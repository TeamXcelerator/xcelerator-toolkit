//! Read-only admission of the producer selected by publication deduplication.
//! An equal numerical payload never substitutes its exact historical ancestry.
use super::read_snapshot::ReadSnapshot;
use super::*;
use std::sync::Mutex;

type ProofKey = (PublicationDestination, String, ContentDigest, ContentDigest);

#[derive(Clone)]
struct ProducerProof {
    heads: BTreeMap<(String, String), String>,
    checked_at: u64,
    valid_until: u64,
    bytes: usize,
}

#[derive(Default)]
struct ProofMemo {
    entries: BTreeMap<ProofKey, ProducerProof>,
    bytes: usize,
}

/// Successful exact ancestry checks, scoped to one publication invocation.
/// Every hit rebinds all observed repository heads in the caller's fresh view.
pub(super) struct ProducerProofCache {
    memo: Mutex<ProofMemo>,
    maximum_bytes: usize,
}

impl ProducerProofCache {
    pub(super) fn new(resources: &xc_core::ResourcePolicy) -> Self {
        Self {
            memo: Mutex::new(ProofMemo::default()),
            maximum_bytes: (resources.maximum_memory_bytes.unwrap_or(256 * 1024 * 1024) / 16)
                .min(8 * 1024 * 1024) as usize,
        }
    }
    fn find(
        &self,
        key: &ProofKey,
        remote: &ReadSnapshot<'_>,
        now: u64,
    ) -> Result<Option<u64>, CacheError> {
        let proof = self
            .memo
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .get(key)
            .cloned();
        let Some(proof) = proof else {
            return Ok(None);
        };
        if now < proof.checked_at || now >= proof.valid_until {
            return Ok(None);
        }
        for ((repository, branch), head) in &proof.heads {
            match remote.read_ref(repository, branch) {
                Ok(current) if current == *head => {}
                Ok(_) | Err(CacheError::NotFound(_)) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        if now_seconds()? >= proof.valid_until {
            return Ok(None);
        }
        Ok(Some(proof.valid_until))
    }
    fn insert(
        &self,
        key: ProofKey,
        heads: BTreeMap<(String, String), String>,
        checked_at: u64,
        valid_until: u64,
    ) {
        if checked_at >= valid_until {
            return;
        }
        let bytes = 512
            + key.1.len()
            + heads
                .iter()
                .map(|((r, b), v)| 128 + r.len() + b.len() + v.len())
                .sum::<usize>();
        if bytes > self.maximum_bytes {
            return;
        }
        let mut memo = self.memo.lock().unwrap_or_else(|e| e.into_inner());
        if memo.entries.len() >= 4096 || memo.bytes.saturating_add(bytes) > self.maximum_bytes {
            *memo = ProofMemo::default();
        }
        if let Some(old) = memo.entries.insert(
            key,
            ProducerProof {
                heads,
                checked_at,
                valid_until,
                bytes,
            },
        ) {
            memo.bytes = memo.bytes.saturating_sub(old.bytes);
        }
        memo.bytes += bytes;
    }
}

fn now_seconds() -> Result<u64, CacheError> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|e| CacheError::Io(e.to_string()))
}

pub(super) struct DestinationProducerAdmission<'a> {
    remote: ReadSnapshot<'a>,
    destination: PublicationDestination,
    cancellation: &'a xc_core::CancellationToken,
    owner: String,
    topologies: BTreeMap<String, crate::bootstrap_topology::BootstrapFamilyTopology>,
    revisions: BTreeMap<String, String>,
    historical: HistoricalDependencyInventories,
    revocations: BTreeMap<(String, String, String), Option<crate::RevocationIndexPartition>>,
    // Scoped to this admission pass and immutable repository revisions. Cache
    // validated metadata/proofs, never a substitute producer with equal values.
    indexes: BTreeMap<(String, String, String, String), Option<crate::ShardIndexPartition>>,
    parents: BTreeMap<ExactDependencyIdentity, (crate::CanonicalArtifactManifest, String)>,
    metadata_bytes: u64,
    admitted: BTreeSet<(ContentDigest, String)>,
    proof_cache: Option<&'a ProducerProofCache>,
    root_topology: crate::bootstrap_topology::BootstrapFamilyTopology,
    valid_until: u64,
}
impl<'a> DestinationProducerAdmission<'a> {
    pub(super) fn new(
        remote: &'a dyn crate::RemoteGitStore,
        destination: PublicationDestination,
        topology: &crate::bootstrap_topology::BootstrapFamilyTopology,
        writable_revision: &str,
        cancellation: &'a xc_core::CancellationToken,
    ) -> Self {
        Self {
            remote: ReadSnapshot::with_head(
                remote,
                &topology.current_writable.repository_url,
                writable_revision,
            ),
            destination,
            cancellation,
            owner: topology
                .current_writable
                .authorized_repository
                .split('/')
                .next()
                .unwrap_or("")
                .into(),
            topologies: BTreeMap::from([(topology.family.clone(), topology.clone())]),
            revisions: BTreeMap::from([(
                topology.current_writable.repository_url.clone(),
                writable_revision.into(),
            )]),
            historical: BTreeMap::new(),
            revocations: BTreeMap::new(),
            indexes: BTreeMap::new(),
            parents: BTreeMap::new(),
            metadata_bytes: 0,
            admitted: BTreeSet::new(),
            proof_cache: None,
            root_topology: topology.clone(),
            valid_until: u64::MAX,
        }
    }
    pub(super) fn with_proof_cache(mut self, cache: Option<&'a ProducerProofCache>) -> Self {
        self.proof_cache = cache;
        self
    }
    fn reserve_metadata_cache(&mut self, bytes: u64) {
        // Each authenticated document is already limited to 16 MiB. Bound the
        // combined metadata cache as well; eviction costs reads, never checks.
        if self.metadata_bytes.saturating_add(bytes) > 64 * 1024 * 1024
            || self.parents.len() + self.indexes.len() >= 4096
        {
            self.parents.clear();
            self.indexes.clear();
            self.metadata_bytes = 0;
        }
        self.metadata_bytes += bytes;
    }
    pub(super) fn prefetch_parents(
        &mut self,
        dependencies: &[crate::PayloadDependencyIdentity],
    ) -> Result<(), CacheError> {
        if dependencies.len() < 2 {
            return Ok(());
        }
        let mut paths = BTreeMap::<String, BTreeSet<String>>::new();
        for dependency in dependencies {
            let identity = (
                dependency.artifact_family.clone(),
                dependency.semantic_digest.clone(),
                dependency.manifest_digest.clone(),
                dependency.payload_digest.clone(),
            );
            if self.parents.contains_key(&identity) {
                continue;
            }
            let Some(topology) = self.topology(&dependency.artifact_family)? else {
                continue;
            };
            for shard in &topology.readable_shards {
                let paths = paths.entry(shard.repository_url.clone()).or_default();
                paths.insert(format!(
                    "manifests/{}/{}.json",
                    &dependency.semantic_digest.0[..2],
                    dependency.manifest_digest
                ));
                paths.insert(format!(
                    "indexes/{}/{}.json",
                    dependency.artifact_family,
                    &dependency.semantic_digest.0[..2]
                ));
                for identity in [
                    &dependency.semantic_digest,
                    &dependency.payload_digest,
                    &dependency.manifest_digest,
                ] {
                    paths.insert(format!("revocations/indexes/{}.json", &identity.0[..2]));
                }
            }
        }
        if dependencies.len() >= 64 {
            xc_core::progress_message!(
                "publication ancestry metadata prefetch: destination={} dependencies={} paths={}",
                visibility_name(self.destination),
                dependencies.len(),
                paths.values().map(BTreeSet::len).sum::<usize>()
            );
        }
        for (repository, paths) in paths {
            let revision = self.revision(&repository)?;
            let paths = paths
                .into_iter()
                .map(|repository_path| crate::RemotePathPrefetch {
                    repository_path,
                    maximum_bytes: 16 * 1024 * 1024,
                })
                .collect::<Vec<_>>();
            for batch in paths.chunks(256) {
                self.cancellation
                    .check()
                    .map_err(|e| CacheError::Cancelled(e.to_string()))?;
                // Hydration is only an optimization. Every document below is
                // still read, hashed, version-gated and checked at this revision.
                if let Err(error) = self.remote.prefetch_present_committed_paths(
                    &repository,
                    &revision,
                    batch,
                    self.cancellation,
                ) {
                    self.cancellation
                        .check()
                        .map_err(|e| CacheError::Cancelled(e.to_string()))?;
                    if matches!(error, CacheError::Cancelled(_)) {
                        return Err(error);
                    }
                    xc_core::progress_message!(
                        "publication ancestry metadata prefetch skipped: {error}"
                    );
                    break;
                }
            }
        }
        Ok(())
    }
    fn topology(
        &mut self,
        family: &str,
    ) -> Result<Option<crate::bootstrap_topology::BootstrapFamilyTopology>, CacheError> {
        if !self.topologies.contains_key(family) {
            match crate::bootstrap_topology::read_bootstrap_family_topology(
                &self.remote,
                &self.owner,
                visibility(self.destination),
                family,
                self.cancellation,
            ) {
                Ok(topology) => {
                    self.topologies.insert(family.into(), topology);
                }
                Err(CacheError::NotFound(_)) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        Ok(self.topologies.get(family).cloned())
    }
    fn revision(&mut self, repository: &str) -> Result<String, CacheError> {
        if !self.revisions.contains_key(repository) {
            self.revisions
                .insert(repository.into(), self.remote.read_ref(repository, "main")?);
        }
        Ok(self.revisions[repository].clone())
    }
    fn revoked(
        &mut self,
        repository: &str,
        revision: &str,
        scope: crate::RevocationScope,
        identity: &ContentDigest,
    ) -> Result<bool, CacheError> {
        let prefix = &identity.0[..2];
        let key = (
            repository.to_owned(),
            revision.to_owned(),
            prefix.to_owned(),
        );
        if !self.revocations.contains_key(&key) {
            let reader = crate::RemoteShardReader::new(&self.remote, 16 * 1024 * 1024)?;
            let partition = match reader.read_json::<crate::RevocationIndexPartition>(
                repository,
                revision,
                &format!("revocations/indexes/{prefix}.json"),
                self.cancellation,
            ) {
                Ok(document) => {
                    document.value.validate()?;
                    if document.value.identity_prefix != prefix {
                        return Err(CacheError::InvalidManifest(
                            "revocation prefix mismatch".into(),
                        ));
                    }
                    Some(document.value)
                }
                Err(CacheError::NotFound(_)) => None,
                Err(error) => return Err(error),
            };
            self.revocations.insert(key.clone(), partition);
        }
        let now = now_seconds()?;
        if let Some(partition) = &self.revocations[&key] {
            if let Some(next) = partition
                .records
                .iter()
                .map(|r| r.effective_unix_seconds)
                .filter(|t| *t > now)
                .min()
            {
                self.valid_until = self.valid_until.min(next);
            }
        }
        Ok(self.revocations[&key]
            .as_ref()
            .is_some_and(|p| p.active(scope, identity, now).is_some()))
    }
    fn blocked(
        &mut self,
        manifest: &crate::CanonicalArtifactManifest,
        topology: &crate::bootstrap_topology::BootstrapFamilyTopology,
        repository: &str,
    ) -> Result<bool, CacheError> {
        let digest = manifest.digest()?;
        let source = topology
            .readable_shards
            .iter()
            .find(|shard| shard.repository_url == repository)
            .ok_or_else(|| {
                CacheError::InvalidManifest(
                    "selected producer repository is outside topology".into(),
                )
            })?;
        let repository_digest = crate::repository_identity_digest(&source.authorized_repository);
        for shard in &topology.readable_shards {
            let revision = self.revision(&shard.repository_url)?;
            for (scope, identity) in [
                (crate::RevocationScope::Semantic, &manifest.semantic_digest),
                (crate::RevocationScope::Payload, &manifest.payload_digest),
                (crate::RevocationScope::Manifest, &digest),
                (crate::RevocationScope::Repository, &repository_digest),
            ]
            .into_iter()
            .chain(
                manifest
                    .transport_digests
                    .iter()
                    .map(|d| (crate::RevocationScope::Transport, d)),
            ) {
                if self.revoked(&shard.repository_url, &revision, scope, identity)? {
                    return Ok(true);
                }
            }
            let prefix = &manifest.semantic_digest.0[..2];
            let key = (
                shard.repository_url.clone(),
                revision.clone(),
                manifest.artifact_family.clone(),
                prefix.to_owned(),
            );
            if !self.indexes.contains_key(&key) {
                let reader = crate::RemoteShardReader::new(&self.remote, 16 * 1024 * 1024)?;
                let (partition, bytes) = match reader.read_json::<crate::ShardIndexPartition>(
                    &shard.repository_url,
                    &revision,
                    &format!(
                        "indexes/{}/{}.json",
                        manifest.artifact_family,
                        &manifest.semantic_digest.0[..2]
                    ),
                    self.cancellation,
                ) {
                    Ok(document) => {
                        document.value.validate()?;
                        if document.value.family != manifest.artifact_family
                            || document.value.semantic_prefix != manifest.semantic_digest.0[..2]
                        {
                            return Err(CacheError::InvalidManifest(
                                "producer admission index identity mismatch".into(),
                            ));
                        }
                        (Some(document.value), document.source.size_bytes)
                    }
                    Err(CacheError::NotFound(_)) => (None, 0),
                    Err(error) => return Err(error),
                };
                self.reserve_metadata_cache(bytes);
                self.indexes.insert(key.clone(), partition);
            }
            if self.indexes[&key].as_ref().is_some_and(|partition| {
                partition.blocks_exact_manifest(&manifest.semantic_digest, &digest)
            }) {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(super) fn admits(
        &mut self,
        root: &crate::CanonicalArtifactManifest,
        repository: &str,
    ) -> Result<bool, CacheError> {
        self.cancellation
            .check()
            .map_err(|e| CacheError::Cancelled(e.to_string()))?;
        let root_digest = (root.digest()?, repository.to_owned());
        if !manifest_admissible_for_current_reader(root)
            || !crate::manifest_admitted_to_destination(root, self.destination)
        {
            return Ok(false);
        }
        let checked_at = now_seconds()?;
        if checked_at >= self.valid_until {
            self.admitted.clear();
            self.valid_until = u64::MAX;
        }
        if self.admitted.contains(&root_digest) {
            return Ok(true);
        }
        let proof_key = (
            self.destination,
            repository.to_owned(),
            root_digest.0.clone(),
            canonical_digest(&self.root_topology)?,
        );
        if let Some(cache) = self.proof_cache {
            if let Some(valid_until) = cache.find(&proof_key, &self.remote, checked_at)? {
                self.cancellation
                    .check()
                    .map_err(|e| CacheError::Cancelled(e.to_string()))?;
                self.valid_until = self.valid_until.min(valid_until);
                self.admitted.insert(root_digest);
                return Ok(true);
            }
        }
        let mut pending = vec![(root.clone(), repository.to_owned())];
        let mut visited = BTreeSet::new();
        while let Some((manifest, repository)) = pending.pop() {
            self.cancellation
                .check()
                .map_err(|e| CacheError::Cancelled(e.to_string()))?;
            if !manifest_admissible_for_current_reader(&manifest)
                || !crate::manifest_admitted_to_destination(&manifest, self.destination)
            {
                return Ok(false);
            }
            let digest = manifest.digest()?;
            if !visited.insert((digest, repository.clone())) {
                continue;
            }
            if visited.len() > 100_000 {
                return Err(CacheError::ResourceLimit(
                    "publication ancestry exceeds 100000 manifests".into(),
                ));
            }
            let Some(topology) = self.topology(&manifest.artifact_family)? else {
                return Ok(false);
            };
            if self.blocked(&manifest, &topology, &repository)? {
                return Ok(false);
            }
            self.prefetch_parents(&manifest.canonical_payload.dependencies)?;
            for dependency in &manifest.canonical_payload.dependencies {
                let identity = (
                    dependency.artifact_family.clone(),
                    dependency.semantic_digest.clone(),
                    dependency.manifest_digest.clone(),
                    dependency.payload_digest.clone(),
                );
                if let Some(parent) = self.parents.get(&identity) {
                    // Walk the exact parent again using validated metadata:
                    // time-dependent revocations and all ancestry checks still
                    // run, without repeating Git reads/publication proofs.
                    pending.push(parent.clone());
                    continue;
                }
                let Some(topology) = self.topology(&dependency.artifact_family)? else {
                    return Ok(false);
                };
                // The helper validates publication evidence for this exact producer,
                // including superseded immutable parents. No same-content fallback.
                if !exact_destination_dependency_exists(
                    &self.remote,
                    self.destination,
                    dependency,
                    &topology,
                    &mut self.revisions,
                    &mut self.historical,
                    &xc_core::ResourcePolicy::default(),
                )? {
                    return Ok(false);
                }
                let mut found = None;
                for shard in &topology.readable_shards {
                    let revision = self.revision(&shard.repository_url)?;
                    let reader = crate::RemoteShardReader::new(&self.remote, 16 * 1024 * 1024)?;
                    match reader.read_json::<crate::CanonicalArtifactManifest>(
                        &shard.repository_url,
                        &revision,
                        &format!(
                            "manifests/{}/{}.json",
                            &dependency.semantic_digest.0[..2],
                            dependency.manifest_digest
                        ),
                        self.cancellation,
                    ) {
                        Ok(document) => {
                            if !manifest_admissible_for_current_reader(&document.value) {
                                return Ok(false);
                            }
                            if document.source.content_digest != dependency.manifest_digest
                                || document.value.digest()? != dependency.manifest_digest
                                || document.value.semantic_digest != dependency.semantic_digest
                                || document.value.payload_digest != dependency.payload_digest
                                || document.value.artifact_family != dependency.artifact_family
                            {
                                return Err(CacheError::InvalidManifest(
                                    "selected producer ancestry binding mismatch".into(),
                                ));
                            }
                            if self.blocked(&document.value, &topology, &shard.repository_url)? {
                                continue;
                            }
                            found = Some((
                                document.value,
                                shard.repository_url.clone(),
                                document.source.size_bytes,
                            ));
                            break;
                        }
                        Err(CacheError::NotFound(_)) => {}
                        Err(error) => return Err(error),
                    }
                }
                let Some((manifest, repository, bytes)) = found else {
                    return Ok(false);
                };
                let parent = (manifest, repository);
                self.reserve_metadata_cache(bytes);
                self.parents.insert(identity, parent.clone());
                pending.push(parent);
            }
        }
        // A scheduled revocation may activate during a long cold traversal.
        // Recheck with the cached immutable documents rather than storing a
        // proof whose validity interval has already ended.
        if now_seconds()? >= self.valid_until {
            return self.admits(root, repository);
        }
        if let Some(cache) = self.proof_cache {
            cache.insert(
                proof_key,
                self.remote.observed_heads(),
                checked_at,
                self.valid_until,
            );
        }
        self.admitted.extend(visited);
        Ok(true)
    }
}

#[cfg(test)]
mod proof_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Heads {
        head: Mutex<String>,
        reads: AtomicUsize,
    }
    impl RemoteGitStore for Heads {
        fn read_committed_path(
            &self,
            _: &str,
            _: &str,
            _: &str,
            _: u64,
            _: &xc_core::CancellationToken,
            _: &mut dyn std::io::Write,
        ) -> Result<crate::RemoteReadReport, CacheError> {
            panic!("proof reuse must not read metadata")
        }
        fn read_ref(&self, _: &str, _: &str) -> Result<String, CacheError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(self.head.lock().unwrap().clone())
        }
        fn immutable_path_digest(
            &self,
            _: &str,
            _: &str,
            _: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            panic!("proof reuse must not read paths")
        }
        fn compare_and_swap_commit(
            &self,
            _: &crate::RemoteCommitRequest,
        ) -> Result<crate::CompareAndSwapResult, CacheError> {
            panic!("proof reuse must not mutate")
        }
        fn verify_committed_part(
            &self,
            _: &str,
            _: &str,
            _: &TransportPart,
        ) -> Result<(), CacheError> {
            panic!("proof reuse must not read payloads")
        }
    }
    #[test]
    fn proofs_bind_all_heads_destination_topology_time_and_memory_budget() {
        let cache = ProducerProofCache::new(&xc_core::ResourcePolicy::default());
        let remote = Heads {
            head: Mutex::new("old".into()),
            reads: AtomicUsize::new(0),
        };
        let key = (
            PublicationDestination::Private,
            "root".into(),
            ContentDigest::sha256(b"manifest"),
            ContentDigest::sha256(b"topology"),
        );
        let now = now_seconds().unwrap();
        cache.insert(
            key.clone(),
            BTreeMap::from([
                (("ancestor".into(), "main".into()), "old".into()),
                (("registry".into(), "main".into()), "old".into()),
            ]),
            now,
            now + 3600,
        );
        let snapshot = ReadSnapshot::new(&remote);
        assert_eq!(cache.find(&key, &snapshot, now).unwrap(), Some(now + 3600));
        assert_eq!(cache.find(&key, &snapshot, now).unwrap(), Some(now + 3600));
        assert_eq!(
            remote.reads.load(Ordering::Relaxed),
            2,
            "heads are pinned within one read-only phase"
        );
        *remote.head.lock().unwrap() = "new".into();
        assert!(cache
            .find(&key, &ReadSnapshot::new(&remote), now)
            .unwrap()
            .is_none());
        let mut other = key.clone();
        other.0 = PublicationDestination::Public;
        assert!(cache.find(&other, &snapshot, now).unwrap().is_none());
        other = key.clone();
        other.3 = ContentDigest::sha256(b"new topology");
        assert!(cache.find(&other, &snapshot, now).unwrap().is_none());
        assert!(cache.find(&key, &snapshot, now - 1).unwrap().is_none());
        assert!(
            cache.find(&key, &snapshot, now + 3600).unwrap().is_none(),
            "scheduled revocations expire proofs even without head changes"
        );
        cache.insert(key.clone(), BTreeMap::new(), now - 2, now - 1);
        assert!(
            cache.find(&key, &snapshot, now - 2).unwrap().is_none(),
            "check the clock again after head validation"
        );
        let tiny = ProducerProofCache::new(&xc_core::ResourcePolicy {
            maximum_memory_bytes: Some(0),
            ..Default::default()
        });
        tiny.insert(key, BTreeMap::new(), now, u64::MAX);
        assert!(tiny.memo.lock().unwrap().entries.is_empty());
    }
}
