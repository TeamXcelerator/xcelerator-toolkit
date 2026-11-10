//! Read-only admission of the producer selected by publication deduplication.
//! An equal numerical payload never substitutes its exact historical ancestry.
use super::*;

pub(super) struct DestinationProducerAdmission<'a> {
    remote: &'a dyn crate::RemoteGitStore,
    destination: PublicationDestination,
    cancellation: &'a xc_core::CancellationToken,
    owner: String,
    topologies: BTreeMap<String, crate::bootstrap_topology::BootstrapFamilyTopology>,
    revisions: BTreeMap<String, String>,
    historical: HistoricalDependencyInventories,
    revocations: BTreeMap<(String, String, String), Option<crate::RevocationIndexPartition>>,
    admitted: BTreeSet<(ContentDigest, String)>,
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
            remote,
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
            admitted: BTreeSet::new(),
        }
    }
    fn topology(
        &mut self,
        family: &str,
    ) -> Result<Option<crate::bootstrap_topology::BootstrapFamilyTopology>, CacheError> {
        if !self.topologies.contains_key(family) {
            match crate::bootstrap_topology::read_bootstrap_family_topology(
                self.remote,
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
            let reader = crate::RemoteShardReader::new(self.remote, 16 * 1024 * 1024)?;
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| CacheError::Io(e.to_string()))?
            .as_secs();
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
            let reader = crate::RemoteShardReader::new(self.remote, 16 * 1024 * 1024)?;
            match reader.read_json::<crate::ShardIndexPartition>(
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
                    if document
                        .value
                        .blocks_exact_manifest(&manifest.semantic_digest, &digest)
                    {
                        return Ok(true);
                    }
                }
                Err(CacheError::NotFound(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(false)
    }
    pub(super) fn admits(
        &mut self,
        root: &crate::CanonicalArtifactManifest,
        repository: &str,
    ) -> Result<bool, CacheError> {
        let root_digest = (root.digest()?, repository.to_owned());
        if self.admitted.contains(&root_digest) {
            return Ok(true);
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
            for dependency in &manifest.canonical_payload.dependencies {
                let Some(topology) = self.topology(&dependency.artifact_family)? else {
                    return Ok(false);
                };
                // The helper validates publication evidence for this exact producer,
                // including superseded immutable parents. No same-content fallback.
                if !exact_destination_dependency_exists(
                    self.remote,
                    self.destination,
                    dependency,
                    &topology,
                    &mut self.revisions,
                    &mut self.historical,
                    &xc_core::ResourcePolicy::default(),
                )? {
                    return Ok(false);
                }
                let reader = crate::RemoteShardReader::new(self.remote, 16 * 1024 * 1024)?;
                let mut found = None;
                for shard in &topology.readable_shards {
                    let revision = self.revision(&shard.repository_url)?;
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
                            found = Some((document.value, shard.repository_url.clone()));
                            break;
                        }
                        Err(CacheError::NotFound(_)) => {}
                        Err(error) => return Err(error),
                    }
                }
                let Some(parent) = found else {
                    return Ok(false);
                };
                pending.push(parent);
            }
        }
        self.admitted.extend(visited);
        Ok(true)
    }
}
