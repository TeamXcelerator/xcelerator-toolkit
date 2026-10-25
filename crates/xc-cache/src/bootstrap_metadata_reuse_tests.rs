use super::*;
use crate::bootstrap_topology::{self, BootstrapDocumentReader};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct MemoryRemote {
    heads: Mutex<HashMap<String, String>>,
    documents: Mutex<HashMap<(String, String, String), Vec<u8>>>,
    head_reads: AtomicUsize,
    document_reads: AtomicUsize,
}

impl RemoteGitStore for MemoryRemote {
    fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError> {
        assert_eq!(branch, "main");
        self.head_reads.fetch_add(1, Ordering::Relaxed);
        self.heads
            .lock()
            .unwrap()
            .get(repository)
            .cloned()
            .ok_or_else(|| CacheError::NotFound(repository.into()))
    }
    fn immutable_path_digest(
        &self,
        _: &str,
        _: &str,
        _: &str,
    ) -> Result<Option<ContentDigest>, CacheError> {
        panic!("topology metadata test must not request payload digests")
    }
    fn read_committed_path(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        maximum_bytes: u64,
        cancellation: &CancellationToken,
        writer: &mut dyn Write,
    ) -> Result<RemoteReadReport, CacheError> {
        cancellation
            .check()
            .map_err(|e| CacheError::Cancelled(e.to_string()))?;
        self.document_reads.fetch_add(1, Ordering::Relaxed);
        let documents = self.documents.lock().unwrap();
        let bytes = documents
            .get(&(repository.into(), revision.into(), path.into()))
            .ok_or_else(|| CacheError::NotFound(path.into()))?;
        if bytes.len() as u64 > maximum_bytes {
            return Err(CacheError::ResourceLimit("mock bounded document".into()));
        }
        writer.write_all(bytes)?;
        Ok(RemoteReadReport {
            repository_path: path.into(),
            revision: revision.into(),
            size_bytes: bytes.len() as u64,
            content_digest: ContentDigest::sha256(bytes),
        })
    }
    fn compare_and_swap_commit(
        &self,
        _: &RemoteCommitRequest,
    ) -> Result<CompareAndSwapResult, CacheError> {
        panic!("metadata qualification must never publish")
    }
    fn verify_committed_part(&self, _: &str, _: &str, _: &TransportPart) -> Result<(), CacheError> {
        panic!("topology metadata test must not request payload parts")
    }
}

struct CachedReader<'a> {
    remote: &'a MemoryRemote,
    cache: Mutex<MetadataDocumentCache>,
    resources: ResourcePolicy,
}
impl<'a> CachedReader<'a> {
    fn new(remote: &'a MemoryRemote) -> Self {
        Self {
            remote,
            cache: Mutex::new(MetadataDocumentCache::default()),
            resources: ResourcePolicy::default(),
        }
    }
    fn read<T: serde::de::DeserializeOwned>(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        maximum_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(T, RemoteReadReport), CacheError> {
        read_cached_metadata_json(
            self.remote,
            &self.cache,
            &self.resources,
            repository,
            revision,
            path,
            maximum_bytes,
            cancellation,
        )
    }
}
impl BootstrapDocumentReader for CachedReader<'_> {
    fn read_topology_json<T: serde::de::DeserializeOwned>(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        cancellation: &CancellationToken,
    ) -> Result<T, CacheError> {
        self.read(repository, revision, path, 4 * 1024 * 1024, cancellation)
            .map(|(v, _)| v)
    }
}

fn add_snapshot(remote: &MemoryRemote, visibility: &str, revision: &str) -> String {
    let registry_id = format!("fixture/xcelerator-cache-{visibility}-registry");
    let repository = format!("https://github.com/{registry_id}.git");
    let mut docs = remote.documents.lock().unwrap();
    let mut put = |path: String, value: serde_json::Value| {
        docs.insert(
            (repository.clone(), revision.into(), path),
            serde_json::to_vec(&value).unwrap(),
        );
    };
    let families = ["ccm-matrices", "ccm-evidence"];
    put(
        "registry.json".into(),
        serde_json::json!({"schema_version":1,"repository":registry_id,
        "default_branch":"main","visibility":visibility,"separate_visibility_inventory":true,
        "families":families.map(|family| serde_json::json!({"family":family,
            "current_writable_shard":format!("fixture/xcelerator-cache-{visibility}-{family}-0001"),
            "metadata":format!("families/{family}.json")}))}),
    );
    for family in families {
        let shard = format!("xcelerator-cache-{visibility}-{family}-0001");
        let authorized = format!("fixture/{shard}");
        put(
            format!("families/{family}.json"),
            serde_json::json!({"schema_version":1,"family":family,
            "display_name":family,"description":"fixture","visibility":visibility,
            "artifact_kinds":["fixture"],"current_writable_shard":authorized,
            "shards":[format!("shards/{shard}.json")]}),
        );
        put(
            format!("shards/{shard}.json"),
            serde_json::json!({"schema_version":1,"repository":authorized,
            "family":family,"visibility":visibility,"shard":1,"default_branch":"main",
            "immutable_objects":true,"writable":true,"capacity_policy":{
                "maximum_reachable_payload_bytes":GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
                "rollover_repository_pattern":format!("xcelerator-cache-{visibility}-{family}-{{shard:04d}}")}}),
        );
    }
    remote
        .heads
        .lock()
        .unwrap()
        .insert(repository.clone(), revision.into());
    repository
}

#[test]
fn topology_reuse_preserves_head_observations_and_exact_topology() {
    let remote = MemoryRemote::default();
    let revision = "a".repeat(40);
    let repository = add_snapshot(&remote, "private", &revision);
    let cached = CachedReader::new(&remote);
    let cancel = CancellationToken::new();
    let preflight_revision = remote.read_ref(&repository, "main").unwrap();
    cached
        .read::<BootstrapRegistry>(
            &repository,
            &preflight_revision,
            "registry.json",
            16 * 1024 * 1024,
            &cancel,
        )
        .unwrap();
    let mut observed = Vec::new();
    for family in ["ccm-matrices", "ccm-evidence"] {
        observed.push(
            bootstrap_topology::read_bootstrap_family_topology_with_reader(
                &remote,
                "fixture",
                CacheVisibility::Private,
                family,
                &cancel,
                &cached,
            )
            .unwrap(),
        );
    }
    assert_eq!(remote.head_reads.load(Ordering::Relaxed), 3);
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 5);
    let baseline = MemoryRemote::default();
    add_snapshot(&baseline, "private", &revision);
    baseline.read_ref(&repository, "main").unwrap();
    let mut ignored = Vec::new();
    baseline
        .read_committed_path(
            &repository,
            &revision,
            "registry.json",
            16 * 1024 * 1024,
            &cancel,
            &mut ignored,
        )
        .unwrap();
    for (family, expected) in ["ccm-matrices", "ccm-evidence"].into_iter().zip(observed) {
        assert_eq!(
            bootstrap_topology::read_bootstrap_family_topology(
                &baseline,
                "fixture",
                CacheVisibility::Private,
                family,
                &cancel
            )
            .unwrap(),
            expected
        );
    }
    assert_eq!(baseline.head_reads.load(Ordering::Relaxed), 3);
    assert_eq!(baseline.document_reads.load(Ordering::Relaxed), 7);
}

#[test]
fn topology_moved_head_reads_new_revision_and_revalidates() {
    let remote = MemoryRemote::default();
    let a = "a".repeat(40);
    let b = "b".repeat(40);
    let repository = add_snapshot(&remote, "private", &a);
    let cached = CachedReader::new(&remote);
    let cancel = CancellationToken::new();
    bootstrap_topology::read_bootstrap_family_topology_with_reader(
        &remote,
        "fixture",
        CacheVisibility::Private,
        "ccm-matrices",
        &cancel,
        &cached,
    )
    .unwrap();
    add_snapshot(&remote, "private", &b);
    let path = "shards/xcelerator-cache-private-ccm-matrices-0001.json";
    let key = (repository, b, path.into());
    let mut docs = remote.documents.lock().unwrap();
    let mut changed: serde_json::Value = serde_json::from_slice(&docs[&key]).unwrap();
    changed["immutable_objects"] = serde_json::json!(false);
    docs.insert(key, serde_json::to_vec(&changed).unwrap());
    drop(docs);
    assert!(matches!(
        bootstrap_topology::read_bootstrap_family_topology_with_reader(
            &remote,
            "fixture",
            CacheVisibility::Private,
            "ccm-matrices",
            &cancel,
            &cached
        ),
        Err(CacheError::InvalidManifest(_))
    ));
    assert_eq!(remote.head_reads.load(Ordering::Relaxed), 2);
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 6);
    // The invalid descriptor is decoded again and still rejected on a hit.
    assert!(
        bootstrap_topology::read_bootstrap_family_topology_with_reader(
            &remote,
            "fixture",
            CacheVisibility::Private,
            "ccm-matrices",
            &cancel,
            &cached
        )
        .is_err()
    );
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 6);
    assert_eq!(remote.head_reads.load(Ordering::Relaxed), 3);
}

#[test]
fn topology_cache_keeps_visibility_repositories_separate() {
    let remote = MemoryRemote::default();
    let revision = "a".repeat(40);
    add_snapshot(&remote, "private", &revision);
    add_snapshot(&remote, "public", &revision);
    let cached = CachedReader::new(&remote);
    let cancel = CancellationToken::new();
    for visibility in [CacheVisibility::Private, CacheVisibility::Public] {
        let result = bootstrap_topology::read_bootstrap_family_topology_with_reader(
            &remote,
            "fixture",
            visibility,
            "ccm-matrices",
            &cancel,
            &cached,
        )
        .unwrap();
        assert_eq!(result.visibility, visibility);
    }
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 6);
}

#[test]
fn cached_metadata_still_enforces_caller_transfer_and_cancellation() {
    let remote = MemoryRemote::default();
    let revision = "a".repeat(40);
    let repository = add_snapshot(&remote, "private", &revision);
    let mut cached = CachedReader::new(&remote);
    let cancel = CancellationToken::new();
    let (_, report) = cached
        .read::<BootstrapRegistry>(
            &repository,
            &revision,
            "registry.json",
            16 * 1024 * 1024,
            &cancel,
        )
        .unwrap();
    assert!(matches!(
        cached.read::<BootstrapRegistry>(
            &repository,
            &revision,
            "registry.json",
            report.size_bytes - 1,
            &cancel
        ),
        Err(CacheError::ResourceLimit(_))
    ));
    cached.resources.maximum_transfer_bytes = Some(report.size_bytes - 1);
    assert!(matches!(
        cached.read::<BootstrapRegistry>(
            &repository,
            &revision,
            "registry.json",
            16 * 1024 * 1024,
            &cancel
        ),
        Err(CacheError::ResourceLimit(_))
    ));
    cached.resources.maximum_transfer_bytes = None;
    cached.resources.maximum_memory_bytes = Some(0);
    assert!(matches!(
        cached.read::<BootstrapRegistry>(
            &repository,
            &revision,
            "registry.json",
            16 * 1024 * 1024,
            &cancel
        ),
        Err(CacheError::ResourceLimit(_))
    ));
    cached.resources.maximum_memory_bytes = None;
    cancel.cancel(xc_core::CancellationReason::UserRequested);
    assert!(matches!(
        cached.read::<BootstrapRegistry>(
            &repository,
            &revision,
            "registry.json",
            16 * 1024 * 1024,
            &cancel
        ),
        Err(CacheError::Cancelled(_))
    ));
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 1);
}

#[test]
fn corrupt_duplicate_and_missing_metadata_are_not_successfully_cached() {
    for bytes in [
        b"{broken".as_slice(),
        b"{\"schema_version\":1,\"schema_version\":1}".as_slice(),
    ] {
        let remote = MemoryRemote::default();
        let revision = "a".repeat(40);
        let repository = add_snapshot(&remote, "private", &revision);
        remote.documents.lock().unwrap().insert(
            (repository.clone(), revision.clone(), "registry.json".into()),
            bytes.to_vec(),
        );
        let cached = CachedReader::new(&remote);
        let cancel = CancellationToken::new();
        for _ in 0..2 {
            assert!(cached
                .read::<BootstrapRegistry>(
                    &repository,
                    &revision,
                    "registry.json",
                    4 * 1024 * 1024,
                    &cancel
                )
                .is_err());
        }
        assert!(cached.cache.lock().unwrap().documents.is_empty());
        assert_eq!(remote.document_reads.load(Ordering::Relaxed), 2);
    }
    let remote = MemoryRemote::default();
    let cached = CachedReader::new(&remote);
    let cancel = CancellationToken::new();
    for _ in 0..2 {
        assert!(matches!(
            cached.read::<BootstrapRegistry>(
                "missing",
                &"a".repeat(40),
                "registry.json",
                4 * 1024 * 1024,
                &cancel
            ),
            Err(CacheError::NotFound(_))
        ));
    }
    assert!(cached.cache.lock().unwrap().documents.is_empty());
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 2);
}

#[test]
fn metadata_retention_respects_available_memory_limit() {
    let remote = MemoryRemote::default();
    let revision = "a".repeat(40);
    let repository = add_snapshot(&remote, "private", &revision);
    let mut cached = CachedReader::new(&remote);
    cached.resources.maximum_memory_bytes = Some(1);
    let cancel = CancellationToken::new();
    for _ in 0..2 {
        cached
            .read::<BootstrapRegistry>(
                &repository,
                &revision,
                "registry.json",
                4 * 1024 * 1024,
                &cancel,
            )
            .unwrap();
    }
    assert!(cached.cache.lock().unwrap().documents.is_empty());
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 2);
}

#[test]
fn lowering_owned_reader_policy_releases_excess_retained_metadata() {
    let root = crate::test_support::temporary_root("metadata-policy-lowering");
    let remote = MemoryRemote::default();
    let revision = "a".repeat(40);
    let repository = add_snapshot(&remote, "public", &revision);
    let mut store = GitHubBootstrapCacheStore::public("fixture", root.join("store")).unwrap();
    let read = |store: &GitHubBootstrapCacheStore| {
        read_cached_metadata_json::<BootstrapRegistry>(
            &remote,
            &store.metadata_documents,
            &store.resources,
            &repository,
            &revision,
            "registry.json",
            4 * 1024 * 1024,
            &store.cancellation,
        )
    };
    let (_, report) = read(&store).unwrap();
    assert!(report.size_bytes > 1);
    assert_eq!(
        store.metadata_documents.lock().unwrap().retained_bytes,
        report.size_bytes
    );

    // An equal allowance retains the authenticated bytes and still hits.
    store = store.with_resource_policy(ResourcePolicy {
        maximum_memory_bytes: Some(report.size_bytes),
        ..ResourcePolicy::default()
    });
    read(&store).unwrap();
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 1);

    // The public owned builder must enforce a smaller allowance immediately,
    // not wait for the next cache miss. A subsequent read is not retained.
    store = store.with_resource_policy(ResourcePolicy {
        maximum_memory_bytes: Some(report.size_bytes - 1),
        ..ResourcePolicy::default()
    });
    assert_eq!(store.metadata_documents.lock().unwrap().retained_bytes, 0);
    assert!(store
        .metadata_documents
        .lock()
        .unwrap()
        .documents
        .is_empty());
    read(&store).unwrap();
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 2);
    assert_eq!(store.metadata_documents.lock().unwrap().retained_bytes, 0);

    // Repopulation after raising the allowance is safe; a zero allowance also
    // releases those bytes and refuses even a previously authenticated path.
    store = store.with_resource_policy(ResourcePolicy::default());
    read(&store).unwrap();
    assert_eq!(
        store.metadata_documents.lock().unwrap().retained_bytes,
        report.size_bytes
    );
    store = store.with_resource_policy(ResourcePolicy {
        maximum_memory_bytes: Some(0),
        ..ResourcePolicy::default()
    });
    assert_eq!(store.metadata_documents.lock().unwrap().retained_bytes, 0);
    assert!(matches!(read(&store), Err(CacheError::ResourceLimit(_))));
    assert_eq!(remote.document_reads.load(Ordering::Relaxed), 3);
    store.finish().unwrap();
    fs::remove_dir_all(root).unwrap();
}
