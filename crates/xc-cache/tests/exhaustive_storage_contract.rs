use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    sync::{Arc, Barrier},
};
use xc_cache::*;
fn draft(kind: &str, name: &str) -> ArtifactDraft {
    let v = ToolkitVersion::parse("0.15.1").unwrap();
    ArtifactDraft {
        schema_version: 1,
        key: ArtifactKey::new(kind, name, b"same").unwrap(),
        producer_toolkit_version: v.clone(),
        minimum_reader_version: v,
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    }
}
fn manifest(kind: &str, name: &str) -> ArtifactManifest {
    let d = draft(kind, name);
    ArtifactManifest {
        schema_version: d.schema_version,
        key: d.key,
        content_digest: ContentDigest::sha256(b"x"),
        size_bytes: 1,
        objects: vec![CacheObjectRef {
            content_digest: ContentDigest::sha256(b"x"),
            size_bytes: 1,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: d.producer_toolkit_version,
        minimum_reader_version: d.minimum_reader_version,
        maximum_reader_version: None,
        quality: d.quality,
        visibility: d.visibility,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    }
}
fn dependency(m: &ArtifactManifest, quality: CacheQuality) -> DependencyRef {
    DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: quality,
    }
}
struct MemoryStore(Vec<ArtifactManifest>);
impl CacheStore for MemoryStore {
    fn name(&self) -> &str {
        "memory"
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        CacheVisibility::Local
    }
    fn put(&self, _: &ArtifactDraft, _: &[u8]) -> Result<ArtifactManifest, CacheError> {
        unreachable!()
    }
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        Ok(self.0.iter().filter(|m| m.key == *key).cloned().collect())
    }
    fn read_payload_to(&self, _: &ArtifactManifest, _: &mut dyn Write) -> Result<(), CacheError> {
        unreachable!()
    }
}
fn policy() -> CachePolicy {
    CachePolicy {
        current_toolkit_version: ToolkitVersion::parse("0.15.1").unwrap(),
        minimum_quality: CacheQuality::Validated,
        allowed_visibilities: [CacheVisibility::Local].into_iter().collect(),
        accepted_schema_versions: [1].into_iter().collect(),
        allow_deprecated: false,
        allow_quarantined: false,
    }
}
#[test]
fn every_dependency_edge_must_satisfy_its_own_quality_floor() {
    let leaf = manifest("test", "leaf");
    let mut a = manifest("test", "a");
    a.dependencies
        .push(dependency(&leaf, CacheQuality::Validated));
    let mut b = manifest("test", "b");
    b.dependencies
        .push(dependency(&leaf, CacheQuality::Certified));
    let mut root = manifest("test", "root");
    root.dependencies = vec![
        dependency(&a, CacheQuality::Validated),
        dependency(&b, CacheQuality::Validated),
    ];
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(MemoryStore(vec![a, b, leaf])),
    }]);
    assert!(resolver
        .validate_dependency_closure(&root, &policy())
        .is_err());
}
#[test]
fn dependency_identity_is_a_tuple_not_colon_concatenation() {
    let a = manifest("a", "b:c");
    let b = manifest("a:b", "c");
    let mut root = manifest("test", "root");
    root.dependencies = vec![
        dependency(&a, CacheQuality::Validated),
        dependency(&b, CacheQuality::Validated),
    ];
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(MemoryStore(vec![a])),
    }]);
    assert!(resolver
        .validate_dependency_closure(&root, &policy())
        .is_err());
}
#[test]
fn closure_uses_a_lower_layer_when_it_satisfies_the_quality_floor() {
    let low = manifest("test", "leaf");
    let mut high = low.clone();
    high.quality = CacheQuality::Certified;
    let mut root = manifest("test", "root");
    root.dependencies = vec![dependency(&low, CacheQuality::Certified)];
    let resolver = CacheResolver::new(vec![
        CacheLayer {
            precedence: 0,
            store: Box::new(MemoryStore(vec![low])),
        },
        CacheLayer {
            precedence: 1,
            store: Box::new(MemoryStore(vec![high])),
        },
    ]);
    assert!(resolver
        .validate_dependency_closure(&root, &policy())
        .is_ok());
}
#[test]
fn saturated_repository_size_does_not_certify_capacity() {
    let shard = RepositoryShard {
        id: "test".into(),
        repository: "test".into(),
        visibility: CacheVisibility::Local,
        artifact_kinds: vec![],
        reachable_payload_bytes: u64::MAX,
        estimated_history_bytes: 0,
        safe_payload_limit_bytes: u64::MAX,
        writable: true,
    };
    assert!(!shard.can_accept("test", CacheVisibility::Local, 1, 0));
}
#[test]
fn impossible_stream_chunk_size_returns_error_without_panicking() {
    let temp = Temp::new();
    let store = FilesystemCacheStore::new("test", temp.path(), true, CacheVisibility::Local);
    let result = std::panic::catch_unwind(|| {
        store.put_reader(
            &draft("test", "capacity"),
            &mut std::io::empty(),
            usize::MAX,
        )
    });
    assert!(result.is_ok());
    assert!(result.unwrap().is_err());
}
#[test]
fn concurrent_writes_retain_every_manifest_in_the_index() {
    let temp = Temp::new();
    let store = Arc::new(FilesystemCacheStore::new(
        "test",
        temp.path(),
        true,
        CacheVisibility::Local,
    ));
    let barrier = Arc::new(Barrier::new(32));
    let seed = store.put(&draft("test", "concurrent"), &[255]).unwrap();
    let workers = (0u8..32)
        .map(|i| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                store.put(&draft("test", "concurrent"), &[i]).unwrap()
            })
        })
        .collect::<Vec<_>>();
    let mut expected = workers
        .into_iter()
        .map(|w| w.join().unwrap().content_digest)
        .collect::<BTreeSet<_>>();
    expected.insert(seed.content_digest);
    let actual = store
        .candidates(&draft("test", "concurrent").key)
        .unwrap()
        .into_iter()
        .map(|m| m.content_digest)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
}
#[test]
fn corrupted_payload_cannot_write_beyond_its_declared_size() {
    let temp = Temp::new();
    let store = FilesystemCacheStore::new("test", temp.path(), true, CacheVisibility::Local);
    let manifest = store.put(&draft("test", "size"), b"x").unwrap();
    let hash = &manifest.objects[0].content_digest.0;
    let object = temp
        .path()
        .join("objects/sha256")
        .join(&hash[..2])
        .join(hash);
    std::fs::write(object, b"corrupt oversized payload").unwrap();
    let mut output = Vec::new();
    assert!(store.read_payload_to(&manifest, &mut output).is_err());
    assert!(output.len() <= 1);
}

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "xc-storage-audit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        }
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn key_paths_do_not_alias_escaped_and_literal_names() {
    let temp = Temp::new();
    let store = FilesystemCacheStore::new("test", temp.path(), true, CacheVisibility::Local);
    let first = store.put(&draft("test", "foo/bar"), b"first").unwrap();
    let second = store.put(&draft("test", "foo_2fbar"), b"second").unwrap();
    assert_eq!(store.candidates(&first.key).unwrap(), vec![first]);
    assert_eq!(store.candidates(&second.key).unwrap(), vec![second]);
}
#[test]
fn dot_keys_remain_inside_the_cache_root() {
    let temp = Temp::new();
    let root = temp.path().join("base/cache");
    std::fs::create_dir_all(&root).unwrap();
    let store = FilesystemCacheStore::new("test", &root, true, CacheVisibility::Local);
    let m = store.put(&draft("..", ".."), b"x").unwrap();
    assert!(!temp
        .path()
        .join("base")
        .join(&m.key.parameters_digest.0)
        .exists());
    assert_eq!(store.candidates(&m.key).unwrap(), vec![m]);
}

#[test]
fn legacy_colliding_keys_and_new_records_remain_discoverable() {
    let temp = Temp::new();
    let a = manifest("test", "foo/bar");
    let b = manifest("test", "foo_2fbar");
    let directory = temp
        .path()
        .join("artifacts/test/foo_2fbar")
        .join(&a.key.parameters_digest.0);
    std::fs::create_dir_all(directory.join("manifests")).unwrap();
    for (name, m) in [("a", &a), ("b", &b)] {
        std::fs::write(
            directory.join(format!("manifests/{name}.json")),
            serde_json::to_vec(m).unwrap(),
        )
        .unwrap();
    }
    let store = FilesystemCacheStore::new("test", temp.path(), true, CacheVisibility::Local);
    assert_eq!(store.candidates(&a.key).unwrap(), vec![a.clone()]);
    assert_eq!(store.candidates(&b.key).unwrap(), vec![b.clone()]);
    let fresh = store.put(&draft("test", "foo/bar"), b"fresh").unwrap();
    let candidates = store.candidates(&a.key).unwrap();
    assert_eq!(candidates.len(), 2);
    assert!(candidates.contains(&a));
    assert!(candidates.contains(&fresh));
    assert_eq!(store.matching_keys("test", "foo", 10).unwrap().len(), 2);
}
#[test]
fn keys_validate_before_their_digest_is_used_as_a_path() {
    let temp = Temp::new();
    let store = FilesystemCacheStore::new("test", temp.path(), false, CacheVisibility::Local);
    let mut key = draft("test", "test").key;
    key.parameters_digest = ContentDigest("../../invalid".into());
    assert!(store.candidates(&key).is_err());
}
#[test]
fn invalid_dependency_key_is_rejected_by_the_manifest() {
    let mut m = manifest("test", "parent");
    let mut child = manifest("test", "child");
    child.key.kind.clear();
    m.dependencies
        .push(dependency(&child, CacheQuality::Validated));
    assert!(m.validate().is_err());
}

fn canonical_adapter(name: &str, parents: &[ArtifactManifest]) -> ArtifactManifest {
    let mut m = manifest("quadrature_rule", name);
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: m.key.kind.clone(),
        mathematical_semantics_version: "audit-v1".into(),
        resolved_mathematical_parameters: serde_json::json!({"name":name}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let mut dependencies = parents
        .iter()
        .map(|parent| {
            let c = retained_canonical_manifest(parent).unwrap().unwrap();
            PayloadDependencyIdentity {
                artifact_family: c.artifact_family.clone(),
                semantic_digest: c.semantic_digest.clone(),
                manifest_digest: c.digest().unwrap(),
                payload_digest: c.payload_digest,
            }
        })
        .collect::<Vec<_>>();
    dependencies.sort_by(|a, b| {
        (
            &a.artifact_family,
            &a.semantic_digest,
            &a.manifest_digest,
            &a.payload_digest,
        )
            .cmp(&(
                &b.artifact_family,
                &b.semantic_digest,
                &b.manifest_digest,
                &b.payload_digest,
            ))
    });
    let payload = CanonicalPayloadEnvelope {
        schema_version: 1,
        scalar_backend: "opaque".into(),
        precision_bits: None,
        scalar_representation: "bytes".into(),
        dimensions: vec![1],
        endianness: "none".into(),
        special_value_encoding: "none".into(),
        ordered_items: vec![LogicalPayloadItem {
            normalized_path: "payload.json".into(),
            content_digest: m.content_digest.clone(),
            size_bytes: 1,
        }],
        dependencies,
    };
    let canonical = CanonicalArtifactManifest {
        schema_version: 1,
        artifact_family: "quadrature".into(),
        semantic_digest: semantic.digest().unwrap(),
        semantic_key: semantic.clone(),
        payload_digest: payload.digest().unwrap(),
        canonical_payload: payload,
        transport_digests: vec![ContentDigest::sha256(b"transport")],
        resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
        producer_toolkit_version: m.producer_toolkit_version.clone(),
        minimum_reader_version: m.minimum_reader_version.clone(),
        maximum_reader_version: None,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        claim_scope: "fixture".into(),
        assumptions: vec![],
    };
    m.key.parameters_digest = semantic.digest().unwrap();
    m.provenance_digest = Some(canonical.digest().unwrap());
    m.tags.insert(
        SEMANTIC_KEY_MANIFEST_TAG.into(),
        serde_json::to_string(&semantic).unwrap(),
    );
    m.tags.insert(
        REMOTE_CANONICAL_MANIFEST_TAG.into(),
        serde_json::to_string(&canonical).unwrap(),
    );
    m
}
#[test]
fn aliases_of_one_published_parent_cannot_cover_a_missing_parent() {
    let a = canonical_adapter("a", &[]);
    let b = canonical_adapter("b", &[]);
    let root = canonical_adapter("root", &[a.clone(), b]);
    let mut alias = a.clone();
    alias.key.logical_key = "alias-of-a".into();
    assert!(!manifest_sources_match(&root, &[&a, &alias]).unwrap());
}
#[test]
fn published_parent_aliases_are_deduplicated_by_canonical_identity() {
    let a = canonical_adapter("a", &[]);
    let root = canonical_adapter("root", std::slice::from_ref(&a));
    let mut alias = a.clone();
    alias.key.logical_key = "alias-of-a".into();
    assert!(manifest_sources_match(&root, &[&a, &alias]).unwrap());
}
