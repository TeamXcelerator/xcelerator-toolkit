use std::collections::BTreeMap;
use xc_cache::*;
#[test]
fn impossible_declared_payload_size_returns_error_instead_of_panicking() {
    let v = ToolkitVersion::parse("0.18.1").unwrap();
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new("test", "test", b"test").unwrap(),
        content_digest: ContentDigest::sha256(b""),
        size_bytes: u64::MAX,
        objects: vec![CacheObjectRef {
            content_digest: ContentDigest::sha256(b""),
            size_bytes: u64::MAX,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: v.clone(),
        minimum_reader_version: v,
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    let scratch = xc_core::test_support::TestDir::new("unused-capacity");
    let store = FilesystemCacheStore::new(
        "capacity",
        scratch.join("unused"),
        false,
        CacheVisibility::Local,
    );
    let outcome = std::panic::catch_unwind(|| store.read_payload(&manifest));
    assert!(outcome.is_ok(), "declared capacity must not panic");
    assert!(outcome.unwrap().is_err());
}
