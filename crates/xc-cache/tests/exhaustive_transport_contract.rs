use std::collections::BTreeMap;
use xc_cache::*;
use xc_core::{CancellationToken, ResourcePolicy};
fn policy() -> TransportPolicy {
    TransportPolicy {
        maximum_file_bytes_exclusive: 16,
        split_part_bytes: 4,
        maximum_batch_payload_bytes: 16,
        maximum_pending_batches: 1,
    }
}
#[test]
fn repeated_content_chunks_preserve_the_encoded_stream() {
    let bytes = b"ABCDABCDABCD";
    let mut stored = BTreeMap::new();
    let result = stream_split_encoded(
        &mut bytes.as_slice(),
        ContentDigest::sha256(b"logical"),
        CURRENT_DETERMINISTIC_ZIP64_PROFILE,
        &policy(),
        &ResourcePolicy::default(),
        &CancellationToken::new(),
        |part, data| {
            stored.insert(part.repository_path.clone(), data.to_vec());
            Ok(())
        },
    );
    assert!(result.is_ok(), "repeated content is valid: {result:?}");
    let record = result.unwrap();
    let reconstructed = record
        .ordered_parts
        .iter()
        .flat_map(|p| stored[&p.repository_path].clone())
        .collect::<Vec<_>>();
    assert_eq!(reconstructed, bytes);
    assert_eq!(stored.len(), 1);
    assert_eq!(record.schema_version, 2);
    let scratch = xc_core::test_support::TestDir::new("repeated-reconstruct");
    let root = scratch.join("root");
    std::fs::create_dir(&root).unwrap();
    for (path, data) in &stored {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, data).unwrap();
    }
    let output = root.join("encoded-package");
    reconstruct_transport_package(
        &record,
        &root,
        &output,
        &ResourcePolicy::default(),
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(std::fs::read(output).unwrap(), bytes);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn publication_batches_count_unique_physical_parts() {
    let first = TransportPart {
        sequence: 0,
        repository_path: "objects/part".into(),
        size_bytes: 4,
        content_digest: ContentDigest::sha256(b"ABCD"),
    };
    let mut second = first.clone();
    second.sequence = 1;
    let batches = plan_publication_batches(&[first, second], &policy()).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].parts.len(), 1);
    assert_eq!(batches[0].payload_bytes, 4);
}

#[test]
fn conflicting_repeated_transport_paths_are_rejected() {
    let a = TransportPart {
        sequence: 0,
        repository_path: "objects/part".into(),
        size_bytes: 4,
        content_digest: ContentDigest::sha256(b"ABCD"),
    };
    let mut b = a.clone();
    b.sequence = 1;
    b.content_digest = ContentDigest::sha256(b"WXYZ");
    let record = TransportEncodingRecord {
        schema_version: 2,
        canonical_payload_digest: ContentDigest::sha256(b"logical"),
        encoder_profile: CURRENT_DETERMINISTIC_ZIP64_PROFILE.into(),
        package_size_bytes: 8,
        package_digest: ContentDigest::sha256(b"ABCDWXYZ"),
        ordered_parts: vec![a, b],
        reconstruction: "concatenate".into(),
    };
    assert!(record.validate().is_err());
}
