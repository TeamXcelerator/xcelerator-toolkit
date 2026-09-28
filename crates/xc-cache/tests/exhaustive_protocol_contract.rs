use std::collections::BTreeMap;
use xc_cache::*;
fn payload() -> CanonicalPayloadEnvelope {
    CanonicalPayloadEnvelope {
        schema_version: 1,
        scalar_backend: "opaque".into(),
        precision_bits: None,
        scalar_representation: "bytes".into(),
        dimensions: vec![],
        endianness: "none".into(),
        special_value_encoding: "none".into(),
        ordered_items: vec![LogicalPayloadItem {
            normalized_path: "a".into(),
            content_digest: ContentDigest::sha256(b"a"),
            size_bytes: 1,
        }],
        dependencies: vec![],
    }
}
#[test]
fn canonical_payload_validation_rejects_total_size_overflow() {
    let mut p = payload();
    p.ordered_items[0].size_bytes = u64::MAX;
    p.ordered_items.push(LogicalPayloadItem {
        normalized_path: "b".into(),
        content_digest: ContentDigest::sha256(b"b"),
        size_bytes: 1,
    });
    assert!(p.validate().is_err());
}
#[test]
fn canonical_payload_paths_reject_windows_drive_and_stream_aliases() {
    for path in ["C:/absolute", "payload:stream", "contains\0nul"] {
        let mut p = payload();
        p.ordered_items[0].normalized_path = path.into();
        assert!(p.validate().is_err(), "accepted {path:?}");
    }
}
#[test]
fn assurance_history_must_end_at_the_reported_level() {
    let s = ArtifactState {
        completion: ArtifactCompletionState::Complete,
        achieved_assurance: ArtifactAssuranceState::Certified,
        assurance_history: vec![AssuranceTransition {
            from: ArtifactAssuranceState::Unchecked,
            to: ArtifactAssuranceState::Computed,
            evidence_digest: ContentDigest::sha256(b"computed"),
        }],
        ..ArtifactState::default()
    };
    assert!(s.validate().is_err());
}
#[test]
fn assurance_history_must_be_monotonic_connected_and_digest_bound() {
    for history in [
        vec![AssuranceTransition {
            from: ArtifactAssuranceState::Certified,
            to: ArtifactAssuranceState::Computed,
            evidence_digest: ContentDigest::sha256(b"x"),
        }],
        vec![AssuranceTransition {
            from: ArtifactAssuranceState::Unchecked,
            to: ArtifactAssuranceState::Computed,
            evidence_digest: ContentDigest("bad".into()),
        }],
        vec![
            AssuranceTransition {
                from: ArtifactAssuranceState::Unchecked,
                to: ArtifactAssuranceState::StructurallyValidated,
                evidence_digest: ContentDigest::sha256(b"x"),
            },
            AssuranceTransition {
                from: ArtifactAssuranceState::Unchecked,
                to: ArtifactAssuranceState::Computed,
                evidence_digest: ContentDigest::sha256(b"x"),
            },
        ],
    ] {
        let s = ArtifactState {
            completion: ArtifactCompletionState::Complete,
            achieved_assurance: ArtifactAssuranceState::Computed,
            assurance_history: history,
            ..ArtifactState::default()
        };
        assert!(s.validate().is_err());
    }
}
#[test]
fn attestation_rejects_malformed_deserialized_version() {
    let v = ToolkitVersion {
        major: 0,
        minor: 15,
        patch: 1,
        prerelease: Some("01".into()),
    };
    let a = AttestationEnvelope {
        schema_version: 1,
        kind: AttestationKind::Validation,
        subject_digest: ContentDigest::sha256(b"subject"),
        actor: "test".into(),
        policy_digest: ContentDigest::sha256(b"policy"),
        execution_fingerprint_digest: ContentDigest::sha256(b"execution"),
        producer_toolkit_version: v,
        dependency_versions: BTreeMap::from([("test".into(), "1".into())]),
        source_revision: "test".into(),
        event_unix_seconds: 0,
        location: None,
        evidence_digests: vec![],
    };
    assert!(a.digest().is_err());
}
