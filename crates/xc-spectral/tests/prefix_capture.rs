#![cfg(feature = "hp")]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::{
    ArtifactKey, ArtifactManifest, CacheObjectRef, CacheQuality, CacheVisibility, ContentDigest,
    ToolkitVersion,
};
use xc_spectral::ccm::prefix::*;

fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(kind, "synthetic-prefix-fixture", kind.as_bytes()).unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.14.3").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    (manifest, bytes)
}
fn matrix() -> (ArtifactManifest, Vec<u8>) {
    source(
        "ccm_even_sector_matrix",
        json!({"schema_version":1,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"dimension":2,"entries":["2","0","0","3"]}),
    )
}
fn options() -> PrefixAnalysisOptions {
    PrefixAnalysisOptions {
        working_precision_bits: 256,
        pivot_margin_bits: 32,
        checkpoint_dimensions: vec![1, 2],
        export_significant_digits: vec![80, 96],
        export_relative_tolerance: "1e-60".into(),
    }
}
fn read_matrix(m: &ArtifactManifest, b: &[u8]) -> RetainedEvenMatrix {
    RetainedEvenMatrix::from_payload(m, b, std::slice::from_ref(&m.content_digest)).unwrap()
}
#[test]
fn retained_source_is_bound_to_bytes_and_allowlist() {
    let (m, b) = matrix();
    assert!(RetainedEvenMatrix::from_payload(&m, &b, &[]).is_err());
    let mut corrupt = b.clone();
    corrupt[0] = b'[';
    assert!(RetainedEvenMatrix::from_payload(
        &m,
        &corrupt,
        std::slice::from_ref(&m.content_digest)
    )
    .is_err());
    let a = read_matrix(&m, &b);
    let before = a.entries().to_vec();
    let r = analyze_retained_prefixes(&a, &options(), &[]).unwrap();
    assert_eq!(a.entries(), before);
    assert_eq!(r.parent_matrix_source, m.content_digest);
    assert!(r.prefixes_are_parent_derived);
    assert!(r.ladder.stopped.is_none());
    assert_eq!(
        r.checkpoints[1].status,
        "innovation_export_passed_eigenpair_not_supplied"
    );
}
#[test]
fn eigenstate_is_normalized_on_a_copy_with_source_bound_overlap() {
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let (e, eb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"2","eigenvector":["0","-5","0"]}),
    );
    let pair =
        RetainedEvenEigenpair::from_payload(&e, &eb, std::slice::from_ref(&e.content_digest))
            .unwrap();
    let r = analyze_retained_prefixes(&a, &options(), &[pair]).unwrap();
    let packet = &r.checkpoints[1];
    assert_eq!(packet.status, "export_checks_passed");
    assert_eq!(packet.eigenpair_source, Some(e.content_digest));
    assert_eq!(
        Float::with_val(
            256,
            Float::parse(&packet.unit_retained_eigenvector[0]).unwrap()
        ),
        1
    );
    assert_eq!(
        Float::with_val(
            256,
            Float::parse(packet.squared_overlap.as_ref().unwrap()).unwrap()
        ),
        0
    );
    assert_eq!(
        eb,
        serde_json::to_vec(
            &json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"2","eigenvector":["0","-5","0"]})
        )
        .unwrap()
    );
}
#[test]
fn asymmetric_and_odd_sources_are_rejected() {
    let (m, b) = source(
        "ccm_even_sector_matrix",
        json!({"schema_version":1,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"dimension":2,"entries":["2","0","1","3"]}),
    );
    assert!(
        RetainedEvenMatrix::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).is_err()
    );
    let (e, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"2","eigenvector":["1","0","-1"]}),
    );
    assert!(
        RetainedEvenEigenpair::from_payload(&e, &b, std::slice::from_ref(&e.content_digest))
            .is_err()
    );
}
#[test]
fn incorrect_eigenstate_does_not_replace_sources() {
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let (e, eb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"999","eigenvector":["0","1","0"]}),
    );
    let pair =
        RetainedEvenEigenpair::from_payload(&e, &eb, std::slice::from_ref(&e.content_digest))
            .unwrap();
    let r = analyze_retained_prefixes(&a, &options(), &[pair]).unwrap();
    assert_eq!(r.checkpoints[1].status, "export_checks_unresolved");
    assert!(r.checkpoints[1].unit_retained_eigenvector.is_empty());
    assert_eq!(a.manifest(), &m);
}
#[test]
fn ten_complete_packets_are_byte_identical() {
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let run =
        || serde_json::to_vec(&analyze_retained_prefixes(&a, &options(), &[]).unwrap()).unwrap();
    let reference = run();
    for _ in 1..10 {
        assert_eq!(reference, run());
    }
}
#[test]
fn new_kind_is_private_only_without_raising_ordinary_compatibility_floors() {
    assert_eq!(
        xc_cache::artifact_compatibility_policy("ccm-evidence", PREFIX_ARTIFACT_KIND)
            .unwrap()
            .minimum_producer_version,
        ToolkitVersion::parse("0.14.4").unwrap()
    );
    assert_eq!(
        xc_cache::artifact_compatibility_policy("ccm-matrices", "ccm_tau_matrix")
            .unwrap()
            .minimum_producer_version,
        ToolkitVersion::parse("0.13.0").unwrap()
    );
}
#[test]
fn derived_cache_reuses_diagnostics_and_rejects_missing_requested_variant() {
    use xc_cache::{
        ArtifactCacheContext, ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver,
        FilesystemCacheStore,
    };
    let root = std::env::temp_dir().join(format!("ccm-prefix-child-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "prefix-test",
            root.clone(),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse("0.14.4").unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let context = |mode, write_on_miss| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["prefix-test".into()],
        mode,
        write_on_miss,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let first = analyze_retained_prefixes_via_cache(
        &a,
        &options(),
        &[],
        &context(ArtifactExecutionCacheMode::PreferReuse, true),
    )
    .unwrap();
    assert!(first.produced_manifest.is_some());
    let reused = analyze_retained_prefixes_via_cache(
        &a,
        &options(),
        &[],
        &context(ArtifactExecutionCacheMode::RequireReuse, false),
    )
    .unwrap();
    assert!(reused.reused_manifest.is_some());
    assert_eq!(first.value, reused.value);
    assert_eq!(
        first.produced_manifest.unwrap(),
        reused.reused_manifest.unwrap()
    );
    let mut other = options();
    other.export_significant_digits = vec![100];
    assert!(analyze_retained_prefixes_via_cache(
        &a,
        &other,
        &[],
        &context(ArtifactExecutionCacheMode::RequireReuse, false)
    )
    .is_err());
    assert_eq!(a.manifest(), &m);
    std::fs::remove_dir_all(root).unwrap();
}
