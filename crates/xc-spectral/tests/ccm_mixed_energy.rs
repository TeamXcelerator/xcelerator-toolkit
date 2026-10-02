//! Both public component representations are valid;
//! explicit operators take precedence in computation and report validation.
#![cfg(feature = "hp")]

use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, retained_evidence::*, state_geometry::RetainedState};

fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    (
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(kind, "fresh-audit-mixed-energy", kind.as_bytes()).unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags: BTreeMap::new(),
            provenance_digest: None,
        },
        bytes,
    )
}

#[test]
fn validated_mixed_component_input_must_not_reject_its_fresh_report() {
    let (matrix_manifest, matrix_bytes) = source(
        "ccm_tau_matrix",
        json!({
            "schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "entries":["4","0","0","0","3","0","0","0","4"]
        }),
    );
    let (mut state_manifest, state_bytes) = source(
        "ccm_weil_eigenpair",
        json!({
            "schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]
        }),
    );
    state_manifest.dependencies.push(DependencyRef {
        key: matrix_manifest.key.clone(),
        content_digest: matrix_manifest.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let state = RetainedState::from_payload(
        &state_manifest,
        &state_bytes,
        std::slice::from_ref(&state_manifest.content_digest),
    )
    .unwrap();
    let matrix = RetainedMatrix::from_payload(
        &matrix_manifest,
        &matrix_bytes,
        std::slice::from_ref(&matrix_manifest.content_digest),
    )
    .unwrap();
    let mut input: ExternalResearchInputs = serde_json::from_value(json!({
        "schema_version":1,"source_eigenpair":state_manifest.content_digest,
        "lambda_squared":"9","n_modes":1,"precision_bits":128,
        "convention_id":"exact diagonal source",
        "definition_digest":ContentDigest::sha256(b"mixed component source"),
        "approximation_scope":"synthetic finite stored points only",
        "components":[{"label":"explicit_tau","source_digest":ContentDigest::sha256(b"explicit tau"),
            "diagonal":["4","3","4"],"dense":[],"rank_one":[]}]
    })).unwrap();
    let context = ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: None,
        ordered_overlays: vec!["disabled".into()],
        mode: ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let options = ExtensionOptions::for_source(&state);
    input.validate().unwrap();
    let baseline = capture_extended(
        "arithmetic_energy",
        &state,
        Some(&matrix),
        None,
        Some(&input),
        &options,
        &[],
        &context,
    )
    .unwrap();
    assert_eq!(baseline.value.data.rows.len(), 1);
    input.run_once = Some(
        serde_json::from_value(json!({"component_actions":[
            {"label":"compact_a","source_digest":ContentDigest::sha256(b"compact a"),
                "action":["0","1","0"],"convention":"center-oriented unit coefficient state"},
            {"label":"compact_b","source_digest":ContentDigest::sha256(b"compact b"),
                "action":["0","2","0"],"convention":"center-oriented unit coefficient state"}
        ]}))
        .unwrap(),
    );
    input
        .validate()
        .expect("the public input contract allows both representations");
    let result = capture_extended(
        "arithmetic_energy",
        &state,
        Some(&matrix),
        None,
        Some(&input),
        &options,
        &[],
        &context,
    );
    assert!(
        result.is_ok(),
        "a validated exact finite input must not fail its own fresh-output gate: {:#}",
        result.as_ref().err().unwrap()
    );
    let report = result.unwrap().value;
    if let Some(path) = std::env::var_os("XC_CCM_REPAIR_SCHEMA_PACKET") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    let mixed = report.data;
    assert_eq!(mixed.rows.len(), 1);
    assert_eq!(mixed.values, baseline.value.data.values);
    assert_eq!(
        serde_json::to_value(&mixed.rows).unwrap(),
        serde_json::to_value(&baseline.value.data.rows).unwrap(),
    );
    let mut bounded = options;
    bounded.maximum_rows = 1;
    bounded.maximum_directional_rows = 1;
    let bounded_result = capture_extended(
        "arithmetic_energy",
        &state,
        Some(&matrix),
        None,
        Some(&input),
        &bounded,
        &[],
        &context,
    )
    .unwrap();
    assert_eq!(bounded_result.value.data.outcome, "point_measurement");
    assert_eq!(
        bounded_result.value.data.rows.len(),
        1,
        "ignored compact actions must not consume the explicit-component row budget"
    );
}
