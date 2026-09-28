#![cfg(feature = "hp")]
use rug::Float;
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
            key: ArtifactKey::new(kind, "synthetic-extension-fixture", kind.as_bytes()).unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.15.1").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.15.1").unwrap(),
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
fn context() -> ArtifactCacheContext<'static> {
    ArtifactCacheContext {
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
    }
}
fn fixture() -> (RetainedState, ArtifactManifest, RetainedMatrix<'static>) {
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["4","0","0","0","3","0","0","0","4"]}),
    );
    let (mut m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    m.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    (s, m, matrix)
}
fn inputs(m: &ArtifactManifest) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":128,"convention_id":"synthetic test values","definition_digest":ContentDigest::sha256(b"synthetic definition"),"approximation_scope":"finite test points"})).unwrap()
}
fn run(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    i: Option<&ExternalResearchInputs>,
) -> ExtendedAnalysis {
    capture_extended(
        id,
        s,
        m,
        None,
        i,
        &ExtensionOptions::for_source(s),
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
}

#[test]
fn allowance_original_counterexamples_preserve_points_and_scale() {
    let (state, manifest, _matrix) = fixture();
    for (case, u, mu, h) in [
        ("unit", "0", "1", "1"),
        ("tiny_common_scale", "0", "1e-200000000", "1e-200000000"),
        ("huge_common_scale", "0", "1e200000000", "1e200000000"),
        (
            "stored_zero_gap",
            "1",
            "1.0000000000000000000000000000000000000001",
            "1",
        ),
        ("equal_scale", "1", "2", "1"),
        (
            "same_stored_equal_scale",
            "1",
            "2.0000000000000000000000000000000000000001",
            "1",
        ),
    ] {
        let mut input = inputs(&manifest);
        input.energy_allowance = Some(EnergyAllowance {
            upper_trial_energy: u.into(),
            low_block_lower_bound: "0".into(),
            high_block_lower_bound: mu.into(),
            cross_block_norm_bound: h.into(),
            hypothesis_record_digest: ContentDigest::sha256(
                b"synthetic finite conditional block assumptions",
            ),
            hypotheses: vec!["declared finite block bounds; not independently certified".into()],
        });
        let report = run("energy_allowance", &state, None, Some(&input));
        if case == "stored_zero_gap" {
            assert_eq!(report.outcome, "sufficient_bound_unavailable");
            assert_eq!(point(&report.values["denominator"], 192), 0);
            assert!(!report.values.contains_key("conditional_energy_allowance"));
        } else if case == "equal_scale" || case == "same_stored_equal_scale" {
            assert_eq!(
                point(&report.values["conditional_energy_allowance"], 192),
                1
            );
            assert_eq!(
                point(
                    &report.values["allowance_below_trial_energy_magnitude"],
                    192
                ),
                0
            );
            assert_eq!(point(&report.values["scale_comparison_resolved"], 192), 1);
        } else {
            assert_eq!(
                point(&report.values["conditional_energy_allowance"], 192),
                point(h, 128)
            );
            assert!(point(&report.values["conditional_energy_allowance"], 192) > 0);
            assert_eq!(
                point(&report.values["conditional_vector_allowance"], 192),
                1
            );
        }
    }
}

fn hp(s: &str) -> Float {
    Float::with_val(768, Float::parse(s).unwrap())
}
fn point(s: &str, p: u32) -> Float {
    Float::with_val(768, Float::with_val(p, Float::parse(s).unwrap()))
}
fn within(map: &BTreeMap<String, String>, name: &str, expected: &Float, p: u32) {
    let lo = hp(&map[&format!("{name}_lower")]);
    let hi = hp(&map[&format!("{name}_upper")]);
    assert!(
        lo <= *expected && *expected <= hi,
        "{name},p={p}: [{lo},{hi}] excludes {expected}"
    );
    let tolerance = (Float::with_val(768, 1) + expected.clone().abs()) >> (p - 8);
    assert!(
        (hp(&map[name]) - expected).abs() <= tolerance,
        "midpoint {name},p={p}"
    );
}
#[test]
fn allowances_match_independent_fraction_formulas_and_strict_comparisons() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/allowance_oracle.json")).unwrap();
    let mut compared = 0;
    let (state, manifest, _) = fixture();
    for case in oracle["cases"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for exponent in [0i32, 400, -400] {
                let scale = |mut v: Float| {
                    if exponent >= 0 {
                        v <<= exponent as u32;
                    } else {
                        v >>= exponent.unsigned_abs();
                    }
                    v
                };
                let mut a = case["input"].clone();
                for value in a.as_object_mut().unwrap().values_mut() {
                    *value = json!(scale(Float::with_val(
                        128,
                        Float::parse(value.as_str().unwrap()).unwrap()
                    ))
                    .to_string());
                }
                a["hypothesis_record_digest"] = json!(ContentDigest::sha256(
                    b"independent rational block expressions"
                ));
                a["hypotheses"] = json!(["external conditional block assumptions"]);
                let mut input = inputs(&manifest);
                input.energy_allowance = Some(serde_json::from_value(a).unwrap());
                let o = ExtensionOptions {
                    working_precision_bits: p,
                    ..ExtensionOptions::for_source(&state)
                };
                let report = capture_extended(
                    "energy_allowance",
                    &state,
                    None,
                    None,
                    Some(&input),
                    &o,
                    &[],
                    &context(),
                )
                .unwrap()
                .value
                .data;
                assert_eq!(
                    report.outcome,
                    if case["valid_margin"].as_bool().unwrap() {
                        "conditional_bound_expression"
                    } else {
                        "sufficient_bound_unavailable"
                    }
                );
                for (name, value) in case["reference"].as_object().unwrap() {
                    let mut expected = hp(value.as_str().unwrap());
                    if ![
                        "conditional_vector_allowance",
                        "allowance_to_trial_energy_magnitude",
                        "allowance_below_trial_energy_magnitude",
                        "scale_comparison_resolved",
                    ]
                    .contains(&name.as_str())
                    {
                        expected = scale(expected);
                    }
                    within(&report.values, name, &expected, p);
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 4185);
}
#[test]
fn allowance_ambiguous_strict_comparison_and_resource_range_limits_are_explicit() {
    let (state, manifest, _) = fixture();
    let mut input = inputs(&manifest);
    let mu = (Float::with_val(128, 1) << 10000u32).to_string();
    let h = (Float::with_val(128, 1) << 5000u32).to_string();
    input.energy_allowance = Some(EnergyAllowance {
        upper_trial_energy: "-1".into(),
        low_block_lower_bound: "-2".into(),
        high_block_lower_bound: mu,
        cross_block_norm_bound: h,
        hypothesis_record_digest: ContentDigest::sha256(b"near strict boundary"),
        hypotheses: vec!["external block assumptions".into()],
    });
    // Exactly mu/(mu+1)<1, but the strict comparison exceeds the guard cap.
    let report = run("energy_allowance", &state, None, Some(&input));
    assert_eq!(report.outcome, "conditional_bound_expression");
    assert_eq!(point(&report.values["scale_comparison_resolved"], 192), 0);
    assert_eq!(
        point(
            &report.values["allowance_below_trial_energy_magnitude"],
            192
        ),
        0
    );
    assert!(report.reason.unwrap().contains("unresolved"));
    assert_eq!(
        point(&report.values["arithmetic_precision_bits"], 192),
        4288
    );
    let o = ExtensionOptions {
        maximum_working_bytes: Some(1),
        ..ExtensionOptions::for_source(&state)
    };
    let report = capture_extended(
        "energy_allowance",
        &state,
        None,
        None,
        Some(&input),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(report.outcome, "unresolved");
    assert!(report.values.is_empty());
    let a = input.energy_allowance.as_mut().unwrap();
    a.upper_trial_energy = "1e-200000000".into();
    a.high_block_lower_bound = "1e200000000".into();
    a.cross_block_norm_bound = "1".into();
    assert!(capture_extended(
        "energy_allowance",
        &state,
        None,
        None,
        Some(&input),
        &ExtensionOptions::for_source(&state),
        &[],
        &context()
    )
    .is_err());
}
