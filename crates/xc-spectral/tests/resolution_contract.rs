#![cfg(feature = "hp")]
use rug::{float::Constant, Float};
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
fn state(c: &str, co: &[String], p: u32) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":p,"force_even":true,"eigenvalue":"3","eigenvector":co}),
    );
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
    )
}
fn inputs(_s: &RetainedState, m: &ArtifactManifest, p: u32) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":p,"convention_id":"synthetic finite transform","definition_digest":ContentDigest::sha256(b"independent resolution test"),"approximation_scope":"finite stored points; external target premises remain conditional"})).unwrap()
}
fn jet(t: String, ordinal: usize) -> ReferenceJet {
    ReferenceJet {
        matched_root_ordinal: None,
        ordinal,
        t,
        reference_window: Jet {
            value: "0".into(),
            derivative: "0".into(),
        },
        reference_full: Jet {
            value: "0".into(),
            derivative: "0".into(),
        },
        exterior_tail: Jet {
            value: "0".into(),
            derivative: "0".into(),
        },
        endpoint_tail_part: None,
        fitted_interior_parts: vec![],
        error_normalization: Some("unit_l2_dx".into()),
        source_value_error: Some("0".into()),
        tail_value_error: None,
        source_derivative_error: Some("0".into()),
        root_separation_radius: Some("0.01".into()),
    }
}
fn report(s: &RetainedState, i: &ExternalResearchInputs, p: u32) -> ExtendedAnalysis {
    capture_extended(
        "resolution_budget",
        s,
        None,
        None,
        Some(i),
        &ExtensionOptions {
            working_precision_bits: p,
            ..ExtensionOptions::for_source(s)
        },
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
}
fn hp(s: &str) -> Float {
    Float::with_val(768, Float::parse(s).unwrap())
}
fn within(values: &BTreeMap<String, String>, name: &str, expected: &str, p: u32) {
    let expected = hp(expected);
    let lo = hp(&values[&format!("{name}_lower")]);
    let hi = hp(&values[&format!("{name}_upper")]);
    let slack = (Float::with_val(768, 1) + expected.clone().abs()) * hp("1e-160");
    assert!(
        Float::with_val(768, &lo) - &slack <= expected
            && expected <= Float::with_val(768, &hi) + &slack,
        "{name},p={p},[{lo},{hi}] excludes {expected}"
    );
    assert!(
        (hp(&values[name]) - &expected).abs()
            <= (Float::with_val(768, 1) + expected.abs()) >> (p - 6),
        "imprecise {name}"
    );
}
#[test]
fn resolution_matches_independent_defining_integrals_under_signed_binary_scaling() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/resolution_oracle.json")).unwrap();
    let mut comparisons = 0;
    for case in oracle["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for exponent in [0i32, 400, -400] {
                let co = case["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        let mut v =
                            Float::with_val(128, Float::parse(v.as_str().unwrap()).unwrap());
                        if exponent > 0 {
                            v <<= exponent as u32;
                            v = -v;
                        } else if exponent < 0 {
                            v >>= exponent.unsigned_abs();
                        }
                        v.to_string()
                    })
                    .collect::<Vec<_>>();
                let (s, m) = state(case["cutoff"].as_str().unwrap(), &co, 128);
                let mut i = inputs(&s, &m, 128);
                i.lambda_squared = case["cutoff"].as_str().unwrap().into();
                i.n_modes = co.len() / 2;
                let mut j = jet(case["t"].as_str().unwrap().into(), 1);
                j.source_value_error = Some("0.0009765625".into());
                j.source_derivative_error = Some("0.00048828125".into());
                j.root_separation_radius = Some("0.0009765625".into());
                i.reference_jets.push(j);
                let r = report(&s, &i, p);
                for (name, expected) in case["expected"].as_object().unwrap() {
                    within(
                        if name == "finite_curvature_expression" {
                            &r.values
                        } else {
                            &r.rows[0].values
                        },
                        name,
                        expected.as_str().unwrap(),
                        p,
                    );
                    comparisons += 1;
                    if name == "conditional_root_distance_allowance" {
                        assert!(
                            hp(&r.rows[0].values[name]) + hp("1e-160")
                                >= hp(expected.as_str().unwrap())
                        );
                    }
                }
            }
        }
    }
    let expected = oracle["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["expected"].as_object().unwrap().len())
        .sum::<usize>()
        * 9;
    assert_eq!(comparisons, expected);
    println!("independent integral/expression comparisons: {comparisons}");
}
#[test]
fn exact_cutoff_error_is_covered_in_the_original_false_bound_example() {
    let cutoff = include_str!("fixtures/resolution_near_one_cutoff.txt").trim();
    let p = 1536;
    let (s, m) = state(cutoff, &["0".into(), "1".into(), "0".into()], 256);
    let c0 = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 250u32);
    let t = Float::with_val(512, Float::with_val(p, Constant::Pi) * 2u32 / c0.ln());
    let mut i = inputs(&s, &m, 512);
    i.lambda_squared = cutoff.into();
    let mut j = jet(t.to_string(), 1);
    j.root_separation_radius = Some("1".into());
    i.reference_jets.push(j);
    let r = report(&s, &i, 512);
    let row = &r.rows[0];
    let l = Float::with_val(p, Float::parse(cutoff).unwrap()).ln();
    let root = Float::with_val(p, Constant::Pi) * 2u32 / &l;
    let distance = (root - Float::with_val(p, &t)).abs();
    let value = Float::with_val(p, 2) * (Float::with_val(p, &t) * &l / 2u32).sin()
        / (Float::with_val(p, &t) * l.clone().sqrt());
    let decode = |s: &str| Float::with_val(p, Float::parse(s).unwrap());
    let bound = decode(&row.values["conditional_root_distance_allowance"]);
    assert!(distance < 1 && bound >= distance && Float::with_val(p, &bound) / &distance < 2);
    assert!(
        decode(&row.values["transform_lower"]) <= value
            && decode(&row.values["transform_upper"]) >= value
    );
    assert!(
        decode(&r.values["finite_curvature_expression"]) > l.clone().square() * l.sqrt() / 12u32
    );
    assert_eq!(row.outcome, "conditional_budget_met");
    assert_eq!(hp(&r.values["conditional_contiguous_prefix"]), 1);
    assert!(hp(&row.values["arithmetic_precision_bits"]) > 576);
}
#[test]
fn original_points_control_ordering_errors_and_indexed_transforms() {
    let alias = "1.0000000000000000000000000000000000000001";
    let (s, m) = state("9", &["0".into(), "1".into(), "0".into()], 128);
    let mut i = inputs(&s, &m, 128);
    i.reference_jets = vec![jet("1".into(), 1), jet("1".into(), 2)];
    let first = report(&s, &i, 192);
    i.reference_jets[0].t = alias.into();
    i.reference_jets[1].t = alias.into();
    let second = report(&s, &i, 192);
    for (a, b) in first.rows.iter().zip(&second.rows) {
        assert_eq!(a.values, b.values);
        assert!(!b.values.contains_key("reference_neighbor_spacing"));
    }
    i.reference_jets[0].source_value_error = Some("1".into());
    let first = report(&s, &i, 192);
    i.reference_jets[0].source_value_error = Some(alias.into());
    let second = report(&s, &i, 192);
    assert_eq!(first.rows[0].values, second.rows[0].values);
    let mut values = Vec::new();
    for t in ["1", alias] {
        let spec = DatasetSpec {
            schema_version: 1,
            role: "evaluation_points".into(),
            attribution: "stored point alias".into(),
            coordinate: "mellin_t".into(),
            precision_bits: 128,
            points: vec![EvaluationPoint {
                ordinal: 1,
                value: Some(t.into()),
                source_status: "supplied".into(),
            }],
        };
        let packet = capture_dataset(&spec, &context()).unwrap().value;
        let (dm, db) = source(
            "research_reference_dataset",
            serde_json::to_value(packet).unwrap(),
        );
        let d = RetainedDataset::from_payload(&dm, &db, std::slice::from_ref(&dm.content_digest))
            .unwrap();
        let r = capture_transforms_at_dataset(
            &s,
            &d,
            &TransformOptions::for_dataset(&s, &d),
            &context(),
        )
        .unwrap();
        assert_eq!(
            r.value.request["transform_arithmetic"],
            "exact_cutoff_stored_points_directed_sinc_v1"
        );
        let row = &r.value.data.rows[0];
        values.push((
            row.value.clone(),
            row.derivative.clone(),
            row.newton_correction.clone(),
        ));
    }
    assert_eq!(values[0], values[1]);
}
#[test]
fn conditional_prefix_gaps_failures_and_resource_limits_remain_explicit() {
    let p = 192;
    let (s, m) = state("9", &["0".into(), "1".into(), "0".into()], 128);
    let mut i = inputs(&s, &m, 128);
    let root = Float::with_val(512, Constant::Pi) * 2u32 / Float::with_val(512, 9).ln();
    i.reference_jets = (1..=3)
        .map(|k| {
            jet(
                Float::with_val(128, Float::with_val(512, &root) * k).to_string(),
                k,
            )
        })
        .collect();
    let r = report(&s, &i, p);
    assert!(r.rows.iter().all(|v| v.outcome == "conditional_budget_met"));
    assert_eq!(hp(&r.values["conditional_contiguous_prefix"]), 3);
    i.reference_jets[1].source_derivative_error = Some("100".into());
    let r = report(&s, &i, p);
    assert_eq!(r.rows[1].outcome, "conditional_budget_unresolved");
    assert_eq!(hp(&r.values["conditional_contiguous_prefix"]), 1);
    i.reference_jets.remove(1);
    let r = report(&s, &i, p);
    assert_eq!(hp(&r.values["conditional_contiguous_prefix"]), 1);
    i.reference_jets.remove(0);
    let r = report(&s, &i, p);
    assert_eq!(hp(&r.values["conditional_contiguous_prefix"]), 0);
    let options = ExtensionOptions {
        maximum_working_bytes: Some(1),
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "resolution_budget",
        &s,
        None,
        None,
        Some(&i),
        &options,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.rows.is_empty());
    i.reference_jets = vec![jet("0".into(), 1)];
    let r = report(&s, &i, p);
    assert_eq!(r.rows[0].outcome, "unresolved_derivative");
    assert!(!r.rows[0]
        .values
        .contains_key("conditional_root_distance_allowance"));
}
#[test]
fn retained_root_join_uses_root_storage_precision_and_keeps_missing_rows() {
    let (s, m) = state("9", &["0".into(), "1".into(), "0".into()], 128);
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenpair_content_digest":m.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":"1.0000000000000000000000000000000000000001"}},{"status":"failed","details":{"root_index":2,"error":"synthetic missing root"}}]}),
    );
    rm.dependencies.push(DependencyRef {
        key: sec.key.clone(),
        content_digest: sec.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let roots = RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &sb,
        &s,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap();
    let mut i = inputs(&s, &m, 192);
    let mut j = jet("1".into(), 1);
    j.matched_root_ordinal = Some(1);
    i.reference_jets.push(j);
    let r = capture_extended(
        "resolution_budget",
        &s,
        None,
        Some(&roots),
        Some(&i),
        &ExtensionOptions::for_source(&s),
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(
        hp(&r.rows[0].values["matched_root_reference_displacement"]),
        0
    );
    let r = capture_extended(
        "resolution_budget",
        &s,
        None,
        Some(&roots),
        None,
        &ExtensionOptions::for_source(&s),
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.rows.len(), 2);
    assert_eq!(hp(&r.rows[0].values["t"]), 1);
    assert_eq!(r.rows[1].outcome, "missing_input");
    assert_eq!(r.rows[1].ordinal, 2);
}
