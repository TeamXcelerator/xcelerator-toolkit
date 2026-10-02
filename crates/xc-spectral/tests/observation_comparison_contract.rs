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
use xc_spectral::ccm::{
    convergence_capture::{RunOnceInputs, SourceUncertainty},
    research_completion::{ComparisonSnapshot, CompletionInputs},
};
fn state(
    c: &str,
    co: &[String],
    e: &str,
    entries: Option<&[String]>,
) -> (
    RetainedState,
    ArtifactManifest,
    Option<RetainedMatrix<'static>>,
) {
    let n = co.len() / 2;
    let mut matrix = None;
    let (mut m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":n,"precision_bits":128,"force_even":true,"eigenvalue":e,"eigenvector":co}),
    );
    if let Some(entries) = entries {
        let (mm, mb) = source(
            "ccm_tau_matrix",
            json!({"schema_version":2,"lambda_squared":c,"n_modes":n,"precision_bits":128,"entries":entries}),
        );
        m.dependencies.push(DependencyRef {
            key: mm.key.clone(),
            content_digest: mm.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        matrix = Some(
            RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest))
                .unwrap(),
        );
    }
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
        matrix,
    )
}
fn input(m: &ArtifactManifest, c: &str, n: usize) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":c,"n_modes":n,"precision_bits":128,"convention_id":"independent comparison fixture","definition_digest":ContentDigest::sha256(b"independent finite algebra"),"approximation_scope":"synthetic stored points; no scientific selection assertion"})).unwrap()
}
fn comparison(c: &str, co: &[String], e: &str) -> ComparisonSnapshot {
    serde_json::from_value(json!({"state":{"source_digest":ContentDigest::sha256(b"comparison"),"matrix_digest":ContentDigest::sha256(b"comparison matrix"),"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":128,"coefficients":co,"matrix":[],"eigenvalue":e,"assembly_policy":"supplied finite points"},"selection_policy":"explicit point","assembly_policy":"same Fourier convention","quadrature_policy":"none","root_branch":"none","root_coordinate":"mellin_t","roots":[]})).unwrap()
}
fn with_comparison(i: &mut ExternalResearchInputs, c: ComparisonSnapshot) {
    i.run_once = Some(RunOnceInputs {
        completion: Some(CompletionInputs {
            comparisons: vec![c],
            ..Default::default()
        }),
        ..Default::default()
    });
}
fn roots(
    s: &RetainedState,
    m: &ArtifactManifest,
    c: &str,
    n: usize,
    points: &[Option<String>],
) -> RetainedRoots {
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":c,"n_modes":n,"precision_bits":128,"force_even":true,"eigenpair_content_digest":m.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let outcomes = points
        .iter()
        .map(|v| {
            if let Some(v) = v {
                json!({"status":"converged","details":{"value":v}})
            } else {
                json!({"status":"failed","details":{"error":"synthetic missing point"}})
            }
        })
        .collect::<Vec<_>>();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":c,"n_modes":n,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":outcomes}),
    );
    rm.dependencies.push(DependencyRef {
        key: sec.key.clone(),
        content_digest: sec.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &sb,
        s,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap()
}
fn run(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    roots: Option<&RetainedRoots>,
    i: &ExternalResearchInputs,
    p: u32,
) -> ExtendedAnalysis {
    capture_extended(
        id,
        s,
        m,
        roots,
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
fn within(v: &BTreeMap<String, String>, name: &str, expected: &str, p: u32) {
    let expected = hp(expected);
    let lo = hp(&v[&format!("{name}_lower")]);
    let hi = hp(&v[&format!("{name}_upper")]);
    let slack = (Float::with_val(768, 1) + expected.clone().abs()) * hp("1e-155");
    assert!(
        Float::with_val(768, &lo) - &slack <= expected
            && expected <= Float::with_val(768, &hi) + &slack,
        "{name}: [{lo},{hi}] excludes {expected}"
    );
    assert!(
        (hp(&v[name]) - &expected).abs() <= (Float::with_val(768, 1) + expected.abs()) >> (p - 6),
        "inaccurate {name}"
    );
}
fn strings(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().into())
        .collect()
}
fn scaled(v: &[String], e: i32) -> Vec<String> {
    v.iter()
        .map(|v| {
            let mut v = Float::with_val(128, Float::parse(v).unwrap());
            if e >= 0 {
                v <<= e as u32;
                v = -v;
            } else {
                v >>= e.unsigned_abs();
            }
            v.to_string()
        })
        .collect()
}
#[test]
fn comparison_matches_independent_fraction_residuals_and_normalized_overlaps() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/comparison_oracle.json")).unwrap();
    let mut count = 0;
    for case in oracle["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for e in [0i32, 400, -400] {
                let c = case["cutoff"].as_str().unwrap();
                let x = scaled(&strings(&case["source_coefficients"]), -e);
                let y = scaled(&strings(&case["comparison_coefficients"]), e);
                let entries = strings(&case["matrix"]);
                let (s, sm, m) = state(
                    c,
                    &x,
                    case["source_eigenvalue"].as_str().unwrap(),
                    Some(&entries),
                );
                let mut i = input(&sm, c, x.len() / 2);
                with_comparison(
                    &mut i,
                    comparison(c, &y, case["comparison_eigenvalue"].as_str().unwrap()),
                );
                let r = run("configuration_comparison", &s, m.as_ref(), None, &i, p);
                assert_eq!(r.rows[0].outcome, "point_measurement");
                for (name, value) in case["expected"].as_object().unwrap() {
                    within(&r.rows[0].values, name, value.as_str().unwrap(), p);
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 1296);
}
#[test]
fn original_scale_energy_and_shifted_residual_failures_are_repaired() {
    let co = vec!["1".into(), "2".into(), "1".into()];
    let (s, m, _) = state("9", &co, "3", None);
    for (y, eigen) in [
        (co, "3"),
        (
            vec!["1".into(), "2".into(), "1".into()],
            "3.0000000000000000000000000000000000000001",
        ),
        (
            vec![
                "1e-200000000".into(),
                "2e-200000000".into(),
                "1e-200000000".into(),
            ],
            "3",
        ),
        (
            vec![
                "1e200000000".into(),
                "2e200000000".into(),
                "1e200000000".into(),
            ],
            "3",
        ),
    ] {
        let mut i = input(&m, "9", 1);
        with_comparison(&mut i, comparison("9", &y, eigen));
        let r = run("configuration_comparison", &s, None, None, &i, 192);
        assert_eq!(r.rows[0].outcome, "point_measurement");
        within(&r.rows[0].values, "absolute_unit_overlap", "1", 192);
        within(&r.rows[0].values, "signed_energy_difference", "0", 192);
    }
    for exponent in [400u32, 10000] {
        let power = Float::with_val(128, 1) << exponent;
        for big in [power.to_string(), power.to_integer().unwrap().to_string()] {
            let co = vec!["1".into(); 3];
            let entries = (0..3)
                .flat_map(|a| {
                    (0..3)
                        .map(|b| if a == b { big.clone() } else { "1".into() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let (s, sm, m) = state("9", &co, &big, Some(&entries));
            let mut i = input(&sm, "9", 1);
            with_comparison(&mut i, comparison("9", &co, &big));
            let r = run("configuration_comparison", &s, m.as_ref(), None, &i, 192);
            within(
                &r.rows[0].values,
                "independent_small_state_low_residual_squared",
                "4",
                192,
            );
            within(
                &r.rows[0].values,
                "independent_small_state_high_forcing_squared",
                "0",
                192,
            );
        }
    }
}
#[test]
fn observation_channels_match_direct_integrals_and_cauchy_schwarz_transport() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/observation_oracle.json")).unwrap();
    let mut count = 0;
    for case in oracle["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            let c = case["cutoff"].as_str().unwrap();
            let co = strings(&case["coefficients"]);
            let (s, m, _) = state(c, &co, "3", None);
            let roots = roots(
                &s,
                &m,
                c,
                co.len() / 2,
                &[Some(case["t"].as_str().unwrap().into())],
            );
            let mut i = input(&m, c, co.len() / 2);
            i.run_once = Some(RunOnceInputs {
                uncertainty: Some(SourceUncertainty {
                    unit_state_l2_error: "0.1875".into(),
                    source_certificate_digest: ContentDigest::sha256(b"external test premise"),
                    hypotheses: vec!["conditional finite-support L2 error".into()],
                }),
                ..Default::default()
            });
            let r = run("observable_budget", &s, None, Some(&roots), &i, p);
            for (name, value) in case["expected"].as_object().unwrap() {
                let values = if [
                    "transform_origin",
                    "declared_origin_error",
                    "conditional_origin_lower_margin",
                ]
                .contains(&name.as_str())
                {
                    &r.values
                } else {
                    &r.rows[0].values
                };
                within(values, name, value.as_str().unwrap(), p);
                count += 1;
                if name.ends_with("_lower_margin") {
                    assert_eq!(values[name], values[&format!("{name}_lower")]);
                } else if name.ends_with("_error") {
                    assert_eq!(values[name], values[&format!("{name}_upper")]);
                }
            }
        }
    }
    assert_eq!(count, 1440);
}
#[test]
fn observation_aliases_missing_rows_and_memory_limits_preserve_qualification() {
    let co = vec!["0".into(), "1".into(), "0".into()];
    let (s, m, _) = state("9", &co, "3", None);
    let mut i = input(&m, "9", 1);
    let mut reports = Vec::new();
    for error in ["1", "0.9999999999999999999999999999999999999999"] {
        i.run_once = Some(RunOnceInputs {
            uncertainty: Some(SourceUncertainty {
                unit_state_l2_error: error.into(),
                source_certificate_digest: ContentDigest::sha256(b"error point alias"),
                hypotheses: vec!["conditional test premise".into()],
            }),
            ..Default::default()
        });
        let r = run("observable_budget", &s, None, None, &i, 192);
        // Error declarations are exact decimal upper bounds, unlike the stored
        // state points. The two declarations must remain distinguishable.
        let length_root = hp("9").ln().sqrt();
        let expected = (Float::with_val(768, 1) - hp(error)) * &length_root;
        let lower = hp(&r.values["conditional_origin_lower_margin"]);
        if error == "1" {
            within(&r.values, "conditional_origin_lower_margin", "0", 192);
            assert!(lower <= 0);
        } else {
            // The producer first rounds the bound upward at working precision.
            // The reported margin is a conservative lower bound, so its narrow
            // arithmetic enclosure need not contain the unrounded exact margin.
            assert!(lower > 0 && lower <= expected);
            assert!(expected - lower <= (Float::with_val(768, 4) >> 192));
        }
        assert!(hp(&r.values["declared_origin_error"]) >= hp(error) * length_root);
        reports.push(r.values);
    }
    assert_ne!(reports[0], reports[1]);
    let roots = roots(
        &s,
        &m,
        "9",
        1,
        &[
            Some("1.0000000000000000000000000000000000000001".into()),
            None,
        ],
    );
    i.run_once = None;
    let r = run("observable_budget", &s, None, Some(&roots), &i, 192);
    assert_eq!(hp(&r.rows[0].values["t"]), 1);
    assert_eq!(r.rows[0].outcome, "channels_resolved_budget_unassessed");
    assert_eq!(r.rows[1].outcome, "missing_input");
    assert!(!r.rows[0].values.contains_key("conditional_value_error"));
    let o = ExtensionOptions {
        maximum_working_bytes: Some(1),
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "observable_budget",
        &s,
        None,
        Some(&roots),
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.values.is_empty());
}
#[test]
fn comparison_root_transforms_match_independent_integrals_at_the_retained_ordinate() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/resolution_oracle.json")).unwrap();
    let cases = oracle["rows"].as_array().unwrap();
    let mut count = 0;
    for profile in 0..12 {
        let a = cases
            .iter()
            .find(|v| v["profile"] == profile && v["ordinal"] == 3)
            .unwrap();
        let b = cases
            .iter()
            .find(|v| v["profile"] == (profile + 1) % 12 && v["ordinal"] == 3)
            .unwrap();
        let cutoff = a["cutoff"].as_str().unwrap();
        let co = strings(&a["coefficients"]);
        let (s, m, _) = state(cutoff, &co, "3", None);
        let roots = roots(
            &s,
            &m,
            cutoff,
            co.len() / 2,
            &[Some(a["t"].as_str().unwrap().into()), None],
        );
        let branch = serde_json::to_string(
            &capture_root_window(&roots, &context())
                .unwrap()
                .value
                .data
                .source_acquisition,
        )
        .unwrap();
        let mut comp = comparison(
            b["cutoff"].as_str().unwrap(),
            &strings(&b["coefficients"]),
            "3",
        );
        comp.root_branch = branch;
        comp.roots = vec![
            EvaluationPoint {
                ordinal: 1,
                value: Some("0.75".into()),
                source_status: "supplied".into(),
            },
            EvaluationPoint {
                ordinal: 2,
                value: Some("3".into()),
                source_status: "supplied".into(),
            },
        ];
        let mut i = input(&m, cutoff, co.len() / 2);
        with_comparison(&mut i, comp);
        for p in [128, 192, 256] {
            let r = run("configuration_comparison", &s, None, Some(&roots), &i, p);
            let values = &r.rows[0].values;
            within(values, "root_1_signed_difference", "-0.25", p);
            assert!(!values.contains_key("root_2_signed_difference"));
            for (field, label) in [("transform", "value"), ("derivative", "slope")] {
                let expected = (hp(a["expected"][field].as_str().unwrap())
                    - hp(b["expected"][field].as_str().unwrap()))
                .to_string();
                within(
                    values,
                    &format!("root_1_transform_{label}_difference"),
                    &expected,
                    p,
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, 72);
}
#[test]
fn comparison_cutoff_equivalence_zero_vectors_policy_gates_and_budgets_are_explicit() {
    let co = vec!["1".into(), "2".into(), "1".into()];
    let (s, m, _) = state("9", &co, "3", None);
    let mut i = input(&m, "9", 1);
    let comp = comparison("9.0", &co, "3");
    with_comparison(&mut i, comp.clone());
    let r = run("configuration_comparison", &s, None, None, &i, 192);
    within(&r.rows[0].values, "absolute_unit_overlap", "1", 192);
    i.run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .comparisons
        .push(comparison("9", &co, "3"));
    let r = run("configuration_comparison", &s, None, None, &i, 192);
    assert_eq!(r.rows[1].outcome, "unresolved_denominator");
    assert!(r.rows[1].values.is_empty());
    let mut zero = comp.clone();
    zero.state.coefficients = vec!["0".into(); 3];
    with_comparison(&mut i, zero);
    let r = run("configuration_comparison", &s, None, None, &i, 192);
    assert_eq!(r.rows[0].outcome, "unresolved_denominator");
    assert!(!r.rows[0].values.contains_key("absolute_unit_overlap"));
    let mut high = comp.clone();
    high.state.precision_bits = 256;
    with_comparison(&mut i, high);
    let r = run("configuration_comparison", &s, None, None, &i, 192);
    assert_eq!(r.rows[0].outcome, "budget_limited");
    with_comparison(&mut i, comp.clone());
    let o = ExtensionOptions {
        maximum_working_bytes: Some(1),
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "configuration_comparison",
        &s,
        None,
        None,
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.rows[0].outcome, "budget_limited");
    let points = (1..=50).map(|j| Some(j.to_string())).collect::<Vec<_>>();
    let roots = roots(&s, &m, "9", 1, &points);
    let mut joined = comp;
    joined.root_branch = serde_json::to_string(
        &capture_root_window(&roots, &context())
            .unwrap()
            .value
            .data
            .source_acquisition,
    )
    .unwrap();
    joined.roots = points
        .iter()
        .enumerate()
        .map(|(j, v)| EvaluationPoint {
            ordinal: j + 1,
            value: v.clone(),
            source_status: "supplied".into(),
        })
        .collect();
    with_comparison(&mut i, joined);
    let o = ExtensionOptions {
        maximum_estimated_output_bytes: 30000,
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "configuration_comparison",
        &s,
        None,
        Some(&roots),
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.rows[0].outcome, "budget_limited");
    i.run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .comparisons[0]
        .root_branch = "different branch".into();
    let r = run("configuration_comparison", &s, None, Some(&roots), &i, 192);
    assert!(!r.rows[0].values.contains_key("root_1_signed_difference"));
}

#[test]
fn comparison_preserves_independently_declared_eigenvalue_and_root_precisions() {
    let co = vec!["1".into(), "2".into(), "1".into()];
    let (s, m, _) = state("9", &co, "3.0000000000000000000000000000000000000001", None);
    let roots = roots(
        &s,
        &m,
        "9",
        1,
        &[Some("1.0000000000000000000000000000000000000001".into())],
    );
    let mut comp = comparison(
        "9",
        &co,
        "3.000000000000000000000000000000000000000000000000000000000001",
    );
    comp.state.precision_bits = 256;
    comp.root_branch = serde_json::to_string(
        &capture_root_window(&roots, &context())
            .unwrap()
            .value
            .data
            .source_acquisition,
    )
    .unwrap();
    comp.roots = vec![EvaluationPoint {
        ordinal: 1,
        value: Some("1.0000000000000000000000000000000000000001".into()),
        source_status: "supplied".into(),
    }];
    let eigen = Float::with_val(256, Float::parse(&comp.state.eigenvalue).unwrap());
    let ordinate = Float::with_val(
        256,
        Float::parse(comp.roots[0].value.as_ref().unwrap()).unwrap(),
    );
    let mut i = input(&m, "9", 1);
    with_comparison(&mut i, comp);
    let r = run("configuration_comparison", &s, None, Some(&roots), &i, 256);
    let values = &r.rows[0].values;
    within(
        values,
        "signed_energy_difference",
        &(Float::with_val(768, 3) - eigen).to_string(),
        256,
    );
    within(
        values,
        "root_1_signed_difference",
        &(Float::with_val(768, 1) - ordinate).to_string(),
        256,
    );
    assert!(
        hp(&values["signed_energy_difference"]) < 0 && hp(&values["root_1_signed_difference"]) < 0
    );
    within(values, "root_1_transform_value_difference", "0", 256);
    within(values, "root_1_transform_slope_difference", "0", 256);
}
