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
fn inputs(m: &ArtifactManifest) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":128,"convention_id":"synthetic test values","definition_digest":ContentDigest::sha256(b"synthetic definition"),"approximation_scope":"finite test points"})).unwrap()
}
#[test]
fn directional_operator_identities_and_resource_limits() {
    let p = 192;
    let co = ["1", "3", "1"];
    let entries = (0..3)
        .flat_map(|a| {
            (0..3).map(move |b| {
                (Float::with_val(p, if a == b { 2 } else { 0 })
                    - Float::with_val(
                        p,
                        co[a].parse::<u32>().unwrap() * co[b].parse::<u32>().unwrap(),
                    ) / 11u32)
                    .to_string()
            })
        })
        .collect::<Vec<_>>();
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":entries}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":co}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let state =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let (mut sec, secbytes) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: sm.key.clone(),
        content_digest: sm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let t = Float::with_val(p, Constant::Pi) * 2u32 * (Float::with_val(p, 3) / 5u32).sqrt()
        / Float::with_val(p, 9).ln();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":t.to_string()}},{"status":"converged","details":{"value":"0"}}]}),
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
        &secbytes,
        &state,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap();

    let mut measurements = Vec::new();
    for case in [
        "baseline",
        "declared_operator_point",
        "rank_action_cancellation",
    ] {
        let mut input = inputs(&sm);
        let diagonal = if case == "declared_operator_point" {
            "1.0000000000000000000000000000000000000001"
        } else {
            "1"
        };
        let mut component = OperatorComponent {
            label: "diagonal".into(),
            source_digest: ContentDigest::sha256(case.as_bytes()),
            diagonal: vec![diagonal.into(), "0".into(), diagonal.into()],
            dense: vec![],
            rank_one: vec![],
        };
        if case == "rank_action_cancellation" {
            component.diagonal.clear();
            let big = (Float::with_val(128, 1) << 400u32).to_string();
            let ranks = [0usize, 2]
                .into_iter()
                .flat_map(|j| {
                    let mut vector = vec!["0".to_string(); 3];
                    vector[j] = "1".into();
                    [big.clone(), "1".into(), format!("-{big}")]
                        .into_iter()
                        .map(move |weight| json!({"weight":weight,"vector":vector}))
                })
                .collect::<Vec<_>>();
            component.rank_one = serde_json::from_value(json!(ranks)).unwrap();
        }
        input.perturbations.push(component);
        let report = capture_extended(
            "directional_response",
            &state,
            Some(&matrix),
            Some(&roots),
            Some(&input),
            &ExtensionOptions::for_source(&state),
            &[],
            &context(),
        )
        .unwrap();
        assert_eq!(report.value.data.rows[1].outcome, "carrier_or_unresolved");
        let values = &report.value.data.rows[0].values;
        assert!((num(&values["conditional_tau_response_diagonal"]) - 0.24).abs() < 1e-14);
        measurements.push(values.clone());
        if case == "baseline" {
            let mut o = ExtensionOptions::for_source(&state);
            o.maximum_working_bytes = Some(1);
            let report = capture_extended(
                "directional_response",
                &state,
                Some(&matrix),
                Some(&roots),
                Some(&input),
                &o,
                &[],
                &context(),
            )
            .unwrap();
            assert_eq!(report.value.data.outcome, "unresolved");
            o.maximum_working_bytes = None;
            o.maximum_directional_rows = 0;
            let report = capture_extended(
                "directional_response",
                &state,
                Some(&matrix),
                Some(&roots),
                Some(&input),
                &o,
                &[],
                &context(),
            )
            .unwrap();
            assert!(report
                .value
                .data
                .rows
                .iter()
                .all(|r| r.outcome == "budget_limited"));
        }
        if case == "rank_action_cancellation" {
            let big = (Float::with_val(128, 1) << 10000u32).to_string();
            for (j, rank) in input.perturbations[0].rank_one.iter_mut().enumerate() {
                if j % 3 == 0 {
                    rank.weight = big.clone();
                } else if j % 3 == 2 {
                    rank.weight = format!("-{big}");
                }
            }
            let report = capture_extended(
                "directional_response",
                &state,
                Some(&matrix),
                Some(&roots),
                Some(&input),
                &ExtensionOptions::for_source(&state),
                &[],
                &context(),
            )
            .unwrap();
            assert_eq!(report.value.data.rows[0].outcome, "cancellation_limited");
            assert!(report.value.data.rows[0].values.is_empty());
        }
    }
    for name in ["forcing_diagonal", "conditional_tau_response_diagonal"] {
        for values in &measurements[1..] {
            let a = hp(&values[name]);
            let b = hp(&measurements[0][name]);
            assert!((a - b).abs() < (Float::with_val(512, 1) >> 175));
        }
    }
}
#[test]
fn directional_curvature_is_scalar_shift_invariant() {
    let mut kappas = Vec::new();
    for case in ["small_scalar_shift", "large_scalar_shift"] {
        let p = 192;
        let big = (Float::with_val(128, 1) << 400u32)
            .to_integer()
            .unwrap()
            .to_string();
        let eigen = if case == "small_scalar_shift" {
            "3".to_string()
        } else {
            big
        };
        let mut entries = vec!["0".to_string(); 25];
        for j in 0..5 {
            entries[5 * j + j] = eigen.clone();
        }
        for (i, j, v) in [(0, 1, "1"), (1, 2, "-1"), (2, 4, "1"), (4, 0, "-1")] {
            entries[5 * i + j] = v.into();
            entries[5 * j + i] = v.into();
        }
        let (mm, mb) = source(
            "ccm_tau_matrix",
            json!({"schema_version":2,"lambda_squared":"9","n_modes":2,"precision_bits":128,"entries":entries}),
        );
        let (mut sm, sb) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":3,"lambda_squared":"9","n_modes":2,"precision_bits":128,"force_even":true,"eigenvalue":eigen,"eigenvector":["1","1","1","1","1"]}),
        );
        sm.dependencies.push(DependencyRef {
            key: mm.key.clone(),
            content_digest: mm.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        let state = RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
            .unwrap();
        let matrix =
            RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest))
                .unwrap();
        let (mut sec, secbytes) = source(
            "ccm_secular_source",
            json!({"schema_version":1,"lambda_squared":"9","n_modes":2,"precision_bits":128,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
        );
        sec.dependencies.push(DependencyRef {
            key: sm.key.clone(),
            content_digest: sm.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        let tau = (Float::with_val(p, 15) - Float::with_val(p, 145).sqrt()) / 10u32;
        let t = Float::with_val(p, Constant::Pi) * 2u32 * tau.sqrt() / Float::with_val(p, 9).ln();
        let (mut rm, rb) = source(
            "ccm_root_discovery_window",
            json!({"schema_version":5,"lambda_squared":"9","n_modes":2,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":t.to_string()}}]}),
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
            &secbytes,
            &state,
            &[rm.content_digest.clone(), sec.content_digest.clone()],
        )
        .unwrap();
        let report = capture_extended(
            "directional_response",
            &state,
            Some(&matrix),
            Some(&roots),
            None,
            &ExtensionOptions::for_source(&state),
            &[],
            &context(),
        )
        .unwrap();
        let v = &report.value.data.rows[0].values;
        let k = hp(&v["directional_energy"]);
        assert!(k > 1.67 && k < 1.69);
        kappas.push(k);
        assert_ne!(report.value.data.rows[0].outcome, "unresolved_denominator");
    }
    assert!(Float::with_val(512, &kappas[0] - &kappas[1]).abs() < (Float::with_val(512, 1) >> 175));
}
#[test]
fn retained_adaptive_roots_preserve_storage_points_and_low_accuracy_targets() {
    let mut reports = Vec::new();
    for (case, p, metadata) in [
        ("plain128", 128, false),
        ("adaptive128", 128, true),
        ("plain64", 64, false),
        ("adaptive64", 64, true),
        ("large_work_metadata", 128, true),
    ] {
        let (sm, sb) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":p,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
        );
        let state = RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
            .unwrap();
        let (mut sec, secbytes) = source(
            "ccm_secular_source",
            json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":p,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
        );
        sec.dependencies.push(DependencyRef {
            key: sm.key.clone(),
            content_digest: sm.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        let mut details = json!({"value":"1.0000000000000000000000000000000000000001"});
        if metadata {
            details["adaptive_precision"] = json!({"source_accuracy_scope":"exact_stored_point_source","target_precision_bits":if p==64{1}else{64},"evaluation_precision_bits":p+128,"verification_precision_bits":p+192,"precision_escalations":2,"verification_correction":"0","stopping_reason":"requested_target_confirmed"});
        }
        if case == "large_work_metadata" {
            details["adaptive_precision"]["evaluation_precision_bits"] = json!(1_000_064);
            details["adaptive_precision"]["verification_precision_bits"] = json!(1_000_128);
        }
        let (mut rm, rb) = source(
            "ccm_root_discovery_window",
            json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":p,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":details}]}),
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
            &secbytes,
            &state,
            &[rm.content_digest.clone(), sec.content_digest.clone()],
        )
        .unwrap();
        let report = capture_root_window(&roots, &context()).unwrap().value.data;
        assert_eq!(
            TransformOptions::for_roots(&state, &roots).working_precision_bits,
            p + 64
        );
        assert!(report
            .window_inverse_moments
            .iter()
            .all(|value| hp(value) == 1));
        reports.push(report.window_inverse_moments);
    }
    assert_eq!(reports[0], reports[1]);
    assert_eq!(reports[2], reports[3]);
    assert_eq!(reports[0], reports[4]);
}

fn hp(s: &str) -> Float {
    Float::with_val(512, Float::parse(s).unwrap())
}
fn num(s: &str) -> f64 {
    hp(s).to_f64()
}
fn within(map: &BTreeMap<String, String>, name: &str, expected: &str, p: u32) {
    let reference = hp(expected);
    let lo = hp(&map[&format!("{name}_lower")]);
    let hi = hp(&map[&format!("{name}_upper")]);
    assert!(
        lo <= reference && reference <= hi,
        "{name},p={p}: [{lo},{hi}] excludes {reference}"
    );
    let tolerance = (Float::with_val(512, 1) + reference.clone().abs()) >> (p - 8);
    assert!(
        (hp(&map[name]) - reference).abs() <= tolerance,
        "midpoint {name},p={p}"
    );
}
#[test]
fn directional_matches_independent_projected_vector_reference() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/directional_oracle.json")).unwrap();
    let mut compared = 0;
    let mut withheld = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let n = case["N"].as_u64().unwrap() as usize;
        for p in [128, 192, 256] {
            for (sign, shift) in [(1i32, 0i32), (1, 400), (-1, -400)] {
                let co = case["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|a| {
                        let mut a =
                            Float::with_val(128, Float::parse(a.as_str().unwrap()).unwrap());
                        a *= sign;
                        if shift >= 0 {
                            a <<= shift as u32;
                        } else {
                            a >>= shift.unsigned_abs();
                        }
                        a.to_string()
                    })
                    .collect::<Vec<_>>();
                let (mm, mb) = source(
                    "ccm_tau_matrix",
                    json!({"schema_version":2,"lambda_squared":"9","n_modes":n,"precision_bits":128,"entries":case["matrix"]}),
                );
                let (mut sm, sb) = source(
                    "ccm_weil_eigenpair",
                    json!({"schema_version":3,"lambda_squared":"9","n_modes":n,"precision_bits":128,"force_even":true,"eigenvalue":case["eigenvalue"],"eigenvector":co}),
                );
                sm.dependencies.push(DependencyRef {
                    key: mm.key.clone(),
                    content_digest: mm.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                });
                let s =
                    RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
                        .unwrap();
                let m = RetainedMatrix::from_payload(
                    &mm,
                    &mb,
                    std::slice::from_ref(&mm.content_digest),
                )
                .unwrap();
                let (mut sec, secbytes) = source(
                    "ccm_secular_source",
                    json!({"schema_version":1,"lambda_squared":"9","n_modes":n,"precision_bits":128,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
                );
                sec.dependencies.push(DependencyRef {
                    key: sm.key.clone(),
                    content_digest: sm.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                });
                let (mut rm, rb) = source(
                    "ccm_root_discovery_window",
                    json!({"schema_version":5,"lambda_squared":"9","n_modes":n,"precision_bits":128,"force_even":true,"first_root_index":7,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":case["root"]}},{"status":"failed","details":{"reason":"retained failure"}}]}),
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
                    &secbytes,
                    &s,
                    &[rm.content_digest.clone(), sec.content_digest.clone()],
                )
                .unwrap();
                let mut op = case["operator"].clone();
                op["label"] = json!("operator");
                op["source_digest"] = json!(ContentDigest::sha256(b"operator"));
                let input:ExternalResearchInputs=serde_json::from_value(json!({"schema_version":1,"source_eigenpair":sm.content_digest,"lambda_squared":"9","n_modes":n,"precision_bits":128,"convention_id":"finite oracle","definition_digest":ContentDigest::sha256(b"independent oracle"),"approximation_scope":"stored point diagnostic","perturbations":[op],"run_once":{"derivative_actions":[{"label":"compact","source_digest":ContentDigest::sha256(b"compact"),"action":case["compact"],"convention":"action on center-oriented unit coefficient state"}]}})).unwrap();
                let o = ExtensionOptions {
                    working_precision_bits: p,
                    ..ExtensionOptions::for_source(&s)
                };
                let report = capture_extended(
                    "directional_response",
                    &s,
                    Some(&m),
                    Some(&roots),
                    Some(&input),
                    &o,
                    &[],
                    &context(),
                )
                .unwrap()
                .value
                .data;
                assert_eq!(report.rows[0].ordinal, 7);
                assert_eq!(report.rows[1].ordinal, 8);
                assert_eq!(report.rows[1].outcome, "missing_input");
                for (name, value) in case["reference"].as_object().unwrap() {
                    let values = &report.rows[0].values;
                    if name.starts_with("conditional_tau_response_") && !values.contains_key(name) {
                        // Generic manufactured matrices need not satisfy the displacement
                        // identity. Keep all unconditional oracle comparisons; do not
                        // relabel their single-direction stationary step as root velocity.
                        assert!(
                            hp(&values["displacement_defect_lower"])
                                > hp(&values["displacement_defect_tolerance_upper"])
                        );
                        assert_eq!(
                            report.rows[0].outcome,
                            "channels_resolved_budget_unassessed"
                        );
                        withheld += 1;
                    } else {
                        within(values, name, value.as_str().unwrap(), p);
                        compared += 1;
                    }
                }
            }
        }
    }
    assert_eq!(compared + withheld, 1962);
    assert!(compared > 1800 && withheld > 0);
}
