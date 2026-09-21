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
fn number(s: &str) -> f64 {
    Float::with_val(256, Float::parse(s).unwrap()).to_f64()
}
fn near(actual: &str, expected: f64, tol: f64) {
    assert!(
        (number(actual) - expected).abs() < tol,
        "{actual} versus {expected}"
    );
}
#[test]
fn compactness_origin_jet_matches_constant_and_cosine_integrals() {
    let (s, _, _) = fixture();
    let r = run("compactness", &s, None, None);
    let l = 9f64.ln();
    near(&r.values["transform_origin"], l.sqrt(), 1e-14);
    near(
        &r.values["transform_second_derivative"],
        -l.powf(2.5) / 12.,
        1e-13,
    );
    near(
        &r.values["transform_fourth_derivative"],
        l.powf(4.5) / 80.,
        1e-13,
    );
    near(&r.values["sigma"], l * l / 24., 1e-14);
    for row in &r.rows {
        let a = number(&row.values["rate"]);
        let exact = ((a * l).exp() - 1.) / (a * l);
        near(&row.values["analytic_integral"], exact, 1e-12);
    }
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["-1","0","-1"]}),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let r = run("compactness", &s, None, None);
    assert!(!r.values.contains_key("sigma"));
    near(
        &r.values["transform_second_derivative"],
        l.powf(2.5) / (2f64.sqrt() * std::f64::consts::PI.powi(2)),
        1e-13,
    );
}
#[test]
fn weighted_nonorthogonal_projection_retains_raw_b2_convention() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    let p = 192;
    let half = Float::with_val(p, 9).ln() / 2u32;
    let n = 32usize;
    let mut values = vec![];
    let mut basis = vec![vec![], vec![]];
    for j in 0..=n {
        let x: Float = Float::with_val(p, &half) * j / n;
        values.push((Float::with_val(p, 1) - Float::with_val(p, &x) / 10u32).to_string());
        basis[0].push((Float::with_val(p, 1) + &x).to_string());
        basis[1].push((Float::with_val(p, 1) - &x).to_string());
    }
    i.target = Some(SampledReference {
        definition_digest: ContentDigest::sha256(b"linear numerical fixture"),
        evaluation_policy: "explicit uniform x points".into(),
        approximation_scope: "point samples".into(),
        intervals: n,
        values,
        basis_values: basis,
        fixed_second_component: Some("0.5".into()),
        raw_normalizer: "1".into(),
        trial_coefficients: None,
    });
    let r = run("weighted_reference_projection", &s, None, Some(&i));
    near(&r.values["a_0"], 0.05, 1e-14);
    near(&r.values["a_1"], -0.05, 1e-14);
    near(&r.values["b2"], -0.075, 1e-14);
    near(&r.values["b_effective"], -1., 1e-13);
    assert!(number(&r.values["fit_residual_norm_squared"]) < 1e-70);
    let t = i.target.as_mut().unwrap();
    t.basis_values[1] = t.basis_values[0].clone();
    assert_eq!(
        run("weighted_reference_projection", &s, None, Some(&i)).outcome,
        "rank_or_precision_unresolved"
    );
}
#[test]
fn signed_reference_channels_keep_cancellation_and_closure_separate() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    let l = Float::with_val(192, 9).ln();
    i.reference_jets.push(ReferenceJet {
        matched_root_ordinal: None,
        ordinal: 200,
        t: "0".into(),
        reference_window: Jet {
            value: (Float::with_val(192, &l) - Float::with_val(192, Float::parse("0.1").unwrap()))
                .to_string(),
            derivative: "0".into(),
        },
        reference_full: Jet {
            value: (Float::with_val(192, &l) + Float::with_val(192, Float::parse("0.1").unwrap()))
                .to_string(),
            derivative: "0".into(),
        },
        exterior_tail: Jet {
            value: "0.2".into(),
            derivative: "0".into(),
        },
        endpoint_tail_part: Some(Jet {
            value: "0.03".into(),
            derivative: "0".into(),
        }),
        fitted_interior_parts: vec![Jet {
            value: "0.08".into(),
            derivative: "0".into(),
        }],
        error_normalization: None,
        source_value_error: None,
        tail_value_error: None,
        source_derivative_error: None,
        root_separation_radius: None,
    });
    let r = run("signed_transform", &s, None, Some(&i));
    let v = &r.rows[0].values;
    assert_eq!(r.rows[0].ordinal, 200);
    near(&v["value_signed_interior"], 0.1, 1e-14);
    near(&v["value_signed_total"], -0.1, 1e-14);
    near(&v["value_unfitted_interior"], 0.02, 1e-14);
    near(&v["value_remaining_tail"], 0.17, 1e-14);
    assert!(number(&v["value_reference_closure_defect"]).abs() < 1e-50);
}
#[test]
fn arithmetic_split_closes_and_preserves_signed_energy_ratios() {
    let (s, m, matrix) = fixture();
    let mut i = inputs(&m);
    for (label, value) in [("positive", "5"), ("negative", "-2")] {
        i.components.push(OperatorComponent {
            label: label.into(),
            source_digest: ContentDigest::sha256(label.as_bytes()),
            diagonal: vec![value.into(); 3],
            dense: vec![],
            rank_one: vec![],
        });
    }
    i.components_are_complete = true;
    i.deficit = Some("0.5".into());
    i.deficit_kind = Some("fuchs_approximation".into());
    let r = run("arithmetic_energy", &s, Some(&matrix), Some(&i));
    near(&r.values["sum_component_energy"], 3., 1e-14);
    near(&r.values["sum_absolute_component_energy"], 7., 1e-14);
    near(&r.values["energy_closure_defect"], 0., 1e-14);
    near(&r.values["signed_weil_over_deficit"], 6., 1e-14);
    i.components[0].diagonal[1] = "6".into();
    near(
        &run("arithmetic_energy", &s, Some(&matrix), Some(&i)).values["energy_closure_defect"],
        1.,
        1e-14,
    );
}
#[test]
fn weighted_tail_keeps_lattice_sign_and_declared_coverage() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.atom_coordinate = Some("test positive coordinate".into());
    i.atom_coverage = Some("finite supplied table, no omitted-tail bound".into());
    i.tail_checkpoints = vec!["1".into(), "3".into()];
    for (ordinal, x, w, family) in [
        (1, "1", "2", "zero"),
        (2, "2", "4", "zero"),
        (1, "3", "-1", "lattice"),
    ] {
        i.atoms.push(WeightedAtom {
            ordinal,
            coordinate: x.into(),
            weight: w.into(),
            family: family.into(),
            partition: "declared_band".into(),
        });
    }
    let r = run("weighted_tail", &s, None, Some(&i));
    let z = r
        .rows
        .iter()
        .find(|r| {
            r.label == "zero/declared_band"
                && r.values["cutoff"]
                    == "3.0000000000000000000000000000000000000000000000000000000000"
        })
        .or_else(|| {
            r.rows
                .iter()
                .rev()
                .find(|r| r.label == "zero/declared_band")
        })
        .unwrap();
    near(&z.values["included_mass"], 6., 1e-14);
    near(&z.values["weighted_inverse_moment_1"], 4., 1e-14);
    let l = r
        .rows
        .iter()
        .rev()
        .find(|r| r.label == "lattice/declared_band")
        .unwrap();
    near(&l.values["included_mass"], -1., 1e-14);
}
#[test]
fn cluster_projection_handles_nonorthogonal_vectors_and_cross_n_embedding() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.cluster = vec![
        ClusterVector {
            source_digest: ContentDigest::sha256(b"v1"),
            n_modes: 1,
            precision_bits: 128,
            eigenvalue: "3".into(),
            coefficients: vec!["0".into(), "2".into(), "0".into()],
            assembly_policy: "fixture".into(),
        },
        ClusterVector {
            source_digest: ContentDigest::sha256(b"v2"),
            n_modes: 1,
            precision_bits: 128,
            eigenvalue: "4".into(),
            coefficients: vec!["1".into(), "1".into(), "1".into()],
            assembly_policy: "fixture".into(),
        },
    ];
    i.previous_cluster = vec![ClusterVector {
        n_modes: 2,
        coefficients: vec!["0".into(), "0".into(), "1".into(), "0".into(), "0".into()],
        ..i.cluster[0].clone()
    }];
    let r = run("spectral_cluster", &s, None, Some(&i));
    assert!(number(&r.values["source_cluster_leakage_squared"]) < 1e-80);
    near(&r.rows[0].values["largest_previous_overlap"], 1., 1e-14);
}
#[test]
fn energy_allowance_never_divides_by_invalid_margin() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.energy_allowance = Some(EnergyAllowance {
        upper_trial_energy: "2".into(),
        low_block_lower_bound: "1".into(),
        high_block_lower_bound: "6".into(),
        cross_block_norm_bound: "3".into(),
        hypothesis_record_digest: ContentDigest::sha256(b"conditional fixture"),
        hypotheses: vec!["supplied point bounds, not certified".into()],
    });
    let r = run("energy_allowance", &s, None, Some(&i));
    near(&r.values["conditional_energy_allowance"], 2.25, 1e-14);
    near(&r.values["conditional_vector_allowance"], 0.75, 1e-14);
    near(
        &r.values["allowance_to_trial_energy_magnitude"],
        1.125,
        1e-14,
    );
    assert_eq!(
        number(&r.values["allowance_below_trial_energy_magnitude"]),
        0.
    );
    assert!(r.reason.as_ref().unwrap().contains("non-informative"));
    i.energy_allowance.as_mut().unwrap().high_block_lower_bound = "2".into();
    let r = run("energy_allowance", &s, None, Some(&i));
    assert_eq!(r.outcome, "sufficient_bound_unavailable");
    assert!(!r.values.contains_key("conditional_energy_allowance"));
    assert!(!r.values.contains_key("allowance_to_trial_energy_magnitude"));
}

#[test]
fn energy_allowance_scale_comparison_is_explicit_and_handles_zero_and_negative_trials() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    for (trial, cross, expected_ratio, below) in [
        ("1e-59", "2", Some(1e59), false),
        ("2", "1", Some(0.125), true),
        ("-2", "1", Some(0.125), true),
        ("2", "0", Some(0.), true),
        ("0", "1", None, false),
        ("0", "0", None, false),
    ] {
        let high: Float = Float::with_val(256, Float::parse(trial).unwrap()) + 4;
        i.energy_allowance = Some(EnergyAllowance {
            upper_trial_energy: trial.into(),
            low_block_lower_bound: "-3".into(),
            high_block_lower_bound: high.to_string_radix(10, None),
            cross_block_norm_bound: cross.into(),
            hypothesis_record_digest: ContentDigest::sha256(b"scale fixture"),
            hypotheses: vec!["finite point fixture".into()],
        });
        let r = run("energy_allowance", &s, None, Some(&i));
        assert_eq!(r.outcome, "conditional_bound_expression");
        assert!(r.convention.contains("not a certificate"));
        if let Some(expected) = expected_ratio {
            near(
                &r.values["allowance_to_trial_energy_magnitude"],
                expected,
                expected.abs() * 1e-14 + 1e-14,
            );
            assert_eq!(
                number(&r.values["allowance_below_trial_energy_magnitude"]),
                f64::from(u32::from(below))
            );
            assert_eq!(
                r.reason.as_ref().unwrap().contains("non-informative"),
                !below
            );
        } else {
            assert!(!r.values.contains_key("allowance_to_trial_energy_magnitude"));
            assert!(!r
                .values
                .contains_key("allowance_below_trial_energy_magnitude"));
            assert!(r.reason.as_ref().unwrap().contains("trial energy is zero"));
        }
    }
}
#[test]
fn missing_inputs_foreign_sources_and_resource_limits_are_explicit() {
    let (s, m, _) = fixture();
    for id in [
        "weighted_reference_projection",
        "signed_transform",
        "weighted_tail",
        "spectral_cluster",
        "energy_allowance",
    ] {
        let r = run(id, &s, None, None);
        assert_eq!(r.outcome, "missing_input");
        assert!(r.reason.is_some());
    }
    let mut i = inputs(&m);
    i.source_eigenpair = ContentDigest::sha256(b"wrong state");
    assert!(capture_extended(
        "weighted_tail",
        &s,
        None,
        None,
        Some(&i),
        &ExtensionOptions::for_source(&s),
        &[],
        &context()
    )
    .is_err());
    let mut o = ExtensionOptions::for_source(&s);
    o.working_precision_bits = 64;
    assert!(capture_extended("compactness", &s, None, None, None, &o, &[], &context()).is_err());
}
#[test]
fn extended_analysis_is_deterministic_across_worker_counts() {
    let (s, _, _) = fixture();
    let run_pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| serde_json::to_value(run("compactness", &s, None, None)).unwrap())
    };
    assert_eq!(run_pool(1), run_pool(4));
}
#[test]
fn resolution_refuses_to_invent_source_error_or_an_n_threshold() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    let t = (Float::with_val(192, Constant::Pi) * 2u32 / Float::with_val(192, 9).ln()).to_string();
    i.reference_jets.push(ReferenceJet {
        matched_root_ordinal: None,
        ordinal: 1,
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
        error_normalization: None,
        source_value_error: None,
        tail_value_error: None,
        source_derivative_error: None,
        root_separation_radius: None,
    });
    let r = run("resolution_budget", &s, None, Some(&i));
    assert_ne!(r.rows[0].outcome, "conditional_budget_met");
    near(&r.values["conditional_contiguous_prefix"], 0., 1e-14);
    assert!(!r.rows[0]
        .values
        .contains_key("spacing_normalized_displacement"));
    let mut next = i.reference_jets[0].clone();
    next.ordinal = 2;
    next.t = (Float::with_val(192, Float::parse(&next.t).unwrap()) + 2u32).to_string();
    i.reference_jets.push(next);
    let r = run("resolution_budget", &s, None, Some(&i));
    near(&r.rows[0].values["reference_neighbor_spacing"], 2., 1e-12);
    assert!(r.rows[0]
        .values
        .contains_key("spacing_normalized_newton_correction"));
    assert!(!r.rows[0]
        .values
        .contains_key("spacing_normalized_displacement"));
    i.reference_jets[0].matched_root_ordinal = Some(1);
    let r = run("resolution_budget", &s, None, Some(&i));
    assert!(r.rows[0]
        .notes
        .iter()
        .any(|s| s.contains("join unavailable")));
}

#[test]
fn directional_energy_matches_independent_projected_resolvent_calculation() {
    let p = 192;
    let co = ["1", "3", "1"];
    let norm = 11f64;
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
    let mut i = inputs(&sm);
    i.perturbations.push(OperatorComponent {
        label: "diagonal".into(),
        source_digest: ContentDigest::sha256(b"D"),
        diagonal: vec!["1".into(), "0".into(), "1".into()],
        dense: vec![],
        rank_one: vec![],
    });
    let o = ExtensionOptions::for_source(&state);
    let result = capture_extended(
        "directional_response",
        &state,
        Some(&matrix),
        Some(&roots),
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    let v = [1f64 / norm.sqrt(), 3f64 / norm.sqrt(), 1f64 / norm.sqrt()];
    let rv = [v[0] / (-0.4), v[1] / 0.6, v[2] / (-0.4)];
    let dot = v.iter().zip(rv).map(|(a, b)| a * b).sum::<f64>();
    let x = [rv[0] - dot * v[0], rv[1] - dot * v[1], rv[2] - dot * v[2]];
    let k = x.iter().map(|x| x * x).sum::<f64>();
    near(&result.rows[0].values["directional_energy"], k, 1e-12);
    near(
        &result.rows[0].values["conditional_tau_response_diagonal"],
        -(x[0] * v[0] + x[2] * v[2]) / k,
        1e-13,
    );
    assert_eq!(result.rows[1].outcome, "carrier_or_unresolved");
    let action = vec![v[0].to_string(), "0".into(), v[2].to_string()];
    i.run_once=Some(serde_json::from_value(json!({"derivative_actions":[
        {"label":"tau_total","source_digest":ContentDigest::sha256(b"total fixture"),"action":action,"convention":"dTau/dlogC synthetic diagonal"},
        {"label":"tau_fixture","source_digest":ContentDigest::sha256(b"component fixture"),"action":action,"convention":"single complete synthetic component"}],"log_cutoff_velocity":"1"})).unwrap());
    let limited = ExtensionOptions {
        maximum_directional_rows: 1,
        ..o.clone()
    };
    let transported = capture_extended(
        "root_transport",
        &state,
        Some(&matrix),
        Some(&roots),
        Some(&i),
        &limited,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(transported.rows.len(), 2);
    assert_eq!(transported.rows[1].outcome, "carrier_or_unresolved"); // Not omitted by the legacy one-row work budget.
    let t = t.to_f64();
    let tau_velocity = -(x[0] * v[0] + x[2] * v[2]) / k;
    near(
        &transported.rows[0].values["conditional_total_physical_velocity"],
        2.0 * std::f64::consts::PI.powi(2) * tau_velocity / (t * 9f64.ln().powi(2)) - t / 9f64.ln(),
        1e-12,
    );
    near(
        &transported.rows[0].values["forcing_closure_defect"],
        0.0,
        1e-14,
    );
    let o = ExtensionOptions {
        maximum_directional_rows: 1,
        ..o
    };
    let result = capture_extended(
        "directional_response",
        &state,
        Some(&matrix),
        Some(&roots),
        None,
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.rows[1].outcome, "budget_limited");
}
#[test]
fn qualified_missing_record_never_becomes_a_completed_measurement() {
    let (s, _, _) = fixture();
    let result = capture_extended(
        "weighted_reference_projection",
        &s,
        None,
        None,
        None,
        &ExtensionOptions::for_source(&s),
        &[],
        &context(),
    )
    .unwrap();
    let d = CapturedDiagnostic::new(&result.value, vec![]).unwrap();
    assert!(matches!(d.qualified(), Err(CaptureFailure::Missing { .. })));
}

#[test]
fn complex_transform_matches_the_entire_constant_integral() {
    let (s, _, _) = fixture();
    let r = run("complex_transform", &s, None, None);
    let l = 9f64.ln();
    assert_eq!(r.rows.len(), 70);
    near(&r.rows[2].values["value_re"], l.sqrt(), 1e-14);
    near(
        &r.rows[4].values["value_re"],
        2.0 * (l / 2.0).sinh() / l.sqrt(),
        1e-14,
    );
    near(&r.rows[4].values["value_im"], 0.0, 1e-14);
    near(
        &r.rows[4].values["derivative_im"],
        -2.0 * ((l / 2.0) * (l / 2.0).cosh() - (l / 2.0).sinh()) / l.sqrt(),
        1e-14,
    );
    assert!(r.convention.contains("not contour certificates"));
    assert_eq!(
        r.rows
            .iter()
            .filter(|r| r.label == "contour_sample")
            .count(),
        65
    );
    assert_eq!(r.rows[5].values, r.rows[69].values);
}
#[test]
fn every_parent_prefix_and_independent_comparison_remain_distinct() {
    let (s, sm, m) = fixture();
    let r = run("finite_section_transfer", &s, Some(&m), None);
    assert_eq!(r.rows.len(), 2);
    for row in &r.rows {
        near(&row.values["retained_mass"], 1.0, 1e-14);
        near(&row.values["high_forcing_squared"], 0.0, 1e-14);
        near(&row.values["truncated_energy"], 3.0, 1e-14);
    }
    let mut i = inputs(&sm);
    i.run_once=Some(serde_json::from_value(json!({"comparison":{"source_digest":ContentDigest::sha256(b"comparison-state"),"matrix_digest":ContentDigest::sha256(b"comparison-matrix"),"lambda_squared":"9","n_modes":0,"precision_bits":128,"coefficients":["1"],"matrix":["2"],"eigenvalue":"2","assembly_policy":"independent synthetic comparison"}})).unwrap());
    let r = run("finite_section_transfer", &s, Some(&m), Some(&i));
    near(
        &r.values["comparison_block_frobenius_difference"],
        1.0,
        1e-14,
    );
    near(
        &r.values["comparison_state_signed_block_defect"],
        1.0,
        1e-14,
    );
}
#[test]
fn cluster_feedback_matches_a_two_by_two_schur_complement() {
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["4","0","1","0","3","0","1","0","6"]}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let m =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let r = run("operator_cluster", &s, Some(&m), None);
    assert_eq!(r.outcome, "point_measurement");
    near(&r.rows[3].values["compressed_operator"], 6.0, 1e-14);
    near(&r.rows[3].values["signed_complement_feedback"], 1.0, 1e-14);
    near(&r.rows[3].values["effective_operator"], 5.0, 1e-14);
}
#[test]
fn fixed_tail_model_solves_without_borrowing_retained_energy() {
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    i.run_once=Some(serde_json::from_value(json!({"tail_form":{"definition_digest":ContentDigest::sha256(b"fixed synthetic basis"),"dimension":2,"finite_zero_form":["1","0","0","3"],"tail_correction":["0.5","0.25","0.25","1"],"lattice_gram":["2","0","0","1"],"tail_operator_error":null,"coverage":"synthetic finite atoms; no infinite tail","hypotheses":[]}})).unwrap());
    let r = run("tail_operator", &s, None, Some(&i));
    near(
        &r.values["model_energy"],
        4.75 - (3.25f64 * 3.25 + 0.125).sqrt(),
        1e-12,
    );
    near(&r.values["model_energy_without_tail"], 1.0, 1e-12);
    near(&r.values["lattice_gram_pivot_ratio"], 2.0, 1e-12);
    near(&r.values["model_vector_lattice_norm_squared"], 1.0, 1e-12);
    assert!(number(&r.values["model_vector_relative_residual"]) < 1e-20);
    near(
        &r.values["tail_energy_lift"],
        number(&r.values["model_energy"]) - 1.0,
        1e-12,
    );
    assert!(
        (number(&r.values["model_vector_zero_energy"])
            + number(&r.values["model_vector_tail_energy"])
            - number(&r.values["model_energy"]))
        .abs()
            < 1e-12
    );
    let (mut other, bytes) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"999","eigenvector":["0","1","0"]}),
    );
    other.dependencies = sm.dependencies.clone();
    let otherstate =
        RetainedState::from_payload(&other, &bytes, std::slice::from_ref(&other.content_digest))
            .unwrap();
    i.source_eigenpair = other.content_digest;
    let b = run("tail_operator", &otherstate, None, Some(&i));
    assert_eq!(r.values["model_energy"], b.values["model_energy"]);
    let mut scaled = i.clone();
    let form = scaled
        .run_once
        .as_mut()
        .unwrap()
        .tail_form
        .as_mut()
        .unwrap();
    for x in form
        .finite_zero_form
        .iter_mut()
        .chain(&mut form.tail_correction)
    {
        *x = (Float::with_val(192, Float::parse(x.as_str()).unwrap())
            * Float::with_val(192, Float::parse("1e-100").unwrap()))
        .to_string();
    }
    let tiny = run("tail_operator", &otherstate, None, Some(&scaled));
    // Scaling the pencil must never turn a poor relative vector into a qualified one.
    let residual = number(&tiny.values["model_vector_relative_residual"]);
    assert!(residual < 1e-25 || tiny.outcome == "partial_unresolved");
    assert!(tiny.values.contains_key("model_energy_without_tail"));

    i.run_once
        .as_mut()
        .unwrap()
        .tail_form
        .as_mut()
        .unwrap()
        .lattice_gram[0] = "0".into();
    assert_eq!(
        run("tail_operator", &otherstate, None, Some(&i)).outcome,
        "unresolved"
    );
}
#[test]
fn missing_error_source_does_not_become_a_certificate() {
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    let a = run("observable_budget", &s, None, None);
    assert!(!a.values.contains_key("conditional_origin_lower_margin"));
    i.run_once=Some(serde_json::from_value(json!({"uncertainty":{"unit_state_l2_error":"0.01","source_certificate_digest":ContentDigest::sha256(b"synthetic conditional source"),"hypotheses":["fixture allowance only"]}})).unwrap());
    let b = run("observable_budget", &s, None, Some(&i));
    near(
        &b.values["conditional_origin_lower_margin"],
        0.99 * 9f64.ln().sqrt(),
        1e-14,
    );
}

#[test]
fn cluster_workspace_limit_keeps_operator_and_coupling() {
    let (s, _, m) = fixture();
    let mut options = ExtensionOptions::for_source(&s);
    options.maximum_working_bytes = Some(1);
    let r = capture_extended(
        "operator_cluster",
        &s,
        Some(&m),
        None,
        None,
        &options,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "partial_unresolved");
    assert!(r.reason.unwrap().contains("workspace budget"));
    assert!(r
        .rows
        .iter()
        .all(|r| r.values.contains_key("coupling_gram")
            && !r.values.contains_key("effective_operator")));
}

#[test]
fn signed_band_recurrence_matches_discrete_gaussian_nodes_and_keeps_failure() {
    use xc_spectral::ccm::research_completion::*;
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    let model = SignedBandModel {
        degree: 2,
        coordinate: "synthetic x".into(),
        definition_digest: ContentDigest::sha256(b"three equal atoms"),
        atoms: ["-1", "0", "1"]
            .iter()
            .map(|x| BandAtom {
                coordinate: (*x).into(),
                signed_weight: "1".into(),
                family: "synthetic".into(),
            })
            .collect(),
        coverage: "complete finite three-atom measure".into(),
        hypotheses: vec!["finite positive measure".into()],
        borrowed_inputs: vec![],
        input_energy: None,
        scoring_roots: vec![],
    };
    let once = xc_spectral::ccm::convergence_capture::RunOnceInputs {
        completion: Some(CompletionInputs {
            band: Some(model),
            ..Default::default()
        }),
        ..Default::default()
    };
    i.run_once = Some(once);
    let result = run("band_reconstruction", &s, None, Some(&i));
    assert_eq!(result.outcome, "point_measurement");
    let mut roots = result
        .rows
        .iter()
        .map(|r| number(&r.values["model_band_root"]))
        .collect::<Vec<_>>();
    roots.sort_by(f64::total_cmp);
    assert!((roots[0] + (2f64 / 3.).sqrt()).abs() < 1e-12);
    assert!((roots[1] - (2f64 / 3.).sqrt()).abs() < 1e-12);
    near(&result.values["band_inverse_moment_one"], 0.0, 1e-12);
    near(&result.values["band_inverse_moment_two"], 3.0, 1e-12);
    near(&result.values["band_inverse_moment_three"], 0.0, 1e-12);
    let m = i
        .run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .band
        .as_mut()
        .unwrap();
    m.atoms[1].signed_weight = "-3".into();
    assert_eq!(
        run("band_reconstruction", &s, None, Some(&i)).outcome,
        "unresolved"
    );
}
#[test]
fn independently_prepared_prime_action_exposes_an_error_hidden_by_reconstruction() {
    use xc_spectral::ccm::{convergence_capture::*, research_completion::*};
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    let action = |label: &str, value: &str| OperatorAction {
        label: label.into(),
        source_digest: ContentDigest::sha256(label.as_bytes()),
        action: vec!["0".into(), value.into(), "0".into()],
        convention: "same unit state".into(),
    };
    i.run_once = Some(RunOnceInputs {
        component_actions: vec![action("tau_prime_reconstructed", "2")],
        completion: Some(CompletionInputs {
            independent_actions: vec![action("tau_prime_direct", "3")],
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("consistency", &s, None, Some(&i));
    near(&r.rows[0].values["signed_energy_difference"], 1., 1e-14);
    near(&r.rows[0].values["action_difference_norm"], 1., 1e-14);
}
#[test]
fn source_independent_reference_preparation_generates_only_a_finite_target() {
    use xc_spectral::ccm::research_completion::*;
    let (s, _, _) = fixture();
    let reference = ReferenceSpec {
        schema_version: 1,
        definition: "finite constant".into(),
        lambda_squared: "9".into(),
        precision_bits: 128,
        coefficients: vec!["1".into()],
        approximation_scope: "finite compact support".into(),
    };
    let prep = ReferencePreparation {
        schema_version: 1,
        finite_reference: Some(ResearchInputs {
            schema_version: 1,
            reference,
            basis: vec![],
            projection: ProjectionOptions {
                working_precision_bits: 192,
                normalization: "center_one".into(),
                fixed_second_component: None,
            },
        }),
        sampled_reference: None,
        lambda_squared: "9".into(),
        precision_bits: 128,
        definition_digest: ContentDigest::sha256(b"external constant"),
        approximation_scope: "finite compact support".into(),
        weighted_atoms: vec![],
        atom_coordinate: None,
        atom_coverage: None,
        tail_form: None,
        tail_recipe: None,
        completion: None,
    };
    let input = prep.prepare(&s, None).unwrap();
    assert!(input
        .target
        .as_ref()
        .unwrap()
        .values
        .iter()
        .all(|x| number(x) == 1.));
    assert!(input.reference_jets.is_empty());
    let r = run("weighted_reference_projection", &s, None, Some(&input));
    near(&r.values["weighted_l2_squared"], 0., 1e-14);
}
#[test]
fn output_budget_exhaustion_is_retained_as_an_explicit_diagnostic_outcome() {
    let (s, _, _) = fixture();
    let mut options = ExtensionOptions::for_source(&s);
    options.maximum_estimated_output_bytes = 1;
    let r = capture_extended(
        "compactness",
        &s,
        None,
        None,
        None,
        &options,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.reason.unwrap().contains("resource budget"));
}
#[test]
#[cfg(feature = "arb")]
fn certified_contour_counts_constant_profile_roots_and_never_certifies_exhausted_boundary() {
    use xc_spectral::ccm::{convergence_capture::*, research_completion::*};
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    i.run_once = Some(RunOnceInputs {
        completion: Some(CompletionInputs {
            contour: Some(ContourPolicy {
                left: "-1".into(),
                right: "7".into(),
                bottom: "-0.5".into(),
                top: "0.5".into(),
                maximum_depth: 20,
                maximum_segments: 2048,
            }),
            ..Default::default()
        }),
        ..Default::default()
    });
    let result = run("transform_enclosure", &s, None, Some(&i));
    assert_eq!(
        result.outcome, "certified_finite_enclosure",
        "{:?}",
        result.reason
    );
    near(&result.values["certified_finite_zero_count"], 2., 1e-14);
    assert!(result
        .convention
        .contains("no eigenstate/assembly accuracy claim"));
    let c = i
        .run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .contour
        .as_mut()
        .unwrap();
    c.bottom = "0".into();
    c.maximum_depth = 0;
    c.maximum_segments = 4;
    let result = run("transform_enclosure", &s, None, Some(&i));
    assert_eq!(result.outcome, "partial_unresolved");
    assert!(!result.values.contains_key("certified_finite_zero_count"));
}

#[test]
fn independent_cohort_measures_forcing_and_withholds_unmatched_roots() {
    use xc_spectral::ccm::{convergence_capture::*, research_completion::*};
    let (s, sm, m) = fixture();
    let mut i = inputs(&sm);
    i.run_once = Some(RunOnceInputs {
        completion: Some(CompletionInputs {
            comparisons: vec![ComparisonSnapshot {
                state: ComparisonState {
                    source_digest: ContentDigest::sha256(b"small state"),
                    matrix_digest: ContentDigest::sha256(b"small matrix"),
                    lambda_squared: "9".into(),
                    n_modes: 0,
                    precision_bits: 128,
                    eigenvalue: "2".into(),
                    coefficients: vec!["1".into()],
                    matrix: vec!["2".into()],
                    assembly_policy: "synthetic".into(),
                },
                selection_policy: "even ground".into(),
                assembly_policy: "synthetic".into(),
                quadrature_policy: "exact".into(),
                root_branch: "different branch".into(),
                root_coordinate: "mellin_t".into(),
                roots: vec![],
            }],
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("configuration_comparison", &s, Some(&m), Some(&i));
    near(&r.rows[0].values["absolute_unit_overlap"], 1., 1e-14);
    near(
        &r.rows[0].values["independent_small_state_low_residual_squared"],
        1.,
        1e-14,
    );
    near(
        &r.rows[0].values["independent_small_state_high_forcing_squared"],
        0.,
        1e-14,
    );
    assert!(r.rows[0]
        .notes
        .iter()
        .any(|n| n.contains("ordinal differences withheld")));
}
#[test]
fn finite_atom_tail_builder_keeps_omitted_tail_qualification() {
    use xc_spectral::ccm::research_completion::*;
    let atoms: Vec<WeightedAtom> = serde_json::from_value(json!([
        {"ordinal":1,"coordinate":"-1","weight":"2","family":"zero","partition":"finite"},
        {"ordinal":2,"coordinate":"1","weight":"2","family":"zero","partition":"finite"},
        {"ordinal":1,"coordinate":"0","weight":"1","family":"lattice","partition":"finite"}
    ]))
    .unwrap();
    let form = prepare_tail_form(
        ContentDigest::sha256(b"finite test"),
        &[vec!["1".into()], vec!["0".into(), "1".into()]],
        &atoms,
        None,
        "three finite atoms",
        &[],
        128,
    )
    .unwrap();
    near(&form.finite_zero_form[0], 4., 1e-14);
    near(&form.finite_zero_form[3], 4., 1e-14);
    near(&form.lattice_gram[0], 1., 1e-14);
    assert!(form
        .hypotheses
        .iter()
        .any(|h| h.contains("unknown, not proved zero")));
    assert!(prepare_tail_form(
        ContentDigest::sha256(b"finite test"),
        &[vec!["1".into()], vec!["0".into(), "1".into()]],
        &atoms,
        Some(&["0".into(), "1".into(), "2".into(), "0".into()]),
        "three finite atoms",
        &[],
        128
    )
    .is_err());
}

#[test]
fn spacing_displacement_requires_an_explicit_join_and_ordered_reference() {
    let (state, sm, _) = fixture();
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: sm.key.clone(),
        content_digest: sm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":3,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":"12"}}]}),
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
        &state,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap();
    let mut i = inputs(&sm);
    for (ordinal, t) in [(10, "11"), (11, "13")] {
        i.reference_jets.push(ReferenceJet {
            matched_root_ordinal: if ordinal == 10 { Some(3) } else { None },
            ordinal,
            t: t.into(),
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
            error_normalization: None,
            source_value_error: None,
            tail_value_error: None,
            source_derivative_error: None,
            root_separation_radius: None,
        });
    }
    let evaluate = |i: &ExternalResearchInputs| {
        capture_extended(
            "resolution_budget",
            &state,
            None,
            Some(&roots),
            Some(i),
            &ExtensionOptions::for_source(&state),
            &[],
            &context(),
        )
        .unwrap()
        .value
        .data
    };
    let r = evaluate(&i);
    near(
        &r.rows[0].values["spacing_normalized_displacement"],
        0.5,
        1e-12,
    );
    i.reference_jets[0].matched_root_ordinal = None;
    assert!(!evaluate(&i).rows[0]
        .values
        .contains_key("matched_root_reference_displacement"));
    i.reference_jets[0].matched_root_ordinal = Some(3);
    i.reference_jets[1].t = "11".into();
    assert!(!evaluate(&i).rows[0]
        .values
        .contains_key("spacing_normalized_displacement"));
}

#[test]
fn signed_atom_sums_keep_exclusion_singularity_and_fixed_cutoff_ladders() {
    use xc_spectral::ccm::research_completion::*;
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    i.atom_coordinate = Some("synthetic x".into());
    i.atom_coverage = Some("finite fixture only".into());
    i.atoms = serde_json::from_value(serde_json::json!([
        {"ordinal":1,"coordinate":"1","weight":"2","family":"zero","partition":"band"},
        {"ordinal":2,"coordinate":"2","weight":"3","family":"zero","partition":"band"},
        {"ordinal":1,"coordinate":"3","weight":"1","family":"lattice","partition":"band"}
    ]))
    .unwrap();
    let policy=serde_json::from_value(serde_json::json!({"maximum_atoms":300000,"maximum_input_bytes":1000000,"evaluations":[
        {"ordinal":1,"coordinate":"3","label":"pole retained"},
        {"ordinal":2,"coordinate":"3","label":"self atom excluded","exclude":[{"family":"lattice","partition":"band","ordinal":1}]}
    ],"cutoffs":["1","2"],"tail_recipe":{"basis_polynomials":[["1"]],"tail_correction":null,"hypotheses":["finite fixture"]}})).unwrap();
    let band = SignedBandModel {
        degree: 1,
        coordinate: "synthetic x".into(),
        definition_digest: ContentDigest::sha256(b"finite band"),
        atoms: vec![
            BandAtom {
                coordinate: "1".into(),
                signed_weight: "2".into(),
                family: "zero".into(),
            },
            BandAtom {
                coordinate: "2".into(),
                signed_weight: "3".into(),
                family: "zero".into(),
            },
        ],
        coverage: "finite fixture".into(),
        hypotheses: vec!["finite positive measure".into()],
        borrowed_inputs: vec![],
        input_energy: None,
        scoring_roots: vec![],
    };
    i.run_once = Some(xc_spectral::ccm::convergence_capture::RunOnceInputs {
        completion: Some(CompletionInputs {
            atom_analysis: Some(policy),
            band: Some(band),
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("weighted_tail", &s, None, Some(&i));
    let zero = r
        .rows
        .iter()
        .find(|r| {
            r.label == "signed_atom_kernel"
                && r.ordinal == 2
                && r.notes.iter().any(|n| n.contains("family zero"))
        })
        .unwrap();
    assert_eq!(number(&zero.values["signed_kernel_1"]), -4.0);
    assert_eq!(number(&zero.values["signed_kernel_2"]), 3.5);
    assert_eq!(number(&zero.values["signed_kernel_3"]), -3.25);
    assert!(r.rows.iter().any(|r| r.label == "signed_atom_kernel"
        && r.ordinal == 1
        && r.outcome == "unresolved_denominator"
        && !r.values.contains_key("signed_kernel_1")));
    let band = run("band_reconstruction", &s, None, Some(&i));
    assert_eq!(
        number(&band.values["supplied_zero_extent_covers_model_band"]),
        1.0
    );
    let cuts = band
        .rows
        .iter()
        .filter(|r| r.label == "band_cutoff")
        .collect::<Vec<_>>();
    assert_eq!(cuts.len(), 2);
    assert_eq!(number(&cuts[0].values["first_model_band_root"]), 1.0);
    assert!((number(&cuts[1].values["first_model_band_root"]) - 1.6).abs() < 1e-14);
    let tail = run("tail_operator", &s, None, Some(&i));
    let cuts = tail
        .rows
        .iter()
        .filter(|r| r.label == "tail_model_cutoff")
        .collect::<Vec<_>>();
    assert_eq!(cuts.len(), 2);
    assert_eq!(number(&cuts[0].values["model_energy"]), 4.0);
    assert_eq!(number(&cuts[1].values["model_energy"]), 10.0);
}

#[test]
fn band_scoring_requires_a_complete_ordered_coordinate_match() {
    use xc_spectral::ccm::research_completion::*;
    let mut c: CompletionInputs = serde_json::from_value(serde_json::json!({"band": {
        "degree":2,"coordinate":"synthetic x","definition_digest":ContentDigest::sha256(b"audit scoring"),
        "atoms":[{"coordinate":"1","signed_weight":"1","family":"zero"},{"coordinate":"2","signed_weight":"1","family":"zero"},{"coordinate":"3","signed_weight":"1","family":"zero"}],
        "coverage":"finite fixture","hypotheses":["finite positive functional"],"borrowed_inputs":[],"input_energy":null,"scoring_roots":[]
    }})).unwrap();
    assert!(c.validate(3, 192).is_ok());
    for bad in [
        vec!["1"],
        vec!["2", "1"],
        vec!["1", "1"],
        vec!["1", "2", "3"],
    ] {
        c.band.as_mut().unwrap().scoring_roots = bad.into_iter().map(str::to_owned).collect();
        assert!(c.validate(3, 192).is_err());
    }
    c.band.as_mut().unwrap().scoring_roots = vec!["1".into(), "2".into()];
    assert!(c.validate(3, 192).is_ok());
}

#[test]
fn capability_dependent_diagnostics_do_not_reuse_other_or_unspecified_backends() {
    let root = std::env::temp_dir().join(format!(
        "ccm-feature-identity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (state, _, _) = fixture();
    let options = ExtensionOptions::for_source(&state);
    for id in ["transform_enclosure", "band_reconstruction"] {
        let initial =
            capture_extended(id, &state, None, None, None, &options, &[], &context()).unwrap();
        assert_eq!(
            initial.value.request["arb_available"],
            json!(cfg!(feature = "arb"))
        );
        let producer = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "feature-test",
                root.join(id),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let mut cache = ArtifactCacheContext {
            resolver: Some(&producer),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["feature-test".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        // Produce an authentic envelope in an isolated cache, then populate a
        // different cache with only the opposite-capability and legacy requests.
        let current =
            capture_extended(id, &state, None, None, None, &options, &[], &cache).unwrap();
        let manifest = current.produced_manifest.unwrap();
        let foreign = FilesystemCacheStore::new(
            "foreign",
            root.join(format!("{id}-foreign")),
            true,
            CacheVisibility::Local,
        );
        for capability in [Some(!cfg!(feature = "arb")), None] {
            let mut value = serde_json::to_value(&current.value).unwrap();
            match capability {
                Some(available) => {
                    value["request"]["arb_available"] = json!(available);
                }
                None => {
                    value["request"]
                        .as_object_mut()
                        .unwrap()
                        .remove("arb_available");
                }
            }
            let mut semantic: SemanticKeyEnvelope =
                serde_json::from_str(&manifest.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
            semantic.resolved_mathematical_parameters["request"] = value["request"].clone();
            let logical = format!(
                "ccm/research/{}/{}",
                manifest.key.kind,
                ContentDigest::sha256(&serde_json::to_vec(&semantic).unwrap()).0
            );
            let mut tags = manifest.tags.clone();
            tags.insert(
                SEMANTIC_KEY_MANIFEST_TAG.into(),
                serde_json::to_string(&semantic).unwrap(),
            );
            foreign
                .put(
                    &ArtifactDraft {
                        schema_version: manifest.schema_version,
                        key: ArtifactKey {
                            kind: manifest.key.kind.clone(),
                            logical_key: logical,
                            parameters_digest: semantic.digest().unwrap(),
                        },
                        producer_toolkit_version: manifest.producer_toolkit_version.clone(),
                        minimum_reader_version: manifest.minimum_reader_version.clone(),
                        maximum_reader_version: None,
                        quality: manifest.quality,
                        visibility: manifest.visibility,
                        immutable: true,
                        dependencies: manifest.dependencies.clone(),
                        tags,
                        provenance_digest: None,
                    },
                    &serde_json::to_vec(&value).unwrap(),
                )
                .unwrap();
        }
        let foreign_resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(foreign),
        }]);
        cache.resolver = Some(&foreign_resolver);
        cache.mode = ArtifactExecutionCacheMode::RequireReuse;
        cache.write_on_miss = false;
        assert!(capture_extended(id, &state, None, None, None, &options, &[], &cache).is_err());
        cache.resolver = Some(&producer);
        let warm = capture_extended(id, &state, None, None, None, &options, &[], &cache).unwrap();
        assert!(warm.reused_manifest.is_some());
        assert_eq!(
            serde_json::to_value(warm.value).unwrap(),
            serde_json::to_value(current.value).unwrap()
        );
    }
    // This is a unique test-owned child of the platform temporary directory.
    std::fs::remove_dir_all(root).unwrap();
}
