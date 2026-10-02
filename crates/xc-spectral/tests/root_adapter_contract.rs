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
#[cfg(feature = "arb")]
fn state(c: &str, co: &[String], p: u32) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":p,"force_even":false,"eigenvalue":"3","eigenvector":co}),
    );
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
    )
}
fn roots(
    s: &RetainedState,
    m: &ArtifactManifest,
    c: &str,
    n: usize,
    source_p: u32,
    root_p: u32,
    points: &[Option<String>],
) -> RetainedRoots {
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":c,"n_modes":n,"precision_bits":source_p,"force_even":false,"eigenpair_content_digest":m.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
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
        json!({"schema_version":5,"lambda_squared":c,"n_modes":n,"precision_bits":root_p,"force_even":false,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":outcomes}),
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

use rug::Integer;
use xc_numerics::prefix::lossless_decimal as dec;
fn number(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn fixture(
    c: &str,
    co: &[String],
    matrix: &[String],
    p: u32,
) -> (RetainedState, ArtifactManifest, RetainedMatrix<'static>) {
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":p,"entries":matrix}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":p,"force_even":false,"eigenvalue":"3","eigenvector":co}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    (
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap(),
        sm,
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap(),
    )
}
fn inputs(m: &ArtifactManifest, c: &str, n: usize, p: u32) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":c,"n_modes":n,"precision_bits":p,"convention_id":"independent transport and enclosure oracle","definition_digest":ContentDigest::sha256(b"adapter oracle"),"approximation_scope":"finite synthetic model"})).unwrap()
}
fn captured(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    r: Option<&RetainedRoots>,
    i: Option<&ExternalResearchInputs>,
    p: u32,
) -> ExtendedAnalysis {
    let mut o = ExtensionOptions::for_source(s);
    o.working_precision_bits = p;
    capture_extended(id, s, m, r, i, &o, &[], &context())
        .unwrap()
        .value
        .data
}
fn enclosed(v: &BTreeMap<String, String>, name: &str, expected: &Float, p: u32) {
    let lo = number(&v[&format!("{name}_lower")], p + 512);
    let hi = number(&v[&format!("{name}_upper")], p + 512);
    assert!(
        lo <= *expected && hi >= *expected,
        "{name}: [{lo}, {hi}] excludes {expected}"
    );
    let width = hi - lo;
    let scale = expected.clone().abs().max(&Float::with_val(p + 512, 1));
    assert!(
        width <= scale * (Float::with_val(p + 512, 1) >> (p - 24)),
        "{name}: interval width {width}"
    );
}

#[test]
fn transport_enclosures_match_independent_projected_resolvent_formulas() {
    let all: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/transport_oracle.json")).unwrap();
    let mut reports = 0;
    let mut comparisons = 0;
    for case in all["cases"].as_array().unwrap() {
        let q = case["precision"].as_u64().unwrap() as u32;
        let p = q + 64;
        let c = case["cutoff"].as_str().unwrap();
        for (shift, sign) in [(-5000i32, 1), (0, -1), (5000, 1)] {
            let co = case["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| dec(&((number(v.as_str().unwrap(), q) << shift) * sign)))
                .collect::<Vec<_>>();
            let matrix = case["matrix"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().into())
                .collect::<Vec<_>>();
            let (s, sm, m) = fixture(c, &co, &matrix, q);
            let t = Float::with_val(
                q,
                case["root"]["mantissa"]
                    .as_str()
                    .unwrap()
                    .parse::<Integer>()
                    .unwrap(),
            ) << case["root"]["exponent"].as_i64().unwrap() as i32;
            let roots = roots(&s, &sm, c, 1, q, q, &[Some(dec(&t))]);
            let mut i = inputs(&sm, c, 1, q);
            i.run_once=Some(serde_json::from_value(json!({"derivative_actions":[
                {"label":"tau_total","source_digest":ContentDigest::sha256(b"total"),"action":["1","0","1"],"convention":"declared unit-state derivative action"},
                {"label":"left","source_digest":ContentDigest::sha256(b"left"),"action":["1","0","0"],"convention":"declared derivative component"},
                {"label":"right","source_digest":ContentDigest::sha256(b"right"),"action":["0","0","1"],"convention":"declared derivative component"}],"log_cutoff_velocity":case["log_cutoff_velocity"]})).unwrap());
            let out = captured("root_transport", &s, Some(&m), Some(&roots), Some(&i), p);
            assert_eq!(
                out.rows[0].outcome, "point_measurement",
                "{:?}",
                out.rows[0].notes
            );
            for (name, expected) in case["expected"].as_object().unwrap() {
                enclosed(
                    &out.rows[0].values,
                    name,
                    &number(expected.as_str().unwrap(), p + 512),
                    p,
                );
                comparisons += 1;
            }
            reports += 1;
        }
    }
    assert_eq!(reports, 327);
    assert_eq!(comparisons, 3270);
}

#[test]
fn transport_velocity_and_retained_response_checks_preserve_owning_precision() {
    let co = vec!["1".into(), "3".into(), "1".into()];
    let matrix = vec![
        "4".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "4".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "4".into(),
    ];
    let (s, sm, m) = fixture("9", &co, &matrix, 128);
    let root = roots(
        &s,
        &sm,
        "9",
        1,
        128,
        128,
        &[Some("1".into()), None, Some("0".into())],
    );
    let mut results = Vec::new();
    for value in ["1", "1.0000000000000000000000000000000000000001"] {
        let mut i = inputs(&sm, "9", 1, 128);
        i.run_once=Some(serde_json::from_value(json!({"log_cutoff_velocity":value,"completion":{"response_checks":[{"ordinal":1,"t":value,"source_digest":ContentDigest::sha256(b"shifted fixture"),"branch":"production_shifted_secular","coordinate":"mellin_t","derivative_parameter":"u=log(lambda_squared)","activation_convention":"analytic_right_continuous_active_prime_set; tau=pole-archimedean-prime; total roots include d(2*pi*n/u)/du=-2*pi*n/u^2","fixed_velocity":"2","support_velocity":value,"total_velocity":"3"}]}})).unwrap());
        results.push(captured(
            "root_transport",
            &s,
            Some(&m),
            Some(&root),
            Some(&i),
            192,
        ));
    }
    assert_eq!(
        serde_json::to_value(&results[0]).unwrap(),
        serde_json::to_value(&results[1]).unwrap()
    );
    assert_eq!(
        number(
            &results[0].rows[0].values["retained_transport_additivity_defect"],
            192
        ),
        0
    );
    assert_eq!(results[0].rows.len(), 3);
    assert_eq!(results[0].rows[1].outcome, "missing_input");
    assert_eq!(results[0].rows[2].outcome, "carrier_or_unresolved");
}

#[test]
fn physical_transport_velocity_aliases_give_identical_enclosures() {
    let co = vec!["1".into(), "3".into(), "1".into()];
    let matrix = vec![
        "4".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "4".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "4".into(),
    ];
    let (s, sm, m) = fixture("9", &co, &matrix, 128);
    let t = Float::with_val(
        128,
        Float::with_val(1024, Constant::Pi) * 2u32 * (Float::with_val(1024, 3) / 5u32).sqrt()
            / Float::with_val(1024, 9).ln(),
    );
    let root = roots(&s, &sm, "9", 1, 128, 128, &[Some(dec(&t))]);
    let mut rows = Vec::new();
    for value in ["1", "1.0000000000000000000000000000000000000001"] {
        let mut i = inputs(&sm, "9", 1, 128);
        i.run_once=Some(serde_json::from_value(json!({"derivative_actions":[{"label":"tau_total","source_digest":ContentDigest::sha256(b"total"),"action":["1","0","1"],"convention":"declared derivative"}],"log_cutoff_velocity":value})).unwrap());
        rows.push(
            captured("root_transport", &s, Some(&m), Some(&root), Some(&i), 192)
                .rows
                .remove(0),
        );
    }
    assert_eq!(
        serde_json::to_value(&rows[0]).unwrap(),
        serde_json::to_value(&rows[1]).unwrap()
    );
    assert!(rows[0]
        .values
        .contains_key("conditional_total_physical_velocity_lower"));
}

#[test]
#[cfg(feature = "arb")]
fn certified_retained_root_rows_enclose_the_original_point_and_keep_aliases_equal() {
    let (s, sm) = state("9", &["1".into()], 128);
    let mut results = Vec::new();
    for t in ["1", "1.0000000000000000000000000000000000000001"] {
        let roots = roots(&s, &sm, "9", 0, 128, 128, &[Some(t.into()), None]);
        results.push(captured(
            "transform_enclosure",
            &s,
            None,
            Some(&roots),
            None,
            192,
        ));
    }
    assert_eq!(
        serde_json::to_value(&results[0]).unwrap(),
        serde_json::to_value(&results[1]).unwrap()
    );
    let row = results[0]
        .rows
        .iter()
        .find(|r| r.label == "retained_root_enclosure")
        .unwrap();
    let half = Float::with_val(1024, 9).ln() / 2u32;
    let expected = half.clone().sin() / &half;
    enclosed(&row.values, "normalized_real", &expected, 192);
    assert_eq!(number(&row.values["t"], 192), 1);
    assert_eq!(number(&row.values["input_point_precision_bits"], 192), 128);
    assert_eq!(results[0].rows.last().unwrap().outcome, "missing_input");
}

#[test]
#[cfg(feature = "arb")]
fn contour_points_follow_external_precision_and_counts_match_sinc_zeros() {
    use xc_spectral::ccm::{
        convergence_capture::RunOnceInputs,
        research_completion::{CompletionInputs, ContourPolicy},
    };
    let mut reports = 0;
    for q in [64, 192] {
        for (c, right, count) in [
            ("2", "3", 0),
            ("2", "10", 1),
            ("2", "28", 3),
            ("9", "1", 0),
            ("9", "3", 1),
            ("9", "7", 2),
            ("9", "12", 4),
            ("100", "1", 0),
            ("100", "3", 2),
            ("100", "7", 5),
        ] {
            for (shift, sign) in [(-500_000_000i32, 1), (0, -1), (500_000_000, 1)] {
                let (s, sm) = state(c, &[dec(&((Float::with_val(q, 1) << shift) * sign))], q);
                let mut i = inputs(&sm, c, 0, q);
                i.run_once = Some(RunOnceInputs {
                    completion: Some(CompletionInputs {
                        contour: Some(ContourPolicy {
                            left: "-1".into(),
                            right: right.into(),
                            bottom: "-0.5".into(),
                            top: "0.5".into(),
                            maximum_depth: 20,
                            maximum_segments: 2048,
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                });
                let out = captured("transform_enclosure", &s, None, None, Some(&i), q + 64);
                assert_eq!(
                    out.outcome, "certified_finite_enclosure",
                    "{:?}",
                    out.reason
                );
                assert_eq!(
                    number(&out.values["certified_finite_zero_count"], q + 64),
                    count
                );
                reports += 1;
            }
        }
    }
    assert_eq!(reports, 60);
    let (s, sm) = state("9", &["1".into()], 128);
    let mut rows = Vec::new();
    for right in ["7", "7.0000000000000000000000000000000000000001"] {
        let mut i = inputs(&sm, "9", 0, 128);
        i.run_once=Some(serde_json::from_value(json!({"completion":{"contour":{"left":"-1","right":right,"bottom":"-0.5","top":"0.5","maximum_depth":20,"maximum_segments":2048}}})).unwrap());
        rows.push(captured(
            "transform_enclosure",
            &s,
            None,
            None,
            Some(&i),
            192,
        ));
    }
    assert_eq!(
        serde_json::to_value(&rows[0]).unwrap(),
        serde_json::to_value(&rows[1]).unwrap()
    );
}

#[test]
#[cfg(feature = "arb")]
fn contour_boundary_zero_never_receives_a_zero_count_certificate() {
    let (s, sm) = state("9", &["1".into(), "0".into(), "-1".into()], 128);
    let mut i = inputs(&sm, "9", 1, 128);
    i.run_once=Some(serde_json::from_value(json!({"completion":{"contour":{"left":"0","right":"1","bottom":"-0.5","top":"0.5","maximum_depth":4,"maximum_segments":32}}})).unwrap());
    let out = captured("transform_enclosure", &s, None, None, Some(&i), 192);
    assert_eq!(out.outcome, "partial_unresolved");
    assert!(!out.values.contains_key("certified_finite_zero_count"));
}

#[test]
#[cfg(feature = "arb")]
fn ordinary_serialized_root_points_have_correct_function_and_derivative_enclosures() {
    for q in [64, 128, 256] {
        let (s, sm) = state("9", &["1".into()], q);
        let p = q + 64;
        let half = Float::with_val(2048, 9).ln() / 2u32;
        let points = vec![
            Float::with_val(q, 0),
            Float::with_val(q, 2).sqrt(),
            number("1.234567890123456789", q),
            Float::with_val(q, -2),
            Float::with_val(q, Float::with_val(2048, Constant::Pi) / &half),
        ];
        let roots = roots(
            &s,
            &sm,
            "9",
            0,
            q,
            q,
            &points.iter().map(|t| Some(dec(t))).collect::<Vec<_>>(),
        );
        let mut i = inputs(&sm, "9", 0, q);
        i.run_once=Some(serde_json::from_value(json!({"completion":{"contour":{"left":"-0.25","right":"0.25","bottom":"-0.25","top":"0.25","maximum_depth":8,"maximum_segments":256}}})).unwrap());
        let result = captured("transform_enclosure", &s, None, Some(&roots), Some(&i), p);
        for (row, t) in result
            .rows
            .iter()
            .filter(|r| r.label == "retained_root_enclosure")
            .zip(points)
        {
            let t = Float::with_val(2048, t);
            let z = Float::with_val(2048, &half) * &t;
            let (value, derivative) = if z == 0 {
                (Float::with_val(2048, 1), Float::with_val(2048, 0))
            } else {
                (
                    z.clone().sin() / &z,
                    ((Float::with_val(2048, &z) * z.clone().cos() - z.clone().sin())
                        / z.clone().square())
                        * &half,
                )
            };
            enclosed(&row.values, "t", &t, p);
            enclosed(&row.values, "normalized_real", &value, p);
            let scale = (Float::with_val(2048, &half) * 2u32).sqrt();
            enclosed(&row.values, "value_real", &(value * &scale), p);
            enclosed(&row.values, "derivative_real", &(derivative * scale), p);
        }
    }
}
