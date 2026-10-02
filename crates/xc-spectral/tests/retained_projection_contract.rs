#![cfg(feature = "hp")]
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{retained_evidence::*, state_geometry::RetainedState};
fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(kind, "synthetic-geometry-fixture", kind.as_bytes()).unwrap(),
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
    };
    (manifest, bytes)
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
fn state(coefficients: &[&str], eigenvalue: &str) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"force_even":true,"eigenvalue":eigenvalue,"eigenvector":coefficients}),
    );
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
    )
}
fn reference(coefficients: &[&str]) -> ReferenceSpec {
    ReferenceSpec {
        schema_version: 1,
        definition: "synthetic Fourier test function".into(),
        lambda_squared: "9".into(),
        precision_bits: 128,
        coefficients: coefficients.iter().map(|s| s.to_string()).collect(),
        approximation_scope: "exact finite test function; serialized coefficients are points"
            .into(),
    }
}
fn retained_reference(spec: &ReferenceSpec) -> RetainedReference {
    let value = capture_reference(spec, &context()).unwrap().value;
    let (m, b) = source("ccm_reference_source", serde_json::to_value(value).unwrap());
    RetainedReference::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap()
}
use rug::Float;
fn float(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn shift(s: &str, e: i32, sign: i32) -> String {
    let mut x = float(s, 128) * sign;
    if e >= 0 {
        x <<= e as u32;
    } else {
        x >>= e.unsigned_abs();
    }
    x.to_string()
}
fn close(actual: &str, expected: &str, p: u32) {
    let a = float(actual, p + 64);
    let e = float(expected, p + 64);
    let tolerance = (Float::with_val(p + 64, 1) + e.clone().abs()) >> (p - 24);
    assert!(
        (Float::with_val(p + 64, &a) - &e).abs() <= tolerance,
        "p={p}: {a} versus {e}"
    );
}
fn unscale(actual: &str, e: i32) -> String {
    let mut x = float(actual, 512);
    if e >= 0 {
        x >>= e as u32;
    } else {
        x <<= e.unsigned_abs();
    }
    x.to_string()
}
fn source_at(coefficients: &[String], cutoff: &str) -> RetainedState {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":cutoff,"n_modes":(coefficients.len()-1)/2,"precision_bits":128,"force_even":false,"eigenvalue":"1","eigenvector":coefficients}),
    );
    RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap()
}
fn reference_at(coefficients: &[String], cutoff: &str) -> RetainedReference {
    let mut spec = reference(&["1"]);
    spec.coefficients = coefficients.to_vec();
    spec.lambda_squared = cutoff.into();
    retained_reference(&spec)
}
#[test]
fn projection_matches_rational_linear_algebra_and_defining_fourier_integrals() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/retained_projection_oracle.json")).unwrap();
    let mut reports = 0;
    for row in fixture["rows"].as_array().unwrap() {
        let cutoff = row["cutoff"].as_str().unwrap();
        let k = row["basis"].as_array().unwrap().len();
        for p in [128, 192, 256] {
            for (se, re, sign, columns_scaled) in [
                (0i32, 0i32, 1i32, false),
                (600_000_000, 0, -1, false),
                (-600_000_000, 0, 1, false),
                (0, 600_000_000, -1, false),
                (-600_000_000, 600_000_000, -1, true),
            ] {
                let values = |v: &serde_json::Value, e: i32, s: i32| {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| shift(x.as_str().unwrap(), e, s))
                        .collect::<Vec<_>>()
                };
                let state = source_at(&values(&row["source"], se, sign), cutoff);
                let reference = reference_at(&values(&row["reference"], re, -sign), cutoff);
                let exponents = (0..k)
                    .map(|j| {
                        if columns_scaled {
                            if j % 2 == 0 {
                                200
                            } else {
                                -200
                            }
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<_>>();
                let basis = row["basis"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(&exponents)
                    .map(|(v, e)| reference_at(&values(v, *e, 1), cutoff))
                    .collect::<Vec<_>>();
                let o = ProjectionOptions {
                    working_precision_bits: p,
                    normalization: row["normalization"].as_str().unwrap().into(),
                    fixed_second_component: if k == 2 { Some("-0.75".into()) } else { None },
                };
                let r = capture_projection(&state, &reference, &basis, &o, &context())
                    .unwrap()
                    .value
                    .data;
                assert_eq!(r.outcome, "point_measurement");
                assert!(r.pivot_metric.contains("independent_binary_column_scaling"));
                close(&r.signed_unit_overlap, row["overlap"].as_str().unwrap(), p);
                close(
                    r.difference_norm_squared.as_ref().unwrap(),
                    row["difference_norm_squared"].as_str().unwrap(),
                    p,
                );
                close(
                    r.fit_residual_norm_squared.as_ref().unwrap(),
                    row["fit_residual_norm_squared"].as_str().unwrap(),
                    p,
                );
                let coefficients = r.coefficients.as_ref().unwrap();
                for i in 0..k {
                    close(
                        &unscale(&coefficients[i], -exponents[i]),
                        row["coefficients"][i].as_str().unwrap(),
                        p,
                    );
                    close(
                        &unscale(&r.rhs[i], exponents[i]),
                        row["rhs"][i].as_str().unwrap(),
                        p,
                    );
                    for j in 0..k {
                        close(
                            &unscale(&r.gram[i * k + j], exponents[i] + exponents[j]),
                            row["gram"][i * k + j].as_str().unwrap(),
                            p,
                        );
                    }
                }
                assert!(float(r.minimum_pivot.as_ref().unwrap(), p) > 0);
                if k == 2 {
                    let a0 = float(
                        &unscale(row["coefficients"][0].as_str().unwrap(), exponents[0]),
                        512,
                    );
                    let a1 = float(
                        &unscale(row["coefficients"][1].as_str().unwrap(), exponents[1]),
                        512,
                    );
                    close(r.b2.as_ref().unwrap(), &(a1 + a0 * 0.75f64).to_string(), p);
                    if let Some(expected) = row["b_effective"].as_str() {
                        close(
                            &unscale(r.b_effective.as_ref().unwrap(), exponents[0] - exponents[1]),
                            expected,
                            p,
                        );
                    }
                }
                reports += 1;
            }
        }
    }
    assert_eq!(reports, 600);
}
#[test]
fn projection_exact_centers_domains_and_exponent_failures_are_explicit() {
    let reference = reference_at(&["0", "1", "0"].map(String::from), "9");
    let mut o = ProjectionOptions {
        working_precision_bits: 192,
        normalization: "unit_l2_dx".into(),
        fixed_second_component: None,
    };
    for coefficient in ["1e200000000", "1e-200000000"] {
        let s = source_at(&["0".into(), coefficient.into(), "0".into()], "9");
        let r = capture_projection(&s, &reference, &[], &o, &context())
            .unwrap()
            .value
            .data;
        close(&r.signed_unit_overlap, "1", 192);
        assert_eq!(float(r.difference_norm_squared.as_ref().unwrap(), 192), 0);
    }
    o.normalization = "center_one".into();
    let s = source_at(
        &["1e100", "-1", "-2e100", "-1", "1e100"].map(String::from),
        "9",
    );
    let r = capture_projection(&s, &reference, &[], &o, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.outcome, "point_measurement");
    assert_eq!(float(&r.source_center, 192), 2);
    let s = source_at(
        &["1e-200000000", "1", "1e-200000000"].map(String::from),
        "9",
    );
    assert!(capture_projection(&s, &reference, &[], &o, &context()).is_err());
    let s = source_at(&["0", "1", "0"].map(String::from), "9");
    let wrong = reference_at(
        &["0", "1", "0"].map(String::from),
        "9.000000000000000000000000000000000000000000000000000000000000000000000001",
    );
    assert!(capture_projection(&s, &wrong, &[], &o, &context()).is_err());
    let equivalent = reference_at(&["0", "1", "0"].map(String::from), "9.000");
    assert!(capture_projection(&s, &equivalent, &[], &o, &context()).is_ok());
    let r = capture_projection(
        &s,
        &reference,
        &[reference.clone(), reference.clone()],
        &o,
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "rank_or_precision_unresolved");
    assert!(r.coefficients.is_none());
}
fn roots_at(values: &[Option<&str>]) -> RetainedRoots {
    let (s, sm) = state(&["0", "1", "0"], "1");
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"normalization":"sum_xi_equals_sqrt_log_lambda_squared","eigenpair_content_digest":sm.content_digest.0}),
    );
    let dep = |m: &ArtifactManifest| DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    };
    sec.dependencies.push(dep(&sm));
    let outcomes = values
        .iter()
        .map(|v| match v {
            Some(v) => json!({"status":"converged","details":{"value":v}}),
            None => json!({"status":"failed","details":{"reason":"missing"}}),
        })
        .collect::<Vec<_>>();
    let (mut rm, rb) = source(
        "ccm_root_refinement",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"reference_seeded_audit","reference_seeds_used":true,"completeness":"partial","outcomes":outcomes}),
    );
    rm.dependencies.push(dep(&sec));
    let allowed = [
        sm.content_digest.clone(),
        sec.content_digest.clone(),
        rm.content_digest.clone(),
    ];
    RetainedRoots::from_payload(&rm, &rb, &sec, &sb, &s, &allowed).unwrap()
}
#[test]
fn finite_root_moments_match_rational_sums_and_fail_on_unrepresentable_terms() {
    let roots = roots_at(&[
        Some("1"),
        Some("2"),
        Some("3"),
        Some("4"),
        Some("0"),
        Some("-1"),
        None,
    ]);
    let r = capture_root_window(&roots, &context()).unwrap().value.data;
    assert_eq!(
        (r.positive_count, r.nonpositive_count, r.missing_count),
        (4, 2, 1)
    );
    for (actual, (n, d)) in
        r.window_inverse_moments
            .iter()
            .zip([(25u32, 12u32), (205, 144), (2035, 1728)])
    {
        close(actual, &(Float::with_val(256, n) / d).to_string(), 128);
    }
    assert_eq!(float(r.minimum_positive_spacing.as_ref().unwrap(), 128), 1);
    for exponent in [-200_000_000i32, 200_000_000] {
        let x = shift("1", exponent, 1);
        let r = capture_root_window(&roots_at(&[Some(&x)]), &context())
            .unwrap()
            .value
            .data;
        for (i, v) in r.window_inverse_moments.iter().enumerate() {
            assert_eq!(float(&unscale(v, -exponent * (i as i32 + 1)), 128), 1);
        }
    }
    assert!(capture_root_window(&roots_at(&[Some("1e200000000")]), &context()).is_err());
    let r = capture_root_window(&roots_at(&[None, Some("0"), Some("-2")]), &context())
        .unwrap()
        .value
        .data;
    assert!(r.window_inverse_moments.iter().all(|x| float(x, 128) == 0));
    let r = capture_root_window(&roots_at(&[Some("2"), Some("2")]), &context())
        .unwrap()
        .value
        .data;
    assert_eq!(float(r.minimum_positive_spacing.as_ref().unwrap(), 128), 0);
}
#[test]
fn stabilization_uses_source_points_and_scaled_relative_changes() {
    let o = StabilizationOptions {
        working_precision_bits: 192,
        relative_tolerance: "1e-45".into(),
        consecutive_steps: 1,
    };
    let (a, _) = state(&["1"], "1");
    let (b, _) = state(
        &["0", "1", "0"],
        "1.0000000000000000000000000000000000000001",
    );
    let r = capture_stabilization(&[a, b], &o, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.outcome, "finite_rule_met");
    assert_eq!(float(r.rows[1].relative_change.as_ref().unwrap(), 192), 0);
    let huge = shift("1", 1_073_741_822, 1);
    let negative = shift("1", 1_073_741_822, -1);
    let (a, _) = state(&["1"], &huge);
    let (b, _) = state(&["0", "1", "0"], &negative);
    let r = capture_stabilization(&[a, b], &o, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.outcome, "finite_rule_not_met");
    assert_eq!(float(r.rows[1].relative_change.as_ref().unwrap(), 192), 2);
    let (a, _) = state(&["1"], "0");
    let (b, _) = state(&["0", "1", "0"], "0");
    let r = capture_stabilization(&[a, b], &o, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.rows[1].status, "zero_denominator");
    assert!(r.rows[1].relative_change.is_none());
}
#[test]
fn retained_statistics_match_independent_fraction_cohorts_under_scaling() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/retained_statistics_oracle.json")).unwrap();
    for row in fixture["roots"].as_array().unwrap() {
        for exponent in [0i32, 200_000_000, -200_000_000] {
            let values = row["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().map(|s| shift(s, exponent, 1)))
                .collect::<Vec<_>>();
            let inputs = values.iter().map(|x| x.as_deref()).collect::<Vec<_>>();
            let r = capture_root_window(&roots_at(&inputs), &context())
                .unwrap()
                .value
                .data;
            assert_eq!(
                r.positive_count as u64,
                row["positive_count"].as_u64().unwrap()
            );
            assert_eq!(
                r.nonpositive_count as u64,
                row["nonpositive_count"].as_u64().unwrap()
            );
            assert_eq!(
                r.missing_count as u64,
                row["missing_count"].as_u64().unwrap()
            );
            for (i, moment) in r.window_inverse_moments.iter().enumerate() {
                close(
                    &unscale(moment, -exponent * (i as i32 + 1)),
                    row["moments"][i].as_str().unwrap(),
                    128,
                );
            }
            if let Some(spacing) = row["spacing"].as_str() {
                close(
                    &unscale(r.minimum_positive_spacing.as_ref().unwrap(), exponent),
                    spacing,
                    128,
                );
            } else {
                assert!(r.minimum_positive_spacing.is_none());
            }
        }
    }
    let o = StabilizationOptions {
        working_precision_bits: 192,
        relative_tolerance: "0.03125".into(),
        consecutive_steps: 2,
    };
    for row in fixture["cohorts"].as_array().unwrap() {
        for exponent in [0i32, 600_000_000, -600_000_000] {
            let states = row["values"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let mut coefficients = vec!["0"; 2 * i + 1];
                    coefficients[i] = "1";
                    state(&coefficients, &shift(v.as_str().unwrap(), exponent, 1)).0
                })
                .collect::<Vec<_>>();
            let r = capture_stabilization(&states, &o, &context())
                .unwrap()
                .value
                .data;
            assert_eq!(r.outcome, row["outcome"].as_str().unwrap());
            for (i, actual) in r.rows.iter().enumerate() {
                assert_eq!(actual.status, row["statuses"][i].as_str().unwrap());
                if let Some(change) = row["relative_changes"][i].as_str() {
                    close(actual.relative_change.as_ref().unwrap(), change, 192);
                } else {
                    assert!(actual.relative_change.is_none());
                }
            }
        }
    }
}

#[test]
fn exhaustive_resumed_stabilization_does_not_round_an_excess_change_into_tolerance() {
    use rug::Rational;
    let exact = Rational::from((1, 7));
    let nearest = Float::with_val(128, &exact);
    assert!(nearest.to_rational().unwrap() < exact);
    // A terminating decimal spelling of this dyadic lies strictly below 1/7.
    let tolerance = nearest.to_string_radix(10, Some(256));
    assert_eq!(
        float(&tolerance, 1024).to_rational().unwrap(),
        nearest.to_rational().unwrap()
    );
    let (a, _) = state(&["1"], "7");
    let (b, _) = state(&["0", "1", "0"], "8");
    let options = StabilizationOptions {
        working_precision_bits: 128,
        relative_tolerance: tolerance,
        consecutive_steps: 1,
    };
    let report = capture_stabilization(&[a, b], &options, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(
        report.outcome, "finite_rule_not_met",
        "exact relative change 1/7 exceeds the exact supplied tolerance"
    );
    assert_eq!(report.rows[1].status, "outside_rule");
}
