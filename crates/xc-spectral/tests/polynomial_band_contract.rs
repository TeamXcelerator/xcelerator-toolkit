#![cfg(all(feature = "hp", feature = "arb"))]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, state_geometry::RetainedState};
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

use rug::{ops::Pow, Integer, Rational};
fn point(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn decimal(s: &str) -> Rational {
    let mut parts = s.split(['e', 'E']);
    let mantissa = parts.next().unwrap();
    let exponent = parts.next().map_or(0, |v| v.parse::<i32>().unwrap());
    let fractional = mantissa.split_once('.').map_or(0, |(_, f)| f.len() as i32);
    let power = exponent - fractional;
    assert!(power.unsigned_abs() < 10000);
    let n: Integer = mantissa.replace('.', "").parse().unwrap();
    let ten = Integer::from(10).pow(power.unsigned_abs());
    if power >= 0 {
        Rational::from(n * ten)
    } else {
        Rational::from((n, ten))
    }
}
fn input(m: &ArtifactManifest, q: u32, matrix: Vec<String>, n: usize) -> ExternalResearchInputs {
    let gram = (0..n * n)
        .map(|j| if j / n == j % n { "1" } else { "0" })
        .collect::<Vec<_>>();
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":0,"precision_bits":q,"convention_id":"independent finite polynomial adapter contract","definition_digest":ContentDigest::sha256(b"independent polynomial adapter contract"),"approximation_scope":"finite rounded polynomial","run_once":{"tail_form":{"definition_digest":ContentDigest::sha256(b"independent polynomial adapter contract"),"dimension":n,"finite_zero_form":matrix,"tail_correction":vec!["0";n*n],"lattice_gram":gram,"coverage":"finite polynomial","hypotheses":["declared finite arithmetic model"],"polynomial_coordinate":"z"}}})).unwrap()
}
fn capture(
    s: &RetainedState,
    i: &ExternalResearchInputs,
    n: usize,
) -> (ExtendedAnalysis, Vec<Rational>) {
    let o = ExtensionOptions::for_source(s);
    let model = capture_extended("tail_operator", s, None, None, Some(i), &o, &[], &context())
        .unwrap()
        .value
        .data;
    let coefficients = (0..n)
        .map(|j| {
            point(
                model.values.get(&format!("model_vector_coefficient_{j}")).unwrap_or_else(||
                    panic!("tail model omitted coefficient {j}: outcome={}, reason={:?}, values={:?}",
                        model.outcome, model.reason, model.values)),
                o.working_precision_bits,
            )
            .to_rational()
            .unwrap()
        })
        .collect::<Vec<_>>();
    let record = capture_extended(
        "band_reconstruction",
        s,
        None,
        None,
        Some(i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value;
    assert_eq!(
        record.request["polynomial_band_arithmetic"],
        "stored_polynomial_exact_newton_inverse_moments_v1"
    );
    assert_eq!(
        record.request["polynomial_root_output"],
        "outward_root_bounds_and_safe_midpoints_v1"
    );
    (record.data, coefficients)
}
#[test]
fn exported_linear_root_intervals_contain_exact_actual_polynomial_roots() {
    let mut checks = 0;
    for q in [64, 128, 256] {
        let (s, m) = state("9", &["1".into()], q);
        for (a, b) in [(1i32, 0), (1, 1), (1, 3), (2, 5), (3, -2), (4, 7)] {
            let matrix = vec![a * a, a * b, a * b, b * b]
                .into_iter()
                .map(|v| v.to_string())
                .collect();
            let (report, co) = capture(&s, &input(&m, q, matrix, 2), 2);
            let root = -co[0].clone() / &co[1];
            assert_eq!(report.rows.len(), 1);
            let values = &report.rows[0].values;
            let lower = decimal(&values["rounded_polynomial_root_lower"]);
            let upper = decimal(&values["rounded_polynomial_root_upper"]);
            assert!(lower <= root && root <= upper);
            let midpoint = point(&values["model_band_root"], q + 64)
                .to_rational()
                .unwrap();
            assert!(lower <= midpoint && midpoint <= upper);
            for (j, name) in ["one", "two", "three"].iter().enumerate() {
                if root == 0 {
                    assert!(
                        !report
                            .values
                            .contains_key(&format!("band_inverse_moment_{name}")),
                        "inverse moments of an exact zero root must be unavailable"
                    );
                    checks += 1;
                    continue;
                }
                assert_eq!(
                    point(
                        &report.values[&format!("band_inverse_moment_{name}")],
                        q + 64
                    ),
                    Float::with_val(q + 64, root.clone().recip().pow(j as u32 + 1))
                );
                checks += 1;
            }
            checks += 2;
        }
    }
    assert_eq!(checks, 90);
}
fn evaluate(co: &[Rational], x: &Rational) -> Rational {
    co.iter()
        .rev()
        .fold(Rational::new(), |value, c| value * x + c)
}
#[test]
fn polynomial_moments_follow_exact_coefficients_through_ill_conditioned_roots() {
    let mut checks = 0;
    for q in [64, 128, 256] {
        let p = q + 64;
        let (s, m) = state("9", &["1".into()], q);
        let tiny = point(&format!("1e-{}", p * 3 / 4), q);
        let vector = [
            -tiny.clone(),
            -tiny,
            Float::with_val(q, 1),
            Float::with_val(q, 1),
        ];
        let norm = vector
            .iter()
            .fold(Float::with_val(q, 0), |a, x| a + Float::with_val(q, x * x));
        for shift in [-500_000_000i32, 0, 500_000_000] {
            let matrix = (0..16)
                .map(|j| {
                    ((Float::with_val(q, i32::from(j / 4 == j % 4))
                        - Float::with_val(q, &vector[j / 4] * &vector[j % 4]) / &norm)
                        << shift)
                        .to_string()
                })
                .collect();
            let (report, mut co) = capture(&s, &input(&m, q, matrix, 4), 4);
            // Eigenvectors are defined up to global sign. Normalize that sign
            // only in this exact polynomial oracle; roots and moments are invariant.
            if co[3] < 0 {
                for value in &mut co {
                    *value = -value.clone();
                }
            }
            // Four exact signs prove three distinct real roots independently
            // of the root isolator or the computed vector's eigenpair accuracy.
            for (x, positive) in [
                (Rational::from(-2), false),
                (Rational::from((-1, 2)), true),
                (Rational::from(0), false),
                (Rational::from(1), true),
            ] {
                assert_eq!(
                    evaluate(&co, &x) > 0,
                    positive,
                    "q={q} shift={shift} x={x} coefficients={co:?}"
                );
                checks += 1;
            }
            assert_eq!(report.rows.len(), 3);
            let a = co[1].clone() / &co[0];
            let b = co[2].clone() / &co[0];
            let c = co[3].clone() / &co[0];
            let first = -a.clone();
            let second: Rational = a.clone() * &a - b.clone() * 2i32;
            let third: Rational = -a.clone() * &a * &a + a * &b * 3i32 - c * 3i32;
            for (name, expected) in ["one", "two", "three"].iter().zip([first, second, third]) {
                assert_eq!(
                    point(&report.values[&format!("band_inverse_moment_{name}")], p),
                    Float::with_val(p, expected)
                );
                checks += 1;
            }
            let mut previous: Option<Rational> = None;
            for row in &report.rows {
                let lower = decimal(&row.values["rounded_polynomial_root_lower"]);
                let upper = decimal(&row.values["rounded_polynomial_root_upper"]);
                assert!(lower <= upper);
                assert!(evaluate(&co, &lower) * evaluate(&co, &upper) <= 0);
                if let Some(previous) = previous {
                    assert!(previous < lower);
                }
                previous = Some(upper);
                checks += 1;
            }
        }
    }
    assert_eq!(checks, 90);
}
