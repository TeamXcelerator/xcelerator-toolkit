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
fn run(
    s: &RetainedState,
    r: &RetainedReference,
    b: &[RetainedReference],
    p: u32,
    normalization: &str,
) -> ProjectionData {
    let o = ProjectionOptions {
        working_precision_bits: p,
        normalization: normalization.into(),
        fixed_second_component: None,
    };
    let result = capture_projection(s, r, b, &o, &context()).unwrap().value;
    assert_eq!(
        result.request["projection_arithmetic"],
        "stable_normalization_interval_gram_v1"
    );
    assert!(result.data.arithmetic_precision_bits > p);
    assert!(result.data.uncertainty.contains("arithmetic_enclosures"));
    result.data
}
fn contains(r: &ProjectionData, name: &str, value: &Float) {
    let [lo, hi] = &r.arithmetic_enclosures[name];
    let tol = (Float::with_val(2000, value.clone().abs()) + 1u32) >> 1000u32;
    assert!(
        float(lo, 2000) <= Float::with_val(2000, value + &tol)
            && float(hi, 2000) >= Float::with_val(2000, value - &tol),
        "{name} bounds [{lo},{hi}] exclude {value}"
    );
}
#[test]
fn public_projection_preserves_center_difference_and_tiny_unit_fit_residual() {
    let reference = reference_at(&["0".into(), "1".into(), "0".into()], "9");
    let basis = vec![
        reference_at(&["1".into(), "0".into(), "0".into()], "9"),
        reference_at(&["0".into(), "0".into(), "1".into()], "9"),
    ];
    for p in [128u32, 192, 256] {
        for bits in [100u32, 350, 1000] {
            let eps = Float::with_val(128, 1) >> bits;
            let source = source_at(
                &[
                    (-eps.clone()).to_string_radix(10, None),
                    "1".into(),
                    (-eps.clone()).to_string_radix(10, None),
                ],
                "9",
            );
            let r = run(&source, &reference, &[], p, "center_one");
            let e = Float::with_val(4000, &eps);
            let c = Float::with_val(4000, 1) + Float::with_val(4000, &e) * 2u32;
            let expected = Float::with_val(4000, 9).ln() * e.square() * 6u32 / c.square();
            contains(&r, "difference_norm_squared", &expected);
            let actual = float(r.difference_norm_squared.as_ref().unwrap(), 4000);
            assert!((actual / &expected - 1u32).abs() < Float::with_val(4000, 1) >> (p - 8));
            if bits <= 350 {
                let r = run(&source, &reference, &basis, p, "unit_l2_dx");
                let q = Float::with_val(4000, eps).square() * 2u32;
                let root = (Float::with_val(4000, 1) + &q).sqrt();
                let expected = (q / (&root * (Float::with_val(4000, 1) + &root))).square();
                contains(&r, "fit_residual_norm_squared", &expected);
                assert!(float(&r.arithmetic_enclosures["fit_residual_norm_squared"][0], p) > 0);
                let actual = float(r.fit_residual_norm_squared.as_ref().unwrap(), 4000);
                assert!((actual / expected - 1u32).abs() < Float::with_val(4000, 1) >> (p - 8));
            }
        }
    }
}
#[test]
fn public_projection_resolves_near_parallel_columns_and_withholds_exhausted_fit() {
    let source = source_at(&["1".into(), "2".into(), "3".into()], "9");
    let reference = reference_at(&["0".into(), "1".into(), "0".into()], "9");
    for p in [128u32, 192, 256] {
        for bits in [20u32, 60, 100, 110, 2500] {
            let eps = Float::with_val(128, 1) >> bits;
            let basis = vec![
                reference_at(
                    &["1".into(), "1".into(), eps.to_string_radix(10, None)],
                    "9",
                ),
                reference_at(
                    &[
                        "2".into(),
                        "2".into(),
                        (eps * 3u32).to_string_radix(10, None),
                    ],
                    "9",
                ),
            ];
            let r = run(&source, &reference, &basis, p, "center_one");
            if bits == 2500 {
                assert_eq!(r.outcome, "rank_or_precision_unresolved");
                assert!(r.coefficients.is_none());
                assert_eq!(r.arithmetic_precision_bits, p + 4096);
            } else {
                assert_eq!(r.outcome, "point_measurement");
                contains(
                    &r,
                    "fit_residual_norm_squared",
                    &(Float::with_val(2000, 9).ln() * 9u32 / 8u32),
                );
            }
        }
    }
}
#[test]
fn public_projection_intervals_contain_fresh_independent_projector_references() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/projection_interval_oracle.json")).unwrap();
    let mut reports = 0;
    for case in data["rows"].as_array().unwrap() {
        for p in [128u32, 192, 256] {
            for (xe, ye, sign, scaled_cols) in [
                (0i32, 0i32, 1, false),
                (600_000_000, 0, -1, false),
                (-600_000_000, 0, 1, false),
                (0, 600_000_000, -1, false),
                (-600_000_000, 600_000_000, -1, true),
            ] {
                let shift = |x: &Float, e: i32| {
                    let mut x = x.clone();
                    if e >= 0 {
                        x <<= e as u32;
                    } else {
                        x >>= (-e) as u32;
                    }
                    x
                };
                let vals = |v: &serde_json::Value, e: i32, s: i32| {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| {
                            (shift(&float(x.as_str().unwrap(), 128), e) * s)
                                .to_string_radix(10, None)
                        })
                        .collect::<Vec<_>>()
                };
                let c = case["cutoff"].as_str().unwrap();
                let source = source_at(&vals(&case["source"], xe, sign), c);
                let reference = reference_at(&vals(&case["reference"], ye, -sign), c);
                let exps = (0..case["basis"].as_array().unwrap().len())
                    .map(|a| {
                        if scaled_cols {
                            if a % 2 == 0 {
                                300
                            } else {
                                -300
                            }
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<_>>();
                let basis = case["basis"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(&exps)
                    .map(|(v, e)| reference_at(&vals(v, *e, 1), c))
                    .collect::<Vec<_>>();
                let r = run(
                    &source,
                    &reference,
                    &basis,
                    p,
                    case["normalization"].as_str().unwrap(),
                );
                assert_eq!(r.outcome, "point_measurement");
                for (key, expected) in case["values"].as_object().unwrap() {
                    let mut v = float(expected.as_str().unwrap(), 2000);
                    let scale = if let Some(a) = key.strip_prefix("a_") {
                        -exps[a.parse::<usize>().unwrap()]
                    } else if let Some(a) = key.strip_prefix("rhs_") {
                        exps[a.parse::<usize>().unwrap()]
                    } else if let Some(a) = key.strip_prefix("gram_") {
                        let ix = a
                            .split('_')
                            .map(|x| x.parse::<usize>().unwrap())
                            .collect::<Vec<_>>();
                        exps[ix[0]] + exps[ix[1]]
                    } else {
                        0
                    };
                    v = shift(&v, scale);
                    contains(&r, key, &v);
                }
                reports += 1;
            }
        }
    }
    assert_eq!(reports, 480);
}
