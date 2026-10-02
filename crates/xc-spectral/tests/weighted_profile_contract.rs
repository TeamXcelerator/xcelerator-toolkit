#![cfg(feature = "hp")]

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
fn f(s: &str) -> Float {
    Float::with_val(128, Float::parse(s).unwrap())
}
fn setup(
    v: Vec<String>,
    target: Vec<String>,
    basis: Vec<Vec<String>>,
) -> (RetainedState, ExternalResearchInputs) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":v.len()/2,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":v}),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let mut i:ExternalResearchInputs=serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":v.len()/2,"precision_bits":128,"convention_id":"synthetic finite weighted profile","definition_digest":ContentDigest::sha256(b"weighted contract"),"approximation_scope":"finite grid"})).unwrap();
    i.target = Some(SampledReference {
        definition_digest: ContentDigest::sha256(b"finite grid"),
        evaluation_policy: "fixed x=j*log(9)/(2n)".into(),
        approximation_scope: "finite stored point grid".into(),
        intervals: target.len() - 1,
        values: target,
        basis_values: basis,
        fixed_second_component: None,
        raw_normalizer: "1".into(),
        trial_coefficients: None,
    });
    (s, i)
}
fn run(s: &RetainedState, i: &ExternalResearchInputs, p: u32) -> ExtendedAnalysis {
    let mut o = ExtensionOptions::for_source(s);
    o.working_precision_bits = p;
    let result = capture_extended(
        "weighted_reference_projection",
        s,
        None,
        None,
        Some(i),
        &o,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(
        result.value.request["weighted_profile_arithmetic"],
        "stored_points_combined_difference_interval_gram_unresolved_v2"
    );
    result.value.data
}
fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).into()).collect()
}
#[test]
fn weighted_profiles_preserve_tiny_signals_and_declared_points() {
    let (s, i) = setup(
        strings(&["-1e-100", "1", "-1e-100"]),
        vec!["1".into(); 9],
        vec![],
    );
    let r = run(&s, &i, 192);
    assert!(f(&r.values["weighted_l1_lower"]) > 0);
    assert!(f(&r.values["weighted_l2_squared_lower"]) > 0);
    assert!(f(&r.values["signed_integral_upper"]) < 0);
    for h in ["1", "-1", "1e200000000", "1e-200000000"] {
        let (s, i) = setup(
            strings(&["0", h, "0"]),
            vec!["1.0000000000000000000000000000000000000001".into(); 9],
            vec![],
        );
        let r = run(&s, &i, 192);
        assert_eq!(f(&r.values["weighted_l1_upper"]), 0);
        assert_eq!(f(&r.values["weighted_l2_squared_upper"]), 0);
    }
}
#[test]
fn weighted_fit_raw_units_zero_denominator_and_budget_are_explicit() {
    let h = f("1") << 400u32;
    let target = (0..=8)
        .map(|j| (f("1") - Float::with_val(128, j) / 16u32).to_string_radix(10, None))
        .collect();
    let basis = vec![
        (0..=8)
            .map(|j| ((f("1") + Float::with_val(128, j) / 8u32) * &h).to_string_radix(10, None))
            .collect(),
        (0..=8)
            .map(|j| (f("1") - Float::with_val(128, j) / 8u32).to_string_radix(10, None))
            .collect(),
    ];
    let (s, mut i) = setup(strings(&["0", "1", "0"]), target, basis);
    i.target.as_mut().unwrap().fixed_second_component = Some("0.5".into());
    let r = run(&s, &i, 192);
    assert_eq!(r.outcome, "point_measurement");
    for (key, expected) in [("a_0", f("1") >> 402u32), ("a_1", f("-0.25"))] {
        assert!(
            f(&r.values[&format!("{key}_lower")]) <= expected
                && f(&r.values[&format!("{key}_upper")]) >= expected
        );
    }
    assert_eq!(f(&r.values["fit_residual_norm_squared_lower"]), 0);
    assert!(r.values.contains_key("minimum_scaled_pivot"));
    assert!(!r.values.contains_key("minimum_pivot"));
    i.target.as_mut().unwrap().values = vec!["1".into(); 9];
    let r = run(&s, &i, 192);
    assert_eq!(f(&r.values["a_0"]), 0);
    assert!(!r.values.contains_key("b_effective"));
    assert!(r.reason.unwrap().contains("contains zero"));
    let mut o = ExtensionOptions::for_source(&s);
    o.maximum_working_bytes = Some(1);
    let result = capture_extended(
        "weighted_reference_projection",
        &s,
        None,
        None,
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(result.value.data.outcome, "unresolved");
    i.target.as_mut().unwrap().basis_values[0] = vec!["1e-200000000".into(); 9];
    let o = ExtensionOptions::for_source(&s);
    let unresolved = capture_extended(
        "weighted_reference_projection",
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
    assert_eq!(unresolved.outcome, "unresolved");
    assert!(unresolved.reason.is_some());
    assert!(!unresolved.values.contains_key("a_0"));
}
#[test]
fn public_weighted_enclosures_contain_independent_grid_references() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/weighted_oracle.json")).unwrap();
    let mut count = 0;
    for case in data["rows"].as_array().unwrap() {
        for p in [128u32, 192, 256] {
            for e in [-400i32, 0, 400] {
                let source = case["source"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        let mut x = f(x.as_str().unwrap());
                        if e >= 0 {
                            x <<= e as u32;
                        } else {
                            x >>= (-e) as u32;
                            x = -x;
                        }
                        x.to_string_radix(10, None)
                    })
                    .collect();
                let target = case["target"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().to_owned())
                    .collect();
                let mut basis = case["basis"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        v.as_array()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_str().unwrap().to_owned())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for x in &mut basis[0] {
                    let mut v = f(x);
                    if e >= 0 {
                        v <<= e as u32;
                    } else {
                        v >>= (-e) as u32;
                    }
                    *x = v.to_string_radix(10, None);
                }
                let (s, i) = setup(source, target, basis);
                let r = run(&s, &i, p);
                assert_eq!(r.outcome, "point_measurement");
                for (key, value) in case["values"].as_object().unwrap() {
                    let parse = |s: &str| Float::with_val(1400, Float::parse(s).unwrap());
                    let mut expected = parse(value.as_str().unwrap());
                    let scale = if key == "a_0" {
                        -e
                    } else if key == "rhs_0" || key == "gram_0_1" || key == "gram_1_0" {
                        e
                    } else if key == "gram_0_0" {
                        2 * e
                    } else {
                        0
                    };
                    if scale >= 0 {
                        expected <<= scale as u32;
                    } else {
                        expected >>= (-scale) as u32;
                    }
                    let tolerance = (Float::with_val(1400, expected.clone().abs()) + 1u32) >> 1000;
                    assert!(
                        parse(&r.values[&format!("{key}_lower")])
                            <= Float::with_val(1400, &expected + &tolerance)
                            && parse(&r.values[&format!("{key}_upper")])
                                >= Float::with_val(1400, &expected - &tolerance),
                        "case {} p {p} e {e} {key}",
                        case["case"]
                    );
                }
                count += 1;
            }
        }
    }
    assert_eq!(count, 144);
}
#[test]
fn public_weighted_fit_checks_near_dependence_and_precision_exhaustion() {
    for bits in [20u32, 60, 100] {
        for p in [128u32, 192, 256] {
            let eps = f("1") >> bits;
            let mut basis = vec![vec![], vec![]];
            for j in 0..=8 {
                let x = Float::with_val(128, j) / 8u32;
                let a = f("1") + &x;
                basis[0].push(a.to_string_radix(10, None));
                basis[1].push((a + Float::with_val(128, &eps) * &x).to_string_radix(10, None));
            }
            let target = (0..=8)
                .map(|j| (f("1") - Float::with_val(128, j) / 16u32).to_string_radix(10, None))
                .collect();
            let (s, i) = setup(strings(&["0", "1", "0"]), target, basis);
            let r = run(&s, &i, p);
            assert_eq!(r.outcome, "point_measurement");
            let expected = f("1") << (bits - 1);
            let parse = |s: &str| Float::with_val(1000, Float::parse(s).unwrap());
            for (key, value) in [("a_0", -expected.clone()), ("a_1", expected)] {
                assert!(
                    parse(&r.values[&format!("{key}_lower")]) <= value
                        && parse(&r.values[&format!("{key}_upper")]) >= value
                );
            }
            assert_eq!(f(&r.values["fit_residual_norm_squared_lower"]), 0);
        }
    }
    let mut basis = vec![vec!["0".into(); 9], vec!["0".into(); 9]];
    basis[0][0] = "1".into();
    basis[0][1] = "3".into();
    basis[1][0] = "2".into();
    basis[1][1] = "6".into();
    basis[1][2] = (f("1") >> 2500u32).to_string_radix(10, None);
    let target = (0..=8)
        .map(|j| (f("1") - Float::with_val(128, j) / 16u32).to_string_radix(10, None))
        .collect();
    let (s, i) = setup(strings(&["0", "1", "0"]), target, basis);
    let r = run(&s, &i, 192);
    assert_eq!(r.outcome, "rank_or_precision_unresolved");
    assert!(!r.values.contains_key("a_0"));
    assert_eq!(f(&r.values["arithmetic_precision_bits"]), 4288);
}
