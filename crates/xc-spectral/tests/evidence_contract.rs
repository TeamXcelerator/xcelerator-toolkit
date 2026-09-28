#![cfg(feature = "hp")]
use rug::{Float, Integer, Rational};
use xc_numerics::interval::RationalInterval as I;
use xc_spectral::ccm::evidence::*;
fn f(p: u32, s: &str) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn bounds(p: u32) -> Vec<ActiveTruncationBound> {
    [
        ComparisonTruncationKind::ValueSpace,
        ComparisonTruncationKind::CoefficientSpace,
        ComparisonTruncationKind::FormNorm,
    ]
    .into_iter()
    .map(|kind| ActiveTruncationBound {
        kind,
        upper_bound: Float::with_val(p, 0),
        source: "explicit finite fixture; no omitted samples".into(),
    })
    .collect()
}
fn shift(v: &[Float], e: i32, sign: i32) -> Vec<Float> {
    v.iter()
        .map(|x| {
            let mut y = x.clone();
            y <<= e;
            y *= sign;
            y
        })
        .collect()
}

#[test]
fn state_comparisons_match_independent_spd_quadratic_forms() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/evidence_oracle.json")).unwrap();
    let mut count = 0;
    for case in oracle["comparison_cases"].as_array().unwrap() {
        for p in [64, 128, 256] {
            let vectors: Vec<Vec<Float>> = case["vectors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| f(p, x.as_str().unwrap()))
                        .collect()
                })
                .collect();
            let matrix: Vec<Float> = case["matrix"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f(p, v.as_str().unwrap()))
                .collect();
            for (we, pe, ae, sign) in [
                (0, 0, 0, 1),
                (-700_000_000, 700_000_000, -600_000_000, -1),
                (700_000_000, -700_000_000, 600_000_000, 1),
                (100, -100, -100, -1),
                (-100, 100, 100, 1),
            ] {
                let got = compare_prolate_weil_states_hp(
                    &shift(&vectors[0], we, sign),
                    &shift(&vectors[1], pe, sign),
                    &shift(&vectors[2], we, sign),
                    &shift(&vectors[3], pe, sign),
                    &shift(&matrix, ae, 1),
                    bounds(p),
                    p,
                )
                .unwrap();
                let values = [
                    got.value_space_overlap,
                    got.value_space_residual,
                    got.coefficient_overlap,
                    got.coefficient_residual,
                    got.prolate_rayleigh_quotient,
                    got.weil_rayleigh_quotient,
                    got.prolate_eigen_residual,
                    got.weil_eigen_residual,
                    got.form_norm_difference,
                ];
                for (i, x) in values.iter().enumerate() {
                    assert_eq!(x.prec(), p);
                    assert!(x.is_finite());
                    let mut actual = Float::with_val(768, x);
                    if (4..=7).contains(&i) {
                        actual >>= ae;
                    } else if i == 8 {
                        actual >>= ae / 2;
                    }
                    let expected = f(768, case["expected"][i].as_str().unwrap());
                    let error = Float::with_val(768, &actual - &expected).abs();
                    let tolerance = expected.clone().abs().max(&Float::with_val(768, 1))
                        * (Float::with_val(768, 1) >> (p - 10));
                    assert!(error<tolerance,"{} p={p} scales={we},{pe},{ae} field={i}: got={actual} expected={expected}",case["name"]);
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 2700);
}

#[test]
fn comparison_rejects_invalid_forms_and_recovers_tiny_nonzero_residuals() {
    let p = 128;
    let v = |xs: &[i32]| {
        xs.iter()
            .map(|x| Float::with_val(p, *x))
            .collect::<Vec<_>>()
    };
    let a = v(&[1, 0, 0, 1]);
    let w = v(&[1, 0]);
    for precision in [0, 63, 1_000_001] {
        assert!(compare_prolate_weil_states_hp(&w, &w, &w, &w, &a, bounds(p), precision).is_err());
    }
    for bad in [
        v(&[1, 2, 0, 1]),
        v(&[-1, 0, 0, 2]),
        v(&[1, 2, 2, 1]),
        v(&[1, 0, 0, 0]),
        vec![Float::with_val(p, rug::float::Special::Nan); 4],
    ] {
        assert!(compare_prolate_weil_states_hp(&w, &w, &w, &w, &bad, bounds(p), p).is_err());
    }
    assert!(compare_prolate_weil_states_hp(&w, &w[..1], &w, &w, &a, bounds(p), p).is_err());
    assert!(compare_prolate_weil_states_hp(&v(&[0, 0]), &w, &w, &w, &a, bounds(p), p).is_err());
    let mut incomplete = bounds(p);
    incomplete.pop();
    assert!(compare_prolate_weil_states_hp(&w, &w, &w, &w, &a, incomplete, p).is_err());
    let mut tiny = Float::with_val(p, 1.75);
    tiny >>= 536_870_913u32;
    let tiny = vec![tiny.clone(), tiny];
    let got = compare_prolate_weil_states_hp(&tiny, &tiny, &tiny, &tiny, &a, bounds(p), p).unwrap();
    assert!((got.value_space_overlap - 1u32).abs() < f(p, "1e-35"));
    assert!(got.weil_eigen_residual.is_zero());
    let scale = f(p, "1e-200000000");
    let matrix = vec![
        scale.clone(),
        Float::with_val(p, 0),
        Float::with_val(p, 0),
        Float::with_val(p, &scale * 2),
    ];
    let mixed = v(&[3, 4]);
    let got = compare_prolate_weil_states_hp(&mixed, &mixed, &mixed, &mixed, &matrix, bounds(p), p)
        .unwrap();
    let mut relative = got.weil_eigen_residual / scale;
    relative -= f(p, "0.48");
    assert!(relative.abs() < f(p, "1e-35"));
}

#[test]
fn asymptotic_and_log_evaluations_match_independent_fuchs_expression() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/evidence_oracle.json")).unwrap();
    for case in oracle["asymptotic_cases"].as_array().unwrap() {
        for p in [64, 128, 256] {
            let c = case["c"].as_str().unwrap().parse::<u64>().unwrap();
            let expected = f(768, case["log"].as_str().unwrap());
            let got = prolate_chi2_log_deficiency_asymptotic(c, p).unwrap();
            let tolerance = expected.clone().abs().max(&Float::with_val(768, 1))
                * (Float::with_val(768, 1) >> (p - 8));
            assert!(Float::with_val(768, got - &expected).abs() < tolerance);
            if let Some(point) = case["point"].as_str() {
                let expected = f(768, point);
                let got = try_prolate_chi2_deficiency_asymptotic(c, p).unwrap();
                let error = (Float::with_val(768, got) - &expected).abs() / expected;
                assert!(
                    error < (Float::with_val(768, 1) >> (p - 8)),
                    "c={c} p={p} relative error={error}"
                );
            } else {
                assert!(try_prolate_chi2_deficiency_asymptotic(c, p).is_err());
                assert!(prolate_chi2_deficiency_asymptotic(c, p).is_nan());
            }
        }
    }
    for p in [0, 1, 63, 1_000_001] {
        assert!(prolate_chi2_log_deficiency_asymptotic(13, p).is_err());
        assert!(try_prolate_chi2_deficiency_asymptotic(13, p).is_err());
        assert!(prolate_chi2_deficiency_asymptotic(13, p).is_nan());
    }
    assert!(try_prolate_chi2_deficiency_asymptotic(0, 128).is_err());
    let finite = CertifiedProlateDeficiency::from_concentration_enclosure(
        I::point(Rational::from((1, 4))),
        64,
    )
    .unwrap();
    assert!(
        ProlateWeilComparison::new(u64::MAX, finite, I::point(Rational::from(0)), 128).is_err()
    );
}

#[test]
fn exact_deficiency_propagation_has_outward_minimal_dyadic_endpoints() {
    for denominator in 1..=32 {
        for numerator in 0..=denominator {
            for bits in [0, 1, 8, 64, 127] {
                let lo = Rational::from((numerator, denominator));
                let hi = Rational::from(((numerator + 1).min(denominator), denominator));
                let got = CertifiedProlateDeficiency::from_concentration_enclosure(
                    I::new(lo.clone(), hi.clone()).unwrap(),
                    bits,
                )
                .unwrap();
                let lower = got.angular_eigenvalue().lower();
                let upper = got.angular_eigenvalue().upper();
                let step = Rational::from((Integer::from(1), Integer::from(1) << bits));
                assert!(Rational::from(lower * lower) <= lo);
                assert!(Rational::from(upper * upper) >= hi);
                let next = Rational::from(lower + &step);
                assert!(Rational::from(&next * &next) > lo);
                let prior = Rational::from(upper - &step);
                assert!(prior < 0 || Rational::from(&prior * &prior) < hi || hi == 0);
                assert_eq!(*got.deficiency().lower(), Rational::from(1) - upper);
                assert_eq!(*got.deficiency().upper(), Rational::from(1) - lower);
                assert_eq!(got.exact_finite(), got.deficiency().is_point());
                assert_eq!(got.fraction_bits(), bits);
            }
        }
    }
    assert!(CertifiedProlateDeficiency::from_concentration_enclosure(
        I::point(Rational::from(-1)),
        64
    )
    .is_err());
    assert!(CertifiedProlateDeficiency::from_concentration_enclosure(
        I::point(Rational::from(2)),
        64
    )
    .is_err());
    assert!(CertifiedProlateDeficiency::from_concentration_enclosure(
        I::point(Rational::from(1)),
        1_000_001
    )
    .is_err());
}

#[test]
fn a_target_eigenvalue_gap_is_reduced_by_eigenvalue_error() {
    // A=diag(0,1), unit v=(3/5,4/5), mu=16/25; r=12/25.
    // A raw eigenvalue gap of 1 would give 12/25, below both actual sin angles.
    let r = Rational::from((12, 25));
    assert!(residual_to_certified_gap(r.clone(), Rational::from(1)).is_err());
    match residual_to_complement_separation(r.clone(), Rational::from((16, 25))).unwrap() {
        ResidualGapConclusion::Reliable {
            sin_angle_upper, ..
        } => assert_eq!(sin_angle_upper, Rational::from((3, 4))),
        _ => panic!("explicit separation should be conclusive"),
    }
    assert!(matches!(
        residual_to_eigenvalue_gap(r.clone(), Rational::from(1), Rational::from((16, 25))).unwrap(),
        ResidualGapConclusion::Inconclusive { .. }
    ));
    match residual_to_eigenvalue_gap(r, Rational::from(1), Rational::from((9, 25))).unwrap() {
        ResidualGapConclusion::Reliable {
            sin_angle_upper,
            gap_lower,
            ..
        } => {
            assert_eq!(sin_angle_upper, Rational::from((3, 4)));
            assert_eq!(gap_lower, Rational::from((16, 25)));
            assert!(sin_angle_upper >= Rational::from((3, 5)));
        }
        _ => panic!("target 1 has valid separation"),
    }
    for (r, g, e) in [(-1, 2, 0), (1, 0, 0), (1, 1, 1), (1, 1, 2), (1, 1, -1)] {
        assert!(residual_to_eigenvalue_gap(
            Rational::from(r),
            Rational::from(g),
            Rational::from(e)
        )
        .is_err());
    }
}

#[test]
fn interval_form_admission_matches_known_exact_congruence_signatures() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/evidence_oracle.json")).unwrap();
    for case in oracle["form_admission_cases"].as_array().unwrap() {
        for p in [64, 128, 256] {
            let n = case["n"].as_u64().unwrap() as usize;
            let v = vec![Float::with_val(p, 1); n];
            let matrix: Vec<Float> = case["matrix"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| f(p, x.as_str().unwrap()))
                .collect();
            let result = compare_prolate_weil_states_hp(&v, &v, &v, &v, &matrix, bounds(p), p);
            assert_eq!(
                result.is_ok(),
                case["positive"].as_bool().unwrap(),
                "n={n} p={p} signature={}",
                case["signature"]
            );
        }
    }
    let mut one_plus = Float::with_val(512, 1);
    one_plus += Float::with_val(512, 1) >> 200;
    let base = vec![Float::with_val(128, 1); 2];
    let close = vec![Float::with_val(512, 1), one_plus];
    let matrix = [1, 0, 0, 1].map(|x| Float::with_val(128, x));
    let got =
        compare_prolate_weil_states_hp(&base, &close, &base, &close, &matrix, bounds(128), 64)
            .unwrap();
    let expected = f(768, oracle["mixed_precision_distance"].as_str().unwrap());
    assert!(got.coefficient_residual > 0 && got.value_space_residual > 0);
    assert!(
        ((Float::with_val(768, got.coefficient_residual) - &expected) / expected).abs()
            < (Float::with_val(768, 1) >> 54)
    );
}
