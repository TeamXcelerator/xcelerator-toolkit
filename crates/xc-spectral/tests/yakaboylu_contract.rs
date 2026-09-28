use xc_spectral::yakaboylu::*;
fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/yakaboylu_kernel_oracle.json")).unwrap()
}
#[test]
fn native_kernel_matches_exact_rational_formula_across_scales() {
    for (index, case) in oracle()["cases"].as_array().unwrap().iter().enumerate() {
        let v: Vec<f64> = case["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().parse().unwrap())
            .collect();
        let got = try_v_r_matrix_element_f64(v[0], v[1], v[2], v[3], v[4]).unwrap();
        for (actual, field) in [(got.0, "real"), (got.1, "imaginary")] {
            let expected = case[field].as_str().unwrap().parse::<f64>().unwrap();
            let tolerance = if expected == 0.0 {
                2e-14
            } else {
                2e-13 * expected.abs()
            };
            assert!(
                (actual - expected).abs() <= tolerance,
                "case {index} {field}: {actual} versus {expected}"
            );
        }
    }
}
#[test]
fn native_matrix_domain_and_limit_diagnostics_do_not_claim_false_positivity() {
    let (_, _, deviation) = try_test_lorentzian_limit_f64(0., 1e-40, 1e-50).unwrap();
    assert!((deviation / 1e-20 - 1.).abs() < 1e-14);
    let result = test_w_positivity_f64(&[1., 1.], 0.1).unwrap();
    assert!(!result.positive_definite && result.positive_semidefinite_with_tolerance);
    assert!(result.condition_number_f64.is_infinite());
    for epsilon in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(try_build_w_matrix_f64(&[1.], epsilon).is_err());
    }
    assert!(try_v_r_matrix_element_f64(1., 0., 1., 0., 1.).is_err());
    assert!(try_build_w_matrix_f64(&[], 1.).is_err());
    assert!(smallest_eigenvalue_f64(&[], usize::MAX).is_err());
    assert!(smallest_eigenvalue_f64(&[], 0).is_err());
    assert!(smallest_eigenvalue_f64(&[f64::NAN], 1).is_err());
    assert!(smallest_eigenvalue_f64(&[1., 99., 0., 2.], 2).is_err());
}
#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::Float;
    #[test]
    fn hp_kernel_matches_exact_formula_and_requested_precision() {
        let p = 256;
        for (index, case) in oracle()["cases"].as_array().unwrap().iter().enumerate() {
            let v: Vec<Float> = case["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| Float::with_val(128, Float::parse(x.as_str().unwrap()).unwrap()))
                .collect();
            let got = try_v_r_matrix_element_hp(&v[0], &v[1], &v[2], &v[3], &v[4], p).unwrap();
            for (actual, field) in [(&got.0, "real"), (&got.1, "imaginary")] {
                assert_eq!(actual.prec(), p);
                let expected =
                    Float::with_val(512, Float::parse(case[field].as_str().unwrap()).unwrap());
                let scale = if expected.is_zero() {
                    Float::with_val(512, 1)
                } else {
                    expected.clone().abs()
                };
                let tolerance = scale >> 220u32;
                assert!(
                    Float::with_val(512, actual - &expected).abs() < tolerance,
                    "case {index} {field}: {actual} versus {expected}"
                );
            }
        }
    }
    #[test]
    fn hp_extreme_exponents_and_mixed_precision_are_not_silently_downgraded() {
        let p = 256;
        let half = Float::with_val(p, 0.5);
        let zero = Float::with_val(p, 0);
        for exponent in [
            rug::float::exp_max() / 2 + 100,
            rug::float::exp_min() / 2 - 100,
        ] {
            let epsilon = Float::with_val(p, 1) << exponent;
            let (real, imaginary) =
                try_v_r_matrix_element_hp(&half, &zero, &half, &epsilon, &epsilon, p).unwrap();
            assert_eq!(real, 0.5);
            assert!(imaginary.is_zero());
        }
        // The difference of two finite MPFR numbers can be below its exponent
        // floor even though the normalized Lorentzian value is representable.
        let minimum = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        let higher = minimum.clone() * 1.5;
        let (real, _) =
            try_v_r_matrix_element_hp(&half, &minimum, &half, &higher, &minimum, p).unwrap();
        assert!((real - Float::with_val(p, 4) / 5u32).abs() < (Float::with_val(p, 1) >> 220u32));
        let epsilon = Float::with_val(53, 0.1);
        let matrix = xc_spectral::yakaboylu::hp::try_build_w_matrix(
            &[Float::with_val(53, 1), Float::with_val(53, 2)],
            &epsilon,
            p,
        )
        .unwrap();
        assert!(matrix.iter().all(|x| x.prec() == p));
        let result = xc_spectral::yakaboylu::hp::test_w_positivity(
            &[Float::with_val(53, 1), Float::with_val(53, 1)],
            &epsilon,
            p,
        )
        .unwrap();
        assert!(!result.positive_definite && result.positive_semidefinite_with_tolerance);
        assert!(result.condition_number.is_infinite());
        assert!(xc_spectral::yakaboylu::hp::smallest_eigenvalue(&[], usize::MAX, p).is_err());
        assert!(xc_spectral::yakaboylu::hp::try_build_w_matrix(&[half], &epsilon, 0).is_err());
    }
}
