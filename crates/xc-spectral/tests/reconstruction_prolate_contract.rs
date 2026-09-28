use xc_spectral::{
    distance::WeilEigenfunctionF64,
    prolate::{compare_xi_to_k_lambda_f64, try_build_pw_matrix_f64, ProlateConfig},
};

#[test]
fn reconstruction_validates_every_coefficient_and_checked_shape() {
    for xi in [[0., 1., 1.], [f64::NAN, 1., 1.], [f64::INFINITY, 1., 1.]] {
        assert!(WeilEigenfunctionF64::from_v_basis(&xi, 1, 2.).is_err());
        assert!(compare_xi_to_k_lambda_f64(&xi, 1, 2., &[1., 2.], &[1., 1.]).is_err());
    }
    assert!(WeilEigenfunctionF64::from_v_basis(&[], usize::MAX, 2.).is_err());
}
#[test]
fn finite_large_lambda_reconstructs_without_overflowing_products() {
    let f = WeilEigenfunctionF64::from_v_basis(&[0.1, 1., 0.1], 1, 1e308).unwrap();
    assert_eq!(f.eval(1.), 1.);
    assert!((f.eval(1e308) - 1.5).abs() < 1e-13);
}
#[test]
fn prolate_fit_is_scale_invariant_and_rejects_nan() {
    let xi = [(2f64 * 2f64.ln()).sqrt()];
    for scale in [1e-200, 1., 1e200] {
        let c = compare_xi_to_k_lambda_f64(&xi, 0, 2., &[1., 2.], &[scale, scale]).unwrap();
        assert!((c.optimal_scalar * scale - 1.).abs() < 1e-14);
        assert!(c.linf_error < 1e-14 && c.l2_error < 1e-14);
    }
    assert!(compare_xi_to_k_lambda_f64(&xi, 0, 2., &[1., 2.], &[f64::NAN, 1.]).is_err());
}
#[test]
fn prolate_checked_build_rejects_invalid_parameters_before_allocation() {
    for lambda in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(try_build_pw_matrix_f64(&ProlateConfig::new(lambda, 17)).is_err());
    }
    assert!(try_build_pw_matrix_f64(&ProlateConfig::new(2., usize::MAX)).is_err());
    let mut cfg = ProlateConfig::new(2., 17);
    cfg.precision_bits = 0;
    assert!(try_build_pw_matrix_f64(&cfg).is_err());
}
fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/prolate_stencil_oracle.json")).unwrap()
}
#[test]
fn prolate_native_matrix_matches_independent_physical_grid_flux() {
    for case in oracle()["cases"].as_array().unwrap() {
        let lambda = case["lambda"].as_str().unwrap().parse().unwrap();
        let n = case["grid"].as_u64().unwrap() as usize;
        let (d, e) = try_build_pw_matrix_f64(&ProlateConfig::new(lambda, n)).unwrap();
        let reference = case["diagonal"]
            .as_array()
            .unwrap()
            .iter()
            .chain(case["off_diagonal"].as_array().unwrap())
            .map(|v| v.as_str().unwrap().parse::<f64>().unwrap());
        for (got, expected) in d.iter().chain(&e).zip(reference) {
            assert!((got - expected).abs() <= 2e-14 * expected.abs().max(1.));
        }
        assert!(d.iter().eq(d.iter().rev()));
        assert!(e.iter().eq(e.iter().rev()));
    }
}

fn state_oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/prolate_state_oracle.json")).unwrap()
}
#[test]
fn native_prolate_selected_states_and_samples_match_independent_dense_oracle() {
    for case in state_oracle()["cases"].as_array().unwrap() {
        let lambda = case["lambda_value"].as_str().unwrap().parse().unwrap();
        let n = case["grid"].as_u64().unwrap() as usize;
        let result = xc_spectral::prolate::compute_k_lambda_finite_dirichlet_f64(
            &ProlateConfig::new(lambda, n).with_n_sample(9),
        )
        .unwrap();
        let check = |got: f64, expected: &serde_json::Value| {
            let expected = expected.as_str().unwrap().parse::<f64>().unwrap();
            assert!(
                (got - expected).abs() < 2e-10 * expected.abs().max(1.),
                "grid={n},lambda={lambda},got={got},expected={expected}"
            );
        };
        check(result.eigenvalue_0, &case["eigenvalue_0"]);
        check(result.eigenvalue_4, &case["eigenvalue_4"]);
        check(result.c_0, &case["c_0"]);
        for (got, expected) in result
            .k_values
            .iter()
            .zip(case["k_values"].as_array().unwrap())
        {
            check(*got, expected);
        }
        for (got, expected) in result.u_grid.iter().zip(case["u_grid"].as_array().unwrap()) {
            check(*got, expected);
        }
    }
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{float::Special, Float};
    use xc_spectral::prolate::hp::{
        compare_xi_to_k_lambda, parity_of, try_build_pw_matrix, HpParity,
    };
    #[test]
    fn hp_prolate_precision_and_independent_flux_are_preserved() {
        let p = 256;
        for case in oracle()["cases"].as_array().unwrap() {
            let lambda =
                Float::with_val(53, Float::parse(case["lambda"].as_str().unwrap()).unwrap());
            let n = case["grid"].as_u64().unwrap() as usize;
            let (d, e) = try_build_pw_matrix(&lambda, n, p).unwrap();
            let reference = case["diagonal"]
                .as_array()
                .unwrap()
                .iter()
                .chain(case["off_diagonal"].as_array().unwrap())
                .map(|v| Float::with_val(512, Float::parse(v.as_str().unwrap()).unwrap()));
            for (got, expected) in d.iter().chain(&e).zip(reference) {
                assert_eq!(got.prec(), p);
                let tolerance = expected.clone().abs().max(&Float::with_val(512, 1)) >> 220u32;
                assert!(Float::with_val(512, got - &expected).abs() < tolerance);
            }
            assert!(d.iter().eq(d.iter().rev()));
            assert!(e.iter().eq(e.iter().rev()));
        }
    }
    #[test]
    fn hp_reflection_checks_the_center_and_is_scale_invariant() {
        let p = 256;
        for exponent in [-20000i32, 0, 20000] {
            let scale: Float = Float::with_val(p, 1) << exponent;
            let z = Float::with_val(p, 0);
            assert_eq!(
                parity_of(&[scale.clone(), scale.clone(), -scale.clone()], p),
                HpParity::Indeterminate
            );
            assert_eq!(
                parity_of(&[z.clone(), scale.clone(), z.clone()], p),
                HpParity::Even
            );
            assert_eq!(parity_of(&[scale.clone(), z, -scale], p), HpParity::Odd);
        }
    }
    #[test]
    fn hp_reconstruction_and_comparison_validate_domains_and_extreme_scales() {
        let p = 256;
        let one = Float::with_val(p, 1);
        let lambda = Float::with_val(p, 2);
        assert!(xc_spectral::distance::hp::WeilEigenfunction::from_v_basis(
            &[Float::with_val(p, Special::Nan), one.clone(), one.clone()],
            1,
            &lambda,
            p
        )
        .is_err());
        assert!(try_build_pw_matrix(&lambda, 17, 0).is_err());
        assert!(try_build_pw_matrix(&lambda, 17, u32::MAX).is_err());
        assert!(try_build_pw_matrix(&lambda, usize::MAX, p).is_err());
        let huge: Float = one.clone() << (rug::float::exp_max() / 2 + 100);
        let f = xc_spectral::distance::hp::WeilEigenfunction::from_v_basis(
            &[one.clone(), Float::with_val(p, 3), one.clone()],
            1,
            &huge,
            p,
        )
        .unwrap();
        assert!((f.eval(&huge) - 5u32).abs() < (one.clone() >> 200u32));
        let xi = [(lambda.clone().ln() * 2u32).sqrt()];
        let grid = [one.clone(), lambda.clone()];
        for scale in [huge.clone(), one.clone(), one.clone() / huge] {
            let c =
                compare_xi_to_k_lambda(&xi, 0, &lambda, &grid, &[scale.clone(), scale.clone()], p)
                    .unwrap();
            assert!((c.optimal_scalar * scale - 1u32).abs() < (one.clone() >> 200u32));
            assert!(c.l2_error < (one.clone() >> 200u32));
        }
    }
    #[test]
    fn hp_node_count_interpolation_and_projected_forms_are_checked() {
        use xc_spectral::prolate::hp::{build_pw_subspace_forms, try_count_nodes, try_interp_grid};
        let p = 256;
        let one = Float::with_val(p, 1);
        for exponent in [-20000i32, 0, 20000] {
            let a = one.clone() << exponent;
            assert_eq!(
                try_count_nodes(&[a.clone(), Float::with_val(p, 0), -a.clone(), a], p).unwrap(),
                2
            );
        }
        assert!(try_count_nodes(std::slice::from_ref(&one), 0).is_err());
        assert!(try_count_nodes(&[Float::with_val(p, Special::Nan)], p).is_err());
        let lambda = Float::with_val(53, 2);
        let h = Float::with_val(53, 1);
        let values = vec![one.clone(), Float::with_val(p, 2), Float::with_val(p, 3)];
        // Exact piecewise-linear values at endpoints, grid points and midpoints.
        for (x, y) in [
            (-3., 0.),
            (-2., 0.),
            (-1.5, 0.5),
            (-1., 1.),
            (-0.5, 1.5),
            (0., 2.),
            (0.5, 2.5),
            (1., 3.),
            (1.5, 1.5),
            (2., 0.),
            (3., 0.),
        ] {
            let value = try_interp_grid(&values, &lambda, &h, &Float::with_val(53, x), p).unwrap();
            assert_eq!(value.prec(), p);
            assert_eq!(value, y);
        }
        assert!(try_interp_grid(&values, &lambda, &(one.clone() >> 100u32), &one, p).is_err());
        assert!(try_interp_grid(&[], &lambda, &h, &one, p).is_err());
        assert!(try_interp_grid(&values, &lambda, &h, &one, 0).is_err());
        let huge = one << (rug::float::exp_max() / 2 + 100);
        assert!(build_pw_subspace_forms(&lambda, 1, &[vec![huge]], p).is_err());
    }
    #[test]
    fn hp_prolate_selected_states_and_samples_match_independent_dense_oracle() {
        let p = 256;
        for case in state_oracle()["cases"].as_array().unwrap() {
            let lambda = Float::with_val(
                53,
                Float::parse(case["lambda_value"].as_str().unwrap()).unwrap(),
            );
            let n = case["grid"].as_u64().unwrap() as usize;
            let result = xc_spectral::prolate::hp::compute_k_lambda_finite_dirichlet(
                &lambda,
                n,
                9,
                p,
                xc_numerics::quadrature::CacheMode::Off,
            )
            .unwrap();
            let check = |got: &Float, expected: &serde_json::Value| {
                assert_eq!(got.prec(), p);
                let expected =
                    Float::with_val(p, Float::parse(expected.as_str().unwrap()).unwrap());
                let tolerance = expected.clone().abs().max(&Float::with_val(p, 1)) >> 120u32;
                assert!(
                    Float::with_val(p, got - &expected).abs() < tolerance,
                    "grid={n},lambda={lambda},got={got},expected={expected}"
                );
            };
            check(&result.eigenvalue_0, &case["eigenvalue_0"]);
            check(&result.eigenvalue_4, &case["eigenvalue_4"]);
            check(&result.c_0, &case["c_0"]);
            for (got, expected) in result
                .k_values
                .iter()
                .zip(case["k_values"].as_array().unwrap())
            {
                check(got, expected);
            }
            for (got, expected) in result.u_grid.iter().zip(case["u_grid"].as_array().unwrap()) {
                check(got, expected);
            }
        }
    }
}

#[test]
fn exhaustive_reconstruction_preserves_subnormal_log_product() {
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/exhaustive_reconstruction_underflow_oracle.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let lambda = case["lambda_value"].as_f64().unwrap();
        let u = f64::from_bits(case["u_bits"].as_u64().unwrap());
        let expected: f64 = case["expected"].as_str().unwrap().parse().unwrap();
        let f = WeilEigenfunctionF64::from_v_basis(&[1., 3., 1.], 1, lambda).unwrap();
        let actual = f.eval(u);
        assert!(
            (actual - expected).abs() < 5e-11,
            "lambda={lambda},u={u},actual={actual},expected={expected}"
        );
    }
}

#[test]
fn exhaustive_resumed_native_coefficient_export_preserves_nonzero_range() {
    let tiny = f64::from_bits(1);
    let results: Vec<bool> = [2.0, 4.0, 3.5]
        .into_iter()
        .map(|center| {
            WeilEigenfunctionF64::from_v_basis(&[tiny, 1.0, center, 1.0, tiny], 2, 2.0).is_err()
        })
        .collect();
    assert_eq!(
        results,
        vec![true; 3],
        "overflow, zero underflow, and partial subnormal loss must be rejected"
    );
    let exact = WeilEigenfunctionF64::from_v_basis(&[tiny, 1.0, 3.0, 1.0, tiny], 2, 2.0).unwrap();
    assert_eq!(
        exact.normalized_coefficients()[2],
        tiny,
        "exact subnormal coefficients remain valid"
    );
}

#[cfg(feature = "hp")]
#[test]
fn exhaustive_resumed_hp_coefficient_export_preserves_nonzero_range() {
    use rug::Float;
    use xc_spectral::distance::hp::WeilEigenfunction;
    let p = 128;
    let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
    let results: Vec<bool> = [2.0, 4.0, 3.5]
        .into_iter()
        .map(|center| {
            WeilEigenfunction::from_v_basis(
                &[
                    tiny.clone(),
                    Float::with_val(p, 1),
                    Float::with_val(p, center),
                    Float::with_val(p, 1),
                    tiny.clone(),
                ],
                2,
                &Float::with_val(p, 2),
                p,
            )
            .is_err()
        })
        .collect();
    assert_eq!(
        results,
        vec![true; 3],
        "overflow, zero underflow, and partial exponent-range loss must be rejected"
    );
    let exact = WeilEigenfunction::from_v_basis(
        &[
            tiny.clone(),
            Float::with_val(p, 1),
            Float::with_val(p, 3),
            Float::with_val(p, 1),
            tiny.clone(),
        ],
        2,
        &Float::with_val(p, 2),
        p,
    )
    .unwrap();
    assert_eq!(exact.normalized_coefficients()[2], tiny);
}
