use xc_solver::diagonal_rank_one_spectrum_f64;

#[test]
fn exact_secular_route_matches_independent_dense_reference_across_ranges() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/rank_one_symmetric_oracle.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let vector = |name: &str| {
            case[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f64::from_bits(v.as_u64().unwrap()))
                .collect::<Vec<_>>()
        };
        let diagonal = vector("diagonal_bits");
        let update = vector("vector_bits");
        let alpha = f64::from_bits(case["alpha_bits"].as_u64().unwrap());
        let tolerance = f64::from_bits(case["tolerance_bits"].as_u64().unwrap());
        let result =
            diagonal_rank_one_spectrum_f64(&diagonal, &update, alpha, tolerance, 256).unwrap();
        assert_eq!(result.eigenvalues.len(), diagonal.len());
        assert!(result.residuals.iter().all(|v| v.is_finite() && *v >= 0.));
        for (got, expected) in result
            .eigenvalues
            .iter()
            .zip(case["roots"].as_array().unwrap())
        {
            let expected = expected.as_str().unwrap().parse::<f64>().unwrap();
            assert!(
                got.is_finite() && (got - expected).abs() <= 1.02 * tolerance * got.abs().max(1.),
                "{}: got={got}, expected={expected}, tolerance={tolerance}",
                case["label"]
            );
        }
    }
}
#[test]
fn zero_poles_and_overflowing_updates_can_have_finite_exact_eigenvalues() {
    for alpha in [-1., 1.] {
        let result = diagonal_rank_one_spectrum_f64(&[0.], &[1.], alpha, 1e-14, 200).unwrap();
        assert_eq!(result.eigenvalues, [alpha]);
        assert_eq!(result.residuals, [0.]);
    }
    let result = diagonal_rank_one_spectrum_f64(&[1.], &[1e200], 1e-300, 1e-14, 200).unwrap();
    assert!(
        result.eigenvalues[0].is_finite() && (result.eigenvalues[0] / 1e100 - 1.).abs() < 2e-14
    );
    let large = 2.0_f64.powi(1023);
    let component = 2.0_f64.powi(512);
    for sign in [-1., 1.] {
        let result =
            diagonal_rank_one_spectrum_f64(&[-sign * large], &[component], sign, 1e-14, 200)
                .unwrap();
        assert_eq!(result.eigenvalues, [sign * large]);
        assert_eq!(result.residuals, [0.]);
    }
}
#[test]
fn budget_range_and_unresolved_pole_gaps_fail_explicitly() {
    assert!(
        diagonal_rank_one_spectrum_f64(&[-2., 1., 4.], &[1., 0.5, 2.], 0.75, 1e-14, 1).is_err()
    );
    assert!(diagonal_rank_one_spectrum_f64(&[0.], &[f64::MAX], f64::MAX, 1e-14, 200).is_err());
    assert!(
        diagonal_rank_one_spectrum_f64(&[0., f64::from_bits(1)], &[1., 1.], 1., 1e-14, 200)
            .is_err()
    );
    assert!(diagonal_rank_one_spectrum_f64(&[1., 2.], &[1., 1.], 1., 1e-14, usize::MAX).is_err());
    for invalid in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(diagonal_rank_one_spectrum_f64(&[0.], &[1.], 1., invalid, 200).is_err());
    }
    assert!(diagonal_rank_one_spectrum_f64(&[1., 1.], &[1., 1.], 1., 1e-14, 200).is_err());
    assert!(diagonal_rank_one_spectrum_f64(&[1., 2.], &[0., 1.], 1., 1e-14, 200).is_err());
}
