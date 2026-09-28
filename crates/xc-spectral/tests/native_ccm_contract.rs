use xc_spectral::ccm::{run_f64, solve_spectrum_f64, CcmParams, LambdaSq};

#[test]
fn native_roots_match_independent_integer_polynomials_across_state_scales() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/native_ccm_oracle.json")).unwrap();
    let mut checked = 0;
    for case in oracle["root_cases"].as_array().unwrap() {
        let positive: Vec<f64> = case["positive_coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as f64)
            .collect();
        let full: Vec<f64> = positive
            .iter()
            .skip(1)
            .rev()
            .chain(positive.iter())
            .copied()
            .collect();
        for length in [0.25, 1.0, 3.0] {
            for scale in [1.0, -1.0, 1e-300, -1e-300, 1e300, -1e300] {
                let state: Vec<_> = full.iter().map(|v| v * scale).collect();
                let roots =
                    solve_spectrum_f64(&state, positive.len() - 1, length, 1e-13, 200).unwrap();
                assert_eq!(roots.len(), positive.len() - 1);
                for (root, reference) in roots.iter().zip(case["roots_t"].as_array().unwrap()) {
                    let expected = 2.0 * std::f64::consts::PI / length
                        * reference.as_str().unwrap().parse::<f64>().unwrap().sqrt();
                    assert!(
                        (root - expected).abs() <= 2e-12 * (1.0 + expected.abs()),
                        "scale={scale} length={length} root={root} expected={expected}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 1134);
    let exterior = solve_spectrum_f64(&[1.0, -3.0, 1.0], 1, 1.0, 1e-13, 200).unwrap();
    assert_eq!(exterior.len(), 1);
    assert!((exterior[0] - 2.0 * std::f64::consts::PI * 3.0_f64.sqrt()).abs() < 1e-11);
}

#[test]
fn native_domains_budgets_and_physical_ordinates_fail_explicitly() {
    for state in [[0.0; 3], [100.0, 1.0, 1.0], [1.0, -2.0, 1.0], [f64::NAN; 3]] {
        assert!(solve_spectrum_f64(&state, 1, 1.0, 1e-12, 200).is_err());
    }
    for budget in [0, 1] {
        assert!(solve_spectrum_f64(&[1.0; 3], 1, 1.0, 1e-12, budget).is_err());
    }
    for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(solve_spectrum_f64(&[1.0; 3], 1, 1.0, tolerance, 200).is_err());
    }
    for length in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
        assert!(solve_spectrum_f64(&[1.0; 3], 1, length, 1e-12, 200).is_err());
    }
    assert_eq!(
        solve_spectrum_f64(&[1.0], 0, 1.0, 1e-12, 200).unwrap(),
        Vec::<f64>::new()
    );
    assert!(solve_spectrum_f64(&[0.0], 0, 1.0, 1e-12, 200).is_err());
    for length in [f64::MIN_POSITIVE, f64::MAX] {
        let actual = solve_spectrum_f64(&[1.0; 3], 1, length, 1e-13, 200).unwrap()[0];
        let expected = (2.0 * std::f64::consts::PI / 3.0_f64.sqrt()) / length;
        assert!(actual.is_finite() && actual > 0.0);
        assert!((actual / expected - 1.0).abs() < 2e-11);
    }
    assert!(run_f64(&CcmParams::from_lambda_sq_integer(u64::MAX, 0)).is_err());
    assert!(run_f64(&CcmParams::from_lambda_sq_integer(5, usize::MAX)).is_err());
    let mismatched = CcmParams {
        lambda_sq: LambdaSq {
            value_u64: 2,
            value_f64: 13.0,
            is_integer: true,
        },
        n_modes: 0,
    };
    assert!(run_f64(&mismatched).is_err());
}

#[test]
fn native_even_state_admission_preserves_resolved_ritz_oracles() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/native_ccm_oracle.json")).unwrap();
    for case in oracle["ritz_cases"].as_array().unwrap() {
        let cutoff = case["cutoff"].as_u64().unwrap();
        let modes = case["modes"].as_u64().unwrap() as usize;
        let result = run_f64(&CcmParams::from_lambda_sq_integer(cutoff, modes));
        // These two states have boundary sums below the dimension/gap-scaled
        // binary64 uncertainty floor. All 24 matrix eigenvalue oracles remain
        // checked directly in ccm::tests, including these rejected states.
        if modes == 3 && matches!(cutoff, 29 | 100) {
            let error = result.expect_err("unresolved boundary normalization must fail");
            assert!(
                error
                    .to_string()
                    .contains("boundary sum unresolved in binary64"),
                "C={cutoff} N={modes}: {error}"
            );
            continue;
        }
        let report = result.unwrap_or_else(|e| panic!("C={cutoff} N={modes}: {e}"));
        let expected = case["even_minimum"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!(
            (report.weil_min_eigenvalue - expected).abs() < 2e-12 * (1.0 + expected.abs()),
            "C={cutoff} N={modes}"
        );
        assert!(report.xi.iter().all(|v| v.is_finite()));
        assert!(report
            .eigenvalues_pos
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
    }
}
