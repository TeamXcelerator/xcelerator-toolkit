use xc_spectral::ccm::rank_one::*;
fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/rank_one_oracle.json")).unwrap()
}

#[test]
fn native_quotient_spectrum_matches_independent_secular_roots_at_common_scales() {
    for case in oracle()["cases"].as_array().unwrap() {
        let weights = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as f64)
            .collect::<Vec<_>>();
        for scale in [1e-300, -1e-300, 1., 1e300, -1e300] {
            let state = weights.iter().map(|v| v * scale).collect::<Vec<_>>();
            let op = FiniteRankOneOperatorF64::from_state(&state).unwrap();
            let roots = op.spectrum(1e-10).unwrap();
            for (got, expected) in roots.iter().zip(case["roots"].as_array().unwrap()) {
                let reference = expected.as_str().unwrap().parse::<f64>().unwrap();
                assert!(
                    (got - reference).abs() < 3e-11,
                    "scale={scale}, got={got}, expected={reference}"
                );
            }
        }
    }
}
#[test]
fn native_rank_one_normalization_and_symbolic_quotient_preserve_range() {
    let state = [1e300, 1., -1e300];
    assert_eq!(
        FiniteRankOneOperatorF64::from_state(&state)
            .unwrap()
            .normalized_state(),
        state
    );
    let op = FiniteRankOneOperatorF64::from_state(&[f64::MAX, -f64::MAX, 1., 0., 0.]).unwrap();
    assert!(op.quotient_matrix().iter().all(|v| v.is_finite()));
    assert_eq!(op.quotient_matrix()[(0, 0)], f64::MAX);
    for state in [[0., 0., 0.], [1., 0., -1.], [1., f64::NAN, 1.]] {
        assert!(FiniteRankOneOperatorF64::from_state(&state).is_err());
    }
    assert!(FiniteRankOneOperatorF64::from_state(&[1., -1., 1.])
        .unwrap()
        .spectrum(1e-12)
        .is_err());
}
#[test]
fn native_physical_ordinates_do_not_return_successful_infinities() {
    let op = FiniteRankOneOperatorF64::from_state(&[1.; 3]).unwrap();
    assert!(op.spectrum_ordinates(f64::from_bits(1), 1e-12).is_err());
    for invalid in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(op.spectrum_ordinates(invalid, 1e-12).is_err());
    }
    let values = op
        .spectrum_ordinates(2. * std::f64::consts::PI, 1e-12)
        .unwrap();
    assert!((values[1] - 1. / 3f64.sqrt()).abs() < 1e-13);
}
#[test]
fn semantic_comparison_rejects_empty_identity_before_evaluation() {
    struct Uncalled(&'static str);
    impl CcmSemanticEvaluatorF64 for Uncalled {
        fn route_id(&self) -> &'static str {
            self.0
        }
        fn independence_class(&self) -> &'static str {
            self.0
        }
        fn evaluate(&self) -> Result<Vec<f64>, RankOneError> {
            panic!("invalid identity should fail before evaluation")
        }
    }
    for empty in ["", " ", "\t"] {
        assert!(compare_three_semantic_evaluators_f64(
            &Uncalled(empty),
            &Uncalled("source"),
            &Uncalled("zero"),
            1e-12
        )
        .is_err());
    }
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{Float, Rational};
    use xc_spectral::ccm::rank_one::hp::spectrum_from_weil_metric;
    fn parse(s: &str, p: u32) -> Float {
        Float::with_val(p, Float::parse(s).unwrap())
    }
    #[test]
    fn hp_metric_spectra_match_independent_secular_roots() {
        let p = 256;
        for case in oracle()["cases"].as_array().unwrap() {
            let state = case["weights"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Float::with_val(p, v.as_u64().unwrap()))
                .collect::<Vec<_>>();
            let matrix = case["metric_exact"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|value| {
                    Float::with_val(p, value.as_str().unwrap().parse::<Rational>().unwrap())
                })
                .collect::<Vec<_>>();
            let length = Float::with_val(p, rug::float::Constant::Pi) * 2u32;
            let result = spectrum_from_weil_metric(
                &matrix,
                &state,
                &Float::with_val(p, 0),
                &length,
                &(Float::with_val(p, 1) >> 180u32),
            )
            .unwrap();
            assert_eq!(result.dimensionless_values.len(), state.len() - 1);
            for (got, expected) in result
                .dimensionless_values
                .iter()
                .zip(case["roots"].as_array().unwrap())
            {
                let expected = parse(expected.as_str().unwrap(), p);
                assert!(
                    Float::with_val(p, got - &expected).abs() < Float::with_val(p, 1) >> 210u32,
                    "got={got}, reference={expected}"
                );
                assert_eq!(got.prec(), p);
            }
        }
    }
    #[test]
    fn hp_matrix_source_precision_and_state_scale_do_not_downgrade_output() {
        let p = 256;
        let length = Float::with_val(p, 1);
        let epsilon = Float::with_val(p, 0);
        let tolerance = Float::with_val(p, 1) >> 180u32;
        for source in [53, 256] {
            let matrix = [2, -1, -1, -1, 2, -1, -1, -1, 2].map(|v| Float::with_val(source, v));
            for exponent in [
                rug::float::exp_min() - 1,
                -900,
                0,
                900,
                rug::float::exp_max() - 1,
            ] {
                let state = vec![Float::with_val(p, 1) << exponent; 3];
                let result =
                    spectrum_from_weil_metric(&matrix, &state, &epsilon, &length, &tolerance)
                        .unwrap();
                let expected = Float::with_val(512, 3).sqrt().recip();
                assert!(
                    Float::with_val(512, &result.dimensionless_values[1] - &expected).abs()
                        < Float::with_val(512, 1) >> 230u32,
                    "source={source}, exponent={exponent}"
                );
                assert_eq!(result.ordinates[1].prec(), p);
            }
        }
    }
    #[test]
    fn hp_rejects_nonfinite_tolerance_hidden_asymmetry_and_unrepresentable_ordinates() {
        let p = 256;
        let matrix = [2, -1, -1, -1, 2, -1, -1, -1, 2].map(|v| Float::with_val(p, v));
        let state = vec![Float::with_val(p, 1); 3];
        let wrong = [
            Float::with_val(p, 1),
            Float::with_val(p, 2),
            Float::with_val(p, 1),
        ];
        let epsilon = Float::with_val(p, 0);
        let length = Float::with_val(p, 1);
        let tolerance = Float::with_val(p, 1) >> 180u32;
        for bad in [f64::NAN, f64::INFINITY, 0., -1.] {
            assert!(spectrum_from_weil_metric(
                &matrix,
                &wrong,
                &epsilon,
                &length,
                &Float::with_val(p, bad)
            )
            .is_err());
        }
        assert!(spectrum_from_weil_metric(&matrix, &wrong, &epsilon, &length, &tolerance).is_err());
        let nonsymmetric = [3, -1, -2, 0, 2, -2, 0, -1, 1].map(|v| Float::with_val(p, v));
        assert!(
            spectrum_from_weil_metric(&nonsymmetric, &state, &epsilon, &length, &tolerance)
                .is_err()
        );
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        assert!(spectrum_from_weil_metric(&matrix, &state, &epsilon, &tiny, &tolerance).is_err());
        let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 1);
        let overflowing = [
            huge.clone(),
            epsilon.clone(),
            epsilon.clone(),
            epsilon.clone(),
            huge.clone(),
            epsilon.clone(),
            epsilon.clone(),
            epsilon.clone(),
            huge.clone(),
        ];
        assert!(
            spectrum_from_weil_metric(&overflowing, &state, &(-huge), &length, &tolerance).is_err()
        );
        let negative = matrix.map(|value| -value);
        assert!(
            spectrum_from_weil_metric(&negative, &state, &epsilon, &length, &tolerance).is_err()
        );
    }
}
