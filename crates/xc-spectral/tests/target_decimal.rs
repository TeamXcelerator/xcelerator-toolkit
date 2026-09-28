//! Adversarial input: decimal underflow occurs before a later normalization.
use xc_spectral::target::{
    GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetEvaluatorF64, TargetProfileSpec,
};

fn tiny_coefficient_profile() -> TargetProfileSpec {
    let mut coefficients = vec!["0".to_owned(); 64];
    coefficients[0] = "1e-300".into();
    coefficients[63] = "1e-330".into();
    TargetProfileSpec {
        schema_version: 1,
        profile_id: "audit-manufactured-underflow".into(),
        base_series: Some(GaussianPolynomialSeriesSpec {
            term_input_power: 0,
            polynomial_coefficients: coefficients,
            polynomial_scale: ScalarScaleSpec::default(),
            parameter_polynomial_coefficients: vec![],
            parameter_polynomial_scale: ScalarScaleSpec::default(),
            minimum_terms: 2,
            maximum_terms: 100,
        }),
        external_profile: None,
        auxiliary_series: None,
    }
}

#[test]
fn native_target_must_not_silently_discard_nonzero_decimal_coefficients() {
    let spec = tiny_coefficient_profile();
    spec.validate().unwrap();
    let result = TargetEvaluatorF64::from_spec(&spec);
    if let Ok(evaluator) = &result {
        eprintln!(
            "native value at 2: {:.17e}",
            evaluator.try_value(2.0).unwrap()
        );
    }
    #[cfg(feature = "hp")]
    {
        let evaluator = xc_spectral::target::hp::TargetEvaluator::from_spec(&spec, 192).unwrap();
        let value = evaluator.try_value(&rug::Float::with_val(192, 2)).unwrap();
        eprintln!("HP value at 2: {value}");
    }
    assert!(result.is_err(), "nonzero coefficients outside binary64 range must not silently change the normalized mathematical target");
}

#[test]
fn native_target_rejects_underflow_during_normalization() {
    let mut spec = tiny_coefficient_profile();
    spec.base_series.as_mut().unwrap().polynomial_coefficients =
        vec!["1e308".into(), "1e-300".into()];
    assert!(TargetEvaluatorF64::from_spec(&spec).is_err());
}

#[test]
fn native_target_rejects_underflowing_auxiliary_coefficients_and_scales() {
    let mut spec = tiny_coefficient_profile();
    spec.base_series.as_mut().unwrap().polynomial_coefficients = vec!["1".into()];
    let mut auxiliary = spec.base_series.clone().unwrap();
    auxiliary.parameter_polynomial_coefficients = vec!["1".into(), "1e-330".into()];
    spec.auxiliary_series = Some(auxiliary.clone());
    assert!(TargetEvaluatorF64::from_spec(&spec).is_err());
    auxiliary.parameter_polynomial_coefficients = vec!["1".into()];
    auxiliary.polynomial_scale = ScalarScaleSpec::Decimal {
        value: "1e-330".into(),
    };
    spec.auxiliary_series = Some(auxiliary);
    assert!(TargetEvaluatorF64::from_spec(&spec).is_err());
}

#[test]
fn literal_zero_coefficients_remain_valid() {
    let mut spec = tiny_coefficient_profile();
    spec.base_series.as_mut().unwrap().polynomial_coefficients =
        vec!["1".into(), "-0.00e-330".into()];
    let evaluator = TargetEvaluatorF64::from_spec(&spec).unwrap();
    assert_eq!(evaluator.try_value(1.0).unwrap(), 1.0);
}

#[cfg(feature = "hp")]
#[test]
fn hp_retains_small_coefficients_and_is_invariant_under_common_decimal_rescaling() {
    use rug::Float;
    let original = tiny_coefficient_profile();
    let mut scaled = original.clone();
    let coefficients = &mut scaled.base_series.as_mut().unwrap().polynomial_coefficients;
    coefficients[0] = "1".into();
    coefficients[63] = "1e-30".into();
    let hp = xc_spectral::target::hp::TargetEvaluator::from_spec(&original, 192).unwrap();
    let hp_scaled = xc_spectral::target::hp::TargetEvaluator::from_spec(&scaled, 192).unwrap();
    let native_scaled = TargetEvaluatorF64::from_spec(&scaled).unwrap();
    for u in [1, 2, 3] {
        let a = hp.try_value(&Float::with_val(192, u)).unwrap();
        let b = hp_scaled.try_value(&Float::with_val(192, u)).unwrap();
        assert!((a.clone() - b).abs() < (Float::with_val(192, 1) >> 170));
        assert!((native_scaled.try_value(f64::from(u)).unwrap() - a.to_f64()).abs() < 1e-12);
    }
}
