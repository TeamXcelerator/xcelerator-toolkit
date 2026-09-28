use xc_spectral::target::{
    GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetEvaluatorF64, TargetProfileSpec,
};

fn series(coefficients: &[&str]) -> GaussianPolynomialSeriesSpec {
    GaussianPolynomialSeriesSpec {
        term_input_power: 0,
        polynomial_coefficients: coefficients.iter().map(|x| x.to_string()).collect(),
        polynomial_scale: ScalarScaleSpec::default(),
        parameter_polynomial_coefficients: vec![],
        parameter_polynomial_scale: ScalarScaleSpec::default(),
        minimum_terms: 2,
        maximum_terms: 32,
    }
}

fn spec() -> TargetProfileSpec {
    let mut auxiliary = series(&["1", "-1"]);
    auxiliary.term_input_power = 1;
    auxiliary.polynomial_scale = ScalarScaleSpec::Decimal {
        value: "2.5".into(),
    };
    auxiliary.parameter_polynomial_coefficients = vec!["0".into(), "1".into(), "0.3".into()];
    auxiliary.parameter_polynomial_scale = ScalarScaleSpec::RationalTimesSquareRoot {
        rational_numerator: -1,
        rational_denominator: 3,
        radicand_numerator: 7,
        radicand_denominator: 2,
    };
    TargetProfileSpec {
        schema_version: 1,
        profile_id: "manufactured-exponent-boundary".into(),
        base_series: Some(series(&["1"])),
        external_profile: None,
        auxiliary_series: Some(auxiliary),
    }
}

#[test]
#[allow(clippy::excessive_precision)]
fn native_auxiliary_underflow_is_a_rounded_zero() {
    let evaluator = TargetEvaluatorF64::from_spec(&spec()).unwrap();
    for u in [20.0, 23.0, 40.0] {
        assert_eq!(evaluator.auxiliary_value(u).unwrap(), 0.0);
    }
    // Independent 120-digit direct positive/negative series summation, before
    // the underflow boundary, guards against returning zero indiscriminately.
    for (u, expected) in [
        (1.1, 0.0033727595690460448495467637725048877),
        (2.5, 8.453944181169113001795074758575056e-7),
        (7.0, 1.5498549082650131013709072507e-62),
    ] {
        assert!((evaluator.auxiliary_value(u).unwrap() / expected - 1.0).abs() < 2e-12);
    }
}

#[cfg(feature = "hp")]
#[test]
fn native_rounded_zero_agrees_with_independent_high_precision_series() {
    use rug::{float::Constant, Float};
    let p = 512;
    let pi = Float::with_val(p, Constant::Pi);
    let sum = |u: u32| {
        let mut a = Float::with_val(p, 0);
        let mut b = Float::with_val(p, 0);
        for n in 1..=32u32 {
            let t = Float::with_val(p, n * u);
            let x = Float::with_val(p, &t * &t) * &pi;
            let common = Float::with_val(p, &t * (-x.clone()).exp());
            a += Float::with_val(p, &common * (Float::with_val(p, 1) - &x));
            b += Float::with_val(
                p,
                &common
                    * (Float::with_val(p, &x)
                        + Float::with_val(p, &x * &x) * Float::with_val(p, 3) / 10),
            );
        }
        (a, b)
    };
    let (a1, b1) = sum(1);
    let coefficient = a1 / b1;
    let evaluator = TargetEvaluatorF64::from_spec(&spec()).unwrap();
    for u in [20u32, 23, 40] {
        let (a, b) = sum(u);
        let value: Float =
            (a - &coefficient * b) * Float::with_val(p, u).sqrt() * Float::with_val(p, 5) / 2;
        assert!(value.clone().abs() < (Float::with_val(p, 1) >> 1075));
        assert_eq!(
            evaluator.auxiliary_value(f64::from(u)).unwrap(),
            value.to_f64()
        );
    }
}

#[cfg(feature = "hp")]
#[test]
fn hp_rejects_floor_rounded_coefficients_and_distinguishes_range_failure() {
    use rug::Float;
    use xc_spectral::target::hp::TargetEvaluator;
    for coefficients in [
        [
            "2.26343665964355341374822468983e-323228497",
            "1.31041069768837319607602998277e-323228497",
        ],
        ["-1.4e-323228497", "1"],
        ["1", "1.8e-323228497"],
    ] {
        let mut input = spec();
        input.auxiliary_series = None;
        input.base_series = Some(series(&coefficients));
        let error = TargetEvaluator::from_spec(&input, 128).unwrap_err();
        assert!(error.to_string().contains("exponent floor"), "{error:#}");
    }
    let mut input = spec();
    input.auxiliary_series = None;
    input.base_series = Some(series(&["1e-400", "2e-400"]));
    let tiny = TargetEvaluator::from_spec(&input, 128).unwrap();
    input.base_series = Some(series(&["1", "2"]));
    let ordinary = TargetEvaluator::from_spec(&input, 128).unwrap();
    for u in [1, 2, 3] {
        let x = Float::with_val(128, u);
        let ratio = tiny.try_value(&x).unwrap() / ordinary.try_value(&x).unwrap();
        assert!((ratio - 1u32).abs() < (Float::with_val(128, 1) >> 115));
    }
    for u in [16000, 20000, 100000] {
        let error = ordinary.try_value(&Float::with_val(128, u)).unwrap_err();
        assert!(
            error.to_string().contains("supported exponent range"),
            "{error:#}"
        );
        assert!(!error.to_string().contains("did not converge"));
    }
}
