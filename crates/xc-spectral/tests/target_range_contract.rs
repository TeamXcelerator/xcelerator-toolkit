//! Public target contracts at arithmetic and precision boundaries.
use xc_spectral::target::{
    ExternalProfileSpec, GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetEvaluatorF64,
    TargetProfileSpec,
};

fn series(coefficients: Vec<String>) -> GaussianPolynomialSeriesSpec {
    GaussianPolynomialSeriesSpec {
        term_input_power: 0,
        polynomial_coefficients: coefficients,
        polynomial_scale: ScalarScaleSpec::default(),
        parameter_polynomial_coefficients: Vec::new(),
        parameter_polynomial_scale: ScalarScaleSpec::default(),
        minimum_terms: 2,
        maximum_terms: 1000,
    }
}

fn spec() -> TargetProfileSpec {
    TargetProfileSpec {
        schema_version: 1,
        profile_id: "generic-range-contract".into(),
        base_series: Some(series(vec!["1".into()])),
        external_profile: None,
        auxiliary_series: None,
    }
}

fn auxiliary(a: &str, b: &str) -> TargetProfileSpec {
    let mut spec = spec();
    let mut aux = series(vec!["1".into()]);
    aux.polynomial_scale = ScalarScaleSpec::Decimal { value: a.into() };
    aux.parameter_polynomial_coefficients = vec!["1".into()];
    aux.parameter_polynomial_scale = ScalarScaleSpec::Decimal { value: b.into() };
    spec.auxiliary_series = Some(aux);
    spec
}

#[test]
fn auxiliary_parameter_rejects_unrepresentable_native_ratios() {
    for (a, b) in [
        ("1e308", "1e-308"),
        ("-1e308", "1e-308"),
        ("1e-308", "1e308"),
    ] {
        assert!(TargetEvaluatorF64::from_spec(&auxiliary(a, b)).is_err());
    }
}

#[test]
fn auxiliary_value_cannot_return_ok_infinity() {
    let mut spec = auxiliary("1e308", "1");
    spec.auxiliary_series
        .as_mut()
        .unwrap()
        .parameter_polynomial_coefficients = vec!["-300".into(), "100".into()];
    let evaluator = TargetEvaluatorF64::from_spec(&spec).unwrap();
    assert!(evaluator.auxiliary_parameter().unwrap().is_finite());
    assert!(evaluator.auxiliary_value(0.25).is_err());
}

#[test]
fn gaussian_and_external_identities_reject_previous_arithmetic_semantics() {
    let spec = spec();
    let old = xc_cache::ContentDigest::sha256(
        &serde_json::to_vec(&("gaussian-series-relative-geometric-tail-v2", &spec)).unwrap(),
    );
    assert_ne!(spec.digest().unwrap(), old.0);
    let mut external = spec;
    external.schema_version = 3;
    external.base_series = None;
    external.external_profile = Some(ExternalProfileSpec {
        lambda_squared: "4".into(),
        evaluation_precision_bits: 256,
        provider_sha256: "a".repeat(64),
        input: serde_json::json!({"fixture":1}),
    });
    let old = xc_cache::ContentDigest::sha256(
        &serde_json::to_vec(&("external-target-provider-protocol-v1", &external)).unwrap(),
    );
    assert_ne!(external.digest().unwrap(), old.0);
    external.auxiliary_series = auxiliary("1", "1").auxiliary_series;
    let old = xc_cache::ContentDigest::sha256(
        &serde_json::to_vec(&(
            "external-target-provider-protocol-v1",
            "gaussian-series-relative-geometric-tail-v2",
            &external,
        ))
        .unwrap(),
    );
    assert_ne!(external.digest().unwrap(), old.0);
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{float::Constant, Float};
    use xc_spectral::target::hp::TargetEvaluator;

    #[test]
    fn gaussian_product_recovers_a_representable_value_after_factor_underflow() {
        // Separate 512-bit positive-series reference. Form each complete
        // monomial logarithm before exponentiation; no implementation helpers.
        let g = 512;
        let p = 128;
        let pi = Float::with_val(g, Constant::Pi);
        let ln2 = Float::with_val(g, 2).ln();
        let x = Float::with_val(g, 1_i64 - i64::from(rug::float::exp_min())) * &ln2 + 400_u32;
        let u = Float::with_val(p, Float::with_val(g, &x / &pi).sqrt());
        let x = Float::with_val(g, &u).square() * &pi;
        let mut denominator = Float::with_val(g, 0);
        for n in 1..=64 {
            let xn = Float::with_val(g, n * n) * &pi;
            denominator += (xn.clone().ln() * 63_u32 - xn).exp();
        }
        let log_value =
            x.clone().ln() * 63_u32 - x + Float::with_val(g, &u).ln() / 2_u32 - denominator.ln();
        let reference = log_value.exp();
        assert!(reference.is_finite() && reference > 0);
        let mut spec = spec();
        let mut coefficients = vec!["0".into(); 64];
        coefficients[63] = "1".into();
        spec.base_series = Some(series(coefficients));
        let actual = TargetEvaluator::from_spec(&spec, p)
            .unwrap()
            .try_value(&u)
            .unwrap();
        let error = (Float::with_val(g, actual / reference) - 1_u32).abs();
        assert!(error < (Float::with_val(g, 1) >> 120));
    }

    #[test]
    fn requested_precision_is_checked_before_allocation() {
        for p in [0, 1_000_001, u32::MAX] {
            assert!(TargetEvaluator::from_spec(&spec(), p).is_err());
        }
        for p in [1, 2, 64, 128] {
            let evaluator = TargetEvaluator::from_spec(&spec(), p).unwrap();
            assert_eq!(evaluator.try_value(&Float::with_val(64, 1)).unwrap(), 1);
        }
    }

    #[test]
    fn nonzero_decimals_and_unresolved_tail_cannot_silently_become_zero() {
        for text in ["1e-999999999", "-1e-999999999"] {
            let mut spec = spec();
            spec.base_series.as_mut().unwrap().polynomial_coefficients =
                vec![text.into(), "1".into()];
            assert!(TargetEvaluator::from_spec(&spec, 128).is_err());
        }
        let mut spec = spec();
        spec.base_series.as_mut().unwrap().maximum_terms = 32;
        let evaluator = TargetEvaluator::from_spec(&spec, 128).unwrap();
        assert!(evaluator.try_value(&Float::with_val(128, 1e300)).is_err());
    }

    #[test]
    fn auxiliary_ratios_respect_the_selected_backend_range() {
        let valid = TargetEvaluator::from_spec(&auxiliary("1e308", "1e-308"), 128).unwrap();
        assert!(valid.auxiliary_parameter().unwrap().is_finite());
        assert!(
            valid
                .auxiliary_value(&Float::with_val(128, 1))
                .unwrap()
                .abs()
                < Float::with_val(128, 1e280)
        );
        for (a, b) in [("1e308", "1e-323228350"), ("1e-323228350", "1e308")] {
            assert!(TargetEvaluator::from_spec(&auxiliary(a, b), 128).is_err());
        }
    }
}

#[test]
fn exhaustive_gaussian_native_partial_underflow_preserves_normalized_value() {
    // Independent 160-digit Decimal series at the exact stored binary64 u.
    let mut input = spec();
    let mut coefficients = vec!["0".to_owned(); 41];
    coefficients[40] = "1".to_owned();
    input.base_series = Some(series(coefficients));
    let evaluator = TargetEvaluatorF64::from_spec(&input).unwrap();
    let cases: &[(f64, f64)] = &[
        (15.033297016639143, 8.623048553505154e-241),
        (15.138795132120961, 6.873856442707763e-245),
        (15.295678028289107, 4.8218716508053874e-251),
        (15.389040103942165, 9.712042328483992e-255),
        (15.399378727952763, 3.771337565411061e-255),
        (15.40971041561482, 1.4643630193624588e-255),
        (15.450968080927584, 3.3262039190179675e-257),
    ];
    for &(u, expected) in cases {
        let got = evaluator.try_value(u).unwrap();
        let relative = (got / expected - 1.0).abs();
        assert!(
            relative < 2e-12,
            "u={u}: got={got:e}, expected={expected:e}, relative={relative:e}"
        );
    }
}
