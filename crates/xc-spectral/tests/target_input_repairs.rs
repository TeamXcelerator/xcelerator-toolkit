use xc_spectral::target::{
    ExternalProfileSpec, GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetEvaluatorF64,
    TargetProfileSpec,
};

fn external(input: serde_json::Value) -> TargetProfileSpec {
    TargetProfileSpec {
        schema_version: 3,
        profile_id: "manufactured-exact-provider-input".into(),
        base_series: None,
        external_profile: Some(ExternalProfileSpec {
            lambda_squared: "13".into(),
            evaluation_precision_bits: 256,
            provider_sha256: "a".repeat(64),
            input,
        }),
        auxiliary_series: None,
    }
}

#[test]
fn external_input_rejects_precision_loss_before_digest_or_provider_launch() {
    for literal in [
        "0.12345678901234567890123456789012345678901234567890",
        "0.123456789012345678",
        "18446744073709551616",
        "-9223372036854775809",
        "1.0",
    ] {
        let nested = format!(r#"{{"nested":[{{"coefficient":{literal}}}]}}"#);
        let spec = external(serde_json::from_str(&nested).unwrap());
        assert!(spec.validate().is_err(), "{literal}");
        assert!(spec.digest().is_err(), "{literal}");
        assert!(TargetProfileSpec::from_json(&serde_json::to_vec(&spec).unwrap()).is_err());
    }
}

#[test]
fn external_decimal_strings_and_integer_limits_are_preserved_and_distinct() {
    let hi = external(
        serde_json::json!({"c":"0.123456789012345678901234567890", "n":u64::MAX, "negative":i64::MIN}),
    );
    let lo = external(
        serde_json::json!({"c":"0.123456789012345678", "n":u64::MAX, "negative":i64::MIN}),
    );
    assert_ne!(hi.digest().unwrap(), lo.digest().unwrap());
    let roundtrip = TargetProfileSpec::from_json(&serde_json::to_vec(&hi).unwrap()).unwrap();
    assert_eq!(hi, roundtrip);
    // Decimal inputs are unchanged, but the nonce-bound provider protocol has a
    // distinct cache identity so older unbound replies cannot supply new artifacts.
    let original_domain =
        serde_json::to_vec(&("external-target-provider-range-checked-v2", &hi)).unwrap();
    assert_ne!(
        hi.digest().unwrap(),
        xc_cache::ContentDigest::sha256(&original_domain).0
    );
    let current_domain =
        serde_json::to_vec(&("external-target-provider-nonce-bound-working-input-v3", &hi))
            .unwrap();
    assert_eq!(
        hi.digest().unwrap(),
        xc_cache::ContentDigest::sha256(&current_domain).0
    );
}

fn polynomial(coefficients: Vec<String>) -> TargetProfileSpec {
    TargetProfileSpec {
        schema_version: 1,
        profile_id: "manufactured-subnormal-coefficient".into(),
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
fn native_target_rejects_partial_subnormal_precision_loss() {
    let mut coefficients = vec!["0".into(); 64];
    coefficients[0] = "7.5e-236".into();
    coefficients[63] = "7.5e-323".into();
    let spec = polynomial(coefficients);
    assert!(TargetEvaluatorF64::from_spec(&spec).is_err());
    // Both source values are normal, but their normalized ratio is subnormal.
    let normalized = polynomial(vec!["1e100".into(), "7.5e-220".into()]);
    assert!(TargetEvaluatorF64::from_spec(&normalized).is_err());
    let valid = polynomial(vec!["1".into(), "0".into()]);
    assert_eq!(
        TargetEvaluatorF64::from_spec(&valid)
            .unwrap()
            .try_value(1.0)
            .unwrap(),
        1.0
    );
}

#[cfg(feature = "hp")]
#[test]
fn hp_subnormal_decimal_case_preserves_common_rescaling_invariance() {
    use rug::Float;
    let mut coefficients = vec!["0".into(); 64];
    coefficients[0] = "7.5e-236".into();
    coefficients[63] = "7.5e-323".into();
    let original = polynomial(coefficients);
    let mut rescaled = original.clone();
    let c = &mut rescaled
        .base_series
        .as_mut()
        .unwrap()
        .polynomial_coefficients;
    c[0] = "1".into();
    c[63] = "1e-87".into();
    let a = xc_spectral::target::hp::TargetEvaluator::from_spec(&original, 256).unwrap();
    let b = xc_spectral::target::hp::TargetEvaluator::from_spec(&rescaled, 256).unwrap();
    for u in [1, 2, 3] {
        let point = Float::with_val(256, u);
        let x = a.try_value(&point).unwrap();
        let y = b.try_value(&point).unwrap();
        assert!((x - y).abs() < (Float::with_val(256, 1) >> 220));
    }
}
