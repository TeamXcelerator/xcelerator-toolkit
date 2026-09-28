use xc_spectral::mellin::*;
const REFERENCE: &str = include_str!("fixtures/mellin-independent-reference.json");

#[test]
fn native_transform_matches_independent_full_fourier_integral() {
    let rows: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
    for row in rows["cases"].as_array().unwrap() {
        let f = |key: &str| row[key].as_str().unwrap().parse::<f64>().unwrap();
        let value = if let Some(coefficients) = row["xi"].as_array() {
            let xi: Vec<f64> = coefficients
                .iter()
                .map(|c| c.as_str().unwrap().parse().unwrap())
                .collect();
            try_xi_weighted_mellin_f64(
                f("sigma"),
                f("t"),
                f("lambda_value"),
                &xi,
                xi.len() / 2,
                256,
            )
            .unwrap()
        } else {
            try_truncated_lambda_f64(f("sigma"), f("t"), f("lambda_value"), 256).unwrap()
        };
        for (actual, expected) in [(value.0, f("re")), (value.1, f("im"))] {
            assert!(
                (actual - expected).abs() <= 2e-12 * expected.abs().max(1e-3),
                "{actual} vs {expected}"
            );
        }
    }
}

#[test]
fn native_transform_rejects_asymmetric_and_invalid_domains() {
    assert!(try_xi_weighted_mellin_f64(0.5, 1.0, 5.0, &[100.0, 1.0, 0.0], 1, 32).is_err());
    assert!(try_xi_weighted_mellin_f64(0.5, 1.0, 5.0, &[], usize::MAX, 32).is_err());
    assert!(try_xi_weighted_mellin_f64(0.5, 1.0, 1.0, &[1.0], 0, 32).is_err());
    for (s, l, n) in [
        (0.5, 5.0, 0),
        (f64::NAN, 5.0, 32),
        (0.5, 0.0, 32),
        (0.5, f64::INFINITY, 32),
    ] {
        assert!(try_truncated_lambda_f64(s, 1.0, l, n).is_err());
        assert!(truncated_lambda_f64(s, 1.0, l, n).0.is_nan());
    }
    assert_eq!(
        try_truncated_lambda_f64(0.5, 1.0, 1.0, 32).unwrap(),
        (0.0, 0.0)
    );
}

#[cfg(feature = "hp")]
#[test]
fn hp_transform_matches_independent_full_fourier_integral_and_checks_rules() {
    use rug::Float;
    use xc_numerics::quadrature::{gauss_legendre_nodes, CacheMode};
    let p = 192;
    let parse = |s: &str| Float::with_val(p, Float::parse(s).unwrap());
    let rows: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
    let (nodes, weights) = gauss_legendre_nodes(256, p, CacheMode::Off);
    for row in rows["cases"].as_array().unwrap().iter().take(2) {
        let f = |key: &str| parse(row[key].as_str().unwrap());
        let value = if let Some(coefficients) = row["xi"].as_array() {
            let xi: Vec<Float> = coefficients
                .iter()
                .map(|c| parse(c.as_str().unwrap()))
                .collect();
            try_xi_weighted_mellin_hp(
                &f("sigma"),
                &f("t"),
                &f("lambda_value"),
                &xi,
                xi.len() / 2,
                &nodes,
                &weights,
            )
            .unwrap()
        } else {
            try_truncated_lambda_hp(&f("sigma"), &f("t"), &f("lambda_value"), &nodes, &weights)
                .unwrap()
        };
        for (actual, expected) in [(value.0, f("re")), (value.1, f("im"))] {
            assert!((actual - expected).abs() < (Float::with_val(p, 1) >> 100u32));
        }
    }
    let sigma = parse("0.5");
    let t = parse("1");
    let lambda = parse("5");
    assert!(try_truncated_lambda_hp(&sigma, &t, &lambda, &[], &[]).is_err());
    assert!(try_truncated_lambda_hp(&sigma, &t, &lambda, &nodes, &weights[..255]).is_err());
    assert!(try_truncated_lambda_hp(
        &sigma,
        &t,
        &lambda,
        &[parse("-0.5"), parse("0.5")],
        &[parse("1"), parse("1")]
    )
    .is_err());
    assert!(try_xi_weighted_mellin_hp(
        &sigma,
        &t,
        &lambda,
        &[parse("100"), parse("1"), parse("0")],
        1,
        &nodes,
        &weights
    )
    .is_err());
}
