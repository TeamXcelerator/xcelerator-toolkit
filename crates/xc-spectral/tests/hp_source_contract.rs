#![cfg(feature = "hp")]
use xc_spectral::ccm::{
    hp::{weil_matrix_hp, weil_plunge_cancellation_hp, weil_spectrum_hp, HighPrecConfig},
    CcmParams,
};

fn config() -> HighPrecConfig {
    let mut c = HighPrecConfig::for_decimal_digits(20);
    c.precision_bits = 64;
    c.quad_points = 8;
    c.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    c
}

#[test]
fn hp_matrix_rejects_invalid_domains_before_arithmetic_or_allocation() {
    let base = config();
    for params in [
        CcmParams::from_lambda_sq_integer(1, 0),
        CcmParams::from_lambda_sq_fractional(f64::NAN, 0),
        CcmParams::from_lambda_sq_fractional(-2.0, 0),
        CcmParams::from_lambda_sq_integer(2, usize::MAX),
    ] {
        let result = std::panic::catch_unwind(|| weil_matrix_hp(&params, &base, false));
        assert!(
            matches!(result, Ok(Err(_))),
            "invalid source did not return Err: {params:?}"
        );
    }
    let params = CcmParams::from_lambda_sq_integer(2, 0);
    let mut bad = base.clone();
    bad.precision_bits = 0;
    let result = std::panic::catch_unwind(|| weil_matrix_hp(&params, &bad, false));
    assert!(
        matches!(result, Ok(Err(_))),
        "invalid precision must not panic"
    );
}

#[test]
fn all_public_weil_matrix_consumers_validate_the_source_domain() {
    let params = CcmParams::from_lambda_sq_integer(1, 0);
    let c = config();
    assert!(weil_spectrum_hp(&params, &c, false).is_err());
    assert!(weil_plunge_cancellation_hp(&params, &c).is_err());
}

#[test]
fn ordinary_and_exact_matrix_routes_match_defining_integral_references() {
    use rug::{ops::Pow, Float};
    use xc_spectral::ccm::hp::{localized_weil_form_exact_hp, ExactLambdaSquaredHp};
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/cutoff_free_oracle.json")).unwrap();
    let mut checked = 0;
    for p in [64, 128, 256] {
        let mut cfg = config();
        cfg.precision_bits = p;
        cfg.quad_points = 192;
        let tolerance = Float::with_val(p, 2).pow(-((p - 24) as i32));
        for case in oracle["cases"].as_array().unwrap() {
            let cutoff = case["cutoff"].as_u64().unwrap();
            let modes = case["modes"].as_u64().unwrap() as usize;
            let params = CcmParams::from_lambda_sq_integer(cutoff, modes);
            let actual = weil_matrix_hp(&params, &cfg, true).unwrap();
            let exact = localized_weil_form_exact_hp(
                ExactLambdaSquaredHp::new(
                    xc_core::DecimalLiteral::new(cutoff.to_string()).unwrap(),
                    cutoff,
                )
                .unwrap(),
                modes,
                &cfg,
                true,
            )
            .unwrap();
            assert_eq!(actual, exact.matrix);
            for (value, entry) in actual.iter().zip(case["entries"].as_array().unwrap()) {
                let reference =
                    Float::with_val(p, Float::parse(entry[3].as_str().unwrap()).unwrap());
                let error = (value.clone() - &reference).abs();
                assert!(
                    error <= Float::with_val(p, &tolerance * (reference.abs() + 1)),
                    "C={cutoff}, N={modes}, p={p}, entry={checked}, error={error}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 882);
}

#[test]
fn exact_source_and_scheduled_quadrature_reject_invalid_controls() {
    use xc_spectral::ccm::hp::{localized_weil_form_exact_hp, ExactLambdaSquaredHp};
    let mut cfg = config();
    cfg.precision_bits = 0;
    let exact = ExactLambdaSquaredHp::new(xc_core::DecimalLiteral::new("2").unwrap(), 2).unwrap();
    let result = std::panic::catch_unwind(|| localized_weil_form_exact_hp(exact, 0, &cfg, false));
    assert!(matches!(result, Ok(Err(_))));
    cfg = config();
    cfg.quad_points = 0;
    assert!(weil_matrix_hp(&CcmParams::from_lambda_sq_integer(2, 0), &cfg, false).is_err());
    assert!(xc_numerics::quadrature::try_gauss_legendre_nodes_scheduled(
        0,
        64,
        xc_numerics::quadrature::CacheMode::Off,
        xc_numerics::hp_runtime::plan_gl_precompute(&[1], 64).root_schedule(1)
    )
    .is_err());
}
