#![cfg(feature = "hp")]
use rug::Float;
use xc_spectral::ccm::{
    hp::{band_concentration_matrix_hp, HighPrecConfig},
    CcmParams,
};

#[test]
fn wide_bands_match_independent_sine_integral_antiderivatives() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/band_resolution_oracle.json")).unwrap();
    for p in [128, 256] {
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = p;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        for case in reference["cases"].as_array().unwrap() {
            let params = CcmParams::from_lambda_sq_integer(
                case["cutoff"].as_u64().unwrap(),
                case["n_modes"].as_u64().unwrap() as usize,
            );
            let omega = Float::with_val(p, case["omega"].as_u64().unwrap());
            let actual = band_concentration_matrix_hp(&params, &cfg, &omega).unwrap();
            for (index, (actual, expected)) in actual
                .iter()
                .zip(case["entries"].as_array().unwrap())
                .enumerate()
            {
                let exact =
                    Float::with_val(p + 64, Float::parse(expected.as_str().unwrap()).unwrap());
                let error = (Float::with_val(p + 64, actual) - exact).abs();
                assert!(
                    error < (Float::with_val(p + 64, 1) >> (p - 24)),
                    "p={p} omega={omega} n={} entry={index} error={error}",
                    params.n_modes
                );
            }
            let dim = 2 * params.n_modes + 1;
            for i in 0..dim {
                assert!(actual[i * dim + i] > 0 && actual[i * dim + i] <= 1);
                for j in 0..dim {
                    assert_eq!(actual[i * dim + j], actual[j * dim + i]);
                }
            }
        }
    }
}

#[test]
fn excessive_bandwidth_is_rejected_before_unbounded_quadrature() {
    let params = CcmParams::from_lambda_sq_integer(13, 0);
    let cfg = HighPrecConfig::for_decimal_digits(40);
    let huge = Float::with_val(cfg.precision_bits, 1) << 100;
    assert!(band_concentration_matrix_hp(&params, &cfg, &huge).is_err());
}
