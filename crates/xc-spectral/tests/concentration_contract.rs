#![cfg(feature = "hp")]
use rug::Float;
use xc_spectral::ccm::{
    hp::{band_concentration_matrix_hp, HighPrecConfig},
    CcmParams,
};
fn config(p: u32) -> HighPrecConfig {
    let mut cfg = HighPrecConfig::for_decimal_digits(20);
    cfg.precision_bits = p;
    cfg.quad_points = 128;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    cfg
}
#[test]
fn finite_concentration_matches_independent_continuum_integrals() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/concentration_oracle.json")).unwrap();
    let mut reports = 0;
    let mut scalars = 0;
    for p in [128, 256, 512] {
        let cfg = config(p);
        for case in oracle["cases"].as_array().unwrap() {
            let n = case["n_modes"].as_u64().unwrap() as usize;
            let params = CcmParams::from_lambda_sq_integer(case["cutoff"].as_u64().unwrap(), n);
            let omega =
                Float::with_val(p, 1) << case["omega_binary_exponent"].as_i64().unwrap() as i32;
            let actual = band_concentration_matrix_hp(&params, &cfg, &omega).unwrap();
            let work = p + 256;
            let expected = case["entries"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(entry, v)| {
                    // The frozen independent integral fixture uses centered
                    // exponentials phi_n. Convert it to the advertised CCM
                    // basis V_n=(-1)^n*phi_n without altering its oracle data.
                    let dim = 2 * n + 1;
                    let value = Float::with_val(work, Float::parse(v.as_str().unwrap()).unwrap());
                    if (entry / dim + entry % dim).is_multiple_of(2) {
                        value
                    } else {
                        -value
                    }
                })
                .collect::<Vec<_>>();
            let scale = expected
                .iter()
                .map(|x| x.clone().abs())
                .max_by(Float::total_cmp)
                .unwrap();
            let tolerance = scale * (Float::with_val(work, 1) >> (p - 16));
            assert_eq!(actual.len(), expected.len());
            for (j, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert!(a.is_finite());
                assert_eq!(a.prec(), p);
                let error = (Float::with_val(work, a) - e).abs();
                assert!(
                    error <= tolerance,
                    "p={p} n={n} omega={omega} entry={j} error={error} tolerance={tolerance}"
                );
                scalars += 1;
            }
            let dim = 2 * n + 1;
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(actual[j * dim + k], actual[k * dim + j]);
                }
            }
            reports += 1;
        }
    }
    assert_eq!(reports, 72);
    assert_eq!(scalars, 840);
}
#[test]
fn concentration_domains_fail_before_quadrature_or_allocation() {
    let valid = CcmParams::from_lambda_sq_integer(9, 0);
    let cfg = config(128);
    for x in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(band_concentration_matrix_hp(&valid, &cfg, &Float::with_val(128, x)).is_err());
    }
    let one = Float::with_val(128, 1);
    for c in [0, 1] {
        assert!(
            band_concentration_matrix_hp(&CcmParams::from_lambda_sq_integer(c, 0), &cfg, &one)
                .is_err()
        );
    }
    for c in [0.0, 1.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(band_concentration_matrix_hp(
            &CcmParams::from_lambda_sq_fractional(c, 0),
            &cfg,
            &one
        )
        .is_err());
    }
    for p in [0, 63, 1_000_001] {
        let mut bad = cfg.clone();
        bad.precision_bits = p;
        assert!(band_concentration_matrix_hp(&valid, &bad, &one).is_err());
    }
    let mut excessive = cfg.clone();
    excessive.quad_points = usize::MAX;
    assert!(band_concentration_matrix_hp(&valid, &excessive, &one).is_err());
    for n in [usize::MAX / 8, usize::MAX / 2, usize::MAX] {
        assert!(
            band_concentration_matrix_hp(&CcmParams::from_lambda_sq_integer(9, n), &cfg, &one)
                .is_err()
        );
    }
    for n in [0, 1, 2] {
        let zero = band_concentration_matrix_hp(
            &CcmParams::from_lambda_sq_integer(9, n),
            &cfg,
            &Float::with_val(128, 0),
        )
        .unwrap();
        assert_eq!(zero.len(), (2 * n + 1) * (2 * n + 1));
        assert!(zero.iter().all(Float::is_zero));
    }
}
