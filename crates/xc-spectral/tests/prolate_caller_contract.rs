#![cfg(feature = "hp")]
use xc_spectral::{
    ccm::{hp::HighPrecConfig, CcmParams},
    prolate::hp::ccm_prolate_distance_hp,
};
#[test]
fn public_prolate_comparison_uses_a_valid_sector_request() {
    let mut cfg = HighPrecConfig::for_decimal_digits(20);
    cfg.precision_bits = 128;
    cfg.quad_points = 96;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let result = ccm_prolate_distance_hp(
        &CcmParams::from_lambda_sq_integer(13, 3),
        &cfg,
        257,
        32,
        xc_numerics::quadrature::CacheMode::Off,
    )
    .unwrap();
    assert_eq!(result.n_grid, 257);
    assert_eq!(
        result.prolate_discretization,
        "prolate-bounded-legendre-even-v2"
    );
    assert!(result.prolate_basis_dimension <= result.n_grid);
    assert!(result.relative_l2_distance.is_finite());
    assert!(result.relative_l2_distance >= 0);
    assert!(result.relative_l2_distance <= 1);
}
