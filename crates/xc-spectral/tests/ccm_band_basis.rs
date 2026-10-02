//! Regression from the defining Fourier integral in the CCM V_n basis.
#![cfg(feature = "hp")]

use rug::Float;
use xc_spectral::ccm::{
    hp::{band_concentration_matrix_hp, weil_spectrum_sonin_hp, HighPrecConfig},
    CcmParams,
};

#[test]
fn fresh_audit_band_matrix_is_in_the_same_basis_as_the_weil_matrix() {
    let p = 128;
    let mut cfg = HighPrecConfig::for_decimal_digits(30);
    cfg.precision_bits = p;
    cfg.quad_points = 128;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let params = CcmParams::from_lambda_sq_integer(13, 1);
    let matrix = band_concentration_matrix_hp(&params, &cfg, &Float::with_val(p, 2)).unwrap();
    // CCM V_n(exp x) = exp(2*pi*i*n*(x+L/2)/L)/sqrt(L).
    // Its transform is (-1)^n sqrt(L) sinc((L/2)*(2*pi*n/L-t)).
    // Thus C[0,1] = -a/pi int_-2^2 sinc(a*t) sinc(pi-a*t) dt.
    // Independent 70-decimal mpmath integral; no toolkit quadrature used.
    let expected = Float::with_val(
        p,
        Float::parse("-0.140800737241771996846508591679892641659625706722944489217141150020058")
            .unwrap(),
    );
    let entry = &matrix[params.idx(0) * params.matrix_size() + params.idx(1)];
    let error = Float::with_val(p, entry - &expected).abs();
    assert!(
        error < (Float::with_val(p, 1) >> 100u32),
        "C[0,1] disagrees with the CCM basis integral: got {entry}, expected {expected}, error {error}"
    );
}

#[test]
fn fresh_audit_sonin_composition_matches_the_common_basis_compression() {
    let p = 128;
    let mut cfg = HighPrecConfig::for_decimal_digits(30);
    cfg.precision_bits = p;
    cfg.quad_points = 128;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let params = CcmParams::from_lambda_sq_integer(13, 1);
    let result = weil_spectrum_sonin_hp(&params, &cfg, &Float::with_val(p, 2), 1).unwrap();
    assert_eq!(result.spectrum.len(), 2);
    // 65-decimal independent integral + symmetric spectral compression in
    // independent_oracles.py::band_basis_composition; the shared CCM V basis
    // is used for both C and pole-minus-archimedean A throughout.
    let expected = Float::with_val(
        p,
        Float::parse("-0.14536610615881212726190861941050413259375435131429").unwrap(),
    );
    let error = Float::with_val(p, &result.spectrum[1] - &expected).abs();
    assert!(
        error < (Float::with_val(p, 1) >> 85u32),
        "Sonin compression uses incompatible source bases: got {}, expected {expected}, error {error}",
        result.spectrum[1]
    );
}
