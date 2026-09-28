#![cfg(feature = "hp")]
use rug::{float::Special, Float};
use xc_numerics::eigen::{tridiag_eigenvalues_hp, tridiag_sturm_count_below_hp};

#[test]
fn qr_rejects_invalid_domain_and_uses_requested_precision() {
    let p = 128;
    for bad in [Special::Nan, Special::Infinity, Special::NegInfinity] {
        assert!(tridiag_eigenvalues_hp(&[Float::with_val(p, bad)], &[], p).is_err());
    }
    for precision in [0, 1, 32, u32::MAX] {
        assert!(tridiag_eigenvalues_hp(&[Float::with_val(p, 1)], &[], precision).is_err());
    }
    assert!(tridiag_eigenvalues_hp(&[], &[Float::with_val(p, 1)], p).is_err());
    assert!(tridiag_eigenvalues_hp(&[], &[], p).unwrap().is_empty());
    let actual = tridiag_eigenvalues_hp(&[Float::with_val(64, 3)], &[], p).unwrap();
    assert_eq!(actual[0], 3);
    assert_eq!(actual[0].prec(), p);
}

#[test]
fn qr_and_sturm_do_not_accept_exponent_overflow_as_a_result() {
    let p = 128;
    let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 1) as u32;
    assert!(tridiag_eigenvalues_hp(
        &[-huge.clone(), huge.clone()],
        std::slice::from_ref(&huge),
        p
    )
    .is_err());
    assert!(tridiag_sturm_count_below_hp(
        &[huge.clone(), huge.clone(), huge.clone()],
        &[huge.clone(), huge],
        &Float::with_val(p, 0),
        p
    )
    .is_err());
    let tiny = Float::with_val(p, 1) >> (-(rug::float::exp_min() + 10)) as u32;
    assert!(!tiny.is_zero());
    assert!(tridiag_sturm_count_below_hp(
        &[tiny.clone(), tiny.clone(), tiny.clone()],
        &[tiny.clone(), tiny],
        &Float::with_val(p, 0),
        p
    )
    .is_err());
}

#[test]
fn wide_but_representable_scales_keep_the_correct_count_and_qr_values() {
    let p = 256;
    for exponent in [-20_000_i32, 0, 20_000] {
        let scale = Float::with_val(p, 1) << exponent;
        let d = vec![-scale.clone(), scale.clone()];
        let e = vec![scale.clone()];
        assert_eq!(
            tridiag_sturm_count_below_hp(&d, &e, &Float::with_val(p, 0), p).unwrap(),
            1
        );
        let values = tridiag_eigenvalues_hp(&d, &e, p).unwrap();
        let expected = Float::with_val(p, 2).sqrt() * scale;
        for (actual, sign) in values.iter().zip([-1_i32, 1]) {
            let relative = (Float::with_val(p, actual * sign) / &expected - 1_i32).abs();
            assert!(relative < Float::with_val(p, 1) >> 220);
        }
    }
}

#[test]
fn interval_sturm_resolves_a_negative_determinant_lost_to_finite_rounding() {
    let p = 64;
    let off: Float = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 63_i32);
    let second: Float = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 62_i32);
    // The exact stored determinant is (1+2^-62)-(1+2^-63)^2=-2^-126.
    // The former point recurrence rounded the square to 1+2^-62 and returned 0.
    assert_eq!(
        tridiag_sturm_count_below_hp(
            &[Float::with_val(p, 1), second],
            &[off],
            &Float::with_val(p, 0),
            p
        )
        .unwrap(),
        1
    );
}
