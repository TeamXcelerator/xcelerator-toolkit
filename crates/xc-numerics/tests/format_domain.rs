#![cfg(feature = "hp")]
use rug::{float::Special, Float};
use xc_numerics::fmt::{matching_digits, relative_difference, sign_of};
#[test]
fn invalid_numbers_have_no_positive_mathematical_sign() {
    assert_eq!(
        sign_of(&Float::with_val(128, Special::Nan)).as_str(),
        "nonfinite"
    );
}
#[test]
fn matching_diagnostics_do_not_overflow_a_finite_relative_difference() {
    let p = 128;
    let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 1);
    let actual = relative_difference(&huge, &-huge.clone()).unwrap();
    assert_eq!(actual, 2);
    let digits = matching_digits(&huge, &-huge.clone());
    assert!(digits.is_finite());
    assert!(
        Float::with_val(p, digits + Float::with_val(p, 2).log10()).abs()
            < (Float::with_val(p, 1) >> 90_u32)
    );
}
#[test]
fn nonzero_difference_below_the_mpfr_exponent_floor_is_not_exact_agreement() {
    let p = 128;
    let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
    let mut adjacent = tiny.clone();
    adjacent.next_up();
    let difference = relative_difference(&adjacent, &tiny).unwrap();
    assert!(difference > 0);
    assert!(matching_digits(&adjacent, &tiny).is_finite());
}
