#![cfg(feature = "hp")]
use rug::{Float, Rational};
use xc_numerics::fmt::{matching_digits, relative_difference};

#[test]
fn lower_precision_reference_does_not_lower_quotient_precision() {
    let a = Float::with_val(256, 1);
    let b = Float::with_val(2, 3);
    let expected = Float::with_val(256, Rational::from((2, 3)));
    assert_eq!(relative_difference(&a, &b).unwrap(), expected);
    // Logarithm of the exact rational quotient, evaluated independently at guard precision.
    let digits = Float::with_val(256, -Float::with_val(1024, Rational::from((2, 3))).log10());
    assert_eq!(matching_digits(&a, &b), digits);
    let b = Float::with_val(2, -3);
    let expected = Float::with_val(256, Rational::from((4, 3)));
    assert_eq!(relative_difference(&a, &b).unwrap(), expected);
}
