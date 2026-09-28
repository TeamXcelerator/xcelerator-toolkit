#![cfg(feature = "hp")]

use rug::Float;
use xc_variational::maynard::MkSymmetricReference;

#[test]
fn rounded_nonzero_floor_products_are_rejected_in_both_forms() {
    let p = 96;
    let reference = MkSymmetricReference::new(3, 0).unwrap();
    let mut smallest = Float::with_val(p, 1);
    smallest <<= rug::float::exp_min() - 1;
    // Exact forms are I=[1/6], J=[1/4]. Products are respectively
    // (2/3)*smallest and (3/4)*smallest, which round to nonzero smallest.
    // Dividing by smallest gives this exact rational oracle without creating
    // a rational whose denominator has billions of bits.
    for sign in [-1, 1] {
        for (metric, multiple) in [(true, 4), (false, 3)] {
            let input = Float::with_val(p, &smallest * (sign * multiple));
            let mut out = vec![Float::with_val(p, 0)];
            let result = if metric {
                reference.apply_i_hp(&[input], &mut out, p)
            } else {
                reference.apply_j_total_hp(&[input], &mut out, p)
            };
            assert!(result.is_err(), "metric={metric}, sign={sign}");
        }
    }
}

#[test]
fn exact_floor_values_and_zero_actions_remain_admissible() {
    let p = 96;
    let reference = MkSymmetricReference::new(3, 0).unwrap();
    let mut smallest = Float::with_val(p, 1);
    smallest <<= rug::float::exp_min() - 1;
    let mut out = vec![Float::with_val(p, 0)];
    // J=[1/4] is exactly representable: no rounding loss at the floor.
    reference
        .apply_j_total_hp(&[Float::with_val(p, &smallest * 4)], &mut out, p)
        .unwrap();
    assert_eq!(out[0], smallest);
    reference
        .apply_i_hp(&[Float::with_val(p, 0)], &mut out, p)
        .unwrap();
    assert!(out[0].is_zero());
}
