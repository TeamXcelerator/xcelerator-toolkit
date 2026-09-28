//! Exact comparisons of stored binary64 bracket widths and stopping thresholds.
use num_bigint::BigInt;

// IEEE binary64 is m * 2^e, including the subnormal range. No conversion
// through decimal or rounded subtraction/multiplication enters acceptance.
fn dyadic_parts(value: f64) -> (BigInt, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mantissa, power) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), exponent - 1023 - 52)
    };
    if mantissa == 0 {
        return (BigInt::from(0), 0);
    }
    let signed = if bits >> 63 == 0 {
        BigInt::from(mantissa)
    } else {
        -BigInt::from(mantissa)
    };
    (signed, power)
}

/// Exact `upper - lower <= tolerance * scale`. Handles widths/products that
/// overflow binary64. Invalid inputs fail closed even outside solver callers.
pub(super) fn width_at_most_product(lower: f64, upper: f64, tolerance: f64, scale: f64) -> bool {
    if [lower, upper, tolerance, scale]
        .iter()
        .any(|x| !x.is_finite())
        || lower > upper
        || tolerance < 0.0
        || scale < 0.0
    {
        return false;
    }
    let [(lm, lp), (um, up), (tm, tp), (sm, sp)] =
        [lower, upper, tolerance, scale].map(dyadic_parts);
    let power = lp.min(up).min(tp + sp);
    let width = (um << ((up - power) as usize)) - (lm << ((lp - power) as usize));
    width <= ((tm * sm) << ((tp + sp - power) as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_comparisons_cover_subnormal_and_overflow_boundaries() {
        let tiny = f64::from_bits(1);
        assert!(width_at_most_product(0.0, tiny, tiny, 1.0));
        assert!(!width_at_most_product(-tiny, tiny, tiny, 1.0));
        assert!(width_at_most_product(-tiny, tiny, tiny, 2.0));
        assert!(!width_at_most_product(0.0, tiny, tiny, 0.5));
        assert!(width_at_most_product(-f64::MAX, f64::MAX, f64::MAX, 2.0));
        assert!(!width_at_most_product(
            -f64::MAX,
            f64::MAX,
            f64::MAX,
            2.0f64.next_down()
        ));
        assert!(!width_at_most_product(0.0, 1.0, f64::NAN, 1.0));
    }
    #[cfg(feature = "hp")]
    #[test]
    fn integer_comparison_matches_independent_mpfr_rationals() {
        use rug::Float;
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut draw = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f64::from_bits(state & 0xffefffffffffffff)
        };
        for _ in 0..12000 {
            let a = draw();
            let b = draw();
            let (lo, hi) = (a.min(b), a.max(b));
            let tolerance = draw().abs();
            let scale = draw().abs();
            let rational = |x| Float::with_val(53, x).to_rational().unwrap();
            let expected = rational(hi) - rational(lo) <= rational(tolerance) * rational(scale);
            assert_eq!(width_at_most_product(lo, hi, tolerance, scale), expected);
        }
    }
}
