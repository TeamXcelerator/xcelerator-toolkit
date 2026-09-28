//! Exact inertia counts for the stored binary64 tridiagonal matrix.
//!
//! Every finite binary64 input is a dyadic rational. Multiplication of the
//! shifted matrix by one positive power of two makes all entries integers
//! without changing inertia. Leading principal determinants then give a
//! Sturm sign sequence with exact integer arithmetic and no pivot floor.
//! Exact zero off-diagonal entries split independent blocks. Inside an
//! irreducible block, adjacent determinants cannot both vanish; an interior
//! zero is skipped and a final zero is excluded, giving strict-below semantics.

use num_bigint::{BigInt, Sign};

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

/// The caller validates dimensions and finiteness before entering this helper.
pub(crate) fn count_below(diagonal: &[f64], off_diagonal: &[f64], threshold: f64) -> usize {
    let values: Vec<_> = diagonal
        .iter()
        .chain(off_diagonal)
        .copied()
        .chain([threshold])
        .map(dyadic_parts)
        .collect();
    let common_power = values.iter().map(|(_, power)| *power).min().unwrap_or(0);
    let integers: Vec<BigInt> = values
        .into_iter()
        .map(|(mantissa, power)| mantissa << ((power - common_power) as usize))
        .collect();
    let n = diagonal.len();
    let shift = &integers[2 * n - 1];
    let d: Vec<BigInt> = integers[..n].iter().map(|x| x - shift).collect();
    let e = &integers[n..2 * n - 1];
    let mut count = 0;
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && e[end - 1].sign() != Sign::NoSign {
            end += 1;
        }
        let mut previous = BigInt::from(1);
        let mut current = d[start].clone();
        let mut previous_nonzero_sign = Sign::Plus;
        let mut record = |value: &BigInt| {
            let sign = value.sign();
            if sign != Sign::NoSign {
                if sign != previous_nonzero_sign {
                    count += 1;
                }
                previous_nonzero_sign = sign;
            }
        };
        record(&current);
        for i in start + 1..end {
            let next = &d[i] * &current - (&e[i - 1] * &e[i - 1]) * &previous;
            record(&next);
            previous = current;
            current = next;
        }
        start = end;
    }
    count
}

/// Compare the exact dyadic endpoint difference with the stored tolerance.
/// A rounded subtraction can equal the tolerance while the exact width exceeds it.
pub(crate) fn bracket_width_at_most(lower: f64, upper: f64, tolerance: f64) -> bool {
    let parts = [lower, upper, tolerance].map(dyadic_parts);
    let common_power = parts.iter().map(|(_, p)| *p).min().unwrap();
    let integers = parts.map(|(m, p)| m << ((p - common_power) as usize));
    &integers[1] - &integers[0] <= integers[2]
}

/// Exact dyadic |a-b| <= tolerance*scale. All inputs are finite and the
/// right-hand factors nonnegative; no rounded product enters acceptance.
pub(crate) fn difference_at_most_scaled(a: f64, b: f64, tolerance: f64, scale: f64) -> bool {
    let [(am, ap), (bm, bp), (tm, tp), (sm, sp)] = [a, b, tolerance, scale].map(dyadic_parts);
    let power = ap.min(bp).min(tp + sp);
    let difference = (am << ((ap - power) as usize)) - (bm << ((bp - power) as usize));
    let magnitude = if difference.sign() == Sign::Minus {
        -difference
    } else {
        difference
    };
    magnitude <= ((tm * sm) << ((tp + sp - power) as usize))
}

/// Exact |a-b| <= absolute + relative*scale for finite nonnegative tolerances.
pub(crate) fn difference_at_most_sum_scaled(
    a: f64,
    b: f64,
    absolute: f64,
    relative: f64,
    scale: f64,
) -> bool {
    let [(am, ap), (bm, bp), (cm, cp), (rm, rp), (sm, sp)] =
        [a, b, absolute, relative, scale].map(dyadic_parts);
    let power = ap.min(bp).min(cp).min(rp + sp);
    let difference = (am << ((ap - power) as usize)) - (bm << ((bp - power) as usize));
    let magnitude = if difference.sign() == Sign::Minus {
        -difference
    } else {
        difference
    };
    magnitude <= (cm << ((cp - power) as usize)) + ((rm * sm) << ((rp + sp - power) as usize))
}
