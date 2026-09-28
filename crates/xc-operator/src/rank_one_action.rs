//! Exact dyadic accumulation for the binary64 rank-one action.
//!
//! Each finite input is an integer times a power of two. Accumulate the dot
//! product exactly, then form each complete `y_i + alpha*v_i*dot` exactly and
//! round once to binary64 (nearest, ties to even). Small terms are retained even
//! when large terms cancel. Storage is O(log(n) + binary64 exponent span).
use crate::OperatorError;
use std::cmp::Ordering;

#[derive(Clone, Default)]
struct Magnitude(Vec<u64>);
impl Magnitude {
    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }
    fn from_u64(v: u64) -> Self {
        if v == 0 {
            Self::default()
        } else {
            Self(vec![v])
        }
    }
    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&rhs.0.len())
            .then_with(|| self.0.iter().rev().cmp(rhs.0.iter().rev()))
    }
    fn add(&mut self, rhs: &Self) {
        self.0.resize(self.0.len().max(rhs.0.len()), 0);
        let mut carry = 0u128;
        for (i, left) in self.0.iter_mut().enumerate() {
            let sum = u128::from(*left) + u128::from(rhs.0.get(i).copied().unwrap_or(0)) + carry;
            *left = sum as u64;
            carry = sum >> 64;
        }
        if carry != 0 {
            self.0.push(carry as u64);
        }
    }
    // Requires self >= rhs.
    fn sub(&mut self, rhs: &Self) {
        let mut borrow = false;
        for (i, left) in self.0.iter_mut().enumerate() {
            let (first, b1) = left.overflowing_sub(rhs.0.get(i).copied().unwrap_or(0));
            let (next, b2) = first.overflowing_sub(u64::from(borrow));
            *left = next;
            borrow = b1 || b2;
        }
        debug_assert!(!borrow);
        self.trim();
    }
    fn mul(&mut self, rhs: u64) {
        if rhs == 0 {
            self.0.clear();
            return;
        }
        let mut carry = 0u128;
        for left in &mut self.0 {
            let product = u128::from(*left) * u128::from(rhs) + carry;
            *left = product as u64;
            carry = product >> 64;
        }
        if carry != 0 {
            self.0.push(carry as u64);
        }
    }
    fn shift(&mut self, bits: usize) {
        if self.is_zero() {
            return;
        }
        let words = bits / 64;
        let remainder = bits % 64;
        if remainder != 0 {
            let mut carry = 0;
            for value in &mut self.0 {
                let next = *value >> (64 - remainder);
                *value = (*value << remainder) | carry;
                carry = next;
            }
            if carry != 0 {
                self.0.push(carry);
            }
        }
        if words != 0 {
            self.0.splice(0..0, std::iter::repeat_n(0, words));
        }
    }
    fn bits(&self) -> usize {
        self.0.last().map_or(0, |last| {
            64 * (self.0.len() - 1) + (64 - last.leading_zeros()) as usize
        })
    }
    fn bit(&self, bit: usize) -> bool {
        self.0
            .get(bit / 64)
            .is_some_and(|x| x & (1u64 << (bit % 64)) != 0)
    }
    fn any_below(&self, bits: usize) -> bool {
        let words = bits / 64;
        let tail = bits % 64;
        self.0.iter().take(words).any(|x| *x != 0)
            || (tail != 0
                && self
                    .0
                    .get(words)
                    .is_some_and(|x| x & ((1u64 << tail) - 1) != 0))
    }
    fn rounded(&self, exponent: i32, negative: bool) -> f64 {
        let bits = self.bits();
        if bits == 0 {
            return 0.;
        }
        let mut leading = exponent + bits as i32 - 1;
        let unit = (leading - 52).max(-1074);
        let shift = unit - exponent;
        let mut q = 0u64;
        if shift >= 0 {
            let shift = shift as usize;
            for i in shift..bits {
                q |= u64::from(self.bit(i)) << (i - shift);
            }
            if shift > 0 && self.bit(shift - 1) && (self.any_below(shift - 1) || q & 1 != 0) {
                q += 1;
            }
        } else {
            // A short exact integer: all significant bits fit the destination.
            q = self.0[0] << (-shift as usize);
        }
        let sign = u64::from(negative) << 63;
        if leading < -1022 {
            return f64::from_bits(sign | q);
        }
        if q == 1u64 << 53 {
            q >>= 1;
            leading += 1;
        }
        if leading > 1023 {
            return f64::INFINITY.copysign(if negative { -1. } else { 1. });
        }
        f64::from_bits(sign | (((leading + 1023) as u64) << 52) | (q & ((1u64 << 52) - 1)))
    }
}

// x = (-1)^sign * integer * 2^exponent, including subnormal inputs.
fn parts(x: f64) -> (bool, u64, i32) {
    let bits = x.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    (
        bits >> 63 != 0,
        if field == 0 {
            fraction
        } else {
            fraction | (1u64 << 52)
        },
        if field == 0 { -1074 } else { field - 1075 },
    )
}
fn difference(mut positive: Magnitude, mut negative: Magnitude) -> (Magnitude, bool) {
    if positive.cmp(&negative) == Ordering::Less {
        negative.sub(&positive);
        (negative, true)
    } else {
        positive.sub(&negative);
        (positive, false)
    }
}

pub(super) fn add(alpha: f64, v: &[f64], x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
    if !alpha.is_finite() || v.iter().chain(x).chain(y.iter()).any(|z| !z.is_finite()) {
        return Err(OperatorError::ApplicationFailed(
            "rank-one action requires finite inputs and base output".into(),
        ));
    }
    if alpha == 0. {
        return Ok(());
    }
    let mut positive = Magnitude::default();
    let mut negative = Magnitude::default();
    for (&vi, &xi) in v.iter().zip(x) {
        let (vs, vm, ve) = parts(vi);
        let (xs, xm, xe) = parts(xi);
        let mut term = Magnitude::from_u64(vm);
        term.mul(xm);
        term.shift((ve + xe + 2148) as usize);
        if vs == xs {
            positive.add(&term);
        } else {
            negative.add(&term);
        }
    }
    let (dot, ds) = difference(positive, negative);
    if dot.is_zero() {
        return Ok(());
    }
    let (asign, am, ae) = parts(alpha);
    for (yi, &vi) in y.iter_mut().zip(v) {
        let (vs, vm, ve) = parts(vi);
        if vm == 0 {
            continue;
        }
        let mut term = dot.clone();
        term.mul(am);
        term.mul(vm);
        let exponent = -2148 + ae + ve;
        let (ys, ym, ye) = parts(*yi);
        let common = exponent.min(ye);
        term.shift((exponent - common) as usize);
        let mut initial = Magnitude::from_u64(ym);
        initial.shift((ye - common) as usize);
        let negative = ds ^ asign ^ vs;
        let (sum, sign) = if negative == ys {
            term.add(&initial);
            (term, negative)
        } else if negative {
            difference(initial, term)
        } else {
            difference(term, initial)
        };
        let value = sum.rounded(common, sign);
        if !value.is_finite() {
            return Err(OperatorError::ApplicationFailed(
                "rank-one result exceeds finite binary64 range".into(),
            ));
        }
        *yi = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_dynamic_range_and_cancellation_are_retained() {
        for (v, x, expected) in [
            (vec![1., 1e-310], vec![1., 1.], vec![1., 1e-310]),
            (vec![1., 1.], vec![1e300, 1e-10], vec![1e300, 1e300]),
            (vec![1., 1e-160], vec![1., 1e-160], vec![1., 1e-160]),
            (vec![1., 1., 1.], vec![1e300, 1., -1e300], vec![1., 1., 1.]),
        ] {
            let mut y = vec![0.; v.len()];
            add(1., &v, &x, &mut y).unwrap();
            assert_eq!(y, expected);
        }
        // The update itself overflows, but cancellation with y leaves a finite value.
        let mut y = [-f64::MAX];
        add(2., &[1.], &[f64::MAX], &mut y).unwrap();
        assert_eq!(y, [f64::MAX]);
    }
    #[test]
    fn final_rounding_is_ties_to_even_including_subnormals() {
        let tiny = f64::from_bits(1);
        let mut y = [0.];
        add(0.5, &[1.], &[tiny], &mut y).unwrap();
        assert_eq!(y, [0.]);
        let mut y = [tiny];
        add(0.5, &[1.], &[tiny], &mut y).unwrap();
        assert_eq!(y, [f64::from_bits(2)]);
        let mut y = [1.];
        add(1., &[1.], &[2f64.powi(-53)], &mut y).unwrap();
        assert_eq!(y, [1.]);
    }
}
