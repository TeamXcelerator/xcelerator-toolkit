//! Guarded point arithmetic. These checks are not forward-error enclosures.
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::cmp::Ordering;

pub(in crate::ccm) fn output(v: &Float, p: u32) -> Result<Float> {
    let out = Float::with_val(p, v);
    if !out.is_finite() || (!v.is_zero() && out.is_zero()) {
        bail!("retained point output exceeds MPFR range");
    }
    Ok(out)
}
pub(super) fn scale(v: &Float, exponent: i64, p: u32) -> Result<Float> {
    let mut out = output(v, p)?;
    if v.is_zero() {
        return Ok(out);
    }
    let amount = u32::try_from(exponent.unsigned_abs())?;
    if exponent >= 0 {
        out <<= amount;
    } else {
        out >>= amount;
    }
    if !out.is_finite() || out.is_zero() {
        bail!("retained binary scale exceeds MPFR range");
    }
    let mut reverse = out.clone();
    if exponent >= 0 {
        reverse >>= amount;
    } else {
        reverse <<= amount;
    }
    if reverse != Float::with_val(p, v) {
        bail!("retained binary scale loses a component");
    }
    Ok(out)
}
pub(in crate::ccm) fn sum(v: &[Float], p: u32) -> Result<Float> {
    let (out, dir) = Float::with_val_round(p, Float::sum(v.iter()), Round::Nearest);
    if !out.is_finite()
        || (out.is_zero() && dir != Ordering::Equal)
        || (out.get_exp() == Some(rug::float::exp_min())
            && Float::with_val_round(p, Float::sum(v.iter()), Round::Zero)
                .0
                .is_zero())
    {
        bail!("retained point sum exceeds MPFR range");
    }
    Ok(out)
}
pub(in crate::ccm) fn product(factors: &[&Float], p: u32) -> Result<Float> {
    if factors.iter().any(|x| !x.is_finite()) {
        bail!("nonfinite retained factor");
    }
    if factors.iter().any(|x| x.is_zero()) {
        return Ok(Float::with_val(p, 0));
    }
    let mut mantissa = Float::with_val(p, 1);
    let mut exponent = 0i64;
    for x in factors {
        let e = i64::from(x.get_exp().unwrap());
        mantissa *= scale(x, -e, p)?;
        exponent = exponent
            .checked_add(e)
            .ok_or_else(|| anyhow::anyhow!("product exponent overflow"))?;
    }
    scale(&mantissa, exponent, p)
}
pub(in crate::ccm) fn quotient(a: &Float, b: &Float, p: u32) -> Result<Float> {
    let (out, dir) = Float::with_val_round(p, a / b, Round::Nearest);
    if !out.is_finite()
        || (out.is_zero() && dir != Ordering::Equal)
        || (out.get_exp() == Some(rug::float::exp_min())
            && Float::with_val_round(p, a / b, Round::Zero).0.is_zero())
    {
        bail!("retained quotient exceeds MPFR range");
    }
    Ok(out)
}
pub(super) fn norm(v: &[Float], p: u32) -> Result<Float> {
    let mut out = Float::with_val(p, 0);
    for x in v {
        if !x.is_finite() {
            bail!("nonfinite retained vector");
        }
        out.hypot_mut(x);
    }
    output(&out, p)
}
pub(in crate::ccm) fn dot(a: &[Float], b: &[Float], p: u32) -> Result<Float> {
    if a.len() != b.len() {
        bail!("retained dot-product shape mismatch");
    }
    let terms = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            product(
                &[x, y],
                x.prec()
                    .checked_add(y.prec())
                    .ok_or_else(|| anyhow::anyhow!("product precision overflow"))?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    sum(&terms, p)
}
pub(in crate::ccm) fn unit(v: &[Float], p: u32) -> Result<Vec<Float>> {
    let exponent = v
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero retained vector"))?;
    let scaled = v
        .iter()
        .map(|x| scale(x, -i64::from(exponent), p))
        .collect::<Result<Vec<_>>>()?;
    let norm = norm(&scaled, p)?;
    scaled.iter().map(|x| quotient(x, &norm, p)).collect()
}
pub(super) fn center_terms(v: &[Float]) -> Vec<Float> {
    let n = v.len() / 2;
    v.iter()
        .enumerate()
        .map(|(i, x)| {
            if i.abs_diff(n).is_multiple_of(2) {
                x.clone()
            } else {
                -x.clone()
            }
        })
        .collect()
}
/// Sign of the exact stored center, including an underflowed nonzero sum.
/// The largest-component convention applies only when that exact sum is zero.
pub(in crate::ccm) fn orientation(v: &[Float], p: u32) -> i32 {
    let terms = center_terms(v);
    let (c, dir) = Float::with_val_round(p, Float::sum(terms.iter()), Round::Nearest);
    if c < 0 || (c.is_zero() && dir == Ordering::Greater) {
        return -1;
    }
    if c > 0 || (c.is_zero() && dir == Ordering::Less) {
        return 1;
    }
    let largest = v
        .iter()
        .max_by(|a, b| (*a).clone().abs().total_cmp(&(*b).clone().abs()));
    if largest.is_some_and(|x| x < &0) {
        -1
    } else {
        1
    }
}

/// Absolute exact-stored center divided by the Euclidean coefficient norm.
/// Sum before coefficient-wise normalization, so a small center does not inherit
/// cancellation between independently rounded unit coefficients. Point output
/// only; callers must supply guard precision and account for their L2(dx) scale.
pub(in crate::ccm) fn unit_center(v: &[Float], p: u32) -> Result<Float> {
    if v.is_empty()
        || v.len().is_multiple_of(2)
        || v.len() > 16385
        || !(64..=1_000_128).contains(&p)
        || v.iter().any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid stored Fourier center input");
    }
    let exponent = v
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero retained vector"))?;
    let scaled = v
        .iter()
        .map(|x| scale(x, -i64::from(exponent), p))
        .collect::<Result<Vec<_>>>()?;
    let center = sum(&center_terms(&scaled), p)?.abs();
    quotient(&center, &norm(&scaled, p)?, p)
}

#[cfg(test)]
mod exhaustive_resumed_range_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_point_sum_rejects_partial_underflow() {
        let p = 128;
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        assert!(sum(&[tiny.clone() * 1.75, -tiny.clone()], p).is_err());
        assert_eq!(sum(&[tiny.clone() * 2u32, -tiny.clone()], p).unwrap(), tiny);
    }
    #[test]
    fn exhaustive_resumed_point_quotient_rejects_partial_underflow() {
        let p = 128;
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        assert!(quotient(&tiny, &Float::with_val(p, 1.5), p).is_err());
        assert_eq!(quotient(&tiny, &Float::with_val(p, 1), p).unwrap(), tiny);
    }
}
