//! Shared directed finite arithmetic; no source or continuum error bound.
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;
pub(in crate::ccm) fn decimal(s: &str, p: u32) -> Result<I> {
    Ok(I::new(
        Float::with_val_round(p, Float::parse(s)?, Round::Down).0,
        Float::with_val_round(p, Float::parse(s)?, Round::Up).0,
    )?)
}
/// Retain a declared nonnegative decimal upper bound without rounding inward.
pub(in crate::ccm) fn decimal_upper(s: &str, p: u32) -> Result<Float> {
    let value = decimal(s, p)?;
    if value.lower() < &0 || !value.upper().is_finite() {
        bail!("declared upper bound must be finite and nonnegative");
    }
    Ok(value.upper().clone())
}
/// Correctly rounded point log of the exact decimal cutoff. Directed endpoints
/// must round to one target point; a fixed guard alone cannot handle C near one.
pub(in crate::ccm) fn rounded_log_cutoff(cutoff: &str, requested: u32) -> Result<Float> {
    if !(32..=1_000_128).contains(&requested) {
        bail!("unsupported logarithm precision");
    }
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let l = decimal(cutoff, requested + guard)?.ln()?;
        if !l.is_strictly_positive() {
            continue;
        }
        let lo = Float::with_val(requested, l.lower());
        let hi = Float::with_val(requested, l.upper());
        if lo == hi && lo.is_finite() && lo > 0 {
            return Ok(lo);
        }
    }
    bail!("exact cutoff logarithm unresolved within 4096 guard bits")
}
pub(in crate::ccm) fn scale_float(x: &Float, e: i64, p: u32) -> Result<Float> {
    let mut out = Float::with_val(p, x);
    if x.is_zero() {
        return Ok(out);
    }
    let amount = u32::try_from(e.unsigned_abs())?;
    if e >= 0 {
        out <<= amount;
    } else {
        out >>= amount;
    }
    if !out.is_finite() || out.is_zero() {
        bail!("weighted profile exceeds exponent range");
    }
    let mut reverse = out.clone();
    if e >= 0 {
        reverse >>= amount;
    } else {
        reverse <<= amount;
    }
    if reverse != *x {
        bail!("weighted profile binary scaling loses a component");
    }
    Ok(out)
}
pub(in crate::ccm) fn scale(x: &I, e: i64) -> Result<I> {
    x.validate()?;
    Ok(I::new(
        scale_float(x.lower(), e, x.precision())?,
        scale_float(x.upper(), e, x.precision())?,
    )?)
}
pub(in crate::ccm) fn abs(x: &I) -> Result<I> {
    x.validate()?;
    let a = x.lower().clone().abs();
    let b = x.upper().clone().abs();
    Ok(I::new(
        if x.contains_zero() {
            Float::with_val(x.precision(), 0)
        } else {
            a.clone().min(&b)
        },
        a.max(&b),
    )?)
}
pub(in crate::ccm) fn exponent(xs: &[I]) -> Result<i64> {
    for x in xs {
        x.validate()?;
    }
    Ok(xs
        .iter()
        .flat_map(|x| [x.lower(), x.upper()])
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from))
}
pub(in crate::ccm) fn normalized(xs: &[I]) -> Result<(Vec<I>, i64)> {
    let e = exponent(xs)?;
    Ok((xs.iter().map(|x| scale(x, -e)).collect::<Result<_>>()?, e))
}
// Width <= 2^(-requested-16) times an explicitly chosen natural scale.
// Values themselves remain enclosures, including cancellation through zero.
pub(in crate::ccm) fn narrow(xs: &[I], natural: &Float, requested: u32) -> Result<bool> {
    for x in xs {
        x.validate()?;
        let width = Float::with_val_round(x.precision(), x.upper() - x.lower(), Round::Up).0;
        if width.is_zero() {
            continue;
        }
        if natural <= &0 {
            return Ok(false);
        }
        let ratio = Float::with_val_round(x.precision(), &width / natural, Round::Up).0;
        if !ratio.is_finite() || ratio > Float::with_val(x.precision(), 1) >> (requested + 16) {
            return Ok(false);
        }
    }
    Ok(true)
}
// Interval Gaussian elimination without pivoting. Positive diagonal pivots
// verify the finite symmetric Gram matrix is SPD; interval updates enclose
// its exact elimination and back substitution. No condition-number heuristic.
pub(in crate::ccm) fn solve(g: &[I], rhs: &[I], p: u32) -> Result<Option<(Vec<I>, I)>> {
    let n = rhs.len();
    let mut a = g.to_vec();
    let mut r = rhs.to_vec();
    let mut minimum: Option<I> = None;
    for k in 0..n {
        let pivot = a[k * n + k].clone();
        pivot.validate()?;
        if !pivot.is_strictly_positive() {
            return Ok(None);
        }
        minimum = Some(if let Some(old) = minimum {
            I::new(
                old.lower().clone().min(pivot.lower()),
                old.upper().clone().min(pivot.upper()),
            )?
        } else {
            pivot.clone()
        });
        for i in k + 1..n {
            let factor = a[i * n + k].div(&pivot)?;
            for j in k + 1..n {
                a[i * n + j] = a[i * n + j].sub(&factor.mul(&a[k * n + j]));
            }
            r[i] = r[i].sub(&factor.mul(&r[k]));
        }
    }
    let mut x = vec![I::from_i64(0, p); n];
    for i in (0..n).rev() {
        let mut value = r[i].clone();
        for j in i + 1..n {
            value = value.sub(&a[i * n + j].mul(&x[j]));
        }
        x[i] = value.div(&a[i * n + i])?;
    }
    Ok(Some((x, minimum.unwrap())))
}
