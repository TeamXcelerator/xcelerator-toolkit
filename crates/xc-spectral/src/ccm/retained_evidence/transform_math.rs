//! Directed finite Fourier transform, including exact decimal cutoff construction.
//! Enclosures exclude source-state construction and infinite-tail errors.
use super::{
    finite_math::{abs, decimal, narrow, scale, scale_float},
    point, precision,
};
use crate::ccm::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;
pub(in crate::ccm) struct Measurement {
    pub value: I,
    pub derivative: I,
    pub absolute_terms: I,
    pub absolute_derivative_terms: I,
    pub curvature: I,
    pub precision: u32,
}
fn widen(value: &I, error: &I) -> Result<I> {
    Ok(value.add(&I::new(-error.upper().clone(), error.upper().clone())?))
}
// sinc(q)=integral_0^1 cos(qs) ds and sinc'(q)=-integral_0^1 s*sin(qs) ds.
// Taylor-Lagrange remainders after k terms are bounded by the next
// term magnitudes: |q|^(2k+2)/(2k+3)! and
// (2k+2)*|q|^(2k+1)/(2k+3)!, respectively. Directed intervals enclose
// those magnitudes for every q in the input; widen before returning.
fn sinc(q: &I) -> Result<Option<(I, I)>> {
    let p = q.precision();
    let magnitude = abs(q)?;
    if magnitude.upper() > &Float::with_val(p, 0.5) {
        return Ok(None);
    }
    if magnitude.upper().is_zero() {
        return Ok(Some((I::from_i64(1, p), I::from_i64(0, p))));
    }
    let square = q.square().neg();
    let mut term = I::from_i64(1, p);
    let mut value = term.clone();
    let mut derivative = q.neg().div(&I::from_i64(3, p))?;
    let mut derivative_term = derivative.clone();
    // Include value k=1 to match the derivative through k=1.
    term = term.mul(&square).div(&I::from_i64(6, p))?;
    value = value.add(&term);
    let tolerance = Float::with_val(p, 1) >> (p + 16);
    for k in 1..=p {
        let k = u64::from(k);
        let next = term
            .mul(&square)
            .div(&I::from_u64((2 * k + 2) * (2 * k + 3), p))?;
        let next_derivative = derivative_term
            .mul(&square)
            .div(&I::from_u64(2 * k * (2 * k + 3), p))?;
        let error = abs(&next)?;
        let derivative_error = abs(&next_derivative)?;
        let relative =
            Float::with_val_round(p, derivative_error.upper() / magnitude.upper(), Round::Up).0;
        if error.upper() <= &tolerance && relative <= tolerance {
            return Ok(Some((
                widen(&value, &error)?,
                widen(&derivative, &derivative_error)?,
            )));
        }
        value = value.add(&next);
        derivative = derivative.add(&next_derivative);
        term = next;
        derivative_term = next_derivative;
    }
    Ok(None)
}
fn sufficiently_narrow(value: &I, requested: u32) -> Result<bool> {
    value.validate()?;
    let natural = if value.contains_zero() {
        Float::with_val(value.precision(), 1)
    } else {
        value
            .lower()
            .clone()
            .abs()
            .max(&value.upper().clone().abs())
    };
    narrow(std::slice::from_ref(value), &natural, requested)
}
fn curvature_from_log(l: &I) -> Result<I> {
    // L^(5/2)/sqrt(80), with an even binary exponent removed before powers.
    let e = i64::from(
        l.upper()
            .get_exp()
            .ok_or_else(|| anyhow::anyhow!("nonpositive Fourier support"))?,
    )
    .div_euclid(2)
        * 2;
    let reduced = scale(l, -e)?;
    let value = reduced
        .square()
        .mul(&reduced.sqrt()?)
        .div(&I::from_i64(80, l.precision()).sqrt()?)?;
    let value = scale(&value, 5 * (e / 2))?;
    if !value.is_strictly_positive() {
        bail!("finite curvature exceeds supported exponent range");
    }
    Ok(value)
}
pub(in crate::ccm) fn curvature(cutoff: &str, requested: u32) -> Result<Option<I>> {
    precision(requested)?;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let l = decimal(cutoff, requested + guard)?.ln()?;
        if !l.is_strictly_positive() {
            continue;
        }
        let k = curvature_from_log(&l)?;
        if sufficiently_narrow(&k, requested)? {
            return Ok(Some(k));
        }
    }
    Ok(None)
}
fn calculate(
    state: &RetainedState,
    t: &Float,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let l = decimal(&state.cutoff, p)?.ln()?;
    if !l.is_strictly_positive() {
        return Ok(None);
    }
    let half = l.div(&I::from_i64(2, p))?;
    let e = i64::from(
        state
            .coefficients
            .iter()
            .filter_map(Float::get_exp)
            .max()
            .ok_or_else(|| anyhow::anyhow!("zero finite transform source"))?,
    );
    let x = state
        .coefficients
        .iter()
        .map(|v| Ok(I::from_float(&scale_float(v, -e, p)?, p)?))
        .collect::<Result<Vec<_>>>()?;
    let q = x.iter().fold(I::from_i64(0, p), |a, x| a.add(&x.square()));
    let factor = l.sqrt()?.div(&q.sqrt()?)?.mul(&I::from_i64(
        i64::from(point::orientation(&state.coefficients, p)),
        p,
    ));
    let phase = I::from_float(t, p)?.mul(&half);
    let sin = phase.sin();
    let cos = phase.cos();
    let pi = I::pi(p);
    let mut value = I::from_i64(0, p);
    let mut derivative = value.clone();
    let mut absolute = value.clone();
    let mut absolute_derivative = value.clone();
    for (idx, (xi, original)) in x.iter().zip(&state.coefficients).enumerate() {
        if original.is_zero() {
            continue;
        }
        let j = i64::try_from(idx)? - i64::try_from(state.modes)?;
        let q = phase.add(&pi.mul(&I::from_i64(j, p)));
        let (a, b) = if abs(&q)?.upper() <= &Float::with_val(p, 0.5) {
            let Some((a, b)) = sinc(&q)? else {
                return Ok(None);
            };
            if j.unsigned_abs().is_multiple_of(2) {
                (a, b)
            } else {
                (a.neg(), b.neg())
            }
        } else {
            if q.contains_zero() {
                return Ok(None);
            }
            let a = sin.div(&q)?;
            let b = cos.sub(&a).div(&q)?;
            (a, b)
        };
        let a = xi.mul(&factor).mul(&a);
        let b = xi.mul(&factor).mul(&half).mul(&b);
        value = value.add(&a);
        derivative = derivative.add(&b);
        absolute = absolute.add(&abs(&a)?);
        absolute_derivative = absolute_derivative.add(&abs(&b)?);
    }
    if t.is_zero()
        && state
            .coefficients
            .iter()
            .eq(state.coefficients.iter().rev())
    {
        // The finite function is exactly even, so its first moment vanishes.
        derivative = I::from_i64(0, p);
    }
    let curvature = curvature_from_log(&l)?;
    for v in [
        &value,
        &derivative,
        &absolute,
        &absolute_derivative,
        &curvature,
    ] {
        if !sufficiently_narrow(v, requested)? {
            return Ok(None);
        }
    }
    Ok(Some(Measurement {
        value,
        derivative,
        absolute_terms: absolute,
        absolute_derivative_terms: absolute_derivative,
        curvature,
        precision: p,
    }))
}
// Coordinated callers use the same bounded guard for both transforms and
// their derived expressions; this does not increase the admitted guard cap.
pub(in crate::ccm) fn measure_at(
    state: &RetainedState,
    t: &Float,
    requested: u32,
    guard: u32,
) -> Result<Option<Measurement>> {
    precision(requested)?;
    if ![64, 128, 256, 512, 1024, 2048, 4096].contains(&guard)
        || requested < state.precision.max(t.prec())
        || !t.is_finite()
    {
        bail!("invalid transform guard or source precision");
    }
    calculate(state, t, requested, requested + guard)
}
pub(in crate::ccm) fn measure(
    state: &RetainedState,
    t: &Float,
    requested: u32,
) -> Result<Option<Measurement>> {
    precision(requested)?;
    if requested < state.precision.max(t.prec()) || !t.is_finite() {
        bail!("finite transform requires finite stored points without precision reduction");
    }
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(value) = measure_at(state, t, requested, guard)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// Production secular roots use exp(-i*t*x). Evaluate the existing
/// plus-convention kernel at -t and reverse its derivative.
pub(in crate::ccm) fn measure_at_root(
    state: &RetainedState,
    t: &Float,
    requested: u32,
    guard: u32,
) -> Result<Option<Measurement>> {
    Ok(
        measure_at(state, &(-t.clone()), requested, guard)?.map(|mut m| {
            m.derivative = m.derivative.neg();
            m
        }),
    )
}
pub(in crate::ccm) fn measure_root(
    state: &RetainedState,
    t: &Float,
    requested: u32,
) -> Result<Option<Measurement>> {
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(m) = measure_at_root(state, t, requested, guard)? {
            return Ok(Some(m));
        }
    }
    Ok(None)
}
