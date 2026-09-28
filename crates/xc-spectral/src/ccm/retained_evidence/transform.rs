//! Point adapter for the directed finite Fourier transform kernel.
use super::{transform_math, RetainedState};
use anyhow::{bail, Result};
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;
fn point(value: &I, p: u32) -> Result<Float> {
    value.validate()?;
    let point = Float::with_val(p, value.midpoint_point().lower());
    if !point.is_finite()
        || (point.is_zero()
            && value.lower() != value.upper()
            && (value.lower() >= &0 || value.upper() <= &0))
    {
        bail!("finite transform point exceeds supported exponent range");
    }
    Ok(point)
}
/// Exact decimal cutoff, original stored coefficient and ordinate points.
/// The returned points approximate finite arithmetic enclosures; source-state
/// construction and infinite-tail errors remain outside this calculation.
pub(super) fn terms(
    state: &RetainedState,
    t: &Float,
    p: u32,
) -> Result<(Float, Float, Float, Float)> {
    let Some(m) = transform_math::measure(state, t, p)? else {
        bail!("finite transform unresolved within 4096 guard bits");
    };
    Ok((
        point(&m.value, p)?,
        point(&m.derivative, p)?,
        point(&m.absolute_terms, p)?,
        point(&m.absolute_derivative_terms, p)?,
    ))
}

pub(super) fn terms_at_root(
    state: &RetainedState,
    t: &Float,
    p: u32,
) -> Result<(Float, Float, Float, Float)> {
    let Some(m) = transform_math::measure_root(state, t, p)? else {
        bail!("finite root transform unresolved within 4096 guard bits");
    };
    Ok((
        point(&m.value, p)?,
        point(&m.derivative, p)?,
        point(&m.absolute_terms, p)?,
        point(&m.absolute_derivative_terms, p)?,
    ))
}
