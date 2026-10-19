//! Point adapter for the directed finite Fourier transform kernel.
use super::{transform_math, RetainedState};
use anyhow::{bail, Result};
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;
#[derive(Debug)]
pub(in crate::ccm) struct GuardExhausted {
    pub maximum: u32,
    pub requested: u32,
}
impl std::fmt::Display for GuardExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "finite transform unresolved within {} guard bits at {} requested bits; increase XC_TRANSFORM_MAX_GUARD_BITS within the run memory budget", self.maximum, self.requested)
    }
}
impl std::error::Error for GuardExhausted {}

pub(in crate::ccm) fn retry_extended(
    result: &Result<impl Sized>,
    exhausted: bool,
    mode: xc_cache::ArtifactExecutionCacheMode,
) -> bool {
    result.as_ref().err().is_some_and(|error| {
        exhausted
            || (mode.requires_reuse()
                && matches!(
                    error.downcast_ref::<xc_cache::CacheError>(),
                    Some(xc_cache::CacheError::NotFound(_))
                ))
    })
}
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
    terms_with_guard_limit(
        state,
        t,
        p,
        transform_math::configured_maximum_guard_bits(p)?,
        false,
    )
}

pub(in crate::ccm) fn terms_with_guard_limit(
    state: &RetainedState,
    t: &Float,
    p: u32,
    maximum: u32,
    root_convention: bool,
) -> Result<(Float, Float, Float, Float)> {
    let Some(m) = transform_math::measure_with_guard_limit(state, t, p, maximum, root_convention)?
    else {
        return Err(GuardExhausted {
            maximum,
            requested: p,
        }
        .into());
    };
    Ok((
        point(&m.value, p)?,
        point(&m.derivative, p)?,
        point(&m.absolute_terms, p)?,
        point(&m.absolute_derivative_terms, p)?,
    ))
}
