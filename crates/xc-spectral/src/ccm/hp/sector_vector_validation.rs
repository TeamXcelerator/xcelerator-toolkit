//! Independent directed checks of retained sector eigenvectors.
use crate::ccm::retained_evidence::finite_math::scale_float;
use anyhow::{bail, Result};
use rug::{float::Round, Float};

/// Verify unit normalization and a residual relative to the matrix scale.
/// The tolerance is 2^-(p-32), reserving 32 bits for the finite solver/reduction
/// error. This validates stored points, not continuum or assembly errors.
pub(super) fn validate(a: &[Float], v: &[Float], lambda: &Float, p: u32) -> Result<()> {
    let n = v.len();
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || !(64..=1_000_000).contains(&p)
        || a.iter()
            .chain(v)
            .chain(std::iter::once(lambda))
            .any(|x| !x.is_finite() || x.prec() > p)
        || v.iter().all(Float::is_zero)
    {
        bail!("invalid stored sector eigenvector inputs");
    }
    let work = p + 64;
    let buffers = (a.len() as u64)
        .saturating_mul(4)
        .saturating_add((n as u64).saturating_mul(12))
        .saturating_add(256);
    if buffers.saturating_mul(u64::from(work).div_ceil(8) + 64) > (8u64 << 30) {
        bail!("sector eigenvector validation exceeds numerical workspace budget");
    }
    let tolerance = Float::with_val(work, 1) >> (p - 32);
    // Directed hypot avoids overflow/underflow in individual squares, including
    // negligible components whose exact squares are outside the MPFR range.
    let mut lower = Float::with_val(work, 0);
    let mut upper = Float::with_val(work, 0);
    for x in v {
        lower.hypot_round(x, Round::Down);
        upper.hypot_round(x, Round::Up);
    }
    let minimum = Float::with_val(work, 1) - &tolerance;
    let maximum = Float::with_val(work, 1) + &tolerance;
    if !lower.is_finite() || !upper.is_finite() || lower < minimum || upper > maximum {
        bail!("stored sector eigenvector is not unit normalized");
    }
    let exponent = a
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = a
        .iter()
        .map(|x| scale_float(x, -exponent, p))
        .collect::<Result<Vec<_>>>()?;
    let lambda = scale_float(lambda, -exponent, p)?;
    let residual = super::state_residual_bounds::evaluate(&normalized, v, &lambda, p)?;
    let mut natural = Float::with_val(work, 1);
    for row in normalized.chunks(n) {
        let terms = row.iter().map(|x| x.clone().abs()).collect::<Vec<_>>();
        let bound = Float::with_val_round(work, Float::sum(terms.iter()), Round::Up).0;
        if bound > natural {
            natural = bound;
        }
    }
    let allowance = natural * tolerance;
    if !allowance.is_finite() || residual.eigenvalue_error_upper > allowance {
        bail!("stored sector eigenvector exceeds directed relative residual tolerance: residual={}, allowance={}, matrix_binary_scale={}", residual.eigenvalue_error_upper, allowance, exponent);
    }
    Ok(())
}
