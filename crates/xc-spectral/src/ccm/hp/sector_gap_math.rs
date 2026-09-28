//! Stored-point sector comparisons and indexed spectrum validation.
use crate::ccm::retained_evidence::finite_math::scale_float;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

/// Correctly rounded logarithmic ratio of the exact retained magnitudes.
/// Near equality uses log1p of the relative difference before any logarithm.
pub(in crate::ccm) fn log_magnitude_ratio(
    numerator: &Float,
    denominator: &Float,
    p: u32,
) -> Result<Float> {
    if !(64..=1_000_000).contains(&p)
        || [numerator, denominator]
            .iter()
            .any(|x| !x.is_finite() || x.is_zero() || x.prec() > p)
    {
        bail!("invalid sector logarithmic ratio inputs");
    }
    let a = denominator.clone().abs();
    let b = numerator.clone().abs();
    if a == b {
        return Ok(Float::with_val(p, 0));
    }
    let ea = i64::from(a.get_exp().unwrap());
    let eb = i64::from(b.get_exp().unwrap());
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p + guard;
        let logarithm = if ea.abs_diff(eb) <= 1 {
            let exponent = ea.max(eb);
            let a = I::point(scale_float(&a, -exponent, work)?);
            let b = I::point(scale_float(&b, -exponent, work)?);
            let delta = b.sub(&a).div(&a)?;
            let mut lower = delta.lower().clone();
            let mut upper = delta.upper().clone();
            lower.ln_1p_round(Round::Down);
            upper.ln_1p_round(Round::Up);
            I::new(lower, upper)?
        } else {
            I::from_float(&b, work)?
                .ln()?
                .sub(&I::from_float(&a, work)?.ln()?)
        };
        let ratio = logarithm.div(&I::from_i64(10, work).ln()?)?;
        let lower = Float::with_val(p, ratio.lower());
        let upper = Float::with_val(p, ratio.upper());
        if lower == upper && lower.is_finite() && !lower.is_zero() {
            return Ok(lower);
        }
    }
    bail!("sector logarithmic ratio unresolved within 4096 guard bits")
}

pub(super) const QR_AGREEMENT_SEMANTICS: &str = "indexed_qr_eight_n_unit_norm_agreement_v1";

/// Small normwise comparison allowance for a computed QR point, not an
/// eigenvalue error certificate. Independently replayed Sturm intervals supply
/// the spectral bounds; downstream source-resolution gates remain mandatory.
pub(super) fn qr_agreement_allowance(diagonal: &[Float], off: &[Float], p: u32) -> Result<Float> {
    if !(64..=1_000_000).contains(&p)
        || diagonal.is_empty()
        || off.len().checked_add(1) != Some(diagonal.len())
        || diagonal
            .iter()
            .chain(off)
            .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid QR normwise comparison source");
    }
    let work = p + 64;
    let mut norm = Float::with_val(work, 0);
    for (index, value) in diagonal.iter().enumerate() {
        let mut row = I::point(Float::with_val(work, value).abs());
        if index > 0 {
            row = row.add(&I::point(Float::with_val(work, &off[index - 1]).abs()));
        }
        if index < off.len() {
            row = row.add(&I::point(Float::with_val(work, &off[index]).abs()));
        }
        norm = norm.max(row.upper());
    }
    let factor = Float::with_val_round(work, 8usize * diagonal.len(), Round::Up).0 >> p;
    let radius = Float::with_val_round(work, norm * factor, Round::Up).0;
    if !radius.is_finite() {
        bail!("QR comparison allowance is unrepresentable");
    }
    Ok(radius)
}

pub(super) fn qr_point_agrees(
    point: &Float,
    lower: &Float,
    upper: &Float,
    allowance: &Float,
) -> bool {
    let work = point
        .prec()
        .max(lower.prec())
        .max(upper.prec())
        .max(allowance.prec());
    let low = Float::with_val_round(work, lower - allowance, Round::Down).0;
    let high = Float::with_val_round(work, upper + allowance, Round::Up).0;
    point.is_finite() && allowance.is_finite() && allowance >= &0 && point >= &low && point <= &high
}

/// Replay algebraic index membership within the small QR comparison allowance.
/// The allowance is a numerical agreement policy, not an assembly-error bound.
pub(super) fn validate_complete_points(
    diagonal: &[Float],
    off: &[Float],
    values: &[Float],
    p: u32,
) -> Result<()> {
    if !(64..=1_000_000).contains(&p)
        || diagonal.is_empty()
        || off.len().checked_add(1) != Some(diagonal.len())
        || values.len() != diagonal.len()
        || diagonal
            .iter()
            .chain(off)
            .chain(values)
            .any(|x| !x.is_finite() || x.prec() > p)
        || values.windows(2).any(|x| x[0] >= x[1])
    {
        bail!("invalid complete spectrum shape, order, precision or range");
    }
    let radius = qr_agreement_allowance(diagonal, off, p)?;
    if diagonal.len() == 1 {
        if qr_point_agrees(&values[0], &diagonal[0], &diagonal[0], &radius) {
            return Ok(());
        }
        bail!("reported scalar eigenvalue leaves the working precision allowance");
    }
    let work = p + 64;
    let exponent = diagonal
        .iter()
        .chain(off)
        .filter_map(Float::get_exp)
        .max()
        .unwrap_or(0);
    let scale = |x: &Float| scale_float(x, -i64::from(exponent), work);
    let diagonal = diagonal.iter().map(scale).collect::<Result<Vec<_>>>()?;
    let off = off.iter().map(scale).collect::<Result<Vec<_>>>()?;
    let radius = scale(&radius)?;
    for (index, value) in values.iter().enumerate() {
        let point = scale(value)?;
        let lower = Float::with_val_round(work, &point - &radius, Round::Down).0;
        let upper = Float::with_val_round(work, &point + &radius, Round::Up).0;
        let lower_count =
            xc_numerics::eigen::tridiag_sturm_count_below_hp(&diagonal, &off, &lower, work)?;
        let upper_count =
            xc_numerics::eigen::tridiag_sturm_count_below_hp(&diagonal, &off, &upper, work)?;
        if lower_count > index || upper_count <= index {
            bail!(
                "reported eigenvalue {index} does not enclose its indexed root within the working precision allowance"
            );
        }
    }
    Ok(())
}

/// Preserve every exact tridiagonal point while removing the common exponent
/// before determinant products in indexed Sturm selection/replay.
pub(super) fn scaled_tridiagonal(
    diagonal: &[Float],
    off: &[Float],
    p: u32,
) -> Result<(Vec<Float>, Vec<Float>, i64)> {
    if !(64..=1_000_000).contains(&p)
        || diagonal.is_empty()
        || off.len().checked_add(1) != Some(diagonal.len())
        || diagonal
            .iter()
            .chain(off)
            .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid tridiagonal scaling inputs");
    }
    if (diagonal.len() as u64)
        .saturating_mul(4)
        .saturating_mul(u64::from(p).div_ceil(8) + 64)
        > (8u64 << 30)
    {
        bail!("tridiagonal scaling exceeds numerical workspace budget");
    }
    let exponent = diagonal
        .iter()
        .chain(off)
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = |values: &[Float]| {
        values
            .iter()
            .map(|x| scale_float(x, -exponent, p))
            .collect::<Result<Vec<_>>>()
    };
    Ok((normalized(diagonal)?, normalized(off)?, exponent))
}
