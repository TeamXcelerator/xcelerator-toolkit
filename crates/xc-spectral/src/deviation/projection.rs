use super::DeviationMetric;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use std::cmp::Ordering;

/// Arithmetic identity for managed discrete projection artifacts.
pub const PROJECTION_ARITHMETIC_V2: &str = "binary_scaled_weighted_hypot_projection_v2";

/// Point measurements of one discrete weighted projection. These values do not
/// enclose quadrature error or establish a law across different profiles.
#[derive(Clone, Debug)]
pub struct DeviationProjection {
    /// Signed coefficient <D,g>/<g,g> in the selected discrete metric.
    pub amplitude: Float,
    /// Discrete weighted norm of D.
    pub deviation_norm: Float,
    /// Discrete weighted norm of g.
    pub reference_norm: Float,
    /// Discrete weighted norm of the internally computed residual.
    pub residual_norm: Float,
    /// Residual norm / deviation norm; zero for identically zero D.
    pub relative_residual: Float,
}

// Binary scaling must be reversible: this also rejects partial underflow that
// rounds a value upward to the smallest representable nonzero MPFR number.
fn scale(value: &Float, exponent: i64, prec: u32) -> Result<Float> {
    let mut out = Float::with_val(prec, value);
    if value.is_zero() {
        return Ok(out);
    }
    let amount = u32::try_from(exponent.unsigned_abs())?;
    if exponent >= 0 {
        out <<= amount;
    } else {
        out >>= amount;
    }
    if !out.is_finite() || out.is_zero() {
        bail!("projection binary scale exceeds MPFR range");
    }
    let mut recovered = out.clone();
    if exponent >= 0 {
        recovered >>= amount;
    } else {
        recovered <<= amount;
    }
    if recovered != *value {
        bail!("projection binary scaling lost a nonzero component");
    }
    Ok(out)
}

fn scale_vector(values: &[Float], prec: u32) -> Result<(Vec<Float>, i64)> {
    let exponent = values
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, |e| i64::from(e) - 1);
    Ok((
        values
            .iter()
            .map(|v| scale(v, -exponent, prec))
            .collect::<Result<_>>()?,
        exponent,
    ))
}

fn norm(values: &[Float], prec: u32) -> Result<Float> {
    let mut n = Float::with_val(prec, 0);
    for v in values {
        n.hypot_mut(v);
    }
    if !n.is_finite() {
        bail!("projection norm exceeds MPFR range");
    }
    Ok(n)
}

// A product of two p-bit stored numbers needs at most 2p significant bits.
// A nonexact result here therefore indicates exponent-range loss, not ordinary
// p-bit rounding. This avoids the intermediate-range exception in Float::dot.
fn exact_product(a: &Float, b: &Float, prec: u32) -> Result<Float> {
    let (out, dir) = Float::with_val_round(2 * prec, a * b, Round::Nearest);
    if !out.is_finite() || dir != Ordering::Equal {
        bail!("projection product exceeds exact intermediate MPFR range");
    }
    Ok(out)
}

fn rounded_nonzero(value: &Float, prec: u32) -> Result<Float> {
    let out = Float::with_val(prec, value);
    if !out.is_finite() || (!value.is_zero() && out.is_zero()) {
        bail!("projection output cannot be represented at the requested precision");
    }
    Ok(out)
}

/// Project a deviation onto a reference using trapezoidal node weights on the
/// supplied grid in [1,infinity). All samples are interpreted as stored numbers.
///
/// Requires finite samples, an ascending grid of at least two points, a nonzero
/// reference, and output precision 32..=1,000,000. Source precision may exceed
/// output precision (up to 1,000,064, including evaluator guard bits); internal
/// precision is their maximum plus
/// 32 guard bits. Outputs are rounded point measurements, not interval bounds.
/// Explicit range failures return errors, including unrepresentable nonzero
/// intermediate products. There is no universal composite rounding guarantee.
pub fn project(
    us: &[Float],
    deviation: &[Float],
    reference: &[Float],
    metric: DeviationMetric,
    prec: u32,
) -> Result<DeviationProjection> {
    if !(32..=1_000_000).contains(&prec) {
        bail!("projection precision must be 32 through 1,000,000 bits");
    }
    if us.len() < 2 || us.len() != deviation.len() || us.len() != reference.len() {
        bail!("projection requires matching sample counts and at least two grid points");
    }
    if us
        .iter()
        .chain(deviation)
        .chain(reference)
        .any(|v| !v.is_finite() || v.prec() > 1_000_064)
    {
        bail!("projection requires finite samples at supported precision");
    }
    if us[0] < 1 || us.windows(2).any(|pair| pair[1] <= pair[0]) {
        bail!("projection requires a strictly ascending grid in [1,infinity)");
    }
    if reference.iter().all(Float::is_zero) {
        bail!("projection reference must be nonzero");
    }
    let working = us
        .iter()
        .chain(deviation)
        .chain(reference)
        .map(Float::prec)
        .fold(prec, u32::max)
        + 32;
    let weights: Vec<Float> = (0..us.len())
        .map(|i| {
            let left = i.saturating_sub(1);
            let right = (i + 1).min(us.len() - 1);
            let mut width = Float::with_val(working, &us[right] - &us[left]);
            width /= 2;
            let u = Float::with_val(working, &us[i]);
            width /= match metric {
                DeviationMetric::FactorWeighted => u,
                DeviationMetric::IntegrandWeighted => u.sqrt(),
            };
            if !width.is_finite() || width <= 0 {
                bail!("projection node weight is unresolved");
            }
            Ok(width)
        })
        .collect::<Result<_>>()?;
    let weight_exp = i64::from(weights.iter().filter_map(Float::get_exp).max().unwrap()) - 1;
    let weight_exp = weight_exp.div_euclid(2) * 2;
    let (d, d_exp) = scale_vector(deviation, working)?;
    let (g, g_exp) = scale_vector(reference, working)?;
    let mut x = Vec::with_capacity(us.len());
    let mut y = Vec::with_capacity(us.len());
    for ((w, a), b) in weights.iter().zip(&d).zip(&g) {
        let root = scale(w, -weight_exp, working)?.sqrt();
        x.push(rounded_nonzero(
            &exact_product(&root, a, working)?,
            working,
        )?);
        y.push(rounded_nonzero(
            &exact_product(&root, b, working)?,
            working,
        )?);
    }
    let nx = norm(&x, working)?;
    let ny = norm(&y, working)?;
    if ny <= 0 {
        bail!("projection reference norm is unresolved");
    }
    let reference_norm = rounded_nonzero(&scale(&ny, g_exp + weight_exp / 2, working)?, prec)?;
    if nx.is_zero() {
        return Ok(DeviationProjection {
            amplitude: Float::with_val(prec, 0),
            deviation_norm: Float::with_val(prec, 0),
            reference_norm,
            residual_norm: Float::with_val(prec, 0),
            relative_residual: Float::with_val(prec, 0),
        });
    }
    let unit_y: Vec<Float> = y
        .iter()
        .map(|v| {
            let out = Float::with_val(working, v / &ny);
            if !out.is_finite() || (!v.is_zero() && out.is_zero()) {
                bail!("projection reference normalization lost a nonzero component");
            }
            Ok(out)
        })
        .collect::<Result<_>>()?;
    let products: Vec<Float> = x
        .iter()
        .zip(&unit_y)
        .map(|(a, b)| exact_product(a, b, working))
        .collect::<Result<_>>()?;
    let (overlap, direction) =
        Float::with_val_round(working, Float::sum(products.iter()), Round::Nearest);
    if !overlap.is_finite() || (overlap.is_zero() && direction != Ordering::Equal) {
        bail!("projection overlap is outside MPFR range");
    }
    let mut residual = Vec::with_capacity(x.len());
    for (a, b) in x.iter().zip(&unit_y) {
        let mut value = -overlap.clone();
        let direction = value.mul_add_round(b, a, Round::Nearest);
        if !value.is_finite() || (value.is_zero() && direction != Ordering::Equal) {
            bail!("projection residual component is outside MPFR range");
        }
        residual.push(value);
    }
    let nr = norm(&residual, working)?;
    let amplitude = Float::with_val(working, &overlap / &ny);
    if !amplitude.is_finite() || (!overlap.is_zero() && amplitude.is_zero()) {
        bail!("projection amplitude is outside MPFR range");
    }
    let relative = Float::with_val(working, &nr / &nx);
    if !relative.is_finite() || (!nr.is_zero() && relative.is_zero()) {
        bail!("projection relative residual is outside MPFR range");
    }
    Ok(DeviationProjection {
        amplitude: rounded_nonzero(&scale(&amplitude, d_exp - g_exp, working)?, prec)?,
        deviation_norm: rounded_nonzero(&scale(&nx, d_exp + weight_exp / 2, working)?, prec)?,
        reference_norm,
        residual_norm: rounded_nonzero(&scale(&nr, d_exp + weight_exp / 2, working)?, prec)?,
        relative_residual: rounded_nonzero(&relative, prec)?,
    })
}
