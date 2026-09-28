//! Pointwise weighted residuals. Logarithmic recovery avoids a premature
//! overflow/underflow in the weight alone. These are rounded point values,
//! not interval enclosures. An unrepresentable nonzero sample is an error.

use anyhow::{ensure, Result};

pub(super) fn native(left: f64, right: f64, u: f64, alpha: f64) -> Result<f64> {
    ensure!(
        left.is_finite() && right.is_finite(),
        "weighted residual samples must be finite"
    );
    ensure!(
        u.is_finite() && u >= 1.0 && alpha.is_finite(),
        "invalid weighted sample domain"
    );
    if left == right {
        return Ok(0.0);
    }
    let difference = left - right;
    let weight = u.powf(-alpha);
    let direct = difference * weight;
    if difference.is_normal() && weight.is_normal() && direct.is_normal() {
        return Ok(direct);
    }
    let log_difference = if difference.is_finite() && difference != 0.0 {
        difference.abs().ln()
    } else {
        let scale = left.abs().max(right.abs());
        ((left / scale) - (right / scale)).abs().ln() + scale.ln()
    };
    let value = (log_difference - alpha * u.ln()).exp();
    ensure!(
        value.is_finite() && value > 0.0,
        "nonzero weighted sample is outside the binary64 range"
    );
    Ok(if left > right { value } else { -value })
}

#[cfg(feature = "hp")]
pub(super) fn high_precision(
    left: &rug::Float,
    right: &rug::Float,
    u: &rug::Float,
    alpha: &rug::Float,
    working: u32,
) -> Result<rug::Float> {
    use rug::{ops::NegAssign, Float};
    ensure!(
        left.is_finite() && right.is_finite(),
        "weighted residual samples must be finite"
    );
    ensure!(
        u.is_finite() && u >= &1 && alpha.is_finite(),
        "invalid weighted sample domain"
    );
    if left == right {
        return Ok(Float::with_val(working, 0));
    }
    let difference = Float::with_val(working, left - right);
    let mut exponent = Float::with_val(working, u).ln();
    exponent *= alpha;
    exponent.neg_assign();
    let weight = exponent.clone().exp();
    let direct = Float::with_val(working, &difference * &weight);
    if difference.is_finite()
        && !difference.is_zero()
        && weight.is_finite()
        && !weight.is_zero()
        && difference.get_exp() != Some(rug::float::exp_min())
        && weight.get_exp() != Some(rug::float::exp_min())
        && direct.is_finite()
        && !direct.is_zero()
    {
        return Ok(direct);
    }
    let mut log_difference = if difference.is_finite()
        && !difference.is_zero()
        && difference.get_exp() != Some(rug::float::exp_min())
    {
        difference.abs().ln()
    } else {
        // A common power of two changes exponents without rounding source
        // significands. Using the maximum value itself as a divisor would
        // introduce cancellation error into a tiny normalized difference.
        let shift = left
            .get_exp()
            .into_iter()
            .chain(right.get_exp())
            .max()
            .expect("distinct finite samples cannot both be zero")
            - 1;
        let normalized_left = left.clone() >> shift;
        let normalized_right = right.clone() >> shift;
        let delta = Float::with_val(working, &normalized_left - &normalized_right);
        ensure!(
            !delta.is_zero() && delta.is_finite(),
            "weighted sample difference is unresolved after exact binary scaling"
        );
        delta.abs().ln() + Float::with_val(working, 2).ln() * shift
    };
    log_difference += exponent;
    let mut value = log_difference.exp();
    ensure!(
        value.is_finite() && !value.is_zero(),
        "nonzero weighted sample is outside the MPFR range"
    );
    if left < right {
        value.neg_assign();
    }
    Ok(value)
}
