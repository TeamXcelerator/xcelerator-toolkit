//! Persisted accuracy and admission of the actual stored eigenstate.
use super::{state_residual_bounds, CcmParityPolicy};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use serde::{Deserialize, Serialize};

pub(super) const ARITHMETIC: &str = "stored_state_residual_and_one_norm_scale_gap_v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredEigenvalueResolution {
    Resolved,
    BelowStorageFloor,
    StoredSignUnresolved,
}

/// Directed bounds for the indexed eigenpair of the stored matrix. The matrix
/// rounding scale is a resolution policy, not a finite-form assembly bound.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmStoredStateResolution {
    pub arithmetic: String,
    pub eigenvalue_resolution: StoredEigenvalueResolution,
    pub eigenvalue_lower: String,
    pub eigenvalue_upper: String,
    pub residual_upper: String,
    /// n * max |A_ij| * 2^-p, rounded upward; no arbitrary safety multiplier.
    pub matrix_rounding_scale_upper: String,
    /// Sum of coefficient magnitudes divided by the magnitude of their sum.
    /// None means the stored sum is exactly zero or its sign is unresolved.
    pub coefficient_sum_condition_upper: Option<String>,
    /// Absent only when the selected subspace has dimension one.
    pub selected_gap_lower: Option<String>,
    pub residual_angle_upper: String,
    /// Ordinary assembly does not supply this bound.
    pub assembly_error_bound: Option<String>,
}

pub(super) fn matrix_rounding_scale(a: &[Float], n: usize, p: u32) -> Result<Float> {
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || !(64..=1_000_000).contains(&p)
        || a.iter().any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid stored-matrix rounding-scale source");
    }
    let work = p + 64;
    let maximum = a
        .iter()
        .map(|x| Float::with_val(work, x).abs())
        .max_by(Float::total_cmp)
        .unwrap();
    let nonzero = !maximum.is_zero();
    let factor = Float::with_val(work, n) >> p;
    let scale = Float::with_val_round(work, maximum * factor, Round::Up).0;
    if !scale.is_finite() || (nonzero && scale.is_zero()) {
        bail!("CCM stored precision resolution limit: matrix rounding scale is unrepresentable");
    }
    Ok(scale)
}

pub(super) struct AdmissionBounds {
    pub lower: Float,
    pub upper: Float,
    pub neighbor_probe: Float,
    pub record: CcmStoredStateResolution,
}

/// Build the same directed endpoints used by ground-index inertia. This helper
/// does not by itself establish the endpoint counts; callers must validate them.
pub(super) fn bounds(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
) -> Result<AdmissionBounds> {
    let residual = state_residual_bounds::evaluate(a, v, lambda, p)?;
    bounds_with_residual(a, v, lambda, p, parity, &residual)
}

/// `bounds` from `state_residual_bounds::evaluate(a, v, lambda, p)`, already
/// evaluated successfully by the caller for exactly these inputs.
pub(super) fn bounds_with_residual(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
    residual: &state_residual_bounds::ResidualBounds,
) -> Result<AdmissionBounds> {
    let work = p + 64;
    let residual = residual.eigenvalue_error_upper.clone();
    let floor = matrix_rounding_scale(a, v.len(), p)?;
    let local = if lambda.is_zero() {
        a.iter()
            .filter(|x| !x.is_zero())
            .map(|x| Float::with_val(work, x).abs())
            .min_by(Float::total_cmp)
            .unwrap_or_else(|| Float::with_val(work, 1))
    } else {
        Float::with_val(work, lambda).abs()
    };
    let local = local >> (p + 8);
    let radius = Float::with_val_round(work, &residual * 8u32, Round::Up)
        .0
        .max(&local);
    let lower = Float::with_val_round(work, lambda - &radius, Round::Down).0;
    let upper = Float::with_val_round(work, lambda + &radius, Round::Up).0;
    let neighbor_probe = Float::with_val_round(work, &upper + &floor, Round::Up).0;
    let value_lower = Float::with_val_round(work, lambda - &residual, Round::Down).0;
    let value_upper = Float::with_val_round(work, lambda + &residual, Round::Up).0;
    let magnitude = Float::with_val(work, lambda).abs();
    let magnitude_lower = Float::with_val_round(work, &magnitude - &residual, Round::Down).0;
    let status = if magnitude_lower <= 0 {
        StoredEigenvalueResolution::StoredSignUnresolved
    } else if magnitude_lower <= floor {
        StoredEigenvalueResolution::BelowStorageFloor
    } else {
        StoredEigenvalueResolution::Resolved
    };
    let dimension = if parity == CcmParityPolicy::EvenSector {
        v.len() / 2 + 1
    } else {
        v.len()
    };
    let (gap, angle) = if dimension == 1 {
        (None, Float::with_val(work, 0))
    } else {
        // A conclusive count of one below neighbor_probe bounds every higher
        // eigenvalue from below. Residual inclusion bounds lambda_0 from above.
        let gap = Float::with_val_round(work, &neighbor_probe - &value_upper, Round::Down).0;
        let separation = Float::with_val_round(work, &gap - &residual, Round::Down).0;
        if separation <= 0 {
            bail!(
                "CCM stored precision resolution limit: residual does not separate the eigenstate"
            );
        }
        let angle = Float::with_val_round(work, &residual / separation, Round::Up).0;
        (Some(gap), angle)
    };
    if [&lower, &upper, &neighbor_probe, &angle]
        .iter()
        .any(|x| !x.is_finite())
        || lower >= upper
    {
        bail!("CCM stored precision resolution limit: ground-index boundaries are unrepresentable");
    }
    let exponent = v
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let mut sum = xc_numerics::mpfr_interval::MpfrInterval::from_i64(0, work);
    let mut magnitude_sum = sum.clone();
    for x in v {
        let value = crate::ccm::retained_evidence::finite_math::scale_float(x, -exponent, work)?;
        sum = sum.add(&xc_numerics::mpfr_interval::MpfrInterval::point(
            value.clone(),
        ));
        magnitude_sum = magnitude_sum.add(&xc_numerics::mpfr_interval::MpfrInterval::point(
            value.abs(),
        ));
    }
    let condition = if sum.contains_zero() {
        None
    } else {
        let denominator = crate::ccm::retained_evidence::finite_math::abs(&sum)?;
        Some(magnitude_sum.div(&denominator)?.upper().clone())
    };
    let down = |x: &Float| x.to_string_radix_round(10, None, Round::Down);
    let up = |x: &Float| x.to_string_radix_round(10, None, Round::Up);
    Ok(AdmissionBounds {
        lower,
        upper,
        neighbor_probe,
        record: CcmStoredStateResolution {
            arithmetic: ARITHMETIC.into(),
            eigenvalue_resolution: status,
            eigenvalue_lower: down(&value_lower),
            eigenvalue_upper: up(&value_upper),
            residual_upper: up(&residual),
            matrix_rounding_scale_upper: up(&floor),
            coefficient_sum_condition_upper: condition.as_ref().map(up),
            selected_gap_lower: gap.as_ref().map(down),
            residual_angle_upper: up(&angle),
            assembly_error_bound: None,
        },
    })
}
