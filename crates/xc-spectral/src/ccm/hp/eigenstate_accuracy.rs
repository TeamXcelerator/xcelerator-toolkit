//! Stored-point accuracy diagnostics and residual-verified Krylov polishing.
use super::{state_residual_bounds, HighPrecResult};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::linalg::LuFactors;

/// Accuracy of an admitted selected-subspace eigenvalue of stored Tau.
/// This is not an error bound for the unrounded finite CCM construction.
#[derive(Clone, Debug)]
pub struct CcmStoredEigenvalueAccuracy {
    /// Residual inclusion radius in eigenvalue units for the selected stored
    /// eigenvalue; the ordinary producer separately validates its index.
    pub absolute_error_upper: Float,
    /// Relative to the exact stored eigenvalue, absent if its sign is unresolved.
    pub relative_error_upper: Option<Float>,
    pub stored_eigenvalue_sign_resolved: bool,
    /// Always None: ordinary point assembly provides no rigorous error bound.
    pub assembly_error_bound: Option<Float>,
}

pub(super) fn diagnostic(result: &HighPrecResult) -> Result<CcmStoredEigenvalueAccuracy> {
    let p = result.precision_bits;
    let residual = &result
        .inverse_iteration_diagnostics
        .final_relative_residual_norm;
    if !(64..=1_000_000).contains(&p)
        || result.xi.is_empty()
        || !residual.is_finite()
        || residual < &0
        || !result.weil_min_eigenvalue.is_finite()
    {
        bail!("invalid stored eigenvalue accuracy inputs");
    }
    let work = p + 64;
    // ||r||2/||v||2 <= sqrt(n) ||r||inf/||v||inf. The persisted diagnostic
    // is a directed bound on the latter, in eigenvalue units (not dimensionless).
    let mut factor = Float::with_val_round(work, result.xi.len(), Round::Up).0;
    factor.sqrt_round(Round::Up);
    let radius = Float::with_val_round(work, residual * &factor, Round::Up).0;
    if !radius.is_finite() || (!residual.is_zero() && radius.is_zero()) {
        bail!("stored eigenvalue accuracy radius is unrepresentable");
    }
    let magnitude = Float::with_val(work, &result.weil_min_eigenvalue).abs();
    let denominator = Float::with_val_round(work, &magnitude - &radius, Round::Down).0;
    let relative = if denominator > 0 {
        let bound = Float::with_val_round(work, &radius / &denominator, Round::Up).0;
        if !bound.is_finite() {
            bail!("stored eigenvalue relative accuracy is unrepresentable");
        }
        Some(bound)
    } else {
        None
    };
    Ok(CcmStoredEigenvalueAccuracy {
        absolute_error_upper: radius,
        stored_eigenvalue_sign_resolved: relative.is_some(),
        relative_error_upper: relative,
        assembly_error_bound: None,
    })
}

/// Numerical storage-floor admission policy; the measured residual itself is
/// directed. This floor is not a bound on transcendental/quad assembly error.
pub(super) fn residual_floor(a: &[Float], n: usize, p: u32) -> Result<Float> {
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || !(64..=1_000_000).contains(&p)
        || a.iter().any(|x| !x.is_finite())
    {
        bail!("invalid eigenstate polishing floor inputs");
    }
    let scale = a
        .iter()
        .map(|x| x.clone().abs())
        .max_by(Float::total_cmp)
        .unwrap();
    let factor = Float::with_val(p + 64, 16 * n);
    let floor = Float::with_val_round(p + 64, scale * factor, Round::Up).0 >> p;
    if !floor.is_finite() || floor.is_zero() {
        bail!("eigenstate polishing floor is unrepresentable");
    }
    Ok(floor)
}

pub(super) struct PolishedState {
    pub value: Float,
    pub vector: Vec<Float>,
    pub diagnostics: xc_numerics::linalg::InverseIterationDiagnostics,
    pub candidate_adopted: bool,
}

pub(super) fn polish(
    a: &[Float],
    factors: &LuFactors,
    value: &Float,
    vector: &[Float],
    p: u32,
    steps: usize,
) -> Result<PolishedState> {
    let n = vector.len();
    let floor = residual_floor(a, n, p)?;
    let mut residual = state_residual_bounds::evaluate(a, vector, value, p)?.eigenvalue_error_upper;
    let (mut value, mut vector) = (value.clone(), vector.to_vec());
    let mut used_steps = 0;
    let mut candidate_adopted = false;
    let diagnostics = loop {
        // A stationary Rayleigh quotient alone does not resolve the vector.
        // In particular, an exactly representable eigenvalue makes the shifted
        // system singular before the unshifted vector necessarily meets the
        // residual floor. Continue from the best state within the original
        // unshifted-step budget; each phase retains its actual shifted outcome.
        let candidate = xc_numerics::linalg::inverse_iteration_from_factors_detailed(
            a,
            factors,
            n,
            p,
            steps.saturating_sub(used_steps),
            false,
            Some(vector.clone()),
        )?;
        if candidate.diagnostics.unshifted_steps == 0
            || candidate.diagnostics.unshifted_steps > steps.saturating_sub(used_steps)
        {
            bail!("Krylov polishing returned invalid unshifted-step progress");
        }
        used_steps += candidate.diagnostics.unshifted_steps;
        let after =
            state_residual_bounds::evaluate(a, &candidate.eigenvector, &candidate.eigenvalue, p)?
                .eigenvalue_error_upper;
        if after <= residual {
            value = candidate.eigenvalue;
            vector = candidate.eigenvector;
            residual = after;
            candidate_adopted = true;
        }
        if residual <= floor {
            // The count is cumulative; convergence/change/refinement describe
            // the final phase. The residual is for the actually retained state.
            let mut diagnostics = candidate.diagnostics;
            diagnostics.configured_step_limit = steps;
            diagnostics.unshifted_steps = used_steps;
            diagnostics.final_relative_residual_norm =
                state_residual_bounds::evaluate(a, &vector, &value, p)?
                    .vector_scaled_residual_upper;
            break diagnostics;
        }
        if used_steps >= steps {
            bail!(
                "Krylov polishing did not resolve the stored eigenstate to its numerical storage floor after {used_steps} unshifted steps"
            );
        }
    };
    Ok(PolishedState {
        value,
        vector,
        diagnostics,
        candidate_adopted,
    })
}
