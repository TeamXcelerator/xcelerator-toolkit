//! Checked agreement of stored generalized eigenpairs and computed metric images.
use super::{
    CrossCheckedGeneralizedEigenpairHp, DenseGeneralizedEigenpairReportHp, HpCrossCheckTolerance,
    MatrixFreeGeneralizedEigenpairReportHp, SolverError,
};
use rug::{float::Round, Float};
use xc_core::{AssuranceLevel, DecimalLiteral, ResultStatus};
use xc_numerics::mpfr_interval::MpfrInterval as I;
use xc_operator::GeneralizedEigenProblem;

fn invalid(message: impl Into<String>) -> SolverError {
    SolverError::CrossCheckDisagreement(message.into())
}

fn validate_values(
    value: &Float,
    vector: &[Float],
    diagnostics: &[&Float],
) -> Result<(), SolverError> {
    let p = value.prec();
    if !(33..=1_000_000).contains(&p)
        || !value.is_finite()
        || vector.is_empty()
        || vector.iter().all(Float::is_zero)
        || vector.iter().any(|x| !x.is_finite() || x.prec() != p)
        || diagnostics
            .iter()
            .any(|x| !x.is_finite() || **x < 0 || x.prec() != p)
    {
        return Err(invalid("generalized reports require finite compatible-precision values, nonzero vectors, and nonnegative finite diagnostics"));
    }
    Ok(())
}

fn tolerance(value: &DecimalLiteral, p: u32) -> Result<Float, SolverError> {
    value.validate().map_err(|e| invalid(e.to_string()))?;
    let parse = || Float::parse(value.as_str()).map_err(|e| invalid(e.to_string()));
    let nearest = Float::with_val(p, parse()?);
    let canonical = value.canonical().map_err(|e| invalid(e.to_string()))?;
    if !nearest.is_finite() || nearest < 0 || (nearest.is_zero() && canonical.as_str() != "0") {
        return Err(invalid(
            "generalized tolerance must be finite, nonnegative and representable",
        ));
    }
    Ok(Float::with_val_round(p, parse()?, Round::Down).0)
}

fn dot(left: &[Float], right: &[Float], p: u32) -> Result<I, SolverError> {
    if left.len() != right.len() {
        return Err(invalid("metric action changed dimension"));
    }
    let mut result = I::from_i64(0, p);
    for (a, b) in left.iter().zip(right) {
        let a = I::from_float(a, p).map_err(|e| invalid(e.to_string()))?;
        let b = I::from_float(b, p).map_err(|e| invalid(e.to_string()))?;
        result = result.add(&a.mul(&b));
    }
    result.validate().map_err(|e| invalid(e.to_string()))?;
    Ok(result)
}

pub(crate) fn check(
    problem: &GeneralizedEigenProblem<'_, Float>,
    matrix_free: &MatrixFreeGeneralizedEigenpairReportHp,
    dense: &DenseGeneralizedEigenpairReportHp,
    limits: &HpCrossCheckTolerance,
) -> Result<CrossCheckedGeneralizedEigenpairHp, SolverError> {
    let n = problem.operator.dimension();
    if n == 0
        || problem.metric.dimension() != n
        || matrix_free.eigenvector.len() != n
        || dense.eigenvector.len() != n
        || matrix_free.target != dense.target
    {
        return Err(invalid(
            "generalized reports must have matching positive dimensions and targets",
        ));
    }
    if matrix_free.status != ResultStatus::Converged || dense.status != ResultStatus::Converged {
        return Err(SolverError::NonConvergence(
            "generalized cross-check requires converged reports".into(),
        ));
    }
    validate_values(
        &matrix_free.eigenvalue,
        &matrix_free.eigenvector,
        &[
            &matrix_free.residual_norm,
            &matrix_free.relative_residual,
            &matrix_free.scaled_backward_error,
            &matrix_free.metric_normalization_error,
            &matrix_free.ritz_value_stability,
            &matrix_free.diagnostics.absolute_residual,
            &matrix_free.diagnostics.relative_residual,
            &matrix_free.diagnostics.scaled_backward_error,
            &matrix_free.diagnostics.orthogonality_error,
        ],
    )?;
    validate_values(
        &dense.eigenvalue,
        &dense.eigenvector,
        &[
            &dense.residual_norm,
            &dense.relative_residual,
            &dense.scaled_backward_error,
            &dense.metric_normalization_error,
            &dense.minimum_cholesky_pivot,
            &dense.diagnostics.absolute_residual,
            &dense.diagnostics.relative_residual,
            &dense.diagnostics.scaled_backward_error,
            &dense.diagnostics.orthogonality_error,
        ],
    )?;
    if dense.precision_bits != dense.eigenvalue.prec()
        || dense.minimum_cholesky_pivot <= 0
        || matrix_free.algorithm.trim().is_empty()
        || dense.algorithm.trim().is_empty()
        || matrix_free.algorithm == dense.algorithm
    {
        return Err(invalid("generalized cross-check needs consistent precision, positive Cholesky evidence and distinct identified routes"));
    }
    // These typed reports have distinct registered producers. Free-text
    // renaming cannot supply an alternative independent implementation.
    if matrix_free.algorithm
        != "matrix_free_generalized_b_orthogonal_rayleigh_ritz_fresh_images_rounding_margin_hp_v4"
        || dense.algorithm
            != format!(
                "dense_generalized_cholesky_whitening_gap_bound_hp_v4:{}",
                xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS
            )
    {
        return Err(invalid(
            "unregistered generalized cross-check implementation identity",
        ));
    }
    super::require_independent_solver_routes(
        super::SolverRoute::HpMatrixFreeGeneralizedRayleighRitz,
        super::SolverRoute::HpDenseGeneralizedWhiteningReference,
        matrix_free.eigenvalue.prec().max(dense.eigenvalue.prec()),
    )?;
    let p = matrix_free.eigenvalue.prec().max(dense.eigenvalue.prec()) + 64;
    let eigenvalue_tolerance = tolerance(&limits.eigenvalue_absolute, p)?;
    let overlap_tolerance = tolerance(&limits.one_minus_overlap_squared, p)?;
    let difference = if matrix_free.eigenvalue >= dense.eigenvalue {
        Float::with_val_round(p, &matrix_free.eigenvalue - &dense.eigenvalue, Round::Up).0
    } else {
        Float::with_val_round(p, &dense.eigenvalue - &matrix_free.eigenvalue, Round::Up).0
    };
    if !difference.is_finite() || difference > eigenvalue_tolerance {
        return Err(invalid(
            "generalized eigenvalue difference exceeds tolerance or is unrepresentable",
        ));
    }
    let mut left_image = vec![Float::with_val(p, 0); n];
    let mut right_image = left_image.clone();
    problem
        .metric
        .apply(&matrix_free.eigenvector, &mut left_image)?;
    problem.metric.apply(&dense.eigenvector, &mut right_image)?;
    let left_norm = dot(&matrix_free.eigenvector, &left_image, p)?;
    let right_norm = dot(&dense.eigenvector, &right_image, p)?;
    if !left_norm.is_strictly_positive() || !right_norm.is_strictly_positive() {
        return Err(invalid(
            "generalized cross-check needs strictly positive computed metric norms",
        ));
    }
    let overlap = dot(&matrix_free.eigenvector, &right_image, p)?
        .square()
        .div(&left_norm.mul(&right_norm))
        .map_err(|e| invalid(e.to_string()))?;
    let discrepancy = I::from_i64(1, p).sub(&overlap);
    discrepancy.validate().map_err(|e| invalid(e.to_string()))?;
    let error = discrepancy
        .lower()
        .clone()
        .abs()
        .max(&discrepancy.upper().clone().abs());
    if !error.is_finite() || error > overlap_tolerance {
        return Err(SolverError::UnresolvedEigenspace(
            "matching eigenvalues do not establish collinearity; a repeated or unresolved eigenspace requires a subspace comparison, not a one-vector verdict".into(),
        ));
    }
    let mut matrix_free = matrix_free.clone();
    matrix_free.assurance = AssuranceLevel::Computed;
    let mut dense_whitening = dense.clone();
    dense_whitening.assurance = AssuranceLevel::Computed;
    Ok(CrossCheckedGeneralizedEigenpairHp {
        matrix_free,
        dense_whitening,
        eigenvalue_absolute_difference: difference,
        one_minus_metric_overlap_squared: error,
        assurance: AssuranceLevel::Computed,
    })
}
