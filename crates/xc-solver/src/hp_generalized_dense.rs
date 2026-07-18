use super::{GeneralizedExtremeConfigHp, HpCrossCheckTolerance, SolverError};
use rug::Float;
use xc_core::{AssuranceLevel, EigenTarget, ResultStatus, SolverProvenance, TerminationReason};
use xc_operator::GeneralizedEigenProblem;

/// Dense real-symmetric generalized problem retained entirely in MPFR.
pub struct DenseGeneralizedProblemHp<'a> {
    pub operator: &'a [Float],
    pub metric: &'a [Float],
    pub dimension: usize,
}

impl<'a> DenseGeneralizedProblemHp<'a> {
    pub fn new(
        operator: &'a [Float],
        metric: &'a [Float],
        dimension: usize,
    ) -> Result<Self, SolverError> {
        let expected = dimension.saturating_mul(dimension);
        if dimension == 0 || operator.len() != expected || metric.len() != expected {
            return Err(SolverError::InvalidConfiguration(format!(
                "dense HP generalized matrices must both contain {expected} entries"
            )));
        }
        Ok(Self {
            operator,
            metric,
            dimension,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DenseGeneralizedEigenpairReportHp {
    pub target: EigenTarget,
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub relative_residual: Float,
    pub scaled_backward_error: Float,
    pub metric_normalization_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
    pub minimum_cholesky_pivot: Float,
    pub precision_bits: u32,
    pub factorization_count: usize,
    pub estimated_peak_memory_bytes: u64,
    pub algorithm: String,
    pub metric_validity_evidence: String,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

#[derive(Clone, Debug)]
pub struct CrossCheckedGeneralizedEigenpairHp {
    pub matrix_free: super::MatrixFreeGeneralizedEigenpairReportHp,
    pub dense_whitening: DenseGeneralizedEigenpairReportHp,
    pub eigenvalue_absolute_difference: Float,
    pub one_minus_metric_overlap_squared: Float,
    pub assurance: AssuranceLevel,
}

fn zero(precision_bits: u32) -> Float {
    Float::with_val(precision_bits, 0)
}

fn parse_positive(
    literal: &xc_core::DecimalLiteral,
    precision_bits: u32,
    name: &str,
) -> Result<Float, SolverError> {
    let parsed = Float::parse(literal.as_str()).map_err(|error| {
        SolverError::InvalidConfiguration(format!("failed to parse {name}: {error}"))
    })?;
    let parsed = Float::with_val(precision_bits, parsed);
    if !parsed.is_finite() || parsed <= 0 {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(parsed)
}

fn dot(left: &[Float], right: &[Float], precision_bits: u32) -> Float {
    let mut sum = zero(precision_bits);
    for (left, right) in left.iter().zip(right) {
        let mut term = Float::with_val(precision_bits, left);
        term *= right;
        sum += term;
    }
    sum
}

fn norm(vector: &[Float], precision_bits: u32) -> Float {
    dot(vector, vector, precision_bits).sqrt()
}

fn matvec(matrix: &[Float], vector: &[Float], dimension: usize, precision_bits: u32) -> Vec<Float> {
    (0..dimension)
        .map(|row| {
            let mut sum = zero(precision_bits);
            for column in 0..dimension {
                let mut term = Float::with_val(precision_bits, &matrix[row * dimension + column]);
                term *= &vector[column];
                sum += term;
            }
            sum
        })
        .collect()
}

fn validate_symmetric_matrix(
    matrix: &[Float],
    dimension: usize,
    name: &str,
) -> Result<(), SolverError> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::InvalidConfiguration(format!(
            "dense HP generalized {name} contains a nonfinite entry"
        )));
    }
    for row in 0..dimension {
        for column in 0..row {
            if matrix[row * dimension + column] != matrix[column * dimension + row] {
                return Err(SolverError::InvalidConfiguration(format!(
                    "dense HP generalized {name} is not exactly symmetric at ({row}, {column})"
                )));
            }
        }
    }
    Ok(())
}

fn cholesky_lower(
    metric: &[Float],
    dimension: usize,
    precision_bits: u32,
) -> Result<(Vec<Float>, Float), SolverError> {
    let mut lower = vec![zero(precision_bits); dimension * dimension];
    let mut minimum_pivot: Option<Float> = None;
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = Float::with_val(precision_bits, &metric[row * dimension + column]);
            for index in 0..column {
                let mut product = lower[row * dimension + index].clone();
                product *= &lower[column * dimension + index];
                value -= product;
            }
            if row == column {
                if !value.is_finite() || value <= 0 {
                    return Err(SolverError::NumericalBreakdown(format!(
                        "dense HP generalized metric is not positive definite at pivot {row}"
                    )));
                }
                value.sqrt_mut();
                if minimum_pivot
                    .as_ref()
                    .is_none_or(|minimum| value < *minimum)
                {
                    minimum_pivot = Some(value.clone());
                }
                lower[row * dimension + column] = value;
            } else {
                value /= &lower[column * dimension + column];
                lower[row * dimension + column] = value;
            }
        }
    }
    Ok((
        lower,
        minimum_pivot.expect("positive dimension produces a Cholesky pivot"),
    ))
}

fn forward_solve(
    lower: &[Float],
    right_hand_side: &[Float],
    dimension: usize,
    precision_bits: u32,
) -> Vec<Float> {
    let mut solution = vec![zero(precision_bits); dimension];
    for row in 0..dimension {
        let mut value = Float::with_val(precision_bits, &right_hand_side[row]);
        for column in 0..row {
            let mut product = lower[row * dimension + column].clone();
            product *= &solution[column];
            value -= product;
        }
        value /= &lower[row * dimension + row];
        solution[row] = value;
    }
    solution
}

fn backward_solve_transpose(
    lower: &[Float],
    right_hand_side: &[Float],
    dimension: usize,
    precision_bits: u32,
) -> Vec<Float> {
    let mut solution = vec![zero(precision_bits); dimension];
    for row in (0..dimension).rev() {
        let mut value = Float::with_val(precision_bits, &right_hand_side[row]);
        for column in row + 1..dimension {
            let mut product = lower[column * dimension + row].clone();
            product *= &solution[column];
            value -= product;
        }
        value /= &lower[row * dimension + row];
        solution[row] = value;
    }
    solution
}

fn whiten_operator(
    operator: &[Float],
    lower: &[Float],
    dimension: usize,
    precision_bits: u32,
) -> Vec<Float> {
    let mut left_whitened = vec![zero(precision_bits); dimension * dimension];
    for column in 0..dimension {
        let right_hand_side: Vec<Float> = (0..dimension)
            .map(|row| operator[row * dimension + column].clone())
            .collect();
        let solution = forward_solve(lower, &right_hand_side, dimension, precision_bits);
        for row in 0..dimension {
            left_whitened[row * dimension + column] = solution[row].clone();
        }
    }
    let mut whitened = vec![zero(precision_bits); dimension * dimension];
    for row in 0..dimension {
        let right_hand_side = &left_whitened[row * dimension..(row + 1) * dimension];
        let solution = forward_solve(lower, right_hand_side, dimension, precision_bits);
        for column in 0..dimension {
            whitened[row * dimension + column] = solution[column].clone();
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            let mut average = whitened[row * dimension + column].clone();
            average += &whitened[column * dimension + row];
            average /= 2u32;
            whitened[row * dimension + column] = average.clone();
            whitened[column * dimension + row] = average;
        }
    }
    whitened
}

fn canonicalize(vector: &mut [Float]) {
    if vector
        .iter()
        .find(|value| !value.is_zero())
        .is_some_and(Float::is_sign_negative)
    {
        for value in vector {
            *value = -value.clone();
        }
    }
}

/// Independent dense MPFR reference for one algebraic generalized extreme.
/// The route uses Cholesky whitening followed by the ordinary dense
/// Householder/QR eigensolver and verifies the result in the original pair.
pub fn solve_dense_generalized_whitening_hp(
    problem: &DenseGeneralizedProblemHp<'_>,
    config: &GeneralizedExtremeConfigHp,
) -> Result<DenseGeneralizedEigenpairReportHp, SolverError> {
    let target_index = match config.target {
        EigenTarget::AlgebraicSmallest => 0,
        EigenTarget::AlgebraicLargest => problem.dimension - 1,
        _ => {
            return Err(SolverError::UnsupportedTarget(
                "dense HP generalized whitening supports algebraic extremes only".to_owned(),
            ));
        }
    };
    if config.precision_bits <= 32 || config.maximum_iterations == 0 {
        return Err(SolverError::InvalidConfiguration(
            "dense HP generalized whitening requires precision above 32 bits and a positive iteration bound"
                .to_owned(),
        ));
    }
    validate_symmetric_matrix(problem.operator, problem.dimension, "operator")?;
    validate_symmetric_matrix(problem.metric, problem.dimension, "metric")?;
    let absolute_tolerance = parse_positive(
        &config.absolute_residual_tolerance,
        config.precision_bits,
        "absolute_residual_tolerance",
    )?;
    let backward_tolerance = parse_positive(
        &config.scaled_backward_error_tolerance,
        config.precision_bits,
        "scaled_backward_error_tolerance",
    )?;
    let (lower, minimum_cholesky_pivot) =
        cholesky_lower(problem.metric, problem.dimension, config.precision_bits)?;
    let whitened = whiten_operator(
        problem.operator,
        &lower,
        problem.dimension,
        config.precision_bits,
    );
    let eigenvalues = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(
        &whitened,
        problem.dimension,
        config.precision_bits,
    )
    .map_err(|error| SolverError::NumericalBreakdown(error.to_string()))?;
    let eigenvalue = eigenvalues[target_index].clone();
    let whitened_vector = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
        &whitened,
        problem.dimension,
        &eigenvalue,
        config.precision_bits,
        config.maximum_iterations,
    )
    .map_err(|error| SolverError::NumericalBreakdown(error.to_string()))?;
    let mut eigenvector = backward_solve_transpose(
        &lower,
        &whitened_vector,
        problem.dimension,
        config.precision_bits,
    );
    let mut applied_metric = matvec(
        problem.metric,
        &eigenvector,
        problem.dimension,
        config.precision_bits,
    );
    let metric_norm_squared = dot(&eigenvector, &applied_metric, config.precision_bits);
    if metric_norm_squared <= 0 || !metric_norm_squared.is_finite() {
        return Err(SolverError::NumericalBreakdown(
            "back-transformed dense HP generalized vector has nonpositive metric norm".to_owned(),
        ));
    }
    let metric_norm = metric_norm_squared.sqrt();
    for value in &mut eigenvector {
        *value /= &metric_norm;
    }
    canonicalize(&mut eigenvector);
    applied_metric = matvec(
        problem.metric,
        &eigenvector,
        problem.dimension,
        config.precision_bits,
    );
    let applied_operator = matvec(
        problem.operator,
        &eigenvector,
        problem.dimension,
        config.precision_bits,
    );
    let residual: Vec<Float> = applied_operator
        .iter()
        .zip(&applied_metric)
        .map(|(operator, metric)| {
            let mut value = metric.clone();
            value *= &eigenvalue;
            value = -value;
            value += operator;
            value
        })
        .collect();
    let residual_norm = norm(&residual, config.precision_bits);
    let operator_norm = norm(&applied_operator, config.precision_bits);
    let metric_image_norm = norm(&applied_metric, config.precision_bits);
    let mut scale = eigenvalue.clone().abs();
    scale *= metric_image_norm;
    scale += operator_norm;
    let mut relative_residual = residual_norm.clone();
    if !scale.is_zero() {
        relative_residual /= &scale;
    }
    let scaled_backward_error = relative_residual.clone();
    let mut metric_normalization_error = dot(&eigenvector, &applied_metric, config.precision_bits);
    metric_normalization_error -= 1u32;
    metric_normalization_error.abs_mut();
    let (status, termination) = if scaled_backward_error <= backward_tolerance {
        (
            ResultStatus::Converged,
            TerminationReason::BackwardErrorTolerance,
        )
    } else if residual_norm <= absolute_tolerance {
        (
            ResultStatus::Converged,
            TerminationReason::ResidualTolerance,
        )
    } else {
        (
            ResultStatus::Approximate,
            TerminationReason::MaximumIterations,
        )
    };
    let bytes_per_value = u64::from(config.precision_bits).div_ceil(8);
    let mut provenance = SolverProvenance::current_package("rug_mpfr");
    provenance.precision_bits = Some(config.precision_bits);
    let diagnostics = super::EigenpairDiagnostics {
        absolute_residual: residual_norm.clone(),
        relative_residual: relative_residual.clone(),
        scaled_backward_error: scaled_backward_error.clone(),
        orthogonality_error: metric_normalization_error.clone(),
    };
    Ok(DenseGeneralizedEigenpairReportHp {
        target: config.target.clone(),
        eigenvalue,
        eigenvector,
        residual_norm,
        relative_residual,
        scaled_backward_error,
        metric_normalization_error,
        diagnostics,
        minimum_cholesky_pivot,
        precision_bits: config.precision_bits,
        factorization_count: 2,
        estimated_peak_memory_bytes: 5u64
            .saturating_mul(problem.dimension as u64)
            .saturating_mul(problem.dimension as u64)
            .saturating_mul(bytes_per_value),
        algorithm: "dense_generalized_cholesky_whitening_householder_qr_hp".to_owned(),
        metric_validity_evidence: "strictly_positive_mpfr_cholesky_pivots".to_owned(),
        status,
        termination,
        assurance: AssuranceLevel::Computed,
        provenance,
    })
}

/// Compare matrix-free and dense-whitened MPFR generalized eigenpairs using a
/// sign-invariant metric overlap and an absolute eigenvalue tolerance.
pub fn cross_check_generalized_hp_reports(
    problem: &GeneralizedEigenProblem<'_, Float>,
    matrix_free: &super::MatrixFreeGeneralizedEigenpairReportHp,
    dense_whitening: &DenseGeneralizedEigenpairReportHp,
    tolerance: &HpCrossCheckTolerance,
) -> Result<CrossCheckedGeneralizedEigenpairHp, SolverError> {
    if matrix_free.eigenvector.len() != dense_whitening.eigenvector.len()
        || matrix_free.eigenvector.len() != problem.operator.dimension()
        || matrix_free.target != dense_whitening.target
    {
        return Err(SolverError::InvalidConfiguration(
            "HP generalized cross-check reports do not describe the same problem and target"
                .to_owned(),
        ));
    }
    if matrix_free.status != ResultStatus::Converged
        || dense_whitening.status != ResultStatus::Converged
    {
        return Err(SolverError::NonConvergence(
            "HP generalized cross-check requires two converged routes".to_owned(),
        ));
    }
    let precision_bits = matrix_free
        .eigenvalue
        .prec()
        .min(dense_whitening.eigenvalue.prec());
    let eigenvalue_tolerance = parse_positive(
        &tolerance.eigenvalue_absolute,
        precision_bits,
        "eigenvalue_absolute",
    )?;
    let overlap_tolerance = parse_positive(
        &tolerance.one_minus_overlap_squared,
        precision_bits,
        "one_minus_overlap_squared",
    )?;
    let mut eigenvalue_absolute_difference =
        Float::with_val(precision_bits, &matrix_free.eigenvalue);
    eigenvalue_absolute_difference -= &dense_whitening.eigenvalue;
    eigenvalue_absolute_difference.abs_mut();
    let mut metric_image = vec![zero(precision_bits); matrix_free.eigenvector.len()];
    problem
        .metric
        .apply(&dense_whitening.eigenvector, &mut metric_image)?;
    for value in &mut metric_image {
        *value = Float::with_val(precision_bits, &*value);
    }
    let overlap = dot(&matrix_free.eigenvector, &metric_image, precision_bits);
    let mut one_minus_metric_overlap_squared = overlap;
    one_minus_metric_overlap_squared *= &one_minus_metric_overlap_squared.clone();
    one_minus_metric_overlap_squared = -one_minus_metric_overlap_squared;
    one_minus_metric_overlap_squared += 1u32;
    one_minus_metric_overlap_squared.abs_mut();
    if eigenvalue_absolute_difference > eigenvalue_tolerance
        || one_minus_metric_overlap_squared > overlap_tolerance
    {
        return Err(SolverError::CrossCheckDisagreement(format!(
            "HP generalized routes disagree: eigenvalue difference={}, one-minus-metric-overlap-squared={}",
            eigenvalue_absolute_difference, one_minus_metric_overlap_squared
        )));
    }
    let mut matrix_free = matrix_free.clone();
    matrix_free.assurance = AssuranceLevel::CrossChecked;
    let mut dense_whitening = dense_whitening.clone();
    dense_whitening.assurance = AssuranceLevel::CrossChecked;
    Ok(CrossCheckedGeneralizedEigenpairHp {
        matrix_free,
        dense_whitening,
        eigenvalue_absolute_difference,
        one_minus_metric_overlap_squared,
        assurance: AssuranceLevel::CrossChecked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_core::DecimalLiteral;
    use xc_operator::{
        DenseSymmetricHp, LinearOperator, OperatorError, OperatorMetadata, PositiveDefiniteMetric,
        SymmetricOperator,
    };

    #[derive(Clone, Debug)]
    struct DenseMetricHp(DenseSymmetricHp);

    impl LinearOperator<Float> for DenseMetricHp {
        fn dimension(&self) -> usize {
            self.0.dimension()
        }

        fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
            self.0.apply(x, y)
        }

        fn metadata(&self) -> OperatorMetadata {
            self.0.metadata()
        }

        fn norm_bound(&self) -> Option<Float> {
            self.0.norm_bound()
        }
    }

    impl SymmetricOperator<Float> for DenseMetricHp {}
    impl PositiveDefiniteMetric<Float> for DenseMetricHp {}

    fn values(precision_bits: u32, entries: &[i32]) -> Vec<Float> {
        entries
            .iter()
            .map(|entry| Float::with_val(precision_bits, *entry))
            .collect()
    }

    fn config(target: EigenTarget) -> GeneralizedExtremeConfigHp {
        GeneralizedExtremeConfigHp {
            target,
            precision_bits: 256,
            absolute_residual_tolerance: DecimalLiteral::new("1e-40").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-40").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-40").unwrap(),
            maximum_iterations: 200,
            minimum_iterations: 2,
        }
    }

    #[test]
    fn dense_hp_whitening_recovers_both_generalized_extremes() {
        let precision = 256;
        let operator = values(precision, &[3, 1, 1, 3]);
        let metric = values(precision, &[2, 1, 1, 2]);
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 2).unwrap();
        for (target, numerator, denominator) in [
            (EigenTarget::AlgebraicSmallest, 4, 3),
            (EigenTarget::AlgebraicLargest, 2, 1),
        ] {
            let report = solve_dense_generalized_whitening_hp(&problem, &config(target)).unwrap();
            let expected = Float::with_val(precision, numerator) / denominator;
            let mut difference = report.eigenvalue.clone();
            difference -= expected;
            difference.abs_mut();
            assert!(difference < Float::with_val(precision, 1e-40));
            assert!(report.residual_norm < Float::with_val(precision, 1e-40));
            assert_eq!(report.diagnostics.absolute_residual, report.residual_norm);
            assert_eq!(
                report.diagnostics.relative_residual,
                report.relative_residual
            );
            assert_eq!(
                report.diagnostics.orthogonality_error,
                report.metric_normalization_error
            );
            assert_eq!(report.status, ResultStatus::Converged);
            assert!(report.minimum_cholesky_pivot > 0);
        }
    }

    #[test]
    fn dense_hp_whitening_rejects_indefinite_metric() {
        let precision = 128;
        let operator = values(precision, &[1, 0, 0, 2]);
        let metric = values(precision, &[1, 0, 0, -1]);
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 2).unwrap();
        let error = solve_dense_generalized_whitening_hp(
            &problem,
            &GeneralizedExtremeConfigHp {
                precision_bits: precision,
                ..config(EigenTarget::AlgebraicLargest)
            },
        )
        .unwrap_err();
        assert!(matches!(error, SolverError::NumericalBreakdown(_)));
    }

    #[test]
    fn dense_hp_whitening_reports_unmet_stopping_checks_without_overclaiming() {
        let precision = 256;
        let operator = values(precision, &[3, 1, 1, 3]);
        let metric = values(precision, &[2, 1, 1, 2]);
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 2).unwrap();
        let report = solve_dense_generalized_whitening_hp(
            &problem,
            &GeneralizedExtremeConfigHp {
                absolute_residual_tolerance: DecimalLiteral::new("1e-100").unwrap(),
                scaled_backward_error_tolerance: DecimalLiteral::new("1e-100").unwrap(),
                maximum_iterations: 1,
                ..config(EigenTarget::AlgebraicLargest)
            },
        )
        .unwrap();
        assert_eq!(report.status, ResultStatus::Approximate);
        assert_eq!(report.termination, TerminationReason::MaximumIterations);
    }

    #[test]
    fn matrix_free_and_dense_hp_generalized_routes_cross_check() {
        let precision = 256;
        let operator_data = values(precision, &[3, 1, 1, 3]);
        let metric_data = values(precision, &[2, 1, 1, 2]);
        let zero = Float::with_val(precision, 0);
        let operator =
            DenseSymmetricHp::new("crosscheck_a", 2, operator_data.clone(), precision, &zero)
                .unwrap();
        let metric = DenseMetricHp(
            DenseSymmetricHp::new("crosscheck_b", 2, metric_data.clone(), precision, &zero)
                .unwrap(),
        );
        let matrix_free_problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let solve_config = config(EigenTarget::AlgebraicLargest);
        let matrix_free = super::super::MatrixFreeGeneralizedRayleighRitzHp
            .solve(&matrix_free_problem, &solve_config)
            .unwrap();
        let dense_problem =
            DenseGeneralizedProblemHp::new(&operator_data, &metric_data, 2).unwrap();
        let dense = solve_dense_generalized_whitening_hp(&dense_problem, &solve_config).unwrap();
        let checked = cross_check_generalized_hp_reports(
            &matrix_free_problem,
            &matrix_free,
            &dense,
            &HpCrossCheckTolerance {
                eigenvalue_absolute: DecimalLiteral::new("1e-35").unwrap(),
                one_minus_overlap_squared: DecimalLiteral::new("1e-35").unwrap(),
            },
        )
        .unwrap();
        assert_eq!(checked.assurance, AssuranceLevel::CrossChecked);
        assert!(checked.eigenvalue_absolute_difference < Float::with_val(precision, 1e-35));
        assert!(checked.one_minus_metric_overlap_squared < Float::with_val(precision, 1e-35));

        let mut inconsistent = dense;
        inconsistent.eigenvalue += 1u32;
        let error = cross_check_generalized_hp_reports(
            &matrix_free_problem,
            &matrix_free,
            &inconsistent,
            &HpCrossCheckTolerance {
                eigenvalue_absolute: DecimalLiteral::new("1e-35").unwrap(),
                one_minus_overlap_squared: DecimalLiteral::new("1e-35").unwrap(),
            },
        )
        .unwrap_err();
        assert!(matches!(error, SolverError::CrossCheckDisagreement(_)));
    }
}
