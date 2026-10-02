use super::{GeneralizedExtremeConfigHp, HpCrossCheckTolerance, SolverError};
use rug::{
    float::Round,
    ops::{AddAssignRound, MulAssignRound, NegAssign},
    Float,
};
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
        let result = Self {
            operator,
            metric,
            dimension,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate public records at every numerical entry boundary.
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.dimension == 0
            || self.dimension.checked_mul(self.dimension) != Some(self.operator.len())
            || self.operator.len() != self.metric.len()
        {
            return Err(SolverError::InvalidConfiguration(
                "dense HP generalized matrices require a positive checked square dimension".into(),
            ));
        }
        validate_symmetric_matrix(self.operator, self.dimension, "operator")?;
        validate_symmetric_matrix(self.metric, self.dimension, "metric")?;
        Ok(())
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
    pub stopping_evidence: super::HpResidualAcceptance,
    pub minimum_cholesky_pivot: Float,
    pub precision_bits: u32,
    /// Completed factorizations: metric Cholesky and the retained shifted recovery factor.
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
    value: &xc_core::DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    super::hp_positive_threshold(value, precision, name, rug::float::Round::Down)
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

pub(super) fn cholesky_lower(
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

pub(super) fn backward_solve_transpose(
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

pub(super) fn whiten_operator(
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
            value.neg_assign();
        }
    }
}

/// Independent dense MPFR reference for one algebraic generalized extreme.
/// The route uses Cholesky whitening followed by the ordinary dense
/// Householder/QR eigensolver and verifies the result in the original pair:
/// residuals are evaluated at the stored inputs' own precision (plus a guard),
/// so inputs finer than the working precision are never silently rounded away
/// before acceptance.
pub fn solve_dense_generalized_whitening_hp(
    problem: &DenseGeneralizedProblemHp<'_>,
    config: &GeneralizedExtremeConfigHp,
) -> Result<DenseGeneralizedEigenpairReportHp, SolverError> {
    problem.validate()?;
    let target_index = match config.target {
        EigenTarget::AlgebraicSmallest => 0,
        EigenTarget::AlgebraicLargest => problem.dimension - 1,
        _ => {
            return Err(SolverError::UnsupportedTarget(
                "dense HP generalized whitening supports algebraic extremes only".to_owned(),
            ));
        }
    };
    if !(33..=1_000_000).contains(&config.precision_bits) || config.maximum_iterations == 0 {
        return Err(SolverError::InvalidConfiguration(
            "dense HP generalized whitening requires precision in 33..=1000000 bits and a positive iteration bound"
                .to_owned(),
        ));
    }
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
    let recovered = xc_numerics::eigen::dense_symmetric_eigenpair_at_index_hp(
        &whitened,
        problem.dimension,
        target_index,
        config.precision_bits,
        config.maximum_iterations,
    )
    .map_err(|error| super::map_hp_recovery_error(error.downcast_ref(), error.to_string()))?;
    let eigenvalue = recovered.eigenvalue;
    let whitened_vector = recovered.eigenvector;
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
    // Verify in the original pair. Every product of stored inputs, the computed
    // vector and eigenvalue is formed exactly, and each residual component,
    // image component and the metric normalization is rounded once (MPFR
    // correctly rounded sum). Inputs finer than the working precision, and
    // small terms beside large ones, therefore survive cancellation.
    let verification_bits = problem
        .operator
        .iter()
        .chain(problem.metric)
        .map(Float::prec)
        .max()
        .unwrap_or(config.precision_bits)
        .max(config.precision_bits)
        .saturating_add(64);
    let dimension = problem.dimension;
    let negative_eigenvalue = Float::with_val(eigenvalue.prec(), -&eigenvalue);
    let mut applied_operator = Vec::with_capacity(dimension);
    let mut residual_upper = Vec::with_capacity(dimension);
    let mut operator_lower = Vec::with_capacity(dimension);
    let mut metric_lower = Vec::with_capacity(dimension);
    let mut metric_bounds = Vec::with_capacity(dimension);
    applied_metric = Vec::with_capacity(dimension);
    for row in 0..dimension {
        let operator_terms = (0..dimension)
            .map(|column| {
                super::exact_product(&[
                    &problem.operator[row * dimension + column],
                    &eigenvector[column],
                ])
            })
            .collect::<Result<Vec<Float>, _>>()?;
        let metric_terms = (0..dimension)
            .map(|column| {
                super::exact_product(&[
                    &problem.metric[row * dimension + column],
                    &eigenvector[column],
                ])
            })
            .collect::<Result<Vec<Float>, _>>()?;
        let shifted_terms = (0..dimension)
            .map(|column| {
                super::exact_product(&[
                    &negative_eigenvalue,
                    &problem.metric[row * dimension + column],
                    &eigenvector[column],
                ])
            })
            .collect::<Result<Vec<Float>, _>>()?;
        let residual_terms = || operator_terms.iter().chain(&shifted_terms);
        residual_upper.push(super::magnitude_bounds(residual_terms, verification_bits).1);
        operator_lower.push(super::magnitude_bounds(|| operator_terms.iter(), verification_bits).0);
        metric_lower.push(super::magnitude_bounds(|| metric_terms.iter(), verification_bits).0);
        metric_bounds.push(super::signed_bounds(
            || metric_terms.iter(),
            verification_bits,
        ));
        applied_operator.push(super::rounded_once_sum(
            operator_terms.iter(),
            verification_bits,
        ));
        applied_metric.push(super::rounded_once_sum(
            metric_terms.iter(),
            verification_bits,
        ));
    }
    // Acceptance uses an upper bound of the residual norm and a lower bound of
    // its scale: components are enclosed by directed rounding of their exact
    // sums, and norms scale every component before squaring, so neither an
    // underflowing square nor a nearest rounding at a threshold can accept.
    let breakdown = || {
        SolverError::NumericalBreakdown(
            "dense HP generalized residual bounds are outside the representable range".into(),
        )
    };
    let residual_norm = super::hp_directed_norm(&residual_upper, verification_bits, Round::Up);
    let mut scale = super::hp_directed_norm(&metric_lower, verification_bits, Round::Down);
    scale.mul_assign_round(
        Float::with_val(verification_bits, eigenvalue.abs_ref()),
        Round::Down,
    );
    scale.add_assign_round(
        super::hp_directed_norm(&operator_lower, verification_bits, Round::Down),
        Round::Down,
    );
    if !residual_norm.is_finite() || !scale.is_finite() {
        return Err(breakdown());
    }
    let relative_residual = if residual_norm.is_zero() {
        Float::with_val(verification_bits, 0)
    } else if scale.is_zero() {
        return Err(breakdown());
    } else {
        Float::with_val_round(verification_bits, &residual_norm / &scale, Round::Up).0
    };
    // Metric normalization diagnostic in O(n) memory: an upper bound of
    // |v^T B v - 1| from the directed enclosures of each (Bv)_i, multiplied
    // exactly by v_i and summed with -1 in each direction.
    let minus_one = Float::with_val(2, -1);
    let mut lower_terms = Vec::with_capacity(dimension);
    let mut upper_terms = Vec::with_capacity(dimension);
    for (component, (down, up)) in eigenvector.iter().zip(&metric_bounds) {
        let first = super::exact_product(&[component, down])?;
        let second = super::exact_product(&[component, up])?;
        if first <= second {
            lower_terms.push(first);
            upper_terms.push(second);
        } else {
            lower_terms.push(second);
            upper_terms.push(first);
        }
    }
    let lower = Float::with_val_round(
        verification_bits,
        Float::sum(lower_terms.iter().chain(std::iter::once(&minus_one))),
        Round::Down,
    )
    .0;
    let upper = Float::with_val_round(
        verification_bits,
        Float::sum(upper_terms.iter().chain(std::iter::once(&minus_one))),
        Round::Up,
    )
    .0;
    let metric_normalization_error = Float::with_val(verification_bits, lower.abs_ref())
        .max(&Float::with_val(verification_bits, upper.abs_ref()));
    // Report the original-pair diagnostics at the working precision, rounded
    // up, and decide acceptance from those reported values: rounding can only
    // make acceptance stricter.
    let report_up =
        |value: &Float| Float::with_val_round(config.precision_bits, value, Round::Up).0;
    let residual_norm = report_up(&residual_norm);
    let relative_residual = report_up(&relative_residual);
    let metric_normalization_error = report_up(&metric_normalization_error);
    let scaled_backward_error = relative_residual.clone();
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
            TerminationReason::MaximumPrecision,
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
    let stopping_evidence = super::hp_residual_acceptance(
        &applied_operator,
        &applied_metric,
        &eigenvalue,
        &residual_norm,
        &scaled_backward_error,
        &absolute_tolerance,
        &backward_tolerance,
        config.precision_bits,
    );
    Ok(DenseGeneralizedEigenpairReportHp {
        target: config.target.clone(),
        eigenvalue,
        eigenvector,
        residual_norm,
        relative_residual,
        scaled_backward_error,
        metric_normalization_error,
        diagnostics,
        stopping_evidence,
        minimum_cholesky_pivot,
        precision_bits: config.precision_bits,
        factorization_count: 2,
        estimated_peak_memory_bytes: 5u64
            .saturating_mul(problem.dimension as u64)
            .saturating_mul(problem.dimension as u64)
            .saturating_mul(bytes_per_value),
        algorithm: format!(
            "dense_generalized_cholesky_whitening_gap_bound_hp_v4:{}",
            xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS
        ),
        metric_validity_evidence: "strictly_positive_mpfr_cholesky_pivots".to_owned(),
        status,
        termination,
        assurance: AssuranceLevel::Computed,
        provenance,
    })
}

/// Compare stored generalized eigenvalues and normalized computed metric overlap.
/// Differences are upward bounds and tolerances round downward. The overlap
/// bounds arithmetic on the returned metric images; it does not bound error
/// inside an arbitrary metric callback. The caller must establish problem and
/// state identity, positive definiteness, and actual algorithm independence.
/// Agreement is not a certificate of an extreme eigenstate index. This report-only
/// comparison retains `Computed`: caller-populated algorithm names cannot prove
/// execution independence or target identity. Its vector test requires a simple,
/// separated state; matching values with noncollinear vectors return
/// `UnresolvedEigenspace`, including valid bases of a repeated eigenspace.
pub fn cross_check_generalized_hp_reports(
    problem: &GeneralizedEigenProblem<'_, Float>,
    matrix_free: &super::MatrixFreeGeneralizedEigenpairReportHp,
    dense_whitening: &DenseGeneralizedEigenpairReportHp,
    tolerance: &HpCrossCheckTolerance,
) -> Result<CrossCheckedGeneralizedEigenpairHp, SolverError> {
    super::hp_generalized_crosscheck::check(problem, matrix_free, dense_whitening, tolerance)
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
    fn residual_is_verified_against_inputs_finer_than_the_working_precision() {
        // A = [1 + 2^-100] stored at 256 bits, B = [1], solved at 64 bits: the
        // working solve sees A = [1], but the original-pair residual is 2^-100.
        let mut entry = Float::with_val(256, 1);
        entry += Float::with_val(256, Float::i_exp(1, -100));
        let operator = vec![entry];
        let metric = values(256, &[1]);
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 1).unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.precision_bits = 64;
        let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
        assert_eq!(
            report.residual_norm,
            Float::with_val(256, Float::i_exp(1, -100))
        );
        assert_eq!(report.status, ResultStatus::Approximate);
        assert!(!report.stopping_evidence.absolute_residual_passed);
        assert!(!report.stopping_evidence.scaled_backward_error_passed);
    }

    /// A = [[3,-1,x,-1],[-1,3,-1,y],[x,-1,3,-1],[-1,y,-1,3]], B = I. The solver
    /// returns lambda = 1, v = (1,1,1,1)/2, with exact residual (x,y,x,y)/2.
    fn boundary_problem(precision: u32, x: Float, y: Float) -> (Vec<Float>, Vec<Float>) {
        let mut operator = values(
            precision,
            &[3, -1, 0, -1, -1, 3, -1, 0, 0, -1, 3, -1, -1, 0, -1, 3],
        );
        operator[2] = x.clone();
        operator[8] = x;
        operator[7] = y.clone();
        operator[13] = y;
        let metric = values(precision, &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        (operator, metric)
    }

    #[test]
    fn residual_norm_is_not_understated_by_an_underflowing_square() {
        // s^2 is the least positive MPFR value and (5s/8)^2 underflows: the
        // true norm sqrt(89/32)*s exceeds a tolerance between it and sqrt(2)*s.
        let s = Float::with_val(64, Float::i_exp(1, -536_870_912));
        let x = Float::with_val(64, &s * 2u32);
        let y = Float::with_val(64, &s * 5u32) / 4u32;
        let (operator, metric) = boundary_problem(64, x, y);
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 4).unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.precision_bits = 64;
        config.absolute_residual_tolerance =
            DecimalLiteral::new("7.52166448181552759435e-161614249").unwrap();
        config.scaled_backward_error_tolerance = DecimalLiteral::new("1e-200000000").unwrap();
        let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
        assert_ne!(report.status, ResultStatus::Converged);
    }

    #[test]
    fn residual_norm_is_not_rounded_down_onto_an_exact_tolerance() {
        // ||r|| = sqrt(d^2 + delta^2) > d with d = 2^-100, delta = 2^-200.
        let d = Float::with_val(128, Float::i_exp(1, -100));
        let delta = Float::with_val(128, Float::i_exp(1, -200));
        let x = Float::with_val(128, &d + &delta);
        let y = Float::with_val(128, &d - &delta);
        let (operator, metric) = boundary_problem(128, x, y);
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 4).unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.precision_bits = 64;
        config.absolute_residual_tolerance =
            DecimalLiteral::new("7.8886090522101180541172856528278622967529296875e-31").unwrap();
        config.scaled_backward_error_tolerance = DecimalLiteral::new("1e-300").unwrap();
        let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
        assert!(report.residual_norm > d);
        assert_ne!(report.status, ResultStatus::Converged);
    }

    fn cycle(precision: u32) -> Vec<Float> {
        values(
            precision,
            &[3, -1, 0, -1, -1, 3, -1, 0, 0, -1, 3, -1, -1, 0, -1, 3],
        )
    }

    #[test]
    fn residual_products_do_not_lose_an_underflowing_partial_product() {
        // A = a*M, B = b*I + a*E (E_02 = E_20 = 1) with a = 2^-900000000 and
        // b = 2^-600000000. lambda*B_02 underflows, although the residual term
        // lambda*B_02*v_2 is representable; the residual must not vanish.
        let a = Float::with_val(64, Float::i_exp(1, -900_000_000));
        let b = Float::with_val(64, Float::i_exp(1, -600_000_000));
        let operator: Vec<Float> = cycle(64)
            .into_iter()
            .map(|entry| Float::with_val(64, entry * &a))
            .collect();
        let mut metric = values(64, &[0; 16]);
        for index in [0, 5, 10, 15] {
            metric[index] = b.clone();
        }
        metric[2] = a.clone();
        metric[8] = a.clone();
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 4).unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.precision_bits = 64;
        config.absolute_residual_tolerance = DecimalLiteral::new("1e-300000000").unwrap();
        config.scaled_backward_error_tolerance = DecimalLiteral::new("1e-100000000").unwrap();
        // Every exact residual product is representable, so the residual is
        // bounded rather than refused.
        let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
        assert!(!report.residual_norm.is_zero());
        assert_ne!(report.status, ResultStatus::Converged);
    }

    #[test]
    fn metric_normalization_bound_keeps_a_small_exact_error() {
        // A = M, B = I + 2^-200 E: v = (1,1,1,1)/2 gives v^T B v - 1 = 2^-201.
        let operator = cycle(64);
        let mut metric = values(64, &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        metric[2] = Float::with_val(64, Float::i_exp(1, -200));
        metric[8] = metric[2].clone();
        let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 4).unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.precision_bits = 64;
        let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
        assert!(report.metric_normalization_error >= Float::with_val(64, Float::i_exp(1, -201)));
    }

    #[test]
    fn shared_residual_norm_keeps_an_underflowing_component() {
        // Matrix-free route, initial candidate lambda = 0, v = e_0: the exact
        // residual (0, s, 5s/8) has norm sqrt(89)*s/8 > 9s/8 although s^2 is
        // the least positive value and (5s/8)^2 underflows.
        let p = 64;
        let s = Float::with_val(p, Float::i_exp(1, -536_870_912));
        let t = Float::with_val(p, &s * 5u32) / 8u32;
        let zero = Float::with_val(p, 0);
        let data = vec![
            zero.clone(),
            s.clone(),
            t.clone(),
            s.clone(),
            Float::with_val(p, &s * 2u32),
            zero.clone(),
            t,
            zero.clone(),
            Float::with_val(p, &s * 3u32),
        ];
        let operator = DenseSymmetricHp::new("shared norm", 3, data, p, &zero).unwrap();
        let metric = DenseMetricHp(
            DenseSymmetricHp::new(
                "identity",
                3,
                values(p, &[1, 0, 0, 0, 1, 0, 0, 0, 1]),
                p,
                &zero,
            )
            .unwrap(),
        );
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.precision_bits = p;
        config.maximum_iterations = 1;
        config.minimum_iterations = 1;
        let threshold = Float::with_val(512, &s) * 9u32 / 8u32;
        config.absolute_residual_tolerance =
            DecimalLiteral::new(threshold.to_string_radix(10, Some(170))).unwrap();
        config.scaled_backward_error_tolerance = DecimalLiteral::new("1e-200000000").unwrap();
        let seed = vec![Float::with_val(p, 1), zero.clone(), zero];
        let report = super::super::MatrixFreeGeneralizedRayleighRitzHp
            .solve_with_initial_vector(&problem, &config, &seed)
            .unwrap();
        assert!(!report.stopping_evidence.absolute_residual_passed);
        assert_ne!(report.status, ResultStatus::Converged);
    }

    #[test]
    fn residual_keeps_small_terms_beside_large_ones() {
        // Exact 64-bit inputs, B = I, tolerances 1e-200. The computed pair
        // lambda = 1, v = (1,1,1,1)/2 leaves the exact residual (d/2, 0, d/2, 0),
        // whose norm d/sqrt(2) is far above the tolerance.
        for exponent in [-200, -300, -600] {
            let d = Float::with_val(64, Float::i_exp(1, exponent));
            let entries = [3, -1, 0, -1, -1, 3, -1, 0, 0, -1, 3, -1, -1, 0, -1, 3];
            let mut operator = values(64, &entries);
            operator[2] = d.clone();
            operator[8] = d.clone();
            let metric = values(64, &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            let problem = DenseGeneralizedProblemHp::new(&operator, &metric, 4).unwrap();
            let mut config = config(EigenTarget::AlgebraicSmallest);
            config.precision_bits = 64;
            config.absolute_residual_tolerance = DecimalLiteral::new("1e-200").unwrap();
            config.scaled_backward_error_tolerance = DecimalLiteral::new("1e-200").unwrap();
            let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
            assert!(!report.residual_norm.is_zero(), "d = 2^{exponent}");
            assert_ne!(report.status, ResultStatus::Converged, "d = 2^{exponent}");
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
        assert_eq!(report.termination, TerminationReason::MaximumPrecision);
    }

    #[test]
    fn repeated_generalized_eigenvalue_requires_a_subspace_comparison() {
        let p = 256;
        let data = values(p, &[1, 0, 0, 1]);
        let operator = DenseSymmetricHp::new("identity", 2, data.clone(), p, &zero(p)).unwrap();
        let metric = DenseMetricHp(operator.clone());
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let cfg = config(EigenTarget::AlgebraicSmallest);
        let mut left = super::super::MatrixFreeGeneralizedRayleighRitzHp
            .solve(&problem, &cfg)
            .unwrap();
        // (I,I) has a two-dimensional repeated eigenspace. The directed
        // member recovery must refuse an individually separated dense vector.
        assert!(matches!(
            solve_dense_generalized_whitening_hp(
                &DenseGeneralizedProblemHp::new(&data, &data, 2).unwrap(),
                &cfg
            ),
            Err(SolverError::UnresolvedEigenspace(_))
        ));
        // Retain the report-only comparison control with independently exact
        // caller-provided vectors. A simple problem supplies the report shape;
        // all state evidence is checked below against the original (I,I).
        let simple = values(p, &[1, 0, 0, 2]);
        let mut right = solve_dense_generalized_whitening_hp(
            &DenseGeneralizedProblemHp::new(&simple, &data, 2).unwrap(),
            &cfg,
        )
        .unwrap();
        // Both vectors are exact, normalized, valid eigenvectors of (I,I).
        left.eigenvector = values(p, &[1, 0]);
        right.eigenvector = values(p, &[0, 1]);
        assert_eq!(left.eigenvalue, 1);
        assert_eq!(right.eigenvalue, 1);
        for vector in [&left.eigenvector, &right.eigenvector] {
            let mut applied = values(p, &[0, 0]);
            operator.apply(vector, &mut applied).unwrap();
            let mut metric_image = values(p, &[0, 0]);
            metric.apply(vector, &mut metric_image).unwrap();
            assert_eq!(applied, metric_image); // Exact A*x = 1*B*x.
            let norm_sq: rug::Rational = vector
                .iter()
                .map(|x| {
                    let exact = x.to_rational().unwrap();
                    exact.clone() * exact
                })
                .sum();
            assert_eq!(norm_sq, 1);
        }

        let limits = HpCrossCheckTolerance {
            eigenvalue_absolute: DecimalLiteral::new("1e-35").unwrap(),
            one_minus_overlap_squared: DecimalLiteral::new("1e-35").unwrap(),
        };
        assert!(matches!(
            cross_check_generalized_hp_reports(&problem, &left, &right, &limits),
            Err(SolverError::UnresolvedEigenspace(_))
        ));
        right.eigenvector = left.eigenvector.clone();
        let agreement =
            cross_check_generalized_hp_reports(&problem, &left, &right, &limits).unwrap();
        assert_eq!(agreement.assurance, AssuranceLevel::Computed);
        assert_eq!(agreement.matrix_free.assurance, AssuranceLevel::Computed);
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
        assert_eq!(checked.assurance, AssuranceLevel::Computed);
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

#[cfg(test)]
mod tolerance_boundary_contract {
    use super::*;
    #[test]
    fn acceptance_threshold_cannot_round_up_to_one() {
        let threshold =
            xc_core::DecimalLiteral::new("0.999999999999999999999999999999999999999999").unwrap();
        // 1 exceeds the exact requested threshold, even though nearest
        // rounding at 64 bits makes the two values indistinguishable.
        assert!(parse_positive(&threshold, 64, "acceptance tolerance").unwrap() < 1);
    }
}
