use super::{check_solver_cancellation, SolverError};
use rug::{
    float::Special,
    ops::{NegAssign, Pow},
    Assign, Float,
};
use serde::{Deserialize, Serialize};
use xc_core::{
    AssuranceLevel, CancellationToken, DecimalLiteral, EigenTarget, PrecisionPolicy, ResultStatus,
    SolverProvenance, TerminationReason,
};
use xc_operator::GeneralizedEigenProblem;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralizedExtremeConfigHp {
    pub target: EigenTarget,
    pub precision_bits: u32,
    pub absolute_residual_tolerance: DecimalLiteral,
    pub scaled_backward_error_tolerance: DecimalLiteral,
    pub ritz_value_stability_tolerance: DecimalLiteral,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
}

#[derive(Clone, Debug)]
pub struct MatrixFreeGeneralizedEigenpairReportHp {
    pub target: EigenTarget,
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub relative_residual: Float,
    pub scaled_backward_error: Float,
    pub metric_normalization_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
    pub ritz_value_stability: Float,
    pub iterations: usize,
    pub operator_applications: usize,
    pub metric_applications: usize,
    pub projected_factorizations: usize,
    pub retained_subspace_vectors: usize,
    pub estimated_peak_memory_bytes: u64,
    pub algorithm: String,
    pub seed_source: String,
    pub metric_validity_evidence: String,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

#[derive(Clone, Debug)]
struct GeneralizedIterateHp {
    vector: Vec<Float>,
    applied_operator: Vec<Float>,
    applied_metric: Vec<Float>,
}

fn zero(precision_bits: u32) -> Float {
    Float::with_val(precision_bits, 0)
}

fn parse_positive(
    value: &DecimalLiteral,
    precision_bits: u32,
    name: &str,
) -> Result<Float, SolverError> {
    let parsed = Float::parse(value.as_str()).map_err(|error| {
        SolverError::InvalidConfiguration(format!("failed to parse {name}: {error}"))
    })?;
    let parsed = Float::with_val(precision_bits, parsed);
    if !parsed.is_finite() || parsed <= 0 {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be finite and strictly positive"
        )));
    }
    Ok(parsed)
}

fn dot(left: &[Float], right: &[Float], precision_bits: u32) -> Float {
    let mut result = zero(precision_bits);
    for (left, right) in left.iter().zip(right) {
        let mut term = Float::with_val(precision_bits, left);
        term *= right;
        result += term;
    }
    result
}

fn norm(vector: &[Float], precision_bits: u32) -> Float {
    dot(vector, vector, precision_bits).sqrt()
}

fn apply<O>(operator: &O, vector: &[Float], precision_bits: u32) -> Result<Vec<Float>, SolverError>
where
    O: xc_operator::LinearOperator<Float> + ?Sized,
{
    let mut output = vec![zero(precision_bits); vector.len()];
    operator.apply(vector, &mut output)?;
    for value in &mut output {
        if !value.is_finite() {
            return Err(SolverError::NumericalBreakdown(
                "HP operator application produced a nonfinite value".to_owned(),
            ));
        }
        super::reprecision_hp_value(value, precision_bits);
    }
    Ok(output)
}

fn canonicalize(iterate: &mut GeneralizedIterateHp) {
    let negative = iterate
        .vector
        .iter()
        .find(|value| !value.is_zero())
        .is_some_and(Float::is_sign_negative);
    if negative {
        for vector in [
            &mut iterate.vector,
            &mut iterate.applied_operator,
            &mut iterate.applied_metric,
        ] {
            for value in vector {
                value.neg_assign();
            }
        }
    }
}

fn normalize_metric(
    iterate: &mut GeneralizedIterateHp,
    precision_bits: u32,
) -> Result<(), SolverError> {
    let metric_norm_squared = dot(&iterate.vector, &iterate.applied_metric, precision_bits);
    if !metric_norm_squared.is_finite() || metric_norm_squared <= 0 {
        return Err(SolverError::NumericalBreakdown(
            "HP generalized iterate has nonpositive metric norm".to_owned(),
        ));
    }
    let scale = metric_norm_squared.sqrt();
    for vector in [
        &mut iterate.vector,
        &mut iterate.applied_operator,
        &mut iterate.applied_metric,
    ] {
        for value in vector {
            *value /= &scale;
        }
    }
    canonicalize(iterate);
    Ok(())
}

fn combine(
    first: &[Float],
    second: &[Float],
    first_coefficient: &Float,
    second_coefficient: &Float,
    precision_bits: u32,
) -> Vec<Float> {
    first
        .iter()
        .zip(second)
        .map(|(first, second)| {
            let mut value = Float::with_val(precision_bits, first);
            value *= first_coefficient;
            let mut term = Float::with_val(precision_bits, second);
            term *= second_coefficient;
            value += term;
            value
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn projected_generalized_extreme_2x2(
    a00: &Float,
    a01: &Float,
    a11: &Float,
    b00: &Float,
    b01: &Float,
    b11: &Float,
    largest: bool,
    precision_bits: u32,
) -> Result<(Float, Float, Float), SolverError> {
    let mut metric_determinant = Float::with_val(precision_bits, b00);
    metric_determinant *= b11;
    let mut off_square = Float::with_val(precision_bits, b01);
    off_square *= b01;
    metric_determinant -= off_square;
    if metric_determinant <= 0 {
        return Err(SolverError::NumericalBreakdown(
            "HP projected metric is not positive definite".to_owned(),
        ));
    }

    let quadratic = metric_determinant;
    let mut linear = Float::with_val(precision_bits, a00);
    linear *= b11;
    let mut term = Float::with_val(precision_bits, a11);
    term *= b00;
    linear += &term;
    term.assign(a01);
    term *= b01;
    term *= 2u32;
    linear -= &term;
    linear = -linear;
    let mut constant = Float::with_val(precision_bits, a00);
    constant *= a11;
    term.assign(a01);
    term *= a01;
    constant -= &term;

    let mut discriminant = linear.clone();
    discriminant *= &linear;
    term.assign(&quadratic);
    term *= &constant;
    term *= 4u32;
    discriminant -= &term;
    if discriminant < 0 {
        return Err(SolverError::NumericalBreakdown(
            "HP projected generalized discriminant is negative".to_owned(),
        ));
    }
    discriminant.sqrt_mut();
    let mut eigenvalue = -linear;
    if largest {
        eigenvalue += discriminant;
    } else {
        eigenvalue -= discriminant;
    }
    let mut denominator = quadratic;
    denominator *= 2u32;
    eigenvalue /= denominator;

    let mut m00 = Float::with_val(precision_bits, b00);
    m00 *= &eigenvalue;
    m00 = -m00;
    m00 += a00;
    let mut m01 = Float::with_val(precision_bits, b01);
    m01 *= &eigenvalue;
    m01 = -m01;
    m01 += a01;
    let mut m11 = Float::with_val(precision_bits, b11);
    m11 *= &eigenvalue;
    m11 = -m11;
    m11 += a11;

    let first_norm = {
        let mut value = m01.clone();
        value *= &m01;
        term.assign(&m00);
        term *= &m00;
        value += &term;
        value
    };
    let second_norm = {
        let mut value = m11.clone();
        value *= &m11;
        term.assign(&m01);
        term *= &m01;
        value += &term;
        value
    };
    let (mut first, mut second) = if first_norm >= second_norm {
        (-m01.clone(), m00)
    } else {
        (-m11, m01)
    };
    let mut metric_norm = first.clone();
    metric_norm *= &first;
    metric_norm *= b00;
    term.assign(&first);
    term *= &second;
    term *= b01;
    term *= 2u32;
    metric_norm += &term;
    term.assign(&second);
    term *= &second;
    term *= b11;
    metric_norm += term;
    if metric_norm <= 0 {
        return Err(SolverError::NumericalBreakdown(
            "HP projected eigenvector has nonpositive metric norm".to_owned(),
        ));
    }
    metric_norm.sqrt_mut();
    first /= &metric_norm;
    second /= metric_norm;
    Ok((eigenvalue, first, second))
}

#[derive(Clone, Debug, Default)]
pub struct MatrixFreeGeneralizedRayleighRitzHp;

impl MatrixFreeGeneralizedRayleighRitzHp {
    pub fn solve(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &GeneralizedExtremeConfigHp,
    ) -> Result<MatrixFreeGeneralizedEigenpairReportHp, SolverError> {
        self.solve_controlled(problem, config, None, &CancellationToken::new())
    }

    pub fn solve_with_initial_vector(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &GeneralizedExtremeConfigHp,
        initial_vector: &[Float],
    ) -> Result<MatrixFreeGeneralizedEigenpairReportHp, SolverError> {
        self.solve_controlled(
            problem,
            config,
            Some(initial_vector),
            &CancellationToken::new(),
        )
    }

    pub fn solve_controlled(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &GeneralizedExtremeConfigHp,
        initial_vector: Option<&[Float]>,
        cancellation: &CancellationToken,
    ) -> Result<MatrixFreeGeneralizedEigenpairReportHp, SolverError> {
        check_solver_cancellation(cancellation)?;
        let largest = match config.target {
            EigenTarget::AlgebraicLargest => true,
            EigenTarget::AlgebraicSmallest => false,
            _ => {
                return Err(SolverError::UnsupportedTarget(
                    "HP generalized Rayleigh-Ritz supports algebraic extremes only".to_owned(),
                ));
            }
        };
        if config.precision_bits <= 32
            || config.maximum_iterations == 0
            || config.minimum_iterations > config.maximum_iterations
        {
            return Err(SolverError::InvalidConfiguration(
                "HP generalized solve requires precision above 32 bits and valid iteration bounds"
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
        let stability_tolerance = parse_positive(
            &config.ritz_value_stability_tolerance,
            config.precision_bits,
            "ritz_value_stability_tolerance",
        )?;
        let dimension = problem.operator.dimension();
        if dimension == 0 || problem.metric.dimension() != dimension {
            return Err(SolverError::InvalidConfiguration(
                "HP generalized operator and metric dimensions must agree and be positive"
                    .to_owned(),
            ));
        }
        let seed_source = if initial_vector.is_some() {
            "caller_hp_warm_start"
        } else {
            "deterministic_hp_seed"
        };
        let vector = initial_vector
            .map(|values| {
                values
                    .iter()
                    .map(|value| Float::with_val(config.precision_bits, value))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                (1..=dimension)
                    .map(|index| Float::with_val(config.precision_bits, index))
                    .collect()
            });
        if vector.len() != dimension || vector.iter().any(|value| !value.is_finite()) {
            return Err(SolverError::InvalidConfiguration(
                "HP generalized initial vector has invalid dimension or values".to_owned(),
            ));
        }
        let mut current = GeneralizedIterateHp {
            applied_operator: apply(problem.operator, &vector, config.precision_bits)?,
            applied_metric: apply(problem.metric, &vector, config.precision_bits)?,
            vector,
        };
        normalize_metric(&mut current, config.precision_bits)?;
        let mut previous_value: Option<Float> = None;

        for iteration in 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            let denominator = dot(
                &current.vector,
                &current.applied_metric,
                config.precision_bits,
            );
            let mut eigenvalue = dot(
                &current.vector,
                &current.applied_operator,
                config.precision_bits,
            );
            eigenvalue /= &denominator;
            let residual: Vec<Float> = current
                .applied_operator
                .iter()
                .zip(&current.applied_metric)
                .map(|(operator, metric)| {
                    let mut value = Float::with_val(config.precision_bits, metric);
                    value *= &eigenvalue;
                    value = -value;
                    value += operator;
                    value
                })
                .collect();
            let residual_norm = norm(&residual, config.precision_bits);
            let operator_norm = norm(&current.applied_operator, config.precision_bits);
            let metric_norm = norm(&current.applied_metric, config.precision_bits);
            let mut scale = eigenvalue.clone().abs();
            scale *= &metric_norm;
            scale += operator_norm;
            let mut relative_residual = residual_norm.clone();
            relative_residual /= &scale;
            let scaled_backward_error = relative_residual.clone();
            let mut metric_normalization_error = denominator;
            metric_normalization_error -= 1u32;
            metric_normalization_error.abs_mut();
            let ritz_value_stability = previous_value
                .as_ref()
                .map(|previous| {
                    let mut difference = eigenvalue.clone();
                    difference -= previous;
                    difference.abs_mut();
                    let mut stability_scale = eigenvalue.clone().abs();
                    let previous_abs = previous.clone().abs();
                    if previous_abs > stability_scale {
                        stability_scale = previous_abs;
                    }
                    if stability_scale < 1 {
                        stability_scale.assign(1);
                    }
                    difference /= stability_scale;
                    difference
                })
                .unwrap_or_else(|| Float::with_val(config.precision_bits, Special::Infinity));
            let converged = iteration >= config.minimum_iterations
                && (residual_norm <= absolute_tolerance
                    || scaled_backward_error <= backward_tolerance)
                && ritz_value_stability <= stability_tolerance;
            if converged || iteration == config.maximum_iterations {
                let (status, termination) = if converged {
                    if scaled_backward_error <= backward_tolerance {
                        (
                            ResultStatus::Converged,
                            TerminationReason::BackwardErrorTolerance,
                        )
                    } else {
                        (
                            ResultStatus::Converged,
                            TerminationReason::ResidualTolerance,
                        )
                    }
                } else {
                    (
                        ResultStatus::Approximate,
                        TerminationReason::MaximumIterations,
                    )
                };
                let mut provenance = SolverProvenance::current_package("rug_mpfr");
                provenance.precision_bits = Some(config.precision_bits);
                let diagnostics = super::EigenpairDiagnostics {
                    absolute_residual: residual_norm.clone(),
                    relative_residual: relative_residual.clone(),
                    scaled_backward_error: scaled_backward_error.clone(),
                    orthogonality_error: metric_normalization_error.clone(),
                };
                return Ok(MatrixFreeGeneralizedEigenpairReportHp {
                    target: config.target.clone(),
                    eigenvalue,
                    eigenvector: current.vector,
                    residual_norm,
                    relative_residual,
                    scaled_backward_error,
                    metric_normalization_error,
                    diagnostics,
                    ritz_value_stability,
                    iterations: iteration,
                    operator_applications: iteration,
                    metric_applications: iteration,
                    projected_factorizations: iteration.saturating_sub(1),
                    retained_subspace_vectors: 2,
                    estimated_peak_memory_bytes: 12u64
                        .saturating_mul(dimension as u64)
                        .saturating_mul(u64::from(config.precision_bits).div_ceil(8)),
                    algorithm: "matrix_free_generalized_b_orthogonal_rayleigh_ritz_hp".to_owned(),
                    seed_source: seed_source.to_owned(),
                    metric_validity_evidence:
                        "positive_definite_metric_trait_plus_positive_projected_gram_checks"
                            .to_owned(),
                    status,
                    termination,
                    assurance: AssuranceLevel::Computed,
                    provenance,
                });
            }

            let mut search_vector = residual;
            let mut search_metric = apply(problem.metric, &search_vector, config.precision_bits)?;
            let unprojected_norm = dot(&search_vector, &search_metric, config.precision_bits);
            if unprojected_norm <= 0 {
                return Err(SolverError::NumericalBreakdown(
                    "HP generalized residual has nonpositive metric norm".to_owned(),
                ));
            }
            for _ in 0..2 {
                let projection = dot(&current.vector, &search_metric, config.precision_bits);
                for index in 0..dimension {
                    let mut correction = current.vector[index].clone();
                    correction *= &projection;
                    search_vector[index] -= &correction;
                    correction.assign(&current.applied_metric[index]);
                    correction *= &projection;
                    search_metric[index] -= correction;
                }
            }
            let projected_norm = dot(&search_vector, &search_metric, config.precision_bits);
            let mut rank_threshold = Float::with_val(config.precision_bits, 2);
            rank_threshold = rank_threshold.pow(-((config.precision_bits / 2) as i32));
            rank_threshold *= &unprojected_norm;
            if projected_norm <= rank_threshold {
                return Err(SolverError::NumericalBreakdown(format!(
                    "HP generalized residual lost metric rank at iteration {iteration}"
                )));
            }
            let search_scale = projected_norm.sqrt();
            for (value, metric) in search_vector.iter_mut().zip(&mut search_metric) {
                *value /= &search_scale;
                *metric /= &search_scale;
            }
            let search_operator = apply(problem.operator, &search_vector, config.precision_bits)?;

            let a00 = dot(
                &current.vector,
                &current.applied_operator,
                config.precision_bits,
            );
            let mut a01 = dot(&current.vector, &search_operator, config.precision_bits);
            a01 += dot(
                &search_vector,
                &current.applied_operator,
                config.precision_bits,
            );
            a01 /= 2u32;
            let a11 = dot(&search_vector, &search_operator, config.precision_bits);
            let b00 = dot(
                &current.vector,
                &current.applied_metric,
                config.precision_bits,
            );
            let mut b01 = dot(&current.vector, &search_metric, config.precision_bits);
            b01 += dot(
                &search_vector,
                &current.applied_metric,
                config.precision_bits,
            );
            b01 /= 2u32;
            let b11 = dot(&search_vector, &search_metric, config.precision_bits);
            let (_, first, second) = projected_generalized_extreme_2x2(
                &a00,
                &a01,
                &a11,
                &b00,
                &b01,
                &b11,
                largest,
                config.precision_bits,
            )?;
            let mut next = GeneralizedIterateHp {
                vector: combine(
                    &current.vector,
                    &search_vector,
                    &first,
                    &second,
                    config.precision_bits,
                ),
                applied_operator: combine(
                    &current.applied_operator,
                    &search_operator,
                    &first,
                    &second,
                    config.precision_bits,
                ),
                applied_metric: combine(
                    &current.applied_metric,
                    &search_metric,
                    &first,
                    &second,
                    config.precision_bits,
                ),
            };
            normalize_metric(&mut next, config.precision_bits)?;
            previous_value = Some(eigenvalue);
            current = next;
        }
        unreachable!("positive maximum_iterations returns from the loop")
    }
}

/// Precision-independent controls for deterministic adaptive execution of the
/// matrix-free MPFR generalized route. Operator source data must retain at
/// least `precision.maximum_bits`; every action is rounded to the current
/// attempt precision at the solver boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveGeneralizedExtremeOptionsHp {
    pub target: EigenTarget,
    pub absolute_residual_tolerance: DecimalLiteral,
    pub scaled_backward_error_tolerance: DecimalLiteral,
    pub ritz_value_stability_tolerance: DecimalLiteral,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
    pub precision: PrecisionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralizedPrecisionAttemptHp {
    pub precision_bits: u32,
    pub status: ResultStatus,
    pub iterations: usize,
    pub operator_applications: usize,
    pub metric_applications: usize,
    pub residual_norm: Option<String>,
    pub scaled_backward_error: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub enum AdaptiveGeneralizedExtremeResultHp {
    Converged {
        result: Box<MatrixFreeGeneralizedEigenpairReportHp>,
        attempts: Vec<GeneralizedPrecisionAttemptHp>,
    },
    Inconclusive {
        last_result: Option<Box<MatrixFreeGeneralizedEigenpairReportHp>>,
        attempts: Vec<GeneralizedPrecisionAttemptHp>,
        reason: String,
    },
}

/// Run the matrix-free generalized MPFR route with deterministic precision
/// escalation and complete attempt history. Approximate results and numerical
/// breakdowns escalate; invalid configuration fails immediately. Reused
/// iterates are always re-applied and residual-verified at the new precision.
pub fn solve_matrix_free_generalized_adaptive_hp(
    problem: &GeneralizedEigenProblem<'_, Float>,
    options: &AdaptiveGeneralizedExtremeOptionsHp,
) -> Result<AdaptiveGeneralizedExtremeResultHp, SolverError> {
    options
        .precision
        .validate()
        .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
    let mut precision_bits = options
        .precision
        .initial_bits
        .saturating_add(options.precision.guard_bits)
        .min(options.precision.maximum_bits);
    if precision_bits <= 32 {
        return Err(SolverError::InvalidConfiguration(
            "adaptive HP generalized precision must exceed 32 bits after guard bits".to_owned(),
        ));
    }
    let mut attempts = Vec::new();
    let mut last_result = None;
    let mut warm_start: Option<Vec<Float>> = None;
    loop {
        let config = GeneralizedExtremeConfigHp {
            target: options.target.clone(),
            precision_bits,
            absolute_residual_tolerance: options.absolute_residual_tolerance.clone(),
            scaled_backward_error_tolerance: options.scaled_backward_error_tolerance.clone(),
            ritz_value_stability_tolerance: options.ritz_value_stability_tolerance.clone(),
            maximum_iterations: options.maximum_iterations,
            minimum_iterations: options.minimum_iterations,
        };
        let outcome = if let Some(vector) = warm_start.as_deref() {
            MatrixFreeGeneralizedRayleighRitzHp.solve_with_initial_vector(problem, &config, vector)
        } else {
            MatrixFreeGeneralizedRayleighRitzHp.solve(problem, &config)
        };
        match outcome {
            Ok(result) => {
                let converged = result.status == ResultStatus::Converged;
                attempts.push(GeneralizedPrecisionAttemptHp {
                    precision_bits,
                    status: result.status.clone(),
                    iterations: result.iterations,
                    operator_applications: result.operator_applications,
                    metric_applications: result.metric_applications,
                    residual_norm: Some(result.residual_norm.to_string()),
                    scaled_backward_error: Some(result.scaled_backward_error.to_string()),
                    reason: if converged {
                        "residual/backward-error and Ritz-stability checks passed".to_owned()
                    } else {
                        "iteration limit reached before all convergence checks passed".to_owned()
                    },
                });
                if converged {
                    return Ok(AdaptiveGeneralizedExtremeResultHp::Converged {
                        result: Box::new(result),
                        attempts,
                    });
                }
                warm_start = Some(result.eigenvector.clone());
                last_result = Some(Box::new(result));
            }
            Err(error @ SolverError::InvalidConfiguration(_))
            | Err(error @ SolverError::UnsupportedTarget(_)) => return Err(error),
            Err(error) => attempts.push(GeneralizedPrecisionAttemptHp {
                precision_bits,
                status: ResultStatus::InsufficientPrecision,
                iterations: 0,
                operator_applications: 0,
                metric_applications: 0,
                residual_norm: None,
                scaled_backward_error: None,
                reason: error.to_string(),
            }),
        }
        let Some(next_bits) = options.precision.next_bits(precision_bits) else {
            return Ok(AdaptiveGeneralizedExtremeResultHp::Inconclusive {
                last_result,
                attempts,
                reason: format!(
                    "matrix-free generalized solve did not converge at maximum precision {}",
                    options.precision.maximum_bits
                ),
            });
        };
        precision_bits = next_bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_operator::{
        LinearOperator, MatrixStructure, OperatorError, OperatorMetadata, PositiveDefiniteMetric,
        SymmetricOperator,
    };

    #[derive(Clone)]
    struct DiagonalHp {
        name: &'static str,
        diagonal: Vec<Float>,
    }

    impl LinearOperator<Float> for DiagonalHp {
        fn dimension(&self) -> usize {
            self.diagonal.len()
        }

        fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
            if x.len() != self.diagonal.len() || y.len() != self.diagonal.len() {
                return Err(OperatorError::DimensionMismatch {
                    expected: self.diagonal.len(),
                    actual: x.len().min(y.len()),
                });
            }
            for ((output, diagonal), input) in y.iter_mut().zip(&self.diagonal).zip(x) {
                *output = diagonal.clone();
                *output *= input;
            }
            Ok(())
        }

        fn metadata(&self) -> OperatorMetadata {
            let mut metadata = OperatorMetadata::new(
                self.name,
                self.diagonal.len(),
                MatrixStructure::Diagonal,
                "rug_mpfr",
            );
            metadata.symmetric = true;
            metadata
        }
    }

    impl SymmetricOperator<Float> for DiagonalHp {}

    struct PositiveDiagonalMetricHp(DiagonalHp);

    impl LinearOperator<Float> for PositiveDiagonalMetricHp {
        fn dimension(&self) -> usize {
            self.0.dimension()
        }

        fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
            self.0.apply(x, y)
        }

        fn metadata(&self) -> OperatorMetadata {
            self.0.metadata()
        }
    }

    impl SymmetricOperator<Float> for PositiveDiagonalMetricHp {}
    impl PositiveDefiniteMetric<Float> for PositiveDiagonalMetricHp {}

    fn config(target: EigenTarget) -> GeneralizedExtremeConfigHp {
        GeneralizedExtremeConfigHp {
            target,
            precision_bits: 256,
            absolute_residual_tolerance: DecimalLiteral::new("1e-50").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-50").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-50").unwrap(),
            maximum_iterations: 100,
            minimum_iterations: 2,
        }
    }

    #[test]
    fn hp_generalized_matrix_free_extremes_match_diagonal_quotients() {
        let precision = 256;
        let operator = DiagonalHp {
            name: "a",
            diagonal: [2, 9, 20]
                .into_iter()
                .map(|value| Float::with_val(precision, value))
                .collect(),
        };
        let metric = PositiveDiagonalMetricHp(DiagonalHp {
            name: "b",
            diagonal: [1, 3, 4]
                .into_iter()
                .map(|value| Float::with_val(precision, value))
                .collect(),
        });
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        for (target, expected) in [
            (EigenTarget::AlgebraicSmallest, 2),
            (EigenTarget::AlgebraicLargest, 5),
        ] {
            let report = MatrixFreeGeneralizedRayleighRitzHp
                .solve(&problem, &config(target))
                .unwrap();
            assert_eq!(report.status, ResultStatus::Converged);
            let mut difference = report.eigenvalue.clone();
            difference -= expected;
            difference.abs_mut();
            assert!(difference < Float::with_val(precision, 1e-45));
            assert!(report.residual_norm < Float::with_val(precision, 1e-45));
            assert_eq!(report.diagnostics.absolute_residual, report.residual_norm);
            assert_eq!(
                report.diagnostics.relative_residual,
                report.relative_residual
            );
            assert_eq!(
                report.diagnostics.orthogonality_error,
                report.metric_normalization_error
            );
            assert_eq!(report.assurance, AssuranceLevel::Computed);
            assert!(report.operator_applications < report.iterations + 2);
        }
    }

    #[test]
    fn hp_generalized_warm_start_is_residual_verified() {
        let precision = 256;
        let operator = DiagonalHp {
            name: "a",
            diagonal: [1, 4]
                .into_iter()
                .map(|value| Float::with_val(precision, value))
                .collect(),
        };
        let metric = PositiveDiagonalMetricHp(DiagonalHp {
            name: "b",
            diagonal: [1, 1]
                .into_iter()
                .map(|value| Float::with_val(precision, value))
                .collect(),
        });
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let warm_start = vec![Float::with_val(precision, 1), Float::with_val(precision, 1)];
        let report = MatrixFreeGeneralizedRayleighRitzHp
            .solve_with_initial_vector(
                &problem,
                &config(EigenTarget::AlgebraicLargest),
                &warm_start,
            )
            .unwrap();
        assert_eq!(report.seed_source, "caller_hp_warm_start");
        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.residual_norm < Float::with_val(precision, 1e-45));
    }
}
