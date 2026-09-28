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
    /// Absolute residual in operator units; acceptance is this OR the scaled
    /// backward-error tolerance, followed by the separate stability guards.
    /// Scaling the operator without scaling this tolerance changes acceptance.
    pub absolute_residual_tolerance: DecimalLiteral,
    /// Dimensionless residual normalized by the computed action norms.
    pub scaled_backward_error_tolerance: DecimalLiteral,
    /// Maximum |lambda_new-lambda_old| / max(1, |lambda_new|, |lambda_old|).
    /// This is iterate agreement, not an eigenvalue error bound.
    pub ritz_value_stability_tolerance: DecimalLiteral,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
}

/// A computed Ritz candidate. Residual convergence does not establish the
/// requested global eigenvalue index. An unvisited eigenspace or a nearly
/// invariant search plateau can hide a better target despite convergence.
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
    pub stopping_evidence: super::HpResidualAcceptance,
    pub ritz_value_stability: Float,
    /// Whether a full-dimensional projected problem established computed ordering.
    /// Even true is point evidence, not an interval/exact index certificate.
    pub target_ordering_established_by_full_space_projection: bool,
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
    value: &xc_core::DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    super::hp_positive_threshold(value, precision, name, rug::float::Round::Down)
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

fn apply<O>(operator: &O, vector: &[Float], precision_bits: u32) -> Result<Vec<Float>, SolverError>
where
    O: xc_operator::LinearOperator<Float> + ?Sized,
{
    super::hp_checked_action(operator, vector, precision_bits)
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
pub(super) fn projected_generalized_extreme_2x2(
    a00: &Float,
    a01: &Float,
    a11: &Float,
    b00: &Float,
    b01: &Float,
    b11: &Float,
    largest: bool,
    precision_bits: u32,
) -> Result<(Float, Float, Float), SolverError> {
    // B=L L^T, C=L^-1 A L^-T, x=L^-T y. Avoid the cancellation
    // in (-b +/- sqrt(b*b-4*a*c))/(2*a), including repeated eigenvalues.
    let operator = [a00.clone(), a01.clone(), a01.clone(), a11.clone()];
    let metric = [b00.clone(), b01.clone(), b01.clone(), b11.clone()];
    super::DenseGeneralizedProblemHp::new(&operator, &metric, 2)?;
    let (lower, _) = super::hp_generalized_dense::cholesky_lower(&metric, 2, precision_bits)?;
    let whitened =
        super::hp_generalized_dense::whiten_operator(&operator, &lower, 2, precision_bits);
    let spectrum = xc_numerics::eigen::dense_symmetric_eigendecomposition_jacobi_hp(
        &whitened,
        2,
        precision_bits,
        32,
    )
    .map_err(|error| SolverError::NumericalBreakdown(error.to_string()))?;
    let index = usize::from(largest);
    let y = [
        spectrum.eigenvectors[index].clone(),
        spectrum.eigenvectors[2 + index].clone(),
    ];
    let x = super::hp_generalized_dense::backward_solve_transpose(&lower, &y, 2, precision_bits);
    if x.iter().any(|v| !v.is_finite()) || x.iter().all(Float::is_zero) {
        return Err(SolverError::NumericalBreakdown(
            "HP projected eigenvector is not representable".into(),
        ));
    }
    Ok((
        spectrum.eigenvalues[index].clone(),
        x[0].clone(),
        x[1].clone(),
    ))
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
        if !(33..=1_000_000).contains(&config.precision_bits)
            || config.maximum_iterations == 0
            || config.minimum_iterations > config.maximum_iterations
        {
            return Err(SolverError::InvalidConfiguration(
                "HP generalized solve requires precision in 33..=1000000 bits and valid iteration bounds"
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
        let mut operator_applications = 1;
        let mut metric_applications = 1;
        let mut projected_factorizations = 0;

        for iteration in 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            // Refresh working images, not only the final diagnostics: otherwise
            // the iteration itself can stall on accumulated recurrence error.
            current.applied_operator =
                apply(problem.operator, &current.vector, config.precision_bits)?;
            operator_applications += 1;
            current.applied_metric = apply(problem.metric, &current.vector, config.precision_bits)?;
            metric_applications += 1;
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
            if !denominator.is_finite() || denominator <= 0 || !eigenvalue.is_finite() {
                return Err(SolverError::NumericalBreakdown(
                    "HP generalized Rayleigh quotient is invalid".into(),
                ));
            }
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
            let (residual_norm, relative_residual) = super::hp_residual_measures(
                &residual,
                &current.applied_operator,
                &current.applied_metric,
                &eigenvalue,
                config.precision_bits,
            )?;
            let scaled_backward_error = relative_residual.clone();
            let mut metric_normalization_error = denominator;
            metric_normalization_error -= 1u32;
            metric_normalization_error.abs_mut();
            let ritz_value_stability = previous_value
                .as_ref()
                .map(|previous| {
                    let mut stability_scale = eigenvalue.clone().abs();
                    let previous_abs = previous.clone().abs();
                    if previous_abs > stability_scale {
                        stability_scale = previous_abs;
                    }
                    if stability_scale < 1 {
                        stability_scale.assign(1);
                    }
                    super::hp_ritz_change(&eigenvalue, previous, Some(&stability_scale))
                })
                .unwrap_or_else(|| Float::with_val(config.precision_bits, Special::Infinity));
            // The computed residual of the stored vector carries its own
            // working-precision rounding error, comparable to the residual at
            // floor-level tolerances. Require each stopping test to hold with
            // a 4*n*2^-p relative margin scaled by ||Av|| + |lambda| ||Bv||.
            let mut rounding_margin = Float::with_val(config.precision_bits, dimension);
            rounding_margin *= 4u32;
            rounding_margin >>= config.precision_bits;
            let mut image_denominator =
                Float::with_val(config.precision_bits, eigenvalue.abs_ref());
            image_denominator *= super::hp_norm(&current.applied_metric, config.precision_bits);
            image_denominator += super::hp_norm(&current.applied_operator, config.precision_bits);
            let absolute_margin =
                Float::with_val(config.precision_bits, &rounding_margin * &image_denominator);
            let backward_met = Float::with_val(
                config.precision_bits,
                &scaled_backward_error + &rounding_margin,
            ) <= backward_tolerance;
            let residual_met =
                Float::with_val(config.precision_bits, &residual_norm + &absolute_margin)
                    <= absolute_tolerance;
            let converged = iteration >= config.minimum_iterations
                && (residual_met || backward_met)
                && ritz_value_stability <= stability_tolerance;
            if converged || iteration == config.maximum_iterations {
                let (status, termination) = if converged {
                    if backward_met {
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
                let stopping_evidence = super::hp_residual_acceptance(
                    &current.applied_operator,
                    &current.applied_metric,
                    &eigenvalue,
                    &residual_norm,
                    &scaled_backward_error,
                    &absolute_tolerance,
                    &backward_tolerance,
                    config.precision_bits,
                );
                return Ok(MatrixFreeGeneralizedEigenpairReportHp {
                    target: config.target.clone(),
                    eigenvalue,
                    eigenvector: current.vector,
                    residual_norm,
                    relative_residual,
                    scaled_backward_error,
                    metric_normalization_error,
                    diagnostics,
                    stopping_evidence,
                    ritz_value_stability,
                    target_ordering_established_by_full_space_projection: dimension == 1
                        || (dimension == 2 && projected_factorizations > 0),
                    iterations: iteration,
                    operator_applications,
                    metric_applications,
                    projected_factorizations,
                    retained_subspace_vectors: dimension.min(2),
                    estimated_peak_memory_bytes: 12u64
                        .saturating_mul(dimension as u64)
                        .saturating_mul(u64::from(config.precision_bits).div_ceil(8)),
                    algorithm:
                        "matrix_free_generalized_b_orthogonal_rayleigh_ritz_fresh_images_rounding_margin_hp_v4"
                            .to_owned(),
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

            // Stability compares successive iterates, so a converged residual
            // still takes a real step. Only a stationary iterate (zero or
            // rank-deficient search direction) is observed again unchanged.
            // This confirms a Ritz pair, not its global index.
            let residual_converged =
                residual_norm <= absolute_tolerance || scaled_backward_error <= backward_tolerance;
            if residual_converged && residual_norm.is_zero() {
                previous_value = Some(eigenvalue);
                continue;
            }
            let mut search_vector = residual;
            let mut search_metric = apply(problem.metric, &search_vector, config.precision_bits)?;
            metric_applications += 1;
            let unprojected_norm = dot(&search_vector, &search_metric, config.precision_bits);
            if !unprojected_norm.is_finite() || unprojected_norm <= 0 {
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
            if !projected_norm.is_finite()
                || !rank_threshold.is_finite()
                || projected_norm <= rank_threshold
            {
                if residual_converged && projected_norm.is_finite() && rank_threshold.is_finite() {
                    previous_value = Some(eigenvalue);
                    continue;
                }
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

            operator_applications += 1;
            projected_factorizations += 1;
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
    /// Absolute residual in operator units; acceptance is this OR the scaled
    /// backward-error tolerance, followed by the separate stability guards.
    /// Scaling the operator without scaling this tolerance changes acceptance.
    pub absolute_residual_tolerance: DecimalLiteral,
    /// Dimensionless residual normalized by the computed action norms.
    pub scaled_backward_error_tolerance: DecimalLiteral,
    /// Maximum |lambda_new-lambda_old| / max(1, |lambda_new|, |lambda_old|).
    /// This is iterate agreement, not an eigenvalue error bound.
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
/// escalation and complete attempt history. Approximate results and explicitly
/// precision-limited failures escalate; other execution failures retain their
/// reason without claiming insufficient precision. Invalid configuration fails immediately. Reused
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
    if !(33..=1_000_000).contains(&precision_bits) {
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
            | Err(error @ SolverError::UnsupportedTarget(_))
            | Err(error @ SolverError::Cancelled(_)) => return Err(error),
            Err(error @ SolverError::PrecisionExhausted(_)) => {
                attempts.push(GeneralizedPrecisionAttemptHp {
                    precision_bits,
                    status: ResultStatus::InsufficientPrecision,
                    iterations: 0,
                    operator_applications: 0,
                    metric_applications: 0,
                    residual_norm: None,
                    scaled_backward_error: None,
                    reason: error.to_string(),
                })
            }
            Err(error) => {
                let reason = error.to_string();
                attempts.push(GeneralizedPrecisionAttemptHp {
                    precision_bits,
                    status: ResultStatus::Failed,
                    iterations: 0,
                    operator_applications: 0,
                    metric_applications: 0,
                    residual_norm: None,
                    scaled_backward_error: None,
                    reason: error.to_string(),
                });
                return Ok(AdaptiveGeneralizedExtremeResultHp::Inconclusive { last_result, attempts,
                    reason: format!("execution failed without evidence that precision escalation remedies it: {reason}") });
            }
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
            assert!(report.operator_applications > report.iterations);
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

#[cfg(test)]
mod operator_precision_contract {
    use super::*;
    #[test]
    fn action_cannot_silently_promote_lower_precision_results() {
        let operator = xc_operator::DenseSymmetricHp::new(
            "fixed 32-bit action",
            2,
            [1, 0, 0, 2].map(|v| Float::with_val(32, v)).to_vec(),
            32,
            &Float::with_val(32, 0),
        )
        .unwrap();
        let vector = [Float::with_val(128, 1), Float::with_val(128, 1)];
        assert!(apply(&operator, &vector, 128).is_err());
    }
}
