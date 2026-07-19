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
pub struct BlockGeneralizedConfigHp {
    pub target: EigenTarget,
    pub precision_bits: u32,
    pub requested_eigenpairs: usize,
    pub guard_eigenpairs: usize,
    pub absolute_residual_tolerance: DecimalLiteral,
    pub scaled_backward_error_tolerance: DecimalLiteral,
    pub ritz_value_stability_tolerance: DecimalLiteral,
    pub boundary_cluster_tolerance: DecimalLiteral,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
    pub maximum_projected_sweeps: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockPreconditionerDescriptorHp {
    pub id: String,
    pub changes_only_convergence: bool,
    pub approximation_error_bound: Option<DecimalLiteral>,
}

impl BlockPreconditionerDescriptorHp {
    pub fn validate(&self, precision_bits: u32) -> Result<(), SolverError> {
        if self.id.trim().is_empty() {
            return Err(SolverError::InvalidConfiguration(
                "HP block preconditioner id must not be empty".to_owned(),
            ));
        }
        let parsed_bound = self
            .approximation_error_bound
            .as_ref()
            .map(|bound| {
                let parsed = Float::parse(bound.as_str()).map_err(|error| {
                    SolverError::InvalidConfiguration(format!(
                        "failed to parse HP block preconditioner bound: {error}"
                    ))
                })?;
                let parsed = Float::with_val(precision_bits, parsed);
                if !parsed.is_finite() || parsed < 0 {
                    return Err(SolverError::InvalidConfiguration(
                        "HP block preconditioner bound must be finite and nonnegative".to_owned(),
                    ));
                }
                Ok(parsed)
            })
            .transpose()?;
        if self.changes_only_convergence
            && parsed_bound.as_ref().is_some_and(|bound| !bound.is_zero())
        {
            return Err(SolverError::InvalidConfiguration(
                "a convergence-only HP block preconditioner cannot declare nonzero approximation error"
                    .to_owned(),
            ));
        }
        if !self.changes_only_convergence && parsed_bound.is_none() {
            return Err(SolverError::InvalidConfiguration(
                "an approximate HP block preconditioner must declare its error bound".to_owned(),
            ));
        }
        Ok(())
    }
}

pub trait BlockGeneralizedPreconditionerHp: Send + Sync {
    fn descriptor(&self) -> BlockPreconditionerDescriptorHp;
    fn apply(
        &self,
        residual: &[Float],
        output: &mut [Float],
        precision_bits: u32,
    ) -> Result<(), SolverError>;
}

#[derive(Clone, Debug)]
pub struct BlockGeneralizedEigenpairHp {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub scaled_backward_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
}

#[derive(Clone, Debug)]
pub struct GeneralizedBoundaryClusterHp {
    pub last_requested_position: usize,
    pub first_guard_position: usize,
    pub first_retained_position: usize,
    pub last_retained_position: usize,
    pub dimension: usize,
    pub requested_members: usize,
    pub lower_eigenvalue: Float,
    pub upper_eigenvalue: Float,
    pub gap: Float,
    pub basis: Vec<Vec<Float>>,
    pub projected_operator: Vec<Float>,
    pub maximum_residual_norm: Float,
}

#[derive(Clone, Debug)]
pub struct BlockGeneralizedEigenReportHp {
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    pub retained_eigenpairs: Vec<BlockGeneralizedEigenpairHp>,
    pub boundary_cluster: Option<GeneralizedBoundaryClusterHp>,
    pub maximum_metric_orthogonality_error: Float,
    pub maximum_ritz_value_stability: Float,
    pub iterations: usize,
    pub operator_applications: usize,
    pub metric_applications: usize,
    pub preconditioner_applications: usize,
    pub projected_diagonalizations: usize,
    pub maximum_trial_dimension: usize,
    pub estimated_peak_memory_bytes: u64,
    pub algorithm: String,
    pub preconditioner: Option<BlockPreconditionerDescriptorHp>,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

/// Precision-independent controls for deterministic adaptive execution of the
/// matrix-free block generalized MPFR route. Operator and metric source data
/// must retain at least `precision.maximum_bits`; each application is rounded
/// to the current attempt precision at the solver boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveBlockGeneralizedOptionsHp {
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    pub guard_eigenpairs: usize,
    pub absolute_residual_tolerance: DecimalLiteral,
    pub scaled_backward_error_tolerance: DecimalLiteral,
    pub ritz_value_stability_tolerance: DecimalLiteral,
    pub boundary_cluster_tolerance: DecimalLiteral,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
    pub maximum_projected_sweeps: usize,
    pub precision: PrecisionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockGeneralizedPrecisionAttemptHp {
    pub precision_bits: u32,
    pub status: ResultStatus,
    pub iterations: usize,
    pub operator_applications: usize,
    pub metric_applications: usize,
    pub preconditioner_applications: usize,
    pub maximum_requested_residual_norm: Option<String>,
    pub maximum_metric_orthogonality_error: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub enum AdaptiveBlockGeneralizedResultHp {
    Converged {
        result: Box<BlockGeneralizedEigenReportHp>,
        attempts: Vec<BlockGeneralizedPrecisionAttemptHp>,
    },
    Inconclusive {
        last_result: Option<Box<BlockGeneralizedEigenReportHp>>,
        attempts: Vec<BlockGeneralizedPrecisionAttemptHp>,
        reason: String,
    },
}

#[derive(Clone, Debug)]
struct BasisVectorHp {
    vector: Vec<Float>,
    applied_operator: Vec<Float>,
    applied_metric: Vec<Float>,
}

fn zero(precision_bits: u32) -> Float {
    Float::with_val(precision_bits, 0)
}

fn parse_positive(
    literal: &DecimalLiteral,
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
        let mut product = Float::with_val(precision_bits, left);
        product *= right;
        sum += product;
    }
    sum
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
                "HP block generalized operator produced a nonfinite value".to_owned(),
            ));
        }
        super::reprecision_hp_value(value, precision_bits);
    }
    Ok(output)
}

fn canonicalize(basis: &mut BasisVectorHp) {
    if basis
        .vector
        .iter()
        .find(|value| !value.is_zero())
        .is_some_and(Float::is_sign_negative)
    {
        for vector in [
            &mut basis.vector,
            &mut basis.applied_operator,
            &mut basis.applied_metric,
        ] {
            for value in vector {
                value.neg_assign();
            }
        }
    }
}

fn add_b_orthonormal_vector(
    problem: &GeneralizedEigenProblem<'_, Float>,
    mut candidate: Vec<Float>,
    basis: &mut Vec<BasisVectorHp>,
    precision_bits: u32,
    relative_rank_threshold: &Float,
    operator_applications: &mut usize,
    metric_applications: &mut usize,
) -> Result<bool, SolverError> {
    let mut applied_metric = apply(problem.metric, &candidate, precision_bits)?;
    *metric_applications += 1;
    let initial_norm_squared = dot(&candidate, &applied_metric, precision_bits);
    if !initial_norm_squared.is_finite() || initial_norm_squared <= 0 {
        return Err(SolverError::NumericalBreakdown(
            "HP block generalized candidate has nonpositive metric norm".to_owned(),
        ));
    }
    for _ in 0..2 {
        for existing in basis.iter() {
            let projection = dot(&existing.vector, &applied_metric, precision_bits);
            for index in 0..candidate.len() {
                let mut correction = existing.vector[index].clone();
                correction *= &projection;
                candidate[index] -= correction;
                let mut metric_correction = existing.applied_metric[index].clone();
                metric_correction *= &projection;
                applied_metric[index] -= metric_correction;
            }
        }
    }
    let norm_squared = dot(&candidate, &applied_metric, precision_bits);
    let mut rank_threshold = relative_rank_threshold.clone();
    rank_threshold *= initial_norm_squared;
    if norm_squared <= rank_threshold {
        return Ok(false);
    }
    if !norm_squared.is_finite() || norm_squared <= 0 {
        return Err(SolverError::NumericalBreakdown(
            "HP block generalized orthogonalization produced a nonpositive norm".to_owned(),
        ));
    }
    let scale = norm_squared.sqrt();
    for value in &mut candidate {
        *value /= &scale;
    }
    for value in &mut applied_metric {
        *value /= &scale;
    }
    let applied_operator = apply(problem.operator, &candidate, precision_bits)?;
    *operator_applications += 1;
    let mut accepted = BasisVectorHp {
        vector: candidate,
        applied_operator,
        applied_metric,
    };
    canonicalize(&mut accepted);
    basis.push(accepted);
    Ok(true)
}

fn maximum_off_diagonal(matrix: &[Float], dimension: usize, precision_bits: u32) -> Float {
    let mut maximum = zero(precision_bits);
    for row in 0..dimension {
        for column in row + 1..dimension {
            let magnitude = matrix[row * dimension + column].clone().abs();
            if magnitude > maximum {
                maximum = magnitude;
            }
        }
    }
    maximum
}

pub(crate) fn symmetric_jacobi_eigensystem(
    input: &[Float],
    dimension: usize,
    precision_bits: u32,
    maximum_sweeps: usize,
) -> Result<(Vec<Float>, Vec<Vec<Float>>), SolverError> {
    let mut matrix: Vec<Float> = input
        .iter()
        .map(|value| Float::with_val(precision_bits, value))
        .collect();
    let mut vectors = vec![zero(precision_bits); dimension * dimension];
    for index in 0..dimension {
        vectors[index * dimension + index].assign(1);
    }
    let mut scale = Float::with_val(precision_bits, 1);
    for value in &matrix {
        let magnitude = value.clone().abs();
        if magnitude > scale {
            scale = magnitude;
        }
    }
    let mut tolerance = Float::with_val(precision_bits, 2);
    tolerance = tolerance.pow(-((precision_bits as i32) - 16));
    tolerance *= scale;
    let one = Float::with_val(precision_bits, 1);
    for _ in 0..maximum_sweeps {
        if maximum_off_diagonal(&matrix, dimension, precision_bits) <= tolerance {
            break;
        }
        for p in 0..dimension.saturating_sub(1) {
            for q in p + 1..dimension {
                let apq = matrix[p * dimension + q].clone();
                if apq.clone().abs() <= tolerance {
                    continue;
                }
                let app = matrix[p * dimension + p].clone();
                let aqq = matrix[q * dimension + q].clone();
                let mut tau = aqq.clone();
                tau -= &app;
                let mut denominator = apq.clone();
                denominator *= 2u32;
                tau /= denominator;
                let mut root = tau.clone();
                root *= &tau;
                root += &one;
                root.sqrt_mut();
                let mut tangent_denominator = tau.clone().abs();
                tangent_denominator += root;
                let mut tangent = one.clone();
                tangent /= tangent_denominator;
                if tau.is_sign_negative() {
                    tangent = -tangent;
                }
                let mut cosine = tangent.clone();
                cosine *= &tangent;
                cosine += &one;
                cosine.sqrt_mut();
                cosine.recip_mut();
                let mut sine = tangent.clone();
                sine *= &cosine;
                for k in 0..dimension {
                    if k != p && k != q {
                        let akp = matrix[k * dimension + p].clone();
                        let akq = matrix[k * dimension + q].clone();
                        let mut new_kp = cosine.clone();
                        new_kp *= &akp;
                        let mut term = sine.clone();
                        term *= &akq;
                        new_kp -= term;
                        let mut new_kq = sine.clone();
                        new_kq *= akp;
                        let mut term = cosine.clone();
                        term *= akq;
                        new_kq += term;
                        matrix[k * dimension + p].assign(&new_kp);
                        matrix[p * dimension + k].assign(new_kp);
                        matrix[k * dimension + q].assign(&new_kq);
                        matrix[q * dimension + k].assign(new_kq);
                    }
                    let vkp = vectors[k * dimension + p].clone();
                    let vkq = vectors[k * dimension + q].clone();
                    let mut new_vkp = cosine.clone();
                    new_vkp *= &vkp;
                    let mut term = sine.clone();
                    term *= &vkq;
                    new_vkp -= term;
                    let mut new_vkq = sine.clone();
                    new_vkq *= vkp;
                    let mut term = cosine.clone();
                    term *= vkq;
                    new_vkq += term;
                    vectors[k * dimension + p].assign(new_vkp);
                    vectors[k * dimension + q].assign(new_vkq);
                }
                let mut diagonal_change = tangent;
                diagonal_change *= &apq;
                matrix[p * dimension + p].assign(app - &diagonal_change);
                matrix[q * dimension + q].assign(aqq + &diagonal_change);
                matrix[p * dimension + q].assign(0);
                matrix[q * dimension + p].assign(0);
            }
        }
    }
    let maximum = maximum_off_diagonal(&matrix, dimension, precision_bits);
    if maximum > tolerance {
        return Err(SolverError::NonConvergence(format!(
            "HP projected Jacobi eigensystem did not converge; maximum off-diagonal={maximum}"
        )));
    }
    let mut ordering: Vec<usize> = (0..dimension).collect();
    ordering.sort_by(|left, right| {
        matrix[*left * dimension + *left]
            .partial_cmp(&matrix[*right * dimension + *right])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues = ordering
        .iter()
        .map(|index| matrix[index * dimension + index].clone())
        .collect();
    let eigenvectors = ordering
        .iter()
        .map(|column| {
            (0..dimension)
                .map(|row| vectors[row * dimension + column].clone())
                .collect()
        })
        .collect();
    Ok((eigenvalues, eigenvectors))
}

fn linear_combination(
    basis: &[BasisVectorHp],
    coefficients: &[Float],
    precision_bits: u32,
    select: fn(&BasisVectorHp) -> &[Float],
) -> Vec<Float> {
    let dimension = basis[0].vector.len();
    let mut result = vec![zero(precision_bits); dimension];
    for (basis_vector, coefficient) in basis.iter().zip(coefficients) {
        for (output, component) in result.iter_mut().zip(select(basis_vector)) {
            let mut term = Float::with_val(precision_bits, component);
            term *= coefficient;
            *output += term;
        }
    }
    result
}

fn extract_ritz_block(
    basis: &[BasisVectorHp],
    retained: usize,
    largest: bool,
    precision_bits: u32,
    maximum_sweeps: usize,
) -> Result<(Vec<BasisVectorHp>, Vec<Float>), SolverError> {
    let projected_dimension = basis.len();
    let mut projected = vec![zero(precision_bits); projected_dimension * projected_dimension];
    for row in 0..projected_dimension {
        for column in 0..=row {
            let value = dot(
                &basis[row].vector,
                &basis[column].applied_operator,
                precision_bits,
            );
            projected[row * projected_dimension + column] = value.clone();
            projected[column * projected_dimension + row] = value;
        }
    }
    let (all_values, all_vectors) = symmetric_jacobi_eigensystem(
        &projected,
        projected_dimension,
        precision_bits,
        maximum_sweeps,
    )?;
    let selected_indices: Vec<usize> = if largest {
        (all_values.len() - retained..all_values.len())
            .rev()
            .collect()
    } else {
        (0..retained).collect()
    };
    let mut states = Vec::with_capacity(retained);
    let mut values = Vec::with_capacity(retained);
    for index in selected_indices {
        let coefficients = &all_vectors[index];
        let mut state = BasisVectorHp {
            vector: linear_combination(basis, coefficients, precision_bits, |item| &item.vector),
            applied_operator: linear_combination(basis, coefficients, precision_bits, |item| {
                &item.applied_operator
            }),
            applied_metric: linear_combination(basis, coefficients, precision_bits, |item| {
                &item.applied_metric
            }),
        };
        let metric_norm = dot(&state.vector, &state.applied_metric, precision_bits).sqrt();
        for vector in [
            &mut state.vector,
            &mut state.applied_operator,
            &mut state.applied_metric,
        ] {
            for value in vector {
                *value /= &metric_norm;
            }
        }
        canonicalize(&mut state);
        states.push(state);
        values.push(all_values[index].clone());
    }
    Ok((states, values))
}

fn metric_orthogonality_errors(
    states: &[BasisVectorHp],
    precision_bits: u32,
) -> (Vec<Float>, Float) {
    let mut errors = vec![zero(precision_bits); states.len()];
    let mut maximum = zero(precision_bits);
    for row in 0..states.len() {
        for column in 0..states.len() {
            let mut value = dot(
                &states[row].vector,
                &states[column].applied_metric,
                precision_bits,
            );
            if row == column {
                value -= 1u32;
            }
            value.abs_mut();
            if value > maximum {
                maximum = value.clone();
            }
            if value > errors[row] {
                errors[row] = value;
            }
        }
    }
    (errors, maximum)
}

#[derive(Clone, Debug, Default)]
pub struct MatrixFreeBlockGeneralizedLobpcgHp;

impl MatrixFreeBlockGeneralizedLobpcgHp {
    pub fn solve(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &BlockGeneralizedConfigHp,
    ) -> Result<BlockGeneralizedEigenReportHp, SolverError> {
        self.solve_controlled_with_preconditioner(problem, config, None, &CancellationToken::new())
    }

    pub fn solve_with_preconditioner(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &BlockGeneralizedConfigHp,
        preconditioner: &dyn BlockGeneralizedPreconditionerHp,
    ) -> Result<BlockGeneralizedEigenReportHp, SolverError> {
        self.solve_controlled_with_preconditioner(
            problem,
            config,
            Some(preconditioner),
            &CancellationToken::new(),
        )
    }

    pub fn solve_controlled(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &BlockGeneralizedConfigHp,
        cancellation: &CancellationToken,
    ) -> Result<BlockGeneralizedEigenReportHp, SolverError> {
        self.solve_controlled_with_preconditioner(problem, config, None, cancellation)
    }

    pub fn solve_controlled_with_preconditioner(
        &self,
        problem: &GeneralizedEigenProblem<'_, Float>,
        config: &BlockGeneralizedConfigHp,
        preconditioner: Option<&dyn BlockGeneralizedPreconditionerHp>,
        cancellation: &CancellationToken,
    ) -> Result<BlockGeneralizedEigenReportHp, SolverError> {
        check_solver_cancellation(cancellation)?;
        let largest = match config.target {
            EigenTarget::AlgebraicLargest => true,
            EigenTarget::AlgebraicSmallest => false,
            _ => {
                return Err(SolverError::UnsupportedTarget(
                    "HP block generalized LOBPCG supports algebraic extremes only".to_owned(),
                ));
            }
        };
        let dimension = problem.operator.dimension();
        let retained = config
            .requested_eigenpairs
            .saturating_add(config.guard_eigenpairs);
        if config.precision_bits <= 32
            || dimension == 0
            || problem.metric.dimension() != dimension
            || config.requested_eigenpairs < 2
            || config.requested_eigenpairs > dimension
            || retained > dimension
            || (config.requested_eigenpairs < dimension && config.guard_eigenpairs == 0)
            || config.maximum_iterations == 0
            || config.minimum_iterations > config.maximum_iterations
            || config.maximum_projected_sweeps == 0
        {
            return Err(SolverError::InvalidConfiguration(
                "HP block generalized LOBPCG requires matching positive dimensions, at least two requested eigenpairs, a guard for partial selection, precision above 32 bits, and valid iteration/sweep bounds"
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
        let cluster_tolerance = parse_positive(
            &config.boundary_cluster_tolerance,
            config.precision_bits,
            "boundary_cluster_tolerance",
        )?;
        let preconditioner_descriptor = preconditioner.map(|value| value.descriptor());
        if let Some(descriptor) = &preconditioner_descriptor {
            descriptor.validate(config.precision_bits)?;
        }
        let mut operator_applications = 0usize;
        let mut metric_applications = 0usize;
        let mut preconditioner_applications = 0usize;
        let relative_rank_threshold =
            Float::with_val(config.precision_bits, 2).pow(-((config.precision_bits / 2) as i32));
        let mut initial_basis = Vec::with_capacity(retained);
        for column in 0..retained {
            let candidate: Vec<Float> = (0..dimension)
                .map(|row| {
                    let base = Float::with_val(config.precision_bits, row + 1);
                    base.pow((column + 1) as u32)
                })
                .collect();
            let _ = add_b_orthonormal_vector(
                problem,
                candidate,
                &mut initial_basis,
                config.precision_bits,
                &relative_rank_threshold,
                &mut operator_applications,
                &mut metric_applications,
            )?;
        }
        for coordinate in 0..dimension {
            if initial_basis.len() >= retained {
                break;
            }
            let mut candidate = vec![zero(config.precision_bits); dimension];
            candidate[coordinate].assign(1);
            let _ = add_b_orthonormal_vector(
                problem,
                candidate,
                &mut initial_basis,
                config.precision_bits,
                &relative_rank_threshold,
                &mut operator_applications,
                &mut metric_applications,
            )?;
        }
        if initial_basis.len() < retained {
            return Err(SolverError::NumericalBreakdown(
                "HP block generalized deterministic seed did not span the retained block"
                    .to_owned(),
            ));
        }
        let (mut states, mut eigenvalues) = extract_ritz_block(
            &initial_basis,
            retained,
            largest,
            config.precision_bits,
            config.maximum_projected_sweeps,
        )?;
        let mut projected_diagonalizations = 1usize;
        let mut maximum_trial_dimension = retained;
        let mut previous_values: Option<Vec<Float>> = None;
        let mut previous_directions: Vec<Vec<Float>> = Vec::new();

        for iteration in 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            let mut residuals = Vec::with_capacity(retained);
            let mut residual_norms = Vec::with_capacity(retained);
            let mut backward_errors = Vec::with_capacity(retained);
            for (state, eigenvalue) in states.iter().zip(&eigenvalues) {
                let residual: Vec<Float> = state
                    .applied_operator
                    .iter()
                    .zip(&state.applied_metric)
                    .map(|(operator, metric)| {
                        let mut value = metric.clone();
                        value *= eigenvalue;
                        value = -value;
                        value += operator;
                        value
                    })
                    .collect();
                let residual_norm = norm(&residual, config.precision_bits);
                let operator_norm = norm(&state.applied_operator, config.precision_bits);
                let metric_norm = norm(&state.applied_metric, config.precision_bits);
                let mut scale = eigenvalue.clone().abs();
                scale *= metric_norm;
                scale += operator_norm;
                let mut backward = residual_norm.clone();
                if !scale.is_zero() {
                    backward /= scale;
                }
                residuals.push(residual);
                residual_norms.push(residual_norm);
                backward_errors.push(backward);
            }
            let maximum_stability = previous_values
                .as_ref()
                .map(|previous| {
                    let mut maximum = zero(config.precision_bits);
                    for index in 0..config.requested_eigenpairs {
                        let mut difference = eigenvalues[index].clone();
                        difference -= &previous[index];
                        difference.abs_mut();
                        let mut scale = eigenvalues[index].clone().abs();
                        let previous_abs = previous[index].clone().abs();
                        if previous_abs > scale {
                            scale = previous_abs;
                        }
                        if scale < 1 {
                            scale.assign(1);
                        }
                        difference /= scale;
                        if difference > maximum {
                            maximum = difference;
                        }
                    }
                    maximum
                })
                .unwrap_or_else(|| Float::with_val(config.precision_bits, Special::Infinity));
            let residuals_converged = (0..config.requested_eigenpairs).all(|index| {
                residual_norms[index] <= absolute_tolerance
                    || backward_errors[index] <= backward_tolerance
            });
            let converged = iteration >= config.minimum_iterations
                && residuals_converged
                && maximum_stability <= stability_tolerance;
            let boundary_cluster = if config.requested_eigenpairs < retained {
                let requested = config.requested_eigenpairs - 1;
                let guard = config.requested_eigenpairs;
                let mut gap = eigenvalues[requested].clone();
                gap -= &eigenvalues[guard];
                gap.abs_mut();
                (gap <= cluster_tolerance).then(|| {
                    let mut first_position = requested;
                    while first_position > 0 {
                        let mut adjacent_gap = eigenvalues[first_position - 1].clone();
                        adjacent_gap -= &eigenvalues[first_position];
                        adjacent_gap.abs_mut();
                        if adjacent_gap > cluster_tolerance {
                            break;
                        }
                        first_position -= 1;
                    }
                    let mut last_position = guard;
                    while last_position + 1 < retained {
                        let mut adjacent_gap = eigenvalues[last_position].clone();
                        adjacent_gap -= &eigenvalues[last_position + 1];
                        adjacent_gap.abs_mut();
                        if adjacent_gap > cluster_tolerance {
                            break;
                        }
                        last_position += 1;
                    }
                    let dimension = last_position - first_position + 1;
                    let mut lower_eigenvalue = eigenvalues[first_position].clone();
                    let mut upper_eigenvalue = lower_eigenvalue.clone();
                    let mut maximum_residual_norm = zero(config.precision_bits);
                    for position in first_position..=last_position {
                        if eigenvalues[position] < lower_eigenvalue {
                            lower_eigenvalue = eigenvalues[position].clone();
                        }
                        if eigenvalues[position] > upper_eigenvalue {
                            upper_eigenvalue = eigenvalues[position].clone();
                        }
                        if residual_norms[position] > maximum_residual_norm {
                            maximum_residual_norm = residual_norms[position].clone();
                        }
                    }
                    let basis: Vec<Vec<Float>> = states[first_position..=last_position]
                        .iter()
                        .map(|state| state.vector.clone())
                        .collect();
                    let mut projected_operator =
                        vec![zero(config.precision_bits); dimension * dimension];
                    for row in 0..dimension {
                        for column in 0..dimension {
                            projected_operator[row * dimension + column] = dot(
                                &states[first_position + row].vector,
                                &states[first_position + column].applied_operator,
                                config.precision_bits,
                            );
                        }
                    }
                    GeneralizedBoundaryClusterHp {
                        last_requested_position: requested,
                        first_guard_position: guard,
                        first_retained_position: first_position,
                        last_retained_position: last_position,
                        dimension,
                        requested_members: config.requested_eigenpairs - first_position,
                        lower_eigenvalue,
                        upper_eigenvalue,
                        gap,
                        basis,
                        projected_operator,
                        maximum_residual_norm,
                    }
                })
            } else {
                None
            };
            if converged || iteration == config.maximum_iterations {
                let (status, termination) = if converged && boundary_cluster.is_some() {
                    (
                        ResultStatus::UnresolvedCluster,
                        TerminationReason::UnresolvedCluster,
                    )
                } else if converged {
                    (
                        ResultStatus::Converged,
                        TerminationReason::BackwardErrorTolerance,
                    )
                } else {
                    (
                        ResultStatus::Approximate,
                        TerminationReason::MaximumIterations,
                    )
                };
                let (orthogonality_errors, maximum_metric_orthogonality_error) =
                    metric_orthogonality_errors(&states, config.precision_bits);
                let cluster_range = boundary_cluster.as_ref().map(|cluster| {
                    cluster.first_retained_position..=cluster.last_retained_position
                });
                let retained_eigenpairs = states
                    .into_iter()
                    .zip(eigenvalues)
                    .zip(residual_norms)
                    .zip(backward_errors)
                    .zip(orthogonality_errors)
                    .enumerate()
                    .filter(|(position, _)| {
                        !cluster_range
                            .as_ref()
                            .is_some_and(|range| range.contains(position))
                    })
                    .map(
                        |(
                            _,
                            (
                                (((state, eigenvalue), residual_norm), scaled_backward_error),
                                orthogonality_error,
                            ),
                        )| {
                            let diagnostics = super::EigenpairDiagnostics {
                                absolute_residual: residual_norm.clone(),
                                relative_residual: scaled_backward_error.clone(),
                                scaled_backward_error: scaled_backward_error.clone(),
                                orthogonality_error,
                            };
                            BlockGeneralizedEigenpairHp {
                                eigenvalue,
                                eigenvector: state.vector,
                                residual_norm,
                                scaled_backward_error,
                                diagnostics,
                            }
                        },
                    )
                    .collect();
                let bytes_per_value = u64::from(config.precision_bits).div_ceil(8);
                let live_vectors = 12u64.saturating_mul(retained as u64);
                let mut provenance = SolverProvenance::current_package("rug_mpfr");
                provenance.precision_bits = Some(config.precision_bits);
                return Ok(BlockGeneralizedEigenReportHp {
                    target: config.target.clone(),
                    requested_eigenpairs: config.requested_eigenpairs,
                    retained_eigenpairs,
                    boundary_cluster,
                    maximum_metric_orthogonality_error,
                    maximum_ritz_value_stability: maximum_stability,
                    iterations: iteration,
                    operator_applications,
                    metric_applications,
                    preconditioner_applications,
                    projected_diagonalizations,
                    maximum_trial_dimension,
                    estimated_peak_memory_bytes: live_vectors
                        .saturating_mul(dimension as u64)
                        .saturating_mul(bytes_per_value),
                    algorithm: "matrix_free_block_generalized_b_orthogonal_lobpcg_hp".to_owned(),
                    preconditioner: preconditioner_descriptor.clone(),
                    status,
                    termination,
                    assurance: AssuranceLevel::Computed,
                    provenance,
                });
            }
            if residuals_converged {
                // Preserve the converged Ritz block for one additional
                // observation when only the explicit stability check remains.
                previous_values = Some(eigenvalues.clone());
                continue;
            }
            previous_values = Some(eigenvalues.clone());
            let old_states = states;
            let search_vectors = if let Some(preconditioner) = preconditioner {
                residuals
                    .iter()
                    .map(|residual| {
                        let mut output = vec![zero(config.precision_bits); dimension];
                        preconditioner.apply(residual, &mut output, config.precision_bits)?;
                        for value in &mut output {
                            if !value.is_finite() {
                                return Err(SolverError::NumericalBreakdown(
                                    "HP block preconditioner produced a nonfinite value".to_owned(),
                                ));
                            }
                            super::reprecision_hp_value(value, config.precision_bits);
                        }
                        preconditioner_applications += 1;
                        Ok(output)
                    })
                    .collect::<Result<Vec<_>, SolverError>>()?
            } else {
                residuals
            };
            let mut trial_basis = Vec::with_capacity(retained.saturating_mul(3));
            for candidate in old_states
                .iter()
                .map(|state| state.vector.clone())
                .chain(search_vectors)
                .chain(previous_directions.iter().cloned())
            {
                let _ = add_b_orthonormal_vector(
                    problem,
                    candidate,
                    &mut trial_basis,
                    config.precision_bits,
                    &relative_rank_threshold,
                    &mut operator_applications,
                    &mut metric_applications,
                )?;
            }
            if trial_basis.len() <= retained {
                return Err(SolverError::NumericalBreakdown(
                    "HP block generalized trial space failed to expand before convergence"
                        .to_owned(),
                ));
            }
            maximum_trial_dimension = maximum_trial_dimension.max(trial_basis.len());
            let (next_states, next_values) = extract_ritz_block(
                &trial_basis,
                retained,
                largest,
                config.precision_bits,
                config.maximum_projected_sweeps,
            )?;
            projected_diagonalizations += 1;
            previous_directions = next_states
                .iter()
                .map(|next| {
                    let mut direction = next.vector.clone();
                    for old in &old_states {
                        let projection =
                            dot(&old.vector, &next.applied_metric, config.precision_bits);
                        for (value, old_component) in direction.iter_mut().zip(&old.vector) {
                            let mut correction = old_component.clone();
                            correction *= &projection;
                            *value -= correction;
                        }
                    }
                    direction
                })
                .collect();
            states = next_states;
            eigenvalues = next_values;
        }
        Err(SolverError::NonConvergence(
            "HP block generalized solver exhausted its iteration loop".to_owned(),
        ))
    }
}

fn maximum_reported_residual(report: &BlockGeneralizedEigenReportHp) -> Option<String> {
    let precision_bits = report.provenance.precision_bits?;
    let mut maximum = zero(precision_bits);
    let mut present = false;
    for pair in &report.retained_eigenpairs {
        if pair.residual_norm > maximum {
            maximum = pair.residual_norm.clone();
        }
        present = true;
    }
    if let Some(cluster) = &report.boundary_cluster {
        if cluster.maximum_residual_norm > maximum {
            maximum = cluster.maximum_residual_norm.clone();
        }
        present = true;
    }
    present.then(|| maximum.to_string())
}

/// Run the matrix-free block generalized MPFR route with deterministic
/// precision escalation and complete attempt history.
pub fn solve_matrix_free_block_generalized_adaptive_hp(
    problem: &GeneralizedEigenProblem<'_, Float>,
    options: &AdaptiveBlockGeneralizedOptionsHp,
) -> Result<AdaptiveBlockGeneralizedResultHp, SolverError> {
    solve_matrix_free_block_generalized_adaptive_with_preconditioner_hp(problem, options, None)
}

/// Adaptive block generalized solve with an optional typed preconditioner.
/// Convergence is always decided from exact operator and metric residuals;
/// the preconditioner only supplies trial directions under its descriptor.
pub fn solve_matrix_free_block_generalized_adaptive_with_preconditioner_hp(
    problem: &GeneralizedEigenProblem<'_, Float>,
    options: &AdaptiveBlockGeneralizedOptionsHp,
    preconditioner: Option<&dyn BlockGeneralizedPreconditionerHp>,
) -> Result<AdaptiveBlockGeneralizedResultHp, SolverError> {
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
            "adaptive HP block generalized precision must exceed 32 bits after guard bits"
                .to_owned(),
        ));
    }
    let mut attempts = Vec::new();
    let mut last_result = None;
    loop {
        let config = BlockGeneralizedConfigHp {
            target: options.target.clone(),
            precision_bits,
            requested_eigenpairs: options.requested_eigenpairs,
            guard_eigenpairs: options.guard_eigenpairs,
            absolute_residual_tolerance: options.absolute_residual_tolerance.clone(),
            scaled_backward_error_tolerance: options.scaled_backward_error_tolerance.clone(),
            ritz_value_stability_tolerance: options.ritz_value_stability_tolerance.clone(),
            boundary_cluster_tolerance: options.boundary_cluster_tolerance.clone(),
            maximum_iterations: options.maximum_iterations,
            minimum_iterations: options.minimum_iterations,
            maximum_projected_sweeps: options.maximum_projected_sweeps,
        };
        let outcome = MatrixFreeBlockGeneralizedLobpcgHp.solve_controlled_with_preconditioner(
            problem,
            &config,
            preconditioner,
            &CancellationToken::new(),
        );
        match outcome {
            Ok(result) => {
                let converged = result.status == ResultStatus::Converged;
                let reason = match &result.status {
                    ResultStatus::Converged => {
                        "all residual, backward-error, Ritz-stability, and boundary checks passed"
                            .to_owned()
                    }
                    ResultStatus::UnresolvedCluster => {
                        "the requested/guard boundary intersects an unresolved invariant subspace"
                            .to_owned()
                    }
                    _ => "the iteration limit was reached before all convergence checks passed"
                        .to_owned(),
                };
                attempts.push(BlockGeneralizedPrecisionAttemptHp {
                    precision_bits,
                    status: result.status.clone(),
                    iterations: result.iterations,
                    operator_applications: result.operator_applications,
                    metric_applications: result.metric_applications,
                    preconditioner_applications: result.preconditioner_applications,
                    maximum_requested_residual_norm: maximum_reported_residual(&result),
                    maximum_metric_orthogonality_error: Some(
                        result.maximum_metric_orthogonality_error.to_string(),
                    ),
                    reason,
                });
                if converged {
                    return Ok(AdaptiveBlockGeneralizedResultHp::Converged {
                        result: Box::new(result),
                        attempts,
                    });
                }
                last_result = Some(Box::new(result));
            }
            Err(error @ SolverError::InvalidConfiguration(_))
            | Err(error @ SolverError::UnsupportedTarget(_)) => return Err(error),
            Err(error) => attempts.push(BlockGeneralizedPrecisionAttemptHp {
                precision_bits,
                status: ResultStatus::InsufficientPrecision,
                iterations: 0,
                operator_applications: 0,
                metric_applications: 0,
                preconditioner_applications: 0,
                maximum_requested_residual_norm: None,
                maximum_metric_orthogonality_error: None,
                reason: error.to_string(),
            }),
        }
        let Some(next_bits) = options.precision.next_bits(precision_bits) else {
            return Ok(AdaptiveBlockGeneralizedResultHp::Inconclusive {
                last_result,
                attempts,
                reason: format!(
                    "matrix-free block generalized solve did not converge at maximum precision {}",
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

    struct IdentityPreconditionerHp {
        descriptor: BlockPreconditionerDescriptorHp,
    }

    impl BlockGeneralizedPreconditionerHp for IdentityPreconditionerHp {
        fn descriptor(&self) -> BlockPreconditionerDescriptorHp {
            self.descriptor.clone()
        }

        fn apply(
            &self,
            residual: &[Float],
            output: &mut [Float],
            precision_bits: u32,
        ) -> Result<(), SolverError> {
            if residual.len() != output.len() {
                return Err(SolverError::InvalidConfiguration(
                    "identity test preconditioner dimension mismatch".to_owned(),
                ));
            }
            for (output, residual) in output.iter_mut().zip(residual) {
                *output = Float::with_val(precision_bits, residual);
            }
            Ok(())
        }
    }

    fn diagonal(precision_bits: u32, values: &[i32]) -> Vec<Float> {
        let dimension = values.len();
        let mut matrix = vec![Float::with_val(precision_bits, 0); dimension * dimension];
        for (index, value) in values.iter().enumerate() {
            matrix[index * dimension + index].assign(*value);
        }
        matrix
    }

    fn config(target: EigenTarget) -> BlockGeneralizedConfigHp {
        BlockGeneralizedConfigHp {
            target,
            precision_bits: 192,
            requested_eigenpairs: 2,
            guard_eigenpairs: 1,
            absolute_residual_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            boundary_cluster_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 20,
            minimum_iterations: 2,
            maximum_projected_sweeps: 100,
        }
    }

    fn solve_diagonal(
        operator_diagonal: &[i32],
        metric_diagonal: &[i32],
        config: &BlockGeneralizedConfigHp,
    ) -> BlockGeneralizedEigenReportHp {
        let precision = config.precision_bits;
        let dimension = operator_diagonal.len();
        let zero = Float::with_val(precision, 0);
        let operator = DenseSymmetricHp::new(
            "block_a",
            dimension,
            diagonal(precision, operator_diagonal),
            precision,
            &zero,
        )
        .unwrap();
        let metric = DenseMetricHp(
            DenseSymmetricHp::new(
                "block_b",
                dimension,
                diagonal(precision, metric_diagonal),
                precision,
                &zero,
            )
            .unwrap(),
        );
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        MatrixFreeBlockGeneralizedLobpcgHp
            .solve(&problem, config)
            .unwrap()
    }

    #[test]
    fn block_hp_generalized_recovers_several_extremes_with_guard() {
        for (target, expected) in [
            (EigenTarget::AlgebraicLargest, [4, 3]),
            (EigenTarget::AlgebraicSmallest, [1, 2]),
        ] {
            let report = solve_diagonal(&[1, 4, 9, 16], &[1, 2, 3, 4], &config(target));
            assert_eq!(report.status, ResultStatus::Converged);
            assert_eq!(report.retained_eigenpairs.len(), 3);
            assert!(report.boundary_cluster.is_none());
            for (pair, expected) in report.retained_eigenpairs.iter().take(2).zip(expected) {
                let mut difference = pair.eigenvalue.clone();
                difference -= expected;
                difference.abs_mut();
                assert!(difference < Float::with_val(192, 1e-30));
                assert!(pair.residual_norm < Float::with_val(192, 1e-30));
                assert_eq!(pair.diagnostics.absolute_residual, pair.residual_norm);
                assert_eq!(
                    pair.diagnostics.scaled_backward_error,
                    pair.scaled_backward_error
                );
                assert!(
                    pair.diagnostics.orthogonality_error
                        <= report.maximum_metric_orthogonality_error
                );
            }
            assert!(report.maximum_metric_orthogonality_error < Float::with_val(192, 1e-30));
            assert!(report.maximum_trial_dimension <= 9);
        }
    }

    #[test]
    fn block_hp_generalized_does_not_split_guard_boundary_cluster() {
        let report = solve_diagonal(
            &[1, 2, 2, 4],
            &[1, 1, 1, 1],
            &config(EigenTarget::AlgebraicLargest),
        );
        assert_eq!(report.status, ResultStatus::UnresolvedCluster);
        assert_eq!(report.termination, TerminationReason::UnresolvedCluster);
        let cluster = report.boundary_cluster.unwrap();
        assert_eq!(cluster.last_requested_position, 1);
        assert_eq!(cluster.first_guard_position, 2);
        assert_eq!(cluster.first_retained_position, 1);
        assert_eq!(cluster.last_retained_position, 2);
        assert_eq!(cluster.dimension, 2);
        assert_eq!(cluster.requested_members, 1);
        assert_eq!(cluster.basis.len(), 2);
        assert_eq!(cluster.projected_operator.len(), 4);
        assert!(cluster.maximum_residual_norm < Float::with_val(192, 1e-30));
        assert!(cluster.gap < Float::with_val(192, 1e-30));
        assert_eq!(report.retained_eigenpairs.len(), 1);
    }

    #[test]
    fn block_hp_generalized_requires_guard_for_partial_selection() {
        let mut invalid = config(EigenTarget::AlgebraicLargest);
        invalid.guard_eigenpairs = 0;
        let precision = invalid.precision_bits;
        let zero = Float::with_val(precision, 0);
        let operator = DenseSymmetricHp::new(
            "invalid_a",
            3,
            diagonal(precision, &[1, 2, 3]),
            precision,
            &zero,
        )
        .unwrap();
        let metric = DenseMetricHp(
            DenseSymmetricHp::new(
                "invalid_b",
                3,
                diagonal(precision, &[1, 1, 1]),
                precision,
                &zero,
            )
            .unwrap(),
        );
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let error = MatrixFreeBlockGeneralizedLobpcgHp
            .solve(&problem, &invalid)
            .unwrap_err();
        assert!(matches!(error, SolverError::InvalidConfiguration(_)));
    }

    #[test]
    fn block_hp_generalized_reports_typed_preconditioner_provenance() {
        let config = config(EigenTarget::AlgebraicLargest);
        let precision = config.precision_bits;
        let zero = Float::with_val(precision, 0);
        let operator = DenseSymmetricHp::new(
            "preconditioned_a",
            4,
            diagonal(precision, &[1, 4, 9, 16]),
            precision,
            &zero,
        )
        .unwrap();
        let metric = DenseMetricHp(
            DenseSymmetricHp::new(
                "preconditioned_b",
                4,
                diagonal(precision, &[1, 2, 3, 4]),
                precision,
                &zero,
            )
            .unwrap(),
        );
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let preconditioner = IdentityPreconditionerHp {
            descriptor: BlockPreconditionerDescriptorHp {
                id: "identity_hp_test".to_owned(),
                changes_only_convergence: true,
                approximation_error_bound: None,
            },
        };
        let report = MatrixFreeBlockGeneralizedLobpcgHp
            .solve_with_preconditioner(&problem, &config, &preconditioner)
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.preconditioner_applications > 0);
        assert_eq!(report.preconditioner.unwrap().id, "identity_hp_test");
    }

    #[test]
    fn block_hp_generalized_rejects_unbounded_approximate_preconditioner() {
        let descriptor = BlockPreconditionerDescriptorHp {
            id: "unbounded".to_owned(),
            changes_only_convergence: false,
            approximation_error_bound: None,
        };
        assert!(descriptor.validate(192).is_err());
        let bounded = BlockPreconditionerDescriptorHp {
            id: "bounded".to_owned(),
            changes_only_convergence: false,
            approximation_error_bound: Some(DecimalLiteral::new("1e-20").unwrap()),
        };
        assert!(bounded.validate(192).is_ok());
    }

    #[test]
    fn adaptive_block_hp_preserves_cluster_attempts_at_precision_ceiling() {
        use xc_core::PrecisionEscalation;

        let source_precision = 192;
        let zero = Float::with_val(source_precision, 0);
        let operator = DenseSymmetricHp::new(
            "adaptive_cluster_a",
            4,
            diagonal(source_precision, &[1, 2, 2, 4]),
            source_precision,
            &zero,
        )
        .unwrap();
        let metric = DenseMetricHp(
            DenseSymmetricHp::new(
                "adaptive_cluster_b",
                4,
                diagonal(source_precision, &[1, 1, 1, 1]),
                source_precision,
                &zero,
            )
            .unwrap(),
        );
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let options = AdaptiveBlockGeneralizedOptionsHp {
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 2,
            guard_eigenpairs: 1,
            absolute_residual_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            boundary_cluster_tolerance: DecimalLiteral::new("1e-25").unwrap(),
            maximum_iterations: 20,
            minimum_iterations: 2,
            maximum_projected_sweeps: 100,
            precision: PrecisionPolicy {
                initial_bits: 64,
                maximum_bits: 192,
                guard_bits: 0,
                escalation: PrecisionEscalation::AddBits(128),
            },
        };
        let outcome = solve_matrix_free_block_generalized_adaptive_hp(&problem, &options).unwrap();
        let AdaptiveBlockGeneralizedResultHp::Inconclusive {
            last_result,
            attempts,
            ..
        } = outcome
        else {
            panic!("boundary cluster must remain typed as inconclusive");
        };
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].precision_bits, 64);
        assert_eq!(attempts[1].precision_bits, 192);
        assert_eq!(attempts[0].status, ResultStatus::InsufficientPrecision);
        assert_eq!(attempts[1].status, ResultStatus::UnresolvedCluster);
        assert!(!attempts[0].reason.is_empty());
        let last_result = last_result.expect("last cluster report must be retained");
        assert!(last_result.boundary_cluster.is_some());
        assert_eq!(last_result.provenance.precision_bits, Some(192));
    }
}
