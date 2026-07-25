use super::{
    check_solver_cancellation, hp_generalized_block::symmetric_jacobi_eigensystem,
    ShiftInvertFactorizationDescriptorHp, ShiftInvertSolveHp, SolverError,
};
use rug::{ops::Pow, Assign, Float};
use serde::{Deserialize, Serialize};
use xc_core::{
    AssuranceLevel, CancellationToken, DecimalLiteral, EigenTarget, ResultStatus, SolverProvenance,
    TerminationReason,
};
use xc_operator::SymmetricOperator;

/// Configuration for retained-factor shift-invert Krylov/Rayleigh-Ritz.
///
/// The shifted factorization is supplied separately and is never rebuilt by
/// this solver. A guard space is mandatory when the requested set does not
/// cover the complete operator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftInvertKrylovConfigHp {
    pub target: EigenTarget,
    pub precision_bits: u32,
    pub requested_eigenpairs: usize,
    pub guard_eigenpairs: usize,
    pub maximum_subspace_dimension: usize,
    pub maximum_restarts: usize,
    pub minimum_restarts: usize,
    pub maximum_projected_sweeps: usize,
    pub absolute_residual_tolerance: DecimalLiteral,
    pub scaled_backward_error_tolerance: DecimalLiteral,
    pub ritz_value_stability_tolerance: DecimalLiteral,
    pub boundary_cluster_tolerance: DecimalLiteral,
}

#[derive(Clone, Debug)]
pub struct ShiftInvertKrylovEigenpairHp {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub scaled_backward_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
}

#[derive(Clone, Debug)]
pub struct ShiftInvertKrylovBoundaryClusterHp {
    pub first_retained_position: usize,
    pub last_retained_position: usize,
    pub requested_members: usize,
    pub lower_eigenvalue: Float,
    pub upper_eigenvalue: Float,
    pub target_distance_gap: Float,
    pub maximum_residual_norm: Float,
}

#[derive(Clone, Debug)]
pub struct ShiftInvertKrylovReportHp {
    pub target: EigenTarget,
    pub factorization: ShiftInvertFactorizationDescriptorHp,
    pub requested_eigenpairs: usize,
    pub retained_eigenpairs: Vec<ShiftInvertKrylovEigenpairHp>,
    pub boundary_cluster: Option<ShiftInvertKrylovBoundaryClusterHp>,
    pub restarts: usize,
    pub shifted_solves: usize,
    pub operator_applications: usize,
    pub projected_diagonalizations: usize,
    pub maximum_subspace_dimension: usize,
    pub maximum_orthogonality_error: Float,
    pub maximum_ritz_value_stability: Float,
    pub estimated_peak_memory_bytes: u64,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

#[derive(Clone, Debug)]
struct RitzState {
    vector: Vec<Float>,
    value: Float,
    residual: Vec<Float>,
    residual_norm: Float,
    backward_error: Float,
    target_distance: Float,
}

fn zero(precision: u32) -> Float {
    Float::with_val(precision, 0)
}

fn parse_finite(value: &DecimalLiteral, precision: u32, name: &str) -> Result<Float, SolverError> {
    let parsed = Float::parse(value.as_str()).map_err(|error| {
        SolverError::InvalidConfiguration(format!("failed to parse {name}: {error}"))
    })?;
    let parsed = Float::with_val(precision, parsed);
    if !parsed.is_finite() {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be finite"
        )));
    }
    Ok(parsed)
}

fn parse_positive(
    value: &DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    let parsed = parse_finite(value, precision, name)?;
    if parsed <= 0 {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be positive"
        )));
    }
    Ok(parsed)
}

fn dot(left: &[Float], right: &[Float], precision: u32) -> Float {
    let mut sum = zero(precision);
    for (left, right) in left.iter().zip(right) {
        let mut product = Float::with_val(precision, left);
        product *= right;
        sum += product;
    }
    sum
}

fn norm(vector: &[Float], precision: u32) -> Float {
    dot(vector, vector, precision).sqrt()
}

fn add_orthonormal(candidate: &[Float], basis: &mut Vec<Vec<Float>>, precision: u32) -> bool {
    let mut candidate: Vec<Float> = candidate
        .iter()
        .map(|value| Float::with_val(precision, value))
        .collect();
    for _ in 0..2 {
        for vector in basis.iter() {
            let projection = dot(vector, &candidate, precision);
            for (value, basis_value) in candidate.iter_mut().zip(vector) {
                let mut correction = basis_value.clone();
                correction *= &projection;
                *value -= correction;
            }
        }
    }
    let length = norm(&candidate, precision);
    let threshold = Float::with_val(precision, 2).pow(-(precision as i32 / 3));
    if !length.is_finite() || length <= threshold {
        return false;
    }
    for value in &mut candidate {
        *value /= &length;
    }
    basis.push(candidate);
    true
}

fn apply(
    operator: &dyn SymmetricOperator<Float>,
    vector: &[Float],
    precision: u32,
) -> Result<Vec<Float>, SolverError> {
    let mut output = vec![zero(precision); vector.len()];
    operator.apply(vector, &mut output)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::NumericalBreakdown(
            "shift-invert Krylov operator application produced a nonfinite value".to_owned(),
        ));
    }
    for value in &mut output {
        *value = Float::with_val(precision, &*value);
    }
    Ok(output)
}

fn target_shift(
    target: &EigenTarget,
    descriptor: &ShiftInvertFactorizationDescriptorHp,
    precision: u32,
) -> Result<Float, SolverError> {
    let factor_shift = parse_finite(&descriptor.shift, precision, "factorization shift")?;
    match target {
        EigenTarget::SmallestMagnitude => {
            if !factor_shift.is_zero() {
                return Err(SolverError::UnsupportedTarget(
                    "SmallestMagnitude requires a zero-shift factorization".to_owned(),
                ));
            }
            Ok(factor_shift)
        }
        EigenTarget::ClosestTo { shift } => {
            let requested = parse_finite(shift, precision, "target shift")?;
            if requested != factor_shift {
                return Err(SolverError::InvalidConfiguration(
                    "target shift must exactly match the retained factorization shift".to_owned(),
                ));
            }
            Ok(requested)
        }
        _ => Err(SolverError::UnsupportedTarget(
            "shift-invert Krylov supports SmallestMagnitude and ClosestTo only".to_owned(),
        )),
    }
}

fn target_distance(value: &Float, shift: &Float) -> Float {
    let mut distance = value.clone();
    distance -= shift;
    distance.abs_mut();
    distance
}

fn orthogonality_errors(states: &[RitzState], precision: u32) -> (Vec<Float>, Float) {
    let mut errors = vec![zero(precision); states.len()];
    let mut maximum = zero(precision);
    for row in 0..states.len() {
        for column in 0..states.len() {
            let mut value = dot(&states[row].vector, &states[column].vector, precision);
            if row == column {
                value -= 1;
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
pub struct ShiftInvertKrylovSolverHp;

impl ShiftInvertKrylovSolverHp {
    pub fn solve(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        shifted_solver: &dyn ShiftInvertSolveHp,
        config: &ShiftInvertKrylovConfigHp,
    ) -> Result<ShiftInvertKrylovReportHp, SolverError> {
        self.solve_with_initial_basis(operator, shifted_solver, config, &[])
    }

    /// Solve using a caller-supplied, ordered starting block.
    ///
    /// Seeds are re-precisioned and deterministically orthonormalized. Invalid
    /// dimensions or nonfinite values are rejected instead of silently
    /// falling back to an unrelated start.
    pub fn solve_with_initial_basis(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        shifted_solver: &dyn ShiftInvertSolveHp,
        config: &ShiftInvertKrylovConfigHp,
        initial_basis: &[Vec<Float>],
    ) -> Result<ShiftInvertKrylovReportHp, SolverError> {
        self.solve_controlled_with_initial_basis(
            operator,
            shifted_solver,
            config,
            initial_basis,
            &CancellationToken::new(),
        )
    }

    pub fn solve_controlled_with_initial_basis(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        shifted_solver: &dyn ShiftInvertSolveHp,
        config: &ShiftInvertKrylovConfigHp,
        initial_basis: &[Vec<Float>],
        cancellation: &CancellationToken,
    ) -> Result<ShiftInvertKrylovReportHp, SolverError> {
        check_solver_cancellation(cancellation)?;
        let dimension = operator.dimension();
        let descriptor = shifted_solver.descriptor();
        let shift = target_shift(&config.target, &descriptor, config.precision_bits)?;
        let retained = config
            .requested_eigenpairs
            .saturating_add(config.guard_eigenpairs);
        if descriptor.dimension != dimension
            || descriptor.factorization_precision_bits < config.precision_bits
            || config.precision_bits <= 32
            || dimension == 0
            || config.requested_eigenpairs == 0
            || retained > dimension
            || (config.requested_eigenpairs < dimension && config.guard_eigenpairs == 0)
            || config.maximum_subspace_dimension <= retained
            || config.maximum_subspace_dimension > dimension
            || config.maximum_restarts == 0
            || config.minimum_restarts > config.maximum_restarts
            || config.maximum_projected_sweeps == 0
        {
            return Err(SolverError::InvalidConfiguration(
                "shift-invert Krylov requires matching operator/factor dimensions, adequate precision, mandatory guards, and a retained block smaller than the bounded subspace"
                    .to_owned(),
            ));
        }
        if initial_basis.iter().any(|vector| {
            vector.len() != dimension || vector.iter().any(|value| !value.is_finite())
        }) {
            return Err(SolverError::InvalidConfiguration(
                "every initial shift-invert Krylov vector must match the operator and be finite"
                    .to_owned(),
            ));
        }
        let absolute_tolerance = parse_positive(
            &config.absolute_residual_tolerance,
            config.precision_bits,
            "absolute residual tolerance",
        )?;
        let backward_tolerance = parse_positive(
            &config.scaled_backward_error_tolerance,
            config.precision_bits,
            "scaled backward-error tolerance",
        )?;
        let stability_tolerance = parse_positive(
            &config.ritz_value_stability_tolerance,
            config.precision_bits,
            "Ritz-value stability tolerance",
        )?;

        let mut retained_states: Vec<RitzState> = Vec::new();
        let mut previous_values: Option<Vec<Float>> = None;
        let mut shifted_solves = 0usize;
        let mut operator_applications = 0usize;

        for restart in 1..=config.maximum_restarts {
            check_solver_cancellation(cancellation)?;
            let mut basis = Vec::new();
            if restart == 1 {
                for seed in initial_basis {
                    let _ = add_orthonormal(seed, &mut basis, config.precision_bits);
                }
            } else {
                for state in &retained_states {
                    let _ = add_orthonormal(&state.vector, &mut basis, config.precision_bits);
                }
                if let Some(worst) = retained_states
                    .iter()
                    .take(config.requested_eigenpairs)
                    .max_by(|left, right| {
                        left.residual_norm
                            .partial_cmp(&right.residual_norm)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    let _ = add_orthonormal(&worst.residual, &mut basis, config.precision_bits);
                }
            }
            if basis.is_empty() {
                let reciprocal: Vec<Float> = (0..dimension)
                    .map(|row| {
                        let mut value = Float::with_val(config.precision_bits, row + 1);
                        value = value.recip();
                        value
                    })
                    .collect();
                let _ = add_orthonormal(&reciprocal, &mut basis, config.precision_bits);
            }
            for coordinate in 0..dimension {
                if !basis.is_empty() {
                    break;
                }
                let mut candidate = vec![zero(config.precision_bits); dimension];
                candidate[coordinate].assign(1);
                let _ = add_orthonormal(&candidate, &mut basis, config.precision_bits);
            }

            while basis.len() < config.maximum_subspace_dimension {
                check_solver_cancellation(cancellation)?;
                let mut candidate = vec![zero(config.precision_bits); dimension];
                shifted_solver.solve_shifted(
                    basis.last().expect("basis is nonempty"),
                    &mut candidate,
                    config.precision_bits,
                )?;
                shifted_solves += 1;
                if add_orthonormal(&candidate, &mut basis, config.precision_bits) {
                    continue;
                }
                let mut expanded = false;
                for coordinate in 0..dimension {
                    let mut fallback = vec![zero(config.precision_bits); dimension];
                    fallback[coordinate].assign(1);
                    if add_orthonormal(&fallback, &mut basis, config.precision_bits) {
                        expanded = true;
                        break;
                    }
                }
                if !expanded {
                    break;
                }
            }
            if basis.len() <= retained {
                return Err(SolverError::NumericalBreakdown(
                    "shift-invert Krylov subspace did not expand beyond its retained block"
                        .to_owned(),
                ));
            }

            let mut applied = Vec::with_capacity(basis.len());
            for vector in &basis {
                applied.push(apply(operator, vector, config.precision_bits)?);
                operator_applications += 1;
            }
            let subspace_dimension = basis.len();
            let mut projected =
                vec![zero(config.precision_bits); subspace_dimension * subspace_dimension];
            for row in 0..subspace_dimension {
                for column in 0..=row {
                    let value = dot(&basis[row], &applied[column], config.precision_bits);
                    projected[row * subspace_dimension + column] = value.clone();
                    projected[column * subspace_dimension + row] = value;
                }
            }
            let (values, vectors) = symmetric_jacobi_eigensystem(
                &projected,
                subspace_dimension,
                config.precision_bits,
                config.maximum_projected_sweeps,
            )?;
            let mut selected: Vec<usize> = (0..subspace_dimension).collect();
            selected.sort_by(|left, right| {
                target_distance(&values[*left], &shift)
                    .partial_cmp(&target_distance(&values[*right], &shift))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        values[*left]
                            .partial_cmp(&values[*right])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            selected.truncate(retained);

            let mut states = Vec::with_capacity(retained);
            for index in selected {
                let coefficients = &vectors[index];
                let mut vector = vec![zero(config.precision_bits); dimension];
                let mut applied_vector = vec![zero(config.precision_bits); dimension];
                for column in 0..subspace_dimension {
                    for row in 0..dimension {
                        let mut contribution = basis[column][row].clone();
                        contribution *= &coefficients[column];
                        vector[row] += contribution;
                        let mut applied_contribution = applied[column][row].clone();
                        applied_contribution *= &coefficients[column];
                        applied_vector[row] += applied_contribution;
                    }
                }
                let value = values[index].clone();
                let residual: Vec<Float> = applied_vector
                    .iter()
                    .zip(&vector)
                    .map(|(applied, component)| {
                        let mut result = component.clone();
                        result *= &value;
                        result = -result;
                        result += applied;
                        result
                    })
                    .collect();
                let residual_norm = norm(&residual, config.precision_bits);
                let mut scale = norm(&applied_vector, config.precision_bits);
                let mut value_scale = value.clone().abs();
                value_scale *= norm(&vector, config.precision_bits);
                scale += value_scale;
                let mut backward_error = residual_norm.clone();
                if !scale.is_zero() {
                    backward_error /= scale;
                }
                states.push(RitzState {
                    vector,
                    value: value.clone(),
                    residual,
                    residual_norm,
                    backward_error,
                    target_distance: target_distance(&value, &shift),
                });
            }
            let maximum_stability = previous_values
                .as_ref()
                .map(|previous| {
                    states
                        .iter()
                        .take(config.requested_eigenpairs)
                        .zip(previous)
                        .fold(
                            zero(config.precision_bits),
                            |mut maximum, (state, previous)| {
                                let mut change = state.value.clone();
                                change -= previous;
                                change.abs_mut();
                                if change > maximum {
                                    maximum = change;
                                }
                                maximum
                            },
                        )
                })
                .unwrap_or_else(|| {
                    Float::with_val(config.precision_bits, rug::float::Special::Infinity)
                });
            let residuals_converged =
                states
                    .iter()
                    .take(config.requested_eigenpairs)
                    .all(|state| {
                        state.residual_norm <= absolute_tolerance
                            || state.backward_error <= backward_tolerance
                    });
            let converged = restart >= config.minimum_restarts
                && residuals_converged
                && maximum_stability <= stability_tolerance;
            if converged || restart == config.maximum_restarts {
                return build_report(
                    config,
                    descriptor,
                    states,
                    restart,
                    shifted_solves,
                    operator_applications,
                    maximum_stability,
                    converged,
                );
            }
            previous_values = Some(
                states
                    .iter()
                    .take(config.requested_eigenpairs)
                    .map(|state| state.value.clone())
                    .collect(),
            );
            retained_states = states;
        }
        unreachable!("positive maximum_restarts returns from the loop")
    }
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    config: &ShiftInvertKrylovConfigHp,
    descriptor: ShiftInvertFactorizationDescriptorHp,
    states: Vec<RitzState>,
    restarts: usize,
    shifted_solves: usize,
    operator_applications: usize,
    maximum_stability: Float,
    converged: bool,
) -> Result<ShiftInvertKrylovReportHp, SolverError> {
    let cluster_tolerance = parse_positive(
        &config.boundary_cluster_tolerance,
        config.precision_bits,
        "boundary cluster tolerance",
    )?;
    let mut boundary_cluster = None;
    if config.requested_eigenpairs < states.len() {
        let requested = config.requested_eigenpairs - 1;
        let guard = config.requested_eigenpairs;
        let mut gap = states[guard].target_distance.clone();
        gap -= &states[requested].target_distance;
        gap.abs_mut();
        if gap <= cluster_tolerance {
            let mut first = requested;
            while first > 0 {
                let mut adjacent = states[first].target_distance.clone();
                adjacent -= &states[first - 1].target_distance;
                adjacent.abs_mut();
                if adjacent > cluster_tolerance {
                    break;
                }
                first -= 1;
            }
            let mut last = guard;
            while last + 1 < states.len() {
                let mut adjacent = states[last + 1].target_distance.clone();
                adjacent -= &states[last].target_distance;
                adjacent.abs_mut();
                if adjacent > cluster_tolerance {
                    break;
                }
                last += 1;
            }
            let mut lower = states[first].value.clone();
            let mut upper = lower.clone();
            let mut maximum_residual = zero(config.precision_bits);
            for state in &states[first..=last] {
                if state.value < lower {
                    lower = state.value.clone();
                }
                if state.value > upper {
                    upper = state.value.clone();
                }
                if state.residual_norm > maximum_residual {
                    maximum_residual = state.residual_norm.clone();
                }
            }
            boundary_cluster = Some(ShiftInvertKrylovBoundaryClusterHp {
                first_retained_position: first,
                last_retained_position: last,
                requested_members: config.requested_eigenpairs - first,
                lower_eigenvalue: lower,
                upper_eigenvalue: upper,
                target_distance_gap: gap,
                maximum_residual_norm: maximum_residual,
            });
        }
    }
    let (orthogonality_errors, maximum_orthogonality_error) =
        orthogonality_errors(&states, config.precision_bits);
    let retained_eigenpairs = states
        .iter()
        .enumerate()
        .map(|(position, state)| ShiftInvertKrylovEigenpairHp {
            eigenvalue: state.value.clone(),
            eigenvector: state.vector.clone(),
            residual_norm: state.residual_norm.clone(),
            scaled_backward_error: state.backward_error.clone(),
            diagnostics: super::EigenpairDiagnostics {
                absolute_residual: state.residual_norm.clone(),
                relative_residual: state.backward_error.clone(),
                scaled_backward_error: state.backward_error.clone(),
                orthogonality_error: orthogonality_errors[position].clone(),
            },
        })
        .collect();
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
    let bytes_per_value = u64::from(config.precision_bits).div_ceil(8);
    let dimension = states.first().map_or(0, |state| state.vector.len());
    let mut provenance = SolverProvenance::current_package("rug_mpfr");
    provenance.precision_bits = Some(config.precision_bits);
    Ok(ShiftInvertKrylovReportHp {
        target: config.target.clone(),
        factorization: descriptor,
        requested_eigenpairs: config.requested_eigenpairs,
        retained_eigenpairs,
        boundary_cluster,
        restarts,
        shifted_solves,
        operator_applications,
        projected_diagonalizations: restarts,
        maximum_subspace_dimension: config.maximum_subspace_dimension,
        maximum_orthogonality_error,
        maximum_ritz_value_stability: maximum_stability,
        estimated_peak_memory_bytes: 3u64
            .saturating_mul(config.maximum_subspace_dimension as u64)
            .saturating_mul(dimension as u64)
            .saturating_mul(bytes_per_value),
        status,
        termination,
        assurance: AssuranceLevel::Computed,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseShiftInvertFactorizationHp;
    use xc_operator::DenseSymmetricHp;

    fn diagonal(precision: u32, values: &[i32]) -> Vec<Float> {
        let n = values.len();
        let mut matrix = vec![zero(precision); n * n];
        for (index, value) in values.iter().enumerate() {
            matrix[index * n + index].assign(*value);
        }
        matrix
    }

    fn config() -> ShiftInvertKrylovConfigHp {
        ShiftInvertKrylovConfigHp {
            target: EigenTarget::SmallestMagnitude,
            precision_bits: 192,
            requested_eigenpairs: 1,
            guard_eigenpairs: 1,
            maximum_subspace_dimension: 6,
            maximum_restarts: 40,
            minimum_restarts: 2,
            maximum_projected_sweeps: 100,
            absolute_residual_tolerance: DecimalLiteral::new("1e-20").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-20").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-20").unwrap(),
            boundary_cluster_tolerance: DecimalLiteral::new("1e-20").unwrap(),
        }
    }

    #[test]
    fn recovers_smallest_magnitude_from_retained_factorization() {
        let precision = 192;
        let values = [-8, -3, 1, 4, 9, 12, 15, 20];
        let matrix = diagonal(precision, &values);
        let operator = DenseSymmetricHp::new(
            "shift-invert-krylov",
            values.len(),
            matrix.clone(),
            precision,
            &zero(precision),
        )
        .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "zero-shift",
            values.len(),
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            precision,
        )
        .unwrap();
        let report = ShiftInvertKrylovSolverHp
            .solve(&operator, &factorization, &config())
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        let pair = &report.retained_eigenpairs[0];
        let mut error = pair.eigenvalue.clone();
        error -= 1;
        error.abs_mut();
        assert!(error < Float::with_val(precision, 1e-18));
        assert!(pair.residual_norm < Float::with_val(precision, 1e-18));
        assert!(report.shifted_solves > 0);
    }

    #[test]
    fn accepts_an_explicit_continuation_seed() {
        let precision = 192;
        let values = [1, 2, 3, 5, 8, 13, 21, 34];
        let matrix = diagonal(precision, &values);
        let operator = DenseSymmetricHp::new(
            "seeded-shift-invert-krylov",
            values.len(),
            matrix.clone(),
            precision,
            &zero(precision),
        )
        .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "zero-shift",
            values.len(),
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            precision,
        )
        .unwrap();
        let mut seed = vec![zero(precision); values.len()];
        seed[0].assign(1);
        let report = ShiftInvertKrylovSolverHp
            .solve_with_initial_basis(&operator, &factorization, &config(), &[seed])
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert_eq!(report.retained_eigenpairs[0].eigenvalue, 1);
    }
}
