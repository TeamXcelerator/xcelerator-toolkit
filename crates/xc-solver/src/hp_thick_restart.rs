use super::{
    check_solver_cancellation, hp_generalized_block::symmetric_jacobi_eigensystem, SolverError,
};
use rug::{ops::Pow, Assign, Float};
use serde::{Deserialize, Serialize};
use xc_core::{
    AssuranceLevel, CancellationToken, DecimalLiteral, EigenTarget, ResultStatus, SolverProvenance,
    TerminationReason,
};
use xc_operator::SymmetricOperator;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThickRestartLanczosConfigHp {
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
pub struct ThickRestartEigenpairHp {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub scaled_backward_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
}

#[derive(Clone, Debug)]
pub struct ThickRestartBoundaryClusterHp {
    pub first_retained_position: usize,
    pub last_retained_position: usize,
    pub requested_members: usize,
    pub dimension: usize,
    pub basis: Vec<Vec<Float>>,
    pub projected_operator: Vec<Float>,
    pub lower_eigenvalue: Float,
    pub upper_eigenvalue: Float,
    pub boundary_gap: Float,
    pub maximum_residual_norm: Float,
}

#[derive(Clone, Debug)]
pub struct ThickRestartLanczosReportHp {
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    pub retained_eigenpairs: Vec<ThickRestartEigenpairHp>,
    pub boundary_cluster: Option<ThickRestartBoundaryClusterHp>,
    pub restarts: usize,
    pub krylov_steps: usize,
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
    applied: Vec<Float>,
    value: Float,
    residual: Vec<Float>,
    residual_norm: Float,
    backward_error: Float,
}

fn zero(precision: u32) -> Float {
    Float::with_val(precision, 0)
}

fn parse_positive(
    value: &DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    let parsed = Float::parse(value.as_str()).map_err(|error| {
        SolverError::InvalidConfiguration(format!("failed to parse {name}: {error}"))
    })?;
    let parsed = Float::with_val(precision, parsed);
    if !parsed.is_finite() || parsed <= 0 {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be finite and positive"
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

fn add_orthonormal(candidate: Vec<Float>, basis: &mut Vec<Vec<Float>>, precision: u32) -> bool {
    let mut candidate = candidate;
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
    for value in &mut output {
        if !value.is_finite() {
            return Err(SolverError::NumericalBreakdown(
                "HP Lanczos operator produced a nonfinite value".to_owned(),
            ));
        }
        *value = Float::with_val(precision, &*value);
    }
    Ok(output)
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
pub struct ThickRestartLanczosHp;

impl ThickRestartLanczosHp {
    pub fn solve(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        config: &ThickRestartLanczosConfigHp,
    ) -> Result<ThickRestartLanczosReportHp, SolverError> {
        self.solve_controlled(operator, config, &CancellationToken::new())
    }

    pub fn solve_controlled(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        config: &ThickRestartLanczosConfigHp,
        cancellation: &CancellationToken,
    ) -> Result<ThickRestartLanczosReportHp, SolverError> {
        check_solver_cancellation(cancellation)?;
        let largest = match config.target {
            EigenTarget::AlgebraicLargest => true,
            EigenTarget::AlgebraicSmallest => false,
            _ => {
                return Err(SolverError::UnsupportedTarget(
                    "HP thick-restart Lanczos supports algebraic extremes only".to_owned(),
                ))
            }
        };
        let dimension = operator.dimension();
        let retained = config
            .requested_eigenpairs
            .saturating_add(config.guard_eigenpairs);
        if config.precision_bits <= 32
            || dimension == 0
            || retained > dimension
            || config.requested_eigenpairs == 0
            || (config.requested_eigenpairs < dimension && config.guard_eigenpairs == 0)
            || config.maximum_subspace_dimension <= retained
            || config.maximum_subspace_dimension > dimension
            || config.maximum_restarts == 0
            || config.minimum_restarts > config.maximum_restarts
            || config.maximum_projected_sweeps == 0
        {
            return Err(SolverError::InvalidConfiguration(
                "HP thick-restart Lanczos requires valid counts, mandatory guards, a retained block smaller than the bounded Krylov subspace, precision above 32 bits, and valid restart limits"
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
        let mut operator_applications = 0usize;
        let mut krylov_steps = 0usize;

        for restart in 1..=config.maximum_restarts {
            check_solver_cancellation(cancellation)?;
            let mut basis: Vec<Vec<Float>> = retained_states
                .iter()
                .map(|state| state.vector.clone())
                .collect();
            let continuation = retained_states
                .iter()
                .take(config.requested_eigenpairs)
                .max_by(|left, right| {
                    left.residual_norm
                        .partial_cmp(&right.residual_norm)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|state| state.residual.clone())
                .unwrap_or_else(|| {
                    (0..dimension)
                        .map(|row| {
                            let mut value = Float::with_val(config.precision_bits, row + 1);
                            value = value.recip();
                            value
                        })
                        .collect()
                });
            let _ = add_orthonormal(continuation, &mut basis, config.precision_bits);
            for coordinate in 0..dimension {
                if basis.len() > retained_states.len() {
                    break;
                }
                let mut candidate = vec![zero(config.precision_bits); dimension];
                candidate[coordinate].assign(1);
                let _ = add_orthonormal(candidate, &mut basis, config.precision_bits);
            }
            while basis.len() < config.maximum_subspace_dimension {
                check_solver_cancellation(cancellation)?;
                let candidate = apply(
                    operator,
                    basis.last().expect("basis is nonempty"),
                    config.precision_bits,
                )?;
                operator_applications += 1;
                if !add_orthonormal(candidate, &mut basis, config.precision_bits) {
                    let mut expanded = false;
                    for coordinate in 0..dimension {
                        let mut fallback = vec![zero(config.precision_bits); dimension];
                        fallback[coordinate].assign(1);
                        if add_orthonormal(fallback, &mut basis, config.precision_bits) {
                            expanded = true;
                            break;
                        }
                    }
                    if !expanded {
                        break;
                    }
                }
                krylov_steps += 1;
            }
            if basis.len() <= retained {
                return Err(SolverError::NumericalBreakdown(
                    "HP thick-restart Krylov subspace did not expand beyond the retained block"
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
            let selected: Vec<usize> = if largest {
                (subspace_dimension - retained..subspace_dimension)
                    .rev()
                    .collect()
            } else {
                (0..retained).collect()
            };
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
                    applied: applied_vector,
                    value,
                    residual,
                    residual_norm,
                    backward_error,
                });
            }
            let maximum_stability = previous_values
                .as_ref()
                .map(|previous| {
                    let mut maximum = zero(config.precision_bits);
                    for index in 0..config.requested_eigenpairs {
                        let mut change = states[index].value.clone();
                        change -= &previous[index];
                        change.abs_mut();
                        if change > maximum {
                            maximum = change;
                        }
                    }
                    maximum
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
                    states,
                    restart,
                    krylov_steps,
                    operator_applications,
                    maximum_stability,
                    converged,
                );
            }
            previous_values = Some(states.iter().map(|state| state.value.clone()).collect());
            retained_states = states;
        }
        unreachable!("positive maximum_restarts returns from the loop")
    }
}

fn build_report(
    config: &ThickRestartLanczosConfigHp,
    states: Vec<RitzState>,
    restarts: usize,
    krylov_steps: usize,
    operator_applications: usize,
    maximum_stability: Float,
    converged: bool,
) -> Result<ThickRestartLanczosReportHp, SolverError> {
    let operator_dimension = states.first().map_or(0, |state| state.vector.len());
    let cluster_tolerance = parse_positive(
        &config.boundary_cluster_tolerance,
        config.precision_bits,
        "boundary cluster tolerance",
    )?;
    let retained = states.len();
    let mut boundary_cluster = None;
    if config.requested_eigenpairs < retained {
        let requested = config.requested_eigenpairs - 1;
        let guard = config.requested_eigenpairs;
        let mut gap = states[requested].value.clone();
        gap -= &states[guard].value;
        gap.abs_mut();
        if gap <= cluster_tolerance {
            let mut first = requested;
            while first > 0 {
                let mut adjacent = states[first - 1].value.clone();
                adjacent -= &states[first].value;
                adjacent.abs_mut();
                if adjacent > cluster_tolerance {
                    break;
                }
                first -= 1;
            }
            let mut last = guard;
            while last + 1 < retained {
                let mut adjacent = states[last].value.clone();
                adjacent -= &states[last + 1].value;
                adjacent.abs_mut();
                if adjacent > cluster_tolerance {
                    break;
                }
                last += 1;
            }
            let dimension = last - first + 1;
            let mut lower = states[first].value.clone();
            let mut upper = lower.clone();
            let mut maximum_residual = zero(config.precision_bits);
            let mut projected_operator = vec![zero(config.precision_bits); dimension * dimension];
            for row in first..=last {
                if states[row].value < lower {
                    lower = states[row].value.clone();
                }
                if states[row].value > upper {
                    upper = states[row].value.clone();
                }
                if states[row].residual_norm > maximum_residual {
                    maximum_residual = states[row].residual_norm.clone();
                }
                for column in first..=last {
                    projected_operator[(row - first) * dimension + column - first] = dot(
                        &states[row].vector,
                        &states[column].applied,
                        config.precision_bits,
                    );
                }
            }
            boundary_cluster = Some(ThickRestartBoundaryClusterHp {
                first_retained_position: first,
                last_retained_position: last,
                requested_members: config.requested_eigenpairs - first,
                dimension,
                basis: states[first..=last]
                    .iter()
                    .map(|state| state.vector.clone())
                    .collect(),
                projected_operator,
                lower_eigenvalue: lower,
                upper_eigenvalue: upper,
                boundary_gap: gap,
                maximum_residual_norm: maximum_residual,
            });
        }
    }
    let (orthogonality_errors, maximum_orthogonality_error) =
        orthogonality_errors(&states, config.precision_bits);
    let cluster_range = boundary_cluster
        .as_ref()
        .map(|cluster| cluster.first_retained_position..=cluster.last_retained_position);
    let retained_eigenpairs = states
        .iter()
        .enumerate()
        .filter(|(position, _)| {
            !cluster_range
                .as_ref()
                .is_some_and(|range| range.contains(position))
        })
        .map(|(position, state)| ThickRestartEigenpairHp {
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
    let mut provenance = SolverProvenance::current_package("rug_mpfr");
    provenance.precision_bits = Some(config.precision_bits);
    Ok(ThickRestartLanczosReportHp {
        target: config.target.clone(),
        requested_eigenpairs: config.requested_eigenpairs,
        retained_eigenpairs,
        boundary_cluster,
        restarts,
        krylov_steps,
        operator_applications,
        projected_diagonalizations: restarts,
        maximum_subspace_dimension: config.maximum_subspace_dimension,
        maximum_orthogonality_error,
        maximum_ritz_value_stability: maximum_stability,
        estimated_peak_memory_bytes: 3u64
            .saturating_mul(config.maximum_subspace_dimension as u64)
            .saturating_mul(operator_dimension as u64)
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
    use xc_operator::DenseSymmetricHp;

    fn diagonal(precision: u32, values: &[i32]) -> Vec<Float> {
        let n = values.len();
        let mut matrix = vec![zero(precision); n * n];
        for (index, value) in values.iter().enumerate() {
            matrix[index * n + index].assign(*value);
        }
        matrix
    }

    fn config(target: EigenTarget) -> ThickRestartLanczosConfigHp {
        ThickRestartLanczosConfigHp {
            target,
            precision_bits: 192,
            requested_eigenpairs: 2,
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
    fn thick_restart_recovers_multiple_extremes_with_bounded_basis() {
        let precision = 192;
        let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let operator = DenseSymmetricHp::new(
            "thick",
            values.len(),
            diagonal(precision, &values),
            precision,
            &zero(precision),
        )
        .unwrap();
        for (target, expected) in [
            (EigenTarget::AlgebraicLargest, [10, 9]),
            (EigenTarget::AlgebraicSmallest, [1, 2]),
        ] {
            let mut config = config(target);
            config.maximum_subspace_dimension = 8;
            config.maximum_restarts = 100;
            let report = ThickRestartLanczosHp.solve(&operator, &config).unwrap();
            assert_eq!(
                report.status,
                ResultStatus::Converged,
                "residuals={:?}, stability={}",
                report
                    .retained_eigenpairs
                    .iter()
                    .map(|pair| pair.residual_norm.to_string())
                    .collect::<Vec<_>>(),
                report.maximum_ritz_value_stability
            );
            for (pair, expected) in report.retained_eigenpairs.iter().take(2).zip(expected) {
                let mut error = pair.eigenvalue.clone();
                error -= expected;
                error.abs_mut();
                assert!(error < Float::with_val(precision, 1e-18));
                assert!(pair.residual_norm < Float::with_val(precision, 1e-18));
                assert_eq!(pair.diagnostics.absolute_residual, pair.residual_norm);
                assert_eq!(
                    pair.diagnostics.scaled_backward_error,
                    pair.scaled_backward_error
                );
                assert!(pair.diagnostics.orthogonality_error <= report.maximum_orthogonality_error);
            }
            assert!(report.restarts > 1);
            assert_eq!(report.maximum_subspace_dimension, 8);
        }
    }

    #[test]
    fn thick_restart_does_not_split_boundary_multiplicity() {
        let precision = 192;
        let values = [1, 2, 2, 4, 5, 6, 7];
        let operator = DenseSymmetricHp::new(
            "cluster",
            values.len(),
            diagonal(precision, &values),
            precision,
            &zero(precision),
        )
        .unwrap();
        let report = ThickRestartLanczosHp
            .solve(&operator, &config(EigenTarget::AlgebraicSmallest))
            .unwrap();
        assert_eq!(report.status, ResultStatus::UnresolvedCluster);
        let cluster = report.boundary_cluster.unwrap();
        assert_eq!(cluster.dimension, 2);
        assert_eq!(cluster.requested_members, 1);
        assert_eq!(cluster.projected_operator.len(), 4);
    }
}
