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
    /// Absolute residual in operator units; acceptance is this OR the scaled
    /// backward-error tolerance, followed by the separate stability guards.
    /// Scaling the operator without scaling this tolerance changes acceptance.
    pub absolute_residual_tolerance: DecimalLiteral,
    /// Dimensionless residual normalized by the computed action norms.
    pub scaled_backward_error_tolerance: DecimalLiteral,
    /// Maximum absolute |lambda_new-lambda_old| in eigenvalue units.
    /// This is iterate agreement, not an eigenvalue error bound.
    pub ritz_value_stability_tolerance: DecimalLiteral,
    /// Absolute projected gap tolerance in eigenvalue/target-distance units.
    pub boundary_cluster_tolerance: DecimalLiteral,
}

#[derive(Clone, Debug)]
pub struct ThickRestartEigenpairHp {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub scaled_backward_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
    pub stopping_evidence: super::HpResidualAcceptance,
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
/// Retained Ritz candidates with residual, stability, and source-bound count evidence.
/// A successful separated request requires a complete count at its boundary;
/// unavailable or mismatched counts retain candidates with an unresolved status.
pub struct ThickRestartLanczosReportHp {
    /// A source-bound count established the number of eigenvalues on the requested side.
    pub global_target_ordering_established: bool,
    pub boundary_count_evidence: super::BoundaryCountEvidenceHp,
    pub algorithm: String,
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    pub retained_eigenpairs: Vec<ThickRestartEigenpairHp>,
    pub boundary_cluster: Option<ThickRestartBoundaryClusterHp>,
    pub effective_boundary_cluster_tolerance: Float,
    /// Whether cluster members also appear in retained_eigenpairs.
    pub cluster_members_in_retained_eigenpairs: bool,
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
    value: &xc_core::DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    super::hp_positive_threshold(value, precision, name, rug::float::Round::Down)
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
    super::hp_norm(vector, precision)
}

fn add_orthonormal(candidate: Vec<Float>, basis: &mut Vec<Vec<Float>>, precision: u32) -> bool {
    let mut candidate = candidate;
    // Rank is invariant under nonzero scalar rescaling. Normalize before
    // projection so the rejection threshold measures relative loss of rank.
    let original_norm = norm(&candidate, precision);
    if !original_norm.is_finite() || original_norm.is_zero() {
        return false;
    }
    for value in &mut candidate {
        *value = Float::with_val(precision, &*value);
        *value /= &original_norm;
    }
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
    super::hp_checked_action(operator, vector, precision)
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
        self.solve_controlled_with_direct_image_reuse(operator, config, cancellation, true)
    }

    fn solve_controlled_with_direct_image_reuse(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        config: &ThickRestartLanczosConfigHp,
        cancellation: &CancellationToken,
        reuse_direct_images: bool,
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
        if !(33..=1_000_000).contains(&config.precision_bits)
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
                "HP thick-restart Lanczos requires valid counts, mandatory guards, a retained block smaller than the bounded Krylov subspace, precision in 33..=1000000 bits, and valid restart limits"
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
        let cluster_tolerance = super::hp_positive_threshold(
            &config.boundary_cluster_tolerance,
            config.precision_bits,
            "boundary cluster tolerance",
            rug::float::Round::Up,
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
            if retained_states.is_empty() {
                for seed in 0..retained {
                    let candidate = (0..dimension)
                        .map(|row| Float::with_val(config.precision_bits, row + seed + 1).recip())
                        .collect();
                    let _ = add_orthonormal(candidate, &mut basis, config.precision_bits);
                }
            }
            let continuation = if retained_states.is_empty() {
                Some(
                    (0..dimension)
                        .map(|row| Float::with_val(config.precision_bits, row + 1).recip())
                        .collect(),
                )
            } else {
                retained_states
                    .iter()
                    .filter(|state| {
                        state.residual_norm > absolute_tolerance
                            && state.backward_error > backward_tolerance
                    })
                    .max_by(|left, right| {
                        left.residual_norm
                            .partial_cmp(&right.residual_norm)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|state| state.residual.clone())
            };
            // Once requested residuals have converged, explore a fresh coordinate.
            // Normalizing roundoff-sized residuals can retain the same invariant
            // space and miss another direction of a repeated eigenvalue.
            if let Some(continuation) = continuation {
                let _ = add_orthonormal(continuation, &mut basis, config.precision_bits);
            }
            for coordinate in 0..dimension {
                if basis.len() > retained_states.len() {
                    break;
                }
                let mut candidate = vec![zero(config.precision_bits); dimension];
                candidate[coordinate].assign(1);
                let _ = add_orthonormal(candidate, &mut basis, config.precision_bits);
            }
            // Retain only direct A*q results computed during this expansion.
            // Reconstructed Ritz images from an earlier restart deliberately
            // are not reused: their different MPFR operation route can differ
            // in low bits from a fresh operator application.
            let mut retained_applied: Vec<Option<Vec<Float>>> =
                (0..basis.len()).map(|_| None).collect();
            let mut expansion_index = 0usize;
            while basis.len() < config.maximum_subspace_dimension {
                check_solver_cancellation(cancellation)?;
                let candidate = apply(operator, &basis[expansion_index], config.precision_bits)?;
                operator_applications += 1;
                let current = expansion_index;
                expansion_index += 1;
                if reuse_direct_images {
                    retained_applied[current] = Some(candidate.clone());
                }
                if add_orthonormal(candidate, &mut basis, config.precision_bits) {
                    retained_applied.push(None);
                } else {
                    let mut expanded = false;
                    for coordinate in 0..dimension {
                        let mut fallback = vec![zero(config.precision_bits); dimension];
                        fallback[coordinate].assign(1);
                        if add_orthonormal(fallback, &mut basis, config.precision_bits) {
                            retained_applied.push(None);
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
            for (vector, retained_image) in basis.iter().zip(retained_applied) {
                if let Some(image) = retained_image {
                    applied.push(image);
                } else {
                    applied.push(apply(operator, vector, config.precision_bits)?);
                    operator_applications += 1;
                }
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
                let (residual_norm, backward_error) = super::hp_residual_measures(
                    &residual,
                    &applied_vector,
                    &vector,
                    &value,
                    config.precision_bits,
                )?;
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
                    for index in 0..states.len() {
                        let change =
                            super::hp_ritz_change(&states[index].value, &previous[index], None);
                        if change > maximum {
                            maximum = change;
                        }
                    }
                    maximum
                })
                .unwrap_or_else(|| {
                    Float::with_val(config.precision_bits, rug::float::Special::Infinity)
                });
            // Refine and qualify the entire retained requested+guard block.
            let residuals_converged = states.iter().all(|state| {
                state.residual_norm <= absolute_tolerance
                    || state.backward_error <= backward_tolerance
            });
            let candidate_converged = restart >= config.minimum_restarts
                && residuals_converged
                && maximum_stability <= stability_tolerance;
            let evidence = if candidate_converged || restart == config.maximum_restarts {
                super::hp_boundary_count::boundary_count(
                    operator,
                    &config.target,
                    config.requested_eigenpairs,
                    &states.iter().map(|s| s.value.clone()).collect::<Vec<_>>(),
                    &cluster_tolerance,
                    config.precision_bits,
                )?
            } else {
                super::BoundaryCountEvidenceHp::Unavailable {
                    reason: "Ritz iteration has not met residual and stability requirements".into(),
                }
            };
            let converged = candidate_converged;
            if converged || restart == config.maximum_restarts {
                return build_report(
                    config,
                    states,
                    restart,
                    krylov_steps,
                    operator_applications,
                    maximum_stability,
                    converged,
                    evidence,
                );
            }
            previous_values = Some(states.iter().map(|state| state.value.clone()).collect());
            retained_states = states;
        }
        unreachable!("positive maximum_restarts returns from the loop")
    }
}

// Keep actual work counters and distinct acceptance evidence explicit here.
#[allow(clippy::too_many_arguments)]
fn build_report(
    config: &ThickRestartLanczosConfigHp,
    states: Vec<RitzState>,
    restarts: usize,
    krylov_steps: usize,
    operator_applications: usize,
    maximum_stability: Float,
    converged: bool,
    boundary_count_evidence: super::BoundaryCountEvidenceHp,
) -> Result<ThickRestartLanczosReportHp, SolverError> {
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
    let operator_dimension = states.first().map_or(0, |state| state.vector.len());
    let cluster_tolerance = super::hp_positive_threshold(
        &config.boundary_cluster_tolerance,
        config.precision_bits,
        "boundary cluster tolerance",
        rug::float::Round::Up,
    )?;
    let retained = states.len();
    let cluster_tolerance = super::hp_effective_cluster_tolerance(
        &cluster_tolerance,
        states.iter().map(|state| &state.value),
        operator_dimension,
        config.precision_bits,
    );
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
            stopping_evidence: super::hp_residual_acceptance(
                &state.applied,
                &state.vector,
                &state.value,
                &state.residual_norm,
                &state.backward_error,
                &absolute_tolerance,
                &backward_tolerance,
                config.precision_bits,
            ),
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
    } else if converged && !boundary_count_evidence.establishes_requested_count() {
        (
            ResultStatus::UnresolvedEigenspace,
            TerminationReason::UnresolvedEigenspace,
        )
    } else if converged {
        (
            ResultStatus::Converged,
            super::hp_block_termination(
                states
                    .iter()
                    .take(config.requested_eigenpairs)
                    .map(|state| (&state.residual_norm, &state.backward_error)),
                &absolute_tolerance,
                &backward_tolerance,
            ),
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
        global_target_ordering_established: boundary_count_evidence.establishes_requested_count(),
        boundary_count_evidence,
        algorithm: super::HP_KRYLOV_COUNT_SEMANTICS.into(),
        target: config.target.clone(),
        requested_eigenpairs: config.requested_eigenpairs,
        retained_eigenpairs,
        boundary_cluster,
        effective_boundary_cluster_tolerance: cluster_tolerance,
        cluster_members_in_retained_eigenpairs: false,
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
        assert_eq!(
            report.status,
            ResultStatus::UnresolvedCluster,
            "pairs={:?}; stability={}",
            report
                .retained_eigenpairs
                .iter()
                .map(|pair| (&pair.eigenvalue, &pair.residual_norm))
                .collect::<Vec<_>>(),
            report.maximum_ritz_value_stability
        );
        let cluster = report.boundary_cluster.unwrap();
        assert_eq!(cluster.dimension, 2);
        assert_eq!(cluster.requested_members, 1);
        assert_eq!(cluster.projected_operator.len(), 4);
    }

    #[test]
    fn retained_direct_operator_images_preserve_every_numerical_output() {
        let precision = 192;
        let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let operator = DenseSymmetricHp::new(
            "direct-image-equivalence",
            values.len(),
            diagonal(precision, &values),
            precision,
            &zero(precision),
        )
        .unwrap();
        let mut config = config(EigenTarget::AlgebraicSmallest);
        config.maximum_subspace_dimension = 8;
        config.maximum_restarts = 12;
        let cancellation = CancellationToken::new();
        let reference = ThickRestartLanczosHp
            .solve_controlled_with_direct_image_reuse(&operator, &config, &cancellation, false)
            .unwrap();
        let optimized = ThickRestartLanczosHp
            .solve_controlled_with_direct_image_reuse(&operator, &config, &cancellation, true)
            .unwrap();

        assert!(optimized.operator_applications < reference.operator_applications);
        assert_eq!(optimized.status, reference.status);
        assert_eq!(optimized.termination, reference.termination);
        assert_eq!(optimized.restarts, reference.restarts);
        assert_eq!(optimized.krylov_steps, reference.krylov_steps);
        assert_eq!(
            optimized.maximum_orthogonality_error,
            reference.maximum_orthogonality_error
        );
        assert_eq!(
            optimized.maximum_ritz_value_stability,
            reference.maximum_ritz_value_stability
        );
        assert_eq!(
            optimized.retained_eigenpairs.len(),
            reference.retained_eigenpairs.len()
        );
        for (actual, expected) in optimized
            .retained_eigenpairs
            .iter()
            .zip(&reference.retained_eigenpairs)
        {
            assert_eq!(actual.eigenvalue, expected.eigenvalue);
            assert_eq!(actual.eigenvector, expected.eigenvector);
            assert_eq!(actual.residual_norm, expected.residual_norm);
            assert_eq!(actual.scaled_backward_error, expected.scaled_backward_error);
            assert_eq!(actual.diagnostics, expected.diagnostics);
        }
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

#[cfg(test)]
mod exhaustive_rank_contract {
    use super::*;
    #[test]
    fn rank_is_invariant_under_nonzero_power_of_two_scaling() {
        for exponent in [-200i32, 0, 200] {
            let scale = Float::with_val(128, 2).pow(exponent);
            let candidate = vec![scale.clone(), scale];
            let mut basis = vec![vec![Float::with_val(128, 1), Float::with_val(128, 0)]];
            assert!(
                add_orthonormal(candidate, &mut basis, 128),
                "scale exponent {exponent}"
            );
            assert_eq!(basis.len(), 2);
            assert!(basis[1][0].clone().abs() < Float::with_val(128, 2).pow(-100));
            assert_eq!(basis[1][1], 1);
        }
    }
}
