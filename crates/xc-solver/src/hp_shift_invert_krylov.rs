use super::{
    check_solver_cancellation, hp_generalized_block::symmetric_jacobi_eigensystem,
    ShiftInvertFactorizationDescriptorHp, ShiftInvertSolveHp, SolverError,
};
use rayon::prelude::*;
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
pub struct ShiftInvertKrylovEigenpairHp {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub scaled_backward_error: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
    pub stopping_evidence: super::HpResidualAcceptance,
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
/// Retained Ritz candidates with residual, stability, and source-bound count evidence.
/// A successful separated request requires a complete count at its boundary;
/// unavailable or mismatched counts retain candidates with an unresolved status.
pub struct ShiftInvertKrylovReportHp {
    /// A source-bound count established the number of eigenvalues on the requested side.
    pub global_target_ordering_established: bool,
    pub boundary_count_evidence: super::BoundaryCountEvidenceHp,
    pub algorithm: String,
    pub target: EigenTarget,
    pub factorization: ShiftInvertFactorizationDescriptorHp,
    pub requested_eigenpairs: usize,
    pub retained_eigenpairs: Vec<ShiftInvertKrylovEigenpairHp>,
    pub boundary_cluster: Option<ShiftInvertKrylovBoundaryClusterHp>,
    pub effective_boundary_cluster_tolerance: Float,
    /// Whether cluster members also appear in retained_eigenpairs.
    pub cluster_members_in_retained_eigenpairs: bool,
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
    applied: Vec<Float>,
    residual_norm: Float,
    backward_error: Float,
    target_distance: Float,
}

fn zero(precision: u32) -> Float {
    Float::with_val(precision, 0)
}

fn parse_finite(value: &DecimalLiteral, precision: u32, name: &str) -> Result<Float, SolverError> {
    let parsed = super::hp_parse_literal(value, precision)?;
    if !parsed.is_finite() {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be finite"
        )));
    }
    Ok(parsed)
}

fn parse_positive(
    value: &xc_core::DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    super::hp_positive_threshold(value, precision, name, rug::float::Round::Down)
}

/// Vectors at least this long form their products and element updates in
/// parallel. Each product or update is the same single correctly rounded
/// operation either way, and sums are always added serially in index order,
/// so the result is bit-identical at every thread count.
const PARALLEL_LENGTH: usize = 32;

fn dot(left: &[Float], right: &[Float], precision: u32) -> Float {
    let product = |(left, right): (&Float, &Float)| {
        let mut product = Float::with_val(precision, left);
        product *= right;
        product
    };
    let products: Vec<Float> = if left.len().min(right.len()) >= PARALLEL_LENGTH {
        left.par_iter().zip(right.par_iter()).map(product).collect()
    } else {
        left.iter().zip(right).map(product).collect()
    };
    let mut sum = zero(precision);
    for product in products {
        sum += product;
    }
    sum
}

/// Apply `update` to every element, in parallel for long vectors.
fn for_each_element(values: &mut [Float], update: impl Fn(usize, &mut Float) + Sync + Send) {
    if values.len() >= PARALLEL_LENGTH {
        values
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| update(index, value));
    } else {
        for (index, value) in values.iter_mut().enumerate() {
            update(index, value);
        }
    }
}

fn norm(vector: &[Float], precision: u32) -> Float {
    super::hp_norm(vector, precision)
}

fn add_orthonormal(candidate: &[Float], basis: &mut Vec<Vec<Float>>, precision: u32) -> bool {
    let mut candidate: Vec<Float> = candidate
        .iter()
        .map(|value| Float::with_val(precision, value))
        .collect();
    // Rank is invariant under nonzero scalar rescaling. Normalize before
    // projection so the rejection threshold measures relative loss of rank.
    let original_norm = norm(&candidate, precision);
    if !original_norm.is_finite() || original_norm.is_zero() {
        return false;
    }
    for_each_element(&mut candidate, |_, value| {
        *value = Float::with_val(precision, &*value);
        *value /= &original_norm;
    });
    for _ in 0..2 {
        for vector in basis.iter() {
            let projection = dot(vector, &candidate, precision);
            for_each_element(&mut candidate, |index, value| {
                if let Some(basis_value) = vector.get(index) {
                    let mut correction = basis_value.clone();
                    correction *= &projection;
                    *value -= correction;
                }
            });
        }
    }
    let length = norm(&candidate, precision);
    let threshold = Float::with_val(precision, 2).pow(-(precision as i32 / 3));
    if !length.is_finite() || length <= threshold {
        return false;
    }
    for_each_element(&mut candidate, |_, value| *value /= &length);
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
        // Solves, products and updates below run on Rayon workers; MPFR
        // exponent bounds are thread-local, so every worker must share the
        // caller's before any parallel arithmetic.
        xc_numerics::mpfr_interval::ensure_uniform_exponent_range()
            .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
        let dimension = operator.dimension();
        let descriptor = shifted_solver.descriptor();
        descriptor.validate(config.precision_bits)?;
        let shift = target_shift(&config.target, &descriptor, config.precision_bits)?;
        let retained = config
            .requested_eigenpairs
            .saturating_add(config.guard_eigenpairs);
        if descriptor.dimension != dimension
            || descriptor.factorization_precision_bits < config.precision_bits
            || !(33..=1_000_000).contains(&config.precision_bits)
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
        if initial_basis.len() > config.maximum_subspace_dimension {
            return Err(SolverError::InvalidConfiguration(
                "initial basis exceeds the configured maximum Krylov subspace dimension".into(),
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

        let cluster_tolerance = super::hp_positive_threshold(
            &config.boundary_cluster_tolerance,
            config.precision_bits,
            "boundary cluster tolerance",
            rug::float::Round::Up,
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
                    .filter(|state| {
                        state.residual_norm > absolute_tolerance
                            && state.backward_error > backward_tolerance
                    })
                    .max_by(|left, right| {
                        left.residual_norm
                            .partial_cmp(&right.residual_norm)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    let mut inverse_image = vec![zero(config.precision_bits); dimension];
                    shifted_solver.solve_shifted(
                        &worst.vector,
                        &mut inverse_image,
                        config.precision_bits,
                    )?;
                    shifted_solves += 1;
                    if inverse_image
                        .iter()
                        .any(|v| !v.is_finite() || v.prec() < config.precision_bits)
                    {
                        return Err(SolverError::NumericalBreakdown("shifted continuation returned nonfinite or insufficient-precision output".into()));
                    }
                    // Form the inverse residual before its normalization.
                    // Normalizing S*x first makes a useful small correction
                    // fail a rank test relative to the dominant Ritz image.
                    for _ in 0..2 {
                        for q in &basis {
                            let coefficient = dot(q, &inverse_image, config.precision_bits);
                            for (value, component) in inverse_image.iter_mut().zip(q) {
                                *value -= Float::with_val(
                                    config.precision_bits,
                                    component * &coefficient,
                                );
                            }
                        }
                    }
                    let _ = add_orthonormal(&inverse_image, &mut basis, config.precision_bits);
                }
            }
            if restart == 1 {
                for seed in 0..retained {
                    if basis.len() >= retained {
                        break;
                    }
                    let candidate: Vec<Float> = (0..dimension)
                        .map(|row| Float::with_val(config.precision_bits, row + seed + 1).recip())
                        .collect();
                    let _ = add_orthonormal(&candidate, &mut basis, config.precision_bits);
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

            let mut expansion_index = 0usize;
            while basis.len() < config.maximum_subspace_dimension {
                check_solver_cancellation(cancellation)?;
                let mut candidate = vec![zero(config.precision_bits); dimension];
                shifted_solver.solve_shifted(
                    &basis[expansion_index],
                    &mut candidate,
                    config.precision_bits,
                )?;
                shifted_solves += 1;
                expansion_index += 1;
                if candidate
                    .iter()
                    .any(|value| !value.is_finite() || value.prec() < config.precision_bits)
                {
                    return Err(SolverError::NumericalBreakdown(
                        "shifted solve returned nonfinite or insufficient-precision output"
                            .to_owned(),
                    ));
                }
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

            // The images are independent solves with one retained factorization.
            // Run them concurrently, then consume them in basis order so the
            // first error and every stored value match the serial loop.
            let images = basis
                .par_iter()
                .map(|vector| {
                    let mut image = vec![zero(config.precision_bits); dimension];
                    shifted_solver
                        .solve_shifted(vector, &mut image, config.precision_bits)
                        .map(|()| image)
                })
                .collect::<Vec<_>>();
            let mut inverse_images = Vec::with_capacity(basis.len());
            for image in images {
                let image = image?;
                shifted_solves += 1;
                if image
                    .iter()
                    .any(|v| !v.is_finite() || v.prec() < config.precision_bits)
                {
                    return Err(SolverError::NumericalBreakdown("projected shifted solve returned nonfinite or insufficient-precision output".into()));
                }
                inverse_images.push(image);
            }
            let subspace_dimension = basis.len();
            let mut projected =
                vec![zero(config.precision_bits); subspace_dimension * subspace_dimension];
            let entries = (0..subspace_dimension)
                .flat_map(|row| (0..=row).map(move |column| (row, column)))
                .collect::<Vec<_>>();
            let values = entries
                .par_iter()
                .map(|&(row, column)| {
                    let mut value =
                        dot(&basis[row], &inverse_images[column], config.precision_bits);
                    value += dot(&basis[column], &inverse_images[row], config.precision_bits);
                    value /= 2u32;
                    value
                })
                .collect::<Vec<_>>();
            for (&(row, column), value) in entries.iter().zip(values) {
                projected[row * subspace_dimension + column] = value.clone();
                projected[column * subspace_dimension + row] = value;
            }
            let (values, vectors) = symmetric_jacobi_eigensystem(
                &projected,
                subspace_dimension,
                config.precision_bits,
                config.maximum_projected_sweeps,
            )?;
            // Inverse Ritz values order interior targets correctly; ordinary
            // A-Ritz values can lie spuriously closer to an interior shift.
            let mut selected: Vec<usize> = (0..subspace_dimension).collect();
            selected.sort_by(|left, right| {
                values[*right]
                    .clone()
                    .abs()
                    .total_cmp(&values[*left].clone().abs())
                    .then_with(|| left.cmp(right))
            });
            selected.truncate(retained);

            let mut states = Vec::with_capacity(retained);
            for index in selected {
                let coefficients = &vectors[index];
                // Each component accumulates its basis columns in column order,
                // exactly as the column-major loop did.
                let mut vector = vec![zero(config.precision_bits); dimension];
                for_each_element(&mut vector, |row, component| {
                    for column in 0..subspace_dimension {
                        let mut contribution = basis[column][row].clone();
                        contribution *= &coefficients[column];
                        *component += contribution;
                    }
                });
                let applied_vector = apply(operator, &vector, config.precision_bits)?;
                operator_applications += 1;
                let norm_squared = dot(&vector, &vector, config.precision_bits);
                if norm_squared <= 0 || !norm_squared.is_finite() {
                    return Err(SolverError::NumericalBreakdown(
                        "inverse Ritz vector has invalid norm".into(),
                    ));
                }
                let value = dot(&vector, &applied_vector, config.precision_bits) / norm_squared;
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
                    value: value.clone(),
                    applied: applied_vector,
                    residual_norm,
                    backward_error,
                    target_distance: target_distance(&value, &shift),
                });
            }
            states.sort_by(|left, right| {
                left.target_distance
                    .total_cmp(&right.target_distance)
                    .then_with(|| left.value.total_cmp(&right.value))
            });
            let maximum_stability = previous_values
                .as_ref()
                .map(|previous| {
                    states.iter().zip(previous).fold(
                        zero(config.precision_bits),
                        |mut maximum, (state, previous)| {
                            let change = super::hp_ritz_change(&state.value, previous, None);
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
                    descriptor,
                    states,
                    restart,
                    shifted_solves,
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
    boundary_count_evidence: super::BoundaryCountEvidenceHp,
) -> Result<ShiftInvertKrylovReportHp, SolverError> {
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
    let cluster_tolerance = super::hp_positive_threshold(
        &config.boundary_cluster_tolerance,
        config.precision_bits,
        "boundary cluster tolerance",
        rug::float::Round::Up,
    )?;
    let cluster_tolerance = super::hp_effective_cluster_tolerance(
        &cluster_tolerance,
        states.iter().map(|state| &state.value),
        states.first().map_or(0, |state| state.vector.len()),
        config.precision_bits,
    );
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
    let dimension = states.first().map_or(0, |state| state.vector.len());
    let mut provenance = SolverProvenance::current_package("rug_mpfr");
    provenance.precision_bits = Some(config.precision_bits);
    Ok(ShiftInvertKrylovReportHp {
        global_target_ordering_established: boundary_count_evidence.establishes_requested_count(),
        boundary_count_evidence,
        algorithm: super::HP_KRYLOV_COUNT_SEMANTICS.into(),
        target: config.target.clone(),
        factorization: descriptor,
        requested_eigenpairs: config.requested_eigenpairs,
        retained_eigenpairs,
        boundary_cluster,
        effective_boundary_cluster_tolerance: cluster_tolerance,
        cluster_members_in_retained_eigenpairs: true,
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
                add_orthonormal(&candidate, &mut basis, 128),
                "scale exponent {exponent}"
            );
            assert_eq!(basis.len(), 2);
            assert!(basis[1][0].clone().abs() < Float::with_val(128, 2).pow(-100));
            assert_eq!(basis[1][1], 1);
        }
    }
}
