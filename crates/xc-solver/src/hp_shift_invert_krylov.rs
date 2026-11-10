use super::{
    check_solver_cancellation, hp_generalized_block::symmetric_jacobi_eigensystem,
    ShiftInvertFactorizationDescriptorHp, ShiftInvertSolveHp, SolverError,
};
use rayon::prelude::*;
use rug::{
    ops::{NegAssign, Pow},
    Assign, Float,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
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
    pub operator_error: super::OperatorErrorEvidenceHp,
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

/// Optional operational telemetry, intentionally absent from numerical reports
/// and managed artifacts. The phases partition this call's wall time; visits
/// count entries into each phase, not arithmetic operations or CPU seconds.
struct KrylovTimings {
    enabled: bool,
    started: Instant,
    phase_started: Instant,
    phase: usize,
    elapsed: [Duration; 8],
    visits: [usize; 8],
    dimension: usize,
    precision: u32,
}

impl KrylovTimings {
    fn new(dimension: usize, precision: u32) -> Self {
        let started = Instant::now();
        Self {
            enabled: std::env::var_os("XC_KRYLOV_TIMING").is_some_and(|v| v == "1"),
            started,
            phase_started: started,
            phase: 0,
            elapsed: [Duration::ZERO; 8],
            visits: [1, 0, 0, 0, 0, 0, 0, 0],
            dimension,
            precision,
        }
    }

    fn enter(&mut self, phase: usize) {
        if self.enabled {
            let now = Instant::now();
            self.elapsed[self.phase] += now.duration_since(self.phase_started);
            self.phase_started = now;
            self.phase = phase;
            self.visits[phase] += 1;
        }
    }
}

impl Drop for KrylovTimings {
    fn drop(&mut self) {
        if self.enabled {
            let now = Instant::now();
            self.elapsed[self.phase] += now.duration_since(self.phase_started);
            const PHASES: [&str; 8] = [
                "setup",
                "basis_and_orthogonalization",
                "shifted_applications",
                "projected_matrix",
                "projected_eigensolve",
                "ritz_construction",
                "operator_residual_validation",
                "source_and_report_validation",
            ];
            let phases = PHASES
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    format!(
                        "{name}={:.6}s/{} visits",
                        self.elapsed[i].as_secs_f64(),
                        self.visits[i]
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            xc_core::progress_message!(
                "[HP Krylov timing] dimension={}, precision={} bits; wall={:.6}s; {phases}",
                self.dimension,
                self.precision,
                now.duration_since(self.started).as_secs_f64()
            );
        }
    }
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

/// Enough work per task to amortize the scheduler, with no worker-count cap.
/// Higher precision admits smaller chunks; a large vector is still shared
/// across the caller's entire Rayon pool. This changes only placement, never
/// arithmetic order. The limb estimate is a scheduling heuristic, not memory
/// admission or a claim about the cost of MPFR multiplication.
fn element_chunk_length(length: usize, precision: u32) -> usize {
    let limbs = (precision as usize).div_ceil(64).max(1);
    let minimum_work = 4096usize.div_ceil(limbs).max(16);
    let pool_grain = length.div_ceil(rayon::current_num_threads().saturating_mul(2).max(1));
    minimum_work.max(pool_grain)
}

#[derive(Default)]
struct DotWorkspace {
    products: Vec<Float>,
}

impl DotWorkspace {
    fn serial_dot(left: &[Float], right: &[Float], precision: u32) -> Float {
        let mut sum = zero(precision);
        let mut product = zero(precision);
        for (left, right) in left.iter().zip(right) {
            product.assign(left);
            product *= right;
            sum += &product;
        }
        sum
    }

    /// Retain MPFR allocations between dot products. The first rounding of
    /// `left` to p bits and the separate multiplication at p bits are both
    /// required: an assignment of `left * right` would fuse those two steps.
    /// Products are always accumulated serially in the original index order.
    fn dot(&mut self, left: &[Float], right: &[Float], precision: u32) -> Float {
        let length = left.len().min(right.len());
        let chunk = element_chunk_length(length, precision);
        if length < chunk.saturating_mul(2) || rayon::current_num_threads() == 1 {
            Self::serial_dot(left, right, precision)
        } else {
            if self.products.len() < length {
                self.products.resize_with(length, || zero(precision));
            }
            self.products[..length]
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(batch, products)| {
                    let offset = batch * chunk;
                    for (index, product) in products.iter_mut().enumerate() {
                        if product.prec() != precision {
                            product.set_prec(precision);
                        }
                        product.assign(&left[offset + index]);
                        *product *= &right[offset + index];
                    }
                });
            let mut sum = zero(precision);
            for product in &self.products[..length] {
                sum += product;
            }
            sum
        }
    }
}

/// One reusable scalar per chunk avoids per-component MPFR allocation.
/// Callers keep each component's operations in their original order.
fn for_each_element(
    values: &mut [Float],
    precision: u32,
    update: impl Fn(usize, &mut Float, &mut Float) + Sync + Send,
) {
    let chunk = element_chunk_length(values.len(), precision);
    let apply_chunk = |batch: usize, values: &mut [Float]| {
        let mut scratch = zero(precision);
        for (index, value) in values.iter_mut().enumerate() {
            update(batch * chunk + index, value, &mut scratch);
        }
    };
    if values.len() >= chunk.saturating_mul(2) && rayon::current_num_threads() > 1 {
        values
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(batch, values)| apply_chunk(batch, values));
    } else {
        apply_chunk(0, values);
    }
}

fn norm(vector: &[Float], precision: u32) -> Float {
    super::hp_norm(vector, precision)
}

fn reset_values(values: &mut [Float], precision: u32) {
    for value in values {
        if value.prec() != precision {
            value.set_prec(precision);
        }
        value.assign(0);
    }
}

fn add_orthonormal(
    candidate: &[Float],
    basis: &mut Vec<Vec<Float>>,
    precision: u32,
    scratch: &mut DotWorkspace,
) -> bool {
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
    for_each_element(&mut candidate, precision, |_, value, _| {
        *value /= &original_norm;
    });
    for _ in 0..2 {
        for vector in basis.iter() {
            let projection = scratch.dot(vector, &candidate, precision);
            for_each_element(&mut candidate, precision, |index, value, correction| {
                if let Some(basis_value) = vector.get(index) {
                    // The old clone inherited the basis component's precision.
                    if correction.prec() != basis_value.prec() {
                        correction.set_prec(basis_value.prec());
                    }
                    correction.assign(basis_value);
                    *correction *= &projection;
                    *value -= &*correction;
                }
            });
        }
    }
    let length = norm(&candidate, precision);
    let threshold = Float::with_val(precision, 2).pow(-(precision as i32 / 3));
    if !length.is_finite() || length <= threshold {
        return false;
    }
    for_each_element(&mut candidate, precision, |_, value, _| *value /= &length);
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
    let mut scratch = DotWorkspace::default();
    let mut errors = vec![zero(precision); states.len()];
    let mut maximum = zero(precision);
    for row in 0..states.len() {
        for column in 0..states.len() {
            let mut value = scratch.dot(&states[row].vector, &states[column].vector, precision);
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

/// Source-bound count taken before any Krylov work, around the target shift
/// `s`, over the open window `(s - r, s + r)` with `r` a quarter of the
/// configured boundary-cluster tolerance.
#[derive(Clone, Debug)]
pub enum KrylovClusterPreflightHp {
    /// More eigenvalues than requested lie in the window, so no Krylov run
    /// with this configuration can be accepted (see
    /// [`ShiftInvertKrylovSolverHp::cluster_preflight`]).
    Precluded { radius: Float, count: usize },
    /// At most the requested number lie in the window; Krylov may succeed.
    Admitted { radius: Float, count: usize },
    /// The source count is unavailable at the window's endpoints.
    Unavailable { radius: Float },
}

impl KrylovClusterPreflightHp {
    pub fn precludes_acceptance(&self) -> bool {
        matches!(self, Self::Precluded { .. })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShiftInvertKrylovSolverHp;

impl ShiftInvertKrylovSolverHp {
    /// Decide, from source-bound counts alone, whether a solve with `config`
    /// is certain to end without an accepted (`Converged`) report.
    ///
    /// A report is `Converged` only with `Verified` boundary-count evidence
    /// for exactly the requested number. That evidence requires the Ritz
    /// target distances `dl <= dr` on either side of the boundary to differ
    /// by more than the effective cluster tolerance, which is at least the
    /// configured tolerance `t`; it then counts eigenvalues of this operator
    /// in `(s - m, s + m)` with `m = (dl + dr) / 2 > t / 2`, up to roundings
    /// of relative size `2^(1-p)`. That window contains `(s - t/4, s + t/4)`.
    /// If this preflight counts more than the requested number there, the
    /// solve's own count exceeds the request too, so its evidence is a count
    /// mismatch or a cluster or unavailable, and the report is never
    /// accepted. Both counts are exact counts of the same operator, so
    /// `Precluded` is a certainty, not an estimate. `Admitted` and
    /// `Unavailable` promise nothing about convergence.
    pub fn cluster_preflight(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        config: &ShiftInvertKrylovConfigHp,
    ) -> Result<KrylovClusterPreflightHp, SolverError> {
        let p = config.precision_bits;
        if !(33..=1_000_000).contains(&p) || config.requested_eigenpairs == 0 {
            return Err(SolverError::InvalidConfiguration(
                "Krylov cluster preflight requires a positive request and adequate precision"
                    .into(),
            ));
        }
        let shift = match &config.target {
            EigenTarget::SmallestMagnitude => zero(p),
            EigenTarget::ClosestTo { shift } => parse_finite(shift, p, "target shift")?,
            _ => {
                return Err(SolverError::UnsupportedTarget(
                    "shift-invert Krylov supports SmallestMagnitude and ClosestTo only".to_owned(),
                ))
            }
        };
        // Rounded down, so the window never exceeds a quarter of the
        // tolerance the solve parses (rounded up).
        let mut radius = parse_positive(
            &config.boundary_cluster_tolerance,
            p,
            "boundary cluster tolerance",
        )?;
        radius >>= 2;
        // The solve rounds s - m and s + m to nearest; with a nonzero shift
        // each rounding moves an endpoint by up to |s| 2^-p, which must stay
        // well inside the margin m - r > r.
        let mut rounding = shift.clone().abs();
        rounding >>= p.saturating_sub(3);
        if radius <= rounding {
            return Ok(KrylovClusterPreflightHp::Unavailable { radius });
        }
        // Directed endpoints keep the counted window inside (s - r, s + r).
        let lower = Float::with_val_round(p, &shift - &radius, rug::float::Round::Up).0;
        let upper = Float::with_val_round(p, &shift + &radius, rug::float::Round::Down).0;
        if lower >= upper {
            return Ok(KrylovClusterPreflightHp::Unavailable { radius });
        }
        let (Some(below_lower), Some(below_upper)) = (
            super::hp_boundary_count::inertia(operator, &lower, p)?,
            super::hp_boundary_count::inertia(operator, &upper, p)?,
        ) else {
            return Ok(KrylovClusterPreflightHp::Unavailable { radius });
        };
        if below_lower.equal != 0 || below_upper.equal != 0 {
            return Ok(KrylovClusterPreflightHp::Unavailable { radius });
        }
        let count = below_upper
            .below
            .checked_sub(below_lower.below)
            .ok_or_else(|| {
                SolverError::NumericalBreakdown("source inertia counts are not monotone".into())
            })?;
        Ok(if count > config.requested_eigenpairs {
            KrylovClusterPreflightHp::Precluded { radius, count }
        } else {
            KrylovClusterPreflightHp::Admitted { radius, count }
        })
    }

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
        let mut timings = KrylovTimings::new(operator.dimension(), config.precision_bits);
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
        let mut scratch = DotWorkspace::default();
        let mut expansion_candidate = vec![zero(config.precision_bits); dimension];

        for restart in 1..=config.maximum_restarts {
            timings.enter(1);
            check_solver_cancellation(cancellation)?;
            let mut basis = Vec::new();
            if restart == 1 {
                for seed in initial_basis {
                    let _ = add_orthonormal(seed, &mut basis, config.precision_bits, &mut scratch);
                }
            } else {
                for state in &retained_states {
                    let _ = add_orthonormal(
                        &state.vector,
                        &mut basis,
                        config.precision_bits,
                        &mut scratch,
                    );
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
                    reset_values(&mut expansion_candidate, config.precision_bits);
                    let inverse_image = &mut expansion_candidate;
                    timings.enter(2);
                    shifted_solver.solve_shifted(
                        &worst.vector,
                        inverse_image,
                        config.precision_bits,
                    )?;
                    timings.enter(1);
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
                            let coefficient = scratch.dot(q, inverse_image, config.precision_bits);
                            for (value, component) in inverse_image.iter_mut().zip(q) {
                                *value -= Float::with_val(
                                    config.precision_bits,
                                    component * &coefficient,
                                );
                            }
                        }
                    }
                    let _ = add_orthonormal(
                        inverse_image,
                        &mut basis,
                        config.precision_bits,
                        &mut scratch,
                    );
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
                    let _ = add_orthonormal(
                        &candidate,
                        &mut basis,
                        config.precision_bits,
                        &mut scratch,
                    );
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
                let _ =
                    add_orthonormal(&reciprocal, &mut basis, config.precision_bits, &mut scratch);
            }
            for coordinate in 0..dimension {
                if !basis.is_empty() {
                    break;
                }
                let mut candidate = vec![zero(config.precision_bits); dimension];
                candidate[coordinate].assign(1);
                let _ =
                    add_orthonormal(&candidate, &mut basis, config.precision_bits, &mut scratch);
            }

            let mut expansion_index = 0usize;
            while basis.len() < config.maximum_subspace_dimension {
                check_solver_cancellation(cancellation)?;
                reset_values(&mut expansion_candidate, config.precision_bits);
                let candidate = &mut expansion_candidate;
                timings.enter(2);
                shifted_solver.solve_shifted(
                    &basis[expansion_index],
                    candidate,
                    config.precision_bits,
                )?;
                timings.enter(1);
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
                if add_orthonormal(candidate, &mut basis, config.precision_bits, &mut scratch) {
                    continue;
                }
                let mut expanded = false;
                for coordinate in 0..dimension {
                    let mut fallback = vec![zero(config.precision_bits); dimension];
                    fallback[coordinate].assign(1);
                    if add_orthonormal(&fallback, &mut basis, config.precision_bits, &mut scratch) {
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
            timings.enter(2);
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
            timings.enter(3);
            let subspace_dimension = basis.len();
            let mut projected =
                vec![zero(config.precision_bits); subspace_dimension * subspace_dimension];
            let entries = (0..subspace_dimension)
                .flat_map(|row| (0..=row).map(move |column| (row, column)))
                .collect::<Vec<_>>();
            let values = entries
                .par_iter()
                .map(|&(row, column)| {
                    // The independent entries already occupy the Rayon pool;
                    // do not recursively schedule every scalar product again.
                    let mut value = DotWorkspace::serial_dot(
                        &basis[row],
                        &inverse_images[column],
                        config.precision_bits,
                    );
                    value += DotWorkspace::serial_dot(
                        &basis[column],
                        &inverse_images[row],
                        config.precision_bits,
                    );
                    value /= 2u32;
                    value
                })
                .collect::<Vec<_>>();
            for (&(row, column), value) in entries.iter().zip(values) {
                projected[row * subspace_dimension + column] = value.clone();
                projected[column * subspace_dimension + row] = value;
            }
            timings.enter(4);
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
            let mut residual = vec![zero(config.precision_bits); dimension];
            for index in selected {
                timings.enter(5);
                let coefficients = &vectors[index];
                // Each component accumulates its basis columns in column order,
                // exactly as the column-major loop did.
                let mut vector = vec![zero(config.precision_bits); dimension];
                for_each_element(
                    &mut vector,
                    config.precision_bits,
                    |row, component, contribution| {
                        for column in 0..subspace_dimension {
                            let basis_value = &basis[column][row];
                            if contribution.prec() != basis_value.prec() {
                                contribution.set_prec(basis_value.prec());
                            }
                            contribution.assign(basis_value);
                            *contribution *= &coefficients[column];
                            *component += &*contribution;
                        }
                    },
                );
                timings.enter(6);
                let applied_vector = apply(operator, &vector, config.precision_bits)?;
                operator_applications += 1;
                let norm_squared = scratch.dot(&vector, &vector, config.precision_bits);
                if norm_squared <= 0 || !norm_squared.is_finite() {
                    return Err(SolverError::NumericalBreakdown(
                        "inverse Ritz vector has invalid norm".into(),
                    ));
                }
                let value =
                    scratch.dot(&vector, &applied_vector, config.precision_bits) / norm_squared;
                for ((result, applied), component) in
                    residual.iter_mut().zip(&applied_vector).zip(&vector)
                {
                    if result.prec() != component.prec() {
                        result.set_prec(component.prec());
                    }
                    result.assign(component);
                    *result *= &value;
                    result.neg_assign();
                    *result += applied;
                }
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
            timings.enter(7);
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
                    operator,
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
    operator: &dyn SymmetricOperator<Float>,
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
    let retained_eigenpairs: Vec<ShiftInvertKrylovEigenpairHp> = states
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
        operator_error: super::OperatorErrorEvidenceHp::for_residuals(
            operator,
            None,
            retained_eigenpairs
                .iter()
                .map(|p: &ShiftInvertKrylovEigenpairHp| (&p.eigenvalue, &p.residual_norm)),
            config.precision_bits,
        )?,
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

    fn diagonal_from(precision: u32, values: &[&str]) -> Vec<Float> {
        let n = values.len();
        let mut matrix = vec![zero(precision); n * n];
        for (index, value) in values.iter().enumerate() {
            matrix[index * n + index] = Float::with_val(precision, Float::parse(value).unwrap());
        }
        matrix
    }

    fn preflight_and_solve(values: &[&str]) -> (KrylovClusterPreflightHp, ResultStatus) {
        let precision = 192;
        let matrix = diagonal_from(precision, values);
        let operator = DenseSymmetricHp::new(
            "preflight",
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
        let preflight = ShiftInvertKrylovSolverHp
            .cluster_preflight(&operator, &config())
            .unwrap();
        let report = ShiftInvertKrylovSolverHp
            .solve(&operator, &factorization, &config())
            .unwrap();
        (preflight, report.status)
    }

    #[test]
    fn preflight_precludes_a_target_cluster_that_krylov_cannot_accept() {
        // Two eigenvalues lie within a quarter of the 1e-20 cluster tolerance.
        let (preflight, status) =
            preflight_and_solve(&["1e-30", "-2e-30", "1", "4", "9", "12", "15", "20"]);
        assert!(
            matches!(
                preflight,
                KrylovClusterPreflightHp::Precluded { count: 2, .. }
            ),
            "{preflight:?}"
        );
        assert_ne!(status, ResultStatus::Converged);
    }

    #[test]
    fn preflight_admits_a_separated_target_that_krylov_accepts() {
        for values in [
            ["-8", "-3", "1", "4", "9", "12", "15", "20"],
            ["1e-30", "-3", "1", "4", "9", "12", "15", "20"],
        ] {
            let (preflight, status) = preflight_and_solve(&values);
            assert!(!preflight.precludes_acceptance(), "{preflight:?}");
            assert_eq!(status, ResultStatus::Converged);
        }
    }

    #[test]
    fn preflight_window_is_a_quarter_of_the_rounded_down_tolerance() {
        let precision = 192;
        // 3e-21 is outside a quarter of 1e-20 (2.5e-21); 2e-21 is inside.
        for (second, precluded) in [("3e-21", false), ("2e-21", true)] {
            let (preflight, _) =
                preflight_and_solve(&["1e-30", second, "1", "4", "9", "12", "15", "20"]);
            assert_eq!(preflight.precludes_acceptance(), precluded, "{preflight:?}");
            let radius = match preflight {
                KrylovClusterPreflightHp::Precluded { radius, .. }
                | KrylovClusterPreflightHp::Admitted { radius, .. }
                | KrylovClusterPreflightHp::Unavailable { radius } => radius,
            };
            let mut expected = Float::with_val_round(
                precision,
                Float::parse("1e-20").unwrap(),
                rug::float::Round::Down,
            )
            .0;
            expected >>= 2;
            assert_eq!(radius, expected);
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
                add_orthonormal(&candidate, &mut basis, 128, &mut DotWorkspace::default()),
                "scale exponent {exponent}"
            );
            assert_eq!(basis.len(), 2);
            assert!(basis[1][0].clone().abs() < Float::with_val(128, 2).pow(-100));
            assert_eq!(basis[1][1], 1);
        }
    }
}

#[cfg(test)]
mod scratch_order_contract {
    use super::*;

    // The pre-optimization arithmetic and scheduling, kept here as an exact
    // regression oracle. In particular, do not replace either product with
    // a fused multiplication/accumulation or change normalization ordering.
    fn original_dot(left: &[Float], right: &[Float], precision: u32) -> Float {
        let product = |(left, right): (&Float, &Float)| {
            let mut value = Float::with_val(precision, left);
            value *= right;
            value
        };
        let products: Vec<_> = if left.len().min(right.len()) >= 32 {
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

    fn original_update(values: &mut [Float], update: impl Fn(usize, &mut Float) + Sync + Send) {
        if values.len() >= 32 {
            values
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, v)| update(i, v));
        } else {
            for (i, v) in values.iter_mut().enumerate() {
                update(i, v);
            }
        }
    }

    fn original_orthonormal(
        candidate: &[Float],
        basis: &mut Vec<Vec<Float>>,
        precision: u32,
    ) -> bool {
        let mut candidate: Vec<_> = candidate
            .iter()
            .map(|v| Float::with_val(precision, v))
            .collect();
        let original_norm = norm(&candidate, precision);
        if !original_norm.is_finite() || original_norm.is_zero() {
            return false;
        }
        original_update(&mut candidate, |_, value| {
            *value = Float::with_val(precision, &*value);
            *value /= &original_norm;
        });
        for _ in 0..2 {
            for vector in basis.iter() {
                let projection = original_dot(vector, &candidate, precision);
                original_update(&mut candidate, |index, value| {
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
        original_update(&mut candidate, |_, value| *value /= &length);
        basis.push(candidate);
        true
    }

    fn sample(length: usize, precision: u32, seed: usize) -> Vec<Float> {
        (0..length)
            .map(|i| {
                let component_precision = precision + [0, 13, 67][i % 3];
                let mut value = Float::with_val(
                    component_precision,
                    ((i * 37 + seed * 17) % 101) as i32 - 50,
                );
                value /= i + seed + 3;
                value <<= (i % 19) as i32 - 9;
                value
            })
            .collect()
    }

    #[test]
    fn workspace_preserves_products_rounding_order_and_precision_changes() {
        for threads in [1, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut scratch = DotWorkspace::default();
                for p in [64, 192, 729, 3386, 6708, 23284, 192] {
                    for n in [0, 1, 31, 32, 127, 501, 2401, 31] {
                        let left = sample(n, p, 1);
                        let right = sample(n + 3, p, 2);
                        let expected = original_dot(&left, &right, p);
                        let actual = scratch.dot(&left, &right, p);
                        assert_eq!(actual, expected, "n={n}, p={p}, threads={threads}");
                        assert_eq!(actual.prec(), expected.prec());
                    }
                }
            });
        }
    }

    #[test]
    fn cancellation_and_signed_zero_keep_exact_product_and_sum_order() {
        for threads in [1, 8] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut scratch = DotWorkspace::default();
                    for p in [63, 64, 65, 127, 128, 129, 3386] {
                        let mut big = Float::with_val(p, 1);
                        big <<= (p + 5) as i32;
                        let left: Vec<_> = (0..2401)
                            .map(|i| match i % 5 {
                                0 => big.clone(),
                                1 => Float::with_val(p, 1),
                                2 => -big.clone(),
                                3 => Float::with_val(p, rug::float::Special::NegZero),
                                _ => zero(p),
                            })
                            .collect();
                        let right = vec![Float::with_val(p + 17, 1); left.len()];
                        let expected = original_dot(&left, &right, p);
                        let actual = scratch.dot(&left, &right, p);
                        assert_eq!(actual, expected);
                        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                        assert_eq!(actual.prec(), expected.prec());
                    }
                });
        }
    }

    #[test]
    fn orthogonalization_matches_original_with_mixed_precision_basis() {
        for threads in [1, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for p in [64, 192, 3386, 6708] {
                    let n = 501;
                    let mut expected_basis = Vec::new();
                    let mut actual_basis = Vec::new();
                    let mut scratch = DotWorkspace::default();
                    for seed in 0..4 {
                        let input = sample(n, p, seed);
                        let expected = original_orthonormal(&input, &mut expected_basis, p);
                        let actual = add_orthonormal(&input, &mut actual_basis, p, &mut scratch);
                        assert_eq!(actual, expected, "p={p}, seed={seed}, threads={threads}");
                        assert_eq!(actual_basis, expected_basis);
                        // Exercise the old clone's inherited component precision.
                        for basis in [&mut expected_basis, &mut actual_basis] {
                            for (index, value) in basis.last_mut().unwrap().iter_mut().enumerate() {
                                value.set_prec(p + [0, 13, 67][index % 3]);
                            }
                        }
                    }
                    for input in [vec![zero(p); n], actual_basis[0].clone()] {
                        assert_eq!(
                            add_orthonormal(&input, &mut actual_basis, p, &mut scratch),
                            original_orthonormal(&input, &mut expected_basis, p)
                        );
                        assert_eq!(actual_basis, expected_basis);
                    }
                }
            });
        }
    }

    #[test]
    #[ignore = "matched wall-clock benchmark; run explicitly in release mode on an otherwise quiet host"]
    fn benchmark_scratch_and_chunk_scheduling() {
        for threads in [1, 8, rayon::current_num_threads()] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for (n, p) in [(501, 5047), (2401, 4217), (501, 23284)] {
                    let candidate = sample(n, p, 20);
                    let basis: Vec<_> = (0..8).map(|seed| {
                        let mut vector = sample(n, p, seed);
                        let length = norm(&vector, p);
                        for value in &mut vector { *value /= &length; }
                        vector
                    }).collect();
                    let mut expected = basis.clone();
                    let accepted = original_orthonormal(&candidate, &mut expected, p);
                    let mut workspace = DotWorkspace::default();
                    let mut actual = basis.clone();
                    assert_eq!(add_orthonormal(&candidate, &mut actual, p, &mut workspace), accepted);
                    assert_eq!(actual, expected);
                    for round in 0..3 {
                        // Alternate order to limit systematic thermal/cache bias.
                        for optimized in if round % 2 == 0 { [false, true] } else { [true, false] } {
                            let start = Instant::now();
                            for _ in 0..4 {
                                let mut basis = basis.clone();
                                let result = if optimized {
                                    add_orthonormal(&candidate, &mut basis, p, &mut workspace)
                                } else {
                                    original_orthonormal(&candidate, &mut basis, p)
                                };
                                std::hint::black_box((result, basis));
                            }
                            println!("krylov_scratch n={n} p={p} threads={threads} round={round} optimized={optimized} seconds={:.9}", start.elapsed().as_secs_f64());
                        }
                    }
                }
            });
        }
    }
}
