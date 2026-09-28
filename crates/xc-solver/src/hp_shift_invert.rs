use super::{
    check_solver_cancellation, hp_generalized_block::symmetric_jacobi_eigensystem, SolverError,
    SolverPerformanceTelemetry,
};
use rug::{ops::Pow, Assign, Float};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::time::Instant;
use xc_core::{
    AssuranceLevel, CancellationToken, DecimalLiteral, EigenTarget, PrecisionPolicy, ResultStatus,
    SolverProvenance, TerminationReason,
};
use xc_operator::SymmetricOperator;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftInvertFactorizationDescriptorHp {
    pub id: String,
    pub dimension: usize,
    pub shift: DecimalLiteral,
    pub factorization_precision_bits: u32,
    pub exact_shifted_solve: bool,
    pub approximation_error_bound: Option<DecimalLiteral>,
}

impl ShiftInvertFactorizationDescriptorHp {
    pub(super) fn validate(&self, working_precision_bits: u32) -> Result<(), SolverError> {
        if !(33..=1_000_000).contains(&working_precision_bits)
            || !(33..=1_000_000).contains(&self.factorization_precision_bits)
            || self.id.trim().is_empty()
            || self.dimension == 0
            || self.factorization_precision_bits < working_precision_bits
        {
            return Err(SolverError::InvalidConfiguration(
                "shift-invert descriptor requires a nonempty id, positive dimension, and factorization precision at least the working precision"
                    .to_owned(),
            ));
        }
        let shift = parse_finite(&self.shift, working_precision_bits, "shift")?;
        if !shift.is_finite() {
            return Err(SolverError::InvalidConfiguration(
                "shift-invert shift must be finite".to_owned(),
            ));
        }
        let bound = self
            .approximation_error_bound
            .as_ref()
            .map(|value| parse_nonnegative(value, working_precision_bits, "approximation bound"))
            .transpose()?;
        if self.exact_shifted_solve && bound.as_ref().is_some_and(|value| !value.is_zero()) {
            return Err(SolverError::InvalidConfiguration(
                "an exact shifted solve cannot declare nonzero approximation error".to_owned(),
            ));
        }
        if !self.exact_shifted_solve && bound.is_none() {
            return Err(SolverError::InvalidConfiguration(
                "an approximate shifted solve must declare a finite nonnegative error bound"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Application of a previously prepared inverse of `A - shift I`.
/// Implementations may be dense, sparse, or matrix-free, but must disclose
/// whether the action is exact at its factorization precision or bounded.
/// A successful action must leave every output finite and represented with at
/// least `working_precision_bits`; invalid output is a solver error, not rank loss.
pub trait ShiftInvertSolveHp: Send + Sync {
    fn descriptor(&self) -> ShiftInvertFactorizationDescriptorHp;
    fn solve_shifted(
        &self,
        right_hand_side: &[Float],
        output: &mut [Float],
        working_precision_bits: u32,
    ) -> Result<(), SolverError>;
}

/// Creates a fresh shifted factorization at each adaptive precision attempt.
pub trait ShiftInvertFactoryHp: Send + Sync {
    fn factor_at_precision(
        &self,
        precision_bits: u32,
    ) -> Result<Box<dyn ShiftInvertSolveHp>, SolverError>;
}

#[derive(Clone, Debug)]
pub struct DenseShiftInvertFactoryHp {
    id: String,
    dimension: usize,
    matrix: Vec<Float>,
    shift: DecimalLiteral,
    source_precision_bits: u32,
}

impl DenseShiftInvertFactoryHp {
    pub fn new(
        id: impl Into<String>,
        dimension: usize,
        matrix: &[Float],
        shift: DecimalLiteral,
        source_precision_bits: u32,
    ) -> Result<Self, SolverError> {
        let id = id.into();
        if id.trim().is_empty()
            || dimension == 0
            || matrix.len() != dimension.saturating_mul(dimension)
            || !(33..=1_000_000).contains(&source_precision_bits)
        {
            return Err(SolverError::InvalidConfiguration(
                "dense shift-invert factory requires a nonempty id, square positive matrix, and source precision in 33..=1000000 bits"
                    .to_owned(),
            ));
        }
        let _ = parse_finite(&shift, source_precision_bits, "factory shift")?;
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(SolverError::InvalidConfiguration(
                "dense shift-invert factory matrix must be finite".to_owned(),
            ));
        }
        Ok(Self {
            id,
            dimension,
            matrix: matrix
                .iter()
                .map(|value| Float::with_val(source_precision_bits, value))
                .collect(),
            shift,
            source_precision_bits,
        })
    }
}

impl ShiftInvertFactoryHp for DenseShiftInvertFactoryHp {
    fn factor_at_precision(
        &self,
        precision_bits: u32,
    ) -> Result<Box<dyn ShiftInvertSolveHp>, SolverError> {
        if !(33..=1_000_000).contains(&precision_bits)
            || precision_bits > self.source_precision_bits
        {
            return Err(SolverError::InvalidConfiguration(format!(
                "requested factorization precision {precision_bits} exceeds retained source precision {}",
                self.source_precision_bits
            )));
        }
        let matrix: Vec<Float> = self
            .matrix
            .iter()
            .map(|value| Float::with_val(precision_bits, value))
            .collect();
        Ok(Box::new(DenseShiftInvertFactorizationHp::factor_owned(
            format!("{}@{precision_bits}", self.id),
            self.dimension,
            matrix,
            self.shift.clone(),
            precision_bits,
        )?))
    }
}

/// Dense MPFR LU factorization with partial pivoting for `A - shift I`.
/// It factors only the shifted system and never computes an eigenspectrum.
#[derive(Clone, Debug)]
pub struct DenseShiftInvertFactorizationHp {
    dimension: usize,
    lu: Vec<Float>,
    pivots: Vec<usize>,
    descriptor: ShiftInvertFactorizationDescriptorHp,
}

impl DenseShiftInvertFactorizationHp {
    pub fn factor(
        id: impl Into<String>,
        dimension: usize,
        matrix: &[Float],
        shift: DecimalLiteral,
        precision_bits: u32,
    ) -> Result<Self, SolverError> {
        if dimension == 0
            || matrix.len() != dimension.saturating_mul(dimension)
            || !(33..=1_000_000).contains(&precision_bits)
        {
            return Err(SolverError::InvalidConfiguration(
                "dense shift-invert factorization requires a square positive matrix and precision in 33..=1000000 bits"
                    .to_owned(),
            ));
        }
        let matrix = matrix
            .iter()
            .map(|value| Float::with_val(precision_bits, value))
            .collect();
        Self::factor_owned(id, dimension, matrix, shift, precision_bits)
    }

    fn factor_owned(
        id: impl Into<String>,
        dimension: usize,
        mut lu: Vec<Float>,
        shift: DecimalLiteral,
        precision_bits: u32,
    ) -> Result<Self, SolverError> {
        if dimension == 0
            || lu.len() != dimension.saturating_mul(dimension)
            || !(33..=1_000_000).contains(&precision_bits)
        {
            return Err(SolverError::InvalidConfiguration(
                "dense shift-invert factorization requires a square positive matrix and precision in 33..=1000000 bits"
                    .to_owned(),
            ));
        }
        let shift_value = parse_finite(&shift, precision_bits, "shift")?;
        for value in &lu {
            if !value.is_finite() {
                return Err(SolverError::InvalidConfiguration(
                    "dense shift-invert matrix must contain only finite values".to_owned(),
                ));
            }
        }
        for index in 0..dimension {
            lu[index * dimension + index] -= &shift_value;
        }
        let mut pivots = Vec::with_capacity(dimension);
        for column in 0..dimension {
            let mut pivot = column;
            let mut pivot_abs = lu[column * dimension + column].clone().abs();
            for row in column + 1..dimension {
                let candidate = lu[row * dimension + column].clone().abs();
                if candidate > pivot_abs {
                    pivot = row;
                    pivot_abs = candidate;
                }
            }
            if pivot_abs.is_zero() || !pivot_abs.is_finite() {
                return Err(SolverError::NumericalBreakdown(format!(
                    "shifted matrix is singular at elimination column {column}"
                )));
            }
            pivots.push(pivot);
            if pivot != column {
                for entry in 0..dimension {
                    lu.swap(column * dimension + entry, pivot * dimension + entry);
                }
            }
            let diagonal = lu[column * dimension + column].clone();
            for row in column + 1..dimension {
                let mut multiplier = lu[row * dimension + column].clone();
                multiplier /= &diagonal;
                lu[row * dimension + column] = multiplier.clone();
                for entry in column + 1..dimension {
                    let mut correction = multiplier.clone();
                    correction *= &lu[column * dimension + entry];
                    lu[row * dimension + entry] -= correction;
                }
            }
        }
        let descriptor = ShiftInvertFactorizationDescriptorHp {
            id: id.into(),
            dimension,
            shift,
            factorization_precision_bits: precision_bits,
            exact_shifted_solve: true,
            approximation_error_bound: None,
        };
        descriptor.validate(precision_bits)?;
        Ok(Self {
            dimension,
            lu,
            pivots,
            descriptor,
        })
    }
}

impl ShiftInvertSolveHp for DenseShiftInvertFactorizationHp {
    fn descriptor(&self) -> ShiftInvertFactorizationDescriptorHp {
        self.descriptor.clone()
    }

    fn solve_shifted(
        &self,
        right_hand_side: &[Float],
        output: &mut [Float],
        working_precision_bits: u32,
    ) -> Result<(), SolverError> {
        self.descriptor.validate(working_precision_bits)?;
        if right_hand_side.len() != self.dimension || output.len() != self.dimension {
            return Err(SolverError::InvalidConfiguration(
                "shifted solve dimension mismatch".to_owned(),
            ));
        }
        let precision = self.descriptor.factorization_precision_bits;
        let mut solution: Vec<Float> = right_hand_side
            .iter()
            .map(|value| Float::with_val(precision, value))
            .collect();
        for (column, pivot) in self.pivots.iter().copied().enumerate() {
            if pivot != column {
                solution.swap(column, pivot);
            }
        }
        for row in 0..self.dimension {
            for column in 0..row {
                let mut correction = self.lu[row * self.dimension + column].clone();
                correction *= &solution[column];
                solution[row] -= correction;
            }
        }
        for row in (0..self.dimension).rev() {
            for column in row + 1..self.dimension {
                let mut correction = self.lu[row * self.dimension + column].clone();
                correction *= &solution[column];
                solution[row] -= correction;
            }
            solution[row] /= &self.lu[row * self.dimension + row];
            if !solution[row].is_finite() {
                return Err(SolverError::NumericalBreakdown(
                    "shifted solve produced a nonfinite value".to_owned(),
                ));
            }
        }
        for (output, value) in output.iter_mut().zip(solution) {
            *output = if value.prec() == working_precision_bits {
                value
            } else {
                Float::with_val(working_precision_bits, value)
            };
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockShiftInvertConfigHp {
    /// Interval selects at most `requested_eigenpairs` observed Ritz values
    /// inside the interval nearest its midpoint; it does not count or enumerate
    /// every eigenvalue in the interval.
    pub target: EigenTarget,
    pub precision_bits: u32,
    pub requested_eigenpairs: usize,
    pub guard_eigenpairs: usize,
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
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
    pub maximum_projected_sweeps: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveBlockShiftInvertOptionsHp {
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    pub guard_eigenpairs: usize,
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
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
    pub maximum_projected_sweeps: usize,
    pub precision: PrecisionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockShiftInvertPrecisionAttemptHp {
    pub precision_bits: u32,
    pub status: ResultStatus,
    pub iterations: usize,
    pub operator_applications: usize,
    pub shifted_solves: usize,
    pub factorizations: usize,
    /// Maximum over requested simple pairs; absent if an unresolved boundary
    /// cluster hides any requested vector. Guards do not enter this value.
    pub maximum_requested_residual_norm: Option<String>,
    /// Maximum over every retained pair and unresolved cluster, including guards.
    #[serde(default)]
    pub maximum_retained_residual_norm: Option<String>,
    pub maximum_orthogonality_error: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub enum AdaptiveBlockShiftInvertResultHp {
    Converged {
        result: Box<BlockShiftInvertReportHp>,
        attempts: Vec<BlockShiftInvertPrecisionAttemptHp>,
    },
    Inconclusive {
        last_result: Option<Box<BlockShiftInvertReportHp>>,
        attempts: Vec<BlockShiftInvertPrecisionAttemptHp>,
        reason: String,
    },
}

#[derive(Clone, Debug)]
pub struct ShiftInvertEigenpairHp {
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub scaled_backward_error: Float,
    pub target_distance: Float,
    pub diagnostics: super::EigenpairDiagnostics<Float>,
    pub stopping_evidence: super::HpResidualAcceptance,
}

#[derive(Clone, Debug)]
pub struct ShiftInvertBoundaryClusterHp {
    pub first_retained_position: usize,
    pub last_retained_position: usize,
    pub requested_members: usize,
    pub dimension: usize,
    pub basis: Vec<Vec<Float>>,
    pub projected_operator: Vec<Float>,
    pub minimum_target_distance: Float,
    pub maximum_target_distance: Float,
    pub boundary_distance_gap: Float,
    pub maximum_residual_norm: Float,
}

#[derive(Clone, Debug)]
/// Convergence concerns retained Ritz residuals and stability. The projected
/// boundary cannot exclude an unvisited eigenspace or establish multiplicity.
pub struct BlockShiftInvertReportHp {
    /// True only after a full-space projection; false means global target
    /// selection and completeness remain unverified, even when Converged.
    pub global_target_ordering_established: bool,
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    /// Number selected inside the target; requested_eigenpairs is an upper bound for intervals.
    pub selected_eigenpairs: usize,
    pub interval_count_evidence: Option<super::BoundaryCountEvidenceHp>,
    pub algorithm: String,
    pub retained_eigenpairs: Vec<ShiftInvertEigenpairHp>,
    pub boundary_cluster: Option<ShiftInvertBoundaryClusterHp>,
    pub effective_boundary_cluster_tolerance: Float,
    pub iterations: usize,
    pub operator_applications: usize,
    pub shifted_solves: usize,
    pub factorizations: usize,
    pub projected_diagonalizations: usize,
    pub maximum_orthogonality_error: Float,
    pub maximum_ritz_value_stability: Float,
    pub estimated_peak_memory_bytes: u64,
    pub performance: SolverPerformanceTelemetry,
    pub factorization: ShiftInvertFactorizationDescriptorHp,
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
    residual_norm: Float,
    backward_error: Float,
    distance: Float,
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

fn parse_nonnegative(
    value: &DecimalLiteral,
    precision: u32,
    name: &str,
) -> Result<Float, SolverError> {
    let parsed = parse_finite(value, precision, name)?;
    if parsed < 0 {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be nonnegative"
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

fn target_shift(
    target: &EigenTarget,
    precision: u32,
) -> Result<(Float, Option<(Float, Float)>), SolverError> {
    match target {
        EigenTarget::SmallestMagnitude => Ok((zero(precision), None)),
        EigenTarget::ClosestTo { shift } => Ok((parse_finite(shift, precision, "target shift")?, None)),
        EigenTarget::Interval { lower, upper } => {
            target
                .validate()
                .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
            // The factorization shift is a point approximation to the interval
            // midpoint. Selection is a separate exact-decimal requirement:
            // round inward so an exterior Ritz point cannot pass membership.
            let lower_point = parse_finite(lower, precision, "interval lower")?;
            let upper_point = parse_finite(upper, precision, "interval upper")?;
            let midpoint = xc_numerics::mpfr_interval::MpfrInterval::new(
                lower_point, upper_point,
            )
            .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?
            .midpoint_point()
            .lower()
            .clone();
            let lower = Float::with_val_round(
                precision,
                Float::parse(lower.as_str()).map_err(|error| {
                    SolverError::InvalidConfiguration(error.to_string())
                })?,
                rug::float::Round::Up,
            ).0;
            let upper = Float::with_val_round(
                precision,
                Float::parse(upper.as_str()).map_err(|error| {
                    SolverError::InvalidConfiguration(error.to_string())
                })?,
                rug::float::Round::Down,
            ).0;
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(SolverError::InvalidConfiguration(
                    "shift-invert interval contains no finite coordinate at the working precision".to_owned(),
                ));
            }
            Ok((midpoint, Some((lower, upper))))
        }
        other => Err(SolverError::UnsupportedTarget(format!(
            "HP block shift-invert supports smallest magnitude, closest-to-shift, and interval targets, got {other:?}"
        ))),
    }
}

fn distance(value: &Float, shift: &Float, precision: u32) -> Float {
    let mut result = Float::with_val(precision, value);
    result -= shift;
    result.abs_mut();
    result
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
pub struct BlockShiftInvertSolverHp;

impl BlockShiftInvertSolverHp {
    pub fn solve(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        shifted_solver: &dyn ShiftInvertSolveHp,
        config: &BlockShiftInvertConfigHp,
    ) -> Result<BlockShiftInvertReportHp, SolverError> {
        self.solve_controlled(operator, shifted_solver, config, &CancellationToken::new())
    }

    pub fn solve_controlled(
        &self,
        operator: &dyn SymmetricOperator<Float>,
        shifted_solver: &dyn ShiftInvertSolveHp,
        config: &BlockShiftInvertConfigHp,
        cancellation: &CancellationToken,
    ) -> Result<BlockShiftInvertReportHp, SolverError> {
        let started = Instant::now();
        check_solver_cancellation(cancellation)?;
        let dimension = operator.dimension();
        let retained = config
            .requested_eigenpairs
            .saturating_add(config.guard_eigenpairs);
        if !(33..=1_000_000).contains(&config.precision_bits)
            || dimension == 0
            || retained > dimension
            || config.requested_eigenpairs == 0
            || (config.requested_eigenpairs < dimension && config.guard_eigenpairs == 0)
            || config.maximum_iterations == 0
            || config.minimum_iterations > config.maximum_iterations
            || config.maximum_projected_sweeps == 0
        {
            return Err(SolverError::InvalidConfiguration(
                "HP block shift-invert requires positive dimensions/counts, a guard for partial selection, precision in 33..=1000000 bits, and valid iteration bounds"
                    .to_owned(),
            ));
        }
        // Retain old directions together with inverse corrections. Account for
        // the complete bounded union and projected eigensolve before allocation.
        let maximum_trial = retained.saturating_mul(2).min(dimension);
        let scalar_bytes = u128::from(config.precision_bits).div_ceil(8) + 40;
        let working_entries = 12u128
            .saturating_mul(dimension as u128)
            .saturating_mul(retained as u128)
            .saturating_add(
                6u128
                    .saturating_mul(maximum_trial as u128)
                    .saturating_mul(maximum_trial as u128),
            );
        let working_bytes = working_entries.saturating_mul(scalar_bytes);
        if working_bytes > 1024u128 * 1024 * 1024 {
            return Err(SolverError::InvalidConfiguration(
                "block shifted-inverse union exceeds the 1 GiB working-memory admission limit"
                    .into(),
            ));
        }
        let mut maximum_trial_dimension = 0usize;
        let descriptor = shifted_solver.descriptor();
        descriptor.validate(config.precision_bits)?;
        if descriptor.dimension != dimension {
            return Err(SolverError::InvalidConfiguration(
                "shifted solver and operator dimensions differ".to_owned(),
            ));
        }
        let (mut shift, interval) = target_shift(&config.target, config.precision_bits)?;
        let declared_shift = parse_finite(
            &descriptor.shift,
            config.precision_bits,
            "factorization shift",
        )?;
        let matches = if let EigenTarget::Interval { lower, upper } = &config.target {
            // Verify the mathematical decimal midpoint before rounding it once.
            // Averaging separately rounded endpoints can differ by one ulp.
            let canonical = descriptor
                .shift
                .canonical()
                .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
            let negative = DecimalLiteral::new(
                if let Some(unsigned) = canonical.as_str().strip_prefix('-') {
                    unsigned.to_owned()
                } else {
                    format!("-{}", canonical.as_str())
                },
            )
            .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
            let zero = DecimalLiteral::new("0").expect("zero literal");
            let exact = zero
                .cmp_sum_many(&[lower, upper, &negative, &negative])
                .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?
                == std::cmp::Ordering::Equal;
            if exact {
                shift = declared_shift.clone();
            }
            exact
        } else {
            declared_shift == shift
        };
        if !matches {
            return Err(SolverError::InvalidConfiguration(
                "factorization shift does not match the target shift or interval midpoint"
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
        let mut basis = Vec::with_capacity(retained);
        for column in 0..retained {
            let candidate = (0..dimension)
                .map(|row| Float::with_val(config.precision_bits, row + 1).pow((column + 1) as u32))
                .collect();
            let _ = add_orthonormal(candidate, &mut basis, config.precision_bits);
        }
        for coordinate in 0..dimension {
            if basis.len() == retained {
                break;
            }
            let mut candidate = vec![zero(config.precision_bits); dimension];
            candidate[coordinate].assign(1);
            let _ = add_orthonormal(candidate, &mut basis, config.precision_bits);
        }
        if basis.len() != retained {
            return Err(SolverError::NumericalBreakdown(
                "deterministic shift-invert seed failed to span the retained block".to_owned(),
            ));
        }
        let mut previous_values: Option<Vec<Float>> = None;
        let mut operator_applications = 0usize;
        let mut shifted_solves = 0usize;
        for iteration in 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            let mut inverse_basis: Vec<Vec<Float>> = Vec::with_capacity(maximum_trial);
            for vector in &basis {
                let mut solved = vec![zero(config.precision_bits); dimension];
                shifted_solver.solve_shifted(vector, &mut solved, config.precision_bits)?;
                shifted_solves += 1;
                if solved
                    .iter()
                    .any(|value| !value.is_finite() || value.prec() < config.precision_bits)
                {
                    return Err(SolverError::NumericalBreakdown(
                        "shifted solve returned nonfinite or insufficient-precision output"
                            .to_owned(),
                    ));
                }
                // Remove the dominant inverse component before normalization;
                // its magnitude must not erase a useful smaller correction.
                for _ in 0..2 {
                    for q in &inverse_basis {
                        let coefficient = dot(q, &solved, config.precision_bits);
                        for (x, b) in solved.iter_mut().zip(q) {
                            *x -= Float::with_val(config.precision_bits, b * &coefficient);
                        }
                    }
                }
                let _ = add_orthonormal(solved, &mut inverse_basis, config.precision_bits);
            }
            for old in &basis {
                if inverse_basis.len() == maximum_trial {
                    break;
                }
                let _ = add_orthonormal(old.clone(), &mut inverse_basis, config.precision_bits);
            }
            if inverse_basis.len() < retained {
                return Err(SolverError::PrecisionExhausted(
                    "shift-invert union cannot resolve the retained rank".into(),
                ));
            }
            let trial_dimension = inverse_basis.len();
            maximum_trial_dimension = maximum_trial_dimension.max(trial_dimension);
            let mut applied = Vec::with_capacity(trial_dimension);
            for vector in &inverse_basis {
                let output = super::hp_checked_action(operator, vector, config.precision_bits)?;
                operator_applications += 1;
                applied.push(output);
            }
            let mut projected =
                vec![zero(config.precision_bits); trial_dimension * trial_dimension];
            for row in 0..trial_dimension {
                for column in 0..=row {
                    let value = dot(&inverse_basis[row], &applied[column], config.precision_bits);
                    projected[row * trial_dimension + column] = value.clone();
                    projected[column * trial_dimension + row] = value;
                }
            }
            let (values, vectors) = symmetric_jacobi_eigensystem(
                &projected,
                trial_dimension,
                config.precision_bits,
                config.maximum_projected_sweeps,
            )?;
            let mut order: Vec<usize> = (0..trial_dimension).collect();
            order.sort_by(|left, right| {
                let left_inside = interval.as_ref().is_none_or(|(lower, upper)| {
                    values[*left] >= *lower && values[*left] <= *upper
                });
                let right_inside = interval.as_ref().is_none_or(|(lower, upper)| {
                    values[*right] >= *lower && values[*right] <= *upper
                });
                right_inside.cmp(&left_inside).then_with(|| {
                    let left_distance = distance(&values[*left], &shift, config.precision_bits);
                    let right_distance = distance(&values[*right], &shift, config.precision_bits);
                    left_distance
                        .partial_cmp(&right_distance)
                        .unwrap_or(Ordering::Equal)
                })
            });
            let mut states = Vec::with_capacity(trial_dimension);
            for index in order.into_iter().take(retained) {
                let mut vector = vec![zero(config.precision_bits); dimension];
                for column in 0..trial_dimension {
                    let coefficient = &vectors[index][column];
                    for row in 0..dimension {
                        let mut contribution = inverse_basis[column][row].clone();
                        contribution *= coefficient;
                        vector[row] += contribution;
                    }
                }
                let applied_vector =
                    super::hp_checked_action(operator, &vector, config.precision_bits)?;
                operator_applications += 1;
                let value = dot(&vector, &applied_vector, config.precision_bits)
                    / dot(&vector, &vector, config.precision_bits);
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
                let target_distance = distance(&value, &shift, config.precision_bits);
                states.push(RitzState {
                    vector,
                    applied: applied_vector,
                    value,
                    residual_norm,
                    backward_error,
                    distance: target_distance,
                });
            }
            let maximum_stability = previous_values
                .as_ref()
                .map(|previous| {
                    let mut maximum = zero(config.precision_bits);
                    for index in 0..retained {
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
            let mut selected_eigenpairs =
                interval
                    .as_ref()
                    .map_or(config.requested_eigenpairs, |(lower, upper)| {
                        states
                            .iter()
                            .take_while(|state| state.value >= *lower && state.value <= *upper)
                            .count()
                            .min(config.requested_eigenpairs)
                    });
            let residuals_converged = states.iter().all(|state| {
                state.residual_norm <= absolute_tolerance
                    || state.backward_error <= backward_tolerance
            });
            let converged = iteration >= config.minimum_iterations
                && residuals_converged
                && maximum_stability <= stability_tolerance;
            if converged || iteration == config.maximum_iterations {
                let cluster_tolerance = super::hp_effective_cluster_tolerance(
                    &cluster_tolerance,
                    states.iter().map(|state| &state.value),
                    dimension,
                    config.precision_bits,
                );
                let interval_selection_requires_count = selected_eigenpairs
                    < config.requested_eigenpairs
                    && matches!(config.target, EigenTarget::Interval { .. });
                let mut interval_count_evidence = None;
                if interval_selection_requires_count {
                    if let EigenTarget::Interval { lower, upper } = &config.target {
                        let mut proof = super::hp_boundary_count::interval_count(
                            operator,
                            lower,
                            upper,
                            selected_eigenpairs,
                            config.precision_bits,
                        )?;
                        if let super::BoundaryCountEvidenceHp::CountMismatch { observed, .. } =
                            proof
                        {
                            // Endpoint roots may have rounded Ritz points just outside.
                            // Exact source counts determine capacity, never a widened point test.
                            selected_eigenpairs = observed.min(config.requested_eigenpairs);
                            proof = super::hp_boundary_count::interval_count(
                                operator,
                                lower,
                                upper,
                                observed,
                                config.precision_bits,
                            )?;
                        }
                        if proof.establishes_requested_count()
                            && selected_eigenpairs > 0
                            && selected_eigenpairs < states.len()
                        {
                            let values: Vec<_> =
                                states.iter().map(|state| state.value.clone()).collect();
                            let prefix = super::hp_boundary_count::boundary_count(
                                operator,
                                &EigenTarget::ClosestTo {
                                    shift: descriptor.shift.clone(),
                                },
                                selected_eigenpairs,
                                &values,
                                &cluster_tolerance,
                                config.precision_bits,
                            )?;
                            if !prefix.establishes_requested_count() {
                                proof = prefix;
                            }
                        }
                        interval_count_evidence = Some(proof);
                    }
                }
                let mut boundary_cluster = None;
                if selected_eigenpairs > 0 && selected_eigenpairs < retained {
                    let requested = selected_eigenpairs - 1;
                    let guard = selected_eigenpairs;
                    let mut gap = states[guard].distance.clone();
                    gap -= &states[requested].distance;
                    gap.abs_mut();
                    if gap <= cluster_tolerance {
                        let mut first = requested;
                        while first > 0 {
                            let mut adjacent = states[first].distance.clone();
                            adjacent -= &states[first - 1].distance;
                            adjacent.abs_mut();
                            if adjacent > cluster_tolerance {
                                break;
                            }
                            first -= 1;
                        }
                        let mut last = guard;
                        while last + 1 < retained {
                            let mut adjacent = states[last + 1].distance.clone();
                            adjacent -= &states[last].distance;
                            adjacent.abs_mut();
                            if adjacent > cluster_tolerance {
                                break;
                            }
                            last += 1;
                        }
                        let cluster_dimension = last - first + 1;
                        let mut projected_operator = vec![
                            zero(config.precision_bits);
                            cluster_dimension * cluster_dimension
                        ];
                        let mut maximum_residual = zero(config.precision_bits);
                        for row in first..=last {
                            if states[row].residual_norm > maximum_residual {
                                maximum_residual = states[row].residual_norm.clone();
                            }
                            for column in first..=last {
                                projected_operator
                                    [(row - first) * cluster_dimension + column - first] = dot(
                                    &states[row].vector,
                                    &states[column].applied,
                                    config.precision_bits,
                                );
                            }
                        }
                        boundary_cluster = Some(ShiftInvertBoundaryClusterHp {
                            first_retained_position: first,
                            last_retained_position: last,
                            requested_members: selected_eigenpairs - first,
                            dimension: cluster_dimension,
                            basis: states[first..=last]
                                .iter()
                                .map(|state| state.vector.clone())
                                .collect(),
                            projected_operator,
                            minimum_target_distance: states[first].distance.clone(),
                            maximum_target_distance: states[last].distance.clone(),
                            boundary_distance_gap: gap,
                            maximum_residual_norm: maximum_residual,
                        });
                    }
                }
                let cluster_range = boundary_cluster.as_ref().map(|cluster| {
                    cluster.first_retained_position..=cluster.last_retained_position
                });
                let (orthogonality_errors, maximum_orthogonality_error) =
                    orthogonality_errors(&states, config.precision_bits);
                let retained_eigenpairs = states
                    .iter()
                    .enumerate()
                    .filter(|(position, _)| {
                        !cluster_range
                            .as_ref()
                            .is_some_and(|range| range.contains(position))
                    })
                    .map(|(position, state)| ShiftInvertEigenpairHp {
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
                        target_distance: state.distance.clone(),
                        diagnostics: super::EigenpairDiagnostics {
                            absolute_residual: state.residual_norm.clone(),
                            relative_residual: state.backward_error.clone(),
                            scaled_backward_error: state.backward_error.clone(),
                            orthogonality_error: orthogonality_errors[position].clone(),
                        },
                    })
                    .collect();
                let incomplete_interval_count = interval_selection_requires_count
                    && !interval_count_evidence
                        .as_ref()
                        .is_some_and(|proof| proof.establishes_requested_count());
                let (status, termination) = if converged && incomplete_interval_count {
                    (
                        ResultStatus::UnresolvedEigenspace,
                        TerminationReason::UnresolvedEigenspace,
                    )
                } else if converged && selected_eigenpairs == 0 {
                    (ResultStatus::Converged, TerminationReason::EmptySelection)
                } else if converged && boundary_cluster.is_some() {
                    (
                        ResultStatus::UnresolvedCluster,
                        TerminationReason::UnresolvedCluster,
                    )
                } else if converged {
                    (
                        ResultStatus::Converged,
                        super::hp_block_termination(
                            states
                                .iter()
                                .take(selected_eigenpairs)
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
                let estimated_peak_memory_bytes = working_bytes as u64;
                let performance = SolverPerformanceTelemetry {
                    operator_applications: operator_applications as u64,
                    metric_applications: 0,
                    preconditioner_applications: 0,
                    factorizations: 0,
                    iterations: iteration as u64,
                    precision_escalations: 0,
                    estimated_peak_memory_bytes,
                    elapsed_nanoseconds: started.elapsed().as_nanos(),
                };
                let mut provenance = SolverProvenance::current_package("rug_mpfr");
                provenance.precision_bits = Some(config.precision_bits);
                return Ok(BlockShiftInvertReportHp {
                    global_target_ordering_established: maximum_trial_dimension == dimension
                        || interval_count_evidence
                            .as_ref()
                            .is_some_and(|proof| proof.establishes_requested_count()),
                    target: config.target.clone(),
                    requested_eigenpairs: config.requested_eigenpairs,
                    selected_eigenpairs,
                    interval_count_evidence,
                    algorithm: "block_shift_invert_union_fresh_images_guard_counts_hp_v3".into(),
                    retained_eigenpairs,
                    boundary_cluster,
                    effective_boundary_cluster_tolerance: cluster_tolerance,
                    iterations: iteration,
                    operator_applications,
                    shifted_solves,
                    factorizations: 0,
                    projected_diagonalizations: iteration,
                    maximum_orthogonality_error,
                    maximum_ritz_value_stability: maximum_stability,
                    estimated_peak_memory_bytes,
                    performance,
                    factorization: descriptor,
                    status,
                    termination,
                    assurance: AssuranceLevel::Computed,
                    provenance,
                });
            }
            previous_values = Some(states.iter().map(|state| state.value.clone()).collect());
            basis = states.into_iter().map(|state| state.vector).collect();
        }
        Err(SolverError::NonConvergence(
            "HP block shift-invert exhausted its iteration loop".to_owned(),
        ))
    }
}

fn maximum_requested_shift_invert_residual(report: &BlockShiftInvertReportHp) -> Option<String> {
    if report
        .boundary_cluster
        .as_ref()
        .is_some_and(|c| c.requested_members != 0)
    {
        return None;
    }
    if report.retained_eigenpairs.len() < report.selected_eigenpairs {
        return None;
    }
    report
        .retained_eigenpairs
        .iter()
        .take(report.selected_eigenpairs)
        .map(|pair| &pair.residual_norm)
        .max_by(|a, b| a.total_cmp(b))
        .map(ToString::to_string)
}

fn maximum_shift_invert_residual(report: &BlockShiftInvertReportHp) -> Option<String> {
    let precision = report.provenance.precision_bits?;
    let mut maximum = zero(precision);
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

fn update_adaptive_performance(
    report: &mut BlockShiftInvertReportHp,
    attempts: &[BlockShiftInvertPrecisionAttemptHp],
    started: &Instant,
) {
    report.performance.operator_applications = attempts
        .iter()
        .map(|attempt| attempt.operator_applications as u64)
        .sum();
    report.performance.factorizations = attempts
        .iter()
        .map(|attempt| attempt.factorizations as u64)
        .sum();
    report.performance.iterations = attempts
        .iter()
        .map(|attempt| attempt.iterations as u64)
        .sum();
    report.performance.precision_escalations = attempts.len().saturating_sub(1) as u64;
    report.performance.elapsed_nanoseconds = started.elapsed().as_nanos();
}

/// Refactor and rerun block shift-invert at each precision in the policy.
/// No lower-precision factorization is reused after escalation.
pub fn solve_block_shift_invert_adaptive_hp(
    operator: &dyn SymmetricOperator<Float>,
    factory: &dyn ShiftInvertFactoryHp,
    options: &AdaptiveBlockShiftInvertOptionsHp,
) -> Result<AdaptiveBlockShiftInvertResultHp, SolverError> {
    let started = Instant::now();
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
            "adaptive HP shift-invert precision must exceed 32 bits after guard bits".to_owned(),
        ));
    }
    let mut attempts = Vec::new();
    let mut last_result = None;
    loop {
        let config = BlockShiftInvertConfigHp {
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
        let mut completed_factorizations = 0;
        let outcome = factory
            .factor_at_precision(precision_bits)
            .and_then(|factorization| {
                completed_factorizations = 1;
                BlockShiftInvertSolverHp.solve(operator, factorization.as_ref(), &config)
            });
        match outcome {
            Ok(mut result) => {
                result.factorizations = completed_factorizations;
                let converged = result.status == ResultStatus::Converged;
                let reason = match &result.status {
                    ResultStatus::Converged => {
                        "each requested residual or backward-error check, stability, interval, and boundary checks passed"
                            .to_owned()
                    }
                    ResultStatus::UnresolvedCluster => {
                        "the requested/guard boundary has indistinguishable target distance"
                            .to_owned()
                    }
                    _ => "the iteration limit was reached before all checks passed".to_owned(),
                };
                attempts.push(BlockShiftInvertPrecisionAttemptHp {
                    precision_bits,
                    status: result.status.clone(),
                    iterations: result.iterations,
                    operator_applications: result.operator_applications,
                    shifted_solves: result.shifted_solves,
                    factorizations: result.factorizations,
                    maximum_requested_residual_norm: maximum_requested_shift_invert_residual(
                        &result,
                    ),
                    maximum_retained_residual_norm: maximum_shift_invert_residual(&result),
                    maximum_orthogonality_error: Some(
                        result.maximum_orthogonality_error.to_string(),
                    ),
                    reason,
                });
                update_adaptive_performance(&mut result, &attempts, &started);
                if converged {
                    return Ok(AdaptiveBlockShiftInvertResultHp::Converged {
                        result: Box::new(result),
                        attempts,
                    });
                }
                last_result = Some(Box::new(result));
            }
            Err(error @ SolverError::InvalidConfiguration(_))
            | Err(error @ SolverError::UnsupportedTarget(_))
            | Err(error @ SolverError::Cancelled(_)) => return Err(error),
            Err(error @ SolverError::PrecisionExhausted(_)) => {
                attempts.push(BlockShiftInvertPrecisionAttemptHp {
                    precision_bits,
                    status: ResultStatus::InsufficientPrecision,
                    iterations: 0,
                    operator_applications: 0,
                    shifted_solves: 0,
                    factorizations: completed_factorizations,
                    maximum_requested_residual_norm: None,
                    maximum_retained_residual_norm: None,
                    maximum_orthogonality_error: None,
                    reason: error.to_string(),
                })
            }
            Err(error) => {
                let reason = error.to_string();
                attempts.push(BlockShiftInvertPrecisionAttemptHp {
                    precision_bits,
                    status: ResultStatus::Failed,
                    iterations: 0,
                    operator_applications: 0,
                    shifted_solves: 0,
                    factorizations: completed_factorizations,
                    maximum_requested_residual_norm: None,
                    maximum_retained_residual_norm: None,
                    maximum_orthogonality_error: None,
                    reason: error.to_string(),
                });
                return Ok(AdaptiveBlockShiftInvertResultHp::Inconclusive { last_result, attempts,
                    reason: format!("execution failed without evidence that precision escalation remedies it: {reason}") });
            }
        }
        let Some(next_bits) = options.precision.next_bits(precision_bits) else {
            if let Some(result) = last_result.as_mut() {
                update_adaptive_performance(result, &attempts, &started);
            }
            return Ok(AdaptiveBlockShiftInvertResultHp::Inconclusive {
                last_result,
                attempts,
                reason: format!(
                    "block shift-invert did not converge at maximum precision {}",
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
    use xc_operator::DenseSymmetricHp;

    fn diagonal(precision: u32, values: &[f64]) -> Vec<Float> {
        let dimension = values.len();
        let mut matrix = vec![zero(precision); dimension * dimension];
        for (index, value) in values.iter().enumerate() {
            matrix[index * dimension + index].assign(*value);
        }
        matrix
    }

    fn config(target: EigenTarget) -> BlockShiftInvertConfigHp {
        BlockShiftInvertConfigHp {
            target,
            precision_bits: 192,
            requested_eigenpairs: 2,
            guard_eigenpairs: 1,
            absolute_residual_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            boundary_cluster_tolerance: DecimalLiteral::new("1e-25").unwrap(),
            maximum_iterations: 40,
            minimum_iterations: 2,
            maximum_projected_sweeps: 100,
        }
    }

    #[test]
    fn adaptive_requested_residual_excludes_guard_and_unresolved_cluster() {
        let p = 192;
        let matrix = diagonal(p, &[0.1, 2.0, 7.0]);
        let operator =
            DenseSymmetricHp::new("residual-scope", 3, matrix.clone(), p, &zero(p)).unwrap();
        let factor = DenseShiftInvertFactorizationHp::factor(
            "residual-scope",
            3,
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            p,
        )
        .unwrap();
        let mut cfg = config(EigenTarget::SmallestMagnitude);
        cfg.requested_eigenpairs = 1;
        cfg.guard_eigenpairs = 2;
        let mut report = BlockShiftInvertSolverHp
            .solve(&operator, &factor, &cfg)
            .unwrap();
        report.retained_eigenpairs[0].residual_norm = Float::with_val(p, 1) / 1024u32;
        report.retained_eigenpairs[1].residual_norm = Float::with_val(p, 1);
        assert_eq!(
            maximum_requested_shift_invert_residual(&report),
            Some((Float::with_val(p, 1) / 1024u32).to_string())
        );
        assert_eq!(
            maximum_shift_invert_residual(&report),
            Some(Float::with_val(p, 1).to_string())
        );
        report.boundary_cluster = Some(ShiftInvertBoundaryClusterHp {
            first_retained_position: 0,
            last_retained_position: 1,
            requested_members: 1,
            dimension: 2,
            basis: vec![],
            projected_operator: vec![],
            minimum_target_distance: zero(p),
            maximum_target_distance: zero(p),
            boundary_distance_gap: zero(p),
            maximum_residual_norm: Float::with_val(p, 2),
        });
        assert_eq!(maximum_requested_shift_invert_residual(&report), None);
        assert_eq!(
            maximum_shift_invert_residual(&report),
            Some(Float::with_val(p, 2).to_string())
        );
    }

    #[test]
    fn block_shift_invert_finds_near_zero_pairs_without_full_spectrum() {
        let precision = 192;
        let values = [-3.0, -0.2, 0.1, 2.0, 7.0];
        let matrix = diagonal(precision, &values);
        let operator = DenseSymmetricHp::new(
            "near_zero",
            values.len(),
            matrix.clone(),
            precision,
            &zero(precision),
        )
        .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "near_zero_lu",
            values.len(),
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            precision,
        )
        .unwrap();
        let report = BlockShiftInvertSolverHp
            .solve(
                &operator,
                &factorization,
                &config(EigenTarget::SmallestMagnitude),
            )
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert_eq!(report.retained_eigenpairs.len(), 3);
        for (pair, expected) in report.retained_eigenpairs.iter().take(2).zip([0.1, -0.2]) {
            let mut error = pair.eigenvalue.clone();
            error -= expected;
            error.abs_mut();
            assert!(error < Float::with_val(precision, 1e-25));
            assert!(pair.residual_norm < Float::with_val(precision, 1e-25));
            assert_eq!(pair.diagnostics.absolute_residual, pair.residual_norm);
            assert_eq!(
                pair.diagnostics.scaled_backward_error,
                pair.scaled_backward_error
            );
            assert!(pair.diagnostics.orthogonality_error <= report.maximum_orthogonality_error);
        }
        assert!(report.shifted_solves > 0);
        assert_eq!(report.factorizations, 0);
        assert_eq!(
            report.performance.operator_applications,
            report.operator_applications as u64
        );
        assert_eq!(report.performance.factorizations, 0);
        assert_eq!(report.performance.iterations, report.iterations as u64);
        assert_eq!(report.performance.precision_escalations, 0);
        assert_eq!(
            report.performance.estimated_peak_memory_bytes,
            report.estimated_peak_memory_bytes
        );
    }

    #[test]
    fn block_shift_invert_interval_returns_values_inside_requested_window() {
        let precision = 192;
        let values = [-3.0, -0.2, 0.1, 2.0, 7.0];
        let matrix = diagonal(precision, &values);
        let operator = DenseSymmetricHp::new(
            "interval",
            values.len(),
            matrix.clone(),
            precision,
            &zero(precision),
        )
        .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "interval_lu",
            values.len(),
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            precision,
        )
        .unwrap();
        let target = EigenTarget::Interval {
            lower: DecimalLiteral::new("-0.3").unwrap(),
            upper: DecimalLiteral::new("0.3").unwrap(),
        };
        let report = BlockShiftInvertSolverHp
            .solve(&operator, &factorization, &config(target))
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report
            .retained_eigenpairs
            .iter()
            .take(2)
            .all(|pair| pair.eigenvalue > -0.3 && pair.eigenvalue < 0.3));
    }

    #[test]
    fn block_shift_invert_reports_equidistant_boundary_subspace() {
        let precision = 192;
        let values = [-2.0, -0.1, 0.1, 3.0];
        let matrix = diagonal(precision, &values);
        let operator = DenseSymmetricHp::new(
            "equidistant",
            values.len(),
            matrix.clone(),
            precision,
            &zero(precision),
        )
        .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "equidistant_lu",
            values.len(),
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            precision,
        )
        .unwrap();
        let mut config = config(EigenTarget::SmallestMagnitude);
        config.requested_eigenpairs = 1;
        let report = BlockShiftInvertSolverHp
            .solve(&operator, &factorization, &config)
            .unwrap();
        assert_eq!(report.status, ResultStatus::UnresolvedCluster);
        let cluster = report.boundary_cluster.unwrap();
        assert_eq!(cluster.dimension, 2);
        assert_eq!(cluster.requested_members, 1);
        assert_eq!(cluster.basis.len(), 2);
        assert_eq!(cluster.projected_operator.len(), 4);
    }

    #[test]
    fn adaptive_shift_invert_refactors_and_preserves_boundary_history() {
        use xc_core::PrecisionEscalation;

        let source_precision = 192;
        let values = [-2.0, -0.1, 0.1, 3.0];
        let matrix = diagonal(source_precision, &values);
        let operator = DenseSymmetricHp::new(
            "adaptive_equidistant",
            values.len(),
            matrix.clone(),
            source_precision,
            &zero(source_precision),
        )
        .unwrap();
        let factory = DenseShiftInvertFactoryHp::new(
            "adaptive_equidistant_lu",
            values.len(),
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            source_precision,
        )
        .unwrap();
        let options = AdaptiveBlockShiftInvertOptionsHp {
            target: EigenTarget::SmallestMagnitude,
            requested_eigenpairs: 1,
            guard_eigenpairs: 1,
            absolute_residual_tolerance: DecimalLiteral::new("1e-25").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-25").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-25").unwrap(),
            boundary_cluster_tolerance: DecimalLiteral::new("1e-20").unwrap(),
            maximum_iterations: 40,
            minimum_iterations: 2,
            maximum_projected_sweeps: 100,
            precision: PrecisionPolicy {
                initial_bits: 64,
                maximum_bits: 192,
                guard_bits: 0,
                escalation: PrecisionEscalation::AddBits(128),
            },
        };
        let outcome = solve_block_shift_invert_adaptive_hp(&operator, &factory, &options).unwrap();
        let AdaptiveBlockShiftInvertResultHp::Inconclusive {
            last_result,
            attempts,
            ..
        } = outcome
        else {
            panic!("equidistant target boundary must remain inconclusive");
        };
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].precision_bits, 64);
        assert_eq!(attempts[1].precision_bits, 192);
        assert_eq!(attempts[1].status, ResultStatus::UnresolvedCluster);
        assert_eq!(attempts[1].factorizations, 1);
        let last_result = last_result.expect("last unresolved report must be retained");
        assert_eq!(last_result.factorization.factorization_precision_bits, 192);
        assert!(last_result.boundary_cluster.is_some());
        assert_eq!(last_result.performance.precision_escalations, 1);
        assert_eq!(last_result.performance.factorizations, 2);
        assert_eq!(
            last_result.performance.operator_applications,
            attempts
                .iter()
                .map(|attempt| attempt.operator_applications as u64)
                .sum::<u64>()
        );
        assert!(last_result.performance.elapsed_nanoseconds > 0);
    }

    #[test]
    fn dense_shift_invert_rejects_a_shift_on_an_eigenvalue() {
        let precision = 128;
        let matrix = diagonal(precision, &[1.0, 2.0, 3.0]);
        let error = DenseShiftInvertFactorizationHp::factor(
            "singular",
            3,
            &matrix,
            DecimalLiteral::new("2").unwrap(),
            precision,
        )
        .unwrap_err();
        assert!(matches!(error, SolverError::NumericalBreakdown(_)));
    }

    #[test]
    fn factorization_shift_must_match_target_shift() {
        let precision = 128;
        let matrix = diagonal(precision, &[1.0, 2.0, 3.0]);
        let operator =
            DenseSymmetricHp::new("mismatch", 3, matrix.clone(), precision, &zero(precision))
                .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "mismatch_lu",
            3,
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            precision,
        )
        .unwrap();
        let target = EigenTarget::ClosestTo {
            shift: DecimalLiteral::new("0.5").unwrap(),
        };
        let error = BlockShiftInvertSolverHp
            .solve(&operator, &factorization, &config(target))
            .unwrap_err();
        assert!(matches!(error, SolverError::InvalidConfiguration(_)));
    }

    #[test]
    fn dense_factorization_solves_shifted_system() {
        let precision = 128;
        let matrix = vec![
            Float::with_val(precision, 4),
            Float::with_val(precision, 1),
            Float::with_val(precision, 1),
            Float::with_val(precision, 3),
        ];
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "dense",
            2,
            &matrix,
            DecimalLiteral::new("1").unwrap(),
            precision,
        )
        .unwrap();
        let right_hand_side = [Float::with_val(precision, 4), Float::with_val(precision, 3)];
        let mut solution = vec![zero(precision); 2];
        factorization
            .solve_shifted(&right_hand_side, &mut solution, precision)
            .unwrap();
        let mut first = solution[0].clone();
        first -= 1;
        let mut second = solution[1].clone();
        second -= 1;
        assert!(first.abs() < Float::with_val(precision, 1e-30));
        assert!(second.abs() < Float::with_val(precision, 1e-30));
        assert_eq!(factorization.descriptor().dimension, 2);
    }

    #[test]
    fn owned_factory_factorization_is_exactly_equivalent_to_borrowed_input() {
        let precision = 257;
        let dimension = 5;
        let matrix = (0..dimension * dimension)
            .map(|index| {
                let row = index / dimension;
                let column = index % dimension;
                let low = row.min(column);
                let high = row.max(column);
                let mut value = Float::with_val(precision, low * 13 + high * 7 + 1);
                value /= 31;
                if row == column {
                    value += 7;
                }
                value
            })
            .collect::<Vec<_>>();
        let shift = DecimalLiteral::new("0.125").unwrap();
        let borrowed = DenseShiftInvertFactorizationHp::factor(
            "equivalent",
            dimension,
            &matrix,
            shift.clone(),
            precision,
        )
        .unwrap();
        let owned = DenseShiftInvertFactorizationHp::factor_owned(
            "equivalent",
            dimension,
            matrix
                .iter()
                .map(|value| Float::with_val(precision, value))
                .collect(),
            shift,
            precision,
        )
        .unwrap();
        assert_eq!(owned.lu, borrowed.lu);
        assert_eq!(owned.pivots, borrowed.pivots);
        assert_eq!(owned.descriptor, borrowed.descriptor);

        let right_hand_side = (0..dimension)
            .map(|index| {
                let mut value = Float::with_val(precision, index + 1);
                value /= 17;
                value
            })
            .collect::<Vec<_>>();
        let mut borrowed_solution = vec![zero(precision); dimension];
        let mut owned_solution = vec![zero(precision); dimension];
        borrowed
            .solve_shifted(&right_hand_side, &mut borrowed_solution, precision)
            .unwrap();
        owned
            .solve_shifted(&right_hand_side, &mut owned_solution, precision)
            .unwrap();
        assert_eq!(owned_solution, borrowed_solution);
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
mod hp_seed_range_contract {
    use super::*;
    #[test]
    fn sixteen_vector_block_does_not_compute_hp_seed_powers_as_machine_integers() {
        let p = 256;
        let n = 16;
        let data = (0..n * n)
            .map(|i| Float::with_val(p, usize::from(i / n == i % n)))
            .collect::<Vec<_>>();
        let operator = xc_operator::DenseSymmetricHp::new(
            "identity",
            n,
            data.clone(),
            p,
            &Float::with_val(p, 0),
        )
        .unwrap();
        let factor = DenseShiftInvertFactorizationHp::factor(
            "identity inverse",
            n,
            &data,
            DecimalLiteral::new("0").unwrap(),
            p,
        )
        .unwrap();
        let tolerance = DecimalLiteral::new("1e-40").unwrap();
        let config = BlockShiftInvertConfigHp {
            target: EigenTarget::SmallestMagnitude,
            precision_bits: p,
            requested_eigenpairs: 15,
            guard_eigenpairs: 1,
            absolute_residual_tolerance: tolerance.clone(),
            scaled_backward_error_tolerance: tolerance.clone(),
            ritz_value_stability_tolerance: tolerance.clone(),
            boundary_cluster_tolerance: tolerance,
            maximum_iterations: 1,
            minimum_iterations: 1,
            maximum_projected_sweeps: 100,
        };
        let report = BlockShiftInvertSolverHp
            .solve(&operator, &factor, &config)
            .unwrap();
        assert!(report.retained_eigenpairs.is_empty());
        let cluster = report.boundary_cluster.expect("repeated identity spectrum");
        assert_eq!(cluster.dimension, n);
        assert_eq!(cluster.basis.len(), n);
        let tolerance = Float::with_val(p, 1) >> 160u32;
        assert!(cluster.maximum_residual_norm < tolerance);
        for (i, value) in cluster.projected_operator.iter().enumerate() {
            let expected = usize::from(i / n == i % n);
            assert!((Float::with_val(p, value) - expected).abs() < tolerance);
        }
    }
}

#[cfg(test)]
mod exact_interval_target_tests {
    use super::*;
    use xc_operator::DenseSymmetricHp;

    fn target(lower: &str, upper: &str) -> EigenTarget {
        EigenTarget::Interval {
            lower: DecimalLiteral::new(lower).unwrap(),
            upper: DecimalLiteral::new(upper).unwrap(),
        }
    }
    fn scalar_report(lower: &str, upper: &str, shift: &str) -> BlockShiftInvertReportHp {
        let p = 64;
        let matrix = vec![Float::with_val(p, 1)];
        let operator = DenseSymmetricHp::new("exact-one", 1, matrix.clone(), p, &zero(p)).unwrap();
        let factor = DenseShiftInvertFactorizationHp::factor(
            "exact-scalar-factor",
            1,
            &matrix,
            DecimalLiteral::new(shift).unwrap(),
            p,
        )
        .unwrap();
        let config = BlockShiftInvertConfigHp {
            target: target(lower, upper),
            precision_bits: p,
            requested_eigenpairs: 1,
            guard_eigenpairs: 0,
            absolute_residual_tolerance: DecimalLiteral::new("1e-15").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-15").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-15").unwrap(),
            boundary_cluster_tolerance: DecimalLiteral::new("1e-12").unwrap(),
            maximum_iterations: 3,
            minimum_iterations: 2,
            maximum_projected_sweeps: 20,
        };
        BlockShiftInvertSolverHp
            .solve(&operator, &factor, &config)
            .unwrap()
    }
    #[test]
    fn upper_decimal_boundary_cannot_admit_one_from_outside() {
        let report = scalar_report(
            "0",
            "0.999999999999999999999999999999",
            "0.4999999999999999999999999999995",
        );
        assert_eq!(report.retained_eigenpairs[0].eigenvalue, 1);
        assert_ne!(report.status, ResultStatus::Converged);
    }
    #[test]
    fn lower_decimal_boundary_cannot_admit_one_from_outside() {
        let report = scalar_report(
            "1.000000000000000000000000000001",
            "2",
            "1.5000000000000000000000000000005",
        );
        assert_eq!(report.retained_eigenpairs[0].eigenvalue, 1);
        assert_ne!(report.status, ResultStatus::Converged);
    }
    #[test]
    fn exact_closed_endpoints_remain_admitted() {
        for (lo, hi, shift) in [("0", "1", "0.5"), ("1", "2", "1.5")] {
            let report = scalar_report(lo, hi, shift);
            assert_eq!(report.status, ResultStatus::Converged);
            assert_eq!(report.retained_eigenpairs[0].eigenvalue, 1);
            assert_eq!(report.retained_eigenpairs[0].residual_norm, 0);
        }
    }
    #[test]
    fn finite_large_interval_has_a_finite_midpoint() {
        let p = 64;
        let lo = Float::with_val(p, 1) << (rug::float::exp_max() - 1);
        let hi = Float::with_val(p, &lo * 1.5);
        assert!(lo.is_finite() && hi.is_finite());
        let (mid, _) = target_shift(&target(&lo.to_string(), &hi.to_string()), p).unwrap();
        assert!(mid.is_finite() && mid >= lo && mid <= hi);
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
