// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Multi-solver framework.
//!
//! The f64 solvers in this crate are deterministic reference and discovery
//! implementations.  They are never a silent replacement for an HP request.
//! The same result contracts are intended for the HP and certified backends.

use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
#[cfg(feature = "hp-reference")]
use xc_core::EigenpairDiagnostics;
use xc_core::{
    AssuranceLevel, CacheAccessMode, CancellationToken, CapabilityCatalog, CertificationCapability,
    ConfigDigest, EigenTarget, ExecutionFingerprint, ExecutionFingerprintDigest,
    PreflightFailureCode, PreflightReport, PreflightRequest, PublicationPreflightRequest,
    ResourceEstimate, ResourcePolicy, ResourceProfile, ResultStatus, RouteEvidence,
    ScalarCapability, SolverCapability, SolverConfig, SolverProvenance, TerminationReason,
};
use xc_operator::{GeneralizedEigenProblem, OperatorError, OperatorMetadata, SymmetricOperator};

#[cfg(feature = "hp-reference")]
mod hp_generalized;
#[cfg(feature = "hp-reference")]
pub use hp_generalized::*;
#[cfg(feature = "hp-reference")]
mod hp_generalized_dense;
#[cfg(feature = "hp-reference")]
pub use hp_generalized_dense::*;
#[cfg(feature = "hp-reference")]
mod hp_generalized_block;
#[cfg(feature = "hp-reference")]
pub use hp_generalized_block::*;
#[cfg(feature = "hp-reference")]
mod hp_shift_invert;
#[cfg(feature = "hp-reference")]
pub use hp_shift_invert::*;
#[cfg(feature = "hp-reference")]
mod hp_thick_restart;
#[cfg(feature = "hp-reference")]
pub use hp_thick_restart::*;
#[cfg(feature = "hp-reference")]
mod hp_shift_invert_krylov;
#[cfg(feature = "hp-reference")]
pub use hp_shift_invert_krylov::*;

#[derive(Clone, Debug)]
pub enum SolverError {
    InvalidConfiguration(String),
    UnsupportedTarget(String),
    Operator(OperatorError),
    NumericalBreakdown(String),
    NonConvergence(String),
    CrossCheckDisagreement(String),
    Cancelled(String),
}

impl Display for SolverError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid solver configuration: {message}")
            }
            Self::UnsupportedTarget(message) => write!(f, "unsupported target: {message}"),
            Self::Operator(error) => Display::fmt(error, f),
            Self::NumericalBreakdown(message) => write!(f, "numerical breakdown: {message}"),
            Self::NonConvergence(message) => write!(f, "solver did not converge: {message}"),
            Self::CrossCheckDisagreement(message) => {
                write!(f, "independent solvers disagree: {message}")
            }
            Self::Cancelled(message) => write!(f, "solver operation cancelled: {message}"),
        }
    }
}

impl Error for SolverError {}

impl From<OperatorError> for SolverError {
    fn from(value: OperatorError) -> Self {
        Self::Operator(value)
    }
}

#[cfg(feature = "hp-reference")]
#[inline]
fn reprecision_hp_value(value: &mut rug::Float, precision_bits: u32) {
    if value.prec() != precision_bits {
        *value = rug::Float::with_val(precision_bits, &*value);
    }
}

/// Route-neutral performance counters. Timing is observational only and must
/// never participate in a numerical acceptance decision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverPerformanceTelemetry {
    pub operator_applications: u64,
    pub metric_applications: u64,
    pub preconditioner_applications: u64,
    pub factorizations: u64,
    pub iterations: u64,
    pub precision_escalations: u64,
    pub estimated_peak_memory_bytes: u64,
    pub elapsed_nanoseconds: u128,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EigenpairReportF64 {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub scaled_backward_error: f64,
    pub iterations: usize,
    pub operator_applications: usize,
    pub algorithm: String,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

impl EigenpairReportF64 {
    pub fn validate_finite(&self) -> Result<(), SolverError> {
        if !self.eigenvalue.is_finite()
            || !self.residual_norm.is_finite()
            || !self.relative_residual.is_finite()
            || !self.scaled_backward_error.is_finite()
            || self.eigenvector.iter().any(|x| !x.is_finite())
        {
            return Err(SolverError::NumericalBreakdown(
                "report contains non-finite values".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossCheckedEigenpairF64 {
    pub accepted: EigenpairReportF64,
    pub independent: EigenpairReportF64,
    pub eigenvalue_difference: f64,
    pub vector_overlap_squared: f64,
    pub tolerance: f64,
}

pub struct SymmetricProblemF64<'a> {
    pub operator: &'a dyn SymmetricOperator<f64>,
}

impl<'a> SymmetricProblemF64<'a> {
    pub fn new(operator: &'a dyn SymmetricOperator<f64>) -> Self {
        Self { operator }
    }
}

pub trait EigenSolverF64: Send + Sync {
    fn name(&self) -> &'static str;

    fn solve(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
    ) -> Result<EigenpairReportF64, SolverError>;

    /// Execute with a cooperative cancellation token. Implementations should
    /// poll it at bounded, safe operation boundaries.
    fn solve_controlled(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
        cancellation: &CancellationToken,
    ) -> Result<EigenpairReportF64, SolverError> {
        check_solver_cancellation(cancellation)?;
        self.solve(problem, config)
    }
}

fn check_solver_cancellation(cancellation: &CancellationToken) -> Result<(), SolverError> {
    cancellation
        .check()
        .map_err(|error| SolverError::Cancelled(error.to_string()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn normalize(x: &mut [f64]) -> Result<(), SolverError> {
    let n = norm(x);
    if !n.is_finite() || n <= f64::MIN_POSITIVE {
        return Err(SolverError::NumericalBreakdown(
            "cannot normalize a zero or non-finite vector".to_owned(),
        ));
    }
    for xi in x {
        *xi /= n;
    }
    Ok(())
}

fn deterministic_seed(n: usize) -> Vec<f64> {
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            let k = (i + 1) as f64;
            1.0 / k + ((i % 7) as f64 - 3.0) * 1e-4
        })
        .collect();
    // n is checked by callers.
    let _ = normalize(&mut x);
    x
}

fn evaluate_eigenpair(
    operator: &dyn SymmetricOperator<f64>,
    vector: &[f64],
    workspace: &mut [f64],
) -> Result<(f64, f64, f64, f64), SolverError> {
    operator.apply(vector, workspace)?;
    let eigenvalue = dot(vector, workspace) / dot(vector, vector);
    let vector_norm = norm(vector);
    let mut residual_sq = 0.0;
    let mut applied_sq = 0.0;
    for (av, v) in workspace.iter().zip(vector) {
        let r = av - eigenvalue * v;
        residual_sq += r * r;
        applied_sq += av * av;
    }
    let residual = residual_sq.sqrt();
    let applied_norm = applied_sq.sqrt();
    let relative =
        residual / (applied_norm + eigenvalue.abs() * vector_norm).max(f64::MIN_POSITIVE);
    let norm_bound = operator
        .norm_bound()
        .unwrap_or(applied_norm / vector_norm.max(f64::MIN_POSITIVE));
    let backward = residual
        / (norm_bound * vector_norm + eigenvalue.abs() * vector_norm).max(f64::MIN_POSITIVE);
    Ok((eigenvalue, residual, relative, backward))
}

fn supported_extreme(target: &EigenTarget) -> Result<bool, SolverError> {
    match target {
        EigenTarget::AlgebraicLargest => Ok(true),
        EigenTarget::AlgebraicSmallest => Ok(false),
        other => Err(SolverError::UnsupportedTarget(format!(
            "{} supports only algebraic smallest/largest, got {other:?}",
            "extreme solver"
        ))),
    }
}

fn stopping_thresholds_f64(config: &SolverConfig) -> Result<(f64, f64), SolverError> {
    let absolute = config
        .stopping
        .absolute_residual
        .parse_f64()
        .map_err(|error| {
            SolverError::InvalidConfiguration(format!(
                "absolute_residual is not representable by the f64 reference solver: {error}"
            ))
        })?;
    let backward = config
        .stopping
        .scaled_backward_error
        .parse_f64()
        .map_err(|error| {
            SolverError::InvalidConfiguration(format!(
                "scaled_backward_error is not representable by the f64 reference solver: {error}"
            ))
        })?;
    Ok((absolute, backward))
}

/// Deterministic shifted power iteration.  The transformation `b I ± A`
/// makes the desired algebraic extreme the dominant nonnegative eigenvalue
/// when `b` is a valid norm bound.
#[derive(Clone, Debug, Default)]
pub struct ShiftedPowerSolverF64;

impl EigenSolverF64 for ShiftedPowerSolverF64 {
    fn name(&self) -> &'static str {
        "shifted_power_f64"
    }

    fn solve(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
    ) -> Result<EigenpairReportF64, SolverError> {
        self.solve_controlled(problem, config, &CancellationToken::new())
    }

    fn solve_controlled(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
        cancellation: &CancellationToken,
    ) -> Result<EigenpairReportF64, SolverError> {
        check_solver_cancellation(cancellation)?;
        config
            .validate()
            .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
        let largest = supported_extreme(&config.target)?;
        let (absolute_residual, backward_tolerance) = stopping_thresholds_f64(config)?;
        let n = problem.operator.dimension();
        if n == 0 {
            return Err(SolverError::InvalidConfiguration(
                "operator dimension must be positive".to_owned(),
            ));
        }
        let bound = problem.operator.norm_bound().ok_or_else(|| {
            SolverError::InvalidConfiguration(
                "shifted power iteration requires a valid operator norm bound".to_owned(),
            )
        })?;
        if !bound.is_finite() || bound < 0.0 {
            return Err(SolverError::InvalidConfiguration(
                "operator norm bound must be finite and nonnegative".to_owned(),
            ));
        }

        let mut x = deterministic_seed(n);
        let mut ax = vec![0.0; n];
        let mut transformed = vec![0.0; n];
        let sign = if largest { 1.0 } else { -1.0 };
        let shift = bound + f64::EPSILON * bound.max(1.0);
        let mut applications = 0usize;

        for iteration in 1..=config.stopping.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            problem.operator.apply(&x, &mut ax)?;
            applications += 1;
            for i in 0..n {
                transformed[i] = shift * x[i] + sign * ax[i];
            }
            normalize(&mut transformed)?;
            std::mem::swap(&mut x, &mut transformed);

            let (lambda, residual, relative, backward) =
                evaluate_eigenpair(problem.operator, &x, &mut ax)?;
            applications += 1;

            if iteration >= config.stopping.minimum_iterations
                && (residual <= absolute_residual || backward <= backward_tolerance)
            {
                let report = EigenpairReportF64 {
                    eigenvalue: lambda,
                    eigenvector: x,
                    residual_norm: residual,
                    relative_residual: relative,
                    scaled_backward_error: backward,
                    iterations: iteration,
                    operator_applications: applications,
                    algorithm: self.name().to_owned(),
                    status: ResultStatus::Converged,
                    termination: if backward <= backward_tolerance {
                        TerminationReason::BackwardErrorTolerance
                    } else {
                        TerminationReason::ResidualTolerance
                    },
                    assurance: AssuranceLevel::Computed,
                    provenance: SolverProvenance::current_package("f64"),
                };
                report.validate_finite()?;
                return Ok(report);
            }
        }

        Err(SolverError::NonConvergence(format!(
            "{} exceeded {} iterations",
            self.name(),
            config.stopping.maximum_iterations
        )))
    }
}

/// Full-reorthogonalized deterministic Lanczos reference solver.
#[derive(Clone, Debug)]
pub struct LanczosSolverF64 {
    pub reorthogonalization_passes: usize,
}

impl Default for LanczosSolverF64 {
    fn default() -> Self {
        Self {
            reorthogonalization_passes: 2,
        }
    }
}

impl LanczosSolverF64 {
    fn ritz_pair(
        &self,
        alphas: &[f64],
        betas: &[f64],
        basis: &[Vec<f64>],
        largest: bool,
    ) -> Result<(f64, Vec<f64>), SolverError> {
        let m = alphas.len();
        if m == 0 || basis.len() < m || betas.len() + 1 < m {
            return Err(SolverError::NumericalBreakdown(
                "inconsistent Lanczos basis".to_owned(),
            ));
        }
        let mut t = DMatrix::<f64>::zeros(m, m);
        for i in 0..m {
            t[(i, i)] = alphas[i];
            if i + 1 < m {
                t[(i, i + 1)] = betas[i];
                t[(i + 1, i)] = betas[i];
            }
        }
        let decomposition = SymmetricEigen::new(t);
        let index = if largest {
            (0..m)
                .max_by(|&a, &b| {
                    decomposition.eigenvalues[a].total_cmp(&decomposition.eigenvalues[b])
                })
                .unwrap()
        } else {
            (0..m)
                .min_by(|&a, &b| {
                    decomposition.eigenvalues[a].total_cmp(&decomposition.eigenvalues[b])
                })
                .unwrap()
        };
        let theta = decomposition.eigenvalues[index];
        let n = basis[0].len();
        if basis
            .iter()
            .take(m)
            .any(|basis_vector| basis_vector.len() != n)
        {
            return Err(SolverError::NumericalBreakdown(
                "Lanczos basis vectors have inconsistent dimensions".to_owned(),
            ));
        }
        let mut vector = vec![0.0; n];
        for (k, basis_vector) in basis.iter().take(m).enumerate() {
            let coefficient = decomposition.eigenvectors[(k, index)];
            for (output, component) in vector.iter_mut().zip(basis_vector) {
                *output += coefficient * component;
            }
        }
        normalize(&mut vector)?;
        Ok((theta, vector))
    }
}

impl EigenSolverF64 for LanczosSolverF64 {
    fn name(&self) -> &'static str {
        "lanczos_full_reorthogonalization_f64"
    }

    fn solve(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
    ) -> Result<EigenpairReportF64, SolverError> {
        self.solve_controlled(problem, config, &CancellationToken::new())
    }

    fn solve_controlled(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
        cancellation: &CancellationToken,
    ) -> Result<EigenpairReportF64, SolverError> {
        check_solver_cancellation(cancellation)?;
        config
            .validate()
            .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
        let largest = supported_extreme(&config.target)?;
        let (absolute_residual, backward_tolerance) = stopping_thresholds_f64(config)?;
        let n = problem.operator.dimension();
        if n == 0 {
            return Err(SolverError::InvalidConfiguration(
                "operator dimension must be positive".to_owned(),
            ));
        }
        let max_basis = config.stopping.maximum_iterations.min(n);
        let mut q = deterministic_seed(n);
        let mut q_prev = vec![0.0; n];
        let mut beta_prev = 0.0;
        let mut basis = vec![q.clone()];
        let mut alphas = Vec::with_capacity(max_basis);
        let mut betas = Vec::with_capacity(max_basis.saturating_sub(1));
        let mut z = vec![0.0; n];
        let mut workspace = vec![0.0; n];
        let mut applications = 0usize;

        for iteration in 1..=max_basis {
            check_solver_cancellation(cancellation)?;
            problem.operator.apply(&q, &mut z)?;
            applications += 1;
            if iteration > 1 {
                for i in 0..n {
                    z[i] -= beta_prev * q_prev[i];
                }
            }
            let alpha = dot(&q, &z);
            for i in 0..n {
                z[i] -= alpha * q[i];
            }

            // Full modified Gram-Schmidt, repeated for difficult clustered spectra.
            for _ in 0..self.reorthogonalization_passes.max(1) {
                check_solver_cancellation(cancellation)?;
                for vector in &basis {
                    let projection = dot(vector, &z);
                    for i in 0..n {
                        z[i] -= projection * vector[i];
                    }
                }
            }
            let beta = norm(&z);
            alphas.push(alpha);

            let (_ritz_theta, vector) = self.ritz_pair(&alphas, &betas, &basis, largest)?;
            let (lambda, residual, relative, backward) =
                evaluate_eigenpair(problem.operator, &vector, &mut workspace)?;
            applications += 1;

            if iteration >= config.stopping.minimum_iterations
                && (residual <= absolute_residual || backward <= backward_tolerance)
            {
                let report = EigenpairReportF64 {
                    eigenvalue: lambda,
                    eigenvector: vector,
                    residual_norm: residual,
                    relative_residual: relative,
                    scaled_backward_error: backward,
                    iterations: iteration,
                    operator_applications: applications,
                    algorithm: self.name().to_owned(),
                    status: ResultStatus::Converged,
                    termination: if backward <= backward_tolerance {
                        TerminationReason::BackwardErrorTolerance
                    } else {
                        TerminationReason::ResidualTolerance
                    },
                    assurance: AssuranceLevel::Computed,
                    provenance: SolverProvenance::current_package("f64"),
                };
                report.validate_finite()?;
                return Ok(report);
            }

            let breakdown_threshold =
                f64::EPSILON.sqrt() * problem.operator.norm_bound().unwrap_or(1.0).max(1.0);
            if !beta.is_finite() || beta <= breakdown_threshold {
                let report = EigenpairReportF64 {
                    eigenvalue: lambda,
                    eigenvector: vector,
                    residual_norm: residual,
                    relative_residual: relative,
                    scaled_backward_error: backward,
                    iterations: iteration,
                    operator_applications: applications,
                    algorithm: self.name().to_owned(),
                    status: if residual <= absolute_residual {
                        ResultStatus::Converged
                    } else {
                        ResultStatus::Approximate
                    },
                    termination: TerminationReason::Breakdown,
                    assurance: AssuranceLevel::Computed,
                    provenance: SolverProvenance::current_package("f64"),
                };
                report.validate_finite()?;
                return Ok(report);
            }

            if iteration < max_basis {
                betas.push(beta);
                q_prev.clone_from(&q);
                for i in 0..n {
                    q[i] = z[i] / beta;
                    z[i] = 0.0;
                }
                beta_prev = beta;
                basis.push(q.clone());
            }
        }

        Err(SolverError::NonConvergence(format!(
            "{} exhausted a basis of dimension {max_basis}",
            self.name()
        )))
    }
}

/// Configuration for deterministic block subspace iteration on one algebraic
/// end of a real symmetric spectrum.
///
/// `block_size` includes guard Ritz values used to decide whether a cluster
/// crosses the requested boundary. It must therefore exceed
/// `requested_count` unless the full operator dimension is requested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockExtremeConfigF64 {
    pub target: EigenTarget,
    pub requested_count: usize,
    pub block_size: usize,
    pub absolute_residual_tolerance: f64,
    pub scaled_backward_error_tolerance: f64,
    pub ritz_value_stability_tolerance: f64,
    pub cluster_absolute_tolerance: f64,
    pub cluster_relative_tolerance: f64,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
}

impl BlockExtremeConfigF64 {
    pub fn validate(&self, dimension: usize) -> Result<(), SolverError> {
        if dimension == 0 {
            return Err(SolverError::InvalidConfiguration(
                "operator dimension must be positive".to_owned(),
            ));
        }
        if !matches!(
            self.target,
            EigenTarget::AlgebraicLargest | EigenTarget::AlgebraicSmallest
        ) {
            return Err(SolverError::UnsupportedTarget(format!(
                "block subspace iteration requires an algebraic extreme, got {:?}",
                self.target
            )));
        }
        if self.requested_count == 0 || self.requested_count > dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "requested_count must be in 1..={dimension}"
            )));
        }
        if self.block_size < self.requested_count || self.block_size > dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "block_size must be in {}..={dimension}",
                self.requested_count
            )));
        }
        if self.requested_count < dimension && self.block_size == self.requested_count {
            return Err(SolverError::InvalidConfiguration(
                "block_size must reserve at least one guard Ritz value when the requested range does not cover the full spectrum"
                    .to_owned(),
            ));
        }
        for (name, value, strictly_positive) in [
            (
                "absolute_residual_tolerance",
                self.absolute_residual_tolerance,
                true,
            ),
            (
                "scaled_backward_error_tolerance",
                self.scaled_backward_error_tolerance,
                true,
            ),
            (
                "ritz_value_stability_tolerance",
                self.ritz_value_stability_tolerance,
                true,
            ),
            (
                "cluster_absolute_tolerance",
                self.cluster_absolute_tolerance,
                false,
            ),
            (
                "cluster_relative_tolerance",
                self.cluster_relative_tolerance,
                false,
            ),
        ] {
            if !value.is_finite() || value < 0.0 || (strictly_positive && value == 0.0) {
                return Err(SolverError::InvalidConfiguration(format!(
                    "{name} must be finite and {}",
                    if strictly_positive {
                        "strictly positive"
                    } else {
                        "nonnegative"
                    }
                )));
            }
        }
        if self.maximum_iterations == 0 || self.minimum_iterations > self.maximum_iterations {
            return Err(SolverError::InvalidConfiguration(
                "maximum_iterations must be positive and not less than minimum_iterations"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// A residual-based spectral window around one converged Ritz cluster.
///
/// This f64 discovery result is deliberately not labeled rigorous. Certified
/// enclosures require the interval/inertia routes described by TD-02.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualSpectralWindowF64 {
    pub lower: f64,
    pub upper: f64,
    pub rigorous: bool,
}

/// Orthonormal basis and projected operator for one selected invariant
/// subspace. When `individual_vectors_resolved` is false, callers must treat
/// `basis` as a subspace rather than attach meaning to its individual vectors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvariantSubspaceF64 {
    pub dimension: usize,
    pub ritz_values: Vec<f64>,
    pub basis: Vec<Vec<f64>>,
    pub projected_operator_row_major: Vec<f64>,
    pub maximum_residual_norm: f64,
    pub residual_frobenius_norm: f64,
    pub spectral_window: ResidualSpectralWindowF64,
    pub individual_vectors_resolved: bool,
}

/// Result of a block selected-extreme solve. The returned count can exceed the
/// request when the requested boundary falls inside a detected cluster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockEigenReportF64 {
    pub target: EigenTarget,
    pub requested_count: usize,
    pub returned_count: usize,
    pub block_size: usize,
    pub invariant_subspaces: Vec<InvariantSubspaceF64>,
    pub target_boundary_separation_established: bool,
    pub maximum_residual_norm: f64,
    pub maximum_scaled_backward_error: f64,
    pub ritz_value_stability: f64,
    pub orthogonality_defect: f64,
    pub iterations: usize,
    pub operator_applications: usize,
    pub estimated_peak_memory_bytes: u64,
    pub algorithm: String,
    pub seed_source: String,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

pub const BLOCK_SUBSPACE_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Complete deterministic continuation state for bounded-memory block
/// subspace iteration.  The retained basis never exceeds the configured
/// block size and includes the current guard Ritz vectors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockSubspaceCheckpointF64 {
    pub schema_version: u32,
    pub algorithm: String,
    pub operator_identity: String,
    pub operator_metadata: OperatorMetadata,
    pub config: BlockExtremeConfigF64,
    pub reorthogonalization_passes: usize,
    pub completed_iterations: usize,
    pub operator_applications: usize,
    pub seed_source: String,
    pub retained_ritz_basis: Vec<Vec<f64>>,
    pub previous_ritz_values: Vec<f64>,
}

impl BlockSubspaceCheckpointF64 {
    pub fn retained_subspace_memory_bytes(&self) -> u64 {
        self.retained_ritz_basis
            .iter()
            .map(|vector| vector.len() as u64)
            .sum::<u64>()
            .saturating_mul(8)
            .saturating_add((self.previous_ritz_values.len() as u64).saturating_mul(8))
    }

    pub fn validate_compatibility(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &BlockExtremeConfigF64,
        operator_identity: &str,
        reorthogonalization_passes: usize,
    ) -> Result<(), SolverError> {
        if self.schema_version != BLOCK_SUBSPACE_CHECKPOINT_SCHEMA_VERSION
            || self.algorithm != "block_subspace_iteration_rayleigh_ritz_f64"
        {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint schema or algorithm is incompatible".to_owned(),
            ));
        }
        if operator_identity.trim().is_empty() || self.operator_identity != operator_identity {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint operator identity is incompatible".to_owned(),
            ));
        }
        if self.operator_metadata != problem.operator.metadata() {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint operator metadata is incompatible".to_owned(),
            ));
        }
        if &self.config != config || self.reorthogonalization_passes != reorthogonalization_passes {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint solver configuration is incompatible".to_owned(),
            ));
        }
        if self.completed_iterations == 0 || self.completed_iterations >= config.maximum_iterations
        {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint iteration is outside the resumable range".to_owned(),
            ));
        }
        if self.seed_source.trim().is_empty() {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint seed provenance is missing".to_owned(),
            ));
        }
        let dimension = problem.operator.dimension();
        if self.retained_ritz_basis.len() != config.block_size
            || self.previous_ritz_values.len() != config.block_size
            || self.retained_ritz_basis.iter().any(|vector| {
                vector.len() != dimension || vector.iter().any(|value| !value.is_finite())
            })
            || self
                .previous_ritz_values
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint retained Ritz state is malformed".to_owned(),
            ));
        }
        let defect = block_orthogonality_defect(&self.retained_ritz_basis);
        if !defect.is_finite() || defect > 4096.0 * f64::EPSILON * dimension.max(1) as f64 {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint retained Ritz basis is not orthonormal".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointDirective {
    Continue,
    StopAfterCheckpoint,
}

pub trait BlockCheckpointSinkF64 {
    fn save(
        &mut self,
        checkpoint: &BlockSubspaceCheckpointF64,
    ) -> Result<CheckpointDirective, SolverError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum BlockSolveOutcomeF64 {
    Complete {
        report: Box<BlockEigenReportF64>,
    },
    Checkpointed {
        checkpoint: Box<BlockSubspaceCheckpointF64>,
    },
}

#[derive(Clone, Debug)]
struct BlockRitzStateF64 {
    values: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    applied_vectors: Vec<Vec<f64>>,
    residual_norms: Vec<f64>,
    scaled_backward_errors: Vec<f64>,
}

fn canonicalize_vector_sign(vector: &mut [f64]) -> bool {
    if let Some(value) = vector
        .iter()
        .find(|value| value.abs() > 32.0 * f64::EPSILON)
    {
        if *value < 0.0 {
            for component in vector {
                *component = -*component;
            }
            return true;
        }
    }
    false
}

fn deterministic_block_seed(dimension: usize, count: usize) -> Vec<Vec<f64>> {
    fn splitmix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    (0..count)
        .map(|column| {
            (0..dimension)
                .map(|row| {
                    let key = ((column as u64) << 32) ^ row as u64 ^ 0x5843_454c_4552_4154;
                    let bits = splitmix64(key) >> 11;
                    (bits as f64) * (1.0 / ((1u64 << 53) as f64)) - 0.5
                })
                .collect()
        })
        .collect()
}

fn orthonormalize_block(
    candidates: Vec<Vec<f64>>,
    dimension: usize,
    count: usize,
    passes: usize,
) -> Result<Vec<Vec<f64>>, SolverError> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(count);
    for mut candidate in candidates {
        if candidate.len() != dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "initial subspace vector has dimension {}, expected {dimension}",
                candidate.len()
            )));
        }
        for _ in 0..passes.max(1) {
            for vector in &basis {
                let projection = dot(vector, &candidate);
                for (value, component) in candidate.iter_mut().zip(vector) {
                    *value -= projection * component;
                }
            }
        }
        let candidate_norm = norm(&candidate);
        if candidate_norm.is_finite() && candidate_norm > 128.0 * f64::EPSILON {
            for value in &mut candidate {
                *value /= candidate_norm;
            }
            let _ = canonicalize_vector_sign(&mut candidate);
            basis.push(candidate);
            if basis.len() == count {
                return Ok(basis);
            }
        }
    }
    Err(SolverError::NumericalBreakdown(format!(
        "block lost rank: produced {} orthonormal vectors, expected {count}",
        basis.len()
    )))
}

fn block_orthogonality_defect(basis: &[Vec<f64>]) -> f64 {
    let mut defect: f64 = 0.0;
    for (i, left) in basis.iter().enumerate() {
        for (j, right) in basis.iter().enumerate() {
            let expected = f64::from(i == j);
            defect = defect.max((dot(left, right) - expected).abs());
        }
    }
    defect
}

fn same_cluster(left: f64, right: f64, config: &BlockExtremeConfigF64) -> bool {
    (left - right).abs()
        <= config.cluster_absolute_tolerance
            + config.cluster_relative_tolerance * left.abs().max(right.abs())
}

/// Deterministic block subspace iteration with repeated orthogonalization and
/// Rayleigh-Ritz extraction. It uses only operator applications and a valid
/// operator norm bound; dense materialization is not required.
#[derive(Clone, Debug)]
pub struct BlockSubspaceIterationF64 {
    pub reorthogonalization_passes: usize,
}

impl Default for BlockSubspaceIterationF64 {
    fn default() -> Self {
        Self {
            reorthogonalization_passes: 2,
        }
    }
}

impl BlockSubspaceIterationF64 {
    pub fn solve(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &BlockExtremeConfigF64,
    ) -> Result<BlockEigenReportF64, SolverError> {
        self.solve_controlled(problem, config, None, &CancellationToken::new())
    }

    pub fn solve_with_initial_subspace(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &BlockExtremeConfigF64,
        initial_subspace: &[Vec<f64>],
    ) -> Result<BlockEigenReportF64, SolverError> {
        self.solve_controlled(
            problem,
            config,
            Some(initial_subspace),
            &CancellationToken::new(),
        )
    }

    pub fn solve_controlled(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &BlockExtremeConfigF64,
        initial_subspace: Option<&[Vec<f64>]>,
        cancellation: &CancellationToken,
    ) -> Result<BlockEigenReportF64, SolverError> {
        match self.solve_engine(
            problem,
            config,
            initial_subspace,
            None,
            None,
            None,
            cancellation,
        )? {
            BlockSolveOutcomeF64::Complete { report } => Ok(*report),
            BlockSolveOutcomeF64::Checkpointed { .. } => Err(SolverError::NumericalBreakdown(
                "non-checkpointed block solve stopped at a checkpoint".to_owned(),
            )),
        }
    }

    /// Run or resume block iteration with durable checkpoint callbacks.  A
    /// sink may request a clean stop after any completed iteration; resuming
    /// the returned state continues with the same Ritz block and stability
    /// history as an uninterrupted solve.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_checkpointed(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &BlockExtremeConfigF64,
        operator_identity: &str,
        initial_subspace: Option<&[Vec<f64>]>,
        resume: Option<&BlockSubspaceCheckpointF64>,
        checkpoint_sink: &mut dyn BlockCheckpointSinkF64,
        cancellation: &CancellationToken,
    ) -> Result<BlockSolveOutcomeF64, SolverError> {
        if operator_identity.trim().is_empty() {
            return Err(SolverError::InvalidConfiguration(
                "checkpointed solve requires a nonempty operator identity".to_owned(),
            ));
        }
        self.solve_engine(
            problem,
            config,
            initial_subspace,
            resume,
            Some(operator_identity),
            Some(checkpoint_sink),
            cancellation,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_engine(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &BlockExtremeConfigF64,
        initial_subspace: Option<&[Vec<f64>]>,
        resume: Option<&BlockSubspaceCheckpointF64>,
        operator_identity: Option<&str>,
        mut checkpoint_sink: Option<&mut dyn BlockCheckpointSinkF64>,
        cancellation: &CancellationToken,
    ) -> Result<BlockSolveOutcomeF64, SolverError> {
        check_solver_cancellation(cancellation)?;
        let dimension = problem.operator.dimension();
        config.validate(dimension)?;
        let norm_bound = problem.operator.norm_bound().ok_or_else(|| {
            SolverError::InvalidConfiguration(
                "block algebraic-extreme iteration requires a valid operator 2-norm bound"
                    .to_owned(),
            )
        })?;
        if !norm_bound.is_finite() || norm_bound < 0.0 {
            return Err(SolverError::InvalidConfiguration(
                "operator norm bound must be finite and nonnegative".to_owned(),
            ));
        }
        let shift = norm_bound + norm_bound.max(1.0) * f64::EPSILON.sqrt();
        if !shift.is_finite() {
            return Err(SolverError::InvalidConfiguration(
                "operator norm bound is too large to construct a finite shifted iteration"
                    .to_owned(),
            ));
        }
        let transform_sign = if config.target == EigenTarget::AlgebraicLargest {
            1.0
        } else {
            -1.0
        };

        if resume.is_some() && initial_subspace.is_some() {
            return Err(SolverError::InvalidConfiguration(
                "a resumed block solve cannot also supply a new initial subspace".to_owned(),
            ));
        }
        let (mut basis, mut previous_values, mut applications, completed_iterations, seed_source) =
            if let Some(checkpoint) = resume {
                let identity = operator_identity.ok_or_else(|| {
                    SolverError::InvalidConfiguration(
                        "resuming a block checkpoint requires an operator identity".to_owned(),
                    )
                })?;
                checkpoint.validate_compatibility(
                    problem,
                    config,
                    identity,
                    self.reorthogonalization_passes,
                )?;
                (
                    checkpoint.retained_ritz_basis.clone(),
                    Some(checkpoint.previous_ritz_values.clone()),
                    checkpoint.operator_applications,
                    checkpoint.completed_iterations,
                    checkpoint.seed_source.clone(),
                )
            } else {
                let mut seed = Vec::new();
                if let Some(initial) = initial_subspace {
                    if initial.len() > config.block_size {
                        return Err(SolverError::InvalidConfiguration(format!(
                            "initial subspace has {} vectors, exceeding block_size {}",
                            initial.len(),
                            config.block_size
                        )));
                    }
                    seed.extend_from_slice(initial);
                }
                let seed_source = if seed.is_empty() {
                    "deterministic_splitmix64_block".to_owned()
                } else if seed.len() == config.block_size {
                    "caller_warm_start".to_owned()
                } else {
                    "caller_warm_start_plus_deterministic_completion".to_owned()
                };
                if seed.len() < config.block_size {
                    // Supply a complete deterministic block after the warm vectors so
                    // collinearity cannot leave the initial subspace one column short.
                    seed.extend(deterministic_block_seed(dimension, config.block_size));
                }
                (
                    orthonormalize_block(
                        seed,
                        dimension,
                        config.block_size,
                        self.reorthogonalization_passes,
                    )?,
                    None,
                    0,
                    0,
                    seed_source,
                )
            };
        if completed_iterations >= config.maximum_iterations {
            return Err(SolverError::InvalidConfiguration(
                "block checkpoint has no remaining iteration budget".to_owned(),
            ));
        }
        if checkpoint_sink.is_some() && operator_identity.is_none() {
            return Err(SolverError::InvalidConfiguration(
                "checkpoint sink requires an operator identity".to_owned(),
            ));
        }

        let mut final_state = None;
        let mut final_stability = f64::INFINITY;
        let mut converged = false;
        let mut iterations = completed_iterations;

        for iteration in completed_iterations + 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            iterations = iteration;
            let mut transformed = Vec::with_capacity(config.block_size);
            for vector in &basis {
                check_solver_cancellation(cancellation)?;
                let mut applied = vec![0.0; dimension];
                problem.operator.apply(vector, &mut applied)?;
                applications += 1;
                for (value, source) in applied.iter_mut().zip(vector) {
                    *value = shift * source + transform_sign * *value;
                }
                transformed.push(applied);
            }
            basis = orthonormalize_block(
                transformed,
                dimension,
                config.block_size,
                self.reorthogonalization_passes,
            )?;
            let state =
                self.extract_ritz(problem.operator, &basis, &config.target, cancellation)?;
            applications += config.block_size;
            final_stability = previous_values
                .as_ref()
                .map(|previous| {
                    state
                        .values
                        .iter()
                        .zip(previous)
                        .map(|(current, prior)| {
                            (current - prior).abs() / current.abs().max(prior.abs()).max(1.0)
                        })
                        .fold(0.0, f64::max)
                })
                .unwrap_or(f64::INFINITY);
            let residual_converged = state
                .residual_norms
                .iter()
                .zip(&state.scaled_backward_errors)
                .all(|(residual, backward)| {
                    *residual <= config.absolute_residual_tolerance
                        || *backward <= config.scaled_backward_error_tolerance
                });
            converged = iteration >= config.minimum_iterations
                && residual_converged
                && final_stability <= config.ritz_value_stability_tolerance;
            previous_values = Some(state.values.clone());
            basis.clone_from(&state.vectors);
            final_state = Some(state);
            if converged {
                break;
            }
            if iteration < config.maximum_iterations {
                if let Some(sink) = checkpoint_sink.as_deref_mut() {
                    let checkpoint = BlockSubspaceCheckpointF64 {
                        schema_version: BLOCK_SUBSPACE_CHECKPOINT_SCHEMA_VERSION,
                        algorithm: "block_subspace_iteration_rayleigh_ritz_f64".to_owned(),
                        operator_identity: operator_identity
                            .expect("checkpoint identity was validated")
                            .to_owned(),
                        operator_metadata: problem.operator.metadata(),
                        config: config.clone(),
                        reorthogonalization_passes: self.reorthogonalization_passes,
                        completed_iterations: iteration,
                        operator_applications: applications,
                        seed_source: seed_source.clone(),
                        retained_ritz_basis: basis.clone(),
                        previous_ritz_values: previous_values
                            .clone()
                            .expect("current Ritz values were just stored"),
                    };
                    if sink.save(&checkpoint)? == CheckpointDirective::StopAfterCheckpoint {
                        return Ok(BlockSolveOutcomeF64::Checkpointed {
                            checkpoint: Box::new(checkpoint),
                        });
                    }
                }
            }
        }

        let state = final_state.ok_or_else(|| {
            SolverError::NumericalBreakdown("block iteration produced no Ritz state".to_owned())
        })?;
        let report = self.build_report(
            config,
            state,
            final_stability,
            iterations,
            applications,
            dimension,
            seed_source,
            converged,
        )?;
        Ok(BlockSolveOutcomeF64::Complete {
            report: Box::new(report),
        })
    }

    fn extract_ritz(
        &self,
        operator: &dyn SymmetricOperator<f64>,
        basis: &[Vec<f64>],
        target: &EigenTarget,
        cancellation: &CancellationToken,
    ) -> Result<BlockRitzStateF64, SolverError> {
        let count = basis.len();
        let dimension = operator.dimension();
        let mut applied_basis = Vec::with_capacity(count);
        for vector in basis {
            check_solver_cancellation(cancellation)?;
            let mut applied = vec![0.0; dimension];
            operator.apply(vector, &mut applied)?;
            applied_basis.push(applied);
        }
        let mut projected = DMatrix::<f64>::zeros(count, count);
        for row in 0..count {
            for column in 0..=row {
                let left = dot(&basis[row], &applied_basis[column]);
                let right = dot(&basis[column], &applied_basis[row]);
                let value = 0.5 * (left + right);
                projected[(row, column)] = value;
                projected[(column, row)] = value;
            }
        }
        let decomposition = SymmetricEigen::new(projected);
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by(|left, right| {
            let order =
                decomposition.eigenvalues[*left].total_cmp(&decomposition.eigenvalues[*right]);
            if *target == EigenTarget::AlgebraicLargest {
                order.reverse()
            } else {
                order
            }
        });
        let norm_bound = operator.norm_bound().unwrap_or(0.0);
        let mut values = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count);
        let mut applied_vectors = Vec::with_capacity(count);
        let mut residual_norms = Vec::with_capacity(count);
        let mut scaled_backward_errors = Vec::with_capacity(count);
        for index in indices {
            let value = decomposition.eigenvalues[index];
            let mut vector = vec![0.0; dimension];
            let mut applied = vec![0.0; dimension];
            for column in 0..count {
                let coefficient = decomposition.eigenvectors[(column, index)];
                for row in 0..dimension {
                    vector[row] += coefficient * basis[column][row];
                    applied[row] += coefficient * applied_basis[column][row];
                }
            }
            let vector_norm = norm(&vector);
            if !vector_norm.is_finite() || vector_norm <= f64::MIN_POSITIVE {
                return Err(SolverError::NumericalBreakdown(
                    "Rayleigh-Ritz extraction produced a zero or non-finite vector".to_owned(),
                ));
            }
            for (component, applied_component) in vector.iter_mut().zip(&mut applied) {
                *component /= vector_norm;
                *applied_component /= vector_norm;
            }
            if canonicalize_vector_sign(&mut vector) {
                for component in &mut applied {
                    *component = -*component;
                }
            }
            let residual = applied
                .iter()
                .zip(&vector)
                .map(|(av, x)| {
                    let value = av - value * x;
                    value * value
                })
                .sum::<f64>()
                .sqrt();
            let backward = residual / (norm_bound + value.abs()).max(f64::MIN_POSITIVE);
            values.push(value);
            vectors.push(vector);
            applied_vectors.push(applied);
            residual_norms.push(residual);
            scaled_backward_errors.push(backward);
        }
        Ok(BlockRitzStateF64 {
            values,
            vectors,
            applied_vectors,
            residual_norms,
            scaled_backward_errors,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_report(
        &self,
        config: &BlockExtremeConfigF64,
        state: BlockRitzStateF64,
        stability: f64,
        iterations: usize,
        applications: usize,
        operator_dimension: usize,
        seed_source: String,
        converged: bool,
    ) -> Result<BlockEigenReportF64, SolverError> {
        let mut returned_count = config.requested_count;
        while returned_count < config.block_size
            && same_cluster(
                state.values[returned_count - 1],
                state.values[returned_count],
                config,
            )
        {
            returned_count += 1;
        }
        let boundary_separation_established = returned_count == operator_dimension
            || (returned_count < config.block_size
                && !same_cluster(
                    state.values[returned_count - 1],
                    state.values[returned_count],
                    config,
                ));

        let mut invariant_subspaces = Vec::new();
        let mut first = 0usize;
        while first < returned_count {
            let mut last = first + 1;
            while last < returned_count
                && same_cluster(state.values[last - 1], state.values[last], config)
            {
                last += 1;
            }
            let cluster_dimension = last - first;
            let basis = state.vectors[first..last].to_vec();
            let mut projected = vec![0.0; cluster_dimension * cluster_dimension];
            for row in 0..cluster_dimension {
                for column in 0..=row {
                    let value = 0.5
                        * (dot(&basis[row], &state.applied_vectors[first + column])
                            + dot(&basis[column], &state.applied_vectors[first + row]));
                    projected[row * cluster_dimension + column] = value;
                    projected[column * cluster_dimension + row] = value;
                }
            }
            let residual_frobenius_norm = state.residual_norms[first..last]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let maximum_residual_norm = state.residual_norms[first..last]
                .iter()
                .copied()
                .fold(0.0, f64::max);
            let minimum_value = state.values[first..last]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let maximum_value = state.values[first..last]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            invariant_subspaces.push(InvariantSubspaceF64 {
                dimension: cluster_dimension,
                ritz_values: state.values[first..last].to_vec(),
                basis,
                projected_operator_row_major: projected,
                maximum_residual_norm,
                residual_frobenius_norm,
                spectral_window: ResidualSpectralWindowF64 {
                    lower: minimum_value - residual_frobenius_norm,
                    upper: maximum_value + residual_frobenius_norm,
                    rigorous: false,
                },
                individual_vectors_resolved: cluster_dimension == 1,
            });
            first = last;
        }

        let maximum_residual_norm = state.residual_norms[..returned_count]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let maximum_scaled_backward_error = state.scaled_backward_errors[..returned_count]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let orthogonality_defect = block_orthogonality_defect(&state.vectors[..returned_count]);
        let (status, termination) = if !converged {
            (
                ResultStatus::Approximate,
                TerminationReason::MaximumIterations,
            )
        } else if !boundary_separation_established {
            (
                ResultStatus::UnresolvedCluster,
                TerminationReason::UnresolvedCluster,
            )
        } else if maximum_scaled_backward_error <= config.scaled_backward_error_tolerance {
            (
                ResultStatus::Converged,
                TerminationReason::BackwardErrorTolerance,
            )
        } else {
            (
                ResultStatus::Converged,
                TerminationReason::ResidualTolerance,
            )
        };
        let scalar_vectors = 6u64
            .saturating_mul(config.block_size as u64)
            .saturating_mul(operator_dimension as u64)
            .saturating_add(4u64.saturating_mul(
                (config.block_size as u64).saturating_mul(config.block_size as u64),
            ));
        Ok(BlockEigenReportF64 {
            target: config.target.clone(),
            requested_count: config.requested_count,
            returned_count,
            block_size: config.block_size,
            invariant_subspaces,
            target_boundary_separation_established: boundary_separation_established,
            maximum_residual_norm,
            maximum_scaled_backward_error,
            ritz_value_stability: stability,
            orthogonality_defect,
            iterations,
            operator_applications: applications,
            estimated_peak_memory_bytes: scalar_vectors.saturating_mul(8),
            algorithm: "block_subspace_iteration_rayleigh_ritz_f64".to_owned(),
            seed_source,
            status,
            termination,
            assurance: AssuranceLevel::Computed,
            provenance: SolverProvenance::current_package("f64"),
        })
    }
}

#[cfg(test)]
mod block_subspace_tests {
    use super::*;
    use xc_operator::{DenseSymmetricF64, DiagonalF64, LinearOperator};

    fn block_config(
        target: EigenTarget,
        requested_count: usize,
        block_size: usize,
    ) -> BlockExtremeConfigF64 {
        BlockExtremeConfigF64 {
            target,
            requested_count,
            block_size,
            absolute_residual_tolerance: 1e-10,
            scaled_backward_error_tolerance: 1e-11,
            ritz_value_stability_tolerance: 1e-12,
            cluster_absolute_tolerance: 1e-9,
            cluster_relative_tolerance: 1e-10,
            maximum_iterations: 600,
            minimum_iterations: 2,
        }
    }

    #[test]
    fn block_iteration_returns_several_extremes_without_materializing() {
        let operator = DiagonalF64::new("diagonal", vec![-3.0, -1.0, 2.0, 4.0, 7.0, 11.0]).unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let report = BlockSubspaceIterationF64::default()
            .solve(&problem, &block_config(EigenTarget::AlgebraicLargest, 3, 4))
            .unwrap();

        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.target_boundary_separation_established);
        assert_eq!(report.returned_count, 3);
        let values: Vec<f64> = report
            .invariant_subspaces
            .iter()
            .flat_map(|cluster| cluster.ritz_values.iter().copied())
            .collect();
        assert_eq!(values.len(), 3);
        for (actual, expected) in values.iter().zip([11.0, 7.0, 4.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!(report.maximum_residual_norm < 1e-9);
        assert!(report.orthogonality_defect < 1e-12);
        assert!(report.operator_applications < operator.dimension() * report.iterations * 3);
    }

    #[test]
    fn exact_multiplicity_is_returned_as_one_invariant_subspace() {
        // Orthogonally rotate diag(1, 2, 5, 5), so the cluster fixture does
        // not depend on coordinate-aligned seed vectors.
        let half = 0.5;
        let q = [
            half, half, half, half, half, -half, half, -half, half, half, -half, -half, half,
            -half, -half, half,
        ];
        let diagonal = [1.0, 2.0, 5.0, 5.0];
        let mut matrix = vec![0.0; 16];
        for row in 0..4 {
            for column in 0..4 {
                matrix[row * 4 + column] = (0..4)
                    .map(|axis| q[row * 4 + axis] * diagonal[axis] * q[column * 4 + axis])
                    .sum();
            }
        }
        let operator = DenseSymmetricF64::new("rotated_cluster", 4, matrix, 1e-14).unwrap();
        let report = BlockSubspaceIterationF64::default()
            .solve(
                &SymmetricProblemF64::new(&operator),
                &block_config(EigenTarget::AlgebraicLargest, 1, 3),
            )
            .unwrap();

        assert_eq!(report.status, ResultStatus::Converged);
        assert_eq!(report.returned_count, 2);
        assert_eq!(report.invariant_subspaces.len(), 1);
        let cluster = &report.invariant_subspaces[0];
        assert_eq!(cluster.dimension, 2);
        assert!(!cluster.individual_vectors_resolved);
        assert!(cluster
            .ritz_values
            .iter()
            .all(|value| (*value - 5.0).abs() < 1e-10));
        assert!(cluster.maximum_residual_norm < 1e-9);
        assert!(cluster.spectral_window.lower <= 5.0);
        assert!(cluster.spectral_window.upper >= 5.0);
        assert!(!cluster.spectral_window.rigorous);
    }

    #[test]
    fn cluster_crossing_guard_boundary_is_not_silently_accepted() {
        let operator = DiagonalF64::new("zero", vec![0.0; 5]).unwrap();
        let report = BlockSubspaceIterationF64::default()
            .solve(
                &SymmetricProblemF64::new(&operator),
                &block_config(EigenTarget::AlgebraicSmallest, 2, 3),
            )
            .unwrap();

        assert_eq!(report.status, ResultStatus::UnresolvedCluster);
        assert_eq!(report.termination, TerminationReason::UnresolvedCluster);
        assert_eq!(report.returned_count, 3);
        assert!(!report.target_boundary_separation_established);
        assert_eq!(report.invariant_subspaces[0].dimension, 3);
    }

    #[test]
    fn partial_selection_requires_a_guard_ritz_value() {
        let config = block_config(EigenTarget::AlgebraicLargest, 2, 2);
        let error = config.validate(4).unwrap_err();
        assert!(error.to_string().contains("guard Ritz value"));
    }

    #[test]
    fn collinear_warm_start_is_completed_by_the_deterministic_block() {
        let operator = DiagonalF64::new("diagonal", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let warm_start = deterministic_block_seed(6, 1);
        let report = BlockSubspaceIterationF64::default()
            .solve_with_initial_subspace(
                &SymmetricProblemF64::new(&operator),
                &block_config(EigenTarget::AlgebraicLargest, 2, 3),
                &warm_start,
            )
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert_eq!(
            report.seed_source,
            "caller_warm_start_plus_deterministic_completion"
        );
    }

    struct StopAtIteration {
        iteration: usize,
    }

    impl BlockCheckpointSinkF64 for StopAtIteration {
        fn save(
            &mut self,
            checkpoint: &BlockSubspaceCheckpointF64,
        ) -> Result<CheckpointDirective, SolverError> {
            Ok(if checkpoint.completed_iterations >= self.iteration {
                CheckpointDirective::StopAfterCheckpoint
            } else {
                CheckpointDirective::Continue
            })
        }
    }

    struct ContinueCheckpointing;

    impl BlockCheckpointSinkF64 for ContinueCheckpointing {
        fn save(
            &mut self,
            _checkpoint: &BlockSubspaceCheckpointF64,
        ) -> Result<CheckpointDirective, SolverError> {
            Ok(CheckpointDirective::Continue)
        }
    }

    #[test]
    fn compatible_checkpoint_resume_is_bitwise_identical_to_uninterrupted_solve() {
        let operator = DiagonalF64::new(
            "checkpoint_diagonal",
            vec![-2.0, -0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0],
        )
        .unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let config = block_config(EigenTarget::AlgebraicLargest, 2, 4);
        let solver = BlockSubspaceIterationF64::default();
        let uninterrupted = solver.solve(&problem, &config).unwrap();

        let mut stop = StopAtIteration { iteration: 3 };
        let first = solver
            .solve_checkpointed(
                &problem,
                &config,
                "sha256:checkpoint-operator-v1",
                None,
                None,
                &mut stop,
                &CancellationToken::new(),
            )
            .unwrap();
        let BlockSolveOutcomeF64::Checkpointed { checkpoint } = first else {
            panic!("the checkpoint sink should have requested a clean stop");
        };
        assert_eq!(checkpoint.retained_ritz_basis.len(), config.block_size);
        assert!(
            checkpoint.retained_subspace_memory_bytes()
                <= ((config.block_size * operator.dimension() + config.block_size) * 8) as u64
        );

        let mut keep_going = ContinueCheckpointing;
        let resumed = solver
            .solve_checkpointed(
                &problem,
                &config,
                "sha256:checkpoint-operator-v1",
                None,
                Some(&checkpoint),
                &mut keep_going,
                &CancellationToken::new(),
            )
            .unwrap();
        let BlockSolveOutcomeF64::Complete { report } = resumed else {
            panic!("the resumed solve should converge");
        };
        assert_eq!(*report, uninterrupted);
    }

    #[test]
    fn checkpoint_resume_rejects_operator_identity_drift() {
        let operator =
            DiagonalF64::new("checkpoint_identity", vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0]).unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let config = block_config(EigenTarget::AlgebraicLargest, 2, 3);
        let solver = BlockSubspaceIterationF64::default();
        let mut stop = StopAtIteration { iteration: 2 };
        let outcome = solver
            .solve_checkpointed(
                &problem,
                &config,
                "sha256:original",
                None,
                None,
                &mut stop,
                &CancellationToken::new(),
            )
            .unwrap();
        let BlockSolveOutcomeF64::Checkpointed { checkpoint } = outcome else {
            panic!("expected a resumable checkpoint");
        };
        let mut keep_going = ContinueCheckpointing;
        let error = solver
            .solve_checkpointed(
                &problem,
                &config,
                "sha256:different",
                None,
                Some(&checkpoint),
                &mut keep_going,
                &CancellationToken::new(),
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("operator identity is incompatible"));
    }
}

/// Independent dense route.  It materializes a matrix by applying the
/// operator to coordinate vectors, checks symmetry, then uses nalgebra's
/// symmetric eigendecomposition.  Intended for validation-scale problems.
#[derive(Clone, Debug)]
pub struct DenseReferenceSolverF64 {
    pub maximum_dimension: usize,
    pub symmetry_tolerance: f64,
}

impl Default for DenseReferenceSolverF64 {
    fn default() -> Self {
        Self {
            maximum_dimension: 2048,
            symmetry_tolerance: 1e-11,
        }
    }
}

impl EigenSolverF64 for DenseReferenceSolverF64 {
    fn name(&self) -> &'static str {
        "dense_materialized_reference_f64"
    }

    fn solve(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
    ) -> Result<EigenpairReportF64, SolverError> {
        self.solve_controlled(problem, config, &CancellationToken::new())
    }

    fn solve_controlled(
        &self,
        problem: &SymmetricProblemF64<'_>,
        config: &SolverConfig,
        cancellation: &CancellationToken,
    ) -> Result<EigenpairReportF64, SolverError> {
        check_solver_cancellation(cancellation)?;
        config
            .validate()
            .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
        let largest = supported_extreme(&config.target)?;
        let (absolute_residual, backward_tolerance) = stopping_thresholds_f64(config)?;
        let n = problem.operator.dimension();
        if n == 0 || n > self.maximum_dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "dense reference dimension {n} is outside 1..={}",
                self.maximum_dimension
            )));
        }
        let mut matrix = DMatrix::<f64>::zeros(n, n);
        let mut e = vec![0.0; n];
        let mut column = vec![0.0; n];
        for j in 0..n {
            check_solver_cancellation(cancellation)?;
            e[j] = 1.0;
            problem.operator.apply(&e, &mut column)?;
            e[j] = 0.0;
            for i in 0..n {
                matrix[(i, j)] = column[i];
            }
        }
        for i in 0..n {
            check_solver_cancellation(cancellation)?;
            for j in 0..i {
                if (matrix[(i, j)] - matrix[(j, i)]).abs() > self.symmetry_tolerance {
                    return Err(SolverError::NumericalBreakdown(format!(
                        "materialized operator is not symmetric at ({i}, {j})"
                    )));
                }
                let average = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
                matrix[(i, j)] = average;
                matrix[(j, i)] = average;
            }
        }
        let decomposition = SymmetricEigen::new(matrix);
        let index = if largest {
            (0..n)
                .max_by(|&a, &b| {
                    decomposition.eigenvalues[a].total_cmp(&decomposition.eigenvalues[b])
                })
                .unwrap()
        } else {
            (0..n)
                .min_by(|&a, &b| {
                    decomposition.eigenvalues[a].total_cmp(&decomposition.eigenvalues[b])
                })
                .unwrap()
        };
        let mut vector: Vec<f64> = decomposition
            .eigenvectors
            .column(index)
            .iter()
            .copied()
            .collect();
        normalize(&mut vector)?;
        let mut workspace = vec![0.0; n];
        let (lambda, residual, relative, backward) =
            evaluate_eigenpair(problem.operator, &vector, &mut workspace)?;
        let (status, termination) = if backward <= backward_tolerance {
            (
                ResultStatus::Converged,
                TerminationReason::BackwardErrorTolerance,
            )
        } else if residual <= absolute_residual {
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
        let report = EigenpairReportF64 {
            eigenvalue: lambda,
            eigenvector: vector,
            residual_norm: residual,
            relative_residual: relative,
            scaled_backward_error: backward,
            iterations: 1,
            operator_applications: n + 1,
            algorithm: self.name().to_owned(),
            status,
            termination,
            assurance: AssuranceLevel::Computed,
            provenance: SolverProvenance::current_package("f64"),
        };
        report.validate_finite()?;
        Ok(report)
    }
}

pub fn cross_check_f64(
    primary: &dyn EigenSolverF64,
    independent: &dyn EigenSolverF64,
    problem: &SymmetricProblemF64<'_>,
    config: &SolverConfig,
    tolerance: f64,
) -> Result<CrossCheckedEigenpairF64, SolverError> {
    cross_check_f64_controlled(
        primary,
        independent,
        problem,
        config,
        tolerance,
        &CancellationToken::new(),
    )
}

pub fn cross_check_f64_controlled(
    primary: &dyn EigenSolverF64,
    independent: &dyn EigenSolverF64,
    problem: &SymmetricProblemF64<'_>,
    config: &SolverConfig,
    tolerance: f64,
    cancellation: &CancellationToken,
) -> Result<CrossCheckedEigenpairF64, SolverError> {
    check_solver_cancellation(cancellation)?;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(SolverError::InvalidConfiguration(
            "cross-check tolerance must be finite and positive".to_owned(),
        ));
    }
    let mut a = primary.solve_controlled(problem, config, cancellation)?;
    check_solver_cancellation(cancellation)?;
    let b = independent.solve_controlled(problem, config, cancellation)?;
    check_solver_cancellation(cancellation)?;
    let scale = a.eigenvalue.abs().max(b.eigenvalue.abs()).max(1.0);
    let difference = (a.eigenvalue - b.eigenvalue).abs();
    let overlap = dot(&a.eigenvector, &b.eigenvector).abs();
    let overlap_sq = overlap * overlap;
    if difference > tolerance * scale {
        return Err(SolverError::CrossCheckDisagreement(format!(
            "{} returned {:.17e}, {} returned {:.17e}; difference {:.3e} exceeds {:.3e}",
            primary.name(),
            a.eigenvalue,
            independent.name(),
            b.eigenvalue,
            difference,
            tolerance * scale
        )));
    }
    a.assurance = AssuranceLevel::CrossChecked;
    Ok(CrossCheckedEigenpairF64 {
        accepted: a,
        independent: b,
        eigenvalue_difference: difference,
        vector_overlap_squared: overlap_sq,
        tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_core::{CancellationReason, PrecisionPolicy, Reproducibility, StoppingPolicy, Subspace};
    use xc_operator::{
        DenseSymmetricF64, DiagonalF64, LinearOperator, MatrixFreeSymmetricF64, PackedSymmetricF64,
        SymmetricBandedF64, SymmetricOperator,
    };

    fn config(target: EigenTarget) -> SolverConfig {
        SolverConfig {
            target,
            subspace: Subspace::Full,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            stopping: StoppingPolicy {
                absolute_residual: xc_core::DecimalLiteral::new("1e-11").unwrap(),
                scaled_backward_error: xc_core::DecimalLiteral::new("1e-11").unwrap(),
                maximum_iterations: 100,
                minimum_iterations: 2,
            },
            reproducibility: Reproducibility::Deterministic,
            algorithm_preferences: Vec::new(),
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        }
    }

    #[test]
    fn shared_rayleigh_image_is_bit_identical_to_two_application_route() {
        let operator = DiagonalF64::new("diagnostic-equivalence", vec![1.25, -3.5, 7.0]).unwrap();
        let vector = [0.25, -0.75, 0.5];
        let mut rayleigh_image = [0.0; 3];
        operator.apply(&vector, &mut rayleigh_image).unwrap();
        let expected_value = dot(&vector, &rayleigh_image) / dot(&vector, &vector);
        let mut diagnostic_image = [0.0; 3];
        operator.apply(&vector, &mut diagnostic_image).unwrap();
        let vector_norm = norm(&vector);
        let mut residual_sq = 0.0;
        let mut applied_sq = 0.0;
        for (applied, component) in diagnostic_image.iter().zip(vector) {
            let residual = applied - expected_value * component;
            residual_sq += residual * residual;
            applied_sq += applied * applied;
        }
        let expected_residual = residual_sq.sqrt();
        let applied_norm = applied_sq.sqrt();
        let expected_relative = expected_residual
            / (applied_norm + expected_value.abs() * vector_norm).max(f64::MIN_POSITIVE);
        let norm_bound = operator.norm_bound().unwrap();
        let expected_backward = expected_residual
            / (norm_bound * vector_norm + expected_value.abs() * vector_norm)
                .max(f64::MIN_POSITIVE);

        let mut optimized_image = [0.0; 3];
        let (value, residual, relative, backward) =
            evaluate_eigenpair(&operator, &vector, &mut optimized_image).unwrap();
        assert_eq!(value.to_bits(), expected_value.to_bits());
        assert_eq!(residual.to_bits(), expected_residual.to_bits());
        assert_eq!(relative.to_bits(), expected_relative.to_bits());
        assert_eq!(backward.to_bits(), expected_backward.to_bits());
        assert_eq!(optimized_image, diagnostic_image);
    }

    #[test]
    fn dense_reference_finds_both_extremes() {
        let d = DiagonalF64::new("diag", vec![-3.0, 1.0, 7.0, 2.0]).unwrap();
        let reference = DenseReferenceSolverF64::default();
        let largest = reference
            .solve(
                &SymmetricProblemF64::new(&d),
                &config(EigenTarget::AlgebraicLargest),
            )
            .unwrap();
        assert!((largest.eigenvalue - 7.0).abs() < 1e-12);
        let smallest = reference
            .solve(
                &SymmetricProblemF64::new(&d),
                &config(EigenTarget::AlgebraicSmallest),
            )
            .unwrap();
        assert!((smallest.eigenvalue + 3.0).abs() < 1e-12);
    }

    #[test]
    fn solver_stops_before_work_when_cancelled() {
        let d = DiagonalF64::new("diag", vec![-3.0, 1.0, 7.0, 2.0]).unwrap();
        let cancellation = CancellationToken::new();
        assert!(cancellation.cancel(CancellationReason::UserRequested));
        let result = LanczosSolverF64::default().solve_controlled(
            &SymmetricProblemF64::new(&d),
            &config(EigenTarget::AlgebraicLargest),
            &cancellation,
        );
        assert!(matches!(result, Err(SolverError::Cancelled(_))));
    }

    #[test]
    fn public_f64_solvers_reject_zero_iterations_before_execution() {
        let operator = DiagonalF64::new("diag", vec![1.0, 2.0]).unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let mut invalid = config(EigenTarget::AlgebraicLargest);
        invalid.stopping.maximum_iterations = 0;
        invalid.stopping.minimum_iterations = 0;
        let solvers: [&dyn EigenSolverF64; 3] = [
            &DenseReferenceSolverF64::default(),
            &LanczosSolverF64::default(),
            &ShiftedPowerSolverF64,
        ];
        for solver in solvers {
            assert!(matches!(
                solver.solve(&problem, &invalid),
                Err(SolverError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn lanczos_cross_checks_dense_route() {
        let a = DenseSymmetricF64::new(
            "laplacian",
            4,
            vec![
                2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0,
                2.0,
            ],
            0.0,
        )
        .unwrap();
        let target = EigenTarget::AlgebraicSmallest;
        let result = cross_check_f64(
            &LanczosSolverF64::default(),
            &DenseReferenceSolverF64::default(),
            &SymmetricProblemF64::new(&a),
            &config(target),
            1e-10,
        )
        .unwrap();
        assert_eq!(result.accepted.assurance, AssuranceLevel::CrossChecked);
        assert!(result.vector_overlap_squared > 1.0 - 1e-12);
    }

    #[test]
    fn shifted_power_uses_algebraic_not_magnitude_target() {
        let d = DiagonalF64::new("diag", vec![-100.0, 1.0, 50.0]).unwrap();
        let solver = ShiftedPowerSolverF64;
        let target = EigenTarget::AlgebraicLargest;
        let result = solver
            .solve(&SymmetricProblemF64::new(&d), &config(target))
            .unwrap();
        assert!((result.eigenvalue - 50.0).abs() < 1e-9);
    }

    #[test]
    fn common_solver_contract_accepts_packed_banded_and_matrix_free() {
        let dense = DenseSymmetricF64::new("dense", 2, vec![1.0, 0.0, 0.0, 4.0], 0.0).unwrap();
        let packed = PackedSymmetricF64::new("packed", 2, vec![1.0, 0.0, 4.0]).unwrap();
        let banded = SymmetricBandedF64::new("banded", vec![vec![1.0, 4.0]]).unwrap();
        let matrix_free =
            MatrixFreeSymmetricF64::exact("matrix-free", 2, Some(4.0), |input, output| {
                output[0] = input[0];
                output[1] = 4.0 * input[1];
                Ok(())
            })
            .unwrap();
        let solver = ShiftedPowerSolverF64;
        for operator in [
            &dense as &dyn SymmetricOperator<f64>,
            &packed,
            &banded,
            &matrix_free,
        ] {
            let result = solver
                .solve(
                    &SymmetricProblemF64::new(operator),
                    &config(EigenTarget::AlgebraicLargest),
                )
                .unwrap();
            assert!((result.eigenvalue - 4.0).abs() < 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// High-precision eigensolver implementation.
// ---------------------------------------------------------------------------

#[cfg(feature = "hp-reference")]
pub use xc_numerics::eigen::{
    HpSelectedTridiagonalEigenpair, HpSelectedTridiagonalEigenpairOptions,
    HpSelectedTridiagonalEigenpairs, HpSelectedTridiagonalItem, HpSelectedTridiagonalSpectrum,
    HpTridiagonalEigenvalueCluster, HpTridiagonalEigenvalueEnclosure, TridiagEigvecOptions,
};

#[cfg(feature = "hp-reference")]
pub struct TridiagonalProblemHp<'a> {
    pub diagonal: &'a [rug::Float],
    pub off_diagonal: &'a [rug::Float],
}

#[cfg(feature = "hp-reference")]
impl<'a> TridiagonalProblemHp<'a> {
    pub fn new(
        diagonal: &'a [rug::Float],
        off_diagonal: &'a [rug::Float],
    ) -> Result<Self, SolverError> {
        if diagonal.is_empty() || off_diagonal.len() + 1 != diagonal.len() {
            return Err(SolverError::InvalidConfiguration(
                "HP tridiagonal problem requires off_diagonal.len() + 1 == diagonal.len() > 0"
                    .to_owned(),
            ));
        }
        if diagonal
            .iter()
            .chain(off_diagonal)
            .any(|value| !value.is_finite())
        {
            return Err(SolverError::InvalidConfiguration(
                "HP tridiagonal entries must be finite".to_owned(),
            ));
        }
        Ok(Self {
            diagonal,
            off_diagonal,
        })
    }

    pub fn dimension(&self) -> usize {
        self.diagonal.len()
    }
}

/// Execute the production HP Sturm route for an inclusive algebraic index
/// range without forming or diagonalizing a dense matrix.
#[cfg(feature = "hp-reference")]
pub fn solve_tridiagonal_selected_hp(
    problem: &TridiagonalProblemHp<'_>,
    first_index: usize,
    last_index: usize,
    absolute_tolerance: &rug::Float,
    maximum_iterations: usize,
    precision_bits: u32,
) -> Result<HpSelectedTridiagonalSpectrum, SolverError> {
    if first_index > last_index
        || last_index >= problem.dimension()
        || precision_bits <= 32
        || maximum_iterations == 0
        || !absolute_tolerance.is_finite()
        || absolute_tolerance <= &rug::Float::with_val(precision_bits.max(2), 0)
    {
        return Err(SolverError::InvalidConfiguration(
            "HP selected tridiagonal solve requires a valid inclusive index range, precision above 32 bits, positive finite tolerance, and a positive iteration limit"
                .to_owned(),
        ));
    }
    xc_numerics::eigen::tridiag_selected_eigenvalues_hp(
        problem.diagonal,
        problem.off_diagonal,
        first_index,
        last_index,
        absolute_tolerance,
        maximum_iterations,
        precision_bits,
    )
    .map_err(|error| SolverError::NonConvergence(error.to_string()))
}

/// Execute selected value isolation followed by residual-verified banded HP
/// inverse iteration for simple values. Unresolved multiplicities are returned
/// as clusters without individual vectors.
#[cfg(feature = "hp-reference")]
pub fn solve_tridiagonal_selected_eigenpairs_hp(
    problem: &TridiagonalProblemHp<'_>,
    options: &HpSelectedTridiagonalEigenpairOptions,
) -> Result<HpSelectedTridiagonalEigenpairs, SolverError> {
    if options.first_index > options.last_index
        || options.last_index >= problem.dimension()
        || options.precision_bits <= 32
        || options.maximum_bisection_iterations == 0
        || options.eigenvector_options.max_steps == 0
        || !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance <= rug::Float::with_val(options.precision_bits.max(2), 0)
    {
        return Err(SolverError::InvalidConfiguration(
            "HP selected tridiagonal eigenpair solve requires a valid inclusive index range, precision above 32 bits, positive finite tolerance, and positive bisection/vector step limits"
                .to_owned(),
        ));
    }
    xc_numerics::eigen::tridiag_selected_eigenpairs_hp(
        problem.diagonal,
        problem.off_diagonal,
        options,
    )
    .map_err(|error| SolverError::NonConvergence(error.to_string()))
}

#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug)]
pub struct HpAdaptiveSelectedTridiagonalOptions {
    pub first_index: usize,
    pub last_index: usize,
    pub absolute_tolerance: xc_core::DecimalLiteral,
    pub maximum_bisection_iterations: usize,
    pub eigenvector_options: TridiagEigvecOptions,
    pub precision: xc_core::PrecisionPolicy,
}

#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HpSelectedPrecisionAttempt {
    pub precision_bits: u32,
    pub status: ResultStatus,
    pub selected_items: usize,
    pub vector_recoveries: usize,
    pub inverse_iteration_runs: usize,
    pub reason: String,
}

#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug)]
pub enum HpAdaptiveSelectedTridiagonalResult {
    Converged {
        result: Box<HpSelectedTridiagonalEigenpairs>,
        attempts: Vec<HpSelectedPrecisionAttempt>,
    },
    Inconclusive {
        last_result: Option<Box<HpSelectedTridiagonalEigenpairs>>,
        attempts: Vec<HpSelectedPrecisionAttempt>,
        reason: String,
    },
}

/// Runs selected HP tridiagonal eigenpairs with deterministic precision escalation.
///
/// # Mathematical semantics
/// Computes the requested indexed eigenpairs of a real symmetric tridiagonal
/// problem, retaining attempt evidence and selection guards.
///
/// # Precision
/// Inputs must retain at least `maximum_bits`. Each attempt rounds them to its
/// declared working precision, and escalation follows the supplied policy; an
/// HP request never falls back to binary64.
///
/// # Failure states
/// Invalid dimensions, indices, precision policies, and backend failures return
/// `SolverError`. Exhausted precision returns an explicit inconclusive result
/// with its attempt history rather than guessed eigenpairs.
///
/// # Assurance and validity
/// The result reports residual and guard evidence for a finite tridiagonal
/// problem. Certification or an independent route is still required when the
/// requested assurance demands it.
///
/// # Cache effects
/// This numerical entry point performs no implicit cache access; callers attach
/// inputs and results to the common artifact plan and provenance model.
///
/// # Example
/// Planning is exercised by `crates/xc-solver/examples/plan.rs`.
#[cfg(feature = "hp-reference")]
pub fn solve_tridiagonal_selected_eigenpairs_adaptive_hp(
    problem: &TridiagonalProblemHp<'_>,
    options: &HpAdaptiveSelectedTridiagonalOptions,
) -> Result<HpAdaptiveSelectedTridiagonalResult, SolverError> {
    options
        .precision
        .validate()
        .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
    if problem
        .diagonal
        .iter()
        .chain(problem.off_diagonal)
        .any(|value| value.prec() < options.precision.maximum_bits)
    {
        return Err(SolverError::InvalidConfiguration(format!(
            "adaptive HP tridiagonal inputs must retain at least maximum_bits={} precision",
            options.precision.maximum_bits
        )));
    }
    let parsed_tolerance = rug::Float::parse(options.absolute_tolerance.as_str())
        .map(|value| rug::Float::with_val(options.precision.maximum_bits, value))
        .map_err(|error| {
            SolverError::InvalidConfiguration(format!(
                "failed to parse adaptive HP tolerance: {error}"
            ))
        })?;
    let mut precision_bits = options
        .precision
        .initial_bits
        .saturating_add(options.precision.guard_bits)
        .min(options.precision.maximum_bits);
    let mut attempts = Vec::new();
    let mut last_result = None;
    loop {
        let attempt_options = HpSelectedTridiagonalEigenpairOptions {
            first_index: options.first_index,
            last_index: options.last_index,
            absolute_tolerance: rug::Float::with_val(precision_bits, &parsed_tolerance),
            maximum_bisection_iterations: options.maximum_bisection_iterations,
            eigenvector_options: options.eigenvector_options,
            precision_bits,
        };
        match solve_tridiagonal_selected_eigenpairs_hp(problem, &attempt_options) {
            Ok(result) => {
                let has_cluster = result
                    .items
                    .iter()
                    .any(|item| matches!(item, HpSelectedTridiagonalItem::Cluster(_)));
                attempts.push(HpSelectedPrecisionAttempt {
                    precision_bits,
                    status: if has_cluster {
                        ResultStatus::UnresolvedCluster
                    } else {
                        ResultStatus::Converged
                    },
                    selected_items: result.items.len(),
                    vector_recoveries: result.vector_recoveries,
                    inverse_iteration_runs: result.inverse_iteration_runs,
                    reason: if has_cluster {
                        "one or more endpoint-count clusters remain unresolved".to_owned()
                    } else {
                        "all requested values are simple and residual-verified".to_owned()
                    },
                });
                if !has_cluster {
                    return Ok(HpAdaptiveSelectedTridiagonalResult::Converged {
                        result: Box::new(result),
                        attempts,
                    });
                }
                last_result = Some(Box::new(result));
            }
            Err(error @ SolverError::InvalidConfiguration(_)) => return Err(error),
            Err(error) => attempts.push(HpSelectedPrecisionAttempt {
                precision_bits,
                status: ResultStatus::InsufficientPrecision,
                selected_items: 0,
                vector_recoveries: 0,
                inverse_iteration_runs: 0,
                reason: error.to_string(),
            }),
        }
        let Some(next_bits) = options.precision.next_bits(precision_bits) else {
            return Ok(HpAdaptiveSelectedTridiagonalResult::Inconclusive {
                last_result,
                attempts,
                reason: format!(
                    "selected HP eigenpairs remain unresolved at maximum precision {}",
                    options.precision.maximum_bits
                ),
            });
        };
        precision_bits = next_bits;
    }
}

#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EigenpairReportHp {
    pub eigenvalue: String,
    pub eigenvector: Vec<String>,
    pub residual_norm: String,
    pub relative_residual: String,
    pub scaled_backward_error: String,
    pub diagnostics: EigenpairDiagnostics<String>,
    pub precision_bits: u32,
    pub algorithm: String,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

#[cfg(feature = "hp-reference")]
pub struct DenseSymmetricProblemHp<'a> {
    pub matrix: &'a [rug::Float],
    pub dimension: usize,
}

#[cfg(feature = "hp-reference")]
impl<'a> DenseSymmetricProblemHp<'a> {
    pub fn new(matrix: &'a [rug::Float], dimension: usize) -> Result<Self, SolverError> {
        if dimension == 0 || matrix.len() != dimension.saturating_mul(dimension) {
            return Err(SolverError::InvalidConfiguration(format!(
                "dense HP matrix length {} does not match dimension {dimension}",
                matrix.len()
            )));
        }
        Ok(Self { matrix, dimension })
    }
}

#[cfg(feature = "hp-reference")]
fn hp_zero(precision_bits: u32) -> rug::Float {
    rug::Float::with_val(precision_bits, 0)
}

#[cfg(feature = "hp-reference")]
fn hp_parse_literal(
    literal: &xc_core::DecimalLiteral,
    precision_bits: u32,
) -> Result<rug::Float, SolverError> {
    let parsed = rug::Float::parse(literal.as_str()).map_err(|error| {
        SolverError::InvalidConfiguration(format!(
            "failed to parse HP decimal literal {:?}: {error}",
            literal.as_str()
        ))
    })?;
    Ok(rug::Float::with_val(precision_bits, parsed))
}

#[cfg(feature = "hp-reference")]
fn hp_norm(values: &[rug::Float], precision_bits: u32) -> rug::Float {
    let mut sum = hp_zero(precision_bits);
    for value in values {
        let mut square = value.clone();
        square *= value;
        sum += square;
    }
    sum.sqrt_mut();
    sum
}

#[cfg(feature = "hp-reference")]
fn hp_matvec(
    matrix: &[rug::Float],
    dimension: usize,
    vector: &[rug::Float],
    precision_bits: u32,
) -> Vec<rug::Float> {
    (0..dimension)
        .map(|row| {
            let mut sum = hp_zero(precision_bits);
            for column in 0..dimension {
                let mut term = matrix[row * dimension + column].clone();
                term *= &vector[column];
                sum += term;
            }
            sum
        })
        .collect()
}

#[cfg(feature = "hp-reference")]
fn hp_decimal(value: &rug::Float, significant_digits: usize) -> String {
    value.to_string_radix(10, Some(significant_digits))
}

/// Run the established dense HP full-spectrum route behind the new typed
/// target and report contracts. This adapter is intentionally a reference
/// implementation: it provides the trusted dense algorithm while the
/// selected-spectrum v0.13.0 solvers are developed independently.
#[cfg(feature = "hp-reference")]
pub fn solve_dense_reference_hp(
    problem: &DenseSymmetricProblemHp<'_>,
    config: &SolverConfig,
) -> Result<EigenpairReportHp, SolverError> {
    solve_dense_reference_hp_controlled(problem, config, &CancellationToken::new())
}

#[cfg(feature = "hp-reference")]
pub fn solve_dense_reference_hp_controlled(
    problem: &DenseSymmetricProblemHp<'_>,
    config: &SolverConfig,
    cancellation: &CancellationToken,
) -> Result<EigenpairReportHp, SolverError> {
    use rug::Float;
    use xc_numerics::eigen::{
        dense_symmetric_eigenvalues_hp, dense_symmetric_eigenvector_for_value_hp,
    };

    check_solver_cancellation(cancellation)?;
    config
        .validate()
        .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
    if !matches!(config.subspace, xc_core::Subspace::Full) {
        return Err(SolverError::UnsupportedTarget(
            "dense HP reference adapter currently requires an already reduced Full subspace"
                .to_owned(),
        ));
    }
    let precision_bits = config.precision.initial_bits;
    let eigenvalues =
        dense_symmetric_eigenvalues_hp(problem.matrix, problem.dimension, precision_bits)
            .map_err(|error| SolverError::NumericalBreakdown(error.to_string()))?;
    check_solver_cancellation(cancellation)?;
    if eigenvalues.is_empty() {
        return Err(SolverError::NumericalBreakdown(
            "HP eigensolver returned no eigenvalues".to_owned(),
        ));
    }

    let index = match &config.target {
        EigenTarget::AlgebraicSmallest => 0,
        EigenTarget::AlgebraicLargest => eigenvalues.len() - 1,
        EigenTarget::SmallestMagnitude => eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (*left)
                    .clone()
                    .abs()
                    .partial_cmp(&(*right).clone().abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .expect("eigenvalues is nonempty"),
        EigenTarget::ClosestTo { shift } => {
            let shift = hp_parse_literal(shift, precision_bits)?;
            eigenvalues
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    let mut left_distance = (*left).clone();
                    left_distance -= &shift;
                    left_distance.abs_mut();
                    let mut right_distance = (*right).clone();
                    right_distance -= &shift;
                    right_distance.abs_mut();
                    left_distance
                        .partial_cmp(&right_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index)
                .expect("eigenvalues is nonempty")
        }
        EigenTarget::IndexRange { .. } | EigenTarget::Interval { .. } => {
            return Err(SolverError::UnsupportedTarget(
                "single-eigenpair HP reference adapter does not accept range targets".to_owned(),
            ))
        }
    };
    let eigenvalue = eigenvalues[index].clone();
    let eigenvector = dense_symmetric_eigenvector_for_value_hp(
        problem.matrix,
        problem.dimension,
        &eigenvalue,
        precision_bits,
        config.stopping.maximum_iterations,
    )
    .map_err(|error| SolverError::NumericalBreakdown(error.to_string()))?;
    check_solver_cancellation(cancellation)?;

    let applied = hp_matvec(
        problem.matrix,
        problem.dimension,
        &eigenvector,
        precision_bits,
    );
    let residual: Vec<Float> = applied
        .iter()
        .zip(&eigenvector)
        .map(|(value, component)| {
            let mut term = eigenvalue.clone();
            term *= component;
            let mut difference = value.clone();
            difference -= term;
            difference
        })
        .collect();
    let residual_norm = hp_norm(&residual, precision_bits);
    let applied_norm = hp_norm(&applied, precision_bits);
    let vector_norm = hp_norm(&eigenvector, precision_bits);
    let mut eigenvalue_abs = eigenvalue.clone();
    eigenvalue_abs.abs_mut();

    let mut denominator = eigenvalue_abs.clone();
    denominator *= &vector_norm;
    denominator += &applied_norm;
    let relative_residual = if denominator.is_zero() {
        residual_norm.clone()
    } else {
        let mut value = residual_norm.clone();
        value /= &denominator;
        value
    };

    let mut infinity_bound = hp_zero(precision_bits);
    for row in 0..problem.dimension {
        check_solver_cancellation(cancellation)?;
        let mut row_sum = hp_zero(precision_bits);
        for column in 0..problem.dimension {
            row_sum += problem.matrix[row * problem.dimension + column]
                .clone()
                .abs();
        }
        if row_sum > infinity_bound {
            infinity_bound = row_sum;
        }
    }
    let mut backward_denominator = infinity_bound;
    backward_denominator *= &vector_norm;
    let mut eigen_term = eigenvalue_abs;
    eigen_term *= &vector_norm;
    backward_denominator += eigen_term;
    let scaled_backward_error = if backward_denominator.is_zero() {
        residual_norm.clone()
    } else {
        let mut value = residual_norm.clone();
        value /= backward_denominator;
        value
    };
    let mut orthogonality_error = vector_norm.clone();
    orthogonality_error *= &vector_norm;
    orthogonality_error -= 1u32;
    orthogonality_error.abs_mut();

    let residual_tolerance = hp_parse_literal(&config.stopping.absolute_residual, precision_bits)?;
    let backward_tolerance =
        hp_parse_literal(&config.stopping.scaled_backward_error, precision_bits)?;
    let (status, termination) = if scaled_backward_error <= backward_tolerance {
        (
            ResultStatus::Converged,
            TerminationReason::BackwardErrorTolerance,
        )
    } else if residual_norm <= residual_tolerance {
        (
            ResultStatus::Converged,
            TerminationReason::ResidualTolerance,
        )
    } else {
        (
            ResultStatus::Approximate,
            TerminationReason::MaximumIterations,
        )
    };

    // Exact-round-trip width; the bare ceiling loses one ulp on decode.
    let decimal_digits = xc_numerics::reduction::roundtrip_decimal_digits(precision_bits).max(32);
    let mut provenance = SolverProvenance::current_package("rug_mpfr");
    provenance.precision_bits = Some(precision_bits);
    let diagnostics = EigenpairDiagnostics {
        absolute_residual: hp_decimal(&residual_norm, decimal_digits),
        relative_residual: hp_decimal(&relative_residual, decimal_digits),
        scaled_backward_error: hp_decimal(&scaled_backward_error, decimal_digits),
        orthogonality_error: hp_decimal(&orthogonality_error, decimal_digits),
    };
    Ok(EigenpairReportHp {
        eigenvalue: hp_decimal(&eigenvalue, decimal_digits),
        eigenvector: eigenvector
            .iter()
            .map(|value| hp_decimal(value, decimal_digits))
            .collect(),
        residual_norm: hp_decimal(&residual_norm, decimal_digits),
        relative_residual: hp_decimal(&relative_residual, decimal_digits),
        scaled_backward_error: hp_decimal(&scaled_backward_error, decimal_digits),
        diagnostics,
        precision_bits,
        algorithm: "xc_numerics_dense_householder_qr_reference_hp".to_owned(),
        status,
        termination,
        assurance: AssuranceLevel::Computed,
        provenance,
    })
}

#[cfg(all(test, feature = "hp-reference"))]
mod hp_reference_tests {
    use super::*;
    use rug::Float;
    use xc_core::{
        AssuranceLevel, DecimalLiteral, PrecisionEscalation, PrecisionPolicy, Reproducibility,
        StoppingPolicy, Subspace,
    };

    fn config(target: EigenTarget) -> SolverConfig {
        SolverConfig {
            target,
            subspace: Subspace::Full,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(256),
            stopping: StoppingPolicy {
                absolute_residual: DecimalLiteral::new("1e-50").unwrap(),
                scaled_backward_error: DecimalLiteral::new("1e-50").unwrap(),
                maximum_iterations: 30,
                minimum_iterations: 2,
            },
            reproducibility: Reproducibility::Deterministic,
            algorithm_preferences: Vec::new(),
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        }
    }

    #[test]
    fn hp_reference_adapter_respects_algebraic_target() {
        let precision = 256;
        let matrix = vec![
            Float::with_val(precision, -3),
            Float::with_val(precision, 0),
            Float::with_val(precision, 0),
            Float::with_val(precision, 7),
        ];
        let problem = DenseSymmetricProblemHp::new(&matrix, 2).unwrap();
        let report =
            solve_dense_reference_hp(&problem, &config(EigenTarget::AlgebraicSmallest)).unwrap();
        let mut difference = Float::with_val(precision, Float::parse(&report.eigenvalue).unwrap());
        difference += 3;
        difference.abs_mut();
        assert!(difference < Float::with_val(precision, 1e-40));
        assert_eq!(report.diagnostics.absolute_residual, report.residual_norm);
        assert_eq!(
            report.diagnostics.relative_residual,
            report.relative_residual
        );
        assert_eq!(
            report.diagnostics.scaled_backward_error,
            report.scaled_backward_error
        );
        let orthogonality = Float::with_val(
            precision,
            Float::parse(&report.diagnostics.orthogonality_error).unwrap(),
        );
        assert!(orthogonality < Float::with_val(precision, 1e-40));
    }

    #[test]
    fn hp_selected_tridiagonal_adapter_executes_index_range() {
        let precision = 256;
        let diagonal = vec![Float::with_val(precision, 2); 8];
        let off_diagonal = vec![Float::with_val(precision, -1); 7];
        let problem = TridiagonalProblemHp::new(&diagonal, &off_diagonal).unwrap();
        let report = solve_tridiagonal_selected_hp(
            &problem,
            1,
            3,
            &Float::with_val(precision, 1e-30),
            200,
            precision,
        )
        .unwrap();
        assert_eq!(report.enclosures.len(), 3);
        assert_eq!(report.enclosures[0].index, 1);
        assert_eq!(report.enclosures[2].index, 3);

        let pairs = solve_tridiagonal_selected_eigenpairs_hp(
            &problem,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: 1,
                last_index: 2,
                absolute_tolerance: Float::with_val(precision, 1e-30),
                maximum_bisection_iterations: 200,
                eigenvector_options: TridiagEigvecOptions::default(),
                precision_bits: precision,
            },
        )
        .unwrap();
        assert_eq!(pairs.vector_recoveries, 2);
        assert!(pairs.inverse_iteration_runs >= pairs.vector_recoveries);
        assert!(pairs
            .items
            .iter()
            .all(|item| matches!(item, HpSelectedTridiagonalItem::SimpleEigenpair(_))));
    }

    #[test]
    fn adaptive_hp_selected_eigenpairs_escalate_after_precision_stagnation() {
        let maximum_precision = 256;
        let diagonal = vec![Float::with_val(maximum_precision, 2); 8];
        let off_diagonal = vec![Float::with_val(maximum_precision, -1); 7];
        let problem = TridiagonalProblemHp::new(&diagonal, &off_diagonal).unwrap();
        let result = solve_tridiagonal_selected_eigenpairs_adaptive_hp(
            &problem,
            &HpAdaptiveSelectedTridiagonalOptions {
                first_index: 1,
                last_index: 2,
                absolute_tolerance: DecimalLiteral::new("1e-40").unwrap(),
                maximum_bisection_iterations: 400,
                eigenvector_options: TridiagEigvecOptions::default(),
                precision: PrecisionPolicy {
                    initial_bits: 64,
                    maximum_bits: maximum_precision,
                    guard_bits: 0,
                    escalation: PrecisionEscalation::AddBits(192),
                },
            },
        )
        .unwrap();
        let HpAdaptiveSelectedTridiagonalResult::Converged { result, attempts } = result else {
            panic!("adaptive selected eigenpairs did not converge");
        };
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].precision_bits, 64);
        assert_eq!(attempts[0].status, ResultStatus::InsufficientPrecision);
        assert_eq!(attempts[1].precision_bits, maximum_precision);
        assert_eq!(attempts[1].status, ResultStatus::Converged);
        assert_eq!(result.vector_recoveries, 2);
    }
}

/// Validation-scale selected-eigenvalue engine for a real symmetric
/// tridiagonal matrix. The certified implementation will use the
/// same Sturm-count contract with interval arithmetic.
#[derive(Clone, Debug)]
pub struct TridiagonalProblemF64<'a> {
    pub diagonal: &'a [f64],
    pub off_diagonal: &'a [f64],
}

impl<'a> TridiagonalProblemF64<'a> {
    pub fn new(diagonal: &'a [f64], off_diagonal: &'a [f64]) -> Result<Self, SolverError> {
        if diagonal.is_empty() || off_diagonal.len() + 1 != diagonal.len() {
            return Err(SolverError::InvalidConfiguration(
                "tridiagonal problem requires off_diagonal.len() + 1 == diagonal.len()".to_owned(),
            ));
        }
        if diagonal
            .iter()
            .chain(off_diagonal)
            .any(|value| !value.is_finite())
        {
            return Err(SolverError::InvalidConfiguration(
                "tridiagonal entries must be finite".to_owned(),
            ));
        }
        Ok(Self {
            diagonal,
            off_diagonal,
        })
    }

    pub fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    pub fn gershgorin_bounds(&self) -> (f64, f64) {
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for index in 0..self.dimension() {
            let mut radius = 0.0;
            if index > 0 {
                radius += self.off_diagonal[index - 1].abs();
            }
            if index + 1 < self.dimension() {
                radius += self.off_diagonal[index].abs();
            }
            lower = lower.min(self.diagonal[index] - radius);
            upper = upper.max(self.diagonal[index] + radius);
        }
        let padding = f64::EPSILON.sqrt() * lower.abs().max(upper.abs()).max(1.0);
        (lower - padding, upper + padding)
    }

    /// Number of eigenvalues strictly below `threshold`, computed from the
    /// signs of the symmetric LDL^T pivots.
    pub fn sturm_count_below(&self, threshold: f64) -> Result<usize, SolverError> {
        if !threshold.is_finite() {
            return Err(SolverError::InvalidConfiguration(
                "Sturm threshold must be finite".to_owned(),
            ));
        }
        let scale = self
            .diagonal
            .iter()
            .chain(self.off_diagonal)
            .map(|value| value.abs())
            .fold(1.0, f64::max);
        let pivot_floor = f64::MIN_POSITIVE.max(f64::EPSILON * scale);
        let mut negative = 0usize;
        let mut pivot = self.diagonal[0] - threshold;
        if pivot < 0.0 {
            negative += 1;
        }
        if pivot.abs() < pivot_floor {
            pivot = if pivot.is_sign_negative() {
                -pivot_floor
            } else {
                pivot_floor
            };
        }
        for index in 1..self.dimension() {
            let off = self.off_diagonal[index - 1];
            pivot = self.diagonal[index] - threshold - off * off / pivot;
            if pivot < 0.0 {
                negative += 1;
            }
            if pivot.abs() < pivot_floor {
                pivot = if pivot.is_sign_negative() {
                    -pivot_floor
                } else {
                    pivot_floor
                };
            }
        }
        Ok(negative)
    }

    /// Enclose the zero-based `index`-th algebraically ordered eigenvalue.
    pub fn bisect_index(
        &self,
        index: usize,
        absolute_tolerance: f64,
        maximum_iterations: usize,
    ) -> Result<(f64, f64), SolverError> {
        if index >= self.dimension() {
            return Err(SolverError::InvalidConfiguration(format!(
                "eigenvalue index {index} is outside 0..{}",
                self.dimension()
            )));
        }
        if !absolute_tolerance.is_finite() || absolute_tolerance <= 0.0 {
            return Err(SolverError::InvalidConfiguration(
                "bisection tolerance must be finite and positive".to_owned(),
            ));
        }
        if maximum_iterations == 0 {
            return Err(SolverError::InvalidConfiguration(
                "maximum_iterations must be positive".to_owned(),
            ));
        }
        let (mut lower, mut upper) = self.gershgorin_bounds();
        for _ in 0..maximum_iterations {
            let midpoint = lower + 0.5 * (upper - lower);
            let count = self.sturm_count_below(midpoint)?;
            if count <= index {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
            if upper - lower <= absolute_tolerance {
                return Ok((lower, upper));
            }
        }
        Err(SolverError::NonConvergence(format!(
            "Sturm bisection did not enclose eigenvalue {index} to {absolute_tolerance:e}"
        )))
    }

    pub fn bisect_range(
        &self,
        first: usize,
        last: usize,
        absolute_tolerance: f64,
        maximum_iterations: usize,
    ) -> Result<Vec<(f64, f64)>, SolverError> {
        if first > last {
            return Err(SolverError::InvalidConfiguration(
                "selected eigenvalue range must satisfy first <= last".to_owned(),
            ));
        }
        (first..=last)
            .map(|index| self.bisect_index(index, absolute_tolerance, maximum_iterations))
            .collect()
    }
}

#[cfg(test)]
mod sturm_reference_tests {
    use super::*;

    #[test]
    fn sturm_bisection_matches_strang_eigenvalues() {
        let dimension = 8usize;
        let diagonal = vec![2.0; dimension];
        let off_diagonal = vec![-1.0; dimension - 1];
        let problem = TridiagonalProblemF64::new(&diagonal, &off_diagonal).unwrap();
        for index in 0..dimension {
            let (lower, upper) = problem.bisect_index(index, 1e-12, 200).unwrap();
            let k = index + 1;
            let expected =
                2.0 - 2.0 * (std::f64::consts::PI * k as f64 / (dimension + 1) as f64).cos();
            assert!(lower <= expected && expected <= upper);
            assert!(upper - lower <= 1e-12);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneralizedEigenpairReportF64 {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub residual_norm: f64,
    pub metric_norm_error: f64,
    pub algorithm: String,
    pub status: ResultStatus,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

/// Typed declaration of an optional f64 discovery preconditioner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreconditionerDescriptorF64 {
    pub id: String,
    pub changes_only_convergence: bool,
    pub approximation_error_bound: Option<f64>,
}

impl PreconditionerDescriptorF64 {
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.id.trim().is_empty() {
            return Err(SolverError::InvalidConfiguration(
                "preconditioner id must not be empty".to_owned(),
            ));
        }
        if let Some(bound) = self.approximation_error_bound {
            if !bound.is_finite() || bound < 0.0 {
                return Err(SolverError::InvalidConfiguration(
                    "preconditioner approximation error bound must be finite and nonnegative"
                        .to_owned(),
                ));
            }
            if self.changes_only_convergence && bound != 0.0 {
                return Err(SolverError::InvalidConfiguration(
                    "a convergence-only preconditioner cannot declare nonzero approximation error"
                        .to_owned(),
                ));
            }
        }
        if !self.changes_only_convergence && self.approximation_error_bound.is_none() {
            return Err(SolverError::InvalidConfiguration(
                "a preconditioner that may introduce approximation error must declare a bound"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Optional residual preconditioner for the f64 generalized discovery route.
pub trait GeneralizedPreconditionerF64: Send + Sync {
    fn descriptor(&self) -> PreconditionerDescriptorF64;
    fn apply(&self, residual: &[f64], output: &mut [f64]) -> Result<(), SolverError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralizedExtremeConfigF64 {
    pub target: EigenTarget,
    pub absolute_residual_tolerance: f64,
    pub scaled_backward_error_tolerance: f64,
    pub ritz_value_stability_tolerance: f64,
    pub maximum_iterations: usize,
    pub minimum_iterations: usize,
}

impl GeneralizedExtremeConfigF64 {
    pub fn validate(&self) -> Result<(), SolverError> {
        supported_extreme(&self.target)?;
        for (name, value) in [
            (
                "absolute_residual_tolerance",
                self.absolute_residual_tolerance,
            ),
            (
                "scaled_backward_error_tolerance",
                self.scaled_backward_error_tolerance,
            ),
            (
                "ritz_value_stability_tolerance",
                self.ritz_value_stability_tolerance,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SolverError::InvalidConfiguration(format!(
                    "{name} must be finite and strictly positive"
                )));
            }
        }
        if self.maximum_iterations == 0 || self.minimum_iterations > self.maximum_iterations {
            return Err(SolverError::InvalidConfiguration(
                "maximum_iterations must be positive and not less than minimum_iterations"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Matrix-free generalized extreme-eigenpair discovery report. This f64 route
/// produces a candidate for later HP repetition and exact/interval quotient
/// verification; it never labels that candidate Certified.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatrixFreeGeneralizedEigenpairReportF64 {
    pub target: EigenTarget,
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub scaled_backward_error: f64,
    pub metric_normalization_error: f64,
    pub ritz_value_stability: f64,
    pub target_ordering_established_by_full_space_projection: bool,
    pub iterations: usize,
    pub operator_applications: usize,
    pub metric_applications: usize,
    pub projected_factorizations: usize,
    pub preconditioner_applications: usize,
    pub retained_subspace_vectors: usize,
    pub estimated_peak_memory_bytes: u64,
    pub algorithm: String,
    pub seed_source: String,
    pub metric_validity_evidence: String,
    pub preconditioner: Option<PreconditionerDescriptorF64>,
    pub status: ResultStatus,
    pub termination: TerminationReason,
    pub assurance: AssuranceLevel,
    pub provenance: SolverProvenance,
}

#[derive(Clone, Debug)]
struct GeneralizedIterateF64 {
    vector: Vec<f64>,
    applied_operator: Vec<f64>,
    applied_metric: Vec<f64>,
}

fn combine_vectors(coefficients: &[f64], vectors: &[&[f64]]) -> Vec<f64> {
    let mut output = vec![0.0; vectors[0].len()];
    for (coefficient, vector) in coefficients.iter().zip(vectors) {
        for (value, component) in output.iter_mut().zip(*vector) {
            *value += coefficient * component;
        }
    }
    output
}

fn normalize_generalized_iterate(iterate: &mut GeneralizedIterateF64) -> Result<(), SolverError> {
    let metric_norm_sq = dot(&iterate.vector, &iterate.applied_metric);
    if !metric_norm_sq.is_finite() || metric_norm_sq <= f64::MIN_POSITIVE {
        return Err(SolverError::NumericalBreakdown(
            "generalized iterate has nonpositive or non-finite metric norm".to_owned(),
        ));
    }
    let scale = metric_norm_sq.sqrt();
    for ((value, applied_operator), applied_metric) in iterate
        .vector
        .iter_mut()
        .zip(&mut iterate.applied_operator)
        .zip(&mut iterate.applied_metric)
    {
        *value /= scale;
        *applied_operator /= scale;
        *applied_metric /= scale;
    }
    if canonicalize_vector_sign(&mut iterate.vector) {
        for value in &mut iterate.applied_operator {
            *value = -*value;
        }
        for value in &mut iterate.applied_metric {
            *value = -*value;
        }
    }
    Ok(())
}

fn projected_generalized_extreme(
    projected_operator: DMatrix<f64>,
    projected_metric: DMatrix<f64>,
    largest: bool,
) -> Result<Vec<f64>, SolverError> {
    use nalgebra::Cholesky;

    let cholesky = Cholesky::new(projected_metric.clone()).ok_or_else(|| {
        SolverError::NumericalBreakdown(
            "projected generalized metric is not numerically positive definite".to_owned(),
        )
    })?;
    let inverse_lower = cholesky.l().try_inverse().ok_or_else(|| {
        SolverError::NumericalBreakdown(
            "failed to invert projected metric Cholesky factor".to_owned(),
        )
    })?;
    let whitened = &inverse_lower * projected_operator * inverse_lower.transpose();
    let decomposition = SymmetricEigen::new(whitened);
    let index = if largest {
        (0..decomposition.eigenvalues.len())
            .max_by(|left, right| {
                decomposition.eigenvalues[*left].total_cmp(&decomposition.eigenvalues[*right])
            })
            .expect("projected dimension is positive")
    } else {
        (0..decomposition.eigenvalues.len())
            .min_by(|left, right| {
                decomposition.eigenvalues[*left].total_cmp(&decomposition.eigenvalues[*right])
            })
            .expect("projected dimension is positive")
    };
    let whitened_vector = decomposition.eigenvectors.column(index).into_owned();
    let coefficients = inverse_lower.transpose() * whitened_vector;
    Ok(coefficients.iter().copied().collect())
}

/// Single-vector locally optimal generalized eigensolver. Its three-vector
/// trial subspace is maintained in the metric inner product, and every large
/// operation is an operator, metric, or optional preconditioner application.
#[derive(Clone, Debug, Default)]
pub struct MatrixFreeLobpcgF64;

impl MatrixFreeLobpcgF64 {
    pub fn solve(
        &self,
        problem: &GeneralizedEigenProblem<'_, f64>,
        config: &GeneralizedExtremeConfigF64,
    ) -> Result<MatrixFreeGeneralizedEigenpairReportF64, SolverError> {
        self.solve_controlled(problem, config, None, None, &CancellationToken::new())
    }

    pub fn solve_with_initial_vector(
        &self,
        problem: &GeneralizedEigenProblem<'_, f64>,
        config: &GeneralizedExtremeConfigF64,
        initial_vector: &[f64],
    ) -> Result<MatrixFreeGeneralizedEigenpairReportF64, SolverError> {
        self.solve_controlled(
            problem,
            config,
            Some(initial_vector),
            None,
            &CancellationToken::new(),
        )
    }

    pub fn solve_controlled(
        &self,
        problem: &GeneralizedEigenProblem<'_, f64>,
        config: &GeneralizedExtremeConfigF64,
        initial_vector: Option<&[f64]>,
        preconditioner: Option<&dyn GeneralizedPreconditionerF64>,
        cancellation: &CancellationToken,
    ) -> Result<MatrixFreeGeneralizedEigenpairReportF64, SolverError> {
        check_solver_cancellation(cancellation)?;
        config.validate()?;
        let dimension = problem.operator.dimension();
        if dimension == 0 || problem.metric.dimension() != dimension {
            return Err(SolverError::InvalidConfiguration(
                "generalized operator and metric require the same positive dimension".to_owned(),
            ));
        }
        let preconditioner_descriptor = preconditioner.map(|value| value.descriptor());
        if let Some(descriptor) = &preconditioner_descriptor {
            descriptor.validate()?;
        }
        let seed_source = if initial_vector.is_some() {
            "caller_warm_start"
        } else {
            "deterministic_reference_seed"
        };
        let vector = initial_vector
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| deterministic_seed(dimension));
        if vector.len() != dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "initial vector has dimension {}, expected {dimension}",
                vector.len()
            )));
        }
        let mut applied_metric = vec![0.0; dimension];
        problem.metric.apply(&vector, &mut applied_metric)?;
        let mut applied_operator = vec![0.0; dimension];
        problem.operator.apply(&vector, &mut applied_operator)?;
        let mut current = GeneralizedIterateF64 {
            vector,
            applied_operator,
            applied_metric,
        };
        normalize_generalized_iterate(&mut current)?;
        let mut direction: Option<GeneralizedIterateF64> = None;
        let mut previous_value: Option<f64> = None;
        let mut last_projected_dimension = 0usize;
        for iteration in 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            let denominator = dot(&current.vector, &current.applied_metric);
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(SolverError::NumericalBreakdown(
                    "generalized Rayleigh denominator is not positive".to_owned(),
                ));
            }
            let eigenvalue = dot(&current.vector, &current.applied_operator) / denominator;
            let residual: Vec<f64> = current
                .applied_operator
                .iter()
                .zip(&current.applied_metric)
                .map(|(operator_value, metric_value)| operator_value - eigenvalue * metric_value)
                .collect();
            let residual_norm = norm(&residual);
            let applied_operator_norm = norm(&current.applied_operator);
            let applied_metric_norm = norm(&current.applied_metric);
            let relative_residual = residual_norm
                / (applied_operator_norm + eigenvalue.abs() * applied_metric_norm)
                    .max(f64::MIN_POSITIVE);
            let scaled_backward_error = relative_residual;
            let metric_normalization_error = (denominator - 1.0).abs();
            let stability = previous_value
                .map(|previous| {
                    (eigenvalue - previous).abs() / eigenvalue.abs().max(previous.abs()).max(1.0)
                })
                .unwrap_or(f64::INFINITY);
            let converged = iteration >= config.minimum_iterations
                && (residual_norm <= config.absolute_residual_tolerance
                    || scaled_backward_error <= config.scaled_backward_error_tolerance)
                && (stability <= config.ritz_value_stability_tolerance
                    || last_projected_dimension == dimension);
            if converged || iteration == config.maximum_iterations {
                let (status, termination) = if converged {
                    if scaled_backward_error <= config.scaled_backward_error_tolerance {
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
                return Ok(MatrixFreeGeneralizedEigenpairReportF64 {
                    target: config.target.clone(),
                    eigenvalue,
                    eigenvector: current.vector,
                    residual_norm,
                    relative_residual,
                    scaled_backward_error,
                    metric_normalization_error,
                    ritz_value_stability: stability,
                    target_ordering_established_by_full_space_projection: last_projected_dimension
                        == dimension,
                    iterations: iteration,
                    operator_applications: iteration,
                    metric_applications: iteration,
                    projected_factorizations: iteration - 1,
                    preconditioner_applications: if preconditioner.is_some() {
                        iteration - 1
                    } else {
                        0
                    },
                    retained_subspace_vectors: if direction.is_some() { 3 } else { 2 },
                    estimated_peak_memory_bytes: (20u64)
                        .saturating_mul(dimension as u64)
                        .saturating_mul(8),
                    algorithm: "matrix_free_lobpcg_single_f64".to_owned(),
                    seed_source: seed_source.to_owned(),
                    metric_validity_evidence:
                        "positive_definite_metric_trait_and_projected_cholesky".to_owned(),
                    preconditioner: preconditioner_descriptor,
                    status,
                    termination,
                    assurance: AssuranceLevel::Computed,
                    provenance: SolverProvenance::current_package("f64"),
                });
            }

            let mut search_vector = vec![0.0; dimension];
            if let Some(preconditioner) = preconditioner {
                preconditioner.apply(&residual, &mut search_vector)?;
            } else {
                search_vector.clone_from(&residual);
            }
            if search_vector.iter().any(|value| !value.is_finite()) {
                return Err(SolverError::NumericalBreakdown(
                    "preconditioned residual contains non-finite values".to_owned(),
                ));
            }
            let mut search_metric = vec![0.0; dimension];
            problem.metric.apply(&search_vector, &mut search_metric)?;
            let unprojected_search_metric_norm_sq = dot(&search_vector, &search_metric);
            if !unprojected_search_metric_norm_sq.is_finite()
                || unprojected_search_metric_norm_sq <= f64::MIN_POSITIVE
            {
                return Err(SolverError::NumericalBreakdown(
                    "preconditioned residual has nonpositive or non-finite metric norm".to_owned(),
                ));
            }

            for _ in 0..2 {
                let projection_on_current = dot(&current.vector, &search_metric);
                for index in 0..dimension {
                    search_vector[index] -= projection_on_current * current.vector[index];
                    search_metric[index] -= projection_on_current * current.applied_metric[index];
                }
                if let Some(previous_direction) = &direction {
                    let projection = dot(&previous_direction.vector, &search_metric);
                    for index in 0..dimension {
                        search_vector[index] -= projection * previous_direction.vector[index];
                        search_metric[index] -=
                            projection * previous_direction.applied_metric[index];
                    }
                }
            }
            let search_metric_norm_sq = dot(&search_vector, &search_metric);
            let rank_threshold =
                unprojected_search_metric_norm_sq * (256.0 * f64::EPSILON) * (256.0 * f64::EPSILON);
            if !search_metric_norm_sq.is_finite()
                || search_metric_norm_sq <= rank_threshold.max(f64::MIN_POSITIVE)
            {
                return Err(SolverError::NumericalBreakdown(
                    format!(
                        "LOBPCG residual search direction lost metric rank before convergence at iteration {iteration}: residual={residual_norm:.3e}, backward={scaled_backward_error:.3e}, stability={stability:.3e}, search_metric_norm_sq={search_metric_norm_sq:.3e}"
                    ),
                ));
            }
            let search_scale = search_metric_norm_sq.sqrt();
            for (value, metric_value) in search_vector.iter_mut().zip(&mut search_metric) {
                *value /= search_scale;
                *metric_value /= search_scale;
            }
            let mut search_operator = vec![0.0; dimension];
            problem
                .operator
                .apply(&search_vector, &mut search_operator)?;
            let search = GeneralizedIterateF64 {
                vector: search_vector,
                applied_operator: search_operator,
                applied_metric: search_metric,
            };

            let mut vectors = vec![current.vector.as_slice(), search.vector.as_slice()];
            let mut applied_operators = vec![
                current.applied_operator.as_slice(),
                search.applied_operator.as_slice(),
            ];
            let mut applied_metrics = vec![
                current.applied_metric.as_slice(),
                search.applied_metric.as_slice(),
            ];
            if let Some(previous_direction) = &direction {
                vectors.push(previous_direction.vector.as_slice());
                applied_operators.push(previous_direction.applied_operator.as_slice());
                applied_metrics.push(previous_direction.applied_metric.as_slice());
            }
            let subspace_dimension = vectors.len();
            let mut projected_operator =
                DMatrix::<f64>::zeros(subspace_dimension, subspace_dimension);
            let mut projected_metric =
                DMatrix::<f64>::zeros(subspace_dimension, subspace_dimension);
            for row in 0..subspace_dimension {
                for column in 0..=row {
                    let operator_value = 0.5
                        * (dot(vectors[row], applied_operators[column])
                            + dot(vectors[column], applied_operators[row]));
                    let metric_value = 0.5
                        * (dot(vectors[row], applied_metrics[column])
                            + dot(vectors[column], applied_metrics[row]));
                    projected_operator[(row, column)] = operator_value;
                    projected_operator[(column, row)] = operator_value;
                    projected_metric[(row, column)] = metric_value;
                    projected_metric[(column, row)] = metric_value;
                }
            }
            let coefficients = projected_generalized_extreme(
                projected_operator,
                projected_metric,
                config.target == EigenTarget::AlgebraicLargest,
            )?;
            last_projected_dimension = subspace_dimension;
            let mut next = GeneralizedIterateF64 {
                vector: combine_vectors(&coefficients, &vectors),
                applied_operator: combine_vectors(&coefficients, &applied_operators),
                applied_metric: combine_vectors(&coefficients, &applied_metrics),
            };
            normalize_generalized_iterate(&mut next)?;

            let direction_coefficients = &coefficients[1..];
            let direction_vectors = &vectors[1..];
            let direction_applied_operators = &applied_operators[1..];
            let direction_applied_metrics = &applied_metrics[1..];
            let mut next_direction = GeneralizedIterateF64 {
                vector: combine_vectors(direction_coefficients, direction_vectors),
                applied_operator: combine_vectors(
                    direction_coefficients,
                    direction_applied_operators,
                ),
                applied_metric: combine_vectors(direction_coefficients, direction_applied_metrics),
            };
            let projection = dot(&next.vector, &next_direction.applied_metric);
            for index in 0..dimension {
                next_direction.vector[index] -= projection * next.vector[index];
                next_direction.applied_operator[index] -= projection * next.applied_operator[index];
                next_direction.applied_metric[index] -= projection * next.applied_metric[index];
            }
            let direction_norm_sq = dot(&next_direction.vector, &next_direction.applied_metric);
            direction = if direction_norm_sq.is_finite() && direction_norm_sq > 256.0 * f64::EPSILON
            {
                normalize_generalized_iterate(&mut next_direction)?;
                Some(next_direction)
            } else {
                None
            };
            previous_value = Some(eigenvalue);
            current = next;
        }

        Err(SolverError::NonConvergence(
            "matrix-free LOBPCG exhausted its iteration loop".to_owned(),
        ))
    }
}

/// Validation-scale dense generalized symmetric problem `A x = lambda B x`
/// with a symmetric positive-definite metric `B`.
pub struct DenseGeneralizedProblemF64<'a> {
    pub operator: &'a [f64],
    pub metric: &'a [f64],
    pub dimension: usize,
}

impl<'a> DenseGeneralizedProblemF64<'a> {
    pub fn new(
        operator: &'a [f64],
        metric: &'a [f64],
        dimension: usize,
        symmetry_tolerance: f64,
    ) -> Result<Self, SolverError> {
        let expected = dimension.saturating_mul(dimension);
        if dimension == 0 || operator.len() != expected || metric.len() != expected {
            return Err(SolverError::InvalidConfiguration(format!(
                "generalized dense matrices must both contain {expected} entries"
            )));
        }
        if !symmetry_tolerance.is_finite() || symmetry_tolerance < 0.0 {
            return Err(SolverError::InvalidConfiguration(
                "symmetry tolerance must be finite and nonnegative".to_owned(),
            ));
        }
        if operator
            .iter()
            .chain(metric)
            .any(|value| !value.is_finite())
        {
            return Err(SolverError::InvalidConfiguration(
                "generalized dense entries must be finite".to_owned(),
            ));
        }
        for row in 0..dimension {
            for column in 0..row {
                for (name, matrix) in [("operator", operator), ("metric", metric)] {
                    if (matrix[row * dimension + column] - matrix[column * dimension + row]).abs()
                        > symmetry_tolerance
                    {
                        return Err(SolverError::InvalidConfiguration(format!(
                            "{name} is not symmetric at ({row}, {column})"
                        )));
                    }
                }
            }
        }
        Ok(Self {
            operator,
            metric,
            dimension,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DenseGeneralizedReferenceSolverF64 {
    pub maximum_dimension: usize,
}

impl Default for DenseGeneralizedReferenceSolverF64 {
    fn default() -> Self {
        Self {
            maximum_dimension: 2048,
        }
    }
}

impl DenseGeneralizedReferenceSolverF64 {
    pub fn solve(
        &self,
        problem: &DenseGeneralizedProblemF64<'_>,
        target: &EigenTarget,
    ) -> Result<GeneralizedEigenpairReportF64, SolverError> {
        self.solve_controlled(problem, target, &CancellationToken::new())
    }

    pub fn solve_controlled(
        &self,
        problem: &DenseGeneralizedProblemF64<'_>,
        target: &EigenTarget,
        cancellation: &CancellationToken,
    ) -> Result<GeneralizedEigenpairReportF64, SolverError> {
        use nalgebra::Cholesky;

        check_solver_cancellation(cancellation)?;
        let largest = supported_extreme(target)?;
        let n = problem.dimension;
        if n > self.maximum_dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "generalized dense dimension {n} exceeds limit {}",
                self.maximum_dimension
            )));
        }
        let operator = DMatrix::from_row_slice(n, n, problem.operator);
        let metric = DMatrix::from_row_slice(n, n, problem.metric);
        let cholesky = Cholesky::new(metric.clone()).ok_or_else(|| {
            SolverError::NumericalBreakdown(
                "generalized metric is not numerically positive definite".to_owned(),
            )
        })?;
        let lower = cholesky.l();
        let inverse_lower = lower.try_inverse().ok_or_else(|| {
            SolverError::NumericalBreakdown(
                "failed to invert generalized metric Cholesky factor".to_owned(),
            )
        })?;
        check_solver_cancellation(cancellation)?;
        let whitened = &inverse_lower * operator.clone() * inverse_lower.transpose();
        let decomposition = SymmetricEigen::new(whitened);
        check_solver_cancellation(cancellation)?;
        let index = if largest {
            (0..n)
                .max_by(|&left, &right| {
                    decomposition.eigenvalues[left].total_cmp(&decomposition.eigenvalues[right])
                })
                .expect("dimension is positive")
        } else {
            (0..n)
                .min_by(|&left, &right| {
                    decomposition.eigenvalues[left].total_cmp(&decomposition.eigenvalues[right])
                })
                .expect("dimension is positive")
        };
        let y = decomposition.eigenvectors.column(index).into_owned();
        let mut x = inverse_lower.transpose() * y;
        let metric_norm_sq = (x.transpose() * &metric * &x)[(0, 0)];
        if !metric_norm_sq.is_finite() || metric_norm_sq <= 0.0 {
            return Err(SolverError::NumericalBreakdown(
                "computed generalized vector has invalid metric norm".to_owned(),
            ));
        }
        x /= metric_norm_sq.sqrt();
        let ax = &operator * &x;
        let bx = &metric * &x;
        let denominator = (x.transpose() * &bx)[(0, 0)];
        let eigenvalue = (x.transpose() * &ax)[(0, 0)] / denominator;
        let residual = ax - bx * eigenvalue;
        let residual_norm = residual.norm();
        let metric_norm_error = (denominator - 1.0).abs();
        Ok(GeneralizedEigenpairReportF64 {
            eigenvalue,
            eigenvector: x.iter().copied().collect(),
            residual_norm,
            metric_norm_error,
            algorithm: "dense_cholesky_whitened_generalized_reference_f64".to_owned(),
            status: ResultStatus::Converged,
            assurance: AssuranceLevel::Computed,
            provenance: SolverProvenance::current_package("f64"),
        })
    }
}

#[cfg(test)]
mod generalized_reference_tests {
    use super::*;
    use xc_operator::{
        DenseSymmetricF64, DiagonalF64, LinearOperator, OperatorMetadata, PositiveDefiniteMetric,
        SymmetricOperator,
    };

    struct PositiveDiagonalMetricF64(DiagonalF64);

    impl PositiveDiagonalMetricF64 {
        fn new(diagonal: Vec<f64>) -> Self {
            assert!(diagonal.iter().all(|value| *value > 0.0));
            Self(DiagonalF64::new("positive_diagonal_metric", diagonal).unwrap())
        }
    }

    impl LinearOperator<f64> for PositiveDiagonalMetricF64 {
        fn dimension(&self) -> usize {
            self.0.dimension()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
            self.0.apply(x, y)
        }

        fn metadata(&self) -> OperatorMetadata {
            self.0.metadata()
        }

        fn norm_bound(&self) -> Option<f64> {
            self.0.norm_bound()
        }
    }

    impl SymmetricOperator<f64> for PositiveDiagonalMetricF64 {}
    impl PositiveDefiniteMetric<f64> for PositiveDiagonalMetricF64 {}

    struct IdentityPreconditionerF64;

    impl GeneralizedPreconditionerF64 for IdentityPreconditionerF64 {
        fn descriptor(&self) -> PreconditionerDescriptorF64 {
            PreconditionerDescriptorF64 {
                id: "identity_test_preconditioner".to_owned(),
                changes_only_convergence: true,
                approximation_error_bound: Some(0.0),
            }
        }

        fn apply(&self, residual: &[f64], output: &mut [f64]) -> Result<(), SolverError> {
            output.clone_from_slice(residual);
            Ok(())
        }
    }

    fn generalized_config(target: EigenTarget) -> GeneralizedExtremeConfigF64 {
        GeneralizedExtremeConfigF64 {
            target,
            absolute_residual_tolerance: 1e-11,
            scaled_backward_error_tolerance: 1e-11,
            ritz_value_stability_tolerance: 1e-13,
            maximum_iterations: 50,
            minimum_iterations: 2,
        }
    }

    #[test]
    fn generalized_dense_solver_whitens_spd_metric() {
        let operator = [2.0, 0.0, 0.0, 6.0];
        let metric = [1.0, 0.0, 0.0, 2.0];
        let problem = DenseGeneralizedProblemF64::new(&operator, &metric, 2, 0.0).unwrap();
        let solver = DenseGeneralizedReferenceSolverF64::default();
        let largest = solver
            .solve(&problem, &EigenTarget::AlgebraicLargest)
            .unwrap();
        assert!((largest.eigenvalue - 3.0).abs() < 1e-12);
        assert!(largest.residual_norm < 1e-12);
        let smallest = solver
            .solve(&problem, &EigenTarget::AlgebraicSmallest)
            .unwrap();
        assert!((smallest.eigenvalue - 2.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_free_lobpcg_matches_dense_generalized_reference() {
        let operator_data = [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0];
        let metric_data = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let operator =
            DenseSymmetricF64::new("generalized_operator", 3, operator_data.to_vec(), 0.0).unwrap();
        let metric = PositiveDiagonalMetricF64::new(vec![1.0, 2.0, 3.0]);
        let matrix_free_problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let dense_problem =
            DenseGeneralizedProblemF64::new(&operator_data, &metric_data, 3, 0.0).unwrap();

        for target in [
            EigenTarget::AlgebraicLargest,
            EigenTarget::AlgebraicSmallest,
        ] {
            let matrix_free = MatrixFreeLobpcgF64
                .solve(&matrix_free_problem, &generalized_config(target.clone()))
                .unwrap();
            let dense = DenseGeneralizedReferenceSolverF64::default()
                .solve(&dense_problem, &target)
                .unwrap();
            assert_eq!(matrix_free.status, ResultStatus::Converged);
            assert!((matrix_free.eigenvalue - dense.eigenvalue).abs() < 1e-10);
            assert!(matrix_free.residual_norm < 1e-10);
            assert!(matrix_free.metric_normalization_error < 1e-12);
            assert_eq!(matrix_free.operator_applications, matrix_free.iterations);
            assert_eq!(matrix_free.metric_applications, matrix_free.iterations);
            assert!(matrix_free.projected_factorizations < matrix_free.iterations);
        }
    }

    #[test]
    fn matrix_free_lobpcg_converges_without_full_space_projection() {
        let operator = DenseSymmetricF64::new(
            "diagonal_generalized_operator",
            6,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 36.0,
            ],
            0.0,
        )
        .unwrap();
        let metric = PositiveDiagonalMetricF64::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let mut config = generalized_config(EigenTarget::AlgebraicLargest);
        config.absolute_residual_tolerance = 1e-9;
        config.scaled_backward_error_tolerance = 1e-10;
        config.ritz_value_stability_tolerance = 1e-11;
        config.maximum_iterations = 200;
        let report = MatrixFreeLobpcgF64
            .solve_controlled(
                &problem,
                &config,
                None,
                Some(&IdentityPreconditionerF64),
                &CancellationToken::new(),
            )
            .unwrap();

        assert_eq!(report.status, ResultStatus::Converged);
        assert!((report.eigenvalue - 6.0).abs() < 1e-9);
        assert!(report.residual_norm < 1e-8);
        assert!(!report.target_ordering_established_by_full_space_projection);
        assert!(report.ritz_value_stability <= config.ritz_value_stability_tolerance);
        assert_eq!(report.preconditioner_applications, report.iterations - 1);
        assert_eq!(
            report.preconditioner.unwrap().id,
            "identity_test_preconditioner"
        );
    }

    #[test]
    fn preconditioner_descriptor_rejects_hidden_approximation_error() {
        let descriptor = PreconditionerDescriptorF64 {
            id: "invalid".to_owned(),
            changes_only_convergence: true,
            approximation_error_bound: Some(1e-3),
        };
        assert!(descriptor.validate().is_err());
        let unbounded = PreconditionerDescriptorF64 {
            id: "unbounded".to_owned(),
            changes_only_convergence: false,
            approximation_error_bound: None,
        };
        assert!(unbounded.validate().is_err());
    }
}

// ===========================================================================
// Typed solver planning
// ===========================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverRoute {
    DenseFullSpectrumReference,
    TridiagonalFullSpectrumReference,
    TridiagonalSturmSelected,
    ShiftedPowerExtremeReference,
    LanczosExtremeReference,
    BlockSubspaceExtremeReference,
    DenseGeneralizedWhiteningReference,
    MatrixFreeGeneralizedLobpcg,
    HpDenseReference,
    HpTridiagonalFullSpectrumReference,
    HpTridiagonalSturmSelected,
    HpMatrixFreeGeneralizedRayleighRitz,
    HpBlockGeneralizedLobpcg,
    HpBlockShiftInvert,
    HpThickRestartLanczos,
    HpDenseGeneralizedWhiteningReference,
    HpSelectedSpectrumPlanned,
    CertifiedInertiaPlanned,
}

impl SolverRoute {
    pub fn id(self) -> &'static str {
        match self {
            Self::DenseFullSpectrumReference => "dense_full_spectrum_reference",
            Self::TridiagonalFullSpectrumReference => "tridiagonal_full_spectrum_reference",
            Self::TridiagonalSturmSelected => "tridiagonal_sturm_selected",
            Self::ShiftedPowerExtremeReference => "shifted_power_extreme_reference",
            Self::LanczosExtremeReference => "lanczos_extreme_reference",
            Self::BlockSubspaceExtremeReference => "block_subspace_extreme_reference",
            Self::DenseGeneralizedWhiteningReference => "dense_generalized_whitening_reference",
            Self::MatrixFreeGeneralizedLobpcg => "matrix_free_generalized_lobpcg",
            Self::HpDenseReference => "hp_dense_reference",
            Self::HpTridiagonalFullSpectrumReference => "hp_tridiagonal_full_spectrum_reference",
            Self::HpTridiagonalSturmSelected => "hp_tridiagonal_sturm_selected",
            Self::HpMatrixFreeGeneralizedRayleighRitz => "hp_matrix_free_generalized_rayleigh_ritz",
            Self::HpBlockGeneralizedLobpcg => "hp_block_generalized_lobpcg",
            Self::HpBlockShiftInvert => "hp_block_shift_invert",
            Self::HpThickRestartLanczos => "hp_thick_restart_lanczos",
            Self::HpDenseGeneralizedWhiteningReference => {
                "hp_dense_generalized_whitening_reference"
            }
            Self::HpSelectedSpectrumPlanned => "hp_selected_spectrum",
            Self::CertifiedInertiaPlanned => "certified_inertia",
        }
    }

    pub fn algorithm_family(self) -> &'static str {
        match self {
            Self::DenseFullSpectrumReference => "dense_symmetric_eigendecomposition",
            Self::TridiagonalFullSpectrumReference => "tridiagonal_ql",
            Self::TridiagonalSturmSelected => "sturm_bisection",
            Self::ShiftedPowerExtremeReference => "shifted_power_iteration",
            Self::LanczosExtremeReference => "lanczos_iteration",
            Self::BlockSubspaceExtremeReference => "block_subspace_iteration",
            Self::DenseGeneralizedWhiteningReference => "cholesky_whitening",
            Self::MatrixFreeGeneralizedLobpcg => "lobpcg",
            Self::HpDenseReference => "hp_dense_symmetric_eigendecomposition",
            Self::HpTridiagonalFullSpectrumReference => "hp_tridiagonal_qr",
            Self::HpTridiagonalSturmSelected => "hp_sturm_bisection",
            Self::HpMatrixFreeGeneralizedRayleighRitz => "hp_b_orthogonal_rayleigh_ritz",
            Self::HpBlockGeneralizedLobpcg => "hp_block_b_orthogonal_lobpcg",
            Self::HpBlockShiftInvert => "hp_block_shift_invert_iteration",
            Self::HpThickRestartLanczos => "hp_thick_restart_lanczos",
            Self::HpDenseGeneralizedWhiteningReference => "hp_cholesky_whitening_householder_qr",
            Self::HpSelectedSpectrumPlanned => "hp_selected_spectrum",
            Self::CertifiedInertiaPlanned => "interval_inertia",
        }
    }

    pub fn formulation(self) -> &'static str {
        match self {
            Self::DenseFullSpectrumReference
            | Self::TridiagonalFullSpectrumReference
            | Self::HpDenseReference
            | Self::HpTridiagonalFullSpectrumReference => "full_spectrum_diagonalization",
            Self::TridiagonalSturmSelected
            | Self::HpTridiagonalSturmSelected
            | Self::CertifiedInertiaPlanned => "threshold_inertia_count",
            Self::ShiftedPowerExtremeReference | Self::LanczosExtremeReference => {
                "extreme_ritz_pair"
            }
            Self::BlockSubspaceExtremeReference => "selected_extreme_invariant_subspaces",
            Self::DenseGeneralizedWhiteningReference => "whitened_generalized_eigenproblem",
            Self::MatrixFreeGeneralizedLobpcg => "metric_orthogonal_generalized_ritz_pair",
            Self::HpMatrixFreeGeneralizedRayleighRitz => {
                "hp_metric_orthogonal_generalized_ritz_pair"
            }
            Self::HpBlockGeneralizedLobpcg => {
                "hp_block_metric_orthogonal_generalized_ritz_subspace"
            }
            Self::HpBlockShiftInvert => "hp_selected_interior_shifted_inverse_subspace",
            Self::HpThickRestartLanczos => "hp_selected_extreme_thick_restart_krylov",
            Self::HpDenseGeneralizedWhiteningReference => "hp_whitened_generalized_full_spectrum",
            Self::HpSelectedSpectrumPlanned => "selected_spectrum_transform",
        }
    }

    pub fn evidence(self, precision_bits: u32, thread_count: Option<usize>) -> RouteEvidence {
        RouteEvidence {
            route_id: self.id().to_owned(),
            algorithm_family: self.algorithm_family().to_owned(),
            formulation: self.formulation().to_owned(),
            implementation_id: format!("xc-solver@{}:{}", env!("CARGO_PKG_VERSION"), self.id()),
            decisive_intermediates: BTreeSet::new(),
            precision_bits: Some(precision_bits),
            seed: None,
            thread_count,
            evidence_digest: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SolverPlan {
    pub primary: SolverRoute,
    pub independent_crosscheck: Option<SolverRoute>,
    pub requested_assurance: AssuranceLevel,
    pub precision_schedule_bits: Vec<u32>,
    pub requires_materialization: bool,
    pub requires_factorization: bool,
    pub resource_estimate: ResourceEstimate,
    pub expected_cached_artifacts: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SolverPlannerInput {
    pub structure: xc_operator::MatrixStructure,
    pub dimension: usize,
    pub target: EigenTarget,
    pub requested_eigenpairs: usize,
    pub assurance: AssuranceLevel,
    pub precision: xc_core::PrecisionPolicy,
    pub matrix_materialized: bool,
    pub generalized: bool,
}

impl SolverPlannerInput {
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.dimension == 0 {
            return Err(SolverError::InvalidConfiguration(
                "solver planner dimension must be positive".to_owned(),
            ));
        }
        if self.requested_eigenpairs == 0 || self.requested_eigenpairs > self.dimension {
            return Err(SolverError::InvalidConfiguration(format!(
                "requested_eigenpairs must be in 1..={}",
                self.dimension
            )));
        }
        self.target
            .validate()
            .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
        if let EigenTarget::IndexRange { first, last } = &self.target {
            if *last >= self.dimension {
                return Err(SolverError::InvalidConfiguration(format!(
                    "index range ends at {last}, outside dimension {}",
                    self.dimension
                )));
            }
            let range_count = last - first + 1;
            if self.requested_eigenpairs != range_count {
                return Err(SolverError::InvalidConfiguration(format!(
                    "requested_eigenpairs={} does not match index-range cardinality {range_count}",
                    self.requested_eigenpairs
                )));
            }
        }
        if self.generalized && self.requested_eigenpairs != 1 && self.precision.initial_bits <= 64 {
            return Err(SolverError::UnsupportedTarget(
                "installed f64 generalized routes currently accept exactly one algebraic extreme"
                    .to_owned(),
            ));
        }
        self.precision
            .validate()
            .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
        Ok(())
    }
}

/// Public adapter used by research domains to translate their own request
/// language into the domain-neutral solver capability request.
pub trait DomainSolverPlanner {
    type Request;

    fn domain_id(&self) -> &'static str;
    fn solver_input(&self, request: &Self::Request) -> Result<SolverPlannerInput, SolverError>;
    fn planning_rationale(&self, request: &Self::Request) -> Vec<String>;

    fn plan(&self, request: &Self::Request) -> Result<DomainSolverPlan, SolverError> {
        let domain_id = self.domain_id();
        if domain_id.trim().is_empty() {
            return Err(SolverError::InvalidConfiguration(
                "domain solver planner identity must be nonempty".to_owned(),
            ));
        }
        let input = self.solver_input(request)?;
        let rationale = self.planning_rationale(request);
        if rationale.is_empty() || rationale.iter().any(|entry| entry.trim().is_empty()) {
            return Err(SolverError::InvalidConfiguration(
                "domain solver planner must provide a nonempty rationale".to_owned(),
            ));
        }
        let solver_plan = plan_symmetric_eigenproblem(&input)?;
        Ok(DomainSolverPlan {
            domain_id: domain_id.to_owned(),
            input,
            solver_plan,
            rationale,
        })
    }
}

/// Persistable result of a domain adapter invoking the common planner.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DomainSolverPlan {
    pub domain_id: String,
    pub input: SolverPlannerInput,
    pub solver_plan: SolverPlan,
    pub rationale: Vec<String>,
}

/// Builds a transparent capability-based symmetric-eigenproblem plan.
///
/// # Mathematical semantics
/// Preserves the requested operator structure, spectral target, eigenpair
/// count, precision, and assurance while selecting a compatible solver route.
///
/// # Precision
/// The precision policy is part of the input and output plan. HP requests are
/// never rewritten as binary64 requests.
///
/// # Failure states
/// Invalid dimensions, targets, or precision policies return `SolverError`.
/// Routes marked `Planned` remain serializable but execution must reject an
/// unavailable capability rather than silently substituting another route.
///
/// # Assurance and validity
/// Planning proves compatibility, not a numerical result. The plan names the
/// cross-check or certification work required by the requested assurance.
///
/// # Cache effects
/// Planning has no cache side effects. Artifact reuse is decided later through
/// the common typed artifact plan and recorded in result provenance.
///
/// # Example
/// Compiled example: `crates/xc-solver/examples/plan.rs`.
pub fn plan_symmetric_eigenproblem(input: &SolverPlannerInput) -> Result<SolverPlan, SolverError> {
    input.validate()?;
    let hp_requested = input.precision.initial_bits > 64;
    let selected_target = input.requested_eigenpairs > 1
        || matches!(
            &input.target,
            EigenTarget::SmallestMagnitude
                | EigenTarget::ClosestTo { .. }
                | EigenTarget::IndexRange { .. }
                | EigenTarget::Interval { .. }
        );
    let algebraic_extreme = matches!(
        &input.target,
        EigenTarget::AlgebraicLargest | EigenTarget::AlgebraicSmallest
    );
    let interior_target = matches!(
        &input.target,
        EigenTarget::SmallestMagnitude
            | EigenTarget::ClosestTo { .. }
            | EigenTarget::Interval { .. }
    );
    let hp_tridiagonal_index_target =
        algebraic_extreme || matches!(&input.target, EigenTarget::IndexRange { .. });
    if input.generalized && !algebraic_extreme {
        return Err(SolverError::UnsupportedTarget(
            "the generalized solver routes currently support algebraic extremes only".to_owned(),
        ));
    }

    let primary = if input.assurance == AssuranceLevel::Certified {
        SolverRoute::CertifiedInertiaPlanned
    } else if input.generalized {
        if hp_requested {
            if input.requested_eigenpairs > 1 {
                SolverRoute::HpBlockGeneralizedLobpcg
            } else {
                SolverRoute::HpMatrixFreeGeneralizedRayleighRitz
            }
        } else if input.matrix_materialized {
            SolverRoute::DenseGeneralizedWhiteningReference
        } else {
            SolverRoute::MatrixFreeGeneralizedLobpcg
        }
    } else {
        match (&input.structure, hp_requested, selected_target) {
            (xc_operator::MatrixStructure::Tridiagonal, true, _) if hp_tridiagonal_index_target => {
                SolverRoute::HpTridiagonalSturmSelected
            }
            (xc_operator::MatrixStructure::Tridiagonal, false, true) => {
                SolverRoute::TridiagonalSturmSelected
            }
            (xc_operator::MatrixStructure::Tridiagonal, false, false) => {
                SolverRoute::TridiagonalFullSpectrumReference
            }
            (_, true, _) if interior_target => SolverRoute::HpBlockShiftInvert,
            (_, true, _) if algebraic_extreme && input.requested_eigenpairs < input.dimension => {
                SolverRoute::HpThickRestartLanczos
            }
            (xc_operator::MatrixStructure::Dense, true, _) if input.matrix_materialized => {
                SolverRoute::HpDenseReference
            }
            (_, true, _) => SolverRoute::HpSelectedSpectrumPlanned,
            (xc_operator::MatrixStructure::Dense, false, _) if input.matrix_materialized => {
                SolverRoute::DenseFullSpectrumReference
            }
            (_, false, _) if input.requested_eigenpairs > 1 && algebraic_extreme => {
                SolverRoute::BlockSubspaceExtremeReference
            }
            (_, false, false) => SolverRoute::LanczosExtremeReference,
            (_, false, true) => SolverRoute::LanczosExtremeReference,
        }
    };

    let crosscheck_candidate = match primary {
        SolverRoute::DenseFullSpectrumReference => Some(SolverRoute::LanczosExtremeReference),
        SolverRoute::TridiagonalFullSpectrumReference => {
            Some(SolverRoute::TridiagonalSturmSelected)
        }
        SolverRoute::TridiagonalSturmSelected => {
            Some(SolverRoute::TridiagonalFullSpectrumReference)
        }
        SolverRoute::LanczosExtremeReference => Some(SolverRoute::ShiftedPowerExtremeReference),
        SolverRoute::ShiftedPowerExtremeReference => Some(SolverRoute::LanczosExtremeReference),
        SolverRoute::BlockSubspaceExtremeReference => input
            .matrix_materialized
            .then_some(SolverRoute::DenseFullSpectrumReference),
        SolverRoute::DenseGeneralizedWhiteningReference => {
            Some(SolverRoute::MatrixFreeGeneralizedLobpcg)
        }
        SolverRoute::MatrixFreeGeneralizedLobpcg => input
            .matrix_materialized
            .then_some(SolverRoute::DenseGeneralizedWhiteningReference),
        SolverRoute::HpDenseReference => Some(SolverRoute::HpSelectedSpectrumPlanned),
        SolverRoute::HpTridiagonalFullSpectrumReference => {
            Some(SolverRoute::HpTridiagonalSturmSelected)
        }
        SolverRoute::HpTridiagonalSturmSelected => {
            Some(SolverRoute::HpTridiagonalFullSpectrumReference)
        }
        SolverRoute::HpMatrixFreeGeneralizedRayleighRitz => input
            .matrix_materialized
            .then_some(SolverRoute::HpDenseGeneralizedWhiteningReference),
        SolverRoute::HpBlockGeneralizedLobpcg => None,
        SolverRoute::HpBlockShiftInvert => input
            .matrix_materialized
            .then_some(SolverRoute::HpDenseReference),
        SolverRoute::HpThickRestartLanczos => input
            .matrix_materialized
            .then_some(SolverRoute::HpDenseReference),
        SolverRoute::HpDenseGeneralizedWhiteningReference => {
            Some(SolverRoute::HpMatrixFreeGeneralizedRayleighRitz)
        }
        SolverRoute::HpSelectedSpectrumPlanned => input
            .matrix_materialized
            .then_some(SolverRoute::HpDenseReference),
        SolverRoute::CertifiedInertiaPlanned => input
            .matrix_materialized
            .then_some(SolverRoute::HpDenseReference),
    };
    let independent_crosscheck = (input.assurance == AssuranceLevel::CrossChecked)
        .then_some(crosscheck_candidate)
        .flatten();

    let mut precision_schedule_bits = vec![input.precision.initial_bits];
    if input.assurance != AssuranceLevel::Computed {
        let repeat = input
            .precision
            .next_bits(input.precision.initial_bits)
            .unwrap_or(input.precision.maximum_bits);
        if repeat > input.precision.initial_bits {
            precision_schedule_bits.push(repeat);
        }
    }

    let route_requires_materialization = |route| {
        matches!(
            route,
            SolverRoute::DenseFullSpectrumReference
                | SolverRoute::TridiagonalFullSpectrumReference
                | SolverRoute::DenseGeneralizedWhiteningReference
                | SolverRoute::HpDenseReference
                | SolverRoute::HpTridiagonalFullSpectrumReference
                | SolverRoute::HpDenseGeneralizedWhiteningReference
                | SolverRoute::CertifiedInertiaPlanned
        )
    };
    let route_requires_factorization = |route| {
        matches!(
            route,
            SolverRoute::DenseGeneralizedWhiteningReference
                | SolverRoute::HpDenseGeneralizedWhiteningReference
                | SolverRoute::HpBlockShiftInvert
                | SolverRoute::HpSelectedSpectrumPlanned
                | SolverRoute::CertifiedInertiaPlanned
        )
    };
    let requires_materialization = route_requires_materialization(primary)
        || independent_crosscheck.is_some_and(route_requires_materialization);
    let requires_factorization = route_requires_factorization(primary)
        || independent_crosscheck.is_some_and(route_requires_factorization);
    let resource_estimate =
        estimate_solver_resources(input, requires_materialization, requires_factorization);

    let mut notes = vec![
        "solver plan is capability-based and must be persisted before execution".to_owned(),
        "unavailable HP or certified routes are errors, never f64 fallbacks".to_owned(),
    ];
    if matches!(primary, SolverRoute::HpSelectedSpectrumPlanned) {
        notes.push(
            "HP selected-spectrum production backend remains an implementation milestone"
                .to_owned(),
        );
    }
    if matches!(primary, SolverRoute::CertifiedInertiaPlanned) {
        notes.push(
            "certified execution requires interval matrix assembly and an interval inertia backend"
                .to_owned(),
        );
    }

    Ok(SolverPlan {
        primary,
        independent_crosscheck,
        requested_assurance: input.assurance,
        precision_schedule_bits,
        requires_materialization,
        requires_factorization,
        resource_estimate,
        expected_cached_artifacts: if requires_factorization {
            vec!["factorization".to_owned()]
        } else {
            Vec::new()
        },
        notes,
    })
}

fn estimate_solver_resources(
    input: &SolverPlannerInput,
    requires_materialization: bool,
    requires_factorization: bool,
) -> ResourceEstimate {
    let dimension = u64::try_from(input.dimension).unwrap_or(u64::MAX);
    let scalar_bytes = u64::from(input.precision.initial_bits)
        .saturating_add(7)
        .saturating_div(8)
        .max(8);
    let vector_bytes = dimension.saturating_mul(scalar_bytes);
    let matrix_bytes = dimension
        .saturating_mul(dimension)
        .saturating_mul(scalar_bytes);
    let iterative_vector_count = if input.requested_eigenpairs > 1 {
        6u64.saturating_mul(input.requested_eigenpairs.saturating_add(1) as u64)
    } else {
        20
    };
    let resident_memory_bytes = if requires_materialization {
        matrix_bytes.saturating_add(6u64.saturating_mul(vector_bytes))
    } else {
        iterative_vector_count.saturating_mul(vector_bytes)
    };
    let temporary_memory_bytes = if requires_factorization {
        Some(matrix_bytes)
    } else {
        Some(4u64.saturating_mul(vector_bytes))
    };

    ResourceEstimate {
        operator_dimension: input.dimension,
        resident_memory_bytes: Some(resident_memory_bytes),
        temporary_memory_bytes,
        temporary_disk_bytes: Some(0),
        persistent_artifact_bytes: Some(vector_bytes.saturating_add(scalar_bytes)),
        transfer_bytes: Some(0),
        estimated_cpu_seconds: None,
        estimated_wall_seconds: None,
        requested_threads: Some(1),
        estimated_operator_applications: None,
        estimated_factorizations: Some(u64::from(requires_factorization)),
        time_class: if requires_materialization {
            "cubic_dense_upper_bound".to_owned()
        } else {
            "iterative_operator_dependent".to_owned()
        },
        notes: vec![
            "preflight estimate is conservative and does not alter solver semantics".to_owned(),
            "CPU and wall estimates require calibration for the selected platform".to_owned(),
        ],
    }
}

/// Non-mathematical inputs needed to prove that a solver plan is executable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SolverPreflightContext {
    pub effective_config_digest: ConfigDigest,
    pub platform: String,
    pub scalar_backend: String,
    pub execution_fingerprint: ExecutionFingerprint,
    pub resources: ResourcePolicy,
    pub requested_threads: usize,
    pub cache_mode: CacheAccessMode,
    pub cache_policy_digest: Option<ConfigDigest>,
    pub cache_validation_mode: Option<xc_core::CacheValidationMode>,
    pub authenticated_principal: Option<String>,
    pub publication: PublicationPreflightRequest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreflightedSolverPlan {
    pub plan: SolverPlan,
    pub preflight: PreflightReport,
    pub resource_alternatives: Vec<SolverResourceAlternative>,
    pub execution_fingerprint_digest: ExecutionFingerprintDigest,
    pub primary_evidence: RouteEvidence,
    pub independent_evidence: Option<RouteEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverResourceAlternativeKind {
    MatrixFreeSelectedSpectrum,
    LargerResourceProfile,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SolverResourceAlternative {
    pub kind: SolverResourceAlternativeKind,
    pub preserves_requested_mathematics: bool,
    pub required_action: String,
    pub route: SolverRoute,
    pub profile: Option<ResourceProfile>,
    pub estimate: ResourceEstimate,
}

impl PreflightedSolverPlan {
    pub fn execution_allowed(&self) -> bool {
        self.preflight.accepted
    }
}

/// Plan first, then check the exact installed catalog. A rejected outcome is
/// returned as data so dry-run commands can show every missing capability.
pub fn plan_and_preflight_symmetric_eigenproblem(
    input: &SolverPlannerInput,
    context: &SolverPreflightContext,
    catalog: &CapabilityCatalog,
) -> Result<PreflightedSolverPlan, SolverError> {
    if context.requested_threads == 0 {
        return Err(SolverError::InvalidConfiguration(
            "requested thread count must be positive".to_owned(),
        ));
    }
    let fingerprint_digest = context
        .execution_fingerprint
        .digest()
        .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
    let resource_digest = context
        .resources
        .digest()
        .map_err(|error| SolverError::InvalidConfiguration(error.to_string()))?;
    if context.execution_fingerprint.effective_configuration_digest
        != context.effective_config_digest
        || context
            .execution_fingerprint
            .resolved_resource_policy_digest
            != resource_digest
        || context.execution_fingerprint.scalar_backend != context.scalar_backend
        || context
            .execution_fingerprint
            .precision
            .working_precision_bits
            != input.precision.initial_bits
        || context.execution_fingerprint.thread_policy.thread_count != context.requested_threads
    {
        return Err(SolverError::InvalidConfiguration(
            "execution fingerprint does not match the resolved solver request".to_owned(),
        ));
    }
    let mut plan = plan_symmetric_eigenproblem(input)?;
    plan.resource_estimate.requested_threads = Some(context.requested_threads);
    let certification_requested = input.assurance == AssuranceLevel::Certified;
    let request = PreflightRequest {
        effective_config_digest: context.effective_config_digest.clone(),
        platform: context.platform.clone(),
        scalar_backend: context.scalar_backend.clone(),
        precision_bits: input.precision.initial_bits,
        operator_representation: operator_representation(&input.structure).to_owned(),
        target_kind: target_kind(&input.target).to_owned(),
        generalized: input.generalized,
        primary_solver: plan.primary.id().to_owned(),
        independent_solver: plan
            .independent_crosscheck
            .map(|route| route.id().to_owned()),
        requested_assurance: input.assurance,
        certification_route: certification_requested.then(|| "interval_inertia".to_owned()),
        certification_claim: certification_requested.then(|| "eigenvalue_enclosure".to_owned()),
        cache_mode: context.cache_mode,
        cache_policy_digest: context.cache_policy_digest.clone(),
        cache_validation_mode: context.cache_validation_mode.clone(),
        authenticated_principal: context.authenticated_principal.clone(),
        publication: context.publication.clone(),
        resources: context.resources.clone(),
        estimate: plan.resource_estimate.clone(),
    };
    let preflight = catalog.preflight(&request);
    let resource_alternatives = solver_resource_alternatives(input, context, &plan, &preflight)?;
    let primary_evidence = plan.primary.evidence(
        input.precision.initial_bits,
        Some(context.requested_threads),
    );
    let independent_evidence = plan.independent_crosscheck.map(|route| {
        route.evidence(
            input.precision.initial_bits,
            Some(context.requested_threads),
        )
    });
    Ok(PreflightedSolverPlan {
        plan,
        preflight,
        resource_alternatives,
        execution_fingerprint_digest: fingerprint_digest,
        primary_evidence,
        independent_evidence,
    })
}

fn solver_resource_alternatives(
    input: &SolverPlannerInput,
    context: &SolverPreflightContext,
    plan: &SolverPlan,
    preflight: &PreflightReport,
) -> Result<Vec<SolverResourceAlternative>, SolverError> {
    if !preflight
        .failures
        .iter()
        .any(|failure| failure.code == PreflightFailureCode::InfeasibleResources)
    {
        return Ok(Vec::new());
    }
    let mut alternatives = Vec::new();
    if plan.requires_materialization && input.requested_eigenpairs < input.dimension {
        let mut matrix_free_input = input.clone();
        matrix_free_input.structure = xc_operator::MatrixStructure::MatrixFree;
        matrix_free_input.matrix_materialized = false;
        if let Ok(matrix_free_plan) = plan_symmetric_eigenproblem(&matrix_free_input) {
            if context
                .resources
                .assess(matrix_free_plan.resource_estimate.clone())
                .feasible
            {
                alternatives.push(SolverResourceAlternative {
                    kind: SolverResourceAlternativeKind::MatrixFreeSelectedSpectrum,
                    preserves_requested_mathematics: true,
                    required_action:
                        "provide the same operator through the matrix-free action contract"
                            .to_owned(),
                    route: matrix_free_plan.primary,
                    profile: Some(context.resources.profile),
                    estimate: matrix_free_plan.resource_estimate,
                });
            }
        }
    }
    for profile in [
        ResourceProfile::HighMemoryWorkstation,
        ResourceProfile::ExternalCompute,
    ] {
        if resource_profile_rank(profile) <= resource_profile_rank(context.resources.profile) {
            continue;
        }
        let policy = ResourcePolicy::for_profile(profile);
        if policy.assess(plan.resource_estimate.clone()).feasible {
            alternatives.push(SolverResourceAlternative {
                kind: SolverResourceAlternativeKind::LargerResourceProfile,
                preserves_requested_mathematics: true,
                required_action: format!(
                    "schedule the unchanged request under profile {:?}",
                    profile
                ),
                route: plan.primary,
                profile: Some(profile),
                estimate: plan.resource_estimate.clone(),
            });
        }
    }
    Ok(alternatives)
}

fn resource_profile_rank(profile: ResourceProfile) -> u8 {
    match profile {
        ResourceProfile::NormalWorkstation => 0,
        ResourceProfile::HighMemoryWorkstation => 1,
        ResourceProfile::ExternalCompute => 2,
    }
}

fn operator_representation(structure: &xc_operator::MatrixStructure) -> &'static str {
    match structure {
        xc_operator::MatrixStructure::Dense => "dense",
        xc_operator::MatrixStructure::PackedSymmetric => "packed_symmetric",
        xc_operator::MatrixStructure::Diagonal => "diagonal",
        xc_operator::MatrixStructure::Tridiagonal => "tridiagonal",
        xc_operator::MatrixStructure::Banded { .. } => "banded",
        xc_operator::MatrixStructure::MatrixFree => "matrix_free",
        xc_operator::MatrixStructure::Composite => "composite",
        xc_operator::MatrixStructure::RankOneUpdate => "rank_one_update",
    }
}

fn target_kind(target: &EigenTarget) -> &'static str {
    match target {
        EigenTarget::AlgebraicSmallest | EigenTarget::AlgebraicLargest => "algebraic_extreme",
        EigenTarget::SmallestMagnitude => "smallest_magnitude",
        EigenTarget::ClosestTo { .. } => "closest_to",
        EigenTarget::IndexRange { .. } => "index_range",
        EigenTarget::Interval { .. } => "interval",
    }
}

fn string_set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn solver_capability(
    route: SolverRoute,
    backends: &[&str],
    representations: &[&str],
    targets: &[&str],
    generalized: bool,
) -> SolverCapability {
    SolverCapability {
        id: route.id().to_owned(),
        algorithm_family: route.algorithm_family().to_owned(),
        scalar_backends: string_set(backends),
        operator_representations: string_set(representations),
        target_kinds: string_set(targets),
        generalized,
        maximum_assurance: AssuranceLevel::CrossChecked,
        checkpoint_supported: route == SolverRoute::BlockSubspaceExtremeReference,
    }
}

/// Capabilities compiled into this crate. Deliberately planned but unimplemented
/// routes are omitted, making preflight reject them instead of falling back.
pub fn installed_solver_capability_catalog() -> CapabilityCatalog {
    let platforms = string_set(&["windows", "linux", "macos"]);
    let scalar_backends = vec![ScalarCapability {
        id: "f64".to_owned(),
        supported_platforms: platforms.clone(),
        maximum_precision_bits: Some(64),
        arbitrary_precision: false,
        rigorous_real_enclosures: false,
        rigorous_complex_enclosures: false,
        exact: false,
    }];
    let all_representations = [
        "dense",
        "packed_symmetric",
        "diagonal",
        "tridiagonal",
        "banded",
        "matrix_free",
        "composite",
        "rank_one_update",
    ];
    let extreme = ["algebraic_extreme"];
    let solvers = vec![
        solver_capability(
            SolverRoute::DenseFullSpectrumReference,
            &["f64"],
            &["dense"],
            &extreme,
            false,
        ),
        solver_capability(
            SolverRoute::TridiagonalFullSpectrumReference,
            &["f64"],
            &["tridiagonal"],
            &["algebraic_extreme", "index_range"],
            false,
        ),
        solver_capability(
            SolverRoute::TridiagonalSturmSelected,
            &["f64"],
            &["tridiagonal"],
            &["algebraic_extreme", "index_range", "interval"],
            false,
        ),
        solver_capability(
            SolverRoute::ShiftedPowerExtremeReference,
            &["f64"],
            &all_representations,
            &extreme,
            false,
        ),
        solver_capability(
            SolverRoute::LanczosExtremeReference,
            &["f64"],
            &all_representations,
            &extreme,
            false,
        ),
        solver_capability(
            SolverRoute::BlockSubspaceExtremeReference,
            &["f64"],
            &all_representations,
            &extreme,
            false,
        ),
        solver_capability(
            SolverRoute::DenseGeneralizedWhiteningReference,
            &["f64"],
            &["dense"],
            &extreme,
            true,
        ),
        solver_capability(
            SolverRoute::MatrixFreeGeneralizedLobpcg,
            &["f64"],
            &all_representations,
            &extreme,
            true,
        ),
    ];

    #[cfg(feature = "hp-reference")]
    let solvers = {
        let mut solvers = solvers;
        solvers.push(solver_capability(
            SolverRoute::HpDenseReference,
            &["rug_mpfr"],
            &["dense"],
            &["algebraic_extreme", "smallest_magnitude", "closest_to"],
            false,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpTridiagonalFullSpectrumReference,
            &["rug_mpfr"],
            &["tridiagonal"],
            &["algebraic_extreme", "index_range"],
            false,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpTridiagonalSturmSelected,
            &["rug_mpfr"],
            &["tridiagonal"],
            &["algebraic_extreme", "index_range"],
            false,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpMatrixFreeGeneralizedRayleighRitz,
            &["rug_mpfr"],
            &all_representations,
            &extreme,
            true,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpBlockGeneralizedLobpcg,
            &["rug_mpfr"],
            &all_representations,
            &extreme,
            true,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpBlockShiftInvert,
            &["rug_mpfr"],
            &all_representations,
            &["smallest_magnitude", "closest_to", "interval"],
            false,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpThickRestartLanczos,
            &["rug_mpfr"],
            &all_representations,
            &extreme,
            false,
        ));
        solvers.push(solver_capability(
            SolverRoute::HpDenseGeneralizedWhiteningReference,
            &["rug_mpfr"],
            &["dense"],
            &extreme,
            true,
        ));
        solvers
    };

    #[cfg(feature = "hp-reference")]
    let scalar_backends = {
        let mut scalar_backends = scalar_backends;
        scalar_backends.push(ScalarCapability {
            id: "rug_mpfr".to_owned(),
            supported_platforms: platforms,
            maximum_precision_bits: None,
            arbitrary_precision: true,
            rigorous_real_enclosures: false,
            rigorous_complex_enclosures: false,
            exact: false,
        });
        scalar_backends
    };

    CapabilityCatalog {
        scalar_backends,
        solvers,
        // No portable interval certification implementation is compiled yet.
        certification_routes: Vec::<CertificationCapability>::new(),
    }
}

#[cfg(test)]
mod planner_tests {
    use super::*;
    use xc_core::{
        PrecisionPolicy, PreflightFailureCode, PublicationPreflightRequest, ResourcePolicy,
    };
    use xc_operator::MatrixStructure;

    #[test]
    fn tridiagonal_selected_plan_uses_sturm_reference() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Tridiagonal,
            dimension: 100,
            target: EigenTarget::IndexRange { first: 0, last: 2 },
            requested_eigenpairs: 3,
            assurance: AssuranceLevel::CrossChecked,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: true,
            generalized: false,
        };
        let plan = plan_symmetric_eigenproblem(&input).unwrap();
        assert_eq!(plan.primary, SolverRoute::TridiagonalSturmSelected);
        assert_eq!(
            plan.independent_crosscheck,
            Some(SolverRoute::TridiagonalFullSpectrumReference)
        );
    }

    #[test]
    fn hp_tridiagonal_selected_plan_uses_hp_sturm_with_full_qr_crosscheck() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Tridiagonal,
            dimension: 100,
            target: EigenTarget::IndexRange { first: 4, last: 7 },
            requested_eigenpairs: 4,
            assurance: AssuranceLevel::CrossChecked,
            precision: PrecisionPolicy::fixed(256),
            matrix_materialized: true,
            generalized: false,
        };
        let plan = plan_symmetric_eigenproblem(&input).unwrap();
        assert_eq!(plan.primary, SolverRoute::HpTridiagonalSturmSelected);
        assert_eq!(
            plan.independent_crosscheck,
            Some(SolverRoute::HpTridiagonalFullSpectrumReference)
        );
        assert!(!plan.requires_factorization);
        assert!(!plan
            .notes
            .iter()
            .any(|note| note.contains("implementation milestone")));
    }

    #[test]
    fn certified_plan_does_not_fall_back_to_f64() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 401,
            target: EigenTarget::AlgebraicSmallest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Certified,
            precision: PrecisionPolicy::fixed(4096),
            matrix_materialized: true,
            generalized: false,
        };
        let plan = plan_symmetric_eigenproblem(&input).unwrap();
        assert_eq!(plan.primary, SolverRoute::CertifiedInertiaPlanned);
        assert_eq!(plan.independent_crosscheck, None);
    }

    fn local_context(backend: &str) -> SolverPreflightContext {
        let resources = ResourcePolicy::default();
        let effective_configuration_digest = ConfigDigest("a".repeat(64));
        SolverPreflightContext {
            effective_config_digest: effective_configuration_digest.clone(),
            platform: "windows".to_owned(),
            scalar_backend: backend.to_owned(),
            execution_fingerprint: ExecutionFingerprint {
                schema_version: 1,
                toolkit_revision: env!("CARGO_PKG_VERSION").to_owned(),
                dependency_revisions: std::collections::BTreeMap::new(),
                compiler: "rustc-test".to_owned(),
                target_triple: "x86_64-pc-windows-msvc".to_owned(),
                native_libraries: std::collections::BTreeMap::new(),
                scalar_backend: backend.to_owned(),
                scalar_backend_version: "test".to_owned(),
                precision: xc_core::PrecisionFingerprint {
                    working_precision_bits: if backend == "f64" { 53 } else { 256 },
                    guard_bits: 0,
                    rounding_policy: "nearest".to_owned(),
                },
                algorithm_semantics_versions: std::collections::BTreeMap::new(),
                cpu_feature_policy: "portable".to_owned(),
                thread_policy: xc_core::ThreadPolicyFingerprint {
                    thread_count: 1,
                    scheduling_policy: "single-thread".to_owned(),
                    reduction_policy: "serial".to_owned(),
                },
                feature_flags: BTreeSet::new(),
                effective_configuration_digest,
                resolved_resource_policy_digest: resources.digest().unwrap(),
                reproducibility: xc_core::Reproducibility::Deterministic,
            },
            resources,
            requested_threads: 1,
            cache_mode: CacheAccessMode::ReadOnly,
            cache_policy_digest: Some(ConfigDigest("c".repeat(64))),
            cache_validation_mode: Some(xc_core::CacheValidationMode::Full),
            authenticated_principal: None,
            publication: PublicationPreflightRequest::default(),
        }
    }

    #[test]
    fn installed_crosscheck_routes_pass_exact_preflight() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Tridiagonal,
            dimension: 100,
            target: EigenTarget::IndexRange { first: 0, last: 2 },
            requested_eigenpairs: 3,
            assurance: AssuranceLevel::CrossChecked,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: true,
            generalized: false,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("f64"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(
            outcome.execution_allowed(),
            "{:?}",
            outcome.preflight.failures
        );
        assert!(outcome.independent_evidence.is_some());
    }

    #[test]
    fn computed_plan_does_not_schedule_unrequested_crosscheck() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 100,
            target: EigenTarget::AlgebraicSmallest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: false,
            generalized: false,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("f64"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(outcome.execution_allowed());
        assert_eq!(outcome.plan.independent_crosscheck, None);
    }

    #[test]
    fn multiple_matrix_free_extremes_select_the_block_route() {
        let catalog = installed_solver_capability_catalog();
        let input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 1_000,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 4,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: false,
            generalized: false,
        };
        let outcome =
            plan_and_preflight_symmetric_eigenproblem(&input, &local_context("f64"), &catalog)
                .unwrap();
        assert!(outcome.execution_allowed());
        assert_eq!(
            outcome.plan.primary,
            SolverRoute::BlockSubspaceExtremeReference
        );
        assert!(
            catalog
                .solvers
                .iter()
                .find(|capability| capability.id == SolverRoute::BlockSubspaceExtremeReference.id())
                .unwrap()
                .checkpoint_supported
        );
        // Six conservative live blocks, each including a guard vector beyond
        // the four requested values.
        assert!(
            outcome
                .plan
                .resource_estimate
                .resident_memory_bytes
                .unwrap()
                >= 6 * 5 * 1_000 * 8
        );
    }

    #[test]
    fn matrix_free_memory_estimate_scales_with_vectors_not_dense_entries() {
        let small_input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 1_000,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 4,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: false,
            generalized: false,
        };
        let mut large_input = small_input.clone();
        large_input.dimension = 2_000;
        let small = plan_symmetric_eigenproblem(&small_input).unwrap();
        let large = plan_symmetric_eigenproblem(&large_input).unwrap();
        assert_eq!(
            large.resource_estimate.resident_memory_bytes,
            small
                .resource_estimate
                .resident_memory_bytes
                .map(|bytes| bytes * 2)
        );
        assert_eq!(
            large.resource_estimate.temporary_memory_bytes,
            small
                .resource_estimate
                .temporary_memory_bytes
                .map(|bytes| bytes * 2)
        );

        let mut dense_small_input = small_input;
        dense_small_input.structure = MatrixStructure::Dense;
        dense_small_input.matrix_materialized = true;
        let mut dense_large_input = dense_small_input.clone();
        dense_large_input.dimension = 2_000;
        let dense_small = plan_symmetric_eigenproblem(&dense_small_input).unwrap();
        let dense_large = plan_symmetric_eigenproblem(&dense_large_input).unwrap();
        assert!(
            dense_large.resource_estimate.resident_memory_bytes.unwrap()
                > 3 * dense_small.resource_estimate.resident_memory_bytes.unwrap()
        );
        assert!(
            large.resource_estimate.resident_memory_bytes
                < dense_small.resource_estimate.resident_memory_bytes
        );
    }

    #[test]
    fn matrix_free_generalized_plan_selects_metric_lobpcg() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 500,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: false,
            generalized: true,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("f64"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(outcome.execution_allowed());
        assert_eq!(
            outcome.plan.primary,
            SolverRoute::MatrixFreeGeneralizedLobpcg
        );
        assert!(!outcome.plan.requires_materialization);
    }

    #[cfg(feature = "hp-reference")]
    #[test]
    fn hp_matrix_free_generalized_plan_selects_real_mpfr_route() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 500,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(256),
            matrix_materialized: false,
            generalized: true,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("rug_mpfr"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(
            outcome.execution_allowed(),
            "{:?}",
            outcome.preflight.failures
        );
        assert_eq!(
            outcome.plan.primary,
            SolverRoute::HpMatrixFreeGeneralizedRayleighRitz
        );
        assert!(!outcome.plan.requires_materialization);
        assert!(!outcome.plan.requires_factorization);
    }

    #[cfg(feature = "hp-reference")]
    #[test]
    fn hp_dense_generalized_crosscheck_pairs_matrix_free_with_whitening() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 50,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::CrossChecked,
            precision: PrecisionPolicy::fixed(256),
            matrix_materialized: true,
            generalized: true,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("rug_mpfr"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(
            outcome.execution_allowed(),
            "{:?}",
            outcome.preflight.failures
        );
        assert_eq!(
            outcome.plan.primary,
            SolverRoute::HpMatrixFreeGeneralizedRayleighRitz
        );
        assert_eq!(
            outcome.plan.independent_crosscheck,
            Some(SolverRoute::HpDenseGeneralizedWhiteningReference)
        );
        assert!(outcome.plan.requires_materialization);
        assert!(outcome.plan.requires_factorization);
        assert!(outcome.independent_evidence.is_some());
    }

    #[test]
    fn dense_generalized_crosscheck_pairs_whitening_with_lobpcg() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 50,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::CrossChecked,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: true,
            generalized: true,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("f64"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(
            outcome.execution_allowed(),
            "{:?}",
            outcome.preflight.failures
        );
        assert_eq!(
            outcome.plan.primary,
            SolverRoute::DenseGeneralizedWhiteningReference
        );
        assert_eq!(
            outcome.plan.independent_crosscheck,
            Some(SolverRoute::MatrixFreeGeneralizedLobpcg)
        );
        assert_ne!(
            outcome.primary_evidence.formulation,
            outcome.independent_evidence.unwrap().formulation
        );
    }

    #[test]
    fn planner_rejects_out_of_range_indices_and_f64_generalized_blocks() {
        let out_of_range = SolverPlannerInput {
            structure: MatrixStructure::Tridiagonal,
            dimension: 3,
            target: EigenTarget::IndexRange { first: 1, last: 3 },
            requested_eigenpairs: 3,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: true,
            generalized: false,
        };
        assert!(out_of_range.validate().is_err());

        let generalized_block = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 10,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 2,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: false,
            generalized: true,
        };
        assert!(matches!(
            generalized_block.validate(),
            Err(SolverError::UnsupportedTarget(_))
        ));
    }

    #[cfg(feature = "hp-reference")]
    #[test]
    fn hp_generalized_block_plan_selects_installed_matrix_free_route() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 1_000,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 4,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(256),
            matrix_materialized: false,
            generalized: true,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("rug_mpfr"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(
            outcome.execution_allowed(),
            "{:?}",
            outcome.preflight.failures
        );
        assert_eq!(outcome.plan.primary, SolverRoute::HpBlockGeneralizedLobpcg);
        assert_eq!(outcome.plan.independent_crosscheck, None);
        assert!(!outcome.plan.requires_materialization);
        assert!(!outcome.plan.requires_factorization);
        assert!(
            outcome
                .plan
                .resource_estimate
                .resident_memory_bytes
                .unwrap()
                >= 6 * 5 * 1_000 * 32
        );
    }

    #[cfg(feature = "hp-reference")]
    #[test]
    fn hp_interior_plan_selects_installed_block_shift_invert_route() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::MatrixFree,
            dimension: 1_000,
            target: EigenTarget::ClosestTo {
                shift: xc_core::DecimalLiteral::new("0.125").unwrap(),
            },
            requested_eigenpairs: 3,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(256),
            matrix_materialized: false,
            generalized: false,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("rug_mpfr"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(
            outcome.execution_allowed(),
            "{:?}",
            outcome.preflight.failures
        );
        assert_eq!(outcome.plan.primary, SolverRoute::HpBlockShiftInvert);
        assert!(!outcome.plan.requires_materialization);
        assert!(outcome.plan.requires_factorization);
        assert_eq!(
            outcome.plan.expected_cached_artifacts,
            vec!["factorization".to_owned()]
        );
    }

    #[test]
    fn uncompiled_hp_route_is_rejected_without_f64_fallback() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 20,
            target: EigenTarget::AlgebraicSmallest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(256),
            matrix_materialized: true,
            generalized: false,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("rug_mpfr"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        #[cfg(not(feature = "hp-reference"))]
        assert!(outcome.preflight.failures.iter().any(|failure| matches!(
            failure.code,
            PreflightFailureCode::UnsupportedScalarBackend
                | PreflightFailureCode::UnsupportedSolver
        )));
        #[cfg(feature = "hp-reference")]
        {
            assert!(outcome.execution_allowed());
            assert_eq!(outcome.plan.primary, SolverRoute::HpThickRestartLanczos);
            assert!(!outcome.plan.requires_factorization);
        }
        assert_ne!(
            outcome.plan.primary,
            SolverRoute::DenseFullSpectrumReference
        );
    }

    #[test]
    fn oversized_dense_plan_fails_resource_preflight() {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 100_000,
            target: EigenTarget::AlgebraicSmallest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(53),
            matrix_materialized: true,
            generalized: false,
        };
        let outcome = plan_and_preflight_symmetric_eigenproblem(
            &input,
            &local_context("f64"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(outcome
            .preflight
            .failures
            .iter()
            .any(|failure| { failure.code == PreflightFailureCode::InfeasibleResources }));
        assert!(!outcome.execution_allowed());
        assert!(outcome.resource_alternatives.iter().any(|alternative| {
            alternative.kind == SolverResourceAlternativeKind::MatrixFreeSelectedSpectrum
                && alternative.preserves_requested_mathematics
                && alternative.estimate.resident_memory_bytes
                    < outcome.plan.resource_estimate.resident_memory_bytes
        }));
        assert!(outcome.resource_alternatives.iter().any(|alternative| {
            alternative.kind == SolverResourceAlternativeKind::LargerResourceProfile
                && alternative.profile == Some(ResourceProfile::ExternalCompute)
                && alternative.estimate == outcome.plan.resource_estimate
        }));
        assert!(outcome.resource_alternatives.iter().any(|alternative| {
            alternative.kind == SolverResourceAlternativeKind::LargerResourceProfile
                && alternative.profile == Some(ResourceProfile::HighMemoryWorkstation)
        }));
    }
}

// ===========================================================================
// High-precision report cross-checking
// ===========================================================================

#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HpCrossCheckTolerance {
    pub eigenvalue_absolute: xc_core::DecimalLiteral,
    pub one_minus_overlap_squared: xc_core::DecimalLiteral,
}

#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossCheckedEigenpairHp {
    pub accepted: EigenpairReportHp,
    pub independent: EigenpairReportHp,
    pub eigenvalue_absolute_difference: String,
    pub vector_overlap_squared: String,
    pub one_minus_overlap_squared: String,
    pub tolerance: HpCrossCheckTolerance,
}

#[cfg(feature = "hp-reference")]
fn hp_parse_string(value: &str, precision_bits: u32) -> Result<rug::Float, SolverError> {
    let parsed = rug::Float::parse(value).map_err(|error| {
        SolverError::InvalidConfiguration(format!(
            "failed to parse HP report value {value:?}: {error}"
        ))
    })?;
    Ok(rug::Float::with_val(precision_bits, parsed))
}

#[cfg(feature = "hp-reference")]
pub fn cross_check_hp_reports(
    primary: &EigenpairReportHp,
    independent: &EigenpairReportHp,
    tolerance: HpCrossCheckTolerance,
) -> Result<CrossCheckedEigenpairHp, SolverError> {
    use rug::Float;
    if primary.eigenvector.len() != independent.eigenvector.len() || primary.eigenvector.is_empty()
    {
        return Err(SolverError::CrossCheckDisagreement(
            "HP eigenvectors must have the same nonzero dimension".to_owned(),
        ));
    }
    let precision_bits = primary
        .precision_bits
        .max(independent.precision_bits)
        .saturating_add(64);
    let primary_value = hp_parse_string(&primary.eigenvalue, precision_bits)?;
    let independent_value = hp_parse_string(&independent.eigenvalue, precision_bits)?;
    let mut eigenvalue_difference = primary_value;
    eigenvalue_difference -= independent_value;
    eigenvalue_difference.abs_mut();

    let left: Vec<Float> = primary
        .eigenvector
        .iter()
        .map(|value| hp_parse_string(value, precision_bits))
        .collect::<Result<_, _>>()?;
    let right: Vec<Float> = independent
        .eigenvector
        .iter()
        .map(|value| hp_parse_string(value, precision_bits))
        .collect::<Result<_, _>>()?;
    let mut dot = Float::with_val(precision_bits, 0);
    let mut left_norm_sq = Float::with_val(precision_bits, 0);
    let mut right_norm_sq = Float::with_val(precision_bits, 0);
    for (left_value, right_value) in left.iter().zip(&right) {
        let mut term = left_value.clone();
        term *= right_value;
        dot += term;
        let mut square = left_value.clone();
        square *= left_value;
        left_norm_sq += square;
        let mut square = right_value.clone();
        square *= right_value;
        right_norm_sq += square;
    }
    if left_norm_sq.is_zero() || right_norm_sq.is_zero() {
        return Err(SolverError::CrossCheckDisagreement(
            "HP cross-check received a zero eigenvector".to_owned(),
        ));
    }
    dot.square_mut();
    let mut norm_product = left_norm_sq;
    norm_product *= right_norm_sq;
    let mut overlap_squared = dot;
    overlap_squared /= norm_product;
    if overlap_squared > 1 {
        // Decimal serialization and independent normalization may place the
        // computed overlap a few ulps above one. Clamp only for reporting the
        // sign-invariant overlap metric, never for eigenvalue acceptance.
        overlap_squared = Float::with_val(precision_bits, 1);
    }
    let mut one_minus_overlap = Float::with_val(precision_bits, 1);
    one_minus_overlap -= &overlap_squared;
    one_minus_overlap.abs_mut();

    let eigenvalue_tolerance = hp_parse_literal(&tolerance.eigenvalue_absolute, precision_bits)?;
    let overlap_tolerance = hp_parse_literal(&tolerance.one_minus_overlap_squared, precision_bits)?;
    if eigenvalue_difference > eigenvalue_tolerance || one_minus_overlap > overlap_tolerance {
        return Err(SolverError::CrossCheckDisagreement(format!(
            "HP reports disagree: eigenvalue difference {}, one-minus-overlap {}",
            eigenvalue_difference.to_string_radix(10, Some(20)),
            one_minus_overlap.to_string_radix(10, Some(20))
        )));
    }

    // Exact-round-trip width; the bare ceiling loses one ulp on decode.
    let digits = xc_numerics::reduction::roundtrip_decimal_digits(precision_bits).max(32);
    let mut accepted = primary.clone();
    accepted.assurance = AssuranceLevel::CrossChecked;
    Ok(CrossCheckedEigenpairHp {
        accepted,
        independent: independent.clone(),
        eigenvalue_absolute_difference: hp_decimal(&eigenvalue_difference, digits),
        vector_overlap_squared: hp_decimal(&overlap_squared, digits),
        one_minus_overlap_squared: hp_decimal(&one_minus_overlap, digits),
        tolerance,
    })
}

#[cfg(all(test, feature = "hp-reference"))]
mod hp_crosscheck_tests {
    use super::*;

    fn report(value: &str, vector: &[&str], precision_bits: u32) -> EigenpairReportHp {
        EigenpairReportHp {
            eigenvalue: value.to_owned(),
            eigenvector: vector.iter().map(|entry| (*entry).to_owned()).collect(),
            residual_norm: "1e-100".to_owned(),
            relative_residual: "1e-100".to_owned(),
            scaled_backward_error: "1e-100".to_owned(),
            diagnostics: EigenpairDiagnostics {
                absolute_residual: "1e-100".to_owned(),
                relative_residual: "1e-100".to_owned(),
                scaled_backward_error: "1e-100".to_owned(),
                orthogonality_error: "0".to_owned(),
            },
            precision_bits,
            algorithm: "fixture".to_owned(),
            status: ResultStatus::Converged,
            termination: TerminationReason::ResidualTolerance,
            assurance: AssuranceLevel::Computed,
            provenance: SolverProvenance::current_package("rug_mpfr"),
        }
    }

    #[test]
    fn hp_crosscheck_is_sign_invariant() {
        let primary = report("1e-400", &["1", "0"], 1024);
        let independent = report("1.0000000001e-400", &["-1", "0"], 2048);
        let result = cross_check_hp_reports(
            &primary,
            &independent,
            HpCrossCheckTolerance {
                eigenvalue_absolute: xc_core::DecimalLiteral::new("1e-409").unwrap(),
                one_minus_overlap_squared: xc_core::DecimalLiteral::new("1e-50").unwrap(),
            },
        )
        .unwrap();
        assert_eq!(result.accepted.assurance, AssuranceLevel::CrossChecked);
    }
}

// ===========================================================================
// Diagonal-plus-rank-one secular reference solver
// ===========================================================================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankOneSecularSpectrumF64 {
    pub eigenvalues: Vec<f64>,
    pub residuals: Vec<f64>,
    pub bisection_iterations: usize,
    pub algorithm: String,
    pub assumptions: Vec<String>,
}

fn rank_one_secular_value(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    lambda: f64,
) -> Result<f64, SolverError> {
    let mut value = 1.0;
    for (&pole, &component) in diagonal.iter().zip(vector) {
        let denominator = pole - lambda;
        if denominator == 0.0 {
            return Err(SolverError::NumericalBreakdown(
                "rank-one secular evaluation reached a diagonal pole".to_owned(),
            ));
        }
        value += alpha * component * component / denominator;
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SolverError::NumericalBreakdown(
            "rank-one secular evaluation was non-finite".to_owned(),
        ))
    }
}

fn adjacent_float_toward(value: f64, toward_positive: bool) -> f64 {
    if value.is_nan()
        || value
            == if toward_positive {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
    {
        return value;
    }
    if value == 0.0 {
        return if toward_positive {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        };
    }
    let bits = value.to_bits();
    let next = if (value > 0.0) == toward_positive {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

fn bisect_monotone_secular(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    mut lower: f64,
    mut upper: f64,
    tolerance: f64,
    maximum_iterations: usize,
) -> Result<(f64, f64, usize), SolverError> {
    let mut f_lower = rank_one_secular_value(diagonal, vector, alpha, lower)?;
    let f_upper = rank_one_secular_value(diagonal, vector, alpha, upper)?;
    if f_lower == 0.0 {
        return Ok((lower, 0.0, 0));
    }
    if f_upper == 0.0 {
        return Ok((upper, 0.0, 0));
    }
    if f_lower.is_sign_positive() == f_upper.is_sign_positive() {
        return Err(SolverError::NumericalBreakdown(format!(
            "rank-one secular bracket does not change sign: [{lower:e}, {upper:e}]"
        )));
    }
    for iteration in 1..=maximum_iterations {
        let midpoint = lower + 0.5 * (upper - lower);
        let f_midpoint = rank_one_secular_value(diagonal, vector, alpha, midpoint)?;
        if f_midpoint == 0.0 || (upper - lower).abs() <= tolerance * midpoint.abs().max(1.0) {
            return Ok((midpoint, f_midpoint.abs(), iteration));
        }
        if f_lower.is_sign_positive() != f_midpoint.is_sign_positive() {
            upper = midpoint;
        } else {
            lower = midpoint;
            f_lower = f_midpoint;
        }
    }
    let midpoint = lower + 0.5 * (upper - lower);
    let residual = rank_one_secular_value(diagonal, vector, alpha, midpoint)?.abs();
    Ok((midpoint, residual, maximum_iterations))
}

/// Enumerate every eigenvalue of `diag(d) + alpha * u u^T` through its
/// secular equation.
///
/// This trusted f64 route requires strictly increasing diagonal entries and
/// nonzero update components. Under those assumptions the secular function is
/// strictly monotone between poles and the rank-one interlacing count is
/// complete. The generic result is useful as an independent route for
/// arrowhead/rank-one formulations; applying it to CCM requires a separately
/// reviewed derivation of the correct finite operator and metric.
pub fn diagonal_rank_one_spectrum_f64(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    tolerance: f64,
    maximum_iterations: usize,
) -> Result<RankOneSecularSpectrumF64, SolverError> {
    if diagonal.is_empty() || diagonal.len() != vector.len() {
        return Err(SolverError::InvalidConfiguration(
            "rank-one secular solver requires equal nonzero diagonal and vector lengths".to_owned(),
        ));
    }
    if diagonal
        .iter()
        .chain(vector)
        .any(|value| !value.is_finite())
        || !alpha.is_finite()
        || alpha == 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || maximum_iterations == 0
    {
        return Err(SolverError::InvalidConfiguration(
            "rank-one secular inputs must be finite with nonzero alpha and positive controls"
                .to_owned(),
        ));
    }
    if diagonal.windows(2).any(|window| window[0] >= window[1]) {
        return Err(SolverError::InvalidConfiguration(
            "rank-one reference route requires strictly increasing diagonal entries".to_owned(),
        ));
    }
    if vector.contains(&0.0) {
        return Err(SolverError::UnsupportedTarget(
            "zero update components create deflated diagonal eigenvalues; split them before using this reference route"
                .to_owned(),
        ));
    }

    let norm_bound = diagonal.iter().map(|value| value.abs()).fold(0.0, f64::max)
        + alpha.abs() * vector.iter().map(|value| value * value).sum::<f64>();
    let expansion = norm_bound.max(1.0).mul_add(2.0, 1.0);
    let mut brackets = Vec::with_capacity(diagonal.len());
    if alpha > 0.0 {
        for window in diagonal.windows(2) {
            brackets.push((
                adjacent_float_toward(window[0], true),
                adjacent_float_toward(window[1], false),
            ));
        }
        brackets.push((
            adjacent_float_toward(*diagonal.last().unwrap(), true),
            *diagonal.last().unwrap() + expansion,
        ));
    } else {
        brackets.push((
            diagonal[0] - expansion,
            adjacent_float_toward(diagonal[0], false),
        ));
        for window in diagonal.windows(2) {
            brackets.push((
                adjacent_float_toward(window[0], true),
                adjacent_float_toward(window[1], false),
            ));
        }
    }

    let mut eigenvalues = Vec::with_capacity(diagonal.len());
    let mut residuals = Vec::with_capacity(diagonal.len());
    let mut total_iterations = 0usize;
    for (lower, upper) in brackets {
        let (root, residual, iterations) = bisect_monotone_secular(
            diagonal,
            vector,
            alpha,
            lower,
            upper,
            tolerance,
            maximum_iterations,
        )?;
        eigenvalues.push(root);
        residuals.push(residual);
        total_iterations += iterations;
    }
    eigenvalues.sort_by(f64::total_cmp);
    Ok(RankOneSecularSpectrumF64 {
        eigenvalues,
        residuals,
        bisection_iterations: total_iterations,
        algorithm: "diagonal_rank_one_secular_bisection_f64".to_owned(),
        assumptions: vec![
            "diagonal entries are strictly increasing".to_owned(),
            "every rank-one update component is nonzero".to_owned(),
            "result is an f64 reference route, not a certified enclosure".to_owned(),
        ],
    })
}

#[cfg(test)]
mod rank_one_secular_tests {
    use super::*;

    #[test]
    fn secular_spectrum_matches_dense_reference() {
        let diagonal = vec![-2.0, 1.0, 4.0];
        let vector = vec![1.0, 0.5, 2.0];
        let alpha = 0.75;
        let secular =
            diagonal_rank_one_spectrum_f64(&diagonal, &vector, alpha, 1e-14, 200).unwrap();
        let mut dense = DMatrix::<f64>::zeros(3, 3);
        for row in 0..3 {
            dense[(row, row)] = diagonal[row];
            for column in 0..3 {
                dense[(row, column)] += alpha * vector[row] * vector[column];
            }
        }
        let reference = SymmetricEigen::new(dense);
        let mut reference_eigenvalues = reference.eigenvalues.iter().copied().collect::<Vec<_>>();
        reference_eigenvalues.sort_by(f64::total_cmp);
        for (observed, expected) in secular.eigenvalues.iter().zip(reference_eigenvalues.iter()) {
            assert!((observed - expected).abs() < 1e-11);
        }
    }

    #[test]
    fn negative_update_places_one_root_below_first_pole() {
        let spectrum =
            diagonal_rank_one_spectrum_f64(&[1.0, 3.0], &[1.0, 1.0], -0.5, 1e-14, 200).unwrap();
        assert!(spectrum.eigenvalues[0] < 1.0);
        assert!(spectrum.eigenvalues[1] > 1.0 && spectrum.eigenvalues[1] < 3.0);
    }
}
