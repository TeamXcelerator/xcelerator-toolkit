// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Multi-solver framework.
//!
//! The f64 solvers in this crate are deterministic reference and discovery
//! implementations.  They are never a silent replacement for an HP request.
//! The same result contracts are intended for the HP and certified backends.

mod exact_sturm_f64;

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
mod hp_boundary_count;
#[cfg(feature = "hp-reference")]
pub use hp_boundary_count::{BoundaryCountEvidenceHp, HP_KRYLOV_COUNT_SEMANTICS};
#[cfg(feature = "hp-reference")]
mod hp_generalized;
#[cfg(feature = "hp-reference")]
pub use hp_generalized::*;
#[cfg(feature = "hp-reference")]
mod hp_generalized_crosscheck;
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
    #[doc(hidden)]
    PremiseNotVerified(String),
    UnsupportedTarget(String),
    Operator(OperatorError),
    NumericalBreakdown(String),
    /// The producer established a precision-dependent resolution or rank limit.
    PrecisionExhausted(String),
    NonConvergence(String),
    /// A fixed work budget ended; precision escalation does not enlarge it.
    IterationBudgetExhausted(String),
    /// A one-vector comparison cannot establish agreement of a non-simple eigenspace.
    UnresolvedEigenspace(String),
    CrossCheckDisagreement(String),
    Cancelled(String),
}

impl Display for SolverError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid solver configuration: {message}")
            }
            Self::PremiseNotVerified(message) => write!(f, "premise not verified: {message}"),
            Self::UnsupportedTarget(message) => write!(f, "unsupported target: {message}"),
            Self::Operator(error) => Display::fmt(error, f),
            Self::NumericalBreakdown(message) => write!(f, "numerical breakdown: {message}"),
            Self::PrecisionExhausted(message) => {
                write!(f, "working precision exhausted: {message}")
            }
            Self::NonConvergence(message) => write!(f, "solver did not converge: {message}"),
            Self::IterationBudgetExhausted(message) => {
                write!(f, "solver work budget exhausted: {message}")
            }
            Self::UnresolvedEigenspace(message) => {
                write!(f, "eigenspace comparison is unresolved: {message}")
            }
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

/// A residual/backward-error report for an approximate eigenpair. `Converged`
/// describes those stopping tests; it does not establish a requested global
/// extremum when the iteration has only visited a proper invariant subspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EigenpairReportF64 {
    /// Full-space numerical selection was performed. This is computed ordering
    /// evidence, not an interval certificate. Missing legacy evidence is false.
    #[serde(default)]
    pub global_target_ordering_established: bool,
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub residual_norm: f64,
    pub relative_residual: f64,
    /// Image-based ratio ||Av-lambda v|| / (||Av||+|lambda| ||v||).
    /// This conservative computable scale does not use a caller's norm upper bound.
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

pub trait EigenSolverF64: Send + Sync + std::any::Any {
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

// Native routes act on the supplied operator. They do not construct a
// requested parity/basis restriction and cannot supply more than 53 bits.
fn validate_native_solver_config(config: &SolverConfig) -> Result<(), SolverError> {
    config
        .validate()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    if config.subspace != xc_core::Subspace::Full {
        return Err(SolverError::UnsupportedTarget(
            "native solver requires an already reduced operator with Full subspace".into(),
        ));
    }
    let working = config
        .precision
        .initial_working_bits()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    if working > f64::MANTISSA_DIGITS {
        return Err(SolverError::InvalidConfiguration(format!(
            "native solver has 53 significand bits; requested {working} working bits require an HP route"
        )));
    }
    Ok(())
}

// Bound every native dense/Ritz eigendecomposition and validate its actual
// arithmetic. The library QR can pair tiny eigenvalues with the wrong columns
// and lose 2x2 eigenvector accuracy, so its orthogonal basis is completed by
// Jacobi rotations: values come back ascending, each with its own column.
fn checked_symmetric_decomposition_f64(
    matrix: DMatrix<f64>,
) -> Result<SymmetricEigen<f64, nalgebra::Dyn>, SolverError> {
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n || matrix.iter().any(|v| !v.is_finite()) {
        return Err(SolverError::NumericalBreakdown(
            "symmetric eigendecomposition requires a finite nonempty square matrix".into(),
        ));
    }
    let mut result = SymmetricEigen::try_new(matrix.clone(), f64::EPSILON, n.saturating_mul(128))
        .ok_or_else(|| {
        SolverError::NonConvergence("native symmetric QR exceeded its iteration budget".into())
    })?;
    if result
        .eigenvalues
        .iter()
        .chain(result.eigenvectors.iter())
        .any(|v| !v.is_finite())
    {
        return Err(SolverError::NumericalBreakdown(
            "native symmetric eigendecomposition produced nonfinite values".into(),
        ));
    }
    let values = xc_numerics::symmetric_f64::complete_symmetric_eigensystem_f64(
        matrix.as_slice(),
        n,
        result.eigenvectors.as_mut_slice(),
    )
    .map_err(|error| SolverError::NumericalBreakdown(error.to_string()))?;
    result.eigenvalues = nalgebra::DVector::from_vec(values);
    Ok(result)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(x: &[f64]) -> f64 {
    if x.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let squared = dot(x, x);
    if squared.is_normal() {
        return squared.sqrt();
    }
    let maximum = x.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if maximum == 0.0 {
        return 0.0;
    }
    let scaled_square: f64 = x.iter().map(|v| (v / maximum).powi(2)).sum();
    maximum * scaled_square.sqrt()
}

fn normalize(x: &mut [f64]) -> Result<(), SolverError> {
    let n = norm(x);
    if !n.is_finite() || n <= 0.0 {
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
    if vector.len() != operator.dimension()
        || workspace.len() != vector.len()
        || vector.is_empty()
        || vector.iter().any(|v| !v.is_finite())
    {
        return Err(SolverError::NumericalBreakdown(
            "invalid eigenpair diagnostic vector".into(),
        ));
    }
    operator.apply(vector, workspace)?;
    let eigenvalue = dot(vector, workspace) / dot(vector, vector);
    let vector_norm = norm(vector);
    let residuals: Vec<_> = workspace
        .iter()
        .zip(vector)
        .map(|(av, v)| av - eigenvalue * v)
        .collect();
    let residual = norm(&residuals);
    let applied_norm = norm(workspace);
    if !eigenvalue.is_finite()
        || !vector_norm.is_finite()
        || vector_norm == 0.0
        || !residual.is_finite()
        || !applied_norm.is_finite()
    {
        return Err(SolverError::NumericalBreakdown(
            "eigenpair diagnostic arithmetic is not finite".into(),
        ));
    }
    // Preserve ordinary arithmetic, but never floor a nonzero denominator
    // to MIN_POSITIVE or divide by an overflowing intermediate sum/product.
    let ratio = |a: f64, b: f64, factor: f64| -> Result<f64, SolverError> {
        let denominator = a + b * factor;
        let value = if denominator.is_normal() {
            residual / denominator
        } else {
            let scale = a.max(b);
            if scale == 0.0 {
                if residual == 0.0 {
                    0.0
                } else {
                    f64::NAN
                }
            } else {
                (residual / scale) / (a / scale + (b / scale) * factor)
            }
        };
        if !value.is_finite() || (value == 0.0 && residual != 0.0) {
            Err(SolverError::NumericalBreakdown(
                "eigenpair diagnostic ratio is not representable".into(),
            ))
        } else {
            Ok(value)
        }
    };
    let relative = ratio(applied_norm, eigenvalue.abs(), vector_norm)?;
    let backward = relative;
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
    fn downward(literal: &xc_core::DecimalLiteral) -> Result<f64, SolverError> {
        let invalid =
            |error: xc_core::ConfigError| SolverError::InvalidConfiguration(error.to_string());
        let mut value = literal.parse_f64().map_err(invalid)?;
        // Every finite binary64 value has a terminating decimal expansion
        // with at most 1074 fractional digits. Compare that exact expansion,
        // not Display's shortest round-trip approximation, to the request.
        let exact = xc_core::DecimalLiteral::from_f64_exact(value).map_err(invalid)?;
        if exact.cmp_numeric(literal).map_err(invalid)? == std::cmp::Ordering::Greater {
            value = value.next_down();
        }
        if !value.is_finite() || value <= 0. {
            return Err(SolverError::InvalidConfiguration(
                "positive native stopping tolerance is outside the supported range".into(),
            ));
        }
        Ok(value)
    }
    Ok((
        downward(&config.stopping.absolute_residual)?,
        downward(&config.stopping.scaled_backward_error)?,
    ))
}

/// Deterministic shifted power iteration.  The transformation `b I ± A`
/// makes the desired algebraic extreme the dominant nonnegative eigenvalue
/// when `b` is a valid norm bound.
#[derive(Clone, Debug, Default)]
pub struct ShiftedPowerSolverF64;

impl EigenSolverF64 for ShiftedPowerSolverF64 {
    fn name(&self) -> &'static str {
        "shifted_power_image_residual_f64_v2"
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
        validate_native_solver_config(config)?;
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
        // ((1 + eps) I +/- A/b) has the same eigenvectors as b I +/- A,
        // without an absolute shift floor or an overflowing shifted image.
        let iteration_scale = if bound == 0.0 { 1.0 } else { bound };
        let shift = 1.0 + f64::EPSILON;
        let mut applications = 0usize;

        for iteration in 1..=config.stopping.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            problem.operator.apply(&x, &mut ax)?;
            applications += 1;
            for i in 0..n {
                transformed[i] = shift * x[i] + sign * (ax[i] / iteration_scale);
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
                    global_target_ordering_established: n == 1,
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
        let decomposition = checked_symmetric_decomposition_f64(t)?;
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
        "lanczos_full_reorthogonalization_image_residual_f64_v2"
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
        validate_native_solver_config(config)?;
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
            // A breakdown tolerance has the same units as A*q. In particular,
            // a tiny but nonzero matrix must not be compared to a unit floor.
            let breakdown_scale = norm(&z);
            if !breakdown_scale.is_finite() || breakdown_scale < 0.0 {
                return Err(SolverError::NumericalBreakdown(
                    "invalid Lanczos breakdown scale".into(),
                ));
            }
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
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(SolverError::NumericalBreakdown(
                    "non-finite Lanczos recurrence coefficient".into(),
                ));
            }
            alphas.push(alpha);

            let (_ritz_theta, vector) = self.ritz_pair(&alphas, &betas, &basis, largest)?;
            let (lambda, residual, relative, backward) =
                evaluate_eigenpair(problem.operator, &vector, &mut workspace)?;
            applications += 1;

            if iteration >= config.stopping.minimum_iterations
                && (residual <= absolute_residual || backward <= backward_tolerance)
            {
                let report = EigenpairReportF64 {
                    global_target_ordering_established: basis.len() == n,
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

            let breakdown_threshold = f64::EPSILON.sqrt() * breakdown_scale;
            if beta <= breakdown_threshold {
                let report = EigenpairReportF64 {
                    global_target_ordering_established: basis.len() == n,
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
    /// Absolute residual tolerance in operator units; accepted as an alternative
    /// to the dimensionless scaled backward-error tolerance.
    pub absolute_residual_tolerance: f64,
    pub scaled_backward_error_tolerance: f64,
    /// |lambda_new-lambda_old| / max(1, |lambda_new|, |lambda_old|).
    /// The unit floor makes this absolute for small eigenvalues.
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
/// Residual convergence and a separated Ritz boundary do not establish global
/// target ordering for a proper subspace. Even full-space identification here
/// is computed evidence; a rigorous index requires separate certification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockEigenReportF64 {
    pub target: EigenTarget,
    pub requested_count: usize,
    pub returned_count: usize,
    pub block_size: usize,
    pub invariant_subspaces: Vec<InvariantSubspaceF64>,
    /// Separation among the Ritz values in the visited subspace only.
    #[serde(default)]
    pub ritz_boundary_separation_established: bool,
    /// Full-space numerical selection was performed; this is not a certificate.
    #[serde(default)]
    pub global_target_ordering_established: bool,
    /// The requested boundary is separated in a full-space numerical solve.
    /// Legacy reports predate the explicit ordering field and must not be used
    /// to infer global ordering unless that field is also true.
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

// Version 3 uses relative rank tests and exact dyadic stability/cluster gates.
pub const BLOCK_SUBSPACE_CHECKPOINT_SCHEMA_VERSION: u32 = 3;

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
            || self.algorithm != "block_subspace_iteration_image_residual_f64_v2"
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
        let initial_norm = norm(&candidate);
        if !initial_norm.is_finite() || initial_norm == 0.0 {
            continue;
        }
        for value in &mut candidate {
            *value /= initial_norm;
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

// Arithmetic helpers for the native block route. These remain point estimates,
// but an intermediate overflow/underflow must not turn a residual into zero.
fn symmetric_average_f64(left: f64, right: f64) -> Result<f64, SolverError> {
    if !left.is_finite() || !right.is_finite() {
        return Err(SolverError::NumericalBreakdown(
            "projected operator contains non-finite arithmetic".into(),
        ));
    }
    let sum = left + right;
    Ok(if sum.is_finite() {
        0.5 * sum
    } else {
        0.5 * left + 0.5 * right
    })
}

fn residual_ratio_f64(residual: f64, bound: f64, eigenvalue_abs: f64) -> Result<f64, SolverError> {
    if !residual.is_finite()
        || !bound.is_finite()
        || bound < 0.0
        || !eigenvalue_abs.is_finite()
        || eigenvalue_abs < 0.0
    {
        return Err(SolverError::NumericalBreakdown(
            "invalid block residual arithmetic".into(),
        ));
    }
    let denominator = bound + eigenvalue_abs;
    let ratio = if denominator.is_normal() {
        residual / denominator
    } else {
        let scale = bound.max(eigenvalue_abs);
        if scale == 0.0 && residual == 0.0 {
            0.0
        } else {
            (residual / scale) / (bound / scale + eigenvalue_abs / scale)
        }
    };
    if !ratio.is_finite() || (ratio == 0.0 && residual != 0.0) {
        return Err(SolverError::NumericalBreakdown(
            "block residual ratio is not representable".into(),
        ));
    }
    Ok(ratio)
}

fn native_ritz_stable(current: f64, previous: f64, tolerance: f64) -> bool {
    exact_sturm_f64::difference_at_most_scaled(
        current,
        previous,
        tolerance,
        current.abs().max(previous.abs()).max(1.0),
    )
}

fn same_cluster(left: f64, right: f64, config: &BlockExtremeConfigF64) -> bool {
    exact_sturm_f64::difference_at_most_sum_scaled(
        left,
        right,
        config.cluster_absolute_tolerance,
        config.cluster_relative_tolerance,
        left.abs().max(right.abs()),
    )
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
        // A positive scalar multiple leaves the iterated subspace unchanged.
        // Normalize by the operator bound BEFORE adding the shift: an absolute
        // floor would erase small A, while b*I +/- A may overflow for large A.
        let iteration_scale = if norm_bound == 0.0 { 1.0 } else { norm_bound };
        let shift = 1.0 + f64::EPSILON.sqrt();
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
                    *value = shift * source + transform_sign * (*value / iteration_scale);
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
                && previous_values.as_ref().is_some_and(|previous| {
                    state.values.iter().zip(previous).all(|(&current, &prior)| {
                        native_ritz_stable(current, prior, config.ritz_value_stability_tolerance)
                    })
                });
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
                        algorithm: "block_subspace_iteration_image_residual_f64_v2".to_owned(),
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
                let value = symmetric_average_f64(left, right)?;
                projected[(row, column)] = value;
                projected[(column, row)] = value;
            }
        }
        let decomposition = checked_symmetric_decomposition_f64(projected)?;
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
            if !value.is_finite() {
                return Err(SolverError::NumericalBreakdown(
                    "Rayleigh-Ritz extraction produced a non-finite eigenvalue".into(),
                ));
            }
            let residual_vector: Vec<_> = applied
                .iter()
                .zip(&vector)
                .map(|(av, x)| av - value * x)
                .collect();
            let residual = norm(&residual_vector);
            let backward = residual_ratio_f64(residual, norm(&applied), value.abs())?;
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
                    let value = symmetric_average_f64(
                        dot(&basis[row], &state.applied_vectors[first + column]),
                        dot(&basis[column], &state.applied_vectors[first + row]),
                    )?;
                    projected[row * cluster_dimension + column] = value;
                    projected[column * cluster_dimension + row] = value;
                }
            }
            let residual_frobenius_norm = norm(&state.residual_norms[first..last]);
            if !residual_frobenius_norm.is_finite() {
                return Err(SolverError::NumericalBreakdown(
                    "block residual Frobenius norm is not representable".into(),
                ));
            }
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
            let lower = minimum_value - residual_frobenius_norm;
            let upper = maximum_value + residual_frobenius_norm;
            if !lower.is_finite() || !upper.is_finite() {
                return Err(SolverError::NumericalBreakdown(
                    "block residual spectral window is not representable".into(),
                ));
            }
            invariant_subspaces.push(InvariantSubspaceF64 {
                dimension: cluster_dimension,
                ritz_values: state.values[first..last].to_vec(),
                basis,
                projected_operator_row_major: projected,
                maximum_residual_norm,
                residual_frobenius_norm,
                spectral_window: ResidualSpectralWindowF64 {
                    lower,
                    upper,
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
        } else if maximum_residual_norm <= config.absolute_residual_tolerance {
            (
                ResultStatus::Converged,
                TerminationReason::ResidualTolerance,
            )
        } else {
            (
                ResultStatus::Converged,
                TerminationReason::ResidualOrBackwardErrorTolerance,
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
            ritz_boundary_separation_established: boundary_separation_established,
            global_target_ordering_established: config.block_size == operator_dimension,
            target_boundary_separation_established: boundary_separation_established
                && config.block_size == operator_dimension,
            maximum_residual_norm,
            maximum_scaled_backward_error,
            ritz_value_stability: stability,
            orthogonality_defect,
            iterations,
            operator_applications: applications,
            estimated_peak_memory_bytes: scalar_vectors.saturating_mul(8),
            algorithm: "block_subspace_iteration_image_residual_f64_v2".to_owned(),
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
    fn block_arithmetic_preserves_range_and_rejects_nonfinite_inputs() {
        // Exact exponent scaling supplies independent dimensionless references.
        let scale = 2.0f64.powi(-537);
        assert!((norm(&[1.25 * scale; 2]) / scale - 1.25 * 2.0f64.sqrt()).abs() < 1e-15);
        let tiny = f64::from_bits(1);
        assert!((residual_ratio_f64(tiny, tiny, 2.0 * tiny).unwrap() - 1.0 / 3.0).abs() < 1e-15);
        assert!(
            (residual_ratio_f64(f64::MAX / 4.0, f64::MAX, f64::MAX).unwrap() - 0.125).abs() < 1e-15
        );
        assert_eq!(symmetric_average_f64(f64::MAX, f64::MAX).unwrap(), f64::MAX);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(residual_ratio_f64(bad, 1.0, 1.0).is_err());
            assert!(symmetric_average_f64(bad, 1.0).is_err());
        }
        assert!(residual_ratio_f64(tiny, f64::MAX, f64::MAX).is_err());
    }

    #[test]
    fn block_checkpoint_rejects_prior_arithmetic_version() {
        let operator = DiagonalF64::new("old-checkpoint", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let config = block_config(EigenTarget::AlgebraicLargest, 1, 2);
        let solver = BlockSubspaceIterationF64::default();
        let mut stop = StopAtIteration { iteration: 1 };
        let BlockSolveOutcomeF64::Checkpointed { mut checkpoint } = solver
            .solve_checkpointed(
                &problem,
                &config,
                "old-arithmetic",
                None,
                None,
                &mut stop,
                &CancellationToken::new(),
            )
            .unwrap()
        else {
            panic!("expected checkpoint")
        };
        checkpoint.schema_version = 1;
        assert!(checkpoint
            .validate_compatibility(&problem, &config, "old-arithmetic", 2)
            .is_err());
    }

    #[test]
    fn block_iteration_returns_several_extremes_without_materializing() {
        let operator = DiagonalF64::new("diagonal", vec![-3.0, -1.0, 2.0, 4.0, 7.0, 11.0]).unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let report = BlockSubspaceIterationF64::default()
            .solve(&problem, &block_config(EigenTarget::AlgebraicLargest, 3, 4))
            .unwrap();

        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.ritz_boundary_separation_established);
        assert!(!report.global_target_ordering_established);
        assert!(!report.target_boundary_separation_established);
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
    /// Relative to the largest absolute materialized entry; no unit scale floor.
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
        "dense_materialized_image_residual_reference_f64_v2"
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
        validate_native_solver_config(config)?;
        let largest = supported_extreme(&config.target)?;
        let (absolute_residual, backward_tolerance) = stopping_thresholds_f64(config)?;
        let n = problem.operator.dimension();
        if !self.symmetry_tolerance.is_finite() || self.symmetry_tolerance < 0.0 {
            return Err(SolverError::InvalidConfiguration(
                "symmetry tolerance must be finite and nonnegative".into(),
            ));
        }
        let shape_valid = n
            .checked_mul(n)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
        if n == 0 || n > self.maximum_dimension || !shape_valid {
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
            if column.iter().any(|v| !v.is_finite()) {
                return Err(SolverError::NumericalBreakdown(
                    "materialized operator contains nonfinite entries".into(),
                ));
            }
            e[j] = 0.0;
            for i in 0..n {
                matrix[(i, j)] = column[i];
            }
        }
        let symmetry_scale = matrix.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        for i in 0..n {
            check_solver_cancellation(cancellation)?;
            for j in 0..i {
                if !exact_sturm_f64::difference_at_most_scaled(
                    matrix[(i, j)],
                    matrix[(j, i)],
                    self.symmetry_tolerance,
                    symmetry_scale,
                ) {
                    return Err(SolverError::NumericalBreakdown(format!(
                        "materialized operator is not symmetric at ({i}, {j})"
                    )));
                }
                let average = symmetric_average_f64(matrix[(i, j)], matrix[(j, i)])?;
                matrix[(i, j)] = average;
                matrix[(j, i)] = average;
            }
        }
        let decomposition = checked_symmetric_decomposition_f64(matrix)?;
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
            global_target_ordering_established: true,
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

fn require_independent_solver_routes(
    primary: SolverRoute,
    independent: SolverRoute,
    precision_bits: u32,
) -> Result<(), SolverError> {
    let assessment = xc_core::assess_route_independence(
        &primary.evidence(precision_bits, None),
        &independent.evidence(precision_bits, None),
        &xc_core::IndependenceDeclaration {
            intended_claim: "agreement on the requested ordered eigenpair".into(),
            rationale: "distinct registered solver formulations with no shared decisive seed"
                .into(),
            accepted_shared_inputs: BTreeSet::new(),
        },
    );
    if !assessment.independent {
        return Err(SolverError::CrossCheckDisagreement(format!(
            "solver routes are not independent: {}",
            assessment.reasons.join("; ")
        )));
    }
    Ok(())
}

fn native_implementation_route(solver: &dyn EigenSolverF64) -> Result<SolverRoute, SolverError> {
    let implementation = solver.type_id();
    if implementation == std::any::TypeId::of::<DenseReferenceSolverF64>() {
        Ok(SolverRoute::DenseFullSpectrumReference)
    } else if implementation == std::any::TypeId::of::<ShiftedPowerSolverF64>() {
        Ok(SolverRoute::ShiftedPowerExtremeReference)
    } else if implementation == std::any::TypeId::of::<LanczosSolverF64>() {
        Ok(SolverRoute::LanczosExtremeReference)
    } else {
        Err(SolverError::CrossCheckDisagreement(
            "unregistered concrete solver implementation cannot establish executable-route independence".into(),
        ))
    }
}

pub const NATIVE_CROSSCHECK_SEMANTICS: &str = "native_concrete_routes_fresh_image_agreement_v2";

/// Compare converged finite reports from caller-selected independent routes.
/// Registered implementation identities and structured route evidence must
/// establish independence, and at least one route must establish full-space
/// target ordering. The accepted report is the ordered route when only one
/// has ordering evidence. Agreement alone does not certify residuals, state selection,
/// or a mathematical eigenvalue. Vector overlap is diagnostic only, allowing
/// different bases in a repeated eigenspace. `tolerance` is relative to the
/// larger of both eigenvalue magnitudes and the operator's norm bound. Acceptance
/// also requires agreement at the scale of freshly applied normalized vectors,
/// so a loose caller-provided norm bound cannot weaken numerical agreement.
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
    // Concrete Rust type provenance is checked before caller-supplied execution.
    let primary_route = native_implementation_route(primary)?;
    let independent_route = native_implementation_route(independent)?;
    require_independent_solver_routes(primary_route, independent_route, 53)?;
    let a = primary.solve_controlled(problem, config, cancellation)?;
    check_solver_cancellation(cancellation)?;
    let b = independent.solve_controlled(problem, config, cancellation)?;
    check_solver_cancellation(cancellation)?;
    a.validate_finite()?;
    b.validate_finite()?;
    if primary.name().trim().is_empty()
        || independent.name().trim().is_empty()
        || primary.name() == independent.name()
    {
        return Err(SolverError::InvalidConfiguration("cross-check requires distinct identified routes; the caller must establish their independence".into()));
    }
    if primary.name() != a.algorithm || independent.name() != b.algorithm {
        return Err(SolverError::CrossCheckDisagreement(
            "solver wrapper identity does not match its reported implementation".into(),
        ));
    }
    if !a.global_target_ordering_established && !b.global_target_ordering_established {
        return Err(SolverError::CrossCheckDisagreement(
            "neither route establishes the requested global target ordering".into(),
        ));
    }
    for report in [&a, &b] {
        if report.status != ResultStatus::Converged
            || report.eigenvector.len() != problem.operator.dimension()
            || report.eigenvector.is_empty()
            || report.residual_norm < 0.0
            || report.relative_residual < 0.0
            || report.scaled_backward_error < 0.0
        {
            return Err(SolverError::CrossCheckDisagreement("cross-check requires converged reports with matching dimensions and nonnegative diagnostics".into()));
        }
    }
    let normalize_report = |values: &[f64]| -> Result<Vec<f64>, SolverError> {
        let maximum = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if maximum == 0.0 {
            return Err(SolverError::CrossCheckDisagreement(
                "cross-check received a zero eigenvector".into(),
            ));
        }
        let mut scaled: Vec<_> = values.iter().map(|v| v / maximum).collect();
        let length = norm(&scaled);
        for v in &mut scaled {
            *v /= length;
        }
        Ok(scaled)
    };
    let av = normalize_report(&a.eigenvector)?;
    let bv = normalize_report(&b.eigenvector)?;
    // A unit floor made the tolerance absolute below magnitude one, so
    // disagreeing eigenvalues of any small-norm operator were cross-checked.
    let operator_scale = problem
        .operator
        .norm_bound()
        .filter(|bound| bound.is_finite() && *bound > 0.0)
        .unwrap_or(0.0);
    let scale = a
        .eigenvalue
        .abs()
        .max(b.eigenvalue.abs())
        .max(operator_scale);
    let mut a_image = vec![0.0; av.len()];
    let mut b_image = vec![0.0; bv.len()];
    problem.operator.apply(&av, &mut a_image)?;
    problem.operator.apply(&bv, &mut b_image)?;
    let image_scale = a
        .eigenvalue
        .abs()
        .max(b.eigenvalue.abs())
        .max(norm(&a_image))
        .max(norm(&b_image));
    if !image_scale.is_finite() {
        return Err(SolverError::NumericalBreakdown(
            "nonfinite cross-check action scale".into(),
        ));
    }
    let difference = (a.eigenvalue - b.eigenvalue).abs();
    if !difference.is_finite() {
        return Err(SolverError::CrossCheckDisagreement(
            "cross-check difference exceeds finite binary64 range".into(),
        ));
    }
    let overlap = dot(&av, &bv).abs().min(1.0);
    let overlap_sq = overlap * overlap;
    if !exact_sturm_f64::difference_at_most_scaled(a.eigenvalue, b.eigenvalue, tolerance, scale)
        || !exact_sturm_f64::difference_at_most_scaled(
            a.eigenvalue,
            b.eigenvalue,
            tolerance,
            image_scale,
        )
    {
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
    let (mut accepted, independent_report) = if a.global_target_ordering_established {
        (a, b)
    } else {
        (b, a)
    };
    accepted.assurance = AssuranceLevel::CrossChecked;
    Ok(CrossCheckedEigenpairF64 {
        accepted,
        independent: independent_report,
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
        let expected_backward = expected_residual
            / (applied_norm + expected_value.abs() * vector_norm).max(f64::MIN_POSITIVE);

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
    fn tiny_native_residual_is_not_erased_by_squaring() {
        let operator = DiagonalF64::new("tiny", vec![1e-200, 2e-200]).unwrap();
        let vector = [std::f64::consts::FRAC_1_SQRT_2; 2];
        let (value, residual, relative, _) =
            evaluate_eigenpair(&operator, &vector, &mut [0.0; 2]).unwrap();
        assert!((value / 1e-200 - 1.5).abs() < 1e-14);
        assert!(
            (residual / 1e-200 - 0.5).abs() < 1e-14,
            "nonzero residual was erased: {residual}"
        );
        assert!(relative > 0.1);
        let mut settings = config(EigenTarget::AlgebraicSmallest);
        settings.stopping.absolute_residual = xc_core::DecimalLiteral::new("1e-250").unwrap();
        settings.stopping.scaled_backward_error = xc_core::DecimalLiteral::new("1e-20").unwrap();
        settings.stopping.maximum_iterations = 2;
        // With the scale-correct shift, the two-eigenvalue example legitimately
        // converges in two steps: the unwanted extreme is nearly annihilated.
        let report = ShiftedPowerSolverF64
            .solve(&SymmetricProblemF64::new(&operator), &settings)
            .unwrap();
        assert!((report.eigenvalue / 1e-200 - 1.0).abs() < 1e-14);
        let scaled_residual = report
            .eigenvector
            .iter()
            .enumerate()
            .map(|(i, x)| ((i as f64 + 1.0 - report.eigenvalue / 1e-200) * x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(scaled_residual / 3.0 <= 1e-20);
        // A third distinct eigenvalue prevents that annihilation; two steps
        // must still reject convergence instead of squaring the residual away.
        let three = DiagonalF64::new("tiny-three", vec![1e-200, 2e-200, 3e-200]).unwrap();
        assert!(ShiftedPowerSolverF64
            .solve(&SymmetricProblemF64::new(&three), &settings)
            .is_err());
    }

    #[test]
    fn native_crosscheck_rejects_malformed_reports() {
        struct BadReport;
        impl EigenSolverF64 for BadReport {
            fn name(&self) -> &'static str {
                "malformed-fixture"
            }
            fn solve(
                &self,
                problem: &SymmetricProblemF64<'_>,
                config: &SolverConfig,
            ) -> Result<EigenpairReportF64, SolverError> {
                let mut report = DenseReferenceSolverF64::default().solve(problem, config)?;
                report.eigenvalue = f64::NAN;
                Ok(report)
            }
        }
        let operator = DiagonalF64::new("diag", vec![1.0, 2.0]).unwrap();
        assert!(cross_check_f64(
            &BadReport,
            &DenseReferenceSolverF64::default(),
            &SymmetricProblemF64::new(&operator),
            &config(EigenTarget::AlgebraicSmallest),
            1e-10
        )
        .is_err());
    }

    #[test]
    fn native_crosscheck_rejects_renamed_implementation_and_checks_small_scale() {
        // Two reports ten percent apart on a norm-2e-12 operator passed the
        // former max(|lambda|, 1) scale as a 1e-13 absolute difference.
        struct Offset(f64);
        impl EigenSolverF64 for Offset {
            fn name(&self) -> &'static str {
                "offset-fixture"
            }
            fn solve(
                &self,
                problem: &SymmetricProblemF64<'_>,
                config: &SolverConfig,
            ) -> Result<EigenpairReportF64, SolverError> {
                let mut report = DenseReferenceSolverF64::default().solve(problem, config)?;
                report.eigenvalue *= self.0;
                Ok(report)
            }
        }
        let operator = DiagonalF64::new("small", vec![1e-12, 2e-12]).unwrap();
        let problem = SymmetricProblemF64::new(&operator);
        let target = config(EigenTarget::AlgebraicSmallest);
        let dense = DenseReferenceSolverF64::default();
        assert!(cross_check_f64(&Offset(1.1), &dense, &problem, &target, 1e-8).is_err());
        // Matching numbers from a renamed copy are still not independent.
        assert!(cross_check_f64(&Offset(1.0), &dense, &problem, &target, 1e-8).is_err());
        let mut precise = target;
        precise.stopping.absolute_residual = xc_core::DecimalLiteral::new("1e-25").unwrap();
        precise.stopping.scaled_backward_error = xc_core::DecimalLiteral::new("1e-13").unwrap();
        assert!(cross_check_f64(
            &LanczosSolverF64::default(),
            &dense,
            &problem,
            &precise,
            1e-8
        )
        .is_ok());
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
    .map_err(|error| {
        if error
            .downcast_ref::<xc_numerics::eigen::SelectedEigenvalueIterationLimit>()
            .is_some()
        {
            SolverError::IterationBudgetExhausted(error.to_string())
        } else {
            SolverError::NonConvergence(error.to_string())
        }
    })
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
    .map_err(|error| {
        if error
            .downcast_ref::<xc_numerics::eigen::SelectedEigenvalueIterationLimit>()
            .is_some()
        {
            SolverError::IterationBudgetExhausted(error.to_string())
        } else {
            SolverError::NonConvergence(error.to_string())
        }
    })
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
    let mut precision_bits = options
        .precision
        .initial_working_bits()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    if !(33..=1_000_000).contains(&precision_bits) || options.precision.maximum_bits > 1_000_000 {
        return Err(SolverError::InvalidConfiguration(
            "adaptive selected precision must be in 33..=1000000 bits".into(),
        ));
    }
    let mut attempts = Vec::new();
    let mut last_result = None;
    loop {
        let attempt_options = HpSelectedTridiagonalEigenpairOptions {
            first_index: options.first_index,
            last_index: options.last_index,
            absolute_tolerance: hp_positive_threshold(
                &options.absolute_tolerance,
                precision_bits,
                "selected eigenvalue tolerance",
                rug::float::Round::Down,
            )?,
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
            Err(error @ SolverError::IterationBudgetExhausted(_)) => {
                attempts.push(HpSelectedPrecisionAttempt {
                    precision_bits,
                    status: ResultStatus::Inconclusive,
                    selected_items: 0,
                    vector_recoveries: 0,
                    inverse_iteration_runs: 0,
                    reason: error.to_string(),
                });
                return Ok(HpAdaptiveSelectedTridiagonalResult::Inconclusive {
                    last_result,
                    attempts,
                    reason: error.to_string(),
                });
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
    literal
        .validate()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    hp_parse_string(literal.as_str(), precision_bits)
}

#[cfg(feature = "hp-reference")]
fn hp_parse_literal_round(
    literal: &xc_core::DecimalLiteral,
    precision_bits: u32,
    round: rug::float::Round,
) -> Result<rug::Float, SolverError> {
    // Retain the common syntax, precision, finite range and nonzero checks.
    hp_parse_literal(literal, precision_bits)?;
    let parsed = rug::Float::parse(literal.as_str())
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    let value = rug::Float::with_val_round(precision_bits, parsed, round).0;
    if !value.is_finite()
        || (value.is_zero()
            && literal
                .canonical()
                .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?
                .as_str()
                != "0")
    {
        return Err(SolverError::InvalidConfiguration(
            "directed HP scalar is outside the supported range".into(),
        ));
    }
    Ok(value)
}

#[cfg(feature = "hp-reference")]
fn hp_positive_threshold(
    literal: &xc_core::DecimalLiteral,
    precision_bits: u32,
    name: &str,
    round: rug::float::Round,
) -> Result<rug::Float, SolverError> {
    let value = hp_parse_literal_round(literal, precision_bits, round)?;
    if value <= 0 {
        return Err(SolverError::InvalidConfiguration(format!(
            "{name} must be positive"
        )));
    }
    Ok(value)
}

#[cfg(feature = "hp-reference")]
fn hp_checked_action<O>(
    operator: &O,
    vector: &[rug::Float],
    precision_bits: u32,
) -> Result<Vec<rug::Float>, SolverError>
where
    O: xc_operator::LinearOperator<rug::Float> + ?Sized,
{
    let mut output = vec![hp_zero(precision_bits); vector.len()];
    operator.apply(vector, &mut output)?;
    for value in &mut output {
        if value.prec() < precision_bits {
            return Err(SolverError::InvalidConfiguration(format!(
                "HP operator returned {}-bit arithmetic for a {precision_bits}-bit solve",
                value.prec()
            )));
        }
        reprecision_hp_value(value, precision_bits);
        if !value.is_finite() {
            return Err(SolverError::NumericalBreakdown(
                "HP operator action is nonfinite at the requested precision".into(),
            ));
        }
    }
    Ok(output)
}

#[cfg(feature = "hp-reference")]
fn hp_norm(values: &[rug::Float], precision_bits: u32) -> rug::Float {
    // Scale every component by the largest before squaring, so an individually
    // underflowing square is never dropped beside a larger one.
    if values.iter().any(|v| !v.is_finite()) {
        return rug::Float::with_val(precision_bits, rug::float::Special::Nan);
    }
    let maximum = values
        .iter()
        .map(|v| v.clone().abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or_else(|| hp_zero(precision_bits));
    if maximum.is_zero() {
        return rug::Float::with_val(precision_bits, maximum);
    }
    let mut sum = hp_zero(precision_bits);
    for value in values {
        let mut scaled = rug::Float::with_val(precision_bits, value / &maximum);
        scaled.square_mut();
        sum += scaled;
    }
    sum.sqrt_mut();
    sum *= maximum;
    sum
}

/// Euclidean norm of `values` with every operation rounded in `round`, each
/// magnitude scaled by the largest before squaring: an upper (`Up`) or lower
/// (`Down`) bound of the norm of the given components.
#[cfg(feature = "hp-reference")]
fn hp_directed_norm(
    values: &[rug::Float],
    precision_bits: u32,
    round: rug::float::Round,
) -> rug::Float {
    use rug::ops::{AddAssignRound, MulAssignRound};
    use rug::Float;
    if values.iter().any(|v| !v.is_finite()) {
        return Float::with_val(precision_bits, rug::float::Special::Nan);
    }
    let Some(scale) = values
        .iter()
        .map(|v| v.clone().abs())
        .max_by(|left, right| left.total_cmp(right))
    else {
        return hp_zero(precision_bits);
    };
    if scale.is_zero() {
        return hp_zero(precision_bits);
    }
    let mut sum = hp_zero(precision_bits);
    for value in values {
        let magnitude = value.clone().abs();
        let mut ratio = Float::with_val_round(precision_bits, &magnitude / &scale, round).0;
        ratio.square_round(round);
        sum.add_assign_round(&ratio, round);
    }
    sum.sqrt_round(round);
    sum.mul_assign_round(&scale, round);
    sum
}

/// The actual computed scale and individual stopping tests for a stored Ritz pair.
/// The working-unit scale is one binary working unit times the image denominator,
/// not a rigorous error bound for matrix assembly or an arbitrary callback.
#[cfg(feature = "hp-reference")]
#[derive(Clone, Debug)]
pub struct HpResidualAcceptance {
    pub absolute_residual_tolerance: rug::Float,
    pub scaled_backward_error_tolerance: rug::Float,
    pub image_norm_denominator: rug::Float,
    pub working_unit_scale: Option<rug::Float>,
    pub absolute_residual_passed: bool,
    pub scaled_backward_error_passed: bool,
}

#[cfg(feature = "hp-reference")]
#[allow(clippy::too_many_arguments)]
fn hp_residual_acceptance(
    left: &[rug::Float],
    right: &[rug::Float],
    eigenvalue: &rug::Float,
    residual: &rug::Float,
    backward: &rug::Float,
    absolute_tolerance: &rug::Float,
    backward_tolerance: &rug::Float,
    p: u32,
) -> HpResidualAcceptance {
    let denominator = hp_norm(left, p) + eigenvalue.clone().abs() * hp_norm(right, p);
    let unit = rug::Float::with_val(p, &denominator) >> p;
    HpResidualAcceptance {
        absolute_residual_tolerance: absolute_tolerance.clone(),
        scaled_backward_error_tolerance: backward_tolerance.clone(),
        working_unit_scale: (!unit.is_zero() || denominator.is_zero()).then_some(unit),
        image_norm_denominator: denominator,
        absolute_residual_passed: residual <= absolute_tolerance,
        scaled_backward_error_passed: backward <= backward_tolerance,
    }
}

/// A conservative working-scale cluster floor. This reports unresolved point
/// resolution; it never replaces the source-bound multiplicity count.
#[cfg(feature = "hp-reference")]
fn hp_effective_cluster_tolerance<'a>(
    requested: &rug::Float,
    values: impl Iterator<Item = &'a rug::Float>,
    dimension: usize,
    p: u32,
) -> rug::Float {
    let mut scale = rug::Float::with_val(p, 0);
    for value in values {
        let a = value.clone().abs();
        if a > scale {
            scale = a;
        }
    }
    scale >>= p;
    scale *= dimension.saturating_mul(8);
    if scale > *requested {
        scale
    } else {
        requested.clone()
    }
}

/// Euclidean residual and relative residual for stored images: an upper bound
/// of the norm of the stored residual components over a lower bound of the
/// stored images' scale. It does not bound operator application error.
#[cfg(feature = "hp-reference")]
fn hp_residual_measures(
    residual: &[rug::Float],
    left_image: &[rug::Float],
    right_image: &[rug::Float],
    eigenvalue: &rug::Float,
    precision_bits: u32,
) -> Result<(rug::Float, rug::Float), SolverError> {
    let invalid = || {
        SolverError::NumericalBreakdown(
            "HP residual diagnostics are nonfinite or outside the representable range".into(),
        )
    };
    if residual.is_empty()
        || residual.len() != left_image.len()
        || residual.len() != right_image.len()
        || !eigenvalue.is_finite()
    {
        return Err(invalid());
    }
    // An upper bound of the residual norm over a lower bound of its scale, so
    // the rounding of these norms can only make acceptance stricter.
    let residual_norm = hp_directed_norm(residual, precision_bits, rug::float::Round::Up);
    let left_norm = hp_directed_norm(left_image, precision_bits, rug::float::Round::Down);
    let right_norm = hp_directed_norm(right_image, precision_bits, rug::float::Round::Down);
    for (values, norm) in [
        (residual, &residual_norm),
        (left_image, &left_norm),
        (right_image, &right_norm),
    ] {
        if !norm.is_finite() || (norm.is_zero() && values.iter().any(|v| !v.is_zero())) {
            return Err(invalid());
        }
    }
    if right_norm.is_zero() {
        return Err(invalid());
    }
    use rug::float::Round;
    use rug::ops::{AddAssignRound, MulAssignRound};
    // The scale is used as a lower bound; its upward rounding detects overflow,
    // which directed downward rounding would otherwise clamp to a finite value.
    let directed_scale = |round: Round| {
        let mut scale = rug::Float::with_val(precision_bits, eigenvalue.abs_ref());
        scale.mul_assign_round(&right_norm, round);
        scale.add_assign_round(&left_norm, round);
        scale
    };
    let scale = directed_scale(Round::Down);
    if !directed_scale(Round::Up).is_finite() || (scale.is_zero() && !residual_norm.is_zero()) {
        return Err(invalid());
    }
    let mut relative = residual_norm.clone();
    if !scale.is_zero() {
        // A ratio that underflows is refused, not reported as the least
        // positive value; otherwise the upward rounding is reported.
        if rug::Float::with_val_round(precision_bits, &residual_norm / &scale, Round::Down)
            .0
            .is_zero()
            && !residual_norm.is_zero()
        {
            return Err(invalid());
        }
        relative = rug::Float::with_val_round(precision_bits, &residual_norm / &scale, Round::Up).0;
    }
    if !relative.is_finite() || (relative.is_zero() && !residual_norm.is_zero()) {
        return Err(invalid());
    }
    Ok((residual_norm, relative))
}

#[cfg(feature = "hp-reference")]
fn hp_ritz_change(
    current: &rug::Float,
    previous: &rug::Float,
    scale: Option<&rug::Float>,
) -> rug::Float {
    use rug::{float::Round, Float};
    let precision = current.prec();
    let difference = if current >= previous {
        Float::with_val_round(precision, current - previous, Round::Up).0
    } else {
        Float::with_val_round(precision, previous - current, Round::Up).0
    };
    if let Some(scale) = scale {
        Float::with_val_round(precision, &difference / scale, Round::Up).0
    } else {
        difference
    }
}

#[cfg(feature = "hp-reference")]
fn hp_block_termination<'a>(
    diagnostics: impl IntoIterator<Item = (&'a rug::Float, &'a rug::Float)>,
    absolute_tolerance: &rug::Float,
    backward_tolerance: &rug::Float,
) -> TerminationReason {
    let (all_absolute, all_backward) =
        diagnostics
            .into_iter()
            .fold((true, true), |(absolute, backward), (residual, error)| {
                (
                    absolute && residual <= absolute_tolerance,
                    backward && error <= backward_tolerance,
                )
            });
    if all_backward {
        TerminationReason::BackwardErrorTolerance
    } else if all_absolute {
        TerminationReason::ResidualTolerance
    } else {
        TerminationReason::ResidualOrBackwardErrorTolerance
    }
}

#[cfg(feature = "hp-reference")]
/// Exact product of stored values. Mantissas are multiplied at a precision
/// holding every operand bit and the exponents are added separately, so no
/// intermediate product can leave the exponent range; a product whose exact
/// value is outside the range is an error, never a rounded or zero term.
pub(crate) fn exact_product(factors: &[&rug::Float]) -> Result<rug::Float, SolverError> {
    let out_of_range = || {
        SolverError::NumericalBreakdown(
            "HP residual product is outside the MPFR exponent range".into(),
        )
    };
    let precision = factors.iter().map(|factor| factor.prec()).sum::<u32>();
    let mut mantissa = rug::Float::with_val(precision, 1);
    let mut exponent = 0i64;
    for factor in factors {
        if factor.is_zero() {
            return Ok(rug::Float::with_val(precision, 0));
        }
        let Some(factor_exponent) = factor.get_exp() else {
            return Err(out_of_range());
        };
        let mut normalized = (*factor).clone();
        normalized >>= factor_exponent;
        mantissa *= &normalized;
        exponent += i64::from(factor_exponent);
    }
    let Some(mantissa_exponent) = mantissa.get_exp() else {
        return Err(out_of_range());
    };
    let expected = exponent + i64::from(mantissa_exponent);
    let shift = i32::try_from(exponent).map_err(|_| out_of_range())?;
    mantissa <<= shift;
    if mantissa.get_exp().map(i64::from) != Some(expected) {
        return Err(out_of_range());
    }
    Ok(mantissa)
}

#[cfg(feature = "hp-reference")]
/// Downward and upward roundings to `precision_bits` of the exact sum of
/// `terms`: an enclosure of that sum.
pub(crate) fn signed_bounds<'a, I: Iterator<Item = &'a rug::Float>>(
    terms: impl Fn() -> I,
    precision_bits: u32,
) -> (rug::Float, rug::Float) {
    (
        rug::Float::with_val_round(
            precision_bits,
            rug::Float::sum(terms()),
            rug::float::Round::Down,
        )
        .0,
        rug::Float::with_val_round(
            precision_bits,
            rug::Float::sum(terms()),
            rug::float::Round::Up,
        )
        .0,
    )
}

#[cfg(feature = "hp-reference")]
/// Lower and upper bounds of the magnitude of the exact sum of `terms`, from
/// its downward and upward roundings to `precision_bits`.
pub(crate) fn magnitude_bounds<'a, I: Iterator<Item = &'a rug::Float>>(
    terms: impl Fn() -> I,
    precision_bits: u32,
) -> (rug::Float, rug::Float) {
    let (down, up) = signed_bounds(terms, precision_bits);
    if down.is_sign_positive() && !down.is_zero() {
        (down, up)
    } else if up.is_sign_negative() && !up.is_zero() {
        (
            rug::Float::with_val(precision_bits, -up),
            rug::Float::with_val(precision_bits, -down),
        )
    } else {
        let upper = rug::Float::with_val(precision_bits, down.abs_ref())
            .max(&rug::Float::with_val(precision_bits, up.abs_ref()));
        (rug::Float::with_val(precision_bits, 0), upper)
    }
}

#[cfg(feature = "hp-reference")]
/// Sum of `terms` with a single rounding to `precision_bits`.
pub(crate) fn rounded_once_sum<'a>(
    terms: impl Iterator<Item = &'a rug::Float>,
    precision_bits: u32,
) -> rug::Float {
    rug::Float::with_val(precision_bits, rug::Float::sum(terms))
}

#[cfg(feature = "hp-reference")]
fn hp_decimal(value: &rug::Float, significant_digits: usize) -> String {
    value.to_string_radix(10, Some(significant_digits))
}

#[cfg(feature = "hp-reference")]
fn map_hp_recovery_error(
    failure: Option<&xc_numerics::eigen::HpEigenvectorRecoveryFailure>,
    message: String,
) -> SolverError {
    use xc_numerics::eigen::HpEigenvectorRecoveryFailure;
    match failure {
        Some(HpEigenvectorRecoveryFailure::InvalidConfiguration(message)) => {
            SolverError::InvalidConfiguration(message.clone())
        }
        Some(HpEigenvectorRecoveryFailure::UnresolvedEigenspace(message)) => {
            SolverError::UnresolvedEigenspace(message.clone())
        }
        Some(HpEigenvectorRecoveryFailure::IterationLimit { .. }) => {
            SolverError::IterationBudgetExhausted(message)
        }
        None => SolverError::NumericalBreakdown(message),
    }
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
        dense_symmetric_eigenpair_at_index_hp, dense_symmetric_eigenvalues_hp,
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
    let precision_bits = config
        .precision
        .initial_working_bits()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
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
    let recovered = dense_symmetric_eigenpair_at_index_hp(
        problem.matrix,
        problem.dimension,
        index,
        precision_bits,
        config.stopping.maximum_iterations,
    )
    .map_err(|error| map_hp_recovery_error(error.downcast_ref(), error.to_string()))?;
    let eigenvalue = recovered.eigenvalue;
    let eigenvector = recovered.eigenvector;
    check_solver_cancellation(cancellation)?;

    // Verify the returned pair against the stored matrix. Every product is
    // exact and each component is enclosed by directed rounding of its exact
    // sum, so contributions smaller than the working precision survive.
    // Acceptance uses an upper bound of the residual norm over lower bounds of
    // its denominators; all reported bounds are rounded up.
    use rug::float::Round;
    use rug::ops::{AddAssignRound, MulAssignRound};
    let dimension = problem.dimension;
    let verification_bits = problem
        .matrix
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(precision_bits)
        .max(precision_bits)
        .saturating_add(64);
    let negative_eigenvalue = Float::with_val(eigenvalue.prec(), -&eigenvalue);
    let mut residual_upper = Vec::with_capacity(dimension);
    let mut applied_lower = Vec::with_capacity(dimension);
    for row in 0..dimension {
        check_solver_cancellation(cancellation)?;
        let mut terms = (0..dimension)
            .map(|column| {
                exact_product(&[
                    &problem.matrix[row * dimension + column],
                    &eigenvector[column],
                ])
            })
            .collect::<Result<Vec<Float>, _>>()?;
        applied_lower.push(magnitude_bounds(|| terms.iter(), verification_bits).0);
        terms.push(exact_product(&[&negative_eigenvalue, &eigenvector[row]])?);
        residual_upper.push(magnitude_bounds(|| terms.iter(), verification_bits).1);
    }
    let residual_norm = hp_directed_norm(&residual_upper, verification_bits, Round::Up);
    let applied_norm = hp_directed_norm(&applied_lower, verification_bits, Round::Down);
    let vector_norm = hp_directed_norm(&eigenvector, verification_bits, Round::Down);
    if [&residual_norm, &applied_norm, &vector_norm, &eigenvalue]
        .iter()
        .any(|v| !v.is_finite())
        || vector_norm.is_zero()
    {
        return Err(SolverError::NumericalBreakdown(
            "invalid HP eigenpair diagnostic norm".into(),
        ));
    }
    let eigenvalue_abs = Float::with_val(verification_bits, eigenvalue.abs_ref());
    let mut denominator = eigenvalue_abs.clone();
    denominator.mul_assign_round(&vector_norm, Round::Down);
    denominator.add_assign_round(&applied_norm, Round::Down);
    let mut infinity_bound = hp_zero(verification_bits);
    for row in 0..dimension {
        check_solver_cancellation(cancellation)?;
        let row_sum = Float::with_val_round(
            verification_bits,
            Float::sum(
                problem.matrix[row * dimension..(row + 1) * dimension]
                    .iter()
                    .map(|entry| entry.clone().abs())
                    .collect::<Vec<_>>()
                    .iter(),
            ),
            Round::Down,
        )
        .0;
        if row_sum > infinity_bound {
            infinity_bound = row_sum;
        }
    }
    let mut backward_denominator = infinity_bound;
    backward_denominator.mul_assign_round(&vector_norm, Round::Down);
    let mut eigen_term = eigenvalue_abs;
    eigen_term.mul_assign_round(&vector_norm, Round::Down);
    backward_denominator.add_assign_round(&eigen_term, Round::Down);
    let upper_ratio = |denominator: &Float| {
        if denominator.is_zero() {
            residual_norm.clone()
        } else {
            Float::with_val_round(verification_bits, &residual_norm / denominator, Round::Up).0
        }
    };
    let relative_residual = upper_ratio(&denominator);
    let scaled_backward_error = upper_ratio(&backward_denominator);
    if [&relative_residual, &scaled_backward_error]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(SolverError::NumericalBreakdown(
            "HP diagnostic ratio is not representable".into(),
        ));
    }
    let minus_one = Float::with_val(2, -1);
    let squares = eigenvector
        .iter()
        .map(|component| exact_product(&[component, component]))
        .collect::<Result<Vec<Float>, _>>()?;
    // Upper bound of |v^T v - 1|: the larger magnitude of the directed
    // enclosure of the exact sum.
    let orthogonality_error = magnitude_bounds(
        || squares.iter().chain(std::iter::once(&minus_one)),
        verification_bits,
    )
    .1;
    // Report and decide at the working precision, rounded up.
    let report_up = |value: &Float| Float::with_val_round(precision_bits, value, Round::Up).0;
    let residual_norm = report_up(&residual_norm);
    let relative_residual = report_up(&relative_residual);
    let scaled_backward_error = report_up(&scaled_backward_error);
    let orthogonality_error = report_up(&orthogonality_error);

    let residual_tolerance = hp_parse_literal_round(
        &config.stopping.absolute_residual,
        precision_bits,
        rug::float::Round::Down,
    )?;
    let backward_tolerance = hp_parse_literal_round(
        &config.stopping.scaled_backward_error,
        precision_bits,
        rug::float::Round::Down,
    )?;
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
            TerminationReason::MaximumPrecision,
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
        algorithm: format!(
            "dense_reference_hp_v3:{}",
            xc_numerics::eigen::DENSE_EIGENVECTOR_SEMANTICS
        ),
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
    fn hp_reference_adapter_verifies_the_returned_pair_against_the_stored_matrix() {
        // A = M + delta*E (E_02 = E_20 = 1) with M the 4-cycle matrix: at 64
        // working bits the rounded image A*v drops delta, but the exact
        // residual of the returned pair is nonzero and far above 1e-200.
        for exponent in [-40, -100, -200, -400] {
            let precision = 64;
            let mut matrix: Vec<Float> = [3, -1, 0, -1, -1, 3, -1, 0, 0, -1, 3, -1, -1, 0, -1, 3]
                .iter()
                .map(|entry| Float::with_val(precision, *entry))
                .collect();
            matrix[2] = Float::with_val(precision, Float::i_exp(1, exponent));
            matrix[8] = matrix[2].clone();
            let problem = DenseSymmetricProblemHp::new(&matrix, 4).unwrap();
            let mut config = config(EigenTarget::AlgebraicSmallest);
            config.precision = PrecisionPolicy::fixed(precision);
            config.stopping.absolute_residual = DecimalLiteral::new("1e-200").unwrap();
            config.stopping.scaled_backward_error = DecimalLiteral::new("1e-200").unwrap();
            let report = solve_dense_reference_hp(&problem, &config).unwrap();
            let residual = Float::with_val(precision, Float::parse(&report.residual_norm).unwrap());
            assert!(!residual.is_zero(), "delta = 2^{exponent}");
            assert_ne!(
                report.status,
                ResultStatus::Converged,
                "delta = 2^{exponent}"
            );
        }
    }

    #[test]
    fn hp_reference_orthogonality_error_is_an_upper_bound() {
        // A = [[d^2, -d, -t], [-d, 1, 0], [-t, 0, 1]] at 64 bits returns
        // v = (1, d, t), whose exact |v^T v - 1| = d^2 + t^2 must be covered.
        for (d_exponent, t_exponent) in [(-40, -200), (-50, -300), (-80, -400), (-100, -300)] {
            let precision = 64;
            let d = Float::with_val(precision, Float::i_exp(1, d_exponent));
            let t = Float::with_val(precision, Float::i_exp(1, t_exponent));
            let zero = Float::with_val(precision, 0);
            let one = Float::with_val(precision, 1);
            let matrix = vec![
                Float::with_val(precision, &d * &d),
                Float::with_val(precision, -&d),
                Float::with_val(precision, -&t),
                Float::with_val(precision, -&d),
                one.clone(),
                zero.clone(),
                Float::with_val(precision, -&t),
                zero,
                one,
            ];
            let problem = DenseSymmetricProblemHp::new(&matrix, 3).unwrap();
            let mut config = config(EigenTarget::AlgebraicSmallest);
            config.precision = PrecisionPolicy::fixed(precision);
            let report = solve_dense_reference_hp(&problem, &config).unwrap();
            let exact = report
                .eigenvector
                .iter()
                .map(|value| {
                    Float::with_val(precision, Float::parse(value).unwrap())
                        .to_rational()
                        .unwrap()
                })
                .fold(rug::Rational::from(-1), |sum, component| {
                    sum + component.clone() * component
                })
                .abs();
            let reported = Float::with_val(
                precision,
                Float::parse(&report.diagnostics.orthogonality_error).unwrap(),
            )
            .to_rational()
            .unwrap();
            assert!(reported >= exact, "d = 2^{d_exponent}, t = 2^{t_exponent}");
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

    /// Outward binary64 Gershgorin bounds for valid finite inputs. An
    /// unrepresentable bound is infinite; `bisect_index` reports that limit.
    /// Public fields are revalidated so malformed values do not cause indexing panics.
    pub fn gershgorin_bounds(&self) -> (f64, f64) {
        if Self::new(self.diagonal, self.off_diagonal).is_err() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for index in 0..self.dimension() {
            let mut radius = 0.0_f64;
            if index > 0 {
                radius = (radius + self.off_diagonal[index - 1].abs()).next_up();
            }
            if index + 1 < self.dimension() {
                radius = (radius + self.off_diagonal[index].abs()).next_up();
            }
            lower = lower.min((self.diagonal[index] - radius).next_down());
            upper = upper.max((self.diagonal[index] + radius).next_up());
        }
        (lower, upper)
    }

    /// Exact number of eigenvalues strictly below the stored finite binary64
    /// threshold. Dyadic input values are scaled to integers and their Sturm
    /// determinant sequence is evaluated without a floating-point pivot floor.
    ///
    /// This validation-scale path trades speed for exact counts. Integer size
    /// grows with dimension and input exponent spread; it is not the large HP
    /// production solver. It certifies only the exact stored finite matrix.
    pub fn sturm_count_below(&self, threshold: f64) -> Result<usize, SolverError> {
        Self::new(self.diagonal, self.off_diagonal)?;
        if !threshold.is_finite() {
            return Err(SolverError::InvalidConfiguration(
                "Sturm threshold must be finite".to_owned(),
            ));
        }
        Ok(exact_sturm_f64::count_below(
            self.diagonal,
            self.off_diagonal,
            threshold,
        ))
    }

    /// Enclose the zero-based `index`-th algebraically ordered eigenvalue.
    /// On success, the exact difference of the stored endpoints is at most
    /// `absolute_tolerance`; an unrepresentable requested width is an error.
    pub fn bisect_index(
        &self,
        index: usize,
        absolute_tolerance: f64,
        maximum_iterations: usize,
    ) -> Result<(f64, f64), SolverError> {
        Self::new(self.diagonal, self.off_diagonal)?;
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
        if !lower.is_finite() || !upper.is_finite() {
            return Err(SolverError::NumericalBreakdown(
                "Gershgorin bounds exceed the finite binary64 range".to_owned(),
            ));
        }
        for _ in 0..maximum_iterations {
            if exact_sturm_f64::bracket_width_at_most(lower, upper, absolute_tolerance) {
                return Ok((lower, upper));
            }
            let midpoint = if lower <= 0.0 && upper >= 0.0 {
                0.5 * lower + 0.5 * upper
            } else {
                lower + 0.5 * (upper - lower)
            };
            if midpoint <= lower || midpoint >= upper {
                return Err(SolverError::NonConvergence(
                    "binary64 eigenvalue bracket stagnated above the requested tolerance"
                        .to_owned(),
                ));
            }
            let count = self.sturm_count_below(midpoint)?;
            if count <= index {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
            if exact_sturm_f64::bracket_width_at_most(lower, upper, absolute_tolerance) {
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
    /// Absolute residual tolerance in operator units; accepted as an alternative
    /// to the dimensionless scaled backward-error tolerance.
    pub absolute_residual_tolerance: f64,
    pub scaled_backward_error_tolerance: f64,
    /// |lambda_new-lambda_old| / max(1, |lambda_new|, |lambda_old|).
    /// The unit floor makes this absolute for small eigenvalues.
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
    /// False for exact stationary/full-space acceptance before a second iterate.
    /// A frozen-vector re-evaluation is never counted as stability evidence.
    #[serde(default)]
    pub ritz_value_stability_observed: bool,
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
    if iterate
        .vector
        .iter()
        .chain(&iterate.applied_operator)
        .chain(&iterate.applied_metric)
        .any(|v| !v.is_finite())
    {
        return Err(SolverError::NumericalBreakdown(
            "generalized iterate contains nonfinite values".into(),
        ));
    }
    let metric_norm_sq = dot(&iterate.vector, &iterate.applied_metric);
    if !metric_norm_sq.is_finite() || metric_norm_sq <= 0.0 {
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

fn generalized_symmetric_decomposition_f64(
    mut matrix: DMatrix<f64>,
) -> Result<SymmetricEigen<f64, nalgebra::Dyn>, SolverError> {
    let invalid = || {
        SolverError::NumericalBreakdown(
            "generalized whitening/eigendecomposition produced invalid arithmetic".into(),
        )
    };
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n || matrix.iter().any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    // C is mathematically symmetric. Average computed roundoff asymmetry;
    // the final residual is evaluated against the original A and B.
    for row in 0..n {
        for column in 0..row {
            let a = matrix[(row, column)];
            let b = matrix[(column, row)];
            let average = if (a + b).is_finite() {
                (a + b) / 2.0
            } else {
                a / 2.0 + b / 2.0
            };
            matrix[(row, column)] = average;
            matrix[(column, row)] = average;
        }
    }
    let result = checked_symmetric_decomposition_f64(matrix)?;
    if result
        .eigenvalues
        .iter()
        .chain(result.eigenvectors.iter())
        .any(|v| !v.is_finite())
    {
        return Err(invalid());
    }
    Ok(result)
}

fn generalized_residual_measures_f64(
    residual: &[f64],
    ax: &[f64],
    bx: &[f64],
    eigenvalue: f64,
) -> Result<(f64, f64), SolverError> {
    let invalid = || {
        SolverError::NumericalBreakdown(
            "generalized residual diagnostic is not finite or representable".into(),
        )
    };
    if residual.is_empty()
        || ax.len() != residual.len()
        || bx.len() != residual.len()
        || !eigenvalue.is_finite()
    {
        return Err(invalid());
    }
    let r = norm(residual);
    let a = norm(ax);
    let b = norm(bx);
    if !r.is_finite() || !a.is_finite() || !b.is_finite() || b == 0.0 {
        return Err(invalid());
    }
    let denominator = a + eigenvalue.abs() * b;
    let relative = if denominator.is_finite() && denominator > 0.0 {
        r / denominator
    } else {
        let scale = a.max(eigenvalue.abs());
        if scale == 0.0 && r == 0.0 {
            0.0
        } else {
            (r / scale) / (a / scale + (eigenvalue.abs() / scale) * b)
        }
    };
    if !relative.is_finite() || (relative == 0.0 && r != 0.0) {
        return Err(invalid());
    }
    Ok((r, relative))
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
    let decomposition = generalized_symmetric_decomposition_f64(whitened)?;
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
    if coefficients.iter().any(|v| !v.is_finite()) {
        return Err(SolverError::NumericalBreakdown(
            "generalized projected vector is nonfinite".into(),
        ));
    }
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
        let mut operator_applications = 1;
        let mut metric_applications = 1;
        let mut projected_factorizations = 0;
        let mut preconditioner_applications = 0;
        // Count performed operations at their call sites; iteration count is
        // not the definition of operator, metric, or factorization work.
        #[allow(clippy::explicit_counter_loop)]
        for iteration in 1..=config.maximum_iterations {
            check_solver_cancellation(cancellation)?;
            // Reapply the stored vectors, replacing recurrence images before
            // either projection or acceptance. Linear-combination updates can
            // accumulate a residual floor that does not belong to the vector.
            problem
                .operator
                .apply(&current.vector, &mut current.applied_operator)?;
            operator_applications += 1;
            problem
                .metric
                .apply(&current.vector, &mut current.applied_metric)?;
            metric_applications += 1;
            if let Some(previous_direction) = &mut direction {
                problem.operator.apply(
                    &previous_direction.vector,
                    &mut previous_direction.applied_operator,
                )?;
                operator_applications += 1;
                problem.metric.apply(
                    &previous_direction.vector,
                    &mut previous_direction.applied_metric,
                )?;
                metric_applications += 1;
            }
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
            let (residual_norm, relative_residual) = generalized_residual_measures_f64(
                &residual,
                &current.applied_operator,
                &current.applied_metric,
                eigenvalue,
            )?;
            let scaled_backward_error = relative_residual;
            let metric_normalization_error = (denominator - 1.0).abs();
            let stability = previous_value
                .map(|previous| {
                    (eigenvalue - previous).abs() / eigenvalue.abs().max(previous.abs()).max(1.0)
                })
                .unwrap_or(f64::INFINITY);
            // Stability is evidence from an actual projected update, including
            // exact-residual warm starts; minimum_iterations is never bypassed.
            let residuals_converged = residual_norm <= config.absolute_residual_tolerance
                || scaled_backward_error <= config.scaled_backward_error_tolerance;
            let converged = iteration >= config.minimum_iterations
                && (residual_norm <= config.absolute_residual_tolerance
                    || scaled_backward_error <= config.scaled_backward_error_tolerance)
                && previous_value.is_some_and(|previous| {
                    native_ritz_stable(eigenvalue, previous, config.ritz_value_stability_tolerance)
                });
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
                    // Keep portable JSON finite; zero is only a placeholder
                    // when the separate observation flag is false.
                    ritz_value_stability: if previous_value.is_some() {
                        stability
                    } else {
                        0.0
                    },
                    ritz_value_stability_observed: previous_value.is_some(),
                    target_ordering_established_by_full_space_projection: last_projected_dimension
                        == dimension,
                    iterations: iteration,
                    operator_applications,
                    metric_applications,
                    projected_factorizations,
                    preconditioner_applications,
                    retained_subspace_vectors: (if direction.is_some() { 3 } else { 2 })
                        .min(dimension),
                    estimated_peak_memory_bytes: (20u64)
                        .saturating_mul(dimension as u64)
                        .saturating_mul(8),
                    algorithm: "matrix_free_lobpcg_fresh_images_real_stationary_updates_f64_v4"
                        .to_owned(),
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

            if dimension == 1 {
                let coefficients = projected_generalized_extreme(
                    DMatrix::from_element(1, 1, dot(&current.vector, &current.applied_operator)),
                    DMatrix::from_element(1, 1, dot(&current.vector, &current.applied_metric)),
                    config.target == EigenTarget::AlgebraicLargest,
                )?;
                let mut next = GeneralizedIterateF64 {
                    vector: current.vector.iter().map(|x| x * coefficients[0]).collect(),
                    applied_operator: current
                        .applied_operator
                        .iter()
                        .map(|x| x * coefficients[0])
                        .collect(),
                    applied_metric: current
                        .applied_metric
                        .iter()
                        .map(|x| x * coefficients[0])
                        .collect(),
                };
                normalize_generalized_iterate(&mut next)?;
                current = next;
                previous_value = Some(eigenvalue);
                last_projected_dimension = 1;
                projected_factorizations += 1;
                continue;
            }
            let mut search_vector = vec![0.0; dimension];
            if residuals_converged {
                // A residual-converged iterate still needs an actual update
                // to observe stability. Use an independent coordinate trial.
                direction = None;
                let coordinate = (0..dimension)
                    .map(|offset| (iteration - 1 + offset) % dimension)
                    .find(|j| {
                        current
                            .vector
                            .iter()
                            .enumerate()
                            .any(|(k, x)| k != *j && *x != 0.0)
                    })
                    .ok_or_else(|| {
                        SolverError::NumericalBreakdown(
                            "no independent stationary complement".into(),
                        )
                    })?;
                search_vector[coordinate] = 1.0;
            } else if let Some(preconditioner) = preconditioner {
                preconditioner.apply(&residual, &mut search_vector)?;
                preconditioner_applications += 1;
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
            metric_applications += 1;
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
            operator_applications += 1;
            projected_factorizations += 1;
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
        if !symmetry_tolerance.is_finite() || symmetry_tolerance < 0.0 {
            return Err(SolverError::InvalidConfiguration(
                "symmetry tolerance must be finite and nonnegative".into(),
            ));
        }
        let problem = Self {
            operator,
            metric,
            dimension,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Require finite, exactly symmetric storage and a positive square shape.
    /// Positive definiteness is checked numerically during Cholesky, not here.
    pub fn validate(&self) -> Result<(), SolverError> {
        let n = self.dimension;
        if n == 0
            || n.checked_mul(n) != Some(self.operator.len())
            || self.metric.len() != self.operator.len()
        {
            return Err(SolverError::InvalidConfiguration(
                "generalized dense matrices have invalid shape".into(),
            ));
        }
        for matrix in [self.operator, self.metric] {
            if matrix.iter().any(|v| !v.is_finite()) {
                return Err(SolverError::InvalidConfiguration(
                    "generalized dense entries must be finite".into(),
                ));
            }
            for row in 0..n {
                for column in 0..row {
                    if matrix[row * n + column] != matrix[column * n + row] {
                        return Err(SolverError::InvalidConfiguration(
                            "generalized dense entries must be exactly symmetric".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Cholesky-whitened dense reference. Whitening is not backward stable for an
/// ill-conditioned metric, so the result is `Converged` only when its scaled
/// backward error `|Ax - lambda Bx| / (|Ax| + |lambda| |Bx|)` meets the tolerance.
#[derive(Clone, Debug)]
pub struct DenseGeneralizedReferenceSolverF64 {
    pub maximum_dimension: usize,
    pub scaled_backward_error_tolerance: f64,
}

impl Default for DenseGeneralizedReferenceSolverF64 {
    fn default() -> Self {
        Self {
            maximum_dimension: 2048,
            scaled_backward_error_tolerance: 1e-12,
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
        problem.validate()?;
        if !self.scaled_backward_error_tolerance.is_finite()
            || self.scaled_backward_error_tolerance <= 0.0
        {
            return Err(SolverError::InvalidConfiguration(
                "generalized dense backward-error tolerance must be finite and positive".to_owned(),
            ));
        }
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
        let decomposition = generalized_symmetric_decomposition_f64(whitened)?;
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
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(SolverError::NumericalBreakdown(
                "generalized Rayleigh denominator is invalid".into(),
            ));
        }
        let residual = &ax - &bx * eigenvalue;
        let (residual_norm, backward_error) = generalized_residual_measures_f64(
            residual.as_slice(),
            ax.as_slice(),
            bx.as_slice(),
            eigenvalue,
        )?;
        let metric_norm_error = (denominator - 1.0).abs();
        let status = if backward_error <= self.scaled_backward_error_tolerance {
            ResultStatus::Converged
        } else {
            ResultStatus::Approximate
        };
        Ok(GeneralizedEigenpairReportF64 {
            eigenvalue,
            eigenvector: x.iter().copied().collect(),
            residual_norm,
            metric_norm_error,
            algorithm: "dense_cholesky_whitened_generalized_reference_f64_v2".to_owned(),
            status,
            assurance: AssuranceLevel::Computed,
            provenance: SolverProvenance::current_package("f64"),
        })
    }
}

#[cfg(test)]
mod generalized_reference_tests {
    use super::*;

    #[test]
    fn whitened_reference_reports_approximate_when_backward_error_fails() {
        // Metric condition number about 5e12: whitening returned a residual
        // near 709 (relative eigenvalue error 1.4e-4) and reported Converged.
        let operator = [
            -0.41273380552487193,
            -0.3864069059005937,
            0.13519410323961334,
            0.2574795313636188,
            -0.28645749404743004,
            -0.3974729854585808,
            -0.3864069059005937,
            0.7781012587036382,
            0.2102849435489973,
            -0.29919540302092984,
            0.7845729071890382,
            0.7864643294524561,
            0.13519410323961334,
            0.2102849435489973,
            -0.9338020277232124,
            -0.2927585002628015,
            0.08654124848273104,
            -0.7882120404837613,
            0.2574795313636188,
            -0.29919540302092984,
            -0.2927585002628015,
            -0.7505410819383027,
            0.18841437891266866,
            -0.28829972970416584,
            -0.28645749404743004,
            0.7845729071890382,
            0.08654124848273104,
            0.18841437891266866,
            -0.7571457271796815,
            -0.17700870892989895,
            -0.3974729854585808,
            0.7864643294524561,
            -0.7882120404837613,
            -0.28829972970416584,
            -0.17700870892989895,
            -0.7432760861978043,
        ];
        let metric = [
            2.3962601867548288,
            0.15602728836631333,
            1.2224482356530937,
            -0.2869233820366388,
            -0.1875536274618723,
            -0.5084455911381273,
            0.15602728836631333,
            2.377614535139449,
            0.30956284397280315,
            -0.09514475975852507,
            1.2190111004396407,
            -0.5303756913017842,
            1.2224482356530937,
            0.30956284397280315,
            3.0094420814517284,
            -0.8123129997359173,
            -0.23479947436886042,
            0.5470044151918012,
            -0.2869233820366388,
            -0.09514475975852507,
            -0.8123129997359173,
            3.83858808970441,
            0.42495453305553754,
            -0.4985793243108785,
            -0.1875536274618723,
            1.2190111004396407,
            -0.23479947436886042,
            0.42495453305553754,
            1.3768392787627177,
            -1.3213864055358115,
            -0.5084455911381273,
            -0.5303756913017842,
            0.5470044151918012,
            -0.4985793243108785,
            -1.3213864055358115,
            2.0012558281878756,
        ];
        let problem = DenseGeneralizedProblemF64::new(&operator, &metric, 6, 0.0).unwrap();
        let solver = DenseGeneralizedReferenceSolverF64::default();
        let result = solver
            .solve(&problem, &EigenTarget::AlgebraicSmallest)
            .unwrap();
        assert_eq!(result.status, ResultStatus::Approximate);
        let identity = [1.0, 0.0, 0.0, 1.0];
        let diagonal = [1.0, 0.0, 0.0, 2.0];
        let easy = DenseGeneralizedProblemF64::new(&diagonal, &identity, 2, 0.0).unwrap();
        let result = solver.solve(&easy, &EigenTarget::AlgebraicLargest).unwrap();
        assert_eq!(result.status, ResultStatus::Converged);
        assert!(DenseGeneralizedReferenceSolverF64 {
            scaled_backward_error_tolerance: f64::NAN,
            ..solver
        }
        .solve(&easy, &EigenTarget::AlgebraicLargest)
        .is_err());
    }

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
            assert!(matrix_free.operator_applications >= 2 * matrix_free.iterations);
            assert_eq!(
                matrix_free.metric_applications,
                matrix_free.operator_applications
            );
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
        // These are identities of deterministic seed constructions, not random
        // numbers. Shared starts are decisive for an iterative target claim.
        let seed = match self {
            Self::ShiftedPowerExtremeReference
            | Self::LanczosExtremeReference
            | Self::MatrixFreeGeneralizedLobpcg => Some(1),
            Self::HpBlockGeneralizedLobpcg | Self::HpBlockShiftInvert => Some(2),
            Self::HpThickRestartLanczos => Some(3),
            Self::BlockSubspaceExtremeReference => Some(4),
            Self::HpMatrixFreeGeneralizedRayleighRitz => Some(5),
            _ => None,
        };
        let decisive_intermediates = seed
            .into_iter()
            .map(|id| format!("xc-solver:deterministic-seed-construction:{id}"))
            .collect();
        RouteEvidence {
            route_id: self.id().to_owned(),
            algorithm_family: self.algorithm_family().to_owned(),
            formulation: self.formulation().to_owned(),
            implementation_id: format!("xc-solver@{}:{}", env!("CARGO_PKG_VERSION"), self.id()),
            decisive_intermediates,
            precision_bits: Some(precision_bits),
            seed,
            thread_count,
            evidence_digest: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SolverPlan {
    #[serde(default)]
    pub semantics_id: String,
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
        let working_bits = self
            .precision
            .initial_working_bits()
            .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
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
        if self.generalized
            && self.requested_eigenpairs != 1
            && working_bits <= f64::MANTISSA_DIGITS
        {
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
    let working_bits = input
        .precision
        .initial_working_bits()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    let hp_requested = working_bits > f64::MANTISSA_DIGITS;
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

    if !hp_requested
        && input.assurance != AssuranceLevel::Certified
        && (!algebraic_extreme || (input.generalized && input.requested_eigenpairs > 1))
    {
        return Err(SolverError::UnsupportedTarget(
            "no installed native eigenpair route implements this target and count; selected Sturm values remain available through the values API".into(),
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
        } else if input.matrix_materialized
            && input.structure == xc_operator::MatrixStructure::Dense
        {
            SolverRoute::DenseGeneralizedWhiteningReference
        } else {
            SolverRoute::MatrixFreeGeneralizedLobpcg
        }
    } else {
        match (&input.structure, hp_requested, selected_target) {
            (xc_operator::MatrixStructure::Tridiagonal, true, _) if hp_tridiagonal_index_target => {
                SolverRoute::HpTridiagonalSturmSelected
            }
            (_, true, _) if interior_target => SolverRoute::HpBlockShiftInvert,
            (_, true, _) if algebraic_extreme && input.requested_eigenpairs < input.dimension => {
                SolverRoute::HpThickRestartLanczos
            }
            (xc_operator::MatrixStructure::Dense, true, _)
                if input.matrix_materialized
                    && input.requested_eigenpairs == 1
                    && !matches!(input.target, EigenTarget::IndexRange { .. }) =>
            {
                SolverRoute::HpDenseReference
            }
            (_, true, _) => SolverRoute::HpSelectedSpectrumPlanned,
            (xc_operator::MatrixStructure::Dense, false, _)
                if input.matrix_materialized && input.requested_eigenpairs == 1 =>
            {
                SolverRoute::DenseFullSpectrumReference
            }
            (_, false, _) if input.requested_eigenpairs > 1 && algebraic_extreme => {
                SolverRoute::BlockSubspaceExtremeReference
            }
            (_, false, false) => SolverRoute::LanczosExtremeReference,
            (_, false, true) => SolverRoute::LanczosExtremeReference,
        }
    };

    if primary == SolverRoute::HpSelectedSpectrumPlanned {
        return Err(SolverError::UnsupportedTarget(
            "no installed HP eigenpair executor supports this structure, target, and count".into(),
        ));
    }
    let crosscheck_candidate = match primary {
        SolverRoute::DenseFullSpectrumReference => Some(SolverRoute::LanczosExtremeReference),
        SolverRoute::TridiagonalFullSpectrumReference => {
            Some(SolverRoute::TridiagonalSturmSelected)
        }
        SolverRoute::TridiagonalSturmSelected => None,
        // These share a decisive seed and cannot independently resolve an
        // unvisited eigenspace. No automatic cross-check is available here.
        SolverRoute::LanczosExtremeReference | SolverRoute::ShiftedPowerExtremeReference => None,
        SolverRoute::BlockSubspaceExtremeReference => None,
        SolverRoute::DenseGeneralizedWhiteningReference => {
            Some(SolverRoute::MatrixFreeGeneralizedLobpcg)
        }
        SolverRoute::MatrixFreeGeneralizedLobpcg => (input.matrix_materialized
            && input.structure == xc_operator::MatrixStructure::Dense)
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
    let crosscheck_candidate = crosscheck_candidate.filter(|route| {
        match route {
            SolverRoute::HpDenseReference | SolverRoute::HpDenseGeneralizedWhiteningReference => {
                input.structure == xc_operator::MatrixStructure::Dense
                    && input.requested_eigenpairs == 1
                    && !matches!(
                        input.target,
                        EigenTarget::IndexRange { .. } | EigenTarget::Interval { .. }
                    )
            }
            // The full tridiagonal reference delivers values only.
            SolverRoute::HpTridiagonalFullSpectrumReference => false,
            _ => true,
        }
    });
    let independent_crosscheck = (input.assurance == AssuranceLevel::CrossChecked)
        .then_some(crosscheck_candidate)
        .flatten();

    let mut precision_schedule_bits = vec![working_bits];
    if input.assurance != AssuranceLevel::Computed {
        let repeat = input
            .precision
            .next_bits(working_bits)
            .unwrap_or(input.precision.maximum_bits);
        if repeat > working_bits {
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
        semantics_id: "executable_eigenpair_vector_count_plan_v2".into(),
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
    // Include Float headers, allocator bookkeeping and limb-rounded mantissas.
    // Plan through the maximum scheduled precision; this is an admission
    // estimate, not an operating-system peak measurement.
    let scalar_bytes = if input.precision.maximum_bits <= f64::MANTISSA_DIGITS {
        8
    } else {
        64u64.saturating_add(
            u64::from(input.precision.maximum_bits)
                .div_ceil(64)
                .saturating_mul(8),
        )
    };
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
        (if input.generalized { 3u64 } else { 2u64 })
            .saturating_mul(matrix_bytes)
            .saturating_add(6u64.saturating_mul(vector_bytes))
    } else {
        iterative_vector_count.saturating_mul(vector_bytes)
    };
    let temporary_memory_bytes = if requires_factorization || requires_materialization {
        Some(4u64.saturating_mul(matrix_bytes))
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
    input.validate()?;
    let working_bits = input
        .precision
        .initial_working_bits()
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
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
            != working_bits
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
        precision_bits: working_bits,
        operator_representation: operator_representation(&input.structure).to_owned(),
        target_kind: target_kind(&input.target).to_owned(),
        generalized: input.generalized,
        require_eigenvectors: true,
        requested_eigenpairs: Some(input.requested_eigenpairs),
        complex_claim: false,
        checkpoint_requested: false,
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
    let primary_evidence = plan
        .primary
        .evidence(working_bits, Some(context.requested_threads));
    let independent_evidence = plan
        .independent_crosscheck
        .map(|route| route.evidence(working_bits, Some(context.requested_threads)));
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
            let catalog = installed_solver_capability_catalog();
            let installed =
                |route: SolverRoute| catalog.solvers.iter().any(|entry| entry.id == route.id());
            let preserves_assurance = input.assurance != AssuranceLevel::CrossChecked
                || matrix_free_plan
                    .independent_crosscheck
                    .is_some_and(installed);
            if installed(matrix_free_plan.primary)
                && preserves_assurance
                && context
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
        delivers_eigenvectors: !matches!(
            route,
            SolverRoute::TridiagonalSturmSelected
                | SolverRoute::TridiagonalFullSpectrumReference
                | SolverRoute::HpTridiagonalFullSpectrumReference
        ),
        maximum_eigenpairs: matches!(
            route,
            SolverRoute::DenseFullSpectrumReference
                | SolverRoute::ShiftedPowerExtremeReference
                | SolverRoute::LanczosExtremeReference
                | SolverRoute::DenseGeneralizedWhiteningReference
                | SolverRoute::MatrixFreeGeneralizedLobpcg
                | SolverRoute::HpDenseReference
                | SolverRoute::HpMatrixFreeGeneralizedRayleighRitz
                | SolverRoute::HpDenseGeneralizedWhiteningReference
        )
        .then_some(1),
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
        maximum_precision_bits: Some(f64::MANTISSA_DIGITS),
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
        SolverCapability {
            id: "diagonal_rank_one_secular".into(),
            algorithm_family: "exact_sign_secular_bisection".into(),
            scalar_backends: string_set(&["f64"]),
            operator_representations: string_set(&["diagonal_rank_one"]),
            target_kinds: string_set(&["full_spectrum"]),
            generalized: false,
            delivers_eigenvectors: false,
            maximum_eigenpairs: None,
            maximum_assurance: AssuranceLevel::Computed,
            checkpoint_supported: false,
        },
        solver_capability(
            SolverRoute::DenseFullSpectrumReference,
            &["f64"],
            &["dense"],
            &extreme,
            false,
        ),
        solver_capability(
            SolverRoute::TridiagonalSturmSelected,
            &["f64"],
            &["tridiagonal"],
            &["algebraic_extreme", "index_range"],
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
        assert!(matches!(
            plan_symmetric_eigenproblem(&input),
            Err(SolverError::UnsupportedTarget(_))
        ));
    }

    #[test]
    fn hp_tridiagonal_selected_plan_does_not_substitute_values_only_crosscheck() {
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
        assert_eq!(plan.independent_crosscheck, None);
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
            structure: MatrixStructure::Dense,
            dimension: 100,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
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
        // A native selected Sturm range has no installed independent full
        // tridiagonal spectrum route; preflight must reject this assurance.
        let unavailable = SolverPlannerInput {
            structure: MatrixStructure::Tridiagonal,
            target: EigenTarget::IndexRange { first: 0, last: 2 },
            requested_eigenpairs: 3,
            ..input
        };
        assert!(matches!(
            plan_and_preflight_symmetric_eigenproblem(
                &unavailable,
                &local_context("f64"),
                &installed_solver_capability_catalog()
            ),
            Err(SolverError::UnsupportedTarget(_))
        ));
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
        // The installed dense reference returns one extreme. Multi-pair
        // requests correctly choose the linear-workspace block route instead.
        dense_small_input.requested_eigenpairs = 1;
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
        // At N=100000, six dense f64 matrix buffers plus vectors require
        // about 480 GB: the 128 GiB profile cannot honestly admit this plan.
        let peak = outcome
            .plan
            .resource_estimate
            .resident_memory_bytes
            .unwrap()
            + outcome
                .plan
                .resource_estimate
                .temporary_memory_bytes
                .unwrap();
        let high_memory = ResourcePolicy::for_profile(ResourceProfile::HighMemoryWorkstation);
        assert!(peak > high_memory.maximum_memory_bytes.unwrap());
        assert!(!outcome.resource_alternatives.iter().any(|alternative| {
            alternative.kind == SolverResourceAlternativeKind::LargerResourceProfile
                && alternative.profile == Some(ResourceProfile::HighMemoryWorkstation)
        }));
        // A 40000-dimensional dense request is about 76.8 GB. It is too large
        // for the normal profile but supplies a positive 128 GiB alternative.
        let mut moderate = input.clone();
        moderate.dimension = 40_000;
        let moderate = plan_and_preflight_symmetric_eigenproblem(
            &moderate,
            &local_context("f64"),
            &installed_solver_capability_catalog(),
        )
        .unwrap();
        assert!(!moderate.execution_allowed());
        assert!(moderate.resource_alternatives.iter().any(|alternative| {
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
/// Historical type name for a report-agreement diagnostic. Caller-supplied
/// reports cannot prove independent executions or target identity; accepted
/// assurance is Computed regardless of the input reports' asserted labels.
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
    if !(32..=1_000_064).contains(&precision_bits) {
        return Err(SolverError::InvalidConfiguration(
            "unsupported HP scalar precision".into(),
        ));
    }
    let literal = xc_core::DecimalLiteral::new(value)
        .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    let parsed =
        rug::Float::parse(value).map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?;
    let parsed = rug::Float::with_val(precision_bits, parsed);
    if !parsed.is_finite()
        || (parsed.is_zero()
            && literal
                .canonical()
                .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?
                .as_str()
                != "0")
    {
        return Err(SolverError::InvalidConfiguration(
            "HP scalar is outside the representable exponent range".into(),
        ));
    }
    Ok(parsed)
}

#[cfg(feature = "hp-reference")]
/// Compare the stored MPFR values decoded at each report's declared precision.
/// Difference and one-minus-overlap are upward bounds; overlap is a lower
/// bound. This prevents rounded boundary coincidences from passing a tolerance.
/// This report-only API does not replay an operator, establish independence,
/// or verify target selection. It returns agreement diagnostics with Computed
/// assurance; editing report names cannot earn CrossChecked assurance.
pub fn cross_check_hp_reports(
    primary: &EigenpairReportHp,
    independent: &EigenpairReportHp,
    tolerance: HpCrossCheckTolerance,
) -> Result<CrossCheckedEigenpairHp, SolverError> {
    use rug::{float::Round, Float};
    use xc_numerics::mpfr_interval::MpfrInterval;
    let invalid = |message: &str| SolverError::CrossCheckDisagreement(message.into());
    if primary.eigenvector.len() != independent.eigenvector.len() || primary.eigenvector.is_empty()
    {
        return Err(invalid(
            "HP eigenvectors must have the same nonzero dimension",
        ));
    }
    for report in [primary, independent] {
        if !(32..=1_000_000).contains(&report.precision_bits)
            || report.status != ResultStatus::Converged
        {
            return Err(invalid(
                "HP cross-check requires converged reports at supported precision",
            ));
        }
        for value in [
            &report.residual_norm,
            &report.relative_residual,
            &report.scaled_backward_error,
            &report.diagnostics.absolute_residual,
            &report.diagnostics.relative_residual,
            &report.diagnostics.scaled_backward_error,
            &report.diagnostics.orthogonality_error,
        ] {
            let parsed = hp_parse_string(value, report.precision_bits)?;
            if !parsed.is_finite() || parsed < 0 {
                return Err(invalid(
                    "HP report diagnostics must be finite and nonnegative",
                ));
            }
        }
    }
    if primary.algorithm.trim().is_empty()
        || independent.algorithm.trim().is_empty()
        || primary.algorithm == independent.algorithm
    {
        return Err(invalid("HP cross-check requires distinct identified routes; the caller must establish independence"));
    }
    let precision_bits = primary
        .precision_bits
        .max(independent.precision_bits)
        .checked_add(64)
        .filter(|p| *p <= rug::float::prec_max())
        .ok_or_else(|| invalid("unsupported HP cross-check guard precision"))?;
    let parse = |value: &str, source_precision| -> Result<Float, SolverError> {
        let parsed = hp_parse_string(value, source_precision)?;
        if !parsed.is_finite() {
            return Err(invalid("HP report contains a nonfinite scalar"));
        }
        // Report strings encode a stored Float, rather than an exact decimal.
        Ok(Float::with_val(precision_bits, parsed))
    };
    let parse_tolerance = |literal: &xc_core::DecimalLiteral| -> Result<Float, SolverError> {
        literal.validate().map_err(|e| invalid(&e.to_string()))?;
        let _representable = hp_parse_literal(literal, precision_bits)?;
        let parsed = Float::parse(literal.as_str()).map_err(|e| invalid(&e.to_string()))?;
        let parsed = Float::with_val_round(precision_bits, parsed, Round::Down).0;
        if !parsed.is_finite() || parsed < 0 {
            return Err(invalid(
                "HP cross-check tolerance must be finite and nonnegative",
            ));
        }
        Ok(parsed)
    };
    let eigenvalue_tolerance = parse_tolerance(&tolerance.eigenvalue_absolute)?;
    let overlap_tolerance = parse_tolerance(&tolerance.one_minus_overlap_squared)?;
    let primary_value = parse(&primary.eigenvalue, primary.precision_bits)?;
    let independent_value = parse(&independent.eigenvalue, independent.precision_bits)?;
    let eigenvalue_difference = if primary_value >= independent_value {
        Float::with_val_round(
            precision_bits,
            &primary_value - &independent_value,
            Round::Up,
        )
        .0
    } else {
        Float::with_val_round(
            precision_bits,
            &independent_value - &primary_value,
            Round::Up,
        )
        .0
    };
    if !eigenvalue_difference.is_finite() {
        return Err(invalid("HP cross-check difference is unrepresentable"));
    }
    let scaled = |report: &EigenpairReportHp| -> Result<Vec<MpfrInterval>, SolverError> {
        let values: Vec<_> = report
            .eigenvector
            .iter()
            .map(|v| parse(v, report.precision_bits))
            .collect::<Result<_, _>>()?;
        let maximum = values
            .iter()
            .map(|v| v.clone().abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        if maximum.is_zero() {
            return Err(invalid("HP cross-check received a zero eigenvector"));
        }
        let divisor = MpfrInterval::point(maximum);
        values
            .into_iter()
            .map(|v| {
                MpfrInterval::point(v)
                    .div(&divisor)
                    .map_err(|e| invalid(&e.to_string()))
            })
            .collect()
    };
    let left = scaled(primary)?;
    let right = scaled(independent)?;
    let mut dot = MpfrInterval::from_i64(0, precision_bits);
    let mut left_norm = dot.clone();
    let mut right_norm = dot.clone();
    for (a, b) in left.iter().zip(&right) {
        dot = dot.add(&a.mul(b));
        left_norm = left_norm.add(&a.square());
        right_norm = right_norm.add(&b.square());
    }
    let overlap = dot
        .square()
        .div(&left_norm.mul(&right_norm))
        .map_err(|e| invalid(&e.to_string()))?;
    overlap.validate().map_err(|e| invalid(&e.to_string()))?;
    let mut overlap_squared = overlap.lower().clone();
    // The exact normalized squared overlap belongs to [0,1]. Intersecting
    // with that mathematical domain preserves a conservative lower bound.
    if overlap_squared < 0 {
        overlap_squared = Float::with_val(precision_bits, 0);
    }
    if overlap_squared > 1 {
        return Err(invalid("HP overlap enclosure contradicts Cauchy-Schwarz"));
    }
    let one = Float::with_val(precision_bits, 1);
    let one_minus_overlap =
        Float::with_val_round(precision_bits, &one - &overlap_squared, Round::Up).0;
    if eigenvalue_difference > eigenvalue_tolerance || one_minus_overlap > overlap_tolerance {
        return Err(invalid("HP report agreement does not establish the requested eigenvalue and overlap tolerances"));
    }
    let digits = xc_numerics::reduction::roundtrip_decimal_digits(precision_bits).max(32);
    let mut accepted = primary.clone();
    accepted.assurance = AssuranceLevel::Computed;
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
    fn hp_crosscheck_rejects_nonfinite_reports_and_tolerances() {
        let tolerance = || HpCrossCheckTolerance {
            eigenvalue_absolute: xc_core::DecimalLiteral::new("1e-30").unwrap(),
            one_minus_overlap_squared: xc_core::DecimalLiteral::new("1e-30").unwrap(),
        };
        let mut good = report("1", &["1", "0"], 128);
        good.algorithm = "independent-fixture".into();
        for bad in [
            report("NaN", &["1", "0"], 128),
            report("1", &["NaN", "0"], 128),
        ] {
            assert!(cross_check_hp_reports(&bad, &good, tolerance()).is_err());
        }
        let mut excessive = tolerance();
        excessive.eigenvalue_absolute = xc_core::DecimalLiteral::new("1e1000000000").unwrap();
        assert!(cross_check_hp_reports(&report("1", &["1", "0"], 128), &good, excessive).is_err());
    }

    #[test]
    fn hp_crosscheck_does_not_round_excess_error_into_tolerance() {
        let tiny: rug::Float = rug::Float::with_val(128, -1) >> 400u32;
        let primary = report("1", &["1", "0"], 128);
        let mut independent = report(&tiny.to_string_radix(10, None), &["1", "0"], 128);
        independent.algorithm = "independent-fixture".into();
        assert!(cross_check_hp_reports(
            &primary,
            &independent,
            HpCrossCheckTolerance {
                eigenvalue_absolute: xc_core::DecimalLiteral::new("1").unwrap(),
                one_minus_overlap_squared: xc_core::DecimalLiteral::new("1e-30").unwrap()
            }
        )
        .is_err());
    }

    #[test]
    fn hp_crosscheck_is_sign_invariant() {
        let primary = report("1e-400", &["1", "0"], 1024);
        let mut independent = report("1.0000000001e-400", &["-1", "0"], 2048);
        independent.algorithm = "independent-fixture".into();
        let result = cross_check_hp_reports(
            &primary,
            &independent,
            HpCrossCheckTolerance {
                eigenvalue_absolute: xc_core::DecimalLiteral::new("1e-409").unwrap(),
                one_minus_overlap_squared: xc_core::DecimalLiteral::new("1e-50").unwrap(),
            },
        )
        .unwrap();
        assert_eq!(result.accepted.assurance, AssuranceLevel::Computed);
    }
}

// ===========================================================================
// Diagonal-plus-rank-one secular reference solver
// ===========================================================================

/// Exact stored-binary width policy: width <= max(absolute_width, relative_width*|point|).
/// Set absolute_width to zero to require relative accuracy without a unit floor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankOneSecularToleranceF64 {
    pub absolute_width: f64,
    pub relative_width: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankOneSecularEnclosureF64 {
    pub lower: f64,
    pub upper: f64,
    pub lower_is_pole: bool,
    pub upper_is_pole: bool,
    /// Upward bound on exact upper-lower, absent only if no finite binary64 bound exists.
    pub absolute_width_upper_bound: Option<f64>,
    /// Upward bound on exact width/|returned point|, absent at zero or outside finite range.
    pub relative_width_upper_bound: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankOneSecularSpectrumF64 {
    pub eigenvalues: Vec<f64>,
    #[serde(default)]
    pub enclosures: Vec<RankOneSecularEnclosureF64>,
    #[serde(default)]
    pub tolerance_policy: Option<RankOneSecularToleranceF64>,
    /// Finite upward-rounded bounds on |1 + alpha*sum(u_i^2/(d_i-lambda))|
    /// for the exact stored binary inputs and each returned point.
    pub residuals: Vec<f64>,
    pub bisection_iterations: usize,
    pub algorithm: String,
    pub assumptions: Vec<String>,
}

mod rank_one_secular_f64;

/// Enumerate every eigenvalue of `diag(d) + alpha * u u^T` through its
/// secular equation.
///
/// This reference route requires strictly increasing diagonal entries and
/// nonzero update components. Under those assumptions the secular function is
/// strictly monotone between poles and the rank-one interlacing count is
/// complete. The generic result is useful as an independent route for
/// arrowhead/rank-one formulations; applying it to CCM requires a separately
/// reviewed derivation of the correct finite operator and metric.
///
/// Stored inputs are exact dyadic rationals for secular sign and outer-bound
/// arithmetic. A returned point is an exact secular zero or lies in a root
/// bracket whose exact width is at most `tolerance * max(1, abs(point))`.
/// Exhausted iterations, unresolved binary64 brackets, and unrepresentable
/// finite output/residual bounds are errors. Exact integer reference arithmetic
/// costs more than a floating-only iteration; no dense eigensolver is used.
pub fn diagonal_rank_one_spectrum_f64(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    tolerance: f64,
    maximum_iterations: usize,
) -> Result<RankOneSecularSpectrumF64, SolverError> {
    diagonal_rank_one_spectrum_with_tolerances_f64(
        diagonal,
        vector,
        alpha,
        RankOneSecularToleranceF64 {
            absolute_width: tolerance,
            relative_width: tolerance,
        },
        maximum_iterations,
    )
}

/// Validated values-only plan for the explicitly supplied diagonal rank-one form.
/// Execution revalidates deserialized plans and applies the retained width policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankOneSecularPlanF64 {
    diagonal: Vec<f64>,
    vector: Vec<f64>,
    alpha: f64,
    tolerance: RankOneSecularToleranceF64,
    maximum_iterations: usize,
}

impl RankOneSecularPlanF64 {
    pub fn route_id(&self) -> &'static str {
        "diagonal_rank_one_secular"
    }
    pub fn semantics_id(&self) -> &'static str {
        "diagonal_rank_one_values_plan_f64_v1"
    }
    pub fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    pub fn tolerance(&self) -> RankOneSecularToleranceF64 {
        self.tolerance
    }
    pub fn execute(&self) -> Result<RankOneSecularSpectrumF64, SolverError> {
        diagonal_rank_one_spectrum_with_tolerances_f64(
            &self.diagonal,
            &self.vector,
            self.alpha,
            self.tolerance,
            self.maximum_iterations,
        )
    }
}

/// Plan every eigenvalue of an explicit diagonal-plus-rank-one matrix.
/// This values-only route is separate from the generic eigenpair planner;
/// generic rank-one metadata does not establish a diagonal base operator.
pub fn plan_diagonal_rank_one_spectrum_f64(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    tolerance: RankOneSecularToleranceF64,
    maximum_iterations: usize,
) -> Result<RankOneSecularPlanF64, SolverError> {
    validate_diagonal_rank_one_inputs(diagonal, vector, alpha, tolerance, maximum_iterations)?;
    Ok(RankOneSecularPlanF64 {
        diagonal: diagonal.to_vec(),
        vector: vector.to_vec(),
        alpha,
        tolerance,
        maximum_iterations,
    })
}

/// Enumerate rank-one roots with an explicit absolute/relative width policy.
/// Returned brackets use exact secular signs for the stored binary inputs;
/// absolute_width=0 requires relative bracket accuracy even for tiny roots.
/// The same interlacing and nonzero-component requirements as
/// diagonal_rank_one_spectrum_f64 apply. Unresolvable widths return an error.
pub fn diagonal_rank_one_spectrum_with_tolerances_f64(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    tolerance: RankOneSecularToleranceF64,
    maximum_iterations: usize,
) -> Result<RankOneSecularSpectrumF64, SolverError> {
    validate_diagonal_rank_one_inputs(diagonal, vector, alpha, tolerance, maximum_iterations)?;
    rank_one_secular_f64::solve(diagonal, vector, alpha, tolerance, maximum_iterations)
}

fn validate_diagonal_rank_one_inputs(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    tolerance: RankOneSecularToleranceF64,
    maximum_iterations: usize,
) -> Result<(), SolverError> {
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
        || !tolerance.absolute_width.is_finite()
        || tolerance.absolute_width < 0.0
        || !tolerance.relative_width.is_finite()
        || tolerance.relative_width < 0.0
        || (tolerance.absolute_width == 0.0 && tolerance.relative_width == 0.0)
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

    Ok(())
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

#[cfg(all(test, feature = "hp-reference"))]
mod hp_diagnostic_contract_tests;

#[cfg(test)]
mod native_tolerance_boundary_contract {
    use super::*;
    #[test]
    fn native_acceptance_threshold_cannot_round_up_to_one() {
        let mut config = SolverConfig {
            target: EigenTarget::AlgebraicLargest,
            subspace: xc_core::Subspace::Full,
            assurance: AssuranceLevel::Computed,
            precision: xc_core::PrecisionPolicy::fixed(53),
            stopping: xc_core::StoppingPolicy::default(),
            reproducibility: xc_core::Reproducibility::Deterministic,
            algorithm_preferences: vec![],
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        };
        config.stopping.absolute_residual =
            xc_core::DecimalLiteral::new("0.999999999999999999999999999999999999999999").unwrap();
        config.stopping.scaled_backward_error = config.stopping.absolute_residual.clone();
        let (absolute, backward) = stopping_thresholds_f64(&config).unwrap();
        assert!(absolute < 1. && backward < 1.);
    }
}

#[cfg(test)]
mod exhaustive_native_iteration_contract {
    use super::*;
    #[test]
    fn native_rank_is_invariant_under_small_scaling() {
        let a = 2.0f64.powi(-200);
        let basis = orthonormalize_block(vec![vec![a, 0.0], vec![0.0, a]], 2, 2, 2).unwrap();
        assert_eq!(basis, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }
    #[test]
    fn ritz_stability_does_not_accept_rounded_product_equality() {
        assert!(!native_ritz_stable(5.0, 3.5, 0.3));
    }
}

#[cfg(all(test, feature = "hp-reference"))]
mod exhaustive_block_termination_contract {
    use super::*;
    use rug::Float;
    #[test]
    fn mixed_block_does_not_claim_either_uniform_stopping_condition() {
        let zero = Float::with_val(128, 0);
        let one = Float::with_val(128, 1);
        let two = Float::with_val(128, 2);
        assert_eq!(
            hp_block_termination([(&zero, &two), (&two, &zero)], &one, &one),
            TerminationReason::ResidualOrBackwardErrorTolerance
        );
        assert_eq!(
            hp_block_termination([(&zero, &two)], &one, &one),
            TerminationReason::ResidualTolerance
        );
        assert_eq!(
            hp_block_termination([(&two, &zero)], &one, &one),
            TerminationReason::BackwardErrorTolerance
        );
    }
}

#[cfg(all(test, feature = "hp-reference"))]
mod exhaustive_hp_stability_contract {
    use super::*;
    use rug::{ops::Pow, Float, Rational};
    #[test]
    fn absolute_stability_cannot_round_a_larger_difference_down_to_tolerance() {
        let one = Float::with_val(64, 1);
        let previous = -Float::with_val(64, 2).pow(-65i32);
        assert!(hp_ritz_change(&one, &previous, None) > one);
    }
    #[test]
    fn relative_stability_is_an_upper_bound_on_the_exact_ratio() {
        let three = Float::with_val(65, 3);
        let two = Float::with_val(65, 2);
        assert!(
            hp_ritz_change(&three, &two, Some(&three))
                .to_rational()
                .unwrap()
                >= Rational::from((1, 3))
        );
    }
}

/// Constrained real, finite-grid weighted L1 fitting and exact verification.
/// Requires `hp-reference` for GMP rational arithmetic; no floating conversion
/// is used by the reference solver or verifier. Bounds concern the exact stored
/// inputs, not input uncertainty, quadrature error, or a continuum optimum.
#[cfg(feature = "hp-reference")]
#[doc(hidden)]
pub mod weighted_l1 {
    use super::SolverError;
    use rug::{ops::Pow, Integer, Rational};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use xc_core::{CancellationToken, DecimalLiteral};

    pub const SEMANTICS: &str = "constrained-real-weighted-l1-exact-inputs-v1";
    pub const SCOPE: &str = "exact_stored_finite_grid_excludes_input_and_quadrature_error";

    /// Minimize `sum_j weights[j]*|target[j] - basis[j]*a|` subject to
    /// constraints*a=rhs. Coefficients a are free signed reals. Scalar strings
    /// are exact finite decimals or signed-integer/positive-integer rationals.
    /// Identities describe caller-supplied sources, not authenticated provenance.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct WeightedL1Problem {
        pub schema_version: u32,
        pub basis_id: String,
        pub target_id: String,
        pub normalization_id: String,
        pub quadrature_id: String,
        pub basis: Vec<Vec<String>>,
        pub target: Vec<String>,
        pub weights: Vec<String>,
        pub constraints: Vec<Vec<String>>,
        pub rhs: Vec<String>,
    }

    /// Explicit bounded-work reference policy. Rational bit limits apply to
    /// each numerator/denominator and intermediate accumulator. Cancellation
    /// (including a token deadline) is checked during parsing and pivot work.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct WeightedL1Options {
        pub absolute_optimality_gap: String,
        /// Acceptance uses gap <= absolute + relative*objective_upper.
        pub relative_optimality_gap: String,
        pub maximum_pivots: usize,
        pub maximum_tableau_cells: usize,
        pub maximum_rational_bits: u32,
    }

    impl Default for WeightedL1Options {
        fn default() -> Self {
            Self {
                absolute_optimality_gap: "0".into(),
                relative_optimality_gap: "0".into(),
                maximum_pivots: 10_000,
                maximum_tableau_cells: 131_072,
                maximum_rational_bits: 32_768,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct WeightedL1Witness {
        pub coefficients: Vec<String>,
        /// |z_j| <= w_j; B^T*z=C^T*nu. Lower bound y^T*z-d^T*nu.
        pub dual_z: Vec<String>,
        pub dual_nu: Vec<String>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum WeightedL1Status {
        ExactFiniteOptimum,
        GapWithinTolerance,
        GapAboveTolerance,
    }

    /// Exact scalar bounds are reduced rational strings. `accepted` concerns
    /// ONLY the requested finite optimization gap, not uniqueness or continuum
    /// accuracy. A feasible candidate can have a valid but unaccepted bound.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct WeightedL1Verification {
        pub objective_lower: String,
        pub objective_upper: String,
        pub optimality_gap: String,
        pub status: WeightedL1Status,
        pub accepted: bool,
    }

    /// Portable result. Replay recomputes input/request identities, primal
    /// feasibility, dual feasibility, bounds, status, and tolerance acceptance.
    /// Backend and pivot count are informational, not certified performance.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct WeightedL1Fit {
        pub schema_version: u32,
        pub semantics: String,
        pub scope: String,
        pub problem_sha256: String,
        pub request_sha256: String,
        pub backend: String,
        pub pivots: usize,
        pub witness: WeightedL1Witness,
        pub verification: WeightedL1Verification,
    }

    /// Backends propose witnesses; only independent verification determines
    /// accuracy. External backends must enforce their own resource/cancellation
    /// policy and declare any floating candidate arithmetic in their name.
    pub trait WeightedL1Backend {
        fn name(&self) -> &str;
        fn propose(
            &self,
            problem: &WeightedL1Problem,
            options: &WeightedL1Options,
            cancellation: &CancellationToken,
        ) -> Result<(WeightedL1Witness, usize), SolverError>;
    }

    /// Dense two-phase rational simplex with Bland's deterministic pivot rule.
    /// A small-problem reference, not a large sparse LP performance promise.
    pub struct ExactWeightedL1Solver;

    fn invalid(message: &str) -> SolverError {
        SolverError::InvalidConfiguration(format!("weighted L1: {message}"))
    }
    fn exhausted(message: &str) -> SolverError {
        SolverError::IterationBudgetExhausted(format!("weighted L1: {message}"))
    }
    fn check_cancel(token: &CancellationToken) -> Result<(), SolverError> {
        if token.is_cancelled() {
            Err(SolverError::Cancelled("weighted L1 work cancelled".into()))
        } else {
            Ok(())
        }
    }
    fn guard(value: &Rational, options: &WeightedL1Options) -> Result<(), SolverError> {
        if value.numer().significant_bits() > options.maximum_rational_bits
            || value.denom().significant_bits() > options.maximum_rational_bits
        {
            return Err(exhausted("rational bit-growth limit reached"));
        }
        Ok(())
    }

    fn scalar(text: &str, options: &WeightedL1Options) -> Result<Rational, SolverError> {
        // Bound text/exponent BEFORE constructing a GMP integer or a power.
        if text.is_empty() || text.len() > 131_072 || text.bytes().any(|b| b.is_ascii_whitespace())
        {
            return Err(invalid("empty, oversized, or whitespace-containing scalar"));
        }
        let integer = |s: &str, signed: bool| -> Result<Integer, SolverError> {
            let digits = if signed {
                s.strip_prefix('-')
                    .or_else(|| s.strip_prefix('+'))
                    .unwrap_or(s)
            } else {
                s
            };
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(invalid("invalid exact integer"));
            }
            Integer::from_str_radix(s, 10).map_err(|_| invalid("invalid exact integer"))
        };
        let value = if let Some((n, d)) = text.split_once('/') {
            let denominator = integer(d, false)?;
            if denominator <= 0 {
                return Err(invalid("positive denominator required"));
            }
            Rational::from((integer(n, true)?, denominator))
        } else {
            let canonical = DecimalLiteral::new(text)
                .and_then(|v| v.canonical())
                .map_err(|_| invalid("finite decimal or rational scalar required"))?;
            let (mantissa, exponent) = canonical
                .as_str()
                .split_once('e')
                .unwrap_or((canonical.as_str(), "0"));
            let exponent: i32 = exponent
                .parse()
                .map_err(|_| invalid("decimal exponent outside supported range"))?;
            if exponent.unsigned_abs() > 16_384 {
                return Err(invalid("decimal exponent limit"));
            }
            let scale = Integer::from(10).pow(exponent.unsigned_abs());
            let mantissa = integer(mantissa, true)?;
            if exponent >= 0 {
                Rational::from(mantissa * scale)
            } else {
                Rational::from((mantissa, scale))
            }
        };
        guard(&value, options)?;
        Ok(value)
    }

    fn validate_options(options: &WeightedL1Options) -> Result<(Rational, Rational), SolverError> {
        if options.maximum_pivots > 1_000_000
            || options.maximum_tableau_cells == 0
            || options.maximum_tableau_cells > 1_048_576
            || !(64..=65_536).contains(&options.maximum_rational_bits)
        {
            return Err(invalid("unsupported work limits"));
        }
        let absolute = scalar(&options.absolute_optimality_gap, options)?;
        let relative = scalar(&options.relative_optimality_gap, options)?;
        if absolute < 0 || relative < 0 {
            return Err(invalid("negative optimality tolerance"));
        }
        Ok((absolute, relative))
    }

    struct Parsed {
        b: Vec<Vec<Rational>>,
        y: Vec<Rational>,
        w: Vec<Rational>,
        c: Vec<Vec<Rational>>,
        d: Vec<Rational>,
    }

    fn parse(
        problem: &WeightedL1Problem,
        options: &WeightedL1Options,
        token: &CancellationToken,
    ) -> Result<Parsed, SolverError> {
        check_cancel(token)?;
        validate_options(options)?;
        let m = problem.basis.len();
        let n = problem.basis.first().map_or(0, Vec::len);
        let k = problem.constraints.len();
        if problem.schema_version != 1
            || m == 0
            || m > 2048
            || n == 0
            || n > 128
            || k > 128
            || m * n > 65_536
            || problem.basis.iter().any(|row| row.len() != n)
            || problem.target.len() != m
            || problem.weights.len() != m
            || problem.rhs.len() != k
            || problem.constraints.iter().any(|row| row.len() != n)
        {
            return Err(invalid("schema, shape, or dimension limit"));
        }
        for id in [
            &problem.basis_id,
            &problem.target_id,
            &problem.normalization_id,
            &problem.quadrature_id,
        ] {
            if id.trim().is_empty() || id.len() > 4096 {
                return Err(invalid("nonempty bounded source identities required"));
            }
        }
        let mut texts = problem
            .basis
            .iter()
            .flatten()
            .chain(&problem.target)
            .chain(&problem.weights)
            .chain(problem.constraints.iter().flatten())
            .chain(&problem.rhs);
        let total_bytes = texts
            .try_fold(0usize, |a, x| a.checked_add(x.len()))
            .ok_or_else(|| invalid("input size overflow"))?;
        if total_bytes > 8_388_608 {
            return Err(invalid("input scalar byte limit"));
        }
        let vector = |values: &[String]| -> Result<Vec<Rational>, SolverError> {
            values
                .iter()
                .map(|v| {
                    check_cancel(token)?;
                    scalar(v, options)
                })
                .collect()
        };
        let data = Parsed {
            b: problem
                .basis
                .iter()
                .map(|row| vector(row))
                .collect::<Result<_, _>>()?,
            y: vector(&problem.target)?,
            w: vector(&problem.weights)?,
            c: problem
                .constraints
                .iter()
                .map(|row| vector(row))
                .collect::<Result<_, _>>()?,
            d: vector(&problem.rhs)?,
        };
        if data.w.iter().any(|w| w <= &0) {
            return Err(invalid("strictly positive weights required"));
        }
        Ok(data)
    }

    fn add_product(
        sum: &mut Rational,
        a: &Rational,
        b: &Rational,
        options: &WeightedL1Options,
    ) -> Result<(), SolverError> {
        let product = Rational::from(a * b);
        guard(&product, options)?;
        *sum += product;
        guard(sum, options)
    }

    fn dot(
        a: &[Rational],
        b: &[Rational],
        options: &WeightedL1Options,
        token: &CancellationToken,
    ) -> Result<Rational, SolverError> {
        let mut sum = Rational::from(0);
        for (x, y) in a.iter().zip(b) {
            check_cancel(token)?;
            add_product(&mut sum, x, y, options)?;
        }
        Ok(sum)
    }

    /// Independent exact witness checker. No simplex tableau, backend success
    /// flag, reported objective, or floating computation enters this calculation.
    pub fn verify_weighted_l1_witness(
        problem: &WeightedL1Problem,
        options: &WeightedL1Options,
        witness: &WeightedL1Witness,
        cancellation: &CancellationToken,
    ) -> Result<WeightedL1Verification, SolverError> {
        let data = parse(problem, options, cancellation)?;
        let m = data.b.len();
        let n = data.b[0].len();
        let k = data.c.len();
        if witness.coefficients.len() != n
            || witness.dual_z.len() != m
            || witness.dual_nu.len() != k
        {
            return Err(invalid("witness shape mismatch"));
        }
        let read = |values: &[String]| -> Result<Vec<Rational>, SolverError> {
            values
                .iter()
                .map(|v| {
                    check_cancel(cancellation)?;
                    scalar(v, options)
                })
                .collect()
        };
        let a = read(&witness.coefficients)?;
        let z = read(&witness.dual_z)?;
        let nu = read(&witness.dual_nu)?;
        for (row, rhs) in data.c.iter().zip(&data.d) {
            if dot(row, &a, options, cancellation)? != *rhs {
                return Err(invalid("primal equality is not exact"));
            }
        }
        if z.iter().zip(&data.w).any(|(v, w)| v.clone().abs() > *w) {
            return Err(invalid("dual weight bound violated"));
        }
        for j in 0..n {
            check_cancel(cancellation)?;
            let mut left = Rational::from(0);
            let mut right = Rational::from(0);
            for (row, zj) in data.b.iter().zip(&z) {
                add_product(&mut left, &row[j], zj, options)?;
            }
            for (row, nuj) in data.c.iter().zip(&nu) {
                add_product(&mut right, &row[j], nuj, options)?;
            }
            if left != right {
                return Err(invalid("dual stationarity is not exact"));
            }
        }
        let mut upper = Rational::from(0);
        for ((row, y), w) in data.b.iter().zip(&data.y).zip(&data.w) {
            let residual = (y - dot(row, &a, options, cancellation)?).abs();
            guard(&residual, options)?;
            add_product(&mut upper, w, &residual, options)?;
        }
        let mut lower =
            dot(&data.y, &z, options, cancellation)? - dot(&data.d, &nu, options, cancellation)?;
        guard(&lower, options)?;
        // Nonnegativity is also an exact lower bound on this objective.
        if lower < 0 {
            lower = Rational::from(0);
        }
        let gap = Rational::from(&upper - &lower);
        guard(&gap, options)?;
        if gap < 0 {
            return Err(SolverError::NumericalBreakdown(
                "weighted L1 weak duality violated".into(),
            ));
        }
        let (mut threshold, relative) = validate_options(options)?;
        add_product(&mut threshold, &relative, &upper, options)?;
        let status = if gap == 0 {
            WeightedL1Status::ExactFiniteOptimum
        } else if gap <= threshold {
            WeightedL1Status::GapWithinTolerance
        } else {
            WeightedL1Status::GapAboveTolerance
        };
        Ok(WeightedL1Verification {
            objective_lower: lower.to_string(),
            objective_upper: upper.to_string(),
            optimality_gap: gap.to_string(),
            accepted: status != WeightedL1Status::GapAboveTolerance,
            status,
        })
    }

    fn hash<T: Serialize>(value: &T) -> Result<String, SolverError> {
        let bytes =
            serde_json::to_vec(value).map_err(|_| invalid("identity serialization failed"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Wrap an externally proposed witness in a replayable Toolkit record.
    /// An above-tolerance bound remains visible with accepted=false.
    pub fn attest_weighted_l1_candidate(
        problem: &WeightedL1Problem,
        options: &WeightedL1Options,
        witness: WeightedL1Witness,
        backend: &str,
        pivots: usize,
        cancellation: &CancellationToken,
    ) -> Result<WeightedL1Fit, SolverError> {
        if backend.trim().is_empty() || backend.len() > 4096 {
            return Err(invalid("backend identity required"));
        }
        let verification = verify_weighted_l1_witness(problem, options, &witness, cancellation)?;
        Ok(WeightedL1Fit {
            schema_version: 1,
            semantics: SEMANTICS.into(),
            scope: SCOPE.into(),
            problem_sha256: hash(&(SEMANTICS, problem))?,
            request_sha256: hash(&(SEMANTICS, problem, options))?,
            backend: backend.into(),
            pivots,
            witness,
            verification,
        })
    }

    pub fn verify_weighted_l1_fit(
        problem: &WeightedL1Problem,
        options: &WeightedL1Options,
        fit: &WeightedL1Fit,
        cancellation: &CancellationToken,
    ) -> Result<WeightedL1Verification, SolverError> {
        // Validate sizes before serializing potentially large caller inputs.
        parse(problem, options, cancellation)?;
        if fit.schema_version != 1
            || fit.semantics != SEMANTICS
            || fit.scope != SCOPE
            || fit.problem_sha256 != hash(&(SEMANTICS, problem))?
            || fit.request_sha256 != hash(&(SEMANTICS, problem, options))?
            || fit.backend.trim().is_empty()
            || fit.backend.len() > 4096
        {
            return Err(invalid("result source, request, or semantics mismatch"));
        }
        let result = verify_weighted_l1_witness(problem, options, &fit.witness, cancellation)?;
        if result != fit.verification {
            return Err(invalid("reported bounds or acceptance mismatch"));
        }
        Ok(result)
    }

    pub fn fit_weighted_l1_with_backend(
        problem: &WeightedL1Problem,
        options: &WeightedL1Options,
        backend: &dyn WeightedL1Backend,
        cancellation: &CancellationToken,
    ) -> Result<WeightedL1Fit, SolverError> {
        parse(problem, options, cancellation)?;
        let (witness, pivots) = backend.propose(problem, options, cancellation)?;
        check_cancel(cancellation)?;
        attest_weighted_l1_candidate(
            problem,
            options,
            witness,
            backend.name(),
            pivots,
            cancellation,
        )
    }

    /// Solve with the exact rational reference and independently replay its
    /// zero-gap witness before returning. Work-limit/cancellation errors return
    /// no accepted result. Use `fit_weighted_l1_with_backend` for other routes.
    ///
    /// ```
    /// use xc_core::CancellationToken;
    /// use xc_solver::weighted_l1::*;
    /// let problem = WeightedL1Problem {
    ///     schema_version: 1,
    ///     basis_id: "constant".into(), target_id: "three-points".into(),
    ///     normalization_id: "unconstrained".into(), quadrature_id: "weights-1-3-1".into(),
    ///     basis: vec![vec!["1".into()]; 3],
    ///     target: vec!["0".into(), "1".into(), "100".into()],
    ///     weights: vec!["1".into(), "3".into(), "1".into()],
    ///     constraints: vec![], rhs: vec![],
    /// };
    /// let options = WeightedL1Options::default();
    /// let token = CancellationToken::new();
    /// let fit = fit_weighted_l1(&problem, &options, &token).unwrap();
    /// assert_eq!(fit.witness.coefficients, vec!["1"]);
    /// assert_eq!(fit.verification.objective_upper, "100");
    /// assert!(verify_weighted_l1_fit(&problem, &options, &fit, &token).unwrap().accepted);
    /// ```
    pub fn fit_weighted_l1(
        problem: &WeightedL1Problem,
        options: &WeightedL1Options,
        cancellation: &CancellationToken,
    ) -> Result<WeightedL1Fit, SolverError> {
        let result =
            fit_weighted_l1_with_backend(problem, options, &ExactWeightedL1Solver, cancellation)?;
        if result.verification.status != WeightedL1Status::ExactFiniteOptimum {
            return Err(SolverError::NumericalBreakdown(
                "exact L1 reference failed independent optimality check".into(),
            ));
        }
        Ok(result)
    }

    struct Tableau<'a> {
        rows: Vec<Vec<Rational>>, // last column is RHS
        basis: Vec<usize>,
        pivots: usize,
        options: &'a WeightedL1Options,
        token: &'a CancellationToken,
    }

    impl Tableau<'_> {
        fn pivot(&mut self, row: usize, column: usize) -> Result<(), SolverError> {
            check_cancel(self.token)?;
            if self.pivots >= self.options.maximum_pivots {
                return Err(exhausted("simplex pivot budget reached"));
            }
            let pivot = self.rows[row][column].clone();
            if pivot == 0 {
                return Err(SolverError::NumericalBreakdown(
                    "zero exact L1 pivot".into(),
                ));
            }
            for x in &mut self.rows[row] {
                check_cancel(self.token)?;
                *x /= &pivot;
                guard(x, self.options)?;
            }
            let normalized = self.rows[row].clone();
            for (i, other) in self.rows.iter_mut().enumerate() {
                check_cancel(self.token)?;
                if i == row {
                    continue;
                }
                let factor = other[column].clone();
                if factor == 0 {
                    continue;
                }
                for (x, y) in other.iter_mut().zip(&normalized) {
                    check_cancel(self.token)?;
                    let product = Rational::from(&factor * y);
                    guard(&product, self.options)?;
                    *x -= product;
                    guard(x, self.options)?;
                }
                if other.last().is_some_and(|v| v < &0) {
                    return Err(SolverError::NumericalBreakdown(
                        "exact L1 pivot lost primal feasibility".into(),
                    ));
                }
            }
            self.basis[row] = column;
            self.pivots += 1;
            Ok(())
        }

        fn simplex(
            &mut self,
            costs: &[Rational],
            allowed_columns: usize,
        ) -> Result<(), SolverError> {
            loop {
                check_cancel(self.token)?;
                let mut entering = None;
                for (column, cost) in costs.iter().enumerate().take(allowed_columns) {
                    check_cancel(self.token)?;
                    if self.basis.contains(&column) {
                        continue;
                    }
                    let mut reduced = cost.clone();
                    for (row, &basic) in self.rows.iter().zip(&self.basis) {
                        let term = Rational::from(&costs[basic] * &row[column]);
                        guard(&term, self.options)?;
                        reduced -= term;
                        guard(&reduced, self.options)?;
                    }
                    if reduced < 0 {
                        entering = Some(column);
                        break;
                    }
                }
                let Some(column) = entering else {
                    return Ok(());
                };
                // Bland: least variable index enters; least BASIC variable
                // index leaves among exactly equal minimum-ratio choices.
                let mut leaving: Option<(usize, Rational)> = None;
                for (i, row) in self.rows.iter().enumerate() {
                    check_cancel(self.token)?;
                    if row[column] <= 0 {
                        continue;
                    }
                    let ratio = Rational::from(row.last().unwrap() / &row[column]);
                    guard(&ratio, self.options)?;
                    if leaving.as_ref().is_none_or(|(j, old)| {
                        ratio < *old || (ratio == *old && self.basis[i] < self.basis[*j])
                    }) {
                        leaving = Some((i, ratio));
                    }
                }
                let Some((row, _)) = leaving else {
                    return Err(SolverError::NumericalBreakdown(
                        "unexpected unbounded L1 phase".into(),
                    ));
                };
                self.pivot(row, column)?;
            }
        }
    }

    impl WeightedL1Backend for ExactWeightedL1Solver {
        fn name(&self) -> &str {
            "gmp-rational-two-phase-simplex-bland-v1"
        }
        fn propose(
            &self,
            problem: &WeightedL1Problem,
            options: &WeightedL1Options,
            token: &CancellationToken,
        ) -> Result<(WeightedL1Witness, usize), SolverError> {
            let data = parse(problem, options, token)?;
            let m = data.b.len();
            let n = data.b[0].len();
            let k = data.c.len();
            let equations = m + k;
            // B(a+ - a-) + r+ - r- = y; C(a+ - a-) = d.
            let ordinary = 2 * n + 2 * m;
            let columns = ordinary + equations;
            let cells = equations
                .checked_mul(columns + 1)
                .ok_or_else(|| invalid("tableau size overflow"))?;
            if cells > options.maximum_tableau_cells {
                return Err(exhausted("dense tableau cell limit"));
            }
            let mut rows = vec![vec![Rational::from(0); columns + 1]; equations];
            let mut signs = vec![1i32; equations];
            for (i, row) in rows.iter_mut().enumerate() {
                check_cancel(token)?;
                let (coefficients, rhs) = if i < m {
                    (&data.b[i], &data.y[i])
                } else {
                    (&data.c[i - m], &data.d[i - m])
                };
                signs[i] = if rhs < &0 { -1 } else { 1 };
                for (j, value) in coefficients.iter().enumerate() {
                    row[j] = value.clone() * signs[i];
                    row[n + j] = -row[j].clone();
                }
                if i < m {
                    row[2 * n + i] = Rational::from(signs[i]);
                    row[2 * n + m + i] = Rational::from(-signs[i]);
                }
                row[ordinary + i] = Rational::from(1);
                row[columns] = rhs.clone() * signs[i];
            }
            let mut tableau = Tableau {
                rows,
                basis: (ordinary..columns).collect(),
                pivots: 0,
                options,
                token,
            };
            let phase_one: Vec<_> = (0..columns)
                .map(|j| Rational::from(i32::from(j >= ordinary)))
                .collect();
            tableau.simplex(&phase_one, columns)?;
            for (row, &basic) in tableau.rows.iter().zip(&tableau.basis) {
                if basic >= ordinary && row[columns] != 0 {
                    return Err(invalid("inconsistent exact equality constraints"));
                }
            }
            // Pivot zero artificial basics out, or remove a redundant equation.
            // Keep artificial COLUMNS: they retain the original-row transform
            // needed to recover the dual multipliers, including RHS signs.
            let mut i = 0;
            while i < tableau.rows.len() {
                check_cancel(token)?;
                if tableau.basis[i] < ordinary {
                    i += 1;
                    continue;
                }
                if let Some(column) = (0..ordinary).find(|&j| tableau.rows[i][j] != 0) {
                    tableau.pivot(i, column)?;
                    i += 1;
                } else {
                    tableau.rows.remove(i);
                    tableau.basis.remove(i);
                }
            }
            let mut costs = vec![Rational::from(0); columns];
            for (i, w) in data.w.iter().enumerate() {
                costs[2 * n + i] = w.clone();
                costs[2 * n + m + i] = w.clone();
            }
            tableau.simplex(&costs, ordinary)?;
            let mut solution = vec![Rational::from(0); ordinary];
            for (row, &basic) in tableau.rows.iter().zip(&tableau.basis) {
                solution[basic] = row[columns].clone();
            }
            let mut multipliers = vec![Rational::from(0); equations];
            for (j, value) in multipliers.iter_mut().enumerate() {
                check_cancel(token)?;
                for (row, &basic) in tableau.rows.iter().zip(&tableau.basis) {
                    add_product(value, &costs[basic], &row[ordinary + j], options)?;
                }
                *value *= signs[j];
            }
            let witness = WeightedL1Witness {
                coefficients: (0..n)
                    .map(|j| Rational::from(&solution[j] - &solution[n + j]).to_string())
                    .collect(),
                dual_z: multipliers[..m].iter().map(ToString::to_string).collect(),
                dual_nu: multipliers[m..]
                    .iter()
                    .map(|v| (-v.clone()).to_string())
                    .collect(),
            };
            // The public wrapper independently checks optimality before returning.
            Ok((witness, tableau.pivots))
        }
    }
}

/// Finite real symmetric trial forms. Input intervals are caller assertions;
/// arithmetic enclosures do not certify their source or continuum validity.
#[cfg(feature = "hp-reference")]
#[doc(hidden)]
pub mod trial_energy {
    use super::SolverError;
    use rug::{ops::Pow, Integer, Rational};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use xc_core::{CancellationToken, DecimalLiteral};
    use xc_numerics::interval::RationalInterval as I;

    pub const SEMANTICS: &str = "real-symmetric-trial-energy-rational-intervals-v1";
    pub const SCOPE: &str =
        "finite_declared_input_enclosures_no_source_authentication_or_continuum_claim";

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ExactBounds {
        pub lower: String,
        pub upper: String,
    }
    impl ExactBounds {
        pub fn point(value: impl Into<String>) -> Self {
            let value = value.into();
            Self {
                lower: value.clone(),
                upper: value,
            }
        }
    }

    /// Row-major symmetric form, not an operator-coordinate matrix G^-1 A.
    /// Both triangles must contain identical intervals. A metric uses this
    /// same representation and must pass a positive-definiteness check.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialForm {
        pub source_id: String,
        pub basis_id: String,
        pub normalization_id: String,
        pub dimension: usize,
        pub entries: Vec<ExactBounds>,
    }

    /// A proposed vector or additive correction in the declared coordinates.
    /// No eigenvalue, convergence flag, or eigenstate classification is needed.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialVector {
        pub label: String,
        pub source_id: String,
        pub operator_id: String,
        pub metric_id: String,
        pub basis_id: String,
        pub normalization_id: String,
        pub coefficients: Vec<ExactBounds>,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialEnergyProblem {
        pub schema_version: u32,
        pub operator: TrialForm,
        pub metric: TrialForm,
        pub baseline: TrialVector,
        pub corrections: Vec<TrialVector>,
    }

    /// Structural work limits, not an exact resident-memory reservation.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialEnergyOptions {
        pub maximum_dimension: usize,
        pub maximum_parts: usize,
        pub maximum_input_bytes: usize,
        pub maximum_interval_operations: u64,
        pub maximum_rational_bits: u32,
    }
    impl Default for TrialEnergyOptions {
        fn default() -> Self {
            Self {
                maximum_dimension: 256,
                maximum_parts: 8,
                maximum_input_bytes: 67_108_864,
                maximum_interval_operations: 20_000_000,
                maximum_rational_bits: 32_768,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum CancellationStatus {
        Resolved,
        ExactZeroTotal,
        TotalContainsZero,
        AllTermsZero,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct CancellationMeasure {
        /// Sum of absolute values in the explicitly stated decomposition.
        pub absolute_term_sum: ExactBounds,
        /// Enclosure of sum(abs(terms))/abs(total), only away from zero.
        /// This is an amplification factor, not a certified lost-digit count.
        pub amplification: Option<ExactBounds>,
        pub status: CancellationStatus,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct FormContribution {
        /// Part zero is the baseline; parts 1.. are corrections in input order.
        pub left: usize,
        pub right: usize,
        /// Q(part) on the diagonal; twice the bilinear pairing off diagonal.
        pub value: ExactBounds,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct FormDecomposition {
        pub contributions: Vec<FormContribution>,
        pub expansion_sum: ExactBounds,
        pub direct_total: ExactBounds,
        /// Intersection of two valid evaluations of the same quadratic form.
        pub total: ExactBounds,
        /// Interval consistency check; dependence can make this nonzero-width.
        pub closure_residual: ExactBounds,
        /// Change from baseline, using the correction terms and direct values.
        pub change_from_baseline: ExactBounds,
        pub correction_cancellation: CancellationMeasure,
        /// Terms A_ii*x_i^2 and 2*A_ij*x_i*x_j, i<j, for the combined vector.
        pub coefficient_cancellation: CancellationMeasure,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialMeasurement {
        pub energy: ExactBounds,
        pub squared_norm: ExactBounds,
        pub rayleigh_quotient: ExactBounds,
        /// Exact rational midpoint and half-width of the Rayleigh enclosure.
        /// Conditional on supplied input enclosures; no extra error is inferred.
        pub rayleigh_midpoint: String,
        pub rayleigh_absolute_error_bound: String,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TrialEnergyReport {
        pub schema_version: u32,
        pub semantics: String,
        pub scope: String,
        pub problem_sha256: String,
        pub request_sha256: String,
        pub metric_check: String,
        pub part_labels: Vec<String>,
        pub energy: FormDecomposition,
        pub norm: FormDecomposition,
        pub baseline: TrialMeasurement,
        pub corrected: TrialMeasurement,
        pub rayleigh_change: ExactBounds,
        /// Every energy term divided by the same corrected squared norm.
        /// The baseline contribution here is generally not its own Rayleigh
        /// quotient, since that uses the baseline's squared norm instead.
        pub common_norm_energy_contributions: Vec<FormContribution>,
        pub common_norm_expansion_sum: ExactBounds,
        pub common_norm_closure_residual: ExactBounds,
        pub interval_operations: u64,
    }

    fn invalid(s: &str) -> SolverError {
        SolverError::InvalidConfiguration(format!("trial energy: {s}"))
    }
    fn limit(s: &str) -> SolverError {
        SolverError::IterationBudgetExhausted(format!("trial energy: {s}"))
    }
    pub(super) fn bounds(x: &I) -> ExactBounds {
        ExactBounds {
            lower: x.lower().to_string(),
            upper: x.upper().to_string(),
        }
    }
    pub(super) fn zero() -> I {
        I::point(Rational::from(0))
    }
    fn two() -> I {
        I::point(Rational::from(2))
    }
    pub(super) fn magnitude(x: &I) -> I {
        if x.contains_zero() {
            I::new(
                Rational::from(0),
                x.lower().clone().abs().max(x.upper().clone().abs()),
            )
            .unwrap()
        } else if x.is_strictly_negative() {
            x.neg()
        } else {
            x.clone()
        }
    }
    pub(super) fn hash<T: Serialize>(x: &T) -> Result<String, SolverError> {
        let bytes = serde_json::to_vec(x).map_err(|_| invalid("identity serialization"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
    pub(super) struct Work<'a> {
        pub(super) options: &'a TrialEnergyOptions,
        pub(super) token: &'a CancellationToken,
        pub(super) operations: u64,
    }
    impl Work<'_> {
        pub(super) fn cancel(&self) -> Result<(), SolverError> {
            if self.token.is_cancelled() {
                Err(SolverError::Cancelled("trial energy cancelled".into()))
            } else {
                Ok(())
            }
        }
        pub(super) fn tick(&mut self) -> Result<(), SolverError> {
            self.cancel()?;
            if self.operations >= self.options.maximum_interval_operations {
                return Err(limit("interval operation limit"));
            }
            self.operations += 1;
            Ok(())
        }
        fn rational(&self, x: &Rational) -> Result<(), SolverError> {
            if x.numer().significant_bits() > self.options.maximum_rational_bits
                || x.denom().significant_bits() > self.options.maximum_rational_bits
            {
                return Err(limit("rational bit-growth limit"));
            }
            Ok(())
        }
        pub(super) fn checked(&self, x: I) -> Result<I, SolverError> {
            self.rational(x.lower())?;
            self.rational(x.upper())?;
            Ok(x)
        }
        pub(super) fn add(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.checked(x.add(y))
        }
        pub(super) fn sub(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.checked(x.sub(y))
        }
        pub(super) fn mul(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.checked(x.mul(y))
        }
        pub(super) fn square(&mut self, x: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.checked(x.square())
        }
        pub(super) fn div(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.checked(x.div(y).map_err(|_| invalid("denominator contains zero"))?)
        }
        pub(super) fn scalar(&self, text: &str) -> Result<Rational, SolverError> {
            self.cancel()?;
            if text.is_empty()
                || text.len() > 131_072
                || text.bytes().any(|b| b.is_ascii_whitespace())
            {
                return Err(invalid("empty, oversized, or whitespace-containing scalar"));
            }
            let integer = |s: &str, signed: bool| -> Result<Integer, SolverError> {
                let digits = if signed {
                    s.strip_prefix('-')
                        .or_else(|| s.strip_prefix('+'))
                        .unwrap_or(s)
                } else {
                    s
                };
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid("invalid exact integer"));
                }
                Integer::from_str_radix(s, 10).map_err(|_| invalid("invalid exact integer"))
            };
            let x = if let Some((a, b)) = text.split_once('/') {
                let denominator = integer(b, false)?;
                if denominator <= 0 {
                    return Err(invalid("positive denominator required"));
                }
                Rational::from((integer(a, true)?, denominator))
            } else {
                let canonical = DecimalLiteral::new(text)
                    .and_then(|v| v.canonical())
                    .map_err(|_| invalid("finite decimal or rational required"))?;
                let (m, e) = canonical
                    .as_str()
                    .split_once('e')
                    .unwrap_or((canonical.as_str(), "0"));
                let e: i32 = e.parse().map_err(|_| invalid("decimal exponent limit"))?;
                if e.unsigned_abs() > 16_384 {
                    return Err(invalid("decimal exponent limit"));
                }
                let scale = Integer::from(10).pow(e.unsigned_abs());
                if e >= 0 {
                    Rational::from(integer(m, true)? * scale)
                } else {
                    Rational::from((integer(m, true)?, scale))
                }
            };
            self.rational(&x)?;
            Ok(x)
        }
        pub(super) fn read(&self, x: &ExactBounds) -> Result<I, SolverError> {
            I::new(self.scalar(&x.lower)?, self.scalar(&x.upper)?)
                .map_err(|_| invalid("reversed interval endpoints"))
        }
    }

    fn validate(
        problem: &TrialEnergyProblem,
        options: &TrialEnergyOptions,
    ) -> Result<(), SolverError> {
        if options.maximum_dimension == 0
            || options.maximum_dimension > 512
            || options.maximum_parts == 0
            || options.maximum_parts > 16
            || options.maximum_input_bytes == 0
            || options.maximum_input_bytes > 134_217_728
            || options.maximum_interval_operations > 1_000_000_000
            || !(64..=65_536).contains(&options.maximum_rational_bits)
        {
            return Err(invalid("unsupported work limits"));
        }
        let n = problem.operator.dimension;
        if problem.schema_version != 1
            || n == 0
            || n > options.maximum_dimension
            || problem.metric.dimension != n
            || problem.corrections.len() >= options.maximum_parts
        {
            return Err(invalid("schema, dimension or part limit"));
        }
        let id = |s: &str| -> Result<(), SolverError> {
            if s.trim().is_empty() || s.len() > 4096 {
                Err(invalid("nonempty bounded identity required"))
            } else {
                Ok(())
            }
        };
        let a = &problem.operator;
        let g = &problem.metric;
        for form in [a, g] {
            id(&form.source_id)?;
            id(&form.basis_id)?;
            id(&form.normalization_id)?;
            if form.entries.len() != n * n {
                return Err(invalid("matrix shape"));
            }
        }
        if a.basis_id != g.basis_id || a.normalization_id != g.normalization_id {
            return Err(invalid("operator/metric coordinate mismatch"));
        }
        let mut labels = std::collections::BTreeSet::new();
        for v in std::iter::once(&problem.baseline).chain(&problem.corrections) {
            id(&v.label)?;
            id(&v.source_id)?;
            if !labels.insert(&v.label)
                || v.coefficients.len() != n
                || v.operator_id != a.source_id
                || v.metric_id != g.source_id
                || v.basis_id != a.basis_id
                || v.normalization_id != a.normalization_id
            {
                return Err(invalid(
                    "vector label, shape, matrix, metric or coordinate mismatch",
                ));
            }
        }
        let scalar_bytes = a
            .entries
            .iter()
            .chain(&g.entries)
            .chain(
                std::iter::once(&problem.baseline)
                    .chain(&problem.corrections)
                    .flat_map(|v| &v.coefficients),
            )
            .try_fold(0usize, |sum, v| {
                sum.checked_add(v.lower.len())?.checked_add(v.upper.len())
            })
            .ok_or_else(|| invalid("input byte overflow"))?;
        if scalar_bytes > options.maximum_input_bytes {
            return Err(limit("input scalar byte limit"));
        }
        Ok(())
    }

    pub(super) fn read_form(form: &TrialForm, work: &Work<'_>) -> Result<Vec<I>, SolverError> {
        let values: Vec<_> = form
            .entries
            .iter()
            .map(|v| work.read(v))
            .collect::<Result<_, _>>()?;
        for i in 0..form.dimension {
            for j in 0..i {
                if values[i * form.dimension + j] != values[j * form.dimension + i] {
                    return Err(invalid("form intervals are not exactly symmetric"));
                }
            }
        }
        Ok(values)
    }

    // Interval LDL without pivoting: positive pivots prove SPD for every
    // symmetric metric in the input box. An unresolved pivot refuses a claim.
    pub(super) fn metric_check(g: &[I], n: usize, w: &mut Work<'_>) -> Result<String, SolverError> {
        let diagonal = (0..n).all(|i| (0..n).all(|j| i == j || g[i * n + j] == zero()));
        if diagonal {
            if (0..n).any(|i| !g[i * n + i].is_strictly_positive()) {
                return Err(invalid("metric positive definiteness unresolved"));
            }
            return Ok("positive_diagonal_interval_metric".into());
        }
        let mut l = vec![zero(); n * n];
        let mut d = vec![zero(); n];
        for i in 0..n {
            w.cancel()?;
            let mut pivot = g[i * n + i].clone();
            for k in 0..i {
                let square = w.square(&l[i * n + k])?;
                let term = w.mul(&square, &d[k])?;
                pivot = w.sub(&pivot, &term)?;
            }
            if !pivot.is_strictly_positive() {
                return Err(invalid("metric positive definiteness unresolved"));
            }
            d[i] = pivot;
            for j in i + 1..n {
                let mut value = g[j * n + i].clone();
                for k in 0..i {
                    let product = w.mul(&l[j * n + k], &l[i * n + k])?;
                    let term = w.mul(&product, &d[k])?;
                    value = w.sub(&value, &term)?;
                }
                l[j * n + i] = w.div(&value, &d[i])?;
            }
        }
        Ok("positive_interval_ldl_metric".into())
    }

    fn pairing(
        a: &[I],
        x: &[I],
        y: &[I],
        same: bool,
        w: &mut Work<'_>,
    ) -> Result<(I, I), SolverError> {
        let n = x.len();
        let mut value = zero();
        let mut absolute = zero();
        for i in 0..n {
            for j in i..n {
                w.cancel()?;
                if a[i * n + j] == zero() {
                    continue;
                }
                let product = if same {
                    if i == j {
                        w.square(&x[i])?
                    } else {
                        let p = w.mul(&x[i], &x[j])?;
                        w.mul(&two(), &p)?
                    }
                } else if i == j {
                    w.mul(&x[i], &y[i])?
                } else {
                    let p = w.mul(&x[i], &y[j])?;
                    let q = w.mul(&x[j], &y[i])?;
                    w.add(&p, &q)?
                };
                let term = w.mul(&a[i * n + j], &product)?;
                value = w.add(&value, &term)?;
                absolute = w.add(&absolute, &magnitude(&term))?;
            }
        }
        Ok((value, absolute))
    }
    fn intersection(x: &I, y: &I) -> Result<I, SolverError> {
        x.intersection(y).ok_or_else(|| {
            SolverError::NumericalBreakdown("trial form evaluations do not overlap".into())
        })
    }
    fn cancellation(
        sum: &I,
        total: &I,
        w: &mut Work<'_>,
    ) -> Result<CancellationMeasure, SolverError> {
        let (status, ratio) = if sum == &zero() {
            (CancellationStatus::AllTermsZero, None)
        } else if total == &zero() {
            (CancellationStatus::ExactZeroTotal, None)
        } else if total.contains_zero() {
            (CancellationStatus::TotalContainsZero, None)
        } else {
            let ratio = w.div(sum, &magnitude(total))?;
            if ratio.upper() < &1 {
                return Err(SolverError::NumericalBreakdown(
                    "invalid cancellation ratio".into(),
                ));
            }
            let ratio = I::new(
                ratio.lower().clone().max(Rational::from(1)),
                ratio.upper().clone(),
            )
            .unwrap();
            (CancellationStatus::Resolved, Some(bounds(&ratio)))
        };
        Ok(CancellationMeasure {
            absolute_term_sum: bounds(sum),
            amplification: ratio,
            status,
        })
    }
    struct Evaluated {
        report: FormDecomposition,
        baseline: I,
        total: I,
    }
    fn decompose(
        a: &[I],
        parts: &[Vec<I>],
        combined: &[I],
        w: &mut Work<'_>,
    ) -> Result<Evaluated, SolverError> {
        let count = parts.len();
        let mut pairings = vec![zero(); count * count];
        for i in 0..count {
            for j in i..count {
                pairings[i * count + j] = pairing(a, &parts[i], &parts[j], i == j, w)?.0;
            }
        }
        let (direct, coefficient_absolute) = pairing(a, combined, combined, true, w)?;
        decompose_pairings(&pairings, count, direct, coefficient_absolute, w)
    }
    fn decompose_pairings(
        pairings: &[I],
        count: usize,
        direct: I,
        coefficient_absolute: I,
        w: &mut Work<'_>,
    ) -> Result<Evaluated, SolverError> {
        let mut contributions = Vec::new();
        let mut expansion = zero();
        let mut change = zero();
        let mut absolute = zero();
        let mut baseline = zero();
        for i in 0..count {
            for j in i..count {
                let mut value = pairings[i * count + j].clone();
                if i != j {
                    value = w.mul(&two(), &value)?;
                }
                if i == 0 && j == 0 {
                    baseline = value.clone();
                } else {
                    change = w.add(&change, &value)?;
                }
                expansion = w.add(&expansion, &value)?;
                absolute = w.add(&absolute, &magnitude(&value))?;
                contributions.push(FormContribution {
                    left: i,
                    right: j,
                    value: bounds(&value),
                });
            }
        }
        let total = intersection(&direct, &expansion)?;
        let closure = w.sub(&direct, &expansion)?;
        let difference = w.sub(&total, &baseline)?;
        let change = intersection(&change, &difference)?;
        let correction_cancellation = cancellation(&absolute, &total, w)?;
        let coefficient_cancellation = cancellation(&coefficient_absolute, &total, w)?;
        let report = FormDecomposition {
            contributions,
            expansion_sum: bounds(&expansion),
            direct_total: bounds(&direct),
            total: bounds(&total),
            closure_residual: bounds(&closure),
            change_from_baseline: bounds(&change),
            correction_cancellation,
            coefficient_cancellation,
        };
        Ok(Evaluated {
            report,
            baseline,
            total,
        })
    }
    fn measurement(
        energy: &I,
        norm: &I,
        w: &mut Work<'_>,
    ) -> Result<TrialMeasurement, SolverError> {
        if !norm.is_strictly_positive() {
            return Err(invalid("trial normalization is not separated from zero"));
        }
        let quotient = w.div(energy, norm)?;
        let midpoint = quotient.midpoint();
        w.rational(&midpoint)?;
        let radius = quotient.width() / 2;
        w.rational(&radius)?;
        Ok(TrialMeasurement {
            energy: bounds(energy),
            squared_norm: bounds(norm),
            rayleigh_quotient: bounds(&quotient),
            rayleigh_midpoint: midpoint.to_string(),
            rayleigh_absolute_error_bound: radius.to_string(),
        })
    }

    /// Analyze baseline plus all corrections using one form and one SPD metric.
    /// Exact point inputs produce zero-width bounds; interval inputs may be
    /// conservative because their dependencies are not asserted. Failure returns
    /// no report. This function does not select an eigenstate or predict a root.
    pub fn analyze_trial_energy(
        problem: &TrialEnergyProblem,
        options: &TrialEnergyOptions,
        token: &CancellationToken,
    ) -> Result<TrialEnergyReport, SolverError> {
        let mut w = Work {
            options,
            token,
            operations: 0,
        };
        w.cancel()?;
        validate(problem, options)?;
        let a = read_form(&problem.operator, &w)?;
        let g = read_form(&problem.metric, &w)?;
        let n = problem.operator.dimension;
        let metric_check = metric_check(&g, n, &mut w)?;
        let vectors: Vec<_> = std::iter::once(&problem.baseline)
            .chain(&problem.corrections)
            .collect();
        let parts: Vec<Vec<I>> = vectors
            .iter()
            .map(|v| v.coefficients.iter().map(|x| w.read(x)).collect())
            .collect::<Result<_, _>>()?;
        let mut combined = vec![zero(); n];
        for part in &parts {
            for (x, y) in combined.iter_mut().zip(part) {
                *x = w.add(x, y)?;
            }
        }
        let energy = decompose(&a, &parts, &combined, &mut w)?;
        let norm = decompose(&g, &parts, &combined, &mut w)?;
        let baseline = measurement(&energy.baseline, &norm.baseline, &mut w)?;
        let corrected = measurement(&energy.total, &norm.total, &mut w)?;
        let before = w.read(&baseline.rayleigh_quotient)?;
        let after = w.read(&corrected.rayleigh_quotient)?;
        let rayleigh_change = bounds(&w.sub(&after, &before)?);
        let mut common_norm_energy_contributions = Vec::new();
        let mut normalized_sum = zero();
        for term in &energy.report.contributions {
            let value = w.read(&term.value)?;
            let normalized = w.div(&value, &norm.total)?;
            normalized_sum = w.add(&normalized_sum, &normalized)?;
            common_norm_energy_contributions.push(FormContribution {
                left: term.left,
                right: term.right,
                value: bounds(&normalized),
            });
        }
        let normalized_closure = w.sub(&normalized_sum, &after)?;
        if !normalized_closure.contains_zero() {
            return Err(SolverError::NumericalBreakdown(
                "normalized trial form expansion does not close".into(),
            ));
        }
        w.cancel()?;
        Ok(TrialEnergyReport {
            schema_version: 1,
            semantics: SEMANTICS.into(),
            scope: SCOPE.into(),
            problem_sha256: hash(&(SEMANTICS, problem))?,
            request_sha256: hash(&(SEMANTICS, problem, options))?,
            metric_check,
            part_labels: vectors.iter().map(|v| v.label.clone()).collect(),
            energy: energy.report,
            norm: norm.report,
            baseline,
            corrected,
            rayleigh_change,
            common_norm_energy_contributions,
            common_norm_expansion_sum: bounds(&normalized_sum),
            common_norm_closure_residual: bounds(&normalized_closure),
            interval_operations: w.operations,
        })
    }

    /// Streaming contractions of a supplied finite symmetric form. The stream
    /// is the packed upper triangle in row order; it defines both triangles.
    #[doc(hidden)]
    pub mod streaming {
        use super::*;
        use rug::{float::Round, Float};
        use std::io::{BufRead, Read};

        pub const STREAM_SEMANTICS: &str = "packed-upper-trial-contractions-v1";
        pub const STREAM_SCOPE: &str =
            "finite_declared_interval_forms_digest_bound_no_assembly_or_continuum_certificate";

        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct FormMetadata {
            pub source_id: String,
            pub basis_id: String,
            pub normalization_id: String,
            pub dimension: usize,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct FormSource {
            pub metadata: FormMetadata,
            /// Digest from fingerprint(), retained independently of replay.
            pub sha256: String,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct FormBlock {
            pub start: u64,
            pub entries: Vec<ExactBounds>,
        }
        /// Readers must honor the requested block limit. The consumer checks
        /// order, coverage, endpoint validity, byte limits and the final digest.
        /// A reader's own allocation/I/O behavior remains its responsibility.
        pub trait FormReader {
            fn next_block(
                &mut self,
                maximum_entries: usize,
                maximum_scalar_bytes: usize,
            ) -> Result<Option<FormBlock>, SolverError>;
        }

        /// One ExactBounds JSON object per line, with no header or blank lines.
        /// A bounded line read prevents an oversized record from growing without
        /// limit. Digest identity is independent of JSON whitespace and blocks,
        /// but binds the exact endpoint strings ("1" and "1.0" differ).
        pub struct JsonLinesReader<R: BufRead> {
            reader: R,
            position: u64,
            pending: Option<ExactBounds>,
        }
        impl<R: BufRead> JsonLinesReader<R> {
            pub fn new(reader: R) -> Self {
                Self {
                    reader,
                    position: 0,
                    pending: None,
                }
            }
        }
        impl<R: BufRead> FormReader for JsonLinesReader<R> {
            fn next_block(
                &mut self,
                maximum_entries: usize,
                maximum_scalar_bytes: usize,
            ) -> Result<Option<FormBlock>, SolverError> {
                if maximum_entries == 0
                    || maximum_entries > 16_384
                    || maximum_scalar_bytes == 0
                    || maximum_scalar_bytes > 134_217_728
                {
                    return Err(invalid("stream block limit"));
                }
                let mut entries = Vec::new();
                let mut bytes = 0usize;
                for _ in 0..maximum_entries {
                    let value = if let Some(value) = self.pending.take() {
                        value
                    } else {
                        let mut line = Vec::new();
                        const MAX_LINE: u64 = 262_400;
                        let read = self
                            .reader
                            .by_ref()
                            .take(MAX_LINE + 1)
                            .read_until(b'\n', &mut line)
                            .map_err(|_| invalid("stream read failed"))?;
                        if read == 0 {
                            break;
                        }
                        if read as u64 > MAX_LINE {
                            return Err(limit("stream JSON record limit"));
                        }
                        serde_json::from_slice::<ExactBounds>(&line)
                            .map_err(|_| invalid("invalid stream JSON record"))?
                    };
                    let size = value.lower.len() + value.upper.len();
                    if size > maximum_scalar_bytes {
                        return Err(limit("stream record scalar byte limit"));
                    }
                    if bytes + size > maximum_scalar_bytes {
                        self.pending = Some(value);
                        break;
                    }
                    bytes += size;
                    entries.push(value);
                }
                if entries.is_empty() {
                    return Ok(None);
                }
                let start = self.position;
                self.position = self
                    .position
                    .checked_add(entries.len() as u64)
                    .ok_or_else(|| invalid("stream position overflow"))?;
                Ok(Some(FormBlock { start, entries }))
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        pub enum Metric {
            Identity {
                metadata: FormMetadata,
            },
            Diagonal {
                metadata: FormMetadata,
                entries: Vec<ExactBounds>,
            },
            Stream {
                source: FormSource,
            },
        }
        impl Metric {
            fn metadata(&self) -> &FormMetadata {
                match self {
                    Self::Identity { metadata } | Self::Diagonal { metadata, .. } => metadata,
                    Self::Stream { source } => &source.metadata,
                }
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Problem {
            pub schema_version: u32,
            pub operator: FormSource,
            pub metric: Metric,
            pub baseline: TrialVector,
            pub corrections: Vec<TrialVector>,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Options {
            pub maximum_dimension: usize,
            pub maximum_parts: usize,
            pub maximum_block_entries: usize,
            pub maximum_block_scalar_bytes: usize,
            /// Total packed entries across the operator and streamed metric.
            pub maximum_stream_entries: u64,
            /// Endpoint-string bytes across all matrix streams.
            pub maximum_stream_scalar_bytes: u64,
            /// Vector and inline-metric endpoint-string bytes.
            pub maximum_vector_scalar_bytes: usize,
            pub maximum_interval_operations: u64,
            pub maximum_rational_bits: u32,
            /// Directed rounding is used only for reported square roots.
            pub norm_precision_bits: u32,
        }
        impl Default for Options {
            fn default() -> Self {
                Self {
                    maximum_dimension: 16_384,
                    maximum_parts: 8,
                    maximum_block_entries: 4096,
                    maximum_block_scalar_bytes: 16_777_216,
                    maximum_stream_entries: 67_108_864,
                    maximum_stream_scalar_bytes: 17_179_869_184,
                    maximum_vector_scalar_bytes: 67_108_864,
                    maximum_interval_operations: 1_000_000_000,
                    maximum_rational_bits: 32_768,
                    norm_precision_bits: 256,
                }
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct StreamRecord {
            pub metadata: FormMetadata,
            pub sha256: String,
            pub entries: u64,
            pub scalar_bytes: u64,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum MetricStatus {
            Identity,
            PositiveDiagonal,
            PositiveStrictDiagonalDominance,
            PositiveOnTrialSpan,
            Unresolved,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum MeasurementStatus {
            CertifiedFinite,
            MetricUnresolved,
            NormNotSeparatedFromZero,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Measurement {
            pub energy: ExactBounds,
            /// A symmetric form value; it is a norm only under metric_status.
            pub squared_norm: ExactBounds,
            pub norm: Option<ExactBounds>,
            pub status: MeasurementStatus,
            pub normalized: Option<TrialMeasurement>,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Stage {
            /// Baseline followed by this many corrections, in request order.
            pub corrections_applied: usize,
            pub measurement: Measurement,
            pub energy_change_from_previous: Option<ExactBounds>,
            pub squared_norm_change_from_previous: Option<ExactBounds>,
            pub rayleigh_change_from_previous: Option<ExactBounds>,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Report {
            pub semantics: String,
            pub scope: String,
            pub problem_sha256: String,
            pub request_sha256: String,
            pub operator: StreamRecord,
            pub metric_stream: Option<StreamRecord>,
            pub metric_status: MetricStatus,
            /// Certified global lower bound only for diagonal/dominance checks.
            pub metric_global_lower_bound: Option<String>,
            pub part_labels: Vec<String>,
            pub parts: Vec<Measurement>,
            pub stages: Vec<Stage>,
            /// Row-major small matrices of all bilinear pairings, not twice
            /// their off-diagonal entries. Parts need not be independent.
            pub energy_pairings: Vec<ExactBounds>,
            pub metric_pairings: Vec<ExactBounds>,
            pub energy: FormDecomposition,
            pub norm: FormDecomposition,
            /// Original-coordinate absolute term sums for each small pairing.
            /// The summation is over all ordered matrix coordinates i,j.
            pub energy_pairing_cancellation: Vec<CancellationMeasure>,
            pub metric_pairing_cancellation: Vec<CancellationMeasure>,
            pub common_norm_energy_contributions: Option<Vec<FormContribution>>,
            pub common_norm_closure_residual: Option<ExactBounds>,
            pub interval_operations: u64,
            pub vector_scalar_bytes: usize,
            pub streamed_entries: u64,
            pub streamed_scalar_bytes: u64,
        }

        fn options(o: &Options) -> Result<TrialEnergyOptions, SolverError> {
            if o.maximum_dimension == 0
                || o.maximum_dimension > 65_536
                || o.maximum_parts == 0
                || o.maximum_parts > 16
                || o.maximum_block_entries == 0
                || o.maximum_block_entries > 16_384
                || o.maximum_block_scalar_bytes == 0
                || o.maximum_block_scalar_bytes > 134_217_728
                || o.maximum_stream_entries == 0
                || o.maximum_stream_entries > 4_295_032_832
                || o.maximum_stream_scalar_bytes == 0
                || o.maximum_stream_scalar_bytes > 1_099_511_627_776
                || o.maximum_vector_scalar_bytes == 0
                || o.maximum_vector_scalar_bytes > 134_217_728
                || o.maximum_interval_operations > 10_000_000_000
                || !(64..=65_536).contains(&o.maximum_rational_bits)
                || !(64..=16_384).contains(&o.norm_precision_bits)
            {
                return Err(invalid("unsupported streaming limits"));
            }
            Ok(TrialEnergyOptions {
                maximum_dimension: o.maximum_dimension,
                maximum_parts: o.maximum_parts,
                maximum_input_bytes: o.maximum_vector_scalar_bytes,
                maximum_interval_operations: o.maximum_interval_operations,
                maximum_rational_bits: o.maximum_rational_bits,
            })
        }
        fn identity(s: &str) -> Result<(), SolverError> {
            if s.trim().is_empty() || s.len() > 4096 {
                Err(invalid("stream source identity"))
            } else {
                Ok(())
            }
        }
        fn metadata(m: &FormMetadata, o: &Options) -> Result<u64, SolverError> {
            identity(&m.source_id)?;
            identity(&m.basis_id)?;
            identity(&m.normalization_id)?;
            if m.dimension == 0 || m.dimension > o.maximum_dimension {
                return Err(invalid("stream dimension"));
            }
            let n = m.dimension as u64;
            Ok(n * (n + 1) / 2)
        }
        fn digest(s: &str) -> Result<(), SolverError> {
            if s.len() != 64
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                Err(invalid("lowercase SHA256 digest required"))
            } else {
                Ok(())
            }
        }
        #[derive(Default)]
        struct Consumption {
            entries: u64,
            bytes: u64,
        }

        // The digest binds domain+metadata followed by length-prefixed endpoint
        // strings. The complete upper triangle is mandatory, including zeros.
        fn consume(
            m: &FormMetadata,
            reader: &mut dyn FormReader,
            o: &Options,
            used: &mut Consumption,
            w: &mut Work<'_>,
            mut entry: impl FnMut(usize, usize, I, &mut Work<'_>) -> Result<(), SolverError>,
        ) -> Result<StreamRecord, SolverError> {
            let count = metadata(m, o)?;
            if count > o.maximum_stream_entries.saturating_sub(used.entries) {
                return Err(limit("stream entry limit"));
            }
            let mut sha = Sha256::new();
            let head = serde_json::to_vec(&(STREAM_SEMANTICS, m))
                .map_err(|_| invalid("stream metadata serialization"))?;
            sha.update((head.len() as u64).to_le_bytes());
            sha.update(head);
            let mut record = StreamRecord {
                metadata: m.clone(),
                sha256: String::new(),
                entries: 0,
                scalar_bytes: 0,
            };
            let (mut i, mut j) = (0usize, 0usize);
            loop {
                w.cancel()?;
                let block =
                    reader.next_block(o.maximum_block_entries, o.maximum_block_scalar_bytes)?;
                w.cancel()?;
                let Some(block) = block else {
                    break;
                };
                let block_bytes = block
                    .entries
                    .iter()
                    .try_fold(0usize, |sum, b| {
                        sum.checked_add(b.lower.len())?.checked_add(b.upper.len())
                    })
                    .ok_or_else(|| limit("stream block byte overflow"))?;
                if block_bytes > o.maximum_block_scalar_bytes {
                    return Err(limit("stream block scalar byte limit"));
                }
                if block.start != record.entries
                    || block.entries.is_empty()
                    || block.entries.len() > o.maximum_block_entries
                    || block.entries.len() as u64 > count - record.entries
                {
                    return Err(invalid("stream order, coverage or block shape"));
                }
                for b in block.entries {
                    w.cancel()?;
                    let bytes = b.lower.len() as u64 + b.upper.len() as u64;
                    if bytes > o.maximum_stream_scalar_bytes.saturating_sub(used.bytes) {
                        return Err(limit("stream scalar byte limit"));
                    }
                    used.bytes += bytes;
                    record.scalar_bytes += bytes;
                    let value = w.read(&b)?;
                    for s in [&b.lower, &b.upper] {
                        sha.update((s.len() as u64).to_le_bytes());
                        sha.update(s.as_bytes());
                    }
                    entry(i, j, value, w)?;
                    used.entries += 1;
                    record.entries += 1;
                    j += 1;
                    if j == m.dimension {
                        i += 1;
                        j = i;
                    }
                }
            }
            if record.entries != count {
                return Err(invalid("incomplete symmetric form stream"));
            }
            record.sha256 = format!("{:x}", sha.finalize());
            Ok(record)
        }
        /// Import-time digest. Keep the result in a source manifest before
        /// analysis/replay; computing it from the replay input cannot establish
        /// independent provenance. This scans and validates every endpoint.
        pub fn fingerprint(
            m: &FormMetadata,
            reader: &mut dyn FormReader,
            o: &Options,
            token: &CancellationToken,
        ) -> Result<StreamRecord, SolverError> {
            let opts = options(o)?;
            let mut w = Work {
                options: &opts,
                token,
                operations: 0,
            };
            w.cancel()?;
            consume(
                m,
                reader,
                o,
                &mut Consumption::default(),
                &mut w,
                |_, _, _, _| Ok(()),
            )
        }

        struct Contracted {
            pairings: Vec<I>,
            absolute: Vec<I>,
            direct: I,
            direct_absolute: I,
            global_lower: Option<Rational>,
        }
        // Columns include the original parts and their separately assembled sum.
        // Exact interval matvecs cost O(n^2 p); the final contractions O(n p^2).
        // Neither the source matrix nor an n-by-n interval copy is retained.
        struct Accumulator<'a> {
            columns: &'a [Vec<I>],
            applied: Vec<Vec<I>>,
            absolute_applied: Vec<Vec<I>>,
            diagonal_lower: Vec<Rational>,
            off_diagonal_upper: Vec<I>,
            check_metric: bool,
        }
        impl<'a> Accumulator<'a> {
            fn new(columns: &'a [Vec<I>], check_metric: bool) -> Self {
                let n = columns[0].len();
                Self {
                    columns,
                    applied: vec![vec![zero(); n]; columns.len()],
                    absolute_applied: vec![vec![zero(); n]; columns.len()],
                    diagonal_lower: vec![Rational::from(0); n],
                    off_diagonal_upper: vec![zero(); n],
                    check_metric,
                }
            }
            fn entry(
                &mut self,
                i: usize,
                j: usize,
                a: I,
                w: &mut Work<'_>,
            ) -> Result<(), SolverError> {
                if self.check_metric {
                    if i == j {
                        self.diagonal_lower[i] = a.lower().clone();
                    } else {
                        let upper = I::point(magnitude(&a).upper().clone());
                        self.off_diagonal_upper[i] = w.add(&self.off_diagonal_upper[i], &upper)?;
                        self.off_diagonal_upper[j] = w.add(&self.off_diagonal_upper[j], &upper)?;
                    }
                }
                if a == zero() {
                    return Ok(());
                }
                for c in 0..self.columns.len() {
                    let v = w.mul(&a, &self.columns[c][j])?;
                    self.applied[c][i] = w.add(&self.applied[c][i], &v)?;
                    self.absolute_applied[c][i] =
                        w.add(&self.absolute_applied[c][i], &magnitude(&v))?;
                    if i != j {
                        let v = w.mul(&a, &self.columns[c][i])?;
                        self.applied[c][j] = w.add(&self.applied[c][j], &v)?;
                        self.absolute_applied[c][j] =
                            w.add(&self.absolute_applied[c][j], &magnitude(&v))?;
                    }
                }
                Ok(())
            }
            fn contract(
                &self,
                left: usize,
                right: usize,
                w: &mut Work<'_>,
            ) -> Result<(I, I), SolverError> {
                let mut value = zero();
                let mut absolute = zero();
                for i in 0..self.columns[0].len() {
                    let term = w.mul(&self.columns[left][i], &self.applied[right][i])?;
                    value = w.add(&value, &term)?;
                    let term = w.mul(
                        &magnitude(&self.columns[left][i]),
                        &self.absolute_applied[right][i],
                    )?;
                    absolute = w.add(&absolute, &term)?;
                }
                Ok((value, absolute))
            }
            fn finish(self, w: &mut Work<'_>) -> Result<Contracted, SolverError> {
                let p = self.columns.len() - 1;
                let mut pairings = vec![zero(); p * p];
                let mut absolute = pairings.clone();
                for a in 0..p {
                    for b in a..p {
                        let (mut v, mut abs) = self.contract(a, b, w)?;
                        if a != b {
                            let (reverse, reverse_abs) = self.contract(b, a, w)?;
                            v = intersection(&v, &reverse)?;
                            // Symmetry makes both ordered-coordinate absolute
                            // sums the same real quantity, also for interval data.
                            abs = intersection(&abs, &reverse_abs)?;
                        }
                        pairings[a * p + b] = v.clone();
                        pairings[b * p + a] = v;
                        absolute[a * p + b] = abs.clone();
                        absolute[b * p + a] = abs;
                    }
                }
                let (direct, direct_absolute) = self.contract(p, p, w)?;
                let mut lower: Option<Rational> = None;
                if self.check_metric {
                    for (d, r) in self.diagonal_lower.iter().zip(&self.off_diagonal_upper) {
                        let bound = w.sub(&I::point(d.clone()), r)?;
                        lower = Some(lower.map_or_else(
                            || bound.lower().clone(),
                            |v| v.min(bound.lower().clone()),
                        ));
                    }
                }
                Ok(Contracted {
                    pairings,
                    absolute,
                    direct,
                    direct_absolute,
                    global_lower: lower,
                })
            }
        }

        fn validate_problem(p: &Problem, o: &Options) -> Result<usize, SolverError> {
            let a = &p.operator.metadata;
            let g = p.metric.metadata();
            metadata(a, o)?;
            metadata(g, o)?;
            digest(&p.operator.sha256)?;
            if let Metric::Stream { source } = &p.metric {
                digest(&source.sha256)?;
            }
            if p.schema_version != 1
                || p.corrections.len() >= o.maximum_parts
                || a.dimension != g.dimension
                || a.basis_id != g.basis_id
                || a.normalization_id != g.normalization_id
            {
                return Err(invalid("stream schema, coordinates or part limit"));
            }
            let mut labels = std::collections::BTreeSet::new();
            let mut bytes = 0usize;
            let mut add_bytes = |b: &ExactBounds| -> Result<(), SolverError> {
                bytes = bytes
                    .checked_add(b.lower.len())
                    .and_then(|v| v.checked_add(b.upper.len()))
                    .ok_or_else(|| limit("vector scalar byte overflow"))?;
                if bytes > o.maximum_vector_scalar_bytes {
                    return Err(limit("vector scalar byte limit"));
                }
                Ok(())
            };
            for v in std::iter::once(&p.baseline).chain(&p.corrections) {
                identity(&v.label)?;
                identity(&v.source_id)?;
                if !labels.insert(&v.label)
                    || v.coefficients.len() != a.dimension
                    || v.operator_id != a.source_id
                    || v.metric_id != g.source_id
                    || v.basis_id != a.basis_id
                    || v.normalization_id != a.normalization_id
                {
                    return Err(invalid("stream vector identity or shape"));
                }
                for b in &v.coefficients {
                    add_bytes(b)?;
                }
            }
            if let Metric::Diagonal { entries, .. } = &p.metric {
                if entries.len() != a.dimension {
                    return Err(invalid("diagonal metric shape"));
                }
                for b in entries {
                    add_bytes(b)?;
                }
            }
            Ok(bytes)
        }
        fn metric_status(
            g: &Contracted,
            p: usize,
            metric: &Metric,
            w: &mut Work<'_>,
        ) -> Result<MetricStatus, SolverError> {
            if matches!(metric, Metric::Identity { .. }) {
                return Ok(MetricStatus::Identity);
            }
            if g.global_lower.as_ref().is_some_and(|v| v > &0) {
                return Ok(if matches!(metric, Metric::Diagonal { .. }) {
                    MetricStatus::PositiveDiagonal
                } else {
                    MetricStatus::PositiveStrictDiagonalDominance
                });
            }
            // An unresolved/negative LDL pivot is evidence of no certificate,
            // not a proof of indefiniteness. Work/cancellation errors propagate.
            match super::metric_check(&g.pairings, p, w) {
                Ok(_) => Ok(MetricStatus::PositiveOnTrialSpan),
                Err(SolverError::InvalidConfiguration(_)) => Ok(MetricStatus::Unresolved),
                Err(e) => Err(e),
            }
        }
        fn measured(
            energy: &I,
            norm: &I,
            metric: &MetricStatus,
            o: &Options,
            w: &mut Work<'_>,
        ) -> Result<Measurement, SolverError> {
            let mut out = Measurement {
                energy: bounds(energy),
                squared_norm: bounds(norm),
                norm: None,
                status: MeasurementStatus::MetricUnresolved,
                normalized: None,
            };
            if *metric == MetricStatus::Unresolved {
                return Ok(out);
            }
            if norm.upper() < &0 {
                return Err(SolverError::NumericalBreakdown(
                    "negative norm with positive metric".into(),
                ));
            }
            let lo = norm.lower().clone().max(Rational::from(0));
            let mut low = Float::with_val_round(o.norm_precision_bits, &lo, Round::Down).0;
            let mut high = Float::with_val_round(o.norm_precision_bits, norm.upper(), Round::Up).0;
            low.sqrt_round(Round::Down);
            high.sqrt_round(Round::Up);
            let low = low
                .to_rational()
                .ok_or_else(|| limit("norm square-root range"))?;
            let high = high
                .to_rational()
                .ok_or_else(|| limit("norm square-root range"))?;
            // A rounded nonzero upper endpoint may never silently become zero.
            if norm.upper() > &0 && high == 0 {
                return Err(limit("norm square-root underflow"));
            }
            let norm_bound = w.checked(I::new(low, high).map_err(|_| invalid("norm bounds"))?)?;
            out.norm = Some(bounds(&norm_bound));
            if norm.is_strictly_positive() {
                out.normalized = Some(super::measurement(energy, norm, w)?);
                out.status = MeasurementStatus::CertifiedFinite;
            } else {
                out.status = MeasurementStatus::NormNotSeparatedFromZero;
            }
            Ok(out)
        }
        fn prefix(
            pairings: &[I],
            p: usize,
            count: usize,
            w: &mut Work<'_>,
        ) -> Result<I, SolverError> {
            let mut value = zero();
            for i in 0..count {
                for j in i..count {
                    let term = if i == j {
                        pairings[i * p + j].clone()
                    } else {
                        w.mul(&two(), &pairings[i * p + j])?
                    };
                    value = w.add(&value, &term)?;
                }
            }
            Ok(value)
        }

        /// All matrix input is consumed and digest-checked before returning a
        /// report. No partial report is returned after failure or cancellation.
        /// Unresolved metric/norm gates retain raw finite diagnostics but never
        /// a certified normalized measurement. Zero/dependent corrections work.
        pub fn analyze(
            problem: &Problem,
            o: &Options,
            operator: &mut dyn FormReader,
            metric_reader: Option<&mut dyn FormReader>,
            token: &CancellationToken,
        ) -> Result<Report, SolverError> {
            let opts = options(o)?;
            let mut w = Work {
                options: &opts,
                token,
                operations: 0,
            };
            w.cancel()?;
            let vector_scalar_bytes = validate_problem(problem, o)?;
            if matches!(problem.metric, Metric::Stream { .. }) != metric_reader.is_some() {
                return Err(invalid("metric reader presence mismatch"));
            }
            let n = problem.operator.metadata.dimension;
            let p = 1 + problem.corrections.len();
            let vectors: Vec<_> = std::iter::once(&problem.baseline)
                .chain(&problem.corrections)
                .collect();
            let mut columns: Vec<Vec<I>> = vectors
                .iter()
                .map(|v| v.coefficients.iter().map(|b| w.read(b)).collect())
                .collect::<Result<_, _>>()?;
            let mut sum = vec![zero(); n];
            for v in &columns {
                for (s, x) in sum.iter_mut().zip(v) {
                    *s = w.add(s, x)?;
                }
            }
            columns.push(sum);
            let mut used = Consumption::default();
            let mut a = Accumulator::new(&columns, false);
            let source = consume(
                &problem.operator.metadata,
                operator,
                o,
                &mut used,
                &mut w,
                |i, j, v, w| a.entry(i, j, v, w),
            )?;
            if source.sha256 != problem.operator.sha256 {
                return Err(invalid("operator source digest mismatch"));
            }
            let a = a.finish(&mut w)?;
            let mut g = Accumulator::new(&columns, true);
            let metric_stream = match (&problem.metric, metric_reader) {
                (Metric::Identity { .. }, None) => {
                    for i in 0..n {
                        w.cancel()?;
                        g.entry(i, i, I::point(Rational::from(1)), &mut w)?;
                    }
                    None
                }
                (Metric::Diagonal { entries, .. }, None) => {
                    for (i, b) in entries.iter().enumerate() {
                        let v = w.read(b)?;
                        g.entry(i, i, v, &mut w)?;
                    }
                    None
                }
                (Metric::Stream { source }, Some(reader)) => {
                    let record = consume(
                        &source.metadata,
                        reader,
                        o,
                        &mut used,
                        &mut w,
                        |i, j, v, w| g.entry(i, j, v, w),
                    )?;
                    if record.sha256 != source.sha256 {
                        return Err(invalid("metric source digest mismatch"));
                    }
                    Some(record)
                }
                _ => return Err(invalid("metric stream mismatch")),
            };
            let g = g.finish(&mut w)?;
            let metric_status = metric_status(&g, p, &problem.metric, &mut w)?;
            let metric_global_lower_bound = g
                .global_lower
                .as_ref()
                .filter(|v| *v > &0)
                .map(ToString::to_string);
            let energy = decompose_pairings(
                &a.pairings,
                p,
                a.direct.clone(),
                a.direct_absolute.clone(),
                &mut w,
            )?;
            let norm = decompose_pairings(
                &g.pairings,
                p,
                g.direct.clone(),
                g.direct_absolute.clone(),
                &mut w,
            )?;
            let mut parts = Vec::new();
            let mut stages: Vec<Stage> = Vec::new();
            for i in 0..p {
                parts.push(measured(
                    &a.pairings[i * p + i],
                    &g.pairings[i * p + i],
                    &metric_status,
                    o,
                    &mut w,
                )?);
                let e = if i + 1 == p {
                    energy.total.clone()
                } else {
                    prefix(&a.pairings, p, i + 1, &mut w)?
                };
                let s = if i + 1 == p {
                    norm.total.clone()
                } else {
                    prefix(&g.pairings, p, i + 1, &mut w)?
                };
                let m = measured(&e, &s, &metric_status, o, &mut w)?;
                let (mut de, mut ds, mut dr) = (None, None, None);
                if let Some(last) = stages.last() {
                    // Use just the new row/column to avoid losing cancellation
                    // by subtracting two large prefix totals.
                    let mut ed = a.pairings[i * p + i].clone();
                    let mut sd = g.pairings[i * p + i].clone();
                    for j in 0..i {
                        let v = w.mul(&two(), &a.pairings[j * p + i])?;
                        ed = w.add(&ed, &v)?;
                        let v = w.mul(&two(), &g.pairings[j * p + i])?;
                        sd = w.add(&sd, &v)?;
                    }
                    let previous_e = w.read(&last.measurement.energy)?;
                    let previous_s = w.read(&last.measurement.squared_norm)?;
                    de = Some(bounds(&intersection(&ed, &w.sub(&e, &previous_e)?)?));
                    ds = Some(bounds(&intersection(&sd, &w.sub(&s, &previous_s)?)?));
                    if let (Some(before), Some(after)) =
                        (&last.measurement.normalized, &m.normalized)
                    {
                        let before = w.read(&before.rayleigh_quotient)?;
                        let after = w.read(&after.rayleigh_quotient)?;
                        dr = Some(bounds(&w.sub(&after, &before)?));
                    }
                }
                stages.push(Stage {
                    corrections_applied: i,
                    measurement: m,
                    energy_change_from_previous: de,
                    squared_norm_change_from_previous: ds,
                    rayleigh_change_from_previous: dr,
                });
            }
            let mut energy_pairing_cancellation = Vec::new();
            let mut metric_pairing_cancellation = Vec::new();
            for i in 0..p * p {
                energy_pairing_cancellation.push(cancellation(
                    &a.absolute[i],
                    &a.pairings[i],
                    &mut w,
                )?);
                metric_pairing_cancellation.push(cancellation(
                    &g.absolute[i],
                    &g.pairings[i],
                    &mut w,
                )?);
            }
            let (mut common, mut closure) = (None, None);
            if let Some(final_value) = &stages[p - 1].measurement.normalized {
                let mut contributions = Vec::new();
                let mut total = zero();
                for term in &energy.report.contributions {
                    let v = w.read(&term.value)?;
                    let v = w.div(&v, &norm.total)?;
                    total = w.add(&total, &v)?;
                    contributions.push(FormContribution {
                        left: term.left,
                        right: term.right,
                        value: bounds(&v),
                    });
                }
                let quotient = w.read(&final_value.rayleigh_quotient)?;
                let difference = w.sub(&total, &quotient)?;
                if !difference.contains_zero() {
                    return Err(SolverError::NumericalBreakdown(
                        "stream normalized closure".into(),
                    ));
                }
                common = Some(contributions);
                closure = Some(bounds(&difference));
            }
            w.cancel()?;
            Ok(Report {
                semantics: STREAM_SEMANTICS.into(),
                scope: STREAM_SCOPE.into(),
                problem_sha256: hash(&(STREAM_SEMANTICS, problem))?,
                request_sha256: hash(&(STREAM_SEMANTICS, problem, o))?,
                operator: source,
                metric_stream,
                metric_status,
                metric_global_lower_bound,
                part_labels: vectors.iter().map(|v| v.label.clone()).collect(),
                parts,
                stages,
                energy_pairings: a.pairings.iter().map(bounds).collect(),
                metric_pairings: g.pairings.iter().map(bounds).collect(),
                energy: energy.report,
                norm: norm.report,
                energy_pairing_cancellation,
                metric_pairing_cancellation,
                common_norm_energy_contributions: common,
                common_norm_closure_residual: closure,
                interval_operations: w.operations,
                vector_scalar_bytes,
                streamed_entries: used.entries,
                streamed_scalar_bytes: used.bytes,
            })
        }
        /// Fresh readers must start at entry zero. Source contents, requested
        /// options, all enclosures, metric gates and diagnostic fields replay.
        /// Reader chunking is transport metadata and may differ on replay.
        pub fn verify(
            problem: &Problem,
            o: &Options,
            operator: &mut dyn FormReader,
            metric: Option<&mut dyn FormReader>,
            report: &Report,
            token: &CancellationToken,
        ) -> Result<(), SolverError> {
            if analyze(problem, o, operator, metric, token)? != *report {
                return Err(invalid("streamed trial report replay mismatch"));
            }
            Ok(())
        }
    }

    /// Replay all identities, metric checks, cross terms, bounds and statuses.
    /// Source labels and caller-supplied interval premises are not authenticated.
    pub fn verify_trial_energy(
        problem: &TrialEnergyProblem,
        options: &TrialEnergyOptions,
        report: &TrialEnergyReport,
        token: &CancellationToken,
    ) -> Result<(), SolverError> {
        if analyze_trial_energy(problem, options, token)? != *report {
            return Err(invalid(
                "report source, request, bounds or diagnostics mismatch",
            ));
        }
        Ok(())
    }
}

/// Bounded reference checks for real finite problems and explicitly supplied
/// analytic envelopes. Labels bind requests; they do not authenticate sources.
#[cfg(feature = "hp-reference")]
#[doc(hidden)]
pub mod convergence {
    /// Largest serialized problem `analyze` accepts. Callers bound their own
    /// workspace below this; the cap only rejects inputs no caller declared.
    pub const MAXIMUM_INPUT_BYTES: usize = 16 << 30;
    use super::trial_energy::{bounds, hash, magnitude, read_form, zero, Work as ExactWork};
    pub use super::trial_energy::{ExactBounds, TrialForm};
    use super::SolverError;
    use rug::{float::Round, Float, Rational};
    use serde::{Deserialize, Serialize};
    use xc_core::CancellationToken;
    use xc_numerics::interval::RationalInterval as I;
    use xc_numerics::mpfr_interval::MpfrInterval as M;

    const SEMANTICS: &str = "finite-convergence-witnesses-v2";
    fn default_precision() -> u32 {
        512
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Options {
        pub maximum_dimension: usize,
        pub maximum_parts: usize,
        pub maximum_input_bytes: usize,
        pub maximum_interval_operations: u64,
        pub maximum_rational_bits: u32,
        #[serde(default = "default_precision")]
        pub working_precision_bits: u32,
    }
    impl Default for Options {
        fn default() -> Self {
            let o = super::trial_energy::TrialEnergyOptions::default();
            Self {
                maximum_dimension: o.maximum_dimension,
                maximum_parts: o.maximum_parts,
                maximum_input_bytes: o.maximum_input_bytes,
                maximum_interval_operations: o.maximum_interval_operations,
                maximum_rational_bits: o.maximum_rational_bits,
                working_precision_bits: default_precision(),
            }
        }
    }
    // This bounded arithmetic is local to convergence checks. Exact trial-energy
    // and LP contracts retain their independent exact-rational implementation.
    struct Work<'a> {
        exact: ExactWork<'a>,
        precision: u32,
    }
    impl<'a> std::ops::Deref for Work<'a> {
        type Target = ExactWork<'a>;
        fn deref(&self) -> &Self::Target {
            &self.exact
        }
    }
    impl std::ops::DerefMut for Work<'_> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.exact
        }
    }
    impl Work<'_> {
        fn mpfr(&self, x: &I) -> Result<M, SolverError> {
            M::new(
                Float::with_val_round(self.precision, x.lower(), Round::Down).0,
                Float::with_val_round(self.precision, x.upper(), Round::Up).0,
            )
            .map_err(|_| invalid("directed conversion range"))
        }
        fn rational_interval(&self, x: M) -> Result<I, SolverError> {
            x.validate()
                .map_err(|_| invalid("directed arithmetic range"))?;
            for f in [x.lower(), x.upper()] {
                // Check before to_rational allocates an exponent-sized integer.
                let e = i64::from(f.get_exp().unwrap_or(0));
                let limit = i64::from(self.options.maximum_rational_bits);
                if e.max(i64::from(f.prec())) > limit || i64::from(f.prec()) - e > limit {
                    return Err(SolverError::IterationBudgetExhausted(
                        "convergence: directed endpoint bit limit".into(),
                    ));
                }
            }
            self.checked(
                I::new(
                    x.lower()
                        .to_rational()
                        .ok_or_else(|| invalid("directed lower range"))?,
                    x.upper()
                        .to_rational()
                        .ok_or_else(|| invalid("directed upper range"))?,
                )
                .unwrap(),
            )
        }
        fn bounded(&self, x: I) -> Result<I, SolverError> {
            let limit = self
                .precision
                .saturating_mul(2)
                .max(self.options.maximum_rational_bits / 4)
                .min(self.options.maximum_rational_bits);
            if [x.lower(), x.upper()].iter().all(|v| {
                v.numer().significant_bits() <= limit && v.denom().significant_bits() <= limit
            }) {
                self.checked(x)
            } else {
                self.rational_interval(self.mpfr(&x)?)
            }
        }
        fn add(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.bounded(x.add(y))
        }
        fn sub(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.bounded(x.sub(y))
        }
        fn mul(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.bounded(x.mul(y))
        }
        fn square(&mut self, x: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.bounded(x.square())
        }
        fn div(&mut self, x: &I, y: &I) -> Result<I, SolverError> {
            self.tick()?;
            self.bounded(x.div(y).map_err(|_| invalid("denominator contains zero"))?)
        }
    }
    fn invalid(s: &str) -> SolverError {
        SolverError::InvalidConfiguration(format!("convergence: {s}"))
    }
    fn premise_metric_check(g: &[I], n: usize, w: &mut Work<'_>) -> Result<String, SolverError> {
        let count = inertia(g, n, w).map_err(|e| match e {
            SolverError::PremiseNotVerified(reason) => SolverError::PremiseNotVerified(format!(
                "positive definiteness unresolved: {reason}"
            )),
            e => e,
        })?;
        if count != 0 {
            return Err(SolverError::PremiseNotVerified(
                "finite form is not positive definite".into(),
            ));
        }
        Ok("positive_directed_interval_ldl".into())
    }
    fn point(v: i32) -> I {
        I::point(Rational::from(v))
    }
    fn upper(v: &I) -> Rational {
        magnitude(v).upper().clone()
    }
    fn widen(x: &I, e: &Rational) -> I {
        x.add(&I::new(-e.clone(), e.clone()).unwrap())
    }
    fn nonnegative(s: &str, w: &Work<'_>) -> Result<Rational, SolverError> {
        let v = w.scalar(s)?;
        if v < 0 {
            return Err(invalid("negative error bound"));
        }
        Ok(v)
    }
    fn sqrt_upper(x: &Rational, w: &mut Work<'_>) -> Result<Rational, SolverError> {
        w.tick()?;
        if x < &0 {
            return Err(invalid("negative squared norm"));
        }
        let mut f = Float::with_val_round(256, x, Round::Up).0;
        f.sqrt_round(Round::Up);
        let r = f
            .to_rational()
            .ok_or_else(|| invalid("square-root range"))?;
        Ok(w.checked(I::point(r))?.upper().clone())
    }
    fn vector(v: &[ExactBounds], w: &Work<'_>) -> Result<Vec<I>, SolverError> {
        if v.is_empty() || v.len() > w.options.maximum_dimension {
            return Err(invalid("vector dimension limit"));
        }
        v.iter().map(|x| w.read(x)).collect()
    }
    fn dot(x: &[I], y: &[I], w: &mut Work<'_>) -> Result<I, SolverError> {
        if x.len() != y.len() {
            return Err(invalid("dot shape"));
        }
        let mut s = zero();
        for (x, y) in x.iter().zip(y) {
            let t = w.mul(x, y)?;
            s = w.add(&s, &t)?;
        }
        Ok(s)
    }
    fn norm2(x: &[I], w: &mut Work<'_>) -> Result<I, SolverError> {
        let mut s = zero();
        for x in x {
            let t = w.square(x)?;
            s = w.add(&s, &t)?;
        }
        Ok(s)
    }
    fn matvec(a: &[I], x: &[I], w: &mut Work<'_>) -> Result<Vec<I>, SolverError> {
        if a.len() != x.len() * x.len() {
            return Err(invalid("matrix/vector shape"));
        }
        a.chunks_exact(x.len()).map(|row| dot(row, x, w)).collect()
    }
    fn form(a: &TrialForm, w: &Work<'_>) -> Result<Vec<I>, SolverError> {
        if a.dimension == 0
            || a.dimension > w.options.maximum_dimension
            || a.entries.len() != a.dimension * a.dimension
            || [&a.source_id, &a.basis_id, &a.normalization_id]
                .iter()
                .any(|s| s.trim().is_empty() || s.len() > 4096)
        {
            return Err(invalid("matrix shape, coordinates or identity"));
        }
        read_form(a, w)
    }
    fn shifted(a: &[I], n: usize, shift: &I, w: &mut Work<'_>) -> Result<Vec<I>, SolverError> {
        let mut b = a.to_vec();
        for i in 0..n {
            b[i * n + i] = w.sub(&b[i * n + i], shift)?;
        }
        Ok(b)
    }
    // No-pivot interval LDL is a sufficient certificate, not a complete
    // inertia algorithm. Any pivot containing zero refuses the certificate.
    fn inertia(a: &[I], n: usize, w: &mut Work<'_>) -> Result<usize, SolverError> {
        let a: Vec<_> = a.iter().map(|x| w.mpfr(x)).collect::<Result<_, _>>()?;
        let zero = M::from_i64(0, w.precision);
        let mut l = vec![zero.clone(); n * n];
        let mut d = vec![zero; n];
        let mut negative = 0;
        for i in 0..n {
            let mut pivot = a[i * n + i].clone();
            for k in 0..i {
                w.tick()?;
                let q = l[i * n + k].square();
                w.tick()?;
                let t = q.mul(&d[k]);
                w.tick()?;
                pivot = pivot.sub(&t);
            }
            pivot
                .validate()
                .map_err(|_| invalid("inertia arithmetic range"))?;
            if pivot.contains_zero() {
                return Err(SolverError::PremiseNotVerified(
                    "shifted inertia pivot unresolved".into(),
                ));
            }
            negative += usize::from(pivot.upper() < &0);
            d[i] = pivot;
            for j in i + 1..n {
                let mut v = a[j * n + i].clone();
                for k in 0..i {
                    w.tick()?;
                    let q = l[j * n + k].mul(&l[i * n + k]);
                    w.tick()?;
                    let t = q.mul(&d[k]);
                    w.tick()?;
                    v = v.sub(&t);
                }
                w.tick()?;
                l[j * n + i] = v
                    .div(&d[i])
                    .map_err(|_| invalid("inertia division range"))?;
            }
        }
        Ok(negative)
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct RootProblem {
        pub source_id: String,
        pub branch_id: String,
        /// A caller's label, never a certified global ordinal.
        pub requested_index: usize,
        pub weights: Vec<ExactBounds>,
        pub poles: Vec<ExactBounds>,
        pub bracket: ExactBounds,
        pub center: String,
        /// Uniform C1 perturbation bounds on the complete bracket. Zero gives
        /// the rational interval family itself. External validity is a premise.
        pub value_error_upper: String,
        pub derivative_error_upper: String,
        pub taylor_order: usize,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RootStatus {
        CertifiedLocal,
        PoleContact,
        UnresolvedDerivative,
        NoSignChange,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct RootReport {
        pub status: RootStatus,
        pub requested_index: usize,
        pub source_id: String,
        pub branch_id: String,
        pub enclosure: Option<ExactBounds>,
        pub absolute_error_upper: Option<String>,
        /// Contributions before intersecting with the isolating bracket.
        /// Evaluation includes uncertainty already present in weights/poles.
        pub evaluation_displacement_upper: Option<String>,
        pub source_displacement_upper: Option<String>,
        pub derivative: Option<ExactBounds>,
        pub endpoint_values: Option<[ExactBounds; 2]>,
        pub global_ordinal_certified: bool,
    }
    fn secular(
        weights: &[I],
        poles: &[I],
        x: &I,
        derivative: bool,
        w: &mut Work<'_>,
    ) -> Result<I, SolverError> {
        let mut s = zero();
        for (a, p) in weights.iter().zip(poles) {
            let d = w.sub(x, p)?;
            let t = if derivative {
                let q = w.square(&d)?;
                w.div(a, &q)?.neg()
            } else {
                w.div(a, &d)?
            };
            s = w.add(&s, &t)?;
        }
        Ok(s)
    }
    fn root(p: &RootProblem, w: &mut Work<'_>) -> Result<RootReport, SolverError> {
        if p.source_id.trim().is_empty()
            || p.branch_id.trim().is_empty()
            || p.requested_index == 0
            || !(2..=64).contains(&p.taylor_order)
        {
            return Err(invalid("root identity/index/order"));
        }
        let a = vector(&p.weights, w)?;
        let poles = vector(&p.poles, w)?;
        if a.len() != poles.len() {
            return Err(invalid("root shape"));
        }
        let b = w.read(&p.bracket)?;
        let c = w.scalar(&p.center)?;
        if b.lower() >= &c || b.upper() <= &c {
            return Err(invalid("root center must be interior"));
        }
        let e = nonnegative(&p.value_error_upper, w)?;
        let de = nonnegative(&p.derivative_error_upper, w)?;
        let mut out = RootReport {
            status: RootStatus::PoleContact,
            requested_index: p.requested_index,
            source_id: p.source_id.clone(),
            branch_id: p.branch_id.clone(),
            enclosure: None,
            absolute_error_upper: None,
            evaluation_displacement_upper: None,
            source_displacement_upper: None,
            derivative: None,
            endpoint_values: None,
            global_ordinal_certified: false,
        };
        if poles
            .iter()
            .any(|p| p.lower() <= b.upper() && p.upper() >= b.lower())
        {
            return Ok(out);
        }
        let center = I::point(c.clone());
        let radius = Rational::from(&c - b.lower()).max(Rational::from(b.upper() - &c));
        let r = I::point(radius);
        let direct = secular(&a, &poles, &b, true, w)?;
        // Cancellation-preserving Taylor coefficients and geometric derivative tail.
        let mut coefficients = vec![zero(); p.taylor_order + 1];
        let mut tail = zero();
        let mut taylor_usable = true;
        for (weight, pole) in a.iter().zip(&poles) {
            let d = w.sub(&center, pole)?;
            let distance = magnitude(&d).lower().clone();
            if r.upper() >= &distance {
                taylor_usable = false;
                break;
            }
            let mut term = w.div(weight, &d)?;
            for coefficient in &mut coefficients {
                *coefficient = w.add(coefficient, &term)?;
                term = w.div(&term, &d)?.neg();
            }
            let q = w.div(&r, &I::point(distance.clone()))?;
            let mut power = point(1);
            for _ in 0..p.taylor_order {
                power = w.mul(&power, &q)?;
            }
            let mq = w.mul(&point(p.taylor_order as i32), &q)?;
            let numerator = w.sub(&point(p.taylor_order as i32 + 1), &mq)?;
            let t = w.mul(&power, &numerator)?;
            let one_minus = w.sub(&point(1), &q)?;
            let square = w.square(&one_minus)?;
            let t = w.div(&t, &square)?;
            let distance2 = w.square(&I::point(distance))?;
            let scale = w.div(&magnitude(weight), &distance2)?;
            let t = w.mul(&scale, &t)?;
            tail = w.add(&tail, &t)?;
        }
        let mut derivative = direct;
        if taylor_usable {
            let mut variation = tail;
            let mut power = r.clone();
            for (m, coefficient) in coefficients.iter().enumerate().skip(2) {
                let t = w.mul(&magnitude(coefficient), &point(m as i32))?;
                let t = w.mul(&t, &power)?;
                variation = w.add(&variation, &t)?;
                power = w.mul(&power, &r)?;
            }
            let taylor = widen(&coefficients[1], variation.upper());
            derivative = I::new(
                derivative.lower().clone().max(taylor.lower().clone()),
                derivative.upper().clone().min(taylor.upper().clone()),
            )
            .map_err(|_| invalid("inconsistent derivative bounds"))?;
        }
        derivative = w.checked(widen(&derivative, &de))?;
        out.derivative = Some(bounds(&derivative));
        let left = secular(&a, &poles, &I::point(b.lower().clone()), false, w)?;
        let left = w.checked(widen(&left, &e))?;
        let right = secular(&a, &poles, &I::point(b.upper().clone()), false, w)?;
        let right = w.checked(widen(&right, &e))?;
        out.endpoint_values = Some([bounds(&left), bounds(&right)]);
        if derivative.contains_zero() {
            out.status = RootStatus::UnresolvedDerivative;
            return Ok(out);
        }
        if !((left.is_strictly_negative() && right.is_strictly_positive())
            || (left.is_strictly_positive() && right.is_strictly_negative()))
        {
            out.status = RootStatus::NoSignChange;
            return Ok(out);
        }
        let value = secular(&a, &poles, &center, false, w)?;
        let floor = I::point(magnitude(&derivative).lower().clone());
        let evaluation = w.div(&I::point(upper(&value)), &floor)?;
        let source = w.div(&I::point(e.clone()), &floor)?;
        out.evaluation_displacement_upper = Some(evaluation.upper().to_string());
        out.source_displacement_upper = Some(source.upper().to_string());
        let value = w.checked(widen(&value, &e))?;
        let err = w.div(
            &I::point(upper(&value)),
            &I::point(magnitude(&derivative).lower().clone()),
        )?;
        let region = widen(&center, err.upper());
        let enclosure = I::new(
            b.lower().clone().max(region.lower().clone()),
            b.upper().clone().min(region.upper().clone()),
        )
        .map_err(|_| invalid("root enclosure intersection"))?;
        let bound =
            Rational::from(&c - enclosure.lower()).max(Rational::from(enclosure.upper() - &c));
        out.status = RootStatus::CertifiedLocal;
        out.enclosure = Some(bounds(&enclosure));
        out.absolute_error_upper = Some(bound.to_string());
        Ok(out)
    }

    /// B x*=b with symmetric positive definite B. Both residuals are recomputed;
    /// no claimed solver convergence or supplied residual is trusted.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct DirectionalProblem {
        pub matrix: TrialForm,
        pub rhs: Vec<ExactBounds>,
        pub approximate_solution: Vec<ExactBounds>,
        pub functional: Vec<ExactBounds>,
        pub approximate_dual: Vec<ExactBounds>,
        pub coercivity_lower: String,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct DirectionalReport {
        pub signed_correction: ExactBounds,
        pub residual_squared_norm: ExactBounds,
        pub dual_residual_squared_norm: ExactBounds,
        pub correction_error_upper: String,
        pub exact_functional_enclosure: ExactBounds,
    }
    fn directional(
        p: &DirectionalProblem,
        w: &mut Work<'_>,
    ) -> Result<DirectionalReport, SolverError> {
        let a = form(&p.matrix, w)?;
        let n = p.matrix.dimension;
        let x = vector(&p.approximate_solution, w)?;
        let z = vector(&p.approximate_dual, w)?;
        let b = vector(&p.rhs, w)?;
        let c = vector(&p.functional, w)?;
        if [&x, &z, &b, &c].iter().any(|v| v.len() != n) {
            return Err(invalid("directional shape"));
        }
        let g = nonnegative(&p.coercivity_lower, w)?;
        if g == 0 {
            return Err(invalid("positive coercivity required"));
        }
        let shifted = shifted(&a, n, &I::point(g.clone()), w)?;
        premise_metric_check(&shifted, n, w)?;
        let ax = matvec(&a, &x, w)?;
        let az = matvec(&a, &z, w)?;
        let r: Vec<_> = b
            .iter()
            .zip(&ax)
            .map(|(b, a)| w.sub(b, a))
            .collect::<Result<_, _>>()?;
        let d: Vec<_> = c
            .iter()
            .zip(&az)
            .map(|(c, a)| w.sub(c, a))
            .collect::<Result<_, _>>()?;
        let rn = norm2(&r, w)?;
        let dn = norm2(&d, w)?;
        let product = w.mul(&rn, &dn)?;
        let error = sqrt_upper(product.upper(), w)?;
        let error = w.div(&I::point(error), &I::point(g))?;
        let signed = dot(&z, &r, w)?;
        let nominal = dot(&c, &x, w)?;
        let corrected = w.add(&nominal, &signed)?;
        let enclosure = w.checked(widen(&corrected, error.upper()))?;
        Ok(DirectionalReport {
            signed_correction: bounds(&signed),
            residual_squared_norm: bounds(&rn),
            dual_residual_squared_norm: bounds(&dn),
            correction_error_upper: error.upper().to_string(),
            exact_functional_enclosure: bounds(&enclosure),
        })
    }

    /// Finite omitted block only: verifies A22-shift I > gap I and bounds the
    /// norm of A12 (A22-shift I)^-1 A21. No unseen infinite block is inferred.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TailBlockProblem {
        pub matrix: TrialForm,
        pub retained_dimension: usize,
        pub shift: ExactBounds,
        pub gap_lower: String,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TailBlockReport {
        pub omitted_dimension: usize,
        pub coupling_squared_upper: String,
        pub schur_correction_norm_upper: String,
    }
    fn tail_block(p: &TailBlockProblem, w: &mut Work<'_>) -> Result<TailBlockReport, SolverError> {
        let a = form(&p.matrix, w)?;
        let n = p.matrix.dimension;
        let k = p.retained_dimension;
        if k == 0 || k >= n {
            return Err(invalid("nonempty retained and omitted blocks required"));
        }
        let shift = w.read(&p.shift)?;
        let g = nonnegative(&p.gap_lower, w)?;
        if g == 0 {
            return Err(invalid("positive tail gap required"));
        }
        let m = n - k;
        let mut block = vec![zero(); m * m];
        let mut coupling = zero();
        for i in k..n {
            for j in k..n {
                block[(i - k) * m + j - k] = a[i * n + j].clone();
            }
        }
        let boundary = w.add(&I::point(shift.upper().clone()), &I::point(g.clone()))?;
        let block = shifted(&block, m, &boundary, w)?;
        premise_metric_check(&block, m, w)?;
        for i in 0..k {
            for j in k..n {
                let t = w.square(&a[i * n + j])?;
                coupling = w.add(&coupling, &t)?;
            }
        }
        let error = w.div(&coupling, &I::point(g))?;
        Ok(TailBlockReport {
            omitted_dimension: m,
            coupling_squared_upper: coupling.upper().to_string(),
            schur_correction_norm_upper: error.upper().to_string(),
        })
    }

    /// Caller-supplied analytic hypothesis |term_j|<=scale/(j+offset)^power
    /// for every j>N. The integral-test calculation is verified; the infinite
    /// hypothesis and its applicability to a root displacement are external.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct TailMajorant {
        pub branch_id: String,
        pub proof_id: String,
        pub hypotheses: Vec<String>,
        pub minimum_modes: usize,
        pub scale: String,
        pub offset: String,
        pub power: u32,
    }
    fn tail_bound(
        p: &TailMajorant,
        n: usize,
        branch: &str,
        w: &mut Work<'_>,
    ) -> Result<Rational, SolverError> {
        if p.branch_id != branch
            || p.proof_id.trim().is_empty()
            || p.hypotheses.is_empty()
            || p.hypotheses.iter().any(|s| s.trim().is_empty())
            || n < p.minimum_modes
            || !(2..=64).contains(&p.power)
        {
            return Err(invalid("tail hypothesis, branch or domain"));
        }
        let scale = nonnegative(&p.scale, w)?;
        let offset = w.scalar(&p.offset)?;
        let base = Rational::from(n) + offset;
        if base <= 0 {
            return Err(invalid("positive tail integration endpoint required"));
        }
        let base = I::point(base);
        let mut denominator = point(p.power as i32 - 1);
        for _ in 0..p.power - 1 {
            denominator = w.mul(&denominator, &base)?;
        }
        Ok(w.div(&I::point(scale), &denominator)?.upper().clone())
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct BudgetCandidate {
        pub modes: usize,
        pub precision_bits: u32,
        pub root: RootProblem,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct DeclaredCutoffBound {
        pub branch_id: String,
        pub proof_id: String,
        pub hypotheses: Vec<String>,
        pub absolute_error_upper: String,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct BudgetProblem {
        pub branch_id: String,
        pub target_absolute_error: String,
        pub candidates: Vec<BudgetCandidate>,
        pub truncation: Option<TailMajorant>,
        pub cutoff: Option<DeclaredCutoffBound>,
        pub reference_assisted: bool,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum BudgetAction {
        SufficientUnderDeclaredHypotheses,
        MissingTruncationBound,
        MissingCutoffBound,
        ResolveRoot,
        RefineFiniteRootOrSource,
        IncreaseModes,
        IncreaseCutoff,
        RefineMultipleComponents,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct CandidateReport {
        pub modes: usize,
        pub precision_bits: u32,
        pub root: RootReport,
        pub truncation_error_upper: Option<String>,
        pub cutoff_error_upper: Option<String>,
        pub total_error_upper: Option<String>,
        pub action: BudgetAction,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct BudgetReport {
        pub rows: Vec<CandidateReport>,
        pub selected_candidate: Option<usize>,
        pub reference_assisted: bool,
    }
    fn budget(p: &BudgetProblem, w: &mut Work<'_>) -> Result<BudgetReport, SolverError> {
        if p.candidates.is_empty() || p.candidates.len() > 64 {
            return Err(invalid("candidate count"));
        }
        let target = nonnegative(&p.target_absolute_error, w)?;
        if target == 0 {
            return Err(invalid("positive target tolerance required"));
        }
        let cutoff = if let Some(c) = &p.cutoff {
            if c.branch_id != p.branch_id
                || c.proof_id.trim().is_empty()
                || c.hypotheses.is_empty()
                || c.hypotheses.iter().any(|x| x.trim().is_empty())
            {
                return Err(invalid("cutoff bound provenance"));
            }
            Some(nonnegative(&c.absolute_error_upper, w)?)
        } else {
            None
        };
        let mut rows = Vec::new();
        let mut selected: Option<usize> = None;
        let mut settings = std::collections::BTreeSet::new();
        for c in &p.candidates {
            if c.modes == 0
                || !(64..=1_000_000).contains(&c.precision_bits)
                || c.root.branch_id != p.branch_id
                || !settings.insert((c.modes, c.precision_bits))
            {
                return Err(invalid("candidate configuration or branch"));
            }
            let r = root(&c.root, w)?;
            let tail = p
                .truncation
                .as_ref()
                .map(|t| tail_bound(t, c.modes, &p.branch_id, w))
                .transpose()?;
            let arithmetic = r
                .absolute_error_upper
                .as_ref()
                .map(|s| w.scalar(s))
                .transpose()?;
            let mut total = None;
            let action = match (&arithmetic, &tail, &cutoff) {
                (None, _, _) => BudgetAction::ResolveRoot,
                (_, None, _) => BudgetAction::MissingTruncationBound,
                (_, _, None) => BudgetAction::MissingCutoffBound,
                (Some(a), Some(t), Some(cut)) => {
                    let sum = w.add(&I::point(a.clone()), &I::point(t.clone()))?;
                    let sum = w.add(&sum, &I::point(cut.clone()))?;
                    total = Some(sum.upper().to_string());
                    if sum.upper() <= &target {
                        BudgetAction::SufficientUnderDeclaredHypotheses
                    } else if [a, t, cut].iter().filter(|x| *x >= &&target).count() > 1 {
                        BudgetAction::RefineMultipleComponents
                    } else if a >= &target {
                        BudgetAction::RefineFiniteRootOrSource
                    } else if t >= &target {
                        BudgetAction::IncreaseModes
                    } else if cut >= &target {
                        BudgetAction::IncreaseCutoff
                    } else {
                        BudgetAction::RefineMultipleComponents
                    }
                }
            };
            if action == BudgetAction::SufficientUnderDeclaredHypotheses
                && selected.is_none_or(|i| {
                    (c.modes, c.precision_bits)
                        < (p.candidates[i].modes, p.candidates[i].precision_bits)
                })
            {
                selected = Some(rows.len());
            }
            rows.push(CandidateReport {
                modes: c.modes,
                precision_bits: c.precision_bits,
                root: r,
                truncation_error_upper: tail.map(|v| v.to_string()),
                cutoff_error_upper: cutoff.as_ref().map(ToString::to_string),
                total_error_upper: total,
                action,
            });
        }
        Ok(BudgetReport {
            rows,
            selected_candidate: selected,
            reference_assisted: p.reference_assisted,
        })
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct NormalizationProblem {
        pub source_id: String,
        pub raw_norm_error_upper: String,
        pub reference_norm_upper: String,
        pub source_normalizer: ExactBounds,
        pub reference_normalizer: ExactBounds,
        /// Separate trace/evaluation control. A nonzero floor is insufficient.
        pub normalizer_difference_upper: String,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct NormalizationReport {
        pub source_floor: String,
        pub reference_floor: String,
        pub normalized_norm_error_upper: String,
    }
    fn normalize(
        p: &NormalizationProblem,
        w: &mut Work<'_>,
    ) -> Result<NormalizationReport, SolverError> {
        if p.source_id.trim().is_empty() {
            return Err(invalid("normalization source identity"));
        }
        let a = w.read(&p.source_normalizer)?;
        let b = w.read(&p.reference_normalizer)?;
        let af = magnitude(&a).lower().clone();
        let bf = magnitude(&b).lower().clone();
        if af == 0 || bf == 0 {
            return Err(SolverError::PremiseNotVerified(
                "normalization floor unresolved".into(),
            ));
        }
        let e = nonnegative(&p.raw_norm_error_upper, w)?;
        let v = nonnegative(&p.reference_norm_upper, w)?;
        let diff = nonnegative(&p.normalizer_difference_upper, w)?;
        let actual = w.sub(&a, &b)?;
        if magnitude(&actual).lower() > &diff {
            return Err(SolverError::PremiseNotVerified(
                "inconsistent normalization difference".into(),
            ));
        }
        let diff = diff.min(upper(&actual));
        let t1 = w.div(&I::point(e), &I::point(af.clone()))?;
        let t2 = w.mul(&I::point(v), &I::point(diff))?;
        let denominator = w.mul(&I::point(af.clone()), &I::point(bf.clone()))?;
        let t2 = w.div(&t2, &denominator)?;
        let result = w.add(&t1, &t2)?;
        Ok(NormalizationReport {
            source_floor: af.to_string(),
            reference_floor: bf.to_string(),
            normalized_norm_error_upper: result.upper().to_string(),
        })
    }
    /// Supremum bounds apply over the entire cell, not only sampled nodes.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ProfileCell {
        pub left: String,
        pub right: String,
        pub left_residual: ExactBounds,
        pub right_residual: ExactBounds,
        pub weight: ExactBounds,
        pub second_derivative_upper: String,
        pub additional_sup_error_upper: String,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ProfileProblem {
        pub source_id: String,
        pub validation_grid_id: String,
        pub training_grid_id: Option<String>,
        pub cells: Vec<ProfileCell>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ProfileReport {
        pub weighted_l1: ExactBounds,
        pub interpolation_error_upper: String,
        pub cell_integrals: Vec<ExactBounds>,
        pub independent_validation_grid: bool,
    }
    fn profile(p: &ProfileProblem, w: &mut Work<'_>) -> Result<ProfileReport, SolverError> {
        if p.source_id.trim().is_empty()
            || p.validation_grid_id.trim().is_empty()
            || p.cells.is_empty()
            || p.cells.len() > 16384
        {
            return Err(invalid("profile identity or cell count"));
        }
        let mut previous: Option<Rational> = None;
        let mut total = zero();
        let mut error_sum = zero();
        let mut cells = Vec::new();
        for cell in &p.cells {
            let l = w.scalar(&cell.left)?;
            let r = w.scalar(&cell.right)?;
            if r <= l || previous.as_ref().is_some_and(|x| x != &l) {
                return Err(invalid(
                    "profile cells must form an ordered contiguous partition",
                ));
            }
            previous = Some(r.clone());
            let h = I::point(r - l);
            let a = w.read(&cell.left_residual)?;
            let b = w.read(&cell.right_residual)?;
            let weight = w.read(&cell.weight)?;
            if weight.lower() < &0 {
                return Err(invalid("negative weight"));
            }
            let midpoint = |x: &I| -> Rational { Rational::from(x.lower() + x.upper()) / 2 };
            let aa = midpoint(&a);
            let bb = midpoint(&b);
            let node_error = Rational::from(a.upper() - &aa).max(Rational::from(b.upper() - &bb));
            let exact = if aa == 0 && bb == 0 {
                zero()
            } else if (aa >= 0 && bb >= 0) || (aa <= 0 && bb <= 0) {
                let s = w.add(&I::point(aa.clone().abs()), &I::point(bb.clone().abs()))?;
                let t = w.mul(&h, &s)?;
                w.div(&t, &point(2))?
            } else {
                let x = w.square(&I::point(aa.clone()))?;
                let y = w.square(&I::point(bb.clone()))?;
                let s = w.add(&x, &y)?;
                let s = w.mul(&h, &s)?;
                let d = w.add(&I::point(aa.abs()), &I::point(bb.abs()))?;
                let d = w.mul(&point(2), &d)?;
                w.div(&s, &d)?
            };
            let m = nonnegative(&cell.second_derivative_upper, w)?;
            let extra = nonnegative(&cell.additional_sup_error_upper, w)?;
            let h2 = w.square(&h)?;
            let t = w.mul(&I::point(m), &h2)?;
            let t = w.div(&t, &point(8))?;
            let t = w.add(&t, &I::point(node_error))?;
            let t = w.add(&t, &I::point(extra))?;
            let t = w.mul(&t, &h)?;
            let err = w.mul(&t, &weight)?;
            let nominal = w.mul(&exact, &weight)?;
            let integral = w.checked(
                I::new(
                    Rational::from(nominal.lower() - err.upper()).max(Rational::from(0)),
                    Rational::from(nominal.upper() + err.upper()),
                )
                .unwrap(),
            )?;
            total = w.add(&total, &integral)?;
            error_sum = w.add(&error_sum, &err)?;
            cells.push(bounds(&integral));
        }
        Ok(ProfileReport {
            weighted_l1: bounds(&total),
            interpolation_error_upper: error_sum.upper().to_string(),
            cell_integrals: cells,
            independent_validation_grid: p
                .training_grid_id
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty() && s != &p.validation_grid_id),
        })
    }

    /// Columns need not be orthonormal. A verified Gram lower bound controls
    /// their conditioning. The selected window is in Euclidean coordinates.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ClusterProblem {
        pub matrix: TrialForm,
        pub spectral_window: ExactBounds,
        pub columns: Vec<Vec<ExactBounds>>,
        pub shifts: Vec<String>,
        pub gram_lower: String,
        /// Optional preceding trial subspace in the identical coordinates.
        /// Tracking is subspace overlap, not an assertion of eigenstate identity.
        pub previous_columns: Option<Vec<Vec<ExactBounds>>>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ClusterReport {
        pub eigenvalues_below: usize,
        pub eigenvalues_in_window: usize,
        pub contains_ground: bool,
        pub simple_ground: bool,
        pub projector_frobenius_squared_upper: String,
        pub residual_squared_norms: Vec<ExactBounds>,
        pub previous_trial_projector_frobenius_squared_upper: Option<String>,
    }
    fn columns(
        raw: &[Vec<ExactBounds>],
        n: usize,
        w: &Work<'_>,
    ) -> Result<Vec<Vec<I>>, SolverError> {
        if raw.is_empty() || raw.len() > n {
            return Err(invalid("cluster rank"));
        }
        raw.iter()
            .map(|v| {
                let x = vector(v, w)?;
                if x.len() != n {
                    return Err(invalid("cluster vector dimension"));
                }
                Ok(x)
            })
            .collect()
    }
    fn gram(q: &[Vec<I>], w: &mut Work<'_>) -> Result<Vec<I>, SolverError> {
        let k = q.len();
        let mut g = vec![zero(); k * k];
        for i in 0..k {
            for j in 0..=i {
                let v = if i == j {
                    norm2(&q[i], w)?
                } else {
                    dot(&q[i], &q[j], w)?
                };
                g[i * k + j] = v.clone();
                g[j * k + i] = v;
            }
        }
        Ok(g)
    }
    fn row_sum_bound(g: &[I], k: usize, w: &mut Work<'_>) -> Result<Rational, SolverError> {
        let mut maximum = Rational::from(0);
        for row in g.chunks_exact(k) {
            let mut sum = zero();
            for x in row {
                sum = w.add(&sum, &magnitude(x))?;
            }
            maximum = maximum.max(sum.upper().clone());
        }
        Ok(maximum)
    }
    fn cluster(p: &ClusterProblem, w: &mut Work<'_>) -> Result<ClusterReport, SolverError> {
        let a = form(&p.matrix, w)?;
        let n = p.matrix.dimension;
        let q = columns(&p.columns, n, w)?;
        let k = q.len();
        if p.shifts.len() != k {
            return Err(invalid("cluster shifts shape"));
        }
        let window = w.read(&p.spectral_window)?;
        if window.lower() >= window.upper() {
            return Err(invalid("empty spectral window"));
        }
        let low = shifted(&a, n, &I::point(window.lower().clone()), w)?;
        let high = shifted(&a, n, &I::point(window.upper().clone()), w)?;
        let below = inertia(&low, n, w)?;
        let above = inertia(&high, n, w)?;
        if above < below || above - below != k {
            return Err(SolverError::PremiseNotVerified(
                "cluster count does not match trial rank".into(),
            ));
        }
        let g = gram(&q, w)?;
        let beta = nonnegative(&p.gram_lower, w)?;
        if beta == 0 {
            return Err(invalid("positive Gram lower bound required"));
        }
        let test = shifted(&g, k, &I::point(beta.clone()), w)?;
        premise_metric_check(&test, k, w)?;
        let mut error = zero();
        let mut residuals = Vec::new();
        for (q, s) in q.iter().zip(&p.shifts) {
            let s = w.scalar(s)?;
            let gap = Rational::from(&s - window.lower()).min(Rational::from(window.upper() - &s));
            if gap <= 0 {
                return Err(invalid("cluster shift outside window"));
            }
            let aq = matvec(&a, q, w)?;
            let mut residual = Vec::new();
            for (a, x) in aq.iter().zip(q) {
                let sx = w.mul(&I::point(s.clone()), x)?;
                residual.push(w.sub(a, &sx)?);
            }
            let nr = norm2(&residual, w)?;
            let gap2 = w.square(&I::point(gap))?;
            let term = w.div(&nr, &gap2)?;
            error = w.add(&error, &term)?;
            residuals.push(bounds(&nr));
        }
        let error = w.div(&error, &I::point(beta))?;
        let distance = error.upper().clone().min(Rational::from(k));
        let previous = if let Some(raw) = &p.previous_columns {
            let v = columns(raw, n, w)?;
            if v.len() != k {
                return Err(invalid("tracking ranks differ"));
            }
            let vg = gram(&v, w)?;
            premise_metric_check(&vg, k, w)?;
            let mut overlap = zero();
            for x in &q {
                for y in &v {
                    let xy = dot(x, y, w)?;
                    let sq = w.square(&xy)?;
                    overlap = w.add(&overlap, &sq)?;
                }
            }
            let g_bound = row_sum_bound(&g, k, w)?;
            let vg_bound = row_sum_bound(&vg, k, w)?;
            let den = w.mul(&I::point(g_bound), &I::point(vg_bound))?;
            let lower = w.div(&overlap, &den)?;
            Some(((Rational::from(k) - lower.lower()).max(Rational::from(0)) * 2i32).to_string())
        } else {
            None
        };
        Ok(ClusterReport {
            eigenvalues_below: below,
            eigenvalues_in_window: k,
            contains_ground: below == 0,
            simple_ground: below == 0 && k == 1,
            projector_frobenius_squared_upper: (distance * 2i32).to_string(),
            residual_squared_norms: residuals,
            previous_trial_projector_frobenius_squared_upper: previous,
        })
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(
        tag = "kind",
        content = "input",
        rename_all = "snake_case",
        deny_unknown_fields
    )]
    pub enum Problem {
        Root(RootProblem),
        Directional(DirectionalProblem),
        TailBlock(TailBlockProblem),
        Budget(BudgetProblem),
        Normalization(NormalizationProblem),
        Profile(ProfileProblem),
        Cluster(ClusterProblem),
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(
        tag = "kind",
        content = "output",
        rename_all = "snake_case",
        deny_unknown_fields
    )]
    pub enum Output {
        Root(RootReport),
        Directional(DirectionalReport),
        TailBlock(TailBlockReport),
        Budget(BudgetReport),
        Normalization(NormalizationReport),
        Profile(ProfileReport),
        Cluster(ClusterReport),
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Report {
        pub semantics: String,
        pub request_sha256: String,
        pub scope: String,
        pub output: Output,
        pub interval_operations: u64,
    }
    pub fn analyze(
        problem: &Problem,
        options: &Options,
        token: &CancellationToken,
    ) -> Result<Report, SolverError> {
        if options.maximum_dimension == 0
            || options.maximum_dimension > 512
            || options.maximum_input_bytes == 0
            || options.maximum_input_bytes > MAXIMUM_INPUT_BYTES
            || options.maximum_interval_operations > 1_000_000_000
            || !(64..=65_536).contains(&options.maximum_rational_bits)
            || !(64..=32_768).contains(&options.working_precision_bits)
            || options.working_precision_bits > options.maximum_rational_bits
        {
            return Err(invalid("unsupported work limits"));
        }
        let limits = super::trial_energy::TrialEnergyOptions {
            maximum_dimension: options.maximum_dimension,
            maximum_parts: options.maximum_parts,
            maximum_input_bytes: options.maximum_input_bytes,
            maximum_interval_operations: options.maximum_interval_operations,
            maximum_rational_bits: options.maximum_rational_bits,
        };
        let mut w = Work {
            exact: ExactWork {
                options: &limits,
                token,
                operations: 0,
            },
            precision: options.working_precision_bits,
        };
        w.cancel()?;
        // Serialization is also the identity representation; input size bounds
        // include strings and analytic hypotheses, not merely scalar payloads.
        let bytes = serde_json::to_vec(problem).map_err(|_| invalid("request serialization"))?;
        if bytes.len() > options.maximum_input_bytes {
            return Err(invalid("input byte limit"));
        }
        let output = match problem {
            Problem::Root(p) => Output::Root(root(p, &mut w)?),
            Problem::Directional(p) => Output::Directional(directional(p, &mut w)?),
            Problem::TailBlock(p) => Output::TailBlock(tail_block(p, &mut w)?),
            Problem::Budget(p) => Output::Budget(budget(p, &mut w)?),
            Problem::Normalization(p) => Output::Normalization(normalize(p, &mut w)?),
            Problem::Profile(p) => Output::Profile(profile(p, &mut w)?),
            Problem::Cluster(p) => Output::Cluster(cluster(p, &mut w)?),
        };
        w.cancel()?;
        Ok(Report{semantics:SEMANTICS.into(),request_sha256:hash(&(SEMANTICS,problem,options))?,
            scope:"finite_interval_arithmetic_conditional_on_declared_input_enclosures_and_explicit_analytic_hypotheses_no_source_authentication".into(),output,interval_operations:w.operations})
    }
    pub fn verify(
        problem: &Problem,
        options: &Options,
        report: &Report,
        token: &CancellationToken,
    ) -> Result<(), SolverError> {
        if analyze(problem, options, token)? != *report {
            return Err(invalid("report replay mismatch"));
        }
        Ok(())
    }
}
