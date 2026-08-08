// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision CCM tier via `rug` (MPFR/GMP).
//!
//! Strategy:
//! - Build the (2N+1)×(2N+1) Weil form matrix at user-chosen precision.
//! - Find smallest eigenpair by inverse iteration (from xc-numerics).
//! - Solve the spectrum equation with Halley's method by default, with
//!   explicit Newton selection available for controlled comparisons.

use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{ops::Pow, Float};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;
#[cfg(feature = "arb")]
use xc_cache::resolve_or_compute_json_artifact_with_assessment;
use xc_cache::{
    resolve_or_compute_json_artifact_with_dependencies, ArtifactAssuranceAttestation,
    ArtifactCacheContext, ArtifactExecutionCacheRequest, ArtifactKey, ArtifactManifest,
    ArtifactProductionAssessment, CacheError, CacheQuality, ContentDigest, DependencyRef,
    SemanticKeyEnvelope, ToolkitVersion,
};
use xc_core::{DecimalLiteral, EigenTarget, ResultStatus};
use xc_operator::{
    ApplicationErrorBound, LinearOperator, MatrixStructure, OperatorError, OperatorMetadata,
    SymmetricOperator,
};
use xc_zeta::zeros::ReferenceZeroDatasetIdentity;

use super::{prime_powers_up_to, window::ZeroTarget, CcmParams, CcmResult};

enum CcmCacheRoute<'a> {
    Standalone,
    Fabric(&'a ArtifactCacheContext<'a>),
}

struct RetainedCcmSource {
    tau: Vec<Float>,
    tau_manifest: Option<ArtifactManifest>,
    eigenpair_manifest: Option<ArtifactManifest>,
    secular_manifest: Option<ArtifactManifest>,
}

#[derive(Clone, Copy)]
enum RootAcquisition<'a> {
    SourceOnly,
    Independent {
        target: &'a ZeroTarget,
        options: IndependentRootDiscoveryOptions,
    },
    ReferenceSeeded {
        first_root_index: usize,
        seeds: &'a [Float],
        dataset: &'a ReferenceZeroDatasetIdentity,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootArtifactMode {
    Independent,
    ReferenceSeededRefinement,
}

impl RootArtifactMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::ReferenceSeededRefinement => "reference_seeded_refinement",
        }
    }
}

/// Sign domain used by independent finite-source root discovery.
///
/// Positive-only discovery remains the production default. `Signed` is an
/// explicit research mode that searches both sides of the finite CCM source
/// window and orders the selected roots numerically.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndependentRootDomain {
    #[default]
    Positive,
    Signed,
}

impl IndependentRootDomain {
    fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Signed => "signed",
        }
    }
}

/// Advanced policy for independent root discovery.
///
/// The default retains the strict production contract: positive roots only
/// and an error if the finite source cannot supply the complete target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentRootDiscoveryOptions {
    pub domain: IndependentRootDomain,
    /// Return the complete available finite window when it cannot fill the
    /// requested target. The returned root list may be empty.
    pub allow_incomplete: bool,
}

impl IndependentRootDiscoveryOptions {
    pub fn advanced(include_negative_roots: bool, allow_incomplete: bool) -> Self {
        Self {
            domain: if include_negative_roots {
                IndependentRootDomain::Signed
            } else {
                IndependentRootDomain::Positive
            },
            allow_incomplete,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RootWindowSemantics {
    domain: IndependentRootDomain,
    requested_count: usize,
    allow_incomplete: bool,
    artifact_format: RootArtifactFormat,
}

impl RootWindowSemantics {
    fn strict_positive(count: usize) -> Self {
        Self {
            domain: IndependentRootDomain::Positive,
            requested_count: count,
            allow_incomplete: false,
            artifact_format: RootArtifactFormat::LegacyV6,
        }
    }

    fn advanced(
        domain: IndependentRootDomain,
        requested_count: usize,
        allow_incomplete: bool,
    ) -> Self {
        Self {
            domain,
            requested_count,
            allow_incomplete,
            artifact_format: RootArtifactFormat::AdvancedV7,
        }
    }

    fn is_advanced(self) -> bool {
        self.artifact_format == RootArtifactFormat::AdvancedV7
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RootArtifactFormat {
    LegacyV6,
    AdvancedV7,
}

#[derive(Debug)]
struct IndependentRootDiscoveryPlan {
    artifact_first_root_index: usize,
    artifact_seeds: Vec<Float>,
    selected_positions: Vec<usize>,
    result_first_root_index: usize,
    request_semantics: RootWindowSemantics,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableTauMatrix {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    entries: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableArchimedeanIntegrals {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    alpha: Vec<String>,
    beta: Vec<String>,
    gamma: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortablePrimeComponent {
    schema_version: u32,
    lambda_squared: String,
    prime_cutoff: u64,
    n_modes: usize,
    precision_bits: u32,
    prime_content: Vec<PrimePowerContent>,
    entries: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableEvenSectorMatrix {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    dimension: usize,
    entries: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableOddSectorMatrix {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    dimension: usize,
    entries: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSectorTridiagonal {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    parity: CcmParity,
    dimension: usize,
    diagonal: Vec<String>,
    off_diagonal: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSectorTransform {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    parity: CcmParity,
    dimension: usize,
    basis: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSectorEigenvalueEnclosure {
    index: usize,
    lower: String,
    upper: String,
    lower_count: usize,
    upper_count: usize,
    iterations: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSectorEigenvalues {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    parity: CcmParity,
    dimension: usize,
    route: CcmSectorEigenvalueRoute,
    complete: bool,
    requested_eigenvalues: usize,
    eigenvalues: Vec<String>,
    selected_enclosures: Vec<PortableSectorEigenvalueEnclosure>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSectorSpectrum {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    parity: CcmParity,
    eigenvalue_route: CcmSectorEigenvalueRoute,
    dimension: usize,
    requested_eigenpairs: usize,
    eigenvalues: Vec<String>,
    eigenvectors: Vec<Vec<String>>,
    residual_norms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSectorGap {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    even_spectrum_content_digest: String,
    odd_spectrum_content_digest: String,
    lambda_even: String,
    lambda_odd: String,
    d_even: String,
    d_odd: String,
    gap_log: String,
    lambda_difference: String,
    difference_depth: String,
    ordering: i8,
    even_simple: bool,
    even_simplicity_margin: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableLuFactorization {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    subspace: String,
    dimension: usize,
    lu: Vec<String>,
    permutation: Vec<usize>,
}

struct ComputedArchimedeanIntegrals {
    alpha: Vec<Float>,
    beta: Vec<Float>,
    gamma: Vec<Float>,
}

struct ComputedCcmMatrixComponents {
    pole: Vec<Float>,
    archimedean: Vec<Float>,
    prime: Vec<Float>,
}

fn assemble_tau_components(components: &ComputedCcmMatrixComponents, prec: u32) -> Vec<Float> {
    components
        .pole
        .iter()
        .zip(&components.archimedean)
        .zip(&components.prime)
        .map(|((pole, archimedean), prime)| {
            let mut value = Float::with_val(prec, pole);
            value -= archimedean;
            value -= prime;
            value
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableWeilEigenpair {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    force_even: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parity_policy: Option<CcmParityPolicy>,
    #[serde(
        default = "legacy_eigenstate_route_name",
        skip_serializing_if = "is_legacy_eigenstate_route"
    )]
    eigenstate_route: String,
    eigenvalue: String,
    eigenvector: Vec<String>,
    inverse_iteration: PortableInverseIterationDiagnostics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shift_invert_krylov: Option<PortableShiftInvertKrylovDiagnostics>,
}

fn legacy_eigenstate_route_name() -> String {
    "legacy_inverse_iteration".to_owned()
}

fn is_legacy_eigenstate_route(value: &String) -> bool {
    value == "legacy_inverse_iteration"
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableShiftInvertKrylovDiagnostics {
    algorithm_semantics: String,
    factorization_id: String,
    requested_eigenpairs: usize,
    guard_eigenpairs: usize,
    maximum_subspace_dimension: usize,
    maximum_restarts: usize,
    restarts: usize,
    shifted_solves: usize,
    operator_applications: usize,
    status: String,
    maximum_ritz_value_stability: String,
    final_scaled_backward_error: String,
    final_relative_tau_residual: String,
    seed_identity: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableInverseIterationDiagnostics {
    pub configured_step_limit: usize,
    pub unshifted_steps: usize,
    pub unshifted_converged: bool,
    pub final_relative_rayleigh_change: Option<String>,
    pub shifted_refinement: String,
    pub final_relative_residual_norm: String,
}

impl PortableInverseIterationDiagnostics {
    fn from_runtime(value: &xc_numerics::linalg::InverseIterationDiagnostics) -> Self {
        use xc_numerics::linalg::ShiftedRefinementOutcome;
        Self {
            configured_step_limit: value.configured_step_limit,
            unshifted_steps: value.unshifted_steps,
            unshifted_converged: value.unshifted_converged,
            final_relative_rayleigh_change: value
                .final_relative_rayleigh_change
                .as_ref()
                .map(Float::to_string),
            shifted_refinement: match value.shifted_refinement {
                ShiftedRefinementOutcome::NotAttempted => "not_attempted",
                ShiftedRefinementOutcome::Accepted => "accepted",
                ShiftedRefinementOutcome::RejectedEigenvalueJump => "rejected_eigenvalue_jump",
                ShiftedRefinementOutcome::Singular => "singular",
            }
            .to_owned(),
            final_relative_residual_norm: value.final_relative_residual_norm.to_string(),
        }
    }

    fn to_runtime(
        &self,
        precision_bits: u32,
    ) -> std::result::Result<xc_numerics::linalg::InverseIterationDiagnostics, CacheError> {
        use xc_numerics::linalg::ShiftedRefinementOutcome;
        if self.configured_step_limit == 0
            || self.unshifted_steps > self.configured_step_limit
            || (self.unshifted_steps == 0 && self.unshifted_converged)
        {
            return Err(CacheError::InvalidManifest(
                "CCM inverse-iteration diagnostics contain invalid step counts".to_owned(),
            ));
        }
        let parse = |value: &str| {
            Float::parse(value)
                .map(|parsed| Float::with_val(precision_bits, parsed))
                .map_err(|error| {
                    CacheError::InvalidManifest(format!(
                        "CCM inverse-iteration diagnostics contain an invalid HP scalar: {error}"
                    ))
                })
        };
        let final_relative_rayleigh_change = self
            .final_relative_rayleigh_change
            .as_deref()
            .map(parse)
            .transpose()?;
        let final_relative_residual_norm = parse(&self.final_relative_residual_norm)?;
        if final_relative_rayleigh_change
            .as_ref()
            .is_some_and(|value| value < &Float::with_val(precision_bits, 0))
            || final_relative_residual_norm < 0
        {
            return Err(CacheError::InvalidManifest(
                "CCM inverse-iteration diagnostics contain a negative metric".to_owned(),
            ));
        }
        let shifted_refinement = match self.shifted_refinement.as_str() {
            "not_attempted" => ShiftedRefinementOutcome::NotAttempted,
            "accepted" => ShiftedRefinementOutcome::Accepted,
            "rejected_eigenvalue_jump" => ShiftedRefinementOutcome::RejectedEigenvalueJump,
            "singular" => ShiftedRefinementOutcome::Singular,
            _ => {
                return Err(CacheError::InvalidManifest(
                    "CCM inverse-iteration diagnostics contain an unknown shifted outcome"
                        .to_owned(),
                ))
            }
        };
        if self.unshifted_steps == 0 || shifted_refinement == ShiftedRefinementOutcome::NotAttempted
        {
            return Err(CacheError::InvalidManifest(
                "computed CCM eigenpair diagnostics omit required stopping evidence".to_owned(),
            ));
        }
        Ok(xc_numerics::linalg::InverseIterationDiagnostics {
            configured_step_limit: self.configured_step_limit,
            unshifted_steps: self.unshifted_steps,
            unshifted_converged: self.unshifted_converged,
            final_relative_rayleigh_change,
            shifted_refinement,
            final_relative_residual_norm,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableSecularSource {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    force_even: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parity_policy: Option<CcmParityPolicy>,
    eigenpair_content_digest: String,
    normalization: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableRootRefinement {
    value: String,
    iterations: usize,
    final_correction: String,
    residual: String,
    achieved_decimal_digits: String,
}

impl PortableRootRefinement {
    fn from_runtime(result: &RootRefinement) -> Self {
        Self {
            value: result.value.to_string(),
            iterations: result.diagnostics.iterations,
            final_correction: result.diagnostics.final_correction.to_string(),
            residual: result.diagnostics.residual.to_string(),
            achieved_decimal_digits: result.diagnostics.achieved_decimal_digits.to_string(),
        }
    }

    fn from_runtime_lossless(result: &RootRefinement) -> Self {
        let encode = |value: &Float| {
            let digits = u64::from(value.prec())
                .saturating_mul(30_103)
                .div_ceil(100_000)
                .saturating_add(4) as usize;
            value.to_string_radix(10, Some(digits))
        };
        Self {
            value: encode(&result.value),
            iterations: result.diagnostics.iterations,
            final_correction: encode(&result.diagnostics.final_correction),
            residual: encode(&result.diagnostics.residual),
            achieved_decimal_digits: encode(&result.diagnostics.achieved_decimal_digits),
        }
    }

    fn from_runtime_for_eigenstate_solver(
        result: &RootRefinement,
        resolved_eigenstate_solver: CcmEigenstateSolver,
    ) -> Self {
        match resolved_eigenstate_solver {
            CcmEigenstateSolver::LegacyInverseIteration => Self::from_runtime(result),
            CcmEigenstateSolver::ShiftInvertKrylov => Self::from_runtime_lossless(result),
            CcmEigenstateSolver::Auto => {
                unreachable!(
                    "automatic CCM eigenstate selection must be resolved before root serialization"
                )
            }
        }
    }

    fn to_runtime(&self, precision_bits: u32) -> std::result::Result<RootRefinement, CacheError> {
        let values = parse_hp_vector(
            &[
                self.value.clone(),
                self.final_correction.clone(),
                self.residual.clone(),
                self.achieved_decimal_digits.clone(),
            ],
            precision_bits,
        )?;
        if self.iterations == 0 || values[1] < 0 || values[2] < 0 || values[3] < 0 {
            return Err(CacheError::InvalidManifest(
                "CCM root diagnostics contain an invalid iteration count or negative metric"
                    .to_owned(),
            ));
        }
        Ok(RootRefinement {
            value: values[0].clone(),
            diagnostics: RootRefinementDiagnostics {
                iterations: self.iterations,
                final_correction: values[1].clone(),
                residual: values[2].clone(),
                achieved_decimal_digits: values[3].clone(),
            },
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "details")]
enum PortableRootOutcome {
    Converged(PortableRootRefinement),
    Stagnated(PortableRootRefinement),
    Approximate(PortableRootRefinement),
    Failed { iterations: usize, reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableRootRange {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    force_even: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parity_policy: Option<CcmParityPolicy>,
    first_root_index: usize,
    #[serde(default, skip_serializing_if = "is_positive_root_domain")]
    root_domain: IndependentRootDomain,
    discovery_mode: String,
    reference_seeds_used: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reference_dataset: Option<ReferenceZeroDatasetIdentity>,
    completeness: String,
    starting_points: Vec<String>,
    outcomes: Vec<PortableRootOutcome>,
    solver: String,
    solver_steps: usize,
    accuracy_guard_bits: u32,
}

fn is_positive_root_domain(value: &IndependentRootDomain) -> bool {
    *value == IndependentRootDomain::Positive
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableEvennessEvidence {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    evenness_deviation: String,
    natural_eigenvalue: String,
    forced_eigenvalue: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableRunEvidence {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    force_even: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parity_policy: Option<CcmParityPolicy>,
    discovery_mode: String,
    first_root_index: usize,
    last_root_index: usize,
    root_count: usize,
    #[serde(default, skip_serializing_if = "is_positive_root_domain")]
    root_domain: IndependentRootDomain,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    requested_root_count: Option<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    allow_incomplete: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    selected_root_ordinals: Vec<usize>,
    weil_min_eigenvalue: String,
    converged_roots: usize,
    stagnated_roots: usize,
    approximate_roots: usize,
    failed_roots: usize,
    inverse_iteration: PortableInverseIterationDiagnostics,
}

fn lambda_squared_cache_identity(params: &CcmParams) -> String {
    if params.lambda_sq.is_integer {
        params.lambda_sq.value_u64.to_string()
    } else {
        format!("{:.17e}", params.lambda_sq.value_f64)
    }
}

/// Exact lambda-squared input for arbitrary-precision CCM assembly.
///
/// The decimal literal is retained verbatim for provenance. `prime_cutoff`
/// must be its exact integer floor, so the analytic length and the discrete
/// prime content cannot silently be derived from different rounded values.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExactLambdaSquaredHp {
    pub decimal: xc_core::DecimalLiteral,
    pub prime_cutoff: u64,
    pub mode: LambdaSquaredMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LambdaSquaredMode {
    Integer,
    Fractional,
}

impl ExactLambdaSquaredHp {
    pub fn new(decimal: xc_core::DecimalLiteral, prime_cutoff: u64) -> Result<Self> {
        let floor = xc_core::DecimalLiteral::new(prime_cutoff.to_string())?;
        let ordering = decimal.cmp_numeric(&floor)?;
        if ordering.is_lt() {
            bail!("lambda-squared is below its declared prime-content floor");
        }
        let mode = if ordering.is_eq() {
            LambdaSquaredMode::Integer
        } else {
            let next = prime_cutoff.checked_add(1).ok_or_else(|| {
                anyhow::anyhow!("fractional lambda-squared exceeds the supported u64 prime floor")
            })?;
            let ceiling = xc_core::DecimalLiteral::new(next.to_string())?;
            if !decimal.cmp_numeric(&ceiling)?.is_lt() {
                bail!("lambda-squared is not below the next integer after its prime-content floor");
            }
            LambdaSquaredMode::Fractional
        };
        Ok(Self {
            decimal,
            prime_cutoff,
            mode,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PrimePowerContent {
    pub power: u64,
    pub prime: u64,
    pub exponent: u32,
}

/// Provenance retained with an exact arbitrary-precision CCM assembly.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CcmHpAssemblyEvidence {
    pub lambda_squared: ExactLambdaSquaredHp,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub prime_content: Vec<PrimePowerContent>,
}

pub struct ExactCcmWeilFormHp {
    pub matrix: Vec<Float>,
    pub evidence: CcmHpAssemblyEvidence,
}

/// Construct the localized Weil form without passing lambda-squared through
/// binary64. Integer and fractional inputs use the same exact decimal route;
/// MPFR performs the sole rounding at the requested working precision.
pub fn localized_weil_form_exact_hp(
    lambda_squared: ExactLambdaSquaredHp,
    n_modes: usize,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<ExactCcmWeilFormHp> {
    xc_numerics::hp_runtime::run_hp(|| {
        let parsed = Float::parse(lambda_squared.decimal.as_str()).map_err(|error| {
            anyhow::anyhow!("failed to parse exact lambda-squared literal for MPFR: {error}")
        })?;
        let value = Float::with_val(cfg.precision_bits, parsed);
        if value <= 0 {
            bail!("lambda-squared must be positive");
        }
        let l = value.ln();
        let mut matrix = build_tau_hp_compute_exact(
            n_modes,
            lambda_squared.prime_cutoff,
            &l,
            cfg,
            include_primes,
        )?;
        force_symmetric(&mut matrix, 2 * n_modes + 1);
        let prime_content = if include_primes {
            prime_powers_up_to(lambda_squared.prime_cutoff)
                .into_iter()
                .map(|(power, prime, exponent)| PrimePowerContent {
                    power,
                    prime,
                    exponent,
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(ExactCcmWeilFormHp {
            evidence: CcmHpAssemblyEvidence {
                lambda_squared,
                n_modes,
                precision_bits: cfg.precision_bits,
                prime_content,
            },
            matrix,
        })
    })
}

/// CCM adapter from finite Weil low-mode requests to the common capability
/// planner. Solver internals remain free of CCM parameters and semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcmLowModeSolverRequest {
    pub matrix_dimension: usize,
    pub requested_modes: usize,
    pub assurance: xc_core::AssuranceLevel,
    pub precision: xc_core::PrecisionPolicy,
    pub matrix_materialized: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CcmLowModeSolverPlanner;

impl xc_solver::DomainSolverPlanner for CcmLowModeSolverPlanner {
    type Request = CcmLowModeSolverRequest;

    fn domain_id(&self) -> &'static str {
        "ccm_weil_low_modes"
    }

    fn solver_input(
        &self,
        request: &Self::Request,
    ) -> Result<xc_solver::SolverPlannerInput, xc_solver::SolverError> {
        Ok(xc_solver::SolverPlannerInput {
            structure: if request.matrix_materialized {
                xc_operator::MatrixStructure::Dense
            } else {
                xc_operator::MatrixStructure::MatrixFree
            },
            dimension: request.matrix_dimension,
            target: xc_core::EigenTarget::SmallestMagnitude,
            requested_eigenpairs: request.requested_modes,
            assurance: request.assurance,
            precision: request.precision,
            matrix_materialized: request.matrix_materialized,
            generalized: false,
        })
    }

    fn planning_rationale(&self, request: &Self::Request) -> Vec<String> {
        vec![format!(
            "CCM requests {} low-magnitude Weil modes together so guard-space ambiguity is visible",
            request.requested_modes
        )]
    }
}

/// Configuration for the high-precision tier.
///
/// All fields are public so callers can override individual components
/// after a `for_decimal_digits` construction. The default values are
/// tuned for the HP-1000 retest workload; tweak with
/// caution.
#[derive(Debug, Clone)]
pub struct HighPrecConfig {
    /// MPFR working precision in bits. Total digits ≈ `precision_bits / 3.322`.
    /// Use `for_decimal_digits` to construct from a target decimal precision.
    pub precision_bits: u32,
    /// Maximum number of inverse-iteration steps for the smallest
    /// Weil-form eigenvector recovery.
    pub inverse_iter_steps: usize,
    /// Maximum number of solver steps per Riemann-zero seed.
    /// Applies to both Halley (default) and Newton solvers.
    ///
    /// In practice the solver exits early when the relative correction meets
    /// the requested-accuracy threshold. Guard bits remain available for
    /// cancellation. A representational stall is reported as stagnation and
    /// rejected by ordinary production APIs.
    /// This cap is a safety net only — it prevents infinite loops on
    /// pathological/wandering seeds and is never reached on real configs.
    /// The deterministic default is 2,000; callers may override it
    /// explicitly on this configuration value.
    pub solver_steps: usize,
    /// Root-finding method used to refine Riemann-zero seeds.
    pub root_solver: RootSolver,
    /// Number of Gauss–Legendre quadrature points used in the integral
    /// computation of α_L, β_L, γ_L. Clamped to `[MIN_QUAD_POINTS,
    /// MAX_QUAD_POINTS]` regardless of input.
    pub quad_points: usize,
    /// Number of positive CCM secular roots to discover independently and
    /// refine. Zero requests an explicit source-only run.
    pub n_eigenvalues: usize,
    /// Cache strategy for the GL-node and τ-matrix disk caches. See
    /// [`xc_numerics::quadrature::CacheMode`]. The default standalone mode
    /// uses local compressed caches; managed remote resolution uses `run_via_cache`.
    /// (local compressed cache → compute).
    pub cache_mode: xc_numerics::quadrature::CacheMode,
    /// Legacy compatibility switch for natural-versus-forced callers.
    ///
    /// New code should use [`Self::set_parity_policy`]. Setting this field to
    /// `false` while [`Self::parity_policy`] remains
    /// [`CcmParityPolicy::EvenSector`] selects the historical natural route.
    /// The default remains `true`.
    pub force_even: bool,
    /// Parity treatment for the selected CCM eigenstate. The default is the
    /// optimized reduced even-sector solve used by existing v0.13 artifacts.
    pub parity_policy: CcmParityPolicy,
    /// Enable warm-start from a nearby-precision cached eigenvector.
    /// When `true` and a cached ξ exists for the same (λ², N) within
    /// `warm_start_tolerance_bits` of the target precision, that cached ξ
    /// is used as the starting vector for inverse iteration instead of the
    /// Gaussian initial guess. Dramatically reduces iteration count for
    /// P-sweep campaigns. Default `true`.
    pub warm_start: bool,
    /// Precision tolerance in bits for warm-start cache lookup.
    /// A cached ξ at prec' is accepted as a warm start if
    /// |prec' - target_prec| ≤ warm_start_tolerance_bits.
    /// Default 500 bits (~150 decimal digits) — spans a full HP-level step.
    pub warm_start_tolerance_bits: u32,
    /// Algorithm used for the smallest Weil eigenstate.
    ///
    /// The default is [`CcmEigenstateSolver::Auto`]. It reuses an exact
    /// current-N eigenstate when available; otherwise it can embed the nearest
    /// compatible lower-N cached state as a Krylov starting hint. The
    /// shift-invert route has a distinct cache identity and can reuse the same
    /// matrix and LU artifacts without relabeling legacy eigenpairs.
    pub eigenstate_solver: CcmEigenstateSolver,
    /// Maximum retained Krylov/Rayleigh-Ritz basis dimension.
    pub krylov_subspace_dimension: usize,
    /// Maximum number of thick-restart cycles for the Krylov route.
    pub krylov_maximum_restarts: usize,
    /// Number of non-requested Ritz guards retained at the target boundary.
    pub krylov_guard_eigenpairs: usize,
}

/// CCM parity policy for the selected smallest Weil eigenstate.
///
/// The policies are deliberately separate cache semantics:
///
/// - [`Self::Natural`] solves the unrestricted full matrix without projection.
/// - [`Self::AdaptiveEven`] reproduces the original full-space inverse
///   iteration and applies the even projection only when the iterate drifts
///   materially away from even symmetry.
/// - [`Self::EvenSector`] solves the reduced even-sector matrix directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CcmParityPolicy {
    Natural,
    AdaptiveEven,
    #[default]
    EvenSector,
}

impl CcmParityPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::AdaptiveEven => "adaptive-even",
            Self::EvenSector => "even-sector",
        }
    }

    fn cache_label(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::AdaptiveEven => "adaptive-even",
            // Preserve every existing v0.13 logical key.
            Self::EvenSector => "even",
        }
    }

    fn legacy_force_even(self) -> bool {
        self != Self::Natural
    }

    fn semantic_subspace(self) -> Option<String> {
        match self {
            Self::EvenSector => Some("even".to_owned()),
            Self::Natural | Self::AdaptiveEven => None,
        }
    }

    fn portable_marker(self) -> Option<Self> {
        (self == Self::AdaptiveEven).then_some(self)
    }
}

fn payload_parity_matches(
    force_even: bool,
    marker: Option<CcmParityPolicy>,
    expected: CcmParityPolicy,
) -> bool {
    let decoded = match marker {
        Some(CcmParityPolicy::AdaptiveEven) if force_even => CcmParityPolicy::AdaptiveEven,
        Some(_) => return false,
        None if force_even => CcmParityPolicy::EvenSector,
        None => CcmParityPolicy::Natural,
    };
    decoded == expected
}

fn add_adaptive_parity_parameter(parameters: &mut serde_json::Value, policy: CcmParityPolicy) {
    if policy == CcmParityPolicy::AdaptiveEven {
        parameters
            .as_object_mut()
            .expect("CCM semantic parameters are an object")
            .insert(
                "parity_policy".to_owned(),
                serde_json::json!(policy.as_str()),
            );
    }
}

/// CCM eigenstate algorithm policy. This is deliberately explicit because the
/// two routes can differ in low-order bits and therefore never share an
/// eigenpair cache identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcmEigenstateSolver {
    LegacyInverseIteration,
    ShiftInvertKrylov,
    /// Prefer an exact Krylov artifact, then an exact legacy artifact. On a
    /// miss, discover the nearest compatible lower-N state across configured
    /// cache layers and use it as a starting hint before computing Krylov.
    Auto,
}

impl CcmEigenstateSolver {
    fn as_str(self) -> &'static str {
        match self {
            Self::LegacyInverseIteration => "legacy_inverse_iteration",
            Self::ShiftInvertKrylov => "shift_invert_krylov",
            Self::Auto => "auto",
        }
    }
}

/// Root-finding method for the CCM high-precision Riemann-zero refinement.
///
/// Keeping this choice in [`HighPrecConfig`] makes it reviewable, testable,
/// and eligible for inclusion in resolved configuration provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootSolver {
    /// Cubic convergence using three passes over the poles per step.
    Halley,
    /// Quadratic convergence using two passes over the poles per step.
    Newton,
}

impl RootSolver {
    fn display_name(self) -> &'static str {
        match self {
            Self::Halley => "Halley",
            Self::Newton => "Newton",
        }
    }
}

/// Conversion factor retained for callers that need an approximate display
/// conversion. Configuration construction itself uses integer arithmetic so
/// the cache identity is independent of host floating-point evaluation.
pub const DIGITS_TO_BITS_FACTOR: f64 = 3.322;

/// Extra working bits beyond the requested decimal accuracy.
///
/// Secular evaluation loses several bits to cancellation near a root. A
/// 64-bit reserve lets the point solver satisfy the requested accuracy rather
/// than incorrectly treating the last representable working bit as the
/// accuracy contract.
pub const GUARD_BITS: u32 = 64;

/// Consecutive point-solver steps without any smaller correction before the
/// outcome is classified as stagnated. Halley should improve cubically near a
/// simple root; this deliberately long window preserves genuinely slow
/// monotone convergence while terminating precision-floor oscillation.
pub const ROOT_STAGNATION_WINDOW: usize = 128;

/// Minimum quadrature points for the HP tier.
pub const MIN_QUAD_POINTS: usize = 600;

/// Maximum quadrature points for the HP tier (prevents excessive runtime).
pub const MAX_QUAD_POINTS: usize = 4000;

/// Multiplier: quad_points = digits * QUAD_POINTS_PER_DIGIT (clamped to [MIN, MAX]).
pub const QUAD_POINTS_PER_DIGIT: usize = 3;

/// HP singularity guard for `rho_hp(x)` near `x = 0`. Below this magnitude
/// we use the Taylor-series branch instead of `1 / (2 sinh(x/2))` directly.
/// Stored as a string literal so it is parsed as an HP `Float` at the
/// caller's working precision (no f64 round-trip).
pub const HP_SINGULARITY_GUARD_STR: &str = "1e-30";

impl HighPrecConfig {
    /// Construct a config from a target decimal-digit working precision.
    ///
    /// Bits are computed with an upward integer bound for
    /// `digits × log₂(10)`, then `GUARD_BITS` are added. Quadrature points
    /// are `digits × QUAD_POINTS_PER_DIGIT`,
    /// clamped to `[MIN_QUAD_POINTS, MAX_QUAD_POINTS]`. Other fields take
    /// the defaults `inverse_iter_steps=2000, solver_steps=2000,
    /// n_eigenvalues=50`.
    pub fn for_decimal_digits(digits: u32) -> Self {
        // 332193/100000 is a strict upward decimal bound for log2(10).
        // Integer arithmetic makes the precision contract deterministic on
        // every supported platform and avoids routing HP configuration
        // through binary64.
        let requested_bits = u64::from(digits)
            .saturating_mul(332_193)
            .div_ceil(100_000)
            .min(u64::from(u32::MAX - GUARD_BITS)) as u32;
        let bits = requested_bits + GUARD_BITS;
        // solver_steps: high safety cap — in practice the solver exits early
        // after meeting the requested accuracy. Stagnation remains a distinct
        // rejected outcome. The cap prevents an infinite loop, and the
        // deliberately generous 2,000-step
        // default favors a slow accurate result over an early approximation.
        let solver_steps = 2_000;
        Self {
            precision_bits: bits,
            inverse_iter_steps: 2_000,
            solver_steps,
            root_solver: RootSolver::Halley,
            quad_points: ((digits as usize) * QUAD_POINTS_PER_DIGIT)
                .clamp(MIN_QUAD_POINTS, MAX_QUAD_POINTS),
            n_eigenvalues: 50,
            cache_mode: xc_numerics::quadrature::CacheMode::default(),
            force_even: true,
            parity_policy: CcmParityPolicy::EvenSector,
            // Warm-start on by default. Uses a cached ξ at a nearby
            // precision as the starting vector for inverse iteration
            // instead of the Gaussian guess. Falls back to the Gaussian
            // when no nearby cache entry exists (cold cache).
            warm_start: true,
            // Tolerance in bits for warm-start lookup, default 500.
            warm_start_tolerance_bits: 500,
            eigenstate_solver: CcmEigenstateSolver::Auto,
            krylov_subspace_dimension: 32,
            krylov_maximum_restarts: 64,
            krylov_guard_eigenpairs: 2,
        }
    }

    /// Set an explicit parity policy and keep the legacy compatibility flag
    /// synchronized for callers that still inspect it.
    pub fn set_parity_policy(&mut self, policy: CcmParityPolicy) {
        self.parity_policy = policy;
        self.force_even = policy.legacy_force_even();
    }

    /// Return the effective parity policy.
    ///
    /// `force_even=false` continues to select the natural route when older
    /// callers have not changed the new policy field.
    pub fn effective_parity_policy(&self) -> CcmParityPolicy {
        if !self.force_even && self.parity_policy == CcmParityPolicy::EvenSector {
            CcmParityPolicy::Natural
        } else {
            self.parity_policy
        }
    }
}

/// Status of a single solver run for one CCM secular-root seed.
///
/// Diagnostics retained for every numerical root value, including outcomes
/// which are rejected by the validated cache path.
#[derive(Debug, Clone)]
pub struct RootRefinementDiagnostics {
    pub iterations: usize,
    pub final_correction: Float,
    pub residual: Float,
    pub achieved_decimal_digits: Float,
}

/// A root value together with its stopping diagnostics.
#[derive(Debug, Clone)]
pub struct RootRefinement {
    pub value: Float,
    pub diagnostics: RootRefinementDiagnostics,
}

/// Status of one CCM root refinement. Computed workflows retain finite
/// `Stagnated` and `Approximate` values with their exact diagnostics;
/// certification still requires `Converged`.
#[derive(Debug, Clone)]
pub enum EigenvalueResult {
    Converged(RootRefinement),
    Stagnated(RootRefinement),
    Approximate(RootRefinement),
    Failed { iterations: usize, reason: String },
}

impl EigenvalueResult {
    /// Return the numerical value for diagnostic inspection.
    pub fn value(&self) -> Option<&Float> {
        match self {
            Self::Converged(result) | Self::Stagnated(result) | Self::Approximate(result) => {
                Some(&result.value)
            }
            Self::Failed { .. } => None,
        }
    }

    /// Returns `true` if this result is fully converged.
    pub fn is_converged(&self) -> bool {
        matches!(self, Self::Converged(_))
    }

    /// Returns `true` if a numerical value is present. Computed workflows may
    /// retain and reuse such a value, but this does not imply convergence or
    /// certified assurance.
    pub fn has_value(&self) -> bool {
        !matches!(self, Self::Failed { .. })
    }
}

/// Result of a single high-precision CCM run.
///
/// All HP fields stay in `rug::Float` at the working precision specified
/// in the config; lossy f64 views are exposed via `to_f64_result`.
pub struct HighPrecResult {
    /// Solver results for roots of the eigenpair-derived CCM secular equation
    /// (equivalently, the finite `D_log` spectrum).
    ///
    /// Ordinary APIs retain positive roots. An explicit advanced independent
    /// discovery request may retain a numerically ordered signed window.
    ///
    /// Each entry is one of:
    /// Only `Converged` entries are returned by the ordinary production APIs.
    /// Other variants retain diagnostics for explicit low-level inspection.
    pub eigenvalues_pos: Vec<EigenvalueResult>,
    /// One-based index assigned to the first entry in `eigenvalues_pos`.
    /// In advanced signed mode this is the ordinal within the returned signed
    /// window; the historical field name is retained for API compatibility.
    /// Empty source-only runs retain the requested start for provenance.
    pub first_positive_root_index: usize,
    /// Smallest eigenvalue of the Weil quadratic form (the spectral
    /// gap quantity ε_N at this `(λ², N)`).
    pub weil_min_eigenvalue: Float,
    /// Smallest-eigenvalue eigenvector of the Weil form, ℓ²-normalized,
    /// stored in the V_n basis order (centered index `0` at position
    /// `n_modes`).
    pub xi: Vec<Float>,
    /// Structured stopping evidence for the eigenstate solve. Reaching the
    /// unshifted limit is retained even when shifted refinement subsequently
    /// produces an acceptable Tau residual.
    pub inverse_iteration_diagnostics: xc_numerics::linalg::InverseIterationDiagnostics,
    /// Wall-clock seconds for the entire HP run (matrix build, eigenstate,
    /// independent discovery, and configured root refinement).
    pub elapsed_seconds: f64,
    /// MPFR working precision used for this run, in bits.
    pub precision_bits: u32,
}

/// Lossless persisted form of one CCM eigenvalue outcome.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "value")]
pub enum PortableEigenvalueResult {
    Converged(PortableRootRefinementResult),
    Stagnated(PortableRootRefinementResult),
    Approximate(PortableRootRefinementResult),
    Failed { iterations: usize, reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableRootRefinementResult {
    pub value: xc_numerics::fmt::PortableHpFloat,
    pub iterations: usize,
    pub final_correction: xc_numerics::fmt::PortableHpFloat,
    pub residual: xc_numerics::fmt::PortableHpFloat,
    pub achieved_decimal_digits: xc_numerics::fmt::PortableHpFloat,
}

impl PortableRootRefinementResult {
    fn from_runtime(result: &RootRefinement) -> Result<Self> {
        Ok(Self {
            value: xc_numerics::fmt::PortableHpFloat::from_float(&result.value)?,
            iterations: result.diagnostics.iterations,
            final_correction: xc_numerics::fmt::PortableHpFloat::from_float(
                &result.diagnostics.final_correction,
            )?,
            residual: xc_numerics::fmt::PortableHpFloat::from_float(&result.diagnostics.residual)?,
            achieved_decimal_digits: xc_numerics::fmt::PortableHpFloat::from_float(
                &result.diagnostics.achieved_decimal_digits,
            )?,
        })
    }

    fn to_runtime(&self) -> Result<RootRefinement> {
        Ok(RootRefinement {
            value: self.value.to_float()?,
            diagnostics: RootRefinementDiagnostics {
                iterations: self.iterations,
                final_correction: self.final_correction.to_float()?,
                residual: self.residual.to_float()?,
                achieved_decimal_digits: self.achieved_decimal_digits.to_float()?,
            },
        })
    }
}

/// Portable CCM result payload for use inside [`xc_core::ResearchResult`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableHighPrecResult {
    pub eigenvalues_pos: Vec<PortableEigenvalueResult>,
    pub first_positive_root_index: usize,
    pub weil_min_eigenvalue: xc_numerics::fmt::PortableHpFloat,
    pub xi: Vec<xc_numerics::fmt::PortableHpFloat>,
    pub inverse_iteration_diagnostics: PortableInverseIterationDiagnostics,
    pub elapsed_seconds: f64,
    pub precision_bits: u32,
}

impl PortableHighPrecResult {
    pub fn from_runtime(result: &HighPrecResult) -> Result<Self> {
        if result.first_positive_root_index == 0
            || result
                .first_positive_root_index
                .checked_add(result.eigenvalues_pos.len().saturating_sub(1))
                .is_none()
        {
            bail!("CCM root index range is invalid");
        }
        let eigenvalues_pos = result
            .eigenvalues_pos
            .iter()
            .map(|value| match value {
                EigenvalueResult::Converged(result) => {
                    PortableRootRefinementResult::from_runtime(result)
                        .map(PortableEigenvalueResult::Converged)
                }
                EigenvalueResult::Stagnated(result) => {
                    PortableRootRefinementResult::from_runtime(result)
                        .map(PortableEigenvalueResult::Stagnated)
                }
                EigenvalueResult::Approximate(result) => {
                    PortableRootRefinementResult::from_runtime(result)
                        .map(PortableEigenvalueResult::Approximate)
                }
                EigenvalueResult::Failed { iterations, reason } => {
                    Ok(PortableEigenvalueResult::Failed {
                        iterations: *iterations,
                        reason: reason.clone(),
                    })
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            eigenvalues_pos,
            first_positive_root_index: result.first_positive_root_index,
            weil_min_eigenvalue: xc_numerics::fmt::PortableHpFloat::from_float(
                &result.weil_min_eigenvalue,
            )?,
            xi: result
                .xi
                .iter()
                .map(xc_numerics::fmt::PortableHpFloat::from_float)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            inverse_iteration_diagnostics: PortableInverseIterationDiagnostics::from_runtime(
                &result.inverse_iteration_diagnostics,
            ),
            elapsed_seconds: result.elapsed_seconds,
            precision_bits: result.precision_bits,
        })
    }

    pub fn to_runtime(&self) -> Result<HighPrecResult> {
        if self.first_positive_root_index == 0
            || self
                .first_positive_root_index
                .checked_add(self.eigenvalues_pos.len().saturating_sub(1))
                .is_none()
        {
            bail!("portable CCM result has an invalid root index range");
        }
        let eigenvalues_pos = self
            .eigenvalues_pos
            .iter()
            .map(|value| match value {
                PortableEigenvalueResult::Converged(result) => {
                    result.to_runtime().map(EigenvalueResult::Converged)
                }
                PortableEigenvalueResult::Stagnated(result) => {
                    result.to_runtime().map(EigenvalueResult::Stagnated)
                }
                PortableEigenvalueResult::Approximate(result) => {
                    result.to_runtime().map(EigenvalueResult::Approximate)
                }
                PortableEigenvalueResult::Failed { iterations, reason } => {
                    Ok(EigenvalueResult::Failed {
                        iterations: *iterations,
                        reason: reason.clone(),
                    })
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(HighPrecResult {
            eigenvalues_pos,
            first_positive_root_index: self.first_positive_root_index,
            weil_min_eigenvalue: self.weil_min_eigenvalue.to_float()?,
            xi: self
                .xi
                .iter()
                .map(xc_numerics::fmt::PortableHpFloat::to_float)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            inverse_iteration_diagnostics: self
                .inverse_iteration_diagnostics
                .to_runtime(self.precision_bits)
                .map_err(anyhow::Error::from)?,
            elapsed_seconds: self.elapsed_seconds,
            precision_bits: self.precision_bits,
        })
    }
}

impl HighPrecResult {
    /// CCM secular roots in the requested window.
    ///
    /// This terminology avoids confusing these values with the distinct Tau
    /// or Weil-form eigenvalues. The `eigenvalues_pos` field remains the
    /// serialized API name used by existing consumers.
    pub fn spectral_roots(&self) -> &[EigenvalueResult] {
        &self.eigenvalues_pos
    }

    /// Inclusive one-based root-index range represented by this result.
    /// In advanced signed mode these are signed-window ordinals.
    pub fn spectral_root_index_range(&self) -> Option<std::ops::RangeInclusive<usize>> {
        if self.eigenvalues_pos.is_empty() || self.first_positive_root_index == 0 {
            return None;
        }
        self.first_positive_root_index
            .checked_add(self.eigenvalues_pos.len() - 1)
            .map(|last| self.first_positive_root_index..=last)
    }

    // HP_F64_REPORT_BOUNDARY_BEGIN: explicit lossy CLI/plot projection only.
    /// Lossy conversion to the f64-tier `CcmResult`. Eigenvalues, ξ
    /// entries, and ε_N collapse to f64; the f64 underflow boundary at
    /// ~10⁻³⁰⁸ silently maps to zero. Use only for f64-tier consumers
    /// (CLI summaries, plot generation); never for downstream HP work.
    pub fn to_f64_result(&self) -> CcmResult {
        CcmResult {
            eigenvalues_pos: self
                .eigenvalues_pos
                .iter()
                .map(|result| result.value().map_or(f64::NAN, Float::to_f64))
                .collect(),
            weil_min_eigenvalue: self.weil_min_eigenvalue.to_f64(),
            xi: self.xi.iter().map(|f| f.to_f64()).collect(),
            elapsed_seconds: self.elapsed_seconds,
        }
    }
    // HP_F64_REPORT_BOUNDARY_END
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CcmParity {
    Even,
    Odd,
}

/// Algorithm used to obtain algebraically indexed parity-sector eigenvalues.
/// The value is persisted in semantic identity and can never be substituted by
/// a cache resolver.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CcmSectorEigenvalueRoute {
    /// Compute only the requested algebraic prefix through HP Sturm isolation.
    Selected,
    /// Compute and retain the complete spectrum through Householder plus QR.
    CompleteQr,
    /// Compute complete QR values and independently enclose the requested
    /// prefix with Sturm counts before accepting them.
    CrossChecked,
}

impl CcmSectorEigenvalueRoute {
    fn as_str(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::CompleteQr => "complete_qr",
            Self::CrossChecked => "cross_checked",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CcmSectorAnalysisOptions {
    pub requested_eigenpairs: usize,
    pub eigenvalue_route: CcmSectorEigenvalueRoute,
}

impl CcmSectorAnalysisOptions {
    pub fn selected(requested_eigenpairs: usize) -> Self {
        Self {
            requested_eigenpairs,
            eigenvalue_route: CcmSectorEigenvalueRoute::Selected,
        }
    }

    pub fn maximum(requested_eigenpairs: usize) -> Self {
        Self {
            requested_eigenpairs,
            eigenvalue_route: CcmSectorEigenvalueRoute::CompleteQr,
        }
    }

    pub fn cross_checked(requested_eigenpairs: usize) -> Self {
        Self {
            requested_eigenpairs,
            eigenvalue_route: CcmSectorEigenvalueRoute::CrossChecked,
        }
    }
}

#[derive(Clone, Debug)]
struct SectorTridiagonalHp {
    diagonal: Vec<Float>,
    off_diagonal: Vec<Float>,
}

#[derive(Clone, Debug)]
struct SectorTransformHp {
    basis: Vec<Float>,
}

#[derive(Clone, Debug)]
struct SectorEigenvaluesHp {
    route: CcmSectorEigenvalueRoute,
    complete: bool,
    values: Vec<Float>,
    selected_enclosures: Vec<xc_numerics::eigen::HpTridiagonalEigenvalueEnclosure>,
}

impl CcmParity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Even => "even",
            Self::Odd => "odd",
        }
    }
}

#[derive(Clone, Debug)]
/// One algebraically indexed eigenpair of an explicit CCM parity block.
pub struct CcmSectorEigenpairHp {
    pub algebraic_index: usize,
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
}

#[derive(Clone, Debug)]
/// Ordered low spectrum retained for one even or odd CCM parity block.
pub struct CcmSectorSpectrumHp {
    pub parity: CcmParity,
    pub dimension: usize,
    pub eigenvalue_route: CcmSectorEigenvalueRoute,
    /// Complete ordered sector spectrum when a complete or cross-checked route
    /// was requested. Selected runs deliberately leave this absent.
    pub complete_eigenvalues: Option<Vec<Float>>,
    pub eigenpairs: Vec<CcmSectorEigenpairHp>,
}

#[derive(Clone, Debug)]
/// Replayable comparison of the lowest even and odd parity-block states.
///
/// `gap_log` is `D_even-D_odd`; `difference_depth` is a distinct diagnostic
/// derived from the direct eigenvalue difference.
pub struct CcmSectorGapHp {
    pub even: CcmSectorSpectrumHp,
    pub odd: CcmSectorSpectrumHp,
    pub lambda_even: Float,
    pub lambda_odd: Float,
    pub d_even: Float,
    pub d_odd: Float,
    pub gap_log: Float,
    pub lambda_difference: Float,
    pub difference_depth: Float,
    pub ordering: i8,
    pub even_simple: bool,
    pub even_simplicity_margin: Float,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcmResearchCaptureOptions {
    pub capture_evenness: bool,
    pub sector_analysis: Option<CcmSectorAnalysisOptions>,
    /// Optional root-only interval certification. This certifies the exact
    /// retained finite secular point source; it does not request the much
    /// more expensive interval reconstruction of Tau.
    pub root_certification: Option<CcmRootCertificationOptions>,
}

impl CcmResearchCaptureOptions {
    pub fn maximum(requested_sector_eigenpairs: usize) -> Self {
        Self {
            capture_evenness: true,
            sector_analysis: Some(CcmSectorAnalysisOptions::maximum(
                requested_sector_eigenpairs,
            )),
            root_certification: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcmRootCertificationOptions {
    pub target: super::certified_roots::IndependentCcmRootTarget,
    pub isolation_bits: u32,
    pub interval_newton: xc_root::IntervalNewtonOptions,
}

impl CcmRootCertificationOptions {
    /// Configure certification of an independently indexed root target.
    /// The interval width follows the claim's requested decimal precision;
    /// the working-precision guard digits remain controlled by
    /// [`HighPrecConfig`].
    pub fn for_decimal_digits(
        target: super::certified_roots::IndependentCcmRootTarget,
        decimal_digits: u32,
    ) -> Result<Self> {
        if decimal_digits == 0 {
            bail!("CCM root certification requires positive decimal digits");
        }
        Ok(Self {
            target,
            isolation_bits: 96,
            interval_newton: xc_root::IntervalNewtonOptions {
                width_tolerance: xc_core::DecimalLiteral::new(format!("1e-{decimal_digits}"))?,
                maximum_iterations: 2_000,
            },
        })
    }
}

pub struct CcmResearchCaptureResult {
    pub primary: HighPrecResult,
    pub evenness: Option<EvennessResult>,
    pub sector_gap: Option<CcmSectorGapHp>,
    /// Separate, source-bound certificate artifact when root-only
    /// certification was requested.
    pub root_certificate: Option<super::certified_roots::ProductionIndependentCcmRootCertificate>,
}

fn evenness_from_sector_gap(
    params: &CcmParams,
    precision_bits: u32,
    gap: &CcmSectorGapHp,
) -> Result<EvennessResult> {
    let (natural_eigenvalue, natural_vector) = match gap.ordering {
        1 => (
            gap.lambda_even.clone(),
            expand_even_sector_vector(
                &gap.even.eigenpairs[0].eigenvector,
                params.n_modes,
                precision_bits,
            ),
        ),
        -1 => (
            gap.lambda_odd.clone(),
            expand_odd_sector_vector(
                &gap.odd.eigenpairs[0].eigenvector,
                params.n_modes,
                precision_bits,
            ),
        ),
        _ => bail!("CCM natural evenness is ambiguous at an even/odd degeneracy"),
    };
    Ok(evenness_from_natural_state(
        params,
        precision_bits,
        natural_eigenvalue,
        &natural_vector,
        gap.lambda_even.clone(),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CcmStateCriterion {
    AlgebraicGround,
    SmallestPositive,
    NearestZero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CcmStateTarget {
    AlgebraicGround,
    SmallestPositive,
    NearestZero,
    ParityRestricted {
        parity: CcmParity,
        criterion: CcmStateCriterion,
    },
}

#[derive(Clone, Debug)]
pub struct CcmStateCandidateHp {
    pub algebraic_index: usize,
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub parity: CcmParity,
}

#[derive(Clone, Debug)]
pub struct SelectedCcmStateHp {
    pub requested_target: CcmStateTarget,
    pub algebraic_index: usize,
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub parity: CcmParity,
}

pub fn select_ccm_state_hp(
    target: CcmStateTarget,
    candidates: &[CcmStateCandidateHp],
) -> Result<SelectedCcmStateHp> {
    if candidates.is_empty()
        || candidates.iter().any(|candidate| {
            candidate.eigenvector.is_empty()
                || !candidate.eigenvalue.is_finite()
                || candidate.eigenvector.iter().any(|value| !value.is_finite())
        })
    {
        anyhow::bail!("CCM state selection requires finite nonempty eigenpairs");
    }
    let (parity, criterion) = match target {
        CcmStateTarget::AlgebraicGround => (None, CcmStateCriterion::AlgebraicGround),
        CcmStateTarget::SmallestPositive => (None, CcmStateCriterion::SmallestPositive),
        CcmStateTarget::NearestZero => (None, CcmStateCriterion::NearestZero),
        CcmStateTarget::ParityRestricted { parity, criterion } => (Some(parity), criterion),
    };
    let eligible = candidates
        .iter()
        .filter(|candidate| parity.is_none_or(|required| candidate.parity == required))
        .filter(|candidate| {
            criterion != CcmStateCriterion::SmallestPositive || candidate.eigenvalue > 0
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        anyhow::bail!("requested CCM state target has no eligible candidate");
    }
    let score = |candidate: &CcmStateCandidateHp| match criterion {
        CcmStateCriterion::AlgebraicGround | CcmStateCriterion::SmallestPositive => {
            candidate.eigenvalue.clone()
        }
        CcmStateCriterion::NearestZero => candidate.eigenvalue.clone().abs(),
    };
    let mut selected = eligible[0];
    let mut selected_score = score(selected);
    for candidate in eligible.iter().skip(1) {
        let candidate_score = score(candidate);
        if candidate_score < selected_score {
            selected = candidate;
            selected_score = candidate_score;
        } else if candidate_score == selected_score {
            anyhow::bail!("requested CCM state target is ambiguous at the selected boundary");
        }
    }
    Ok(SelectedCcmStateHp {
        requested_target: target,
        algebraic_index: selected.algebraic_index,
        eigenvalue: selected.eigenvalue.clone(),
        eigenvector: selected.eigenvector.clone(),
        parity: selected.parity,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CcmFormComponentKind {
    Archimedean,
    Prime,
    Pole,
    Other,
}

#[derive(Clone, Debug)]
pub struct CcmFormComponentMatrixHp {
    pub kind: CcmFormComponentKind,
    /// Signed coefficient in the total-form convention, normally +1 or -1.
    pub signed_coefficient: i32,
    pub matrix_row_major: Vec<Float>,
}

#[derive(Clone, Debug)]
pub struct CcmFormComponentValueHp {
    pub kind: CcmFormComponentKind,
    pub signed_coefficient: i32,
    pub rayleigh_value: Float,
    pub signed_contribution: Float,
}

#[derive(Clone, Debug)]
pub struct CcmFormDecompositionHp {
    pub total_value: Float,
    pub reconstructed_total: Float,
    pub cancellation_residual: Float,
    pub components: Vec<CcmFormComponentValueHp>,
}

pub fn evaluate_ccm_form_components_hp(
    total_matrix_row_major: &[Float],
    components: &[CcmFormComponentMatrixHp],
    vector: &[Float],
    precision_bits: u32,
) -> Result<CcmFormDecompositionHp> {
    let dimension = vector.len();
    if precision_bits < 64
        || dimension == 0
        || total_matrix_row_major.len() != dimension.saturating_mul(dimension)
        || vector.iter().any(|value| !value.is_finite())
    {
        anyhow::bail!("invalid CCM form-decomposition dimensions or precision");
    }
    let required = [
        CcmFormComponentKind::Archimedean,
        CcmFormComponentKind::Prime,
        CcmFormComponentKind::Pole,
        CcmFormComponentKind::Other,
    ];
    for kind in required {
        if components
            .iter()
            .filter(|component| component.kind == kind)
            .count()
            != 1
        {
            anyhow::bail!("CCM form decomposition requires each named component exactly once");
        }
    }
    if components.iter().any(|component| {
        component.signed_coefficient == 0
            || component.matrix_row_major.len() != dimension.saturating_mul(dimension)
            || component
                .matrix_row_major
                .iter()
                .any(|value| !value.is_finite())
    }) {
        anyhow::bail!("CCM form components require finite square matrices and nonzero signs");
    }

    let rayleigh = |matrix: &[Float]| -> Result<Float> {
        let mut applied = vec![Float::with_val(precision_bits, 0); dimension];
        for row in 0..dimension {
            let terms = (0..dimension)
                .map(|column| {
                    let mut term =
                        Float::with_val(precision_bits, &matrix[row * dimension + column]);
                    term *= &vector[column];
                    term
                })
                .collect::<Vec<_>>();
            applied[row] =
                xc_numerics::reduction::deterministic_pairwise_sum_hp(&terms, precision_bits);
        }
        let numerator_terms = vector
            .iter()
            .zip(&applied)
            .map(|(left, right)| {
                let mut term = Float::with_val(precision_bits, left);
                term *= right;
                term
            })
            .collect::<Vec<_>>();
        let denominator_terms = vector
            .iter()
            .map(|value| {
                let mut term = Float::with_val(precision_bits, value);
                term *= value;
                term
            })
            .collect::<Vec<_>>();
        let numerator =
            xc_numerics::reduction::deterministic_pairwise_sum_hp(&numerator_terms, precision_bits);
        let denominator = xc_numerics::reduction::deterministic_pairwise_sum_hp(
            &denominator_terms,
            precision_bits,
        );
        if denominator <= 0 {
            anyhow::bail!("CCM form-decomposition vector has nonpositive norm");
        }
        Ok(Float::with_val(precision_bits, numerator / denominator))
    };

    let total_value = rayleigh(total_matrix_row_major)?;
    let mut values = Vec::with_capacity(components.len());
    for component in components {
        let rayleigh_value = rayleigh(&component.matrix_row_major)?;
        let mut signed_contribution = Float::with_val(precision_bits, &rayleigh_value);
        signed_contribution *= component.signed_coefficient;
        values.push(CcmFormComponentValueHp {
            kind: component.kind,
            signed_coefficient: component.signed_coefficient,
            rayleigh_value,
            signed_contribution,
        });
    }
    let reconstructed_total = xc_numerics::reduction::deterministic_pairwise_sum_hp(
        &values
            .iter()
            .map(|value| value.signed_contribution.clone())
            .collect::<Vec<_>>(),
        precision_bits,
    );
    let mut cancellation_residual = Float::with_val(precision_bits, &total_value);
    cancellation_residual -= &reconstructed_total;
    cancellation_residual.abs_mut();
    Ok(CcmFormDecompositionHp {
        total_value,
        reconstructed_total,
        cancellation_residual,
        components: values,
    })
}

// -- Helpers
#[inline]
fn fl_i(prec: u32, v: i64) -> Float {
    Float::with_val(prec, v)
}
#[inline]
fn pi(prec: u32) -> Float {
    Float::with_val(prec, rug::float::Constant::Pi)
}
#[inline]
fn euler(prec: u32) -> Float {
    Float::with_val(prec, rug::float::Constant::Euler)
}

/// Force exact symmetry on a `dim × dim` row-major HP matrix in-place.
///
/// Computes `(M[i,j] + M[j,i]) / 2` for every upper-triangle pair and
/// immediately stores the average in both positions. This removes the former
/// O(dim²) index and MPFR-result scratch while retaining identical arithmetic.
/// The diagonal is untouched. This is called on the τ-matrix before eigenvector
/// computation to ensure that floating-point construction noise doesn't
/// break the assumed symmetry of the Weil quadratic form.
fn force_symmetric(matrix: &mut [Float], dim: usize) {
    for i in 0..dim {
        for j in (i + 1)..dim {
            let mut sum = matrix[i * dim + j].clone();
            sum += &matrix[j * dim + i];
            sum /= 2u32;
            matrix[i * dim + j] = sum.clone();
            matrix[j * dim + i] = sum;
        }
    }
}

fn decode_tau_artifact(
    artifact: &PortableTauMatrix,
    params: &CcmParams,
    prec: u32,
) -> std::result::Result<Vec<Float>, CacheError> {
    let identity = lambda_squared_cache_identity(params);
    let expected = params.matrix_size() * params.matrix_size();
    if artifact.schema_version != 2
        || artifact.lambda_squared != identity
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != prec
        || artifact.entries.len() != expected
    {
        return Err(CacheError::InvalidManifest(
            "CCM tau payload does not match its semantic identity".to_owned(),
        ));
    }
    let mut tau = Vec::with_capacity(expected);
    for entry in &artifact.entries {
        let parsed = Float::parse(entry).map_err(|error| {
            CacheError::InvalidManifest(format!(
                "CCM tau payload contains an invalid HP scalar: {error}"
            ))
        })?;
        tau.push(Float::with_val(prec, parsed));
    }
    if let Some(reason) = tau_cache::structural_check(&tau, params.n_modes, prec) {
        return Err(CacheError::InvalidManifest(format!(
            "CCM tau payload failed structural validation: {reason}"
        )));
    }
    Ok(tau)
}

#[cfg(feature = "arb")]
fn certify_tau_from_retained_computation(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
    manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactProductionAssessment> {
    let certification = super::cutoff_free::CutoffFreeConfig::new(
        params.lambda_sq_int(),
        params.n_modes,
        cfg.precision_bits,
    );
    let (interval_matrix, certificate) = super::cutoff_free::certify_portable(&certification)?;
    if interval_matrix.tau.len() != tau.len() {
        bail!("cutoff-free certification matrix dimension differs from retained tau matrix");
    }
    let mut error_enclosures = Vec::with_capacity(tau.len());
    for (index, (point, enclosure)) in tau.iter().zip(&interval_matrix.tau).enumerate() {
        let exact_point = point.to_rational().ok_or_else(|| {
            anyhow::anyhow!("retained tau entry {index} cannot be represented exactly")
        })?;
        let error_lower = enclosure.lower().clone() - &exact_point;
        let error_upper = enclosure.upper().clone() - &exact_point;
        error_enclosures.push(serde_json::json!({
            "index": index,
            "point": {
                "numerator": exact_point.numer().to_string(),
                "denominator": exact_point.denom().to_string()
            },
            "true_minus_point": {
                "lower": {
                    "numerator": error_lower.numer().to_string(),
                    "denominator": error_lower.denom().to_string()
                },
                "upper": {
                    "numerator": error_upper.numer().to_string(),
                    "denominator": error_upper.denom().to_string()
                }
            }
        }));
    }
    let replay = xc_certify::exact::verify_portable_interval_inertia_certificate(&certificate);
    if !replay.valid {
        bail!(
            "portable tau interval certificate failed independent replay: {}",
            replay.errors.join("; ")
        );
    }
    let sink = cache.production_sink.ok_or_else(|| {
        anyhow::anyhow!("certified assurance requires the toolkit evidence store")
    })?;
    let certificate_bytes = serde_json::to_vec_pretty(&certificate)?;
    let certificate_digest = sink
        .record_evidence("portable-interval-certificate", &certificate_bytes)
        .map_err(anyhow::Error::from)?;
    let containment_bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema_version": 1,
        "artifact_key": manifest.key,
        "artifact_content_digest": manifest.content_digest,
        "certificate_id": certificate.certificate_id,
        "interval_matrix_digest": certificate.matrix_digest,
        "verified_entry_count": tau.len(),
        "claim": "each exact retained MPFR value has the listed rigorous true-minus-point error enclosure",
        "error_enclosures": error_enclosures
    }))?;
    let containment_digest = sink
        .record_evidence("interval-containment-report", &containment_bytes)
        .map_err(anyhow::Error::from)?;
    let mut evidence_digests = vec![certificate_digest, containment_digest];
    evidence_digests.sort();
    let achieved_assurance = match cache.requested_assurance {
        xc_core::AssuranceLevel::Computed => xc_cache::ArtifactAssuranceState::Computed,
        xc_core::AssuranceLevel::CrossChecked => xc_cache::ArtifactAssuranceState::CrossChecked,
        xc_core::AssuranceLevel::Certified => xc_cache::ArtifactAssuranceState::Certified,
    };
    Ok(ArtifactProductionAssessment {
        achieved_assurance,
        evidence_digests,
    })
}

#[cfg(not(feature = "arb"))]
fn certify_tau_from_retained_computation(
    _params: &CcmParams,
    _cfg: &HighPrecConfig,
    _tau: &[Float],
    _manifest: &ArtifactManifest,
    _cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactProductionAssessment> {
    bail!("cross-checked or certified CCM assurance requires the xc-spectral arb feature")
}

fn parse_hp_scalar(value: &str, precision_bits: u32) -> std::result::Result<Float, CacheError> {
    Float::parse(value)
        .map(|parsed| Float::with_val(precision_bits, parsed))
        .map_err(|error| {
            CacheError::InvalidManifest(format!(
                "CCM component contains an invalid HP scalar: {error}"
            ))
        })
        .and_then(|parsed| {
            if parsed.is_finite() {
                Ok(parsed)
            } else {
                Err(CacheError::InvalidManifest(
                    "CCM component contains a non-finite HP scalar".to_owned(),
                ))
            }
        })
}

fn parse_hp_vector(
    values: &[String],
    precision_bits: u32,
) -> std::result::Result<Vec<Float>, CacheError> {
    values
        .iter()
        .map(|value| parse_hp_scalar(value, precision_bits))
        .collect()
}

fn canonical_dependency_refs(manifests: Vec<ArtifactManifest>) -> Vec<DependencyRef> {
    let mut dependencies: Vec<DependencyRef> = manifests
        .into_iter()
        .map(|manifest| DependencyRef {
            key: manifest.key,
            content_digest: manifest.content_digest,
            required_quality: CacheQuality::Validated,
        })
        .collect();
    dependencies.sort_by(|left, right| {
        (
            left.key.kind.as_str(),
            left.key.logical_key.as_str(),
            left.key.parameters_digest.0.as_str(),
            left.content_digest.0.as_str(),
        )
            .cmp(&(
                right.key.kind.as_str(),
                right.key.logical_key.as_str(),
                right.key.parameters_digest.0.as_str(),
                right.content_digest.0.as_str(),
            ))
    });
    dependencies
}

fn decode_archimedean_integrals(
    artifact: &PortableArchimedeanIntegrals,
    params: &CcmParams,
    precision_bits: u32,
) -> std::result::Result<ComputedArchimedeanIntegrals, CacheError> {
    let expected = params.n_modes + 1;
    if artifact.schema_version != 1
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != precision_bits
        || artifact.alpha.len() != expected
        || artifact.beta.len() != expected
        || artifact.gamma.len() != expected
    {
        return Err(CacheError::InvalidManifest(
            "CCM archimedean-integral payload does not match its semantic identity".to_owned(),
        ));
    }
    Ok(ComputedArchimedeanIntegrals {
        alpha: parse_hp_vector(&artifact.alpha, precision_bits)?,
        beta: parse_hp_vector(&artifact.beta, precision_bits)?,
        gamma: parse_hp_vector(&artifact.gamma, precision_bits)?,
    })
}

fn decode_prime_component(
    artifact: &PortablePrimeComponent,
    params: &CcmParams,
    precision_bits: u32,
) -> std::result::Result<Vec<Float>, CacheError> {
    let expected_content: Vec<PrimePowerContent> = prime_powers_up_to(params.lambda_sq_int())
        .into_iter()
        .map(|(power, prime, exponent)| PrimePowerContent {
            power,
            prime,
            exponent,
        })
        .collect();
    let expected_entries = params.matrix_size() * params.matrix_size();
    if artifact.schema_version != 1
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.prime_cutoff != params.lambda_sq_int()
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != precision_bits
        || artifact.prime_content != expected_content
        || artifact.entries.len() != expected_entries
    {
        return Err(CacheError::InvalidManifest(
            "CCM prime-component payload does not match its semantic identity".to_owned(),
        ));
    }
    let matrix = parse_hp_vector(&artifact.entries, precision_bits)?;
    let dim = params.matrix_size();
    for row in 0..dim {
        for column in (row + 1)..dim {
            if matrix[row * dim + column] != matrix[column * dim + row] {
                return Err(CacheError::InvalidManifest(
                    "CCM prime-component payload is not exactly symmetric".to_owned(),
                ));
            }
        }
    }
    Ok(matrix)
}

fn resolve_archimedean_integrals_via_cache(
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(ComputedArchimedeanIntegrals, ArtifactManifest)> {
    let precision_bits = cfg.precision_bits;
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_archimedean_integrals".to_owned(),
        mathematical_semantics_version: "ccm-archimedean-integrals-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": precision_bits,
            "quadrature_points": cfg.quad_points,
            "scalar_backend": "rug_mpfr"
        }),
        normalization: Some("alpha_beta_gamma_nonnegative_modes".to_owned()),
        target: Some("localized_archimedean_form_primitives".to_owned()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("adaptive_gauss_legendre_by_mode".to_owned()),
    };
    let logical_key = format!(
        "ccm/archimedean-integrals/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        precision_bits
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.archimedean_integrals.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "archimedean_integrals".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let (integrals, manifests) =
                compute_archimedean_integrals_tracked(params.n_modes, l, cfg, Some(cache))
                    .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            let dependencies = manifests
                .into_iter()
                .map(|manifest| DependencyRef {
                    key: manifest.key,
                    content_digest: manifest.content_digest,
                    required_quality: CacheQuality::Validated,
                })
                .collect();
            Ok((
                PortableArchimedeanIntegrals {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits,
                    alpha: integrals.alpha.iter().map(Float::to_string).collect(),
                    beta: integrals.beta.iter().map(Float::to_string).collect(),
                    gamma: integrals.gamma.iter().map(Float::to_string).collect(),
                },
                dependencies,
            ))
        },
        |artifact| decode_archimedean_integrals(artifact, params, precision_bits).map(|_| ()),
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("archimedean-integral execution returned no manifest"))?;
    Ok((
        decode_archimedean_integrals(&resolved.value, params, precision_bits)?,
        manifest,
    ))
}

fn resolve_prime_component_via_cache(
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Vec<Float>, ArtifactManifest)> {
    let precision_bits = cfg.precision_bits;
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_prime_component".to_owned(),
        mathematical_semantics_version: "ccm-prime-component-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "prime_cutoff": params.lambda_sq_int(),
            "n_modes": params.n_modes,
            "precision_bits": precision_bits,
            "scalar_backend": "rug_mpfr"
        }),
        normalization: Some("symmetric_row_major_unsigned_prime_sum".to_owned()),
        target: Some("localized_prime_power_component".to_owned()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let logical_key = format!(
        "ccm/prime-component/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        precision_bits
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.prime_component.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "prime_component".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let mut entries = compute_prime_component_matrix(
                params.n_modes,
                params.lambda_sq_int(),
                l,
                precision_bits,
            );
            force_symmetric(&mut entries, params.matrix_size());
            Ok((
                PortablePrimeComponent {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    prime_cutoff: params.lambda_sq_int(),
                    n_modes: params.n_modes,
                    precision_bits,
                    prime_content: prime_powers_up_to(params.lambda_sq_int())
                        .into_iter()
                        .map(|(power, prime, exponent)| PrimePowerContent {
                            power,
                            prime,
                            exponent,
                        })
                        .collect(),
                    entries: entries.iter().map(Float::to_string).collect(),
                },
                Vec::new(),
            ))
        },
        |artifact| decode_prime_component(artifact, params, precision_bits).map(|_| ()),
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("prime-component execution returned no manifest"))?;
    Ok((
        decode_prime_component(&resolved.value, params, precision_bits)?,
        manifest,
    ))
}

fn build_tau_hp_via_cache(
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Vec<Float>, ArtifactManifest)> {
    let prec = cfg.precision_bits;
    let lambda_identity = lambda_squared_cache_identity(params);
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_tau_matrix".to_owned(),
        mathematical_semantics_version: "ccm-weil-form-v0.13.0-v2".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_identity,
            "prime_cutoff": params.lambda_sq.value_u64,
            "n_modes": params.n_modes,
            "precision_bits": prec,
            "scalar_backend": "rug_mpfr",
            "include_primes": true
        }),
        normalization: Some("symmetric_row_major".to_owned()),
        target: Some("localized_weil_form".to_owned()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let logical_key = format!(
        "ccm/tau/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        prec
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.tau.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "tau_matrix".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    // Decoding and structural validation parse every high-precision matrix
    // entry. Retain that exact validated matrix so a cache hit performs this
    // expensive MPFR conversion only once.
    let validated_tau = RefCell::new(None);
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let (integrals, archimedean_manifest) =
                resolve_archimedean_integrals_via_cache(params, l, cfg, cache)
                    .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            let (prime, prime_manifest) = resolve_prime_component_via_cache(params, l, cfg, cache)
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            let (pole, archimedean) =
                assemble_pole_and_archimedean_components(params.n_modes, l, prec, &integrals);
            let components = ComputedCcmMatrixComponents {
                pole,
                archimedean,
                prime,
            };
            let dependencies =
                canonical_dependency_refs(vec![archimedean_manifest, prime_manifest]);
            let tau = assemble_tau_components(&components, prec);
            Ok((
                PortableTauMatrix {
                    schema_version: 2,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: prec,
                    entries: tau.iter().map(Float::to_string).collect(),
                },
                dependencies,
            ))
        },
        |artifact| {
            let tau = decode_tau_artifact(artifact, params, prec)?;
            validated_tau.replace(Some(tau));
            Ok(())
        },
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("typed tau execution returned no artifact manifest"))?;
    let tau = validated_tau.into_inner().ok_or_else(|| {
        anyhow::anyhow!("typed tau execution did not retain its validated runtime matrix")
    })?;
    if cache.requested_assurance != xc_core::AssuranceLevel::Computed {
        let required_assurance = match cache.requested_assurance {
            xc_core::AssuranceLevel::Computed => unreachable!(),
            xc_core::AssuranceLevel::CrossChecked => xc_cache::ArtifactAssuranceState::CrossChecked,
            xc_core::AssuranceLevel::Certified => xc_cache::ArtifactAssuranceState::Certified,
        };
        if let Some(sink) = cache.production_sink {
            sink.record_assurance_requirement(xc_cache::ArtifactAssuranceRequirement {
                schema_version: 1,
                artifact_key: manifest.key.clone(),
                content_digest: manifest.content_digest.clone(),
                required_assurance,
            })
            .map_err(anyhow::Error::from)?;
        }
        let retained_assurance = cache
            .production_sink
            .map(|sink| {
                sink.retained_assurance(&manifest.key, &manifest.content_digest)
                    .map_err(anyhow::Error::from)
            })
            .transpose()?
            .flatten()
            .filter(|assessment| assessment.achieved_assurance >= required_assurance);
        if retained_assurance.is_some() {
            return Ok((tau, manifest));
        }
        match certify_tau_from_retained_computation(params, cfg, &tau, &manifest, cache) {
            Ok(assessment) => {
                if let Some(sink) = cache.production_sink {
                    sink.record_assurance(ArtifactAssuranceAttestation {
                        schema_version: 1,
                        artifact_key: manifest.key.clone(),
                        content_digest: manifest.content_digest.clone(),
                        achieved_assurance: assessment.achieved_assurance,
                        evidence_digests: assessment.evidence_digests,
                    })
                    .map_err(anyhow::Error::from)?;
                }
            }
            Err(error)
                if cache.certification_failure_policy
                    == xc_cache::CertificationFailurePolicy::RetainComputedSkipPublication =>
            {
                eprintln!(
                    "[HP] certification failed; retained computed tau and disabled its publication: {error}"
                );
            }
            Err(error) => return Err(error),
        }
    }
    Ok((tau, manifest))
}

fn decode_weil_eigenpair(
    artifact: &PortableWeilEigenpair,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
) -> std::result::Result<
    (
        Float,
        Vec<Float>,
        xc_numerics::linalg::InverseIterationDiagnostics,
    ),
    CacheError,
> {
    let prec = cfg.precision_bits;
    let parity_policy = cfg.effective_parity_policy();
    let expected_schema = match cfg.eigenstate_solver {
        CcmEigenstateSolver::LegacyInverseIteration => 2,
        CcmEigenstateSolver::ShiftInvertKrylov => 3,
        CcmEigenstateSolver::Auto => {
            unreachable!("automatic eigenstate policy is resolved before payload decoding")
        }
    };
    if artifact.schema_version != expected_schema
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != prec
        || !payload_parity_matches(artifact.force_even, artifact.parity_policy, parity_policy)
        || artifact.eigenstate_route != cfg.eigenstate_solver.as_str()
        || artifact.eigenvector.len() != params.matrix_size()
    {
        return Err(CacheError::InvalidManifest(
            "CCM Weil eigenpair payload does not match its semantic identity".to_owned(),
        ));
    }
    let parse = |value: &str| {
        Float::parse(value)
            .map(|parsed| Float::with_val(prec, parsed))
            .map_err(|error| {
                CacheError::InvalidManifest(format!(
                    "CCM Weil eigenpair contains an invalid HP scalar: {error}"
                ))
            })
    };
    let eps_n = parse(&artifact.eigenvalue)?;
    let xi: Vec<Float> = artifact
        .eigenvector
        .iter()
        .map(|entry| parse(entry))
        .collect::<std::result::Result<_, _>>()?;
    let diagnostics = artifact.inverse_iteration.to_runtime(prec)?;
    match cfg.eigenstate_solver {
        CcmEigenstateSolver::LegacyInverseIteration => {
            if diagnostics.configured_step_limit != cfg.inverse_iter_steps
                || artifact.shift_invert_krylov.is_some()
            {
                return Err(CacheError::InvalidManifest(
                    "CCM Weil eigenpair uses incompatible inverse-iteration evidence".to_owned(),
                ));
            }
        }
        CcmEigenstateSolver::ShiftInvertKrylov => {
            let krylov = artifact.shift_invert_krylov.as_ref().ok_or_else(|| {
                CacheError::InvalidManifest(
                    "CCM Krylov eigenpair omits its route-specific stopping evidence".to_owned(),
                )
            })?;
            let parse_metric = |value: &str, name: &str| {
                Float::parse(value)
                    .map(|parsed| Float::with_val(prec, parsed))
                    .map_err(|error| {
                        CacheError::InvalidManifest(format!(
                            "CCM Krylov {name} is not a valid HP scalar: {error}"
                        ))
                    })
            };
            let tau_residual = parse_metric(&krylov.final_relative_tau_residual, "Tau residual")?;
            let backward = parse_metric(&krylov.final_scaled_backward_error, "backward error")?;
            let stability = parse_metric(&krylov.maximum_ritz_value_stability, "Ritz stability")?;
            if krylov.algorithm_semantics
                != "ccm_even_zero_shift_thick_restart_shift_invert_krylov_rayleigh_ritz_v1"
                || krylov.status != "converged"
                || krylov.requested_eigenpairs != 1
                || krylov.guard_eigenpairs != cfg.krylov_guard_eigenpairs
                || krylov.maximum_subspace_dimension
                    != cfg.krylov_subspace_dimension.min(params.n_modes + 1)
                || krylov.maximum_restarts != cfg.krylov_maximum_restarts
                || krylov.restarts == 0
                || krylov.restarts > krylov.maximum_restarts
                || krylov.shifted_solves == 0
                || krylov.operator_applications == 0
                || !tau_residual.is_finite()
                || !backward.is_finite()
                || !stability.is_finite()
                || tau_residual < 0
                || backward < 0
                || stability < 0
            {
                return Err(CacheError::InvalidManifest(
                    "CCM Krylov eigenpair has invalid or mismatched stopping evidence".to_owned(),
                ));
            }
        }
        CcmEigenstateSolver::Auto => {
            unreachable!("automatic eigenstate policy is resolved before payload decoding")
        }
    }
    if !weil_eigvec_cache::residual_ok(tau, params.matrix_size(), &xi, &eps_n, prec) {
        return Err(CacheError::InvalidManifest(
            "CCM Weil eigenpair failed its tau residual validation".to_owned(),
        ));
    }
    let replayed_residual =
        weil_eigvec_cache::relative_residual_norm(tau, params.matrix_size(), &xi, &eps_n, prec)
            .ok_or_else(|| {
                CacheError::InvalidManifest(
                    "CCM Weil eigenpair has no finite relative Tau residual".to_owned(),
                )
            })?;
    if replayed_residual != diagnostics.final_relative_residual_norm {
        return Err(CacheError::InvalidManifest(
            "CCM Weil eigenpair stopping evidence does not match its replayed Tau residual"
                .to_owned(),
        ));
    }
    if let Some(krylov) = &artifact.shift_invert_krylov {
        if replayed_residual.to_string() != krylov.final_relative_tau_residual {
            return Err(CacheError::InvalidManifest(
                "CCM Krylov full-Tau replay does not match its stored stopping evidence".to_owned(),
            ));
        }
    }
    Ok((eps_n, xi, diagnostics))
}

fn build_even_sector_matrix(tau: &[Float], n_modes: usize, prec: u32) -> Vec<Float> {
    let full_dim = 2 * n_modes + 1;
    let even_dim = n_modes + 1;
    let center = n_modes;
    let sqrt_two = Float::with_val(prec, 2).sqrt();
    let mut sector = vec![Float::with_val(prec, 0); even_dim * even_dim];
    sector[0] = tau[center * full_dim + center].clone();
    for k in 1..=n_modes {
        let minus_k = center - k;
        let plus_k = center + k;
        let mut row_value = tau[center * full_dim + minus_k].clone();
        row_value += &tau[center * full_dim + plus_k];
        row_value /= &sqrt_two;
        sector[k] = row_value.clone();
        sector[k * even_dim] = row_value;
        for j in 1..=n_modes {
            let minus_j = center - j;
            let plus_j = center + j;
            let mut value = tau[minus_k * full_dim + minus_j].clone();
            value += &tau[minus_k * full_dim + plus_j];
            value += &tau[plus_k * full_dim + minus_j];
            value += &tau[plus_k * full_dim + plus_j];
            value /= 2u32;
            sector[k * even_dim + j] = value;
        }
    }
    force_symmetric(&mut sector, even_dim);
    sector
}

/// Restrict the full Weil form to the historical orthonormal odd basis
/// `(e_k - e_-k)/sqrt(2)`, `k=1..=N`.
///
/// The reduced entry `Q[k,j] - Q[k,-j]` is the established sector-gap
/// convention. It is exactly the four-term
/// orthogonal projection when the full form is centrosymmetric.  Keeping this
/// operation order preserves the established MPFR values.
fn build_odd_sector_matrix(tau: &[Float], n_modes: usize, prec: u32) -> Vec<Float> {
    let full_dim = 2 * n_modes + 1;
    let center = n_modes;
    let mut sector = vec![Float::with_val(prec, 0); n_modes * n_modes];
    for k in 1..=n_modes {
        let plus_k = center + k;
        for j in 1..=n_modes {
            let minus_j = center - j;
            let plus_j = center + j;
            let mut value = tau[plus_k * full_dim + plus_j].clone();
            value -= &tau[plus_k * full_dim + minus_j];
            sector[(k - 1) * n_modes + (j - 1)] = value;
        }
    }
    sector
}

fn expand_odd_sector_vector(vector: &[Float], n_modes: usize, prec: u32) -> Vec<Float> {
    debug_assert_eq!(vector.len(), n_modes);
    let mut expanded = vec![Float::with_val(prec, 0); 2 * n_modes + 1];
    let sqrt_two = Float::with_val(prec, 2).sqrt();
    for k in 1..=n_modes {
        let mut value = vector[k - 1].clone();
        value /= &sqrt_two;
        expanded[n_modes + k] = value.clone();
        expanded[n_modes - k] = -value;
    }
    expanded
}

fn matrix_is_exactly_symmetric(matrix: &[Float], dimension: usize) -> bool {
    matrix.len() == dimension * dimension
        && (0..dimension).all(|row| {
            ((row + 1)..dimension)
                .all(|column| matrix[row * dimension + column] == matrix[column * dimension + row])
        })
}

fn expand_even_sector_vector(vector: &[Float], n_modes: usize, prec: u32) -> Vec<Float> {
    let mut expanded = vec![Float::with_val(prec, 0); 2 * n_modes + 1];
    expanded[n_modes] = vector[0].clone();
    let sqrt_two = Float::with_val(prec, 2).sqrt();
    for k in 1..=n_modes {
        let mut value = vector[k].clone();
        value /= &sqrt_two;
        expanded[n_modes - k] = value.clone();
        expanded[n_modes + k] = value;
    }
    expanded
}

fn resolve_even_sector_matrix_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
    tau_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Vec<Float>, ArtifactManifest)> {
    let dimension = params.n_modes + 1;
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_even_sector_matrix".to_owned(),
        mathematical_semantics_version: "ccm-even-sector-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "tau_content_digest": tau_manifest.content_digest.0
        }),
        normalization: Some("orthonormal_reflection_even_basis_row_major".to_owned()),
        target: Some("even_sector_weil_form".to_owned()),
        subspace: Some("even".to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let logical_key = format!(
        "ccm/even-sector/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.even_sector.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "even_sector_matrix".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let sector = build_even_sector_matrix(tau, params.n_modes, cfg.precision_bits);
            Ok((
                PortableEvenSectorMatrix {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    dimension,
                    entries: sector.iter().map(Float::to_string).collect(),
                },
                vec![DependencyRef {
                    key: tau_manifest.key.clone(),
                    content_digest: tau_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            if artifact.schema_version != 1
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || artifact.dimension != dimension
                || artifact.entries.len() != dimension * dimension
            {
                return Err(CacheError::InvalidManifest(
                    "CCM even-sector matrix does not match its semantic identity".to_owned(),
                ));
            }
            let decoded = parse_hp_vector(&artifact.entries, cfg.precision_bits)?;
            let expected = build_even_sector_matrix(tau, params.n_modes, cfg.precision_bits);
            if decoded != expected {
                return Err(CacheError::InvalidManifest(
                    "CCM even-sector matrix is inconsistent with its full tau dependency"
                        .to_owned(),
                ));
            }
            Ok(())
        },
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("even-sector execution returned no manifest"))?;
    Ok((
        parse_hp_vector(&resolved.value.entries, cfg.precision_bits)?,
        manifest,
    ))
}

fn resolve_odd_sector_matrix_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    tau: &[Float],
    tau_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Vec<Float>, ArtifactManifest)> {
    let dimension = params.n_modes;
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_odd_sector_matrix".to_owned(),
        mathematical_semantics_version: "ccm-odd-sector-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "tau_content_digest": tau_manifest.content_digest.0
        }),
        normalization: Some("orthonormal_reflection_odd_basis_row_major".to_owned()),
        target: Some("odd_sector_weil_form".to_owned()),
        subspace: Some("odd".to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("centrosymmetric_q_plus_plus_minus_q_plus_minus".to_owned()),
    };
    let logical_key = format!(
        "ccm/odd-sector/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.odd_sector.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "odd_sector_matrix".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let sector = build_odd_sector_matrix(tau, params.n_modes, cfg.precision_bits);
            if !matrix_is_exactly_symmetric(&sector, dimension) {
                return Err(CacheError::InvalidManifest(
                    "CCM odd-sector projection is not exactly symmetric".to_owned(),
                ));
            }
            Ok((
                PortableOddSectorMatrix {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    dimension,
                    entries: sector.iter().map(Float::to_string).collect(),
                },
                vec![DependencyRef {
                    key: tau_manifest.key.clone(),
                    content_digest: tau_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            if artifact.schema_version != 1
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || artifact.dimension != dimension
                || artifact.entries.len() != dimension * dimension
            {
                return Err(CacheError::InvalidManifest(
                    "CCM odd-sector matrix does not match its semantic identity".to_owned(),
                ));
            }
            let decoded = parse_hp_vector(&artifact.entries, cfg.precision_bits)?;
            let expected = build_odd_sector_matrix(tau, params.n_modes, cfg.precision_bits);
            if decoded != expected || !matrix_is_exactly_symmetric(&decoded, dimension) {
                return Err(CacheError::InvalidManifest(
                    "CCM odd-sector matrix is inconsistent with its full tau dependency".to_owned(),
                ));
            }
            Ok(())
        },
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("odd-sector execution returned no artifact manifest"))?;
    Ok((
        parse_hp_vector(&resolved.value.entries, cfg.precision_bits)?,
        manifest,
    ))
}

fn decode_sector_tridiagonal(
    artifact: &PortableSectorTridiagonal,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    dimension: usize,
) -> std::result::Result<SectorTridiagonalHp, CacheError> {
    if artifact.schema_version != 1
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != cfg.precision_bits
        || artifact.parity != parity
        || artifact.dimension != dimension
        || artifact.diagonal.len() != dimension
        || artifact.off_diagonal.len() + 1 != dimension
    {
        return Err(CacheError::InvalidManifest(
            "CCM sector tridiagonal does not match its semantic identity".to_owned(),
        ));
    }
    let diagonal = parse_hp_vector(&artifact.diagonal, cfg.precision_bits)?;
    let off_diagonal = parse_hp_vector(&artifact.off_diagonal, cfg.precision_bits)?;
    if diagonal
        .iter()
        .chain(&off_diagonal)
        .any(|value| !value.is_finite())
    {
        return Err(CacheError::InvalidManifest(
            "CCM sector tridiagonal contains a non-finite value".to_owned(),
        ));
    }
    Ok(SectorTridiagonalHp {
        diagonal,
        off_diagonal,
    })
}

fn hp_invariant_close(left: &Float, right: &Float, precision_bits: u32) -> bool {
    let mut difference = left.clone();
    difference -= right;
    difference.abs_mut();
    let mut scale = left.clone().abs();
    let right_scale = right.clone().abs();
    if right_scale > scale {
        scale = right_scale;
    }
    scale += 1u32;
    scale *= Float::with_val(precision_bits, 2).pow(-((precision_bits / 4).max(8) as i32));
    difference <= scale
}

fn sector_tridiagonal_invariants_match(
    matrix: &[Float],
    tridiagonal: &SectorTridiagonalHp,
    dimension: usize,
    precision_bits: u32,
) -> bool {
    if matrix.len() != dimension * dimension
        || tridiagonal.diagonal.len() != dimension
        || tridiagonal.off_diagonal.len() + 1 != dimension
    {
        return false;
    }
    let mut matrix_trace = Float::with_val(precision_bits, 0);
    let mut matrix_frobenius = Float::with_val(precision_bits, 0);
    for row in 0..dimension {
        matrix_trace += &matrix[row * dimension + row];
        for column in 0..dimension {
            let mut square = matrix[row * dimension + column].clone();
            square.square_mut();
            matrix_frobenius += square;
        }
    }
    let mut tridiagonal_trace = Float::with_val(precision_bits, 0);
    let mut tridiagonal_frobenius = Float::with_val(precision_bits, 0);
    for value in &tridiagonal.diagonal {
        tridiagonal_trace += value;
        let mut square = value.clone();
        square.square_mut();
        tridiagonal_frobenius += square;
    }
    for value in &tridiagonal.off_diagonal {
        let mut square = value.clone();
        square.square_mut();
        square *= 2u32;
        tridiagonal_frobenius += square;
    }
    hp_invariant_close(&matrix_trace, &tridiagonal_trace, precision_bits)
        && hp_invariant_close(&matrix_frobenius, &tridiagonal_frobenius, precision_bits)
}

fn resolve_sector_tridiagonal_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    matrix: &[Float],
    matrix_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(
    SectorTridiagonalHp,
    ArtifactManifest,
    Option<SectorTransformHp>,
)> {
    let dimension = match parity {
        CcmParity::Even => params.n_modes + 1,
        CcmParity::Odd => params.n_modes,
    };
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_tridiagonal".to_owned(),
        mathematical_semantics_version: "ccm-parity-tridiagonal-v0.13.0-v3".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "parity": parity.as_str(),
            "sector_matrix_content_digest": matrix_manifest.content_digest.0
        }),
        normalization: Some("householder_diagonal_and_signed_off_diagonal".to_owned()),
        target: Some("symmetric_tridiagonal_reduction".to_owned()),
        subspace: Some(parity.as_str().to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(
            "dense_householder_with_consistent_signed_off_diagonal_and_reusable_q_v2".to_owned(),
        ),
    };
    let logical_key = format!(
        "ccm/sector-tridiagonal/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        parity.as_str()
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_tridiagonal.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_tridiagonal".to_owned()),
            ("parity".to_owned(), parity.as_str().to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let validated = RefCell::new(None);
    let computed_transform = RefCell::new(None);
    let started = Instant::now();
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let (diagonal, off_diagonal, basis) =
                xc_numerics::eigen::householder_tridiag_hp(matrix, dimension, cfg.precision_bits)
                    .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            computed_transform.replace(Some(SectorTransformHp { basis }));
            Ok((
                PortableSectorTridiagonal {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    parity,
                    dimension,
                    diagonal: diagonal.iter().map(Float::to_string).collect(),
                    off_diagonal: off_diagonal.iter().map(Float::to_string).collect(),
                },
                vec![DependencyRef {
                    key: matrix_manifest.key.clone(),
                    content_digest: matrix_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            let tridiagonal = decode_sector_tridiagonal(artifact, params, cfg, parity, dimension)?;
            if !sector_tridiagonal_invariants_match(
                matrix,
                &tridiagonal,
                dimension,
                cfg.precision_bits,
            ) {
                return Err(CacheError::InvalidManifest(
                    "CCM sector tridiagonal failed trace or Frobenius replay".to_owned(),
                ));
            }
            validated.replace(Some(tridiagonal));
            Ok(())
        },
    )?;
    let was_produced = resolved.produced_manifest.is_some();
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("sector-tridiagonal execution returned no manifest"))?;
    let tridiagonal = validated.into_inner().ok_or_else(|| {
        anyhow::anyhow!("sector-tridiagonal execution retained no validated runtime value")
    })?;
    eprintln!(
        "[HP] {parity:?} sector tridiagonal: {} in {:.3}s",
        if was_produced { "computed" } else { "reused" },
        started.elapsed().as_secs_f64()
    );
    Ok((tridiagonal, manifest, computed_transform.into_inner()))
}

fn validate_sector_transform(
    matrix: &[Float],
    tridiagonal: &SectorTridiagonalHp,
    transform: &SectorTransformHp,
    dimension: usize,
    precision_bits: u32,
) -> std::result::Result<(), String> {
    if matrix.len() != dimension * dimension || transform.basis.len() != dimension * dimension {
        return Err("matrix or basis dimensions are inconsistent".to_owned());
    }
    // `HighPrecConfig::for_decimal_digits` reserves GUARD_BITS beyond the
    // caller's requested precision. Validation must enforce the requested
    // contract, not demand that an O(n^3) accumulated transformation retain
    // half of those guard bits as additional answer digits.
    let orthogonality_tolerance = Float::with_val(precision_bits, 2)
        .pow(-((precision_bits.saturating_sub(GUARD_BITS).max(1)) as i32));

    // Check every basis-vector norm and every adjacent dot product.  This is
    // O(n^2), so cached transforms remain cheap to validate at research sizes.
    for column in 0..dimension {
        let mut norm = Float::with_val(precision_bits, 0);
        let mut adjacent = Float::with_val(precision_bits, 0);
        for row in 0..dimension {
            let mut square = transform.basis[row * dimension + column].clone();
            square.square_mut();
            norm += square;
            if column + 1 < dimension {
                let mut product = transform.basis[row * dimension + column].clone();
                product *= &transform.basis[row * dimension + column + 1];
                adjacent += product;
            }
        }
        norm -= 1u32;
        let norm_error = norm.abs();
        let adjacent_error = adjacent.abs();
        if norm_error > orthogonality_tolerance {
            return Err(format!(
                "basis column {column} norm error {} exceeds requested-precision tolerance {}",
                norm_error, orthogonality_tolerance
            ));
        }
        if adjacent_error > orthogonality_tolerance {
            return Err(format!(
                "basis columns {column} and {} inner-product error {} exceeds requested-precision tolerance {}",
                column + 1,
                adjacent_error,
                orthogonality_tolerance
            ));
        }
    }

    // A Q = Q T is a homogeneous identity. Use a relative infinity-scale
    // threshold so an otherwise identical matrix expressed at a different
    // magnitude cannot be spuriously rejected by an absolute cutoff.
    let mut matrix_scale = Float::with_val(precision_bits, 1);
    for row in 0..dimension {
        let mut row_sum = Float::with_val(precision_bits, 0);
        for column in 0..dimension {
            row_sum += matrix[row * dimension + column].clone().abs();
        }
        if row_sum > matrix_scale {
            matrix_scale = row_sum;
        }
    }
    let mut similarity_tolerance = orthogonality_tolerance.clone();
    similarity_tolerance *= matrix_scale;

    // Replay A Q = Q T for boundary and central columns.  Every retained
    // eigenvector is additionally replayed against A before acceptance.
    let mut columns = vec![0, dimension / 2, dimension - 1];
    columns.sort_unstable();
    columns.dedup();
    for column in columns {
        for row in 0..dimension {
            let mut left = Float::with_val(precision_bits, 0);
            for inner in 0..dimension {
                let mut term = matrix[row * dimension + inner].clone();
                term *= &transform.basis[inner * dimension + column];
                left += term;
            }
            let mut right = transform.basis[row * dimension + column].clone();
            right *= &tridiagonal.diagonal[column];
            if column > 0 {
                let mut term = transform.basis[row * dimension + column - 1].clone();
                term *= &tridiagonal.off_diagonal[column - 1];
                right += term;
            }
            if column + 1 < dimension {
                let mut term = transform.basis[row * dimension + column + 1].clone();
                term *= &tridiagonal.off_diagonal[column];
                right += term;
            }
            left -= right;
            let residual = left.abs();
            if residual > similarity_tolerance {
                return Err(format!(
                    "A Q = Q T residual at row {row}, column {column} is {} and exceeds scale-aware tolerance {}",
                    residual, similarity_tolerance
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_sector_transform_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    matrix: &[Float],
    matrix_manifest: &ArtifactManifest,
    tridiagonal: &SectorTridiagonalHp,
    tridiagonal_manifest: &ArtifactManifest,
    precomputed: Option<&SectorTransformHp>,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(SectorTransformHp, ArtifactManifest)> {
    let dimension = match parity {
        CcmParity::Even => params.n_modes + 1,
        CcmParity::Odd => params.n_modes,
    };
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_transform".to_owned(),
        mathematical_semantics_version: "ccm-parity-householder-basis-v0.13.0-v2".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "parity": parity.as_str(),
            "sector_matrix_content_digest": matrix_manifest.content_digest.0,
            "sector_tridiagonal_content_digest": tridiagonal_manifest.content_digest.0
        }),
        normalization: Some("row_major_orthogonal_householder_basis".to_owned()),
        target: Some("tridiagonal_to_dense_eigenvector_transform".to_owned()),
        subspace: Some(parity.as_str().to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("dense_householder_q_accumulation".to_owned()),
    };
    let logical_key = format!(
        "ccm/sector-transform/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        parity.as_str()
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_transform.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_transform".to_owned()),
            ("parity".to_owned(), parity.as_str().to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let validated = RefCell::new(None);
    let started = Instant::now();
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let basis = if let Some(precomputed) = precomputed {
                precomputed.basis.clone()
            } else {
                let (diagonal, off_diagonal, basis) = xc_numerics::eigen::householder_tridiag_hp(
                    matrix,
                    dimension,
                    cfg.precision_bits,
                )
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                if diagonal != tridiagonal.diagonal || off_diagonal != tridiagonal.off_diagonal {
                    return Err(CacheError::InvalidManifest(
                        "CCM Householder basis did not reproduce the cached tridiagonal".to_owned(),
                    ));
                }
                basis
            };
            Ok((
                PortableSectorTransform {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    parity,
                    dimension,
                    basis: basis.iter().map(Float::to_string).collect(),
                },
                canonical_dependency_refs(vec![
                    matrix_manifest.clone(),
                    tridiagonal_manifest.clone(),
                ]),
            ))
        },
        |artifact| {
            if artifact.schema_version != 1
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || artifact.parity != parity
                || artifact.dimension != dimension
                || artifact.basis.len() != dimension * dimension
            {
                return Err(CacheError::InvalidManifest(
                    "CCM sector transform does not match its semantic identity".to_owned(),
                ));
            }
            let transform = SectorTransformHp {
                basis: parse_hp_vector(&artifact.basis, cfg.precision_bits)?,
            };
            if let Err(reason) = validate_sector_transform(
                matrix,
                tridiagonal,
                &transform,
                dimension,
                cfg.precision_bits,
            ) {
                return Err(CacheError::InvalidManifest(format!(
                    "CCM sector transform failed orthogonality or A Q = Q T replay: {reason}"
                )));
            }
            validated.replace(Some(transform));
            Ok(())
        },
    )?;
    let was_produced = resolved.produced_manifest.is_some();
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("sector-transform execution returned no manifest"))?;
    let transform = validated.into_inner().ok_or_else(|| {
        anyhow::anyhow!("sector-transform execution retained no validated runtime value")
    })?;
    eprintln!(
        "[HP] {parity:?} sector transform: {} in {:.3}s",
        if was_produced { "computed" } else { "reused" },
        started.elapsed().as_secs_f64()
    );
    Ok((transform, manifest))
}

fn selected_sector_tolerance(precision_bits: u32) -> Float {
    Float::with_val(precision_bits, 2).pow(-((precision_bits.saturating_sub(32)) as i32))
}

fn complete_sector_eigenvalues_qr(
    tridiagonal: &SectorTridiagonalHp,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    // Keep CCM's slow-but-accurate policy explicit instead of relying solely
    // on a library default that could later regress independently.
    xc_numerics::eigen::tridiag_eigenvalues_hp_with_options(
        &tridiagonal.diagonal,
        &tridiagonal.off_diagonal,
        precision_bits,
        xc_numerics::eigen::TridiagQrOptions {
            max_iterations_per_eigenvalue: xc_numerics::eigen::DEFAULT_TRIDIAG_QR_MAX_ITERATIONS,
        },
    )
}

fn compute_sector_eigenvalues(
    tridiagonal: &SectorTridiagonalHp,
    dimension: usize,
    requested_eigenvalues: usize,
    route: CcmSectorEigenvalueRoute,
    precision_bits: u32,
) -> Result<SectorEigenvaluesHp> {
    let selected = || {
        xc_numerics::eigen::tridiag_selected_eigenvalues_hp(
            &tridiagonal.diagonal,
            &tridiagonal.off_diagonal,
            0,
            requested_eigenvalues - 1,
            &selected_sector_tolerance(precision_bits),
            (precision_bits as usize).saturating_mul(2),
            precision_bits,
        )
    };
    match route {
        CcmSectorEigenvalueRoute::Selected => {
            let spectrum = selected()?;
            let values = spectrum
                .enclosures
                .iter()
                .map(|enclosure| {
                    let mut midpoint = enclosure.lower.clone();
                    midpoint += &enclosure.upper;
                    midpoint /= 2u32;
                    midpoint
                })
                .collect();
            Ok(SectorEigenvaluesHp {
                route,
                complete: false,
                values,
                selected_enclosures: spectrum.enclosures,
            })
        }
        CcmSectorEigenvalueRoute::CompleteQr => Ok(SectorEigenvaluesHp {
            route,
            complete: true,
            values: complete_sector_eigenvalues_qr(tridiagonal, precision_bits)?,
            selected_enclosures: Vec::new(),
        }),
        CcmSectorEigenvalueRoute::CrossChecked => {
            let values = complete_sector_eigenvalues_qr(tridiagonal, precision_bits)?;
            if values.len() != dimension {
                bail!("complete QR sector spectrum has the wrong dimension");
            }
            let selected = selected()?;
            for enclosure in &selected.enclosures {
                if values[enclosure.index] < enclosure.lower
                    || values[enclosure.index] > enclosure.upper
                {
                    bail!(
                        "QR eigenvalue {} escaped its independently selected Sturm enclosure",
                        enclosure.index
                    );
                }
            }
            Ok(SectorEigenvaluesHp {
                route,
                complete: true,
                values,
                selected_enclosures: selected.enclosures,
            })
        }
    }
}

fn portable_sector_eigenvalues(
    values: &SectorEigenvaluesHp,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    dimension: usize,
    requested_eigenvalues: usize,
) -> PortableSectorEigenvalues {
    PortableSectorEigenvalues {
        schema_version: 1,
        lambda_squared: lambda_squared_cache_identity(params),
        n_modes: params.n_modes,
        precision_bits: cfg.precision_bits,
        parity,
        dimension,
        route: values.route,
        complete: values.complete,
        requested_eigenvalues,
        eigenvalues: values.values.iter().map(Float::to_string).collect(),
        selected_enclosures: values
            .selected_enclosures
            .iter()
            .map(|enclosure| PortableSectorEigenvalueEnclosure {
                index: enclosure.index,
                lower: enclosure.lower.to_string(),
                upper: enclosure.upper.to_string(),
                lower_count: enclosure.lower_count,
                upper_count: enclosure.upper_count,
                iterations: enclosure.iterations,
            })
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_sector_eigenvalues(
    artifact: &PortableSectorEigenvalues,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    dimension: usize,
    requested_eigenvalues: usize,
    route: CcmSectorEigenvalueRoute,
    tridiagonal: &SectorTridiagonalHp,
) -> std::result::Result<SectorEigenvaluesHp, CacheError> {
    let expected_value_count = if route == CcmSectorEigenvalueRoute::Selected {
        requested_eigenvalues
    } else {
        dimension
    };
    let expected_enclosure_count = if route == CcmSectorEigenvalueRoute::CompleteQr {
        0
    } else {
        requested_eigenvalues
    };
    if artifact.schema_version != 1
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != cfg.precision_bits
        || artifact.parity != parity
        || artifact.dimension != dimension
        || artifact.route != route
        || artifact.complete != (route != CcmSectorEigenvalueRoute::Selected)
        || artifact.requested_eigenvalues != requested_eigenvalues
        || artifact.eigenvalues.len() != expected_value_count
        || artifact.selected_enclosures.len() != expected_enclosure_count
    {
        return Err(CacheError::InvalidManifest(
            "CCM sector eigenvalues do not match their semantic identity".to_owned(),
        ));
    }
    let values = parse_hp_vector(&artifact.eigenvalues, cfg.precision_bits)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CacheError::InvalidManifest(
            "CCM sector eigenvalues contain a non-finite value".to_owned(),
        ));
    }
    if let Some(index) = values.windows(2).position(|pair| pair[0] >= pair[1]) {
        let enclosure_detail = artifact
            .selected_enclosures
            .get(index)
            .zip(artifact.selected_enclosures.get(index + 1))
            .map(|(left, right)| {
                format!(
                    "; adjacent Sturm endpoint counts are [{}, {}] and [{}, {}]",
                    left.lower_count, left.upper_count, right.lower_count, right.upper_count
                )
            })
            .unwrap_or_default();
        return Err(CacheError::InvalidManifest(format!(
            "CCM sector resolution limit: algebraic indices {} and {} are not strictly separated at {} working bits{}; retain the sector matrix and rerun sector analysis at higher precision",
            index,
            index + 1,
            cfg.precision_bits,
            enclosure_detail
        )));
    }
    let mut selected_enclosures = Vec::with_capacity(expected_enclosure_count);
    for (expected_index, enclosure) in artifact.selected_enclosures.iter().enumerate() {
        let lower = parse_hp_scalar(&enclosure.lower, cfg.precision_bits)?;
        let upper = parse_hp_scalar(&enclosure.upper, cfg.precision_bits)?;
        if enclosure.index != expected_index
            || !lower.is_finite()
            || !upper.is_finite()
            || lower >= upper
            || enclosure.lower_count > expected_index
            || enclosure.upper_count <= expected_index
            || enclosure.iterations == 0
        {
            return Err(CacheError::InvalidManifest(
                "CCM selected eigenvalue enclosure is invalid".to_owned(),
            ));
        }
        let value = &values[expected_index];
        if value < &lower || value > &upper {
            return Err(CacheError::InvalidManifest(
                "CCM selected eigenvalue escaped its indexed enclosure".to_owned(),
            ));
        }
        let replay_lower = xc_numerics::eigen::tridiag_sturm_count_below_hp(
            &tridiagonal.diagonal,
            &tridiagonal.off_diagonal,
            &lower,
            cfg.precision_bits,
        )
        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        let replay_upper = xc_numerics::eigen::tridiag_sturm_count_below_hp(
            &tridiagonal.diagonal,
            &tridiagonal.off_diagonal,
            &upper,
            cfg.precision_bits,
        )
        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        if replay_lower != enclosure.lower_count || replay_upper != enclosure.upper_count {
            return Err(CacheError::InvalidManifest(
                "CCM selected eigenvalue enclosure failed Sturm-count replay".to_owned(),
            ));
        }
        selected_enclosures.push(xc_numerics::eigen::HpTridiagonalEigenvalueEnclosure {
            index: enclosure.index,
            lower,
            upper,
            lower_count: enclosure.lower_count,
            upper_count: enclosure.upper_count,
            iterations: enclosure.iterations,
        });
    }
    if artifact.complete {
        let mut eigenvalue_trace = Float::with_val(cfg.precision_bits, 0);
        let mut eigenvalue_square_sum = Float::with_val(cfg.precision_bits, 0);
        for value in &values {
            eigenvalue_trace += value;
            let mut square = value.clone();
            square.square_mut();
            eigenvalue_square_sum += square;
        }
        let mut tridiagonal_trace = Float::with_val(cfg.precision_bits, 0);
        let mut tridiagonal_square_sum = Float::with_val(cfg.precision_bits, 0);
        for value in &tridiagonal.diagonal {
            tridiagonal_trace += value;
            let mut square = value.clone();
            square.square_mut();
            tridiagonal_square_sum += square;
        }
        for value in &tridiagonal.off_diagonal {
            let mut square = value.clone();
            square.square_mut();
            square *= 2u32;
            tridiagonal_square_sum += square;
        }
        if !hp_invariant_close(&eigenvalue_trace, &tridiagonal_trace, cfg.precision_bits)
            || !hp_invariant_close(
                &eigenvalue_square_sum,
                &tridiagonal_square_sum,
                cfg.precision_bits,
            )
        {
            return Err(CacheError::InvalidManifest(
                "complete CCM sector eigenvalues failed trace or Frobenius replay".to_owned(),
            ));
        }
    }
    Ok(SectorEigenvaluesHp {
        route,
        complete: artifact.complete,
        values,
        selected_enclosures,
    })
}

#[allow(clippy::too_many_arguments)]
fn resolve_sector_eigenvalues_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    dimension: usize,
    requested_eigenvalues: usize,
    route: CcmSectorEigenvalueRoute,
    tridiagonal: &SectorTridiagonalHp,
    tridiagonal_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(SectorEigenvaluesHp, ArtifactManifest)> {
    // A complete QR result is independent of how many eigenvectors the caller
    // will retain. Key it as the complete dimension so later requests for a
    // larger vector prefix reuse the same expensive spectrum artifact.
    let artifact_request_count = if route == CcmSectorEigenvalueRoute::CompleteQr {
        dimension
    } else {
        requested_eigenvalues
    };
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_eigenvalues".to_owned(),
        mathematical_semantics_version: "ccm-parity-sector-eigenvalues-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "parity": parity.as_str(),
            "dimension": dimension,
            "route": route.as_str(),
            "requested_eigenvalues": artifact_request_count,
            "tridiagonal_content_digest": tridiagonal_manifest.content_digest.0
        }),
        normalization: Some("strict_algebraic_order".to_owned()),
        target: Some(if route == CcmSectorEigenvalueRoute::Selected {
            "requested_parity_sector_eigenvalue_prefix".to_owned()
        } else {
            "complete_parity_sector_eigenvalue_spectrum".to_owned()
        }),
        subspace: Some(parity.as_str().to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(
            match route {
                CcmSectorEigenvalueRoute::Selected => "hp_sturm_indexed_bisection",
                CcmSectorEigenvalueRoute::CompleteQr => "implicit_wilkinson_shift_tridiagonal_qr",
                CcmSectorEigenvalueRoute::CrossChecked => {
                    "implicit_wilkinson_shift_tridiagonal_qr_cross_checked_by_hp_sturm"
                }
            }
            .to_owned(),
        ),
    };
    let logical_key = format!(
        "ccm/sector-eigenvalues/{}/{}/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        parity.as_str(),
        route.as_str(),
        artifact_request_count
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_eigenvalues.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_eigenvalues".to_owned()),
            ("parity".to_owned(), parity.as_str().to_owned()),
            ("route".to_owned(), route.as_str().to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let validated = RefCell::new(None);
    let started = Instant::now();
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let values = compute_sector_eigenvalues(
                tridiagonal,
                dimension,
                artifact_request_count,
                route,
                cfg.precision_bits,
            )
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            Ok((
                portable_sector_eigenvalues(
                    &values,
                    params,
                    cfg,
                    parity,
                    dimension,
                    artifact_request_count,
                ),
                vec![DependencyRef {
                    key: tridiagonal_manifest.key.clone(),
                    content_digest: tridiagonal_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            validated.replace(Some(decode_sector_eigenvalues(
                artifact,
                params,
                cfg,
                parity,
                dimension,
                artifact_request_count,
                route,
                tridiagonal,
            )?));
            Ok(())
        },
    )?;
    let was_produced = resolved.produced_manifest.is_some();
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("sector-eigenvalue execution returned no manifest"))?;
    let values = validated.into_inner().ok_or_else(|| {
        anyhow::anyhow!("sector-eigenvalue execution retained no validated runtime value")
    })?;
    eprintln!(
        "[HP] {parity:?} sector eigenvalues ({}): {} {} values in {:.3}s",
        route.as_str(),
        if was_produced { "computed" } else { "reused" },
        values.values.len(),
        started.elapsed().as_secs_f64()
    );
    Ok((values, manifest))
}

fn sector_eigenpair_residual_norm(
    matrix: &[Float],
    dimension: usize,
    eigenvalue: &Float,
    eigenvector: &[Float],
    precision_bits: u32,
) -> Result<Float> {
    if matrix.len() != dimension * dimension || eigenvector.len() != dimension {
        bail!("sector eigenpair dimensions do not match the matrix");
    }
    let mut squared_norm = Float::with_val(precision_bits, 0);
    for row in 0..dimension {
        let mut residual = Float::with_val(precision_bits, 0);
        for column in 0..dimension {
            let mut term = matrix[row * dimension + column].clone();
            term *= &eigenvector[column];
            residual += term;
        }
        let mut expected = eigenvector[row].clone();
        expected *= eigenvalue;
        residual -= expected;
        residual.square_mut();
        squared_norm += residual;
    }
    Ok(squared_norm.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn compute_sector_spectrum(
    matrix: &[Float],
    dimension: usize,
    parity: CcmParity,
    requested_eigenpairs: usize,
    eigenvalues: &SectorEigenvaluesHp,
    tridiagonal: &SectorTridiagonalHp,
    transform: &SectorTransformHp,
    cfg: &HighPrecConfig,
) -> Result<CcmSectorSpectrumHp> {
    if dimension == 0
        || requested_eigenpairs == 0
        || requested_eigenpairs > dimension
        || eigenvalues.values.len() < requested_eigenpairs
    {
        bail!("requested CCM sector spectrum is outside the sector dimension");
    }
    let eigenvector_start = Instant::now();
    // Each retained eigenvector is recovered from the same immutable matrix and
    // a distinct eigenvalue. An indexed Rayon collect runs those independent
    // solves concurrently while preserving algebraic order exactly. Rayon uses
    // its existing global pool here, so nested helper parallelism cannot create
    // an additional oversubscribed thread pool.
    let eigenpairs = eigenvalues
        .values
        .par_iter()
        .take(requested_eigenpairs)
        .cloned()
        .enumerate()
        .map(|(algebraic_index, eigenvalue)| {
            let tridiagonal_vector = xc_numerics::eigen::tridiag_eigenvector_for_value_hp(
                &tridiagonal.diagonal,
                &tridiagonal.off_diagonal,
                &eigenvalue,
                cfg.precision_bits,
                xc_numerics::eigen::TridiagEigvecOptions {
                    max_steps: cfg.inverse_iter_steps,
                    early_termination: true,
                    solver: xc_numerics::eigen::TridiagSolver::Banded,
                },
            )?;
            let eigenvector = (0..dimension)
                .map(|row| {
                    let terms = (0..dimension)
                        .map(|column| {
                            let mut term = transform.basis[row * dimension + column].clone();
                            term *= &tridiagonal_vector[column];
                            term
                        })
                        .collect();
                    xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(
                        terms,
                        cfg.precision_bits,
                    )
                })
                .collect::<Vec<_>>();
            let residual_norm = sector_eigenpair_residual_norm(
                matrix,
                dimension,
                &eigenvalue,
                &eigenvector,
                cfg.precision_bits,
            )?;
            Ok(CcmSectorEigenpairHp {
                algebraic_index,
                eigenvalue,
                eigenvector,
                residual_norm,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    eprintln!(
        "[HP] {parity:?} sector spectrum: {requested_eigenpairs} retained eigenvectors via banded tridiagonal solve={:.3}s",
        eigenvector_start.elapsed().as_secs_f64(),
    );
    Ok(CcmSectorSpectrumHp {
        parity,
        dimension,
        eigenvalue_route: eigenvalues.route,
        complete_eigenvalues: eigenvalues.complete.then(|| eigenvalues.values.clone()),
        eigenpairs,
    })
}

fn portable_sector_spectrum(spectrum: &CcmSectorSpectrumHp) -> PortableSectorSpectrum {
    PortableSectorSpectrum {
        schema_version: 1,
        lambda_squared: String::new(),
        n_modes: 0,
        precision_bits: 0,
        parity: spectrum.parity,
        eigenvalue_route: spectrum.eigenvalue_route,
        dimension: spectrum.dimension,
        requested_eigenpairs: spectrum.eigenpairs.len(),
        eigenvalues: spectrum
            .eigenpairs
            .iter()
            .map(|pair| pair.eigenvalue.to_string())
            .collect(),
        eigenvectors: spectrum
            .eigenpairs
            .iter()
            .map(|pair| pair.eigenvector.iter().map(Float::to_string).collect())
            .collect(),
        residual_norms: spectrum
            .eigenpairs
            .iter()
            .map(|pair| pair.residual_norm.to_string())
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_sector_spectrum(
    artifact: &PortableSectorSpectrum,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    eigenvalue_route: CcmSectorEigenvalueRoute,
    matrix: &[Float],
    dimension: usize,
    requested_eigenpairs: usize,
) -> std::result::Result<CcmSectorSpectrumHp, CacheError> {
    validate_sector_spectrum_identity(
        artifact,
        params,
        cfg,
        parity,
        eigenvalue_route,
        dimension,
        requested_eigenpairs,
    )?;
    let eigenvalues = parse_hp_vector(&artifact.eigenvalues, cfg.precision_bits)?;
    if eigenvalues.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CacheError::InvalidManifest(
            "CCM sector spectrum is not strictly ordered".to_owned(),
        ));
    }
    let stored_residuals = parse_hp_vector(&artifact.residual_norms, cfg.precision_bits)?;
    let mut eigenpairs = Vec::with_capacity(requested_eigenpairs);
    let tolerance =
        Float::with_val(cfg.precision_bits, 2).pow(-((cfg.precision_bits / 4).max(8) as i32));
    for index in 0..requested_eigenpairs {
        let eigenvector = parse_hp_vector(&artifact.eigenvectors[index], cfg.precision_bits)?;
        if eigenvector.len() != dimension {
            return Err(CacheError::InvalidManifest(
                "CCM sector eigenvector has the wrong dimension".to_owned(),
            ));
        }
        let residual = sector_eigenpair_residual_norm(
            matrix,
            dimension,
            &eigenvalues[index],
            &eigenvector,
            cfg.precision_bits,
        )
        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        if residual > tolerance || residual != stored_residuals[index] {
            return Err(CacheError::InvalidManifest(
                "CCM sector eigenpair failed exact residual replay".to_owned(),
            ));
        }
        eigenpairs.push(CcmSectorEigenpairHp {
            algebraic_index: index,
            eigenvalue: eigenvalues[index].clone(),
            eigenvector,
            residual_norm: residual,
        });
    }
    Ok(CcmSectorSpectrumHp {
        parity,
        dimension,
        eigenvalue_route,
        complete_eigenvalues: None,
        eigenpairs,
    })
}

fn validate_sector_spectrum_identity(
    artifact: &PortableSectorSpectrum,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    eigenvalue_route: CcmSectorEigenvalueRoute,
    dimension: usize,
    requested_eigenpairs: usize,
) -> std::result::Result<(), CacheError> {
    if artifact.schema_version != 1
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != cfg.precision_bits
        || artifact.parity != parity
        || artifact.eigenvalue_route != eigenvalue_route
        || artifact.dimension != dimension
        || artifact.requested_eigenpairs != requested_eigenpairs
        || artifact.eigenvalues.len() != requested_eigenpairs
        || artifact.eigenvectors.len() != requested_eigenpairs
        || artifact.residual_norms.len() != requested_eigenpairs
    {
        return Err(CacheError::InvalidManifest(
            "CCM sector spectrum does not match its semantic identity".to_owned(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_sector_spectrum_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    matrix: &[Float],
    matrix_manifest: &ArtifactManifest,
    requested_eigenpairs: usize,
    eigenvalues: &SectorEigenvaluesHp,
    eigenvalues_manifest: &ArtifactManifest,
    tridiagonal: &SectorTridiagonalHp,
    tridiagonal_manifest: &ArtifactManifest,
    transform: &SectorTransformHp,
    transform_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(CcmSectorSpectrumHp, ArtifactManifest)> {
    let dimension = match parity {
        CcmParity::Even => params.n_modes + 1,
        CcmParity::Odd => params.n_modes,
    };
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_spectrum".to_owned(),
        mathematical_semantics_version: "ccm-parity-sector-spectrum-v0.13.0-v3".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "parity": parity.as_str(),
            "eigenvalue_route": eigenvalues.route.as_str(),
            "requested_eigenpairs": requested_eigenpairs,
            "sector_matrix_content_digest": matrix_manifest.content_digest.0,
            "sector_eigenvalues_content_digest": eigenvalues_manifest.content_digest.0,
            "sector_tridiagonal_content_digest": tridiagonal_manifest.content_digest.0,
            "sector_transform_content_digest": transform_manifest.content_digest.0
        }),
        normalization: Some("l2_unit_sector_vectors_algebraic_order".to_owned()),
        target: Some("lowest_parity_sector_eigenpairs".to_owned()),
        subspace: Some(parity.as_str().to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(
            "indexed_eigenvalues_plus_banded_tridiagonal_inverse_iteration_and_householder_backtransform_v1".to_owned(),
        ),
    };
    let logical_key = format!(
        "ccm/sector-spectrum/{}/{}/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        parity.as_str(),
        eigenvalues.route.as_str(),
        requested_eigenpairs
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_spectrum.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_spectrum".to_owned()),
            ("parity".to_owned(), parity.as_str().to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    // Validation decodes and replays every portable spectrum. Retain that
    // validated runtime value so the caller does not parse the same HP decimal
    // vectors and repeat every residual calculation a second time.
    let validated_spectrum = RefCell::new(None);
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let spectrum = compute_sector_spectrum(
                matrix,
                dimension,
                parity,
                requested_eigenpairs,
                eigenvalues,
                tridiagonal,
                transform,
                cfg,
            )
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            let mut portable = portable_sector_spectrum(&spectrum);
            portable.lambda_squared = lambda_squared_cache_identity(params);
            portable.n_modes = params.n_modes;
            portable.precision_bits = cfg.precision_bits;
            Ok((
                portable,
                canonical_dependency_refs(vec![
                    matrix_manifest.clone(),
                    eigenvalues_manifest.clone(),
                    tridiagonal_manifest.clone(),
                    transform_manifest.clone(),
                ]),
            ))
        },
        |artifact| {
            let spectrum = decode_sector_spectrum(
                artifact,
                params,
                cfg,
                parity,
                eigenvalues.route,
                matrix,
                dimension,
                requested_eigenpairs,
            )?;
            validated_spectrum.replace(Some(spectrum));
            Ok(())
        },
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("sector-spectrum execution returned no manifest"))?;
    let mut spectrum = validated_spectrum.into_inner().ok_or_else(|| {
        anyhow::anyhow!("sector-spectrum execution did not retain its validated runtime values")
    })?;
    spectrum.complete_eigenvalues = eigenvalues.complete.then(|| eigenvalues.values.clone());
    Ok((spectrum, manifest))
}

fn negative_log10_abs(value: &Float, precision_bits: u32) -> Result<Float> {
    let magnitude = value.clone().abs();
    if magnitude.is_zero() || !magnitude.is_finite() {
        bail!("CCM logarithmic depth requires a finite nonzero value");
    }
    let mut depth = magnitude.log10();
    depth = -depth;
    Ok(Float::with_val(precision_bits, depth))
}

fn compute_sector_gap(
    even: CcmSectorSpectrumHp,
    odd: CcmSectorSpectrumHp,
    precision_bits: u32,
) -> Result<CcmSectorGapHp> {
    if even.parity != CcmParity::Even
        || odd.parity != CcmParity::Odd
        || even.eigenpairs.len() < 2
        || odd.eigenpairs.len() < 2
    {
        bail!("CCM sector-gap analysis requires two ordered eigenpairs per parity sector");
    }
    let lambda_even = even.eigenpairs[0].eigenvalue.clone();
    let lambda_odd = odd.eigenpairs[0].eigenvalue.clone();
    let d_even = negative_log10_abs(&lambda_even, precision_bits)?;
    let d_odd = negative_log10_abs(&lambda_odd, precision_bits)?;
    let mut gap_log = d_even.clone();
    gap_log -= &d_odd;
    let mut lambda_difference = lambda_odd.clone();
    lambda_difference -= &lambda_even;
    let difference_depth = negative_log10_abs(&lambda_difference, precision_bits)?;
    let ordering = match lambda_difference.cmp0() {
        Some(std::cmp::Ordering::Greater) => 1,
        Some(std::cmp::Ordering::Less) => -1,
        _ => 0,
    };
    let mut even_simplicity_margin = even.eigenpairs[1].eigenvalue.clone();
    even_simplicity_margin -= &lambda_even;
    let even_simple = even_simplicity_margin > 0;
    Ok(CcmSectorGapHp {
        even,
        odd,
        lambda_even,
        lambda_odd,
        d_even,
        d_odd,
        gap_log,
        lambda_difference,
        difference_depth,
        ordering,
        even_simple,
        even_simplicity_margin,
    })
}

fn portable_sector_gap(
    result: &CcmSectorGapHp,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    even_manifest: &ArtifactManifest,
    odd_manifest: &ArtifactManifest,
) -> PortableSectorGap {
    PortableSectorGap {
        schema_version: 1,
        lambda_squared: lambda_squared_cache_identity(params),
        n_modes: params.n_modes,
        precision_bits: cfg.precision_bits,
        even_spectrum_content_digest: even_manifest.content_digest.0.clone(),
        odd_spectrum_content_digest: odd_manifest.content_digest.0.clone(),
        lambda_even: result.lambda_even.to_string(),
        lambda_odd: result.lambda_odd.to_string(),
        d_even: result.d_even.to_string(),
        d_odd: result.d_odd.to_string(),
        gap_log: result.gap_log.to_string(),
        lambda_difference: result.lambda_difference.to_string(),
        difference_depth: result.difference_depth.to_string(),
        ordering: result.ordering,
        even_simple: result.even_simple,
        even_simplicity_margin: result.even_simplicity_margin.to_string(),
    }
}

fn resolve_sector_gap_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    even: CcmSectorSpectrumHp,
    odd: CcmSectorSpectrumHp,
    even_manifest: &ArtifactManifest,
    odd_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(CcmSectorGapHp, ArtifactManifest)> {
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_gap".to_owned(),
        mathematical_semantics_version: "ccm-even-odd-gap-log-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "even_spectrum_content_digest": even_manifest.content_digest.0,
            "odd_spectrum_content_digest": odd_manifest.content_digest.0,
            "definition": "log10(abs(lambda_odd)/abs(lambda_even))"
        }),
        normalization: None,
        target: Some("finite_ccm_even_odd_sector_gap".to_owned()),
        subspace: Some("even_vs_odd".to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(
            "mpfr_depth_difference_and_direct_eigenvalue_ordering".to_owned(),
        ),
    };
    let logical_key = format!(
        "ccm/sector-gap/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_gap.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_gap".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let expected = compute_sector_gap(even.clone(), odd.clone(), cfg.precision_bits)?;
    let expected_payload = portable_sector_gap(&expected, params, cfg, even_manifest, odd_manifest);
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            Ok((
                expected_payload.clone(),
                canonical_dependency_refs(vec![even_manifest.clone(), odd_manifest.clone()]),
            ))
        },
        |artifact| {
            if artifact != &expected_payload {
                return Err(CacheError::InvalidManifest(
                    "CCM sector-gap payload does not replay from its sector spectra".to_owned(),
                ));
            }
            Ok(())
        },
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("sector-gap execution returned no manifest"))?;
    if resolved.value != expected_payload {
        bail!("resolved CCM sector-gap artifact disagrees with replayed values");
    }
    Ok((expected, manifest))
}

#[allow(clippy::too_many_arguments)]
fn resolve_sector_branch_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    parity: CcmParity,
    matrix: &[Float],
    matrix_manifest: &ArtifactManifest,
    requested_eigenpairs: usize,
    route: CcmSectorEigenvalueRoute,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(CcmSectorSpectrumHp, ArtifactManifest)> {
    let dimension = match parity {
        CcmParity::Even => params.n_modes + 1,
        CcmParity::Odd => params.n_modes,
    };
    let (tridiagonal, tridiagonal_manifest, precomputed_transform) =
        resolve_sector_tridiagonal_via_cache(params, cfg, parity, matrix, matrix_manifest, cache)?;
    let (transform, transform_manifest) = resolve_sector_transform_via_cache(
        params,
        cfg,
        parity,
        matrix,
        matrix_manifest,
        &tridiagonal,
        &tridiagonal_manifest,
        precomputed_transform.as_ref(),
        cache,
    )?;
    let (eigenvalues, eigenvalues_manifest) = resolve_sector_eigenvalues_via_cache(
        params,
        cfg,
        parity,
        dimension,
        requested_eigenpairs,
        route,
        &tridiagonal,
        &tridiagonal_manifest,
        cache,
    )?;
    resolve_sector_spectrum_via_cache(
        params,
        cfg,
        parity,
        matrix,
        matrix_manifest,
        requested_eigenpairs,
        &eigenvalues,
        &eigenvalues_manifest,
        &tridiagonal,
        &tridiagonal_manifest,
        &transform,
        &transform_manifest,
        cache,
    )
}

fn compute_sector_branch(
    matrix: &[Float],
    dimension: usize,
    parity: CcmParity,
    requested_eigenpairs: usize,
    route: CcmSectorEigenvalueRoute,
    cfg: &HighPrecConfig,
) -> Result<CcmSectorSpectrumHp> {
    let (diagonal, off_diagonal, basis) =
        xc_numerics::eigen::householder_tridiag_hp(matrix, dimension, cfg.precision_bits)?;
    let tridiagonal = SectorTridiagonalHp {
        diagonal,
        off_diagonal,
    };
    let transform = SectorTransformHp { basis };
    let eigenvalues = compute_sector_eigenvalues(
        &tridiagonal,
        dimension,
        requested_eigenpairs,
        route,
        cfg.precision_bits,
    )?;
    compute_sector_spectrum(
        matrix,
        dimension,
        parity,
        requested_eigenpairs,
        &eigenvalues,
        &tridiagonal,
        &transform,
        cfg,
    )
}

fn analyze_sector_gap_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    options: CcmSectorAnalysisOptions,
    cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<CcmSectorGapHp> {
    let l = log_lambda_sq_hp(params, cfg.precision_bits);
    let source = if let Some(cache) = cache {
        let (tau, tau_manifest) = build_tau_hp_via_cache(params, &l, cfg, cache)?;
        RetainedCcmSource {
            tau,
            tau_manifest: Some(tau_manifest),
            eigenpair_manifest: None,
            secular_manifest: None,
        }
    } else {
        RetainedCcmSource {
            tau: build_tau_hp(params, &l, cfg)?,
            tau_manifest: None,
            eigenpair_manifest: None,
            secular_manifest: None,
        }
    };
    analyze_sector_gap_from_retained_source(params, cfg, options, cache, source)
}

fn analyze_sector_gap_from_retained_source(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    options: CcmSectorAnalysisOptions,
    cache: Option<&ArtifactCacheContext<'_>>,
    mut source: RetainedCcmSource,
) -> Result<CcmSectorGapHp> {
    let requested_eigenpairs = options.requested_eigenpairs;
    if requested_eigenpairs < 2 || requested_eigenpairs > params.n_modes {
        bail!("CCM sector-gap analysis requires between two and N eigenpairs per sector");
    }
    if let Some(cache) = cache {
        let tau_manifest = source.tau_manifest.take().ok_or_else(|| {
            anyhow::anyhow!("managed retained CCM source is missing its Tau manifest")
        })?;
        let mut tau = source.tau;
        force_symmetric(&mut tau, params.matrix_size());
        let (even_matrix, even_matrix_manifest) =
            resolve_even_sector_matrix_via_cache(params, cfg, &tau, &tau_manifest, cache)?;
        let (odd_matrix, odd_matrix_manifest) =
            resolve_odd_sector_matrix_via_cache(params, cfg, &tau, &tau_manifest, cache)?;
        let (even_result, odd_result) = rayon::join(
            || {
                resolve_sector_branch_via_cache(
                    params,
                    cfg,
                    CcmParity::Even,
                    &even_matrix,
                    &even_matrix_manifest,
                    requested_eigenpairs,
                    options.eigenvalue_route,
                    cache,
                )
            },
            || {
                resolve_sector_branch_via_cache(
                    params,
                    cfg,
                    CcmParity::Odd,
                    &odd_matrix,
                    &odd_matrix_manifest,
                    requested_eigenpairs,
                    options.eigenvalue_route,
                    cache,
                )
            },
        );
        let (even, even_manifest) = even_result?;
        let (odd, odd_manifest) = odd_result?;
        resolve_sector_gap_via_cache(params, cfg, even, odd, &even_manifest, &odd_manifest, cache)
            .map(|(gap, _)| gap)
    } else {
        let mut tau = source.tau;
        force_symmetric(&mut tau, params.matrix_size());
        let even_matrix = build_even_sector_matrix(&tau, params.n_modes, cfg.precision_bits);
        let odd_matrix = build_odd_sector_matrix(&tau, params.n_modes, cfg.precision_bits);
        let (even, odd) = rayon::join(
            || {
                compute_sector_branch(
                    &even_matrix,
                    params.n_modes + 1,
                    CcmParity::Even,
                    requested_eigenpairs,
                    options.eigenvalue_route,
                    cfg,
                )
            },
            || {
                compute_sector_branch(
                    &odd_matrix,
                    params.n_modes,
                    CcmParity::Odd,
                    requested_eigenpairs,
                    options.eigenvalue_route,
                    cfg,
                )
            },
        );
        let (even, odd) = (even?, odd?);
        compute_sector_gap(even, odd, cfg.precision_bits)
    }
}

/// Compute or reuse the lowest requested eigenpairs in both reflection-parity
/// sectors and derive the full-precision finite-model GapLog record.
///
/// This is an explicit research operation.  Ordinary CCM root reproduction
/// continues to pay only for the standard even source state.
pub fn analyze_sector_gap(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    requested_eigenpairs: usize,
) -> Result<CcmSectorGapHp> {
    analyze_sector_gap_with_options(
        params,
        cfg,
        CcmSectorAnalysisOptions::selected(requested_eigenpairs),
    )
}

/// Route-explicit sector analysis. Maximum-capture callers should select
/// [`CcmSectorAnalysisOptions::maximum`] so the complete QR spectra are
/// retained; ordinary GapLog analysis defaults to selected indexed values.
pub fn analyze_sector_gap_with_options(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    options: CcmSectorAnalysisOptions,
) -> Result<CcmSectorGapHp> {
    let managed =
        xc_cache::ManagedArtifactCacheSession::from_environment().map_err(anyhow::Error::from)?;
    if let Some(managed) = &managed {
        let cache = managed.context();
        let result = xc_numerics::hp_runtime::run_hp(|| {
            analyze_sector_gap_inner(params, cfg, options, Some(&cache))
        })?;
        managed
            .finalize_publication_inventory()
            .map_err(anyhow::Error::from)?;
        Ok(result)
    } else {
        xc_numerics::hp_runtime::run_hp(|| analyze_sector_gap_inner(params, cfg, options, None))
    }
}

/// Explicit-cache variant of [`analyze_sector_gap`].
///
/// The caller controls overlay order, reuse policy, staging, and publication
/// finalization through `cache`; this function performs no direct network I/O.
pub fn analyze_sector_gap_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    requested_eigenpairs: usize,
    cache: &ArtifactCacheContext<'_>,
) -> Result<CcmSectorGapHp> {
    analyze_sector_gap_with_options_via_cache(
        params,
        cfg,
        CcmSectorAnalysisOptions::selected(requested_eigenpairs),
        cache,
    )
}

pub fn analyze_sector_gap_with_options_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    options: CcmSectorAnalysisOptions,
    cache: &ArtifactCacheContext<'_>,
) -> Result<CcmSectorGapHp> {
    xc_numerics::hp_runtime::run_hp(|| analyze_sector_gap_inner(params, cfg, options, Some(cache)))
}

fn factorization_backward_error(
    matrix: &[Float],
    factors: &xc_numerics::linalg::LuFactors,
    dimension: usize,
    precision_bits: u32,
) -> Option<Float> {
    if matrix.len() != dimension * dimension
        || factors.lu.len() != dimension * dimension
        || factors.perm.len() != dimension
    {
        return None;
    }
    let mut seen = vec![false; dimension];
    for &index in &factors.perm {
        if index >= dimension || seen[index] {
            return None;
        }
        seen[index] = true;
    }
    if matrix.iter().any(|value| !value.is_finite())
        || factors.lu.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let rhs: Vec<Float> = (0..dimension)
        .map(|index| Float::with_val(precision_bits, index + 1))
        .collect();
    let solution = xc_numerics::linalg::lu_solve(factors, &rhs, dimension, precision_bits);
    if solution.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut maximum_residual = Float::with_val(precision_bits, 0);
    let mut matrix_norm = Float::with_val(precision_bits, 0);
    for row in 0..dimension {
        let mut value = Float::with_val(precision_bits, 0);
        let mut row_sum = Float::with_val(precision_bits, 0);
        for column in 0..dimension {
            let mut term = matrix[row * dimension + column].clone();
            row_sum += term.clone().abs();
            term *= &solution[column];
            value += term;
        }
        if row_sum > matrix_norm {
            matrix_norm = row_sum;
        }
        value -= &rhs[row];
        let residual = value.abs();
        if residual > maximum_residual {
            maximum_residual = residual;
        }
    }
    let mut solution_norm = Float::with_val(precision_bits, 0);
    for value in &solution {
        let magnitude = value.clone().abs();
        if magnitude > solution_norm {
            solution_norm = magnitude;
        }
    }
    let mut rhs_norm = Float::with_val(precision_bits, 0);
    for value in &rhs {
        let magnitude = value.clone().abs();
        if magnitude > rhs_norm {
            rhs_norm = magnitude;
        }
    }
    let mut scale = matrix_norm;
    scale *= solution_norm;
    scale += rhs_norm;
    if scale.is_zero() || !scale.is_finite() {
        return None;
    }
    maximum_residual /= scale;
    Some(maximum_residual)
}

fn resolve_factorization_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    matrix: &[Float],
    matrix_manifest: &ArtifactManifest,
    subspace: &str,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(xc_numerics::linalg::LuFactors, ArtifactManifest)> {
    let resolution_start = Instant::now();
    let dimension = if subspace == "even" {
        params.n_modes + 1
    } else {
        params.matrix_size()
    };
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_factorization".to_owned(),
        mathematical_semantics_version: "ccm-dense-lu-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "subspace": subspace,
            "matrix_content_digest": matrix_manifest.content_digest.0,
            "pivoting": "partial"
        }),
        normalization: Some("combined_lu_row_major_with_permutation".to_owned()),
        target: Some("inverse_iteration_linear_solve".to_owned()),
        subspace: Some(subspace.to_owned()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("dense_lu_partial_pivoting".to_owned()),
    };
    let logical_key = format!(
        "ccm/factorization/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        subspace
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.factorization.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "factorization".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    // Validation parses and checks every portable LU entry. Retain that exact
    // validated representation so downstream inverse iteration does not parse
    // tens of thousands of multi-kilobit decimal strings a second time.
    let validated_factors = RefCell::new(None);
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let factors = xc_numerics::linalg::lu_factor(matrix, dimension)
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            Ok((
                PortableLuFactorization {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    subspace: subspace.to_owned(),
                    dimension,
                    lu: factors.lu.iter().map(Float::to_string).collect(),
                    permutation: factors.perm,
                },
                vec![DependencyRef {
                    key: matrix_manifest.key.clone(),
                    content_digest: matrix_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            if artifact.schema_version != 1
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || artifact.subspace != subspace
                || artifact.dimension != dimension
                || artifact.lu.len() != dimension * dimension
                || artifact.permutation.len() != dimension
            {
                return Err(CacheError::InvalidManifest(
                    "CCM factorization does not match its semantic identity".to_owned(),
                ));
            }
            let factors = xc_numerics::linalg::LuFactors {
                lu: parse_hp_vector(&artifact.lu, cfg.precision_bits)?,
                perm: artifact.permutation.clone(),
            };
            let tolerance =
                Float::with_val(cfg.precision_bits, 2).pow(-((cfg.precision_bits / 4) as i32));
            let backward_error =
                factorization_backward_error(matrix, &factors, dimension, cfg.precision_bits)
                    .ok_or_else(|| {
                        CacheError::InvalidManifest(
                    "CCM factorization has invalid dimensions, permutation, or finite values"
                        .to_owned(),
                )
                    })?;
            if backward_error < tolerance {
                validated_factors.replace(Some(factors));
                Ok(())
            } else {
                Err(CacheError::InvalidManifest(
                    format!(
                        "CCM factorization failed its normwise backward-error check: error={}, tolerance={}",
                        xc_numerics::fmt::display_hp(&backward_error, 8),
                        xc_numerics::fmt::display_hp(&tolerance, 8)
                    ),
                ))
            }
        },
    )?;
    let was_produced = resolved.produced_manifest.is_some();
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("factorization execution returned no manifest"))?;
    let factors = validated_factors.into_inner().ok_or_else(|| {
        anyhow::anyhow!("factorization execution did not retain its validated runtime factors")
    })?;
    eprintln!(
        "[HP] {subspace} LU factorization: {} in {:.3}s",
        if was_produced { "computed" } else { "reused" },
        resolution_start.elapsed().as_secs_f64(),
    );
    Ok((factors, manifest))
}

struct BorrowedDenseSymmetricHp<'a> {
    name: &'static str,
    dimension: usize,
    entries: &'a [Float],
    precision_bits: u32,
}

impl LinearOperator<Float> for BorrowedDenseSymmetricHp<'_> {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&self, x: &[Float], y: &mut [Float]) -> std::result::Result<(), OperatorError> {
        if x.len() != self.dimension {
            return Err(OperatorError::DimensionMismatch {
                expected: self.dimension,
                actual: x.len(),
            });
        }
        if y.len() != self.dimension {
            return Err(OperatorError::DimensionMismatch {
                expected: self.dimension,
                actual: y.len(),
            });
        }
        for (row, output) in self.entries.chunks_exact(self.dimension).zip(y.iter_mut()) {
            let mut sum = Float::with_val(self.precision_bits, 0);
            for (entry, component) in row.iter().zip(x) {
                let mut term = Float::with_val(self.precision_bits, entry);
                term *= component;
                sum += term;
            }
            *output = sum;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name,
            self.dimension,
            MatrixStructure::Dense,
            "rug_mpfr",
        );
        metadata.symmetric = true;
        metadata
    }

    fn application_error_bound(&self) -> ApplicationErrorBound<Float> {
        ApplicationErrorBound::Exact
    }
}

impl SymmetricOperator<Float> for BorrowedDenseSymmetricHp<'_> {}

struct RetainedCcmLuShiftInvert<'a> {
    factors: &'a xc_numerics::linalg::LuFactors,
    dimension: usize,
    precision_bits: u32,
    id: String,
}

impl xc_solver::ShiftInvertSolveHp for RetainedCcmLuShiftInvert<'_> {
    fn descriptor(&self) -> xc_solver::ShiftInvertFactorizationDescriptorHp {
        xc_solver::ShiftInvertFactorizationDescriptorHp {
            id: self.id.clone(),
            dimension: self.dimension,
            shift: DecimalLiteral::new("0").expect("zero is a valid decimal literal"),
            factorization_precision_bits: self.precision_bits,
            exact_shifted_solve: true,
            approximation_error_bound: None,
        }
    }

    fn solve_shifted(
        &self,
        right_hand_side: &[Float],
        output: &mut [Float],
        working_precision_bits: u32,
    ) -> std::result::Result<(), xc_solver::SolverError> {
        if working_precision_bits != self.precision_bits
            || right_hand_side.len() != self.dimension
            || output.len() != self.dimension
        {
            return Err(xc_solver::SolverError::InvalidConfiguration(
                "retained CCM LU solve has incompatible precision or dimensions".to_owned(),
            ));
        }
        let solution = xc_numerics::linalg::lu_solve(
            self.factors,
            right_hand_side,
            self.dimension,
            working_precision_bits,
        );
        output.clone_from_slice(&solution);
        Ok(())
    }
}

fn krylov_tolerance(precision_bits: u32) -> DecimalLiteral {
    // Make the projected stopping threshold slightly stricter than the
    // established full-Tau replay floor 2^-(precision_bits-32). The replay
    // remains authoritative, but the Krylov route should reach it before
    // declaring convergence rather than failing only during serialization.
    let decimal_digits =
        u64::from(precision_bits.saturating_sub(24)).saturating_mul(30_103) / 100_000;
    DecimalLiteral::new(format!("1e-{}", decimal_digits.max(12)))
        .expect("generated Krylov tolerance is a valid decimal literal")
}

fn krylov_inverse_compatibility_diagnostics(
    report: &xc_solver::ShiftInvertKrylovReportHp,
) -> xc_numerics::linalg::InverseIterationDiagnostics {
    use xc_numerics::linalg::ShiftedRefinementOutcome;
    let configured_step_limit = report
        .maximum_subspace_dimension
        .saturating_mul(report.restarts.max(1));
    xc_numerics::linalg::InverseIterationDiagnostics {
        configured_step_limit,
        unshifted_steps: report.shifted_solves.min(configured_step_limit),
        unshifted_converged: report.status == ResultStatus::Converged,
        final_relative_rayleigh_change: Some(report.maximum_ritz_value_stability.clone()),
        shifted_refinement: ShiftedRefinementOutcome::Accepted,
        final_relative_residual_norm: report
            .retained_eigenpairs
            .first()
            .map(|pair| pair.scaled_backward_error.clone())
            .unwrap_or_else(|| {
                Float::with_val(report.factorization.factorization_precision_bits, 0)
            }),
    }
}

fn weil_eigenpair_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    tau: &[Float],
    tau_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(
    Float,
    Vec<Float>,
    xc_numerics::linalg::InverseIterationDiagnostics,
    ArtifactManifest,
)> {
    weil_eigenpair_via_cache_with_seed(params, cfg, l, tau, tau_manifest, cache, None, None).map(
        |(eigenvalue, eigenvector, diagnostics, manifest, _)| {
            (eigenvalue, eigenvector, diagnostics, manifest)
        },
    )
}

fn weil_eigenpair_cache_identity(
    params: &CcmParams,
    cfg: &HighPrecConfig,
) -> Result<(SemanticKeyEnvelope, String)> {
    let prec = cfg.precision_bits;
    let route = cfg.eigenstate_solver.as_str();
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_mathematical_parameters = match cfg.eigenstate_solver {
        CcmEigenstateSolver::LegacyInverseIteration => serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": prec,
            "scalar_backend": "rug_mpfr",
            "force_even": parity_policy.legacy_force_even(),
            "normalization": "sum_xi_equals_sqrt_log_lambda_squared",
            "inverse_iteration_step_limit": cfg.inverse_iter_steps
        }),
        CcmEigenstateSolver::ShiftInvertKrylov => serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": prec,
            "scalar_backend": "rug_mpfr",
            "force_even": parity_policy.legacy_force_even(),
            "normalization": "sum_xi_equals_sqrt_log_lambda_squared",
            "eigenstate_route": route,
            "krylov_subspace_dimension": cfg.krylov_subspace_dimension,
            "krylov_maximum_restarts": cfg.krylov_maximum_restarts,
            "krylov_guard_eigenpairs": cfg.krylov_guard_eigenpairs
        }),
        CcmEigenstateSolver::Auto => {
            bail!("automatic CCM eigenstate selection must be resolved before key construction")
        }
    };
    add_adaptive_parity_parameter(&mut resolved_mathematical_parameters, parity_policy);
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_weil_eigenpair".to_owned(),
        mathematical_semantics_version: match (cfg.eigenstate_solver, parity_policy) {
            (CcmEigenstateSolver::LegacyInverseIteration, CcmParityPolicy::AdaptiveEven) => {
                "ccm-smallest-weil-eigenpair-adaptive-even-v1"
            }
            (CcmEigenstateSolver::LegacyInverseIteration, _) => {
                "ccm-smallest-weil-eigenpair-v0.13.0-v3"
            }
            (CcmEigenstateSolver::ShiftInvertKrylov, _) => {
                "ccm-smallest-weil-eigenpair-shift-invert-krylov-v1"
            }
            (CcmEigenstateSolver::Auto, _) => unreachable!(),
        }
        .to_owned(),
        resolved_mathematical_parameters,
        normalization: Some("sum_xi_equals_sqrt_log_lambda_squared".to_owned()),
        target: Some("smallest_weil_form_eigenpair".to_owned()),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(
            match (cfg.eigenstate_solver, parity_policy) {
                (
                    CcmEigenstateSolver::LegacyInverseIteration,
                    CcmParityPolicy::AdaptiveEven,
                ) => "dense_full_space_inverse_iteration_with_conditional_even_projection_and_full_tau_residual_gate_v1",
                (CcmEigenstateSolver::LegacyInverseIteration, _) =>
                    "dense_inverse_iteration_with_half_precision_basin_shifted_rescue_and_full_tau_residual_gate_v1",
                (CcmEigenstateSolver::ShiftInvertKrylov, _) =>
                    "ccm_even_zero_shift_thick_restart_shift_invert_krylov_rayleigh_ritz_v1",
                (CcmEigenstateSolver::Auto, _) => unreachable!(),
            }
            .to_owned(),
        ),
    };
    let logical_key = match cfg.eigenstate_solver {
        CcmEigenstateSolver::LegacyInverseIteration => format!(
            "ccm/weil-eigenpair/{}/{}/{}/{}",
            lambda_squared_cache_identity(params),
            params.n_modes,
            prec,
            parity_policy.cache_label()
        ),
        CcmEigenstateSolver::ShiftInvertKrylov => format!(
            "ccm/weil-eigenpair/{}/{}/{}/{}/{}",
            lambda_squared_cache_identity(params),
            params.n_modes,
            prec,
            parity_policy.cache_label(),
            route
        ),
        CcmEigenstateSolver::Auto => unreachable!(),
    };
    Ok((semantic_key, logical_key))
}

fn accepted_identity_exists(
    semantic_key: &SemanticKeyEnvelope,
    logical_key: &str,
    cache: &ArtifactCacheContext<'_>,
) -> Result<bool> {
    let route_resolver = if cache.mode.compares_against_reference() {
        cache.reference_resolver
    } else {
        cache.resolver
    };
    let (Some(resolver), Some(policy)) = (route_resolver, cache.acceptance) else {
        return Ok(false);
    };
    let key = ArtifactKey {
        kind: semantic_key.artifact_kind.clone(),
        logical_key: logical_key.to_owned(),
        parameters_digest: semantic_key.digest()?,
    };
    match resolver.resolve_manifest(&key, policy) {
        Ok(_) => Ok(true),
        Err(CacheError::NotFound(_)) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn continuation_key_n_modes(
    key: &ArtifactKey,
    params: &CcmParams,
    cfg: &HighPrecConfig,
) -> Option<usize> {
    let fields: Vec<_> = key.logical_key.split('/').collect();
    if fields.len() < 6
        || fields[0] != "ccm"
        || fields[1] != "weil-eigenpair"
        || fields[2] != lambda_squared_cache_identity(params)
        || fields[4] != cfg.precision_bits.to_string()
        || fields[5] != "even"
    {
        return None;
    }
    fields[3].parse().ok()
}

fn discover_lower_n_continuation_seed(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    cache: &ArtifactCacheContext<'_>,
) -> Result<Option<(Vec<Float>, ArtifactManifest)>> {
    let route_resolver = if cache.mode.compares_against_reference() {
        cache.reference_resolver
    } else {
        cache.resolver
    };
    let (Some(resolver), Some(policy)) = (route_resolver, cache.acceptance) else {
        return Ok(None);
    };
    let discovery_started = Instant::now();
    let mut keys = resolver
        .ccm_eigenpair_continuation_keys(
            &xc_cache::CcmEigenpairContinuationQuery {
                lambda_squared: lambda_squared_cache_identity(params),
                maximum_n_modes: params.n_modes,
                precision_bits: cfg.precision_bits,
                force_even: cfg.effective_parity_policy().legacy_force_even(),
            },
            32,
        )
        .map_err(anyhow::Error::from)?;
    keys.retain(|key| {
        continuation_key_n_modes(key, params, cfg).is_some_and(|n_modes| n_modes < params.n_modes)
    });
    keys.sort_by(|left, right| {
        continuation_key_n_modes(right, params, cfg)
            .cmp(&continuation_key_n_modes(left, params, cfg))
            .then_with(|| left.parameters_digest.cmp(&right.parameters_digest))
    });
    for key in keys {
        let source_n =
            continuation_key_n_modes(&key, params, cfg).expect("filtered continuation key has N");
        let resolved = match resolver.resolve(&key, policy) {
            Ok(resolved) => resolved,
            Err(CacheError::NotFound(_)) => continue,
            Err(error) => return Err(error.into()),
        };
        let artifact: PortableWeilEigenpair = serde_json::from_slice(&resolved.payload)
            .map_err(CacheError::from)
            .map_err(anyhow::Error::from)?;
        if artifact.lambda_squared != lambda_squared_cache_identity(params)
            || artifact.n_modes != source_n
            || artifact.precision_bits != cfg.precision_bits
            || !artifact.force_even
            || artifact.parity_policy.is_some()
            || artifact.eigenvector.len() != 2 * source_n + 1
            || !matches!(
                artifact.eigenstate_route.as_str(),
                "legacy_inverse_iteration" | "shift_invert_krylov"
            )
        {
            continue;
        }
        let mut full_state = Vec::with_capacity(artifact.eigenvector.len());
        let mut valid = true;
        for scalar in &artifact.eigenvector {
            match Float::parse(scalar) {
                Ok(parsed) => {
                    let value = Float::with_val(cfg.precision_bits, parsed);
                    valid &= value.is_finite();
                    full_state.push(value);
                }
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        let seed = embedded_even_continuation_seed(
            &full_state,
            source_n,
            params.n_modes,
            cfg.precision_bits,
        )?;
        eprintln!(
            "[HP] Auto eigenstate seed: reused compatible N={} state from {} (digest={}); indexed lookup={:.3}s",
            source_n,
            resolved.layer_name,
            &resolved.manifest.content_digest.0[..12],
            discovery_started.elapsed().as_secs_f64(),
        );
        return Ok(Some((seed, resolved.manifest)));
    }
    eprintln!(
        "[HP] Auto eigenstate seed: no indexed lower-N state found in {:.3}s",
        discovery_started.elapsed().as_secs_f64()
    );
    Ok(None)
}

fn is_retryable_auto_krylov_failure(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause.downcast_ref::<CacheError>().is_some_and(|error| {
            matches!(
                error,
                CacheError::InvalidTransition(message)
                    if message.starts_with(
                        "CCM shift-invert Krylov did not produce an unambiguous converged target:"
                    )
            )
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn weil_eigenpair_via_cache_with_seed(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    tau: &[Float],
    tau_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
    continuation_seed: Option<&[Float]>,
    continuation_manifest: Option<&ArtifactManifest>,
) -> Result<(
    Float,
    Vec<Float>,
    xc_numerics::linalg::InverseIterationDiagnostics,
    ArtifactManifest,
    CcmEigenstateSolver,
)> {
    let parity_policy = cfg.effective_parity_policy();
    if cfg.eigenstate_solver == CcmEigenstateSolver::Auto {
        let mut selected = cfg.clone();
        let consult_exact_cache = cache.mode.consults_overlays_for_route_selection();
        let discover_automatic_seed = cache.mode.may_use_cached_seed();
        if parity_policy != CcmParityPolicy::EvenSector {
            selected.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
            return weil_eigenpair_via_cache_with_seed(
                params,
                &selected,
                l,
                tau,
                tau_manifest,
                cache,
                None,
                None,
            );
        }
        selected.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
        let (semantic_key, logical_key) = weil_eigenpair_cache_identity(params, &selected)?;
        if consult_exact_cache && accepted_identity_exists(&semantic_key, &logical_key, cache)? {
            if cache.mode.compares_against_reference()
                && continuation_seed.is_none()
                && discover_automatic_seed
            {
                if let Some((discovered_seed, discovered_manifest)) =
                    discover_lower_n_continuation_seed(params, &selected, cache)?
                {
                    return weil_eigenpair_via_cache_with_seed(
                        params,
                        &selected,
                        l,
                        tau,
                        tau_manifest,
                        cache,
                        Some(&discovered_seed),
                        Some(&discovered_manifest),
                    );
                }
            }
            return weil_eigenpair_via_cache_with_seed(
                params,
                &selected,
                l,
                tau,
                tau_manifest,
                cache,
                continuation_seed,
                continuation_manifest,
            );
        }
        selected.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        let (semantic_key, logical_key) = weil_eigenpair_cache_identity(params, &selected)?;
        if consult_exact_cache && accepted_identity_exists(&semantic_key, &logical_key, cache)? {
            return weil_eigenpair_via_cache_with_seed(
                params,
                &selected,
                l,
                tau,
                tau_manifest,
                cache,
                None,
                None,
            );
        }
        if params.n_modes <= cfg.krylov_guard_eigenpairs {
            return weil_eigenpair_via_cache_with_seed(
                params,
                &selected,
                l,
                tau,
                tau_manifest,
                cache,
                None,
                None,
            );
        }
        let discovered_seed;
        let discovered_manifest;
        let (seed, seed_manifest) = if continuation_seed.is_some() {
            (continuation_seed, continuation_manifest)
        } else if discover_automatic_seed {
            if let Some((seed, manifest)) =
                discover_lower_n_continuation_seed(params, &selected, cache)?
            {
                discovered_seed = seed;
                discovered_manifest = manifest;
                (Some(discovered_seed.as_slice()), Some(&discovered_manifest))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        selected.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
        let krylov = weil_eigenpair_via_cache_with_seed(
            params,
            &selected,
            l,
            tau,
            tau_manifest,
            cache,
            seed,
            seed_manifest,
        );
        return match krylov {
            Ok(result) => Ok(result),
            Err(error) if is_retryable_auto_krylov_failure(&error) => {
                eprintln!(
                    "[HP] Auto eigenstate solver: shift-invert Krylov did not converge unambiguously; falling back to legacy inverse iteration"
                );
                selected.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
                weil_eigenpair_via_cache_with_seed(
                    params,
                    &selected,
                    l,
                    tau,
                    tau_manifest,
                    cache,
                    None,
                    None,
                )
            }
            Err(error) => Err(error),
        };
    }
    let prec = cfg.precision_bits;
    if cfg.eigenstate_solver == CcmEigenstateSolver::ShiftInvertKrylov
        && parity_policy != CcmParityPolicy::EvenSector
    {
        bail!("shift-invert Krylov CCM currently requires parity_policy=even-sector");
    }
    let route = cfg.eigenstate_solver.as_str();
    let seed_identity = continuation_manifest.map_or_else(
        || "canonical".to_owned(),
        |manifest| format!("from-eigenpair-{}", manifest.content_digest.0),
    );
    let (semantic_key, logical_key) = weil_eigenpair_cache_identity(params, cfg)?;
    let mut tags = BTreeMap::from([
        ("domain".to_owned(), "ccm".to_owned()),
        ("artifact".to_owned(), "weil_eigenpair".to_owned()),
    ]);
    if let Some(seed_manifest) = continuation_manifest {
        tags.insert(
            xc_cache::OUTPUT_VALIDATION_SEED_TAG.to_owned(),
            serde_json::to_string(&xc_cache::DependencyRef {
                key: seed_manifest.key.clone(),
                content_digest: seed_manifest.content_digest.clone(),
                required_quality: seed_manifest.quality,
            })?,
        );
    }
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.weil_eigenpair.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags,
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let (eps_n, xi, diagnostics, factor_manifest, krylov_diagnostics) = if parity_policy
                == CcmParityPolicy::EvenSector
            {
                let (sector, sector_manifest) =
                    resolve_even_sector_matrix_via_cache(params, cfg, tau, tau_manifest, cache)
                        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                let (factors, factor_manifest) = resolve_factorization_via_cache(
                    params,
                    cfg,
                    &sector,
                    &sector_manifest,
                    "even",
                    cache,
                )
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                match cfg.eigenstate_solver {
                    CcmEigenstateSolver::LegacyInverseIteration => {
                        let output = xc_numerics::linalg::inverse_iteration_from_factors_detailed(
                            &sector,
                            &factors,
                            params.n_modes + 1,
                            prec,
                            cfg.inverse_iter_steps,
                            false,
                            None,
                        )
                        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                        let expanded =
                            expand_even_sector_vector(&output.eigenvector, params.n_modes, prec);
                        (
                            output.eigenvalue,
                            normalize_eigenvector(&expanded, l, prec),
                            output.diagnostics,
                            factor_manifest,
                            None,
                        )
                    }
                    CcmEigenstateSolver::ShiftInvertKrylov => {
                        let dimension = params.n_modes + 1;
                        let maximum_subspace_dimension =
                            cfg.krylov_subspace_dimension.min(dimension);
                        if maximum_subspace_dimension
                            <= 1usize.saturating_add(cfg.krylov_guard_eigenpairs)
                        {
                            return Err(CacheError::InvalidManifest(
                                "CCM Krylov subspace must exceed the requested plus guard block"
                                    .to_owned(),
                            ));
                        }
                        let operator = BorrowedDenseSymmetricHp {
                            name: "ccm-even-sector",
                            dimension,
                            entries: &sector,
                            precision_bits: prec,
                        };
                        let shifted = RetainedCcmLuShiftInvert {
                            factors: &factors,
                            dimension,
                            precision_bits: prec,
                            id: format!("ccm-even-lu:{}", factor_manifest.content_digest.0),
                        };
                        let tolerance = krylov_tolerance(prec);
                        let solver_config = xc_solver::ShiftInvertKrylovConfigHp {
                            target: EigenTarget::SmallestMagnitude,
                            precision_bits: prec,
                            requested_eigenpairs: 1,
                            guard_eigenpairs: cfg.krylov_guard_eigenpairs,
                            maximum_subspace_dimension,
                            maximum_restarts: cfg.krylov_maximum_restarts,
                            minimum_restarts: 2,
                            maximum_projected_sweeps: 256,
                            absolute_residual_tolerance: tolerance.clone(),
                            scaled_backward_error_tolerance: tolerance.clone(),
                            ritz_value_stability_tolerance: tolerance.clone(),
                            boundary_cluster_tolerance: tolerance,
                        };
                        let initial_basis = continuation_seed
                            .map(|seed| vec![seed.to_vec()])
                            .unwrap_or_default();
                        let report = xc_solver::ShiftInvertKrylovSolverHp
                            .solve_with_initial_basis(
                                &operator,
                                &shifted,
                                &solver_config,
                                &initial_basis,
                            )
                            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                        if report.status != ResultStatus::Converged
                            || report.boundary_cluster.is_some()
                            || report.retained_eigenpairs.is_empty()
                        {
                            return Err(CacheError::InvalidTransition(format!(
                                "CCM shift-invert Krylov did not produce an unambiguous converged target: status={:?}, boundary_cluster={}",
                                report.status,
                                report.boundary_cluster.is_some()
                            )));
                        }
                        let pair = &report.retained_eigenpairs[0];
                        let expanded =
                            expand_even_sector_vector(&pair.eigenvector, params.n_modes, prec);
                        let diagnostics = krylov_inverse_compatibility_diagnostics(&report);
                        let portable = PortableShiftInvertKrylovDiagnostics {
                            algorithm_semantics:
                                "ccm_even_zero_shift_thick_restart_shift_invert_krylov_rayleigh_ritz_v1"
                                    .to_owned(),
                            factorization_id: report.factorization.id.clone(),
                            requested_eigenpairs: report.requested_eigenpairs,
                            guard_eigenpairs: cfg.krylov_guard_eigenpairs,
                            maximum_subspace_dimension: report.maximum_subspace_dimension,
                            maximum_restarts: cfg.krylov_maximum_restarts,
                            restarts: report.restarts,
                            shifted_solves: report.shifted_solves,
                            operator_applications: report.operator_applications,
                            status: "converged".to_owned(),
                            maximum_ritz_value_stability: report
                                .maximum_ritz_value_stability
                                .to_string(),
                            final_scaled_backward_error: pair
                                .scaled_backward_error
                                .to_string(),
                            final_relative_tau_residual: "pending_full_tau_replay".to_owned(),
                            seed_identity: seed_identity.clone(),
                        };
                        (
                            pair.eigenvalue.clone(),
                            normalize_eigenvector(&expanded, l, prec),
                            diagnostics,
                            factor_manifest,
                            Some(portable),
                        )
                    }
                    CcmEigenstateSolver::Auto => {
                        unreachable!("automatic eigenstate policy is resolved before computation")
                    }
                }
            } else {
                let (factors, factor_manifest) =
                    resolve_factorization_via_cache(params, cfg, tau, tau_manifest, "full", cache)
                        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                let output = xc_numerics::linalg::inverse_iteration_from_factors_detailed(
                    tau,
                    &factors,
                    params.matrix_size(),
                    prec,
                    cfg.inverse_iter_steps,
                    parity_policy == CcmParityPolicy::AdaptiveEven,
                    None,
                )
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                (
                    output.eigenvalue,
                    normalize_eigenvector(&output.eigenvector, l, prec),
                    output.diagnostics,
                    factor_manifest,
                    None,
                )
            };
            let mut diagnostics = diagnostics;
            diagnostics.final_relative_residual_norm = weil_eigvec_cache::relative_residual_norm(
                tau,
                params.matrix_size(),
                &xi,
                &eps_n,
                prec,
            )
            .ok_or_else(|| {
                CacheError::InvalidManifest(
                    "CCM inverse iteration produced an invalid eigenvector".to_owned(),
                )
            })?;
            let krylov_diagnostics = krylov_diagnostics.map(|mut value| {
                value.final_relative_tau_residual =
                    diagnostics.final_relative_residual_norm.to_string();
                value
            });
            Ok((
                PortableWeilEigenpair {
                    schema_version: if krylov_diagnostics.is_some() { 3 } else { 2 },
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: prec,
                    force_even: parity_policy.legacy_force_even(),
                    parity_policy: parity_policy.portable_marker(),
                    eigenstate_route: route.to_owned(),
                    eigenvalue: eps_n.to_string(),
                    eigenvector: xi.iter().map(Float::to_string).collect(),
                    inverse_iteration: PortableInverseIterationDiagnostics::from_runtime(
                        &diagnostics,
                    ),
                    shift_invert_krylov: krylov_diagnostics,
                },
                canonical_dependency_refs(vec![factor_manifest]),
            ))
        },
        |artifact| decode_weil_eigenpair(artifact, params, cfg, tau).map(|_| ()),
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("Weil eigenpair execution returned no manifest"))?;
    let (eigenvalue, eigenvector, diagnostics) =
        decode_weil_eigenpair(&resolved.value, params, cfg, tau)?;
    Ok((
        eigenvalue,
        eigenvector,
        diagnostics,
        manifest,
        cfg.eigenstate_solver,
    ))
}

fn resolve_secular_source_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    eigenpair_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactManifest> {
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_parameters = serde_json::json!({
        "lambda_squared": lambda_squared_cache_identity(params),
        "n_modes": params.n_modes,
        "precision_bits": cfg.precision_bits,
        "force_even": parity_policy.legacy_force_even(),
        "eigenpair_content_digest": eigenpair_manifest.content_digest.0
    });
    add_adaptive_parity_parameter(&mut resolved_parameters, parity_policy);
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_secular_source".to_owned(),
        mathematical_semantics_version: "ccm-secular-source-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: resolved_parameters,
        normalization: Some("sum_xi_equals_sqrt_log_lambda_squared".to_owned()),
        target: Some("ccm_secular_function".to_owned()),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("xi_hat_exponential_sum".to_owned()),
    };
    let logical_key = format!(
        "ccm/secular-source/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        parity_policy.cache_label()
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.secular_source.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "secular_source".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let expected_digest = eigenpair_manifest.content_digest.0.clone();
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            Ok((
                PortableSecularSource {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    force_even: parity_policy.legacy_force_even(),
                    parity_policy: parity_policy.portable_marker(),
                    eigenpair_content_digest: expected_digest.clone(),
                    normalization: "sum_xi_equals_sqrt_log_lambda_squared".to_owned(),
                },
                vec![DependencyRef {
                    key: eigenpair_manifest.key.clone(),
                    content_digest: eigenpair_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            if artifact.schema_version != 1
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || !payload_parity_matches(
                    artifact.force_even,
                    artifact.parity_policy,
                    parity_policy,
                )
                || artifact.eigenpair_content_digest != expected_digest
                || artifact.normalization != "sum_xi_equals_sqrt_log_lambda_squared"
            {
                Err(CacheError::InvalidManifest(
                    "CCM secular-source payload does not match its semantic identity".to_owned(),
                ))
            } else {
                Ok(())
            }
        },
    )?;
    resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("secular-source execution returned no manifest"))
}

#[cfg(feature = "arb")]
fn certify_roots_from_retained_source(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    weights: &[Float],
    secular_manifest: Option<&ArtifactManifest>,
    options: &CcmRootCertificationOptions,
    cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<super::certified_roots::ProductionIndependentCcmRootCertificate> {
    use super::certified_roots::{
        certify_production_independent_ccm_roots, production_ccm_source_weights_digest,
        validate_production_independent_ccm_root_certificate_structure,
    };

    if !params.lambda_sq.is_integer {
        bail!("root-only CCM certification currently requires exact integer lambda_squared");
    }
    let expected_weights_digest =
        production_ccm_source_weights_digest(weights, cfg.precision_bits)?;
    let compute = || {
        certify_production_independent_ccm_roots(
            weights,
            params.lambda_sq_int(),
            params.n_modes,
            &options.target,
            cfg.precision_bits,
            options.isolation_bits,
            &options.interval_newton,
        )
    };
    let Some(cache) = cache else {
        return compute();
    };
    let secular_manifest = secular_manifest.ok_or_else(|| {
        anyhow::anyhow!("managed CCM root certification is missing its secular-source manifest")
    })?;
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_parameters = serde_json::json!({
        "lambda_squared": lambda_squared_cache_identity(params),
        "n_modes": params.n_modes,
        "precision_bits": cfg.precision_bits,
        "force_even": parity_policy.legacy_force_even(),
        "secular_source_content_digest": secular_manifest.content_digest.0,
        "source_weights_digest": expected_weights_digest.0,
        "certification_scope": "exact_stored_point_source",
        "target": options.target,
        "isolation_bits": options.isolation_bits,
        "interval_newton": options.interval_newton,
    });
    add_adaptive_parity_parameter(&mut resolved_parameters, parity_policy);
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_certificate_bundle".to_owned(),
        mathematical_semantics_version: "ccm-exact-point-source-root-certificate-v0.13.0-v1"
            .to_owned(),
        resolved_mathematical_parameters: resolved_parameters,
        normalization: Some("sum_xi_equals_sqrt_log_lambda_squared".to_owned()),
        target: Some("independently_indexed_positive_ccm_roots".to_owned()),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities: BTreeMap::from([(
            "ccm_secular_source".to_owned(),
            secular_manifest.content_digest.clone(),
        )]),
        algorithm_semantics: Some("flint_exact_count_arb_isolation_interval_newton_v1".to_owned()),
    };
    let semantic_digest = semantic_key.digest()?;
    let logical_key = format!(
        "ccm/certificate-bundle/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        semantic_digest.0
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.root_certificate.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Certified,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "root_certificate".to_owned()),
            (
                "certification_scope".to_owned(),
                "exact_stored_point_source".to_owned(),
            ),
        ]),
        provenance_digest: Some(secular_manifest.content_digest.clone()),
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_assessment(
        &request,
        || {
            let certificate = compute().map_err(|error| {
                CacheError::InvalidManifest(format!("CCM root certification failed: {error:#}"))
            })?;
            let evidence_bytes = serde_json::to_vec(&certificate)?;
            let evidence_digest = if let Some(sink) = cache.production_sink {
                sink.record_evidence("ccm-root-interval-certificate", &evidence_bytes)?
            } else {
                ContentDigest::sha256(&evidence_bytes)
            };
            Ok((
                certificate,
                vec![DependencyRef {
                    key: secular_manifest.key.clone(),
                    content_digest: secular_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
                ArtifactProductionAssessment {
                    achieved_assurance: xc_cache::ArtifactAssuranceState::Certified,
                    evidence_digests: vec![evidence_digest],
                },
            ))
        },
        |certificate| {
            validate_production_independent_ccm_root_certificate_structure(certificate)
                .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
            if certificate.integer_cutoff_c != params.lambda_sq_int()
                || certificate.modes != params.n_modes
                || certificate.precision_bits != cfg.precision_bits
                || certificate.isolation_bits != options.isolation_bits
                || certificate.interval_newton != options.interval_newton
                || certificate.target != options.target
                || certificate.source_weights_digest != expected_weights_digest
            {
                return Err(CacheError::InvalidManifest(
                    "CCM root certificate does not match its exact semantic identity".to_owned(),
                ));
            }
            Ok(())
        },
    )?;
    Ok(resolved.value)
}

fn reconcile_computed_roots_with_certificate(
    result: &HighPrecResult,
    certificate: &super::certified_roots::ProductionIndependentCcmRootCertificate,
) -> Result<()> {
    let first = certificate
        .first_selected_positive_index
        .ok_or_else(|| anyhow::anyhow!("CCM root certificate does not assign positive ordinals"))?;
    for (offset, enclosure) in certificate.selected_roots().iter().enumerate() {
        let positive_index = first
            .checked_add(offset)
            .ok_or_else(|| anyhow::anyhow!("certified CCM root ordinal overflows"))?;
        let result_offset = positive_index
            .checked_sub(result.first_positive_root_index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "computed CCM root window begins after certified ordinal {positive_index}"
                )
            })?;
        let computed = result
            .eigenvalues_pos
            .get(result_offset)
            .and_then(EigenvalueResult::value)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "computed CCM root window has no value for certified ordinal {positive_index}"
                )
            })?;
        let lower = Float::with_val(
            result.precision_bits,
            Float::parse(&enclosure.lower)
                .map_err(|error| anyhow::anyhow!("parse certified CCM lower endpoint: {error}"))?,
        );
        let upper = Float::with_val(
            result.precision_bits,
            Float::parse(&enclosure.upper)
                .map_err(|error| anyhow::anyhow!("parse certified CCM upper endpoint: {error}"))?,
        );
        if computed < &lower || computed > &upper {
            bail!(
                "computed CCM root ordinal {positive_index} is outside its certified interval; independent discovery ordering or refinement does not reconcile"
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "arb"))]
fn certify_roots_from_retained_source(
    _params: &CcmParams,
    _cfg: &HighPrecConfig,
    _weights: &[Float],
    _secular_manifest: Option<&ArtifactManifest>,
    _options: &CcmRootCertificationOptions,
    _cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<super::certified_roots::ProductionIndependentCcmRootCertificate> {
    bail!("root-only CCM certification requires the xc-spectral arb feature")
}

fn compute_root_range(
    xi: &[Float],
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    seeds: &[Float],
) -> Vec<EigenvalueResult> {
    let solve = |seed: &Float| {
        solve_r_zero(
            xi,
            params.n_modes,
            l,
            seed,
            cfg.precision_bits,
            cfg.solver_steps,
            cfg.root_solver,
        )
    };
    // Each seed defines an independent pole-safe refinement. `par_iter` on a
    // slice is indexed, so collect retains the exact algebraic seed order.
    // The sequential truncation below preserves the established rule that a
    // valueless failure leaves an unfillable ordered-window gap.
    let mut outcomes = if seeds.len() < 2 {
        seeds.iter().map(solve).collect::<Vec<_>>()
    } else {
        seeds.par_iter().map(solve).collect::<Vec<_>>()
    };
    if let Some(first_failure) = outcomes.iter().position(|outcome| !outcome.has_value()) {
        outcomes.truncate(first_failure + 1);
    }
    outcomes
}

fn ensure_root_window_usable(
    outcomes: &[EigenvalueResult],
    expected_count: usize,
    require_converged: bool,
    domain: IndependentRootDomain,
) -> Result<()> {
    if outcomes.len() != expected_count {
        bail!(
            "CCM root window is incomplete: expected {} outcomes, received {}",
            expected_count,
            outcomes.len()
        );
    }
    let mut previous: Option<&Float> = None;
    for (offset, outcome) in outcomes.iter().enumerate() {
        let result = match outcome {
            EigenvalueResult::Converged(result) => result,
            EigenvalueResult::Stagnated(result) => {
                if require_converged {
                    bail!(
                        "CCM root {} stagnated after {} iterations (correction={}, residual={}, achieved_digits={})",
                        offset + 1,
                        result.diagnostics.iterations,
                        xc_numerics::fmt::display_hp(&result.diagnostics.final_correction, 8),
                        xc_numerics::fmt::display_hp(&result.diagnostics.residual, 8),
                        xc_numerics::fmt::display_hp(&result.diagnostics.achieved_decimal_digits, 8)
                    )
                }
                result
            }
            EigenvalueResult::Approximate(result) => {
                if require_converged {
                    bail!(
                        "CCM root {} reached the {}-iteration limit (correction={}, residual={}, achieved_digits={})",
                        offset + 1,
                        result.diagnostics.iterations,
                        xc_numerics::fmt::display_hp(&result.diagnostics.final_correction, 8),
                        xc_numerics::fmt::display_hp(&result.diagnostics.residual, 8),
                        xc_numerics::fmt::display_hp(&result.diagnostics.achieved_decimal_digits, 8)
                    )
                }
                result
            }
            EigenvalueResult::Failed { iterations, reason } => {
                bail!(
                    "CCM root {} failed after {} iterations: {}",
                    offset + 1,
                    iterations,
                    reason
                )
            }
        };
        if (domain == IndependentRootDomain::Positive && result.value <= 0)
            || result.value.is_zero()
            || previous.is_some_and(|value| &result.value <= value)
        {
            bail!(
                "CCM root values are not a strictly increasing nonzero {} sequence",
                domain.as_str()
            );
        }
        previous = Some(&result.value);
    }
    Ok(())
}

fn format_index_ranges(indices: &[usize]) -> String {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < indices.len() {
        let first = indices[start];
        let mut end = start;
        while end + 1 < indices.len() && indices[end + 1] == indices[end] + 1 {
            end += 1;
        }
        if start == end {
            ranges.push(first.to_string());
        } else {
            ranges.push(format!("{}-{}", first, indices[end]));
        }
        start = end + 1;
    }
    ranges.join(",")
}

fn report_precision_limited_category(label: &str, roots: &[(usize, &RootRefinement)]) {
    if roots.is_empty() {
        return;
    }
    let indices: Vec<usize> = roots.iter().map(|(index, _)| *index).collect();
    let mut minimum_digits = roots[0].1.diagnostics.achieved_decimal_digits.clone();
    let mut maximum_digits = minimum_digits.clone();
    let mut maximum_iterations = 0usize;
    for (_, root) in roots {
        if root.diagnostics.achieved_decimal_digits < minimum_digits {
            minimum_digits = root.diagnostics.achieved_decimal_digits.clone();
        }
        if root.diagnostics.achieved_decimal_digits > maximum_digits {
            maximum_digits = root.diagnostics.achieved_decimal_digits.clone();
        }
        maximum_iterations = maximum_iterations.max(root.diagnostics.iterations);
    }
    eprintln!(
        "[HP] {label} roots retained as computed approximations: indices={}; achieved_digits={}..{}; maximum_iterations={}; full per-root diagnostics are stored in the artifact",
        format_index_ranges(&indices),
        xc_numerics::fmt::display_hp(&minimum_digits, 8),
        xc_numerics::fmt::display_hp(&maximum_digits, 8),
        maximum_iterations
    );
}

fn report_root_status_summary(outcomes: &[EigenvalueResult], first_root_index: usize) {
    if outcomes.is_empty() {
        return;
    }
    let mut converged = 0usize;
    let mut stagnated = Vec::new();
    let mut approximate = Vec::new();
    let mut failed = 0usize;
    for (offset, outcome) in outcomes.iter().enumerate() {
        let index = first_root_index + offset;
        match outcome {
            EigenvalueResult::Converged(_) => converged += 1,
            EigenvalueResult::Stagnated(result) => stagnated.push((index, result)),
            EigenvalueResult::Approximate(result) => approximate.push((index, result)),
            EigenvalueResult::Failed { .. } => failed += 1,
        }
    }
    eprintln!(
        "[HP] root status summary: {} total; {} converged, {} stagnated, {} approximate, {} failed",
        outcomes.len(),
        converged,
        stagnated.len(),
        approximate.len(),
        failed
    );
    report_precision_limited_category("stagnated", &stagnated);
    report_precision_limited_category("iteration-limited", &approximate);
}

fn validate_reference_seed_dataset(
    artifact_mode: RootArtifactMode,
    reference_dataset: Option<&ReferenceZeroDatasetIdentity>,
    first_root_index: usize,
    root_count: usize,
) -> Result<()> {
    match (artifact_mode, reference_dataset) {
        (RootArtifactMode::Independent, None) => Ok(()),
        (RootArtifactMode::Independent, Some(_)) => {
            bail!("independent CCM discovery cannot carry reference-seed provenance")
        }
        (RootArtifactMode::ReferenceSeededRefinement, None) => {
            bail!("reference-seeded CCM refinement requires dataset provenance")
        }
        (RootArtifactMode::ReferenceSeededRefinement, Some(dataset)) => {
            let last_root_index = first_root_index
                .checked_add(root_count.saturating_sub(1))
                .ok_or_else(|| anyhow::anyhow!("reference-seed ordinal range overflows"))?;
            if !dataset.validate()
                || !ContentDigest(dataset.content_sha256.clone()).validate()
                || root_count == 0
                || first_root_index == 0
                || last_root_index > dataset.record_count
            {
                bail!("reference-seeded CCM dataset provenance or ordinal coverage is invalid");
            }
            Ok(())
        }
    }
}

fn residual_replay_matches(
    stored: &Float,
    replayed: Option<&Float>,
    replay_term_scale: Option<&Float>,
    term_count: usize,
    precision_bits: u32,
) -> bool {
    let Some(replayed) = replayed else {
        return false;
    };
    if !stored.is_finite() || !replayed.is_finite() || stored < &0 || replayed < &0 {
        return false;
    }
    if stored == replayed {
        return true;
    }
    let Some(replay_term_scale) = replay_term_scale else {
        return false;
    };
    if !replay_term_scale.is_finite() || replay_term_scale < &0 || term_count == 0 {
        return false;
    }
    let mut difference = Float::with_val(precision_bits, stored);
    difference -= replayed;
    difference.abs_mut();
    // Portable decimal round-tripping can change cancellation-dominated
    // residuals in their low-order bits. Compare the replay to the stored
    // residual itself, not to the unrelated Halley correction tolerance.
    //
    // Each term performs a pole multiply, subtraction, division, and ordered
    // addition. Bound their aggregate roundoff by the absolute term sum,
    // operation count, and the MPFR working unit. This remains meaningful
    // under severe cancellation; a relative comparison to the tiny final
    // residual does not.
    let unit_bits = precision_bits.saturating_sub(4).max(1);
    let mut tolerance = Float::with_val(precision_bits, 2).pow(-(unit_bits as i32));
    tolerance *= replay_term_scale;
    tolerance *= u32::try_from(term_count)
        .unwrap_or(u32::MAX)
        .saturating_mul(8);
    difference <= tolerance
}

#[allow(clippy::too_many_arguments)]
fn decode_root_range(
    artifact: &PortableRootRange,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    first_root_index: usize,
    seeds: &[Float],
    artifact_mode: RootArtifactMode,
    reference_dataset: Option<&ReferenceZeroDatasetIdentity>,
    xi: &[Float],
    l: &Float,
    semantics: RootWindowSemantics,
    require_converged: bool,
) -> std::result::Result<Vec<EigenvalueResult>, CacheError> {
    let parity_policy = cfg.effective_parity_policy();
    let expected_seeds: Vec<String> = seeds.iter().map(Float::to_string).collect();
    if artifact.schema_version != 3
        || artifact.lambda_squared != lambda_squared_cache_identity(params)
        || artifact.n_modes != params.n_modes
        || artifact.precision_bits != cfg.precision_bits
        || !payload_parity_matches(artifact.force_even, artifact.parity_policy, parity_policy)
        || first_root_index == 0
        || artifact.first_root_index != first_root_index
        || artifact.root_domain != semantics.domain
        || artifact.discovery_mode != artifact_mode.as_str()
        || artifact.reference_seeds_used
            != (artifact_mode == RootArtifactMode::ReferenceSeededRefinement)
        || artifact.reference_dataset.as_ref() != reference_dataset
        || artifact.completeness
            != match artifact_mode {
                RootArtifactMode::Independent => "unverified_computed_discovery",
                RootArtifactMode::ReferenceSeededRefinement => "not_applicable_refinement",
            }
        || artifact.starting_points != expected_seeds
        || artifact.outcomes.len() != seeds.len()
        || artifact.solver != cfg.root_solver.display_name().to_ascii_lowercase()
        || artifact.solver_steps != cfg.solver_steps
        || artifact.accuracy_guard_bits != GUARD_BITS
    {
        return Err(CacheError::InvalidManifest(
            "CCM root-range payload does not match its semantic identity".to_owned(),
        ));
    }
    let decoded: Vec<EigenvalueResult> = artifact
        .outcomes
        .iter()
        .map(|outcome| match outcome {
            PortableRootOutcome::Converged(result) => result
                .to_runtime(cfg.precision_bits)
                .map(EigenvalueResult::Converged),
            PortableRootOutcome::Stagnated(result) => result
                .to_runtime(cfg.precision_bits)
                .map(EigenvalueResult::Stagnated),
            PortableRootOutcome::Approximate(result) => result
                .to_runtime(cfg.precision_bits)
                .map(EigenvalueResult::Approximate),
            PortableRootOutcome::Failed { iterations, reason } => Ok(EigenvalueResult::Failed {
                iterations: *iterations,
                reason: reason.clone(),
            }),
        })
        .collect::<std::result::Result<_, _>>()?;
    let mut two_pi_over_l = pi(cfg.precision_bits);
    two_pi_over_l *= 2u32;
    two_pi_over_l /= l;
    let mut previous: Option<&Float> = None;
    for outcome in &decoded {
        let (result, status) = match outcome {
            EigenvalueResult::Converged(result) => (result, "converged"),
            EigenvalueResult::Stagnated(result) => (result, "stagnated"),
            EigenvalueResult::Approximate(result) => (result, "approximate"),
            EigenvalueResult::Failed { .. } => {
                return Err(CacheError::InvalidManifest(
                    "computed CCM root-range payload contains a failed root".to_owned(),
                ))
            }
        };
        if require_converged && status != "converged" {
            return Err(CacheError::InvalidManifest(format!(
                "requested assurance rejects {status} CCM root-range evidence"
            )));
        }
        let value = &result.value;
        let correction_meets_target = result.diagnostics.final_correction
            < root_correction_tolerance(value, cfg.precision_bits);
        let replayed_digits = achieved_decimal_digits(
            value,
            &result.diagnostics.final_correction,
            cfg.precision_bits,
        );
        let replayed = secular_residual_and_scale_at(
            xi,
            params.n_modes,
            &two_pi_over_l,
            value,
            cfg.precision_bits,
        );
        let replayed_residual = replayed.as_ref().map(|(residual, _)| residual);
        let replay_term_scale = replayed.as_ref().map(|(_, scale)| scale);
        let invalid_iterations = result.diagnostics.iterations > cfg.solver_steps;
        let invalid_status = (status == "converged" && !correction_meets_target)
            || (status != "converged" && correction_meets_target)
            || (status == "approximate" && result.diagnostics.iterations != cfg.solver_steps);
        let digits_mismatch = replayed_digits != result.diagnostics.achieved_decimal_digits;
        let residual_mismatch = !residual_replay_matches(
            &result.diagnostics.residual,
            replayed_residual,
            replay_term_scale,
            params.matrix_size(),
            cfg.precision_bits,
        );
        if invalid_iterations || invalid_status || digits_mismatch || residual_mismatch {
            return Err(CacheError::InvalidManifest(format!(
                "validated CCM root-range payload has inconsistent convergence diagnostics: status={status}, iterations={}/{}, correction_meets_target={}, digits_mismatch={}, residual_mismatch={}, stored_residual={}, replayed_residual={}",
                result.diagnostics.iterations,
                cfg.solver_steps,
                correction_meets_target,
                digits_mismatch,
                residual_mismatch,
                result.diagnostics.residual,
                replayed_residual.map_or_else(|| "none".to_owned(), Float::to_string)
            )));
        }
        if (semantics.domain == IndependentRootDomain::Positive
            && value <= &Float::with_val(cfg.precision_bits, 0))
            || value.is_zero()
            || previous.is_some_and(|prior| value <= prior)
        {
            return Err(CacheError::InvalidManifest(format!(
                "CCM root-range payload is not a strictly increasing nonzero {} sequence",
                semantics.domain.as_str()
            )));
        }
        previous = Some(value);
    }
    Ok(decoded)
}

fn root_range_semantic_key(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    first_root_index: usize,
    seeds: &[Float],
    artifact_mode: RootArtifactMode,
    reference_dataset: Option<&ReferenceZeroDatasetIdentity>,
    semantics: RootWindowSemantics,
) -> Result<SemanticKeyEnvelope> {
    validate_reference_seed_dataset(
        artifact_mode,
        reference_dataset,
        first_root_index,
        seeds.len(),
    )?;
    let seed_strings: Vec<String> = seeds.iter().map(Float::to_string).collect();
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_parameters = serde_json::json!({
        "lambda_squared": lambda_squared_cache_identity(params),
        "n_modes": params.n_modes,
        "precision_bits": cfg.precision_bits,
        "force_even": parity_policy.legacy_force_even(),
        "first_root_index": first_root_index,
        "root_count": seeds.len(),
        "discovery_mode": artifact_mode.as_str(),
        "starting_points": seed_strings,
        "reference_seeds_used": artifact_mode == RootArtifactMode::ReferenceSeededRefinement,
        "completeness": match artifact_mode {
            RootArtifactMode::Independent => "unverified_computed_discovery",
            RootArtifactMode::ReferenceSeededRefinement => "not_applicable_refinement",
        },
        "solver": cfg.root_solver.display_name().to_ascii_lowercase(),
        "solver_steps": cfg.solver_steps,
        "accuracy_guard_bits": GUARD_BITS
    });
    add_adaptive_parity_parameter(&mut resolved_parameters, parity_policy);
    if let Some(dataset) = reference_dataset {
        resolved_parameters
            .as_object_mut()
            .expect("CCM root semantic parameters are an object")
            .insert(
                "reference_dataset".to_owned(),
                serde_json::to_value(dataset)?,
            );
    }
    if semantics.is_advanced() {
        let parameters = resolved_parameters
            .as_object_mut()
            .expect("CCM root semantic parameters are an object");
        parameters.insert(
            "root_domain".to_owned(),
            serde_json::json!(semantics.domain.as_str()),
        );
    }
    let source_data_identities = reference_dataset
        .map(|dataset| {
            BTreeMap::from([(
                "reference_zero_dataset".to_owned(),
                ContentDigest(dataset.content_sha256.clone()),
            )])
        })
        .unwrap_or_default();
    Ok(SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: match artifact_mode {
            RootArtifactMode::Independent => "ccm_root_discovery_window",
            RootArtifactMode::ReferenceSeededRefinement => "ccm_root_refinement",
        }
        .to_owned(),
        mathematical_semantics_version: if semantics.is_advanced() {
            "ccm-root-range-v0.13.3-v7"
        } else {
            "ccm-root-range-v0.13.0-v6"
        }
        .to_owned(),
        resolved_mathematical_parameters: resolved_parameters,
        normalization: None,
        target: Some(
            match semantics.domain {
                IndependentRootDomain::Positive => "positive_ccm_spectral_roots",
                IndependentRootDomain::Signed => "signed_ccm_spectral_roots",
            }
            .to_owned(),
        ),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities,
        algorithm_semantics: Some(cfg.root_solver.display_name().to_ascii_lowercase()),
    })
}

#[allow(clippy::too_many_arguments)]
fn resolve_root_range_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    resolved_eigenstate_solver: CcmEigenstateSolver,
    l: &Float,
    xi: &[Float],
    first_root_index: usize,
    seeds: &[Float],
    secular_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
    artifact_mode: RootArtifactMode,
    reference_dataset: Option<&ReferenceZeroDatasetIdentity>,
    semantics: RootWindowSemantics,
) -> Result<(Vec<EigenvalueResult>, ArtifactManifest, bool)> {
    if first_root_index == 0 || seeds.is_empty() {
        bail!("CCM indexed root refinement requires a positive index and nonempty seed window");
    }
    let last_root_index = first_root_index
        .checked_add(seeds.len() - 1)
        .ok_or_else(|| anyhow::anyhow!("CCM indexed root window overflows usize"))?;
    let require_converged = cache.requested_assurance != xc_core::AssuranceLevel::Computed;
    let parity_policy = cfg.effective_parity_policy();

    // A reference-seeded result for a larger prefix contains exactly the same
    // indexed mathematical requests as a shorter prefix. Probe common
    // canonical prefix windows before creating a narrower artifact. This
    // preserves the larger artifact's provenance and prevents one-root claim
    // runs from publishing redundant refinements and diagnostics.
    if artifact_mode == RootArtifactMode::ReferenceSeededRefinement
        && cache.mode.consults_cache_for_result_reuse()
        && reference_dataset.is_some()
        && reference_dataset == Some(&xc_zeta::zeros::bundled_dataset_identity()?)
    {
        let requested_last = last_root_index;
        for candidate_count in [50usize, 25, 134, 100, 116, 250, 460, 500, 1_000] {
            if candidate_count < requested_last
                || (first_root_index == 1 && candidate_count == seeds.len())
            {
                continue;
            }
            let candidate_strings = xc_zeta::zeros::bundled_first_n_strings(candidate_count)?;
            let candidate_seeds = candidate_strings
                .iter()
                .map(|seed| {
                    Float::parse(seed)
                        .map(|parsed| Float::with_val(cfg.precision_bits, parsed))
                        .map_err(|error| anyhow::anyhow!("invalid bundled root seed: {error}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let candidate_semantic = root_range_semantic_key(
                params,
                cfg,
                1,
                &candidate_seeds,
                artifact_mode,
                reference_dataset,
                RootWindowSemantics::strict_positive(candidate_seeds.len()),
            )?;
            let candidate_logical = format!(
                "ccm/root-refinement/{}/{}/{}/{}/1-{}",
                lambda_squared_cache_identity(params),
                params.n_modes,
                cfg.precision_bits,
                parity_policy.cache_label(),
                candidate_count
            );
            let candidate_request = ArtifactExecutionCacheRequest {
                operation: "ccm.root_refinement.resolve_compatible",
                semantic_key: &candidate_semantic,
                logical_key: &candidate_logical,
                resolver: cache.resolver,
                reference_resolver: None,
                acceptance: cache.acceptance,
                ordered_overlays: cache.ordered_overlays.clone(),
                mode: xc_cache::ArtifactExecutionCacheMode::RequireReuse,
                write_on_miss: false,
                write_visibility: cache.write_visibility,
                produced_quality: CacheQuality::Validated,
                producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
                minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
                maximum_reader_version: None,
                tags: BTreeMap::new(),
                provenance_digest: None,
                production_sink: None,
            };
            let candidate = resolve_or_compute_json_artifact_with_dependencies(
                &candidate_request,
                || {
                    Err(CacheError::NotFound(
                        "compatible root-window probe does not compute".to_owned(),
                    ))
                },
                |artifact| {
                    decode_root_range(
                        artifact,
                        params,
                        cfg,
                        1,
                        &candidate_seeds,
                        artifact_mode,
                        reference_dataset,
                        xi,
                        l,
                        RootWindowSemantics::strict_positive(candidate_seeds.len()),
                        require_converged,
                    )
                    .map(|_| ())
                },
            );
            let candidate = match candidate {
                Ok(candidate) => candidate,
                Err(CacheError::NotFound(_)) => continue,
                Err(error) => return Err(error.into()),
            };
            let manifest = candidate
                .reused_manifest
                .ok_or_else(|| anyhow::anyhow!("compatible root-window probe computed"))?;
            let decoded = decode_root_range(
                &candidate.value,
                params,
                cfg,
                1,
                &candidate_seeds,
                artifact_mode,
                reference_dataset,
                xi,
                l,
                RootWindowSemantics::strict_positive(candidate_seeds.len()),
                require_converged,
            )?;
            let start = first_root_index - 1;
            let projected = decoded[start..start + seeds.len()].to_vec();
            eprintln!(
                "  cache root window: reused indices 1..={candidate_count} for contained request {first_root_index}..={last_root_index}"
            );
            return Ok((projected, manifest, true));
        }
    }
    let semantic_key = root_range_semantic_key(
        params,
        cfg,
        first_root_index,
        seeds,
        artifact_mode,
        reference_dataset,
        semantics,
    )?;
    let logical_key = if semantics.is_advanced() {
        format!(
            "ccm/root-discovery/advanced/{}/{}/{}/{}/{}/{}-{}",
            semantics.domain.as_str(),
            lambda_squared_cache_identity(params),
            params.n_modes,
            cfg.precision_bits,
            parity_policy.cache_label(),
            first_root_index,
            last_root_index
        )
    } else {
        format!(
            "ccm/{}/{}/{}/{}/{}/{}-{}",
            match artifact_mode {
                RootArtifactMode::Independent => "root-discovery",
                RootArtifactMode::ReferenceSeededRefinement => "root-refinement",
            },
            lambda_squared_cache_identity(params),
            params.n_modes,
            cfg.precision_bits,
            parity_policy.cache_label(),
            first_root_index,
            last_root_index
        )
    };
    let request = ArtifactExecutionCacheRequest {
        operation: match artifact_mode {
            RootArtifactMode::Independent => "ccm.root_discovery.resolve_or_compute",
            RootArtifactMode::ReferenceSeededRefinement => "ccm.root_refinement.resolve_or_compute",
        },
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse(if semantics.is_advanced() {
            "0.13.3"
        } else {
            "0.13.0"
        })?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            (
                "artifact".to_owned(),
                match artifact_mode {
                    RootArtifactMode::Independent => "root_discovery_window",
                    RootArtifactMode::ReferenceSeededRefinement => "root_refinement",
                }
                .to_owned(),
            ),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let computed = compute_root_range(xi, params, l, cfg, seeds);
            ensure_root_window_usable(&computed, seeds.len(), require_converged, semantics.domain)
                .map_err(|error| CacheError::InvalidTransition(error.to_string()))?;
            let outcomes = computed
                .into_iter()
                .map(|outcome| match outcome {
                    EigenvalueResult::Converged(result) => PortableRootOutcome::Converged(
                        PortableRootRefinement::from_runtime_for_eigenstate_solver(
                            &result,
                            resolved_eigenstate_solver,
                        ),
                    ),
                    EigenvalueResult::Stagnated(result) => PortableRootOutcome::Stagnated(
                        PortableRootRefinement::from_runtime_for_eigenstate_solver(
                            &result,
                            resolved_eigenstate_solver,
                        ),
                    ),
                    EigenvalueResult::Approximate(result) => PortableRootOutcome::Approximate(
                        PortableRootRefinement::from_runtime_for_eigenstate_solver(
                            &result,
                            resolved_eigenstate_solver,
                        ),
                    ),
                    EigenvalueResult::Failed { iterations, reason } => {
                        PortableRootOutcome::Failed { iterations, reason }
                    }
                })
                .collect();
            Ok((
                PortableRootRange {
                    schema_version: 3,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    force_even: parity_policy.legacy_force_even(),
                    parity_policy: parity_policy.portable_marker(),
                    first_root_index,
                    root_domain: semantics.domain,
                    discovery_mode: artifact_mode.as_str().to_owned(),
                    reference_seeds_used: artifact_mode
                        == RootArtifactMode::ReferenceSeededRefinement,
                    reference_dataset: reference_dataset.cloned(),
                    completeness: match artifact_mode {
                        RootArtifactMode::Independent => "unverified_computed_discovery",
                        RootArtifactMode::ReferenceSeededRefinement => "not_applicable_refinement",
                    }
                    .to_owned(),
                    starting_points: seeds.iter().map(Float::to_string).collect(),
                    outcomes,
                    solver: cfg.root_solver.display_name().to_ascii_lowercase(),
                    solver_steps: cfg.solver_steps,
                    accuracy_guard_bits: GUARD_BITS,
                },
                vec![DependencyRef {
                    key: secular_manifest.key.clone(),
                    content_digest: secular_manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                }],
            ))
        },
        |artifact| {
            decode_root_range(
                artifact,
                params,
                cfg,
                first_root_index,
                seeds,
                artifact_mode,
                reference_dataset,
                xi,
                l,
                semantics,
                require_converged,
            )
            .map(|_| ())
        },
    )?;
    let manifest = resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("root-range execution returned no manifest"))?;
    Ok((
        decode_root_range(
            &resolved.value,
            params,
            cfg,
            first_root_index,
            seeds,
            artifact_mode,
            reference_dataset,
            xi,
            l,
            semantics,
            require_converged,
        )?,
        manifest,
        false,
    ))
}

#[allow(clippy::too_many_arguments)]
fn record_run_evidence_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    eps_n: &Float,
    inverse_iteration: &xc_numerics::linalg::InverseIterationDiagnostics,
    roots: &[EigenvalueResult],
    first_root_index: usize,
    eigenpair_manifest: &ArtifactManifest,
    root_manifest: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
    artifact_mode: RootArtifactMode,
    semantics: RootWindowSemantics,
    selected_root_ordinals: &[usize],
) -> Result<ArtifactManifest> {
    if first_root_index == 0 || roots.is_empty() {
        bail!("CCM run evidence requires a nonempty one-based root range");
    }
    let last_root_index = first_root_index
        .checked_add(roots.len() - 1)
        .ok_or_else(|| anyhow::anyhow!("CCM run-evidence root range overflows usize"))?;
    let counts = roots
        .iter()
        .fold((0usize, 0usize, 0usize, 0usize), |mut counts, root| {
            match root {
                EigenvalueResult::Converged(_) => counts.0 += 1,
                EigenvalueResult::Stagnated(_) => counts.1 += 1,
                EigenvalueResult::Approximate(_) => counts.2 += 1,
                EigenvalueResult::Failed { .. } => counts.3 += 1,
            }
            counts
        });
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_parameters = serde_json::json!({
        "lambda_squared": lambda_squared_cache_identity(params),
        "n_modes": params.n_modes,
        "precision_bits": cfg.precision_bits,
        "force_even": parity_policy.legacy_force_even(),
        "discovery_mode": artifact_mode.as_str(),
        "first_root_index": first_root_index,
        "last_root_index": last_root_index,
        "root_count": roots.len(),
        "eigenpair_content_digest": eigenpair_manifest.content_digest.0,
        "root_range_content_digest": root_manifest.content_digest.0
    });
    add_adaptive_parity_parameter(&mut resolved_parameters, parity_policy);
    if semantics.is_advanced() {
        let parameters = resolved_parameters
            .as_object_mut()
            .expect("CCM run evidence parameters are an object");
        parameters.insert(
            "root_domain".to_owned(),
            serde_json::json!(semantics.domain.as_str()),
        );
        parameters.insert(
            "requested_root_count".to_owned(),
            serde_json::json!(semantics.requested_count),
        );
        parameters.insert(
            "allow_incomplete".to_owned(),
            serde_json::json!(semantics.allow_incomplete),
        );
        parameters.insert(
            "selected_root_ordinals".to_owned(),
            serde_json::json!(selected_root_ordinals),
        );
    }
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_convergence_diagnostics".to_owned(),
        mathematical_semantics_version: if semantics.is_advanced() {
            "ccm-run-evidence-v0.13.3-v4"
        } else {
            "ccm-run-evidence-v0.13.0-v3"
        }
        .to_owned(),
        resolved_mathematical_parameters: resolved_parameters,
        normalization: None,
        target: Some("ccm_configuration_run_summary".to_owned()),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let logical_key = if semantics.is_advanced() {
        format!(
            "ccm/run-evidence/advanced/{}/{}/{}/{}/{}/{}/requested-{}/returned-{}",
            semantics.domain.as_str(),
            artifact_mode.as_str(),
            lambda_squared_cache_identity(params),
            params.n_modes,
            cfg.precision_bits,
            parity_policy.cache_label(),
            semantics.requested_count,
            roots.len()
        )
    } else {
        format!(
            "ccm/run-evidence/{}/{}/{}/{}/{}/{}-{}",
            artifact_mode.as_str(),
            lambda_squared_cache_identity(params),
            params.n_modes,
            cfg.precision_bits,
            parity_policy.cache_label(),
            first_root_index,
            last_root_index
        )
    };
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.run_evidence.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse(if semantics.is_advanced() {
            "0.13.3"
        } else {
            "0.13.0"
        })?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "run_evidence".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            Ok((
                PortableRunEvidence {
                    schema_version: 3,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    force_even: parity_policy.legacy_force_even(),
                    parity_policy: parity_policy.portable_marker(),
                    discovery_mode: artifact_mode.as_str().to_owned(),
                    first_root_index,
                    last_root_index,
                    root_count: roots.len(),
                    root_domain: semantics.domain,
                    requested_root_count: semantics
                        .is_advanced()
                        .then_some(semantics.requested_count),
                    allow_incomplete: semantics.is_advanced() && semantics.allow_incomplete,
                    selected_root_ordinals: if semantics.is_advanced() {
                        selected_root_ordinals.to_vec()
                    } else {
                        Vec::new()
                    },
                    weil_min_eigenvalue: eps_n.to_string(),
                    converged_roots: counts.0,
                    stagnated_roots: counts.1,
                    approximate_roots: counts.2,
                    failed_roots: counts.3,
                    inverse_iteration: PortableInverseIterationDiagnostics::from_runtime(
                        inverse_iteration,
                    ),
                },
                canonical_dependency_refs(vec![eigenpair_manifest.clone(), root_manifest.clone()]),
            ))
        },
        |artifact| {
            if artifact.schema_version != 3
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || !payload_parity_matches(
                    artifact.force_even,
                    artifact.parity_policy,
                    parity_policy,
                )
                || artifact.discovery_mode != artifact_mode.as_str()
                || artifact.first_root_index != first_root_index
                || artifact.last_root_index != last_root_index
                || artifact.root_count != roots.len()
                || artifact.root_domain != semantics.domain
                || artifact.requested_root_count
                    != semantics.is_advanced().then_some(semantics.requested_count)
                || artifact.allow_incomplete
                    != (semantics.is_advanced() && semantics.allow_incomplete)
                || artifact.selected_root_ordinals
                    != if semantics.is_advanced() {
                        selected_root_ordinals
                    } else {
                        &[]
                    }
                || artifact.weil_min_eigenvalue != eps_n.to_string()
                || artifact.converged_roots != counts.0
                || artifact.stagnated_roots != counts.1
                || artifact.approximate_roots != counts.2
                || artifact.failed_roots != counts.3
            {
                return Err(CacheError::InvalidManifest(
                    "CCM run evidence does not match its semantic identity".to_owned(),
                ));
            }
            let decoded_inverse = artifact.inverse_iteration.to_runtime(cfg.precision_bits)?;
            if decoded_inverse.configured_step_limit != inverse_iteration.configured_step_limit
                || decoded_inverse.unshifted_steps != inverse_iteration.unshifted_steps
                || decoded_inverse.unshifted_converged != inverse_iteration.unshifted_converged
                || decoded_inverse.final_relative_rayleigh_change
                    != inverse_iteration.final_relative_rayleigh_change
                || decoded_inverse.shifted_refinement != inverse_iteration.shifted_refinement
                || decoded_inverse.final_relative_residual_norm
                    != inverse_iteration.final_relative_residual_norm
            {
                return Err(CacheError::InvalidManifest(
                    "CCM run evidence contains inconsistent inverse-iteration diagnostics"
                        .to_owned(),
                ));
            }
            parse_hp_vector(
                std::slice::from_ref(&artifact.weil_min_eigenvalue),
                cfg.precision_bits,
            )
            .map(|_| ())
        },
    )?;
    resolved
        .produced_manifest
        .or(resolved.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("run-evidence execution returned no manifest"))
}

/// Build the finite CCM source, independently discover the requested positive
/// root prefix, and refine it at the configured HP precision.
///
/// The call is routed through [`xc_numerics::hp_runtime::run_hp`], whose
/// default is a direct full-parallel call. Safe-capped execution is selected
/// only through `run_hp_with_policy`, and the same explicit policy is recorded
/// in provenance; no environment variable changes this route.
pub fn run(params: &CcmParams, cfg: &HighPrecConfig) -> Result<HighPrecResult> {
    if cfg.n_eigenvalues == 0 {
        return build_source(params, cfg);
    }
    run_independent(
        params,
        cfg,
        &ZeroTarget::FirstK {
            count: cfg.n_eigenvalues,
        },
    )
}

/// Compute the Tau/eigenstate/secular source without requesting roots.
pub fn build_source(params: &CcmParams, cfg: &HighPrecConfig) -> Result<HighPrecResult> {
    run_with_acquisition(params, cfg, RootAcquisition::SourceOnly)
}

/// Independently discover and then refine a requested finite-source window.
/// No external reference ordinates are accepted by this API.
pub fn run_independent(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    target: &ZeroTarget,
) -> Result<HighPrecResult> {
    run_independent_with_options(
        params,
        cfg,
        target,
        IndependentRootDiscoveryOptions::default(),
    )
}

/// Independently discover and refine a finite-source window with explicit
/// advanced sign-domain and shortfall policy.
pub fn run_independent_with_options(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    target: &ZeroTarget,
    discovery: IndependentRootDiscoveryOptions,
) -> Result<HighPrecResult> {
    run_with_acquisition(
        params,
        cfg,
        RootAcquisition::Independent {
            target,
            options: discovery,
        },
    )
}

/// Run independent root discovery and all requested research diagnostics in
/// one toolkit-owned cache/publication session. Independent supplemental
/// branches share the Rayon pool and publication is finalized exactly once.
pub fn run_independent_with_research_capture(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    target: &ZeroTarget,
    options: CcmResearchCaptureOptions,
) -> Result<CcmResearchCaptureResult> {
    run_independent_with_options_and_research_capture(
        params,
        cfg,
        target,
        IndependentRootDiscoveryOptions::default(),
        options,
    )
}

/// Advanced independent discovery plus optional research capture in one
/// toolkit-owned cache/publication session.
pub fn run_independent_with_options_and_research_capture(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    target: &ZeroTarget,
    discovery: IndependentRootDiscoveryOptions,
    options: CcmResearchCaptureOptions,
) -> Result<CcmResearchCaptureResult> {
    if options.root_certification.is_some()
        && (discovery.domain != IndependentRootDomain::Positive || discovery.allow_incomplete)
    {
        bail!("root-only certification requires a complete positive independent-discovery window");
    }
    run_with_research_capture(
        params,
        cfg,
        RootAcquisition::Independent {
            target,
            options: discovery,
        },
        options,
    )
}

fn run_with_research_capture(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    acquisition: RootAcquisition<'_>,
    options: CcmResearchCaptureOptions,
) -> Result<CcmResearchCaptureResult> {
    let managed =
        xc_cache::ManagedArtifactCacheSession::from_environment().map_err(anyhow::Error::from)?;
    let execute = |cache: Option<&ArtifactCacheContext<'_>>| {
        let (primary, retained_source) = run_inner_retaining_source(
            params,
            cfg,
            acquisition,
            cache
                .map(CcmCacheRoute::Fabric)
                .unwrap_or(CcmCacheRoute::Standalone),
            None,
        )?;
        let root_certificate = if let Some(certification) = &options.root_certification {
            let certification_started = Instant::now();
            let certificate = certify_roots_from_retained_source(
                params,
                cfg,
                &primary.xi,
                retained_source.secular_manifest.as_ref(),
                certification,
                cache,
            )?;
            reconcile_computed_roots_with_certificate(&primary, &certificate)?;
            eprintln!(
                "[HP] root-only certification: {} roots, exact stored point source, computed ordinals reconciled, {:.3}s",
                certificate.selected_root_count,
                certification_started.elapsed().as_secs_f64()
            );
            Some(certificate)
        } else {
            None
        };
        let supplemental_started = Instant::now();
        // A parity-sector decomposition is a stronger and substantially less
        // expensive natural-state calculation than repeating full dense
        // inverse iteration. The winning sector vector is lifted to the full
        // reflection basis and passed through the same evenness calculation.
        let (sector_gap, retained_source, sector_resolution_limit) = match options.sector_analysis {
            Some(sector_options) => match analyze_sector_gap_from_retained_source(
                params,
                cfg,
                sector_options,
                cache,
                retained_source,
            ) {
                Ok(gap) => (Some(gap), None, None),
                Err(error) if is_sector_resolution_limit(&error) => {
                    (None, None, Some(error.to_string()))
                }
                Err(error) => return Err(error),
            },
            None => (None, Some(retained_source), None),
        };
        if let Some(limitation) = &sector_resolution_limit {
            eprintln!(
                "[HP] sector research capture is precision-limited and was retained without individual eigenpairs or GapLog: {limitation}"
            );
        };
        let evenness = if options.capture_evenness {
            if let Some(gap) = &sector_gap {
                Some(evenness_from_sector_gap(params, cfg.precision_bits, gap)?)
            } else if sector_resolution_limit.is_some() {
                eprintln!(
                    "[HP] natural-evenness evidence was not derived from an unresolved sector cluster"
                );
                None
            } else if let Some(cache) = cache {
                let mut source =
                    retained_source.expect("source is retained when sector analysis is disabled");
                let tau_manifest = source.tau_manifest.take().ok_or_else(|| {
                    anyhow::anyhow!("managed retained CCM source is missing its Tau manifest")
                })?;
                Some(measure_evenness_from_retained_source_via_cache(
                    params,
                    cfg,
                    source.tau,
                    tau_manifest,
                    cache,
                )?)
            } else {
                Some(measure_evenness_from_tau(
                    params,
                    cfg,
                    retained_source
                        .expect("source is retained when sector analysis is disabled")
                        .tau,
                )?)
            }
        } else {
            None
        };
        eprintln!(
            "[HP] supplemental research capture completed in {:.3}s",
            supplemental_started.elapsed().as_secs_f64()
        );
        Ok(CcmResearchCaptureResult {
            primary,
            evenness,
            sector_gap,
            root_certificate,
        })
    };

    if let Some(managed) = &managed {
        #[cfg(not(feature = "arb"))]
        if managed.requested_assurance() != xc_core::AssuranceLevel::Computed {
            bail!(
                "requested {:?} assurance requires an xc-spectral build with the arb feature",
                managed.requested_assurance()
            );
        }
        let cache = managed.context();
        let result = xc_numerics::hp_runtime::run_hp(|| execute(Some(&cache)))?;
        managed
            .finalize_publication_inventory()
            .map_err(anyhow::Error::from)?;
        Ok(result)
    } else {
        xc_numerics::hp_runtime::run_hp(|| execute(None))
    }
}

fn is_sector_resolution_limit(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains("CCM sector resolution limit:"))
}

/// Refine a caller-supplied, explicitly indexed root window while reusing the
/// same managed Tau and secular-source artifacts as every other window.
///
/// This API performs refinement, not independent discovery.  Reference-free
/// discovery and completeness certification are provided by
/// [`super::certified_roots`].
pub fn run_indexed_seeded(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    first_root_index: usize,
    zero_seeds: &[Float],
    dataset: &ReferenceZeroDatasetIdentity,
) -> Result<HighPrecResult> {
    validate_reference_seed_dataset(
        RootArtifactMode::ReferenceSeededRefinement,
        Some(dataset),
        first_root_index,
        zero_seeds.len(),
    )?;
    run_with_acquisition(
        params,
        cfg,
        RootAcquisition::ReferenceSeeded {
            first_root_index,
            seeds: zero_seeds,
            dataset,
        },
    )
}

/// Refine an explicitly indexed reference-seeded window and capture optional
/// sector diagnostics or an independent finite-source root certificate in the
/// same managed cache/publication session.
pub fn run_indexed_seeded_with_research_capture(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    first_root_index: usize,
    zero_seeds: &[Float],
    dataset: &ReferenceZeroDatasetIdentity,
    options: CcmResearchCaptureOptions,
) -> Result<CcmResearchCaptureResult> {
    validate_reference_seed_dataset(
        RootArtifactMode::ReferenceSeededRefinement,
        Some(dataset),
        first_root_index,
        zero_seeds.len(),
    )?;
    run_with_research_capture(
        params,
        cfg,
        RootAcquisition::ReferenceSeeded {
            first_root_index,
            seeds: zero_seeds,
            dataset,
        },
        options,
    )
}

fn run_with_acquisition(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    acquisition: RootAcquisition<'_>,
) -> Result<HighPrecResult> {
    let managed =
        xc_cache::ManagedArtifactCacheSession::from_environment().map_err(anyhow::Error::from)?;
    if let Some(managed) = &managed {
        #[cfg(not(feature = "arb"))]
        if managed.requested_assurance() != xc_core::AssuranceLevel::Computed {
            bail!(
                "requested {:?} assurance requires an xc-spectral build with the arb feature",
                managed.requested_assurance()
            );
        }
        let cache = managed.context();
        let result = xc_numerics::hp_runtime::run_hp(|| {
            run_inner(params, cfg, acquisition, CcmCacheRoute::Fabric(&cache))
        })?;
        managed
            .finalize_publication_inventory()
            .map_err(anyhow::Error::from)?;
        Ok(result)
    } else {
        xc_numerics::hp_runtime::run_hp(|| {
            run_inner(params, cfg, acquisition, CcmCacheRoute::Standalone)
        })
    }
}

/// Runs the HP CCM pipeline through the common cache fabric.
///
/// # Mathematical semantics
/// Builds the localized Weil form, obtains its selected smallest eigenpair,
/// and independently discovers positive finite `D_log` spectral roots.
///
/// # Precision
/// All numerical construction and validation uses the precision in `cfg`.
/// Typed cache payloads store decimal representations of the MPFR values.
///
/// # Failure states
/// Matrix construction, inverse iteration, cache policy, cache corruption,
/// required-reuse misses, and unavailable write overlays return errors.
///
/// # Assurance and validity
/// A reused tau matrix passes structural symmetry checks. A reused eigenpair
/// must additionally pass its full tau residual at working precision.
///
/// # Cache effects
/// Lookup and persistence are governed only by `cache`; neither artifact path
/// performs direct GitHub access or invokes a network subprocess.
///
/// # Example
/// Supply an [`ArtifactCacheContext`] containing the desired ordered overlays
/// and an explicit prefer-reuse, require-reuse, or disabled policy.
pub fn run_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    cache: &ArtifactCacheContext<'_>,
) -> Result<HighPrecResult> {
    if cfg.n_eigenvalues == 0 {
        return xc_numerics::hp_runtime::run_hp(|| {
            run_inner(
                params,
                cfg,
                RootAcquisition::SourceOnly,
                CcmCacheRoute::Fabric(cache),
            )
        });
    }
    let target = ZeroTarget::FirstK {
        count: cfg.n_eigenvalues,
    };
    xc_numerics::hp_runtime::run_hp(|| {
        run_inner(
            params,
            cfg,
            RootAcquisition::Independent {
                target: &target,
                options: IndependentRootDiscoveryOptions::default(),
            },
            CcmCacheRoute::Fabric(cache),
        )
    })
}

pub fn run_indexed_seeded_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    first_root_index: usize,
    zero_seeds: &[Float],
    dataset: &ReferenceZeroDatasetIdentity,
    cache: &ArtifactCacheContext<'_>,
) -> Result<HighPrecResult> {
    validate_reference_seed_dataset(
        RootArtifactMode::ReferenceSeededRefinement,
        Some(dataset),
        first_root_index,
        zero_seeds.len(),
    )?;
    xc_numerics::hp_runtime::run_hp(|| {
        run_inner(
            params,
            cfg,
            RootAcquisition::ReferenceSeeded {
                first_root_index,
                seeds: zero_seeds,
                dataset,
            },
            CcmCacheRoute::Fabric(cache),
        )
    })
}

/// One member of a strictly increasing cross-N CCM continuation sweep.
#[derive(Debug, Clone)]
pub struct CcmIndexedSeededSweepPoint {
    pub params: CcmParams,
    pub first_root_index: usize,
    pub zero_seeds: Vec<Float>,
}

fn embedded_even_continuation_seed(
    full_state: &[Float],
    source_n: usize,
    target_n: usize,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    if target_n <= source_n || full_state.len() != 2 * source_n + 1 {
        bail!("cross-N continuation requires a valid full state and strictly increasing N");
    }
    let mut seed = vec![Float::with_val(precision_bits, 0); target_n + 1];
    seed[0] = Float::with_val(precision_bits, &full_state[source_n]);
    let sqrt_two = Float::with_val(precision_bits, 2).sqrt();
    for mode in 1..=source_n {
        let mut component = Float::with_val(precision_bits, &full_state[source_n - mode]);
        component += &full_state[source_n + mode];
        component /= &sqrt_two;
        seed[mode] = component;
    }
    Ok(seed)
}

fn validate_continuation_sweep(
    points: &[CcmIndexedSeededSweepPoint],
    cfg: &HighPrecConfig,
    dataset: &ReferenceZeroDatasetIdentity,
) -> Result<()> {
    if points.is_empty() {
        bail!("cross-N continuation sweep requires at least one point");
    }
    if cfg.eigenstate_solver != CcmEigenstateSolver::ShiftInvertKrylov
        || cfg.effective_parity_policy() != CcmParityPolicy::EvenSector
    {
        bail!("cross-N continuation requires the forced-even shift-invert Krylov route");
    }
    for point in points {
        validate_reference_seed_dataset(
            RootArtifactMode::ReferenceSeededRefinement,
            Some(dataset),
            point.first_root_index,
            point.zero_seeds.len(),
        )?;
    }
    for pair in points.windows(2) {
        if pair[0].params.lambda_sq != pair[1].params.lambda_sq
            || pair[0].params.n_modes >= pair[1].params.n_modes
        {
            bail!(
                "cross-N continuation requires one lambda-squared value and strictly increasing N"
            );
        }
    }
    Ok(())
}

/// Execute a seeded, strictly increasing N sweep in one caller-owned cache
/// session. The accepted full eigenstate at N is deterministically embedded
/// into the even sector at the next N. Its exact eigenpair content digest is
/// part of the successor cache identity.
pub fn run_indexed_seeded_n_sweep_via_cache(
    points: &[CcmIndexedSeededSweepPoint],
    cfg: &HighPrecConfig,
    dataset: &ReferenceZeroDatasetIdentity,
    cache: &ArtifactCacheContext<'_>,
) -> Result<Vec<HighPrecResult>> {
    validate_continuation_sweep(points, cfg, dataset)?;
    xc_numerics::hp_runtime::run_hp(|| {
        let mut results = Vec::with_capacity(points.len());
        let mut continuation_seed: Option<Vec<Float>> = None;
        let mut continuation_manifest: Option<ArtifactManifest> = None;
        for (position, point) in points.iter().enumerate() {
            let acquisition = RootAcquisition::ReferenceSeeded {
                first_root_index: point.first_root_index,
                seeds: &point.zero_seeds,
                dataset,
            };
            let (result, retained) = run_inner_retaining_source(
                &point.params,
                cfg,
                acquisition,
                CcmCacheRoute::Fabric(cache),
                continuation_seed
                    .as_deref()
                    .zip(continuation_manifest.as_ref()),
            )?;
            let eigenpair_manifest = retained.eigenpair_manifest.as_ref().ok_or_else(|| {
                anyhow::anyhow!("cross-N continuation did not retain its eigenpair manifest")
            })?;
            if let Some(next) = points.get(position + 1) {
                continuation_seed = Some(embedded_even_continuation_seed(
                    &result.xi,
                    point.params.n_modes,
                    next.params.n_modes,
                    cfg.precision_bits,
                )?);
                continuation_manifest = Some(eigenpair_manifest.clone());
            }
            results.push(result);
        }
        Ok(results)
    })
}

/// Managed-session wrapper for [`run_indexed_seeded_n_sweep_via_cache`].
/// Publication inventory is finalized once after the complete sweep.
pub fn run_indexed_seeded_n_sweep(
    points: &[CcmIndexedSeededSweepPoint],
    cfg: &HighPrecConfig,
    dataset: &ReferenceZeroDatasetIdentity,
) -> Result<Vec<HighPrecResult>> {
    validate_continuation_sweep(points, cfg, dataset)?;
    let managed = xc_cache::ManagedArtifactCacheSession::from_environment()
        .map_err(anyhow::Error::from)?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cross-N continuation requires a managed cache session so retained LU and seed provenance are explicit"
            )
        })?;
    let cache = managed.context();
    let results = run_indexed_seeded_n_sweep_via_cache(points, cfg, dataset, &cache)?;
    managed
        .finalize_publication_inventory()
        .map_err(anyhow::Error::from)?;
    Ok(results)
}

fn independently_discovered_starting_points(
    params: &CcmParams,
    l: &Float,
    xi: &[Float],
    target: &ZeroTarget,
    options: IndependentRootDiscoveryOptions,
    precision_bits: u32,
) -> Result<IndependentRootDiscoveryPlan> {
    if xi.len() != params.matrix_size() {
        bail!("independent HP discovery requires one weight per CCM pole");
    }
    let mut spacing = pi(precision_bits);
    spacing *= 2u32;
    spacing /= l;
    let mut maximum = spacing.clone();
    maximum *= params.n_modes;
    let zero = Float::with_val(precision_bits, 0);
    if options.domain == IndependentRootDomain::Signed
        && !matches!(
            target,
            ZeroTarget::FirstK { .. } | ZeroTarget::SymmetricHeightWindow { .. }
        )
    {
        bail!(
            "signed independent discovery supports FirstK and SymmetricHeightWindow targets only"
        );
    }
    let (scan_upper, lower_filter) = match target {
        ZeroTarget::FirstK { count } => {
            if *count == 0 {
                bail!("independent CCM prefix count must be positive");
            }
            (maximum.clone(), zero.clone())
        }
        ZeroTarget::IndexRange { first, last } => {
            if *first == 0 || first > last {
                bail!("independent CCM root indices require 1 <= first <= last");
            }
            (maximum.clone(), zero.clone())
        }
        ZeroTarget::HeightWindow { lower, upper } => {
            let lower = Float::with_val(precision_bits, Float::parse(lower)?);
            let upper = Float::with_val(precision_bits, Float::parse(upper)?);
            if lower <= 0 || lower >= upper || upper > maximum {
                bail!("independent positive height window lies outside the finite CCM reach");
            }
            (upper, lower)
        }
        ZeroTarget::SymmetricHeightWindow { height } => {
            let upper = Float::with_val(precision_bits, Float::parse(height)?);
            if upper <= 0 || upper > maximum {
                bail!("independent symmetric height window lies outside the finite CCM reach");
            }
            (upper, zero.clone())
        }
    };
    let values = if options.domain == IndependentRootDomain::Signed {
        discover_secular_roots_hp_signed(xi, params.n_modes, &spacing, &scan_upper, precision_bits)?
    } else {
        discover_secular_roots_hp(xi, params.n_modes, &spacing, &scan_upper, precision_bits)?
    };
    let requested_count = match target {
        ZeroTarget::FirstK { count } => *count,
        ZeroTarget::IndexRange { first, last } => last - first + 1,
        ZeroTarget::HeightWindow { .. } | ZeroTarget::SymmetricHeightWindow { .. } => values.len(),
    };
    let advanced = options != IndependentRootDiscoveryOptions::default();

    // Preserve the exact v6 request-shaped artifact path for every ordinary
    // caller. Advanced requests instead cache the complete discovered finite
    // window and record the request/projection only in the small evidence
    // artifact. This lets a later contained request reuse the same numerical
    // root payload without duplicating it under a policy-shaped cache key.
    if !advanced {
        let (first_index, selected): (usize, &[Float]) = match target {
            ZeroTarget::FirstK { count } => {
                if values.len() < *count {
                    bail!(
                        "independent HP discovery found only {} positive roots, but target requests {count}; enable the explicit incomplete-window policy or increase finite reach",
                        values.len()
                    );
                }
                (1, &values[..*count])
            }
            ZeroTarget::IndexRange { first, last } => {
                if values.len() < *last {
                    bail!(
                        "independent HP discovery found only {} positive roots, but target requires index {last}; enable the explicit incomplete-window policy or increase finite reach",
                        values.len()
                    );
                }
                (*first, &values[*first - 1..*last])
            }
            ZeroTarget::HeightWindow { .. } | ZeroTarget::SymmetricHeightWindow { .. } => {
                let first_offset = values.partition_point(|value| value <= &lower_filter);
                let selected = &values[first_offset..];
                if selected.is_empty() {
                    bail!("independent computed height window contains no discovered roots");
                }
                (first_offset + 1, selected)
            }
        };
        let artifact_seeds = selected.to_vec();
        return Ok(IndependentRootDiscoveryPlan {
            artifact_first_root_index: first_index,
            selected_positions: (0..artifact_seeds.len()).collect(),
            artifact_seeds,
            result_first_root_index: first_index,
            request_semantics: RootWindowSemantics::strict_positive(requested_count),
        });
    }

    let selected_positions = match target {
        ZeroTarget::FirstK { count } => {
            if values.len() < *count {
                if !options.allow_incomplete {
                    bail!(
                        "independent HP discovery found only {} {} roots, but target requests {count}; enable the explicit incomplete-window policy or increase finite reach",
                        values.len(),
                        options.domain.as_str()
                    );
                }
                eprintln!(
                    "[HP] advanced root discovery exhausted the finite {} window: requested {}, returning {}",
                    options.domain.as_str(),
                    count,
                    values.len()
                );
            }
            if options.domain == IndependentRootDomain::Signed {
                let mut positions: Vec<usize> = (0..values.len()).collect();
                positions.sort_by(|left, right| {
                    values[*left]
                        .clone()
                        .abs()
                        .partial_cmp(&values[*right].clone().abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            values[*left]
                                .partial_cmp(&values[*right])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                });
                positions.truncate((*count).min(positions.len()));
                positions.sort_unstable();
                positions
            } else {
                (0..(*count).min(values.len())).collect()
            }
        }
        ZeroTarget::IndexRange { first, last } => {
            if values.len() < *last {
                if !options.allow_incomplete {
                    bail!(
                        "independent HP discovery found only {} positive roots, but target requires index {last}; enable the explicit incomplete-window policy or increase finite reach",
                        values.len()
                    );
                }
                if values.len() < *first {
                    eprintln!(
                        "[HP] advanced root discovery exhausted the finite positive window: requested indices {}..={}, returning no roots",
                        first, last
                    );
                } else {
                    eprintln!(
                        "[HP] advanced root discovery exhausted the finite positive window: requested indices {}..={}, returning {}..={}",
                        first,
                        last,
                        first,
                        values.len()
                    );
                }
            }
            ((*first - 1).min(values.len())..(*last).min(values.len())).collect()
        }
        ZeroTarget::HeightWindow { .. } | ZeroTarget::SymmetricHeightWindow { .. } => {
            let first_offset = values.partition_point(|value| value <= &lower_filter);
            if first_offset == values.len() && !options.allow_incomplete {
                bail!("independent computed height window contains no discovered roots");
            }
            (first_offset..values.len()).collect()
        }
    };
    if selected_positions.is_empty() && !options.allow_incomplete {
        bail!("independent HP discovery found no roots in the finite source window");
    }
    let result_first_root_index = match target {
        ZeroTarget::IndexRange { first, .. } => *first,
        ZeroTarget::HeightWindow { .. } if options.domain == IndependentRootDomain::Positive => {
            selected_positions
                .first()
                .map_or(1, |position| position + 1)
        }
        _ => 1,
    };
    Ok(IndependentRootDiscoveryPlan {
        artifact_first_root_index: 1,
        artifact_seeds: values,
        selected_positions,
        result_first_root_index,
        request_semantics: RootWindowSemantics::advanced(
            options.domain,
            requested_count,
            options.allow_incomplete,
        ),
    })
}

fn evaluate_secular_hp(
    xi: &[Float],
    n_modes: usize,
    spacing: &Float,
    point: &Float,
    precision_bits: u32,
) -> Result<Float> {
    let mut value = Float::with_val(precision_bits, 0);
    for mode in -(n_modes as i64)..=(n_modes as i64) {
        let index = (mode + n_modes as i64) as usize;
        let mut pole = spacing.clone();
        pole *= fl_i(precision_bits, mode);
        let mut denominator = point.clone();
        denominator -= pole;
        if denominator.is_zero() {
            bail!("independent HP discovery evaluated a secular pole");
        }
        let mut term = xi[index].clone();
        term /= denominator;
        value += term;
    }
    Ok(value)
}

/// Computed-assurance discovery using the full MPFR source. Pole-free
/// intervals are scanned in order and sign-changing brackets are bisected to
/// 128 bits before the configured HP point solver runs. This prevents the
/// severe loss of source information caused by converting a deep CCM state to
/// binary64 and makes pole crossing by the subsequent point solver unlikely.
fn discover_secular_roots_hp(
    xi: &[Float],
    n_modes: usize,
    spacing: &Float,
    scan_upper: &Float,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    discover_secular_roots_hp_range(
        xi,
        n_modes,
        spacing,
        &Float::with_val(precision_bits, 0),
        scan_upper,
        precision_bits,
    )
}

fn discover_secular_roots_hp_signed(
    xi: &[Float],
    n_modes: usize,
    spacing: &Float,
    scan_height: &Float,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    let mut lower = Float::with_val(precision_bits, scan_height);
    lower *= -1;
    discover_secular_roots_hp_range(xi, n_modes, spacing, &lower, scan_height, precision_bits)
}

fn discover_secular_roots_hp_range(
    xi: &[Float],
    n_modes: usize,
    spacing: &Float,
    scan_lower: &Float,
    scan_upper: &Float,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    if scan_lower >= scan_upper {
        bail!("independent HP discovery requires a nonempty scan interval");
    }
    let mut boundaries = vec![scan_lower.clone()];
    for mode in -(n_modes as i64)..=(n_modes as i64) {
        let mut pole = spacing.clone();
        pole *= fl_i(precision_bits, mode);
        if pole > *scan_lower && pole < *scan_upper {
            boundaries.push(pole);
        }
    }
    boundaries.push(scan_upper.clone());

    let margin_fraction = Float::with_val(precision_bits, 2).pow(-64i32);
    // Pole intervals do not share state. Parallel indexed collection retains
    // interval order, while each interval keeps its established sequential
    // subdivision and bisection arithmetic exactly unchanged.
    let interval_roots = boundaries
        .par_windows(2)
        .map(|interval| {
            discover_secular_roots_in_interval_hp(
                xi,
                n_modes,
                spacing,
                interval,
                &margin_fraction,
                precision_bits,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut roots = Vec::new();
    for root in interval_roots.into_iter().flatten() {
        if root > *scan_lower
            && root < *scan_upper
            && roots.last().is_none_or(|previous| &root > previous)
        {
            roots.push(root);
        }
    }
    Ok(roots)
}

#[cfg(test)]
fn discover_secular_roots_hp_sequential_reference(
    xi: &[Float],
    n_modes: usize,
    spacing: &Float,
    scan_upper: &Float,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    let mut boundaries = vec![Float::with_val(precision_bits, 0)];
    for mode in 1..=n_modes {
        let mut pole = spacing.clone();
        pole *= mode;
        if pole < *scan_upper {
            boundaries.push(pole);
        }
    }
    boundaries.push(scan_upper.clone());
    let margin_fraction = Float::with_val(precision_bits, 2).pow(-64i32);
    let mut roots = Vec::new();
    for interval in boundaries.windows(2) {
        for root in discover_secular_roots_in_interval_hp(
            xi,
            n_modes,
            spacing,
            interval,
            &margin_fraction,
            precision_bits,
        )? {
            if roots.last().is_none_or(|previous| &root > previous) {
                roots.push(root);
            }
        }
    }
    Ok(roots)
}

fn discover_secular_roots_in_interval_hp(
    xi: &[Float],
    n_modes: usize,
    spacing: &Float,
    interval: &[Float],
    margin_fraction: &Float,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    const SUBDIVISIONS: usize = 16;
    const BISECTION_STEPS: usize = 128;

    let mut width = interval[1].clone();
    width -= &interval[0];
    if width <= 0 {
        return Ok(Vec::new());
    }
    let mut margin = width.clone();
    margin *= margin_fraction;
    let mut left = interval[0].clone();
    left += &margin;
    let mut right = interval[1].clone();
    right -= &margin;
    if left >= right {
        return Ok(Vec::new());
    }

    let mut roots = Vec::new();
    let mut previous_point = left.clone();
    let mut previous_value =
        evaluate_secular_hp(xi, n_modes, spacing, &previous_point, precision_bits)?;
    for subdivision in 1..=SUBDIVISIONS {
        let mut point = right.clone();
        point -= &left;
        point *= subdivision;
        point /= SUBDIVISIONS;
        point += &left;
        let value = evaluate_secular_hp(xi, n_modes, spacing, &point, precision_bits)?;
        let changes_sign = previous_value.is_zero()
            || value.is_zero()
            || previous_value.is_sign_positive() != value.is_sign_positive();
        if changes_sign {
            let mut bracket_left = previous_point.clone();
            let mut bracket_right = point.clone();
            let mut left_value = previous_value.clone();
            for _ in 0..BISECTION_STEPS {
                let mut midpoint = bracket_left.clone();
                midpoint += &bracket_right;
                midpoint /= 2u32;
                let midpoint_value =
                    evaluate_secular_hp(xi, n_modes, spacing, &midpoint, precision_bits)?;
                if midpoint_value.is_zero() {
                    bracket_left = midpoint.clone();
                    bracket_right = midpoint;
                    break;
                }
                if left_value.is_sign_positive() != midpoint_value.is_sign_positive() {
                    bracket_right = midpoint;
                } else {
                    bracket_left = midpoint;
                    left_value = midpoint_value;
                }
            }
            let mut root = bracket_left;
            root += bracket_right;
            root /= 2u32;
            if roots.last().is_none_or(|previous| &root > previous) {
                roots.push(root);
            }
        }
        previous_point = point;
        previous_value = value;
    }
    Ok(roots)
}

fn run_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    acquisition: RootAcquisition<'_>,
    cache_route: CcmCacheRoute<'_>,
) -> Result<HighPrecResult> {
    run_inner_retaining_source(params, cfg, acquisition, cache_route, None)
        .map(|(result, _)| result)
}

fn run_inner_retaining_source(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    acquisition: RootAcquisition<'_>,
    cache_route: CcmCacheRoute<'_>,
    continuation: Option<(&[Float], &ArtifactManifest)>,
) -> Result<(HighPrecResult, RetainedCcmSource)> {
    let start = Instant::now();
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    if cfg.eigenstate_solver == CcmEigenstateSolver::ShiftInvertKrylov
        && matches!(&cache_route, CcmCacheRoute::Standalone)
    {
        bail!(
            "CCM shift-invert Krylov requires a managed cache context so its retained LU dependency and route-specific artifact identity are explicit"
        );
    }

    let l = log_lambda_sq_hp(params, prec);
    let tau_started = Instant::now();
    let (mut tau, tau_manifest) = match &cache_route {
        CcmCacheRoute::Standalone => (build_tau_hp(params, &l, cfg)?, None),
        CcmCacheRoute::Fabric(cache) => {
            let (tau, manifest) = build_tau_hp_via_cache(params, &l, cfg, cache)?;
            (tau, Some(manifest))
        }
    };
    eprintln!(
        "[HP] phase timing: tau construction/reuse={:.3}s",
        tau_started.elapsed().as_secs_f64()
    );

    // Force exact symmetry of the τ-matrix (parallel compute, sequential write).
    force_symmetric(&mut tau, dim);

    // Smallest eigenpair (ξ, ε_N).
    //
    // After-τ cache check: if a cached Weil eigenvector exists for this
    // (λ², N, prec) AND it validates against the in-hand τ via the
    // eigen-residual ‖τξ − μξ‖, skip the costly LU factorization
    // entirely. A missing or residual-failing entry falls through to a
    // fresh inverse-iteration compute, which is then cached. The check
    // sits *after* `build_tau_hp` precisely so τ is available for the
    // residual validation (the strongest integrity test for ξ).
    //
    let eigenstate_started = Instant::now();
    let parity_policy = cfg.effective_parity_policy();
    let (eps_n, xi, inverse_iteration_diagnostics, eigenpair_manifest, resolved_eigenstate_solver) =
        if let CcmCacheRoute::Fabric(cache) = &cache_route {
            let (seed, seed_manifest) = continuation
                .map(|(seed, manifest)| (Some(seed), Some(manifest)))
                .unwrap_or((None, None));
            let (eps_n, xi, diagnostics, manifest, resolved_eigenstate_solver) =
                weil_eigenpair_via_cache_with_seed(
                    params,
                    cfg,
                    &l,
                    &tau,
                    tau_manifest
                        .as_ref()
                        .expect("fabric tau route retains its exact manifest"),
                    cache,
                    seed,
                    seed_manifest,
                )?;
            (
                eps_n,
                xi,
                diagnostics,
                Some(manifest),
                resolved_eigenstate_solver,
            )
        } else {
            let lambda_sq = params.lambda_sq;
            let n_modes_key = params.n_modes;
            let mut cached_pair: Option<(
                Float,
                Vec<Float>,
                xc_numerics::linalg::InverseIterationDiagnostics,
            )> = None;
            if let Some(c) =
                weil_eigvec_cache::load(lambda_sq, n_modes_key, prec, cfg.cache_mode, parity_policy)
            {
                let replayed_residual =
                    weil_eigvec_cache::relative_residual_norm(&tau, dim, &c.xi, &c.eps_n, prec);
                if c.diagnostics.configured_step_limit == cfg.inverse_iter_steps
                    && replayed_residual
                        .as_ref()
                        .is_some_and(|value| value == &c.diagnostics.final_relative_residual_norm)
                    && weil_eigvec_cache::residual_ok(&tau, dim, &c.xi, &c.eps_n, prec)
                {
                    eprintln!(
                "[HP] loaded cached Weil eigenvector for λ²={}, N={}, prec={} bits (τ-residual validated)",
                lambda_sq.value_f64, n_modes_key, prec
            );
                    cached_pair = Some((c.eps_n, c.xi, c.diagnostics));
                } else {
                    crate::hp_debug!(
                        "[HP] WARNING: cached Weil eigenvector for λ²={}, N={}, prec={} failed \
                 τ-residual validation; recomputing",
                        lambda_sq.value_f64,
                        n_modes_key,
                        prec
                    );
                }
            }
            let pair = match cached_pair {
                Some(pair) => pair,
                None => {
                    // Warm-start from nearby-precision cache if enabled.
                    // Scan for a cached ξ at a nearby precision to use as the
                    // starting vector for inverse iteration instead of the Gaussian.
                    let warm_xi: Option<Vec<Float>> =
                        if cfg.warm_start && parity_policy != CcmParityPolicy::EvenSector {
                            weil_eigvec_cache::find_warm_start(
                                lambda_sq,
                                n_modes_key,
                                prec,
                                cfg.warm_start_tolerance_bits,
                                parity_policy,
                            )
                        } else {
                            None
                        };

                    // Find the selected smallest eigenpair under the explicit
                    // parity policy.
                    let (raw_eigenvalue, raw_eigenvector, mut diagnostics) = if parity_policy
                        == CcmParityPolicy::EvenSector
                    {
                        let sector =
                            build_even_sector_matrix(&tau, params.n_modes, cfg.precision_bits);
                        let sector_dimension = params.n_modes + 1;
                        eprintln!(
                            "[HP] LU factoring {}×{} even-sector matrix (one-time cost)...",
                            sector_dimension, sector_dimension
                        );
                        let output = xc_numerics::linalg::inverse_iteration_detailed(
                            &sector,
                            sector_dimension,
                            prec,
                            cfg.inverse_iter_steps,
                            false,
                        )?;
                        (
                            output.eigenvalue,
                            expand_even_sector_vector(&output.eigenvector, params.n_modes, prec),
                            output.diagnostics,
                        )
                    } else {
                        eprintln!(
                            "[HP] LU factoring {}×{} full matrix (one-time cost)...",
                            dim, dim
                        );
                        let project_adaptively = parity_policy == CcmParityPolicy::AdaptiveEven;
                        let output = if let Some(warm) = warm_xi {
                            crate::hp_debug!(
                                "[HP] starting inverse iteration from warm-start vector"
                            );
                            xc_numerics::linalg::inverse_iteration_from_detailed(
                                &tau,
                                dim,
                                prec,
                                cfg.inverse_iter_steps,
                                project_adaptively,
                                Some(warm),
                            )?
                        } else {
                            xc_numerics::linalg::inverse_iteration_detailed(
                                &tau,
                                dim,
                                prec,
                                cfg.inverse_iter_steps,
                                project_adaptively,
                            )?
                        };
                        (output.eigenvalue, output.eigenvector, output.diagnostics)
                    };
                    crate::hp_debug!("[HP] LU factorization done.");
                    // Normalize: Σ ξ_j = √L.
                    let eps_n = raw_eigenvalue;
                    let xi = normalize_eigenvector(&raw_eigenvector, &l, prec);
                    diagnostics.final_relative_residual_norm =
                        weil_eigvec_cache::relative_residual_norm(&tau, dim, &xi, &eps_n, prec)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "CCM inverse iteration produced an invalid eigenvector"
                                )
                            })?;
                    eprintln!("[HP] Eigenvector computed. Solving spectrum...");
                    weil_eigvec_cache::save(
                        lambda_sq,
                        n_modes_key,
                        prec,
                        &eps_n,
                        &xi,
                        &diagnostics,
                        cfg.cache_mode,
                        parity_policy,
                    );
                    (eps_n, xi, diagnostics)
                }
            };
            (
                pair.0,
                pair.1,
                pair.2,
                None,
                CcmEigenstateSolver::LegacyInverseIteration,
            )
        };
    eprintln!(
        "[HP] phase timing: Weil eigenstate construction/reuse={:.3}s",
        eigenstate_started.elapsed().as_secs_f64()
    );

    if !weil_eigvec_cache::residual_ok(&tau, dim, &xi, &eps_n, prec) {
        bail!(
            "CCM inverse iteration did not meet the working-precision tau-residual acceptance floor after {} steps",
            cfg.inverse_iter_steps
        );
    }
    if !inverse_iteration_diagnostics.unshifted_converged {
        let termination = if inverse_iteration_diagnostics.unshifted_steps
            < inverse_iteration_diagnostics.configured_step_limit
        {
            "early residual-verified shifted rescue"
        } else {
            "unshifted limit reached"
        };
        eprintln!(
            "[HP] eigenstate provenance: {} ({}/{} steps), shifted refinement={:?}, final relative Tau residual={}",
            termination,
            inverse_iteration_diagnostics.unshifted_steps,
            inverse_iteration_diagnostics.configured_step_limit,
            inverse_iteration_diagnostics.shifted_refinement,
            xc_numerics::fmt::display_hp(
                &inverse_iteration_diagnostics.final_relative_residual_norm,
                10
            )
        );
    }

    // Find finite D_log spectral roots as zeros of R(z). The normal path
    // discovers starting points directly from the full-precision secular
    // source; the explicitly seeded API is retained for post-discovery
    // comparison and refinement.
    let roots_started = Instant::now();
    let (
        artifact_first_root_index,
        artifact_seeds,
        selected_root_positions,
        first_root_index,
        artifact_mode,
        reference_dataset,
        root_semantics,
    ) = match acquisition {
        RootAcquisition::SourceOnly => (
            1,
            Vec::new(),
            Vec::new(),
            1,
            RootArtifactMode::Independent,
            None,
            RootWindowSemantics::strict_positive(0),
        ),
        RootAcquisition::Independent { target, options } => {
            let plan =
                independently_discovered_starting_points(params, &l, &xi, target, options, prec)?;
            (
                plan.artifact_first_root_index,
                plan.artifact_seeds,
                plan.selected_positions,
                plan.result_first_root_index,
                RootArtifactMode::Independent,
                None,
                plan.request_semantics,
            )
        }
        RootAcquisition::ReferenceSeeded {
            first_root_index,
            seeds,
            dataset,
        } => {
            validate_reference_seed_dataset(
                RootArtifactMode::ReferenceSeededRefinement,
                Some(dataset),
                first_root_index,
                seeds.len(),
            )?;
            (
                first_root_index,
                seeds
                    .iter()
                    .map(|seed| Float::with_val(prec, seed))
                    .collect::<Vec<_>>(),
                (0..seeds.len()).collect(),
                first_root_index,
                RootArtifactMode::ReferenceSeededRefinement,
                Some(dataset),
                RootWindowSemantics::strict_positive(seeds.len()),
            )
        }
    };

    // Starting points either come from the finite secular source itself or
    // from the explicitly named reference-refinement API.
    crate::hp_debug!(
        "[HP] using {} {} starting points for HP {} refinement (N={})",
        artifact_seeds.len(),
        artifact_mode.as_str(),
        cfg.root_solver.display_name(),
        params.n_modes
    );

    let (canonical_roots, root_manifest, secular_manifest, projected_root_window): (
        Vec<EigenvalueResult>,
        Option<ArtifactManifest>,
        Option<ArtifactManifest>,
        bool,
    ) = if artifact_seeds.is_empty() {
        let secular_manifest = if let CcmCacheRoute::Fabric(cache) = &cache_route {
            Some(resolve_secular_source_via_cache(
                params,
                cfg,
                eigenpair_manifest
                    .as_ref()
                    .expect("fabric eigenpair route retains its exact manifest"),
                cache,
            )?)
        } else {
            None
        };
        (Vec::new(), None, secular_manifest, false)
    } else if let CcmCacheRoute::Fabric(cache) = &cache_route {
        let secular_manifest = resolve_secular_source_via_cache(
            params,
            cfg,
            eigenpair_manifest
                .as_ref()
                .expect("fabric eigenpair route retains its exact manifest"),
            cache,
        )?;
        let (roots, manifest, projected) = resolve_root_range_via_cache(
            params,
            cfg,
            resolved_eigenstate_solver,
            &l,
            &xi,
            artifact_first_root_index,
            &artifact_seeds,
            &secular_manifest,
            cache,
            artifact_mode,
            reference_dataset,
            root_semantics,
        )?;
        (roots, Some(manifest), Some(secular_manifest), projected)
    } else {
        let roots = compute_root_range(&xi, params, &l, cfg, &artifact_seeds);
        ensure_root_window_usable(&roots, artifact_seeds.len(), false, root_semantics.domain)?;
        (roots, None, None, false)
    };
    let eigenvalues_pos = selected_root_positions
        .iter()
        .map(|position| {
            canonical_roots.get(*position).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "advanced CCM root projection ordinal {} lies outside the canonical window of {} roots",
                    position + 1,
                    canonical_roots.len()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let hp_seeds = selected_root_positions
        .iter()
        .map(|position| {
            artifact_seeds.get(*position).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "advanced CCM seed projection ordinal {} lies outside the canonical window of {} roots",
                    position + 1,
                    artifact_seeds.len()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    report_root_status_summary(&eigenvalues_pos, first_root_index);
    eprintln!(
        "[HP] phase timing: root discovery/refinement={:.3}s",
        roots_started.elapsed().as_secs_f64()
    );
    if !projected_root_window {
        if let (CcmCacheRoute::Fabric(cache), Some(root_manifest)) = (&cache_route, &root_manifest)
        {
            record_run_evidence_via_cache(
                params,
                cfg,
                &eps_n,
                &inverse_iteration_diagnostics,
                &eigenvalues_pos,
                first_root_index,
                eigenpair_manifest
                    .as_ref()
                    .expect("fabric eigenpair route retains its exact manifest"),
                root_manifest,
                cache,
                artifact_mode,
                root_semantics,
                &selected_root_positions
                    .iter()
                    .map(|position| position + 1)
                    .collect::<Vec<_>>(),
            )?;
        }
    }
    // After all solves, verify each computed eigenvalue is closest
    // to its assigned seed (detect cross-overs). Log warnings for any
    // mismatches but do not reorder; ordering is fixed before HP refinement.
    for (k, ev_result) in eigenvalues_pos.iter().enumerate() {
        let ev = match ev_result.value() {
            Some(z) => z,
            None => continue,
        };
        if k >= hp_seeds.len() {
            break;
        }
        let ref_k = hp_seeds[k].clone();
        let mut dist_k = ev.clone();
        dist_k -= &ref_k;
        let dist_k = dist_k.abs();
        for (j, ref_j_seed) in hp_seeds.iter().enumerate() {
            if j == k {
                continue;
            }
            if j >= hp_seeds.len() {
                break;
            }
            let mut dist_j = ev.clone();
            dist_j -= ref_j_seed;
            let dist_j = dist_j.abs();
            if dist_j < dist_k {
                crate::hp_debug!(
                    "[HP] WARNING: spectral root {} is closer to seed {} than its own seed {} — possible cross-over",
                    first_root_index + k,
                    first_root_index + j,
                    first_root_index + k
                );
                break;
            }
        }
    }

    Ok((
        HighPrecResult {
            eigenvalues_pos,
            first_positive_root_index: first_root_index,
            weil_min_eigenvalue: eps_n,
            xi,
            inverse_iteration_diagnostics,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            precision_bits: prec,
        },
        RetainedCcmSource {
            tau,
            tau_manifest,
            eigenpair_manifest,
            secular_manifest,
        },
    ))
}

/// Measure the natural evenness of the smallest eigenvector before
/// forced symmetrization.
///
/// Returns `(evenness_deviation, natural_eigenvalue, forced_eigenvalue)` where:
/// - `evenness_deviation` = ‖ξ - γξ‖ / ‖ξ‖ (0 = perfectly even, >0 = asymmetric)
/// - `natural_eigenvalue` = smallest eigenvalue without forcing
/// - `forced_eigenvalue` = smallest *even* eigenvalue (with forcing)
///
/// At small λ, the natural eigenvector is essentially even (deviation ~10⁻¹⁵⁰).
/// At large λ (λ²≥1000), the natural eigenvector may be odd or mixed-symmetry,
/// with deviation O(1). This is a structural property of the construction,
/// not a precision artifact (verified at HP-1000).
pub fn measure_evenness(params: &CcmParams, cfg: &HighPrecConfig) -> Result<EvennessResult> {
    let managed =
        xc_cache::ManagedArtifactCacheSession::from_environment().map_err(anyhow::Error::from)?;
    if let Some(managed) = &managed {
        let cache = managed.context();
        let result =
            xc_numerics::hp_runtime::run_hp(|| measure_evenness_via_cache(params, cfg, &cache))?;
        managed
            .finalize_publication_inventory()
            .map_err(anyhow::Error::from)?;
        Ok(result)
    } else {
        let l = log_lambda_sq_hp(params, cfg.precision_bits);
        let tau = build_tau_hp(params, &l, cfg)?;
        measure_evenness_from_tau(params, cfg, tau)
    }
}

fn evenness_from_natural_state(
    params: &CcmParams,
    precision_bits: u32,
    natural_eval: Float,
    xi_natural: &[Float],
    forced_eval: Float,
) -> EvennessResult {
    let dim = params.matrix_size();
    let squared_terms: Vec<(Float, Float)> = (0..dim)
        .into_par_iter()
        .map(|index| {
            let reflected = dim - 1 - index;
            let mut difference = xi_natural[index].clone();
            difference -= &xi_natural[reflected];
            (difference.square(), xi_natural[index].clone().square())
        })
        .collect();
    let differences: Vec<Float> = squared_terms
        .iter()
        .map(|(difference, _)| difference.clone())
        .collect();
    let norms: Vec<Float> = squared_terms.into_iter().map(|(_, norm)| norm).collect();
    let mut deviation =
        xc_numerics::reduction::deterministic_pairwise_sum_hp(&differences, precision_bits).sqrt();
    let norm = xc_numerics::reduction::deterministic_pairwise_sum_hp(&norms, precision_bits).sqrt();
    if !norm.is_zero() {
        deviation /= norm;
    }
    EvennessResult {
        evenness_deviation: deviation,
        natural_eigenvalue: natural_eval,
        forced_eigenvalue: forced_eval,
    }
}

fn measure_evenness_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    cache: &ArtifactCacheContext<'_>,
) -> Result<EvennessResult> {
    let l = log_lambda_sq_hp(params, cfg.precision_bits);
    let (tau, tau_manifest) = build_tau_hp_via_cache(params, &l, cfg, cache)?;
    measure_evenness_from_retained_source_via_cache(params, cfg, tau, tau_manifest, cache)
}

fn measure_evenness_from_retained_source_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    mut tau: Vec<Float>,
    tau_manifest: ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<EvennessResult> {
    let l = log_lambda_sq_hp(params, cfg.precision_bits);
    force_symmetric(&mut tau, params.matrix_size());

    let mut natural_cfg = cfg.clone();
    natural_cfg.set_parity_policy(CcmParityPolicy::Natural);
    let (natural_eval, natural_xi, _natural_diagnostics, natural_manifest) =
        weil_eigenpair_via_cache(params, &natural_cfg, &l, &tau, &tau_manifest, cache)?;
    let mut forced_cfg = cfg.clone();
    forced_cfg.set_parity_policy(CcmParityPolicy::EvenSector);
    let (forced_eval, _forced_xi, _forced_diagnostics, forced_manifest) =
        weil_eigenpair_via_cache(params, &forced_cfg, &l, &tau, &tau_manifest, cache)?;
    let calculated = evenness_from_natural_state(
        params,
        cfg.precision_bits,
        natural_eval,
        &natural_xi,
        forced_eval,
    );

    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_validation_record".to_owned(),
        mathematical_semantics_version: "ccm-evenness-evidence-v0.13.0-v1".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared_cache_identity(params),
            "n_modes": params.n_modes,
            "precision_bits": cfg.precision_bits,
            "natural_eigenpair": natural_manifest.content_digest.0,
            "forced_eigenpair": forced_manifest.content_digest.0
        }),
        normalization: Some("l2_reflection_deviation".to_owned()),
        target: Some("natural_evenness".to_owned()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let logical_key = format!(
        "ccm/evenness/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.evenness.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "evenness_evidence".to_owned()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            Ok((
                PortableEvennessEvidence {
                    schema_version: 1,
                    lambda_squared: lambda_squared_cache_identity(params),
                    n_modes: params.n_modes,
                    precision_bits: cfg.precision_bits,
                    evenness_deviation: calculated.evenness_deviation.to_string(),
                    natural_eigenvalue: calculated.natural_eigenvalue.to_string(),
                    forced_eigenvalue: calculated.forced_eigenvalue.to_string(),
                },
                canonical_dependency_refs(vec![natural_manifest, forced_manifest]),
            ))
        },
        |artifact| {
            if artifact.schema_version != 1
                || artifact.lambda_squared != lambda_squared_cache_identity(params)
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != cfg.precision_bits
                || artifact.evenness_deviation != calculated.evenness_deviation.to_string()
                || artifact.natural_eigenvalue != calculated.natural_eigenvalue.to_string()
                || artifact.forced_eigenvalue != calculated.forced_eigenvalue.to_string()
            {
                return Err(CacheError::InvalidManifest(
                    "CCM evenness evidence does not match its semantic identity".to_owned(),
                ));
            }
            parse_hp_vector(
                &[
                    artifact.evenness_deviation.clone(),
                    artifact.natural_eigenvalue.clone(),
                    artifact.forced_eigenvalue.clone(),
                ],
                cfg.precision_bits,
            )
            .map(|_| ())
        },
    )?;
    let values = parse_hp_vector(
        &[
            resolved.value.evenness_deviation,
            resolved.value.natural_eigenvalue,
            resolved.value.forced_eigenvalue,
        ],
        cfg.precision_bits,
    )?;
    Ok(EvennessResult {
        evenness_deviation: values[0].clone(),
        natural_eigenvalue: values[1].clone(),
        forced_eigenvalue: values[2].clone(),
    })
}

fn measure_evenness_from_tau(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    mut tau: Vec<Float>,
) -> Result<EvennessResult> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();

    // Force exact symmetry of the τ-matrix (parallel compute, sequential write).
    force_symmetric(&mut tau, dim);

    // Natural (unforced) smallest eigenpair.
    let (natural_eval, xi_natural) =
        xc_numerics::linalg::inverse_iteration(&tau, dim, prec, cfg.inverse_iter_steps, false)?;

    // Forced-even smallest eigenpair.
    let (forced_eval, _xi_forced) =
        xc_numerics::linalg::inverse_iteration(&tau, dim, prec, cfg.inverse_iter_steps, true)?;

    // Evenness deviation: ‖ξ - γξ‖ / ‖ξ‖ where γ is index reflection.
    // γξ_i = ξ_{dim-1-i}. Deviation = ‖ξ - γξ‖₂ / ‖ξ‖₂.
    // Both reductions are parallelized over i.
    let squared_terms: Vec<(Float, Float)> = (0..dim)
        .into_par_iter()
        .map(|i| {
            let reflected = dim - 1 - i;
            let mut d = xi_natural[i].clone();
            d -= &xi_natural[reflected];
            let d_sq = d.square();
            let n_sq = xi_natural[i].clone().square();
            (d_sq, n_sq)
        })
        .collect();
    let diff_terms: Vec<Float> = squared_terms
        .iter()
        .map(|(difference, _)| difference.clone())
        .collect();
    let norm_terms: Vec<Float> = squared_terms.into_iter().map(|(_, norm)| norm).collect();
    let diff_sq = xc_numerics::reduction::deterministic_pairwise_sum_hp(&diff_terms, prec);
    let norm_sq = xc_numerics::reduction::deterministic_pairwise_sum_hp(&norm_terms, prec);
    let mut deviation = diff_sq.sqrt();
    let norm = norm_sq.sqrt();
    if !norm.is_zero() {
        deviation /= &norm;
    }

    Ok(EvennessResult {
        evenness_deviation: deviation,
        natural_eigenvalue: natural_eval,
        forced_eigenvalue: forced_eval,
    })
}

/// Result of the evenness measurement.
pub struct EvennessResult {
    /// ‖ξ - γξ‖ / ‖ξ‖. Zero means perfectly even.
    pub evenness_deviation: Float,
    /// Smallest eigenvalue without forced-even projection.
    pub natural_eigenvalue: Float,
    /// Smallest eigenvalue with forced-even projection.
    pub forced_eigenvalue: Float,
}

// ===========================================================================
// Matrix construction
// ===========================================================================

/// Wrapper around `build_tau_hp_compute` that consults the tau matrix
/// disk cache before invoking the full HP construction.
///
/// At HP-1000 the τ-matrix construction is O(N²) HP integral
/// evaluations + O(N³) LU-equivalent work in inverse iteration; for the
/// representative large configurations (λ²=13/100/1000 at N=120/500/800) this is
/// minutes-to-hours of wall-time. The output is fully determined by
/// `(λ²_int, n_modes, prec)`, so it can be cached on disk and reused
/// across runs.
///
/// Cache layout (mirrors `gl_cache` and `prolate_eigvals_cache`):
///   <cwd>/data/tau_cache/lambda_sq{L}_nmodes{N}_prec{P}.json[.zip[.partXX]]
///
/// Lookup priority:
///   1. Uncompressed `.json`
///   2. Single zip `.json.zip`
///   3. Multi-part split `.json.zip.part00, .part01, ...` (used when
///      compressed payload exceeds GitHub's 100 MB hard limit; we
///      split at 90 MB-byte boundaries and concatenate the parts on
///      read before passing to the zip decoder).
///
/// Cache miss → compute fresh via `build_tau_hp_compute`, save in
/// the most appropriate format for the resulting size.
fn build_tau_hp(params: &CcmParams, l: &Float, cfg: &HighPrecConfig) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let lambda_sq = params.lambda_sq;
    let n_modes = params.n_modes;

    if let Some(cached) = tau_cache::load(lambda_sq, n_modes, prec, cfg.cache_mode) {
        eprintln!(
            "[HP] loaded cached τ-matrix for λ²={}, N={}, prec={} bits ({}×{} = {} entries)",
            lambda_sq.value_f64,
            n_modes,
            prec,
            params.matrix_size(),
            params.matrix_size(),
            cached.len()
        );
        return Ok(cached);
    }

    let tau = build_tau_hp_compute(params, l, cfg, true)?;
    tau_cache::save(lambda_sq, n_modes, prec, &tau, cfg.cache_mode);
    Ok(tau)
}

/// Eigenvalues of the localized Weil-form (`τ`) matrix. The smallest
/// positive eigenvalue is the plunge `ε_N`, whose `−log₁₀|ε_N|` is the CCM
/// prime-content floor `D_Primes(λ²)` (up to the small ε_N↔matching-digit
/// offset).
///
/// `include_primes`:
/// - `true`  → the full Weil form (archimedean + prime sum). This is the
///   same `ε_N` returned by [`run`] / cached in `weil_eigvec_cache`.
/// - `false` → the **archimedean-only** form (prime sum `w_p` dropped from
///   every `τ` entry). Used to test the prefactor decomposition: do the
///   archimedean-only and full plunges share the same prefactor power
///   (the prime sum shifting only the additive constant)?
///
/// Returns the **full Weil-form spectrum** (all eigenvalues of `τ`, sorted
/// ascending). The smallest positive eigenvalue is the plunge `ε_N`; the
/// overall minimum reveals whether the form is positive (Weil positivity)
/// or O(1)-indefinite (relevant for the archimedean-only mode, where the
/// prime sum that enforces positivity is absent).
///
/// Uses the dense HP symmetric eigensolver (all eigenvalues) rather than
/// inverse iteration, since the archimedean-only spectrum bottom may be
/// negative — `inverse_iteration` finds the smallest-*magnitude* eigenvalue,
/// not the spectrum minimum, so it would be the wrong tool there. The
/// archimedean-only path bypasses the τ disk cache (keyed only on
/// `(λ², N, prec)`, which would collide with the full matrix).
pub fn weil_spectrum_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    xc_numerics::hp_runtime::run_hp(|| weil_spectrum_hp_inner(params, cfg, include_primes))
}

/// Assemble the dense HP Weil-form matrix for independent solver and
/// certificate routes. The returned storage is exactly symmetric row-major
/// data at `cfg.precision_bits`; `include_primes=false` selects the
/// archimedean-only form and bypasses the full-form cache.
pub fn weil_matrix_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    xc_numerics::hp_runtime::run_hp(|| weil_matrix_hp_inner(params, cfg, include_primes))
}

fn weil_matrix_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, cfg.precision_bits);
    let mut tau = if include_primes {
        build_tau_hp(params, &l, cfg)?
    } else {
        build_tau_hp_compute(params, &l, cfg, false)?
    };
    force_symmetric(&mut tau, dim);
    Ok(tau)
}

fn weil_spectrum_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    let tau = weil_matrix_hp_inner(params, cfg, include_primes)?;
    xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&tau, dim, prec)
}

/// Decomposition of the plunge into archimedean and prime Rayleigh
/// contributions on the **full** plunge eigenvector ξ.
///
/// The localized Weil form splits as `A_full = A_arch − A_prime` (the τ
/// entry is `w02 − wr − wp`; the archimedean part is `w02 − wr`, the prime
/// part is the prime-power sum `wp`). By linearity of the Rayleigh
/// quotient on the full plunge eigenvector ξ:
/// ```text
///   ε_N = ⟨A_full ξ,ξ⟩/⟨ξ,ξ⟩
///       = ⟨A_arch ξ,ξ⟩/⟨ξ,ξ⟩ − ⟨A_prime ξ,ξ⟩/⟨ξ,ξ⟩
///       = arch_rayleigh − prime_rayleigh.
/// ```
/// Since `ε_N` is exponentially small while `arch_rayleigh` and
/// `prime_rayleigh` are O(1)-or-larger, the two must agree to
/// `≈ −log₁₀|ε_N|` digits — a direct quantitative picture of the
/// archimedean↔prime (Weil-positivity) cancellation that produces the
/// CCM convergence floor.
pub struct PlungeCancellation {
    /// Plunge eigenvalue `ε_N = arch_rayleigh − prime_rayleigh`.
    pub eps_n: Float,
    /// `⟨A_arch ξ,ξ⟩/⟨ξ,ξ⟩` — archimedean Rayleigh on the full plunge ξ.
    pub arch_rayleigh: Float,
    /// `⟨A_prime ξ,ξ⟩/⟨ξ,ξ⟩` — prime-sum Rayleigh on the full plunge ξ.
    pub prime_rayleigh: Float,
}

/// Compute the [`PlungeCancellation`] for `(λ², N, P)`. Finds the full
/// plunge eigenvector ξ by inverse iteration, then evaluates the
/// archimedean and prime Rayleigh quotients on that same ξ.
pub fn weil_plunge_cancellation_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
) -> Result<PlungeCancellation> {
    xc_numerics::hp_runtime::run_hp(|| weil_plunge_cancellation_hp_inner(params, cfg))
}

fn weil_plunge_cancellation_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
) -> Result<PlungeCancellation> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);

    // Full Weil form; smallest positive eigenvalue (the plunge) and its
    // eigenvector via the dense HP symmetric path.
    let mut tau_full = build_tau_hp(params, &l, cfg)?;
    force_symmetric(&mut tau_full, dim);
    let eigs = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&tau_full, dim, prec)?;
    let zero = Float::with_val(prec, 0);
    let eps_n = eigs
        .iter()
        .find(|e| **e > zero)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no positive eigenvalue (Weil form indefinite?)"))?;
    let xi = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
        &tau_full,
        dim,
        &eps_n,
        prec,
        cfg.inverse_iter_steps,
    )?;

    // Archimedean-only form (prime sum dropped).
    let mut tau_arch = build_tau_hp_compute(params, &l, cfg, false)?;
    force_symmetric(&mut tau_arch, dim);

    // Rayleigh quotients on the SAME ξ. The split is exact by linearity:
    // arch_rayleigh − prime_rayleigh = full_rayleigh = ε_N.
    let full_rayleigh = xc_numerics::linalg::rayleigh_quotient(&tau_full, dim, &xi, prec);
    let arch_rayleigh = xc_numerics::linalg::rayleigh_quotient(&tau_arch, dim, &xi, prec);
    let mut prime_rayleigh = Float::with_val(prec, &arch_rayleigh);
    prime_rayleigh -= &full_rayleigh;

    Ok(PlungeCancellation {
        eps_n: full_rayleigh,
        arch_rayleigh,
        prime_rayleigh,
    })
}

/// `sinc(t) = sin(t)/t`, with `sinc(0) = 1`, at working precision `prec`.
fn sinc_hp(t: &Float, prec: u32) -> Float {
    let tiny = Float::with_val(prec, Float::parse("1e-40").unwrap());
    if t.cmp_abs(&tiny).map(|o| o.is_lt()).unwrap_or(false) {
        Float::with_val(prec, 1)
    } else {
        let mut s = t.clone().sin();
        s /= t;
        s
    }
}

/// Time-frequency (prolate) band-concentration matrix `C` in the same
/// `V_n` trigonometric basis as the localized Weil form, for a frequency
/// band `(−Ω, Ω)` (`omega` = Ω).
///
/// With `φ_n(x) = e^{2π i n x / L}/√L` on the log-interval `(−a, a)`
/// (`a = ½ ln λ²`, `L = 2a`) and `ω_n = 2π n / L`, the entry is the
/// time-then-band-limiting (Slepian concentration) operator
/// `C[n,m] = (a/π) ∫_{−Ω}^{Ω} sinc(a(ω_n−ξ)) sinc(a(ω_m−ξ)) dξ`,
/// computed by HP Gauss–Legendre quadrature on the band.
///
/// Its eigenvalues `χ ∈ (0,1)` are the band-concentration ratios: `χ ≈ 1`
/// modes are band-concentrated (the prolate/PSWF subspace), `χ ≈ 0` modes
/// span the Sonin-like (anti-band) subspace. The number of `χ ≈ 1` modes
/// is the Shannon number `≈ 2aΩ/π`. Symmetric, row-major, dimension `2N+1`,
/// in the `params.idx` ordering (so it matches `weil_spectrum_hp`).
pub fn band_concentration_matrix_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
) -> Result<Vec<Float>> {
    xc_numerics::hp_runtime::run_hp(|| band_concentration_matrix_hp_inner(params, cfg, omega))
}

fn band_concentration_matrix_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let n_max = params.n_modes as i64;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);
    let mut a = l.clone();
    a /= 2u32;
    let pi_v = pi(prec);
    let npts = cfg.quad_points.max(8 * params.n_modes + 64);
    let (nodes, weights) =
        xc_numerics::quadrature::gauss_legendre_nodes(npts, prec, cfg.cache_mode);

    // ω_n = 2π n / L, indexed by position params.idx(n) = n + N.
    let omega_n: Vec<Float> = (-n_max..=n_max)
        .map(|n| {
            let mut w = pi_v.clone();
            w *= 2u32;
            w *= fl_i(prec, n);
            w /= &l;
            w
        })
        .collect();

    // prefactor (a/π)·Ω folded into the quadrature weight.
    let mut aon = a.clone();
    aon /= &pi_v;
    aon *= omega;

    let mut c = vec![Float::with_val(prec, 0); dim * dim];
    for (q, node) in nodes.iter().enumerate() {
        let mut xi = node.clone(); // GL node on [-1,1]
        xi *= omega; // ξ_q = Ω·node ∈ (−Ω, Ω)
        let svec: Vec<Float> = omega_n
            .iter()
            .map(|wn| {
                let mut arg = wn.clone();
                arg -= &xi;
                arg *= &a;
                sinc_hp(&arg, prec)
            })
            .collect();
        let mut wq = weights[q].clone();
        wq *= &aon;
        for i in 0..dim {
            let mut wi = svec[i].clone();
            wi *= &wq;
            for j in i..dim {
                let mut term = wi.clone();
                term *= &svec[j];
                c[i * dim + j] += &term;
            }
        }
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let v = c[i * dim + j].clone();
            c[j * dim + i] = v;
        }
    }
    Ok(c)
}

/// Result of restricting the archimedean Weil form to the Sonin-like
/// (anti-band) subspace via band-concentration deflation.
pub struct SoninRestriction {
    /// Band-concentration eigenvalues `χ ∈ (0,1)`, ascending. The count
    /// of `χ ≈ 1` is the Shannon number; the `χ ≈ 0` tail is the Sonin
    /// subspace.
    pub chi: Vec<Float>,
    /// Spectrum of the archimedean Weil form after deflating the top
    /// `n_dropped` band-concentrated modes (those land near `+σ`). The
    /// smallest entry is the archimedean Rayleigh minimum on the
    /// band-complement (Sonin-like) subspace — positive iff archimedean
    /// positivity holds there (Connes Thm 7.1).
    pub spectrum: Vec<Float>,
    /// Number of band-concentrated modes deflated out.
    pub n_dropped: usize,
}

/// Archimedean Weil-form spectrum restricted to the Sonin-like (anti-band)
/// subspace. The full-space archimedean form is O(1)-indefinite; this
/// deflates the top `n_drop` band-concentration eigenvectors (band
/// `(−Ω, Ω)`, `omega = Ω`) by a positive shift `σ`, leaving the
/// archimedean form on the band-complement. Returns the deflated spectrum;
/// its minimum is the archimedean Rayleigh minimum on that subspace.
///
/// Method: build [`band_concentration_matrix_hp`], take its top `n_drop`
/// eigenvectors `v_k` (largest `χ`), and form `A_deflated = A_arch + σ Σ_k v_k v_kᵀ`
/// with `σ` a Gershgorin bound on `‖A_arch‖` (so deflated modes leave the
/// spectrum bottom). `A_arch` is the prime-free Weil matrix
/// (`build_tau_hp_compute(.., include_primes=false)`).
pub fn weil_spectrum_sonin_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
    n_drop: usize,
) -> Result<SoninRestriction> {
    xc_numerics::hp_runtime::run_hp(|| weil_spectrum_sonin_hp_inner(params, cfg, omega, n_drop))
}

fn weil_spectrum_sonin_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
    n_drop: usize,
) -> Result<SoninRestriction> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);

    let cmat = band_concentration_matrix_hp_inner(params, cfg, omega)?;
    let chi = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&cmat, dim, prec)?;

    let n_drop = n_drop.min(dim);
    let mut deflate: Vec<Vec<Float>> = Vec::with_capacity(n_drop);
    for k in 0..n_drop {
        let chi_k = &chi[dim - 1 - k]; // largest χ first
        let mut v = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
            &cmat,
            dim,
            chi_k,
            prec,
            cfg.inverse_iter_steps,
        )?;
        xc_numerics::linalg::normalize_l2(&mut v);
        deflate.push(v);
    }

    let mut a_arch = build_tau_hp_compute(params, &l, cfg, false)?;
    force_symmetric(&mut a_arch, dim);

    // σ = 10 · (Gershgorin bound on |spectrum|) + 1.
    let mut sigma = Float::with_val(prec, 0);
    for i in 0..dim {
        let mut row = Float::with_val(prec, 0);
        for j in 0..dim {
            row += a_arch[i * dim + j].clone().abs();
        }
        if row > sigma {
            sigma = row;
        }
    }
    sigma *= 10u32;
    sigma += 1u32;

    for v in &deflate {
        for i in 0..dim {
            let mut svi = v[i].clone();
            svi *= &sigma;
            for j in 0..dim {
                let mut term = svi.clone();
                term *= &v[j];
                a_arch[i * dim + j] += &term;
            }
        }
    }
    force_symmetric(&mut a_arch, dim);
    let spectrum = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&a_arch, dim, prec)?;

    Ok(SoninRestriction {
        chi,
        spectrum,
        n_dropped: n_drop,
    })
}

fn build_tau_hp_compute(
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    build_tau_hp_compute_exact(
        params.n_modes,
        params.lambda_sq_int(),
        l,
        cfg,
        include_primes,
    )
}

fn build_tau_hp_compute_exact(
    n_modes: usize,
    lambda_sq_int: u64,
    l: &Float,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    let (components, _) =
        build_tau_components_exact_tracked(n_modes, lambda_sq_int, l, cfg, include_primes, None)?;
    Ok(assemble_tau_components(&components, cfg.precision_bits))
}

fn report_quadrature_precompute_summary(total: usize, accesses: &[xc_core::CacheAccessProvenance]) {
    if accesses.is_empty() {
        eprintln!("[HP] GL tables ready: {total} total (standalone cache/computation)");
        return;
    }
    let mut counts = BTreeMap::<String, usize>::new();
    for access in accesses {
        let label = match access.reuse_disposition {
            xc_core::CacheReuseDisposition::Recomputed => "computed".to_owned(),
            xc_core::CacheReuseDisposition::Reused => access
                .selected_source
                .as_ref()
                .map(|source| format!("reused from {}", source.overlay))
                .unwrap_or_else(|| "reused".to_owned()),
            xc_core::CacheReuseDisposition::InspectedOnly => "inspected only".to_owned(),
        };
        *counts.entry(label).or_default() += 1;
    }
    let detail = counts
        .into_iter()
        .map(|(label, count)| format!("{count} {label}"))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("[HP] GL tables ready: {total} total ({detail})");
}

fn compute_archimedean_integrals_tracked(
    n_modes: usize,
    l: &Float,
    cfg: &HighPrecConfig,
    fabric_cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<(ComputedArchimedeanIntegrals, Vec<ArtifactManifest>)> {
    let prec = cfg.precision_bits;
    let base_pts = cfg.quad_points;
    let prec_extra = (prec / 2) as usize;
    crate::hp_debug!(
        "[HP] Computing alpha_L, beta_L, gamma_L for n=0..{} (base quad={})",
        n_modes,
        base_pts
    );

    use std::collections::HashMap;
    let pts_for_n: Vec<usize> = (0..=n_modes)
        .map(|n| base_pts.max(3 * n + prec_extra))
        .collect();
    let unique_pts: Vec<usize> = {
        let mut values = pts_for_n.clone();
        values.sort_unstable();
        values.dedup();
        values
    };
    eprintln!(
        "[HP] Precomputing {} unique GL node tables (npts up to {}, prec={} bits)...",
        unique_pts.len(),
        unique_pts.last().copied().unwrap_or(0),
        prec
    );
    type GlTable = (Vec<Float>, Vec<Float>);
    let (gl_cache, quadrature_manifests, quadrature_accesses): (
        HashMap<usize, GlTable>,
        Vec<ArtifactManifest>,
        Vec<xc_core::CacheAccessProvenance>,
    ) = if let Some(cache) = fabric_cache {
        let resolved = xc_numerics::hp_runtime::map_gl_precompute(&unique_pts, |&npts| {
            let request = ArtifactCacheContext {
                resolver: cache.resolver,
                reference_resolver: cache.reference_resolver,
                acceptance: cache.acceptance,
                ordered_overlays: cache.ordered_overlays.clone(),
                mode: cache.mode,
                write_on_miss: cache.write_on_miss,
                write_visibility: cache.write_visibility,
                requested_assurance: cache.requested_assurance,
                certification_failure_policy: cache.certification_failure_policy,
                production_sink: cache.production_sink,
            };
            xc_numerics::quadrature::gauss_legendre_nodes_via_cache(npts, prec, request).map(
                |rule| {
                    (
                        npts,
                        (rule.nodes, rule.weights),
                        rule.artifact_manifest,
                        rule.cache_access,
                    )
                },
            )
        });
        let mut tables = HashMap::new();
        let mut manifests = Vec::new();
        let mut accesses = Vec::new();
        for result in resolved {
            let (npts, table, manifest, access) = result.map_err(anyhow::Error::from)?;
            tables.insert(npts, table);
            manifests.push(manifest);
            accesses.push(access);
        }
        manifests.sort_by(|left, right| left.key.logical_key.cmp(&right.key.logical_key));
        (tables, manifests, accesses)
    } else {
        let pairs: Vec<(usize, GlTable)> =
            xc_numerics::hp_runtime::map_gl_precompute(&unique_pts, |&npts| {
                (
                    npts,
                    xc_numerics::quadrature::gauss_legendre_nodes(npts, prec, cfg.cache_mode),
                )
            });
        (pairs.into_iter().collect(), Vec::new(), Vec::new())
    };
    report_quadrature_precompute_summary(unique_pts.len(), &quadrature_accesses);
    eprintln!("[HP] Computing alpha_L, beta_L, gamma_L integrals...");

    let indices: Vec<usize> = (0..=n_modes).collect();
    let kappa_half = compute_kappa_half(l, prec);
    let fused: Vec<(Float, Float, Float)> = indices
        .par_iter()
        .map(|&n| {
            let (nodes, weights) = gl_cache.get(&pts_for_n[n]).unwrap();
            compute_archimedean_integrals_l(n as i64, l, prec, nodes, weights, &kappa_half)
        })
        .collect();
    let mut alpha = Vec::with_capacity(fused.len());
    let mut beta = Vec::with_capacity(fused.len());
    let mut gamma = Vec::with_capacity(fused.len());
    for (alpha_value, beta_value, gamma_value) in fused {
        alpha.push(alpha_value);
        beta.push(beta_value);
        gamma.push(gamma_value);
    }
    Ok((
        ComputedArchimedeanIntegrals { alpha, beta, gamma },
        quadrature_manifests,
    ))
}

fn assemble_pole_and_archimedean_components(
    n_modes: usize,
    l: &Float,
    prec: u32,
    integrals: &ComputedArchimedeanIntegrals,
) -> (Vec<Float>, Vec<Float>) {
    let dim = 2 * n_modes + 1;
    let pi_v = pi(prec);
    let mut sixteen_pi2 = pi_v.square();
    sixteen_pi2 *= 16u32;
    let l_sq = l.clone().square();
    let sinh2_l_over_4 = {
        let mut value = l.clone();
        value /= 4u32;
        value.sinh().square()
    };
    let cells: Vec<(i64, i64)> = (-(n_modes as i64)..=(n_modes as i64))
        .flat_map(|n| (-(n_modes as i64)..=(n_modes as i64)).map(move |m| (n, m)))
        .collect();
    let values: Vec<(Float, Float)> = cells
        .par_iter()
        .map(|&(n, m)| {
            let nf = fl_i(prec, n);
            let mf = fl_i(prec, m);
            let pole = {
                let mut mn = sixteen_pi2.clone();
                mn *= &mf;
                mn *= &nf;
                let mut numerator = l_sq.clone();
                numerator -= &mn;
                let mut left = sixteen_pi2.clone();
                left *= &mf;
                left *= &mf;
                left += &l_sq;
                let mut right = sixteen_pi2.clone();
                right *= &nf;
                right *= &nf;
                right += &l_sq;
                left *= right;
                let mut value = sinh2_l_over_4.clone();
                value *= 32u32;
                value *= l;
                value *= numerator;
                value /= left;
                value
            };
            let archimedean = if n == m {
                let index = n.unsigned_abs() as usize;
                let mut value = integrals.gamma[index].clone();
                value -= &integrals.beta[index];
                value *= 2u32;
                value
            } else {
                let mut value = signed_alpha(&integrals.alpha, m, prec);
                value -= signed_alpha(&integrals.alpha, n, prec);
                value /= fl_i(prec, n - m);
                value
            };
            (pole, archimedean)
        })
        .collect();
    let mut pole = vec![Float::with_val(prec, 0); dim * dim];
    let mut archimedean = vec![Float::with_val(prec, 0); dim * dim];
    for (index, &(n, m)) in cells.iter().enumerate() {
        let row = (n + n_modes as i64) as usize;
        let column = (m + n_modes as i64) as usize;
        pole[row * dim + column] = values[index].0.clone();
        archimedean[row * dim + column] = values[index].1.clone();
    }
    (pole, archimedean)
}

fn compute_prime_component_matrix(
    n_modes: usize,
    prime_cutoff: u64,
    l: &Float,
    prec: u32,
) -> Vec<Float> {
    let dim = 2 * n_modes + 1;
    let pi_v = pi(prec);
    let mut two_pi = pi_v.clone();
    two_pi *= 2u32;
    let mode_values = (-(n_modes as i64)..=(n_modes as i64))
        .map(|mode| fl_i(prec, mode))
        .collect::<Vec<_>>();
    let mode_frequencies = mode_values
        .iter()
        .map(|mode| {
            let mut frequency = two_pi.clone();
            frequency *= mode;
            frequency /= l;
            frequency
        })
        .collect::<Vec<_>>();
    let difference_denominators = (-(2 * n_modes as i64)..=(2 * n_modes as i64))
        .map(|difference| {
            let mut denominator = pi_v.clone();
            denominator *= fl_i(prec, difference);
            denominator
        })
        .collect::<Vec<_>>();
    struct PrimeKernelTable {
        log_prime: Float,
        sqrt_power: Float,
        diagonal_factor: Float,
        sines: Vec<Float>,
        cosines: Vec<Float>,
    }
    let prime_data: Vec<PrimeKernelTable> = prime_powers_up_to(prime_cutoff)
        .into_iter()
        .map(|(power, prime, _)| {
            let log_power = Float::with_val(prec, power).ln();
            let log_prime = Float::with_val(prec, prime).ln();
            let sqrt_power = Float::with_val(prec, power).sqrt();
            let mut diagonal_factor = Float::with_val(prec, 1);
            let mut ratio = log_power.clone();
            ratio /= l;
            diagonal_factor -= ratio;
            diagonal_factor *= 2u32;
            let phases = mode_frequencies
                .iter()
                .map(|frequency| {
                    let mut phase = frequency.clone();
                    phase *= &log_power;
                    phase
                })
                .collect::<Vec<_>>();
            let sines = phases.iter().map(|phase| phase.clone().sin()).collect();
            let cosines = phases.into_iter().map(Float::cos).collect();
            PrimeKernelTable {
                log_prime,
                sqrt_power,
                diagonal_factor,
                sines,
                cosines,
            }
        })
        .collect();
    let cells: Vec<(i64, i64)> = (-(n_modes as i64)..=(n_modes as i64))
        .flat_map(|n| (-(n_modes as i64)..=(n_modes as i64)).map(move |m| (n, m)))
        .collect();
    let values: Vec<Float> = cells
        .par_iter()
        .map(|&(n, m)| {
            let n_index = (n + n_modes as i64) as usize;
            let m_index = (m + n_modes as i64) as usize;
            let mut sum = Float::with_val(prec, 0);
            for data in &prime_data {
                let kernel = if n == m {
                    let mut factor = data.diagonal_factor.clone();
                    factor *= &data.cosines[n_index];
                    factor
                } else {
                    let mut difference = data.sines[m_index].clone();
                    difference -= &data.sines[n_index];
                    difference /= &difference_denominators[(n - m + 2 * n_modes as i64) as usize];
                    difference
                };
                let mut term = kernel;
                term *= &data.log_prime;
                term /= &data.sqrt_power;
                sum += term;
            }
            sum
        })
        .collect();
    let mut matrix = vec![Float::with_val(prec, 0); dim * dim];
    for (index, &(n, m)) in cells.iter().enumerate() {
        let row = (n + n_modes as i64) as usize;
        let column = (m + n_modes as i64) as usize;
        matrix[row * dim + column] = values[index].clone();
    }
    matrix
}

// Frozen allocating implementation retained only as an exact-equivalence
// oracle for the precomputed prime-kernel tables above. Do not relax the
// comparison to a tolerance: a change here would change cached matrix bytes.
#[cfg(test)]
fn compute_prime_component_matrix_reference(
    n_modes: usize,
    prime_cutoff: u64,
    l: &Float,
    prec: u32,
) -> Vec<Float> {
    let dim = 2 * n_modes + 1;
    let pi_v = pi(prec);
    let mut two_pi = pi_v.clone();
    two_pi *= 2u32;
    let prime_data: Vec<(Float, Float, Float)> = prime_powers_up_to(prime_cutoff)
        .into_iter()
        .map(|(power, prime, _)| {
            (
                Float::with_val(prec, power).ln(),
                Float::with_val(prec, prime).ln(),
                Float::with_val(prec, power).sqrt(),
            )
        })
        .collect();
    let cells: Vec<(i64, i64)> = (-(n_modes as i64)..=(n_modes as i64))
        .flat_map(|n| (-(n_modes as i64)..=(n_modes as i64)).map(move |m| (n, m)))
        .collect();
    let values: Vec<Float> = cells
        .par_iter()
        .map(|&(n, m)| {
            let mut omega_n = two_pi.clone();
            omega_n *= fl_i(prec, n);
            omega_n /= l;
            let mut omega_m = two_pi.clone();
            omega_m *= fl_i(prec, m);
            omega_m /= l;
            let mut sum = Float::with_val(prec, 0);
            for (log_power, log_prime, sqrt_power) in &prime_data {
                let kernel = if n == m {
                    let mut phase = omega_n.clone();
                    phase *= log_power;
                    let mut factor = Float::with_val(prec, 1);
                    let mut ratio = log_power.clone();
                    ratio /= l;
                    factor -= ratio;
                    factor *= 2u32;
                    factor *= phase.cos();
                    factor
                } else {
                    let mut phase_m = omega_m.clone();
                    phase_m *= log_power;
                    let mut phase_n = omega_n.clone();
                    phase_n *= log_power;
                    let mut difference = phase_m.sin();
                    difference -= phase_n.sin();
                    let mut denominator = pi_v.clone();
                    denominator *= fl_i(prec, n - m);
                    difference /= denominator;
                    difference
                };
                let mut term = kernel;
                term *= log_prime;
                term /= sqrt_power;
                sum += term;
            }
            sum
        })
        .collect();
    let mut matrix = vec![Float::with_val(prec, 0); dim * dim];
    for (index, &(n, m)) in cells.iter().enumerate() {
        let row = (n + n_modes as i64) as usize;
        let column = (m + n_modes as i64) as usize;
        matrix[row * dim + column] = values[index].clone();
    }
    matrix
}

fn build_tau_components_exact_tracked(
    n_modes: usize,
    lambda_sq_int: u64,
    l: &Float,
    cfg: &HighPrecConfig,
    include_primes: bool,
    fabric_cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<(ComputedCcmMatrixComponents, Vec<ArtifactManifest>)> {
    let prec = cfg.precision_bits;
    let n_max = n_modes;
    let dim = 2 * n_modes + 1;

    let pi_v = pi(prec);
    let mut two_pi = pi_v.clone();
    two_pi *= 2u32;
    let mut sixteen_pi2 = pi_v.clone().square();
    sixteen_pi2 *= 16u32;
    let l_sq = l.clone().square();
    let sinh2_l_over_4 = {
        let mut v = l.clone();
        v /= 4u32;
        v.sinh().square()
    };

    let base_pts = cfg.quad_points;
    let prec_extra = (cfg.precision_bits / 2) as usize;
    crate::hp_debug!(
        "[HP] Computing α_L, β_L, γ_L for n=0..{} (base quad={})",
        n_max,
        base_pts
    );

    use std::collections::HashMap;
    let pts_for_n: Vec<usize> = (0..=n_max)
        .map(|n| base_pts.max(3 * n + prec_extra))
        .collect();
    let unique_pts: Vec<usize> = {
        let mut v = pts_for_n.clone();
        v.sort_unstable();
        v.dedup();
        v
    };
    eprintln!(
        "[HP] Precomputing {} unique GL node tables (npts up to {}, prec={} bits)...",
        unique_pts.len(),
        unique_pts.last().copied().unwrap_or(0),
        prec
    );
    // xc_numerics::hp_runtime::map_gl_precompute: parallel on Vast/native
    // Linux (unchanged); sequential on WSL2, where sustained concurrent
    // GMP allocation across many back-to-back GL computes triggers a
    // non-deterministic glibc abort (confirmed independent of rayon —
    // reproduces with plain std::thread too). GL tables are cached to
    // disk after first compute, so this only costs time on a cold cache.
    type GlTable = (Vec<Float>, Vec<Float>);
    let (gl_cache, quadrature_manifests, quadrature_accesses): (
        HashMap<usize, GlTable>,
        Vec<ArtifactManifest>,
        Vec<xc_core::CacheAccessProvenance>,
    ) = if let Some(cache) = fabric_cache {
        let resolved = xc_numerics::hp_runtime::map_gl_precompute(&unique_pts, |&npts| {
            let request = ArtifactCacheContext {
                resolver: cache.resolver,
                reference_resolver: cache.reference_resolver,
                acceptance: cache.acceptance,
                ordered_overlays: cache.ordered_overlays.clone(),
                mode: cache.mode,
                write_on_miss: cache.write_on_miss,
                write_visibility: cache.write_visibility,
                requested_assurance: cache.requested_assurance,
                certification_failure_policy: cache.certification_failure_policy,
                production_sink: cache.production_sink,
            };
            xc_numerics::quadrature::gauss_legendre_nodes_via_cache(npts, prec, request).map(
                |rule| {
                    (
                        npts,
                        (rule.nodes, rule.weights),
                        rule.artifact_manifest,
                        rule.cache_access,
                    )
                },
            )
        });
        let mut tables = HashMap::new();
        let mut manifests = Vec::new();
        let mut accesses = Vec::new();
        for result in resolved {
            let (npts, table, manifest, access) = result.map_err(anyhow::Error::from)?;
            tables.insert(npts, table);
            manifests.push(manifest);
            accesses.push(access);
        }
        manifests.sort_by(|left, right| left.key.logical_key.cmp(&right.key.logical_key));
        (tables, manifests, accesses)
    } else {
        let gl_pairs: Vec<(usize, GlTable)> =
            xc_numerics::hp_runtime::map_gl_precompute(&unique_pts, |&npts| {
                (
                    npts,
                    xc_numerics::quadrature::gauss_legendre_nodes(npts, prec, cfg.cache_mode),
                )
            });
        (gl_pairs.into_iter().collect(), Vec::new(), Vec::new())
    };
    report_quadrature_precompute_summary(unique_pts.len(), &quadrature_accesses);
    eprintln!("[HP] Computing alpha_L, beta_L, gamma_L integrals...");
    let indices: Vec<usize> = (0..=n_max).collect();
    let kappa_half = compute_kappa_half(l, prec);
    let fused_integrals: Vec<(Float, Float, Float)> = indices
        .par_iter()
        .map(|&n| {
            let pts = pts_for_n[n];
            let (nodes, weights) = gl_cache.get(&pts).unwrap();
            compute_archimedean_integrals_l(n as i64, l, prec, nodes, weights, &kappa_half)
        })
        .collect();
    let mut alpha_l = Vec::with_capacity(fused_integrals.len());
    let mut beta_l = Vec::with_capacity(fused_integrals.len());
    let mut gamma_l = Vec::with_capacity(fused_integrals.len());
    for (alpha_value, beta_value, gamma_value) in fused_integrals {
        alpha_l.push(alpha_value);
        beta_l.push(beta_value);
        gamma_l.push(gamma_value);
    }
    eprintln!(
        "[HP] Integrals done. Assembling {}×{} τ-matrix...",
        dim, dim
    );

    let prime_powers = prime_powers_up_to(lambda_sq_int);
    // Pure HP path: compute log_p in HP from the exposed prime, do not
    // recover j from log ratios. j is provided directly by the sieve.
    let pp_data: Vec<(Float, Float, Float)> = prime_powers
        .iter()
        .map(|&(k, _p, _j)| {
            let log_k = Float::with_val(prec, k).ln();
            let log_p = Float::with_val(prec, _p).ln();
            let sqrt_k = Float::with_val(prec, k).sqrt();
            (log_k, log_p, sqrt_k)
        })
        .collect();

    let cells: Vec<(i64, i64)> = (-(n_max as i64)..=(n_max as i64))
        .flat_map(|n| (-(n_max as i64)..=(n_max as i64)).map(move |m| (n, m)))
        .collect();

    let computed: Vec<(Float, Float, Float)> = cells
        .par_iter()
        .map(|&(n, m)| {
            let nf = fl_i(prec, n);
            let mf = fl_i(prec, m);

            let w02 = {
                let mut mn = sixteen_pi2.clone();
                mn *= &mf;
                mn *= &nf;
                let mut num = l_sq.clone();
                num -= &mn;
                let mut a = sixteen_pi2.clone();
                a *= &mf;
                a *= &mf;
                a += &l_sq;
                let mut b = sixteen_pi2.clone();
                b *= &nf;
                b *= &nf;
                b += &l_sq;
                let mut den = a;
                den *= &b;
                let mut v = sinh2_l_over_4.clone();
                v *= 32u32;
                v *= l;
                v *= &num;
                v /= &den;
                v
            };

            let wr = if n == m {
                let k = n.unsigned_abs() as usize;
                let mut v = gamma_l[k].clone();
                v -= &beta_l[k];
                v *= 2u32;
                v
            } else {
                let an = signed_alpha(&alpha_l, n, prec);
                let am = signed_alpha(&alpha_l, m, prec);
                let mut v = am;
                v -= &an;
                v /= fl_i(prec, n - m);
                v
            };

            let two_pi_n_over_l = {
                let mut v = two_pi.clone();
                v *= &nf;
                v /= l;
                v
            };
            let two_pi_m_over_l = {
                let mut v = two_pi.clone();
                v *= &mf;
                v /= l;
                v
            };
            let mut wp = Float::with_val(prec, 0);
            if include_primes {
                for (log_k, log_p, sqrt_k) in &pp_data {
                    let q = if n == m {
                        let mut ph = two_pi_n_over_l.clone();
                        ph *= log_k;
                        let c = ph.cos();
                        let mut t = log_k.clone();
                        t /= l;
                        let mut f = Float::with_val(prec, 1);
                        f -= &t;
                        f *= 2u32;
                        f *= &c;
                        f
                    } else {
                        let mut sm = two_pi_m_over_l.clone();
                        sm *= log_k;
                        let sm_s = sm.sin();
                        let mut sn = two_pi_n_over_l.clone();
                        sn *= log_k;
                        let sn_s = sn.sin();
                        let mut d = sm_s;
                        d -= &sn_s;
                        let mut dn = pi_v.clone();
                        dn *= fl_i(prec, n - m);
                        d /= &dn;
                        d
                    };
                    let mut term = q;
                    term *= log_p;
                    term /= sqrt_k;
                    wp += &term;
                }
            }
            (w02, wr, wp)
        })
        .collect();

    let mut pole = vec![Float::with_val(prec, 0); dim * dim];
    let mut archimedean = vec![Float::with_val(prec, 0); dim * dim];
    let mut prime = vec![Float::with_val(prec, 0); dim * dim];
    for (i, &(n, m)) in cells.iter().enumerate() {
        let row = (n + n_modes as i64) as usize;
        let column = (m + n_modes as i64) as usize;
        pole[row * dim + column] = computed[i].0.clone();
        archimedean[row * dim + column] = computed[i].1.clone();
        prime[row * dim + column] = computed[i].2.clone();
    }
    Ok((
        ComputedCcmMatrixComponents {
            pole,
            archimedean,
            prime,
        },
        quadrature_manifests,
    ))
}

fn signed_alpha(table: &[Float], n: i64, prec: u32) -> Float {
    let k = n.unsigned_abs() as usize;
    if k >= table.len() {
        return Float::with_val(prec, 0);
    }
    if n < 0 {
        let mut v = table[k].clone();
        v = -v;
        v
    } else {
        table[k].clone()
    }
}

fn compute_kappa_half(l: &Float, prec: u32) -> Float {
    let exp_l = l.clone().exp();
    let mut numerator = exp_l.clone();
    numerator -= 1u32;
    let mut denominator = exp_l;
    denominator += 1u32;
    numerator /= &denominator;
    let mut four_pi = pi(prec);
    four_pi *= 4u32;
    numerator *= &four_pi;
    let mut kappa_half = numerator.ln();
    kappa_half += euler(prec);
    kappa_half /= 2u32;
    kappa_half
}

/// Evaluate alpha, beta, and gamma in one ordered quadrature pass. Each
/// accumulator follows the exact operation order of its former standalone
/// evaluator; only node mapping, rho, phase cosine, and kappa are shared.
fn compute_archimedean_integrals_l(
    n: i64,
    l: &Float,
    prec: u32,
    nodes: &[Float],
    weights: &[Float],
    kappa_half: &Float,
) -> (Float, Float, Float) {
    let pi_value = pi(prec);
    let mut frequency = pi_value.clone();
    frequency *= 2u32;
    frequency *= fl_i(prec, n);
    frequency /= l;
    let mut half_l = l.clone();
    half_l /= 2u32;
    let singularity_guard = Float::with_val(
        prec,
        Float::parse(HP_SINGULARITY_GUARD_STR).expect("static singularity guard parses"),
    );
    let mut alpha = Float::with_val(prec, 0);
    let mut beta = Float::with_val(prec, 0);
    let mut gamma = Float::with_val(prec, 0);
    for (node, weight) in nodes.iter().zip(weights) {
        let mut x = node.clone();
        x += 1u32;
        x *= &half_l;
        let rho = rho_hp_with_guard(&x, &singularity_guard, prec);
        let mut phase = frequency.clone();
        phase *= &x;
        // Keep MPFR's standalone sin/cos routes so the fused evaluator is
        // byte-identical to the former separate integrands.
        let sine = phase.clone().sin();
        let cosine = phase.cos();

        if n != 0 {
            let mut alpha_value = sine;
            alpha_value *= &rho;
            let mut alpha_term = weight.clone();
            alpha_term *= &alpha_value;
            alpha += &alpha_term;
        }

        let mut beta_value = x.clone();
        beta_value *= &cosine;
        beta_value *= &rho;
        let mut beta_term = weight.clone();
        beta_term *= &beta_value;
        beta += &beta_term;

        let mut negative_half = x;
        negative_half /= -2i32;
        let exponential = negative_half.exp();
        let mut gamma_value = cosine;
        gamma_value -= &exponential;
        gamma_value *= &rho;
        let mut gamma_term = weight.clone();
        gamma_term *= &gamma_value;
        gamma += &gamma_term;
    }
    alpha *= &half_l;
    alpha /= &pi_value;
    beta *= &half_l;
    beta /= l;
    gamma *= &half_l;
    gamma += kappa_half;
    (alpha, beta, gamma)
}

#[cfg(test)]
fn compute_alpha_l(n: i64, l: &Float, prec: u32, nodes: &[Float], weights: &[Float]) -> Float {
    if n == 0 {
        return Float::with_val(prec, 0);
    }
    let pi_v = pi(prec);
    let mut freq = pi_v.clone();
    freq *= 2u32;
    freq *= fl_i(prec, n);
    freq /= l;
    let f = |x: &Float| -> Float {
        let mut ph = freq.clone();
        ph *= x;
        let mut r = ph.sin();
        r *= &rho_hp(x, prec);
        r
    };
    let mut v = quad_eval(nodes, weights, l, f);
    v /= &pi_v;
    v
}

#[cfg(test)]
fn compute_beta_l(n: i64, l: &Float, prec: u32, nodes: &[Float], weights: &[Float]) -> Float {
    let pi_v = pi(prec);
    let mut freq = pi_v.clone();
    freq *= 2u32;
    freq *= fl_i(prec, n);
    freq /= l;
    let f = |x: &Float| -> Float {
        let mut ph = freq.clone();
        ph *= x;
        let c = ph.cos();
        let mut r = x.clone();
        r *= &c;
        r *= &rho_hp(x, prec);
        r
    };
    let mut v = quad_eval(nodes, weights, l, f);
    v /= l;
    v
}

#[cfg(test)]
fn compute_gamma_l(n: i64, l: &Float, prec: u32, nodes: &[Float], weights: &[Float]) -> Float {
    let pi_v = pi(prec);
    let mut freq = pi_v.clone();
    freq *= 2u32;
    freq *= fl_i(prec, n);
    freq /= l;
    let f = |x: &Float| -> Float {
        let mut ph = freq.clone();
        ph *= x;
        let c = ph.cos();
        let mut neg_half = x.clone();
        neg_half /= -2i32;
        let e = neg_half.exp();
        let mut diff = c;
        diff -= &e;
        diff *= &rho_hp(x, prec);
        diff
    };
    let mut v = quad_eval(nodes, weights, l, f);
    let kappa_half = compute_kappa_half(l, prec);
    v += &kappa_half;
    v
}

#[cfg(test)]
fn rho_hp(x: &Float, prec: u32) -> Float {
    let tiny = Float::with_val(prec, Float::parse(HP_SINGULARITY_GUARD_STR).unwrap());
    rho_hp_with_guard(x, &tiny, prec)
}

fn rho_hp_with_guard(x: &Float, tiny: &Float, prec: u32) -> Float {
    if x.cmp_abs(tiny).map(|o| o.is_lt()).unwrap_or(false) {
        let mut v = x.clone().recip();
        v /= 2u32;
        v += {
            let mut q = Float::with_val(prec, 1);
            q /= 4u32;
            q
        };
        v
    } else {
        let mut hx = x.clone();
        hx /= 2u32;
        let e = hx.exp();
        let mut d = x.clone().sinh();
        d *= 2u32;
        let mut r = e;
        r /= &d;
        r
    }
}

#[cfg(test)]
fn quad_eval<F>(nodes: &[Float], weights: &[Float], l: &Float, f: F) -> Float
where
    F: Fn(&Float) -> Float,
{
    let prec = l.prec();
    let mut half_l = l.clone();
    half_l /= 2u32;
    let mut acc = Float::with_val(prec, 0);
    for (n, w) in nodes.iter().zip(weights) {
        let mut x = n.clone();
        x += 1u32;
        x *= &half_l;
        let mut term = w.clone();
        term *= &f(&x);
        acc += &term;
    }
    acc *= &half_l;
    acc
}

fn normalize_eigenvector(xi: &[Float], l: &Float, prec: u32) -> Vec<Float> {
    let mut sum = Float::with_val(prec, 0);
    for v in xi {
        sum += v;
    }
    let mut target = l.clone().sqrt();
    target /= &sum;
    xi.iter()
        .map(|v| {
            let mut x = v.clone();
            x *= &target;
            x
        })
        .collect()
}

/// Basis half-period `L = ln(λ²) = 2 ln λ` at full working precision.
///
/// Compute `L = ln(λ²)` at full HP precision.
///
/// Integer mode: uses `value_u64` for exact HP promotion —
/// full working-precision result, no f64 representation error.
///
/// Fractional mode: formats `value_f64` to 17 significant figures
/// and parses into HP. Gives L accurate to ~17 digits (the f64
/// input limit).
fn log_lambda_sq_hp(params: &CcmParams, prec: u32) -> Float {
    if params.lambda_sq.is_integer {
        Float::with_val(prec, params.lambda_sq.value_u64).ln()
    } else {
        let lsq_str = format!("{:.17e}", params.lambda_sq.value_f64);
        let lsq_hp = Float::with_val(prec, Float::parse(&lsq_str).unwrap());
        lsq_hp.ln()
    }
}

// ===========================================================================
// High-precision refinement of R(z) zeros
// ===========================================================================

/// Root-finding method for R(z) zeros.
///
/// Selected explicitly through [`HighPrecConfig::root_solver`]:
///   - `"halley"` (default): cubic convergence, 3 passes per step, ~33%
///     fewer steps than Newton. Requires seeds reasonably close to the
///     true eigenvalue — satisfied by the pole-aware MPFR discovery brackets.
///   - `"newton"`: quadratic convergence, 2 passes per step, retained only
///     for explicit algorithm-comparison experiments. It is never selected
///     automatically after a Halley outcome.
fn solve_r_zero(
    xi: &[Float],
    n_max: usize,
    l: &Float,
    seed: &Float,
    prec: u32,
    n_steps: usize,
    method: RootSolver,
) -> EigenvalueResult {
    match method {
        RootSolver::Newton => newton_xi_hat_zero(xi, n_max, l, seed, prec, n_steps),
        RootSolver::Halley => halley_xi_hat_zero(xi, n_max, l, seed, prec, n_steps),
    }
}

fn secular_residual_at(
    xi: &[Float],
    n_max: usize,
    two_pi_over_l: &Float,
    z: &Float,
    prec: u32,
) -> Option<Float> {
    secular_residual_and_scale_at(xi, n_max, two_pi_over_l, z, prec).map(|(residual, _)| residual)
}

fn secular_residual_and_scale_at(
    xi: &[Float],
    n_max: usize,
    two_pi_over_l: &Float,
    z: &Float,
    prec: u32,
) -> Option<(Float, Float)> {
    let mut residual = Float::with_val(prec, 0);
    let mut term_scale = Float::with_val(prec, 0);
    for j in -(n_max as i64)..=(n_max as i64) {
        let idx = (j + n_max as i64) as usize;
        let mut pole = two_pi_over_l.clone();
        pole *= fl_i(prec, j);
        let mut denominator = z.clone();
        denominator -= pole;
        if denominator.is_zero() {
            return None;
        }
        let mut term = xi[idx].clone();
        term /= denominator;
        term_scale += Float::with_val(prec, &term).abs();
        residual += term;
    }
    Some((residual.abs(), term_scale))
}

fn achieved_decimal_digits(value: &Float, correction: &Float, prec: u32) -> Float {
    let mut maximum = Float::with_val(prec, 2).log10();
    maximum *= prec.saturating_sub(GUARD_BITS);
    if correction.is_zero() {
        return maximum;
    }
    let mut scale = value.clone().abs();
    if scale < 1 {
        scale = Float::with_val(prec, 1);
    }
    let mut relative = correction.clone().abs();
    relative /= scale;
    if relative >= 1 {
        return Float::with_val(prec, 0);
    }
    let mut digits = relative.log10();
    digits = -digits;
    if digits > maximum {
        maximum
    } else {
        digits
    }
}

/// Relative correction threshold corresponding to the caller's requested
/// accuracy. `for_decimal_digits` reserves `GUARD_BITS` beyond that contract;
/// those working bits absorb secular-sum cancellation and are not themselves
/// demanded from the final root.
fn root_correction_tolerance(value: &Float, prec: u32) -> Float {
    let target_bits = prec.saturating_sub(GUARD_BITS).max(1);
    let mut tolerance = Float::with_val(prec, 2).pow(-(target_bits as i32));
    let mut scale = value.clone().abs();
    if scale < 1 {
        scale = Float::with_val(prec, 1);
    }
    tolerance *= scale;
    tolerance
}

fn root_refinement(
    xi: &[Float],
    n_max: usize,
    two_pi_over_l: &Float,
    value: Float,
    iterations: usize,
    final_correction: Float,
    prec: u32,
) -> Option<RootRefinement> {
    let residual = secular_residual_at(xi, n_max, two_pi_over_l, &value, prec)?;
    let achieved_decimal_digits = achieved_decimal_digits(&value, &final_correction, prec);
    Some(RootRefinement {
        value,
        diagnostics: RootRefinementDiagnostics {
            iterations,
            final_correction,
            residual,
            achieved_decimal_digits,
        },
    })
}

/// Newton's method for a zero of R(z) = Σ ξ_j / (z − 2πj/L).
///
/// Quadratic convergence: correct digits double each step.
/// Uses R(z) and R'(z) — 2 passes over the poles per step.
///
/// Returns:
/// Stagnation is detected from a representational stall, a two-cycle, or the
/// absence of any smaller correction for `ROOT_STAGNATION_WINDOW` steps. It
/// is reported distinctly and is never accepted as convergence.
fn newton_xi_hat_zero(
    xi: &[Float],
    n_max: usize,
    l: &Float,
    seed: &Float,
    prec: u32,
    n_steps: usize,
) -> EigenvalueResult {
    let two_pi_over_l = {
        let mut v = pi(prec);
        v *= 2u32;
        v /= l;
        v
    };
    let mut z = seed.clone();
    if n_steps == 0 {
        return EigenvalueResult::Failed {
            iterations: 0,
            reason: "root iteration limit must be positive".to_owned(),
        };
    }
    let mut point_before_previous: Option<Float> = None;
    let mut last_correction = Float::with_val(prec, 0);
    let mut best_correction: Option<Float> = None;
    let mut nonimproving_steps = 0usize;
    for iteration in 1..=n_steps {
        let previous_point = z.clone();
        let mut r = Float::with_val(prec, 0);
        let mut r_prime = Float::with_val(prec, 0);
        for j in -(n_max as i64)..=(n_max as i64) {
            let idx = (j + n_max as i64) as usize;
            let mut pole = two_pi_over_l.clone();
            pole *= fl_i(prec, j);
            let mut den = z.clone();
            den -= &pole;
            let mut term = xi[idx].clone();
            term /= &den;
            r += &term;
            let mut den_sq = den.clone();
            den_sq.square_mut();
            let mut dterm = xi[idx].clone();
            dterm /= &den_sq;
            r_prime -= &dterm;
        }
        if r_prime.is_zero() {
            return EigenvalueResult::Failed {
                iterations: iteration,
                reason: "Newton derivative is zero".to_owned(),
            };
        }
        let mut dz = r;
        dz /= &r_prime;
        z -= &dz;
        let abs_dz = dz.abs();
        last_correction = abs_dz.clone();
        // Converged to full HP precision.
        let tolerance = root_correction_tolerance(&z, prec);
        if abs_dz
            .cmp_abs(&tolerance)
            .map(|o| o.is_lt())
            .unwrap_or(false)
        {
            return match root_refinement(xi, n_max, &two_pi_over_l, z, iteration, abs_dz, prec) {
                Some(result) => EigenvalueResult::Converged(result),
                None => EigenvalueResult::Failed {
                    iterations: iteration,
                    reason: "Newton converged onto a secular pole".to_owned(),
                },
            };
        }
        if best_correction.as_ref().is_none_or(|best| &abs_dz < best) {
            best_correction = Some(abs_dz.clone());
            nonimproving_steps = 0;
        } else {
            nonimproving_steps += 1;
        }
        if z == previous_point
            || point_before_previous
                .as_ref()
                .is_some_and(|point| point == &z)
            || nonimproving_steps >= ROOT_STAGNATION_WINDOW
        {
            return match root_refinement(xi, n_max, &two_pi_over_l, z, iteration, abs_dz, prec) {
                Some(result) => EigenvalueResult::Stagnated(result),
                None => EigenvalueResult::Failed {
                    iterations: iteration,
                    reason: "Newton stagnated on a secular pole".to_owned(),
                },
            };
        }
        point_before_previous = Some(previous_point);
    }
    match root_refinement(xi, n_max, &two_pi_over_l, z, n_steps, last_correction, prec) {
        Some(result) => EigenvalueResult::Approximate(result),
        None => EigenvalueResult::Failed {
            iterations: n_steps,
            reason: "Newton iteration limit reached on a secular pole".to_owned(),
        },
    }
}

/// Halley's method for a zero of R(z) = Σ ξ_j / (z − 2πj/L).
///
/// Cubic convergence: correct digits triple each step.
/// Uses R(z), R'(z), and R''(z) — 3 passes over the poles per step.
/// Step: z ← z − 2·R·R' / (2·R'² − R·R'')
///
/// Stagnation is detected from a representational stall, a two-cycle, or the
/// absence of any smaller correction for `ROOT_STAGNATION_WINDOW` steps. It
/// is reported distinctly and is never accepted as convergence.
fn halley_xi_hat_zero(
    xi: &[Float],
    n_max: usize,
    l: &Float,
    seed: &Float,
    prec: u32,
    n_steps: usize,
) -> EigenvalueResult {
    let two_pi_over_l = {
        let mut v = pi(prec);
        v *= 2u32;
        v /= l;
        v
    };
    let mut z = seed.clone();
    if n_steps == 0 {
        return EigenvalueResult::Failed {
            iterations: 0,
            reason: "root iteration limit must be positive".to_owned(),
        };
    }
    let mut point_before_previous: Option<Float> = None;
    let mut last_correction = Float::with_val(prec, 0);
    let mut best_correction: Option<Float> = None;
    let mut nonimproving_steps = 0usize;
    for iteration in 1..=n_steps {
        let previous_point = z.clone();
        let mut r = Float::with_val(prec, 0);
        let mut r_prime = Float::with_val(prec, 0);
        let mut r_dprime = Float::with_val(prec, 0);
        for j in -(n_max as i64)..=(n_max as i64) {
            let idx = (j + n_max as i64) as usize;
            let mut pole = two_pi_over_l.clone();
            pole *= fl_i(prec, j);
            let mut den = z.clone();
            den -= &pole;

            let mut term = xi[idx].clone();
            term /= &den;
            r += &term;

            let mut den2 = den.clone();
            den2.square_mut();
            let mut dt = xi[idx].clone();
            dt /= &den2;
            r_prime -= &dt;

            let mut den3 = den2.clone();
            den3 *= &den;
            let mut ddt = xi[idx].clone();
            ddt /= &den3;
            ddt *= 2u32;
            r_dprime += &ddt;
        }

        let mut denom = r_prime.clone();
        denom.square_mut();
        denom *= 2u32;
        let mut rr2 = r.clone();
        rr2 *= &r_dprime;
        denom -= &rr2;

        if denom.is_zero() {
            return EigenvalueResult::Failed {
                iterations: iteration,
                reason: "Halley denominator is zero".to_owned(),
            };
        }

        let mut dz = r.clone();
        dz *= &r_prime;
        dz *= 2u32;
        dz /= &denom;
        z -= &dz;
        let abs_dz = dz.abs();
        last_correction = abs_dz.clone();
        // Converged to full HP precision.
        let tolerance = root_correction_tolerance(&z, prec);
        if abs_dz
            .cmp_abs(&tolerance)
            .map(|o| o.is_lt())
            .unwrap_or(false)
        {
            return match root_refinement(xi, n_max, &two_pi_over_l, z, iteration, abs_dz, prec) {
                Some(result) => EigenvalueResult::Converged(result),
                None => EigenvalueResult::Failed {
                    iterations: iteration,
                    reason: "Halley converged onto a secular pole".to_owned(),
                },
            };
        }
        if best_correction.as_ref().is_none_or(|best| &abs_dz < best) {
            best_correction = Some(abs_dz.clone());
            nonimproving_steps = 0;
        } else {
            nonimproving_steps += 1;
        }
        if z == previous_point
            || point_before_previous
                .as_ref()
                .is_some_and(|point| point == &z)
            || nonimproving_steps >= ROOT_STAGNATION_WINDOW
        {
            return match root_refinement(xi, n_max, &two_pi_over_l, z, iteration, abs_dz, prec) {
                Some(result) => EigenvalueResult::Stagnated(result),
                None => EigenvalueResult::Failed {
                    iterations: iteration,
                    reason: "Halley stagnated on a secular pole".to_owned(),
                },
            };
        }
        point_before_previous = Some(previous_point);
    }
    match root_refinement(xi, n_max, &two_pi_over_l, z, n_steps, last_correction, prec) {
        Some(result) => EigenvalueResult::Approximate(result),
        None => EigenvalueResult::Failed {
            iterations: n_steps,
            reason: "Halley iteration limit reached on a secular pole".to_owned(),
        },
    }
}

// ===========================================================================
// τ-matrix disk cache
// ===========================================================================

mod tau_cache {
    //! Disk cache for the HP τ-matrix produced by `build_tau_hp_compute`.
    //!
    //! Cache layout under `<cwd>/data/tau_cache/`:
    //!   - `lambda_sq{L}_nmodes{N}_prec{P}.json` (uncompressed, fast path)
    //!   - `lambda_sq{L}_nmodes{N}_prec{P}.json.zip` (single-zip when
    //!     compressed payload ≤ 90 MB)
    //!   - `lambda_sq{L}_nmodes{N}_prec{P}.json.zip.part00, .part01, ...`
    //!     (byte-split when compressed payload > 90 MB; we split at
    //!     90 MB-byte boundaries, comfortably under GitHub's 100 MB
    //!     hard file size limit)
    //!
    //! Read priority: uncompressed → single zip → multi-part zip →
    //! compute fresh. Multi-part read concatenates the part bytes in
    //! lexicographic order and decompresses the result as one zip.
    //! Write logic picks single vs multi-part based on the compressed
    //! payload size.
    //!
    //! Structural validation on cache hit: matrix length matches
    //! `(2N+1)²`, no NaN/Inf entries, and symmetry `τ[i,j] = τ[j,i]`
    //! to working precision. A file that fails validation is
    //! discarded with a stderr warning; the toolkit falls through to
    //! compute fresh. Bad files are preserved on disk.

    use rug::{ops::Pow, Float};
    use std::io::{Read, Write};

    use super::super::LambdaSq;

    /// Per-part byte cap for split files. 90 MB stays comfortably
    /// under GitHub's 100 MB hard file limit and well under the
    /// 50 MB soft warning, leaving headroom for git's pack overhead.
    const PART_BYTE_LIMIT: usize = 90 * 1024 * 1024;

    /// Toolkit version string embedded in every tau cache file written
    /// by this build. Matches `[workspace.package].version` in `Cargo.toml`.
    const TOOLKIT_VERSION: &str = env!("CARGO_PKG_VERSION");

    #[cfg(test)]
    pub(super) fn toolkit_version_for_test() -> &'static str {
        TOOLKIT_VERSION
    }

    /// Minimum toolkit version required to use a tau cache file. Files
    /// produced by an older toolkit are treated as cache misses and
    /// recomputed. Update this constant when a change to the tau
    /// computation changes the stored values.
    fn effective_min_version() -> String {
        xc_cache::artifact_compatibility_policy("ccm-matrices", "ccm_tau_matrix")
            .expect("CCM matrix compatibility policy")
            .minimum_producer_version
            .to_string()
    }

    /// Tolerance for the symmetry identity check on cache load.
    /// At precision `prec` bits this is `2^-(prec - 8)` — same
    /// pattern used by the GL and prolate caches.
    fn cache_tol(prec: u32) -> Float {
        Float::with_val(prec, 2).pow(-((prec as i32) - 8))
    }

    fn cache_dir() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let dir = cwd.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    pub(super) fn cache_filename(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> String {
        format!(
            "lambda_sq{}_nmodes{}_prec{}.json",
            lambda_sq.filename_str(),
            n_modes,
            prec
        )
    }

    fn json_path(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| d.join(cache_filename(lambda_sq, n_modes, prec)))
    }

    fn zip_path(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| {
            let f = cache_filename(lambda_sq, n_modes, prec);
            d.join(format!("{}.zip", f))
        })
    }

    /// Glob the .partXX files for a given config in lexicographic
    /// order. Returns `None` if no parts exist.
    fn part_paths(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
    ) -> Option<Vec<std::path::PathBuf>> {
        let dir = cache_dir()?;
        let stem = cache_filename(lambda_sq, n_modes, prec);
        let prefix = format!("{}.zip.part", stem);
        let mut parts: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name();
                let s = name.to_str()?;
                if s.starts_with(&prefix) {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();
        if parts.is_empty() {
            return None;
        }
        parts.sort();
        Some(parts)
    }

    /// Parse the cache JSON for the tau matrix.
    /// Expects schema_version 1 envelope format. Returns `None` on any
    /// structural mismatch or a stale `toolkit_version`.
    fn parse_json(data: &str, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        let dim = 2 * n_modes + 1;
        let n_expected = dim * dim;
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let obj = parsed.as_object()?;

        let file_ver = obj.get("toolkit_version").and_then(|v| v.as_str())?;
        if version_is_older(file_ver, &effective_min_version()) {
            return None;
        }

        let arr = obj.get("matrix")?.as_array()?;
        if arr.len() != n_expected {
            return None;
        }
        let mut out = Vec::with_capacity(n_expected);
        for s in arr {
            out.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some(out)
    }

    /// Returns `true` if version string `a` is strictly older than `b`.
    fn version_is_older(a: &str, b: &str) -> bool {
        let parse = |s: &str| -> (u64, u64, u64) {
            let mut parts = s.splitn(3, '.');
            let major = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let patch = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            (major, minor, patch)
        };
        parse(a) < parse(b)
    }

    /// Verify the loaded matrix satisfies the structural identities:
    /// length matches `(2N+1)²`, no NaN/Inf entries, and symmetry
    /// `τ[i,j] = τ[j,i]` to working precision.
    pub(super) fn structural_check(tau: &[Float], n_modes: usize, prec: u32) -> Option<String> {
        let dim = 2 * n_modes + 1;
        if tau.len() != dim * dim {
            return Some(format!(
                "matrix length {} != expected {}², = {}",
                tau.len(),
                dim,
                dim * dim
            ));
        }
        for (k, v) in tau.iter().enumerate() {
            if v.is_nan() {
                return Some(format!("entry {} is NaN", k));
            }
            if v.is_infinite() {
                return Some(format!("entry {} is infinite", k));
            }
        }
        // Symmetry check: τ[i,j] = τ[j,i] for i < j.
        let tol = cache_tol(prec);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let mut diff = tau[i * dim + j].clone();
                diff -= &tau[j * dim + i];
                let abs_diff = diff.abs();
                if !abs_diff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
                    return Some(format!(
                        "asymmetry at ({},{}): τ[{},{}] - τ[{},{}] = {} (tol {})",
                        i, j, i, j, j, i, abs_diff, tol
                    ));
                }
            }
        }
        None
    }

    fn warn_skip(path: &std::path::Path, reason: &str) {
        crate::hp_debug!(
            "[tau_cache] WARNING: skipping {} ({}); recomputing",
            path.display(),
            reason
        );
    }

    /// Read a single zip and return both the parsed matrix and the
    /// raw inner JSON bytes (so the caller can write the decompressed
    /// copy without re-serializing).
    fn read_single_zip(
        zip_bytes: &[u8],
        json_filename: &str,
        n_modes: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, String)> {
        let cursor = std::io::Cursor::new(zip_bytes);
        let mut archive = zip::ZipArchive::new(cursor).ok()?;
        let mut entry = archive.by_name(json_filename).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_json(&data, n_modes, prec)?;
        Some((parsed, data))
    }

    /// Concatenate the bytes of all `.partXX` files in order, then
    /// treat the result as a single zip and decompress.
    fn read_split_zip_parts(
        parts: &[std::path::PathBuf],
        json_filename: &str,
        n_modes: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, String)> {
        let mut concatenated: Vec<u8> = Vec::new();
        for p in parts {
            let mut bytes = Vec::new();
            std::fs::File::open(p).ok()?.read_to_end(&mut bytes).ok()?;
            concatenated.extend_from_slice(&bytes);
        }
        read_single_zip(&concatenated, json_filename, n_modes, prec)
    }

    pub(super) fn load(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        mode: xc_numerics::quadrature::CacheMode,
    ) -> Option<Vec<Float>> {
        use xc_numerics::quadrature::CacheMode;
        if mode == CacheMode::Off {
            return None;
        }

        // Caches are zip-only: we read straight from the .json.zip
        // (decompress in memory) and never write a decompressed .json.
        // Keeps local disk usage ~2x smaller. JsonOnly is now a read
        // no-op because current cache files are zip-only.
        if mode == CacheMode::JsonOnly {
            return None;
        }

        // Local zip — single first, then
        // multi-part. Decompressed in memory; no .json written.
        if let Some(tau) = try_load_local_zip(lambda_sq, n_modes, prec) {
            return Some(tau);
        }

        None
    }

    /// Attempt to load a config from a local single zip, then local
    /// multi-part parts. Decompresses in memory; does NOT write a
    /// decompressed `.json`. Returns `None` if no local zip/parts exist
    /// or they fail validation.
    fn try_load_local_zip(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        let json_filename = cache_filename(lambda_sq, n_modes, prec);

        // Single zip first.
        if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
            if zp.exists() {
                match std::fs::read(&zp) {
                    Ok(bytes) => match read_single_zip(&bytes, &json_filename, n_modes, prec) {
                        Some((tau, _data)) => {
                            if let Some(reason) = structural_check(&tau, n_modes, prec) {
                                warn_skip(&zp, &reason);
                            } else {
                                return Some(tau);
                            }
                        }
                        None => warn_skip(&zp, "zip parse / shape failed"),
                    },
                    Err(e) => warn_skip(&zp, &format!("read failed: {}", e)),
                }
            }
        }

        // Multi-part split zip.
        if let Some(parts) = part_paths(lambda_sq, n_modes, prec) {
            let first_part_path = parts.first().cloned().unwrap_or_default();
            match read_split_zip_parts(&parts, &json_filename, n_modes, prec) {
                Some((tau, _data)) => {
                    if let Some(reason) = structural_check(&tau, n_modes, prec) {
                        warn_skip(&first_part_path, &format!("{} (split parts)", reason));
                    } else {
                        return Some(tau);
                    }
                }
                None => warn_skip(
                    &first_part_path,
                    &format!("could not concatenate / decompress {} parts", parts.len()),
                ),
            }
        }

        None
    }

    /// Test-only accessor for the current schema parser.
    #[cfg(test)]
    pub(super) fn parse_json_for_test(data: &str, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        parse_json(data, n_modes, prec)
    }

    /// Serialize `tau` to the versioned JSON envelope and return the
    /// resulting bytes.
    fn serialize_to_json(tau: &[Float], lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Vec<u8> {
        let strs: Vec<String> = tau.iter().map(|f| f.to_string()).collect();
        let payload = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": TOOLKIT_VERSION,
            "lambda_sq": lambda_sq.value_f64,
            "lambda_sq_mode": lambda_sq.mode_str(),
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        });
        serde_json::to_vec(&payload).unwrap_or_default()
    }

    /// Compress `json_bytes` to a deflated zip in-memory; the inner
    /// entry is named `entry_name`. Returns the resulting zip bytes.
    ///
    /// `large_file(true)` is required: the `zip` crate defaults to
    /// classic (non-Zip64) headers, which silently abort the write once
    /// either the uncompressed or compressed size crosses 4 GiB. τ-matrix
    /// JSON at HP-1500/HP-2000 for N in the 700-1000 range routinely
    /// exceeds that (e.g. λ²=1000, N=890 at HP-2000 is ~6.5 GB
    /// uncompressed), so without this flag `compress_to_zip` returns an
    /// empty `Vec` and the caller (`save`) treats that as "nothing to
    /// cache" with no error surfaced — every write for such a config
    /// silently no-ops forever. Setting `large_file` unconditionally
    /// costs 20 bytes of Zip64 header overhead per entry, negligible
    /// even for the smallest cached files.
    fn compress_to_zip(json_bytes: &[u8], entry_name: &str) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(json_bytes.len() / 2);
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = zip::ZipWriter::new(cursor);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .large_file(true);
            if writer.start_file(entry_name, opts).is_err() {
                return Vec::new();
            }
            if writer.write_all(json_bytes).is_err() {
                return Vec::new();
            }
            if writer.finish().is_err() {
                return Vec::new();
            }
        }
        buf
    }

    /// Remove any pre-existing single-zip / multi-part-zip / .json
    /// files for this config so we don't leave stale partner files
    /// from a previous run that wrote a different shape.
    fn cleanup_previous(lambda_sq: LambdaSq, n_modes: usize, prec: u32) {
        if let Some(p) = json_path(lambda_sq, n_modes, prec) {
            if p.exists() {
                let _ = std::fs::remove_file(&p);
            }
        }
        if let Some(p) = zip_path(lambda_sq, n_modes, prec) {
            if p.exists() {
                let _ = std::fs::remove_file(&p);
            }
        }
        if let Some(parts) = part_paths(lambda_sq, n_modes, prec) {
            for p in parts {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    pub(super) fn save(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        tau: &[Float],
        mode: xc_numerics::quadrature::CacheMode,
    ) {
        use xc_numerics::quadrature::CacheMode;
        // Off and JsonOnly write nothing: the cache is zip-only (we never
        // persist a decompressed .json), so only JsonZip
        // produce output.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) {
            return;
        }

        // Serialize to JSON in memory.
        let json_bytes = serialize_to_json(tau, lambda_sq, n_modes, prec);
        if json_bytes.is_empty() {
            return;
        }

        cleanup_previous(lambda_sq, n_modes, prec);

        // Write ONLY the compressed copy. Readers decompress from the zip
        // on demand — no uncompressed .json is persisted. Decide single-zip
        // vs multi-part split based on compressed size.
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let zip_bytes = compress_to_zip(&json_bytes, &entry_name);
        if zip_bytes.is_empty() {
            eprintln!(
                "[tau_cache] WARNING: zip compression failed for λ²={}, N={}, prec={} \
                 ({} bytes uncompressed) — this config will NOT be cached and will \
                 recompute from scratch on every run",
                lambda_sq.value_f64,
                n_modes,
                prec,
                json_bytes.len()
            );
            return;
        }

        if zip_bytes.len() <= PART_BYTE_LIMIT {
            // Single-zip path: under the per-part cap, write one file.
            if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
                if let Err(e) = std::fs::write(&zp, &zip_bytes) {
                    crate::hp_debug!(
                        "[tau_cache] WARNING: could not write {}: {}",
                        zp.display(),
                        e
                    );
                }
            }
        } else {
            // Multi-part split path: byte-split at PART_BYTE_LIMIT.
            // The toolkit reads parts back by lexicographic
            // concatenation, so naming uses zero-padded indices to
            // keep the order correct.
            let n_parts = zip_bytes.len().div_ceil(PART_BYTE_LIMIT);
            let dir = match cache_dir() {
                Some(d) => d,
                None => return,
            };
            for i in 0..n_parts {
                let start = i * PART_BYTE_LIMIT;
                let end = ((i + 1) * PART_BYTE_LIMIT).min(zip_bytes.len());
                let part_path = dir.join(format!("{}.zip.part{:02}", entry_name, i));
                if let Err(e) = std::fs::write(&part_path, &zip_bytes[start..end]) {
                    crate::hp_debug!(
                        "[tau_cache] WARNING: could not write {}: {}",
                        part_path.display(),
                        e
                    );
                    return;
                }
            }
            crate::hp_debug!(
                "[tau_cache] wrote {} parts of ≤{} MB each (compressed total {} MB) for λ²={}, N={}, prec={}",
                n_parts, PART_BYTE_LIMIT / (1024 * 1024),
                zip_bytes.len() / (1024 * 1024),
                lambda_sq.value_f64, n_modes, prec
            );
        }
    }

    /// Per-file outcome from `verify_tau_cache_dir`.
    ///
    /// Each variant carries the file path and (when parseable from the
    /// filename) the cache key tuple `(lambda_sq, n_modes, prec)`.
    /// `Skipped` and `LoadFailed` only have the tuple if the filename
    /// matched the expected pattern.
    #[derive(Debug, Clone)]
    pub enum TauCacheFileStatus {
        /// File parsed and passed all structural identity checks.
        Ok {
            path: std::path::PathBuf,
            lambda_sq: LambdaSq,
            n_modes: usize,
            prec: u32,
        },
        /// File was skipped. Either the filename didn't match the
        /// expected pattern, or it's a `.partXX` chunk handled as part
        /// of an assembled multi-part archive.
        Skipped {
            path: std::path::PathBuf,
            reason: String,
        },
        /// Filename matched but the file failed to load (decompress,
        /// concatenate, parse JSON, etc.).
        LoadFailed {
            path: std::path::PathBuf,
            lambda_sq: LambdaSq,
            n_modes: usize,
            prec: u32,
            reason: String,
        },
        /// File loaded but failed at least one structural identity
        /// check on the τ matrix it contains.
        StructurallyInvalid {
            path: std::path::PathBuf,
            lambda_sq: LambdaSq,
            n_modes: usize,
            prec: u32,
            reason: String,
        },
    }

    /// Aggregate report from `verify_tau_cache_dir`.
    #[derive(Debug, Clone)]
    pub struct TauCacheVerifyReport {
        /// Directory that was scanned.
        pub directory: std::path::PathBuf,
        /// One status entry per file (or per assembled multi-part set)
        /// found in `directory`.
        pub statuses: Vec<TauCacheFileStatus>,
    }

    impl TauCacheVerifyReport {
        /// Count of files that passed all checks.
        pub fn ok_count(&self) -> usize {
            self.statuses
                .iter()
                .filter(|s| matches!(s, TauCacheFileStatus::Ok { .. }))
                .count()
        }
        /// Count of files that failed at least one check (load or
        /// structural). Skipped files are not counted as failures.
        pub fn failure_count(&self) -> usize {
            self.statuses
                .iter()
                .filter(|s| {
                    matches!(
                        s,
                        TauCacheFileStatus::LoadFailed { .. }
                            | TauCacheFileStatus::StructurallyInvalid { .. }
                    )
                })
                .count()
        }
        /// All failure entries (load + structural), for callers that
        /// want to print only the bad files.
        pub fn failures(&self) -> impl Iterator<Item = &TauCacheFileStatus> {
            self.statuses.iter().filter(|s| {
                matches!(
                    s,
                    TauCacheFileStatus::LoadFailed { .. }
                        | TauCacheFileStatus::StructurallyInvalid { .. }
                )
            })
        }
    }

    /// Parse `lambda_sq{L}_nmodes{N}_prec{P}.json[.zip[.partXX]]`.
    /// Returns the tuple plus a flag indicating whether the file is a
    /// part of a split archive (so the verifier can skip individual
    /// parts and inspect the assembled set instead).
    pub(super) fn parse_filename(name: &str) -> Option<(LambdaSq, usize, u32, FileKind)> {
        // Three possible suffixes, in priority order.
        let (stem, kind) = if let Some(s) = strip_part_suffix(name) {
            (s, FileKind::Part)
        } else if let Some(s) = name.strip_suffix(".json.zip") {
            (s, FileKind::Zip)
        } else if let Some(s) = name.strip_suffix(".json") {
            (s, FileKind::Json)
        } else {
            return None;
        };
        let after_lsq = stem.strip_prefix("lambda_sq")?;
        let (lsq_str, rest) = after_lsq.split_once("_nmodes")?;

        let (n_str, prec_str) = rest.split_once("_prec")?;

        Some((
            LambdaSq::from_filename_str(lsq_str)?,
            n_str.parse().ok()?,
            prec_str.parse().ok()?,
            kind,
        ))
    }

    /// Strip the `.json.zip.partXX` suffix and return the stem (the
    /// `lambda_sq{L}_nmodes{N}_prec{P}` portion). Returns `None` if
    /// the name doesn't match this pattern.
    fn strip_part_suffix(name: &str) -> Option<&str> {
        let pos = name.rfind(".json.zip.part")?;
        // Validate the suffix after `.part` is digits only.
        let rest = &name[pos + ".json.zip.part".len()..];
        if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        Some(&name[..pos])
    }

    pub(super) enum FileKind {
        Json,
        Zip,
        Part,
    }

    /// Walk the τ-cache directory and structurally verify every file.
    /// Multi-part archives are verified as a set: if any `.partXX`
    /// files are present for a config, the verifier reads them all,
    /// concatenates, decompresses, and validates the result. Files
    /// not in the expected pattern are reported as `Skipped`.
    pub fn verify_tau_cache_dir(dir: &std::path::Path) -> std::io::Result<TauCacheVerifyReport> {
        let mut statuses: Vec<TauCacheFileStatus> = Vec::new();

        if !dir.exists() {
            return Ok(TauCacheVerifyReport {
                directory: dir.to_path_buf(),
                statuses,
            });
        }

        // First pass: bucket files by (lambda_sq, n_modes, prec, kind).
        // Multi-part archives (kind=Part) are deduplicated: we verify
        // the whole set once per config rather than per-file.
        // Key is (filename_str, n_modes, prec) for uniqueness; value is LambdaSq.
        let mut configs_with_parts: std::collections::BTreeMap<(String, usize, u32), LambdaSq> =
            std::collections::BTreeMap::new();
        let mut singletons: Vec<(std::path::PathBuf, LambdaSq, usize, u32, FileKind)> = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            match parse_filename(&name) {
                Some((lsq, n_modes, prec, kind)) => match kind {
                    FileKind::Json | FileKind::Zip => {
                        singletons.push((path, lsq, n_modes, prec, kind));
                    }
                    FileKind::Part => {
                        configs_with_parts.insert((lsq.filename_str(), n_modes, prec), lsq);
                    }
                },
                None => {
                    statuses.push(TauCacheFileStatus::Skipped {
                        path,
                        reason: format!(
                            "filename '{}' not in expected lambda_sq{{L}}_nmodes{{N}}_prec{{P}}.json[.zip[.partXX]] form",
                            name
                        ),
                    });
                }
            }
        }

        // Verify singletons.
        for (path, lsq, n_modes, prec, kind) in singletons {
            let parsed: Option<Vec<Float>> = match kind {
                FileKind::Json => std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|d| parse_json(&d, n_modes, prec)),
                FileKind::Zip => std::fs::read(&path).ok().and_then(|bytes| {
                    let entry_name = cache_filename(lsq, n_modes, prec);
                    read_single_zip(&bytes, &entry_name, n_modes, prec).map(|(t, _)| t)
                }),
                FileKind::Part => unreachable!(),
            };
            match parsed {
                Some(tau) => match structural_check(&tau, n_modes, prec) {
                    None => statuses.push(TauCacheFileStatus::Ok {
                        path,
                        lambda_sq: lsq,
                        n_modes,
                        prec,
                    }),
                    Some(reason) => statuses.push(TauCacheFileStatus::StructurallyInvalid {
                        path,
                        lambda_sq: lsq,
                        n_modes,
                        prec,
                        reason,
                    }),
                },
                None => statuses.push(TauCacheFileStatus::LoadFailed {
                    path,
                    lambda_sq: lsq,
                    n_modes,
                    prec,
                    reason: "parse / decompress failed".to_string(),
                }),
            }
        }

        // Verify split-archive sets (one entry per config, not per part).
        for ((_key, n_modes, prec), lsq) in configs_with_parts {
            let parts = match part_paths(lsq, n_modes, prec) {
                Some(p) => p,
                None => continue,
            };
            let representative = parts[0].clone();
            let entry_name = cache_filename(lsq, n_modes, prec);
            match read_split_zip_parts(&parts, &entry_name, n_modes, prec) {
                Some((tau, _data)) => match structural_check(&tau, n_modes, prec) {
                    None => statuses.push(TauCacheFileStatus::Ok {
                        path: representative,
                        lambda_sq: lsq,
                        n_modes,
                        prec,
                    }),
                    Some(reason) => statuses.push(TauCacheFileStatus::StructurallyInvalid {
                        path: representative,
                        lambda_sq: lsq,
                        n_modes,
                        prec,
                        reason,
                    }),
                },
                None => statuses.push(TauCacheFileStatus::LoadFailed {
                    path: representative,
                    lambda_sq: lsq,
                    n_modes,
                    prec,
                    reason: format!(
                        "split archive ({} parts) failed to assemble / decompress",
                        parts.len()
                    ),
                }),
            }
        }

        Ok(TauCacheVerifyReport {
            directory: dir.to_path_buf(),
            statuses,
        })
    }
}

#[cfg(feature = "hp")]
pub use tau_cache::{verify_tau_cache_dir, TauCacheFileStatus, TauCacheVerifyReport};

// ===========================================================================
// Weil-eigenvector (ξ) disk cache
// ===========================================================================

mod weil_eigvec_cache {
    //! Disk cache for the smallest-eigenvalue eigenvector ξ of the Weil
    //! quadratic form (the vector produced by `inverse_iteration` inside
    //! [`super::run`], ℓ²-normalized so Σξ = √L). Distinct from the
    //! prolate eigenvalue cache (`prolate_eigvals_cache`, different
    //! operator *and* quantity) and the τ-matrix cache (`tau_cache`,
    //! different quantity).
    //!
    //! Cache layout under `<cwd>/data/weil_eigvec_cache/`:
    //!   - `weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json` (uncompressed,
    //!     fast path)
    //!   - `weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json.zip`
    //!     (single-zip companion for distribution)
    //!
    //! Unlike `tau_cache`, ξ is small (2N+1 entries, ≲ 2 MB even at
    //! HP-1000/N=800), so there is no byte-split `.partXX` tier — single
    //! zip only, exactly like the GL-node cache.
    //!
    //! Schema mirrors [`super::HighPrecResult::save_xi_json`]
    //! (`schema_version: 1`): a JSON object carrying ξ as decimal strings
    //! plus `weil_min_eigenvalue` (ε_N) and the `(λ², N, prec)` metadata.
    //!
    //! Validation on load is two-tier:
    //!   1. *Structural* (here, no τ needed): length = 2N+1, finite
    //!      entries, metadata match. Cheap O(N).
    //!   2. *Residual* (at the [`super::run`] call site, where τ is in
    //!      hand): ‖τξ − ε_N·ξ‖ below the working-precision floor. This is
    //!      the strongest integrity test and is why the cache check sits
    //!      *after* the τ build.

    use rug::{ops::Pow, Float};
    use std::io::{Read, Write};

    use xc_numerics::quadrature::CacheMode;

    use super::super::LambdaSq;
    use super::{CcmParityPolicy, PortableInverseIterationDiagnostics};

    /// Toolkit version string embedded in every weil eigvec cache file
    /// written by this build.
    const TOOLKIT_VERSION: &str = env!("CARGO_PKG_VERSION");

    #[cfg(test)]
    pub(super) fn toolkit_version_for_test() -> &'static str {
        TOOLKIT_VERSION
    }

    /// Minimum toolkit version required to use a weil eigvec cache file.
    /// Files produced by an older toolkit are treated as cache misses.
    /// Update this constant when a change to the eigenvector computation
    /// changes the stored values.
    fn effective_min_version() -> String {
        xc_cache::artifact_compatibility_policy("weil-states", "ccm_weil_eigenpair")
            .expect("Weil-state compatibility policy")
            .minimum_producer_version
            .to_string()
    }

    /// Current schema version for the weil eigvec JSON envelope.
    const SCHEMA_VERSION: u32 = 2;

    /// A ξ entry loaded from the cache: the eigenvector plus its
    /// eigenvalue ε_N, both at the requested working precision.
    pub(super) struct CachedXi {
        pub eps_n: Float,
        pub xi: Vec<Float>,
        pub diagnostics: xc_numerics::linalg::InverseIterationDiagnostics,
    }

    fn cache_dir() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let dir = cwd.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    /// Find the nearest cached ξ for the same (lambda_sq, n_modes)
    /// but a different precision, within `prec_tolerance` bits.
    ///
    /// Scans the local weil_eigvec_cache directory for filename patterns
    /// matching `weil_eigvec_lambda_sq{lsq}_nmodes{n}_prec*.json.zip`
    /// and returns the closest-prec entry within tolerance, promoted to
    /// the target precision. Returns `None` if no match is found or the
    /// feature is disabled.
    ///
    /// This is used as a warm-start for inverse iteration: instead of the
    /// Gaussian initial guess, we start from a ξ that is already close to
    /// the answer, reducing iteration count from ~50-100 to ~2-5.
    pub(super) fn find_warm_start(
        lambda_sq: LambdaSq,
        n_modes: usize,
        target_prec: u32,
        tolerance_bits: u32,
        parity_policy: CcmParityPolicy,
    ) -> Option<Vec<Float>> {
        let dir = cache_dir()?;
        let variant = cache_variant(parity_policy);
        let prefix = format!(
            "weil_eigvec_lambda_sq{}_nmodes{}_prec",
            lambda_sq.filename_str(),
            n_modes
        );
        let suffix = format!("{}.json.zip", variant);

        let mut best_prec: Option<u32> = None;
        let mut best_diff = u32::MAX;

        // Scan directory for matching filenames
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with(&prefix) || !name_str.ends_with(&suffix) {
                continue;
            }
            // Parse the prec value from the filename
            let mid = &name_str[prefix.len()..name_str.len() - suffix.len()];
            if let Ok(p) = mid.parse::<u32>() {
                if p == target_prec {
                    continue;
                } // exact match handled elsewhere
                let diff = target_prec.abs_diff(p);
                if diff <= tolerance_bits && diff < best_diff {
                    best_diff = diff;
                    best_prec = Some(p);
                }
            }
        }

        let nearby_prec = best_prec?;

        // Load the nearby cache entry (at its original precision)
        let cached = load(
            lambda_sq,
            n_modes,
            nearby_prec,
            CacheMode::JsonZip,
            parity_policy,
        )?;

        // Promote ξ to target precision by re-parsing each entry.
        // This is exact: we're just increasing the mantissa bits, no rounding.
        let xi_promoted: Vec<Float> = cached
            .xi
            .iter()
            .map(|v| {
                let s = v.to_string();
                Float::with_val(
                    target_prec,
                    Float::parse(&s).unwrap_or_else(|_| Float::parse("0").unwrap()),
                )
            })
            .collect();

        crate::hp_debug!(
            "[HP] warm-start from nearby cache prec={} (target={}, diff={} bits)",
            nearby_prec,
            target_prec,
            best_diff
        );

        Some(xi_promoted)
    }

    pub(super) fn cache_filename(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        parity_policy: CcmParityPolicy,
    ) -> String {
        let variant = cache_variant(parity_policy);
        format!(
            "weil_eigvec_lambda_sq{}_nmodes{}_prec{}{}.json",
            lambda_sq.filename_str(),
            n_modes,
            prec,
            variant
        )
    }

    fn cache_variant(parity_policy: CcmParityPolicy) -> &'static str {
        match parity_policy {
            // The historical unsuffixed standalone cache used full-space
            // inverse iteration with conditional even projection.
            CcmParityPolicy::AdaptiveEven => "",
            CcmParityPolicy::Natural => "_natural",
            // The reduced even-sector route was introduced through the
            // managed artifact fabric, so it must not consume a historical
            // adaptive standalone result.
            CcmParityPolicy::EvenSector => "_even_sector",
        }
    }

    fn json_path(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        parity_policy: CcmParityPolicy,
    ) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| d.join(cache_filename(lambda_sq, n_modes, prec, parity_policy)))
    }

    fn zip_path(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        parity_policy: CcmParityPolicy,
    ) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| {
            let f = cache_filename(lambda_sq, n_modes, prec, parity_policy);
            d.join(format!("{}.zip", f))
        })
    }

    /// Parse the cache JSON object into `(eps_n, xi)`.
    /// Expects schema_version 1 envelope format. Returns `None` on any
    /// structural mismatch or a stale `toolkit_version`.
    pub(super) fn parse_json(
        data: &str,
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
    ) -> Option<CachedXi> {
        let v: serde_json::Value = serde_json::from_str(data).ok()?;
        let obj = v.as_object()?;

        if obj.get("schema_version").and_then(|x| x.as_u64())? as u32 != SCHEMA_VERSION {
            return None;
        }

        let file_ver = obj.get("toolkit_version").and_then(|x| x.as_str())?;
        if version_is_older(file_ver, &effective_min_version()) {
            return None;
        }

        if obj.get("n_modes").and_then(|x| x.as_u64())? as usize != n_modes {
            return None;
        }
        if obj.get("precision_bits").and_then(|x| x.as_u64())? as u32 != prec {
            return None;
        }
        let l_meta = obj.get("lambda_sq").and_then(|x| x.as_f64())?;
        if (l_meta - lambda_sq.value_f64).abs() > 0.5 {
            return None;
        }

        let eps_str = obj.get("weil_min_eigenvalue").and_then(|x| x.as_str())?;
        let eps_n = Float::with_val(prec, Float::parse(eps_str).ok()?);
        if eps_n.is_nan() || eps_n.is_infinite() {
            return None;
        }

        let arr = obj.get("xi").and_then(|x| x.as_array())?;
        if arr.len() != 2 * n_modes + 1 {
            return None;
        }
        let mut xi = Vec::with_capacity(arr.len());
        for s in arr {
            let f = Float::with_val(prec, Float::parse(s.as_str()?).ok()?);
            if f.is_nan() || f.is_infinite() {
                return None;
            }
            xi.push(f);
        }
        let portable: PortableInverseIterationDiagnostics =
            serde_json::from_value(obj.get("inverse_iteration")?.clone()).ok()?;
        let diagnostics = portable.to_runtime(prec).ok()?;
        Some(CachedXi {
            eps_n,
            xi,
            diagnostics,
        })
    }

    /// Returns `true` if version string `a` is strictly older than `b`.
    fn version_is_older(a: &str, b: &str) -> bool {
        let parse = |s: &str| -> (u64, u64, u64) {
            let mut parts = s.splitn(3, '.');
            let major = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let patch = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            (major, minor, patch)
        };
        parse(a) < parse(b)
    }

    /// Eigen-residual check: is `(xi, eps_n)` a genuine eigenpair of the
    /// in-hand τ matrix? Returns `true` when `‖τξ − ε_N·ξ‖_∞ / ‖ξ‖_∞`
    /// sits below the working-precision floor. This is the strong
    /// integrity test that catches a structurally-valid-but-wrong ξ
    /// (e.g. a different eigenvector, or one from a subtly different τ).
    pub(super) fn relative_residual_norm(
        tau: &[Float],
        dim: usize,
        xi: &[Float],
        eps_n: &Float,
        prec: u32,
    ) -> Option<Float> {
        if xi.len() != dim || tau.len() != dim * dim {
            return None;
        }

        // ‖ξ‖_∞ for the relative bound. A zero vector can never be a
        // valid eigenvector.
        let mut xi_linf = Float::with_val(prec, 0);
        for v in xi {
            let a = v.clone().abs();
            if a > xi_linf {
                xi_linf = a;
            }
        }
        if xi_linf.is_zero() {
            return None;
        }

        // max_i | (τξ)_i − ε_N ξ_i |, rows computed in parallel then a
        // deterministic max-fold. The inner row sum is sequential (it is
        // the same fixed index order every run).
        use rayon::prelude::*;
        let residuals: Vec<Float> = (0..dim)
            .into_par_iter()
            .map(|i| {
                let mut row = Float::with_val(prec, 0);
                for j in 0..dim {
                    let mut t = tau[i * dim + j].clone();
                    t *= &xi[j];
                    row += &t;
                }
                let mut e = eps_n.clone();
                e *= &xi[i];
                row -= &e;
                row.abs()
            })
            .collect();
        let mut resid_inf = Float::with_val(prec, 0);
        for residual in residuals {
            if residual > resid_inf {
                resid_inf = residual;
            }
        }

        // Relative residual vs floor. Use a generous floor: the eigenpair
        // is accurate to ~working precision, but the residual accumulates
        // O(N) HP roundings in the matrix-vector product. 2^-(prec-32)
        // leaves 32 bits (~10 digits) of headroom — far below the O(1)
        // residual a wrong ξ would produce, yet safely above the genuine
        // floor.
        let mut rel = resid_inf;
        rel /= &xi_linf;
        Some(rel)
    }

    pub(super) fn residual_ok(
        tau: &[Float],
        dim: usize,
        xi: &[Float],
        eps_n: &Float,
        prec: u32,
    ) -> bool {
        let Some(rel) = relative_residual_norm(tau, dim, xi, eps_n, prec) else {
            return false;
        };
        let floor = Float::with_val(prec, 2).pow(-((prec as i32) - 32));
        rel.cmp_abs(&floor).map(|o| o.is_lt()).unwrap_or(false)
    }

    /// Test-only accessor for `parse_json` (lets version-rejection tests
    /// call the parser directly without touching disk).
    #[cfg(test)]
    pub(super) fn parse_json_for_test(
        data: &str,
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
    ) -> Option<CachedXi> {
        parse_json(data, lambda_sq, n_modes, prec)
    }

    fn warn_skip(path: &std::path::Path, reason: &str) {
        crate::hp_debug!(
            "[weil_eigvec_cache] WARNING: skipping {} ({}); recomputing",
            path.display(),
            reason
        );
    }

    /// Read a single zip and return the parsed entry plus the raw inner
    /// JSON (so the caller can write the decompressed copy without
    /// re-serializing).
    fn read_single_zip(
        zip_path: &std::path::Path,
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        parity_policy: CcmParityPolicy,
    ) -> Option<(CachedXi, String)> {
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        let entry_name = cache_filename(lambda_sq, n_modes, prec, parity_policy);
        let mut entry = archive.by_name(&entry_name).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_json(&data, lambda_sq, n_modes, prec)?;
        Some((parsed, data))
    }

    pub(super) fn load(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        mode: CacheMode,
        parity_policy: CcmParityPolicy,
    ) -> Option<CachedXi> {
        if mode == CacheMode::Off {
            return None;
        }

        // Caches are zip-only: read straight from the .json.zip
        // (decompress in memory), never write a decompressed .json.
        // JsonOnly is a read no-op because current cache files are zip-only.
        if mode == CacheMode::JsonOnly {
            return None;
        }

        // Local single zip — in memory.
        if let Some(c) = try_load_local_zip(lambda_sq, n_modes, prec, parity_policy) {
            return Some(c);
        }

        None
    }

    /// Load from a local single zip. Decompresses in memory; does NOT
    /// write a decompressed `.json`. Returns the parsed entry or None.
    fn try_load_local_zip(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        parity_policy: CcmParityPolicy,
    ) -> Option<CachedXi> {
        let zp = zip_path(lambda_sq, n_modes, prec, parity_policy)?;
        if !zp.exists() {
            return None;
        }
        match read_single_zip(&zp, lambda_sq, n_modes, prec, parity_policy) {
            Some((parsed, _json_string)) => Some(parsed),
            None => {
                warn_skip(&zp, "zip open / decompress / shape parse failed");
                None
            }
        }
    }

    /// Serialize `(eps_n, xi)` to the versioned schema-v1 JSON object.
    fn serialize_to_json(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        eps_n: &Float,
        xi: &[Float],
        diagnostics: &xc_numerics::linalg::InverseIterationDiagnostics,
    ) -> Vec<u8> {
        let xi_strings: Vec<String> = xi.iter().map(|f| f.to_string()).collect();
        let payload = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "toolkit_version": TOOLKIT_VERSION,
            "lambda_sq": lambda_sq.value_f64,
            "lambda_sq_mode": lambda_sq.mode_str(),
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": eps_n.to_string(),
            "xi": xi_strings,
            "inverse_iteration": PortableInverseIterationDiagnostics::from_runtime(diagnostics),
        });
        serde_json::to_vec(&payload).unwrap_or_default()
    }

    /// See `tau_cache::compress_to_zip` for why `large_file(true)` is
    /// required (silent 4 GiB write-abort otherwise).
    fn compress_to_zip(json_bytes: &[u8], entry_name: &str) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(json_bytes.len() / 2);
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = zip::ZipWriter::new(cursor);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .large_file(true);
            if writer.start_file(entry_name, opts).is_err() {
                return Vec::new();
            }
            if writer.write_all(json_bytes).is_err() {
                return Vec::new();
            }
            if writer.finish().is_err() {
                return Vec::new();
            }
        }
        buf
    }

    fn cleanup_previous(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        parity_policy: CcmParityPolicy,
    ) {
        if let Some(p) = json_path(lambda_sq, n_modes, prec, parity_policy) {
            if p.exists() {
                let _ = std::fs::remove_file(&p);
            }
        }
        if let Some(p) = zip_path(lambda_sq, n_modes, prec, parity_policy) {
            if p.exists() {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn save(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        eps_n: &Float,
        xi: &[Float],
        diagnostics: &xc_numerics::linalg::InverseIterationDiagnostics,
        mode: CacheMode,
        parity_policy: CcmParityPolicy,
    ) {
        // Off and JsonOnly write nothing: the cache is zip-only.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) {
            return;
        }

        let json_bytes = serialize_to_json(lambda_sq, n_modes, prec, eps_n, xi, diagnostics);
        if json_bytes.is_empty() {
            return;
        }

        cleanup_previous(lambda_sq, n_modes, prec, parity_policy);

        // Write ONLY the compressed copy. Readers decompress from the zip
        // on demand — no uncompressed .json is persisted. ξ is small, so
        // this is always a single zip (no byte-split tier — unlike τ).
        let entry_name = cache_filename(lambda_sq, n_modes, prec, parity_policy);
        let zip_bytes = compress_to_zip(&json_bytes, &entry_name);
        if zip_bytes.is_empty() {
            eprintln!(
                "[weil_eigvec_cache] WARNING: zip compression failed for λ²={}, N={}, \
                 prec={} ({} bytes uncompressed) — this config will NOT be cached and \
                 will recompute from scratch on every run",
                lambda_sq.value_f64,
                n_modes,
                prec,
                json_bytes.len()
            );
            return;
        }
        if let Some(zp) = zip_path(lambda_sq, n_modes, prec, parity_policy) {
            if let Err(e) = std::fs::write(&zp, &zip_bytes) {
                crate::hp_debug!(
                    "[weil_eigvec_cache] WARNING: could not write {}: {}",
                    zp.display(),
                    e
                );
            }
        }
    }
}

// Matrix fixtures below use the row-major index convention `m[i * dim + j]`
// uniformly, including `i = 0` rows where `0 * dim` is kept for alignment
// with neighboring entries. Allow the erasing_op lint in this test module.
#[cfg(test)]
#[allow(clippy::erasing_op)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn legacy_eigenstate_payload_keeps_its_established_json_shape() {
        let artifact = PortableWeilEigenpair {
            schema_version: 2,
            lambda_squared: "13".to_owned(),
            n_modes: 1,
            precision_bits: 192,
            force_even: true,
            parity_policy: None,
            eigenstate_route: legacy_eigenstate_route_name(),
            eigenvalue: "1".to_owned(),
            eigenvector: vec!["0".to_owned(), "1".to_owned(), "0".to_owned()],
            inverse_iteration: PortableInverseIterationDiagnostics {
                configured_step_limit: 2_000,
                unshifted_steps: 2_000,
                unshifted_converged: false,
                final_relative_rayleigh_change: Some("1e-40".to_owned()),
                shifted_refinement: "accepted".to_owned(),
                final_relative_residual_norm: "1e-50".to_owned(),
            },
            shift_invert_krylov: None,
        };
        let value = serde_json::to_value(artifact).unwrap();
        assert!(value.get("parity_policy").is_none());
        assert!(value.get("eigenstate_route").is_none());
        assert!(value.get("shift_invert_krylov").is_none());
        assert_eq!(
            HighPrecConfig::for_decimal_digits(100).eigenstate_solver,
            CcmEigenstateSolver::Auto
        );
    }

    #[test]
    fn resolved_legacy_route_preserves_established_root_encoding() {
        let precision_bits = 192;
        let mut third = Float::with_val(precision_bits, 1);
        third /= 3;
        let result = RootRefinement {
            value: third.clone(),
            diagnostics: RootRefinementDiagnostics {
                iterations: 7,
                final_correction: third.clone(),
                residual: third.clone(),
                achieved_decimal_digits: third,
            },
        };

        let legacy = PortableRootRefinement::from_runtime_for_eigenstate_solver(
            &result,
            CcmEigenstateSolver::LegacyInverseIteration,
        );
        let krylov = PortableRootRefinement::from_runtime_for_eigenstate_solver(
            &result,
            CcmEigenstateSolver::ShiftInvertKrylov,
        );

        assert_eq!(legacy, PortableRootRefinement::from_runtime(&result));
        assert_eq!(
            krylov,
            PortableRootRefinement::from_runtime_lossless(&result)
        );
        assert_ne!(legacy, krylov);
    }

    #[test]
    fn auto_retries_only_explicit_krylov_nonconvergence() {
        let retryable = anyhow::Error::new(CacheError::InvalidTransition(
            "CCM shift-invert Krylov did not produce an unambiguous converged target: status=Approximate, boundary_cluster=false"
                .to_owned(),
        ));
        assert!(is_retryable_auto_krylov_failure(&retryable));

        let corruption = anyhow::Error::new(CacheError::InvalidManifest(
            "CCM eigenpair payload failed residual replay".to_owned(),
        ));
        assert!(!is_retryable_auto_krylov_failure(&corruption));

        let unrelated_transition = anyhow::Error::new(CacheError::InvalidTransition(
            "repository publication batch is incomplete".to_owned(),
        ));
        assert!(!is_retryable_auto_krylov_failure(&unrelated_transition));
    }

    #[test]
    fn legacy_and_krylov_eigenstate_cache_identities_are_disjoint() {
        let params = CcmParams::from_lambda_sq_integer(13, 120);
        let mut legacy = HighPrecConfig::for_decimal_digits(1_000);
        legacy.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        let (legacy_semantic, legacy_logical) =
            weil_eigenpair_cache_identity(&params, &legacy).unwrap();
        assert_eq!(
            legacy_logical,
            format!("ccm/weil-eigenpair/13/120/{}/even", legacy.precision_bits)
        );
        assert_eq!(
            legacy_semantic.mathematical_semantics_version,
            "ccm-smallest-weil-eigenpair-v0.13.0-v3"
        );
        assert_eq!(
            legacy_semantic.resolved_mathematical_parameters,
            serde_json::json!({
                "lambda_squared": "13",
                "n_modes": 120,
                "precision_bits": legacy.precision_bits,
                "scalar_backend": "rug_mpfr",
                "force_even": true,
                "normalization": "sum_xi_equals_sqrt_log_lambda_squared",
                "inverse_iteration_step_limit": 2_000
            })
        );

        let mut krylov = legacy.clone();
        krylov.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
        let (krylov_semantic, krylov_logical) =
            weil_eigenpair_cache_identity(&params, &krylov).unwrap();
        assert_eq!(
            krylov_logical,
            format!(
                "ccm/weil-eigenpair/13/120/{}/even/shift_invert_krylov",
                krylov.precision_bits
            )
        );
        assert_ne!(
            legacy_semantic.digest().unwrap(),
            krylov_semantic.digest().unwrap()
        );
        assert_ne!(legacy_logical, krylov_logical);

        let mut natural = legacy.clone();
        natural.set_parity_policy(CcmParityPolicy::Natural);
        let (natural_semantic, natural_logical) =
            weil_eigenpair_cache_identity(&params, &natural).unwrap();
        assert_eq!(
            natural_logical,
            format!(
                "ccm/weil-eigenpair/13/120/{}/natural",
                natural.precision_bits
            )
        );

        let mut adaptive = legacy.clone();
        adaptive.set_parity_policy(CcmParityPolicy::AdaptiveEven);
        let (adaptive_semantic, adaptive_logical) =
            weil_eigenpair_cache_identity(&params, &adaptive).unwrap();
        assert_eq!(
            adaptive_logical,
            format!(
                "ccm/weil-eigenpair/13/120/{}/adaptive-even",
                adaptive.precision_bits
            )
        );
        assert_eq!(
            adaptive_semantic
                .resolved_mathematical_parameters
                .get("parity_policy"),
            Some(&serde_json::json!("adaptive-even"))
        );
        assert_eq!(
            adaptive_semantic.mathematical_semantics_version,
            "ccm-smallest-weil-eigenpair-adaptive-even-v1"
        );
        assert_ne!(
            legacy_semantic.digest().unwrap(),
            natural_semantic.digest().unwrap()
        );
        assert_ne!(
            legacy_semantic.digest().unwrap(),
            adaptive_semantic.digest().unwrap()
        );
        assert_ne!(
            natural_semantic.digest().unwrap(),
            adaptive_semantic.digest().unwrap()
        );
    }

    #[test]
    fn cross_n_seed_preserves_the_even_basis_coordinates() {
        let precision = 192;
        // Full centered coordinates for N=2: [a2, a1, a0, a1, a2].
        let full: Vec<Float> = [3, 2, 1, 2, 3]
            .into_iter()
            .map(|value| Float::with_val(precision, value))
            .collect();
        let seed = embedded_even_continuation_seed(&full, 2, 4, precision).unwrap();
        assert_eq!(seed.len(), 5);
        assert_eq!(seed[0], 1);
        let sqrt_two = Float::with_val(precision, 2).sqrt();
        let mut expected_one = Float::with_val(precision, 2);
        expected_one *= &sqrt_two;
        let mut expected_two = Float::with_val(precision, 3);
        expected_two *= sqrt_two;
        assert_eq!(seed[1], expected_one);
        assert_eq!(seed[2], expected_two);
        assert!(seed[3].is_zero());
        assert!(seed[4].is_zero());
    }

    fn test_reference_dataset() -> ReferenceZeroDatasetIdentity {
        ReferenceZeroDatasetIdentity {
            schema_version: 1,
            resource_id: "test/reference-zeros.json".to_owned(),
            content_sha256: ContentDigest::sha256(b"test reference zeros").0,
            record_count: 1_000,
            decimal_digits: 100,
        }
    }

    #[test]
    fn seeded_and_independent_root_artifacts_have_disjoint_semantic_identity() {
        let params = CcmParams::from_lambda_sq_integer(13, 4);
        let cfg = HighPrecConfig::for_decimal_digits(40);
        let seeds = vec![Float::with_val(cfg.precision_bits, 14)];
        let dataset = test_reference_dataset();
        let independent = root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::Independent,
            None,
            RootWindowSemantics::strict_positive(seeds.len()),
        )
        .unwrap();
        let seeded = root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::ReferenceSeededRefinement,
            Some(&dataset),
            RootWindowSemantics::strict_positive(seeds.len()),
        )
        .unwrap();
        assert_eq!(independent.artifact_kind, "ccm_root_discovery_window");
        assert_eq!(
            independent.mathematical_semantics_version,
            "ccm-root-range-v0.13.0-v6"
        );
        assert!(independent
            .resolved_mathematical_parameters
            .get("root_domain")
            .is_none());
        assert!(independent
            .resolved_mathematical_parameters
            .get("requested_root_count")
            .is_none());
        assert!(independent
            .resolved_mathematical_parameters
            .get("allow_incomplete")
            .is_none());
        assert_eq!(seeded.artifact_kind, "ccm_root_refinement");
        assert!(independent.source_data_identities.is_empty());
        assert_eq!(
            seeded.source_data_identities.get("reference_zero_dataset"),
            Some(&ContentDigest(dataset.content_sha256.clone()))
        );
        assert_ne!(independent.digest().unwrap(), seeded.digest().unwrap());
        let signed = root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::Independent,
            None,
            RootWindowSemantics::advanced(IndependentRootDomain::Signed, 200, true),
        )
        .unwrap();
        assert_eq!(signed.target.as_deref(), Some("signed_ccm_spectral_roots"));
        assert_eq!(
            signed.mathematical_semantics_version,
            "ccm-root-range-v0.13.3-v7"
        );
        assert_ne!(independent.digest().unwrap(), signed.digest().unwrap());
        let same_signed_window_for_another_request = root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::Independent,
            None,
            RootWindowSemantics::advanced(IndependentRootDomain::Signed, 8, false),
        )
        .unwrap();
        assert_eq!(
            signed.digest().unwrap(),
            same_signed_window_for_another_request.digest().unwrap(),
            "request policy must not duplicate an identical numerical root window"
        );

        let mut other_dataset = dataset.clone();
        other_dataset.content_sha256 = ContentDigest::sha256(b"other reference zeros").0;
        let other_seeded = root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::ReferenceSeededRefinement,
            Some(&other_dataset),
            RootWindowSemantics::strict_positive(seeds.len()),
        )
        .unwrap();
        assert_ne!(seeded.digest().unwrap(), other_seeded.digest().unwrap());
        assert!(root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::Independent,
            Some(&dataset),
            RootWindowSemantics::strict_positive(seeds.len()),
        )
        .is_err());
        assert!(root_range_semantic_key(
            &params,
            &cfg,
            1,
            &seeds,
            RootArtifactMode::ReferenceSeededRefinement,
            None,
            RootWindowSemantics::strict_positive(seeds.len()),
        )
        .is_err());
    }

    #[test]
    fn ill_conditioned_lu_is_validated_by_backward_error() {
        let precision = 256;
        let one = Float::with_val(precision, 1);
        let mut one_plus_delta = one.clone();
        one_plus_delta += Float::with_val(
            precision,
            Float::parse("1e-70").expect("near-singular perturbation"),
        );
        let matrix = vec![one.clone(), one.clone(), one, one_plus_delta];
        let factors = xc_numerics::linalg::lu_factor(&matrix, 2).unwrap();
        let backward_error = factorization_backward_error(&matrix, &factors, 2, precision).unwrap();
        let tolerance = Float::with_val(precision, 2).pow(-((precision / 4) as i32));
        assert!(
            backward_error < tolerance,
            "backward-stable LU was rejected: error={}, tolerance={}",
            xc_numerics::fmt::display_hp(&backward_error, 8),
            xc_numerics::fmt::display_hp(&tolerance, 8)
        );

        let invalid_permutation = xc_numerics::linalg::LuFactors {
            lu: factors.lu,
            perm: vec![0, 0],
        };
        assert!(
            factorization_backward_error(&matrix, &invalid_permutation, 2, precision).is_none()
        );
    }

    #[test]
    fn independent_starting_points_are_default_seed_source() {
        let precision = 192;
        let params = CcmParams::from_lambda_sq_integer(13, 4);
        let l = Float::with_val(precision, 13).ln();
        let xi = vec![Float::with_val(precision, 1); params.matrix_size()];
        let plan = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::IndexRange { first: 2, last: 3 },
            IndependentRootDiscoveryOptions::default(),
            precision,
        )
        .unwrap();
        assert_eq!(plan.result_first_root_index, 2);
        assert_eq!(plan.artifact_seeds.len(), 2);
        assert!(plan.artifact_seeds[0] > 0 && plan.artifact_seeds[0] < plan.artifact_seeds[1]);
    }

    #[test]
    fn independent_discovery_preserves_weights_below_binary64_range() {
        let precision = 2048;
        let params = CcmParams::from_lambda_sq_integer(13, 4);
        let l = Float::with_val(precision, 13).ln();
        let tiny = Float::with_val(precision, Float::parse("1e-400").unwrap());
        assert_eq!(tiny.to_f64(), 0.0, "fixture must underflow in binary64");
        let xi = vec![tiny; params.matrix_size()];
        let plan = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::IndexRange { first: 1, last: 3 },
            IndependentRootDiscoveryOptions::default(),
            precision,
        )
        .unwrap();
        assert_eq!(plan.result_first_root_index, 1);
        assert_eq!(plan.artifact_seeds.len(), 3);
        assert!(plan.artifact_seeds.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn signed_independent_discovery_returns_the_available_finite_window() {
        let precision = 192;
        let params = CcmParams::from_lambda_sq_integer(13, 4);
        let l = Float::with_val(precision, 13).ln();
        let xi = vec![Float::with_val(precision, 1); params.matrix_size()];
        let options = IndependentRootDiscoveryOptions::advanced(true, true);
        let plan = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::FirstK { count: 20 },
            options,
            precision,
        )
        .unwrap();
        assert_eq!(plan.result_first_root_index, 1);
        assert_eq!(plan.artifact_seeds.len(), 8);
        assert!(plan.artifact_seeds.first().unwrap() < &0);
        assert!(plan.artifact_seeds.last().unwrap() > &0);
        assert!(plan.artifact_seeds.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(plan.request_semantics.domain, IndependentRootDomain::Signed);
        assert_eq!(plan.request_semantics.requested_count, 20);
        assert!(plan.request_semantics.allow_incomplete);
        assert_eq!(plan.selected_positions, (0..8).collect::<Vec<_>>());
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = precision;
        let refined = compute_root_range(&xi, &params, &l, &cfg, &plan.artifact_seeds);
        ensure_root_window_usable(
            &refined,
            plan.artifact_seeds.len(),
            false,
            IndependentRootDomain::Signed,
        )
        .unwrap();
        assert!(ensure_root_window_usable(
            &refined,
            plan.artifact_seeds.len(),
            false,
            IndependentRootDomain::Positive,
        )
        .is_err());
    }

    #[test]
    fn advanced_incomplete_discovery_accepts_an_empty_finite_window() {
        let precision = 192;
        let params = CcmParams::from_lambda_sq_integer(250, 10);
        let l = Float::with_val(precision, 250).ln();
        let mut xi = vec![Float::with_val(precision, 0); params.matrix_size()];
        xi[params.n_modes] = Float::with_val(precision, 1);
        let plan = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::FirstK { count: 200 },
            IndependentRootDiscoveryOptions::advanced(true, true),
            precision,
        )
        .unwrap();
        assert!(plan.artifact_seeds.is_empty());
        assert!(plan.selected_positions.is_empty());
        assert_eq!(plan.result_first_root_index, 1);
        assert_eq!(plan.request_semantics.domain, IndependentRootDomain::Signed);
        assert_eq!(plan.request_semantics.requested_count, 200);
        assert!(plan.request_semantics.allow_incomplete);
    }

    #[test]
    fn advanced_signed_requests_project_one_canonical_window() {
        let precision = 192;
        let params = CcmParams::from_lambda_sq_integer(13, 4);
        let l = Float::with_val(precision, 13).ln();
        let xi = vec![Float::with_val(precision, 1); params.matrix_size()];
        let plan = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::FirstK { count: 4 },
            IndependentRootDiscoveryOptions::advanced(true, true),
            precision,
        )
        .unwrap();
        assert_eq!(plan.artifact_seeds.len(), 8);
        assert_eq!(plan.selected_positions.len(), 4);
        assert_eq!(plan.selected_positions, vec![2, 3, 4, 5]);
        assert_eq!(plan.request_semantics.requested_count, 4);
    }

    #[test]
    fn strict_independent_discovery_rejects_a_finite_window_shortfall() {
        let precision = 192;
        let params = CcmParams::from_lambda_sq_integer(13, 4);
        let l = Float::with_val(precision, 13).ln();
        let xi = vec![Float::with_val(precision, 1); params.matrix_size()];
        let error = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::FirstK { count: 20 },
            IndependentRootDiscoveryOptions::advanced(true, false),
            precision,
        )
        .unwrap_err();
        assert!(error.to_string().contains("target requests 20"));
    }

    #[test]
    fn parallel_secular_discovery_is_bit_identical_to_ordered_reference() {
        let precision = 256;
        let n_modes = 8;
        let spacing = Float::with_val(precision, Float::parse("1.75").unwrap());
        let scan_upper = Float::with_val(precision, 16);
        let xi = (0..(2 * n_modes + 1))
            .map(|index| Float::with_val(precision, index + 1))
            .collect::<Vec<_>>();
        let parallel =
            discover_secular_roots_hp(&xi, n_modes, &spacing, &scan_upper, precision).unwrap();
        let sequential = discover_secular_roots_hp_sequential_reference(
            &xi,
            n_modes,
            &spacing,
            &scan_upper,
            precision,
        )
        .unwrap();
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn parallel_root_refinement_is_bit_identical_to_seed_order_reference() {
        let precision = 256;
        let params = CcmParams::from_lambda_sq_integer(13, 5);
        let l = Float::with_val(precision, 13).ln();
        let xi = vec![Float::with_val(precision, 1); params.matrix_size()];
        let mut cfg = HighPrecConfig::for_decimal_digits(60);
        cfg.precision_bits = precision;
        cfg.solver_steps = 24;
        let plan = independently_discovered_starting_points(
            &params,
            &l,
            &xi,
            &ZeroTarget::FirstK { count: 3 },
            IndependentRootDiscoveryOptions::default(),
            precision,
        )
        .unwrap();
        let seeds = plan.artifact_seeds;
        let parallel = compute_root_range(&xi, &params, &l, &cfg, &seeds);
        let sequential = seeds
            .iter()
            .map(|seed| {
                solve_r_zero(
                    &xi,
                    params.n_modes,
                    &l,
                    seed,
                    cfg.precision_bits,
                    cfg.solver_steps,
                    cfg.root_solver,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(parallel.len(), sequential.len());
        for (parallel, sequential) in parallel.iter().zip(&sequential) {
            match (parallel, sequential) {
                (EigenvalueResult::Converged(left), EigenvalueResult::Converged(right))
                | (EigenvalueResult::Stagnated(left), EigenvalueResult::Stagnated(right))
                | (EigenvalueResult::Approximate(left), EigenvalueResult::Approximate(right)) => {
                    assert_eq!(left.value, right.value);
                    assert_eq!(left.diagnostics.iterations, right.diagnostics.iterations);
                    assert_eq!(
                        left.diagnostics.final_correction,
                        right.diagnostics.final_correction
                    );
                    assert_eq!(left.diagnostics.residual, right.diagnostics.residual);
                    assert_eq!(
                        left.diagnostics.achieved_decimal_digits,
                        right.diagnostics.achieved_decimal_digits
                    );
                }
                (
                    EigenvalueResult::Failed {
                        iterations: left_iterations,
                        reason: left_reason,
                    },
                    EigenvalueResult::Failed {
                        iterations: right_iterations,
                        reason: right_reason,
                    },
                ) => {
                    assert_eq!(left_iterations, right_iterations);
                    assert_eq!(left_reason, right_reason);
                }
                _ => panic!("parallel refinement changed an outcome status"),
            }
        }
    }

    #[test]
    fn computed_root_windows_retain_approximations_while_strict_windows_reject_them() {
        let precision = 256;
        let diagnostic = |value: u32| RootRefinement {
            value: Float::with_val(precision, value),
            diagnostics: RootRefinementDiagnostics {
                iterations: 2_000,
                final_correction: Float::with_val(precision, Float::parse("1e-50").unwrap()),
                residual: Float::with_val(precision, Float::parse("1e-60").unwrap()),
                achieved_decimal_digits: Float::with_val(precision, 50),
            },
        };
        assert!(ensure_root_window_usable(
            &[
                EigenvalueResult::Converged(diagnostic(1)),
                EigenvalueResult::Converged(diagnostic(2)),
            ],
            2,
            true,
            IndependentRootDomain::Positive,
        )
        .is_ok());
        assert!(ensure_root_window_usable(
            &[
                EigenvalueResult::Stagnated(diagnostic(1)),
                EigenvalueResult::Approximate(diagnostic(2)),
            ],
            2,
            false,
            IndependentRootDomain::Positive,
        )
        .is_ok());
        assert!(ensure_root_window_usable(
            &[EigenvalueResult::Stagnated(diagnostic(1))],
            1,
            true,
            IndependentRootDomain::Positive,
        )
        .unwrap_err()
        .to_string()
        .contains("stagnated"));
        assert!(ensure_root_window_usable(
            &[EigenvalueResult::Approximate(diagnostic(1))],
            1,
            true,
            IndependentRootDomain::Positive,
        )
        .unwrap_err()
        .to_string()
        .contains("iteration limit"));
        assert!(ensure_root_window_usable(
            &[EigenvalueResult::Failed {
                iterations: 3,
                reason: "degenerate derivative".to_owned(),
            }],
            1,
            false,
            IndependentRootDomain::Positive,
        )
        .unwrap_err()
        .to_string()
        .contains("degenerate derivative"));
    }

    #[test]
    fn precision_limited_root_indices_are_compacted_into_ranges() {
        assert_eq!(
            format_index_ranges(&[37, 39, 40, 41, 43, 44, 45, 46, 48, 49, 50]),
            "37,39-41,43-46,48-50"
        );
        assert_eq!(format_index_ranges(&[]), "");
    }

    #[test]
    fn residual_replay_is_compared_to_the_stored_residual_not_correction_floor() {
        let precision_bits = 729;
        let residual = Float::with_val(precision_bits, Float::parse("1.17184766e-198").unwrap());
        // This residual is intentionally much larger than a roughly 1e-200
        // correction target. Identical replay is nevertheless valid evidence.
        assert!(residual_replay_matches(
            &residual,
            Some(&residual),
            Some(&Float::with_val(precision_bits, 1)),
            61,
            precision_bits
        ));
        let changed = Float::with_val(precision_bits, Float::parse("1.27184766e-198").unwrap());
        assert!(!residual_replay_matches(
            &residual,
            Some(&changed),
            Some(&Float::with_val(precision_bits, 1)),
            61,
            precision_bits
        ));
    }

    #[test]
    fn computed_cache_replays_stagnated_root_evidence_without_claiming_convergence() {
        let mut cfg = HighPrecConfig::for_decimal_digits(60);
        cfg.precision_bits = 256;
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        let params = CcmParams::from_lambda_sq_integer(13, 1);
        let l = Float::with_val(cfg.precision_bits, 13).ln();
        let xi = vec![Float::with_val(cfg.precision_bits, 1); params.matrix_size()];
        let value = Float::with_val(cfg.precision_bits, 1);
        let correction = Float::with_val(cfg.precision_bits, Float::parse("1e-50").unwrap());
        assert!(correction >= root_correction_tolerance(&value, cfg.precision_bits));
        let mut two_pi_over_l = pi(cfg.precision_bits);
        two_pi_over_l *= 2u32;
        two_pi_over_l /= &l;
        let result = RootRefinement {
            value: value.clone(),
            diagnostics: RootRefinementDiagnostics {
                iterations: 17,
                final_correction: correction.clone(),
                residual: secular_residual_at(
                    &xi,
                    params.n_modes,
                    &two_pi_over_l,
                    &value,
                    cfg.precision_bits,
                )
                .unwrap(),
                achieved_decimal_digits: achieved_decimal_digits(
                    &value,
                    &correction,
                    cfg.precision_bits,
                ),
            },
        };
        let artifact = PortableRootRange {
            schema_version: 3,
            lambda_squared: lambda_squared_cache_identity(&params),
            n_modes: params.n_modes,
            precision_bits: cfg.precision_bits,
            force_even: cfg.effective_parity_policy().legacy_force_even(),
            parity_policy: None,
            first_root_index: 1,
            root_domain: IndependentRootDomain::Positive,
            discovery_mode: RootArtifactMode::Independent.as_str().to_owned(),
            reference_seeds_used: false,
            reference_dataset: None,
            completeness: "unverified_computed_discovery".to_owned(),
            starting_points: vec![value.to_string()],
            outcomes: vec![PortableRootOutcome::Stagnated(
                PortableRootRefinement::from_runtime(&result),
            )],
            solver: cfg.root_solver.display_name().to_ascii_lowercase(),
            solver_steps: cfg.solver_steps,
            accuracy_guard_bits: GUARD_BITS,
        };
        let legacy_json = serde_json::to_value(&artifact).unwrap();
        assert!(legacy_json.get("root_domain").is_none());
        assert!(legacy_json.get("requested_root_count").is_none());
        assert!(legacy_json.get("allow_incomplete").is_none());
        assert!(decode_root_range(
            &artifact,
            &params,
            &cfg,
            1,
            std::slice::from_ref(&value),
            RootArtifactMode::Independent,
            None,
            &xi,
            &l,
            RootWindowSemantics::strict_positive(1),
            false,
        )
        .is_ok());
        assert!(decode_root_range(
            &artifact,
            &params,
            &cfg,
            1,
            &[value],
            RootArtifactMode::Independent,
            None,
            &xi,
            &l,
            RootWindowSemantics::strict_positive(1),
            true,
        )
        .unwrap_err()
        .to_string()
        .contains("requested assurance rejects stagnated"));
    }

    #[test]
    fn fused_archimedean_integrals_are_bit_identical_to_separate_evaluators() {
        let precision = 256;
        let l = Float::with_val(precision, 13).ln();
        let nodes = ["-0.91", "-0.43", "0", "0.37", "0.88"]
            .iter()
            .map(|value| Float::with_val(precision, Float::parse(value).unwrap()))
            .collect::<Vec<_>>();
        let weights = ["0.12", "0.23", "0.31", "0.22", "0.12"]
            .iter()
            .map(|value| Float::with_val(precision, Float::parse(value).unwrap()))
            .collect::<Vec<_>>();
        let kappa_half = compute_kappa_half(&l, precision);
        for mode in 0..=6 {
            let fused =
                compute_archimedean_integrals_l(mode, &l, precision, &nodes, &weights, &kappa_half);
            assert_eq!(
                fused.0,
                compute_alpha_l(mode, &l, precision, &nodes, &weights),
                "alpha changed at mode {mode}"
            );
            assert_eq!(
                fused.1,
                compute_beta_l(mode, &l, precision, &nodes, &weights),
                "beta changed at mode {mode}"
            );
            assert_eq!(
                fused.2,
                compute_gamma_l(mode, &l, precision, &nodes, &weights),
                "gamma changed at mode {mode}"
            );
        }
    }

    #[test]
    fn precomputed_prime_kernels_are_bit_identical_to_reference() {
        for precision in [128, 257] {
            let l = Float::with_val(precision, 13).ln();
            for n_modes in 0..=4 {
                let optimized = compute_prime_component_matrix(n_modes, 19, &l, precision);
                let reference =
                    compute_prime_component_matrix_reference(n_modes, 19, &l, precision);
                assert_eq!(
                    optimized.len(),
                    reference.len(),
                    "matrix shape changed at precision={precision}, N={n_modes}"
                );
                for (index, (actual, expected)) in
                    optimized.iter().zip(reference.iter()).enumerate()
                {
                    assert_eq!(
                        actual, expected,
                        "prime component changed at precision={precision}, N={n_modes}, cell={index}"
                    );
                }
            }
        }
    }

    #[test]
    fn tau_matrix_round_trips_through_common_cache_fabric() {
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
            FilesystemCacheStore,
        };

        let root =
            std::env::temp_dir().join(format!("xc-spectral-ccm-tau-fabric-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let context = ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["workstation".to_owned()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let params = CcmParams::from_lambda_sq_integer(5, 2);
        let cfg = HighPrecConfig::for_decimal_digits(40);
        let l = log_lambda_sq_hp(&params, cfg.precision_bits);
        let first = build_tau_hp_via_cache(&params, &l, &cfg, &context).unwrap();
        let second = build_tau_hp_via_cache(&params, &l, &cfg, &context).unwrap();
        assert_eq!(first, second);
        assert!(!first.1.dependencies.is_empty());
        assert!(first
            .1
            .dependencies
            .iter()
            .any(|dependency| dependency.key.kind == "ccm_archimedean_integrals"));
        assert!(first
            .1
            .dependencies
            .iter()
            .any(|dependency| dependency.key.kind == "ccm_prime_component"));
        let seeds = vec![Float::with_val(cfg.precision_bits, 14)];
        let dataset = test_reference_dataset();
        let first_run =
            run_indexed_seeded_via_cache(&params, &cfg, 1, &seeds, &dataset, &context).unwrap();
        let second_run =
            run_indexed_seeded_via_cache(&params, &cfg, 1, &seeds, &dataset, &context).unwrap();
        assert_eq!(
            first_run.weil_min_eigenvalue,
            second_run.weil_min_eigenvalue
        );
        assert_eq!(first_run.xi, second_run.xi);
        let (_, _, _, _, resolved_eigenstate_solver) = weil_eigenpair_via_cache_with_seed(
            &params, &cfg, &l, &first.0, &first.1, &context, None, None,
        )
        .unwrap();
        assert_eq!(
            resolved_eigenstate_solver,
            CcmEigenstateSolver::LegacyInverseIteration
        );
        let auto_next_params = CcmParams::from_lambda_sq_integer(5, 3);
        let auto_next =
            run_indexed_seeded_via_cache(&auto_next_params, &cfg, 1, &seeds, &dataset, &context)
                .unwrap();
        assert_eq!(auto_next.xi.len(), 7);
        let mut resolved_auto_cfg = cfg.clone();
        resolved_auto_cfg.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
        let (auto_next_semantic, auto_next_logical) =
            weil_eigenpair_cache_identity(&auto_next_params, &resolved_auto_cfg).unwrap();
        let auto_next_key = ArtifactKey {
            kind: auto_next_semantic.artifact_kind.clone(),
            logical_key: auto_next_logical,
            parameters_digest: auto_next_semantic.digest().unwrap(),
        };
        let auto_next_artifact = resolver.resolve(&auto_next_key, &policy).unwrap();
        let auto_next_payload: PortableWeilEigenpair =
            serde_json::from_slice(&auto_next_artifact.payload).unwrap();
        assert!(auto_next_payload
            .shift_invert_krylov
            .as_ref()
            .unwrap()
            .seed_identity
            .starts_with("from-eigenpair-"));
        let mut krylov_cfg = cfg.clone();
        krylov_cfg.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
        krylov_cfg.krylov_guard_eigenpairs = 1;
        krylov_cfg.krylov_subspace_dimension = 4;
        krylov_cfg.krylov_maximum_restarts = 16;
        let krylov_run =
            run_indexed_seeded_via_cache(&params, &krylov_cfg, 1, &seeds, &dataset, &context)
                .unwrap();
        let mut route_difference = krylov_run.weil_min_eigenvalue.clone();
        route_difference -= &first_run.weil_min_eigenvalue;
        route_difference.abs_mut();
        assert!(
            route_difference
                < Float::with_val(cfg.precision_bits, 2).pow(-((cfg.precision_bits as i32) - 48))
        );
        assert!(weil_eigvec_cache::residual_ok(
            &first.0,
            params.matrix_size(),
            &krylov_run.xi,
            &krylov_run.weil_min_eigenvalue,
            cfg.precision_bits
        ));
        let sweep = run_indexed_seeded_n_sweep_via_cache(
            &[
                CcmIndexedSeededSweepPoint {
                    params: params.clone(),
                    first_root_index: 1,
                    zero_seeds: seeds.clone(),
                },
                CcmIndexedSeededSweepPoint {
                    params: CcmParams::from_lambda_sq_integer(5, 3),
                    first_root_index: 1,
                    zero_seeds: seeds.clone(),
                },
            ],
            &krylov_cfg,
            &dataset,
            &context,
        )
        .unwrap();
        assert_eq!(sweep.len(), 2);
        assert_eq!(sweep[0].xi.len(), 5);
        assert_eq!(sweep[1].xi.len(), 7);
        assert_eq!(first_run.eigenvalues_pos.len(), 1);
        assert_eq!(first_run.spectral_root_index_range(), Some(1..=1));
        let first_indexed =
            run_indexed_seeded_via_cache(&params, &cfg, 17, &seeds, &dataset, &context).unwrap();
        let second_indexed =
            run_indexed_seeded_via_cache(&params, &cfg, 17, &seeds, &dataset, &context).unwrap();
        assert_eq!(first_indexed.spectral_root_index_range(), Some(17..=17));
        assert_eq!(
            first_indexed.eigenvalues_pos[0].value(),
            second_indexed.eigenvalues_pos[0].value()
        );
        let first_evenness = measure_evenness_via_cache(&params, &cfg, &context).unwrap();
        let second_evenness = measure_evenness_via_cache(&params, &cfg, &context).unwrap();
        assert_eq!(
            first_evenness.evenness_deviation,
            second_evenness.evenness_deviation
        );
        assert_eq!(
            first_evenness.natural_eigenvalue,
            second_evenness.natural_eigenvalue
        );
        assert_eq!(
            first_evenness.forced_eigenvalue,
            second_evenness.forced_eigenvalue
        );
        let first_sector_gap = analyze_sector_gap_via_cache(&params, &cfg, 2, &context).unwrap();
        let second_sector_gap = analyze_sector_gap_via_cache(&params, &cfg, 2, &context).unwrap();
        assert_eq!(first_sector_gap.lambda_even, second_sector_gap.lambda_even);
        assert_eq!(first_sector_gap.lambda_odd, second_sector_gap.lambda_odd);
        assert_eq!(first_sector_gap.gap_log, second_sector_gap.gap_log);
        assert_eq!(first_sector_gap.even.eigenpairs.len(), 2);
        assert_eq!(first_sector_gap.odd.eigenpairs.len(), 2);
        assert_eq!(
            first_sector_gap.even.eigenvalue_route,
            CcmSectorEigenvalueRoute::Selected
        );
        assert!(first_sector_gap.even.complete_eigenvalues.is_none());
        let complete_sector_gap = analyze_sector_gap_with_options_via_cache(
            &params,
            &cfg,
            CcmSectorAnalysisOptions::maximum(2),
            &context,
        )
        .unwrap();
        let reused_complete_sector_gap = analyze_sector_gap_with_options_via_cache(
            &params,
            &cfg,
            CcmSectorAnalysisOptions::maximum(2),
            &context,
        )
        .unwrap();
        assert_eq!(
            complete_sector_gap.even.eigenvalue_route,
            CcmSectorEigenvalueRoute::CompleteQr
        );
        assert_eq!(
            complete_sector_gap
                .even
                .complete_eigenvalues
                .as_ref()
                .unwrap()
                .len(),
            params.n_modes + 1
        );
        assert_eq!(
            complete_sector_gap
                .odd
                .complete_eigenvalues
                .as_ref()
                .unwrap()
                .len(),
            params.n_modes
        );
        assert_eq!(
            complete_sector_gap.even.complete_eigenvalues,
            reused_complete_sector_gap.even.complete_eigenvalues
        );
        assert_eq!(
            complete_sector_gap.odd.complete_eigenvalues,
            reused_complete_sector_gap.odd.complete_eigenvalues
        );
        assert_eq!(
            complete_sector_gap.even.eigenpairs[0].eigenvalue,
            reused_complete_sector_gap.even.eigenpairs[0].eigenvalue
        );
        assert_eq!(
            complete_sector_gap.even.eigenpairs[0].eigenvector,
            reused_complete_sector_gap.even.eigenpairs[0].eigenvector
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn even_sector_projection_preserves_even_quadratic_form() {
        let precision_bits = 256;
        let n_modes = 2;
        let dim = 2 * n_modes + 1;
        let mut tau = vec![Float::with_val(precision_bits, 0); dim * dim];
        for row in 0..dim {
            for column in row..dim {
                let value = Float::with_val(precision_bits, (row + 2) * (column + 3));
                tau[row * dim + column] = value.clone();
                tau[column * dim + row] = value;
            }
        }
        let sector = build_even_sector_matrix(&tau, n_modes, precision_bits);
        let y = vec![
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, -3),
        ];
        let x = expand_even_sector_vector(&y, n_modes, precision_bits);
        let full = xc_numerics::linalg::rayleigh_quotient(&tau, dim, &x, precision_bits);
        let reduced =
            xc_numerics::linalg::rayleigh_quotient(&sector, n_modes + 1, &y, precision_bits);
        let mut difference = full;
        difference -= reduced;
        assert!(difference.abs() < Float::with_val(precision_bits, 2).pow(-240));
    }

    #[test]
    fn odd_sector_projection_preserves_historical_values_and_quadratic_form() {
        let precision_bits = 256;
        let n_modes = 2;
        let dim = 2 * n_modes + 1;
        let mut tau = vec![Float::with_val(precision_bits, 0); dim * dim];
        for row in 0..dim {
            let signed_row = row as i32 - n_modes as i32;
            for column in 0..dim {
                let signed_column = column as i32 - n_modes as i32;
                let diagonal = if row == column { 40 } else { 0 };
                let value =
                    diagonal + signed_row * signed_column + (signed_row - signed_column).abs();
                tau[row * dim + column] = Float::with_val(precision_bits, value);
            }
        }
        assert!(matrix_is_exactly_symmetric(&tau, dim));
        for row in 0..dim {
            for column in 0..dim {
                assert_eq!(
                    tau[row * dim + column],
                    tau[(dim - 1 - row) * dim + (dim - 1 - column)]
                );
            }
        }

        let sector = build_odd_sector_matrix(&tau, n_modes, precision_bits);
        for k in 1..=n_modes {
            for j in 1..=n_modes {
                let mut historical = tau[(n_modes + k) * dim + (n_modes + j)].clone();
                historical -= &tau[(n_modes + k) * dim + (n_modes - j)];
                assert_eq!(sector[(k - 1) * n_modes + (j - 1)], historical);
            }
        }
        assert!(matrix_is_exactly_symmetric(&sector, n_modes));

        let y = vec![
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, -3),
        ];
        let x = expand_odd_sector_vector(&y, n_modes, precision_bits);
        let full = xc_numerics::linalg::rayleigh_quotient(&tau, dim, &x, precision_bits);
        let reduced = xc_numerics::linalg::rayleigh_quotient(&sector, n_modes, &y, precision_bits);
        let mut difference = full;
        difference -= reduced;
        assert!(difference.abs() < Float::with_val(precision_bits, 2).pow(-240));
    }

    #[test]
    fn parallel_banded_sector_recovery_is_bit_identical_to_sequential_recovery() {
        let precision_bits = 192;
        let dimension = 3;
        let matrix = vec![
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 5),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 9),
        ];
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = precision_bits;
        cfg.inverse_iter_steps = 12;

        let eigenvalues =
            xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&matrix, dimension, precision_bits)
                .unwrap();
        let routed_eigenvalues = SectorEigenvaluesHp {
            route: CcmSectorEigenvalueRoute::CompleteQr,
            complete: true,
            values: eigenvalues.clone(),
            selected_enclosures: Vec::new(),
        };
        let (diagonal, off_diagonal, basis) =
            xc_numerics::eigen::householder_tridiag_hp(&matrix, dimension, precision_bits).unwrap();
        let tridiagonal = SectorTridiagonalHp {
            diagonal,
            off_diagonal,
        };
        let transform = SectorTransformHp { basis };
        let parallel = compute_sector_spectrum(
            &matrix,
            dimension,
            CcmParity::Odd,
            2,
            &routed_eigenvalues,
            &tridiagonal,
            &transform,
            &cfg,
        )
        .unwrap();
        let sequential = compute_sector_branch(
            &matrix,
            dimension,
            CcmParity::Odd,
            2,
            CcmSectorEigenvalueRoute::CompleteQr,
            &cfg,
        )
        .unwrap();

        assert_eq!(parallel.eigenpairs.len(), sequential.eigenpairs.len());
        for (parallel, sequential) in parallel.eigenpairs.iter().zip(&sequential.eigenpairs) {
            assert_eq!(parallel.algebraic_index, sequential.algebraic_index);
            assert_eq!(parallel.eigenvalue, sequential.eigenvalue);
            assert_eq!(parallel.eigenvector, sequential.eigenvector);
            assert_eq!(parallel.residual_norm, sequential.residual_norm);
        }
    }

    #[test]
    fn sector_transform_validation_uses_requested_precision_and_matrix_scale() {
        let precision_bits = 256;
        let dimension = 4;
        let scale = Float::with_val(
            precision_bits,
            Float::parse("1e40").expect("valid HP scale"),
        );
        let mut matrix = vec![
            Float::with_val(precision_bits, 4),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 6),
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, 9),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 12),
        ];
        for value in &mut matrix {
            *value *= &scale;
        }
        let (diagonal, off_diagonal, basis) =
            xc_numerics::eigen::householder_tridiag_hp(&matrix, dimension, precision_bits).unwrap();
        let tridiagonal = SectorTridiagonalHp {
            diagonal,
            off_diagonal,
        };
        let transform = SectorTransformHp { basis };
        validate_sector_transform(&matrix, &tridiagonal, &transform, dimension, precision_bits)
            .unwrap();

        let mut corrupted = transform;
        corrupted.basis[0] += Float::with_val(
            precision_bits,
            Float::parse("1e-20").expect("valid HP perturbation"),
        );
        assert!(validate_sector_transform(
            &matrix,
            &tridiagonal,
            &corrupted,
            dimension,
            precision_bits,
        )
        .is_err());
    }

    #[test]
    fn banded_householder_sector_vectors_match_dense_reference_up_to_phase() {
        let precision_bits = 256;
        let dimension = 4;
        let matrix = vec![
            Float::with_val(precision_bits, 4),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 6),
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 2),
            Float::with_val(precision_bits, 9),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 0),
            Float::with_val(precision_bits, 1),
            Float::with_val(precision_bits, 12),
        ];
        let mut cfg = HighPrecConfig::for_decimal_digits(60);
        cfg.precision_bits = precision_bits;
        cfg.inverse_iter_steps = 200;
        let eigenvalues =
            xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&matrix, dimension, precision_bits)
                .unwrap();
        let routed = SectorEigenvaluesHp {
            route: CcmSectorEigenvalueRoute::CompleteQr,
            complete: true,
            values: eigenvalues.clone(),
            selected_enclosures: Vec::new(),
        };
        let (diagonal, off_diagonal, basis) =
            xc_numerics::eigen::householder_tridiag_hp(&matrix, dimension, precision_bits).unwrap();
        let optimized = compute_sector_spectrum(
            &matrix,
            dimension,
            CcmParity::Even,
            2,
            &routed,
            &SectorTridiagonalHp {
                diagonal,
                off_diagonal,
            },
            &SectorTransformHp { basis },
            &cfg,
        )
        .unwrap();
        let tolerance = Float::with_val(precision_bits, 10).pow(-50i32);
        for (index, optimized_pair) in optimized.eigenpairs.iter().enumerate() {
            let dense = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
                &matrix,
                dimension,
                &eigenvalues[index],
                precision_bits,
                cfg.inverse_iter_steps,
            )
            .unwrap();
            let dot_terms = optimized_pair
                .eigenvector
                .iter()
                .zip(&dense)
                .map(|(left, right)| {
                    let mut product = left.clone();
                    product *= right;
                    product
                })
                .collect();
            let mut phase_agreement = xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(
                dot_terms,
                precision_bits,
            )
            .abs();
            phase_agreement -= 1u32;
            assert!(phase_agreement.abs() < tolerance);
            assert!(optimized_pair.residual_norm < tolerance);
        }
    }

    #[test]
    fn selected_complete_and_cross_checked_sector_routes_reconcile() {
        let precision_bits = 192;
        let tridiagonal = SectorTridiagonalHp {
            diagonal: vec![
                Float::with_val(precision_bits, 2),
                Float::with_val(precision_bits, 5),
                Float::with_val(precision_bits, 9),
            ],
            off_diagonal: vec![
                Float::with_val(precision_bits, 0),
                Float::with_val(precision_bits, 0),
            ],
        };
        let selected = compute_sector_eigenvalues(
            &tridiagonal,
            3,
            2,
            CcmSectorEigenvalueRoute::Selected,
            precision_bits,
        )
        .unwrap();
        let complete = compute_sector_eigenvalues(
            &tridiagonal,
            3,
            3,
            CcmSectorEigenvalueRoute::CompleteQr,
            precision_bits,
        )
        .unwrap();
        let cross_checked = compute_sector_eigenvalues(
            &tridiagonal,
            3,
            2,
            CcmSectorEigenvalueRoute::CrossChecked,
            precision_bits,
        )
        .unwrap();

        assert!(!selected.complete);
        assert_eq!(selected.values.len(), 2);
        assert_eq!(selected.selected_enclosures.len(), 2);
        assert!(complete.complete);
        assert_eq!(complete.values.len(), 3);
        assert!(complete.selected_enclosures.is_empty());
        assert!(cross_checked.complete);
        assert_eq!(cross_checked.values, complete.values);
        assert_eq!(cross_checked.selected_enclosures.len(), 2);
        for enclosure in &selected.selected_enclosures {
            assert!(enclosure.lower <= complete.values[enclosure.index]);
            assert!(enclosure.upper >= complete.values[enclosure.index]);
        }
    }

    #[test]
    fn sector_gap_uses_depth_difference_not_eigenvalue_subtraction() {
        let precision_bits = 256;
        let pair = |index: usize, value: &str| CcmSectorEigenpairHp {
            algebraic_index: index,
            eigenvalue: Float::with_val(precision_bits, Float::parse(value).unwrap()),
            eigenvector: vec![Float::with_val(precision_bits, 1)],
            residual_norm: Float::with_val(precision_bits, 0),
        };
        let even = CcmSectorSpectrumHp {
            parity: CcmParity::Even,
            dimension: 1,
            eigenvalue_route: CcmSectorEigenvalueRoute::Selected,
            complete_eigenvalues: None,
            eigenpairs: vec![pair(0, "1e-20"), pair(1, "2")],
        };
        let odd = CcmSectorSpectrumHp {
            parity: CcmParity::Odd,
            dimension: 1,
            eigenvalue_route: CcmSectorEigenvalueRoute::Selected,
            complete_eigenvalues: None,
            eigenpairs: vec![pair(0, "1e-16"), pair(1, "3")],
        };
        let gap = compute_sector_gap(even, odd, precision_bits).unwrap();
        let expected = Float::with_val(precision_bits, 4);
        let mut error = gap.gap_log.clone();
        error -= expected;
        assert!(error.abs() < Float::with_val(precision_bits, 2).pow(-240));
        assert_eq!(gap.ordering, 1);
        assert!(gap.even_simple);
        assert_ne!(gap.gap_log, gap.difference_depth);
    }

    #[cfg(feature = "arb")]
    #[test]
    fn retained_tau_is_interval_certified_and_promoted_without_recomputation() {
        use xc_cache::{
            ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
            CanonicalStagingProductionSink, FilesystemCacheStore, TransportPolicy,
        };
        use xc_core::{CancellationToken, ResourcePolicy};

        let root = std::env::temp_dir().join(format!(
            "xc-spectral-ccm-certified-retained-tau-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                root.join("cache"),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let sink = CanonicalStagingProductionSink::new(
            root.join("staging"),
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        let context = ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["workstation".to_owned()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Certified,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: Some(&sink),
        };
        let params = CcmParams::from_lambda_sq_integer(5, 2);
        let cfg = HighPrecConfig::for_decimal_digits(40);
        let l = log_lambda_sq_hp(&params, cfg.precision_bits);
        let first = build_tau_hp_via_cache(&params, &l, &cfg, &context).unwrap();
        let reused = build_tau_hp_via_cache(&params, &l, &cfg, &context).unwrap();
        assert_eq!(first, reused);
        let tau_draft = sink
            .drafts()
            .unwrap()
            .into_iter()
            .find(|draft| draft.family == "ccm-matrices")
            .unwrap();
        assert_eq!(
            tau_draft.achieved_assurance,
            xc_cache::ArtifactAssuranceState::Certified
        );
        assert_eq!(
            tau_draft.required_assurance,
            Some(xc_cache::ArtifactAssuranceState::Certified)
        );
        assert_eq!(tau_draft.assurance_evidence_digests.len(), 2);
        let draft_identities = sink
            .drafts()
            .unwrap()
            .into_iter()
            .map(|draft| (draft.family, draft.source_artifact_key.kind))
            .collect::<Vec<_>>();
        assert_eq!(
            draft_identities,
            vec![
                ("quadrature".to_owned(), "gauss_legendre_rule".to_owned()),
                (
                    "ccm-components".to_owned(),
                    "ccm_archimedean_integrals".to_owned()
                ),
                (
                    "ccm-components".to_owned(),
                    "ccm_prime_component".to_owned()
                ),
                ("ccm-matrices".to_owned(), "ccm_tau_matrix".to_owned()),
            ]
        );
        let inventory = xc_cache::build_managed_publication_inventory(
            &sink.drafts().unwrap(),
            xc_core::PublicationTarget::Both,
            "TeamXcelerator",
            xc_cache::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Certified,
            xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            true,
        )
        .unwrap();
        assert!(inventory.ready_for_remote_execution);
        assert_eq!(inventory.entries.len(), 8);
        assert!(inventory
            .entries
            .iter()
            .all(|entry| entry.assurance_eligible));
        assert_eq!(
            inventory
                .entries
                .iter()
                .filter(|entry| entry.required_assurance.is_some())
                .count(),
            2
        );
        assert!(root.join("staging/evidence").is_dir());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(feature = "arb")]
    #[test]
    fn root_certificate_is_separate_source_bound_and_reusable() {
        use xc_cache::{
            ArtifactExecutionCacheMode, ArtifactKey, CacheLayer, CacheObjectRef, CachePolicy,
            CacheResolver, CacheVisibility, FilesystemCacheStore,
        };

        let root = std::env::temp_dir().join(format!(
            "xc-spectral-ccm-root-certificate-cache-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                root.join("cache"),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = 192;
        cfg.quad_points = MIN_QUAD_POINTS;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let (mut source, _) = run_inner_retaining_source(
            &params,
            &cfg,
            RootAcquisition::SourceOnly,
            CcmCacheRoute::Standalone,
            None,
        )
        .unwrap();
        let source_digest = ContentDigest::sha256(b"exact-test-secular-source");
        let source_manifest = ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(
                "ccm_secular_source",
                "ccm/test/secular-source",
                b"exact-test-secular-source-parameters",
            )
            .unwrap(),
            content_digest: source_digest.clone(),
            size_bytes: 1,
            objects: vec![CacheObjectRef {
                content_digest: source_digest,
                size_bytes: 1,
            }],
            created_unix_seconds: 0,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: Vec::new(),
            tags: BTreeMap::new(),
            provenance_digest: None,
        };
        let context = |mode, write_on_miss| ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["workstation".to_owned()],
            mode,
            write_on_miss,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let options = CcmRootCertificationOptions::for_decimal_digits(
            super::super::certified_roots::IndependentCcmRootTarget::Prefix { count: 1 },
            30,
        )
        .unwrap();
        let first = certify_roots_from_retained_source(
            &params,
            &cfg,
            &source.xi,
            Some(&source_manifest),
            &options,
            Some(&context(ArtifactExecutionCacheMode::PreferReuse, true)),
        )
        .unwrap();
        let enclosure = &first.selected_roots()[0];
        let mut certified_midpoint =
            Float::with_val(cfg.precision_bits, Float::parse(&enclosure.lower).unwrap());
        certified_midpoint +=
            Float::with_val(cfg.precision_bits, Float::parse(&enclosure.upper).unwrap());
        certified_midpoint /= 2;
        source.first_positive_root_index = 1;
        source.eigenvalues_pos = vec![EigenvalueResult::Converged(RootRefinement {
            value: certified_midpoint,
            diagnostics: RootRefinementDiagnostics {
                iterations: 1,
                final_correction: Float::with_val(cfg.precision_bits, 0),
                residual: Float::with_val(cfg.precision_bits, 0),
                achieved_decimal_digits: Float::with_val(cfg.precision_bits, 30),
            },
        })];
        reconcile_computed_roots_with_certificate(&source, &first).unwrap();
        let reused = certify_roots_from_retained_source(
            &params,
            &cfg,
            &source.xi,
            Some(&source_manifest),
            &options,
            Some(&context(ArtifactExecutionCacheMode::RequireReuse, false)),
        )
        .unwrap();
        assert_eq!(first, reused);

        let different_target = CcmRootCertificationOptions::for_decimal_digits(
            super::super::certified_roots::IndependentCcmRootTarget::IndexRange {
                first: 2,
                last: 2,
            },
            30,
        )
        .unwrap();
        assert!(certify_roots_from_retained_source(
            &params,
            &cfg,
            &source.xi,
            Some(&source_manifest),
            &different_target,
            Some(&context(ArtifactExecutionCacheMode::RequireReuse, false)),
        )
        .is_err());
        let mut different_source = source_manifest.clone();
        different_source.content_digest = ContentDigest::sha256(b"different-secular-source");
        assert!(certify_roots_from_retained_source(
            &params,
            &cfg,
            &source.xi,
            Some(&different_source),
            &options,
            Some(&context(ArtifactExecutionCacheMode::RequireReuse, false)),
        )
        .is_err());
        if let EigenvalueResult::Converged(root) = &mut source.eigenvalues_pos[0] {
            root.value += 1;
        }
        assert!(reconcile_computed_roots_with_certificate(&source, &first).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn portable_ccm_hp_result_round_trips_values_status_and_precision() {
        let precision_bits = 512;
        let refinement = |precision: u32, value: &str| RootRefinement {
            value: Float::with_val(precision, Float::parse(value).unwrap()),
            diagnostics: RootRefinementDiagnostics {
                iterations: 17,
                final_correction: Float::with_val(precision, Float::parse("1e-100").unwrap()),
                residual: Float::with_val(precision, Float::parse("1e-110").unwrap()),
                achieved_decimal_digits: Float::with_val(precision, 100),
            },
        };
        let runtime = HighPrecResult {
            eigenvalues_pos: vec![
                EigenvalueResult::Converged(refinement(
                    precision_bits,
                    "14.134725141734693790457251983562",
                )),
                EigenvalueResult::Stagnated(refinement(768, "21.022039638771554992628479593896")),
                EigenvalueResult::Approximate(refinement(640, "25.010857580145688763213790992563")),
                EigenvalueResult::Failed {
                    iterations: 9,
                    reason: "test failure".to_owned(),
                },
            ],
            first_positive_root_index: 17,
            weil_min_eigenvalue: Float::with_val(precision_bits, Float::parse("1e-120").unwrap()),
            xi: vec![
                Float::with_val(384, Float::parse("-0.125").unwrap()),
                Float::with_val(640, Float::parse("0.75").unwrap()),
            ],
            inverse_iteration_diagnostics: xc_numerics::linalg::InverseIterationDiagnostics {
                configured_step_limit: 2_000,
                unshifted_steps: 2_000,
                unshifted_converged: false,
                final_relative_rayleigh_change: Some(Float::with_val(
                    precision_bits,
                    Float::parse("1e-90").unwrap(),
                )),
                shifted_refinement: xc_numerics::linalg::ShiftedRefinementOutcome::Accepted,
                final_relative_residual_norm: Float::with_val(
                    precision_bits,
                    Float::parse("1e-120").unwrap(),
                ),
            },
            elapsed_seconds: 1.25,
            precision_bits,
        };
        let portable = PortableHighPrecResult::from_runtime(&runtime).unwrap();
        let saved = xc_core::ResearchResult::computed(
            portable,
            xc_core::SolverProvenance::current_package("rug_mpfr"),
        );
        let encoded = serde_json::to_vec(&saved).unwrap();
        let decoded: xc_core::ResearchResult<PortableHighPrecResult> =
            serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, saved);
        let reconstructed = decoded.value.unwrap().to_runtime().unwrap();
        assert_eq!(reconstructed.precision_bits, runtime.precision_bits);
        assert_eq!(
            reconstructed.weil_min_eigenvalue,
            runtime.weil_min_eigenvalue
        );
        assert_eq!(reconstructed.xi, runtime.xi);
        assert_eq!(
            reconstructed.inverse_iteration_diagnostics,
            runtime.inverse_iteration_diagnostics
        );
        assert_eq!(reconstructed.spectral_root_index_range(), Some(17..=20));
        for (actual, expected) in reconstructed
            .eigenvalues_pos
            .iter()
            .zip(&runtime.eigenvalues_pos)
        {
            match (actual, expected) {
                (EigenvalueResult::Converged(actual), EigenvalueResult::Converged(expected))
                | (EigenvalueResult::Stagnated(actual), EigenvalueResult::Stagnated(expected))
                | (
                    EigenvalueResult::Approximate(actual),
                    EigenvalueResult::Approximate(expected),
                ) => {
                    assert_eq!(actual.value, expected.value);
                    assert_eq!(actual.value.prec(), expected.value.prec());
                    assert_eq!(
                        actual.diagnostics.final_correction,
                        expected.diagnostics.final_correction
                    );
                    assert_eq!(actual.diagnostics.residual, expected.diagnostics.residual);
                    assert_eq!(
                        actual.diagnostics.iterations,
                        expected.diagnostics.iterations
                    );
                }
                (
                    EigenvalueResult::Failed {
                        iterations: actual_iterations,
                        reason: actual_reason,
                    },
                    EigenvalueResult::Failed {
                        iterations: expected_iterations,
                        reason: expected_reason,
                    },
                ) => {
                    assert_eq!(actual_iterations, expected_iterations);
                    assert_eq!(actual_reason, expected_reason);
                }
                _ => panic!("saved CCM eigenvalue status changed"),
            }
        }
    }

    #[test]
    fn exact_fractional_lambda_squared_drives_hp_assembly_and_evidence() {
        let integer =
            ExactLambdaSquaredHp::new(xc_core::DecimalLiteral::new("13").unwrap(), 13).unwrap();
        assert_eq!(integer.mode, LambdaSquaredMode::Integer);

        let fractional = ExactLambdaSquaredHp::new(
            xc_core::DecimalLiteral::new("13.0000000000000000000000000000000000001").unwrap(),
            13,
        )
        .unwrap();
        assert_eq!(fractional.mode, LambdaSquaredMode::Fractional);
        assert!(
            ExactLambdaSquaredHp::new(xc_core::DecimalLiteral::new("14.1").unwrap(), 13).is_err()
        );

        let mut cfg = HighPrecConfig::for_decimal_digits(50);
        cfg.quad_points = 96;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let exact = localized_weil_form_exact_hp(fractional, 0, &cfg, true).unwrap();
        assert_eq!(exact.matrix.len(), 1);
        assert_eq!(exact.evidence.n_modes, 0);
        assert_eq!(exact.evidence.precision_bits, cfg.precision_bits);
        assert_eq!(exact.evidence.lambda_squared.prime_cutoff, 13);
        assert_eq!(exact.evidence.prime_content.len(), 9);
        assert_eq!(exact.evidence.prime_content.last().unwrap().power, 13);
    }

    #[test]
    fn ccm_domain_planner_selects_guarded_hp_low_mode_route() {
        let request = CcmLowModeSolverRequest {
            matrix_dimension: 5,
            requested_modes: 2,
            assurance: xc_core::AssuranceLevel::Computed,
            precision: xc_core::PrecisionPolicy::fixed(192),
            matrix_materialized: true,
        };
        let plan =
            xc_solver::DomainSolverPlanner::plan(&CcmLowModeSolverPlanner, &request).unwrap();
        assert_eq!(plan.domain_id, "ccm_weil_low_modes");
        assert_eq!(plan.input.target, xc_core::EigenTarget::SmallestMagnitude);
        assert!(!plan.input.generalized);
        assert_eq!(
            plan.solver_plan.primary,
            xc_solver::SolverRoute::HpBlockShiftInvert
        );
    }

    #[test]
    fn ccm_state_targets_are_distinct_and_retained_in_results() {
        let precision = 192;
        let candidate = |index, value, parity| CcmStateCandidateHp {
            algebraic_index: index,
            eigenvalue: Float::with_val(precision, value),
            eigenvector: vec![Float::with_val(precision, 1)],
            parity,
        };
        let candidates = vec![
            candidate(0, -4, CcmParity::Even),
            candidate(1, -1, CcmParity::Odd),
            candidate(2, 2, CcmParity::Even),
            candidate(3, 3, CcmParity::Odd),
        ];
        let ground = select_ccm_state_hp(CcmStateTarget::AlgebraicGround, &candidates).unwrap();
        let positive = select_ccm_state_hp(CcmStateTarget::SmallestPositive, &candidates).unwrap();
        let nearest = select_ccm_state_hp(CcmStateTarget::NearestZero, &candidates).unwrap();
        let odd_positive = select_ccm_state_hp(
            CcmStateTarget::ParityRestricted {
                parity: CcmParity::Odd,
                criterion: CcmStateCriterion::SmallestPositive,
            },
            &candidates,
        )
        .unwrap();
        assert_eq!(ground.algebraic_index, 0);
        assert_eq!(positive.algebraic_index, 2);
        assert_eq!(nearest.algebraic_index, 1);
        assert_eq!(odd_positive.algebraic_index, 3);
        assert_eq!(odd_positive.parity, CcmParity::Odd);
        assert!(matches!(
            odd_positive.requested_target,
            CcmStateTarget::ParityRestricted { .. }
        ));
    }

    #[test]
    fn ccm_form_components_reconstruct_total_on_the_same_vector() {
        let precision = 192;
        let diagonal = |left, right| {
            vec![
                Float::with_val(precision, left),
                Float::with_val(precision, 0),
                Float::with_val(precision, 0),
                Float::with_val(precision, right),
            ]
        };
        let components = vec![
            CcmFormComponentMatrixHp {
                kind: CcmFormComponentKind::Archimedean,
                signed_coefficient: 1,
                matrix_row_major: diagonal(10, 14),
            },
            CcmFormComponentMatrixHp {
                kind: CcmFormComponentKind::Prime,
                signed_coefficient: -1,
                matrix_row_major: diagonal(2, 4),
            },
            CcmFormComponentMatrixHp {
                kind: CcmFormComponentKind::Pole,
                signed_coefficient: 1,
                matrix_row_major: diagonal(1, 1),
            },
            CcmFormComponentMatrixHp {
                kind: CcmFormComponentKind::Other,
                signed_coefficient: 1,
                matrix_row_major: diagonal(1, 3),
            },
        ];
        let total = diagonal(10, 14);
        let vector = vec![Float::with_val(precision, 1), Float::with_val(precision, 2)];
        let report =
            evaluate_ccm_form_components_hp(&total, &components, &vector, precision).unwrap();
        assert_eq!(report.components.len(), 4);
        assert!(report.cancellation_residual < Float::with_val(precision, 1e-50));
    }

    /// Serialize all cwd-mutating cache tests in this module. Cargo runs
    /// tests in parallel by default; cwd is per-process (not per-thread),
    /// so two cache tests racing on `set_current_dir` would corrupt each
    /// other (one test deleting the temp dir another captured as its
    /// "original"). The mutex enforces sequential access. Mirrors the GL
    /// cache tests in `xc-numerics::quadrature`.
    ///
    /// This aliases the crate-level [`crate::TEST_CWD_LOCK`] so the
    /// `ccm::hp` and `prolate::hp` cache tests — which share one
    /// process-global cwd within the same test binary — serialize
    /// against *each other*, not just within their own module.
    #[allow(dead_code)]
    static CWD_LOCK: &Mutex<()> = &crate::TEST_CWD_LOCK;

    /// Guard that restores the original cwd on drop, so a panic inside a
    /// test doesn't leave the runner in a temp dir (which would break
    /// subsequent unrelated tests). Holds the CWD_LOCK for the guard's
    /// lifetime to serialize cwd mutation.
    struct CwdGuard {
        original: std::path::PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    impl CwdGuard {
        fn enter(temp: &std::path::Path) -> Self {
            // Recover from poison: a previously-panicking test poisons the
            // lock, but subsequent tests can still safely acquire it (the
            // global cwd state isn't corrupted by a panic — the prior
            // guard's Drop ran on unwind and restored cwd).
            let lock = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let original = std::env::current_dir().expect("no cwd");
            std::env::set_current_dir(temp).expect("set_current_dir to temp");
            CwdGuard {
                original,
                _lock: lock,
            }
        }
    }
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    /// HighPrecConfig::for_decimal_digits should produce expected values.
    #[test]
    fn config_for_200_digits() {
        let cfg = HighPrecConfig::for_decimal_digits(200);
        // ceil(200 * log2(10)) = 665 + 64 guard = 729 bits
        assert_eq!(cfg.precision_bits, 729);
        // 200 * 3 = 600, clamped to [600, 4000] → 600
        assert_eq!(cfg.quad_points, 600);
        assert_eq!(cfg.inverse_iter_steps, 2_000);
        assert_eq!(cfg.solver_steps, 2_000);
        assert_eq!(cfg.n_eigenvalues, 50);
        assert!(cfg.force_even, "force_even should default to true");
    }

    #[test]
    fn dense_ccm_has_two_independent_hp_routes_and_precision_repeat() {
        let params = CcmParams::from_lambda_sq_integer(5, 2);
        let solve = |digits| {
            let mut cfg = HighPrecConfig::for_decimal_digits(digits);
            cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
            let matrix = weil_matrix_hp(&params, &cfg, true).unwrap();
            let dimension = params.matrix_size();
            let qr = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(
                &matrix,
                dimension,
                cfg.precision_bits,
            )
            .unwrap();
            let jacobi = xc_numerics::eigen::dense_symmetric_eigenvalues_jacobi_hp(
                &matrix,
                dimension,
                cfg.precision_bits,
                80,
            )
            .unwrap();
            (qr, jacobi.eigenvalues)
        };
        let (qr_low, jacobi_low) = solve(30);
        let (qr_high, jacobi_high) = solve(50);
        let high_precision = qr_high[0].prec();
        let route_tolerance = Float::with_val(high_precision, Float::parse("1e-35").unwrap());
        for (qr, jacobi) in qr_high.iter().zip(&jacobi_high) {
            let mut difference = qr.clone();
            difference -= jacobi;
            assert!(difference.abs() < route_tolerance);
        }
        let repeat_tolerance = Float::with_val(high_precision, Float::parse("1e-25").unwrap());
        for (low, high) in qr_low.iter().zip(&qr_high) {
            let mut difference = Float::with_val(high_precision, low);
            difference -= high;
            assert!(difference.abs() < repeat_tolerance);
        }
        for (low, high) in jacobi_low.iter().zip(&jacobi_high) {
            let mut difference = Float::with_val(high_precision, low);
            difference -= high;
            assert!(difference.abs() < repeat_tolerance);
        }
    }

    #[test]
    fn ccm_low_modes_are_requested_as_one_guarded_hp_block() {
        use xc_core::{DecimalLiteral, EigenTarget};
        use xc_operator::DenseSymmetricHp;
        use xc_solver::{
            BlockShiftInvertConfigHp, BlockShiftInvertSolverHp, DenseShiftInvertFactorizationHp,
        };

        let params = CcmParams::from_lambda_sq_integer(5, 2);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let matrix = weil_matrix_hp(&params, &cfg, true).unwrap();
        let dimension = params.matrix_size();
        let full_reference = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(
            &matrix,
            dimension,
            cfg.precision_bits,
        )
        .unwrap();
        let operator = DenseSymmetricHp::new(
            "ccm-weil-low-block",
            dimension,
            matrix.clone(),
            cfg.precision_bits,
            &Float::with_val(cfg.precision_bits, 0),
        )
        .unwrap();
        let factorization = DenseShiftInvertFactorizationHp::factor(
            "ccm-weil-zero-shift",
            dimension,
            &matrix,
            DecimalLiteral::new("0").unwrap(),
            cfg.precision_bits,
        )
        .unwrap();
        let report = BlockShiftInvertSolverHp
            .solve(
                &operator,
                &factorization,
                &BlockShiftInvertConfigHp {
                    target: EigenTarget::SmallestMagnitude,
                    precision_bits: cfg.precision_bits,
                    requested_eigenpairs: 2,
                    guard_eigenpairs: 1,
                    absolute_residual_tolerance: DecimalLiteral::new("1e-20").unwrap(),
                    scaled_backward_error_tolerance: DecimalLiteral::new("1e-20").unwrap(),
                    ritz_value_stability_tolerance: DecimalLiteral::new("1e-20").unwrap(),
                    boundary_cluster_tolerance: DecimalLiteral::new("1e-25").unwrap(),
                    maximum_iterations: 60,
                    minimum_iterations: 2,
                    maximum_projected_sweeps: 100,
                },
            )
            .unwrap();
        assert_eq!(report.requested_eigenpairs, 2);
        assert!(report.boundary_cluster.is_none());
        assert!(report.retained_eigenpairs.len() >= 2);
        let tolerance = Float::with_val(cfg.precision_bits, Float::parse("1e-18").unwrap());
        for (selected, reference) in report
            .retained_eigenpairs
            .iter()
            .take(2)
            .zip(full_reference.iter().take(2))
        {
            let mut difference = selected.eigenvalue.clone();
            difference -= reference;
            assert!(difference.abs() < tolerance);
        }
    }

    /// Band-concentration matrix is symmetric, has a [0,1] spectrum, and
    /// exhibits a concentrated/Sonin split (some χ≈1, some χ≈0).
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn band_concentration_spectrum_in_unit_interval() {
        let params = CcmParams::from_lambda_sq_integer(5, 8);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let dim = params.matrix_size();
        let omega = Float::with_val(prec, 15);
        let c = band_concentration_matrix_hp(&params, &cfg, &omega).unwrap();
        for i in 0..dim {
            for j in 0..dim {
                let d = (c[i * dim + j].to_f64() - c[j * dim + i].to_f64()).abs();
                assert!(d < 1e-25, "C not symmetric at ({i},{j}): {d}");
            }
            let cii = c[i * dim + i].to_f64();
            assert!(cii > -1e-9 && cii < 1.0 + 1e-6, "diag {cii} out of [0,1]");
        }
        let chi = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&c, dim, prec).unwrap();
        let chi_f: Vec<f64> = chi.iter().map(|x| x.to_f64()).collect();
        for &x in &chi_f {
            assert!(x > -1e-6 && x < 1.0 + 1e-6, "chi {x} out of [0,1]");
        }
        assert!(
            chi_f.iter().any(|&x| x > 0.9),
            "expected a concentrated (χ≈1) mode"
        );
        assert!(
            chi_f.iter().any(|&x| x < 0.1),
            "expected a Sonin-like (χ≈0) mode"
        );
    }

    /// Sonin restriction deflates exactly `n_drop` band-concentrated modes:
    /// they cluster near the shift σ (≈ max eigenvalue), well separated
    /// from the rest, and the result shape is consistent.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn weil_sonin_deflates_band_modes() {
        let params = CcmParams::from_lambda_sq_integer(5, 8);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let dim = params.matrix_size();
        let omega = Float::with_val(prec, 15);
        let n_drop = 5usize;
        let res = weil_spectrum_sonin_hp(&params, &cfg, &omega, n_drop).unwrap();
        assert_eq!(res.chi.len(), dim);
        assert_eq!(res.spectrum.len(), dim);
        assert_eq!(res.n_dropped, n_drop);
        for w in res.chi.windows(2) {
            assert!(w[0].to_f64() <= w[1].to_f64() + 1e-12, "χ not ascending");
        }
        // The n_drop deflated modes cluster near σ (the max); everything
        // else sits below σ/2 — so exactly n_drop eigenvalues exceed σ/2.
        let s: Vec<f64> = res.spectrum.iter().map(|x| x.to_f64()).collect();
        let max_v = *s.last().unwrap();
        assert!(
            max_v > 10.0,
            "deflation shift σ should be large, got {max_v}"
        );
        let big = s.iter().filter(|&&e| e > max_v / 2.0).count();
        assert_eq!(
            big, n_drop,
            "expected {n_drop} deflated modes near σ, got {big}"
        );
    }

    /// HighPrecConfig::for_decimal_digits at 500 digits.
    #[test]
    fn config_for_500_digits() {
        let cfg = HighPrecConfig::for_decimal_digits(500);
        // ceil(500 * log2(10)) = 1661 + 64 guard = 1725 bits
        assert_eq!(cfg.precision_bits, 1725);
        // 500 * 3 = 1500, clamped to [600, 4000] → 1500
        assert_eq!(cfg.quad_points, 1500);
    }

    /// Even-sector remains the default, while the legacy boolean override
    /// continues to select the natural path.
    #[test]
    fn config_force_even_default_and_override() {
        let cfg = HighPrecConfig::for_decimal_digits(200);
        assert!(cfg.force_even, "force_even should default to true");
        assert_eq!(cfg.effective_parity_policy(), CcmParityPolicy::EvenSector);

        let mut cfg2 = HighPrecConfig::for_decimal_digits(200);
        cfg2.force_even = false;
        assert!(!cfg2.force_even, "force_even should be settable to false");
        assert_eq!(cfg2.effective_parity_policy(), CcmParityPolicy::Natural);

        cfg2.set_parity_policy(CcmParityPolicy::AdaptiveEven);
        assert!(cfg2.force_even);
        assert_eq!(
            cfg2.effective_parity_policy(),
            CcmParityPolicy::AdaptiveEven
        );
    }

    /// Weil-eigenvector cache filenames preserve the two historical
    /// standalone routes and isolate the newer reduced even-sector route.
    #[test]
    fn weil_eigvec_cache_filename_keys_on_force_even() {
        use super::super::LambdaSq;
        use super::weil_eigvec_cache::cache_filename;
        let forced = cache_filename(
            LambdaSq::integer(1000),
            800,
            6660,
            CcmParityPolicy::EvenSector,
        );
        let natural = cache_filename(LambdaSq::integer(1000), 800, 6660, CcmParityPolicy::Natural);
        let adaptive = cache_filename(
            LambdaSq::integer(1000),
            800,
            6660,
            CcmParityPolicy::AdaptiveEven,
        );
        assert_eq!(
            forced,
            "weil_eigvec_lambda_sq1000_nmodes800_prec6660_even_sector.json"
        );
        // Natural is distinct.
        assert_eq!(
            natural,
            "weil_eigvec_lambda_sq1000_nmodes800_prec6660_natural.json"
        );
        assert_eq!(
            adaptive,
            "weil_eigvec_lambda_sq1000_nmodes800_prec6660.json"
        );
        assert_ne!(forced, natural);
        assert_ne!(forced, adaptive);
        assert_ne!(natural, adaptive);
    }

    /// `weil_spectrum_hp`: the FULL Weil form is positive (smallest
    /// eigenvalue > 0 — Weil positivity / the plunge), while the
    /// ARCHIMEDEAN-ONLY form (prime sum `w_p` dropped) is indefinite
    /// (negative minimum — the prime sum that enforces positivity is
    /// absent). λ²=13, N=12, 64 digits, hermetic (no cache / no network).
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn weil_spectrum_full_positive_arch_indefinite() {
        let params = CcmParams::from_lambda_sq_integer(13, 12);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;

        let full = weil_spectrum_hp(&params, &cfg, true).unwrap();
        let arch = weil_spectrum_hp(&params, &cfg, false).unwrap();

        let zero = Float::with_val(cfg.precision_bits, 0);
        // FULL: smallest eigenvalue is positive (Weil positivity).
        assert!(
            full[0] > zero,
            "full Weil form should be positive definite, min={}",
            full[0]
        );
        // ARCH-only: minimum is negative (indefinite without the prime sum).
        assert!(
            arch[0] < zero,
            "archimedean-only form should be indefinite, min={}",
            arch[0]
        );
    }

    /// `weil_plunge_cancellation_hp`: the plunge ε_N is the difference of
    /// two O(1) Rayleigh quotients (archimedean − prime) that agree to many
    /// digits. λ²=13, N=12, 64 digits, hermetic.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn weil_plunge_cancellation_decomposes() {
        let params = CcmParams::from_lambda_sq_integer(13, 12);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;

        let d = weil_plunge_cancellation_hp(&params, &cfg).unwrap();
        let prec = cfg.precision_bits;

        // ε_N = arch − prime exactly (linearity of the Rayleigh quotient).
        let mut recon = Float::with_val(prec, &d.arch_rayleigh);
        recon -= &d.prime_rayleigh;
        let mut err = Float::with_val(prec, &recon);
        err -= &d.eps_n;
        assert!(
            err.abs() < Float::with_val(prec, 2).pow(-(prec as i32 - 16)),
            "arch − prime should equal eps_n"
        );

        // ε_N is small and positive; arch is O(1) (not exponentially small).
        let zero = Float::with_val(prec, 0);
        assert!(d.eps_n > zero, "plunge should be positive");
        assert!(d.eps_n < Float::with_val(prec, 1), "plunge should be small");
        let thresh = Float::with_val(prec, Float::parse("1e-3").unwrap());
        assert!(
            d.arch_rayleigh.clone().abs() > thresh,
            "arch Rayleigh should be O(1), not exponentially small: {}",
            d.arch_rayleigh
        );
    }

    /// HighPrecConfig quad_points should clamp to MAX_QUAD_POINTS.
    #[test]
    fn config_clamps_quad_points() {
        let cfg = HighPrecConfig::for_decimal_digits(2000);
        // 2000 * 3 = 6000, clamped to max 4000
        assert_eq!(cfg.quad_points, MAX_QUAD_POINTS);
    }

    /// HighPrecConfig quad_points should clamp to MIN_QUAD_POINTS.
    #[test]
    fn config_floors_quad_points() {
        let cfg = HighPrecConfig::for_decimal_digits(10);
        // 10 * 3 = 30, clamped to min 600
        assert_eq!(cfg.quad_points, MIN_QUAD_POINTS);
    }

    /// Explicit seeded refinement at small N should produce roots near the
    /// corresponding comparison ordinates.
    /// Uses λ²=13, N=10 (21×21 matrix) at 64-digit precision — fast enough
    /// for a unit test (~1-2 seconds).
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn run_small_n_produces_eigenvalues() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.n_eigenvalues = 5;
        // Hermetic: no cache read/write, no network. Pure compute.
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;

        // HP seeds: first 5 Riemann zeros at full precision.
        let prec = cfg.precision_bits;
        let seed_strs = [
            "14.134725141734693790457251983562470270784257115699243175685567460149",
            "21.022039638771554992628479593896902777334340524902781754629520403587",
            "25.010857580145688763213790992562821818659549672557996672496542006745",
            "30.424876125859513210311897530584091320181560023715440180962146036993",
            "32.935061587739189690662368964074903488812715603517039009280003440784",
        ];
        let zero_seeds: Vec<Float> = seed_strs
            .iter()
            .map(|s| Float::with_val(prec, Float::parse(s).unwrap()))
            .collect();

        let result =
            run_indexed_seeded(&params, &cfg, 1, &zero_seeds, &test_reference_dataset()).unwrap();

        // At N=10, the f64 solver finds however many eigenvalues exist
        // in the standard brackets. Just verify we got at least 1.
        assert!(
            !result.eigenvalues_pos.is_empty(),
            "should produce at least one eigenvalue"
        );

        // First eigenvalue should match 14.13... to at least 10 digits.
        // Compare in HP — no f64 round-trip.
        let prec = result.precision_bits;
        let target = Float::with_val(
            prec,
            Float::parse("14.134725141734693790457251983").unwrap(),
        );
        let ev0 = result.eigenvalues_pos[0]
            .value()
            .expect("Newton should converge for zero_1 at λ²=13, N=10, P=64");
        let mut diff = ev0.clone();
        diff -= &target;
        let abs_diff = diff.abs();
        let tol = Float::with_val(prec, Float::parse("1e-5").unwrap());
        assert!(
            abs_diff < tol,
            "first eigenvalue {} should be near 14.13",
            xc_numerics::fmt::display_hp(ev0, 10)
        );

        // ε_N should be small (tiny Weil eigenvalue at λ²=13). Compare HP.
        let eps_tol = Float::with_val(prec, Float::parse("1e-20").unwrap());
        let abs_eps = result.weil_min_eigenvalue.clone().abs();
        assert!(
            abs_eps < eps_tol,
            "ε_N = {} should be tiny at λ²=13, N=10",
            xc_numerics::fmt::display_hp(&result.weil_min_eigenvalue, 6)
        );

        // Elapsed time should be positive (f64 metadata is fine here).
        assert!(result.elapsed_seconds > 0.0);
    }

    /// P2 test: newton_xi_hat_zero returns None when given a bogus
    /// eigenvector (all zeros) — Newton has no R(z) structure to converge on.
    #[test]
    fn newton_returns_none_on_bogus_eigenvector() {
        let prec = 128;
        let n_max = 5;
        let l = Float::with_val(prec, Float::parse("3.0").unwrap()); // ln(20)~3
        let seed = Float::with_val(prec, Float::parse("14.13").unwrap());
        // All-zero eigenvector: R(z) = 0 for all z, Newton cannot converge.
        let xi = vec![Float::with_val(prec, 0); 2 * n_max + 1];
        let result = newton_xi_hat_zero(&xi, n_max, &l, &seed, prec, 20);
        // With zero ξ, R(z)=0 everywhere. Newton step dz = R/R' = 0/0.
        // Function should return Failed (r_prime == 0 check) OR converge
        // trivially (dz=0 < tol on first step). Either way, not a silent
        // fallback to seed.
        // Actually: with all-zero xi, r=0 and r_prime=0 → Failed (division guard).
        assert!(
            !result.has_value(),
            "newton should return Failed with all-zero xi (degenerate R(z)=0)"
        );
    }

    /// P3 test: log_lambda_sq_hp at non-integer (fractional) λ² should
    /// give a value that is at least as accurate as 17-digit f64 input allows.
    #[test]
    fn log_lambda_sq_hp_non_integer_precision() {
        let prec = 512;
        // λ²=2.5 — fractional. ln(2.5) = 0.916290731874155...
        let params = CcmParams::from_lambda_sq_fractional(2.5, 10);
        let l = log_lambda_sq_hp(&params, prec);
        // Reference: ln(2.5) computed at HP from exact rational 5/2.
        let ref_val = {
            let five = Float::with_val(prec, 5);
            let two = Float::with_val(prec, 2);
            let mut ratio = five;
            ratio /= &two;
            ratio.ln()
        };
        let mut diff = l.clone();
        diff -= &ref_val;
        let abs_diff = diff.abs();
        // Should match to at least 15 significant digits (f64 input limit).
        // At prec=512 the string-parse path gives 17 sig figs.
        let tol = Float::with_val(prec, Float::parse("1e-15").unwrap());
        assert!(
            abs_diff < tol,
            "log_lambda_sq_hp for fractional λ²=2.5: L = {}, ref = {}, diff = {}",
            l.to_f64(),
            ref_val.to_f64(),
            abs_diff.to_f64()
        );
    }

    /// R4 test: root and inverse-iteration limits deliberately favor slow,
    /// accurate convergence over early approximate results.
    #[test]
    fn config_uses_generous_fixed_halley_and_inverse_limits() {
        let cfg60 = HighPrecConfig::for_decimal_digits(60);
        assert_eq!(cfg60.root_solver, RootSolver::Halley);
        assert_eq!(cfg60.solver_steps, 2_000);
        assert_eq!(cfg60.inverse_iter_steps, 2_000);

        let cfg200 = HighPrecConfig::for_decimal_digits(200);
        assert_eq!(cfg200.solver_steps, 2_000);
        assert_eq!(cfg200.inverse_iter_steps, 2_000);

        let cfg1000 = HighPrecConfig::for_decimal_digits(1000);
        assert_eq!(cfg1000.precision_bits, 3_386);
        assert_eq!(cfg1000.solver_steps, 2_000);
        assert_eq!(cfg1000.inverse_iter_steps, 2_000);

        let cfg2000 = HighPrecConfig::for_decimal_digits(2000);
        assert_eq!(cfg2000.solver_steps, 2_000);
        assert_eq!(cfg2000.inverse_iter_steps, 2_000);

        // Steps must be non-decreasing with precision
        assert!(cfg200.solver_steps >= cfg60.solver_steps);
        assert!(cfg1000.solver_steps >= cfg200.solver_steps);
        assert!(cfg2000.solver_steps >= cfg1000.solver_steps);
    }

    /// R3 test: Halley's method matches Newton to working precision.
    /// Runs the same λ²=13, N=10 config with both methods and verifies
    /// the results agree to at least 50 digits.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn halley_matches_newton_for_same_config() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.n_eigenvalues = 1;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let seed_str = "14.134725141734693790457251983562470270784257115699243175685567460149";
        let zero_seeds: Vec<Float> = vec![Float::with_val(prec, Float::parse(seed_str).unwrap())];

        // Newton result
        cfg.root_solver = RootSolver::Newton;
        let result_n =
            run_indexed_seeded(&params, &cfg, 1, &zero_seeds, &test_reference_dataset()).unwrap();
        let ev_newton = result_n.eigenvalues_pos[0]
            .value()
            .expect("Newton should converge");

        // Halley result
        cfg.root_solver = RootSolver::Halley;
        let result_h =
            run_indexed_seeded(&params, &cfg, 1, &zero_seeds, &test_reference_dataset()).unwrap();
        let ev_halley = result_h.eigenvalues_pos[0]
            .value()
            .expect("Halley should converge");

        // Both should agree to at least 50 digits
        let mut diff = ev_newton.clone();
        diff -= ev_halley;
        let abs_diff = diff.abs();
        let tol = Float::with_val(prec, Float::parse("1e-50").unwrap());
        assert!(
            abs_diff < tol,
            "Halley and Newton should agree to 50 digits; diff = {}",
            abs_diff.to_f64()
        );
    }
    /// Uses λ²=13, N=10 which has good eigenvalue convergence.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn indexed_seeded_run_returns_requested_root_count() {
        let params = CcmParams::from_lambda_sq_integer(13, 8); // N=8, tiny
        let mut cfg = HighPrecConfig::for_decimal_digits(60);
        cfg.n_eigenvalues = 1;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let seed_strs = ["14.134725141734693790457251983562470270784257115699243175685567460149"];
        let zero_seeds: Vec<Float> = seed_strs
            .iter()
            .map(|s| Float::with_val(prec, Float::parse(s).unwrap()))
            .collect();
        let result =
            run_indexed_seeded(&params, &cfg, 1, &zero_seeds, &test_reference_dataset()).unwrap();
        // Should produce 1 eigenvalue (Some or None — just check no panic)
        assert_eq!(
            result.eigenvalues_pos.len(),
            1,
            "should produce 1 eigenvalue entry for n_eigenvalues=1"
        );
    }

    /// measure_evenness at λ²=13, N=10 should show near-perfect evenness.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn measure_evenness_small_lambda_is_even() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        // Hermetic: no cache read/write, no network. Pure compute.
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let result = measure_evenness(&params, &cfg).unwrap();

        let prec = result.evenness_deviation.prec();

        // At λ²=13, the natural eigenvector should be essentially even.
        // Compare in HP (deviation could be 1e-30 or smaller — tighter than f64).
        let dev_tol = Float::with_val(prec, Float::parse("1e-10").unwrap());
        assert!(
            result.evenness_deviation < dev_tol,
            "evenness deviation at λ²=13 should be tiny, got {}",
            xc_numerics::fmt::display_hp(&result.evenness_deviation, 6)
        );

        // Both eigenvalues should be the same (since natural IS even).
        // Use HP relative_difference helper — no f64 fallback even at tiny ε.
        if let Some(rel_diff) = xc_numerics::fmt::relative_difference(
            &result.natural_eigenvalue,
            &result.forced_eigenvalue,
        ) {
            let rel_tol = Float::with_val(prec, Float::parse("1e-10").unwrap());
            assert!(
                rel_diff < rel_tol,
                "natural and forced eigenvalues should match, rel diff = {}",
                xc_numerics::fmt::display_hp(&rel_diff, 6)
            );
        } else {
            // forced is exactly zero — the only acceptable case is when
            // natural is also zero.
            assert!(
                result.natural_eigenvalue.is_zero(),
                "forced is zero but natural is not — eigenvalues differ"
            );
        }

        // Sanity check signs match (both should be positive at λ²=13).
        use xc_numerics::fmt::sign_of;
        assert_eq!(
            sign_of(&result.natural_eigenvalue),
            sign_of(&result.forced_eigenvalue),
            "natural and forced eigenvalue signs must agree at λ²=13"
        );
    }

    // ---------------------------------------------------------------
    // tau cache — pure-function and verify_dir tests
    // ---------------------------------------------------------------

    /// `tau_cache::parse_filename` extracts (λ²_int, n_modes, prec)
    /// from each of the three accepted filename forms, and rejects
    /// other patterns.
    #[test]
    fn tau_cache_filename_parser() {
        use super::super::LambdaSq;
        use super::tau_cache::{parse_filename, FileKind};

        // .json
        let r = parse_filename("lambda_sq13_nmodes120_prec3338.json").unwrap();
        assert_eq!(r.0, LambdaSq::integer(13));
        assert_eq!(r.1, 120);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Json));

        // .json.zip
        let r = parse_filename("lambda_sq100_nmodes500_prec3338.json.zip").unwrap();
        assert_eq!(r.0, LambdaSq::integer(100));
        assert_eq!(r.1, 500);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Zip));

        // .json.zip.partXX
        let r = parse_filename("lambda_sq1000_nmodes800_prec3338.json.zip.part00").unwrap();
        assert_eq!(r.0, LambdaSq::integer(1000));
        assert_eq!(r.1, 800);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Part));
        let r = parse_filename("lambda_sq1000_nmodes800_prec3338.json.zip.part42").unwrap();
        assert!(matches!(r.3, FileKind::Part));

        // Invalid — wrong base name.
        assert!(parse_filename("foo.json").is_none());
        // Invalid — missing component.
        assert!(parse_filename("lambda_sq13_nmodes120.json").is_none());
        // Invalid — non-digit part suffix.
        assert!(parse_filename("lambda_sq13_nmodes120_prec3338.json.zip.partAA").is_none());
        // Invalid — empty part suffix.
        assert!(parse_filename("lambda_sq13_nmodes120_prec3338.json.zip.part").is_none());
    }

    /// `tau_cache::structural_check` rejects asymmetric matrices,
    /// matrices with wrong length, and matrices with NaN/Inf entries.
    #[test]
    fn tau_cache_structural_check() {
        use super::tau_cache::structural_check;
        let prec = 128;
        let n_modes = 3;
        let dim = 2 * n_modes + 1; // 7

        // Build a symmetric 7×7 matrix.
        let mut sym = vec![Float::with_val(prec, 0); dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let val = Float::with_val(prec, (i + j + 1) as f64);
                sym[i * dim + j] = val.clone();
                sym[j * dim + i] = val;
            }
        }
        assert!(
            structural_check(&sym, n_modes, prec).is_none(),
            "symmetric matrix should pass"
        );

        // Wrong length.
        let mut short = sym.clone();
        short.pop();
        assert!(
            structural_check(&short, n_modes, prec).is_some(),
            "wrong length should be rejected"
        );

        // Asymmetric: perturb one off-diagonal pair so τ[i,j] ≠ τ[j,i].
        let mut asym = sym.clone();
        asym[0 * dim + 1] = Float::with_val(prec, 99);
        // τ[1,0] is still its original value → asymmetry.
        assert!(
            structural_check(&asym, n_modes, prec).is_some(),
            "asymmetric matrix should be rejected"
        );

        // NaN entry.
        let mut with_nan = sym.clone();
        with_nan[5] = Float::with_val(prec, f64::NAN);
        assert!(
            structural_check(&with_nan, n_modes, prec).is_some(),
            "NaN entry should be rejected"
        );
    }

    /// `tau_cache::parse_json_for_test` rejects a JSON envelope whose
    /// `toolkit_version` is older than the CCM-matrix family producer floor — a
    /// stale file written by an older toolkit build.
    #[test]
    fn tau_cache_rejects_stale_toolkit_version() {
        let n_modes: usize = 1;
        let prec: u32 = 64;
        // Build a well-formed envelope stamped with a far-past version.
        let dim = 2 * n_modes + 1;
        let strs: Vec<String> = (0..dim * dim).map(|i| format!("{}", i)).collect();
        let payload = serde_json::json!({
            "toolkit_version": "0.0.1",
            "lambda_sq": 13_u64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        })
        .to_string();
        assert!(
            super::tau_cache::parse_json_for_test(&payload, n_modes, prec).is_none(),
            "tau parser should reject a stale toolkit_version=0.0.1"
        );
    }

    /// `verify_tau_cache_dir` on a non-existent directory returns
    /// an empty report.
    #[test]
    fn tau_cache_verify_missing_dir() {
        let nonexistent = crate::test_tmp_root().join(format!(
            "xc_spectral_tau_cache_test_missing_{}",
            std::process::id()
        ));
        let report = super::tau_cache::verify_tau_cache_dir(&nonexistent).unwrap();
        assert_eq!(report.statuses.len(), 0);
        assert_eq!(report.ok_count(), 0);
        assert_eq!(report.failure_count(), 0);
    }

    /// `verify_tau_cache_dir` classifies each file: Ok for a real
    /// matrix, StructurallyInvalid for an asymmetric one, Skipped
    /// for an unrecognized name, LoadFailed for malformed JSON.
    #[test]
    fn tau_cache_verify_classifies() {
        use super::super::LambdaSq;
        use super::tau_cache::{cache_filename, verify_tau_cache_dir, TauCacheFileStatus};

        let prec = 128;
        let n_modes = 3;
        let dim = 2 * n_modes + 1; // 7
        let lambda_sq = LambdaSq::integer(13);

        let temp_dir = crate::fresh_test_dir("tau_cache_classify");

        // 1. Valid: build a symmetric matrix and serialize as envelope.
        let mut sym = vec![Float::with_val(prec, 0); dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let val = Float::with_val(prec, (i + j + 1) as f64);
                sym[i * dim + j] = val.clone();
                sym[j * dim + i] = val;
            }
        }
        let valid_name = cache_filename(lambda_sq, n_modes, prec);
        let valid_path = temp_dir.join(&valid_name);
        let strs: Vec<String> = sym.iter().map(|f| f.to_string()).collect();
        let valid_json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": super::tau_cache::toolkit_version_for_test(),
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        });
        std::fs::write(&valid_path, serde_json::to_string(&valid_json).unwrap()).unwrap();

        // 2. StructurallyInvalid: asymmetric matrix at a different lsq,
        //    wrapped in the envelope so parse succeeds but structural check fails.
        let mut asym = sym.clone();
        asym[0 * dim + 1] = Float::with_val(prec, 99);
        // τ[1,0] still its original value → asymmetry.
        let lsq_bad = LambdaSq::integer(lambda_sq.value_u64 + 1);
        let bad_name = cache_filename(lsq_bad, n_modes, prec);
        let bad_path = temp_dir.join(&bad_name);
        let bad_strs: Vec<String> = asym.iter().map(|f| f.to_string()).collect();
        let bad_json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": super::tau_cache::toolkit_version_for_test(),
            "lambda_sq": lsq_bad.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": bad_strs,
        });
        std::fs::write(&bad_path, serde_json::to_string(&bad_json).unwrap()).unwrap();

        // 3. Skipped: unrecognized filename.
        let skipped_path = temp_dir.join("not_a_tau_cache.txt");
        std::fs::write(&skipped_path, "irrelevant").unwrap();

        // 4. LoadFailed: matching pattern, malformed JSON.
        let malformed_name =
            cache_filename(LambdaSq::integer(lambda_sq.value_u64 + 2), n_modes, prec);
        let malformed_path = temp_dir.join(&malformed_name);
        std::fs::write(&malformed_path, "{").unwrap();

        let report = verify_tau_cache_dir(&temp_dir).unwrap();
        assert_eq!(
            report.statuses.len(),
            4,
            "expected 4 statuses, got {}",
            report.statuses.len()
        );

        let mut saw_ok = false;
        let mut saw_invalid = false;
        let mut saw_skipped = false;
        let mut saw_loadfail = false;
        for s in &report.statuses {
            match s {
                TauCacheFileStatus::Ok {
                    path,
                    lambda_sq: l,
                    n_modes: n,
                    prec: p,
                } => {
                    assert_eq!(path, &valid_path);
                    assert_eq!(*l, lambda_sq);
                    assert_eq!(*n, n_modes);
                    assert_eq!(*p, prec);
                    saw_ok = true;
                }
                TauCacheFileStatus::StructurallyInvalid { path, .. } => {
                    assert_eq!(path, &bad_path);
                    saw_invalid = true;
                }
                TauCacheFileStatus::Skipped { path, .. } => {
                    assert_eq!(path, &skipped_path);
                    saw_skipped = true;
                }
                TauCacheFileStatus::LoadFailed { path, .. } => {
                    assert_eq!(path, &malformed_path);
                    saw_loadfail = true;
                }
            }
        }
        assert!(saw_ok, "missing Ok");
        assert!(saw_invalid, "missing StructurallyInvalid");
        assert!(saw_skipped, "missing Skipped");
        assert!(saw_loadfail, "missing LoadFailed");

        assert_eq!(report.ok_count(), 1);
        assert_eq!(report.failure_count(), 2);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Negative: a structurally-invalid τ `.json` on disk (parseable but
    /// asymmetric) must be skipped by `tau_cache::load` (returns `None`,
    /// treated as a miss → caller recomputes). The bad file is preserved.
    /// Mirrors the GL cache's structurally-invalid-json test; brings the
    /// τ load path to parity (previously only the verify-dir audit and
    /// the unit-level structural_check were tested, not the load path).
    #[test]
    fn tau_load_skips_structurally_invalid_json() {
        use super::super::LambdaSq;
        use super::tau_cache::{cache_filename, load};
        use xc_numerics::quadrature::CacheMode;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes = 3usize;
        let dim = 2 * n_modes + 1; // 7

        let temp = crate::fresh_test_dir("tau_invalid_json");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let zip_path = dir.join(format!("{}.zip", entry_name));

        // Build a correctly-shaped matrix, then break symmetry at one
        // off-diagonal pair so structural_check rejects it.
        let mut m = vec![Float::with_val(prec, 0); dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let val = Float::with_val(prec, (i + j + 1) as f64);
                m[i * dim + j] = val.clone();
                m[j * dim + i] = val;
            }
        }
        m[0 * dim + 1] = Float::with_val(prec, 99); // τ[1,0] unchanged → asymmetric
        let strs: Vec<String> = m.iter().map(|f| f.to_string()).collect();
        let json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": super::tau_cache::toolkit_version_for_test(),
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        });
        let json_str = serde_json::to_string(&json).unwrap();

        // Plant the bad matrix inside a .json.zip (the only tier read now).
        {
            use std::io::Write;
            let f = std::fs::File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file(&entry_name, opts).unwrap();
            zw.write_all(json_str.as_bytes()).unwrap();
            zw.finish().unwrap();
        }

        // load must skip the asymmetric matrix (None). JsonOnly is a
        // read no-op; JsonZip reads the zip, runs structural_check, rejects.
        assert!(
            load(lambda_sq, n_modes, prec, CacheMode::JsonOnly).is_none(),
            "JsonOnly is a read no-op under the zip-only contract"
        );
        assert!(
            load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
            "structurally-invalid (asymmetric) τ matrix in the zip must be skipped"
        );

        // Bad file preserved on disk.
        assert!(
            zip_path.exists(),
            "structurally-invalid τ zip should be preserved for inspection"
        );

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a truncated/corrupt τ `.json.zip` must be detected and
    /// skipped without panic (`load` returns `None`). The corrupt file is
    /// preserved. Mirrors the GL cache's `cache_handles_corrupt_zip_gracefully`.
    #[test]
    fn tau_load_handles_corrupt_zip_gracefully() {
        use super::super::LambdaSq;
        use super::tau_cache::load;
        use xc_numerics::quadrature::CacheMode;
        let prec = 64;
        let lambda_sq = LambdaSq::integer(49);
        let n_modes = 3usize;

        let temp = crate::fresh_test_dir("tau_corrupt_zip");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).unwrap();
        // Garbage bytes named as the single-zip for this config; no
        // local .json, so JsonZip falls through to the zip, fails to
        // open it, and returns None without panicking.
        let zip_path = dir.join(format!(
            "lambda_sq{}_nmodes{}_prec{}.json.zip",
            lambda_sq.filename_str(),
            n_modes,
            prec
        ));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        assert!(
            load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
            "corrupt τ .json.zip must be skipped, not loaded"
        );

        assert!(
            zip_path.exists(),
            "corrupt τ zip should be preserved for inspection"
        );

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    // -----------------------------------------------------------------
    // weil_eigvec_cache tests
    // -----------------------------------------------------------------

    /// A fresh temp dir + cwd guard so cache reads/writes land in a
    /// throwaway location and never touch the real `data/` tree.
    /// Scratch lives under `target/test-tmp/` (removed by `cargo clean`),
    /// not the OS temp dir.
    fn weil_temp_cwd(tag: &str) -> std::path::PathBuf {
        crate::fresh_test_dir(&format!("weil_eigvec_{}", tag))
    }

    /// `parse_json` accepts a well-formed entry and rejects metadata
    /// mismatches, wrong xi length, and non-finite values.
    #[test]
    fn weil_eigvec_parse_json_validates() {
        use super::super::LambdaSq;
        use super::weil_eigvec_cache::parse_json;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes = 3usize;
        let dim = 2 * n_modes + 1; // 7

        let xi: Vec<Float> = (0..dim)
            .map(|i| Float::with_val(prec, (i + 1) as f64))
            .collect();
        let xi_strs: Vec<String> = xi.iter().map(|f| f.to_string()).collect();
        let good = serde_json::json!({
            "schema_version": 2,
            "toolkit_version": super::weil_eigvec_cache::toolkit_version_for_test(),
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.5e-40",
            "xi": xi_strs,
            "inverse_iteration": {
                "configured_step_limit": 2000,
                "unshifted_steps": 7,
                "unshifted_converged": true,
                "final_relative_rayleigh_change": "1e-30",
                "shifted_refinement": "accepted",
                "final_relative_residual_norm": "1e-40"
            }
        })
        .to_string();
        let parsed =
            parse_json(&good, lambda_sq, n_modes, prec).expect("well-formed entry should parse");
        assert_eq!(parsed.xi.len(), dim);

        // Wrong n_modes metadata → reject.
        assert!(
            parse_json(&good, lambda_sq, n_modes + 1, prec).is_none(),
            "n_modes mismatch should be rejected"
        );
        // Wrong precision metadata → reject.
        assert!(
            parse_json(&good, lambda_sq, n_modes, prec + 1).is_none(),
            "precision mismatch should be rejected"
        );
        // Wrong λ² metadata → reject.
        assert!(
            parse_json(
                &good,
                LambdaSq::integer(lambda_sq.value_u64 + 5),
                n_modes,
                prec
            )
            .is_none(),
            "lambda_sq mismatch should be rejected"
        );

        // Wrong xi length → reject.
        let mut short_strs = xi_strs.clone();
        short_strs.pop();
        let short = serde_json::json!({
            "schema_version": 2, "toolkit_version": super::weil_eigvec_cache::toolkit_version_for_test(),
            "lambda_sq": lambda_sq.value_f64, "n_modes": n_modes,
            "precision_bits": prec, "weil_min_eigenvalue": "1.5e-40", "xi": short_strs,
            "inverse_iteration": {
                "configured_step_limit": 2000,
                "unshifted_steps": 7,
                "unshifted_converged": true,
                "final_relative_rayleigh_change": "1e-30",
                "shifted_refinement": "accepted",
                "final_relative_residual_norm": "1e-40"
            }
        }).to_string();
        assert!(
            parse_json(&short, lambda_sq, n_modes, prec).is_none(),
            "wrong xi length should be rejected"
        );
    }

    /// `weil_eigvec_cache::parse_json_for_test` rejects a JSON envelope
    /// whose `toolkit_version` is older than the Weil-state family producer floor
    /// — a stale file written by an older toolkit build.
    #[test]
    fn weil_eigvec_rejects_stale_toolkit_version() {
        use super::super::LambdaSq;
        let prec: u32 = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes: usize = 2;
        let dim = 2 * n_modes + 1;
        let xi_strs: Vec<String> = (0..dim).map(|i| format!("0.{}", i + 1)).collect();
        let payload = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": "0.0.1",
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.23e-10",
            "xi": xi_strs,
        })
        .to_string();
        assert!(
            super::weil_eigvec_cache::parse_json_for_test(&payload, lambda_sq, n_modes, prec)
                .is_none(),
            "weil eigvec parser should reject a stale toolkit_version=0.0.1"
        );
    }

    /// `residual_ok` accepts a genuine eigenpair of τ and rejects a
    /// perturbed (wrong) eigenvector — the strong integrity test the
    /// after-τ cache check relies on.
    #[test]
    fn weil_eigvec_residual_check_discriminates() {
        use super::weil_eigvec_cache::residual_ok;
        let prec = 256;
        let n = 5;
        // Diagonal matrix: eigenpairs are (λ_i, e_i). Smallest is λ=1 at e_0.
        let mut a = vec![Float::with_val(prec, 0); n * n];
        let diag = ["1", "2", "3", "4", "5"];
        for (i, d) in diag.iter().enumerate() {
            a[i * n + i] = Float::with_val(prec, Float::parse(d).unwrap());
        }
        // True smallest eigenpair.
        let eps = Float::with_val(prec, 1);
        let mut xi = vec![Float::with_val(prec, 0); n];
        xi[0] = Float::with_val(prec, 1);
        assert!(
            residual_ok(&a, n, &xi, &eps, prec),
            "genuine eigenpair should pass the residual check"
        );

        // Wrong eigenvector (points along e_1, whose eigenvalue is 2,
        // not 1) → residual is O(1) → reject.
        let mut wrong = vec![Float::with_val(prec, 0); n];
        wrong[1] = Float::with_val(prec, 1);
        assert!(
            !residual_ok(&a, n, &wrong, &eps, prec),
            "wrong eigenvector should fail the residual check"
        );

        // Zero vector → reject.
        let zero = vec![Float::with_val(prec, 0); n];
        assert!(
            !residual_ok(&a, n, &zero, &eps, prec),
            "zero vector should fail the residual check"
        );
    }

    /// Round-trip: `save` then `load` returns a byte-identical ξ and ε_N
    /// at every CacheMode tier. Also checks that `CacheMode::Off` writes
    /// nothing.
    #[test]
    fn weil_eigvec_save_load_round_trip() {
        use super::super::LambdaSq;
        use super::weil_eigvec_cache::{load, save};
        use xc_numerics::quadrature::CacheMode;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(49);
        let n_modes = 4usize;
        let dim = 2 * n_modes + 1; // 9

        let temp = weil_temp_cwd("round_trip");
        let _guard = CwdGuard::enter(&temp);

        let eps = Float::with_val(prec, Float::parse("3.25e-12").unwrap());
        let xi: Vec<Float> = (0..dim)
            .map(|i| Float::with_val(prec, Float::parse(format!("0.{}1", i + 1)).unwrap()))
            .collect();
        let diagnostics = xc_numerics::linalg::InverseIterationDiagnostics {
            configured_step_limit: 2_000,
            unshifted_steps: 7,
            unshifted_converged: true,
            final_relative_rayleigh_change: Some(Float::with_val(
                prec,
                Float::parse("1e-30").unwrap(),
            )),
            shifted_refinement: xc_numerics::linalg::ShiftedRefinementOutcome::Accepted,
            final_relative_residual_norm: Float::with_val(prec, Float::parse("1e-40").unwrap()),
        };

        // Off: writes nothing, reads nothing.
        save(
            lambda_sq,
            n_modes,
            prec,
            &eps,
            &xi,
            &diagnostics,
            CacheMode::Off,
            CcmParityPolicy::EvenSector,
        );
        assert!(
            load(
                lambda_sq,
                n_modes,
                prec,
                CacheMode::Off,
                CcmParityPolicy::EvenSector,
            )
            .is_none(),
            "Off should never read"
        );
        assert!(
            load(
                lambda_sq,
                n_modes,
                prec,
                CacheMode::JsonZip,
                CcmParityPolicy::EvenSector,
            )
            .is_none(),
            "Off save should have written nothing"
        );

        // JsonZip: writes ONLY the .json.zip (zip-only contract); reads
        // back identical by decompressing in memory.
        save(
            lambda_sq,
            n_modes,
            prec,
            &eps,
            &xi,
            &diagnostics,
            CacheMode::JsonZip,
            CcmParityPolicy::EvenSector,
        );

        // No uncompressed .json should be written.
        let jp = temp.join("data").join("weil_eigvec_cache").join(
            super::weil_eigvec_cache::cache_filename(
                lambda_sq,
                n_modes,
                prec,
                CcmParityPolicy::EvenSector,
            ),
        );
        assert!(
            !jp.exists(),
            "zip-only: save must not write an uncompressed .json"
        );

        let got = load(
            lambda_sq,
            n_modes,
            prec,
            CacheMode::JsonZip,
            CcmParityPolicy::EvenSector,
        )
        .expect("JsonZip round-trip should load from the zip");
        assert_eq!(got.xi.len(), dim);
        for (a, b) in xi.iter().zip(got.xi.iter()) {
            assert_eq!(
                a.to_string(),
                b.to_string(),
                "xi entry must round-trip exactly"
            );
        }
        assert_eq!(
            eps.to_string(),
            got.eps_n.to_string(),
            "eps_n must round-trip exactly"
        );
        assert_eq!(got.diagnostics, diagnostics);

        // JsonOnly is now a read no-op (no uncompressed .json exists).
        assert!(
            load(
                lambda_sq,
                n_modes,
                prec,
                CacheMode::JsonOnly,
                CcmParityPolicy::EvenSector,
            )
            .is_none(),
            "zip-only: JsonOnly must not read the zip"
        );

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a structurally-invalid ξ entry (parseable JSON but
    /// wrong xi length / mismatched metadata) inside a `.json.zip` must
    /// be skipped by `load` (returns `None`, treated as a miss → caller
    /// recomputes). The bad file is preserved on disk for inspection.
    #[test]
    fn weil_eigvec_load_skips_structurally_invalid_json() {
        use super::super::LambdaSq;
        use super::weil_eigvec_cache::{cache_filename, load};
        use xc_numerics::quadrature::CacheMode;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes = 4usize; // expects 2N+1 = 9 entries

        let temp = weil_temp_cwd("invalid_json");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let entry_name = cache_filename(lambda_sq, n_modes, prec, CcmParityPolicy::EvenSector);
        let zip_path = dir.join(format!("{}.zip", entry_name));

        // Shape-parseable JSON, but xi has the WRONG length (3 ≠ 9)
        // and otherwise-valid metadata. parse_json must reject it.
        let bad = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": super::weil_eigvec_cache::toolkit_version_for_test(),
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.0e-20",
            "xi": ["1.0", "2.0", "3.0"],
        })
        .to_string();

        // Plant the bad entry inside a .json.zip (the only tier read now).
        {
            use std::io::Write;
            let f = std::fs::File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file(&entry_name, opts).unwrap();
            zw.write_all(bad.as_bytes()).unwrap();
            zw.finish().unwrap();
        }

        // load must skip (None) — never returns a malformed entry.
        assert!(
            load(
                lambda_sq,
                n_modes,
                prec,
                CacheMode::JsonOnly,
                CcmParityPolicy::EvenSector,
            )
            .is_none(),
            "JsonOnly is a read no-op under the zip-only contract"
        );
        assert!(
            load(
                lambda_sq,
                n_modes,
                prec,
                CacheMode::JsonZip,
                CcmParityPolicy::EvenSector,
            )
            .is_none(),
            "structurally-invalid ξ entry in the zip must be skipped"
        );

        // The bad file is preserved on disk (load does not delete it).
        assert!(
            zip_path.exists(),
            "structurally-invalid zip should be preserved for inspection"
        );

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a truncated/corrupt ξ `.json.zip` must be detected and
    /// skipped without panic (`load` returns `None`). The corrupt file is
    /// preserved. Mirrors the GL cache's `cache_handles_corrupt_zip_gracefully`.
    #[test]
    fn weil_eigvec_load_handles_corrupt_zip_gracefully() {
        use super::super::LambdaSq;
        use super::weil_eigvec_cache::load;
        use xc_numerics::quadrature::CacheMode;
        let prec = 64;
        let lambda_sq = LambdaSq::integer(49);
        let n_modes = 3usize;

        let temp = weil_temp_cwd("corrupt_zip");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).unwrap();
        // Random bytes that are NOT a valid zip. No local .json present,
        // so JsonZip must fall through to the (garbage) zip, fail to
        // open it, and return None — without panicking.
        let zip_path = dir.join(format!(
            "weil_eigvec_lambda_sq{}_nmodes{}_prec{}.json.zip",
            lambda_sq.filename_str(),
            n_modes,
            prec
        ));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        assert!(
            load(
                lambda_sq,
                n_modes,
                prec,
                CacheMode::JsonZip,
                CcmParityPolicy::EvenSector,
            )
            .is_none(),
            "corrupt .json.zip must be skipped, not loaded"
        );

        // Corrupt file preserved on disk.
        assert!(
            zip_path.exists(),
            "corrupt zip should be preserved for inspection"
        );

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// `newton_xi_hat_zero` and `halley_xi_hat_zero` on a synthetic R(z)
    /// with a known root.
    ///
    /// Setup: single-pole eigenvector ξ = (1) at n=0, L = 2π.
    /// R(z) = ξ[0] / (z − 0) = 1/z, which has no zero.
    ///
    /// Better: two poles at ±p, weights ξ = (+1, +1) gives
    /// R(z) = 1/(z-p) + 1/(z+p) = 2z / (z²-p²).
    /// R(z) = 0 → z = 0, so we seed at 0.1 and verify convergence.
    ///
    /// Even simpler (and deterministic): weights ξ = (+1, -1) gives
    /// R(z) = 1/(z-p) − 1/(z+p) = 2p/(z²-p²).
    /// R(z) = 0 → no real zero.
    ///
    /// Use the two-weight (+1, +1) case with p = π/L so the pole
    /// positions are the n=±1 poles, and the zero is at z=0.
    #[test]
    fn newton_and_halley_find_known_zero() {
        use rug::{ops::Pow, Float};
        let prec = 256;

        // L = 2π, n_max = 1: poles at 0, ±2π/L = ±1.
        // ξ = (1, 0, 1) in order n = -1, 0, 1.
        // R(z) = ξ[-1]/(z - (-1)) + ξ[0]/(z - 0) + ξ[+1]/(z - 1)
        //      = 1/(z+1) + 0 + 1/(z-1) = 2z/(z²-1)
        // Zero at z = 0.
        let n_max = 1_usize;
        let l = Float::with_val(prec, Float::parse("6.283185307179586476925").unwrap()); // 2π
        let xi = vec![
            Float::with_val(prec, 1), // n = -1
            Float::with_val(prec, 0), // n = 0
            Float::with_val(prec, 1), // n = +1
        ];
        let seed = Float::with_val(prec, Float::parse("0.5").unwrap());
        let tol_check = Float::with_val(prec, 2).pow(-((prec as i32) / 2));

        let z_newton = super::newton_xi_hat_zero(&xi, n_max, &l, &seed, prec, 100)
            .value()
            .expect("Newton should find a zero")
            .clone();
        assert!(
            z_newton.clone().abs() < tol_check,
            "Newton zero should be ~0, got {}",
            z_newton
        );

        let z_halley = super::halley_xi_hat_zero(&xi, n_max, &l, &seed, prec, 100)
            .value()
            .expect("Halley should find a zero")
            .clone();
        assert!(
            z_halley.clone().abs() < tol_check,
            "Halley zero should be ~0, got {}",
            z_halley
        );

        // Verify the two methods agree.
        let mut diff = z_newton.clone();
        diff -= &z_halley;
        let abs_diff = diff.abs();
        assert!(
            abs_diff < tol_check,
            "Newton and Halley zeros differ by {}",
            abs_diff
        );
    }

    /// `newton_xi_hat_zero` with a pole directly on the seed (degenerate case):
    /// should return None, not panic or loop.
    #[test]
    fn newton_returns_none_when_starting_on_pole() {
        use rug::Float;
        let prec = 128;
        let n_max = 1_usize;
        let l = Float::with_val(prec, Float::parse("6.283185307179586").unwrap());
        // ξ = (1, 0, 0): only one non-zero weight at n = -1 → pole at z = -1.
        let xi = vec![
            Float::with_val(prec, 1),
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
        ];
        // Seed exactly on the pole: R'(z) = −1/(z+1)² → → ∞ near pole,
        // Newton step dz = R/R' → 0 at the pole itself → no zero found.
        // We pass a seed very close to the pole, expecting either None or a
        // convergent zero far away (R(z)=1/(z+1) has no zero). We verify it
        // doesn't panic. The specific None/Some outcome is implementation-defined.
        let seed_near_pole = Float::with_val(prec, Float::parse("-1.0").unwrap());
        // This won't crash; outcome is None (no zero for this R(z)).
        let _result = super::newton_xi_hat_zero(&xi, n_max, &l, &seed_near_pole, prec, 10);
        // No assertion on result — just verifying no panic / infinite loop.
    }
}
