//! Multi-zero CCM observation contracts and independent f64 discovery tools.
//!
//! Reference zeta zeros are deliberately absent from the discovery API.  A
//! caller may compare certified or cross-checked roots with references only
//! after discovery and ordering are complete.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};
use xc_certify::DecimalInterval;
use xc_core::{AssuranceLevel, ConvergenceTableRow};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CcmContinuumStatus {
    FiniteConfiguration,
    FiniteSequenceEvidence,
    ContinuumCertified,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmFiniteClaimContext {
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmPublicationClaim {
    pub claim_id: String,
    pub statement: String,
    pub continuum_status: CcmContinuumStatus,
    pub assurance: AssuranceLevel,
    pub finite_context: Option<CcmFiniteClaimContext>,
    pub convergence_certificate_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmPublicationClaimArtifact {
    pub schema_version: u32,
    pub claims: Vec<CcmPublicationClaim>,
}

impl CcmPublicationClaimArtifact {
    pub fn validate(&self) -> Result<(), WindowError> {
        if self.schema_version != 1 || self.claims.is_empty() {
            return Err(WindowError::InvalidRequest(
                "CCM publication claims require schema 1 and at least one claim".to_owned(),
            ));
        }
        let mut identifiers = std::collections::BTreeSet::new();
        for claim in &self.claims {
            if claim.claim_id.trim().is_empty()
                || claim.statement.trim().is_empty()
                || !identifiers.insert(claim.claim_id.as_str())
            {
                return Err(WindowError::InvalidRequest(
                    "CCM publication claim identifiers and statements must be nonempty and unique"
                        .to_owned(),
                ));
            }
            match claim.continuum_status {
                CcmContinuumStatus::FiniteConfiguration => {
                    let context = claim.finite_context.as_ref().ok_or_else(|| {
                        WindowError::InvalidRequest(
                            "finite CCM claims require their exact finite context".to_owned(),
                        )
                    })?;
                    validate_finite_claim_context(context)?;
                }
                CcmContinuumStatus::FiniteSequenceEvidence => {
                    let context = claim.finite_context.as_ref().ok_or_else(|| {
                        WindowError::InvalidRequest(
                            "finite sequence claims require a representative finite context"
                                .to_owned(),
                        )
                    })?;
                    validate_finite_claim_context(context)?;
                }
                CcmContinuumStatus::ContinuumCertified => {
                    if claim.assurance != AssuranceLevel::Certified
                        || claim
                            .convergence_certificate_id
                            .as_ref()
                            .is_none_or(|identifier| identifier.trim().is_empty())
                    {
                        return Err(WindowError::InvalidRequest(
                            "continuum-certified CCM claims require Certified assurance and a convergence certificate"
                                .to_owned(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Render status-separated publication text so finite evidence cannot be
    /// presented in a continuum-certified section by formatting alone.
    pub fn render_markdown(&self) -> Result<String, WindowError> {
        self.validate()?;
        let mut output = String::from("# CCM claim scopes\n");
        for (heading, status) in [
            (
                "Finite configurations",
                CcmContinuumStatus::FiniteConfiguration,
            ),
            (
                "Finite sequence evidence",
                CcmContinuumStatus::FiniteSequenceEvidence,
            ),
            (
                "Continuum-certified claims",
                CcmContinuumStatus::ContinuumCertified,
            ),
        ] {
            output.push_str(&format!("\n## {heading}\n"));
            for claim in self
                .claims
                .iter()
                .filter(|claim| claim.continuum_status == status)
            {
                output.push_str(&format!("\n- `{}`: {}\n", claim.claim_id, claim.statement));
            }
        }
        Ok(output)
    }
}

fn validate_finite_claim_context(context: &CcmFiniteClaimContext) -> Result<(), WindowError> {
    if context.lambda_squared.trim().is_empty()
        || context.n_modes == 0
        || context.precision_bits < 53
    {
        return Err(WindowError::InvalidRequest(
            "finite CCM claim context requires lambda-squared, positive modes, and at least 53 bits"
                .to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum ZeroTarget {
    FirstK { count: usize },
    IndexRange { first: usize, last: usize },
    HeightWindow { lower: String, upper: String },
    SymmetricHeightWindow { height: String },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMode {
    #[default]
    Independent,
    ReferenceSeededAudit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcmObservationRequest {
    pub target: ZeroTarget,
    pub target_digits: u32,
    pub lambda_sq: f64,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub assurance: AssuranceLevel,
    #[serde(default)]
    pub discovery_mode: DiscoveryMode,
}

impl CcmObservationRequest {
    /// Construct a reference-free observation request. Independent discovery
    /// is deliberately the ordinary API; reference-seeded refinement must be
    /// selected explicitly with [`Self::reference_seeded_audit`].
    pub fn independent(
        target: ZeroTarget,
        target_digits: u32,
        lambda_sq: f64,
        n_modes: usize,
        precision_bits: u32,
        assurance: AssuranceLevel,
    ) -> Self {
        Self {
            target,
            target_digits,
            lambda_sq,
            n_modes,
            precision_bits,
            assurance,
            discovery_mode: DiscoveryMode::Independent,
        }
    }

    /// Explicit opt-in for a reference-seeded comparison/refinement run.
    pub fn reference_seeded_audit(mut self) -> Self {
        self.discovery_mode = DiscoveryMode::ReferenceSeededAudit;
        self
    }
}

impl CcmObservationRequest {
    pub fn validate(&self) -> Result<(), WindowError> {
        if !self.lambda_sq.is_finite() || self.lambda_sq <= 1.0 {
            return Err(WindowError::InvalidRequest(
                "lambda_sq must be finite and greater than one".to_owned(),
            ));
        }
        if self.n_modes == 0 {
            return Err(WindowError::InvalidRequest(
                "n_modes must be positive".to_owned(),
            ));
        }
        if self.precision_bits < 53 {
            return Err(WindowError::InvalidRequest(
                "production observation precision must be at least 53 bits".to_owned(),
            ));
        }
        match &self.target {
            ZeroTarget::FirstK { count } if *count == 0 => Err(WindowError::InvalidRequest(
                "FirstK count must be positive".to_owned(),
            )),
            ZeroTarget::IndexRange { first, last } if *first == 0 || *first > *last => {
                Err(WindowError::InvalidRequest(
                    "positive root indices are one-based and require first <= last".to_owned(),
                ))
            }
            ZeroTarget::HeightWindow { lower, upper } => {
                let lower = parse_finite(lower, "lower height")?;
                let upper = parse_finite(upper, "upper height")?;
                if lower >= upper {
                    return Err(WindowError::InvalidRequest(
                        "height window must have lower < upper".to_owned(),
                    ));
                }
                Ok(())
            }
            ZeroTarget::SymmetricHeightWindow { height } => {
                if parse_finite(height, "height")? <= 0.0 {
                    return Err(WindowError::InvalidRequest(
                        "symmetric height must be positive".to_owned(),
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectralReach {
    pub lambda_sq: f64,
    pub n_modes: usize,
    pub largest_positive_pole: f64,
    pub requested_height: f64,
    pub minimum_modes_for_reach: usize,
    pub reaches_window: bool,
}

pub fn minimum_modes_for_height(lambda_sq: f64, height: f64) -> Result<usize, WindowError> {
    if !lambda_sq.is_finite() || lambda_sq <= 1.0 {
        return Err(WindowError::InvalidRequest(
            "lambda_sq must be finite and greater than one".to_owned(),
        ));
    }
    if !height.is_finite() || height < 0.0 {
        return Err(WindowError::InvalidRequest(
            "height must be finite and nonnegative".to_owned(),
        ));
    }
    Ok((height * lambda_sq.ln() / (2.0 * PI)).ceil() as usize)
}

pub fn spectral_reach(
    lambda_sq: f64,
    n_modes: usize,
    requested_height: f64,
) -> Result<SpectralReach, WindowError> {
    let minimum = minimum_modes_for_height(lambda_sq, requested_height)?;
    let largest_positive_pole = 2.0 * PI * n_modes as f64 / lambda_sq.ln();
    Ok(SpectralReach {
        lambda_sq,
        n_modes,
        largest_positive_pole,
        requested_height,
        minimum_modes_for_reach: minimum,
        reaches_window: n_modes >= minimum,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationPlan {
    pub estimated_height: f64,
    pub minimum_modes_for_reach: usize,
    pub recommended_precision_bits: u32,
    pub guard_digits: u32,
    pub analytic_context: xc_core::AnalyticProblemContext,
}

/// Leading Riemann-von Mangoldt inversion used only for capacity planning.
/// It is not a reference zero and is never used as a root seed.
pub fn estimate_nth_zero_height(index: usize) -> Result<f64, WindowError> {
    if index == 0 {
        return Err(WindowError::InvalidRequest(
            "positive zero index is one-based".to_owned(),
        ));
    }
    let target = index as f64 - 0.875;
    let mut t = if index <= 5 {
        14.0 + 7.0 * (index.saturating_sub(1)) as f64
    } else {
        (2.0 * PI * index as f64 / (index as f64).ln()).max(14.0)
    };
    for _ in 0..30 {
        let log_term = (t / (2.0 * PI)).ln();
        let f = t / (2.0 * PI) * (log_term - 1.0) - target;
        let derivative = log_term / (2.0 * PI);
        if derivative.abs() < 1e-12 {
            break;
        }
        let next = t - f / derivative;
        if !next.is_finite() || next <= 2.0 * PI {
            t = 0.5 * (t + 2.0 * PI + 1.0);
        } else {
            if (next - t).abs() <= 1e-12 * t.max(1.0) {
                t = next;
                break;
            }
            t = next;
        }
    }
    Ok(t)
}

/// Plans the finite CCM window needed for a requested zero target.
///
/// # Mathematical semantics
/// Converts a first-count, index-range, or height target into a conservative
/// finite mode count, spectral reach, working precision, and explicit guards.
/// It plans an observation; it does not use reference zeros as solver seeds.
///
/// # Precision
/// Decimal target and guard digits determine the recommended binary precision.
/// This binary64 planning calculation does not perform the later HP solve.
///
/// # Failure states
/// Non-finite or nonpositive parameters, empty targets, invalid ranges, and
/// unrepresentable sizes return `WindowError` before computation.
///
/// # Assurance and validity
/// The plan explicitly records that a finite CCM observation neither proves RH
/// nor establishes convergence to an infinite-dimensional operator.
///
/// # Cache effects
/// Planning performs no cache lookup or publication. Executions bind reused or
/// generated artifacts later through the common provenance contract.
///
/// # Example
/// Compiled example: `crates/xc-spectral/examples/ccm_window_plan.rs`.
pub fn plan_observation(
    lambda_sq: f64,
    target: &ZeroTarget,
    target_digits: u32,
    guard_digits: u32,
) -> Result<ObservationPlan, WindowError> {
    let estimated_height = match target {
        ZeroTarget::FirstK { count } => estimate_nth_zero_height(*count)?,
        ZeroTarget::IndexRange { last, .. } => estimate_nth_zero_height(*last)?,
        ZeroTarget::HeightWindow { upper, .. } => parse_finite(upper, "upper height")?,
        ZeroTarget::SymmetricHeightWindow { height } => parse_finite(height, "height")?,
    };
    let minimum_modes_for_reach = minimum_modes_for_height(lambda_sq, estimated_height)?;
    let decimal_digits = target_digits.saturating_add(guard_digits);
    let recommended_precision_bits =
        ((decimal_digits as f64) * std::f64::consts::LOG2_10).ceil() as u32;
    let plan = ObservationPlan {
        estimated_height,
        minimum_modes_for_reach,
        recommended_precision_bits: recommended_precision_bits.max(64),
        guard_digits,
        analytic_context: xc_core::AnalyticProblemContext {
            schema_version: 1,
            domain: "ccm_observation_planning".to_owned(),
            requested_assurance: xc_core::AssuranceLevel::Computed,
            assumptions: vec![
                xc_core::AnalyticAssumption {
                    assumption_id: "leading_zero_count".to_owned(),
                    kind: xc_core::AnalyticAssumptionKind::AsymptoticEstimate,
                    statement: "height planning uses only the leading Riemann-von Mangoldt count"
                        .to_owned(),
                    scope: "capacity planning; never root discovery or certification".to_owned(),
                    evidence_ids: Vec::new(),
                },
                xc_core::AnalyticAssumption {
                    assumption_id: "mode_reach_only".to_owned(),
                    kind: xc_core::AnalyticAssumptionKind::Model,
                    statement: "mode reach is necessary but does not establish accuracy".to_owned(),
                    scope: "finite CCM observation plan".to_owned(),
                    evidence_ids: Vec::new(),
                },
            ],
            error_budget: xc_core::AnalyticErrorBudget {
                quantity: "estimated height, reach, and recommended precision".to_owned(),
                terms: vec![
                    xc_core::AnalyticErrorTerm {
                        term_id: "zero_count_remainder".to_owned(),
                        source: "omitted lower-order Riemann-von Mangoldt terms".to_owned(),
                        affects: "estimated target height".to_owned(),
                        decisive_for_claim: true,
                        rigorous: false,
                        absolute_bound: None,
                        proof_method: None,
                        evidence_sha256: None,
                    },
                    xc_core::AnalyticErrorTerm {
                        term_id: "construction_conditioning".to_owned(),
                        source: "unmeasured matrix construction and root conditioning".to_owned(),
                        affects: "recommended precision and attainable digits".to_owned(),
                        decisive_for_claim: true,
                        rigorous: false,
                        absolute_bound: None,
                        proof_method: None,
                        evidence_sha256: None,
                    },
                ],
                total_absolute_bound: None,
                aggregation_method:
                    "unbounded planning diagnostics; measure before assurance promotion".to_owned(),
            },
        },
    };
    plan.analytic_context
        .validate()
        .map_err(|error| WindowError::InvalidRequest(error.to_string()))?;
    Ok(plan)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmPlannerCandidate {
    pub candidate_id: String,
    pub lambda_squared: String,
    pub n_modes: u64,
    pub precision_bits: u64,
    pub calibrated_root_count: u64,
    pub calibrated_uniform_digits: u32,
    pub calibration_evidence_digest: String,
}

impl CcmPlannerCandidate {
    fn validate(&self) -> Result<f64, WindowError> {
        let lambda_squared = parse_finite(&self.lambda_squared, "planner lambda squared")?;
        if self.candidate_id.trim().is_empty()
            || lambda_squared <= 1.0
            || self.n_modes == 0
            || self.precision_bits < 53
            || self.calibrated_root_count == 0
            || self.calibrated_uniform_digits == 0
            || !is_lower_hex_digest(&self.calibration_evidence_digest)
        {
            return Err(WindowError::InvalidRequest(
                "CCM planner candidate lacks configuration or calibration evidence".to_owned(),
            ));
        }
        Ok(lambda_squared)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmFirstKPlanningRequest {
    pub requested_roots: u64,
    pub target_uniform_digits: u32,
    pub precision_guard_digits: u32,
    pub minimum_reach_margin_modes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmPlannerCandidateRejection {
    pub candidate_id: String,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmFirstKObservationPlan {
    pub schema_version: u32,
    pub request: CcmFirstKPlanningRequest,
    pub selected: CcmPlannerCandidate,
    pub estimated_height: String,
    pub minimum_modes_for_reach: u64,
    pub reach_margin_modes: u64,
    pub recommended_precision_bits: u64,
    pub precision_margin_bits: u64,
    pub rejected_candidates: Vec<CcmPlannerCandidateRejection>,
    pub finite_planning_statement: String,
}

fn recommended_precision_bits(request: &CcmFirstKPlanningRequest) -> u64 {
    let decimal_digits = request
        .target_uniform_digits
        .saturating_add(request.precision_guard_digits);
    (((decimal_digits as f64) * std::f64::consts::LOG2_10).ceil() as u64).max(64)
}

fn validate_first_k_plan(plan: &CcmFirstKObservationPlan) -> Result<f64, WindowError> {
    if plan.schema_version != 1
        || plan.request.requested_roots == 0
        || plan.request.target_uniform_digits == 0
        || plan.finite_planning_statement.trim().is_empty()
    {
        return Err(WindowError::InvalidRequest(
            "CCM observation plan identity is invalid".to_owned(),
        ));
    }
    let lambda_squared = plan.selected.validate()?;
    if plan.selected.calibrated_root_count < plan.request.requested_roots
        || plan.selected.calibrated_uniform_digits < plan.request.target_uniform_digits
    {
        return Err(WindowError::InvalidRequest(
            "selected CCM configuration exceeds its calibration evidence".to_owned(),
        ));
    }
    let requested_roots = usize::try_from(plan.request.requested_roots).map_err(|_| {
        WindowError::InvalidRequest("requested root count does not fit this platform".to_owned())
    })?;
    let estimated_height = estimate_nth_zero_height(requested_roots)?;
    let minimum_modes = u64::try_from(minimum_modes_for_height(lambda_squared, estimated_height)?)
        .map_err(|_| {
            WindowError::InvalidRequest(
                "minimum mode count does not fit the plan schema".to_owned(),
            )
        })?;
    let precision_bits = recommended_precision_bits(&plan.request);
    let expected_height = estimated_height.to_string();
    let expected_reach_margin = plan
        .selected
        .n_modes
        .checked_sub(minimum_modes)
        .ok_or_else(|| {
            WindowError::InvalidRequest("selected CCM mode count cannot reach K".to_owned())
        })?;
    let expected_precision_margin = plan
        .selected
        .precision_bits
        .checked_sub(precision_bits)
        .ok_or_else(|| {
            WindowError::InvalidRequest(
                "selected CCM precision is below the recommendation".to_owned(),
            )
        })?;
    if plan.estimated_height != expected_height
        || plan.minimum_modes_for_reach != minimum_modes
        || plan.reach_margin_modes != expected_reach_margin
        || plan.reach_margin_modes < plan.request.minimum_reach_margin_modes
        || plan.recommended_precision_bits != precision_bits
        || plan.precision_margin_bits != expected_precision_margin
        || plan.rejected_candidates.iter().any(|rejection| {
            rejection.candidate_id.trim().is_empty()
                || rejection.candidate_id == plan.selected.candidate_id
                || rejection.reasons.is_empty()
                || rejection
                    .reasons
                    .iter()
                    .any(|reason| reason.trim().is_empty())
        })
    {
        return Err(WindowError::InvalidRequest(
            "CCM observation plan does not match its replayed safety calculations".to_owned(),
        ));
    }
    Ok(lambda_squared)
}

/// Select the least estimated-work calibrated configuration that satisfies
/// requested root-count, uniform-digit, reach, and precision safety margins.
/// Calibration limits are eligibility evidence, not a promise of achieved
/// accuracy; the returned plan must be verified after execution.
pub fn plan_first_k_ccm_observation(
    request: CcmFirstKPlanningRequest,
    candidates: &[CcmPlannerCandidate],
) -> Result<CcmFirstKObservationPlan, WindowError> {
    if request.requested_roots == 0 || request.target_uniform_digits == 0 || candidates.is_empty() {
        return Err(WindowError::InvalidRequest(
            "CCM first-K planning requires roots, digits, and calibrated candidates".to_owned(),
        ));
    }
    let requested_roots = usize::try_from(request.requested_roots).map_err(|_| {
        WindowError::InvalidRequest("requested root count does not fit this platform".to_owned())
    })?;
    let estimated_height = estimate_nth_zero_height(requested_roots)?;
    let recommended_precision_bits = recommended_precision_bits(&request);
    let mut eligible = Vec::new();
    let mut rejected_candidates = Vec::new();
    let mut identifiers = std::collections::BTreeSet::new();
    for candidate in candidates {
        let lambda_squared = candidate.validate()?;
        if !identifiers.insert(candidate.candidate_id.as_str()) {
            return Err(WindowError::InvalidRequest(
                "CCM planner candidate identifiers must be unique".to_owned(),
            ));
        }
        let minimum_modes =
            u64::try_from(minimum_modes_for_height(lambda_squared, estimated_height)?).map_err(
                |_| {
                    WindowError::InvalidRequest(
                        "minimum mode count does not fit the plan schema".to_owned(),
                    )
                },
            )?;
        let required_modes = minimum_modes.saturating_add(request.minimum_reach_margin_modes);
        let mut reasons = Vec::new();
        if candidate.calibrated_root_count < request.requested_roots {
            reasons.push("calibrated root-count range is too small".to_owned());
        }
        if candidate.calibrated_uniform_digits < request.target_uniform_digits {
            reasons.push("calibrated uniform-digit range is too small".to_owned());
        }
        if candidate.n_modes < required_modes {
            reasons.push(format!(
                "mode count {} is below reach requirement {required_modes}",
                candidate.n_modes
            ));
        }
        if candidate.precision_bits < recommended_precision_bits {
            reasons.push(format!(
                "precision {} is below recommendation {recommended_precision_bits}",
                candidate.precision_bits
            ));
        }
        if reasons.is_empty() {
            let work_score =
                u128::from(candidate.n_modes).saturating_mul(u128::from(candidate.precision_bits));
            eligible.push((
                work_score,
                candidate.candidate_id.as_str(),
                candidate,
                minimum_modes,
            ));
        } else {
            rejected_candidates.push(CcmPlannerCandidateRejection {
                candidate_id: candidate.candidate_id.clone(),
                reasons,
            });
        }
    }
    let (_, _, selected, minimum_modes_for_reach) = eligible
        .into_iter()
        .min_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)))
        .ok_or_else(|| {
            WindowError::InvalidRequest(
                "no calibrated CCM candidate satisfies the requested safety margins".to_owned(),
            )
        })?;
    Ok(CcmFirstKObservationPlan {
        schema_version: 1,
        request,
        selected: selected.clone(),
        estimated_height: estimated_height.to_string(),
        minimum_modes_for_reach,
        reach_margin_modes: selected.n_modes - minimum_modes_for_reach,
        recommended_precision_bits,
        precision_margin_bits: selected.precision_bits - recommended_precision_bits,
        rejected_candidates,
        finite_planning_statement: "calibrated finite CCM configuration proposal; achieved K, D_min, reach, and assurance require post-run verification"
            .to_owned(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmObservationPlanVerification {
    pub schema_version: u32,
    pub plan: CcmFirstKObservationPlan,
    pub measured: ConvergenceTableRow,
    pub configuration_matches: bool,
    pub root_target_met: bool,
    pub uniform_digit_target_met: bool,
    pub spectral_reach_verified: bool,
    pub accepted: bool,
    pub reasons: Vec<String>,
}

pub fn verify_first_k_ccm_observation_plan(
    plan: &CcmFirstKObservationPlan,
    measured: ConvergenceTableRow,
) -> Result<CcmObservationPlanVerification, WindowError> {
    let lambda_squared = validate_first_k_plan(plan)?;
    let measured_lambda = parse_finite(&measured.lambda_squared, "measured lambda squared")?;
    let measured_digits = measured_minimum_digits(&measured)?;
    let estimated_height = parse_finite(&plan.estimated_height, "planned height")?;
    let measured_modes = usize::try_from(measured.n_modes).map_err(|_| {
        WindowError::InvalidRequest("measured mode count does not fit this platform".to_owned())
    })?;
    let reach = spectral_reach(measured_lambda, measured_modes, estimated_height)?;
    let configuration_matches = measured.lambda_squared == plan.selected.lambda_squared
        && measured.n_modes == plan.selected.n_modes
        && measured.precision_bits == plan.selected.precision_bits
        && measured_lambda == lambda_squared;
    let root_target_met = measured.root_count >= plan.request.requested_roots;
    let uniform_digit_target_met = measured_digits >= plan.request.target_uniform_digits;
    let spectral_reach_verified = reach.reaches_window
        && measured.n_modes
            >= plan
                .minimum_modes_for_reach
                .saturating_add(plan.request.minimum_reach_margin_modes);
    let mut reasons = Vec::new();
    if !configuration_matches {
        reasons.push("measured effective configuration differs from the plan".to_owned());
    }
    if !root_target_met {
        reasons.push("measured root count did not meet K".to_owned());
    }
    if !uniform_digit_target_met {
        reasons.push("measured D_min did not meet the uniform digit target".to_owned());
    }
    if !spectral_reach_verified {
        reasons.push("measured configuration did not preserve the reach margin".to_owned());
    }
    let accepted = reasons.is_empty();
    Ok(CcmObservationPlanVerification {
        schema_version: 1,
        plan: plan.clone(),
        measured,
        configuration_matches,
        root_target_met,
        uniform_digit_target_met,
        spectral_reach_verified,
        accepted,
        reasons,
    })
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootStatus {
    Discovered,
    Refined,
    CrossChecked,
    Certified,
    Duplicate,
    Crossover,
    Skipped,
    Unresolved,
    TooCloseToPole,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcmRootRecord {
    pub positive_index: Option<usize>,
    pub midpoint: String,
    pub enclosure: Option<DecimalInterval>,
    pub residual_bound: Option<String>,
    pub derivative_magnitude: Option<String>,
    #[serde(default)]
    pub conditioning: Option<String>,
    #[serde(default)]
    pub isolation_distance: Option<String>,
    pub nearest_left_pole: Option<String>,
    pub nearest_right_pole: Option<String>,
    pub precision_bits: u32,
    #[serde(default)]
    pub precision_history_bits: Vec<u32>,
    #[serde(default)]
    pub certified_digits: Option<u32>,
    pub status: RootStatus,
    pub discovery_method: String,
    pub crosscheck_method: Option<String>,
    pub reference_comparison_digits: Option<f64>,
}

/// Per-root adaptive-precision policy.  Precision changes are local to one
/// root; a difficult root cannot silently raise the working precision of its
/// well-conditioned neighbors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AdaptivePrecisionPolicy {
    pub initial_precision_bits: u32,
    pub maximum_precision_bits: u32,
    /// Multiplicative growth, represented exactly as a rational pair.
    pub growth_numerator: u32,
    pub growth_denominator: u32,
    pub maximum_attempts_per_root: usize,
}

impl AdaptivePrecisionPolicy {
    pub fn validate(&self) -> Result<(), WindowError> {
        if self.initial_precision_bits < 53
            || self.maximum_precision_bits < self.initial_precision_bits
            || self.growth_denominator == 0
            || self.growth_numerator <= self.growth_denominator
            || self.maximum_attempts_per_root == 0
        {
            return Err(WindowError::InvalidRequest(
                "invalid adaptive per-root precision policy".to_owned(),
            ));
        }
        Ok(())
    }

    fn next_precision(&self, current: u32) -> u32 {
        let grown = u64::from(current)
            .saturating_mul(u64::from(self.growth_numerator))
            .div_ceil(u64::from(self.growth_denominator));
        grown
            .min(u64::from(self.maximum_precision_bits))
            .max(u64::from(current) + 1) as u32
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRootCandidate {
    pub candidate_id: String,
    pub seed: String,
    pub requested_digits: u32,
}

/// Evidence returned by one root-refinement attempt.  The callback producing
/// this record is responsible for constructing/evaluating the source at
/// `precision_bits`; promoting lower-precision coefficients is not allowed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootPrecisionAttempt {
    pub precision_bits: u32,
    pub midpoint: String,
    pub enclosure: Option<DecimalInterval>,
    pub residual_bound: String,
    pub conditioning: String,
    pub isolation_distance: String,
    pub nearest_pole_distance: String,
    pub certified_digits: Option<u32>,
    pub converged: bool,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRootOutcome {
    pub candidate: AdaptiveRootCandidate,
    pub attempts: Vec<RootPrecisionAttempt>,
    pub achieved: bool,
    pub limiting_reason: Option<String>,
}

/// Refine candidates independently, escalating only the root whose latest
/// attempt has not met its requested digit target.
pub fn refine_roots_adaptively<F>(
    candidates: &[AdaptiveRootCandidate],
    policy: &AdaptivePrecisionPolicy,
    mut refine_one: F,
) -> Result<Vec<AdaptiveRootOutcome>, WindowError>
where
    F: FnMut(&AdaptiveRootCandidate, u32) -> Result<RootPrecisionAttempt, WindowError>,
{
    policy.validate()?;
    let mut outcomes = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.candidate_id.trim().is_empty()
            || candidate.seed.trim().is_empty()
            || candidate.requested_digits == 0
        {
            return Err(WindowError::InvalidRequest(
                "adaptive root candidates require id, seed, and positive digit target".to_owned(),
            ));
        }
        let mut precision = policy.initial_precision_bits;
        let mut attempts = Vec::new();
        let mut achieved = false;
        let mut limiting_reason = None;
        for _ in 0..policy.maximum_attempts_per_root {
            let attempt = refine_one(candidate, precision)?;
            if attempt.precision_bits != precision {
                return Err(WindowError::EvaluationFailed(format!(
                    "root {} refinement reported precision {} but {} was requested",
                    candidate.candidate_id, attempt.precision_bits, precision
                )));
            }
            let meets_digits = attempt
                .certified_digits
                .is_some_and(|digits| digits >= candidate.requested_digits);
            let attempt_converged = attempt.converged && meets_digits;
            limiting_reason.clone_from(&attempt.diagnostic);
            attempts.push(attempt);
            if attempt_converged {
                achieved = true;
                limiting_reason = None;
                break;
            }
            if precision == policy.maximum_precision_bits {
                limiting_reason.get_or_insert_with(|| {
                    "maximum per-root precision reached before target evidence".to_owned()
                });
                break;
            }
            precision = policy.next_precision(precision);
        }
        if !achieved && limiting_reason.is_none() {
            limiting_reason = Some("per-root attempt limit reached".to_owned());
        }
        outcomes.push(AdaptiveRootOutcome {
            candidate: candidate.clone(),
            attempts,
            achieved,
            limiting_reason,
        });
    }
    Ok(outcomes)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowCompleteness {
    Unverified,
    CountMatched,
    Certified,
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcmSpectralWindow {
    pub request: CcmObservationRequest,
    pub reach: SpectralReach,
    pub roots: Vec<CcmRootRecord>,
    pub independently_counted_roots: Option<usize>,
    pub completeness: WindowCompleteness,
    pub source_artifact_digest: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmResearchTarget {
    pub sequence_index: u64,
    pub lambda_squared: String,
    pub n_modes: u64,
    pub precision_bits: u64,
    pub root_count: u64,
    pub uniform_accuracy_target_digits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum CcmResearchSequenceOutcome {
    Achieved,
    Failed {
        first_failed_sequence_index: u64,
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmResearchObservation {
    pub target: CcmResearchTarget,
    pub measured: ConvergenceTableRow,
    pub limiting_reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmResearchSequence {
    pub schema_version: u32,
    pub observations: Vec<CcmResearchObservation>,
    pub windows_strictly_increase: bool,
    pub accuracy_targets_strictly_increase: bool,
    pub measured_minimum_accuracy_strictly_increases: bool,
    pub outcome: CcmResearchSequenceOutcome,
    pub finite_scope_statement: String,
}

fn measured_minimum_digits(row: &ConvergenceTableRow) -> Result<u32, WindowError> {
    row.minimum_accuracy_digits.parse::<u32>().map_err(|error| {
        WindowError::InvalidRequest(format!(
            "minimum measured accuracy must be a nonnegative integer: {error}"
        ))
    })
}

/// Build the serialized growing-window experiment required by ACC-005.
///
/// The requested `K` and uniform digit targets must increase strictly. A
/// measured plateau or regression is retained as a valid failed research
/// outcome, including its first failure, rather than rejected as malformed
/// data. This finite sequence never represents an infinite-limit claim.
pub fn build_ccm_research_sequence(
    observations: Vec<CcmResearchObservation>,
) -> Result<CcmResearchSequence, WindowError> {
    if observations.len() < 2 {
        return Err(WindowError::InvalidRequest(
            "a CCM research sequence requires at least two observations".to_owned(),
        ));
    }
    let mut previous_k = 0;
    let mut previous_target = 0;
    let mut previous_measured = None;
    let mut failure = None;
    for (offset, observation) in observations.iter().enumerate() {
        let expected_index = offset as u64 + 1;
        let target = &observation.target;
        let measured = &observation.measured;
        if target.sequence_index != expected_index
            || measured.sequence_index != expected_index
            || target.lambda_squared != measured.lambda_squared
            || target.n_modes != measured.n_modes
            || target.precision_bits != measured.precision_bits
            || target.root_count != measured.root_count
            || target.root_count == 0
            || target.n_modes == 0
            || target.precision_bits < 53
            || target.uniform_accuracy_target_digits == 0
        {
            return Err(WindowError::InvalidRequest(format!(
                "CCM research observation {expected_index} target and measured identities disagree"
            )));
        }
        if offset > 0
            && (target.root_count <= previous_k
                || target.uniform_accuracy_target_digits <= previous_target)
        {
            return Err(WindowError::InvalidRequest(
                "CCM research K and uniform accuracy targets must increase strictly".to_owned(),
            ));
        }
        let measured_digits = measured_minimum_digits(measured)?;
        if measured_digits < target.uniform_accuracy_target_digits && failure.is_none() {
            failure = Some((
                expected_index,
                format!(
                    "measured D_min={measured_digits} did not meet target {}",
                    target.uniform_accuracy_target_digits
                ),
            ));
        }
        if previous_measured.is_some_and(|previous| measured_digits <= previous)
            && failure.is_none()
        {
            failure = Some((
                expected_index,
                format!(
                    "measured D_min={measured_digits} did not increase above the preceding value {}",
                    previous_measured.unwrap_or(0)
                ),
            ));
        }
        if failure.is_none() {
            failure = observation
                .limiting_reason
                .as_ref()
                .filter(|reason| !reason.trim().is_empty())
                .map(|reason| (expected_index, reason.clone()));
        }
        previous_k = target.root_count;
        previous_target = target.uniform_accuracy_target_digits;
        previous_measured = Some(measured_digits);
    }
    let outcome = failure.map_or(CcmResearchSequenceOutcome::Achieved, |(index, reason)| {
        CcmResearchSequenceOutcome::Failed {
            first_failed_sequence_index: index,
            reason,
        }
    });
    Ok(CcmResearchSequence {
        schema_version: 1,
        observations,
        windows_strictly_increase: true,
        accuracy_targets_strictly_increase: true,
        measured_minimum_accuracy_strictly_increases: matches!(
            &outcome,
            CcmResearchSequenceOutcome::Achieved
        ),
        outcome,
        finite_scope_statement: "finite measured CCM sequence; no finite trend establishes a K-to-infinity or continuum limit"
            .to_owned(),
    })
}

pub fn verify_ccm_research_sequence(sequence: &CcmResearchSequence) -> Result<(), WindowError> {
    let replay = build_ccm_research_sequence(sequence.observations.clone())?;
    if sequence.schema_version != 1
        || sequence.finite_scope_statement.trim().is_empty()
        || &replay != sequence
    {
        return Err(WindowError::InvalidRequest(
            "CCM research sequence does not match its replayed measurements".to_owned(),
        ));
    }
    Ok(())
}

/// Convert an accepted positive-root window into the fixed publication convergence
/// row from TD-05. Accuracy summaries come only from each root's recorded
/// `certified_digits`; reference-comparison digits are deliberately ignored.
/// The index penalty is defined explicitly as first-root digits minus
/// last-root digits, so the exported statistic has one stable meaning.
pub fn convergence_table_row(
    sequence_index: u64,
    request: &CcmObservationRequest,
    roots: &[CcmRootRecord],
) -> Result<ConvergenceTableRow, WindowError> {
    if sequence_index == 0 {
        return Err(WindowError::InvalidRequest(
            "convergence sequence index is one-based".to_owned(),
        ));
    }
    let positive: Vec<_> = roots
        .iter()
        .filter(|root| root.positive_index.is_some())
        .collect();
    if positive.is_empty() {
        return Err(WindowError::InvalidRequest(
            "a convergence row requires positive indexed roots".to_owned(),
        ));
    }
    for (offset, root) in positive.iter().enumerate() {
        if root.positive_index != Some(offset + 1) {
            return Err(WindowError::InvalidRequest(
                "convergence roots must have consecutive one-based positive indices".to_owned(),
            ));
        }
        if matches!(
            root.status,
            RootStatus::Duplicate
                | RootStatus::Crossover
                | RootStatus::Skipped
                | RootStatus::Unresolved
                | RootStatus::TooCloseToPole
                | RootStatus::Failed
        ) {
            return Err(WindowError::InvalidRequest(format!(
                "root {} has nonconverged status {:?}",
                offset + 1,
                root.status
            )));
        }
    }
    let digits: Vec<u32> = positive
        .iter()
        .map(|root| {
            root.certified_digits.ok_or_else(|| {
                WindowError::InvalidRequest(format!(
                    "root {} lacks recorded certified digits",
                    root.positive_index.unwrap_or(0)
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    let minimum = digits.iter().copied().min().unwrap_or(0);
    let mut ordered = digits.clone();
    ordered.sort_unstable();
    let median = exact_integer_median(&ordered);
    let penalty = i64::from(digits[0]) - i64::from(digits[digits.len() - 1]);
    let precision_bits = positive
        .iter()
        .map(|root| u64::from(root.precision_bits))
        .max()
        .unwrap_or(u64::from(request.precision_bits));
    let completion_status = if positive
        .iter()
        .all(|root| root.status == RootStatus::Certified)
    {
        "certified"
    } else if positive.iter().all(|root| {
        matches!(
            root.status,
            RootStatus::Certified | RootStatus::CrossChecked
        )
    }) {
        "cross_checked"
    } else {
        "converged"
    };
    Ok(ConvergenceTableRow {
        sequence_index,
        lambda_squared: request.lambda_sq.to_string(),
        n_modes: request.n_modes as u64,
        precision_bits,
        root_count: positive.len() as u64,
        minimum_accuracy_digits: minimum.to_string(),
        median_accuracy_digits: median,
        index_penalty_digits: penalty.to_string(),
        completion_status: completion_status.to_owned(),
    })
}

fn exact_integer_median(ordered: &[u32]) -> String {
    let middle = ordered.len() / 2;
    if ordered.len() % 2 == 1 {
        return ordered[middle].to_string();
    }
    let sum = u64::from(ordered[middle - 1]) + u64::from(ordered[middle]);
    if sum % 2 == 0 {
        (sum / 2).to_string()
    } else {
        format!("{}.5", sum / 2)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum WindowError {
    InvalidRequest(String),
    InvalidSecularFunction(String),
    EvaluationFailed(String),
}

impl Display for WindowError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid CCM window request: {message}"),
            Self::InvalidSecularFunction(message) => {
                write!(f, "invalid secular function: {message}")
            }
            Self::EvaluationFailed(message) => write!(f, "secular evaluation failed: {message}"),
        }
    }
}

impl Error for WindowError {}

/// f64 discovery-only secular function.  Publication and certified paths use
/// the HP/ball implementations; this type is useful for independent seeds and
/// regression tests.
#[derive(Clone, Debug)]
pub struct SecularFunctionF64 {
    poles: Vec<f64>,
    weights: Vec<f64>,
}

impl SecularFunctionF64 {
    pub fn new(poles: Vec<f64>, weights: Vec<f64>) -> Result<Self, WindowError> {
        if poles.is_empty() || poles.len() != weights.len() {
            return Err(WindowError::InvalidSecularFunction(
                "poles and weights must have equal nonzero length".to_owned(),
            ));
        }
        if poles.iter().chain(&weights).any(|value| !value.is_finite()) {
            return Err(WindowError::InvalidSecularFunction(
                "poles and weights must be finite".to_owned(),
            ));
        }
        if poles.windows(2).any(|w| w[0] >= w[1]) {
            return Err(WindowError::InvalidSecularFunction(
                "poles must be strictly increasing".to_owned(),
            ));
        }
        Ok(Self { poles, weights })
    }

    pub fn poles(&self) -> &[f64] {
        &self.poles
    }

    pub fn evaluate(&self, x: f64) -> Result<f64, WindowError> {
        if !x.is_finite() {
            return Err(WindowError::EvaluationFailed(
                "evaluation point must be finite".to_owned(),
            ));
        }
        let mut value = 0.0;
        for (&pole, &weight) in self.poles.iter().zip(&self.weights) {
            let denominator = x - pole;
            if denominator == 0.0 {
                return Err(WindowError::EvaluationFailed(
                    "evaluation point coincides with a pole".to_owned(),
                ));
            }
            value += weight / denominator;
        }
        if value.is_finite() {
            Ok(value)
        } else {
            Err(WindowError::EvaluationFailed(
                "secular evaluation produced a non-finite value".to_owned(),
            ))
        }
    }

    pub fn derivative(&self, x: f64) -> Result<f64, WindowError> {
        if !x.is_finite() {
            return Err(WindowError::EvaluationFailed(
                "evaluation point must be finite".to_owned(),
            ));
        }
        let mut value = 0.0;
        for (&pole, &weight) in self.poles.iter().zip(&self.weights) {
            let denominator = x - pole;
            if denominator == 0.0 {
                return Err(WindowError::EvaluationFailed(
                    "evaluation point coincides with a pole".to_owned(),
                ));
            }
            value -= weight / (denominator * denominator);
        }
        Ok(value)
    }
}

#[derive(Clone, Debug)]
pub struct DiscoveryOptionsF64 {
    pub subdivisions_per_pole_interval: usize,
    pub bisection_iterations: usize,
    pub pole_margin_fraction: f64,
    pub zero_tolerance: f64,
}

impl Default for DiscoveryOptionsF64 {
    fn default() -> Self {
        Self {
            subdivisions_per_pole_interval: 16,
            bisection_iterations: 100,
            pole_margin_fraction: 1e-10,
            zero_tolerance: 1e-13,
        }
    }
}

/// Discover sign-change roots on pole-free intervals.  This routine makes no
/// completeness claim; it intentionally returns `Unverified` evidence.
pub fn discover_roots_f64(
    secular: &SecularFunctionF64,
    lower: f64,
    upper: f64,
    options: &DiscoveryOptionsF64,
) -> Result<Vec<CcmRootRecord>, WindowError> {
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(WindowError::InvalidRequest(
            "discovery range must have finite lower < upper".to_owned(),
        ));
    }
    if options.subdivisions_per_pole_interval == 0
        || options.bisection_iterations == 0
        || !(0.0..0.1).contains(&options.pole_margin_fraction)
        || !options.zero_tolerance.is_finite()
        || options.zero_tolerance <= 0.0
    {
        return Err(WindowError::InvalidRequest(
            "invalid discovery options".to_owned(),
        ));
    }

    let mut boundaries = vec![lower];
    boundaries.extend(
        secular
            .poles()
            .iter()
            .copied()
            .filter(|pole| *pole > lower && *pole < upper),
    );
    boundaries.push(upper);
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|a, b| *a == *b);

    let mut roots = Vec::new();
    for pair in boundaries.windows(2) {
        let interval_width = pair[1] - pair[0];
        if interval_width <= 0.0 {
            continue;
        }
        let margin = (options.pole_margin_fraction * interval_width)
            .max(f64::EPSILON * pair[0].abs().max(pair[1].abs()).max(1.0));
        let left = pair[0] + margin;
        let right = pair[1] - margin;
        if left >= right {
            continue;
        }
        let subdivisions = options.subdivisions_per_pole_interval;
        let mut x0 = left;
        let mut f0 = secular.evaluate(x0)?;
        for step in 1..=subdivisions {
            let x1 = left + (right - left) * step as f64 / subdivisions as f64;
            let f1 = secular.evaluate(x1)?;
            if f0 == 0.0 || f1 == 0.0 || f0.is_sign_positive() != f1.is_sign_positive() {
                let (mut a, mut b, mut fa) = if f0 == 0.0 {
                    (
                        x0 - options.zero_tolerance,
                        x0 + options.zero_tolerance,
                        secular.evaluate(x0 - options.zero_tolerance)?,
                    )
                } else {
                    (x0, x1, f0)
                };
                for _ in 0..options.bisection_iterations {
                    let midpoint = a + 0.5 * (b - a);
                    let fm = secular.evaluate(midpoint)?;
                    if fm.abs() <= options.zero_tolerance || b - a <= options.zero_tolerance {
                        a = midpoint;
                        b = midpoint;
                        break;
                    }
                    if fa.is_sign_positive() != fm.is_sign_positive() {
                        b = midpoint;
                    } else {
                        a = midpoint;
                        fa = fm;
                    }
                }
                let midpoint = a + 0.5 * (b - a);
                if roots.iter().all(|root: &CcmRootRecord| {
                    root.midpoint.parse::<f64>().map_or(true, |prior| {
                        (prior - midpoint).abs() > 10.0 * options.zero_tolerance
                    })
                }) {
                    roots.push(CcmRootRecord {
                        positive_index: None,
                        midpoint: format!("{midpoint:.17e}"),
                        enclosure: Some(DecimalInterval {
                            lower: format!("{a:.17e}"),
                            upper: format!("{b:.17e}"),
                        }),
                        residual_bound: secular
                            .evaluate(midpoint)
                            .ok()
                            .map(|value| format!("{:.17e}", value.abs())),
                        derivative_magnitude: secular
                            .derivative(midpoint)
                            .ok()
                            .map(|value| format!("{:.17e}", value.abs())),
                        conditioning: secular
                            .derivative(midpoint)
                            .ok()
                            .filter(|value| *value != 0.0)
                            .map(|value| format!("{:.17e}", value.abs().recip())),
                        isolation_distance: Some(format!("{:.17e}", (b - a).abs())),
                        nearest_left_pole: Some(format!("{:.17e}", pair[0])),
                        nearest_right_pole: Some(format!("{:.17e}", pair[1])),
                        precision_bits: 53,
                        precision_history_bits: vec![53],
                        certified_digits: None,
                        status: RootStatus::Discovered,
                        discovery_method: "pole_aware_sign_change_bisection_f64".to_owned(),
                        crosscheck_method: None,
                        reference_comparison_digits: None,
                    });
                }
            }
            x0 = x1;
            f0 = f1;
        }
    }
    roots.sort_by(|a, b| {
        let a = a.midpoint.parse::<f64>().unwrap_or(f64::NAN);
        let b = b.midpoint.parse::<f64>().unwrap_or(f64::NAN);
        a.total_cmp(&b)
    });
    for (index, root) in roots
        .iter_mut()
        .filter(|root| root.midpoint.parse::<f64>().is_ok_and(|value| value > 0.0))
        .enumerate()
    {
        root.positive_index = Some(index + 1);
    }
    Ok(roots)
}

fn parse_finite(value: &str, name: &str) -> Result<f64, WindowError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|e| WindowError::InvalidRequest(format!("invalid {name} decimal: {e}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(WindowError::InvalidRequest(format!(
            "{name} must be finite"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_requests_default_to_independent_discovery() {
        let request = CcmObservationRequest::independent(
            ZeroTarget::FirstK { count: 3 },
            20,
            13.0,
            120,
            256,
            AssuranceLevel::Computed,
        );
        assert_eq!(request.discovery_mode, DiscoveryMode::Independent);

        let encoded = serde_json::to_value(&request).unwrap();
        let mut without_mode = encoded.as_object().unwrap().clone();
        without_mode.remove("discovery_mode");
        let decoded: CcmObservationRequest =
            serde_json::from_value(serde_json::Value::Object(without_mode)).unwrap();
        assert_eq!(decoded.discovery_mode, DiscoveryMode::Independent);
        assert_eq!(
            request.clone().reference_seeded_audit().discovery_mode,
            DiscoveryMode::ReferenceSeededAudit
        );
    }

    #[test]
    fn publication_claim_artifact_separates_finite_sequence_and_continuum_status() {
        let context = CcmFiniteClaimContext {
            lambda_squared: "13".to_owned(),
            n_modes: 50,
            precision_bits: 256,
        };
        let artifact = CcmPublicationClaimArtifact {
            schema_version: 1,
            claims: vec![
                CcmPublicationClaim {
                    claim_id: "finite-plunge".to_owned(),
                    statement: "the finite Weil plunge is positive".to_owned(),
                    continuum_status: CcmContinuumStatus::FiniteConfiguration,
                    assurance: AssuranceLevel::Certified,
                    finite_context: Some(context.clone()),
                    convergence_certificate_id: None,
                },
                CcmPublicationClaim {
                    claim_id: "measured-trend".to_owned(),
                    statement: "the measured finite sequence decreases".to_owned(),
                    continuum_status: CcmContinuumStatus::FiniteSequenceEvidence,
                    assurance: AssuranceLevel::CrossChecked,
                    finite_context: Some(context),
                    convergence_certificate_id: None,
                },
                CcmPublicationClaim {
                    claim_id: "continuum-limit".to_owned(),
                    statement: "the certified convergence theorem supplies the limit".to_owned(),
                    continuum_status: CcmContinuumStatus::ContinuumCertified,
                    assurance: AssuranceLevel::Certified,
                    finite_context: None,
                    convergence_certificate_id: Some("sha256:convergence-proof".to_owned()),
                },
            ],
        };
        let encoded = serde_json::to_string(&artifact).unwrap();
        let decoded: CcmPublicationClaimArtifact = serde_json::from_str(&encoded).unwrap();
        let rendered = decoded.render_markdown().unwrap();
        let finite = rendered.find("## Finite configurations").unwrap();
        let sequence = rendered.find("## Finite sequence evidence").unwrap();
        let continuum = rendered.find("## Continuum-certified claims").unwrap();
        assert!(finite < sequence && sequence < continuum);
        assert!(rendered[finite..sequence].contains("finite-plunge"));
        assert!(!rendered[finite..sequence].contains("continuum-limit"));
        assert!(rendered[continuum..].contains("continuum-limit"));
    }

    #[test]
    fn continuum_label_without_convergence_certificate_is_rejected() {
        let artifact = CcmPublicationClaimArtifact {
            schema_version: 1,
            claims: vec![CcmPublicationClaim {
                claim_id: "unsupported-limit".to_owned(),
                statement: "finite evidence is relabelled as a limit".to_owned(),
                continuum_status: CcmContinuumStatus::ContinuumCertified,
                assurance: AssuranceLevel::Certified,
                finite_context: None,
                convergence_certificate_id: None,
            }],
        };
        assert!(artifact.render_markdown().is_err());
    }

    fn convergence_root(index: usize, digits: u32, precision_bits: u32) -> CcmRootRecord {
        CcmRootRecord {
            positive_index: Some(index),
            midpoint: format!("{}.0", index + 13),
            enclosure: None,
            residual_bound: Some("1e-40".to_owned()),
            derivative_magnitude: Some("1".to_owned()),
            conditioning: Some("1".to_owned()),
            isolation_distance: Some("0.1".to_owned()),
            nearest_left_pole: Some("12".to_owned()),
            nearest_right_pole: Some("15".to_owned()),
            precision_bits,
            precision_history_bits: vec![precision_bits],
            certified_digits: Some(digits),
            status: RootStatus::Certified,
            discovery_method: "independent-test".to_owned(),
            crosscheck_method: Some("second-test-route".to_owned()),
            reference_comparison_digits: Some(200.0),
        }
    }

    #[test]
    fn convergence_row_uses_root_evidence_not_reference_digits() {
        let request = CcmObservationRequest {
            target: ZeroTarget::FirstK { count: 4 },
            target_digits: 10,
            lambda_sq: 13.0,
            n_modes: 120,
            precision_bits: 256,
            assurance: AssuranceLevel::Certified,
            discovery_mode: DiscoveryMode::Independent,
        };
        let roots = vec![
            convergence_root(1, 40, 256),
            convergence_root(2, 30, 384),
            convergence_root(3, 25, 384),
            convergence_root(4, 20, 512),
        ];
        let row = convergence_table_row(2, &request, &roots).unwrap();
        assert_eq!(row.lambda_squared, "13");
        assert_eq!(row.root_count, 4);
        assert_eq!(row.precision_bits, 512);
        assert_eq!(row.minimum_accuracy_digits, "20");
        assert_eq!(row.median_accuracy_digits, "27.5");
        assert_eq!(row.index_penalty_digits, "20");
        assert_eq!(row.completion_status, "certified");
    }

    fn research_observation(
        index: u64,
        k: u64,
        target: u32,
        measured: u32,
    ) -> CcmResearchObservation {
        let precision_bits = 128 + 32 * index;
        let n_modes = k + 10;
        CcmResearchObservation {
            target: CcmResearchTarget {
                sequence_index: index,
                lambda_squared: "5".to_owned(),
                n_modes,
                precision_bits,
                root_count: k,
                uniform_accuracy_target_digits: target,
            },
            measured: ConvergenceTableRow {
                sequence_index: index,
                lambda_squared: "5".to_owned(),
                n_modes,
                precision_bits,
                root_count: k,
                minimum_accuracy_digits: measured.to_string(),
                median_accuracy_digits: measured.to_string(),
                index_penalty_digits: "0".to_owned(),
                completion_status: "certified".to_owned(),
            },
            limiting_reason: None,
        }
    }

    #[test]
    fn growing_window_sequence_records_success_and_replays() {
        let sequence = build_ccm_research_sequence(vec![
            research_observation(1, 10, 8, 12),
            research_observation(2, 25, 14, 20),
            research_observation(3, 50, 24, 31),
        ])
        .unwrap();
        assert_eq!(sequence.outcome, CcmResearchSequenceOutcome::Achieved);
        assert!(sequence.measured_minimum_accuracy_strictly_increases);
        verify_ccm_research_sequence(&sequence).unwrap();
        let decoded: CcmResearchSequence =
            serde_json::from_slice(&serde_json::to_vec(&sequence).unwrap()).unwrap();
        verify_ccm_research_sequence(&decoded).unwrap();
    }

    #[test]
    fn growing_window_sequence_preserves_first_measured_failure() {
        let sequence = build_ccm_research_sequence(vec![
            research_observation(1, 10, 8, 12),
            research_observation(2, 25, 14, 12),
            research_observation(3, 50, 24, 30),
        ])
        .unwrap();
        assert_eq!(
            sequence.outcome,
            CcmResearchSequenceOutcome::Failed {
                first_failed_sequence_index: 2,
                reason: "measured D_min=12 did not meet target 14".to_owned(),
            }
        );
        assert!(!sequence.measured_minimum_accuracy_strictly_increases);
        verify_ccm_research_sequence(&sequence).unwrap();
    }

    #[test]
    fn reach_is_separate_from_accuracy() {
        let reach = spectral_reach(100.0, 500, 680.0).unwrap();
        assert!(reach.reaches_window);
        assert!(reach.largest_positive_pole > 680.0);
    }

    #[test]
    fn planner_does_not_need_reference_zeros() {
        let plan = plan_observation(100.0, &ZeroTarget::FirstK { count: 50 }, 100, 50).unwrap();
        assert!(plan.estimated_height > 100.0);
        assert!(plan.minimum_modes_for_reach > 0);
        assert!(plan.recommended_precision_bits >= 498);
        plan.analytic_context.validate().unwrap();
        assert_eq!(plan.analytic_context.error_budget.terms.len(), 2);
        assert!(plan
            .analytic_context
            .error_budget
            .terms
            .iter()
            .all(|term| !term.rigorous && term.absolute_bound.is_none()));
        let encoded = serde_json::to_vec(&plan).unwrap();
        let decoded: ObservationPlan = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.analytic_context, plan.analytic_context);
        assert_eq!(
            decoded.analytic_context.canonical_sha256().unwrap(),
            plan.analytic_context.canonical_sha256().unwrap()
        );
    }

    fn planner_candidate(
        candidate_id: &str,
        n_modes: u64,
        precision_bits: u64,
        calibrated_root_count: u64,
    ) -> CcmPlannerCandidate {
        CcmPlannerCandidate {
            candidate_id: candidate_id.to_owned(),
            lambda_squared: "5".to_owned(),
            n_modes,
            precision_bits,
            calibrated_root_count,
            calibrated_uniform_digits: 30,
            calibration_evidence_digest: "a".repeat(64),
        }
    }

    #[test]
    fn calibrated_first_k_planner_selects_least_work_and_verifies_measurement() {
        let request = CcmFirstKPlanningRequest {
            requested_roots: 50,
            target_uniform_digits: 20,
            precision_guard_digits: 10,
            minimum_reach_margin_modes: 10,
        };
        let plan = plan_first_k_ccm_observation(
            request,
            &[
                planner_candidate("under-calibrated", 80, 160, 25),
                planner_candidate("wide", 120, 192, 100),
                planner_candidate("efficient", 80, 160, 50),
            ],
        )
        .unwrap();
        assert_eq!(plan.selected.candidate_id, "efficient");
        assert!(plan.reach_margin_modes >= 10);
        assert_eq!(plan.rejected_candidates.len(), 1);
        assert_eq!(plan.rejected_candidates[0].candidate_id, "under-calibrated");

        let measured = ConvergenceTableRow {
            sequence_index: 1,
            lambda_squared: "5".to_owned(),
            n_modes: 80,
            precision_bits: 160,
            root_count: 50,
            minimum_accuracy_digits: "22".to_owned(),
            median_accuracy_digits: "24".to_owned(),
            index_penalty_digits: "3".to_owned(),
            completion_status: "certified".to_owned(),
        };
        let verification = verify_first_k_ccm_observation_plan(&plan, measured.clone()).unwrap();
        assert!(verification.accepted);
        let decoded: CcmObservationPlanVerification =
            serde_json::from_slice(&serde_json::to_vec(&verification).unwrap()).unwrap();
        assert_eq!(decoded, verification);

        let mut insufficient = measured;
        insufficient.minimum_accuracy_digits = "19".to_owned();
        let failed = verify_first_k_ccm_observation_plan(&plan, insufficient).unwrap();
        assert!(!failed.accepted);
        assert!(!failed.uniform_digit_target_met);
    }

    #[test]
    fn calibrated_first_k_planner_rejects_tampered_safety_calculation() {
        let mut plan = plan_first_k_ccm_observation(
            CcmFirstKPlanningRequest {
                requested_roots: 50,
                target_uniform_digits: 20,
                precision_guard_digits: 10,
                minimum_reach_margin_modes: 10,
            },
            &[planner_candidate("eligible", 80, 160, 50)],
        )
        .unwrap();
        plan.reach_margin_modes += 1;
        let measured = ConvergenceTableRow {
            sequence_index: 1,
            lambda_squared: "5".to_owned(),
            n_modes: 80,
            precision_bits: 160,
            root_count: 50,
            minimum_accuracy_digits: "22".to_owned(),
            median_accuracy_digits: "24".to_owned(),
            index_penalty_digits: "3".to_owned(),
            completion_status: "certified".to_owned(),
        };
        assert!(verify_first_k_ccm_observation_plan(&plan, measured).is_err());
    }

    #[test]
    fn pole_aware_discovery_finds_simple_sign_change_root() {
        // R(x) = 1/(x+1) + 1/(x-1) = 2x/(x^2-1), root x=0.
        let secular = SecularFunctionF64::new(vec![-1.0, 1.0], vec![1.0, 1.0]).unwrap();
        let roots =
            discover_roots_f64(&secular, -0.9, 0.9, &DiscoveryOptionsF64::default()).unwrap();
        assert_eq!(roots.len(), 1);
        assert!(roots[0].midpoint.parse::<f64>().unwrap().abs() < 1e-12);
    }

    #[test]
    fn adaptive_precision_escalates_only_the_difficult_root() {
        let candidates = vec![
            AdaptiveRootCandidate {
                candidate_id: "easy".to_owned(),
                seed: "1.0".to_owned(),
                requested_digits: 20,
            },
            AdaptiveRootCandidate {
                candidate_id: "difficult".to_owned(),
                seed: "2.0".to_owned(),
                requested_digits: 50,
            },
        ];
        let policy = AdaptivePrecisionPolicy {
            initial_precision_bits: 80,
            maximum_precision_bits: 320,
            growth_numerator: 2,
            growth_denominator: 1,
            maximum_attempts_per_root: 4,
        };
        let outcomes = refine_roots_adaptively(&candidates, &policy, |candidate, precision| {
            let digits = if candidate.candidate_id == "easy" {
                24
            } else {
                precision / 3
            };
            Ok(RootPrecisionAttempt {
                precision_bits: precision,
                midpoint: candidate.seed.clone(),
                enclosure: None,
                residual_bound: "1e-60".to_owned(),
                conditioning: "1".to_owned(),
                isolation_distance: "0.1".to_owned(),
                nearest_pole_distance: "0.2".to_owned(),
                certified_digits: Some(digits),
                converged: true,
                diagnostic: (digits < candidate.requested_digits)
                    .then(|| "insufficient certified digits".to_owned()),
            })
        })
        .unwrap();
        assert!(outcomes.iter().all(|outcome| outcome.achieved));
        assert_eq!(outcomes[0].attempts.len(), 1);
        assert_eq!(outcomes[0].attempts[0].precision_bits, 80);
        assert_eq!(
            outcomes[1]
                .attempts
                .iter()
                .map(|attempt| attempt.precision_bits)
                .collect::<Vec<_>>(),
            vec![80, 160]
        );
    }

    #[test]
    fn reconciliation_is_independent_and_reference_comparison_is_late() {
        let secular = SecularFunctionF64::new(vec![-1.0, 0.0, 1.0], vec![1.0, 1.0, 1.0]).unwrap();
        let discovered =
            discover_roots_f64(&secular, -0.99, 0.99, &DiscoveryOptionsF64::default()).unwrap();
        assert_eq!(discovered.len(), 2);
        assert!(discovered
            .iter()
            .all(|root| root.reference_comparison_digits.is_none()));

        // The count is supplied as separate evidence, not derived from the
        // discovered vector inside reconciliation.
        let mut reconciled = reconcile_window_roots(&discovered, 2, true, 1e-12).unwrap();
        assert_eq!(reconciled.completeness, WindowCompleteness::Certified);
        assert_eq!(reconciled.isolated_unique_roots, 2);
        compare_ordered_roots_to_references(&mut reconciled.roots, &[1.0 / 3.0_f64.sqrt()])
            .unwrap();
        let positive = reconciled
            .roots
            .iter()
            .find(|root| root.positive_index == Some(1))
            .unwrap();
        assert!(positive.reference_comparison_digits.unwrap() > 10.0);

        let mut with_duplicate = discovered.clone();
        with_duplicate.push(discovered[1].clone());
        let duplicate_report = reconcile_window_roots(&with_duplicate, 2, true, 1e-12).unwrap();
        assert_eq!(duplicate_report.duplicate_count, 1);
        assert_eq!(
            duplicate_report.completeness,
            WindowCompleteness::Inconclusive
        );
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowReconciliation {
    pub roots: Vec<CcmRootRecord>,
    pub isolated_unique_roots: usize,
    pub independently_counted_roots: usize,
    pub duplicate_count: usize,
    pub crossover_count: usize,
    pub skipped_index_count: usize,
    pub completeness: WindowCompleteness,
    pub diagnostics: Vec<String>,
}

/// Reconcile an independently obtained root count with a discovered list.
///
/// The count is an input from a distinct route; it is never inferred from the
/// candidate list.  Ordering, duplicates, index gaps, and overlapping
/// enclosures fail closed before completeness can be reported.
pub fn reconcile_window_roots(
    roots: &[CcmRootRecord],
    independently_counted_roots: usize,
    count_is_certified: bool,
    duplicate_tolerance: f64,
) -> Result<WindowReconciliation, WindowError> {
    if independently_counted_roots == 0
        || !duplicate_tolerance.is_finite()
        || duplicate_tolerance <= 0.0
    {
        return Err(WindowError::InvalidRequest(
            "window reconciliation requires a positive independent count and tolerance".to_owned(),
        ));
    }
    let mut ordered = roots.to_vec();
    for root in &ordered {
        parse_finite(&root.midpoint, "root midpoint")?;
    }
    ordered.sort_by(|left, right| {
        let left = left.midpoint.parse::<f64>().expect("validated midpoint");
        let right = right.midpoint.parse::<f64>().expect("validated midpoint");
        left.total_cmp(&right)
    });

    let mut duplicate_count = 0usize;
    for index in 1..ordered.len() {
        let left_midpoint = ordered[index - 1]
            .midpoint
            .parse::<f64>()
            .expect("validated midpoint");
        let right_midpoint = ordered[index]
            .midpoint
            .parse::<f64>()
            .expect("validated midpoint");
        let interval_overlap = match (&ordered[index - 1].enclosure, &ordered[index].enclosure) {
            (Some(left), Some(right)) => !left
                .is_disjoint_from(right)
                .map_err(|error| WindowError::EvaluationFailed(error.to_string()))?,
            _ => false,
        };
        if interval_overlap || right_midpoint - left_midpoint <= duplicate_tolerance {
            ordered[index].status = RootStatus::Duplicate;
            duplicate_count += 1;
        }
    }

    let declared_indices = ordered
        .iter()
        .filter(|root| root.status != RootStatus::Duplicate)
        .filter_map(|root| root.positive_index)
        .collect::<Vec<_>>();
    let crossover_count = declared_indices
        .windows(2)
        .filter(|pair| pair[0] >= pair[1])
        .count();
    let skipped_index_count = declared_indices
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0] + 1))
        .sum::<usize>();
    if crossover_count > 0 {
        for root in ordered
            .iter_mut()
            .filter(|root| root.status != RootStatus::Duplicate && root.positive_index.is_some())
        {
            root.status = RootStatus::Crossover;
        }
    } else if skipped_index_count > 0 {
        for root in ordered
            .iter_mut()
            .filter(|root| root.status != RootStatus::Duplicate && root.positive_index.is_some())
        {
            root.status = RootStatus::Skipped;
        }
    }

    for (index, root) in ordered
        .iter_mut()
        .filter(|root| {
            root.midpoint
                .parse::<f64>()
                .is_ok_and(|midpoint| midpoint > 0.0)
                && root.status != RootStatus::Duplicate
                && root.status != RootStatus::Crossover
                && root.status != RootStatus::Skipped
        })
        .enumerate()
    {
        root.positive_index = Some(index + 1);
    }
    let isolated_unique_roots = ordered
        .iter()
        .filter(|root| {
            root.status != RootStatus::Duplicate
                && root.status != RootStatus::Crossover
                && root.status != RootStatus::Skipped
                && !matches!(
                    root.status,
                    RootStatus::Unresolved | RootStatus::TooCloseToPole | RootStatus::Failed
                )
                && root.enclosure.is_some()
        })
        .count();
    let has_unresolved = ordered.iter().any(|root| {
        matches!(
            root.status,
            RootStatus::Duplicate
                | RootStatus::Crossover
                | RootStatus::Skipped
                | RootStatus::Unresolved
                | RootStatus::TooCloseToPole
                | RootStatus::Failed
        ) || root.enclosure.is_none()
    });
    let count_matches = isolated_unique_roots == independently_counted_roots;
    let completeness = if has_unresolved || !count_matches {
        WindowCompleteness::Inconclusive
    } else if count_is_certified {
        WindowCompleteness::Certified
    } else {
        WindowCompleteness::CountMatched
    };
    let mut diagnostics = Vec::new();
    if duplicate_count > 0 {
        diagnostics.push(format!(
            "detected {duplicate_count} duplicate/overlapping roots"
        ));
    }
    if crossover_count > 0 {
        diagnostics.push(format!("detected {crossover_count} root-order crossovers"));
    }
    if skipped_index_count > 0 {
        diagnostics.push(format!(
            "detected {skipped_index_count} skipped root indices"
        ));
    }
    if !count_matches {
        diagnostics.push(format!(
            "independent count {independently_counted_roots} does not match {isolated_unique_roots} isolated unique roots"
        ));
    }
    Ok(WindowReconciliation {
        roots: ordered,
        isolated_unique_roots,
        independently_counted_roots,
        duplicate_count,
        crossover_count,
        skipped_index_count,
        completeness,
        diagnostics,
    })
}

/// Compare only after reference-free discovery and ordering.  References are
/// never accepted as seeds by this function and cannot change an enclosure.
pub fn compare_ordered_roots_to_references(
    roots: &mut [CcmRootRecord],
    positive_reference_values: &[f64],
) -> Result<(), WindowError> {
    if positive_reference_values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(WindowError::InvalidRequest(
            "reference ordinates must be finite and positive".to_owned(),
        ));
    }
    for root in roots {
        let Some(index) = root.positive_index else {
            continue;
        };
        let Some(reference) = positive_reference_values.get(index - 1) else {
            continue;
        };
        let midpoint = parse_finite(&root.midpoint, "root midpoint")?;
        let error = (midpoint - reference).abs();
        root.reference_comparison_digits = Some(if error == 0.0 {
            f64::INFINITY
        } else {
            -error.log10()
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reusable xc-root adapter. The direct discovery routine above remains as an
// independent numerical regression route.
// ---------------------------------------------------------------------------

impl xc_root::RealFunctionF64 for SecularFunctionF64 {
    fn evaluate(&self, x: f64) -> Result<f64, xc_root::RootError> {
        SecularFunctionF64::evaluate(self, x)
            .map_err(|error| xc_root::RootError::Evaluation(error.to_string()))
    }

    fn derivative(&self, x: f64) -> Result<f64, xc_root::RootError> {
        SecularFunctionF64::derivative(self, x)
            .map_err(|error| xc_root::RootError::Evaluation(error.to_string()))
    }
}

impl xc_root::MeromorphicFunctionF64 for SecularFunctionF64 {
    fn real_poles(&self) -> &[f64] {
        self.poles()
    }
}

pub fn discover_roots_with_xc_root_f64(
    secular: &SecularFunctionF64,
    lower: f64,
    upper: f64,
    options: &DiscoveryOptionsF64,
) -> Result<Vec<CcmRootRecord>, WindowError> {
    let generic_options = xc_root::PoleAwareDiscoveryOptionsF64 {
        subdivisions_per_interval: options.subdivisions_per_pole_interval,
        pole_margin_fraction: options.pole_margin_fraction,
        stopping: xc_root::RootStoppingF64 {
            absolute_x_tolerance: options.zero_tolerance,
            relative_x_tolerance: options.zero_tolerance,
            residual_tolerance: options.zero_tolerance,
            maximum_iterations: options.bisection_iterations,
        },
        duplicate_tolerance: 10.0 * options.zero_tolerance,
    };
    let generic =
        xc_root::discover_pole_aware_sign_changes_f64(secular, lower, upper, &generic_options)
            .map_err(|error| WindowError::EvaluationFailed(error.to_string()))?;

    let mut boundaries = vec![lower];
    boundaries.extend(
        secular
            .poles()
            .iter()
            .copied()
            .filter(|pole| *pole > lower && *pole < upper),
    );
    boundaries.push(upper);

    let mut records = Vec::with_capacity(generic.len());
    for root in generic {
        let neighboring = boundaries
            .windows(2)
            .find(|window| root.midpoint > window[0] && root.midpoint < window[1]);
        records.push(CcmRootRecord {
            positive_index: None,
            midpoint: format!("{:.17e}", root.midpoint),
            enclosure: Some(root.decimal_interval()),
            residual_bound: Some(format!("{:.17e}", root.residual)),
            derivative_magnitude: root
                .derivative_magnitude
                .map(|value| format!("{:.17e}", value)),
            conditioning: root
                .derivative_magnitude
                .filter(|value| *value != 0.0)
                .map(|value| format!("{:.17e}", value.abs().recip())),
            isolation_distance: Some(format!(
                "{:.17e}",
                (root.bracket.upper - root.bracket.lower).abs()
            )),
            nearest_left_pole: neighboring.map(|window| format!("{:.17e}", window[0])),
            nearest_right_pole: neighboring.map(|window| format!("{:.17e}", window[1])),
            precision_bits: 53,
            precision_history_bits: vec![53],
            certified_digits: None,
            status: RootStatus::Discovered,
            discovery_method: "xc_root_pole_aware_sign_change_bisection_f64".to_owned(),
            crosscheck_method: Some("direct_ccm_pole_aware_sign_change_bisection_f64".to_owned()),
            reference_comparison_digits: None,
        });
    }
    records.sort_by(|left, right| {
        let left = left.midpoint.parse::<f64>().unwrap_or(f64::NAN);
        let right = right.midpoint.parse::<f64>().unwrap_or(f64::NAN);
        left.total_cmp(&right)
    });
    for (index, root) in records
        .iter_mut()
        .filter(|root| root.midpoint.parse::<f64>().is_ok_and(|value| value > 0.0))
        .enumerate()
    {
        root.positive_index = Some(index + 1);
    }
    Ok(records)
}

#[cfg(test)]
mod xc_root_adapter_tests {
    use super::*;

    #[test]
    fn reusable_and_direct_discovery_agree_on_fixture() {
        let secular = SecularFunctionF64::new(vec![-1.0, 1.0], vec![1.0, 1.0]).unwrap();
        let options = DiscoveryOptionsF64::default();
        let direct = discover_roots_f64(&secular, -0.9, 0.9, &options).unwrap();
        let reusable = discover_roots_with_xc_root_f64(&secular, -0.9, 0.9, &options).unwrap();
        assert_eq!(direct.len(), reusable.len());
        let left = direct[0].midpoint.parse::<f64>().unwrap();
        let right = reusable[0].midpoint.parse::<f64>().unwrap();
        assert!((left - right).abs() < 1e-12);
    }
}
