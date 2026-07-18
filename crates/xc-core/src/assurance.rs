//! Requested and achieved assurance, plus route-independence evaluation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApproximationKind {
    Banding,
    Compression,
    RandomizedSketch,
    MixedPrecision,
    Truncation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigorousApproximationBound {
    pub absolute_error_bound: String,
    pub proof_method: String,
    pub evidence_sha256: String,
}

impl RigorousApproximationBound {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        crate::DecimalLiteral::new(&self.absolute_error_bound)?;
        if self.absolute_error_bound.trim_start().starts_with('-')
            || self.proof_method.trim().is_empty()
            || !lower_sha256(&self.evidence_sha256)
        {
            return Err(crate::ConfigError::new(
                "rigorous approximation bound requires a nonnegative decimal, proof method, and evidence SHA-256",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproximationEvidence {
    pub kind: ApproximationKind,
    pub purpose: String,
    /// False only when exact final verification proves this approximation did
    /// not participate in the accepted result.
    pub decisive_for_accepted_result: bool,
    pub rigorous_bound: Option<RigorousApproximationBound>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproximationLedger {
    pub entries: Vec<ApproximationEvidence>,
}

impl ApproximationLedger {
    pub fn validate_for_assurance(
        &self,
        assurance: AssuranceLevel,
    ) -> Result<(), crate::ConfigError> {
        for entry in &self.entries {
            if entry.purpose.trim().is_empty() {
                return Err(crate::ConfigError::new(
                    "approximation evidence purpose must be nonempty",
                ));
            }
            if let Some(bound) = &entry.rigorous_bound {
                bound.validate()?;
            }
            if assurance == AssuranceLevel::Certified
                && entry.decisive_for_accepted_result
                && entry.rigorous_bound.is_none()
            {
                return Err(crate::ConfigError::new(format!(
                    "Certified assurance requires a rigorous bound for decisive {:?} approximation",
                    entry.kind
                )));
            }
        }
        Ok(())
    }
}

fn lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Strength of the mathematical evidence attached to a result.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum AssuranceLevel {
    /// One explicitly identified numerical route with diagnostics.
    #[default]
    Computed,
    /// Two algorithmically or formulation-independent routes agree.
    CrossChecked,
    /// Portable exact or interval evidence establishes the stated claim.
    Certified,
}

/// Reproducibility expectation for reductions and output ordering.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reproducibility {
    /// Deterministic mathematical output and stable ordering.
    #[default]
    Deterministic,
    /// Bit-identical serialized output for the same complete fingerprint.
    Bitwise,
    /// Explicitly exploratory. Not admissible for certified acceptance.
    Exploratory,
}

/// Evidence-bearing identity of one solution route.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RouteEvidence {
    pub route_id: String,
    pub algorithm_family: String,
    pub formulation: String,
    pub implementation_id: String,
    pub decisive_intermediates: BTreeSet<String>,
    pub precision_bits: Option<u32>,
    pub seed: Option<u64>,
    pub thread_count: Option<usize>,
    pub evidence_digest: Option<String>,
}

/// Human-reviewed declaration for the intended comparison claim.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IndependenceDeclaration {
    pub intended_claim: String,
    pub rationale: String,
    pub accepted_shared_inputs: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IndependenceAssessment {
    pub independent: bool,
    pub intended_claim: String,
    pub reasons: Vec<String>,
    pub stability_evidence: Vec<String>,
    pub shared_decisive_intermediates: Vec<String>,
}

/// Evaluate whether two routes are independent for one declared claim.
pub fn assess_route_independence(
    primary: &RouteEvidence,
    secondary: &RouteEvidence,
    declaration: &IndependenceDeclaration,
) -> IndependenceAssessment {
    let mut reasons = Vec::new();
    let mut stability_evidence = Vec::new();

    if declaration.intended_claim.trim().is_empty() {
        reasons.push("independence declaration has no intended claim".to_owned());
    }
    if declaration.rationale.trim().is_empty() {
        reasons.push("independence declaration has no rationale".to_owned());
    }
    if primary.route_id.trim().is_empty() || secondary.route_id.trim().is_empty() {
        reasons.push("both route identifiers must be nonempty".to_owned());
    }
    if primary.route_id == secondary.route_id {
        reasons.push("both evidence records identify the same route".to_owned());
    }

    let same_algorithm = primary.algorithm_family == secondary.algorithm_family;
    let same_formulation = primary.formulation == secondary.formulation;
    if same_algorithm && same_formulation {
        reasons.push(
            "routes share the same decisive algorithm family and mathematical formulation"
                .to_owned(),
        );
    }
    if primary.implementation_id == secondary.implementation_id {
        reasons.push("routes share the same implementation identity".to_owned());
    }

    let shared_decisive_intermediates: Vec<_> = primary
        .decisive_intermediates
        .intersection(&secondary.decisive_intermediates)
        .cloned()
        .collect();
    if !shared_decisive_intermediates.is_empty() {
        reasons.push(format!(
            "routes share decisive intermediate results: {}",
            shared_decisive_intermediates.join(", ")
        ));
    }

    if primary.precision_bits != secondary.precision_bits {
        stability_evidence.push("precision changed between route executions".to_owned());
    }
    if primary.seed != secondary.seed {
        stability_evidence.push("seed changed between route executions".to_owned());
    }
    if primary.thread_count != secondary.thread_count {
        stability_evidence.push("thread count changed between route executions".to_owned());
    }
    if same_algorithm && same_formulation && !stability_evidence.is_empty() {
        reasons.push(
            "precision, seed, or thread changes are stability evidence, not an independent route"
                .to_owned(),
        );
    }

    IndependenceAssessment {
        independent: reasons.is_empty(),
        intended_claim: declaration.intended_claim.clone(),
        reasons,
        stability_evidence,
        shared_decisive_intermediates,
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssuranceEvidence {
    pub computation_valid: bool,
    pub stability_checks: Vec<String>,
    pub independence: Option<IndependenceAssessment>,
    pub certificate_verified: bool,
    pub certificate_claim_scope: Option<String>,
    #[serde(default)]
    pub approximations: ApproximationLedger,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssuranceEvaluation {
    pub requested: AssuranceLevel,
    pub achieved: Option<AssuranceLevel>,
    pub completed_checks: Vec<String>,
    pub missing_checks: Vec<String>,
}

/// Derive achieved assurance from completed evidence rather than the request.
pub fn evaluate_assurance(
    requested: AssuranceLevel,
    completion_successful: bool,
    evidence: &AssuranceEvidence,
) -> AssuranceEvaluation {
    let mut completed_checks = evidence.stability_checks.clone();
    let mut missing_checks = Vec::new();
    let mut achieved = evidence
        .computation_valid
        .then_some(AssuranceLevel::Computed);

    if evidence.computation_valid {
        completed_checks.push("primary computation diagnostics accepted".to_owned());
    } else {
        missing_checks.push("valid primary computation".to_owned());
    }

    let independent_routes_accepted = evidence
        .independence
        .as_ref()
        .is_some_and(|assessment| assessment.independent);
    let approximation_evidence_accepted = evidence
        .approximations
        .validate_for_assurance(AssuranceLevel::Certified)
        .is_ok();
    let certificate_accepted = evidence.certificate_verified
        && evidence
            .certificate_claim_scope
            .as_ref()
            .is_some_and(|scope| !scope.trim().is_empty())
        && approximation_evidence_accepted;

    if completion_successful && evidence.computation_valid {
        if independent_routes_accepted {
            completed_checks.push("independent route comparison accepted".to_owned());
            achieved = Some(AssuranceLevel::CrossChecked);
        }

        if certificate_accepted {
            completed_checks.push(format!(
                "portable certificate verified for {}",
                evidence
                    .certificate_claim_scope
                    .as_deref()
                    .unwrap_or_default()
            ));
            achieved = Some(AssuranceLevel::Certified);
        }
    }

    if requested == AssuranceLevel::CrossChecked && !independent_routes_accepted {
        missing_checks.push("accepted independent route comparison".to_owned());
    }
    if requested == AssuranceLevel::Certified && !certificate_accepted {
        missing_checks.push("verified certificate with explicit claim scope".to_owned());
    }
    if requested == AssuranceLevel::Certified && !approximation_evidence_accepted {
        missing_checks.push("rigorous bounds for every decisive approximation".to_owned());
    }
    if !completion_successful {
        missing_checks.push("successful completion".to_owned());
    }

    AssuranceEvaluation {
        requested,
        achieved,
        completed_checks,
        missing_checks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(id: &str, family: &str, formulation: &str) -> RouteEvidence {
        RouteEvidence {
            route_id: id.to_owned(),
            algorithm_family: family.to_owned(),
            formulation: formulation.to_owned(),
            implementation_id: id.to_owned(),
            decisive_intermediates: BTreeSet::new(),
            precision_bits: Some(256),
            seed: None,
            thread_count: Some(1),
            evidence_digest: None,
        }
    }

    fn declaration() -> IndependenceDeclaration {
        IndependenceDeclaration {
            intended_claim: "selected eigenvalue".to_owned(),
            rationale: "dense reduction and Sturm count use distinct decisive algorithms"
                .to_owned(),
            accepted_shared_inputs: ["canonical operator".to_owned()].into_iter().collect(),
        }
    }

    #[test]
    fn higher_precision_repeat_is_not_independent() {
        let primary = route("solve-p256", "dense_qr", "matrix_eigensolve");
        let mut secondary = route("solve-p512", "dense_qr", "matrix_eigensolve");
        secondary.precision_bits = Some(512);
        let assessment = assess_route_independence(&primary, &secondary, &declaration());
        assert!(!assessment.independent);
        assert_eq!(assessment.stability_evidence.len(), 1);
    }

    #[test]
    fn distinct_algorithmic_routes_are_independent() {
        let primary = route("dense", "householder_qr", "matrix_eigensolve");
        let secondary = route("count", "sturm_count", "threshold_count");
        let assessment = assess_route_independence(&primary, &secondary, &declaration());
        assert!(assessment.independent, "{:?}", assessment.reasons);
    }

    #[test]
    fn shared_decisive_intermediate_rejects_independence() {
        let mut primary = route("secular", "interval_newton", "secular_function");
        let mut secondary = route("operator", "dense_qr", "rank_one_operator");
        primary
            .decisive_intermediates
            .insert("precomputed-root-list".to_owned());
        secondary
            .decisive_intermediates
            .insert("precomputed-root-list".to_owned());
        let assessment = assess_route_independence(&primary, &secondary, &declaration());
        assert!(!assessment.independent);
        assert_eq!(
            assessment.shared_decisive_intermediates,
            vec!["precomputed-root-list"]
        );
    }

    #[test]
    fn requested_certified_does_not_copy_into_achieved() {
        let evaluation = evaluate_assurance(
            AssuranceLevel::Certified,
            true,
            &AssuranceEvidence {
                computation_valid: true,
                ..AssuranceEvidence::default()
            },
        );
        assert_eq!(evaluation.achieved, Some(AssuranceLevel::Computed));
        assert!(evaluation
            .missing_checks
            .iter()
            .any(|check| check.contains("certificate")));
    }

    #[test]
    fn invalid_primary_cannot_be_promoted_by_certificate_or_independence_flags() {
        let evaluation = evaluate_assurance(
            AssuranceLevel::Certified,
            true,
            &AssuranceEvidence {
                computation_valid: false,
                independence: Some(IndependenceAssessment {
                    independent: true,
                    intended_claim: "eigenvalue".to_owned(),
                    reasons: Vec::new(),
                    stability_evidence: Vec::new(),
                    shared_decisive_intermediates: Vec::new(),
                }),
                certificate_verified: true,
                certificate_claim_scope: Some("eigenvalue enclosure".to_owned()),
                ..AssuranceEvidence::default()
            },
        );
        assert_eq!(evaluation.achieved, None);
        assert!(evaluation
            .missing_checks
            .iter()
            .any(|check| check.contains("primary computation")));
    }

    #[test]
    fn every_decisive_approximation_kind_requires_a_rigorous_bound_for_certified() {
        for kind in [
            ApproximationKind::Banding,
            ApproximationKind::Compression,
            ApproximationKind::RandomizedSketch,
            ApproximationKind::MixedPrecision,
            ApproximationKind::Truncation,
        ] {
            let evidence = AssuranceEvidence {
                computation_valid: true,
                certificate_verified: true,
                certificate_claim_scope: Some("finite matrix claim".to_owned()),
                approximations: ApproximationLedger {
                    entries: vec![ApproximationEvidence {
                        kind,
                        purpose: "accelerate decisive finite computation".to_owned(),
                        decisive_for_accepted_result: true,
                        rigorous_bound: None,
                    }],
                },
                ..AssuranceEvidence::default()
            };
            let evaluation = evaluate_assurance(AssuranceLevel::Certified, true, &evidence);
            assert_ne!(evaluation.achieved, Some(AssuranceLevel::Certified));
            assert!(evaluation
                .missing_checks
                .iter()
                .any(|check| check.contains("rigorous bounds")));
        }
    }

    #[test]
    fn bounded_decisive_and_nondecisive_seed_approximations_are_admissible() {
        let evidence = AssuranceEvidence {
            computation_valid: true,
            certificate_verified: true,
            certificate_claim_scope: Some("finite matrix claim".to_owned()),
            approximations: ApproximationLedger {
                entries: vec![
                    ApproximationEvidence {
                        kind: ApproximationKind::Truncation,
                        purpose: "finite tail enclosure".to_owned(),
                        decisive_for_accepted_result: true,
                        rigorous_bound: Some(RigorousApproximationBound {
                            absolute_error_bound: "1e-80".to_owned(),
                            proof_method: "interval tail majorant".to_owned(),
                            evidence_sha256: "b".repeat(64),
                        }),
                    },
                    ApproximationEvidence {
                        kind: ApproximationKind::RandomizedSketch,
                        purpose: "nondecisive warm start discarded before exact verification"
                            .to_owned(),
                        decisive_for_accepted_result: false,
                        rigorous_bound: None,
                    },
                ],
            },
            ..AssuranceEvidence::default()
        };
        let evaluation = evaluate_assurance(AssuranceLevel::Certified, true, &evidence);
        assert_eq!(evaluation.achieved, Some(AssuranceLevel::Certified));
    }
}
