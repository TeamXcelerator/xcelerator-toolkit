use crate::{AssuranceLevel, ConfigError, DecimalLiteral};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticAssumptionKind {
    TheoremDomain,
    Model,
    ExternalData,
    AsymptoticEstimate,
    NumericalRegularity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticAssumption {
    pub assumption_id: String,
    pub kind: AnalyticAssumptionKind,
    pub statement: String,
    pub scope: String,
    pub evidence_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticErrorTerm {
    pub term_id: String,
    pub source: String,
    pub affects: String,
    pub decisive_for_claim: bool,
    pub rigorous: bool,
    pub absolute_bound: Option<DecimalLiteral>,
    pub proof_method: Option<String>,
    pub evidence_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticErrorBudget {
    pub quantity: String,
    pub terms: Vec<AnalyticErrorTerm>,
    pub total_absolute_bound: Option<DecimalLiteral>,
    pub aggregation_method: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticProblemContext {
    pub schema_version: u32,
    pub domain: String,
    pub requested_assurance: AssuranceLevel,
    pub assumptions: Vec<AnalyticAssumption>,
    pub error_budget: AnalyticErrorBudget,
}

impl AnalyticProblemContext {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != 1
            || self.domain.trim().is_empty()
            || self.error_budget.quantity.trim().is_empty()
            || self.error_budget.aggregation_method.trim().is_empty()
        {
            return Err(ConfigError::new(
                "analytic problem context identity and error-budget scope must be nonempty",
            ));
        }
        let mut assumption_ids = BTreeSet::new();
        for assumption in &self.assumptions {
            let mut evidence_ids = BTreeSet::new();
            if assumption.assumption_id.trim().is_empty()
                || assumption.statement.trim().is_empty()
                || assumption.scope.trim().is_empty()
                || !assumption_ids.insert(&assumption.assumption_id)
                || assumption
                    .evidence_ids
                    .iter()
                    .any(|value| value.trim().is_empty() || !evidence_ids.insert(value))
            {
                return Err(ConfigError::new(
                    "analytic assumptions require unique ids, statements, scopes, and unique nonempty evidence ids",
                ));
            }
        }
        if self
            .error_budget
            .total_absolute_bound
            .as_ref()
            .is_some_and(invalid_bound)
        {
            return Err(ConfigError::new(
                "analytic error-budget bounds must be nonnegative decimal literals",
            ));
        }
        let mut term_ids = BTreeSet::new();
        for term in &self.error_budget.terms {
            if term.term_id.trim().is_empty()
                || term.source.trim().is_empty()
                || term.affects.trim().is_empty()
                || !term_ids.insert(&term.term_id)
            {
                return Err(ConfigError::new(
                    "analytic error terms require unique ids, sources, and affected quantities",
                ));
            }
            if term.absolute_bound.as_ref().is_some_and(invalid_bound) {
                return Err(ConfigError::new(
                    "analytic error-term bounds must be nonnegative decimal literals",
                ));
            }
            if term.rigorous {
                if term.absolute_bound.is_none()
                    || term
                        .proof_method
                        .as_ref()
                        .is_none_or(|value| value.trim().is_empty())
                    || term
                        .evidence_sha256
                        .as_ref()
                        .is_none_or(|value| !lower_sha256(value))
                {
                    return Err(ConfigError::new(
                        "rigorous analytic error terms require a bound, proof method, and evidence SHA-256",
                    ));
                }
            } else if term.proof_method.is_some() || term.evidence_sha256.is_some() {
                return Err(ConfigError::new(
                    "nonrigorous analytic error terms must not carry proof evidence",
                ));
            }
        }
        if self.requested_assurance == AssuranceLevel::Certified
            && (self.error_budget.total_absolute_bound.is_none()
                || self
                    .error_budget
                    .terms
                    .iter()
                    .any(|term| term.decisive_for_claim && !term.rigorous))
        {
            return Err(ConfigError::new(
                "Certified analytic context requires a total bound and rigorous decisive terms",
            ));
        }
        Ok(())
    }

    pub fn canonical_sha256(&self) -> Result<String, ConfigError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| {
            ConfigError::new(format!("analytic context serialization failed: {error}"))
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

fn lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_bound(value: &DecimalLiteral) -> bool {
    value.validate().is_err() || value.as_str().starts_with('-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(assurance: AssuranceLevel, rigorous: bool) -> AnalyticProblemContext {
        AnalyticProblemContext {
            schema_version: 1,
            domain: "finite-example".to_owned(),
            requested_assurance: assurance,
            assumptions: vec![AnalyticAssumption {
                assumption_id: "domain".to_owned(),
                kind: AnalyticAssumptionKind::TheoremDomain,
                statement: "the parameter lies in the theorem domain".to_owned(),
                scope: "finite claim".to_owned(),
                evidence_ids: vec!["theorem:1".to_owned()],
            }],
            error_budget: AnalyticErrorBudget {
                quantity: "claim value".to_owned(),
                terms: vec![AnalyticErrorTerm {
                    term_id: "tail".to_owned(),
                    source: "truncated analytic tail".to_owned(),
                    affects: "claim value".to_owned(),
                    decisive_for_claim: true,
                    rigorous,
                    absolute_bound: rigorous.then(|| DecimalLiteral::new("1e-30").unwrap()),
                    proof_method: rigorous.then(|| "theorem tail inequality".to_owned()),
                    evidence_sha256: rigorous.then(|| "a".repeat(64)),
                }],
                total_absolute_bound: rigorous.then(|| DecimalLiteral::new("1e-30").unwrap()),
                aggregation_method: "triangle inequality".to_owned(),
            },
        }
    }

    #[test]
    fn analytic_problem_context_round_trips_and_has_stable_identity() {
        let original = context(AssuranceLevel::Certified, true);
        let digest = original.canonical_sha256().unwrap();
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: AnalyticProblemContext = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.canonical_sha256().unwrap(), digest);
    }

    #[test]
    fn certified_problem_rejects_an_unbounded_decisive_analytic_error() {
        let certified = context(AssuranceLevel::Certified, false);
        assert!(certified.validate().is_err());
        let exploratory = context(AssuranceLevel::Computed, false);
        exploratory.validate().unwrap();
    }

    #[test]
    fn analytic_problem_rejects_negative_bounds_and_repeated_evidence() {
        let mut negative = context(AssuranceLevel::Certified, true);
        negative.error_budget.terms[0].absolute_bound =
            Some(DecimalLiteral::new("-1e-30").unwrap());
        assert!(negative.validate().is_err());

        let mut duplicate = context(AssuranceLevel::Certified, true);
        duplicate.assumptions[0]
            .evidence_ids
            .push("theorem:1".to_owned());
        assert!(duplicate.validate().is_err());
    }
}
