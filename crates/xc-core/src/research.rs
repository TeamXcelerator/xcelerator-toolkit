// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Domain-independent research result and evidence contracts.
//!
//! A numerical value is not sufficient evidence by itself. These records keep
//! the achieved assurance, diagnostics, artifacts, and provenance attached to
//! the value through serialization and publication workflows.

use crate::{
    evaluate_assurance, AssuranceEvidence, AssuranceLevel, CompletionStatus, ResultStatus,
    SolverProvenance,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub kind: String,
    pub identifier: String,
    pub digest: Option<String>,
    pub description: String,
}

impl EvidenceRef {
    pub fn new(
        kind: impl Into<String>,
        identifier: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            kind: kind.into(),
            identifier: identifier.into(),
            digest: None,
            description: description.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub kind: String,
    pub logical_key: String,
    pub semantic_digest: String,
    pub payload_digest: String,
    /// Completion of the computation which produced this artifact.
    pub completion: CompletionStatus,
    /// Mathematical assurance actually established for this artifact.
    pub assurance: Option<AssuranceLevel>,
    /// Cache lifecycle state, independent from completion and assurance.
    pub disposition: String,
    /// Local or remote locations known to contain the artifact.
    pub locations: Vec<String>,
    /// Publication state keyed by target repository or channel.
    pub publication_states: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Diagnostics {
    pub scalars: BTreeMap<String, String>,
    pub counters: BTreeMap<String, u64>,
    pub flags: BTreeMap<String, bool>,
    pub notes: Vec<String>,
}

impl Diagnostics {
    pub fn insert_scalar(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.scalars.insert(name.into(), value.into());
    }

    pub fn insert_counter(&mut self, name: impl Into<String>, value: u64) {
        self.counters.insert(name.into(), value);
    }

    pub fn insert_flag(&mut self, name: impl Into<String>, value: bool) {
        self.flags.insert(name.into(), value);
    }
}

/// Identity of terminal completion and assurance consistency semantics.
pub const RESEARCH_RESULT_SEMANTICS: &str = "research-result-terminal-completion-v2";

fn legacy_research_result_semantics() -> String {
    "research-result-unversioned-v1".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResearchResult<T> {
    /// Producer semantics; absent historical markers retain their legacy identity.
    #[serde(default = "legacy_research_result_semantics")]
    pub semantics: String,
    pub value: Option<T>,
    pub completion: CompletionStatus,
    pub status: ResultStatus,
    pub requested_assurance: AssuranceLevel,
    pub achieved_assurance: Option<AssuranceLevel>,
    pub completed_assurance_checks: Vec<String>,
    pub missing_assurance_checks: Vec<String>,
    pub diagnostics: Diagnostics,
    pub evidence: Vec<EvidenceRef>,
    pub artifacts: Vec<ArtifactRef>,
    pub provenance: SolverProvenance,
}

impl<T> ResearchResult<T> {
    pub fn computed(value: T, provenance: SolverProvenance) -> Self {
        Self::for_request(value, AssuranceLevel::Computed, provenance)
    }

    pub fn for_request(
        value: T,
        requested_assurance: AssuranceLevel,
        provenance: SolverProvenance,
    ) -> Self {
        Self {
            semantics: RESEARCH_RESULT_SEMANTICS.to_owned(),
            value: Some(value),
            completion: CompletionStatus::Successful,
            status: ResultStatus::Converged,
            requested_assurance,
            achieved_assurance: Some(AssuranceLevel::Computed),
            completed_assurance_checks: vec!["primary computation diagnostics accepted".to_owned()],
            missing_assurance_checks: Vec::new(),
            diagnostics: Diagnostics::default(),
            evidence: Vec::new(),
            artifacts: Vec::new(),
            provenance,
        }
    }

    pub fn without_value(
        completion: CompletionStatus,
        status: ResultStatus,
        requested_assurance: AssuranceLevel,
        provenance: SolverProvenance,
    ) -> Self {
        Self {
            semantics: RESEARCH_RESULT_SEMANTICS.to_owned(),
            value: None,
            completion,
            status,
            requested_assurance,
            achieved_assurance: None,
            completed_assurance_checks: Vec::new(),
            missing_assurance_checks: Vec::new(),
            diagnostics: Diagnostics::default(),
            evidence: Vec::new(),
            artifacts: Vec::new(),
            provenance,
        }
    }

    pub fn with_status(mut self, status: ResultStatus) -> Self {
        self.semantics = RESEARCH_RESULT_SEMANTICS.to_owned();
        if matches!(
            self.completion,
            CompletionStatus::Failed | CompletionStatus::Cancelled
        ) {
            self.achieved_assurance = None;
        } else if matches!(
            status,
            ResultStatus::Failed | ResultStatus::InvalidConfiguration
        ) {
            self.completion = CompletionStatus::Failed;
            self.achieved_assurance = None;
        } else if status != ResultStatus::Converged {
            self.completion = CompletionStatus::Inconclusive;
            self.achieved_assurance = self
                .achieved_assurance
                .map(|level| level.min(AssuranceLevel::Computed));
        }
        if status != ResultStatus::Converged {
            self.completed_assurance_checks.retain(|check| {
                !check.starts_with("portable certificate verified for ")
                    && check != "independent route comparison accepted"
            });
            if !self
                .missing_assurance_checks
                .iter()
                .any(|check| check == "successful completion")
            {
                self.missing_assurance_checks
                    .push("successful completion".to_owned());
            }
        }
        self.status = status;
        self
    }

    /// Derive achieved assurance and its audit trail from completed evidence.
    /// The supplied evidence must come from the caller's completed checks;
    /// this method does not independently replay the underlying mathematics.
    pub fn with_assurance_evidence(mut self, evidence: &AssuranceEvidence) -> Self {
        self.semantics = RESEARCH_RESULT_SEMANTICS.to_owned();
        let evaluation = evaluate_assurance(
            self.requested_assurance,
            self.completion == CompletionStatus::Successful
                && self.status == ResultStatus::Converged,
            evidence,
        );
        self.achieved_assurance = if matches!(
            self.completion,
            CompletionStatus::Failed | CompletionStatus::Cancelled
        ) || matches!(
            self.status,
            ResultStatus::Failed | ResultStatus::InvalidConfiguration
        ) {
            None
        } else {
            self.value.as_ref().and(evaluation.achieved)
        };
        self.completed_assurance_checks = evaluation.completed_checks;
        self.missing_assurance_checks = evaluation.missing_checks;
        self
    }

    pub fn add_evidence(&mut self, evidence: EvidenceRef) {
        self.evidence.push(evidence);
    }

    pub fn add_artifact(&mut self, artifact: ArtifactRef) {
        self.artifacts.push(artifact);
    }
}

impl<T: Serialize> ResearchResult<T> {
    /// Checks internal assurance consistency; caller attestations are not proof replay.
    pub fn validate_assurance_consistency(&self) -> Result<(), crate::ConfigError> {
        let invalid = (self.value.is_none() && self.achieved_assurance.is_some())
            || (self.achieved_assurance.is_some()
                && (matches!(
                    self.completion,
                    CompletionStatus::Failed | CompletionStatus::Cancelled
                ) || self
                    .missing_assurance_checks
                    .iter()
                    .any(|check| check == "valid primary computation")
                    || !self
                        .completed_assurance_checks
                        .iter()
                        .any(|check| check == "primary computation diagnostics accepted")))
            || (matches!(
                self.status,
                ResultStatus::Failed | ResultStatus::InvalidConfiguration
            ) && (self.completion == CompletionStatus::Successful
                || self.achieved_assurance.is_some()))
            || (self
                .achieved_assurance
                .is_some_and(|level| level > AssuranceLevel::Computed)
                && (self.completion != CompletionStatus::Successful
                    || self.status != ResultStatus::Converged));
        if invalid {
            return Err(crate::ConfigError::new(
                "research result assurance contradicts value, status, or completion",
            ));
        }
        let accepted = match self.achieved_assurance {
            Some(AssuranceLevel::Certified) => {
                self.completed_assurance_checks.iter().any(|check| {
                    check
                        .strip_prefix("portable certificate verified for ")
                        .is_some_and(|scope| !scope.trim().is_empty())
                }) && !self.missing_assurance_checks.iter().any(|check| {
                    check.contains("certificate")
                        || check.contains("decisive approximation")
                        || check == "successful completion"
                })
            }
            Some(AssuranceLevel::CrossChecked) => {
                self.completed_assurance_checks
                    .iter()
                    .any(|check| check == "independent route comparison accepted")
                    && !self.missing_assurance_checks.iter().any(|check| {
                        check.contains("independent route comparison")
                            || check == "successful completion"
                    })
            }
            _ => true,
        };
        if !accepted {
            return Err(crate::ConfigError::new(
                "research result assurance lacks consistent completed checks",
            ));
        }
        Ok(())
    }

    /// Checks consistency and secret-free persistence, not the truth of attestations.
    pub fn validate_for_persistence(&self) -> Result<(), crate::ConfigError> {
        self.validate_assurance_consistency()?;
        crate::validate_secret_free(self, "research result")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_round_trip_preserves_evidence() {
        let mut result = ResearchResult::computed(
            "1.2345e-1000".to_owned(),
            SolverProvenance::current_package("rug_mpfr"),
        );
        result.add_evidence(EvidenceRef::new(
            "residual",
            "residual.json",
            "high-precision residual record",
        ));
        result.diagnostics.insert_flag("cross_checked", true);
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: ResearchResult<String> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, result);
        result.validate_for_persistence().unwrap();
    }

    #[test]
    fn persistence_rejects_credentials_in_diagnostics_without_echoing_them() {
        let mut result = ResearchResult::computed(
            "value".to_owned(),
            SolverProvenance::current_package("test"),
        );
        result.diagnostics.insert_scalar("password", "do-not-echo");
        let error = result.validate_for_persistence().unwrap_err().to_string();
        assert!(error.contains("credential-bearing field"));
        assert!(!error.contains("do-not-echo"));
    }

    #[test]
    fn failed_result_has_no_value_or_achieved_assurance() {
        let result = ResearchResult::<String>::without_value(
            CompletionStatus::Failed,
            ResultStatus::InsufficientPrecision,
            AssuranceLevel::Certified,
            SolverProvenance::current_package("rug_mpfr"),
        );
        assert!(result.value.is_none());
        assert_eq!(result.achieved_assurance, None);
        assert_eq!(result.requested_assurance, AssuranceLevel::Certified);
    }

    #[test]
    fn inconclusive_certified_request_serializes_derived_lower_assurance_and_check_lists() {
        let result = ResearchResult::<String>::without_value(
            CompletionStatus::Inconclusive,
            ResultStatus::InsufficientPrecision,
            AssuranceLevel::Certified,
            SolverProvenance::current_package("rug_mpfr"),
        )
        .with_assurance_evidence(&AssuranceEvidence {
            computation_valid: true,
            ..AssuranceEvidence::default()
        });
        assert_eq!(result.requested_assurance, AssuranceLevel::Certified);
        assert_eq!(result.achieved_assurance, None);
        assert!(result
            .completed_assurance_checks
            .iter()
            .any(|check| check.contains("primary computation")));
        assert!(result
            .missing_assurance_checks
            .iter()
            .any(|check| check.contains("successful completion")));

        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: ResearchResult<String> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, result);
        assert!(!encoded.contains("\"achieved_assurance\":\"certified\""));
    }
    #[test]
    fn contradictory_saved_assurance_and_failed_status_are_rejected() {
        let mut result = ResearchResult::computed(1, SolverProvenance::current_package("f64"));
        result.achieved_assurance = Some(AssuranceLevel::Certified);
        assert!(result.validate_for_persistence().is_err());
        result
            .completed_assurance_checks
            .push("portable certificate verified for stored source".into());
        assert!(result.validate_for_persistence().is_ok());
        result.value = None;
        assert!(result.validate_for_persistence().is_err());
        let failed = ResearchResult::computed(1, SolverProvenance::current_package("f64"))
            .with_status(ResultStatus::Failed);
        assert_eq!(failed.achieved_assurance, None);
        assert_eq!(failed.completion, CompletionStatus::Failed);
        failed.validate_for_persistence().unwrap();
        let failed = failed.with_assurance_evidence(&AssuranceEvidence {
            computation_valid: true,
            ..AssuranceEvidence::default()
        });
        assert_eq!(failed.achieved_assurance, None);
        failed.validate_for_persistence().unwrap();
    }

    #[test]
    fn missing_approximation_ledger_is_not_an_empty_ledger() {
        let mut value = serde_json::to_value(AssuranceEvidence::default()).unwrap();
        value.as_object_mut().unwrap().remove("approximations");
        assert!(serde_json::from_value::<AssuranceEvidence>(value).is_err());
    }
}
