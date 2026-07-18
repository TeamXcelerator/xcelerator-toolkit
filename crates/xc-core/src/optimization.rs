use crate::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Evidence for one prerequisite that must precede accepting an optimized
/// backend as a default route.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationMilestoneGate {
    pub milestone_id: String,
    pub passed: bool,
    pub evidence_ids: Vec<String>,
}

impl OptimizationMilestoneGate {
    fn validate(&self, label: &str) -> Result<(), ConfigError> {
        if self.milestone_id.trim().is_empty() {
            return Err(ConfigError::new(format!(
                "{label} optimization milestone id must be nonempty"
            )));
        }
        if self.passed && self.evidence_ids.is_empty() {
            return Err(ConfigError::new(format!(
                "passed {label} optimization milestone requires evidence"
            )));
        }
        let mut unique = BTreeSet::new();
        for evidence in &self.evidence_ids {
            if evidence.trim().is_empty() || !unique.insert(evidence) {
                return Err(ConfigError::new(format!(
                    "{label} optimization milestone evidence must be nonempty and unique"
                )));
            }
        }
        Ok(())
    }
}

/// Replayable decision controlling whether an optimized route may become the
/// default for its exact mathematical capability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationDefaultDecision {
    pub capability_id: String,
    pub reference_route_id: String,
    pub optimized_route_id: String,
    pub correctness: OptimizationMilestoneGate,
    pub baseline_parity: OptimizationMilestoneGate,
    pub independent_cross_validation: OptimizationMilestoneGate,
    pub certification_required: bool,
    pub certification: OptimizationMilestoneGate,
    pub accepted_as_default: bool,
    pub blocking_gates: Vec<String>,
}

impl OptimizationDefaultDecision {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        capability_id: impl Into<String>,
        reference_route_id: impl Into<String>,
        optimized_route_id: impl Into<String>,
        correctness: OptimizationMilestoneGate,
        baseline_parity: OptimizationMilestoneGate,
        independent_cross_validation: OptimizationMilestoneGate,
        certification_required: bool,
        certification: OptimizationMilestoneGate,
    ) -> Result<Self, ConfigError> {
        let mut decision = Self {
            capability_id: capability_id.into(),
            reference_route_id: reference_route_id.into(),
            optimized_route_id: optimized_route_id.into(),
            correctness,
            baseline_parity,
            independent_cross_validation,
            certification_required,
            certification,
            accepted_as_default: false,
            blocking_gates: Vec::new(),
        };
        decision.validate_inputs()?;
        decision.blocking_gates = decision.expected_blocking_gates();
        decision.accepted_as_default = decision.blocking_gates.is_empty();
        Ok(decision)
    }

    /// Reject a serialized record whose claimed decision no longer follows
    /// from its retained prerequisite evidence.
    pub fn verify(&self) -> Result<(), ConfigError> {
        self.validate_inputs()?;
        let expected = self.expected_blocking_gates();
        if self.blocking_gates != expected || self.accepted_as_default != expected.is_empty() {
            return Err(ConfigError::new(
                "optimized-default decision does not match its milestone evidence",
            ));
        }
        Ok(())
    }

    fn validate_inputs(&self) -> Result<(), ConfigError> {
        if self.capability_id.trim().is_empty()
            || self.reference_route_id.trim().is_empty()
            || self.optimized_route_id.trim().is_empty()
        {
            return Err(ConfigError::new(
                "optimization decision requires nonempty capability and route ids",
            ));
        }
        if self.reference_route_id == self.optimized_route_id {
            return Err(ConfigError::new(
                "optimized and trusted-reference route ids must differ",
            ));
        }
        self.correctness.validate("correctness")?;
        self.baseline_parity.validate("baseline parity")?;
        self.independent_cross_validation
            .validate("independent cross-validation")?;
        self.certification.validate("certification")?;
        Ok(())
    }

    fn expected_blocking_gates(&self) -> Vec<String> {
        let mut blocking = Vec::new();
        if !self.correctness.passed {
            blocking.push("correctness".to_owned());
        }
        if !self.baseline_parity.passed {
            blocking.push("baseline_parity".to_owned());
        }
        if !self.independent_cross_validation.passed {
            blocking.push("independent_cross_validation".to_owned());
        }
        if self.certification_required && !self.certification.passed {
            blocking.push("certification".to_owned());
        }
        blocking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(id: &str, passed: bool) -> OptimizationMilestoneGate {
        OptimizationMilestoneGate {
            milestone_id: id.to_owned(),
            passed,
            evidence_ids: if passed {
                vec![format!("evidence:{id}")]
            } else {
                Vec::new()
            },
        }
    }

    #[test]
    fn optimized_default_requires_every_applicable_milestone() {
        let blocked = OptimizationDefaultDecision::evaluate(
            "symmetric-extremes",
            "dense-reference",
            "matrix-free-optimized",
            gate("correctness-v1", true),
            gate("baseline-current", false),
            gate("independent-crosscheck-v1", true),
            true,
            gate("portable-certificate-v1", false),
        )
        .unwrap();
        assert!(!blocked.accepted_as_default);
        assert_eq!(blocked.blocking_gates, ["baseline_parity", "certification"]);
        blocked.verify().unwrap();

        let accepted = OptimizationDefaultDecision::evaluate(
            "symmetric-extremes",
            "dense-reference",
            "matrix-free-optimized",
            gate("correctness-v1", true),
            gate("baseline-current", true),
            gate("independent-crosscheck-v1", true),
            true,
            gate("portable-certificate-v1", true),
        )
        .unwrap();
        assert!(accepted.accepted_as_default);
        accepted.verify().unwrap();
    }

    #[test]
    fn serialized_optimization_decision_is_replayable_and_fail_closed() {
        let decision = OptimizationDefaultDecision::evaluate(
            "batch-action",
            "sequential",
            "rayon-indexed",
            gate("correctness-v1", true),
            gate("baseline-current", true),
            gate("independent-crosscheck-v1", false),
            false,
            gate("not-applicable", false),
        )
        .unwrap();
        let encoded = serde_json::to_string(&decision).unwrap();
        let mut decoded: OptimizationDefaultDecision = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, decision);
        decoded.accepted_as_default = true;
        assert!(decoded.verify().is_err());
    }
}
