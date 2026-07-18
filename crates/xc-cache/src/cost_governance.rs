//! Explainable storage-cost, quota, retention, and paid-backend governance.

use crate::{canonical_digest, CacheError, ContentDigest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageBackendClass {
    GithubPublic,
    GithubPrivate,
    LocalFilesystem,
    ExportBundle,
    PaidExternal,
}

impl StorageBackendClass {
    fn is_github(self) -> bool {
        matches!(self, Self::GithubPublic | Self::GithubPrivate)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageRetentionClass {
    TransactionStaging,
    ProjectCache,
    DurableCache,
    PublicationArchive,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageCostEstimate {
    pub schema_version: u32,
    pub currency: String,
    pub upfront_cost_micros: u64,
    pub monthly_storage_cost_micros: u64,
    pub transfer_cost_micros: u64,
    pub pricing_basis_digest: ContentDigest,
}

impl StorageCostEstimate {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.currency.trim().is_empty()
            || !self.pricing_basis_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "storage cost estimate requires currency and pricing evidence".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn is_zero_cost(&self) -> bool {
        self.upfront_cost_micros == 0
            && self.monthly_storage_cost_micros == 0
            && self.transfer_cost_micros == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PaidBackendApproval {
    pub schema_version: u32,
    pub backend_id: String,
    pub estimate_digest: ContentDigest,
    pub governance_policy_digest: ContentDigest,
    pub approver: String,
    pub approval_evidence_digest: ContentDigest,
    pub maximum_upfront_cost_micros: u64,
    pub maximum_monthly_storage_cost_micros: u64,
    pub maximum_transfer_cost_micros: u64,
    pub authorizes_github_override: bool,
    pub justification: String,
}

impl PaidBackendApproval {
    fn validate_for(
        &self,
        backend_id: &str,
        estimate: &StorageCostEstimate,
        governance_policy_digest: &ContentDigest,
    ) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.backend_id != backend_id
            || self.estimate_digest != estimate.digest()?
            || &self.governance_policy_digest != governance_policy_digest
            || self.approver.trim().is_empty()
            || !self.approval_evidence_digest.validate()
            || self.justification.trim().is_empty()
            || estimate.upfront_cost_micros > self.maximum_upfront_cost_micros
            || estimate.monthly_storage_cost_micros > self.maximum_monthly_storage_cost_micros
            || estimate.transfer_cost_micros > self.maximum_transfer_cost_micros
        {
            return Err(CacheError::PermissionDenied(
                "paid backend is not covered by the exact governance approval".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoragePlacementCandidate {
    pub candidate_id: String,
    pub backend_id: String,
    pub backend_class: StorageBackendClass,
    pub destination: String,
    pub logical_bytes: u64,
    pub unique_added_bytes: u64,
    pub transfer_bytes: u64,
    pub retention: StorageRetentionClass,
    pub available: bool,
    pub operationally_suitable: bool,
    pub cost: StorageCostEstimate,
    pub paid_approval: Option<PaidBackendApproval>,
}

impl StoragePlacementCandidate {
    fn validate(&self) -> Result<(), CacheError> {
        self.cost.validate()?;
        if self.candidate_id.trim().is_empty()
            || self.backend_id.trim().is_empty()
            || self.destination.trim().is_empty()
            || self.logical_bytes == 0
            || self.unique_added_bytes > self.logical_bytes
        {
            return Err(CacheError::InvalidManifest(
                "storage placement candidate identity or size estimate is invalid".to_owned(),
            ));
        }
        if self.backend_class == StorageBackendClass::PaidExternal && self.cost.is_zero_cost() {
            return Err(CacheError::InvalidManifest(
                "a paid backend must declare its nonzero cost estimate".to_owned(),
            ));
        }
        if self.backend_class != StorageBackendClass::PaidExternal && self.paid_approval.is_some() {
            return Err(CacheError::InvalidManifest(
                "paid-backend approval is invalid for a nonpaid backend".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoragePlacementPolicy {
    pub schema_version: u32,
    pub governance_policy_digest: ContentDigest,
    pub github_first: bool,
    pub maximum_unique_added_bytes: Option<u64>,
    pub maximum_transfer_bytes: Option<u64>,
    pub allowed_retention: Vec<StorageRetentionClass>,
}

impl StoragePlacementPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        let retention = self
            .allowed_retention
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if self.schema_version != 1
            || !self.governance_policy_digest.validate()
            || !self.github_first
            || self.allowed_retention.is_empty()
            || retention.len() != self.allowed_retention.len()
            || self.maximum_unique_added_bytes == Some(0)
            || self.maximum_transfer_bytes == Some(0)
        {
            return Err(CacheError::InvalidManifest(
                "storage placement policy must be GitHub-first with explicit nonzero quotas and retention"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoragePlacementRequest {
    pub schema_version: u32,
    pub policy: StoragePlacementPolicy,
    pub candidates: Vec<StoragePlacementCandidate>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoragePlacementDecision {
    pub candidate_id: String,
    pub backend_id: String,
    pub backend_class: StorageBackendClass,
    pub destination: String,
    pub logical_bytes: u64,
    pub unique_added_bytes: u64,
    pub transfer_bytes: u64,
    pub retention: StorageRetentionClass,
    pub cost: StorageCostEstimate,
    pub admissible: bool,
    pub selected: bool,
    pub paid_approval_evidence_digest: Option<ContentDigest>,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoragePlacementPlan {
    pub schema_version: u32,
    pub selected_candidate_id: Option<String>,
    pub github_was_available_and_suitable: bool,
    pub paid_backend_selected: bool,
    pub decisions: Vec<StoragePlacementDecision>,
}

pub fn plan_storage_placement(
    request: &StoragePlacementRequest,
) -> Result<StoragePlacementPlan, CacheError> {
    if request.schema_version != 1 || request.candidates.is_empty() {
        return Err(CacheError::InvalidManifest(
            "storage placement request requires candidates".to_owned(),
        ));
    }
    request.policy.validate()?;
    let mut ids = BTreeSet::new();
    for candidate in &request.candidates {
        candidate.validate()?;
        if !ids.insert(&candidate.candidate_id) {
            return Err(CacheError::InvalidManifest(
                "storage placement candidate identities must be unique".to_owned(),
            ));
        }
    }
    let github_was_available_and_suitable = request.candidates.iter().any(|candidate| {
        candidate.backend_class.is_github()
            && candidate.available
            && candidate.operationally_suitable
            && request
                .policy
                .maximum_unique_added_bytes
                .is_none_or(|maximum| candidate.unique_added_bytes <= maximum)
            && request
                .policy
                .maximum_transfer_bytes
                .is_none_or(|maximum| candidate.transfer_bytes <= maximum)
            && request
                .policy
                .allowed_retention
                .contains(&candidate.retention)
    });
    let mut decisions = request
        .candidates
        .iter()
        .map(|candidate| {
            evaluate_placement_candidate(
                &request.policy,
                candidate,
                github_was_available_and_suitable,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected_index = decisions
        .iter()
        .enumerate()
        .filter(|(_, decision)| decision.admissible)
        .min_by_key(|(_, decision)| {
            let class_rank = if decision.backend_class.is_github() {
                0
            } else if decision.backend_class == StorageBackendClass::PaidExternal {
                2
            } else {
                1
            };
            (
                class_rank,
                decision.cost.upfront_cost_micros,
                decision.cost.monthly_storage_cost_micros,
                decision.cost.transfer_cost_micros,
                decision.unique_added_bytes,
                decision.transfer_bytes,
                decision.candidate_id.clone(),
            )
        })
        .map(|(index, _)| index);
    let selected_candidate_id = selected_index.map(|index| {
        decisions[index].selected = true;
        decisions[index].candidate_id.clone()
    });
    let paid_backend_selected = selected_index
        .is_some_and(|index| decisions[index].backend_class == StorageBackendClass::PaidExternal);
    decisions.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    Ok(StoragePlacementPlan {
        schema_version: 1,
        selected_candidate_id,
        github_was_available_and_suitable,
        paid_backend_selected,
        decisions,
    })
}

fn evaluate_placement_candidate(
    policy: &StoragePlacementPolicy,
    candidate: &StoragePlacementCandidate,
    github_available: bool,
) -> Result<StoragePlacementDecision, CacheError> {
    let mut reasons = Vec::new();
    if !candidate.available {
        reasons.push("backend is unavailable".to_owned());
    }
    if !candidate.operationally_suitable {
        reasons.push("backend is not operationally suitable".to_owned());
    }
    if policy
        .maximum_unique_added_bytes
        .is_some_and(|maximum| candidate.unique_added_bytes > maximum)
    {
        reasons.push("projected unique added bytes exceed quota".to_owned());
    }
    if policy
        .maximum_transfer_bytes
        .is_some_and(|maximum| candidate.transfer_bytes > maximum)
    {
        reasons.push("projected transfer bytes exceed quota".to_owned());
    }
    if !policy.allowed_retention.contains(&candidate.retention) {
        reasons.push("retention class is disallowed by policy".to_owned());
    }
    let paid_approval_evidence_digest = if candidate.backend_class
        == StorageBackendClass::PaidExternal
    {
        match &candidate.paid_approval {
            Some(approval) => match approval.validate_for(
                &candidate.backend_id,
                &candidate.cost,
                &policy.governance_policy_digest,
            ) {
                Ok(()) => {
                    if github_available && !approval.authorizes_github_override {
                        reasons.push(
                            "GitHub is free and suitable; approval does not authorize override"
                                .to_owned(),
                        );
                    }
                    Some(approval.approval_evidence_digest.clone())
                }
                Err(error) => {
                    reasons.push(error.to_string());
                    None
                }
            },
            None => {
                reasons
                    .push("paid backend requires explicit cost and governance approval".to_owned());
                None
            }
        }
    } else {
        None
    };
    Ok(StoragePlacementDecision {
        candidate_id: candidate.candidate_id.clone(),
        backend_id: candidate.backend_id.clone(),
        backend_class: candidate.backend_class,
        destination: candidate.destination.clone(),
        logical_bytes: candidate.logical_bytes,
        unique_added_bytes: candidate.unique_added_bytes,
        transfer_bytes: candidate.transfer_bytes,
        retention: candidate.retention,
        cost: candidate.cost.clone(),
        admissible: reasons.is_empty(),
        selected: false,
        paid_approval_evidence_digest,
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(seed: &[u8], monthly: u64) -> StorageCostEstimate {
        StorageCostEstimate {
            schema_version: 1,
            currency: "USD".to_owned(),
            upfront_cost_micros: 0,
            monthly_storage_cost_micros: monthly,
            transfer_cost_micros: monthly / 10,
            pricing_basis_digest: ContentDigest::sha256(seed),
        }
    }

    fn candidate(id: &str, class: StorageBackendClass, monthly: u64) -> StoragePlacementCandidate {
        StoragePlacementCandidate {
            candidate_id: id.to_owned(),
            backend_id: id.to_owned(),
            backend_class: class,
            destination: format!("destination/{id}"),
            logical_bytes: 1_000,
            unique_added_bytes: 800,
            transfer_bytes: 900,
            retention: StorageRetentionClass::DurableCache,
            available: true,
            operationally_suitable: true,
            cost: estimate(id.as_bytes(), monthly),
            paid_approval: None,
        }
    }

    fn policy() -> StoragePlacementPolicy {
        StoragePlacementPolicy {
            schema_version: 1,
            governance_policy_digest: ContentDigest::sha256(b"cost-governance"),
            github_first: true,
            maximum_unique_added_bytes: Some(1_000),
            maximum_transfer_bytes: Some(1_000),
            allowed_retention: vec![StorageRetentionClass::DurableCache],
        }
    }

    #[test]
    fn github_remains_default_and_unapproved_paid_backend_is_explained() {
        let request = StoragePlacementRequest {
            schema_version: 1,
            policy: policy(),
            candidates: vec![
                candidate("github", StorageBackendClass::GithubPrivate, 0),
                candidate("paid", StorageBackendClass::PaidExternal, 50_000),
            ],
        };
        let plan = plan_storage_placement(&request).unwrap();
        assert_eq!(plan.selected_candidate_id.as_deref(), Some("github"));
        assert!(!plan.paid_backend_selected);
        assert!(plan
            .decisions
            .iter()
            .find(|decision| decision.candidate_id == "paid")
            .unwrap()
            .reasons
            .iter()
            .any(|reason| reason.contains("explicit cost")));
    }

    #[test]
    fn exact_approved_paid_backend_is_eligible_when_github_is_unavailable() {
        let policy = policy();
        let mut github = candidate("github", StorageBackendClass::GithubPrivate, 0);
        github.available = false;
        let mut paid = candidate("paid", StorageBackendClass::PaidExternal, 50_000);
        paid.paid_approval = Some(PaidBackendApproval {
            schema_version: 1,
            backend_id: paid.backend_id.clone(),
            estimate_digest: paid.cost.digest().unwrap(),
            governance_policy_digest: policy.governance_policy_digest.clone(),
            approver: "storage-governor".to_owned(),
            approval_evidence_digest: ContentDigest::sha256(b"approval"),
            maximum_upfront_cost_micros: 0,
            maximum_monthly_storage_cost_micros: 50_000,
            maximum_transfer_cost_micros: 5_000,
            authorizes_github_override: false,
            justification: "GitHub is temporarily unsuitable".to_owned(),
        });
        let plan = plan_storage_placement(&StoragePlacementRequest {
            schema_version: 1,
            policy,
            candidates: vec![github, paid],
        })
        .unwrap();
        assert_eq!(plan.selected_candidate_id.as_deref(), Some("paid"));
        assert!(plan.paid_backend_selected);
    }

    #[test]
    fn quotas_reject_large_materialization_before_execution() {
        let mut github = candidate("github", StorageBackendClass::GithubPrivate, 0);
        github.logical_bytes = 2_000;
        github.unique_added_bytes = 1_001;
        let plan = plan_storage_placement(&StoragePlacementRequest {
            schema_version: 1,
            policy: policy(),
            candidates: vec![github],
        })
        .unwrap();
        assert!(plan.selected_candidate_id.is_none());
        assert!(plan.decisions[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("quota")));
    }
}
