//! Deterministic dependency-aware cache derivation planning.

use crate::{ArtifactAssuranceState, CacheError, ContentDigest};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePlanAction {
    Load,
    Derive,
    Recompute,
    Certify,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePlanTrust {
    Unverified,
    HashVerified,
    PolicyTrusted,
    IndependentlyArchived,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePlanLocality {
    ProcessLocal,
    WorkstationLocal,
    ProjectPrivate,
    TeamPrivateRemote,
    PublicRemote,
    ExternalRemote,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePreferenceCriterion {
    Trust,
    Precision,
    Locality,
    TransferBytes,
    RecomputationCost,
    ActionPreference,
}

const ALL_CRITERIA: [CachePreferenceCriterion; 6] = [
    CachePreferenceCriterion::Trust,
    CachePreferenceCriterion::Precision,
    CachePreferenceCriterion::Locality,
    CachePreferenceCriterion::TransferBytes,
    CachePreferenceCriterion::RecomputationCost,
    CachePreferenceCriterion::ActionPreference,
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePlannerPolicy {
    pub schema_version: u32,
    pub minimum_trust: CachePlanTrust,
    pub maximum_total_transfer_bytes: Option<u64>,
    pub maximum_total_recomputation_units: Option<u64>,
    pub preference_order: Vec<CachePreferenceCriterion>,
    pub action_preference: Vec<CachePlanAction>,
}

impl CachePlannerPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        let criteria = self
            .preference_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let actions = self
            .action_preference
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if self.schema_version == 0
            || self.preference_order.len() != ALL_CRITERIA.len()
            || criteria != BTreeSet::from(ALL_CRITERIA)
            || self.action_preference.is_empty()
            || actions.len() != self.action_preference.len()
        {
            return Err(CacheError::InvalidManifest(
                "cache planner policy must contain every preference criterion once and a unique action preference"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn action_rank(&self, action: CachePlanAction) -> Option<usize> {
        self.action_preference
            .iter()
            .position(|candidate| *candidate == action)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePlanCandidate {
    pub candidate_id: String,
    pub action: CachePlanAction,
    pub source: Option<String>,
    pub trust: CachePlanTrust,
    pub precision_bits: u64,
    pub locality: CachePlanLocality,
    pub transfer_bytes: u64,
    pub recomputation_units: u64,
    pub projected_assurance: ArtifactAssuranceState,
    pub dependency_node_ids: BTreeSet<String>,
    pub compatible: bool,
    pub revoked: bool,
    pub available: bool,
}

impl CachePlanCandidate {
    fn validate(&self) -> Result<(), CacheError> {
        if self.candidate_id.trim().is_empty()
            || self.precision_bits == 0
            || self
                .source
                .as_ref()
                .is_some_and(|source| source.trim().is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "cache plan candidate identity, source, or precision is invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePlanNode {
    pub node_id: String,
    pub artifact_family: String,
    pub semantic_digest: ContentDigest,
    pub required_assurance: ArtifactAssuranceState,
    pub minimum_precision_bits: u64,
    pub dependency_node_ids: BTreeSet<String>,
    pub candidates: Vec<CachePlanCandidate>,
}

impl CachePlanNode {
    fn validate(&self) -> Result<(), CacheError> {
        if self.node_id.trim().is_empty()
            || self.artifact_family.trim().is_empty()
            || !self.semantic_digest.validate()
            || self.minimum_precision_bits == 0
            || self.candidates.is_empty()
            || self.dependency_node_ids.contains(&self.node_id)
        {
            return Err(CacheError::InvalidManifest(
                "cache plan node identity, family, precision, dependencies, or candidates are invalid"
                    .to_owned(),
            ));
        }
        let mut candidate_ids = BTreeSet::new();
        for candidate in &self.candidates {
            candidate.validate()?;
            if !candidate_ids.insert(&candidate.candidate_id) {
                return Err(CacheError::InvalidManifest(
                    "cache plan node contains duplicate candidate identities".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePlanRequest {
    pub schema_version: u32,
    pub policy: CachePlannerPolicy,
    pub nodes: Vec<CachePlanNode>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheCandidatePlanDecision {
    pub candidate_id: String,
    pub action: CachePlanAction,
    pub source: Option<String>,
    pub admissible: bool,
    pub selected: bool,
    pub trust: CachePlanTrust,
    pub precision_bits: u64,
    pub locality: CachePlanLocality,
    pub transfer_bytes: u64,
    pub recomputation_units: u64,
    pub projected_assurance: ArtifactAssuranceState,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheNodePlan {
    pub node_id: String,
    pub artifact_family: String,
    pub semantic_digest: ContentDigest,
    pub selected_candidate_id: Option<String>,
    pub selected_action: Option<CachePlanAction>,
    pub decisions: Vec<CacheCandidatePlanDecision>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheDerivationPlan {
    pub schema_version: u32,
    pub complete: bool,
    pub execution_order: Vec<String>,
    pub total_transfer_bytes: u64,
    pub total_recomputation_units: u64,
    pub nodes: Vec<CacheNodePlan>,
}

pub fn plan_cache_derivations(
    request: &CachePlanRequest,
) -> Result<CacheDerivationPlan, CacheError> {
    if request.schema_version == 0 || request.nodes.is_empty() {
        return Err(CacheError::InvalidManifest(
            "cache plan request schema and nodes are required".to_owned(),
        ));
    }
    request.policy.validate()?;
    let mut nodes = BTreeMap::new();
    for node in &request.nodes {
        node.validate()?;
        if nodes.insert(node.node_id.clone(), node).is_some() {
            return Err(CacheError::InvalidManifest(
                "cache plan contains duplicate node identities".to_owned(),
            ));
        }
    }
    for node in nodes.values() {
        if node
            .dependency_node_ids
            .iter()
            .any(|dependency| !nodes.contains_key(dependency))
        {
            return Err(CacheError::InvalidManifest(format!(
                "cache plan node {:?} names an unknown dependency",
                node.node_id
            )));
        }
    }
    let execution_order = topological_order(&nodes)?;
    let mut selected_nodes = BTreeSet::new();
    let mut total_transfer_bytes = 0u64;
    let mut total_recomputation_units = 0u64;
    let mut plans = Vec::new();
    for node_id in &execution_order {
        let node = nodes[node_id];
        let unavailable_dependencies = node
            .dependency_node_ids
            .iter()
            .filter(|dependency| !selected_nodes.contains(*dependency))
            .cloned()
            .collect::<Vec<_>>();
        let mut decisions = node
            .candidates
            .iter()
            .map(|candidate| {
                evaluate_candidate(
                    &request.policy,
                    node,
                    candidate,
                    &unavailable_dependencies,
                    total_transfer_bytes,
                    total_recomputation_units,
                )
            })
            .collect::<Vec<_>>();
        let selected_index = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| decision.admissible)
            .min_by(|(_, left), (_, right)| compare_candidates(&request.policy, left, right))
            .map(|(index, _)| index);
        let (selected_candidate_id, selected_action) = if let Some(index) = selected_index {
            decisions[index].selected = true;
            total_transfer_bytes = total_transfer_bytes
                .checked_add(decisions[index].transfer_bytes)
                .ok_or_else(|| {
                    CacheError::ResourceLimit("cache plan transfer bytes exceed u64".to_owned())
                })?;
            total_recomputation_units = total_recomputation_units
                .checked_add(decisions[index].recomputation_units)
                .ok_or_else(|| {
                    CacheError::ResourceLimit(
                        "cache plan recomputation units exceed u64".to_owned(),
                    )
                })?;
            selected_nodes.insert(node_id.clone());
            (
                Some(decisions[index].candidate_id.clone()),
                Some(decisions[index].action),
            )
        } else {
            (None, None)
        };
        decisions.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
        plans.push(CacheNodePlan {
            node_id: node_id.clone(),
            artifact_family: node.artifact_family.clone(),
            semantic_digest: node.semantic_digest.clone(),
            selected_candidate_id,
            selected_action,
            decisions,
        });
    }
    Ok(CacheDerivationPlan {
        schema_version: 1,
        complete: selected_nodes.len() == nodes.len(),
        execution_order,
        total_transfer_bytes,
        total_recomputation_units,
        nodes: plans,
    })
}

fn evaluate_candidate(
    policy: &CachePlannerPolicy,
    node: &CachePlanNode,
    candidate: &CachePlanCandidate,
    unavailable_dependencies: &[String],
    current_transfer_bytes: u64,
    current_recomputation_units: u64,
) -> CacheCandidatePlanDecision {
    let mut reasons = Vec::new();
    if policy.action_rank(candidate.action).is_none() {
        reasons.push(format!(
            "action {:?} is disabled by policy",
            candidate.action
        ));
    }
    if !candidate.available {
        reasons.push("candidate source or implementation is unavailable".to_owned());
    }
    if candidate.revoked {
        reasons.push("candidate is revoked".to_owned());
    }
    if !candidate.compatible {
        reasons.push("candidate is incompatible with the request".to_owned());
    }
    if candidate.trust < policy.minimum_trust {
        reasons.push(format!(
            "candidate trust {:?} is below {:?}",
            candidate.trust, policy.minimum_trust
        ));
    }
    if candidate.precision_bits < node.minimum_precision_bits {
        reasons.push(format!(
            "candidate precision {} is below {} bits",
            candidate.precision_bits, node.minimum_precision_bits
        ));
    }
    if candidate.projected_assurance < node.required_assurance {
        reasons.push(format!(
            "candidate assurance {:?} is below {:?}",
            candidate.projected_assurance, node.required_assurance
        ));
    }
    if candidate.dependency_node_ids != node.dependency_node_ids {
        reasons.push("candidate dependency identity set does not match the node".to_owned());
    }
    if !unavailable_dependencies.is_empty() {
        reasons.push(format!(
            "required dependency nodes are unavailable: {}",
            unavailable_dependencies.join(", ")
        ));
    }
    let projected_transfer = current_transfer_bytes.saturating_add(candidate.transfer_bytes);
    if policy
        .maximum_total_transfer_bytes
        .is_some_and(|maximum| projected_transfer > maximum)
    {
        reasons.push(format!(
            "projected transfer {projected_transfer} exceeds policy"
        ));
    }
    let projected_recomputation =
        current_recomputation_units.saturating_add(candidate.recomputation_units);
    if policy
        .maximum_total_recomputation_units
        .is_some_and(|maximum| projected_recomputation > maximum)
    {
        reasons.push(format!(
            "projected recomputation {projected_recomputation} exceeds policy"
        ));
    }
    CacheCandidatePlanDecision {
        candidate_id: candidate.candidate_id.clone(),
        action: candidate.action,
        source: candidate.source.clone(),
        admissible: reasons.is_empty(),
        selected: false,
        trust: candidate.trust,
        precision_bits: candidate.precision_bits,
        locality: candidate.locality,
        transfer_bytes: candidate.transfer_bytes,
        recomputation_units: candidate.recomputation_units,
        projected_assurance: candidate.projected_assurance,
        reasons,
    }
}

fn compare_candidates(
    policy: &CachePlannerPolicy,
    left: &CacheCandidatePlanDecision,
    right: &CacheCandidatePlanDecision,
) -> Ordering {
    for criterion in &policy.preference_order {
        let ordering = match criterion {
            CachePreferenceCriterion::Trust => right.trust.cmp(&left.trust),
            CachePreferenceCriterion::Precision => right.precision_bits.cmp(&left.precision_bits),
            CachePreferenceCriterion::Locality => left.locality.cmp(&right.locality),
            CachePreferenceCriterion::TransferBytes => {
                left.transfer_bytes.cmp(&right.transfer_bytes)
            }
            CachePreferenceCriterion::RecomputationCost => {
                left.recomputation_units.cmp(&right.recomputation_units)
            }
            CachePreferenceCriterion::ActionPreference => policy
                .action_rank(left.action)
                .cmp(&policy.action_rank(right.action)),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.candidate_id.cmp(&right.candidate_id)
}

fn topological_order(nodes: &BTreeMap<String, &CachePlanNode>) -> Result<Vec<String>, CacheError> {
    let mut incoming = nodes
        .iter()
        .map(|(node_id, node)| (node_id.clone(), node.dependency_node_ids.len()))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = BTreeMap::<String, BTreeSet<String>>::new();
    for (node_id, node) in nodes {
        for dependency in &node.dependency_node_ids {
            dependents
                .entry(dependency.clone())
                .or_default()
                .insert(node_id.clone());
        }
    }
    let mut ready = incoming
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(node_id, _)| node_id.clone())
        .collect::<BTreeSet<_>>();
    let mut order = Vec::new();
    while let Some(node_id) = ready.pop_first() {
        order.push(node_id.clone());
        for dependent in dependents.get(&node_id).into_iter().flatten() {
            let count = incoming
                .get_mut(dependent)
                .expect("validated dependency graph contains every dependent");
            *count -= 1;
            if *count == 0 {
                ready.insert(dependent.clone());
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(CacheError::InvalidManifest(
            "cache plan dependency graph contains a cycle".to_owned(),
        ));
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CachePlannerPolicy {
        CachePlannerPolicy {
            schema_version: 1,
            minimum_trust: CachePlanTrust::HashVerified,
            maximum_total_transfer_bytes: Some(100),
            maximum_total_recomputation_units: Some(1_000),
            preference_order: ALL_CRITERIA.to_vec(),
            action_preference: vec![
                CachePlanAction::Load,
                CachePlanAction::Derive,
                CachePlanAction::Recompute,
                CachePlanAction::Certify,
            ],
        }
    }

    fn candidate(
        id: &str,
        action: CachePlanAction,
        trust: CachePlanTrust,
        locality: CachePlanLocality,
        transfer: u64,
        recomputation: u64,
        dependencies: BTreeSet<String>,
    ) -> CachePlanCandidate {
        CachePlanCandidate {
            candidate_id: id.to_owned(),
            action,
            source: Some(id.to_owned()),
            trust,
            precision_bits: 256,
            locality,
            transfer_bytes: transfer,
            recomputation_units: recomputation,
            projected_assurance: ArtifactAssuranceState::CrossChecked,
            dependency_node_ids: dependencies,
            compatible: true,
            revoked: false,
            available: true,
        }
    }

    fn node(
        id: &str,
        dependencies: BTreeSet<String>,
        candidates: Vec<CachePlanCandidate>,
    ) -> CachePlanNode {
        CachePlanNode {
            node_id: id.to_owned(),
            artifact_family: "ccm".to_owned(),
            semantic_digest: ContentDigest::sha256(id.as_bytes()),
            required_assurance: ArtifactAssuranceState::Computed,
            minimum_precision_bits: 128,
            dependency_node_ids: dependencies,
            candidates,
        }
    }

    #[test]
    fn planner_prefers_trust_before_cheaper_transfer() {
        let request = CachePlanRequest {
            schema_version: 1,
            policy: policy(),
            nodes: vec![node(
                "matrix",
                BTreeSet::new(),
                vec![
                    candidate(
                        "cheap-untrusted",
                        CachePlanAction::Load,
                        CachePlanTrust::Unverified,
                        CachePlanLocality::WorkstationLocal,
                        0,
                        0,
                        BTreeSet::new(),
                    ),
                    candidate(
                        "trusted-public",
                        CachePlanAction::Load,
                        CachePlanTrust::PolicyTrusted,
                        CachePlanLocality::PublicRemote,
                        20,
                        0,
                        BTreeSet::new(),
                    ),
                ],
            )],
        };
        let plan = plan_cache_derivations(&request).unwrap();
        assert!(plan.complete);
        assert_eq!(
            plan.nodes[0].selected_candidate_id.as_deref(),
            Some("trusted-public")
        );
        assert!(plan.nodes[0]
            .decisions
            .iter()
            .find(|decision| decision.candidate_id == "cheap-untrusted")
            .is_some_and(|decision| !decision.admissible));
    }

    #[test]
    fn transfer_quota_selects_recomputation_and_preserves_dependency_order() {
        let source = node(
            "source",
            BTreeSet::new(),
            vec![candidate(
                "source-local",
                CachePlanAction::Load,
                CachePlanTrust::PolicyTrusted,
                CachePlanLocality::WorkstationLocal,
                0,
                0,
                BTreeSet::new(),
            )],
        );
        let dependencies = BTreeSet::from(["source".to_owned()]);
        let derived = node(
            "derived",
            dependencies.clone(),
            vec![
                candidate(
                    "remote-large",
                    CachePlanAction::Load,
                    CachePlanTrust::PolicyTrusted,
                    CachePlanLocality::PublicRemote,
                    200,
                    0,
                    dependencies.clone(),
                ),
                candidate(
                    "derive-local",
                    CachePlanAction::Derive,
                    CachePlanTrust::PolicyTrusted,
                    CachePlanLocality::WorkstationLocal,
                    0,
                    10,
                    dependencies,
                ),
            ],
        );
        let plan = plan_cache_derivations(&CachePlanRequest {
            schema_version: 1,
            policy: policy(),
            nodes: vec![derived, source],
        })
        .unwrap();
        assert_eq!(plan.execution_order, vec!["source", "derived"]);
        assert_eq!(
            plan.nodes[1].selected_candidate_id.as_deref(),
            Some("derive-local")
        );
    }

    #[test]
    fn missing_dependency_selection_is_explained_downstream() {
        let mut unavailable = candidate(
            "missing",
            CachePlanAction::Load,
            CachePlanTrust::PolicyTrusted,
            CachePlanLocality::PublicRemote,
            0,
            0,
            BTreeSet::new(),
        );
        unavailable.available = false;
        let dependency = node("dependency", BTreeSet::new(), vec![unavailable]);
        let dependencies = BTreeSet::from(["dependency".to_owned()]);
        let derived = node(
            "derived",
            dependencies.clone(),
            vec![candidate(
                "derive",
                CachePlanAction::Derive,
                CachePlanTrust::PolicyTrusted,
                CachePlanLocality::WorkstationLocal,
                0,
                1,
                dependencies,
            )],
        );
        let plan = plan_cache_derivations(&CachePlanRequest {
            schema_version: 1,
            policy: policy(),
            nodes: vec![derived, dependency],
        })
        .unwrap();
        assert!(!plan.complete);
        assert!(plan.nodes[1].decisions[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("dependency")));
    }

    #[test]
    fn cyclic_plan_is_rejected() {
        let left_dependencies = BTreeSet::from(["right".to_owned()]);
        let right_dependencies = BTreeSet::from(["left".to_owned()]);
        let request = CachePlanRequest {
            schema_version: 1,
            policy: policy(),
            nodes: vec![
                node(
                    "left",
                    left_dependencies.clone(),
                    vec![candidate(
                        "left-candidate",
                        CachePlanAction::Derive,
                        CachePlanTrust::PolicyTrusted,
                        CachePlanLocality::WorkstationLocal,
                        0,
                        1,
                        left_dependencies,
                    )],
                ),
                node(
                    "right",
                    right_dependencies.clone(),
                    vec![candidate(
                        "right-candidate",
                        CachePlanAction::Derive,
                        CachePlanTrust::PolicyTrusted,
                        CachePlanLocality::WorkstationLocal,
                        0,
                        1,
                        right_dependencies,
                    )],
                ),
            ],
        };
        assert!(plan_cache_derivations(&request).is_err());
    }
}
