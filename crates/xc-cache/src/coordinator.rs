//! Fail-closed publication planning across policy, topology, and capacity.

use crate::{
    authorize_target_publication, plan_publication_batches, select_write_shard_from_local_ledgers,
    AuthenticatedGitHubSession, CacheError, CacheNetworkRegistry, CachePublicationCandidate,
    CachePublicationPolicy, CacheVisibility, CapacityAdmission, CapacityLedger, ContentDigest,
    ContributorPublicationEvidence, PublicationAuthorizationDecision, PublicationDestination,
    PublicationReviewEvidence, PublicationTargetState, PublicationTransactionJournal,
    RemoteFabricTrustPolicy, RemoteGitStore, RemoteReadReport, RemoteShardReader,
    TargetPublicationAuditEvidence, TargetPublicationAuthorizationRequest, TopologyRegistry,
    TopologyShardStatus, TopologyTrustPolicy, TransportEncodingRecord, TransportPolicy,
    TrustedRepositoryRole, DEFAULT_CAPACITY_LEDGER_PATH,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use xc_core::{CancellationToken, PublicationAuthority, PublicationTarget};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteTopologySource {
    pub repository: String,
    pub branch: String,
    pub registry_path: String,
    pub maximum_registry_bytes: u64,
    pub capacity_ledger_path: String,
    pub maximum_capacity_ledger_bytes: u64,
}

impl RemoteTopologySource {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || !crate::protocol::normalized_relative_path(&self.registry_path)
            || !crate::protocol::normalized_relative_path(&self.capacity_ledger_path)
            || self.maximum_registry_bytes == 0
            || self.maximum_capacity_ledger_bytes == 0
        {
            return Err(CacheError::InvalidManifest(
                "remote topology and capacity-ledger source is invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for RemoteTopologySource {
    fn default() -> Self {
        Self {
            repository: String::new(),
            branch: "main".to_owned(),
            registry_path: "registry/topology.json".to_owned(),
            maximum_registry_bytes: 4 * 1024 * 1024,
            capacity_ledger_path: DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
            maximum_capacity_ledger_bytes: 4 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedPublicationAddition {
    pub unique_payload_bytes: u64,
    pub metadata_bytes: u64,
    pub projected_history_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemotePublicationRouteSelection {
    pub destination: PublicationDestination,
    pub shard_id: String,
    pub endpoint_id: String,
    pub authorized_repository: String,
    pub repository_url: String,
    pub branch: String,
    pub expected_head: String,
    pub admission: CapacityAdmission,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemotePublicationRoutingPlan {
    pub schema_version: u32,
    pub family: String,
    pub target: PublicationTarget,
    pub topology_revision: String,
    pub topology_source: RemoteReadReport,
    pub topology_digest: ContentDigest,
    pub topology: TopologyRegistry,
    pub ledgers: BTreeMap<String, CapacityLedger>,
    pub ledger_sources: BTreeMap<String, RemoteReadReport>,
    pub shard_heads: BTreeMap<String, String>,
    pub selections: BTreeMap<PublicationDestination, RemotePublicationRouteSelection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteTargetPublicationPlanningInput {
    pub candidate: CachePublicationCandidate,
    pub authority: PublicationAuthority,
    pub contributor: Option<ContributorPublicationEvidence>,
    pub reviews: Vec<PublicationReviewEvidence>,
    pub projected_metadata_bytes: u64,
    pub projected_history_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationRouteValidationReport {
    pub topology_revision: String,
    pub topology_source: RemoteReadReport,
    pub topology_digest: ContentDigest,
    pub validated_targets: BTreeMap<PublicationDestination, String>,
}

/// Re-read the trusted family topology and prove that every incomplete
/// journal target is still the current writable endpoint before resuming it.
/// This prevents a durable transaction from silently writing to a shard that
/// was retired, sealed, or placed on incident hold after initial planning.
#[allow(clippy::too_many_arguments)]
pub fn validate_remote_publication_routes(
    remote: &dyn RemoteGitStore,
    cancellation: &CancellationToken,
    journal: &PublicationTransactionJournal,
    family: &str,
    topology_source: &RemoteTopologySource,
    topology_trust: &TopologyTrustPolicy,
    fabric_trust: &RemoteFabricTrustPolicy,
    evaluation_unix_seconds: u64,
    network: &CacheNetworkRegistry,
) -> Result<PublicationRouteValidationReport, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    if family.trim().is_empty() {
        return Err(CacheError::InvalidManifest(
            "publication route validation family is required".to_owned(),
        ));
    }
    topology_source.validate()?;
    network.validate()?;
    let topology_revision =
        remote.read_ref(&topology_source.repository, &topology_source.branch)?;
    fabric_trust.verify_repository(
        TrustedRepositoryRole::Registry,
        &topology_source.repository,
        &topology_source.branch,
        &topology_revision,
        evaluation_unix_seconds,
    )?;
    let topology_document = RemoteShardReader::new(remote, topology_source.maximum_registry_bytes)?
        .load_trusted_topology(
            &topology_source.repository,
            &topology_revision,
            &topology_source.registry_path,
            topology_trust,
            cancellation,
        )?;
    if topology_document.registry.policy_digest != journal.policy_digest {
        return Err(CacheError::InvalidManifest(format!(
            "current topology policy {} does not match journal policy {}",
            topology_document.registry.policy_digest, journal.policy_digest
        )));
    }
    fabric_trust.verify_policy_digest(&topology_document.registry.policy_digest)?;
    let mut validated_targets = BTreeMap::new();
    for (destination, target) in &journal.targets {
        if matches!(
            target.state,
            PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
        ) {
            continue;
        }
        let visibility = destination_visibility(*destination);
        let route = topology_document
            .registry
            .route(family, visibility)
            .ok_or_else(|| {
                CacheError::NoWritableShard(format!(
                    "current topology has no {family:?} {visibility:?} route"
                ))
            })?;
        let shard = route
            .ordered_shards
            .iter()
            .find(|shard| shard.shard_id == target.shard_id)
            .ok_or_else(|| {
                CacheError::NoWritableShard(format!(
                    "journal shard {:?} is absent from the current family route",
                    target.shard_id
                ))
            })?;
        if shard.status != TopologyShardStatus::Writable {
            return Err(CacheError::NoWritableShard(format!(
                "journal shard {:?} is now {:?}",
                target.shard_id, shard.status
            )));
        }
        let endpoint = network
            .endpoint_for_shard(&shard.endpoint_id)
            .ok_or_else(|| {
                CacheError::NoWritableShard(format!(
                    "current topology endpoint {:?} is absent from the network registry",
                    shard.endpoint_id
                ))
            })?;
        let authorized_repository = format!("{}/{}", endpoint.owner, endpoint.repository);
        if !endpoint.enabled_for_write
            || endpoint.visibility != visibility
            || target.authorized_repository != authorized_repository
            || target.repository != endpoint.preferred_clone_url()
            || target.branch != endpoint.branch
        {
            return Err(CacheError::NoWritableShard(format!(
                "journal target {destination:?} no longer matches its writable network endpoint"
            )));
        }
        let current_head = remote.read_ref(&target.repository, &target.branch)?;
        fabric_trust.verify_repository(
            TrustedRepositoryRole::Shard,
            &authorized_repository,
            &target.branch,
            &current_head,
            evaluation_unix_seconds,
        )?;
        validated_targets.insert(*destination, target.shard_id.clone());
    }
    Ok(PublicationRouteValidationReport {
        topology_revision,
        topology_source: topology_document.source,
        topology_digest: topology_document.topology_digest,
        validated_targets,
    })
}

/// Fetch only the trusted topology document and shard-local capacity ledgers,
/// then select the fullest admissible writable shard. No repository is cloned
/// and the top-level registry remains independent of artifact-level changes.
#[allow(clippy::too_many_arguments)]
pub fn discover_remote_publication_routing(
    remote: &dyn RemoteGitStore,
    cancellation: &CancellationToken,
    family: &str,
    target: PublicationTarget,
    topology_source: &RemoteTopologySource,
    topology_trust: &TopologyTrustPolicy,
    fabric_trust: &RemoteFabricTrustPolicy,
    evaluation_unix_seconds: u64,
    network: &CacheNetworkRegistry,
    additions: &BTreeMap<PublicationDestination, ProjectedPublicationAddition>,
) -> Result<RemotePublicationRoutingPlan, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if family.trim().is_empty() {
        return Err(CacheError::InvalidManifest(
            "publication artifact family must be explicit".to_owned(),
        ));
    }
    topology_source.validate()?;
    network.validate()?;
    let destinations = target_destinations(target);
    if destinations.is_empty() || additions.keys().copied().collect::<Vec<_>>() != destinations {
        return Err(CacheError::InvalidManifest(
            "projected additions must exactly match the publication targets".to_owned(),
        ));
    }
    let topology_revision =
        remote.read_ref(&topology_source.repository, &topology_source.branch)?;
    fabric_trust.verify_repository(
        TrustedRepositoryRole::Registry,
        &topology_source.repository,
        &topology_source.branch,
        &topology_revision,
        evaluation_unix_seconds,
    )?;
    let topology_document = RemoteShardReader::new(remote, topology_source.maximum_registry_bytes)?
        .load_trusted_topology(
            &topology_source.repository,
            &topology_revision,
            &topology_source.registry_path,
            topology_trust,
            cancellation,
        )?;
    let topology = topology_document.registry;
    fabric_trust.verify_policy_digest(&topology.policy_digest)?;
    let topology_digest = topology_document.topology_digest;
    let topology_read_source = topology_document.source;
    let ledger_reader =
        RemoteShardReader::new(remote, topology_source.maximum_capacity_ledger_bytes)?;
    let mut ledgers = BTreeMap::new();
    let mut ledger_sources = BTreeMap::new();
    let mut shard_heads = BTreeMap::new();
    let mut selections = BTreeMap::new();

    for destination in destinations {
        let visibility = destination_visibility(destination);
        let route = topology.route(family, visibility).ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "no topology route for family={family:?}, visibility={visibility:?}"
            ))
        })?;
        for shard in &route.ordered_shards {
            if shard.status != TopologyShardStatus::Writable {
                continue;
            }
            let endpoint = network
                .endpoint_for_shard(&shard.endpoint_id)
                .ok_or_else(|| {
                    CacheError::NoWritableShard(format!(
                        "topology endpoint {:?} is absent from the network registry",
                        shard.endpoint_id
                    ))
                })?;
            if !endpoint.enabled_for_write || endpoint.visibility != visibility {
                return Err(CacheError::NoWritableShard(format!(
                    "topology endpoint {:?} is not writable for {visibility:?}",
                    shard.endpoint_id
                )));
            }
            let repository_url = endpoint.preferred_clone_url();
            let head = remote.read_ref(&repository_url, &endpoint.branch)?;
            fabric_trust.verify_repository(
                TrustedRepositoryRole::Shard,
                &format!("{}/{}", endpoint.owner, endpoint.repository),
                &endpoint.branch,
                &head,
                evaluation_unix_seconds,
            )?;
            let document = ledger_reader.read_json::<CapacityLedger>(
                &repository_url,
                &head,
                &topology_source.capacity_ledger_path,
                cancellation,
            )?;
            document.value.validate()?;
            if document.value.shard_id != shard.shard_id {
                return Err(CacheError::InvalidManifest(format!(
                    "capacity ledger {:?} does not match topology shard {:?}",
                    document.value.shard_id, shard.shard_id
                )));
            }
            if ledgers
                .insert(shard.shard_id.clone(), document.value)
                .is_some()
                || ledger_sources
                    .insert(shard.shard_id.clone(), document.source)
                    .is_some()
                || shard_heads.insert(shard.shard_id.clone(), head).is_some()
            {
                return Err(CacheError::InvalidManifest(format!(
                    "topology shard {:?} was fetched more than once",
                    shard.shard_id
                )));
            }
        }
        let addition = &additions[&destination];
        let (selected, admission) = select_write_shard_from_local_ledgers(
            &topology,
            &ledgers,
            family,
            visibility,
            addition.unique_payload_bytes,
            addition.metadata_bytes,
            addition.projected_history_bytes,
        )?;
        let endpoint = network
            .endpoint_for_shard(&selected.endpoint_id)
            .expect("selected shard endpoint was fetched");
        selections.insert(
            destination,
            RemotePublicationRouteSelection {
                destination,
                shard_id: selected.shard_id.clone(),
                endpoint_id: selected.endpoint_id.clone(),
                authorized_repository: format!("{}/{}", endpoint.owner, endpoint.repository),
                repository_url: endpoint.preferred_clone_url(),
                branch: endpoint.branch.clone(),
                expected_head: shard_heads[&selected.shard_id].clone(),
                admission,
            },
        );
    }

    Ok(RemotePublicationRoutingPlan {
        schema_version: 1,
        family: family.to_owned(),
        target,
        topology_revision,
        topology_source: topology_read_source,
        topology_digest,
        topology,
        ledgers,
        ledger_sources,
        shard_heads,
        selections,
    })
}

/// Bind live credential evidence and target-specific authorization to a
/// previously discovered remote routing plan, producing the durable journal
/// without performing any remote mutation.
#[allow(clippy::too_many_arguments)]
pub fn coordinate_discovered_publication(
    routing: &RemotePublicationRoutingPlan,
    policy: &CachePublicationPolicy,
    topology_trust: &TopologyTrustPolicy,
    network: &CacheNetworkRegistry,
    encoding: &TransportEncodingRecord,
    transport_policy: &TransportPolicy,
    target_inputs: &BTreeMap<PublicationDestination, RemoteTargetPublicationPlanningInput>,
    authenticated_sessions: &BTreeMap<PublicationDestination, AuthenticatedGitHubSession>,
) -> Result<CoordinatedPublicationPlan, CacheError> {
    if routing.schema_version != 1
        || routing.topology_digest != topology_trust.verify(&routing.topology)?
        || routing.selections.keys().copied().collect::<Vec<_>>()
            != target_destinations(routing.target)
        || target_inputs.keys().copied().collect::<Vec<_>>() != target_destinations(routing.target)
    {
        return Err(CacheError::InvalidManifest(
            "remote publication routing plan identity or targets are invalid".to_owned(),
        ));
    }
    let mut coordinated_inputs = BTreeMap::new();
    for (destination, input) in target_inputs {
        let selection = &routing.selections[destination];
        coordinated_inputs.insert(
            *destination,
            TargetPublicationPlanningInput {
                candidate: input.candidate.clone(),
                authorization: TargetPublicationAuthorizationRequest {
                    destination: *destination,
                    repository: selection.authorized_repository.clone(),
                    authority: input.authority.clone(),
                    contributor: input.contributor.clone(),
                    reviews: input.reviews.clone(),
                },
                expected_head: selection.expected_head.clone(),
                projected_metadata_bytes: input.projected_metadata_bytes,
                projected_history_bytes: input.projected_history_bytes,
            },
        );
    }
    let plan = coordinate_publication(
        &routing.family,
        routing.target,
        policy,
        &routing.topology,
        topology_trust,
        network,
        &routing.ledgers,
        encoding,
        transport_policy,
        &coordinated_inputs,
        authenticated_sessions,
    )?;
    for (destination, report) in &plan.target_reports {
        let selected = &routing.selections[destination];
        if report.accepted()
            && (report.shard_id.as_deref() != Some(&selected.shard_id)
                || report.repository_url.as_deref() != Some(&selected.repository_url)
                || report.branch.as_deref() != Some(&selected.branch))
        {
            return Err(CacheError::InvalidManifest(
                "coordinated publication no longer matches remote shard selection".to_owned(),
            ));
        }
    }
    Ok(plan)
}

fn destination_visibility(destination: PublicationDestination) -> CacheVisibility {
    match destination {
        PublicationDestination::Private => CacheVisibility::Private,
        PublicationDestination::Public => CacheVisibility::Public,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPublicationPlanningInput {
    pub candidate: CachePublicationCandidate,
    pub authorization: TargetPublicationAuthorizationRequest,
    pub expected_head: String,
    pub projected_metadata_bytes: u64,
    pub projected_history_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetPublicationPlanningReport {
    pub destination: PublicationDestination,
    pub authorization: PublicationAuthorizationDecision,
    pub capacity: Option<CapacityAdmission>,
    pub shard_id: Option<String>,
    pub repository_url: Option<String>,
    pub branch: Option<String>,
    pub reasons: Vec<String>,
}

impl TargetPublicationPlanningReport {
    pub fn accepted(&self) -> bool {
        self.authorization.authorized && self.reasons.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatedPublicationPlan {
    pub topology_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub target_reports: BTreeMap<PublicationDestination, TargetPublicationPlanningReport>,
    /// Present only when every requested target passed before remote mutation.
    pub journal: Option<PublicationTransactionJournal>,
}

impl CoordinatedPublicationPlan {
    pub fn authorized(&self) -> bool {
        self.journal.is_some()
            && self
                .target_reports
                .values()
                .all(TargetPublicationPlanningReport::accepted)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn coordinate_publication(
    family: &str,
    target: PublicationTarget,
    policy: &CachePublicationPolicy,
    topology: &TopologyRegistry,
    topology_trust: &TopologyTrustPolicy,
    network: &CacheNetworkRegistry,
    ledgers: &BTreeMap<String, CapacityLedger>,
    encoding: &TransportEncodingRecord,
    transport_policy: &TransportPolicy,
    target_inputs: &BTreeMap<PublicationDestination, TargetPublicationPlanningInput>,
    authenticated_sessions: &BTreeMap<PublicationDestination, AuthenticatedGitHubSession>,
) -> Result<CoordinatedPublicationPlan, CacheError> {
    if family.trim().is_empty() {
        return Err(CacheError::InvalidManifest(
            "publication artifact family must be explicit".to_owned(),
        ));
    }
    let publication_policy_digest = policy.digest()?;
    if topology.policy_digest != publication_policy_digest {
        return Err(CacheError::InvalidManifest(format!(
            "topology policy {} does not match resolved publication policy {}",
            topology.policy_digest, publication_policy_digest
        )));
    }
    let topology_digest = topology_trust.verify(topology)?;
    network.validate()?;
    encoding.validate()?;
    let transport_digest = encoding.digest()?;
    let batches = plan_publication_batches(&encoding.ordered_parts, transport_policy)?;
    let destinations = target_destinations(target);
    if destinations.is_empty()
        || target_inputs.keys().copied().collect::<Vec<_>>() != destinations
        || authenticated_sessions.keys().copied().collect::<Vec<_>>() != destinations
    {
        return Err(CacheError::InvalidManifest(
            "publication inputs and live authenticated sessions must exactly match the requested targets"
                .to_owned(),
        ));
    }

    let mut reports = BTreeMap::new();
    let mut manifest_digests = BTreeMap::new();
    let mut repositories = BTreeMap::new();
    let mut authorized_repositories = BTreeMap::new();
    let mut permission_evidence = BTreeMap::new();
    let mut shard_ids = BTreeMap::new();
    let mut branches = BTreeMap::new();
    let mut heads = BTreeMap::new();
    let mut common_semantic: Option<ContentDigest> = None;
    let mut common_payload: Option<ContentDigest> = None;

    for destination in destinations.iter().copied() {
        let input = &target_inputs[&destination];
        let authorization = authorize_target_publication(
            policy,
            &input.candidate,
            &input.authorization,
            &authenticated_sessions[&destination],
        )?;
        let mut report = TargetPublicationPlanningReport {
            destination,
            authorization,
            capacity: None,
            shard_id: None,
            repository_url: None,
            branch: None,
            reasons: Vec::new(),
        };
        if input.candidate.payload_digest != encoding.canonical_payload_digest {
            report
                .reasons
                .push("target candidate payload does not match the transport encoding".to_owned());
        }
        if common_semantic
            .as_ref()
            .is_some_and(|digest| digest != &input.candidate.semantic_digest)
        {
            report.reasons.push(
                "private and public target candidates have different semantic identities"
                    .to_owned(),
            );
        }
        if common_payload
            .as_ref()
            .is_some_and(|digest| digest != &input.candidate.payload_digest)
        {
            report.reasons.push(
                "private and public target candidates have different canonical payload identities"
                    .to_owned(),
            );
        }
        common_semantic.get_or_insert_with(|| input.candidate.semantic_digest.clone());
        common_payload.get_or_insert_with(|| input.candidate.payload_digest.clone());

        if report.authorization.authorized && report.reasons.is_empty() {
            let visibility = match destination {
                PublicationDestination::Private => CacheVisibility::Private,
                PublicationDestination::Public => CacheVisibility::Public,
            };
            match select_write_shard_from_local_ledgers(
                topology,
                ledgers,
                family,
                visibility,
                encoding.package_size_bytes,
                input.projected_metadata_bytes,
                input.projected_history_bytes,
            ) {
                Ok((shard, admission)) => {
                    let endpoint = network.endpoint_for_shard(&shard.endpoint_id);
                    match endpoint {
                        None => report.reasons.push(format!(
                            "topology endpoint {:?} is absent from the network registry",
                            shard.endpoint_id
                        )),
                        Some(endpoint)
                            if !endpoint.enabled_for_write || endpoint.visibility != visibility =>
                        {
                            report.reasons.push(format!(
                                "network endpoint {:?} is not writable for {visibility:?}",
                                shard.endpoint_id
                            ));
                        }
                        Some(endpoint) => {
                            let authority_repository =
                                format!("{}/{}", endpoint.owner, endpoint.repository);
                            if authority_repository != input.authorization.repository {
                                report.reasons.push(format!(
                                    "authorized repository {:?} does not match selected endpoint {:?}",
                                    input.authorization.repository, authority_repository
                                ));
                            } else {
                                report.capacity = Some(admission);
                                report.shard_id = Some(shard.shard_id.clone());
                                report.repository_url = Some(endpoint.preferred_clone_url());
                                report.branch = Some(endpoint.branch.clone());
                            }
                        }
                    }
                }
                Err(error) => report.reasons.push(error.to_string()),
            }
        }
        report.reasons.sort();
        report.reasons.dedup();
        if report.accepted() {
            manifest_digests.insert(destination, input.candidate.manifest_digest.clone());
            repositories.insert(
                destination,
                report.repository_url.clone().unwrap_or_default(),
            );
            authorized_repositories.insert(destination, input.authorization.repository.clone());
            permission_evidence.insert(
                destination,
                report.authorization.repository_permission.clone(),
            );
            shard_ids.insert(destination, report.shard_id.clone().unwrap_or_default());
            branches.insert(destination, report.branch.clone().unwrap_or_default());
            heads.insert(destination, input.expected_head.clone());
        }
        reports.insert(destination, report);
    }

    let all_accepted = reports
        .values()
        .all(TargetPublicationPlanningReport::accepted);
    let journal = if all_accepted {
        let mut journal = PublicationTransactionJournal::new(
            common_semantic.unwrap_or_else(|| ContentDigest::sha256(b"missing-semantic")),
            manifest_digests,
            common_payload.unwrap_or_else(|| ContentDigest::sha256(b"missing-payload")),
            publication_policy_digest,
            target,
            repositories,
            authorized_repositories,
            permission_evidence,
            shard_ids,
            branches,
            heads,
            &batches,
        )?;
        for destination in destinations.iter().copied() {
            let input = &target_inputs[&destination];
            let decision = &reports[&destination].authorization;
            journal.attach_target_audit_evidence(
                destination,
                TargetPublicationAuditEvidence {
                    policy_id: policy.policy_id.clone(),
                    authority_mode: input.authorization.authority.mode,
                    validation_evidence_digests: input
                        .candidate
                        .validator_evidence
                        .iter()
                        .filter(|evidence| evidence.passed && evidence.evidence_digest.validate())
                        .map(|evidence| evidence.evidence_digest.clone())
                        .collect(),
                    contributor_authorization_digest: decision
                        .contributor_authorization_digest
                        .clone(),
                    reviewer_approvals: input
                        .authorization
                        .reviews
                        .iter()
                        .filter(|review| review.approved && review.evidence_digest.validate())
                        .cloned()
                        .collect(),
                },
            )?;
        }
        Some(journal)
    } else {
        None
    };
    Ok(CoordinatedPublicationPlan {
        topology_digest,
        transport_digest,
        target_reports: reports,
        journal,
    })
}

fn target_destinations(target: PublicationTarget) -> Vec<PublicationDestination> {
    match target {
        PublicationTarget::None => Vec::new(),
        PublicationTarget::Private => vec![PublicationDestination::Private],
        PublicationTarget::Public => vec![PublicationDestination::Public],
        PublicationTarget::Both => vec![
            PublicationDestination::Private,
            PublicationDestination::Public,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArtifactAssuranceState, ArtifactCompletionState, ArtifactDisposition, ArtifactFamilyRoute,
        CachePublicationCandidate, GitHubRepositoryEndpoint, PublicSanitizerProfile,
        RepositoryPermission, TargetPublicationAuthorizationRequest, TopologyShardRoute,
        TopologyShardStatus, TransportPart, ValidatorEvidence,
        GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use serde_json::{json, Value};
    use std::collections::BTreeSet;
    use xc_core::{PublicationAuthority, PublicationAuthorityMode};

    fn policy() -> CachePublicationPolicy {
        CachePublicationPolicy {
            policy_id: "owner-policy-v1".to_owned(),
            owner_principals: ["test-owner".to_owned()].into_iter().collect(),
            allow_owner_direct: true,
            minimum_assurance: BTreeMap::from([
                (
                    PublicationDestination::Private,
                    ArtifactAssuranceState::Computed,
                ),
                (
                    PublicationDestination::Public,
                    ArtifactAssuranceState::CrossChecked,
                ),
            ]),
            required_validators: BTreeMap::from([
                (
                    PublicationDestination::Private,
                    ["manifest".to_owned()].into_iter().collect(),
                ),
                (
                    PublicationDestination::Public,
                    ["manifest".to_owned(), "public-sanitizer".to_owned()]
                        .into_iter()
                        .collect(),
                ),
            ]),
            minimum_unique_contributor_reviews: 1,
            sanitizer: PublicSanitizerProfile {
                allowed_leaf_fields: ["artifact.kind".to_owned()].into_iter().collect(),
                allowed_source_identifiers: BTreeSet::new(),
                allowed_repository_names: ["example-org/public-shard".to_owned()]
                    .into_iter()
                    .collect(),
                ..PublicSanitizerProfile::default()
            },
        }
    }

    fn topology(policy: &CachePublicationPolicy) -> TopologyRegistry {
        TopologyRegistry {
            schema_version: 1,
            generation: 1,
            previous_registry_digest: None,
            policy_digest: policy.digest().unwrap(),
            trust_anchor_ids: vec!["release-key".to_owned()],
            family_routes: vec![
                ArtifactFamilyRoute {
                    family: "ccm".to_owned(),
                    visibility: CacheVisibility::Private,
                    ordered_shards: vec![TopologyShardRoute {
                        shard_id: "private-001".to_owned(),
                        endpoint_id: "private-001".to_owned(),
                        sequence: 1,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: None,
                    }],
                },
                ArtifactFamilyRoute {
                    family: "ccm".to_owned(),
                    visibility: CacheVisibility::Public,
                    ordered_shards: vec![TopologyShardRoute {
                        shard_id: "public-001".to_owned(),
                        endpoint_id: "public-001".to_owned(),
                        sequence: 1,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: None,
                    }],
                },
            ],
        }
    }

    fn ledger(id: &str) -> CapacityLedger {
        CapacityLedger {
            schema_version: 1,
            shard_id: id.to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 0,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 1_000_000_000,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: "a".repeat(40),
            reconciliation_digest: ContentDigest::sha256(id.as_bytes()),
        }
    }

    fn network() -> CacheNetworkRegistry {
        CacheNetworkRegistry {
            schema_version: 1,
            repositories: vec![
                GitHubRepositoryEndpoint {
                    shard_id: "private-001".to_owned(),
                    owner: "example-org".to_owned(),
                    repository: "restricted-cache".to_owned(),
                    branch: "main".to_owned(),
                    visibility: CacheVisibility::Private,
                    enabled_for_read: true,
                    enabled_for_write: true,
                    clone_via_ssh: false,
                },
                GitHubRepositoryEndpoint {
                    shard_id: "public-001".to_owned(),
                    owner: "example-org".to_owned(),
                    repository: "public-shard".to_owned(),
                    branch: "main".to_owned(),
                    visibility: CacheVisibility::Public,
                    enabled_for_read: true,
                    enabled_for_write: true,
                    clone_via_ssh: false,
                },
            ],
        }
    }

    fn encoding() -> TransportEncodingRecord {
        let bytes = b"encoded";
        let digest = ContentDigest::sha256(bytes);
        TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: ContentDigest::sha256(b"payload"),
            encoder_profile: "fixture-v1".to_owned(),
            package_size_bytes: bytes.len() as u64,
            package_digest: digest.clone(),
            ordered_parts: vec![TransportPart {
                sequence: 0,
                repository_path: format!("objects/{}.part", digest.0),
                size_bytes: bytes.len() as u64,
                content_digest: digest,
            }],
            reconstruction: "concatenate".to_owned(),
        }
    }

    fn input(
        policy: &CachePublicationPolicy,
        destination: PublicationDestination,
    ) -> TargetPublicationPlanningInput {
        let repository = match destination {
            PublicationDestination::Private => "example-org/restricted-cache",
            PublicationDestination::Public => "example-org/public-shard",
        };
        let required_sanitizer = destination == PublicationDestination::Public;
        let mut validators = vec![ValidatorEvidence {
            validator_id: "manifest".to_owned(),
            passed: true,
            evidence_digest: ContentDigest::sha256(b"manifest-evidence"),
            establishes_assurance: Some(ArtifactAssuranceState::CrossChecked),
        }];
        if required_sanitizer {
            validators.push(ValidatorEvidence {
                validator_id: "public-sanitizer".to_owned(),
                passed: true,
                evidence_digest: ContentDigest::sha256(b"sanitizer-evidence"),
                establishes_assurance: None,
            });
        }
        TargetPublicationPlanningInput {
            candidate: CachePublicationCandidate {
                semantic_digest: ContentDigest::sha256(b"semantic"),
                manifest_digest: ContentDigest::sha256(match destination {
                    PublicationDestination::Private => b"private-manifest",
                    PublicationDestination::Public => b"public-manifest",
                }),
                payload_digest: ContentDigest::sha256(b"payload"),
                completion: ArtifactCompletionState::Complete,
                achieved_assurance: ArtifactAssuranceState::CrossChecked,
                disposition: ArtifactDisposition::Active,
                validator_evidence: validators,
                public_metadata: BTreeMap::from([(
                    "artifact".to_owned(),
                    json!({"kind": "ccm_matrix"}),
                )]),
            },
            authorization: TargetPublicationAuthorizationRequest {
                destination,
                repository: repository.to_owned(),
                authority: PublicationAuthority {
                    principal: "test-owner".to_owned(),
                    mode: PublicationAuthorityMode::OwnerDirect,
                    allowed_targets: [PublicationTarget::Both].into_iter().collect(),
                    allowed_repositories: [repository.to_owned()].into_iter().collect(),
                    policy_digest: policy.digest().unwrap().0,
                },
                contributor: None,
                reviews: Vec::new(),
            },
            expected_head: "b".repeat(40),
            projected_metadata_bytes: 1_000,
            projected_history_bytes: 1_000,
        }
    }

    fn authenticated_sessions() -> BTreeMap<PublicationDestination, AuthenticatedGitHubSession> {
        BTreeMap::from([
            (
                PublicationDestination::Private,
                AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    "example-org/restricted-cache",
                    RepositoryPermission::Write,
                ),
            ),
            (
                PublicationDestination::Public,
                AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    "example-org/public-shard",
                    RepositoryPermission::Write,
                ),
            ),
        ])
    }

    #[test]
    fn dual_target_plan_uses_independent_manifests_and_one_family_topology() {
        let policy = policy();
        let inputs = BTreeMap::from([
            (
                PublicationDestination::Private,
                input(&policy, PublicationDestination::Private),
            ),
            (
                PublicationDestination::Public,
                input(&policy, PublicationDestination::Public),
            ),
        ]);
        let plan = coordinate_publication(
            "ccm",
            PublicationTarget::Both,
            &policy,
            &topology(&policy),
            &TopologyTrustPolicy {
                minimum_generation: 1,
                pinned_registry_digest: None,
                required_trust_anchor: Some("release-key".to_owned()),
            },
            &network(),
            &BTreeMap::from([
                ("private-001".to_owned(), ledger("private-001")),
                ("public-001".to_owned(), ledger("public-001")),
            ]),
            &encoding(),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 90,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
            &inputs,
            &authenticated_sessions(),
        )
        .unwrap();
        assert!(plan.authorized(), "{:?}", plan.target_reports);
        let journal = plan.journal.unwrap();
        assert_ne!(
            journal.target_manifest_digests[&PublicationDestination::Private],
            journal.target_manifest_digests[&PublicationDestination::Public]
        );
        assert_eq!(journal.targets.len(), 2);
    }

    #[test]
    fn one_rejected_target_prevents_all_remote_mutation_during_preflight() {
        let policy = policy();
        let private = input(&policy, PublicationDestination::Private);
        let mut public = input(&policy, PublicationDestination::Public);
        public.candidate.public_metadata.insert(
            "token".to_owned(),
            Value::String("github_pat_secret".to_owned()),
        );
        let inputs = BTreeMap::from([
            (PublicationDestination::Private, private),
            (PublicationDestination::Public, public),
        ]);
        let plan = coordinate_publication(
            "ccm",
            PublicationTarget::Both,
            &policy,
            &topology(&policy),
            &TopologyTrustPolicy {
                minimum_generation: 1,
                pinned_registry_digest: None,
                required_trust_anchor: None,
            },
            &network(),
            &BTreeMap::from([
                ("private-001".to_owned(), ledger("private-001")),
                ("public-001".to_owned(), ledger("public-001")),
            ]),
            &encoding(),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 90,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
            &inputs,
            &authenticated_sessions(),
        )
        .unwrap();
        assert!(plan.journal.is_none());
        assert!(plan.target_reports[&PublicationDestination::Private].accepted());
        assert!(!plan.target_reports[&PublicationDestination::Public].accepted());
    }

    #[test]
    fn one_read_only_target_prevents_dual_target_journal_creation() {
        let policy = policy();
        let inputs = BTreeMap::from([
            (
                PublicationDestination::Private,
                input(&policy, PublicationDestination::Private),
            ),
            (
                PublicationDestination::Public,
                input(&policy, PublicationDestination::Public),
            ),
        ]);
        let mut sessions = authenticated_sessions();
        sessions.insert(
            PublicationDestination::Public,
            AuthenticatedGitHubSession::verified_for_test(
                "test-owner",
                "example-org/public-shard",
                RepositoryPermission::Read,
            ),
        );
        let plan = coordinate_publication(
            "ccm",
            PublicationTarget::Both,
            &policy,
            &topology(&policy),
            &TopologyTrustPolicy {
                minimum_generation: 1,
                pinned_registry_digest: None,
                required_trust_anchor: None,
            },
            &network(),
            &BTreeMap::from([
                ("private-001".to_owned(), ledger("private-001")),
                ("public-001".to_owned(), ledger("public-001")),
            ]),
            &encoding(),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 90,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
            &inputs,
            &sessions,
        )
        .unwrap();
        assert!(plan.journal.is_none());
        assert!(plan.target_reports[&PublicationDestination::Private].accepted());
        assert!(!plan.target_reports[&PublicationDestination::Public].accepted());
        assert!(plan.target_reports[&PublicationDestination::Public]
            .authorization
            .reasons
            .iter()
            .any(|reason| reason.contains("permission")));
    }

    struct PlanningRemote {
        heads: BTreeMap<(String, String), String>,
        documents: BTreeMap<(String, String, String), Vec<u8>>,
    }

    impl RemoteGitStore for PlanningRemote {
        fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError> {
            self.heads
                .get(&(repository.to_owned(), branch.to_owned()))
                .cloned()
                .ok_or_else(|| CacheError::NotFound(format!("{repository}:{branch}")))
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            _revision: &str,
            _path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(None)
        }

        fn read_committed_path(
            &self,
            repository: &str,
            revision: &str,
            path: &str,
            maximum_bytes: u64,
            cancellation: &CancellationToken,
            writer: &mut dyn std::io::Write,
        ) -> Result<RemoteReadReport, CacheError> {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let bytes = self
                .documents
                .get(&(repository.to_owned(), revision.to_owned(), path.to_owned()))
                .ok_or_else(|| CacheError::NotFound(format!("{revision}:{path}")))?;
            if bytes.len() as u64 > maximum_bytes {
                return Err(CacheError::ResourceLimit(
                    "planning fixture exceeds read limit".to_owned(),
                ));
            }
            writer.write_all(bytes)?;
            Ok(RemoteReadReport {
                repository_path: path.to_owned(),
                revision: revision.to_owned(),
                size_bytes: bytes.len() as u64,
                content_digest: ContentDigest::sha256(bytes),
            })
        }

        fn compare_and_swap_commit(
            &self,
            _request: &crate::RemoteCommitRequest,
        ) -> Result<crate::CompareAndSwapResult, CacheError> {
            panic!("remote routing discovery must not mutate repositories")
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            _revision: &str,
            _part: &TransportPart,
        ) -> Result<(), CacheError> {
            panic!("remote routing discovery must not verify payloads")
        }
    }

    #[test]
    fn remote_routing_reads_only_topology_and_shard_local_ledgers() {
        let policy = policy();
        let topology = TopologyRegistry {
            schema_version: 1,
            generation: 7,
            previous_registry_digest: Some(ContentDigest::sha256(b"generation-6")),
            policy_digest: policy.digest().unwrap(),
            trust_anchor_ids: vec!["release-key".to_owned()],
            family_routes: vec![ArtifactFamilyRoute {
                family: "ccm".to_owned(),
                visibility: CacheVisibility::Private,
                ordered_shards: vec![
                    TopologyShardRoute {
                        shard_id: "private-001".to_owned(),
                        endpoint_id: "private-001".to_owned(),
                        sequence: 1,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: Some("private-002".to_owned()),
                    },
                    TopologyShardRoute {
                        shard_id: "private-002".to_owned(),
                        endpoint_id: "private-002".to_owned(),
                        sequence: 2,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: None,
                    },
                ],
            }],
        };
        let network = CacheNetworkRegistry {
            schema_version: 1,
            repositories: [
                ("private-001", "restricted-cache-001"),
                ("private-002", "restricted-cache-002"),
            ]
            .into_iter()
            .map(|(shard_id, repository)| GitHubRepositoryEndpoint {
                shard_id: shard_id.to_owned(),
                owner: "example-org".to_owned(),
                repository: repository.to_owned(),
                branch: "main".to_owned(),
                visibility: CacheVisibility::Private,
                enabled_for_read: true,
                enabled_for_write: true,
                clone_via_ssh: false,
            })
            .collect(),
        };
        let topology_repository = "example-org/topology".to_owned();
        let topology_head = "a".repeat(40);
        let first_url = network.repositories[0].preferred_clone_url();
        let second_url = network.repositories[1].preferred_clone_url();
        let first_head = "b".repeat(40);
        let second_head = "c".repeat(40);
        let trust_root = |role, repository: &str, revision: &str| {
            let protection = crate::ProtectedBranchStatement {
                schema_version: 1,
                repository: repository.to_owned(),
                branch: "main".to_owned(),
                observed_revision: revision.to_owned(),
                force_pushes_prohibited: true,
                pushes_restricted: true,
                observed_at_unix_seconds: 100,
                valid_until_unix_seconds: 300,
                trust_anchor_id: "release-key".to_owned(),
            };
            crate::TrustedRepositoryRoot {
                role,
                repository: repository.to_owned(),
                owner: repository
                    .split_once('/')
                    .map_or(repository, |(owner, _)| owner)
                    .to_owned(),
                branch: "main".to_owned(),
                revision_policy: crate::TrustedRevisionPolicy::Exact {
                    revision: revision.to_owned(),
                },
                branch_protection_digest: protection.digest().unwrap(),
                branch_protection: protection,
            }
        };
        let fabric_trust = crate::RemoteFabricTrustPolicy {
            schema_version: 1,
            approved_trust_anchor_ids: ["release-key".to_owned()].into_iter().collect(),
            approved_policy_digests: [topology.policy_digest.clone()].into_iter().collect(),
            repositories: vec![
                trust_root(
                    crate::TrustedRepositoryRole::Registry,
                    &topology_repository,
                    &topology_head,
                ),
                trust_root(
                    crate::TrustedRepositoryRole::Shard,
                    "example-org/restricted-cache-001",
                    &first_head,
                ),
                trust_root(
                    crate::TrustedRepositoryRole::Shard,
                    "example-org/restricted-cache-002",
                    &second_head,
                ),
            ],
        };
        let mut first_ledger = ledger("private-001");
        first_ledger.first_seen_immutable_payload_bytes = 60_000_000_000;
        let second_ledger = ledger("private-002");
        let remote = PlanningRemote {
            heads: BTreeMap::from([
                (
                    (topology_repository.clone(), "main".to_owned()),
                    topology_head.clone(),
                ),
                ((first_url.clone(), "main".to_owned()), first_head.clone()),
                ((second_url.clone(), "main".to_owned()), second_head.clone()),
            ]),
            documents: BTreeMap::from([
                (
                    (
                        topology_repository.clone(),
                        topology_head.clone(),
                        "registry/topology.json".to_owned(),
                    ),
                    serde_json::to_vec(&topology).unwrap(),
                ),
                (
                    (
                        first_url,
                        first_head.clone(),
                        DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                    ),
                    serde_json::to_vec(&first_ledger).unwrap(),
                ),
                (
                    (
                        second_url,
                        second_head,
                        DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                    ),
                    serde_json::to_vec(&second_ledger).unwrap(),
                ),
            ]),
        };
        let routing = discover_remote_publication_routing(
            &remote,
            &CancellationToken::new(),
            "ccm",
            PublicationTarget::Private,
            &RemoteTopologySource {
                repository: topology_repository.clone(),
                ..RemoteTopologySource::default()
            },
            &TopologyTrustPolicy {
                minimum_generation: 7,
                pinned_registry_digest: Some(topology.digest().unwrap()),
                required_trust_anchor: Some("release-key".to_owned()),
            },
            &fabric_trust,
            200,
            &network,
            &BTreeMap::from([(
                PublicationDestination::Private,
                ProjectedPublicationAddition {
                    unique_payload_bytes: 30_000_000_000,
                    metadata_bytes: 1_000,
                    projected_history_bytes: 1_000,
                },
            )]),
        )
        .unwrap();
        let selected = &routing.selections[&PublicationDestination::Private];
        assert_eq!(selected.shard_id, "private-001");
        assert_eq!(selected.expected_head, first_head);
        assert_eq!(routing.ledgers.len(), 2);
        assert_eq!(routing.ledger_sources.len(), 2);

        let ordinary_input = input(&policy, PublicationDestination::Private);
        let target_inputs = BTreeMap::from([(
            PublicationDestination::Private,
            RemoteTargetPublicationPlanningInput {
                candidate: ordinary_input.candidate,
                authority: PublicationAuthority {
                    principal: "test-owner".to_owned(),
                    mode: PublicationAuthorityMode::OwnerDirect,
                    allowed_targets: [PublicationTarget::Private].into_iter().collect(),
                    allowed_repositories: [selected.authorized_repository.clone()]
                        .into_iter()
                        .collect(),
                    policy_digest: policy.digest().unwrap().0,
                },
                contributor: None,
                reviews: Vec::new(),
                projected_metadata_bytes: 1_000,
                projected_history_bytes: 1_000,
            },
        )]);
        let sessions = BTreeMap::from([(
            PublicationDestination::Private,
            AuthenticatedGitHubSession::verified_for_test(
                "test-owner",
                &selected.authorized_repository,
                RepositoryPermission::Write,
            ),
        )]);
        let trust = TopologyTrustPolicy {
            minimum_generation: 7,
            pinned_registry_digest: Some(topology.digest().unwrap()),
            required_trust_anchor: Some("release-key".to_owned()),
        };
        let coordinated = coordinate_discovered_publication(
            &routing,
            &policy,
            &trust,
            &network,
            &encoding(),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 90,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
            &target_inputs,
            &sessions,
        )
        .unwrap();
        let journal = coordinated.journal.unwrap();
        let validation = validate_remote_publication_routes(
            &remote,
            &CancellationToken::new(),
            &journal,
            "ccm",
            &RemoteTopologySource {
                repository: topology_repository,
                ..RemoteTopologySource::default()
            },
            &trust,
            &fabric_trust,
            200,
            &network,
        )
        .unwrap();
        assert_eq!(
            validation.validated_targets[&PublicationDestination::Private],
            "private-001"
        );
    }
}
