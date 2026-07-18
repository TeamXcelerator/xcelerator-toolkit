//! Verified shard-successor activation and atomic topology rollover.

use crate::protocol::canonical_json_bytes;
use crate::publication_staging::stage_publication_bytes;
use crate::{
    AuthenticatedGitHubSession, CacheError, CacheVisibility, CompareAndSwapResult, ContentDigest,
    RemoteCommitRequest, RemoteGitStore, RemoteShardAuditReport, RemoteShardReader,
    TopologyRegistry, TopologyShardStatus, TopologyTrustPolicy, DEFAULT_CAPACITY_LEDGER_PATH,
    GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuccessorShardReadinessEvidence {
    pub schema_version: u32,
    pub endpoint_id: String,
    pub repository_owner: String,
    pub repository_name: String,
    pub visibility: CacheVisibility,
    pub ownership_and_visibility_evidence_digest: ContentDigest,
    pub branch_protection_evidence_digest: ContentDigest,
    pub trust_metadata_evidence_digest: ContentDigest,
    pub read_write_health_evidence_digest: ContentDigest,
    pub reviewer_approval_digest: ContentDigest,
    pub audited_at_unix_seconds: u64,
    pub audit: RemoteShardAuditReport,
}

impl SuccessorShardReadinessEvidence {
    pub fn validate(&self) -> Result<ContentDigest, CacheError> {
        let evidence_digests = [
            &self.ownership_and_visibility_evidence_digest,
            &self.branch_protection_evidence_digest,
            &self.trust_metadata_evidence_digest,
            &self.read_write_health_evidence_digest,
            &self.reviewer_approval_digest,
        ];
        let ledger = self.audit.capacity_ledger.as_ref().ok_or_else(|| {
            CacheError::InvalidManifest("successor audit has no capacity ledger".to_owned())
        })?;
        ledger.validate()?;
        if self.schema_version != 1
            || self.endpoint_id.trim().is_empty()
            || self.repository_owner.trim().is_empty()
            || self.repository_name.trim().is_empty()
            || evidence_digests.iter().any(|digest| !digest.validate())
            || self.audited_at_unix_seconds == 0
            || self.audit.schema_version != 1
            || self.audit.repository.trim().is_empty()
            || self.audit.branch.trim().is_empty()
            || self.audit.revision.trim().is_empty()
            || self.audit.shard_id.trim().is_empty()
            || !self.audit.issues.is_empty()
            || self.audit.index_entry_count != 0
            || self.audit.complete_artifact_count != 0
            || self.audit.logical_payload_bytes != 0
            || self.audit.unique_transport_object_count != 0
            || self.audit.unique_transport_object_bytes != 0
            || self.audit.unreferenced_manifest_count != 0
            || self.audit.unreferenced_encoding_count != 0
            || self.audit.unreferenced_receipt_count != 0
            || self.audit.unreferenced_object_count != 0
            || !self.audit.reconstructed_partitions.is_empty()
            || ledger.shard_id != self.audit.shard_id
            || ledger.hard_capacity_bytes != GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES
            || ledger.first_seen_immutable_payload_bytes != 0
            || ledger.abandoned_reachable_bytes != 0
            || ledger.last_reconciled_commit != self.audit.revision
            || !self.audit.capacity.ledger_covers_referenced_transport
            || !self.audit.capacity.durable_record_coverage_complete
            || !self.audit.capacity.ledger_matches_durable_payload_history
            || !self.audit.capacity.exact_history_rebuild_available
        {
            return Err(CacheError::InvalidManifest(
                "successor readiness does not prove an empty, governed, healthy shard".to_owned(),
            ));
        }
        Ok(ContentDigest::sha256(&canonical_json_bytes(self)?))
    }

    pub fn authorized_repository(&self) -> String {
        format!("{}/{}", self.repository_owner, self.repository_name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyRolloverPlan {
    pub schema_version: u32,
    pub authorized_registry_repository: String,
    pub registry_repository: String,
    pub registry_branch: String,
    pub registry_path: String,
    pub expected_head: String,
    pub observed_registry_digest: ContentDigest,
    pub prior_topology_digest: ContentDigest,
    pub replacement_registry_digest: ContentDigest,
    pub replacement_size_bytes: u64,
    pub family: String,
    pub visibility: CacheVisibility,
    pub prior_writable_shard_id: String,
    pub successor_shard_id: String,
    pub successor_readiness_digest: ContentDigest,
    pub replacement: TopologyRegistry,
}

impl TopologyRolloverPlan {
    pub fn validate(&self) -> Result<(), CacheError> {
        self.replacement.validate()?;
        let bytes = canonical_json_bytes(&self.replacement)?;
        if self.schema_version != 1
            || self.authorized_registry_repository.trim().is_empty()
            || self.registry_repository.trim().is_empty()
            || self.registry_branch.trim().is_empty()
            || self.registry_path.trim().is_empty()
            || self.expected_head.trim().is_empty()
            || self.family.trim().is_empty()
            || self.prior_writable_shard_id.trim().is_empty()
            || self.successor_shard_id.trim().is_empty()
            || !self.observed_registry_digest.validate()
            || !self.prior_topology_digest.validate()
            || !self.successor_readiness_digest.validate()
            || self.replacement_registry_digest != ContentDigest::sha256(&bytes)
            || self.replacement_size_bytes != bytes.len() as u64
            || self.replacement.previous_registry_digest.as_ref()
                != Some(&self.prior_topology_digest)
        {
            return Err(CacheError::InvalidManifest(
                "topology rollover plan is incomplete or digest-inconsistent".to_owned(),
            ));
        }
        let route = self
            .replacement
            .route(&self.family, self.visibility)
            .ok_or_else(|| CacheError::InvalidManifest("replacement route is absent".to_owned()))?;
        let prior = route
            .ordered_shards
            .iter()
            .find(|shard| shard.shard_id == self.prior_writable_shard_id);
        let successor = route
            .ordered_shards
            .iter()
            .find(|shard| shard.shard_id == self.successor_shard_id);
        if prior.is_none_or(|shard| {
            shard.status != TopologyShardStatus::ReadOnly
                || shard.successor_shard_id.as_deref() != Some(&self.successor_shard_id)
        }) || successor.is_none_or(|shard| shard.status != TopologyShardStatus::Writable)
        {
            return Err(CacheError::InvalidTransition(
                "rollover replacement does not switch the declared successor".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        Ok(ContentDigest::sha256(&canonical_json_bytes(self)?))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum TopologyRolloverOutcome {
    RefConflict {
        current_head: String,
    },
    CommittedAndVerified {
        commit_id: String,
        plan_digest: ContentDigest,
        principal: String,
        repository_permission_evidence_digest: ContentDigest,
        replacement_registry_digest: ContentDigest,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn plan_topology_rollover(
    remote: &dyn RemoteGitStore,
    authorized_registry_repository: &str,
    registry_repository: &str,
    registry_branch: &str,
    registry_path: &str,
    maximum_registry_bytes: u64,
    topology_trust: &TopologyTrustPolicy,
    family: &str,
    visibility: CacheVisibility,
    prior_writable_shard_id: &str,
    successor: &SuccessorShardReadinessEvidence,
    cancellation: &CancellationToken,
) -> Result<TopologyRolloverPlan, CacheError> {
    let successor_readiness_digest = successor.validate()?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let successor_head = remote.read_ref(&successor.audit.repository, &successor.audit.branch)?;
    if successor_head != successor.audit.revision {
        return Err(CacheError::InvalidTransition(
            "successor shard advanced after its readiness audit".to_owned(),
        ));
    }
    let ledger = successor
        .audit
        .capacity_ledger
        .as_ref()
        .expect("readiness validation requires a ledger");
    let expected_ledger_digest = ContentDigest::sha256(&canonical_json_bytes(ledger)?);
    let current_ledger_digest = remote.immutable_path_digest(
        &successor.audit.repository,
        &successor.audit.revision,
        DEFAULT_CAPACITY_LEDGER_PATH,
    )?;
    if current_ledger_digest.as_ref() != Some(&expected_ledger_digest) {
        return Err(CacheError::DigestMismatch {
            expected: expected_ledger_digest.to_string(),
            actual: current_ledger_digest
                .map_or_else(|| "missing".to_owned(), |digest| digest.to_string()),
        });
    }
    let expected_head = remote.read_ref(registry_repository, registry_branch)?;
    let document = RemoteShardReader::new(remote, maximum_registry_bytes)?
        .read_json::<TopologyRegistry>(
            registry_repository,
            &expected_head,
            registry_path,
            cancellation,
        )?;
    let prior_topology_digest = topology_trust.verify(&document.value)?;
    if document.source.content_digest != prior_topology_digest {
        return Err(CacheError::DigestMismatch {
            expected: prior_topology_digest.to_string(),
            actual: document.source.content_digest.to_string(),
        });
    }
    let mut replacement = document.value;
    let route = replacement.route(family, visibility).ok_or_else(|| {
        CacheError::InvalidTransition("rollover family route is absent".to_owned())
    })?;
    let prior_index = route
        .ordered_shards
        .iter()
        .position(|shard| shard.shard_id == prior_writable_shard_id)
        .ok_or_else(|| {
            CacheError::InvalidTransition("prior writable shard is absent".to_owned())
        })?;
    let successor_id = route.ordered_shards[prior_index]
        .successor_shard_id
        .clone()
        .ok_or_else(|| CacheError::InvalidTransition("prior shard has no successor".to_owned()))?;
    let successor_index = route
        .ordered_shards
        .iter()
        .position(|shard| shard.shard_id == successor_id)
        .ok_or_else(|| CacheError::InvalidTransition("declared successor is absent".to_owned()))?;
    let prior = &route.ordered_shards[prior_index];
    let next = &route.ordered_shards[successor_index];
    if prior.status != TopologyShardStatus::Writable
        || next.status != TopologyShardStatus::ReadOnly
        || next.sequence != prior.sequence.saturating_add(1)
        || next.endpoint_id != successor.endpoint_id
        || next.shard_id != successor.audit.shard_id
        || successor.visibility != visibility
    {
        return Err(CacheError::InvalidTransition(
            "successor identity, order, visibility, or preactivation state is invalid".to_owned(),
        ));
    }
    replacement.generation = replacement
        .generation
        .checked_add(1)
        .ok_or_else(|| CacheError::InvalidTransition("topology generation overflow".to_owned()))?;
    replacement.previous_registry_digest = Some(prior_topology_digest.clone());
    let route = replacement
        .family_routes
        .iter_mut()
        .find(|route| route.family == family && route.visibility == visibility)
        .expect("route was checked");
    route.ordered_shards[prior_index].status = TopologyShardStatus::ReadOnly;
    route.ordered_shards[successor_index].status = TopologyShardStatus::Writable;
    replacement.validate()?;
    let bytes = canonical_json_bytes(&replacement)?;
    if bytes.len() as u64 > maximum_registry_bytes {
        return Err(CacheError::ResourceLimit(
            "replacement topology exceeds its configured byte bound".to_owned(),
        ));
    }
    let plan = TopologyRolloverPlan {
        schema_version: 1,
        authorized_registry_repository: authorized_registry_repository.to_owned(),
        registry_repository: registry_repository.to_owned(),
        registry_branch: registry_branch.to_owned(),
        registry_path: registry_path.to_owned(),
        expected_head,
        observed_registry_digest: document.source.content_digest,
        prior_topology_digest,
        replacement_registry_digest: ContentDigest::sha256(&bytes),
        replacement_size_bytes: bytes.len() as u64,
        family: family.to_owned(),
        visibility,
        prior_writable_shard_id: prior_writable_shard_id.to_owned(),
        successor_shard_id: successor_id,
        successor_readiness_digest,
        replacement,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn execute_topology_rollover(
    remote: &dyn RemoteGitStore,
    session: &AuthenticatedGitHubSession,
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    plan: &TopologyRolloverPlan,
) -> Result<TopologyRolloverOutcome, CacheError> {
    plan.validate()?;
    session.require_write_for(
        session.evidence().principal.as_str(),
        &plan.authorized_registry_repository,
    )?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let current_head = remote.read_ref(&plan.registry_repository, &plan.registry_branch)?;
    if current_head != plan.expected_head {
        return Ok(TopologyRolloverOutcome::RefConflict { current_head });
    }
    let current_digest = remote
        .immutable_path_digest(
            &plan.registry_repository,
            &plan.expected_head,
            &plan.registry_path,
        )?
        .ok_or_else(|| CacheError::NotFound(plan.registry_path.clone()))?;
    if current_digest != plan.observed_registry_digest {
        return Err(CacheError::DigestMismatch {
            expected: plan.observed_registry_digest.to_string(),
            actual: current_digest.to_string(),
        });
    }
    let bytes = canonical_json_bytes(&plan.replacement)?;
    let mut staged_bytes = 0;
    let part = stage_publication_bytes(
        staging_root,
        &plan.registry_path,
        &bytes,
        resources,
        cancellation,
        &mut staged_bytes,
    )?;
    let plan_digest = plan.digest()?;
    let request = RemoteCommitRequest {
        repository: plan.registry_repository.clone(),
        branch: plan.registry_branch.clone(),
        expected_head: plan.expected_head.clone(),
        message: format!("activate cache shard successor {plan_digest}"),
        parts: vec![part.clone()],
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            Ok(TopologyRolloverOutcome::RefConflict { current_head })
        }
        CompareAndSwapResult::Committed { commit_id } => {
            remote.verify_committed_part(&plan.registry_repository, &commit_id, &part)?;
            Ok(TopologyRolloverOutcome::CommittedAndVerified {
                commit_id,
                plan_digest,
                principal: session.evidence().principal.clone(),
                repository_permission_evidence_digest: session.evidence().evidence_digest.clone(),
                replacement_registry_digest: plan.replacement_registry_digest.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArtifactFamilyRoute, CapacityLedger, RemoteReadReport, RepositoryPermission,
        ShardCapacityAudit, TopologyShardRoute,
    };
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::Mutex;

    struct MemoryRemote {
        heads: Mutex<BTreeMap<String, String>>,
        documents: Mutex<BTreeMap<(String, String), Vec<u8>>>,
        digests: Mutex<BTreeMap<(String, String), ContentDigest>>,
    }

    impl RemoteGitStore for MemoryRemote {
        fn read_ref(&self, repository: &str, _branch: &str) -> Result<String, CacheError> {
            self.heads
                .lock()
                .unwrap()
                .get(repository)
                .cloned()
                .ok_or_else(|| CacheError::NotFound(repository.to_owned()))
        }

        fn immutable_path_digest(
            &self,
            repository: &str,
            _revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(self
                .digests
                .lock()
                .unwrap()
                .get(&(repository.to_owned(), path.to_owned()))
                .cloned())
        }

        fn read_committed_path(
            &self,
            repository: &str,
            revision: &str,
            path: &str,
            maximum_bytes: u64,
            _cancellation: &CancellationToken,
            writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            let documents = self.documents.lock().unwrap();
            let bytes = documents
                .get(&(repository.to_owned(), path.to_owned()))
                .ok_or_else(|| CacheError::NotFound(path.to_owned()))?;
            if bytes.len() as u64 > maximum_bytes {
                return Err(CacheError::ResourceLimit("test read bound".to_owned()));
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
            request: &RemoteCommitRequest,
        ) -> Result<CompareAndSwapResult, CacheError> {
            let mut heads = self.heads.lock().unwrap();
            let head = heads
                .get_mut(&request.repository)
                .ok_or_else(|| CacheError::NotFound(request.repository.clone()))?;
            if head != &request.expected_head {
                return Ok(CompareAndSwapResult::RefConflict {
                    current_head: head.clone(),
                });
            }
            for part in &request.parts {
                self.digests.lock().unwrap().insert(
                    (request.repository.clone(), part.repository_path.clone()),
                    part.content_digest.clone(),
                );
            }
            *head = "registry-commit-2".to_owned();
            Ok(CompareAndSwapResult::Committed {
                commit_id: head.clone(),
            })
        }

        fn verify_committed_part(
            &self,
            repository: &str,
            revision: &str,
            part: &crate::TransportPart,
        ) -> Result<(), CacheError> {
            if revision != "registry-commit-2"
                || self
                    .digests
                    .lock()
                    .unwrap()
                    .get(&(repository.to_owned(), part.repository_path.clone()))
                    != Some(&part.content_digest)
            {
                return Err(CacheError::DigestMismatch {
                    expected: part.content_digest.to_string(),
                    actual: "missing".to_owned(),
                });
            }
            Ok(())
        }
    }

    fn topology() -> TopologyRegistry {
        TopologyRegistry {
            schema_version: 1,
            generation: 7,
            previous_registry_digest: Some(ContentDigest::sha256(b"generation-6")),
            policy_digest: ContentDigest::sha256(b"policy"),
            trust_anchor_ids: vec!["release-root".to_owned()],
            family_routes: vec![ArtifactFamilyRoute {
                family: "ccm".to_owned(),
                visibility: CacheVisibility::Private,
                ordered_shards: vec![
                    TopologyShardRoute {
                        shard_id: "ccm-private-001".to_owned(),
                        endpoint_id: "ccm-private-001".to_owned(),
                        sequence: 1,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: Some("ccm-private-002".to_owned()),
                    },
                    TopologyShardRoute {
                        shard_id: "ccm-private-002".to_owned(),
                        endpoint_id: "ccm-private-002".to_owned(),
                        sequence: 2,
                        status: TopologyShardStatus::ReadOnly,
                        successor_shard_id: None,
                    },
                ],
            }],
        }
    }

    fn readiness() -> SuccessorShardReadinessEvidence {
        let ledger = CapacityLedger {
            schema_version: 1,
            shard_id: "ccm-private-002".to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 5_000_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 256,
            estimated_history_bytes: 128,
            emergency_reserve_bytes: 1_000_000_000,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: "successor-head".to_owned(),
            reconciliation_digest: ContentDigest::sha256(b"empty-successor-audit"),
        };
        SuccessorShardReadinessEvidence {
            schema_version: 1,
            endpoint_id: "ccm-private-002".to_owned(),
            repository_owner: "TeamXcelerator".to_owned(),
            repository_name: "ccm-private-002".to_owned(),
            visibility: CacheVisibility::Private,
            ownership_and_visibility_evidence_digest: ContentDigest::sha256(b"ownership"),
            branch_protection_evidence_digest: ContentDigest::sha256(b"protection"),
            trust_metadata_evidence_digest: ContentDigest::sha256(b"trust"),
            read_write_health_evidence_digest: ContentDigest::sha256(b"read-write"),
            reviewer_approval_digest: ContentDigest::sha256(b"review"),
            audited_at_unix_seconds: 100,
            audit: RemoteShardAuditReport {
                schema_version: 1,
                repository: "successor-repository".to_owned(),
                branch: "main".to_owned(),
                revision: "successor-head".to_owned(),
                shard_id: "ccm-private-002".to_owned(),
                listed_path_count: 1,
                listed_path_bytes: 20,
                verified_metadata_bytes_read: 256,
                index_partition_count: 0,
                index_entry_count: 0,
                complete_artifact_count: 0,
                logical_payload_bytes: 0,
                unique_transport_object_count: 0,
                unique_transport_object_bytes: 0,
                unreferenced_manifest_count: 0,
                unreferenced_encoding_count: 0,
                unreferenced_receipt_count: 0,
                unreferenced_object_count: 0,
                reconstructed_partitions: Vec::new(),
                capacity_ledger: Some(ledger),
                capacity: ShardCapacityAudit {
                    ledger_accounted_bytes: Some(1_000_000_384),
                    ledger_first_seen_immutable_payload_bytes: Some(0),
                    referenced_unique_transport_bytes: 0,
                    ledger_covers_referenced_transport: true,
                    ledger_remaining_capacity_bytes: Some(98_999_999_616),
                    durable_batch_record_count: 0,
                    durable_recorded_new_payload_bytes: 0,
                    durable_record_coverage_complete: true,
                    ledger_matches_durable_payload_history: true,
                    exact_history_rebuild_available: true,
                },
                issues: Vec::new(),
            },
        }
    }

    fn remote(
        topology: &TopologyRegistry,
        readiness: &SuccessorShardReadinessEvidence,
    ) -> MemoryRemote {
        let registry_bytes = canonical_json_bytes(topology).unwrap();
        let ledger = readiness.audit.capacity_ledger.as_ref().unwrap();
        let ledger_bytes = canonical_json_bytes(ledger).unwrap();
        MemoryRemote {
            heads: Mutex::new(BTreeMap::from([
                ("registry-repository".to_owned(), "registry-head".to_owned()),
                (
                    readiness.audit.repository.clone(),
                    readiness.audit.revision.clone(),
                ),
            ])),
            documents: Mutex::new(BTreeMap::from([(
                ("registry-repository".to_owned(), "registry.json".to_owned()),
                registry_bytes.clone(),
            )])),
            digests: Mutex::new(BTreeMap::from([
                (
                    ("registry-repository".to_owned(), "registry.json".to_owned()),
                    ContentDigest::sha256(&registry_bytes),
                ),
                (
                    (
                        readiness.audit.repository.clone(),
                        DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                    ),
                    ContentDigest::sha256(&ledger_bytes),
                ),
            ])),
        }
    }

    #[test]
    fn dirty_successor_cannot_be_activated() {
        let mut evidence = readiness();
        evidence.audit.unreferenced_object_count = 1;
        assert!(evidence.validate().is_err());
    }

    #[test]
    fn verified_successor_rollover_is_one_reviewed_compare_and_swap() {
        let topology = topology();
        let readiness = readiness();
        let remote = remote(&topology, &readiness);
        let plan = plan_topology_rollover(
            &remote,
            "example-org/restricted-registry",
            "registry-repository",
            "main",
            "registry.json",
            1_000_000,
            &TopologyTrustPolicy {
                minimum_generation: 7,
                pinned_registry_digest: Some(topology.digest().unwrap()),
                required_trust_anchor: Some("release-root".to_owned()),
            },
            "ccm",
            CacheVisibility::Private,
            "ccm-private-001",
            &readiness,
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(plan.replacement.generation, 8);
        let route = plan
            .replacement
            .route("ccm", CacheVisibility::Private)
            .unwrap();
        assert_eq!(
            route.ordered_shards[0].status,
            TopologyShardStatus::ReadOnly
        );
        assert_eq!(
            route.ordered_shards[1].status,
            TopologyShardStatus::Writable
        );
        let session = AuthenticatedGitHubSession::verified_for_test(
            "TeamXcelerator",
            "example-org/restricted-registry",
            RepositoryPermission::Admin,
        );
        let staging = std::env::temp_dir().join(format!("xc-rollover-{}", plan.digest().unwrap()));
        let outcome = execute_topology_rollover(
            &remote,
            &session,
            &staging,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &plan,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            TopologyRolloverOutcome::CommittedAndVerified { .. }
        ));
        let _ = std::fs::remove_dir_all(staging);
    }
}
