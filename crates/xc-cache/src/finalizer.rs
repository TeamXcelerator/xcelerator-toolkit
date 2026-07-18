//! Resumable metadata, shard-discoverability, ledger, and receipt finalization.

use crate::protocol::{canonical_digest, canonical_json_bytes, normalized_relative_path};
use crate::{
    AuthenticatedGitHubSession, CacheError, CapacityAdmission, CapacityLedger,
    CompareAndSwapResult, ContentDigest, PublicationCommitState, PublicationDestination,
    PublicationJournalStore, PublicationReviewEvidence, PublicationTargetState,
    PublicationTransactionJournal, RemoteCommitRequest, RemoteGitStore, TransportPart,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use xc_core::{CancellationToken, PublicationAuthorityMode};

pub const DEFAULT_CAPACITY_LEDGER_PATH: &str = "ledger/capacity.json";
pub const DEFAULT_MAXIMUM_CAPACITY_LEDGER_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationFinalizationPolicy {
    pub capacity_ledger_path: String,
    pub maximum_capacity_ledger_bytes: u64,
    pub projected_history_bytes: u64,
}

impl Default for PublicationFinalizationPolicy {
    fn default() -> Self {
        Self {
            capacity_ledger_path: DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
            maximum_capacity_ledger_bytes: DEFAULT_MAXIMUM_CAPACITY_LEDGER_BYTES,
            projected_history_bytes: 0,
        }
    }
}

impl PublicationFinalizationPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if !normalized_relative_path(&self.capacity_ledger_path)
            || self.maximum_capacity_ledger_bytes == 0
        {
            return Err(CacheError::InvalidManifest(
                "publication finalization capacity policy is invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Immutable, per-target evidence. It names the payload and metadata commits;
/// the later discoverability commit contains these bytes together with the
/// updated shard index and capacity ledger, so the receipt need not contain
/// its own (self-referential) Git commit identifier.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationReceipt {
    pub schema_version: u32,
    pub transaction_id: String,
    pub idempotency_key: ContentDigest,
    pub destination: PublicationDestination,
    pub principal: String,
    pub authorized_repository: String,
    pub repository_permission_evidence_digest: ContentDigest,
    pub shard_id: String,
    pub branch: String,
    pub semantic_digest: ContentDigest,
    pub canonical_payload_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub policy_digest: ContentDigest,
    pub policy_id: String,
    pub authority_mode: PublicationAuthorityMode,
    pub validation_evidence_digests: Vec<ContentDigest>,
    pub contributor_authorization_digest: Option<ContentDigest>,
    pub reviewer_approvals: Vec<PublicationReviewEvidence>,
    pub payload_commit_ids: Vec<String>,
    #[serde(default)]
    pub payload_batch_record_commit_ids: Vec<String>,
    #[serde(default)]
    pub payload_batch_record_digests: BTreeMap<String, ContentDigest>,
    pub metadata_commit_id: String,
    pub metadata_file_digests: BTreeMap<String, ContentDigest>,
    pub discoverability_subject_digests: BTreeMap<String, ContentDigest>,
    pub remote_verification_results: Vec<RemoteCommitVerificationResult>,
    pub verified_at_unix_seconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteCommitVerificationResult {
    pub phase: String,
    pub sequence: u64,
    pub commit_id: String,
    pub verified: bool,
    pub content_digests: Vec<ContentDigest>,
}

impl PublicationReceipt {
    pub fn from_verified_metadata(
        journal: &PublicationTransactionJournal,
        destination: PublicationDestination,
        transport_digest: ContentDigest,
        discoverability_subject_digests: BTreeMap<String, ContentDigest>,
        verified_at_unix_seconds: u64,
    ) -> Result<Self, CacheError> {
        journal.validate()?;
        if !transport_digest.validate() || discoverability_subject_digests.is_empty() {
            return Err(CacheError::InvalidManifest(
                "publication receipt transport and discoverability identities are required"
                    .to_owned(),
            ));
        }
        if discoverability_subject_digests
            .iter()
            .any(|(path, digest)| !normalized_relative_path(path) || !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "publication receipt contains an invalid discoverability subject".to_owned(),
            ));
        }
        let target = journal.targets.get(&destination).ok_or_else(|| {
            CacheError::InvalidTransition(format!(
                "transaction has no {destination:?} receipt target"
            ))
        })?;
        if target.state != PublicationTargetState::RemoteVerified {
            return Err(CacheError::InvalidTransition(
                "receipt construction requires remotely verified immutable metadata".to_owned(),
            ));
        }
        let metadata = target.metadata_commit.as_ref().ok_or_else(|| {
            CacheError::InvalidTransition("publication metadata was not planned".to_owned())
        })?;
        if metadata.state != PublicationCommitState::RemoteVerified {
            return Err(CacheError::InvalidTransition(
                "publication metadata is not remotely verified".to_owned(),
            ));
        }
        let metadata_commit_id = metadata.commit_id.clone().ok_or_else(|| {
            CacheError::InvalidTransition("verified metadata has no commit identity".to_owned())
        })?;
        let audit = target.audit_evidence.as_ref().ok_or_else(|| {
            CacheError::InvalidTransition(
                "receipt construction requires target authority and validation evidence".to_owned(),
            )
        })?;
        audit.validate()?;
        let mut payload_commit_ids = Vec::new();
        let mut payload_batch_record_commit_ids = Vec::new();
        let mut payload_batch_record_digests = BTreeMap::new();
        let mut remote_verification_results = Vec::new();
        for batch in &target.batches {
            let commit_id = batch.commit_id.clone().ok_or_else(|| {
                CacheError::InvalidTransition("verified payload batch has no commit id".to_owned())
            })?;
            if payload_commit_ids.last() != Some(&commit_id) {
                payload_commit_ids.push(commit_id.clone());
            }
            remote_verification_results.push(RemoteCommitVerificationResult {
                phase: "payload_batch".to_owned(),
                sequence: batch.plan.sequence,
                commit_id,
                verified: batch.state == crate::PublicationBatchState::RemoteVerified,
                content_digests: batch
                    .plan
                    .parts
                    .iter()
                    .map(|part| part.content_digest.clone())
                    .collect(),
            });
            if !batch.newly_committed_digests.is_empty() {
                let record = batch.record_commit.as_ref().ok_or_else(|| {
                    CacheError::InvalidTransition(
                        "new payload commit has no durable remote batch record".to_owned(),
                    )
                })?;
                if record.state != PublicationCommitState::RemoteVerified {
                    return Err(CacheError::InvalidTransition(
                        "payload batch record is not remotely verified".to_owned(),
                    ));
                }
                let record_commit_id = record.commit_id.clone().ok_or_else(|| {
                    CacheError::InvalidTransition(
                        "verified payload batch record has no commit identity".to_owned(),
                    )
                })?;
                if payload_batch_record_commit_ids.last() != Some(&record_commit_id) {
                    payload_batch_record_commit_ids.push(record_commit_id.clone());
                }
                let record_file = &record.files[0];
                if payload_batch_record_digests
                    .insert(
                        record_file.repository_path.clone(),
                        record_file.content_digest.clone(),
                    )
                    .is_some()
                {
                    return Err(CacheError::InvalidManifest(
                        "publication contains duplicate payload batch record paths".to_owned(),
                    ));
                }
                remote_verification_results.push(RemoteCommitVerificationResult {
                    phase: "payload_batch_record".to_owned(),
                    sequence: batch.plan.sequence,
                    commit_id: record_commit_id,
                    verified: record.state == PublicationCommitState::RemoteVerified,
                    content_digests: vec![record_file.content_digest.clone()],
                });
            }
        }
        remote_verification_results.push(RemoteCommitVerificationResult {
            phase: "immutable_metadata".to_owned(),
            sequence: 0,
            commit_id: metadata_commit_id.clone(),
            verified: metadata.state == PublicationCommitState::RemoteVerified,
            content_digests: metadata
                .files
                .iter()
                .map(|file| file.content_digest.clone())
                .collect(),
        });
        let receipt = Self {
            schema_version: 1,
            transaction_id: journal.transaction_id.clone(),
            idempotency_key: journal.idempotency_key.clone(),
            destination,
            principal: target.permission_evidence.principal.clone(),
            authorized_repository: target.authorized_repository.clone(),
            repository_permission_evidence_digest: target
                .permission_evidence
                .evidence_digest
                .clone(),
            shard_id: target.shard_id.clone(),
            branch: target.branch.clone(),
            semantic_digest: journal.semantic_digest.clone(),
            canonical_payload_digest: journal.payload_digest.clone(),
            manifest_digest: journal.target_manifest_digests[&destination].clone(),
            transport_digest,
            policy_digest: journal.policy_digest.clone(),
            policy_id: audit.policy_id.clone(),
            authority_mode: audit.authority_mode,
            validation_evidence_digests: audit.validation_evidence_digests.clone(),
            contributor_authorization_digest: audit.contributor_authorization_digest.clone(),
            reviewer_approvals: audit.reviewer_approvals.clone(),
            payload_commit_ids,
            payload_batch_record_commit_ids,
            payload_batch_record_digests,
            metadata_commit_id,
            metadata_file_digests: metadata
                .files
                .iter()
                .map(|file| (file.repository_path.clone(), file.content_digest.clone()))
                .collect(),
            discoverability_subject_digests,
            remote_verification_results,
            verified_at_unix_seconds,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.transaction_id != self.idempotency_key.0
            || self.principal.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || self.authorized_repository.trim().is_empty()
            || self.shard_id.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.payload_commit_ids.is_empty()
            || self
                .payload_commit_ids
                .iter()
                .any(|commit| commit.trim().is_empty())
            || self
                .payload_batch_record_commit_ids
                .iter()
                .any(|commit| commit.trim().is_empty())
            || self
                .payload_commit_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.payload_commit_ids.len()
            || self
                .payload_batch_record_commit_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.payload_batch_record_commit_ids.len()
            || self.payload_batch_record_digests.is_empty()
                != self.payload_batch_record_commit_ids.is_empty()
            || self.metadata_commit_id.trim().is_empty()
            || self.metadata_file_digests.is_empty()
            || self.discoverability_subject_digests.is_empty()
            || self.validation_evidence_digests.is_empty()
            || self.remote_verification_results.is_empty()
            || self.remote_verification_results.iter().any(|result| {
                result.phase.trim().is_empty()
                    || result.commit_id.trim().is_empty()
                    || !result.verified
                    || result.content_digests.is_empty()
                    || result
                        .content_digests
                        .iter()
                        .any(|digest| !digest.validate())
            })
            || self.verified_at_unix_seconds == 0
            || self.reviewer_approvals.iter().any(|review| {
                !review.approved
                    || review.reviewer_principal.trim().is_empty()
                    || review.pull_request_number == 0
                    || review.reviewed_head_revision.len() != 40
                    || !review
                        .reviewed_head_revision
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
            || (self.authority_mode == PublicationAuthorityMode::OwnerDirect
                && self.contributor_authorization_digest.is_some())
            || (self.authority_mode == PublicationAuthorityMode::ContributorReviewed
                && (self.contributor_authorization_digest.is_none()
                    || self.reviewer_approvals.is_empty()))
        {
            return Err(CacheError::InvalidManifest(
                "publication receipt is incomplete".to_owned(),
            ));
        }
        if [
            &self.idempotency_key,
            &self.repository_permission_evidence_digest,
            &self.semantic_digest,
            &self.canonical_payload_digest,
            &self.manifest_digest,
            &self.transport_digest,
            &self.policy_digest,
        ]
        .into_iter()
        .chain(self.metadata_file_digests.values())
        .chain(self.validation_evidence_digests.iter())
        .chain(self.contributor_authorization_digest.iter())
        .chain(
            self.reviewer_approvals
                .iter()
                .map(|review| &review.evidence_digest),
        )
        .chain(self.payload_batch_record_digests.values())
        .chain(self.discoverability_subject_digests.values())
        .any(|digest| !digest.validate())
            || self
                .metadata_file_digests
                .keys()
                .chain(self.payload_batch_record_digests.keys())
                .chain(self.discoverability_subject_digests.keys())
                .any(|path| !normalized_relative_path(path))
        {
            return Err(CacheError::InvalidManifest(
                "publication receipt contains an invalid path or digest".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn validate_for_transaction(
        &self,
        journal: &PublicationTransactionJournal,
        destination: PublicationDestination,
    ) -> Result<(), CacheError> {
        self.validate()?;
        journal.validate()?;
        let target = journal.targets.get(&destination).ok_or_else(|| {
            CacheError::InvalidTransition(format!(
                "transaction has no {destination:?} receipt target"
            ))
        })?;
        let metadata = target.metadata_commit.as_ref().ok_or_else(|| {
            CacheError::InvalidTransition("publication metadata was not planned".to_owned())
        })?;
        let audit = target.audit_evidence.as_ref().ok_or_else(|| {
            CacheError::InvalidTransition("publication target has no audit evidence".to_owned())
        })?;
        if self.transaction_id != journal.transaction_id
            || self.idempotency_key != journal.idempotency_key
            || self.destination != destination
            || self.principal != target.permission_evidence.principal
            || self.authorized_repository != target.authorized_repository
            || self.repository_permission_evidence_digest
                != target.permission_evidence.evidence_digest
            || self.shard_id != target.shard_id
            || self.branch != target.branch
            || self.semantic_digest != journal.semantic_digest
            || self.canonical_payload_digest != journal.payload_digest
            || self.manifest_digest != journal.target_manifest_digests[&destination]
            || self.policy_id != audit.policy_id
            || self.authority_mode != audit.authority_mode
            || self.validation_evidence_digests != audit.validation_evidence_digests
            || self.contributor_authorization_digest != audit.contributor_authorization_digest
            || self.reviewer_approvals != audit.reviewer_approvals
            || metadata.state != PublicationCommitState::RemoteVerified
            || metadata.commit_id.as_ref() != Some(&self.metadata_commit_id)
            || metadata
                .files
                .iter()
                .map(|file| (file.repository_path.clone(), file.content_digest.clone()))
                .collect::<BTreeMap<_, _>>()
                != self.metadata_file_digests
        {
            return Err(CacheError::InvalidManifest(
                "publication receipt does not match the verified target transaction".to_owned(),
            ));
        }
        let payload_commit_ids = target
            .batches
            .iter()
            .map(|batch| batch.commit_id.clone())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                CacheError::InvalidTransition(
                    "verified payload batch has no commit identity".to_owned(),
                )
            })?;
        let payload_commit_ids =
            payload_commit_ids
                .into_iter()
                .fold(Vec::<String>::new(), |mut commits, commit| {
                    if commits.last() != Some(&commit) {
                        commits.push(commit);
                    }
                    commits
                });
        if payload_commit_ids != self.payload_commit_ids {
            return Err(CacheError::InvalidManifest(
                "publication receipt payload commits do not match the journal".to_owned(),
            ));
        }
        let mut record_commit_ids = Vec::new();
        let mut record_digests = BTreeMap::new();
        for batch in &target.batches {
            if batch.newly_committed_digests.is_empty() {
                continue;
            }
            let record = batch.record_commit.as_ref().ok_or_else(|| {
                CacheError::InvalidTransition(
                    "new payload commit has no durable batch record".to_owned(),
                )
            })?;
            let commit_id = record.commit_id.clone().ok_or_else(|| {
                CacheError::InvalidTransition(
                    "payload batch record has no commit identity".to_owned(),
                )
            })?;
            if record_commit_ids.last() != Some(&commit_id) {
                record_commit_ids.push(commit_id);
            }
            let file = &record.files[0];
            record_digests.insert(file.repository_path.clone(), file.content_digest.clone());
        }
        if record_commit_ids != self.payload_batch_record_commit_ids
            || record_digests != self.payload_batch_record_digests
        {
            return Err(CacheError::InvalidManifest(
                "publication receipt batch records do not match the journal".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CacheError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    pub fn repository_path(&self) -> String {
        format!(
            "transactions/{}/{}/receipt.json",
            self.transaction_id,
            match self.destination {
                PublicationDestination::Private => "private",
                PublicationDestination::Public => "public",
            }
        )
    }

    pub fn as_transport_file(&self, sequence: u64) -> Result<(TransportPart, Vec<u8>), CacheError> {
        let bytes = self.canonical_bytes()?;
        let digest = ContentDigest::sha256(&bytes);
        debug_assert_eq!(digest, self.digest()?);
        Ok((
            TransportPart {
                sequence,
                repository_path: self.repository_path(),
                size_bytes: bytes.len() as u64,
                content_digest: digest,
            },
            bytes,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationInventoryTargetRecord {
    pub destination: PublicationDestination,
    pub completed_successfully: bool,
    pub authenticated_actor: String,
    pub authority_mode: PublicationAuthorityMode,
    pub policy_id: String,
    pub policy_digest: ContentDigest,
    pub authorized_repository: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub canonical_payload_digest: ContentDigest,
    pub transport_digest: Option<ContentDigest>,
    pub validation_evidence_digests: Vec<ContentDigest>,
    pub contributor_authorization_digest: Option<ContentDigest>,
    pub reviewer_approvals: Vec<PublicationReviewEvidence>,
    pub github_commit_ids: Vec<String>,
    pub remote_verification_results: Vec<RemoteCommitVerificationResult>,
    pub receipt_digest: Option<ContentDigest>,
    pub publication_time_unix_seconds: Option<u64>,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationInventoryRecord {
    pub schema_version: u32,
    pub transaction_id: String,
    pub targets: BTreeMap<PublicationDestination, PublicationInventoryTargetRecord>,
}

pub fn build_publication_inventory(
    journal: &PublicationTransactionJournal,
    receipts: &BTreeMap<PublicationDestination, PublicationReceipt>,
) -> Result<PublicationInventoryRecord, CacheError> {
    journal.validate()?;
    let mut targets = BTreeMap::new();
    for (destination, target) in &journal.targets {
        let audit = target.audit_evidence.as_ref().ok_or_else(|| {
            CacheError::InvalidManifest(format!(
                "publication target {destination:?} has no audit evidence"
            ))
        })?;
        audit.validate()?;
        let receipt = receipts.get(destination);
        if target.state == PublicationTargetState::ReceiptComplete {
            let receipt = receipt.ok_or_else(|| {
                CacheError::InvalidManifest(format!(
                    "completed publication target {destination:?} has no receipt"
                ))
            })?;
            receipt.validate_for_transaction(journal, *destination)?;
            if target.receipt_digest.as_ref() != Some(&receipt.digest()?) {
                return Err(CacheError::InvalidManifest(
                    "publication inventory receipt digest differs from the journal".to_owned(),
                ));
            }
        } else if receipt.is_some() {
            return Err(CacheError::InvalidManifest(format!(
                "incomplete publication target {destination:?} cannot claim a receipt"
            )));
        }

        let mut github_commit_ids = Vec::new();
        let mut push_commit = |commit: &str| {
            if github_commit_ids
                .last()
                .is_none_or(|previous| previous != commit)
            {
                github_commit_ids.push(commit.to_owned());
            }
        };
        for batch in &target.batches {
            if let Some(commit) = &batch.commit_id {
                push_commit(commit);
            }
            if let Some(commit) = batch
                .record_commit
                .as_ref()
                .and_then(|record| record.commit_id.as_ref())
            {
                push_commit(commit);
            }
        }
        if let Some(commit) = target
            .metadata_commit
            .as_ref()
            .and_then(|metadata| metadata.commit_id.as_ref())
        {
            push_commit(commit);
        }
        if let Some(commit) = target
            .discoverability_commit
            .as_ref()
            .and_then(|discoverability| discoverability.commit_id.as_ref())
        {
            push_commit(commit);
        }
        let mut remote_verification_results = receipt
            .map(|receipt| receipt.remote_verification_results.clone())
            .unwrap_or_default();
        if let Some(discoverability) = &target.discoverability_commit {
            if let Some(commit_id) = &discoverability.commit_id {
                remote_verification_results.push(RemoteCommitVerificationResult {
                    phase: "discoverability".to_owned(),
                    sequence: 0,
                    commit_id: commit_id.clone(),
                    verified: discoverability.state == PublicationCommitState::RemoteVerified,
                    content_digests: discoverability
                        .files
                        .iter()
                        .map(|file| file.content_digest.clone())
                        .collect(),
                });
            }
        }
        targets.insert(
            *destination,
            PublicationInventoryTargetRecord {
                destination: *destination,
                completed_successfully: target.state == PublicationTargetState::ReceiptComplete,
                authenticated_actor: target.permission_evidence.principal.clone(),
                authority_mode: audit.authority_mode,
                policy_id: audit.policy_id.clone(),
                policy_digest: journal.policy_digest.clone(),
                authorized_repository: target.authorized_repository.clone(),
                semantic_digest: journal.semantic_digest.clone(),
                manifest_digest: journal.target_manifest_digests[destination].clone(),
                canonical_payload_digest: journal.payload_digest.clone(),
                transport_digest: receipt.map(|receipt| receipt.transport_digest.clone()),
                validation_evidence_digests: audit.validation_evidence_digests.clone(),
                contributor_authorization_digest: audit.contributor_authorization_digest.clone(),
                reviewer_approvals: audit.reviewer_approvals.clone(),
                github_commit_ids,
                remote_verification_results,
                receipt_digest: target.receipt_digest.clone(),
                publication_time_unix_seconds: receipt
                    .map(|receipt| receipt.verified_at_unix_seconds),
                failure: target.failure.clone(),
            },
        );
    }
    if receipts
        .keys()
        .any(|destination| !journal.targets.contains_key(destination))
    {
        return Err(CacheError::InvalidManifest(
            "publication inventory contains a receipt for an unrequested target".to_owned(),
        ));
    }
    Ok(PublicationInventoryRecord {
        schema_version: 1,
        transaction_id: journal.transaction_id.clone(),
        targets,
    })
}

pub fn plan_discoverability_commit(
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    files: Vec<TransportPart>,
    receipt: &PublicationReceipt,
) -> Result<ContentDigest, CacheError> {
    receipt.validate_for_transaction(journal, destination)?;
    let receipt_digest = receipt.digest()?;
    let receipt_path = receipt.repository_path();
    let files_by_path: BTreeMap<_, _> = files
        .iter()
        .map(|file| (file.repository_path.as_str(), &file.content_digest))
        .collect();
    if files_by_path.get(receipt_path.as_str()).copied() != Some(&receipt_digest)
        || receipt
            .discoverability_subject_digests
            .iter()
            .any(|(path, digest)| files_by_path.get(path.as_str()).copied() != Some(digest))
    {
        return Err(CacheError::InvalidManifest(
            "discoverability files do not match the index, ledger, and receipt identities"
                .to_owned(),
        ));
    }
    journal
        .targets
        .get_mut(&destination)
        .expect("receipt validation checked target existence")
        .plan_discoverability_commit(files, receipt_digest.clone())?;
    Ok(receipt_digest)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicationFinalizationOutcome {
    AwaitingMetadataPlan,
    MetadataRefConflict {
        current_head: String,
    },
    MetadataCommittedAndVerified {
        commit_id: String,
    },
    MetadataAlreadyVerified {
        commit_id: String,
    },
    AwaitingDiscoverabilityPlan {
        metadata_commit_id: String,
    },
    DiscoverabilityRefConflictRequiresReplan {
        current_head: String,
    },
    ReceiptCommittedAndVerified {
        commit_id: String,
        receipt_digest: ContentDigest,
    },
    ReceiptAlreadyVerified {
        commit_id: String,
        receipt_digest: ContentDigest,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn execute_next_finalization_step(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    authenticated_session: &AuthenticatedGitHubSession,
    policy: &PublicationFinalizationPolicy,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
) -> Result<PublicationFinalizationOutcome, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    policy.validate()?;
    journal.validate()?;
    refresh_live_permission(checkpoints, authenticated_session, journal, destination)?;
    let target = &journal.targets[&destination];
    if target
        .batches
        .iter()
        .any(|batch| batch.state != crate::PublicationBatchState::RemoteVerified)
    {
        return Err(CacheError::InvalidTransition(
            "publication finalization requires all payload batches to be verified".to_owned(),
        ));
    }
    let Some(metadata) = target.metadata_commit.as_ref() else {
        return Ok(PublicationFinalizationOutcome::AwaitingMetadataPlan);
    };
    if metadata.state != PublicationCommitState::RemoteVerified {
        return execute_metadata_commit(
            remote,
            checkpoints,
            cancellation,
            policy,
            journal,
            destination,
        );
    }
    let metadata_commit_id = metadata
        .commit_id
        .clone()
        .expect("validated verified metadata has a commit id");
    let Some(discoverability) = target.discoverability_commit.as_ref() else {
        return Ok(
            PublicationFinalizationOutcome::AwaitingDiscoverabilityPlan { metadata_commit_id },
        );
    };
    if discoverability.state == PublicationCommitState::RemoteVerified {
        return Ok(PublicationFinalizationOutcome::ReceiptAlreadyVerified {
            commit_id: discoverability
                .commit_id
                .clone()
                .expect("validated verified discoverability has a commit id"),
            receipt_digest: discoverability
                .receipt_digest
                .clone()
                .expect("validated discoverability has a receipt digest"),
        });
    }
    execute_discoverability_commit(
        remote,
        checkpoints,
        cancellation,
        policy,
        journal,
        destination,
    )
}

fn refresh_live_permission(
    checkpoints: &PublicationJournalStore,
    authenticated_session: &AuthenticatedGitHubSession,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
) -> Result<(), CacheError> {
    let target = journal.targets.get(&destination).ok_or_else(|| {
        CacheError::InvalidTransition(format!(
            "transaction has no {destination:?} publication target"
        ))
    })?;
    authenticated_session.require_write_for(
        &target.permission_evidence.principal,
        &target.authorized_repository,
    )?;
    if target.permission_evidence != *authenticated_session.evidence() {
        journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked")
            .permission_evidence = authenticated_session.evidence().clone();
        checkpoints.save(journal)?;
    }
    Ok(())
}

fn execute_metadata_commit(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    policy: &PublicationFinalizationPolicy,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
) -> Result<PublicationFinalizationOutcome, CacheError> {
    let target = &journal.targets[&destination];
    let repository = target.repository.clone();
    let branch = target.branch.clone();
    let expected_head = target.expected_head.clone();
    let metadata = target
        .metadata_commit
        .as_ref()
        .expect("metadata existence was checked");
    let files = metadata.files.clone();
    if metadata.state == PublicationCommitState::Committed {
        let commit_id = metadata
            .commit_id
            .clone()
            .expect("validated committed metadata has a commit id");
        verify_files(remote, &repository, &commit_id, &files, cancellation)?;
        mark_metadata_verified(checkpoints, journal, destination, &commit_id)?;
        return Ok(PublicationFinalizationOutcome::MetadataCommittedAndVerified { commit_id });
    }

    let current_head = remote.read_ref(&repository, &branch)?;
    if current_head != expected_head {
        if files_match(
            remote,
            &repository,
            &current_head,
            &files,
            true,
            cancellation,
        )? {
            mark_metadata_reused(checkpoints, journal, destination, &current_head)?;
            return Ok(PublicationFinalizationOutcome::MetadataAlreadyVerified {
                commit_id: current_head,
            });
        }
        let target = journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked");
        target.expected_head = current_head.clone();
        target.retry_count = target.retry_count.saturating_add(1);
        checkpoints.save(journal)?;
        return Ok(PublicationFinalizationOutcome::MetadataRefConflict { current_head });
    }
    let missing_files =
        missing_immutable_files(remote, &repository, &current_head, &files, cancellation)?;
    if missing_files.is_empty() {
        mark_metadata_reused(checkpoints, journal, destination, &current_head)?;
        return Ok(PublicationFinalizationOutcome::MetadataAlreadyVerified {
            commit_id: current_head,
        });
    }
    revalidate_publication_capacity(remote, journal, destination, policy, 0, cancellation)?;
    let request = RemoteCommitRequest {
        repository: repository.clone(),
        branch,
        expected_head: expected_head.clone(),
        message: format!(
            "cache publication {} immutable metadata",
            journal.idempotency_key
        ),
        parts: missing_files,
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            let target = journal
                .targets
                .get_mut(&destination)
                .expect("target existence was checked");
            target.expected_head = current_head.clone();
            target.retry_count = target.retry_count.saturating_add(1);
            checkpoints.save(journal)?;
            Ok(PublicationFinalizationOutcome::MetadataRefConflict { current_head })
        }
        CompareAndSwapResult::Committed { commit_id } => {
            {
                let target = journal
                    .targets
                    .get_mut(&destination)
                    .expect("target existence was checked");
                target
                    .metadata_commit
                    .as_mut()
                    .expect("metadata existence was checked")
                    .mark_committed(
                        &expected_head,
                        &commit_id,
                        request
                            .parts
                            .iter()
                            .map(|file| file.content_digest.clone())
                            .collect(),
                    )?;
                target.expected_head = commit_id.clone();
            }
            checkpoints.save(journal)?;
            verify_files(remote, &repository, &commit_id, &files, cancellation)?;
            mark_metadata_verified(checkpoints, journal, destination, &commit_id)?;
            Ok(PublicationFinalizationOutcome::MetadataCommittedAndVerified { commit_id })
        }
    }
}

fn execute_discoverability_commit(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    policy: &PublicationFinalizationPolicy,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
) -> Result<PublicationFinalizationOutcome, CacheError> {
    let target = &journal.targets[&destination];
    let repository = target.repository.clone();
    let branch = target.branch.clone();
    let expected_head = target.expected_head.clone();
    let finalization = target
        .discoverability_commit
        .as_ref()
        .expect("discoverability existence was checked");
    let files = finalization.files.clone();
    let receipt_digest = finalization
        .receipt_digest
        .clone()
        .expect("validated discoverability has a receipt digest");
    if finalization.state == PublicationCommitState::Committed {
        let commit_id = finalization
            .commit_id
            .clone()
            .expect("validated committed discoverability has a commit id");
        verify_files(remote, &repository, &commit_id, &files, cancellation)?;
        mark_receipt_verified(
            checkpoints,
            journal,
            destination,
            &commit_id,
            receipt_digest.clone(),
        )?;
        return Ok(
            PublicationFinalizationOutcome::ReceiptCommittedAndVerified {
                commit_id,
                receipt_digest,
            },
        );
    }

    let current_head = remote.read_ref(&repository, &branch)?;
    if files_match(
        remote,
        &repository,
        &current_head,
        &files,
        false,
        cancellation,
    )? {
        mark_receipt_reused(
            checkpoints,
            journal,
            destination,
            &current_head,
            receipt_digest.clone(),
        )?;
        return Ok(PublicationFinalizationOutcome::ReceiptAlreadyVerified {
            commit_id: current_head,
            receipt_digest,
        });
    }
    if current_head != expected_head {
        journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked")
            .discard_planned_discoverability_after_conflict(current_head.clone())?;
        checkpoints.save(journal)?;
        return Ok(
            PublicationFinalizationOutcome::DiscoverabilityRefConflictRequiresReplan {
                current_head,
            },
        );
    }
    revalidate_publication_capacity(remote, journal, destination, policy, 0, cancellation)?;
    let request = RemoteCommitRequest {
        repository: repository.clone(),
        branch,
        expected_head: expected_head.clone(),
        message: format!(
            "cache publication {} index ledger receipt",
            journal.idempotency_key
        ),
        parts: files.clone(),
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            journal
                .targets
                .get_mut(&destination)
                .expect("target existence was checked")
                .discard_planned_discoverability_after_conflict(current_head.clone())?;
            checkpoints.save(journal)?;
            Ok(
                PublicationFinalizationOutcome::DiscoverabilityRefConflictRequiresReplan {
                    current_head,
                },
            )
        }
        CompareAndSwapResult::Committed { commit_id } => {
            {
                let target = journal
                    .targets
                    .get_mut(&destination)
                    .expect("target existence was checked");
                target
                    .discoverability_commit
                    .as_mut()
                    .expect("discoverability existence was checked")
                    .mark_committed(
                        &expected_head,
                        &commit_id,
                        request
                            .parts
                            .iter()
                            .map(|file| file.content_digest.clone())
                            .collect(),
                    )?;
                target.expected_head = commit_id.clone();
            }
            checkpoints.save(journal)?;
            verify_files(remote, &repository, &commit_id, &files, cancellation)?;
            mark_receipt_verified(
                checkpoints,
                journal,
                destination,
                &commit_id,
                receipt_digest.clone(),
            )?;
            Ok(
                PublicationFinalizationOutcome::ReceiptCommittedAndVerified {
                    commit_id,
                    receipt_digest,
                },
            )
        }
    }
}

fn mark_metadata_reused(
    checkpoints: &PublicationJournalStore,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    revision: &str,
) -> Result<(), CacheError> {
    let target = journal
        .targets
        .get_mut(&destination)
        .expect("target existence was checked");
    target
        .metadata_commit
        .as_mut()
        .expect("metadata existence was checked")
        .mark_reused(revision)?;
    target.expected_head = revision.to_owned();
    target.mark_remote_verified()?;
    checkpoints.save(journal)?;
    Ok(())
}

fn mark_metadata_verified(
    checkpoints: &PublicationJournalStore,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    commit_id: &str,
) -> Result<(), CacheError> {
    let target = journal
        .targets
        .get_mut(&destination)
        .expect("target existence was checked");
    target
        .metadata_commit
        .as_mut()
        .expect("metadata existence was checked")
        .mark_remote_verified(commit_id)?;
    target.mark_remote_verified()?;
    checkpoints.save(journal)?;
    Ok(())
}

fn mark_receipt_reused(
    checkpoints: &PublicationJournalStore,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    revision: &str,
    receipt_digest: ContentDigest,
) -> Result<(), CacheError> {
    let target = journal
        .targets
        .get_mut(&destination)
        .expect("target existence was checked");
    target
        .discoverability_commit
        .as_mut()
        .expect("discoverability existence was checked")
        .mark_reused(revision)?;
    target.expected_head = revision.to_owned();
    target.complete_receipt(receipt_digest)?;
    checkpoints.save(journal)?;
    Ok(())
}

fn mark_receipt_verified(
    checkpoints: &PublicationJournalStore,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    commit_id: &str,
    receipt_digest: ContentDigest,
) -> Result<(), CacheError> {
    let target = journal
        .targets
        .get_mut(&destination)
        .expect("target existence was checked");
    target
        .discoverability_commit
        .as_mut()
        .expect("discoverability existence was checked")
        .mark_remote_verified(commit_id)?;
    target.complete_receipt(receipt_digest)?;
    checkpoints.save(journal)?;
    Ok(())
}

fn files_match(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    files: &[TransportPart],
    immutable: bool,
    cancellation: &CancellationToken,
) -> Result<bool, CacheError> {
    let mut all_match = true;
    for file in files {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        match remote.immutable_path_digest(repository, revision, &file.repository_path)? {
            Some(actual) if actual == file.content_digest => {}
            Some(actual) if immutable => {
                return Err(CacheError::DigestMismatch {
                    expected: file.content_digest.to_string(),
                    actual: actual.to_string(),
                });
            }
            Some(_) | None => all_match = false,
        }
    }
    Ok(all_match)
}

fn missing_immutable_files(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    files: &[TransportPart],
    cancellation: &CancellationToken,
) -> Result<Vec<TransportPart>, CacheError> {
    let mut missing = Vec::new();
    for file in files {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        match remote.immutable_path_digest(repository, revision, &file.repository_path)? {
            Some(actual) if actual == file.content_digest => {}
            Some(actual) => {
                return Err(CacheError::DigestMismatch {
                    expected: file.content_digest.to_string(),
                    actual: actual.to_string(),
                });
            }
            None => missing.push(file.clone()),
        }
    }
    Ok(missing)
}

fn verify_files(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    files: &[TransportPart],
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    for file in files {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        remote.verify_committed_part(repository, revision, file)?;
    }
    Ok(())
}

pub(crate) fn revalidate_publication_capacity(
    remote: &dyn RemoteGitStore,
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    policy: &PublicationFinalizationPolicy,
    pending_unique_payload_bytes: u64,
    cancellation: &CancellationToken,
) -> Result<CapacityAdmission, CacheError> {
    let target = &journal.targets[&destination];
    let mut bytes = Vec::new();
    let source = remote.read_committed_path(
        &target.repository,
        &target.expected_head,
        &policy.capacity_ledger_path,
        policy.maximum_capacity_ledger_bytes,
        cancellation,
        &mut bytes,
    )?;
    if source.revision != target.expected_head
        || source.repository_path != policy.capacity_ledger_path
    {
        return Err(CacheError::InvalidManifest(
            "capacity ledger was not read from the expected shard revision".to_owned(),
        ));
    }
    let ledger: CapacityLedger = serde_json::from_slice(&bytes)?;
    ledger.validate()?;
    if ledger.shard_id != target.shard_id {
        return Err(CacheError::InvalidManifest(format!(
            "capacity ledger belongs to shard {:?}, expected {:?}",
            ledger.shard_id, target.shard_id
        )));
    }
    let part_sizes: BTreeMap<_, _> = target
        .batches
        .iter()
        .flat_map(|batch| &batch.plan.parts)
        .map(|part| (part.content_digest.clone(), part.size_bytes))
        .collect();
    let unique_payload_bytes = target
        .batches
        .iter()
        .flat_map(|batch| &batch.newly_committed_digests)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|digest| part_sizes.get(digest).copied().unwrap_or(0))
        .fold(0u64, u64::saturating_add)
        .saturating_add(pending_unique_payload_bytes);
    let metadata_bytes = target
        .batches
        .iter()
        .filter_map(|batch| batch.record_commit.as_ref())
        .flat_map(|commit| &commit.files)
        .chain(
            target
                .metadata_commit
                .iter()
                .chain(target.discoverability_commit.iter())
                .flat_map(|commit| &commit.files),
        )
        .map(|file| file.size_bytes)
        .fold(0u64, u64::saturating_add);
    let admission = ledger.assess_addition(
        unique_payload_bytes,
        metadata_bytes,
        policy.projected_history_bytes,
    )?;
    if !admission.accepted {
        return Err(CacheError::NoWritableShard(format!(
            "capacity changed before publication finalization: {}",
            admission.reasons.join("; ")
        )));
    }
    Ok(admission)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plan_publication_batches, PublicationBatchPlan, PublicationTransactionJournal,
        RemoteReadReport, RepositoryPermission, TransportPolicy,
        GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use std::io::Write;
    use std::sync::Mutex;
    use xc_core::PublicationTarget;

    struct MemoryRemote {
        head: Mutex<String>,
        sequence: Mutex<u64>,
        revisions: Mutex<BTreeMap<String, BTreeMap<String, Vec<u8>>>>,
        staged: Mutex<BTreeMap<String, Vec<u8>>>,
    }

    impl MemoryRemote {
        fn stage(&self, path: &str, bytes: Vec<u8>) -> TransportPart {
            let digest = ContentDigest::sha256(&bytes);
            self.staged
                .lock()
                .unwrap()
                .insert(path.to_owned(), bytes.clone());
            TransportPart {
                sequence: 0,
                repository_path: path.to_owned(),
                size_bytes: bytes.len() as u64,
                content_digest: digest,
            }
        }

        fn advance_unrelated(&self) {
            let current = self.head.lock().unwrap().clone();
            let current_tree = self.revisions.lock().unwrap()[&current].clone();
            let mut sequence = self.sequence.lock().unwrap();
            *sequence += 1;
            let next = format!("head-{sequence}");
            self.revisions
                .lock()
                .unwrap()
                .insert(next.clone(), current_tree);
            *self.head.lock().unwrap() = next;
        }
    }

    impl RemoteGitStore for MemoryRemote {
        fn read_ref(&self, _repository: &str, _branch: &str) -> Result<String, CacheError> {
            Ok(self.head.lock().unwrap().clone())
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(self
                .revisions
                .lock()
                .unwrap()
                .get(revision)
                .and_then(|tree| tree.get(path))
                .map(|bytes| ContentDigest::sha256(bytes)))
        }

        fn read_committed_path(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
            maximum_bytes: u64,
            cancellation: &CancellationToken,
            writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let revisions = self.revisions.lock().unwrap();
            let bytes = revisions
                .get(revision)
                .and_then(|tree| tree.get(path))
                .ok_or_else(|| CacheError::NotFound(format!("{revision}:{path}")))?;
            if bytes.len() as u64 > maximum_bytes {
                return Err(CacheError::ResourceLimit(path.to_owned()));
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
            let current = self.head.lock().unwrap().clone();
            if current != request.expected_head {
                return Ok(CompareAndSwapResult::RefConflict {
                    current_head: current,
                });
            }
            let mut tree = self.revisions.lock().unwrap()[&current].clone();
            let staged = self.staged.lock().unwrap();
            for file in &request.parts {
                let bytes = staged
                    .get(&file.repository_path)
                    .ok_or_else(|| CacheError::NotFound(file.repository_path.clone()))?;
                if bytes.len() as u64 != file.size_bytes
                    || ContentDigest::sha256(bytes) != file.content_digest
                {
                    return Err(CacheError::DigestMismatch {
                        expected: file.content_digest.to_string(),
                        actual: ContentDigest::sha256(bytes).to_string(),
                    });
                }
                tree.insert(file.repository_path.clone(), bytes.clone());
            }
            drop(staged);
            let mut sequence = self.sequence.lock().unwrap();
            *sequence += 1;
            let next = format!("head-{sequence}");
            self.revisions.lock().unwrap().insert(next.clone(), tree);
            *self.head.lock().unwrap() = next.clone();
            Ok(CompareAndSwapResult::Committed { commit_id: next })
        }

        fn verify_committed_part(
            &self,
            repository: &str,
            revision: &str,
            part: &TransportPart,
        ) -> Result<(), CacheError> {
            match self.immutable_path_digest(repository, revision, &part.repository_path)? {
                Some(actual) if actual == part.content_digest => Ok(()),
                Some(actual) => Err(CacheError::DigestMismatch {
                    expected: part.content_digest.to_string(),
                    actual: actual.to_string(),
                }),
                None => Err(CacheError::NotFound(part.repository_path.clone())),
            }
        }
    }

    fn ledger(last_commit: &str) -> CapacityLedger {
        CapacityLedger {
            schema_version: 1,
            shard_id: "private-001".to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 0,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 0,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: last_commit.to_owned(),
            reconciliation_digest: ContentDigest::sha256(b"reconciliation"),
        }
    }

    fn payload_batches(part: &TransportPart) -> Vec<PublicationBatchPlan> {
        plan_publication_batches(
            std::slice::from_ref(part),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 10,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
        )
        .unwrap()
    }

    fn fixture() -> (
        MemoryRemote,
        PublicationTransactionJournal,
        AuthenticatedGitHubSession,
    ) {
        let payload = b"payload".to_vec();
        let payload_part = TransportPart {
            sequence: 0,
            repository_path: "objects/payload.part".to_owned(),
            size_bytes: payload.len() as u64,
            content_digest: ContentDigest::sha256(&payload),
        };
        let ledger_bytes = serde_json::to_vec(&ledger("head-0")).unwrap();
        let remote = MemoryRemote {
            head: Mutex::new("head-0".to_owned()),
            sequence: Mutex::new(0),
            revisions: Mutex::new(BTreeMap::from([(
                "head-0".to_owned(),
                BTreeMap::from([
                    (payload_part.repository_path.clone(), payload.clone()),
                    (DEFAULT_CAPACITY_LEDGER_PATH.to_owned(), ledger_bytes),
                ]),
            )])),
            staged: Mutex::new(BTreeMap::new()),
        };
        let session = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            RepositoryPermission::Write,
        );
        let mut journal = PublicationTransactionJournal::new(
            ContentDigest::sha256(b"semantic"),
            BTreeMap::from([(
                PublicationDestination::Private,
                ContentDigest::sha256(b"manifest"),
            )]),
            ContentDigest::sha256(b"payload"),
            ContentDigest::sha256(b"policy"),
            PublicationTarget::Private,
            BTreeMap::from([(PublicationDestination::Private, "memory".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "team/private".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, session.evidence().clone())]),
            BTreeMap::from([(PublicationDestination::Private, "private-001".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "main".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "head-0".to_owned())]),
            &payload_batches(&payload_part),
        )
        .unwrap();
        crate::attach_test_owner_audit_evidence(&mut journal, PublicationDestination::Private);
        let target = journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap();
        target.start().unwrap();
        target.mark_batch_reused(0, "head-0").unwrap();
        (remote, journal, session)
    }

    fn checkpoint_store(name: &str) -> PublicationJournalStore {
        let root =
            std::env::temp_dir().join(format!("xc-cache-finalizer-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        PublicationJournalStore::new(root)
    }

    #[test]
    fn finalization_commits_metadata_then_index_ledger_and_receipt() {
        let (remote, mut journal, session) = fixture();
        let checkpoints = checkpoint_store("complete");
        let mut metadata = vec![
            remote.stage("transport/record.json", b"transport".to_vec()),
            remote.stage("manifests/manifest.json", b"manifest".to_vec()),
        ];
        for (sequence, file) in metadata.iter_mut().enumerate() {
            file.sequence = sequence as u64;
        }
        journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap()
            .plan_metadata_commit(metadata)
            .unwrap();
        let outcome = execute_next_finalization_step(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &session,
            &PublicationFinalizationPolicy::default(),
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            PublicationFinalizationOutcome::MetadataCommittedAndVerified { .. }
        ));

        let index_bytes = b"index".to_vec();
        let ledger_bytes = serde_json::to_vec(&ledger("head-1")).unwrap();
        let subjects = BTreeMap::from([
            (
                "indexes/ccm/aa.json".to_owned(),
                ContentDigest::sha256(&index_bytes),
            ),
            (
                DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                ContentDigest::sha256(&ledger_bytes),
            ),
        ]);
        let receipt = PublicationReceipt::from_verified_metadata(
            &journal,
            PublicationDestination::Private,
            ContentDigest::sha256(b"transport-record"),
            subjects,
            123,
        )
        .unwrap();
        let receipt_digest = receipt.digest().unwrap();
        let (mut receipt_file, receipt_bytes) = receipt.as_transport_file(2).unwrap();
        remote
            .staged
            .lock()
            .unwrap()
            .insert(receipt_file.repository_path.clone(), receipt_bytes);
        let mut final_files = vec![
            remote.stage("indexes/ccm/aa.json", index_bytes),
            remote.stage(DEFAULT_CAPACITY_LEDGER_PATH, ledger_bytes),
        ];
        receipt_file.sequence = 2;
        final_files.push(receipt_file);
        for (sequence, file) in final_files.iter_mut().enumerate() {
            file.sequence = sequence as u64;
        }
        assert_eq!(
            plan_discoverability_commit(
                &mut journal,
                PublicationDestination::Private,
                final_files,
                &receipt,
            )
            .unwrap(),
            receipt_digest
        );
        let outcome = execute_next_finalization_step(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &session,
            &PublicationFinalizationPolicy::default(),
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            PublicationFinalizationOutcome::ReceiptCommittedAndVerified { .. }
        ));
        assert!(journal.complete());
        assert_eq!(
            checkpoints.load_latest(&journal.transaction_id).unwrap(),
            journal
        );
        assert_eq!(
            remote
                .immutable_path_digest(
                    "memory",
                    &journal.targets[&PublicationDestination::Private].expected_head,
                    &receipt.repository_path(),
                )
                .unwrap(),
            Some(receipt_digest)
        );
        let inventory = build_publication_inventory(
            &journal,
            &BTreeMap::from([(PublicationDestination::Private, receipt)]),
        )
        .unwrap();
        let target = &inventory.targets[&PublicationDestination::Private];
        assert!(target.completed_successfully);
        assert_eq!(target.authenticated_actor, "test-owner");
        assert_eq!(target.policy_id, "fixture-owner-policy");
        assert_eq!(target.authority_mode, PublicationAuthorityMode::OwnerDirect);
        assert!(target.reviewer_approvals.is_empty());
        assert_eq!(target.publication_time_unix_seconds, Some(123));
        assert!(target.github_commit_ids.len() >= 3);
        assert!(target
            .remote_verification_results
            .iter()
            .any(|result| result.phase == "discoverability" && result.verified));
    }

    #[test]
    fn concurrent_discoverability_update_requires_a_fresh_plan() {
        let (remote, mut journal, session) = fixture();
        let checkpoints = checkpoint_store("conflict");
        let mut metadata = vec![remote.stage("transport/record.json", b"transport".to_vec())];
        metadata[0].sequence = 0;
        journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap()
            .plan_metadata_commit(metadata)
            .unwrap();
        execute_next_finalization_step(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &session,
            &PublicationFinalizationPolicy::default(),
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        let receipt = PublicationReceipt::from_verified_metadata(
            &journal,
            PublicationDestination::Private,
            ContentDigest::sha256(b"transport-record"),
            BTreeMap::from([(
                "indexes/ccm/aa.json".to_owned(),
                ContentDigest::sha256(b"index"),
            )]),
            123,
        )
        .unwrap();
        let (mut receipt_file, receipt_bytes) = receipt.as_transport_file(1).unwrap();
        remote
            .staged
            .lock()
            .unwrap()
            .insert(receipt_file.repository_path.clone(), receipt_bytes);
        let mut index = remote.stage("indexes/ccm/aa.json", b"index".to_vec());
        index.sequence = 0;
        receipt_file.sequence = 1;
        plan_discoverability_commit(
            &mut journal,
            PublicationDestination::Private,
            vec![index, receipt_file],
            &receipt,
        )
        .unwrap();
        remote.advance_unrelated();
        let outcome = execute_next_finalization_step(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &session,
            &PublicationFinalizationPolicy::default(),
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            PublicationFinalizationOutcome::DiscoverabilityRefConflictRequiresReplan { .. }
        ));
        let target = &journal.targets[&PublicationDestination::Private];
        assert!(target.discoverability_commit.is_none());
        assert_eq!(target.state, PublicationTargetState::RemoteVerified);
    }

    #[test]
    fn finalization_refuses_remote_access_without_fresh_write_permission() {
        let (remote, mut journal, _session) = fixture();
        let checkpoints = checkpoint_store("permission");
        let mut metadata = vec![remote.stage("transport/record.json", b"transport".to_vec())];
        metadata[0].sequence = 0;
        journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap()
            .plan_metadata_commit(metadata)
            .unwrap();
        let read_only = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            RepositoryPermission::Read,
        );
        let error = execute_next_finalization_step(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &read_only,
            &PublicationFinalizationPolicy::default(),
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::PermissionDenied(_)));
        assert_eq!(remote.read_ref("memory", "main").unwrap(), "head-0");
    }

    #[test]
    fn finalization_revalidates_capacity_before_mutation() {
        let (remote, mut journal, session) = fixture();
        let checkpoints = checkpoint_store("capacity");
        let mut full_ledger = ledger("head-0");
        full_ledger.first_seen_immutable_payload_bytes =
            full_ledger.hard_capacity_bytes.saturating_sub(1);
        remote
            .revisions
            .lock()
            .unwrap()
            .get_mut("head-0")
            .unwrap()
            .insert(
                DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                serde_json::to_vec(&full_ledger).unwrap(),
            );
        let mut metadata = vec![remote.stage("transport/record.json", b"transport".to_vec())];
        metadata[0].sequence = 0;
        journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap()
            .plan_metadata_commit(metadata)
            .unwrap();
        let error = execute_next_finalization_step(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &session,
            &PublicationFinalizationPolicy::default(),
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::NoWritableShard(_)));
        assert_eq!(remote.read_ref("memory", "main").unwrap(), "head-0");
    }
}
