//! Durable publication recovery inspection and fail-closed abandonment.

use crate::{
    AuthenticatedGitHubSession, CacheError, ContentDigest, PublicationBatchState,
    PublicationCommitState, PublicationDestination, PublicationJournalStore,
    PublicationTargetState, PublicationTransactionJournal, RemoteGitStore,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use xc_core::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationResumePhase {
    VerifyCommittedPayload,
    PublishPayloadBatchRecord,
    VerifyPayloadBatchRecord,
    UploadPayload,
    StageImmutableMetadata,
    PublishImmutableMetadata,
    VerifyImmutableMetadata,
    StageDiscoverability,
    PublishDiscoverability,
    VerifyDiscoverability,
    Complete,
    Abandoned,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationTargetRecoveryStatus {
    pub destination: PublicationDestination,
    pub repository: String,
    pub branch: String,
    pub state: PublicationTargetState,
    pub resume_phase: PublicationResumePhase,
    pub verified_payload_batches: u64,
    pub total_payload_batches: u64,
    pub newly_committed_bytes: u64,
    pub reachable_commit_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationRecoveryReport {
    pub schema_version: u32,
    pub transaction_id: String,
    pub complete: bool,
    /// Local staging can be considered for explicit policy-controlled cleanup
    /// only after no target can still need it.
    pub local_staging_cleanup_eligible: bool,
    pub targets: BTreeMap<PublicationDestination, PublicationTargetRecoveryStatus>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationAbandonmentReport {
    pub schema_version: u32,
    pub transaction_id: String,
    pub destination: PublicationDestination,
    pub repository: String,
    pub inspected_head: String,
    pub previous_state: PublicationTargetState,
    pub state: PublicationTargetState,
    pub newly_committed_bytes_retained: u64,
    pub reachable_commit_ids: Vec<String>,
    pub temporary_reservation_released: bool,
    pub local_staging_cleanup_eligible: bool,
    pub checkpoint_path: PathBuf,
}

pub fn inspect_publication_recovery(
    journal: &PublicationTransactionJournal,
) -> Result<PublicationRecoveryReport, CacheError> {
    journal.validate()?;
    let mut targets = BTreeMap::new();
    for (destination, target) in &journal.targets {
        let (newly_committed_bytes, reachable_commit_ids) = committed_footprint(target)?;
        targets.insert(
            *destination,
            PublicationTargetRecoveryStatus {
                destination: *destination,
                repository: target.repository.clone(),
                branch: target.branch.clone(),
                state: target.state,
                resume_phase: resume_phase(target),
                verified_payload_batches: target
                    .batches
                    .iter()
                    .filter(|batch| batch.state == PublicationBatchState::RemoteVerified)
                    .count() as u64,
                total_payload_batches: target.batches.len() as u64,
                newly_committed_bytes,
                reachable_commit_ids,
            },
        );
    }
    Ok(PublicationRecoveryReport {
        schema_version: 1,
        transaction_id: journal.transaction_id.clone(),
        complete: journal.complete(),
        local_staging_cleanup_eligible: journal.targets.values().all(|target| {
            matches!(
                target.state,
                PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
            )
        }),
        targets,
    })
}

/// Mark one target abandoned only after proving that its immutable receipt is
/// absent from the current remote head. This prevents a lost compare-and-swap
/// response from turning an already discoverable artifact into a misleading
/// local `abandoned` state. Remote objects already introduced by the
/// transaction are retained and reported for capacity reconciliation.
#[allow(clippy::too_many_arguments)]
pub fn abandon_publication_target(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    authenticated_session: &AuthenticatedGitHubSession,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    reason: &str,
) -> Result<PublicationAbandonmentReport, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    if reason.trim().is_empty() || reason.len() > 4_096 {
        return Err(CacheError::InvalidManifest(
            "publication abandonment reason must contain 1 to 4096 bytes".to_owned(),
        ));
    }
    let target = journal.targets.get(&destination).ok_or_else(|| {
        CacheError::InvalidTransition(format!(
            "transaction has no {destination:?} publication target"
        ))
    })?;
    if target.state == PublicationTargetState::ReceiptComplete {
        return Err(CacheError::InvalidTransition(
            "a receipt-complete publication target cannot be abandoned".to_owned(),
        ));
    }
    if target.state == PublicationTargetState::Abandoned {
        return Err(CacheError::InvalidTransition(
            "publication target is already abandoned".to_owned(),
        ));
    }
    authenticated_session.require_write_for(
        &target.permission_evidence.principal,
        &target.authorized_repository,
    )?;

    let repository = target.repository.clone();
    let branch = target.branch.clone();
    let inspected_head = remote.read_ref(&repository, &branch)?;
    let receipt_path = format!(
        "transactions/{}/{}/receipt.json",
        journal.transaction_id,
        match destination {
            PublicationDestination::Private => "private",
            PublicationDestination::Public => "public",
        }
    );
    if let Some(receipt_digest) =
        remote.immutable_path_digest(&repository, &inspected_head, &receipt_path)?
    {
        return Err(CacheError::InvalidTransition(format!(
            "remote receipt {receipt_path:?} with digest {receipt_digest} already exists at {inspected_head}; resume verification instead of abandoning"
        )));
    }

    let target = journal
        .targets
        .get_mut(&destination)
        .expect("target existence was checked");
    let previous_state = target.state;
    if target.permission_evidence != *authenticated_session.evidence() {
        target.permission_evidence = authenticated_session.evidence().clone();
    }
    let (newly_committed_bytes_retained, reachable_commit_ids) = committed_footprint(target)?;
    target.abandon(reason)?;
    let local_staging_cleanup_eligible = journal.targets.values().all(|target| {
        matches!(
            target.state,
            PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
        )
    });
    let checkpoint_path = checkpoints.save(journal)?;
    Ok(PublicationAbandonmentReport {
        schema_version: 1,
        transaction_id: journal.transaction_id.clone(),
        destination,
        repository,
        inspected_head,
        previous_state,
        state: PublicationTargetState::Abandoned,
        newly_committed_bytes_retained,
        reachable_commit_ids,
        temporary_reservation_released: true,
        local_staging_cleanup_eligible,
        checkpoint_path,
    })
}

fn resume_phase(target: &crate::TargetPublicationJournal) -> PublicationResumePhase {
    if target.state == PublicationTargetState::Abandoned {
        return PublicationResumePhase::Abandoned;
    }
    if target.state == PublicationTargetState::ReceiptComplete {
        return PublicationResumePhase::Complete;
    }
    if target
        .batches
        .iter()
        .any(|batch| batch.state == PublicationBatchState::Committed)
    {
        return PublicationResumePhase::VerifyCommittedPayload;
    }
    if target
        .batches
        .iter()
        .any(|batch| batch.state == PublicationBatchState::PayloadVerified)
    {
        return PublicationResumePhase::PublishPayloadBatchRecord;
    }
    if target
        .batches
        .iter()
        .any(|batch| batch.state == PublicationBatchState::RecordCommitted)
    {
        return PublicationResumePhase::VerifyPayloadBatchRecord;
    }
    if target
        .batches
        .iter()
        .any(|batch| batch.state == PublicationBatchState::Planned)
    {
        return PublicationResumePhase::UploadPayload;
    }
    match target.metadata_commit.as_ref() {
        None => PublicationResumePhase::StageImmutableMetadata,
        Some(commit) if commit.state == PublicationCommitState::Planned => {
            PublicationResumePhase::PublishImmutableMetadata
        }
        Some(commit) if commit.state == PublicationCommitState::Committed => {
            PublicationResumePhase::VerifyImmutableMetadata
        }
        Some(_) => match target.discoverability_commit.as_ref() {
            None => PublicationResumePhase::StageDiscoverability,
            Some(commit) if commit.state == PublicationCommitState::Planned => {
                PublicationResumePhase::PublishDiscoverability
            }
            Some(commit) if commit.state == PublicationCommitState::Committed => {
                PublicationResumePhase::VerifyDiscoverability
            }
            Some(_) => PublicationResumePhase::VerifyDiscoverability,
        },
    }
}

fn committed_footprint(
    target: &crate::TargetPublicationJournal,
) -> Result<(u64, Vec<String>), CacheError> {
    let mut sizes = BTreeMap::<ContentDigest, u64>::new();
    let mut commit_ids = BTreeSet::new();
    for batch in &target.batches {
        record_committed_files(
            &batch.plan.parts,
            &batch.newly_committed_digests,
            batch.commit_id.as_deref(),
            &mut sizes,
            &mut commit_ids,
        )?;
        if let Some(record) = &batch.record_commit {
            record_committed_files(
                &record.files,
                &record.newly_committed_digests,
                record.commit_id.as_deref(),
                &mut sizes,
                &mut commit_ids,
            )?;
        }
    }
    for commit in [
        target.metadata_commit.as_ref(),
        target.discoverability_commit.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        record_committed_files(
            &commit.files,
            &commit.newly_committed_digests,
            commit.commit_id.as_deref(),
            &mut sizes,
            &mut commit_ids,
        )?;
    }
    let bytes = sizes.values().try_fold(0u64, |total, size| {
        total.checked_add(*size).ok_or_else(|| {
            CacheError::ResourceLimit("committed publication footprint exceeds u64".to_owned())
        })
    })?;
    Ok((bytes, commit_ids.into_iter().collect()))
}

fn record_committed_files(
    files: &[crate::TransportPart],
    newly_committed_digests: &[ContentDigest],
    commit_id: Option<&str>,
    sizes: &mut BTreeMap<ContentDigest, u64>,
    commit_ids: &mut BTreeSet<String>,
) -> Result<(), CacheError> {
    if newly_committed_digests.is_empty() {
        return Ok(());
    }
    let commit_id = commit_id.ok_or_else(|| {
        CacheError::InvalidManifest(
            "newly committed publication objects have no commit identity".to_owned(),
        )
    })?;
    commit_ids.insert(commit_id.to_owned());
    for digest in newly_committed_digests {
        let file = files
            .iter()
            .find(|file| &file.content_digest == digest)
            .ok_or_else(|| {
                CacheError::InvalidManifest(
                    "newly committed publication object is absent from its commit plan".to_owned(),
                )
            })?;
        match sizes.insert(digest.clone(), file.size_bytes) {
            Some(previous) if previous != file.size_bytes => {
                return Err(CacheError::InvalidManifest(
                    "one publication content identity has conflicting byte counts".to_owned(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plan_publication_batches, CompareAndSwapResult, PublicationTransactionJournal,
        RemoteCommitRequest, RemoteReadReport, RepositoryPermission, TransportPart,
        TransportPolicy,
    };
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::Path;
    use xc_core::PublicationTarget;

    fn journal() -> PublicationTransactionJournal {
        let part = TransportPart {
            sequence: 0,
            repository_path: "objects/payload.part".to_owned(),
            size_bytes: 7,
            content_digest: ContentDigest::sha256(b"payload"),
        };
        let batches = plan_publication_batches(
            &[part],
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 90,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
        )
        .unwrap();
        let permission = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            RepositoryPermission::Write,
        )
        .evidence()
        .clone();
        PublicationTransactionJournal::new(
            ContentDigest::sha256(b"semantic"),
            BTreeMap::from([(
                PublicationDestination::Private,
                ContentDigest::sha256(b"manifest"),
            )]),
            ContentDigest::sha256(b"payload"),
            ContentDigest::sha256(b"policy"),
            PublicationTarget::Private,
            BTreeMap::from([(PublicationDestination::Private, "remote.git".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "team/private".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, permission)]),
            BTreeMap::from([(PublicationDestination::Private, "private-001".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "main".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "head-0".to_owned())]),
            &batches,
        )
        .unwrap()
    }

    struct ReceiptRemote {
        receipt: Option<ContentDigest>,
    }

    impl RemoteGitStore for ReceiptRemote {
        fn read_ref(&self, _repository: &str, _branch: &str) -> Result<String, CacheError> {
            Ok("head-current".to_owned())
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            if path.ends_with("/receipt.json") {
                Ok(self.receipt.clone())
            } else {
                Ok(None)
            }
        }

        fn read_committed_path(
            &self,
            _repository: &str,
            _revision: &str,
            _path: &str,
            _maximum_bytes: u64,
            _cancellation: &CancellationToken,
            _writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            Err(CacheError::NotFound("fixture".to_owned()))
        }

        fn compare_and_swap_commit(
            &self,
            _request: &RemoteCommitRequest,
        ) -> Result<CompareAndSwapResult, CacheError> {
            panic!("abandonment must not mutate the remote")
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            _revision: &str,
            _part: &TransportPart,
        ) -> Result<(), CacheError> {
            panic!("abandonment must not verify payload parts")
        }
    }

    fn session() -> AuthenticatedGitHubSession {
        AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            RepositoryPermission::Write,
        )
    }

    fn mark_batch_record_verified(journal: &mut PublicationTransactionJournal) {
        let record =
            crate::build_payload_batch_record(journal, PublicationDestination::Private, 0).unwrap();
        let file = TransportPart {
            sequence: 0,
            repository_path: record.repository_path(),
            size_bytes: serde_json::to_vec(&record).unwrap().len() as u64,
            content_digest: record.digest().unwrap(),
        };
        let target = journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap();
        target
            .plan_verified_batch_record(0, "head-1", file)
            .unwrap();
        target.mark_batch_record_reused(0, "head-1").unwrap();
    }

    #[test]
    fn recovery_report_identifies_the_exact_next_phase() {
        let mut journal = journal();
        let report = inspect_publication_recovery(&journal).unwrap();
        assert_eq!(
            report.targets[&PublicationDestination::Private].resume_phase,
            PublicationResumePhase::UploadPayload
        );
        let target = journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap();
        target.start().unwrap();
        let digest = target.batches[0].plan.parts[0].content_digest.clone();
        target
            .mark_batch_committed(0, "head-0", "head-1", vec![digest])
            .unwrap();
        let report = inspect_publication_recovery(&journal).unwrap();
        let target = &report.targets[&PublicationDestination::Private];
        assert_eq!(
            target.resume_phase,
            PublicationResumePhase::VerifyCommittedPayload
        );
        assert_eq!(target.newly_committed_bytes, 7);
        assert_eq!(target.reachable_commit_ids, vec!["head-1"]);
    }

    #[test]
    fn failed_target_restores_the_phase_implied_by_durable_substate() {
        let mut journal = journal();
        let target = journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap();
        target.start().unwrap();
        let digest = target.batches[0].plan.parts[0].content_digest.clone();
        target
            .mark_batch_committed(0, "head-0", "head-1", vec![digest])
            .unwrap();
        mark_batch_record_verified(&mut journal);
        let target = journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap();
        target.fail("interrupted before metadata staging");
        target.restore_failed_resume_state().unwrap();
        assert_eq!(target.state, PublicationTargetState::BatchVerified);
        assert_eq!(target.failure, None);
    }

    #[test]
    fn abandon_is_append_only_and_retains_remote_object_accounting() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("abandon-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = PublicationJournalStore::new(&root);
        let mut journal = journal();
        store.save(&journal).unwrap();
        {
            let target = journal
                .targets
                .get_mut(&PublicationDestination::Private)
                .unwrap();
            target.start().unwrap();
            let digest = target.batches[0].plan.parts[0].content_digest.clone();
            target
                .mark_batch_committed(0, "head-0", "head-1", vec![digest])
                .unwrap();
        }
        mark_batch_record_verified(&mut journal);
        let report = abandon_publication_target(
            &ReceiptRemote { receipt: None },
            &store,
            &CancellationToken::new(),
            &session(),
            &mut journal,
            PublicationDestination::Private,
            "operator cancelled publication",
        )
        .unwrap();
        assert_eq!(report.newly_committed_bytes_retained, 7);
        assert_eq!(report.reachable_commit_ids, vec!["head-1"]);
        assert!(report.temporary_reservation_released);
        assert!(report.local_staging_cleanup_eligible);
        assert_eq!(
            store.load_latest(&journal.transaction_id).unwrap().targets
                [&PublicationDestination::Private]
                .state,
            PublicationTargetState::Abandoned
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn abandon_refuses_a_receipt_that_may_already_be_discoverable() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("abandon-receipt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = PublicationJournalStore::new(&root);
        let mut journal = journal();
        let error = abandon_publication_target(
            &ReceiptRemote {
                receipt: Some(ContentDigest::sha256(b"receipt")),
            },
            &store,
            &CancellationToken::new(),
            &session(),
            &mut journal,
            PublicationDestination::Private,
            "operator cancelled publication",
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::InvalidTransition(_)));
        assert_eq!(
            journal.targets[&PublicationDestination::Private].state,
            PublicationTargetState::Planned
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
