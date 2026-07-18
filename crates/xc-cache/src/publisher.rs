//! Durable append-only journal checkpoints and transport orchestration.

use crate::{
    build_payload_batch_record, revalidate_publication_capacity, stage_payload_batch_record,
    AuthenticatedGitHubSession, CacheError, CompareAndSwapResult, ContentDigest,
    PublicationBatchState, PublicationDestination, PublicationFinalizationPolicy,
    PublicationTargetState, PublicationTransactionJournal, RemoteCommitRequest, RemoteGitStore,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug)]
pub struct PublicationJournalStore {
    root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JournalCheckpoint {
    schema_version: u32,
    sequence: u64,
    journal_digest: ContentDigest,
    journal: PublicationTransactionJournal,
}

impl PublicationJournalStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn save(&self, journal: &PublicationTransactionJournal) -> Result<PathBuf, CacheError> {
        journal.validate()?;
        let directory = self.transaction_directory(&journal.transaction_id)?;
        fs::create_dir_all(&directory)?;
        let sequence = next_checkpoint_sequence(&directory)?;
        let digest = journal.digest()?;
        let checkpoint = JournalCheckpoint {
            schema_version: 1,
            sequence,
            journal_digest: digest.clone(),
            journal: journal.clone(),
        };
        let path = directory.join(format!("{sequence:020}-{}.json", digest.0));
        let bytes = serde_json::to_vec_pretty(&checkpoint)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(path)
    }

    pub fn load_latest(
        &self,
        transaction_id: &str,
    ) -> Result<PublicationTransactionJournal, CacheError> {
        let directory = self.transaction_directory(transaction_id)?;
        self.load_latest_from_directory(transaction_id, &directory)
    }

    pub fn load_if_exists(
        &self,
        transaction_id: &str,
    ) -> Result<Option<PublicationTransactionJournal>, CacheError> {
        let directory = self.transaction_directory(transaction_id)?;
        if !directory.exists() {
            return Ok(None);
        }
        self.load_latest_from_directory(transaction_id, &directory)
            .map(Some)
    }

    fn load_latest_from_directory(
        &self,
        transaction_id: &str,
        directory: &Path,
    ) -> Result<PublicationTransactionJournal, CacheError> {
        let mut paths: Vec<_> = fs::read_dir(directory)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .collect();
        paths.sort();
        let path = paths.last().ok_or_else(|| {
            CacheError::NotFound(format!(
                "publication journal checkpoint for {transaction_id}"
            ))
        })?;
        let checkpoint: JournalCheckpoint = serde_json::from_slice(&fs::read(path)?)?;
        if checkpoint.schema_version != 1
            || checkpoint.journal.transaction_id != transaction_id
            || checkpoint.journal.digest()? != checkpoint.journal_digest
        {
            return Err(CacheError::InvalidManifest(
                "publication journal checkpoint failed identity verification".to_owned(),
            ));
        }
        Ok(checkpoint.journal)
    }

    fn transaction_directory(&self, transaction_id: &str) -> Result<PathBuf, CacheError> {
        if transaction_id.len() != 64
            || !transaction_id
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CacheError::InvalidManifest(
                "publication transaction id must be a lowercase SHA-256 digest".to_owned(),
            ));
        }
        Ok(self.root.join(transaction_id).join("checkpoints"))
    }
}

fn next_checkpoint_sequence(directory: &Path) -> Result<u64, CacheError> {
    let mut maximum = None;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(sequence) = name
            .split_once('-')
            .and_then(|(sequence, _)| sequence.parse::<u64>().ok())
        {
            maximum = Some(maximum.map_or(sequence, |current: u64| current.max(sequence)));
        }
    }
    maximum
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| CacheError::ResourceLimit("journal sequence exhausted u64".to_owned()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicationStepOutcome {
    RefConflict { current_head: String },
    ExistingBatchVerified { sequence: u64 },
    PayloadVerifiedAndRecordStaged { sequence: u64, commit_id: String },
    ExistingBatchRecordVerified { sequence: u64 },
    BatchRecordCommittedAndVerified { sequence: u64, commit_id: String },
    PayloadBatchesComplete,
}

/// Execute at most one remote mutation. Every accepted remote mutation is
/// checkpointed before independent verification. Callers re-run authority and
/// capacity checks before each invocation.
#[allow(clippy::too_many_arguments)]
pub fn execute_next_payload_batch(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    authenticated_session: &AuthenticatedGitHubSession,
    capacity_policy: &PublicationFinalizationPolicy,
    staging_root: &Path,
    resources: &ResourcePolicy,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
) -> Result<PublicationStepOutcome, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    let target = journal.targets.get(&destination).ok_or_else(|| {
        CacheError::InvalidTransition(format!(
            "transaction has no {destination:?} publication target"
        ))
    })?;
    authenticated_session.require_write_for(
        &target.permission_evidence.principal,
        &target.authorized_repository,
    )?;
    let refreshed_permission = target.permission_evidence != *authenticated_session.evidence();
    if refreshed_permission {
        journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked")
            .permission_evidence = authenticated_session.evidence().clone();
        checkpoints.save(journal)?;
    }
    let target = journal
        .targets
        .get(&destination)
        .expect("target existence was checked");

    if let Some(sequence) = target
        .batches
        .iter()
        .position(|batch| batch.state == PublicationBatchState::RecordCommitted)
    {
        let record = target.batches[sequence]
            .record_commit
            .as_ref()
            .expect("record-committed batch has a record journal");
        let commit_id = record.commit_id.clone().ok_or_else(|| {
            CacheError::InvalidTransition("committed batch record has no commit id".to_owned())
        })?;
        let record_file = record.files[0].clone();
        let repository = target.repository.clone();
        remote.verify_committed_part(&repository, &commit_id, &record_file)?;
        journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked")
            .mark_batch_record_verified(sequence, &commit_id)?;
        checkpoints.save(journal)?;
        return Ok(PublicationStepOutcome::BatchRecordCommittedAndVerified {
            sequence: sequence as u64,
            commit_id,
        });
    }

    if let Some(sequence) = target
        .batches
        .iter()
        .position(|batch| batch.state == PublicationBatchState::PayloadVerified)
    {
        let expected_head = target.expected_head.clone();
        let repository = target.repository.clone();
        let branch = target.branch.clone();
        let payload_parts = target.batches[sequence].plan.parts.clone();
        let record_file = target.batches[sequence]
            .record_commit
            .as_ref()
            .expect("payload-verified batch has a planned record")
            .files[0]
            .clone();
        for part in &payload_parts {
            remote.verify_committed_part(&repository, &expected_head, part)?;
        }
        match remote.immutable_path_digest(
            &repository,
            &expected_head,
            &record_file.repository_path,
        )? {
            Some(digest) if digest == record_file.content_digest => {
                journal
                    .targets
                    .get_mut(&destination)
                    .expect("target existence was checked")
                    .mark_batch_record_reused(sequence, &expected_head)?;
                checkpoints.save(journal)?;
                return Ok(PublicationStepOutcome::ExistingBatchRecordVerified {
                    sequence: sequence as u64,
                });
            }
            Some(actual) => {
                return Err(CacheError::DigestMismatch {
                    expected: record_file.content_digest.to_string(),
                    actual: actual.to_string(),
                });
            }
            None => {}
        }
        revalidate_publication_capacity(
            remote,
            journal,
            destination,
            capacity_policy,
            0,
            cancellation,
        )?;
        let request = RemoteCommitRequest {
            repository: repository.clone(),
            branch,
            expected_head: expected_head.clone(),
            message: format!(
                "cache publication {} batch {} durable record",
                journal.idempotency_key, sequence
            ),
            parts: vec![record_file.clone()],
            delete_paths: Vec::new(),
        };
        return match remote.compare_and_swap_commit(&request)? {
            CompareAndSwapResult::RefConflict { current_head } => {
                journal
                    .targets
                    .get_mut(&destination)
                    .expect("target existence was checked")
                    .record_ref_conflict(current_head.clone())?;
                checkpoints.save(journal)?;
                Ok(PublicationStepOutcome::RefConflict { current_head })
            }
            CompareAndSwapResult::Committed { commit_id } => {
                journal
                    .targets
                    .get_mut(&destination)
                    .expect("target existence was checked")
                    .mark_batch_record_committed(
                        sequence,
                        &expected_head,
                        &commit_id,
                        vec![record_file.content_digest.clone()],
                    )?;
                checkpoints.save(journal)?;
                remote.verify_committed_part(&repository, &commit_id, &record_file)?;
                journal
                    .targets
                    .get_mut(&destination)
                    .expect("target existence was checked")
                    .mark_batch_record_verified(sequence, &commit_id)?;
                checkpoints.save(journal)?;
                Ok(PublicationStepOutcome::BatchRecordCommittedAndVerified {
                    sequence: sequence as u64,
                    commit_id,
                })
            }
        };
    }

    if let Some(sequence) = target
        .batches
        .iter()
        .position(|batch| batch.state == PublicationBatchState::Committed)
    {
        let commit_id = target.batches[sequence].commit_id.clone().ok_or_else(|| {
            CacheError::InvalidTransition("committed batch has no commit id".to_owned())
        })?;
        let repository = target.repository.clone();
        let parts = target.batches[sequence].plan.parts.clone();
        for part in &parts {
            remote.verify_committed_part(&repository, &commit_id, part)?;
        }
        stage_verified_payload_record(
            checkpoints,
            cancellation,
            staging_root,
            resources,
            journal,
            destination,
            sequence,
            &commit_id,
        )?;
        return Ok(PublicationStepOutcome::PayloadVerifiedAndRecordStaged {
            sequence: sequence as u64,
            commit_id,
        });
    }

    let Some(sequence) = target
        .batches
        .iter()
        .position(|batch| batch.state == PublicationBatchState::Planned)
    else {
        return Ok(PublicationStepOutcome::PayloadBatchesComplete);
    };
    if target.state != PublicationTargetState::Uploading {
        journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked")
            .start()?;
        checkpoints.save(journal)?;
    }
    let target = journal
        .targets
        .get(&destination)
        .expect("target existence was checked");
    let expected_head = target.expected_head.clone();
    let repository = target.repository.clone();
    let branch = target.branch.clone();
    let parts = target.batches[sequence].plan.parts.clone();
    let mut missing = Vec::new();
    let mut missing_paths = std::collections::BTreeSet::new();
    for part in &parts {
        match remote.immutable_path_digest(&repository, &expected_head, &part.repository_path)? {
            Some(digest) if digest == part.content_digest => {}
            Some(actual) => {
                return Err(CacheError::DigestMismatch {
                    expected: part.content_digest.to_string(),
                    actual: actual.to_string(),
                });
            }
            None if missing_paths.insert(part.repository_path.clone()) => {
                missing.push(part.clone());
            }
            None => {}
        }
    }
    if missing.is_empty() {
        journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked")
            .mark_batch_reused(sequence, &expected_head)?;
        checkpoints.save(journal)?;
        return Ok(PublicationStepOutcome::ExistingBatchVerified {
            sequence: sequence as u64,
        });
    }

    let pending_unique_payload_bytes = missing
        .iter()
        .map(|part| part.size_bytes)
        .fold(0u64, u64::saturating_add);
    revalidate_publication_capacity(
        remote,
        journal,
        destination,
        capacity_policy,
        pending_unique_payload_bytes,
        cancellation,
    )?;

    let request = RemoteCommitRequest {
        repository: repository.clone(),
        branch,
        expected_head: expected_head.clone(),
        message: format!(
            "cache publication {} batch {}",
            journal.idempotency_key, sequence
        ),
        parts: missing,
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            journal
                .targets
                .get_mut(&destination)
                .expect("target existence was checked")
                .record_ref_conflict(current_head.clone())?;
            checkpoints.save(journal)?;
            Ok(PublicationStepOutcome::RefConflict { current_head })
        }
        CompareAndSwapResult::Committed { commit_id } => {
            let newly_committed_digests = request
                .parts
                .iter()
                .map(|part| part.content_digest.clone())
                .collect();
            journal
                .targets
                .get_mut(&destination)
                .expect("target existence was checked")
                .mark_batch_committed(
                    sequence,
                    &expected_head,
                    &commit_id,
                    newly_committed_digests,
                )?;
            checkpoints.save(journal)?;
            for part in &parts {
                remote.verify_committed_part(&repository, &commit_id, part)?;
            }
            stage_verified_payload_record(
                checkpoints,
                cancellation,
                staging_root,
                resources,
                journal,
                destination,
                sequence,
                &commit_id,
            )?;
            Ok(PublicationStepOutcome::PayloadVerifiedAndRecordStaged {
                sequence: sequence as u64,
                commit_id,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_verified_payload_record(
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    staging_root: &Path,
    resources: &ResourcePolicy,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    sequence: usize,
    payload_commit_id: &str,
) -> Result<(), CacheError> {
    let record = build_payload_batch_record(journal, destination, sequence)?;
    let record_file = stage_payload_batch_record(staging_root, resources, cancellation, &record)?;
    journal
        .targets
        .get_mut(&destination)
        .expect("target existence was checked")
        .plan_verified_batch_record(sequence, payload_commit_id, record_file)?;
    checkpoints.save(journal)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plan_publication_batches, CapacityLedger, PublicationBatchPlan,
        PublicationTransactionJournal, TransportPart, TransportPolicy,
        DEFAULT_CAPACITY_LEDGER_PATH, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;
    use xc_core::{CancellationReason, PublicationTarget};

    fn parts() -> Vec<TransportPart> {
        [b"first".as_slice(), b"second".as_slice()]
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                let digest = ContentDigest::sha256(bytes);
                TransportPart {
                    sequence: index as u64,
                    repository_path: format!("objects/{}.part", digest.0),
                    size_bytes: bytes.len() as u64,
                    content_digest: digest,
                }
            })
            .collect()
    }

    fn batches() -> Vec<PublicationBatchPlan> {
        plan_publication_batches(
            &parts(),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 10,
                split_part_bytes: 5,
                maximum_batch_payload_bytes: 6,
                maximum_pending_batches: 1,
            },
        )
        .unwrap()
    }

    fn journal() -> PublicationTransactionJournal {
        let permission = crate::AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            crate::RepositoryPermission::Write,
        )
        .evidence()
        .clone();
        PublicationTransactionJournal::new(
            ContentDigest::sha256(b"semantic"),
            BTreeMap::from([(
                PublicationDestination::Private,
                ContentDigest::sha256(b"private-manifest"),
            )]),
            ContentDigest::sha256(b"payload"),
            ContentDigest::sha256(b"policy"),
            PublicationTarget::Private,
            BTreeMap::from([(PublicationDestination::Private, "team/private".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "team/private".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, permission)]),
            BTreeMap::from([(PublicationDestination::Private, "private-001".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "main".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "head-0".to_owned())]),
            &batches(),
        )
        .unwrap()
    }

    fn authenticated_session() -> AuthenticatedGitHubSession {
        AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            crate::RepositoryPermission::Write,
        )
    }

    fn temporary_root(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("{name}-{}", std::process::id()))
    }

    #[test]
    fn append_only_checkpoint_store_loads_latest_verified_snapshot() {
        let root = temporary_root("publication-journal");
        let _ = fs::remove_dir_all(&root);
        let store = PublicationJournalStore::new(&root);
        let mut journal = journal();
        assert_eq!(store.load_if_exists(&journal.transaction_id).unwrap(), None);
        store.save(&journal).unwrap();
        journal
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap()
            .start()
            .unwrap();
        store.save(&journal).unwrap();
        assert_eq!(
            store.load_if_exists(&journal.transaction_id).unwrap(),
            Some(journal.clone())
        );
        assert_eq!(store.load_latest(&journal.transaction_id).unwrap(), journal);
        let _ = fs::remove_dir_all(root);
    }

    struct FakeRemote {
        head: Mutex<String>,
        paths: Mutex<HashMap<(String, String), ContentDigest>>,
        conflict_once: Mutex<bool>,
        calls: Mutex<usize>,
    }

    impl FakeRemote {
        fn new() -> Self {
            Self {
                head: Mutex::new("head-0".to_owned()),
                paths: Mutex::new(HashMap::new()),
                conflict_once: Mutex::new(true),
                calls: Mutex::new(0),
            }
        }
    }

    impl RemoteGitStore for FakeRemote {
        fn read_ref(&self, _repository: &str, _branch: &str) -> Result<String, CacheError> {
            *self.calls.lock().unwrap() += 1;
            Ok(self.head.lock().unwrap().clone())
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            *self.calls.lock().unwrap() += 1;
            Ok(self
                .paths
                .lock()
                .unwrap()
                .get(&(revision.to_owned(), path.to_owned()))
                .cloned())
        }

        fn read_committed_path(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
            maximum_bytes: u64,
            cancellation: &CancellationToken,
            writer: &mut dyn Write,
        ) -> Result<crate::RemoteReadReport, CacheError> {
            *self.calls.lock().unwrap() += 1;
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            if path != DEFAULT_CAPACITY_LEDGER_PATH {
                return Err(CacheError::NotFound(format!("{revision}:{path}")));
            }
            let bytes = serde_json::to_vec(&CapacityLedger {
                schema_version: 1,
                shard_id: "private-001".to_owned(),
                hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
                warning_reserve_bytes: 1_000_000,
                first_seen_immutable_payload_bytes: 0,
                manifest_index_receipt_bytes: 0,
                estimated_history_bytes: 0,
                emergency_reserve_bytes: 0,
                abandoned_reachable_bytes: 0,
                last_reconciled_commit: revision.to_owned(),
                reconciliation_digest: ContentDigest::sha256(b"publisher-test-ledger"),
            })?;
            if bytes.len() as u64 > maximum_bytes {
                return Err(CacheError::ResourceLimit(path.to_owned()));
            }
            writer.write_all(&bytes)?;
            Ok(crate::RemoteReadReport {
                repository_path: path.to_owned(),
                revision: revision.to_owned(),
                size_bytes: bytes.len() as u64,
                content_digest: ContentDigest::sha256(&bytes),
            })
        }

        fn compare_and_swap_commit(
            &self,
            request: &RemoteCommitRequest,
        ) -> Result<CompareAndSwapResult, CacheError> {
            *self.calls.lock().unwrap() += 1;
            let mut conflict = self.conflict_once.lock().unwrap();
            if *conflict {
                *conflict = false;
                *self.head.lock().unwrap() = "head-1".to_owned();
                return Ok(CompareAndSwapResult::RefConflict {
                    current_head: "head-1".to_owned(),
                });
            }
            let mut head = self.head.lock().unwrap();
            if *head != request.expected_head {
                return Ok(CompareAndSwapResult::RefConflict {
                    current_head: head.clone(),
                });
            }
            let commit = format!("commit-{}", request.expected_head);
            let mut paths = self.paths.lock().unwrap();
            let inherited = paths
                .iter()
                .filter(|((revision, _), _)| revision == &request.expected_head)
                .map(|((_, path), digest)| (path.clone(), digest.clone()))
                .collect::<Vec<_>>();
            for (path, digest) in inherited {
                paths.insert((commit.clone(), path), digest);
            }
            for part in &request.parts {
                paths.insert(
                    (commit.clone(), part.repository_path.clone()),
                    part.content_digest.clone(),
                );
            }
            *head = commit.clone();
            Ok(CompareAndSwapResult::Committed { commit_id: commit })
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            revision: &str,
            part: &TransportPart,
        ) -> Result<(), CacheError> {
            *self.calls.lock().unwrap() += 1;
            let actual = self
                .paths
                .lock()
                .unwrap()
                .get(&(revision.to_owned(), part.repository_path.clone()))
                .cloned()
                .ok_or_else(|| CacheError::NotFound(part.repository_path.clone()))?;
            if actual == part.content_digest {
                Ok(())
            } else {
                Err(CacheError::DigestMismatch {
                    expected: part.content_digest.to_string(),
                    actual: actual.to_string(),
                })
            }
        }
    }

    #[test]
    fn executor_retries_ref_conflict_then_verifies_each_batch() {
        let remote = FakeRemote::new();
        let mut journal = journal();
        let root = temporary_root("publication-executor");
        let _ = fs::remove_dir_all(&root);
        let checkpoints = PublicationJournalStore::new(&root);
        let staging = root.join("staging");
        let resources = ResourcePolicy::default();
        let first = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(first, PublicationStepOutcome::RefConflict { .. }));
        let second = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            second,
            PublicationStepOutcome::PayloadVerifiedAndRecordStaged { sequence: 0, .. }
        ));
        let third = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            third,
            PublicationStepOutcome::BatchRecordCommittedAndVerified { sequence: 0, .. }
        ));
        let fourth = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            fourth,
            PublicationStepOutcome::PayloadVerifiedAndRecordStaged { sequence: 1, .. }
        ));
        let fifth = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        )
        .unwrap();
        assert!(matches!(
            fifth,
            PublicationStepOutcome::BatchRecordCommittedAndVerified { sequence: 1, .. }
        ));
        assert!(journal.targets[&PublicationDestination::Private]
            .batches
            .iter()
            .all(|batch| batch.state == PublicationBatchState::RemoteVerified));
        assert_eq!(
            checkpoints.load_latest(&journal.transaction_id).unwrap(),
            journal
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn executor_honors_cancellation_before_remote_mutation() {
        let remote = FakeRemote::new();
        let mut journal = journal();
        let root = temporary_root("publication-executor-cancel");
        let _ = fs::remove_dir_all(&root);
        let checkpoints = PublicationJournalStore::new(&root);
        let staging = root.join("staging");
        let resources = ResourcePolicy::default();
        let cancellation = CancellationToken::new();
        cancellation.cancel(CancellationReason::UserRequested);
        let result = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &cancellation,
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        );
        assert!(matches!(result, Err(CacheError::Cancelled(_))));
        assert_eq!(remote.read_ref("team/private", "main").unwrap(), "head-0");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn elapsed_budget_stops_publication_before_remote_mutation() {
        let remote = FakeRemote::new();
        let mut journal = journal();
        let root = temporary_root("publication-executor-deadline");
        let _ = fs::remove_dir_all(&root);
        let checkpoints = PublicationJournalStore::new(&root);
        let staging = root.join("staging");
        let resources = ResourcePolicy {
            maximum_wall_seconds: Some(0),
            ..ResourcePolicy::default()
        };
        let cancellation = CancellationToken::for_policy(&resources);
        let result = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &cancellation,
            &authenticated_session(),
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        );
        assert!(matches!(result, Err(CacheError::Cancelled(_))));
        assert!(matches!(
            cancellation.state().reason,
            Some(CancellationReason::ResourceBudgetReached(
                xc_core::ResourceKind::WallTime
            ))
        ));
        assert_eq!(remote.read_ref("team/private", "main").unwrap(), "head-0");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn executor_requires_live_write_permission_before_remote_access() {
        let remote = FakeRemote::new();
        let mut journal = journal();
        let root = temporary_root("publication-executor-permission");
        let _ = fs::remove_dir_all(&root);
        let checkpoints = PublicationJournalStore::new(&root);
        let staging = root.join("staging");
        let resources = ResourcePolicy::default();
        let read_only = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            crate::RepositoryPermission::Read,
        );
        let result = execute_next_payload_batch(
            &remote,
            &checkpoints,
            &CancellationToken::new(),
            &read_only,
            &PublicationFinalizationPolicy::default(),
            &staging,
            &resources,
            &mut journal,
            PublicationDestination::Private,
        );
        assert!(matches!(result, Err(CacheError::PermissionDenied(_))));
        assert_eq!(*remote.calls.lock().unwrap(), 0);
        let _ = fs::remove_dir_all(root);
    }
}
