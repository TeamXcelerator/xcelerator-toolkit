//! Explicit, fail-closed publication of audit-rebuilt shard index projections.

use crate::protocol::{canonical_digest, canonical_json_bytes, normalized_relative_path};
use crate::{
    stage_publication_bytes, AuthenticatedGitHubSession, CacheError, CompareAndSwapResult,
    ContentDigest, RemoteCommitRequest, RemoteGitStore, RemoteShardAuditReport,
    ShardAuditIssueKind, ShardAuditSeverity, ShardIndexPartition,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardIndexRepairPolicy {
    pub allow_entry_removal: bool,
    pub allow_repair_with_unresolved_errors: bool,
    pub maximum_repair_files: u64,
    pub maximum_repair_bytes: u64,
}

impl ShardIndexRepairPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.maximum_repair_files == 0 || self.maximum_repair_bytes == 0 {
            return Err(CacheError::ResourceLimit(
                "shard index repair bounds must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardIndexRepairDocument {
    pub repository_path: String,
    pub observed_digest: ContentDigest,
    pub replacement_digest: ContentDigest,
    pub replacement_size_bytes: u64,
    pub partition: ShardIndexPartition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardIndexRepairPlan {
    pub schema_version: u32,
    pub repository: String,
    pub authorized_repository: String,
    pub branch: String,
    pub expected_head: String,
    pub shard_id: String,
    pub audited_index_entry_count: u64,
    pub replacement_index_entry_count: u64,
    pub removes_entries: bool,
    pub documents: Vec<ShardIndexRepairDocument>,
}

impl ShardIndexRepairPlan {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.repository.trim().is_empty()
            || self.authorized_repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.expected_head.trim().is_empty()
            || self.shard_id.trim().is_empty()
            || self.removes_entries
                != (self.replacement_index_entry_count < self.audited_index_entry_count)
        {
            return Err(CacheError::InvalidManifest(
                "shard index repair plan identity or entry accounting is invalid".to_owned(),
            ));
        }
        let mut previous_path: Option<&str> = None;
        let mut replacement_entries = 0u64;
        for document in &self.documents {
            document.partition.validate()?;
            let expected_path = format!(
                "indexes/{}/{}.json",
                document.partition.family, document.partition.semantic_prefix
            );
            let bytes = canonical_json_bytes(&document.partition)?;
            if !normalized_relative_path(&document.repository_path)
                || document.repository_path != expected_path
                || previous_path
                    .is_some_and(|previous| previous >= document.repository_path.as_str())
                || !document.observed_digest.validate()
                || !document.replacement_digest.validate()
                || document.observed_digest == document.replacement_digest
                || document.replacement_digest != ContentDigest::sha256(&bytes)
                || document.replacement_size_bytes != bytes.len() as u64
                || document.replacement_size_bytes == 0
            {
                return Err(CacheError::InvalidManifest(
                    "shard index repair document is invalid, unchanged, or unordered".to_owned(),
                ));
            }
            previous_path = Some(&document.repository_path);
            replacement_entries = replacement_entries
                .checked_add(document.partition.entries.len() as u64)
                .ok_or_else(|| {
                    CacheError::ResourceLimit("repair entry count exceeds u64".to_owned())
                })?;
        }
        let unchanged_entries = self
            .replacement_index_entry_count
            .checked_sub(replacement_entries)
            .ok_or_else(|| {
                CacheError::InvalidManifest(
                    "repair documents exceed replacement entry count".to_owned(),
                )
            })?;
        if unchanged_entries > self.audited_index_entry_count {
            return Err(CacheError::InvalidManifest(
                "repair unchanged-entry accounting is invalid".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn total_repair_bytes(&self) -> u64 {
        self.documents
            .iter()
            .map(|document| document.replacement_size_bytes)
            .fold(0u64, u64::saturating_add)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardIndexRepairOutcome {
    NoChanges,
    RefConflict {
        current_head: String,
    },
    CommittedAndVerified {
        commit_id: String,
        plan_digest: ContentDigest,
        principal: String,
        repository_permission_evidence_digest: ContentDigest,
        repaired_paths: Vec<String>,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn plan_shard_index_repair(
    remote: &dyn RemoteGitStore,
    report: &RemoteShardAuditReport,
    authorized_repository: &str,
    policy: &ShardIndexRepairPolicy,
    cancellation: &CancellationToken,
) -> Result<ShardIndexRepairPlan, CacheError> {
    policy.validate()?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if authorized_repository.trim().is_empty() {
        return Err(CacheError::InvalidManifest(
            "authorized shard repair repository is required".to_owned(),
        ));
    }
    if report.schema_version == 0
        || report.repository.trim().is_empty()
        || report.branch.trim().is_empty()
        || report.revision.trim().is_empty()
        || report.shard_id.trim().is_empty()
    {
        return Err(CacheError::InvalidManifest(
            "shard audit report identity is incomplete".to_owned(),
        ));
    }
    let unresolved_errors = report.issues.iter().filter(|issue| {
        issue.severity == ShardAuditSeverity::Error
            && !matches!(
                issue.kind,
                ShardAuditIssueKind::ProjectionDrift | ShardAuditIssueKind::LedgerDrift
            )
    });
    if !policy.allow_repair_with_unresolved_errors {
        if let Some(issue) = unresolved_errors.into_iter().next() {
            return Err(CacheError::InvalidTransition(format!(
                "index repair is blocked by unresolved {:?}: {}",
                issue.kind, issue.reason
            )));
        }
    }
    let drift_paths = report
        .issues
        .iter()
        .filter(|issue| issue.kind == ShardAuditIssueKind::ProjectionDrift)
        .filter_map(|issue| issue.repository_path.clone())
        .collect::<BTreeSet<_>>();
    let replacement_index_entry_count =
        report
            .reconstructed_partitions
            .iter()
            .try_fold(0u64, |total, partition| {
                total
                    .checked_add(partition.entries.len() as u64)
                    .ok_or_else(|| {
                        CacheError::ResourceLimit(
                            "rebuilt index entry count exceeds u64".to_owned(),
                        )
                    })
            })?;
    let removes_entries = replacement_index_entry_count < report.index_entry_count;
    if removes_entries && !policy.allow_entry_removal {
        return Err(CacheError::InvalidTransition(format!(
            "audit rebuild would remove {} index entries; explicit removal authorization is required",
            report
                .index_entry_count
                .saturating_sub(replacement_index_entry_count)
        )));
    }
    let mut documents = Vec::new();
    for partition in &report.reconstructed_partitions {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let repository_path = format!(
            "indexes/{}/{}.json",
            partition.family, partition.semantic_prefix
        );
        if !drift_paths.contains(&repository_path) {
            continue;
        }
        let observed_digest = remote
            .immutable_path_digest(&report.repository, &report.revision, &repository_path)?
            .ok_or_else(|| CacheError::NotFound(repository_path.clone()))?;
        let bytes = canonical_json_bytes(partition)?;
        let replacement_digest = ContentDigest::sha256(&bytes);
        if observed_digest == replacement_digest {
            continue;
        }
        documents.push(ShardIndexRepairDocument {
            repository_path,
            observed_digest,
            replacement_digest,
            replacement_size_bytes: bytes.len() as u64,
            partition: partition.clone(),
        });
    }
    documents.sort_by(|left, right| left.repository_path.cmp(&right.repository_path));
    let planned_paths = documents
        .iter()
        .map(|document| document.repository_path.clone())
        .collect::<BTreeSet<_>>();
    if planned_paths != drift_paths {
        return Err(CacheError::InvalidTransition(
            "one or more drifted index paths cannot be reconstructed safely".to_owned(),
        ));
    }
    let plan = ShardIndexRepairPlan {
        schema_version: 1,
        repository: report.repository.clone(),
        authorized_repository: authorized_repository.to_owned(),
        branch: report.branch.clone(),
        expected_head: report.revision.clone(),
        shard_id: report.shard_id.clone(),
        audited_index_entry_count: report.index_entry_count,
        replacement_index_entry_count,
        removes_entries,
        documents,
    };
    plan.validate()?;
    if plan.documents.len() as u64 > policy.maximum_repair_files
        || plan.total_repair_bytes() > policy.maximum_repair_bytes
    {
        return Err(CacheError::ResourceLimit(
            "shard index repair exceeds the configured file or byte bound".to_owned(),
        ));
    }
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_shard_index_repair(
    remote: &dyn RemoteGitStore,
    session: &AuthenticatedGitHubSession,
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    plan: &ShardIndexRepairPlan,
) -> Result<ShardIndexRepairOutcome, CacheError> {
    plan.validate()?;
    session.require_write_for(
        session.evidence().principal.as_str(),
        &plan.authorized_repository,
    )?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if plan.documents.is_empty() {
        return Ok(ShardIndexRepairOutcome::NoChanges);
    }
    let current_head = remote.read_ref(&plan.repository, &plan.branch)?;
    if current_head != plan.expected_head {
        return Ok(ShardIndexRepairOutcome::RefConflict { current_head });
    }
    let mut parts = Vec::with_capacity(plan.documents.len());
    let mut staged_bytes = 0u64;
    for (sequence, document) in plan.documents.iter().enumerate() {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        match remote.immutable_path_digest(
            &plan.repository,
            &plan.expected_head,
            &document.repository_path,
        )? {
            Some(actual) if actual == document.observed_digest => {}
            Some(actual) => {
                return Err(CacheError::DigestMismatch {
                    expected: document.observed_digest.to_string(),
                    actual: actual.to_string(),
                });
            }
            None => return Err(CacheError::NotFound(document.repository_path.clone())),
        }
        let bytes = canonical_json_bytes(&document.partition)?;
        let mut part = stage_publication_bytes(
            staging_root,
            &document.repository_path,
            &bytes,
            resources,
            cancellation,
            &mut staged_bytes,
        )?;
        part.sequence = sequence as u64;
        if part.content_digest != document.replacement_digest
            || part.size_bytes != document.replacement_size_bytes
        {
            return Err(CacheError::DigestMismatch {
                expected: document.replacement_digest.to_string(),
                actual: part.content_digest.to_string(),
            });
        }
        parts.push(part);
    }
    let plan_digest = plan.digest()?;
    let request = RemoteCommitRequest {
        repository: plan.repository.clone(),
        branch: plan.branch.clone(),
        expected_head: plan.expected_head.clone(),
        message: format!("repair shard index projection {plan_digest}"),
        parts: parts.clone(),
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            Ok(ShardIndexRepairOutcome::RefConflict { current_head })
        }
        CompareAndSwapResult::Committed { commit_id } => {
            for part in &parts {
                remote.verify_committed_part(&plan.repository, &commit_id, part)?;
            }
            Ok(ShardIndexRepairOutcome::CommittedAndVerified {
                commit_id,
                plan_digest,
                principal: session.evidence().principal.clone(),
                repository_permission_evidence_digest: session.evidence().evidence_digest.clone(),
                repaired_paths: parts.into_iter().map(|part| part.repository_path).collect(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RemoteReadReport, RepositoryPermission, ShardAuditIssue, ShardCapacityAudit};
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::Mutex;

    struct MemoryRemote {
        head: Mutex<String>,
        paths: Mutex<BTreeMap<(String, String), ContentDigest>>,
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
                .paths
                .lock()
                .unwrap()
                .get(&(revision.to_owned(), path.to_owned()))
                .cloned())
        }

        fn read_committed_path(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
            _maximum_bytes: u64,
            _cancellation: &CancellationToken,
            _writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            Err(CacheError::NotFound(path.to_owned()))
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
            let commit_id = "repair-commit".to_owned();
            let mut paths = self.paths.lock().unwrap();
            for part in &request.parts {
                paths.insert(
                    (commit_id.clone(), part.repository_path.clone()),
                    part.content_digest.clone(),
                );
            }
            *self.head.lock().unwrap() = commit_id.clone();
            Ok(CompareAndSwapResult::Committed { commit_id })
        }

        fn verify_committed_part(
            &self,
            repository: &str,
            revision: &str,
            part: &crate::TransportPart,
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

    fn audit_report(observed: &ContentDigest) -> (MemoryRemote, RemoteShardAuditReport) {
        let partition = ShardIndexPartition::rebuild("fixture", "aa", Vec::new()).unwrap();
        let path = "indexes/fixture/aa.json".to_owned();
        let head = "a".repeat(40);
        let remote = MemoryRemote {
            head: Mutex::new(head.clone()),
            paths: Mutex::new(BTreeMap::from([(
                (head.clone(), path.clone()),
                observed.clone(),
            )])),
        };
        let report = RemoteShardAuditReport {
            schema_version: 1,
            repository: "memory".to_owned(),
            branch: "main".to_owned(),
            revision: head,
            shard_id: "fixture-001".to_owned(),
            listed_path_count: 1,
            listed_path_bytes: path.len() as u64,
            verified_metadata_bytes_read: 1,
            index_partition_count: 1,
            index_entry_count: 1,
            complete_artifact_count: 0,
            logical_payload_bytes: 0,
            unique_transport_object_count: 0,
            unique_transport_object_bytes: 0,
            unreferenced_manifest_count: 0,
            unreferenced_encoding_count: 0,
            unreferenced_receipt_count: 0,
            unreferenced_object_count: 0,
            reconstructed_partitions: vec![partition],
            capacity_ledger: None,
            capacity: ShardCapacityAudit {
                ledger_accounted_bytes: None,
                ledger_first_seen_immutable_payload_bytes: None,
                referenced_unique_transport_bytes: 0,
                ledger_covers_referenced_transport: false,
                ledger_remaining_capacity_bytes: None,
                durable_batch_record_count: 0,
                durable_recorded_new_payload_bytes: 0,
                durable_record_coverage_complete: true,
                ledger_matches_durable_payload_history: false,
                exact_history_rebuild_available: true,
            },
            issues: vec![ShardAuditIssue {
                severity: ShardAuditSeverity::Error,
                kind: ShardAuditIssueKind::ProjectionDrift,
                repository_path: Some(path),
                identity_digest: None,
                reason: "fixture drift".to_owned(),
            }],
        };
        (remote, report)
    }

    fn policy(allow_entry_removal: bool) -> ShardIndexRepairPolicy {
        ShardIndexRepairPolicy {
            allow_entry_removal,
            allow_repair_with_unresolved_errors: false,
            maximum_repair_files: 10,
            maximum_repair_bytes: 1024 * 1024,
        }
    }

    #[test]
    fn repair_requires_explicit_authority_to_remove_entries() {
        let observed = ContentDigest::sha256(b"old-index");
        let (remote, report) = audit_report(&observed);
        let error = plan_shard_index_repair(
            &remote,
            &report,
            "team/shard",
            &policy(false),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::InvalidTransition(_)));
    }

    #[test]
    fn authorized_repair_is_one_verified_compare_and_swap_commit() {
        let observed = ContentDigest::sha256(b"old-index");
        let (remote, report) = audit_report(&observed);
        let plan = plan_shard_index_repair(
            &remote,
            &report,
            "team/shard",
            &policy(true),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(plan.removes_entries);
        let session = AuthenticatedGitHubSession::verified_for_test(
            "auditor",
            "team/shard",
            RepositoryPermission::Maintain,
        );
        let root =
            std::env::temp_dir().join(format!("xc-shard-index-repair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let outcome = execute_shard_index_repair(
            &remote,
            &session,
            &root,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &plan,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            ShardIndexRepairOutcome::CommittedAndVerified { .. }
        ));
        assert_eq!(remote.read_ref("memory", "main").unwrap(), "repair-commit");
        let _ = std::fs::remove_dir_all(root);
    }
}
