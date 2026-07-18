//! Auditable remote governance-record planning and mutation.

use crate::protocol::canonical_json_bytes;
use crate::publication_staging::stage_publication_bytes;
use crate::{
    AuthenticatedGitHubSession, CacheError, CompareAndSwapResult, ContentDigest,
    RemoteCommitRequest, RemoteGitStore, RemoteShardReader, RevocationIndexPartition,
    RevocationRecord, TransportPart,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevocationUpdatePlan {
    pub schema_version: u32,
    pub authorized_repository: String,
    pub repository: String,
    pub branch: String,
    pub expected_head: String,
    pub repository_path: String,
    pub observed_digest: Option<ContentDigest>,
    pub replacement_digest: ContentDigest,
    pub replacement_size_bytes: u64,
    pub partition: RevocationIndexPartition,
    pub changes_remote_state: bool,
}

impl RevocationUpdatePlan {
    pub fn validate(&self) -> Result<(), CacheError> {
        self.partition.validate()?;
        let bytes = canonical_json_bytes(&self.partition)?;
        let prefix = &self.partition.identity_prefix;
        if self.schema_version != 1
            || self.authorized_repository.trim().is_empty()
            || self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.expected_head.trim().is_empty()
            || self.repository_path != format!("revocations/indexes/{prefix}.json")
            || self.replacement_digest != ContentDigest::sha256(&bytes)
            || self.replacement_size_bytes != bytes.len() as u64
            || self.replacement_size_bytes == 0
            || self.changes_remote_state
                != (self.observed_digest.as_ref() != Some(&self.replacement_digest))
        {
            return Err(CacheError::InvalidManifest(
                "revocation update plan identity or replacement is invalid".to_owned(),
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
pub enum RevocationUpdateOutcome {
    NoChanges {
        plan_digest: ContentDigest,
    },
    RefConflict {
        current_head: String,
    },
    CommittedAndVerified {
        commit_id: String,
        plan_digest: ContentDigest,
        principal: String,
        repository_permission_evidence_digest: ContentDigest,
        repository_path: String,
        revocation_digest: ContentDigest,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupersessionRecord {
    pub schema_version: u32,
    pub prior_semantic_digest: ContentDigest,
    pub prior_manifest_digest: ContentDigest,
    pub prior_payload_digest: ContentDigest,
    pub replacement_semantic_digest: ContentDigest,
    pub replacement_manifest_digest: ContentDigest,
    pub replacement_payload_digest: ContentDigest,
    pub reason: String,
    pub effective_unix_seconds: u64,
    pub authorizing_evidence_digest: ContentDigest,
}

impl SupersessionRecord {
    pub fn validate(&self) -> Result<(), CacheError> {
        let digests = [
            &self.prior_semantic_digest,
            &self.prior_manifest_digest,
            &self.prior_payload_digest,
            &self.replacement_semantic_digest,
            &self.replacement_manifest_digest,
            &self.replacement_payload_digest,
            &self.authorizing_evidence_digest,
        ];
        if self.schema_version != 1
            || digests.iter().any(|digest| !digest.validate())
            || self.reason.trim().is_empty()
            || self.reason.len() > 4_096
            || (self.prior_semantic_digest == self.replacement_semantic_digest
                && self.prior_manifest_digest == self.replacement_manifest_digest
                && self.prior_payload_digest == self.replacement_payload_digest)
        {
            return Err(CacheError::InvalidManifest(
                "supersession identities, reason, or authority evidence are invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupersessionIndexPartition {
    pub schema_version: u32,
    pub semantic_prefix: String,
    pub records: Vec<SupersessionRecord>,
}

impl SupersessionIndexPartition {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.semantic_prefix.len() != 2
            || !self
                .semantic_prefix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CacheError::InvalidManifest(
                "supersession partition prefix is invalid".to_owned(),
            ));
        }
        let mut previous: Option<(&ContentDigest, &ContentDigest)> = None;
        for record in &self.records {
            record.validate()?;
            if !record
                .prior_semantic_digest
                .0
                .starts_with(&self.semantic_prefix)
            {
                return Err(CacheError::InvalidManifest(
                    "supersession record is stored in the wrong partition".to_owned(),
                ));
            }
            let current = (&record.prior_semantic_digest, &record.prior_manifest_digest);
            if previous.is_some_and(|previous| previous >= current) {
                return Err(CacheError::InvalidManifest(
                    "supersession partition is duplicated or not canonically ordered".to_owned(),
                ));
            }
            previous = Some(current);
        }
        Ok(())
    }

    pub fn replacement_for(
        &self,
        semantic_digest: &ContentDigest,
        manifest_digest: &ContentDigest,
    ) -> Option<&SupersessionRecord> {
        self.records.iter().find(|record| {
            &record.prior_semantic_digest == semantic_digest
                && &record.prior_manifest_digest == manifest_digest
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupersessionUpdatePlan {
    pub schema_version: u32,
    pub authorized_repository: String,
    pub repository: String,
    pub branch: String,
    pub expected_head: String,
    pub repository_path: String,
    pub observed_digest: Option<ContentDigest>,
    pub replacement_digest: ContentDigest,
    pub replacement_size_bytes: u64,
    pub partition: SupersessionIndexPartition,
    pub changes_remote_state: bool,
}

impl SupersessionUpdatePlan {
    pub fn validate(&self) -> Result<(), CacheError> {
        self.partition.validate()?;
        let bytes = canonical_json_bytes(&self.partition)?;
        if self.schema_version != 1
            || self.authorized_repository.trim().is_empty()
            || self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.expected_head.trim().is_empty()
            || self.repository_path
                != format!(
                    "supersessions/indexes/{}.json",
                    self.partition.semantic_prefix
                )
            || self.replacement_digest != ContentDigest::sha256(&bytes)
            || self.replacement_size_bytes != bytes.len() as u64
            || self.replacement_size_bytes == 0
            || self.changes_remote_state
                != (self.observed_digest.as_ref() != Some(&self.replacement_digest))
        {
            return Err(CacheError::InvalidManifest(
                "supersession update plan identity or replacement is invalid".to_owned(),
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
pub enum SupersessionUpdateOutcome {
    NoChanges {
        plan_digest: ContentDigest,
    },
    RefConflict {
        current_head: String,
    },
    CommittedAndVerified {
        commit_id: String,
        plan_digest: ContentDigest,
        principal: String,
        repository_permission_evidence_digest: ContentDigest,
        repository_path: String,
        supersession_digest: ContentDigest,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn load_supersession_chain(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    initial_semantic_digest: &ContentDigest,
    initial_manifest_digest: &ContentDigest,
    maximum_partition_bytes: u64,
    maximum_depth: u32,
    cancellation: &CancellationToken,
) -> Result<Vec<SupersessionRecord>, CacheError> {
    if repository.trim().is_empty()
        || revision.trim().is_empty()
        || !initial_semantic_digest.validate()
        || !initial_manifest_digest.validate()
        || maximum_partition_bytes == 0
        || maximum_depth == 0
    {
        return Err(CacheError::InvalidManifest(
            "supersession lookup requires exact identities and positive read bounds".to_owned(),
        ));
    }
    let reader = RemoteShardReader::new(remote, maximum_partition_bytes)?;
    let mut semantic_digest = initial_semantic_digest.clone();
    let mut manifest_digest = initial_manifest_digest.clone();
    let mut visited = BTreeSet::new();
    let mut chain = Vec::new();
    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        if !visited.insert((semantic_digest.clone(), manifest_digest.clone())) {
            return Err(CacheError::InvalidTransition(
                "supersession chain contains a cycle".to_owned(),
            ));
        }
        let prefix = semantic_digest.0[..2].to_owned();
        let path = format!("supersessions/indexes/{prefix}.json");
        let document = match reader.read_json::<SupersessionIndexPartition>(
            repository,
            revision,
            &path,
            cancellation,
        ) {
            Ok(document) => document,
            Err(CacheError::NotFound(_)) => break,
            Err(error) => return Err(error),
        };
        document.value.validate()?;
        if document.value.semantic_prefix != prefix {
            return Err(CacheError::InvalidManifest(
                "supersession partition was returned from the wrong path".to_owned(),
            ));
        }
        let Some(record) = document
            .value
            .replacement_for(&semantic_digest, &manifest_digest)
            .cloned()
        else {
            break;
        };
        if chain.len() as u32 == maximum_depth {
            return Err(CacheError::ResourceLimit(
                "supersession chain exceeds its configured depth bound".to_owned(),
            ));
        }
        semantic_digest = record.replacement_semantic_digest.clone();
        manifest_digest = record.replacement_manifest_digest.clone();
        chain.push(record);
    }
    Ok(chain)
}

fn merge_revocation_record(
    mut partition: RevocationIndexPartition,
    record: RevocationRecord,
) -> Result<RevocationIndexPartition, CacheError> {
    partition.validate()?;
    record.validate()?;
    if !record
        .identity_digest
        .0
        .starts_with(&partition.identity_prefix)
    {
        return Err(CacheError::InvalidManifest(
            "revocation record does not belong to the selected partition".to_owned(),
        ));
    }
    if let Some(existing) = partition.records.iter().find(|existing| {
        existing.scope == record.scope && existing.identity_digest == record.identity_digest
    }) {
        if existing == &record {
            return Ok(partition);
        }
        return Err(CacheError::InvalidTransition(
            "an existing revocation identity cannot be silently rewritten".to_owned(),
        ));
    }
    partition.records.push(record);
    partition.records.sort_by(|left, right| {
        (left.scope, &left.identity_digest).cmp(&(right.scope, &right.identity_digest))
    });
    partition.validate()?;
    Ok(partition)
}

#[allow(clippy::too_many_arguments)]
pub fn plan_revocation_update(
    remote: &dyn RemoteGitStore,
    authorized_repository: &str,
    repository: &str,
    branch: &str,
    maximum_partition_bytes: u64,
    record: RevocationRecord,
    cancellation: &CancellationToken,
) -> Result<RevocationUpdatePlan, CacheError> {
    record.validate()?;
    if authorized_repository.trim().is_empty()
        || repository.trim().is_empty()
        || branch.trim().is_empty()
        || maximum_partition_bytes == 0
    {
        return Err(CacheError::InvalidManifest(
            "revocation planning requires repository identity, branch, and a read bound".to_owned(),
        ));
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let expected_head = remote.read_ref(repository, branch)?;
    let prefix = record.identity_digest.0[..2].to_owned();
    let repository_path = format!("revocations/indexes/{prefix}.json");
    let observed_digest =
        remote.immutable_path_digest(repository, &expected_head, &repository_path)?;
    let partition = if let Some(expected_digest) = observed_digest.as_ref() {
        let document = RemoteShardReader::new(remote, maximum_partition_bytes)?
            .read_json::<RevocationIndexPartition>(
            repository,
            &expected_head,
            &repository_path,
            cancellation,
        )?;
        if &document.source.content_digest != expected_digest {
            return Err(CacheError::DigestMismatch {
                expected: expected_digest.to_string(),
                actual: document.source.content_digest.to_string(),
            });
        }
        document.value
    } else {
        RevocationIndexPartition {
            schema_version: 1,
            identity_prefix: prefix,
            records: Vec::new(),
        }
    };
    let partition = merge_revocation_record(partition, record)?;
    let bytes = canonical_json_bytes(&partition)?;
    if bytes.len() as u64 > maximum_partition_bytes {
        return Err(CacheError::ResourceLimit(
            "updated revocation partition exceeds its configured byte bound".to_owned(),
        ));
    }
    let replacement_digest = ContentDigest::sha256(&bytes);
    let plan = RevocationUpdatePlan {
        schema_version: 1,
        authorized_repository: authorized_repository.to_owned(),
        repository: repository.to_owned(),
        branch: branch.to_owned(),
        expected_head,
        repository_path,
        changes_remote_state: observed_digest.as_ref() != Some(&replacement_digest),
        observed_digest,
        replacement_digest,
        replacement_size_bytes: bytes.len() as u64,
        partition,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn execute_revocation_update(
    remote: &dyn RemoteGitStore,
    session: &AuthenticatedGitHubSession,
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    plan: &RevocationUpdatePlan,
) -> Result<RevocationUpdateOutcome, CacheError> {
    plan.validate()?;
    session.require_write_for(
        session.evidence().principal.as_str(),
        &plan.authorized_repository,
    )?;
    let plan_digest = plan.digest()?;
    if !plan.changes_remote_state {
        return Ok(RevocationUpdateOutcome::NoChanges { plan_digest });
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let current_head = remote.read_ref(&plan.repository, &plan.branch)?;
    if current_head != plan.expected_head {
        return Ok(RevocationUpdateOutcome::RefConflict { current_head });
    }
    let current_digest = remote.immutable_path_digest(
        &plan.repository,
        &plan.expected_head,
        &plan.repository_path,
    )?;
    if current_digest != plan.observed_digest {
        return Err(CacheError::DigestMismatch {
            expected: format!("{:?}", plan.observed_digest),
            actual: format!("{current_digest:?}"),
        });
    }
    let bytes = canonical_json_bytes(&plan.partition)?;
    let mut staged_bytes = 0;
    let part: TransportPart = stage_publication_bytes(
        staging_root,
        &plan.repository_path,
        &bytes,
        resources,
        cancellation,
        &mut staged_bytes,
    )?;
    let request = RemoteCommitRequest {
        repository: plan.repository.clone(),
        branch: plan.branch.clone(),
        expected_head: plan.expected_head.clone(),
        message: format!("publish revocation record {plan_digest}"),
        parts: vec![part.clone()],
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            Ok(RevocationUpdateOutcome::RefConflict { current_head })
        }
        CompareAndSwapResult::Committed { commit_id } => {
            remote.verify_committed_part(&plan.repository, &commit_id, &part)?;
            Ok(RevocationUpdateOutcome::CommittedAndVerified {
                commit_id,
                plan_digest,
                principal: session.evidence().principal.clone(),
                repository_permission_evidence_digest: session.evidence().evidence_digest.clone(),
                repository_path: plan.repository_path.clone(),
                revocation_digest: plan.replacement_digest.clone(),
            })
        }
    }
}

fn merge_supersession_record(
    mut partition: SupersessionIndexPartition,
    record: SupersessionRecord,
) -> Result<SupersessionIndexPartition, CacheError> {
    partition.validate()?;
    record.validate()?;
    if !record
        .prior_semantic_digest
        .0
        .starts_with(&partition.semantic_prefix)
    {
        return Err(CacheError::InvalidManifest(
            "supersession record does not belong to the selected partition".to_owned(),
        ));
    }
    if let Some(existing) = partition.records.iter().find(|existing| {
        existing.prior_semantic_digest == record.prior_semantic_digest
            && existing.prior_manifest_digest == record.prior_manifest_digest
    }) {
        if existing == &record {
            return Ok(partition);
        }
        return Err(CacheError::InvalidTransition(
            "an existing supersession edge cannot be silently rewritten".to_owned(),
        ));
    }
    partition.records.push(record);
    partition.records.sort_by(|left, right| {
        (&left.prior_semantic_digest, &left.prior_manifest_digest)
            .cmp(&(&right.prior_semantic_digest, &right.prior_manifest_digest))
    });
    partition.validate()?;
    Ok(partition)
}

fn require_manifest_at_revision(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    semantic_digest: &ContentDigest,
    manifest_digest: &ContentDigest,
) -> Result<(), CacheError> {
    let path = format!(
        "manifests/{}/{}.json",
        &semantic_digest.0[..2],
        manifest_digest.0
    );
    match remote.immutable_path_digest(repository, revision, &path)? {
        Some(actual) if &actual == manifest_digest => Ok(()),
        Some(actual) => Err(CacheError::DigestMismatch {
            expected: manifest_digest.to_string(),
            actual: actual.to_string(),
        }),
        None => Err(CacheError::NotFound(path)),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plan_supersession_update(
    remote: &dyn RemoteGitStore,
    authorized_repository: &str,
    repository: &str,
    branch: &str,
    maximum_partition_bytes: u64,
    record: SupersessionRecord,
    cancellation: &CancellationToken,
) -> Result<SupersessionUpdatePlan, CacheError> {
    record.validate()?;
    if authorized_repository.trim().is_empty()
        || repository.trim().is_empty()
        || branch.trim().is_empty()
        || maximum_partition_bytes == 0
    {
        return Err(CacheError::InvalidManifest(
            "supersession planning requires repository identity, branch, and a read bound"
                .to_owned(),
        ));
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let expected_head = remote.read_ref(repository, branch)?;
    require_manifest_at_revision(
        remote,
        repository,
        &expected_head,
        &record.prior_semantic_digest,
        &record.prior_manifest_digest,
    )?;
    require_manifest_at_revision(
        remote,
        repository,
        &expected_head,
        &record.replacement_semantic_digest,
        &record.replacement_manifest_digest,
    )?;
    let prefix = record.prior_semantic_digest.0[..2].to_owned();
    let repository_path = format!("supersessions/indexes/{prefix}.json");
    let observed_digest =
        remote.immutable_path_digest(repository, &expected_head, &repository_path)?;
    let partition = if let Some(expected_digest) = observed_digest.as_ref() {
        let document = RemoteShardReader::new(remote, maximum_partition_bytes)?
            .read_json::<SupersessionIndexPartition>(
            repository,
            &expected_head,
            &repository_path,
            cancellation,
        )?;
        if &document.source.content_digest != expected_digest {
            return Err(CacheError::DigestMismatch {
                expected: expected_digest.to_string(),
                actual: document.source.content_digest.to_string(),
            });
        }
        document.value
    } else {
        SupersessionIndexPartition {
            schema_version: 1,
            semantic_prefix: prefix,
            records: Vec::new(),
        }
    };
    let partition = merge_supersession_record(partition, record)?;
    let bytes = canonical_json_bytes(&partition)?;
    if bytes.len() as u64 > maximum_partition_bytes {
        return Err(CacheError::ResourceLimit(
            "updated supersession partition exceeds its configured byte bound".to_owned(),
        ));
    }
    let replacement_digest = ContentDigest::sha256(&bytes);
    let plan = SupersessionUpdatePlan {
        schema_version: 1,
        authorized_repository: authorized_repository.to_owned(),
        repository: repository.to_owned(),
        branch: branch.to_owned(),
        expected_head,
        repository_path,
        changes_remote_state: observed_digest.as_ref() != Some(&replacement_digest),
        observed_digest,
        replacement_digest,
        replacement_size_bytes: bytes.len() as u64,
        partition,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn execute_supersession_update(
    remote: &dyn RemoteGitStore,
    session: &AuthenticatedGitHubSession,
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    plan: &SupersessionUpdatePlan,
) -> Result<SupersessionUpdateOutcome, CacheError> {
    plan.validate()?;
    session.require_write_for(
        session.evidence().principal.as_str(),
        &plan.authorized_repository,
    )?;
    let plan_digest = plan.digest()?;
    if !plan.changes_remote_state {
        return Ok(SupersessionUpdateOutcome::NoChanges { plan_digest });
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let current_head = remote.read_ref(&plan.repository, &plan.branch)?;
    if current_head != plan.expected_head {
        return Ok(SupersessionUpdateOutcome::RefConflict { current_head });
    }
    let current_digest = remote.immutable_path_digest(
        &plan.repository,
        &plan.expected_head,
        &plan.repository_path,
    )?;
    if current_digest != plan.observed_digest {
        return Err(CacheError::DigestMismatch {
            expected: format!("{:?}", plan.observed_digest),
            actual: format!("{current_digest:?}"),
        });
    }
    let bytes = canonical_json_bytes(&plan.partition)?;
    let mut staged_bytes = 0;
    let part = stage_publication_bytes(
        staging_root,
        &plan.repository_path,
        &bytes,
        resources,
        cancellation,
        &mut staged_bytes,
    )?;
    let request = RemoteCommitRequest {
        repository: plan.repository.clone(),
        branch: plan.branch.clone(),
        expected_head: plan.expected_head.clone(),
        message: format!("publish supersession record {plan_digest}"),
        parts: vec![part.clone()],
        delete_paths: Vec::new(),
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::RefConflict { current_head } => {
            Ok(SupersessionUpdateOutcome::RefConflict { current_head })
        }
        CompareAndSwapResult::Committed { commit_id } => {
            remote.verify_committed_part(&plan.repository, &commit_id, &part)?;
            Ok(SupersessionUpdateOutcome::CommittedAndVerified {
                commit_id,
                plan_digest,
                principal: session.evidence().principal.clone(),
                repository_permission_evidence_digest: session.evidence().evidence_digest.clone(),
                repository_path: plan.repository_path.clone(),
                supersession_digest: plan.replacement_digest.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RemoteReadReport, RepositoryPermission, RevocationScope};
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemoryRemote {
        head: Mutex<String>,
        paths: Mutex<BTreeMap<String, ContentDigest>>,
        documents: Mutex<BTreeMap<String, Vec<u8>>>,
    }

    impl RemoteGitStore for MemoryRemote {
        fn read_ref(&self, _repository: &str, _branch: &str) -> Result<String, CacheError> {
            Ok(self.head.lock().unwrap().clone())
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(self.paths.lock().unwrap().get(path).cloned())
        }

        fn read_committed_path(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
            _maximum_bytes: u64,
            _cancellation: &CancellationToken,
            writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            let documents = self.documents.lock().unwrap();
            let bytes = documents
                .get(path)
                .ok_or_else(|| CacheError::NotFound(format!("{revision}:{path}")))?;
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
            let mut head = self.head.lock().unwrap();
            if *head != request.expected_head {
                return Ok(CompareAndSwapResult::RefConflict {
                    current_head: head.clone(),
                });
            }
            for part in &request.parts {
                self.paths
                    .lock()
                    .unwrap()
                    .insert(part.repository_path.clone(), part.content_digest.clone());
            }
            *head = "commit-1".to_owned();
            Ok(CompareAndSwapResult::Committed {
                commit_id: head.clone(),
            })
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            revision: &str,
            part: &TransportPart,
        ) -> Result<(), CacheError> {
            if revision != "commit-1"
                || self.paths.lock().unwrap().get(&part.repository_path)
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

    fn record(reason: &str) -> RevocationRecord {
        RevocationRecord {
            schema_version: 1,
            scope: RevocationScope::Semantic,
            identity_digest: ContentDigest::sha256(b"identity"),
            reason: reason.to_owned(),
            effective_unix_seconds: 10,
            replacement_digest: Some(ContentDigest::sha256(b"replacement")),
            incident_reference: Some("incident-7".to_owned()),
            authorizing_evidence_digest: ContentDigest::sha256(b"authority"),
        }
    }

    fn supersession_record(reason: &str) -> SupersessionRecord {
        SupersessionRecord {
            schema_version: 1,
            prior_semantic_digest: ContentDigest::sha256(b"prior-semantic"),
            prior_manifest_digest: ContentDigest::sha256(b"prior-manifest"),
            prior_payload_digest: ContentDigest::sha256(b"prior-payload"),
            replacement_semantic_digest: ContentDigest::sha256(b"replacement-semantic"),
            replacement_manifest_digest: ContentDigest::sha256(b"replacement-manifest"),
            replacement_payload_digest: ContentDigest::sha256(b"replacement-payload"),
            reason: reason.to_owned(),
            effective_unix_seconds: 20,
            authorizing_evidence_digest: ContentDigest::sha256(b"supersession-authority"),
        }
    }

    fn manifest_path(semantic_digest: &ContentDigest, manifest_digest: &ContentDigest) -> String {
        format!(
            "manifests/{}/{}.json",
            &semantic_digest.0[..2],
            manifest_digest.0
        )
    }

    #[test]
    fn revocation_merge_is_idempotent_and_rejects_rewrite() {
        let prefix = record("reason").identity_digest.0[..2].to_owned();
        let partition = RevocationIndexPartition {
            schema_version: 1,
            identity_prefix: prefix,
            records: Vec::new(),
        };
        let merged = merge_revocation_record(partition, record("reason")).unwrap();
        assert_eq!(merged.records.len(), 1);
        assert_eq!(
            merge_revocation_record(merged.clone(), record("reason")).unwrap(),
            merged
        );
        assert!(merge_revocation_record(merged, record("different reason")).is_err());
    }

    #[test]
    fn revocation_update_is_planned_committed_and_remotely_verified() {
        let remote = MemoryRemote {
            head: Mutex::new("head-0".to_owned()),
            ..MemoryRemote::default()
        };
        let cancellation = CancellationToken::new();
        let plan = plan_revocation_update(
            &remote,
            "owner/repo",
            "https://example.invalid/owner/repo.git",
            "main",
            1_000_000,
            record("incident"),
            &cancellation,
        )
        .unwrap();
        assert!(plan.changes_remote_state);
        let staging =
            std::env::temp_dir().join(format!("xc-revocation-{}", plan.replacement_digest.0));
        let session = AuthenticatedGitHubSession::verified_for_test(
            "owner",
            "owner/repo",
            RepositoryPermission::Admin,
        );
        let outcome = execute_revocation_update(
            &remote,
            &session,
            &staging,
            &ResourcePolicy::default(),
            &cancellation,
            &plan,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            RevocationUpdateOutcome::CommittedAndVerified { .. }
        ));
        let _ = std::fs::remove_dir_all(staging);
    }

    #[test]
    fn supersession_merge_is_idempotent_and_rejects_rewrite() {
        let record = supersession_record("correction");
        let partition = SupersessionIndexPartition {
            schema_version: 1,
            semantic_prefix: record.prior_semantic_digest.0[..2].to_owned(),
            records: Vec::new(),
        };
        let merged = merge_supersession_record(partition, record.clone()).unwrap();
        assert_eq!(merged.records.len(), 1);
        assert_eq!(
            merge_supersession_record(merged.clone(), record.clone()).unwrap(),
            merged
        );
        let mut rewrite = record;
        rewrite.reason = "silently rewritten".to_owned();
        assert!(merge_supersession_record(merged, rewrite).is_err());
    }

    #[test]
    fn supersession_update_requires_both_manifests_and_is_remotely_verified() {
        let record = supersession_record("validated correction");
        let mut paths = BTreeMap::new();
        paths.insert(
            manifest_path(&record.prior_semantic_digest, &record.prior_manifest_digest),
            record.prior_manifest_digest.clone(),
        );
        paths.insert(
            manifest_path(
                &record.replacement_semantic_digest,
                &record.replacement_manifest_digest,
            ),
            record.replacement_manifest_digest.clone(),
        );
        let remote = MemoryRemote {
            head: Mutex::new("head-0".to_owned()),
            paths: Mutex::new(paths),
            ..MemoryRemote::default()
        };
        let cancellation = CancellationToken::new();
        let plan = plan_supersession_update(
            &remote,
            "owner/repo",
            "https://example.invalid/owner/repo.git",
            "main",
            1_000_000,
            record,
            &cancellation,
        )
        .unwrap();
        assert!(plan.changes_remote_state);
        let staging =
            std::env::temp_dir().join(format!("xc-supersession-{}", plan.replacement_digest.0));
        let session = AuthenticatedGitHubSession::verified_for_test(
            "owner",
            "owner/repo",
            RepositoryPermission::Admin,
        );
        let outcome = execute_supersession_update(
            &remote,
            &session,
            &staging,
            &ResourcePolicy::default(),
            &cancellation,
            &plan,
        )
        .unwrap();
        assert!(matches!(
            outcome,
            SupersessionUpdateOutcome::CommittedAndVerified { .. }
        ));
        let _ = std::fs::remove_dir_all(staging);
    }

    #[test]
    fn supersession_planning_rejects_an_unpublished_replacement_manifest() {
        let record = supersession_record("invalid correction");
        let mut paths = BTreeMap::new();
        paths.insert(
            manifest_path(&record.prior_semantic_digest, &record.prior_manifest_digest),
            record.prior_manifest_digest.clone(),
        );
        let remote = MemoryRemote {
            head: Mutex::new("head-0".to_owned()),
            paths: Mutex::new(paths),
            ..MemoryRemote::default()
        };
        assert!(matches!(
            plan_supersession_update(
                &remote,
                "owner/repo",
                "https://example.invalid/owner/repo.git",
                "main",
                1_000_000,
                record,
                &CancellationToken::new(),
            ),
            Err(CacheError::NotFound(_))
        ));
    }

    #[test]
    fn consumer_follows_a_bounded_supersession_chain_without_losing_old_identity() {
        let first = supersession_record("first correction");
        let second = SupersessionRecord {
            prior_semantic_digest: first.replacement_semantic_digest.clone(),
            prior_manifest_digest: first.replacement_manifest_digest.clone(),
            prior_payload_digest: first.replacement_payload_digest.clone(),
            replacement_semantic_digest: ContentDigest::sha256(b"final-semantic"),
            replacement_manifest_digest: ContentDigest::sha256(b"final-manifest"),
            replacement_payload_digest: ContentDigest::sha256(b"final-payload"),
            reason: "second correction".to_owned(),
            ..first.clone()
        };
        let first_partition = SupersessionIndexPartition {
            schema_version: 1,
            semantic_prefix: first.prior_semantic_digest.0[..2].to_owned(),
            records: vec![first.clone()],
        };
        let second_partition = SupersessionIndexPartition {
            schema_version: 1,
            semantic_prefix: second.prior_semantic_digest.0[..2].to_owned(),
            records: vec![second.clone()],
        };
        let mut documents = BTreeMap::new();
        documents.insert(
            format!(
                "supersessions/indexes/{}.json",
                first_partition.semantic_prefix
            ),
            canonical_json_bytes(&first_partition).unwrap(),
        );
        documents.insert(
            format!(
                "supersessions/indexes/{}.json",
                second_partition.semantic_prefix
            ),
            canonical_json_bytes(&second_partition).unwrap(),
        );
        let remote = MemoryRemote {
            head: Mutex::new("head-0".to_owned()),
            documents: Mutex::new(documents),
            ..MemoryRemote::default()
        };
        let chain = load_supersession_chain(
            &remote,
            "owner/repo",
            "head-0",
            &first.prior_semantic_digest,
            &first.prior_manifest_digest,
            1_000_000,
            2,
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(chain, vec![first.clone(), second]);
        assert_eq!(chain[0].prior_payload_digest, first.prior_payload_digest);
    }
}
