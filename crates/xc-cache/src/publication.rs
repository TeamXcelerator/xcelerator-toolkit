//! Streaming transport parts and resumable, per-target publication journals.

use crate::protocol::{
    canonical_digest, normalized_relative_path, PublicationDestination, PublicationTargetState,
};
use crate::{CacheError, ContentDigest, PublicationReviewEvidence, RepositoryPermissionEvidence};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use xc_core::{CancellationToken, PublicationAuthorityMode, PublicationTarget, ResourcePolicy};

/// Ordinary Git-managed files must remain strictly below this boundary.
pub const GITHUB_HARD_FILE_BOUNDARY_BYTES: u64 = 100_000_000;
/// Project default deterministic byte-split part size (90 MiB).
pub const DEFAULT_SPLIT_PART_BYTES: u64 = 94_371_840;
/// Maximum new payload introduced by one commit and push.
pub const GITHUB_MAX_PUBLICATION_BATCH_BYTES: u64 = 1_000_000_000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportPolicy {
    pub maximum_file_bytes_exclusive: u64,
    pub split_part_bytes: u64,
    pub maximum_batch_payload_bytes: u64,
    pub maximum_pending_batches: usize,
}

impl Default for TransportPolicy {
    fn default() -> Self {
        Self {
            maximum_file_bytes_exclusive: GITHUB_HARD_FILE_BOUNDARY_BYTES,
            split_part_bytes: DEFAULT_SPLIT_PART_BYTES,
            maximum_batch_payload_bytes: GITHUB_MAX_PUBLICATION_BATCH_BYTES,
            maximum_pending_batches: 1,
        }
    }
}

impl TransportPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.maximum_file_bytes_exclusive == 0
            || self.maximum_file_bytes_exclusive > GITHUB_HARD_FILE_BOUNDARY_BYTES
        {
            return Err(CacheError::InvalidManifest(format!(
                "file boundary must be at most {GITHUB_HARD_FILE_BOUNDARY_BYTES} bytes"
            )));
        }
        if self.split_part_bytes == 0
            || self.split_part_bytes >= self.maximum_file_bytes_exclusive
            || self.split_part_bytes > DEFAULT_SPLIT_PART_BYTES
        {
            return Err(CacheError::InvalidManifest(format!(
                "split part size must be positive, below the file boundary, and no larger than {DEFAULT_SPLIT_PART_BYTES}"
            )));
        }
        if self.maximum_batch_payload_bytes == 0
            || self.maximum_batch_payload_bytes > GITHUB_MAX_PUBLICATION_BATCH_BYTES
            || self.maximum_batch_payload_bytes < self.split_part_bytes
        {
            return Err(CacheError::InvalidManifest(format!(
                "batch payload must contain one part and may not exceed {GITHUB_MAX_PUBLICATION_BATCH_BYTES} bytes"
            )));
        }
        if self.maximum_pending_batches == 0 {
            return Err(CacheError::InvalidManifest(
                "at least one pending publication batch must be allowed".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportPart {
    pub sequence: u64,
    pub repository_path: String,
    pub size_bytes: u64,
    pub content_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportEncodingRecord {
    pub schema_version: u32,
    pub canonical_payload_digest: ContentDigest,
    pub encoder_profile: String,
    pub package_size_bytes: u64,
    pub package_digest: ContentDigest,
    pub ordered_parts: Vec<TransportPart>,
    pub reconstruction: String,
}

impl TransportEncodingRecord {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.encoder_profile.trim().is_empty()
            || self.reconstruction.trim().is_empty()
            || !self.canonical_payload_digest.validate()
            || !self.package_digest.validate()
            || self.ordered_parts.is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "transport encoding record is incomplete".to_owned(),
            ));
        }
        let mut total = 0u64;
        for (index, part) in self.ordered_parts.iter().enumerate() {
            if part.sequence != index as u64
                || part.repository_path.trim().is_empty()
                || !normalized_relative_path(&part.repository_path)
                || part.size_bytes == 0
                || part.size_bytes >= GITHUB_HARD_FILE_BOUNDARY_BYTES
                || !part.content_digest.validate()
            {
                return Err(CacheError::InvalidManifest(format!(
                    "transport part {index} is invalid"
                )));
            }
            total = total.checked_add(part.size_bytes).ok_or_else(|| {
                CacheError::InvalidManifest("transport size overflows u64".to_owned())
            })?;
        }
        if total != self.package_size_bytes {
            return Err(CacheError::InvalidManifest(format!(
                "transport parts total {total}, expected {}",
                self.package_size_bytes
            )));
        }
        Ok(())
    }
}

/// Split an already-canonical encoded stream into bounded, hashed parts. The
/// sink may persist each part atomically or enqueue it for direct publication.
pub fn stream_split_encoded<R, F>(
    reader: &mut R,
    canonical_payload_digest: ContentDigest,
    encoder_profile: impl Into<String>,
    transport_policy: &TransportPolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    mut sink: F,
) -> Result<TransportEncodingRecord, CacheError>
where
    R: Read,
    F: FnMut(&TransportPart, &[u8]) -> Result<(), CacheError>,
{
    transport_policy.validate()?;
    if !canonical_payload_digest.validate() {
        return Err(CacheError::InvalidManifest(
            "canonical payload digest is invalid".to_owned(),
        ));
    }
    if resources
        .maximum_memory_bytes
        .is_some_and(|maximum| transport_policy.split_part_bytes > maximum)
    {
        return Err(CacheError::ResourceLimit(format!(
            "one split part requires {} bytes, above the memory budget",
            transport_policy.split_part_bytes
        )));
    }
    let part_capacity = usize::try_from(transport_policy.split_part_bytes).map_err(|_| {
        CacheError::ResourceLimit("split part size does not fit this platform".to_owned())
    })?;
    let mut package_hasher = Sha256::new();
    let mut package_size = 0u64;
    let mut ordered_parts = Vec::new();

    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let mut bytes = Vec::with_capacity(part_capacity);
        let mut limited = reader.take(transport_policy.split_part_bytes);
        limited.read_to_end(&mut bytes)?;
        if bytes.is_empty() {
            break;
        }
        package_size = package_size
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| CacheError::ResourceLimit("package size exceeds u64".to_owned()))?;
        if resources
            .maximum_transfer_bytes
            .is_some_and(|maximum| package_size > maximum)
        {
            return Err(CacheError::ResourceLimit(format!(
                "encoded transfer {package_size} bytes exceeds policy"
            )));
        }
        package_hasher.update(&bytes);
        let content_digest = ContentDigest::sha256(&bytes);
        let prefix = &content_digest.0[..2];
        let part = TransportPart {
            sequence: ordered_parts.len() as u64,
            repository_path: format!("objects/sha256/{prefix}/{}.part", content_digest.0),
            size_bytes: bytes.len() as u64,
            content_digest,
        };
        sink(&part, &bytes)?;
        ordered_parts.push(part);
    }

    if ordered_parts.is_empty() {
        return Err(CacheError::InvalidManifest(
            "encoded package must not be empty".to_owned(),
        ));
    }
    let package_digest = ContentDigest(format!("{:x}", package_hasher.finalize()));
    let record = TransportEncodingRecord {
        schema_version: 1,
        canonical_payload_digest,
        encoder_profile: encoder_profile.into(),
        package_size_bytes: package_size,
        package_digest,
        ordered_parts,
        reconstruction: "concatenate ordered parts, then decode the named deterministic profile"
            .to_owned(),
    };
    record.validate()?;
    Ok(record)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationBatchPlan {
    pub sequence: u64,
    pub payload_bytes: u64,
    pub parts: Vec<TransportPart>,
}

pub fn plan_publication_batches(
    parts: &[TransportPart],
    policy: &TransportPolicy,
) -> Result<Vec<PublicationBatchPlan>, CacheError> {
    policy.validate()?;
    if parts.is_empty() {
        return Err(CacheError::InvalidManifest(
            "publication requires at least one transport part".to_owned(),
        ));
    }
    let mut batches = Vec::new();
    let mut current_parts = Vec::new();
    let mut current_bytes = 0u64;
    for (index, part) in parts.iter().enumerate() {
        if part.sequence != index as u64
            || part.size_bytes == 0
            || part.size_bytes >= policy.maximum_file_bytes_exclusive
            || part.size_bytes > policy.maximum_batch_payload_bytes
        {
            return Err(CacheError::InvalidManifest(format!(
                "part {index} violates publication limits"
            )));
        }
        if !current_parts.is_empty()
            && current_bytes.saturating_add(part.size_bytes) > policy.maximum_batch_payload_bytes
        {
            batches.push(PublicationBatchPlan {
                sequence: batches.len() as u64,
                payload_bytes: current_bytes,
                parts: std::mem::take(&mut current_parts),
            });
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(part.size_bytes);
        current_parts.push(part.clone());
    }
    if !current_parts.is_empty() {
        batches.push(PublicationBatchPlan {
            sequence: batches.len() as u64,
            payload_bytes: current_bytes,
            parts: current_parts,
        });
    }
    Ok(batches)
}

/// Plan repository commits across every artifact in one family publication.
///
/// Logical artifacts retain independent manifests and semantic identities, but
/// their physical files share the same one-gigabyte Git commit budget. A path
/// reused by multiple artifacts is transferred once. Conflicting bytes at one
/// repository path fail closed before any remote mutation.
pub fn plan_family_publication_batches(
    artifact_parts: &[Vec<TransportPart>],
    policy: &TransportPolicy,
) -> Result<Vec<PublicationBatchPlan>, CacheError> {
    policy.validate()?;
    let mut unique = BTreeMap::<String, TransportPart>::new();
    for parts in artifact_parts {
        for part in parts {
            if !normalized_relative_path(&part.repository_path)
                || part.size_bytes == 0
                || part.size_bytes >= policy.maximum_file_bytes_exclusive
                || part.size_bytes > policy.maximum_batch_payload_bytes
                || !part.content_digest.validate()
            {
                return Err(CacheError::InvalidManifest(format!(
                    "family publication part {:?} violates publication limits",
                    part.repository_path
                )));
            }
            match unique.get(&part.repository_path) {
                Some(existing)
                    if existing.size_bytes != part.size_bytes
                        || existing.content_digest != part.content_digest =>
                {
                    return Err(CacheError::DigestMismatch {
                        expected: existing.content_digest.to_string(),
                        actual: part.content_digest.to_string(),
                    });
                }
                Some(_) => {}
                None => {
                    unique.insert(part.repository_path.clone(), part.clone());
                }
            }
        }
    }
    if unique.is_empty() {
        return Err(CacheError::InvalidManifest(
            "family publication requires at least one transport part".to_owned(),
        ));
    }
    let ordered = unique
        .into_values()
        .enumerate()
        .map(|(sequence, mut part)| {
            part.sequence = sequence as u64;
            part
        })
        .collect::<Vec<_>>();
    plan_publication_batches(&ordered, policy)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationBatchState {
    Planned,
    Committed,
    PayloadVerified,
    RecordCommitted,
    RemoteVerified,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadBatchObjectRecord {
    pub repository_path: String,
    pub size_bytes: u64,
    pub content_digest: ContentDigest,
    pub newly_introduced: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadBatchRecord {
    pub schema_version: u32,
    pub transaction_id: String,
    pub idempotency_key: ContentDigest,
    pub destination: PublicationDestination,
    pub authorized_repository: String,
    pub shard_id: String,
    pub branch: String,
    pub sequence: u64,
    pub payload_parent_head: String,
    pub payload_commit_id: String,
    pub planned_payload_bytes: u64,
    pub newly_committed_payload_bytes: u64,
    pub objects: Vec<PayloadBatchObjectRecord>,
}

impl PayloadBatchRecord {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn repository_path(&self) -> String {
        let destination = match self.destination {
            PublicationDestination::Private => "private",
            PublicationDestination::Public => "public",
        };
        format!(
            "transactions/{}/{destination}/batches/{:020}.json",
            self.transaction_id, self.sequence,
        )
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.transaction_id != self.idempotency_key.0
            || self.authorized_repository.trim().is_empty()
            || self.shard_id.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.payload_parent_head.trim().is_empty()
            || self.payload_commit_id.trim().is_empty()
            || self.objects.is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "payload batch record is incomplete".to_owned(),
            ));
        }
        let mut planned_bytes = 0u64;
        let mut new_bytes = 0u64;
        let mut previous_path: Option<&str> = None;
        let mut object_digests = BTreeSet::new();
        for object in &self.objects {
            if !normalized_relative_path(&object.repository_path)
                || object.size_bytes == 0
                || object.size_bytes >= GITHUB_HARD_FILE_BOUNDARY_BYTES
                || !object.content_digest.validate()
                || !object_digests.insert(&object.content_digest)
                || previous_path.is_some_and(|previous| previous >= object.repository_path.as_str())
            {
                return Err(CacheError::InvalidManifest(
                    "payload batch objects are invalid, duplicated, or unordered".to_owned(),
                ));
            }
            previous_path = Some(&object.repository_path);
            planned_bytes = planned_bytes
                .checked_add(object.size_bytes)
                .ok_or_else(|| {
                    CacheError::ResourceLimit("payload batch planned bytes exceed u64".to_owned())
                })?;
            if object.newly_introduced {
                new_bytes = new_bytes.checked_add(object.size_bytes).ok_or_else(|| {
                    CacheError::ResourceLimit("payload batch new bytes exceed u64".to_owned())
                })?;
            }
        }
        if planned_bytes != self.planned_payload_bytes
            || planned_bytes > GITHUB_MAX_PUBLICATION_BATCH_BYTES
            || new_bytes != self.newly_committed_payload_bytes
            || new_bytes == 0
            || new_bytes > GITHUB_MAX_PUBLICATION_BATCH_BYTES
        {
            return Err(CacheError::InvalidManifest(
                "payload batch record byte accounting is inconsistent".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationBatchJournal {
    pub plan: PublicationBatchPlan,
    pub state: PublicationBatchState,
    pub parent_head: Option<String>,
    pub commit_id: Option<String>,
    /// Content identities first introduced by this transaction's accepted
    /// commit. Existing remote objects are intentionally absent.
    pub newly_committed_digests: Vec<ContentDigest>,
    /// Separate required commit that durably records payload sizes and identities.
    pub record_commit: Option<PublicationCommitJournal>,
}

impl PublicationBatchJournal {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.plan.parts.is_empty()
            || self.plan.payload_bytes == 0
            || self.plan.payload_bytes > GITHUB_MAX_PUBLICATION_BATCH_BYTES
        {
            return Err(CacheError::InvalidManifest(
                "publication batch plan is empty or exceeds the payload limit".to_owned(),
            ));
        }
        let planned_bytes = self.plan.parts.iter().try_fold(0u64, |total, part| {
            if !normalized_relative_path(&part.repository_path)
                || part.size_bytes == 0
                || part.size_bytes >= GITHUB_HARD_FILE_BOUNDARY_BYTES
                || !part.content_digest.validate()
            {
                return Err(CacheError::InvalidManifest(
                    "publication batch contains an invalid part".to_owned(),
                ));
            }
            total.checked_add(part.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit("publication batch bytes exceed u64".to_owned())
            })
        })?;
        if planned_bytes != self.plan.payload_bytes {
            return Err(CacheError::InvalidManifest(
                "publication batch byte accounting does not match its parts".to_owned(),
            ));
        }
        let parent_present = self
            .parent_head
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());
        let commit_present = self
            .commit_id
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());
        if (self.state == PublicationBatchState::Planned && (parent_present || commit_present))
            || (self.state != PublicationBatchState::Planned && !(parent_present && commit_present))
        {
            return Err(CacheError::InvalidManifest(
                "publication batch state and payload commit identities disagree".to_owned(),
            ));
        }
        let planned_digests = self
            .plan
            .parts
            .iter()
            .map(|part| &part.content_digest)
            .collect::<BTreeSet<_>>();
        let new_digests = self.newly_committed_digests.iter().collect::<BTreeSet<_>>();
        if new_digests.len() != self.newly_committed_digests.len()
            || !new_digests.is_subset(&planned_digests)
            || (self.state == PublicationBatchState::Planned && !new_digests.is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "publication batch new-object accounting is invalid".to_owned(),
            ));
        }
        if let Some(record) = &self.record_commit {
            record.validate()?;
            if record.files.len() != 1 || record.receipt_digest.is_some() {
                return Err(CacheError::InvalidManifest(
                    "payload batch record must be one non-receipt metadata file".to_owned(),
                ));
            }
        }
        let record_state = self.record_commit.as_ref().map(|record| record.state);
        let state_valid = match self.state {
            PublicationBatchState::Planned | PublicationBatchState::Committed => {
                self.record_commit.is_none()
            }
            PublicationBatchState::PayloadVerified => {
                record_state == Some(PublicationCommitState::Planned)
            }
            PublicationBatchState::RecordCommitted => {
                record_state == Some(PublicationCommitState::Committed)
            }
            PublicationBatchState::RemoteVerified => {
                record_state == Some(PublicationCommitState::RemoteVerified)
                    || self.record_commit.is_none()
            }
        };
        if !state_valid {
            return Err(CacheError::InvalidManifest(
                "publication batch and durable record states disagree".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationCommitState {
    Planned,
    Committed,
    RemoteVerified,
}

/// One resumable metadata commit. `files` reuse the transport descriptor
/// because the remote boundary needs the same path, byte-count, and digest
/// contract for both large payload parts and small metadata objects.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationCommitJournal {
    pub files: Vec<TransportPart>,
    pub state: PublicationCommitState,
    pub parent_head: Option<String>,
    pub commit_id: Option<String>,
    /// File identities first introduced by this accepted commit.
    pub newly_committed_digests: Vec<ContentDigest>,
    /// Present only for the discoverability commit that makes the receipt
    /// visible together with the shard index and capacity ledger.
    pub receipt_digest: Option<ContentDigest>,
}

impl PublicationCommitJournal {
    pub fn planned(
        files: Vec<TransportPart>,
        receipt_digest: Option<ContentDigest>,
    ) -> Result<Self, CacheError> {
        validate_commit_files(&files)?;
        if receipt_digest
            .as_ref()
            .is_some_and(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "publication commit receipt digest is invalid".to_owned(),
            ));
        }
        Ok(Self {
            files,
            state: PublicationCommitState::Planned,
            parent_head: None,
            commit_id: None,
            newly_committed_digests: Vec::new(),
            receipt_digest,
        })
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        validate_commit_files(&self.files)?;
        if self
            .receipt_digest
            .as_ref()
            .is_some_and(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "publication commit receipt digest is invalid".to_owned(),
            ));
        }
        let parent_present = self
            .parent_head
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());
        let commit_present = self
            .commit_id
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());
        if (self.state == PublicationCommitState::Planned && (parent_present || commit_present))
            || (self.state != PublicationCommitState::Planned
                && !(parent_present && commit_present))
        {
            return Err(CacheError::InvalidManifest(
                "publication commit state and commit identities disagree".to_owned(),
            ));
        }
        let planned_digests: std::collections::BTreeSet<_> =
            self.files.iter().map(|file| &file.content_digest).collect();
        let new_digests: std::collections::BTreeSet<_> =
            self.newly_committed_digests.iter().collect();
        if new_digests.len() != self.newly_committed_digests.len()
            || !new_digests.is_subset(&planned_digests)
            || (self.state == PublicationCommitState::Planned
                && !self.newly_committed_digests.is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "publication commit new-object accounting is invalid".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn mark_committed(
        &mut self,
        parent_head: &str,
        commit_id: &str,
        newly_committed_digests: Vec<ContentDigest>,
    ) -> Result<(), CacheError> {
        if self.state != PublicationCommitState::Planned
            || parent_head.trim().is_empty()
            || commit_id.trim().is_empty()
        {
            return Err(CacheError::InvalidTransition(
                "metadata commit must advance from a planned state".to_owned(),
            ));
        }
        let planned_digests: std::collections::BTreeSet<_> =
            self.files.iter().map(|file| &file.content_digest).collect();
        let new_digests: std::collections::BTreeSet<_> = newly_committed_digests.iter().collect();
        if newly_committed_digests.is_empty()
            || new_digests.len() != newly_committed_digests.len()
            || !new_digests.is_subset(&planned_digests)
        {
            return Err(CacheError::InvalidTransition(
                "metadata commit must identify its newly written file identities".to_owned(),
            ));
        }
        self.parent_head = Some(parent_head.to_owned());
        self.commit_id = Some(commit_id.to_owned());
        self.newly_committed_digests = newly_committed_digests;
        self.state = PublicationCommitState::Committed;
        Ok(())
    }

    pub fn mark_reused(&mut self, revision: &str) -> Result<(), CacheError> {
        if self.state != PublicationCommitState::Planned || revision.trim().is_empty() {
            return Err(CacheError::InvalidTransition(
                "only planned metadata may be reused".to_owned(),
            ));
        }
        self.parent_head = Some(revision.to_owned());
        self.commit_id = Some(revision.to_owned());
        self.newly_committed_digests.clear();
        self.state = PublicationCommitState::Committed;
        self.state = PublicationCommitState::RemoteVerified;
        Ok(())
    }

    pub fn mark_remote_verified(&mut self, commit_id: &str) -> Result<(), CacheError> {
        if self.state != PublicationCommitState::Committed
            || self.commit_id.as_deref() != Some(commit_id)
        {
            return Err(CacheError::InvalidTransition(
                "metadata verification must match its recorded commit".to_owned(),
            ));
        }
        self.state = PublicationCommitState::RemoteVerified;
        Ok(())
    }
}

fn validate_commit_files(files: &[TransportPart]) -> Result<u64, CacheError> {
    if files.is_empty() {
        return Err(CacheError::InvalidManifest(
            "publication metadata commit must contain at least one file".to_owned(),
        ));
    }
    let mut total = 0u64;
    let mut paths = std::collections::BTreeSet::new();
    for (index, file) in files.iter().enumerate() {
        if file.sequence != index as u64
            || !normalized_relative_path(&file.repository_path)
            || file.size_bytes == 0
            || file.size_bytes >= GITHUB_HARD_FILE_BOUNDARY_BYTES
            || !file.content_digest.validate()
            || !paths.insert(&file.repository_path)
        {
            return Err(CacheError::InvalidManifest(format!(
                "publication metadata file {index} is invalid or duplicated"
            )));
        }
        total = total.checked_add(file.size_bytes).ok_or_else(|| {
            CacheError::ResourceLimit("publication metadata bytes exceed u64".to_owned())
        })?;
    }
    if total > GITHUB_MAX_PUBLICATION_BATCH_BYTES {
        return Err(CacheError::ResourceLimit(format!(
            "publication metadata commit {total} exceeds {GITHUB_MAX_PUBLICATION_BATCH_BYTES} bytes"
        )));
    }
    Ok(total)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPublicationJournal {
    pub destination: PublicationDestination,
    /// Stable owner/name policy identity, distinct from the transport URL.
    pub authorized_repository: String,
    pub repository: String,
    pub permission_evidence: RepositoryPermissionEvidence,
    pub shard_id: String,
    pub branch: String,
    pub expected_head: String,
    #[serde(default)]
    pub audit_evidence: Option<TargetPublicationAuditEvidence>,
    pub state: PublicationTargetState,
    pub batches: Vec<PublicationBatchJournal>,
    pub metadata_commit: Option<PublicationCommitJournal>,
    pub discoverability_commit: Option<PublicationCommitJournal>,
    pub receipt_digest: Option<ContentDigest>,
    pub retry_count: u32,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPublicationAuditEvidence {
    pub policy_id: String,
    pub authority_mode: PublicationAuthorityMode,
    pub validation_evidence_digests: Vec<ContentDigest>,
    pub contributor_authorization_digest: Option<ContentDigest>,
    pub reviewer_approvals: Vec<PublicationReviewEvidence>,
}

impl TargetPublicationAuditEvidence {
    pub fn validate(&self) -> Result<(), CacheError> {
        let valid_review = |review: &PublicationReviewEvidence| {
            review.approved
                && !review.reviewer_principal.trim().is_empty()
                && review.pull_request_number > 0
                && review.reviewed_head_revision.len() == 40
                && review
                    .reviewed_head_revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                && review.evidence_digest.validate()
        };
        if self.policy_id.trim().is_empty()
            || self.validation_evidence_digests.is_empty()
            || self
                .validation_evidence_digests
                .iter()
                .any(|digest| !digest.validate())
            || self
                .contributor_authorization_digest
                .as_ref()
                .is_some_and(|digest| !digest.validate())
            || self
                .reviewer_approvals
                .iter()
                .any(|review| !valid_review(review))
            || (self.authority_mode == PublicationAuthorityMode::OwnerDirect
                && self.contributor_authorization_digest.is_some())
            || (self.authority_mode == PublicationAuthorityMode::ContributorReviewed
                && (self.contributor_authorization_digest.is_none()
                    || self.reviewer_approvals.is_empty()))
        {
            return Err(CacheError::InvalidManifest(
                "publication audit evidence is incomplete or inconsistent with authority mode"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn attach_test_owner_audit_evidence(
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
) {
    journal
        .attach_target_audit_evidence(
            destination,
            TargetPublicationAuditEvidence {
                policy_id: "fixture-owner-policy".to_owned(),
                authority_mode: PublicationAuthorityMode::OwnerDirect,
                validation_evidence_digests: vec![ContentDigest::sha256(b"fixture-validation")],
                contributor_authorization_digest: None,
                reviewer_approvals: Vec::new(),
            },
        )
        .unwrap();
}

impl TargetPublicationJournal {
    /// Record that every payload part for this logical artifact was verified
    /// in a repository-wide family commit. Multiple logical artifacts may
    /// therefore share one physical Git commit without inventing per-artifact
    /// commits or batch-record commits.
    pub fn mark_payload_verified_by_family_commit(
        &mut self,
        parent_head: &str,
        commit_id: &str,
    ) -> Result<(), CacheError> {
        if !matches!(
            self.state,
            PublicationTargetState::Planned | PublicationTargetState::Uploading
        ) || self.expected_head != parent_head
            || parent_head.trim().is_empty()
            || commit_id.trim().is_empty()
            || self.batches.is_empty()
            || self
                .batches
                .iter()
                .any(|batch| batch.state != PublicationBatchState::Planned)
        {
            return Err(CacheError::InvalidTransition(
                "family payload verification requires an untouched planned target".to_owned(),
            ));
        }
        for batch in &mut self.batches {
            batch.state = PublicationBatchState::RemoteVerified;
            batch.parent_head = Some(parent_head.to_owned());
            batch.commit_id = Some(commit_id.to_owned());
            batch.newly_committed_digests.clear();
            batch.record_commit = None;
        }
        self.expected_head = commit_id.to_owned();
        self.state = PublicationTargetState::BatchVerified;
        Ok(())
    }

    pub fn plan_metadata_commit(&mut self, files: Vec<TransportPart>) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::BatchVerified
            || self
                .batches
                .iter()
                .any(|batch| batch.state != PublicationBatchState::RemoteVerified)
            || self.metadata_commit.is_some()
        {
            return Err(CacheError::InvalidTransition(
                "metadata may be planned only after every payload batch is verified".to_owned(),
            ));
        }
        self.metadata_commit = Some(PublicationCommitJournal::planned(files, None)?);
        Ok(())
    }

    pub(crate) fn plan_discoverability_commit(
        &mut self,
        files: Vec<TransportPart>,
        receipt_digest: ContentDigest,
    ) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::RemoteVerified
            || self
                .metadata_commit
                .as_ref()
                .is_none_or(|commit| commit.state != PublicationCommitState::RemoteVerified)
            || self.discoverability_commit.is_some()
        {
            return Err(CacheError::InvalidTransition(
                "discoverability may be planned only after metadata is remotely verified"
                    .to_owned(),
            ));
        }
        if !files
            .iter()
            .any(|file| file.content_digest == receipt_digest)
        {
            return Err(CacheError::InvalidManifest(
                "discoverability commit does not contain its publication receipt".to_owned(),
            ));
        }
        self.discoverability_commit = Some(PublicationCommitJournal::planned(
            files,
            Some(receipt_digest),
        )?);
        Ok(())
    }

    pub fn discard_planned_discoverability_after_conflict(
        &mut self,
        current_head: impl Into<String>,
    ) -> Result<(), CacheError> {
        if self
            .discoverability_commit
            .as_ref()
            .is_none_or(|commit| commit.state != PublicationCommitState::Planned)
        {
            return Err(CacheError::InvalidTransition(
                "only an uncommitted discoverability plan may be rebuilt".to_owned(),
            ));
        }
        self.expected_head = current_head.into();
        self.retry_count = self.retry_count.saturating_add(1);
        self.discoverability_commit = None;
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), CacheError> {
        if !matches!(
            self.state,
            PublicationTargetState::Planned
                | PublicationTargetState::BatchVerified
                | PublicationTargetState::Failed
        ) {
            return Err(CacheError::InvalidTransition(format!(
                "target {:?} cannot start from {:?}",
                self.destination, self.state
            )));
        }
        self.state = PublicationTargetState::Uploading;
        self.failure = None;
        Ok(())
    }

    pub fn mark_batch_committed(
        &mut self,
        sequence: usize,
        parent_head: &str,
        commit_id: &str,
        newly_committed_digests: Vec<ContentDigest>,
    ) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::Uploading
            || parent_head != self.expected_head
            || commit_id.trim().is_empty()
        {
            return Err(CacheError::InvalidTransition(
                "batch commit must be an uploading compare-and-swap from the expected head"
                    .to_owned(),
            ));
        }
        let batch = self.batches.get_mut(sequence).ok_or_else(|| {
            CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
        })?;
        if batch.state != PublicationBatchState::Planned {
            return Err(CacheError::InvalidTransition(format!(
                "batch {sequence} is already {:?}",
                batch.state
            )));
        }
        let planned_digests: std::collections::BTreeSet<_> = batch
            .plan
            .parts
            .iter()
            .map(|part| &part.content_digest)
            .collect();
        let new_digests: std::collections::BTreeSet<_> = newly_committed_digests.iter().collect();
        if newly_committed_digests.is_empty()
            || new_digests.len() != newly_committed_digests.len()
            || !new_digests.is_subset(&planned_digests)
        {
            return Err(CacheError::InvalidTransition(
                "committed batch must identify its unique newly introduced objects".to_owned(),
            ));
        }
        batch.state = PublicationBatchState::Committed;
        batch.parent_head = Some(parent_head.to_owned());
        batch.commit_id = Some(commit_id.to_owned());
        batch.newly_committed_digests = newly_committed_digests;
        self.expected_head = commit_id.to_owned();
        Ok(())
    }

    pub fn plan_verified_batch_record(
        &mut self,
        sequence: usize,
        commit_id: &str,
        record_file: TransportPart,
    ) -> Result<(), CacheError> {
        let batch = self.batches.get_mut(sequence).ok_or_else(|| {
            CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
        })?;
        if batch.state != PublicationBatchState::Committed
            || batch.commit_id.as_deref() != Some(commit_id)
            || batch.record_commit.is_some()
        {
            return Err(CacheError::InvalidTransition(
                "batch-record planning must match the verified payload commit".to_owned(),
            ));
        }
        let record_commit = PublicationCommitJournal::planned(vec![record_file], None)?;
        batch.record_commit = Some(record_commit);
        batch.state = PublicationBatchState::PayloadVerified;
        Ok(())
    }

    pub fn mark_batch_record_committed(
        &mut self,
        sequence: usize,
        parent_head: &str,
        commit_id: &str,
        newly_committed_digests: Vec<ContentDigest>,
    ) -> Result<(), CacheError> {
        if self.expected_head != parent_head {
            return Err(CacheError::InvalidTransition(
                "batch-record commit must use the target's expected head".to_owned(),
            ));
        }
        let batch = self.batches.get_mut(sequence).ok_or_else(|| {
            CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
        })?;
        if batch.state != PublicationBatchState::PayloadVerified {
            return Err(CacheError::InvalidTransition(
                "batch record may be committed only after payload verification".to_owned(),
            ));
        }
        batch
            .record_commit
            .as_mut()
            .ok_or_else(|| {
                CacheError::InvalidTransition("batch record was not planned".to_owned())
            })?
            .mark_committed(parent_head, commit_id, newly_committed_digests)?;
        batch.state = PublicationBatchState::RecordCommitted;
        self.expected_head = commit_id.to_owned();
        Ok(())
    }

    pub fn mark_batch_record_reused(
        &mut self,
        sequence: usize,
        revision: &str,
    ) -> Result<(), CacheError> {
        let batch = self.batches.get_mut(sequence).ok_or_else(|| {
            CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
        })?;
        if batch.state != PublicationBatchState::PayloadVerified {
            return Err(CacheError::InvalidTransition(
                "only a planned verified-payload record may be reused".to_owned(),
            ));
        }
        batch
            .record_commit
            .as_mut()
            .ok_or_else(|| {
                CacheError::InvalidTransition("batch record was not planned".to_owned())
            })?
            .mark_reused(revision)?;
        batch.state = PublicationBatchState::RemoteVerified;
        self.state = PublicationTargetState::BatchVerified;
        Ok(())
    }

    pub fn mark_batch_record_verified(
        &mut self,
        sequence: usize,
        commit_id: &str,
    ) -> Result<(), CacheError> {
        let batch = self.batches.get_mut(sequence).ok_or_else(|| {
            CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
        })?;
        if batch.state != PublicationBatchState::RecordCommitted {
            return Err(CacheError::InvalidTransition(
                "batch-record verification requires its committed state".to_owned(),
            ));
        }
        batch
            .record_commit
            .as_mut()
            .ok_or_else(|| {
                CacheError::InvalidTransition("batch record was not planned".to_owned())
            })?
            .mark_remote_verified(commit_id)?;
        batch.state = PublicationBatchState::RemoteVerified;
        self.state = PublicationTargetState::BatchVerified;
        Ok(())
    }

    pub fn mark_batch_reused(
        &mut self,
        sequence: usize,
        verified_revision: &str,
    ) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::Uploading || verified_revision.trim().is_empty() {
            return Err(CacheError::InvalidTransition(
                "reused batch must be verified while the target is uploading".to_owned(),
            ));
        }
        let batch = self.batches.get_mut(sequence).ok_or_else(|| {
            CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
        })?;
        if batch.state != PublicationBatchState::Planned {
            return Err(CacheError::InvalidTransition(format!(
                "batch {sequence} is already {:?}",
                batch.state
            )));
        }
        batch.state = PublicationBatchState::RemoteVerified;
        batch.parent_head = Some(verified_revision.to_owned());
        batch.commit_id = Some(verified_revision.to_owned());
        batch.newly_committed_digests.clear();
        self.state = PublicationTargetState::BatchVerified;
        Ok(())
    }

    pub fn record_ref_conflict(&mut self, new_head: impl Into<String>) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::Uploading {
            return Err(CacheError::InvalidTransition(
                "ref conflict is valid only while uploading".to_owned(),
            ));
        }
        self.expected_head = new_head.into();
        self.retry_count = self.retry_count.saturating_add(1);
        Ok(())
    }

    pub fn mark_remote_verified(&mut self) -> Result<(), CacheError> {
        if self
            .batches
            .iter()
            .any(|batch| batch.state != PublicationBatchState::RemoteVerified)
        {
            return Err(CacheError::InvalidTransition(
                "all batches must be remotely verified first".to_owned(),
            ));
        }
        self.state = PublicationTargetState::RemoteVerified;
        Ok(())
    }

    pub fn complete_receipt(&mut self, receipt: ContentDigest) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::RemoteVerified
            || !receipt.validate()
            || self.discoverability_commit.as_ref().is_none_or(|commit| {
                commit.state != PublicationCommitState::RemoteVerified
                    || commit.receipt_digest.as_ref() != Some(&receipt)
            })
        {
            return Err(CacheError::InvalidTransition(
                "receipt completion requires a remotely verified discoverability commit".to_owned(),
            ));
        }
        self.receipt_digest = Some(receipt);
        self.state = PublicationTargetState::ReceiptComplete;
        Ok(())
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        self.failure = Some(message.into());
        self.state = PublicationTargetState::Failed;
    }

    pub(crate) fn abandon(&mut self, reason: &str) -> Result<(), CacheError> {
        if reason.trim().is_empty() || reason.len() > 4_096 {
            return Err(CacheError::InvalidManifest(
                "publication abandonment reason must contain 1 to 4096 bytes".to_owned(),
            ));
        }
        if self.state == PublicationTargetState::ReceiptComplete {
            return Err(CacheError::InvalidTransition(
                "a receipt-complete publication target cannot be abandoned".to_owned(),
            ));
        }
        self.failure = Some(reason.trim().to_owned());
        self.state = PublicationTargetState::Abandoned;
        Ok(())
    }

    pub(crate) fn restore_failed_resume_state(&mut self) -> Result<(), CacheError> {
        if self.state != PublicationTargetState::Failed {
            return Err(CacheError::InvalidTransition(
                "only a failed publication target needs resume-state restoration".to_owned(),
            ));
        }
        self.state = if self
            .batches
            .iter()
            .any(|batch| batch.state != PublicationBatchState::RemoteVerified)
        {
            PublicationTargetState::Uploading
        } else if self
            .metadata_commit
            .as_ref()
            .is_some_and(|commit| commit.state == PublicationCommitState::RemoteVerified)
        {
            PublicationTargetState::RemoteVerified
        } else {
            PublicationTargetState::BatchVerified
        };
        self.failure = None;
        Ok(())
    }
}

pub fn build_payload_batch_record(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    sequence: usize,
) -> Result<PayloadBatchRecord, CacheError> {
    let target = journal.targets.get(&destination).ok_or_else(|| {
        CacheError::InvalidTransition(format!(
            "transaction has no {destination:?} publication target"
        ))
    })?;
    let batch = target.batches.get(sequence).ok_or_else(|| {
        CacheError::InvalidTransition(format!("unknown publication batch {sequence}"))
    })?;
    if batch.state != PublicationBatchState::Committed {
        return Err(CacheError::InvalidTransition(
            "payload batch record requires a committed, independently verified payload".to_owned(),
        ));
    }
    let parent_head = batch.parent_head.clone().ok_or_else(|| {
        CacheError::InvalidTransition("payload batch has no parent head".to_owned())
    })?;
    let commit_id = batch.commit_id.clone().ok_or_else(|| {
        CacheError::InvalidTransition("payload batch has no commit identity".to_owned())
    })?;
    let newly_committed = batch
        .newly_committed_digests
        .iter()
        .collect::<BTreeSet<_>>();
    let mut objects = batch
        .plan
        .parts
        .iter()
        .map(|part| PayloadBatchObjectRecord {
            repository_path: part.repository_path.clone(),
            size_bytes: part.size_bytes,
            content_digest: part.content_digest.clone(),
            newly_introduced: newly_committed.contains(&part.content_digest),
        })
        .collect::<Vec<_>>();
    objects.sort_by(|left, right| left.repository_path.cmp(&right.repository_path));
    let newly_committed_payload_bytes = objects
        .iter()
        .filter(|object| object.newly_introduced)
        .try_fold(0u64, |total, object| {
            total.checked_add(object.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit("payload batch new bytes exceed u64".to_owned())
            })
        })?;
    let record = PayloadBatchRecord {
        schema_version: 1,
        transaction_id: journal.transaction_id.clone(),
        idempotency_key: journal.idempotency_key.clone(),
        destination,
        authorized_repository: target.authorized_repository.clone(),
        shard_id: target.shard_id.clone(),
        branch: target.branch.clone(),
        sequence: sequence as u64,
        payload_parent_head: parent_head,
        payload_commit_id: commit_id,
        planned_payload_bytes: batch.plan.payload_bytes,
        newly_committed_payload_bytes,
        objects,
    };
    record.validate()?;
    Ok(record)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationTransactionJournal {
    pub schema_version: u32,
    pub transaction_id: String,
    pub idempotency_key: ContentDigest,
    pub semantic_digest: ContentDigest,
    pub target_manifest_digests: BTreeMap<PublicationDestination, ContentDigest>,
    pub payload_digest: ContentDigest,
    pub policy_digest: ContentDigest,
    pub targets: BTreeMap<PublicationDestination, TargetPublicationJournal>,
}

#[derive(Serialize)]
struct IdempotencyEnvelope<'a> {
    schema_version: u32,
    semantic_digest: &'a ContentDigest,
    target_manifest_digests: &'a BTreeMap<PublicationDestination, ContentDigest>,
    payload_digest: &'a ContentDigest,
    policy_digest: &'a ContentDigest,
    target: PublicationTarget,
    repositories: &'a BTreeMap<PublicationDestination, String>,
    authorized_repositories: &'a BTreeMap<PublicationDestination, String>,
    shard_ids: &'a BTreeMap<PublicationDestination, String>,
    branches: &'a BTreeMap<PublicationDestination, String>,
}

impl PublicationTransactionJournal {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        semantic_digest: ContentDigest,
        target_manifest_digests: BTreeMap<PublicationDestination, ContentDigest>,
        payload_digest: ContentDigest,
        policy_digest: ContentDigest,
        target: PublicationTarget,
        repositories: BTreeMap<PublicationDestination, String>,
        authorized_repositories: BTreeMap<PublicationDestination, String>,
        permission_evidence: BTreeMap<PublicationDestination, RepositoryPermissionEvidence>,
        shard_ids: BTreeMap<PublicationDestination, String>,
        branches: BTreeMap<PublicationDestination, String>,
        expected_heads: BTreeMap<PublicationDestination, String>,
        batches: &[PublicationBatchPlan],
    ) -> Result<Self, CacheError> {
        for digest in [&semantic_digest, &payload_digest, &policy_digest] {
            if !digest.validate() {
                return Err(CacheError::InvalidManifest(
                    "publication transaction contains an invalid identity".to_owned(),
                ));
            }
        }
        let required = destinations(target);
        if required.is_empty()
            || repositories.keys().copied().collect::<Vec<_>>() != required
            || authorized_repositories.keys().copied().collect::<Vec<_>>() != required
            || permission_evidence.keys().copied().collect::<Vec<_>>() != required
            || shard_ids.keys().copied().collect::<Vec<_>>() != required
            || branches.keys().copied().collect::<Vec<_>>() != required
            || expected_heads.keys().copied().collect::<Vec<_>>() != required
            || target_manifest_digests.keys().copied().collect::<Vec<_>>() != required
        {
            return Err(CacheError::InvalidManifest(
                "publication repositories, shards, branches, and heads must exactly match the requested targets"
                    .to_owned(),
            ));
        }
        if target_manifest_digests
            .values()
            .any(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "publication target manifest digest is invalid".to_owned(),
            ));
        }
        let idempotency_key = canonical_digest(&IdempotencyEnvelope {
            schema_version: 1,
            semantic_digest: &semantic_digest,
            target_manifest_digests: &target_manifest_digests,
            payload_digest: &payload_digest,
            policy_digest: &policy_digest,
            target,
            repositories: &repositories,
            authorized_repositories: &authorized_repositories,
            shard_ids: &shard_ids,
            branches: &branches,
        })?;
        let mut journals = BTreeMap::new();
        for destination in required {
            let repository = repositories.get(&destination).cloned().unwrap_or_default();
            let authorized_repository = authorized_repositories
                .get(&destination)
                .cloned()
                .unwrap_or_default();
            let target_permission_evidence = permission_evidence
                .get(&destination)
                .cloned()
                .ok_or_else(|| {
                    CacheError::InvalidManifest(
                        "publication target permission evidence is missing".to_owned(),
                    )
                })?;
            let shard_id = shard_ids.get(&destination).cloned().unwrap_or_default();
            let branch = branches.get(&destination).cloned().unwrap_or_default();
            let expected_head = expected_heads
                .get(&destination)
                .cloned()
                .unwrap_or_default();
            if repository.trim().is_empty()
                || authorized_repository.trim().is_empty()
                || shard_id.trim().is_empty()
                || branch.trim().is_empty()
                || expected_head.trim().is_empty()
            {
                return Err(CacheError::InvalidManifest(
                    "publication target repository, shard, branch, and expected head are required"
                        .to_owned(),
                ));
            }
            target_permission_evidence.validate()?;
            if !target_permission_evidence
                .repository
                .eq_ignore_ascii_case(&authorized_repository)
            {
                return Err(CacheError::InvalidManifest(
                    "permission evidence repository does not match publication authority"
                        .to_owned(),
                ));
            }
            journals.insert(
                destination,
                TargetPublicationJournal {
                    destination,
                    authorized_repository,
                    repository,
                    permission_evidence: target_permission_evidence,
                    shard_id,
                    branch,
                    expected_head,
                    audit_evidence: None,
                    state: PublicationTargetState::Planned,
                    batches: batches
                        .iter()
                        .cloned()
                        .map(|plan| PublicationBatchJournal {
                            plan,
                            state: PublicationBatchState::Planned,
                            parent_head: None,
                            commit_id: None,
                            newly_committed_digests: Vec::new(),
                            record_commit: None,
                        })
                        .collect(),
                    metadata_commit: None,
                    discoverability_commit: None,
                    receipt_digest: None,
                    retry_count: 0,
                    failure: None,
                },
            );
        }
        Ok(Self {
            schema_version: 1,
            transaction_id: idempotency_key.0.clone(),
            idempotency_key,
            semantic_digest,
            target_manifest_digests,
            payload_digest,
            policy_digest,
            targets: journals,
        })
    }

    pub fn complete(&self) -> bool {
        self.targets
            .values()
            .all(|target| target.state == PublicationTargetState::ReceiptComplete)
    }

    pub fn attach_target_audit_evidence(
        &mut self,
        destination: PublicationDestination,
        evidence: TargetPublicationAuditEvidence,
    ) -> Result<(), CacheError> {
        evidence.validate()?;
        let target = self.targets.get_mut(&destination).ok_or_else(|| {
            CacheError::InvalidTransition(format!(
                "transaction has no {destination:?} audit target"
            ))
        })?;
        if target.state != PublicationTargetState::Planned || target.audit_evidence.is_some() {
            return Err(CacheError::InvalidTransition(
                "publication audit evidence is immutable after target execution begins".to_owned(),
            ));
        }
        target.audit_evidence = Some(evidence);
        Ok(())
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.transaction_id != self.idempotency_key.0
            || !self.idempotency_key.validate()
            || !self.semantic_digest.validate()
            || !self.payload_digest.validate()
            || !self.policy_digest.validate()
            || self.targets.is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "publication transaction journal identity is invalid".to_owned(),
            ));
        }
        let target = match (
            self.targets.contains_key(&PublicationDestination::Private),
            self.targets.contains_key(&PublicationDestination::Public),
        ) {
            (true, true) => PublicationTarget::Both,
            (true, false) => PublicationTarget::Private,
            (false, true) => PublicationTarget::Public,
            (false, false) => PublicationTarget::None,
        };
        let repositories: BTreeMap<_, _> = self
            .targets
            .iter()
            .map(|(destination, journal)| (*destination, journal.repository.clone()))
            .collect();
        let authorized_repositories: BTreeMap<_, _> = self
            .targets
            .iter()
            .map(|(destination, journal)| (*destination, journal.authorized_repository.clone()))
            .collect();
        let shard_ids: BTreeMap<_, _> = self
            .targets
            .iter()
            .map(|(destination, journal)| (*destination, journal.shard_id.clone()))
            .collect();
        let branches: BTreeMap<_, _> = self
            .targets
            .iter()
            .map(|(destination, journal)| (*destination, journal.branch.clone()))
            .collect();
        let expected = canonical_digest(&IdempotencyEnvelope {
            schema_version: 1,
            semantic_digest: &self.semantic_digest,
            target_manifest_digests: &self.target_manifest_digests,
            payload_digest: &self.payload_digest,
            policy_digest: &self.policy_digest,
            target,
            repositories: &repositories,
            authorized_repositories: &authorized_repositories,
            shard_ids: &shard_ids,
            branches: &branches,
        })?;
        if expected != self.idempotency_key {
            return Err(CacheError::InvalidManifest(
                "publication journal no longer matches its idempotency identity".to_owned(),
            ));
        }
        if self
            .target_manifest_digests
            .keys()
            .copied()
            .collect::<Vec<_>>()
            != destinations(target)
            || self
                .target_manifest_digests
                .values()
                .any(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "publication target manifest identities do not match the targets".to_owned(),
            ));
        }
        for (destination, journal) in &self.targets {
            if destination != &journal.destination
                || journal.authorized_repository.trim().is_empty()
                || journal.repository.trim().is_empty()
                || journal.shard_id.trim().is_empty()
                || journal.branch.trim().is_empty()
                || journal.expected_head.trim().is_empty()
                || journal
                    .audit_evidence
                    .as_ref()
                    .is_some_and(|evidence| evidence.validate().is_err())
                || journal.batches.iter().enumerate().any(|(index, batch)| {
                    batch.plan.sequence != index as u64
                        || batch.validate().is_err()
                        || batch.record_commit.as_ref().is_some_and(|record| {
                            record.files[0].repository_path
                                != format!(
                                    "transactions/{}/{}/batches/{:020}.json",
                                    self.transaction_id,
                                    match destination {
                                        PublicationDestination::Private => "private",
                                        PublicationDestination::Public => "public",
                                    },
                                    index
                                )
                        })
                })
            {
                return Err(CacheError::InvalidManifest(format!(
                    "publication target journal {destination:?} is invalid"
                )));
            }
            journal.permission_evidence.validate()?;
            if !journal
                .permission_evidence
                .repository
                .eq_ignore_ascii_case(&journal.authorized_repository)
            {
                return Err(CacheError::InvalidManifest(format!(
                    "publication target journal {destination:?} permission repository is invalid"
                )));
            }
            if let Some(commit) = &journal.metadata_commit {
                commit.validate()?;
            }
            if let Some(commit) = &journal.discoverability_commit {
                commit.validate()?;
                if commit.receipt_digest.as_ref().is_none_or(|receipt_digest| {
                    !commit
                        .files
                        .iter()
                        .any(|file| &file.content_digest == receipt_digest)
                }) {
                    return Err(CacheError::InvalidManifest(
                        "discoverability commit does not contain its receipt identity".to_owned(),
                    ));
                }
            }
            if journal.discoverability_commit.is_some()
                && journal
                    .metadata_commit
                    .as_ref()
                    .is_none_or(|commit| commit.state != PublicationCommitState::RemoteVerified)
            {
                return Err(CacheError::InvalidManifest(
                    "discoverability commit precedes verified immutable metadata".to_owned(),
                ));
            }
            if journal.state == PublicationTargetState::ReceiptComplete
                && (journal.receipt_digest.is_none()
                    || journal
                        .discoverability_commit
                        .as_ref()
                        .is_none_or(|commit| {
                            commit.state != PublicationCommitState::RemoteVerified
                                || commit.receipt_digest != journal.receipt_digest
                        }))
            {
                return Err(CacheError::InvalidManifest(
                    "receipt-complete target lacks a verified discoverability commit".to_owned(),
                ));
            }
            if journal
                .discoverability_commit
                .as_ref()
                .is_some_and(|commit| commit.state == PublicationCommitState::RemoteVerified)
                && journal.state != PublicationTargetState::ReceiptComplete
            {
                return Err(CacheError::InvalidManifest(
                    "a remotely verified discoverability commit must be receipt-complete"
                        .to_owned(),
                ));
            }
        }
        Ok(())
    }
}

fn destinations(target: PublicationTarget) -> Vec<PublicationDestination> {
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteCommitRequest {
    pub repository: String,
    pub branch: String,
    pub expected_head: String,
    pub message: String,
    pub parts: Vec<TransportPart>,
    /// Existing paths removed from the new tree. The prior blobs remain in
    /// Git history and continue to count toward repository capacity.
    #[serde(default)]
    pub delete_paths: Vec<String>,
}

impl RemoteCommitRequest {
    pub fn validate_limits(&self) -> Result<u64, CacheError> {
        if self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.expected_head.trim().is_empty()
            || self.message.trim().is_empty()
            || (self.parts.is_empty() && self.delete_paths.is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "remote commit request is incomplete".to_owned(),
            ));
        }
        let mut paths = std::collections::BTreeSet::new();
        let mut total = 0u64;
        for part in &self.parts {
            if part.size_bytes == 0
                || part.size_bytes >= GITHUB_HARD_FILE_BOUNDARY_BYTES
                || !part.content_digest.validate()
                || !normalized_relative_path(&part.repository_path)
                || !paths.insert(&part.repository_path)
            {
                return Err(CacheError::InvalidManifest(
                    "remote commit contains an invalid or duplicate transport part".to_owned(),
                ));
            }
            total = total.checked_add(part.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit("remote commit size exceeds u64".to_owned())
            })?;
        }
        let mut deletions = std::collections::BTreeSet::new();
        let mut deletion_path_bytes = 0u64;
        for path in &self.delete_paths {
            if !normalized_relative_path(path) || !deletions.insert(path) || paths.contains(path) {
                return Err(CacheError::InvalidManifest(
                    "remote commit contains an invalid, duplicate, or replaced-and-deleted path"
                        .to_owned(),
                ));
            }
            deletion_path_bytes = deletion_path_bytes
                .checked_add(path.len() as u64)
                .ok_or_else(|| {
                    CacheError::ResourceLimit("deletion path bytes exceed u64".to_owned())
                })?;
        }
        if self.delete_paths.len() > 1_000_000 || deletion_path_bytes > 128 * 1024 * 1024 {
            return Err(CacheError::ResourceLimit(
                "remote deletion commit exceeds its bounded path inventory".to_owned(),
            ));
        }
        if total > GITHUB_MAX_PUBLICATION_BATCH_BYTES {
            return Err(CacheError::ResourceLimit(format!(
                "remote commit payload {total} exceeds {GITHUB_MAX_PUBLICATION_BATCH_BYTES} bytes"
            )));
        }
        Ok(total)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompareAndSwapResult {
    Committed { commit_id: String },
    RefConflict { current_head: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteRefCreationRequest {
    pub repository: String,
    pub branch: String,
    pub message: String,
    pub parts: Vec<TransportPart>,
}

impl RemoteRefCreationRequest {
    pub fn validate_limits(&self) -> Result<u64, CacheError> {
        if self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.message.trim().is_empty()
            || self.parts.is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "remote ref creation request is incomplete".to_owned(),
            ));
        }
        let request = RemoteCommitRequest {
            repository: self.repository.clone(),
            branch: self.branch.clone(),
            // Limit validation does not interpret the revision. A syntactically
            // valid sentinel keeps creation requests subject to exactly the
            // same path and byte bounds as ordinary commits.
            expected_head: "0000000000000000000000000000000000000000".to_owned(),
            message: self.message.clone(),
            parts: self.parts.clone(),
            delete_paths: Vec::new(),
        };
        request.validate_limits()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CreateRefResult {
    Created { commit_id: String },
    RefExists { current_head: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtomicRemoteCommitRequest {
    pub repository: String,
    pub commits: Vec<RemoteCommitRequest>,
}

impl AtomicRemoteCommitRequest {
    pub fn validate_limits(&self) -> Result<(), CacheError> {
        if self.repository.trim().is_empty() || self.commits.len() < 2 {
            return Err(CacheError::InvalidManifest(
                "atomic remote commit requires one repository and at least two refs".to_owned(),
            ));
        }
        let mut branches = std::collections::BTreeSet::new();
        for commit in &self.commits {
            if commit.repository != self.repository || !branches.insert(commit.branch.as_str()) {
                return Err(CacheError::InvalidManifest(
                    "atomic remote commits must target distinct refs in one repository".to_owned(),
                ));
            }
            commit.validate_limits()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AtomicCompareAndSwapResult {
    Committed {
        commit_ids: std::collections::BTreeMap<String, String>,
    },
    RefConflict {
        current_heads: std::collections::BTreeMap<String, String>,
    },
}

/// Minimal no-checkout transport boundary. Implementations own a verified
/// local part source keyed by each descriptor's SHA-256 digest and may use a
/// bounded temporary bare object database or an API for small metadata.
pub trait RemoteGitStore: Send + Sync {
    fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError>;
    fn immutable_path_digest(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
    ) -> Result<Option<ContentDigest>, CacheError>;
    fn read_committed_path(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        maximum_bytes: u64,
        cancellation: &CancellationToken,
        writer: &mut dyn Write,
    ) -> Result<RemoteReadReport, CacheError>;
    fn list_committed_paths(
        &self,
        _repository: &str,
        _revision: &str,
        _prefix: &str,
        _maximum_paths: u64,
        _maximum_total_path_bytes: u64,
        _cancellation: &CancellationToken,
    ) -> Result<RemotePathListReport, CacheError> {
        Err(CacheError::ReadOnlyLayer(
            "remote transport does not support bounded tree enumeration".to_owned(),
        ))
    }
    fn compare_and_swap_commit(
        &self,
        request: &RemoteCommitRequest,
    ) -> Result<CompareAndSwapResult, CacheError>;
    fn create_ref_commit_if_absent(
        &self,
        _request: &RemoteRefCreationRequest,
    ) -> Result<CreateRefResult, CacheError> {
        Err(CacheError::ReadOnlyLayer(
            "remote transport does not support atomic ref creation".to_owned(),
        ))
    }
    fn compare_and_swap_commits_atomically(
        &self,
        _request: &AtomicRemoteCommitRequest,
    ) -> Result<AtomicCompareAndSwapResult, CacheError> {
        Err(CacheError::ReadOnlyLayer(
            "remote transport does not support atomic multi-ref commits".to_owned(),
        ))
    }
    fn verify_committed_part(
        &self,
        repository: &str,
        revision: &str,
        part: &TransportPart,
    ) -> Result<(), CacheError>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteReadReport {
    pub repository_path: String,
    pub revision: String,
    pub size_bytes: u64,
    pub content_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemotePathListReport {
    pub prefix: String,
    pub revision: String,
    pub paths: Vec<String>,
    pub total_path_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_core::{CancellationReason, ResourceProfile};

    fn test_policy() -> TransportPolicy {
        TransportPolicy {
            maximum_file_bytes_exclusive: 8,
            split_part_bytes: 4,
            maximum_batch_payload_bytes: 6,
            maximum_pending_batches: 1,
        }
    }

    #[test]
    fn hard_limits_cannot_be_configured_upward() {
        let policy = TransportPolicy {
            split_part_bytes: DEFAULT_SPLIT_PART_BYTES + 1,
            ..TransportPolicy::default()
        };
        assert!(policy.validate().is_err());
        let policy = TransportPolicy {
            maximum_batch_payload_bytes: GITHUB_MAX_PUBLICATION_BATCH_BYTES + 1,
            ..TransportPolicy::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn encoded_stream_is_hashed_split_and_batched_within_bounds() {
        let mut input = &b"abcdefghij"[..];
        let mut captured = Vec::new();
        let mut resources = ResourcePolicy::for_profile(ResourceProfile::NormalWorkstation);
        resources.maximum_memory_bytes = Some(16);
        resources.maximum_transfer_bytes = Some(16);
        let record = stream_split_encoded(
            &mut input,
            ContentDigest::sha256(b"logical payload"),
            "test-encoder-v1",
            &test_policy(),
            &resources,
            &CancellationToken::new(),
            |part, bytes| {
                captured.push((part.clone(), bytes.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(record.package_size_bytes, 10);
        assert_eq!(record.ordered_parts.len(), 3);
        assert_eq!(captured[2].1, b"ij");
        let batches = plan_publication_batches(&record.ordered_parts, &test_policy()).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].payload_bytes, 4);
        assert_eq!(batches[1].payload_bytes, 6);
    }

    #[test]
    fn transport_identity_preserves_canonical_identity_and_rejects_part_reordering() {
        let mut input = &b"abcdefghij"[..];
        let canonical_payload_digest = ContentDigest::sha256(b"exact hp payload envelope");
        let record = stream_split_encoded(
            &mut input,
            canonical_payload_digest.clone(),
            "deterministic-encoder-v1",
            &test_policy(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            |_, _| Ok(()),
        )
        .unwrap();
        let first_transport_digest = record.digest().unwrap();

        let mut alternate_transport = record.clone();
        alternate_transport.encoder_profile = "deterministic-encoder-v2".to_owned();
        assert_eq!(
            alternate_transport.canonical_payload_digest,
            canonical_payload_digest
        );
        assert_ne!(
            alternate_transport.digest().unwrap(),
            first_transport_digest
        );

        let mut reordered = record;
        reordered.ordered_parts.swap(0, 1);
        assert!(reordered.validate().is_err());
    }

    #[test]
    fn cancellation_stops_before_emitting_parts() {
        let token = CancellationToken::new();
        token.cancel(CancellationReason::UserRequested);
        let mut input = &b"abcdefghij"[..];
        let result = stream_split_encoded(
            &mut input,
            ContentDigest::sha256(b"logical payload"),
            "test-encoder-v1",
            &test_policy(),
            &ResourcePolicy::default(),
            &token,
            |_, _| panic!("cancelled splitter must not emit a part"),
        );
        assert!(matches!(result, Err(CacheError::Cancelled(_))));
    }

    fn simulated_parts(sizes: &[u64]) -> Vec<TransportPart> {
        sizes
            .iter()
            .enumerate()
            .map(|(index, size)| TransportPart {
                sequence: index as u64,
                repository_path: format!("objects/{index}.part"),
                size_bytes: *size,
                content_digest: ContentDigest::sha256(format!("part-{index}").as_bytes()),
            })
            .collect()
    }

    #[test]
    fn one_billion_byte_batch_boundary_is_inclusive() {
        let policy = TransportPolicy::default();
        let parts = simulated_parts(&[
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            DEFAULT_SPLIT_PART_BYTES,
            GITHUB_MAX_PUBLICATION_BATCH_BYTES - 10 * DEFAULT_SPLIT_PART_BYTES,
            1,
        ]);
        let batches = plan_publication_batches(&parts, &policy).unwrap();
        assert_eq!(batches[0].payload_bytes, GITHUB_MAX_PUBLICATION_BATCH_BYTES);
        assert_eq!(batches[1].payload_bytes, 1);
    }

    #[test]
    fn family_batching_does_not_create_one_commit_per_artifact() {
        let artifacts = (0..863)
            .map(|artifact| {
                vec![TransportPart {
                    sequence: 0,
                    repository_path: format!("objects/{artifact:04}.part"),
                    size_bytes: 1_000_000,
                    content_digest: ContentDigest::sha256(
                        format!("quadrature-artifact-{artifact}").as_bytes(),
                    ),
                }]
            })
            .collect::<Vec<_>>();
        let batches = plan_family_publication_batches(&artifacts, &TransportPolicy::default())
            .expect("863 small artifacts must share repository batches");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].parts.len(), 863);
        assert_eq!(batches[0].payload_bytes, 863_000_000);
    }

    #[test]
    fn family_batching_uses_payload_bytes_not_artifact_count() {
        let artifacts = (0..1_001)
            .map(|artifact| {
                vec![TransportPart {
                    sequence: 0,
                    repository_path: format!("objects/{artifact:04}.part"),
                    size_bytes: 1_000_000,
                    content_digest: ContentDigest::sha256(
                        format!("quadrature-artifact-{artifact}").as_bytes(),
                    ),
                }]
            })
            .collect::<Vec<_>>();
        let batches = plan_family_publication_batches(&artifacts, &TransportPolicy::default())
            .expect("the aggregate payload should cross one commit boundary");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].payload_bytes, 1_000_000_000);
        assert_eq!(batches[1].payload_bytes, 1_000_000);
    }

    #[test]
    fn dual_target_journals_resume_independently_and_are_idempotent() {
        let parts = simulated_parts(&[4, 4]);
        let batches = plan_publication_batches(&parts, &test_policy()).unwrap();
        let repositories = BTreeMap::from([
            (PublicationDestination::Private, "team/private".to_owned()),
            (PublicationDestination::Public, "team/public".to_owned()),
        ]);
        let permission_evidence = BTreeMap::from([
            (
                PublicationDestination::Private,
                crate::AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    "team/private",
                    crate::RepositoryPermission::Write,
                )
                .evidence()
                .clone(),
            ),
            (
                PublicationDestination::Public,
                crate::AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    "team/public",
                    crate::RepositoryPermission::Write,
                )
                .evidence()
                .clone(),
            ),
        ]);
        let shards = BTreeMap::from([
            (PublicationDestination::Private, "private-001".to_owned()),
            (PublicationDestination::Public, "public-001".to_owned()),
        ]);
        let heads = BTreeMap::from([
            (PublicationDestination::Private, "private-head".to_owned()),
            (PublicationDestination::Public, "public-head".to_owned()),
        ]);
        let build = || {
            PublicationTransactionJournal::new(
                ContentDigest::sha256(b"semantic"),
                BTreeMap::from([
                    (
                        PublicationDestination::Private,
                        ContentDigest::sha256(b"private-manifest"),
                    ),
                    (
                        PublicationDestination::Public,
                        ContentDigest::sha256(b"public-manifest"),
                    ),
                ]),
                ContentDigest::sha256(b"payload"),
                ContentDigest::sha256(b"policy"),
                PublicationTarget::Both,
                repositories.clone(),
                repositories.clone(),
                permission_evidence.clone(),
                shards.clone(),
                BTreeMap::from([
                    (PublicationDestination::Private, "main".to_owned()),
                    (PublicationDestination::Public, "main".to_owned()),
                ]),
                heads.clone(),
                &batches,
            )
            .unwrap()
        };
        let first = build();
        let mut second = build();
        assert_eq!(first.idempotency_key, second.idempotency_key);
        second
            .targets
            .get_mut(&PublicationDestination::Private)
            .unwrap()
            .start()
            .unwrap();
        second
            .targets
            .get_mut(&PublicationDestination::Public)
            .unwrap()
            .fail("sanitizer rejected public metadata");
        assert_eq!(
            second.targets[&PublicationDestination::Private].state,
            PublicationTargetState::Uploading
        );
        assert_eq!(
            second.targets[&PublicationDestination::Public].state,
            PublicationTargetState::Failed
        );
    }
}
