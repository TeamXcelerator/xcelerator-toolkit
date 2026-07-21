//! Deterministic staging of target-specific shard metadata and projections.

use crate::protocol::{canonical_digest, canonical_json_bytes, normalized_relative_path};
use crate::{
    plan_discoverability_commit, ArtifactAssuranceState, ArtifactDisposition, AttestationEnvelope,
    AttestationKind, CacheError, CanonicalArtifactManifest, CapacityAdmission, CapacityLedger,
    ContentDigest, PayloadBatchRecord, PublicSanitizerProfile, PublicationDestination,
    PublicationReceipt, PublicationTransactionJournal, RemoteDocument, ShardIndexEntry,
    ShardIndexPartition, TransportEncodingRecord, TransportPart, ValidatorEvidence,
    DEFAULT_CAPACITY_LEDGER_PATH, GITHUB_HARD_FILE_BOUNDARY_BYTES,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xc_core::{CancellationToken, ResourcePolicy};

static STAGING_TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationMetadataBundle {
    pub schema_version: u32,
    pub family: String,
    pub manifest: CanonicalArtifactManifest,
    pub encoding: TransportEncodingRecord,
    pub validation_attestations: Vec<AttestationEnvelope>,
    pub validator_evidence: Vec<ValidatorEvidence>,
    /// Target-specific metadata that was evaluated during authorization. A
    /// public target rechecks it together with every staged document.
    pub target_metadata: BTreeMap<String, Value>,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemotePublicationTransactionRecord {
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
    pub payload_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub policy_digest: ContentDigest,
    pub validation_attestation_digests: Vec<ContentDigest>,
    pub validator_evidence_digests: Vec<ContentDigest>,
}

impl RemotePublicationTransactionRecord {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        if self.schema_version == 0
            || self.transaction_id != self.idempotency_key.0
            || self.principal.trim().is_empty()
            || self.authorized_repository.trim().is_empty()
            || self.shard_id.trim().is_empty()
            || self.branch.trim().is_empty()
            || self.validation_attestation_digests.is_empty()
            || self.validator_evidence_digests.is_empty()
            || [
                &self.idempotency_key,
                &self.repository_permission_evidence_digest,
                &self.semantic_digest,
                &self.payload_digest,
                &self.manifest_digest,
                &self.transport_digest,
                &self.policy_digest,
            ]
            .into_iter()
            .chain(self.validation_attestation_digests.iter())
            .chain(self.validator_evidence_digests.iter())
            .any(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "remote publication transaction record is incomplete".to_owned(),
            ));
        }
        canonical_digest(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImmutableMetadataStageReport {
    pub manifest_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub validation_attestation_digests: Vec<ContentDigest>,
    pub transaction_record_digest: ContentDigest,
    pub files: Vec<TransportPart>,
    pub staged_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiscoverabilityStageReport {
    pub index_digest: ContentDigest,
    pub ledger_digest: ContentDigest,
    pub receipt_digest: ContentDigest,
    pub unique_payload_bytes_added: u64,
    pub metadata_bytes_added: u64,
    pub projected_history_bytes_added: u64,
    pub capacity: CapacityAdmission,
    pub files: Vec<TransportPart>,
    pub staged_bytes: u64,
}

#[derive(Serialize)]
struct CapacityReconciliationEvidence<'a> {
    schema_version: u32,
    transaction_id: &'a str,
    base_revision: &'a str,
    base_ledger_digest: &'a ContentDigest,
    index_digest: &'a ContentDigest,
    receipt_digest: &'a ContentDigest,
    unique_payload_bytes_added: u64,
    metadata_bytes_added: u64,
    projected_history_bytes_added: u64,
}

pub fn stage_payload_batch_record(
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    record: &PayloadBatchRecord,
) -> Result<TransportPart, CacheError> {
    record.validate()?;
    let mut staged_bytes = 0;
    stage_canonical_document(
        staging_root,
        &record.repository_path(),
        record,
        resources,
        cancellation,
        &mut staged_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn stage_immutable_publication_metadata(
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    public_sanitizer: Option<&PublicSanitizerProfile>,
) -> Result<ImmutableMetadataStageReport, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    validate_bundle(journal, destination, bundle)?;
    validate_public_documents(journal, destination, bundle, public_sanitizer)?;
    let target = &journal.targets[&destination];
    if target.metadata_commit.is_none()
        && target.state != crate::PublicationTargetState::BatchVerified
    {
        return Err(CacheError::InvalidTransition(
            "immutable metadata staging requires verified payload batches".to_owned(),
        ));
    }
    let manifest_digest = bundle.manifest.digest()?;
    let transport_digest = bundle.encoding.digest()?;
    let validation_attestation_digests = bundle
        .validation_attestations
        .iter()
        .map(AttestationEnvelope::digest)
        .collect::<Result<Vec<_>, _>>()?;
    let validator_evidence_digests = bundle
        .validator_evidence
        .iter()
        .map(|evidence| evidence.evidence_digest.clone())
        .collect::<Vec<_>>();
    let transaction_record = RemotePublicationTransactionRecord {
        schema_version: 1,
        transaction_id: journal.transaction_id.clone(),
        idempotency_key: journal.idempotency_key.clone(),
        destination,
        principal: target.permission_evidence.principal.clone(),
        authorized_repository: target.authorized_repository.clone(),
        repository_permission_evidence_digest: target.permission_evidence.evidence_digest.clone(),
        shard_id: target.shard_id.clone(),
        branch: target.branch.clone(),
        semantic_digest: journal.semantic_digest.clone(),
        payload_digest: journal.payload_digest.clone(),
        manifest_digest: manifest_digest.clone(),
        transport_digest: transport_digest.clone(),
        policy_digest: journal.policy_digest.clone(),
        validation_attestation_digests: validation_attestation_digests.clone(),
        validator_evidence_digests,
    };
    let transaction_record_digest = transaction_record.digest()?;
    let semantic_prefix = &journal.semantic_digest.0[..2];
    let payload_prefix = &journal.payload_digest.0[..2];
    let mut staged_bytes = 0u64;
    let mut files = Vec::new();
    files.push(stage_canonical_document(
        staging_root,
        &format!("encodings/{payload_prefix}/{}.json", transport_digest.0),
        &bundle.encoding,
        resources,
        cancellation,
        &mut staged_bytes,
    )?);
    files.push(stage_canonical_document(
        staging_root,
        &format!("manifests/{semantic_prefix}/{}.json", manifest_digest.0),
        &bundle.manifest,
        resources,
        cancellation,
        &mut staged_bytes,
    )?);
    for (attestation, digest) in bundle
        .validation_attestations
        .iter()
        .zip(&validation_attestation_digests)
    {
        files.push(stage_canonical_document(
            staging_root,
            &format!("attestations/validation/{}.json", digest.0),
            attestation,
            resources,
            cancellation,
            &mut staged_bytes,
        )?);
    }
    files.push(stage_canonical_document(
        staging_root,
        &format!(
            "transactions/{}/{}/plan.json",
            journal.transaction_id,
            match destination {
                PublicationDestination::Private => "private",
                PublicationDestination::Public => "public",
            }
        ),
        &transaction_record,
        resources,
        cancellation,
        &mut staged_bytes,
    )?);
    canonicalize_file_sequence(&mut files);
    match &journal.targets[&destination].metadata_commit {
        Some(existing) if existing.files == files => {}
        Some(_) => {
            return Err(CacheError::InvalidTransition(
                "a different immutable metadata plan already exists".to_owned(),
            ));
        }
        None => journal
            .targets
            .get_mut(&destination)
            .expect("bundle validation checked target existence")
            .plan_metadata_commit(files.clone())?,
    }
    Ok(ImmutableMetadataStageReport {
        manifest_digest,
        transport_digest,
        validation_attestation_digests,
        transaction_record_digest,
        files,
        staged_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn stage_publication_discoverability(
    staging_root: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    index_document: &RemoteDocument<ShardIndexPartition>,
    ledger_document: &RemoteDocument<CapacityLedger>,
    verified_at_unix_seconds: u64,
    projected_history_bytes: u64,
    replace_existing_semantic: bool,
) -> Result<DiscoverabilityStageReport, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    validate_bundle(journal, destination, bundle)?;
    let target = &journal.targets[&destination];
    let metadata = target.metadata_commit.as_ref().ok_or_else(|| {
        CacheError::InvalidTransition("immutable publication metadata was not planned".to_owned())
    })?;
    if target.state != crate::PublicationTargetState::RemoteVerified
        || metadata.state != crate::PublicationCommitState::RemoteVerified
    {
        return Err(CacheError::InvalidTransition(
            "discoverability staging requires remotely verified immutable metadata".to_owned(),
        ));
    }
    let semantic_prefix = &journal.semantic_digest.0[..2];
    let index_path = format!("indexes/{}/{semantic_prefix}.json", bundle.family);
    for source in [&index_document.source, &ledger_document.source] {
        if source.revision != target.expected_head {
            return Err(CacheError::InvalidManifest(
                "shard projections were not read from the target's current revision".to_owned(),
            ));
        }
    }
    if index_document.source.repository_path != index_path
        || ledger_document.source.repository_path != DEFAULT_CAPACITY_LEDGER_PATH
    {
        return Err(CacheError::InvalidManifest(
            "shard index or capacity ledger was read from an unexpected path".to_owned(),
        ));
    }
    index_document.value.validate()?;
    ledger_document.value.validate()?;
    if index_document.value.family != bundle.family
        || index_document.value.semantic_prefix != semantic_prefix
        || ledger_document.value.shard_id != target.shard_id
    {
        return Err(CacheError::InvalidManifest(
            "shard projection identity does not match the publication target".to_owned(),
        ));
    }
    let manifest_digest = bundle.manifest.digest()?;
    let transport_digest = bundle.encoding.digest()?;
    let mut index_entries = index_document.value.entries.clone();
    index_document.value.ensure_monotonic_producer(
        &journal.semantic_digest,
        &bundle.manifest.producer_toolkit_version,
    )?;
    if replace_existing_semantic {
        index_entries.retain(|entry| entry.semantic_digest != journal.semantic_digest);
    } else {
        index_entries.retain(|entry| entry.manifest_digest != manifest_digest);
    }
    index_entries.push(ShardIndexEntry {
        semantic_digest: journal.semantic_digest.clone(),
        canonical_payload_digest: journal.payload_digest.clone(),
        manifest_digest: manifest_digest.clone(),
        achieved_assurance: bundle.achieved_assurance,
        disposition: bundle.disposition,
        producer_toolkit_version: bundle.manifest.producer_toolkit_version.clone(),
        minimum_reader_version: bundle.manifest.minimum_reader_version.clone(),
        transport_digests: vec![transport_digest.clone()],
        publication_transaction_id: journal.transaction_id.clone(),
    });
    let index = ShardIndexPartition::rebuild(
        bundle.family.clone(),
        semantic_prefix.to_owned(),
        index_entries,
    )?;
    let index_bytes = canonical_json_bytes(&index)?;
    let index_digest = ContentDigest::sha256(&index_bytes);
    let receipt = PublicationReceipt::from_verified_metadata(
        journal,
        destination,
        transport_digest,
        BTreeMap::from([(index_path.clone(), index_digest.clone())]),
        verified_at_unix_seconds,
    )?;
    let receipt_bytes = receipt.canonical_bytes()?;
    let receipt_digest = receipt.digest()?;
    let unique_payload_bytes_added = newly_committed_payload_bytes(target)?;
    let immutable_metadata_bytes_added = newly_committed_metadata_bytes(metadata)?
        .saturating_add(newly_committed_batch_record_bytes(target)?);
    let (ledger, ledger_bytes, metadata_bytes_added, capacity) = build_updated_ledger(
        journal,
        target,
        &ledger_document.value,
        &ledger_document.source.content_digest,
        &index_digest,
        index_bytes.len() as u64,
        &receipt_digest,
        receipt_bytes.len() as u64,
        unique_payload_bytes_added,
        immutable_metadata_bytes_added,
        projected_history_bytes,
    )?;
    let ledger_digest = ContentDigest::sha256(&ledger_bytes);
    ledger.validate()?;
    let mut staged_bytes = 0u64;
    let mut files = vec![
        stage_bytes(
            staging_root,
            &index_path,
            &index_bytes,
            resources,
            cancellation,
            &mut staged_bytes,
        )?,
        stage_bytes(
            staging_root,
            DEFAULT_CAPACITY_LEDGER_PATH,
            &ledger_bytes,
            resources,
            cancellation,
            &mut staged_bytes,
        )?,
        stage_bytes(
            staging_root,
            &receipt.repository_path(),
            &receipt_bytes,
            resources,
            cancellation,
            &mut staged_bytes,
        )?,
    ];
    canonicalize_file_sequence(&mut files);
    match &journal.targets[&destination].discoverability_commit {
        Some(existing)
            if existing.files == files
                && existing.receipt_digest.as_ref() == Some(&receipt_digest) =>
        {
            receipt.validate_for_transaction(journal, destination)?;
        }
        Some(_) => {
            return Err(CacheError::InvalidTransition(
                "a different discoverability plan already exists".to_owned(),
            ));
        }
        None => {
            let planned_receipt_digest =
                plan_discoverability_commit(journal, destination, files.clone(), &receipt)?;
            debug_assert_eq!(planned_receipt_digest, receipt_digest);
        }
    }
    Ok(DiscoverabilityStageReport {
        index_digest,
        ledger_digest,
        receipt_digest,
        unique_payload_bytes_added,
        metadata_bytes_added,
        projected_history_bytes_added: projected_history_bytes,
        capacity,
        files,
        staged_bytes,
    })
}

fn validate_bundle(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
) -> Result<(), CacheError> {
    if bundle.schema_version == 0
        || !safe_family(&bundle.family)
        || bundle.family != bundle.manifest.artifact_family
    {
        return Err(CacheError::InvalidManifest(
            "publication metadata family is invalid or disagrees with the manifest".to_owned(),
        ));
    }
    bundle.manifest.validate()?;
    bundle.encoding.validate()?;
    let target = journal.targets.get(&destination).ok_or_else(|| {
        CacheError::InvalidTransition(format!(
            "transaction has no {destination:?} metadata target"
        ))
    })?;
    let manifest_digest = bundle.manifest.digest()?;
    let transport_digest = bundle.encoding.digest()?;
    if bundle.manifest.semantic_digest != journal.semantic_digest
        || bundle.manifest.payload_digest != journal.payload_digest
        || bundle.encoding.canonical_payload_digest != journal.payload_digest
        || journal.target_manifest_digests[&destination] != manifest_digest
        || !bundle
            .manifest
            .transport_digests
            .contains(&transport_digest)
        || target.permission_evidence.principal.trim().is_empty()
    {
        return Err(CacheError::InvalidManifest(
            "publication metadata identities do not match the transaction".to_owned(),
        ));
    }
    if bundle.validation_attestations.is_empty() {
        return Err(CacheError::InvalidManifest(
            "publication requires validation or certification attestation".to_owned(),
        ));
    }
    let evidence_assurance = bundle
        .validator_evidence
        .iter()
        .filter(|evidence| evidence.passed && evidence.evidence_digest.validate())
        .filter_map(|evidence| evidence.establishes_assurance)
        .max()
        .unwrap_or(ArtifactAssuranceState::Unchecked);
    if bundle.disposition != ArtifactDisposition::Active
        || bundle.validator_evidence.is_empty()
        || bundle.validator_evidence.iter().any(|evidence| {
            evidence.validator_id.trim().is_empty()
                || !evidence.passed
                || !evidence.evidence_digest.validate()
        })
        || bundle.achieved_assurance > evidence_assurance
    {
        return Err(CacheError::InvalidManifest(
            "publication state or achieved assurance is not supported by validator evidence"
                .to_owned(),
        ));
    }
    for attestation in &bundle.validation_attestations {
        attestation.digest()?;
        if !matches!(
            attestation.kind,
            AttestationKind::Validation | AttestationKind::Certification
        ) || (attestation.subject_digest != manifest_digest
            && attestation.subject_digest != journal.payload_digest)
            || attestation.producer_toolkit_version != bundle.manifest.producer_toolkit_version
        {
            return Err(CacheError::InvalidManifest(
                "publication attestation does not validate this manifest or payload".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_public_documents(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    public_sanitizer: Option<&PublicSanitizerProfile>,
) -> Result<(), CacheError> {
    if destination == PublicationDestination::Private {
        return Ok(());
    }
    let sanitizer = public_sanitizer.ok_or_else(|| {
        CacheError::PermissionDenied(
            "public publication metadata requires an allowlist sanitizer".to_owned(),
        )
    })?;
    let mut material = bundle.target_metadata.clone();
    material.insert(
        "manifest".to_owned(),
        serde_json::to_value(&bundle.manifest)?,
    );
    material.insert(
        "encoding".to_owned(),
        serde_json::to_value(&bundle.encoding)?,
    );
    material.insert(
        "attestations".to_owned(),
        serde_json::to_value(&bundle.validation_attestations)?,
    );
    material.insert(
        "validator_evidence".to_owned(),
        serde_json::to_value(&bundle.validator_evidence)?,
    );
    let target = &journal.targets[&destination];
    material.insert(
        "publication_principal".to_owned(),
        Value::String(target.permission_evidence.principal.clone()),
    );
    material.insert(
        "publication_repository".to_owned(),
        Value::String(target.authorized_repository.clone()),
    );
    material.insert(
        "publication_shard".to_owned(),
        Value::String(target.shard_id.clone()),
    );
    material.insert(
        "publication_branch".to_owned(),
        Value::String(target.branch.clone()),
    );
    let repository = &target.authorized_repository;
    let report = sanitizer.inspect(&material, repository);
    if !report.accepted {
        return Err(CacheError::PermissionDenied(format!(
            "public publication documents failed sanitization: {}",
            report.reasons.join("; ")
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_updated_ledger(
    journal: &PublicationTransactionJournal,
    target: &crate::TargetPublicationJournal,
    base: &CapacityLedger,
    base_digest: &ContentDigest,
    index_digest: &ContentDigest,
    index_bytes: u64,
    receipt_digest: &ContentDigest,
    receipt_bytes: u64,
    unique_payload_bytes_added: u64,
    immutable_metadata_bytes_added: u64,
    projected_history_bytes: u64,
) -> Result<(CapacityLedger, Vec<u8>, u64, CapacityAdmission), CacheError> {
    let mut ledger_size = 0u64;
    for _ in 0..16 {
        let metadata_bytes_added = immutable_metadata_bytes_added
            .saturating_add(index_bytes)
            .saturating_add(receipt_bytes)
            .saturating_add(ledger_size);
        let evidence = CapacityReconciliationEvidence {
            schema_version: 1,
            transaction_id: &journal.transaction_id,
            base_revision: &target.expected_head,
            base_ledger_digest: base_digest,
            index_digest,
            receipt_digest,
            unique_payload_bytes_added,
            metadata_bytes_added,
            projected_history_bytes_added: projected_history_bytes,
        };
        let mut ledger = base.clone();
        ledger.first_seen_immutable_payload_bytes = ledger
            .first_seen_immutable_payload_bytes
            .saturating_add(unique_payload_bytes_added);
        ledger.manifest_index_receipt_bytes = ledger
            .manifest_index_receipt_bytes
            .saturating_add(metadata_bytes_added);
        ledger.estimated_history_bytes = ledger
            .estimated_history_bytes
            .saturating_add(projected_history_bytes);
        ledger.last_reconciled_commit = target.expected_head.clone();
        ledger.reconciliation_digest = canonical_digest(&evidence)?;
        ledger.validate()?;
        let bytes = canonical_json_bytes(&ledger)?;
        let actual_size = bytes.len() as u64;
        if actual_size == ledger_size {
            let capacity = base.assess_addition(
                unique_payload_bytes_added,
                metadata_bytes_added,
                projected_history_bytes,
            )?;
            if !capacity.accepted {
                return Err(CacheError::NoWritableShard(format!(
                    "publication projections exceed capacity: {}",
                    capacity.reasons.join("; ")
                )));
            }
            return Ok((ledger, bytes, metadata_bytes_added, capacity));
        }
        ledger_size = actual_size;
    }
    Err(CacheError::InvalidManifest(
        "capacity ledger byte accounting did not converge".to_owned(),
    ))
}

fn newly_committed_payload_bytes(
    target: &crate::TargetPublicationJournal,
) -> Result<u64, CacheError> {
    let sizes: BTreeMap<_, _> = target
        .batches
        .iter()
        .flat_map(|batch| &batch.plan.parts)
        .map(|part| (part.content_digest.clone(), part.size_bytes))
        .collect();
    target
        .batches
        .iter()
        .flat_map(|batch| &batch.newly_committed_digests)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .try_fold(0u64, |total, digest| {
            sizes
                .get(digest)
                .copied()
                .map(|size| total.saturating_add(size))
                .ok_or_else(|| {
                    CacheError::InvalidManifest(
                        "new payload accounting names an unknown digest".to_owned(),
                    )
                })
        })
}

fn newly_committed_metadata_bytes(
    metadata: &crate::PublicationCommitJournal,
) -> Result<u64, CacheError> {
    let sizes: BTreeMap<_, _> = metadata
        .files
        .iter()
        .map(|file| (file.content_digest.clone(), file.size_bytes))
        .collect();
    metadata
        .newly_committed_digests
        .iter()
        .try_fold(0u64, |total, digest| {
            sizes
                .get(digest)
                .copied()
                .map(|size| total.saturating_add(size))
                .ok_or_else(|| {
                    CacheError::InvalidManifest(
                        "new metadata accounting names an unknown digest".to_owned(),
                    )
                })
        })
}

fn newly_committed_batch_record_bytes(
    target: &crate::TargetPublicationJournal,
) -> Result<u64, CacheError> {
    target
        .batches
        .iter()
        .filter_map(|batch| batch.record_commit.as_ref())
        .try_fold(0u64, |total, record| {
            newly_committed_metadata_bytes(record).map(|bytes| total.saturating_add(bytes))
        })
}

fn stage_canonical_document<T: Serialize>(
    root: &Path,
    repository_path: &str,
    value: &T,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    staged_bytes: &mut u64,
) -> Result<TransportPart, CacheError> {
    let bytes = canonical_json_bytes(value)?;
    stage_bytes(
        root,
        repository_path,
        &bytes,
        resources,
        cancellation,
        staged_bytes,
    )
}

fn stage_bytes(
    root: &Path,
    repository_path: &str,
    bytes: &[u8],
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    staged_bytes: &mut u64,
) -> Result<TransportPart, CacheError> {
    if !normalized_relative_path(repository_path)
        || bytes.is_empty()
        || bytes.len() as u64 >= GITHUB_HARD_FILE_BOUNDARY_BYTES
    {
        return Err(CacheError::InvalidManifest(format!(
            "staged publication path {repository_path:?} or size is invalid"
        )));
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let next_total = staged_bytes.saturating_add(bytes.len() as u64);
    if resources
        .maximum_memory_bytes
        .is_some_and(|maximum| bytes.len() as u64 > maximum)
        || resources
            .maximum_temporary_disk_bytes
            .is_some_and(|maximum| bytes.len() as u64 > maximum)
        || resources
            .maximum_permanent_disk_bytes
            .is_some_and(|maximum| next_total > maximum)
    {
        return Err(CacheError::ResourceLimit(format!(
            "staging {repository_path:?} exceeds the resolved resource policy"
        )));
    }
    let destination = resolve_staging_path(root, repository_path)?;
    let digest = ContentDigest::sha256(bytes);
    if destination.exists() {
        verify_existing_file(&destination, bytes.len() as u64, &digest, cancellation)?;
    } else {
        let parent = destination.parent().unwrap_or(root);
        let name = destination
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("metadata"))
            .to_string_lossy();
        let sequence = STAGING_TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{name}.xc-stage-{}-{sequence}",
            std::process::id()
        ));
        let result = (|| {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            match fs::hard_link(&temporary, &destination) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    verify_existing_file(&destination, bytes.len() as u64, &digest, cancellation)
                }
                Err(error) => Err(error.into()),
            }
        })();
        let _ = fs::remove_file(temporary);
        result?;
    }
    *staged_bytes = next_total;
    Ok(TransportPart {
        sequence: 0,
        repository_path: repository_path.to_owned(),
        size_bytes: bytes.len() as u64,
        content_digest: digest,
    })
}

pub(crate) fn stage_publication_bytes(
    root: &Path,
    repository_path: &str,
    bytes: &[u8],
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    staged_bytes: &mut u64,
) -> Result<TransportPart, CacheError> {
    stage_bytes(
        root,
        repository_path,
        bytes,
        resources,
        cancellation,
        staged_bytes,
    )
}

fn resolve_staging_path(root: &Path, repository_path: &str) -> Result<PathBuf, CacheError> {
    fs::create_dir_all(root)?;
    if fs::symlink_metadata(root)?.file_type().is_symlink() {
        return Err(CacheError::InvalidManifest(
            "publication staging root may not be a symlink".to_owned(),
        ));
    }
    let mut path = root.to_owned();
    let components: Vec<_> = repository_path.split('/').collect();
    for component in &components[..components.len().saturating_sub(1)] {
        path.push(component);
        if path.exists() {
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(CacheError::InvalidManifest(format!(
                    "publication staging parent {} is unsafe",
                    path.display()
                )));
            }
        } else {
            fs::create_dir(&path)?;
        }
    }
    path.push(components.last().copied().unwrap_or("metadata"));
    Ok(path)
}

fn verify_existing_file(
    path: &Path,
    expected_size: u64,
    expected_digest: &ContentDigest,
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != expected_size {
        return Err(CacheError::DigestMismatch {
            expected: format!("{expected_digest} ({expected_size} bytes)"),
            actual: format!("unsafe or differently sized file {}", path.display()),
        });
    }
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 64 * 1024];
    let mut hasher = Sha256::new();
    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = ContentDigest(format!("{:x}", hasher.finalize()));
    if &actual != expected_digest {
        return Err(CacheError::DigestMismatch {
            expected: expected_digest.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn canonicalize_file_sequence(files: &mut [TransportPart]) {
    files.sort_by(|left, right| left.repository_path.cmp(&right.repository_path));
    for (sequence, file) in files.iter_mut().enumerate() {
        file.sequence = sequence as u64;
    }
}

fn safe_family(family: &str) -> bool {
    !family.is_empty()
        && family
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plan_publication_batches, AuthenticatedGitHubSession, CanonicalPayloadEnvelope,
        LogicalPayloadItem, PublicationBatchState, PublicationCommitState, RemoteReadReport,
        RepositoryPermission, SemanticKeyEnvelope, TransportPolicy,
        GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use xc_core::{AssuranceLevel, PublicationTarget};

    fn fixture(
        destination: PublicationDestination,
    ) -> (PublicationTransactionJournal, PublicationMetadataBundle) {
        let payload_bytes = b"payload-part";
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "staging_fixture".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: json!({"case": "staging"}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "bytes".to_owned(),
            dimensions: vec![payload_bytes.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "fixture.bin".to_owned(),
                content_digest: ContentDigest::sha256(payload_bytes),
                size_bytes: payload_bytes.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let part = TransportPart {
            sequence: 0,
            repository_path: format!(
                "objects/sha256/aa/{}.part",
                ContentDigest::sha256(payload_bytes)
            ),
            size_bytes: payload_bytes.len() as u64,
            content_digest: ContentDigest::sha256(payload_bytes),
        };
        let encoding = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: payload_digest.clone(),
            encoder_profile: "fixture-v1".to_owned(),
            package_size_bytes: part.size_bytes,
            package_digest: part.content_digest.clone(),
            ordered_parts: vec![part.clone()],
            reconstruction: "concatenate".to_owned(),
        };
        let transport_digest = encoding.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm".to_owned(),
            semantic_key,
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![transport_digest],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"configuration"),
            producer_toolkit_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let attestation = AttestationEnvelope {
            schema_version: 1,
            kind: AttestationKind::Validation,
            subject_digest: manifest_digest.clone(),
            actor: "fixture-validator".to_owned(),
            policy_digest: ContentDigest::sha256(b"policy"),
            execution_fingerprint_digest: ContentDigest::sha256(b"fingerprint"),
            producer_toolkit_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            dependency_versions: BTreeMap::from([("xc-cache".to_owned(), "0.13.0".to_owned())]),
            source_revision: "toolkit-revision".to_owned(),
            event_unix_seconds: 1,
            location: None,
            evidence_digests: vec![ContentDigest::sha256(b"evidence")],
        };
        let repository = match destination {
            PublicationDestination::Private => "team/private",
            PublicationDestination::Public => "team/public",
        };
        let session = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            repository,
            RepositoryPermission::Write,
        );
        let batches = plan_publication_batches(
            std::slice::from_ref(&part),
            &TransportPolicy {
                maximum_file_bytes_exclusive: 100,
                split_part_bytes: 10,
                maximum_batch_payload_bytes: 100,
                maximum_pending_batches: 1,
            },
        )
        .unwrap();
        let target = match destination {
            PublicationDestination::Private => PublicationTarget::Private,
            PublicationDestination::Public => PublicationTarget::Public,
        };
        let mut journal = PublicationTransactionJournal::new(
            semantic_digest,
            BTreeMap::from([(destination, manifest_digest)]),
            payload_digest,
            ContentDigest::sha256(b"policy"),
            target,
            BTreeMap::from([(destination, repository.to_owned())]),
            BTreeMap::from([(destination, repository.to_owned())]),
            BTreeMap::from([(destination, session.evidence().clone())]),
            BTreeMap::from([(destination, "shard-001".to_owned())]),
            BTreeMap::from([(destination, "main".to_owned())]),
            BTreeMap::from([(destination, "head-0".to_owned())]),
            &batches,
        )
        .unwrap();
        crate::attach_test_owner_audit_evidence(&mut journal, destination);
        {
            let target_journal = journal.targets.get_mut(&destination).unwrap();
            target_journal.start().unwrap();
            target_journal
                .mark_batch_committed(0, "head-0", "head-1", vec![part.content_digest.clone()])
                .unwrap();
        }
        let batch_record = crate::build_payload_batch_record(&journal, destination, 0).unwrap();
        let record_file = TransportPart {
            sequence: 0,
            repository_path: batch_record.repository_path(),
            size_bytes: serde_json::to_vec(&batch_record).unwrap().len() as u64,
            content_digest: batch_record.digest().unwrap(),
        };
        let target_journal = journal.targets.get_mut(&destination).unwrap();
        target_journal
            .plan_verified_batch_record(0, "head-1", record_file)
            .unwrap();
        target_journal
            .mark_batch_record_reused(0, "head-1")
            .unwrap();
        (
            journal,
            PublicationMetadataBundle {
                schema_version: 1,
                family: "ccm".to_owned(),
                manifest,
                encoding,
                validation_attestations: vec![attestation],
                validator_evidence: vec![ValidatorEvidence {
                    validator_id: "fixture-validator".to_owned(),
                    passed: true,
                    evidence_digest: ContentDigest::sha256(b"evidence"),
                    establishes_assurance: Some(ArtifactAssuranceState::Computed),
                }],
                target_metadata: BTreeMap::new(),
                achieved_assurance: ArtifactAssuranceState::Computed,
                disposition: ArtifactDisposition::Active,
            },
        )
    }

    fn temporary_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "xc-cache-publication-staging-{name}-{}",
            std::process::id()
        ))
    }

    fn ledger() -> CapacityLedger {
        CapacityLedger {
            schema_version: 1,
            shard_id: "shard-001".to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 0,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 0,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: "head-0".to_owned(),
            reconciliation_digest: ContentDigest::sha256(b"base-reconciliation"),
        }
    }

    #[test]
    fn stages_exact_metadata_index_ledger_and_receipt_documents() {
        let (mut journal, bundle) = fixture(PublicationDestination::Private);
        let root = temporary_root("complete");
        let _ = fs::remove_dir_all(&root);
        let metadata = stage_immutable_publication_metadata(
            &root,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &mut journal,
            PublicationDestination::Private,
            &bundle,
            None,
        )
        .unwrap();
        {
            let target = journal
                .targets
                .get_mut(&PublicationDestination::Private)
                .unwrap();
            let digests = metadata
                .files
                .iter()
                .map(|file| file.content_digest.clone())
                .collect();
            target
                .metadata_commit
                .as_mut()
                .unwrap()
                .mark_committed("head-1", "head-2", digests)
                .unwrap();
            target
                .metadata_commit
                .as_mut()
                .unwrap()
                .mark_remote_verified("head-2")
                .unwrap();
            target.expected_head = "head-2".to_owned();
            target.mark_remote_verified().unwrap();
        }
        let prefix = journal.semantic_digest.0[..2].to_owned();
        let prior_entry = ShardIndexEntry {
            semantic_digest: journal.semantic_digest.clone(),
            canonical_payload_digest: ContentDigest::sha256(b"prior-payload"),
            manifest_digest: ContentDigest::sha256(b"prior-manifest"),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            transport_digests: vec![ContentDigest::sha256(b"prior-transport")],
            publication_transaction_id: ContentDigest::sha256(b"prior-transaction").0,
        };
        let prior_index =
            ShardIndexPartition::rebuild("ccm", prefix.clone(), vec![prior_entry]).unwrap();
        let index_bytes = canonical_json_bytes(&prior_index).unwrap();
        let base_ledger = ledger();
        let ledger_bytes = canonical_json_bytes(&base_ledger).unwrap();
        let index_document = RemoteDocument {
            source: RemoteReadReport {
                repository_path: format!("indexes/ccm/{prefix}.json"),
                revision: "head-2".to_owned(),
                size_bytes: index_bytes.len() as u64,
                content_digest: ContentDigest::sha256(&index_bytes),
            },
            value: prior_index,
        };
        let ledger_document = RemoteDocument {
            source: RemoteReadReport {
                repository_path: DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                revision: "head-2".to_owned(),
                size_bytes: ledger_bytes.len() as u64,
                content_digest: ContentDigest::sha256(&ledger_bytes),
            },
            value: base_ledger,
        };
        let report = stage_publication_discoverability(
            &root,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &mut journal,
            PublicationDestination::Private,
            &bundle,
            &index_document,
            &ledger_document,
            123,
            500,
            true,
        )
        .unwrap();
        let payload_size = bundle.encoding.ordered_parts[0].size_bytes;
        assert_eq!(report.unique_payload_bytes_added, payload_size);
        assert!(report.capacity.accepted);
        let staged_index: ShardIndexPartition = serde_json::from_slice(
            &fs::read(root.join(format!("indexes/ccm/{prefix}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(staged_index.entries.len(), 1);
        assert_eq!(
            staged_index.entries[0].manifest_digest,
            bundle.manifest.digest().unwrap()
        );
        let staged_ledger: CapacityLedger =
            serde_json::from_slice(&fs::read(root.join(DEFAULT_CAPACITY_LEDGER_PATH)).unwrap())
                .unwrap();
        assert_eq!(
            staged_ledger.first_seen_immutable_payload_bytes,
            payload_size
        );
        assert_eq!(
            staged_ledger.manifest_index_receipt_bytes,
            report.metadata_bytes_added
        );
        assert_eq!(staged_ledger.estimated_history_bytes, 500);
        let target = &journal.targets[&PublicationDestination::Private];
        assert_eq!(
            target.batches[0].state,
            PublicationBatchState::RemoteVerified
        );
        assert_eq!(
            target.metadata_commit.as_ref().unwrap().state,
            PublicationCommitState::RemoteVerified
        );
        assert!(target.discoverability_commit.is_some());
        assert!(root
            .join("transactions")
            .join(&journal.transaction_id)
            .join("private")
            .join("receipt.json")
            .is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn older_toolkit_cannot_publish_after_a_newer_active_artifact() {
        let (mut journal, bundle) = fixture(PublicationDestination::Private);
        let root = temporary_root("producer-downgrade");
        let _ = fs::remove_dir_all(&root);
        let metadata = stage_immutable_publication_metadata(
            &root,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &mut journal,
            PublicationDestination::Private,
            &bundle,
            None,
        )
        .unwrap();
        {
            let target = journal
                .targets
                .get_mut(&PublicationDestination::Private)
                .unwrap();
            let digests = metadata
                .files
                .iter()
                .map(|file| file.content_digest.clone())
                .collect();
            target
                .metadata_commit
                .as_mut()
                .unwrap()
                .mark_committed("head-1", "head-2", digests)
                .unwrap();
            target
                .metadata_commit
                .as_mut()
                .unwrap()
                .mark_remote_verified("head-2")
                .unwrap();
            target.expected_head = "head-2".to_owned();
            target.mark_remote_verified().unwrap();
        }
        let prefix = journal.semantic_digest.0[..2].to_owned();
        let newer_entry = ShardIndexEntry {
            semantic_digest: journal.semantic_digest.clone(),
            canonical_payload_digest: ContentDigest::sha256(b"newer-payload"),
            manifest_digest: ContentDigest::sha256(b"newer-manifest"),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: crate::ToolkitVersion::parse("0.14.0").unwrap(),
            minimum_reader_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            transport_digests: vec![ContentDigest::sha256(b"newer-transport")],
            publication_transaction_id: ContentDigest::sha256(b"newer-transaction").0,
        };
        let index = ShardIndexPartition::rebuild("ccm", prefix.clone(), vec![newer_entry]).unwrap();
        let index_bytes = canonical_json_bytes(&index).unwrap();
        let base_ledger = ledger();
        let ledger_bytes = canonical_json_bytes(&base_ledger).unwrap();
        let error = stage_publication_discoverability(
            &root,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &mut journal,
            PublicationDestination::Private,
            &bundle,
            &RemoteDocument {
                source: RemoteReadReport {
                    repository_path: format!("indexes/ccm/{prefix}.json"),
                    revision: "head-2".to_owned(),
                    size_bytes: index_bytes.len() as u64,
                    content_digest: ContentDigest::sha256(&index_bytes),
                },
                value: index,
            },
            &RemoteDocument {
                source: RemoteReadReport {
                    repository_path: DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                    revision: "head-2".to_owned(),
                    size_bytes: ledger_bytes.len() as u64,
                    content_digest: ContentDigest::sha256(&ledger_bytes),
                },
                value: base_ledger,
            },
            123,
            0,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("publication downgrade rejected"));
        assert!(!root.join(format!("indexes/ccm/{prefix}.json")).is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn public_staging_fails_closed_without_document_sanitizer() {
        let (mut journal, bundle) = fixture(PublicationDestination::Public);
        let root = temporary_root("public-sanitizer");
        let _ = fs::remove_dir_all(&root);
        let error = stage_immutable_publication_metadata(
            &root,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            &mut journal,
            PublicationDestination::Public,
            &bundle,
            None,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::PermissionDenied(_)));
        assert!(!root.exists());
    }
}
