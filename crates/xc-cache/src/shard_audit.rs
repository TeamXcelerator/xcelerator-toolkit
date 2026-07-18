//! Bounded, read-only remote shard inventory and projection reconciliation.

use crate::{
    CacheError, CanonicalArtifactManifest, CapacityLedger, ContentDigest, PayloadBatchRecord,
    PublicationReceipt, RemoteDocument, RemoteGitStore, RemotePathListReport, RemoteShardReader,
    ShardIndexEntry, ShardIndexPartition, TransportEncodingRecord, DEFAULT_CAPACITY_LEDGER_PATH,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use xc_core::CancellationToken;

const AUDIT_PREFIXES: [&str; 8] = [
    "indexes",
    "manifests",
    "encodings",
    "objects",
    "transactions",
    "attestations",
    "revocations",
    "ledger",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardAuditPolicy {
    pub maximum_paths_per_prefix: u64,
    pub maximum_path_bytes_per_prefix: u64,
    pub maximum_document_bytes: u64,
    pub maximum_total_metadata_bytes: u64,
}

impl ShardAuditPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if [
            self.maximum_paths_per_prefix,
            self.maximum_path_bytes_per_prefix,
            self.maximum_document_bytes,
            self.maximum_total_metadata_bytes,
        ]
        .contains(&0)
        {
            return Err(CacheError::ResourceLimit(
                "shard audit bounds must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardAuditSeverity {
    Warning,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardAuditIssueKind {
    InvalidIndex,
    MissingManifest,
    InvalidManifest,
    MissingEncoding,
    InvalidEncoding,
    MissingReceipt,
    InvalidReceipt,
    MissingMetadata,
    MissingObject,
    ProjectionDrift,
    LedgerDrift,
    UnreferencedManifest,
    UnreferencedEncoding,
    UnreferencedReceipt,
    UnreferencedObject,
    IncompleteTransaction,
    InvalidPayloadBatchRecord,
    PayloadHistoryGap,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardAuditIssue {
    pub severity: ShardAuditSeverity,
    pub kind: ShardAuditIssueKind,
    pub repository_path: Option<String>,
    pub identity_digest: Option<ContentDigest>,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardCapacityAudit {
    pub ledger_accounted_bytes: Option<u64>,
    pub ledger_first_seen_immutable_payload_bytes: Option<u64>,
    pub referenced_unique_transport_bytes: u64,
    pub ledger_covers_referenced_transport: bool,
    pub ledger_remaining_capacity_bytes: Option<u64>,
    pub durable_batch_record_count: u64,
    pub durable_recorded_new_payload_bytes: u64,
    pub durable_record_coverage_complete: bool,
    pub ledger_matches_durable_payload_history: bool,
    pub exact_history_rebuild_available: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteShardAuditReport {
    pub schema_version: u32,
    pub repository: String,
    pub branch: String,
    pub revision: String,
    pub shard_id: String,
    pub listed_path_count: u64,
    pub listed_path_bytes: u64,
    pub verified_metadata_bytes_read: u64,
    pub index_partition_count: u64,
    pub index_entry_count: u64,
    pub complete_artifact_count: u64,
    pub logical_payload_bytes: u64,
    pub unique_transport_object_count: u64,
    pub unique_transport_object_bytes: u64,
    pub unreferenced_manifest_count: u64,
    pub unreferenced_encoding_count: u64,
    pub unreferenced_receipt_count: u64,
    pub unreferenced_object_count: u64,
    pub reconstructed_partitions: Vec<ShardIndexPartition>,
    pub capacity_ledger: Option<CapacityLedger>,
    pub capacity: ShardCapacityAudit,
    pub issues: Vec<ShardAuditIssue>,
}

/// Audit one exact shard revision. No remote mutation is possible through this
/// API. Tree enumeration is path-only, so ordinary audits never download
/// payload blobs. Append-only payload-batch records provide the exact
/// first-seen size history, including incomplete transactions, when their
/// coverage is complete.
#[allow(clippy::too_many_arguments)]
pub fn audit_remote_shard(
    remote: &dyn RemoteGitStore,
    repository: &str,
    branch: &str,
    revision: &str,
    shard_id: &str,
    policy: &ShardAuditPolicy,
    cancellation: &CancellationToken,
) -> Result<RemoteShardAuditReport, CacheError> {
    policy.validate()?;
    if repository.trim().is_empty() || branch.trim().is_empty() || shard_id.trim().is_empty() {
        return Err(CacheError::InvalidManifest(
            "shard audit repository, branch, and shard identity are required".to_owned(),
        ));
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;

    let mut listings = BTreeMap::new();
    let mut all_paths = BTreeSet::new();
    let mut listed_path_bytes = 0u64;
    for prefix in AUDIT_PREFIXES {
        let listing = remote.list_committed_paths(
            repository,
            revision,
            prefix,
            policy.maximum_paths_per_prefix,
            policy.maximum_path_bytes_per_prefix,
            cancellation,
        )?;
        validate_listing(&listing, revision, prefix)?;
        listed_path_bytes = listed_path_bytes
            .checked_add(listing.total_path_bytes)
            .ok_or_else(|| CacheError::ResourceLimit("audit path bytes exceed u64".to_owned()))?;
        all_paths.extend(listing.paths.iter().cloned());
        listings.insert(prefix.to_owned(), listing);
    }

    let mut metadata_bytes = 0u64;
    let mut issues = Vec::new();
    let payload_history = audit_payload_batch_records(
        remote,
        repository,
        branch,
        revision,
        shard_id,
        &listings["transactions"],
        &listings["objects"],
        &all_paths,
        policy,
        cancellation,
        &mut metadata_bytes,
        &mut issues,
    )?;
    let mut originals = BTreeMap::<(String, String), ShardIndexPartition>::new();
    let mut index_sources = BTreeMap::<String, ContentDigest>::new();
    for path in paths_with_suffix(&listings["indexes"], ".json") {
        match read_json::<ShardIndexPartition>(
            remote,
            repository,
            revision,
            path,
            policy,
            cancellation,
            &mut metadata_bytes,
        ) {
            Ok(document) => {
                let expected_path = format!(
                    "indexes/{}/{}.json",
                    document.value.family, document.value.semantic_prefix
                );
                if document.value.validate().is_err() || path != &expected_path {
                    issue(
                        &mut issues,
                        ShardAuditSeverity::Error,
                        ShardAuditIssueKind::InvalidIndex,
                        Some(path),
                        None,
                        "index partition is invalid or stored at the wrong path",
                    );
                    continue;
                }
                let key = (
                    document.value.family.clone(),
                    document.value.semantic_prefix.clone(),
                );
                if originals.insert(key, document.value).is_some() {
                    issue(
                        &mut issues,
                        ShardAuditSeverity::Error,
                        ShardAuditIssueKind::InvalidIndex,
                        Some(path),
                        None,
                        "duplicate index partition identity",
                    );
                }
                index_sources.insert(path.clone(), document.source.content_digest);
            }
            Err(error) if recoverable_document_error(&error) => issue(
                &mut issues,
                ShardAuditSeverity::Error,
                ShardAuditIssueKind::InvalidIndex,
                Some(path),
                None,
                &error.to_string(),
            ),
            Err(error) => return Err(error),
        }
    }

    let index_entry_count = originals
        .values()
        .map(|partition| partition.entries.len() as u64)
        .sum();
    let mut rebuilt_entries = BTreeMap::<(String, String), Vec<ShardIndexEntry>>::new();
    let mut referenced_paths = BTreeSet::new();
    let mut verified_metadata = BTreeMap::<String, ContentDigest>::new();
    let mut accepted_manifests = BTreeSet::new();
    let mut unique_objects = BTreeMap::<ContentDigest, (String, u64)>::new();
    let mut logical_payload_bytes = 0u64;
    let mut complete_artifact_count = 0u64;

    for ((family, prefix), partition) in &originals {
        let index_path = format!("indexes/{family}/{prefix}.json");
        referenced_paths.insert(index_path.clone());
        for entry in &partition.entries {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let manifest_path = format!(
                "manifests/{}/{}.json",
                &entry.semantic_digest.0[..2],
                entry.manifest_digest.0
            );
            referenced_paths.insert(manifest_path.clone());
            let manifest = match read_json::<CanonicalArtifactManifest>(
                remote,
                repository,
                revision,
                &manifest_path,
                policy,
                cancellation,
                &mut metadata_bytes,
            ) {
                Ok(document) => document,
                Err(error) if recoverable_document_error(&error) => {
                    issue(
                        &mut issues,
                        ShardAuditSeverity::Error,
                        if matches!(error, CacheError::NotFound(_)) {
                            ShardAuditIssueKind::MissingManifest
                        } else {
                            ShardAuditIssueKind::InvalidManifest
                        },
                        Some(&manifest_path),
                        Some(entry.manifest_digest.clone()),
                        &error.to_string(),
                    );
                    continue;
                }
                Err(error) => return Err(error),
            };
            let manifest_digest = manifest.value.digest();
            if manifest_digest.as_ref() != Ok(&entry.manifest_digest)
                || manifest.source.content_digest != entry.manifest_digest
                || manifest.value.artifact_family != *family
                || manifest.value.semantic_digest != entry.semantic_digest
                || manifest.value.payload_digest != entry.canonical_payload_digest
                || manifest.value.transport_digests != entry.transport_digests
            {
                issue(
                    &mut issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::InvalidManifest,
                    Some(&manifest_path),
                    Some(entry.manifest_digest.clone()),
                    "manifest identity does not match its index entry",
                );
                continue;
            }
            verified_metadata.insert(manifest_path.clone(), entry.manifest_digest.clone());

            let receipt_candidates = [
                format!(
                    "transactions/{}/private/receipt.json",
                    entry.publication_transaction_id
                ),
                format!(
                    "transactions/{}/public/receipt.json",
                    entry.publication_transaction_id
                ),
                // Read-only compatibility for pre-destination-namespaced
                // receipts. New publications never write this path.
                format!(
                    "transactions/{}/receipt.json",
                    entry.publication_transaction_id
                ),
            ];
            let matching_receipts = receipt_candidates
                .into_iter()
                .filter(|path| all_paths.contains(path))
                .collect::<Vec<_>>();
            if matching_receipts.len() != 1 {
                issue(
                    &mut issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::MissingReceipt,
                    None,
                    Some(entry.manifest_digest.clone()),
                    "index entry does not resolve to exactly one target-specific receipt",
                );
                continue;
            }
            let receipt_path = matching_receipts[0].clone();
            referenced_paths.insert(receipt_path.clone());
            let receipt = match read_json::<PublicationReceipt>(
                remote,
                repository,
                revision,
                &receipt_path,
                policy,
                cancellation,
                &mut metadata_bytes,
            ) {
                Ok(document) => document,
                Err(error) if recoverable_document_error(&error) => {
                    issue(
                        &mut issues,
                        ShardAuditSeverity::Error,
                        if matches!(error, CacheError::NotFound(_)) {
                            ShardAuditIssueKind::MissingReceipt
                        } else {
                            ShardAuditIssueKind::InvalidReceipt
                        },
                        Some(&receipt_path),
                        None,
                        &error.to_string(),
                    );
                    continue;
                }
                Err(error) => return Err(error),
            };
            let index_digest = &index_sources[&index_path];
            if receipt.value.validate().is_err()
                || receipt.value.digest().as_ref() != Ok(&receipt.source.content_digest)
                || receipt.value.transaction_id != entry.publication_transaction_id
                || receipt.value.shard_id != shard_id
                || receipt.value.branch != branch
                || receipt.value.semantic_digest != entry.semantic_digest
                || receipt.value.canonical_payload_digest != entry.canonical_payload_digest
                || receipt.value.manifest_digest != entry.manifest_digest
                || !entry
                    .transport_digests
                    .contains(&receipt.value.transport_digest)
                || receipt
                    .value
                    .discoverability_subject_digests
                    .get(&index_path)
                    != Some(index_digest)
                || receipt.value.metadata_file_digests.get(&manifest_path)
                    != Some(&entry.manifest_digest)
            {
                issue(
                    &mut issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::InvalidReceipt,
                    Some(&receipt_path),
                    None,
                    "receipt does not prove its index entry and immutable metadata",
                );
                continue;
            }

            let mut complete = true;
            for (path, expected_digest) in &receipt.value.payload_batch_record_digests {
                referenced_paths.insert(path.clone());
                if payload_history.record_digests.get(path) != Some(expected_digest) {
                    issue(
                        &mut issues,
                        ShardAuditSeverity::Error,
                        ShardAuditIssueKind::InvalidPayloadBatchRecord,
                        Some(path),
                        Some(expected_digest.clone()),
                        "receipt payload-batch record is absent or has a different digest",
                    );
                    complete = false;
                }
            }
            for transport_digest in &entry.transport_digests {
                let encoding_path = format!(
                    "encodings/{}/{}.json",
                    &entry.canonical_payload_digest.0[..2],
                    transport_digest.0
                );
                referenced_paths.insert(encoding_path.clone());
                let encoding = match read_json::<TransportEncodingRecord>(
                    remote,
                    repository,
                    revision,
                    &encoding_path,
                    policy,
                    cancellation,
                    &mut metadata_bytes,
                ) {
                    Ok(document) => document,
                    Err(error) if recoverable_document_error(&error) => {
                        issue(
                            &mut issues,
                            ShardAuditSeverity::Error,
                            if matches!(error, CacheError::NotFound(_)) {
                                ShardAuditIssueKind::MissingEncoding
                            } else {
                                ShardAuditIssueKind::InvalidEncoding
                            },
                            Some(&encoding_path),
                            Some(transport_digest.clone()),
                            &error.to_string(),
                        );
                        complete = false;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                if encoding.value.digest().as_ref() != Ok(transport_digest)
                    || encoding.source.content_digest != *transport_digest
                    || encoding.value.canonical_payload_digest != entry.canonical_payload_digest
                    || receipt.value.metadata_file_digests.get(&encoding_path)
                        != Some(transport_digest)
                {
                    issue(
                        &mut issues,
                        ShardAuditSeverity::Error,
                        ShardAuditIssueKind::InvalidEncoding,
                        Some(&encoding_path),
                        Some(transport_digest.clone()),
                        "transport record does not match the index, receipt, or payload",
                    );
                    complete = false;
                    continue;
                }
                verified_metadata.insert(encoding_path, transport_digest.clone());
                for part in &encoding.value.ordered_parts {
                    referenced_paths.insert(part.repository_path.clone());
                    if !all_paths.contains(&part.repository_path) {
                        issue(
                            &mut issues,
                            ShardAuditSeverity::Error,
                            ShardAuditIssueKind::MissingObject,
                            Some(&part.repository_path),
                            Some(part.content_digest.clone()),
                            "transport object is absent from the audited revision",
                        );
                        complete = false;
                    }
                    match unique_objects.get(&part.content_digest) {
                        Some((path, size))
                            if path != &part.repository_path || *size != part.size_bytes =>
                        {
                            issue(
                                &mut issues,
                                ShardAuditSeverity::Error,
                                ShardAuditIssueKind::InvalidEncoding,
                                Some(&part.repository_path),
                                Some(part.content_digest.clone()),
                                "one object digest has conflicting path or size declarations",
                            );
                            complete = false;
                        }
                        Some(_) => {}
                        None => {
                            unique_objects.insert(
                                part.content_digest.clone(),
                                (part.repository_path.clone(), part.size_bytes),
                            );
                        }
                    }
                }
            }

            for (path, expected_digest) in &receipt.value.metadata_file_digests {
                referenced_paths.insert(path.clone());
                if let Some(actual_digest) = verified_metadata.get(path) {
                    if actual_digest != expected_digest {
                        complete = false;
                    }
                    continue;
                }
                match read_bytes(
                    remote,
                    repository,
                    revision,
                    path,
                    policy,
                    cancellation,
                    &mut metadata_bytes,
                ) {
                    Ok(actual_digest) if &actual_digest == expected_digest => {
                        verified_metadata.insert(path.clone(), actual_digest);
                    }
                    Ok(actual_digest) => {
                        issue(
                            &mut issues,
                            ShardAuditSeverity::Error,
                            ShardAuditIssueKind::MissingMetadata,
                            Some(path),
                            Some(expected_digest.clone()),
                            &format!("metadata digest is {actual_digest}"),
                        );
                        complete = false;
                    }
                    Err(error) if recoverable_document_error(&error) => {
                        issue(
                            &mut issues,
                            ShardAuditSeverity::Error,
                            ShardAuditIssueKind::MissingMetadata,
                            Some(path),
                            Some(expected_digest.clone()),
                            &error.to_string(),
                        );
                        complete = false;
                    }
                    Err(error) => return Err(error),
                }
            }
            if !complete {
                continue;
            }
            complete_artifact_count = complete_artifact_count.saturating_add(1);
            if accepted_manifests.insert(entry.manifest_digest.clone()) {
                logical_payload_bytes = logical_payload_bytes
                    .checked_add(manifest.value.canonical_payload.logical_size_bytes())
                    .ok_or_else(|| {
                        CacheError::ResourceLimit("audit logical bytes exceed u64".to_owned())
                    })?;
            }
            rebuilt_entries
                .entry((family.clone(), prefix.clone()))
                .or_default()
                .push(entry.clone());
        }
    }

    let mut reconstructed_partitions = Vec::new();
    for ((family, prefix), original) in &originals {
        let rebuilt = ShardIndexPartition::rebuild(
            family.clone(),
            prefix.clone(),
            rebuilt_entries
                .remove(&(family.clone(), prefix.clone()))
                .unwrap_or_default(),
        )?;
        if &rebuilt != original {
            issue(
                &mut issues,
                ShardAuditSeverity::Error,
                ShardAuditIssueKind::ProjectionDrift,
                Some(&format!("indexes/{family}/{prefix}.json")),
                None,
                "derived complete entries do not reproduce the stored index partition",
            );
        }
        reconstructed_partitions.push(rebuilt);
    }

    let capacity_ledger = if all_paths.contains(DEFAULT_CAPACITY_LEDGER_PATH) {
        match read_json::<CapacityLedger>(
            remote,
            repository,
            revision,
            DEFAULT_CAPACITY_LEDGER_PATH,
            policy,
            cancellation,
            &mut metadata_bytes,
        ) {
            Ok(document)
                if document.value.validate().is_ok() && document.value.shard_id == shard_id =>
            {
                referenced_paths.insert(DEFAULT_CAPACITY_LEDGER_PATH.to_owned());
                Some(document.value)
            }
            Ok(_) => {
                issue(
                    &mut issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::LedgerDrift,
                    Some(DEFAULT_CAPACITY_LEDGER_PATH),
                    None,
                    "capacity ledger is invalid or names another shard",
                );
                None
            }
            Err(error) if recoverable_document_error(&error) => {
                issue(
                    &mut issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::LedgerDrift,
                    Some(DEFAULT_CAPACITY_LEDGER_PATH),
                    None,
                    &error.to_string(),
                );
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        issue(
            &mut issues,
            ShardAuditSeverity::Error,
            ShardAuditIssueKind::LedgerDrift,
            Some(DEFAULT_CAPACITY_LEDGER_PATH),
            None,
            "capacity ledger is absent",
        );
        None
    };

    let unreferenced_manifest_count = report_unreferenced(
        &listings["manifests"],
        &referenced_paths,
        ShardAuditIssueKind::UnreferencedManifest,
        &mut issues,
    );
    let unreferenced_encoding_count = report_unreferenced(
        &listings["encodings"],
        &referenced_paths,
        ShardAuditIssueKind::UnreferencedEncoding,
        &mut issues,
    );
    let unreferenced_object_count = report_unreferenced(
        &listings["objects"],
        &referenced_paths,
        ShardAuditIssueKind::UnreferencedObject,
        &mut issues,
    );
    let receipt_paths = listings["transactions"]
        .paths
        .iter()
        .filter(|path| path.ends_with("/receipt.json"))
        .cloned()
        .collect::<Vec<_>>();
    let unreferenced_receipt_count = receipt_paths
        .iter()
        .filter(|path| !referenced_paths.contains(*path))
        .count() as u64;
    for path in receipt_paths
        .iter()
        .filter(|path| !referenced_paths.contains(*path))
    {
        issue(
            &mut issues,
            ShardAuditSeverity::Warning,
            ShardAuditIssueKind::UnreferencedReceipt,
            Some(path),
            None,
            "receipt is not referenced by a current index entry",
        );
    }
    for path in listings["transactions"]
        .paths
        .iter()
        .filter(|path| path.ends_with("/plan.json"))
    {
        let receipt_path = path.trim_end_matches("plan.json").to_owned() + "receipt.json";
        if !all_paths.contains(&receipt_path) {
            issue(
                &mut issues,
                ShardAuditSeverity::Warning,
                ShardAuditIssueKind::IncompleteTransaction,
                Some(path),
                None,
                "transaction plan has no committed receipt",
            );
        }
    }

    let unique_transport_object_bytes = unique_objects
        .values()
        .try_fold(0u64, |total, (_, size)| total.checked_add(*size))
        .ok_or_else(|| CacheError::ResourceLimit("audit transport bytes exceed u64".to_owned()))?;
    let ledger_covers_referenced_transport = capacity_ledger.as_ref().is_some_and(|ledger| {
        ledger.first_seen_immutable_payload_bytes >= unique_transport_object_bytes
    });
    if capacity_ledger.is_some() && !ledger_covers_referenced_transport {
        issue(
            &mut issues,
            ShardAuditSeverity::Error,
            ShardAuditIssueKind::LedgerDrift,
            Some(DEFAULT_CAPACITY_LEDGER_PATH),
            None,
            "ledger first-seen payload bytes are below referenced unique transport bytes",
        );
    }
    let ledger_matches_durable_payload_history = payload_history.coverage_complete
        && capacity_ledger.as_ref().is_some_and(|ledger| {
            ledger
                .first_seen_immutable_payload_bytes
                .saturating_add(ledger.abandoned_reachable_bytes)
                == payload_history.newly_introduced_bytes
        });
    if payload_history.coverage_complete
        && capacity_ledger.is_some()
        && !ledger_matches_durable_payload_history
    {
        issue(
            &mut issues,
            ShardAuditSeverity::Error,
            ShardAuditIssueKind::LedgerDrift,
            Some(DEFAULT_CAPACITY_LEDGER_PATH),
            None,
            "ledger completed-plus-abandoned payload bytes do not match durable batch history",
        );
    }
    let capacity = ShardCapacityAudit {
        ledger_accounted_bytes: capacity_ledger
            .as_ref()
            .map(CapacityLedger::accounted_bytes),
        ledger_first_seen_immutable_payload_bytes: capacity_ledger
            .as_ref()
            .map(|ledger| ledger.first_seen_immutable_payload_bytes),
        referenced_unique_transport_bytes: unique_transport_object_bytes,
        ledger_covers_referenced_transport,
        ledger_remaining_capacity_bytes: capacity_ledger.as_ref().map(|ledger| {
            ledger
                .hard_capacity_bytes
                .saturating_sub(ledger.accounted_bytes())
        }),
        durable_batch_record_count: payload_history.valid_record_count,
        durable_recorded_new_payload_bytes: payload_history.newly_introduced_bytes,
        durable_record_coverage_complete: payload_history.coverage_complete,
        ledger_matches_durable_payload_history,
        exact_history_rebuild_available: payload_history.coverage_complete,
    };
    issues.sort_by(|left, right| {
        (&left.repository_path, &left.reason).cmp(&(&right.repository_path, &right.reason))
    });
    Ok(RemoteShardAuditReport {
        schema_version: 1,
        repository: repository.to_owned(),
        branch: branch.to_owned(),
        revision: revision.to_owned(),
        shard_id: shard_id.to_owned(),
        listed_path_count: all_paths.len() as u64,
        listed_path_bytes,
        verified_metadata_bytes_read: metadata_bytes,
        index_partition_count: originals.len() as u64,
        index_entry_count,
        complete_artifact_count,
        logical_payload_bytes,
        unique_transport_object_count: unique_objects.len() as u64,
        unique_transport_object_bytes,
        unreferenced_manifest_count,
        unreferenced_encoding_count,
        unreferenced_receipt_count,
        unreferenced_object_count,
        reconstructed_partitions,
        capacity_ledger,
        capacity,
        issues,
    })
}

struct PayloadHistoryAudit {
    valid_record_count: u64,
    newly_introduced_bytes: u64,
    coverage_complete: bool,
    record_digests: BTreeMap<String, ContentDigest>,
}

#[allow(clippy::too_many_arguments)]
fn audit_payload_batch_records(
    remote: &dyn RemoteGitStore,
    repository: &str,
    branch: &str,
    revision: &str,
    shard_id: &str,
    transaction_listing: &RemotePathListReport,
    object_listing: &RemotePathListReport,
    all_paths: &BTreeSet<String>,
    policy: &ShardAuditPolicy,
    cancellation: &CancellationToken,
    metadata_bytes: &mut u64,
    issues: &mut Vec<ShardAuditIssue>,
) -> Result<PayloadHistoryAudit, CacheError> {
    let record_paths = transaction_listing
        .paths
        .iter()
        .filter(|path| path.contains("/batches/") && path.ends_with(".json"))
        .collect::<Vec<_>>();
    let mut valid_record_count = 0u64;
    let mut coverage_complete = true;
    let mut record_digests = BTreeMap::new();
    let mut declared_by_path = BTreeMap::<String, (ContentDigest, u64)>::new();
    let mut declared_by_digest = BTreeMap::<ContentDigest, (String, u64)>::new();
    let mut newly_introduced = BTreeMap::<ContentDigest, (String, u64)>::new();

    for path in record_paths {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let document = match read_json::<PayloadBatchRecord>(
            remote,
            repository,
            revision,
            path,
            policy,
            cancellation,
            metadata_bytes,
        ) {
            Ok(document) => document,
            Err(error) if recoverable_document_error(&error) => {
                issue(
                    issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::InvalidPayloadBatchRecord,
                    Some(path),
                    None,
                    &error.to_string(),
                );
                coverage_complete = false;
                continue;
            }
            Err(error) => return Err(error),
        };
        let record = &document.value;
        let mut record_valid = record.validate().is_ok()
            && record.digest().as_ref() == Ok(&document.source.content_digest)
            && record.repository_path() == **path
            && record.shard_id == shard_id
            && record.branch == branch;
        if !record_valid {
            issue(
                issues,
                ShardAuditSeverity::Error,
                ShardAuditIssueKind::InvalidPayloadBatchRecord,
                Some(path),
                None,
                "payload-batch record identity, digest, shard, branch, or path is invalid",
            );
            coverage_complete = false;
            continue;
        }

        for object in &record.objects {
            if !object.repository_path.starts_with("objects/")
                || !all_paths.contains(&object.repository_path)
            {
                issue(
                    issues,
                    ShardAuditSeverity::Error,
                    if all_paths.contains(&object.repository_path) {
                        ShardAuditIssueKind::InvalidPayloadBatchRecord
                    } else {
                        ShardAuditIssueKind::MissingObject
                    },
                    Some(&object.repository_path),
                    Some(object.content_digest.clone()),
                    "payload-batch record names a missing or non-object path",
                );
                record_valid = false;
                continue;
            }
            if declared_by_path
                .get(&object.repository_path)
                .is_some_and(|identity| {
                    identity != &(object.content_digest.clone(), object.size_bytes)
                })
                || declared_by_digest
                    .get(&object.content_digest)
                    .is_some_and(|identity| {
                        identity != &(object.repository_path.clone(), object.size_bytes)
                    })
                || (object.newly_introduced
                    && newly_introduced.contains_key(&object.content_digest))
            {
                issue(
                    issues,
                    ShardAuditSeverity::Error,
                    ShardAuditIssueKind::InvalidPayloadBatchRecord,
                    Some(path),
                    Some(object.content_digest.clone()),
                    "payload object identity conflicts with history or is marked newly introduced more than once",
                );
                record_valid = false;
            }
        }
        if !record_valid {
            coverage_complete = false;
            continue;
        }

        for object in &record.objects {
            declared_by_path.insert(
                object.repository_path.clone(),
                (object.content_digest.clone(), object.size_bytes),
            );
            declared_by_digest.insert(
                object.content_digest.clone(),
                (object.repository_path.clone(), object.size_bytes),
            );
            if object.newly_introduced {
                newly_introduced.insert(
                    object.content_digest.clone(),
                    (object.repository_path.clone(), object.size_bytes),
                );
            }
        }
        valid_record_count = valid_record_count.saturating_add(1);
        record_digests.insert(path.clone(), document.source.content_digest);
    }

    for path in &object_listing.paths {
        if !declared_by_path.contains_key(path) {
            issue(
                issues,
                ShardAuditSeverity::Warning,
                ShardAuditIssueKind::PayloadHistoryGap,
                Some(path),
                None,
                "reachable payload object has no durable size-bearing batch record",
            );
            coverage_complete = false;
        }
    }
    for (digest, (path, _)) in &declared_by_digest {
        if !newly_introduced.contains_key(digest) {
            issue(
                issues,
                ShardAuditSeverity::Warning,
                ShardAuditIssueKind::PayloadHistoryGap,
                Some(path),
                Some(digest.clone()),
                "payload history never records this object as newly introduced",
            );
            coverage_complete = false;
        }
    }
    let newly_introduced_bytes = newly_introduced
        .values()
        .try_fold(0u64, |total, (_, size)| {
            total.checked_add(*size).ok_or_else(|| {
                CacheError::ResourceLimit("durable payload history bytes exceed u64".to_owned())
            })
        })?;
    Ok(PayloadHistoryAudit {
        valid_record_count,
        newly_introduced_bytes,
        coverage_complete,
        record_digests,
    })
}

fn validate_listing(
    listing: &RemotePathListReport,
    revision: &str,
    prefix: &str,
) -> Result<(), CacheError> {
    if listing.revision != revision
        || listing.prefix != prefix
        || listing.paths.windows(2).any(|pair| pair[0] >= pair[1])
        || listing.total_path_bytes
            != listing
                .paths
                .iter()
                .map(|path| path.len() as u64)
                .sum::<u64>()
    {
        return Err(CacheError::InvalidManifest(
            "remote path listing failed canonical validation".to_owned(),
        ));
    }
    Ok(())
}

fn paths_with_suffix<'a>(
    listing: &'a RemotePathListReport,
    suffix: &'a str,
) -> impl Iterator<Item = &'a String> + 'a {
    listing
        .paths
        .iter()
        .filter(move |path| path.ends_with(suffix))
}

#[allow(clippy::too_many_arguments)]
fn read_json<T: DeserializeOwned>(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    path: &str,
    policy: &ShardAuditPolicy,
    cancellation: &CancellationToken,
    metadata_bytes: &mut u64,
) -> Result<RemoteDocument<T>, CacheError> {
    let remaining = policy
        .maximum_total_metadata_bytes
        .checked_sub(*metadata_bytes)
        .ok_or_else(|| {
            CacheError::ResourceLimit("shard audit metadata budget exhausted".to_owned())
        })?;
    if remaining == 0 {
        return Err(CacheError::ResourceLimit(
            "shard audit metadata budget exhausted".to_owned(),
        ));
    }
    let document = RemoteShardReader::new(remote, policy.maximum_document_bytes.min(remaining))?
        .read_json(repository, revision, path, cancellation)?;
    *metadata_bytes = metadata_bytes
        .checked_add(document.source.size_bytes)
        .ok_or_else(|| CacheError::ResourceLimit("audit metadata bytes exceed u64".to_owned()))?;
    Ok(document)
}

#[allow(clippy::too_many_arguments)]
fn read_bytes(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    path: &str,
    policy: &ShardAuditPolicy,
    cancellation: &CancellationToken,
    metadata_bytes: &mut u64,
) -> Result<ContentDigest, CacheError> {
    let remaining = policy
        .maximum_total_metadata_bytes
        .checked_sub(*metadata_bytes)
        .ok_or_else(|| {
            CacheError::ResourceLimit("shard audit metadata budget exhausted".to_owned())
        })?;
    if remaining == 0 {
        return Err(CacheError::ResourceLimit(
            "shard audit metadata budget exhausted".to_owned(),
        ));
    }
    let maximum = policy.maximum_document_bytes.min(remaining);
    let mut sink = Vec::new();
    let source =
        remote.read_committed_path(repository, revision, path, maximum, cancellation, &mut sink)?;
    *metadata_bytes = metadata_bytes
        .checked_add(source.size_bytes)
        .ok_or_else(|| CacheError::ResourceLimit("audit metadata bytes exceed u64".to_owned()))?;
    Ok(source.content_digest)
}

fn recoverable_document_error(error: &CacheError) -> bool {
    matches!(
        error,
        CacheError::Serialization(_)
            | CacheError::InvalidManifest(_)
            | CacheError::DigestMismatch { .. }
            | CacheError::NotFound(_)
    )
}

fn report_unreferenced(
    listing: &RemotePathListReport,
    referenced_paths: &BTreeSet<String>,
    kind: ShardAuditIssueKind,
    issues: &mut Vec<ShardAuditIssue>,
) -> u64 {
    let paths = listing
        .paths
        .iter()
        .filter(|path| !referenced_paths.contains(*path))
        .collect::<Vec<_>>();
    for path in &paths {
        issue(
            issues,
            ShardAuditSeverity::Warning,
            kind,
            Some(path),
            None,
            "reachable path is not referenced by the current discoverable inventory",
        );
    }
    paths.len() as u64
}

fn issue(
    issues: &mut Vec<ShardAuditIssue>,
    severity: ShardAuditSeverity,
    kind: ShardAuditIssueKind,
    repository_path: Option<&str>,
    identity_digest: Option<ContentDigest>,
    reason: &str,
) {
    issues.push(ShardAuditIssue {
        severity,
        kind,
        repository_path: repository_path.map(str::to_owned),
        identity_digest,
        reason: reason.to_owned(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical_json_bytes;
    use crate::{
        ArtifactAssuranceState, ArtifactDisposition, CanonicalPayloadEnvelope,
        CompareAndSwapResult, LogicalPayloadItem, PayloadBatchObjectRecord, PublicationDestination,
        RemoteCommitRequest, RemoteReadReport, SemanticKeyEnvelope, ToolkitVersion, TransportPart,
        GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use serde::Serialize;
    use serde_json::json;
    use std::io::Write;
    use xc_core::{AssuranceLevel, PublicationAuthorityMode};

    struct MemoryRemote {
        revision: String,
        paths: BTreeMap<String, Vec<u8>>,
    }

    impl RemoteGitStore for MemoryRemote {
        fn read_ref(&self, _repository: &str, _branch: &str) -> Result<String, CacheError> {
            Ok(self.revision.clone())
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(self
                .paths
                .get(path)
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
            if revision != self.revision {
                return Err(CacheError::NotFound(revision.to_owned()));
            }
            let bytes = self
                .paths
                .get(path)
                .ok_or_else(|| CacheError::NotFound(path.to_owned()))?;
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

        fn list_committed_paths(
            &self,
            _repository: &str,
            revision: &str,
            prefix: &str,
            maximum_paths: u64,
            maximum_total_path_bytes: u64,
            _cancellation: &CancellationToken,
        ) -> Result<RemotePathListReport, CacheError> {
            if revision != self.revision {
                return Err(CacheError::NotFound(revision.to_owned()));
            }
            let paths = self
                .paths
                .keys()
                .filter(|path| *path == prefix || path.starts_with(&format!("{prefix}/")))
                .cloned()
                .collect::<Vec<_>>();
            let total_path_bytes = paths.iter().map(|path| path.len() as u64).sum();
            if paths.len() as u64 > maximum_paths || total_path_bytes > maximum_total_path_bytes {
                return Err(CacheError::ResourceLimit(prefix.to_owned()));
            }
            Ok(RemotePathListReport {
                prefix: prefix.to_owned(),
                revision: revision.to_owned(),
                paths,
                total_path_bytes,
            })
        }

        fn compare_and_swap_commit(
            &self,
            _request: &RemoteCommitRequest,
        ) -> Result<CompareAndSwapResult, CacheError> {
            Err(CacheError::ReadOnlyLayer("audit fixture".to_owned()))
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            _revision: &str,
            _part: &TransportPart,
        ) -> Result<(), CacheError> {
            Ok(())
        }
    }

    fn insert_document<T: Serialize>(
        paths: &mut BTreeMap<String, Vec<u8>>,
        path: impl Into<String>,
        value: &T,
    ) -> ContentDigest {
        let bytes = canonical_json_bytes(value).unwrap();
        let digest = ContentDigest::sha256(&bytes);
        paths.insert(path.into(), bytes);
        digest
    }

    fn fixture() -> MemoryRemote {
        let revision = "a".repeat(40);
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "audit_fixture".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: json!({"n": 1}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let payload_bytes = b"payload";
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "bytes".to_owned(),
            dimensions: vec![payload_bytes.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(payload_bytes),
                size_bytes: payload_bytes.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let object_digest = ContentDigest::sha256(b"encoded");
        let object_path = format!(
            "objects/sha256/{}/{}.part",
            &object_digest.0[..2],
            object_digest.0
        );
        let encoding = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: payload_digest.clone(),
            encoder_profile: "fixture-v1".to_owned(),
            package_size_bytes: 7,
            package_digest: object_digest.clone(),
            ordered_parts: vec![TransportPart {
                sequence: 0,
                repository_path: object_path.clone(),
                size_bytes: 7,
                content_digest: object_digest.clone(),
            }],
            reconstruction: "concatenate".to_owned(),
        };
        let transport_digest = encoding.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "fixture".to_owned(),
            semantic_key,
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![transport_digest.clone()],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "audit fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let transaction_id = ContentDigest::sha256(b"transaction").0;
        let index = ShardIndexPartition::rebuild(
            "fixture",
            semantic_digest.0[..2].to_owned(),
            vec![ShardIndexEntry {
                semantic_digest: semantic_digest.clone(),
                canonical_payload_digest: payload_digest.clone(),
                manifest_digest: manifest_digest.clone(),
                achieved_assurance: ArtifactAssuranceState::Computed,
                disposition: ArtifactDisposition::Active,
                producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
                transport_digests: vec![transport_digest.clone()],
                publication_transaction_id: transaction_id.clone(),
            }],
        )
        .unwrap();
        let prefix = &semantic_digest.0[..2];
        let index_path = format!("indexes/fixture/{prefix}.json");
        let manifest_path = format!("manifests/{prefix}/{manifest_digest}.json");
        let encoding_path = format!(
            "encodings/{}/{transport_digest}.json",
            &payload_digest.0[..2]
        );
        let receipt_path = format!("transactions/{transaction_id}/receipt.json");
        let mut paths = BTreeMap::new();
        let index_digest = insert_document(&mut paths, &index_path, &index);
        insert_document(&mut paths, &manifest_path, &manifest);
        insert_document(&mut paths, &encoding_path, &encoding);
        paths.insert(object_path.clone(), b"encoded".to_vec());
        let batch_record = PayloadBatchRecord {
            schema_version: 1,
            transaction_id: transaction_id.clone(),
            idempotency_key: ContentDigest(transaction_id.clone()),
            destination: PublicationDestination::Public,
            authorized_repository: "team/shard".to_owned(),
            shard_id: "fixture-001".to_owned(),
            branch: "main".to_owned(),
            sequence: 0,
            payload_parent_head: "payload-parent".to_owned(),
            payload_commit_id: "payload-commit".to_owned(),
            planned_payload_bytes: 7,
            newly_committed_payload_bytes: 7,
            objects: vec![PayloadBatchObjectRecord {
                repository_path: object_path,
                size_bytes: 7,
                content_digest: object_digest,
                newly_introduced: true,
            }],
        };
        let batch_record_path = batch_record.repository_path();
        let batch_record_digest = insert_document(&mut paths, &batch_record_path, &batch_record);
        let receipt = PublicationReceipt {
            schema_version: 1,
            transaction_id: transaction_id.clone(),
            idempotency_key: ContentDigest(transaction_id),
            destination: PublicationDestination::Public,
            principal: "auditor".to_owned(),
            authorized_repository: "team/shard".to_owned(),
            repository_permission_evidence_digest: ContentDigest::sha256(b"permission"),
            shard_id: "fixture-001".to_owned(),
            branch: "main".to_owned(),
            semantic_digest,
            canonical_payload_digest: payload_digest,
            manifest_digest: manifest_digest.clone(),
            transport_digest: transport_digest.clone(),
            policy_digest: ContentDigest::sha256(b"policy"),
            policy_id: "fixture-owner-policy".to_owned(),
            authority_mode: PublicationAuthorityMode::OwnerDirect,
            validation_evidence_digests: vec![ContentDigest::sha256(b"validation")],
            contributor_authorization_digest: None,
            reviewer_approvals: Vec::new(),
            payload_commit_ids: vec!["payload-commit".to_owned()],
            payload_batch_record_commit_ids: vec!["batch-record-commit".to_owned()],
            payload_batch_record_digests: BTreeMap::from([(
                batch_record_path,
                batch_record_digest,
            )]),
            metadata_commit_id: "metadata-commit".to_owned(),
            metadata_file_digests: BTreeMap::from([
                (manifest_path, manifest_digest),
                (encoding_path, transport_digest.clone()),
            ]),
            discoverability_subject_digests: BTreeMap::from([(index_path, index_digest)]),
            remote_verification_results: vec![crate::RemoteCommitVerificationResult {
                phase: "immutable_metadata".to_owned(),
                sequence: 0,
                commit_id: "metadata-commit".to_owned(),
                verified: true,
                content_digests: vec![transport_digest],
            }],
            verified_at_unix_seconds: 1,
        };
        insert_document(&mut paths, receipt_path, &receipt);
        let ledger = CapacityLedger {
            schema_version: 1,
            shard_id: "fixture-001".to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000,
            first_seen_immutable_payload_bytes: 7,
            manifest_index_receipt_bytes: 10_000,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 0,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: revision.clone(),
            reconciliation_digest: ContentDigest::sha256(b"reconciliation"),
        };
        insert_document(&mut paths, DEFAULT_CAPACITY_LEDGER_PATH, &ledger);
        MemoryRemote { revision, paths }
    }

    fn policy() -> ShardAuditPolicy {
        ShardAuditPolicy {
            maximum_paths_per_prefix: 1_000,
            maximum_path_bytes_per_prefix: 1024 * 1024,
            maximum_document_bytes: 1024 * 1024,
            maximum_total_metadata_bytes: 16 * 1024 * 1024,
        }
    }

    #[test]
    fn audit_rebuilds_clean_inventory_without_reading_payload_blobs() {
        let remote = fixture();
        let report = audit_remote_shard(
            &remote,
            "team/shard",
            "main",
            &remote.revision,
            "fixture-001",
            &policy(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(report.complete_artifact_count, 1);
        assert_eq!(report.logical_payload_bytes, 7);
        assert_eq!(report.unique_transport_object_bytes, 7);
        assert_eq!(report.reconstructed_partitions.len(), 1);
        assert!(report.issues.is_empty());
        assert!(report.capacity.ledger_covers_referenced_transport);
        assert!(report.capacity.exact_history_rebuild_available);
        assert!(report.capacity.ledger_matches_durable_payload_history);
        assert_eq!(report.capacity.durable_recorded_new_payload_bytes, 7);
    }

    #[test]
    fn audit_reports_reachable_unreferenced_objects() {
        let mut remote = fixture();
        let digest = ContentDigest::sha256(b"orphan");
        remote.paths.insert(
            format!("objects/sha256/{}/{}.part", &digest.0[..2], digest.0),
            b"orphan".to_vec(),
        );
        let report = audit_remote_shard(
            &remote,
            "team/shard",
            "main",
            &remote.revision,
            "fixture-001",
            &policy(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(report.unreferenced_object_count, 1);
        assert!(!report.capacity.durable_record_coverage_complete);
        assert!(!report.capacity.exact_history_rebuild_available);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.kind == ShardAuditIssueKind::UnreferencedObject));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.kind == ShardAuditIssueKind::PayloadHistoryGap));
    }

    #[test]
    fn audit_rejects_a_corrupt_payload_batch_record() {
        let mut remote = fixture();
        let record_path = remote
            .paths
            .keys()
            .find(|path| path.contains("/batches/"))
            .cloned()
            .unwrap();
        remote
            .paths
            .insert(record_path.clone(), b"{not-json".to_vec());
        let report = audit_remote_shard(
            &remote,
            "team/shard",
            "main",
            &remote.revision,
            "fixture-001",
            &policy(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(!report.capacity.durable_record_coverage_complete);
        assert!(report.issues.iter().any(|issue| {
            issue.kind == ShardAuditIssueKind::InvalidPayloadBatchRecord
                && issue.repository_path.as_deref() == Some(record_path.as_str())
        }));
    }
}
