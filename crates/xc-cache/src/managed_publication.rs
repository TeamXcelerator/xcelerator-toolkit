//! Toolkit-owned adaptation from canonical numerical drafts to the existing
//! policy-gated, journaled publication engine.

use crate::{
    canonical_digest, ArtifactAssuranceState, ArtifactCompletionState, ArtifactDisposition,
    ArtifactFamilyRoute, AttestationEnvelope, AttestationKind, AuthenticatedGitHubSession,
    CacheError, CacheNetworkRegistry, CachePublicationCandidate, CachePublicationPolicy,
    CacheVisibility, CanonicalProductionDraft, CapacityLedger, ContentDigest,
    CoordinatedPublicationPlan, GitHubRepositoryEndpoint, PublicSanitizerProfile,
    PublicationDestination, PublicationMetadataBundle, RemoteGitStore,
    TargetPublicationAuthorizationRequest, TargetPublicationPlanningInput, ToolkitVersion,
    TopologyRegistry, TopologyShardRoute, TopologyShardStatus, TopologyTrustPolicy, TransportPart,
    TransportPolicy, ValidatorEvidence,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use xc_core::{PublicationAuthority, PublicationAuthorityMode, PublicationTarget};

pub const MANAGED_VALIDATOR_ID: &str = "toolkit-integrated-validation-v1";

fn producer_source_revision() -> Result<String, CacheError> {
    let revision = option_env!("XC_SOURCE_REVISION").ok_or_else(|| {
        CacheError::InvalidManifest(
            "author publication requires an exact toolkit Git revision; build from a Git checkout or set XC_SOURCE_REVISION to the full commit hash".to_owned(),
        )
    })?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CacheError::InvalidManifest(
            "author publication requires XC_SOURCE_REVISION to be a full 40-character lowercase hexadecimal Git commit".to_owned(),
        ));
    }
    Ok(revision.to_owned())
}

#[derive(Clone, Debug)]
pub struct ManagedPublicationPlanningContext {
    pub owner: String,
    pub principal: String,
    pub target: PublicationTarget,
    pub target_heads: BTreeMap<PublicationDestination, String>,
    pub capacity_ledgers: BTreeMap<PublicationDestination, CapacityLedger>,
    pub event_unix_seconds: u64,
}

#[derive(Clone, Debug)]
pub struct ManagedPreparedArtifactPublication {
    pub policy: CachePublicationPolicy,
    pub topology: TopologyRegistry,
    pub topology_trust: TopologyTrustPolicy,
    pub network: CacheNetworkRegistry,
    pub bundles: BTreeMap<PublicationDestination, PublicationMetadataBundle>,
    pub coordinated: CoordinatedPublicationPlan,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManagedPublicationExecutionReport {
    pub transaction_id: String,
    pub completed: bool,
    pub steps_executed: usize,
    pub final_journal_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManagedRunPublicationReport {
    pub schema_version: u32,
    pub transactions: Vec<ManagedPublicationExecutionReport>,
    pub all_completed: bool,
    #[serde(default)]
    pub current_tree_paths_removed: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManagedPublicationExecutionPhaseSummary {
    pub phase_index: usize,
    pub phase_digest: ContentDigest,
    pub relative_report_path: String,
    pub transaction_count: usize,
    pub completed_transaction_count: usize,
    pub all_completed: bool,
    pub current_tree_paths_removed: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManagedCumulativePublicationExecutionReport {
    pub schema_version: u32,
    pub phases: Vec<ManagedPublicationExecutionPhaseSummary>,
    pub transactions: Vec<ManagedPublicationExecutionReport>,
    pub all_completed: bool,
    pub current_tree_paths_removed: usize,
}

fn advance_capacity_ledger_for_repository_batch(
    ledger: &mut CapacityLedger,
    transaction_id: &ContentDigest,
    sequence: u64,
    expected_head: &str,
    parts: &[TransportPart],
) -> Result<(), CacheError> {
    let payload_bytes_added = parts
        .iter()
        .filter(|part| part.repository_path.starts_with("objects/"))
        .map(|part| part.size_bytes)
        .sum::<u64>();
    let metadata_bytes_added = parts
        .iter()
        .filter(|part| !part.repository_path.starts_with("objects/"))
        .map(|part| part.size_bytes)
        .sum::<u64>();
    let admission = ledger.assess_addition(payload_bytes_added, metadata_bytes_added, 0)?;
    if !admission.accepted {
        return Err(CacheError::ResourceLimit(format!(
            "shard capacity admission failed before batch {sequence}: {}",
            admission.reasons.join("; ")
        )));
    }
    ledger.first_seen_immutable_payload_bytes = ledger
        .first_seen_immutable_payload_bytes
        .saturating_add(payload_bytes_added);
    ledger.manifest_index_receipt_bytes = ledger
        .manifest_index_receipt_bytes
        .saturating_add(metadata_bytes_added);
    ledger.last_reconciled_commit = expected_head.to_owned();
    ledger.reconciliation_digest =
        ContentDigest::sha256(format!("{transaction_id}:{sequence}:{expected_head}").as_bytes());
    ledger.validate()
}

fn prune_unreferenced_current_tree(
    remote: &crate::GitCliRemoteStore,
    repository: &str,
    family: &str,
    destination: PublicationDestination,
    resources: &xc_core::ResourcePolicy,
) -> Result<usize, CacheError> {
    let cancellation = xc_core::CancellationToken::for_policy(resources);
    for _ in 0..3 {
        let head = remote.read_ref(repository, "main")?;
        let report = crate::audit_remote_shard(
            remote,
            repository,
            "main",
            &head,
            &shard_id(family, destination),
            &crate::ShardAuditPolicy {
                maximum_paths_per_prefix: 1_000_000,
                maximum_path_bytes_per_prefix: 128 * 1024 * 1024,
                maximum_document_bytes: 16 * 1024 * 1024,
                maximum_total_metadata_bytes: 512 * 1024 * 1024,
            },
            &cancellation,
        )?;
        if report
            .issues
            .iter()
            .any(|issue| issue.severity == crate::ShardAuditSeverity::Error)
        {
            return Err(CacheError::InvalidManifest(
                "replacement cleanup refused a shard with unresolved audit errors".to_owned(),
            ));
        }
        let delete_paths = report
            .issues
            .iter()
            .filter(|issue| {
                matches!(
                    issue.kind,
                    crate::ShardAuditIssueKind::UnreferencedManifest
                        | crate::ShardAuditIssueKind::UnreferencedEncoding
                        | crate::ShardAuditIssueKind::UnreferencedObject
                )
            })
            .filter_map(|issue| issue.repository_path.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if delete_paths.is_empty() {
            return Ok(0);
        }
        let request = crate::RemoteCommitRequest {
            repository: repository.to_owned(),
            branch: "main".to_owned(),
            expected_head: head,
            message: format!(
                "prune unreferenced current-tree cache data for {family} after forced replacement"
            ),
            parts: Vec::new(),
            delete_paths: delete_paths.clone(),
        };
        match remote.compare_and_swap_commit(&request)? {
            crate::CompareAndSwapResult::RefConflict { .. } => continue,
            crate::CompareAndSwapResult::Committed { commit_id } => {
                for path in &delete_paths {
                    if remote
                        .immutable_path_digest(repository, &commit_id, path)?
                        .is_some()
                    {
                        return Err(CacheError::InvalidManifest(format!(
                            "replacement cleanup did not remove current-tree path {path:?}"
                        )));
                    }
                }
                return Ok(delete_paths.len());
            }
        }
    }
    Err(CacheError::InvalidTransition(
        "replacement cleanup exceeded its compare-and-swap retry bound".to_owned(),
    ))
}

fn repository_url(owner: &str, family: &str, destination: PublicationDestination) -> String {
    format!(
        "https://github.com/{owner}/xcelerator-cache-{}-{family}-0001.git",
        visibility_name(destination)
    )
}

fn target_staging_root(root: &Path, destination: PublicationDestination) -> PathBuf {
    root.join("managed-targets").join(match destination {
        PublicationDestination::Private => "private",
        PublicationDestination::Public => "public",
    })
}

fn read_capacity_ledger(
    remote: &dyn crate::RemoteGitStore,
    repository: &str,
    revision: &str,
    cancellation: &xc_core::CancellationToken,
) -> Result<CapacityLedger, CacheError> {
    let document = crate::RemoteShardReader::new(remote, 4 * 1024 * 1024)?
        .read_json::<CapacityLedger>(
            repository,
            revision,
            crate::DEFAULT_CAPACITY_LEDGER_PATH,
            cancellation,
        )?;
    document.value.validate()?;
    Ok(document.value)
}

fn preflight_producer_monotonicity(
    remote: &dyn crate::RemoteGitStore,
    repository: &str,
    revision: &str,
    family: &str,
    semantic_digest: &ContentDigest,
    producer_toolkit_version: &ToolkitVersion,
    cancellation: &xc_core::CancellationToken,
) -> Result<(), CacheError> {
    let path = format!("indexes/{family}/{}.json", &semantic_digest.0[..2]);
    let index =
        match crate::RemoteShardReader::new(remote, 4 * 1024 * 1024)?
            .read_json::<crate::ShardIndexPartition>(repository, revision, &path, cancellation)
        {
            // A clean shard has no partition for an unseen semantic prefix. It is
            // an empty monotonicity domain, not a publication failure.
            Ok(index) => index,
            Err(CacheError::NotFound(_)) => return Ok(()),
            Err(error) => return Err(error),
        };
    index.value.validate()?;
    if index.value.family != family {
        return Err(CacheError::InvalidManifest(
            "publication preflight read an index for a different artifact family".to_owned(),
        ));
    }
    index
        .value
        .ensure_monotonic_producer(semantic_digest, producer_toolkit_version)
}

fn completed_remote_publication(
    prepared: &ManagedPreparedArtifactPublication,
    remotes: &BTreeMap<PublicationDestination, &dyn crate::RemoteGitStore>,
    cancellation: &xc_core::CancellationToken,
) -> Result<Option<ManagedPublicationExecutionReport>, CacheError> {
    let journal = prepared.coordinated.journal.as_ref().ok_or_else(|| {
        CacheError::InvalidTransition(
            "managed publication has no authorized transaction journal".to_owned(),
        )
    })?;
    let mut receipt_digests = BTreeMap::new();
    for (destination, target) in &journal.targets {
        let remote = remotes.get(destination).ok_or_else(|| {
            CacheError::InvalidManifest(
                "managed publication is missing a destination remote".to_owned(),
            )
        })?;
        let revision = remote.read_ref(&target.repository, &target.branch)?;
        let index_path = format!(
            "indexes/{}/{}.json",
            prepared.bundles[destination].family,
            &journal.semantic_digest.0[..2]
        );
        let index = match crate::RemoteShardReader::new(*remote, 4 * 1024 * 1024)?
            .read_json::<crate::ShardIndexPartition>(
            &target.repository,
            &revision,
            &index_path,
            cancellation,
        ) {
            Ok(index) => index,
            Err(CacheError::NotFound(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        index.value.validate()?;
        let manifest_digest = &journal.target_manifest_digests[destination];
        let transport_digest = prepared.bundles[destination].encoding.digest()?;
        let Some(entry) = index.value.lookup(&journal.semantic_digest).find(|entry| {
            entry.canonical_payload_digest == journal.payload_digest
                && &entry.manifest_digest == manifest_digest
                && entry.transport_digests.contains(&transport_digest)
                && entry.publication_transaction_id == journal.transaction_id
                && entry.disposition == ArtifactDisposition::Active
        }) else {
            return Ok(None);
        };
        let receipt_path = format!(
            "transactions/{}/{}/receipt.json",
            journal.transaction_id,
            visibility_name(*destination)
        );
        let receipt = match crate::RemoteShardReader::new(*remote, 4 * 1024 * 1024)?
            .read_json::<crate::PublicationReceipt>(
            &target.repository,
            &revision,
            &receipt_path,
            cancellation,
        ) {
            Ok(receipt) => receipt,
            Err(CacheError::NotFound(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        receipt.value.validate()?;
        let receipt_digest = receipt.value.digest()?;
        let manifest_path = format!(
            "manifests/{}/{}.json",
            &journal.semantic_digest.0[..2],
            manifest_digest.0
        );
        if receipt.source.content_digest != receipt_digest
            || receipt.value.transaction_id != journal.transaction_id
            || receipt.value.idempotency_key != journal.idempotency_key
            || receipt.value.destination != *destination
            || receipt.value.principal != target.permission_evidence.principal
            || receipt.value.authorized_repository != target.authorized_repository
            || receipt.value.shard_id != target.shard_id
            || receipt.value.branch != target.branch
            || receipt.value.semantic_digest != journal.semantic_digest
            || receipt.value.canonical_payload_digest != journal.payload_digest
            || &receipt.value.manifest_digest != manifest_digest
            || receipt.value.transport_digest != transport_digest
            || receipt.value.policy_digest != journal.policy_digest
            || receipt.value.metadata_file_digests.get(&manifest_path)
                != Some(&entry.manifest_digest)
        {
            return Ok(None);
        }
        // The receipt binds the index snapshot created by this transaction.
        // The live partition legitimately changes as later artifacts are
        // appended, so its current file digest must not equal that historical
        // snapshot. The exact active entry above establishes continued
        // discoverability in the current partition.
        receipt_digests.insert(*destination, receipt_digest);
    }
    Ok(Some(ManagedPublicationExecutionReport {
        transaction_id: journal.transaction_id.clone(),
        completed: true,
        steps_executed: 0,
        final_journal_digest: canonical_digest(&receipt_digests)?,
    }))
}

struct DestinationDraftSelection<'a> {
    pending: Vec<&'a CanonicalProductionDraft>,
    already_present: usize,
}

fn destination_manifest_dominates(
    existing: &crate::CanonicalArtifactManifest,
    staged: &crate::CanonicalArtifactManifest,
) -> bool {
    if existing.artifact_family != staged.artifact_family
        || existing.semantic_digest != staged.semantic_digest
        || existing.semantic_key != staged.semantic_key
        || existing.resolved_mathematical_configuration_digest
            != staged.resolved_mathematical_configuration_digest
        || existing.minimum_reader_version != staged.minimum_reader_version
        || existing.maximum_reader_version != staged.maximum_reader_version
        || existing.producer_toolkit_version < staged.producer_toolkit_version
    {
        return false;
    }
    let mut existing_payload = existing.canonical_payload.clone();
    let existing_dependencies = std::mem::take(&mut existing_payload.dependencies);
    let mut staged_payload = staged.canonical_payload.clone();
    let staged_dependencies = std::mem::take(&mut staged_payload.dependencies);
    existing_payload == staged_payload
        && staged_dependencies
            .iter()
            .all(|dependency| existing_dependencies.contains(dependency))
}

fn select_missing_destination_drafts<'a>(
    remote: &dyn crate::RemoteGitStore,
    repository: &str,
    revision: &str,
    family: &str,
    destination: PublicationDestination,
    drafts: &[&'a CanonicalProductionDraft],
    cancellation: &xc_core::CancellationToken,
) -> Result<DestinationDraftSelection<'a>, CacheError> {
    let mut partitions = BTreeMap::<String, crate::ShardIndexPartition>::new();
    let prefixes = drafts
        .iter()
        .map(|draft| draft.manifest.semantic_digest.0[..2].to_owned())
        .collect::<BTreeSet<_>>();
    let reader = crate::RemoteShardReader::new(remote, 16 * 1024 * 1024)?;
    for prefix in prefixes {
        let path = format!("indexes/{family}/{prefix}.json");
        match reader.read_json::<crate::ShardIndexPartition>(
            repository,
            revision,
            &path,
            cancellation,
        ) {
            Ok(partition) => {
                partition.value.validate()?;
                if partition.value.family != family || partition.value.semantic_prefix != prefix {
                    return Err(CacheError::InvalidManifest(format!(
                        "destination index partition {path:?} has the wrong family or prefix"
                    )));
                }
                partitions.insert(prefix, partition.value);
            }
            Err(CacheError::NotFound(_)) => {}
            Err(error) => return Err(error),
        }
    }

    let mut pending = Vec::with_capacity(drafts.len());
    let mut already_present = 0usize;
    let mut batches = BTreeMap::<String, Option<crate::RepositoryPublicationBatch>>::new();
    for draft in drafts {
        let manifest = target_manifest(draft, destination)?;
        let manifest_digest = manifest.digest()?;
        let transport_digest = draft.encoding.digest()?;
        let prefix = &manifest.semantic_digest.0[..2];
        if let Some(partition) = partitions.get(prefix) {
            partition.ensure_monotonic_producer(
                &manifest.semantic_digest,
                &manifest.producer_toolkit_version,
            )?;
        }
        let mut existing_candidates = Vec::new();
        if let Some(partition) = partitions.get(prefix) {
            for entry in partition.lookup(&manifest.semantic_digest).filter(|entry| {
                entry.achieved_assurance >= draft.achieved_assurance
                    && entry.disposition == ArtifactDisposition::Active
                    && entry.producer_toolkit_version >= manifest.producer_toolkit_version
                    && entry.minimum_reader_version == manifest.minimum_reader_version
            }) {
                let path = format!(
                    "manifests/{}/{}.json",
                    &entry.semantic_digest.0[..2],
                    entry.manifest_digest
                );
                let document = reader.read_json::<crate::CanonicalArtifactManifest>(
                    repository,
                    revision,
                    &path,
                    cancellation,
                )?;
                document.value.validate()?;
                if document.source.content_digest != entry.manifest_digest
                    || document.value.digest()? != entry.manifest_digest
                    || document.value.payload_digest != entry.canonical_payload_digest
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "destination index entry points to a non-canonical manifest {path:?}"
                    )));
                }
                let exact = entry.manifest_digest == manifest_digest
                    && entry.canonical_payload_digest == manifest.payload_digest
                    && entry.transport_digests.contains(&transport_digest);
                let dominates = exact || destination_manifest_dominates(&document.value, &manifest);
                if dominates {
                    existing_candidates.push(entry);
                }
            }
        }
        let mut proven = false;
        for entry in existing_candidates {
            let transaction_id = entry.publication_transaction_id.clone();
            if !batches.contains_key(&transaction_id) {
                let path = format!(
                    "transactions/batches/{transaction_id}/{}.json",
                    visibility_name(destination)
                );
                let batch = match reader.read_json::<crate::RepositoryPublicationBatch>(
                    repository,
                    revision,
                    &path,
                    cancellation,
                ) {
                    Ok(document) => {
                        document.value.validate()?;
                        if document.source.content_digest != document.value.digest()? {
                            return Err(CacheError::InvalidManifest(format!(
                                "destination repository batch {path:?} is not canonical"
                            )));
                        }
                        Some(document.value)
                    }
                    Err(CacheError::NotFound(_)) => None,
                    Err(error) => return Err(error),
                };
                batches.insert(transaction_id.clone(), batch);
            }
            proven = batches
                .get(&transaction_id)
                .and_then(Option::as_ref)
                .is_some_and(|batch| {
                    batch.batch_id.0 == transaction_id
                        && batch.destination == destination
                        && batch.family == family
                        && batch.artifacts.iter().any(|artifact| {
                            artifact.semantic_digest == manifest.semantic_digest
                                && artifact.canonical_payload_digest
                                    == entry.canonical_payload_digest
                                && artifact.manifest_digest == entry.manifest_digest
                                && entry.transport_digests.contains(&artifact.transport_digest)
                                && artifact.achieved_assurance >= draft.achieved_assurance
                                && artifact.producer_toolkit_version
                                    == entry.producer_toolkit_version
                        })
                });
            if proven {
                break;
            }
        }
        if proven {
            already_present = already_present.saturating_add(1);
        } else {
            pending.push(*draft);
        }
    }
    Ok(DestinationDraftSelection {
        pending,
        already_present,
    })
}

#[derive(Clone, Debug)]
struct PreparedFamilyCandidate<'a> {
    draft: &'a CanonicalProductionDraft,
    artifact: ManagedPreparedArtifactPublication,
    immutable_files: Vec<TransportPart>,
}

fn stage_family_document(
    staging_root: &Path,
    path: String,
    bytes: Vec<u8>,
    files: &mut BTreeMap<String, TransportPart>,
) -> Result<(), CacheError> {
    let digest = ContentDigest::sha256(&bytes);
    let target = staging_root.join(&path);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if target.exists() {
        let existing = std::fs::read(&target)?;
        let actual = ContentDigest::sha256(&existing);
        if actual != digest {
            return Err(CacheError::DigestMismatch {
                expected: digest.to_string(),
                actual: actual.to_string(),
            });
        }
    } else {
        std::fs::write(&target, &bytes)?;
    }
    files.entry(path.clone()).or_insert(TransportPart {
        sequence: 0,
        repository_path: path,
        size_bytes: bytes.len() as u64,
        content_digest: digest,
    });
    Ok(())
}

fn prepare_family_candidates<'a>(
    pending_drafts: &[&'a CanonicalProductionDraft],
    destination: PublicationDestination,
    context: &ManagedPublicationPlanningContext,
    sessions: &BTreeMap<PublicationDestination, AuthenticatedGitHubSession>,
    staging_root: &Path,
) -> Result<Vec<PreparedFamilyCandidate<'a>>, CacheError> {
    let mut candidates = Vec::with_capacity(pending_drafts.len());
    for draft in pending_drafts {
        let sessions_for_target = BTreeMap::from([(destination, sessions[&destination].clone())]);
        let artifact = prepare_managed_artifact_publication(draft, context, &sessions_for_target)?;
        crate::publication_staging::validate_public_documents(
            artifact
                .coordinated
                .journal
                .as_ref()
                .expect("authorized journal"),
            destination,
            &artifact.bundles[&destination],
            if destination == PublicationDestination::Public {
                Some(&artifact.policy.sanitizer)
            } else {
                None
            },
        )?;

        let bundle = &artifact.bundles[&destination];
        let journal = artifact
            .coordinated
            .journal
            .as_ref()
            .expect("authorized journal");
        let manifest_digest = bundle.manifest.digest()?;
        let transport_digest = bundle.encoding.digest()?;
        let manifest_path = format!(
            "manifests/{}/{}.json",
            &journal.semantic_digest.0[..2],
            manifest_digest.0
        );
        let encoding_path = format!(
            "encodings/{}/{}.json",
            &journal.payload_digest.0[..2],
            transport_digest.0
        );
        let mut files = BTreeMap::new();
        stage_family_document(
            staging_root,
            manifest_path,
            crate::protocol::canonical_json_bytes(&bundle.manifest)?,
            &mut files,
        )?;
        stage_family_document(
            staging_root,
            encoding_path,
            crate::protocol::canonical_json_bytes(&bundle.encoding)?,
            &mut files,
        )?;
        for (attestation, digest) in bundle.validation_attestations.iter().zip(
            bundle
                .validation_attestations
                .iter()
                .map(|item| item.digest())
                .collect::<Result<Vec<_>, _>>()?,
        ) {
            stage_family_document(
                staging_root,
                format!("attestations/validation/{}.json", digest.0),
                crate::protocol::canonical_json_bytes(attestation)?,
                &mut files,
            )?;
        }
        for part in &bundle.encoding.ordered_parts {
            let source = part
                .repository_path
                .split('/')
                .fold(draft.staged_parts_root.to_path_buf(), |path, component| {
                    path.join(component)
                });
            let target = part
                .repository_path
                .split('/')
                .fold(staging_root.to_path_buf(), |path, component| {
                    path.join(component)
                });
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !target.exists() {
                std::fs::copy(&source, &target)?;
            }
            files
                .entry(part.repository_path.clone())
                .or_insert_with(|| part.clone());
        }
        candidates.push(PreparedFamilyCandidate {
            draft,
            artifact,
            immutable_files: files.into_values().collect(),
        });
    }
    Ok(candidates)
}

fn prepared_candidate_files(candidates: &[PreparedFamilyCandidate<'_>]) -> Vec<TransportPart> {
    candidates
        .iter()
        .flat_map(|candidate| candidate.immutable_files.iter().cloned())
        .map(|part| (part.repository_path.clone(), part))
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect()
}

/// Publish a family as repository-level byte batches. Logical manifests and
/// index entries remain one-per-artifact; only the Git transport transaction is
/// shared. This is the normal path for families containing more than one
/// artifact.
#[allow(clippy::too_many_arguments)]
fn execute_family_batch_publication(
    drafts: &[&CanonicalProductionDraft],
    destination: PublicationDestination,
    owner: &str,
    principal: &str,
    sessions: &BTreeMap<PublicationDestination, AuthenticatedGitHubSession>,
    journal_root: &Path,
    resources: &xc_core::ResourcePolicy,
    replace_existing_semantic: bool,
    event_unix_seconds: u64,
) -> Result<Vec<ManagedPublicationExecutionReport>, CacheError> {
    let first = drafts.first().ok_or_else(|| {
        CacheError::InvalidTransition("family batch publication received no drafts".to_owned())
    })?;
    let authorized = repository(owner, &first.family, destination);
    let repository_url = repository_url(owner, &first.family, destination);
    let shard = shard_id(&first.family, destination);
    let cancellation = xc_core::CancellationToken::for_policy(resources);
    let staging_root = journal_root
        .join("family-batches")
        .join(&first.family)
        .join(visibility_name(destination));
    std::fs::create_dir_all(&staging_root)?;
    let remote = crate::GitCliRemoteStore::new(
        journal_root
            .join("git-transport")
            .join("family-batch")
            .join(&first.family)
            .join(visibility_name(destination)),
        staging_root.clone(),
        format!("Xcelerator Toolkit ({principal})"),
        "xcelerator-toolkit@users.noreply.github.com",
    )?
    .with_resource_policy(resources.clone());
    let session = sessions.get(&destination).ok_or_else(|| {
        CacheError::Authentication("family batch is missing its write session".to_owned())
    })?;
    session.require_write_for(principal, &authorized)?;
    verify_bootstrap_registry_route(&remote, owner, &first.family, destination, &cancellation)?;
    let observed_head = remote.read_ref(&repository_url, "main")?;
    let mut pending_drafts = drafts.to_vec();
    if !replace_existing_semantic {
        let selection = select_missing_destination_drafts(
            &remote,
            &repository_url,
            &observed_head,
            &first.family,
            destination,
            &pending_drafts,
            &cancellation,
        )?;
        if selection.already_present > 0 {
            eprintln!(
                "publication destination {} family {}: {} already present, {} pending",
                visibility_name(destination),
                first.family,
                selection.already_present,
                selection.pending.len()
            );
        }
        pending_drafts = selection.pending;
        if pending_drafts.is_empty() {
            remote.cleanup_session(&repository_url)?;
            return Ok(Vec::new());
        }
    }
    let private_lease_policy = crate::PrivatePublicationLeasePolicy::default();
    let preparation_started = std::time::Instant::now();
    let preparation_ledger = if remote
        .immutable_path_digest(
            &repository_url,
            &observed_head,
            crate::DEFAULT_CAPACITY_LEDGER_PATH,
        )?
        .is_some()
    {
        read_capacity_ledger(&remote, &repository_url, &observed_head, &cancellation)?
    } else {
        // Derive the same conservative ledger that shard initialization will
        // commit later, but do so read-only. This lets all immutable artifact
        // preparation and Git object hashing finish before the one private
        // publication lease is acquired, even for a brand-new shard.
        reconciled_capacity_ledger(
            &remote,
            &repository_url,
            &observed_head,
            &shard,
            &cancellation,
        )?
    };
    if preparation_ledger.shard_id != shard {
        return Err(CacheError::InvalidManifest(
            "family batch capacity ledger belongs to another shard".to_owned(),
        ));
    }
    if replace_existing_semantic {
        for draft in &pending_drafts {
            preflight_producer_monotonicity(
                &remote,
                &repository_url,
                &observed_head,
                &first.family,
                &draft.manifest.semantic_digest,
                &draft.manifest.producer_toolkit_version,
                &cancellation,
            )?;
        }
    }
    let preparation_context = ManagedPublicationPlanningContext {
        owner: owner.to_owned(),
        principal: principal.to_owned(),
        target: if destination == PublicationDestination::Private {
            PublicationTarget::Private
        } else {
            PublicationTarget::Public
        },
        target_heads: BTreeMap::from([(destination, observed_head.clone())]),
        capacity_ledgers: BTreeMap::from([(destination, preparation_ledger)]),
        event_unix_seconds,
    };
    let mut prepared_candidates = prepare_family_candidates(
        &pending_drafts,
        destination,
        &preparation_context,
        sessions,
        &staging_root,
    )?;
    let prepared_files = prepared_candidate_files(&prepared_candidates);
    remote.prepare_staged_parts(&repository_url, &prepared_files)?;
    eprintln!(
        "publication family {}: prepared {} candidate(s) and {} immutable file(s) before lock in {:.3}s",
        first.family,
        prepared_candidates.len(),
        prepared_files.len(),
        preparation_started.elapsed().as_secs_f64()
    );

    let mut private_lease = if destination == PublicationDestination::Private {
        let lease_owner = crate::PrivatePublicationLeaseOwner::for_current_process(
            &repository_url,
            principal,
            event_unix_seconds,
        )?;
        Some(crate::acquire_private_publication_lease(
            &remote,
            &repository_url,
            "main",
            &lease_owner,
            &format!("pending-family-{}-{event_unix_seconds}", first.family),
            &staging_root,
            &cancellation,
            &private_lease_policy,
        )?)
    } else {
        None
    };
    let mut active_session = session.clone();
    let publication_result = (|| -> Result<Vec<ManagedPublicationExecutionReport>, CacheError> {
        if active_session.requires_refresh(60)? {
            let refreshed =
                crate::GitHubCredentialApiProbe::default().probe_repository(&authorized)?;
            if refreshed.evidence().principal != principal {
                return Err(CacheError::Authentication(
                    "post-lock GitHub permission resolved a different principal".to_owned(),
                ));
            }
            active_session = refreshed;
        }
        active_session.require_write_for(principal, &authorized)?;
        // A clean bootstrap shard has no ledger or index yet. Initialize those
        // sidecars before planning the first repository batch; subsequent runs
        // simply reuse the verified sidecars.
        let (head, mut ledger) = ensure_managed_shard_sidecars(
            &remote,
            &active_session,
            &repository_url,
            &authorized,
            "main",
            &shard,
            &first.family,
            &pending_drafts[0].manifest.semantic_digest.0[..2],
            &staging_root,
            resources,
            &cancellation,
            private_lease.as_mut(),
            &private_lease_policy,
        )?;
        if !replace_existing_semantic && head != observed_head {
            let selection = select_missing_destination_drafts(
                &remote,
                &repository_url,
                &head,
                &first.family,
                destination,
                &pending_drafts,
                &cancellation,
            )?;
            if selection.already_present > 0 {
                eprintln!(
                    "publication destination {} family {}: {} became present before mutation, {} pending",
                    visibility_name(destination),
                    first.family,
                    selection.already_present,
                    selection.pending.len()
                );
            }
            pending_drafts = selection.pending;
            if pending_drafts.is_empty() {
                return Ok(Vec::new());
            }
            let retained = pending_drafts
                .iter()
                .map(|draft| draft.manifest.semantic_digest.clone())
                .collect::<BTreeSet<_>>();
            prepared_candidates
                .retain(|candidate| retained.contains(&candidate.draft.manifest.semantic_digest));
        }
        if let Some(lease) = private_lease.as_mut() {
            crate::renew_private_publication_lease(
                &remote,
                lease,
                &head,
                &staging_root,
                &private_lease_policy,
            )?;
        }
        if replace_existing_semantic {
            for draft in &pending_drafts {
                preflight_producer_monotonicity(
                    &remote,
                    &repository_url,
                    &head,
                    &first.family,
                    &draft.manifest.semantic_digest,
                    &draft.manifest.producer_toolkit_version,
                    &cancellation,
                )?;
            }
        }
        if ledger.shard_id != shard {
            return Err(CacheError::InvalidManifest(
                "family batch capacity ledger belongs to a different shard".to_owned(),
            ));
        }
        if prepared_candidates.len() != pending_drafts.len() {
            return Err(CacheError::InvalidTransition(
                "destination recheck did not retain one prepared candidate per pending draft"
                    .to_owned(),
            ));
        }
        let prepared = prepared_candidates
            .iter()
            .map(|candidate| &candidate.artifact)
            .collect::<Vec<_>>();
        let mut files = prepared_candidate_files(&prepared_candidates)
            .into_iter()
            .map(|part| (part.repository_path.clone(), part))
            .collect::<BTreeMap<_, _>>();
        let mut artifacts = Vec::with_capacity(prepared.len());
        for artifact in &prepared {
            let bundle = &artifact.bundles[&destination];
            let journal = artifact
                .coordinated
                .journal
                .as_ref()
                .expect("authorized journal");
            let manifest_digest = bundle.manifest.digest()?;
            let transport_digest = bundle.encoding.digest()?;
            let manifest_path = format!(
                "manifests/{}/{}.json",
                &journal.semantic_digest.0[..2],
                manifest_digest.0
            );
            let mut provenance_evidence_digests = bundle
                .validation_attestations
                .iter()
                .map(|attestation| attestation.digest())
                .collect::<Result<Vec<_>, _>>()?;
            provenance_evidence_digests.sort();
            artifacts.push(crate::RepositoryBatchArtifact {
                semantic_digest: journal.semantic_digest.clone(),
                canonical_payload_digest: journal.payload_digest.clone(),
                manifest_digest,
                transport_digest,
                manifest_path,
                achieved_assurance: bundle.achieved_assurance,
                producer_toolkit_version: bundle.manifest.producer_toolkit_version.clone(),
                provenance_evidence_digests,
            });
        }
        let policy_digest = prepared[0].policy.digest()?;
        let batch = crate::RepositoryPublicationBatch::new(
            destination,
            first.family.clone(),
            principal,
            authorized.clone(),
            "main",
            policy_digest,
            private_lease
                .as_ref()
                .map(|lease| lease.lock.fencing_generation),
            artifacts,
            event_unix_seconds,
        )?;
        if let Some(lease) = private_lease.as_mut() {
            lease.set_transaction_id(batch.batch_id.0.clone());
        }
        match crate::RemoteShardReader::new(&remote, 16 * 1024 * 1024)?
            .read_json::<crate::RepositoryPublicationBatch>(
            &repository_url,
            &head,
            &batch.repository_path(),
            &cancellation,
        ) {
            Ok(existing) => {
                existing.value.validate()?;
                if existing.value.batch_id != batch.batch_id {
                    return Err(CacheError::InvalidManifest(
                        "remote batch path contains a different immutable batch identity"
                            .to_owned(),
                    ));
                }
                return Ok(vec![ManagedPublicationExecutionReport {
                    transaction_id: batch.batch_id.0.clone(),
                    completed: true,
                    steps_executed: 0,
                    final_journal_digest: batch.batch_id.clone(),
                }]);
            }
            Err(CacheError::NotFound(_)) => {}
            Err(error) => return Err(error),
        }
        let batch_bytes = crate::protocol::canonical_json_bytes(&batch)?;
        let batch_path = batch.repository_path();
        files.insert(
            batch_path.clone(),
            TransportPart {
                sequence: 0,
                repository_path: batch_path,
                size_bytes: batch_bytes.len() as u64,
                content_digest: ContentDigest::sha256(&batch_bytes),
            },
        );
        let batch_file_path = staging_root.join(batch.repository_path());
        if let Some(parent) = batch_file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(batch_file_path, batch_bytes)?;

        let semantic_prefixes = prepared
            .iter()
            .map(|artifact| {
                artifact
                    .coordinated
                    .journal
                    .as_ref()
                    .unwrap()
                    .semantic_digest
                    .0[..2]
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        for prefix in semantic_prefixes {
            let path = format!("indexes/{}/{prefix}.json", first.family);
            let mut entries = match crate::RemoteShardReader::new(&remote, 16 * 1024 * 1024)?
                .read_json::<crate::ShardIndexPartition>(
                &repository_url,
                &head,
                &path,
                &cancellation,
            ) {
                Ok(current) => current.value.entries,
                Err(CacheError::NotFound(_)) => Vec::new(),
                Err(error) => return Err(error),
            };
            for artifact in &prepared {
                let journal = artifact.coordinated.journal.as_ref().unwrap();
                if journal.semantic_digest.0.get(..2) != Some(prefix.as_str()) {
                    continue;
                }
                let bundle = &artifact.bundles[&destination];
                let manifest_digest = bundle.manifest.digest()?;
                let transport_digest = bundle.encoding.digest()?;
                entries.retain(|entry| entry.semantic_digest != journal.semantic_digest);
                entries.push(crate::ShardIndexEntry {
                    semantic_digest: journal.semantic_digest.clone(),
                    canonical_payload_digest: journal.payload_digest.clone(),
                    manifest_digest,
                    achieved_assurance: bundle.achieved_assurance,
                    disposition: bundle.disposition,
                    producer_toolkit_version: bundle.manifest.producer_toolkit_version.clone(),
                    minimum_reader_version: bundle.manifest.minimum_reader_version.clone(),
                    transport_digests: vec![transport_digest],
                    publication_transaction_id: batch.batch_id.0.clone(),
                });
            }
            let index = crate::ShardIndexPartition::rebuild(first.family.clone(), prefix, entries)?;
            let bytes = crate::protocol::canonical_json_bytes(&index)?;
            files.insert(
                path.clone(),
                TransportPart {
                    sequence: 0,
                    repository_path: path.clone(),
                    size_bytes: bytes.len() as u64,
                    content_digest: ContentDigest::sha256(&bytes),
                },
            );
            let target = staging_root.join(&path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(target, bytes)?;
        }
        // Maintain a compact compatibility inventory for cross-N CCM
        // continuation. The primary shard index is hash-partitioned and
        // deliberately does not support semantic range queries; without this
        // derived inventory a reader would have to open every eigenpair
        // manifest merely to find the nearest lower N.
        let mut continuation_updates = BTreeMap::<
            String,
            (
                crate::CcmEigenpairContinuationQuery,
                Vec<crate::CcmEigenpairContinuationEntry>,
            ),
        >::new();
        for (draft, artifact) in pending_drafts.iter().zip(&prepared) {
            let bundle = &artifact.bundles[&destination];
            let manifest_digest = bundle.manifest.digest()?;
            let Some((query, entry)) = crate::ccm_eigenpair_continuation_entry(
                &bundle.manifest.semantic_key,
                &draft.source_logical_key,
                manifest_digest,
                bundle.achieved_assurance,
                bundle.disposition,
                bundle.manifest.producer_toolkit_version.clone(),
                bundle.manifest.minimum_reader_version.clone(),
            )?
            else {
                continue;
            };
            let path = query.repository_path()?;
            continuation_updates
                .entry(path)
                .or_insert_with(|| (query, Vec::new()))
                .1
                .push(entry);
        }
        for (path, (query, additions)) in continuation_updates {
            let mut entries = match crate::RemoteShardReader::new(&remote, 16 * 1024 * 1024)?
                .read_json::<crate::CcmEigenpairContinuationIndex>(
                &repository_url,
                &head,
                &path,
                &cancellation,
            ) {
                Ok(current) => {
                    current.value.validate()?;
                    current.value.entries
                }
                Err(CacheError::NotFound(_)) => Vec::new(),
                Err(error) => return Err(error),
            };
            for addition in additions {
                entries.retain(|entry| entry.semantic_digest != addition.semantic_digest);
                entries.push(addition);
            }
            let inventory = crate::CcmEigenpairContinuationIndex::rebuild(&query, entries)?;
            let bytes = crate::protocol::canonical_json_bytes(&inventory)?;
            files.insert(
                path.clone(),
                TransportPart {
                    sequence: 0,
                    repository_path: path.clone(),
                    size_bytes: bytes.len() as u64,
                    content_digest: ContentDigest::sha256(&bytes),
                },
            );
            let target = staging_root.join(&path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(target, bytes)?;
        }
        // Commit immutable payload first. Metadata and live indexes are kept
        // together at the end so an interrupted multi-gigabyte publication
        // cannot make an artifact discoverable before all of its objects are
        // reachable. Every individual commit receives its own ledger snapshot.
        let parts = files.into_values().collect::<Vec<_>>();
        let ordered_parts = parts
            .iter()
            .filter(|part| part.repository_path.starts_with("objects/"))
            .cloned()
            .chain(
                parts
                    .iter()
                    .filter(|part| !part.repository_path.starts_with("objects/"))
                    .cloned(),
            )
            .enumerate()
            .map(|(sequence, mut part)| {
                part.sequence = sequence as u64;
                part
            })
            .collect::<Vec<_>>();
        let mut batch_policy = crate::TransportPolicy::default();
        // Reserve ample room for the per-commit ledger document while keeping
        // the complete Git push below the established one-gigabyte boundary.
        batch_policy.maximum_batch_payload_bytes = batch_policy
            .maximum_batch_payload_bytes
            .saturating_sub(1024 * 1024);
        let batches = crate::plan_publication_batches(&ordered_parts, &batch_policy)?;
        let total_batches = batches.len();
        let mut current_head = head;
        let mut steps = 0;
        for batch_plan in batches {
            let mut commit_parts = Vec::with_capacity(batch_plan.parts.len() + 1);
            for part in batch_plan.parts {
                match remote.immutable_path_digest(
                    &repository_url,
                    &current_head,
                    &part.repository_path,
                )? {
                    Some(existing) if existing == part.content_digest => {
                        // An earlier transaction or an interrupted retry
                        // already committed these exact bytes.
                        continue;
                    }
                    Some(existing) if part.repository_path.starts_with("objects/") => {
                        return Err(CacheError::DigestMismatch {
                            expected: part.content_digest.to_string(),
                            actual: existing.to_string(),
                        });
                    }
                    Some(_) | None => {}
                }
                commit_parts.push(part);
            }
            if commit_parts.is_empty() {
                continue;
            }
            advance_capacity_ledger_for_repository_batch(
                &mut ledger,
                &batch.batch_id,
                batch_plan.sequence,
                &current_head,
                &commit_parts,
            )?;
            let ledger_bytes = crate::protocol::canonical_json_bytes(&ledger)?;
            let ledger_target = staging_root.join(crate::DEFAULT_CAPACITY_LEDGER_PATH);
            if let Some(parent) = ledger_target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&ledger_target, &ledger_bytes)?;
            commit_parts.push(TransportPart {
                sequence: commit_parts.len() as u64,
                repository_path: crate::DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                size_bytes: ledger_bytes.len() as u64,
                content_digest: ContentDigest::sha256(&ledger_bytes),
            });
            for (sequence, part) in commit_parts.iter_mut().enumerate() {
                part.sequence = sequence as u64;
            }
            if active_session.requires_refresh(60)? {
                let refreshed =
                    crate::GitHubCredentialApiProbe::default().probe_repository(&authorized)?;
                if refreshed.evidence().principal != principal {
                    return Err(CacheError::Authentication(
                        "publication batch permission refresh resolved a different principal"
                            .to_owned(),
                    ));
                }
                active_session = refreshed;
            }
            active_session.require_write_for(principal, &authorized)?;
            let request = crate::RemoteCommitRequest {
                repository: repository_url.clone(),
                branch: "main".to_owned(),
                expected_head: current_head.clone(),
                message: format!(
                    "batch publish {} artifacts ({}/{})",
                    pending_drafts.len(),
                    batch_plan.sequence + 1,
                    total_batches
                ),
                parts: commit_parts,
                delete_paths: Vec::new(),
            };
            if let Some(lease) = private_lease.as_mut() {
                current_head = crate::commit_private_batch_atomically(
                    &remote,
                    lease,
                    request,
                    &staging_root,
                    &private_lease_policy,
                )?;
                steps += 1;
            } else {
                match remote.compare_and_swap_commit(&request)? {
                    crate::CompareAndSwapResult::Committed { commit_id } => {
                        current_head = commit_id;
                        steps += 1;
                    }
                    crate::CompareAndSwapResult::RefConflict {
                        current_head: remote_head,
                    } => {
                        return Err(CacheError::InvalidTransition(format!(
                            "family batch publication conflicted at {remote_head}"
                        )))
                    }
                }
            }
        }
        Ok(vec![ManagedPublicationExecutionReport {
            transaction_id: batch.batch_id.0.clone(),
            completed: true,
            steps_executed: steps,
            final_journal_digest: batch.batch_id.clone(),
        }])
    })();
    let completed = publication_result.is_ok();
    let release_result = private_lease.as_ref().map(|lease| {
        crate::release_private_publication_lease(&remote, lease, &staging_root, completed)
    });
    match (publication_result, release_result) {
        (Ok(report), None | Some(Ok(()))) => Ok(report),
        (Ok(_), Some(Err(error))) => Err(error),
        (Err(error), None | Some(Ok(()))) => Err(error),
        (Err(error), Some(Err(release_error))) => {
            eprintln!(
                "private publication failed and its lease release also failed: {release_error}"
            );
            Err(error)
        }
    }
}

fn bounded_prefix_bytes(
    remote: &dyn crate::RemoteGitStore,
    repository: &str,
    revision: &str,
    prefix: &str,
    canonical_objects_only: bool,
    cancellation: &xc_core::CancellationToken,
) -> Result<u64, CacheError> {
    let listing = remote.list_committed_paths(
        repository,
        revision,
        prefix,
        1_000_000,
        128 * 1024 * 1024,
        cancellation,
    )?;
    listing
        .paths
        .iter()
        .filter(|path| !canonical_objects_only || canonical_payload_object_path(path))
        .try_fold(0u64, |total, path| {
            let mut sink = std::io::sink();
            let report = remote.read_committed_path(
                repository,
                revision,
                path,
                crate::GITHUB_HARD_FILE_BOUNDARY_BYTES - 1,
                cancellation,
                &mut sink,
            )?;
            total.checked_add(report.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit("bootstrap shard byte accounting exceeds u64".to_owned())
            })
        })
}

fn canonical_payload_object_path(path: &str) -> bool {
    let mut components = path.split('/');
    let (Some("objects"), Some("sha256"), Some(prefix), Some(file), None) = (
        components.next(),
        components.next(),
        components.next(),
        components.next(),
        components.next(),
    ) else {
        return false;
    };
    let Some(digest) = file.strip_suffix(".part") else {
        return false;
    };
    digest.len() == 64
        && prefix.len() == 2
        && prefix == &digest[..2]
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Serialize)]
struct BootstrapReconciliation<'a> {
    schema_version: u32,
    shard_id: &'a str,
    revision: &'a str,
    observed_payload_bytes: u64,
    observed_metadata_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapRegistry {
    schema_version: u32,
    repository: String,
    default_branch: String,
    families: Vec<BootstrapRegistryFamily>,
    #[serde(default)]
    separate_visibility_inventory: Option<bool>,
    #[serde(default)]
    visibility: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapRegistryFamily {
    family: String,
    current_writable_shard: String,
    metadata: String,
}

fn verify_bootstrap_registry_route(
    remote: &dyn crate::RemoteGitStore,
    owner: &str,
    family: &str,
    destination: PublicationDestination,
    cancellation: &xc_core::CancellationToken,
) -> Result<(), CacheError> {
    let visibility = visibility_name(destination);
    let authorized_registry = format!("{owner}/xcelerator-cache-{visibility}-registry");
    let registry_url = format!("https://github.com/{authorized_registry}.git");
    let revision = remote.read_ref(&registry_url, "main")?;
    let mut bytes = Vec::new();
    remote.read_committed_path(
        &registry_url,
        &revision,
        "registry.json",
        4 * 1024 * 1024,
        cancellation,
        &mut bytes,
    )?;
    let registry: BootstrapRegistry = serde_json::from_slice(&bytes)?;
    if registry.schema_version != 1
        || registry.repository != authorized_registry
        || registry.default_branch != "main"
        || registry.families.iter().any(|entry| {
            entry.family.trim().is_empty()
                || entry.current_writable_shard.trim().is_empty()
                || entry.metadata.trim().is_empty()
        })
        || registry
            .separate_visibility_inventory
            .is_some_and(|separate| !separate)
        || registry
            .visibility
            .as_ref()
            .is_some_and(|declared| declared != visibility)
    {
        return Err(CacheError::InvalidManifest(format!(
            "{visibility} bootstrap registry identity or family entries are invalid"
        )));
    }
    let expected_shard = repository(owner, family, destination);
    let route = registry
        .families
        .iter()
        .find(|entry| entry.family == family)
        .ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "{visibility} registry has no route for family {family:?}"
            ))
        })?;
    if route.current_writable_shard != expected_shard {
        return Err(CacheError::NoWritableShard(format!(
            "{visibility} registry routes {family:?} to {:?}, not expected {:?}",
            route.current_writable_shard, expected_shard
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ensure_managed_shard_sidecars(
    remote: &dyn crate::RemoteGitStore,
    session: &AuthenticatedGitHubSession,
    repository: &str,
    authorized_repository: &str,
    branch: &str,
    shard_id: &str,
    _family: &str,
    _semantic_prefix: &str,
    staging_root: &Path,
    resources: &xc_core::ResourcePolicy,
    cancellation: &xc_core::CancellationToken,
    mut private_lease: Option<&mut crate::PrivatePublicationLease>,
    private_lease_policy: &crate::PrivatePublicationLeasePolicy,
) -> Result<(String, CapacityLedger), CacheError> {
    session.require_write_for(session.evidence().principal.as_str(), authorized_repository)?;
    for _ in 0..4 {
        let head = remote.read_ref(repository, branch)?;
        let ledger_digest =
            remote.immutable_path_digest(repository, &head, crate::DEFAULT_CAPACITY_LEDGER_PATH)?;
        if ledger_digest.is_some() {
            let ledger = read_capacity_ledger(remote, repository, &head, cancellation)?;
            if ledger.shard_id != shard_id {
                return Err(CacheError::InvalidManifest(format!(
                    "live capacity ledger belongs to {:?}, expected {shard_id:?}",
                    ledger.shard_id
                )));
            }
            return Ok((head, ledger));
        }

        let ledger = reconciled_capacity_ledger(remote, repository, &head, shard_id, cancellation)?;
        let mut parts = Vec::new();
        let mut staged_bytes = 0u64;
        parts.push(crate::publication_staging::stage_publication_bytes(
            staging_root,
            crate::DEFAULT_CAPACITY_LEDGER_PATH,
            &crate::protocol::canonical_json_bytes(&ledger)?,
            resources,
            cancellation,
            &mut staged_bytes,
        )?);
        parts.sort_by(|left, right| left.repository_path.cmp(&right.repository_path));
        for (sequence, part) in parts.iter_mut().enumerate() {
            part.sequence = sequence as u64;
        }
        let staged_paths = parts
            .iter()
            .map(|part| {
                part.repository_path
                    .split('/')
                    .fold(staging_root.to_owned(), |path, component| {
                        path.join(component)
                    })
            })
            .collect::<Vec<_>>();
        let verification_parts = parts.clone();
        let request = crate::RemoteCommitRequest {
            repository: repository.to_owned(),
            branch: branch.to_owned(),
            expected_head: head.clone(),
            message: "initialize Xcelerator v0.13.0 shard capacity ledger".to_owned(),
            parts,
            delete_paths: Vec::new(),
        };
        let outcome = if let Some(lease) = private_lease.as_deref_mut() {
            crate::CompareAndSwapResult::Committed {
                commit_id: crate::commit_private_batch_atomically(
                    remote,
                    lease,
                    request,
                    staging_root,
                    private_lease_policy,
                )?,
            }
        } else {
            remote.compare_and_swap_commit(&request)?
        };
        for path in staged_paths {
            if path.is_file() {
                std::fs::remove_file(path)?;
            }
        }
        match outcome {
            crate::CompareAndSwapResult::Committed { commit_id } => {
                for part in &verification_parts {
                    remote.verify_committed_part(repository, &commit_id, part)?;
                }
                let ledger = read_capacity_ledger(remote, repository, &commit_id, cancellation)?;
                return Ok((commit_id, ledger));
            }
            crate::CompareAndSwapResult::RefConflict { .. } => continue,
        }
    }
    Err(CacheError::InvalidTransition(format!(
        "could not initialize shard {authorized_repository} after repeated ref conflicts"
    )))
}

fn reconciled_capacity_ledger(
    remote: &dyn crate::RemoteGitStore,
    repository: &str,
    revision: &str,
    shard_id: &str,
    cancellation: &xc_core::CancellationToken,
) -> Result<CapacityLedger, CacheError> {
    let payload_bytes =
        bounded_prefix_bytes(remote, repository, revision, "objects", true, cancellation)?;
    let metadata_bytes = [
        "manifests",
        "encodings",
        "transactions",
        "indexes",
        "attestations",
        "revocations",
    ]
    .into_iter()
    .try_fold(0u64, |total, prefix| {
        total
            .checked_add(bounded_prefix_bytes(
                remote,
                repository,
                revision,
                prefix,
                false,
                cancellation,
            )?)
            .ok_or_else(|| {
                CacheError::ResourceLimit("bootstrap metadata accounting exceeds u64".to_owned())
            })
    })?;
    let reconciliation_digest = canonical_digest(&BootstrapReconciliation {
        schema_version: 1,
        shard_id,
        revision,
        observed_payload_bytes: payload_bytes,
        observed_metadata_bytes: metadata_bytes,
    })?;
    let ledger = CapacityLedger {
        schema_version: 1,
        shard_id: shard_id.to_owned(),
        hard_capacity_bytes: crate::GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
        warning_reserve_bytes: 5_000_000_000,
        first_seen_immutable_payload_bytes: payload_bytes,
        manifest_index_receipt_bytes: metadata_bytes,
        // Conservatively reserve another full reachable snapshot for
        // pre-ledger history rather than claiming exact old history.
        estimated_history_bytes: payload_bytes.saturating_add(metadata_bytes),
        emergency_reserve_bytes: 0,
        abandoned_reachable_bytes: 0,
        last_reconciled_commit: revision.to_owned(),
        reconciliation_digest,
    };
    ledger.validate()?;
    Ok(ledger)
}

#[derive(Serialize)]
struct ManagedValidationEvidence<'a> {
    schema_version: u32,
    manifest_digest: &'a ContentDigest,
    payload_digest: &'a ContentDigest,
    achieved_assurance: ArtifactAssuranceState,
    assurance_evidence_digests: &'a [ContentDigest],
}

fn destinations(target: PublicationTarget) -> Result<Vec<PublicationDestination>, CacheError> {
    match target {
        PublicationTarget::None => Err(CacheError::InvalidManifest(
            "managed publication requires a non-none target".to_owned(),
        )),
        PublicationTarget::Private => Ok(vec![PublicationDestination::Private]),
        PublicationTarget::Public => Ok(vec![PublicationDestination::Public]),
        PublicationTarget::Both => Ok(vec![
            PublicationDestination::Private,
            PublicationDestination::Public,
        ]),
    }
}

fn visibility(destination: PublicationDestination) -> CacheVisibility {
    match destination {
        PublicationDestination::Private => CacheVisibility::Private,
        PublicationDestination::Public => CacheVisibility::Public,
    }
}

fn visibility_name(destination: PublicationDestination) -> &'static str {
    match destination {
        PublicationDestination::Private => "private",
        PublicationDestination::Public => "public",
    }
}

fn repository(owner: &str, family: &str, destination: PublicationDestination) -> String {
    format!(
        "{owner}/xcelerator-cache-{}-{family}-0001",
        visibility_name(destination)
    )
}

fn shard_id(family: &str, destination: PublicationDestination) -> String {
    format!("{}-{family}-0001", visibility_name(destination))
}

fn target_manifest(
    draft: &CanonicalProductionDraft,
    destination: PublicationDestination,
) -> Result<crate::CanonicalArtifactManifest, CacheError> {
    let mut manifest = destination_neutral_manifest(draft)?;
    manifest.assumptions.push(format!(
        "publication_visibility={}",
        visibility_name(destination)
    ));
    manifest.assumptions.sort();
    manifest.assumptions.dedup();
    manifest.validate()?;
    Ok(manifest)
}

fn destination_neutral_manifest(
    draft: &CanonicalProductionDraft,
) -> Result<crate::CanonicalArtifactManifest, CacheError> {
    let mut manifest = draft.manifest.clone();
    manifest
        .assumptions
        .retain(|assumption| !assumption.starts_with("publication_visibility="));
    manifest.assumptions.sort();
    manifest.assumptions.dedup();
    manifest.validate()?;
    Ok(manifest)
}

fn remap_destination_drafts_with_existing(
    drafts: &[CanonicalProductionDraft],
    destination: PublicationDestination,
    mut existing_dependency: impl FnMut(&crate::PayloadDependencyIdentity) -> Result<bool, CacheError>,
) -> Result<Vec<CanonicalProductionDraft>, CacheError> {
    type Identity = (String, ContentDigest, ContentDigest, ContentDigest);

    let mut source_identities = BTreeMap::<Identity, usize>::new();
    for (index, draft) in drafts.iter().enumerate() {
        draft.manifest.validate()?;
        let mut manifests = vec![draft.manifest.clone()];
        let neutral = destination_neutral_manifest(draft)?;
        if neutral != draft.manifest {
            manifests.push(neutral);
        }
        for manifest in manifests {
            let identity = (
                manifest.artifact_family.clone(),
                manifest.semantic_digest.clone(),
                manifest.digest()?,
                manifest.payload_digest.clone(),
            );
            if let Some(previous) = source_identities.insert(identity, index) {
                if previous != index {
                    return Err(CacheError::InvalidManifest(
                        "managed publication closure contains an ambiguous exact or destination-neutral artifact identity"
                            .to_owned(),
                    ));
                }
            }
        }
    }

    let mut remapped = vec![None::<CanonicalProductionDraft>; drafts.len()];
    let mut external_identities = BTreeSet::<Identity>::new();
    let mut remaining = (0..drafts.len()).collect::<BTreeSet<_>>();
    while !remaining.is_empty() {
        let mut progressed = false;
        for index in remaining.iter().copied().collect::<Vec<_>>() {
            let draft = &drafts[index];
            let mut dependencies =
                Vec::with_capacity(draft.manifest.canonical_payload.dependencies.len());
            let mut ready = true;
            for dependency in &draft.manifest.canonical_payload.dependencies {
                let identity = (
                    dependency.artifact_family.clone(),
                    dependency.semantic_digest.clone(),
                    dependency.manifest_digest.clone(),
                    dependency.payload_digest.clone(),
                );
                if let Some(dependency_index) = source_identities.get(&identity) {
                    let Some(destination_dependency) = remapped[*dependency_index].as_ref() else {
                        ready = false;
                        break;
                    };
                    dependencies.push(crate::PayloadDependencyIdentity {
                        artifact_family: destination_dependency.manifest.artifact_family.clone(),
                        semantic_digest: destination_dependency.manifest.semantic_digest.clone(),
                        manifest_digest: destination_dependency.manifest.digest()?,
                        payload_digest: destination_dependency.manifest.payload_digest.clone(),
                    });
                } else if existing_dependency(dependency)? {
                    external_identities.insert(identity);
                    dependencies.push(dependency.clone());
                } else {
                    return Err(CacheError::InvalidManifest(format!(
                        "managed publication closure is missing exact dependency {}/{}",
                        dependency.artifact_family, dependency.semantic_digest.0
                    )));
                }
            }
            if !ready {
                continue;
            }

            let mut destination_draft = draft.clone();
            destination_draft.manifest.canonical_payload.dependencies = dependencies;
            destination_draft.manifest.payload_digest =
                destination_draft.manifest.canonical_payload.digest()?;
            destination_draft.encoding.canonical_payload_digest =
                destination_draft.manifest.payload_digest.clone();
            destination_draft.manifest.transport_digests =
                vec![destination_draft.encoding.digest()?];
            destination_draft.manifest = target_manifest(&destination_draft, destination)?;
            destination_draft.manifest.validate()?;
            remapped[index] = Some(destination_draft);
            remaining.remove(&index);
            progressed = true;
        }
        if !progressed {
            return Err(CacheError::InvalidManifest(
                "managed publication dependency closure contains a cycle".to_owned(),
            ));
        }
    }

    let remapped = remapped
        .into_iter()
        .map(|draft| draft.expect("all destination drafts were remapped"))
        .collect::<Vec<_>>();
    let identities = remapped
        .iter()
        .map(|draft| {
            Ok((
                draft.manifest.artifact_family.clone(),
                draft.manifest.semantic_digest.clone(),
                draft.manifest.digest()?,
                draft.manifest.payload_digest.clone(),
            ))
        })
        .collect::<Result<BTreeSet<_>, CacheError>>()?
        .union(&external_identities)
        .cloned()
        .collect::<BTreeSet<_>>();
    for draft in &remapped {
        for dependency in &draft.manifest.canonical_payload.dependencies {
            let identity = (
                dependency.artifact_family.clone(),
                dependency.semantic_digest.clone(),
                dependency.manifest_digest.clone(),
                dependency.payload_digest.clone(),
            );
            if !identities.contains(&identity) {
                return Err(CacheError::InvalidManifest(format!(
                    "destination publication closure has a dangling dependency {}/{}",
                    dependency.artifact_family, dependency.semantic_digest.0
                )));
            }
        }
    }
    Ok(remapped)
}

#[cfg(test)]
fn remap_destination_drafts(
    drafts: &[CanonicalProductionDraft],
    destination: PublicationDestination,
) -> Result<Vec<CanonicalProductionDraft>, CacheError> {
    remap_destination_drafts_with_existing(drafts, destination, |_| Ok(false))
}

fn exact_destination_dependency_exists(
    remote: &dyn crate::RemoteGitStore,
    owner: &str,
    destination: PublicationDestination,
    dependency: &crate::PayloadDependencyIdentity,
    revisions: &mut BTreeMap<String, String>,
    resources: &xc_core::ResourcePolicy,
) -> Result<bool, CacheError> {
    let repository = repository_url(owner, &dependency.artifact_family, destination);
    let revision = if let Some(revision) = revisions.get(&repository) {
        revision.clone()
    } else {
        let revision = remote.read_ref(&repository, "main")?;
        revisions.insert(repository.clone(), revision.clone());
        revision
    };
    let cancellation = xc_core::CancellationToken::for_policy(resources);
    let reader = crate::RemoteShardReader::new(remote, 16 * 1024 * 1024)?;
    let manifest_path = format!(
        "manifests/{}/{}.json",
        &dependency.semantic_digest.0[..2],
        dependency.manifest_digest
    );
    let manifest = match reader.read_json::<crate::CanonicalArtifactManifest>(
        &repository,
        &revision,
        &manifest_path,
        &cancellation,
    ) {
        Ok(document) => document,
        Err(CacheError::NotFound(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    manifest.value.validate()?;
    if manifest.source.content_digest != dependency.manifest_digest
        || manifest.value.digest()? != dependency.manifest_digest
        || manifest.value.artifact_family != dependency.artifact_family
        || manifest.value.semantic_digest != dependency.semantic_digest
        || manifest.value.payload_digest != dependency.payload_digest
    {
        return Err(CacheError::InvalidManifest(format!(
            "destination dependency manifest {manifest_path:?} does not match its exact identity"
        )));
    }

    let index_path = format!(
        "indexes/{}/{}.json",
        dependency.artifact_family,
        &dependency.semantic_digest.0[..2]
    );
    let index = match reader.read_json::<crate::ShardIndexPartition>(
        &repository,
        &revision,
        &index_path,
        &cancellation,
    ) {
        Ok(document) => document,
        Err(CacheError::NotFound(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    index.value.validate()?;
    if index.value.family != dependency.artifact_family {
        return Err(CacheError::InvalidManifest(format!(
            "destination dependency index {index_path:?} belongs to the wrong family"
        )));
    }
    let exists = index
        .value
        .lookup(&dependency.semantic_digest)
        .any(|entry| {
            entry.disposition == ArtifactDisposition::Active
                && entry.manifest_digest == dependency.manifest_digest
                && entry.canonical_payload_digest == dependency.payload_digest
        });
    Ok(exists)
}

fn collect_leaf_paths(prefix: &str, value: &Value, paths: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                collect_leaf_paths(&path, value, paths);
            }
        }
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                collect_leaf_paths(&format!("{prefix}[{index}]"), value, paths);
            }
        }
        _ => {
            paths.insert(prefix.to_owned());
        }
    }
}

fn public_leaf_allowlist(
    repository: &str,
    manifest: &crate::CanonicalArtifactManifest,
    encoding: &crate::TransportEncodingRecord,
    attestation: &AttestationEnvelope,
    validator: &ValidatorEvidence,
) -> Result<BTreeSet<String>, CacheError> {
    let mut material = BTreeMap::from([
        ("manifest".to_owned(), serde_json::to_value(manifest)?),
        ("encoding".to_owned(), serde_json::to_value(encoding)?),
        (
            "attestations".to_owned(),
            serde_json::to_value(vec![attestation])?,
        ),
        (
            "validator_evidence".to_owned(),
            serde_json::to_value(vec![validator])?,
        ),
        (
            "publication_principal".to_owned(),
            Value::String("managed-owner".to_owned()),
        ),
        (
            "publication_repository".to_owned(),
            Value::String(repository.to_owned()),
        ),
        (
            "publication_shard".to_owned(),
            Value::String("managed-shard".to_owned()),
        ),
        (
            "publication_branch".to_owned(),
            Value::String("main".to_owned()),
        ),
    ]);
    let mut paths = BTreeSet::new();
    for (key, value) in &mut material {
        collect_leaf_paths(key, value, &mut paths);
    }
    Ok(paths)
}

#[allow(clippy::too_many_lines)]
pub fn prepare_managed_artifact_publication(
    draft: &CanonicalProductionDraft,
    context: &ManagedPublicationPlanningContext,
    sessions: &BTreeMap<PublicationDestination, AuthenticatedGitHubSession>,
) -> Result<ManagedPreparedArtifactPublication, CacheError> {
    if context.owner.trim().is_empty()
        || context.owner.contains('/')
        || context.principal.trim().is_empty()
        || draft.family.trim().is_empty()
    {
        return Err(CacheError::InvalidManifest(
            "managed publication owner, principal, or family is invalid".to_owned(),
        ));
    }
    let destinations = destinations(context.target)?;
    if context.target_heads.keys().copied().collect::<Vec<_>>() != destinations
        || context.capacity_ledgers.keys().copied().collect::<Vec<_>>() != destinations
        || sessions.keys().copied().collect::<Vec<_>>() != destinations
    {
        return Err(CacheError::InvalidManifest(
            "managed publication heads, ledgers, and sessions must match the target".to_owned(),
        ));
    }

    let mut manifests = BTreeMap::new();
    let mut validation_digests = BTreeMap::new();
    for destination in destinations.iter().copied() {
        let manifest = target_manifest(draft, destination)?;
        let manifest_digest = manifest.digest()?;
        let validation_digest = canonical_digest(&ManagedValidationEvidence {
            schema_version: 1,
            manifest_digest: &manifest_digest,
            payload_digest: &manifest.payload_digest,
            achieved_assurance: draft.achieved_assurance,
            assurance_evidence_digests: &draft.assurance_evidence_digests,
        })?;
        manifests.insert(destination, manifest);
        validation_digests.insert(destination, validation_digest);
    }

    // Construct the fixed-shape attestation once to derive the public schema
    // allowlist. The final policy digest changes values, never leaf paths.
    let placeholder_policy = ContentDigest::sha256(b"managed-policy-placeholder");
    let public_repository = repository(
        &context.owner,
        &draft.family,
        PublicationDestination::Public,
    );
    let public_manifest = manifests
        .get(&PublicationDestination::Public)
        .or_else(|| manifests.values().next())
        .expect("a non-none target has a manifest");
    let public_validation = validation_digests
        .get(&PublicationDestination::Public)
        .or_else(|| validation_digests.values().next())
        .expect("a non-none target has validation evidence");
    let mut public_evidence = draft.assurance_evidence_digests.clone();
    public_evidence.push(public_validation.clone());
    public_evidence.sort();
    public_evidence.dedup();
    let source_revision = producer_source_revision()?;
    let placeholder_attestation = AttestationEnvelope {
        schema_version: 1,
        kind: if draft.achieved_assurance == ArtifactAssuranceState::Certified {
            AttestationKind::Certification
        } else {
            AttestationKind::Validation
        },
        subject_digest: public_manifest.digest()?,
        actor: context.principal.clone(),
        policy_digest: placeholder_policy,
        execution_fingerprint_digest: canonical_digest(&draft.source_artifact_key)?,
        producer_toolkit_version: public_manifest.producer_toolkit_version.clone(),
        dependency_versions: BTreeMap::from([(
            "xcelerator-toolkit".to_owned(),
            env!("CARGO_PKG_VERSION").to_owned(),
        )]),
        source_revision,
        event_unix_seconds: context.event_unix_seconds,
        location: None,
        evidence_digests: public_evidence,
    };
    let placeholder_validator = ValidatorEvidence {
        validator_id: MANAGED_VALIDATOR_ID.to_owned(),
        passed: true,
        evidence_digest: public_validation.clone(),
        establishes_assurance: Some(draft.achieved_assurance),
    };
    let allowed_leaf_fields = public_leaf_allowlist(
        &public_repository,
        public_manifest,
        &draft.encoding,
        &placeholder_attestation,
        &placeholder_validator,
    )?;
    let policy = CachePublicationPolicy {
        policy_id: "xcelerator-managed-owner-publication-v1".to_owned(),
        owner_principals: [context.principal.clone()].into_iter().collect(),
        allow_owner_direct: true,
        minimum_assurance: destinations
            .iter()
            .copied()
            .map(|destination| (destination, ArtifactAssuranceState::Computed))
            .collect(),
        required_validators: destinations
            .iter()
            .copied()
            .map(|destination| {
                (
                    destination,
                    [MANAGED_VALIDATOR_ID.to_owned()].into_iter().collect(),
                )
            })
            .collect(),
        minimum_unique_contributor_reviews: 1,
        sanitizer: PublicSanitizerProfile {
            allowed_leaf_fields,
            allowed_source_identifiers: BTreeSet::new(),
            allowed_repository_names: destinations
                .iter()
                .copied()
                .map(|destination| repository(&context.owner, &draft.family, destination))
                .collect(),
            ..PublicSanitizerProfile::default()
        },
    };
    let policy_digest = policy.digest()?;

    let mut routes = Vec::new();
    let mut endpoints = Vec::new();
    let mut ledgers = BTreeMap::new();
    let mut inputs = BTreeMap::new();
    let mut bundles = BTreeMap::new();
    for destination in destinations.iter().copied() {
        let shard_id = shard_id(&draft.family, destination);
        let authorized_repository = repository(&context.owner, &draft.family, destination);
        let repository_name = authorized_repository
            .split_once('/')
            .expect("managed repository contains owner separator")
            .1
            .to_owned();
        routes.push(ArtifactFamilyRoute {
            family: draft.family.clone(),
            visibility: visibility(destination),
            ordered_shards: vec![TopologyShardRoute {
                shard_id: shard_id.clone(),
                endpoint_id: shard_id.clone(),
                sequence: 1,
                status: TopologyShardStatus::Writable,
                successor_shard_id: None,
            }],
        });
        endpoints.push(GitHubRepositoryEndpoint {
            shard_id: shard_id.clone(),
            owner: context.owner.clone(),
            repository: repository_name,
            branch: "main".to_owned(),
            visibility: visibility(destination),
            enabled_for_read: true,
            enabled_for_write: true,
            clone_via_ssh: false,
        });
        let ledger = context.capacity_ledgers[&destination].clone();
        ledger.validate()?;
        if ledger.shard_id != shard_id {
            return Err(CacheError::InvalidManifest(format!(
                "managed capacity ledger {:?} does not match expected shard {shard_id:?}",
                ledger.shard_id
            )));
        }
        ledgers.insert(shard_id, ledger);

        let manifest = manifests[&destination].clone();
        let manifest_digest = manifest.digest()?;
        let validation_digest = validation_digests[&destination].clone();
        let validator = ValidatorEvidence {
            validator_id: MANAGED_VALIDATOR_ID.to_owned(),
            passed: true,
            evidence_digest: validation_digest.clone(),
            establishes_assurance: Some(draft.achieved_assurance),
        };
        let mut evidence_digests = draft.assurance_evidence_digests.clone();
        evidence_digests.push(validation_digest.clone());
        evidence_digests.sort();
        evidence_digests.dedup();
        let attestation = AttestationEnvelope {
            policy_digest: policy_digest.clone(),
            subject_digest: manifest_digest.clone(),
            evidence_digests,
            ..placeholder_attestation.clone()
        };
        attestation.digest()?;
        let candidate = CachePublicationCandidate {
            semantic_digest: manifest.semantic_digest.clone(),
            manifest_digest,
            payload_digest: manifest.payload_digest.clone(),
            completion: ArtifactCompletionState::Complete,
            achieved_assurance: draft.achieved_assurance,
            disposition: ArtifactDisposition::Active,
            validator_evidence: vec![validator.clone()],
            public_metadata: BTreeMap::new(),
        };
        inputs.insert(
            destination,
            TargetPublicationPlanningInput {
                candidate: candidate.clone(),
                authorization: TargetPublicationAuthorizationRequest {
                    destination,
                    repository: authorized_repository.clone(),
                    authority: PublicationAuthority {
                        principal: context.principal.clone(),
                        mode: PublicationAuthorityMode::OwnerDirect,
                        allowed_targets: [context.target].into_iter().collect(),
                        allowed_repositories: [authorized_repository].into_iter().collect(),
                        policy_digest: policy_digest.0.clone(),
                    },
                    contributor: None,
                    reviews: Vec::new(),
                },
                expected_head: context.target_heads[&destination].clone(),
                projected_metadata_bytes: 4 * 1024 * 1024,
                projected_history_bytes: draft.encoding.package_size_bytes / 10,
            },
        );
        bundles.insert(
            destination,
            PublicationMetadataBundle {
                schema_version: 1,
                family: draft.family.clone(),
                manifest,
                encoding: draft.encoding.clone(),
                validation_attestations: vec![attestation],
                validator_evidence: vec![validator],
                target_metadata: BTreeMap::new(),
                achieved_assurance: draft.achieved_assurance,
                disposition: ArtifactDisposition::Active,
            },
        );
    }
    let topology = TopologyRegistry {
        schema_version: 1,
        generation: 1,
        previous_registry_digest: None,
        policy_digest,
        trust_anchor_ids: vec!["managed-owner-direct-v1".to_owned()],
        family_routes: routes,
    };
    let topology_trust = TopologyTrustPolicy {
        minimum_generation: 1,
        pinned_registry_digest: Some(topology.digest()?),
        required_trust_anchor: Some("managed-owner-direct-v1".to_owned()),
    };
    let network = CacheNetworkRegistry {
        schema_version: 1,
        repositories: endpoints,
    };
    let coordinated = crate::coordinate_publication(
        &draft.family,
        context.target,
        &policy,
        &topology,
        &topology_trust,
        &network,
        &ledgers,
        &draft.encoding,
        &TransportPolicy::default(),
        &inputs,
        sessions,
    )?;
    if !coordinated.authorized() {
        let reasons = coordinated
            .target_reports
            .iter()
            .filter(|(_, report)| !report.accepted())
            .map(|(destination, report)| {
                let mut reasons = report
                    .authorization
                    .reasons
                    .iter()
                    .chain(&report.reasons)
                    .cloned()
                    .collect::<Vec<_>>();
                reasons.sort();
                reasons.dedup();
                let reasons = if reasons.is_empty() {
                    format!(
                        "authorization denied for principal {:?} with {:?} permission on {:?}",
                        report.authorization.repository_permission.principal,
                        report.authorization.repository_permission.permission,
                        report.authorization.repository_permission.repository
                    )
                } else {
                    reasons.join("; ")
                };
                format!("{destination:?}: {reasons}")
            })
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(CacheError::PermissionDenied(format!(
            "managed publication preflight was rejected: {reasons}"
        )));
    }
    Ok(ManagedPreparedArtifactPublication {
        policy,
        topology,
        topology_trust,
        network,
        bundles,
        coordinated,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn execute_prepared_managed_artifact_publication(
    prepared: &mut ManagedPreparedArtifactPublication,
    remotes: &BTreeMap<PublicationDestination, &dyn crate::RemoteGitStore>,
    checkpoints: &crate::PublicationJournalStore,
    cancellation: &xc_core::CancellationToken,
    sessions: &BTreeMap<PublicationDestination, AuthenticatedGitHubSession>,
    staging_root: &std::path::Path,
    resources: &xc_core::ResourcePolicy,
    maximum_steps: usize,
    receipt_verified_at_unix_seconds: u64,
    replace_existing_semantic: bool,
) -> Result<ManagedPublicationExecutionReport, CacheError> {
    if receipt_verified_at_unix_seconds == 0 {
        return Err(CacheError::InvalidManifest(
            "managed publication receipt time must be positive".to_owned(),
        ));
    }
    let journal = prepared.coordinated.journal.as_mut().ok_or_else(|| {
        CacheError::InvalidTransition(
            "managed publication has no authorized transaction journal".to_owned(),
        )
    })?;
    if remotes.keys().copied().collect::<Vec<_>>()
        != journal.targets.keys().copied().collect::<Vec<_>>()
        || sessions.keys().copied().collect::<Vec<_>>()
            != journal.targets.keys().copied().collect::<Vec<_>>()
    {
        return Err(CacheError::InvalidManifest(
            "managed publication remotes and sessions must match journal targets".to_owned(),
        ));
    }
    let finalization_policy = crate::PublicationFinalizationPolicy::default();
    if maximum_steps == 0 {
        return Err(CacheError::ResourceLimit(
            "managed publication requires a positive step ceiling".to_owned(),
        ));
    }
    let mut target_staging_roots = BTreeMap::new();
    for destination in journal.targets.keys().copied() {
        let target_root = staging_root
            .join("managed-targets")
            .join(match destination {
                PublicationDestination::Private => "private",
                PublicationDestination::Public => "public",
            });
        for part in &prepared.bundles[&destination].encoding.ordered_parts {
            let source = part
                .repository_path
                .split('/')
                .fold(staging_root.to_owned(), |path, component| {
                    path.join(component)
                });
            let target = part
                .repository_path
                .split('/')
                .fold(target_root.clone(), |path, component| path.join(component));
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !target.exists() && std::fs::hard_link(&source, &target).is_err() {
                std::fs::copy(&source, &target)?;
            }
            let bytes = std::fs::read(&target)?;
            if bytes.len() as u64 != part.size_bytes
                || ContentDigest::sha256(&bytes) != part.content_digest
            {
                return Err(CacheError::DigestMismatch {
                    expected: part.content_digest.to_string(),
                    actual: ContentDigest::sha256(&bytes).to_string(),
                });
            }
        }
        target_staging_roots.insert(destination, target_root);
    }
    let mut steps_executed = 0usize;
    while !journal.complete() && steps_executed < maximum_steps {
        let destination = [
            PublicationDestination::Private,
            PublicationDestination::Public,
        ]
        .into_iter()
        .find(|destination| {
            journal.targets.get(destination).is_some_and(|target| {
                !matches!(
                    target.state,
                    crate::PublicationTargetState::ReceiptComplete
                        | crate::PublicationTargetState::Abandoned
                )
            })
        })
        .ok_or_else(|| {
            CacheError::InvalidTransition(
                "managed journal has no resumable target but is incomplete".to_owned(),
            )
        })?;
        let execution = crate::PublicationTargetExecution {
            staging_root: &target_staging_roots[&destination],
            resources,
            finalization_policy: &finalization_policy,
            bundle: &prepared.bundles[&destination],
            public_sanitizer: (destination == PublicationDestination::Public)
                .then_some(&prepared.policy.sanitizer),
            maximum_index_bytes: 4 * 1024 * 1024,
            receipt_verified_at_unix_seconds,
            replace_existing_semantic,
        };
        crate::advance_publication_target(
            remotes[&destination],
            checkpoints,
            cancellation,
            &sessions[&destination],
            journal,
            destination,
            &execution,
        )?;
        steps_executed += 1;
    }
    if !journal.complete() {
        return Err(CacheError::ResourceLimit(format!(
            "managed publication transaction {} did not complete within {maximum_steps} steps",
            journal.transaction_id
        )));
    }
    let final_journal_digest = journal.digest()?;
    Ok(ManagedPublicationExecutionReport {
        transaction_id: journal.transaction_id.clone(),
        completed: true,
        steps_executed,
        final_journal_digest,
    })
}

#[allow(unreachable_code, unused_variables, unused_mut)]
fn execute_managed_family_drafts_on_github(
    drafts: &[&CanonicalProductionDraft],
    target: PublicationTarget,
    owner: &str,
    journal_root: &Path,
    resources: &xc_core::ResourcePolicy,
    replace_existing_semantic: bool,
) -> Result<ManagedRunPublicationReport, CacheError> {
    let destinations = destinations(target)?;
    let event_unix_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| CacheError::InvalidTransition(error.to_string()))?
        .as_secs()
        .max(1);
    let cancellation = xc_core::CancellationToken::for_policy(resources);
    let probe = crate::GitHubCredentialApiProbe::default();
    let checkpoints = crate::PublicationJournalStore::new(journal_root);
    let mut reports = Vec::new();
    let mut remotely_completed = 0usize;
    let mut replacement_families = BTreeSet::new();
    let first_draft = drafts.first().ok_or_else(|| {
        CacheError::InvalidTransition("managed family publication received no drafts".to_owned())
    })?;
    if drafts
        .iter()
        .any(|draft| draft.family != first_draft.family)
    {
        return Err(CacheError::InvalidTransition(
            "managed family publication requires exactly one artifact family".to_owned(),
        ));
    }
    // Authenticate and initialize each destination once for the complete
    // family. Every artifact retains its own transaction journal and audit
    // receipt, but no longer pays for a fresh bare repository and fetch.
    let mut sessions = BTreeMap::new();
    for destination in destinations.iter().copied() {
        let authorized_repository = repository(owner, &first_draft.family, destination);
        sessions.insert(destination, probe.probe_repository(&authorized_repository)?);
    }
    let principals = sessions
        .values()
        .map(|session| session.evidence().principal.as_str())
        .collect::<BTreeSet<_>>();
    if principals.len() != 1 {
        return Err(CacheError::Authentication(
            "managed dual publication resolved different GitHub principals".to_owned(),
        ));
    }
    let principal = principals
        .into_iter()
        .next()
        .expect("target has a session")
        .to_owned();
    // Repository batches are the sole managed publication format. A one-
    // artifact family uses the same path so publication never regresses to
    // the obsolete per-artifact receipt/commit transaction.
    let mut batch_reports = Vec::new();
    for destination in destinations.iter().copied() {
        batch_reports.extend(execute_family_batch_publication(
            drafts,
            destination,
            owner,
            &principal,
            &sessions,
            journal_root,
            resources,
            replace_existing_semantic,
            event_unix_seconds,
        )?);
    }
    return Ok(ManagedRunPublicationReport {
        schema_version: 1,
        all_completed: batch_reports.iter().all(|report| report.completed),
        transactions: batch_reports,
        current_tree_paths_removed: 0,
    });

    #[allow(unreachable_code)]
    let mut remotes = BTreeMap::new();
    for destination in destinations.iter().copied() {
        let target_roots = drafts
            .iter()
            .map(|draft| target_staging_root(&draft.staged_parts_root, destination))
            .collect::<Vec<_>>();
        let temporary_root = journal_root
            .join("git-transport")
            .join("family")
            .join(&first_draft.family)
            .join(visibility_name(destination));
        let remote = crate::GitCliRemoteStore::new(
            temporary_root,
            target_roots[0].clone(),
            format!("Xcelerator Toolkit ({principal})"),
            "xcelerator-toolkit@users.noreply.github.com",
        )?
        .with_additional_staged_parts_roots(target_roots.into_iter().skip(1))?
        .with_resource_policy(resources.clone());
        remotes.insert(destination, remote);
    }

    for draft in drafts {
        // Refresh before any per-artifact remote access. Without this check a
        // large family can reach its next artifact with evidence that expired
        // while the previous artifact was being committed.
        for destination in destinations.iter().copied() {
            if sessions[&destination].requires_refresh(60)? {
                let authorized_repository = repository(owner, &draft.family, destination);
                let refreshed = probe.probe_repository(&authorized_repository)?;
                if refreshed.evidence().principal != principal {
                    return Err(CacheError::Authentication(
                        "refreshed GitHub permission resolved a different principal".to_owned(),
                    ));
                }
                sessions.insert(destination, refreshed);
            }
        }
        let semantic_prefix = &draft.manifest.semantic_digest.0[..2];
        let mut heads = BTreeMap::new();
        let mut ledgers = BTreeMap::new();
        for destination in destinations.iter().copied() {
            verify_bootstrap_registry_route(
                &remotes[&destination],
                owner,
                &draft.family,
                destination,
                &cancellation,
            )?;
            let authorized_repository = repository(owner, &draft.family, destination);
            let repository_url = repository_url(owner, &draft.family, destination);
            let shard_id = shard_id(&draft.family, destination);
            let target_root = target_staging_root(&draft.staged_parts_root, destination);
            let (head, ledger) = ensure_managed_shard_sidecars(
                &remotes[&destination],
                &sessions[&destination],
                &repository_url,
                &authorized_repository,
                "main",
                &shard_id,
                &draft.family,
                semantic_prefix,
                &target_root,
                resources,
                &cancellation,
                None,
                &crate::PrivatePublicationLeasePolicy::default(),
            )?;
            preflight_producer_monotonicity(
                &remotes[&destination],
                &repository_url,
                &head,
                &draft.family,
                &draft.manifest.semantic_digest,
                &draft.manifest.producer_toolkit_version,
                &cancellation,
            )?;
            heads.insert(destination, head);
            ledgers.insert(destination, ledger);
        }

        // A family can contain many artifacts. Refresh the live permission
        // evidence before planning the next artifact rather than allowing the
        // family-wide session to expire merely because earlier artifacts took
        // longer than the five-minute freshness window.
        for destination in destinations.iter().copied() {
            if sessions[&destination].requires_refresh(60)? {
                let authorized_repository = repository(owner, &draft.family, destination);
                let refreshed = probe.probe_repository(&authorized_repository)?;
                if refreshed.evidence().principal != principal {
                    return Err(CacheError::Authentication(
                        "refreshed GitHub permission resolved a different principal".to_owned(),
                    ));
                }
                sessions.insert(destination, refreshed);
            }
        }
        let context = ManagedPublicationPlanningContext {
            owner: owner.to_owned(),
            principal: principal.clone(),
            target,
            target_heads: heads,
            capacity_ledgers: ledgers,
            event_unix_seconds,
        };
        let mut prepared = prepare_managed_artifact_publication(draft, &context, &sessions)?;
        let remote_refs: BTreeMap<PublicationDestination, &dyn crate::RemoteGitStore> = remotes
            .iter()
            .map(|(destination, remote)| (*destination, remote as &dyn crate::RemoteGitStore))
            .collect();
        if let Some(report) = completed_remote_publication(&prepared, &remote_refs, &cancellation)?
        {
            reports.push(report);
            remotely_completed += 1;
            continue;
        }
        let planned = prepared
            .coordinated
            .journal
            .as_ref()
            .expect("authorized managed publication has a journal");
        if let Some(existing) = checkpoints.load_if_exists(&planned.transaction_id)? {
            prepared.coordinated.journal = Some(existing);
        } else {
            checkpoints.save(planned)?;
        }
        let maximum_steps = draft
            .encoding
            .ordered_parts
            .len()
            .saturating_mul(4)
            .saturating_add(32);
        let mut refresh_attempts = 0usize;
        let report = loop {
            match execute_prepared_managed_artifact_publication(
                &mut prepared,
                &remote_refs,
                &checkpoints,
                &cancellation,
                &sessions,
                &draft.staged_parts_root,
                resources,
                maximum_steps,
                event_unix_seconds,
                replace_existing_semantic,
            ) {
                Ok(report) => break report,
                Err(error)
                    if error
                        .to_string()
                        .contains(crate::STALE_AUTHORITY_PROBE_MESSAGE)
                        && refresh_attempts < 16 =>
                {
                    refresh_attempts += 1;
                    for destination in destinations.iter().copied() {
                        let authorized_repository = repository(owner, &draft.family, destination);
                        let refreshed = probe.probe_repository(&authorized_repository)?;
                        if refreshed.evidence().principal != principal {
                            return Err(CacheError::Authentication(
                                "refreshed GitHub permission resolved a different principal"
                                    .to_owned(),
                            ));
                        }
                        sessions.insert(destination, refreshed);
                    }
                    eprintln!(
                        "publication authorization refreshed for family {} after a long-running artifact step",
                        draft.family
                    );
                }
                Err(error) => return Err(error),
            }
        };
        reports.push(report);
        if replace_existing_semantic {
            replacement_families.insert(draft.family.clone());
        }
    }
    if remotely_completed > 0 {
        eprintln!(
            "publication family {}: reused {remotely_completed} completed remote transaction(s)",
            first_draft.family
        );
    }
    for destination in destinations.iter().copied() {
        remotes[&destination].cleanup_session(&repository_url(
            owner,
            &first_draft.family,
            destination,
        ))?;
    }
    let mut current_tree_paths_removed = 0usize;
    if replace_existing_semantic {
        for family in replacement_families {
            for destination in destinations.iter().copied() {
                let authorized_repository = repository(owner, &family, destination);
                let session = probe.probe_repository(&authorized_repository)?;
                if !session.evidence().permission.permits_write() {
                    return Err(CacheError::PermissionDenied(format!(
                        "replacement cleanup lacks write permission for {authorized_repository}"
                    )));
                }
                let repository_url = repository_url(owner, &family, destination);
                let cleanup_staging = journal_root
                    .join("replacement-cleanup-staging")
                    .join(&family)
                    .join(visibility_name(destination));
                let remote = crate::GitCliRemoteStore::new(
                    journal_root
                        .join("git-transport")
                        .join("replacement-cleanup")
                        .join(&family)
                        .join(visibility_name(destination)),
                    &cleanup_staging,
                    format!("Xcelerator Toolkit ({})", session.evidence().principal),
                    "xcelerator-toolkit@users.noreply.github.com",
                )?
                .with_resource_policy(resources.clone());
                current_tree_paths_removed =
                    current_tree_paths_removed.saturating_add(prune_unreferenced_current_tree(
                        &remote,
                        &repository_url,
                        &family,
                        destination,
                        resources,
                    )?);
                remote.cleanup_session(&repository_url)?;
            }
        }
    }
    Ok(ManagedRunPublicationReport {
        schema_version: 1,
        all_completed: reports.iter().all(|report| report.completed),
        transactions: reports,
        current_tree_paths_removed,
    })
}

pub fn execute_managed_drafts_on_github(
    drafts: &[CanonicalProductionDraft],
    target: PublicationTarget,
    owner: &str,
    journal_root: &Path,
    resources: &xc_core::ResourcePolicy,
    replace_existing_semantic: bool,
) -> Result<ManagedRunPublicationReport, CacheError> {
    let requested_destinations = destinations(target)?;
    let mut groups = Vec::<(PublicationTarget, Vec<CanonicalProductionDraft>)>::new();
    for destination in requested_destinations.iter().copied() {
        let destination_target = match destination {
            PublicationDestination::Private => PublicationTarget::Private,
            PublicationDestination::Public => PublicationTarget::Public,
        };
        let dependency_remote = crate::GitCliRemoteStore::new(
            journal_root
                .join("git-transport")
                .join("dependency-preflight")
                .join(visibility_name(destination)),
            journal_root
                .join("dependency-preflight-staging")
                .join(visibility_name(destination)),
            "Xcelerator Toolkit",
            "xcelerator-toolkit@users.noreply.github.com",
        )?
        .with_resource_policy(resources.clone());
        let mut dependency_revisions = BTreeMap::new();
        let mut dependency_results =
            BTreeMap::<(String, ContentDigest, ContentDigest, ContentDigest), bool>::new();
        let remapped = remap_destination_drafts_with_existing(drafts, destination, |dependency| {
            let identity = (
                dependency.artifact_family.clone(),
                dependency.semantic_digest.clone(),
                dependency.manifest_digest.clone(),
                dependency.payload_digest.clone(),
            );
            if let Some(exists) = dependency_results.get(&identity) {
                return Ok(*exists);
            }
            let exists = exact_destination_dependency_exists(
                &dependency_remote,
                owner,
                destination,
                dependency,
                &mut dependency_revisions,
                resources,
            )?;
            dependency_results.insert(identity, exists);
            Ok(exists)
        })?;
        let mut by_family = BTreeMap::<String, Vec<CanonicalProductionDraft>>::new();
        for draft in remapped {
            by_family
                .entry(draft.family.clone())
                .or_default()
                .push(draft);
        }
        groups.extend(
            by_family
                .into_values()
                .map(|family_drafts| (destination_target, family_drafts)),
        );
    }

    // Fail the complete multi-family run before the first remote mutation if
    // any destination repository is absent from the PAT or lacks write access.
    // Per-family probes are repeated immediately before mutation so their
    // evidence remains fresh during long publication runs.
    let probe = crate::GitHubCredentialApiProbe::default();
    let mut principals = BTreeSet::new();
    for (group_target, family_drafts) in &groups {
        let family = family_drafts
            .first()
            .map(|draft| draft.family.as_str())
            .ok_or_else(|| {
                CacheError::InvalidTransition(
                    "managed publication received an empty artifact family".to_owned(),
                )
            })?;
        for destination in destinations(*group_target)? {
            let authorized_repository = repository(owner, family, destination);
            let session = probe.probe_repository(&authorized_repository)?;
            principals.insert(session.evidence().principal.clone());
        }
    }
    if principals.len() != 1 {
        return Err(CacheError::Authentication(
            "managed multi-family publication resolved different GitHub principals".to_owned(),
        ));
    }

    let started = std::time::Instant::now();
    let reports = groups
        .par_iter()
        .map(|(group_target, family_drafts)| {
            let family = family_drafts
                .first()
                .map(|draft| draft.family.as_str())
                .unwrap_or("empty");
            let family_started = std::time::Instant::now();
            let family_draft_refs = family_drafts.iter().collect::<Vec<_>>();
            let report = execute_managed_family_drafts_on_github(
                &family_draft_refs,
                *group_target,
                owner,
                journal_root,
                resources,
                replace_existing_semantic,
            )?;
            eprintln!(
                "publication family {family}: evaluated {} staged candidate(s) in {:.3}s",
                family_drafts.len(),
                family_started.elapsed().as_secs_f64()
            );
            Ok(report)
        })
        .collect::<Result<Vec<_>, CacheError>>()?;

    let transactions = reports
        .iter()
        .flat_map(|report| report.transactions.iter().cloned())
        .collect::<Vec<_>>();
    let current_tree_paths_removed = reports.iter().fold(0usize, |total, report| {
        total.saturating_add(report.current_tree_paths_removed)
    });
    eprintln!(
        "publication execution: evaluated {} destination candidate(s), created {} transaction(s) across {} destination shard groups in {:.3}s",
        drafts.len().saturating_mul(requested_destinations.len()),
        transactions.len(),
        reports.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(ManagedRunPublicationReport {
        schema_version: 1,
        all_completed: reports.iter().all(|report| report.all_completed),
        transactions,
        current_tree_paths_removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        protocol::canonical_json_bytes, CanonicalArtifactManifest, CanonicalPayloadEnvelope,
        CompareAndSwapResult, LogicalPayloadItem, PublicationJournalStore, RemoteCommitRequest,
        RemoteGitStore, RemotePathListReport, RemoteReadReport, RepositoryPermission,
        SemanticKeyEnvelope, ShardIndexPartition, TransportEncodingRecord, TransportPart,
        GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use std::fs;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use xc_core::{AssuranceLevel, CancellationToken, ResourcePolicy};

    struct RepositoryState {
        head: String,
        sequence: u64,
        revisions: BTreeMap<String, BTreeMap<String, Vec<u8>>>,
    }

    struct FilesystemMemoryRemote {
        staging_root: PathBuf,
        repositories: Mutex<BTreeMap<String, RepositoryState>>,
    }

    impl FilesystemMemoryRemote {
        fn new(staging_root: PathBuf) -> Self {
            Self {
                staging_root,
                repositories: Mutex::new(BTreeMap::new()),
            }
        }

        fn insert_repository(
            &self,
            repository: String,
            head: String,
            tree: BTreeMap<String, Vec<u8>>,
        ) {
            self.repositories.lock().unwrap().insert(
                repository,
                RepositoryState {
                    head: head.clone(),
                    sequence: 0,
                    revisions: BTreeMap::from([(head, tree)]),
                },
            );
        }

        fn staged_path(&self, repository: &str, repository_path: &str) -> PathBuf {
            repository_path.split('/').fold(
                self.staging_root
                    .join("managed-targets")
                    .join(visibility_token(repository)),
                |path, part| path.join(part),
            )
        }
    }

    impl RemoteGitStore for FilesystemMemoryRemote {
        fn read_ref(&self, repository: &str, _branch: &str) -> Result<String, CacheError> {
            self.repositories
                .lock()
                .unwrap()
                .get(repository)
                .map(|state| state.head.clone())
                .ok_or_else(|| CacheError::NotFound(repository.to_owned()))
        }

        fn immutable_path_digest(
            &self,
            repository: &str,
            revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(self
                .repositories
                .lock()
                .unwrap()
                .get(repository)
                .and_then(|state| state.revisions.get(revision))
                .and_then(|tree| tree.get(path))
                .map(|bytes| ContentDigest::sha256(bytes)))
        }

        fn read_committed_path(
            &self,
            repository: &str,
            revision: &str,
            path: &str,
            maximum_bytes: u64,
            cancellation: &CancellationToken,
            writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let repositories = self.repositories.lock().unwrap();
            let bytes = repositories
                .get(repository)
                .and_then(|state| state.revisions.get(revision))
                .and_then(|tree| tree.get(path))
                .ok_or_else(|| CacheError::NotFound(format!("{repository}:{revision}:{path}")))?;
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
            request.validate_limits()?;
            let mut repositories = self.repositories.lock().unwrap();
            let state = repositories
                .get_mut(&request.repository)
                .ok_or_else(|| CacheError::NotFound(request.repository.clone()))?;
            if state.head != request.expected_head {
                return Ok(CompareAndSwapResult::RefConflict {
                    current_head: state.head.clone(),
                });
            }
            let mut tree = state.revisions[&state.head].clone();
            for part in &request.parts {
                let bytes = fs::read(self.staged_path(&request.repository, &part.repository_path))?;
                if bytes.len() as u64 != part.size_bytes
                    || ContentDigest::sha256(&bytes) != part.content_digest
                {
                    return Err(CacheError::DigestMismatch {
                        expected: part.content_digest.to_string(),
                        actual: ContentDigest::sha256(&bytes).to_string(),
                    });
                }
                tree.insert(part.repository_path.clone(), bytes);
            }
            state.sequence += 1;
            let commit = format!(
                "{}-commit-{}",
                visibility_token(&request.repository),
                state.sequence
            );
            state.revisions.insert(commit.clone(), tree);
            state.head = commit.clone();
            Ok(CompareAndSwapResult::Committed { commit_id: commit })
        }

        fn list_committed_paths(
            &self,
            repository: &str,
            revision: &str,
            prefix: &str,
            maximum_paths: u64,
            maximum_total_path_bytes: u64,
            _cancellation: &CancellationToken,
        ) -> Result<RemotePathListReport, CacheError> {
            let repositories = self.repositories.lock().unwrap();
            let tree = repositories
                .get(repository)
                .and_then(|state| state.revisions.get(revision))
                .ok_or_else(|| CacheError::NotFound(format!("{repository}:{revision}")))?;
            let paths = tree
                .keys()
                .filter(|path| *path == prefix || path.starts_with(&format!("{prefix}/")))
                .cloned()
                .collect::<Vec<_>>();
            let total_path_bytes = paths.iter().map(|path| path.len() as u64).sum::<u64>();
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

    fn visibility_token(repository: &str) -> &'static str {
        if repository.contains("private") || repository.contains("restricted") {
            "private"
        } else {
            "public"
        }
    }

    fn fixture_draft(root: &Path) -> CanonicalProductionDraft {
        let payload = b"managed-publication-payload";
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_tau_matrix".to_owned(),
            mathematical_semantics_version: "managed-fixture-v1".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({"n_modes": 2}),
            normalization: Some("row_major".to_owned()),
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
            dimensions: vec![payload.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "payload.bin".to_owned(),
                content_digest: ContentDigest::sha256(payload),
                size_bytes: payload.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let part_digest = ContentDigest::sha256(payload);
        let part = TransportPart {
            sequence: 0,
            repository_path: format!(
                "objects/sha256/{}/{}.part",
                &part_digest.0[..2],
                part_digest.0
            ),
            size_bytes: payload.len() as u64,
            content_digest: part_digest.clone(),
        };
        let part_path = part
            .repository_path
            .split('/')
            .fold(root.to_owned(), |path, component| path.join(component));
        fs::create_dir_all(part_path.parent().unwrap()).unwrap();
        fs::write(part_path, payload).unwrap();
        let encoding = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: payload_digest.clone(),
            encoder_profile: "managed-fixture-v1".to_owned(),
            package_size_bytes: payload.len() as u64,
            package_digest: part_digest,
            ordered_parts: vec![part],
            reconstruction: "concatenate".to_owned(),
        };
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm-matrices".to_owned(),
            semantic_key,
            semantic_digest,
            canonical_payload,
            payload_digest,
            transport_digests: vec![encoding.digest().unwrap()],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Certified,
            claim_scope: "managed integration fixture".to_owned(),
            assumptions: Vec::new(),
        };
        CanonicalProductionDraft {
            schema_version: 1,
            family: "ccm-matrices".to_owned(),
            source_operation: "fixture".to_owned(),
            source_logical_key: "fixture/tau".to_owned(),
            source_artifact_key: crate::ArtifactKey::new("ccm_tau_matrix", "fixture/tau", b"p")
                .unwrap(),
            source_content_digest: ContentDigest::sha256(payload),
            source_manifest_digest: ContentDigest::sha256(b"source-manifest"),
            manifest,
            encoding,
            staged_parts_root: root.to_owned(),
            achieved_assurance: ArtifactAssuranceState::Certified,
            required_assurance: Some(ArtifactAssuranceState::Certified),
            assurance_evidence_digests: vec![ContentDigest::sha256(b"interval-certificate")],
        }
    }

    fn fixture_draft_with_n(root: &Path, n_modes: u64) -> CanonicalProductionDraft {
        let mut draft = fixture_draft(root);
        draft.manifest.semantic_key.resolved_mathematical_parameters =
            serde_json::json!({"n_modes": n_modes});
        let semantic_digest = draft.manifest.semantic_key.digest().unwrap();
        draft.manifest.semantic_digest = semantic_digest;
        draft.source_logical_key = format!("fixture/tau/{n_modes}");
        draft.source_artifact_key =
            crate::ArtifactKey::new("ccm_tau_matrix", &draft.source_logical_key, b"p").unwrap();
        draft
    }

    fn fixture_draft_with_dependency(
        mut draft: CanonicalProductionDraft,
        dependency: &CanonicalProductionDraft,
    ) -> CanonicalProductionDraft {
        draft
            .manifest
            .canonical_payload
            .dependencies
            .push(crate::PayloadDependencyIdentity {
                artifact_family: dependency.manifest.artifact_family.clone(),
                semantic_digest: dependency.manifest.semantic_digest.clone(),
                manifest_digest: dependency.manifest.digest().unwrap(),
                payload_digest: dependency.manifest.payload_digest.clone(),
            });
        draft.manifest.payload_digest = draft.manifest.canonical_payload.digest().unwrap();
        draft.encoding.canonical_payload_digest = draft.manifest.payload_digest.clone();
        draft.manifest.transport_digests = vec![draft.encoding.digest().unwrap()];
        draft
    }

    fn fixture_draft_family(
        mut draft: CanonicalProductionDraft,
        family: &str,
    ) -> CanonicalProductionDraft {
        draft.family = family.to_owned();
        draft.manifest.artifact_family = family.to_owned();
        draft
    }

    #[test]
    fn destination_remapping_rewrites_the_complete_dependency_closure() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-destination-dependency-remap-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let mut leaf = fixture_draft_family(fixture_draft_with_n(&root, 2), "ccm-components");
        leaf.manifest
            .assumptions
            .push("publication_visibility=public".to_owned());
        let middle = fixture_draft_family(
            fixture_draft_with_dependency(fixture_draft_with_n(&root, 3), &leaf),
            "ccm-matrices",
        );
        let root_draft = fixture_draft_family(
            fixture_draft_with_dependency(fixture_draft_with_n(&root, 4), &middle),
            "ccm-roots",
        );

        let remapped = remap_destination_drafts(
            &[root_draft.clone(), leaf.clone(), middle.clone()],
            PublicationDestination::Private,
        )
        .unwrap();
        let destination_leaf = remapped
            .iter()
            .find(|draft| draft.family == "ccm-components")
            .unwrap();
        let destination_middle = remapped
            .iter()
            .find(|draft| draft.family == "ccm-matrices")
            .unwrap();
        let destination_root = remapped
            .iter()
            .find(|draft| draft.family == "ccm-roots")
            .unwrap();

        let middle_dependency = &destination_middle.manifest.canonical_payload.dependencies[0];
        assert_eq!(
            middle_dependency.manifest_digest,
            destination_leaf.manifest.digest().unwrap()
        );
        assert_eq!(
            middle_dependency.payload_digest,
            destination_leaf.manifest.payload_digest
        );
        let root_dependency = &destination_root.manifest.canonical_payload.dependencies[0];
        assert_eq!(
            root_dependency.manifest_digest,
            destination_middle.manifest.digest().unwrap()
        );
        assert_eq!(
            root_dependency.payload_digest,
            destination_middle.manifest.payload_digest
        );
        assert_ne!(
            root_dependency.manifest_digest,
            middle.manifest.digest().unwrap()
        );
        for draft in &remapped {
            assert!(draft
                .manifest
                .assumptions
                .contains(&"publication_visibility=private".to_owned()));
            assert!(!draft
                .manifest
                .assumptions
                .contains(&"publication_visibility=public".to_owned()));
            assert_eq!(
                draft.encoding.canonical_payload_digest,
                draft.manifest.payload_digest
            );
            assert_eq!(
                draft.manifest.transport_digests,
                vec![draft.encoding.digest().unwrap()]
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn destination_remapping_rejects_an_incomplete_dependency_closure() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-destination-incomplete-closure-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let dependency = fixture_draft_with_n(&root, 2);
        let root_draft = fixture_draft_with_dependency(fixture_draft_with_n(&root, 3), &dependency);

        let error =
            remap_destination_drafts(&[root_draft], PublicationDestination::Private).unwrap_err();
        assert!(error
            .to_string()
            .contains("closure is missing exact dependency"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn destination_remapping_accepts_an_exact_indexed_destination_dependency() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-destination-external-dependency-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let dependency = fixture_draft_family(fixture_draft_with_n(&root, 2), "weil-states");
        let root_draft = fixture_draft_family(
            fixture_draft_with_dependency(fixture_draft_with_n(&root, 3), &dependency),
            "ccm-roots",
        );
        let expected = root_draft.manifest.canonical_payload.dependencies[0].clone();
        let mut resolutions = 0usize;

        let remapped = remap_destination_drafts_with_existing(
            &[root_draft],
            PublicationDestination::Private,
            |candidate| {
                resolutions = resolutions.saturating_add(1);
                Ok(candidate == &expected)
            },
        )
        .unwrap();

        assert_eq!(resolutions, 1);
        assert_eq!(
            remapped[0].manifest.canonical_payload.dependencies,
            vec![expected]
        );
        assert!(remapped[0]
            .manifest
            .assumptions
            .contains(&"publication_visibility=private".to_owned()));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn destination_remapping_resolves_a_neutral_alias_to_its_staged_private_manifest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-destination-neutral-alias-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut dependency = fixture_draft_family(fixture_draft_with_n(&root, 2), "weil-states");
        dependency
            .manifest
            .assumptions
            .push("publication_visibility=private".to_owned());
        dependency.manifest.validate().unwrap();
        let neutral = destination_neutral_manifest(&dependency).unwrap();
        assert_ne!(
            neutral.digest().unwrap(),
            dependency.manifest.digest().unwrap()
        );

        let mut root_draft = fixture_draft_family(
            fixture_draft_with_dependency(fixture_draft_with_n(&root, 3), &dependency),
            "ccm-roots",
        );
        root_draft.manifest.canonical_payload.dependencies[0].manifest_digest =
            neutral.digest().unwrap();
        root_draft.manifest.payload_digest =
            root_draft.manifest.canonical_payload.digest().unwrap();
        root_draft.encoding.canonical_payload_digest = root_draft.manifest.payload_digest.clone();
        root_draft.manifest.transport_digests = vec![root_draft.encoding.digest().unwrap()];

        let remapped = remap_destination_drafts_with_existing(
            &[root_draft, dependency.clone()],
            PublicationDestination::Private,
            |_| panic!("a staged neutral alias must not require remote resolution"),
        )
        .unwrap();
        let remapped_root = remapped
            .iter()
            .find(|draft| draft.family == "ccm-roots")
            .unwrap();
        let remapped_dependency = remapped
            .iter()
            .find(|draft| draft.family == "weil-states")
            .unwrap();
        assert_eq!(
            remapped_root.manifest.canonical_payload.dependencies[0].manifest_digest,
            remapped_dependency.manifest.digest().unwrap()
        );
        assert_eq!(
            remapped_dependency.manifest.digest().unwrap(),
            dependency.manifest.digest().unwrap()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn exact_destination_dependency_requires_canonical_manifest_and_active_index() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-exact-destination-dependency-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        let staging = root.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let draft = fixture_draft_with_n(&staging, 2);
        let manifest = target_manifest(&draft, PublicationDestination::Private).unwrap();
        let dependency = crate::PayloadDependencyIdentity {
            artifact_family: manifest.artifact_family.clone(),
            semantic_digest: manifest.semantic_digest.clone(),
            manifest_digest: manifest.digest().unwrap(),
            payload_digest: manifest.payload_digest.clone(),
        };
        let repository =
            "https://github.com/example-org/xcelerator-cache-private-ccm-matrices-0001.git";
        let head = "existing-head";
        let remote = FilesystemMemoryRemote::new(staging);
        remote.insert_repository(
            repository.to_owned(),
            head.to_owned(),
            published_destination_tree(&[&draft], PublicationDestination::Private),
        );

        assert!(exact_destination_dependency_exists(
            &remote,
            "example-org",
            PublicationDestination::Private,
            &dependency,
            &mut BTreeMap::new(),
            &ResourcePolicy::default(),
        )
        .unwrap());

        let mut missing = dependency;
        missing.manifest_digest = ContentDigest::sha256(b"not-published");
        assert!(!exact_destination_dependency_exists(
            &remote,
            "example-org",
            PublicationDestination::Private,
            &missing,
            &mut BTreeMap::new(),
            &ResourcePolicy::default(),
        )
        .unwrap());
        let _ = fs::remove_dir_all(root);
    }

    fn active_index_entry(
        draft: &CanonicalProductionDraft,
        destination: PublicationDestination,
        publication_transaction_id: String,
    ) -> crate::ShardIndexEntry {
        let manifest = target_manifest(draft, destination).unwrap();
        crate::ShardIndexEntry {
            semantic_digest: manifest.semantic_digest.clone(),
            canonical_payload_digest: manifest.payload_digest.clone(),
            manifest_digest: manifest.digest().unwrap(),
            achieved_assurance: draft.achieved_assurance,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: manifest.producer_toolkit_version.clone(),
            minimum_reader_version: manifest.minimum_reader_version.clone(),
            transport_digests: vec![draft.encoding.digest().unwrap()],
            publication_transaction_id,
        }
    }

    fn published_destination_tree(
        drafts: &[&CanonicalProductionDraft],
        destination: PublicationDestination,
    ) -> BTreeMap<String, Vec<u8>> {
        let artifacts = drafts
            .iter()
            .map(|draft| {
                let manifest = target_manifest(draft, destination).unwrap();
                crate::RepositoryBatchArtifact {
                    semantic_digest: manifest.semantic_digest.clone(),
                    canonical_payload_digest: manifest.payload_digest.clone(),
                    manifest_digest: manifest.digest().unwrap(),
                    transport_digest: draft.encoding.digest().unwrap(),
                    manifest_path: format!(
                        "manifests/{}/{}.json",
                        &manifest.semantic_digest.0[..2],
                        manifest.digest().unwrap()
                    ),
                    achieved_assurance: draft.achieved_assurance,
                    producer_toolkit_version: manifest.producer_toolkit_version,
                    provenance_evidence_digests: draft.assurance_evidence_digests.clone(),
                }
            })
            .collect();
        let batch = crate::RepositoryPublicationBatch::new(
            destination,
            "ccm-matrices",
            "test-owner",
            "example-org/restricted-ccm-matrices-0001",
            "main",
            ContentDigest::sha256(b"policy"),
            (destination == PublicationDestination::Private).then_some(1),
            artifacts,
            123,
        )
        .unwrap();
        let transaction_id = batch.batch_id.0.clone();
        let mut partitions = BTreeMap::<String, Vec<crate::ShardIndexEntry>>::new();
        for draft in drafts {
            let entry = active_index_entry(draft, destination, transaction_id.clone());
            partitions
                .entry(entry.semantic_digest.0[..2].to_owned())
                .or_default()
                .push(entry);
        }
        let mut tree = partitions
            .into_iter()
            .map(|(prefix, entries)| {
                let partition =
                    ShardIndexPartition::rebuild("ccm-matrices", &prefix, entries).unwrap();
                (
                    format!("indexes/ccm-matrices/{prefix}.json"),
                    canonical_json_bytes(&partition).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for draft in drafts {
            let manifest = target_manifest(draft, destination).unwrap();
            let manifest_digest = manifest.digest().unwrap();
            tree.insert(
                format!(
                    "manifests/{}/{}.json",
                    &manifest.semantic_digest.0[..2],
                    manifest_digest
                ),
                canonical_json_bytes(&manifest).unwrap(),
            );
        }
        tree.insert(
            batch.repository_path(),
            canonical_json_bytes(&batch).unwrap(),
        );
        tree
    }

    fn ledger(shard: &str, head: &str) -> CapacityLedger {
        CapacityLedger {
            schema_version: 1,
            shard_id: shard.to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 0,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 0,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: head.to_owned(),
            reconciliation_digest: ContentDigest::sha256(shard.as_bytes()),
        }
    }

    #[test]
    fn managed_publication_embeds_a_full_source_commit() {
        let revision = producer_source_revision().unwrap();
        assert_eq!(revision.len(), 40);
        assert!(revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }

    #[test]
    fn destination_selection_excludes_exact_active_index_entries() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!("managed-destination-filter-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let staging = root.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let existing = fixture_draft_with_n(&staging, 2);
        let missing = fixture_draft_with_n(&staging, 3);
        let repository = "https://github.com/example-org/restricted-ccm-matrices-0001.git";
        let head = "existing-head";
        let remote = FilesystemMemoryRemote::new(staging);
        remote.insert_repository(
            repository.to_owned(),
            head.to_owned(),
            published_destination_tree(&[&existing], PublicationDestination::Private),
        );

        let selection = select_missing_destination_drafts(
            &remote,
            repository,
            head,
            "ccm-matrices",
            PublicationDestination::Private,
            &[&existing, &missing],
            &CancellationToken::new(),
        )
        .unwrap();

        assert_eq!(selection.already_present, 1);
        assert_eq!(selection.pending, vec![&missing]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn destination_manifest_with_stronger_dependencies_dominates_a_stripped_wrapper() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-destination-dependency-dominance-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let draft = fixture_draft_with_n(&root, 2);
        let staged = target_manifest(&draft, PublicationDestination::Private).unwrap();
        let mut existing = staged.clone();
        existing
            .canonical_payload
            .dependencies
            .push(crate::PayloadDependencyIdentity {
                artifact_family: "ccm-components".to_owned(),
                semantic_digest: ContentDigest::sha256(b"dependency-semantic"),
                manifest_digest: ContentDigest::sha256(b"dependency-manifest"),
                payload_digest: ContentDigest::sha256(b"dependency-payload"),
            });
        existing.payload_digest = existing.canonical_payload.digest().unwrap();
        assert!(destination_manifest_dominates(&existing, &staged));
        assert!(!destination_manifest_dominates(&staged, &existing));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn destination_selection_republishes_an_unproven_index_entry() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!(
                "managed-destination-unproven-{}",
                std::process::id()
            ));
        let _ = fs::remove_dir_all(&root);
        let staging = root.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let draft = fixture_draft_with_n(&staging, 2);
        let repository = "https://github.com/example-org/restricted-ccm-matrices-0001.git";
        let head = "existing-head";
        let entry = active_index_entry(
            &draft,
            PublicationDestination::Private,
            ContentDigest::sha256(b"missing-batch").0,
        );
        let prefix = entry.semantic_digest.0[..2].to_owned();
        let partition = ShardIndexPartition::rebuild("ccm-matrices", &prefix, vec![entry]).unwrap();
        let manifest = target_manifest(&draft, PublicationDestination::Private).unwrap();
        let manifest_path = format!(
            "manifests/{}/{}.json",
            &manifest.semantic_digest.0[..2],
            manifest.digest().unwrap()
        );
        let remote = FilesystemMemoryRemote::new(staging);
        remote.insert_repository(
            repository.to_owned(),
            head.to_owned(),
            BTreeMap::from([
                (
                    format!("indexes/ccm-matrices/{prefix}.json"),
                    canonical_json_bytes(&partition).unwrap(),
                ),
                (manifest_path, canonical_json_bytes(&manifest).unwrap()),
            ]),
        );

        let selection = select_missing_destination_drafts(
            &remote,
            repository,
            head,
            "ccm-matrices",
            PublicationDestination::Private,
            &[&draft],
            &CancellationToken::new(),
        )
        .unwrap();

        assert_eq!(selection.already_present, 0);
        assert_eq!(selection.pending, vec![&draft]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn destination_selection_makes_an_all_existing_family_a_no_op() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!("managed-destination-noop-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let staging = root.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let first = fixture_draft_with_n(&staging, 2);
        let second = fixture_draft_with_n(&staging, 3);
        let repository = "https://github.com/example-org/restricted-ccm-matrices-0001.git";
        let head = "existing-head";
        let remote = FilesystemMemoryRemote::new(staging);
        remote.insert_repository(
            repository.to_owned(),
            head.to_owned(),
            published_destination_tree(&[&first, &second], PublicationDestination::Private),
        );

        let selection = select_missing_destination_drafts(
            &remote,
            repository,
            head,
            "ccm-matrices",
            PublicationDestination::Private,
            &[&first, &second],
            &CancellationToken::new(),
        )
        .unwrap();

        assert_eq!(selection.already_present, 2);
        assert!(selection.pending.is_empty());
        assert_eq!(remote.read_ref(repository, "main").unwrap(), head);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repository_batch_ledgers_advance_at_each_committed_boundary() {
        let mut capacity = ledger("private-test-0001", "a".repeat(40).as_str());
        let transaction = ContentDigest::sha256(b"transaction");
        let first = vec![
            TransportPart {
                sequence: 0,
                repository_path: "objects/sha256/aa/payload.part".to_owned(),
                size_bytes: 90,
                content_digest: ContentDigest::sha256(b"payload"),
            },
            TransportPart {
                sequence: 1,
                repository_path: "manifests/aa/manifest.json".to_owned(),
                size_bytes: 10,
                content_digest: ContentDigest::sha256(b"manifest"),
            },
        ];
        advance_capacity_ledger_for_repository_batch(
            &mut capacity,
            &transaction,
            0,
            &"a".repeat(40),
            &first,
        )
        .unwrap();
        assert_eq!(capacity.first_seen_immutable_payload_bytes, 90);
        assert_eq!(capacity.manifest_index_receipt_bytes, 10);
        let first_reconciliation = capacity.reconciliation_digest.clone();

        let second = vec![TransportPart {
            sequence: 0,
            repository_path: "indexes/family/aa.json".to_owned(),
            size_bytes: 12,
            content_digest: ContentDigest::sha256(b"index"),
        }];
        advance_capacity_ledger_for_repository_batch(
            &mut capacity,
            &transaction,
            1,
            &"b".repeat(40),
            &second,
        )
        .unwrap();
        assert_eq!(capacity.first_seen_immutable_payload_bytes, 90);
        assert_eq!(capacity.manifest_index_receipt_bytes, 22);
        assert_eq!(capacity.last_reconciled_commit, "b".repeat(40));
        assert_ne!(capacity.reconciliation_digest, first_reconciliation);
    }

    #[test]
    fn missing_live_sidecars_are_initialized_without_replacing_bootstrap_content() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!("managed-bootstrap-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let staging = root.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let remote = FilesystemMemoryRemote::new(staging.clone());
        let repository = "https://github.com/example-org/restricted-quadrature-0001.git";
        let authorized = "example-org/restricted-quadrature-0001";
        let stored_object_digest = ContentDigest::sha256(b"stored-object");
        let stored_object_path = format!(
            "objects/sha256/{}/{}.part",
            &stored_object_digest.0[..2],
            stored_object_digest.0
        );
        remote.insert_repository(
            repository.to_owned(),
            "bootstrap-head".to_owned(),
            BTreeMap::from([
                ("README.md".to_owned(), b"bootstrap repository".to_vec()),
                (stored_object_path, b"stored-object".to_vec()),
            ]),
        );
        let session = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            authorized,
            RepositoryPermission::Write,
        );
        let (head, capacity) = ensure_managed_shard_sidecars(
            &remote,
            &session,
            repository,
            authorized,
            "main",
            "restricted-quadrature-0001",
            "quadrature",
            "ab",
            &staging.join("managed-targets/private"),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            None,
            &crate::PrivatePublicationLeasePolicy::default(),
        )
        .unwrap();
        assert_eq!(capacity.first_seen_immutable_payload_bytes, 13);
        assert!(remote
            .immutable_path_digest(repository, &head, "README.md")
            .unwrap()
            .is_some());
        assert!(remote
            .immutable_path_digest(repository, &head, crate::DEFAULT_CAPACITY_LEDGER_PATH)
            .unwrap()
            .is_some());
        // Index partitions are created by the first family batch that uses a
        // prefix; ledger bootstrap must not create empty prefixes or replay
        // an initialization commit on every later family.
        assert!(remote
            .immutable_path_digest(repository, &head, "indexes/quadrature/ab.json")
            .unwrap()
            .is_none());
        assert!(!staging
            .join("managed-targets/private")
            .join(crate::DEFAULT_CAPACITY_LEDGER_PATH)
            .exists());
        assert!(!staging
            .join("managed-targets/private/indexes/quadrature/ab.json")
            .exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn managed_adapter_completes_dual_target_resumable_publication() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!("managed-publication-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let staging = root.join("staging");
        fs::create_dir_all(&staging).unwrap();
        let draft = fixture_draft(&staging);
        let private_authorized = repository(
            "example-org",
            "ccm-matrices",
            PublicationDestination::Private,
        );
        let public_authorized = repository(
            "example-org",
            "ccm-matrices",
            PublicationDestination::Public,
        );
        let private_repository = format!("https://github.com/{private_authorized}.git");
        let public_repository = format!("https://github.com/{public_authorized}.git");
        let private_head = "private-head-0".to_owned();
        let public_head = "public-head-0".to_owned();
        let private_shard = shard_id("ccm-matrices", PublicationDestination::Private);
        let public_shard = shard_id("ccm-matrices", PublicationDestination::Public);
        let prefix = &draft.manifest.semantic_digest.0[..2];
        let remote = FilesystemMemoryRemote::new(staging.clone());
        for (repository, head, shard) in [
            (
                private_repository.as_str(),
                &private_head,
                private_shard.as_str(),
            ),
            (
                public_repository.as_str(),
                &public_head,
                public_shard.as_str(),
            ),
        ] {
            let index = ShardIndexPartition::rebuild("ccm-matrices", prefix, Vec::new()).unwrap();
            remote.insert_repository(
                repository.to_owned(),
                head.clone(),
                BTreeMap::from([
                    (
                        format!("indexes/ccm-matrices/{prefix}.json"),
                        canonical_json_bytes(&index).unwrap(),
                    ),
                    (
                        crate::DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
                        canonical_json_bytes(&ledger(shard, head)).unwrap(),
                    ),
                ]),
            );
        }
        let sessions = BTreeMap::from([
            (
                PublicationDestination::Private,
                AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    &private_authorized,
                    RepositoryPermission::Write,
                ),
            ),
            (
                PublicationDestination::Public,
                AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    &public_authorized,
                    RepositoryPermission::Write,
                ),
            ),
        ]);
        let context = ManagedPublicationPlanningContext {
            owner: "example-org".to_owned(),
            principal: "test-owner".to_owned(),
            target: PublicationTarget::Both,
            target_heads: BTreeMap::from([
                (PublicationDestination::Private, private_head.clone()),
                (PublicationDestination::Public, public_head.clone()),
            ]),
            capacity_ledgers: BTreeMap::from([
                (
                    PublicationDestination::Private,
                    ledger(&private_shard, &private_head),
                ),
                (
                    PublicationDestination::Public,
                    ledger(&public_shard, &public_head),
                ),
            ]),
            event_unix_seconds: 123,
        };
        let mut prepared =
            prepare_managed_artifact_publication(&draft, &context, &sessions).unwrap();
        let checkpoints = PublicationJournalStore::new(root.join("journals"));
        let transaction_id = prepared
            .coordinated
            .journal
            .as_ref()
            .unwrap()
            .transaction_id
            .clone();
        checkpoints
            .save(prepared.coordinated.journal.as_ref().unwrap())
            .unwrap();
        let remote_map: BTreeMap<PublicationDestination, &dyn RemoteGitStore> = BTreeMap::from([
            (
                PublicationDestination::Private,
                &remote as &dyn RemoteGitStore,
            ),
            (
                PublicationDestination::Public,
                &remote as &dyn RemoteGitStore,
            ),
        ]);
        let report = execute_prepared_managed_artifact_publication(
            &mut prepared,
            &remote_map,
            &checkpoints,
            &CancellationToken::new(),
            &sessions,
            &staging,
            &ResourcePolicy::default(),
            40,
            123,
            false,
        )
        .unwrap();
        assert!(report.completed);
        assert_eq!(report.transaction_id, transaction_id);
        let journal = prepared.coordinated.journal.as_ref().unwrap();
        assert!(journal.complete());
        assert_eq!(checkpoints.load_latest(&transaction_id).unwrap(), *journal);
        for destination in [
            PublicationDestination::Private,
            PublicationDestination::Public,
        ] {
            let target = &journal.targets[&destination];
            let head = remote.read_ref(&target.repository, "main").unwrap();
            assert!(remote
                .immutable_path_digest(
                    &target.repository,
                    &head,
                    &format!(
                        "transactions/{transaction_id}/{}/receipt.json",
                        match destination {
                            PublicationDestination::Private => "private",
                            PublicationDestination::Public => "public",
                        }
                    ),
                )
                .unwrap()
                .is_some());
        }
        // A later author session has fresh authorization evidence, but the
        // artifact and its stable publication identity are unchanged. The
        // completed remote receipt must win over regenerated local plan bytes.
        for target in prepared
            .coordinated
            .journal
            .as_mut()
            .unwrap()
            .targets
            .values_mut()
        {
            target.permission_evidence.evidence_digest =
                ContentDigest::sha256(b"refreshed authorization evidence");
        }
        let resumed =
            completed_remote_publication(&prepared, &remote_map, &CancellationToken::new())
                .unwrap()
                .expect("the completed remote transaction should be reusable");
        assert!(resumed.completed);
        assert_eq!(resumed.transaction_id, transaction_id);
        assert_eq!(resumed.steps_executed, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore = "explicit read-only live GitHub acceptance preflight"]
    fn live_managed_routes_and_owner_permissions_are_read_only() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target/test-tmp")
            .join(format!("managed-live-preflight-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let probe = crate::GitHubCredentialApiProbe::default();
        let cancellation = CancellationToken::new();
        let mut principals = BTreeSet::new();
        for destination in [
            PublicationDestination::Private,
            PublicationDestination::Public,
        ] {
            let authorized = repository("TeamXcelerator", "quadrature", destination);
            let session = probe.probe_repository(&authorized).unwrap();
            principals.insert(session.evidence().principal.clone());
            let remote = crate::GitCliRemoteStore::new(
                root.join(visibility_name(destination)).join("transport"),
                root.join(visibility_name(destination)).join("staging"),
                "Xcelerator live preflight",
                "xcelerator-toolkit@users.noreply.github.com",
            )
            .unwrap();
            verify_bootstrap_registry_route(
                &remote,
                "TeamXcelerator",
                "quadrature",
                destination,
                &cancellation,
            )
            .unwrap();
            let shard_url = repository_url("TeamXcelerator", "quadrature", destination);
            let head = remote.read_ref(&shard_url, "main").unwrap();
            let has_ledger = remote
                .immutable_path_digest(&shard_url, &head, crate::DEFAULT_CAPACITY_LEDGER_PATH)
                .unwrap()
                .is_some();
            eprintln!(
                "managed live preflight: destination={destination:?} principal={} head={head} ledger_present={has_ledger}",
                session.evidence().principal
            );
            remote.cleanup_session(&shard_url).unwrap();
        }
        assert_eq!(principals.len(), 1);
        let _ = fs::remove_dir_all(root);
    }
}
