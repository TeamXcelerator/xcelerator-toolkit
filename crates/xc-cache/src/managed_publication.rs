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
    TopologyRegistry, TopologyShardRoute, TopologyShardStatus, TopologyTrustPolicy,
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
    pub artifacts: Vec<ManagedPublicationExecutionReport>,
    pub all_completed: bool,
    #[serde(default)]
    pub current_tree_paths_removed: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManagedPublicationExecutionPhaseSummary {
    pub phase_index: usize,
    pub phase_digest: ContentDigest,
    pub relative_report_path: String,
    pub artifact_count: usize,
    pub completed_artifact_count: usize,
    pub all_completed: bool,
    pub current_tree_paths_removed: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManagedCumulativePublicationExecutionReport {
    pub schema_version: u32,
    pub phases: Vec<ManagedPublicationExecutionPhaseSummary>,
    pub artifacts: Vec<ManagedPublicationExecutionReport>,
    pub all_completed: bool,
    pub current_tree_paths_removed: usize,
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
        crate::RemoteShardReader::new(remote, 4 * 1024 * 1024)?
            .read_json::<crate::ShardIndexPartition>(repository, revision, &path, cancellation)?;
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
    family: &str,
    semantic_prefix: &str,
    staging_root: &Path,
    resources: &xc_core::ResourcePolicy,
    cancellation: &xc_core::CancellationToken,
) -> Result<(String, CapacityLedger), CacheError> {
    session.require_write_for(session.evidence().principal.as_str(), authorized_repository)?;
    for _ in 0..4 {
        let head = remote.read_ref(repository, branch)?;
        let ledger_digest =
            remote.immutable_path_digest(repository, &head, crate::DEFAULT_CAPACITY_LEDGER_PATH)?;
        let index_path = format!("indexes/{family}/{semantic_prefix}.json");
        let index_digest = remote.immutable_path_digest(repository, &head, &index_path)?;
        if ledger_digest.is_some() && index_digest.is_some() {
            let ledger = read_capacity_ledger(remote, repository, &head, cancellation)?;
            if ledger.shard_id != shard_id {
                return Err(CacheError::InvalidManifest(format!(
                    "live capacity ledger belongs to {:?}, expected {shard_id:?}",
                    ledger.shard_id
                )));
            }
            return Ok((head, ledger));
        }

        let mut parts = Vec::new();
        let mut staged_bytes = 0u64;
        if ledger_digest.is_none() {
            let payload_bytes =
                bounded_prefix_bytes(remote, repository, &head, "objects", true, cancellation)?;
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
                        &head,
                        prefix,
                        false,
                        cancellation,
                    )?)
                    .ok_or_else(|| {
                        CacheError::ResourceLimit(
                            "bootstrap metadata accounting exceeds u64".to_owned(),
                        )
                    })
            })?;
            let reconciliation_digest = canonical_digest(&BootstrapReconciliation {
                schema_version: 1,
                shard_id,
                revision: &head,
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
                last_reconciled_commit: head.clone(),
                reconciliation_digest,
            };
            ledger.validate()?;
            parts.push(crate::publication_staging::stage_publication_bytes(
                staging_root,
                crate::DEFAULT_CAPACITY_LEDGER_PATH,
                &crate::protocol::canonical_json_bytes(&ledger)?,
                resources,
                cancellation,
                &mut staged_bytes,
            )?);
        }
        if index_digest.is_none() {
            let index = crate::ShardIndexPartition::rebuild(
                family.to_owned(),
                semantic_prefix.to_owned(),
                Vec::new(),
            )?;
            parts.push(crate::publication_staging::stage_publication_bytes(
                staging_root,
                &index_path,
                &crate::protocol::canonical_json_bytes(&index)?,
                resources,
                cancellation,
                &mut staged_bytes,
            )?);
        }
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
        let outcome = remote.compare_and_swap_commit(&crate::RemoteCommitRequest {
            repository: repository.to_owned(),
            branch: branch.to_owned(),
            expected_head: head,
            message: format!("initialize Xcelerator v0.13.0 shard metadata for {family}"),
            parts,
            delete_paths: Vec::new(),
        })?;
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
    let mut manifest = draft.manifest.clone();
    manifest.assumptions.push(format!(
        "publication_visibility={}",
        visibility_name(destination)
    ));
    manifest.assumptions.sort();
    manifest.assumptions.dedup();
    manifest.validate()?;
    Ok(manifest)
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
            .map(|(destination, report)| format!("{destination:?}: {}", report.reasons.join("; ")))
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
        let context = ManagedPublicationPlanningContext {
            owner: owner.to_owned(),
            principal: principal.clone(),
            target,
            target_heads: heads,
            capacity_ledgers: ledgers,
            event_unix_seconds,
        };
        let mut prepared = prepare_managed_artifact_publication(draft, &context, &sessions)?;
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
        let remote_refs: BTreeMap<PublicationDestination, &dyn crate::RemoteGitStore> = remotes
            .iter()
            .map(|(destination, remote)| (*destination, remote as &dyn crate::RemoteGitStore))
            .collect();
        let maximum_steps = draft
            .encoding
            .ordered_parts
            .len()
            .saturating_mul(4)
            .saturating_add(32);
        let report = execute_prepared_managed_artifact_publication(
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
        )?;
        reports.push(report);
        if replace_existing_semantic {
            replacement_families.insert(draft.family.clone());
        }
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
        artifacts: reports,
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
    let mut by_family = BTreeMap::<String, Vec<&CanonicalProductionDraft>>::new();
    for draft in drafts {
        by_family
            .entry(draft.family.clone())
            .or_default()
            .push(draft);
    }
    let groups = by_family.into_values().collect::<Vec<_>>();
    let started = std::time::Instant::now();
    let reports = groups
        .par_iter()
        .map(|family_drafts| {
            let family = family_drafts
                .first()
                .map(|draft| draft.family.as_str())
                .unwrap_or("empty");
            let family_started = std::time::Instant::now();
            let report = execute_managed_family_drafts_on_github(
                family_drafts,
                target,
                owner,
                journal_root,
                resources,
                replace_existing_semantic,
            )?;
            eprintln!(
                "publication family {family}: {} artifacts in {:.3}s",
                report.artifacts.len(),
                family_started.elapsed().as_secs_f64()
            );
            Ok(report)
        })
        .collect::<Result<Vec<_>, CacheError>>()?;

    let artifacts = reports
        .iter()
        .flat_map(|report| report.artifacts.iter().cloned())
        .collect::<Vec<_>>();
    let current_tree_paths_removed = reports.iter().fold(0usize, |total, report| {
        total.saturating_add(report.current_tree_paths_removed)
    });
    eprintln!(
        "publication execution: {} artifacts across {} shard families in {:.3}s",
        artifacts.len(),
        reports.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(ManagedRunPublicationReport {
        schema_version: 1,
        all_completed: reports.iter().all(|report| report.all_completed),
        artifacts,
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
        assert!(remote
            .immutable_path_digest(repository, &head, "indexes/quadrature/ab.json")
            .unwrap()
            .is_some());
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
