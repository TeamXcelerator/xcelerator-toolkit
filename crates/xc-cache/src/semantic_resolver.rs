//! Complete metadata resolution over bounded, selective remote shard reads.

use crate::{
    ArtifactAssuranceState, ArtifactDisposition, CacheError, CacheNetworkRegistry, CacheVisibility,
    CanonicalArtifactManifest, ContentDigest, PublicationDestination, PublicationReceipt,
    RemoteDocument, RemoteFabricTrustPolicy, RemoteGitStore, RemotePublicationEvidence,
    RemoteReadReport, RemoteShardReader, RemoteTopologySource, RepositoryPublicationBatch,
    RevocationIndexPartition, RevocationRecord, RevocationScope, SemanticKeyEnvelope,
    ShardIndexEntry, ToolkitVersion, TopologyShardStatus, TopologyTrustPolicy,
    TransportEncodingRecord, TrustedRepositoryRole,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use xc_core::CancellationToken;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteResolverOverlay {
    pub name: String,
    pub visibility: CacheVisibility,
    pub topology_source: RemoteTopologySource,
    pub topology_trust: TopologyTrustPolicy,
    pub fabric_trust: RemoteFabricTrustPolicy,
    pub network: CacheNetworkRegistry,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteSemanticQuery {
    pub family: String,
    pub semantic_key: SemanticKeyEnvelope,
    pub minimum_assurance: ArtifactAssuranceState,
    pub allowed_scalar_backends: BTreeSet<String>,
    pub minimum_precision_bits: Option<u64>,
    pub required_configuration_digest: Option<ContentDigest>,
    pub required_provenance_evidence_digests: BTreeSet<ContentDigest>,
    pub current_toolkit_version: ToolkitVersion,
    pub accepted_publication_policy_digests: BTreeSet<ContentDigest>,
    pub allow_deprecated: bool,
    pub evaluation_unix_seconds: u64,
    pub maximum_topology_bytes: u64,
    pub maximum_index_bytes: u64,
    pub maximum_manifest_bytes: u64,
    pub maximum_encoding_bytes: u64,
    pub maximum_receipt_bytes: u64,
    pub maximum_revocation_partition_bytes: u64,
    pub maximum_dependency_depth: u32,
    pub maximum_dependency_count: u64,
}

impl RemoteSemanticQuery {
    pub fn validate(&self) -> Result<ContentDigest, CacheError> {
        if self.family.trim().is_empty()
            || self.accepted_publication_policy_digests.is_empty()
            || self
                .accepted_publication_policy_digests
                .iter()
                .any(|digest| !digest.validate())
            || self
                .allowed_scalar_backends
                .iter()
                .any(|backend| backend.trim().is_empty())
            || self.minimum_precision_bits == Some(0)
            || self
                .required_configuration_digest
                .as_ref()
                .is_some_and(|digest| !digest.validate())
            || self
                .required_provenance_evidence_digests
                .iter()
                .any(|digest| !digest.validate())
            || [
                self.maximum_topology_bytes,
                self.maximum_index_bytes,
                self.maximum_manifest_bytes,
                self.maximum_encoding_bytes,
                self.maximum_receipt_bytes,
                self.maximum_revocation_partition_bytes,
                self.maximum_dependency_count,
            ]
            .contains(&0)
        {
            return Err(CacheError::InvalidManifest(
                "remote semantic query identity, policy, or bounds are invalid".to_owned(),
            ));
        }
        self.semantic_key.digest()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionRejectionStage {
    Topology,
    Route,
    Index,
    Policy,
    Disposition,
    Assurance,
    Compatibility,
    Revocation,
    Manifest,
    Receipt,
    Encoding,
    Dependency,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionRejection {
    pub overlay: String,
    pub shard_id: Option<String>,
    pub repository: Option<String>,
    pub manifest_digest: Option<ContentDigest>,
    pub stage: ResolutionRejectionStage,
    pub reason: String,
    pub revocation: Option<RevocationRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedRemoteArtifact {
    pub family: String,
    pub semantic_digest: ContentDigest,
    pub overlay: String,
    pub visibility: CacheVisibility,
    pub shard_id: String,
    pub authorized_repository: String,
    pub repository: String,
    pub revision: String,
    pub index: ShardIndexEntry,
    pub manifest: CanonicalArtifactManifest,
    pub encoding: TransportEncodingRecord,
    pub receipt: RemotePublicationEvidence,
    pub index_source: RemoteReadReport,
    pub manifest_source: RemoteReadReport,
    pub encoding_source: RemoteReadReport,
    pub receipt_source: RemoteReadReport,
    pub dependencies: Vec<ResolvedRemoteArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticResolutionReport {
    pub schema_version: u32,
    pub semantic_digest: ContentDigest,
    pub resolved_semantic_key: SemanticKeyEnvelope,
    pub selected: Option<ResolvedRemoteArtifact>,
    pub rejections: Vec<ResolutionRejection>,
}

struct ResolutionContext<'a> {
    remote: &'a dyn RemoteGitStore,
    cancellation: &'a CancellationToken,
    query: &'a RemoteSemanticQuery,
    overlays: &'a [RemoteResolverOverlay],
    rejections: Vec<ResolutionRejection>,
    revocations: BTreeMap<(String, String, String), Option<RevocationIndexPartition>>,
    active_dependencies: BTreeSet<(String, ContentDigest)>,
    dependency_count: u64,
}

#[derive(Clone)]
struct ExpectedArtifact<'a> {
    manifest_digest: &'a ContentDigest,
    payload_digest: &'a ContentDigest,
}

/// Resolves one semantic artifact through registry routing and shard-local indexes.
///
/// # Mathematical semantics
/// Uses the canonical semantic-key digest and compatibility policy to select an
/// immutable manifest without placing object identities in the top-level
/// family registry.
///
/// # Precision
/// Backend and minimum precision constraints are checked from the query and
/// manifest. Resolution never converts payload scalars or lowers precision.
///
/// # Failure states
/// Invalid queries, cancellation, topology or index corruption, rollback,
/// revocation, incompatible candidates, and bounded-read violations return
/// `CacheError` with no payload presented as accepted.
///
/// # Assurance and validity
/// The report proves metadata, index, manifest, encoding, and receipt checks.
/// Consumers must still materialize and validate payload bytes at the requested
/// validation level.
///
/// # Cache effects
/// Performs bounded immutable remote reads only. It does not clone a full shard,
/// write a local artifact, mutate GitHub, or publish a receipt.
///
/// # Example
/// Overlay use is shown in `crates/xc-cache/examples/cache_overlay.rs`.
pub fn resolve_remote_semantic_artifact(
    remote: &dyn RemoteGitStore,
    cancellation: &CancellationToken,
    query: &RemoteSemanticQuery,
    overlays: &[RemoteResolverOverlay],
) -> Result<SemanticResolutionReport, CacheError> {
    let semantic_digest = query.validate()?;
    if overlays.is_empty() {
        return Err(CacheError::InvalidManifest(
            "remote semantic resolution requires at least one overlay".to_owned(),
        ));
    }
    let mut names = BTreeSet::new();
    for overlay in overlays {
        if overlay.name.trim().is_empty()
            || !names.insert(overlay.name.as_str())
            || !matches!(
                overlay.visibility,
                CacheVisibility::Private | CacheVisibility::Public
            )
        {
            return Err(CacheError::InvalidManifest(
                "remote resolver overlays must be uniquely named private/public sources".to_owned(),
            ));
        }
        overlay.topology_source.validate()?;
        overlay.network.validate()?;
    }
    let mut context = ResolutionContext {
        remote,
        cancellation,
        query,
        overlays,
        rejections: Vec::new(),
        revocations: BTreeMap::new(),
        active_dependencies: BTreeSet::new(),
        dependency_count: 0,
    };
    let selected = resolve_identity(&mut context, &query.family, &semantic_digest, None, 0)?;
    Ok(SemanticResolutionReport {
        schema_version: 1,
        semantic_digest,
        resolved_semantic_key: query.semantic_key.clone(),
        selected,
        rejections: context.rejections,
    })
}

fn resolve_identity(
    context: &mut ResolutionContext<'_>,
    family: &str,
    semantic_digest: &ContentDigest,
    expected: Option<ExpectedArtifact<'_>>,
    depth: u32,
) -> Result<Option<ResolvedRemoteArtifact>, CacheError> {
    context
        .cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if depth > context.query.maximum_dependency_depth {
        return Err(CacheError::ResourceLimit(format!(
            "dependency resolution depth {depth} exceeds {}",
            context.query.maximum_dependency_depth
        )));
    }
    for overlay in context.overlays {
        let topology_revision = match context.remote.read_ref(
            &overlay.topology_source.repository,
            &overlay.topology_source.branch,
        ) {
            Ok(revision) => revision,
            Err(error) => {
                reject(
                    context,
                    overlay,
                    None,
                    None,
                    None,
                    ResolutionRejectionStage::Topology,
                    error.to_string(),
                    None,
                );
                continue;
            }
        };
        if let Err(error) = overlay.fabric_trust.verify_repository(
            TrustedRepositoryRole::Registry,
            &overlay.topology_source.repository,
            &overlay.topology_source.branch,
            &topology_revision,
            context.query.evaluation_unix_seconds,
        ) {
            reject(
                context,
                overlay,
                None,
                Some(&overlay.topology_source.repository),
                None,
                ResolutionRejectionStage::Topology,
                error.to_string(),
                None,
            );
            continue;
        }
        let topology = match RemoteShardReader::new(
            context.remote,
            context
                .query
                .maximum_topology_bytes
                .min(overlay.topology_source.maximum_registry_bytes),
        )
        .and_then(|reader| {
            reader.load_trusted_topology(
                &overlay.topology_source.repository,
                &topology_revision,
                &overlay.topology_source.registry_path,
                &overlay.topology_trust,
                context.cancellation,
            )
        }) {
            Ok(topology) => topology,
            Err(error) => {
                reject(
                    context,
                    overlay,
                    None,
                    None,
                    None,
                    ResolutionRejectionStage::Topology,
                    error.to_string(),
                    None,
                );
                continue;
            }
        };
        if let Err(error) = overlay
            .fabric_trust
            .verify_policy_digest(&topology.registry.policy_digest)
        {
            reject(
                context,
                overlay,
                None,
                Some(&overlay.topology_source.repository),
                None,
                ResolutionRejectionStage::Policy,
                error.to_string(),
                None,
            );
            continue;
        }
        if !context
            .query
            .accepted_publication_policy_digests
            .contains(&topology.registry.policy_digest)
        {
            reject(
                context,
                overlay,
                None,
                None,
                None,
                ResolutionRejectionStage::Policy,
                format!(
                    "topology policy {} is not accepted by the query",
                    topology.registry.policy_digest
                ),
                None,
            );
            continue;
        }
        let Some(route) = topology.registry.route(family, overlay.visibility) else {
            reject(
                context,
                overlay,
                None,
                None,
                None,
                ResolutionRejectionStage::Route,
                format!("no {family:?} route for {:?}", overlay.visibility),
                None,
            );
            continue;
        };
        let mut shards = route.ordered_shards.iter().collect::<Vec<_>>();
        shards.sort_by_key(|shard| shard.sequence);
        for shard in shards {
            if shard.status == TopologyShardStatus::IncidentHold {
                reject(
                    context,
                    overlay,
                    Some(&shard.shard_id),
                    None,
                    None,
                    ResolutionRejectionStage::Route,
                    "shard is on incident hold".to_owned(),
                    None,
                );
                continue;
            }
            let Some(endpoint) = overlay.network.endpoint_for_shard(&shard.endpoint_id) else {
                reject(
                    context,
                    overlay,
                    Some(&shard.shard_id),
                    None,
                    None,
                    ResolutionRejectionStage::Route,
                    "shard endpoint is absent from the network registry".to_owned(),
                    None,
                );
                continue;
            };
            if !endpoint.enabled_for_read || endpoint.visibility != overlay.visibility {
                reject(
                    context,
                    overlay,
                    Some(&shard.shard_id),
                    None,
                    None,
                    ResolutionRejectionStage::Route,
                    "shard endpoint is not readable for this overlay".to_owned(),
                    None,
                );
                continue;
            }
            let repository = endpoint.preferred_clone_url();
            let authorized_repository = format!("{}/{}", endpoint.owner, endpoint.repository);
            let revision = match context.remote.read_ref(&repository, &endpoint.branch) {
                Ok(revision) => revision,
                Err(error) => {
                    reject(
                        context,
                        overlay,
                        Some(&shard.shard_id),
                        Some(&repository),
                        None,
                        ResolutionRejectionStage::Index,
                        error.to_string(),
                        None,
                    );
                    continue;
                }
            };
            if let Err(error) = overlay.fabric_trust.verify_repository(
                TrustedRepositoryRole::Shard,
                &authorized_repository,
                &endpoint.branch,
                &revision,
                context.query.evaluation_unix_seconds,
            ) {
                reject(
                    context,
                    overlay,
                    Some(&shard.shard_id),
                    Some(&authorized_repository),
                    None,
                    ResolutionRejectionStage::Route,
                    error.to_string(),
                    None,
                );
                continue;
            }
            let semantic_prefix = &semantic_digest.0[..2];
            let index_path = format!("indexes/{family}/{semantic_prefix}.json");
            let index =
                match RemoteShardReader::new(context.remote, context.query.maximum_index_bytes)
                    .and_then(|reader| {
                        reader.load_index_partition(
                            &repository,
                            &revision,
                            &index_path,
                            family,
                            semantic_prefix,
                            context.cancellation,
                        )
                    }) {
                    Ok(index) => index,
                    Err(error) => {
                        reject(
                            context,
                            overlay,
                            Some(&shard.shard_id),
                            Some(&repository),
                            None,
                            ResolutionRejectionStage::Index,
                            error.to_string(),
                            None,
                        );
                        continue;
                    }
                };
            if let Some(revocation) = active_revocation(
                context,
                &repository,
                &revision,
                RevocationScope::Semantic,
                semantic_digest,
            )? {
                reject(
                    context,
                    overlay,
                    Some(&shard.shard_id),
                    Some(&repository),
                    None,
                    ResolutionRejectionStage::Revocation,
                    revocation.reason.clone(),
                    Some(revocation),
                );
                continue;
            }
            let mut candidates = index
                .value
                .lookup(semantic_digest)
                .cloned()
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                right
                    .producer_toolkit_version
                    .cmp(&left.producer_toolkit_version)
                    .then_with(|| right.achieved_assurance.cmp(&left.achieved_assurance))
                    .then_with(|| left.manifest_digest.cmp(&right.manifest_digest))
            });
            if candidates.is_empty() {
                reject(
                    context,
                    overlay,
                    Some(&shard.shard_id),
                    Some(&repository),
                    None,
                    ResolutionRejectionStage::Index,
                    "semantic identity is absent from the shard partition".to_owned(),
                    None,
                );
                continue;
            }
            for entry in candidates {
                if let Some((stage, reason)) =
                    entry_rejection(context.query, &entry, expected.as_ref())
                {
                    reject(
                        context,
                        overlay,
                        Some(&shard.shard_id),
                        Some(&repository),
                        Some(&entry.manifest_digest),
                        stage,
                        reason,
                        None,
                    );
                    continue;
                }
                let repository_digest = repository_identity_digest(&authorized_repository);
                let mut active = None;
                for (scope, identity) in [
                    (RevocationScope::Payload, &entry.canonical_payload_digest),
                    (RevocationScope::Manifest, &entry.manifest_digest),
                    (RevocationScope::Policy, &topology.registry.policy_digest),
                    (RevocationScope::Repository, &repository_digest),
                ]
                .into_iter()
                .chain(
                    entry
                        .transport_digests
                        .iter()
                        .map(|identity| (RevocationScope::Transport, identity)),
                ) {
                    if let Some(revocation) =
                        active_revocation(context, &repository, &revision, scope, identity)?
                    {
                        active = Some(revocation);
                        break;
                    }
                }
                if let Some(revocation) = active {
                    reject(
                        context,
                        overlay,
                        Some(&shard.shard_id),
                        Some(&repository),
                        Some(&entry.manifest_digest),
                        ResolutionRejectionStage::Revocation,
                        revocation.reason.clone(),
                        Some(revocation),
                    );
                    continue;
                }
                match resolve_candidate(
                    context,
                    overlay,
                    family,
                    semantic_digest,
                    expected.clone(),
                    depth,
                    &shard.shard_id,
                    &authorized_repository,
                    &repository,
                    &endpoint.branch,
                    &revision,
                    &index_path,
                    &index,
                    &entry,
                ) {
                    Ok(artifact) => return Ok(Some(artifact)),
                    Err(error) => reject(
                        context,
                        overlay,
                        Some(&shard.shard_id),
                        Some(&repository),
                        Some(&entry.manifest_digest),
                        classify_candidate_error(&error),
                        error.to_string(),
                        None,
                    ),
                }
            }
        }
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn resolve_candidate(
    context: &mut ResolutionContext<'_>,
    overlay: &RemoteResolverOverlay,
    family: &str,
    semantic_digest: &ContentDigest,
    expected: Option<ExpectedArtifact<'_>>,
    depth: u32,
    shard_id: &str,
    authorized_repository: &str,
    repository: &str,
    branch: &str,
    revision: &str,
    _index_path: &str,
    index: &RemoteDocument<crate::ShardIndexPartition>,
    entry: &ShardIndexEntry,
) -> Result<ResolvedRemoteArtifact, CacheError> {
    entry.validate()?;
    if entry.disposition == ArtifactDisposition::Revoked
        || entry.disposition == ArtifactDisposition::Quarantined
        || (entry.disposition == ArtifactDisposition::Deprecated && !context.query.allow_deprecated)
    {
        return Err(CacheError::InvalidManifest(format!(
            "candidate disposition {:?} is not admissible",
            entry.disposition
        )));
    }
    if entry.achieved_assurance < context.query.minimum_assurance {
        return Err(CacheError::InvalidManifest(format!(
            "candidate assurance {:?} is below {:?}",
            entry.achieved_assurance, context.query.minimum_assurance
        )));
    }
    if context.query.current_toolkit_version < entry.minimum_reader_version {
        return Err(CacheError::InvalidManifest(
            "candidate requires a newer toolkit reader".to_owned(),
        ));
    }
    let family_policy =
        crate::artifact_compatibility_policy(family, &context.query.semantic_key.artifact_kind)?;
    if entry.producer_toolkit_version < family_policy.minimum_producer_version {
        return Err(CacheError::InvalidManifest(format!(
            "candidate producer toolkit {} precedes the family floor {}",
            entry.producer_toolkit_version, family_policy.minimum_producer_version
        )));
    }
    if expected.as_ref().is_some_and(|expected| {
        expected.manifest_digest != &entry.manifest_digest
            || expected.payload_digest != &entry.canonical_payload_digest
    }) {
        return Err(CacheError::InvalidManifest(
            "dependency candidate does not match its exact manifest and payload identities"
                .to_owned(),
        ));
    }
    for (scope, identity) in [
        (RevocationScope::Payload, &entry.canonical_payload_digest),
        (RevocationScope::Manifest, &entry.manifest_digest),
    ] {
        if let Some(revocation) = active_revocation(context, repository, revision, scope, identity)?
        {
            return Err(CacheError::InvalidManifest(format!(
                "candidate is revoked: {}",
                revocation.reason
            )));
        }
    }
    let manifest_path = format!(
        "manifests/{}/{}.json",
        &semantic_digest.0[..2],
        entry.manifest_digest.0
    );
    let manifest: RemoteDocument<CanonicalArtifactManifest> = RemoteShardReader::new(
        context.remote,
        context.query.maximum_manifest_bytes,
    )?
    .read_json(repository, revision, &manifest_path, context.cancellation)?;
    let manifest_digest = manifest.value.digest()?;
    if manifest_digest != entry.manifest_digest
        || manifest.source.content_digest != manifest_digest
        || manifest.value.artifact_family != family
        || manifest.value.semantic_digest != *semantic_digest
        || manifest.value.payload_digest != entry.canonical_payload_digest
        || manifest.value.producer_toolkit_version != entry.producer_toolkit_version
        || manifest.value.minimum_reader_version != entry.minimum_reader_version
        || context.query.current_toolkit_version < manifest.value.minimum_reader_version
        || manifest
            .value
            .maximum_reader_version
            .as_ref()
            .is_some_and(|maximum| &context.query.current_toolkit_version > maximum)
    {
        return Err(CacheError::InvalidManifest(
            "canonical manifest does not match its index entry or reader compatibility".to_owned(),
        ));
    }
    if !context.query.allowed_scalar_backends.is_empty()
        && !context
            .query
            .allowed_scalar_backends
            .contains(&manifest.value.canonical_payload.scalar_backend)
    {
        return Err(CacheError::InvalidManifest(format!(
            "candidate scalar backend {:?} is outside the consumption policy",
            manifest.value.canonical_payload.scalar_backend
        )));
    }
    if context.query.minimum_precision_bits.is_some_and(|minimum| {
        manifest.value.canonical_payload.precision_bits.unwrap_or(0) < minimum
    }) {
        return Err(CacheError::InvalidManifest(
            "candidate precision is below the consumption policy".to_owned(),
        ));
    }
    if context
        .query
        .required_configuration_digest
        .as_ref()
        .is_some_and(|required| {
            required != &manifest.value.resolved_mathematical_configuration_digest
        })
    {
        return Err(CacheError::InvalidManifest(
            "candidate resolved configuration does not match consumption policy".to_owned(),
        ));
    }
    let expected_destination = match overlay.visibility {
        CacheVisibility::Private => PublicationDestination::Private,
        CacheVisibility::Public => PublicationDestination::Public,
        _ => unreachable!("overlay validation accepts only private/public"),
    };
    let receipt_path = format!(
        "transactions/{}/{}/receipt.json",
        entry.publication_transaction_id,
        match expected_destination {
            PublicationDestination::Private => "private",
            PublicationDestination::Public => "public",
        }
    );
    let reader = RemoteShardReader::new(context.remote, context.query.maximum_receipt_bytes)?;
    let (
        publication_evidence,
        receipt_source,
        policy_digest,
        transport_digest,
        legacy_metadata,
        batch_evidence,
    ) = match reader.read_json::<PublicationReceipt>(
        repository,
        revision,
        &receipt_path,
        context.cancellation,
    ) {
        Ok(receipt) => {
            receipt.value.validate()?;
            if receipt.source.content_digest != receipt.value.digest()?
                || receipt.value.transaction_id != entry.publication_transaction_id
                || receipt.value.destination != expected_destination
                || receipt.value.authorized_repository != authorized_repository
                || receipt.value.shard_id != shard_id
                || receipt.value.branch != branch
                || receipt.value.semantic_digest != *semantic_digest
                || receipt.value.canonical_payload_digest != entry.canonical_payload_digest
                || receipt.value.manifest_digest != entry.manifest_digest
                || receipt.value.metadata_file_digests.get(&manifest_path)
                    != Some(&entry.manifest_digest)
            {
                return Err(CacheError::InvalidManifest(
                    "publication receipt does not prove this discoverable index entry".to_owned(),
                ));
            }
            let metadata = Some(receipt.value.metadata_file_digests.clone());
            let policy = receipt.value.policy_digest.clone();
            let transport = receipt.value.transport_digest.clone();
            (
                RemotePublicationEvidence::ArtifactReceipt(Box::new(receipt.value)),
                receipt.source,
                policy,
                transport,
                metadata,
                None,
            )
        }
        Err(CacheError::NotFound(_)) => {
            let batch_path = format!(
                "transactions/batches/{}/{}.json",
                entry.publication_transaction_id,
                match expected_destination {
                    PublicationDestination::Private => "private",
                    PublicationDestination::Public => "public",
                }
            );
            let batch: RemoteDocument<RepositoryPublicationBatch> =
                reader.read_json(repository, revision, &batch_path, context.cancellation)?;
            batch.value.validate()?;
            if batch.source.content_digest != batch.value.digest()? {
                return Err(CacheError::InvalidManifest(
                    "repository batch bytes are not canonical".to_owned(),
                ));
            }
            let artifact = batch
                .value
                .artifacts
                .iter()
                .find(|artifact| {
                    artifact.semantic_digest == *semantic_digest
                        && artifact.manifest_digest == entry.manifest_digest
                })
                .ok_or_else(|| {
                    CacheError::InvalidManifest(
                        "repository batch does not contain the selected artifact".to_owned(),
                    )
                })?;
            if batch.value.batch_id.0 != entry.publication_transaction_id
                || batch.value.destination != expected_destination
                || batch.value.family != family
                || batch.value.authorized_repository != authorized_repository
                || batch.value.branch != branch
                || artifact.canonical_payload_digest != entry.canonical_payload_digest
                || artifact.transport_digest
                    != *entry.transport_digests.first().ok_or_else(|| {
                        CacheError::InvalidManifest(
                            "repository batch index has no transport".to_owned(),
                        )
                    })?
                || artifact.manifest_path != manifest_path
            {
                return Err(CacheError::InvalidManifest(
                    "repository batch does not prove this discoverable index entry".to_owned(),
                ));
            }
            let policy = batch.value.policy_digest.clone();
            let transport = artifact.transport_digest.clone();
            let evidence = artifact.provenance_evidence_digests.clone();
            (
                RemotePublicationEvidence::RepositoryBatch(Box::new(batch.value)),
                batch.source,
                policy,
                transport,
                None,
                Some(evidence),
            )
        }
        Err(error) => return Err(error),
    };
    if !context
        .query
        .accepted_publication_policy_digests
        .contains(&policy_digest)
    {
        return Err(CacheError::InvalidManifest(
            "publication policy is not accepted".to_owned(),
        ));
    }
    if !context
        .query
        .required_provenance_evidence_digests
        .is_empty()
        && match (legacy_metadata.as_ref(), batch_evidence.as_ref()) {
            (Some(metadata), _) => {
                let evidence = metadata.values().collect::<BTreeSet<_>>();
                !context
                    .query
                    .required_provenance_evidence_digests
                    .iter()
                    .all(|required| evidence.contains(required))
            }
            (None, Some(evidence)) => !context
                .query
                .required_provenance_evidence_digests
                .iter()
                .all(|required| evidence.contains(required)),
            (None, None) => true,
        }
    {
        return Err(CacheError::InvalidManifest(
            "publication evidence lacks required provenance evidence".to_owned(),
        ));
    }
    if !entry.transport_digests.contains(&transport_digest)
        || !manifest.value.transport_digests.contains(&transport_digest)
    {
        return Err(CacheError::InvalidManifest(
            "receipt transport is absent from the index or manifest".to_owned(),
        ));
    }
    if let Some(revocation) = active_revocation(
        context,
        repository,
        revision,
        RevocationScope::Transport,
        &transport_digest,
    )? {
        return Err(CacheError::InvalidManifest(format!(
            "transport is revoked: {}",
            revocation.reason
        )));
    }
    if let Some(revocation) = active_revocation(
        context,
        repository,
        revision,
        RevocationScope::Policy,
        &policy_digest,
    )? {
        return Err(CacheError::InvalidManifest(format!(
            "publication policy is revoked: {}",
            revocation.reason
        )));
    }
    let repository_digest = repository_identity_digest(authorized_repository);
    if let Some(revocation) = active_revocation(
        context,
        repository,
        revision,
        RevocationScope::Repository,
        &repository_digest,
    )? {
        return Err(CacheError::InvalidManifest(format!(
            "repository is revoked: {}",
            revocation.reason
        )));
    }
    let encoding_path = format!(
        "encodings/{}/{}.json",
        &entry.canonical_payload_digest.0[..2],
        transport_digest.0
    );
    if legacy_metadata
        .as_ref()
        .is_some_and(|metadata| metadata.get(&encoding_path) != Some(&transport_digest))
    {
        return Err(CacheError::InvalidManifest(
            "receipt does not bind the selected transport record".to_owned(),
        ));
    }
    let encoding = RemoteShardReader::new(context.remote, context.query.maximum_encoding_bytes)?
        .load_transport_encoding(
            repository,
            revision,
            &encoding_path,
            &transport_digest,
            context.cancellation,
        )?;
    if encoding.source.content_digest != transport_digest
        || encoding.value.canonical_payload_digest != entry.canonical_payload_digest
    {
        return Err(CacheError::InvalidManifest(
            "transport record does not match the canonical payload".to_owned(),
        ));
    }

    let dependency_key = (family.to_owned(), semantic_digest.clone());
    if !context.active_dependencies.insert(dependency_key.clone()) {
        return Err(CacheError::InvalidManifest(
            "artifact dependency graph contains a cycle".to_owned(),
        ));
    }
    let dependencies_result = (|| {
        let mut dependencies = Vec::new();
        for dependency in &manifest.value.canonical_payload.dependencies {
            context.dependency_count = context.dependency_count.saturating_add(1);
            if context.dependency_count > context.query.maximum_dependency_count {
                return Err(CacheError::ResourceLimit(format!(
                    "dependency count exceeds {}",
                    context.query.maximum_dependency_count
                )));
            }
            let resolved = resolve_identity(
                context,
                &dependency.artifact_family,
                &dependency.semantic_digest,
                Some(ExpectedArtifact {
                    manifest_digest: &dependency.manifest_digest,
                    payload_digest: &dependency.payload_digest,
                }),
                depth.saturating_add(1),
            )?
            .ok_or_else(|| {
                CacheError::NotFound(format!(
                    "dependency {} / {} / {}",
                    dependency.artifact_family,
                    dependency.semantic_digest,
                    dependency.manifest_digest
                ))
            })?;
            dependencies.push(resolved);
        }
        Ok(dependencies)
    })();
    context.active_dependencies.remove(&dependency_key);
    let dependencies = dependencies_result?;
    Ok(ResolvedRemoteArtifact {
        family: family.to_owned(),
        semantic_digest: semantic_digest.clone(),
        overlay: overlay.name.clone(),
        visibility: overlay.visibility,
        shard_id: shard_id.to_owned(),
        authorized_repository: authorized_repository.to_owned(),
        repository: repository.to_owned(),
        revision: revision.to_owned(),
        index: entry.clone(),
        manifest: manifest.value,
        encoding: encoding.value,
        receipt: publication_evidence,
        index_source: index.source.clone(),
        manifest_source: manifest.source,
        encoding_source: encoding.source,
        receipt_source,
        dependencies,
    })
}

fn active_revocation(
    context: &mut ResolutionContext<'_>,
    repository: &str,
    revision: &str,
    scope: RevocationScope,
    identity: &ContentDigest,
) -> Result<Option<RevocationRecord>, CacheError> {
    let prefix = identity.0[..2].to_owned();
    let key = (repository.to_owned(), revision.to_owned(), prefix.clone());
    if !context.revocations.contains_key(&key) {
        let path = format!("revocations/indexes/{prefix}.json");
        let document = RemoteShardReader::new(
            context.remote,
            context.query.maximum_revocation_partition_bytes,
        )?
        .read_json::<RevocationIndexPartition>(
            repository,
            revision,
            &path,
            context.cancellation,
        );
        let partition = match document {
            Ok(document) => {
                document.value.validate()?;
                if document.value.identity_prefix != prefix {
                    return Err(CacheError::InvalidManifest(
                        "revocation partition prefix does not match its path".to_owned(),
                    ));
                }
                Some(document.value)
            }
            Err(CacheError::NotFound(_)) => None,
            Err(error) => return Err(error),
        };
        context.revocations.insert(key.clone(), partition);
    }
    Ok(context.revocations[&key]
        .as_ref()
        .and_then(|partition| {
            partition.active(scope, identity, context.query.evaluation_unix_seconds)
        })
        .cloned())
}

fn entry_rejection(
    query: &RemoteSemanticQuery,
    entry: &ShardIndexEntry,
    expected: Option<&ExpectedArtifact<'_>>,
) -> Option<(ResolutionRejectionStage, String)> {
    if entry.disposition == ArtifactDisposition::Revoked
        || entry.disposition == ArtifactDisposition::Quarantined
        || (entry.disposition == ArtifactDisposition::Deprecated && !query.allow_deprecated)
    {
        return Some((
            ResolutionRejectionStage::Disposition,
            format!(
                "candidate disposition {:?} is not admissible",
                entry.disposition
            ),
        ));
    }
    if entry.achieved_assurance < query.minimum_assurance {
        return Some((
            ResolutionRejectionStage::Assurance,
            format!(
                "candidate assurance {:?} is below {:?}",
                entry.achieved_assurance, query.minimum_assurance
            ),
        ));
    }
    if query.current_toolkit_version < entry.minimum_reader_version {
        return Some((
            ResolutionRejectionStage::Compatibility,
            "candidate requires a newer toolkit reader".to_owned(),
        ));
    }
    if expected.is_some_and(|expected| {
        expected.manifest_digest != &entry.manifest_digest
            || expected.payload_digest != &entry.canonical_payload_digest
    }) {
        return Some((
            ResolutionRejectionStage::Dependency,
            "dependency candidate does not match its exact manifest and payload identities"
                .to_owned(),
        ));
    }
    None
}

pub fn repository_identity_digest(authorized_repository: &str) -> ContentDigest {
    ContentDigest::sha256(format!("github-repository-v1:{authorized_repository}").as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn reject(
    context: &mut ResolutionContext<'_>,
    overlay: &RemoteResolverOverlay,
    shard_id: Option<&str>,
    repository: Option<&str>,
    manifest_digest: Option<&ContentDigest>,
    stage: ResolutionRejectionStage,
    reason: String,
    revocation: Option<RevocationRecord>,
) {
    context.rejections.push(ResolutionRejection {
        overlay: overlay.name.clone(),
        shard_id: shard_id.map(str::to_owned),
        repository: repository.map(str::to_owned),
        manifest_digest: manifest_digest.cloned(),
        stage,
        reason,
        revocation,
    });
}

fn classify_candidate_error(error: &CacheError) -> ResolutionRejectionStage {
    match error {
        CacheError::NotFound(message) if message.contains("dependency") => {
            ResolutionRejectionStage::Dependency
        }
        CacheError::NotFound(_) => ResolutionRejectionStage::Manifest,
        CacheError::DigestMismatch { .. } => ResolutionRejectionStage::Manifest,
        CacheError::ResourceLimit(_) | CacheError::Cancelled(_) => {
            ResolutionRejectionStage::Dependency
        }
        _ => ResolutionRejectionStage::Manifest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical_json_bytes;
    use crate::{
        ArtifactFamilyRoute, CanonicalPayloadEnvelope, CompareAndSwapResult,
        GitHubRepositoryEndpoint, LogicalPayloadItem, RemoteCommitRequest, ShardIndexPartition,
        TopologyRegistry, TopologyShardRoute, TransportPart,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::Mutex;
    use xc_core::{AssuranceLevel, PublicationAuthorityMode};

    struct MemoryRemote {
        heads: BTreeMap<(String, String), String>,
        documents: BTreeMap<(String, String, String), Vec<u8>>,
        reads: Mutex<Vec<String>>,
    }

    impl RemoteGitStore for MemoryRemote {
        fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError> {
            self.heads
                .get(&(repository.to_owned(), branch.to_owned()))
                .cloned()
                .ok_or_else(|| CacheError::NotFound(format!("{repository}:{branch}")))
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            _revision: &str,
            _path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(None)
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
            self.reads.lock().unwrap().push(path.to_owned());
            let bytes = self
                .documents
                .get(&(repository.to_owned(), revision.to_owned(), path.to_owned()))
                .ok_or_else(|| CacheError::NotFound(format!("{revision}:{path}")))?;
            if bytes.len() as u64 > maximum_bytes {
                return Err(CacheError::ResourceLimit(format!(
                    "fixture path {path:?} exceeds bound"
                )));
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
            _request: &RemoteCommitRequest,
        ) -> Result<CompareAndSwapResult, CacheError> {
            panic!("semantic resolution must never mutate a repository")
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            _revision: &str,
            _part: &TransportPart,
        ) -> Result<(), CacheError> {
            panic!("semantic metadata resolution does not fetch payload parts")
        }
    }

    struct Fixture {
        remote: MemoryRemote,
        query: RemoteSemanticQuery,
        overlays: Vec<RemoteResolverOverlay>,
        semantic_digest: ContentDigest,
        manifest_digest: ContentDigest,
    }

    fn fixture(revoke_semantic: bool) -> Fixture {
        let topology_repository = "example-org/topology".to_owned();
        let topology_revision = "a".repeat(40);
        let shard_revision = "b".repeat(40);
        let policy_digest = ContentDigest::sha256(b"publication-policy");
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_matrix".to_owned(),
            mathematical_semantics_version: "ccm-v2".to_owned(),
            resolved_mathematical_parameters: json!({"n": 120}),
            normalization: Some("unit-trace".to_owned()),
            target: Some("algebraic_minimum".to_owned()),
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "mpfr".to_owned(),
            precision_bits: Some(256),
            scalar_representation: "bytes".to_owned(),
            dimensions: vec![15],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "fixture.bin".to_owned(),
                content_digest: ContentDigest::sha256(b"decoded-fixture"),
                size_bytes: 15,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let part_digest = ContentDigest::sha256(b"encoded-package");
        let encoding = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: payload_digest.clone(),
            encoder_profile: "deterministic-zip64-v1".to_owned(),
            package_size_bytes: 15,
            package_digest: part_digest.clone(),
            ordered_parts: vec![TransportPart {
                sequence: 0,
                repository_path: format!("objects/{}.part", part_digest.0),
                size_bytes: 15,
                content_digest: part_digest,
            }],
            reconstruction: "concatenate".to_owned(),
        };
        let transport_digest = encoding.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm".to_owned(),
            semantic_key: semantic_key.clone(),
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![transport_digest.clone()],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: Some(ToolkitVersion::parse("0.13.99").unwrap()),
            requested_assurance: AssuranceLevel::CrossChecked,
            claim_scope: "finite CCM fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let transaction_id = ContentDigest::sha256(b"transaction").0;
        let index = ShardIndexPartition::rebuild(
            "ccm",
            semantic_digest.0[..2].to_owned(),
            vec![ShardIndexEntry {
                semantic_digest: semantic_digest.clone(),
                canonical_payload_digest: payload_digest.clone(),
                manifest_digest: manifest_digest.clone(),
                achieved_assurance: ArtifactAssuranceState::CrossChecked,
                disposition: ArtifactDisposition::Active,
                producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
                transport_digests: vec![transport_digest.clone()],
                publication_transaction_id: transaction_id.clone(),
            }],
        )
        .unwrap();
        let index_bytes = canonical_json_bytes(&index).unwrap();
        let index_digest = ContentDigest::sha256(&index_bytes);
        let manifest_path = format!(
            "manifests/{}/{}.json",
            &semantic_digest.0[..2],
            manifest_digest.0
        );
        let encoding_path = format!(
            "encodings/{}/{}.json",
            &payload_digest.0[..2],
            transport_digest.0
        );
        let index_path = format!("indexes/ccm/{}.json", &semantic_digest.0[..2]);
        let receipt = PublicationReceipt {
            schema_version: 1,
            transaction_id: transaction_id.clone(),
            idempotency_key: ContentDigest(transaction_id.clone()),
            destination: PublicationDestination::Private,
            principal: "test-owner".to_owned(),
            authorized_repository: "example-org/restricted-cache".to_owned(),
            repository_permission_evidence_digest: ContentDigest::sha256(b"permission"),
            shard_id: "private-001".to_owned(),
            branch: "main".to_owned(),
            semantic_digest: semantic_digest.clone(),
            canonical_payload_digest: payload_digest.clone(),
            manifest_digest: manifest_digest.clone(),
            transport_digest: transport_digest.clone(),
            policy_digest: policy_digest.clone(),
            policy_id: "fixture-owner-policy".to_owned(),
            authority_mode: PublicationAuthorityMode::OwnerDirect,
            validation_evidence_digests: vec![ContentDigest::sha256(b"validator-evidence")],
            contributor_authorization_digest: None,
            reviewer_approvals: Vec::new(),
            payload_commit_ids: vec!["payload-commit".to_owned()],
            payload_batch_record_commit_ids: Vec::new(),
            payload_batch_record_digests: BTreeMap::new(),
            metadata_commit_id: "metadata-commit".to_owned(),
            metadata_file_digests: BTreeMap::from([
                (manifest_path.clone(), manifest_digest.clone()),
                (encoding_path.clone(), transport_digest.clone()),
                (
                    "attestations/provenance.json".to_owned(),
                    ContentDigest::sha256(b"provenance-evidence"),
                ),
            ]),
            discoverability_subject_digests: BTreeMap::from([(index_path.clone(), index_digest)]),
            remote_verification_results: vec![crate::RemoteCommitVerificationResult {
                phase: "immutable_metadata".to_owned(),
                sequence: 0,
                commit_id: "metadata-commit".to_owned(),
                verified: true,
                content_digests: vec![manifest_digest.clone(), transport_digest.clone()],
            }],
            verified_at_unix_seconds: 100,
        };
        receipt.validate().unwrap();
        let topology = TopologyRegistry {
            schema_version: 1,
            generation: 3,
            previous_registry_digest: Some(ContentDigest::sha256(b"generation-2")),
            policy_digest: policy_digest.clone(),
            trust_anchor_ids: vec!["release-key".to_owned()],
            family_routes: vec![ArtifactFamilyRoute {
                family: "ccm".to_owned(),
                visibility: CacheVisibility::Private,
                ordered_shards: vec![TopologyShardRoute {
                    shard_id: "private-001".to_owned(),
                    endpoint_id: "private-001".to_owned(),
                    sequence: 1,
                    status: TopologyShardStatus::Writable,
                    successor_shard_id: None,
                }],
            }],
        };
        let network = CacheNetworkRegistry {
            schema_version: 1,
            repositories: vec![GitHubRepositoryEndpoint {
                shard_id: "private-001".to_owned(),
                owner: "example-org".to_owned(),
                repository: "restricted-cache".to_owned(),
                branch: "main".to_owned(),
                visibility: CacheVisibility::Private,
                enabled_for_read: true,
                enabled_for_write: true,
                clone_via_ssh: false,
            }],
        };
        let shard_repository = network.repositories[0].preferred_clone_url();
        let mut documents = BTreeMap::from([
            (
                (
                    topology_repository.clone(),
                    topology_revision.clone(),
                    "registry/topology.json".to_owned(),
                ),
                canonical_json_bytes(&topology).unwrap(),
            ),
            (
                (shard_repository.clone(), shard_revision.clone(), index_path),
                index_bytes,
            ),
            (
                (
                    shard_repository.clone(),
                    shard_revision.clone(),
                    manifest_path,
                ),
                canonical_json_bytes(&manifest).unwrap(),
            ),
            (
                (
                    shard_repository.clone(),
                    shard_revision.clone(),
                    encoding_path,
                ),
                canonical_json_bytes(&encoding).unwrap(),
            ),
            (
                (
                    shard_repository.clone(),
                    shard_revision.clone(),
                    format!(
                        "transactions/{transaction_id}/{}/receipt.json",
                        match receipt.destination {
                            PublicationDestination::Private => "private",
                            PublicationDestination::Public => "public",
                        }
                    ),
                ),
                canonical_json_bytes(&receipt).unwrap(),
            ),
        ]);
        if revoke_semantic {
            let revocations = RevocationIndexPartition {
                schema_version: 1,
                identity_prefix: semantic_digest.0[..2].to_owned(),
                records: vec![RevocationRecord {
                    schema_version: 1,
                    scope: RevocationScope::Semantic,
                    identity_digest: semantic_digest.clone(),
                    reason: "superseded after validation incident".to_owned(),
                    effective_unix_seconds: 150,
                    replacement_digest: Some(ContentDigest::sha256(b"replacement")),
                    incident_reference: Some("INC-42".to_owned()),
                    authorizing_evidence_digest: ContentDigest::sha256(b"revocation-authority"),
                }],
            };
            documents.insert(
                (
                    shard_repository.clone(),
                    shard_revision.clone(),
                    format!("revocations/indexes/{}.json", &semantic_digest.0[..2]),
                ),
                canonical_json_bytes(&revocations).unwrap(),
            );
        }
        let query = RemoteSemanticQuery {
            family: "ccm".to_owned(),
            semantic_key,
            minimum_assurance: ArtifactAssuranceState::Computed,
            allowed_scalar_backends: BTreeSet::new(),
            minimum_precision_bits: None,
            required_configuration_digest: None,
            required_provenance_evidence_digests: BTreeSet::new(),
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            accepted_publication_policy_digests: [policy_digest].into_iter().collect(),
            allow_deprecated: false,
            evaluation_unix_seconds: 200,
            maximum_topology_bytes: 1024 * 1024,
            maximum_index_bytes: 1024 * 1024,
            maximum_manifest_bytes: 1024 * 1024,
            maximum_encoding_bytes: 1024 * 1024,
            maximum_receipt_bytes: 1024 * 1024,
            maximum_revocation_partition_bytes: 1024 * 1024,
            maximum_dependency_depth: 8,
            maximum_dependency_count: 100,
        };
        let overlays = vec![RemoteResolverOverlay {
            name: "private".to_owned(),
            visibility: CacheVisibility::Private,
            topology_source: RemoteTopologySource {
                repository: topology_repository.clone(),
                ..RemoteTopologySource::default()
            },
            topology_trust: TopologyTrustPolicy {
                minimum_generation: 3,
                pinned_registry_digest: Some(topology.digest().unwrap()),
                required_trust_anchor: Some("release-key".to_owned()),
            },
            fabric_trust: {
                let root = |role, repository: &str, owner: &str, revision: &str| {
                    let protection = crate::ProtectedBranchStatement {
                        schema_version: 1,
                        repository: repository.to_owned(),
                        branch: "main".to_owned(),
                        observed_revision: revision.to_owned(),
                        force_pushes_prohibited: true,
                        pushes_restricted: true,
                        observed_at_unix_seconds: 100,
                        valid_until_unix_seconds: 300,
                        trust_anchor_id: "release-key".to_owned(),
                    };
                    crate::TrustedRepositoryRoot {
                        role,
                        repository: repository.to_owned(),
                        owner: owner.to_owned(),
                        branch: "main".to_owned(),
                        revision_policy: crate::TrustedRevisionPolicy::Exact {
                            revision: revision.to_owned(),
                        },
                        branch_protection_digest: protection.digest().unwrap(),
                        branch_protection: protection,
                    }
                };
                crate::RemoteFabricTrustPolicy {
                    schema_version: 1,
                    approved_trust_anchor_ids: ["release-key".to_owned()].into_iter().collect(),
                    approved_policy_digests: [topology.policy_digest.clone()].into_iter().collect(),
                    repositories: vec![
                        root(
                            crate::TrustedRepositoryRole::Registry,
                            &topology_repository,
                            "example-org",
                            &topology_revision,
                        ),
                        root(
                            crate::TrustedRepositoryRole::Shard,
                            "example-org/restricted-cache",
                            "example-org",
                            &shard_revision,
                        ),
                    ],
                }
            },
            network,
        }];
        Fixture {
            remote: MemoryRemote {
                heads: BTreeMap::from([
                    ((topology_repository, "main".to_owned()), topology_revision),
                    ((shard_repository, "main".to_owned()), shard_revision),
                ]),
                documents,
                reads: Mutex::new(Vec::new()),
            },
            query,
            overlays,
            semantic_digest,
            manifest_digest,
        }
    }

    #[test]
    fn resolver_proves_index_manifest_encoding_and_receipt_without_payload_clone() {
        let fixture = fixture(false);
        let report = resolve_remote_semantic_artifact(
            &fixture.remote,
            &CancellationToken::new(),
            &fixture.query,
            &fixture.overlays,
        )
        .unwrap();
        let provenance =
            crate::record_remote_cache_access(crate::RemoteCacheAccessProvenanceRequest {
                operation: "ccm.resolve",
                family: &fixture.query.family,
                overlays: &fixture.overlays,
                resolution: &report,
                reuse_disposition: xc_core::CacheReuseDisposition::Reused,
                validation_mode: xc_core::CacheValidationMode::Fast,
                validation_outcome: xc_core::CacheValidationOutcome::Passed,
                validation_detail: None,
                materialization: None,
            })
            .unwrap();
        assert_eq!(
            provenance.selected_source.as_ref().unwrap().repository,
            report.selected.as_ref().unwrap().repository
        );
        let unified = crate::resolve_semantic_artifact(
            Some(&fixture.remote),
            &fixture.query,
            &[crate::SemanticArtifactSource::GitHub {
                overlay_class: crate::SemanticArtifactOverlayClass::TeamPrivate,
                overlay: &fixture.overlays[0],
            }],
            &xc_core::ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(
            unified.selected.as_ref().unwrap().semantic_digest,
            fixture.semantic_digest
        );
        let selected = report.selected.unwrap();
        assert_eq!(selected.semantic_digest, fixture.semantic_digest);
        assert_eq!(selected.manifest.digest().unwrap(), fixture.manifest_digest);
        assert!(matches!(
            &selected.receipt,
            crate::RemotePublicationEvidence::ArtifactReceipt(receipt)
                if receipt.transaction_id == selected.index.publication_transaction_id
        ));
        assert!(selected.dependencies.is_empty());
        assert!(fixture
            .remote
            .reads
            .lock()
            .unwrap()
            .iter()
            .all(|path| !path.starts_with("objects/")));
    }

    #[test]
    fn active_semantic_revocation_excludes_candidate_with_reason_and_replacement() {
        let fixture = fixture(true);
        let report = resolve_remote_semantic_artifact(
            &fixture.remote,
            &CancellationToken::new(),
            &fixture.query,
            &fixture.overlays,
        )
        .unwrap();
        assert!(report.selected.is_none());
        let provenance =
            crate::record_remote_cache_access(crate::RemoteCacheAccessProvenanceRequest {
                operation: "ccm.resolve",
                family: &fixture.query.family,
                overlays: &fixture.overlays,
                resolution: &report,
                reuse_disposition: xc_core::CacheReuseDisposition::Recomputed,
                validation_mode: xc_core::CacheValidationMode::Fast,
                validation_outcome: xc_core::CacheValidationOutcome::Failed,
                validation_detail: Some("no admissible candidate".to_owned()),
                materialization: None,
            })
            .unwrap();
        assert!(provenance.selected_source.is_none());
        assert!(!provenance.rejected_candidates.is_empty());
        let rejection = report
            .rejections
            .iter()
            .find(|rejection| rejection.revocation.is_some())
            .unwrap();
        assert_eq!(rejection.stage, ResolutionRejectionStage::Revocation);
        assert_eq!(
            rejection
                .revocation
                .as_ref()
                .unwrap()
                .incident_reference
                .as_deref(),
            Some("INC-42")
        );
    }

    #[test]
    fn consumption_policy_enforces_backend_precision_provenance_and_assurance() {
        let mut fixture = fixture(false);
        fixture.query.allowed_scalar_backends = ["mpfr".to_owned()].into_iter().collect();
        fixture.query.minimum_precision_bits = Some(256);
        fixture.query.required_configuration_digest = Some(ContentDigest::sha256(b"config"));
        fixture.query.required_provenance_evidence_digests =
            [ContentDigest::sha256(b"provenance-evidence")]
                .into_iter()
                .collect();
        let accepted = resolve_remote_semantic_artifact(
            &fixture.remote,
            &CancellationToken::new(),
            &fixture.query,
            &fixture.overlays,
        )
        .unwrap();
        assert!(accepted.selected.is_some());

        fixture.query.minimum_precision_bits = Some(257);
        let under_precision = resolve_remote_semantic_artifact(
            &fixture.remote,
            &CancellationToken::new(),
            &fixture.query,
            &fixture.overlays,
        )
        .unwrap();
        assert!(under_precision.selected.is_none());
        assert!(under_precision
            .rejections
            .iter()
            .any(|rejection| rejection.reason.contains("precision")));

        fixture.query.minimum_precision_bits = Some(256);
        fixture.query.minimum_assurance = ArtifactAssuranceState::Certified;
        let under_assured = resolve_remote_semantic_artifact(
            &fixture.remote,
            &CancellationToken::new(),
            &fixture.query,
            &fixture.overlays,
        )
        .unwrap();
        assert!(under_assured.selected.is_none());
        assert!(under_assured
            .rejections
            .iter()
            .any(|rejection| rejection.reason.contains("assurance")));
    }

    #[test]
    fn payload_dependency_identity_requires_a_family_route() {
        let dependency = crate::PayloadDependencyIdentity {
            artifact_family: String::new(),
            semantic_digest: ContentDigest::sha256(b"semantic"),
            manifest_digest: ContentDigest::sha256(b"manifest"),
            payload_digest: ContentDigest::sha256(b"payload"),
        };
        let envelope = crate::CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "ieee754".to_owned(),
            precision_bits: Some(64),
            scalar_representation: "f64-le".to_owned(),
            dimensions: vec![1],
            endianness: "little".to_owned(),
            special_value_encoding: "ieee754".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(b"value"),
                size_bytes: 5,
            }],
            dependencies: vec![dependency],
        };
        assert!(envelope.validate().is_err());
    }
}
