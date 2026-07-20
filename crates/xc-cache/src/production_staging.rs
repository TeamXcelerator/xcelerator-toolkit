//! Conversion of validated typed-production records into canonical transport
//! drafts. This stage is local-only: it selects no visibility or repository
//! and performs no remote operation.

use crate::{
    canonical_digest, package_canonical_payload_bytes_zip64, stream_split_encoded,
    ArtifactAssuranceState, CacheError, CanonicalArtifactManifest, CanonicalPayloadEnvelope,
    ContentDigest, LogicalPayloadItem, ProducedArtifactRecord, TransportEncodingRecord,
    TransportPolicy,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use xc_core::{AssuranceLevel, CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalProductionDraft {
    pub schema_version: u32,
    pub family: String,
    pub source_operation: String,
    pub source_logical_key: String,
    pub source_artifact_key: crate::ArtifactKey,
    pub source_content_digest: ContentDigest,
    pub source_manifest_digest: ContentDigest,
    pub manifest: CanonicalArtifactManifest,
    pub encoding: TransportEncodingRecord,
    pub staged_parts_root: PathBuf,
    pub achieved_assurance: ArtifactAssuranceState,
    #[serde(default)]
    pub required_assurance: Option<ArtifactAssuranceState>,
    #[serde(default)]
    pub assurance_evidence_digests: Vec<ContentDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedPublicationInventoryEntry {
    pub family: String,
    pub destination: crate::PublicationDestination,
    pub repository: String,
    pub branch: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub payload_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub package_size_bytes: u64,
    pub ordered_part_count: usize,
    pub achieved_assurance: ArtifactAssuranceState,
    #[serde(default)]
    pub required_assurance: Option<ArtifactAssuranceState>,
    pub assurance_evidence_digests: Vec<ContentDigest>,
    pub assurance_eligible: bool,
    pub ineligibility_reasons: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedPublicationInventory {
    pub schema_version: u32,
    pub profile: crate::ManagedRunProfile,
    pub requested_assurance: xc_core::AssuranceLevel,
    pub certification_failure_policy: crate::CertificationFailurePolicy,
    pub target: xc_core::PublicationTarget,
    pub entries: Vec<ManagedPublicationInventoryEntry>,
    pub total_target_package_bytes: u64,
    pub remote_mutation_enabled: bool,
    pub remote_mutation_requested: bool,
    pub ready_for_remote_execution: bool,
}

pub fn build_managed_publication_inventory(
    drafts: &[CanonicalProductionDraft],
    target: xc_core::PublicationTarget,
    owner: &str,
    profile: crate::ManagedRunProfile,
    requested_assurance: xc_core::AssuranceLevel,
    certification_failure_policy: crate::CertificationFailurePolicy,
    execute_remote_mutations: bool,
) -> Result<ManagedPublicationInventory, CacheError> {
    if owner.trim().is_empty() || owner.contains('/') {
        return Err(CacheError::InvalidManifest(
            "managed publication repository owner is invalid".to_owned(),
        ));
    }
    let destinations = match target {
        xc_core::PublicationTarget::None => Vec::new(),
        xc_core::PublicationTarget::Private => vec![crate::PublicationDestination::Private],
        xc_core::PublicationTarget::Public => vec![crate::PublicationDestination::Public],
        xc_core::PublicationTarget::Both => vec![
            crate::PublicationDestination::Private,
            crate::PublicationDestination::Public,
        ],
    };
    let mut entries = Vec::new();
    for draft in drafts {
        draft.manifest.validate()?;
        draft.encoding.validate()?;
        for destination in &destinations {
            let visibility = match destination {
                crate::PublicationDestination::Private => "private",
                crate::PublicationDestination::Public => "public",
            };
            entries.push(ManagedPublicationInventoryEntry {
                family: draft.family.clone(),
                destination: *destination,
                repository: format!(
                    "{owner}/xcelerator-cache-{visibility}-{}-0001",
                    draft.family
                ),
                branch: "main".to_owned(),
                semantic_digest: draft.manifest.semantic_digest.clone(),
                manifest_digest: draft.manifest.digest()?,
                payload_digest: draft.manifest.payload_digest.clone(),
                transport_digest: draft.encoding.digest()?,
                package_size_bytes: draft.encoding.package_size_bytes,
                ordered_part_count: draft.encoding.ordered_parts.len(),
                achieved_assurance: draft.achieved_assurance,
                required_assurance: draft.required_assurance,
                assurance_evidence_digests: draft.assurance_evidence_digests.clone(),
                assurance_eligible: draft
                    .required_assurance
                    .is_none_or(|required| draft.achieved_assurance >= required),
                ineligibility_reasons: if draft
                    .required_assurance
                    .is_none_or(|required| draft.achieved_assurance >= required)
                {
                    Vec::new()
                } else {
                    vec![format!(
                        "achieved assurance {:?} is below artifact requirement {:?}",
                        draft.achieved_assurance, draft.required_assurance
                    )]
                },
            });
        }
    }
    entries.sort_by(|left, right| {
        (&left.family, left.destination, &left.semantic_digest).cmp(&(
            &right.family,
            right.destination,
            &right.semantic_digest,
        ))
    });
    let total_target_package_bytes = entries.iter().try_fold(0u64, |total, entry| {
        total.checked_add(entry.package_size_bytes).ok_or_else(|| {
            CacheError::ResourceLimit("managed publication inventory size overflow".to_owned())
        })
    })?;
    let ready_for_remote_execution =
        !entries.is_empty() && entries.iter().all(|entry| entry.assurance_eligible);
    Ok(ManagedPublicationInventory {
        schema_version: 1,
        profile,
        requested_assurance,
        certification_failure_policy,
        target,
        entries,
        total_target_package_bytes,
        remote_mutation_enabled: execute_remote_mutations && ready_for_remote_execution,
        remote_mutation_requested: execute_remote_mutations,
        ready_for_remote_execution,
    })
}

/// Same-process production sink that converts every freshly validated typed
/// artifact directly into its deterministic canonical transport draft. The
/// retained drafts provide exact dependency identities to later artifacts in
/// the same computation; no queue-draining process is required.
pub struct CanonicalStagingProductionSink {
    staging_root: PathBuf,
    transport_policy: TransportPolicy,
    resources: ResourcePolicy,
    cancellation: CancellationToken,
    drafts: Mutex<Vec<CanonicalProductionDraft>>,
}

impl CanonicalStagingProductionSink {
    pub fn new(
        staging_root: impl Into<PathBuf>,
        transport_policy: TransportPolicy,
        resources: ResourcePolicy,
        cancellation: CancellationToken,
    ) -> Result<Self, CacheError> {
        let staging_root = staging_root.into();
        if staging_root.as_os_str().is_empty() {
            return Err(CacheError::InvalidManifest(
                "canonical production staging root is required".to_owned(),
            ));
        }
        transport_policy.validate()?;
        fs::create_dir_all(&staging_root)?;
        let mut drafts = Vec::new();
        let drafts_root = staging_root.join("drafts");
        if drafts_root.is_dir() {
            let mut pending = vec![drafts_root];
            while let Some(directory) = pending.pop() {
                for entry in fs::read_dir(directory)? {
                    let entry = entry?;
                    if entry.file_type()?.is_dir() {
                        pending.push(entry.path());
                    } else if entry.file_name() == "draft.json" {
                        let draft: CanonicalProductionDraft =
                            serde_json::from_slice(&fs::read(entry.path())?)?;
                        draft.manifest.validate()?;
                        draft.encoding.validate()?;
                        crate::ArtifactProductionAssessment {
                            achieved_assurance: draft.achieved_assurance,
                            evidence_digests: draft.assurance_evidence_digests.clone(),
                        }
                        .validate()?;
                        for part in &draft.encoding.ordered_parts {
                            let bytes =
                                fs::read(draft.staged_parts_root.join(&part.repository_path))?;
                            if bytes.len() as u64 != part.size_bytes
                                || ContentDigest::sha256(&bytes) != part.content_digest
                            {
                                return Err(CacheError::InvalidManifest(format!(
                                    "existing canonical staging part failed verification for {}",
                                    entry.path().display()
                                )));
                            }
                        }
                        drafts.push(draft);
                    }
                }
            }
            drafts.sort_by(|left, right| {
                (&left.family, &left.manifest.semantic_digest)
                    .cmp(&(&right.family, &right.manifest.semantic_digest))
            });
        }
        Ok(Self {
            staging_root,
            transport_policy,
            resources,
            cancellation,
            drafts: Mutex::new(drafts),
        })
    }

    pub fn staging_root(&self) -> &Path {
        &self.staging_root
    }

    pub fn drafts(&self) -> Result<Vec<CanonicalProductionDraft>, CacheError> {
        self.drafts
            .lock()
            .map(|drafts| drafts.clone())
            .map_err(|_| CacheError::Io("canonical staging sink lock is poisoned".to_owned()))
    }
}

impl crate::ArtifactProductionSink for CanonicalStagingProductionSink {
    fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
        let mut drafts = self
            .drafts
            .lock()
            .map_err(|_| CacheError::Io("canonical staging sink lock is poisoned".to_owned()))?;
        if drafts.iter().any(|draft| {
            draft.source_artifact_key == artifact.manifest.key
                && draft.source_content_digest == artifact.manifest.content_digest
        }) {
            return Ok(());
        }
        let draft = stage_produced_artifact_with_dependencies(
            &artifact,
            &drafts,
            &self.staging_root,
            &self.transport_policy,
            &self.resources,
            &self.cancellation,
        )?;
        drafts.push(draft);
        Ok(())
    }

    fn contains_artifact(
        &self,
        key: &crate::ArtifactKey,
        content_digest: &ContentDigest,
    ) -> Result<bool, CacheError> {
        Ok(self
            .drafts
            .lock()
            .map_err(|_| CacheError::Io("canonical staging sink lock is poisoned".to_owned()))?
            .iter()
            .any(|draft| {
                &draft.source_artifact_key == key && &draft.source_content_digest == content_digest
            }))
    }

    fn retained_assurance(
        &self,
        key: &crate::ArtifactKey,
        content_digest: &ContentDigest,
    ) -> Result<Option<crate::ArtifactProductionAssessment>, CacheError> {
        let assessment = {
            let drafts = self.drafts.lock().map_err(|_| {
                CacheError::Io("canonical staging sink lock is poisoned".to_owned())
            })?;
            let Some(draft) = drafts.iter().find(|draft| {
                &draft.source_artifact_key == key && &draft.source_content_digest == content_digest
            }) else {
                return Ok(None);
            };
            crate::ArtifactProductionAssessment {
                achieved_assurance: draft.achieved_assurance,
                evidence_digests: draft.assurance_evidence_digests.clone(),
            }
        };
        assessment.validate()?;
        if assessment.achieved_assurance == crate::ArtifactAssuranceState::Computed {
            return Ok(Some(assessment));
        }
        for digest in &assessment.evidence_digests {
            let mut found = false;
            let evidence_root = self.staging_root.join("evidence");
            if evidence_root.is_dir() {
                for kind in fs::read_dir(&evidence_root)? {
                    let path = kind?.path().join(&digest.0[0..2]).join(&digest.0);
                    if !path.is_file() {
                        continue;
                    }
                    let bytes = fs::read(&path)?;
                    let actual = ContentDigest::sha256(&bytes);
                    if &actual != digest {
                        return Err(CacheError::DigestMismatch {
                            expected: digest.to_string(),
                            actual: actual.to_string(),
                        });
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(None);
            }
        }
        Ok(Some(assessment))
    }

    fn record_assurance(
        &self,
        attestation: crate::ArtifactAssuranceAttestation,
    ) -> Result<(), CacheError> {
        attestation.validate()?;
        let mut drafts = self
            .drafts
            .lock()
            .map_err(|_| CacheError::Io("canonical staging sink lock is poisoned".to_owned()))?;
        let draft = drafts
            .iter_mut()
            .find(|draft| {
                draft.source_artifact_key == attestation.artifact_key
                    && draft.source_content_digest == attestation.content_digest
            })
            .ok_or_else(|| {
                CacheError::NotFound(format!(
                    "canonical staging draft is missing for assurance promotion {} / {}",
                    attestation.artifact_key.logical_key, attestation.content_digest
                ))
            })?;
        if attestation.achieved_assurance < draft.achieved_assurance {
            return Err(CacheError::InvalidTransition(
                "artifact assurance cannot be downgraded".to_owned(),
            ));
        }
        draft.achieved_assurance = attestation.achieved_assurance;
        draft.assurance_evidence_digests = attestation.evidence_digests.clone();
        let attestation_digest = canonical_digest(&attestation)?;
        let attestation_path = self
            .staging_root
            .join("assurance")
            .join(&attestation.artifact_key.parameters_digest.0)
            .join(&attestation.content_digest.0)
            .join(format!("{}.json", attestation_digest.0));
        if attestation_path.exists() {
            if serde_json::from_slice::<crate::ArtifactAssuranceAttestation>(&fs::read(
                &attestation_path,
            )?)? != attestation
            {
                return Err(CacheError::InvalidManifest(
                    "assurance attestation digest collision".to_owned(),
                ));
            }
        } else {
            let bytes = serde_json::to_vec_pretty(&attestation)?;
            let parent = attestation_path.parent().ok_or_else(|| {
                CacheError::InvalidManifest("assurance attestation path has no parent".to_owned())
            })?;
            fs::create_dir_all(parent)?;
            fs::write(&attestation_path, bytes)?;
        }
        let draft_path = self
            .staging_root
            .join("drafts")
            .join(&draft.manifest.semantic_digest.0)
            .join(&draft.source_content_digest.0)
            .join("draft.json");
        crate::atomic_replace(&draft_path, &serde_json::to_vec_pretty(draft)?)
    }

    fn record_assurance_requirement(
        &self,
        requirement: crate::ArtifactAssuranceRequirement,
    ) -> Result<(), CacheError> {
        requirement.validate()?;
        let mut drafts = self
            .drafts
            .lock()
            .map_err(|_| CacheError::Io("canonical staging sink lock is poisoned".to_owned()))?;
        let draft = drafts
            .iter_mut()
            .find(|draft| {
                draft.source_artifact_key == requirement.artifact_key
                    && draft.source_content_digest == requirement.content_digest
            })
            .ok_or_else(|| {
                CacheError::NotFound(format!(
                    "canonical staging draft is missing for assurance requirement {} / {}",
                    requirement.artifact_key.logical_key, requirement.content_digest
                ))
            })?;
        if draft
            .required_assurance
            .is_some_and(|current| current != requirement.required_assurance)
        {
            return Err(CacheError::InvalidTransition(
                "artifact assurance requirement cannot change within a run".to_owned(),
            ));
        }
        draft.required_assurance = Some(requirement.required_assurance);
        let path = self
            .staging_root
            .join("drafts")
            .join(&draft.manifest.semantic_digest.0)
            .join(&draft.source_content_digest.0)
            .join("draft.json");
        crate::atomic_replace(&path, &serde_json::to_vec_pretty(draft)?)
    }

    fn record_evidence(&self, kind: &str, bytes: &[u8]) -> Result<ContentDigest, CacheError> {
        if kind.trim().is_empty()
            || !kind
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(CacheError::InvalidManifest(
                "artifact evidence kind is invalid".to_owned(),
            ));
        }
        let digest = ContentDigest::sha256(bytes);
        let path = self
            .staging_root
            .join("evidence")
            .join(kind)
            .join(&digest.0[0..2])
            .join(&digest.0);
        if path.exists() {
            if fs::read(&path)? != bytes {
                return Err(CacheError::DigestMismatch {
                    expected: digest.to_string(),
                    actual: ContentDigest::sha256(&fs::read(&path)?).to_string(),
                });
            }
        } else {
            let parent = path.parent().ok_or_else(|| {
                CacheError::InvalidManifest("artifact evidence path has no parent".to_owned())
            })?;
            fs::create_dir_all(parent)?;
            fs::write(path, bytes)?;
        }
        Ok(digest)
    }
}

pub fn family_for_artifact_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "gauss_legendre_rule"
        | "quadrature_rule"
        | "quadrature_reference_table"
        | "quadrature_validation" => Some("quadrature"),
        "ccm_prime_enumeration"
        | "ccm_archimedean_integrals"
        | "ccm_archimedean_component"
        | "ccm_prime_component"
        | "ccm_pole_component" => Some("ccm-components"),
        "ccm_tau_matrix"
        | "ccm_even_sector_matrix"
        | "ccm_odd_sector_matrix"
        | "ccm_sector_tridiagonal"
        | "ccm_sector_transform"
        | "ccm_reduced_operator"
        | "ccm_factorization" => Some("ccm-matrices"),
        "ccm_weil_eigenpair"
        | "ccm_sector_eigenvalues"
        | "ccm_sector_spectrum"
        | "ccm_weil_plunge_state"
        | "ccm_weil_sonin_state"
        | "ccm_source_eigenbasis" => Some("weil-states"),
        "prolate_eigenvalue_spectrum"
        | "ccm_prolate_spectrum"
        | "ccm_prolate_basis"
        | "ccm_prolate_candidate"
        | "ccm_band_concentration" => Some("prolate"),
        "ccm_secular_source"
        | "ccm_root_count_window"
        | "ccm_root_discovery_window"
        | "ccm_root_refinement"
        | "ccm_spectral_window" => Some("ccm-roots"),
        "ccm_convergence_diagnostics"
        | "ccm_sector_gap"
        | "ccm_post_discovery_comparison"
        | "ccm_cross_check_record"
        | "ccm_validation_record"
        | "ccm_certificate_bundle" => Some("ccm-evidence"),
        _ => None,
    }
}

pub fn stage_produced_artifact(
    record: &ProducedArtifactRecord,
    staging_root: &Path,
    transport_policy: &TransportPolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<CanonicalProductionDraft, CacheError> {
    stage_produced_artifact_with_dependencies(
        record,
        &[],
        staging_root,
        transport_policy,
        resources,
        cancellation,
    )
}

pub fn stage_produced_artifact_with_dependencies(
    record: &ProducedArtifactRecord,
    dependency_drafts: &[CanonicalProductionDraft],
    staging_root: &Path,
    transport_policy: &TransportPolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<CanonicalProductionDraft, CacheError> {
    record.semantic_key.validate()?;
    record.manifest.validate()?;
    crate::ArtifactProductionAssessment {
        achieved_assurance: record.achieved_assurance,
        evidence_digests: record.assurance_evidence_digests.clone(),
    }
    .validate()?;
    let family = family_for_artifact_kind(&record.semantic_key.artifact_kind).ok_or_else(|| {
        CacheError::InvalidManifest(format!(
            "artifact kind {:?} has no publication family mapping",
            record.semantic_key.artifact_kind
        ))
    })?;
    if record.operation.trim().is_empty()
        || record.logical_key.trim().is_empty()
        || record.manifest.key.parameters_digest != record.semantic_key.digest()?
        || record.manifest.content_digest != ContentDigest::sha256(&record.payload)
        || record.manifest.size_bytes != record.payload.len() as u64
    {
        return Err(CacheError::InvalidManifest(
            "produced artifact cannot be canonically staged because its identities disagree"
                .to_owned(),
        ));
    }
    transport_policy.validate()?;
    let semantic_digest = record.semantic_key.digest()?;
    let draft_root = staging_root
        .join("drafts")
        .join(&semantic_digest.0)
        .join(&record.manifest.content_digest.0);
    if draft_root.exists() {
        let draft_path = draft_root.join("draft.json");
        let draft: CanonicalProductionDraft =
            serde_json::from_slice(&fs::read(&draft_path).map_err(|_| {
                CacheError::InvalidManifest(format!(
                    "existing canonical production draft is incomplete: {}",
                    draft_root.display()
                ))
            })?)?;
        draft.manifest.validate()?;
        draft.encoding.validate()?;
        crate::ArtifactProductionAssessment {
            achieved_assurance: draft.achieved_assurance,
            evidence_digests: draft.assurance_evidence_digests.clone(),
        }
        .validate()?;
        if draft.source_artifact_key != record.manifest.key
            || draft.source_content_digest != record.manifest.content_digest
            || draft.source_manifest_digest != canonical_digest(&record.manifest)?
            || draft.family != family
            || draft.manifest.semantic_digest != semantic_digest
        {
            return Err(CacheError::InvalidManifest(format!(
                "existing canonical production draft disagrees with produced artifact: {}",
                draft_root.display()
            )));
        }
        for part in &draft.encoding.ordered_parts {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let path = draft.staged_parts_root.join(&part.repository_path);
            let bytes = fs::read(&path)?;
            if bytes.len() as u64 != part.size_bytes
                || ContentDigest::sha256(&bytes) != part.content_digest
            {
                return Err(CacheError::InvalidManifest(format!(
                    "existing canonical production part failed verification: {}",
                    path.display()
                )));
            }
        }
        return Ok(draft);
    }
    fs::create_dir_all(&draft_root)?;
    let precision_bits = record
        .semantic_key
        .resolved_mathematical_parameters
        .get("precision_bits")
        .and_then(serde_json::Value::as_u64);
    let scalar_backend = record
        .semantic_key
        .resolved_mathematical_parameters
        .get("scalar_backend")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("canonical_json")
        .to_owned();
    let mut dependencies = Vec::new();
    for dependency in &record.manifest.dependencies {
        let draft = dependency_drafts
            .iter()
            .find(|draft| {
                draft.source_artifact_key == dependency.key
                    && draft.source_content_digest == dependency.content_digest
            })
            .ok_or_else(|| {
                CacheError::NotFound(format!(
                    "canonical dependency draft is missing for {} / {} / {}",
                    dependency.key.kind, dependency.key.logical_key, dependency.content_digest
                ))
            })?;
        dependencies.push(crate::PayloadDependencyIdentity {
            artifact_family: draft.family.clone(),
            semantic_digest: draft.manifest.semantic_digest.clone(),
            manifest_digest: draft.manifest.digest()?,
            payload_digest: draft.manifest.payload_digest.clone(),
        });
    }
    dependencies.sort_by(|left, right| {
        (
            &left.artifact_family,
            &left.semantic_digest,
            &left.manifest_digest,
            &left.payload_digest,
        )
            .cmp(&(
                &right.artifact_family,
                &right.semantic_digest,
                &right.manifest_digest,
                &right.payload_digest,
            ))
    });
    let canonical_payload = CanonicalPayloadEnvelope {
        schema_version: 1,
        scalar_backend,
        precision_bits,
        scalar_representation: "canonical-json-utf8-v1".to_owned(),
        dimensions: Vec::new(),
        endianness: "not-applicable".to_owned(),
        special_value_encoding: "decimal-string-or-json-number-v1".to_owned(),
        ordered_items: vec![LogicalPayloadItem {
            normalized_path: "payload.json".to_owned(),
            content_digest: ContentDigest::sha256(&record.payload),
            size_bytes: record.payload.len() as u64,
        }],
        dependencies,
    };
    let payload_digest = canonical_payload.digest()?;
    let parts_root = draft_root.join("parts");
    let archive_path = draft_root.join("payload.zip");
    let package = package_canonical_payload_bytes_zip64(
        &canonical_payload,
        "payload.json",
        &record.payload,
        &archive_path,
        resources,
        cancellation,
    )?;
    let mut archive = BufReader::new(File::open(&archive_path)?);
    let encoding = stream_split_encoded(
        &mut archive,
        package.canonical_payload_digest,
        package.encoder_profile,
        transport_policy,
        resources,
        cancellation,
        |part, bytes| {
            let path = parts_root.join(&part.repository_path);
            if path.exists() {
                return Err(CacheError::InvalidManifest(format!(
                    "staged part already exists: {}",
                    path.display()
                )));
            }
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(path, bytes)?;
            Ok(())
        },
    )?;
    drop(archive);
    fs::remove_file(&archive_path)?;
    let transport_digest = encoding.digest()?;
    let manifest = CanonicalArtifactManifest {
        schema_version: 1,
        artifact_family: family.to_owned(),
        semantic_key: record.semantic_key.clone(),
        semantic_digest,
        canonical_payload,
        payload_digest,
        transport_digests: vec![transport_digest],
        resolved_mathematical_configuration_digest: canonical_digest(
            &record.semantic_key.resolved_mathematical_parameters,
        )?,
        producer_toolkit_version: record.manifest.producer_toolkit_version.clone(),
        minimum_reader_version: record.manifest.minimum_reader_version.clone(),
        maximum_reader_version: record.manifest.maximum_reader_version.clone(),
        requested_assurance: AssuranceLevel::Computed,
        claim_scope: "validated typed numerical cache artifact".to_owned(),
        assumptions: Vec::new(),
    };
    manifest.validate()?;
    let draft = CanonicalProductionDraft {
        schema_version: 1,
        family: family.to_owned(),
        source_operation: record.operation.clone(),
        source_logical_key: record.logical_key.clone(),
        source_artifact_key: record.manifest.key.clone(),
        source_content_digest: record.manifest.content_digest.clone(),
        source_manifest_digest: canonical_digest(&record.manifest)?,
        manifest,
        encoding,
        staged_parts_root: parts_root,
        achieved_assurance: record.achieved_assurance,
        required_assurance: None,
        assurance_evidence_digests: record.assurance_evidence_digests.clone(),
    };
    fs::write(
        draft_root.join("draft.json"),
        serde_json::to_vec_pretty(&draft)?,
    )?;
    Ok(draft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtifactKey, ArtifactManifest, CacheQuality, CacheVisibility, ToolkitVersion};
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn granular_ccm_sector_artifacts_route_to_existing_shards() {
        assert_eq!(
            family_for_artifact_kind("ccm_sector_tridiagonal"),
            Some("ccm-matrices")
        );
        assert_eq!(
            family_for_artifact_kind("ccm_sector_transform"),
            Some("ccm-matrices")
        );
        assert_eq!(
            family_for_artifact_kind("ccm_sector_eigenvalues"),
            Some("weil-states")
        );
        assert_eq!(
            family_for_artifact_kind("ccm_sector_spectrum"),
            Some("weil-states")
        );
    }

    fn record() -> ProducedArtifactRecord {
        let semantic_key = crate::SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_tau_matrix".to_owned(),
            mathematical_semantics_version: "ccm-weil-form-v0.13.0-v1".to_owned(),
            resolved_mathematical_parameters: json!({
                "lambda_squared": 13,
                "n_modes": 2,
                "precision_bits": 128,
                "scalar_backend": "rug_mpfr"
            }),
            normalization: Some("symmetric_row_major".to_owned()),
            target: Some("localized_weil_form".to_owned()),
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let payload = br#"{"entries":["1.0","0.0","0.0","1.0"]}"#.to_vec();
        let parameters_digest = semantic_key.digest().unwrap();
        ProducedArtifactRecord {
            operation: "ccm.tau.resolve_or_compute".to_owned(),
            semantic_key,
            logical_key: "ccm/tau/13/2/128".to_owned(),
            manifest: ArtifactManifest {
                schema_version: 1,
                key: ArtifactKey {
                    kind: "ccm_tau_matrix".to_owned(),
                    logical_key: "ccm/tau/13/2/128".to_owned(),
                    parameters_digest,
                },
                content_digest: ContentDigest::sha256(&payload),
                size_bytes: payload.len() as u64,
                objects: vec![crate::CacheObjectRef {
                    content_digest: ContentDigest::sha256(&payload),
                    size_bytes: payload.len() as u64,
                }],
                created_unix_seconds: 1,
                producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies: Vec::new(),
                tags: BTreeMap::new(),
                provenance_digest: None,
            },
            achieved_assurance: ArtifactAssuranceState::Computed,
            assurance_evidence_digests: Vec::new(),
            payload,
        }
    }

    #[test]
    fn typed_record_stages_as_deterministic_bounded_canonical_draft() {
        let first_root =
            std::env::temp_dir().join(format!("xc-production-stage-first-{}", std::process::id()));
        let second_root =
            std::env::temp_dir().join(format!("xc-production-stage-second-{}", std::process::id()));
        let _ = fs::remove_dir_all(&first_root);
        let _ = fs::remove_dir_all(&second_root);
        let record = record();
        let first = stage_produced_artifact(
            &record,
            &first_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let second = stage_produced_artifact(
            &record,
            &second_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(first.family, "ccm-matrices");
        assert_eq!(first.manifest, second.manifest);
        assert_eq!(first.encoding, second.encoding);
        assert!(first
            .encoding
            .ordered_parts
            .iter()
            .all(|part| part.size_bytes < crate::GITHUB_HARD_FILE_BOUNDARY_BYTES));
        for part in &first.encoding.ordered_parts {
            let bytes = fs::read(first.staged_parts_root.join(&part.repository_path)).unwrap();
            assert_eq!(ContentDigest::sha256(&bytes), part.content_digest);
        }
        let _ = fs::remove_dir_all(first_root);
        let _ = fs::remove_dir_all(second_root);
    }

    #[test]
    fn canonical_staging_binds_exact_dependency_draft() {
        let root =
            std::env::temp_dir().join(format!("xc-production-dependencies-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let mut dependency = record();
        dependency.semantic_key.artifact_kind = "gauss_legendre_rule".to_owned();
        dependency.manifest.key.kind = "gauss_legendre_rule".to_owned();
        dependency.manifest.key.logical_key = "quadrature/gauss-legendre/64/128".to_owned();
        dependency.logical_key = dependency.manifest.key.logical_key.clone();
        dependency.manifest.key.parameters_digest = dependency.semantic_key.digest().unwrap();
        let dependency_draft = stage_produced_artifact(
            &dependency,
            &root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();

        let mut dependent = record();
        dependent.manifest.dependencies.push(crate::DependencyRef {
            key: dependency.manifest.key.clone(),
            content_digest: dependency.manifest.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        let dependent_draft = stage_produced_artifact_with_dependencies(
            &dependent,
            std::slice::from_ref(&dependency_draft),
            &root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let identities = &dependent_draft.manifest.canonical_payload.dependencies;
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].artifact_family, "quadrature");
        assert_eq!(
            identities[0].semantic_digest,
            dependency_draft.manifest.semantic_digest
        );
        assert_eq!(
            identities[0].manifest_digest,
            dependency_draft.manifest.digest().unwrap()
        );
        assert_eq!(
            identities[0].payload_digest,
            dependency_draft.manifest.payload_digest
        );

        let missing_root = root.with_extension("missing");
        let _ = fs::remove_dir_all(&missing_root);
        let error = stage_produced_artifact(
            &dependent,
            &missing_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::NotFound(_)));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(missing_root);
    }

    #[test]
    fn integrated_sink_stages_dependency_chain_in_process() {
        let root = std::env::temp_dir().join(format!(
            "xc-integrated-production-sink-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let sink = CanonicalStagingProductionSink::new(
            &root,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();

        let mut dependency = record();
        dependency.semantic_key.artifact_kind = "gauss_legendre_rule".to_owned();
        dependency.manifest.key.kind = "gauss_legendre_rule".to_owned();
        dependency.manifest.key.logical_key = "quadrature/gauss-legendre/64/128".to_owned();
        dependency.logical_key = dependency.manifest.key.logical_key.clone();
        dependency.manifest.key.parameters_digest = dependency.semantic_key.digest().unwrap();
        crate::ArtifactProductionSink::record(&sink, dependency.clone()).unwrap();
        crate::ArtifactProductionSink::record(&sink, dependency.clone()).unwrap();
        assert_eq!(sink.drafts().unwrap().len(), 1);

        let mut dependent = record();
        dependent.manifest.dependencies.push(crate::DependencyRef {
            key: dependency.manifest.key.clone(),
            content_digest: dependency.manifest.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        let dependent_key = dependent.manifest.key.clone();
        let dependent_content = dependent.manifest.content_digest.clone();
        crate::ArtifactProductionSink::record(&sink, dependent).unwrap();
        crate::ArtifactProductionSink::record_assurance_requirement(
            &sink,
            crate::ArtifactAssuranceRequirement {
                schema_version: 1,
                artifact_key: dependent_key.clone(),
                content_digest: dependent_content.clone(),
                required_assurance: ArtifactAssuranceState::Certified,
            },
        )
        .unwrap();

        let drafts = sink.drafts().unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].family, "quadrature");
        assert_eq!(drafts[1].family, "ccm-matrices");
        assert_eq!(drafts[1].manifest.canonical_payload.dependencies.len(), 1);
        assert_eq!(
            drafts[1].manifest.canonical_payload.dependencies[0].manifest_digest,
            drafts[0].manifest.digest().unwrap()
        );
        let inventory = build_managed_publication_inventory(
            &drafts,
            xc_core::PublicationTarget::Both,
            "example-org",
            crate::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Certified,
            crate::CertificationFailurePolicy::RetainComputedFailRun,
            false,
        )
        .unwrap();
        assert_eq!(inventory.entries.len(), 4);
        assert!(!inventory.remote_mutation_enabled);
        assert!(!inventory.ready_for_remote_execution);
        assert_eq!(
            inventory
                .entries
                .iter()
                .filter(|entry| !entry.assurance_eligible)
                .count(),
            2
        );
        assert!(inventory.entries.iter().any(|entry| {
            entry.family == "quadrature"
                && entry.destination == crate::PublicationDestination::Private
                && entry.repository.starts_with("example-org/")
        }));
        assert!(inventory.entries.iter().any(|entry| {
            entry.family == "ccm-matrices"
                && entry.destination == crate::PublicationDestination::Public
                && entry.repository.starts_with("example-org/")
        }));

        let certificate_digest = crate::ArtifactProductionSink::record_evidence(
            &sink,
            "portable-interval-certificate",
            b"portable interval certificate",
        )
        .unwrap();
        crate::ArtifactProductionSink::record_assurance(
            &sink,
            crate::ArtifactAssuranceAttestation {
                schema_version: 1,
                artifact_key: dependent_key,
                content_digest: dependent_content,
                achieved_assurance: ArtifactAssuranceState::Certified,
                evidence_digests: vec![certificate_digest.clone()],
            },
        )
        .unwrap();
        let promoted = sink
            .drafts()
            .unwrap()
            .into_iter()
            .find(|draft| draft.family == "ccm-matrices")
            .unwrap();
        assert_eq!(
            promoted.achieved_assurance,
            ArtifactAssuranceState::Certified
        );
        assert_eq!(
            promoted.assurance_evidence_digests,
            vec![certificate_digest.clone()]
        );
        assert_eq!(
            crate::ArtifactProductionSink::retained_assurance(
                &sink,
                &promoted.source_artifact_key,
                &promoted.source_content_digest,
            )
            .unwrap(),
            Some(crate::ArtifactProductionAssessment {
                achieved_assurance: ArtifactAssuranceState::Certified,
                evidence_digests: vec![certificate_digest],
            })
        );
        let promoted_inventory = build_managed_publication_inventory(
            &sink.drafts().unwrap(),
            xc_core::PublicationTarget::Both,
            "example-org",
            crate::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Certified,
            crate::CertificationFailurePolicy::RetainComputedFailRun,
            false,
        )
        .unwrap();
        assert!(promoted_inventory.ready_for_remote_execution);
        let private_shard = format!("{}-ccm-matrices-0001", "private");
        let public_shard = format!("{}-ccm-matrices-0001", "public");
        let ledger = |shard: &str, head: &str| crate::CapacityLedger {
            schema_version: 1,
            shard_id: shard.to_owned(),
            hard_capacity_bytes: crate::GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 0,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 0,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: head.to_owned(),
            reconciliation_digest: ContentDigest::sha256(shard.as_bytes()),
        };
        let private_repository = format!("example-org/xcelerator-cache-{private_shard}");
        let public_repository = format!("example-org/xcelerator-cache-{public_shard}");
        let sessions = BTreeMap::from([
            (
                crate::PublicationDestination::Private,
                crate::AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    &private_repository,
                    crate::RepositoryPermission::Write,
                ),
            ),
            (
                crate::PublicationDestination::Public,
                crate::AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    &public_repository,
                    crate::RepositoryPermission::Write,
                ),
            ),
        ]);
        let private_head = "a".repeat(40);
        let public_head = "b".repeat(40);
        let prepared = crate::prepare_managed_artifact_publication(
            &promoted,
            &crate::ManagedPublicationPlanningContext {
                owner: "example-org".to_owned(),
                principal: "test-owner".to_owned(),
                target: xc_core::PublicationTarget::Both,
                target_heads: BTreeMap::from([
                    (crate::PublicationDestination::Private, private_head.clone()),
                    (crate::PublicationDestination::Public, public_head.clone()),
                ]),
                capacity_ledgers: BTreeMap::from([
                    (
                        crate::PublicationDestination::Private,
                        ledger(&private_shard, &private_head),
                    ),
                    (
                        crate::PublicationDestination::Public,
                        ledger(&public_shard, &public_head),
                    ),
                ]),
                event_unix_seconds: 123,
            },
            &sessions,
        )
        .unwrap();
        assert!(prepared.coordinated.authorized());
        assert_ne!(
            prepared.bundles[&crate::PublicationDestination::Private]
                .manifest
                .digest()
                .unwrap(),
            prepared.bundles[&crate::PublicationDestination::Public]
                .manifest
                .digest()
                .unwrap()
        );
        assert_eq!(promoted.manifest, drafts[1].manifest);

        let resumed = CanonicalStagingProductionSink::new(
            &root,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        crate::ArtifactProductionSink::record(&resumed, dependency).unwrap();
        assert!(resumed
            .drafts()
            .unwrap()
            .iter()
            .any(|draft| draft == &drafts[0]));
        let _ = fs::remove_dir_all(root);
    }
}
