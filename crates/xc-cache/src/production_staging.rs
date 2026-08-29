//! Conversion of validated typed-production records into canonical transport
//! drafts. This stage is local-only: it selects no visibility or repository
//! and performs no remote operation.

use crate::{
    canonical_digest, package_canonical_payload_bytes_zip64, stream_split_encoded,
    ArtifactAssuranceState, CacheError, CanonicalArtifactManifest, CanonicalPayloadEnvelope,
    ContentDigest, LogicalPayloadItem, ProducedArtifactRecord, TransportEncodingRecord,
    TransportPolicy, REMOTE_CANONICAL_MANIFEST_TAG,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
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
    // Private-only kinds are routed rather than rejected: they keep their
    // private destination and are withheld from the public one, so a mixed
    // draft set publishes gracefully under `Both`. The only hard failure is
    // an explicit public-only request in which nothing at all is
    // public-eligible -- publishing nothing under an explicit request would
    // be a silent no-op.
    if target == xc_core::PublicationTarget::Public
        && !drafts.is_empty()
        && drafts.iter().all(|draft| {
            artifact_kind_is_private_only(draft.manifest.semantic_key.artifact_kind.as_str())
        })
    {
        let restricted = drafts
            .iter()
            .map(|draft| draft.manifest.semantic_key.artifact_kind.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        return Err(CacheError::PermissionDenied(format!(
            "runtime-target-derived artifacts are private-only and cannot be published to a public destination: {}",
            restricted.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    let mut entries = Vec::new();
    for draft in drafts {
        draft.manifest.validate()?;
        draft.encoding.validate()?;
        for destination in &destinations {
            if !artifact_kind_admitted_to_destination(
                draft.manifest.semantic_key.artifact_kind.as_str(),
                *destination,
            ) {
                continue;
            }
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
    /// Exact artifacts touched by this process. Existing drafts loaded from a
    /// prior process are deliberately not counted: remote execution must not
    /// succeed merely because stale staging happens to be nonempty.
    observed_artifacts: Mutex<BTreeSet<String>>,
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
            observed_artifacts: Mutex::new(BTreeSet::new()),
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

    pub fn observed_artifact_count(&self) -> Result<usize, CacheError> {
        self.observed_artifacts
            .lock()
            .map(|observed| observed.len())
            .map_err(|_| {
                CacheError::Io("canonical staging observation lock is poisoned".to_owned())
            })
    }

    fn mark_observed(&self, artifact: &ProducedArtifactRecord) -> Result<(), CacheError> {
        self.observed_artifacts
            .lock()
            .map_err(|_| {
                CacheError::Io("canonical staging observation lock is poisoned".to_owned())
            })?
            .insert(format!(
                "{}\n{}\n{}\n{}",
                artifact.manifest.key.kind,
                artifact.manifest.key.logical_key,
                artifact.manifest.key.parameters_digest,
                artifact.manifest.content_digest
            ));
        Ok(())
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
            self.mark_observed(&artifact)?;
            return Ok(());
        }
        // Deduplicate by exact published identity as well as by source key.
        // The key-based and identity-based closure walks can meet on a
        // diamond: the same artifact staged first through an identity
        // resolution carries a synthetic logical key, and a later key-based
        // reference must be recognized as the same draft rather than rejected
        // as a disagreeing one. The comparison uses the full identity tuple
        // -- family, semantic digest, canonical manifest digest, canonical
        // payload digest -- which is computable up front only for records
        // that carry a retained canonical manifest; those are exactly the
        // records both walks can produce for the same artifact. Anything
        // weaker (for example semantic digest plus raw payload digest) would
        // silently discard a second artifact whose canonical dependencies
        // differ, leaving the staged closure incomplete.
        if let Some(encoded) = artifact.manifest.tags.get(REMOTE_CANONICAL_MANIFEST_TAG) {
            let canonical: CanonicalArtifactManifest = serde_json::from_str(encoded)?;
            let family = family_for_artifact_kind(&artifact.semantic_key.artifact_kind)
                .ok_or_else(|| {
                    CacheError::InvalidManifest(format!(
                        "artifact kind {:?} has no publication family mapping",
                        artifact.semantic_key.artifact_kind
                    ))
                })?;
            crate::validate_retained_canonical_binding(
                &canonical,
                &artifact.semantic_key,
                family,
                &artifact.manifest.content_digest,
                artifact.manifest.size_bytes,
                artifact.manifest.provenance_digest.as_ref(),
            )?;
            let manifest_digest = canonical.digest()?;
            if let Some(existing) = drafts.iter_mut().find(|draft| {
                draft.family == canonical.artifact_family
                    && draft.manifest.semantic_digest == canonical.semantic_digest
                    && draft.manifest.payload_digest == canonical.payload_digest
                    && draft
                        .manifest
                        .digest()
                        .map(|digest| digest == manifest_digest)
                        .unwrap_or(false)
            }) {
                // An identity-addressed GitHub closure lookup necessarily
                // creates a local-only `closure/...` adapter key because a
                // canonical shard manifest carries no logical key. If the
                // key-based walk later reaches the same published artifact,
                // retain its real adapter provenance on the existing draft.
                // A freshly produced child names dependencies by that real
                // key, so merely deduplicating here would leave the child
                // unable to match its already-staged dependency.
                //
                // Promote only the explicit identity-closure provenance. Two
                // genuinely different non-synthetic keys remain deduplicated
                // by their full published identity without silently choosing
                // one as canonical.
                if existing.source_operation == "cache.dependency.closure"
                    && artifact.operation != "cache.dependency.closure"
                    && existing.source_artifact_key != artifact.manifest.key
                {
                    let mut promoted = existing.clone();
                    promoted.source_operation = artifact.operation.clone();
                    promoted.source_logical_key = artifact.logical_key.clone();
                    promoted.source_artifact_key = artifact.manifest.key.clone();
                    promoted.source_content_digest = artifact.manifest.content_digest.clone();
                    promoted.source_manifest_digest = canonical_digest(&artifact.manifest)?;
                    let draft_path = self
                        .staging_root
                        .join("drafts")
                        .join(&promoted.manifest.semantic_digest.0)
                        .join(&promoted.manifest.payload_digest.0)
                        .join("draft.json");
                    crate::atomic_replace(&draft_path, &serde_json::to_vec_pretty(&promoted)?)?;
                    *existing = promoted;
                }
                self.mark_observed(&artifact)?;
                return Ok(());
            }
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
        self.mark_observed(&artifact)?;
        Ok(())
    }

    fn contains_dependency_identity(
        &self,
        identity: &crate::PayloadDependencyIdentity,
    ) -> Result<bool, CacheError> {
        let drafts = self
            .drafts
            .lock()
            .map_err(|_| CacheError::Io("canonical staging sink lock is poisoned".to_owned()))?;
        for draft in drafts.iter() {
            if draft.family == identity.artifact_family
                && draft.manifest.semantic_digest == identity.semantic_digest
                && draft.manifest.payload_digest == identity.payload_digest
                && draft.manifest.digest()? == identity.manifest_digest
            {
                return Ok(true);
            }
        }
        Ok(false)
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
            .join(&draft.manifest.payload_digest.0)
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
            .join(&draft.manifest.payload_digest.0)
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

const QUADRATURE_KINDS: &[&str] = &[
    "gauss_legendre_rule",
    "quadrature_rule",
    "quadrature_reference_table",
    "quadrature_validation",
];
const CCM_COMPONENT_KINDS: &[&str] = &[
    "ccm_prime_enumeration",
    "ccm_archimedean_integrals",
    "ccm_archimedean_component",
    "ccm_prime_component",
    "ccm_pole_component",
];
const CCM_MATRIX_KINDS: &[&str] = &[
    "ccm_tau_matrix",
    "ccm_even_sector_matrix",
    "ccm_odd_sector_matrix",
    "ccm_sector_tridiagonal",
    "ccm_sector_transform",
    "ccm_reduced_operator",
    "ccm_factorization",
];
const WEIL_STATE_KINDS: &[&str] = &[
    "ccm_weil_eigenpair",
    "ccm_sector_eigenvalues",
    "ccm_sector_spectrum",
    "ccm_weil_plunge_state",
    "ccm_weil_sonin_state",
    "ccm_source_eigenbasis",
];
const PROLATE_KINDS: &[&str] = &[
    "prolate_eigenvalue_spectrum",
    "ccm_prolate_spectrum",
    "ccm_prolate_basis",
    "ccm_prolate_candidate",
    "ccm_band_concentration",
];
const CCM_ROOT_KINDS: &[&str] = &[
    "ccm_secular_source",
    "ccm_root_count_window",
    "ccm_root_discovery_window",
    "ccm_root_refinement",
    "ccm_spectral_window",
];
const CCM_EVIDENCE_KINDS: &[&str] = &[
    "ccm_convergence_diagnostics",
    "ccm_root_conditioning_analysis",
    "ccm_prime_power_response_analysis",
    "ccm_u_flow_response_analysis",
    "ccm_sector_gap",
    "ccm_sector_gap_certificate",
    "ccm_post_discovery_comparison",
    "ccm_cross_check_record",
    "ccm_validation_record",
    "ccm_certificate_bundle",
];
/// Eigenfunction profiles and target-distance measurements. These are
/// derived measurement products: they are reproducible from a retained
/// eigenstate plus a stated quadrature convention, and are published so that
/// downstream analysis does not have to repeat the spectral solve.
const CCM_DISTANCE_KINDS: &[&str] = &[
    "ccm_deviation_decomposition",
    "ccm_discretization_distance",
    "ccm_distance_resolution_evidence",
    "ccm_eigenfunction_profile",
    "ccm_target_distance",
    "ccm_target_residual_analysis",
];

/// Artifact kinds whose values depend on a private runtime target definition.
///
/// This is a hard confidentiality boundary, not a configurable publication
/// preference. Public reads ignore these kinds. Managed publication withholds
/// them from every public leg: under `Both` they ride only the private leg,
/// and an explicit public-only request fails when nothing staged is
/// public-eligible.
const PRIVATE_ONLY_ARTIFACT_KINDS: &[&str] = &[
    "ccm_deviation_decomposition",
    "ccm_distance_resolution_evidence",
    "ccm_target_distance",
    "ccm_target_residual_analysis",
];

pub fn artifact_kind_is_private_only(kind: &str) -> bool {
    PRIVATE_ONLY_ARTIFACT_KINDS.contains(&kind)
}

/// Whether an artifact kind may be published to a destination.
///
/// Private-only kinds are admitted to the private destination and silently
/// withheld from the public one; every other kind is admitted everywhere.
/// Routing callers use this to split a mixed draft set across destinations
/// instead of failing the whole publication, while the staging, planning, and
/// bootstrap guards remain hard backstops should a private-only artifact ever
/// reach a public surface anyway.
pub fn artifact_kind_admitted_to_destination(
    kind: &str,
    destination: crate::PublicationDestination,
) -> bool {
    destination == crate::PublicationDestination::Private || !artifact_kind_is_private_only(kind)
}
const MAYNARD_TAO_KINDS: &[&str] = &[
    "maynard_basis",
    "maynard_moment_table",
    "maynard_operator",
    "maynard_candidate",
    "maynard_bound",
    "maynard_certificate",
];

pub fn artifact_kinds_for_family(family: &str) -> Option<&'static [&'static str]> {
    match family {
        "quadrature" => Some(QUADRATURE_KINDS),
        "ccm-components" => Some(CCM_COMPONENT_KINDS),
        "ccm-matrices" => Some(CCM_MATRIX_KINDS),
        "weil-states" => Some(WEIL_STATE_KINDS),
        "prolate" => Some(PROLATE_KINDS),
        "ccm-roots" => Some(CCM_ROOT_KINDS),
        "ccm-evidence" => Some(CCM_EVIDENCE_KINDS),
        "ccm-distance" => Some(CCM_DISTANCE_KINDS),
        "maynard-tao" => Some(MAYNARD_TAO_KINDS),
        _ => None,
    }
}

pub fn family_for_artifact_kind(kind: &str) -> Option<&'static str> {
    [
        "quadrature",
        "ccm-components",
        "ccm-matrices",
        "weil-states",
        "prolate",
        "ccm-roots",
        "ccm-evidence",
        "ccm-distance",
        "maynard-tao",
    ]
    .into_iter()
    .find(|family| artifact_kinds_for_family(family).is_some_and(|kinds| kinds.contains(&kind)))
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
    let retained_remote_manifest = record
        .manifest
        .tags
        .get(REMOTE_CANONICAL_MANIFEST_TAG)
        .map(|encoded| serde_json::from_str::<CanonicalArtifactManifest>(encoded))
        .transpose()?;
    let canonical_payload = if let Some(remote) = &retained_remote_manifest {
        crate::validate_retained_canonical_binding(
            remote,
            &record.semantic_key,
            family,
            &ContentDigest::sha256(&record.payload),
            record.payload.len() as u64,
            record.manifest.provenance_digest.as_ref(),
        )?;
        if remote.semantic_digest != semantic_digest {
            return Err(CacheError::InvalidManifest(
                "retained remote canonical manifest disagrees with its validated payload"
                    .to_owned(),
            ));
        }
        remote.canonical_payload.clone()
    } else {
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
        CanonicalPayloadEnvelope {
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
        }
    };
    let payload_digest = canonical_payload.digest()?;
    let draft_root = staging_root
        .join("drafts")
        .join(&semantic_digest.0)
        .join(&payload_digest.0);
    if draft_root.exists() {
        let draft_path = draft_root.join("draft.json");
        // draft.json is written only after the archive and every split part
        // have completed.  Its absence therefore identifies an interrupted
        // staging attempt, not a reusable canonical draft.  Rebuild it from
        // the still-validated produced artifact instead of permanently
        // wedging subsequent author runs after a reboot or process kill.
        if !draft_path.is_file() {
            fs::remove_dir_all(&draft_root)?;
        }
    }
    if draft_root.exists() {
        let draft_path = draft_root.join("draft.json");
        let draft: CanonicalProductionDraft = serde_json::from_slice(&fs::read(&draft_path)?)?;
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
            || draft.manifest.payload_digest != payload_digest
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
    let manifest = if let Some(remote) = retained_remote_manifest {
        if remote.payload_digest != payload_digest
            || !remote.transport_digests.contains(&transport_digest)
        {
            return Err(CacheError::InvalidManifest(
                "retained remote manifest cannot be reproduced by the deterministic transport"
                    .to_owned(),
            ));
        }
        remote
    } else {
        CanonicalArtifactManifest {
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
        }
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
        assert_eq!(
            family_for_artifact_kind("ccm_prime_power_response_analysis"),
            Some("ccm-evidence")
        );
        assert_eq!(
            family_for_artifact_kind("ccm_u_flow_response_analysis"),
            Some("ccm-evidence")
        );
    }

    /// Every artifact kind a family declares must have a compatibility floor.
    ///
    /// A kind can otherwise be declared, routed, and published to a registry
    /// while no code path is able to admit it -- the declaration and the
    /// implementation drift apart silently, and the failure only appears when
    /// a consumer tries to resolve an artifact that was never producible.
    #[test]
    fn every_declared_kind_has_a_compatibility_floor() {
        for family in [
            "quadrature",
            "ccm-components",
            "ccm-matrices",
            "weil-states",
            "prolate",
            "ccm-roots",
            "ccm-evidence",
            "ccm-distance",
        ] {
            for kind in artifact_kinds_for_family(family).unwrap() {
                assert!(
                    crate::artifact_compatibility_policy(family, kind).is_ok(),
                    "family {family} declares {kind} with no compatibility floor"
                );
            }
        }
    }

    #[test]
    fn family_declarations_and_reverse_routing_are_exact() {
        let families = [
            "quadrature",
            "ccm-components",
            "ccm-matrices",
            "weil-states",
            "prolate",
            "ccm-roots",
            "ccm-evidence",
            "ccm-distance",
            "maynard-tao",
        ];
        let mut seen = std::collections::BTreeSet::new();
        for family in families {
            let kinds = artifact_kinds_for_family(family).unwrap();
            assert!(!kinds.is_empty());
            for kind in kinds {
                assert!(seen.insert(*kind), "duplicate artifact kind {kind}");
                assert_eq!(family_for_artifact_kind(kind), Some(family));
            }
        }
    }

    #[test]
    fn runtime_target_artifacts_are_private_only() {
        for kind in [
            "ccm_target_distance",
            "ccm_distance_resolution_evidence",
            "ccm_target_residual_analysis",
            "ccm_deviation_decomposition",
        ] {
            assert!(artifact_kind_is_private_only(kind));
        }
        assert!(!artifact_kind_is_private_only("ccm_eigenfunction_profile"));
        assert!(!artifact_kind_is_private_only(
            "ccm_discretization_distance"
        ));
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
    fn managed_inventory_rejects_public_target_derived_artifacts() {
        let root =
            std::env::temp_dir().join(format!("xc-production-private-only-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut restricted = record();
        restricted.operation = "ccm.target_distance.resolve_or_compute".to_owned();
        restricted.logical_key = "ccm/target-distance/fixture".to_owned();
        restricted.semantic_key.artifact_kind = "ccm_target_distance".to_owned();
        restricted.semantic_key.mathematical_semantics_version =
            "ccm-runtime-target-distance-v0.14.1-v2".to_owned();
        restricted.manifest.key.kind = restricted.semantic_key.artifact_kind.clone();
        restricted.manifest.key.logical_key = restricted.logical_key.clone();
        restricted.manifest.key.parameters_digest = restricted.semantic_key.digest().unwrap();
        restricted.manifest.producer_toolkit_version = ToolkitVersion::parse("0.14.1").unwrap();
        restricted.manifest.minimum_reader_version = ToolkitVersion::parse("0.14.1").unwrap();
        let draft = stage_produced_artifact(
            &restricted,
            &root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();

        let private = build_managed_publication_inventory(
            std::slice::from_ref(&draft),
            xc_core::PublicationTarget::Private,
            "example-org",
            crate::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Computed,
            crate::CertificationFailurePolicy::RetainComputedFailRun,
            false,
        )
        .unwrap();
        assert_eq!(private.entries.len(), 1);
        assert_eq!(
            private.entries[0].destination,
            crate::PublicationDestination::Private
        );

        // An explicit public-only request in which nothing is public-eligible
        // still fails loudly rather than publishing nothing.
        let error = build_managed_publication_inventory(
            std::slice::from_ref(&draft),
            xc_core::PublicationTarget::Public,
            "example-org",
            crate::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Computed,
            crate::CertificationFailurePolicy::RetainComputedFailRun,
            false,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::PermissionDenied(_)));

        // Under `Both`, the private-only draft is routed to the private
        // destination and withheld from the public one instead of failing the
        // publication.
        let both = build_managed_publication_inventory(
            std::slice::from_ref(&draft),
            xc_core::PublicationTarget::Both,
            "example-org",
            crate::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Computed,
            crate::CertificationFailurePolicy::RetainComputedFailRun,
            false,
        )
        .unwrap();
        assert_eq!(both.entries.len(), 1);
        assert_eq!(
            both.entries[0].destination,
            crate::PublicationDestination::Private
        );
        let _ = fs::remove_dir_all(root);
    }

    /// A mixed draft set under `Both` splits by destination: public-eligible
    /// kinds go everywhere, private-only kinds go private, and the run
    /// proceeds.
    #[test]
    fn managed_inventory_routes_mixed_drafts_across_destinations() {
        let root = std::env::temp_dir().join(format!(
            "xc-production-mixed-routing-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut restricted = record();
        restricted.operation = "ccm.target_distance.resolve_or_compute".to_owned();
        restricted.logical_key = "ccm/target-distance/fixture".to_owned();
        restricted.semantic_key.artifact_kind = "ccm_target_distance".to_owned();
        restricted.semantic_key.mathematical_semantics_version =
            "ccm-runtime-target-distance-v0.14.1-v2".to_owned();
        restricted.manifest.key.kind = restricted.semantic_key.artifact_kind.clone();
        restricted.manifest.key.logical_key = restricted.logical_key.clone();
        restricted.manifest.key.parameters_digest = restricted.semantic_key.digest().unwrap();
        restricted.manifest.producer_toolkit_version = ToolkitVersion::parse("0.14.1").unwrap();
        restricted.manifest.minimum_reader_version = ToolkitVersion::parse("0.14.1").unwrap();
        let restricted_draft = stage_produced_artifact(
            &restricted,
            &root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();

        let mut eligible = record();
        eligible.operation = "ccm.eigenfunction_profile.resolve_or_compute".to_owned();
        eligible.logical_key = "ccm/profile/fixture".to_owned();
        eligible.semantic_key.artifact_kind = "ccm_eigenfunction_profile".to_owned();
        eligible.manifest.key.kind = eligible.semantic_key.artifact_kind.clone();
        eligible.manifest.key.logical_key = eligible.logical_key.clone();
        eligible.manifest.key.parameters_digest = eligible.semantic_key.digest().unwrap();
        eligible.manifest.producer_toolkit_version = ToolkitVersion::parse("0.14.1").unwrap();
        eligible.manifest.minimum_reader_version = ToolkitVersion::parse("0.14.0").unwrap();
        let eligible_draft = stage_produced_artifact(
            &eligible,
            &root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();

        let inventory = build_managed_publication_inventory(
            &[restricted_draft, eligible_draft],
            xc_core::PublicationTarget::Both,
            "example-org",
            crate::ManagedRunProfile::Author,
            xc_core::AssuranceLevel::Computed,
            crate::CertificationFailurePolicy::RetainComputedFailRun,
            false,
        )
        .unwrap();
        // restricted: private only. eligible: private and public.
        assert_eq!(inventory.entries.len(), 3);
        let public: Vec<_> = inventory
            .entries
            .iter()
            .filter(|entry| entry.destination == crate::PublicationDestination::Public)
            .collect();
        assert_eq!(public.len(), 1);
        assert_eq!(public[0].family, "ccm-distance");
        let private_kinds = inventory
            .entries
            .iter()
            .filter(|entry| entry.destination == crate::PublicationDestination::Private)
            .count();
        assert_eq!(private_kinds, 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn remotely_reused_record_preserves_its_canonical_dependency_identity() {
        let source_root = std::env::temp_dir().join(format!(
            "xc-production-remote-source-{}",
            std::process::id()
        ));
        let replay_root = std::env::temp_dir().join(format!(
            "xc-production-remote-replay-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&source_root);
        let _ = fs::remove_dir_all(&replay_root);
        let base_record = record();
        let mut source = stage_produced_artifact(
            &base_record,
            &source_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        source
            .manifest
            .canonical_payload
            .dependencies
            .push(crate::PayloadDependencyIdentity {
                artifact_family: "ccm-components".to_owned(),
                semantic_digest: ContentDigest::sha256(b"dependency-semantic"),
                manifest_digest: ContentDigest::sha256(b"dependency-manifest"),
                payload_digest: ContentDigest::sha256(b"dependency-payload"),
            });
        source.manifest.payload_digest = source.manifest.canonical_payload.digest().unwrap();
        source.encoding.canonical_payload_digest = source.manifest.payload_digest.clone();
        let archive = replay_root.join("expected.zip");
        let package = package_canonical_payload_bytes_zip64(
            &source.manifest.canonical_payload,
            "payload.json",
            &base_record.payload,
            &archive,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let mut encoded_bytes = BufReader::new(File::open(&archive).unwrap());
        source.encoding = stream_split_encoded(
            &mut encoded_bytes,
            package.canonical_payload_digest,
            package.encoder_profile,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            |_part, _bytes| Ok(()),
        )
        .unwrap();
        source.manifest.transport_digests = vec![source.encoding.digest().unwrap()];
        source.manifest.validate().unwrap();

        let mut reused = base_record;
        reused.manifest.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&source.manifest).unwrap(),
        );
        let staged = stage_produced_artifact(
            &reused,
            &replay_root.join("staging"),
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(staged.manifest, source.manifest);
        assert_eq!(
            staged.manifest.canonical_payload.dependencies,
            source.manifest.canonical_payload.dependencies
        );
        let _ = fs::remove_dir_all(source_root);
        let _ = fs::remove_dir_all(replay_root);
    }

    #[test]
    fn historical_and_active_manifests_with_same_semantic_and_item_coexist() {
        let root = std::env::temp_dir().join(format!(
            "xc-production-same-item-distinct-closure-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let base = record();
        let historical = stage_produced_artifact(
            &base,
            &root.join("historical-source"),
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();

        // Reproduce the production-corpus shape: a later manifest retains the
        // exact same semantic key and payload.json bytes but adds a canonical
        // dependency, so its canonical payload and manifest identities differ.
        let mut active = historical.clone();
        active
            .manifest
            .canonical_payload
            .dependencies
            .push(crate::PayloadDependencyIdentity {
                artifact_family: "ccm-matrices".to_owned(),
                semantic_digest: ContentDigest::sha256(b"active-parent-semantic"),
                manifest_digest: ContentDigest::sha256(b"active-parent-manifest"),
                payload_digest: ContentDigest::sha256(b"active-parent-payload"),
            });
        active.manifest.payload_digest = active.manifest.canonical_payload.digest().unwrap();
        let archive = root.join("active-source.zip");
        let package = package_canonical_payload_bytes_zip64(
            &active.manifest.canonical_payload,
            "payload.json",
            &base.payload,
            &archive,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let mut encoded_bytes = BufReader::new(File::open(&archive).unwrap());
        active.encoding = stream_split_encoded(
            &mut encoded_bytes,
            package.canonical_payload_digest,
            package.encoder_profile,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            |_part, _bytes| Ok(()),
        )
        .unwrap();
        active.manifest.transport_digests = vec![active.encoding.digest().unwrap()];
        active.manifest.validate().unwrap();
        assert_eq!(
            historical.manifest.semantic_digest,
            active.manifest.semantic_digest
        );
        assert_eq!(
            historical.manifest.canonical_payload.ordered_items,
            active.manifest.canonical_payload.ordered_items
        );
        assert_ne!(
            historical.manifest.payload_digest,
            active.manifest.payload_digest
        );
        assert_ne!(
            historical.manifest.digest().unwrap(),
            active.manifest.digest().unwrap()
        );

        let mut historical_adapter = base.clone();
        historical_adapter.operation = "cache.dependency.closure".to_owned();
        historical_adapter.logical_key = "closure/historical-matrix".to_owned();
        historical_adapter.manifest.key.logical_key = historical_adapter.logical_key.clone();
        historical_adapter.manifest.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&historical.manifest).unwrap(),
        );
        historical_adapter.manifest.provenance_digest = Some(historical.manifest.digest().unwrap());

        let mut active_adapter = base;
        active_adapter.operation = "ccm.even-sector.resolve".to_owned();
        active_adapter.manifest.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&active.manifest).unwrap(),
        );
        active_adapter.manifest.provenance_digest = Some(active.manifest.digest().unwrap());

        let staging_root = root.join("staging");
        let sink = CanonicalStagingProductionSink::new(
            &staging_root,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        crate::ArtifactProductionSink::record(&sink, historical_adapter).unwrap();
        crate::ArtifactProductionSink::record(&sink, active_adapter).unwrap();
        let drafts = sink.drafts().unwrap();
        assert_eq!(drafts.len(), 2);
        for draft in drafts {
            assert!(staging_root
                .join("drafts")
                .join(&draft.manifest.semantic_digest.0)
                .join(&draft.manifest.payload_digest.0)
                .join("draft.json")
                .is_file());
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn identity_first_dedup_persists_real_key_for_fresh_child() {
        let root = std::env::temp_dir().join(format!(
            "xc-production-identity-first-key-promotion-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let source_root = root.join("source");
        let staging_root = root.join("staging");

        let dependency = record();
        let published = stage_produced_artifact(
            &dependency,
            &source_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let published_manifest_digest = published.manifest.digest().unwrap();
        let retained = serde_json::to_string(&published.manifest).unwrap();

        let mut synthetic = dependency.clone();
        synthetic.operation = "cache.dependency.closure".to_owned();
        synthetic.logical_key = "closure/synthetic-identity".to_owned();
        synthetic.manifest.key.logical_key = synthetic.logical_key.clone();
        synthetic
            .manifest
            .tags
            .insert(REMOTE_CANONICAL_MANIFEST_TAG.to_owned(), retained.clone());
        synthetic.manifest.provenance_digest = Some(published_manifest_digest.clone());

        let mut real = dependency.clone();
        real.operation = "cache.dependency.resolve".to_owned();
        real.manifest
            .tags
            .insert(REMOTE_CANONICAL_MANIFEST_TAG.to_owned(), retained);
        real.manifest.provenance_digest = Some(published_manifest_digest);

        let sink = CanonicalStagingProductionSink::new(
            &staging_root,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        crate::ArtifactProductionSink::record(&sink, synthetic).unwrap();
        crate::ArtifactProductionSink::record(&sink, real.clone()).unwrap();
        let promoted = sink.drafts().unwrap();
        assert_eq!(promoted.len(), 1, "published identity remains deduplicated");
        assert_eq!(
            promoted[0].source_artifact_key, real.manifest.key,
            "the key-based adapter must replace synthetic closure provenance"
        );
        drop(sink);

        // Reopen the staging directory to prove the promotion was persisted,
        // then stage a newly computed child that names the dependency by its
        // real adapter key -- the production failure this regression covers.
        let sink = CanonicalStagingProductionSink::new(
            &staging_root,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(
            sink.drafts().unwrap()[0].source_artifact_key,
            real.manifest.key
        );

        let mut child = record();
        child.operation = "ccm.factorization.resolve_or_compute".to_owned();
        child.logical_key = "ccm/factorization/fixture".to_owned();
        child.semantic_key.artifact_kind = "ccm_factorization".to_owned();
        child.semantic_key.resolved_mathematical_parameters = json!({"role": "child"});
        child.manifest.key.kind = child.semantic_key.artifact_kind.clone();
        child.manifest.key.logical_key = child.logical_key.clone();
        child.manifest.key.parameters_digest = child.semantic_key.digest().unwrap();
        child.payload = br#"{"child":"fresh"}"#.to_vec();
        child.manifest.content_digest = ContentDigest::sha256(&child.payload);
        child.manifest.size_bytes = child.payload.len() as u64;
        child.manifest.objects = vec![crate::CacheObjectRef {
            content_digest: child.manifest.content_digest.clone(),
            size_bytes: child.manifest.size_bytes,
        }];
        child.manifest.dependencies = vec![crate::DependencyRef {
            key: real.manifest.key.clone(),
            content_digest: real.manifest.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        }];
        child.manifest.tags.clear();
        child.manifest.provenance_digest = None;

        crate::ArtifactProductionSink::record(&sink, child).unwrap();
        let drafts = sink.drafts().unwrap();
        assert_eq!(drafts.len(), 2);
        let child = drafts
            .iter()
            .find(|draft| draft.source_artifact_key.kind == "ccm_factorization")
            .unwrap();
        assert_eq!(child.manifest.canonical_payload.dependencies.len(), 1);
        assert_eq!(
            child.manifest.canonical_payload.dependencies[0],
            crate::PayloadDependencyIdentity {
                artifact_family: promoted[0].family.clone(),
                semantic_digest: promoted[0].manifest.semantic_digest.clone(),
                manifest_digest: promoted[0].manifest.digest().unwrap(),
                payload_digest: promoted[0].manifest.payload_digest.clone(),
            }
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn interrupted_canonical_draft_is_rebuilt_from_validated_record() {
        let root = std::env::temp_dir().join(format!(
            "xc-production-interrupted-stage-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let record = record();
        let reference_root = root.with_extension("reference");
        let _ = fs::remove_dir_all(&reference_root);
        let reference = stage_produced_artifact(
            &record,
            &reference_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let draft_root = root
            .join("drafts")
            .join(record.semantic_key.digest().unwrap().0)
            .join(&reference.manifest.payload_digest.0);
        fs::create_dir_all(draft_root.join("parts")).unwrap();
        fs::write(draft_root.join("interrupted.part"), b"partial").unwrap();

        let rebuilt = stage_produced_artifact(
            &record,
            &root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();

        assert!(draft_root.join("draft.json").is_file());
        assert!(!draft_root.join("interrupted.part").exists());
        assert_eq!(
            rebuilt.source_content_digest,
            record.manifest.content_digest
        );
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(reference_root);
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
