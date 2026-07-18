//! Durability-copy assessment and fail-closed local prune planning.

use crate::{
    ArtifactAssuranceState, ArtifactDisposition, CacheError, ContentDigest, PublicationDestination,
    PublicationTargetState,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use xc_core::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurabilityClass {
    Recomputable,
    ExpensiveReproducible,
    IrreplaceableSource,
    PublicationOrCertificateRecord,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactCopyLocation {
    ProcessLocal,
    WorkstationLocal,
    ProjectPrivate,
    TeamPrivateRemote,
    PublicRemote,
    ExternalMirror,
    Archive,
}

impl ArtifactCopyLocation {
    fn is_remote(self) -> bool {
        matches!(
            self,
            Self::TeamPrivateRemote | Self::PublicRemote | Self::ExternalMirror | Self::Archive
        )
    }

    fn is_locally_prunable(self) -> bool {
        matches!(
            self,
            Self::ProcessLocal | Self::WorkstationLocal | Self::ProjectPrivate
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurabilityPolicy {
    pub schema_version: u32,
    pub artifact_family: String,
    pub class: DurabilityClass,
    pub minimum_verified_copies: u32,
    pub minimum_independent_failure_domains: u32,
    pub require_archive_copy: bool,
    pub allowed_locations: BTreeSet<ArtifactCopyLocation>,
}

impl DurabilityPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.artifact_family.trim().is_empty()
            || self.minimum_verified_copies == 0
            || self.minimum_independent_failure_domains == 0
            || self.minimum_independent_failure_domains > self.minimum_verified_copies
            || self.allowed_locations.is_empty()
            || (self.require_archive_copy
                && !self
                    .allowed_locations
                    .contains(&ArtifactCopyLocation::Archive))
        {
            return Err(CacheError::InvalidManifest(
                "durability policy identity, copy count, independence, or locations are invalid"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactCopyEvidence {
    pub schema_version: u32,
    pub copy_id: String,
    pub location: ArtifactCopyLocation,
    pub locator: String,
    pub failure_domain: String,
    pub canonical_payload_digest: ContentDigest,
    /// Publication receipt for GitHub copies or an equivalent immutable
    /// placement/deposit receipt for mirrors and archives.
    pub placement_receipt_digest: Option<ContentDigest>,
    pub verification_evidence_digest: ContentDigest,
    pub verified_at_unix_seconds: u64,
    pub revoked: bool,
}

impl ArtifactCopyEvidence {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.copy_id.trim().is_empty()
            || self.locator.trim().is_empty()
            || self.failure_domain.trim().is_empty()
            || !self.canonical_payload_digest.validate()
            || !self.verification_evidence_digest.validate()
            || self.verified_at_unix_seconds == 0
            || self
                .placement_receipt_digest
                .as_ref()
                .is_some_and(|digest| !digest.validate())
            || (self.location.is_remote() && self.placement_receipt_digest.is_none())
        {
            return Err(CacheError::InvalidManifest(
                "artifact copy identity, verification, failure domain, or remote receipt is invalid"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurabilityCopyRejectionReason {
    InvalidEvidence,
    WrongPayload,
    DisallowedLocation,
    Revoked,
    DuplicateCopyId,
    DuplicateLocator,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurabilityCopyRejection {
    pub copy_id: String,
    pub reason: DurabilityCopyRejectionReason,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurabilityAssessment {
    pub schema_version: u32,
    pub artifact_family: String,
    pub class: DurabilityClass,
    pub canonical_payload_digest: ContentDigest,
    pub satisfied: bool,
    pub verified_copy_count: u32,
    pub independent_failure_domain_count: u32,
    pub archive_copy_present: bool,
    pub accepted_copy_ids: Vec<String>,
    pub rejections: Vec<DurabilityCopyRejection>,
    pub deficits: Vec<String>,
}

pub fn assess_durability(
    policy: &DurabilityPolicy,
    canonical_payload_digest: &ContentDigest,
    copies: &[ArtifactCopyEvidence],
) -> Result<DurabilityAssessment, CacheError> {
    policy.validate()?;
    if !canonical_payload_digest.validate() {
        return Err(CacheError::InvalidManifest(
            "durability subject payload digest is invalid".to_owned(),
        ));
    }
    let mut accepted_copy_ids = Vec::new();
    let mut accepted_ids = BTreeSet::new();
    let mut accepted_locators = BTreeSet::new();
    let mut failure_domains = BTreeSet::new();
    let mut archive_copy_present = false;
    let mut rejections = Vec::new();
    for copy in copies {
        let rejected = if let Err(error) = copy.validate() {
            Some((
                DurabilityCopyRejectionReason::InvalidEvidence,
                error.to_string(),
            ))
        } else if &copy.canonical_payload_digest != canonical_payload_digest {
            Some((
                DurabilityCopyRejectionReason::WrongPayload,
                "copy names another canonical payload".to_owned(),
            ))
        } else if !policy.allowed_locations.contains(&copy.location) {
            Some((
                DurabilityCopyRejectionReason::DisallowedLocation,
                "copy location is excluded by durability policy".to_owned(),
            ))
        } else if copy.revoked {
            Some((
                DurabilityCopyRejectionReason::Revoked,
                "copy or its evidence is revoked".to_owned(),
            ))
        } else if !accepted_ids.insert(copy.copy_id.clone()) {
            Some((
                DurabilityCopyRejectionReason::DuplicateCopyId,
                "copy identity is duplicated".to_owned(),
            ))
        } else if !accepted_locators.insert(copy.locator.clone()) {
            Some((
                DurabilityCopyRejectionReason::DuplicateLocator,
                "copy locator is duplicated".to_owned(),
            ))
        } else {
            None
        };
        if let Some((reason, detail)) = rejected {
            rejections.push(DurabilityCopyRejection {
                copy_id: copy.copy_id.clone(),
                reason,
                detail,
            });
            continue;
        }
        failure_domains.insert(copy.failure_domain.clone());
        archive_copy_present |= copy.location == ArtifactCopyLocation::Archive;
        accepted_copy_ids.push(copy.copy_id.clone());
    }
    accepted_copy_ids.sort();
    let verified_copy_count = u32::try_from(accepted_copy_ids.len()).unwrap_or(u32::MAX);
    let independent_failure_domain_count = u32::try_from(failure_domains.len()).unwrap_or(u32::MAX);
    let mut deficits = Vec::new();
    if verified_copy_count < policy.minimum_verified_copies {
        deficits.push(format!(
            "verified copies {verified_copy_count} are below required {}",
            policy.minimum_verified_copies
        ));
    }
    if independent_failure_domain_count < policy.minimum_independent_failure_domains {
        deficits.push(format!(
            "independent failure domains {independent_failure_domain_count} are below required {}",
            policy.minimum_independent_failure_domains
        ));
    }
    if policy.require_archive_copy && !archive_copy_present {
        deficits.push("an independently verified archive copy is required".to_owned());
    }
    rejections.sort_by(|left, right| left.copy_id.cmp(&right.copy_id));
    Ok(DurabilityAssessment {
        schema_version: 1,
        artifact_family: policy.artifact_family.clone(),
        class: policy.class,
        canonical_payload_digest: canonical_payload_digest.clone(),
        satisfied: deficits.is_empty(),
        verified_copy_count,
        independent_failure_domain_count,
        archive_copy_present,
        accepted_copy_ids,
        rejections,
        deficits,
    })
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecomputationCost {
    Trivial,
    Moderate,
    Expensive,
    Irreplaceable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPrunePolicy {
    pub schema_version: u32,
    pub minimum_age_seconds: u64,
    pub maximum_recomputation_cost: RecomputationCost,
    pub allowed_dispositions: BTreeSet<ArtifactDisposition>,
    pub protect_assurance_at_or_above: Option<ArtifactAssuranceState>,
}

impl LocalPrunePolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0 || self.allowed_dispositions.is_empty() {
            return Err(CacheError::InvalidManifest(
                "local prune policy schema and dispositions are required".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPruneCandidate {
    pub schema_version: u32,
    pub copy_id: String,
    pub local_path: PathBuf,
    pub local_file_digest: ContentDigest,
    pub local_file_size_bytes: u64,
    pub location: ArtifactCopyLocation,
    pub artifact_family: String,
    pub canonical_payload_digest: ContentDigest,
    pub age_seconds: u64,
    pub project_pinned: bool,
    pub dependency_reachable: bool,
    pub active_transaction: bool,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub recomputation_cost: RecomputationCost,
    pub required_publication_targets: BTreeSet<PublicationDestination>,
    pub publication_states: BTreeMap<PublicationDestination, PublicationTargetState>,
}

impl LocalPruneCandidate {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.copy_id.trim().is_empty()
            || self.local_path.as_os_str().is_empty()
            || !self.local_file_digest.validate()
            || !self.location.is_locally_prunable()
            || self.artifact_family.trim().is_empty()
            || !self.canonical_payload_digest.validate()
            || self
                .required_publication_targets
                .iter()
                .any(|target| !self.publication_states.contains_key(target))
        {
            return Err(CacheError::InvalidManifest(
                "local prune candidate identity, path, location, or target state is invalid"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPrunePlan {
    pub schema_version: u32,
    pub dry_run: bool,
    pub copy_id: String,
    pub local_path: PathBuf,
    pub removable: bool,
    pub durability_after_removal: DurabilityAssessment,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPruneExecutionReport {
    pub schema_version: u32,
    pub plan: LocalPrunePlan,
    pub deleted: bool,
    pub deleted_file_digest: ContentDigest,
    pub deleted_file_size_bytes: u64,
}

pub fn plan_local_prune(
    policy: &LocalPrunePolicy,
    durability_policy: &DurabilityPolicy,
    candidate: &LocalPruneCandidate,
    copies: &[ArtifactCopyEvidence],
) -> Result<LocalPrunePlan, CacheError> {
    policy.validate()?;
    durability_policy.validate()?;
    candidate.validate()?;
    if durability_policy.artifact_family != candidate.artifact_family {
        return Err(CacheError::InvalidManifest(
            "prune candidate and durability policy name different artifact families".to_owned(),
        ));
    }
    let retained_copies = copies
        .iter()
        .filter(|copy| copy.copy_id != candidate.copy_id)
        .cloned()
        .collect::<Vec<_>>();
    let durability_after_removal = assess_durability(
        durability_policy,
        &candidate.canonical_payload_digest,
        &retained_copies,
    )?;
    let mut reasons = Vec::new();
    if candidate.age_seconds < policy.minimum_age_seconds {
        reasons.push(format!(
            "candidate age {} seconds is below policy minimum {}",
            candidate.age_seconds, policy.minimum_age_seconds
        ));
    }
    if candidate.project_pinned {
        reasons.push("candidate is protected by a project pin".to_owned());
    }
    if candidate.dependency_reachable {
        reasons.push("candidate remains reachable from a retained dependency".to_owned());
    }
    if candidate.active_transaction {
        reasons.push("candidate is used by an active publication transaction".to_owned());
    }
    if !policy.allowed_dispositions.contains(&candidate.disposition) {
        reasons.push(format!(
            "candidate disposition {:?} is protected",
            candidate.disposition
        ));
    }
    if candidate.recomputation_cost > policy.maximum_recomputation_cost {
        reasons.push(format!(
            "candidate recomputation cost {:?} exceeds policy maximum {:?}",
            candidate.recomputation_cost, policy.maximum_recomputation_cost
        ));
    }
    if policy
        .protect_assurance_at_or_above
        .is_some_and(|minimum| candidate.achieved_assurance >= minimum)
    {
        reasons.push(format!(
            "candidate assurance {:?} is protected by policy",
            candidate.achieved_assurance
        ));
    }
    for target in &candidate.required_publication_targets {
        if candidate.publication_states.get(target)
            != Some(&PublicationTargetState::ReceiptComplete)
        {
            reasons.push(format!(
                "required {target:?} publication is not receipt-complete"
            ));
        }
    }
    reasons.extend(
        durability_after_removal
            .deficits
            .iter()
            .map(|deficit| format!("durability deficit after removal: {deficit}")),
    );
    reasons.sort();
    reasons.dedup();
    Ok(LocalPrunePlan {
        schema_version: 1,
        dry_run: true,
        copy_id: candidate.copy_id.clone(),
        local_path: candidate.local_path.clone(),
        removable: reasons.is_empty(),
        durability_after_removal,
        reasons,
    })
}

pub fn execute_local_prune(
    policy: &LocalPrunePolicy,
    durability_policy: &DurabilityPolicy,
    candidate: &LocalPruneCandidate,
    copies: &[ArtifactCopyEvidence],
    cancellation: &CancellationToken,
) -> Result<LocalPruneExecutionReport, CacheError> {
    let plan = plan_local_prune(policy, durability_policy, candidate, copies)?;
    if !plan.removable {
        return Err(CacheError::PermissionDenied(format!(
            "local prune is refused: {}",
            plan.reasons.join("; ")
        )));
    }
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    let metadata = std::fs::symlink_metadata(&candidate.local_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CacheError::NotFound(candidate.local_path.display().to_string())
        } else {
            error.into()
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CacheError::InvalidManifest(
            "local prune target must be one regular non-symlink file".to_owned(),
        ));
    }
    if metadata.len() != candidate.local_file_size_bytes {
        return Err(CacheError::DigestMismatch {
            expected: format!("{} bytes", candidate.local_file_size_bytes),
            actual: format!("{} bytes", metadata.len()),
        });
    }
    let mut reader = BufReader::new(File::open(&candidate.local_path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut observed_size = 0_u64;
    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        observed_size = observed_size.checked_add(read as u64).ok_or_else(|| {
            CacheError::ResourceLimit("local prune verification size exceeds u64".to_owned())
        })?;
        hasher.update(&buffer[..read]);
    }
    let observed_digest = ContentDigest(format!("{:x}", hasher.finalize()));
    if observed_size != candidate.local_file_size_bytes
        || observed_digest != candidate.local_file_digest
    {
        return Err(CacheError::DigestMismatch {
            expected: format!(
                "{} ({} bytes)",
                candidate.local_file_digest, candidate.local_file_size_bytes
            ),
            actual: format!("{observed_digest} ({observed_size} bytes)"),
        });
    }
    drop(reader);
    std::fs::remove_file(&candidate.local_path)?;
    Ok(LocalPruneExecutionReport {
        schema_version: 1,
        plan,
        deleted: true,
        deleted_file_digest: observed_digest,
        deleted_file_size_bytes: observed_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &[u8]) -> ContentDigest {
        ContentDigest::sha256(value)
    }

    fn copy(
        copy_id: &str,
        location: ArtifactCopyLocation,
        failure_domain: &str,
    ) -> ArtifactCopyEvidence {
        ArtifactCopyEvidence {
            schema_version: 1,
            copy_id: copy_id.to_owned(),
            location,
            locator: format!("{failure_domain}/{copy_id}"),
            failure_domain: failure_domain.to_owned(),
            canonical_payload_digest: digest(b"payload"),
            placement_receipt_digest: location.is_remote().then(|| digest(copy_id.as_bytes())),
            verification_evidence_digest: digest(format!("verify-{copy_id}").as_bytes()),
            verified_at_unix_seconds: 1,
            revoked: false,
        }
    }

    fn durability_policy(class: DurabilityClass, copies: u32, domains: u32) -> DurabilityPolicy {
        DurabilityPolicy {
            schema_version: 1,
            artifact_family: "ccm".to_owned(),
            class,
            minimum_verified_copies: copies,
            minimum_independent_failure_domains: domains,
            require_archive_copy: class == DurabilityClass::IrreplaceableSource,
            allowed_locations: BTreeSet::from([
                ArtifactCopyLocation::WorkstationLocal,
                ArtifactCopyLocation::TeamPrivateRemote,
                ArtifactCopyLocation::PublicRemote,
                ArtifactCopyLocation::Archive,
            ]),
        }
    }

    fn candidate() -> LocalPruneCandidate {
        LocalPruneCandidate {
            schema_version: 1,
            copy_id: "local".to_owned(),
            local_path: PathBuf::from("staging/artifact.zip"),
            local_file_digest: digest(b"staged artifact"),
            local_file_size_bytes: 15,
            location: ArtifactCopyLocation::WorkstationLocal,
            artifact_family: "ccm".to_owned(),
            canonical_payload_digest: digest(b"payload"),
            age_seconds: 10_000,
            project_pinned: false,
            dependency_reachable: false,
            active_transaction: false,
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            recomputation_cost: RecomputationCost::Moderate,
            required_publication_targets: BTreeSet::from([PublicationDestination::Private]),
            publication_states: BTreeMap::from([(
                PublicationDestination::Private,
                PublicationTargetState::ReceiptComplete,
            )]),
        }
    }

    fn prune_policy() -> LocalPrunePolicy {
        LocalPrunePolicy {
            schema_version: 1,
            minimum_age_seconds: 100,
            maximum_recomputation_cost: RecomputationCost::Moderate,
            allowed_dispositions: BTreeSet::from([ArtifactDisposition::Active]),
            protect_assurance_at_or_above: Some(ArtifactAssuranceState::Certified),
        }
    }

    #[test]
    fn recomputable_staging_is_removable_after_verified_remote_receipt() {
        let plan = plan_local_prune(
            &prune_policy(),
            &durability_policy(DurabilityClass::Recomputable, 1, 1),
            &candidate(),
            &[
                copy(
                    "local",
                    ArtifactCopyLocation::WorkstationLocal,
                    "workstation",
                ),
                copy(
                    "private",
                    ArtifactCopyLocation::TeamPrivateRemote,
                    "github-private",
                ),
            ],
        )
        .unwrap();
        assert!(plan.removable);
        assert!(plan.durability_after_removal.satisfied);
    }

    #[test]
    fn eligible_exact_local_file_is_deleted_after_revalidation() {
        let root = std::env::temp_dir().join(format!(
            "xc-prune-{}",
            ContentDigest::sha256(format!("{:?}", std::thread::current().id()).as_bytes())
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("artifact.zip");
        std::fs::write(&path, b"staged artifact").unwrap();
        let mut candidate = candidate();
        candidate.local_path = path.clone();
        let report = execute_local_prune(
            &prune_policy(),
            &durability_policy(DurabilityClass::Recomputable, 1, 1),
            &candidate,
            &[
                copy(
                    "local",
                    ArtifactCopyLocation::WorkstationLocal,
                    "workstation",
                ),
                copy(
                    "private",
                    ArtifactCopyLocation::TeamPrivateRemote,
                    "github-private",
                ),
            ],
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(report.deleted);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn changed_local_file_is_never_deleted() {
        let root = std::env::temp_dir().join("xc-prune-changed-file");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("artifact.zip");
        std::fs::write(&path, b"different bytes").unwrap();
        let mut candidate = candidate();
        candidate.local_path = path.clone();
        candidate.local_file_size_bytes = 15;
        assert!(execute_local_prune(
            &prune_policy(),
            &durability_policy(DurabilityClass::Recomputable, 1, 1),
            &candidate,
            &[
                copy(
                    "local",
                    ArtifactCopyLocation::WorkstationLocal,
                    "workstation",
                ),
                copy(
                    "private",
                    ArtifactCopyLocation::TeamPrivateRemote,
                    "github-private",
                ),
            ],
            &CancellationToken::new(),
        )
        .is_err());
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn irreplaceable_staging_is_retained_without_independent_archive() {
        let plan = plan_local_prune(
            &prune_policy(),
            &durability_policy(DurabilityClass::IrreplaceableSource, 2, 2),
            &candidate(),
            &[
                copy(
                    "local",
                    ArtifactCopyLocation::WorkstationLocal,
                    "workstation",
                ),
                copy("private", ArtifactCopyLocation::TeamPrivateRemote, "github"),
                copy("public", ArtifactCopyLocation::PublicRemote, "github"),
            ],
        )
        .unwrap();
        assert!(!plan.removable);
        assert!(plan.reasons.iter().any(|reason| reason.contains("archive")));
        assert!(plan
            .reasons
            .iter()
            .any(|reason| reason.contains("failure domains")));
    }

    #[test]
    fn prune_plan_reports_every_active_protection_axis() {
        let mut candidate = candidate();
        candidate.age_seconds = 1;
        candidate.project_pinned = true;
        candidate.dependency_reachable = true;
        candidate.active_transaction = true;
        candidate.recomputation_cost = RecomputationCost::Expensive;
        candidate.publication_states.insert(
            PublicationDestination::Private,
            PublicationTargetState::Failed,
        );
        let plan = plan_local_prune(
            &prune_policy(),
            &durability_policy(DurabilityClass::Recomputable, 1, 1),
            &candidate,
            &[copy(
                "local",
                ArtifactCopyLocation::WorkstationLocal,
                "workstation",
            )],
        )
        .unwrap();
        assert!(!plan.removable);
        assert!(plan.reasons.len() >= 6);
    }
}
