//! Portable, dependency-complete cache bundles for disconnected operation.

use crate::protocol::{canonical_digest, canonical_json_bytes, normalized_relative_path};
use crate::{
    reconstruct_transport_package, verify_canonical_payload_zip64, ArtifactAssuranceState,
    ArtifactDisposition, CacheError, CanonicalArtifactManifest, ContentDigest,
    DeterministicPackageReport, PublicationReceipt, ToolkitVersion, TransportEncodingRecord,
    TransportPart, VerifiedPackageReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xc_core::{CancellationToken, ResourcePolicy};

pub const CACHE_BUNDLE_MANIFEST_PATH: &str = "bundle.json";
pub const CACHE_BUNDLE_PARTS_DIRECTORY: &str = "parts";

const COPY_BUFFER_BYTES: u64 = 1024 * 1024;
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleArtifactIdentity {
    pub artifact_family: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub payload_digest: ContentDigest,
}

impl CacheBundleArtifactIdentity {
    pub fn from_manifest(manifest: &CanonicalArtifactManifest) -> Result<Self, CacheError> {
        Ok(Self {
            artifact_family: manifest.artifact_family.clone(),
            semantic_digest: manifest.semantic_digest.clone(),
            manifest_digest: manifest.digest()?,
            payload_digest: manifest.payload_digest.clone(),
        })
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.artifact_family.trim().is_empty()
            || !self.semantic_digest.validate()
            || !self.manifest_digest.validate()
            || !self.payload_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "cache bundle artifact identity is incomplete".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleArtifactRecord {
    pub identity: CacheBundleArtifactIdentity,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub manifest: CanonicalArtifactManifest,
    pub encoding: TransportEncodingRecord,
    pub origin_receipt: Option<PublicationReceipt>,
}

impl CacheBundleArtifactRecord {
    pub fn validate(&self) -> Result<(), CacheError> {
        self.identity.validate()?;
        self.manifest.validate()?;
        self.encoding.validate()?;
        let identity = CacheBundleArtifactIdentity::from_manifest(&self.manifest)?;
        let transport_digest = self.encoding.digest()?;
        if identity != self.identity
            || self.encoding.canonical_payload_digest != self.manifest.payload_digest
            || !self.manifest.transport_digests.contains(&transport_digest)
        {
            return Err(CacheError::InvalidManifest(
                "cache bundle record identities do not match its manifest and encoding".to_owned(),
            ));
        }
        if let Some(receipt) = &self.origin_receipt {
            receipt.validate()?;
            if receipt.semantic_digest != self.identity.semantic_digest
                || receipt.canonical_payload_digest != self.identity.payload_digest
                || receipt.manifest_digest != self.identity.manifest_digest
                || receipt.transport_digest != transport_digest
            {
                return Err(CacheError::InvalidManifest(
                    "cache bundle origin receipt does not bind the exported artifact".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleManifest {
    pub schema_version: u32,
    pub roots: Vec<CacheBundleArtifactIdentity>,
    pub artifacts: Vec<CacheBundleArtifactRecord>,
}

impl CacheBundleManifest {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0 || self.roots.is_empty() || self.artifacts.is_empty() {
            return Err(CacheError::InvalidManifest(
                "cache bundle requires a schema, roots, and artifacts".to_owned(),
            ));
        }
        if self.roots.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .artifacts
                .windows(2)
                .any(|pair| pair[0].identity >= pair[1].identity)
        {
            return Err(CacheError::InvalidManifest(
                "cache bundle roots and artifacts must be uniquely and canonically ordered"
                    .to_owned(),
            ));
        }
        let mut identities = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            identities.insert(artifact.identity.clone());
        }
        for root in &self.roots {
            root.validate()?;
            if !identities.contains(root) {
                return Err(CacheError::InvalidManifest(format!(
                    "cache bundle root {:?} is absent",
                    root.semantic_digest
                )));
            }
        }
        let mut dependencies =
            BTreeMap::<CacheBundleArtifactIdentity, Vec<CacheBundleArtifactIdentity>>::new();
        for artifact in &self.artifacts {
            let mut artifact_dependencies = Vec::new();
            for dependency in &artifact.manifest.canonical_payload.dependencies {
                let identity = CacheBundleArtifactIdentity {
                    artifact_family: dependency.artifact_family.clone(),
                    semantic_digest: dependency.semantic_digest.clone(),
                    manifest_digest: dependency.manifest_digest.clone(),
                    payload_digest: dependency.payload_digest.clone(),
                };
                if !identities.contains(&identity) {
                    return Err(CacheError::InvalidManifest(format!(
                        "cache bundle artifact {:?} has a missing exact dependency {:?}",
                        artifact.identity.semantic_digest, identity.semantic_digest
                    )));
                }
                artifact_dependencies.push(identity);
            }
            dependencies.insert(artifact.identity.clone(), artifact_dependencies);
        }
        let mut reachable = BTreeSet::new();
        let mut pending = self.roots.clone();
        while let Some(identity) = pending.pop() {
            if reachable.insert(identity.clone()) {
                pending.extend(dependencies[&identity].iter().cloned());
            }
        }
        if reachable != identities {
            return Err(CacheError::InvalidManifest(
                "cache bundle contains artifacts outside the declared root closure".to_owned(),
            ));
        }
        let mut incoming = dependencies
            .iter()
            .map(|(identity, values)| (identity.clone(), values.len()))
            .collect::<BTreeMap<_, _>>();
        let mut dependents =
            BTreeMap::<CacheBundleArtifactIdentity, BTreeSet<CacheBundleArtifactIdentity>>::new();
        for (identity, values) in &dependencies {
            for dependency in values {
                dependents
                    .entry(dependency.clone())
                    .or_default()
                    .insert(identity.clone());
            }
        }
        let mut ready = incoming
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(identity, _)| identity.clone())
            .collect::<BTreeSet<_>>();
        let mut visited = 0usize;
        while let Some(identity) = ready.pop_first() {
            visited += 1;
            for dependent in dependents.get(&identity).into_iter().flatten() {
                let count = incoming
                    .get_mut(dependent)
                    .expect("validated cache bundle contains every dependent");
                *count -= 1;
                if *count == 0 {
                    ready.insert(dependent.clone());
                }
            }
        }
        if visited != identities.len() {
            return Err(CacheError::InvalidManifest(
                "cache bundle dependency graph contains a cycle".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBundleVerificationMode {
    Transport,
    Full,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundlePolicy {
    pub schema_version: u32,
    pub maximum_manifest_bytes: u64,
    pub maximum_artifacts: usize,
    pub maximum_parts: usize,
    pub verification_mode: CacheBundleVerificationMode,
}

impl Default for CacheBundlePolicy {
    fn default() -> Self {
        Self {
            schema_version: 1,
            maximum_manifest_bytes: 64 * 1024 * 1024,
            maximum_artifacts: 100_000,
            maximum_parts: 1_000_000,
            verification_mode: CacheBundleVerificationMode::Full,
        }
    }
}

impl CacheBundlePolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.maximum_manifest_bytes == 0
            || self.maximum_artifacts == 0
            || self.maximum_parts == 0
        {
            return Err(CacheError::InvalidManifest(
                "cache bundle policy limits must be positive".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_manifest(&self, manifest: &CacheBundleManifest) -> Result<(), CacheError> {
        self.validate()?;
        manifest.validate()?;
        let part_count = manifest
            .artifacts
            .iter()
            .try_fold(0usize, |count, artifact| {
                count
                    .checked_add(artifact.encoding.ordered_parts.len())
                    .ok_or_else(|| {
                        CacheError::ResourceLimit(
                            "cache bundle part count exceeds this platform".to_owned(),
                        )
                    })
            })?;
        if manifest.artifacts.len() > self.maximum_artifacts || part_count > self.maximum_parts {
            return Err(CacheError::ResourceLimit(format!(
                "cache bundle contains {} artifacts and {part_count} part references above policy",
                manifest.artifacts.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleExportSource {
    pub artifact: CacheBundleArtifactRecord,
    pub parts_root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleExportRequest {
    pub schema_version: u32,
    pub roots: Vec<CacheBundleArtifactIdentity>,
    pub sources: Vec<CacheBundleExportSource>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleExportReport {
    pub schema_version: u32,
    pub bundle_digest: ContentDigest,
    pub destination: PathBuf,
    pub root_count: usize,
    pub artifact_count: usize,
    pub part_reference_count: usize,
    pub stored_part_count: usize,
    pub referenced_package_bytes: u64,
    pub projected_output_bytes: u64,
    pub hard_link_reuses: usize,
    pub verification: CacheBundleVerificationReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleVerificationReport {
    pub schema_version: u32,
    pub bundle_digest: ContentDigest,
    pub artifact_count: usize,
    pub root_count: usize,
    pub part_reference_count: usize,
    pub stored_part_count: usize,
    pub referenced_package_bytes: u64,
    pub stored_part_bytes: u64,
    pub decoded_artifact_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleConsumptionPolicy {
    pub schema_version: u32,
    pub reader_version: ToolkitVersion,
    pub minimum_assurance: ArtifactAssuranceState,
    pub allow_deprecated: bool,
}

impl CacheBundleConsumptionPolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0 {
            return Err(CacheError::InvalidManifest(
                "cache bundle consumption policy schema is required".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleMaterializationReport {
    pub schema_version: u32,
    pub bundle_digest: ContentDigest,
    pub selected: CacheBundleArtifactIdentity,
    pub dependency_artifact_count: usize,
    pub package: DeterministicPackageReport,
    pub decoded_verification: VerifiedPackageReport,
    pub bundle_verification: CacheBundleVerificationReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheBundleSemanticResolutionReport {
    pub schema_version: u32,
    pub semantic_digest: ContentDigest,
    pub selected: Option<CacheBundleArtifactRecord>,
    pub rejected_manifest_digests: BTreeMap<ContentDigest, String>,
    pub verification: CacheBundleVerificationReport,
}

pub struct CacheBundleSemanticResolutionRequest<'a> {
    pub bundle_root: &'a Path,
    pub scratch_root: &'a Path,
    pub family: &'a str,
    pub semantic_digest: &'a ContentDigest,
    pub bundle_policy: &'a CacheBundlePolicy,
    pub consumption_policy: &'a CacheBundleConsumptionPolicy,
    pub resources: &'a ResourcePolicy,
    pub cancellation: &'a CancellationToken,
}

#[derive(Clone)]
struct PlannedPart {
    part: TransportPart,
    source_path: PathBuf,
}

struct VerifiedBundle {
    manifest: CacheBundleManifest,
    report: CacheBundleVerificationReport,
}

pub fn export_cache_bundle(
    request: &CacheBundleExportRequest,
    destination: &Path,
    policy: &CacheBundlePolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<CacheBundleExportReport, CacheError> {
    check_cancelled(cancellation)?;
    if request.schema_version == 0
        || request.sources.is_empty()
        || destination.as_os_str().is_empty()
        || destination.exists()
    {
        return Err(CacheError::InvalidManifest(
            "cache bundle export requires sources and an absent explicit destination".to_owned(),
        ));
    }
    policy.validate()?;
    let mut artifacts = request
        .sources
        .iter()
        .map(|source| source.artifact.clone())
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| left.identity.cmp(&right.identity));
    let mut roots = request.roots.clone();
    roots.sort();
    let manifest = CacheBundleManifest {
        schema_version: 1,
        roots,
        artifacts,
    };
    policy.validate_manifest(&manifest)?;
    let manifest_bytes = canonical_json_bytes(&manifest)?;
    if manifest_bytes.len() as u64 > policy.maximum_manifest_bytes {
        return Err(CacheError::ResourceLimit(format!(
            "cache bundle manifest requires {} bytes above policy",
            manifest_bytes.len()
        )));
    }
    let planned_parts = plan_export_parts(request)?;
    let referenced_package_bytes =
        manifest
            .artifacts
            .iter()
            .try_fold(0u64, |total, artifact| {
                total
                    .checked_add(artifact.encoding.package_size_bytes)
                    .ok_or_else(|| {
                        CacheError::ResourceLimit(
                            "cache bundle referenced package bytes exceed u64".to_owned(),
                        )
                    })
            })?;
    let projected_output_bytes =
        planned_parts
            .values()
            .try_fold(manifest_bytes.len() as u64, |total, planned| {
                total.checked_add(planned.part.size_bytes).ok_or_else(|| {
                    CacheError::ResourceLimit("cache bundle output bytes exceed u64".to_owned())
                })
            })?;
    enforce_output_budget(projected_output_bytes, resources)?;

    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = create_staging_directory(destination)?;
    let verification_scratch = match create_staging_directory(&parent.join("bundle-verification")) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    };
    let result = (|| {
        let mut first_by_digest = BTreeMap::<ContentDigest, PathBuf>::new();
        let mut hard_link_reuses = 0usize;
        for (repository_path, planned) in &planned_parts {
            check_cancelled(cancellation)?;
            let output = resolve_bundle_part_path(&staging, repository_path)?;
            if let Some(existing) = first_by_digest.get(&planned.part.content_digest) {
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)?;
                }
                if fs::hard_link(existing, &output).is_ok() {
                    hard_link_reuses += 1;
                    continue;
                }
            }
            copy_verified_part(
                &planned.source_path,
                &output,
                &planned.part,
                resources,
                cancellation,
            )?;
            first_by_digest
                .entry(planned.part.content_digest.clone())
                .or_insert(output);
        }
        let bundle_path = staging.join(CACHE_BUNDLE_MANIFEST_PATH);
        write_new_file(&bundle_path, &manifest_bytes)?;
        let verification = verify_cache_bundle(
            &staging,
            &verification_scratch,
            policy,
            resources,
            cancellation,
        )?;
        let _ = fs::remove_dir_all(&verification_scratch);
        fs::rename(&staging, destination)?;
        Ok(CacheBundleExportReport {
            schema_version: 1,
            bundle_digest: verification.bundle_digest.clone(),
            destination: destination.to_owned(),
            root_count: manifest.roots.len(),
            artifact_count: manifest.artifacts.len(),
            part_reference_count: manifest
                .artifacts
                .iter()
                .map(|artifact| artifact.encoding.ordered_parts.len())
                .sum(),
            stored_part_count: planned_parts.len(),
            referenced_package_bytes,
            projected_output_bytes,
            hard_link_reuses,
            verification,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    let _ = fs::remove_dir_all(&verification_scratch);
    result
}

pub fn verify_cache_bundle(
    bundle_root: &Path,
    scratch_root: &Path,
    policy: &CacheBundlePolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<CacheBundleVerificationReport, CacheError> {
    Ok(load_and_verify_bundle(bundle_root, scratch_root, policy, resources, cancellation)?.report)
}

pub fn resolve_cache_bundle_semantic_artifact(
    request: CacheBundleSemanticResolutionRequest<'_>,
) -> Result<CacheBundleSemanticResolutionReport, CacheError> {
    if request.family.trim().is_empty() || !request.semantic_digest.validate() {
        return Err(CacheError::InvalidManifest(
            "bundle semantic query requires a family and digest".to_owned(),
        ));
    }
    request.consumption_policy.validate()?;
    let verified = load_and_verify_bundle(
        request.bundle_root,
        request.scratch_root,
        request.bundle_policy,
        request.resources,
        request.cancellation,
    )?;
    let artifacts = verified
        .manifest
        .artifacts
        .iter()
        .map(|artifact| (artifact.identity.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    let mut accepted = Vec::new();
    let mut rejected_manifest_digests = BTreeMap::new();
    for artifact in verified.manifest.artifacts.iter().filter(|artifact| {
        artifact.identity.artifact_family == request.family
            && artifact.identity.semantic_digest == *request.semantic_digest
    }) {
        match validate_consumable_closure(artifact, &artifacts, request.consumption_policy) {
            Ok(_) => accepted.push(artifact),
            Err(error) => {
                rejected_manifest_digests
                    .insert(artifact.identity.manifest_digest.clone(), error.to_string());
            }
        }
    }
    accepted.sort_by(|left, right| {
        right
            .achieved_assurance
            .cmp(&left.achieved_assurance)
            .then_with(|| {
                left.identity
                    .manifest_digest
                    .cmp(&right.identity.manifest_digest)
            })
    });
    Ok(CacheBundleSemanticResolutionReport {
        schema_version: 1,
        semantic_digest: request.semantic_digest.clone(),
        selected: accepted.first().map(|artifact| (*artifact).clone()),
        rejected_manifest_digests,
        verification: verified.report,
    })
}

pub fn materialize_cache_bundle_artifact(
    bundle_root: &Path,
    identity: &CacheBundleArtifactIdentity,
    destination: &Path,
    bundle_policy: &CacheBundlePolicy,
    consumption_policy: &CacheBundleConsumptionPolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<CacheBundleMaterializationReport, CacheError> {
    identity.validate()?;
    consumption_policy.validate()?;
    let mut transport_policy = bundle_policy.clone();
    transport_policy.verification_mode = CacheBundleVerificationMode::Transport;
    let verified = load_and_verify_bundle(
        bundle_root,
        bundle_root,
        &transport_policy,
        resources,
        cancellation,
    )?;
    let artifacts = verified
        .manifest
        .artifacts
        .iter()
        .map(|artifact| (artifact.identity.clone(), artifact))
        .collect::<BTreeMap<_, _>>();
    let selected = artifacts.get(identity).copied().ok_or_else(|| {
        CacheError::NotFound(format!(
            "bundle artifact {}:{}",
            identity.artifact_family, identity.semantic_digest
        ))
    })?;
    let dependency_artifact_count =
        validate_consumable_closure(selected, &artifacts, consumption_policy)?;
    let package = reconstruct_transport_package(
        &selected.encoding,
        &bundle_root.join(CACHE_BUNDLE_PARTS_DIRECTORY),
        destination,
        resources,
        cancellation,
    )?;
    let decoded_verification = match verify_canonical_payload_zip64(
        &selected.manifest.canonical_payload,
        &selected.encoding,
        destination,
        cancellation,
    ) {
        Ok(report) => report,
        Err(error) => {
            let _ = fs::remove_file(destination);
            return Err(error);
        }
    };
    Ok(CacheBundleMaterializationReport {
        schema_version: 1,
        bundle_digest: verified.report.bundle_digest.clone(),
        selected: identity.clone(),
        dependency_artifact_count,
        package,
        decoded_verification,
        bundle_verification: verified.report,
    })
}

fn load_and_verify_bundle(
    bundle_root: &Path,
    scratch_root: &Path,
    policy: &CacheBundlePolicy,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<VerifiedBundle, CacheError> {
    check_cancelled(cancellation)?;
    policy.validate()?;
    if bundle_root.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "cache bundle root must be explicit".to_owned(),
        ));
    }
    let bundle_path = bundle_root.join(CACHE_BUNDLE_MANIFEST_PATH);
    let metadata = fs::symlink_metadata(&bundle_path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > policy.maximum_manifest_bytes
        || resources
            .maximum_memory_bytes
            .is_some_and(|maximum| metadata.len() > maximum)
    {
        return Err(CacheError::ResourceLimit(
            "cache bundle manifest is unsafe or exceeds the metadata budget".to_owned(),
        ));
    }
    let bytes = fs::read(&bundle_path)?;
    let manifest: CacheBundleManifest = serde_json::from_slice(&bytes)?;
    policy.validate_manifest(&manifest)?;
    let canonical = canonical_json_bytes(&manifest)?;
    if canonical != bytes {
        return Err(CacheError::InvalidManifest(
            "cache bundle manifest is not in canonical JSON form".to_owned(),
        ));
    }
    let bundle_digest = ContentDigest::sha256(&canonical);
    let planned_parts = plan_manifest_parts(&manifest)?;
    let buffer_len = copy_buffer_len(resources)?;
    let mut buffer = vec![0u8; buffer_len];
    let mut stored_part_bytes = 0u64;
    for (repository_path, part) in &planned_parts {
        let path = resolve_bundle_part_path(bundle_root, repository_path)?;
        verify_part_file(&path, part, cancellation, &mut buffer)?;
        stored_part_bytes = stored_part_bytes
            .checked_add(part.size_bytes)
            .ok_or_else(|| CacheError::ResourceLimit("bundle bytes exceed u64".to_owned()))?;
    }
    for artifact in &manifest.artifacts {
        verify_package_digest(
            &bundle_root.join(CACHE_BUNDLE_PARTS_DIRECTORY),
            &artifact.encoding,
            cancellation,
            &mut buffer,
        )?;
    }
    let decoded_artifact_count = match policy.verification_mode {
        CacheBundleVerificationMode::Transport => 0,
        CacheBundleVerificationMode::Full => {
            verify_decoded_artifacts(
                bundle_root,
                scratch_root,
                &manifest,
                resources,
                cancellation,
            )?;
            manifest.artifacts.len()
        }
    };
    let referenced_package_bytes =
        manifest
            .artifacts
            .iter()
            .try_fold(0u64, |total, artifact| {
                total
                    .checked_add(artifact.encoding.package_size_bytes)
                    .ok_or_else(|| {
                        CacheError::ResourceLimit("bundle package bytes exceed u64".to_owned())
                    })
            })?;
    Ok(VerifiedBundle {
        report: CacheBundleVerificationReport {
            schema_version: 1,
            bundle_digest,
            artifact_count: manifest.artifacts.len(),
            root_count: manifest.roots.len(),
            part_reference_count: manifest
                .artifacts
                .iter()
                .map(|artifact| artifact.encoding.ordered_parts.len())
                .sum(),
            stored_part_count: planned_parts.len(),
            referenced_package_bytes,
            stored_part_bytes,
            decoded_artifact_count,
        },
        manifest,
    })
}

fn plan_export_parts(
    request: &CacheBundleExportRequest,
) -> Result<BTreeMap<String, PlannedPart>, CacheError> {
    let mut planned = BTreeMap::<String, PlannedPart>::new();
    for source in &request.sources {
        source.artifact.validate()?;
        if source.parts_root.as_os_str().is_empty() {
            return Err(CacheError::InvalidManifest(
                "cache bundle source parts root must be explicit".to_owned(),
            ));
        }
        for part in &source.artifact.encoding.ordered_parts {
            let source_path =
                resolve_transport_part_path(&source.parts_root, &part.repository_path)?;
            match planned.get(&part.repository_path) {
                Some(existing) if existing.part != *part => {
                    return Err(CacheError::InvalidManifest(format!(
                        "bundle part path {:?} has conflicting identities",
                        part.repository_path
                    )));
                }
                Some(_) => {}
                None => {
                    planned.insert(
                        part.repository_path.clone(),
                        PlannedPart {
                            part: part.clone(),
                            source_path,
                        },
                    );
                }
            }
        }
    }
    Ok(planned)
}

fn plan_manifest_parts(
    manifest: &CacheBundleManifest,
) -> Result<BTreeMap<String, TransportPart>, CacheError> {
    let mut planned = BTreeMap::<String, TransportPart>::new();
    for artifact in &manifest.artifacts {
        for part in &artifact.encoding.ordered_parts {
            match planned.get(&part.repository_path) {
                Some(existing) if existing != part => {
                    return Err(CacheError::InvalidManifest(format!(
                        "bundle part path {:?} has conflicting identities",
                        part.repository_path
                    )));
                }
                Some(_) => {}
                None => {
                    planned.insert(part.repository_path.clone(), part.clone());
                }
            }
        }
    }
    Ok(planned)
}

fn validate_consumable_closure(
    selected: &CacheBundleArtifactRecord,
    artifacts: &BTreeMap<CacheBundleArtifactIdentity, &CacheBundleArtifactRecord>,
    policy: &CacheBundleConsumptionPolicy,
) -> Result<usize, CacheError> {
    if selected.achieved_assurance < policy.minimum_assurance {
        return Err(CacheError::InvalidManifest(format!(
            "bundle artifact assurance {:?} is below {:?}",
            selected.achieved_assurance, policy.minimum_assurance
        )));
    }
    let mut pending = vec![selected.identity.clone()];
    let mut visited = BTreeSet::new();
    while let Some(identity) = pending.pop() {
        if !visited.insert(identity.clone()) {
            continue;
        }
        let artifact = artifacts.get(&identity).copied().ok_or_else(|| {
            CacheError::InvalidManifest("bundle dependency disappeared after validation".to_owned())
        })?;
        match artifact.disposition {
            ArtifactDisposition::Active => {}
            ArtifactDisposition::Deprecated if policy.allow_deprecated => {}
            ArtifactDisposition::Deprecated => {
                return Err(CacheError::InvalidManifest(
                    "deprecated bundle dependency is disabled by policy".to_owned(),
                ));
            }
            ArtifactDisposition::Quarantined | ArtifactDisposition::Revoked => {
                return Err(CacheError::InvalidManifest(
                    "quarantined or revoked bundle artifacts cannot be consumed".to_owned(),
                ));
            }
        }
        if policy.reader_version < artifact.manifest.minimum_reader_version
            || artifact
                .manifest
                .maximum_reader_version
                .as_ref()
                .is_some_and(|maximum| &policy.reader_version > maximum)
        {
            return Err(CacheError::InvalidManifest(
                "bundle artifact is incompatible with the requested reader version".to_owned(),
            ));
        }
        pending.extend(
            artifact
                .manifest
                .canonical_payload
                .dependencies
                .iter()
                .map(|dependency| CacheBundleArtifactIdentity {
                    artifact_family: dependency.artifact_family.clone(),
                    semantic_digest: dependency.semantic_digest.clone(),
                    manifest_digest: dependency.manifest_digest.clone(),
                    payload_digest: dependency.payload_digest.clone(),
                }),
        );
    }
    Ok(visited.len().saturating_sub(1))
}

fn verify_decoded_artifacts(
    bundle_root: &Path,
    scratch_root: &Path,
    manifest: &CacheBundleManifest,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    if scratch_root.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "full bundle verification requires an explicit scratch root".to_owned(),
        ));
    }
    let scratch_existed = scratch_root.exists();
    fs::create_dir_all(scratch_root)?;
    let bundle_root = fs::canonicalize(bundle_root)?;
    let scratch_root_canonical = fs::canonicalize(scratch_root)?;
    if scratch_root_canonical.starts_with(&bundle_root) {
        if !scratch_existed {
            let _ = fs::remove_dir(scratch_root);
        }
        return Err(CacheError::InvalidManifest(
            "full bundle verification scratch space must be outside the immutable bundle"
                .to_owned(),
        ));
    }
    for artifact in &manifest.artifacts {
        check_cancelled(cancellation)?;
        let package_path = unique_scratch_path(scratch_root, &artifact.identity.manifest_digest);
        let result = reconstruct_transport_package(
            &artifact.encoding,
            &bundle_root.join(CACHE_BUNDLE_PARTS_DIRECTORY),
            &package_path,
            resources,
            cancellation,
        )
        .and_then(|_| {
            verify_canonical_payload_zip64(
                &artifact.manifest.canonical_payload,
                &artifact.encoding,
                &package_path,
                cancellation,
            )
        });
        let _ = fs::remove_file(&package_path);
        result?;
    }
    Ok(())
}

fn verify_package_digest(
    parts_root: &Path,
    encoding: &TransportEncodingRecord,
    cancellation: &CancellationToken,
    buffer: &mut [u8],
) -> Result<(), CacheError> {
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    for part in &encoding.ordered_parts {
        check_cancelled(cancellation)?;
        let path = resolve_transport_part_path(parts_root, &part.repository_path)?;
        let mut input = BufReader::new(File::open(path)?);
        loop {
            check_cancelled(cancellation)?;
            let read = input.read(buffer)?;
            if read == 0 {
                break;
            }
            size = size.checked_add(read as u64).ok_or_else(|| {
                CacheError::ResourceLimit("bundle package size exceeds u64".to_owned())
            })?;
            hasher.update(&buffer[..read]);
        }
    }
    let digest = ContentDigest(format!("{:x}", hasher.finalize()));
    if size != encoding.package_size_bytes || digest != encoding.package_digest {
        return Err(CacheError::DigestMismatch {
            expected: format!(
                "{} ({} bytes)",
                encoding.package_digest, encoding.package_size_bytes
            ),
            actual: format!("{digest} ({size} bytes)"),
        });
    }
    Ok(())
}

fn copy_verified_part(
    source: &Path,
    destination: &Path,
    part: &TransportPart,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    let metadata = fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != part.size_bytes
    {
        return Err(CacheError::InvalidManifest(format!(
            "bundle source part {:?} is not the recorded regular file",
            part.repository_path
        )));
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let input = File::open(source)?;
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut input = BufReader::new(input);
    let mut output = BufWriter::new(output);
    let mut buffer = vec![0u8; copy_buffer_len(resources)?];
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    loop {
        check_cancelled(cancellation)?;
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        hasher.update(&buffer[..read]);
        size = size
            .checked_add(read as u64)
            .ok_or_else(|| CacheError::ResourceLimit("bundle part size exceeds u64".to_owned()))?;
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    let digest = ContentDigest(format!("{:x}", hasher.finalize()));
    if size != part.size_bytes || digest != part.content_digest {
        drop(output);
        let _ = fs::remove_file(destination);
        return Err(CacheError::DigestMismatch {
            expected: format!("{} ({} bytes)", part.content_digest, part.size_bytes),
            actual: format!("{digest} ({size} bytes)"),
        });
    }
    Ok(())
}

fn verify_part_file(
    path: &Path,
    part: &TransportPart,
    cancellation: &CancellationToken,
    buffer: &mut [u8],
) -> Result<(), CacheError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != part.size_bytes
    {
        return Err(CacheError::InvalidManifest(format!(
            "bundle part {:?} is not the recorded regular file",
            part.repository_path
        )));
    }
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    loop {
        check_cancelled(cancellation)?;
        let read = input.read(buffer)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .ok_or_else(|| CacheError::ResourceLimit("bundle part size exceeds u64".to_owned()))?;
        hasher.update(&buffer[..read]);
    }
    let digest = ContentDigest(format!("{:x}", hasher.finalize()));
    if size != part.size_bytes || digest != part.content_digest {
        return Err(CacheError::DigestMismatch {
            expected: format!("{} ({} bytes)", part.content_digest, part.size_bytes),
            actual: format!("{digest} ({size} bytes)"),
        });
    }
    Ok(())
}

fn enforce_output_budget(bytes: u64, resources: &ResourcePolicy) -> Result<(), CacheError> {
    for (name, maximum) in [
        ("permanent-disk", resources.maximum_permanent_disk_bytes),
        ("transfer", resources.maximum_transfer_bytes),
    ] {
        if maximum.is_some_and(|maximum| bytes > maximum) {
            return Err(CacheError::ResourceLimit(format!(
                "cache bundle projects {bytes} output bytes above the {name} budget"
            )));
        }
    }
    Ok(())
}

fn resolve_bundle_part_path(root: &Path, repository_path: &str) -> Result<PathBuf, CacheError> {
    resolve_transport_part_path(&root.join(CACHE_BUNDLE_PARTS_DIRECTORY), repository_path)
}

fn resolve_transport_part_path(root: &Path, repository_path: &str) -> Result<PathBuf, CacheError> {
    if !normalized_relative_path(repository_path) {
        return Err(CacheError::InvalidManifest(format!(
            "cache bundle part path {repository_path:?} is unsafe"
        )));
    }
    Ok(repository_path
        .split('/')
        .fold(root.to_owned(), |path, component| path.join(component)))
}

fn copy_buffer_len(resources: &ResourcePolicy) -> Result<usize, CacheError> {
    let bytes = resources
        .maximum_memory_bytes
        .unwrap_or(COPY_BUFFER_BYTES)
        .clamp(1, COPY_BUFFER_BYTES);
    usize::try_from(bytes).map_err(|_| {
        CacheError::ResourceLimit("cache bundle buffer does not fit this platform".to_owned())
    })
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), CacheError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn create_staging_directory(destination: &Path) -> Result<PathBuf, CacheError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("bundle");
    for _ in 0..1024 {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{name}.xc-bundle-{}-{sequence}.tmp",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(CacheError::Io(
        "could not allocate a unique cache bundle staging directory".to_owned(),
    ))
}

fn unique_scratch_path(root: &Path, digest: &ContentDigest) -> PathBuf {
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    root.join(format!(
        ".xc-bundle-verify-{}-{}-{sequence}.zip",
        &digest.0[..12],
        std::process::id()
    ))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        package_canonical_payload_zip64, resolve_semantic_artifact, stream_split_encoded,
        CanonicalPayloadEnvelope, LogicalPayloadItem, PayloadDependencyIdentity, PayloadFileSource,
        RemoteSemanticQuery, SemanticArtifactOverlayClass, SemanticArtifactSource,
        SemanticArtifactSourceKind, SemanticKeyEnvelope, TransportPolicy,
        CURRENT_DETERMINISTIC_ZIP64_PROFILE,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use xc_core::AssuranceLevel;

    fn temporary_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("xc-cache-bundle-{name}-{}", std::process::id()))
    }

    fn fixture(
        root: &Path,
        name: &str,
        dependency: Option<PayloadDependencyIdentity>,
    ) -> CacheBundleExportSource {
        let artifact_root = root.join(name);
        fs::create_dir_all(&artifact_root).unwrap();
        let logical_bytes = format!("offline payload for {name}").into_bytes();
        let source_path = artifact_root.join("source.bin");
        fs::write(&source_path, &logical_bytes).unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "bytes".to_owned(),
            dimensions: vec![logical_bytes.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(&logical_bytes),
                size_bytes: logical_bytes.len() as u64,
            }],
            dependencies: dependency.into_iter().collect(),
        };
        let package_path = artifact_root.join("package.zip");
        let package = package_canonical_payload_zip64(
            &canonical_payload,
            &[PayloadFileSource {
                normalized_path: "value.bin".to_owned(),
                source_path,
            }],
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(package.encoder_profile, CURRENT_DETERMINISTIC_ZIP64_PROFILE);
        let parts_root = artifact_root.join("parts");
        let mut input = File::open(&package_path).unwrap();
        let encoding = stream_split_encoded(
            &mut input,
            canonical_payload.digest().unwrap(),
            package.encoder_profile,
            &TransportPolicy {
                maximum_file_bytes_exclusive: 1024,
                split_part_bytes: 64,
                maximum_batch_payload_bytes: 1024,
                maximum_pending_batches: 1,
            },
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            |part, bytes| {
                let path = resolve_transport_part_path(&parts_root, &part.repository_path)?;
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(path, bytes)?;
                Ok(())
            },
        )
        .unwrap();
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: name.to_owned(),
            mathematical_semantics_version: "1".to_owned(),
            resolved_mathematical_parameters: json!({"name": name}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let transport_digest = encoding.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "fixture".to_owned(),
            semantic_digest: semantic_key.digest().unwrap(),
            semantic_key,
            payload_digest: canonical_payload.digest().unwrap(),
            canonical_payload,
            transport_digests: vec![transport_digest],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "offline fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let identity = CacheBundleArtifactIdentity::from_manifest(&manifest).unwrap();
        CacheBundleExportSource {
            artifact: CacheBundleArtifactRecord {
                identity,
                achieved_assurance: ArtifactAssuranceState::Computed,
                disposition: ArtifactDisposition::Active,
                manifest,
                encoding,
                origin_receipt: None,
            },
            parts_root,
        }
    }

    #[test]
    fn bundle_exports_verifies_and_materializes_without_network() {
        let root = temporary_root("round-trip");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let dependency_source = fixture(&root, "dependency", None);
        let dependency = PayloadDependencyIdentity {
            artifact_family: dependency_source.artifact.identity.artifact_family.clone(),
            semantic_digest: dependency_source.artifact.identity.semantic_digest.clone(),
            manifest_digest: dependency_source.artifact.identity.manifest_digest.clone(),
            payload_digest: dependency_source.artifact.identity.payload_digest.clone(),
        };
        let source = fixture(&root, "root", Some(dependency));
        let identity = source.artifact.identity.clone();
        let semantic_key = source.artifact.manifest.semantic_key.clone();
        let bundle = root.join("export.bundle");
        let policy = CacheBundlePolicy {
            maximum_manifest_bytes: 1024 * 1024,
            maximum_artifacts: 10,
            maximum_parts: 100,
            ..CacheBundlePolicy::default()
        };
        let export = export_cache_bundle(
            &CacheBundleExportRequest {
                schema_version: 1,
                roots: vec![identity.clone()],
                sources: vec![source, dependency_source],
            },
            &bundle,
            &policy,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(export.verification.decoded_artifact_count, 2);
        let query = RemoteSemanticQuery {
            family: identity.artifact_family.clone(),
            semantic_key,
            minimum_assurance: ArtifactAssuranceState::Computed,
            allowed_scalar_backends: BTreeSet::new(),
            minimum_precision_bits: None,
            required_configuration_digest: None,
            required_provenance_evidence_digests: BTreeSet::new(),
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            accepted_publication_policy_digests: [ContentDigest::sha256(b"policy")]
                .into_iter()
                .collect(),
            allow_deprecated: false,
            evaluation_unix_seconds: 1,
            maximum_topology_bytes: 1,
            maximum_index_bytes: 1,
            maximum_manifest_bytes: 1,
            maximum_encoding_bytes: 1,
            maximum_receipt_bytes: 1,
            maximum_revocation_partition_bytes: 1,
            maximum_dependency_depth: 8,
            maximum_dependency_count: 10,
        };
        let local_scratch = root.join("ordered-local-scratch");
        let export_scratch = root.join("ordered-export-scratch");
        let ordered = resolve_semantic_artifact(
            None,
            &query,
            &[
                SemanticArtifactSource::LocalFilesystem {
                    name: "workstation-first",
                    overlay_class: SemanticArtifactOverlayClass::WorkstationLocal,
                    root: &bundle,
                    scratch_root: &local_scratch,
                    policy: &policy,
                },
                SemanticArtifactSource::ExportBundle {
                    name: "archive-second",
                    overlay_class: SemanticArtifactOverlayClass::OptionalRemote,
                    root: &bundle,
                    scratch_root: &export_scratch,
                    policy: &policy,
                },
            ],
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(
            ordered.ordered_sources,
            vec!["workstation-first", "archive-second"]
        );
        assert_eq!(
            ordered.selected.unwrap().source_kind,
            SemanticArtifactSourceKind::LocalFilesystem
        );
        let overlay_classes = [
            SemanticArtifactOverlayClass::ProcessLocal,
            SemanticArtifactOverlayClass::WorkstationLocal,
            SemanticArtifactOverlayClass::ProjectPrivate,
            SemanticArtifactOverlayClass::TeamPrivate,
            SemanticArtifactOverlayClass::PublicPublished,
            SemanticArtifactOverlayClass::OptionalRemote,
        ];
        let overlay_names = [
            "process",
            "workstation",
            "project-private",
            "team-private",
            "public-published",
            "optional-remote",
        ];
        let overlay_roots = (0..overlay_classes.len())
            .map(|index| {
                if index + 1 == overlay_classes.len() {
                    bundle.clone()
                } else {
                    root.join(format!("unpopulated-overlay-{index}"))
                }
            })
            .collect::<Vec<_>>();
        let overlay_scratch = (0..overlay_classes.len())
            .map(|index| root.join(format!("six-overlay-scratch-{index}")))
            .collect::<Vec<_>>();
        let six_sources = (0..overlay_classes.len())
            .map(|index| SemanticArtifactSource::LocalFilesystem {
                name: overlay_names[index],
                overlay_class: overlay_classes[index],
                root: &overlay_roots[index],
                scratch_root: &overlay_scratch[index],
                policy: &policy,
            })
            .collect::<Vec<_>>();
        let six_overlay_resolution = resolve_semantic_artifact(
            None,
            &query,
            &six_sources,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(six_overlay_resolution.ordered_sources, overlay_names);
        assert_eq!(
            six_overlay_resolution.ordered_overlay_classes,
            overlay_classes
        );
        assert_eq!(six_overlay_resolution.rejections.len(), 5);
        let six_overlay_selection = six_overlay_resolution.selected.unwrap();
        assert_eq!(
            six_overlay_selection.overlay_class,
            SemanticArtifactOverlayClass::OptionalRemote
        );
        assert_eq!(
            six_overlay_selection.semantic_digest,
            identity.semantic_digest
        );
        for (name, source_kind) in [
            ("workstation", SemanticArtifactSourceKind::LocalFilesystem),
            ("archive", SemanticArtifactSourceKind::ExportBundle),
        ] {
            let scratch = root.join(format!("{name}-semantic-scratch"));
            let source = match source_kind {
                SemanticArtifactSourceKind::LocalFilesystem => {
                    SemanticArtifactSource::LocalFilesystem {
                        name,
                        overlay_class: SemanticArtifactOverlayClass::WorkstationLocal,
                        root: &bundle,
                        scratch_root: &scratch,
                        policy: &policy,
                    }
                }
                SemanticArtifactSourceKind::ExportBundle => SemanticArtifactSource::ExportBundle {
                    name,
                    overlay_class: SemanticArtifactOverlayClass::OptionalRemote,
                    root: &bundle,
                    scratch_root: &scratch,
                    policy: &policy,
                },
                _ => unreachable!(),
            };
            let semantic = resolve_semantic_artifact(
                None,
                &query,
                &[source],
                &ResourcePolicy::default(),
                &CancellationToken::new(),
            )
            .unwrap();
            let selected = semantic.selected.unwrap();
            assert_eq!(selected.source_kind, source_kind);
            assert_eq!(selected.semantic_digest, identity.semantic_digest);
            assert_eq!(selected.manifest_digest, identity.manifest_digest);
        }
        let verification = verify_cache_bundle(
            &bundle,
            &root.join("verification-scratch"),
            &policy,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(verification.bundle_digest, export.bundle_digest);
        let materialized = materialize_cache_bundle_artifact(
            &bundle,
            &identity,
            &root.join("offline.zip"),
            &policy,
            &CacheBundleConsumptionPolicy {
                schema_version: 1,
                reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
                minimum_assurance: ArtifactAssuranceState::Computed,
                allow_deprecated: false,
            },
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(materialized.selected, identity);
        assert_eq!(materialized.dependency_artifact_count, 1);
        assert_eq!(materialized.decoded_verification.item_count, 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundle_rejects_missing_dependency_before_visibility() {
        let root = temporary_root("missing-dependency");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let dependency = PayloadDependencyIdentity {
            artifact_family: "fixture".to_owned(),
            semantic_digest: ContentDigest::sha256(b"missing-semantic"),
            manifest_digest: ContentDigest::sha256(b"missing-manifest"),
            payload_digest: ContentDigest::sha256(b"missing-payload"),
        };
        let source = fixture(&root, "derived", Some(dependency));
        let destination = root.join("rejected.bundle");
        let result = export_cache_bundle(
            &CacheBundleExportRequest {
                schema_version: 1,
                roots: vec![source.artifact.identity.clone()],
                sources: vec![source],
            },
            &destination,
            &CacheBundlePolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(CacheError::InvalidManifest(_))));
        assert!(!destination.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundle_verification_detects_corrupt_exported_part() {
        let root = temporary_root("corrupt");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source = fixture(&root, "root", None);
        let identity = source.artifact.identity.clone();
        let first_part = source.artifact.encoding.ordered_parts[0].clone();
        let bundle = root.join("export.bundle");
        let policy = CacheBundlePolicy {
            verification_mode: CacheBundleVerificationMode::Transport,
            ..CacheBundlePolicy::default()
        };
        export_cache_bundle(
            &CacheBundleExportRequest {
                schema_version: 1,
                roots: vec![identity],
                sources: vec![source],
            },
            &bundle,
            &policy,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let corrupt = resolve_bundle_part_path(&bundle, &first_part.repository_path).unwrap();
        fs::write(&corrupt, vec![0u8; first_part.size_bytes as usize]).unwrap();
        let result = verify_cache_bundle(
            &bundle,
            &root.join("scratch"),
            &policy,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(CacheError::DigestMismatch { .. })));
        fs::remove_dir_all(root).unwrap();
    }
}
