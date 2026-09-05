// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Granular content-addressed cache fabric.
//!
//! Cache reuse is foundational plumbing, not a late performance feature. The
//! types in this crate support ordered local/private/public overlays, immutable
//! content objects, quality gates, compatibility policy, dependency records,
//! chunked payloads, and deterministic GitHub repository capacity planning.

mod artifact_validation;
mod batch_publication;
mod bootstrap_remote_store;
mod bootstrap_topology;
mod bundle;
mod cache_provenance;
mod compatibility;
mod coordinator;
mod cost_governance;
mod dedup_governance;
mod durability;
mod execution_cache;
mod finalizer;
mod git_transport;
mod github_auth;
mod governance;
mod governance_records;
mod large_corpus_acceptance;
mod live_github_acceptance;
mod managed_publication;
mod materialization;
mod output_validation;
mod packaging;
mod planner;
mod private_coordination;
mod production_staging;
mod protocol;
mod publication;
mod publication_orchestrator;
mod publication_recovery;
mod publication_staging;
mod publisher;
mod registry;
mod remote_reader;
mod rollover;
mod semantic_api;
mod semantic_resolver;
mod shard_audit;
mod shard_repair;
mod trust;

pub use artifact_validation::*;
pub use batch_publication::*;
pub use bootstrap_remote_store::*;
pub use bundle::*;
pub use cache_provenance::*;
pub use compatibility::*;
pub use coordinator::*;
pub use cost_governance::*;
pub use dedup_governance::*;
pub use durability::*;
pub use execution_cache::*;
pub use finalizer::*;
pub use git_transport::*;
pub use github_auth::*;
pub use governance::*;
pub use governance_records::*;
pub use large_corpus_acceptance::*;
pub use live_github_acceptance::*;
pub use managed_publication::*;
pub use materialization::*;
pub use output_validation::*;
pub use packaging::*;
pub use planner::*;
pub use private_coordination::*;
pub use production_staging::*;
pub use protocol::*;
pub use publication::*;
pub use publication_orchestrator::*;
pub use publication_recovery::*;
pub use publication_staging::*;
pub use publisher::*;
pub use registry::*;
pub use remote_reader::*;
pub use rollover::*;
pub use semantic_api::*;
pub use semantic_resolver::*;
pub use shard_audit::*;
pub use shard_repair::*;
pub use trust::*;

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Approved safe reachable-payload threshold for a GitHub cache repository.
///
/// Repository history remains part of capacity planning. The shard planner
/// therefore combines reachable payload and estimated history before selecting
/// a write destination.
pub const GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES: u64 = 100_000_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CacheError {
    Io(String),
    Serialization(String),
    InvalidManifest(String),
    DigestMismatch { expected: String, actual: String },
    NotFound(String),
    ReadOnlyLayer(String),
    NoWritableShard(String),
    InvalidTransition(String),
    ResourceLimit(String),
    Cancelled(String),
    Authentication(String),
    PermissionDenied(String),
}

impl Display for CacheError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(f, "cache I/O error: {message}"),
            Self::Serialization(message) => write!(f, "cache serialization error: {message}"),
            Self::InvalidManifest(message) => write!(f, "invalid cache manifest: {message}"),
            Self::DigestMismatch { expected, actual } => {
                write!(
                    f,
                    "cache digest mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NotFound(message) => write!(f, "cache artifact not found: {message}"),
            Self::ReadOnlyLayer(layer) => write!(f, "cache layer {layer:?} is read-only"),
            Self::NoWritableShard(message) => write!(f, "no writable cache shard: {message}"),
            Self::InvalidTransition(message) => {
                write!(f, "invalid cache state transition: {message}")
            }
            Self::ResourceLimit(message) => write!(f, "cache resource limit reached: {message}"),
            Self::Cancelled(message) => write!(f, "cache operation cancelled: {message}"),
            Self::Authentication(message) => {
                write!(f, "cache authentication failed: {message}")
            }
            Self::PermissionDenied(message) => {
                write!(f, "cache publication permission denied: {message}")
            }
        }
    }
}

impl Error for CacheError {}

impl From<std::io::Error> for CacheError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for CacheError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ContentDigest(pub String);

impl ContentDigest {
    pub fn sha256(payload: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        Self(hex_digest(hasher.finalize().as_slice()))
    }

    pub fn sha256_chunks<'a, I>(chunks: I) -> Self
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = Sha256::new();
        for chunk in chunks {
            hasher.update(chunk);
        }
        Self(hex_digest(hasher.finalize().as_slice()))
    }

    pub fn validate(&self) -> bool {
        self.0.len() == 64
            && self
                .0
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }
}

impl Display for ContentDigest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolkitVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
}

impl ToolkitVersion {
    pub fn parse(value: &str) -> Result<Self, CacheError> {
        let (core, prerelease) = match value.split_once('-') {
            Some((core, prerelease)) if !prerelease.is_empty() => {
                (core, Some(prerelease.to_owned()))
            }
            Some(_) => {
                return Err(CacheError::InvalidManifest(format!(
                    "invalid toolkit version {value:?}"
                )))
            }
            None => (value, None),
        };
        let parts: Vec<_> = core.split('.').collect();
        if parts.len() != 3 {
            return Err(CacheError::InvalidManifest(format!(
                "toolkit version must be major.minor.patch: {value:?}"
            )));
        }
        let parse = |part: &str| {
            part.parse::<u64>().map_err(|_| {
                CacheError::InvalidManifest(format!("invalid toolkit version {value:?}"))
            })
        };
        Ok(Self {
            major: parse(parts[0])?,
            minor: parse(parts[1])?,
            patch: parse(parts[2])?,
            prerelease,
        })
    }
}

impl Display for ToolkitVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(prerelease) = &self.prerelease {
            write!(f, "-{prerelease}")?;
        }
        Ok(())
    }
}

impl Ord for ToolkitVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (&self.prerelease, &other.prerelease) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(left), Some(right)) => left.cmp(right),
            })
    }
}

impl PartialOrd for ToolkitVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheQuality {
    Quarantined,
    Staged,
    Validated,
    CrossChecked,
    Certified,
    Published,
    Deprecated,
}

impl CacheQuality {
    pub fn admissible_rank(self) -> u8 {
        match self {
            Self::Quarantined | Self::Deprecated => 0,
            Self::Staged => 1,
            Self::Validated => 2,
            Self::CrossChecked => 3,
            Self::Certified => 4,
            Self::Published => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheVisibility {
    Local,
    Private,
    Team,
    Public,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ArtifactKey {
    pub kind: String,
    pub logical_key: String,
    pub parameters_digest: ContentDigest,
}

impl ArtifactKey {
    pub fn new(
        kind: impl Into<String>,
        logical_key: impl Into<String>,
        canonical_parameters: &[u8],
    ) -> Result<Self, CacheError> {
        let kind = kind.into();
        let logical_key = logical_key.into();
        if kind.trim().is_empty() || logical_key.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "artifact kind and logical key must be nonempty".to_owned(),
            ));
        }
        Ok(Self {
            kind,
            logical_key,
            parameters_digest: ContentDigest::sha256(canonical_parameters),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DependencyRef {
    pub key: ArtifactKey,
    pub content_digest: ContentDigest,
    pub required_quality: CacheQuality,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheObjectRef {
    pub content_digest: ContentDigest,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub key: ArtifactKey,
    pub content_digest: ContentDigest,
    pub size_bytes: u64,
    /// Ordered immutable objects whose concatenation is the logical payload.
    pub objects: Vec<CacheObjectRef>,
    pub created_unix_seconds: u64,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub maximum_reader_version: Option<ToolkitVersion>,
    pub quality: CacheQuality,
    pub visibility: CacheVisibility,
    pub immutable: bool,
    pub dependencies: Vec<DependencyRef>,
    pub tags: BTreeMap<String, String>,
    pub provenance_digest: Option<ContentDigest>,
}

impl ArtifactManifest {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1 {
            return Err(CacheError::InvalidManifest(
                "only manifest schema_version 1 is supported".to_owned(),
            ));
        }
        if !self.content_digest.validate() || !self.key.parameters_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "manifest contains an invalid SHA-256 digest".to_owned(),
            ));
        }
        let mut previous_dependency: Option<(&str, &str, &ContentDigest, &ContentDigest)> = None;
        for dependency in &self.dependencies {
            if !dependency.content_digest.validate() || !dependency.key.parameters_digest.validate()
            {
                return Err(CacheError::InvalidManifest(
                    "manifest contains an invalid dependency digest".to_owned(),
                ));
            }
            let identity = (
                dependency.key.kind.as_str(),
                dependency.key.logical_key.as_str(),
                &dependency.key.parameters_digest,
                &dependency.content_digest,
            );
            if previous_dependency.is_some_and(|previous| previous >= identity) {
                return Err(CacheError::InvalidManifest(
                    "manifest dependencies must be unique and canonically ordered".to_owned(),
                ));
            }
            previous_dependency = Some(identity);
        }
        if let Some(provenance) = &self.provenance_digest {
            if !provenance.validate() {
                return Err(CacheError::InvalidManifest(
                    "manifest contains an invalid provenance digest".to_owned(),
                ));
            }
        }
        if self.key.kind.trim().is_empty() || self.key.logical_key.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "artifact key fields must be nonempty".to_owned(),
            ));
        }
        if self.objects.is_empty() {
            return Err(CacheError::InvalidManifest(
                "manifest must explicitly list its immutable objects".to_owned(),
            ));
        } else {
            let mut total = 0u64;
            for object in &self.objects {
                if !object.content_digest.validate() {
                    return Err(CacheError::InvalidManifest(
                        "manifest contains an invalid object digest".to_owned(),
                    ));
                }
                total = total.checked_add(object.size_bytes).ok_or_else(|| {
                    CacheError::InvalidManifest("object sizes overflow u64".to_owned())
                })?;
            }
            let zip_json = self
                .tags
                .get("xc_storage_encoding")
                .is_some_and(|value| value == "zip-json-entry-v1");
            if !zip_json && total != self.size_bytes {
                return Err(CacheError::InvalidManifest(format!(
                    "object sizes total {total}, expected {}",
                    self.size_bytes
                )));
            }
        }
        if let Some(maximum) = &self.maximum_reader_version {
            if maximum < &self.minimum_reader_version {
                return Err(CacheError::InvalidManifest(
                    "maximum_reader_version precedes minimum_reader_version".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactDraft {
    pub schema_version: u32,
    pub key: ArtifactKey,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub maximum_reader_version: Option<ToolkitVersion>,
    pub quality: CacheQuality,
    pub visibility: CacheVisibility,
    pub immutable: bool,
    pub dependencies: Vec<DependencyRef>,
    pub tags: BTreeMap<String, String>,
    pub provenance_digest: Option<ContentDigest>,
}

#[derive(Clone, Debug)]
pub struct CachePolicy {
    pub current_toolkit_version: ToolkitVersion,
    pub minimum_quality: CacheQuality,
    pub accepted_schema_versions: Vec<u32>,
    pub allow_deprecated: bool,
    pub allow_quarantined: bool,
    pub allowed_visibilities: Vec<CacheVisibility>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheAcceptanceDecision {
    pub accepted: bool,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
}

impl CachePolicy {
    pub fn assess(&self, manifest: &ArtifactManifest) -> CacheAcceptanceDecision {
        let mut reasons = Vec::new();
        let mut warnings = Vec::new();
        if let Err(error) = manifest.validate() {
            reasons.push(format!("manifest validation failed: {error}"));
        }
        if !self
            .accepted_schema_versions
            .contains(&manifest.schema_version)
        {
            reasons.push(format!(
                "schema version {} is not accepted",
                manifest.schema_version
            ));
        }
        if self.current_toolkit_version < manifest.minimum_reader_version {
            reasons.push(format!(
                "current toolkit {:?} precedes minimum reader {:?}",
                self.current_toolkit_version, manifest.minimum_reader_version
            ));
        }
        if manifest
            .maximum_reader_version
            .as_ref()
            .is_some_and(|maximum| &self.current_toolkit_version > maximum)
        {
            reasons.push(format!(
                "current toolkit {:?} exceeds maximum reader {:?}",
                self.current_toolkit_version, manifest.maximum_reader_version
            ));
        }
        if !self.allowed_visibilities.contains(&manifest.visibility) {
            reasons.push(format!(
                "visibility {:?} is not allowed by this resolver",
                manifest.visibility
            ));
        }
        match manifest.quality {
            CacheQuality::Deprecated if !self.allow_deprecated => {
                reasons.push("deprecated artifacts are disabled".to_owned())
            }
            CacheQuality::Quarantined if !self.allow_quarantined => {
                reasons.push("quarantined artifacts are disabled".to_owned())
            }
            CacheQuality::Deprecated => warnings.push(
                "deprecated artifact accepted only because policy explicitly allows it".to_owned(),
            ),
            CacheQuality::Quarantined => warnings.push(
                "quarantined artifact accepted only because policy explicitly allows it".to_owned(),
            ),
            _ => {}
        }
        if manifest.quality.admissible_rank() < self.minimum_quality.admissible_rank() {
            reasons.push(format!(
                "artifact quality {:?} is below required {:?}",
                manifest.quality, self.minimum_quality
            ));
        }
        // Producer age is not judged here against the running toolkit's
        // release line. The explicit compatibility floors are the authority:
        // canonical manifest validation enforces the per-family
        // `minimum_producer_version`, and the reader-range checks above
        // enforce the artifact's own declared reader window. This matches the
        // original floor-based contract (a 0.12.0 artifact remained valid
        // under every later toolkit until a floor was deliberately raised).
        CacheAcceptanceDecision {
            accepted: reasons.is_empty(),
            reasons,
            warnings,
        }
    }

    pub fn accepts(&self, manifest: &ArtifactManifest) -> bool {
        self.assess(manifest).accepted
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromotionReview {
    pub reviewer: String,
    pub approved: bool,
    pub evidence_digest: Option<ContentDigest>,
    pub notes: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CachePromotionRequest {
    pub source_manifest_digest: ContentDigest,
    pub source_quality: CacheQuality,
    pub target_quality: CacheQuality,
    pub source_visibility: CacheVisibility,
    pub target_visibility: CacheVisibility,
    pub reviews: Vec<PromotionReview>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CachePromotionPolicy {
    pub minimum_unique_approvals: usize,
    pub require_certified_before_publication: bool,
    pub allow_private_to_public: bool,
}

impl CachePromotionPolicy {
    pub fn validate_request(&self, request: &CachePromotionRequest) -> Result<(), CacheError> {
        if !request.source_manifest_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "promotion source manifest digest is invalid".to_owned(),
            ));
        }
        if matches!(
            request.target_quality,
            CacheQuality::Quarantined | CacheQuality::Deprecated
        ) {
            return Err(CacheError::InvalidManifest(
                "quarantine and deprecation are governance actions, not promotions".to_owned(),
            ));
        }
        if request.target_quality.admissible_rank() < request.source_quality.admissible_rank() {
            return Err(CacheError::InvalidManifest(
                "promotion target quality is below source quality".to_owned(),
            ));
        }
        if request.source_visibility != CacheVisibility::Public
            && request.target_visibility == CacheVisibility::Public
            && !self.allow_private_to_public
        {
            return Err(CacheError::InvalidManifest(
                "policy does not allow private-to-public promotion".to_owned(),
            ));
        }
        if self.require_certified_before_publication
            && request.target_quality == CacheQuality::Published
            && request.source_quality.admissible_rank() < CacheQuality::Certified.admissible_rank()
        {
            return Err(CacheError::InvalidManifest(
                "publication policy requires a certified source artifact".to_owned(),
            ));
        }
        let approvals: BTreeSet<&str> = request
            .reviews
            .iter()
            .filter(|review| review.approved && !review.reviewer.trim().is_empty())
            .map(|review| review.reviewer.as_str())
            .collect();
        if approvals.len() < self.minimum_unique_approvals {
            return Err(CacheError::InvalidManifest(format!(
                "promotion has {} unique approvals, requires {}",
                approvals.len(),
                self.minimum_unique_approvals
            )));
        }
        for review in &request.reviews {
            if let Some(digest) = &review.evidence_digest {
                if !digest.validate() {
                    return Err(CacheError::InvalidManifest(
                        "promotion review contains an invalid evidence digest".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestIndex {
    schema_version: u32,
    manifests: Vec<String>,
}

impl Default for ManifestIndex {
    fn default() -> Self {
        Self {
            schema_version: 1,
            manifests: Vec::new(),
        }
    }
}

pub trait CacheStore: Send + Sync {
    fn name(&self) -> &str;
    fn writable(&self) -> bool;
    fn visibility(&self) -> CacheVisibility;
    fn put(&self, draft: &ArtifactDraft, payload: &[u8]) -> Result<ArtifactManifest, CacheError>;
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError>;
    /// Enumerate artifact keys whose kind and logical key match a bounded
    /// semantic prefix. Implementations may return an empty set when their
    /// backing format predates searchable logical-key metadata.
    fn matching_keys(
        &self,
        _kind: &str,
        _logical_key_prefix: &str,
        _maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        Ok(Vec::new())
    }
    /// Return manifests matching an exact published dependency identity
    /// (family, semantic digest, canonical manifest digest, payload digest),
    /// regardless of logical key.
    ///
    /// Publication closures reference dependencies by identity, and a
    /// dependency reused from a published shard carries no key-based
    /// dependency list of its own, so staging its transitive closure requires
    /// looking artifacts up by identity rather than by key. Implementations
    /// that cannot search by identity return an empty set and the resolver
    /// consults other layers.
    fn identity_candidates(
        &self,
        _identity: &crate::PayloadDependencyIdentity,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        Ok(Vec::new())
    }

    /// Prepare the encoded transports for already selected manifests without
    /// decoding their logical payloads. Remote stores use this bounded hook to
    /// batch exact dependency downloads; local stores need no preparation.
    fn prefetch_manifests(&self, _manifests: &[ArtifactManifest]) -> Result<(), CacheError> {
        Ok(())
    }

    /// Return nearest compatible CCM eigenpair keys through a purpose-built
    /// secondary index. Remote implementations must not emulate this by
    /// crawling canonical manifests.
    fn ccm_eigenpair_continuation_keys(
        &self,
        _query: &CcmEigenpairContinuationQuery,
        _maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        Ok(Vec::new())
    }
    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError>;

    /// Return an already verified encoded representation of the logical
    /// payload, when the store retains one. This lets a writable cache adopt a
    /// remote deterministic ZIP without decoding and recompressing it.
    fn verified_encoded_payload(
        &self,
        _manifest: &ArtifactManifest,
    ) -> Result<Option<VerifiedEncodedPayload>, CacheError> {
        Ok(None)
    }

    /// Return exact split parts verified during materialization, when still
    /// available. This is an optional staging accelerator; the encoded package
    /// remains the portable fallback.
    fn verified_transport_parts(
        &self,
        _manifest: &ArtifactManifest,
    ) -> Result<Option<VerifiedTransportParts>, CacheError> {
        Ok(None)
    }

    /// Adopt an already verified encoded payload without changing its logical
    /// identity. Stores that do not support this representation return
    /// `Ok(None)` and callers fall back to `put`.
    fn put_verified_encoded_payload(
        &self,
        _draft: &ArtifactDraft,
        _logical_digest: &ContentDigest,
        _logical_size_bytes: u64,
        _encoded: &VerifiedEncodedPayload,
    ) -> Result<Option<ArtifactManifest>, CacheError> {
        Ok(None)
    }

    fn read_payload(&self, manifest: &ArtifactManifest) -> Result<Vec<u8>, CacheError> {
        let capacity = usize::try_from(manifest.size_bytes).unwrap_or(0);
        let mut payload = Vec::with_capacity(capacity);
        self.read_payload_to(manifest, &mut payload)?;
        Ok(payload)
    }

    /// Read and verify the logical payload while returning any already
    /// verified encoded representation from the same pass. Stores with no
    /// combined path retain the conservative two-operation default.
    fn read_payload_and_encoded(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<(Vec<u8>, Option<VerifiedEncodedPayload>), CacheError> {
        let payload = self.read_payload(manifest)?;
        let encoded = self.verified_encoded_payload(manifest)?;
        Ok((payload, encoded))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedEncodedPayload {
    pub path: PathBuf,
    pub encoding: String,
    /// Exact byte-affecting encoder profile. Legacy objects that predate this
    /// provenance field are reusable as logical payloads but are not eligible
    /// for direct publication adoption.
    pub encoder_profile: Option<String>,
    pub content_digest: ContentDigest,
    pub size_bytes: u64,
}

pub const STORAGE_ENCODER_PROFILE_TAG: &str = "xc_storage_encoder_profile";

fn supported_storage_encoder_profile(profile: &str) -> bool {
    matches!(
        profile,
        DETERMINISTIC_ZIP64_PROFILE_V1 | DETERMINISTIC_ZIP64_PROFILE_V2
    )
}

/// Split transport parts retained by a cache layer for one exact artifact.
///
/// `parts_verified` is true only when this process verified every part's
/// digest (a download or a reconstruction from parts). A layer that merely
/// finds size-matching part files next to an already verified package reports
/// false, and staging then hashes each part before linking it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedTransportParts {
    pub encoding: TransportEncodingRecord,
    pub parts_root: PathBuf,
    pub parts_verified: bool,
}

/// Process-local memo of one exact identity query. It is reused only while
/// the on-disk identity inventory for that semantic digest is byte-for-byte
/// the file it was computed against, so a manifest written by another process
/// becomes visible without a restart.
#[derive(Clone)]
struct IdentityQueryMemo {
    inventory_signature: Option<(u64, SystemTime)>,
    manifests: Vec<ArtifactManifest>,
}

/// Persistent secondary index from one semantic digest to every retained
/// manifest that carries a canonical publication identity. Maintained by
/// every write; repaired once per process per semantic digest from the
/// artifact directory when an entry is absent.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityInventory {
    schema_version: u32,
    entries: Vec<IdentityInventoryEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityInventoryEntry {
    artifact_family: String,
    manifest_digest: ContentDigest,
    payload_digest: ContentDigest,
    kind: String,
    logical_key: String,
    content_digest: ContentDigest,
    /// Manifest record path relative to the store root, `/`-separated.
    manifest_path: String,
}

/// Local filesystem implementation used for working caches and checked-out
/// public/private GitHub cache repositories.
pub struct FilesystemCacheStore {
    name: String,
    root: PathBuf,
    writable: bool,
    visibility: CacheVisibility,
    identity_query_cache: Mutex<HashMap<String, IdentityQueryMemo>>,
    scanned_semantic_digests: Mutex<HashSet<ContentDigest>>,
}

impl FilesystemCacheStore {
    pub fn new(
        name: impl Into<String>,
        root: impl Into<PathBuf>,
        writable: bool,
        visibility: CacheVisibility,
    ) -> Self {
        Self {
            name: name.into(),
            root: root.into(),
            writable,
            visibility,
            identity_query_cache: Mutex::new(HashMap::new()),
            scanned_semantic_digests: Mutex::new(HashSet::new()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn object_path(&self, digest: &ContentDigest) -> PathBuf {
        let prefix = digest.0.get(0..2).unwrap_or("00");
        self.root
            .join("objects")
            .join("sha256")
            .join(prefix)
            .join(&digest.0)
    }

    fn key_directory(&self, key: &ArtifactKey) -> PathBuf {
        self.root
            .join("artifacts")
            .join(encode_component(&key.kind))
            .join(encode_component(&key.logical_key))
            .join(&key.parameters_digest.0)
    }

    fn index_path(&self, key: &ArtifactKey) -> PathBuf {
        self.key_directory(key).join("index.json")
    }

    fn load_index(&self, key: &ArtifactKey) -> Result<ManifestIndex, CacheError> {
        let path = self.index_path(key);
        if !path.exists() {
            let manifest_directory = self.key_directory(key).join("manifests");
            if !manifest_directory.exists() {
                return Ok(ManifestIndex::default());
            }
            let mut manifests = Vec::new();
            for entry in fs::read_dir(&manifest_directory)? {
                let entry = entry?;
                if entry.file_type()?.is_file()
                    && entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "json")
                {
                    manifests.push(format!("manifests/{}", entry.file_name().to_string_lossy()));
                }
            }
            manifests.sort();
            return Ok(ManifestIndex {
                schema_version: 1,
                manifests,
            });
        }
        let bytes = fs::read(&path)?;
        let index: ManifestIndex = serde_json::from_slice(&bytes)?;
        if index.schema_version != 1 {
            return Err(CacheError::InvalidManifest(format!(
                "unsupported manifest index schema {} at {}",
                index.schema_version,
                path.display()
            )));
        }
        Ok(index)
    }

    fn identity_inventory_key(identity: &crate::PayloadDependencyIdentity) -> String {
        format!(
            "{}:{}:{}:{}",
            identity.artifact_family,
            identity.semantic_digest,
            identity.manifest_digest,
            identity.payload_digest
        )
    }

    fn manifest_dependency_identity(
        manifest: &ArtifactManifest,
    ) -> Option<crate::PayloadDependencyIdentity> {
        let encoded = manifest.tags.get(crate::REMOTE_CANONICAL_MANIFEST_TAG)?;
        let Ok(canonical) = serde_json::from_str::<crate::CanonicalArtifactManifest>(encoded)
        else {
            return None;
        };
        let identity = crate::PayloadDependencyIdentity {
            artifact_family: canonical.artifact_family.clone(),
            semantic_digest: canonical.semantic_digest.clone(),
            manifest_digest: canonical.digest().ok()?,
            payload_digest: canonical.payload_digest.clone(),
        };
        identity.validate().ok()?;
        if !manifest_matches_dependency_identity(manifest, &identity).ok()? {
            return None;
        }
        Some(identity)
    }

    fn verify_object(&self, object: &CacheObjectRef) -> Result<(), CacheError> {
        let path = self.object_path(&object.content_digest);
        if !path.exists() {
            return Err(CacheError::NotFound(path.display().to_string()));
        }
        let (digest, size) = digest_file(&path)?;
        if digest != object.content_digest {
            return Err(CacheError::DigestMismatch {
                expected: object.content_digest.to_string(),
                actual: digest.to_string(),
            });
        }
        if size != object.size_bytes {
            return Err(CacheError::InvalidManifest(format!(
                "object {} has size {size}, expected {}",
                object.content_digest, object.size_bytes
            )));
        }
        Ok(())
    }

    fn ensure_object(&self, payload: &[u8]) -> Result<CacheObjectRef, CacheError> {
        let object = CacheObjectRef {
            content_digest: ContentDigest::sha256(payload),
            size_bytes: payload.len() as u64,
        };
        let path = self.object_path(&object.content_digest);
        if path.exists() {
            self.verify_object(&object)?;
        } else {
            atomic_write(&path, payload)?;
        }
        Ok(object)
    }

    fn ensure_verified_object(
        &self,
        encoded: &VerifiedEncodedPayload,
    ) -> Result<CacheObjectRef, CacheError> {
        let metadata = fs::symlink_metadata(&encoded.path)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() != encoded.size_bytes
            || !encoded.content_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "verified encoded cache object has invalid filesystem metadata".to_owned(),
            ));
        }
        let object = CacheObjectRef {
            content_digest: encoded.content_digest.clone(),
            size_bytes: encoded.size_bytes,
        };
        let path = self.object_path(&object.content_digest);
        if path.exists() {
            self.verify_object(&object)?;
            return Ok(object);
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Verify under a private sibling name before exposing the canonical
        // content-addressed path. A same-filesystem source remains zero-copy;
        // other filesystems fall back to one verified copy.
        let temporary = path.with_extension(format!(
            "xc-adopt-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| CacheError::Io(error.to_string()))?
                .as_nanos()
        ));
        let result = (|| {
            match fs::hard_link(&encoded.path, &temporary) {
                Ok(()) => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::CrossesDevices
                            | std::io::ErrorKind::PermissionDenied
                            | std::io::ErrorKind::Unsupported
                    ) =>
                {
                    fs::copy(&encoded.path, &temporary)?;
                }
                Err(error) => return Err(error.into()),
            }

            let (adopted_digest, adopted_size) = digest_file(&temporary)?;
            if adopted_digest != object.content_digest || adopted_size != object.size_bytes {
                return Err(CacheError::DigestMismatch {
                    expected: format!("{} ({} bytes)", object.content_digest, object.size_bytes),
                    actual: format!("{adopted_digest} ({adopted_size} bytes)"),
                });
            }
            match fs::hard_link(&temporary, &path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    self.verify_object(&object)
                }
                Err(error) => Err(error.into()),
            }
        })();
        let _ = fs::remove_file(&temporary);
        result?;
        Ok(object)
    }

    fn publish_manifest(
        &self,
        draft: &ArtifactDraft,
        objects: Vec<CacheObjectRef>,
        content_digest: ContentDigest,
        size_bytes: u64,
    ) -> Result<ArtifactManifest, CacheError> {
        if !self.writable {
            return Err(CacheError::ReadOnlyLayer(self.name.clone()));
        }
        if draft.visibility != self.visibility {
            return Err(CacheError::InvalidManifest(format!(
                "draft visibility {:?} does not match store visibility {:?}",
                draft.visibility, self.visibility
            )));
        }
        let created_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| CacheError::Io(error.to_string()))?
            .as_secs();
        let manifest = ArtifactManifest {
            schema_version: draft.schema_version,
            key: draft.key.clone(),
            content_digest,
            size_bytes,
            objects,
            created_unix_seconds,
            producer_toolkit_version: draft.producer_toolkit_version.clone(),
            minimum_reader_version: draft.minimum_reader_version.clone(),
            maximum_reader_version: draft.maximum_reader_version.clone(),
            quality: draft.quality,
            visibility: draft.visibility,
            immutable: draft.immutable,
            dependencies: draft.dependencies.clone(),
            tags: draft.tags.clone(),
            provenance_digest: draft.provenance_digest.clone(),
        };
        manifest.validate()?;

        let key_directory = self.key_directory(&manifest.key);
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        let manifest_record_digest = ContentDigest::sha256(&manifest_bytes);
        let manifest_name = format!(
            "{}-{}.json",
            manifest.created_unix_seconds, manifest_record_digest.0
        );
        let relative_manifest = PathBuf::from("manifests").join(manifest_name);
        let manifest_path = key_directory.join(&relative_manifest);
        atomic_write(&manifest_path, &manifest_bytes)?;

        let mut index = self.load_index(&manifest.key)?;
        let relative_string = relative_manifest.to_string_lossy().replace('\\', "/");
        if !index
            .manifests
            .iter()
            .any(|entry| entry == &relative_string)
        {
            index.manifests.push(relative_string);
        }
        let index_bytes = serde_json::to_vec_pretty(&index)?;
        atomic_replace(&self.index_path(&manifest.key), &index_bytes)?;
        self.update_ccm_eigenpair_continuation_inventory(&manifest, manifest_record_digest)?;
        let root_relative_manifest = self.root_relative_path(&manifest_path)?;
        if let Some(entry) = Self::identity_inventory_entry(&manifest, &root_relative_manifest) {
            self.record_identity_inventory_entries(&manifest.key.parameters_digest, vec![entry])?;
        }
        Ok(manifest)
    }

    fn root_relative_path(&self, path: &Path) -> Result<String, CacheError> {
        let relative = path.strip_prefix(&self.root).map_err(|_| {
            CacheError::Io(format!(
                "path {} is outside cache root {}",
                path.display(),
                self.root.display()
            ))
        })?;
        Ok(relative.to_string_lossy().replace('\\', "/"))
    }

    fn identity_inventory_path(&self, semantic_digest: &ContentDigest) -> PathBuf {
        let prefix = semantic_digest.0.get(0..2).unwrap_or("00");
        self.root
            .join("identities")
            .join(prefix)
            .join(format!("{}.json", semantic_digest.0))
    }

    fn identity_inventory_signature(path: &Path) -> Result<Option<(u64, SystemTime)>, CacheError> {
        match fs::metadata(path) {
            Ok(metadata) => Ok(Some((metadata.len(), metadata.modified()?))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn load_identity_inventory(path: &Path) -> Result<IdentityInventory, CacheError> {
        match fs::read(path) {
            Ok(bytes) => {
                let inventory: IdentityInventory = serde_json::from_slice(&bytes)?;
                if inventory.schema_version != 1 {
                    return Err(CacheError::InvalidManifest(format!(
                        "unsupported identity inventory schema {} at {}",
                        inventory.schema_version,
                        path.display()
                    )));
                }
                Ok(inventory)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(IdentityInventory {
                schema_version: 1,
                entries: Vec::new(),
            }),
            Err(error) => Err(error.into()),
        }
    }

    fn identity_inventory_entry(
        manifest: &ArtifactManifest,
        root_relative_manifest_path: &str,
    ) -> Option<IdentityInventoryEntry> {
        let identity = Self::manifest_dependency_identity(manifest)?;
        if identity.semantic_digest != manifest.key.parameters_digest {
            return None;
        }
        Some(IdentityInventoryEntry {
            artifact_family: identity.artifact_family,
            manifest_digest: identity.manifest_digest,
            payload_digest: identity.payload_digest,
            kind: manifest.key.kind.clone(),
            logical_key: manifest.key.logical_key.clone(),
            content_digest: manifest.content_digest.clone(),
            manifest_path: root_relative_manifest_path.to_owned(),
        })
    }

    /// Append entries to the semantic digest's identity inventory. The
    /// read-modify-write is serialized within the process and across
    /// processes (advisory file lock), and the inventory only ever grows, so
    /// its byte length is a monotone change signature.
    fn record_identity_inventory_entries(
        &self,
        semantic_digest: &ContentDigest,
        additions: Vec<IdentityInventoryEntry>,
    ) -> Result<(), CacheError> {
        if additions.is_empty() {
            return Ok(());
        }
        static IDENTITY_INVENTORY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = IDENTITY_INVENTORY_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| CacheError::Io("local identity inventory lock was poisoned".to_owned()))?;
        let path = self.identity_inventory_path(semantic_digest);
        let parent = path
            .parent()
            .ok_or_else(|| CacheError::Io(format!("path has no parent: {}", path.display())))?;
        fs::create_dir_all(parent)?;
        // Windows requires read or write access on the handle for LockFileEx;
        // append-only access is refused. Nothing is ever written to the lock
        // file itself, and it is never truncated.
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path.with_extension("lock"))?;
        lock_file.lock_exclusive()?;
        let result = (|| {
            let mut inventory = Self::load_identity_inventory(&path)?;
            let mut changed = false;
            for addition in additions {
                if !inventory
                    .entries
                    .iter()
                    .any(|entry| entry.manifest_path == addition.manifest_path)
                {
                    inventory.entries.push(addition);
                    changed = true;
                }
            }
            if changed {
                inventory
                    .entries
                    .sort_by(|left, right| left.manifest_path.cmp(&right.manifest_path));
                atomic_replace(&path, &serde_json::to_vec_pretty(&inventory)?)?;
            }
            Ok(())
        })();
        let _ = FileExt::unlock(&lock_file);
        result
    }

    fn identity_matches_from_inventory(
        &self,
        identity: &crate::PayloadDependencyIdentity,
        inventory_path: &Path,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        let inventory = Self::load_identity_inventory(inventory_path)?;
        let mut matches = Vec::new();
        for entry in inventory.entries.iter().filter(|entry| {
            entry.artifact_family == identity.artifact_family
                && entry.manifest_digest == identity.manifest_digest
                && entry.payload_digest == identity.payload_digest
        }) {
            if entry.manifest_path.contains(['\\', ':'])
                || entry
                    .manifest_path
                    .split('/')
                    .any(|component| component.is_empty() || component == "." || component == "..")
                || Path::new(&entry.manifest_path)
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
            {
                return Err(CacheError::InvalidManifest(format!(
                    "identity inventory entry has an unsafe manifest path {:?}",
                    entry.manifest_path
                )));
            }
            let manifest_path = entry
                .manifest_path
                .split('/')
                .fold(self.root.clone(), |path, component| path.join(component));
            let bytes = match fs::read(&manifest_path) {
                Ok(bytes) => bytes,
                // A pruned manifest leaves a dangling inventory entry; it is
                // simply not a candidate.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            let manifest: ArtifactManifest = serde_json::from_slice(&bytes)?;
            manifest.validate()?;
            if manifest.key.parameters_digest == identity.semantic_digest
                && manifest_matches_dependency_identity(&manifest, identity)?
            {
                matches.push(manifest);
            }
        }
        Ok(matches)
    }

    /// Enumerate every retained manifest under one semantic digest by
    /// entering only the matching `<logical-key>/<semantic-digest>/manifests`
    /// directories. This is the bounded legacy path for caches written before
    /// the identity inventory existed or by an older writer.
    fn scan_semantic_digest_manifests(
        &self,
        semantic_digest: &ContentDigest,
    ) -> Result<Vec<(ArtifactManifest, String)>, CacheError> {
        let artifacts_root = self.root.join("artifacts");
        let mut discovered = Vec::new();
        if !artifacts_root.is_dir() {
            return Ok(discovered);
        }
        for kind_entry in fs::read_dir(&artifacts_root)? {
            let kind_path = kind_entry?.path();
            if !kind_path.is_dir() {
                continue;
            }
            for logical_entry in fs::read_dir(&kind_path)? {
                let logical_path = logical_entry?.path();
                if !logical_path.is_dir() {
                    continue;
                }
                let manifest_directory = logical_path.join(&semantic_digest.0).join("manifests");
                if !manifest_directory.is_dir() {
                    continue;
                }
                for entry in fs::read_dir(&manifest_directory)? {
                    let entry = entry?;
                    if !entry.file_type()?.is_file()
                        || entry
                            .path()
                            .extension()
                            .is_none_or(|extension| extension != "json")
                    {
                        continue;
                    }
                    let manifest: ArtifactManifest =
                        serde_json::from_slice(&fs::read(entry.path())?)?;
                    manifest.validate()?;
                    if manifest.key.parameters_digest != *semantic_digest {
                        continue;
                    }
                    let relative = self.root_relative_path(&entry.path())?;
                    discovered.push((manifest, relative));
                }
            }
        }
        Ok(discovered)
    }

    fn update_ccm_eigenpair_continuation_inventory(
        &self,
        manifest: &ArtifactManifest,
        manifest_record_digest: ContentDigest,
    ) -> Result<(), CacheError> {
        if manifest.key.kind != CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND {
            return Ok(());
        }
        let Some(encoded) = manifest.tags.get(SEMANTIC_KEY_MANIFEST_TAG) else {
            // Compatibility-only stores may contain opaque test or imported
            // records. They remain exactly addressable but are not eligible
            // for semantic continuation discovery.
            return Ok(());
        };
        let semantic_key: SemanticKeyEnvelope = serde_json::from_str(encoded)?;
        let assurance = match manifest.quality {
            CacheQuality::Certified => ArtifactAssuranceState::Certified,
            CacheQuality::CrossChecked => ArtifactAssuranceState::CrossChecked,
            CacheQuality::Quarantined | CacheQuality::Deprecated => return Ok(()),
            _ => ArtifactAssuranceState::Computed,
        };
        let Some((query, addition)) = ccm_eigenpair_continuation_entry(
            &semantic_key,
            &manifest.key.logical_key,
            manifest_record_digest,
            assurance,
            ArtifactDisposition::Active,
            manifest.producer_toolkit_version.clone(),
            manifest.minimum_reader_version.clone(),
        )?
        else {
            return Ok(());
        };
        static INVENTORY_UPDATE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = INVENTORY_UPDATE_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| {
                CacheError::Io("local continuation inventory lock was poisoned".to_owned())
            })?;
        let path = query
            .repository_path()?
            .split('/')
            .fold(self.root.clone(), |path, component| path.join(component));
        let mut entries = if path.exists() {
            let existing: CcmEigenpairContinuationIndex =
                serde_json::from_slice(&fs::read(&path)?)?;
            existing.validate()?;
            existing.entries
        } else {
            Vec::new()
        };
        entries.retain(|entry| entry.semantic_digest != addition.semantic_digest);
        entries.push(addition);
        let inventory = CcmEigenpairContinuationIndex::rebuild(&query, entries)?;
        atomic_replace(&path, &serde_json::to_vec_pretty(&inventory)?)
    }

    /// Store an artifact as multiple immutable content-addressed objects.
    pub fn put_chunks<I, B>(
        &self,
        draft: &ArtifactDraft,
        chunks: I,
    ) -> Result<ArtifactManifest, CacheError>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        if !self.writable {
            return Err(CacheError::ReadOnlyLayer(self.name.clone()));
        }
        let mut whole_hasher = Sha256::new();
        let mut size_bytes = 0u64;
        let mut objects = Vec::new();
        for chunk in chunks {
            let bytes = chunk.as_ref();
            whole_hasher.update(bytes);
            size_bytes = size_bytes
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| CacheError::Io("payload size exceeds u64".to_owned()))?;
            objects.push(self.ensure_object(bytes)?);
        }
        if objects.is_empty() {
            objects.push(self.ensure_object(&[])?);
        }
        let content_digest = ContentDigest(hex_digest(whole_hasher.finalize().as_slice()));
        self.publish_manifest(draft, objects, content_digest, size_bytes)
    }

    /// Stream a large payload into fixed-size immutable objects without loading
    /// the complete payload into memory.
    pub fn put_reader(
        &self,
        draft: &ArtifactDraft,
        reader: &mut dyn Read,
        chunk_size: usize,
    ) -> Result<ArtifactManifest, CacheError> {
        if chunk_size == 0 {
            return Err(CacheError::InvalidManifest(
                "chunk_size must be positive".to_owned(),
            ));
        }
        if !self.writable {
            return Err(CacheError::ReadOnlyLayer(self.name.clone()));
        }
        let mut whole_hasher = Sha256::new();
        let mut size_bytes = 0u64;
        let mut objects = Vec::new();
        let mut buffer = vec![0u8; chunk_size];
        loop {
            let mut filled = 0usize;
            while filled < buffer.len() {
                let count = reader.read(&mut buffer[filled..])?;
                if count == 0 {
                    break;
                }
                filled += count;
            }
            if filled == 0 {
                break;
            }
            let chunk = &buffer[..filled];
            whole_hasher.update(chunk);
            size_bytes = size_bytes
                .checked_add(filled as u64)
                .ok_or_else(|| CacheError::Io("payload size exceeds u64".to_owned()))?;
            objects.push(self.ensure_object(chunk)?);
            if filled < buffer.len() {
                break;
            }
        }
        if objects.is_empty() {
            objects.push(self.ensure_object(&[])?);
        }
        let content_digest = ContentDigest(hex_digest(whole_hasher.finalize().as_slice()));
        self.publish_manifest(draft, objects, content_digest, size_bytes)
    }
}

/// Whether a locally retained manifest is the artifact a published
/// dependency identity names.
///
/// Only manifests adopted from a published shard carry the retained canonical
/// manifest tag; a manifest without it cannot prove identity equality and is
/// never matched. The canonical manifest digest is recomputed rather than
/// trusted from any index.
fn manifest_matches_dependency_identity(
    manifest: &ArtifactManifest,
    identity: &crate::PayloadDependencyIdentity,
) -> Result<bool, CacheError> {
    let Some(encoded) = manifest.tags.get(crate::REMOTE_CANONICAL_MANIFEST_TAG) else {
        return Ok(false);
    };
    let Ok(canonical) = serde_json::from_str::<crate::CanonicalArtifactManifest>(encoded) else {
        return Ok(false);
    };
    let Some(encoded_semantic_key) = manifest.tags.get(SEMANTIC_KEY_MANIFEST_TAG) else {
        return Ok(false);
    };
    let Ok(semantic_key) = serde_json::from_str::<SemanticKeyEnvelope>(encoded_semantic_key) else {
        return Ok(false);
    };
    if semantic_key.validate().is_err()
        || semantic_key.artifact_kind != manifest.key.kind
        || semantic_key.digest()? != manifest.key.parameters_digest
    {
        return Ok(false);
    }
    let Some(family) = production_staging::family_for_artifact_kind(&semantic_key.artifact_kind)
    else {
        return Ok(false);
    };
    if validate_retained_canonical_binding(
        &canonical,
        &semantic_key,
        family,
        &manifest.content_digest,
        manifest.size_bytes,
        manifest.provenance_digest.as_ref(),
    )
    .is_err()
    {
        return Ok(false);
    }
    Ok(canonical.artifact_family == identity.artifact_family
        && canonical.semantic_digest == identity.semantic_digest
        && canonical.digest()? == identity.manifest_digest
        && canonical.payload_digest == identity.payload_digest)
}

impl CacheStore for FilesystemCacheStore {
    fn identity_candidates(
        &self,
        identity: &crate::PayloadDependencyIdentity,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        identity.validate()?;
        let key = Self::identity_inventory_key(identity);
        let inventory_path = self.identity_inventory_path(&identity.semantic_digest);
        let signature = Self::identity_inventory_signature(&inventory_path)?;
        if let Some(memo) = self
            .identity_query_cache
            .lock()
            .map_err(|_| CacheError::Io("cache identity query lock poisoned".to_owned()))?
            .get(&key)
        {
            if memo.inventory_signature == signature {
                return Ok(memo.manifests.clone());
            }
        }

        // The persistent inventory answers exact identity queries without
        // touching the artifact directory tree. An absent entry triggers at
        // most one bounded directory scan per semantic digest per process,
        // which also repairs the inventory for manifests written before it
        // existed or by an older toolkit.
        let mut matches = self.identity_matches_from_inventory(identity, &inventory_path)?;
        if matches.is_empty() {
            // A read-only store cannot persist the rebuilt inventory. Scan on
            // every inventory miss so a first query for one retained identity
            // cannot hide a second identity under the same semantic digest.
            let first_scan = if self.writable {
                self.scanned_semantic_digests
                    .lock()
                    .map_err(|_| CacheError::Io("cache identity scan lock poisoned".to_owned()))?
                    .insert(identity.semantic_digest.clone())
            } else {
                true
            };
            if first_scan {
                let discovered = self.scan_semantic_digest_manifests(&identity.semantic_digest)?;
                let additions = discovered
                    .iter()
                    .filter_map(|(manifest, relative)| {
                        Self::identity_inventory_entry(manifest, relative)
                    })
                    .collect::<Vec<_>>();
                if !additions.is_empty() && self.writable {
                    self.record_identity_inventory_entries(&identity.semantic_digest, additions)?;
                }
                for (manifest, _) in discovered {
                    if manifest_matches_dependency_identity(&manifest, identity)? {
                        matches.push(manifest);
                    }
                }
            }
        }
        let mut final_signature = Self::identity_inventory_signature(&inventory_path)?;
        let mut stable = final_signature == signature;
        if !stable && self.writable {
            // A different process changed the inventory while this query was
            // running. Re-read it once; never stamp pre-change results with a
            // post-change signature, which would turn the race into a durable
            // false cache hit.
            matches = self.identity_matches_from_inventory(identity, &inventory_path)?;
            let after_retry = Self::identity_inventory_signature(&inventory_path)?;
            stable = after_retry == final_signature;
            final_signature = after_retry;
        }
        if stable {
            self.identity_query_cache
                .lock()
                .map_err(|_| CacheError::Io("cache identity query lock poisoned".to_owned()))?
                .insert(
                    key,
                    IdentityQueryMemo {
                        inventory_signature: final_signature,
                        manifests: matches.clone(),
                    },
                );
        }
        Ok(matches)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn writable(&self) -> bool {
        self.writable
    }

    fn visibility(&self) -> CacheVisibility {
        self.visibility
    }

    fn put(&self, draft: &ArtifactDraft, payload: &[u8]) -> Result<ArtifactManifest, CacheError> {
        self.put_chunks(draft, std::iter::once(payload))
    }

    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        let index = self.load_index(key)?;
        let key_directory = self.key_directory(key);
        let mut manifests = Vec::new();
        for relative in index.manifests {
            let path = key_directory.join(relative);
            if !path.exists() {
                return Err(CacheError::NotFound(path.display().to_string()));
            }
            let manifest: ArtifactManifest = serde_json::from_slice(&fs::read(&path)?)?;
            manifest.validate()?;
            if &manifest.key != key {
                return Err(CacheError::InvalidManifest(format!(
                    "manifest key mismatch at {}",
                    path.display()
                )));
            }
            manifests.push(manifest);
        }
        manifests.sort_by(|left, right| {
            right
                .quality
                .admissible_rank()
                .cmp(&left.quality.admissible_rank())
                .then_with(|| right.created_unix_seconds.cmp(&left.created_unix_seconds))
                .then_with(|| right.content_digest.0.cmp(&left.content_digest.0))
        });
        Ok(manifests)
    }

    fn matching_keys(
        &self,
        kind: &str,
        logical_key_prefix: &str,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        if maximum_keys == 0 {
            return Ok(Vec::new());
        }
        let kind_root = self.root.join("artifacts").join(encode_component(kind));
        if !kind_root.exists() {
            return Ok(Vec::new());
        }
        let mut keys = BTreeSet::new();
        for logical_entry in fs::read_dir(kind_root)? {
            let logical_entry = logical_entry?;
            if !logical_entry.file_type()?.is_dir() {
                continue;
            }
            for parameter_entry in fs::read_dir(logical_entry.path())? {
                let parameter_entry = parameter_entry?;
                if !parameter_entry.file_type()?.is_dir() {
                    continue;
                }
                let manifest_root = parameter_entry.path().join("manifests");
                if !manifest_root.exists() {
                    continue;
                }
                let Some(manifest_entry) = fs::read_dir(manifest_root)?
                    .filter_map(Result::ok)
                    .find(|entry| {
                        entry.file_type().is_ok_and(|kind| kind.is_file())
                            && entry
                                .path()
                                .extension()
                                .is_some_and(|value| value == "json")
                    })
                else {
                    continue;
                };
                let manifest: ArtifactManifest =
                    serde_json::from_slice(&fs::read(manifest_entry.path())?)?;
                manifest.validate()?;
                if manifest.key.kind == kind
                    && manifest.key.logical_key.starts_with(logical_key_prefix)
                {
                    keys.insert((
                        manifest.key.logical_key.clone(),
                        manifest.key.parameters_digest.clone(),
                    ));
                    if keys.len() >= maximum_keys {
                        break;
                    }
                }
            }
            if keys.len() >= maximum_keys {
                break;
            }
        }
        Ok(keys
            .into_iter()
            .map(|(logical_key, parameters_digest)| ArtifactKey {
                kind: kind.to_owned(),
                logical_key,
                parameters_digest,
            })
            .collect())
    }

    fn ccm_eigenpair_continuation_keys(
        &self,
        query: &CcmEigenpairContinuationQuery,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        query.validate()?;
        if maximum_keys == 0 {
            return Ok(Vec::new());
        }
        let inventory_path = query
            .repository_path()?
            .split('/')
            .fold(self.root.clone(), |path, component| path.join(component));
        if inventory_path.exists() {
            let inventory: CcmEigenpairContinuationIndex =
                serde_json::from_slice(&fs::read(inventory_path)?)?;
            return inventory.query(query, &current_toolkit_version()?, maximum_keys);
        }
        // One bounded legacy fallback permits reuse of workstation artifacts
        // written before the secondary inventory existed. New writes maintain
        // the inventory directly and never take this path.
        let prefix = format!("ccm/weil-eigenpair/{}/", query.lambda_squared);
        let mut keys = self.matching_keys(
            CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND,
            &prefix,
            usize::MAX,
        )?;
        keys.retain(|key| {
            let fields = key.logical_key.split('/').collect::<Vec<_>>();
            fields.len() >= 6
                && fields[3]
                    .parse::<usize>()
                    .is_ok_and(|n_modes| n_modes < query.maximum_n_modes)
                && fields[4] == query.precision_bits.to_string()
                && fields[5] == if query.force_even { "even" } else { "natural" }
        });
        keys.sort_by(|left, right| {
            let n = |key: &ArtifactKey| {
                key.logical_key
                    .split('/')
                    .nth(3)
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0)
            };
            n(right)
                .cmp(&n(left))
                .then_with(|| left.parameters_digest.cmp(&right.parameters_digest))
        });
        keys.truncate(maximum_keys);
        Ok(keys)
    }

    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        manifest.validate()?;
        let objects = &manifest.objects;

        let mut whole_hasher = Sha256::new();
        let mut total_size = 0u64;
        let mut buffer = vec![0u8; 1024 * 1024];
        for object in objects {
            let path = self.object_path(&object.content_digest);
            if !path.exists() {
                return Err(CacheError::NotFound(path.display().to_string()));
            }
            let mut file = fs::File::open(&path)?;
            let mut object_hasher = Sha256::new();
            let mut object_size = 0u64;
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                object_hasher.update(&buffer[..count]);
                whole_hasher.update(&buffer[..count]);
                writer.write_all(&buffer[..count])?;
                object_size = object_size
                    .checked_add(count as u64)
                    .ok_or_else(|| CacheError::Io("object size exceeds u64".to_owned()))?;
            }
            let actual = ContentDigest(hex_digest(object_hasher.finalize().as_slice()));
            if actual != object.content_digest {
                return Err(CacheError::DigestMismatch {
                    expected: object.content_digest.to_string(),
                    actual: actual.to_string(),
                });
            }
            if object_size != object.size_bytes {
                return Err(CacheError::InvalidManifest(format!(
                    "object {} has size {object_size}, expected {}",
                    object.content_digest, object.size_bytes
                )));
            }
            total_size = total_size
                .checked_add(object_size)
                .ok_or_else(|| CacheError::Io("payload size exceeds u64".to_owned()))?;
        }

        let actual = ContentDigest(hex_digest(whole_hasher.finalize().as_slice()));
        if actual != manifest.content_digest {
            return Err(CacheError::DigestMismatch {
                expected: manifest.content_digest.to_string(),
                actual: actual.to_string(),
            });
        }
        if total_size != manifest.size_bytes {
            return Err(CacheError::InvalidManifest(format!(
                "payload has size {total_size}, expected {}",
                manifest.size_bytes
            )));
        }
        Ok(())
    }
}

/// Local JSON cache whose stored object is a deterministic ZIP containing one
/// `payload.json` entry. Logical identity remains the uncompressed JSON digest;
/// reuse streams the entry directly from the archive without extracting it.
pub struct ZipJsonFilesystemCacheStore {
    inner: FilesystemCacheStore,
    in_memory_zip_budget: Arc<InMemoryZipBudget>,
    single_pass_object_limit: u64,
}

impl ZipJsonFilesystemCacheStore {
    pub fn new(
        name: impl Into<String>,
        root: impl Into<PathBuf>,
        writable: bool,
        visibility: CacheVisibility,
    ) -> Self {
        Self {
            inner: FilesystemCacheStore::new(name, root, writable, visibility),
            in_memory_zip_budget: InMemoryZipBudget::global(),
            single_pass_object_limit: single_pass_zip_object_limit(),
        }
    }

    #[cfg(test)]
    fn with_in_memory_zip_limits(
        mut self,
        budget: Arc<InMemoryZipBudget>,
        single_pass_object_limit: u64,
    ) -> Self {
        self.in_memory_zip_budget = budget;
        self.single_pass_object_limit = single_pass_object_limit;
        self
    }

    pub fn root(&self) -> &Path {
        self.inner.root()
    }

    /// Workstation objects are encoded by the same deterministic encoder as
    /// publication packages, so a verified local object can be published
    /// byte-for-byte without recompression. `zip_store_and_publication_encoder_agree_byte_for_byte`
    /// guards that equality.
    fn encode(payload: &[u8]) -> Result<Vec<u8>, CacheError> {
        Ok(write_deterministic_zip_entry(
            std::io::Cursor::new(Vec::new()),
            "payload.json",
            payload,
        )?
        .into_inner())
    }

    /// Verify and decode one ZIP JSON object, reporting which read strategy
    /// was used. A single pass is taken only when the object fits the
    /// per-object limit and a reservation against the aggregate in-memory
    /// allowance succeeds immediately; otherwise the object streams.
    fn read_zip_object(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<ZipReadStrategy, CacheError> {
        manifest.validate()?;
        if manifest
            .tags
            .get("xc_storage_encoding")
            .is_none_or(|value| value != "zip-json-entry-v1")
            || manifest.objects.len() != 1
        {
            return Err(CacheError::InvalidManifest(
                "ZIP JSON cache manifest lacks its exact storage encoding".to_owned(),
            ));
        }
        let object = &manifest.objects[0];
        let object_path = self.inner.object_path(&object.content_digest);
        if object.size_bytes <= self.single_pass_object_limit {
            if let Some(_reservation) = self.in_memory_zip_budget.try_reserve(object.size_bytes) {
                let metadata = match fs::symlink_metadata(&object_path) {
                    Ok(metadata) => metadata,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        return Err(CacheError::NotFound(object_path.display().to_string()));
                    }
                    Err(error) => return Err(error.into()),
                };
                if metadata.file_type().is_symlink()
                    || !metadata.is_file()
                    || metadata.len() != object.size_bytes
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "object {} has invalid type or size at {}",
                        object.content_digest,
                        object_path.display()
                    )));
                }
                let capacity = usize::try_from(object.size_bytes).map_err(|_| {
                    CacheError::ResourceLimit(
                        "single-pass ZIP object size does not fit this platform".to_owned(),
                    )
                })?;
                let mut input = fs::File::open(&object_path)?;
                let mut bytes = Vec::with_capacity(capacity);
                (&mut input)
                    .take(object.size_bytes)
                    .read_to_end(&mut bytes)?;
                let mut trailing = [0u8; 1];
                let trailing_bytes = input.read(&mut trailing)?;
                let digest = ContentDigest::sha256(&bytes);
                if digest != object.content_digest {
                    return Err(CacheError::DigestMismatch {
                        expected: object.content_digest.to_string(),
                        actual: digest.to_string(),
                    });
                }
                if bytes.len() as u64 != object.size_bytes || trailing_bytes != 0 {
                    return Err(CacheError::InvalidManifest(format!(
                        "object {} changed size while being read; observed at least {}, expected {}",
                        object.content_digest,
                        bytes.len() + trailing_bytes,
                        object.size_bytes
                    )));
                }
                let archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                    .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                Self::decode_zip_json_entry(archive, manifest, writer)?;
                return Ok(ZipReadStrategy::SinglePass);
            }
        }
        self.inner.verify_object(object)?;
        let file = fs::File::open(&object_path)?;
        let archive = zip::ZipArchive::new(file)
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        Self::decode_zip_json_entry(archive, manifest, writer)?;
        Ok(ZipReadStrategy::Streamed)
    }

    fn decode_zip_json_entry<R: Read + Seek>(
        mut archive: zip::ZipArchive<R>,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        if archive.len() != 1 {
            return Err(CacheError::InvalidManifest(
                "ZIP JSON cache object must contain exactly one entry".to_owned(),
            ));
        }
        let mut entry = archive
            .by_name("payload.json")
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        if entry.size() != manifest.size_bytes {
            return Err(CacheError::InvalidManifest(format!(
                "ZIP payload declares {} decoded bytes, expected {}",
                entry.size(),
                manifest.size_bytes
            )));
        }
        let mut hasher = Sha256::new();
        let mut size = 0u64;
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let count = entry.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            size = size.checked_add(count as u64).ok_or_else(|| {
                CacheError::ResourceLimit("decoded ZIP payload exceeds u64".to_owned())
            })?;
            if size > manifest.size_bytes {
                return Err(CacheError::ResourceLimit(format!(
                    "decoded ZIP payload exceeds declared {}-byte limit",
                    manifest.size_bytes
                )));
            }
            hasher.update(&buffer[..count]);
            writer.write_all(&buffer[..count])?;
        }
        let digest = ContentDigest(hex_digest(hasher.finalize().as_slice()));
        if digest != manifest.content_digest || size != manifest.size_bytes {
            return Err(CacheError::DigestMismatch {
                expected: manifest.content_digest.to_string(),
                actual: digest.to_string(),
            });
        }
        Ok(())
    }
}

/// Default size up to which one compressed ZIP object is read in a single
/// pass: read once into a bounded buffer, hashed from that buffer, then
/// inflated from memory instead of a second seeking pass over the file. It
/// matches the split part size, so a
/// single-pass buffer is never larger than a part the toolkit already holds
/// in memory while splitting. Override with `XC_CACHE_SINGLE_PASS_ZIP_BYTES`.
pub const DEFAULT_SINGLE_PASS_ZIP_OBJECT_BYTES: u64 = 90 * 1024 * 1024;

/// Default aggregate bytes that single-pass reads may hold in memory at the
/// same time, shared by every ZIP store in the process. Override with
/// `XC_CACHE_IN_MEMORY_ZIP_BYTES`.
pub const DEFAULT_IN_MEMORY_ZIP_BYTES: u64 = 256 * 1024 * 1024;

/// Hard ceiling for process-local ZIP memory overrides. Values above this are
/// treated as invalid rather than becoming effectively unbounded.
pub const MAXIMUM_IN_MEMORY_ZIP_OVERRIDE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

pub const SINGLE_PASS_ZIP_OBJECT_BYTES_ENV: &str = "XC_CACHE_SINGLE_PASS_ZIP_BYTES";
pub const IN_MEMORY_ZIP_BYTES_ENV: &str = "XC_CACHE_IN_MEMORY_ZIP_BYTES";

/// Parse a byte-limit override. Anything that is not a positive integer,
/// including an empty, negative, overflowing, or unparsable value, yields
/// the default deterministically. An override can therefore never fail a
/// run and never produce an unbounded limit.
fn parse_byte_limit(value: Option<&str>, default: u64) -> u64 {
    value
        .map(str::trim)
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|limit| *limit > 0 && *limit <= MAXIMUM_IN_MEMORY_ZIP_OVERRIDE_BYTES)
        .unwrap_or(default)
}

fn single_pass_zip_object_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        parse_byte_limit(
            std::env::var(SINGLE_PASS_ZIP_OBJECT_BYTES_ENV)
                .ok()
                .as_deref(),
            DEFAULT_SINGLE_PASS_ZIP_OBJECT_BYTES,
        )
    })
}

/// Aggregate allowance for compressed ZIP objects that single-pass reads hold
/// in memory. Reservation never waits: a read that does not fit streams
/// through the two-pass path instead, so a saturated allowance costs disk
/// reads, never latency or memory.
pub struct InMemoryZipBudget {
    limit: u64,
    in_use: Mutex<u64>,
}

impl InMemoryZipBudget {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            in_use: Mutex::new(0),
        }
    }

    /// The process-wide allowance every store created with `new` shares.
    pub fn global() -> Arc<Self> {
        static GLOBAL: OnceLock<Arc<InMemoryZipBudget>> = OnceLock::new();
        Arc::clone(GLOBAL.get_or_init(|| {
            Arc::new(Self::new(parse_byte_limit(
                std::env::var(IN_MEMORY_ZIP_BYTES_ENV).ok().as_deref(),
                DEFAULT_IN_MEMORY_ZIP_BYTES,
            )))
        }))
    }

    fn try_reserve(self: &Arc<Self>, bytes: u64) -> Option<InMemoryZipReservation> {
        let mut in_use = self
            .in_use
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let projected = in_use.checked_add(bytes)?;
        if projected > self.limit {
            return None;
        }
        *in_use = projected;
        Some(InMemoryZipReservation {
            budget: Arc::clone(self),
            bytes,
        })
    }

    #[cfg(test)]
    fn in_use(&self) -> u64 {
        *self
            .in_use
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Bytes of one compressed object held in memory; released on drop.
struct InMemoryZipReservation {
    budget: Arc<InMemoryZipBudget>,
    bytes: u64,
}

impl Drop for InMemoryZipReservation {
    fn drop(&mut self) {
        let mut in_use = self
            .budget
            .in_use
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *in_use = in_use.saturating_sub(self.bytes);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ZipReadStrategy {
    SinglePass,
    Streamed,
}

impl CacheStore for ZipJsonFilesystemCacheStore {
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn writable(&self) -> bool {
        self.inner.writable
    }

    fn visibility(&self) -> CacheVisibility {
        self.inner.visibility
    }

    fn put(&self, draft: &ArtifactDraft, payload: &[u8]) -> Result<ArtifactManifest, CacheError> {
        if !self.inner.writable {
            return Err(CacheError::ReadOnlyLayer(self.inner.name.clone()));
        }
        let encoded = Self::encode(payload)?;
        let object = self.inner.ensure_object(&encoded)?;
        let mut encoded_draft = draft.clone();
        encoded_draft.tags.insert(
            "xc_storage_encoding".to_owned(),
            "zip-json-entry-v1".to_owned(),
        );
        encoded_draft.tags.insert(
            STORAGE_ENCODER_PROFILE_TAG.to_owned(),
            CURRENT_DETERMINISTIC_ZIP64_PROFILE.to_owned(),
        );
        self.inner.publish_manifest(
            &encoded_draft,
            vec![object],
            ContentDigest::sha256(payload),
            payload.len() as u64,
        )
    }

    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        self.inner.candidates(key)
    }

    fn matching_keys(
        &self,
        kind: &str,
        logical_key_prefix: &str,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        self.inner
            .matching_keys(kind, logical_key_prefix, maximum_keys)
    }

    fn identity_candidates(
        &self,
        identity: &crate::PayloadDependencyIdentity,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        self.inner.identity_candidates(identity)
    }

    fn ccm_eigenpair_continuation_keys(
        &self,
        query: &CcmEigenpairContinuationQuery,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        self.inner
            .ccm_eigenpair_continuation_keys(query, maximum_keys)
    }

    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        self.read_zip_object(manifest, writer).map(|_| ())
    }

    fn read_payload_and_encoded(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<(Vec<u8>, Option<VerifiedEncodedPayload>), CacheError> {
        let capacity = usize::try_from(manifest.size_bytes).unwrap_or(0);
        let mut payload = Vec::with_capacity(capacity);
        // `read_payload_to` verifies the compressed object before opening it
        // and then verifies the logical JSON while decoding. Constructing the
        // descriptor from that same verified object avoids hashing the full
        // ZIP a second time on every local reuse hit.
        self.read_payload_to(manifest, &mut payload)?;
        let object = &manifest.objects[0];
        Ok((
            payload,
            manifest
                .tags
                .get(STORAGE_ENCODER_PROFILE_TAG)
                .filter(|profile| supported_storage_encoder_profile(profile))
                .map(|profile| VerifiedEncodedPayload {
                    path: self.inner.object_path(&object.content_digest),
                    encoding: "zip-json-entry-v1".to_owned(),
                    encoder_profile: Some(profile.clone()),
                    content_digest: object.content_digest.clone(),
                    size_bytes: object.size_bytes,
                }),
        ))
    }

    fn verified_encoded_payload(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<Option<VerifiedEncodedPayload>, CacheError> {
        manifest.validate()?;
        if manifest
            .tags
            .get("xc_storage_encoding")
            .is_none_or(|value| value != "zip-json-entry-v1")
            || manifest.objects.len() != 1
        {
            return Ok(None);
        }
        let Some(encoder_profile) = manifest.tags.get(STORAGE_ENCODER_PROFILE_TAG) else {
            // Legacy local ZIP objects remain valid logical cache entries, but
            // their writer profile was not persisted and must not be asserted
            // when adopting their bytes for publication.
            return Ok(None);
        };
        if !supported_storage_encoder_profile(encoder_profile) {
            return Ok(None);
        }
        let object = &manifest.objects[0];
        self.inner.verify_object(object)?;
        Ok(Some(VerifiedEncodedPayload {
            path: self.inner.object_path(&object.content_digest),
            encoding: "zip-json-entry-v1".to_owned(),
            encoder_profile: Some(encoder_profile.clone()),
            content_digest: object.content_digest.clone(),
            size_bytes: object.size_bytes,
        }))
    }

    fn put_verified_encoded_payload(
        &self,
        draft: &ArtifactDraft,
        logical_digest: &ContentDigest,
        logical_size_bytes: u64,
        encoded: &VerifiedEncodedPayload,
    ) -> Result<Option<ArtifactManifest>, CacheError> {
        if !self.inner.writable {
            return Err(CacheError::ReadOnlyLayer(self.inner.name.clone()));
        }
        let Some(encoder_profile) = encoded.encoder_profile.as_deref() else {
            return Ok(None);
        };
        if encoded.encoding != "zip-json-entry-v1"
            || !supported_storage_encoder_profile(encoder_profile)
        {
            return Ok(None);
        }
        let object = self.inner.ensure_verified_object(encoded)?;
        let mut encoded_draft = draft.clone();
        encoded_draft.tags.insert(
            "xc_storage_encoding".to_owned(),
            "zip-json-entry-v1".to_owned(),
        );
        encoded_draft.tags.insert(
            STORAGE_ENCODER_PROFILE_TAG.to_owned(),
            encoder_profile.to_owned(),
        );
        Ok(Some(self.inner.publish_manifest(
            &encoded_draft,
            vec![object],
            logical_digest.clone(),
            logical_size_bytes,
        )?))
    }
}

pub struct CacheLayer {
    pub precedence: u32,
    pub store: Box<dyn CacheStore>,
}

pub struct ResolvedArtifact {
    pub layer_name: String,
    pub manifest: ArtifactManifest,
    pub payload: Vec<u8>,
    pub encoded_payload: Option<VerifiedEncodedPayload>,
}

/// An exact artifact resolved as metadata plus its verified encoded
/// representation. No logical payload was read: publication staging can
/// split the encoded object and, when parts are retained, link them
/// directly.
pub struct ResolvedEncodedArtifact {
    pub layer_name: String,
    pub manifest: ArtifactManifest,
    pub encoded: VerifiedEncodedPayload,
    pub transport: Option<VerifiedTransportParts>,
}

pub struct CacheResolver {
    layers: Vec<CacheLayer>,
}

impl CacheResolver {
    pub fn new(mut layers: Vec<CacheLayer>) -> Self {
        layers.sort_by_key(|layer| layer.precedence);
        Self { layers }
    }

    pub fn verified_transport_parts(
        &self,
        layer_name: &str,
        manifest: &ArtifactManifest,
    ) -> Result<Option<VerifiedTransportParts>, CacheError> {
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.store.name() == layer_name)
            .ok_or_else(|| CacheError::NotFound(format!("cache layer {layer_name:?}")))?;
        layer.store.verified_transport_parts(manifest)
    }

    /// Resolve exact key-based dependency metadata in precedence order, then
    /// ask each selected layer to prepare all of its transports as one batch.
    /// No logical payload is decoded here.
    pub fn prefetch_exact_dependencies(
        &self,
        dependencies: &[DependencyRef],
        policy: &CachePolicy,
    ) -> Result<(), CacheError> {
        let mut selected = (0..self.layers.len())
            .map(|_| Vec::<ArtifactManifest>::new())
            .collect::<Vec<_>>();
        for dependency in dependencies {
            let mut found = false;
            for (index, layer) in self.layers.iter().enumerate() {
                if let Some(manifest) =
                    layer
                        .store
                        .candidates(&dependency.key)?
                        .into_iter()
                        .find(|manifest| {
                            manifest.content_digest == dependency.content_digest
                                && manifest.quality.admissible_rank()
                                    >= dependency.required_quality.admissible_rank()
                                && policy.accepts(manifest)
                        })
                {
                    selected[index].push(manifest);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(CacheError::NotFound(format!(
                    "{} / {} with digest {}",
                    dependency.key.kind, dependency.key.logical_key, dependency.content_digest
                )));
            }
        }
        for (layer, manifests) in self.layers.iter().zip(selected) {
            if !manifests.is_empty() {
                layer.store.prefetch_manifests(&manifests)?;
            }
        }
        Ok(())
    }

    /// Resolve exact canonical dependency metadata in precedence order, then
    /// prepare sibling transports together. This forms a bounded progressive
    /// wavefront: each verified canonical manifest reveals the next exact set.
    pub fn prefetch_dependency_identities(
        &self,
        identities: &[crate::PayloadDependencyIdentity],
        policy: &CachePolicy,
    ) -> Result<(), CacheError> {
        let mut selected = (0..self.layers.len())
            .map(|_| Vec::<ArtifactManifest>::new())
            .collect::<Vec<_>>();
        for identity in identities {
            identity.validate()?;
            let mut found = false;
            for (index, layer) in self.layers.iter().enumerate() {
                for manifest in layer.store.identity_candidates(identity)? {
                    if !policy.accepts(&manifest) {
                        continue;
                    }
                    if !manifest_matches_dependency_identity(&manifest, identity)? {
                        return Err(CacheError::InvalidManifest(format!(
                            "cache layer {} returned a candidate that does not match dependency identity {}/{}",
                            layer.store.name(),
                            identity.artifact_family,
                            identity.semantic_digest.0
                        )));
                    }
                    selected[index].push(manifest);
                    found = true;
                    break;
                }
                if found {
                    break;
                }
            }
            if !found {
                return Err(CacheError::NotFound(format!(
                    "published dependency {}/{} (manifest {})",
                    identity.artifact_family,
                    identity.semantic_digest.0,
                    identity.manifest_digest.0
                )));
            }
        }
        for (layer, manifests) in self.layers.iter().zip(selected) {
            if !manifests.is_empty() {
                layer.store.prefetch_manifests(&manifests)?;
            }
        }
        Ok(())
    }

    pub fn resolve_manifest(
        &self,
        key: &ArtifactKey,
        policy: &CachePolicy,
    ) -> Result<(String, ArtifactManifest), CacheError> {
        let mut rejected = Vec::new();
        for layer in &self.layers {
            for manifest in layer.store.candidates(key)? {
                if policy.accepts(&manifest) {
                    return Ok((layer.store.name().to_owned(), manifest));
                }
                rejected.push(format!(
                    "{}:{}:{:?}",
                    layer.store.name(),
                    manifest.content_digest,
                    manifest.quality
                ));
            }
        }
        Err(CacheError::NotFound(format!(
            "{} / {}; rejected candidates: {}",
            key.kind,
            key.logical_key,
            rejected.join(", ")
        )))
    }

    pub fn resolve_exact_manifest(
        &self,
        key: &ArtifactKey,
        content_digest: &ContentDigest,
        policy: &CachePolicy,
    ) -> Result<(String, ArtifactManifest), CacheError> {
        for layer in &self.layers {
            for manifest in layer.store.candidates(key)? {
                if &manifest.content_digest == content_digest && policy.accepts(&manifest) {
                    return Ok((layer.store.name().to_owned(), manifest));
                }
            }
        }
        Err(CacheError::NotFound(format!(
            "{} / {} with digest {}",
            key.kind, key.logical_key, content_digest
        )))
    }

    /// Verify that every exact dependency named by the selected artifact is
    /// present, compatible, and at least as strong as the dependency's stated
    /// quality requirement. Payloads are not loaded.
    pub fn validate_dependency_closure(
        &self,
        manifest: &ArtifactManifest,
        policy: &CachePolicy,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        let mut visited = BTreeSet::new();
        let mut ordered = Vec::new();
        self.visit_dependencies(manifest, policy, &mut visited, &mut ordered)?;
        Ok(ordered)
    }

    fn visit_dependencies(
        &self,
        manifest: &ArtifactManifest,
        policy: &CachePolicy,
        visited: &mut BTreeSet<String>,
        ordered: &mut Vec<ArtifactManifest>,
    ) -> Result<(), CacheError> {
        for dependency in &manifest.dependencies {
            let identity = format!(
                "{}:{}:{}:{}",
                dependency.key.kind,
                dependency.key.logical_key,
                dependency.key.parameters_digest,
                dependency.content_digest
            );
            if !visited.insert(identity) {
                continue;
            }
            let (_, resolved) =
                self.resolve_exact_manifest(&dependency.key, &dependency.content_digest, policy)?;
            if resolved.quality.admissible_rank() < dependency.required_quality.admissible_rank() {
                return Err(CacheError::InvalidManifest(format!(
                    "dependency {} has quality {:?}, requires {:?}",
                    resolved.content_digest, resolved.quality, dependency.required_quality
                )));
            }
            self.visit_dependencies(&resolved, policy, visited, ordered)?;
            ordered.push(resolved);
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        key: &ArtifactKey,
        policy: &CachePolicy,
    ) -> Result<ResolvedArtifact, CacheError> {
        let mut rejected = Vec::new();
        for layer in &self.layers {
            for manifest in layer.store.candidates(key)? {
                if !policy.accepts(&manifest) {
                    rejected.push(format!(
                        "{}:{}:{:?}",
                        layer.store.name(),
                        manifest.content_digest,
                        manifest.quality
                    ));
                    continue;
                }
                let (payload, encoded_payload) = layer.store.read_payload_and_encoded(&manifest)?;
                return Ok(ResolvedArtifact {
                    layer_name: layer.store.name().to_owned(),
                    manifest,
                    payload,
                    encoded_payload,
                });
            }
        }
        Err(CacheError::NotFound(format!(
            "{} / {}; rejected candidates: {}",
            key.kind,
            key.logical_key,
            rejected.join(", ")
        )))
    }

    /// Resolve and load one exact key/content pair. Dependency closure must
    /// not accidentally select a newer candidate for the same semantic key,
    /// and it selects the same candidate `prefetch_exact_dependencies`
    /// prepared: the first layer whose copy has the exact digest and at least
    /// the required quality.
    pub fn resolve_exact(
        &self,
        key: &ArtifactKey,
        content_digest: &ContentDigest,
        required_quality: CacheQuality,
        policy: &CachePolicy,
    ) -> Result<ResolvedArtifact, CacheError> {
        for layer in &self.layers {
            for manifest in layer.store.candidates(key)? {
                if !Self::exact_candidate_accepted(
                    &manifest,
                    content_digest,
                    required_quality,
                    policy,
                ) {
                    continue;
                }
                let (payload, encoded_payload) = layer.store.read_payload_and_encoded(&manifest)?;
                return Ok(ResolvedArtifact {
                    layer_name: layer.store.name().to_owned(),
                    manifest,
                    payload,
                    encoded_payload,
                });
            }
        }
        Err(CacheError::NotFound(format!(
            "{} / {} with digest {}",
            key.kind, key.logical_key, content_digest
        )))
    }

    fn exact_candidate_accepted(
        manifest: &ArtifactManifest,
        content_digest: &ContentDigest,
        required_quality: CacheQuality,
        policy: &CachePolicy,
    ) -> bool {
        &manifest.content_digest == content_digest
            && manifest.quality.admissible_rank() >= required_quality.admissible_rank()
            && policy.accepts(manifest)
    }

    /// Resolve one exact key/content pair as metadata plus its verified
    /// encoded representation, without reading the logical payload. Returns
    /// `Ok(None)` when the selected layer retains no encoded form, in which
    /// case `resolve_exact` selects the same layer and loads the payload.
    pub fn resolve_exact_encoded(
        &self,
        key: &ArtifactKey,
        content_digest: &ContentDigest,
        required_quality: CacheQuality,
        policy: &CachePolicy,
    ) -> Result<Option<ResolvedEncodedArtifact>, CacheError> {
        for layer in &self.layers {
            for manifest in layer.store.candidates(key)? {
                if !Self::exact_candidate_accepted(
                    &manifest,
                    content_digest,
                    required_quality,
                    policy,
                ) {
                    continue;
                }
                let Some(encoded) = layer.store.verified_encoded_payload(&manifest)? else {
                    return Ok(None);
                };
                let transport = layer.store.verified_transport_parts(&manifest)?;
                return Ok(Some(ResolvedEncodedArtifact {
                    layer_name: layer.store.name().to_owned(),
                    manifest,
                    encoded,
                    transport,
                }));
            }
        }
        Ok(None)
    }

    /// Identity-addressed counterpart of `resolve_exact_encoded`. Candidates
    /// are re-verified against the requested identity exactly as in
    /// `resolve_dependency_identity`.
    pub fn resolve_dependency_identity_encoded(
        &self,
        identity: &crate::PayloadDependencyIdentity,
        policy: &CachePolicy,
    ) -> Result<Option<ResolvedEncodedArtifact>, CacheError> {
        identity.validate()?;
        for layer in &self.layers {
            for manifest in layer.store.identity_candidates(identity)? {
                if !policy.accepts(&manifest) {
                    continue;
                }
                if !manifest_matches_dependency_identity(&manifest, identity)? {
                    return Err(CacheError::InvalidManifest(format!(
                        "cache layer {} returned a candidate that does not match dependency \
                         identity {}/{}",
                        layer.store.name(),
                        identity.artifact_family,
                        identity.semantic_digest.0
                    )));
                }
                let Some(encoded) = layer.store.verified_encoded_payload(&manifest)? else {
                    return Ok(None);
                };
                let transport = layer.store.verified_transport_parts(&manifest)?;
                return Ok(Some(ResolvedEncodedArtifact {
                    layer_name: layer.store.name().to_owned(),
                    manifest,
                    encoded,
                    transport,
                }));
            }
        }
        Ok(None)
    }

    /// Return distinct keys visible through all configured layers. Layer
    /// precedence is preserved, while duplicate semantic identities are
    /// collapsed.
    /// Resolve an artifact by its exact published dependency identity.
    ///
    /// Layers are consulted in precedence order, so a copy already adopted
    /// into the local working cache wins over a remote fetch. Returns
    /// `Ok(None)` when no layer can produce the identity, leaving the caller
    /// to report which closure member is unreachable.
    pub fn resolve_dependency_identity(
        &self,
        identity: &crate::PayloadDependencyIdentity,
        policy: &CachePolicy,
    ) -> Result<Option<ResolvedArtifact>, CacheError> {
        identity.validate()?;
        for layer in &self.layers {
            for manifest in layer.store.identity_candidates(identity)? {
                if !policy.accepts(&manifest) {
                    continue;
                }
                // Stores are not trusted to have matched the identity: the
                // resolver re-verifies every candidate against the retained
                // canonical manifest before returning it.
                if !manifest_matches_dependency_identity(&manifest, identity)? {
                    return Err(CacheError::InvalidManifest(format!(
                        "cache layer {} returned a candidate that does not match dependency \
                         identity {}/{}",
                        layer.store.name(),
                        identity.artifact_family,
                        identity.semantic_digest.0
                    )));
                }
                let (payload, encoded_payload) = layer.store.read_payload_and_encoded(&manifest)?;
                return Ok(Some(ResolvedArtifact {
                    layer_name: layer.store.name().to_owned(),
                    manifest,
                    payload,
                    encoded_payload,
                }));
            }
        }
        Ok(None)
    }

    pub fn matching_keys(
        &self,
        kind: &str,
        logical_key_prefix: &str,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        let mut seen = BTreeSet::new();
        let mut keys = Vec::new();
        for layer in &self.layers {
            for key in layer
                .store
                .matching_keys(kind, logical_key_prefix, maximum_keys)?
            {
                let identity = (
                    key.kind.clone(),
                    key.logical_key.clone(),
                    key.parameters_digest.clone(),
                );
                if seen.insert(identity) {
                    keys.push(key);
                    if keys.len() >= maximum_keys {
                        return Ok(keys);
                    }
                }
            }
        }
        Ok(keys)
    }

    pub fn ccm_eigenpair_continuation_keys(
        &self,
        query: &CcmEigenpairContinuationQuery,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        let mut seen = BTreeSet::new();
        let mut keys = Vec::new();
        for layer in &self.layers {
            for key in layer
                .store
                .ccm_eigenpair_continuation_keys(query, maximum_keys)?
            {
                let identity = (
                    key.kind.clone(),
                    key.logical_key.clone(),
                    key.parameters_digest.clone(),
                );
                if seen.insert(identity) {
                    keys.push(key);
                }
            }
        }
        keys.sort_by(|left, right| {
            let n = |key: &ArtifactKey| {
                key.logical_key
                    .split('/')
                    .nth(3)
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0)
            };
            n(right)
                .cmp(&n(left))
                .then_with(|| left.parameters_digest.cmp(&right.parameters_digest))
        });
        keys.truncate(maximum_keys);
        Ok(keys)
    }

    pub fn first_writable(&self, visibility: CacheVisibility) -> Option<&dyn CacheStore> {
        self.layers
            .iter()
            .map(|layer| layer.store.as_ref())
            .find(|store| store.writable() && store.visibility() == visibility)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RepositoryShard {
    pub id: String,
    pub repository: String,
    pub visibility: CacheVisibility,
    pub artifact_kinds: Vec<String>,
    pub reachable_payload_bytes: u64,
    pub estimated_history_bytes: u64,
    pub safe_payload_limit_bytes: u64,
    pub writable: bool,
}

impl RepositoryShard {
    pub fn projected_total_bytes(&self, new_payload_bytes: u64, new_history_bytes: u64) -> u64 {
        self.reachable_payload_bytes
            .saturating_add(self.estimated_history_bytes)
            .saturating_add(new_payload_bytes)
            .saturating_add(new_history_bytes)
    }

    pub fn can_accept(
        &self,
        artifact_kind: &str,
        visibility: CacheVisibility,
        new_payload_bytes: u64,
        new_history_bytes: u64,
    ) -> bool {
        self.writable
            && self.visibility == visibility
            && (self.artifact_kinds.is_empty()
                || self.artifact_kinds.iter().any(|kind| kind == artifact_kind))
            && self.projected_total_bytes(new_payload_bytes, new_history_bytes)
                <= self.safe_payload_limit_bytes
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheRepositoryRegistry {
    pub schema_version: u32,
    pub shards: Vec<RepositoryShard>,
}

impl CacheRepositoryRegistry {
    pub fn select_write_shard(
        &self,
        artifact_kind: &str,
        visibility: CacheVisibility,
        new_payload_bytes: u64,
        new_history_bytes: u64,
    ) -> Result<&RepositoryShard, CacheError> {
        // Fill the fullest acceptable shard first. This keeps the number of
        // repositories small while respecting the approved 100 GB threshold.
        self.shards
            .iter()
            .filter(|shard| {
                shard.can_accept(
                    artifact_kind,
                    visibility,
                    new_payload_bytes,
                    new_history_bytes,
                )
            })
            .max_by_key(|shard| shard.projected_total_bytes(new_payload_bytes, new_history_bytes))
            .ok_or_else(|| {
                CacheError::NoWritableShard(format!(
                    "kind={artifact_kind}, visibility={visibility:?}, payload={new_payload_bytes}, history={new_history_bytes}"
                ))
            })
    }
}

fn encode_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("_{byte:02x}"));
        }
    }
    if out.is_empty() {
        "_empty".to_owned()
    } else {
        out
    }
}

fn digest_file(path: &Path) -> Result<(ContentDigest, u64), CacheError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        size = size
            .checked_add(count as u64)
            .ok_or_else(|| CacheError::Io("file size exceeds u64".to_owned()))?;
    }
    Ok((
        ContentDigest(hex_digest(hasher.finalize().as_slice())),
        size,
    ))
}

fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), CacheError> {
    let parent = path
        .parent()
        .ok_or_else(|| CacheError::Io(format!("path has no parent: {}", path.display())))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".replace-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| CacheError::Io(error.to_string()))?
            .as_nanos()
    ));
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        CacheError::Io(format!("failed to replace {}: {error}", path.display()))
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CacheError> {
    let parent = path
        .parent()
        .ok_or_else(|| CacheError::Io(format!("path has no parent: {}", path.display())))?;
    fs::create_dir_all(parent)?;
    let unique = format!(
        ".tmp-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| CacheError::Io(error.to_string()))?
            .as_nanos()
    );
    let temporary = parent.join(unique);
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(error) if path.exists() => {
            let _ = fs::remove_file(&temporary);
            let (existing_digest, existing_size) = digest_file(path)?;
            let expected_digest = ContentDigest::sha256(bytes);
            if existing_digest == expected_digest && existing_size == bytes.len() as u64 {
                Ok(())
            } else {
                Err(CacheError::Io(format!(
                    "atomic write collision at {}: {error}",
                    path.display()
                )))
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(CacheError::Io(format!(
                "failed to publish {}: {error}",
                path.display()
            )))
        }
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

pub fn sha256_hex(input: &[u8]) -> String {
    ContentDigest::sha256(input).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(value: &str) -> ToolkitVersion {
        ToolkitVersion::parse(value).unwrap()
    }

    fn draft(
        key: ArtifactKey,
        quality: CacheQuality,
        visibility: CacheVisibility,
    ) -> ArtifactDraft {
        ArtifactDraft {
            schema_version: 1,
            key,
            producer_toolkit_version: version("0.13.0"),
            minimum_reader_version: version("0.13.0"),
            maximum_reader_version: None,
            quality,
            visibility,
            immutable: true,
            dependencies: Vec::new(),
            tags: BTreeMap::new(),
            provenance_digest: None,
        }
    }

    fn temporary_root(name: &str) -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("{name}-{}", std::process::id()))
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn version_order_handles_prereleases() {
        assert!(version("0.13.0-rc.1") < version("0.13.0"));
        assert!(version("0.13.0") < version("0.14.0"));
    }

    #[test]
    fn chunked_payload_round_trips_and_reuses_objects() {
        let root = temporary_root("xc-cache-chunks");
        let _ = fs::remove_dir_all(&root);
        let store = FilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let key_a = ArtifactKey::new("tau", "a", br#"{"n":120}"#).unwrap();
        let key_b = ArtifactKey::new("tau", "b", br#"{"n":121}"#).unwrap();
        let manifest_a = store
            .put_chunks(
                &draft(key_a, CacheQuality::Validated, CacheVisibility::Local),
                [&b"shared"[..], &b"-a"[..]],
            )
            .unwrap();
        let manifest_b = store
            .put_chunks(
                &draft(key_b, CacheQuality::Validated, CacheVisibility::Local),
                [&b"shared"[..], &b"-b"[..]],
            )
            .unwrap();
        assert_eq!(manifest_a.objects[0], manifest_b.objects[0]);
        assert_eq!(store.read_payload(&manifest_a).unwrap(), b"shared-a");
        assert_eq!(store.read_payload(&manifest_b).unwrap(), b"shared-b");
        let object_path = store.object_path(&manifest_a.objects[0].content_digest);
        assert!(object_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn filesystem_store_discovers_bounded_logical_key_prefixes() {
        let root = temporary_root("xc-cache-prefix-discovery");
        let _ = fs::remove_dir_all(&root);
        let store = FilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        for n_modes in [10, 20, 30] {
            let logical_key =
                format!("ccm/weil-eigenpair/13/{n_modes}/256/even/shift_invert_krylov");
            let key = ArtifactKey::new(
                "ccm_weil_eigenpair",
                logical_key,
                format!("N={n_modes}").as_bytes(),
            )
            .unwrap();
            store
                .put(
                    &draft(key, CacheQuality::Validated, CacheVisibility::Local),
                    b"{}",
                )
                .unwrap();
        }
        let keys = store
            .matching_keys("ccm_weil_eigenpair", "ccm/weil-eigenpair/13/", 2)
            .unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys
            .iter()
            .all(|key| key.logical_key.starts_with("ccm/weil-eigenpair/13/")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zip_store_memoizes_exact_identity_queries_without_scanning_unrelated_manifests() {
        let root = temporary_root("xc-cache-zip-identity-inventory");
        let _ = fs::remove_dir_all(&root);
        let store = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let payload = br#"{"fixture":true}"#;
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_tau_matrix".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({"case": "identity-index"}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "json".to_owned(),
            precision_bits: None,
            scalar_representation: "json".to_owned(),
            dimensions: vec![payload.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "payload.json".to_owned(),
                content_digest: ContentDigest::sha256(payload),
                size_bytes: payload.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let canonical = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm-matrices".to_owned(),
            semantic_key: semantic_key.clone(),
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![ContentDigest::sha256(b"transport")],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"configuration"),
            producer_toolkit_version: version("0.14.1"),
            minimum_reader_version: version("0.14.1"),
            maximum_reader_version: None,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            claim_scope: "identity inventory fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let identity = PayloadDependencyIdentity {
            artifact_family: canonical.artifact_family.clone(),
            semantic_digest: semantic_digest.clone(),
            manifest_digest: canonical.digest().unwrap(),
            payload_digest,
        };

        let unrelated = root
            .join("artifacts")
            .join("unrelated")
            .join("logical")
            .join(ContentDigest::sha256(b"unrelated-semantic").0)
            .join("manifests")
            .join("invalid.json");
        fs::create_dir_all(unrelated.parent().unwrap()).unwrap();
        fs::write(&unrelated, b"not json").unwrap();

        // Cache the empty exact query first. The unrelated corrupt manifest
        // must not be parsed, and a later same-process write must update the
        // memoized result rather than remaining invisible until restart.
        assert!(store.identity_candidates(&identity).unwrap().is_empty());
        let key = ArtifactKey {
            kind: semantic_key.artifact_kind.clone(),
            logical_key: "ccm/fixture/identity-index".to_owned(),
            parameters_digest: semantic_digest,
        };
        let mut artifact = draft(key, CacheQuality::Validated, CacheVisibility::Local);
        artifact.tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&semantic_key).unwrap(),
        );
        artifact.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&canonical).unwrap(),
        );
        let honest = store.put(&artifact, payload).unwrap();
        assert_eq!(store.identity_candidates(&identity).unwrap().len(), 1);

        // A local adapter cannot claim the honest published identity while
        // carrying different payload bytes. The inventory is only a lookup
        // accelerator; every candidate is rebound to its semantic envelope,
        // payload digest/size, and canonical-manifest provenance before use.
        let tampered = store
            .put(&artifact, br#"{"fixture":"different bytes"}"#)
            .unwrap();
        assert_ne!(tampered.content_digest, honest.content_digest);
        let found = store.identity_candidates(&identity).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].content_digest, honest.content_digest);

        let reopened =
            ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        assert_eq!(reopened.identity_candidates(&identity).unwrap().len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zip_store_and_publication_encoder_agree_byte_for_byte() {
        // Staging publishes a workstation ZIP object directly, so the store
        // encoder and the deterministic packager must never diverge.
        let root = temporary_root("xc-cache-encoder-agreement");
        let _ = fs::remove_dir_all(&root);
        let payload = br#"{"entries":["1.0","0.0","0.0","1.0"],"note":"encoder agreement"}"#;
        let envelope = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "json".to_owned(),
            precision_bits: None,
            scalar_representation: "json".to_owned(),
            dimensions: vec![payload.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "payload.json".to_owned(),
                content_digest: ContentDigest::sha256(payload),
                size_bytes: payload.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let package_path = root.join("package.zip");
        let package = package_canonical_payload_bytes_zip64(
            &envelope,
            "payload.json",
            payload,
            &package_path,
            &xc_core::ResourcePolicy::default(),
            &xc_core::CancellationToken::new(),
        )
        .unwrap();
        let store_bytes = ZipJsonFilesystemCacheStore::encode(payload).unwrap();
        assert_eq!(store_bytes, fs::read(&package_path).unwrap());
        assert_eq!(ContentDigest::sha256(&store_bytes), package.package_digest);
        assert_eq!(package.encoder_profile, CURRENT_DETERMINISTIC_ZIP64_PROFILE);
        let source_path = root.join("payload.json");
        fs::write(&source_path, payload).unwrap();
        let file_package_path = root.join("file-package.zip");
        let file_package = package_canonical_payload_zip64(
            &envelope,
            &[PayloadFileSource {
                normalized_path: "payload.json".to_owned(),
                source_path,
            }],
            &file_package_path,
            &xc_core::ResourcePolicy::default(),
            &xc_core::CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(
            file_package.encoder_profile,
            CURRENT_DETERMINISTIC_ZIP64_PROFILE
        );
        assert_eq!(package.encoder_profile, DETERMINISTIC_ZIP64_PROFILE_V1);
        assert_eq!(fs::read(file_package_path).unwrap(), store_bytes);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zip_store_persists_encoder_provenance_and_legacy_objects_are_decode_only() {
        let root = temporary_root("xc-cache-encoder-provenance");
        let _ = fs::remove_dir_all(&root);
        let store = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let key = ArtifactKey::new("ccm_tau_matrix", "ccm/fixture/profile", b"profile").unwrap();
        let payload = br#"{"profile":"bound"}"#;
        let manifest = store
            .put(
                &draft(key, CacheQuality::Validated, CacheVisibility::Local),
                payload,
            )
            .unwrap();
        assert_eq!(
            manifest
                .tags
                .get(STORAGE_ENCODER_PROFILE_TAG)
                .map(String::as_str),
            Some(CURRENT_DETERMINISTIC_ZIP64_PROFILE)
        );
        assert_eq!(
            store
                .verified_encoded_payload(&manifest)
                .unwrap()
                .unwrap()
                .encoder_profile
                .as_deref(),
            Some(CURRENT_DETERMINISTIC_ZIP64_PROFILE)
        );

        let mut unknown = manifest.clone();
        unknown.tags.insert(
            STORAGE_ENCODER_PROFILE_TAG.to_owned(),
            "unrecognized-test-encoder".to_owned(),
        );
        assert!(store.verified_encoded_payload(&unknown).unwrap().is_none());
        let (decoded, encoded) = store.read_payload_and_encoded(&unknown).unwrap();
        assert_eq!(decoded, payload);
        assert!(
            encoded.is_none(),
            "an unrecognized encoder profile may be decoded but not adopted"
        );

        let mut legacy = manifest;
        legacy.tags.remove(STORAGE_ENCODER_PROFILE_TAG);
        assert!(store.verified_encoded_payload(&legacy).unwrap().is_none());
        let (decoded, encoded) = store.read_payload_and_encoded(&legacy).unwrap();
        assert_eq!(decoded, payload);
        assert!(
            encoded.is_none(),
            "an unprofiled legacy object may be decoded but not adopted for publication"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verified_object_adoption_hashes_the_source_before_exposing_it() {
        let root = temporary_root("xc-cache-verified-object-source-hash");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let store = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let source = root.join("offered.zip");
        fs::write(&source, b"bytes that do not match the offered digest").unwrap();
        let offered_digest = ContentDigest::sha256(b"different bytes");
        let offered = VerifiedEncodedPayload {
            path: source,
            encoding: "zip-json-entry-v1".to_owned(),
            encoder_profile: Some(CURRENT_DETERMINISTIC_ZIP64_PROFILE.to_owned()),
            content_digest: offered_digest.clone(),
            size_bytes: b"bytes that do not match the offered digest".len() as u64,
        };
        let key = ArtifactKey::new("ccm_tau_matrix", "ccm/fixture/adopt", b"adopt").unwrap();
        let mut unsupported = offered.clone();
        unsupported.encoder_profile = Some("unrecognized-test-encoder".to_owned());
        assert!(store
            .put_verified_encoded_payload(
                &draft(key.clone(), CacheQuality::Validated, CacheVisibility::Local,),
                &ContentDigest::sha256(b"logical"),
                b"logical".len() as u64,
                &unsupported,
            )
            .unwrap()
            .is_none());
        let result = store.put_verified_encoded_payload(
            &draft(key, CacheQuality::Validated, CacheVisibility::Local),
            &ContentDigest::sha256(b"logical"),
            b"logical".len() as u64,
            &offered,
        );
        assert!(matches!(result, Err(CacheError::DigestMismatch { .. })));
        assert!(
            !store.inner.object_path(&offered_digest).exists(),
            "unverified bytes must never become visible at a content-addressed destination"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zip_decode_rejects_declared_logical_size_before_inflating() {
        let budget = Arc::new(InMemoryZipBudget::new(1024 * 1024));
        let (store, manifest, root) =
            zip_store_with_object("xc-cache-zip-logical-bound", budget, 1024 * 1024);
        let mut inconsistent = manifest;
        inconsistent.size_bytes -= 1;
        let mut output = Vec::new();
        assert!(matches!(
            store.read_zip_object(&inconsistent, &mut output),
            Err(CacheError::InvalidManifest(_))
        ));
        assert!(
            output.is_empty(),
            "declared-size disagreement must be rejected before decompression writes bytes"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_exact_selects_the_exact_digest_not_the_newest_candidate() {
        let root = temporary_root("xc-cache-resolve-exact");
        let _ = fs::remove_dir_all(&root);
        let store = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let key = ArtifactKey::new("ccm_tau_matrix", "ccm/fixture/exact", b"exact").unwrap();
        let older = store
            .put(
                &draft(key.clone(), CacheQuality::Validated, CacheVisibility::Local),
                br#"{"generation":1}"#,
            )
            .unwrap();
        let newer = store
            .put(
                &draft(key.clone(), CacheQuality::Validated, CacheVisibility::Local),
                br#"{"generation":2}"#,
            )
            .unwrap();
        assert_ne!(older.content_digest, newer.content_digest);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(store),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: version("0.13.0"),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        // A child that recorded the older generation as its dependency must
        // resolve exactly that generation even though a newer one is live.
        let resolved = resolver
            .resolve_exact(
                &key,
                &older.content_digest,
                CacheQuality::Validated,
                &policy,
            )
            .unwrap();
        assert_eq!(resolved.manifest.content_digest, older.content_digest);
        assert_eq!(resolved.payload, br#"{"generation":1}"#);
        let resolved = resolver
            .resolve_exact(
                &key,
                &newer.content_digest,
                CacheQuality::Validated,
                &policy,
            )
            .unwrap();
        assert_eq!(resolved.manifest.content_digest, newer.content_digest);
        // The metadata-only route selects the same candidate and never reads
        // the logical payload.
        let encoded = resolver
            .resolve_exact_encoded(
                &key,
                &older.content_digest,
                CacheQuality::Validated,
                &policy,
            )
            .unwrap()
            .unwrap();
        assert_eq!(encoded.manifest.content_digest, older.content_digest);
        assert_eq!(
            encoded.encoded.content_digest,
            older.objects[0].content_digest
        );
        // A quality floor above the candidate's quality is a miss, matching
        // the prefetch selection rule.
        assert!(matches!(
            resolver.resolve_exact(
                &key,
                &older.content_digest,
                CacheQuality::Certified,
                &policy
            ),
            Err(CacheError::NotFound(_))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn identity_inventory_persists_and_sees_writes_from_another_store_instance() {
        let root = temporary_root("xc-cache-identity-inventory-cross-instance");
        let _ = fs::remove_dir_all(&root);
        let payload = br#"{"fixture":"cross-instance"}"#;
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_tau_matrix".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({"case": "cross-instance"}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "json".to_owned(),
            precision_bits: None,
            scalar_representation: "json".to_owned(),
            dimensions: vec![payload.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "payload.json".to_owned(),
                content_digest: ContentDigest::sha256(payload),
                size_bytes: payload.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let canonical = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm-matrices".to_owned(),
            semantic_key: semantic_key.clone(),
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![ContentDigest::sha256(b"transport")],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"configuration"),
            producer_toolkit_version: version("0.14.1"),
            minimum_reader_version: version("0.14.1"),
            maximum_reader_version: None,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            claim_scope: "identity inventory fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let identity = PayloadDependencyIdentity {
            artifact_family: canonical.artifact_family.clone(),
            semantic_digest: semantic_digest.clone(),
            manifest_digest: canonical.digest().unwrap(),
            payload_digest,
        };
        let key = ArtifactKey {
            kind: semantic_key.artifact_kind.clone(),
            logical_key: "ccm/fixture/cross-instance".to_owned(),
            parameters_digest: semantic_digest.clone(),
        };
        let mut artifact = draft(key, CacheQuality::Validated, CacheVisibility::Local);
        artifact.tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&semantic_key).unwrap(),
        );
        artifact.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&canonical).unwrap(),
        );

        // Instance A memoizes a miss. Instance B (another process in
        // practice) writes the artifact. A's next query must see it without
        // a restart because the inventory file changed under its memo.
        let reader = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        assert!(reader.identity_candidates(&identity).unwrap().is_empty());
        let writer = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let manifest = writer.put(&artifact, payload).unwrap();
        let mut second_canonical = canonical.clone();
        second_canonical.producer_toolkit_version = version("0.14.2");
        let second_identity = PayloadDependencyIdentity {
            artifact_family: second_canonical.artifact_family.clone(),
            semantic_digest: semantic_digest.clone(),
            manifest_digest: second_canonical.digest().unwrap(),
            payload_digest: second_canonical.payload_digest.clone(),
        };
        let mut second_artifact = artifact.clone();
        second_artifact.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&second_canonical).unwrap(),
        );
        writer.put(&second_artifact, payload).unwrap();
        let inventory_path = writer
            .inner
            .identity_inventory_path(&identity.semantic_digest);
        assert!(inventory_path.is_file(), "writes maintain the inventory");
        let found = reader.identity_candidates(&identity).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].content_digest, manifest.content_digest);

        // A cache written before the inventory existed (or by an older
        // writer) is repaired by one bounded scan of the semantic digest.
        fs::remove_dir_all(root.join("identities")).unwrap();
        let legacy = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let found = legacy.identity_candidates(&identity).unwrap();
        assert_eq!(found.len(), 1);
        assert!(
            inventory_path.is_file(),
            "the legacy scan rebuilds the inventory for later queries"
        );
        let inventory: IdentityInventory =
            serde_json::from_slice(&fs::read(&inventory_path).unwrap()).unwrap();
        assert_eq!(inventory.entries.len(), 2);
        assert!(inventory
            .entries
            .iter()
            .any(|entry| entry.content_digest == manifest.content_digest));

        // Read-only stores cannot repair the persistent inventory. They must
        // therefore rescan after an inventory miss: finding one identity may
        // not mark the shared semantic digest as exhausted and hide a second
        // retained manifest under that digest.
        fs::remove_dir_all(root.join("identities")).unwrap();
        let read_only =
            ZipJsonFilesystemCacheStore::new("local", &root, false, CacheVisibility::Local);
        assert_eq!(read_only.identity_candidates(&identity).unwrap().len(), 1);
        assert_eq!(
            read_only
                .identity_candidates(&second_identity)
                .unwrap()
                .len(),
            1
        );
        assert!(!inventory_path.exists());

        // A drive-qualified path is absolute on Windows even though it does
        // not begin with a slash. Inventory data is untrusted and may never
        // escape the cache root on any platform.
        fs::create_dir_all(inventory_path.parent().unwrap()).unwrap();
        fs::write(
            &inventory_path,
            serde_json::to_vec_pretty(&IdentityInventory {
                schema_version: 1,
                entries: vec![IdentityInventoryEntry {
                    artifact_family: identity.artifact_family.clone(),
                    manifest_digest: identity.manifest_digest.clone(),
                    payload_digest: identity.payload_digest.clone(),
                    kind: semantic_key.artifact_kind.clone(),
                    logical_key: "ccm/fixture/cross-instance".to_owned(),
                    content_digest: manifest.content_digest.clone(),
                    manifest_path: "C:/outside-cache/manifest.json".to_owned(),
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let unsafe_reader =
            ZipJsonFilesystemCacheStore::new("local", &root, false, CacheVisibility::Local);
        assert!(matches!(
            unsafe_reader.identity_candidates(&identity),
            Err(CacheError::InvalidManifest(_))
        ));
        let _ = fs::remove_dir_all(root);
    }

    fn zip_store_with_object(
        name: &str,
        budget: Arc<InMemoryZipBudget>,
        single_pass_object_limit: u64,
    ) -> (ZipJsonFilesystemCacheStore, ArtifactManifest, PathBuf) {
        let root = temporary_root(name);
        let _ = fs::remove_dir_all(&root);
        let store = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local)
            .with_in_memory_zip_limits(budget, single_pass_object_limit);
        let key = ArtifactKey::new("ccm_tau_matrix", "ccm/fixture/zip-memory", b"zip").unwrap();
        let manifest = store
            .put(
                &draft(key, CacheQuality::Validated, CacheVisibility::Local),
                br#"{"zip":"memory budget fixture"}"#,
            )
            .unwrap();
        (store, manifest, root)
    }

    #[test]
    fn zip_objects_above_the_single_pass_limit_stream() {
        let budget = Arc::new(InMemoryZipBudget::new(1024 * 1024));
        let (store, manifest, root) = zip_store_with_object(
            "xc-cache-zip-single-pass-limit",
            budget.clone(),
            1024 * 1024,
        );
        let object_bytes = manifest.objects[0].size_bytes;
        let mut payload = Vec::new();
        assert_eq!(
            store.read_zip_object(&manifest, &mut payload).unwrap(),
            ZipReadStrategy::SinglePass
        );
        assert_eq!(payload, br#"{"zip":"memory budget fixture"}"#);
        assert_eq!(
            budget.in_use(),
            0,
            "the reservation is released after the read"
        );

        let store = store.with_in_memory_zip_limits(budget.clone(), object_bytes - 1);
        let mut streamed = Vec::new();
        assert_eq!(
            store.read_zip_object(&manifest, &mut streamed).unwrap(),
            ZipReadStrategy::Streamed
        );
        assert_eq!(streamed, payload, "both strategies decode the same bytes");
        assert_eq!(budget.in_use(), 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn single_pass_zip_read_is_bounded_by_the_declared_object_size() {
        let budget = Arc::new(InMemoryZipBudget::new(1024 * 1024));
        let (store, manifest, root) =
            zip_store_with_object("xc-cache-zip-bounded-read", budget.clone(), 1024 * 1024);
        let object = &manifest.objects[0];
        let object_path = store.inner.object_path(&object.content_digest);
        let mut enlarged = fs::read(&object_path).unwrap();
        enlarged.extend(std::iter::repeat_n(0u8, 1024 * 1024));
        fs::write(&object_path, enlarged).unwrap();

        assert!(matches!(
            store.read_zip_object(&manifest, &mut Vec::new()),
            Err(CacheError::InvalidManifest(_))
        ));
        assert_eq!(budget.in_use(), 0, "failed reads release their allowance");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_single_pass_reads_never_exceed_the_aggregate_allowance() {
        let budget = Arc::new(InMemoryZipBudget::new(350));
        let attempts = 8;
        let barrier = std::sync::Barrier::new(attempts);
        let granted = std::sync::atomic::AtomicUsize::new(0);
        let peak = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|scope| {
            for _ in 0..attempts {
                scope.spawn(|| {
                    barrier.wait();
                    let reservation = budget.try_reserve(100);
                    if reservation.is_some() {
                        granted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    peak.fetch_max(budget.in_use(), std::sync::atomic::Ordering::SeqCst);
                    // Hold every successful reservation until all attempts
                    // have been made, so the grants really overlap.
                    barrier.wait();
                    drop(reservation);
                });
            }
        });
        let granted = granted.load(std::sync::atomic::Ordering::SeqCst);
        assert!((1..=3).contains(&granted), "granted {granted} of 8");
        assert!(peak.load(std::sync::atomic::Ordering::SeqCst) <= 350);
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn aggregate_allowance_is_shared_across_store_instances() {
        // Default construction shares the one process-wide allowance.
        let default_store = ZipJsonFilesystemCacheStore::new(
            "local",
            temporary_root("xc-cache-zip-global-budget"),
            false,
            CacheVisibility::Local,
        );
        assert!(Arc::ptr_eq(
            &default_store.in_memory_zip_budget,
            &InMemoryZipBudget::global()
        ));

        // Two instances over one private allowance: a reservation held on
        // behalf of the first forces the second to stream.
        let budget = Arc::new(InMemoryZipBudget::new(4096));
        let (first, manifest, root) =
            zip_store_with_object("xc-cache-zip-shared-budget", budget.clone(), 4096);
        let second =
            ZipJsonFilesystemCacheStore::new("local", &root, false, CacheVisibility::Local)
                .with_in_memory_zip_limits(budget.clone(), 4096);
        assert!(Arc::ptr_eq(
            &first.in_memory_zip_budget,
            &second.in_memory_zip_budget
        ));
        let held = budget.try_reserve(4096 - 1).unwrap();
        let mut payload = Vec::new();
        assert_eq!(
            second.read_zip_object(&manifest, &mut payload).unwrap(),
            ZipReadStrategy::Streamed
        );
        drop(held);
        assert_eq!(
            second.read_zip_object(&manifest, &mut Vec::new()).unwrap(),
            ZipReadStrategy::SinglePass
        );
        assert_eq!(budget.in_use(), 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_byte_limit_overrides_fall_back_to_the_default_deterministically() {
        assert_eq!(parse_byte_limit(None, 7), 7);
        assert_eq!(parse_byte_limit(Some(""), 7), 7);
        assert_eq!(parse_byte_limit(Some("   "), 7), 7);
        assert_eq!(parse_byte_limit(Some("abc"), 7), 7);
        assert_eq!(parse_byte_limit(Some("0"), 7), 7);
        assert_eq!(parse_byte_limit(Some("-1"), 7), 7);
        assert_eq!(parse_byte_limit(Some("1.5"), 7), 7);
        assert_eq!(parse_byte_limit(Some("18446744073709551616"), 7), 7);
        assert_eq!(parse_byte_limit(Some(" 4096 "), 7), 4096);
        assert_eq!(parse_byte_limit(Some("18446744073709551615"), 7), 7);
        assert_eq!(
            DEFAULT_SINGLE_PASS_ZIP_OBJECT_BYTES,
            90 * 1024 * 1024,
            "the single-pass limit tracks the split part size"
        );
    }

    #[test]
    fn a_failed_reservation_streams_immediately_instead_of_waiting() {
        let budget = Arc::new(InMemoryZipBudget::new(10));
        let held = budget.try_reserve(10).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let probe = budget.clone();
        std::thread::spawn(move || {
            let outcome = probe.try_reserve(1).is_some();
            let _ = sender.send(outcome);
        });
        let outcome = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reservation must return at once, never block");
        assert!(!outcome, "a saturated allowance refuses rather than waits");

        let (store, manifest, root) =
            zip_store_with_object("xc-cache-zip-no-wait", budget.clone(), 1024 * 1024);
        let mut payload = Vec::new();
        assert_eq!(
            store.read_zip_object(&manifest, &mut payload).unwrap(),
            ZipReadStrategy::Streamed
        );
        assert_eq!(payload, br#"{"zip":"memory budget fixture"}"#);
        drop(held);
        assert_eq!(budget.in_use(), 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn filesystem_store_maintains_a_direct_continuation_inventory() {
        let root = temporary_root("xc-cache-continuation-inventory");
        let _ = fs::remove_dir_all(&root);
        let store = FilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        for n_modes in [10, 20] {
            let logical_key =
                format!("ccm/weil-eigenpair/13/{n_modes}/729/even/shift_invert_krylov");
            let semantic_key = SemanticKeyEnvelope {
                schema_version: 1,
                artifact_kind: "ccm_weil_eigenpair".to_owned(),
                mathematical_semantics_version: "fixture".to_owned(),
                resolved_mathematical_parameters: serde_json::json!({
                    "lambda_squared": "13",
                    "n_modes": n_modes,
                    "precision_bits": 729,
                    "force_even": true,
                    "eigenstate_route": "shift_invert_krylov"
                }),
                normalization: Some("fixture".to_owned()),
                target: Some("smallest_weil_form_eigenpair".to_owned()),
                subspace: Some("even".to_owned()),
                source_data_identities: BTreeMap::new(),
                algorithm_semantics: Some("fixture".to_owned()),
            };
            let key = ArtifactKey {
                kind: "ccm_weil_eigenpair".to_owned(),
                logical_key,
                parameters_digest: semantic_key.digest().unwrap(),
            };
            let mut artifact = draft(key, CacheQuality::Validated, CacheVisibility::Local);
            artifact.tags.insert(
                SEMANTIC_KEY_MANIFEST_TAG.to_owned(),
                serde_json::to_string(&semantic_key).unwrap(),
            );
            store.put(&artifact, b"{}").unwrap();
        }
        let query = CcmEigenpairContinuationQuery {
            lambda_squared: "13".to_owned(),
            maximum_n_modes: 30,
            precision_bits: 729,
            force_even: true,
        };
        let keys = store.ccm_eigenpair_continuation_keys(&query, 1).unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys[0].logical_key.contains("/20/"));
        assert!(root.join(query.repository_path().unwrap()).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reader_storage_streams_fixed_size_chunks() {
        let root = temporary_root("xc-cache-reader");
        let _ = fs::remove_dir_all(&root);
        let store = FilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let key = ArtifactKey::new("matrix", "large", br#"{"n":8}"#).unwrap();
        let payload = b"abcdefghij";
        let mut reader = &payload[..];
        let manifest = store
            .put_reader(
                &draft(key, CacheQuality::Validated, CacheVisibility::Local),
                &mut reader,
                4,
            )
            .unwrap();
        assert_eq!(manifest.objects.len(), 3);
        assert_eq!(store.read_payload(&manifest).unwrap(), payload);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zip_json_store_reads_entry_without_extracting_logical_payload() {
        let root = temporary_root("xc-cache-zip-json");
        let _ = fs::remove_dir_all(&root);
        let store = ZipJsonFilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let key = ArtifactKey::new("matrix", "zip", br#"{"n":8}"#).unwrap();
        let payload = br#"{"entries":["1.0","0.0","0.0","1.0"]}"#;
        let manifest = store
            .put(
                &draft(key, CacheQuality::Validated, CacheVisibility::Local),
                payload,
            )
            .unwrap();
        assert_eq!(manifest.content_digest, ContentDigest::sha256(payload));
        assert_eq!(store.read_payload(&manifest).unwrap(), payload);
        assert_eq!(manifest.objects.len(), 1);
        assert_ne!(manifest.objects[0].content_digest, manifest.content_digest);
        assert!(!root.join("payload.json").exists());
        let object_path = store.inner.object_path(&manifest.objects[0].content_digest);
        let archive = zip::ZipArchive::new(fs::File::open(object_path).unwrap()).unwrap();
        assert_eq!(archive.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zip_json_store_adopts_a_verified_archive_without_recompression() {
        let root = temporary_root("xc-cache-zip-json-adopt");
        let _ = fs::remove_dir_all(&root);
        let source = ZipJsonFilesystemCacheStore::new(
            "source",
            root.join("source"),
            true,
            CacheVisibility::Local,
        );
        let destination = ZipJsonFilesystemCacheStore::new(
            "destination",
            root.join("destination"),
            true,
            CacheVisibility::Local,
        );
        let key = ArtifactKey::new("matrix", "zip-adopt", br#"{"n":8}"#).unwrap();
        let payload = br#"{"entries":["1.0","0.0","0.0","1.0"]}"#;
        let source_manifest = source
            .put(
                &draft(key.clone(), CacheQuality::Validated, CacheVisibility::Local),
                payload,
            )
            .unwrap();
        let encoded = source
            .verified_encoded_payload(&source_manifest)
            .unwrap()
            .unwrap();
        let adopted = destination
            .put_verified_encoded_payload(
                &draft(key, CacheQuality::Validated, CacheVisibility::Local),
                &source_manifest.content_digest,
                source_manifest.size_bytes,
                &encoded,
            )
            .unwrap()
            .unwrap();
        assert_eq!(adopted.content_digest, source_manifest.content_digest);
        assert_eq!(adopted.objects, source_manifest.objects);
        assert_eq!(destination.read_payload(&adopted).unwrap(), payload);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn multiple_versions_of_one_logical_key_are_indexed() {
        let root = temporary_root("xc-cache-index-update");
        let _ = fs::remove_dir_all(&root);
        let store = FilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let key = ArtifactKey::new("tau", "same", br#"{"n":120}"#).unwrap();
        store
            .put(
                &draft(key.clone(), CacheQuality::Validated, CacheVisibility::Local),
                b"first",
            )
            .unwrap();
        store
            .put(
                &draft(key.clone(), CacheQuality::Certified, CacheVisibility::Local),
                b"second",
            )
            .unwrap();
        let candidates = store.candidates(&key).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].quality, CacheQuality::Certified);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn filesystem_overlay_respects_quality_and_precedence() {
        let root = temporary_root("xc-cache-overlay");
        let _ = fs::remove_dir_all(&root);
        let local =
            FilesystemCacheStore::new("local", root.join("local"), true, CacheVisibility::Local);
        let public =
            FilesystemCacheStore::new("public", root.join("public"), true, CacheVisibility::Public);
        let key = ArtifactKey::new("tau", "lambda13-n120", br#"{"n":120}"#).unwrap();
        local
            .put(
                &draft(key.clone(), CacheQuality::Staged, CacheVisibility::Local),
                b"local",
            )
            .unwrap();
        public
            .put(
                &draft(
                    key.clone(),
                    CacheQuality::Published,
                    CacheVisibility::Public,
                ),
                b"public",
            )
            .unwrap();
        let resolver = CacheResolver::new(vec![
            CacheLayer {
                precedence: 0,
                store: Box::new(local),
            },
            CacheLayer {
                precedence: 10,
                store: Box::new(public),
            },
        ]);
        let policy = CachePolicy {
            current_toolkit_version: version("0.13.0"),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local, CacheVisibility::Public],
        };
        let resolved = resolver.resolve(&key, &policy).unwrap();
        assert_eq!(resolved.layer_name, "public");
        assert_eq!(resolved.payload, b"public");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dependency_closure_requires_exact_compatible_artifact() {
        let root = temporary_root("xc-cache-dependencies");
        let _ = fs::remove_dir_all(&root);
        let store = FilesystemCacheStore::new("local", &root, true, CacheVisibility::Local);
        let dependency_key = ArtifactKey::new("gl", "n=64", br#"{"n":64}"#).unwrap();
        let dependency_manifest = store
            .put(
                &draft(
                    dependency_key.clone(),
                    CacheQuality::Certified,
                    CacheVisibility::Local,
                ),
                b"nodes-and-weights",
            )
            .unwrap();
        let root_key = ArtifactKey::new("tau", "n=120", br#"{"n":120}"#).unwrap();
        let mut root_draft = draft(
            root_key.clone(),
            CacheQuality::Validated,
            CacheVisibility::Local,
        );
        root_draft.dependencies.push(DependencyRef {
            key: dependency_key,
            content_digest: dependency_manifest.content_digest.clone(),
            required_quality: CacheQuality::Certified,
        });
        let root_manifest = store.put(&root_draft, b"tau").unwrap();
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(store),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: version("0.13.0"),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let dependencies = resolver
            .validate_dependency_closure(&root_manifest, &policy)
            .unwrap();
        assert_eq!(dependencies.len(), 1);
        assert_eq!(
            dependencies[0].content_digest,
            dependency_manifest.content_digest
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn public_promotion_requires_certified_source_and_review() {
        let policy = CachePromotionPolicy {
            minimum_unique_approvals: 1,
            require_certified_before_publication: true,
            allow_private_to_public: true,
        };
        let mut request = CachePromotionRequest {
            source_manifest_digest: ContentDigest::sha256(b"manifest"),
            source_quality: CacheQuality::CrossChecked,
            target_quality: CacheQuality::Published,
            source_visibility: CacheVisibility::Private,
            target_visibility: CacheVisibility::Public,
            reviews: vec![PromotionReview {
                reviewer: "reviewer@example.org".to_owned(),
                approved: true,
                evidence_digest: Some(ContentDigest::sha256(b"evidence")),
                notes: "reviewed".to_owned(),
            }],
        };
        assert!(policy.validate_request(&request).is_err());
        request.source_quality = CacheQuality::Certified;
        assert!(policy.validate_request(&request).is_ok());
    }

    #[test]
    fn shard_planner_uses_full_safe_repository_capacity() {
        let registry = CacheRepositoryRegistry {
            schema_version: 1,
            shards: vec![
                RepositoryShard {
                    id: "tau-000".to_owned(),
                    repository: "example-org/public-shard-a".to_owned(),
                    visibility: CacheVisibility::Public,
                    artifact_kinds: vec!["tau".to_owned()],
                    reachable_payload_bytes: 62_000_000_000,
                    estimated_history_bytes: 2_000_000_000,
                    safe_payload_limit_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
                    writable: true,
                },
                RepositoryShard {
                    id: "tau-001".to_owned(),
                    repository: "example-org/public-shard-b".to_owned(),
                    visibility: CacheVisibility::Public,
                    artifact_kinds: vec!["tau".to_owned()],
                    reachable_payload_bytes: 0,
                    estimated_history_bytes: 0,
                    safe_payload_limit_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
                    writable: true,
                },
            ],
        };
        let selected = registry
            .select_write_shard("tau", CacheVisibility::Public, 30_000_000_000, 0)
            .unwrap();
        assert_eq!(selected.id, "tau-000");
    }
}

// ===========================================================================
// GitHub repository registry descriptors and publication planning
// ===========================================================================

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GitHubRepositoryEndpoint {
    pub shard_id: String,
    pub owner: String,
    pub repository: String,
    pub branch: String,
    pub visibility: CacheVisibility,
    pub enabled_for_read: bool,
    pub enabled_for_write: bool,
    pub clone_via_ssh: bool,
}

impl GitHubRepositoryEndpoint {
    pub fn validate(&self) -> Result<(), CacheError> {
        for (name, value) in [
            ("shard_id", &self.shard_id),
            ("owner", &self.owner),
            ("repository", &self.repository),
            ("branch", &self.branch),
        ] {
            if value.trim().is_empty() {
                return Err(CacheError::InvalidManifest(format!(
                    "GitHub repository endpoint {name} must be nonempty"
                )));
            }
        }
        Ok(())
    }

    pub fn https_clone_url(&self) -> String {
        format!("https://github.com/{}/{}.git", self.owner, self.repository)
    }

    pub fn ssh_clone_url(&self) -> String {
        format!("git@github.com:{}/{}.git", self.owner, self.repository)
    }

    pub fn preferred_clone_url(&self) -> String {
        if self.clone_via_ssh {
            self.ssh_clone_url()
        } else {
            self.https_clone_url()
        }
    }

    pub fn raw_content_url(&self, repository_relative_path: &str) -> Result<String, CacheError> {
        self.validate()?;
        let path = repository_relative_path.trim_matches('/');
        if path.is_empty() || path.split('/').any(|component| component == "..") {
            return Err(CacheError::InvalidManifest(
                "raw GitHub path must be nonempty and may not contain '..'".to_owned(),
            ));
        }
        Ok(format!(
            "https://raw.githubusercontent.com/{}/{}/{}/{}",
            self.owner, self.repository, self.branch, path
        ))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheNetworkRegistry {
    pub schema_version: u32,
    pub repositories: Vec<GitHubRepositoryEndpoint>,
}

impl CacheNetworkRegistry {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0 {
            return Err(CacheError::InvalidManifest(
                "cache network registry schema_version must be positive".to_owned(),
            ));
        }
        let mut shard_ids = BTreeSet::new();
        for endpoint in &self.repositories {
            endpoint.validate()?;
            if !shard_ids.insert(endpoint.shard_id.clone()) {
                return Err(CacheError::InvalidManifest(format!(
                    "duplicate cache network shard id {:?}",
                    endpoint.shard_id
                )));
            }
        }
        Ok(())
    }

    pub fn endpoint_for_shard(&self, shard_id: &str) -> Option<&GitHubRepositoryEndpoint> {
        self.repositories
            .iter()
            .find(|endpoint| endpoint.shard_id == shard_id)
    }

    pub fn readable_by_visibility(
        &self,
        visibility: CacheVisibility,
    ) -> impl Iterator<Item = &GitHubRepositoryEndpoint> {
        self.repositories
            .iter()
            .filter(move |endpoint| endpoint.enabled_for_read && endpoint.visibility == visibility)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GitHubPublicationPlan {
    pub shard_id: String,
    pub repository: String,
    pub branch: String,
    pub artifact_kind: String,
    pub visibility: CacheVisibility,
    pub payload_bytes: u64,
    pub estimated_history_bytes: u64,
    pub projected_repository_bytes: u64,
    pub safe_repository_limit_bytes: u64,
    pub requires_pull_request: bool,
    pub notes: Vec<String>,
}

/// Selects a safe GitHub shard for a proposed cache publication.
///
/// # Mathematical semantics
/// Matches artifact family and visibility, then accounts for payload, projected
/// history, and the configured safe repository ceiling without changing the
/// artifact's semantic or payload identity.
///
/// # Precision
/// The planner never transforms numerical content. Scalar backend and precision
/// remain properties of the artifact manifest supplied to later stages.
///
/// # Failure states
/// Invalid registries, disabled or mismatched endpoints, arithmetic overflow,
/// and insufficient shard capacity return `CacheError`; no remote mutation is
/// attempted.
///
/// # Assurance and validity
/// The output proves routing and capacity eligibility only. Validation,
/// authorization, receipt finalization, and any scientific certification are
/// independent gates.
///
/// # Cache effects
/// This function is a side-effect-free dry-run planner. It neither clones a
/// repository nor uploads data.
///
/// # Example
/// Compiled example: `crates/xc-cache/examples/github_publication_plan.rs`.
pub fn plan_github_publication(
    capacity_registry: &CacheRepositoryRegistry,
    network_registry: &CacheNetworkRegistry,
    artifact_kind: &str,
    visibility: CacheVisibility,
    payload_bytes: u64,
    estimated_history_bytes: u64,
    requires_pull_request: bool,
) -> Result<GitHubPublicationPlan, CacheError> {
    if visibility == CacheVisibility::Public && artifact_kind_is_private_only(artifact_kind) {
        return Err(CacheError::PermissionDenied(format!(
            "artifact kind {artifact_kind:?} is private-only"
        )));
    }
    capacity_registry.shards.iter().try_for_each(|shard| {
        if shard.safe_payload_limit_bytes > GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES {
            return Err(CacheError::InvalidManifest(format!(
                "shard {:?} exceeds the approved GitHub safe threshold",
                shard.id
            )));
        }
        Ok(())
    })?;
    network_registry.validate()?;
    let shard = capacity_registry.select_write_shard(
        artifact_kind,
        visibility,
        payload_bytes,
        estimated_history_bytes,
    )?;
    let endpoint = network_registry
        .endpoint_for_shard(&shard.id)
        .ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "selected capacity shard {:?} has no GitHub endpoint",
                shard.id
            ))
        })?;
    if !endpoint.enabled_for_write {
        return Err(CacheError::ReadOnlyLayer(endpoint.shard_id.clone()));
    }
    if endpoint.visibility != visibility {
        return Err(CacheError::InvalidManifest(format!(
            "capacity and network visibility disagree for shard {:?}",
            shard.id
        )));
    }
    let projected = shard.projected_total_bytes(payload_bytes, estimated_history_bytes);
    Ok(GitHubPublicationPlan {
        shard_id: shard.id.clone(),
        repository: endpoint.https_clone_url(),
        branch: endpoint.branch.clone(),
        artifact_kind: artifact_kind.to_owned(),
        visibility,
        payload_bytes,
        estimated_history_bytes,
        projected_repository_bytes: projected,
        safe_repository_limit_bytes: shard.safe_payload_limit_bytes,
        requires_pull_request,
        notes: vec![
            "repository capacity includes reachable payload and estimated Git history".to_owned(),
            "large immutable objects should be finalized before their first binary commit"
                .to_owned(),
            "publication changes artifact visibility only after review policy passes".to_owned(),
        ],
    })
}

#[cfg(test)]
mod github_registry_tests {
    use super::*;

    #[test]
    fn public_repository_plan_respects_capacity_and_endpoint() {
        let capacity = CacheRepositoryRegistry {
            schema_version: 1,
            shards: vec![RepositoryShard {
                id: "tau-public-001".to_owned(),
                repository: "example-org/public-shard-a".to_owned(),
                visibility: CacheVisibility::Public,
                artifact_kinds: vec!["ccm_tau_matrix".to_owned()],
                reachable_payload_bytes: 62_000_000_000,
                estimated_history_bytes: 1_000_000_000,
                safe_payload_limit_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
                writable: true,
            }],
        };
        let network = CacheNetworkRegistry {
            schema_version: 1,
            repositories: vec![GitHubRepositoryEndpoint {
                shard_id: "tau-public-001".to_owned(),
                owner: "TeamXcelerator".to_owned(),
                repository: "public-shard-a".to_owned(),
                branch: "main".to_owned(),
                visibility: CacheVisibility::Public,
                enabled_for_read: true,
                enabled_for_write: true,
                clone_via_ssh: true,
            }],
        };
        let plan = plan_github_publication(
            &capacity,
            &network,
            "ccm_tau_matrix",
            CacheVisibility::Public,
            5_000_000_000,
            500_000_000,
            true,
        )
        .unwrap();
        assert_eq!(plan.shard_id, "tau-public-001");
        assert!(plan.projected_repository_bytes < GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES);
        assert!(plan.requires_pull_request);

        let error = plan_github_publication(
            &capacity,
            &network,
            "ccm_target_distance",
            CacheVisibility::Public,
            1,
            0,
            true,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::PermissionDenied(_)));
    }

    #[test]
    fn raw_path_rejects_parent_traversal() {
        let endpoint = GitHubRepositoryEndpoint {
            shard_id: "x".to_owned(),
            owner: "TeamXcelerator".to_owned(),
            repository: "cache".to_owned(),
            branch: "main".to_owned(),
            visibility: CacheVisibility::Public,
            enabled_for_read: true,
            enabled_for_write: false,
            clone_via_ssh: false,
        };
        assert!(endpoint.raw_content_url("../secret").is_err());
    }
}

#[cfg(test)]
mod cache_acceptance_tests {
    use super::*;

    fn manifest_for_version(version: &str, quality: CacheQuality) -> ArtifactManifest {
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new("fixture", "fixture-key", b"{}").unwrap(),
            content_digest: ContentDigest::sha256(b"payload"),
            size_bytes: 7,
            objects: vec![CacheObjectRef {
                content_digest: ContentDigest::sha256(b"payload"),
                size_bytes: 7,
            }],
            created_unix_seconds: 0,
            producer_toolkit_version: ToolkitVersion::parse(version).unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.10.0").unwrap(),
            maximum_reader_version: None,
            quality,
            visibility: CacheVisibility::Private,
            immutable: true,
            dependencies: Vec::new(),
            tags: BTreeMap::new(),
            provenance_digest: None,
        }
    }

    #[test]
    fn assessment_rejects_insufficient_quality_with_reason() {
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_quality: CacheQuality::Certified,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Private],
        };
        let decision = policy.assess(&manifest_for_version("0.13.0", CacheQuality::Validated));
        assert!(!decision.accepted);
        assert!(decision
            .reasons
            .iter()
            .any(|reason| reason.contains("below required")));
    }
}
