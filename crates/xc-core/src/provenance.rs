use crate::config_resolution::{canonical_json, sha256_hex};
use crate::{ConfigDigest, ConfigError, Reproducibility};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ExecutionFingerprintDigest(pub String);

impl ExecutionFingerprintDigest {
    pub fn is_sha256(&self) -> bool {
        self.0.len() == 64
            && self
                .0
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionFingerprint {
    pub working_precision_bits: u32,
    pub guard_bits: u32,
    pub rounding_policy: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadPolicyFingerprint {
    pub thread_count: usize,
    pub scheduling_policy: String,
    pub reduction_policy: String,
}

pub const DETERMINISTIC_REDUCTION_SCHEDULING_V1: &str = "rayon-indexed-fixed-chunks-v1";
pub const DETERMINISTIC_REDUCTION_ALGORITHM_V1: &str = "deterministic-indexed-chunks-pairwise-v1";

/// Identity-bearing controls for the canonical parallel reduction tree.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeterministicReductionPolicy {
    pub chunk_elements: usize,
}

impl DeterministicReductionPolicy {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.chunk_elements == 0 {
            return Err(ConfigError::new(
                "deterministic reduction chunk_elements must be positive",
            ));
        }
        Ok(())
    }

    pub fn fingerprint_name(&self) -> String {
        format!(
            "{DETERMINISTIC_REDUCTION_ALGORITHM_V1}:chunk_elements={}",
            self.chunk_elements
        )
    }
}

impl Default for DeterministicReductionPolicy {
    fn default() -> Self {
        Self {
            chunk_elements: 4096,
        }
    }
}

/// Complete identity required by a byte-reproducibility claim.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionFingerprint {
    pub schema_version: u32,
    pub toolkit_revision: String,
    pub dependency_revisions: BTreeMap<String, String>,
    pub compiler: String,
    pub target_triple: String,
    pub native_libraries: BTreeMap<String, String>,
    pub scalar_backend: String,
    pub scalar_backend_version: String,
    pub precision: PrecisionFingerprint,
    pub algorithm_semantics_versions: BTreeMap<String, String>,
    pub cpu_feature_policy: String,
    pub thread_policy: ThreadPolicyFingerprint,
    pub feature_flags: BTreeSet<String>,
    pub effective_configuration_digest: ConfigDigest,
    pub resolved_resource_policy_digest: ConfigDigest,
    pub reproducibility: Reproducibility,
}

impl ExecutionFingerprint {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version == 0 {
            return Err(ConfigError::new(
                "execution fingerprint schema_version must be positive",
            ));
        }
        for (name, value) in [
            ("toolkit_revision", &self.toolkit_revision),
            ("compiler", &self.compiler),
            ("target_triple", &self.target_triple),
            ("scalar_backend", &self.scalar_backend),
            ("scalar_backend_version", &self.scalar_backend_version),
            ("rounding_policy", &self.precision.rounding_policy),
            ("cpu_feature_policy", &self.cpu_feature_policy),
            ("scheduling_policy", &self.thread_policy.scheduling_policy),
            ("reduction_policy", &self.thread_policy.reduction_policy),
        ] {
            if value.trim().is_empty() {
                return Err(ConfigError::new(format!(
                    "execution fingerprint {name} must be nonempty"
                )));
            }
        }
        for (description, values) in [
            ("dependency revision", &self.dependency_revisions),
            ("native library", &self.native_libraries),
            ("algorithm semantics", &self.algorithm_semantics_versions),
        ] {
            if values
                .iter()
                .any(|(name, version)| name.trim().is_empty() || version.trim().is_empty())
            {
                return Err(ConfigError::new(format!(
                    "execution fingerprint contains an empty {description} identity"
                )));
            }
        }
        if self.precision.working_precision_bits < 32 {
            return Err(ConfigError::new(
                "execution fingerprint precision must be at least 32 bits",
            ));
        }
        if self.thread_policy.thread_count == 0 {
            return Err(ConfigError::new(
                "execution fingerprint thread count must be positive",
            ));
        }
        if self.reproducibility == Reproducibility::Bitwise {
            if self.dependency_revisions.is_empty() {
                return Err(ConfigError::new(
                    "bitwise execution fingerprint requires dependency-lock/revision identities",
                ));
            }
            if self.algorithm_semantics_versions.is_empty() {
                return Err(ConfigError::new(
                    "bitwise execution fingerprint requires algorithm-semantics identities",
                ));
            }
            if self.scalar_backend.to_ascii_lowercase().contains("mpfr")
                && self.native_libraries.is_empty()
            {
                return Err(ConfigError::new(
                    "bitwise MPFR fingerprint requires native-library versions",
                ));
            }
        }
        if !self.effective_configuration_digest.is_sha256()
            || !self.resolved_resource_policy_digest.is_sha256()
        {
            return Err(ConfigError::new(
                "execution fingerprint configuration digests must be lowercase SHA-256",
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ExecutionFingerprintDigest, ConfigError> {
        self.validate()?;
        let value = serde_json::to_value(self).map_err(|error| {
            ConfigError::new(format!(
                "execution fingerprint serialization failed: {error}"
            ))
        })?;
        let canonical = canonical_json(&value).map_err(|error| {
            ConfigError::new(format!(
                "execution fingerprint canonicalization failed: {error}"
            ))
        })?;
        Ok(ExecutionFingerprintDigest(sha256_hex(canonical.as_bytes())))
    }

    pub fn comparison_plan(
        &self,
        other: &Self,
    ) -> Result<ReproducibilityComparisonPlan, ConfigError> {
        let left = self.digest()?;
        let right = other.digest()?;
        let fingerprints_match = left == right;
        let byte_identity_permitted = fingerprints_match
            && self.reproducibility == Reproducibility::Bitwise
            && other.reproducibility == Reproducibility::Bitwise;
        Ok(ReproducibilityComparisonPlan {
            left_fingerprint: left,
            right_fingerprint: right,
            fingerprints_match,
            byte_identity_permitted,
            required_comparison: if byte_identity_permitted {
                CrossFingerprintComparison::ByteIdentity
            } else {
                CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure
            },
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossFingerprintComparison {
    ByteIdentity,
    DeclaredNumericalEquivalenceOrEnclosure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReproducibilityComparisonPlan {
    pub left_fingerprint: ExecutionFingerprintDigest,
    pub right_fingerprint: ExecutionFingerprintDigest,
    pub fingerprints_match: bool,
    pub byte_identity_permitted: bool,
    pub required_comparison: CrossFingerprintComparison,
}

/// Canonical saved evidence for one deterministic scalar reduction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReproducibleReductionArtifact {
    pub schema_version: u32,
    pub execution_fingerprint_digest: ExecutionFingerprintDigest,
    pub policy: DeterministicReductionPolicy,
    pub scalar_backend: String,
    pub precision_bits: u32,
    pub scalar_encoding: String,
    pub value: String,
    pub payload_sha256: String,
}

#[derive(Serialize)]
struct ReproducibleReductionPayload<'a> {
    schema_version: u32,
    execution_fingerprint_digest: &'a ExecutionFingerprintDigest,
    policy: &'a DeterministicReductionPolicy,
    scalar_backend: &'a str,
    precision_bits: u32,
    scalar_encoding: &'a str,
    value: &'a str,
}

impl ReproducibleReductionArtifact {
    pub fn new(
        fingerprint: &ExecutionFingerprint,
        policy: DeterministicReductionPolicy,
        scalar_encoding: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, ConfigError> {
        fingerprint.validate()?;
        policy.validate()?;
        if fingerprint.reproducibility != Reproducibility::Bitwise {
            return Err(ConfigError::new(
                "a reproducible reduction artifact requires bitwise reproducibility",
            ));
        }
        if fingerprint.thread_policy.scheduling_policy != DETERMINISTIC_REDUCTION_SCHEDULING_V1
            || fingerprint.thread_policy.reduction_policy != policy.fingerprint_name()
        {
            return Err(ConfigError::new(
                "execution fingerprint does not match the deterministic reduction policy",
            ));
        }
        let scalar_encoding = scalar_encoding.into();
        let value = value.into();
        if scalar_encoding.trim().is_empty() || value.trim().is_empty() {
            return Err(ConfigError::new(
                "reproducible reduction scalar encoding and value must be nonempty",
            ));
        }
        let mut artifact = Self {
            schema_version: 1,
            execution_fingerprint_digest: fingerprint.digest()?,
            policy,
            scalar_backend: fingerprint.scalar_backend.clone(),
            precision_bits: fingerprint.precision.working_precision_bits,
            scalar_encoding,
            value,
            payload_sha256: String::new(),
        };
        artifact.payload_sha256 = artifact.compute_payload_sha256()?;
        Ok(artifact)
    }

    pub fn verify(&self, fingerprint: &ExecutionFingerprint) -> Result<(), ConfigError> {
        if self.schema_version != 1 {
            return Err(ConfigError::new(
                "unsupported reproducible reduction artifact schema",
            ));
        }
        self.policy.validate()?;
        if self.execution_fingerprint_digest != fingerprint.digest()?
            || self.scalar_backend != fingerprint.scalar_backend
            || self.precision_bits != fingerprint.precision.working_precision_bits
            || fingerprint.reproducibility != Reproducibility::Bitwise
            || fingerprint.thread_policy.scheduling_policy != DETERMINISTIC_REDUCTION_SCHEDULING_V1
            || fingerprint.thread_policy.reduction_policy != self.policy.fingerprint_name()
        {
            return Err(ConfigError::new(
                "reproducible reduction artifact does not match its execution fingerprint",
            ));
        }
        if !self
            .payload_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || self.payload_sha256.len() != 64
            || self.payload_sha256 != self.compute_payload_sha256()?
        {
            return Err(ConfigError::new(
                "reproducible reduction artifact payload digest mismatch",
            ));
        }
        Ok(())
    }

    fn compute_payload_sha256(&self) -> Result<String, ConfigError> {
        let value = serde_json::to_value(ReproducibleReductionPayload {
            schema_version: self.schema_version,
            execution_fingerprint_digest: &self.execution_fingerprint_digest,
            policy: &self.policy,
            scalar_backend: &self.scalar_backend,
            precision_bits: self.precision_bits,
            scalar_encoding: &self.scalar_encoding,
            value: &self.value,
        })
        .map_err(|error| {
            ConfigError::new(format!(
                "reproducible reduction payload serialization failed: {error}"
            ))
        })?;
        let canonical = canonical_json(&value).map_err(|error| {
            ConfigError::new(format!(
                "reproducible reduction payload canonicalization failed: {error}"
            ))
        })?;
        Ok(sha256_hex(canonical.as_bytes()))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HpRuntimeMode {
    #[default]
    FullParallel,
    SafeCapped,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HpRuntimePolicy {
    pub mode: HpRuntimeMode,
    pub worker_threads: Option<usize>,
    pub stack_bytes: Option<usize>,
    pub sequential_precompute: bool,
    pub issue_reference: Option<String>,
}

impl Default for HpRuntimePolicy {
    fn default() -> Self {
        Self {
            mode: HpRuntimeMode::FullParallel,
            worker_threads: None,
            stack_bytes: None,
            sequential_precompute: false,
            issue_reference: None,
        }
    }
}

impl HpRuntimePolicy {
    pub fn safe_capped(
        worker_threads: usize,
        stack_bytes: usize,
        issue_reference: impl Into<String>,
    ) -> Result<Self, ConfigError> {
        let policy = Self {
            mode: HpRuntimeMode::SafeCapped,
            worker_threads: Some(worker_threads),
            stack_bytes: Some(stack_bytes),
            sequential_precompute: true,
            issue_reference: Some(issue_reference.into()),
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let valid = match self.mode {
            HpRuntimeMode::FullParallel => {
                self.worker_threads.is_none()
                    && self.stack_bytes.is_none()
                    && !self.sequential_precompute
                    && self.issue_reference.is_none()
            }
            HpRuntimeMode::SafeCapped => {
                self.worker_threads.is_some_and(|value| value > 0)
                    && self.stack_bytes.is_some_and(|value| value >= 1024 * 1024)
                    && self.sequential_precompute
                    && self
                        .issue_reference
                        .as_deref()
                        .is_some_and(|value| !value.trim().is_empty())
            }
        };
        if !valid {
            return Err(ConfigError::new(
                "HP runtime policy is inconsistent with its full or safe-capped mode",
            ));
        }
        Ok(())
    }

    pub fn thread_policy(&self, full_parallel_threads: usize) -> ThreadPolicyFingerprint {
        match self.mode {
            HpRuntimeMode::FullParallel => ThreadPolicyFingerprint {
                thread_count: full_parallel_threads,
                scheduling_policy: "hp_full_parallel".to_owned(),
                reduction_policy: "deterministic_indexed".to_owned(),
            },
            HpRuntimeMode::SafeCapped => ThreadPolicyFingerprint {
                thread_count: self.worker_threads.unwrap_or(1),
                scheduling_policy: "hp_safe_capped_pool".to_owned(),
                reduction_policy: "deterministic_indexed_sequential_precompute".to_owned(),
            },
        }
    }
}

/// Machine-readable provenance attached to every publication-grade artifact.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverProvenance {
    pub schema_version: u32,
    pub toolkit_version: String,
    pub git_commit: Option<String>,
    pub dependency_lock_digest: Option<String>,
    pub build_profile: String,
    pub platform: Option<String>,
    pub target_triple: Option<String>,
    pub rustc_version: Option<String>,
    pub feature_flags: Vec<String>,
    pub native_libraries: BTreeMap<String, String>,
    pub cpu_feature_policy: Option<String>,
    pub scalar_backend: String,
    pub precision_bits: Option<u32>,
    pub thread_count: Option<usize>,
    pub thread_policy: Option<ThreadPolicyFingerprint>,
    #[serde(default)]
    pub hp_runtime_policy: Option<HpRuntimePolicy>,
    pub solver_configuration: Option<Value>,
    pub domain_configuration: Option<Value>,
    pub resolved_configuration_digest: Option<ConfigDigest>,
    pub deterministic: bool,
    pub execution_fingerprint: Option<ExecutionFingerprint>,
    pub execution_fingerprint_digest: Option<ExecutionFingerprintDigest>,
    /// Explicitly allowlisted, non-secret operational context only.
    pub environment: BTreeMap<String, String>,
    /// Ordered cache decisions made during this run. These are operational
    /// provenance and do not participate in mathematical artifact identity.
    pub cache_accesses: Vec<CacheAccessProvenance>,
    /// Exact canonical semantic keys for every consumed or produced artifact.
    pub artifact_semantics: Vec<SemanticArtifactProvenance>,
    pub artifact_hashes: ResearchArtifactHashes,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheLookupOutcome {
    Hit,
    Miss,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheReuseDisposition {
    Reused,
    Recomputed,
    InspectedOnly,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheValidationMode {
    None,
    Fast,
    Full,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheValidationOutcome {
    NotRequested,
    Passed,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheSourceProvenance {
    pub overlay: String,
    pub location_kind: String,
    pub repository: String,
    pub revision: String,
    /// Role to exact repository path.
    pub document_paths: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheCandidateRejectionProvenance {
    pub overlay: String,
    pub source: Option<String>,
    pub stage: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheValidatedArtifactProvenance {
    pub semantic_digest: String,
    pub manifest_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheAccessProvenance {
    pub schema_version: u32,
    pub operation: String,
    pub artifact_family: String,
    pub semantic_digest: String,
    pub semantic_key_schema_version: u32,
    pub resolved_semantic_key: Value,
    #[serde(default)]
    pub selected_manifest_digest: Option<String>,
    pub ordered_overlays: Vec<String>,
    pub lookup_outcome: CacheLookupOutcome,
    pub reuse_disposition: CacheReuseDisposition,
    pub selected_source: Option<CacheSourceProvenance>,
    pub rejected_candidates: Vec<CacheCandidateRejectionProvenance>,
    pub validation_mode: CacheValidationMode,
    pub validation_outcome: CacheValidationOutcome,
    pub validation_detail: Option<String>,
    pub validated_artifacts: Vec<CacheValidatedArtifactProvenance>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactProvenanceDirection {
    Consumed,
    Produced,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticArtifactProvenance {
    pub direction: ArtifactProvenanceDirection,
    pub artifact_family: String,
    pub semantic_key_schema_version: u32,
    pub resolved_semantic_key: Value,
    pub semantic_digest: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchArtifactHashes {
    pub input_cache_manifests: BTreeMap<String, String>,
    pub generated_operator_artifacts: BTreeMap<String, String>,
}

impl ResearchArtifactHashes {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (kind, entries) in [
            ("input cache", &self.input_cache_manifests),
            ("generated operator", &self.generated_operator_artifacts),
        ] {
            if entries.iter().any(|(label, digest)| {
                label.trim().is_empty()
                    || digest.len() != 64
                    || !digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }) {
                return Err(ConfigError::new(format!(
                    "{kind} artifact hash provenance contains an invalid label or SHA-256"
                )));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<ConfigDigest, ConfigError> {
        self.validate()?;
        let value = serde_json::to_value(self).map_err(|error| {
            ConfigError::new(format!("artifact hash serialization failed: {error}"))
        })?;
        let canonical = canonical_json(&value).map_err(|error| {
            ConfigError::new(format!("artifact hash canonicalization failed: {error}"))
        })?;
        Ok(ConfigDigest(sha256_hex(canonical.as_bytes())))
    }

    pub fn verify_exact(&self, expected: &Self) -> Result<(), ConfigError> {
        self.validate()?;
        expected.validate()?;
        if self != expected {
            return Err(ConfigError::new(
                "input cache or generated operator hashes differ from the saved result",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheDuplicationAudit {
    pub schema_version: u32,
    pub avoidable_recomputation_semantic_digests: Vec<String>,
}

impl SolverProvenance {
    pub fn current_package(scalar_backend: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            toolkit_version: env!("CARGO_PKG_VERSION").to_owned(),
            build_profile: if cfg!(debug_assertions) {
                "debug".to_owned()
            } else {
                "release".to_owned()
            },
            scalar_backend: scalar_backend.into(),
            deterministic: true,
            ..Self::default()
        }
    }

    pub fn with_execution_fingerprint(
        mut self,
        fingerprint: &ExecutionFingerprint,
    ) -> Result<Self, ConfigError> {
        fingerprint.validate()?;
        self.git_commit = Some(fingerprint.toolkit_revision.clone());
        self.target_triple = Some(fingerprint.target_triple.clone());
        self.rustc_version = Some(fingerprint.compiler.clone());
        self.feature_flags = fingerprint.feature_flags.iter().cloned().collect();
        self.native_libraries = fingerprint.native_libraries.clone();
        self.cpu_feature_policy = Some(fingerprint.cpu_feature_policy.clone());
        self.scalar_backend = fingerprint.scalar_backend.clone();
        self.precision_bits = Some(fingerprint.precision.working_precision_bits);
        self.thread_count = Some(fingerprint.thread_policy.thread_count);
        self.thread_policy = Some(fingerprint.thread_policy.clone());
        self.resolved_configuration_digest =
            Some(fingerprint.effective_configuration_digest.clone());
        self.deterministic = fingerprint.reproducibility != Reproducibility::Exploratory;
        self.execution_fingerprint_digest = Some(fingerprint.digest()?);
        self.execution_fingerprint = Some(fingerprint.clone());
        Ok(self)
    }

    pub fn record_hp_runtime_policy(
        &mut self,
        policy: HpRuntimePolicy,
        full_parallel_threads: usize,
    ) -> Result<(), ConfigError> {
        policy.validate()?;
        let thread_policy = policy.thread_policy(full_parallel_threads);
        if let Some(fingerprint) = &self.execution_fingerprint {
            if fingerprint.thread_policy != thread_policy {
                return Err(ConfigError::new(
                    "HP runtime policy does not match the execution fingerprint thread policy",
                ));
            }
        }
        self.thread_count = Some(thread_policy.thread_count);
        self.thread_policy = Some(thread_policy);
        self.hp_runtime_policy = Some(policy);
        Ok(())
    }

    pub fn with_saved_result_context(
        self,
        fingerprint: &ExecutionFingerprint,
        dependency_lock_digest: impl Into<String>,
        platform: impl Into<String>,
        solver_configuration: Value,
        domain_configuration: Value,
    ) -> Result<Self, ConfigError> {
        let mut provenance = self.with_execution_fingerprint(fingerprint)?;
        provenance.dependency_lock_digest = Some(dependency_lock_digest.into());
        provenance.platform = Some(platform.into());
        provenance.solver_configuration = Some(solver_configuration);
        provenance.domain_configuration = Some(domain_configuration);
        provenance.validate_saved_result()?;
        Ok(provenance)
    }

    pub fn validate_saved_result(&self) -> Result<(), ConfigError> {
        crate::validate_secret_free(self, "saved-result provenance")?;
        let fingerprint = self.execution_fingerprint.as_ref().ok_or_else(|| {
            ConfigError::new("saved-result provenance requires a complete execution fingerprint")
        })?;
        let fingerprint_digest = fingerprint.digest()?;
        let sha256 = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        let lock_digest = self.dependency_lock_digest.as_deref().unwrap_or_default();
        let resolved_digest = self.resolved_configuration_digest.as_ref();
        if self.schema_version != 1
            || self.toolkit_version.trim().is_empty()
            || self.build_profile.trim().is_empty()
            || self
                .platform
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            || !sha256(lock_digest)
            || self.git_commit.as_deref() != Some(fingerprint.toolkit_revision.as_str())
            || self.target_triple.as_deref() != Some(fingerprint.target_triple.as_str())
            || self.rustc_version.as_deref() != Some(fingerprint.compiler.as_str())
            || self.feature_flags
                != fingerprint
                    .feature_flags
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
            || self.native_libraries != fingerprint.native_libraries
            || self.cpu_feature_policy.as_deref() != Some(fingerprint.cpu_feature_policy.as_str())
            || self.scalar_backend != fingerprint.scalar_backend
            || self.precision_bits != Some(fingerprint.precision.working_precision_bits)
            || self.thread_count != Some(fingerprint.thread_policy.thread_count)
            || self.thread_policy.as_ref() != Some(&fingerprint.thread_policy)
            || resolved_digest != Some(&fingerprint.effective_configuration_digest)
            || self.execution_fingerprint_digest.as_ref() != Some(&fingerprint_digest)
            || self
                .solver_configuration
                .as_ref()
                .is_none_or(Value::is_null)
            || self
                .domain_configuration
                .as_ref()
                .is_none_or(Value::is_null)
        {
            return Err(ConfigError::new(
                "saved-result provenance is incomplete or differs from its execution fingerprint",
            ));
        }
        for (name, configuration) in [
            ("solver", self.solver_configuration.as_ref().unwrap()),
            ("domain", self.domain_configuration.as_ref().unwrap()),
        ] {
            canonical_json(configuration).map_err(|error| {
                ConfigError::new(format!(
                    "saved-result {name} configuration is not canonicalizable: {error}"
                ))
            })?;
        }
        if let Some(policy) = &self.hp_runtime_policy {
            policy.validate()?;
            if policy.thread_policy(fingerprint.thread_policy.thread_count)
                != fingerprint.thread_policy
            {
                return Err(ConfigError::new(
                    "saved-result HP runtime policy differs from its execution fingerprint",
                ));
            }
        }
        self.artifact_hashes.validate()?;
        Ok(())
    }

    pub fn record_cache_access(
        &mut self,
        access: CacheAccessProvenance,
    ) -> Result<(), ConfigError> {
        access.validate()?;
        self.artifact_semantics.push(SemanticArtifactProvenance {
            direction: ArtifactProvenanceDirection::Consumed,
            artifact_family: access.artifact_family.clone(),
            semantic_key_schema_version: access.semantic_key_schema_version,
            resolved_semantic_key: access.resolved_semantic_key.clone(),
            semantic_digest: access.semantic_digest.clone(),
        });
        if let Some(manifest_digest) = &access.selected_manifest_digest {
            insert_artifact_hash(
                &mut self.artifact_hashes.input_cache_manifests,
                format!(
                    "{}:{}:{}",
                    access.operation, access.artifact_family, access.semantic_digest
                ),
                manifest_digest.clone(),
            )?;
        }
        self.cache_accesses.push(access);
        Ok(())
    }

    pub fn record_generated_operator_hash(
        &mut self,
        label: impl Into<String>,
        digest: impl Into<String>,
    ) -> Result<(), ConfigError> {
        insert_artifact_hash(
            &mut self.artifact_hashes.generated_operator_artifacts,
            label.into(),
            digest.into(),
        )
    }

    pub fn record_produced_artifact(
        &mut self,
        artifact: SemanticArtifactProvenance,
    ) -> Result<(), ConfigError> {
        if artifact.direction != ArtifactProvenanceDirection::Produced {
            return Err(ConfigError::new(
                "produced-artifact provenance must use the produced direction",
            ));
        }
        artifact.validate()?;
        self.artifact_semantics.push(artifact);
        Ok(())
    }

    pub fn cache_duplication_audit(&self) -> CacheDuplicationAudit {
        CacheDuplicationAudit {
            schema_version: 1,
            avoidable_recomputation_semantic_digests: self
                .cache_accesses
                .iter()
                .filter(|access| {
                    access.lookup_outcome == CacheLookupOutcome::Hit
                        && access.reuse_disposition == CacheReuseDisposition::Recomputed
                })
                .map(|access| access.semantic_digest.clone())
                .collect(),
        }
    }
}

impl CacheAccessProvenance {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let sha256 = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if self.schema_version != 1
            || self.operation.trim().is_empty()
            || self.artifact_family.trim().is_empty()
            || !sha256(&self.semantic_digest)
            || validate_semantic_key_record(
                self.semantic_key_schema_version,
                &self.resolved_semantic_key,
                &self.semantic_digest,
            )
            .is_err()
            || self.ordered_overlays.is_empty()
            || self
                .ordered_overlays
                .iter()
                .any(|value| value.trim().is_empty())
        {
            return Err(ConfigError::new(
                "cache access provenance identity is invalid",
            ));
        }
        if (self.lookup_outcome == CacheLookupOutcome::Hit) != self.selected_source.is_some() {
            return Err(ConfigError::new(
                "cache hit/miss provenance does not match selected source",
            ));
        }
        if (self.lookup_outcome == CacheLookupOutcome::Hit)
            != self.selected_manifest_digest.is_some()
            || self
                .selected_manifest_digest
                .as_deref()
                .is_some_and(|digest| !sha256(digest))
        {
            return Err(ConfigError::new(
                "cache hit/miss provenance does not match selected manifest identity",
            ));
        }
        if self.reuse_disposition == CacheReuseDisposition::Reused
            && self.lookup_outcome != CacheLookupOutcome::Hit
        {
            return Err(ConfigError::new(
                "a cache miss cannot be recorded as reused",
            ));
        }
        if self.validation_outcome == CacheValidationOutcome::NotRequested
            && self.validation_mode != CacheValidationMode::None
        {
            return Err(ConfigError::new(
                "cache validation mode requires a validation outcome",
            ));
        }
        if self.validated_artifacts.iter().any(|artifact| {
            !sha256(&artifact.semantic_digest) || !sha256(&artifact.manifest_digest)
        }) {
            return Err(ConfigError::new(
                "cache validation provenance contains an invalid digest",
            ));
        }
        Ok(())
    }
}

impl SemanticArtifactProvenance {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.artifact_family.trim().is_empty() {
            return Err(ConfigError::new(
                "semantic artifact provenance family is empty",
            ));
        }
        validate_semantic_key_record(
            self.semantic_key_schema_version,
            &self.resolved_semantic_key,
            &self.semantic_digest,
        )
    }
}

fn validate_semantic_key_record(
    schema_version: u32,
    resolved_semantic_key: &Value,
    semantic_digest: &str,
) -> Result<(), ConfigError> {
    if schema_version == 0
        || resolved_semantic_key
            .get("schema_version")
            .and_then(Value::as_u64)
            != Some(u64::from(schema_version))
    {
        return Err(ConfigError::new(
            "semantic-key provenance schema does not match its resolved key",
        ));
    }
    let canonical = canonical_json(resolved_semantic_key).map_err(|error| {
        ConfigError::new(format!(
            "semantic-key provenance is not canonicalizable: {error}"
        ))
    })?;
    if semantic_digest != sha256_hex(canonical.as_bytes()) {
        return Err(ConfigError::new(
            "semantic-key provenance digest does not match the resolved key",
        ));
    }
    Ok(())
}

fn insert_artifact_hash(
    entries: &mut BTreeMap<String, String>,
    label: String,
    digest: String,
) -> Result<(), ConfigError> {
    let valid = !label.trim().is_empty()
        && digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(ConfigError::new(
            "artifact hash requires a nonempty label and lowercase SHA-256",
        ));
    }
    if entries
        .get(&label)
        .is_some_and(|existing| existing != &digest)
    {
        return Err(ConfigError::new(format!(
            "artifact hash label {label:?} was already bound to different bytes"
        )));
    }
    entries.insert(label, digest);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> ExecutionFingerprint {
        ExecutionFingerprint {
            schema_version: 1,
            toolkit_revision: "a029c1d".to_owned(),
            dependency_revisions: BTreeMap::from([("nalgebra".to_owned(), "0.33.2".to_owned())]),
            compiler: "rustc 1.95.0".to_owned(),
            target_triple: "x86_64-pc-windows-msvc".to_owned(),
            native_libraries: BTreeMap::from([("mpfr".to_owned(), "4.2.2".to_owned())]),
            scalar_backend: "rug_mpfr".to_owned(),
            scalar_backend_version: "1.28.0".to_owned(),
            precision: PrecisionFingerprint {
                working_precision_bits: 4096,
                guard_bits: 128,
                rounding_policy: "nearest_then_directed_verification".to_owned(),
            },
            algorithm_semantics_versions: BTreeMap::from([(
                "selected_eigensolver".to_owned(),
                "block-shift-invert-v1".to_owned(),
            )]),
            cpu_feature_policy: "portable-x86_64".to_owned(),
            thread_policy: ThreadPolicyFingerprint {
                thread_count: 8,
                scheduling_policy: "fixed-partition".to_owned(),
                reduction_policy: "deterministic-pairwise-v1".to_owned(),
            },
            feature_flags: ["hp-reference".to_owned()].into_iter().collect(),
            effective_configuration_digest: ConfigDigest("a".repeat(64)),
            resolved_resource_policy_digest: ConfigDigest("b".repeat(64)),
            reproducibility: Reproducibility::Bitwise,
        }
    }

    #[test]
    fn canonical_fingerprint_digest_is_stable() {
        let first = fingerprint();
        let mut second = fingerprint();
        second.dependency_revisions = BTreeMap::new();
        second
            .dependency_revisions
            .insert("nalgebra".to_owned(), "0.33.2".to_owned());
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn different_thread_policy_forbids_byte_identity_claim() {
        let first = fingerprint();
        let mut second = fingerprint();
        second.thread_policy.thread_count = 16;
        let plan = first.comparison_plan(&second).unwrap();
        assert!(!plan.fingerprints_match);
        assert!(!plan.byte_identity_permitted);
        assert_eq!(
            plan.required_comparison,
            CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure
        );
    }

    #[test]
    fn matching_bitwise_fingerprints_permit_byte_comparison() {
        let first = fingerprint();
        let plan = first.comparison_plan(&first).unwrap();
        assert!(plan.byte_identity_permitted);
        assert_eq!(
            plan.required_comparison,
            CrossFingerprintComparison::ByteIdentity
        );
    }

    #[test]
    fn bitwise_fingerprint_rejects_missing_decisive_identities() {
        let mut incomplete = fingerprint();
        incomplete.dependency_revisions.clear();
        assert!(incomplete.validate().is_err());

        let mut incomplete = fingerprint();
        incomplete.algorithm_semantics_versions.clear();
        assert!(incomplete.validate().is_err());

        let mut incomplete = fingerprint();
        incomplete.native_libraries.clear();
        assert!(incomplete.validate().is_err());
    }

    #[test]
    fn saved_result_provenance_round_trips_complete_machine_readable_context() {
        let provenance = SolverProvenance::current_package("placeholder")
            .with_saved_result_context(
                &fingerprint(),
                "c".repeat(64),
                "windows-11-x86_64",
                serde_json::json!({"solver": "block_shift_invert", "target": "smallest"}),
                serde_json::json!({"family": "ccm", "window": [14, 50]}),
            )
            .unwrap();
        provenance.validate_saved_result().unwrap();
        let bytes = serde_json::to_vec(&provenance).unwrap();
        let decoded: SolverProvenance = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, provenance);
        decoded.validate_saved_result().unwrap();
        assert_eq!(decoded.scalar_backend, "rug_mpfr");
        assert_eq!(decoded.precision_bits, Some(4096));
        assert_eq!(decoded.thread_count, Some(8));
    }

    #[test]
    fn saved_result_provenance_rejects_a_mirrored_field_mismatch() {
        let mut provenance = SolverProvenance::current_package("placeholder")
            .with_saved_result_context(
                &fingerprint(),
                "c".repeat(64),
                "linux-x86_64",
                serde_json::json!({"solver": "block_shift_invert"}),
                serde_json::json!({"family": "ccm"}),
            )
            .unwrap();
        provenance.thread_count = Some(1);
        assert!(provenance
            .validate_saved_result()
            .unwrap_err()
            .to_string()
            .contains("differs from its execution fingerprint"));
    }

    #[test]
    fn safe_hp_runtime_policy_is_explicit_and_fingerprint_bound() {
        let policy =
            HpRuntimePolicy::safe_capped(2, 64 * 1024 * 1024, "wsl-gmp-thread-instability-review")
                .unwrap();
        let mut execution = fingerprint();
        execution.thread_policy = policy.thread_policy(8);
        let mut provenance = SolverProvenance::current_package("rug_mpfr")
            .with_saved_result_context(
                &execution,
                "c".repeat(64),
                "linux-wsl2-x86_64",
                serde_json::json!({"solver": "hp_reference"}),
                serde_json::json!({"family": "ccm"}),
            )
            .unwrap();
        provenance
            .record_hp_runtime_policy(policy.clone(), 8)
            .unwrap();
        provenance.validate_saved_result().unwrap();
        assert_eq!(provenance.hp_runtime_policy, Some(policy.clone()));
        assert_eq!(provenance.thread_count, Some(2));

        let mut mismatched = provenance;
        mismatched
            .execution_fingerprint
            .as_mut()
            .unwrap()
            .thread_policy = HpRuntimePolicy::default().thread_policy(8);
        assert!(mismatched.validate_saved_result().is_err());
    }

    #[test]
    fn cache_provenance_exposes_avoidable_recomputation() {
        let semantic_key = serde_json::json!({
            "schema_version": 1,
            "artifact_kind": "ccm_form",
            "resolved_mathematical_parameters": {"parity": "natural"}
        });
        let digest = sha256_hex(canonical_json(&semantic_key).unwrap().as_bytes());
        let mut provenance = SolverProvenance::current_package("test");
        provenance
            .record_cache_access(CacheAccessProvenance {
                schema_version: 1,
                operation: "ccm.form_components".to_owned(),
                artifact_family: "ccm-form".to_owned(),
                semantic_digest: digest.clone(),
                semantic_key_schema_version: 1,
                resolved_semantic_key: semantic_key,
                selected_manifest_digest: Some("b".repeat(64)),
                ordered_overlays: vec!["private".to_owned(), "public".to_owned()],
                lookup_outcome: CacheLookupOutcome::Hit,
                reuse_disposition: CacheReuseDisposition::Recomputed,
                selected_source: Some(CacheSourceProvenance {
                    overlay: "private".to_owned(),
                    location_kind: "github_remote".to_owned(),
                    repository: "example-org/restricted-cache".to_owned(),
                    revision: "b".repeat(40),
                    document_paths: BTreeMap::from([(
                        "manifest".to_owned(),
                        "manifests/a.json".to_owned(),
                    )]),
                }),
                rejected_candidates: Vec::new(),
                validation_mode: CacheValidationMode::Fast,
                validation_outcome: CacheValidationOutcome::Passed,
                validation_detail: None,
                validated_artifacts: Vec::new(),
            })
            .unwrap();
        assert_eq!(
            provenance
                .cache_duplication_audit()
                .avoidable_recomputation_semantic_digests,
            vec![digest]
        );
        assert_eq!(provenance.artifact_semantics.len(), 1);
        assert_eq!(
            provenance.artifact_semantics[0].direction,
            ArtifactProvenanceDirection::Consumed
        );
        assert_eq!(provenance.artifact_hashes.input_cache_manifests.len(), 1);
    }

    #[test]
    fn produced_artifact_retains_the_complete_canonical_semantic_key() {
        let key = serde_json::json!({
            "schema_version": 1,
            "artifact_kind": "ccm_matrix",
            "mathematical_semantics_version": "ccm-v2",
            "resolved_mathematical_parameters": {
                "parity_mode": "forced_odd",
                "natural_parity": "even",
                "precision_bits": 512,
                "basis_size": 4096,
                "cutoff_mode": "exact_fractional_lambda_squared",
                "solver_semantics": "block_shift_invert_v1"
            }
        });
        let digest = sha256_hex(canonical_json(&key).unwrap().as_bytes());
        let mut provenance = SolverProvenance::current_package("rug_mpfr");
        provenance
            .record_produced_artifact(SemanticArtifactProvenance {
                direction: ArtifactProvenanceDirection::Produced,
                artifact_family: "ccm".to_owned(),
                semantic_key_schema_version: 1,
                resolved_semantic_key: key.clone(),
                semantic_digest: digest,
            })
            .unwrap();
        assert_eq!(provenance.artifact_semantics[0].resolved_semantic_key, key);

        let mut tampered = provenance.artifact_semantics[0].clone();
        tampered.resolved_semantic_key["resolved_mathematical_parameters"]["basis_size"] =
            serde_json::json!(2048);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn artifact_hash_inventory_verifies_inputs_and_generated_operators_exactly() {
        let mut provenance = SolverProvenance::current_package("rug_mpfr");
        provenance
            .record_generated_operator_hash("ccm/operator/n4096", "d".repeat(64))
            .unwrap();
        let expected = provenance.artifact_hashes.clone();
        provenance.artifact_hashes.verify_exact(&expected).unwrap();
        assert!(provenance.artifact_hashes.digest().unwrap().is_sha256());

        let mut changed = expected;
        changed
            .generated_operator_artifacts
            .insert("ccm/operator/n4096".to_owned(), "e".repeat(64));
        assert!(provenance
            .artifact_hashes
            .verify_exact(&changed)
            .unwrap_err()
            .to_string()
            .contains("differ"));
        assert!(provenance
            .record_generated_operator_hash("ccm/operator/n4096", "f".repeat(64))
            .is_err());
    }
}
