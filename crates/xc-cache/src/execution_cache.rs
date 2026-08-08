//! Typed resolve-or-compute execution over the common cache fabric.

use crate::{
    ArtifactDraft, ArtifactKey, ArtifactManifest, CacheError, CachePolicy, CacheQuality,
    CacheResolver, CacheVisibility, ContentDigest, SemanticKeyEnvelope, ToolkitVersion,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
#[cfg(test)]
use std::io::Cursor;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use xc_core::{
    CacheAccessProvenance, CacheCandidateRejectionProvenance, CacheLookupOutcome,
    CacheReuseDisposition, CacheSourceProvenance, CacheValidatedArtifactProvenance,
    CacheValidationMode, CacheValidationOutcome,
};

pub const SEMANTIC_KEY_MANIFEST_TAG: &str = "xc.semantic_key.v1";
pub const REMOTE_CANONICAL_MANIFEST_TAG: &str = "xc.remote_canonical_manifest.v1";
pub const OUTPUT_VALIDATION_SEED_TAG: &str = "xc.output_validation.seed_source.v1";
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, DateTime, ZipArchive, ZipWriter};

/// Whether a computation may consult or populate the cache fabric.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactExecutionCacheMode {
    Disabled,
    PreferReuse,
    RequireReuse,
    /// Skip every cache lookup, compute a fresh value, then write and stage it.
    Refresh,
    /// Recompute into an isolated writable cache and compare every logical
    /// payload with the same identity in read-only reference overlays.
    VerifyAgainstReference,
}

impl ArtifactExecutionCacheMode {
    pub fn fabric_enabled(self) -> bool {
        match self {
            Self::Disabled => false,
            Self::PreferReuse
            | Self::RequireReuse
            | Self::Refresh
            | Self::VerifyAgainstReference => true,
        }
    }

    pub fn consults_cache_for_result_reuse(self) -> bool {
        match self {
            Self::PreferReuse | Self::RequireReuse => true,
            Self::Disabled | Self::Refresh | Self::VerifyAgainstReference => false,
        }
    }

    pub fn consults_overlays_for_route_selection(self) -> bool {
        match self {
            Self::PreferReuse | Self::RequireReuse | Self::VerifyAgainstReference => true,
            Self::Disabled | Self::Refresh => false,
        }
    }

    pub fn may_use_cached_seed(self) -> bool {
        match self {
            Self::PreferReuse | Self::VerifyAgainstReference => true,
            Self::Disabled | Self::RequireReuse | Self::Refresh => false,
        }
    }

    pub fn writes_computed_artifacts(self) -> bool {
        match self {
            Self::PreferReuse | Self::Refresh | Self::VerifyAgainstReference => true,
            Self::Disabled | Self::RequireReuse => false,
        }
    }

    pub fn compares_against_reference(self) -> bool {
        match self {
            Self::VerifyAgainstReference => true,
            Self::Disabled | Self::PreferReuse | Self::RequireReuse | Self::Refresh => false,
        }
    }

    pub fn may_stage_for_publication(self) -> bool {
        match self {
            Self::PreferReuse | Self::Refresh => true,
            Self::Disabled | Self::RequireReuse | Self::VerifyAgainstReference => false,
        }
    }

    pub fn requires_reuse(self) -> bool {
        match self {
            Self::RequireReuse => true,
            Self::Disabled | Self::PreferReuse | Self::Refresh | Self::VerifyAgainstReference => {
                false
            }
        }
    }

    pub fn is_refresh(self) -> bool {
        match self {
            Self::Refresh => true,
            Self::Disabled
            | Self::PreferReuse
            | Self::RequireReuse
            | Self::VerifyAgainstReference => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedRunProfile {
    Normal,
    Author,
}

/// Remote overlays consulted after the automatic workstation cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedRemoteCacheMode {
    None,
    Public,
    Private,
    PrivateThenPublic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputValidationConfig {
    pub validation_root: PathBuf,
    pub report_root: PathBuf,
    pub reference_mode: ManagedRemoteCacheMode,
}

fn parse_cache_mode(value: Option<&str>) -> Result<ArtifactExecutionCacheMode, CacheError> {
    match value.unwrap_or("reuse") {
        "reuse" | "prefer_reuse" | "prefer-reuse" => Ok(ArtifactExecutionCacheMode::PreferReuse),
        "refresh" => Ok(ArtifactExecutionCacheMode::Refresh),
        "require_reuse" | "require-reuse" => Ok(ArtifactExecutionCacheMode::RequireReuse),
        "verify" | "verify_against_reference" | "verify-against-reference" => {
            Ok(ArtifactExecutionCacheMode::VerifyAgainstReference)
        }
        other => Err(CacheError::InvalidManifest(format!(
            "unsupported XC_CACHE_MODE {other:?}; expected reuse, refresh, require_reuse, or verify"
        ))),
    }
}

fn parse_validation_reference_mode(
    value: Option<&str>,
) -> Result<ManagedRemoteCacheMode, CacheError> {
    match value.unwrap_or("private_public") {
        "public" => Ok(ManagedRemoteCacheMode::Public),
        "private" => Ok(ManagedRemoteCacheMode::Private),
        "private_public" | "private-public" => Ok(ManagedRemoteCacheMode::PrivateThenPublic),
        other => Err(CacheError::InvalidManifest(format!(
            "unsupported XC_VALIDATION_REFERENCE {other:?}; expected public, private, or private_public"
        ))),
    }
}

fn parse_requested_assurance(value: Option<&str>) -> Result<xc_core::AssuranceLevel, CacheError> {
    match value.unwrap_or("computed") {
        "computed" => Ok(xc_core::AssuranceLevel::Computed),
        "cross_checked" | "cross-checked" => Ok(xc_core::AssuranceLevel::CrossChecked),
        "certified" => Ok(xc_core::AssuranceLevel::Certified),
        other => Err(CacheError::InvalidManifest(format!(
            "unsupported XC_ASSURANCE {other:?}; expected computed, cross_checked, or certified"
        ))),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificationFailurePolicy {
    RetainComputedFailRun,
    RetainComputedSkipPublication,
}

/// Resolver, acceptance, and write controls shared by numerical domains.
///
/// Domain APIs construct their own semantic identity and combine it with this
/// context, preventing callers from accidentally supplying a key that does not
/// describe the requested computation.
pub struct ArtifactCacheContext<'a> {
    pub resolver: Option<&'a CacheResolver>,
    /// Reference-only resolver used by output-preservation validation. It
    /// never contains the writable validation-computed layer.
    pub reference_resolver: Option<&'a CacheResolver>,
    pub acceptance: Option<&'a CachePolicy>,
    pub ordered_overlays: Vec<String>,
    pub mode: ArtifactExecutionCacheMode,
    pub write_on_miss: bool,
    pub write_visibility: CacheVisibility,
    pub requested_assurance: xc_core::AssuranceLevel,
    pub certification_failure_policy: CertificationFailurePolicy,
    /// Optional toolkit-owned handoff for validated publication candidates.
    /// Both fresh computations and validated reuse hits are emitted so an
    /// author run can package or certify an existing local result without
    /// repeating the numerical computation.
    pub production_sink: Option<&'a dyn ArtifactProductionSink>,
}

/// Toolkit-owned configuration for typed numerical caching. Applications do
/// not construct stores or production sinks; they may expose ordinary command
/// line flags by setting these values before invoking a numerical API.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedArtifactCacheConfig {
    pub profile: ManagedRunProfile,
    pub requested_assurance: xc_core::AssuranceLevel,
    pub certification_failure_policy: CertificationFailurePolicy,
    pub cache_root: PathBuf,
    pub staging_root: Option<PathBuf>,
    pub publication_target: xc_core::PublicationTarget,
    pub repository_owner: String,
    pub remote_cache_mode: ManagedRemoteCacheMode,
    pub cache_mode: ArtifactExecutionCacheMode,
    pub replace_existing_publication: bool,
    pub execute_remote_mutations: bool,
    pub output_validation: Option<OutputValidationConfig>,
}

impl ManagedArtifactCacheConfig {
    /// Loads the standard process configuration.
    ///
    /// The managed cache is always enabled. `XC_CACHE_ROOT` is an optional
    /// author/operator override; ordinary consumers receive a platform-native
    /// per-user cache without configuring environment variables. Publication
    /// remains opt-in and is never inferred from credentials.
    pub fn from_environment() -> Result<Option<Self>, CacheError> {
        let cache_root =
            std::env::var_os("XC_CACHE_ROOT").or_else(|| std::env::var_os("XC_TYPED_CACHE_ROOT"));
        let cache_root = cache_root.unwrap_or_else(default_managed_cache_root);
        if cache_root.is_empty() {
            return Err(CacheError::InvalidManifest(
                "XC_CACHE_ROOT must not be empty".to_owned(),
            ));
        }
        let profile = match std::env::var("XC_RUN_PROFILE")
            .unwrap_or_else(|_| "normal".to_owned())
            .as_str()
        {
            "normal" => ManagedRunProfile::Normal,
            "author" => ManagedRunProfile::Author,
            other => {
                return Err(CacheError::InvalidManifest(format!(
                    "unsupported XC_RUN_PROFILE {other:?}; expected normal or author"
                )))
            }
        };
        let assurance_value = std::env::var("XC_ASSURANCE").ok();
        let requested_assurance = parse_requested_assurance(assurance_value.as_deref())?;
        let cache_mode_value = std::env::var("XC_CACHE_MODE").ok();
        let cache_mode = parse_cache_mode(cache_mode_value.as_deref())?;
        let certification_failure_policy = match std::env::var("XC_CERTIFICATION_FAILURE_POLICY")
            .unwrap_or_else(|_| "retain_computed_fail_run".to_owned())
            .as_str()
        {
            "retain_computed_fail_run" => CertificationFailurePolicy::RetainComputedFailRun,
            "retain_computed_skip_publication" => {
                CertificationFailurePolicy::RetainComputedSkipPublication
            }
            other => {
                return Err(CacheError::InvalidManifest(format!(
                    "unsupported XC_CERTIFICATION_FAILURE_POLICY {other:?}"
                )))
            }
        };
        let explicit_staging_root = std::env::var_os("XC_PUBLISH_STAGING_ROOT")
            .or_else(|| std::env::var_os("XC_PUBLICATION_QUEUE"))
            .map(PathBuf::from);
        let mut staging_root = explicit_staging_root.clone();
        if staging_root
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "XC_PUBLISH_STAGING_ROOT must not be empty".to_owned(),
            ));
        }
        let publication_target = match std::env::var("XC_PUBLISH_TARGET")
            .unwrap_or_else(|_| "none".to_owned())
            .as_str()
        {
            "none" => xc_core::PublicationTarget::None,
            "private" => xc_core::PublicationTarget::Private,
            "public" => xc_core::PublicationTarget::Public,
            "both" => xc_core::PublicationTarget::Both,
            other => {
                return Err(CacheError::InvalidManifest(format!(
                "unsupported XC_PUBLISH_TARGET {other:?}; expected none, private, public, or both"
            )))
            }
        };
        if staging_root.is_none()
            && !cache_mode.compares_against_reference()
            && (profile == ManagedRunProfile::Author
                || requested_assurance != xc_core::AssuranceLevel::Computed)
        {
            staging_root = Some(PathBuf::from(&cache_root).join("publication"));
        }
        let repository_owner = std::env::var("XC_CACHE_REPOSITORY_OWNER")
            .unwrap_or_else(|_| "TeamXcelerator".to_owned());
        let remote_cache_mode = match std::env::var("XC_CACHE_REMOTE")
            .unwrap_or_else(|_| "public".to_owned())
            .as_str()
        {
            "none" => ManagedRemoteCacheMode::None,
            "public" => ManagedRemoteCacheMode::Public,
            "private" => ManagedRemoteCacheMode::Private,
            "private_public" | "private-public" => ManagedRemoteCacheMode::PrivateThenPublic,
            other => {
                return Err(CacheError::InvalidManifest(format!(
                    "unsupported XC_CACHE_REMOTE {other:?}; expected none, public, private, or private_public"
                )))
            }
        };
        let replace_existing_publication = match std::env::var("XC_PUBLISH_REPLACE")
            .unwrap_or_else(|_| "false".to_owned())
            .as_str()
        {
            "false" | "0" => false,
            "true" | "1" => true,
            other => {
                return Err(CacheError::InvalidManifest(format!(
                    "unsupported XC_PUBLISH_REPLACE {other:?}; expected true or false"
                )))
            }
        };
        let execute_remote_mutations = match std::env::var("XC_PUBLISH_EXECUTE")
            .unwrap_or_else(|_| "false".to_owned())
            .as_str()
        {
            "false" | "0" => false,
            "true" | "1" => true,
            other => {
                return Err(CacheError::InvalidManifest(format!(
                    "unsupported XC_PUBLISH_EXECUTE {other:?}; expected true or false"
                )))
            }
        };
        if profile == ManagedRunProfile::Normal
            && (publication_target != xc_core::PublicationTarget::None
                || execute_remote_mutations
                || cache_mode.is_refresh()
                || replace_existing_publication)
        {
            return Err(CacheError::InvalidManifest(
                "normal runs cannot request publication; use XC_RUN_PROFILE=author".to_owned(),
            ));
        }
        if execute_remote_mutations && publication_target == xc_core::PublicationTarget::None {
            return Err(CacheError::InvalidManifest(
                "XC_PUBLISH_EXECUTE=true requires a non-none publication target".to_owned(),
            ));
        }
        if replace_existing_publication
            && (!execute_remote_mutations
                || publication_target == xc_core::PublicationTarget::None
                || !cache_mode.is_refresh())
        {
            return Err(CacheError::InvalidManifest(
                "XC_PUBLISH_REPLACE=true requires author refresh mode, remote execution, and a publication target"
                    .to_owned(),
            ));
        }
        let output_validation = if cache_mode.compares_against_reference() {
            if publication_target != xc_core::PublicationTarget::None
                || execute_remote_mutations
                || replace_existing_publication
                || explicit_staging_root.is_some()
                || staging_root.is_some()
            {
                return Err(CacheError::InvalidManifest(
                    "XC_CACHE_MODE=verify cannot stage, publish, replace, or execute remote mutations"
                        .to_owned(),
                ));
            }
            let validation_root = std::env::var_os("XC_VALIDATION_CACHE_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| default_validation_cache_root(Path::new(&cache_root)));
            let report_root = std::env::var_os("XC_VALIDATION_REPORT_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| validation_root.join("reports"));
            validate_separate_cache_roots(Path::new(&cache_root), &validation_root)?;
            let reference_value = std::env::var("XC_VALIDATION_REFERENCE").ok();
            Some(OutputValidationConfig {
                validation_root,
                report_root,
                reference_mode: parse_validation_reference_mode(reference_value.as_deref())?,
            })
        } else {
            None
        };
        Ok(Some(Self {
            profile,
            requested_assurance,
            certification_failure_policy,
            cache_root: PathBuf::from(cache_root),
            staging_root,
            publication_target,
            repository_owner,
            remote_cache_mode,
            cache_mode,
            replace_existing_publication,
            execute_remote_mutations,
            output_validation,
        }))
    }
}

fn default_managed_cache_root() -> std::ffi::OsString {
    if let Some(root) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(root).join("xcelerator").into_os_string();
    }
    if cfg!(windows) {
        if let Some(root) = std::env::var_os("LOCALAPPDATA").filter(|value| !value.is_empty()) {
            return PathBuf::from(root)
                .join("Xcelerator")
                .join("cache")
                .into_os_string();
        }
    }
    if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home)
            .join(".cache")
            .join("xcelerator")
            .into_os_string();
    }
    PathBuf::from(".xcelerator-cache").into_os_string()
}

fn default_validation_cache_root(cache_root: &Path) -> PathBuf {
    let file_name = cache_root
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("cache");
    cache_root.with_file_name(format!("{file_name}-validation"))
}

fn absolute_lexical_path(path: &Path) -> Result<PathBuf, CacheError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn physical_lexical_path(path: &Path) -> Result<PathBuf, CacheError> {
    let normalized = absolute_lexical_path(path)?;
    let mut existing = normalized.clone();
    let mut missing_suffix = Vec::new();
    while !existing.exists() {
        let component = existing.file_name().ok_or_else(|| {
            CacheError::InvalidManifest(format!(
                "cache root {} has no existing filesystem ancestor",
                normalized.display()
            ))
        })?;
        missing_suffix.push(component.to_os_string());
        if !existing.pop() {
            return Err(CacheError::InvalidManifest(format!(
                "cache root {} has no existing filesystem ancestor",
                normalized.display()
            )));
        }
    }
    let mut physical = fs::canonicalize(existing)?;
    for component in missing_suffix.into_iter().rev() {
        physical.push(component);
    }
    Ok(physical)
}

fn validate_separate_cache_roots(
    production_root: &Path,
    validation_root: &Path,
) -> Result<(), CacheError> {
    if production_root.as_os_str().is_empty() || validation_root.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "production and validation cache roots must be nonempty".to_owned(),
        ));
    }
    let production = physical_lexical_path(production_root)?;
    let validation = physical_lexical_path(validation_root)?;
    if production == validation
        || production.starts_with(&validation)
        || validation.starts_with(&production)
    {
        return Err(CacheError::InvalidManifest(format!(
            "production cache {} and validation cache {} must be separate, non-nested roots",
            production.display(),
            validation.display()
        )));
    }
    Ok(())
}

/// Owns the resolver, acceptance policy, and optional integrated production
/// sink for one numerical run.
pub struct ManagedArtifactCacheSession {
    profile: ManagedRunProfile,
    requested_assurance: xc_core::AssuranceLevel,
    certification_failure_policy: CertificationFailurePolicy,
    resolver: CacheResolver,
    reference_resolver: Option<CacheResolver>,
    policy: CachePolicy,
    production_sink: Option<crate::CanonicalStagingProductionSink>,
    publication_target: xc_core::PublicationTarget,
    repository_owner: String,
    execute_remote_mutations: bool,
    cache_mode: ArtifactExecutionCacheMode,
    replace_existing_publication: bool,
    output_validation: Option<OutputValidationConfig>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompletedManagedPublicationKey {
    staging_root: PathBuf,
    repository_owner: String,
    target: xc_core::PublicationTarget,
    manifest_digest: ContentDigest,
    transport_digest: ContentDigest,
}

fn completed_managed_publications() -> &'static Mutex<BTreeSet<CompletedManagedPublicationKey>> {
    static COMPLETED: OnceLock<Mutex<BTreeSet<CompletedManagedPublicationKey>>> = OnceLock::new();
    COMPLETED.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn managed_publication_key(
    draft: &crate::CanonicalProductionDraft,
    staging_root: &Path,
    repository_owner: &str,
    target: xc_core::PublicationTarget,
) -> Result<CompletedManagedPublicationKey, CacheError> {
    Ok(CompletedManagedPublicationKey {
        staging_root: staging_root.to_path_buf(),
        repository_owner: repository_owner.to_owned(),
        target,
        manifest_digest: draft.manifest.digest()?,
        transport_digest: draft.encoding.digest()?,
    })
}

fn build_required_reference_layers(
    repository_owner: &str,
    validation_root: &Path,
    mode: ManagedRemoteCacheMode,
) -> Result<Vec<crate::CacheLayer>, CacheError> {
    let mut layers = Vec::new();
    let mut precedence = 0;
    if matches!(
        mode,
        ManagedRemoteCacheMode::Private | ManagedRemoteCacheMode::PrivateThenPublic
    ) {
        let private = crate::GitHubBootstrapCacheStore::private(
            repository_owner,
            validation_root.join("reference-private"),
        )?;
        private.preflight()?;
        layers.push(crate::CacheLayer {
            precedence,
            store: Box::new(private),
        });
        precedence += 1;
    }
    if matches!(
        mode,
        ManagedRemoteCacheMode::Public | ManagedRemoteCacheMode::PrivateThenPublic
    ) {
        let public = crate::GitHubBootstrapCacheStore::public_required(
            repository_owner,
            validation_root.join("reference-public"),
        )?;
        public.preflight()?;
        layers.push(crate::CacheLayer {
            precedence,
            store: Box::new(public),
        });
    }
    Ok(layers)
}

fn reference_overlay_names(mode: ManagedRemoteCacheMode) -> Vec<String> {
    match mode {
        ManagedRemoteCacheMode::None => Vec::new(),
        ManagedRemoteCacheMode::Public => vec!["github-public".to_owned()],
        ManagedRemoteCacheMode::Private => vec!["github-private".to_owned()],
        ManagedRemoteCacheMode::PrivateThenPublic => {
            vec!["github-private".to_owned(), "github-public".to_owned()]
        }
    }
}

impl ManagedArtifactCacheSession {
    pub fn new(config: ManagedArtifactCacheConfig) -> Result<Self, CacheError> {
        let output_validation = config.output_validation.clone();
        let production_cache_installed = output_validation.is_none();
        let remote_cache_mode = output_validation
            .as_ref()
            .map_or(config.remote_cache_mode, |validation| {
                validation.reference_mode
            });
        if matches!(
            remote_cache_mode,
            ManagedRemoteCacheMode::Private | ManagedRemoteCacheMode::PrivateThenPublic
        ) {
            // Private bootstrap reads are non-interactive. Prime Git's
            // credential provider from GITHUB_TOKEN/GH_TOKEN before the first
            // registry read, including immediately after a machine restart.
            crate::GitHubCredentialApiProbe::default().prepare_git_transport()?;
        }
        let (layers, reference_resolver) = if let Some(validation) = &output_validation {
            validate_separate_cache_roots(&config.cache_root, &validation.validation_root)?;
            let execution_layers = vec![crate::CacheLayer {
                precedence: 0,
                store: Box::new(crate::ZipJsonFilesystemCacheStore::new(
                    "validation-computed",
                    validation.validation_root.join("computed"),
                    true,
                    CacheVisibility::Local,
                )),
            }];
            let reference_layers = build_required_reference_layers(
                &config.repository_owner,
                &validation.validation_root,
                validation.reference_mode,
            )?;
            (execution_layers, Some(CacheResolver::new(reference_layers)))
        } else {
            let mut ordinary_layers = vec![crate::CacheLayer {
                precedence: 0,
                store: Box::new(crate::ZipJsonFilesystemCacheStore::new(
                    "workstation",
                    &config.cache_root,
                    true,
                    CacheVisibility::Local,
                )),
            }];
            if matches!(
                remote_cache_mode,
                ManagedRemoteCacheMode::Private | ManagedRemoteCacheMode::PrivateThenPublic
            ) {
                let private_store = crate::GitHubBootstrapCacheStore::private(
                    &config.repository_owner,
                    config.cache_root.join("remote-private"),
                )?;
                private_store.preflight()?;
                ordinary_layers.push(crate::CacheLayer {
                    precedence: 1,
                    store: Box::new(private_store),
                });
            }
            if matches!(
                remote_cache_mode,
                ManagedRemoteCacheMode::Public | ManagedRemoteCacheMode::PrivateThenPublic
            ) {
                ordinary_layers.push(crate::CacheLayer {
                    precedence: 2,
                    store: Box::new(crate::GitHubBootstrapCacheStore::public(
                        &config.repository_owner,
                        config.cache_root.join("remote-public"),
                    )?),
                });
            }
            (ordinary_layers, None)
        };
        let resolver = CacheResolver::new(layers);
        let policy = CachePolicy {
            current_toolkit_version: crate::current_toolkit_version()?,
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: match remote_cache_mode {
                ManagedRemoteCacheMode::None => vec![CacheVisibility::Local],
                ManagedRemoteCacheMode::Public => {
                    vec![CacheVisibility::Local, CacheVisibility::Public]
                }
                ManagedRemoteCacheMode::Private => {
                    vec![CacheVisibility::Local, CacheVisibility::Private]
                }
                ManagedRemoteCacheMode::PrivateThenPublic => vec![
                    CacheVisibility::Local,
                    CacheVisibility::Private,
                    CacheVisibility::Public,
                ],
            },
        };
        let production_sink = config
            .cache_mode
            .may_stage_for_publication()
            .then_some(config.staging_root)
            .flatten()
            .map(|root| {
                crate::CanonicalStagingProductionSink::new(
                    root,
                    crate::TransportPolicy::default(),
                    xc_core::ResourcePolicy::default(),
                    xc_core::CancellationToken::new(),
                )
            })
            .transpose()?;
        if let Some(validation) = &output_validation {
            crate::begin_output_validation_run(crate::OutputValidationRunConfig {
                validation_root: validation.validation_root.clone(),
                report_root: validation.report_root.clone(),
                toolkit_version: crate::current_toolkit_version()?,
                reference_mode: match validation.reference_mode {
                    ManagedRemoteCacheMode::Public => "public",
                    ManagedRemoteCacheMode::Private => "private",
                    ManagedRemoteCacheMode::PrivateThenPublic => "private_public",
                    ManagedRemoteCacheMode::None => "none",
                }
                .to_owned(),
                ordered_reference_overlays: reference_overlay_names(validation.reference_mode),
                production_cache_installed,
                remote_publication_enabled: production_sink.is_some()
                    || config.execute_remote_mutations
                    || config.publication_target != xc_core::PublicationTarget::None,
            })?;
        }
        Ok(Self {
            profile: config.profile,
            requested_assurance: config.requested_assurance,
            certification_failure_policy: config.certification_failure_policy,
            resolver,
            reference_resolver,
            policy,
            production_sink,
            publication_target: config.publication_target,
            repository_owner: config.repository_owner,
            execute_remote_mutations: config.execute_remote_mutations,
            cache_mode: config.cache_mode,
            replace_existing_publication: config.replace_existing_publication,
            output_validation,
        })
    }

    pub fn from_environment() -> Result<Option<Self>, CacheError> {
        ManagedArtifactCacheConfig::from_environment()?
            .map(Self::new)
            .transpose()
    }

    pub fn context(&self) -> ArtifactCacheContext<'_> {
        ArtifactCacheContext {
            resolver: Some(&self.resolver),
            reference_resolver: self.reference_resolver.as_ref(),
            acceptance: Some(&self.policy),
            ordered_overlays: {
                let mut overlays = vec![if self.cache_mode.compares_against_reference() {
                    "validation-computed".to_owned()
                } else {
                    "workstation".to_owned()
                }];
                if self
                    .policy
                    .allowed_visibilities
                    .contains(&CacheVisibility::Private)
                {
                    overlays.push("github-private".to_owned());
                }
                if self
                    .policy
                    .allowed_visibilities
                    .contains(&CacheVisibility::Public)
                {
                    overlays.push("github-public".to_owned());
                }
                overlays
            },
            mode: self.cache_mode,
            write_on_miss: self.cache_mode.writes_computed_artifacts(),
            write_visibility: CacheVisibility::Local,
            requested_assurance: self.requested_assurance,
            certification_failure_policy: self.certification_failure_policy,
            production_sink: self
                .production_sink
                .as_ref()
                .map(|sink| sink as &dyn ArtifactProductionSink),
        }
    }

    pub fn profile(&self) -> ManagedRunProfile {
        self.profile
    }

    pub fn requested_assurance(&self) -> xc_core::AssuranceLevel {
        self.requested_assurance
    }

    pub fn certification_failure_policy(&self) -> CertificationFailurePolicy {
        self.certification_failure_policy
    }

    pub fn execute_remote_mutations(&self) -> bool {
        self.execute_remote_mutations
    }

    pub fn staged_drafts(&self) -> Result<Vec<crate::CanonicalProductionDraft>, CacheError> {
        self.production_sink
            .as_ref()
            .map(crate::CanonicalStagingProductionSink::drafts)
            .transpose()
            .map(Option::unwrap_or_default)
    }

    pub fn publication_inventory(&self) -> Result<crate::ManagedPublicationInventory, CacheError> {
        crate::build_managed_publication_inventory(
            &self.staged_drafts()?,
            self.publication_target,
            &self.repository_owner,
            self.profile,
            self.requested_assurance,
            self.certification_failure_policy,
            self.execute_remote_mutations,
        )
    }

    fn pending_staged_drafts(&self) -> Result<Vec<crate::CanonicalProductionDraft>, CacheError> {
        let Some(sink) = &self.production_sink else {
            return Ok(Vec::new());
        };
        let completed = completed_managed_publications().lock().map_err(|_| {
            CacheError::InvalidTransition(
                "managed publication completion registry lock was poisoned".to_owned(),
            )
        })?;
        self.staged_drafts()?
            .into_iter()
            .filter_map(|draft| {
                let key = managed_publication_key(
                    &draft,
                    sink.staging_root(),
                    &self.repository_owner,
                    self.publication_target,
                );
                match key {
                    Ok(key) if completed.contains(&key) => None,
                    Ok(_) => Some(Ok(draft)),
                    Err(error) => Some(Err(error)),
                }
            })
            .collect()
    }

    fn mark_staged_drafts_completed(
        &self,
        drafts: &[crate::CanonicalProductionDraft],
    ) -> Result<(), CacheError> {
        let Some(sink) = &self.production_sink else {
            return Ok(());
        };
        let mut completed = completed_managed_publications().lock().map_err(|_| {
            CacheError::InvalidTransition(
                "managed publication completion registry lock was poisoned".to_owned(),
            )
        })?;
        for draft in drafts {
            completed.insert(managed_publication_key(
                draft,
                sink.staging_root(),
                &self.repository_owner,
                self.publication_target,
            )?);
        }
        Ok(())
    }

    /// Finalizes the exact target inventory and, when the author explicitly
    /// enabled remote mutation, executes every eligible draft through the
    /// toolkit-owned resumable GitHub publisher before returning.
    pub fn finalize_publication_inventory(&self) -> Result<Option<PathBuf>, CacheError> {
        if self.output_validation.is_some() {
            return if crate::output_validation_claim_scope_is_active()? {
                crate::checkpoint_output_validation_run().map(Some)
            } else {
                crate::finalize_output_validation_run().map(Some)
            };
        }
        let Some(sink) = &self.production_sink else {
            return Ok(None);
        };
        // A CCM run creates several managed sessions over one staging root.
        // Keep every draft on disk for dependency resolution and crash resume,
        // but do not repeat successful GitHub orchestration later in the same
        // process. The registry is intentionally process-local: a restarted
        // process sees every staged draft and safely resumes from its journals.
        let pending_drafts = self.pending_staged_drafts()?;
        let pending_inventory = crate::build_managed_publication_inventory(
            &pending_drafts,
            self.publication_target,
            &self.repository_owner,
            self.profile,
            self.requested_assurance,
            self.certification_failure_policy,
            self.execute_remote_mutations,
        )?;
        // The durable inventory remains cumulative and therefore describes
        // the complete run. Only remote execution is filtered to pending work.
        let inventory = self.publication_inventory()?;
        let path = sink.staging_root().join("publication-inventory.json");
        let bytes = serde_json::to_vec_pretty(&inventory)?;
        crate::atomic_replace(&path, &bytes)?;
        if self.execute_remote_mutations
            && !pending_drafts.is_empty()
            && !pending_inventory.ready_for_remote_execution
        {
            match self.certification_failure_policy {
                CertificationFailurePolicy::RetainComputedFailRun => {
                    let ineligible = pending_inventory
                        .entries
                        .iter()
                        .filter(|entry| !entry.assurance_eligible)
                        .map(|entry| {
                            format!(
                                "{} in {} achieved {:?}",
                                entry.family, entry.repository, entry.achieved_assurance
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(CacheError::InvalidTransition(format!(
                        "publication was not executed because staged artifacts did not reach \
                         requested {:?} assurance; retained computed artifacts and inventory at \
                         {}: {ineligible}",
                        self.requested_assurance,
                        path.display()
                    )));
                }
                CertificationFailurePolicy::RetainComputedSkipPublication => {}
            }
        }
        if self.execute_remote_mutations && pending_inventory.ready_for_remote_execution {
            let report = crate::execute_managed_drafts_on_github(
                &pending_drafts,
                self.publication_target,
                &self.repository_owner,
                &sink.staging_root().join("journals"),
                &xc_core::ResourcePolicy::default(),
                self.replace_existing_publication,
            )?;
            let persisted = persist_publication_execution_report(sink.staging_root(), &report)?;
            if report.all_completed {
                self.mark_staged_drafts_completed(&pending_drafts)?;
            }
            eprintln!(
                "{}",
                format_publication_completion(
                    self.requested_assurance,
                    self.publication_target,
                    pending_drafts.len(),
                    pending_inventory.entries.len(),
                    pending_inventory.total_target_package_bytes,
                    report
                        .transactions
                        .iter()
                        .filter(|item| item.completed)
                        .count(),
                    report.transactions.len(),
                    report.current_tree_paths_removed,
                    &persisted.phase_path,
                    &persisted.cumulative_path,
                    persisted.cumulative.transactions.len(),
                    persisted.cumulative.phases.len(),
                )
            );
        }
        Ok(Some(path))
    }
}

struct PersistedPublicationExecutionReport {
    phase_path: PathBuf,
    cumulative_path: PathBuf,
    cumulative: crate::ManagedCumulativePublicationExecutionReport,
}

fn persist_publication_execution_report(
    staging_root: &Path,
    report: &crate::ManagedRunPublicationReport,
) -> Result<PersistedPublicationExecutionReport, CacheError> {
    let phase_bytes = serde_json::to_vec_pretty(report)?;
    let phase_digest = ContentDigest::sha256(&serde_json::to_vec(report)?);
    let phase_directory = staging_root.join("publication-execution-reports");
    fs::create_dir_all(&phase_directory)?;
    let mut existing_phase_paths = fs::read_dir(&phase_directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    existing_phase_paths.sort();
    let digest_suffix = format!("-{}.json", phase_digest.0);
    let phase_path = existing_phase_paths
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(&digest_suffix))
        })
        .cloned()
        .unwrap_or_else(|| {
            let next_index = existing_phase_paths
                .iter()
                .filter_map(|path| {
                    path.file_name()
                        .and_then(|value| value.to_str())
                        .and_then(|name| name.split_once('-'))
                        .and_then(|(index, _)| index.parse::<usize>().ok())
                })
                .max()
                .unwrap_or(0)
                + 1;
            phase_directory.join(format!("{next_index:020}-{}.json", phase_digest.0))
        });
    if phase_path.exists() {
        let existing: crate::ManagedRunPublicationReport =
            serde_json::from_slice(&fs::read(&phase_path)?)?;
        if existing != *report {
            return Err(CacheError::InvalidTransition(format!(
                "publication phase report digest collision at {}",
                phase_path.display()
            )));
        }
    } else {
        crate::atomic_replace(&phase_path, &phase_bytes)?;
    }

    let mut phase_paths = fs::read_dir(&phase_directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    phase_paths.sort();

    let mut phases = Vec::with_capacity(phase_paths.len());
    let mut transactions = BTreeMap::<String, crate::ManagedPublicationExecutionReport>::new();
    let mut current_tree_paths_removed = 0usize;
    for path in phase_paths {
        let bytes = fs::read(&path)?;
        let phase: crate::ManagedRunPublicationReport = serde_json::from_slice(&bytes)?;
        let digest = ContentDigest::sha256(&serde_json::to_vec(&phase)?);
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                CacheError::InvalidTransition(format!(
                    "publication phase report has an invalid filename at {}",
                    path.display()
                ))
            })?;
        let (phase_index, digest_name) = name.split_once('-').ok_or_else(|| {
            CacheError::InvalidTransition(format!(
                "publication phase report filename has no sequence at {}",
                path.display()
            ))
        })?;
        let phase_index = phase_index.parse::<usize>().map_err(|_| {
            CacheError::InvalidTransition(format!(
                "publication phase report has an invalid sequence at {}",
                path.display()
            ))
        })?;
        let expected_digest_name = format!("{}.json", digest.0);
        if digest_name != expected_digest_name {
            return Err(CacheError::InvalidTransition(format!(
                "publication phase report filename does not match its content at {}",
                path.display()
            )));
        }
        current_tree_paths_removed = current_tree_paths_removed
            .checked_add(phase.current_tree_paths_removed)
            .ok_or_else(|| {
                CacheError::InvalidTransition(
                    "cumulative publication cleanup count overflow".to_owned(),
                )
            })?;
        for transaction in &phase.transactions {
            match transactions.get(&transaction.transaction_id) {
                None => {
                    transactions.insert(transaction.transaction_id.clone(), transaction.clone());
                }
                Some(existing)
                    if existing.final_journal_digest == transaction.final_journal_digest
                        && existing.completed == transaction.completed => {}
                Some(existing) if !existing.completed && transaction.completed => {
                    transactions.insert(transaction.transaction_id.clone(), transaction.clone());
                }
                Some(existing) if existing.completed && !transaction.completed => {}
                Some(_) => {
                    return Err(CacheError::InvalidTransition(format!(
                        "conflicting publication reports for transaction {}",
                        transaction.transaction_id
                    )));
                }
            }
        }
        let completed_transaction_count = phase
            .transactions
            .iter()
            .filter(|transaction| transaction.completed)
            .count();
        let relative_report_path = path
            .strip_prefix(staging_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        phases.push(crate::ManagedPublicationExecutionPhaseSummary {
            phase_index,
            phase_digest: digest,
            relative_report_path,
            transaction_count: phase.transactions.len(),
            completed_transaction_count,
            all_completed: phase.all_completed,
            current_tree_paths_removed: phase.current_tree_paths_removed,
        });
    }
    let transactions = transactions.into_values().collect::<Vec<_>>();
    let cumulative = crate::ManagedCumulativePublicationExecutionReport {
        schema_version: 1,
        all_completed: transactions.iter().all(|transaction| transaction.completed),
        phases,
        transactions,
        current_tree_paths_removed,
    };
    let cumulative_path = staging_root.join("publication-execution-report.json");
    crate::atomic_replace(&cumulative_path, &serde_json::to_vec_pretty(&cumulative)?)?;
    Ok(PersistedPublicationExecutionReport {
        phase_path,
        cumulative_path,
        cumulative,
    })
}

#[allow(clippy::too_many_arguments)]
fn format_publication_completion(
    assurance: xc_core::AssuranceLevel,
    target: xc_core::PublicationTarget,
    staged_candidate_count: usize,
    eligible_target_candidate_count: usize,
    eligible_target_candidate_bytes: u64,
    completed_transactions: usize,
    transaction_count: usize,
    current_tree_paths_removed: usize,
    phase_report_path: &Path,
    cumulative_report_path: &Path,
    cumulative_transaction_count: usize,
    phase_count: usize,
) -> String {
    let assurance = match assurance {
        xc_core::AssuranceLevel::Computed => "computed",
        xc_core::AssuranceLevel::CrossChecked => "cross_checked",
        xc_core::AssuranceLevel::Certified => "certified",
    };
    let target = match target {
        xc_core::PublicationTarget::None => "none",
        xc_core::PublicationTarget::Private => "private",
        xc_core::PublicationTarget::Public => "public",
        xc_core::PublicationTarget::Both => "public + private",
    };
    format!(
        "Publication complete:\n  staged candidates evaluated: {staged_candidate_count}\n  \
         eligible target candidates before destination deduplication: \
         {eligible_target_candidate_count}\n  targets: {target}\n  assurance: {assurance}\n  \
         eligible packaged bytes before destination deduplication: \
         {eligible_target_candidate_bytes} bytes ({:.1} MB)\n  completed transactions: \
         {completed_transactions}/{transaction_count}\n  old current-tree paths removed: \
         {current_tree_paths_removed}\n  phase report: {}\n  cumulative report: {} \
         ({cumulative_transaction_count} transactions across {phase_count} phases)",
        eligible_target_candidate_bytes as f64 / 1_000_000.0,
        phase_report_path.display(),
        cumulative_report_path.display()
    )
}

/// Exact owned record handed from a typed numerical producer to publication
/// staging. It retains the semantic envelope that the local compatibility
/// manifest alone cannot reconstruct.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducedArtifactRecord {
    pub operation: String,
    pub semantic_key: SemanticKeyEnvelope,
    pub logical_key: String,
    pub manifest: ArtifactManifest,
    #[serde(default = "computed_artifact_assurance")]
    pub achieved_assurance: crate::ArtifactAssuranceState,
    #[serde(default)]
    pub assurance_evidence_digests: Vec<ContentDigest>,
    pub payload: Vec<u8>,
}

fn computed_artifact_assurance() -> crate::ArtifactAssuranceState {
    crate::ArtifactAssuranceState::Computed
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactProductionAssessment {
    pub achieved_assurance: crate::ArtifactAssuranceState,
    pub evidence_digests: Vec<ContentDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactAssuranceAttestation {
    pub schema_version: u32,
    pub artifact_key: ArtifactKey,
    pub content_digest: ContentDigest,
    pub achieved_assurance: crate::ArtifactAssuranceState,
    pub evidence_digests: Vec<ContentDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactAssuranceRequirement {
    pub schema_version: u32,
    pub artifact_key: ArtifactKey,
    pub content_digest: ContentDigest,
    pub required_assurance: crate::ArtifactAssuranceState,
}

impl ArtifactAssuranceRequirement {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || !self.content_digest.validate()
            || self.required_assurance == crate::ArtifactAssuranceState::Computed
        {
            return Err(CacheError::InvalidManifest(
                "artifact assurance requirement is invalid or redundant".to_owned(),
            ));
        }
        Ok(())
    }
}

impl ArtifactAssuranceAttestation {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1 || !self.content_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "artifact assurance attestation identity is invalid".to_owned(),
            ));
        }
        ArtifactProductionAssessment {
            achieved_assurance: self.achieved_assurance,
            evidence_digests: self.evidence_digests.clone(),
        }
        .validate()
    }
}

impl Default for ArtifactProductionAssessment {
    fn default() -> Self {
        Self {
            achieved_assurance: crate::ArtifactAssuranceState::Computed,
            evidence_digests: Vec::new(),
        }
    }
}

impl ArtifactProductionAssessment {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self
            .evidence_digests
            .iter()
            .any(|digest| !digest.validate())
            || self
                .evidence_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || (self.achieved_assurance == crate::ArtifactAssuranceState::CrossChecked
                && self.evidence_digests.len() < 2)
            || (self.achieved_assurance == crate::ArtifactAssuranceState::Certified
                && self.evidence_digests.is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "artifact assurance is not supported by canonical ordered evidence".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueuedProducedArtifactRecord {
    pub schema_version: u32,
    pub operation: String,
    pub semantic_key: SemanticKeyEnvelope,
    pub logical_key: String,
    pub manifest: ArtifactManifest,
    #[serde(default = "computed_artifact_assurance")]
    pub achieved_assurance: crate::ArtifactAssuranceState,
    #[serde(default)]
    pub assurance_evidence_digests: Vec<ContentDigest>,
    pub payload_file: String,
}

impl QueuedProducedArtifactRecord {
    pub fn load(&self, record_path: &Path) -> Result<ProducedArtifactRecord, CacheError> {
        if self.schema_version != 1 || self.payload_file != "payload.json.zip" {
            return Err(CacheError::InvalidManifest(
                "queued production record schema or payload path is invalid".to_owned(),
            ));
        }
        let parent = record_path.parent().ok_or_else(|| {
            CacheError::InvalidManifest("queued production record has no parent".to_owned())
        })?;
        let file = fs::File::open(parent.join(&self.payload_file))?;
        let mut archive = ZipArchive::new(file)
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        if archive.len() != 1 {
            return Err(CacheError::InvalidManifest(
                "queued production ZIP must contain exactly one entry".to_owned(),
            ));
        }
        let mut entry = archive
            .by_name("payload.json")
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        let mut payload = Vec::with_capacity(self.manifest.size_bytes as usize);
        entry.read_to_end(&mut payload)?;
        Ok(ProducedArtifactRecord {
            operation: self.operation.clone(),
            semantic_key: self.semantic_key.clone(),
            logical_key: self.logical_key.clone(),
            manifest: self.manifest.clone(),
            achieved_assurance: self.achieved_assurance,
            assurance_evidence_digests: self.assurance_evidence_digests.clone(),
            payload,
        })
    }
}

pub fn load_queued_produced_artifact(path: &Path) -> Result<ProducedArtifactRecord, CacheError> {
    let queued: QueuedProducedArtifactRecord = serde_json::from_slice(&fs::read(path)?)?;
    let artifact = queued.load(path)?;
    artifact.semantic_key.validate()?;
    artifact.manifest.validate()?;
    if artifact.manifest.key.parameters_digest != artifact.semantic_key.digest()?
        || artifact.manifest.content_digest != ContentDigest::sha256(&artifact.payload)
        || artifact.manifest.size_bytes != artifact.payload.len() as u64
    {
        return Err(CacheError::InvalidManifest(
            "queued produced artifact failed identity verification".to_owned(),
        ));
    }
    Ok(artifact)
}

pub trait ArtifactProductionSink: Send + Sync {
    fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError>;

    fn contains_artifact(
        &self,
        _key: &ArtifactKey,
        _content_digest: &ContentDigest,
    ) -> Result<bool, CacheError> {
        Ok(false)
    }

    fn retained_assurance(
        &self,
        _key: &ArtifactKey,
        _content_digest: &ContentDigest,
    ) -> Result<Option<ArtifactProductionAssessment>, CacheError> {
        Ok(None)
    }

    fn record_assurance_requirement(
        &self,
        requirement: ArtifactAssuranceRequirement,
    ) -> Result<(), CacheError>;

    fn record_assurance(&self, attestation: ArtifactAssuranceAttestation)
        -> Result<(), CacheError>;

    fn record_evidence(&self, kind: &str, bytes: &[u8]) -> Result<ContentDigest, CacheError>;
}

/// Durable application-owned queue for validated artifacts awaiting explicit
/// publication policy resolution. Queueing is local only and never implies a
/// remote publication target.
pub struct DirectoryArtifactProductionSink {
    root: PathBuf,
}

impl DirectoryArtifactProductionSink {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, CacheError> {
        let root = root.into();
        if root.as_os_str().is_empty() {
            return Err(CacheError::InvalidManifest(
                "artifact production queue root is required".to_owned(),
            ));
        }
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn artifact_root(&self, artifact: &ProducedArtifactRecord) -> Result<PathBuf, CacheError> {
        Ok(self
            .root
            .join("pending")
            .join(artifact.semantic_key.digest()?.0)
            .join(&artifact.manifest.content_digest.0))
    }

    fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), CacheError> {
        if path.exists() {
            if fs::read(path)? == bytes {
                return Ok(());
            }
            return Err(CacheError::InvalidManifest(format!(
                "artifact production queue path already contains different bytes: {}",
                path.display()
            )));
        }
        let parent = path.parent().ok_or_else(|| {
            CacheError::InvalidManifest("queued artifact path has no parent".to_owned())
        })?;
        fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(
            ".{}.{}.tmp",
            path.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        fs::write(&temporary, bytes)?;
        match fs::rename(&temporary, path) {
            Ok(()) => Ok(()),
            Err(_error) if path.exists() && fs::read(path)? == bytes => {
                let _ = fs::remove_file(&temporary);
                Ok(())
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                Err(CacheError::Io(error.to_string()))
            }
        }
    }

    #[cfg(test)]
    fn encode_payload_zip(payload: &[u8]) -> Result<Vec<u8>, CacheError> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(6))
            .last_modified_time(DateTime::default())
            .unix_permissions(0o644)
            .large_file(true);
        writer
            .start_file("payload.json", options)
            .map_err(|error| CacheError::Io(error.to_string()))?;
        writer.write_all(payload)?;
        Ok(writer
            .finish()
            .map_err(|error| CacheError::Io(error.to_string()))?
            .into_inner())
    }

    fn zip_contains_payload(path: &Path, payload: &[u8]) -> Result<bool, CacheError> {
        let file = fs::File::open(path)?;
        let mut archive =
            ZipArchive::new(file).map_err(|error| CacheError::Io(error.to_string()))?;
        if archive.len() != 1 {
            return Ok(false);
        }
        let mut entry = match archive.by_name("payload.json") {
            Ok(entry) => entry,
            Err(_) => return Ok(false),
        };
        if entry.size() != payload.len() as u64 {
            return Ok(false);
        }
        let mut offset = 0usize;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = entry.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            if payload.get(offset..offset + read) != Some(&buffer[..read]) {
                return Ok(false);
            }
            offset += read;
        }
        Ok(offset == payload.len())
    }

    fn write_payload_zip_immutable(path: &Path, payload: &[u8]) -> Result<(), CacheError> {
        if path.exists() {
            if Self::zip_contains_payload(path, payload)? {
                return Ok(());
            }
            return Err(CacheError::InvalidManifest(format!(
                "artifact production queue path already contains a different payload: {}",
                path.display()
            )));
        }
        let parent = path.parent().ok_or_else(|| {
            CacheError::InvalidManifest("queued artifact path has no parent".to_owned())
        })?;
        fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(
            ".{}.{}.tmp",
            path.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        let output = fs::File::create(&temporary)?;
        let mut writer = ZipWriter::new(output);
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(6))
            .last_modified_time(DateTime::default())
            .unix_permissions(0o644)
            .large_file(true);
        writer
            .start_file("payload.json", options)
            .map_err(|error| CacheError::Io(error.to_string()))?;
        writer.write_all(payload)?;
        writer
            .finish()
            .map_err(|error| CacheError::Io(error.to_string()))?;
        match fs::rename(&temporary, path) {
            Ok(()) => Ok(()),
            Err(_error) if path.exists() && Self::zip_contains_payload(path, payload)? => {
                let _ = fs::remove_file(&temporary);
                Ok(())
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                Err(CacheError::Io(error.to_string()))
            }
        }
    }
}

impl ArtifactProductionSink for DirectoryArtifactProductionSink {
    fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
        artifact.semantic_key.validate()?;
        artifact.manifest.validate()?;
        ArtifactProductionAssessment {
            achieved_assurance: artifact.achieved_assurance,
            evidence_digests: artifact.assurance_evidence_digests.clone(),
        }
        .validate()?;
        if artifact.operation.trim().is_empty()
            || artifact.logical_key.trim().is_empty()
            || artifact.manifest.key.parameters_digest != artifact.semantic_key.digest()?
            || artifact.manifest.content_digest != ContentDigest::sha256(&artifact.payload)
            || artifact.manifest.size_bytes != artifact.payload.len() as u64
        {
            return Err(CacheError::InvalidManifest(
                "produced artifact record identity or payload is inconsistent".to_owned(),
            ));
        }
        let artifact_root = self.artifact_root(&artifact)?;
        Self::write_payload_zip_immutable(
            &artifact_root.join("payload.json.zip"),
            &artifact.payload,
        )?;
        let queued = QueuedProducedArtifactRecord {
            schema_version: 1,
            operation: artifact.operation,
            semantic_key: artifact.semantic_key,
            logical_key: artifact.logical_key,
            manifest: artifact.manifest,
            achieved_assurance: artifact.achieved_assurance,
            assurance_evidence_digests: artifact.assurance_evidence_digests,
            payload_file: "payload.json.zip".to_owned(),
        };
        Self::write_immutable(
            &artifact_root.join("record.json"),
            &serde_json::to_vec_pretty(&queued)?,
        )
    }

    fn contains_artifact(
        &self,
        key: &ArtifactKey,
        content_digest: &ContentDigest,
    ) -> Result<bool, CacheError> {
        Ok(self
            .root
            .join("pending")
            .join(&key.parameters_digest.0)
            .join(&content_digest.0)
            .join("record.json")
            .is_file())
    }

    fn record_assurance(
        &self,
        attestation: ArtifactAssuranceAttestation,
    ) -> Result<(), CacheError> {
        attestation.validate()?;
        let root = self
            .root
            .join("assurance")
            .join(&attestation.artifact_key.parameters_digest.0)
            .join(&attestation.content_digest.0);
        let digest = crate::canonical_digest(&attestation)?;
        Self::write_immutable(
            &root.join(format!("{}.json", digest.0)),
            &serde_json::to_vec_pretty(&attestation)?,
        )
    }

    fn record_assurance_requirement(
        &self,
        requirement: ArtifactAssuranceRequirement,
    ) -> Result<(), CacheError> {
        requirement.validate()?;
        let root = self
            .root
            .join("assurance-requirements")
            .join(&requirement.artifact_key.parameters_digest.0)
            .join(&requirement.content_digest.0);
        let digest = crate::canonical_digest(&requirement)?;
        Self::write_immutable(
            &root.join(format!("{}.json", digest.0)),
            &serde_json::to_vec_pretty(&requirement)?,
        )
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
        Self::write_immutable(
            &self
                .root
                .join("evidence")
                .join(kind)
                .join(&digest.0[0..2])
                .join(&digest.0),
            bytes,
        )?;
        Ok(digest)
    }
}

/// Complete cache policy for one typed numerical artifact execution.
pub struct ArtifactExecutionCacheRequest<'a> {
    pub operation: &'a str,
    pub semantic_key: &'a SemanticKeyEnvelope,
    pub logical_key: &'a str,
    pub resolver: Option<&'a CacheResolver>,
    pub reference_resolver: Option<&'a CacheResolver>,
    pub acceptance: Option<&'a CachePolicy>,
    pub ordered_overlays: Vec<String>,
    pub mode: ArtifactExecutionCacheMode,
    pub write_on_miss: bool,
    pub write_visibility: CacheVisibility,
    pub produced_quality: CacheQuality,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub maximum_reader_version: Option<ToolkitVersion>,
    pub tags: BTreeMap<String, String>,
    pub provenance_digest: Option<ContentDigest>,
    pub production_sink: Option<&'a dyn ArtifactProductionSink>,
}

/// Result plus the exact cache/provenance decision made during execution.
pub struct ArtifactExecutionCacheResult<T> {
    pub value: T,
    pub access: CacheAccessProvenance,
    pub reused_manifest: Option<ArtifactManifest>,
    pub produced_manifest: Option<ArtifactManifest>,
}

fn validate_request(request: &ArtifactExecutionCacheRequest<'_>) -> Result<(), CacheError> {
    request.semantic_key.validate()?;
    if request.operation.trim().is_empty()
        || request.logical_key.trim().is_empty()
        || request.ordered_overlays.is_empty()
        || request
            .ordered_overlays
            .iter()
            .any(|overlay| overlay.trim().is_empty())
    {
        return Err(CacheError::InvalidManifest(
            "artifact execution cache request has an empty identity or overlay".to_owned(),
        ));
    }
    let fabric_enabled = request.mode.fabric_enabled();
    if fabric_enabled != request.resolver.is_some()
        || fabric_enabled != request.acceptance.is_some()
    {
        return Err(CacheError::InvalidManifest(
            "enabled artifact execution requires exactly one resolver and acceptance policy"
                .to_owned(),
        ));
    }
    if !request.mode.fabric_enabled() && request.write_on_miss {
        return Err(CacheError::InvalidManifest(
            "disabled cache execution cannot request a write".to_owned(),
        ));
    }
    if request.mode.requires_reuse() && request.write_on_miss {
        return Err(CacheError::InvalidManifest(
            "require-reuse execution cannot compute and write on a miss".to_owned(),
        ));
    }
    if request.mode.compares_against_reference()
        && (!request.write_on_miss
            || request.reference_resolver.is_none()
            || request.production_sink.is_some())
    {
        return Err(CacheError::InvalidManifest(
            "verify execution requires a writable isolated cache, a reference-only resolver, and no production sink"
                .to_owned(),
        ));
    }
    if !request.mode.compares_against_reference() && request.reference_resolver.is_some() {
        return Err(CacheError::InvalidManifest(
            "a reference-only resolver is valid only for verify execution".to_owned(),
        ));
    }
    if let Some(digest) = &request.provenance_digest {
        if !digest.validate() {
            return Err(CacheError::InvalidManifest(
                "artifact execution provenance digest is invalid".to_owned(),
            ));
        }
    }
    Ok(())
}

fn semantic_value(
    request: &ArtifactExecutionCacheRequest<'_>,
) -> Result<serde_json::Value, CacheError> {
    serde_json::to_value(request.semantic_key).map_err(CacheError::from)
}

fn validation_seed_source(
    tags: &BTreeMap<String, String>,
) -> Result<Option<crate::DependencyRef>, CacheError> {
    tags.get(OUTPUT_VALIDATION_SEED_TAG)
        .map(|encoded| serde_json::from_str(encoded).map_err(CacheError::from))
        .transpose()
}

fn manifest_digest(manifest: &ArtifactManifest) -> Result<ContentDigest, CacheError> {
    Ok(ContentDigest::sha256(&serde_json::to_vec(manifest)?))
}

fn manifest_semantic_key(manifest: &ArtifactManifest) -> Result<SemanticKeyEnvelope, CacheError> {
    let encoded = manifest
        .tags
        .get(SEMANTIC_KEY_MANIFEST_TAG)
        .ok_or_else(|| {
            CacheError::NotFound(format!(
            "cache dependency {} / {} predates persisted semantic envelopes; reuse the retained \
             publication staging directory once or recompute this dependency under the current schema",
            manifest.key.kind, manifest.key.logical_key
        ))
        })?;
    let semantic_key: SemanticKeyEnvelope = serde_json::from_str(encoded).map_err(|error| {
        CacheError::InvalidManifest(format!(
            "persisted dependency semantic envelope is invalid: {error}"
        ))
    })?;
    semantic_key.validate()?;
    if semantic_key.digest()? != manifest.key.parameters_digest
        || semantic_key.artifact_kind != manifest.key.kind
    {
        return Err(CacheError::InvalidManifest(
            "persisted dependency semantic envelope disagrees with its artifact key".to_owned(),
        ));
    }
    Ok(semantic_key)
}

fn emit_dependency_closure(
    resolver: &CacheResolver,
    acceptance: &CachePolicy,
    sink: &dyn ArtifactProductionSink,
    manifest: &ArtifactManifest,
    visiting: &mut BTreeSet<String>,
) -> Result<(), CacheError> {
    for dependency in &manifest.dependencies {
        if sink.contains_artifact(&dependency.key, &dependency.content_digest)? {
            continue;
        }
        let identity = format!(
            "{}\n{}\n{}\n{}",
            dependency.key.kind,
            dependency.key.logical_key,
            dependency.key.parameters_digest,
            dependency.content_digest
        );
        if !visiting.insert(identity.clone()) {
            return Err(CacheError::InvalidManifest(format!(
                "cache dependency cycle reaches {} / {}",
                dependency.key.kind, dependency.key.logical_key
            )));
        }
        let resolved = resolver.resolve(&dependency.key, acceptance)?;
        if resolved.manifest.content_digest != dependency.content_digest
            || resolved.manifest.quality < dependency.required_quality
        {
            return Err(CacheError::InvalidManifest(format!(
                "resolved dependency {} / {} does not match the exact required content or quality",
                dependency.key.kind, dependency.key.logical_key
            )));
        }
        emit_dependency_closure(resolver, acceptance, sink, &resolved.manifest, visiting)?;
        let semantic_key = manifest_semantic_key(&resolved.manifest)?;
        sink.record(ProducedArtifactRecord {
            operation: "cache.dependency.resolve".to_owned(),
            semantic_key,
            logical_key: resolved.manifest.key.logical_key.clone(),
            manifest: resolved.manifest,
            achieved_assurance: crate::ArtifactAssuranceState::Computed,
            assurance_evidence_digests: Vec::new(),
            payload: resolved.payload,
        })?;
        visiting.remove(&identity);
    }
    Ok(())
}

fn access_record(
    request: &ArtifactExecutionCacheRequest<'_>,
    semantic_digest: &ContentDigest,
    selected: Option<(&str, &ArtifactManifest)>,
    reused: bool,
    rejection: Option<String>,
) -> Result<CacheAccessProvenance, CacheError> {
    let selected_manifest_digest = selected
        .map(|(_, manifest)| manifest_digest(manifest))
        .transpose()?;
    let selected_source = selected.map(|(layer, _)| CacheSourceProvenance {
        overlay: layer.to_owned(),
        location_kind: "cache_fabric".to_owned(),
        repository: layer.to_owned(),
        revision: selected_manifest_digest
            .as_ref()
            .map_or_else(|| "unresolved".to_owned(), |digest| digest.0.clone()),
        document_paths: BTreeMap::from([(
            "payload".to_owned(),
            "content-addressed objects".to_owned(),
        )]),
    });
    let validated_artifacts = selected_manifest_digest
        .as_ref()
        .map(|digest| {
            vec![CacheValidatedArtifactProvenance {
                semantic_digest: semantic_digest.0.clone(),
                manifest_digest: digest.0.clone(),
            }]
        })
        .unwrap_or_default();
    let access = CacheAccessProvenance {
        schema_version: 1,
        operation: request.operation.to_owned(),
        artifact_family: request.semantic_key.artifact_kind.clone(),
        semantic_digest: semantic_digest.0.clone(),
        semantic_key_schema_version: request.semantic_key.schema_version,
        resolved_semantic_key: semantic_value(request)?,
        selected_manifest_digest: selected_manifest_digest.map(|digest| digest.0),
        ordered_overlays: request.ordered_overlays.clone(),
        lookup_outcome: if selected.is_some() {
            CacheLookupOutcome::Hit
        } else {
            CacheLookupOutcome::Miss
        },
        reuse_disposition: if reused {
            CacheReuseDisposition::Reused
        } else {
            CacheReuseDisposition::Recomputed
        },
        selected_source,
        rejected_candidates: rejection
            .map(|reason| {
                vec![CacheCandidateRejectionProvenance {
                    overlay: request.ordered_overlays.join(" -> "),
                    source: None,
                    stage: "resolution".to_owned(),
                    reason,
                }]
            })
            .unwrap_or_default(),
        validation_mode: CacheValidationMode::Full,
        validation_outcome: CacheValidationOutcome::Passed,
        validation_detail: Some(if reused {
            "typed payload decoded and domain validator passed".to_owned()
        } else {
            "fresh typed payload passed domain validator".to_owned()
        }),
        validated_artifacts,
    };
    access
        .validate()
        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
    Ok(access)
}

/// Resolves a validated JSON artifact or computes it under an explicit miss policy.
///
/// Cache corruption and policy failures never turn into a silent recomputation.
/// Only a typed `NotFound` outcome may reach `compute`, and a requested write
/// fails when no compatible writable overlay exists.
pub fn resolve_or_compute_json_artifact<T, Compute, Validate>(
    request: &ArtifactExecutionCacheRequest<'_>,
    compute: Compute,
    validate: Validate,
) -> Result<ArtifactExecutionCacheResult<T>, CacheError>
where
    T: Serialize + DeserializeOwned,
    Compute: FnOnce() -> Result<T, CacheError>,
    Validate: Fn(&T) -> Result<(), CacheError>,
{
    resolve_or_compute_json_artifact_with_dependencies(
        request,
        || compute().map(|value| (value, Vec::new())),
        validate,
    )
}

/// Dependency-aware form of [`resolve_or_compute_json_artifact`]. A fresh
/// computation returns the exact manifests it consumed; those immutable
/// identities are committed atomically with the new artifact manifest.
pub fn resolve_or_compute_json_artifact_with_dependencies<T, Compute, Validate>(
    request: &ArtifactExecutionCacheRequest<'_>,
    compute: Compute,
    validate: Validate,
) -> Result<ArtifactExecutionCacheResult<T>, CacheError>
where
    T: Serialize + DeserializeOwned,
    Compute: FnOnce() -> Result<(T, Vec<crate::DependencyRef>), CacheError>,
    Validate: Fn(&T) -> Result<(), CacheError>,
{
    resolve_or_compute_json_artifact_with_assessment(
        request,
        || {
            compute().map(|(value, dependencies)| {
                (value, dependencies, ArtifactProductionAssessment::default())
            })
        },
        validate,
    )
}

/// Assurance-aware production boundary. Certification or independent
/// cross-checking is performed from the retained computation state and bound
/// to the exact produced artifact before the production sink can package it.
pub fn resolve_or_compute_json_artifact_with_assessment<T, Compute, Validate>(
    request: &ArtifactExecutionCacheRequest<'_>,
    compute: Compute,
    validate: Validate,
) -> Result<ArtifactExecutionCacheResult<T>, CacheError>
where
    T: Serialize + DeserializeOwned,
    Compute:
        FnOnce()
            -> Result<(T, Vec<crate::DependencyRef>, ArtifactProductionAssessment), CacheError>,
    Validate: Fn(&T) -> Result<(), CacheError>,
{
    validate_request(request)?;
    let semantic_digest = request.semantic_key.digest()?;
    let key = ArtifactKey {
        kind: request.semantic_key.artifact_kind.clone(),
        logical_key: request.logical_key.to_owned(),
        parameters_digest: semantic_digest.clone(),
    };
    let mut miss_reason = None;
    if request.mode.consults_cache_for_result_reuse() {
        if let (Some(resolver), Some(acceptance)) = (request.resolver, request.acceptance) {
            match resolver.resolve(&key, acceptance) {
                Ok(resolved) => {
                    let value: T = serde_json::from_slice(&resolved.payload).map_err(|error| {
                        CacheError::InvalidManifest(format!(
                            "cached typed payload failed JSON decoding: {error}"
                        ))
                    })?;
                    validate(&value)?;
                    let access = access_record(
                        request,
                        &semantic_digest,
                        Some((&resolved.layer_name, &resolved.manifest)),
                        true,
                        None,
                    )?;
                    report_managed_cache_decision(request, &resolved.layer_name, "reused");
                    // A validated remote hit is promoted into the ordinary local
                    // working cache immediately. Subsequent runs are therefore
                    // offline and do not repeatedly download a large artifact.
                    if resolved.manifest.visibility != CacheVisibility::Local
                        && request.write_on_miss
                    {
                        if let Some(store) = resolver.first_writable(request.write_visibility) {
                            let mut draft = ArtifactDraft {
                                schema_version: resolved.manifest.schema_version,
                                key: resolved.manifest.key.clone(),
                                producer_toolkit_version: resolved
                                    .manifest
                                    .producer_toolkit_version
                                    .clone(),
                                minimum_reader_version: resolved
                                    .manifest
                                    .minimum_reader_version
                                    .clone(),
                                maximum_reader_version: resolved
                                    .manifest
                                    .maximum_reader_version
                                    .clone(),
                                quality: resolved.manifest.quality,
                                visibility: request.write_visibility,
                                immutable: true,
                                dependencies: resolved.manifest.dependencies.clone(),
                                tags: resolved.manifest.tags.clone(),
                                provenance_digest: resolved.manifest.provenance_digest.clone(),
                            };
                            draft
                                .tags
                                .insert("xc.cached_from".to_owned(), resolved.layer_name.clone());
                            let adopted = resolved
                                .encoded_payload
                                .as_ref()
                                .map(|encoded| {
                                    store.put_verified_encoded_payload(
                                        &draft,
                                        &resolved.manifest.content_digest,
                                        resolved.manifest.size_bytes,
                                        encoded,
                                    )
                                })
                                .transpose()?
                                .flatten();
                            if adopted.is_none() {
                                store.put(&draft, &resolved.payload)?;
                            }
                        }
                    }
                    // Publication is a boundary for newly computed artifacts,
                    // not a replay mechanism for cache hits. Re-authoring a
                    // validated remote hit changes its provenance envelope and
                    // creates redundant Git history despite identical logical
                    // payload bytes. Explicit promotion remains a separate
                    // workflow.
                    return Ok(ArtifactExecutionCacheResult {
                        value,
                        access,
                        reused_manifest: Some(resolved.manifest),
                        produced_manifest: None,
                    });
                }
                Err(CacheError::NotFound(reason)) => {
                    if request.mode.requires_reuse() {
                        return Err(CacheError::NotFound(reason));
                    }
                    miss_reason = Some(reason);
                }
                Err(error) => return Err(error),
            }
        }
    } else if request.mode.is_refresh() {
        miss_reason = Some("author refresh explicitly bypassed all cache overlays".to_owned());
    } else if request.mode.compares_against_reference() {
        miss_reason =
            Some("output validation explicitly bypassed completed-result reuse".to_owned());
    }

    let compute_started = std::time::Instant::now();
    let (value, dependencies, assessment) = compute()?;
    let compute_duration_millis = compute_started.elapsed().as_millis() as u64;
    assessment.validate()?;
    for dependency in &dependencies {
        if !dependency.key.parameters_digest.validate() || !dependency.content_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "produced artifact dependency contains an invalid digest".to_owned(),
            ));
        }
    }
    validate(&value)?;
    let payload = serde_json::to_vec(&value)?;
    let computed_dependencies = dependencies.clone();
    let produced_manifest = if request.write_on_miss {
        let resolver = request.resolver.ok_or_else(|| {
            CacheError::InvalidManifest("cache write lacks a resolver".to_owned())
        })?;
        let store = resolver
            .first_writable(request.write_visibility)
            .ok_or_else(|| CacheError::ReadOnlyLayer(format!("{:?}", request.write_visibility)))?;
        let mut tags = request.tags.clone();
        tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.to_owned(),
            String::from_utf8(crate::protocol::canonical_json_bytes(request.semantic_key)?)
                .map_err(|error| CacheError::Serialization(error.to_string()))?,
        );
        let draft = ArtifactDraft {
            schema_version: request.semantic_key.schema_version,
            key,
            producer_toolkit_version: request.producer_toolkit_version.clone(),
            minimum_reader_version: request.minimum_reader_version.clone(),
            maximum_reader_version: request.maximum_reader_version.clone(),
            quality: request.produced_quality,
            visibility: request.write_visibility,
            immutable: true,
            dependencies,
            tags,
            provenance_digest: request.provenance_digest.clone(),
        };
        let manifest = store.put(&draft, &payload)?;
        if let Some(sink) = request.production_sink {
            let acceptance = request.acceptance.ok_or_else(|| {
                CacheError::InvalidManifest(
                    "publication dependency closure lacks an acceptance policy".to_owned(),
                )
            })?;
            emit_dependency_closure(resolver, acceptance, sink, &manifest, &mut BTreeSet::new())?;
            sink.record(ProducedArtifactRecord {
                operation: request.operation.to_owned(),
                semantic_key: request.semantic_key.clone(),
                logical_key: request.logical_key.to_owned(),
                manifest: manifest.clone(),
                achieved_assurance: assessment.achieved_assurance,
                assurance_evidence_digests: assessment.evidence_digests.clone(),
                payload: payload.clone(),
            })?;
        }
        Some(manifest)
    } else {
        None
    };
    let access = if request.mode.compares_against_reference() {
        let computed_manifest = produced_manifest.as_ref().ok_or_else(|| {
            CacheError::InvalidTransition(
                "verify execution did not produce a real isolated manifest".to_owned(),
            )
        })?;
        let reference_resolver = request.reference_resolver.ok_or_else(|| {
            CacheError::InvalidManifest(
                "verify execution lacks its reference-only resolver".to_owned(),
            )
        })?;
        let acceptance = request.acceptance.ok_or_else(|| {
            CacheError::InvalidManifest("verify execution lacks cache policy".to_owned())
        })?;
        let fetch_started = std::time::Instant::now();
        let resolved_reference =
            match reference_resolver.resolve(&computed_manifest.key, acceptance) {
                Ok(reference) => Some(reference),
                Err(CacheError::NotFound(reason)) => {
                    let status = crate::record_output_comparison(crate::OutputComparisonInput {
                        operation: request.operation,
                        semantic_key: request.semantic_key,
                        computed_manifest,
                        computed_payload: &payload,
                        computed_dependencies: &computed_dependencies,
                        reference: None,
                        reference_absence_reason: Some(reason.clone()),
                        seed_source: validation_seed_source(&request.tags)?,
                        compute_duration_millis,
                        reference_fetch_duration_millis: fetch_started.elapsed().as_millis() as u64,
                    })?;
                    debug_assert_eq!(
                        status,
                        crate::ArtifactOutputComparisonStatus::ReferenceAbsent
                    );
                    None
                }
                Err(error) => return Err(error),
            };
        let (selected, status, rejection) = if let Some(reference) = &resolved_reference {
            let status = crate::record_output_comparison(crate::OutputComparisonInput {
                operation: request.operation,
                semantic_key: request.semantic_key,
                computed_manifest,
                computed_payload: &payload,
                computed_dependencies: &computed_dependencies,
                reference: Some((
                    &reference.layer_name,
                    &reference.manifest,
                    &reference.payload,
                )),
                reference_absence_reason: None,
                seed_source: validation_seed_source(&request.tags)?,
                compute_duration_millis,
                reference_fetch_duration_millis: fetch_started.elapsed().as_millis() as u64,
            })?;
            (
                Some((reference.layer_name.as_str(), &reference.manifest)),
                status,
                None,
            )
        } else {
            (
                None,
                crate::ArtifactOutputComparisonStatus::ReferenceAbsent,
                Some("reference artifact was absent".to_owned()),
            )
        };
        let mut access = access_record(request, &semantic_digest, selected, false, rejection)?;
        if selected.is_some() {
            access.reuse_disposition = CacheReuseDisposition::InspectedOnly;
        }
        access.validation_outcome = if status == crate::ArtifactOutputComparisonStatus::Match {
            CacheValidationOutcome::Passed
        } else {
            CacheValidationOutcome::Failed
        };
        access.validation_detail = Some(match status {
            crate::ArtifactOutputComparisonStatus::Match => {
                "fresh logical payload is byte-identical to the inspected reference".to_owned()
            }
            crate::ArtifactOutputComparisonStatus::Mismatch => {
                "fresh logical payload differs from the inspected reference".to_owned()
            }
            crate::ArtifactOutputComparisonStatus::ReferenceAbsent => {
                "fresh logical payload has no reference artifact at the same identity".to_owned()
            }
        });
        access
            .validate()
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        access
    } else {
        access_record(request, &semantic_digest, None, false, miss_reason)?
    };
    let report_source = if request.mode.compares_against_reference() {
        "validation-computed"
    } else {
        "workstation"
    };
    report_managed_cache_decision(request, report_source, "computed");
    Ok(ArtifactExecutionCacheResult {
        value,
        access,
        reused_manifest: None,
        produced_manifest,
    })
}

fn report_managed_cache_decision(
    request: &ArtifactExecutionCacheRequest<'_>,
    source: &str,
    outcome: &str,
) {
    // CCM summarizes large quadrature precompute batches once; printing one
    // line per rule can produce hundreds of low-value lines.
    if request.semantic_key.artifact_kind == "gauss_legendre_rule" {
        return;
    }
    if request
        .ordered_overlays
        .iter()
        .any(|overlay| overlay.starts_with("github-"))
    {
        eprintln!(
            "  cache artifact: {} ({outcome}, source={source})",
            request.semantic_key.artifact_kind
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheLayer, DependencyRef, FilesystemCacheStore};
    use serde_json::json;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    struct FaultingReferenceStore;

    impl crate::CacheStore for FaultingReferenceStore {
        fn name(&self) -> &str {
            "faulting-reference"
        }

        fn writable(&self) -> bool {
            false
        }

        fn visibility(&self) -> CacheVisibility {
            CacheVisibility::Local
        }

        fn put(
            &self,
            _draft: &ArtifactDraft,
            _payload: &[u8],
        ) -> Result<ArtifactManifest, CacheError> {
            Err(CacheError::ReadOnlyLayer(self.name().to_owned()))
        }

        fn candidates(&self, _key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
            Err(CacheError::Io(
                "simulated reference transport failure".to_owned(),
            ))
        }

        fn read_payload_to(
            &self,
            _manifest: &ArtifactManifest,
            _writer: &mut dyn Write,
        ) -> Result<(), CacheError> {
            Err(CacheError::Io(
                "simulated reference transport failure".to_owned(),
            ))
        }
    }

    #[test]
    fn computed_assurance_is_the_managed_workflow_default() {
        assert_eq!(
            parse_requested_assurance(None).unwrap(),
            xc_core::AssuranceLevel::Computed
        );
        assert_eq!(
            parse_requested_assurance(Some("cross_checked")).unwrap(),
            xc_core::AssuranceLevel::CrossChecked
        );
        assert_eq!(
            parse_requested_assurance(Some("certified")).unwrap(),
            xc_core::AssuranceLevel::Certified
        );
    }

    #[test]
    fn cache_mode_predicates_keep_verify_compute_compare_and_seed_semantics() {
        let verify = ArtifactExecutionCacheMode::VerifyAgainstReference;
        assert!(verify.fabric_enabled());
        assert!(!verify.consults_cache_for_result_reuse());
        assert!(verify.consults_overlays_for_route_selection());
        assert!(verify.may_use_cached_seed());
        assert!(verify.writes_computed_artifacts());
        assert!(verify.compares_against_reference());
        assert!(!verify.may_stage_for_publication());
        assert!(!verify.requires_reuse());
        assert!(!verify.is_refresh());

        assert_eq!(parse_cache_mode(Some("verify")).unwrap(), verify);
        assert_eq!(
            parse_validation_reference_mode(None).unwrap(),
            ManagedRemoteCacheMode::PrivateThenPublic
        );
        assert!(parse_validation_reference_mode(Some("none")).is_err());
    }

    #[test]
    fn validation_cache_root_cannot_overlap_the_production_cache() {
        let production = root("production-cache-root");
        assert!(validate_separate_cache_roots(&production, &production).is_err());
        assert!(
            validate_separate_cache_roots(&production, &production.join("validation")).is_err()
        );
        assert!(
            validate_separate_cache_roots(&production.join("nested-production"), &production)
                .is_err()
        );
        assert!(validate_separate_cache_roots(
            &production,
            &production.with_file_name("validation-cache-root")
        )
        .is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn validation_cache_root_rejects_a_symlink_alias_of_production() {
        use std::os::unix::fs::symlink;

        let root = root("validation-cache-symlink-alias");
        let _ = fs::remove_dir_all(&root);
        let production = root.join("production");
        let alias = root.join("validation-alias");
        fs::create_dir_all(&production).unwrap();
        symlink(&production, &alias).unwrap();
        assert!(validate_separate_cache_roots(&production, &alias).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn completed_publication_has_a_visible_ascii_summary() {
        let summary = format_publication_completion(
            xc_core::AssuranceLevel::Computed,
            xc_core::PublicationTarget::Both,
            3,
            6,
            61_082_844,
            2,
            2,
            7,
            Path::new("staging/publication-execution-reports/phase.json"),
            Path::new("staging/publication-execution-report.json"),
            10,
            3,
        );
        assert_eq!(
            summary,
            concat!(
                "Publication complete:\n",
                "  staged candidates evaluated: 3\n",
                "  eligible target candidates before destination deduplication: 6\n",
                "  targets: public + private\n",
                "  assurance: computed\n",
                "  eligible packaged bytes before destination deduplication: ",
                "61082844 bytes (61.1 MB)\n",
                "  completed transactions: 2/2\n",
                "  old current-tree paths removed: 7\n",
                "  phase report: staging/publication-execution-reports/phase.json\n",
                "  cumulative report: staging/publication-execution-report.json ",
                "(10 transactions across 3 phases)"
            )
        );
        assert!(summary.is_ascii());
    }

    #[test]
    fn publication_reports_preserve_phases_and_build_a_cumulative_run_report() {
        fn phase(ids: &[&str], removed: usize) -> crate::ManagedRunPublicationReport {
            crate::ManagedRunPublicationReport {
                schema_version: 1,
                transactions: ids
                    .iter()
                    .map(|id| crate::ManagedPublicationExecutionReport {
                        transaction_id: (*id).to_owned(),
                        completed: true,
                        steps_executed: 4,
                        final_journal_digest: ContentDigest::sha256(id.as_bytes()),
                    })
                    .collect(),
                all_completed: true,
                current_tree_paths_removed: removed,
            }
        }

        let root = root("cumulative-publication-report");
        let _ = fs::remove_dir_all(&root);
        let first = phase(&["transaction-a", "transaction-b"], 2);
        let second = phase(&["transaction-c"], 3);

        let first_persisted = persist_publication_execution_report(&root, &first).unwrap();
        assert!(first_persisted.phase_path.exists());
        assert_eq!(first_persisted.cumulative.phases.len(), 1);
        assert_eq!(first_persisted.cumulative.transactions.len(), 2);
        assert_eq!(first_persisted.cumulative.current_tree_paths_removed, 2);

        let second_persisted = persist_publication_execution_report(&root, &second).unwrap();
        assert!(second_persisted.phase_path.exists());
        assert_eq!(second_persisted.cumulative.phases.len(), 2);
        assert_eq!(second_persisted.cumulative.phases[0].phase_index, 1);
        assert_eq!(second_persisted.cumulative.phases[1].phase_index, 2);
        assert_eq!(second_persisted.cumulative.transactions.len(), 3);
        assert_eq!(second_persisted.cumulative.current_tree_paths_removed, 5);
        assert!(second_persisted.cumulative.all_completed);

        let repeated = persist_publication_execution_report(&root, &first).unwrap();
        assert_eq!(repeated.cumulative.phases.len(), 2);
        assert_eq!(repeated.cumulative.transactions.len(), 3);
        let durable: crate::ManagedCumulativePublicationExecutionReport =
            serde_json::from_slice(&fs::read(&repeated.cumulative_path).unwrap()).unwrap();
        assert_eq!(durable, repeated.cumulative);
        assert_eq!(
            fs::read_dir(root.join("publication-execution-reports"))
                .unwrap()
                .count(),
            2
        );
        let _ = fs::remove_dir_all(root);
    }

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<ProducedArtifactRecord>>);

    impl ArtifactProductionSink for RecordingSink {
        fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
            self.0.lock().unwrap().push(artifact);
            Ok(())
        }

        fn contains_artifact(
            &self,
            _key: &ArtifactKey,
            _content_digest: &ContentDigest,
        ) -> Result<bool, CacheError> {
            Ok(false)
        }

        fn record_assurance(
            &self,
            _attestation: ArtifactAssuranceAttestation,
        ) -> Result<(), CacheError> {
            Ok(())
        }

        fn record_assurance_requirement(
            &self,
            _requirement: ArtifactAssuranceRequirement,
        ) -> Result<(), CacheError> {
            Ok(())
        }

        fn record_evidence(&self, _kind: &str, bytes: &[u8]) -> Result<ContentDigest, CacheError> {
            Ok(ContentDigest::sha256(bytes))
        }
    }

    fn root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{name}-{}", std::process::id()))
    }

    fn semantic_key() -> SemanticKeyEnvelope {
        SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "quadrature_rule".to_owned(),
            mathematical_semantics_version: "gauss-legendre-v1".to_owned(),
            resolved_mathematical_parameters: json!({"order": 4, "precision_bits": 128}),
            normalization: Some("nodes_on_minus_one_one".to_owned()),
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        }
    }

    fn policy() -> CachePolicy {
        CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        }
    }

    fn request<'a>(
        key: &'a SemanticKeyEnvelope,
        resolver: &'a CacheResolver,
        policy: &'a CachePolicy,
        mode: ArtifactExecutionCacheMode,
    ) -> ArtifactExecutionCacheRequest<'a> {
        ArtifactExecutionCacheRequest {
            operation: "quadrature.load_or_compute",
            semantic_key: key,
            logical_key: "gauss-legendre/4/128",
            resolver: Some(resolver),
            reference_resolver: None,
            acceptance: Some(policy),
            ordered_overlays: vec!["workstation".to_owned()],
            mode,
            write_on_miss: mode.writes_computed_artifacts(),
            write_visibility: CacheVisibility::Local,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            tags: BTreeMap::new(),
            provenance_digest: None,
            production_sink: None,
        }
    }

    #[test]
    fn one_contract_computes_writes_reuses_and_records_provenance() {
        let root = root("execution-cache-roundtrip");
        let _ = fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let key = semantic_key();
        let policy = policy();
        let calls = AtomicUsize::new(0);
        let first = resolve_or_compute_json_artifact(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::PreferReuse,
            ),
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(vec!["node-a".to_owned(), "node-b".to_owned()])
            },
            |value| {
                if value.len() == 2 {
                    Ok(())
                } else {
                    Err(CacheError::InvalidManifest(
                        "wrong vector length".to_owned(),
                    ))
                }
            },
        )
        .unwrap();
        assert_eq!(first.access.lookup_outcome, CacheLookupOutcome::Miss);
        assert!(first.produced_manifest.is_some());

        let second = resolve_or_compute_json_artifact(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::PreferReuse,
            ),
            || -> Result<Vec<String>, CacheError> { panic!("cache hit must not recompute") },
            |value| {
                if value.len() == 2 {
                    Ok(())
                } else {
                    Err(CacheError::InvalidManifest(
                        "wrong vector length".to_owned(),
                    ))
                }
            },
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.value, first.value);
        assert_eq!(second.access.lookup_outcome, CacheLookupOutcome::Hit);
        assert_eq!(
            second.access.reuse_disposition,
            CacheReuseDisposition::Reused
        );
        assert!(second.reused_manifest.is_some());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn refresh_bypasses_an_existing_hit_and_publishes_the_fresh_value_locally() {
        let root = root("execution-cache-refresh");
        let _ = fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let key = semantic_key();
        let policy = policy();
        resolve_or_compute_json_artifact(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::PreferReuse,
            ),
            || Ok(vec!["old".to_owned()]),
            |_| Ok(()),
        )
        .unwrap();
        let refreshed = resolve_or_compute_json_artifact(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::Refresh,
            ),
            || Ok(vec!["new".to_owned()]),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(refreshed.value, vec!["new"]);
        assert_eq!(refreshed.access.lookup_outcome, CacheLookupOutcome::Miss);
        assert_eq!(
            refreshed.access.reuse_disposition,
            CacheReuseDisposition::Recomputed
        );
        assert!(refreshed.reused_manifest.is_none());
        assert!(refreshed.produced_manifest.is_some());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn require_reuse_miss_never_computes() {
        let root = root("execution-cache-required-miss");
        let _ = fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let key = semantic_key();
        let policy = policy();
        let result = resolve_or_compute_json_artifact(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::RequireReuse,
            ),
            || -> Result<Vec<String>, CacheError> { panic!("required miss must not compute") },
            |_| Ok(()),
        );
        assert!(matches!(result, Err(CacheError::NotFound(_))));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn produced_dependencies_are_committed_and_reused_exactly() {
        let root = root("execution-cache-dependencies");
        let _ = fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let key = semantic_key();
        let policy = policy();
        let dependency = DependencyRef {
            key: ArtifactKey::new("gauss_legendre_rule", "gl/4/128", b"gl").unwrap(),
            content_digest: ContentDigest::sha256(b"quadrature payload"),
            required_quality: CacheQuality::Validated,
        };
        let first = resolve_or_compute_json_artifact_with_dependencies(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::PreferReuse,
            ),
            || Ok((vec!["matrix".to_owned()], vec![dependency.clone()])),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(
            first.produced_manifest.as_ref().unwrap().dependencies,
            vec![dependency.clone()]
        );
        let second = resolve_or_compute_json_artifact_with_dependencies(
            &request(
                &key,
                &resolver,
                &policy,
                ArtifactExecutionCacheMode::PreferReuse,
            ),
            || -> Result<(Vec<String>, Vec<DependencyRef>), CacheError> {
                panic!("cache hit must retain persisted dependencies")
            },
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(
            second.reused_manifest.as_ref().unwrap().dependencies,
            vec![dependency]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn fresh_publication_staging_reconstructs_cached_dependency_closure() {
        let root = root("execution-cache-fresh-staging-closure");
        let _ = fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                root.join("cache"),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = policy();
        let dependency_key = semantic_key();
        let dependency_request = request(
            &dependency_key,
            &resolver,
            &policy,
            ArtifactExecutionCacheMode::PreferReuse,
        );
        let dependency = resolve_or_compute_json_artifact(
            &dependency_request,
            || Ok(vec!["quadrature".to_owned()]),
            |_| Ok(()),
        )
        .unwrap()
        .produced_manifest
        .unwrap();

        let parent_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_tau_matrix".to_owned(),
            mathematical_semantics_version: "ccm-fixture-v1".to_owned(),
            resolved_mathematical_parameters: json!({"n_modes": 2}),
            normalization: Some("row_major".to_owned()),
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let mut parent_request = request(
            &parent_key,
            &resolver,
            &policy,
            ArtifactExecutionCacheMode::PreferReuse,
        );
        parent_request.logical_key = "ccm/tau/fixture";
        let sink = crate::CanonicalStagingProductionSink::new(
            root.join("fresh-staging"),
            crate::TransportPolicy::default(),
            xc_core::ResourcePolicy::default(),
            xc_core::CancellationToken::new(),
        )
        .unwrap();
        parent_request.production_sink = Some(&sink);
        resolve_or_compute_json_artifact_with_dependencies(
            &parent_request,
            || {
                Ok((
                    vec!["tau".to_owned()],
                    vec![DependencyRef {
                        key: dependency.key.clone(),
                        content_digest: dependency.content_digest.clone(),
                        required_quality: CacheQuality::Validated,
                    }],
                ))
            },
            |_| Ok(()),
        )
        .unwrap();
        let drafts = sink.drafts().unwrap();
        assert_eq!(drafts.len(), 2);
        assert!(drafts.iter().any(|draft| draft.family == "quadrature"));
        assert!(drafts.iter().any(|draft| draft.family == "ccm-matrices"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn production_sink_records_fresh_but_not_reused_artifacts() {
        let root = root("execution-cache-production-sink");
        let _ = fs::remove_dir_all(&root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let key = semantic_key();
        let policy = policy();
        let sink = RecordingSink::default();
        let mut first_request = request(
            &key,
            &resolver,
            &policy,
            ArtifactExecutionCacheMode::PreferReuse,
        );
        first_request.production_sink = Some(&sink);
        let expected = vec!["node-a".to_owned(), "node-b".to_owned()];
        resolve_or_compute_json_artifact(&first_request, || Ok(expected.clone()), |_| Ok(()))
            .unwrap();

        let mut second_request = request(
            &key,
            &resolver,
            &policy,
            ArtifactExecutionCacheMode::PreferReuse,
        );
        second_request.production_sink = Some(&sink);
        resolve_or_compute_json_artifact(
            &second_request,
            || -> Result<Vec<String>, CacheError> { panic!("cache hit must not recompute") },
            |_| Ok(()),
        )
        .unwrap();

        let records = sink.0.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].semantic_key, key);
        assert_eq!(records[0].logical_key, "gauss-legendre/4/128");
        assert_eq!(records[0].payload, serde_json::to_vec(&expected).unwrap());
        assert_eq!(
            records[0].manifest.content_digest,
            ContentDigest::sha256(&records[0].payload)
        );
        drop(records);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn certified_assessment_is_bound_before_packaging() {
        let first_root = root("execution-cache-certified-assessment");
        let _ = fs::remove_dir_all(&first_root);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &first_root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let key = semantic_key();
        let policy = policy();
        let sink = RecordingSink::default();
        let mut cache_request = request(
            &key,
            &resolver,
            &policy,
            ArtifactExecutionCacheMode::PreferReuse,
        );
        cache_request.production_sink = Some(&sink);
        let certificate = ContentDigest::sha256(b"portable interval certificate");
        resolve_or_compute_json_artifact_with_assessment(
            &cache_request,
            || {
                Ok((
                    vec!["certified-matrix".to_owned()],
                    Vec::new(),
                    ArtifactProductionAssessment {
                        achieved_assurance: crate::ArtifactAssuranceState::Certified,
                        evidence_digests: vec![certificate.clone()],
                    },
                ))
            },
            |_| Ok(()),
        )
        .unwrap();
        let records = sink.0.lock().unwrap();
        assert_eq!(
            records[0].achieved_assurance,
            crate::ArtifactAssuranceState::Certified
        );
        assert_eq!(records[0].assurance_evidence_digests, vec![certificate]);
        drop(records);

        let missing_root = root("execution-cache-certified-missing-evidence");
        let _ = fs::remove_dir_all(&missing_root);
        let missing_resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                &missing_root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let missing_request = request(
            &key,
            &missing_resolver,
            &policy,
            ArtifactExecutionCacheMode::PreferReuse,
        );
        let error = match resolve_or_compute_json_artifact_with_assessment(
            &missing_request,
            || {
                Ok((
                    vec!["not-certified".to_owned()],
                    Vec::new(),
                    ArtifactProductionAssessment {
                        achieved_assurance: crate::ArtifactAssuranceState::Certified,
                        evidence_digests: Vec::new(),
                    },
                ))
            },
            |_| Ok(()),
        ) {
            Ok(_) => panic!("certified assessment without evidence must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, CacheError::InvalidManifest(_)));
        let _ = fs::remove_dir_all(first_root);
        let _ = fs::remove_dir_all(missing_root);
    }

    #[test]
    fn directory_production_sink_queues_identity_bound_payload_once() {
        let root = root("execution-cache-directory-sink");
        let _ = fs::remove_dir_all(&root);
        let sink = DirectoryArtifactProductionSink::new(&root).unwrap();
        let semantic_key = semantic_key();
        let payload = serde_json::to_vec(&vec!["node-a", "node-b"]).unwrap();
        let expected_zip = DirectoryArtifactProductionSink::encode_payload_zip(&payload).unwrap();
        let semantic_digest = semantic_key.digest().unwrap();
        let record = ProducedArtifactRecord {
            operation: "quadrature.load_or_compute".to_owned(),
            semantic_key,
            logical_key: "gauss-legendre/4/128".to_owned(),
            manifest: ArtifactManifest {
                schema_version: 1,
                key: ArtifactKey {
                    kind: "quadrature_rule".to_owned(),
                    logical_key: "gauss-legendre/4/128".to_owned(),
                    parameters_digest: semantic_digest.clone(),
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
            achieved_assurance: crate::ArtifactAssuranceState::Computed,
            assurance_evidence_digests: Vec::new(),
            payload: payload.clone(),
        };
        sink.record(record.clone()).unwrap();
        sink.record(record.clone()).unwrap();
        let artifact_root = root
            .join("pending")
            .join(semantic_digest.0)
            .join(&record.manifest.content_digest.0);
        let saved = load_queued_produced_artifact(&artifact_root.join("record.json")).unwrap();
        assert_eq!(saved, record);
        let payload_zip = artifact_root.join("payload.json.zip");
        assert!(payload_zip.is_file());
        assert_eq!(fs::read(payload_zip).unwrap(), expected_zip);
        assert!(!artifact_root.join("payload.json").exists());
        assert!(
            fs::metadata(artifact_root.join("record.json"))
                .unwrap()
                .len()
                < 4096
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn successful_same_process_publication_filter_retains_restart_staging() {
        let root = root("managed-completed-publication-filter");
        let _ = fs::remove_dir_all(&root);
        let session = ManagedArtifactCacheSession::new(ManagedArtifactCacheConfig {
            profile: ManagedRunProfile::Author,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            cache_root: root.join("cache"),
            staging_root: Some(root.join("staging")),
            publication_target: xc_core::PublicationTarget::Public,
            repository_owner: "ManagedCompletionFilterFixture".to_owned(),
            remote_cache_mode: ManagedRemoteCacheMode::None,
            cache_mode: ArtifactExecutionCacheMode::PreferReuse,
            replace_existing_publication: false,
            execute_remote_mutations: false,
            output_validation: None,
        })
        .unwrap();
        let key = semantic_key();
        let cache = session.context();
        let request = ArtifactExecutionCacheRequest {
            operation: "quadrature.load_or_compute",
            semantic_key: &key,
            logical_key: "gauss-legendre/4/128",
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays,
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            tags: BTreeMap::new(),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };
        resolve_or_compute_json_artifact(
            &request,
            || Ok(vec!["node-a".to_owned(), "node-b".to_owned()]),
            |_| Ok(()),
        )
        .unwrap();

        let pending = session.pending_staged_drafts().unwrap();
        assert_eq!(pending.len(), 1);
        session.mark_staged_drafts_completed(&pending).unwrap();
        assert!(session.pending_staged_drafts().unwrap().is_empty());
        assert_eq!(session.staged_drafts().unwrap().len(), 1);
        assert_eq!(session.publication_inventory().unwrap().entries.len(), 1);
        session.finalize_publication_inventory().unwrap();
        let durable_inventory: crate::ManagedPublicationInventory = serde_json::from_slice(
            &fs::read(root.join("staging/publication-inventory.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(durable_inventory.entries.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn managed_author_run_retains_ineligible_artifacts_and_obeys_failure_policy() {
        fn stage_computed_quadrature(
            session: &ManagedArtifactCacheSession,
        ) -> Result<ArtifactManifest, CacheError> {
            let key = semantic_key();
            let cache = session.context();
            let request = ArtifactExecutionCacheRequest {
                operation: "quadrature.load_or_compute",
                semantic_key: &key,
                logical_key: "gauss-legendre/4/128",
                resolver: cache.resolver,
                reference_resolver: cache.reference_resolver,
                acceptance: cache.acceptance,
                ordered_overlays: cache.ordered_overlays,
                mode: cache.mode,
                write_on_miss: cache.write_on_miss,
                write_visibility: cache.write_visibility,
                produced_quality: CacheQuality::Validated,
                producer_toolkit_version: ToolkitVersion::parse("0.13.0")?,
                minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
                maximum_reader_version: None,
                tags: BTreeMap::new(),
                provenance_digest: None,
                production_sink: cache.production_sink,
            };
            let resolved = resolve_or_compute_json_artifact(
                &request,
                || Ok(vec!["node-a".to_owned(), "node-b".to_owned()]),
                |_| Ok(()),
            )?;
            resolved
                .produced_manifest
                .or(resolved.reused_manifest)
                .ok_or_else(|| CacheError::NotFound("managed test artifact manifest".to_owned()))
        }

        for (policy, must_fail) in [
            (CertificationFailurePolicy::RetainComputedFailRun, true),
            (
                CertificationFailurePolicy::RetainComputedSkipPublication,
                false,
            ),
        ] {
            let root = root(match policy {
                CertificationFailurePolicy::RetainComputedFailRun => "managed-fail-run",
                CertificationFailurePolicy::RetainComputedSkipPublication => {
                    "managed-skip-publication"
                }
            });
            let _ = fs::remove_dir_all(&root);
            let staging_root = root.join("staging");
            let session = ManagedArtifactCacheSession::new(ManagedArtifactCacheConfig {
                profile: ManagedRunProfile::Author,
                requested_assurance: xc_core::AssuranceLevel::Certified,
                certification_failure_policy: policy,
                cache_root: root.join("cache"),
                staging_root: Some(staging_root.clone()),
                publication_target: xc_core::PublicationTarget::Both,
                repository_owner: "TeamXcelerator".to_owned(),
                remote_cache_mode: ManagedRemoteCacheMode::None,
                cache_mode: ArtifactExecutionCacheMode::PreferReuse,
                replace_existing_publication: false,
                execute_remote_mutations: true,
                output_validation: None,
            })
            .unwrap();
            let manifest = stage_computed_quadrature(&session).unwrap();
            session
                .context()
                .production_sink
                .unwrap()
                .record_assurance_requirement(ArtifactAssuranceRequirement {
                    schema_version: 1,
                    artifact_key: manifest.key,
                    content_digest: manifest.content_digest,
                    required_assurance: crate::ArtifactAssuranceState::Certified,
                })
                .unwrap();
            let result = session.finalize_publication_inventory();
            assert_eq!(result.is_err(), must_fail);
            let inventory: crate::ManagedPublicationInventory = serde_json::from_slice(
                &fs::read(staging_root.join("publication-inventory.json")).unwrap(),
            )
            .unwrap();
            assert!(!inventory.ready_for_remote_execution);
            assert!(inventory.entries.iter().all(|entry| {
                !entry.assurance_eligible
                    && entry.achieved_assurance == crate::ArtifactAssuranceState::Computed
            }));
            assert!(staging_root.join("drafts").is_dir());
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn verify_mode_writes_real_artifacts_compares_and_rejects_empty_runs() {
        let _validation_guard = crate::output_validation_test_lock().lock().unwrap();
        let root = root("execution-cache-output-validation");
        let _ = fs::remove_dir_all(&root);
        let reference_root = root.join("reference");
        let reference_writer = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "reference-writer",
                &reference_root,
                true,
                CacheVisibility::Local,
            )),
        }]);
        let semantic = semantic_key();
        let policy = policy();
        let expected = vec!["node-a".to_owned(), "node-b".to_owned()];
        resolve_or_compute_json_artifact(
            &request(
                &semantic,
                &reference_writer,
                &policy,
                ArtifactExecutionCacheMode::PreferReuse,
            ),
            || Ok(expected.clone()),
            |_| Ok(()),
        )
        .unwrap();

        let run_once = |name: &str,
                        run_semantic: &SemanticKeyEnvelope,
                        computed_value: Vec<String>| {
            let validation_root = root.join(name);
            let report_root = validation_root.join("reports");
            crate::begin_output_validation_run(crate::OutputValidationRunConfig {
                validation_root: validation_root.clone(),
                report_root: report_root.clone(),
                toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
                reference_mode: "local_fixture".to_owned(),
                ordered_reference_overlays: vec!["reference".to_owned()],
                production_cache_installed: false,
                remote_publication_enabled: false,
            })
            .unwrap();
            let computed_resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(crate::ZipJsonFilesystemCacheStore::new(
                    "validation-computed",
                    validation_root.join("computed"),
                    true,
                    CacheVisibility::Local,
                )),
            }]);
            let reference_resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "reference",
                    &reference_root,
                    false,
                    CacheVisibility::Local,
                )),
            }]);
            let calls = AtomicUsize::new(0);
            let verify_request = ArtifactExecutionCacheRequest {
                operation: "quadrature.load_or_compute",
                semantic_key: run_semantic,
                logical_key: "gauss-legendre/4/128",
                resolver: Some(&computed_resolver),
                reference_resolver: Some(&reference_resolver),
                acceptance: Some(&policy),
                ordered_overlays: vec!["validation-computed".to_owned(), "reference".to_owned()],
                mode: ArtifactExecutionCacheMode::VerifyAgainstReference,
                write_on_miss: true,
                write_visibility: CacheVisibility::Local,
                produced_quality: CacheQuality::Validated,
                producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
                maximum_reader_version: None,
                tags: BTreeMap::new(),
                provenance_digest: None,
                production_sink: None,
            };
            let result = resolve_or_compute_json_artifact(
                &verify_request,
                || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(computed_value)
                },
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(result.reused_manifest.is_none());
            let manifest = result.produced_manifest.unwrap();
            assert_eq!(manifest.visibility, CacheVisibility::Local);
            let computed = computed_resolver.resolve(&manifest.key, &policy).unwrap();
            assert_eq!(computed.manifest.content_digest, manifest.content_digest);
            assert_eq!(
                result.access.reuse_disposition,
                if run_semantic == &semantic {
                    CacheReuseDisposition::InspectedOnly
                } else {
                    CacheReuseDisposition::Recomputed
                }
            );
            (report_root, crate::finalize_output_validation_run())
        };

        let (matching_report_root, matching_result) =
            run_once("matching", &semantic, expected.clone());
        assert!(matching_result.is_ok());
        let matching: crate::OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(matching_report_root.join("latest.json")).unwrap())
                .unwrap();
        assert!(matching.output_preserving);
        assert_eq!(matching.totals.matched, 1);

        let (mismatch_report_root, mismatch_result) =
            run_once("mismatch", &semantic, vec!["different".to_owned()]);
        assert!(mismatch_result.is_err());
        let mismatch: crate::OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(mismatch_report_root.join("latest.json")).unwrap())
                .unwrap();
        assert!(!mismatch.output_preserving);
        assert_eq!(mismatch.totals.mismatched, 1);
        assert_eq!(mismatch.totals.first_divergences, 1);

        let mut absent_semantic = semantic.clone();
        absent_semantic.resolved_mathematical_parameters =
            json!({"order": 8, "precision_bits": 128});
        let (absent_report_root, absent_result) =
            run_once("absent", &absent_semantic, vec!["new-node".to_owned()]);
        assert!(absent_result.is_err());
        let absent: crate::OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(absent_report_root.join("latest.json")).unwrap())
                .unwrap();
        assert_eq!(absent.totals.reference_absent, 1);
        assert!(!absent.output_preserving);

        let empty_root = root.join("empty");
        crate::begin_output_validation_run(crate::OutputValidationRunConfig {
            validation_root: empty_root.clone(),
            report_root: empty_root.join("reports"),
            toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            reference_mode: "local_fixture".to_owned(),
            ordered_reference_overlays: vec!["reference".to_owned()],
            production_cache_installed: false,
            remote_publication_enabled: false,
        })
        .unwrap();
        assert!(crate::finalize_output_validation_run().is_err());
        let empty: crate::OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(empty_root.join("reports/latest.json")).unwrap())
                .unwrap();
        assert_eq!(empty.totals.compared, 0);
        assert!(!empty.output_preserving);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verify_mode_never_reports_operational_reference_errors_as_absence() {
        let _validation_guard = crate::output_validation_test_lock().lock().unwrap();
        let root = root("execution-cache-output-validation-operational-error");
        let _ = fs::remove_dir_all(&root);
        crate::begin_output_validation_run(crate::OutputValidationRunConfig {
            validation_root: root.clone(),
            report_root: root.join("reports"),
            toolkit_version: ToolkitVersion::parse("0.13.3").unwrap(),
            reference_mode: "faulting_fixture".to_owned(),
            ordered_reference_overlays: vec!["faulting-reference".to_owned()],
            production_cache_installed: false,
            remote_publication_enabled: false,
        })
        .unwrap();
        let computed_resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(crate::ZipJsonFilesystemCacheStore::new(
                "validation-computed",
                root.join("computed"),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let reference_resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FaultingReferenceStore),
        }]);
        let semantic = semantic_key();
        let policy = policy();
        let request = ArtifactExecutionCacheRequest {
            operation: "quadrature.load_or_compute",
            semantic_key: &semantic,
            logical_key: "gauss-legendre/4/128",
            resolver: Some(&computed_resolver),
            reference_resolver: Some(&reference_resolver),
            acceptance: Some(&policy),
            ordered_overlays: vec![
                "validation-computed".to_owned(),
                "faulting-reference".to_owned(),
            ],
            mode: ArtifactExecutionCacheMode::VerifyAgainstReference,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse("0.13.3").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            tags: BTreeMap::new(),
            provenance_digest: None,
            production_sink: None,
        };
        let error = match resolve_or_compute_json_artifact(
            &request,
            || Ok(vec!["computed".to_owned()]),
            |_| Ok(()),
        ) {
            Ok(_) => panic!("operational reference failure unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(matches!(error, CacheError::Io(_)));
        assert!(crate::finalize_output_validation_run().is_err());
        let report: crate::OutputPreservationValidationReport =
            serde_json::from_slice(&fs::read(root.join("reports/latest.json")).unwrap()).unwrap();
        assert_eq!(report.totals.compared, 0);
        assert_eq!(report.totals.reference_absent, 0);
        let _ = fs::remove_dir_all(root);
    }
}
