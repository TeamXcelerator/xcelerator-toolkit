//! Backend-neutral semantic artifact resolution.

use crate::{
    resolve_cache_bundle_semantic_artifact, resolve_remote_semantic_artifact,
    ArtifactAssuranceState, ArtifactDisposition, CacheBundleConsumptionPolicy, CacheBundlePolicy,
    CacheBundleSemanticResolutionRequest, CacheError, CanonicalArtifactManifest, ContentDigest,
    RemoteGitStore, RemoteResolverOverlay, RemoteSemanticQuery,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticArtifactSourceKind {
    LocalFilesystem,
    ExportBundle,
    GitHubPrivate,
    GitHubPublic,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticArtifactOverlayClass {
    ProcessLocal,
    WorkstationLocal,
    ProjectPrivate,
    TeamPrivate,
    PublicPublished,
    OptionalRemote,
}

pub enum SemanticArtifactSource<'a> {
    LocalFilesystem {
        name: &'a str,
        overlay_class: SemanticArtifactOverlayClass,
        root: &'a Path,
        scratch_root: &'a Path,
        policy: &'a CacheBundlePolicy,
    },
    ExportBundle {
        name: &'a str,
        overlay_class: SemanticArtifactOverlayClass,
        root: &'a Path,
        scratch_root: &'a Path,
        policy: &'a CacheBundlePolicy,
    },
    GitHub {
        overlay_class: SemanticArtifactOverlayClass,
        overlay: &'a RemoteResolverOverlay,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticArtifactSelection {
    pub source_name: String,
    pub overlay_class: SemanticArtifactOverlayClass,
    pub source_kind: SemanticArtifactSourceKind,
    pub source_locator: String,
    pub immutable_revision: Option<String>,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub manifest: CanonicalArtifactManifest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticArtifactSourceRejection {
    pub source_name: String,
    pub overlay_class: SemanticArtifactOverlayClass,
    pub source_kind: SemanticArtifactSourceKind,
    pub stage: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnifiedSemanticArtifactResolutionReport {
    pub schema_version: u32,
    pub family: String,
    pub semantic_digest: ContentDigest,
    pub ordered_sources: Vec<String>,
    pub ordered_overlay_classes: Vec<SemanticArtifactOverlayClass>,
    pub selected: Option<SemanticArtifactSelection>,
    pub rejections: Vec<SemanticArtifactSourceRejection>,
}

pub fn resolve_semantic_artifact(
    remote: Option<&dyn RemoteGitStore>,
    query: &RemoteSemanticQuery,
    sources: &[SemanticArtifactSource<'_>],
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<UnifiedSemanticArtifactResolutionReport, CacheError> {
    let semantic_digest = query.validate()?;
    if sources.is_empty() {
        return Err(CacheError::InvalidManifest(
            "semantic resolution requires at least one backend source".to_owned(),
        ));
    }
    let mut names = BTreeSet::new();
    let source_descriptors = sources
        .iter()
        .map(source_name_kind_class)
        .map(|(name, kind, overlay_class)| {
            if name.trim().is_empty() || !names.insert(name) {
                Err(CacheError::InvalidManifest(
                    "semantic backend source names must be nonempty and unique".to_owned(),
                ))
            } else {
                Ok((name.to_owned(), kind, overlay_class))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if source_descriptors
        .windows(2)
        .any(|pair| pair[0].2 > pair[1].2)
    {
        return Err(CacheError::InvalidManifest(
            "semantic overlays must follow process, workstation, project-private, team-private, public-published, optional-remote order".to_owned(),
        ));
    }
    let ordered_sources = source_descriptors
        .iter()
        .map(|(name, _, _)| name.clone())
        .collect::<Vec<_>>();
    let ordered_overlay_classes = source_descriptors
        .iter()
        .map(|(_, _, overlay_class)| *overlay_class)
        .collect::<Vec<_>>();
    let mut rejections = Vec::new();
    for source in sources {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let (name, kind, overlay_class) = source_name_kind_class(source);
        let selected = match source {
            SemanticArtifactSource::GitHub { overlay, .. } => {
                let remote = remote.ok_or_else(|| {
                    CacheError::InvalidManifest(
                        "GitHub semantic source requires a remote transport".to_owned(),
                    )
                })?;
                let report = resolve_remote_semantic_artifact(
                    remote,
                    cancellation,
                    query,
                    std::slice::from_ref(overlay),
                )?;
                rejections.extend(report.rejections.into_iter().map(|rejection| {
                    SemanticArtifactSourceRejection {
                        source_name: name.to_owned(),
                        overlay_class,
                        source_kind: kind,
                        stage: format!("{:?}", rejection.stage).to_ascii_lowercase(),
                        reason: rejection.reason,
                    }
                }));
                if let Some(artifact) = report.selected {
                    let manifest_digest = artifact.manifest.digest()?;
                    Some(SemanticArtifactSelection {
                        source_name: name.to_owned(),
                        overlay_class,
                        source_kind: kind,
                        source_locator: artifact.repository,
                        immutable_revision: Some(artifact.revision),
                        semantic_digest: artifact.semantic_digest,
                        manifest_digest,
                        achieved_assurance: artifact.index.achieved_assurance,
                        disposition: artifact.index.disposition,
                        manifest: artifact.manifest,
                    })
                } else {
                    None
                }
            }
            SemanticArtifactSource::LocalFilesystem {
                root,
                scratch_root,
                policy,
                ..
            }
            | SemanticArtifactSource::ExportBundle {
                root,
                scratch_root,
                policy,
                ..
            } => {
                let consumption = CacheBundleConsumptionPolicy {
                    schema_version: 1,
                    reader_version: query.current_toolkit_version.clone(),
                    minimum_assurance: query.minimum_assurance,
                    allow_deprecated: query.allow_deprecated,
                };
                match resolve_cache_bundle_semantic_artifact(CacheBundleSemanticResolutionRequest {
                    bundle_root: root,
                    scratch_root,
                    family: &query.family,
                    semantic_digest: &semantic_digest,
                    bundle_policy: policy,
                    consumption_policy: &consumption,
                    resources,
                    cancellation,
                }) {
                    Ok(report) => {
                        rejections.extend(report.rejected_manifest_digests.into_iter().map(
                            |(digest, reason)| SemanticArtifactSourceRejection {
                                source_name: name.to_owned(),
                                overlay_class,
                                source_kind: kind,
                                stage: "policy".to_owned(),
                                reason: format!("manifest {}: {reason}", digest.0),
                            },
                        ));
                        report.selected.map(|artifact| SemanticArtifactSelection {
                            source_name: name.to_owned(),
                            overlay_class,
                            source_kind: kind,
                            source_locator: path_locator(root),
                            immutable_revision: Some(report.verification.bundle_digest.0),
                            semantic_digest: artifact.identity.semantic_digest,
                            manifest_digest: artifact.identity.manifest_digest,
                            achieved_assurance: artifact.achieved_assurance,
                            disposition: artifact.disposition,
                            manifest: artifact.manifest,
                        })
                    }
                    Err(CacheError::Cancelled(message)) => {
                        return Err(CacheError::Cancelled(message));
                    }
                    Err(error) => {
                        rejections.push(SemanticArtifactSourceRejection {
                            source_name: name.to_owned(),
                            overlay_class,
                            source_kind: kind,
                            stage: "source_validation".to_owned(),
                            reason: error.to_string(),
                        });
                        None
                    }
                }
            }
        };
        if let Some(selected) = selected {
            return Ok(UnifiedSemanticArtifactResolutionReport {
                schema_version: 1,
                family: query.family.clone(),
                semantic_digest,
                ordered_sources: ordered_sources.clone(),
                ordered_overlay_classes: ordered_overlay_classes.clone(),
                selected: Some(selected),
                rejections,
            });
        }
    }
    Ok(UnifiedSemanticArtifactResolutionReport {
        schema_version: 1,
        family: query.family.clone(),
        semantic_digest,
        ordered_sources,
        ordered_overlay_classes,
        selected: None,
        rejections,
    })
}

fn source_name_kind_class<'a>(
    source: &'a SemanticArtifactSource<'a>,
) -> (
    &'a str,
    SemanticArtifactSourceKind,
    SemanticArtifactOverlayClass,
) {
    match source {
        SemanticArtifactSource::LocalFilesystem {
            name,
            overlay_class,
            ..
        } => (
            name,
            SemanticArtifactSourceKind::LocalFilesystem,
            *overlay_class,
        ),
        SemanticArtifactSource::ExportBundle {
            name,
            overlay_class,
            ..
        } => (
            name,
            SemanticArtifactSourceKind::ExportBundle,
            *overlay_class,
        ),
        SemanticArtifactSource::GitHub {
            overlay_class,
            overlay,
        } => (
            &overlay.name,
            match overlay.visibility {
                crate::CacheVisibility::Private => SemanticArtifactSourceKind::GitHubPrivate,
                crate::CacheVisibility::Public => SemanticArtifactSourceKind::GitHubPublic,
                _ => SemanticArtifactSourceKind::GitHubPrivate,
            },
            *overlay_class,
        ),
    }
}

fn path_locator(path: &Path) -> String {
    PathBuf::from(path).to_string_lossy().into_owned()
}
