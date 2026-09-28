//! Stable conversion of cache decisions into run provenance.

use crate::{
    CacheError, RemoteArtifactClosureMaterializationReport, RemoteArtifactMaterializationReport,
    RemoteResolverOverlay, SemanticResolutionReport,
};
use std::collections::{BTreeMap, BTreeSet};
use xc_core::{
    CacheAccessProvenance, CacheCandidateRejectionProvenance, CacheLookupOutcome,
    CacheReuseDisposition, CacheSourceProvenance, CacheValidatedArtifactProvenance,
    CacheValidationMode, CacheValidationOutcome,
};

pub struct RemoteCacheAccessProvenanceRequest<'a> {
    pub operation: &'a str,
    pub family: &'a str,
    pub overlays: &'a [RemoteResolverOverlay],
    pub resolution: &'a SemanticResolutionReport,
    pub reuse_disposition: CacheReuseDisposition,
    pub validation_mode: CacheValidationMode,
    pub validation_outcome: CacheValidationOutcome,
    pub validation_detail: Option<String>,
    pub root_materialization: Option<&'a RemoteArtifactMaterializationReport>,
    pub materialization: Option<&'a RemoteArtifactClosureMaterializationReport>,
}

pub fn record_remote_cache_access(
    request: RemoteCacheAccessProvenanceRequest<'_>,
) -> Result<CacheAccessProvenance, CacheError> {
    if request.validation_outcome == CacheValidationOutcome::Passed
        && matches!(
            request.validation_mode,
            CacheValidationMode::Root | CacheValidationMode::Full
        )
    {
        let selected = request.resolution.selected.as_ref().ok_or_else(|| {
            CacheError::InvalidManifest(
                "payload validation requires a selected artifact".to_owned(),
            )
        })?;
        let manifest_digest = selected.manifest.digest()?;
        let matches_root = |artifact: &RemoteArtifactMaterializationReport| {
            artifact.semantic_digest == selected.semantic_digest
                && artifact.manifest_digest == manifest_digest
                && artifact.canonical_payload_digest == selected.manifest.payload_digest
        };
        let mut expected_closure = BTreeSet::new();
        if request.validation_mode == CacheValidationMode::Full {
            let mut pending = vec![selected];
            while let Some(artifact) = pending.pop() {
                if expected_closure.insert((
                    artifact.semantic_digest.clone(),
                    artifact.manifest.digest()?,
                    artifact.manifest.payload_digest.clone(),
                )) {
                    pending.extend(&artifact.dependencies);
                }
            }
        }
        let valid_scope = match request.validation_mode {
            CacheValidationMode::Root => request.root_materialization.is_some_and(matches_root),
            CacheValidationMode::Full => request.materialization.is_some_and(|report| {
                report.dependency_closure_fully_validated
                    && report.artifacts_dependency_first.len() == expected_closure.len()
                    && report
                        .artifacts_dependency_first
                        .iter()
                        .map(|artifact| {
                            (
                                artifact.semantic_digest.clone(),
                                artifact.manifest_digest.clone(),
                                artifact.canonical_payload_digest.clone(),
                            )
                        })
                        .collect::<BTreeSet<_>>()
                        == expected_closure
                    && report.root_semantic_digest == selected.semantic_digest
                    && report.root_manifest_digest == manifest_digest
                    && report.dependency_count.checked_add(1)
                        == Some(report.artifacts_dependency_first.len())
                    && report
                        .artifacts_dependency_first
                        .iter()
                        .filter(|artifact| matches_root(artifact))
                        .count()
                        == 1
            }),
            _ => false,
        };
        if !valid_scope {
            return Err(CacheError::InvalidManifest(
                "payload validation scope requires a matching completed materialization report"
                    .to_owned(),
            ));
        }
    }
    let selected_source =
        request
            .resolution
            .selected
            .as_ref()
            .map(|artifact| CacheSourceProvenance {
                overlay: artifact.overlay.clone(),
                location_kind: match artifact.visibility {
                    crate::CacheVisibility::Private => "github_private_remote",
                    crate::CacheVisibility::Public => "github_public_remote",
                    _ => "remote",
                }
                .to_owned(),
                repository: artifact.repository.clone(),
                revision: artifact.revision.clone(),
                document_paths: BTreeMap::from([
                    (
                        "index".to_owned(),
                        artifact.index_source.repository_path.clone(),
                    ),
                    (
                        "manifest".to_owned(),
                        artifact.manifest_source.repository_path.clone(),
                    ),
                    (
                        "encoding".to_owned(),
                        artifact.encoding_source.repository_path.clone(),
                    ),
                    (
                        "receipt".to_owned(),
                        artifact.receipt_source.repository_path.clone(),
                    ),
                ]),
            });
    let rejected_candidates = request
        .resolution
        .rejections
        .iter()
        .map(|rejection| CacheCandidateRejectionProvenance {
            overlay: rejection.overlay.clone(),
            source: rejection
                .repository
                .clone()
                .or_else(|| rejection.shard_id.clone()),
            stage: format!("{:?}", rejection.stage).to_ascii_lowercase(),
            reason: rejection.reason.clone(),
        })
        .collect();
    let checked_artifacts: Vec<_> = if request.validation_outcome == CacheValidationOutcome::Passed
    {
        match request.validation_mode {
            CacheValidationMode::Full => request
                .materialization
                .into_iter()
                .flat_map(|report| &report.artifacts_dependency_first)
                .collect(),
            CacheValidationMode::Root => request.root_materialization.into_iter().collect(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };
    let validated_artifacts = checked_artifacts
        .into_iter()
        .map(|artifact| CacheValidatedArtifactProvenance {
            semantic_digest: artifact.semantic_digest.0.clone(),
            manifest_digest: artifact.manifest_digest.0.clone(),
        })
        .collect();
    let provenance = CacheAccessProvenance {
        schema_version: 1,
        operation: request.operation.to_owned(),
        artifact_family: request.family.to_owned(),
        semantic_digest: request.resolution.semantic_digest.0.clone(),
        semantic_key_schema_version: request.resolution.resolved_semantic_key.schema_version,
        resolved_semantic_key: serde_json::to_value(&request.resolution.resolved_semantic_key)?,
        selected_manifest_digest: request
            .resolution
            .selected
            .as_ref()
            .map(|artifact| artifact.index.manifest_digest.0.clone()),
        ordered_overlays: request
            .overlays
            .iter()
            .map(|overlay| overlay.name.clone())
            .collect(),
        lookup_outcome: if request.resolution.selected.is_some() {
            CacheLookupOutcome::Hit
        } else {
            CacheLookupOutcome::Miss
        },
        reuse_disposition: request.reuse_disposition,
        selected_source,
        rejected_candidates,
        validation_mode: request.validation_mode,
        validation_outcome: request.validation_outcome,
        validation_detail: request.validation_detail,
        validated_artifacts,
    };
    provenance
        .validate()
        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
    Ok(provenance)
}
