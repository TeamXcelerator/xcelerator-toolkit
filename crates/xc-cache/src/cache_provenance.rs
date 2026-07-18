//! Stable conversion of cache decisions into run provenance.

use crate::{
    CacheError, RemoteArtifactClosureMaterializationReport, RemoteResolverOverlay,
    SemanticResolutionReport,
};
use std::collections::BTreeMap;
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
    pub materialization: Option<&'a RemoteArtifactClosureMaterializationReport>,
}

pub fn record_remote_cache_access(
    request: RemoteCacheAccessProvenanceRequest<'_>,
) -> Result<CacheAccessProvenance, CacheError> {
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
    let validated_artifacts = request
        .materialization
        .into_iter()
        .flat_map(|report| &report.artifacts_dependency_first)
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
