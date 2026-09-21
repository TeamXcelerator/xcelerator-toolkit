//! Metadata-only source checks for local artifacts and published shard adapters.
use crate::*;

/// Read a published manifest only after binding it to the adapter's semantic key,
/// payload bytes, and recorded provenance. An empty adapter dependency list does
/// not mean a published artifact has no parents.
pub fn retained_canonical_manifest(
    manifest: &ArtifactManifest,
) -> Result<Option<CanonicalArtifactManifest>, CacheError> {
    manifest.validate()?;
    let Some(encoded) = manifest.tags.get(REMOTE_CANONICAL_MANIFEST_TAG) else {
        return Ok(None);
    };
    let canonical: CanonicalArtifactManifest = serde_json::from_str(encoded)?;
    let semantic: SemanticKeyEnvelope = serde_json::from_str(
        manifest
            .tags
            .get(SEMANTIC_KEY_MANIFEST_TAG)
            .ok_or_else(|| {
                CacheError::InvalidManifest("published source has no semantic envelope".into())
            })?,
    )?;
    if semantic.artifact_kind != manifest.key.kind
        || semantic.digest()? != manifest.key.parameters_digest
    {
        return Err(CacheError::InvalidManifest(
            "published source semantic key differs from its adapter".into(),
        ));
    }
    let family = family_for_artifact_kind(&manifest.key.kind).ok_or_else(|| {
        CacheError::InvalidManifest("published source kind has no artifact family".into())
    })?;
    execution_cache::validate_retained_canonical_binding(
        &canonical,
        &semantic,
        family,
        &manifest.content_digest,
        manifest.size_bytes,
        manifest.provenance_digest.as_ref(),
    )?;
    Ok(Some(canonical))
}

fn identity(
    canonical: &CanonicalArtifactManifest,
) -> Result<PayloadDependencyIdentity, CacheError> {
    Ok(PayloadDependencyIdentity {
        artifact_family: canonical.artifact_family.clone(),
        semantic_digest: canonical.semantic_digest.clone(),
        manifest_digest: canonical.digest()?,
        payload_digest: canonical.payload_digest.clone(),
    })
}

/// Verify a direct source edge, retaining exact published identities across
/// adapter logical-key aliases. Matching numerical bytes alone is insufficient.
pub fn manifest_depends_on(
    manifest: &ArtifactManifest,
    source: &ArtifactManifest,
) -> Result<bool, CacheError> {
    source.validate()?;
    if let Some(canonical) = retained_canonical_manifest(manifest)? {
        let Some(parent) = retained_canonical_manifest(source)? else {
            return Ok(false);
        };
        return Ok(
            source.quality.admissible_rank() >= CacheQuality::Validated.admissible_rank()
                && canonical
                    .canonical_payload
                    .dependencies
                    .contains(&identity(&parent)?),
        );
    }
    Ok(manifest.dependencies.iter().any(|d| {
        d.key == source.key
            && d.content_digest == source.content_digest
            && source.quality.admissible_rank() >= d.required_quality.admissible_rank()
    }))
}

/// Check the complete direct parent set, rather than treating an adopted shard's
/// empty local dependency list as either a mismatch or permission to skip checks.
pub fn manifest_sources_match(
    manifest: &ArtifactManifest,
    sources: &[&ArtifactManifest],
) -> Result<bool, CacheError> {
    let count = match retained_canonical_manifest(manifest)? {
        Some(canonical) => canonical.canonical_payload.dependencies.len(),
        None => manifest.dependencies.len(),
    };
    let mut unique = Vec::<&ArtifactManifest>::new();
    for source in sources {
        if !unique
            .iter()
            .any(|s| s.key == source.key && s.content_digest == source.content_digest)
        {
            unique.push(source);
        }
    }
    if count != unique.len() {
        return Ok(false);
    }
    for source in unique {
        if !manifest_depends_on(manifest, source)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Resolve exact direct parent metadata without reading numerical payloads.
/// Supplied manifests permit offline ancestry walks; missing or invalid metadata
/// remains an error and is never replaced by a same-configuration guess.
pub fn resolve_manifest_sources(
    manifest: &ArtifactManifest,
    provided: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<Vec<ArtifactManifest>, CacheError> {
    let mut sources = Vec::new();
    if let Some(canonical) = retained_canonical_manifest(manifest)? {
        for dependency in &canonical.canonical_payload.dependencies {
            let mut found = None;
            for candidate in provided {
                if let Some(parent) = retained_canonical_manifest(candidate)? {
                    if identity(&parent)? == *dependency {
                        found = Some(candidate.clone());
                        break;
                    }
                }
            }
            if found.is_none() {
                if let (Some(resolver), Some(policy)) = (cache.resolver, cache.acceptance) {
                    found = resolver
                        .resolve_dependency_identity_manifest(dependency, policy)?
                        .map(|(_, m)| m);
                }
            }
            let source = found.ok_or_else(|| {
                CacheError::NotFound(format!(
                    "source metadata {}/{} (manifest {})",
                    dependency.artifact_family,
                    dependency.semantic_digest,
                    dependency.manifest_digest,
                ))
            })?;
            if source.quality.admissible_rank() < CacheQuality::Validated.admissible_rank() {
                return Err(CacheError::InvalidManifest(
                    "inadmissible published source".into(),
                ));
            }
            sources.push(source);
        }
    } else {
        for dependency in &manifest.dependencies {
            let mut found = provided
                .iter()
                .find(|m| m.key == dependency.key && m.content_digest == dependency.content_digest)
                .cloned();
            if found.is_none() {
                if let (Some(resolver), Some(policy)) = (cache.resolver, cache.acceptance) {
                    found = Some(
                        resolver
                            .resolve_exact_manifest(
                                &dependency.key,
                                &dependency.content_digest,
                                policy,
                            )?
                            .1,
                    );
                }
            }
            let source = found.ok_or_else(|| {
                CacheError::NotFound(format!(
                    "source metadata {} / {}",
                    dependency.key.kind, dependency.key.logical_key,
                ))
            })?;
            source.validate()?;
            if source.quality.admissible_rank()
                < dependency
                    .required_quality
                    .admissible_rank()
                    .max(CacheQuality::Validated.admissible_rank())
            {
                return Err(CacheError::InvalidManifest(
                    "inadmissible local source".into(),
                ));
            }
            sources.push(source);
        }
    }
    Ok(sources)
}
