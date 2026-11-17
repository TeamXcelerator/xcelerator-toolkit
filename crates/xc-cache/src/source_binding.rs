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
    execution_cache::validate_retained_canonical_binding(&canonical, &semantic, family, manifest)?;
    Ok(Some(canonical))
}

pub(crate) fn identity(
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
        return Ok(source.quality.satisfies(CacheQuality::Validated)
            && canonical
                .canonical_payload
                .dependencies
                .contains(&identity(&parent)?));
    }
    Ok(manifest.dependencies.iter().any(|d| {
        d.key == source.key
            && d.content_digest == source.content_digest
            && source.quality.satisfies(d.required_quality)
    }))
}

/// Check the complete direct parent set, rather than treating an adopted shard's
/// empty local dependency list as either a mismatch or permission to skip checks.
pub fn manifest_sources_match(
    manifest: &ArtifactManifest,
    sources: &[&ArtifactManifest],
) -> Result<bool, CacheError> {
    if let Some(canonical) = retained_canonical_manifest(manifest)? {
        let key = |d: &PayloadDependencyIdentity| {
            (
                d.artifact_family.clone(),
                d.semantic_digest.clone(),
                d.manifest_digest.clone(),
                d.payload_digest.clone(),
            )
        };
        let expected = canonical
            .canonical_payload
            .dependencies
            .iter()
            .map(&key)
            .collect::<std::collections::BTreeSet<_>>();
        let mut actual = std::collections::BTreeSet::new();
        for source in sources {
            let Some(parent) = retained_canonical_manifest(source)? else {
                return Ok(false);
            };
            if !source.quality.satisfies(CacheQuality::Validated) {
                return Ok(false);
            }
            actual.insert(key(&identity(&parent)?));
        }
        return Ok(actual == expected);
    }
    let mut unique = std::collections::BTreeSet::new();
    for source in sources {
        if !manifest_depends_on(manifest, source)? {
            return Ok(false);
        }
        unique.insert((
            source.key.kind.clone(),
            source.key.logical_key.clone(),
            source.key.parameters_digest.clone(),
            source.content_digest.clone(),
        ));
    }
    Ok(unique.len() == manifest.dependencies.len())
}

/// Admit an immutable validated source under the complete reader policy.
/// Without a caller policy, use this Toolkit's version and allow all storage
/// visibilities. Admission never grants publication or disclosure permission.
/// Errors include the artifact kind/digests and specific policy reasons.
pub fn admit_retained_source(
    manifest: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<(), CacheError> {
    let default;
    let policy = match policy {
        Some(policy) => policy,
        None => {
            default = CachePolicy {
                current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
                minimum_quality: CacheQuality::Validated,
                accepted_schema_versions: vec![1],
                allow_deprecated: false,
                allow_quarantined: false,
                allowed_visibilities: vec![
                    CacheVisibility::Local,
                    CacheVisibility::Public,
                    CacheVisibility::Private,
                    CacheVisibility::Team,
                ],
            };
            &default
        }
    };
    let mut decision = policy.assess(manifest);
    if !manifest.immutable {
        decision
            .reasons
            .push("retained source must be immutable".into());
    }
    if !manifest.quality.satisfies(CacheQuality::Validated) {
        decision
            .reasons
            .push("retained source must have validated evidence".into());
    }
    if !decision.reasons.is_empty() {
        return Err(CacheError::InvalidManifest(format!(
            "source {} (semantic {}, payload {}) rejected: {}",
            manifest.key.kind,
            manifest.key.parameters_digest,
            manifest.content_digest,
            decision.reasons.join("; ")
        )));
    }
    Ok(())
}

fn missing_source(message: String, rejected: Option<String>) -> CacheError {
    match rejected {
        Some(reason) => CacheError::InvalidManifest(format!(
            "{message}; first rejected source candidate: {reason}"
        )),
        None => CacheError::NotFound(message),
    }
}

enum SourceRequest<'a> {
    Published(&'a PayloadDependencyIdentity),
    Native(&'a DependencyRef),
}

// Source reads require immutable validated evidence in addition to the caller
// policy. Apply that complete contract while scanning resolver candidates too,
// so an ineligible first hit cannot hide an admissible source in a later layer.
fn lookup_retained_source(
    resolver: &CacheResolver,
    request: SourceRequest<'_>,
    policy: Option<&CachePolicy>,
    rejected: &mut Option<String>,
) -> Result<Option<ArtifactManifest>, CacheError> {
    for layer in &resolver.layers {
        let candidates = match request {
            SourceRequest::Published(id) => layer.store.identity_candidates(id)?,
            SourceRequest::Native(dep) => layer.store.candidates(&dep.key)?,
        };
        for candidate in candidates {
            if let SourceRequest::Native(dep) = request {
                if candidate.key != dep.key || candidate.content_digest != dep.content_digest {
                    continue;
                }
            }
            if let Err(error) = admit_retained_source(&candidate, policy) {
                rejected
                    .get_or_insert_with(|| format!("cache layer {}: {error}", layer.store.name()));
                continue;
            }
            match request {
                SourceRequest::Native(dep)
                    if !candidate.quality.satisfies(dep.required_quality) =>
                {
                    rejected.get_or_insert_with(|| {
                        format!(
                            "source {} fails dependency quality {:?}",
                            candidate.key.kind, dep.required_quality
                        )
                    });
                    continue;
                }
                SourceRequest::Published(id)
                    if !manifest_matches_dependency_identity(&candidate, id)? =>
                {
                    return Err(CacheError::InvalidManifest(format!("cache layer {} returned a candidate that does not match dependency identity {}/{}",layer.store.name(),id.artifact_family,id.semantic_digest)));
                }
                _ => {}
            }
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// Resolve exact direct parent metadata without reading numerical payloads.
/// Supplied manifests permit offline ancestry walks; missing or invalid metadata
/// remains an error and is never replaced by a same-configuration guess.
/// The full caller policy applies to supplied candidates and resolver results.
/// Inadmissible aliases do not mask an admissible exact source. When no caller
/// policy exists, supplied sources still pass current-reader admission.
pub fn resolve_manifest_sources(
    manifest: &ArtifactManifest,
    provided: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<Vec<ArtifactManifest>, CacheError> {
    let mut sources = Vec::new();
    if let Some(canonical) = retained_canonical_manifest(manifest)? {
        for dependency in &canonical.canonical_payload.dependencies {
            let mut found = None;
            let mut rejected = None;
            for candidate in provided {
                if candidate.key.parameters_digest != dependency.semantic_digest
                    || family_for_artifact_kind(&candidate.key.kind)
                        != Some(dependency.artifact_family.as_str())
                {
                    continue;
                }
                let admitted: Result<bool, CacheError> = (|| {
                    admit_retained_source(candidate, cache.acceptance)?;
                    match retained_canonical_manifest(candidate)? {
                        Some(parent) => Ok(identity(&parent)? == *dependency),
                        None => Ok(false),
                    }
                })();
                match admitted {
                    Ok(true) => {
                        found = Some(candidate.clone());
                        break;
                    }
                    Ok(false) => {}
                    Err(error) => {
                        rejected.get_or_insert_with(|| error.to_string());
                    }
                }
            }
            if found.is_none() {
                if let Some(resolver) = cache.resolver {
                    found = lookup_retained_source(
                        resolver,
                        SourceRequest::Published(dependency),
                        cache.acceptance,
                        &mut rejected,
                    )?;
                }
            }
            let source = found.ok_or_else(|| {
                missing_source(
                    format!(
                        "source metadata {}/{} (manifest {})",
                        dependency.artifact_family,
                        dependency.semantic_digest,
                        dependency.manifest_digest,
                    ),
                    rejected,
                )
            })?;
            admit_retained_source(&source, cache.acceptance)?;
            sources.push(source);
        }
    } else {
        for dependency in &manifest.dependencies {
            let minimum_quality = CacheQuality::Validated
                .combined_minimum(dependency.required_quality)
                .ok_or_else(|| {
                    CacheError::InvalidManifest(
                        "source validation and dependency quality requirements are incompatible"
                            .to_owned(),
                    )
                })?;
            let mut rejected = None;
            let mut found = None;
            for candidate in provided.iter().filter(|m| {
                m.key == dependency.key && m.content_digest == dependency.content_digest
            }) {
                let admitted = admit_retained_source(candidate, cache.acceptance).and_then(|()| {
                    if candidate.quality.satisfies(minimum_quality) {
                        Ok(())
                    } else {
                        Err(CacheError::InvalidManifest(format!(
                            "source {} fails dependency quality {:?}",
                            candidate.key.kind, minimum_quality
                        )))
                    }
                });
                match admitted {
                    Ok(()) => {
                        found = Some(candidate.clone());
                        break;
                    }
                    Err(error) => {
                        rejected.get_or_insert_with(|| error.to_string());
                    }
                }
            }
            if found.is_none() {
                if let Some(resolver) = cache.resolver {
                    found = lookup_retained_source(
                        resolver,
                        SourceRequest::Native(dependency),
                        cache.acceptance,
                        &mut rejected,
                    )?;
                }
            }
            let source = found.ok_or_else(|| {
                missing_source(
                    format!(
                        "source metadata {} / {}",
                        dependency.key.kind, dependency.key.logical_key,
                    ),
                    rejected,
                )
            })?;
            admit_retained_source(&source, cache.acceptance)?;
            if !source.quality.satisfies(dependency.required_quality)
                || !source.quality.satisfies(CacheQuality::Validated)
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
