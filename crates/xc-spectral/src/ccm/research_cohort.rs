//! Bounded metadata discovery followed by exact-source reads. No primary solve.
use super::{
    convergence_capture::ComparisonState, research_completion::ComparisonSnapshot,
    retained_evidence::*, state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use xc_cache::*;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortRegistration {
    pub schema_version: u32,
    pub eigenpair: ArtifactManifest,
    pub matrix: ArtifactManifest,
    pub root: Option<ArtifactManifest>,
    pub secular: Option<ArtifactManifest>,
    pub assembly_policy: String,
    pub quadrature_policy: String,
}
fn directory() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("XC_RESEARCH_COHORT_DIR") {
        return Ok(p.into());
    }
    Ok(ManagedArtifactCacheConfig::from_environment()?
        .ok_or_else(|| anyhow::anyhow!("managed cache root unavailable"))?
        .cache_root
        .join("research-cohorts"))
}
pub fn register(record: &CohortRegistration) -> Result<()> {
    if record.schema_version != 1
        || record.assembly_policy.is_empty()
        || record.quadrature_policy.is_empty()
    {
        bail!("cohort policies absent");
    }
    record.eigenpair.validate()?;
    record.matrix.validate()?;
    let root = directory()?;
    std::fs::create_dir_all(&root)?;
    let bytes = serde_json::to_vec(record)?;
    let path = root.join(format!("{}.json", ContentDigest::sha256(&bytes).0));
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
    {
        Ok(mut f) => {
            use std::io::Write;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    };
    Ok(())
}
fn metadata(root: &Path, limit: usize) -> Result<Vec<ArtifactManifest>> {
    let mut pending = vec![root.to_path_buf()];
    let mut manifests = Vec::new();
    let mut visited = 0;
    while let Some(path) = pending.pop() {
        if !path.exists() {
            continue;
        }
        let mut entries = std::fs::read_dir(&path)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            visited += 1;
            if visited > 100000 {
                bail!("cohort metadata traversal limit exceeded");
            }
            let t = entry.file_type()?;
            if t.is_symlink() {
                continue;
            }
            if t.is_dir() {
                pending.push(entry.path());
            } else if entry.path().extension().is_some_and(|s| s == "json")
                && entry.metadata()?.len() < 1024 * 1024
            {
                if let Ok(m) = serde_json::from_reader::<_, ArtifactManifest>(
                    std::io::BufReader::new(std::fs::File::open(entry.path())?),
                ) {
                    if m.key.kind == "ccm_weil_eigenpair" {
                        manifests.push(m);
                        if manifests.len() > limit {
                            bail!("cohort candidate limit exceeded; supply a bounded XC_RESEARCH_COHORT_DIR");
                        }
                    }
                }
            }
        }
    }
    Ok(manifests)
}

// Resolve only the admitted matrix/factor/sector path. Published adapters keep
// exact canonical dependencies in metadata rather than the local dependency list.
fn matrix_source(
    state: &ArtifactManifest,
    provided: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<(ArtifactManifest, Vec<ArtifactManifest>)> {
    let mut pending = vec![(state.clone(), Vec::new())];
    let mut visited = std::collections::HashSet::new();
    while let Some((manifest, path)) = pending.pop() {
        if path.len() > 16 || visited.len() > 64 {
            bail!("cohort matrix ancestry exceeds metadata budget");
        }
        for parent in resolve_manifest_sources(&manifest, provided, cache)? {
            if parent.key.kind == "ccm_tau_matrix" {
                return Ok((parent, path));
            }
            if [
                "ccm_factorization",
                "ccm_even_sector_matrix",
                "ccm_odd_sector_matrix",
            ]
            .contains(&parent.key.kind.as_str())
                && visited.insert((parent.key.clone(), parent.content_digest.clone()))
            {
                let mut next_path = path.clone();
                next_path.push(parent.clone());
                pending.push((parent, next_path));
            }
        }
    }
    bail!("cohort eigenstate has no authenticated retained matrix ancestor")
}

fn same_cutoff(manifest: &ArtifactManifest, cutoff: &str) -> Result<bool> {
    if let Some(canonical) = retained_canonical_manifest(manifest)? {
        return Ok(
            canonical.semantic_key.resolved_mathematical_parameters["lambda_squared"].as_str()
                == Some(cutoff),
        );
    }
    let parts = manifest.key.logical_key.split('/').collect::<Vec<_>>();
    Ok(parts.len() >= 6 && parts[2] == cutoff)
}
pub(crate) fn discover(
    s: &RetainedState,
    cache: &ArtifactCacheContext<'_>,
) -> Result<(Vec<ComparisonSnapshot>, Vec<ArtifactManifest>, Vec<String>)> {
    let (Some(resolver), Some(policy)) = (cache.resolver, cache.acceptance) else {
        return Ok((
            vec![],
            vec![],
            vec!["cohort discovery requires an exact-source cache resolver".into()],
        ));
    };
    let mut registrations = Vec::new();
    let mut notes = Vec::new();
    let dir = directory()?;
    if dir.exists() {
        let mut entries = std::fs::read_dir(&dir)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            if registrations.len() >= 512 {
                notes.push("cohort discovery selected the first 512 registrations in digest filename order; additional files remain on disk. Supply a bounded cohort directory for a different selection".into());
                break;
            }
            if entry.file_type()?.is_file()
                && entry.path().extension().is_some_and(|s| s == "json")
                && entry.metadata()?.len() < 4 * 1024 * 1024
            {
                registrations.push(serde_json::from_reader::<_, CohortRegistration>(
                    std::io::BufReader::new(std::fs::File::open(entry.path())?),
                )?);
            }
        }
    }
    // Existing cache states need no new registration or primary computation.
    if std::env::var_os("XC_RESEARCH_COHORT_DIR").is_none() {
        if let Some(config) = ManagedArtifactCacheConfig::from_environment()? {
            for manifest in metadata(
                &config.cache_root.join("artifacts/ccm_weil_eigenpair"),
                4096,
            )? {
                if manifest.content_digest == s.manifest.content_digest
                    || registrations
                        .iter()
                        .any(|c| c.eigenpair.content_digest == manifest.content_digest)
                {
                    continue;
                }
                // Filter by declared logical coordinates before decoding a large state.
                if !same_cutoff(&manifest, &s.cutoff)? {
                    continue;
                }
                match matrix_source(&manifest, &[], cache) {
                    Ok((matrix, _)) => {
                        registrations.push(CohortRegistration{schema_version:1,eigenpair:manifest,matrix,root:None,secular:None,assembly_policy:"historical assembly policy not retained in cohort index; exact matrix identity recorded".into(),quadrature_policy:"unassessed historical quadrature policy".into()});
                    }
                    Err(e) => notes.push(format!(
                        "historical comparison {} rejected: {e}",
                        manifest.content_digest.0
                    )),
                }
            }
        }
    }
    registrations.sort_by(|a, b| a.eigenpair.content_digest.cmp(&b.eigenpair.content_digest));
    registrations.dedup_by(|a, b| a.eigenpair.content_digest == b.eigenpair.content_digest);
    let mut comparisons = Vec::new();
    let mut parents = Vec::new();
    for entry in registrations {
        if entry.eigenpair.content_digest == s.manifest.content_digest {
            continue;
        }
        if comparisons.len() >= 64 {
            notes.push("comparison source limit reached; remaining candidates not decoded".into());
            break;
        }
        let result = (|| -> Result<_> {
            if entry.eigenpair.size_bytes > 64 * 1024 * 1024 {
                bail!("comparison eigenpair exceeds 64 MiB");
            }
            let state = resolver.resolve_exact(
                &entry.eigenpair.key,
                &entry.eigenpair.content_digest,
                CacheQuality::Validated,
                policy,
            )?;
            let state = RetainedState::from_payload(
                &state.manifest,
                &state.payload,
                std::slice::from_ref(&state.manifest.content_digest),
            )?;
            // A direct matrix ancestor or an authenticated factor/sector chain.
            let (matrix, ancestry) =
                matrix_source(&state.manifest, std::slice::from_ref(&entry.matrix), cache)?;
            if matrix.key != entry.matrix.key
                || matrix.content_digest != entry.matrix.content_digest
                || retained_canonical_manifest(&matrix)?
                    .map(|m| m.digest())
                    .transpose()?
                    != retained_canonical_manifest(&entry.matrix)?
                        .map(|m| m.digest())
                        .transpose()?
            {
                bail!("comparison matrix is not the recorded eigenstate ancestor");
            }
            let (branch, points) =
                if let (Some(root), Some(secular)) = (&entry.root, &entry.secular) {
                    let rb = resolver.resolve_exact(
                        &root.key,
                        &root.content_digest,
                        CacheQuality::Validated,
                        policy,
                    )?;
                    let sb = resolver.resolve_exact(
                        &secular.key,
                        &secular.content_digest,
                        CacheQuality::Validated,
                        policy,
                    )?;
                    let roots = RetainedRoots::from_payload(
                        root,
                        &rb.payload,
                        secular,
                        &sb.payload,
                        &state,
                        &[root.content_digest.clone(), secular.content_digest.clone()],
                    )?;
                    (
                        serde_json::to_string(&roots.acquisition)?,
                        roots.dataset.points,
                    )
                } else {
                    ("unassessed".into(), vec![])
                };
            let snapshot = ComparisonSnapshot {
                state: ComparisonState {
                    source_digest: state.manifest.content_digest.clone(),
                    matrix_digest: entry.matrix.content_digest.clone(),
                    lambda_squared: state.cutoff,
                    n_modes: state.modes,
                    precision_bits: state.precision,
                    coefficients: state
                        .coefficients
                        .iter()
                        .map(xc_numerics::prefix::lossless_decimal)
                        .collect(),
                    matrix: vec![],
                    eigenvalue: state.eigenvalue,
                    assembly_policy: entry.assembly_policy.clone(),
                },
                selection_policy: state
                    .selection_policy
                    .unwrap_or_else(|| "unassessed".into()),
                assembly_policy: entry.assembly_policy,
                quadrature_policy: entry.quadrature_policy,
                root_branch: branch,
                root_coordinate: "mellin_t".into(),
                roots: points,
            };
            let mut sources = ancestry;
            sources.extend([state.manifest, matrix]);
            sources.extend(entry.root);
            sources.extend(entry.secular);
            Ok((snapshot, sources))
        })();
        match result {
            Ok((c, m)) => {
                comparisons.push(c);
                parents.extend(m);
            }
            Err(e) => notes.push(format!("comparison rejected: {e}")),
        }
    }
    Ok((comparisons, parents, notes))
}

#[cfg(test)]
mod tests {
    use super::*;
    mod published_sources {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/published_sources.rs"
        ));
    }
    fn source(kind: &str) -> ArtifactManifest {
        serde_json::from_value(serde_json::json!({
            "schema_version":1,"key":ArtifactKey::new(kind,kind,kind.as_bytes()).unwrap(),
            "content_digest":ContentDigest::sha256(kind.as_bytes()),"size_bytes":kind.len(),
            "objects":[{"content_digest":ContentDigest::sha256(kind.as_bytes()),"size_bytes":kind.len()}],"created_unix_seconds":1,
            "producer_toolkit_version":ToolkitVersion::parse("0.15.1").unwrap(),
            "minimum_reader_version":ToolkitVersion::parse("0.15.1").unwrap(),
            "maximum_reader_version":null,"quality":"validated","visibility":"private",
            "immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
        })).unwrap()
    }
    #[test]
    fn cohort_follows_canonical_factor_chain_and_rejects_substituted_matrix() {
        use published_sources::published;
        let matrix = published(source("ccm_tau_matrix"), &[]);
        let sector = published(source("ccm_even_sector_matrix"), &[&matrix]);
        let factor = published(source("ccm_factorization"), &[&sector]);
        let state = published(source("ccm_weil_eigenpair"), &[&factor]);
        let cache = ArtifactCacheContext {
            resolver: None,
            reference_resolver: None,
            acceptance: None,
            ordered_overlays: vec![],
            mode: ArtifactExecutionCacheMode::Disabled,
            write_on_miss: false,
            write_visibility: CacheVisibility::Private,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let provided = vec![matrix.clone(), sector.clone(), factor.clone()];
        let (found, chain) = matrix_source(&state, &provided, &cache).unwrap();
        assert_eq!(found.content_digest, matrix.content_digest);
        assert_eq!(chain.len(), 2);
        let mut other = source("ccm_tau_matrix");
        other.key.logical_key = "other assembly".into();
        let other = published(other, &[]);
        assert!(matrix_source(&state, &[other, sector, factor], &cache).is_err());
    }

    #[test]
    fn published_cohort_filter_uses_semantic_coordinates_not_adapter_aliases() {
        let mut state = source("ccm_weil_eigenpair");
        let semantic = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: state.key.kind.clone(),
            mathematical_semantics_version: "fixture".into(),
            resolved_mathematical_parameters: serde_json::json!({"lambda_squared":"13"}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: Default::default(),
            algorithm_semantics: None,
        };
        state.tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.into(),
            serde_json::to_string(&semantic).unwrap(),
        );
        let mut state = published_sources::published(state, &[]);
        state.key.logical_key = "local-shard/weil-states/opaque".into();
        assert!(same_cutoff(&state, "13").unwrap());
        assert!(!same_cutoff(&state, "100").unwrap());
    }
}
