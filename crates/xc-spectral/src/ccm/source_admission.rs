//! Metadata admission for consumed CCM values. Publication ancestry remains a
//! separate exact-provenance contract in xc-cache.
#[cfg(test)]
#[path = "source_admission_tests.rs"]
mod tests;
use anyhow::{ensure, Result};
use serde_json::Value;
use xc_cache::*;

pub(crate) fn admit_manifest(m: &ArtifactManifest, policy: Option<&CachePolicy>) -> Result<()> {
    admit_retained_source(m, None)?;
    if let Some(policy) = policy {
        admit_retained_source(m, Some(policy))?;
    }
    Ok(())
}

pub(crate) fn semantic(
    m: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<Option<SemanticKeyEnvelope>> {
    if !m.tags.contains_key(SEMANTIC_KEY_MANIFEST_TAG)
        && !m.tags.contains_key(REMOTE_CANONICAL_MANIFEST_TAG)
    {
        admit_manifest(m, policy)?;
        return Ok(None);
    }
    let _reservation = xc_core::RunResources::current()
        .map(|r| {
            let bytes = m.tags.values().try_fold(65536u64, |n, value| {
                n.checked_add((value.len() as u64).saturating_mul(16))
                    .ok_or_else(|| anyhow::anyhow!("source metadata workspace overflow"))
            })?;
            r.try_reserve_working(bytes).map_err(anyhow::Error::from)
        })
        .transpose()?;
    admit_manifest(m, policy)?;
    let Some(raw) = m.tags.get(SEMANTIC_KEY_MANIFEST_TAG) else {
        return Ok(None);
    };
    let value: SemanticKeyEnvelope = serde_json::from_str(raw)?;
    ensure!(
        value.artifact_kind == m.key.kind && value.digest()? == m.key.parameters_digest,
        "retained source semantic envelope differs from its manifest"
    );
    Ok(Some(value))
}

enum TauIdentity {
    Numerical(NumericalInputIdentity),
    Legacy(DependencyRef),
}
impl TauIdentity {
    fn parse(value: &serde_json::Value) -> Result<Self> {
        if value.get("role").is_some() {
            let id: NumericalInputIdentity = serde_json::from_value(value.clone())?;
            ensure!(
                id.role == "tau"
                    && id.kind == "ccm_tau_matrix"
                    && id.meaning == "stored-tau-matrix-v1"
                    && id.content_digest.validate(),
                "invalid numerical Tau role, kind, meaning or digest"
            );
            Ok(Self::Numerical(id))
        } else {
            let id: DependencyRef = serde_json::from_value(value.clone())?;
            id.key.validate()?;
            ensure!(
                id.key.kind == "ccm_tau_matrix"
                    && id.content_digest.validate()
                    && id.required_quality == CacheQuality::Validated,
                "invalid legacy Tau identity"
            );
            Ok(Self::Legacy(id))
        }
    }
    fn digest(&self) -> &ContentDigest {
        match self {
            Self::Numerical(i) => &i.content_digest,
            Self::Legacy(i) => &i.content_digest,
        }
    }
}

pub(crate) fn declared_tau_digest(
    state: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<Option<ContentDigest>> {
    let Some(s) = semantic(state, policy)? else {
        return Ok(None);
    };
    s.resolved_mathematical_parameters
        .get("tau_source_identity")
        .map(|value| Ok(TauIdentity::parse(value)?.digest().clone()))
        .transpose()
}

fn same_configuration(a: &SemanticKeyEnvelope, b: &SemanticKeyEnvelope) -> Result<()> {
    for name in ["lambda_squared", "n_modes", "precision_bits"] {
        let value = &a.resolved_mathematical_parameters[name];
        ensure!(
            !value.is_null() && value == &b.resolved_mathematical_parameters[name],
            "retained source {name} differs from its declared numerical parent"
        );
    }
    Ok(())
}

/// Older sources can omit a shape declaration. When present, authenticate it
/// against the actual decoded coefficients or matrix header.
pub(crate) fn check_payload_shape(
    manifest: &ArtifactManifest,
    cutoff: &str,
    modes: usize,
    precision: u32,
) -> Result<()> {
    let Some(semantic) = semantic(manifest, None)? else {
        return Ok(());
    };
    let parameters = &semantic.resolved_mathematical_parameters;
    if let Some(value) = parameters.get("lambda_squared") {
        let declared = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid retained cutoff declaration"))?;
        ensure!(
            declared == cutoff
                || xc_core::DecimalLiteral::new(declared)?
                    .cmp_numeric(&xc_core::DecimalLiteral::new(cutoff)?)?
                    == std::cmp::Ordering::Equal,
            "retained payload and semantic lambda_squared disagree"
        );
    }
    for (field, observed) in [
        ("n_modes", modes as u64),
        ("precision_bits", precision.into()),
    ] {
        if let Some(value) = parameters.get(field) {
            ensure!(
                value.as_u64() == Some(observed),
                "retained payload and semantic {field} disagree"
            );
        }
    }
    Ok(())
}

/// False means no explicit numerical declaration: the caller must use its
/// historical exact-ancestry route. A present but invalid declaration is fatal.
pub(crate) fn declared_tau_matches(
    state: &ArtifactManifest,
    tau: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<bool> {
    let Some(s) = semantic(state, policy)? else {
        return Ok(false);
    };
    let Some(value) = s
        .resolved_mathematical_parameters
        .get("tau_source_identity")
    else {
        return Ok(false);
    };
    let id = TauIdentity::parse(value)?;
    let t = semantic(tau, policy)?
        .ok_or_else(|| anyhow::anyhow!("declared Tau lacks its semantic envelope"))?;
    ensure!(
        state.key.kind == "ccm_weil_eigenpair" && tau.key.kind == "ccm_tau_matrix",
        "invalid state/Tau source kinds"
    );
    same_configuration(&s, &t)?;
    ensure!(
        s.normalization.as_deref() == Some("sum_xi_equals_sqrt_log_lambda_squared")
            && t.normalization.as_deref() == Some("symmetric_row_major"),
        "retained state/Tau normalization differs"
    );
    match id {
        TauIdentity::Numerical(id) => ensure!(
            id == numerical_source_identity(tau, "tau", "stored-tau-matrix-v1"),
            "declared numerical Tau payload differs"
        ),
        TauIdentity::Legacy(id) => ensure!(
            id.content_digest == tau.content_digest
                && id.key.kind == tau.key.kind
                && id.key.parameters_digest == tau.key.parameters_digest,
            "declared legacy Tau identity differs"
        ),
    }
    Ok(true)
}

/// Authenticate a Tau edge while allowing local labels for the same semantic
/// identity and payload. Published edges use canonical payload digests.
pub(super) fn tau_edge_matches(parent: &ArtifactManifest, tau: &ArtifactManifest) -> Result<bool> {
    if parent.dependencies.iter().any(|edge| {
        edge.key.kind == tau.key.kind
            && edge.key.parameters_digest == tau.key.parameters_digest
            && edge.content_digest == tau.content_digest
    }) {
        return Ok(true);
    }
    let (Some(parent), Some(source)) = (
        retained_canonical_manifest(parent)?,
        retained_canonical_manifest(tau)?,
    ) else {
        return Ok(false);
    };
    Ok(parent.canonical_payload.dependencies.iter().any(|edge| {
        edge.artifact_family == source.artifact_family
            && edge.semantic_digest == source.semantic_digest
            && edge.payload_digest == source.payload_digest
    }))
}

pub(crate) fn secular_matches_state(
    secular: &ArtifactManifest,
    state: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<bool> {
    Ok(declared_secular_matches(secular, state, policy)? || manifest_depends_on(secular, state)?)
}

/// Secular payloads explicitly bind both the state bytes and its validation key.
/// Publication-manifest changes do not alter either consumed value.
pub(crate) fn declared_secular_matches(
    secular: &ArtifactManifest,
    state: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<bool> {
    let Some(s) = semantic(secular, policy)? else {
        return Ok(false);
    };
    let p = &s.resolved_mathematical_parameters;
    if p.get("eigenpair_content_digest").is_none() && p.get("eigenpair_semantic_digest").is_none() {
        return Ok(false);
    }
    let e = semantic(state, policy)?
        .ok_or_else(|| anyhow::anyhow!("declared state lacks its semantic envelope"))?;
    ensure!(
        secular.key.kind == "ccm_secular_source" && state.key.kind == "ccm_weil_eigenpair",
        "invalid secular/state source kinds"
    );
    same_configuration(&s, &e)?;
    ensure!(
        p["eigenpair_content_digest"] == state.content_digest.0
            && p["eigenpair_semantic_digest"] == state.key.parameters_digest.0,
        "retained secular declaration names another state"
    );
    Ok(true)
}

fn root_secular_digest(s: &SemanticKeyEnvelope) -> Result<Option<ContentDigest>> {
    let p = &s.resolved_mathematical_parameters;
    let Some(value) = p.get("secular_source_content_digest") else {
        return Ok(None);
    };
    let digest: ContentDigest = serde_json::from_value(value.clone())?;
    ensure!(
        digest.validate(),
        "invalid declared secular-source content digest"
    );
    let seeded = match s.artifact_kind.as_str() {
        "ccm_root_refinement" => true,
        "ccm_root_discovery_window" => false,
        _ => anyhow::bail!("invalid root source kind {}", s.artifact_kind),
    };
    ensure!(
        p["reference_seeds_used"].as_bool() == Some(seeded)
            && p["discovery_mode"]
                == if seeded {
                    "reference_seeded_refinement"
                } else {
                    "independent"
                },
        "retained root acquisition mode differs from its artifact kind"
    );
    Ok(Some(digest))
}

/// Match the exact secular bytes consumed by a current root producer. Missing
/// declarations retain historical exact ancestry; invalid declarations fail.
pub(crate) fn declared_root_secular_matches(
    root: &ArtifactManifest,
    secular: &ArtifactManifest,
    policy: Option<&CachePolicy>,
) -> Result<bool> {
    let Some(r) = semantic(root, policy)? else {
        return Ok(false);
    };
    let Some(digest) = root_secular_digest(&r)? else {
        return Ok(false);
    };
    let s = semantic(secular, policy)?
        .ok_or_else(|| anyhow::anyhow!("declared secular source lacks its semantic envelope"))?;
    ensure!(
        secular.key.kind == "ccm_secular_source",
        "invalid secular source kind"
    );
    ensure!(
        digest == secular.content_digest,
        "declared numerical secular payload differs"
    );
    same_configuration(&r, &s)?;
    for field in ["force_even", "parity_policy"] {
        if let Some(value) = s.resolved_mathematical_parameters.get(field) {
            ensure!(
                r.resolved_mathematical_parameters.get(field) == Some(value),
                "retained root/secular {field} differs"
            );
        }
    }
    ensure!(
        r.subspace == s.subspace,
        "retained root/secular subspace differs"
    );
    ensure!(
        s.normalization.as_deref() == Some("sum_xi_equals_sqrt_log_lambda_squared"),
        "unsupported secular normalization"
    );
    Ok(true)
}

pub(crate) fn authenticate_root_policy(
    root: &ArtifactManifest,
    semantic: &SemanticKeyEnvelope,
    raw: &Value,
) -> Result<()> {
    let parameters = &semantic.resolved_mathematical_parameters;
    for field in [
        "force_even",
        "parity_policy",
        "first_root_index",
        "discovery_mode",
        "reference_seeds_used",
        "completeness",
        "secular_source_content_digest",
        "starting_points",
        "solver",
        "solver_steps",
        "accuracy_guard_bits",
        "reference_dataset",
    ] {
        if let Some(declared) = parameters.get(field) {
            ensure!(
                raw.get(field) == Some(declared),
                "root payload and semantic {field} disagree"
            );
        }
    }
    let seeded = root.key.kind == "ccm_root_refinement";
    ensure!(
        matches!(
            root.key.kind.as_str(),
            "ccm_root_refinement" | "ccm_root_discovery_window"
        ) && parameters["discovery_mode"]
            == if seeded {
                "reference_seeded_refinement"
            } else {
                "independent"
            }
            && parameters["reference_seeds_used"] == seeded,
        "root artifact kind and acquisition policy disagree"
    );
    let schema = raw["schema_version"].as_u64();
    // The producer's semantic policy spelling differs deliberately from the
    // serde enum spelling in schemas 5/7. Schemas 6/8 fixed-guard payloads omit all
    // adaptive fields. Do not accept arbitrary aliases or infer a policy.
    match (
        schema,
        parameters
            .get("root_precision_policy")
            .and_then(Value::as_str),
    ) {
        (Some(5 | 7), Some("adaptive_v1")) => {
            ensure!(
                schema != Some(7) || seeded,
                "schema-7 roots require seeded acquisition"
            );
            ensure!(
                raw["root_precision_policy"] == "adaptive",
                "adaptive root payload policy mismatch"
            );
            for field in [
                "target_precision_bits",
                "maximum_extra_precision_bits",
                "verification_precision_bits",
            ] {
                ensure!(
                    parameters[field].as_u64().is_some() && raw[field] == parameters[field],
                    "adaptive root payload and semantic {field} disagree"
                );
            }
        }
        (Some(3 | 6 | 8), None) => {
            ensure!(
                schema != Some(8) || seeded,
                "schema-8 roots require seeded acquisition"
            );
            for field in [
                "root_precision_policy",
                "target_precision_bits",
                "maximum_extra_precision_bits",
                "verification_precision_bits",
            ] {
                ensure!(
                    raw.get(field).is_none_or(Value::is_null)
                        && parameters.get(field).is_none_or(Value::is_null),
                    "fixed-guard root payload has an unbound adaptive {field}"
                );
            }
        }
        _ => anyhow::bail!("unsupported or inconsistent root schema/precision policy"),
    }
    let domain = |v: &Value| -> Result<String> {
        match v.get("root_domain") {
            None => Ok("positive".into()),
            Some(Value::String(s)) => Ok(s.clone()),
            _ => anyhow::bail!("invalid root domain representation"),
        }
    };
    let declared_domain = domain(parameters)?;
    let stored_domain = domain(raw)?;
    ensure!(
        matches!(declared_domain.as_str(), "positive" | "signed")
            && declared_domain == stored_domain,
        "root domain binding mismatch"
    );
    if let Some(count) = parameters.get("root_count") {
        ensure!(
            count.as_u64().is_some()
                && count.as_u64() == raw["outcomes"].as_array().map(|r| r.len() as u64),
            "root count binding mismatch"
        );
    }
    if parameters.get("reference_dataset").is_none() {
        ensure!(
            raw.get("reference_dataset").is_none_or(Value::is_null),
            "root reference dataset absent from semantic policy"
        );
    }
    Ok(())
}

pub(crate) fn resolve_root_secular(
    root: &ArtifactManifest,
    state: &ArtifactManifest,
    provided: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactManifest> {
    let check = |source: &ArtifactManifest| -> Result<bool> {
        let declared = declared_root_secular_matches(root, source, cache.acceptance)?;
        ensure!(
            secular_matches_state(source, state, cache.acceptance)?,
            "roots belong to another state"
        );
        Ok(declared)
    };
    let digest = semantic(root, cache.acceptance)?
        .as_ref()
        .map(root_secular_digest)
        .transpose()?
        .flatten();
    let mut rejected = None;
    if let Some(digest) = digest {
        for source in provided
            .iter()
            .filter(|m| m.key.kind == "ccm_secular_source" && m.content_digest == digest)
        {
            match check(source) {
                Ok(true) => {
                    if let (Some(resolver), Some(policy)) = (cache.resolver, cache.acceptance) {
                        match resolver.resolve_exact_manifest(
                            &source.key,
                            &source.content_digest,
                            policy,
                        ) {
                            Ok(_) => return Ok(source.clone()),
                            Err(xc_cache::CacheError::NotFound(_)) => continue,
                            Err(error) => return Err(error.into()),
                        }
                    }
                    return Ok(source.clone());
                }
                Ok(false) => unreachable!("root declaration was authenticated above"),
                Err(error) => {
                    rejected.get_or_insert(error);
                }
            }
        }
    }
    let parents = resolve_manifest_sources(root, provided, cache).map_err(|error| {
        if let Some(rejected) = rejected {
            anyhow::anyhow!(
                "root secular source unavailable: {error}; supplied candidate rejected: {rejected}"
            )
        } else {
            error.into()
        }
    })?;
    let mut secular = parents
        .into_iter()
        .filter(|m| m.key.kind == "ccm_secular_source");
    let source = secular
        .next()
        .ok_or_else(|| anyhow::anyhow!("root secular ancestry missing"))?;
    ensure!(secular.next().is_none(), "ambiguous root secular ancestry");
    // A declared mismatch must not be rescued by an exact publication edge.
    check(&source)?;
    Ok(source)
}
