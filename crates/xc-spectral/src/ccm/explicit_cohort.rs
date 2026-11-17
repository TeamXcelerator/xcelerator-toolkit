//! Optional frozen retained-source comparisons. Nothing in capture invokes this module.
//! Source stamps authenticate finite point inputs; they are not proof certificates.
use anyhow::{ensure, Result};
use rug::{Float, Rational};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, io::Read, path::Path};
use xc_cache::*;
use xc_core::{MemoryReservation, RunResources};
use xc_numerics::mpfr_interval::MpfrInterval as I;

use super::{
    retained_evidence::{self as retained, RetainedMatrix, RetainedRoots},
    state_geometry::RetainedState,
};

pub const SEMANTICS: &str = "ccm-explicit-retained-cohort-v4";
const KIND: &str = "ccm_finite_diagnostic_analysis";
const SCOPE: &str = "finite retained point comparisons; directed arithmetic bounds exclude source error; no ground-index, global root ordinal, convergence, continuum, or infinite-tail certificate";

/// Label-free source binding. Canonical manifest identity is required for published sources.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub kind: String,
    pub semantic_digest: ContentDigest,
    pub payload_digest: ContentDigest,
    pub canonical_manifest_digest: Option<ContentDigest>,
}
impl SourceIdentity {
    fn numerical_value(&self) -> Value {
        json!({"kind":self.kind,"payload_digest":self.payload_digest})
    }
    pub fn from_manifest(manifest: &ArtifactManifest) -> Result<Self> {
        semantic(manifest)?;
        Ok(Self {
            kind: manifest.key.kind.clone(),
            semantic_digest: manifest.key.parameters_digest.clone(),
            payload_digest: manifest.content_digest.clone(),
            canonical_manifest_digest: retained_canonical_manifest(manifest)?
                .map(|m| m.digest())
                .transpose()?,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortMemberRequest {
    pub id: String,
    pub state: SourceIdentity,
    pub matrix: Option<SourceIdentity>,
    pub roots: Option<SourceIdentity>,
}
/// Freeze one explicitly selected journal/inventory. Ambiguous distinct states or
/// root windows require the caller to narrow that selection; no cache discovery occurs.
pub fn freeze_member(
    id: impl Into<String>,
    selected_manifests: &[ArtifactManifest],
) -> Result<CohortMemberRequest> {
    let unique = |kinds: &[&str]| -> Result<Vec<SourceIdentity>> {
        let mut identities = Vec::new();
        for m in selected_manifests
            .iter()
            .filter(|m| kinds.contains(&m.key.kind.as_str()))
        {
            let identity = SourceIdentity::from_manifest(m)?;
            if !identities.contains(&identity) {
                identities.push(identity);
            }
        }
        Ok(identities)
    };
    let states = unique(&["ccm_weil_eigenpair"])?;
    let matrices = unique(&["ccm_tau_matrix"])?;
    ensure!(
        matrices.len() <= 1,
        "explicit member selection requires at most one distinct Tau"
    );
    let roots = unique(&["ccm_root_refinement", "ccm_root_discovery_window"])?;
    ensure!(states.len()==1 && roots.len()<=1,"explicit member selection requires one distinct state and at most one distinct root source");
    Ok(CohortMemberRequest {
        id: id.into(),
        state: states[0].clone(),
        matrix: matrices.into_iter().next(),
        roots: roots.into_iter().next(),
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortPairRequest {
    pub left: String,
    pub right: String,
    pub root_ordinals: Vec<usize>,
    #[serde(default)]
    pub embedded_residual: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenCohortPlan {
    pub schema_version: u32,
    pub members: Vec<CohortMemberRequest>,
    pub pairs: Vec<CohortPairRequest>,
}
impl FrozenCohortPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && self.members.len() >= 2
                && self.members.len() <= 1024
                && !self.pairs.is_empty()
                && self.pairs.len() <= 4096,
            "invalid frozen cohort schema or request size"
        );
        let mut ids = BTreeSet::new();
        for m in &self.members {
            ensure!(
                !m.id.is_empty() && m.id.len() <= 256 && ids.insert(&m.id),
                "duplicate or invalid cohort member ID"
            );
            ensure!(
                m.state.kind == "ccm_weil_eigenpair",
                "cohort member must name an eigenpair"
            );
            ensure!(
                m.matrix.as_ref().is_none_or(|m| m.kind == "ccm_tau_matrix"),
                "cohort matrix selection must name Tau"
            );
            for identity in std::iter::once(&m.state)
                .chain(m.roots.iter())
                .chain(m.matrix.iter())
            {
                ensure!(
                    identity.semantic_digest.validate()
                        && identity.payload_digest.validate()
                        && identity
                            .canonical_manifest_digest
                            .as_ref()
                            .is_none_or(ContentDigest::validate),
                    "invalid cohort source digest"
                );
            }
            ensure!(
                m.roots.as_ref().is_none_or(|r| matches!(
                    r.kind.as_str(),
                    "ccm_root_refinement" | "ccm_root_discovery_window"
                )),
                "invalid cohort roots kind"
            );
        }
        let mut pairs = BTreeSet::new();
        for pair in &self.pairs {
            ensure!(
                pair.left != pair.right && ids.contains(&pair.left) && ids.contains(&pair.right),
                "unknown or identical cohort pair endpoints"
            );
            let endpoints = if pair.left < pair.right {
                (&pair.left, &pair.right)
            } else {
                (&pair.right, &pair.left)
            };
            ensure!(pairs.insert(endpoints), "duplicate cohort pair");
            ensure!(
                pair.root_ordinals.len() <= 100000
                    && pair.root_ordinals.iter().all(|ordinal| *ordinal > 0)
                    && pair.root_ordinals.windows(2).all(|w| w[0] < w[1]),
                "root ordinals must be strictly increasing"
            );
        }
        ensure!(
            self.members
                .iter()
                .all(|m| self.pairs.iter().any(|p| p.left == m.id || p.right == m.id)),
            "every frozen member must participate in an explicit pair"
        );
        Ok(())
    }
    fn numerical_request(&self) -> Value {
        let index = |id: &str| self.members.iter().position(|m| m.id == id).unwrap();
        json!({"semantics":SEMANTICS,"members":self.members.iter().map(|m|json!({"state":m.state.numerical_value(),"roots":m.roots.as_ref().map(SourceIdentity::numerical_value)})).collect::<Vec<_>>(),
            "pairs":self.pairs.iter().map(|p|json!({"left":index(&p.left),"right":index(&p.right),"root_ordinals":p.root_ordinals,"embedded_residual":p.embedded_residual})).collect::<Vec<_>>()})
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortMeasurements {
    pub semantics: String,
    pub scope: String,
    pub numerical_identity: Option<ContentDigest>,
    pub execution_complete: bool,
    pub requested_coverage_complete: bool,
    pub members: Vec<Value>,
    pub pairs: Vec<Value>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CohortOperations {
    /// Logical decoded bytes actually opened, including repeated pair reads. Not network traffic.
    pub decoded_payload_bytes_opened: u64,
    pub dense_matrix_payload_opens: u64,
    pub peak_reserved_bytes: u64,
    pub elapsed_seconds: f64,
    pub report_bytes: u64,
    pub failures: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortExecution {
    pub measurements: CohortMeasurements,
    pub operations: CohortOperations,
    pub artifact: Option<ArtifactManifest>,
    pub cache_status: String,
}

fn semantic(m: &ArtifactManifest) -> Result<SemanticKeyEnvelope> {
    super::source_admission::semantic(m, None)?
        .ok_or_else(|| anyhow::anyhow!("retained source lacks authenticated semantic envelope"))
}
fn available_alias(m: &ArtifactManifest, cache: &ArtifactCacheContext<'_>) -> Result<bool> {
    let (Some(resolver), Some(policy)) = (cache.resolver, cache.acceptance) else {
        return Ok(false);
    };
    match resolver.resolve_exact_manifest(&m.key, &m.content_digest, policy) {
        Ok(_) => Ok(true),
        Err(CacheError::NotFound(_)) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn select<'a>(
    id: &SourceIdentity,
    manifests: &'a [ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<&'a ArtifactManifest> {
    fn parent_values(m: &ArtifactManifest) -> BTreeSet<(&str, &ContentDigest, &ContentDigest)> {
        m.dependencies
            .iter()
            .map(|d| {
                (
                    d.key.kind.as_str(),
                    &d.key.parameters_digest,
                    &d.content_digest,
                )
            })
            .collect()
    }

    let mut rejected = None;
    let mut selected = None;
    for m in manifests.iter().filter(|m| {
        m.key.kind == id.kind
            && m.key.parameters_digest == id.semantic_digest
            && m.content_digest == id.payload_digest
    }) {
        let candidate = super::source_admission::admit_manifest(m, cache.acceptance)
            .and_then(|()| SourceIdentity::from_manifest(m));
        match candidate {
            // Only the frozen selection is an observation. Other publications
            // may coexist in the inventory, but cannot replace that selection.
            Ok(candidate) if candidate == *id => {
                if let Some(first) = selected {
                    // Older native selections carry no manifest hash. When
                    // they also lack an explicit consumed-value declaration,
                    // conflicting parent values would make order pick a result.
                    let semantic = semantic(m)?;
                    let parameters = &semantic.resolved_mathematical_parameters;
                    let field = if id.kind == "ccm_weil_eigenpair" {
                        "tau_source_identity"
                    } else {
                        "secular_source_content_digest"
                    };
                    ensure!(
                        id.canonical_manifest_digest.is_some()
                            || parameters.get(field).is_some()
                            || parent_values(first) == parent_values(m),
                        "ambiguous legacy cohort source ancestry for {} {}",
                        id.kind,
                        id.semantic_digest
                    );
                    if !available_alias(first, cache)? && available_alias(m, cache)? {
                        selected = Some(m);
                    }
                } else {
                    selected = Some(m);
                }
            }
            Ok(_) => {
                rejected.get_or_insert_with(|| "canonical cohort source identity differs".into());
            }
            Err(error) => {
                rejected.get_or_insert_with(|| error.to_string());
            }
        }
    }
    if let Some(selected) = selected {
        return Ok(selected);
    }
    anyhow::bail!(
        "explicit cohort source unavailable: {} {}; {}",
        id.kind,
        id.semantic_digest.0,
        rejected
            .as_deref()
            .unwrap_or("no matching source candidate")
    )
}
struct Metadata {
    state: ArtifactManifest,
    tau: ArtifactManifest,
    roots: Option<ArtifactManifest>,
    secular: Option<ArtifactManifest>,
    ancestry: Vec<ArtifactManifest>,
    policy: Value,
}
impl Metadata {
    fn same_cutoff(&self, other: &Self) -> Result<bool> {
        let cutoff = |m: &Self| {
            xc_core::DecimalLiteral::new(
                m.policy["lambda_squared"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("missing cohort cutoff"))?,
            )
            .map_err(anyhow::Error::from)
        };
        Ok(cutoff(self)?.cmp_numeric(&cutoff(other)?)? == std::cmp::Ordering::Equal)
    }

    fn same_source(&self, other: &Self) -> bool {
        let same_bytes = |a: &ArtifactManifest, b: &ArtifactManifest| {
            a.key.kind == b.key.kind && a.content_digest == b.content_digest
        };
        same_bytes(&self.state, &other.state)
            && same_bytes(&self.tau, &other.tau)
            && match (&self.roots, &other.roots) {
                (Some(a), Some(b)) => same_bytes(a, b),
                (None, None) => true,
                _ => false,
            }
            && self.policy == other.policy
    }
}
fn metadata(
    request: &CohortMemberRequest,
    manifests: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<Metadata> {
    let state = select(&request.state, manifests, cache)?.clone();
    let state_semantic = semantic(&state)?;
    let mut queue = vec![(state.clone(), Vec::new())];
    let mut seen = BTreeSet::new();
    let mut chain = None;
    let mut rejected_tau = None;
    let selected_tau = request
        .matrix
        .as_ref()
        .map(|id| select(id, manifests, cache))
        .transpose()?;
    if let Some(digest) = super::source_admission::declared_tau_digest(&state, cache.acceptance)? {
        for tau in manifests
            .iter()
            .filter(|m| m.key.kind == "ccm_tau_matrix" && m.content_digest == digest)
        {
            if selected_tau.is_some_and(|selected| selected != tau) {
                continue;
            }
            if let Err(error) =
                super::source_admission::declared_tau_matches(&state, tau, cache.acceptance)
            {
                rejected_tau.get_or_insert(error);
                continue;
            }
            if !super::source_admission::tau_edge_matches(&state, tau)? {
                rejected_tau.get_or_insert_with(|| {
                    anyhow::anyhow!("supplied Tau lacks a matching direct producer edge")
                });
                continue;
            }
            if let Some((old, _)) = &chain {
                // Provenance aliases are interchangeable. Different operator
                // policies still need an explicit selection for a cohort.
                let old: &ArtifactManifest = old;
                ensure!(
                    old.key.parameters_digest == tau.key.parameters_digest,
                    "ambiguous declared Tau operator policies"
                );
                if !available_alias(old, cache)? && available_alias(tau, cache)? {
                    chain = Some((tau.clone(), vec![tau.clone()]));
                }
            } else {
                chain = Some((tau.clone(), vec![tau.clone()]));
            }
        }
    }
    if chain.is_none() {
        while let Some((m, path)) = queue.pop() {
            ensure!(
                path.len() <= 16 && seen.len() <= 64,
                "cohort matrix ancestry exceeds metadata budget"
            );
            if !seen.insert((
                m.key.kind.clone(),
                m.key.parameters_digest.clone(),
                m.content_digest.clone(),
            )) {
                continue;
            }
            let parents = resolve_manifest_sources(&m, manifests, cache).map_err(|error| {
                if let Some(rejected) = &rejected_tau {
                    anyhow::anyhow!(
                        "Tau source unavailable: {error}; supplied candidate rejected: {rejected}"
                    )
                } else {
                    error.into()
                }
            })?;
            for next in parents {
                super::source_admission::admit_manifest(&next, cache.acceptance)?;
                semantic(&next)?;
                let mut ancestry = path.clone();
                ancestry.push(next.clone());
                if next.key.kind == "ccm_tau_matrix" {
                    let next = if let Some(selected) = selected_tau {
                        if next.key.parameters_digest != selected.key.parameters_digest
                            || next.content_digest != selected.content_digest
                        {
                            continue;
                        }
                        super::source_admission::admit_manifest(selected, cache.acceptance)?;
                        semantic(selected)?;
                        *ancestry.last_mut().unwrap() = selected.clone();
                        selected.clone()
                    } else {
                        next
                    };
                    if let Some((old, _)) = &chain {
                        ensure!(
                            SourceIdentity::from_manifest(old)?
                                == SourceIdentity::from_manifest(&next)?,
                            "ambiguous Tau ancestry"
                        );
                    } else {
                        chain = Some((next, ancestry));
                    }
                } else if matches!(
                    next.key.kind.as_str(),
                    "ccm_factorization" | "ccm_even_sector_matrix" | "ccm_odd_sector_matrix"
                ) {
                    queue.push((next, ancestry));
                }
            }
        }
    }
    let (tau, ancestry) = chain.ok_or_else(|| match rejected_tau {
        Some(reason) => anyhow::anyhow!("missing authenticated Tau source: {reason}"),
        None => anyhow::anyhow!("missing authenticated matrix/factorization/sector ancestry"),
    })?;
    super::source_admission::admit_manifest(&tau, cache.acceptance)?;
    let tau_semantic = semantic(&tau)?;
    let a = &state_semantic.resolved_mathematical_parameters;
    let b = &tau_semantic.resolved_mathematical_parameters;
    for field in ["lambda_squared", "n_modes", "precision_bits"] {
        ensure!(
            !a[field].is_null() && a[field] == b[field],
            "matrix/state semantic {field} mismatch"
        );
    }
    super::source_admission::declared_tau_matches(&state, &tau, cache.acceptance)?;
    ensure!(
        state_semantic.normalization.as_deref() == Some("sum_xi_equals_sqrt_log_lambda_squared")
            && a["normalization"] == "sum_xi_equals_sqrt_log_lambda_squared",
        "unsupported state normalization"
    );
    ensure!(
        tau_semantic.normalization.as_deref() == Some("symmetric_row_major"),
        "unsupported matrix coefficient basis/normalization"
    );
    let orders = b["resolved_quadrature_orders"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing actual quadrature orders"))?;
    ensure!(
        !orders.is_empty()
            && orders.len() <= 8193
            && orders.iter().all(|q| q.as_u64().is_some_and(|v| v > 0)),
        "invalid actual quadrature orders"
    );
    let (roots, secular) = match &request.roots {
        Some(id) => {
            let root = select(id, manifests, cache)?.clone();
            super::source_admission::admit_manifest(&root, cache.acceptance)?;
            let secular =
                super::source_admission::resolve_root_secular(&root, &state, manifests, cache)?;
            super::source_admission::admit_manifest(&secular, cache.acceptance)?;
            semantic(&secular)?;
            (Some(root), Some(secular))
        }
        None => (None, None),
    };
    // Historical Tau declarations included the adapter's store key. Admission
    // has already checked the retained digest and semantics. Keep those fields
    // in the comparison policy without making store aliases part of the report.
    let mut state_policy = a.clone();
    if let Some(key) = state_policy
        .get_mut("tau_source_identity")
        .and_then(|identity| identity.get_mut("key"))
        .and_then(Value::as_object_mut)
    {
        key.remove("logical_key");
    }
    let policy = json!({"lambda_squared":a["lambda_squared"],"n_modes":a["n_modes"],"precision_bits":a["precision_bits"],
        "normalization":state_semantic.normalization,"state_subspace":state_semantic.subspace,
        "state_semantics":state_semantic.mathematical_semantics_version,"state_route":a.get("eigenstate_route").cloned().unwrap_or(json!("legacy_inverse_iteration")),
        "state_policy":state_policy,"operator_semantics":tau_semantic.mathematical_semantics_version,"operator_algorithm":tau_semantic.algorithm_semantics,
        "operator_policy":b,"operator_normalization":tau_semantic.normalization,"operator_source_data":tau_semantic.source_data_identities,
        "basis":"signed_fourier_modes_minus_N_through_N; zero_padding_at_equal_cutoff","metric":"identity_coefficient_metric"});
    Ok(Metadata {
        state,
        tau,
        roots,
        secular,
        ancestry,
        policy,
    })
}
fn parents(metadata: &[Metadata]) -> Vec<ArtifactManifest> {
    let mut out = Vec::new();
    for m in metadata {
        out.push(m.state.clone());
        out.extend(m.ancestry.clone());
        out.extend(m.roots.clone());
        out.extend(m.secular.clone());
    }
    out.sort_by(|a, b| {
        (
            &a.key.kind,
            &a.key.logical_key,
            &a.key.parameters_digest,
            &a.content_digest,
        )
            .cmp(&(
                &b.key.kind,
                &b.key.logical_key,
                &b.key.parameters_digest,
                &b.content_digest,
            ))
    });
    out.dedup_by(|a, b| a.key == b.key && a.content_digest == b.content_digest);
    out
}
struct Loaded {
    state: RetainedState,
    roots: Option<RetainedRoots>,
    evidence: Value,
    _reservation: MemoryReservation,
}

fn read(
    m: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
    resources: &RunResources,
    scratch: &Path,
    operations: &mut CohortOperations,
) -> Result<Vec<u8>> {
    let _reservation = resources.try_reserve_working(
        m.size_bytes
            .checked_add(2 << 20)
            .ok_or_else(|| anyhow::anyhow!("input byte overflow"))?,
    )?;
    let mut source = cache
        .resolver
        .ok_or_else(|| anyhow::anyhow!("cohort requires an exact retained source resolver"))?
        .open_exact_payload(
            &m.key,
            &m.content_digest,
            m.quality,
            cache
                .acceptance
                .ok_or_else(|| anyhow::anyhow!("cohort requires source acceptance policy"))?,
            scratch,
            m.size_bytes,
        )?;
    operations.decoded_payload_bytes_opened = operations
        .decoded_payload_bytes_opened
        .checked_add(m.size_bytes)
        .ok_or_else(|| anyhow::anyhow!("byte counter overflow"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(m.size_bytes)?)?;
    source.payload.read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == m.size_bytes && ContentDigest::sha256(&bytes) == m.content_digest,
        "retained payload size/digest mismatch"
    );
    Ok(bytes)
}
fn load(
    m: &Metadata,
    cache: &ArtifactCacheContext<'_>,
    resources: &RunResources,
    scratch: &Path,
    operations: &mut CohortOperations,
) -> Result<Loaded> {
    let modes = m.policy["n_modes"]
        .as_u64()
        .filter(|n| *n <= 8192)
        .ok_or_else(|| anyhow::anyhow!("unsupported cohort state dimension"))?;
    let precision = m.policy["precision_bits"]
        .as_u64()
        .filter(|p| (64..=1_000_000).contains(p))
        .ok_or_else(|| anyhow::anyhow!("unsupported cohort state precision"))?;
    // Short scalar strings still allocate full-precision MPFR mantissas. Round
    // up to 64-bit limbs and include per-scalar overhead and arithmetic scratch.
    let scalar_bytes = (precision + 64).div_ceil(64) * 8 + 128;
    let decoded_bytes = (2 * modes + 1 + 32) * scalar_bytes;
    // In addition to decoded coefficients, cover JSON buffers, parsed root
    // records and metadata. Keep the charge as long as this member is loaded.
    let bytes = m
        .state
        .size_bytes
        .checked_add(m.roots.as_ref().map_or(0, |m| m.size_bytes))
        .and_then(|n| n.checked_add(m.secular.as_ref().map_or(0, |m| m.size_bytes)))
        .ok_or_else(|| anyhow::anyhow!("source allocation overflow"))?;
    let reservation = resources.try_reserve_retained(
        bytes
            .checked_mul(24)
            .and_then(|n| n.checked_add(decoded_bytes))
            .and_then(|n| n.checked_add(2 << 20))
            .ok_or_else(|| anyhow::anyhow!("source allocation overflow"))?,
    )?;
    let state_bytes = read(&m.state, cache, resources, scratch, operations)?;
    let raw: Value = serde_json::from_slice(&state_bytes)?;
    let p = &m.policy;
    ensure!(
        raw["lambda_squared"] == p["lambda_squared"]
            && raw["n_modes"] == p["n_modes"]
            && raw["precision_bits"] == p["precision_bits"],
        "state payload and authenticated policy disagree"
    );
    for field in ["force_even", "parity_policy", "eigenstate_route"] {
        if let Some(declared) = p["state_policy"].get(field) {
            ensure!(
                raw.get(field) == Some(declared),
                "state payload and semantic {field} disagree"
            );
        }
    }
    let state = RetainedState::from_payload(
        &m.state,
        &state_bytes,
        std::slice::from_ref(&m.state.content_digest),
    )?;
    let roots = match (&m.roots, &m.secular) {
        (Some(root), Some(sec)) => {
            let root_bytes = read(root, cache, resources, scratch, operations)?;
            let secular_bytes = read(sec, cache, resources, scratch, operations)?;
            let root_raw: Value = serde_json::from_slice(&root_bytes)?;
            let secular_raw: Value = serde_json::from_slice(&secular_bytes)?;
            let root_semantic = semantic(root)?;
            let secular_semantic = semantic(sec)?;
            for field in ["lambda_squared", "n_modes", "precision_bits"] {
                ensure!(
                    root_semantic.resolved_mathematical_parameters[field] == raw[field]
                        && secular_semantic.resolved_mathematical_parameters[field] == raw[field],
                    "root/secular semantic source shape mismatch"
                );
            }
            // Current root declarations are checked by the shared decoder.
            // Historical cohort inputs still require a full acquisition policy.
            if root_semantic
                .resolved_mathematical_parameters
                .get("secular_source_content_digest")
                .is_none()
            {
                super::source_admission::authenticate_root_policy(root, &root_semantic, &root_raw)?;
            }
            ensure!(
                secular_semantic.normalization.as_deref()
                    == Some("sum_xi_equals_sqrt_log_lambda_squared")
                    && secular_raw["normalization"] == "sum_xi_equals_sqrt_log_lambda_squared",
                "secular semantic normalization mismatch"
            );
            Some(RetainedRoots::from_payload(
                root,
                &root_bytes,
                sec,
                &secular_bytes,
                &state,
                &[root.content_digest.clone(), sec.content_digest.clone()],
            )?)
        }
        _ => None,
    };
    let evidence = json!({"ground_index":raw["ground_index"],"stored_state_resolution":raw["stored_state_resolution"],
        "ground_evidence_scope":"inherited authenticated payload fields; not independently replayed; absence remains unresolved"});
    Ok(Loaded {
        state,
        roots,
        evidence,
        _reservation: reservation,
    })
}
fn exact(x: &Float) -> Result<Rational> {
    ensure!(
        x.get_exp().is_none_or(|e| e.unsigned_abs() <= 131072),
        "exact scalar exceeds exponent budget"
    );
    x.to_rational()
        .ok_or_else(|| anyhow::anyhow!("nonfinite scalar"))
}
/// Bound decimal expansion from the actual numerator and denominator before formatting.
struct ExactReportText<'a> {
    resources: &'a RunResources,
    remaining: u64,
    reservations: Vec<MemoryReservation>,
}
impl ExactReportText<'_> {
    fn decimal(&mut self, value: &Rational) -> Result<String> {
        let bits = u64::from(value.numer().significant_bits())
            + u64::from(value.denom().significant_bits());
        // Three binary bits need at most one decimal digit, plus sign and slash.
        // Allow both the formatted string and its owned JSON copy.
        let bytes = (bits.div_ceil(3) + 6) * 2;
        if bytes > self.remaining {
            let additional = (bytes - self.remaining).max(4096);
            self.reservations
                .push(self.resources.try_reserve_retained(additional)?);
            self.remaining += additional;
        }
        self.remaining -= bytes;
        Ok(value.to_string())
    }
}

fn bounds(x: &I, text: &mut ExactReportText<'_>) -> Result<Value> {
    x.validate()?;
    Ok(json!({"lower":text.decimal(&exact(x.lower())?)?,"upper":text.decimal(&exact(x.upper())?)?}))
}
fn delta(a: &Float, b: &Float, text: &mut ExactReportText<'_>) -> Result<Value> {
    let a = exact(a)?;
    let b = exact(b)?;
    let signed = Rational::from(&b - &a);
    let absolute = signed.clone().abs();
    Ok(
        json!({"signed":text.decimal(&signed)?,"absolute":text.decimal(&absolute)?,
        "relative_to_left":if a==0 {None}else{Some(text.decimal(&(&absolute/a.clone().abs()))?)},
        "relative_denominator_zero":a==0,"sign_crossing":(a<0 && b>0)||(a>0 && b<0)}),
    )
}
fn overlap(
    a: &RetainedState,
    b: &RetainedState,
    p: u32,
    text: &mut ExactReportText<'_>,
) -> Result<Value> {
    let mut dot = I::from_i64(0, p);
    let mut aa = I::from_i64(0, p);
    let mut bb = I::from_i64(0, p);
    for x in &a.coefficients {
        aa = aa.add(&I::from_float(x, p)?.square());
    }
    for x in &b.coefficients {
        bb = bb.add(&I::from_float(x, p)?.square());
    }
    let n = a.modes.min(b.modes);
    for i in 0..=2 * n {
        let x = I::from_float(&a.coefficients[a.modes - n + i], p)?;
        let y = I::from_float(&b.coefficients[b.modes - n + i], p)?;
        dot = dot.add(&x.mul(&y));
    }
    let absolute = retained::finite_math::abs(&dot)?.div(&aa.mul(&bb).sqrt()?)?;
    // Intersect with a theorem-backed range, retaining both raw and tightened bounds.
    let unit = I::new(Float::with_val(p, 0), Float::with_val(p, 1))?;
    let bounded = absolute
        .intersection(&unit)
        .ok_or_else(|| anyhow::anyhow!("overlap range inconsistent"))?;
    let squared = I::from_i64(2, p).sub(&I::from_i64(2, p).mul(&bounded));
    Ok(
        json!({"normalized_absolute_overlap":bounds(&bounded, text)?,"raw_overlap_arithmetic_enclosure":bounds(&absolute, text)?,
        "sign_aligned_distance_squared":bounds(&squared, text)?,"sign_aligned_distance":bounds(&squared.sqrt()?, text)?,
        "normalization":"both retained coefficient vectors normalized in identity metric; smaller signed Fourier vector zero padded",
        "range_tightening":"Cauchy-Schwarz intersection with [0,1]; no acceptance test",
        "branch_qualification":"single-vector rotation in an unresolved cluster remains possible; no target-branch assertion"}),
    )
}

fn changed(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    if a == b {
        return;
    }
    if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
        let keys = a.keys().chain(b.keys()).collect::<BTreeSet<_>>();
        for k in keys {
            changed(
                a.get(k).unwrap_or(&Value::Null),
                b.get(k).unwrap_or(&Value::Null),
                &format!("{path}{k}."),
                out,
            );
        }
    } else {
        out.push(path.trim_end_matches('.').into());
    }
}

#[allow(clippy::too_many_arguments)]
fn residual(
    left: &Loaded,
    right: &Loaded,
    meta: &Metadata,
    p: u32,
    cache: &ArtifactCacheContext<'_>,
    resources: &RunResources,
    scratch: &Path,
    operations: &mut CohortOperations,
    text: &mut ExactReportText<'_>,
) -> Result<Value> {
    ensure!(
        left.state.modes < right.state.modes,
        "embedded residual requires a smaller left state"
    );
    let resolver = cache
        .resolver
        .ok_or_else(|| anyhow::anyhow!("missing source resolver"))?;
    let _spool = resources.try_reserve_working(2 << 20)?;
    let mut source = resolver.open_exact_payload(
        &meta.tau.key,
        &meta.tau.content_digest,
        meta.tau.quality,
        cache
            .acceptance
            .ok_or_else(|| anyhow::anyhow!("missing acceptance policy"))?,
        scratch,
        meta.tau.size_bytes,
    )?;
    operations.dense_matrix_payload_opens += 1;
    operations.decoded_payload_bytes_opened += meta.tau.size_bytes;
    let matrix = RetainedMatrix::from_reader_with_resources(
        &meta.tau,
        &mut source.payload,
        std::slice::from_ref(&meta.tau.content_digest),
        resources,
    )?;
    matrix.match_state(&right.state)?;
    let dim = 2 * right.state.modes + 1;
    let n = left.state.modes;
    let offset = right.state.modes - n;
    let lambda = I::from_float(
        &retained::scalar(&left.state.eigenvalue, left.state.precision)?,
        p,
    )?;
    let mut low = I::from_i64(0, p);
    let mut high = I::from_i64(0, p);
    let mut norm = I::from_i64(0, p);
    for x in &left.state.coefficients {
        norm = norm.add(&I::from_float(x, p)?.square());
    }
    for i in 0..dim {
        resources.check_cancelled()?;
        let mut r = I::from_i64(0, p);
        for j in 0..=2 * n {
            r = r.add(
                &I::from_float(&matrix.entries[i * dim + offset + j], p)?
                    .mul(&I::from_float(&left.state.coefficients[j], p)?),
            );
        }
        if (offset..=offset + 2 * n).contains(&i) {
            r = r.sub(&lambda.mul(&I::from_float(&left.state.coefficients[i - offset], p)?));
            low = low.add(&r.square());
        } else {
            high = high.add(&r.square());
        }
    }
    low = low.div(&norm)?;
    high = high.div(&norm)?;
    Ok(
        json!({"status":"measured","matrix":SourceIdentity::from_manifest(&meta.tau)?.numerical_value(),"low_mode_residual_squared":bounds(&low, text)?,"high_mode_forcing_squared":bounds(&high, text)?,"total_residual_squared":bounds(&low.add(&high), text)?,
        "scope":"directed residual of exact stored larger point matrix and stored smaller state/eigenvalue; assembly error and analytic source uncertainty excluded",
        "state_error_bound":"unresolved: matching matrix separation and target-state identification not established",
        "low_mode_qualification":"includes changed assembly/quadrature and smaller state's residual; not pure truncation error"}),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "fresh measurement and source replay share explicit plan, cache, budget and comparison inputs"
)]
fn measure(
    plan: &FrozenCohortPlan,
    metadata: &[Metadata],
    identity: ContentDigest,
    cache: &ArtifactCacheContext<'_>,
    resources: &RunResources,
    scratch: &Path,
    operations: &mut CohortOperations,
    expected: Option<&CohortMeasurements>,
) -> Result<CohortMeasurements> {
    // Reserve structural storage before constructing rows. Exact decimal text
    // is charged separately from measured numerator and denominator widths.
    let max_precision = metadata
        .iter()
        .filter_map(|m| m.policy["precision_bits"].as_u64())
        .max()
        .unwrap_or(128);
    ensure!(max_precision <= 1_000_000, "unsupported cohort precision");
    // During source replay, compare and drop each root row immediately.
    // Only one root row is constructed alongside the pair measurements.
    let rows = if expected.is_some() {
        plan.pairs.len() as u64 + u64::from(plan.pairs.iter().any(|p| !p.root_ordinals.is_empty()))
    } else {
        plan.pairs.iter().try_fold(0u64, |n, p| {
            n.checked_add(p.root_ordinals.len() as u64 + 1)
                .ok_or_else(|| anyhow::anyhow!("report row count overflow"))
        })?
    };
    let metadata_bytes = metadata
        .iter()
        .flat_map(|m| {
            std::iter::once(&m.state)
                .chain(m.ancestry.iter())
                .chain(m.roots.iter())
                .chain(m.secular.iter())
        })
        .flat_map(|m| m.tags.values())
        .try_fold(0u64, |n, s| {
            n.checked_add(s.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("report metadata size overflow"))
        })?;
    // Row keys, JSON nodes and live scalar scratch. Exact decimal output is
    // reserved separately below from its measured numerator/denominator width.
    let estimate = max_precision
        .div_ceil(8)
        .checked_mul(16)
        .and_then(|bytes| bytes.checked_add(2048))
        .and_then(|n| n.checked_mul(rows))
        .and_then(|n| {
            metadata_bytes
                .checked_mul(16)
                .and_then(|m| n.checked_add(m))
        })
        .ok_or_else(|| anyhow::anyhow!("report workspace estimate overflow"))?;
    let _output_reservation = resources.try_reserve_retained(estimate)?;
    let mut text = ExactReportText {
        resources,
        remaining: 0,
        reservations: Vec::new(),
    };
    let mut members=metadata.iter().map(|m|Ok(json!({"state":SourceIdentity::from_manifest(&m.state)?.numerical_value(),"matrix":SourceIdentity::from_manifest(&m.tau)?.numerical_value(),"roots":m.roots.as_ref().map(|m|SourceIdentity::from_manifest(m).map(|i|i.numerical_value())).transpose()?,"policy":m.policy,
        "matrix_scope":"authenticated metadata ancestry only; matrix payload unopened unless residual requested"}))).collect::<Result<Vec<_>>>()?;
    let mut pairs = Vec::new();
    let mut coverage = true;
    for (pair_index, pair) in plan.pairs.iter().enumerate() {
        resources.check_cancelled()?;
        let expected_pair = expected.map(|report| &report.pairs[pair_index]);
        let ai = plan.members.iter().position(|m| m.id == pair.left).unwrap();
        let bi = plan
            .members
            .iter()
            .position(|m| m.id == pair.right)
            .unwrap();
        let a = load(&metadata[ai], cache, resources, scratch, operations)?;
        let b = load(&metadata[bi], cache, resources, scratch, operations)?;
        members[ai]["retained_evidence"] = a.evidence.clone();
        members[bi]["retained_evidence"] = b.evidence.clone();
        for (index, source) in [(ai, &a), (bi, &b)] {
            if let Some(r) = &source.roots {
                members[index]["root_acquisition"] = r.acquisition.clone();
                members[index]["root_completeness"] = json!(r.completeness);
            }
        }
        let p = a
            .state
            .precision
            .max(b.state.precision)
            .checked_add(64)
            .ok_or_else(|| anyhow::anyhow!("working precision overflow"))?;
        let mut differences = Vec::new();
        changed(
            &metadata[ai].policy,
            &metadata[bi].policy,
            "",
            &mut differences,
        );
        let same_cutoff = metadata[ai].same_cutoff(&metadata[bi])?;
        if !same_cutoff {
            coverage = false;
        }
        let state = if same_cutoff {
            overlap(&a.state, &b.state, p, &mut text)?
        } else {
            json!({"status":"incompatible","reason":"different cutoff: zero padding is not a common physical Fourier basis"})
        };
        let mut root_rows = Vec::new();
        for (row_index, ordinal) in pair.root_ordinals.iter().enumerate() {
            let mut scratch_text = ExactReportText {
                resources,
                remaining: 0,
                reservations: Vec::new(),
            };
            let row_text = if expected_pair.is_some() {
                &mut scratch_text
            } else {
                &mut text
            };
            let find = |s: &Loaded| {
                s.roots
                    .as_ref()
                    .and_then(|r| r.dataset.points.iter().find(|r| r.ordinal == *ordinal))
                    .cloned()
            };
            let left = find(&a);
            let right = find(&b);
            let values = left
                .as_ref()
                .and_then(|r| r.value.as_ref())
                .zip(right.as_ref().and_then(|r| r.value.as_ref()));
            let same_domain = a.roots.as_ref().map(|r| &r.acquisition["root_domain"])
                == b.roots.as_ref().map(|r| &r.acquisition["root_domain"]);
            let row = if let Some((x, y)) = values.filter(|_| same_domain) {
                coverage &= left.as_ref().unwrap().source_status == "converged"
                    && right.as_ref().unwrap().source_status == "converged";
                json!({"ordinal":ordinal,"status":"point_comparison","left_status":left.as_ref().unwrap().source_status,"right_status":right.as_ref().unwrap().source_status,"change":delta(&retained::scalar(x,a.state.precision)?,&retained::scalar(y,b.state.precision)?, row_text)?,"certified_difference_enclosure":null})
            } else {
                coverage = false;
                json!({"ordinal":ordinal,"status":"unresolved","left_status":left.map(|r|r.source_status),"right_status":right.map(|r|r.source_status),"reason":"requested retained root absent or failed; no silent window intersection"})
            };
            if let Some(cached) = expected_pair {
                ensure!(
                    cached["roots"][row_index] == row,
                    "cohort cached root measurement differs from retained source replay"
                );
            } else {
                root_rows.push(row);
            }
        }
        let embedded = if pair.embedded_residual {
            residual(
                &a,
                &b,
                &metadata[bi],
                p,
                cache,
                resources,
                scratch,
                operations,
                &mut text,
            )?
        } else {
            json!({"status":"not_requested"})
        };
        let q_a = &metadata[ai].policy["operator_policy"]["resolved_quadrature_orders"];
        let q_b = &metadata[bi].policy["operator_policy"]["resolved_quadrature_orders"];
        let uniform = |q: &Value| {
            q.as_array()
                .filter(|q| !q.is_empty() && q.iter().all(|x| x == &q[0]))
                .map(|q| q[0].clone())
        };
        let q_same = q_a == q_b || uniform(q_a).zip(uniform(q_b)).is_some_and(|(x, y)| x == y);
        let measured = json!({"left":ai,"right":bi,"same_source_replay":metadata[ai].same_source(&metadata[bi]),
            "changed_fields":differences,"comparison_class":if !same_cutoff {"parameter_change"}else if a.state.precision!=b.state.precision && !q_same {"coupled_precision_quadrature_change"}else{"finite_configuration_comparison"},
            "eigenvalue_change":delta(&retained::scalar(&a.state.eigenvalue,a.state.precision)?,&retained::scalar(&b.state.eigenvalue,b.state.precision)?, &mut text)?,
            "eigenvalue_qualification":"stored point changes only; retained ground/precision-floor evidence is inherited, not newly certified",
            "state":state,"roots":root_rows,"requested_root_count":pair.root_ordinals.len(),"ordinal_scope":"fixed retained acquisition ordinals; seeded and independent routes retained; no global kth-root or branch certification",
            "root_enclosure_scope":"stored point inputs only; converged status is not an interval certificate","embedded_residual":embedded,
            "numerical_acceptance":"not_requested; no tolerance, extrapolation, improving-chain, or convergence predicate"});
        if let Some(cached) = expected_pair {
            let fields = measured.as_object().expect("pair measurement is an object");
            ensure!(
                cached
                    .as_object()
                    .is_some_and(|object| object.len() == fields.len())
                    && fields
                        .iter()
                        .all(|(key, value)| key == "roots" || cached.get(key) == Some(value)),
                "cohort cached pair measurements differ from retained source replay"
            );
        } else {
            pairs.push(measured);
        }
    }
    if let Some(cached) = expected {
        ensure!(
            cached.members == members && cached.requested_coverage_complete == coverage,
            "cohort cached member measurements differ from retained source replay"
        );
    }
    Ok(CohortMeasurements {
        semantics: SEMANTICS.into(),
        scope: SCOPE.into(),
        numerical_identity: Some(identity),
        execution_complete: true,
        requested_coverage_complete: coverage,
        members,
        pairs,
    })
}

// Only reports independently calculated by this process enter this bounded memo.
// Including the exact report digest forces replay if any scientific field changes.
fn verified_reports(
) -> &'static std::sync::Mutex<std::collections::VecDeque<(ContentDigest, ContentDigest)>> {
    static REPORTS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::VecDeque<(ContentDigest, ContentDigest)>>,
    > = std::sync::OnceLock::new();
    REPORTS.get_or_init(Default::default)
}
fn remember_report(identity: &ContentDigest, report: &CohortMeasurements) -> Result<()> {
    let entry = (identity.clone(), canonical_digest(report)?);
    let mut verified = verified_reports()
        .lock()
        .map_err(|_| anyhow::anyhow!("cohort verification lock poisoned"))?;
    if !verified.contains(&entry) {
        if verified.len() >= 64 {
            verified.pop_front();
        }
        verified.push_back(entry);
    }
    Ok(())
}

/// Analyze exactly the named members and edges. Metadata walks never decode Tau.
/// Each pair owns at most two lightweight states and its explicitly requested larger matrix.
/// The supplied cache controls reuse/staging; no primary provider is reachable here.
pub fn analyze_frozen_cohort(
    plan: &FrozenCohortPlan,
    manifests: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
    resources: &RunResources,
    scratch_directory: &Path,
) -> Result<CohortExecution> {
    plan.validate()?;
    std::fs::create_dir_all(scratch_directory)?;
    let _scope = resources.enter();
    let started = std::time::Instant::now();
    let mut operations = CohortOperations::default();
    let metadata = plan
        .members
        .iter()
        .map(|m| metadata(m, manifests, cache))
        .collect::<Result<Vec<_>>>()
        .and_then(|rows| {
            let index = |id: &str| plan.members.iter().position(|m| m.id == id).unwrap();
            for pair in plan.pairs.iter().filter(|pair| pair.embedded_residual) {
                let left = &rows[index(&pair.left)];
                let right = &rows[index(&pair.right)];
                let increasing = left.policy["n_modes"]
                    .as_u64()
                    .zip(right.policy["n_modes"].as_u64())
                    .is_some_and(|(a, b)| a < b);
                ensure!(
                    increasing && left.same_cutoff(right)?,
                    "embedded residual requires smaller left state at equal cutoff"
                );
            }
            Ok(rows)
        });
    let mut request = plan.numerical_request();
    let matrices = metadata
        .as_ref()
        .ok()
        .map(|rows| {
            rows.iter()
                .map(|m| SourceIdentity::from_manifest(&m.tau))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    request["matrices"] = json!(matrices
        .iter()
        .map(SourceIdentity::numerical_value)
        .collect::<Vec<_>>());
    request["policy_digests"] = json!(metadata
        .as_ref()
        .ok()
        .map(|rows| rows
            .iter()
            .map(|m| canonical_digest(&m.policy))
            .collect::<std::result::Result<Vec<_>, _>>())
        .transpose()?
        .unwrap_or_default());
    let identity = ContentDigest::sha256(&serde_json::to_vec(&request)?);
    let incomplete = || {
        CohortMeasurements {semantics:SEMANTICS.into(),scope:SCOPE.into(),numerical_identity:None,execution_complete:false,requested_coverage_complete:false,
        members:plan.members.iter().map(|m|json!({"state":m.state,"matrix":m.matrix,"roots":m.roots,"status":"unavailable_or_not_evaluated"})).collect(),
        pairs:plan.pairs.iter().map(|p|json!({"left":plan.members.iter().position(|m|m.id==p.left).unwrap(),"right":plan.members.iter().position(|m|m.id==p.right).unwrap(),"status":"unresolved","requested_root_ordinals":p.root_ordinals,"embedded_residual_requested":p.embedded_residual})).collect()}
    };
    let mut artifact = None;
    let mut cache_status = "incomplete_not_persisted".to_string();
    let measurements = match metadata {
        Err(e) => {
            operations.failures.push(e.to_string());
            incomplete()
        }
        Ok(metadata) => {
            let parents = parents(&metadata);
            let counters = std::cell::RefCell::new(&mut operations);
            let calculate = |expected: Option<&CohortMeasurements>| {
                measure(
                    plan,
                    &metadata,
                    identity.clone(),
                    cache,
                    resources,
                    scratch_directory,
                    &mut counters.borrow_mut(),
                    expected,
                )
            };
            let result = retained::managed_with_input_identities(
                KIND,
                request,
                &parents,
                cache,
                &std::collections::BTreeMap::from([
                    (
                        "cohort_source_selection".into(),
                        serde_json::to_string(plan)?,
                    ),
                    (
                        "cohort_matrix_selection".into(),
                        serde_json::to_string(&matrices)?,
                    ),
                    (
                        "cohort_producer_edges".into(),
                        serde_json::to_string(
                            &parents
                                .iter()
                                .map(|parent| {
                                    Ok((
                                        SourceIdentity::from_manifest(parent)?,
                                        &parent.dependencies,
                                    ))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        )?,
                    ),
                ]),
                Some(&[]),
                || {
                    let report = calculate(None)?;
                    remember_report(&identity, &report)?;
                    Ok(report)
                },
                |r: &CohortMeasurements| {
                    ensure!(
                        r.semantics == SEMANTICS
                            && r.numerical_identity.as_ref() == Some(&identity)
                            && r.execution_complete
                            && r.scope == SCOPE
                            && r.members.len() == plan.members.len()
                            && r.pairs.len() == plan.pairs.len(),
                        "cohort cached report binding mismatch"
                    );
                    for (i, member) in r.members.iter().enumerate() {
                        let request = &plan.members[i];
                        ensure!(
                            member["state"] == request.state.numerical_value()
                                && member["matrix"] == matrices[i].numerical_value()
                                && member["policy"] == metadata[i].policy
                                && member["roots"]
                                    == serde_json::to_value(
                                        request.roots.as_ref().map(SourceIdentity::numerical_value)
                                    )?,
                            "cohort cached member binding mismatch"
                        );
                    }
                    let mut coverage = true;
                    for (row, request) in r.pairs.iter().zip(&plan.pairs) {
                        let left = plan
                            .members
                            .iter()
                            .position(|m| m.id == request.left)
                            .unwrap();
                        let right = plan
                            .members
                            .iter()
                            .position(|m| m.id == request.right)
                            .unwrap();
                        let residual = &row["embedded_residual"];
                        let residual_bound = if request.embedded_residual {
                            residual["status"] == "measured"
                                && residual["matrix"] == matrices[right].numerical_value()
                        } else {
                            residual["status"] == "not_requested"
                        };
                        let same_cutoff = metadata[left].same_cutoff(&metadata[right])?;
                        let root_rows = row["roots"]
                            .as_array()
                            .ok_or_else(|| anyhow::anyhow!("cohort cached root rows missing"))?;
                        ensure!(
                            row["left"] == left
                                && row["right"] == right
                                && row["same_source_replay"]
                                    == metadata[left].same_source(&metadata[right])
                                && (row["state"]["status"] == "incompatible") != same_cutoff
                                && residual_bound
                                && row["requested_root_count"] == request.root_ordinals.len()
                                && root_rows.len() == request.root_ordinals.len(),
                            "cohort cached pair binding mismatch"
                        );
                        coverage &= same_cutoff;
                        for (root, ordinal) in root_rows.iter().zip(&request.root_ordinals) {
                            ensure!(
                                root["ordinal"] == *ordinal
                                    && matches!(
                                        root["status"].as_str(),
                                        Some("point_comparison" | "unresolved")
                                    ),
                                "cohort cached root row binding mismatch"
                            );
                            coverage &= root["status"] == "point_comparison"
                                && root["left_status"] == "converged"
                                && root["right_status"] == "converged";
                        }
                    }
                    ensure!(
                        r.requested_coverage_complete == coverage,
                        "cohort cached coverage mismatch"
                    );
                    let entry = (identity.clone(), canonical_digest(r)?);
                    let verified = verified_reports()
                        .lock()
                        .map_err(|_| anyhow::anyhow!("cohort verification lock poisoned"))?
                        .contains(&entry);
                    if !verified {
                        calculate(Some(r))?;
                        remember_report(&identity, r)?;
                    }
                    Ok(())
                },
            );
            match result {
                Ok(r) => {
                    cache_status = if r.reused_manifest.is_some() {
                        "reused"
                    } else {
                        "computed"
                    }
                    .into();
                    artifact = r.produced_manifest.or(r.reused_manifest);
                    r.value.data
                }
                Err(e) => {
                    operations.failures.push(e.to_string());
                    incomplete()
                }
            }
        }
    };
    operations.peak_reserved_bytes = resources.usage().peak_bytes;
    operations.elapsed_seconds = started.elapsed().as_secs_f64();
    operations.report_bytes = super::capture_runtime::serialized_json_bytes(&measurements)?;
    Ok(CohortExecution {
        measurements,
        operations,
        artifact,
        cache_status,
    })
}

#[cfg(test)]
use super::published_source_fixtures;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn cohort_value_request_excludes_producer_manifest_but_binds_values_and_coverage() {
        let source = SourceIdentity {
            kind: "ccm_weil_eigenpair".into(),
            semantic_digest: ContentDigest::sha256(b"semantic"),
            payload_digest: ContentDigest::sha256(b"value"),
            canonical_manifest_digest: Some(ContentDigest::sha256(b"producer-one")),
        };
        let mut plan = FrozenCohortPlan {
            schema_version: 1,
            members: vec![
                CohortMemberRequest {
                    matrix: None,
                    id: "a".into(),
                    state: source.clone(),
                    roots: None,
                },
                CohortMemberRequest {
                    matrix: None,
                    id: "b".into(),
                    state: source,
                    roots: None,
                },
            ],
            pairs: vec![CohortPairRequest {
                left: "a".into(),
                right: "b".into(),
                root_ordinals: vec![],
                embedded_residual: false,
            }],
        };
        let original = plan.numerical_request();
        plan.members[0].state.canonical_manifest_digest =
            Some(ContentDigest::sha256(b"producer-two"));
        assert_eq!(plan.numerical_request(), original);
        plan.members[0].state.payload_digest = ContentDigest::sha256(b"changed-values");
        assert_ne!(plan.numerical_request(), original);
        let changed = plan.numerical_request();
        plan.pairs[0].root_ordinals.push(1);
        assert_ne!(plan.numerical_request(), changed);
    }

    #[test]
    fn explicit_cohort_published_tau_alias_authenticates_exact_ancestry_without_payloads() {
        fn manifest(kind: &str, parameters: Value) -> ArtifactManifest {
            let s = SemanticKeyEnvelope {
                schema_version: 1,
                artifact_kind: kind.into(),
                mathematical_semantics_version: "metadata-alias-fixture-v1".into(),
                resolved_mathematical_parameters: parameters,
                normalization: Some(
                    if kind == "ccm_tau_matrix" {
                        "symmetric_row_major"
                    } else {
                        "sum_xi_equals_sqrt_log_lambda_squared"
                    }
                    .into(),
                ),
                target: None,
                subspace: None,
                source_data_identities: BTreeMap::new(),
                algorithm_semantics: None,
            };
            let bytes = b"payload deliberately unavailable";
            let digest = ContentDigest::sha256(bytes);
            ArtifactManifest {
                schema_version: 1,
                key: ArtifactKey {
                    kind: kind.into(),
                    logical_key: format!("original/{kind}"),
                    parameters_digest: s.digest().unwrap(),
                },
                content_digest: digest.clone(),
                size_bytes: bytes.len() as u64,
                objects: vec![CacheObjectRef {
                    content_digest: digest,
                    size_bytes: bytes.len() as u64,
                }],
                created_unix_seconds: 1,
                producer_toolkit_version: ToolkitVersion::parse("0.17.0").unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Private,
                immutable: true,
                dependencies: vec![],
                tags: BTreeMap::from([(
                    SEMANTIC_KEY_MANIFEST_TAG.into(),
                    serde_json::to_string(&s).unwrap(),
                )]),
                provenance_digest: None,
            }
        }
        let tau = published_source_fixtures::published(
            manifest(
                "ccm_tau_matrix",
                json!({"lambda_squared":"13","n_modes":1,"precision_bits":128,"resolved_quadrature_orders":[8,8]}),
            ),
            &[],
        );
        let state = published_source_fixtures::published(
            manifest(
                "ccm_weil_eigenpair",
                json!({"lambda_squared":"13","n_modes":1,"precision_bits":128,"normalization":"sum_xi_equals_sqrt_log_lambda_squared","tau_source_identity":DependencyRef{key:tau.key.clone(),content_digest:tau.content_digest.clone(),required_quality:CacheQuality::Validated}}),
            ),
            &[&tau],
        );
        let request = CohortMemberRequest {
            matrix: None,
            id: "named".into(),
            state: SourceIdentity::from_manifest(&state).unwrap(),
            roots: None,
        };
        let mut alias = tau.clone();
        alias.key.logical_key = "closure/synthetic-adapter-key".into();
        assert_eq!(
            SourceIdentity::from_manifest(&tau).unwrap(),
            SourceIdentity::from_manifest(&alias).unwrap()
        );
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
        let original = metadata(&request, &[state.clone(), tau.clone()], &cache).unwrap();
        let adopted = metadata(&request, &[state.clone(), alias], &cache).unwrap();
        assert_eq!(original.policy, adopted.policy);
        assert_eq!(
            SourceIdentity::from_manifest(&original.tau).unwrap(),
            SourceIdentity::from_manifest(&adopted.tau).unwrap()
        );
        // Current value identity admits a different publication of exactly the
        // same matrix, without asking a resolver to open any numerical payload.
        let mut current = manifest(
            "ccm_weil_eigenpair",
            json!({
                "lambda_squared":"13", "n_modes":1, "precision_bits":128,
                "normalization":"sum_xi_equals_sqrt_log_lambda_squared",
                "tau_source_identity":xc_cache::numerical_source_identity(&tau, "tau", "stored-tau-matrix-v1")
            }),
        );
        current = published_source_fixtures::published(current, &[&tau]);
        let mut variant = tau.clone();
        variant.producer_toolkit_version = ToolkitVersion::parse("0.17.1").unwrap();
        variant = published_source_fixtures::published(variant, &[]);
        assert!(!xc_cache::manifest_depends_on(&current, &variant).unwrap());
        let request = CohortMemberRequest {
            matrix: None,
            id: "numerical".into(),
            state: SourceIdentity::from_manifest(&current).unwrap(),
            roots: None,
        };
        let original = metadata(&request, &[current.clone(), tau.clone()], &cache).unwrap();
        let adopted = metadata(&request, &[current.clone(), variant.clone()], &cache).unwrap();
        assert_eq!(original.policy, adopted.policy);

        // A combined inventory may contain another publication of the same
        // state. Only the exact frozen selection is an observation; unrelated
        // aliases must neither replace it nor prevent finding it.
        let mut other_state = current.clone();
        other_state.producer_toolkit_version = ToolkitVersion::parse("0.17.2").unwrap();
        other_state = published_source_fixtures::published(other_state, &[&tau]);
        assert_ne!(
            SourceIdentity::from_manifest(&other_state).unwrap(),
            request.state
        );
        for sources in [
            vec![other_state.clone(), current.clone(), variant.clone()],
            vec![current.clone(), other_state.clone(), variant.clone()],
        ] {
            assert_eq!(
                metadata(&request, &sources, &cache).unwrap().policy,
                original.policy
            );
        }
        assert!(metadata(&request, &[other_state, variant.clone()], &cache).is_err());

        // Reader-ineligible Tau aliases must not mask an admissible numerical
        // source. Without an admissible source the specific reason survives.
        let mut future = variant.clone();
        future.minimum_reader_version = ToolkitVersion::parse("999.0.0").unwrap();
        future = published_source_fixtures::published(future, &[]);
        let sources = [current.clone(), future.clone(), variant];
        assert_eq!(
            metadata(&request, &sources, &cache).unwrap().policy,
            original.policy
        );
        let error = metadata(&request, &[current, future], &cache)
            .err()
            .unwrap();
        assert!(error.to_string().contains("minimum reader"), "{error}");
    }
}
