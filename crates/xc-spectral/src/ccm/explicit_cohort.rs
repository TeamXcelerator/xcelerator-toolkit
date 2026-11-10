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

pub const SEMANTICS: &str = "ccm-explicit-retained-cohort-v1";
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
    let roots = unique(&["ccm_root_refinement", "ccm_root_discovery_window"])?;
    ensure!(states.len()==1 && roots.len()<=1,"explicit member selection requires one distinct state and at most one distinct root source");
    Ok(CohortMemberRequest {
        id: id.into(),
        state: states[0].clone(),
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
            for identity in std::iter::once(&m.state).chain(m.roots.iter()) {
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
        json!({"semantics":SEMANTICS,"members":self.members.iter().map(|m|json!({"state":m.state,"roots":m.roots})).collect::<Vec<_>>(),
            "pairs":self.pairs.iter().map(|p|json!({"left":index(&p.left),"right":index(&p.right),"root_ordinals":p.root_ordinals,"embedded_residual":p.embedded_residual})).collect::<Vec<_>>()})
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortMeasurements {
    pub semantics: String,
    pub scope: String,
    pub numerical_identity: ContentDigest,
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
    let _parse_reservation = RunResources::current()
        .map(|r| {
            let bytes = m.tags.values().try_fold(0u64, |n, s| {
                n.checked_add(s.len() as u64)
                    .ok_or_else(|| anyhow::anyhow!("semantic metadata size overflow"))
            })?;
            Ok::<_, anyhow::Error>(
                r.try_reserve_working(
                    bytes
                        .checked_mul(16)
                        .and_then(|n| n.checked_add(65536))
                        .ok_or_else(|| anyhow::anyhow!("semantic parse workspace overflow"))?,
                )?,
            )
        })
        .transpose()?;
    m.validate()?;
    ensure!(
        m.immutable && m.quality.satisfies(CacheQuality::Validated),
        "cohort requires immutable validated source metadata"
    );
    let s: SemanticKeyEnvelope =
        serde_json::from_str(m.tags.get(SEMANTIC_KEY_MANIFEST_TAG).ok_or_else(|| {
            anyhow::anyhow!("retained source lacks authenticated semantic envelope")
        })?)?;
    ensure!(
        s.artifact_kind == m.key.kind && s.digest()? == m.key.parameters_digest,
        "cohort source semantic identity mismatch"
    );
    retained_canonical_manifest(m)?;
    Ok(s)
}
fn select<'a>(
    id: &SourceIdentity,
    manifests: &'a [ArtifactManifest],
) -> Result<&'a ArtifactManifest> {
    let mut found = None;
    for m in manifests.iter().filter(|m| {
        m.key.kind == id.kind
            && m.key.parameters_digest == id.semantic_digest
            && m.content_digest == id.payload_digest
    }) {
        ensure!(
            SourceIdentity::from_manifest(m)? == *id,
            "canonical cohort source identity differs"
        );
        // Equal exact source replays are aliases, not independent observations.
        if found.is_none() {
            found = Some(m);
        }
    }
    found.ok_or_else(|| {
        anyhow::anyhow!(
            "explicit cohort source unavailable: {} {}",
            id.kind,
            id.semantic_digest.0
        )
    })
}
struct Metadata {
    state: ArtifactManifest,
    tau: ArtifactManifest,
    roots: Option<ArtifactManifest>,
    secular: Option<ArtifactManifest>,
    ancestry: Vec<ArtifactManifest>,
    policy: Value,
}
fn metadata(
    request: &CohortMemberRequest,
    manifests: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<Metadata> {
    let state = select(&request.state, manifests)?.clone();
    let state_semantic = semantic(&state)?;
    let mut queue = vec![(state.clone(), Vec::new())];
    let mut seen = BTreeSet::new();
    let mut chain = None;
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
        for next in resolve_manifest_sources(&m, manifests, cache)? {
            semantic(&next)?;
            let mut ancestry = path.clone();
            ancestry.push(next.clone());
            if next.key.kind == "ccm_tau_matrix" {
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
    let (tau, ancestry) = chain.ok_or_else(|| {
        anyhow::anyhow!("missing authenticated matrix/factorization/sector ancestry")
    })?;
    let tau_semantic = semantic(&tau)?;
    let a = &state_semantic.resolved_mathematical_parameters;
    let b = &tau_semantic.resolved_mathematical_parameters;
    for field in ["lambda_squared", "n_modes", "precision_bits"] {
        ensure!(
            !a[field].is_null() && a[field] == b[field],
            "matrix/state semantic {field} mismatch"
        );
    }
    if let Some(declared) = a.get("tau_source_identity") {
        let declared: DependencyRef = serde_json::from_value(declared.clone())?;
        ensure!(
            declared.content_digest == tau.content_digest
                && declared.key.kind == tau.key.kind
                && declared.key.parameters_digest == tau.key.parameters_digest
                && declared.required_quality == CacheQuality::Validated,
            "declared Tau identity differs from actual ancestry"
        );
    }
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
            let root = select(id, manifests)?.clone();
            let parents = resolve_manifest_sources(&root, manifests, cache)?;
            let secular = parents
                .iter()
                .find(|m| m.key.kind == "ccm_secular_source")
                .ok_or_else(|| anyhow::anyhow!("root secular ancestry missing"))?
                .clone();
            semantic(&secular)?;
            ensure!(
                manifest_depends_on(&secular, &state)?,
                "roots belong to another state"
            );
            (Some(root), Some(secular))
        }
        None => (None, None),
    };
    let policy = json!({"lambda_squared":a["lambda_squared"],"n_modes":a["n_modes"],"precision_bits":a["precision_bits"],
        "normalization":state_semantic.normalization,"state_subspace":state_semantic.subspace,
        "state_semantics":state_semantic.mathematical_semantics_version,"state_route":a.get("eigenstate_route").cloned().unwrap_or(json!("legacy_inverse_iteration")),
        "state_policy":a,"operator_semantics":tau_semantic.mathematical_semantics_version,"operator_algorithm":tau_semantic.algorithm_semantics,
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

fn authenticate_root_policy(
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
    // serde enum spelling in schema 5. Schema 6 fixed-guard payloads omit all
    // adaptive fields. Do not accept arbitrary aliases or infer a policy.
    match (
        schema,
        parameters
            .get("root_precision_policy")
            .and_then(Value::as_str),
    ) {
        (Some(5), Some("adaptive_v1")) => {
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
        (Some(3 | 6), None) => {
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
    // Includes temporary parse strings, retained vectors and roots, and scalar arithmetic.
    let bytes = m
        .state
        .size_bytes
        .checked_add(m.roots.as_ref().map_or(0, |m| m.size_bytes))
        .and_then(|n| n.checked_add(m.secular.as_ref().map_or(0, |m| m.size_bytes)))
        .ok_or_else(|| anyhow::anyhow!("source allocation overflow"))?;
    let reservation = resources.try_reserve_retained(
        bytes
            .checked_mul(24)
            .and_then(|n| n.checked_add(2 << 20))
            .ok_or_else(|| anyhow::anyhow!("source allocation overflow"))?,
    )?;
    let state_bytes = read(&m.state, cache, resources, scratch, operations)?;
    let state = RetainedState::from_payload(
        &m.state,
        &state_bytes,
        std::slice::from_ref(&m.state.content_digest),
    )?;
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
            authenticate_root_policy(root, &root_semantic, &root_raw)?;
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
fn bounds(x: &I) -> Result<Value> {
    x.validate()?;
    Ok(json!({"lower":exact(x.lower())?.to_string(),"upper":exact(x.upper())?.to_string()}))
}
fn delta(a: &Float, b: &Float) -> Result<Value> {
    let a = exact(a)?;
    let b = exact(b)?;
    let signed = Rational::from(&b - &a);
    let absolute = signed.clone().abs();
    Ok(
        json!({"signed":signed.to_string(),"absolute":absolute.to_string(),"relative_to_left":if a==0 {None}else{Some((&absolute/a.clone().abs()).to_string())},
        "relative_denominator_zero":a==0,"sign_crossing":(a<0 && b>0)||(a>0 && b<0)}),
    )
}
fn overlap(a: &RetainedState, b: &RetainedState, p: u32) -> Result<Value> {
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
        json!({"normalized_absolute_overlap":bounds(&bounded)?,"raw_overlap_arithmetic_enclosure":bounds(&absolute)?,
        "sign_aligned_distance_squared":bounds(&squared)?,"sign_aligned_distance":bounds(&squared.sqrt()?)?,
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
) -> Result<Value> {
    ensure!(
        left.state.modes < right.state.modes && left.state.cutoff == right.state.cutoff,
        "embedded residual requires smaller left state at equal cutoff"
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
        json!({"status":"measured","matrix":SourceIdentity::from_manifest(&meta.tau)?,"low_mode_residual_squared":bounds(&low)?,"high_mode_forcing_squared":bounds(&high)?,"total_residual_squared":bounds(&low.add(&high))?,
        "scope":"directed residual of exact stored larger point matrix and stored smaller state/eigenvalue; assembly error and analytic source uncertainty excluded",
        "state_error_bound":"unresolved: matching matrix separation and target-state identification not established",
        "low_mode_qualification":"includes changed assembly/quadrature and smaller state's residual; not pure truncation error"}),
    )
}

fn measure(
    plan: &FrozenCohortPlan,
    metadata: &[Metadata],
    identity: ContentDigest,
    cache: &ArtifactCacheContext<'_>,
    resources: &RunResources,
    scratch: &Path,
    operations: &mut CohortOperations,
) -> Result<CohortMeasurements> {
    // Admit conservative retained report storage before constructing rows. The
    // exponent term covers exact rational endpoint expansion, not just displays.
    let max_precision = metadata
        .iter()
        .filter_map(|m| m.policy["precision_bits"].as_u64())
        .max()
        .unwrap_or(128);
    ensure!(max_precision <= 1_000_000, "unsupported cohort precision");
    let rows = plan.pairs.iter().try_fold(0u64, |n, p| {
        n.checked_add(p.root_ordinals.len() as u64 + 1)
            .ok_or_else(|| anyhow::anyhow!("report row count overflow"))
    })?;
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
    let estimate = (max_precision / 8 + 32768)
        .checked_mul(64)
        .and_then(|n| n.checked_mul(rows))
        .and_then(|n| {
            metadata_bytes
                .checked_mul(16)
                .and_then(|m| n.checked_add(m))
        })
        .ok_or_else(|| anyhow::anyhow!("report workspace estimate overflow"))?;
    let _output_reservation = resources.try_reserve_retained(estimate)?;
    let mut members=metadata.iter().map(|m|Ok(json!({"state":SourceIdentity::from_manifest(&m.state)?,"matrix":SourceIdentity::from_manifest(&m.tau)?,"roots":m.roots.as_ref().map(SourceIdentity::from_manifest).transpose()?,"policy":m.policy,
        "ancestry":m.ancestry.iter().map(SourceIdentity::from_manifest).collect::<Result<Vec<_>>>()?,"matrix_scope":"authenticated metadata ancestry only; matrix payload unopened unless residual requested"}))).collect::<Result<Vec<_>>>()?;
    let mut pairs = Vec::new();
    let mut coverage = true;
    for pair in &plan.pairs {
        resources.check_cancelled()?;
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
        let same_cutoff = xc_core::DecimalLiteral::new(&a.state.cutoff)?
            .cmp_numeric(&xc_core::DecimalLiteral::new(&b.state.cutoff)?)?
            == std::cmp::Ordering::Equal;
        if !same_cutoff {
            coverage = false;
        }
        let state = if same_cutoff {
            overlap(&a.state, &b.state, p)?
        } else {
            json!({"status":"incompatible","reason":"different cutoff: zero padding is not a common physical Fourier basis"})
        };
        let mut root_rows = Vec::new();
        for ordinal in &pair.root_ordinals {
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
            let row = if let Some((x, y)) = values {
                json!({"ordinal":ordinal,"status":"point_comparison","left_status":left.as_ref().unwrap().source_status,"right_status":right.as_ref().unwrap().source_status,"change":delta(&retained::scalar(x,a.state.precision)?,&retained::scalar(y,b.state.precision)?)?,"certified_difference_enclosure":null})
            } else {
                coverage = false;
                json!({"ordinal":ordinal,"status":"unresolved","left_status":left.map(|r|r.source_status),"right_status":right.map(|r|r.source_status),"reason":"requested retained root absent or failed; no silent window intersection"})
            };
            root_rows.push(row);
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
        pairs.push(json!({"left":ai,"right":bi,"same_source_replay":plan.members[ai].state==plan.members[bi].state,
            "changed_fields":differences,"comparison_class":if !same_cutoff {"parameter_change"}else if a.state.precision!=b.state.precision && !q_same {"coupled_precision_quadrature_change"}else{"finite_configuration_comparison"},
            "eigenvalue_change":delta(&retained::scalar(&a.state.eigenvalue,a.state.precision)?,&retained::scalar(&b.state.eigenvalue,b.state.precision)?)?,
            "eigenvalue_qualification":"stored point changes only; retained ground/precision-floor evidence is inherited, not newly certified",
            "state":state,"roots":root_rows,"requested_root_count":pair.root_ordinals.len(),"ordinal_scope":"fixed retained acquisition ordinals; seeded and independent routes retained; no global kth-root or branch certification",
            "root_enclosure_scope":"stored point inputs only; converged status is not an interval certificate","embedded_residual":embedded,
            "numerical_acceptance":"not_requested; no tolerance, extrapolation, improving-chain, or convergence predicate"}));
    }
    Ok(CohortMeasurements {
        semantics: SEMANTICS.into(),
        scope: SCOPE.into(),
        numerical_identity: identity,
        execution_complete: true,
        requested_coverage_complete: coverage,
        members,
        pairs,
    })
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
    let request = plan.numerical_request();
    let identity = ContentDigest::sha256(&serde_json::to_vec(&request)?);
    let incomplete = || {
        CohortMeasurements {semantics:SEMANTICS.into(),scope:SCOPE.into(),numerical_identity:identity.clone(),execution_complete:false,requested_coverage_complete:false,
        members:plan.members.iter().map(|m|json!({"state":m.state,"roots":m.roots,"status":"unavailable_or_not_evaluated"})).collect(),
        pairs:plan.pairs.iter().map(|p|json!({"left":plan.members.iter().position(|m|m.id==p.left).unwrap(),"right":plan.members.iter().position(|m|m.id==p.right).unwrap(),"status":"unresolved","requested_root_ordinals":p.root_ordinals,"embedded_residual_requested":p.embedded_residual})).collect()}
    };
    let metadata = plan
        .members
        .iter()
        .map(|m| metadata(m, manifests, cache))
        .collect::<Result<Vec<_>>>();
    let mut artifact = None;
    let mut cache_status = "incomplete_not_persisted".to_string();
    let measurements = match metadata {
        Err(e) => {
            operations.failures.push(e.to_string());
            incomplete()
        }
        Ok(metadata) => {
            let parents = parents(&metadata);
            let result = retained::managed(
                KIND,
                request,
                &parents,
                cache,
                || {
                    measure(
                        plan,
                        &metadata,
                        identity.clone(),
                        cache,
                        resources,
                        scratch_directory,
                        &mut operations,
                    )
                },
                |r: &CohortMeasurements| {
                    ensure!(
                        r.semantics == SEMANTICS
                            && r.numerical_identity == identity
                            && r.execution_complete
                            && r.scope == SCOPE
                            && r.members.len() == plan.members.len()
                            && r.pairs.len() == plan.pairs.len(),
                        "cohort cached report binding mismatch"
                    );
                    for (member, request) in r.members.iter().zip(&plan.members) {
                        ensure!(
                            member["state"] == serde_json::to_value(&request.state)?
                                && member["roots"] == serde_json::to_value(&request.roots)?,
                            "cohort cached member binding mismatch"
                        );
                    }
                    for (row, request) in r.pairs.iter().zip(&plan.pairs) {
                        ensure!(
                            row["left"]
                                == plan
                                    .members
                                    .iter()
                                    .position(|m| m.id == request.left)
                                    .unwrap()
                                && row["right"]
                                    == plan
                                        .members
                                        .iter()
                                        .position(|m| m.id == request.right)
                                        .unwrap()
                                && row["roots"].as_array().is_some_and(|rows| rows.len()
                                    == request.root_ordinals.len()
                                    && rows
                                        .iter()
                                        .zip(&request.root_ordinals)
                                        .all(|(r, k)| r["ordinal"] == *k)),
                            "cohort cached pair binding mismatch"
                        );
                    }
                    Ok(())
                },
            );
            match result {
                Ok(r) => {
                    artifact = r.produced_manifest.or(r.reused_manifest);
                    cache_status = if operations.decoded_payload_bytes_opened == 0 {
                        "reused"
                    } else {
                        "computed"
                    }
                    .into();
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
    operations.report_bytes = serde_json::to_vec(&measurements)?.len() as u64;
    Ok(CohortExecution {
        measurements,
        operations,
        artifact,
        cache_status,
    })
}

#[cfg(test)]
#[path = "../../tests/common/published_sources.rs"]
mod published_source_fixtures;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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
        let original = metadata(&request, &[state.clone(), tau], &cache).unwrap();
        let adopted = metadata(&request, &[state, alias], &cache).unwrap();
        assert_eq!(original.policy, adopted.policy);
        assert_eq!(
            SourceIdentity::from_manifest(&original.tau).unwrap(),
            SourceIdentity::from_manifest(&adopted.tau).unwrap()
        );
    }
}
