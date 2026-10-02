//! Managed finite-state research observations. Inputs are explicit, numerical
//! scope travels with each report, and no primary solve is available here.
use super::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::Float;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use xc_cache::*;
use xc_numerics::prefix::lossless_decimal as dec;

#[path = "retained_evidence/energy.rs"]
mod energy;
#[path = "retained_evidence/finite_math.rs"]
pub(super) mod finite_math;
#[path = "retained_evidence/point.rs"]
pub(super) mod point;
#[path = "retained_evidence/projection.rs"]
mod projection;
#[path = "retained_evidence/projection_math.rs"]
mod projection_math;
#[path = "retained_evidence/transform.rs"]
mod transform;
#[path = "retained_evidence/transform_math.rs"]
pub(super) mod transform_math;
pub const SEMANTICS: &str = "ccm-retained-research-observations-v22";
const GUARDED_RANGE_POINT_SEMANTICS: &str = "ccm-retained-research-observations-v21";
const SIGNED_BAND_POINT_SEMANTICS: &str = "ccm-retained-research-observations-v20";
const ROOT_ADAPTER_POINT_SEMANTICS: &str = "ccm-retained-research-observations-v19";
const TAIL_MODEL_POINT_SEMANTICS: &str = "ccm-retained-research-observations-v18";
const COMPLEX_TRANSFORM_INTERVAL_SEMANTICS: &str = "ccm-retained-research-observations-v17";
const OPERATOR_CLUSTER_INTERVAL_SEMANTICS: &str = "ccm-retained-research-observations-v16";
const TRANSFER_INTERVAL_SEMANTICS: &str = "ccm-retained-research-observations-v15";
const OBSERVATION_INTERVAL_SEMANTICS: &str = "ccm-retained-research-observations-v14";
const TRANSFORM_INTERVAL_SEMANTICS: &str = "ccm-retained-research-observations-v13";
const ALLOWANCE_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v12";
const DIRECTIONAL_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v11";
const ENERGY_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v10";
const SIGNED_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v9";
const PROJECTION_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v8";
const WEIGHTED_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v7";
const CLUSTER_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v6";
const ATOM_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v5";
const COMPACTNESS_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v4";
const LATEST_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v3";
const PREVIOUS_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v2";
const LEGACY_DEFINITION_SEMANTICS: &str = "ccm-retained-research-observations-v1";
const SCOPE: &str = "finite_point_inputs; no_source_error_enclosure; no_ground_selection_or_convergence_certificate";
fn equal_cutoff(a: &str, b: &str) -> Result<bool> {
    Ok(
        xc_core::DecimalLiteral::new(a)?.cmp_numeric(&xc_core::DecimalLiteral::new(b)?)?
            == std::cmp::Ordering::Equal,
    )
}
fn canonical_decimal(value: &str) -> Result<String> {
    Ok(xc_core::DecimalLiteral::new(value)?
        .canonical()?
        .as_str()
        .to_owned())
}
pub(super) fn scalar(s: &str, p: u32) -> Result<Float> {
    // Public request/source precision is checked separately; internal arithmetic
    // may add up to 128 guard bits without changing the input precision contract.
    if !(64..=1_000_128).contains(&p) {
        bail!("unsupported scalar precision");
    }
    if s.len() > 1_048_576 {
        bail!("research scalar exceeds the one MiB input budget");
    }
    let decimal = xc_core::DecimalLiteral::new(s)?;
    let v = Float::with_val(p, Float::parse(s)?);
    if !v.is_finite()
        || (v.is_zero() && decimal.canonical()?.as_str() != "0")
        // MPFR's smallest exponent has no subnormal significands. A nonzero
        // nearest result can conceal severe relative loss below that floor.
        || (v.get_exp() == Some(rug::float::exp_min())
            && Float::with_val_round(p, Float::parse(s)?, rug::float::Round::Zero)
                .0
                .is_zero())
    {
        bail!("research scalar is outside the finite exponent range");
    }
    Ok(v)
}
pub(super) fn precision(p: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&p) {
        bail!("unsupported research precision");
    }
    Ok(())
}
fn dep(m: &ArtifactManifest) -> DependencyRef {
    DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    }
}
fn admits(m: &ArtifactManifest, b: &[u8], allowed: &[ContentDigest], kind: &str) -> Result<()> {
    m.validate()?;
    if m.key.kind != kind
        || !m.immutable
        || m.quality.admissible_rank() < CacheQuality::Validated.admissible_rank()
        || m.size_bytes != b.len() as u64
        || ContentDigest::sha256(b) != m.content_digest
        || !allowed.contains(&m.content_digest)
    {
        bail!("unapproved or invalid retained research source");
    }
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchRecord<T> {
    pub schema_version: u32,
    pub semantics: String,
    pub kind: String,
    pub scope: String,
    pub source_dependencies: Vec<DependencyRef>,
    pub request: Value,
    pub data: T,
}
pub(super) fn managed<T, F, V>(
    kind: &str,
    request: Value,
    sources: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
    compute: F,
    validate: V,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<T>>>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> Result<T>,
    V: Fn(&T) -> Result<()>,
{
    if cache.requested_assurance != xc_core::AssuranceLevel::Computed {
        bail!("research observations are computed, not certificates");
    }
    for s in sources {
        s.validate()?;
        if s.quality.admissible_rank() < CacheQuality::Validated.admissible_rank() {
            bail!("inadmissible research parent quality");
        }
    }
    let scope = if kind == "ccm_transform_enclosure" {
        "finite_retained_function_enclosures; source_scope_explicit; no_infinite_limit_claim"
    } else {
        SCOPE
    };
    let public = sources
        .iter()
        .all(|m| m.visibility == CacheVisibility::Public);
    if cache.write_visibility == CacheVisibility::Public
        && (!public || artifact_kind_is_private_only(kind))
    {
        bail!("research observation requires independently public-admissible inputs");
    }
    let mut dependencies = sources.iter().map(dep).collect::<Vec<_>>();
    dependencies.sort_by(|a, b| {
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
    dependencies.dedup();
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: SEMANTICS.into(),
        resolved_mathematical_parameters: json!({"request":request,"source_dependencies":dependencies,"source_parents_are_public":public}),
        normalization: None,
        target: Some(kind.into()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(SEMANTICS.into()),
    };
    let key = ContentDigest::sha256(&serde_json::to_vec(&semantic)?);
    let logical = format!("ccm/research/{kind}/{}", key.0);
    let req = ArtifactExecutionCacheRequest {
        operation: "ccm.research.retained",
        semantic_key: &semantic,
        logical_key: &logical,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Validated,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse(xc_cache::CLEAN_SLATE)?,
        maximum_reader_version: None,
        tags: BTreeMap::from([("assurance".into(), "computed_not_certified".into())]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    Ok(resolve_or_compute_json_artifact_with_dependencies(
        &req,
        || {
            let data = compute().map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            Ok((
                ResearchRecord {
                    schema_version: 1,
                    semantics: SEMANTICS.into(),
                    kind: kind.into(),
                    scope: scope.into(),
                    source_dependencies: dependencies.clone(),
                    request: request.clone(),
                    data,
                },
                dependencies.clone(),
            ))
        },
        |r| {
            if r.schema_version != 1
                || r.semantics != SEMANTICS
                || r.kind != kind
                || r.scope != scope
                || r.source_dependencies != dependencies
                || r.request != request
            {
                return Err(CacheError::InvalidManifest(
                    "retained research identity mismatch".into(),
                ));
            }
            validate(&r.data).map_err(|e| CacheError::InvalidManifest(e.to_string()))
        },
    )?)
}
/// Explicit finite Fourier reference. This is not an automatically certified
/// prolate function or a reader of the legacy private target specification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSpec {
    pub schema_version: u32,
    pub definition: String,
    pub lambda_squared: String,
    pub precision_bits: u32,
    pub coefficients: Vec<String>,
    pub approximation_scope: String,
}
impl ReferenceSpec {
    fn validate(&self) -> Result<()> {
        precision(self.precision_bits)?;
        if self.schema_version != 1
            || self.definition.trim().is_empty()
            || self.definition.len() > 16384
            || self.approximation_scope.trim().is_empty()
            || self.approximation_scope.len() > 16384
            || self.coefficients.is_empty()
            || self.coefficients.len() > 16385
            || self.coefficients.len().is_multiple_of(2)
            || scalar(&self.lambda_squared, self.precision_bits)? <= 1
        {
            bail!("invalid finite reference definition");
        }
        let v = self.values(self.precision_bits)?;
        if v.iter().all(|x| x == &0) {
            bail!("zero reference");
        }
        xc_core::validate_secret_free(self, "reference definition")?;
        Ok(())
    }
    pub(super) fn values(&self, p: u32) -> Result<Vec<Float>> {
        if p < self.precision_bits {
            bail!("reference coefficient precision reduction");
        }
        self.coefficients
            .iter()
            .map(|s| Ok(Float::with_val(p, scalar(s, self.precision_bits)?)))
            .collect()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceData {
    pub spec: ReferenceSpec,
    pub definition_digest: ContentDigest,
    pub basis: String,
    pub certification: String,
}
pub fn capture_reference(
    spec: &ReferenceSpec,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ReferenceData>>> {
    spec.validate()?;
    let digest = ContentDigest::sha256(&serde_json::to_vec(spec)?);
    managed(
        "ccm_reference_source",
        serde_json::to_value(spec)?,
        &[],
        cache,
        || {
            Ok(ReferenceData {
                spec: spec.clone(),
                definition_digest: digest.clone(),
                basis: "centered_full_V_fourier; coefficient_points; zero_extension".into(),
                certification: "not_certified; approximation_error_not_enclosed".into(),
            })
        },
        |r| {
            r.spec.validate()?;
            if r.spec != *spec
                || r.definition_digest != digest
                || r.basis != "centered_full_V_fourier; coefficient_points; zero_extension"
                || r.certification != "not_certified; approximation_error_not_enclosed"
            {
                bail!("reference source mismatch");
            }
            Ok(())
        },
    )
}
#[derive(Clone, Debug)]
/// Authenticated reference definition. Modify a separate spec and recapture it
/// to obtain a different definition; admitted source identities are immutable.
/// ```compile_fail
/// use xc_spectral::ccm::retained_evidence::RetainedReference;
/// fn alter(source: &mut RetainedReference) { source.spec.coefficients.clear(); }
/// ```
pub struct RetainedReference {
    manifest: ArtifactManifest,
    spec: ReferenceSpec,
}
impl RetainedReference {
    pub fn spec(&self) -> &ReferenceSpec {
        &self.spec
    }
    pub fn from_payload(m: &ArtifactManifest, b: &[u8], allowed: &[ContentDigest]) -> Result<Self> {
        admits(m, b, allowed, "ccm_reference_source")?;
        let r: ResearchRecord<ReferenceData> = serde_json::from_slice(b)?;
        r.data.spec.validate()?;
        if r.schema_version != 1
            || ![
                SEMANTICS,
                GUARDED_RANGE_POINT_SEMANTICS,
                LEGACY_DEFINITION_SEMANTICS,
                PREVIOUS_DEFINITION_SEMANTICS,
                LATEST_DEFINITION_SEMANTICS,
                COMPACTNESS_DEFINITION_SEMANTICS,
                ATOM_DEFINITION_SEMANTICS,
                CLUSTER_DEFINITION_SEMANTICS,
                WEIGHTED_DEFINITION_SEMANTICS,
                PROJECTION_DEFINITION_SEMANTICS,
                SIGNED_DEFINITION_SEMANTICS,
                ENERGY_DEFINITION_SEMANTICS,
                DIRECTIONAL_DEFINITION_SEMANTICS,
                ALLOWANCE_DEFINITION_SEMANTICS,
                TRANSFORM_INTERVAL_SEMANTICS,
                OBSERVATION_INTERVAL_SEMANTICS,
                TRANSFER_INTERVAL_SEMANTICS,
                SIGNED_BAND_POINT_SEMANTICS,
                ROOT_ADAPTER_POINT_SEMANTICS,
                TAIL_MODEL_POINT_SEMANTICS,
                COMPLEX_TRANSFORM_INTERVAL_SEMANTICS,
                OPERATOR_CLUSTER_INTERVAL_SEMANTICS,
            ]
            .contains(&r.semantics.as_str())
            || r.kind != "ccm_reference_source"
            || r.scope != SCOPE
            || !r.source_dependencies.is_empty()
            || !m.dependencies.is_empty()
            || r.request != serde_json::to_value(&r.data.spec)?
            || r.data.definition_digest != ContentDigest::sha256(&serde_json::to_vec(&r.data.spec)?)
            || r.data.basis != "centered_full_V_fourier; coefficient_points; zero_extension"
            || r.data.certification != "not_certified; approximation_error_not_enclosed"
        {
            bail!("invalid reference record");
        }
        Ok(Self {
            manifest: m.clone(),
            spec: r.data.spec,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPoint {
    pub ordinal: usize,
    pub value: Option<String>,
    pub source_status: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSpec {
    pub schema_version: u32,
    pub role: String,
    pub attribution: String,
    pub coordinate: String,
    pub precision_bits: u32,
    pub points: Vec<EvaluationPoint>,
}
impl DatasetSpec {
    fn validate(&self) -> Result<()> {
        precision(self.precision_bits)?;
        if self.schema_version != 1
            || self.coordinate != "mellin_t"
            || ![
                "known_zeta_ordinates",
                "evaluation_points",
                "retained_ccm_roots",
            ]
            .contains(&self.role.as_str())
            || self.attribution.trim().is_empty()
            || self.attribution.len() > 16384
            || self.points.len() > 100000
            || self.points.windows(2).any(|w| w[0].ordinal >= w[1].ordinal)
        {
            bail!("invalid evaluation dataset");
        }
        for row in &self.points {
            if row.ordinal == 0
                || row.source_status.trim().is_empty()
                || row.source_status.len() > 512
            {
                bail!("invalid evaluation row");
            }
            if let Some(v) = &row.value {
                scalar(v, self.precision_bits)?;
            }
        }
        xc_core::validate_secret_free(self, "evaluation dataset")?;
        Ok(())
    }
}
pub fn capture_dataset(
    spec: &DatasetSpec,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<DatasetSpec>>> {
    spec.validate()?;
    if spec.role == "retained_ccm_roots" {
        bail!("retained CCM roots require authenticated root-window input");
    }
    managed(
        "research_reference_dataset",
        serde_json::to_value(spec)?,
        &[],
        cache,
        || Ok(spec.clone()),
        |v| {
            v.validate()?;
            if v != spec {
                bail!("dataset mismatch");
            }
            Ok(())
        },
    )
}
#[derive(Clone, Debug)]
/// Immutable authenticated evaluation points.
/// ```compile_fail
/// use xc_spectral::ccm::retained_evidence::RetainedDataset;
/// fn alter(source: &mut RetainedDataset) { source.spec.points.clear(); }
/// ```
pub struct RetainedDataset {
    pub(crate) manifest: ArtifactManifest,
    spec: DatasetSpec,
}
impl RetainedDataset {
    pub fn spec(&self) -> &DatasetSpec {
        &self.spec
    }
    pub fn from_payload(m: &ArtifactManifest, b: &[u8], allowed: &[ContentDigest]) -> Result<Self> {
        admits(m, b, allowed, "research_reference_dataset")?;
        let r: ResearchRecord<DatasetSpec> = serde_json::from_slice(b)?;
        r.data.validate()?;
        if r.schema_version != 1
            || ![
                SEMANTICS,
                GUARDED_RANGE_POINT_SEMANTICS,
                LEGACY_DEFINITION_SEMANTICS,
                PREVIOUS_DEFINITION_SEMANTICS,
                LATEST_DEFINITION_SEMANTICS,
                COMPACTNESS_DEFINITION_SEMANTICS,
                ATOM_DEFINITION_SEMANTICS,
                CLUSTER_DEFINITION_SEMANTICS,
                WEIGHTED_DEFINITION_SEMANTICS,
                PROJECTION_DEFINITION_SEMANTICS,
                SIGNED_DEFINITION_SEMANTICS,
                ENERGY_DEFINITION_SEMANTICS,
                DIRECTIONAL_DEFINITION_SEMANTICS,
                ALLOWANCE_DEFINITION_SEMANTICS,
                TRANSFORM_INTERVAL_SEMANTICS,
                OBSERVATION_INTERVAL_SEMANTICS,
                TRANSFER_INTERVAL_SEMANTICS,
                SIGNED_BAND_POINT_SEMANTICS,
                ROOT_ADAPTER_POINT_SEMANTICS,
                TAIL_MODEL_POINT_SEMANTICS,
                COMPLEX_TRANSFORM_INTERVAL_SEMANTICS,
                OPERATOR_CLUSTER_INTERVAL_SEMANTICS,
            ]
            .contains(&r.semantics.as_str())
            || r.kind != "research_reference_dataset"
            || r.scope != SCOPE
            || r.request != serde_json::to_value(&r.data)?
            || !r.source_dependencies.is_empty()
            || !m.dependencies.is_empty()
            || r.data.role == "retained_ccm_roots"
        {
            bail!("invalid explicit dataset record");
        }
        Ok(Self {
            manifest: m.clone(),
            spec: r.data,
        })
    }
}
/// Retained roots include every original outcome, including failed rows.
#[derive(Clone, Debug)]
pub struct RetainedRoots {
    pub(crate) manifest: ArtifactManifest,
    pub(crate) secular_manifest: ArtifactManifest,
    pub(crate) dataset: DatasetSpec,
    pub(crate) completeness: String,
    pub(crate) acquisition: Value,
    pub(crate) cutoff: String,
    pub(crate) modes: usize,
}
impl RetainedRoots {
    pub fn from_payload(
        m: &ArtifactManifest,
        b: &[u8],
        secular: &ArtifactManifest,
        secular_bytes: &[u8],
        state: &RetainedState,
        allowed: &[ContentDigest],
    ) -> Result<Self> {
        if !["ccm_root_discovery_window", "ccm_root_refinement"].contains(&m.key.kind.as_str()) {
            bail!("unsupported root source kind");
        }
        admits(m, b, allowed, &m.key.kind)?;
        admits(secular, secular_bytes, allowed, "ccm_secular_source")?;
        let r: Value = serde_json::from_slice(b)?;
        let sec: Value = serde_json::from_slice(secular_bytes)?;
        let require = |v: &Value, name: &str| -> Result<String> {
            Ok(v.get(name)
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing {name}"))?
                .to_owned())
        };
        if !xc_cache::manifest_depends_on(m, secular)?
            || !xc_cache::manifest_depends_on(secular, &state.manifest)?
            || sec["eigenpair_content_digest"].as_str() != Some(&state.manifest.content_digest.0)
            || r["lambda_squared"].as_str() != Some(&state.cutoff)
            || sec["lambda_squared"] != r["lambda_squared"]
            || r["n_modes"].as_u64() != Some(state.modes as u64)
            || sec["n_modes"] != r["n_modes"]
            || r["precision_bits"].as_u64() != Some(state.precision as u64)
            || sec["precision_bits"] != r["precision_bits"]
        {
            bail!("root/state dependency closure mismatch");
        }
        if !matches!(r["schema_version"].as_u64(), Some(3 | 5 | 6))
            || ((r["schema_version"].as_u64() == Some(6)
                || r.get("secular_source_content_digest").is_some())
                && r["secular_source_content_digest"].as_str() != Some(&secular.content_digest.0))
            || sec["schema_version"].as_u64() != Some(1)
            || sec["normalization"].as_str() != Some("sum_xi_equals_sqrt_log_lambda_squared")
            || r["force_even"] != sec["force_even"]
            || r["parity_policy"] != sec["parity_policy"]
            || r["discovery_mode"].as_str().is_none()
            || r["reference_seeds_used"].as_bool().is_none()
        {
            bail!("unsupported root schema or acquisition policy");
        }
        let acquisition = json!({"source_schema":r["schema_version"],"discovery_mode":r["discovery_mode"],"reference_seeds_used":r["reference_seeds_used"],"reference_dataset":r["reference_dataset"],"root_domain":r.get("root_domain").cloned().unwrap_or(json!("positive")),"force_even":r["force_even"],"parity_policy":r["parity_policy"],"root_precision_policy":r["root_precision_policy"]});
        // Adaptive solver metadata describes work and accuracy, never storage.
        // hp::adaptive_root_outcome downrounds every numeric value to this precision.
        let root_precision = state.precision;
        let first = r["first_root_index"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| anyhow::anyhow!("missing root ordinal"))?;
        let rows = r["outcomes"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing root outcomes"))?;
        if rows.len() > 100000 {
            bail!("root row budget exceeded");
        }
        let mut points = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            let status = require(row, "status")?;
            if !["converged", "stagnated", "approximate", "failed"].contains(&status.as_str()) {
                bail!("unsupported root status");
            }
            let value = if status == "failed" {
                None
            } else {
                Some(require(&row["details"], "value")?)
            };
            for name in [
                "target_precision_bits",
                "evaluation_precision_bits",
                "verification_precision_bits",
            ] {
                if let Some(v) = row["details"]["adaptive_precision"].get(name) {
                    let bits = v
                        .as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| anyhow::anyhow!("invalid root precision"))?;
                    if name == "target_precision_bits" {
                        if !(1..=1_000_000).contains(&bits) {
                            bail!("unsupported root accuracy target");
                        }
                    } else if bits < 64 {
                        // Metadata only: the producer may work above the retained
                        // input precision cap. No allocation uses these counts.
                        bail!("unsupported root working precision metadata");
                    }
                }
            }
            points.push(EvaluationPoint {
                ordinal: first
                    .checked_add(i)
                    .ok_or_else(|| anyhow::anyhow!("ordinal overflow"))?,
                value,
                source_status: status,
            });
        }
        let dataset = DatasetSpec {
            schema_version: 1,
            role: "retained_ccm_roots".into(),
            attribution: m.content_digest.0.clone(),
            coordinate: "mellin_t".into(),
            precision_bits: root_precision,
            points,
        };
        dataset.validate()?;
        Ok(Self {
            manifest: m.clone(),
            secular_manifest: secular.clone(),
            dataset,
            completeness: require(&r, "completeness")?,
            acquisition,
            cutoff: state.cutoff.clone(),
            modes: state.modes,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RootWindowData {
    pub lambda_squared: String,
    pub n_modes: usize,
    pub source_completeness: String,
    pub source_acquisition: Value,
    pub ordinal_scope: String,
    pub points: Vec<EvaluationPoint>,
    pub positive_count: usize,
    pub nonpositive_count: usize,
    pub missing_count: usize,
    pub minimum_positive_spacing: Option<String>,
    pub window_inverse_moments: Vec<String>,
    pub omitted_tail: String,
}
pub fn capture_root_window(
    roots: &RetainedRoots,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<RootWindowData>>> {
    let spec = &roots.dataset;
    let p = spec.precision_bits;
    managed(
        "ccm_root_band_analysis",
        json!({"rule":"entire_retained_window; no_equilibrium_band_classification","precision_bits":p}),
        &[roots.manifest.clone(), roots.secular_manifest.clone()],
        cache,
        || root_window_data(roots),
        |r| validate_root_window(r, roots),
    )
}

fn root_window_data(roots: &RetainedRoots) -> Result<RootWindowData> {
    let spec = &roots.dataset;
    let p = spec.precision_bits;
    let mut positive = Vec::new();
    let mut nonpositive = 0;
    let mut missing = 0;
    for row in &spec.points {
        match &row.value {
            Some(v) => {
                let v = Float::with_val(p + 64, scalar(v, p)?);
                if v > 0 {
                    positive.push(v);
                } else {
                    nonpositive += 1;
                }
            }
            None => missing += 1,
        }
    }
    let work = p + 64;
    let one = Float::with_val(work, 1);
    let mut terms = (0..3)
        .map(|_| Vec::with_capacity(positive.len()))
        .collect::<Vec<_>>();
    for x in &positive {
        let inverse = point::quotient(&one, x, work)?;
        terms[0].push(inverse.clone());
        terms[1].push(point::product(&[&inverse, &inverse], work)?);
        terms[2].push(point::product(&[&inverse, &inverse, &inverse], work)?);
    }
    let sums = terms
        .iter()
        .map(|v| point::output(&point::sum(v, work)?, p))
        .collect::<Result<Vec<_>>>()?;
    positive.sort_by(|a, b| a.total_cmp(b));
    let gaps = positive
        .windows(2)
        .map(|w| point::output(&point::sum(&[w[1].clone(), -w[0].clone()], work)?, p))
        .collect::<Result<Vec<_>>>()?;
    let gap = gaps.into_iter().min_by(|a, b| a.total_cmp(b));
    Ok(RootWindowData {
        lambda_squared: roots.cutoff.clone(),
        n_modes: roots.modes,
        source_completeness: roots.completeness.clone(),
        source_acquisition: roots.acquisition.clone(),
        ordinal_scope: "preserved_source_window_ordinals; not_zeta_identification".into(),
        points: spec.points.clone(),
        positive_count: positive.len(),
        nonpositive_count: nonpositive,
        missing_count: missing,
        minimum_positive_spacing: gap.as_ref().map(dec),
        window_inverse_moments: sums.iter().map(dec).collect(),
        omitted_tail: "not_bounded; window_moments_are_not_infinite_band_moments".into(),
    })
}

fn validate_root_window(r: &RootWindowData, roots: &RetainedRoots) -> Result<()> {
    // Replay cheap finite statistics from the admitted source points. A matching
    // count total or recognized status alone does not authenticate a conclusion.
    let expected = root_window_data(roots)?;
    if r != &expected {
        bail!("root_window summary disagrees with source replay");
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TransformOptions {
    pub working_precision_bits: u32,
    pub maximum_rows: usize,
    #[serde(default = "default_transform_byte_budget")]
    pub maximum_estimated_output_bytes: u64,
}
fn default_transform_byte_budget() -> u64 {
    256 * 1024 * 1024
}
impl TransformOptions {
    pub fn for_source(s: &RetainedState) -> Self {
        Self {
            working_precision_bits: s.precision.saturating_add(64).min(1_000_000),
            maximum_rows: 100000,
            maximum_estimated_output_bytes: default_transform_byte_budget(),
        }
    }
}
impl TransformOptions {
    pub fn for_roots(s: &RetainedState, r: &RetainedRoots) -> Self {
        Self {
            working_precision_bits: s
                .precision
                .max(r.dataset.precision_bits)
                .saturating_add(64)
                .min(1_000_000),
            ..Self::for_source(s)
        }
    }
    pub fn for_dataset(s: &RetainedState, d: &RetainedDataset) -> Self {
        Self {
            working_precision_bits: s
                .precision
                .max(d.spec.precision_bits)
                .saturating_add(64)
                .min(1_000_000),
            ..Self::for_source(s)
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformRow {
    pub ordinal: usize,
    pub source_status: String,
    pub t: Option<String>,
    pub outcome: String,
    pub value: Option<String>,
    pub derivative: Option<String>,
    pub sum_absolute_terms: Option<String>,
    pub sum_absolute_derivative_terms: Option<String>,
    pub cancellation_digits: Option<String>,
    pub newton_correction: Option<String>,
    pub newton_remainder: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformData {
    pub coordinate: String,
    pub normalization: String,
    pub input_role: String,
    pub source_precision_bits: u32,
    pub working_precision_bits: u32,
    pub finite_support: String,
    pub physical_tail: String,
    pub second_derivative_cauchy_schwarz_expression: String,
    pub rows: Vec<TransformRow>,
}
pub(super) fn orientation(x: &[Float], p: u32) -> i32 {
    point::orientation(x, p)
}
pub(super) fn norm2(x: &[Float], p: u32) -> Result<Float> {
    point::dot(x, x, p)
}
pub(super) fn transform_terms(
    state: &RetainedState,
    t: &Float,
    p: u32,
) -> Result<(Float, Float, Float, Float)> {
    transform::terms(state, t, p)
}
pub(super) fn transform_terms_at_root(
    state: &RetainedState,
    t: &Float,
    p: u32,
) -> Result<(Float, Float, Float, Float)> {
    transform::terms_at_root(state, t, p)
}
fn transforms(
    state: &RetainedState,
    dataset: &DatasetSpec,
    o: &TransformOptions,
    extra: &[ArtifactManifest],
    root_convention: bool,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<TransformData>>> {
    dataset.validate()?;
    precision(o.working_precision_bits)?;
    if o.working_precision_bits < state.precision.max(dataset.precision_bits)
        || o.maximum_rows == 0
        || dataset.points.len() > o.maximum_rows
    {
        bail!("transform precision or row budget exceeded");
    }
    // Seven high-precision decimals per row plus conservative JSON/label overhead.
    // This is a resource bound, not an estimate of scientific source accuracy.
    let estimated_bytes =
        dataset.points.len() as u64 * (4096 + 8 * (u64::from(o.working_precision_bits) / 3 + 32));
    if estimated_bytes > o.maximum_estimated_output_bytes {
        bail!("transform estimated output byte budget exceeded");
    }
    let coordinate = if root_convention {
        "mellin_t; exp(-i*t*log(u)); retained_secular_root_convention"
    } else {
        "mellin_t; exp(i*t*log(u))"
    };
    let formula = if root_convention {
        "integral_-L/2^L/2 f(x)*exp(-i*t*x) dx; analytic_sinc_and_derivative"
    } else {
        "integral_-L/2^L/2 f(x)*exp(i*t*x) dx; analytic_sinc_and_derivative"
    };
    let arithmetic = if root_convention {
        "exact_cutoff_stored_points_directed_sinc_root_convention_v2"
    } else {
        "exact_cutoff_stored_points_directed_sinc_v1"
    };
    let mut sources = vec![state.manifest.clone()];
    sources.extend_from_slice(extra);
    managed(
        "ccm_indexed_transform_analysis",
        json!({"options":o,"dataset":dataset,"formula":formula,"transform_arithmetic":arithmetic,"maximum_transform_guard_bits":4096,"curvature_output":"outward_upper_endpoint_v1"}),
        &sources,
        cache,
        || {
            let p = o.working_precision_bits;
            let curvature = transform_math::curvature(&state.cutoff, p)?
                .ok_or_else(|| anyhow::anyhow!("curvature unresolved within 4096 guard bits"))?;
            let curvature = Float::with_val_round(p, curvature.upper(), rug::float::Round::Up).0;
            let rows = dataset
                .points
                .par_iter()
                .map(|r| -> Result<_> {
                    let mut row = TransformRow {
                        ordinal: r.ordinal,
                        source_status: r.source_status.clone(),
                        t: r.value.clone(),
                        outcome: "missing_input".into(),
                        value: None,
                        derivative: None,
                        sum_absolute_terms: None,
                        sum_absolute_derivative_terms: None,
                        cancellation_digits: None,
                        newton_correction: None,
                        newton_remainder:
                            "not_bounded; local_Newton_diagnostic_not_a_zero_error_certificate"
                                .into(),
                    };
                    if let Some(t) = &r.value {
                        let t = scalar(t, dataset.precision_bits)?;
                        let (v, d, a, ad) = if root_convention {
                            transform_terms_at_root(state, &t, p)?
                        } else {
                            transform_terms(state, &t, p)?
                        };
                        let scale = (Float::with_val(p, &ad) + 1) >> (p - 32);
                        let floor = (Float::with_val(p, &a) + 1) >> (p - 32);
                        row.outcome = if d.clone().abs() <= scale {
                            "unresolved_derivative"
                        } else if v.clone().abs() <= floor {
                            "cancellation_limited"
                        } else {
                            "point_measurement"
                        }
                        .into();
                        if row.outcome == "point_measurement" {
                            row.newton_correction = Some(dec(&(-point::quotient(&v, &d, p)?)));
                        }
                        if v != 0 && a > 0 {
                            let c = a.clone().log10() - v.clone().abs().log10();
                            row.cancellation_digits = Some(dec(&c.max(&Float::with_val(p, 0))));
                        }
                        row.value = Some(dec(&v));
                        row.derivative = Some(dec(&d));
                        row.sum_absolute_terms = Some(dec(&a));
                        row.sum_absolute_derivative_terms = Some(dec(&ad));
                    }
                    Ok(row)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(TransformData {
                coordinate: coordinate.into(),
                normalization: "unit_L2_dx; center_positive_else_largest_coefficient_positive"
                    .into(),
                input_role: dataset.role.clone(),
                source_precision_bits: state.precision,
                working_precision_bits: p,
                finite_support: "[-log(C)/2,log(C)/2]; zero_extended_finite_Fourier_state".into(),
                physical_tail:
                    "not_available; zero_extension_is_not_an_infinite_ground_state_tail_bound"
                        .into(),
                second_derivative_cauchy_schwarz_expression: dec(&curvature),
                rows,
            })
        },
        |r| {
            if r.rows.len() != dataset.points.len()
                || r.source_precision_bits != state.precision
                || r.working_precision_bits != o.working_precision_bits
                || r.input_role != dataset.role
                || r.coordinate != coordinate
                || r.normalization
                    != "unit_L2_dx; center_positive_else_largest_coefficient_positive"
                || r.finite_support != "[-log(C)/2,log(C)/2]; zero_extended_finite_Fourier_state"
                || r.physical_tail
                    != "not_available; zero_extension_is_not_an_infinite_ground_state_tail_bound"
            {
                bail!("transform report identity mismatch");
            }
            if scalar(
                &r.second_derivative_cauchy_schwarz_expression,
                o.working_precision_bits,
            )? <= 0
            {
                bail!("invalid curvature expression");
            }
            for (row, input) in r.rows.iter().zip(&dataset.points) {
                if row.ordinal != input.ordinal
                    || row.source_status != input.source_status
                    || row.t != input.value
                    || row.newton_remainder
                        != "not_bounded; local_Newton_diagnostic_not_a_zero_error_certificate"
                {
                    bail!("transform row identity mismatch");
                }
                if ![
                    "missing_input",
                    "point_measurement",
                    "unresolved_derivative",
                    "cancellation_limited",
                ]
                .contains(&row.outcome.as_str())
                    || (row.outcome == "missing_input") != input.value.is_none()
                    || row.value.is_some() != input.value.is_some()
                    || row.derivative.is_some() != input.value.is_some()
                    || row.sum_absolute_terms.is_some() != input.value.is_some()
                    || row.sum_absolute_derivative_terms.is_some() != input.value.is_some()
                    || row.newton_correction.is_some() != (row.outcome == "point_measurement")
                {
                    bail!("invalid transform row status");
                }
                for v in [
                    &row.value,
                    &row.derivative,
                    &row.sum_absolute_terms,
                    &row.sum_absolute_derivative_terms,
                    &row.cancellation_digits,
                    &row.newton_correction,
                ]
                .into_iter()
                .flatten()
                {
                    scalar(v, o.working_precision_bits)?;
                }
            }
            Ok(())
        },
    )
}
pub fn capture_transforms_at_roots(
    s: &RetainedState,
    r: &RetainedRoots,
    o: &TransformOptions,
    c: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<TransformData>>> {
    if !xc_cache::manifest_depends_on(&r.secular_manifest, &s.manifest)? {
        bail!("transform roots belong to another state");
    }
    transforms(
        s,
        &r.dataset,
        o,
        &[r.manifest.clone(), r.secular_manifest.clone()],
        true,
        c,
    )
}
pub fn capture_transforms_at_dataset(
    s: &RetainedState,
    d: &RetainedDataset,
    o: &TransformOptions,
    c: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<TransformData>>> {
    transforms(s, &d.spec, o, std::slice::from_ref(&d.manifest), false, c)
}

#[derive(Clone, Debug)]
pub struct RetainedMatrix<'a> {
    pub(crate) manifest: ArtifactManifest,
    pub(crate) cutoff: String,
    pub(crate) modes: usize,
    pub(crate) precision: u32,
    pub(crate) entries: std::borrow::Cow<'a, [Float]>,
}
fn retained_matrix_shape(modes: usize, p: u32, entries: usize) -> Result<usize> {
    precision(p)?;
    let dim = modes
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("retained matrix dimension overflow"))?;
    if modes > 8192
        || dim.checked_mul(dim) != Some(entries)
        || entries as u128 * (u128::from(p).div_ceil(8) + 96) > (8u128 << 30)
    {
        bail!("retained matrix shape or workspace exceeds its supported domain");
    }
    Ok(dim)
}
impl<'a> RetainedMatrix<'a> {
    pub fn from_payload(m: &ArtifactManifest, b: &[u8], allowed: &[ContentDigest]) -> Result<Self> {
        admits(m, b, allowed, "ccm_tau_matrix")?;
        #[derive(Deserialize)]
        struct Payload {
            schema_version: u32,
            lambda_squared: String,
            n_modes: usize,
            precision_bits: u32,
            entries: Vec<String>,
        }
        let r: Payload = serde_json::from_slice(b)?;
        let n = retained_matrix_shape(r.n_modes, r.precision_bits, r.entries.len())?;
        super::research::ExactCutoff::parse(&r.lambda_squared)?;
        if r.schema_version != 2 {
            bail!("invalid retained matrix shape");
        }
        let entries = r
            .entries
            .iter()
            .map(|v| scalar(v, r.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..n {
            for j in 0..i {
                if entries[i * n + j] != entries[j * n + i] {
                    bail!("retained matrix is not exactly symmetric");
                }
            }
        }
        Ok(Self {
            manifest: m.clone(),
            cutoff: r.lambda_squared,
            modes: r.n_modes,
            precision: r.precision_bits,
            entries: std::borrow::Cow::Owned(entries),
        })
    }
    // The private live adapter borrows immutable, reader-authenticated bytes
    // already decoded by the owning run. Recheck numeric/metadata domains here.
    // Preserve its exact stored entries, including historical tiny asymmetry;
    // quadratic-form diagnostics explicitly use the symmetric part.
    pub(crate) fn from_admitted_runtime(
        m: ArtifactManifest,
        cutoff: String,
        modes: usize,
        precision: u32,
        entries: &'a [Float],
    ) -> Result<Self> {
        m.validate()?;
        retained_matrix_shape(modes, precision, entries.len())?;
        super::research::ExactCutoff::parse(&cutoff)?;
        if m.key.kind != "ccm_tau_matrix"
            || !m.immutable
            || m.quality.admissible_rank() < CacheQuality::Validated.admissible_rank()
            || entries
                .iter()
                .any(|v| !v.is_finite() || v.prec() != precision)
        {
            bail!("invalid admitted matrix");
        }
        Ok(Self {
            manifest: m,
            cutoff,
            modes,
            precision,
            entries: std::borrow::Cow::Borrowed(entries),
        })
    }
    pub(crate) fn match_state(&self, s: &RetainedState) -> Result<()> {
        if !equal_cutoff(&self.cutoff, &s.cutoff)?
            || self.modes != s.modes
            || self.precision != s.precision
        {
            bail!("matrix is not the exact retained eigenstate parent");
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyData {
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub source_eigenvalue: String,
    pub coefficient_norm_squared: String,
    pub rayleigh_quotient: String,
    pub eigenvalue_defect: String,
    /// Residual of the raw retained matrix; normalization is explicit below.
    pub relative_residual: String,
    pub residual_normalization: String,
    pub sum_absolute_energy_terms: String,
    pub cancellation_digits: Option<String>,
    pub component_decomposition: String,
    pub ground_selection: String,
}
pub(super) fn matrix_ancestry(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    provided: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<Vec<ArtifactManifest>> {
    let mut queue = vec![(s.manifest.clone(), Vec::new())];
    let mut visited = std::collections::HashSet::new();
    while let Some((parent, path)) = queue.pop() {
        if path.len() > 16 || visited.len() > 64 {
            bail!("matrix ancestry exceeds metadata budget");
        }
        if xc_cache::manifest_depends_on(&parent, &m.manifest)? {
            return Ok(path);
        }
        for next in xc_cache::resolve_manifest_sources(&parent, provided, cache)? {
            if ![
                "ccm_factorization",
                "ccm_even_sector_matrix",
                "ccm_odd_sector_matrix",
            ]
            .contains(&next.key.kind.as_str())
                || !visited.insert((next.key.clone(), next.content_digest.clone()))
            {
                continue;
            }
            let mut chain = path.clone();
            chain.push(next.clone());
            queue.push((next, chain));
        }
    }
    bail!(
        "matrix is not the exact retained eigenstate ancestor; supply its factorization/sector manifests"
    )
}
pub fn capture_operator_energy(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<EnergyData>>> {
    capture_operator_energy_with_ancestry(s, m, &[], cache)
}
/// Metadata-only exact ancestry for offline sources whose factorization chain is
/// not present in the child cache. No factorization payload is read or rebuilt.
pub fn capture_operator_energy_with_ancestry(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    parents: &[ArtifactManifest],
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<EnergyData>>> {
    m.match_state(s)?;
    let mut sources = matrix_ancestry(s, m, parents, cache)?;
    sources.extend([s.manifest.clone(), m.manifest.clone()]);
    let p = s.precision.saturating_add(32).min(1_000_000);
    precision(p)?;
    managed(
        "ccm_operator_energy_analysis",
        json!({"precision_bits":p,"formula":"xi^T Tau xi / xi^T xi; exact_retained_parent","residual_semantics":"raw_matrix_explicit_zero_eigenvalue_normalization_v2"}),
        &sources,
        cache,
        || energy::calculate(s, m, p),
        |r| {
            if r.lambda_squared != s.cutoff
                || r.n_modes != s.modes
                || r.precision_bits != p
                || r.source_eigenvalue != s.eigenvalue
                || r.component_decomposition
                    != "not_captured; total_Tau_only; no_channel_sign_inference"
                || r.ground_selection != "not_established_by_this_diagnostic"
                || r.residual_normalization
                    != energy::residual_normalization(&s.eigenvalue, s.precision)?
            {
                bail!("operator energy identity mismatch");
            }
            for v in [
                &r.coefficient_norm_squared,
                &r.rayleigh_quotient,
                &r.eigenvalue_defect,
                &r.relative_residual,
                &r.sum_absolute_energy_terms,
            ]
            .into_iter()
            .chain(r.cancellation_digits.iter())
            {
                scalar(v, p)?;
            }
            if scalar(&r.coefficient_norm_squared, p)? <= 0
                || scalar(&r.relative_residual, p)? < 0
                || scalar(&r.sum_absolute_energy_terms, p)? < 0
            {
                bail!("invalid energy diagnostics");
            }
            Ok(())
        },
    )
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProjectionOptions {
    pub working_precision_bits: u32,
    pub normalization: String,
    pub fixed_second_component: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionData {
    pub outcome: String,
    pub metric: String,
    pub normalization: String,
    pub source_center: String,
    pub reference_center: String,
    pub signed_unit_overlap: String,
    pub difference_norm_squared: Option<String>,
    pub gram: Vec<String>,
    pub rhs: Vec<String>,
    pub coefficients: Option<Vec<String>>,
    pub fit_residual_norm_squared: Option<String>,
    pub minimum_pivot: Option<String>,
    pub pivot_metric: String,
    pub fixed_second_component: Option<String>,
    pub b_effective: Option<String>,
    pub b2: Option<String>,
    pub uncertainty: String,
    /// Internal arithmetic precision used for the finite enclosures.
    pub arithmetic_precision_bits: u32,
    /// Outward real-decimal bounds for each measured midpoint field.
    pub arithmetic_enclosures: BTreeMap<String, [String; 2]>,
}
pub(super) fn center(v: &[Float], p: u32) -> Result<Float> {
    point::sum(&point::center_terms(v), p)
}
pub(super) fn dot(a: &[Float], b: &[Float], p: u32) -> Result<Float> {
    point::dot(a, b, p)
}
pub(super) fn solve_gram_checked(
    gram: &[Float],
    rhs: &[Float],
    p: u32,
) -> Result<Option<(Vec<Float>, Float)>> {
    projection::solve(gram, rhs, p)
}
pub fn capture_projection(
    state: &RetainedState,
    reference: &RetainedReference,
    basis: &[RetainedReference],
    o: &ProjectionOptions,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ProjectionData>>> {
    let mut canonical_options = o.clone();
    canonical_options.fixed_second_component = o
        .fixed_second_component
        .as_deref()
        .map(canonical_decimal)
        .transpose()?;
    let o = &canonical_options;
    precision(o.working_precision_bits)?;
    let p = o.working_precision_bits;
    if p < state.precision
        || basis.len() > 8
        || !["unit_l2_dx", "center_one"].contains(&o.normalization.as_str())
        || o.fixed_second_component.is_some() && basis.len() != 2
    {
        bail!("unsupported projection policy");
    }
    for r in std::iter::once(reference).chain(basis.iter()) {
        r.spec.validate()?;
        if xc_core::DecimalLiteral::new(&r.spec.lambda_squared)?
            .canonical()?
            .as_str()
            != xc_core::DecimalLiteral::new(&state.cutoff)?
                .canonical()?
                .as_str()
            || r.spec.precision_bits > p
        {
            bail!("projection reference support or precision mismatch");
        }
    }
    if let Some(b) = &o.fixed_second_component {
        scalar(b, p)?;
    }
    let mut sources = vec![state.manifest.clone(), reference.manifest.clone()];
    sources.extend(basis.iter().map(|b| b.manifest.clone()));
    managed(
        "ccm_reference_projection_analysis",
        json!({"options":o,"reference":reference.manifest.content_digest,"ordered_basis":basis.iter().map(|b|&b.manifest.content_digest).collect::<Vec<_>>(),"metric":"full_support_L2_dx; finite_Fourier_coefficients","projection_arithmetic":"stable_normalization_interval_gram_v1","maximum_additional_guard_bits":4096,"projection_output":"midpoints_with_outward_decimal_enclosures_v1"}),
        &sources,
        cache,
        || projection::calculate(state, reference, basis, o),
        |r| {
            if ![
                "normalization_unresolved",
                "rank_or_precision_unresolved",
                "point_measurement",
            ]
            .contains(&r.outcome.as_str())
                || r.metric != "full_support_L2_dx; finite_Fourier_coefficients"
                || r.normalization != o.normalization
                || r.pivot_metric != projection::pivot_metric()
                || r.fixed_second_component != o.fixed_second_component
                || r.uncertainty != projection::uncertainty()
            {
                bail!("invalid projection scope");
            }
            for v in [
                &r.source_center,
                &r.reference_center,
                &r.signed_unit_overlap,
            ]
            .into_iter()
            .chain(r.gram.iter())
            .chain(r.rhs.iter())
            .chain(r.coefficients.iter().flatten())
            .chain(r.fit_residual_norm_squared.iter())
            .chain(r.minimum_pivot.iter())
            .chain(r.b_effective.iter())
            .chain(r.b2.iter())
            {
                scalar(v, p)?;
            }
            if r.outcome == "normalization_unresolved" {
                if r.difference_norm_squared.is_some()
                    || r.coefficients.is_some()
                    || !r.gram.is_empty()
                    || !r.rhs.is_empty()
                {
                    bail!("invalid unresolved projection");
                }
            } else {
                if scalar(
                    r.difference_norm_squared
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("missing projection distance"))?,
                    p,
                )? < 0
                    || r.gram.len() != basis.len().pow(2)
                    || r.rhs.len() != basis.len()
                    || r.coefficients.is_some() != (r.outcome == "point_measurement")
                {
                    bail!("invalid projection shape");
                }
                if let Some(c) = &r.coefficients {
                    if c.len() != basis.len() {
                        bail!("invalid projection coefficient count");
                    }
                }
            }
            projection::validate_enclosures(r, p)?;
            Ok(())
        },
    )
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalObservation {
    pub schema_version: u32,
    pub original_utf8: String,
    pub attribution: String,
    pub definition: String,
    pub hypotheses: Vec<String>,
    pub borrowed_inputs: Vec<String>,
    pub limitations: Vec<String>,
}
impl ExternalObservation {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.original_utf8.len() > 4 * 1024 * 1024
            || self.original_utf8.is_empty()
            || self.attribution.trim().is_empty()
            || self.definition.trim().is_empty()
        {
            bail!("invalid external observation");
        }
        xc_core::validate_secret_free(self, "external observation")?;
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationData {
    pub origin: String,
    pub validation: String,
    pub original_digest: ContentDigest,
    pub observation: ExternalObservation,
}
pub fn capture_external_observation(
    o: &ExternalObservation,
    c: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ObservationData>>> {
    o.validate()?;
    let hash = ContentDigest::sha256(o.original_utf8.as_bytes());
    managed(
        "research_observation_packet",
        serde_json::to_value(o)?,
        &[],
        c,
        || {
            Ok(ObservationData {
                origin: "externally_reported".into(),
                validation: "structure_and_bytes_only; numerical_claims_not_replayed".into(),
                original_digest: hash.clone(),
                observation: o.clone(),
            })
        },
        |r| {
            r.observation.validate()?;
            if r.origin != "externally_reported"
                || r.validation != "structure_and_bytes_only; numerical_claims_not_replayed"
                || r.original_digest != hash
                || serde_json::to_value(&r.observation)? != serde_json::to_value(o)?
            {
                bail!("external observation identity mismatch");
            }
            Ok(())
        },
    )
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StabilizationOptions {
    pub working_precision_bits: u32,
    pub relative_tolerance: String,
    pub consecutive_steps: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StabilizationRow {
    pub source: ContentDigest,
    pub n_modes: usize,
    pub value: String,
    /// Outward upper bound for the relative change of stored eigenvalue points.
    pub relative_change: Option<String>,
    pub status: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StabilizationData {
    pub lambda_squared: String,
    pub observable: String,
    pub qualification: String,
    pub outcome: String,
    pub rows: Vec<StabilizationRow>,
}
pub fn capture_stabilization(
    states: &[RetainedState],
    o: &StabilizationOptions,
    c: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<StabilizationData>>> {
    let mut canonical_options = o.clone();
    canonical_options.relative_tolerance = canonical_decimal(&o.relative_tolerance)?;
    let o = &canonical_options;
    precision(o.working_precision_bits)?;
    let p = o.working_precision_bits;
    let tolerance = acceptance_tolerance(&o.relative_tolerance, p)?;
    if states.len() < 2
        || states.len() > 10000
        || o.consecutive_steps == 0
        || o.consecutive_steps >= states.len()
        || tolerance <= 0
        || tolerance >= 1
    {
        bail!("invalid stabilization cohort or rule");
    }
    let first = &states[0];
    if first.selection_policy.is_none() {
        bail!("cohort source parity policy is unspecified");
    }
    for s in states {
        if s.selection_policy != first.selection_policy
            || !equal_cutoff(&s.cutoff, &first.cutoff)?
            || s.precision != first.precision
            || p < s.precision
        {
            bail!("stabilization requires fixed cutoff and construction precision");
        }
    }
    if states.windows(2).any(|w| w[0].modes >= w[1].modes) {
        bail!("stabilization dimensions must strictly increase");
    }
    managed(
        "ccm_stabilization_analysis",
        json!({"options":o,"acceptance_rounding":"exact_decimal_down","relative_change_arithmetic":"exact_stored_points_outward_upper_v2","ordered_sources":states.iter().map(|s|&s.manifest.content_digest).collect::<Vec<_>>()}),
        &states
            .iter()
            .map(|s| s.manifest.clone())
            .collect::<Vec<_>>(),
        c,
        || stabilization_data(states, o, &tolerance),
        |r| validate_stabilization(r, states, o, &tolerance),
    )
}

fn stabilization_data(
    states: &[RetainedState],
    o: &StabilizationOptions,
    tolerance: &Float,
) -> Result<StabilizationData> {
    let first = &states[0];
    let p = o.working_precision_bits;
    let mut rows = Vec::new();
    let mut previous: Option<Float> = None;
    let mut streak = 0;
    for s in states {
        let value = Float::with_val(p + 64, scalar(&s.eigenvalue, s.precision)?);
        let change = if let Some(old) = &previous {
            if old.is_zero() {
                None
            } else {
                let exponent = old
                    .get_exp()
                    .into_iter()
                    .chain(value.get_exp())
                    .max()
                    .unwrap();
                let a = point::scale(old, -i64::from(exponent), p + 64)?;
                let b = point::scale(&value, -i64::from(exponent), p + 64)?;
                use xc_numerics::mpfr_interval::MpfrInterval as I;
                let denominator = I::point(a);
                let change = I::point(b).sub(&denominator).div(&denominator)?;
                let absolute = finite_math::abs(&change)?;
                // The rule must not accept a rounded-down excess change.
                // Both subtraction/division and the final stored bound are outward.
                Some(Float::with_val_round(p, absolute.upper(), rug::float::Round::Up).0)
            }
        } else {
            None
        };
        let status = if previous.is_none() {
            "initial"
        } else if change.is_none() {
            "zero_denominator"
        } else if change.as_ref().unwrap() <= tolerance {
            streak += 1;
            "within_rule"
        } else {
            streak = 0;
            "outside_rule"
        };
        if status == "zero_denominator" {
            streak = 0;
        }
        rows.push(StabilizationRow {
            source: s.manifest.content_digest.clone(),
            n_modes: s.modes,
            value: s.eigenvalue.clone(),
            relative_change: change.as_ref().map(dec),
            status: status.into(),
        });
        previous = Some(value);
    }
    Ok(StabilizationData{lambda_squared:first.cutoff.clone(),observable:"serialized_retained_Weil_eigenvalue".into(),qualification:"finite_frozen_rule_only; branch_and_assembly_policy_comparability_not_established; not_N_infinity".into(),outcome:if streak>=o.consecutive_steps{"finite_rule_met"}else{"finite_rule_not_met"}.into(),rows})
}

fn validate_stabilization(
    r: &StabilizationData,
    states: &[RetainedState],
    o: &StabilizationOptions,
    tolerance: &Float,
) -> Result<()> {
    // Replay cheap finite statistics from the admitted source points. A matching
    // count total or recognized status alone does not authenticate a conclusion.
    let expected = stabilization_data(states, o, tolerance)?;
    if r != &expected {
        bail!("stabilization summary disagrees with source replay");
    }
    Ok(())
}

/// Optional, explicit additional sources for the live adapter. No scripts execute.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchInputs {
    pub schema_version: u32,
    pub reference: ReferenceSpec,
    pub basis: Vec<ReferenceSpec>,
    pub projection: ProjectionOptions,
}
impl ResearchInputs {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 || self.basis.len() > 8 {
            bail!("unsupported research input bundle");
        }
        self.reference.validate()?;
        for b in &self.basis {
            b.validate()?;
        }
        precision(self.projection.working_precision_bits)?;
        Ok(())
    }
}
fn admitted_reference(
    result: ArtifactExecutionCacheResult<ResearchRecord<ReferenceData>>,
) -> Result<RetainedReference> {
    let m = result
        .produced_manifest
        .or(result.reused_manifest)
        .ok_or_else(|| anyhow::anyhow!("reference capture requires a managed manifest"))?;
    let bytes = serde_json::to_vec(&result.value)?;
    RetainedReference::from_payload(&m, &bytes, std::slice::from_ref(&m.content_digest))
}
pub fn capture_configured_projection(
    state: &RetainedState,
    input: &ResearchInputs,
    cache: &ArtifactCacheContext<'_>,
) -> Result<CapturedDiagnostic> {
    input.validate()?;
    let reference = admitted_reference(capture_reference(&input.reference, cache)?)?;
    let basis = input
        .basis
        .iter()
        .map(|b| admitted_reference(capture_reference(b, cache)?))
        .collect::<Result<Vec<_>>>()?;
    Ok(CapturedDiagnostic::from_cached(capture_projection(
        state,
        &reference,
        &basis,
        &input.projection,
        cache,
    )?)?)
}

fn acceptance_tolerance(text: &str, p: u32) -> Result<Float> {
    scalar(text, p)?;
    // Accepting an error must not relax the exact supplied decimal limit.
    Ok(Float::with_val_round(p, Float::parse(text)?, rug::float::Round::Down).0)
}

#[cfg(test)]
mod renewed_boundary_contract {
    use super::*;
    #[test]
    fn acceptance_decimal_cannot_round_up_to_half() {
        let value = acceptance_tolerance("0.499999999999999999999999999999999999999", 64).unwrap();
        assert!(value < 0.5);
    }
}

#[cfg(test)]
mod exhaustive_resumed_validation_contract {
    use super::*;
    fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
        let bytes = serde_json::to_vec(&value).unwrap();
        let digest = ContentDigest::sha256(&bytes);
        let manifest = ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(kind, "synthetic-geometry-fixture", kind.as_bytes()).unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags: BTreeMap::new(),
            provenance_digest: None,
        };
        (manifest, bytes)
    }

    fn context() -> ArtifactCacheContext<'static> {
        ArtifactCacheContext {
            resolver: None,
            reference_resolver: None,
            acceptance: None,
            ordered_overlays: vec!["disabled".into()],
            mode: ArtifactExecutionCacheMode::Disabled,
            write_on_miss: false,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        }
    }
    fn state(coefficients: &[&str], eigenvalue: &str) -> (RetainedState, ArtifactManifest) {
        let (m, b) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":3,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"force_even":true,"eigenvalue":eigenvalue,"eigenvector":coefficients}),
        );
        (
            RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
            m,
        )
    }
    fn roots_at(values: &[Option<&str>]) -> RetainedRoots {
        let (s, sm) = state(&["0", "1", "0"], "1");
        let (mut sec, sb) = source(
            "ccm_secular_source",
            json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"normalization":"sum_xi_equals_sqrt_log_lambda_squared","eigenpair_content_digest":sm.content_digest.0}),
        );
        let dep = |m: &ArtifactManifest| DependencyRef {
            key: m.key.clone(),
            content_digest: m.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        };
        sec.dependencies.push(dep(&sm));
        let outcomes = values
            .iter()
            .map(|v| match v {
                Some(v) => json!({"status":"converged","details":{"value":v}}),
                None => json!({"status":"failed","details":{"reason":"missing"}}),
            })
            .collect::<Vec<_>>();
        let (mut rm, rb) = source(
            "ccm_root_refinement",
            json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"reference_seeded_audit","reference_seeds_used":true,"completeness":"partial","outcomes":outcomes}),
        );
        rm.dependencies.push(dep(&sec));
        let allowed = [
            sm.content_digest.clone(),
            sec.content_digest.clone(),
            rm.content_digest.clone(),
        ];
        RetainedRoots::from_payload(&rm, &rb, &sec, &sb, &s, &allowed).unwrap()
    }

    #[test]
    fn exhaustive_resumed_root_window_replays_classification() {
        let roots = roots_at(&[Some("1"), Some("2"), Some("0"), None]);
        let mut data = capture_root_window(&roots, &context()).unwrap().value.data;
        data.positive_count = 1;
        data.nonpositive_count = 2;
        assert!(validate_root_window(&data, &roots).is_err());
    }
    #[test]
    fn exhaustive_resumed_root_window_rejects_changed_statistics() {
        let roots = roots_at(&[Some("1"), Some("2"), Some("0"), None]);
        let original = capture_root_window(&roots, &context()).unwrap().value.data;
        let mut accepted = Vec::new();
        for k in 0..5 {
            let mut data = original.clone();
            match k {
                0..=2 => data.window_inverse_moments[k] = "0".into(),
                3 => data.minimum_positive_spacing = None,
                _ => data.minimum_positive_spacing = Some("100".into()),
            }
            if validate_root_window(&data, &roots).is_ok() {
                accepted.push(k);
            }
        }
        assert!(
            accepted.is_empty(),
            "accepted altered statistics: {accepted:?}"
        );
    }
    #[test]
    fn exhaustive_resumed_root_window_count_overflow_returns_error() {
        let roots = roots_at(&[Some("1"), Some("0"), None]);
        let mut data = capture_root_window(&roots, &context()).unwrap().value.data;
        data.positive_count = usize::MAX;
        let result = std::panic::catch_unwind(|| validate_root_window(&data, &roots));
        assert!(
            matches!(result, Ok(Err(_))),
            "untrusted counts must fail without panicking"
        );
    }
    #[test]
    fn exhaustive_resumed_stabilization_replays_rule_and_streak() {
        let states = [
            state(&["1"], "1").0,
            state(&["0", "1", "0"], "2").0,
            state(&["0", "0", "1", "0", "0"], "2").0,
        ];
        let o = StabilizationOptions {
            working_precision_bits: 128,
            relative_tolerance: "0.1".into(),
            consecutive_steps: 2,
        };
        let tolerance = acceptance_tolerance(&o.relative_tolerance, 128).unwrap();
        let original = capture_stabilization(&states, &o, &context())
            .unwrap()
            .value
            .data;
        let mut accepted = Vec::new();
        for k in 0..5 {
            let mut data = original.clone();
            match k {
                0 => data.outcome = "finite_rule_met".into(),
                1 => data.rows[1].status = "within_rule".into(),
                2 => data.rows[1].relative_change = Some("0".into()),
                3 => data.rows[0].relative_change = Some("0".into()),
                _ => data.rows[2].relative_change = None,
            }
            if validate_stabilization(&data, &states, &o, &tolerance).is_ok() {
                accepted.push(k);
            }
        }
        assert!(
            accepted.is_empty(),
            "accepted altered rule fields: {accepted:?}"
        );
    }
}

#[cfg(test)]
mod exhaustive_scalar_floor {
    use super::*;
    #[test]
    fn exhaustive_scalar_floor_rejects_partial_underflow() {
        let p = 128;
        let minimum = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        let text = minimum.to_string();
        let (mantissa, exponent) = text.split_once('e').unwrap();
        let mantissa = Float::with_val(512, Float::parse(mantissa).unwrap()) * 0.75;
        let below = format!("{mantissa}e{exponent}");
        let rounded = Float::with_val(p, Float::parse(&below).unwrap());
        assert_eq!(rounded, minimum); // The old parser accepts this large relative error.
        assert!(scalar(&below, p).is_err());
        assert!(scalar(&format!("-{below}"), p).is_err());
    }
    #[test]
    fn exhaustive_scalar_floor_requires_decimal_literal_grammar() {
        assert!(scalar("1@2", 128).is_err());
        assert!(scalar(" 1", 128).is_err());
    }
    #[test]
    fn exhaustive_scalar_floor_input_budget_precedes_parsing() {
        let text = format!("1.{}", "0".repeat(1_048_576));
        assert!(scalar(&text, 128).is_err());
    }
    #[test]
    fn exhaustive_scalar_floor_domain_and_range_controls() {
        for p in [64, 128, 1_000_128] {
            for s in ["+1.25", "1.250E+0", ".125e1", "125.e-2"] {
                assert_eq!(scalar(s, p).unwrap(), Float::with_val(p, 1.25));
            }
            assert_eq!(scalar("-1.25", p).unwrap(), Float::with_val(p, -1.25));
            for s in ["0", "-0.000", "+0e-999999999", "0e+999999999"] {
                assert!(scalar(s, p).unwrap().is_zero());
            }
        }
        for s in [
            "",
            "1\n",
            "1 0",
            "NaN",
            "inf",
            "-Inf",
            "１２",
            "0x1",
            "1e9999999999",
            "1e-9999999999",
        ] {
            assert!(scalar(s, 128).is_err(), "{s:?}");
        }
        for p in [0, 63, 1_000_129, u32::MAX] {
            assert!(scalar("1", p).is_err());
        }
        let within_budget = format!("1.{}", "0".repeat(1_048_574));
        assert_eq!(within_budget.len(), 1_048_576);
        assert_eq!(scalar(&within_budget, 128).unwrap(), 1);
        let minimum = Float::with_val(128, 1) << (rug::float::exp_min() - 1);
        let text = minimum.to_string();
        let (mantissa, exponent) = text.split_once('e').unwrap();
        let mantissa = Float::with_val(512, Float::parse(mantissa).unwrap()) * 1.25;
        let above = format!("{mantissa}e{exponent}");
        let expected = Float::with_val(128, Float::parse(&above).unwrap());
        assert!(expected > minimum);
        assert_eq!(scalar(&above, 128).unwrap(), expected);
        assert_eq!(scalar(&format!("-{above}"), 128).unwrap(), -expected);
    }
}

#[cfg(test)]
mod exhaustive_runtime_matrix {
    use super::*;
    fn manifest() -> ArtifactManifest {
        let bytes = b"admitted runtime matrix boundary";
        let digest = ContentDigest::sha256(bytes);
        serde_json::from_value(json!({"schema_version":1,"key":ArtifactKey::new("ccm_tau_matrix","runtime-boundary",bytes).unwrap(),"content_digest":digest,"size_bytes":bytes.len(),"objects":[{"content_digest":digest,"size_bytes":bytes.len()}],"created_unix_seconds":1,"producer_toolkit_version":ToolkitVersion::parse("0.16.0").unwrap(),"minimum_reader_version":ToolkitVersion::parse("0.16.0").unwrap(),"maximum_reader_version":null,"quality":"validated","visibility":"private","immutable":true,"dependencies":[],"tags":{},"provenance_digest":null})).unwrap()
    }
    #[test]
    fn exhaustive_runtime_matrix_shape_overflow_returns_error() {
        assert!(RetainedMatrix::from_admitted_runtime(
            manifest(),
            "13".into(),
            usize::MAX,
            128,
            &[]
        )
        .is_err());
    }
    #[test]
    fn exhaustive_runtime_matrix_rejects_invalid_precision() {
        assert!(RetainedMatrix::from_admitted_runtime(
            manifest(),
            "13".into(),
            0,
            0,
            &[Float::with_val(128, 1)]
        )
        .is_err());
    }
    #[test]
    fn exhaustive_runtime_matrix_rejects_false_precision_stamp() {
        assert!(RetainedMatrix::from_admitted_runtime(
            manifest(),
            "13".into(),
            0,
            128,
            &[Float::with_val(512, 1) + (Float::with_val(512, 1) >> 400)]
        )
        .is_err());
    }
    #[test]
    fn exhaustive_runtime_matrix_rejects_invalid_cutoff() {
        assert!(RetainedMatrix::from_admitted_runtime(
            manifest(),
            "NaN".into(),
            0,
            128,
            &[Float::with_val(128, 1)]
        )
        .is_err());
    }
    #[test]
    fn exhaustive_runtime_matrix_rejects_mutable_source() {
        let mut m = manifest();
        m.immutable = false;
        assert!(RetainedMatrix::from_admitted_runtime(
            m,
            "13".into(),
            0,
            128,
            &[Float::with_val(128, 1)]
        )
        .is_err());
    }
}

#[cfg(test)]
mod exhaustive_remaining_dot {
    use super::*;
    #[test]
    fn exhaustive_remaining_dot_preserves_cancelled_unit() {
        let p = 128;
        let x = Float::with_val(p, 1) << 400u32;
        assert_eq!(
            dot(
                &[x.clone(), Float::with_val(p, 1), -x],
                &vec![Float::with_val(p, 1); 3],
                p
            )
            .unwrap(),
            1
        );
    }
}

#[cfg(test)]
#[path = "retained_evidence/retained_regression_tests.rs"]
mod retained_regression_tests;
