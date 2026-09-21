//! Managed finite-state research observations. Inputs are explicit, numerical
//! scope travels with each report, and no primary solve is available here.
use super::state_geometry::RetainedState;
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Constant, Float};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use xc_cache::*;
use xc_numerics::prefix::lossless_decimal as dec;

pub const SEMANTICS: &str = "ccm-retained-research-observations-v1";
const SCOPE: &str = "finite_point_inputs; no_source_error_enclosure; no_ground_selection_or_convergence_certificate";
pub(super) fn scalar(s: &str, p: u32) -> Result<Float> {
    let v = Float::with_val(p, Float::parse(s)?);
    if !v.is_finite() {
        bail!("nonfinite research scalar");
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
        minimum_reader_version: ToolkitVersion::parse("0.15.1")?,
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
    fn values(&self, p: u32) -> Result<Vec<Float>> {
        self.coefficients.iter().map(|s| scalar(s, p)).collect()
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
pub struct RetainedReference {
    manifest: ArtifactManifest,
    pub spec: ReferenceSpec,
}
impl RetainedReference {
    pub fn from_payload(m: &ArtifactManifest, b: &[u8], allowed: &[ContentDigest]) -> Result<Self> {
        admits(m, b, allowed, "ccm_reference_source")?;
        let r: ResearchRecord<ReferenceData> = serde_json::from_slice(b)?;
        r.data.spec.validate()?;
        if r.schema_version != 1
            || r.semantics != SEMANTICS
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
pub struct RetainedDataset {
    pub(crate) manifest: ArtifactManifest,
    pub spec: DatasetSpec,
}
impl RetainedDataset {
    pub fn from_payload(m: &ArtifactManifest, b: &[u8], allowed: &[ContentDigest]) -> Result<Self> {
        admits(m, b, allowed, "research_reference_dataset")?;
        let r: ResearchRecord<DatasetSpec> = serde_json::from_slice(b)?;
        r.data.validate()?;
        if r.schema_version != 1
            || r.semantics != SEMANTICS
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
        let mut root_precision = state.precision;
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
                    precision(bits)?;
                    root_precision = root_precision.max(bits);
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        || {
            let mut positive = Vec::new();
            let mut nonpositive = 0;
            let mut missing = 0;
            for row in &spec.points {
                match &row.value {
                    Some(v) => {
                        let v = scalar(v, p)?;
                        if v > 0 {
                            positive.push(v);
                        } else {
                            nonpositive += 1;
                        }
                    }
                    None => missing += 1,
                }
            }
            let mut sums = vec![Float::with_val(p, 0); 3];
            for x in &positive {
                let inverse = Float::with_val(p, 1) / x;
                let mut term = inverse.clone();
                for sum in &mut sums {
                    *sum += &term;
                    term *= &inverse;
                }
            }
            positive.sort_by(|a, b| a.total_cmp(b));
            let gap = positive
                .windows(2)
                .map(|w| Float::with_val(p, &w[1]) - &w[0])
                .min_by(|a, b| a.total_cmp(b));
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
        },
        |r| {
            if r.lambda_squared != roots.cutoff
                || r.n_modes != roots.modes
                || r.source_completeness != roots.completeness
                || r.source_acquisition != roots.acquisition
                || r.points != spec.points
                || r.positive_count + r.nonpositive_count + r.missing_count != spec.points.len()
                || r.window_inverse_moments.len() != 3
                || r.omitted_tail != "not_bounded; window_moments_are_not_infinite_band_moments"
                || r.ordinal_scope != "preserved_source_window_ordinals; not_zeta_identification"
            {
                bail!("invalid root-window summary");
            }
            for v in r
                .window_inverse_moments
                .iter()
                .chain(r.minimum_positive_spacing.iter())
            {
                if scalar(v, p)? < 0 {
                    bail!("negative positive-root statistic");
                }
            }
            Ok(())
        },
    )
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
            working_precision_bits: s.precision.saturating_add(64),
            maximum_rows: 100000,
            maximum_estimated_output_bytes: default_transform_byte_budget(),
        }
    }
}
impl TransformOptions {
    pub fn for_roots(s: &RetainedState, r: &RetainedRoots) -> Self {
        Self {
            working_precision_bits: s.precision.max(r.dataset.precision_bits).saturating_add(64),
            ..Self::for_source(s)
        }
    }
    pub fn for_dataset(s: &RetainedState, d: &RetainedDataset) -> Self {
        Self {
            working_precision_bits: s.precision.max(d.spec.precision_bits).saturating_add(64),
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
    let n = x.len() / 2;
    let mut center = Float::with_val(p, 0);
    for (i, v) in x.iter().enumerate() {
        if i.abs_diff(n).is_multiple_of(2) {
            center += v;
        } else {
            center -= v;
        }
    }
    let v = if center == 0 {
        x.iter()
            .max_by(|a, b| (*a).clone().abs().total_cmp(&(*b).clone().abs()))
            .unwrap()
    } else {
        &center
    };
    if v < &0 {
        -1
    } else {
        1
    }
}
pub(super) fn norm2(x: &[Float], p: u32) -> Float {
    x.iter().fold(Float::with_val(p, 0), |mut s, v| {
        s += v * v;
        s
    })
}
fn sinc_pair(q: &Float, p: u32) -> (Float, Float) {
    if q.clone().abs() < Float::with_val(p, 0.5) {
        let q2 = Float::with_val(p, q * q);
        let mut sum = Float::with_val(p, 1);
        let mut derivative = Float::with_val(p, 0);
        let mut term = Float::with_val(p, 1);
        if q == &0 {
            return (sum, derivative);
        }
        let tolerance = Float::with_val(p, 1) >> p;
        for m in 1u32..=p {
            term *= &q2;
            term = -term;
            term /= 2 * m;
            term /= 2 * m + 1;
            sum += &term;
            derivative += Float::with_val(p, &term) * (2 * m) / q;
            if term.clone().abs() < tolerance {
                break;
            }
        }
        (sum, derivative)
    } else {
        let (sin, cos) = q.clone().sin_cos(Float::new(p));
        (
            Float::with_val(p, &sin) / q,
            (cos - Float::with_val(p, &sin) / q) / q,
        )
    }
}
/// Real transform of a Hermitian Fourier state: integral f(x) exp(i*t*x) dx.
/// The small-argument branch avoids removable carrier singularities.
pub(super) fn transform_terms(
    state: &RetainedState,
    t: &Float,
    p: u32,
) -> Result<(Float, Float, Float, Float)> {
    let l = scalar(&state.cutoff, p)?.ln();
    let half = Float::with_val(p, &l) / 2u32;
    let pi = Float::with_val(p, Constant::Pi);
    let scale = Float::with_val(p, orientation(&state.coefficients, p))
        / (norm2(&state.coefficients, p) * &l).sqrt();
    let v = Float::with_val(p, t) * &half;
    let (sinv, cosv) = v.clone().sin_cos(Float::new(p));
    let mut value = Float::with_val(p, 0);
    let mut derivative = Float::with_val(p, 0);
    let mut abs = Float::with_val(p, 0);
    let mut absd = Float::with_val(p, 0);
    for (i, xi) in state.coefficients.iter().enumerate() {
        let j = i as i64 - state.modes as i64;
        let q = Float::with_val(p, &pi) * j + &v;
        let (mut a, mut b) = if q.clone().abs() < Float::with_val(p, 0.5) {
            let (a, b) = sinc_pair(&q, p);
            let sign = if j.unsigned_abs().is_multiple_of(2) {
                1
            } else {
                -1
            };
            (a * sign, b * sign)
        } else {
            (
                Float::with_val(p, &sinv) / &q,
                (Float::with_val(p, &cosv) - Float::with_val(p, &sinv) / &q) / &q,
            )
        };
        a *= xi;
        a *= &l;
        a *= &scale;
        b *= xi;
        b *= &l;
        b *= &half;
        b *= &scale;
        abs += a.clone().abs();
        absd += b.clone().abs();
        value += a;
        derivative += b;
    }
    Ok((value, derivative, abs, absd))
}
fn transforms(
    state: &RetainedState,
    dataset: &DatasetSpec,
    o: &TransformOptions,
    extra: &[ArtifactManifest],
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
    let mut sources = vec![state.manifest.clone()];
    sources.extend_from_slice(extra);
    managed(
        "ccm_indexed_transform_analysis",
        json!({"options":o,"dataset":dataset,"formula":"integral_-L/2^L/2 f(x)*exp(i*t*x) dx; analytic_sinc_and_derivative"}),
        &sources,
        cache,
        || {
            let p = o.working_precision_bits;
            let l = scalar(&state.cutoff, p)?.ln();
            let curvature = (Float::with_val(p, &l) * &l * &l * &l * &l / 80u32).sqrt();
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
                        let (v, d, a, ad) = transform_terms(state, &scalar(t, p)?, p)?;
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
                            row.newton_correction = Some(dec(&(-Float::with_val(p, &v) / &d)));
                        }
                        if v != 0 && a > 0 {
                            let c = (Float::with_val(p, &a) / v.clone().abs()).log10();
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
                coordinate: "mellin_t; exp(i*t*log(u))".into(),
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
                || r.coordinate != "mellin_t; exp(i*t*log(u))"
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
        c,
    )
}
pub fn capture_transforms_at_dataset(
    s: &RetainedState,
    d: &RetainedDataset,
    o: &TransformOptions,
    c: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<TransformData>>> {
    transforms(s, &d.spec, o, std::slice::from_ref(&d.manifest), c)
}

#[derive(Clone, Debug)]
pub struct RetainedMatrix<'a> {
    pub(crate) manifest: ArtifactManifest,
    pub(crate) cutoff: String,
    pub(crate) modes: usize,
    pub(crate) precision: u32,
    pub(crate) entries: std::borrow::Cow<'a, [Float]>,
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
        precision(r.precision_bits)?;
        if r.schema_version != 2
            || r.n_modes > 8192
            || scalar(&r.lambda_squared, r.precision_bits)? <= 1
            || r.entries.len() != (2 * r.n_modes + 1).pow(2)
        {
            bail!("invalid retained matrix shape");
        }
        let entries = r
            .entries
            .iter()
            .map(|v| scalar(v, r.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        let n = 2 * r.n_modes + 1;
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
    // The live adapter already has an authenticated matrix from its managed reader.
    pub(crate) fn from_admitted_runtime(
        m: ArtifactManifest,
        cutoff: String,
        modes: usize,
        precision: u32,
        entries: &'a [Float],
    ) -> Result<Self> {
        m.validate()?;
        if m.key.kind != "ccm_tau_matrix"
            || entries.len() != (2 * modes + 1).pow(2)
            || entries.iter().any(|v| !v.is_finite())
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
        if self.cutoff != s.cutoff || self.modes != s.modes || self.precision != s.precision {
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
    pub relative_residual: String,
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
    bail!("matrix is not the exact retained eigenstate ancestor; supply its factorization/sector manifests")
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
    let p = s.precision.saturating_add(32);
    precision(p)?;
    managed(
        "ccm_operator_energy_analysis",
        json!({"precision_bits":p,"formula":"xi^T Tau xi / xi^T xi; exact_retained_parent"}),
        &sources,
        cache,
        || {
            let n = s.coefficients.len();
            let norm = norm2(&s.coefficients, p);
            let eigen = scalar(&s.eigenvalue, p)?;
            let actions = m
                .entries
                .par_chunks(n)
                .enumerate()
                .map(|(i, row)| {
                    let mut v = Float::with_val(p, 0);
                    let mut a = Float::with_val(p, 0);
                    for (x, y) in row.iter().zip(&s.coefficients) {
                        let product = Float::with_val(p, x * y);
                        v += &product;
                        a += product.abs();
                    }
                    a *= s.coefficients[i].clone().abs();
                    (v, a)
                })
                .collect::<Vec<_>>();
            let mut energy = Float::with_val(p, 0);
            let mut abs = Float::with_val(p, 0);
            let mut residual = Float::with_val(p, 0);
            for ((ax, a), x) in actions.iter().zip(&s.coefficients) {
                energy += Float::with_val(p, ax * x);
                abs += a;
                residual += (Float::with_val(p, ax) - Float::with_val(p, &eigen) * x).square();
            }
            let rayleigh = Float::with_val(p, &energy) / &norm;
            let defect = Float::with_val(p, &rayleigh) - &eigen;
            let scale = if eigen == 0 {
                Float::with_val(p, 1)
            } else {
                eigen.clone().abs()
            };
            let cancellation = if energy != 0 && abs > 0 {
                Some(dec(&(Float::with_val(p, &abs) / energy.abs())
                    .log10()
                    .max(&Float::with_val(p, 0))))
            } else {
                None
            };
            Ok(EnergyData {
                lambda_squared: s.cutoff.clone(),
                n_modes: s.modes,
                precision_bits: p,
                source_eigenvalue: s.eigenvalue.clone(),
                coefficient_norm_squared: dec(&norm),
                rayleigh_quotient: dec(&rayleigh),
                eigenvalue_defect: dec(&defect),
                relative_residual: dec(&(residual.sqrt() / norm.sqrt() / scale)),
                sum_absolute_energy_terms: dec(&abs),
                cancellation_digits: cancellation,
                component_decomposition: "not_captured; total_Tau_only; no_channel_sign_inference"
                    .into(),
                ground_selection: "not_established_by_this_diagnostic".into(),
            })
        },
        |r| {
            if r.lambda_squared != s.cutoff
                || r.n_modes != s.modes
                || r.precision_bits != p
                || r.source_eigenvalue != s.eigenvalue
                || r.component_decomposition
                    != "not_captured; total_Tau_only; no_channel_sign_inference"
                || r.ground_selection != "not_established_by_this_diagnostic"
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
    pub fixed_second_component: Option<String>,
    pub b_effective: Option<String>,
    pub b2: Option<String>,
    pub uncertainty: String,
}
fn padded(v: &[Float], modes: usize, p: u32) -> Vec<Float> {
    let mut out = vec![Float::with_val(p, 0); 2 * modes + 1];
    let n = v.len() / 2;
    for (i, x) in v.iter().enumerate() {
        out[modes - n + i] = Float::with_val(p, x);
    }
    out
}
pub(super) fn center(v: &[Float], p: u32) -> Float {
    let n = v.len() / 2;
    v.iter()
        .enumerate()
        .fold(Float::with_val(p, 0), |mut s, (i, x)| {
            if i.abs_diff(n).is_multiple_of(2) {
                s += x;
            } else {
                s -= x;
            }
            s
        })
}
pub(super) fn dot(a: &[Float], b: &[Float], p: u32) -> Float {
    a.iter()
        .zip(b)
        .fold(Float::with_val(p, 0), |mut s, (x, y)| {
            s += x * y;
            s
        })
}
pub(super) fn solve_small(gram: &[Float], rhs: &[Float], p: u32) -> Option<(Vec<Float>, Float)> {
    let n = rhs.len();
    if n == 0 {
        return Some((vec![], Float::with_val(p, 0)));
    }
    let mut a = gram.to_vec();
    let mut b = rhs.to_vec();
    let max = a
        .iter()
        .fold(Float::with_val(p, 0), |m, x| m.max(&x.clone().abs()));
    let tolerance = max >> (p - 32);
    let mut smallest = Float::with_val(p, f64::INFINITY);
    for k in 0..n {
        let pivot = a[k * n + k].clone();
        if pivot <= tolerance {
            return None;
        }
        smallest = smallest.min(&pivot);
        for i in k + 1..n {
            let factor = Float::with_val(p, &a[i * n + k]) / &pivot;
            for j in k..n {
                let product = Float::with_val(p, &factor) * &a[k * n + j];
                a[i * n + j] -= product;
            }
            let product = Float::with_val(p, &factor) * &b[k];
            b[i] -= product;
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            let product = Float::with_val(p, &a[i * n + j]) * &b[j];
            b[i] -= product;
        }
        b[i] /= &a[i * n + i];
    }
    Some((b, smallest))
}
pub fn capture_projection(
    state: &RetainedState,
    reference: &RetainedReference,
    basis: &[RetainedReference],
    o: &ProjectionOptions,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<ResearchRecord<ProjectionData>>> {
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
        if scalar(&r.spec.lambda_squared, p)? != scalar(&state.cutoff, p)?
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
        json!({"options":o,"reference":reference.manifest.content_digest,"ordered_basis":basis.iter().map(|b|&b.manifest.content_digest).collect::<Vec<_>>(),"metric":"full_support_L2_dx; finite_Fourier_coefficients"}),
        &sources,
        cache,
        || {
            let n = std::iter::once(state.modes)
                .chain(std::iter::once(reference.spec.coefficients.len() / 2))
                .chain(basis.iter().map(|b| b.spec.coefficients.len() / 2))
                .max()
                .unwrap();
            let l = scalar(&state.cutoff, p)?.ln();
            let mut u = padded(&state.coefficients, n, p);
            let mut v = padded(&reference.spec.values(p)?, n, p);
            let uc = center(&u, p);
            let vc = center(&v, p);
            let overlap = dot(&u, &v, p) / (norm2(&u, p) * norm2(&v, p)).sqrt()
                * orientation(&u, p)
                * orientation(&v, p);
            let unresolved = o.normalization == "center_one" && (uc == 0 || vc == 0);
            let mut report = ProjectionData {
                outcome: "normalization_unresolved".into(),
                metric: "full_support_L2_dx; finite_Fourier_coefficients".into(),
                normalization: o.normalization.clone(),
                source_center: dec(&uc),
                reference_center: dec(&vc),
                signed_unit_overlap: dec(&overlap),
                difference_norm_squared: None,
                gram: vec![],
                rhs: vec![],
                coefficients: None,
                fit_residual_norm_squared: None,
                minimum_pivot: None,
                fixed_second_component: o.fixed_second_component.clone(),
                b_effective: None,
                b2: None,
                uncertainty:
                    "point_inputs; no_reference_approximation_or_construction_error_enclosure"
                        .into(),
            };
            if unresolved {
                report.difference_norm_squared = None;
                return Ok(report);
            }
            let us = if o.normalization == "center_one" {
                Float::with_val(p, 1) / &uc
            } else {
                Float::with_val(p, orientation(&u, p)) / (norm2(&u, p) * &l).sqrt()
            };
            let vs = if o.normalization == "center_one" {
                Float::with_val(p, 1) / &vc
            } else {
                Float::with_val(p, orientation(&v, p)) / (norm2(&v, p) * &l).sqrt()
            };
            for x in &mut u {
                *x *= &us;
            }
            for x in &mut v {
                *x *= &vs;
            }
            let h = u
                .iter()
                .zip(&v)
                .map(|(a, b)| Float::with_val(p, a) - b)
                .collect::<Vec<_>>();
            let h2 = norm2(&h, p) * &l;
            report.difference_norm_squared = Some(dec(&h2));
            let vectors = basis
                .iter()
                .map(|b| Ok(padded(&b.spec.values(p)?, n, p)))
                .collect::<Result<Vec<_>>>()?;
            let mut gram = Vec::new();
            for a in &vectors {
                for b in &vectors {
                    gram.push(dot(a, b, p) * &l);
                }
            }
            let rhs = vectors
                .iter()
                .map(|a| dot(a, &h, p) * &l)
                .collect::<Vec<_>>();
            report.gram = gram.iter().map(dec).collect();
            report.rhs = rhs.iter().map(dec).collect();
            if let Some((coefficients, pivot)) = solve_small(&gram, &rhs, p) {
                let mut residual = h;
                for (a, v) in coefficients.iter().zip(&vectors) {
                    for (x, y) in residual.iter_mut().zip(v) {
                        *x -= Float::with_val(p, a * y);
                    }
                }
                report.fit_residual_norm_squared = Some(dec(&(norm2(&residual, p) * &l)));
                report.minimum_pivot = Some(dec(&pivot));
                if let Some(b) = &o.fixed_second_component {
                    report.b2 =
                        Some(dec(&(Float::with_val(p, &coefficients[1])
                            - scalar(b, p)? * &coefficients[0])));
                    if coefficients[0] != 0 {
                        report.b_effective = Some(dec(
                            &(Float::with_val(p, &coefficients[1]) / &coefficients[0])
                        ));
                    }
                }
                report.coefficients = Some(coefficients.iter().map(dec).collect());
                report.outcome = "point_measurement".into();
            } else {
                report.outcome = "rank_or_precision_unresolved".into();
            }
            Ok(report)
        },
        |r| {
            if ![
                "normalization_unresolved",
                "rank_or_precision_unresolved",
                "point_measurement",
            ]
            .contains(&r.outcome.as_str())
                || r.metric != "full_support_L2_dx; finite_Fourier_coefficients"
                || r.normalization != o.normalization
                || r.fixed_second_component != o.fixed_second_component
                || r.uncertainty
                    != "point_inputs; no_reference_approximation_or_construction_error_enclosure"
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StabilizationRow {
    pub source: ContentDigest,
    pub n_modes: usize,
    pub value: String,
    pub relative_change: Option<String>,
    pub status: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    precision(o.working_precision_bits)?;
    let p = o.working_precision_bits;
    let tolerance = scalar(&o.relative_tolerance, p)?;
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
            || s.cutoff != first.cutoff
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
        json!({"options":o,"ordered_sources":states.iter().map(|s|&s.manifest.content_digest).collect::<Vec<_>>()}),
        &states
            .iter()
            .map(|s| s.manifest.clone())
            .collect::<Vec<_>>(),
        c,
        || {
            let mut rows = Vec::new();
            let mut previous: Option<Float> = None;
            let mut streak = 0;
            for s in states {
                let value = scalar(&s.eigenvalue, p)?;
                let change = previous.as_ref().and_then(|old| {
                    if old == &0 {
                        None
                    } else {
                        Some(((Float::with_val(p, &value) - old) / old).abs())
                    }
                });
                let status = if previous.is_none() {
                    "initial"
                } else if change.is_none() {
                    "zero_denominator"
                } else if change.as_ref().unwrap() <= &tolerance {
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
        },
        |r| {
            if r.lambda_squared!=first.cutoff||r.rows.len()!=states.len()||r.observable!="serialized_retained_Weil_eigenvalue"||r.qualification!="finite_frozen_rule_only; branch_and_assembly_policy_comparability_not_established; not_N_infinity"||!["finite_rule_met","finite_rule_not_met"].contains(&r.outcome.as_str()){bail!("invalid stabilization report");}
            for (row, s) in r.rows.iter().zip(states) {
                if row.source != s.manifest.content_digest
                    || row.n_modes != s.modes
                    || row.value != s.eigenvalue
                    || !["initial", "zero_denominator", "within_rule", "outside_rule"]
                        .contains(&row.status.as_str())
                {
                    bail!("stabilization source mismatch");
                }
                if let Some(v) = &row.relative_change {
                    if scalar(v, p)? < 0 {
                        bail!("negative relative change");
                    }
                }
            }
            Ok(())
        },
    )
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
