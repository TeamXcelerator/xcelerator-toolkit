// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.
//! Source-bound geometry of the actual retained Fourier state. No eigensolve,
//! target function, reference zeros, or assertion of ground-state selection.
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Constant, Assign, Float};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use xc_cache::{
    resolve_or_compute_json_artifact_with_dependencies, ArtifactCacheContext,
    ArtifactExecutionCacheRequest, ArtifactExecutionCacheResult, ArtifactManifest, CacheError,
    CacheQuality, CacheVisibility, ContentDigest, DependencyRef, SemanticKeyEnvelope,
    ToolkitVersion,
};
use xc_numerics::prefix::lossless_decimal;

pub const ARTIFACT_KIND: &str = "ccm_state_geometry_analysis";
pub const SEMANTICS: &str = "ccm-retained-fourier-state-geometry-v1";
const CONVENTION: &str = "x=log(u); L=log(lambda_squared); sum_j xi_j exp(2*pi*i*j*(x/L+1/2)); unit_L2_dx; center_positive_else_largest_coefficient_positive";
const ASSURANCE: &str = "computed_point_diagnostics; source_error_and_quadrature_error_not_enclosed; no_ground_selection_or_global_sign_certificate";

fn scalar(s: &str, p: u32) -> Result<Float> {
    let value = Float::with_val(p, Float::parse(s)?);
    if !value.is_finite() {
        bail!("nonfinite state geometry scalar");
    }
    Ok(value)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryOptions {
    pub working_precision_bits: u32,
    /// Refined grid has twice this many intervals. Must resolve all retained modes.
    pub base_intervals: usize,
}
impl GeometryOptions {
    pub fn for_source(source: &RetainedState) -> Self {
        Self {
            working_precision_bits: source.precision,
            base_intervals: 8 * (source.modes + 1),
        }
    }
    fn validate(&self, source: &RetainedState) -> Result<()> {
        if !(source.precision..=1_000_000).contains(&self.working_precision_bits)
            || self.base_intervals < 8 * (source.modes + 1)
            || self.base_intervals > 131072
            || !self.base_intervals.is_multiple_of(8)
        {
            bail!("state geometry precision or grid budget is invalid");
        }
        Ok(())
    }
}
/// Authenticated logical payload, not a profile that discarded its raw normalizer.
#[derive(Clone, Debug)]
pub struct RetainedState {
    pub(crate) manifest: ArtifactManifest,
    pub(crate) cutoff: String,
    pub(crate) modes: usize,
    pub(crate) precision: u32,
    pub(crate) coefficients: Vec<Float>,
    pub(crate) eigenvalue: String,
    pub(crate) selection_policy: Option<String>,
}
impl RetainedState {
    pub fn from_payload(
        manifest: &ArtifactManifest,
        bytes: &[u8],
        approved: &[ContentDigest],
    ) -> Result<Self> {
        manifest.validate()?;
        if manifest.key.kind != "ccm_weil_eigenpair"
            || !manifest.immutable
            || manifest.quality.admissible_rank() < CacheQuality::Validated.admissible_rank()
            || manifest.size_bytes != bytes.len() as u64
            || ContentDigest::sha256(bytes) != manifest.content_digest
            || !approved.contains(&manifest.content_digest)
        {
            bail!("state geometry requires an approved immutable validated eigenpair payload");
        }
        #[derive(Deserialize)]
        struct Payload {
            schema_version: u32,
            lambda_squared: String,
            n_modes: usize,
            precision_bits: u32,
            pub(crate) eigenvalue: String,
            eigenvector: Vec<String>,
            force_even: Option<bool>,
            parity_policy: Option<serde_json::Value>,
        }
        let v: Payload = serde_json::from_slice(bytes)?;
        if ![2, 3].contains(&v.schema_version)
            || v.n_modes > 8192
            || !(64..=1_000_000).contains(&v.precision_bits)
            || v.eigenvector.len() != 2 * v.n_modes + 1
            || scalar(&v.lambda_squared, v.precision_bits)? <= 1
        {
            bail!("unsupported retained state shape, precision, or cutoff");
        }
        scalar(&v.eigenvalue, v.precision_bits)?;
        let coefficients = v
            .eigenvector
            .iter()
            .map(|s| scalar(s, v.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        if coefficients.iter().all(|x| x == &0) {
            bail!("zero retained state");
        }
        Ok(Self {
            manifest: manifest.clone(),
            cutoff: v.lambda_squared,
            modes: v.n_modes,
            precision: v.precision_bits,
            coefficients,
            eigenvalue: v.eigenvalue,
            selection_policy: v.force_even.map(|even| {
                format!(
                    "force_even={even};parity={}",
                    v.parity_policy
                        .map_or("legacy".to_owned(), |v| v.to_string())
                )
            }),
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryGrid {
    pub intervals: usize,
    pub l2_mass: String,
    /// Integral x^j |f(x)|^2 dx on [-L/2,L/2], j=1,2,4.
    pub spatial_moments: Vec<String>,
    /// Periodic trapezoid approximations to mass outside |x| >= L/8,L/4,3L/8.
    pub outer_shell_masses: Vec<String>,
    pub sampled_real_minimum: Option<String>,
    /// Integral max(-f,0) dx, available only for exactly even real coefficients.
    pub sampled_negative_part_l1: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateGeometryAnalysis {
    pub schema_version: u32,
    pub semantics: String,
    pub source: ContentDigest,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub source_precision_bits: u32,
    pub source_eigenvalue: String,
    pub options: GeometryOptions,
    pub convention: String,
    pub assurance: String,
    pub physical_sign_status: String,
    pub coefficient_norm: String,
    pub raw_center: String,
    pub unit_l2_center: String,
    pub unit_l2_signed_mass: Option<String>,
    pub coefficient_evenness_defect: String,
    pub orientation: i32,
    pub coarse: GeometryGrid,
    pub refined: GeometryGrid,
    /// Absolute fine-minus-coarse differences: mass, moments 1/2/4, shells 1/2/3.
    /// These are sensitivity measurements, never rigorous error bounds.
    pub refinement_absolute_differences: Vec<String>,
}
fn grid(
    source: &RetainedState,
    options: &GeometryOptions,
    intervals: usize,
    scale: &Float,
    even: bool,
) -> Result<GeometryGrid> {
    let p = options.working_precision_bits;
    let l = scalar(&source.cutoff, p)?.ln();
    let two_pi = Float::with_val(p, Constant::Pi) * 2;
    let pairs: Vec<_> = (1..=source.modes)
        .map(|n| {
            (
                Float::with_val(p, &source.coefficients[source.modes + n])
                    + &source.coefficients[source.modes - n],
                Float::with_val(p, &source.coefficients[source.modes + n])
                    - &source.coefficients[source.modes - n],
            )
        })
        .collect();
    // Fixed chunks and fixed merge order make results independent of worker count.
    // Only chunk summaries are retained, not a potentially huge sampled profile.
    let chunks: Vec<_> = (0..intervals.div_ceil(64))
        .into_par_iter()
        .map(|chunk| {
            let mut sums = vec![Float::with_val(p, 0); 8];
            let mut minimum: Option<Float> = None;
            for j in chunk * 64..((chunk + 1) * 64).min(intervals) {
                // Periodic trapezoid nodes include the center. Shell boundaries receive
                // half weight, matching their piecewise trapezoid interpretation.
                let t = Float::with_val(p, j) / intervals;
                let x = (Float::with_val(p, &t) - Float::with_val(p, 0.5)) * &l;
                let angle = Float::with_val(p, &two_pi) * &t;
                let (sin, cos) = angle.sin_cos(Float::new(p));
                let mut cn = Float::with_val(p, 1);
                let mut sn = Float::with_val(p, 0);
                let mut re = Float::with_val(p, &source.coefficients[source.modes]);
                let mut im = Float::with_val(p, 0);
                let mut next_cos = Float::new(p);
                let mut next_sin = Float::new(p);
                for (pair, difference) in &pairs {
                    next_cos.assign(&cn * &cos);
                    next_cos -= &sn * &sin;
                    next_sin.assign(&sn * &cos);
                    next_sin += &cn * &sin;
                    cn.assign(&next_cos);
                    sn.assign(&next_sin);
                    re += pair * &cn;
                    im += difference * &sn;
                }
                re *= scale;
                im *= scale;
                let energy = Float::with_val(p, &re * &re) + Float::with_val(p, &im * &im);
                sums[0] += &energy;
                // Average the two endpoint values for the nonperiodic odd moment.
                // All even moments and the Fourier energy have equal endpoints.
                if j != 0 {
                    sums[1] += Float::with_val(p, &energy * &x);
                }
                let x2 = Float::with_val(p, &x * &x);
                sums[2] += Float::with_val(p, &energy * &x2);
                sums[3] += Float::with_val(p, &energy) * x2.square();
                let distance = j.abs_diff(intervals / 2);
                for h in 1..=3 {
                    let boundary = intervals * h / 8;
                    if distance > boundary {
                        sums[3 + h] += &energy;
                    } else if distance == boundary {
                        sums[3 + h] += Float::with_val(p, &energy) / 2;
                    }
                }
                if even {
                    if minimum.as_ref().is_none_or(|m| re < *m) {
                        minimum = Some(re.clone());
                    }
                    if re < 0 {
                        sums[7] -= re;
                    }
                }
            }
            (sums, minimum)
        })
        .collect();
    let mut sums = vec![Float::with_val(p, 0); 8];
    let mut minimum: Option<Float> = None;
    for (values, m) in chunks {
        for (sum, value) in sums.iter_mut().zip(values) {
            *sum += value;
        }
        if let Some(m) = m {
            if minimum.as_ref().is_none_or(|a| m < *a) {
                minimum = Some(m);
            }
        }
    }
    for sum in &mut sums {
        *sum *= &l;
        *sum /= intervals;
    }
    Ok(GeometryGrid {
        intervals,
        l2_mass: lossless_decimal(&sums[0]),
        spatial_moments: sums[1..4].iter().map(lossless_decimal).collect(),
        outer_shell_masses: sums[4..7].iter().map(lossless_decimal).collect(),
        sampled_real_minimum: minimum.as_ref().map(lossless_decimal),
        sampled_negative_part_l1: even.then(|| lossless_decimal(&sums[7])),
    })
}
fn grid_values(g: &GeometryGrid) -> Vec<&str> {
    std::iter::once(g.l2_mass.as_str())
        .chain(g.spatial_moments.iter().map(String::as_str))
        .chain(g.outer_shell_masses.iter().map(String::as_str))
        .collect()
}
/// Analyze the actual retained state with unit L2(dx) normalization. No replacement source.
pub fn analyze_state_geometry(
    source: &RetainedState,
    options: &GeometryOptions,
) -> Result<StateGeometryAnalysis> {
    options.validate(source)?;
    let p = options.working_precision_bits;
    let mut norm2 = Float::with_val(p, 0);
    let mut center = Float::with_val(p, &source.coefficients[source.modes]);
    let mut defect = Float::with_val(p, 0);
    for v in &source.coefficients {
        norm2 += Float::with_val(p, v * v);
    }
    for n in 1..=source.modes {
        let a = &source.coefficients[source.modes + n];
        let b = &source.coefficients[source.modes - n];
        let pair = Float::with_val(p, a) + b;
        if n.is_multiple_of(2) {
            center += pair;
        } else {
            center -= pair;
        }
        defect += (Float::with_val(p, a) - b).square() * 2;
    }
    if !norm2.is_finite() || norm2 <= 0 || !center.is_finite() || !defect.is_finite() {
        bail!("state normalization exceeds the arithmetic range");
    }
    let even = source
        .coefficients
        .iter()
        .eq(source.coefficients.iter().rev());
    let orient_value = if center == 0 {
        source
            .coefficients
            .iter()
            .max_by(|a, b| (*a).clone().abs().total_cmp(&(*b).clone().abs()))
            .unwrap()
    } else {
        &center
    };
    let orientation = if orient_value < &0 { -1 } else { 1 };
    let l = scalar(&source.cutoff, p)?.ln();
    let scale = Float::with_val(p, orientation) / (Float::with_val(p, &norm2) * &l).sqrt();
    let coarse = grid(source, options, options.base_intervals, &scale, even)?;
    let refined = grid(source, options, 2 * options.base_intervals, &scale, even)?;
    let differences = grid_values(&coarse)
        .iter()
        .zip(grid_values(&refined))
        .map(|(a, b)| Ok(lossless_decimal(&(scalar(b, p)? - scalar(a, p)?).abs())))
        .collect::<Result<Vec<_>>>()?;
    let report = StateGeometryAnalysis {
        schema_version: 1,
        semantics: SEMANTICS.into(),
        source: source.manifest.content_digest.clone(),
        lambda_squared: source.cutoff.clone(),
        n_modes: source.modes,
        source_precision_bits: source.precision,
        source_eigenvalue: source.eigenvalue.clone(),
        options: options.clone(),
        convention: CONVENTION.into(),
        assurance: ASSURANCE.into(),
        physical_sign_status: if even {
            "sampled_real_state_only"
        } else {
            "unavailable_nonreal_fourier_state"
        }
        .into(),
        coefficient_norm: lossless_decimal(&norm2.clone().sqrt()),
        raw_center: lossless_decimal(&center),
        unit_l2_center: lossless_decimal(&(center * &scale)),
        unit_l2_signed_mass: even.then(|| {
            lossless_decimal(
                &(Float::with_val(p, &source.coefficients[source.modes]) * &l * &scale),
            )
        }),
        coefficient_evenness_defect: lossless_decimal(&(defect / norm2).sqrt()),
        orientation,
        coarse,
        refined,
        refinement_absolute_differences: differences,
    };
    validate_report(&report, source, options)?;
    Ok(report)
}
fn validate_report(
    r: &StateGeometryAnalysis,
    s: &RetainedState,
    o: &GeometryOptions,
) -> Result<()> {
    o.validate(s)?;
    let even = s.coefficients.iter().eq(s.coefficients.iter().rev());
    if r.schema_version != 1
        || r.semantics != SEMANTICS
        || r.source != s.manifest.content_digest
        || r.lambda_squared != s.cutoff
        || r.n_modes != s.modes
        || r.source_precision_bits != s.precision
        || r.source_eigenvalue != s.eigenvalue
        || r.options != *o
        || r.convention != CONVENTION
        || r.assurance != ASSURANCE
        || r.physical_sign_status
            != if even {
                "sampled_real_state_only"
            } else {
                "unavailable_nonreal_fourier_state"
            }
        || ![-1, 1].contains(&r.orientation)
        || r.unit_l2_signed_mass.is_some() != even
        || r.refinement_absolute_differences.len() != 7
    {
        bail!("state geometry identity or scope mismatch");
    }
    for v in [
        &r.coefficient_norm,
        &r.raw_center,
        &r.unit_l2_center,
        &r.coefficient_evenness_defect,
    ]
    .into_iter()
    .chain(r.unit_l2_signed_mass.iter())
    .chain(r.refinement_absolute_differences.iter())
    {
        scalar(v, o.working_precision_bits)?;
    }
    for (g, n) in [
        (&r.coarse, o.base_intervals),
        (&r.refined, 2 * o.base_intervals),
    ] {
        if g.intervals != n
            || g.spatial_moments.len() != 3
            || g.outer_shell_masses.len() != 3
            || g.sampled_real_minimum.is_some() != even
            || g.sampled_negative_part_l1.is_some() != even
        {
            bail!("state geometry grid shape mismatch");
        }
        for v in grid_values(g)
            .into_iter()
            .chain(g.sampled_real_minimum.as_deref())
            .chain(g.sampled_negative_part_l1.as_deref())
        {
            scalar(v, o.working_precision_bits)?;
        }
        if scalar(&g.l2_mass, o.working_precision_bits)? <= 0 {
            bail!("invalid state geometry mass");
        }
        for v in g
            .outer_shell_masses
            .iter()
            .chain(g.sampled_negative_part_l1.iter())
        {
            if scalar(v, o.working_precision_bits)? < 0 {
                bail!("negative mass");
            }
        }
    }
    if scalar(&r.coefficient_evenness_defect, o.working_precision_bits)? < 0
        || r.refinement_absolute_differences
            .iter()
            .any(|v| scalar(v, o.working_precision_bits).is_ok_and(|v| v < 0))
    {
        bail!("negative geometry defect or refinement difference");
    }
    for (i, (coarse, refined)) in grid_values(&r.coarse)
        .iter()
        .zip(grid_values(&r.refined))
        .enumerate()
    {
        let difference = (scalar(refined, o.working_precision_bits)?
            - scalar(coarse, o.working_precision_bits)?)
        .abs();
        if scalar(
            &r.refinement_absolute_differences[i],
            o.working_precision_bits,
        )? != difference
        {
            bail!("geometry refinement difference is inconsistent with retained grids");
        }
    }
    if scalar(&r.coefficient_norm, o.working_precision_bits)? <= 0 {
        bail!("invalid state geometry norm");
    }
    Ok(())
}
/// Cache only the new child; its primary source has already been authenticated.
pub fn analyze_state_geometry_via_cache(
    source: &RetainedState,
    options: &GeometryOptions,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<StateGeometryAnalysis>> {
    options.validate(source)?;
    if cache.requested_assurance != xc_core::AssuranceLevel::Computed {
        bail!("state geometry is computed evidence, not a certificate");
    }
    let public = source.manifest.visibility == CacheVisibility::Public;
    if cache.write_visibility == CacheVisibility::Public && !public {
        bail!("public state geometry requires a public source manifest");
    }
    let dependencies = vec![DependencyRef {
        key: source.manifest.key.clone(),
        content_digest: source.manifest.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    }];
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: ARTIFACT_KIND.into(),
        mathematical_semantics_version: SEMANTICS.into(),
        resolved_mathematical_parameters: serde_json::json!({"source_dependencies":dependencies,"options":options,"source_parents_are_public":public}),
        normalization: Some(CONVENTION.into()),
        target: Some("actual_retained_state_geometry".into()),
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("periodic_trapezoid_two_grids_fixed_chunks_v1".into()),
    };
    let logical = format!("ccm/state-geometry/{}", source.manifest.content_digest.0);
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.state_geometry.analyze_retained",
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
        tags: BTreeMap::from([
            ("domain".into(), "ccm".into()),
            ("assurance".into(), "computed_not_certified".into()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    Ok(resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            analyze_state_geometry(source, options)
                .map(|r| (r, dependencies.clone()))
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))
        },
        |r| {
            validate_report(r, source, options)
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))
        },
    )?)
}
