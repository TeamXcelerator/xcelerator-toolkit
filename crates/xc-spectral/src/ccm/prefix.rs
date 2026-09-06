// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Source-bound prefix diagnostics. Source acquisition is intentionally absent:
//! callers supply approved, retained payloads and their immutable manifests.
//! This module cannot build a missing Tau matrix or replace an eigenstate.
//! Mathematical construction and the caller's trust in a source manifest are
//! separate from checking the payload's exact byte digest here.

use anyhow::{bail, Result};
use rug::Float;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use xc_cache::{
    resolve_or_compute_json_artifact_with_dependencies, ArtifactCacheContext,
    ArtifactExecutionCacheRequest, ArtifactExecutionCacheResult, ArtifactManifest, CacheError,
    CacheQuality, CacheVisibility, ContentDigest, DependencyRef, SemanticKeyEnvelope,
    ToolkitVersion,
};
use xc_numerics::prefix::{
    analyze_prefixes, checked_decimal_export, lossless_decimal, PrefixAnalysisReport,
};
use xc_numerics::reduction::deterministic_pairwise_sum_hp_owned;

pub const PREFIX_SEMANTICS: &str = "ccm-retained-even-prefix-moments-checked-exports-v1";
pub const PREFIX_ARTIFACT_KIND: &str = "ccm_prefix_analysis";
pub const EVEN_BASIS: &str = "orthonormal_reflection_even_basis_zero_then_positive_modes";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixAnalysisOptions {
    pub working_precision_bits: u32,
    pub pivot_margin_bits: u32,
    /// Generic dimension k=N+1, NOT the CCM mode index N.
    pub checkpoint_dimensions: Vec<usize>,
    pub export_significant_digits: Vec<usize>,
    /// Relative backward-error/norm tolerance for the actual decoded packet.
    /// This is an export acceptance policy, not an assembly error bound.
    pub export_relative_tolerance: String,
}

#[derive(Clone, Debug)]
pub struct RetainedEvenMatrix {
    manifest: ArtifactManifest,
    cutoff: String,
    modes: usize,
    precision: u32,
    entries: Vec<Float>,
}

fn authenticate(
    manifest: &ArtifactManifest,
    bytes: &[u8],
    allowed: &[ContentDigest],
) -> Result<()> {
    manifest.validate()?;
    if manifest.quality.admissible_rank() < CacheQuality::Validated.admissible_rank()
        || !manifest.immutable
        || manifest.size_bytes != bytes.len() as u64
        || ContentDigest::sha256(bytes) != manifest.content_digest
        || !allowed.contains(&manifest.content_digest)
    {
        bail!("source must match an explicitly approved immutable payload digest and byte count");
    }
    Ok(())
}

fn scalar(text: &str, p: u32) -> Result<Float> {
    let value = Float::with_val(p, Float::parse(text)?);
    if !value.is_finite() {
        bail!("nonfinite diagnostic/source scalar");
    }
    Ok(value)
}

fn precision(p: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&p) {
        bail!("unsupported diagnostic precision");
    }
    Ok(())
}

impl RetainedEvenMatrix {
    /// Decode the existing ccm_even_sector_matrix payload without rewriting it.
    /// The allowlist must come from the run/campaign's source-selection policy.
    pub fn from_payload(
        manifest: &ArtifactManifest,
        bytes: &[u8],
        allowed: &[ContentDigest],
    ) -> Result<Self> {
        authenticate(manifest, bytes, allowed)?;
        if manifest.key.kind != "ccm_even_sector_matrix" {
            bail!("an ordered even-sector matrix is required");
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Payload {
            schema_version: u32,
            lambda_squared: String,
            n_modes: usize,
            precision_bits: u32,
            dimension: usize,
            entries: Vec<String>,
        }
        let payload: Payload = serde_json::from_slice(bytes)?;
        precision(payload.precision_bits)?;
        if payload.schema_version != 1
            || payload.dimension == 0
            || payload.dimension > 8193
            || payload.n_modes.checked_add(1) != Some(payload.dimension)
            || payload.dimension.checked_mul(payload.dimension) != Some(payload.entries.len())
            || payload.lambda_squared.trim().is_empty()
        {
            bail!("invalid retained even-matrix shape or metadata");
        }
        let entries = payload
            .entries
            .iter()
            .map(|v| scalar(v, payload.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..payload.dimension {
            for j in 0..i {
                if entries[i * payload.dimension + j] != entries[j * payload.dimension + i] {
                    bail!("source storage is not exactly symmetric");
                }
            }
        }
        Ok(Self {
            manifest: manifest.clone(),
            cutoff: payload.lambda_squared,
            modes: payload.n_modes,
            precision: payload.precision_bits,
            entries,
        })
    }
    pub fn dimension(&self) -> usize {
        self.modes + 1
    }
    pub fn entries(&self) -> &[Float] {
        &self.entries
    }
    pub fn source_precision_bits(&self) -> u32 {
        self.precision
    }
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }
}

#[derive(Clone, Debug)]
pub struct RetainedEvenEigenpair {
    manifest: ArtifactManifest,
    cutoff: String,
    modes: usize,
    precision: u32,
    eigenvalue: Float,
    vector: Vec<Float>,
}
impl RetainedEvenEigenpair {
    /// Preserve the existing full coefficient source; make an even-coordinate
    /// COPY only after exact reflection symmetry has been checked.
    pub fn from_payload(
        manifest: &ArtifactManifest,
        bytes: &[u8],
        allowed: &[ContentDigest],
    ) -> Result<Self> {
        authenticate(manifest, bytes, allowed)?;
        if manifest.key.kind != "ccm_weil_eigenpair" {
            bail!("a retained Weil eigenpair is required");
        }
        #[derive(Deserialize)]
        struct Payload {
            schema_version: u32,
            lambda_squared: String,
            n_modes: usize,
            precision_bits: u32,
            eigenvalue: String,
            eigenvector: Vec<String>,
        }
        let v: Payload = serde_json::from_slice(bytes)?;
        precision(v.precision_bits)?;
        if ![2, 3].contains(&v.schema_version)
            || v.n_modes > 8192
            || v.n_modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(v.eigenvector.len())
        {
            bail!("invalid eigenpair shape or schema");
        }
        let vector = v
            .eigenvector
            .iter()
            .map(|x| scalar(x, v.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..v.n_modes {
            if vector[i] != vector[2 * v.n_modes - i] {
                bail!("checkpoint eigenstate is not exactly even; no silent parity projection");
            }
        }
        if vector.iter().all(Float::is_zero) {
            bail!("zero eigenvector");
        }
        Ok(Self {
            manifest: manifest.clone(),
            cutoff: v.lambda_squared,
            modes: v.n_modes,
            precision: v.precision_bits,
            eigenvalue: scalar(&v.eigenvalue, v.precision_bits)?,
            vector,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointExport {
    pub dimension: usize,
    pub n_modes: usize,
    pub status: String,
    pub eigenpair_source: Option<ContentDigest>,
    pub eigenpair_precision_bits: Option<u32>,
    pub accepted_significant_digits: Option<usize>,
    pub raw_innovation: Vec<String>,
    pub unit_innovation: Vec<String>,
    pub unit_retained_eigenvector: Vec<String>,
    pub signed_overlap: Option<String>,
    pub squared_overlap: Option<String>,
    pub decoded_innovation_backward_error: Option<String>,
    pub decoded_eigenpair_backward_error: Option<String>,
    pub eigenpair_residual_matrix: String,
    pub sign_convention: String,
    pub diagnostic: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmPrefixAnalysis {
    pub schema_version: u32,
    pub semantics: String,
    pub basis: String,
    pub parent_matrix_source: ContentDigest,
    pub parent_n_modes: usize,
    pub cutoff: String,
    pub source_precision_bits: u32,
    pub options: PrefixAnalysisOptions,
    /// This is a prefix of THIS parent, not an asserted canonical smaller matrix.
    pub prefixes_are_parent_derived: bool,
    pub ladder: PrefixAnalysisReport,
    pub checkpoints: Vec<CheckpointExport>,
    pub assurance: String,
}

fn dot(a: &[Float], b: &[Float], p: u32) -> Float {
    deterministic_pairwise_sum_hp_owned(
        a.iter()
            .zip(b)
            .map(|(a, b)| {
                let mut v = Float::with_val(p, a);
                v *= b;
                v
            })
            .collect(),
        p,
    )
}
fn norm(a: &[Float], p: u32) -> Float {
    dot(a, a, p).sqrt()
}
fn unit(v: &[Float], p: u32) -> Result<Vec<Float>> {
    let n = norm(v, p);
    if !n.is_finite() || n.is_zero() {
        bail!("unresolved vector normalization");
    }
    Ok(v.iter()
        .map(|x| {
            let mut v = Float::with_val(p, x);
            v /= &n;
            v
        })
        .collect())
}
fn residual(a: &[Float], stride: usize, v: &[Float], rhs: &[Float], p: u32) -> Float {
    let n = v.len();
    let r = (0..n)
        .map(|i| {
            let mut x = dot(&a[i * stride..i * stride + n], v, p);
            x -= &rhs[i];
            x
        })
        .collect::<Vec<_>>();
    let row_squares = (0..n)
        .map(|i| {
            let row = &a[i * stride..i * stride + n];
            dot(row, row, p)
        })
        .collect();
    let mut scale = deterministic_pairwise_sum_hp_owned(row_squares, p).sqrt();
    scale *= norm(v, p);
    scale += norm(rhs, p);
    let mut result = norm(&r, p);
    if !scale.is_zero() {
        result /= &scale;
    }
    // Failed normalization must not pass a `NaN > tolerance` comparison.
    if !result.is_finite() || !scale.is_finite() {
        return Float::with_val(p, rug::float::Special::Infinity);
    }
    result
}

/// O(D^3) prefix analysis; only explicitly supplied retained eigenstates are
/// exported. No source mutation, numerical solver invocation, or network access.
pub fn analyze_retained_prefixes(
    matrix: &RetainedEvenMatrix,
    options: &PrefixAnalysisOptions,
    eigenpairs: &[RetainedEvenEigenpair],
) -> Result<CcmPrefixAnalysis> {
    let p = options.working_precision_bits;
    precision(p)?;
    if p < matrix.precision {
        bail!("analysis cannot down-round the retained source");
    }
    let tolerance = scalar(&options.export_relative_tolerance, p)?;
    if tolerance <= 0 || tolerance >= 1 {
        bail!("export tolerance must be in (0,1)");
    }
    // Validate the schedule even for an empty checkpoint set.
    checked_decimal_export(
        &[Float::with_val(p, 1)],
        &options.export_significant_digits,
        |_| Ok(true),
    )?;
    let mut by_dimension = BTreeMap::new();
    for pair in eigenpairs {
        if pair.cutoff != matrix.cutoff
            || pair.precision > p
            || !options.checkpoint_dimensions.contains(&(pair.modes + 1))
            || by_dimension.insert(pair.modes + 1, pair).is_some()
        {
            bail!("mismatched, duplicate, or unrequested checkpoint eigenstate");
        }
    }
    let ladder = analyze_prefixes(
        &matrix.entries,
        matrix.dimension(),
        p,
        options.pivot_margin_bits,
        &options.checkpoint_dimensions,
    )?;
    let mut checkpoints = Vec::new();
    for &k in &options.checkpoint_dimensions {
        let mut packet=CheckpointExport {dimension:k,n_modes:k-1,status:"unresolved_prefix".into(),
            eigenpair_source:by_dimension.get(&k).map(|v|v.manifest.content_digest.clone()),
            eigenpair_precision_bits:by_dimension.get(&k).map(|v|v.precision),
            accepted_significant_digits:None,raw_innovation:vec![],unit_innovation:vec![],
            unit_retained_eigenvector:vec![],signed_overlap:None,squared_overlap:None,
            decoded_innovation_backward_error:None,decoded_eigenpair_backward_error:None,
            eigenpair_residual_matrix:"actual_largest_parent_prefix_not_asserted_canonical_checkpoint_matrix".into(),
            sign_convention:"innovation_last_coefficient_positive_one; eigenstate_largest_absolute_even_coefficient_positive_first_index_tie".into(), diagnostic:None};
        let Some(raw) = ladder.checkpoint_innovations.get(&k) else {
            checkpoints.push(packet);
            continue;
        };
        let innovation = raw
            .iter()
            .map(|v| scalar(v, p))
            .collect::<Result<Vec<_>>>()?;
        let normalized = unit(&innovation, p)?;
        let sigma = scalar(&ladder.rows[k - 1].sigma, p)?;
        let pair = by_dimension.get(&k);
        let mut xi = Vec::new();
        if let Some(pair) = pair {
            let sqrt2 = Float::with_val(p, 2).sqrt();
            xi.push(Float::with_val(p, &pair.vector[pair.modes]));
            for j in 1..k {
                let mut x = Float::with_val(p, &pair.vector[pair.modes + j]);
                x *= &sqrt2;
                xi.push(x);
            }
            xi = unit(&xi, p)?;
            let mut pivot = 0;
            for j in 1..k {
                if xi[j].clone().abs() > xi[pivot].clone().abs() {
                    pivot = j;
                }
            }
            if xi[pivot] < 0 {
                for x in &mut xi {
                    *x = -x.clone();
                }
            }
        }
        let mut values = innovation.clone();
        values.extend(normalized);
        values.extend(xi);
        // Use exactly the decoded packet in all norm, equation, and overlap tests.
        let checks = |decoded: &[Float]| -> Result<bool> {
            // The supplied eigenstate must remain the same vector after
            // printing, even when several small eigenvalues are unresolved.
            if decoded[k..]
                .iter()
                .zip(&values[k..])
                .any(|(a, b)| Float::with_val(p, a - b).abs() > tolerance)
            {
                return Ok(false);
            }
            let mut rhs = vec![Float::with_val(p, 0); k];
            rhs[k - 1] = sigma.clone();
            if decoded[k - 1] != 1 {
                return Ok(false);
            }
            let mut mass_error = norm(&decoded[k..2 * k], p);
            mass_error -= 1;
            mass_error.abs_mut();
            if mass_error > tolerance
                || residual(&matrix.entries, matrix.dimension(), &decoded[..k], &rhs, p) > tolerance
            {
                return Ok(false);
            }
            let renormalized = unit(&decoded[..k], p)?;
            if renormalized
                .iter()
                .zip(&decoded[k..2 * k])
                .any(|(a, b)| Float::with_val(p, a - b).abs() > tolerance)
            {
                return Ok(false);
            }
            if let Some(pair) = pair {
                let vector = &decoded[2 * k..];
                let mut error = norm(vector, p);
                error -= 1;
                error.abs_mut();
                let rhs = vector
                    .iter()
                    .map(|x| {
                        let mut v = Float::with_val(p, x);
                        v *= &pair.eigenvalue;
                        v
                    })
                    .collect::<Vec<_>>();
                if error > tolerance
                    || residual(&matrix.entries, matrix.dimension(), vector, &rhs, p) > tolerance
                {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match checked_decimal_export(&values, &options.export_significant_digits, checks) {
            Ok((digits, encoded)) => {
                let decoded = encoded
                    .iter()
                    .map(|s| scalar(s, p))
                    .collect::<Result<Vec<_>>>()?;
                let mut rhs = vec![Float::with_val(p, 0); k];
                rhs[k - 1] = sigma.clone();
                packet.decoded_innovation_backward_error = Some(lossless_decimal(&residual(
                    &matrix.entries,
                    matrix.dimension(),
                    &decoded[..k],
                    &rhs,
                    p,
                )));
                packet.status = if pair.is_some() {
                    "export_checks_passed"
                } else {
                    "innovation_export_passed_eigenpair_not_supplied"
                }
                .into();
                packet.accepted_significant_digits = Some(digits);
                packet.raw_innovation = encoded[..k].to_vec();
                packet.unit_innovation = encoded[k..2 * k].to_vec();
                if let Some(pair) = pair {
                    packet.unit_retained_eigenvector = encoded[2 * k..].to_vec();
                    let overlap = dot(&decoded[k..2 * k], &decoded[2 * k..], p);
                    packet.signed_overlap = Some(lossless_decimal(&overlap));
                    packet.squared_overlap = Some(lossless_decimal(&overlap.square()));
                    let rhs = decoded[2 * k..]
                        .iter()
                        .map(|x| {
                            let mut v = Float::with_val(p, x);
                            v *= &pair.eigenvalue;
                            v
                        })
                        .collect::<Vec<_>>();
                    packet.decoded_eigenpair_backward_error = Some(lossless_decimal(&residual(
                        &matrix.entries,
                        matrix.dimension(),
                        &decoded[2 * k..],
                        &rhs,
                        p,
                    )));
                }
            }
            Err(error) => {
                packet.status = "export_checks_unresolved".into();
                packet.diagnostic = Some(error.to_string());
            }
        }
        checkpoints.push(packet);
    }
    Ok(CcmPrefixAnalysis {
        schema_version: 1,
        semantics: PREFIX_SEMANTICS.into(),
        basis: EVEN_BASIS.into(),
        parent_matrix_source: matrix.manifest.content_digest.clone(),
        parent_n_modes: matrix.modes,
        cutoff: matrix.cutoff.clone(),
        source_precision_bits: matrix.precision,
        options: options.clone(),
        prefixes_are_parent_derived: true,
        ladder,
        checkpoints,
        assurance: "computed_point_diagnostics_and_export_checks_not_certified".into(),
    })
}

fn dependency(manifest: &ArtifactManifest) -> DependencyRef {
    DependencyRef {
        key: manifest.key.clone(),
        content_digest: manifest.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    }
}

/// Managed caching of NEW diagnostic children only. The separately supplied
/// approved sources are never rebuilt or rewritten here. Publication may reuse
/// their existing dependency closure without changing its identities.
/// This artifact kind is private-only. Registry registration is an explicit
/// deployment operation; this API never modifies a private registry.
pub fn analyze_retained_prefixes_via_cache(
    matrix: &RetainedEvenMatrix,
    options: &PrefixAnalysisOptions,
    eigenpairs: &[RetainedEvenEigenpair],
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<CcmPrefixAnalysis>> {
    if cache.write_visibility == CacheVisibility::Public
        || cache.requested_assurance != xc_core::AssuranceLevel::Computed
    {
        bail!("prefix diagnostics are private/local computed evidence, not certified outputs");
    }
    let mut dependencies = vec![dependency(&matrix.manifest)];
    dependencies.extend(eigenpairs.iter().map(|e| dependency(&e.manifest)));
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
        artifact_kind: PREFIX_ARTIFACT_KIND.into(),
        mathematical_semantics_version: PREFIX_SEMANTICS.into(),
        resolved_mathematical_parameters: serde_json::json!({"source_dependencies":dependencies,"options":options}),
        normalization: Some(EVEN_BASIS.into()),
        target: Some("parent_derived_prefix_moments_and_checkpoint_exports".into()),
        subspace: Some("even".into()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("unpivoted_ldlt_fixed_reduction_order".into()),
    };
    let logical = format!(
        "ccm/prefix/{}/{}",
        matrix.manifest.content_digest.0,
        matrix.dimension()
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.prefix.analyze_retained",
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
        minimum_reader_version: ToolkitVersion::parse("0.14.4")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".into(), "ccm".into()),
            ("assurance".into(), "computed_not_certified".into()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    let result = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            analyze_retained_prefixes(matrix, options, eigenpairs)
                .map(|r| (r, dependencies.clone()))
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))
        },
        |r| {
            if r.schema_version != 1
                || r.semantics != PREFIX_SEMANTICS
                || r.basis != EVEN_BASIS
                || r.parent_matrix_source != matrix.manifest.content_digest
                || r.options != *options
                || r.parent_n_modes != matrix.modes
                || r.cutoff != matrix.cutoff
                || r.source_precision_bits != matrix.precision
                || !r.prefixes_are_parent_derived
                || r.ladder.precision_bits != options.working_precision_bits
                || r.ladder.requested_dimension != matrix.dimension()
                || r.ladder.rows.len() > matrix.dimension()
                || (r.ladder.stopped.is_none() && r.ladder.rows.len() != matrix.dimension())
                || r.checkpoints
                    .iter()
                    .map(|r| r.dimension)
                    .collect::<Vec<_>>()
                    != options.checkpoint_dimensions
                || r.assurance != "computed_point_diagnostics_and_export_checks_not_certified"
            {
                return Err(CacheError::InvalidManifest(
                    "prefix diagnostic identity or shape mismatch".into(),
                ));
            }
            for (i, row) in r.ladder.rows.iter().enumerate() {
                if row.dimension != i + 1 {
                    return Err(CacheError::InvalidManifest(
                        "prefix rows are not ordered".into(),
                    ));
                }
                for value in [
                    &row.sigma,
                    &row.innovation_mass,
                    &row.inverse_trace,
                    &row.inverse_square_trace,
                ] {
                    if scalar(value, options.working_precision_bits)
                        .map_err(|e| CacheError::InvalidManifest(e.to_string()))?
                        <= 0
                    {
                        return Err(CacheError::InvalidManifest(
                            "nonpositive prefix metric".into(),
                        ));
                    }
                }
            }
            Ok(())
        },
    )?;
    if let Some(manifest) = result
        .produced_manifest
        .as_ref()
        .or(result.reused_manifest.as_ref())
    {
        if manifest.dependencies != dependencies {
            bail!("cached prefix source dependency bindings differ");
        }
    }
    Ok(result)
}
