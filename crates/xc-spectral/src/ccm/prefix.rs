// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Source-bound prefix diagnostics. Source acquisition is intentionally absent:
//! callers supply approved, retained payloads and their immutable manifests.
//! This module cannot build a missing Tau matrix or replace an eigenstate.
//! Mathematical construction and the caller's trust in a source manifest are
//! separate from checking the payload's exact byte digest here.

mod numerical_checks;

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
    analyze_prefixes_with_policy, checked_decimal_export, lossless_decimal, PrefixAnalysisReport,
};

pub const PREFIX_SEMANTICS: &str = "ccm-retained-even-prefix-moments-checked-exports-v8";
pub const LEGACY_PREFIX_SEMANTICS: &str = "ccm-retained-even-prefix-moments-checked-exports-v1";
pub const EXTENDED_PREFIX_SEMANTICS: &str = "ccm-retained-even-prefix-moments-checked-exports-v10";
pub use xc_core::PrefixDiagnosticPolicy;
pub const PREFIX_ARTIFACT_KIND: &str = "ccm_prefix_analysis";
pub const RETAINED_REDUCTION_SEMANTICS: &str = "ccm-retained-reduction-v0.15.1-v3";
const REDUCTION_ASSURANCE: &str = "computed stored-matrix checks; no construction, branch, positivity or continuum certificate; Q computed but not retained in this report";
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
    #[serde(
        default,
        skip_serializing_if = "PrefixDiagnosticPolicy::is_legacy_default"
    )]
    pub diagnostics: PrefixDiagnosticPolicy,
}

fn prefix_semantics(options: &PrefixAnalysisOptions) -> &'static str {
    if options.diagnostics.is_legacy_default() {
        PREFIX_SEMANTICS
    } else {
        EXTENDED_PREFIX_SEMANTICS
    }
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
    if bytes.len() as u128 > (2u128 << 30) {
        bail!("retained prefix payload exceeds the encoded byte budget");
    }
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
    precision(p)?;
    if text.len() > 1_048_576 {
        bail!("prefix scalar exceeds the input budget");
    }
    super::retained_evidence::scalar(text, p)
}
fn validate_cutoff(text: &str) -> Result<()> {
    if text.len() > 1_048_576
        || xc_core::DecimalLiteral::new(text)?.cmp_numeric(&xc_core::DecimalLiteral::new("1")?)?
            != std::cmp::Ordering::Greater
    {
        bail!("retained prefix cutoff must be an exact finite decimal greater than one");
    }
    Ok(())
}
fn decode_budget(count: usize, p: u32, encoded_bytes: usize) -> Result<()> {
    precision(p)?;
    let bytes = count as u128 * (u128::from(p).div_ceil(8) + 96) * 2 + encoded_bytes as u128 * 3;
    if bytes > (8u128 << 30) {
        bail!("retained prefix decoding exceeds the combined workspace budget");
    }
    Ok(())
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
        validate_cutoff(&payload.lambda_squared)?;
        decode_budget(payload.entries.len(), payload.precision_bits, bytes.len())?;
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
    pub(crate) fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

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
        if ![2, 3, 4, 5].contains(&v.schema_version)
            || v.n_modes > 8192
            || v.n_modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(v.eigenvector.len())
        {
            bail!("invalid eigenpair shape or schema");
        }
        validate_cutoff(&v.lambda_squared)?;
        decode_budget(v.eigenvector.len(), v.precision_bits, bytes.len())?;
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
    /// Optional comparisons of explicitly supplied retained point sources.
    /// Equality does not prove assembly accuracy or an infinite nesting theorem.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nesting_checks: Vec<PrefixNestingCheck>,
    pub ladder: PrefixAnalysisReport,
    pub checkpoints: Vec<CheckpointExport>,
    pub assurance: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixNestingCheck {
    pub semantics: String,
    pub smaller_source: ContentDigest,
    pub larger_source: ContentDigest,
    pub smaller_key: xc_cache::ArtifactKey,
    pub larger_key: xc_cache::ArtifactKey,
    pub smaller_manifest_provenance: Option<ContentDigest>,
    pub larger_manifest_provenance: Option<ContentDigest>,
    pub smaller_dimension: usize,
    pub larger_dimension: usize,
    pub smaller_precision_bits: u32,
    pub larger_precision_bits: u32,
    pub entries_compared: usize,
    pub mismatching_entries: usize,
    pub first_mismatch: Option<[usize; 2]>,
    pub exactly_nested: bool,
    pub assurance: String,
}

/// Compare decoded stored binary points exactly, at their native precisions.
/// Matching a block establishes equality of these two retained inputs only.
pub fn check_prefix_nesting(
    smaller: &RetainedEvenMatrix,
    larger: &RetainedEvenMatrix,
) -> Result<PrefixNestingCheck> {
    if smaller.dimension() > larger.dimension()
        || xc_core::DecimalLiteral::new(&smaller.cutoff)?.canonical()?
            != xc_core::DecimalLiteral::new(&larger.cutoff)?.canonical()?
    {
        bail!("prefix nesting requires the same exact cutoff and ordered dimensions");
    }
    let mut mismatching_entries = 0;
    let mut first_mismatch = None;
    for i in 0..smaller.dimension() {
        for j in 0..smaller.dimension() {
            if smaller.entries[i * smaller.dimension() + j]
                != larger.entries[i * larger.dimension() + j]
            {
                mismatching_entries += 1;
                first_mismatch.get_or_insert([i, j]);
            }
        }
    }
    Ok(PrefixNestingCheck {
        semantics: "retained-even-prefix-exact-point-comparison-v1".into(),
        smaller_source: smaller.manifest.content_digest.clone(),
        larger_source: larger.manifest.content_digest.clone(),
        smaller_key: smaller.manifest.key.clone(),
        larger_key: larger.manifest.key.clone(),
        smaller_manifest_provenance: smaller.manifest.provenance_digest.clone(),
        larger_manifest_provenance: larger.manifest.provenance_digest.clone(),
        smaller_dimension: smaller.dimension(),
        larger_dimension: larger.dimension(),
        smaller_precision_bits: smaller.precision,
        larger_precision_bits: larger.precision,
        entries_compared: smaller.entries.len(),
        mismatching_entries,
        first_mismatch,
        exactly_nested: mismatching_entries == 0,
        assurance: "exact_stored_point_equality_not_assembly_accuracy_or_a_general_nesting_theorem"
            .into(),
    })
}

fn dot(a: &[Float], b: &[Float], p: u32) -> Result<Float> {
    numerical_checks::workspace(a.len(), p)?;
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite() || x.prec() > p) {
        bail!("invalid prefix dot-product shape or source precision");
    }
    super::retained_evidence::point::dot(a, b, p)
}

fn unit(v: &[Float], p: u32) -> Result<Vec<Float>> {
    numerical_checks::unit(v, p)
}
fn residual(a: &[Float], stride: usize, v: &[Float], rhs: &[Float], p: u32) -> Float {
    numerical_checks::backward_error(
        a,
        stride,
        v,
        numerical_checks::RightHandSide::Vector(rhs),
        p,
    )
    .unwrap_or_else(|_| Float::with_val(p, rug::float::Special::Infinity))
}
fn eigenpair_residual(a: &[Float], stride: usize, v: &[Float], lambda: &Float, p: u32) -> Float {
    numerical_checks::backward_error(
        a,
        stride,
        v,
        numerical_checks::RightHandSide::Eigenvalue(lambda),
        p,
    )
    .unwrap_or_else(|_| Float::with_val(p, rug::float::Special::Infinity))
}
fn upper_decimal(value: &Float) -> String {
    value.to_string_radix_round(10, None, rug::float::Round::Up)
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
    let tolerance = acceptance_tolerance(&options.export_relative_tolerance, p)?;
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
        if xc_core::DecimalLiteral::new(&pair.cutoff)?
            .cmp_numeric(&xc_core::DecimalLiteral::new(&matrix.cutoff)?)?
            != std::cmp::Ordering::Equal
            || pair.precision > p
            || !options.checkpoint_dimensions.contains(&(pair.modes + 1))
            || by_dimension.insert(pair.modes + 1, pair).is_some()
        {
            bail!("mismatched, duplicate, or unrequested checkpoint eigenstate");
        }
    }
    let ladder = analyze_prefixes_with_policy(
        &matrix.entries,
        matrix.dimension(),
        p,
        options.pivot_margin_bits,
        &options.checkpoint_dimensions,
        &options.diagnostics,
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
            let source_unit = unit(&pair.vector, p)?;
            let sqrt2 = Float::with_val(p, 2).sqrt();
            xi.push(Float::with_val(p, &source_unit[pair.modes]));
            for j in 1..k {
                let mut x = Float::with_val(p, &source_unit[pair.modes + j]);
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
                .any(|(a, b)| !numerical_checks::component_close(a, b, &tolerance, p))
            {
                return Ok(false);
            }
            let mut rhs = vec![Float::with_val(p, 0); k];
            rhs[k - 1] = sigma.clone();
            if decoded[k - 1] != 1 {
                return Ok(false);
            }
            let mass_error = numerical_checks::unit_deviation_upper(&decoded[k..2 * k], p)?;
            if mass_error > tolerance
                || residual(&matrix.entries, matrix.dimension(), &decoded[..k], &rhs, p) > tolerance
            {
                return Ok(false);
            }
            let renormalized = unit(&decoded[..k], p)?;
            if renormalized
                .iter()
                .zip(&decoded[k..2 * k])
                .any(|(a, b)| !numerical_checks::component_close(a, b, &tolerance, p))
            {
                return Ok(false);
            }
            if let Some(pair) = pair {
                let vector = &decoded[2 * k..];
                let error = numerical_checks::unit_deviation_upper(vector, p)?;
                if error > tolerance
                    || eigenpair_residual(
                        &matrix.entries,
                        matrix.dimension(),
                        vector,
                        &pair.eigenvalue,
                        p,
                    ) > tolerance
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
                packet.decoded_innovation_backward_error = Some(upper_decimal(&residual(
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
                    let overlap = dot(&decoded[k..2 * k], &decoded[2 * k..], p)?;
                    packet.signed_overlap = Some(lossless_decimal(&overlap));
                    packet.squared_overlap = Some(lossless_decimal(
                        &super::retained_evidence::point::product(&[&overlap, &overlap], p)?,
                    ));
                    packet.decoded_eigenpair_backward_error =
                        Some(upper_decimal(&eigenpair_residual(
                            &matrix.entries,
                            matrix.dimension(),
                            &decoded[2 * k..],
                            &pair.eigenvalue,
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
        semantics: prefix_semantics(options).into(),
        basis: EVEN_BASIS.into(),
        parent_matrix_source: matrix.manifest.content_digest.clone(),
        parent_n_modes: matrix.modes,
        cutoff: matrix.cutoff.clone(),
        source_precision_bits: matrix.precision,
        options: options.clone(),
        prefixes_are_parent_derived: true,
        nesting_checks: Vec::new(),
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
/// Source-only diagnostics may be public only when all supplied parents are
/// public. Registry registration is separate from numerical execution.
/// Warm reuse replays every numerical field from the supplied retained points
/// in O(D^3) arithmetic; it does not reconstruct Tau matrices or eigenstates.
pub fn analyze_retained_prefixes_via_cache(
    matrix: &RetainedEvenMatrix,
    options: &PrefixAnalysisOptions,
    eigenpairs: &[RetainedEvenEigenpair],
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<CcmPrefixAnalysis>> {
    let mut canonical_options = options.clone();
    canonical_options.export_relative_tolerance =
        xc_core::DecimalLiteral::new(&options.export_relative_tolerance)?
            .canonical()?
            .to_string();
    let options = &canonical_options;
    if cache.requested_assurance != xc_core::AssuranceLevel::Computed {
        bail!("prefix diagnostics are computed evidence, not certified outputs");
    }
    if cache.write_visibility == CacheVisibility::Public
        && (matrix.manifest.visibility != CacheVisibility::Public
            || eigenpairs
                .iter()
                .any(|e| e.manifest.visibility != CacheVisibility::Public))
    {
        bail!("public prefix diagnostics require public source manifests");
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
        mathematical_semantics_version: prefix_semantics(options).into(),
        resolved_mathematical_parameters: serde_json::json!({"source_dependencies":dependencies,"options":options,"source_parents_are_public":matrix.manifest.visibility == CacheVisibility::Public && eigenpairs.iter().all(|e| e.manifest.visibility == CacheVisibility::Public)}),
        normalization: Some(EVEN_BASIS.into()),
        target: Some("parent_derived_prefix_moments_and_checkpoint_exports".into()),
        subspace: Some("even".into()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(if options.diagnostics.is_legacy_default() {
            "unpivoted_ldlt_directed_scaled_exports_and_full_source_replay_v2".into()
        } else {
            format!(
                "unpivoted_ldlt_directed_scaled_exports_and_full_source_replay_v2+{}",
                xc_numerics::prefix::TWO_MODE_MOMENT_SEMANTICS
            )
        }),
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
        minimum_reader_version: ToolkitVersion::parse(xc_cache::CLEAN_SLATE)?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".into(), "ccm".into()),
            ("assurance".into(), "computed_not_certified".into()),
        ]),
        provenance_digest: None,
        production_sink: cache.production_sink,
    };
    // A fresh payload can use its exact process-local seal. Cache reuse has
    // no such seal and must replay every field against the retained inputs.
    let fresh = std::cell::RefCell::new(None);
    let result = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let report = analyze_retained_prefixes(matrix, options, eigenpairs)
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            if !cache.mode.compares_against_reference() {
                let bytes = serde_json::to_vec(&report)
                    .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
                fresh.replace(Some(ContentDigest::sha256(&bytes)));
            }
            Ok((report, dependencies.clone()))
        },
        |report| {
            let bytes = serde_json::to_vec(report)
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            if fresh.borrow().as_ref() == Some(&ContentDigest::sha256(&bytes)) {
                return Ok(());
            }
            let expected = analyze_retained_prefixes(matrix, options, eigenpairs)
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            if report != &expected {
                return Err(CacheError::InvalidManifest(
                    "retained prefix numerical rows or exports fail exact source replay".into(),
                ));
            }
            Ok(())
        },
    )?;
    if let Some(manifest) = result
        .produced_manifest
        .as_ref()
        .or(result.reused_manifest.as_ref())
    {
        let mut sources = vec![&matrix.manifest];
        sources.extend(eigenpairs.iter().map(|e| &e.manifest));
        if !xc_cache::manifest_sources_match(manifest, &sources)? {
            bail!("cached prefix source dependency bindings differ");
        }
    }
    Ok(result)
}

/// Typed meanings for the scalar ladder. Moment-based eigenvalue endpoints
/// remain estimates and never acquire root-depth or certified-bound semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixObservable {
    SchurPivot,
    InnovationMass,
    InverseTraceIncrement,
    InverseTrace,
    InverseSquareTrace,
    InverseCubeTrace,
    SmallestEigenvalueCubeLowerEstimate,
    SmallestEigenvalueCubeRatioUpperEstimate,
    TwoModeSmallestEigenvalueEstimate,
    TwoModeSecondEigenvalueEstimate,
    TwoModeGapRatioEstimate,
    TwoModeTraceClosureResidual,
    EffectiveInverseRank,
    NewestInverseTraceFraction,
    SmallestEigenvalueLowerEstimate,
    SmallestEigenvalueUpperEstimate,
    GapRatioEstimate,
    SmallestEigenvalueSecondOrderEstimate,
    PivotCancellationDigits,
    InnovationCancellationDigits,
    EigenvalueDepthLowerEstimate,
    EigenvalueDepthUpperEstimate,
}
/// Interpret an existing prefix report without refactoring or loading sources.
/// The caller supplies resolution evidence in this observable's units. Unknown
/// construction errors stay unknown. The report hash and largest-parent N
/// remain in every design: these are NOT canonical independently built prefixes.
/// Numerical authenticity remains that of the supplied report/source chain.
pub fn prefix_observations(
    report: &CcmPrefixAnalysis,
    observable: PrefixObservable,
    resolution: &xc_core::ObservableResolution,
) -> Result<Vec<xc_core::ObservationPayload>> {
    use xc_core::*;
    resolution.validate()?;
    precision(report.source_precision_bits)?;
    precision(report.ladder.precision_bits)?;
    precision(report.options.working_precision_bits)?;
    validate_cutoff(&report.cutoff)?;
    if report.parent_n_modes > 8192
        || report.ladder.precision_bits != report.options.working_precision_bits
        || report.ladder.precision_bits < report.source_precision_bits
        || resolution.analysis_precision_bits < report.ladder.precision_bits
    {
        bail!("incoherent prefix report dimensions or analysis precision");
    }

    if report.schema_version != 1
        || ![
            PREFIX_SEMANTICS,
            EXTENDED_PREFIX_SEMANTICS,
            LEGACY_PREFIX_SEMANTICS,
            "ccm-retained-even-prefix-moments-checked-exports-v6",
            "ccm-retained-even-prefix-moments-checked-exports-v7",
            "ccm-retained-even-prefix-moments-checked-exports-v2",
            "ccm-retained-even-prefix-moments-checked-exports-v3",
        ]
        .contains(&report.semantics.as_str())
        || report.basis != EVEN_BASIS
        || !report.prefixes_are_parent_derived
        || resolution.source_precision_bits != report.source_precision_bits
    {
        bail!("unsupported prefix report semantics, basis or source precision");
    }
    let source = ConfigDigest(ContentDigest::sha256(&serde_json::to_vec(report)?).0);
    let mut observations = Vec::with_capacity(report.ladder.rows.len());
    let mut previous = 0;
    for row in &report.ladder.rows {
        if row.dimension <= previous || row.dimension > report.parent_n_modes.saturating_add(1) {
            bail!("invalid prefix dimension sequence");
        }
        previous = row.dimension;
        // A one-dimensional matrix has no second mode. Do not invent a gap.
        if matches!(
            observable,
            PrefixObservable::GapRatioEstimate
                | PrefixObservable::TwoModeSecondEigenvalueEstimate
                | PrefixObservable::TwoModeGapRatioEstimate
        ) && row.dimension == 1
        {
            continue;
        }
        let derived;
        let (name, value, depth) = match observable {
            PrefixObservable::SchurPivot => ("schur-pivot", &row.sigma, false),
            PrefixObservable::InnovationMass => ("innovation-mass", &row.innovation_mass, false),
            PrefixObservable::InverseTraceIncrement => (
                "inverse-trace-increment",
                &row.inverse_trace_increment,
                false,
            ),
            PrefixObservable::InverseTrace => ("inverse-trace", &row.inverse_trace, false),
            PrefixObservable::InverseSquareTrace => {
                ("inverse-square-trace", &row.inverse_square_trace, false)
            }
            PrefixObservable::InverseCubeTrace
            | PrefixObservable::SmallestEigenvalueCubeLowerEstimate
            | PrefixObservable::SmallestEigenvalueCubeRatioUpperEstimate
            | PrefixObservable::TwoModeSmallestEigenvalueEstimate
            | PrefixObservable::TwoModeSecondEigenvalueEstimate
            | PrefixObservable::TwoModeGapRatioEstimate
            | PrefixObservable::TwoModeTraceClosureResidual => {
                let third=row.third_inverse_moment.as_ref().ok_or_else(|| anyhow::anyhow!("third moment was not captured at prefix {}",row.dimension))?;
                let fit=&third.two_mode_fit;
                let (name,value)=match observable {
                    PrefixObservable::InverseCubeTrace=>("inverse-cube-trace",Some(&third.inverse_cube_trace)),
                    PrefixObservable::SmallestEigenvalueCubeLowerEstimate=>("smallest-eigenvalue-cube-lower-estimate",Some(&third.smallest_eigenvalue_lower_estimate)),
                    PrefixObservable::SmallestEigenvalueCubeRatioUpperEstimate=>("smallest-eigenvalue-cube-ratio-upper-estimate",Some(&third.smallest_eigenvalue_upper_estimate)),
                    PrefixObservable::TwoModeSmallestEigenvalueEstimate=>("two-mode-smallest-eigenvalue-estimate",fit.smallest_eigenvalue_estimate.as_ref()),
                    PrefixObservable::TwoModeSecondEigenvalueEstimate=>("two-mode-second-eigenvalue-estimate",fit.second_eigenvalue_estimate.as_ref()),
                    PrefixObservable::TwoModeGapRatioEstimate=>("two-mode-gap-ratio-estimate",fit.gap_ratio_estimate.as_ref()),
                    _=>("two-mode-relative-trace-closure-residual",fit.relative_trace_closure_residual.as_ref()),
                };
                (name,value.ok_or_else(||anyhow::anyhow!("two-mode fit unavailable at prefix {}: {:?}",row.dimension,fit.status))?,false)
            },
            PrefixObservable::EffectiveInverseRank => {
                ("effective-inverse-rank", &row.effective_inverse_rank, false)
            }
            PrefixObservable::NewestInverseTraceFraction => (
                "newest-inverse-trace-fraction",
                &row.newest_inverse_trace_fraction,
                false,
            ),
            PrefixObservable::SmallestEigenvalueLowerEstimate => (
                "smallest-eigenvalue-lower-estimate",
                &row.smallest_eigenvalue_lower_estimate,
                false,
            ),
            PrefixObservable::SmallestEigenvalueUpperEstimate => (
                "smallest-eigenvalue-upper-estimate",
                &row.smallest_eigenvalue_upper_estimate,
                false,
            ),
            PrefixObservable::EigenvalueDepthLowerEstimate => (
                "eigenvalue-depth-lower-estimate",
                &row.eigenvalue_depth_lower_estimate,
                true,
            ),
            PrefixObservable::EigenvalueDepthUpperEstimate => (
                "eigenvalue-depth-upper-estimate",
                &row.eigenvalue_depth_upper_estimate,
                true,
            ),
            PrefixObservable::GapRatioEstimate => (
                "gap-ratio-two-mode-estimate",
                row.gap_ratio_estimate.as_ref().ok_or_else(|| anyhow::anyhow!("gap proxy unavailable at prefix {}; v2 moments and a resolved nonnegative width are required", row.dimension))?,
                false,
            ),
            PrefixObservable::SmallestEigenvalueSecondOrderEstimate => (
                "smallest-eigenvalue-second-order-two-mode-estimate",
                row.smallest_eigenvalue_second_order_estimate.as_ref().ok_or_else(|| anyhow::anyhow!("second-order estimate unavailable at prefix {}", row.dimension))?,
                false,
            ),
            PrefixObservable::PivotCancellationDigits => {
                let numerator = scalar(&row.pivot_cancellation_scale, report.ladder.precision_bits)?;
                let denominator = scalar(&row.sigma, report.ladder.precision_bits)?;
                if numerator <= 0 || denominator <= 0 { bail!("prefix cancellation ratio requires positive values"); }
                derived = lossless_decimal(&super::hp::sector_gap_math::log_magnitude_ratio(&numerator,&denominator,report.ladder.precision_bits)?);
                ("pivot-cancellation-decimal-digits", &derived, false)
            },
            PrefixObservable::InnovationCancellationDigits => (
                "innovation-cancellation-decimal-digits",
                row.innovation_cancellation.as_ref().and_then(|c| c.decimal_digits_lost.as_ref())
                    .ok_or_else(|| anyhow::anyhow!("finite innovation cancellation unavailable at prefix {}; inspect its raw cancellation status", row.dimension))?,
                false,
            ),
        };
        let payload = ObservationPayload {
            schema_version: 1,
            observable: ObservableContract {
                definition_id: format!("ccm.parent-prefix.{name}"),
                definition_version: 1,
                target: ObservableTarget::Finite,
                transform: if depth {
                    ObservableTransform::PositiveLog {
                        base: LogBase::Decimal,
                        negative: true,
                    }
                } else {
                    ObservableTransform::Identity
                },
                normalization: if matches!(
                    observable,
                    PrefixObservable::TwoModeSmallestEigenvalueEstimate
                        | PrefixObservable::TwoModeSecondEigenvalueEstimate
                        | PrefixObservable::TwoModeGapRatioEstimate
                        | PrefixObservable::TwoModeTraceClosureResidual
                ) {
                    "exactly two positive inverse modes fitted to computed T2,T3; T1 closure is diagnostic; estimates_not_bounds; not full-spectrum identification".into()
                } else if matches!(
                    observable,
                    PrefixObservable::GapRatioEstimate
                        | PrefixObservable::SmallestEigenvalueSecondOrderEstimate
                ) {
                    "two-mode asymptotic model; a dominant second inverse mode is assumed; estimate_not_bound".into()
                } else {
                    "raw innovation last coefficient +1; scalar moments in orthonormal coordinates"
                        .into()
                },
                metric: "Euclidean in the orthonormal even sector".into(),
                target_definition: None,
                derivative_coordinate: None,
                root: None,
                prolate: None,
            },
            design: ObservationDesign {
                coordinates: BTreeMap::from([
                    ("c=lambda^2".into(), DecimalLiteral::new(&report.cutoff)?),
                    (
                        "parent_N".into(),
                        DecimalLiteral::new(report.parent_n_modes.to_string())?,
                    ),
                ]),
                n_modes: Some(row.dimension - 1),
                dimension: row.dimension,
                parity: "reflection-even".into(),
                basis: EVEN_BASIS.into(),
                method: report.semantics.clone(),
                quadrature_identity: format!(
                    "reported-parent-payload:{};construction-not-rechecked",
                    report.parent_matrix_source.0
                ),
                source_precision_bits: report.source_precision_bits,
                exact_input_identity: source.clone(),
            },
            value: ObservedScalar::Finite {
                value: DecimalLiteral::new(value)?,
            },
            resolution: resolution.clone(),
        };
        payload.validate()?;
        observations.push(payload);
    }
    Ok(observations)
}

/// Stable reduction checks of an approved retained matrix. This is a new
/// analysis of its stored point entries, not a canonical matrix/eigenstate
/// replacement or an interval certificate of the original Weil form.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedReductionCheck {
    pub schema_version: u32,
    pub algorithm_semantics: String,
    pub matrix_source: ContentDigest,
    pub cutoff: String,
    pub n_modes: usize,
    pub dimension: usize,
    pub source_precision_bits: u32,
    pub working_precision_bits: u32,
    pub authorized_maximum_dimension: usize,
    pub estimated_matrix_working_bytes: u64,
    pub acceptance_tolerance: String,
    pub checks_passed: bool,
    pub diagnostics: xc_core::SymmetricReductionDiagnostics<String>,
    pub diagonal: Vec<String>,
    pub off_diagonal: Vec<String>,
    pub computed_eigenvalues: Vec<String>,
    pub assurance: String,
}
/// Persist a source-bound stable-reduction diagnostic without rebuilding a source.
/// A warm hit replays the complete computed report from the supplied retained
/// matrix in O(d^3) arithmetic. This does not rebuild Tau or change assurance.
pub fn check_retained_reduction_via_cache(
    matrix: &RetainedEvenMatrix,
    working_precision_bits: u32,
    maximum_dimension: usize,
    relative_tolerance: &str,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<RetainedReductionCheck>> {
    let canonical_tolerance = xc_core::DecimalLiteral::new(relative_tolerance)?
        .canonical()?
        .to_string();
    let relative_tolerance = canonical_tolerance.as_str();
    if cache.requested_assurance != xc_core::AssuranceLevel::Computed {
        bail!("retained reduction is computed evidence only");
    }
    precision(working_precision_bits)?;
    if matrix.dimension() > maximum_dimension || working_precision_bits < matrix.precision {
        bail!("retained reduction exceeds dimension or precision budget");
    }
    let tolerance = acceptance_tolerance(relative_tolerance, working_precision_bits)?;
    if tolerance <= 0 || tolerance >= 1 {
        bail!("invalid reduction tolerance");
    }
    if cache.write_visibility == CacheVisibility::Public
        && matrix.manifest.visibility != CacheVisibility::Public
    {
        bail!("public reduction diagnostics require a public source manifest");
    }
    let estimated_bytes = 2
        * matrix.dimension() as u128
        * matrix.dimension() as u128
        * (u128::from(working_precision_bits).div_ceil(8) + 96);
    if estimated_bytes > 16 * 1024_u128.pow(3) {
        bail!("retained reduction exceeds matrix-storage budget");
    }
    let dependencies = vec![dependency(&matrix.manifest)];
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_retained_reduction_check".into(),
        mathematical_semantics_version: RETAINED_REDUCTION_SEMANTICS.into(),
        resolved_mathematical_parameters: serde_json::json!({
            "source_dependencies": dependencies, "source_parents_are_public": matrix.manifest.visibility == CacheVisibility::Public, "working_precision_bits": working_precision_bits,
            "maximum_dimension": maximum_dimension, "relative_tolerance": relative_tolerance,
        }),
        normalization: Some(EVEN_BASIS.into()),
        target: Some("stored_point_matrix_reduction_diagnostics".into()),
        subspace: Some("even".into()),
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some(format!(
            "{};full_source_replay_v2",
            xc_numerics::eigen::STABLE_HOUSEHOLDER_SEMANTICS
        )),
    };
    let logical = format!(
        "ccm/retained-reduction/{}",
        matrix.manifest.content_digest.0
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.retained_reduction.check",
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
    let fresh = std::cell::RefCell::new(None);
    let result = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let report = check_retained_reduction(
                matrix,
                working_precision_bits,
                maximum_dimension,
                relative_tolerance,
            )
            .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            if !cache.mode.compares_against_reference() {
                let bytes = serde_json::to_vec(&report)
                    .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
                fresh.replace(Some(ContentDigest::sha256(&bytes)));
            }
            Ok((report, dependencies.clone()))
        },
        |report: &RetainedReductionCheck| {
            let bytes = serde_json::to_vec(report)
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            if fresh.borrow().as_ref() == Some(&ContentDigest::sha256(&bytes)) {
                return Ok(());
            }
            let expected = check_retained_reduction(
                matrix,
                working_precision_bits,
                maximum_dimension,
                relative_tolerance,
            )
            .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
            if report != &expected {
                return Err(CacheError::InvalidManifest(
                    "retained reduction numerical fields fail exact source replay".into(),
                ));
            }
            Ok(())
        },
    )?;
    if let Some(manifest) = result
        .produced_manifest
        .as_ref()
        .or(result.reused_manifest.as_ref())
    {
        if !xc_cache::manifest_sources_match(manifest, &[&matrix.manifest])? {
            bail!("retained reduction dependency closure mismatch");
        }
    }
    Ok(result)
}

/// Explicitly authorized dense diagnostic: O(d^3) arithmetic and O(d^2)
/// working storage. No source lookup, replacement, publication or automatic
/// Ultra backfill. The dimension budget is checked before decomposition; an
/// operational 16-GiB matrix-storage screen also applies. Extra working bits
/// do not recover construction digits missing from the approved source.
pub fn check_retained_reduction(
    matrix: &RetainedEvenMatrix,
    working_precision_bits: u32,
    maximum_dimension: usize,
    relative_tolerance: &str,
) -> Result<RetainedReductionCheck> {
    use xc_numerics::eigen::{
        assess_symmetric_reduction_hp, householder_tridiag_hp_stable, tridiag_eigenvalues_hp,
        STABLE_HOUSEHOLDER_SEMANTICS,
    };
    precision(working_precision_bits)?;
    let d = matrix.dimension();
    if maximum_dimension == 0
        || d > maximum_dimension
        || working_precision_bits < matrix.source_precision_bits()
    {
        bail!("retained reduction exceeds the dimension budget or would down-round the source");
    }
    let estimated_bytes =
        2 * d as u128 * d as u128 * (u128::from(working_precision_bits).div_ceil(8) + 96);
    if estimated_bytes > 16 * 1024_u128.pow(3) {
        bail!("retained reduction exceeds estimated matrix-storage budget");
    }
    let tolerance = acceptance_tolerance(relative_tolerance, working_precision_bits)?;
    if tolerance <= 0 || tolerance >= 1 {
        bail!("relative diagnostic tolerance must lie strictly between zero and one");
    }
    let (diag, off_diag, q) =
        householder_tridiag_hp_stable(matrix.entries(), d, working_precision_bits)?;
    let diagnostics = assess_symmetric_reduction_hp(
        matrix.entries(),
        &diag,
        &off_diag,
        &q,
        working_precision_bits,
    )?;
    let passed = diagnostics.relative_similarity_residual <= tolerance
        && diagnostics.relative_orthogonality_residual <= tolerance;
    let eigenvalues = tridiag_eigenvalues_hp(&diag, &off_diag, working_precision_bits)?;
    if eigenvalues.iter().any(|x| !x.is_finite()) {
        bail!("nonfinite retained spectrum check");
    }
    Ok(RetainedReductionCheck {
        schema_version: 1,
        algorithm_semantics: STABLE_HOUSEHOLDER_SEMANTICS.into(),
        matrix_source: matrix.manifest().content_digest.clone(),
        cutoff: matrix.cutoff.clone(),
        n_modes: matrix.modes,
        dimension: d,
        source_precision_bits: matrix.precision,
        working_precision_bits,
        authorized_maximum_dimension: maximum_dimension,
        estimated_matrix_working_bytes: estimated_bytes as u64,
        acceptance_tolerance: relative_tolerance.into(),
        checks_passed: passed,
        diagnostics: xc_core::SymmetricReductionDiagnostics {
            absolute_similarity_residual: lossless_decimal(
                &diagnostics.absolute_similarity_residual,
            ),
            relative_similarity_residual: lossless_decimal(
                &diagnostics.relative_similarity_residual,
            ),
            absolute_orthogonality_residual: lossless_decimal(
                &diagnostics.absolute_orthogonality_residual,
            ),
            relative_orthogonality_residual: lossless_decimal(
                &diagnostics.relative_orthogonality_residual,
            ),
            source_frobenius_norm: lossless_decimal(&diagnostics.source_frobenius_norm),
            tridiagonal_frobenius_norm: lossless_decimal(&diagnostics.tridiagonal_frobenius_norm),
            basis_frobenius_norm: lossless_decimal(&diagnostics.basis_frobenius_norm),
        },
        diagonal: diag.iter().map(lossless_decimal).collect(),
        off_diagonal: off_diag.iter().map(lossless_decimal).collect(),
        computed_eigenvalues: eigenvalues.iter().map(lossless_decimal).collect(),
        assurance: REDUCTION_ASSURANCE.into(),
    })
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
mod exhaustive_retained_prefix {
    use super::*;
    use serde_json::json;
    use xc_cache::{ArtifactKey, CacheObjectRef};
    fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
        let bytes = serde_json::to_vec(&value).unwrap();
        let digest = ContentDigest::sha256(&bytes);
        (
            ArtifactManifest {
                schema_version: 1,
                key: ArtifactKey::new(kind, "prefix-boundary-fixture", kind.as_bytes()).unwrap(),
                content_digest: digest.clone(),
                size_bytes: bytes.len() as u64,
                objects: vec![CacheObjectRef {
                    content_digest: digest,
                    size_bytes: bytes.len() as u64,
                }],
                created_unix_seconds: 1,
                producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies: vec![],
                tags: BTreeMap::new(),
                provenance_digest: None,
            },
            bytes,
        )
    }
    fn matrix(cutoff: &str) -> (ArtifactManifest, Vec<u8>) {
        source(
            "ccm_even_sector_matrix",
            json!({"schema_version":1,"lambda_squared":cutoff,"n_modes":1,"precision_bits":128,"dimension":2,"entries":["1","0","0","1"]}),
        )
    }
    fn options() -> PrefixAnalysisOptions {
        PrefixAnalysisOptions {
            working_precision_bits: 128,
            pivot_margin_bits: 32,
            checkpoint_dimensions: vec![2],
            export_significant_digits: vec![40, 48],
            export_relative_tolerance: "1e-30".into(),
            diagnostics: PrefixDiagnosticPolicy::default(),
        }
    }
    fn report() -> CcmPrefixAnalysis {
        let (m, b) = matrix("13");
        let a = RetainedEvenMatrix::from_payload(&m, &b, std::slice::from_ref(&m.content_digest))
            .unwrap();
        analyze_retained_prefixes(&a, &options(), &[]).unwrap()
    }
    fn resolution() -> xc_core::ObservableResolution {
        use xc_core::*;
        ObservableResolution {
            source_precision_bits: 128,
            analysis_precision_bits: 128,
            export_significant_digits: 40,
            components: RESOLUTION_AXES
                .iter()
                .map(|&axis| ResolutionComponent {
                    axis,
                    classification: ResolutionClass::Unknown,
                    absolute: None,
                    explanation: "not evaluated for supplied report".into(),
                    evidence: vec![],
                    dependency_groups: Default::default(),
                })
                .collect(),
        }
    }
    #[test]
    fn current_retained_eigenpair_schemas_preserve_exact_points_and_authentication() {
        for schema in [2, 3, 4, 5, 6] {
            let (manifest, bytes) = source(
                "ccm_weil_eigenpair",
                json!({
                    "schema_version": schema, "lambda_squared": "13", "n_modes": 1,
                    "precision_bits": 128, "eigenvalue": "1", "eigenvector": ["1", "2", "1"],
                    "force_even": true, "parity_policy": "even_sector",
                }),
            );
            let approved = std::slice::from_ref(&manifest.content_digest);
            let pair = RetainedEvenEigenpair::from_payload(&manifest, &bytes, approved);
            let state = crate::ccm::state_geometry::RetainedState::from_payload(
                &manifest, &bytes, approved,
            );
            if schema == 6 {
                assert!(pair.is_err());
                assert!(state.is_err());
                continue;
            }
            let pair = pair.unwrap();
            let state = state.unwrap();
            // Exact integer point oracle; format support never transforms
            // or silently projects the retained source coefficients.
            assert_eq!(pair.eigenvalue, 1);
            for (actual, expected) in pair.vector.iter().zip([1, 2, 1]) {
                assert_eq!(actual.to_rational().unwrap(), rug::Rational::from(expected));
            }
            assert_eq!(state.coefficients, pair.vector);
            let mut changed = bytes.clone();
            changed[0] ^= 1;
            assert!(RetainedEvenEigenpair::from_payload(&manifest, &changed, approved).is_err());
            assert!(crate::ccm::state_geometry::RetainedState::from_payload(
                &manifest, &changed, approved,
            )
            .is_err());
        }
    }
    #[test]
    fn exhaustive_retained_prefix_nonzero_decimal_underflow() {
        assert!(scalar("1e-400000000", 128).is_err());
    }
    #[test]
    fn exhaustive_retained_prefix_scalar_precision_is_fallible() {
        assert!(std::panic::catch_unwind(|| scalar("1", 0)).is_ok_and(|x| x.is_err()));
    }
    #[test]
    fn exhaustive_retained_prefix_dot_preserves_exact_cancellation() {
        let p = 64;
        let big = Float::with_val(p, 1) << 100u32;
        let a = vec![big.clone(), Float::with_val(p, 1), -big];
        let b = vec![Float::with_val(p, 1); 3];
        assert_eq!(dot(&a, &b, p).unwrap(), 1);
    }
    #[test]
    fn exhaustive_retained_prefix_matrix_cutoff_is_mathematical() {
        let (m, b) = matrix("NaN");
        assert!(
            RetainedEvenMatrix::from_payload(&m, &b, std::slice::from_ref(&m.content_digest))
                .is_err()
        );
    }
    #[test]
    fn exhaustive_retained_prefix_eigenpair_cutoff_is_mathematical() {
        let (m, b) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":2,"lambda_squared":"1","n_modes":1,"precision_bits":128,"eigenvalue":"1","eigenvector":["0","1","0"]}),
        );
        assert!(RetainedEvenEigenpair::from_payload(
            &m,
            &b,
            std::slice::from_ref(&m.content_digest)
        )
        .is_err());
    }
    #[test]
    fn exhaustive_retained_prefix_restriction_normalizes_before_overflow() {
        let (m, b) = matrix("13");
        let a = RetainedEvenMatrix::from_payload(&m, &b, std::slice::from_ref(&m.content_digest))
            .unwrap();
        let big = Float::with_val(128, 0.75) << rug::float::exp_max();
        assert!(big.is_finite());
        let (m, b) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":2,"lambda_squared":"13","n_modes":1,"precision_bits":128,"eigenvalue":"1","eigenvector":[big.to_string(),"0",big.to_string()]}),
        );
        let pair =
            RetainedEvenEigenpair::from_payload(&m, &b, std::slice::from_ref(&m.content_digest))
                .unwrap();
        let r = analyze_retained_prefixes(&a, &options(), &[pair]);
        assert!(r.is_ok(), "{r:?}");
        assert_eq!(r.unwrap().checkpoints[0].status, "export_checks_passed");
    }
    #[test]
    fn exhaustive_retained_prefix_observation_precision_is_fallible() {
        let mut r = report();
        r.ladder.precision_bits = 0;
        assert!(std::panic::catch_unwind(|| prefix_observations(
            &r,
            PrefixObservable::PivotCancellationDigits,
            &resolution()
        ))
        .is_ok_and(|x| x.is_err()));
    }
    #[test]
    fn exhaustive_retained_prefix_log_ratio_preserves_finite_result() {
        let mut r = report();
        for row in &mut r.ladder.rows {
            row.sigma = (Float::with_val(128, 1) >> 700_000_000u32).to_string();
            row.pivot_cancellation_scale = (Float::with_val(128, 1) << 700_000_000u32).to_string();
        }
        let values =
            prefix_observations(&r, PrefixObservable::PivotCancellationDigits, &resolution());
        assert!(values.is_ok(), "{values:?}");
        let expected = Float::with_val(128, Float::with_val(512, 2).log10() * 1_400_000_000u32);
        for item in values.unwrap() {
            let xc_core::ObservedScalar::Finite { value } = item.value else {
                panic!("finite observation required")
            };
            assert_eq!(scalar(value.as_str(), 128).unwrap(), expected);
        }
    }
    #[test]
    fn exhaustive_retained_prefix_unit_rejects_partial_component_underflow() {
        let p = 64;
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        assert!(tiny > 0);
        assert!(unit(
            &[Float::with_val(p, 0.75), Float::with_val(p, 0.75), tiny],
            p
        )
        .is_err());
    }

    #[test]
    fn exhaustive_retained_prefix_exact_dot_oracles() {
        use rug::Rational;
        for p in [64, 128, 256] {
            for exponent in [-900i32, 0, 900] {
                for case in 1i32..=16 {
                    let a: Vec<_> = (0..7)
                        .map(|i| Float::with_val(p, (case + 1) * (i - 3)) << exponent)
                        .collect();
                    let b: Vec<_> = (0..7)
                        .map(|i| Float::with_val(p, (case - 2) * (i + 1) + 1) >> exponent)
                        .collect();
                    let exact = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| x.to_rational().unwrap() * y.to_rational().unwrap())
                        .sum::<Rational>();
                    assert_eq!(dot(&a, &b, p).unwrap(), Float::with_val(p, exact));
                }
            }
            assert!(dot(&[], &[], p).is_err());
            assert!(dot(&[Float::with_val(p, 1)], &[], p).is_err());
        }
    }
    #[test]
    fn exhaustive_retained_prefix_normalization_and_outward_norm_oracles() {
        use rug::{float::Round, Rational};
        for p in [64, 128, 256] {
            for exponent in [-700_000_000i32, 0, 700_000_000] {
                let vector = [
                    Float::with_val(p, 3) << exponent,
                    Float::with_val(p, 4) << exponent,
                ];
                let actual = unit(&vector, p).unwrap();
                assert_eq!(actual[0], Float::with_val(p, Rational::from((3, 5))));
                assert_eq!(actual[1], Float::with_val(p, Rational::from((4, 5))));
                let q = actual
                    .iter()
                    .map(|x| x.to_rational().unwrap().square())
                    .sum::<Rational>();
                let work = p * 4;
                let mut lo = Float::with_val_round(work, &q, Round::Down).0;
                let mut hi = Float::with_val_round(work, &q, Round::Up).0;
                lo.sqrt_round(Round::Down);
                hi.sqrt_round(Round::Up);
                let lo = (lo - 1u32).abs();
                let hi = (hi - 1u32).abs();
                let oracle = lo.max(&hi);
                assert!(numerical_checks::unit_deviation_upper(&actual, p).unwrap() >= oracle);
            }
        }
    }
    #[test]
    fn exhaustive_retained_prefix_exact_domains_and_budgets() {
        assert!(validate_cutoff("1.00000000000000000000000000000000000000001").is_ok());
        for text in [
            "1",
            "0",
            "-2",
            "NaN",
            "inf",
            "0.99999999999999999999999999999999999999",
        ] {
            assert!(validate_cutoff(text).is_err(), "{text}");
        }
        assert!(decode_budget(4, 128, 1000).is_ok());
        assert!(decode_budget(8193usize * 8193, 1_000_000, 1000).is_err());
        assert!(decode_budget(1, 128, usize::MAX).is_err());
        assert!(numerical_checks::workspace(16385, 64).is_ok());
        assert!(numerical_checks::workspace(16386, 64).is_err());
        assert!(numerical_checks::workspace(16385, 1_000_000).is_err());
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(scalar("1", p).is_err());
        }
        for text in ["NaN", "inf", "-inf", "1e400000000", "-1e-400000000"] {
            assert!(scalar(text, 128).is_err());
        }
    }
    #[test]
    fn exhaustive_retained_prefix_export_distance_uses_outward_endpoint() {
        let p = 64;
        let one = Float::with_val(p, 1);
        let tiny = Float::with_val(p, 1) >> 100u32;
        assert!(!numerical_checks::component_close(&one, &(-tiny), &one, p));
        assert!(numerical_checks::component_close(
            &one,
            &Float::with_val(p, 0),
            &one,
            p
        ));
        assert!(numerical_checks::component_close(
            &one,
            &one,
            &Float::with_val(p, 0),
            p
        ));
    }
}
