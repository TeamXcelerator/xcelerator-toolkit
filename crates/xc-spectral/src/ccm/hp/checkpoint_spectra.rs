//! Directed low-spectrum enclosures of leading even-sector blocks.
//!
//! The retained even-sector matrix nests exactly in N: its leading k x k block
//! is the even-sector matrix of the smaller configuration. Each requested
//! block receives directed enclosures of its lowest eigenvalues (Householder
//! source allowance plus directed Sturm bisection), giving rigorous gap and
//! gap-ratio intervals along the dimension ladder.
use super::*;
use rug::float::Round;

pub const CHECKPOINT_SPECTRA_SEMANTICS: &str = "ccm-checkpoint-low-spectrum-v0.16.0-v1";
const CHECKPOINT_SPECTRA_SCOPE: &str = "directed enclosures of the lowest eigenvalues of leading blocks of the exact stored even-sector matrix; widen eigenvalues by ccm_assembly_error_analysis.even_sector.spectral_upper (gaps by twice that) for exact finite-form enclosures";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmCheckpointEigenvalue {
    pub index: usize,
    pub lower: String,
    pub upper: String,
    pub resolution: StoredEigenvalueResolution,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmCheckpointSpectrumRow {
    /// Even-sector block dimension k, the even dimension of N = k - 1 modes.
    pub dimension: usize,
    pub modes: usize,
    /// `certified_finite_enclosure` or `unresolved`.
    pub outcome: String,
    pub eigenvalues: Vec<CcmCheckpointEigenvalue>,
    /// Enclosure of lambda_1 - lambda_0.
    pub gap_lower: Option<String>,
    pub gap_upper: Option<String>,
    /// Enclosure of lambda_0 / lambda_1 when both are resolved positive.
    pub gap_ratio_lower: Option<String>,
    pub gap_ratio_upper: Option<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmCheckpointSpectra {
    pub schema_version: u32,
    pub semantics: String,
    pub claim_scope: String,
    pub source_matrix_digest: String,
    pub source_dimension: usize,
    pub precision_bits: u32,
    pub eigenvalue_count: usize,
    pub rows: Vec<CcmCheckpointSpectrumRow>,
    pub outcome: String,
}

fn digits(value: &Float, round: Round) -> String {
    let digits = xc_numerics::reduction::roundtrip_decimal_digits(value.prec());
    value.to_string_radix_round(10, Some(digits), round)
}

fn row(entries: &[Float], full: usize, k: usize, p: u32, count: usize) -> CcmCheckpointSpectrumRow {
    let mut result = CcmCheckpointSpectrumRow {
        dimension: k,
        modes: k - 1,
        outcome: "unresolved".to_owned(),
        eigenvalues: Vec::new(),
        gap_lower: None,
        gap_upper: None,
        gap_ratio_lower: None,
        gap_ratio_upper: None,
        reason: None,
    };
    let block = (0..k)
        .flat_map(|i| entries[i * full..i * full + k].iter().cloned())
        .collect::<Vec<_>>();
    let spectrum = match super::spectrum_accuracy::lowest_spectrum(&block, k, p, count.min(k)) {
        Ok(spectrum) => spectrum,
        Err(error) => {
            result.reason = Some(format!("{error:#}"));
            return result;
        }
    };
    let bounds = spectrum.eigenvalue_bounds;
    result.eigenvalues = bounds
        .iter()
        .map(|bound| CcmCheckpointEigenvalue {
            index: bound.algebraic_index,
            lower: digits(&bound.lower, Round::Down),
            upper: digits(&bound.upper, Round::Up),
            resolution: bound.resolution.clone(),
        })
        .collect();
    if let [first, second, ..] = bounds.as_slice() {
        let work = first.lower.prec().max(second.lower.prec());
        let lower = Float::with_val_round(work, &second.lower - &first.upper, Round::Down).0;
        let upper = Float::with_val_round(work, &second.upper - &first.lower, Round::Up).0;
        result.gap_lower = Some(digits(&lower, Round::Down));
        result.gap_upper = Some(digits(&upper, Round::Up));
        if first.lower > 0 && second.lower > 0 {
            let ratio_lower =
                Float::with_val_round(work, &first.lower / &second.upper, Round::Down).0;
            let ratio_upper =
                Float::with_val_round(work, &first.upper / &second.lower, Round::Up).0;
            result.gap_ratio_lower = Some(digits(&ratio_lower, Round::Down));
            result.gap_ratio_upper = Some(digits(&ratio_upper, Round::Up));
        }
    }
    result.outcome = "certified_finite_enclosure".to_owned();
    result
}

/// Enclose the lowest `count` eigenvalues of each requested leading block.
/// A block whose enclosure fails is retained as an unresolved row.
pub fn checkpoint_low_spectra(
    matrix: &super::super::prefix::RetainedEvenMatrix,
    dimensions: &[usize],
    count: usize,
) -> Result<CcmCheckpointSpectra> {
    checkpoint_low_spectra_from_entries(
        matrix.entries(),
        matrix.dimension(),
        matrix.source_precision_bits(),
        &matrix.manifest().content_digest,
        dimensions,
        count,
    )
}

fn checkpoint_low_spectra_from_entries(
    entries: &[Float],
    full: usize,
    p: u32,
    source_digest: &ContentDigest,
    dimensions: &[usize],
    count: usize,
) -> Result<CcmCheckpointSpectra> {
    if entries.len() != full * full {
        bail!("checkpoint source matrix shape is inconsistent");
    }
    if count == 0
        || dimensions.is_empty()
        || dimensions.windows(2).any(|pair| pair[0] >= pair[1])
        || dimensions.iter().any(|&k| k == 0 || k > full)
    {
        bail!("checkpoint dimensions must be increasing within the retained even dimension");
    }
    let rows = dimensions
        .iter()
        .map(|&k| row(entries, full, k, p, count))
        .collect::<Vec<_>>();
    let resolved = rows
        .iter()
        .all(|row| row.outcome == "certified_finite_enclosure");
    Ok(CcmCheckpointSpectra {
        schema_version: 1,
        semantics: CHECKPOINT_SPECTRA_SEMANTICS.to_owned(),
        claim_scope: CHECKPOINT_SPECTRA_SCOPE.to_owned(),
        source_matrix_digest: source_digest.0.clone(),
        source_dimension: full,
        precision_bits: p,
        eigenvalue_count: count,
        rows,
        outcome: if resolved {
            "certified_finite_enclosure"
        } else {
            "partial_unresolved"
        }
        .to_owned(),
    })
}

/// Resolve or compute checkpoint spectra through the managed cache.
pub fn checkpoint_low_spectra_via_cache(
    matrix: &super::super::prefix::RetainedEvenMatrix,
    dimensions: &[usize],
    count: usize,
    cache: &ArtifactCacheContext<'_>,
) -> Result<xc_cache::ArtifactExecutionCacheResult<CcmCheckpointSpectra>> {
    let source = matrix.manifest();
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_checkpoint_spectra".to_owned(),
        mathematical_semantics_version: CHECKPOINT_SPECTRA_SEMANTICS.to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "source_matrix_digest": source.content_digest.0,
            "source_dimension": matrix.dimension(),
            "precision_bits": matrix.source_precision_bits(),
            "dimensions": dimensions,
            "eigenvalue_count": count,
        }),
        normalization: None,
        target: Some("lowest_eigenvalues_of_leading_even_blocks".to_owned()),
        subspace: Some("even".to_owned()),
        source_data_identities: BTreeMap::from([(
            "ccm_even_sector_matrix".to_owned(),
            source.content_digest.clone(),
        )]),
        algorithm_semantics: Some(format!(
            "{}+leading_block_lowest_selected_v1",
            super::spectrum_accuracy::WEIL_SPECTRUM_SEMANTICS
        )),
    };
    let semantic_digest = semantic_key.digest()?;
    let logical_key = format!(
        "ccm/checkpoint-spectra/{}/{}",
        matrix.dimension(),
        semantic_digest.0
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.checkpoint_spectra.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
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
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "checkpoint_spectra".to_owned()),
        ]),
        provenance_digest: Some(source.content_digest.clone()),
        production_sink: cache.production_sink,
    };
    Ok(resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let spectra = checkpoint_low_spectra(matrix, dimensions, count)
                .map_err(|error| CacheError::InvalidManifest(format!("{error:#}")))?;
            Ok((spectra, canonical_dependency_refs(vec![source.clone()])))
        },
        |spectra| {
            if spectra.schema_version != 1
                || spectra.semantics != CHECKPOINT_SPECTRA_SEMANTICS
                || spectra.claim_scope != CHECKPOINT_SPECTRA_SCOPE
                || spectra.source_matrix_digest != source.content_digest.0
                || spectra.eigenvalue_count != count
                || spectra.rows.len() != dimensions.len()
                || spectra
                    .rows
                    .iter()
                    .zip(dimensions)
                    .any(|(row, &k)| row.dimension != k)
            {
                return Err(CacheError::InvalidManifest(
                    "CCM checkpoint spectra do not match their semantic identity".to_owned(),
                ));
            }
            Ok(())
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_block_enclosures_contain_exact_tridiagonal_spectra() {
        let p = 192;
        let full = 24;
        let mut entries = vec![Float::with_val(p, 0); full * full];
        for i in 0..full {
            entries[i * full + i] = Float::with_val(p, 2);
            if i + 1 < full {
                entries[i * full + i + 1] = Float::with_val(p, -1);
                entries[(i + 1) * full + i] = Float::with_val(p, -1);
            }
        }
        let digest = ContentDigest::sha256(b"tridiagonal");
        let dimensions = [3, 6, 12, 24];
        let spectra =
            checkpoint_low_spectra_from_entries(&entries, full, p, &digest, &dimensions, 3)
                .unwrap();
        assert_eq!(spectra.outcome, "certified_finite_enclosure");
        let pi = Float::with_val(p + 64, rug::float::Constant::Pi);
        for (row, &k) in spectra.rows.iter().zip(&dimensions) {
            assert_eq!(row.eigenvalues.len(), 3);
            let exact = |j: usize| {
                let angle = Float::with_val(p + 64, &pi * j as u32) / (k as u32 + 1);
                Float::with_val(p + 64, 2) - Float::with_val(p + 64, angle.cos()) * 2u32
            };
            // The reference trigonometric values carry about 2^-(p+64) error.
            let slack = Float::with_val(p + 64, 1) >> (p + 32);
            for (index, eigenvalue) in row.eigenvalues.iter().enumerate() {
                let value = exact(index + 1);
                let (below, above) = (
                    Float::with_val(p + 64, &value + &slack),
                    Float::with_val(p + 64, &value - &slack),
                );
                assert!(Float::with_val(p + 64, Float::parse(&eigenvalue.lower).unwrap()) <= below);
                assert!(Float::with_val(p + 64, Float::parse(&eigenvalue.upper).unwrap()) >= above);
            }
            let gap = Float::with_val(p + 64, exact(2) - exact(1));
            assert!(
                Float::with_val(
                    p + 64,
                    Float::parse(row.gap_lower.as_ref().unwrap()).unwrap()
                ) <= gap
            );
            assert!(
                Float::with_val(
                    p + 64,
                    Float::parse(row.gap_upper.as_ref().unwrap()).unwrap()
                ) >= gap
            );
            let ratio = Float::with_val(p + 64, exact(1) / exact(2));
            assert!(
                Float::with_val(
                    p + 64,
                    Float::parse(row.gap_ratio_lower.as_ref().unwrap()).unwrap()
                ) <= ratio
            );
            assert!(
                Float::with_val(
                    p + 64,
                    Float::parse(row.gap_ratio_upper.as_ref().unwrap()).unwrap()
                ) >= ratio
            );
        }
        assert!(
            checkpoint_low_spectra_from_entries(&entries, full, p, &digest, &[6, 3], 3).is_err()
        );
    }
}
