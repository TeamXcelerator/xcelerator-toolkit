//! Managed production of compact analytic certificates. Only the compact
//! record reaches the cache and publication sink; full proof inputs are local
//! temporary values. Legacy certificate keys and readers remain separate.

use super::*;
use crate::ccm::cutoff_free::CutoffFreeConfig;
use std::collections::BTreeMap;
use xc_cache::{
    resolve_or_compute_json_artifact_with_assessment, ArtifactAssuranceState, ArtifactCacheContext,
    ArtifactExecutionCacheRequest, ArtifactExecutionCacheResult, ArtifactManifest,
    ArtifactProductionAssessment, CacheError, CacheQuality, DependencyRef, SemanticKeyEnvelope,
    ToolkitVersion,
};

const MAX_ASSEMBLY_BITS: u32 = 1_000_000;

fn assembly_precisions(requested: u32) -> Vec<u32> {
    let mut precisions = vec![requested];
    let mut guard = 64;
    let mut bits = requested;
    while bits < MAX_ASSEMBLY_BITS {
        bits = (requested + guard).min(MAX_ASSEMBLY_BITS);
        precisions.push(bits);
        guard *= 2;
    }
    precisions
}

fn try_assembly_precisions<T>(
    precisions: &[u32],
    mut attempt: impl FnMut(u32) -> Result<T>,
) -> Result<T> {
    let mut remaining = precisions.iter();
    let mut bits = *remaining
        .next()
        .context("empty assembly precision ladder")?;
    loop {
        match attempt(bits) {
            Err(error) if error.is::<AssemblyResolutionLimit>() => {
                let Some(&next) = remaining.next() else {
                    return Err(error.context(format!(
                        "compact sector-gap assembly exhausted the supported precision after attempts at {precisions:?} bits"
                    )));
                };
                xc_core::progress_message!(
                    "compact sector-gap assembly inconclusive at {bits} bits; retrying at {next} bits: {error}"
                );
                bits = next;
            }
            result => return result,
        }
    }
}

/// Resolve or compute an optional compact certificate from retained sector
/// guides. The guides choose a request identity; the actual proof uses freshly
/// assembled analytic interval matrices and their own midpoint guides.
///
/// The source identity must name an archived source revision or snapshot that
/// the caller preserves. It is provenance, not a substitute for proof replay.
/// New records use the existing certificate family with a distinct semantic
/// key and representation. They cannot satisfy a legacy full-payload request.
#[allow(clippy::too_many_arguments)]
pub fn resolve_or_compute(
    params: &super::super::super::CcmParams,
    precision_bits: u32,
    even_ground_guide: &Float,
    even_first_excited_guide: &Float,
    odd_ground_guide: &Float,
    even_manifest: &ArtifactManifest,
    odd_manifest: &ArtifactManifest,
    gap_manifest: &ArtifactManifest,
    options: CcmSectorGapCertificationOptions,
    source_identity: ReconstructionSourceIdentity,
    cache: &ArtifactCacheContext<'_>,
) -> Result<ArtifactExecutionCacheResult<CompactCcmSectorGapCertificate>> {
    options.validate()?;
    if !params.lambda_sq.is_integer || params.lambda_sq_int() <= 1 || params.n_modes == 0 {
        bail!("compact sector certification requires integer lambda_squared > 1 and N >= 1");
    }
    if !(64..=MAX_ASSEMBLY_BITS).contains(&precision_bits)
        || !source_identity.source_tree_digest.validate()
        || source_identity.toolkit_revision.trim().is_empty()
        || source_identity.build_identity.trim().is_empty()
    {
        bail!("compact sector certification precision or reconstruction identity is invalid");
    }
    if even_manifest.key.kind != "ccm_sector_spectrum"
        || odd_manifest.key.kind != "ccm_sector_spectrum"
        || gap_manifest.key.kind != "ccm_sector_gap"
        || !even_ground_guide.is_finite()
        || !even_first_excited_guide.is_finite()
        || !odd_ground_guide.is_finite()
    {
        bail!("compact sector certification requires retained finite even/odd/gap guides");
    }
    let scalar_backend = format!(
        "system-flint-arb-{}",
        super::super::super::arb_bridge::backend_version()
    );
    let manifests = [even_manifest, odd_manifest, gap_manifest];
    let precisions = assembly_precisions(precision_bits);
    let identity = |bits| {
        let config = CutoffFreeConfig::new(params.lambda_sq_int(), params.n_modes, bits);
        let (semantic, logical) = certificate_key(&config, &scalar_backend, manifests, options)?;
        Ok::<_, anyhow::Error>((config, semantic, logical))
    };
    // Prefer the lowest admitted existing rung. The normal resolver still checks
    // the payload and dependencies, promotes remote hits, and stages publication.
    let mut first = 0;
    if cache.mode.consults_overlays_for_route_selection() {
        for (index, &bits) in precisions.iter().enumerate() {
            let (_, semantic, logical) = identity(bits)?;
            if crate::ccm::hp::accepted_identity_exists(&semantic, &logical, cache)? {
                first = index;
                break;
            }
        }
    }
    try_assembly_precisions(&precisions[first..], |bits| {
        let (config, semantic, logical) = identity(bits)?;
        resolve_at_precision(
            &config,
            &semantic,
            &logical,
            &scalar_backend,
            manifests,
            [
                even_ground_guide,
                even_first_excited_guide,
                odd_ground_guide,
            ],
            options,
            &source_identity,
            cache,
            bits > precision_bits,
        )
    })
}

fn certificate_key(
    config: &CutoffFreeConfig,
    scalar_backend: &str,
    [even_manifest, odd_manifest, gap_manifest]: [&ArtifactManifest; 3],
    options: CcmSectorGapCertificationOptions,
) -> Result<(SemanticKeyEnvelope, String)> {
    let semantic_key = semantic_key(
        config,
        scalar_backend,
        [
            &even_manifest.content_digest,
            &odd_manifest.content_digest,
            &gap_manifest.content_digest,
        ],
        options,
    );
    let logical_key = format!(
        "ccm/sector-gap-replay-certificate/{}/{}/{}/{}",
        config.integer_cutoff_c,
        config.modes,
        config.precision_bits,
        semantic_key.digest()?.0,
    );
    Ok((semantic_key, logical_key))
}

pub(super) fn semantic_key(
    config: &CutoffFreeConfig,
    scalar_backend: &str,
    [even, odd, gap]: [&ContentDigest; 3],
    options: CcmSectorGapCertificationOptions,
) -> SemanticKeyEnvelope {
    SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_gap_certificate".to_owned(),
        mathematical_semantics_version: COMPACT_SEMANTICS.to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "representation": "compact_analytic_replay_v1",
            "integer_cutoff_c": config.integer_cutoff_c,
            "n_modes": config.modes,
            "precision_bits": config.precision_bits,
            "geometric_terms": config.geometric_terms,
            "assembly_semantics": ASSEMBLY_SEMANTICS,
            "full_inertia_semantics": INERTIA_SEMANTICS,
            "selected_inertia_semantics": SELECTED_INERTIA_SEMANTICS,
            "interval_matrix_semantics": INTERVAL_MATRIX_SEMANTICS,
            "relative_enclosure_bits": options.relative_enclosure_bits,
            "maximum_bracket_expansions": options.maximum_bracket_expansions,
            "scalar_backend": scalar_backend,
        }),
        normalization: Some("orthonormal_reflection_parity_sectors".to_owned()),
        target: Some(
            "finite_ccm_conditional_parity_simplicity_and_unconditional_full_inertia".to_owned(),
        ),
        subspace: Some("even_vs_odd".to_owned()),
        source_data_identities: BTreeMap::from([
            ("ccm_even_sector_spectrum".to_owned(), even.clone()),
            ("ccm_odd_sector_spectrum".to_owned(), odd.clone()),
            ("ccm_sector_gap".to_owned(), gap.clone()),
        ]),
        algorithm_semantics: Some(COMPACT_SEMANTICS.to_owned()),
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_at_precision(
    config: &CutoffFreeConfig,
    semantic_key: &SemanticKeyEnvelope,
    logical_key: &str,
    scalar_backend: &str,
    [even_manifest, odd_manifest, gap_manifest]: [&ArtifactManifest; 3],
    [even_ground_guide, even_first_excited_guide, odd_ground_guide]: [&Float; 3],
    options: CcmSectorGapCertificationOptions,
    source_identity: &ReconstructionSourceIdentity,
    cache: &ArtifactCacheContext<'_>,
    escalated: bool,
) -> Result<ArtifactExecutionCacheResult<CompactCcmSectorGapCertificate>> {
    let precision_bits = config.precision_bits;
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_gap_replay_certificate.resolve_or_compute",
        semantic_key,
        logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Certified,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.17.1")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_gap_certificate".to_owned()),
            (
                "representation".to_owned(),
                "compact_analytic_replay_v1".to_owned(),
            ),
            (
                "certification_scope".to_owned(),
                "finite_cutoff_free_matrix".to_owned(),
            ),
        ]),
        provenance_digest: Some(gap_manifest.content_digest.clone()),
        production_sink: cache.production_sink,
    };
    let produced_digest = std::cell::RefCell::new(None);
    let prove_reused = cache.mode.compares_against_reference()
        || cache.requested_assurance != xc_core::AssuranceLevel::Computed;
    let mut resolution_limit = None;
    let result = resolve_or_compute_json_artifact_with_assessment(
        &request,
        || {
            let produce = || -> Result<CompactCcmSectorGapCertificate> {
                // Charge the larger computation only on a fresh fallback.
                let resources = escalated.then(runtime_resources).transpose()?;
                let _scope = resources.as_ref().map(|r| r.enter());
                let _workspace = if escalated {
                    reserve_certificate_workspace(
                        config.modes,
                        (precision_bits + 1024).min(MAX_ASSEMBLY_BITS),
                        config.integer_cutoff_c,
                        config.geometric_terms,
                    )?
                } else {
                    None
                };
                let matrix = super::super::super::cutoff_free::assemble(config)?;
                let component_digest = matrix.component_evidence_digest()?;
                matrix.retain_binding(component_digest.clone())?;
                let full_inertia = matrix.portable_inertia_certificate()?;
                if full_inertia.zero_or_unresolved != 0 {
                    return Err(AssemblyResolutionLimit(format!(
                        "compact certification requires conclusive full inertia; {} of {} counts unresolved",
                        full_inertia.zero_or_unresolved, full_inertia.dimension
                    )).into());
                }
                let terms = matrix.config.geometric_terms;
                let backend = matrix.scalar_backend.clone();
                drop(matrix);
                let full = build_certificate_from_tau(
                    config.integer_cutoff_c.to_string(),
                    config.integer_cutoff_c,
                    config.modes,
                    precision_bits,
                    terms,
                    backend,
                    component_digest,
                    full_inertia,
                    even_ground_guide,
                    even_first_excited_guide,
                    odd_ground_guide,
                    even_manifest.content_digest.clone(),
                    odd_manifest.content_digest.clone(),
                    gap_manifest.content_digest.clone(),
                    options,
                )?;
                let proof = verify_retained_sector_gap(&full, true);
                if !proof.valid || !proof.mathematical_claim_verified {
                    bail!(
                        "compact certificate production proof failed: {}",
                        proof.errors.join("; ")
                    );
                }
                // No serialized legacy certificate, pivot trace or dense matrix
                // is sent to the cache or staging sink.
                CompactCcmSectorGapCertificate::from_portable(&full, source_identity.clone())
            };
            let value = produce().map_err(|error| {
                let failure = CacheError::InvalidManifest(format!(
                    "compact sector-gap certification failed: {error:#}"
                ));
                // Preserve the numerical result across the cache callback. A
                // cache, replay, or staging error must never advance the ladder.
                resolution_limit = error.downcast::<AssemblyResolutionLimit>().ok();
                failure
            })?;
            let digest = ContentDigest::sha256_json(&value)?;
            let evidence = if let Some(sink) = cache.production_sink {
                sink.record_evidence(
                    "ccm-sector-gap-replay-certificate",
                    &serde_json::to_vec(&value)?,
                )?
            } else {
                digest.clone()
            };
            *produced_digest.borrow_mut() = Some(digest);
            let mut dependencies = [even_manifest, odd_manifest, gap_manifest]
                .into_iter()
                .map(|manifest| DependencyRef {
                    key: manifest.key.clone(),
                    content_digest: manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                })
                .collect::<Vec<_>>();
            dependencies.sort_by(|left, right| {
                (
                    &left.key.kind,
                    &left.key.logical_key,
                    &left.key.parameters_digest.0,
                    &left.content_digest.0,
                )
                    .cmp(&(
                        &right.key.kind,
                        &right.key.logical_key,
                        &right.key.parameters_digest.0,
                        &right.content_digest.0,
                    ))
            });
            Ok((
                value,
                dependencies,
                ArtifactProductionAssessment {
                    achieved_assurance: ArtifactAssuranceState::Certified,
                    evidence_digests: vec![evidence],
                },
            ))
        },
        |value| {
            let validate = || -> Result<()> {
                value.validate_structure()?;
                if value.integer_cutoff_c != config.integer_cutoff_c
                    || value.n_modes != config.modes
                    || value.precision_bits != precision_bits
                    || value.geometric_terms != config.geometric_terms
                    || value.scalar_backend != scalar_backend
                    || value.full_inertia_semantics != INERTIA_SEMANTICS
                    || value.relative_enclosure_bits != options.relative_enclosure_bits
                    || value.maximum_bracket_expansions != options.maximum_bracket_expansions
                    || value.guide_even_spectrum_content_digest != even_manifest.content_digest
                    || value.guide_odd_spectrum_content_digest != odd_manifest.content_digest
                    || value.guide_sector_gap_content_digest != gap_manifest.content_digest
                {
                    bail!("compact certificate differs from its request identity");
                }
                let fresh =
                    produced_digest.borrow().as_ref() == Some(&ContentDigest::sha256_json(value)?);
                if !fresh && prove_reused {
                    let proof = verify_compact_ccm_sector_gap_certificate(value);
                    if !proof.valid || !proof.mathematical_claim_verified {
                        bail!(
                            "compact certificate replay failed: {}",
                            proof.errors.join("; ")
                        );
                    }
                }
                Ok(())
            };
            validate().map_err(|error| {
                CacheError::InvalidManifest(format!(
                    "compact sector-gap admission failed: {error:#}"
                ))
            })
        },
    );
    if let Some(limit) = resolution_limit {
        return Err(limit.into());
    }
    Ok(result?)
}

#[cfg(test)]
mod tests;
