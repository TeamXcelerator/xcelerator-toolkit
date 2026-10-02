//! Root-by-root certification for retained CCM runs.
//!
//! Every requested ordinal receives a row. Certifiable ordinals carry a
//! replayable finite-source enclosure; every other ordinal keeps its computed
//! value and status with the reason it was not certified. Numerical
//! limitations never discard retained computed data.
use super::super::certified_roots::{
    validate_production_independent_ccm_root_certificate_structure, IndependentCcmRootTarget,
    PartialIndependentCcmRootCertification,
};
use super::*;

pub const ROOT_CERTIFICATION_REPORT_SEMANTICS: &str = "ccm-root-certification-report-v0.16.0-v1";
const ROOT_CERTIFICATION_REPORT_SCOPE: &str = "certified rows are interval enclosures of roots of the exact stored point secular source; computed_not_certified rows retain the computed value and solver status without a certificate";

/// One requested root ordinal.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmRootCertificationRow {
    pub ordinal: usize,
    /// `certified_finite_enclosure`, `computed_not_certified`, or
    /// `not_computed_not_certified`.
    pub outcome: String,
    /// `converged`, `stagnated`, `approximate`, `failed`, or `absent`.
    pub computed_status: String,
    pub computed_value: Option<String>,
    pub certified_lower: Option<String>,
    pub certified_upper: Option<String>,
    pub certified_precision_bits: Option<u32>,
    /// `inside`, `outside`, `no_computed_value`, or `not_certified`.
    pub computed_agreement: String,
    /// When a computed value lies in another ordinal's certified enclosure.
    pub computed_matches_certified_ordinal: Option<usize>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmRootCertificationRequest {
    pub target: IndependentCcmRootTarget,
    pub first_index: usize,
    pub last_index: usize,
    pub expected_rows: usize,
    pub isolation_bits: u32,
    pub interval_newton: xc_root::IntervalNewtonOptions,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmRootCertificationReport {
    pub schema_version: u32,
    pub semantics: String,
    pub claim_scope: String,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub secular_source_content_digest: String,
    pub root_range_content_digest: String,
    pub root_selection_digest: String,
    pub request: CcmRootCertificationRequest,
    /// `certified_finite_enclosure` when every row is certified, otherwise
    /// `partial_certification`.
    pub outcome: String,
    /// Whole-source limitation, when certification could not start.
    pub reason: Option<String>,
    pub certified_rows: usize,
    pub computed_not_certified_rows: usize,
    pub unresolved_rows: usize,
    pub computed_outside_certified_rows: usize,
    pub rows: Vec<CcmRootCertificationRow>,
    pub certification: PartialIndependentCcmRootCertification,
}

impl CcmRootCertificationReport {
    /// The single certificate covering the whole request, when one exists.
    pub fn complete_certificate(
        &self,
    ) -> Option<&super::super::certified_roots::ProductionIndependentCcmRootCertificate> {
        (self.outcome == "certified_finite_enclosure" && self.certification.certificates.len() == 1)
            .then(|| &self.certification.certificates[0])
    }
}

fn computed_status(result: &EigenvalueResult) -> &'static str {
    match result {
        EigenvalueResult::Converged(_) => "converged",
        EigenvalueResult::Stagnated(_) => "stagnated",
        EigenvalueResult::Approximate(_) => "approximate",
        EigenvalueResult::Failed { .. } => "failed",
    }
}

fn requested_ordinals(
    target: &IndependentCcmRootTarget,
    primary: &HighPrecResult,
) -> (usize, usize) {
    match target {
        IndependentCcmRootTarget::Prefix { count } => (1, *count),
        IndependentCcmRootTarget::IndexRange { first, last } => (*first, *last),
        // A height window has no ordinals before counting; report the
        // computed roots retained for it.
        IndependentCcmRootTarget::PositiveHeightWindow { .. } => (
            primary.first_positive_root_index,
            primary.first_positive_root_index + primary.eigenvalues_pos.len().max(1) - 1,
        ),
    }
}

#[cfg(feature = "arb")]
fn certify_partially(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    weights: &[Float],
    options: &CcmRootCertificationOptions,
    first: usize,
    last: usize,
) -> Result<PartialIndependentCcmRootCertification> {
    use super::super::certified_roots::{
        certify_production_independent_ccm_roots_at_cutoff,
        certify_production_independent_ccm_roots_partially,
    };
    let cutoff = lambda_squared_cache_identity(params);
    if let IndependentCcmRootTarget::PositiveHeightWindow { .. } = options.target {
        let mut partial = unavailable(first, last, String::new());
        partial.source_limitation = None;
        partial.uncertified.clear();
        match certify_production_independent_ccm_roots_at_cutoff(
            weights,
            &cutoff,
            params.n_modes,
            &options.target,
            cfg.precision_bits,
            options.isolation_bits,
            &options.interval_newton,
        ) {
            Ok(certificate) => partial.certificates.push(certificate),
            Err(error) => return Ok(unavailable(first, last, format!("{error:#}"))),
        }
        return Ok(partial);
    }
    certify_production_independent_ccm_roots_partially(
        weights,
        &cutoff,
        params.n_modes,
        first,
        last,
        &options.target,
        cfg.precision_bits,
        options.isolation_bits,
        &options.interval_newton,
    )
}

#[cfg(not(feature = "arb"))]
fn certify_partially(
    _params: &CcmParams,
    _cfg: &HighPrecConfig,
    _weights: &[Float],
    _options: &CcmRootCertificationOptions,
    first: usize,
    last: usize,
) -> Result<PartialIndependentCcmRootCertification> {
    Ok(unavailable(
        first,
        last,
        "root certification requires the xc-spectral arb feature".to_owned(),
    ))
}

fn unavailable(
    first: usize,
    last: usize,
    reason: String,
) -> PartialIndependentCcmRootCertification {
    PartialIndependentCcmRootCertification {
        first_index: first,
        last_index: last,
        available_positive_roots: None,
        source_limitation: Some(reason.clone()),
        certificates: Vec::new(),
        uncertified: (first..=last)
            .map(|ordinal| (ordinal, reason.clone()))
            .collect(),
    }
}

/// Build the per-root report without a cache.
pub(super) fn build_root_certification_report(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    primary: &HighPrecResult,
    weights: &[Float],
    options: &CcmRootCertificationOptions,
    secular_digest: &ContentDigest,
    root_digest: &ContentDigest,
) -> Result<CcmRootCertificationReport> {
    let (mut first, mut last) = requested_ordinals(&options.target, primary);
    if first == 0 || first > last {
        bail!("CCM root certification request has no ordinal range");
    }
    let mut certification = certify_partially(params, cfg, weights, options, first, last)?;
    let mut enclosures = BTreeMap::new();
    for certificate in &certification.certificates {
        let start = certificate.first_selected_positive_index.ok_or_else(|| {
            anyhow::anyhow!("CCM root certificate does not assign positive ordinals")
        })?;
        for (offset, root) in certificate.try_selected_roots()?.iter().enumerate() {
            enclosures.insert(start + offset, root);
        }
    }
    // A height window's ordinals are known only after counting: rows cover
    // every computed and every certified ordinal.
    if let IndependentCcmRootTarget::PositiveHeightWindow { .. } = options.target {
        if let (Some(&low), Some(&high)) = (enclosures.keys().next(), enclosures.keys().last()) {
            first = first.min(low);
            last = last.max(high);
            certification.first_index = first;
            certification.last_index = last;
        }
    }
    let computed_at = |ordinal: usize| {
        ordinal
            .checked_sub(primary.first_positive_root_index)
            .and_then(|offset| primary.eigenvalues_pos.get(offset))
    };
    let mut rows = Vec::with_capacity(last - first + 1);
    for ordinal in first..=last {
        let computed = computed_at(ordinal);
        let value = computed.and_then(EigenvalueResult::value);
        let mut row = CcmRootCertificationRow {
            ordinal,
            outcome: String::new(),
            computed_status: computed.map_or("absent", computed_status).to_owned(),
            computed_value: value.map(lossless_hp_decimal),
            certified_lower: None,
            certified_upper: None,
            certified_precision_bits: None,
            computed_agreement: "not_certified".to_owned(),
            computed_matches_certified_ordinal: None,
            reason: None,
        };
        if let Some(root) = enclosures.get(&ordinal) {
            row.outcome = "certified_finite_enclosure".to_owned();
            row.certified_lower = Some(root.lower.clone());
            row.certified_upper = Some(root.upper.clone());
            row.certified_precision_bits = Some(root.precision_bits);
            row.computed_agreement = match value {
                None => "no_computed_value".to_owned(),
                Some(v)
                    if stored_root_in_decimal_interval(
                        v,
                        &root.lower,
                        &root.upper,
                        root.precision_bits,
                    )? =>
                {
                    "inside".to_owned()
                }
                Some(_) => "outside".to_owned(),
            };
        } else {
            row.outcome = if value.is_some() {
                "computed_not_certified"
            } else {
                "not_computed_not_certified"
            }
            .to_owned();
            row.reason = certification.uncertified.get(&ordinal).cloned();
        }
        if row.computed_agreement == "outside" || row.computed_agreement == "not_certified" {
            if let Some(v) = value {
                for (other, root) in &enclosures {
                    if *other != ordinal
                        && stored_root_in_decimal_interval(
                            v,
                            &root.lower,
                            &root.upper,
                            root.precision_bits,
                        )?
                    {
                        row.computed_matches_certified_ordinal = Some(*other);
                        break;
                    }
                }
            }
        }
        rows.push(row);
    }
    let count = |outcome: &str| rows.iter().filter(|row| row.outcome == outcome).count();
    let certified_rows = count("certified_finite_enclosure");
    Ok(CcmRootCertificationReport {
        schema_version: 1,
        semantics: ROOT_CERTIFICATION_REPORT_SEMANTICS.to_owned(),
        claim_scope: ROOT_CERTIFICATION_REPORT_SCOPE.to_owned(),
        lambda_squared: lambda_squared_cache_identity(params),
        n_modes: params.n_modes,
        precision_bits: cfg.precision_bits,
        secular_source_content_digest: secular_digest.0.clone(),
        root_range_content_digest: root_digest.0.clone(),
        root_selection_digest: root_selection_digest(&primary.eigenvalues_pos)?.0,
        request: CcmRootCertificationRequest {
            target: options.target.clone(),
            first_index: first,
            last_index: last,
            expected_rows: last - first + 1,
            isolation_bits: options.isolation_bits,
            interval_newton: options.interval_newton.clone(),
        },
        outcome: if certified_rows == rows.len() {
            "certified_finite_enclosure"
        } else {
            "partial_certification"
        }
        .to_owned(),
        reason: certification.source_limitation.clone(),
        certified_rows,
        computed_not_certified_rows: count("computed_not_certified"),
        unresolved_rows: count("not_computed_not_certified"),
        computed_outside_certified_rows: rows
            .iter()
            .filter(|row| row.computed_agreement == "outside")
            .count(),
        rows,
        certification,
    })
}

fn validate_root_certification_report(
    report: &CcmRootCertificationReport,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    options: &CcmRootCertificationOptions,
    secular_digest: &ContentDigest,
    root_digest: &ContentDigest,
    selection_digest: &ContentDigest,
) -> std::result::Result<(), CacheError> {
    let invalid = |reason: &str| Err(CacheError::InvalidManifest(reason.to_owned()));
    if report.schema_version != 1
        || report.semantics != ROOT_CERTIFICATION_REPORT_SEMANTICS
        || report.claim_scope != ROOT_CERTIFICATION_REPORT_SCOPE
        || report.lambda_squared != lambda_squared_cache_identity(params)
        || report.n_modes != params.n_modes
        || report.precision_bits != cfg.precision_bits
        || report.secular_source_content_digest != secular_digest.0
        || report.root_range_content_digest != root_digest.0
        || report.root_selection_digest != selection_digest.0
        || report.request.target != options.target
        || report.request.isolation_bits != options.isolation_bits
        || report.request.interval_newton != options.interval_newton
        || report.request.first_index == 0
        || report.request.first_index > report.request.last_index
        || report.request.expected_rows != report.rows.len()
        || report.request.last_index - report.request.first_index + 1 != report.rows.len()
    {
        return invalid("CCM root certification report does not match its semantic identity");
    }
    let mut certified = 0;
    for (offset, row) in report.rows.iter().enumerate() {
        let certified_row = row.outcome == "certified_finite_enclosure";
        certified += usize::from(certified_row);
        if row.ordinal != report.request.first_index + offset
            || certified_row != row.certified_lower.is_some()
            || !matches!(
                row.outcome.as_str(),
                "certified_finite_enclosure"
                    | "computed_not_certified"
                    | "not_computed_not_certified"
            )
        {
            return invalid("CCM root certification report rows are inconsistent");
        }
    }
    if certified != report.certified_rows {
        return invalid("CCM root certification report counts are inconsistent");
    }
    for certificate in &report.certification.certificates {
        validate_production_independent_ccm_root_certificate_structure(certificate)
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
        if certificate.modes != params.n_modes
            || certificate.precision_bits != cfg.precision_bits
            || certificate.cutoff_text() != lambda_squared_cache_identity(params)
        {
            return invalid("embedded CCM root certificate does not match the report source");
        }
    }
    Ok(())
}

/// Resolve or compute the per-root certification report for a retained run,
/// returning it with its retained manifest. With no cache the report is
/// computed directly and has no manifest.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_root_certification_report_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    primary: &HighPrecResult,
    weights: &[Float],
    root_manifest: Option<&ArtifactManifest>,
    secular_manifest: Option<&ArtifactManifest>,
    options: &CcmRootCertificationOptions,
    cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<(CcmRootCertificationReport, Option<ArtifactManifest>)> {
    let digest = |manifest: Option<&ArtifactManifest>| {
        manifest.map_or_else(
            || ContentDigest("unretained".to_owned()),
            |manifest| manifest.content_digest.clone(),
        )
    };
    let compute = || {
        build_root_certification_report(
            params,
            cfg,
            primary,
            weights,
            options,
            &digest(secular_manifest),
            &digest(root_manifest),
        )
    };
    let Some(cache) = cache else {
        return Ok((compute()?, None));
    };
    // The secular source is required; a run that acquired no roots has no
    // root-range artifact and certifies from the source alone.
    let Some(secular_manifest) = secular_manifest else {
        bail!("managed CCM root certification requires a retained secular-source manifest");
    };
    let root_digest = root_manifest.map(|manifest| manifest.content_digest.clone());
    let selection_digest = root_selection_digest(&primary.eigenvalues_pos)?;
    let parity_policy = cfg.effective_parity_policy();
    let mut resolved_parameters = serde_json::json!({
        "lambda_squared": lambda_squared_cache_identity(params),
        "n_modes": params.n_modes,
        "precision_bits": cfg.precision_bits,
        "force_even": parity_policy.legacy_force_even(),
        "root_range_content_digest": root_digest.as_ref().map(|digest| digest.0.clone()),
        "secular_source_content_digest": secular_manifest.content_digest.0,
        "root_selection_digest": selection_digest.0,
        "target": options.target,
        "isolation_bits": options.isolation_bits,
        "interval_newton": options.interval_newton,
    });
    add_adaptive_parity_parameter(&mut resolved_parameters, parity_policy);
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_root_certification_report".to_owned(),
        mathematical_semantics_version: ROOT_CERTIFICATION_REPORT_SEMANTICS.to_owned(),
        resolved_mathematical_parameters: resolved_parameters,
        normalization: Some("sum_xi_equals_sqrt_log_lambda_squared".to_owned()),
        target: Some("per_ordinal_finite_source_root_certification".to_owned()),
        subspace: parity_policy.semantic_subspace(),
        source_data_identities: root_digest
            .iter()
            .map(|digest| ("ccm_root_range".to_owned(), digest.clone()))
            .chain(std::iter::once((
                "ccm_secular_source".to_owned(),
                secular_manifest.content_digest.clone(),
            )))
            .collect(),
        algorithm_semantics: Some(
            "whole_range_then_bisected_flint_arb_or_pole_gap_census_isolation_and_interval_newton_v2".to_owned(),
        ),
    };
    let semantic_digest = semantic_key.digest()?;
    let logical_key = format!(
        "ccm/root-certification-report/{}/{}/{}/{}/{}",
        lambda_squared_cache_identity(params),
        params.n_modes,
        cfg.precision_bits,
        parity_policy.cache_label(),
        semantic_digest.0
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.root_certification_report.resolve_or_compute",
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
            (
                "artifact".to_owned(),
                "root_certification_report".to_owned(),
            ),
        ]),
        provenance_digest: Some(
            root_digest
                .clone()
                .unwrap_or_else(|| secular_manifest.content_digest.clone()),
        ),
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_dependencies(
        &request,
        || {
            let report = compute().map_err(|error| {
                CacheError::InvalidManifest(format!("CCM root certification report: {error:#}"))
            })?;
            Ok((
                report,
                canonical_dependency_refs(
                    root_manifest
                        .iter()
                        .copied()
                        .cloned()
                        .chain(std::iter::once(secular_manifest.clone()))
                        .collect(),
                ),
            ))
        },
        |report| {
            validate_root_certification_report(
                report,
                params,
                cfg,
                options,
                &secular_manifest.content_digest,
                &digest(root_manifest),
                &selection_digest,
            )
        },
    )?;
    let manifest = resolved.produced_manifest.or(resolved.reused_manifest);
    Ok((resolved.value, manifest))
}

#[cfg(all(test, feature = "arb"))]
mod tests {
    use super::super::super::certified_roots::verify_production_independent_ccm_root_certificate;
    use super::*;

    fn source_only(params: &CcmParams) -> (HighPrecConfig, HighPrecResult) {
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = 192;
        cfg.quad_points = MIN_QUAD_POINTS;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let (source, _) = run_inner_retaining_source(
            params,
            &cfg,
            RootAcquisition::SourceOnly,
            CcmCacheRoute::Standalone,
            None,
        )
        .unwrap();
        (cfg, source)
    }

    fn computed(outcome: &str, value: Float) -> EigenvalueResult {
        let refinement = RootRefinement {
            value: value.clone(),
            diagnostics: RootRefinementDiagnostics {
                iterations: 1,
                final_correction: Float::with_val(value.prec(), 0),
                residual: Float::with_val(value.prec(), 0),
                achieved_decimal_digits: Float::with_val(value.prec(), 30),
            },
        };
        match outcome {
            "converged" => EigenvalueResult::Converged(refinement),
            "stagnated" => EigenvalueResult::Stagnated(refinement),
            _ => EigenvalueResult::Approximate(refinement),
        }
    }

    fn midpoint(row: &CcmRootCertificationRow, p: u32) -> Float {
        let lower = Float::with_val(
            p,
            Float::parse(row.certified_lower.as_ref().unwrap()).unwrap(),
        );
        let upper = Float::with_val(
            p,
            Float::parse(row.certified_upper.as_ref().unwrap()).unwrap(),
        );
        (lower + upper) / 2u32
    }

    fn report(
        params: &CcmParams,
        cfg: &HighPrecConfig,
        primary: &HighPrecResult,
        options: &CcmRootCertificationOptions,
    ) -> CcmRootCertificationReport {
        resolve_root_certification_report_via_cache(
            params,
            cfg,
            primary,
            &primary.xi,
            None,
            None,
            options,
            None,
        )
        .unwrap()
        .0
    }

    #[test]
    fn partial_requests_keep_every_computed_root_and_certify_the_rest() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let (cfg, mut primary) = source_only(&params);
        let p = cfg.precision_bits;
        primary.first_positive_root_index = 1;
        primary.eigenvalues_pos.clear();
        let probe = report(
            &params,
            &cfg,
            &primary,
            &CcmRootCertificationOptions::for_decimal_digits(
                IndependentCcmRootTarget::Prefix { count: 1 },
                30,
            )
            .unwrap(),
        );
        let available = probe.certification.available_positive_roots.unwrap();
        assert!(available >= 2);
        let options = CcmRootCertificationOptions::for_decimal_digits(
            IndependentCcmRootTarget::Prefix {
                count: available + 2,
            },
            30,
        )
        .unwrap();
        let blank = report(&params, &cfg, &primary, &options);
        assert_eq!(blank.certified_rows, available);
        assert_eq!(blank.unresolved_rows, 2);
        // Converged values at the certified midpoints, one stagnated value at
        // the wrong root, and one approximate root beyond certification reach.
        let mut values = (0..available)
            .map(|k| computed("converged", midpoint(&blank.rows[k], p)))
            .collect::<Vec<_>>();
        values[1] = computed("stagnated", midpoint(&blank.rows[0], p));
        values.push(computed("approximate", Float::with_val(p, 1_000_000)));
        primary.eigenvalues_pos = values;
        let result = report(&params, &cfg, &primary, &options);
        assert_eq!(result.rows.len(), available + 2);
        assert_eq!(result.outcome, "partial_certification");
        assert_eq!(result.certified_rows, available);
        assert_eq!(result.computed_not_certified_rows, 1);
        assert_eq!(result.unresolved_rows, 1);
        assert_eq!(result.computed_outside_certified_rows, 1);
        assert_eq!(result.rows[0].computed_agreement, "inside");
        assert_eq!(result.rows[1].computed_status, "stagnated");
        assert_eq!(result.rows[1].computed_agreement, "outside");
        assert_eq!(result.rows[1].computed_matches_certified_ordinal, Some(1));
        let beyond = &result.rows[available];
        assert_eq!(beyond.outcome, "computed_not_certified");
        assert_eq!(beyond.computed_status, "approximate");
        assert!(beyond.computed_value.is_some());
        assert!(beyond.reason.as_ref().unwrap().contains("beyond"));
        assert_eq!(
            result.rows[available + 1].outcome,
            "not_computed_not_certified"
        );
        assert!(result.complete_certificate().is_none());
        for certificate in &result.certification.certificates {
            verify_production_independent_ccm_root_certificate(certificate).unwrap();
        }
        let coverage =
            xc_cache::NumericalCoverage::from_value(&serde_json::to_value(&result).unwrap());
        assert_eq!(coverage.resolved_rows, available);
        assert_eq!(coverage.qualified_rows, 1);
        // A height window covering every root reports every certified
        // ordinal, not only the computed ones.
        let upper = Float::with_val(p, midpoint(&blank.rows[available - 1], p) + 1u32);
        primary.eigenvalues_pos = vec![computed("converged", midpoint(&blank.rows[0], p))];
        let window = CcmRootCertificationOptions::for_decimal_digits(
            IndependentCcmRootTarget::PositiveHeightWindow {
                lower: "0.001".into(),
                upper: upper.to_string_radix(10, Some(30)),
            },
            30,
        )
        .unwrap();
        let windowed = report(&params, &cfg, &primary, &window);
        assert_eq!(windowed.rows.len(), available, "{windowed:?}");
        assert_eq!(windowed.certified_rows, available);
        assert_eq!(windowed.rows[0].computed_agreement, "inside");
        assert_eq!(coverage.unresolved_rows, 1);
        assert_eq!(coverage.outcome, "partial_unresolved");

        // Whatever a stricter request certifies, every computed value is
        // retained and every uncertified row states its reason.
        let mut strict = options.clone();
        strict.interval_newton.width_tolerance = xc_core::DecimalLiteral::new("1e-400").unwrap();
        let strict = report(&params, &cfg, &primary, &strict);
        assert_eq!(strict.rows.len(), available + 2);
        for (row, value) in strict.rows.iter().zip(&primary.eigenvalues_pos) {
            assert_eq!(
                row.computed_value.as_deref(),
                Some(lossless_hp_decimal(value.value().unwrap()).as_str())
            );
            assert!(row.outcome == "certified_finite_enclosure" || row.reason.is_some());
        }
    }

    #[test]
    fn fractional_cutoffs_are_certified_and_replay() {
        let params = CcmParams::from_lambda_sq_fractional(12.5, 6);
        let (cfg, mut primary) = source_only(&params);
        primary.first_positive_root_index = 1;
        primary.eigenvalues_pos.clear();
        let options = CcmRootCertificationOptions::for_decimal_digits(
            IndependentCcmRootTarget::Prefix { count: 1 },
            30,
        )
        .unwrap();
        let result = report(&params, &cfg, &primary, &options);
        assert_eq!(
            result.certified_rows,
            1,
            "{:?} {:?} {:?}",
            result.reason,
            result.certification.available_positive_roots,
            result
                .rows
                .iter()
                .map(|row| &row.reason)
                .collect::<Vec<_>>()
        );
        let certificate = result.complete_certificate().unwrap();
        assert_eq!(
            certificate.lambda_squared_decimal.as_deref(),
            Some(lambda_squared_cache_identity(&params).as_str())
        );
        assert_eq!(certificate.integer_cutoff_c, 12);
        verify_production_independent_ccm_root_certificate(certificate).unwrap();
        let integer = CcmParams::from_lambda_sq_integer(13, 10);
        let (cfg, mut primary) = source_only(&integer);
        primary.eigenvalues_pos.clear();
        let result = report(&integer, &cfg, &primary, &options);
        let complete = result.complete_certificate();
        assert!(
            complete.is_some(),
            "{} {:?} {:?}",
            result.outcome,
            result.certification.available_positive_roots,
            result.rows
        );
        assert!(complete.unwrap().lambda_squared_decimal.is_none());
    }
}
