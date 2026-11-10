//! Small real capture integration; no remote or publication path.
use super::*;
use crate::ccm::sector_gap_certificate::{
    compact::ReconstructionSourceIdentity, CcmSectorGapCertificationOptions,
    SectorGapCertificateRef,
};
use xc_cache::{
    ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
    ZipJsonFilesystemCacheStore,
};

#[test]
fn compact_sector_certificate_capture_is_optional_reference_bound_and_request_fixed() {
    let dir = xc_core::test_support::TestDir::new("compact-retained-capture");
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(ZipJsonFilesystemCacheStore::new(
            "local",
            dir.to_path_buf(),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let cache = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let params = CcmParams::from_lambda_sq_integer(13, 4);
    let mut cfg = HighPrecConfig::for_decimal_digits(30);
    cfg.precision_bits = 192;
    cfg.n_eigenvalues = 1;
    cfg.quad_points = 32;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
    let seeds = xc_zeta::zeros::bundled_first_n_strings(1)
        .unwrap()
        .iter()
        .map(|s| Float::with_val(192, Float::parse(s).unwrap()))
        .collect::<Vec<_>>();
    let mut run = RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap();
    let mut absent = CcmResearchCaptureOptions::maximum(2);
    absent.distance_capture = None;
    absent.sector_gap_certificate_format = CcmSectorGapCertificateFormat::Compact(
        ReconstructionSourceIdentity::current("capture-fixture"),
    );
    assert!(absent.sector_gap_certification.is_none());
    assert!(run.ensure_sector_certificate(&absent, &cache).is_err());
    assert!(run.sector_certificate.is_none() && run.compact_sector_certificate.is_none());
    let options = absent.with_compact_sector_gap_certification(
        CcmSectorGapCertificationOptions::default(),
        ReconstructionSourceIdentity::current("capture-fixture"),
    );
    let first = run
        .capture_diagnostic("sector_gap_certificate", &options, &cache)
        .unwrap();
    assert!(first.value_reference.is_some());
    assert!(
        run.sector_certificate.is_none(),
        "compact path never memoizes a full portable proof"
    );
    let resolved = run.compact_sector_certificate.as_ref().unwrap();
    let manifest = resolved
        .produced_manifest
        .as_ref()
        .or(resolved.reused_manifest.as_ref())
        .unwrap()
        .clone();
    let compact = resolved.value.clone();
    assert!(xc_cache::json_payload_matches_manifest(&manifest, &compact).unwrap());
    assert!(serde_json::from_value::<
        crate::ccm::sector_gap_certificate::PortableCcmSectorGapCertificate,
    >(serde_json::to_value(&compact).unwrap())
    .is_err());
    let again = run
        .capture_diagnostic("sector_gap_certificate", &options, &cache)
        .unwrap();
    assert_eq!(first.value_reference, again.value_reference);
    let receipt = xc_cache::collect_capture(
        &serde_json::json!({}),
        vec!["sector_gap_certificate".into()],
        |_| Ok(first.clone()),
    )
    .unwrap();
    assert!(receipt.receipt.is_complete());
    assert!(receipt.measurements["sector_gap_certificate"]
        .value
        .is_null());
    assert!(receipt.measurements["sector_gap_certificate"]
        .value_reference
        .is_some());
    assert_eq!(
        receipt.numerical_coverage["sector_gap_certificate"].resolved_rows,
        1
    );

    // A separately requested legacy proof provides the comparison oracle, not
    // a payload paired with the compact manifest or published by its path.
    let full_resolved = certify_sector_gap_from_resolution(
        &params,
        &cfg,
        CcmSectorGapCertificationOptions::default(),
        run.sectors.as_ref().unwrap(),
        Some(&cache),
    )
    .unwrap();
    let full_manifest = full_resolved
        .produced_manifest
        .as_ref()
        .or(full_resolved.reused_manifest.as_ref())
        .unwrap();
    let full = &full_resolved.value;
    let input: crate::ccm::extended_research::ExternalResearchInputs = serde_json::from_value(serde_json::json!({
        "schema_version":1,"source_eigenpair":run.eigenpair.as_ref().unwrap().manifest.content_digest,
        "lambda_squared":"13","n_modes":4,"precision_bits":192,
        "convention_id":"compact-capture-fixture","definition_digest":ContentDigest::sha256(b"compact-capture-fixture"),
        "approximation_scope":"source-only qualification input; no analytic continuum or tail bound",
        "run_once":{"completion":{"contour":{"left":"-1","right":"1","bottom":"-0.1","top":"0.1","maximum_depth":12,"maximum_segments":256}}}
    })).unwrap();
    run.set_extended_research_inputs(input.clone()).unwrap();
    let state = run.admitted_state().unwrap();
    let full_error = crate::ccm::transform_enclosure::verified_state_error_ref(
        &state,
        SectorGapCertificateRef::Full(full),
        256,
    )
    .unwrap();
    let compact_error = crate::ccm::transform_enclosure::verified_state_error_ref(
        &state,
        SectorGapCertificateRef::Compact(&compact),
        256,
    )
    .unwrap();
    assert_eq!(full_error.lower(), compact_error.lower());
    assert_eq!(full_error.upper(), compact_error.upper());
    let mismatch = crate::ccm::extended_research::capture_extended_with_certificate_ref(
        "transform_enclosure",
        &state,
        None,
        None,
        Some(&input),
        Some((SectorGapCertificateRef::Compact(&compact), full_manifest)),
        &crate::ccm::extended_research::ExtensionOptions::for_source(&state),
        &[],
        &cache,
    )
    .err()
    .expect("compact payload must not pair with legacy full manifest");
    assert!(mismatch.to_string().contains("does not match its manifest"));
    drop(state);
    for id in [
        "capture_preflight",
        "transform_enclosure",
        "finite_root_budget",
    ] {
        let record = run.capture_diagnostic(id, &options, &cache).unwrap();
        // from_cached references the diagnostic artifact itself. Its manifest
        // carries the exact certificate source edge; receipts do not repeat
        // every ancestor as a top-level source.
        assert!(
            record
                .sources
                .iter()
                .any(|m| xc_cache::manifest_depends_on(m, &manifest).unwrap()),
            "{id} artifact must depend on the actual compact certificate manifest"
        );
        assert!(run.sector_certificate.is_none());
    }
    let mut full_request = options.clone();
    full_request.sector_gap_certificate_format = CcmSectorGapCertificateFormat::Full;
    assert!(run
        .capture_diagnostic("sector_gap_certificate", &full_request, &cache)
        .err()
        .expect("changed request rejected")
        .to_string()
        .contains("certification options changed"));
    let mut changed_source = options.clone();
    changed_source.sector_gap_certificate_format = CcmSectorGapCertificateFormat::Compact(
        ReconstructionSourceIdentity::current("different-source"),
    );
    assert!(run
        .capture_diagnostic("sector_gap_certificate", &changed_source, &cache)
        .err()
        .expect("changed request rejected")
        .to_string()
        .contains("certification options changed"));
}
