#![cfg(feature = "hp")]
use rug::Float;
use xc_cache::*;
use xc_spectral::ccm::{
    hp::{
        capture_run::RetainedCcmRun, CcmResearchCaptureOptions, CcmSectorAnalysisOptions,
        HighPrecConfig,
    },
    CcmParams,
};

#[test]
fn changing_sector_capture_request_must_not_silently_reuse_the_old_request() {
    let root_dir = xc_core::test_support::TestDir::new("fresh-capture-options");
    let root = root_dir.to_path_buf();
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(ZipJsonFilesystemCacheStore::new(
            "local",
            &root,
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
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let params = CcmParams::from_lambda_sq_integer(13, 16);
    let cfg = HighPrecConfig {
        n_eigenvalues: 1,
        ..HighPrecConfig::for_decimal_digits(40)
    };
    let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
    let seeds = xc_zeta::zeros::bundled_first_n_strings(1)
        .unwrap()
        .iter()
        .map(|s| Float::with_val(cfg.precision_bits, Float::parse(s).unwrap()))
        .collect::<Vec<_>>();
    let mut run = RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap();
    let mut options = CcmResearchCaptureOptions::maximum(2);
    options.sector_analysis = Some(CcmSectorAnalysisOptions::selected(2));
    let initial = run
        .capture_diagnostic("sector_analysis", &options, &cache)
        .unwrap();
    let repeated = run
        .capture_diagnostic("sector_analysis", &options, &cache)
        .unwrap();
    assert_eq!(initial.value, repeated.value);
    options.sector_analysis = Some(CcmSectorAnalysisOptions::cross_checked(3));
    let changed = run.capture_diagnostic("sector_analysis", &options, &cache);
    if let Err(error) = &changed {
        assert!(error.to_string().contains("options changed"));
    }
    options.sector_analysis = Some(CcmSectorAnalysisOptions::selected(2));
    assert_eq!(
        run.capture_diagnostic("sector_analysis", &options, &cache)
            .unwrap()
            .value,
        initial.value
    );
    // Recomputing the requested route or rejecting a changed request is safe.
    // An Ok result carrying the original count and selected-only route is not.
    std::fs::remove_dir_all(&root).unwrap();
    if let Ok(changed) = changed {
        let spectra = changed
            .value
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v.get("requested_eigenpairs").is_some())
            .collect::<Vec<_>>();
        assert_eq!(spectra.len(), 2, "both parity spectra must be present");
        let observed = spectra
            .iter()
            .map(|v| (&v["requested_eigenpairs"], &v["eigenvalue_route"]))
            .collect::<Vec<_>>();
        eprintln!("old and new payload equal={}; requested three cross-checked eigenpairs; observed={observed:?}",initial.value==changed.value);
        assert!(spectra.iter().all(|v|v["requested_eigenpairs"]==3 && v["eigenvalue_route"]=="cross_checked"),"changed sector count/route was silently ignored");
    }
}
