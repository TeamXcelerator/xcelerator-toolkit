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
    // A real retained run invokes every new Ultra producer and persists all
    // outcomes, including qualified absence, without another primary solve.
    let plan = xc_spectral::ccm::capture::CcmCapturePlan::ultra(2, 17).unwrap();
    let ids = xc_spectral::ccm::capture::FINITE_DIAGNOSTICS;
    let source = run
        .primary_sources()
        .into_iter()
        .find(|m| m.key.kind == "ccm_weil_eigenpair")
        .unwrap();
    let mut target = vec!["0"; 33];
    target[16] = "2";
    let input = serde_json::from_value(serde_json::json!({"schema_version":1,
        "source_eigenpair":source.content_digest,"lambda_squared":"13","n_modes":16,
        "precision_bits":cfg.precision_bits,"convention_id":"synthetic finite target",
        "definition_digest":ContentDigest::sha256(b"finite capture target"),"approximation_scope":"finite coefficients",
        "target":{"definition_digest":ContentDigest::sha256(b"target"),"evaluation_policy":"finite synthetic",
            "approximation_scope":"finite coefficient test","intervals":8,"values":vec!["2";9],"basis_values":[],
            "fixed_second_component":null,"raw_normalizer":"2","trial_coefficients":target}})).unwrap();
    run.set_extended_research_inputs(input).unwrap();
    let requested = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>();
    let result = capture_and_persist(
        &plan,
        requested.clone(),
        |id| run.capture_diagnostic_outcome(id, &options, &cache),
        &cache,
    )
    .unwrap();
    assert!(result.produced_manifest.is_some());
    assert_eq!(result.value.measurements.len(), ids.len());
    for id in ids {
        assert!(plan.receipt().unwrap().outcomes().contains_key(*id));
        let measurement = &result.value.measurements[*id];
        assert!(measurement.value_reference.is_some(), "{id}");
        let value = measurement_value(measurement, &resolver, &policy).unwrap();
        assert_eq!(value["data"]["diagnostic"], *id);
        if *id == "normalization_error_bound" {
            assert_eq!(value["data"]["outcome"], "computed");
            assert!(value["data"]["result"]["squared_difference"].is_string());
        }
        if *id == "trial_vector_energy" {
            assert_eq!(
                value["data"]["result"]["trial_series"]["provenance"]["target_center_normalizer"],
                "2"
            );
        }
        assert!(value["source_dependencies"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["key"]["kind"] == "ccm_weil_eigenpair"));
    }
    // A supplied target gives constrained_l1_fit its automatic reference
    // amplitude baseline, a certified finite enclosure.
    assert_eq!(
        result.value.coverage()["constrained_l1_fit"].outcome,
        "certified_finite_enclosure"
    );
    let replay = capture_and_persist(
        &plan,
        requested,
        |id| run.capture_diagnostic_outcome(id, &options, &cache),
        &cache,
    )
    .unwrap();
    assert_eq!(result.value, replay.value);
    assert!(replay.reused_manifest.is_some());
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
