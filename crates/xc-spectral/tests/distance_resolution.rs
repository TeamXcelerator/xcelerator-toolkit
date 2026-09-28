#![cfg(feature = "hp")]
use rug::Float;
use xc_cache::{
    ArtifactCacheContext, ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheQuality,
    CacheResolver, CacheVisibility, CertificationFailurePolicy, FilesystemCacheStore,
    ToolkitVersion,
};
use xc_numerics::grid_integral::{GridVariable, UniformGridScheme};
use xc_spectral::distance::{
    hp::capture_ccm_distance_with_resolution_evidence_via_cache, WeightedIntegrationRule,
};

struct LocalFixture(std::path::PathBuf);
impl Drop for LocalFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Real public capture route, including a warm reuse. The synthetic target is
// the normalized Gaussian sum with P=1, matching the reference mathematical case;
// no oracle or artifact bytes are copied from external records.
#[test]
#[ignore = "explicit HP capture qualification: Q=12000 and Q=30000 refinements"]
fn returned_grid_verdict_and_ladder_verdict_are_distinct_through_cache() {
    let root = std::env::temp_dir().join(format!(
        "xc-confirmed-distance-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let fixture = LocalFixture(root);
    let spec = fixture.0.join("target.json");
    std::fs::write(&spec, serde_json::to_vec(&serde_json::json!({
        "schema_version":1, "profile_id":"confirmed-distance-manufactured-v1",
        "base_series":{"term_input_power":0,"polynomial_coefficients":["1"],"minimum_terms":1,"maximum_terms":1000},
        "auxiliary_series":{"term_input_power":0,"polynomial_coefficients":["0","1"],"parameter_polynomial_coefficients":["1"],"minimum_terms":1,"maximum_terms":1000}
    })).unwrap()).unwrap();
    std::env::set_var("XC_TARGET_SPEC_FILE", &spec);
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "workstation",
            fixture.0.join("artifacts"),
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
        ordered_overlays: vec!["workstation".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let params = xc_spectral::ccm::CcmParams::from_lambda_sq_integer(5, 8);
    let mut cfg = xc_spectral::ccm::hp::HighPrecConfig::for_decimal_digits(60);
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let alpha = Float::with_val(cfg.precision_bits, 0.5);
    for (q, expected_base) in [(12_000, false), (30_000, true)] {
        let rules = [WeightedIntegrationRule::UniformGrid {
            scheme: UniformGridScheme::Trapezoid,
            variable: GridVariable::U,
            steps: q,
        }];
        let cold = capture_ccm_distance_with_resolution_evidence_via_cache(
            &params, &cfg, &alpha, &rules, 20, &cache,
        )
        .unwrap();
        let warm = capture_ccm_distance_with_resolution_evidence_via_cache(
            &params, &cfg, &alpha, &rules, 20, &cache,
        )
        .unwrap();
        assert_eq!(cold.distances[0].rule, rules[0]);
        assert_eq!(cold.distances[0].value, warm.distances[0].value);
        assert_eq!(cold.resolution_tolerance_met, Some(expected_base));
        assert_eq!(warm.resolution_tolerance_met, Some(expected_base));
        assert_eq!(cold.resolution_ladder_tolerance_met, Some(true));
        assert_eq!(warm.resolution_ladder_tolerance_met, Some(true));
        println!(
            "Q={q}: reported={:?}, final_pair={:?}, value={}",
            cold.resolution_tolerance_met,
            cold.resolution_ladder_tolerance_met,
            cold.distances[0].value.to_string_radix(10, Some(30))
        );
    }
}
