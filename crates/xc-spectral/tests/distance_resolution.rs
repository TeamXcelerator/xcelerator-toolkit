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
    let root_dir = xc_core::test_support::TestDir::new("confirmed-distance");
    let root = root_dir.to_path_buf();
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

#[test]
fn derived_distance_captures_reuse_adopted_canonical_dependencies() {
    use xc_cache::*;
    use xc_spectral::distance::hp::capture_ccm_distance_with_derived_via_cache;
    let root = xc_core::test_support::TestDir::new("distance-published-parents");
    let spec = root.join("target.json");
    std::fs::write(&spec, serde_json::to_vec(&serde_json::json!({
        "schema_version":1, "profile_id":"distance-published-parents-v1",
        "base_series":{"term_input_power":0,"polynomial_coefficients":["1"],"minimum_terms":1,"maximum_terms":1000},
        "auxiliary_series":{"term_input_power":0,"polynomial_coefficients":["0","1"],"parameter_polynomial_coefficients":["1"],"minimum_terms":1,"maximum_terms":1000}
    })).unwrap()).unwrap();
    std::env::set_var("XC_TARGET_SPEC_FILE", &spec);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let author = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "local",
            root.join("author"),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let sink = CanonicalStagingProductionSink::new(
        root.join("staging"),
        TransportPolicy::default(),
        xc_core::ResourcePolicy::default(),
        xc_core::CancellationToken::new(),
    )
    .unwrap();
    let context = ArtifactCacheContext {
        resolver: Some(&author),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: Some(&sink),
    };
    let params = xc_spectral::ccm::CcmParams::from_lambda_sq_integer(5, 2);
    let mut cfg = xc_spectral::ccm::hp::HighPrecConfig::for_decimal_digits(40);
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let alpha = Float::with_val(cfg.precision_bits, 0.5);
    let rules = [
        WeightedIntegrationRule::UniformGrid {
            scheme: UniformGridScheme::Trapezoid,
            variable: GridVariable::U,
            steps: 32,
        },
        WeightedIntegrationRule::GaussLegendre {
            points: 16,
            variable: GridVariable::U,
        },
    ];
    let cold = capture_ccm_distance_with_derived_via_cache(
        &params, &cfg, &alpha, &rules, 16, &context, true, true, true,
    )
    .unwrap();
    let store = FilesystemCacheStore::new(
        "adopted",
        root.join("adopted"),
        true,
        CacheVisibility::Local,
    );
    let drafts = sink.drafts().unwrap();
    let child_kinds = [
        "ccm_distance_resolution_evidence",
        "ccm_target_residual_analysis",
        "ccm_deviation_decomposition",
    ];
    for kind in child_kinds {
        assert!(drafts.iter().any(|d| d.source_artifact_key.kind == kind));
    }
    for draft in drafts {
        let retained = author
            .resolve_exact(
                &draft.source_artifact_key,
                &draft.source_content_digest,
                CacheQuality::Validated,
                &policy,
            )
            .unwrap();
        let mut tags = retained.manifest.tags.clone();
        tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.into(),
            serde_json::to_string(&draft.manifest.semantic_key).unwrap(),
        );
        tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.into(),
            serde_json::to_string(&draft.manifest).unwrap(),
        );
        // This is the representation emitted by the GitHub shard adapter and
        // preserved when its verified payload is adopted into a local store.
        store
            .put(
                &ArtifactDraft {
                    schema_version: 1,
                    key: retained.manifest.key,
                    producer_toolkit_version: retained.manifest.producer_toolkit_version,
                    minimum_reader_version: retained.manifest.minimum_reader_version,
                    maximum_reader_version: retained.manifest.maximum_reader_version,
                    quality: retained.manifest.quality,
                    visibility: CacheVisibility::Local,
                    immutable: true,
                    dependencies: vec![],
                    tags,
                    provenance_digest: Some(draft.manifest.digest().unwrap()),
                },
                &retained.payload,
            )
            .unwrap();
    }
    let adopted = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(store),
    }]);
    let context = ArtifactCacheContext {
        resolver: Some(&adopted),
        ordered_overlays: vec!["adopted".into()],
        mode: ArtifactExecutionCacheMode::RequireReuse,
        write_on_miss: false,
        production_sink: None,
        ..context
    };
    // Exercise each public producer independently. RequireReuse forbids a
    // silent numerical fallback masking failure to accept the retained child.
    for (resolution, residual, decomposition) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let warm = capture_ccm_distance_with_derived_via_cache(
            &params,
            &cfg,
            &alpha,
            &rules,
            16,
            &context,
            resolution,
            residual,
            decomposition,
        )
        .unwrap();
        assert_eq!(warm.eigenvalue, cold.eigenvalue);
        for (a, b) in warm.distances.iter().zip(&cold.distances) {
            assert_eq!(a.value, b.value);
        }
        if resolution {
            assert_eq!(warm.resolution_tolerance_met, cold.resolution_tolerance_met);
            assert_eq!(
                warm.resolution_ladder_tolerance_met,
                cold.resolution_ladder_tolerance_met
            );
        }
    }
    std::env::remove_var("XC_TARGET_SPEC_FILE");
}
