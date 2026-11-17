use super::*;

#[test]
fn root_window_reuse_ignores_only_obsolete_dataset_attribution() {
    let scratch = xc_core::test_support::TestDir::new("root-window-attribution");
    let policy = policy();
    let original = resolver(store_at(&scratch.join("original")));
    let (state, state_manifest) = state(&["0", "1", "0"], "3");
    let (mut secular, secular_bytes) = source(
        "ccm_secular_source",
        json!({
            "schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "force_even":true,"normalization":"sum_xi_equals_sqrt_log_lambda_squared",
            "eigenpair_content_digest":state_manifest.content_digest
        }),
    );
    let dependency = |m: &ArtifactManifest| DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    };
    secular.dependencies.push(dependency(&state_manifest));
    let (mut root_manifest, root_bytes) = source(
        "ccm_root_refinement",
        json!({
            "schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "force_even":true,"first_root_index":1,"discovery_mode":"reference_seeded_audit",
            "reference_seeds_used":true,"starting_points":["14"],"completeness":"partial",
            "outcomes":[{"status":"converged","details":{"value":"14"}}]
        }),
    );
    root_manifest.dependencies.push(dependency(&secular));
    let allowed = vec![
        root_manifest.content_digest.clone(),
        secular.content_digest.clone(),
        state_manifest.content_digest.clone(),
    ];
    let roots = RetainedRoots::from_payload(
        &root_manifest,
        &root_bytes,
        &secular,
        &secular_bytes,
        &state,
        &allowed,
    )
    .unwrap();
    let cold = capture_root_window(&roots, &cached(&original, &policy)).unwrap();
    let manifest = cold.produced_manifest.unwrap();
    for mutation in 0..5 {
        let mut value = serde_json::to_value(&cold.value).unwrap();
        value["data"]["source_acquisition"]["reference_dataset"] =
            json!({"label":"historical attribution"});
        match mutation {
            1 => value["data"]["source_acquisition"]["reference_seeds_used"] = json!(false),
            2 => value["data"]["points"][0]["value"] = json!("15"),
            3 => value["data"]["source_acquisition"]["unreviewed"] = json!(true),
            4 => value["data"]["missing_count"] = json!(1),
            _ => {}
        }
        let variant = store_at(&scratch.join(format!("variant-{mutation}")));
        put_variant(&variant, &manifest, &serde_json::to_vec(&value).unwrap());
        let variant = resolver(variant);
        let mut cache = cached(&variant, &policy);
        cache.mode = ArtifactExecutionCacheMode::RequireReuse;
        cache.write_on_miss = false;
        let result = capture_root_window(&roots, &cache);
        assert_eq!(
            result.is_ok(),
            mutation == 0,
            "mutation {mutation}: {:?}",
            result.err()
        );
    }
}

fn policy() -> CachePolicy {
    CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    }
}

fn cached<'a>(resolver: &'a CacheResolver, policy: &'a CachePolicy) -> ArtifactCacheContext<'a> {
    ArtifactCacheContext {
        resolver: Some(resolver),
        acceptance: Some(policy),
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        ordered_overlays: vec!["fixture".into()],
        ..context()
    }
}

fn store_at(path: &std::path::Path) -> FilesystemCacheStore {
    FilesystemCacheStore::new("fixture", path, true, CacheVisibility::Local)
}

fn resolver(store: FilesystemCacheStore) -> CacheResolver {
    CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(store),
    }])
}

fn put_variant(
    store: &FilesystemCacheStore,
    manifest: &ArtifactManifest,
    bytes: &[u8],
) -> ArtifactManifest {
    store
        .put(
            &ArtifactDraft {
                schema_version: manifest.schema_version,
                key: manifest.key.clone(),
                producer_toolkit_version: manifest.producer_toolkit_version.clone(),
                minimum_reader_version: manifest.minimum_reader_version.clone(),
                maximum_reader_version: None,
                quality: manifest.quality,
                visibility: manifest.visibility,
                immutable: true,
                dependencies: manifest.dependencies.clone(),
                tags: manifest.tags.clone(),
                provenance_digest: None,
            },
            bytes,
        )
        .unwrap()
}

#[test]
fn configured_projection_authenticates_original_reference_json_bytes() {
    let scratch = xc_core::test_support::TestDir::new("reference-json-reordering");
    let policy = policy();
    let original = resolver(store_at(&scratch.join("original")));
    let spec = reference(&["0", "1", "0"]);
    let captured = capture_reference(&spec, &cached(&original, &policy)).unwrap();
    let manifest = captured.produced_manifest.unwrap();
    // Sorted JSON object keys plus whitespace preserve the exact values but
    // cannot reproduce the original typed encoder's byte digest.
    let value = serde_json::to_value(&captured.value).unwrap();
    let bytes = serde_json::to_vec_pretty(&value).unwrap();
    assert_ne!(ContentDigest::sha256(&bytes), manifest.content_digest);
    let migrated = store_at(&scratch.join("migrated"));
    let admitted = put_variant(&migrated, &manifest, &bytes);
    let migrated = resolver(migrated);
    let mut cache = cached(&migrated, &policy);
    let reused = capture_reference(&spec, &cache).unwrap();
    assert_eq!(
        reused.reused_manifest.unwrap().content_digest,
        admitted.content_digest
    );
    let (state, _) = state(&["0", "1", "0"], "1");
    let input = ResearchInputs {
        schema_version: 1,
        reference: spec.clone(),
        basis: vec![],
        projection: ProjectionOptions {
            working_precision_bits: 192,
            normalization: "center_one".into(),
            fixed_second_component: None,
        },
    };
    assert!(capture_configured_projection(&state, &input, &cache).is_ok());
    cache.mode = ArtifactExecutionCacheMode::RequireReuse;
    cache.write_on_miss = false;
    assert!(capture_configured_projection(&state, &input, &cache).is_ok());

    let mut wrong = value;
    wrong["data"]["spec"]["coefficients"][1] = json!("2");
    let corrupt = store_at(&scratch.join("wrong-values"));
    put_variant(&corrupt, &manifest, &serde_json::to_vec(&wrong).unwrap());
    let corrupt = resolver(corrupt);
    let mut required = cached(&corrupt, &policy);
    required.mode = ArtifactExecutionCacheMode::RequireReuse;
    required.write_on_miss = false;
    assert!(capture_configured_projection(&state, &input, &required).is_err());
    let repaired = capture_reference(&spec, &cached(&corrupt, &policy)).unwrap();
    assert_eq!(
        serde_json::to_value(&repaired.value.data).unwrap(),
        serde_json::to_value(&captured.value.data).unwrap()
    );
    assert!(repaired.reused_manifest.is_none());
}

#[test]
fn transform_reuse_accepts_only_exactly_equal_decimal_coordinates() {
    let scratch = xc_core::test_support::TestDir::new("transform-decimal-reuse");
    let policy = policy();
    let original = resolver(store_at(&scratch.join("original")));
    let (state, _) = state(&["0", "1", "0"], "1");
    let dataset = DatasetSpec {
        schema_version: 1,
        role: "evaluation_points".into(),
        attribution: "synthetic exact inputs".into(),
        coordinate: "mellin_t".into(),
        precision_bits: 192,
        points: vec![EvaluationPoint {
            ordinal: 1,
            value: Some("1.25".into()),
            source_status: "supplied".into(),
        }],
    };
    let record = capture_dataset(&dataset, &context()).unwrap().value;
    let (manifest, bytes) = source(
        "research_reference_dataset",
        serde_json::to_value(record).unwrap(),
    );
    let retained = RetainedDataset::from_payload(
        &manifest,
        &bytes,
        std::slice::from_ref(&manifest.content_digest),
    )
    .unwrap();
    let options = TransformOptions {
        working_precision_bits: 192,
        maximum_rows: 1,
        maximum_estimated_output_bytes: 1 << 20,
    };
    let cold =
        capture_transforms_at_dataset(&state, &retained, &options, &cached(&original, &policy))
            .unwrap();
    let manifest = cold.produced_manifest.unwrap();
    for (index, coordinate, accepted) in [
        (0, "1.25000", true),
        (1, "125e-2", true),
        (
            2,
            "1.25000000000000000000000000000000000000000000000000000000000000001",
            false,
        ),
        (3, "NaN", false),
    ] {
        let mut value = serde_json::to_value(&cold.value).unwrap();
        value["data"]["rows"][0]["t"] = json!(coordinate);
        let variant = store_at(&scratch.join(format!("variant-{index}")));
        put_variant(&variant, &manifest, &serde_json::to_vec(&value).unwrap());
        let variant = resolver(variant);
        let mut cache = cached(&variant, &policy);
        cache.mode = ArtifactExecutionCacheMode::RequireReuse;
        cache.write_on_miss = false;
        let result = capture_transforms_at_dataset(&state, &retained, &options, &cache);
        assert_eq!(result.is_ok(), accepted, "{coordinate}: {:?}", result.err());
    }
}
