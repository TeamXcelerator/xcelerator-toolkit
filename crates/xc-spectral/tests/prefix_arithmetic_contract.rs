#![cfg(feature = "hp")]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::{
    ArtifactKey, ArtifactManifest, CacheObjectRef, CacheQuality, CacheVisibility, ContentDigest,
    ToolkitVersion,
};
use xc_spectral::ccm::prefix::*;
fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(kind, "synthetic-prefix-fixture", kind.as_bytes()).unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    (manifest, bytes)
}
fn matrix() -> (ArtifactManifest, Vec<u8>) {
    source(
        "ccm_even_sector_matrix",
        json!({"schema_version":1,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"dimension":2,"entries":["2","0","0","3"]}),
    )
}
fn options() -> PrefixAnalysisOptions {
    PrefixAnalysisOptions {
        diagnostics: PrefixDiagnosticPolicy::default(),
        working_precision_bits: 256,
        pivot_margin_bits: 32,
        checkpoint_dimensions: vec![1, 2],
        export_significant_digits: vec![80, 96],
        export_relative_tolerance: "1e-60".into(),
    }
}
fn read_matrix(m: &ArtifactManifest, b: &[u8]) -> RetainedEvenMatrix {
    RetainedEvenMatrix::from_payload(m, b, std::slice::from_ref(&m.content_digest)).unwrap()
}

#[test]
fn retained_prefix_export_rejects_scale_hidden_residuals_and_normalizes_extreme_states() {
    for p in [128, 256, 512] {
        let epsilon = Float::with_val(p, 1) >> 100i32;
        let tolerance = Float::with_val(p, 1) >> 120i32;
        for shift in [0i32, -536_870_880, 536_870_880] {
            let scale = Float::with_val(p, 1) << shift;
            let (m, bytes) = source(
                "ccm_even_sector_matrix",
                json!({"schema_version":1,"lambda_squared":"13","n_modes":1,"precision_bits":p,"dimension":2,"entries":[scale.to_string(),"0","0",Float::with_val(p,&scale*2).to_string()]}),
            );
            let matrix = read_matrix(&m, &bytes);
            let (e, bytes) = source(
                "ccm_weil_eigenpair",
                json!({"schema_version":2,"lambda_squared":"13","n_modes":1,"precision_bits":p,"eigenvalue":scale.to_string(),"eigenvector":[epsilon.to_string(),"1",epsilon.to_string()]}),
            );
            let pair = RetainedEvenEigenpair::from_payload(
                &e,
                &bytes,
                std::slice::from_ref(&e.content_digest),
            )
            .unwrap();
            let settings = PrefixAnalysisOptions {
                working_precision_bits: p,
                pivot_margin_bits: 32,
                checkpoint_dimensions: vec![2],
                export_significant_digits: vec![200, 220],
                export_relative_tolerance: tolerance.to_string(),
                diagnostics: PrefixDiagnosticPolicy::default(),
            };
            let report = analyze_retained_prefixes(&matrix, &settings, &[pair]).unwrap();
            assert_eq!(
                report.checkpoints[0].status, "export_checks_unresolved",
                "p={p} scale={shift}"
            );
            assert!(report.checkpoints[0]
                .decoded_eigenpair_backward_error
                .is_none());
        }
    }
    let (m, bytes) = matrix();
    let matrix = read_matrix(&m, &bytes);
    for exponent in [-700_000_000i32, 0, 700_000_000] {
        let v = Float::with_val(256, -1) << exponent;
        let (e, bytes) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":2,"lambda_squared":"13","n_modes":1,"precision_bits":256,"eigenvalue":"2","eigenvector":["0",v.to_string(),"0"]}),
        );
        let pair = RetainedEvenEigenpair::from_payload(
            &e,
            &bytes,
            std::slice::from_ref(&e.content_digest),
        )
        .unwrap();
        let report = analyze_retained_prefixes(&matrix, &options(), &[pair]).unwrap();
        assert_eq!(report.checkpoints[1].status, "export_checks_passed");
        assert_eq!(
            Float::with_val(
                256,
                Float::parse(
                    report.checkpoints[1]
                        .decoded_eigenpair_backward_error
                        .as_ref()
                        .unwrap()
                )
                .unwrap()
            ),
            0
        );
    }
}

#[test]
fn prefix_cache_replays_every_numerical_row_and_export_field() {
    use xc_cache::*;
    let root_dir = xc_core::test_support::TestDir::new("prefix-numerical-replay");
    let root = root_dir.join(format!("xc-prefix-numerical-replay-{}", std::process::id()));
    assert!(!root.exists());
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "test",
            root.join("producer"),
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
    let mut context = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["test".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let (m, bytes) = matrix();
    let source = read_matrix(&m, &bytes);
    let settings = options();
    let fresh = analyze_retained_prefixes_via_cache(&source, &settings, &[], &context).unwrap();
    let manifest = fresh.produced_manifest.unwrap();
    let semantic: SemanticKeyEnvelope =
        serde_json::from_str(&manifest.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
    assert_eq!(semantic.mathematical_semantics_version, PREFIX_SEMANTICS);
    assert_eq!(
        manifest.minimum_reader_version,
        ToolkitVersion::parse(xc_cache::CLEAN_SLATE).unwrap()
    );
    context.mode = ArtifactExecutionCacheMode::RequireReuse;
    context.write_on_miss = false;
    let warm = analyze_retained_prefixes_via_cache(&source, &settings, &[], &context).unwrap();
    assert_eq!(warm.value, fresh.value);
    let payload = serde_json::to_value(&fresh.value).unwrap();
    let mutations = [
        ("/ladder/rows/0/sigma", json!("999")),
        ("/ladder/rows/0/innovation_mass", json!("999")),
        ("/ladder/rows/0/inverse_trace_increment", json!("999")),
        ("/ladder/rows/0/inverse_trace", json!("999")),
        ("/ladder/rows/0/inverse_square_trace", json!("999")),
        ("/ladder/rows/0/pivot_cancellation_scale", json!("999")),
        ("/ladder/rows/0/effective_inverse_rank", json!("999")),
        ("/ladder/rows/0/newest_inverse_trace_fraction", json!("999")),
        (
            "/ladder/rows/0/smallest_eigenvalue_lower_estimate",
            json!("999"),
        ),
        (
            "/ladder/rows/0/smallest_eigenvalue_upper_estimate",
            json!("999"),
        ),
        (
            "/ladder/rows/0/eigenvalue_depth_lower_estimate",
            json!("999"),
        ),
        (
            "/ladder/rows/0/eigenvalue_depth_upper_estimate",
            json!("999"),
        ),
        ("/checkpoints/0/status", json!("export_checks_passed")),
        ("/checkpoints/0/raw_innovation/0", json!("123")),
        ("/checkpoints/0/unit_innovation/0", json!("123")),
        (
            "/checkpoints/0/decoded_innovation_backward_error",
            json!("-777"),
        ),
        ("/checkpoints/0/accepted_significant_digits", json!(1)),
    ];
    for (index, (pointer, value)) in mutations.into_iter().enumerate() {
        let mut bad = payload.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        let store = FilesystemCacheStore::new(
            "test",
            root.join(format!("mutation-{index}")),
            true,
            CacheVisibility::Local,
        );
        store
            .put(
                &ArtifactDraft {
                    schema_version: manifest.schema_version,
                    key: manifest.key.clone(),
                    producer_toolkit_version: manifest.producer_toolkit_version.clone(),
                    minimum_reader_version: manifest.minimum_reader_version.clone(),
                    maximum_reader_version: manifest.maximum_reader_version.clone(),
                    quality: manifest.quality,
                    visibility: manifest.visibility,
                    immutable: manifest.immutable,
                    dependencies: manifest.dependencies.clone(),
                    tags: manifest.tags.clone(),
                    provenance_digest: manifest.provenance_digest.clone(),
                },
                &serde_json::to_vec(&bad).unwrap(),
            )
            .unwrap();
        let altered = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(store),
        }]);
        let altered_context = ArtifactCacheContext {
            resolver: Some(&altered),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["test".into()],
            mode: ArtifactExecutionCacheMode::RequireReuse,
            write_on_miss: false,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let result = analyze_retained_prefixes_via_cache(&source, &settings, &[], &altered_context);
        assert!(result.is_err(), "accepted changed field {pointer}");
    }
    assert!(root.starts_with(std::env::temp_dir()));
    assert!(root
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("xc-prefix-numerical-replay-"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn retained_reduction_cache_replays_spectrum_norms_and_residuals() {
    use xc_cache::*;
    let root_dir = xc_core::test_support::TestDir::new("reduction-numerical-repl");
    let root = root_dir.join(format!(
        "xc-reduction-numerical-replay-{}",
        std::process::id()
    ));
    assert!(!root.exists());
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "test",
            root.join("producer"),
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
    let mut context = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["test".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let (m, bytes) = matrix();
    let source = read_matrix(&m, &bytes);
    let fresh = check_retained_reduction_via_cache(&source, 256, 2, "1e-60", &context).unwrap();
    let manifest = fresh.produced_manifest.unwrap();
    let semantic: SemanticKeyEnvelope =
        serde_json::from_str(&manifest.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
    assert_eq!(
        semantic.mathematical_semantics_version,
        RETAINED_REDUCTION_SEMANTICS
    );
    assert_eq!(
        manifest.minimum_reader_version,
        ToolkitVersion::parse(xc_cache::CLEAN_SLATE).unwrap()
    );
    context.mode = ArtifactExecutionCacheMode::RequireReuse;
    context.write_on_miss = false;
    let warm = check_retained_reduction_via_cache(&source, 256, 2, "1e-60", &context).unwrap();
    assert_eq!(warm.value, fresh.value);
    let payload = serde_json::to_value(&fresh.value).unwrap();
    assert_eq!(
        fresh
            .value
            .computed_eigenvalues
            .iter()
            .map(|x| Float::with_val(256, Float::parse(x).unwrap()))
            .collect::<Vec<_>>(),
        vec![Float::with_val(256, 2), Float::with_val(256, 3)]
    );
    assert!(fresh.value.checks_passed);
    let mutations = [
        ("/diagonal/0", json!("999")),
        ("/off_diagonal/0", json!("222")),
        ("/computed_eigenvalues/0", json!("-999")),
        ("/computed_eigenvalues/1", json!("999")),
        ("/diagnostics/absolute_similarity_residual", json!("1")),
        ("/diagnostics/relative_similarity_residual", json!("1e-62")),
        ("/diagnostics/absolute_orthogonality_residual", json!("1")),
        (
            "/diagnostics/relative_orthogonality_residual",
            json!("1e-62"),
        ),
        ("/diagnostics/source_frobenius_norm", json!("0")),
        ("/diagnostics/tridiagonal_frobenius_norm", json!("0")),
        ("/diagnostics/basis_frobenius_norm", json!("0")),
    ];
    for (index, (pointer, value)) in mutations.into_iter().enumerate() {
        let mut bad = payload.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        let store = FilesystemCacheStore::new(
            "test",
            root.join(format!("mutation-{index}")),
            true,
            CacheVisibility::Local,
        );
        store
            .put(
                &ArtifactDraft {
                    schema_version: manifest.schema_version,
                    key: manifest.key.clone(),
                    producer_toolkit_version: manifest.producer_toolkit_version.clone(),
                    minimum_reader_version: manifest.minimum_reader_version.clone(),
                    maximum_reader_version: manifest.maximum_reader_version.clone(),
                    quality: manifest.quality,
                    visibility: manifest.visibility,
                    immutable: manifest.immutable,
                    dependencies: manifest.dependencies.clone(),
                    tags: manifest.tags.clone(),
                    provenance_digest: manifest.provenance_digest.clone(),
                },
                &serde_json::to_vec(&bad).unwrap(),
            )
            .unwrap();
        let altered = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(store),
        }]);
        let altered_context = ArtifactCacheContext {
            resolver: Some(&altered),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["test".into()],
            mode: ArtifactExecutionCacheMode::RequireReuse,
            write_on_miss: false,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        let result = check_retained_reduction_via_cache(&source, 256, 2, "1e-60", &altered_context);
        assert!(result.is_err(), "accepted changed field {pointer}");
    }
    assert!(root.starts_with(std::env::temp_dir()));
    assert!(root
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("xc-reduction-numerical-replay-"));
    std::fs::remove_dir_all(root).unwrap();
}
