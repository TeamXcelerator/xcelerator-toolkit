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
        producer_toolkit_version: ToolkitVersion::parse("0.14.3").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
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
fn nesting_checks_exact_points_and_retains_both_source_identities() {
    let (small_m, small_b) = matrix();
    let small = read_matrix(&small_m, &small_b);
    let build = |center: &str| {
        let (m, b) = source(
            "ccm_even_sector_matrix",
            json!({"schema_version":1,"lambda_squared":"13","n_modes":2,"dimension":3,"precision_bits":256,"entries":["2","0","0","0",center,"0","0","0","4"]}),
        );
        read_matrix(&m, &b)
    };
    let larger = build("3");
    let result = check_prefix_nesting(&small, &larger).unwrap();
    assert!(result.exactly_nested);
    assert_eq!(result.entries_compared, 4);
    assert_eq!(result.smaller_source, small_m.content_digest);
    assert_eq!(result.larger_source, larger.manifest().content_digest);
    let changed = check_prefix_nesting(&small, &build("3.00000000000000000000001")).unwrap();
    assert!(!changed.exactly_nested);
    assert_eq!(changed.mismatching_entries, 1);
    assert_eq!(changed.first_mismatch, Some([1, 1]));
    assert!(check_prefix_nesting(&larger, &small).is_err());
}
#[test]
fn retained_source_is_bound_to_bytes_and_allowlist() {
    let (m, b) = matrix();
    assert!(RetainedEvenMatrix::from_payload(&m, &b, &[]).is_err());
    let mut corrupt = b.clone();
    corrupt[0] = b'[';
    assert!(RetainedEvenMatrix::from_payload(
        &m,
        &corrupt,
        std::slice::from_ref(&m.content_digest)
    )
    .is_err());
    let a = read_matrix(&m, &b);
    let before = a.entries().to_vec();
    let r = analyze_retained_prefixes(&a, &options(), &[]).unwrap();
    assert_eq!(a.entries(), before);
    assert_eq!(r.parent_matrix_source, m.content_digest);
    assert!(r.prefixes_are_parent_derived);
    assert!(r.ladder.stopped.is_none());
    assert_eq!(
        r.checkpoints[1].status,
        "innovation_export_passed_eigenpair_not_supplied"
    );
}
#[test]
fn eigenstate_is_normalized_on_a_copy_with_source_bound_overlap() {
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let (e, eb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"2","eigenvector":["0","-5","0"]}),
    );
    let pair =
        RetainedEvenEigenpair::from_payload(&e, &eb, std::slice::from_ref(&e.content_digest))
            .unwrap();
    let r = analyze_retained_prefixes(&a, &options(), &[pair]).unwrap();
    let packet = &r.checkpoints[1];
    assert_eq!(packet.status, "export_checks_passed");
    assert_eq!(packet.eigenpair_source, Some(e.content_digest));
    assert_eq!(
        Float::with_val(
            256,
            Float::parse(&packet.unit_retained_eigenvector[0]).unwrap()
        ),
        1
    );
    assert_eq!(
        Float::with_val(
            256,
            Float::parse(packet.squared_overlap.as_ref().unwrap()).unwrap()
        ),
        0
    );
    assert_eq!(
        eb,
        serde_json::to_vec(
            &json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"2","eigenvector":["0","-5","0"]})
        )
        .unwrap()
    );
}
#[test]
fn asymmetric_and_odd_sources_are_rejected() {
    let (m, b) = source(
        "ccm_even_sector_matrix",
        json!({"schema_version":1,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"dimension":2,"entries":["2","0","1","3"]}),
    );
    assert!(
        RetainedEvenMatrix::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).is_err()
    );
    let (e, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"2","eigenvector":["1","0","-1"]}),
    );
    assert!(
        RetainedEvenEigenpair::from_payload(&e, &b, std::slice::from_ref(&e.content_digest))
            .is_err()
    );
}
#[test]
fn incorrect_eigenstate_does_not_replace_sources() {
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let (e, eb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":2,"lambda_squared":"13","n_modes":1,
        "precision_bits":256,"eigenvalue":"999","eigenvector":["0","1","0"]}),
    );
    let pair =
        RetainedEvenEigenpair::from_payload(&e, &eb, std::slice::from_ref(&e.content_digest))
            .unwrap();
    let r = analyze_retained_prefixes(&a, &options(), &[pair]).unwrap();
    assert_eq!(r.checkpoints[1].status, "export_checks_unresolved");
    assert!(r.checkpoints[1].unit_retained_eigenvector.is_empty());
    assert_eq!(a.manifest(), &m);
}
#[test]
fn ten_complete_packets_are_byte_identical() {
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let run =
        || serde_json::to_vec(&analyze_retained_prefixes(&a, &options(), &[]).unwrap()).unwrap();
    let reference = run();
    for _ in 1..10 {
        assert_eq!(reference, run());
    }
}
#[test]
fn new_kind_is_private_only_without_raising_ordinary_compatibility_floors() {
    assert_eq!(
        xc_cache::artifact_compatibility_policy("ccm-evidence", PREFIX_ARTIFACT_KIND)
            .unwrap()
            .minimum_producer_version,
        ToolkitVersion::parse("0.15.0").unwrap()
    );
    assert_eq!(
        xc_cache::artifact_compatibility_policy("ccm-matrices", "ccm_tau_matrix")
            .unwrap()
            .minimum_producer_version,
        ToolkitVersion::parse("0.13.0").unwrap()
    );
}
#[test]
fn derived_cache_reuses_diagnostics_and_rejects_missing_requested_variant() {
    use xc_cache::{
        ArtifactCacheContext, ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver,
        FilesystemCacheStore,
    };
    let root = std::env::temp_dir().join(format!("ccm-prefix-child-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "prefix-test",
            root.clone(),
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
    let context = |mode, write_on_miss| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["prefix-test".into()],
        mode,
        write_on_miss,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let first = analyze_retained_prefixes_via_cache(
        &a,
        &options(),
        &[],
        &context(ArtifactExecutionCacheMode::PreferReuse, true),
    )
    .unwrap();
    assert!(first.produced_manifest.is_some());
    let mut equivalent = options();
    equivalent.export_relative_tolerance = "0.1E-59".into();
    let reused = analyze_retained_prefixes_via_cache(
        &a,
        &equivalent,
        &[],
        &context(ArtifactExecutionCacheMode::RequireReuse, false),
    )
    .unwrap();
    assert!(reused.reused_manifest.is_some());
    assert_eq!(first.value, reused.value);
    assert_eq!(
        first.produced_manifest.unwrap(),
        reused.reused_manifest.unwrap()
    );
    let mut other = options();
    other.export_significant_digits = vec![100];
    assert!(analyze_retained_prefixes_via_cache(
        &a,
        &other,
        &[],
        &context(ArtifactExecutionCacheMode::RequireReuse, false)
    )
    .is_err());
    for policy in [
        PrefixDiagnosticPolicy::full(),
        PrefixDiagnosticPolicy {
            third_inverse_moment: true,
            innovation_cancellation: false,
        },
        PrefixDiagnosticPolicy {
            third_inverse_moment: false,
            innovation_cancellation: false,
        },
    ] {
        let mut extended = options();
        extended.diagnostics = policy;
        assert!(analyze_retained_prefixes_via_cache(
            &a,
            &extended,
            &[],
            &context(ArtifactExecutionCacheMode::RequireReuse, false)
        )
        .is_err());
        let cold = analyze_retained_prefixes_via_cache(
            &a,
            &extended,
            &[],
            &context(ArtifactExecutionCacheMode::PreferReuse, true),
        )
        .unwrap();
        let warm = analyze_retained_prefixes_via_cache(
            &a,
            &extended,
            &[],
            &context(ArtifactExecutionCacheMode::RequireReuse, false),
        )
        .unwrap();
        assert_eq!(cold.value, warm.value);
        assert_eq!(cold.value.semantics, EXTENDED_PREFIX_SEMANTICS);
        assert!(cold
            .value
            .ladder
            .rows
            .iter()
            .all(
                |r| r.third_inverse_moment.is_some() == policy.third_inverse_moment
                    && r.innovation_cancellation.is_some() == policy.innovation_cancellation
            ));
    }
    let reduction = check_retained_reduction_via_cache(
        &a,
        256,
        2,
        "1e-40",
        &context(ArtifactExecutionCacheMode::PreferReuse, true),
    )
    .unwrap();
    assert!(reduction.value.checks_passed);
    let reused_reduction = check_retained_reduction_via_cache(
        &a,
        256,
        2,
        "0.1E-39",
        &context(ArtifactExecutionCacheMode::RequireReuse, false),
    )
    .unwrap();
    assert_eq!(reduction.value, reused_reduction.value);
    assert_eq!(
        reduction.produced_manifest,
        reused_reduction.reused_manifest
    );
    assert!(check_retained_reduction_via_cache(
        &a,
        256,
        1,
        "1e-40",
        &context(ArtifactExecutionCacheMode::RequireReuse, false)
    )
    .is_err());
    assert!(check_retained_reduction_via_cache(
        &a,
        512,
        2,
        "1e-40",
        &context(ArtifactExecutionCacheMode::RequireReuse, false)
    )
    .is_err());
    let mut public = context(ArtifactExecutionCacheMode::PreferReuse, true);
    public.write_visibility = CacheVisibility::Public;
    assert!(analyze_retained_prefixes_via_cache(&a, &options(), &[], &public).is_err());
    assert!(check_retained_reduction_via_cache(&a, 256, 2, "1e-40", &public).is_err());
    assert_eq!(a.manifest(), &m);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn prefix_observable_adapter_keeps_parent_and_estimate_semantics() {
    use xc_core::*;
    let (m, b) = matrix();
    let retained = read_matrix(&m, &b);
    let report = analyze_retained_prefixes(&retained, &options(), &[]).unwrap();
    let before = serde_json::to_vec(&report).unwrap();
    let resolution = ObservableResolution {
        source_precision_bits: 256,
        analysis_precision_bits: 256,
        export_significant_digits: 80,
        components: RESOLUTION_AXES
            .iter()
            .map(|&axis| ResolutionComponent {
                axis,
                classification: ResolutionClass::Unknown,
                absolute: None,
                explanation: "not evaluated for this supplied report".into(),
                evidence: vec![],
                dependency_groups: Default::default(),
            })
            .collect(),
    };
    let values = prefix_observations(
        &report,
        PrefixObservable::EigenvalueDepthLowerEstimate,
        &resolution,
    )
    .unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values[1].design.n_modes, Some(1));
    assert_eq!(values[1].design.dimension, 2);
    assert_eq!(values[1].design.coordinates["parent_N"].as_str(), "1");
    assert!(values[1]
        .observable
        .definition_id
        .ends_with("lower-estimate"));
    assert!(values[1].observable.root.is_none());
    assert_eq!(values[1].resolution, resolution);
    for observable in [
        PrefixObservable::GapRatioEstimate,
        PrefixObservable::SmallestEigenvalueSecondOrderEstimate,
        PrefixObservable::PivotCancellationDigits,
        PrefixObservable::InnovationCancellationDigits,
    ] {
        let derived = prefix_observations(&report, observable, &resolution).unwrap();
        assert_eq!(
            derived.len(),
            if observable == PrefixObservable::GapRatioEstimate {
                1
            } else {
                2
            }
        );
        assert!(derived
            .iter()
            .all(|v| v.resolution == resolution && v.observable.root.is_none()));
        if matches!(
            observable,
            PrefixObservable::GapRatioEstimate
                | PrefixObservable::SmallestEigenvalueSecondOrderEstimate
        ) {
            assert!(derived[0]
                .observable
                .normalization
                .contains("dominant second inverse mode"));
        }
    }
    let mut full = options();
    full.diagnostics = PrefixDiagnosticPolicy::full();
    let extended = analyze_retained_prefixes(&retained, &full, &[]).unwrap();
    for observable in [
        PrefixObservable::InverseCubeTrace,
        PrefixObservable::SmallestEigenvalueCubeLowerEstimate,
        PrefixObservable::SmallestEigenvalueCubeRatioUpperEstimate,
        PrefixObservable::TwoModeSmallestEigenvalueEstimate,
        PrefixObservable::TwoModeSecondEigenvalueEstimate,
        PrefixObservable::TwoModeGapRatioEstimate,
        PrefixObservable::TwoModeTraceClosureResidual,
    ] {
        let observations = prefix_observations(&extended, observable, &resolution).unwrap();
        let has_second = matches!(
            observable,
            PrefixObservable::TwoModeSecondEigenvalueEstimate
                | PrefixObservable::TwoModeGapRatioEstimate
        );
        assert_eq!(observations.len(), if has_second { 1 } else { 2 });
        assert!(observations.iter().all(|v| v.resolution == resolution));
    }
    assert!(prefix_observations(&report, PrefixObservable::InverseCubeTrace, &resolution).is_err());
    let mut old = serde_json::to_value(&report).unwrap();
    old["semantics"] = LEGACY_PREFIX_SEMANTICS.into();
    for row in old["ladder"]["rows"].as_array_mut().unwrap() {
        for field in [
            "innovation_cancellation",
            "gap_ratio_estimate",
            "smallest_eigenvalue_second_order_estimate",
        ] {
            row.as_object_mut().unwrap().remove(field);
        }
    }
    let old: CcmPrefixAnalysis = serde_json::from_value(old).unwrap();
    assert!(prefix_observations(&old, PrefixObservable::InverseTrace, &resolution).is_ok());
    assert!(prefix_observations(
        &old,
        PrefixObservable::InnovationCancellationDigits,
        &resolution
    )
    .is_err());
    assert_eq!(before, serde_json::to_vec(&report).unwrap());
}

#[test]
fn retained_reduction_check_obeys_budget_and_preserves_parent() {
    let (m, b) = matrix();
    let retained = read_matrix(&m, &b);
    let before = retained.entries().to_vec();
    assert!(check_retained_reduction(&retained, 256, 1, "1e-40").is_err());
    assert!(check_retained_reduction(&retained, 128, 2, "1e-20").is_err());
    assert!(check_retained_reduction(&retained, 256, 2, "nan").is_err());
    let report = check_retained_reduction(&retained, 256, 2, "1e-40").unwrap();
    assert!(report.checks_passed);
    assert_eq!(report.matrix_source, m.content_digest);
    assert_eq!(report.source_precision_bits, 256);
    assert_eq!(report.working_precision_bits, 256);
    assert_eq!(retained.entries(), before);
    assert_eq!(
        serde_json::to_vec(&report).unwrap(),
        serde_json::to_vec(&check_retained_reduction(&retained, 256, 2, "1e-40").unwrap()).unwrap()
    );
    let encoded = serde_json::to_vec(&report).unwrap();
    let decoded: RetainedReductionCheck = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, report);
    let elevated = check_retained_reduction(&retained, 512, 2, "1e-80").unwrap();
    assert_eq!(elevated.source_precision_bits, 256);
    assert_eq!(elevated.working_precision_bits, 512);
}

fn capture_context() -> xc_cache::ArtifactCacheContext<'static> {
    xc_cache::ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: None,
        ordered_overlays: vec!["disabled".into()],
        mode: xc_cache::ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    }
}
#[test]
fn capture_runner_accounts_for_missing_sources_and_continues_primary_diagnostics() {
    use xc_spectral::ccm::capture::*;
    let plan = CcmCapturePlan::ultra(2, 2).unwrap();
    let expected = plan.receipt().unwrap();
    let mut calls = 0;
    let result = plan
        .execute_with_receipt(None, &capture_context(), |_| {
            calls += 1;
            Err(xc_cache::CaptureFailure::Missing {
                reason: "synthetic test has no primary sources".into(),
            })
        })
        .unwrap();
    result.value.receipt.validate_against(&expected).unwrap();
    assert_eq!(calls, expected.outcomes().len() - 2);
    assert!(matches!(
        result.value.receipt.outcomes()["prefix_ladder"],
        xc_core::DiagnosticOutcome::Missing { .. }
    ));
    assert!(matches!(
        result.value.receipt.outcomes()["prefix_checkpoint_2"],
        xc_core::DiagnosticOutcome::Blocked { .. }
    ));
    assert!(!result.value.receipt.is_complete());
}
#[test]
fn capture_runner_saves_prefixes_and_accounts_for_optional_reduction_budget() {
    use xc_spectral::ccm::capture::*;
    let (m, b) = matrix();
    let a = read_matrix(&m, &b);
    let plan = CcmCapturePlan::resolve(CcmCaptureLevel::Claim, 2, 2)
        .unwrap()
        .with_prefix_checkpoints(vec![2])
        .unwrap();
    let budget = RetainedReductionRequest {
        working_precision_bits: 256,
        maximum_dimension: 1,
        relative_tolerance: "1e-50".into(),
    };
    let result = plan
        .execute_with_receipt_and_reduction(
            Some((&a, &[])),
            Some(&budget),
            &capture_context(),
            |_| panic!("no primary groups requested"),
        )
        .unwrap();
    assert!(matches!(
        result.value.receipt.outcomes()["retained_reduction"],
        xc_core::DiagnosticOutcome::Blocked { .. }
    ));
    assert!(matches!(
        result.value.receipt.outcomes()["prefix_checkpoint_2"],
        xc_core::DiagnosticOutcome::Missing { .. }
    ));
    assert!(result.value.measurements.contains_key("prefix_ladder"));
    assert_eq!(result.value.source_dependencies.len(), 1);
    assert_eq!(
        result.value.source_dependencies[0].content_digest,
        m.content_digest
    );

    let budget = RetainedReductionRequest {
        maximum_dimension: 2,
        ..budget
    };
    let result = plan
        .execute_with_receipt_and_reduction(
            Some((&a, &[])),
            Some(&budget),
            &capture_context(),
            |_| panic!("no primary groups requested"),
        )
        .unwrap();
    assert_eq!(
        result.value.measurements["retained_reduction"].value["checks_passed"],
        true
    );
    result.value.validate().unwrap();
}
