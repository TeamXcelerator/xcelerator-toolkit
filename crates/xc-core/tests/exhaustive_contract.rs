use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use xc_core::*;

fn route(id: &str) -> RouteEvidence {
    RouteEvidence {
        route_id: id.into(),
        algorithm_family: format!("algorithm-{id}"),
        formulation: format!("formulation-{id}"),
        implementation_id: format!("implementation-{id}"),
        decisive_intermediates: BTreeSet::new(),
        precision_bits: Some(128),
        seed: None,
        thread_count: Some(1),
        evidence_digest: None,
    }
}
#[test]
fn incomplete_route_identity_cannot_establish_independence() {
    let declaration = IndependenceDeclaration {
        intended_claim: "same finite eigenvalue".into(),
        rationale: "independent algorithms".into(),
        accepted_shared_inputs: BTreeSet::new(),
    };
    let secondary = route("secondary");
    assert!(assess_route_independence(&route("primary"), &secondary, &declaration).independent);
    for field in 0..3 {
        let mut primary = route("primary");
        match field {
            0 => primary.algorithm_family.clear(),
            1 => primary.formulation.clear(),
            _ => primary.implementation_id.clear(),
        }
        assert!(
            !assess_route_independence(&primary, &secondary, &declaration).independent,
            "accepted missing route field {field}"
        );
    }
}
#[test]
fn resource_sum_above_u64_max_does_not_fit_a_u64_max_budget() {
    let policy = ResourcePolicy {
        maximum_memory_bytes: Some(u64::MAX),
        ..ResourcePolicy::default()
    };
    let estimate = ResourceEstimate {
        resident_memory_bytes: Some(u64::MAX),
        temporary_memory_bytes: Some(1),
        ..ResourceEstimate::default()
    };
    let report = policy.assess(estimate);
    assert!(
        !report.feasible,
        "overflowing exact sum was reported feasible"
    );
    assert!(report
        .violations
        .iter()
        .any(|v| v.resource == ResourceKind::Memory));
    let exact = ResourceEstimate {
        resident_memory_bytes: Some(u64::MAX - 1),
        temporary_memory_bytes: Some(1),
        ..ResourceEstimate::default()
    };
    assert!(policy.assess(exact).feasible);
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FlexibleConfiguration {
    shape: Value,
}
impl ValidateResolvedConfig for FlexibleConfiguration {
    fn validate_resolved(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}
#[test]
fn empty_object_cannot_bypass_environment_override_allowlist() {
    let built_in = ConfigurationLayer::from_serializable(
        ConfigSource::BuiltIn,
        "baseline",
        &FlexibleConfiguration { shape: json!(5) },
    )
    .unwrap();
    let environment =
        ConfigurationLayer::environment("environment", json!({"shape":{}}), Vec::<String>::new());
    let result = resolve_configuration::<FlexibleConfiguration>([built_in, environment]);
    assert!(
        matches!(result, Err(ConfigResolutionError::ForbiddenOverride { .. })),
        "unpermitted structural override accepted: {result:?}"
    );
}
#[test]
fn replacing_a_configuration_container_removes_obsolete_leaf_paths() {
    for (before, after) in [
        (json!({"old":1}), json!(5)),
        (json!(5), json!({"new":2})),
        (json!(5), json!({})),
    ] {
        let built_in = ConfigurationLayer::from_serializable(
            ConfigSource::Run,
            "old",
            &FlexibleConfiguration { shape: before },
        )
        .unwrap();
        let replacement = ConfigurationLayer::from_serializable(
            ConfigSource::CommandLine,
            "new",
            &FlexibleConfiguration {
                shape: after.clone(),
            },
        )
        .unwrap();
        let resolved =
            resolve_configuration::<FlexibleConfiguration>([built_in, replacement]).unwrap();
        assert_eq!(resolved.resolved.shape, after);
        assert!(!resolved.resolved_paths.contains_key("shape.old"));
    }
}

fn fingerprint() -> ExecutionFingerprint {
    use std::collections::BTreeMap;
    let policy = DeterministicReductionPolicy::default();
    ExecutionFingerprint {
        schema_version: 1,
        toolkit_revision: "audit-fixture".into(),
        dependency_revisions: BTreeMap::from([("lock".into(), "fixture".into())]),
        compiler: "fixture compiler".into(),
        target_triple: "fixture target".into(),
        native_libraries: BTreeMap::new(),
        scalar_backend: "binary64".into(),
        scalar_backend_version: "IEEE-754".into(),
        precision: PrecisionFingerprint {
            working_precision_bits: 53,
            guard_bits: 0,
            rounding_policy: "nearest".into(),
        },
        algorithm_semantics_versions: BTreeMap::from([("reduction".into(), "v1".into())]),
        cpu_feature_policy: "portable".into(),
        thread_policy: ThreadPolicyFingerprint {
            thread_count: 1,
            scheduling_policy: DETERMINISTIC_REDUCTION_SCHEDULING_V1.into(),
            reduction_policy: policy.fingerprint_name(),
        },
        feature_flags: BTreeSet::new(),
        effective_configuration_digest: ConfigDigest("a".repeat(64)),
        resolved_resource_policy_digest: ConfigDigest("b".repeat(64)),
        reproducibility: Reproducibility::Bitwise,
    }
}
fn digest_json(value: &Value) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}
#[test]
fn reduction_import_rechecks_required_scalar_fields_after_hash_replay() {
    let fp = fingerprint();
    let original = ReproducibleReductionArtifact::new(
        &fp,
        DeterministicReductionPolicy::default(),
        "decimal",
        "1",
    )
    .unwrap();
    original.verify(&fp).unwrap();
    for field in ["scalar_encoding", "value"] {
        let mut payload = serde_json::to_value(&original).unwrap();
        payload[field] = json!("");
        payload.as_object_mut().unwrap().remove("payload_sha256");
        let digest = digest_json(&payload);
        payload["payload_sha256"] = json!(digest);
        let altered: ReproducibleReductionArtifact = serde_json::from_value(payload).unwrap();
        assert!(
            altered.verify(&fp).is_err(),
            "accepted empty {field} after replaying its hash"
        );
    }
}
fn saved_provenance() -> SolverProvenance {
    SolverProvenance::current_package("binary64")
        .with_saved_result_context(
            &fingerprint(),
            "c".repeat(64),
            "fixture",
            json!({}),
            json!({}),
        )
        .unwrap()
}
#[test]
fn saved_provenance_rejects_mirrored_determinism_mismatch() {
    let mut saved = saved_provenance();
    saved.deterministic = false;
    assert!(saved.validate_saved_result().is_err());
}
#[test]
fn saved_provenance_checks_nested_semantic_records() {
    let mut saved = saved_provenance();
    saved.artifact_semantics.push(SemanticArtifactProvenance {
        direction: ArtifactProvenanceDirection::Produced,
        artifact_family: "fixture".into(),
        semantic_key_schema_version: 1,
        resolved_semantic_key: json!({"schema_version":1}),
        semantic_digest: "invalid".into(),
    });
    assert!(saved.validate_saved_result().is_err());
}
#[test]
fn failed_cache_provenance_record_does_not_partially_mutate_the_result() {
    use std::collections::BTreeMap;
    let key = json!({"schema_version":1});
    let mut access = CacheAccessProvenance {
        schema_version: 1,
        operation: "fixture".into(),
        artifact_family: "fixture".into(),
        semantic_digest: digest_json(&key),
        semantic_key_schema_version: 1,
        resolved_semantic_key: key,
        selected_manifest_digest: Some("a".repeat(64)),
        ordered_overlays: vec!["local".into()],
        lookup_outcome: CacheLookupOutcome::Hit,
        reuse_disposition: CacheReuseDisposition::Reused,
        selected_source: Some(CacheSourceProvenance {
            overlay: "local".into(),
            location_kind: "local".into(),
            repository: "fixture".into(),
            revision: "fixture".into(),
            document_paths: BTreeMap::new(),
        }),
        rejected_candidates: vec![],
        validation_mode: CacheValidationMode::Full,
        validation_outcome: CacheValidationOutcome::Passed,
        validation_detail: None,
        validated_artifacts: vec![],
    };
    let mut saved = SolverProvenance::current_package("binary64");
    saved.record_cache_access(access.clone()).unwrap();
    let before = saved.clone();
    access.selected_manifest_digest = Some("b".repeat(64));
    assert!(saved.record_cache_access(access).is_err());
    assert_eq!(saved, before);
}
#[test]
fn contradictory_independence_evidence_cannot_promote_assurance() {
    let evidence = AssuranceEvidence {
        computation_valid: true,
        independence: Some(IndependenceAssessment {
            algorithm_semantics: "test-attestation-v1".into(),
            independent: true,
            intended_claim: "fixture".into(),
            reasons: vec!["routes share decisive intermediates".into()],
            stability_evidence: vec![],
            shared_decisive_intermediates: vec!["same result".into()],
        }),
        ..AssuranceEvidence::default()
    };
    assert_eq!(
        evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
        Some(AssuranceLevel::Computed)
    );
}
fn publication_provenance() -> PublicationProvenance {
    PublicationProvenance {
        toolkit_version: "0.15.1".into(),
        release_tag: "v0.15.1".into(),
        source_revision: "fixture".into(),
        resolved_configuration_digest: "a".repeat(64),
        execution_fingerprint_digest: "b".repeat(64),
        input_artifact_digests: vec![],
    }
}
fn convergence_row() -> ConvergenceTableRow {
    ConvergenceTableRow {
        sequence_index: 1,
        lambda_squared: "13".into(),
        n_modes: 3,
        precision_bits: 128,
        root_count: 3,
        minimum_accuracy_digits: "2".into(),
        median_accuracy_digits: "3.5".into(),
        index_penalty_digits: "-1".into(),
        completion_status: "certified".into(),
        accuracy_scope: "caller_attested_root_records".to_owned(),
        source_weights_digest: None,
    }
}
#[test]
fn convergence_export_rejects_nonfinite_or_inconsistent_numeric_summaries() {
    let build = |rows: &[ConvergenceTableRow]| {
        ccm_convergence_publication_table(
            "fixture",
            "finite observations",
            rows,
            publication_provenance(),
        )
    };
    assert!(build(&[convergence_row()]).is_ok());
    for field in 0..6 {
        let mut row = convergence_row();
        match field {
            0 => row.lambda_squared = "NaN".into(),
            1 => row.minimum_accuracy_digits = "inf".into(),
            2 => row.median_accuracy_digits = "-inf".into(),
            3 => row.index_penalty_digits = "NaN".into(),
            4 => row.lambda_squared = "1".into(),
            _ => row.minimum_accuracy_digits = "4".into(),
        }
        assert!(
            build(&[row]).is_err(),
            "accepted malformed summary field {field}"
        );
    }
}
#[test]
fn convergence_export_requires_strict_sequence_order() {
    let build = |rows: &[ConvergenceTableRow]| {
        ccm_convergence_publication_table(
            "fixture",
            "finite observations",
            rows,
            publication_provenance(),
        )
    };
    let first = convergence_row();
    let mut second = first.clone();
    second.sequence_index = 2;
    assert!(build(&[first.clone(), second.clone()]).is_ok());
    assert!(build(&[second, first.clone()]).is_err());
    assert!(build(&[first.clone(), first]).is_err());
}

#[test]
fn cancellation_snapshots_keep_the_flag_and_reason_consistent() {
    for _ in 0..300 {
        let token = CancellationToken::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                token.cancel(CancellationReason::UserRequested);
            });
            loop {
                let state = token.state();
                assert_eq!(state.requested, state.reason.is_some());
                if state.requested {
                    break;
                }
                std::thread::yield_now();
            }
        });
    }
}
