use super::*;
use serde_json::{json, Value};

fn capture(id: &str, value: &Value) -> CaptureArtifact {
    collect_capture(&json!({}), vec![id.into()], |_| {
        CapturedDiagnostic::new(value, vec![]).map_err(CaptureFailure::failed)
    })
    .unwrap()
}

fn complete_rows() -> Value {
    json!({"expected_rows":2,"data":{"rows":[{"outcome":"point_measurement"},{"outcome":"point_measurement"}]}})
}

fn repair(
    original: &CaptureArtifact,
    id: &str,
    replacement: CapturedMeasurement,
) -> CaptureMeasurementRepair {
    CaptureMeasurementRepair {
        diagnostic: id.into(),
        original_evidence_digest: measurement_evidence(id, &original.measurements[id])
            .unwrap()
            .digest
            .unwrap(),
        replacement,
    }
}

// Model a previous writer that sealed an internally consistent completed receipt.
fn reseal(id: &str, measurement: CapturedMeasurement) -> CaptureArtifact {
    let mut record = capture(id, &complete_rows());
    record.receipt =
        CaptureReceipt::new(&record.resolved_plan, record.requested_diagnostics.clone()).unwrap();
    record
        .receipt
        .record(
            id,
            DiagnosticOutcome::Completed {
                evidence: vec![measurement_evidence(id, &measurement).unwrap()],
            },
        )
        .unwrap();
    record.source_dependencies = measurement.source_dependencies.clone();
    record.measurements.insert(id.into(), measurement);
    record.numerical_coverage = record.coverage();
    record
}

#[test]
fn fresh_repaired_and_reopened_measurements_apply_the_same_completion_rules() {
    for (id, value) in [
        (
            "finite_root_budget",
            json!({"kind":"ccm_finite_diagnostic_analysis","expected_rows":2,"data":{"outcome":"missing_input","rows":[{"outcome":"unresolved"}]}}),
        ),
        (
            "root_certificate",
            json!({"outcome":"partial_certification","rows":[{"outcome":"certified_finite_enclosure"}]}),
        ),
        (
            "root_certificate",
            json!({"outcome":"certified_finite_enclosure","computed_outside_certified_rows":1}),
        ),
        (
            "root_certificate_tight",
            json!({"outcome":"certified_finite_enclosure","computed_outside_certified_rows":0,"rows":[{"computed_agreement":"outside"}]}),
        ),
        (
            "finite_root_budget",
            json!([{"data":{"execution_budget_limited":true}}]),
        ),
        (
            "root_transport",
            json!({"expected_rows":1,"rows":[{"outcome":"conditional_budget_met"}]}),
        ),
        (
            "root_transport",
            json!({"expected_rows":2,"rows":[{"outcome":"point_measurement"}]}),
        ),
        (
            "directional_response_full",
            json!({"expected_rows":1,"rows":[{"outcome":"point_measurement"},{"outcome":"unresolved"}]}),
        ),
        (
            "directional_response_full",
            json!({"expected_rows":0,"rows":[]}),
        ),
        (
            "directional_response_full",
            json!({"rows":[{"outcome":"point_measurement"}]}),
        ),
    ] {
        assert!(!capture(id, &value).receipt.is_complete(), "{id}: {value}");
        let measured = CapturedMeasurement {
            value,
            value_reference: None,
            source_dependencies: vec![],
        };
        assert!(reseal(id, measured.clone()).validate().is_err(), "{id}");
        let good = capture(id, &complete_rows());
        assert!(repair_capture_measurements(&good, vec![repair(&good, id, measured)]).is_err());
    }
    let partial = json!({"expected_rows":1,"data":{"outcome":"partial_unresolved","rows":[{"outcome":"unresolved"}]}});
    for id in ["directional_response", "spectral_cluster_full"] {
        let qualified = capture(id, &partial);
        assert!(qualified.receipt.is_complete());
        assert_eq!(qualified.coverage()[id].unresolved_rows, 1);
    }
}

fn draft(key: ArtifactKey) -> ArtifactDraft {
    ArtifactDraft {
        schema_version: 1,
        key,
        producer_toolkit_version: current_toolkit_version().unwrap(),
        minimum_reader_version: ToolkitVersion::parse(CLEAN_SLATE).unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    }
}

#[test]
fn source_bytes_decide_repair_cold_admission_and_strict_reuse_of_resealed_receipts() {
    let scratch = crate::test_support::temporary_root("source-bound-completion");
    let store = FilesystemCacheStore::new("receipts", scratch.path(), true, CacheVisibility::Local);
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "receipts",
            scratch.path(),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: current_toolkit_version().unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let context = |mode: ArtifactExecutionCacheMode| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["receipts".into()],
        mode,
        write_on_miss: !mode.requires_reuse(),
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let id = "root_transport";
    let good = capture(id, &complete_rows());
    let cold = context(ArtifactExecutionCacheMode::PreferReuse);
    let warm = context(ArtifactExecutionCacheMode::RequireReuse);
    let template = persist_capture(&good, &cold)
        .unwrap()
        .produced_manifest
        .unwrap();
    for (name, actual) in [
        ("missing", json!({"data":{"outcome":"missing_input"}})),
        (
            "partial",
            json!({"outcome":"partial_certification","rows":[{"outcome":"certified_finite_enclosure"}]}),
        ),
        (
            "outside",
            json!({"outcome":"certified_finite_enclosure","rows":[{"outcome":"certified_finite_enclosure","computed_agreement":"outside"}]}),
        ),
        (
            "short",
            json!({"expected_rows":2,"rows":[{"outcome":"unresolved"}]}),
        ),
        ("good", complete_rows()),
    ] {
        let bytes = serde_json::to_vec(&actual).unwrap();
        let source = store
            .put(
                &draft(ArtifactKey::new("fixture", name, b"fixture").unwrap()),
                &bytes,
            )
            .unwrap();
        let deps = research_source_dependencies(std::slice::from_ref(&source)).unwrap();
        let measured = CapturedMeasurement {
            value: Value::Null,
            source_dependencies: deps.clone(),
            value_reference: Some(MeasurementValueReference {
                array: false,
                artifacts: deps.clone(),
                value_digest: xc_core::research_digest_value(&actual).unwrap().0,
                coverage: NumericalCoverage::from_value(&complete_rows()),
            }),
        };
        let record = reseal(id, measured.clone());
        let fresh = capture_and_persist(
            &json!({}),
            vec![id.into()],
            |_| {
                CapturedDiagnostic::by_reference(
                    &complete_rows(),
                    vec![source.clone()],
                    false,
                    vec![],
                )
                .map_err(CaptureFailure::failed)
            },
            &cold,
        );
        assert_eq!(fresh.is_ok(), name == "good", "{name}");
        // Structural validation alone cannot distinguish two sources with the same declared coverage.
        record.validate().unwrap();
        assert!(
            repair_capture_measurements(&good, vec![repair(&good, id, measured.clone())]).is_err()
        );
        let replaced = repair_capture_measurements_with_sources(
            &good,
            vec![repair(&good, id, measured.clone())],
            |_| Ok(bytes.clone()),
        );
        assert_eq!(
            repair_capture_measurements_with_sources(&record, vec![], |_| Ok(bytes.clone()))
                .is_ok(),
            name == "good",
            "inherited {name}"
        );
        assert!(repair_capture_measurements_with_sources(
            &good,
            vec![repair(&good, id, measured)],
            |_| Ok(b"null".to_vec())
        )
        .is_err());
        if name == "good" {
            assert_eq!(replaced.unwrap(), record);
            record.validate_with_sources(&resolver, &policy).unwrap();
            persist_capture(&record, &cold).unwrap();
            assert!(persist_capture(&record, &warm)
                .unwrap()
                .reused_manifest
                .is_some());
            continue;
        }
        assert!(replaced.is_err());
        assert!(record.validate_with_sources(&resolver, &policy).is_err());
        assert!(
            persist_research_artifact(CAPTURE_RECEIPT_KIND, &record, &deps, &cold, |_| Ok(()))
                .is_err()
        );
        // Insert the record under the exact key an older writer used, then prove it
        // really exists before attempting strict warm reuse.
        let digest = xc_core::research_digest(&record).unwrap();
        let mut semantic: SemanticKeyEnvelope =
            serde_json::from_str(&template.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
        semantic.resolved_mathematical_parameters =
            json!({"record_digest":digest,"source_dependencies":deps});
        let mut old = draft(ArtifactKey {
            kind: CAPTURE_RECEIPT_KIND.into(),
            parameters_digest: semantic.digest().unwrap(),
            logical_key: format!(
                "{}attempt/{}",
                capture_receipt_plan_prefix(record.receipt.plan_digest()).unwrap(),
                digest.0
            ),
        });
        old.tags = template.tags.clone();
        old.tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.into(),
            serde_json::to_string(&semantic).unwrap(),
        );
        old.dependencies = deps;
        let stored = store
            .put(&old, &serde_json::to_vec(&record).unwrap())
            .unwrap();
        resolver
            .resolve_exact_manifest(&old.key, &stored.content_digest, &policy)
            .unwrap();
        let err = persist_capture(&record, &warm).err().unwrap();
        assert!(err.to_string().contains("measurement"), "{name}: {err}");
    }
}
