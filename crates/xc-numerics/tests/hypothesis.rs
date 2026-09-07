#![cfg(feature = "hp")]
use std::collections::BTreeSet;
use xc_cache::ContentDigest;
use xc_core::*;
use xc_numerics::hypothesis::*;

fn d(s: &str) -> DecimalLiteral {
    DecimalLiteral::new(s).unwrap()
}

#[test]
fn scheduled_n_ladder_retains_realized_quadrature_and_rejects_control_drift() {
    let mut a = payload();
    a.design.n_modes = Some(4);
    a.design.dimension = 5;
    a.design.quadrature_identity = "gl-order-20".into();
    let mut b = a.clone();
    b.design.n_modes = Some(8);
    b.design.dimension = 9;
    b.design.quadrature_identity = "gl-order-36".into();
    b.value = ObservedScalar::Finite {
        value: d("0.12502"),
    };
    let mut ladder = vec![a, b];
    let schedule = StabilizationQuadratureSchedule {
        policy_identity: "gauss-legendre order=4*(N+1)".into(),
        identities_by_n_modes: [(4, "gl-order-20".into()), (8, "gl-order-36".into())].into(),
    };
    assert!(stabilization_ladder(&ladder, 256).is_err());
    let result = stabilization_ladder_with_quadrature_schedule(&ladder, 256, &schedule).unwrap();
    assert_eq!(result.quadrature_schedule, schedule);
    assert_eq!(
        result.observation_digests[1],
        research_digest(&ladder[1]).unwrap()
    );
    assert!(result.steps[0].interpretation.contains("coupled"));
    ladder[1].design.quadrature_identity = "gl-order-40".into();
    assert!(stabilization_ladder_with_quadrature_schedule(&ladder, 256, &schedule).is_err());
    ladder[1].design.quadrature_identity = "gl-order-36".into();
    ladder[1].design.method = "different assembly".into();
    assert!(stabilization_ladder_with_quadrature_schedule(&ladder, 256, &schedule).is_err());
}
fn spec() -> HypothesisSpec {
    serde_json::from_str(include_str!(
        "../../xc-core/tests/fixtures/hypothesis-spec.json"
    ))
    .unwrap()
}
fn payload() -> ObservationPayload {
    let s = spec();
    ObservationPayload {
        schema_version: 1,
        observable: s.observable,
        design: s.cases["dev"].design.clone(),
        value: ObservedScalar::Finite {
            value: d("0.12501"),
        },
        resolution: ObservableResolution {
            source_precision_bits: 256,
            analysis_precision_bits: 256,
            export_significant_digits: 80,
            components: RESOLUTION_AXES
                .iter()
                .map(|&axis| ResolutionComponent {
                    axis,
                    classification: ResolutionClass::EmpiricalDiscrepancy,
                    absolute: Some(d("0.0000001")),
                    explanation: "synthetic supplied uncertainty, not a certificate".into(),
                    evidence: vec![EvidenceRef::new("synthetic", "repeat", "not independent")],
                    dependency_groups: BTreeSet::from(["shared-parent".into()]),
                })
                .collect(),
        },
    }
}
fn metadata(id: &str, p: &ObservationPayload, bytes: &[u8]) -> ObservationMetadata {
    ObservationMetadata {
        case_id: id.into(),
        family_id: "c13".into(),
        observable: p.observable.clone(),
        design: p.design.clone(),
        payload_sha256: ConfigDigest(ContentDigest::sha256(bytes).0),
        sources: vec![ArtifactRef {
            kind: "synthetic".into(),
            logical_key: "case13".into(),
            semantic_digest: "d".repeat(64),
            payload_digest: "e".repeat(64),
            completion: CompletionStatus::Successful,
            assurance: Some(AssuranceLevel::Computed),
            disposition: "retained".into(),
            locations: vec![],
            publication_states: Default::default(),
        }],
    }
}
fn run(s: HypothesisSpec, p: ObservationPayload) -> HypothesisScore {
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    score_frozen_hypothesis(
        &s.freeze().unwrap(),
        &[m],
        DatasetPartition::Development,
        false,
        256,
        |_| Ok(bytes.clone()),
    )
    .unwrap()
    .value
    .unwrap()
}
#[test]
fn pass_fail_and_unresolved_are_scientific_not_rmse_labels() {
    assert_eq!(
        run(spec(), payload()).verdict,
        HypothesisVerdict::PassOnTestedDomain
    );
    let mut p = payload();
    p.value = ObservedScalar::Finite { value: d("0.126") };
    assert_eq!(
        run(spec(), p).verdict,
        HypothesisVerdict::FailOnTestedDomain
    );
    let mut p = payload();
    p.resolution.components[0].absolute = Some(d("0.001"));
    assert_eq!(
        run(spec(), p).verdict,
        HypothesisVerdict::UnresolvedAtCurrentResolution
    );
}
#[test]
fn unknown_finite_n_tail_and_missing_axis_cannot_be_zero() {
    let mut p = payload();
    let c = p
        .resolution
        .components
        .iter_mut()
        .find(|c| c.axis == ResolutionAxis::FiniteN)
        .unwrap();
    c.classification = ResolutionClass::Unknown;
    c.absolute = None;
    let r = run(spec(), p.clone());
    assert_eq!(r.verdict, HypothesisVerdict::UnresolvedAtCurrentResolution);
    assert!(r.cases[0].signed_residual.is_none());
    p.resolution.components.pop();
    assert_eq!(run(spec(), p).verdict, HypothesisVerdict::IneligibleData);
}
#[test]
fn unknown_zero_and_sign_are_preserved() {
    for value in [
        ObservedScalar::RoundedZero,
        ObservedScalar::UnresolvedSign {
            approximation: d("-1e-90"),
        },
    ] {
        let mut p = payload();
        p.value = value.clone();
        let r = run(spec(), p);
        assert_eq!(r.verdict, HypothesisVerdict::UnresolvedAtCurrentResolution);
        assert_eq!(r.cases[0].observed, Some(value));
    }
}
#[test]
fn metadata_gates_precede_payload_access_including_protected_cases() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let mut s = spec();
    let mut c = s.cases["dev"].clone();
    c.family_id = "c20".into();
    c.design.coordinates.insert("c=lambda^2".into(), d("20"));
    c.design.exact_input_identity = ConfigDigest("f".repeat(64));
    c.partition = DatasetPartition::ProtectedValidation;
    s.cases.insert("protected".into(), c);
    let mut protected = metadata("protected", &p, &bytes);
    protected.family_id = "c20".into();
    protected.payload_sha256 = ConfigDigest("f".repeat(64));
    let m = metadata("dev", &p, &bytes);
    let mut calls = 0;
    let frozen = s.freeze().unwrap();
    let r = score_frozen_hypothesis(
        &frozen,
        &[protected, m],
        DatasetPartition::Development,
        false,
        256,
        |m| {
            calls += 1;
            assert_eq!(m.case_id, "dev");
            Ok(bytes.clone())
        },
    )
    .unwrap()
    .value
    .unwrap();
    assert_eq!(calls, 1);
    assert!(r.complete);
    assert!(score_frozen_hypothesis(
        &frozen,
        &[],
        DatasetPartition::ProtectedValidation,
        false,
        256,
        |_| panic!("must not load")
    )
    .is_err());
    let mut m = metadata("dev", &p, &bytes);
    m.observable.metric = "incompatible".into();
    let r = score_frozen_hypothesis(
        &frozen,
        &[m],
        DatasetPartition::Development,
        false,
        256,
        |_| panic!("must not load"),
    )
    .unwrap()
    .value
    .unwrap();
    assert_eq!(r.verdict, HypothesisVerdict::IneligibleData);
}
#[test]
fn missing_required_points_never_pass_a_smaller_cohort() {
    let mut s = spec();
    let mut c = s.cases["dev"].clone();
    c.family_id = "c20".into();
    c.design.coordinates.insert("c=lambda^2".into(), d("20"));
    s.cases.insert("missing".into(), c);
    let r = run(s, payload());
    assert!(!r.complete);
    assert_eq!(r.required_observations, 2);
    assert_eq!(r.loaded_observations, 1);
    assert_eq!(r.verdict, HypothesisVerdict::UnresolvedAtCurrentResolution);
}
#[test]
fn duplicate_acquisition_is_not_counted_twice() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    let r = score_frozen_hypothesis(
        &spec().freeze().unwrap(),
        &[m.clone(), m],
        DatasetPartition::Development,
        false,
        256,
        |_| panic!("duplicate must not load"),
    )
    .unwrap()
    .value
    .unwrap();
    assert_eq!(r.verdict, HypothesisVerdict::IneligibleData);
    assert_eq!(r.loaded_observations, 0);
}
#[test]
fn source_bytes_and_decoded_design_are_both_authenticated() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    let r = score_frozen_hypothesis(
        &spec().freeze().unwrap(),
        &[m],
        DatasetPartition::Development,
        false,
        256,
        |_| Ok(b"wrong".to_vec()),
    )
    .unwrap()
    .value
    .unwrap();
    assert_eq!(r.verdict, HypothesisVerdict::IneligibleData);
    let mut altered = p.clone();
    altered.design.method = "different".into();
    let b = serde_json::to_vec(&altered).unwrap();
    let m = metadata("dev", &p, &b);
    let r = score_frozen_hypothesis(
        &spec().freeze().unwrap(),
        &[m],
        DatasetPartition::Development,
        false,
        256,
        |_| Ok(b.clone()),
    )
    .unwrap()
    .value
    .unwrap();
    assert_eq!(r.verdict, HypothesisVerdict::IneligibleData);
}
#[test]
fn factor_ten_log_units_and_depth_sign_agree() {
    let natural = convert_log_units(
        &d("1"),
        LogBase::Decimal,
        false,
        LogBase::Natural,
        false,
        256,
    )
    .unwrap();
    let ln10 = rug::Float::with_val(256, 10).ln();
    let lo = enclose_decimal(&d(&natural.lower), 256).unwrap();
    let hi = enclose_decimal(&d(&natural.upper), 256).unwrap();
    assert!(lo.lower() < &ln10 && hi.upper() > &ln10);
    let neg = convert_log_units(
        &d("1"),
        LogBase::Decimal,
        false,
        LogBase::Decimal,
        true,
        256,
    )
    .unwrap();
    assert!(d(&neg.lower).cmp_numeric(&d("-1")).unwrap().is_lt());
    assert!(d(&neg.upper).cmp_numeric(&d("-1")).unwrap().is_gt());
}
#[test]
fn cancellation_sensitive_correction_keeps_input_uncertainty() {
    let mut p = payload();
    p.value = ObservedScalar::Finite {
        value: d("0.1250000000000000000000000000000000000001"),
    };
    for c in &mut p.resolution.components {
        c.absolute = Some(d("1e-35"));
    }
    let r = scaled_correction(&p, &d("0.125"), &d("1e-40"), 256)
        .unwrap()
        .unwrap();
    assert!(d(&r.lower).cmp_numeric(&d("0")).unwrap().is_lt());
    assert!(d(&r.upper).cmp_numeric(&d("1")).unwrap().is_gt());
    for c in &mut p.resolution.components {
        c.absolute = Some(d("1e-60"));
    }
    let r = scaled_correction(&p, &d("0.125"), &d("1e-40"), 256)
        .unwrap()
        .unwrap();
    assert!(d(&r.lower)
        .cmp_numeric(&d("0.999999999999"))
        .unwrap()
        .is_gt());
    assert!(d(&r.upper)
        .cmp_numeric(&d("1.000000000001"))
        .unwrap()
        .is_lt());
}
#[test]
fn repeat_scores_are_byte_identical_and_do_not_mutate_sources() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let before = bytes.clone();
    let m = metadata("dev", &p, &bytes);
    let s = spec().freeze().unwrap();
    let mut reports = vec![];
    for _ in 0..3 {
        reports.push(
            serde_json::to_vec(
                &score_frozen_hypothesis(
                    &s,
                    std::slice::from_ref(&m),
                    DatasetPartition::Development,
                    false,
                    256,
                    |_| Ok(bytes.clone()),
                )
                .unwrap()
                .value,
            )
            .unwrap(),
        );
    }
    assert!(reports.windows(2).all(|x| x[0] == x[1]));
    assert_eq!(bytes, before);
}

#[test]
fn original_sign_survives_signed_log_and_disagrees_explicitly() {
    let mut s = spec();
    s.observable.transform = ObservableTransform::SignedLogMagnitude {
        base: LogBase::Decimal,
        negative: true,
    };
    s.cases.get_mut("dev").unwrap().prediction_sign = Some(NonzeroSign::Positive);
    s.cases.get_mut("dev").unwrap().prediction = d("1000");
    let mut p = payload();
    p.observable = s.observable.clone();
    p.value = ObservedScalar::SignedLogMagnitude {
        sign: NonzeroSign::Negative,
        log_magnitude: d("1000"),
    };
    assert_eq!(
        run(s.clone(), p.clone()).verdict,
        HypothesisVerdict::FailOnTestedDomain
    );
    p.value = ObservedScalar::Finite { value: d("1000") };
    assert_eq!(run(s, p).verdict, HypothesisVerdict::IneligibleData);
}
#[test]
fn signed_adjacent_changes_keep_denominator_and_reject_mixed_precision() {
    let mut ladder = Vec::new();
    for (i, v) in ["0.13", "0.126", "0.125"].iter().enumerate() {
        let mut p = payload();
        p.value = ObservedScalar::Finite { value: d(v) };
        p.design.n_modes = Some(4 + i);
        p.design.dimension = 9 + 2 * i;
        ladder.push(p);
    }
    let r = stabilization_ladder(&ladder, 256).unwrap();
    assert!(r[0].signed_change_ratio.is_none());
    let denom = r[1].ratio_denominator.as_ref().unwrap();
    assert!(d(&denom.lower).cmp_numeric(&d("-0.004")).unwrap().is_lt());
    let ratio = r[1].signed_change_ratio.as_ref().unwrap();
    assert!(d(&ratio.lower).cmp_numeric(&d("0.24")).unwrap().is_gt());
    assert!(d(&ratio.upper).cmp_numeric(&d("0.26")).unwrap().is_lt());
    ladder[2].design.source_precision_bits = 512;
    ladder[2].resolution.source_precision_bits = 512;
    assert!(stabilization_ladder(&ladder, 256).is_err());
}
#[test]
fn conversion_and_scoring_precision_cannot_recover_missing_construction_digits() {
    let mut p = payload();
    p.resolution.components[0].classification = ResolutionClass::Unknown;
    p.resolution.components[0].absolute = None;
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    for bits in [128, 256, 512] {
        let r = score_frozen_hypothesis(
            &spec().freeze().unwrap(),
            std::slice::from_ref(&m),
            DatasetPartition::Development,
            false,
            bits,
            |_| Ok(bytes.clone()),
        )
        .unwrap();
        assert_eq!(r.achieved_assurance, Some(AssuranceLevel::Computed));
        assert_eq!(
            r.value.unwrap().verdict,
            HypothesisVerdict::UnresolvedAtCurrentResolution
        );
    }
}

#[test]
fn protected_payload_alias_cannot_be_loaded_under_a_development_id() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let dev = metadata("dev", &p, &bytes);
    let mut s = spec();
    let mut planned = s.cases["dev"].clone();
    planned.family_id = "c20".into();
    planned.partition = DatasetPartition::ProtectedValidation;
    planned
        .design
        .coordinates
        .insert("c=lambda^2".into(), d("20"));
    s.cases.insert("protected".into(), planned);
    let mut protected = dev.clone();
    protected.case_id = "protected".into();
    protected.family_id = "c20".into();
    let r = score_frozen_hypothesis(
        &s.freeze().unwrap(),
        &[dev, protected],
        DatasetPartition::Development,
        false,
        256,
        |_| panic!("aliased protected bytes must not load"),
    )
    .unwrap()
    .value
    .unwrap();
    assert_eq!(r.verdict, HypothesisVerdict::IneligibleData);
    assert_eq!(r.loaded_observations, 0);
}

#[test]
fn evaluation_packet_replays_exact_selected_bytes_and_rejects_score_or_coverage_tampering() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    let frozen = spec().freeze().unwrap();
    let packet = evaluate_hypothesis_packet(
        &frozen,
        &[m],
        DatasetPartition::Development,
        false,
        256,
        |_| Ok(bytes.clone()),
    )
    .unwrap();
    packet.validate(false).unwrap();
    assert_eq!(packet.selected_observations.len(), 1);
    let encoded = serde_json::to_vec(&packet).unwrap();
    serde_json::from_slice::<HypothesisEvaluationPacket>(&encoded)
        .unwrap()
        .validate(false)
        .unwrap();
    let mut tampered = packet.clone();
    tampered.score.cases[0].reason = "invented conclusion".into();
    assert!(tampered.validate(false).is_err());
    let mut tampered = packet.clone();
    tampered.selected_observations[0].payload = Some("{}".into());
    assert!(tampered.validate(false).is_err());
    let mut tampered = packet;
    tampered.selected_observations.clear();
    assert!(tampered.validate(false).is_err());
}
#[test]
fn evaluation_packets_keep_unavailable_outcomes_and_never_read_unselected_payloads() {
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    let frozen = spec().freeze().unwrap();
    let unavailable = evaluate_hypothesis_packet(
        &frozen,
        std::slice::from_ref(&m),
        DatasetPartition::Development,
        false,
        256,
        |_| anyhow::bail!("unavailable"),
    )
    .unwrap();
    assert!(!unavailable.score.complete);
    unavailable.validate(false).unwrap();
    let mut dropped = unavailable.clone();
    dropped.selected_observations.clear();
    assert!(dropped.validate(false).is_err());
    let mut other = m.clone();
    other.case_id = "unplanned".into();
    let packet = evaluate_hypothesis_packet(
        &frozen,
        &[m, other],
        DatasetPartition::Development,
        false,
        256,
        |selected| {
            assert_eq!(selected.case_id, "dev");
            Ok(bytes.clone())
        },
    )
    .unwrap();
    assert_eq!(packet.selected_observations.len(), 1);
    packet.validate(false).unwrap();
    assert!(evaluate_hypothesis_packet(
        &frozen,
        &[],
        DatasetPartition::ProtectedValidation,
        false,
        256,
        |_| panic!("protected read")
    )
    .is_err());
}

#[test]
fn managed_evaluation_requires_exact_sources_and_roundtrips_with_required_reuse() {
    use xc_cache::*;
    let p = payload();
    let bytes = serde_json::to_vec(&p).unwrap();
    let m = metadata("dev", &p, &bytes);
    let packet = evaluate_hypothesis_packet(
        &spec().freeze().unwrap(),
        &[m],
        DatasetPartition::Development,
        false,
        256,
        |_| Ok(bytes.clone()),
    )
    .unwrap();
    let source = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey {
            kind: "synthetic".into(),
            logical_key: "case13".into(),
            parameters_digest: ContentDigest("d".repeat(64)),
        },
        content_digest: ContentDigest("e".repeat(64)),
        size_bytes: 1,
        objects: vec![CacheObjectRef {
            content_digest: ContentDigest("e".repeat(64)),
            size_bytes: 1,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.15.0").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: Default::default(),
        provenance_digest: None,
    };
    let root = std::env::temp_dir().join(format!(
        "xc-evaluation-record-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "evaluation",
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
    let context = |mode: ArtifactExecutionCacheMode| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["evaluation".into()],
        mode,
        write_on_miss: !mode.requires_reuse(),
        write_visibility: CacheVisibility::Local,
        requested_assurance: AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let cold_context = context(ArtifactExecutionCacheMode::PreferReuse);
    assert!(persist_hypothesis_evaluation(&packet, &[], false, &cold_context).is_err());
    let cold =
        persist_hypothesis_evaluation(&packet, std::slice::from_ref(&source), false, &cold_context)
            .unwrap();
    let warm = persist_hypothesis_evaluation(
        &packet,
        std::slice::from_ref(&source),
        false,
        &context(ArtifactExecutionCacheMode::RequireReuse),
    )
    .unwrap();
    assert_eq!(cold.value, warm.value);
    assert_eq!(cold.produced_manifest, warm.reused_manifest);
    for quality in [CacheQuality::CrossChecked, CacheQuality::Certified] {
        let mut promoted = source.clone();
        promoted.quality = quality;
        let reused = persist_hypothesis_evaluation(
            &packet,
            &[promoted],
            false,
            &context(ArtifactExecutionCacheMode::RequireReuse),
        )
        .unwrap();
        assert_eq!(reused.reused_manifest, warm.reused_manifest);
    }
    let mut wrong = source;
    wrong.content_digest = ContentDigest("f".repeat(64));
    assert!(persist_hypothesis_evaluation(&packet, &[wrong], false, &cold_context).is_err());
    drop(resolver);
    std::fs::remove_dir_all(root).unwrap();
}
