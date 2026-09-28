use serde::Serialize;
use xc_core::*;

fn provenance() -> SolverProvenance {
    SolverProvenance::current_package("manufactured-finite-control")
}

#[test]
fn terminal_completions_are_absorbing_under_every_status_and_evidence_update() {
    let statuses = [
        ResultStatus::Converged,
        ResultStatus::Approximate,
        ResultStatus::Failed,
        ResultStatus::Inconclusive,
        ResultStatus::UnresolvedCluster,
        ResultStatus::UnresolvedEigenspace,
        ResultStatus::InsufficientPrecision,
        ResultStatus::InvalidConfiguration,
    ];
    for terminal in [CompletionStatus::Failed, CompletionStatus::Cancelled] {
        for first in &statuses {
            for second in &statuses {
                let mut result = ResearchResult::computed(17u32, provenance());
                result.completion = terminal;
                let result = result
                    .with_status(first.clone())
                    .with_status(second.clone())
                    .with_assurance_evidence(&AssuranceEvidence {
                        computation_valid: true,
                        ..AssuranceEvidence::default()
                    });
                assert_eq!(result.completion, terminal);
                assert_eq!(result.achieved_assurance, None);
                result.validate_for_persistence().unwrap();
            }
        }
    }
}

#[test]
fn actual_evidence_labels_pass_but_missing_primary_computation_never_does() {
    for level in [
        AssuranceLevel::Computed,
        AssuranceLevel::CrossChecked,
        AssuranceLevel::Certified,
    ] {
        let mut evidence = AssuranceEvidence {
            computation_valid: true,
            ..AssuranceEvidence::default()
        };
        if level == AssuranceLevel::CrossChecked {
            evidence.independence = Some(IndependenceAssessment {
                algorithm_semantics: "test-attestation-v1".into(),
                independent: true,
                intended_claim: "exact manufactured integer".into(),
                reasons: vec![],
                stability_evidence: vec![],
                shared_decisive_intermediates: vec![],
            });
            evidence.comparison = Some(RouteComparisonEvidence {
                intended_claim: "exact manufactured integer".into(),
                left_result_digest: "a".repeat(64),
                right_result_digest: "b".repeat(64),
                comparison_rule: "equal exact integers".into(),
                agreement_accepted: true,
            });
        }
        if level == AssuranceLevel::Certified {
            evidence.certificate_verified = true;
            evidence.certificate_claim_scope = Some("exact manufactured integer".into());
        }
        let mut result = ResearchResult::for_request(17u32, level, provenance())
            .with_assurance_evidence(&evidence);
        assert_eq!(result.achieved_assurance, Some(level));
        result.validate_for_persistence().unwrap();
        result
            .missing_assurance_checks
            .push("valid primary computation".into());
        assert!(result.validate_assurance_consistency().is_err());
        assert!(result.validate_for_persistence().is_err());
        result.missing_assurance_checks.clear();
        result
            .completed_assurance_checks
            .retain(|s| s != "primary computation diagnostics accepted");
        assert!(result.validate_for_persistence().is_err());
    }
}

#[derive(Serialize)]
enum NestedMeasurement {
    Wide { samples: Vec<Option<f64>> },
    Narrow(Option<f32>),
}

#[test]
fn nonfinite_values_are_rejected_before_null_and_finite_ieee_values_round_trip() {
    for x in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff0_0000_0000_0001),
    ] {
        let value = NestedMeasurement::Wide {
            samples: vec![None, Some(x)],
        };
        assert!(research_digest(&value).is_err());
        assert!(ResearchResult::computed(value, provenance())
            .validate_for_persistence()
            .is_err());
    }
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(research_digest(&NestedMeasurement::Narrow(Some(x))).is_err());
    }
    let mut bits = 0x9e37_79b9_7f4a_7c15u64;
    let mut values = vec![0.0, -0.0, f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX];
    for _ in 0..2048 {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        let value = f64::from_bits(bits);
        if value.is_finite() {
            values.push(value);
        }
    }
    for x in values {
        let record = ResearchResult::computed(vec![x], provenance());
        record.validate_for_persistence().unwrap();
        let bytes = finite_json::to_vec(&record).unwrap();
        let decoded: ResearchResult<Vec<f64>> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.value.unwrap()[0].to_bits(), x.to_bits());
        assert_eq!(
            finite_json::to_value(&x).unwrap(),
            serde_json::to_value(x).unwrap()
        );
    }
    assert!(research_digest(&Option::<f64>::None).is_ok());
}

#[test]
fn historical_semantics_are_preserved_until_a_new_transition_is_performed() {
    let current = ResearchResult::computed(17u32, provenance());
    assert_eq!(current.semantics, RESEARCH_RESULT_SEMANTICS);
    let mut old = serde_json::to_value(&current).unwrap();
    old.as_object_mut().unwrap().remove("semantics");
    let old: ResearchResult<u32> = serde_json::from_value(old).unwrap();
    assert_eq!(old.semantics, "research-result-unversioned-v1");
    old.validate_for_persistence().unwrap();
    assert_eq!(
        old.with_status(ResultStatus::Approximate).semantics,
        RESEARCH_RESULT_SEMANTICS
    );
}
