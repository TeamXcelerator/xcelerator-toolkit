//! Contract regression: independent route identities do not establish agreement.
use std::collections::BTreeSet;
use xc_core::{
    assess_route_independence, evaluate_assurance, AssuranceEvidence, AssuranceLevel,
    IndependenceDeclaration, RouteEvidence,
};

#[test]
fn route_independence_without_a_result_comparison_must_not_promote_assurance() {
    let route = |id: &str, family: &str| RouteEvidence {
        route_id: id.into(),
        algorithm_family: family.into(),
        formulation: "same stored matrix".into(),
        implementation_id: id.into(),
        decisive_intermediates: BTreeSet::new(),
        precision_bits: Some(256),
        seed: None,
        thread_count: Some(1),
        evidence_digest: None,
    };
    let independence = assess_route_independence(
        &route("route-a", "qr"),
        &route("route-b", "jacobi"),
        &IndependenceDeclaration {
            intended_claim: "smallest eigenvalue of the supplied matrix".into(),
            rationale: "independent algorithms and implementations".into(),
            accepted_shared_inputs: BTreeSet::new(),
        },
    );
    assert!(independence.independent);
    // No route result values, residuals, tolerance, comparison decision, or
    // agreement evidence have been supplied or evaluated anywhere above.
    let evaluation = evaluate_assurance(
        AssuranceLevel::CrossChecked,
        true,
        &AssuranceEvidence {
            computation_valid: true,
            independence: Some(independence),
            ..AssuranceEvidence::default()
        },
    );
    assert_ne!(
        evaluation.achieved,
        Some(AssuranceLevel::CrossChecked),
        "structural independence alone was promoted to result agreement: {evaluation:?}"
    );
}

#[test]
fn cross_check_requires_matching_explicit_accepted_result_evidence() {
    use xc_core::{IndependenceAssessment, RouteComparisonEvidence};
    let mut evidence = AssuranceEvidence {
        computation_valid: true,
        independence: Some(IndependenceAssessment {
            algorithm_semantics: "test-attestation-v1".into(),
            independent: true,
            intended_claim: "stored eigenvalue".into(),
            reasons: vec![],
            stability_evidence: vec![],
            shared_decisive_intermediates: vec![],
        }),
        comparison: Some(RouteComparisonEvidence {
            intended_claim: "stored eigenvalue".into(),
            left_result_digest: "a".repeat(64),
            right_result_digest: "b".repeat(64),
            comparison_rule: "exact equality of enclosed rational singleton results".into(),
            agreement_accepted: true,
        }),
        ..AssuranceEvidence::default()
    };
    assert_eq!(
        evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
        Some(AssuranceLevel::CrossChecked)
    );
    evidence.comparison.as_mut().unwrap().agreement_accepted = false;
    assert_eq!(
        evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
        Some(AssuranceLevel::Computed)
    );
    evidence.comparison.as_mut().unwrap().agreement_accepted = true;
    evidence.comparison.as_mut().unwrap().intended_claim = "different claim".into();
    assert_eq!(
        evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
        Some(AssuranceLevel::Computed)
    );
    evidence.comparison.as_mut().unwrap().intended_claim = "stored eigenvalue".into();
    evidence
        .comparison
        .as_mut()
        .unwrap()
        .left_result_digest
        .clear();
    assert_eq!(
        evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
        Some(AssuranceLevel::Computed)
    );
}
