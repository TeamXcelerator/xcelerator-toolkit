use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use xc_core::*;

fn decimal(text: &str) -> DecimalLiteral {
    DecimalLiteral::new(text).unwrap()
}

#[test]
fn binary64_decimal_underflow_is_not_a_successful_zero() {
    for input in ["1e-400", "-1e-400", "9e-999999"] {
        assert!(decimal(input).parse_f64().is_err(), "{input}");
    }
    assert_eq!(decimal("-0.00e-400").parse_f64().unwrap(), -0.0);
    assert!(decimal("5e-324").parse_f64().is_err());
    assert_eq!(
        decimal("2.2250738585072014e-308").parse_f64().unwrap(),
        f64::MIN_POSITIVE
    );
}

#[test]
fn exact_multi_term_comparison_preserves_cancellation_and_sparse_exponents() {
    let terms = [
        decimal("1e1000000"),
        decimal("-1e1000000"),
        decimal("0.1"),
        decimal("0.2"),
    ];
    assert_eq!(
        decimal("0.3")
            .cmp_sum_many(&terms.iter().collect::<Vec<_>>())
            .unwrap(),
        Ordering::Equal
    );
    assert_eq!(
        decimal("0.2999999999999999999999999999")
            .cmp_sum_many(&[&decimal("0.1"), &decimal("0.2")])
            .unwrap(),
        Ordering::Less
    );
}

fn budget() -> AnalyticProblemContext {
    AnalyticProblemContext {
        schema_version: 1,
        domain: "manufactured additive bound".into(),
        requested_assurance: AssuranceLevel::Certified,
        assumptions: vec![],
        error_budget: AnalyticErrorBudget {
            quantity: "value".into(),
            aggregation_method: "triangle inequality".into(),
            total_absolute_bound: Some(decimal("3e-1000")),
            terms: (0..3)
                .map(|i| AnalyticErrorTerm {
                    term_id: format!("term-{i}"),
                    source: "exact manufactured bound".into(),
                    affects: "value".into(),
                    decisive_for_claim: true,
                    rigorous: true,
                    absolute_bound: Some(decimal("1e-1000")),
                    proof_method: Some("hypothesis".into()),
                    evidence_sha256: Some("a".repeat(64)),
                })
                .collect(),
        },
    }
}

#[test]
fn certified_error_budget_checks_exact_sum_and_quantity() {
    let mut c = budget();
    c.validate().unwrap();
    c.error_budget.total_absolute_bound = Some(decimal("2.9999999999999999999999e-1000"));
    assert!(c.validate().is_err());
    c = budget();
    c.error_budget.terms[0].affects = "another quantity".into();
    assert!(c.validate().is_err());
    c = budget();
    c.error_budget.aggregation_method = "unverified cancellation".into();
    assert!(c.validate().is_err());
    c.requested_assurance = AssuranceLevel::Computed;
    c.validate().unwrap();
}

fn access() -> CacheAccessProvenance {
    let key = serde_json::json!({"schema_version":1});
    CacheAccessProvenance {
        schema_version: 1,
        operation: "manufactured.lookup".into(),
        artifact_family: "manufactured".into(),
        semantic_digest: format!("{:x}", Sha256::digest(serde_json::to_vec(&key).unwrap())),
        semantic_key_schema_version: 1,
        resolved_semantic_key: key,
        selected_manifest_digest: Some("b".repeat(64)),
        ordered_overlays: vec!["local".into()],
        lookup_outcome: CacheLookupOutcome::Hit,
        reuse_disposition: CacheReuseDisposition::Reused,
        selected_source: Some(CacheSourceProvenance {
            overlay: "local".into(),
            location_kind: "local".into(),
            repository: "manufactured".into(),
            revision: "c".repeat(40),
            document_paths: Default::default(),
        }),
        rejected_candidates: vec![],
        validation_mode: CacheValidationMode::Fast,
        validation_outcome: CacheValidationOutcome::Passed,
        validation_detail: None,
        validated_artifacts: vec![],
    }
}

#[test]
fn contradictory_validation_cannot_be_recorded_as_successful_reuse() {
    let valid = access();
    valid.validate().unwrap();
    let mut failed = valid.clone();
    failed.validation_outcome = CacheValidationOutcome::Failed;
    assert!(failed.validate().is_err());
    let mut none = valid.clone();
    none.validation_mode = CacheValidationMode::None;
    assert!(none.validate().is_err());
    none.validation_outcome = CacheValidationOutcome::NotRequested;
    none.validate().unwrap();
    none.validated_artifacts
        .push(CacheValidatedArtifactProvenance {
            semantic_digest: "d".repeat(64),
            manifest_digest: "e".repeat(64),
        });
    assert!(none.validate().is_err());
    none.validation_mode = CacheValidationMode::Root;
    none.validation_outcome = CacheValidationOutcome::Passed;
    none.validate().unwrap();
    let mut inspected = none;
    inspected.reuse_disposition = CacheReuseDisposition::InspectedOnly;
    inspected.validation_outcome = CacheValidationOutcome::Failed;
    assert!(inspected.validate().is_err());
    inspected.validated_artifacts.clear();
    inspected.validate().unwrap();
}
