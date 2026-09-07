use std::collections::BTreeMap;
use xc_core::*;

fn spec() -> HypothesisSpec {
    serde_json::from_str(include_str!("fixtures/hypothesis-spec.json")).unwrap()
}
#[test]
fn scientific_meaning_cannot_be_silently_pooled() {
    let a = spec().observable;
    let mut b = a.clone();
    b.target = ObservableTarget::SampledMinimum;
    assert!(a
        .check_compatible(&b)
        .unwrap_err()
        .to_string()
        .contains("target"));
    b = a.clone();
    b.metric = "different metric".into();
    assert!(a.check_compatible(&b).is_err());
    b = a.clone();
    b.target_definition = Some(ConfigDigest("d".repeat(64)));
    assert!(a.check_compatible(&b).is_err());
    b = a.clone();
    b.derivative_coordinate = Some("c".into());
    let mut c = b.clone();
    c.derivative_coordinate = Some("ln(c)".into());
    assert!(b.check_compatible(&c).is_err());
    b = a.clone();
    b.root = Some(RootObservable {
        index: 1,
        reference_identity: ConfigDigest("d".repeat(64)),
        error: RootErrorConvention::Absolute,
    });
    c = b.clone();
    c.root.as_mut().unwrap().error = RootErrorConvention::Relative;
    assert!(b
        .check_compatible(&c)
        .unwrap_err()
        .to_string()
        .contains("root"));
    b = a.clone();
    b.prolate = Some(ProlateObservable {
        index: 4,
        indexing_convention: "zero-based full".into(),
        deficiency_convention: "1-amplitude".into(),
        asymptotic_approximation: false,
    });
    c = b.clone();
    c.prolate.as_mut().unwrap().index = 2;
    assert!(b.check_compatible(&c).is_err());
}
#[test]
fn every_scientific_change_has_a_new_frozen_identity() {
    let a = spec();
    let frozen = a.clone().freeze().unwrap();
    let mut variants = vec![];
    let mut b = a.clone();
    b.parameters.get_mut("leading").unwrap().value = DecimalLiteral::new("0.126").unwrap();
    variants.push(b);
    let mut b = a.clone();
    b.regime.push_str(" changed");
    variants.push(b);
    let mut b = a.clone();
    b.selection_policy.push_str(" changed");
    variants.push(b);
    let mut b = a.clone();
    b.cases.get_mut("dev").unwrap().absolute_tolerance = DecimalLiteral::new("0.001").unwrap();
    variants.push(b);
    let mut b = a.clone();
    b.cases.get_mut("dev").unwrap().prediction = DecimalLiteral::new("0.126").unwrap();
    variants.push(b);
    for b in variants {
        assert_ne!(frozen.digest(), b.freeze().unwrap().digest());
    }
    let mut encoded = serde_json::to_value(&frozen).unwrap();
    encoded["specification"]["regime"] = "tampered".into();
    assert!(serde_json::from_value::<FrozenHypothesis>(encoded)
        .unwrap()
        .validate()
        .is_err());
    frozen.validate().unwrap();
}
#[test]
fn design_families_cannot_leak_between_partitions() {
    let mut s = spec();
    let mut other = s.cases["dev"].clone();
    other.partition = DatasetPartition::ProtectedValidation;
    other
        .design
        .coordinates
        .insert("c=lambda^2".into(), DecimalLiteral::new("20").unwrap());
    s.cases.insert("protected".into(), other);
    assert!(s.freeze().unwrap_err().to_string().contains("family"));
}
#[test]
fn frozen_spec_rejects_invalid_decimals_even_after_serde() {
    let mut s = serde_json::to_value(spec()).unwrap();
    s["cases"]["dev"]["prediction"] = "nan".into();
    assert!(serde_json::from_value::<HypothesisSpec>(s)
        .unwrap()
        .freeze()
        .is_err());
}
#[test]
fn capture_receipt_is_append_only_and_complete_only_with_all_evidence() {
    let mut r = CaptureReceipt::new(
        &BTreeMap::from([("plan", "test")]),
        ["a".into(), "b".into()],
    )
    .unwrap();
    assert!(!r.is_complete());
    assert!(r
        .record("not-requested", DiagnosticOutcome::Pending)
        .is_err());
    assert!(r
        .record("a", DiagnosticOutcome::Completed { evidence: vec![] })
        .is_err());
    let done = DiagnosticOutcome::Completed {
        evidence: vec![EvidenceRef::new("synthetic", "one", "test")],
    };
    r.record("a", done.clone()).unwrap();
    assert!(!r.is_complete());
    assert!(r.record("a", done.clone()).is_err());
    r.record("b", done).unwrap();
    assert!(r.is_complete());
}
