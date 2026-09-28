use std::collections::BTreeMap;
use xc_cache::{ArtifactKey, ContentDigest, ToolkitVersion};
use xc_certify::{
    verify_bundle, CertificateBundle, CertificateClaim, CertifiedArtifactRef, InertiaCertificate,
};
use xc_core::{ApproximationLedger, AssuranceLevel, SolverProvenance};

fn bundle() -> CertificateBundle {
    let mut b = CertificateBundle {
        schema_version: 1,
        certificate_id: ContentDigest::sha256(b"placeholder"),
        claim: CertificateClaim::MatrixInertia,
        assurance: AssuranceLevel::Certified,
        toolkit_version: ToolkitVersion::parse("0.15.1").unwrap(),
        citation: None,
        provenance: SolverProvenance::current_package("exact_fixture"),
        inputs: vec![],
        inertia: Some(InertiaCertificate {
            dimension: 1,
            positive: 1,
            negative: 0,
            zero_or_unresolved: 0,
            matrix_digest: ContentDigest::sha256(b"matrix"),
            scalar_backend: "exact_fixture".into(),
            precision_bits: 128,
            pivot_enclosures_digest: None,
        }),
        eigenvalue_enclosures: vec![],
        spectral_gap: None,
        exact_records: BTreeMap::new(),
        evidence_digests: BTreeMap::new(),
        approximation_ledger: ApproximationLedger::default(),
        assumptions: vec![],
        notes: vec![],
    };
    b.refresh_certificate_id().unwrap();
    assert!(verify_bundle(&b).valid);
    b
}
fn reject(mut b: CertificateBundle) {
    // A newly computed identity must not legitimize malformed record contents.
    b.refresh_certificate_id().unwrap();
    assert!(!verify_bundle(&b).valid, "accepted invalid bundle: {b:?}");
}
#[test]
fn matrix_inertia_claim_requires_its_record() {
    let mut b = bundle();
    b.inertia = None;
    reject(b);
}
#[test]
fn eigenvalue_claim_requires_an_enclosure() {
    let mut b = bundle();
    b.claim = CertificateClaim::EigenvalueEnclosure;
    reject(b);
}
#[test]
fn spectral_gap_claim_requires_a_gap_record() {
    let mut b = bundle();
    b.claim = CertificateClaim::SpectralGap;
    reject(b);
}
#[test]
fn named_evidence_digest_must_be_well_formed() {
    let mut b = bundle();
    b.evidence_digests
        .insert("proof".into(), ContentDigest("invalid".into()));
    reject(b);
}
#[test]
fn inertia_pivot_digest_must_be_well_formed() {
    let mut b = bundle();
    b.inertia.as_mut().unwrap().pivot_enclosures_digest = Some(ContentDigest("invalid".into()));
    reject(b);
}
#[test]
fn input_parameter_digest_must_be_well_formed() {
    let mut b = bundle();
    b.inputs.push(CertifiedArtifactRef {
        key: ArtifactKey {
            kind: "matrix".into(),
            logical_key: "one".into(),
            parameters_digest: ContentDigest("invalid".into()),
        },
        content_digest: ContentDigest::sha256(b"matrix"),
    });
    reject(b);
}
