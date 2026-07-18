use std::collections::BTreeMap;
use xc_cache::{ContentDigest, ToolkitVersion};
use xc_certify::{verify_bundle, CertificateBundle, CertificateClaim, InertiaCertificate};
use xc_core::{
    ApproximationLedger, ArchiveAuthor, ArtifactCitationMetadata, AssuranceLevel, SolverProvenance,
};

fn main() {
    let digest = ContentDigest::sha256(b"example symmetric matrix");
    let mut bundle = CertificateBundle {
        schema_version: 1,
        certificate_id: ContentDigest("00".repeat(32)),
        claim: CertificateClaim::PositiveDefiniteMatrix,
        assurance: AssuranceLevel::Certified,
        toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
        citation: None,
        provenance: SolverProvenance::current_package("exact_rational_fixture"),
        inputs: Vec::new(),
        inertia: Some(InertiaCertificate {
            dimension: 3,
            positive: 3,
            negative: 0,
            zero_or_unresolved: 0,
            matrix_digest: digest,
            scalar_backend: "exact_rational_fixture".to_owned(),
            precision_bits: 256,
            pivot_enclosures_digest: None,
        }),
        eigenvalue_enclosures: Vec::new(),
        spectral_gap: None,
        exact_records: BTreeMap::new(),
        evidence_digests: BTreeMap::new(),
        approximation_ledger: ApproximationLedger::default(),
        assumptions: vec!["example finite matrix is represented exactly".to_owned()],
        notes: vec!["documentation example only".to_owned()],
    };
    bundle
        .attach_citation(ArtifactCitationMetadata {
            schema_version: 1,
            artifact_title: "Example positive-definite matrix certificate".to_owned(),
            artifact_type: "certificate_bundle".to_owned(),
            authors: vec![ArchiveAuthor {
                given_names: "Ronnie".to_owned(),
                family_names: "Andrews".to_owned(),
                name_suffix: Some("Jr.".to_owned()),
                orcid: "https://orcid.org/0009-0003-9724-3104".to_owned(),
            }],
            software_title: "Xcelerator Toolkit".to_owned(),
            software_version: "0.13.0".to_owned(),
            repository: "https://github.com/TeamXcelerator/xcelerator-toolkit".to_owned(),
            preferred_citation: "Andrews, Example positive-definite matrix certificate (2026)."
                .to_owned(),
            software_doi: None,
            artifact_doi: None,
        })
        .unwrap();
    let report = verify_bundle(&bundle);
    assert!(report.valid, "certificate should verify: {report:?}");
    println!("{}", serde_json::to_string_pretty(&bundle).unwrap());
}
