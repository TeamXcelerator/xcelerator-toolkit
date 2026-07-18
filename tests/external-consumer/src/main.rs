use std::collections::BTreeMap;
use xc_cache::{ContentDigest, ToolkitVersion};
use xc_certify::{verify_bundle, CertificateBundle, CertificateClaim, InertiaCertificate};
use xc_core::{
    ApproximationLedger, AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionPolicy,
    Reproducibility, SolverConfig, SolverProvenance, StoppingPolicy, Subspace,
};
use xc_operator::DiagonalF64;
use xc_solver::{DenseReferenceSolverF64, EigenSolverF64, SymmetricProblemF64};

fn public_solver_and_certificate_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let operator = DiagonalF64::new("external-positive-diagonal", vec![2.0, 3.0])?;
    let configuration = SolverConfig {
        target: EigenTarget::AlgebraicSmallest,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        stopping: StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-12")?,
            scaled_backward_error: DecimalLiteral::new("1e-12")?,
            maximum_iterations: 20,
            minimum_iterations: 1,
        },
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: vec!["dense_materialized_reference_f64".to_owned()],
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    };
    let solved = DenseReferenceSolverF64::default()
        .solve(&SymmetricProblemF64::new(&operator), &configuration)?;
    assert!((solved.eigenvalue - 2.0).abs() < 1e-12);
    assert!(solved.scaled_backward_error <= 1e-12);

    let matrix_digest = ContentDigest::sha256(b"exact diagonal matrix [2, 3]");
    let mut bundle = CertificateBundle {
        schema_version: 1,
        certificate_id: ContentDigest("00".repeat(32)),
        claim: CertificateClaim::PositiveDefiniteMatrix,
        assurance: AssuranceLevel::Certified,
        toolkit_version: ToolkitVersion::parse("0.13.0")?,
        citation: None,
        provenance: SolverProvenance::current_package("exact_integer_fixture"),
        inputs: Vec::new(),
        inertia: Some(InertiaCertificate {
            dimension: 2,
            positive: 2,
            negative: 0,
            zero_or_unresolved: 0,
            matrix_digest,
            scalar_backend: "exact_integer_fixture".to_owned(),
            precision_bits: 64,
            pivot_enclosures_digest: None,
        }),
        eigenvalue_enclosures: Vec::new(),
        spectral_gap: None,
        exact_records: BTreeMap::new(),
        evidence_digests: BTreeMap::new(),
        approximation_ledger: ApproximationLedger::default(),
        assumptions: vec!["the supplied diagonal entries are exact integers".to_owned()],
        notes: vec!["standalone non-domain-specific consumer".to_owned()],
    };
    bundle.refresh_certificate_id()?;
    let verification = verify_bundle(&bundle);
    assert!(verification.valid, "{verification:?}");

    #[cfg(feature = "hp")]
    verify_portable_exact_certificate()?;
    Ok(())
}

#[cfg(feature = "hp")]
fn verify_portable_exact_certificate() -> Result<(), Box<dyn std::error::Error>> {
    use rug::Rational;
    use xc_certify::exact::{
        build_portable_interval_inertia_certificate, verify_portable_interval_inertia_certificate,
    };
    use xc_numerics::interval::RationalInterval;

    let exact = |numerator| RationalInterval::point(Rational::from(numerator));
    let matrix = vec![exact(2), exact(0), exact(0), exact(3)];
    let certificate = build_portable_interval_inertia_certificate(
        &matrix,
        2,
        256,
        "exact_rational",
        ContentDigest::sha256(b"external exact assembly evidence"),
        BTreeMap::from([("problem".to_owned(), "positive diagonal".to_owned())]),
        vec!["finite two-dimensional claim".to_owned()],
    )?;
    let report = verify_portable_interval_inertia_certificate(&certificate);
    assert!(report.valid, "{report:?}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    public_solver_and_certificate_workflow()
}

#[cfg(test)]
mod tests {
    #[test]
    fn consumes_public_solver_and_certification_apis() {
        super::public_solver_and_certificate_workflow().unwrap();
    }
}
