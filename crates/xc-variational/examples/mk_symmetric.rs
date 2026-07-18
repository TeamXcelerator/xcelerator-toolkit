#[cfg(feature = "hp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rug::Rational;
    use xc_core::EigenTarget;
    use xc_operator::GeneralizedEigenProblem;
    use xc_solver::{GeneralizedExtremeConfigF64, MatrixFreeLobpcgF64};
    use xc_variational::maynard::{
        MkSymmetricIMetricF64, MkSymmetricJOperatorF64, MkSymmetricReference,
    };

    let reference = MkSymmetricReference::new(3, 3)?;
    let operator = MkSymmetricJOperatorF64::new(&reference)?;
    let metric = MkSymmetricIMetricF64::new(&reference)?;
    let problem = GeneralizedEigenProblem::new(&operator, &metric)?;
    let candidate = MatrixFreeLobpcgF64.solve(
        &problem,
        &GeneralizedExtremeConfigF64 {
            target: EigenTarget::AlgebraicLargest,
            absolute_residual_tolerance: 1e-11,
            scaled_backward_error_tolerance: 1e-11,
            ritz_value_stability_tolerance: 1e-12,
            maximum_iterations: 500,
            minimum_iterations: 2,
        },
    )?;

    // Discovery is f64, but the published lower-bound candidate is evaluated
    // exactly after explicit rationalization in the declared symmetric space.
    let exact_coefficients: Vec<Rational> = candidate
        .eigenvector
        .iter()
        .map(|value| {
            let numerator = (value * 1_000_000_000_000.0).round() as i64;
            Rational::from((numerator, 1_000_000_000_000i64))
        })
        .collect();
    let certificate = reference.certificate(&exact_coefficients)?;
    println!(
        "symmetric Mk discovery: k={}, degree={}, dimension={}, lambda~{:.15}; exact lower bound={}/{}",
        reference.k(),
        reference.degree(),
        reference.dimension(),
        candidate.eigenvalue,
        certificate.quotient.numerator,
        certificate.quotient.denominator,
    );
    Ok(())
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("run with: cargo run -p xc-variational --example mk_symmetric --features hp");
}
