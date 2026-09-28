#![cfg(feature = "hp")]
use rug::{ops::Pow, Integer, Rational};
use xc_core::DecimalLiteral;
use xc_variational::maynard::{
    run_mk_three_route_acceptance, verify_mk_three_route_acceptance, MkThreeRouteAcceptanceOptions,
};
fn options(k: usize, degree: usize, maximum_iterations: usize) -> MkThreeRouteAcceptanceOptions {
    let lit = |s| DecimalLiteral::new(s).unwrap();
    MkThreeRouteAcceptanceOptions {
        k,
        degree,
        precision_bits: 256,
        initial_precision_bits: 64,
        absolute_residual_tolerance: lit("1e-45"),
        scaled_backward_error_tolerance: lit("1e-45"),
        ritz_value_stability_tolerance: lit("1e-45"),
        eigenvalue_agreement_tolerance: lit("1e-35"),
        overlap_tolerance: lit("1e-35"),
        candidate_quotient_agreement_tolerance: lit("1e-35"),
        maximum_iterations,
    }
}
#[test]
fn maynard_k1_large_iteration_budget_has_bounded_exact_records() {
    // Cauchy-Schwarz gives (integral F)^2 <= integral F^2, with equality
    // for a constant: the exact maximum is1 in every k=1 polynomial space.
    for degree in 1..=4 {
        for maximum_iterations in [400, 20000] {
            let cfg = options(1, degree, maximum_iterations);
            let record = run_mk_three_route_acceptance(&cfg).unwrap();
            verify_mk_three_route_acceptance(&record, &cfg).unwrap();
            // Integrate the recorded univariate polynomial independently:
            // integral t^j=1/(j+1), so this oracle does not call Toolkit forms.
            let coefficients: Vec<Rational> = record
                .candidate_coefficients
                .iter()
                .map(|c| {
                    Rational::from((
                        Integer::from_str_radix(&c.numerator, 10).unwrap(),
                        Integer::from_str_radix(&c.denominator, 10).unwrap(),
                    ))
                })
                .collect();
            let mut integral = Rational::new();
            let mut norm = Rational::new();
            for (i, a) in coefficients.iter().enumerate() {
                integral += a.clone() / (i + 1);
                for (j, b) in coefficients.iter().enumerate() {
                    norm += Rational::from(a * b) / (i + j + 1);
                }
            }
            let quotient = Rational::from(&integral * &integral) / norm;
            let encoded = Rational::from((
                Integer::from_str_radix(&record.candidate_certificate.quotient.numerator, 10)
                    .unwrap(),
                Integer::from_str_radix(&record.candidate_certificate.quotient.denominator, 10)
                    .unwrap(),
            ));
            assert_eq!(quotient, encoded);
            let gap = Rational::from(1) - quotient;
            assert!(
                gap >= 0 && gap <= Rational::from((Integer::from(1), Integer::from(10).pow(70)))
            );
            assert_eq!(record.adaptive_attempt_precisions.last(), Some(&256));
            let encoded = serde_json::to_vec(&record).unwrap();
            assert!(
                encoded.len() < 65536,
                "degree{degree}, maximum_iterations{maximum_iterations}: {} bytes",
                encoded.len()
            );
        }
    }
}
#[test]
fn maynard_small_spaces_reach_the_declared_source_precision() {
    for degree in [0, 1] {
        let cfg = options(2, degree, 1000);
        let record = run_mk_three_route_acceptance(&cfg).unwrap();
        verify_mk_three_route_acceptance(&record, &cfg).unwrap();
        assert_eq!(record.adaptive_attempt_precisions.last(), Some(&256));
        // Constant k=2 source forms integrate exactly to quotient4/3.
        if degree == 0 {
            assert_eq!(record.candidate_certificate.quotient.numerator, "4");
            assert_eq!(record.candidate_certificate.quotient.denominator, "3");
        }
    }
}
