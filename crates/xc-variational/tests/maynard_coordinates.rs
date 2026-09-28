#![cfg(feature = "hp")]
use xc_core::DecimalLiteral;
use xc_variational::maynard::*;
#[test]
fn difficult_small_spaces_converge_after_exact_i_coordinate_change() {
    let d = |s| DecimalLiteral::new(s).unwrap();
    // Independent characteristic-polynomial reference maxima for these spaces.
    for (k, degree, maximum) in [
        (2, 3, "1.3859093264936133"),
        (2, 4, "1.3859309471589758"),
        (3, 3, "1.6460312569616813"),
    ] {
        let options = MkThreeRouteAcceptanceOptions {
            k,
            degree,
            precision_bits: 192,
            initial_precision_bits: 192,
            absolute_residual_tolerance: d("1e-35"),
            scaled_backward_error_tolerance: d("1e-35"),
            ritz_value_stability_tolerance: d("1e-35"),
            eigenvalue_agreement_tolerance: d("1e-25"),
            overlap_tolerance: d("1e-25"),
            candidate_quotient_agreement_tolerance: d("1e-25"),
            maximum_iterations: 3000,
        };
        let record = run_mk_three_route_acceptance(&options).unwrap();
        assert_eq!(record.schema_version, 3);
        assert_eq!(
            record.matrix_free_route,
            "adaptive_exact_i_orthogonal_streamed_j_fresh_images_bounded_records_hp_v4"
        );
        verify_mk_three_route_acceptance(&record, &options).unwrap();
        assert_eq!(
            record.adaptive_attempt_precisions.last(),
            Some(&options.precision_bits)
        );
        let exact = &record.candidate_certificate.quotient;
        let quotient = rug::Rational::from((
            rug::Integer::from_str_radix(&exact.numerator, 10).unwrap(),
            rug::Integer::from_str_radix(&exact.denominator, 10).unwrap(),
        ));
        let independent = rug::Float::with_val(512, rug::Float::parse(maximum).unwrap());
        let error = (rug::Float::with_val(512, quotient) - independent).abs();
        assert!(error < rug::Float::with_val(512, rug::Float::parse("1e-12").unwrap()));

        let bytes = serde_json::to_vec(&record).unwrap();
        let replay: MkThreeRouteAcceptanceRecord = serde_json::from_slice(&bytes).unwrap();
        verify_mk_three_route_acceptance(&replay, &options).unwrap();
    }
}
