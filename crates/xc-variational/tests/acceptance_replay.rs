#![cfg(feature = "hp")]
use xc_core::DecimalLiteral as D;
use xc_variational::maynard::*;

fn three_options() -> MkThreeRouteAcceptanceOptions {
    MkThreeRouteAcceptanceOptions {
        k: 2,
        degree: 2,
        precision_bits: 256,
        initial_precision_bits: 64,
        absolute_residual_tolerance: D::new("1e-45").unwrap(),
        scaled_backward_error_tolerance: D::new("1e-45").unwrap(),
        ritz_value_stability_tolerance: D::new("1e-45").unwrap(),
        eigenvalue_agreement_tolerance: D::new("1e-35").unwrap(),
        overlap_tolerance: D::new("1e-35").unwrap(),
        candidate_quotient_agreement_tolerance: D::new("1e-35").unwrap(),
        maximum_iterations: 5000,
    }
}

#[test]
fn three_route_acceptance_replays_values_instead_of_trusting_error_fields() {
    let options = three_options();
    let record = run_mk_three_route_acceptance(&options).unwrap();
    let mut bad = record.clone();
    bad.matrix_free_eigenvalue = "999".into();
    bad.eigenvalue_absolute_difference = "0".into();
    bad.candidate_absolute_difference = "0".into();
    assert!(
        verify_mk_three_route_acceptance(&bad, &options).is_err(),
        "fabricated scalar accepted"
    );
    for field in [
        "dense_eigenvalue",
        "matrix_free_residual_norm",
        "dense_residual_norm",
        "candidate_absolute_difference",
        "one_minus_metric_overlap_squared",
    ] {
        let mut value = serde_json::to_value(&record).unwrap();
        value[field] = "-1".into();
        let bad = serde_json::from_value(value).unwrap();
        assert!(
            verify_mk_three_route_acceptance(&bad, &options).is_err(),
            "invalid {field} accepted"
        );
    }
}

#[test]
fn scale_acceptance_recomputes_quotient_difference() {
    let options = MkScaleAcceptanceOptions {
        historical_dense_degree_limit: 3,
        target_degree: 4,
        precision_bits: 192,
        minimum_exact_lower_bound: xc_certify::ExactRationalRecord {
            numerator: "2".into(),
            denominator: "1".into(),
        },
        quotient_agreement_tolerance: D::new("1e-45").unwrap(),
    };
    let mut record = run_mk_scale_acceptance(&options).unwrap();
    record.matrix_free_quotient = "999".into();
    record.quotient_absolute_difference = "0".into();
    assert!(verify_mk_scale_acceptance(&record, &options).is_err());
}

#[test]
fn prolongation_rejects_duplicate_bases_and_nonfinite_coefficients() {
    let constant = IntegerPartition(vec![]);
    assert!(prolong_symmetric_warm_start(
        &[constant.clone(), constant.clone()],
        &[1.0, 2.0],
        std::slice::from_ref(&constant),
        0,
        1
    )
    .is_err());
    assert!(prolong_symmetric_warm_start(
        std::slice::from_ref(&constant),
        &[1.0],
        &[constant.clone(), constant.clone()],
        0,
        1
    )
    .is_err());
    assert!(prolong_symmetric_warm_start(
        std::slice::from_ref(&constant),
        &[f64::NAN],
        std::slice::from_ref(&constant),
        0,
        1
    )
    .is_err());
}

#[test]
fn exact_constant_space_checks_and_mutated_state_evidence() {
    // On the k-simplex, constant F has I=1/k!, sum J_i=2k/(k+1)!,
    // hence the one-dimensional quotient is exactly 2k/(k+1).
    for k in 1..=4 {
        let mut options = three_options();
        options.k = k;
        options.degree = 0;
        options.initial_precision_bits = 128;
        options.absolute_residual_tolerance = D::new("1e-30").unwrap();
        options.scaled_backward_error_tolerance = D::new("1e-30").unwrap();
        options.ritz_value_stability_tolerance = D::new("1e-30").unwrap();
        options.eigenvalue_agreement_tolerance = D::new("1e-30").unwrap();
        options.candidate_quotient_agreement_tolerance = D::new("1e-30").unwrap();
        let record = run_mk_three_route_acceptance(&options).unwrap();
        let q = &record.candidate_certificate.quotient;
        let actual = rug::Rational::from((
            rug::Integer::from_str_radix(&q.numerator, 10).unwrap(),
            rug::Integer::from_str_radix(&q.denominator, 10).unwrap(),
        ));
        assert_eq!(actual, rug::Rational::from((2 * k, k + 1)));
        assert_eq!(record.schema_version, 3);
        // Initial convergence is followed by real repetition at the declared
        // source precision before exact-source acceptance.
        assert_eq!(
            record.adaptive_attempt_precisions,
            vec![options.initial_precision_bits, options.precision_bits]
        );
        assert_eq!(record.precision_bits, options.precision_bits);
        verify_mk_three_route_acceptance(&record, &options).unwrap();
        for field in ["dense_coefficients", "adaptive_attempt_precisions"] {
            let mut value = serde_json::to_value(&record).unwrap();
            value[field] = serde_json::json!([]);
            assert!(verify_mk_three_route_acceptance(
                &serde_json::from_value(value).unwrap(),
                &options
            )
            .is_err());
        }
    }
    let options = three_options();
    let record = run_mk_three_route_acceptance(&options).unwrap();
    let mut altered = record.clone();
    altered.dense_coefficients[0].numerator = "999".into();
    assert!(verify_mk_three_route_acceptance(&altered, &options).is_err());
    for field in ["matrix_free_residual_norm", "dense_residual_norm"] {
        let mut value = serde_json::to_value(&record).unwrap();
        value[field] = "0".into();
        assert!(verify_mk_three_route_acceptance(
            &serde_json::from_value(value).unwrap(),
            &options
        )
        .is_err());
    }
    let mut legacy = record.clone();
    legacy.schema_version = 1;
    assert!(verify_mk_three_route_acceptance(&legacy, &options).is_err());
    for p in [0, 32, 64, u32::MAX] {
        let mut bad = options.clone();
        bad.precision_bits = p;
        assert!(verify_mk_three_route_acceptance(&record, &bad).is_err());
    }
    let mut bad = options.clone();
    bad.overlap_tolerance = D::new("-1").unwrap();
    assert!(verify_mk_three_route_acceptance(&record, &bad).is_err());
}

#[test]
fn valid_prolongation_preserves_polynomial_coordinates_under_reordering() {
    let parent = [IntegerPartition(vec![]), IntegerPartition(vec![1])];
    let child = [
        IntegerPartition(vec![2]),
        IntegerPartition(vec![1]),
        IntegerPartition(vec![]),
    ];
    let result = prolong_symmetric_warm_start(&parent, &[2.0, 3.0], &child, 0, 1).unwrap();
    assert_eq!(result.coefficients, vec![0.0, 3.0, 2.0]);
    for x in [-2.0, 0.0, 0.5, 3.0] {
        assert_eq!(
            2.0 + 3.0 * x,
            result.coefficients[0] * x * x + result.coefficients[1] * x + result.coefficients[2]
        );
    }
    let malformed = [IntegerPartition(vec![0])];
    assert!(prolong_symmetric_warm_start(&malformed, &[1.0], &malformed, 0, 1).is_err());
}
