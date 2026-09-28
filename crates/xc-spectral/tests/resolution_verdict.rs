#![cfg(feature = "hp")]
include!("fixtures/resolution_validator/extracted-resolution-validator.rs");

fn check_verdict(precision: u32, base_distance: &str, supplied_difference: &str) {
    // Access the private validator without changing production source. Ensure
    // the mechanically extracted bodies are still byte-for-byte source matches.
    let original = include_str!("../src/distance.rs").replace("\r\n", "\n");
    let bodies: Vec<String> = serde_json::from_str(include_str!(
        "fixtures/resolution_validator/extracted-resolution-validator.json"
    ))
    .unwrap();
    for body in bodies {
        assert!(
            original.contains(&body),
            "extracted snapshot must match production"
        );
    }
    use xc_numerics::grid_integral::{GridVariable, UniformGridScheme};
    let alpha = Float::with_val(precision, 0);
    let rule = WeightedIntegrationRule::UniformGrid {
        scheme: UniformGridScheme::Trapezoid,
        variable: GridVariable::U,
        steps: 2,
    };
    let coefficient_tail = [15, 30, 45].map(|n| {
        serde_json::json!({
        "threshold":format!("1e-{n}"),"effective_bandwidth":null,
        "discarded_one_sided_l1":"0","discarded_cosine_pointwise_bound":"0",
        "discarded_cosine_l2":"0"})
    });
    let value = serde_json::json!({
        "schema_version":2,"target_definition_digest":"audit_target","lambda_squared":"4",
        "n_modes":0,"precision_bits":precision,"alpha":alpha_identity(&alpha,precision),
        "normalization":"f(1)=1","coefficient_count":1,
        "coefficient_tail":coefficient_tail,
        "refinement_factor":2,"maximum_refinement_multiplier":4,
        "relative_tolerance":"1e-8","relative_difference_denominator":"absolute_finer_distance",
        "zero_denominator_fallback":"absolute_difference",
        "refinements":[{"rule_family":rule.family(),"quadrature_rule":rule.rule(),
            "grid_variable":rule.variable().as_str(),"base_resolution":2,
            "base_distance":base_distance,"twice_resolution":4,"twice_distance":"1",
            "q_to_2q_absolute_difference":supplied_difference,"q_to_2q_relative_difference":supplied_difference,
            "four_times_resolution":null,"four_times_distance":null,
            "final_absolute_difference":supplied_difference,"final_relative_difference":supplied_difference,
            "final_resolution":4,"tolerance_met":true}]
    });
    let artifact: PortableDistanceResolutionEvidence = serde_json::from_value(value).unwrap();
    // Independent rational comparison on the actual stored binary points.
    // No validator tolerance or discrepancy implementation is used here.
    let exact_discrepancy = Float::with_val(precision, Float::parse(base_distance).unwrap())
        .to_rational()
        .unwrap()
        - rug::Rational::from(1);
    let claimed_discrepancy =
        Float::with_val(precision, Float::parse(supplied_difference).unwrap())
            .to_rational()
            .unwrap();
    assert!(exact_discrepancy * 100_000_000u32 > 1);
    assert!(claimed_discrepancy * 100_000_000u32 < 1);
    let outcome = validate_portable_distance_resolution_evidence(
        &artifact,
        "audit_target",
        "4",
        0,
        precision,
        &alpha,
        &[rule],
    );
    eprintln!("p={precision}; exact retained discrepancy > 1e-8 but claimed discrepancy < 1e-8; forged verdict validation={outcome:?}");
    assert!(
        outcome.is_err(),
        "false difference and false tolerance verdict were accepted"
    );
}

#[test]
fn validator_internal_p32_verdict_must_follow_replayed_measurements() {
    check_verdict(32, "1.00000095367431640625", "0");
}

#[test]
fn current_capture_p64_verdict_must_follow_replayed_measurements() {
    check_verdict(64, "1.0000000100000005", "0.0000000099999995");
}

#[test]
fn resolution_decision_matches_independent_exact_point_comparisons() {
    for (left, right) in [
        ("0", "0"),
        ("0.000000001", "0"),
        ("0.0000001", "0"),
        ("1", "1"),
        ("1.0000000099999995", "1"),
        ("1.0000000100000005", "1"),
        ("0.9999999900000005", "1"),
        ("0.9999999899999995", "1"),
        ("0", "1"),
    ] {
        for p in [64, 128, 256] {
            let a = Float::with_val(p, Float::parse(left).unwrap());
            let b = Float::with_val(p, Float::parse(right).unwrap());
            let ar = a.to_rational().unwrap();
            let br = b.to_rational().unwrap();
            let denominator = if br == 0 {
                rug::Rational::from(1)
            } else {
                br.clone().abs()
            };
            let exact = (ar - br).abs() * 100_000_000u32 <= denominator;
            assert_eq!(
                resolution_tolerance_met(&a, &b, p),
                exact,
                "p={p}, {left}, {right}"
            );
        }
    }
}
