use super::*;
use rug::Rational;

// Independent exact Euler-Maclaurin identity: trapezoid for u^2 on [1,2]
// equals 7/3 + 1/(6 Q^2). The chosen Q values straddle the 1e-8 policy.
fn fixture(q: usize) -> (PortableDistanceResolutionEvidence, PortableTargetDistance) {
    let p = 192;
    let rule = |steps| WeightedIntegrationRule::UniformGrid {
        scheme: UniformGridScheme::Trapezoid,
        variable: GridVariable::U,
        steps,
    };
    let exact = |steps: usize| Rational::from((14 * steps * steps + 1, 6 * steps * steps));
    let values = [q, 2 * q, 4 * q].map(|steps| {
        let result = weighted_alpha_norm(
            |u: &Float| u.clone().square(),
            &Float::with_val(p, 2),
            &Float::with_val(p, 0),
            rule(steps),
            p,
        )
        .unwrap();
        let expected = Float::with_val(p, exact(steps));
        assert!(
            (Float::with_val(p, &result.value - expected)).abs() < (Float::with_val(p, 1) >> 170)
        );
        result.value
    });
    let mut relative = exact(q) - exact(2 * q);
    relative /= exact(2 * q);
    let base_met = relative <= Rational::from((1, 100_000_000));
    let mut final_relative = exact(2 * q) - exact(4 * q);
    final_relative /= exact(4 * q);
    assert!(final_relative <= Rational::from((1, 100_000_000)));
    let (absolute, relative) = absolute_and_relative_difference(&values[0], &values[1], p);
    let (last_absolute, last_relative) = if base_met {
        (absolute.clone(), relative.clone())
    } else {
        absolute_and_relative_difference(&values[1], &values[2], p)
    };
    let entry = PortableRuleResolutionEvidence {
        rule_family: "uniform_grid".into(),
        quadrature_rule: rule(q).rule().into(),
        grid_variable: rule(q).variable().as_str().into(),
        base_resolution: q,
        base_distance: retained_decimal(&values[0], p),
        twice_resolution: 2 * q,
        twice_distance: retained_decimal(&values[1], p),
        q_to_2q_absolute_difference: retained_decimal(&absolute, p),
        q_to_2q_relative_difference: retained_decimal(&relative, p),
        four_times_resolution: (!base_met).then_some(4 * q),
        four_times_distance: (!base_met).then(|| retained_decimal(&values[2], p)),
        final_absolute_difference: retained_decimal(&last_absolute, p),
        final_relative_difference: retained_decimal(&last_relative, p),
        final_resolution: if base_met { 2 * q } else { 4 * q },
        tolerance_met: true,
    };
    let target_digest = "a".repeat(64);
    let evidence = PortableDistanceResolutionEvidence {
        schema_version: 2,
        target_definition_digest: target_digest.clone(),
        lambda_squared: "4".into(),
        n_modes: 0,
        precision_bits: p,
        alpha: alpha_identity(&Float::with_val(p, 0), p),
        normalization: "f(1)=1".into(),
        coefficient_count: 1,
        coefficient_tail: coefficient_tail_evidence(&[Float::with_val(p, 1)], p).unwrap(),
        refinement_factor: 2,
        maximum_refinement_multiplier: 4,
        relative_tolerance: "1e-8".into(),
        relative_difference_denominator: "absolute_finer_distance".into(),
        zero_denominator_fallback: "absolute_difference".into(),
        refinements: vec![entry.clone()],
    };
    validate_portable_distance_resolution_evidence(
        &evidence,
        &target_digest,
        "4",
        0,
        p,
        &Float::with_val(p, 0),
        &[rule(q)],
    )
    .unwrap();
    let distance = PortableTargetDistance {
        schema_version: 2,
        target_definition_digest: target_digest,
        lambda_squared: "4".into(),
        n_modes: 0,
        precision_bits: p,
        alpha: evidence.alpha.clone(),
        eigenvalue: "0".into(),
        measurements: vec![PortableRuleMeasurement {
            rule_family: entry.rule_family,
            quadrature_rule: entry.quadrature_rule,
            grid_variable: entry.grid_variable,
            resolution: q,
            distance_to_target: entry.base_distance,
            eigenfunction_norm: "1".into(),
        }],
    };
    (evidence, distance)
}

#[test]
fn later_ladder_agreement_does_not_qualify_returned_q() {
    let (evidence, distance) = fixture(2000);
    assert_eq!(evidence.refinements[0].four_times_resolution, Some(8000));
    assert_eq!(resolution_ladder_verdict(&evidence), Some(true));
    assert_eq!(
        resolution_verdict(&evidence, &distance).unwrap(),
        Some(false)
    );
}

#[test]
fn original_q_agreement_remains_a_positive_control() {
    let (evidence, distance) = fixture(3000);
    assert_eq!(evidence.refinements[0].four_times_resolution, None);
    assert_eq!(resolution_ladder_verdict(&evidence), Some(true));
    assert_eq!(
        resolution_verdict(&evidence, &distance).unwrap(),
        Some(true)
    );
}

#[test]
fn verdict_rejects_substituted_returned_measurement_or_convention() {
    let (evidence, mut distance) = fixture(3000);
    distance.measurements[0].distance_to_target = "2".into();
    assert!(resolution_verdict(&evidence, &distance).is_err());
    distance.measurements[0].distance_to_target = evidence.refinements[0].base_distance.clone();
    distance.measurements[0].resolution += 1;
    assert!(resolution_verdict(&evidence, &distance).is_err());
}
