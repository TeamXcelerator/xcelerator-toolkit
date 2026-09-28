#![cfg(feature = "hp")]
use rug::{Float, Integer, Rational};
use xc_spectral::ccm::research::{
    analyze_nested_gram_schur_hp, analyze_nested_schur_hp, analyze_root_transfer_hp,
};
fn f(v: i32) -> Float {
    Float::with_val(192, v)
}
fn parsed(value: &str) -> Float {
    Float::with_val(64, Float::parse(value).unwrap())
}
#[test]
fn exhaustive_research_point_secular_sum_keeps_cancelled_signal() {
    let big = f(1) << 200u32;
    let r = analyze_root_transfer_hp(
        &[big.clone() * 3, f(2), -big],
        &[f(0), f(1), f(2)],
        &f(3),
        None,
        64,
    )
    .unwrap();
    assert_eq!(parsed(&r.function_value), Float::with_val(64, 1));
    assert_ne!(parsed(&r.predicted_displacement), 0);
}
#[test]
fn exhaustive_research_point_preserves_high_precision_pole_distance() {
    let pole = f(1) << 100u32;
    let point = pole.clone() + 1;
    let r =
        analyze_root_transfer_hp(&[f(1)], std::slice::from_ref(&pole), &point, None, 64).unwrap();
    assert_eq!(parsed(&r.nearest_pole_distance), 1);
    assert_eq!(parsed(&r.function_value), 1);
}
#[test]
fn exhaustive_research_point_pole_crossing_uses_exact_predicted_target() {
    let delta = Float::with_val(192, Rational::from((1, Integer::from(1) << 100)));
    let r = analyze_root_transfer_hp(
        &[f(4), Float::with_val(192, -1.5) - delta],
        &[f(0), f(1)],
        &f(2),
        None,
        64,
    )
    .unwrap();
    assert!(!r.predicted_step_crosses_pole);
}
#[test]
fn exhaustive_research_point_observed_displacement_preserves_low_bits() {
    let old = f(1) << 100u32;
    let target = old.clone() + 1;
    let r = analyze_root_transfer_hp(&[f(1)], &[f(0)], &old, Some(&target), 64).unwrap();
    assert_eq!(parsed(r.observed_displacement.as_ref().unwrap()), 1);
}
#[test]
fn exhaustive_research_point_prefix_decision_preserves_high_precision_defect() {
    let delta = Float::with_val(192, Rational::from((1, Integer::from(1) << 100)));
    let r = analyze_nested_schur_hp(
        &[f(1)],
        &[f(1) + &delta, f(0), f(0), f(2)],
        1,
        &f(0),
        &f(0),
        64,
    )
    .unwrap();
    assert!(!r.prefix_within_tolerance);
    assert_eq!(
        parsed(&r.prefix_maximum_absolute_defect),
        Float::with_val(64, delta)
    );
}
#[test]
fn exhaustive_research_point_generalized_pencil_preserves_exact_cancellation() {
    let a = f(1) << 100u32;
    let g = a.clone() + 1;
    let r = analyze_nested_gram_schur_hp(
        std::slice::from_ref(&a),
        std::slice::from_ref(&g),
        &[a.clone(), f(0), f(0), f(3)],
        &[g.clone(), f(0), f(0), f(1)],
        1,
        &f(1),
        &f(0),
        64,
    )
    .unwrap();
    assert_eq!(parsed(&r.schur_complement), 2);
}

#[test]
fn exact_schur_matches_known_solutions_across_precision_and_scale() {
    for p in [64, 128, 256] {
        for n in 1..=6usize {
            for exponent in [-1000i32, 0, 1000] {
                let scale = if exponent >= 0 {
                    Rational::from(Integer::from(1) << exponent as u32)
                } else {
                    Rational::from((1, Integer::from(1) << (-exponent) as u32))
                };
                let a = (0..n * n)
                    .map(|k| Rational::from(if k / n == k % n { k / n + 2 } else { 1 }) * &scale)
                    .collect::<Vec<_>>();
                let x = (0..n)
                    .map(|j| Rational::from((j as i32 - 2, 8)))
                    .collect::<Vec<_>>();
                let b = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| a[i * n + j].clone() * &x[j])
                            .sum::<Rational>()
                    })
                    .collect::<Vec<_>>();
                let expected = Rational::from((7, 8)) * &scale;
                let last = b
                    .iter()
                    .zip(&x)
                    .map(|(b, x)| b.clone() * x)
                    .sum::<Rational>()
                    + &expected;
                let mut large = vec![Float::with_val(p, 0); (n + 1) * (n + 1)];
                for i in 0..n {
                    for j in 0..n {
                        large[i * (n + 1) + j] = Float::with_val(p, &a[i * n + j]);
                    }
                    large[i * (n + 1) + n] = Float::with_val(p, &b[i]);
                    large[n * (n + 1) + i] = Float::with_val(p, &b[i]);
                }
                large[n * (n + 1) + n] = Float::with_val(p, last);
                let small = a.iter().map(|v| Float::with_val(p, v)).collect::<Vec<_>>();
                let zero = Float::with_val(p, 0);
                let report = analyze_nested_schur_hp(&small, &large, n, &zero, &zero, p).unwrap();
                let parse = |s: &str| Float::with_val(p, Float::parse(s).unwrap());
                assert_eq!(
                    parse(&report.schur_complement),
                    Float::with_val(p, expected),
                    "p={p},n={n},e={exponent}"
                );
                assert_eq!(parse(&report.solve_relative_residual), 0);
                assert!(report.prefix_within_tolerance);
                let norm = b.iter().map(|x| x.clone() * x).sum::<Rational>();
                assert_eq!(parse(&report.border_norm), Float::with_val(p, norm).sqrt());
            }
        }
    }
}
#[test]
fn exact_transfer_matches_symmetric_closed_forms_at_multiple_scales() {
    for p in [64, 128, 256] {
        for exponent in [-1000i32, 0, 1000] {
            for numerator in [-3, -1, 0, 1, 3] {
                let a = Float::with_val(p, 1) << exponent;
                let t = Float::with_val(p, numerator) * &a / 4u32;
                let q_a = a.to_rational().unwrap();
                let q_t = t.to_rational().unwrap();
                let a2 = q_a.clone() * &q_a;
                let t2 = q_t.clone() * &q_t;
                let expected = q_t.clone() * (t2.clone() - &a2) / (t2 + &a2);
                let report = analyze_root_transfer_hp(
                    &[Float::with_val(p, 1), Float::with_val(p, 1)],
                    &[-a.clone(), a],
                    &t,
                    Some(&Float::with_val(p, 0)),
                    p,
                )
                .unwrap();
                let parse = |s: &str| Float::with_val(p, Float::parse(s).unwrap());
                assert_eq!(
                    parse(&report.predicted_displacement),
                    Float::with_val(p, expected)
                );
                assert_eq!(parse(report.observed_displacement.as_ref().unwrap()), -t);
                assert_eq!(
                    parse(report.supplied_target_relative_residual.as_ref().unwrap()),
                    0
                );
                assert!(!report.predicted_step_crosses_pole);
            }
        }
    }
}
#[test]
fn exact_research_singular_budget_and_invalid_inputs_fail_explicitly() {
    assert!(
        analyze_nested_schur_hp(&[f(0)], &[f(0), f(1), f(1), f(2)], 1, &f(0), &f(0), 64).is_err()
    );
    assert!(
        analyze_nested_schur_hp(&[f(1)], &[f(1), f(0), f(0), f(2)], 1, &f(0), &f(-1), 64).is_err()
    );
    assert!(analyze_root_transfer_hp(&[f(1)], &[f(0)], &f(0), None, 64).is_err());
    assert!(analyze_root_transfer_hp(&[f(0)], &[f(0)], &f(1), None, 64).is_err());
    assert!(
        analyze_root_transfer_hp(&[f(1)], &[f(0)], &(f(1) << 100_000_000u32), None, 64).is_err()
    );
    for p in [0, 63, 1_000_001, u32::MAX] {
        assert!(analyze_root_transfer_hp(&[f(1)], &[f(0)], &f(1), None, p).is_err());
    }
    // Invertible prefix requiring a row interchange: A=[[0,1],[1,2]],
    // x=[1,2], b=[2,5], d=15 gives Schur=15-(2+10)=3.
    let r = analyze_nested_schur_hp(
        &[f(0), f(1), f(1), f(2)],
        &[f(0), f(1), f(2), f(1), f(2), f(5), f(2), f(5), f(15)],
        2,
        &f(0),
        &f(0),
        64,
    )
    .unwrap();
    assert_eq!(parsed(&r.schur_complement), 3);
}
