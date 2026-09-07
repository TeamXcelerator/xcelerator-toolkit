#![cfg(feature = "hp")]

use rug::{float::Special, ops::Pow, Float, Integer, Rational};
use xc_numerics::prefix::{
    analyze_prefixes, analyze_prefixes_with_policy, checked_decimal_export, PrefixDiagnosticPolicy,
    TwoModeMomentStatus,
};

// Independent exact Gauss-Jordan inverse, used only for small test fixtures.
fn inverse(a: &[Rational], n: usize) -> Vec<Rational> {
    let mut v = vec![Rational::from(0); 2 * n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * 2 * n + j] = a[i * n + j].clone();
        }
        v[i * 2 * n + n + i] = Rational::from(1);
    }
    for j in 0..n {
        let selected = (j..n).find(|&i| v[i * 2 * n + j] != 0).unwrap();
        for k in 0..2 * n {
            v.swap(j * 2 * n + k, selected * 2 * n + k);
        }
        let pivot = v[j * 2 * n + j].clone();
        for k in 0..2 * n {
            v[j * 2 * n + k] /= &pivot;
        }
        for i in 0..n {
            if i != j {
                let factor = v[i * 2 * n + j].clone();
                for k in 0..2 * n {
                    let mut correction = v[j * 2 * n + k].clone();
                    correction *= &factor;
                    v[i * 2 * n + k] -= correction;
                }
            }
        }
    }
    (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| v[i * 2 * n + n + j].clone())
        .collect()
}
fn parse(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn compare(s: &str, exact: &Rational) {
    let p = 512;
    let reference = Float::with_val(p, exact);
    let mut error = parse(s, p);
    error -= &reference;
    error.abs_mut();
    let mut scale = reference.abs();
    if scale.is_zero() {
        scale = Float::with_val(p, 1);
    }
    error /= scale;
    assert!(
        error < Float::with_val(p, Float::parse("1e-100").unwrap()),
        "{s} vs {exact}: {error}"
    );
}
fn check_exact(a: Vec<Rational>, n: usize) {
    let p = 512;
    let hp: Vec<Float> = a.iter().map(|v| Float::with_val(p, v)).collect();
    let before = hp.clone();
    let report = analyze_prefixes(&hp, n, p, 32, &(1..=n).collect::<Vec<_>>()).unwrap();
    assert!(report.stopped.is_none());
    assert_eq!(report.rows.len(), n);
    assert_eq!(hp, before, "the retained input matrix must not be mutated");
    let full =
        analyze_prefixes_with_policy(&hp, n, p, 32, &[], &PrefixDiagnosticPolicy::full()).unwrap();
    let mut previous_trace = Rational::from(0);
    for k in 1..=n {
        let prefix: Vec<Rational> = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].clone())
            .collect();
        let inv = inverse(&prefix, k);
        let mut trace = Rational::from(0);
        let mut trace2 = Rational::from(0);
        let mut trace3 = Rational::from(0);
        for i in 0..k {
            trace += &inv[i * k + i];
        }
        for x in &inv {
            let mut square = x.clone();
            square *= x;
            trace2 += square;
        }
        // Independent exact trace(A^{-3}), not a replay of the Gram recurrence.
        for i in 0..k {
            for j in 0..k {
                for l in 0..k {
                    trace3 += inv[i * k + j].clone() * &inv[j * k + l] * &inv[l * k + i];
                }
            }
        }
        compare(
            &full.rows[k - 1]
                .third_inverse_moment
                .as_ref()
                .unwrap()
                .inverse_cube_trace,
            &trace3,
        );
        let mut sigma = Rational::from(1);
        sigma /= &inv[k * k - 1];
        let innovation: Vec<Rational> = (0..k)
            .map(|i| {
                let mut x = inv[i * k + k - 1].clone();
                x *= &sigma;
                x
            })
            .collect();
        let mut mass = Rational::from(0);
        for x in &innovation {
            let mut square = x.clone();
            square *= x;
            mass += square;
        }
        let row = &report.rows[k - 1];
        compare(&row.sigma, &sigma);
        compare(&row.innovation_mass, &mass);
        compare(&row.inverse_trace, &trace);
        compare(&row.inverse_square_trace, &trace2);
        let mut increment = trace.clone();
        increment -= &previous_trace;
        compare(&row.inverse_trace_increment, &increment);
        for (s, x) in report.checkpoint_innovations[&k].iter().zip(&innovation) {
            compare(s, x);
        }
        previous_trace = trace;
    }
}

#[test]
fn single_prefix_matches_exact_inverse() {
    check_exact(vec![Rational::from((3, 7))], 1);
}
#[test]
fn diagonal_dynamic_range_matches_exact_inverse() {
    let n = 5;
    let mut a = vec![Rational::from(0); n * n];
    for i in 0..n {
        a[i * n + i] = Rational::from((Integer::from(1), Integer::from(1) << (40 * i)));
    }
    check_exact(a, n);
}
#[test]
fn coupled_matrix_matches_exact_inverse() {
    check_exact(
        [4, 12, -16, 12, 37, -43, -16, -43, 98]
            .map(Rational::from)
            .to_vec(),
        3,
    );
}
#[test]
fn hilbert_ladder_matches_exact_inverse_at_all_ten_prefixes() {
    let n = 10;
    let a = (0..n)
        .flat_map(|i| (0..n).map(move |j| Rational::from((1, (i + j + 1) as i32))))
        .collect();
    check_exact(a, n);
}
#[test]
fn near_degenerate_dyadic_gram_matches_exact_inverse() {
    let tiny = Rational::from((Integer::from(1), Integer::from(1) << 160));
    let a = vec![
        Rational::from(1),
        Rational::from(1),
        Rational::from(1),
        Rational::from(1) + tiny,
    ];
    check_exact(a, 2);
}
#[test]
fn zero_and_negative_pivots_stop_without_regularization() {
    for last in [0, -1] {
        let a = [2, 0, 0, last].map(|x| Float::with_val(128, x));
        let report = analyze_prefixes(&a, 2, 128, 32, &[1, 2]).unwrap();
        assert_eq!(report.rows.len(), 1);
        assert_eq!(report.stopped.unwrap().attempted_dimension, 2);
        assert!(!report.checkpoint_innovations.contains_key(&2));
    }
}
#[test]
fn positive_but_unresolved_pivot_is_not_silently_accepted() {
    let p = 128;
    let tiny = Float::with_val(p, 2).pow(-110);
    let a = vec![
        Float::with_val(p, 1),
        Float::with_val(p, 1),
        Float::with_val(p, 1),
        Float::with_val(p, 1) + tiny,
    ];
    let result = analyze_prefixes(&a, 2, p, 32, &[1, 2]).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.stopped.unwrap().reason,
        "insufficient_computed_pivot_margin"
    );
}
#[test]
fn malformed_input_is_rejected() {
    let p = 128;
    assert!(analyze_prefixes(&[], 0, p, 32, &[]).is_err());
    assert!(analyze_prefixes(&[Float::with_val(p, 1)], usize::MAX, p, 32, &[]).is_err());
    let asymmetric = [1, 0, 1, 2].map(|x| Float::with_val(p, x));
    assert!(analyze_prefixes(&asymmetric, 2, p, 32, &[]).is_err());
    for bad in [Special::Nan, Special::Infinity] {
        assert!(analyze_prefixes(&[Float::with_val(p, bad)], 1, p, 32, &[]).is_err());
    }
    assert!(analyze_prefixes(&[Float::with_val(256, 1)], 1, p, 32, &[]).is_err());
    assert!(analyze_prefixes(&[Float::with_val(p, 1)], 1, p, 32, &[1, 1]).is_err());
    assert!(analyze_prefixes(&[Float::with_val(p, 1)], 1, p, 32, &[2]).is_err());
}
#[test]
fn ten_serialized_runs_are_identical() {
    let a = [4, 1, 1, 2].map(|x| Float::with_val(256, x));
    let snapshot =
        || serde_json::to_vec(&analyze_prefixes(&a, 2, 256, 32, &[1, 2]).unwrap()).unwrap();
    let first = snapshot();
    for _ in 1..10 {
        assert_eq!(first, snapshot());
    }
}

#[test]
fn parallel_prefix_cross_terms_match_serial_bytes_at_odd_block_boundaries() {
    // Signed dense, strictly diagonally dominant SPD source. Odd lengths and
    // alternating signs exercise partial blocks and cancellation in dot sums.
    let n = 137;
    for p in [64, 128, 256, 512] {
        let matrix: Vec<_> = (0..n * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                let value = if i == j {
                    2048
                } else {
                    let magnitude = ((i + j) % 7 + 1) as i32;
                    if (i + j) % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                };
                Float::with_val(p, value)
            })
            .collect();
        let run = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let result =
                        analyze_prefixes(&matrix, n, p, 32, &[127, 128, 129, 137]).unwrap();
                    assert!(result.stopped.is_none());
                    serde_json::to_vec(&result).unwrap()
                })
        };
        let serial = run(1);
        for workers in [2, 4] {
            assert_eq!(serial, run(workers));
        }
    }
}
#[test]
fn generator_identity_is_checked_after_decimal_serialization() {
    let p = 1024;
    let large = Integer::from(10).pow(100);
    let base = Float::with_val(p, &large);
    let mut adjacent = base.clone();
    adjacent += 1;
    let mut expected = adjacent.clone().square();
    expected -= base.clone().square();
    let values = vec![adjacent, base];
    let check = |v: &[Float]| -> anyhow::Result<bool> {
        let mut difference = v[0].clone().square();
        difference -= v[1].clone().square();
        Ok(difference == expected)
    };
    assert!(checked_decimal_export(&values, &[80, 96], check).is_err());
    let (digits, _) = checked_decimal_export(&values, &[80, 96, 112], check).unwrap();
    assert_eq!(digits, 112);
}

#[test]
fn source_that_fails_before_export_cannot_be_rescued_by_rounding() {
    let values = [Float::with_val(128, Float::parse("1.01").unwrap())];
    assert!(checked_decimal_export(&values, &[1, 20], |v| Ok(v[0] == 1)).is_err());
}

#[test]
fn inverse_moment_estimates_bound_diagonal_minimum() {
    let a = [1, 0, 0, 4].map(|x| Float::with_val(256, x));
    let report = analyze_prefixes(&a, 2, 256, 32, &[2]).unwrap();
    let row = &report.rows[1];
    assert!(parse(&row.smallest_eigenvalue_lower_estimate, 256) <= 1);
    assert!(parse(&row.smallest_eigenvalue_upper_estimate, 256) >= 1);
    assert!(parse(&row.effective_inverse_rank, 256) >= 1);
}

#[test]
fn innovation_cancellation_records_a_zero_result_without_inventing_finite_digits() {
    // A=LL^T, L=[[1,0,0],[1,1,0],[1,1,1]]. Last innovation is [0,-1,1]:
    // its first back-substitution inner sum cancels two nonzero terms exactly.
    let a = [1, 1, 1, 1, 2, 2, 1, 2, 3].map(|v| Float::with_val(128, v));
    let report = analyze_prefixes(&a, 3, 128, 32, &[3]).unwrap();
    let c = report.rows[2].innovation_cancellation.as_ref().unwrap();
    assert_eq!(c.worst_component, Some(0));
    assert!(c.zero_result_with_nonzero_terms);
    assert_eq!(parse(&c.absolute_term_sum, 128), 2);
    assert_eq!(parse(&c.absolute_result, 128), 0);
    assert!(c.ratio.is_none() && c.decimal_digits_lost.is_none());
    assert!(report.stopped.is_none());
    let first = report.rows[0].innovation_cancellation.as_ref().unwrap();
    assert_eq!(first.worst_component, None);
    assert_eq!(parse(first.ratio.as_ref().unwrap(), 128), 1);
}

#[test]
fn two_mode_correction_is_an_estimate_and_moments_do_not_identify_a_gap() {
    let p = 256;
    let a = [1, 0, 0, 1000].map(|v| Float::with_val(p, v));
    let report = analyze_prefixes(&a, 2, p, 32, &[]).unwrap();
    let row = &report.rows[1];
    let lower = parse(&row.smallest_eigenvalue_lower_estimate, p);
    let corrected = parse(
        row.smallest_eigenvalue_second_order_estimate
            .as_ref()
            .unwrap(),
        p,
    );
    assert!((corrected - 1_i32).abs() < (lower - 1_i32).abs() / 100_i32);
    assert!(report.rows[0].gap_ratio_estimate.is_none());
    // Inverse spectra {3,2,1} and {19/7,17/7,6/7} have T1=6,T2=14,
    // but different first/second eigenvalue ratios: 2/3 and 17/19.
    let traces = |mu: &[Rational]| {
        let t1: Rational = mu.iter().cloned().sum();
        let t2: Rational = mu.iter().map(|v| v.clone() * v).sum();
        (t1, t2)
    };
    assert_eq!(
        traces(&[Rational::from(3), Rational::from(2), Rational::from(1)]),
        traces(&[
            Rational::from((19, 7)),
            Rational::from((17, 7)),
            Rational::from((6, 7))
        ])
    );
    assert_ne!(Rational::from((2, 3)), Rational::from((17, 19)));
}

#[test]
fn three_moments_fit_two_modes_but_do_not_identify_a_general_spectrum() {
    let p = 512;
    let a = [1, 0, 0, 1000].map(|v| Float::with_val(p, v));
    let report =
        analyze_prefixes_with_policy(&a, 2, p, 32, &[], &PrefixDiagnosticPolicy::full()).unwrap();
    let m = report.rows[1].third_inverse_moment.as_ref().unwrap();
    assert_eq!(m.two_mode_fit.status, TwoModeMomentStatus::Resolved);
    m.validate_for_moments(
        &parse(&report.rows[1].inverse_trace, p),
        &parse(&report.rows[1].inverse_square_trace, p),
        2,
    )
    .unwrap();
    let mut altered = m.clone();
    altered.two_mode_fit.second_eigenvalue_estimate = Some("42".into());
    assert!(altered
        .validate_for_moments(
            &parse(&report.rows[1].inverse_trace, p),
            &parse(&report.rows[1].inverse_square_trace, p),
            2
        )
        .is_err());
    compare(
        m.two_mode_fit
            .smallest_eigenvalue_estimate
            .as_ref()
            .unwrap(),
        &Rational::from(1),
    );
    compare(
        m.two_mode_fit.second_eigenvalue_estimate.as_ref().unwrap(),
        &Rational::from(1000),
    );
    compare(
        m.two_mode_fit.gap_ratio_estimate.as_ref().unwrap(),
        &Rational::from((1, 1000)),
    );
    assert!(
        parse(
            m.two_mode_fit
                .relative_trace_closure_residual
                .as_ref()
                .unwrap(),
            p
        )
        .abs()
            < Float::with_val(p, 2).pow(-450)
    );
    assert!(parse(&m.smallest_eigenvalue_lower_estimate, p) <= 1);
    assert!(parse(&m.smallest_eigenvalue_upper_estimate, p) >= 1);
    let traces = |mu: [i32; 4]| {
        (1..=3)
            .map(|power| {
                mu.iter()
                    .map(|v| Integer::from(*v).pow(power))
                    .sum::<Integer>()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(traces([2, 3, 9, 10]), traces([1, 6, 6, 11]));
    assert_eq!(
        traces([2, 3, 9, 10]),
        vec![Integer::from(24), Integer::from(194), Integer::from(1764)]
    );
    assert_ne!(Rational::from((9, 10)), Rational::from((6, 11)));
    // Three comparable inverse modes can lie outside the exactly-two-mode range.
    let eye = [1, 0, 0, 0, 1, 0, 0, 0, 1].map(|v| Float::with_val(p, v));
    let report =
        analyze_prefixes_with_policy(&eye, 3, p, 32, &[], &PrefixDiagnosticPolicy::full()).unwrap();
    assert_eq!(
        report.rows[0]
            .third_inverse_moment
            .as_ref()
            .unwrap()
            .two_mode_fit
            .status,
        TwoModeMomentStatus::SingleDimension
    );
    assert_eq!(
        report.rows[2]
            .third_inverse_moment
            .as_ref()
            .unwrap()
            .two_mode_fit
            .status,
        TwoModeMomentStatus::OutsideTwoModeRange
    );
}
#[test]
fn third_moment_and_omission_policies_are_deterministic() {
    let d = 137;
    for p in [64, 256, 512] {
        let matrix: Vec<_> = (0..d * d)
            .map(|i| {
                Float::with_val(
                    p,
                    if i / d == i % d {
                        4096
                    } else {
                        ((i / d + i % d) % 9) as i32 - 4
                    },
                )
            })
            .collect();
        for cancellation in [false, true] {
            let policy = PrefixDiagnosticPolicy {
                third_inverse_moment: true,
                innovation_cancellation: cancellation,
            };
            let run = |workers| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| {
                        analyze_prefixes_with_policy(&matrix, d, p, 32, &[127, 137], &policy)
                            .unwrap()
                    })
            };
            let one = run(1);
            assert_eq!(one, run(4));
            assert!(one.rows.iter().all(|r| r.third_inverse_moment.is_some()
                && r.innovation_cancellation.is_some() == cancellation));
        }
    }
}
#[test]
fn corrected_two_moment_residual_includes_the_cubic_two_mode_term() {
    let p = 512;
    for (second, third) in [(1_000_000, None), (1_000_000, Some(1_000_000_000_000_i64))] {
        let values = if let Some(third) = third {
            vec![1_i64, second, third]
        } else {
            vec![1_i64, second]
        };
        let d = values.len();
        let a: Vec<_> = (0..d * d)
            .map(|i| Float::with_val(p, if i / d == i % d { values[i / d] } else { 0 }))
            .collect();
        let r = analyze_prefixes(&a, d, p, 32, &[]).unwrap();
        let corrected = parse(
            r.rows
                .last()
                .unwrap()
                .smallest_eigenvalue_second_order_estimate
                .as_ref()
                .unwrap(),
            p,
        );
        let ratios: Vec<_> = values[1..]
            .iter()
            .map(|v| Float::with_val(p, 1) / Float::with_val(p, *v))
            .collect();
        let s1: Float = ratios.iter().fold(Float::with_val(p, 0), |a, b| a + b);
        let s2: Float = ratios
            .iter()
            .fold(Float::with_val(p, 0), |a, b| a + b.clone().square());
        let mut leading = s1.clone().square() - &s2;
        leading /= 2;
        leading -= s1.clone() * s2 / 2_i32;
        let residual = corrected - 1_i32;
        assert!((residual - leading).abs() < s1.pow(4));
    }
}

#[test]
fn trace_closure_estimates_a_dominant_third_mode_but_aggregates_a_tail() {
    let p = 512;
    // A separated third inverse mode, then two equal tail modes. In the latter
    // case the closure measures their sum, not either individual eigenvalue.
    for tail_count in [1, 2] {
        let mut eigenvalues = vec![1_i64, 1_000_000];
        eigenvalues.extend(std::iter::repeat_n(1_000_000_000_000_i64, tail_count));
        let d = eigenvalues.len();
        let matrix: Vec<_> = (0..d * d)
            .map(|i| {
                Float::with_val(
                    p,
                    if i / d == i % d {
                        eigenvalues[i / d]
                    } else {
                        0
                    },
                )
            })
            .collect();
        let report =
            analyze_prefixes_with_policy(&matrix, d, p, 32, &[], &PrefixDiagnosticPolicy::full())
                .unwrap();
        let row = report.rows.last().unwrap();
        let fit = &row.third_inverse_moment.as_ref().unwrap().two_mode_fit;
        assert_eq!(fit.status, TwoModeMomentStatus::Resolved);
        let closure = parse(fit.relative_trace_closure_residual.as_ref().unwrap(), p);
        let tail = Float::with_val(p, tail_count) / Float::with_val(p, 1_000_000_000_000_i64);
        assert!((closure.clone() / &tail - 1_i32).abs() < Float::with_val(p, 4) / 1_000_000_i32);
        let deficit = closure * parse(&row.inverse_trace, p);
        let scale = Float::with_val(p, 1) / deficit;
        let expected_scale = Float::with_val(p, 1_000_000_000_000_i64) / tail_count;
        assert!((scale / expected_scale - 1_i32).abs() < Float::with_val(p, 2) / 1_000_000_i32);
    }
}
