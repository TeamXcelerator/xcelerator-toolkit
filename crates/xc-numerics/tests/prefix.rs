#![cfg(feature = "hp")]

use rug::{float::Special, ops::Pow, Float, Integer, Rational};
use xc_numerics::prefix::{analyze_prefixes, checked_decimal_export};

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
    let mut previous_trace = Rational::from(0);
    for k in 1..=n {
        let prefix: Vec<Rational> = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].clone())
            .collect();
        let inv = inverse(&prefix, k);
        let mut trace = Rational::from(0);
        let mut trace2 = Rational::from(0);
        for i in 0..k {
            trace += &inv[i * k + i];
        }
        for x in &inv {
            let mut square = x.clone();
            square *= x;
            trace2 += square;
        }
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
