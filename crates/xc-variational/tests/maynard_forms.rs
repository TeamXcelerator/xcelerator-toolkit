#![cfg(feature = "hp")]
use rug::{Float, Rational};
use std::collections::BTreeMap;
use xc_variational::maynard::MkMonomialReference;
use xc_variational::maynard::MkSymmetricReference;
type Polynomial = BTreeMap<Vec<u32>, Rational>;
fn multiply(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let mut out = Polynomial::new();
    for (i, x) in a {
        for (j, y) in b {
            let e = i.iter().zip(j).map(|(i, j)| i + j).collect();
            *out.entry(e).or_default() += x.clone() * y;
        }
    }
    out
}
// Integrate the last coordinate over [0, 1-sum(other coordinates)].
// Only polynomial multiplication and elementary antiderivatives are used;
// no factorial, beta, or toolkit integration formula enters this oracle.
fn eliminate(p: Polynomial, n: usize) -> Polynomial {
    let mut remainder = Polynomial::from([(vec![0; n - 1], Rational::from(1))]);
    for axis in 0..n - 1 {
        let mut e = vec![0; n - 1];
        e[axis] = 1;
        remainder.insert(e, Rational::from(-1));
    }
    let mut out = Polynomial::new();
    for (e, mut c) in p {
        let power = e[n - 1] + 1;
        c /= power;
        let mut term = Polynomial::from([(e[..n - 1].to_vec(), c)]);
        for _ in 0..power {
            term = multiply(&term, &remainder);
        }
        for (e, c) in term {
            *out.entry(e).or_default() += c;
        }
    }
    out
}
fn integral(mut p: Polynomial, n: usize) -> Rational {
    for dimension in (1..=n).rev() {
        p = eliminate(p, dimension);
    }
    p.get(&Vec::new()).cloned().unwrap_or_default()
}
#[test]
fn monomial_forms_equal_fresh_exact_iterated_antiderivatives() {
    let mut comparisons = 0;
    for k in 1..=3 {
        let r = MkMonomialReference::new(k, 3).unwrap();
        for a in r.indices() {
            for b in r.indices() {
                let e = a.0.iter().zip(&b.0).map(|(x, y)| x + y).collect();
                let expected = integral(Polynomial::from([(e, Rational::from(1))]), k);
                assert_eq!(r.i_entry(a, b).unwrap(), expected);
                comparisons += 1;
                for axis in 0..k {
                    let marginal = |v: &[u32]| {
                        let mut e: Vec<_> = v
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != axis)
                            .map(|(_, v)| *v)
                            .collect();
                        e.push(v[axis]);
                        eliminate(Polynomial::from([(e, Rational::from(1))]), k)
                    };
                    let expected = integral(multiply(&marginal(&a.0), &marginal(&b.0)), k - 1);
                    assert_eq!(r.j_entry(axis, a, b).unwrap(), expected);
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!(
        "{comparisons} exact I/J comparisons against freshly implemented iterated antiderivatives"
    );
}
#[test]
fn nonzero_mpfr_streamed_actions_must_report_exponent_underflow() {
    let p = 96;
    let r = MkSymmetricReference::new(3, 0).unwrap();
    let mut smallest = Float::with_val(p, 1);
    smallest <<= rug::float::exp_min() - 1;
    assert!(smallest.is_finite() && !smallest.is_zero());
    // Exact I=[1/6], J_total=[1/4]. Both exact products are nonzero,
    // below the MPFR exponent floor. A successful zero action loses the metric.
    for metric in [true, false] {
        let mut out = vec![Float::with_val(p, 1)];
        let result = if metric {
            r.apply_i_hp(&[smallest.clone()], &mut out, p)
        } else {
            r.apply_j_total_hp(&[smallest.clone()], &mut out, p)
        };
        eprintln!(
            "metric={metric}, input_exp={:?}, result={result:?}, output={}",
            smallest.get_exp(),
            out[0]
        );
        assert!(
            result.is_err(),
            "nonzero exact streamed action silently underflowed to zero"
        );
    }
}

#[test]
fn mpfr_streamed_actions_distinguish_exact_cancellation_from_exponent_loss() {
    let p = 96;
    let r = MkSymmetricReference::new(1, 1).unwrap();
    let mut out = vec![Float::with_val(p, 0); 2];
    r.apply_i_hp(
        &[Float::with_val(p, 1), Float::with_val(p, -2)],
        &mut out,
        p,
    )
    .unwrap();
    assert!(out[0].is_zero()); // I_00 - 2 I_01 = 1 - 1, exactly.
    assert!(out[1] < 0);
    r.apply_i_hp(&[Float::with_val(p, 0), Float::with_val(p, 0)], &mut out, p)
        .unwrap();
    assert!(out.iter().all(Float::is_zero));
    let mut a = Float::with_val(p, 1);
    a <<= rug::float::exp_min();
    let mut adjacent = a.clone();
    adjacent.next_up();
    assert!(adjacent > a);
    let mut negative_twice = -adjacent;
    negative_twice *= 2;
    let outcome = r.apply_i_hp(&[a, negative_twice], &mut out, p);
    assert!(
        outcome.is_err(),
        "unequal nonzero opposite terms were rounded to zero without error"
    );
    let mut safe = Float::with_val(p, 1);
    safe <<= rug::float::exp_min() + 4;
    let negative = -Float::with_val(p, &safe * 2);
    r.apply_i_hp(&[safe, negative], &mut out, p).unwrap();
    assert!(out[0].is_zero() && out[1] < 0);
}
