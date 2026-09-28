//! Conditional tail perturbation bounds for the exact stored Gram matrix.
//! If ||Delta A||_2 <= epsilon and G is positive definite, each exact generalized
//! energy E=2*lambda changes by at most 2*epsilon*trace(G^-1).
use anyhow::{bail, Result};
use rug::{float::Round, Float, Rational};

fn check<'a>(values: impl Iterator<Item = &'a Rational>, limit: u64) -> Result<()> {
    let values = values.collect::<Vec<_>>();
    let bits = values
        .iter()
        .map(|x| {
            u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits())
        })
        .sum::<u128>();
    if bits > 67_108_864 {
        bail!("exact tail perturbation exceeds 64 Mbit rational budget");
    }
    if bits.div_ceil(8) * 4 + values.len() as u128 * 128 > u128::from(limit) {
        bail!("exact tail perturbation algebra exceeds workspace budget");
    }
    Ok(())
}

pub(super) fn bound(gram: &[Float], error: &Float, n: usize, p: u32, limit: u64) -> Result<Float> {
    if n == 0 || n > 128 || n.checked_mul(n) != Some(gram.len()) || error < &0 {
        bail!("invalid tail perturbation shape or error");
    }
    if !(64..=1_000_000).contains(&p)
        || !(64..=1_000_000).contains(&error.prec())
        || !error.is_finite()
        || gram
            .iter()
            .any(|x| !x.is_finite() || x.prec() < p || x.prec() > 1_000_000)
    {
        bail!("invalid tail perturbation source precision or value");
    }
    crate::ccm::certified_roots::boundary::rational_point_vector_budget(
        gram.iter().chain(std::iter::once(error)),
    )?;
    let g = gram
        .iter()
        .map(|x| x.to_rational().expect("finite Gram point"))
        .collect::<Vec<_>>();
    let error = error.to_rational().expect("finite error point");
    check(g.iter().chain(std::iter::once(&error)), limit)?;
    if (0..n).any(|r| (0..r).any(|c| g[r * n + c] != g[c * n + r])) {
        bail!("tail perturbation Gram matrix must be symmetric");
    }
    let mut l = vec![Rational::from(0); n * n];
    let mut d = vec![Rational::from(0); n];
    // Exact LDL^T: positivity of every pivot proves positive definiteness.
    for r in 0..n {
        l[r * n + r] = Rational::from(1);
        for c in 0..=r {
            let mut value = g[r * n + c].clone();
            for k in 0..c {
                value -= l[r * n + k].clone() * &l[c * n + k] * &d[k];
                check(std::iter::once(&value), limit)?;
            }
            if r == c {
                if value <= 0 {
                    bail!("exact tail perturbation Gram matrix is not positive definite");
                }
                d[r] = value;
            } else {
                l[r * n + c] = value / &d[c];
            }
        }
        check(g.iter().chain(&l).chain(&d), limit)?;
    }
    let mut inverse = vec![Rational::from(0); n * n];
    for r in 0..n {
        inverse[r * n + r] = Rational::from(1);
        for c in 0..r {
            let mut value = Rational::from(0);
            for k in c..r {
                value -= l[r * n + k].clone() * &inverse[k * n + c];
                check(std::iter::once(&value), limit)?;
            }
            inverse[r * n + c] = value;
        }
        check(g.iter().chain(&l).chain(&d).chain(&inverse), limit)?;
    }
    let mut trace = Rational::from(0);
    for r in 0..n {
        for c in 0..=r {
            trace += inverse[r * n + c].clone().square() / &d[r];
            check(std::iter::once(&trace), limit)?;
        }
    }
    let exact = trace * error * 2;
    check(std::iter::once(&exact), limit)?;
    let result = Float::with_val_round(p, &exact, Round::Up).0;
    if !result.is_finite() || (result == 0 && exact != 0) {
        bail!("tail perturbation bound exceeds output exponent range");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tail_bound_admits_declared_large_diagonal_grams() {
        for n in [80usize, 96, 128] {
            let mut gram = vec![Float::with_val(128, 0); n * n];
            for i in 0..n {
                gram[i * n + i] = Float::with_val(128, 2);
            }
            // trace((2I)^-1)=n/2, so 2*epsilon*trace=7n/10.
            let error = Float::with_val_round(128, Rational::from((7, 10)), Round::Up).0;
            let upper = bound(&gram, &error, n, 128, 8 << 30).unwrap();
            assert!(upper >= Rational::from((7 * n, 10)));
            assert!(bound(&gram, &error, n, 128, 1024).is_err());
        }
    }
    #[test]
    fn inverse_trace_bound_matches_closed_two_by_two_formula() {
        for p in [64, 128, 256] {
            for (a, b, c) in [(2, 1, 3), (5, 2, 7), (17, -3, 4)] {
                let g = [a, b, b, c].map(|v| Float::with_val(p, v));
                let actual = bound(&g, &Float::with_val(p, 1), 2, p, 1 << 20).unwrap();
                let exact = Rational::from((2 * (a + c), a * c - b * b));
                assert_eq!(actual, Float::with_val_round(p, exact, Round::Up).0);
            }
        }
    }
    #[test]
    fn invalid_gram_error_and_resource_limit_fail_closed() {
        let f = |v| Float::with_val(128, v);
        for g in [[1, 2, 2, 1], [1, 1, 1, 1], [2, 1, 0, 2]] {
            assert!(bound(&g.map(f), &f(1), 2, 128, 1 << 20).is_err());
        }
        assert!(bound(&[f(1)], &f(-1), 1, 128, 1 << 20).is_err());
        assert!(bound(&[f(1)], &f(1), 1, 128, 1).is_err());
    }
}
