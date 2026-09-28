//! Explicit finite point stages for automatic polynomial preparation.
//! Each transcendental output requires directed bounds to agree after rounding.
//! Exact dyadic form splitting is separate from continuum/source accuracy.
use anyhow::{bail, ensure, Result};
use rug::{float::Round, ops::Pow, Float, Integer, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;
const BIT_BUDGET: u64 = 67_108_864;
fn check(x: &Rational) -> Result<()> {
    ensure!(
        u64::from(x.numer().significant_bits()) + u64::from(x.denom().significant_bits())
            <= BIT_BUDGET,
        "preparation exact rational exceeds bit budget"
    );
    Ok(())
}
fn source(x: &Float) -> Result<Rational> {
    super::certified_roots::boundary::rational_budget(std::iter::once(x), 1)?;
    let q = x.to_rational().unwrap();
    check(&q)?;
    Ok(q)
}
pub(super) fn round(x: &Rational, p: u32) -> Result<Float> {
    check(x)?;
    let v = Float::with_val(p, x);
    ensure!(
        v.is_finite() && (!v.is_zero() || x == &0),
        "preparation point exceeds exponent range"
    );
    ensure!(
        v.get_exp() != Some(rug::float::exp_min())
            || !Float::with_val_round(p, x, Round::Zero).0.is_zero(),
        "preparation point partially underflows"
    );
    Ok(v)
}
fn rounded(x: &I, p: u32) -> Result<Option<Float>> {
    x.validate()?;
    let lo = Float::with_val(p, x.lower());
    let hi = Float::with_val(p, x.upper());
    Ok((lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (x.lower().is_zero() && x.upper().is_zero())))
    .then_some(lo))
}
fn shape(n: usize, p: u32) -> Result<()> {
    ensure!(
        (1..=512).contains(&n) && (64..=995_904).contains(&p),
        "preparation dimension/precision exceeds supported domain"
    );
    Ok(())
}
pub(super) fn lattice(k: usize, n: usize) -> Rational {
    Rational::from((k * k, n * n))
}
pub(super) fn square_sum(x: &[Float], p: u32) -> Result<Float> {
    let mut q = Rational::new();
    for x in x {
        let v = source(x)?;
        q += v.square();
        check(&q)?;
    }
    round(&q, p)
}
pub(super) fn coordinates(t: &[Float], l: &Float, n: usize, p: u32) -> Result<Vec<Float>> {
    shape(n, p)?;
    ensure!(
        l > &0 && !t.is_empty() && t.len() <= 1_000_000,
        "invalid preparation coordinates"
    );
    let l = source(l)?;
    t.iter()
        .map(|t| {
            let q = source(t)? * &l / (2 * n);
            for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
                let w = p + guard;
                let value = I::from_rational(&q, w).div(&I::pi(w))?.square();
                if let Some(v) = rounded(&value, p)? {
                    return Ok(v);
                }
            }
            bail!("preparation coordinate rounding unresolved")
        })
        .collect()
}
pub(super) fn basis(z: &[Float], l: &Float, n: usize, d: usize, p: u32) -> Result<Vec<Float>> {
    shape(n, p)?;
    ensure!(
        (1..=65).contains(&d) && d <= n + 1 && z.len() >= n + 1 - d && l > &0,
        "invalid preparation basis shape"
    );
    let prefix = n + 1 - d;
    let roots = z[..prefix].iter().map(source).collect::<Result<Vec<_>>>()?;
    let length = source(l)?;
    let mut facts = vec![Integer::from(1); 2 * n + 1];
    for k in 1..=2 * n {
        facts[k] = facts[k - 1].clone() * k;
    }
    let power = Integer::from(n).pow(2 * n as u32);
    let mut v = vec![Float::with_val(p, 0); (2 * n + 1) * d];
    for k in 0..=n {
        let a = lattice(k, n);
        let mut q = Rational::from((power.clone(), facts[n - k].clone() * &facts[n + k]));
        if k % 2 == 1 {
            q = -q;
        }
        for r in &roots {
            q *= a.clone() - r;
            check(&q)?;
        }
        for j in 0..d {
            let mut value = None;
            for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
                let w = p + guard;
                let r = I::from_rational(&q, w).div(&I::from_rational(&length, w).sqrt()?)?;
                if let Some(r) = rounded(&r, p)? {
                    value = Some(r);
                    break;
                }
            }
            let value =
                value.ok_or_else(|| anyhow::anyhow!("preparation basis rounding unresolved"))?;
            v[(n + k) * d + j] = value.clone();
            v[(n - k) * d + j] = value;
            q *= &a;
            check(&q)?;
        }
    }
    Ok(v)
}
pub(super) fn forms(
    matrix: &[Float],
    v: &[Float],
    dim: usize,
    d: usize,
    p: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    ensure!(
        dim <= 1025
            && dim > 0
            && d > 0
            && d <= 65
            && matrix.len() == dim * dim
            && v.len() == dim * d,
        "preparation form shape mismatch"
    );
    let estimate = matrix
        .iter()
        .chain(v)
        .try_fold(0u128, |sum, x| -> Result<_> {
            ensure!(x.is_finite(), "nonfinite preparation form source");
            Ok(sum
                + u128::from(x.prec())
                + u128::from(i64::from(x.get_exp().unwrap_or(0)).unsigned_abs())
                + 2)
        })?;
    ensure!(
        estimate * (d as u128 + 4) <= 8u128 << 33,
        "preparation exact form exceeds 8 GiB workspace budget"
    );
    let v = v.iter().map(source).collect::<Result<Vec<_>>>()?;
    let mut av = vec![Rational::new(); dim * d];
    for r in 0..dim {
        for k in 0..dim {
            let a: Rational = (source(&matrix[r * dim + k])? + source(&matrix[k * dim + r])?) / 2;
            for j in 0..d {
                av[r * d + j] += a.clone() * &v[k * d + j];
                check(&av[r * d + j])?;
            }
        }
    }
    let mut gram = vec![Float::with_val(p, 0); d * d];
    let mut actual = gram.clone();
    for i in 0..d {
        for j in 0..=i {
            let mut g = Rational::new();
            let mut a = Rational::new();
            for k in 0..dim {
                g += v[k * d + i].clone() * &v[k * d + j];
                a += v[k * d + i].clone() * &av[k * d + j];
                check(&g)?;
                check(&a)?;
            }
            let g = round(&g, p)?;
            let a = round(&(a / 2), p)?;
            gram[i * d + j] = g.clone();
            gram[j * d + i] = g;
            actual[i * d + j] = a.clone();
            actual[j * d + i] = a;
        }
    }
    Ok((gram, actual))
}
pub(super) fn head(
    t: &[Float],
    z: &[Float],
    l: &Float,
    n: usize,
    d: usize,
    p: u32,
) -> Result<Vec<Float>> {
    shape(n, p)?;
    ensure!(
        d > 0
            && d <= 65
            && d <= n + 1
            && t.len() == z.len()
            && t.len() >= n + 1 - d
            && t.len() <= 1_000_000,
        "invalid preparation head shape"
    );
    let prefix = n + 1 - d;
    let l = source(l)?;
    super::certified_roots::boundary::rational_budget(z.iter(), 1)?;
    let z = z.iter().map(source).collect::<Result<Vec<_>>>()?;
    // Stream exact coefficients: their aggregate size can exceed the budget
    // even though a single coefficient and the moment accumulators fit. Keep
    // the same exact factors and directed summation order at each guard tier.
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let w = p + guard;
        let mut moments = vec![I::from_i64(0, w); 2 * d - 1];
        for (j, t) in t.iter().enumerate().skip(prefix) {
            let theta = source(t)? * &l / 2;
            let mut factor = Rational::from(1);
            for r in &z[..prefix] {
                factor *= z[j].clone() - r;
                check(&factor)?;
            }
            for k in 1..=n {
                let denominator = z[j].clone() - lattice(k, n);
                ensure!(
                    denominator != 0,
                    "stored preparation coordinate is a rational pole"
                );
                factor /= denominator;
                check(&factor)?;
            }
            let factor = factor.square();
            check(&factor)?;
            let theta_i = I::from_rational(&theta, w);
            let sinc = if theta == 0 {
                I::from_i64(1, w)
            } else {
                theta_i.sin().div(&theta_i)?
            };
            let mut weight = sinc.square().mul(&I::from_rational(&factor, w));
            let z = I::from_rational(&z[j], w);
            for m in &mut moments {
                *m = m.add(&weight);
                weight = weight.mul(&z);
            }
        }
        let values = moments
            .iter()
            .map(|x| rounded(x, p))
            .collect::<Result<Option<Vec<_>>>>()?;
        if let Some(values) = values {
            return Ok((0..d * d).map(|k| values[k / d + k % d].clone()).collect());
        }
    }
    bail!("preparation head rounding unresolved within guard budget")
}
/// Preserve the exact difference of the stored actual/head points by promoting
/// representation precision. The precision increase does not add source accuracy.
pub(super) fn split(actual: &[Float], head: &[Float], p: u32) -> Result<(u32, Vec<Float>)> {
    ensure!(
        actual.len() == head.len(),
        "preparation tail shape mismatch"
    );
    let tail = actual
        .iter()
        .zip(head)
        .map(|(a, h)| Ok(source(a)? - source(h)?))
        .collect::<Result<Vec<_>>>()?;
    let target = tail
        .iter()
        .map(|x| x.numer().significant_bits())
        .max()
        .unwrap_or(p)
        .max(p);
    ensure!(
        target <= 1_000_000,
        "exact preparation tail exceeds representation precision budget"
    );
    let values = tail
        .iter()
        .map(|q| {
            let v = round(q, target)?;
            ensure!(
                v.to_rational().as_ref() == Some(q),
                "tail split lost exact dyadic difference"
            );
            Ok(v)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((target, values))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preparation_basis_matches_closed_rational_coefficients() {
        for p in [64, 128, 256] {
            let v = basis(&[], &Float::with_val(p, 4), 2, 3, p).unwrap();
            let coefficient = [
                Rational::from((1, 3)),
                Rational::from((-4, 3)),
                Rational::from(2),
                Rational::from((-4, 3)),
                Rational::from((1, 3)),
            ];
            for row in 0..5 {
                let coordinate = Rational::from(((row as i32 - 2).pow(2), 4));
                for j in 0..3 {
                    assert_eq!(
                        v[row * 3 + j],
                        Float::with_val(
                            p,
                            coefficient[row].clone() * coordinate.clone().pow(j as u32)
                        )
                    );
                }
            }
        }
    }
    #[test]
    fn preparation_forms_match_exact_closed_quadratic_values() {
        for p in [64, 128, 256] {
            for shift in [0u32, 200, 600] {
                let a = Float::with_val(p, 1) << shift;
                let v = vec![
                    Float::with_val(p, 1),
                    a.clone(),
                    Float::with_val(p, 2),
                    Float::with_val(p, 1),
                    Float::with_val(p, 3),
                    -a.clone(),
                ];
                let matrix = [2, 7, 0, -7, 4, 5, 0, -5, 6].map(|x| Float::with_val(p, x));
                let (g, f) = forms(&matrix, &v, 3, 2, p).unwrap();
                let a = a.to_rational().unwrap();
                let want_g = [
                    Rational::from(14),
                    a.clone() * -2 + 2,
                    a.clone() * -2 + 2,
                    a.clone().square() * 2 + 1,
                ];
                let want_f = [
                    Rational::from(36),
                    a.clone() * -8 + 4,
                    a.clone() * -8 + 4,
                    a.square() * 4 + 2,
                ];
                for k in 0..4 {
                    assert_eq!(g[k], Float::with_val(p, &want_g[k]));
                    assert_eq!(f[k], Float::with_val(p, &want_f[k]));
                }
            }
        }
    }
    #[test]
    fn preparation_head_matches_independent_high_precision_formula() {
        for p in [64, 128, 256] {
            for n in [1, 2] {
                let l = Float::with_val(p, 4);
                let t = [1, 2, 3].map(|x| Float::with_val(p, x));
                let z = [0.125, 0.375, 0.75].map(|x| Float::with_val(p, x));
                let got = head(&t, &z, &l, n, 2, p).unwrap();
                let mut moments = vec![Float::with_val(2048, 0); 3];
                for j in n - 1..3 {
                    let theta = Float::with_val(2048, &t[j]) * 2u32;
                    let mut f = theta.clone().sin() / theta;
                    if n == 2 {
                        f *= Float::with_val(2048, &z[j]) - &z[0];
                    }
                    for k in 1..=n {
                        f /= Float::with_val(2048, &z[j])
                            - Float::with_val(2048, Rational::from((k * k, n * n)));
                    }
                    let mut weight = f.square();
                    for m in &mut moments {
                        *m += &weight;
                        weight *= &z[j];
                    }
                }
                for k in 0..4 {
                    assert_eq!(got[k], Float::with_val(p, &moments[k / 2 + k % 2]));
                }
            }
        }
    }
    #[test]
    fn preparation_head_streams_coefficients_beyond_aggregate_storage_budget() {
        // Each exact squared factor uses about 4 million bits; retaining all
        // twenty exceeded the 64-Mibit coefficient budget. Streaming needs one.
        // H_k=20*z^k/(1-z)^2 rounds to 20*z^k at 64 bits for this tiny z.
        let p = 64;
        let z = Float::with_val(p, 1) >> 1_000_000u32;
        let head = head(
            &vec![Float::with_val(p, 0); 20],
            &vec![z.clone(); 20],
            &Float::with_val(p, 1),
            1,
            2,
            p,
        )
        .unwrap();
        for (k, value) in head.iter().enumerate() {
            assert_eq!(
                *value,
                Float::with_val(p, 20) * z.clone().pow((k / 2 + k % 2) as u32)
            );
        }
    }
    #[test]
    fn preparation_tail_split_handles_cancellation_and_rejects_outside_domain() {
        for p in [64, 128, 256] {
            let a = Float::with_val(p, 1);
            let h = Float::with_val(p, 1) << 400u32;
            let (target, t) = split(std::slice::from_ref(&a), std::slice::from_ref(&h), p).unwrap();
            assert!(target >= 400);
            assert_eq!(
                a.to_rational().unwrap(),
                h.to_rational().unwrap() + t[0].to_rational().unwrap()
            );
        }
        assert!(coordinates(&[Float::with_val(64, 1)], &Float::with_val(64, 1), 0, 64).is_err());
        assert!(basis(&[], &Float::with_val(64, 1), 513, 1, 64).is_err());
        assert!(head(
            &[Float::with_val(64, 1)],
            &[Float::with_val(64, 1)],
            &Float::with_val(64, 1),
            1,
            2,
            64
        )
        .is_err());
        assert!(square_sum(&[Float::with_val(64, rug::float::Special::Nan)], 64).is_err());
    }
}
