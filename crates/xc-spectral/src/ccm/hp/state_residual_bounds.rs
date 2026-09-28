//! Directed residual bounds for actual finite symmetric source matrices.
use super::super::retained_evidence::finite_math::{abs, scale, scale_float};
use anyhow::{bail, Result};
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) struct ResidualBounds {
    /// Upper bound on ||A v-lambda v||_2 / ||v||_2, in eigenvalue units.
    pub eigenvalue_error_upper: Float,
    /// Upper bound on ||A v-lambda v||_infinity / ||v||_infinity.
    pub vector_scaled_residual_upper: Float,
}

pub(super) fn evaluate(a: &[Float], v: &[Float], lambda: &Float, p: u32) -> Result<ResidualBounds> {
    let n = v.len();
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || !(64..=1_000_000).contains(&p)
        || a.iter()
            .chain(v)
            .chain(std::iter::once(lambda))
            .any(|x| !x.is_finite() || x.prec() > p)
        || v.iter().all(Float::is_zero)
    {
        bail!("invalid finite source-residual dimensions, precision or vector");
    }
    for i in 0..n {
        for j in i + 1..n {
            if a[i * n + j] != a[j * n + i] {
                bail!("source eigenvalue residual requires exact stored symmetry");
            }
        }
    }
    let work = p + 64;
    let buffers = (a.len() as u64)
        .saturating_mul(2)
        .saturating_add((n as u64).saturating_mul(8))
        .saturating_add(192);
    if buffers.saturating_mul(u64::from(work).div_ceil(8) + 64) > (8u64 << 30) {
        bail!("source residual exceeds numerical workspace budget");
    }
    let ae = a
        .iter()
        .chain(std::iter::once(lambda))
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let ve = i64::from(v.iter().filter_map(Float::get_exp).max().unwrap());
    let normalized = |xs: &[Float], exponent: i64| -> Result<Vec<I>> {
        xs.iter()
            .map(|x| Ok(I::from_float(&scale_float(x, -exponent, work)?, work)?))
            .collect()
    };
    let a = normalized(a, ae)?;
    let v = normalized(v, ve)?;
    let lambda = I::from_float(&scale_float(lambda, -ae, work)?, work)?;
    let mut residuals = Vec::with_capacity(n);
    let mut norm_squared = I::from_i64(0, work);
    let mut vector_max_lower = Float::with_val(work, 0);
    for (i, row) in a.chunks(n).enumerate() {
        let mut value = I::from_i64(0, work);
        for (entry, x) in row.iter().zip(&v) {
            value = value.add(&entry.mul(x));
        }
        residuals.push(abs(&value.sub(&lambda.mul(&v[i])))?);
        let x = abs(&v[i])?;
        norm_squared = norm_squared.add(&x.mul(&x));
        if x.lower() > &vector_max_lower {
            vector_max_lower = x.lower().clone();
        }
    }
    // Normalize residual intervals before squaring. Combine their exponent
    // with the matrix exponent only at the final step, preserving compensated
    // finite results rather than underflowing an intermediate residual square.
    let re = residuals
        .iter()
        .filter_map(|x| x.upper().get_exp())
        .max()
        .map_or(0, i64::from);
    let mut residual_squared = I::from_i64(0, work);
    let mut residual_max_upper = Float::with_val(work, 0);
    for residual in residuals {
        let normalized = scale(&residual, -re)?;
        residual_squared = residual_squared.add(&normalized.mul(&normalized));
        if normalized.upper() > &residual_max_upper {
            residual_max_upper = normalized.upper().clone();
        }
    }
    let two_norm = residual_squared.div(&norm_squared)?.sqrt()?;
    let infinity_norm =
        I::from_float(&residual_max_upper, work)?.div(&I::from_float(&vector_max_lower, work)?)?;
    Ok(ResidualBounds {
        eigenvalue_error_upper: scale_float(two_norm.upper(), ae + re, work)?,
        vector_scaled_residual_upper: scale_float(infinity_norm.upper(), ae + re, work)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;
    #[test]
    fn directed_source_residuals_bound_independent_exact_rational_norms_and_scales() {
        let mut reports = 0;
        for p in [64, 128, 256] {
            for n in 1usize..=5 {
                for case in 0i32..8 {
                    let a = (0..n * n)
                        .map(|j| {
                            Float::with_val(p, ((j / n + j % n) as i32 * 3 + case) % 11 - 5) / 13u32
                        })
                        .collect::<Vec<_>>();
                    let v = (0..n)
                        .map(|j| Float::with_val(p, j as i32 * 2 + case + 1) / 7u32)
                        .collect::<Vec<_>>();
                    let lambda = Float::with_val(p, case - 4) / 3u32;
                    let av = a
                        .iter()
                        .map(|x| x.to_rational().unwrap())
                        .collect::<Vec<_>>();
                    let vv = v
                        .iter()
                        .map(|x| x.to_rational().unwrap())
                        .collect::<Vec<_>>();
                    let l = lambda.to_rational().unwrap();
                    let residual = (0..n)
                        .map(|i| {
                            (0..n)
                                .map(|j| av[i * n + j].clone() * &vv[j])
                                .sum::<Rational>()
                                - l.clone() * &vv[i]
                        })
                        .collect::<Vec<_>>();
                    let squared = residual
                        .iter()
                        .map(|x| x.clone().square())
                        .sum::<Rational>()
                        / vv.iter().map(|x| x.clone().square()).sum::<Rational>();
                    let infinity = residual.iter().map(|x| x.clone().abs()).max().unwrap()
                        / vv.iter().map(|x| x.clone().abs()).max().unwrap();
                    let base = evaluate(&a, &v, &lambda, p).unwrap();
                    for ascale in [-700_000_000i32, 0, 700_000_000] {
                        for vscale in [-700_000_000i32, 0, 700_000_000] {
                            let a = a.iter().map(|x| x.clone() << ascale).collect::<Vec<_>>();
                            let v = v.iter().map(|x| x.clone() << vscale).collect::<Vec<_>>();
                            let actual = evaluate(&a, &v, &(lambda.clone() << ascale), p).unwrap();
                            let two = (actual.eigenvalue_error_upper.clone() >> ascale)
                                .to_rational()
                                .unwrap();
                            let inf = (actual.vector_scaled_residual_upper.clone() >> ascale)
                                .to_rational()
                                .unwrap();
                            assert!(two >= 0 && two.square() >= squared);
                            assert!(inf >= infinity);
                            assert_eq!(
                                actual.eigenvalue_error_upper,
                                base.eigenvalue_error_upper.clone() << ascale
                            );
                            assert_eq!(
                                actual.vector_scaled_residual_upper,
                                base.vector_scaled_residual_upper.clone() << ascale
                            );
                            reports += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(reports, 1080);
    }
    #[test]
    fn source_residual_zero_and_domain_contracts() {
        let p = 128;
        let one = Float::with_val(p, 1);
        let zero = Float::with_val(p, 0);
        let exact = evaluate(
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            &one,
            p,
        )
        .unwrap();
        assert_eq!(exact.eigenvalue_error_upper, 0);
        assert_eq!(exact.vector_scaled_residual_upper, 0);
        assert!(evaluate(&[], &[], &one, p).is_err());
        assert!(evaluate(std::slice::from_ref(&one), &[zero], &one, p).is_err());
        assert!(evaluate(
            &[Float::with_val(p, f64::NAN)],
            std::slice::from_ref(&one),
            &one,
            p
        )
        .is_err());
        assert!(evaluate(
            &[one.clone(), one.clone(), Float::with_val(p, 0), one.clone()],
            &[one.clone(), one.clone()],
            &one,
            p
        )
        .is_err());
    }
}
