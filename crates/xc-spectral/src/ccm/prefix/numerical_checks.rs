//! Directed checks on the actual stored prefix matrix and exported points.
use super::super::retained_evidence::finite_math::{abs, scale, scale_float};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) fn workspace(count: usize, p: u32) -> Result<()> {
    if count == 0
        || count > 16385
        || !(64..=1_000_000).contains(&p)
        || count as u128 * (u128::from(p + 64).div_ceil(8) + 96) * 8 > (8u128 << 30)
    {
        bail!("prefix vector exceeds supported dimensions or workspace budget");
    }
    Ok(())
}

// Accept only when the outward distance upper bound is within the actual limit.
pub(super) fn component_close(a: &Float, b: &Float, tolerance: &Float, p: u32) -> bool {
    if !(64..=1_000_000).contains(&p)
        || [a, b, tolerance]
            .iter()
            .any(|x| !x.is_finite() || x.prec() > p)
        || tolerance < &0
    {
        return false;
    }
    let lower = Float::with_val_round(p, a - b, Round::Down).0.abs();
    let upper = Float::with_val_round(p, a - b, Round::Up).0.abs();
    lower.is_finite() && upper.is_finite() && lower <= *tolerance && upper <= *tolerance
}

pub(super) enum RightHandSide<'a> {
    Vector(&'a [Float]),
    Eigenvalue(&'a Float),
}

fn norm(values: &[I], p: u32) -> Result<I> {
    let mut squared = I::from_i64(0, p);
    for x in values {
        let x = abs(x)?;
        squared = squared.add(&x.mul(&x));
    }
    Ok(squared.sqrt()?)
}

/// Upper bound on ||A v-b||_2/(||A||_F ||v||_2+||b||_2), where b is
/// either an actual stored vector or the exact product lambda*v.
pub(super) fn backward_error(
    matrix: &[Float],
    stride: usize,
    vector: &[Float],
    rhs: RightHandSide<'_>,
    p: u32,
) -> Result<Float> {
    let n = vector.len();
    if n == 0
        || n > stride
        || stride.checked_mul(stride) != Some(matrix.len())
        || !(64..=1_000_000).contains(&p)
        || vector.iter().all(Float::is_zero)
    {
        bail!("invalid prefix residual dimensions, precision or vector");
    }
    let work = p + 64;
    let buffers = (n as u64)
        .saturating_mul(n as u64)
        .saturating_mul(2)
        .saturating_add((n as u64).saturating_mul(12))
        .saturating_add(128);
    if buffers.saturating_mul(u64::from(work).div_ceil(8) + 64) > (8u64 << 30) {
        bail!("prefix residual exceeds numerical buffer budget");
    }
    let entries = matrix
        .chunks(stride)
        .take(n)
        .flat_map(|row| row.iter().take(n))
        .collect::<Vec<_>>();
    let right = match &rhs {
        RightHandSide::Vector(v) if v.len() == n => *v,
        RightHandSide::Eigenvalue(lambda) => std::slice::from_ref(*lambda),
        _ => bail!("prefix residual right-hand-side dimension mismatch"),
    };
    if entries
        .iter()
        .copied()
        .chain(vector)
        .chain(right)
        .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("prefix residual requires finite points without precision loss");
    }
    let ve = i64::from(vector.iter().filter_map(Float::get_exp).max().unwrap());
    let ae = entries
        .iter()
        .filter_map(|x| x.get_exp())
        .max()
        .map(i64::from);
    let be = right.iter().filter_map(Float::get_exp).max().map(i64::from);
    let common = match &rhs {
        RightHandSide::Vector(_) => ae.map(|x| x + ve).into_iter().chain(be).max().unwrap_or(0),
        RightHandSide::Eigenvalue(_) => ae.into_iter().chain(be).max().unwrap_or(0) + ve,
    };
    let point = |x: &Float, shift: i64| -> Result<I> {
        Ok(I::from_float(&scale_float(x, shift, work)?, work)?)
    };
    let a = entries
        .iter()
        .map(|x| point(x, ve - common))
        .collect::<Result<Vec<_>>>()?;
    let v = vector
        .iter()
        .map(|x| point(x, -ve))
        .collect::<Result<Vec<_>>>()?;
    let right = match rhs {
        RightHandSide::Vector(b) => b
            .iter()
            .map(|x| point(x, -common))
            .collect::<Result<Vec<_>>>()?,
        RightHandSide::Eigenvalue(lambda) => {
            let lambda = point(lambda, ve - common)?;
            v.iter().map(|x| lambda.mul(x)).collect()
        }
    };
    let mut residuals = Vec::with_capacity(n);
    for (i, row) in a.chunks(n).enumerate() {
        let mut value = I::from_i64(0, work);
        for (a, x) in row.iter().zip(&v) {
            value = value.add(&a.mul(x));
        }
        residuals.push(abs(&value.sub(&right[i]))?);
    }
    let denominator = norm(&a, work)?
        .mul(&norm(&v, work)?)
        .add(&norm(&right, work)?);
    if denominator.upper() == &0 && residuals.iter().all(|x| x.upper() == &0) {
        return Ok(Float::with_val(p, 0));
    }
    if denominator.lower() <= &0 {
        bail!("unresolved prefix residual normalization");
    }
    // Residual components can be vastly smaller than A or b. Scaling these
    // intervals before squaring avoids an intermediate squared-norm underflow.
    let re = residuals
        .iter()
        .filter_map(|x| x.upper().get_exp())
        .max()
        .map_or(0, i64::from);
    let residuals = residuals
        .iter()
        .map(|x| scale(x, -re))
        .collect::<Result<Vec<_>>>()?;
    let ratio = norm(&residuals, work)?.div(&denominator)?;
    let upper = scale_float(ratio.upper(), re, work)?;
    let upper = Float::with_val_round(p, upper, Round::Up).0;
    if !upper.is_finite() {
        bail!("prefix backward-error bound exceeded finite range");
    }
    Ok(upper)
}

/// Conservative norm-one check for the actual exported vector.
pub(super) fn unit_deviation_upper(vector: &[Float], p: u32) -> Result<Float> {
    if vector.is_empty()
        || !(64..=1_000_000).contains(&p)
        || vector.iter().any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid exported norm input");
    }
    workspace(vector.len(), p)?;
    let work = p + 64;
    let mut lower = Float::with_val(work, 0);
    let mut upper = Float::with_val(work, 0);
    for x in vector {
        lower.hypot_round(x, Round::Down);
        upper.hypot_round(x, Round::Up);
    }
    let length = I::new(lower, upper)?;
    let difference = abs(&length.sub(&I::from_i64(1, work)))?;
    let bound = Float::with_val_round(p, difference.upper(), Round::Up).0;
    if !bound.is_finite() {
        bail!("prefix norm-deviation bound is unrepresentable");
    }
    Ok(bound)
}

/// Computed normalization after exact binary scaling; no exact-unit-norm claim.
pub(super) fn unit(vector: &[Float], p: u32) -> Result<Vec<Float>> {
    if vector.is_empty()
        || !(64..=1_000_000).contains(&p)
        || vector.iter().any(|x| !x.is_finite() || x.prec() > p)
        || vector.iter().all(Float::is_zero)
    {
        bail!("invalid finite prefix vector normalization");
    }
    workspace(vector.len(), p)?;
    super::super::retained_evidence::point::unit(vector, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;

    fn exact_norm(v: &[Rational], p: u32) -> I {
        let squared = v.iter().map(|x| x.clone().square()).sum::<Rational>();
        I::from_rational(&squared, p).sqrt().unwrap()
    }

    #[test]
    fn prefix_backward_errors_enclose_exact_rational_equations_at_independent_scales() {
        let mut checks = 0;
        for p in [64, 128, 256] {
            for n in 1usize..=4 {
                for case in 0i32..12 {
                    let a = (0..n * n)
                        .map(|j| Float::with_val(p, (j as i32 * 3 + case) % 17 - 8) / 11u32)
                        .collect::<Vec<_>>();
                    let v = (0..n)
                        .map(|j| Float::with_val(p, j as i32 + case + 1) / 7u32)
                        .collect::<Vec<_>>();
                    let b = (0..n)
                        .map(|j| Float::with_val(p, j as i32 * 2 - case) / 13u32)
                        .collect::<Vec<_>>();
                    let lambda = Float::with_val(p, case - 5) / 3u32;
                    let ar = a
                        .iter()
                        .map(|x| x.to_rational().unwrap())
                        .collect::<Vec<_>>();
                    let vr = v
                        .iter()
                        .map(|x| x.to_rational().unwrap())
                        .collect::<Vec<_>>();
                    for eigen in [false, true] {
                        let br = if eigen {
                            vr.iter()
                                .map(|x| x.clone() * lambda.to_rational().unwrap())
                                .collect::<Vec<_>>()
                        } else {
                            b.iter()
                                .map(|x| x.to_rational().unwrap())
                                .collect::<Vec<_>>()
                        };
                        let residual = (0..n)
                            .map(|i| {
                                (0..n)
                                    .map(|j| ar[i * n + j].clone() * &vr[j])
                                    .sum::<Rational>()
                                    - &br[i]
                            })
                            .collect::<Vec<_>>();
                        let work = p + 512;
                        let denominator = exact_norm(&ar, work)
                            .mul(&exact_norm(&vr, work))
                            .add(&exact_norm(&br, work));
                        let reference = exact_norm(&residual, work).div(&denominator).unwrap();
                        let rhs = if eigen {
                            RightHandSide::Eigenvalue(&lambda)
                        } else {
                            RightHandSide::Vector(&b)
                        };
                        let base = backward_error(&a, n, &v, rhs, p).unwrap();
                        // A rigorous higher-precision interval of the exact rational
                        // contractions supplies an independent numerical reference.
                        let sumsq = |values: &[Rational]| {
                            values.iter().map(|x| x.clone().square()).sum::<Rational>()
                        };
                        let av = sumsq(&ar) * sumsq(&vr);
                        let bs = sumsq(&br);
                        let rs = sumsq(&residual);
                        let x2 = base.to_rational().unwrap().square();
                        let difference = rs - x2.clone() * (av.clone() + &bs);
                        if difference > 0 {
                            assert!(difference.square() <= x2.square() * av * bs * 4);
                        }
                        let excess = Float::with_val(work, &base) - reference.upper();
                        assert!(excess < (Float::with_val(work, 1) >> (p - 4)));
                        for ascale in [-500_000_000i32, 0, 500_000_000] {
                            for vscale in [-300_000_000i32, 0, 300_000_000] {
                                let aa = a.iter().map(|x| x.clone() << ascale).collect::<Vec<_>>();
                                let vv = v.iter().map(|x| x.clone() << vscale).collect::<Vec<_>>();
                                let bb = b
                                    .iter()
                                    .map(|x| x.clone() << (ascale + vscale))
                                    .collect::<Vec<_>>();
                                let ll = lambda.clone() << ascale;
                                let rhs = if eigen {
                                    RightHandSide::Eigenvalue(&ll)
                                } else {
                                    RightHandSide::Vector(&bb)
                                };
                                let bound = backward_error(&aa, n, &vv, rhs, p).unwrap();
                                assert_eq!(bound, base);
                                checks += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(checks, 2592);
    }

    #[test]
    fn prefix_underflow_counterexample_and_exact_lambda_product_are_detected() {
        let p = 256;
        let epsilon = Float::with_val(p, 1) >> 100i32;
        let tolerance = Float::with_val(p, 1) >> 120i32;
        let vector = unit(&[Float::with_val(p, 1), epsilon.clone()], p).unwrap();
        for exponent in [0i32, -536_870_880] {
            let s = Float::with_val(p, 1) << exponent;
            let a = vec![
                s.clone(),
                Float::with_val(p, 0),
                Float::with_val(p, 0),
                Float::with_val(p, &s * 2),
            ];
            let bound = backward_error(&a, 2, &vector, RightHandSide::Eigenvalue(&s), p).unwrap();
            assert!(bound > tolerance);
            assert!(bound > epsilon.clone() / 8);
        }
        // Rounded A*v and lambda*v coincide, but exact stored-point contractions
        // leave residual (1,1). Directed evaluation must not accept a false zero.
        let p = 64;
        let lambda = Float::with_val(p, 1) << 100i32;
        let one = Float::with_val(p, 1);
        let a = vec![lambda.clone(), one.clone(), one.clone(), lambda.clone()];
        let v = vec![one.clone(), one];
        let bound = backward_error(&a, 2, &v, RightHandSide::Eigenvalue(&lambda), p).unwrap();
        assert!(bound > (Float::with_val(p, 1) >> 103i32));
    }

    #[test]
    fn prefix_normalization_and_norm_bounds_cover_extreme_vectors_and_shapes() {
        for p in [64, 128, 256] {
            let v = vec![Float::with_val(p, 3), Float::with_val(p, 4)];
            let expected = unit(&v, p).unwrap();
            let squared = expected
                .iter()
                .map(|x| x.to_rational().unwrap().square())
                .sum::<Rational>();
            let actual = unit_deviation_upper(&expected, p).unwrap();
            let bound = actual.to_rational().unwrap();
            assert!((Rational::from(1) + &bound).square() >= squared);
            if bound < 1 {
                assert!((Rational::from(1) - &bound).square() <= squared);
            }
            for shift in [-700_000_000i32, 700_000_000] {
                let scaled = v.iter().map(|x| x.clone() << shift).collect::<Vec<_>>();
                assert_eq!(unit(&scaled, p).unwrap(), expected);
            }
            let z = Float::with_val(p, 0);
            let one = Float::with_val(p, 1);
            assert_eq!(
                backward_error(
                    std::slice::from_ref(&z),
                    1,
                    std::slice::from_ref(&one),
                    RightHandSide::Vector(std::slice::from_ref(&z)),
                    p
                )
                .unwrap(),
                0
            );
            assert_eq!(
                unit_deviation_upper(std::slice::from_ref(&one), p).unwrap(),
                0
            );
            assert!(unit(std::slice::from_ref(&z), p).is_err());
            assert!(backward_error(&[], 0, &[], RightHandSide::Vector(&[]), p).is_err());
            assert!(backward_error(
                std::slice::from_ref(&one),
                1,
                std::slice::from_ref(&one),
                RightHandSide::Vector(&[]),
                p
            )
            .is_err());
            assert!(backward_error(
                &[Float::with_val(p, f64::NAN)],
                1,
                std::slice::from_ref(&one),
                RightHandSide::Eigenvalue(&one),
                p
            )
            .is_err());
            // Entries outside the requested leading block do not affect its norm.
            let a = vec![
                one.clone(),
                Float::with_val(p, 1) << 700_000_000i32,
                z.clone(),
                one.clone(),
            ];
            assert_eq!(
                backward_error(
                    &a,
                    2,
                    std::slice::from_ref(&one),
                    RightHandSide::Eigenvalue(&one),
                    p
                )
                .unwrap(),
                0
            );
        }
    }
}
