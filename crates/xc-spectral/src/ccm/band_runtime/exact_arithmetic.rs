//! Exact contraction of finite binary points, rounded once at the end.
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{Assign, Float};

const BLOCK: usize = 512;
const MAX_BITS: i64 = 8_000_000;

fn scale(x: &Float, exponent: i64) -> Result<Float> {
    let mut out = x.clone();
    if out.is_zero() {
        return Ok(out);
    }
    let shift = u32::try_from(exponent.unsigned_abs())?;
    if exponent >= 0 {
        out <<= shift;
    } else {
        out >>= shift;
    }
    if !out.is_finite() || out.is_zero() {
        bail!("band contraction output exceeds exponent range");
    }
    let mut reverse = out.clone();
    if exponent >= 0 {
        reverse >>= shift;
    } else {
        reverse <<= shift;
    }
    if reverse != *x {
        bail!("band contraction binary scaling lost precision");
    }
    Ok(out)
}

/// For each nonzero triple, let E_i be the sum of its binary exponents and
/// L_i = E_i - sum(input precisions). Its exact product is a multiple of 2^L_i
/// and has magnitude below 2^E_i. After scaling by 2^-max(E_i), every partial
/// sum fits in max(E_i)-min(L_i)+ceil(log2(n))+2 bits. Thus all products and
/// reductions below are exact, including cancellation. Fixed block order is
/// retained, but neither scheduling nor block boundaries affect the result.
pub(crate) fn inner(a: &[Float], b: &[Float], w: &[Float], p: u32, bytes: u64) -> Result<Float> {
    if !(64..=1_000_000).contains(&p) {
        bail!("unsupported contraction output precision");
    }
    let (sum, exponent) = exact_contraction(a, b, w, p, bytes)?;
    scale(&Float::with_val(p, sum), exponent)
}

fn exact_contraction(
    a: &[Float],
    b: &[Float],
    w: &[Float],
    p: u32,
    bytes: u64,
) -> Result<(Float, i64)> {
    if a.len() != b.len()
        || a.len() != w.len()
        || !(64..=1_000_032).contains(&p)
        || a.iter()
            .chain(b)
            .chain(w)
            .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid finite band contraction shape or precision");
    }
    let mut high = i64::MIN;
    let mut low = i64::MAX;
    let mut count = 0u64;
    for ((a, b), w) in a.iter().zip(b).zip(w) {
        if let (Some(ea), Some(eb), Some(ew)) = (a.get_exp(), b.get_exp(), w.get_exp()) {
            let e = i64::from(ea) + i64::from(eb) + i64::from(ew);
            high = high.max(e);
            low = low.min(e - i64::from(a.prec()) - i64::from(b.prec()) - i64::from(w.prec()));
            count += 1;
        }
    }
    if count == 0 {
        return Ok((Float::with_val(p, 0), 0));
    }
    let carry = i64::from(u64::BITS - (count - 1).leading_zeros());
    let bits = high - low + carry + 2;
    if bits > MAX_BITS {
        bail!("band contraction exceeds eight-million-bit exact span budget");
    }
    let bits = u32::try_from(bits)?;
    let blocks = a.len().div_ceil(BLOCK) as u64;
    // Block results plus per-worker promoted inputs, products and scaling copies.
    // This bounds arithmetic buffers, not the entire process or Rayon runtime.
    let needed = (u64::from(bits).div_ceil(8) + 64).saturating_mul(
        blocks
            .saturating_add(
                // At most one active workspace per block, independent of the host pool.
                blocks.saturating_mul(12),
            )
            .saturating_add(8),
    );
    if needed > bytes {
        bail!("band contraction exact workspace budget exceeded");
    }
    let sums = a
        .par_chunks(BLOCK)
        .zip(b.par_chunks(BLOCK))
        .zip(w.par_chunks(BLOCK))
        .map(|((a, b), w)| -> Result<Float> {
            let mut sum = Float::with_val(bits, 0);
            let mut term = Float::with_val(bits, 0);
            for ((a, b), w) in a.iter().zip(b).zip(w) {
                let (Some(ea), Some(eb), Some(ew)) = (a.get_exp(), b.get_exp(), w.get_exp()) else {
                    continue;
                };
                term.assign(a);
                term >>= ea;
                let mut right = Float::with_val(bits, b);
                right >>= eb;
                term *= &right;
                right.assign(w);
                right >>= ew;
                term *= &right;
                term <<= i32::try_from(i64::from(ea) + i64::from(eb) + i64::from(ew) - high)?;
                sum += &term;
            }
            Ok(sum)
        })
        .collect::<Result<Vec<_>>>()?;
    let sum = sums
        .into_iter()
        .fold(Float::with_val(bits, 0), |s, x| s + x);
    Ok((sum, high))
}

/// Correctly rounded finite quadratic quotient in the actual binary inputs.
/// Exact numerator and denominator stay normalized until one final division.
/// Also return the correctly rounded signed physical component, without first
/// rounding the unsigned quotient. The independent reported component sum is
/// a separate point diagnostic.
pub(crate) fn rayleigh(
    matrix: &[Float],
    vector: &[Float],
    p: u32,
    coefficient: i32,
    bytes: u64,
) -> Result<(Float, Float)> {
    let n = vector.len();
    if n == 0
        || n.checked_mul(n) != Some(matrix.len())
        || !(64..=1_000_000).contains(&p)
        || coefficient == 0
        || matrix
            .iter()
            .chain(vector)
            .any(|x| !x.is_finite() || x.prec() > 1_000_000)
    {
        bail!("invalid finite quadratic form shape, points, precision or coefficient");
    }
    let work = matrix
        .iter()
        .chain(vector)
        .map(Float::prec)
        .max()
        .unwrap()
        .max(p);
    let fixed = (matrix.len() as u64)
        .saturating_mul(2)
        .saturating_add(n as u64)
        .saturating_mul(u64::from(work).div_ceil(8) + 64);
    // Retain both exact sums, denominator operands and signed-numerator copies
    // in addition to the contraction's own per-worker arithmetic accounting.
    let retained = (MAX_BITS as u64 / 8 + 128).saturating_mul(8);
    if fixed.saturating_add(retained) >= bytes {
        bail!("finite quadratic form workspace budget exceeded");
    }
    let scratch = bytes - fixed - retained;
    let one = vec![Float::with_val(work, 1); n];
    let (den, de) = exact_contraction(vector, vector, &one, work, scratch)?;
    if den <= 0 {
        bail!("finite quadratic form requires a nonzero vector");
    }
    let left = (0..matrix.len())
        .map(|j| vector[j / n].clone())
        .collect::<Vec<_>>();
    let right = (0..matrix.len())
        .map(|j| vector[j % n].clone())
        .collect::<Vec<_>>();
    let (num, ne) = exact_contraction(&left, &right, matrix, work, scratch)?;
    let signed = Float::with_val(num.prec() + 32, &num * coefficient);
    let value = scale(&Float::with_val(p, &num / &den), ne - de)?;
    let contribution = scale(&Float::with_val(p, &signed / &den), ne - de)?;
    Ok((value, contribution))
}

/// Newton identities for all roots of the finite polynomial in its actual
/// stored coefficients. This avoids conditioning losses from rounded root
/// midpoints. Real-root coverage and model qualifications remain with callers.
#[cfg(any(feature = "arb", test))]
pub(crate) fn polynomial_inverse_moments(
    coefficients: &[Float],
    p: u32,
    bytes: u64,
) -> Result<[Float; 3]> {
    if coefficients.is_empty()
        || coefficients[0].is_zero()
        || !(64..=1_000_000).contains(&p)
        || coefficients.iter().any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!(
            "finite polynomial inverse moments require finite coefficients and a nonzero constant"
        );
    }
    let retained = (MAX_BITS as u64 / 8 + 128) * 8;
    if bytes <= retained {
        bail!("finite polynomial inverse-moment workspace budget exceeded");
    }
    let work = p + 2;
    let exponent = coefficients
        .iter()
        .take(4)
        .filter_map(Float::get_exp)
        .max()
        .unwrap();
    let c = (0..4)
        .map(|j| {
            let value = coefficients
                .get(j)
                .map_or_else(|| Float::with_val(work, 0), |x| Float::with_val(work, x));
            scale(&value, -i64::from(exponent))
        })
        .collect::<Result<Vec<_>>>()?;
    let one = Float::with_val(work, 1);
    let ratio = |a: &[Float], b: &[Float], w: &[Float], den: [Float; 3]| -> Result<Float> {
        let (num, ne) = exact_contraction(a, b, w, work, bytes - retained)?;
        let (den, de) =
            exact_contraction(&den[..1], &den[1..2], &den[2..], work, bytes - retained)?;
        if den.is_zero() {
            bail!("polynomial inverse-moment denominator is zero");
        }
        scale(&Float::with_val(p, &num / &den), ne - de)
    };
    let first = ratio(
        &[-c[1].clone()],
        std::slice::from_ref(&one),
        std::slice::from_ref(&one),
        [c[0].clone(), one.clone(), one.clone()],
    )?;
    let second = ratio(
        &[c[1].clone(), -c[0].clone() * 2u32],
        &[c[1].clone(), c[2].clone()],
        &[one.clone(), one.clone()],
        [c[0].clone(), c[0].clone(), one.clone()],
    )?;
    let third = ratio(
        &[-c[1].clone(), c[0].clone() * 3u32, -c[0].clone() * 3u32],
        &[c[1].clone(), c[1].clone(), c[0].clone()],
        &[c[1].clone(), c[2].clone(), c[3].clone()],
        [c[0].clone(), c[0].clone(), c[0].clone()],
    )?;
    Ok([first, second, third])
}

/// Finite sums over the computed model-root points. A relative zero guard is a
/// resolution heuristic, not a root enclosure or a certificate of separation.
pub(crate) fn inverse_moments(roots: &[Float], p: u32, bytes: u64) -> Result<[Float; 3]> {
    use super::super::retained_evidence::finite_math::{
        abs, narrow, scale as scale_interval, scale_float,
    };
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    if roots.is_empty()
        || roots
            .iter()
            .any(|x| !x.is_finite() || x.is_zero() || x.prec() > p)
    {
        bail!("a model root is zero or nonfinite");
    }
    let maximum = roots
        .iter()
        .map(|x| x.clone().abs())
        .max_by(Float::total_cmp)
        .unwrap();
    let guard = Float::with_val(p, 1) >> (p - 32);
    if roots.iter().any(|x| x.clone().abs() / &maximum <= guard) {
        bail!("a model root is unresolved relative to the model coordinate scale");
    }
    let exponent = i64::from(maximum.get_exp().unwrap());
    for extra in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p + extra;
        if (u64::from(work).div_ceil(8) + 64).saturating_mul(96) > bytes {
            bail!("band inverse-moment workspace budget exceeded");
        }
        let mut sums = std::array::from_fn::<_, 3, _>(|_| I::from_i64(0, work));
        let mut absolute = sums.clone();
        for x in roots {
            let normalized = I::from_float(&scale_float(x, -exponent, work)?, work)?;
            let inverse = I::from_i64(1, work).div(&normalized)?;
            let mut term = I::from_i64(1, work);
            for j in 0..3 {
                term = term.mul(&inverse);
                sums[j] = sums[j].add(&term);
                absolute[j] = absolute[j].add(&abs(&term)?);
            }
        }
        if (0..3)
            .map(|j| narrow(std::slice::from_ref(&sums[j]), absolute[j].upper(), p))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .all(|x| *x)
        {
            let mut result = std::array::from_fn(|_| Float::with_val(p, 0));
            for j in 0..3 {
                let value = scale_interval(&sums[j], -exponent * (j as i64 + 1))?;
                result[j] = Float::with_val(p, value.midpoint_point().lower());
                if !result[j].is_finite() {
                    bail!("band inverse moment exceeds exponent range");
                }
            }
            return Ok(result);
        }
    }
    bail!("band inverse-moment arithmetic unresolved within 4096 guard bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::{ops::Pow, Integer, Rational};
    #[test]
    fn exact_triples_match_independent_rationals_and_worker_counts() {
        for p in [64, 128, 256] {
            for count in [1usize, 17, 513, 1031] {
                let mut a = Vec::new();
                let mut b = Vec::new();
                let mut w = Vec::new();
                let mut expected = Rational::new();
                for j in 0..count {
                    let av = Float::with_val(p, (j as i32 % 13) - 6) << ((j % 57) as i32 - 28);
                    let bv = Float::with_val(p, (j as i32 % 17) - 8) << ((j % 63) as i32 - 31);
                    let wv = Float::with_val(p, (j as i32 % 19) - 9) << ((j % 71) as i32 - 35);
                    expected += av.to_rational().unwrap()
                        * bv.to_rational().unwrap()
                        * wv.to_rational().unwrap();
                    a.push(av);
                    b.push(bv);
                    w.push(wv);
                }
                for workers in [1, 2, 3] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap();
                    assert_eq!(
                        pool.install(|| inner(&a, &b, &w, p, 1 << 28)).unwrap(),
                        Float::with_val(p, &expected)
                    );
                }
            }
            let big = Float::with_val(p, Integer::from(1) << 3333);
            assert_eq!(
                inner(
                    &[big.clone(), Float::with_val(p, 1), -big],
                    &vec![Float::with_val(p, 1); 3],
                    &vec![Float::with_val(p, 1); 3],
                    p,
                    1 << 28
                )
                .unwrap(),
                1
            );
        }
    }
    #[test]
    fn compensating_exponents_and_explicit_limits() {
        let p = 128;
        for shift in [-600_000_000, 600_000_000] {
            let a = Float::with_val(p, 3) << shift;
            let b = Float::with_val(p, 5) << shift;
            let w = Float::with_val(p, 7) >> shift;
            assert_eq!(
                inner(&[a], &[b], &[w], p, 1 << 20).unwrap(),
                Float::with_val(p, 105) << shift
            );
        }
        let a = vec![Float::with_val(p, 1), Float::with_val(p, 1) << 8_000_001];
        assert!(inner(&a, &a, &a, p, 1 << 28)
            .unwrap_err()
            .to_string()
            .contains("span"));
        assert!(inner(&a[..1], &a, &a, p, 1 << 28).is_err());
        assert!(inner(&a[..1], &a[..1], &a[..1], p, 1)
            .unwrap_err()
            .to_string()
            .contains("workspace"));
        let huge = vec![Float::with_val(p, 1) << 600_000_000];
        assert!(inner(&huge, &huge, &huge, p, 1 << 28)
            .unwrap_err()
            .to_string()
            .contains("exponent"));
        assert!(inner(
            &[Float::with_val(p, f64::NAN)],
            &a[..1],
            &a[..1],
            p,
            1 << 28
        )
        .is_err());
    }
    #[test]
    fn exact_quadratic_quotients_match_independent_rational_values() {
        let mut count = 0;
        for p in [64, 128, 256] {
            for n in 1usize..=5 {
                for case in 0i32..17 {
                    let v = (0..n)
                        .map(|j| Float::with_val(p, (j as i32 + 1) * (case - 7) + 3) / 8u32)
                        .collect::<Vec<_>>();
                    if v.iter().all(Float::is_zero) {
                        continue;
                    }
                    let a = (0..n * n)
                        .map(|j| Float::with_val(p, ((j as i32 * 7 + case) % 19) - 9) / 16u32)
                        .collect::<Vec<_>>();
                    let denominator = v
                        .iter()
                        .map(|x| x.to_rational().unwrap().square())
                        .sum::<Rational>();
                    let numerator = (0..n * n)
                        .map(|j| {
                            a[j].to_rational().unwrap()
                                * v[j / n].to_rational().unwrap()
                                * v[j % n].to_rational().unwrap()
                        })
                        .sum::<Rational>();
                    let expected = numerator / denominator;
                    for sign in [-3, 1, 7] {
                        let (value, contribution) = rayleigh(&a, &v, p, sign, 1 << 28).unwrap();
                        assert_eq!(value, Float::with_val(p, &expected));
                        assert_eq!(contribution, Float::with_val(p, expected.clone() * sign));
                        count += 1;
                    }
                }
            }
        }
        assert!(count > 700);
    }
    #[test]
    fn exact_quadratic_values_survive_compensating_scales_and_validate_domains() {
        let p = 128;
        for vector_scale in [-700_000_000, 0, 700_000_000] {
            for matrix_scale in [-300_000_000, 0, 300_000_000] {
                let matrix = vec![Float::with_val(p, 3) << matrix_scale];
                let vector = vec![Float::with_val(p, 7) << vector_scale];
                let (value, signed) = rayleigh(&matrix, &vector, p, -3, 1 << 28).unwrap();
                assert_eq!(value, Float::with_val(p, 3) << matrix_scale);
                assert_eq!(signed, Float::with_val(p, -9) << matrix_scale);
            }
        }
        assert!(rayleigh(
            &[Float::with_val(p, f64::NAN)],
            &[Float::with_val(p, 1)],
            p,
            1,
            1 << 28
        )
        .is_err());
        assert!(rayleigh(
            &[Float::with_val(p, 1)],
            &[Float::with_val(p, 0)],
            p,
            1,
            1 << 28
        )
        .is_err());
        assert!(rayleigh(&[Float::with_val(p, 1)], &[Float::with_val(p, 1)], p, 1, 1).is_err());
        // Input points may have more precision than the requested result.
        let a: Float = Float::with_val(256, 1) + (Float::with_val(256, 1) >> 100u32);
        assert_eq!(
            rayleigh(
                std::slice::from_ref(&a),
                &[Float::with_val(256, 1)],
                64,
                1,
                1 << 28
            )
            .unwrap()
            .0,
            Float::with_val(64, a)
        );
    }
    #[test]
    fn coefficient_newton_sums_match_independent_factor_roots_exactly() {
        let mut checks = 0;
        for p in [64, 128, 256] {
            for n in 1usize..=6 {
                for case in 0i32..13 {
                    let roots = (0..n)
                        .map(|j| {
                            let k = ((case * 3 + j as i32 * 5) % 19) - 9;
                            Rational::from((if k == 0 { 1 } else { k }, 4))
                        })
                        .collect::<Vec<_>>();
                    let mut co = vec![Rational::from(1)];
                    for root in &roots {
                        let mut next = vec![Rational::new(); co.len() + 1];
                        for (j, c) in co.iter().enumerate() {
                            next[j] -= c.clone() * root;
                            next[j + 1] += c;
                        }
                        co = next;
                    }
                    let expected = (1u32..=3)
                        .map(|k| {
                            roots
                                .iter()
                                .map(|r| r.clone().recip().pow(k))
                                .sum::<Rational>()
                        })
                        .collect::<Vec<_>>();
                    for scale in [-700_000_000i32, 0, 700_000_000] {
                        let coefficients = co
                            .iter()
                            .map(|c| Float::with_val(p, c) << scale)
                            .collect::<Vec<_>>();
                        let result = polynomial_inverse_moments(&coefficients, p, 1 << 28).unwrap();
                        for j in 0..3 {
                            assert_eq!(result[j], Float::with_val(p, &expected[j]));
                            checks += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checks, 2106);
    }
    #[test]
    fn coefficient_newton_sums_keep_large_odd_cancellation_and_fail_explicitly() {
        for p in [64, 128, 256] {
            let small = Float::with_val(p, 1) >> 6000u32;
            let co = vec![
                -small.clone(),
                -small,
                Float::with_val(p, 1),
                Float::with_val(p, 1),
            ];
            let sums = polynomial_inverse_moments(&co, p, 1 << 28).unwrap();
            assert_eq!(sums[0], -1);
            assert_eq!(sums[2], -1);
            assert_eq!(
                sums[1],
                Float::with_val(p, 1) + (Float::with_val(p, 1) << 6001u32)
            );
            assert!(polynomial_inverse_moments(
                &[Float::with_val(p, 0), Float::with_val(p, 1)],
                p,
                1 << 28
            )
            .is_err());
            assert!(
                polynomial_inverse_moments(&[Float::with_val(p, f64::NAN)], p, 1 << 28).is_err()
            );
            assert!(polynomial_inverse_moments(&co, p, 1).is_err());
        }
    }
    #[test]
    fn exact_contraction_admission_does_not_depend_on_host_threads() {
        let values = vec![Float::with_val(128, 1); 1025];
        for budget in [1, 4000, 8000, 1 << 20] {
            let calculate = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| inner(&values, &values, &values, 128, budget))
                    .map(|x| x.to_string())
            };
            let single = calculate(1);
            let parallel = calculate(4);
            assert_eq!(single.is_ok(), parallel.is_ok());
            if let Ok(single) = single {
                assert_eq!(single, parallel.unwrap());
            }
        }
    }
}
