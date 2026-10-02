//! Full finite stored-point Gram and similarity checks, with directed arithmetic.
use super::super::retained_evidence::finite_math::{abs, scale_float};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

#[derive(Clone, Debug)]
pub(super) struct ValidationBounds {
    #[cfg(test)]
    pub gram_defect_upper: Float,
    #[cfg(test)]
    pub normalized_residual_norm_upper: Float,
    #[cfg(test)]
    pub normalized_tridiagonal_norm_upper: Float,
    #[cfg(test)]
    pub matrix_binary_exponent: i64,
    pub eigenvalue_allowance: Float,
}

// Successful proofs are pure functions of these exact points and arithmetic
// environment. Keep only small proof results, never source matrices, and never
// retain failures. A different byte of the source requires a fresh proof.
std::thread_local! {
    static CHECKED_TRANSFORMS: std::cell::RefCell<Vec<(xc_cache::ContentDigest, ValidationBounds)>> = const { std::cell::RefCell::new(Vec::new()) };
    #[cfg(test)]
    static FULL_TRANSFORM_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    #[cfg(test)]
    pub(super) static SANITY_TRANSFORM_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn source_digest(
    namespace: &[u8],
    groups: &[&[Float]],
    p: u32,
) -> Result<xc_cache::ContentDigest> {
    use sha2::{Digest, Sha256};
    if !(64..=1_000_000).contains(&p) {
        bail!("invalid validation precision");
    }
    let mut hash = Sha256::new();
    hash.update(namespace.len().to_le_bytes());
    hash.update(namespace);
    hash.update(p.to_le_bytes());
    hash.update(rug::float::exp_min().to_le_bytes());
    hash.update(rug::float::exp_max().to_le_bytes());
    hash.update(groups.len().to_le_bytes());
    for group in groups {
        hash.update(group.len().to_le_bytes());
        for x in *group {
            if !x.is_finite() || x.prec() > p {
                bail!("nonfinite or overprecision validation source");
            }
            hash.update(x.prec().to_le_bytes());
            let text = x.to_string_radix(16, None);
            hash.update(text.len().to_le_bytes());
            hash.update(text.as_bytes());
        }
    }
    Ok(xc_cache::ContentDigest(format!("{:x}", hash.finalize())))
}

#[cfg(test)]
pub(super) fn validate(
    matrix: &[Float],
    diagonal: &[Float],
    off_diagonal: &[Float],
    basis: &[Float],
    n: usize,
    p: u32,
) -> std::result::Result<(), String> {
    bounds(matrix, diagonal, off_diagonal, basis, n, p).map(|_| ())
}

pub(super) fn bounds(
    matrix: &[Float],
    diagonal: &[Float],
    off_diagonal: &[Float],
    basis: &[Float],
    n: usize,
    p: u32,
) -> std::result::Result<ValidationBounds, String> {
    if n == 0
        || n.checked_mul(n) != Some(matrix.len())
        || basis.len() != matrix.len()
        || diagonal.len() != n
        || off_diagonal.len() != n - 1
    {
        return Err("invalid finite sector-transform dimensions".into());
    }
    let namespace = format!("full-sector-transform-v1:{n}");
    let digest = source_digest(
        namespace.as_bytes(),
        &[matrix, diagonal, off_diagonal, basis],
        p,
    )
    .map_err(|e| format!("{e:#}"))?;
    if let Some(checked) = CHECKED_TRANSFORMS.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(key, _)| key == &digest)
            .map(|(_, value)| value.clone())
    }) {
        return Ok(checked);
    }
    let checked =
        check(matrix, diagonal, off_diagonal, basis, n, p).map_err(|e| format!("{e:#}"))?;
    CHECKED_TRANSFORMS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= 8 {
            cache.remove(0);
        }
        cache.push((digest, checked.clone()));
    });
    Ok(checked)
}

/// Ordinary computed-reuse admission of a retained transform whose full proof
/// recorded `allowance` when it was produced. It checks the shape, finite
/// points, exact stored symmetry, the unit-column domain and, for three fixed
/// columns, the complete Gram column and the complete A Q - Q T column at the
/// same directed tolerances as the full proof. Cost is O(n^2). It detects
/// storage and association errors; it is not a proof of the allowance, which
/// explicit verification replays with [`bounds`].
pub(super) fn sanity(
    a: &[Float],
    d: &[Float],
    e: &[Float],
    q: &[Float],
    n: usize,
    p: u32,
    allowance: &Float,
) -> Result<()> {
    #[cfg(test)]
    SANITY_TRANSFORM_CHECKS.with(|count| count.set(count.get() + 1));
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || a.len() != q.len()
        || d.len() != n
        || e.len() != n - 1
        || !(64..=1_000_000).contains(&p)
        || a.iter()
            .chain(d)
            .chain(e)
            .chain(q)
            .any(|x| !x.is_finite() || x.prec() > p)
        || !allowance.is_finite()
        || allowance.is_sign_negative()
    {
        bail!("invalid retained sector-transform dimensions, points, precision or allowance");
    }
    for i in 0..n {
        for j in i + 1..n {
            if a[i * n + j] != a[j * n + i] {
                bail!("retained sector transform requires exact stored matrix symmetry");
            }
        }
    }
    if q.iter().any(|x| x.clone().abs() > 2) {
        bail!("sector-transform basis exceeds the unit-column domain");
    }
    let work = p + 64;
    let epsilon = Float::with_val(work, 1) >> p.saturating_sub(64).max(1);
    let mut probes = vec![0, n / 2, n - 1];
    probes.dedup();
    let column = |j: usize| -> Result<Vec<I>> {
        (0..n)
            .map(|i| Ok(I::from_float(&q[i * n + j], work)?))
            .collect()
    };
    for &j in &probes {
        let qj = column(j)?;
        let gram = (0..n)
            .into_par_iter()
            .map(|k| -> Result<Float> {
                let mut dot = I::from_i64(0, work);
                for (i, value) in qj.iter().enumerate() {
                    dot = dot.add(&value.mul(&I::from_float(&q[i * n + k], work)?));
                }
                Ok(abs(&dot.sub(&I::from_i64(i64::from(j == k), work)))?
                    .upper()
                    .clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut bound = Float::with_val(work, 0);
        for value in gram {
            bound = Float::with_val_round(work, &bound + &value, Round::Up).0;
        }
        if !bound.is_finite() || bound > epsilon {
            bail!(
                "retained sector-transform Gram column {j} exceeds requested-precision tolerance"
            );
        }
    }
    let exponent = a
        .iter()
        .chain(d)
        .chain(e)
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let scaled =
        |x: &Float| -> Result<I> { Ok(I::from_float(&scale_float(x, -exponent, work)?, work)?) };
    let mut matrix_norm_lower = Float::with_val(work, 0);
    for row in a.chunks(n) {
        let mut sum = Float::with_val(work, 0);
        for x in row {
            sum = Float::with_val_round(work, &sum + abs(&scaled(x)?)?.lower(), Round::Down).0;
        }
        if sum > matrix_norm_lower {
            matrix_norm_lower = sum;
        }
    }
    let threshold = Float::with_val_round(work, &epsilon * &matrix_norm_lower, Round::Down).0;
    for &j in &probes {
        let qj = column(j)?;
        let neighbors = [(j > 0).then(|| j - 1), (j + 1 < n).then_some(j + 1)];
        let dj = scaled(&d[j])?;
        let left_e = if j > 0 {
            Some(scaled(&e[j - 1])?)
        } else {
            None
        };
        let right_e = if j + 1 < n {
            Some(scaled(&e[j])?)
        } else {
            None
        };
        (0..n).into_par_iter().try_for_each(|i| -> Result<()> {
            let mut left = I::from_i64(0, work);
            for (k, value) in qj.iter().enumerate() {
                left = left.add(&scaled(&a[i * n + k])?.mul(value));
            }
            let mut right = I::from_float(&q[i * n + j], work)?.mul(&dj);
            if let (Some(k), Some(coupling)) = (neighbors[0], &left_e) {
                right = right.add(&I::from_float(&q[i * n + k], work)?.mul(coupling));
            }
            if let (Some(k), Some(coupling)) = (neighbors[1], &right_e) {
                right = right.add(&I::from_float(&q[i * n + k], work)?.mul(coupling));
            }
            let entry = abs(&left.sub(&right))?;
            if !entry.upper().is_finite() || entry.upper() > &threshold {
                bail!("retained A Q = Q T entry ({i}, {j}) exceeds scale-relative requested-precision tolerance");
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn check(
    a: &[Float],
    d: &[Float],
    e: &[Float],
    q: &[Float],
    n: usize,
    p: u32,
) -> Result<ValidationBounds> {
    #[cfg(test)]
    FULL_TRANSFORM_CHECKS.with(|count| count.set(count.get() + 1));
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || a.len() != q.len()
        || d.len() != n
        || e.len() != n - 1
        || !(64..=1_000_000).contains(&p)
        || a.iter()
            .chain(d)
            .chain(e)
            .chain(q)
            .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid finite sector-transform dimensions, points or precision");
    }
    for i in 0..n {
        for j in i + 1..n {
            if a[i * n + j] != a[j * n + i] {
                bail!("source eigenvalue allowance requires exact stored matrix symmetry");
            }
        }
    }
    // A column with an entry of magnitude >2 cannot satisfy the Gram bound.
    if q.iter().any(|x| x.clone().abs() > 2) {
        bail!("sector-transform basis exceeds the unit-column domain");
    }
    let work = p + 64;
    let buffers = (a.len() as u64)
        .saturating_mul(4)
        .saturating_add((n as u64).saturating_mul(6))
        .saturating_add((rayon::current_num_threads() as u64).saturating_mul(192));
    if buffers.saturating_mul(u64::from(work).div_ceil(8) + 64) as u128
        > super::source_working_budget()?
    {
        bail!("sector-transform validation exceeds numerical workspace budget");
    }
    let epsilon = Float::with_val(work, 1) >> p.saturating_sub(64).max(1);
    let q = q
        .iter()
        .map(|x| Ok(I::from_float(x, work)?))
        .collect::<Result<Vec<_>>>()?;
    // Bound the full induced infinity norm of Q^T Q-I, not just adjacent
    // products. Epsilon <=1/2 also excludes rank-deficient accepted bases.
    let gram_rows = (0..n)
        .into_par_iter()
        .map(|j| -> Result<Float> {
            let mut row_bound = Float::with_val(work, 0);
            for k in 0..n {
                let mut dot = I::from_i64(0, work);
                for i in 0..n {
                    dot = dot.add(&q[i * n + j].mul(&q[i * n + k]));
                }
                let error = abs(&dot.sub(&I::from_i64(i64::from(j == k), work)))?;
                row_bound = Float::with_val_round(work, &row_bound + error.upper(), Round::Up).0;
                if !row_bound.is_finite() || row_bound > epsilon {
                    bail!("full basis Gram row {j} exceeds requested-precision tolerance");
                }
            }
            Ok(row_bound)
        })
        .collect::<Result<Vec<_>>>()?;
    let gram_defect_upper = gram_rows.into_iter().max_by(Float::total_cmp).unwrap();
    // Normalize A and T together by an exact binary power. Never insert a
    // coordinate-dependent unit floor into this homogeneous comparison.
    let exponent = a
        .iter()
        .chain(d)
        .chain(e)
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = |xs: &[Float]| -> Result<Vec<I>> {
        xs.iter()
            .map(|x| Ok(I::from_float(&scale_float(x, -exponent, work)?, work)?))
            .collect()
    };
    let a = normalized(a)?;
    let d = normalized(d)?;
    let e = normalized(e)?;
    let mut matrix_norm_lower = Float::with_val(work, 0);
    for row in a.chunks(n) {
        let mut sum = Float::with_val(work, 0);
        for x in row {
            sum = Float::with_val_round(work, &sum + abs(x)?.lower(), Round::Down).0;
        }
        if sum > matrix_norm_lower {
            matrix_norm_lower = sum;
        }
    }
    let threshold = Float::with_val_round(work, &epsilon * &matrix_norm_lower, Round::Down).0;
    // Every residual entry contributes to an outward row-sum bound. An
    // accepted result establishes ||AQ-QT||_inf <= epsilon*||A||_inf for
    // these stored finite matrices, including A=0. This is not a continuum
    // eigenvalue certificate or proof of state selection by downstream code.
    let residual_rows=(0..n).into_par_iter().map(|i|->Result<Float> {
        let mut row_bound=Float::with_val(work,0);
        for j in 0..n {
            let mut left=I::from_i64(0,work);
            for k in 0..n {left=left.add(&a[i*n+k].mul(&q[k*n+j]));}
            let mut right=q[i*n+j].mul(&d[j]);
            if j>0 {right=right.add(&q[i*n+j-1].mul(&e[j-1]));}
            if j+1<n {right=right.add(&q[i*n+j+1].mul(&e[j]));}
            row_bound=Float::with_val_round(work,&row_bound+abs(&left.sub(&right))?.upper(),Round::Up).0;
            if !row_bound.is_finite() || row_bound>threshold
            {bail!("full A Q = Q T row {i} exceeds scale-relative requested-precision tolerance");}
        }
        Ok(row_bound)
    }).collect::<Result<Vec<_>>>()?;
    let normalized_residual_norm_upper =
        residual_rows.into_iter().max_by(Float::total_cmp).unwrap();
    let mut normalized_tridiagonal_norm_upper = Float::with_val(work, 0);
    for j in 0..n {
        let mut row = abs(&d[j])?;
        if j > 0 {
            row = row.add(&abs(&e[j - 1])?);
        }
        if j + 1 < n {
            row = row.add(&abs(&e[j])?);
        }
        if row.upper() > &normalized_tridiagonal_norm_upper {
            normalized_tridiagonal_norm_upper = row.upper().clone();
        }
    }
    // Q=U H, H=(Q^T Q)^(1/2): ||H-I||_2 <= eta and
    // ||H^-1||_2 <= 1/sqrt(1-eta). Here rho is the complete induced
    // infinity norm of R=AQ-QT. Thus ||R||_2 <= sqrt(n)*rho (use
    // ||R||_1 <= n*||R||_inf), not a maximum-entry estimate.
    // B=U^T A U satisfies B-T=((H-I)T-T(H-I)+U^T R)H^-1.
    // Weyl then encloses every ordered eigenvalue of stored symmetric A.
    let rho = I::from_float(&normalized_residual_norm_upper, work)?;
    let tau = I::from_float(&normalized_tridiagonal_norm_upper, work)?;
    let eta = I::from_float(&gram_defect_upper, work)?;
    let root_dimension = I::from_float(&Float::with_val(work, n), work)?.sqrt()?;
    let denominator = I::from_i64(1, work).sub(&eta).sqrt()?;
    let delta = rho
        .mul(&root_dimension)
        .add(&tau.mul(&eta).mul(&I::from_i64(2, work)))
        .div(&denominator)?;
    delta.validate()?;
    let eigenvalue_allowance = scale_float(delta.upper(), exponent, work)?;
    Ok(ValidationBounds {
        #[cfg(test)]
        gram_defect_upper,
        #[cfg(test)]
        normalized_residual_norm_upper,
        #[cfg(test)]
        normalized_tridiagonal_norm_upper,
        #[cfg(test)]
        matrix_binary_exponent: exponent,
        eigenvalue_allowance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;
    #[test]
    fn successful_transform_proof_reuses_only_identical_sources() {
        CHECKED_TRANSFORMS.with(|cache| cache.borrow_mut().clear());
        FULL_TRANSFORM_CHECKS.with(|count| count.set(0));
        let (a, d, e, q) = fixture(4, 3, 128);
        let first = bounds(&a, &d, &e, &q, 4, 128).unwrap();
        let second = bounds(&a, &d, &e, &q, 4, 128).unwrap();
        assert_eq!(first.eigenvalue_allowance, second.eigenvalue_allowance);
        assert_eq!(FULL_TRANSFORM_CHECKS.with(|count| count.get()), 1);
        assert!(bounds(&a, &d, &e, &q, 3, 128).is_err());
        let mut wrong = q.clone();
        wrong[0] = Float::with_val(128, 3);
        for expected in [2, 3] {
            assert!(bounds(&a, &d, &e, &wrong, 4, 128).is_err());
            assert_eq!(FULL_TRANSFORM_CHECKS.with(|count| count.get()), expected);
        }
        bounds(&a, &d, &e, &q, 4, 160).unwrap();
        assert_eq!(FULL_TRANSFORM_CHECKS.with(|count| count.get()), 4);
    }
    #[test]
    fn retained_sanity_probes_columns_without_the_full_proof() {
        let p = 128;
        for n in [1, 2, 5, 8] {
            let (a, d, e, q) = fixture(n, 2, p);
            let allowance = check(&a, &d, &e, &q, n, p).unwrap().eigenvalue_allowance;
            FULL_TRANSFORM_CHECKS.with(|count| count.set(0));
            sanity(&a, &d, &e, &q, n, p, &allowance).unwrap();
            assert_eq!(FULL_TRANSFORM_CHECKS.with(|count| count.get()), 0);
            assert!(sanity(&a, &d, &e, &q, n, p, &Float::with_val(p, -1)).is_err());
            if n > 1 {
                // A corrupted probed column fails its Gram or similarity row.
                let mut wrong = q.clone();
                wrong[n - 1] += Float::with_val(p, 1) >> 20;
                assert!(sanity(&a, &d, &e, &wrong, n, p, &allowance).is_err());
                let mut asymmetric = a.clone();
                asymmetric[1] += 1;
                assert!(sanity(&asymmetric, &d, &e, &q, n, p, &allowance).is_err());
            }
        }
        // Sanity is not a proof: a column outside the probe set is invisible
        // to it, while the full replay rejects the same basis.
        let (n, p) = (8, 128);
        let (a, d, e, q) = fixture(n, 2, p);
        let allowance = check(&a, &d, &e, &q, n, p).unwrap().eigenvalue_allowance;
        let mut unsampled = q.clone();
        for row in 0..n {
            unsampled[row * n + 2] *= 2;
        }
        assert!(check(&a, &d, &e, &unsampled, n, p).is_err());
        assert!(sanity(&a, &d, &e, &unsampled, n, p, &allowance).is_ok());
    }

    fn fixture(n: usize, case: usize, p: u32) -> (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
        let mut q = (0..n * n)
            .map(|j| Rational::from(i32::from(j / n == j % n)))
            .collect::<Vec<_>>();
        // Exact products of rational Givens rotations: c^2+s^2=1.
        if n > 1 {
            for k in 0..n + case {
                let a = k % n;
                let b = (a + 1) % n;
                let c = Rational::from((3, 5));
                let s = Rational::from((if k % 2 == 0 { 4 } else { -4 }, 5));
                for row in 0..n {
                    let x = q[row * n + a].clone();
                    let y = q[row * n + b].clone();
                    q[row * n + a] = x.clone() * &c - y.clone() * &s;
                    q[row * n + b] = x * &s + y * &c;
                }
            }
        }
        let d = (0..n)
            .map(|j| Rational::from((j as i32 * 3 + case as i32) % 13 - 6))
            .collect::<Vec<_>>();
        let e = (0..n - 1)
            .map(|j| Rational::from((j as i32 + case as i32) % 5 - 2))
            .collect::<Vec<_>>();
        let mut a = vec![Rational::new(); n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    a[i * n + j] += q[i * n + k].clone() * &d[k] * &q[j * n + k];
                    if k + 1 < n {
                        a[i * n + j] += (q[i * n + k].clone() * &q[j * n + k + 1]
                            + q[i * n + k + 1].clone() * &q[j * n + k])
                            * &e[k];
                    }
                }
            }
        }
        let points = |v: Vec<Rational>| v.into_iter().map(|x| Float::with_val(p, x)).collect();
        (points(a), points(d), points(e), points(q))
    }
    #[test]
    fn full_transform_accepts_independent_exact_similarity_fixtures_at_all_scales() {
        let mut cases = 0;
        for p in [64, 128, 256] {
            for n in 1..=7 {
                for case in 0..8 {
                    let (a, d, e, q) = fixture(n, case, p);
                    for shift in [-500_000_000i32, 0, 500_000_000] {
                        let scaled =
                            |v: &[Float]| v.iter().map(|x| x.clone() << shift).collect::<Vec<_>>();
                        validate(&scaled(&a), &scaled(&d), &scaled(&e), &q, n, p).unwrap();
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 504);
    }
    #[test]
    fn full_transform_rejects_omitted_columns_duplicates_and_unit_floor_counterexamples() {
        let p = 128;
        let n = 6;
        let identity = (0..n * n)
            .map(|j| Float::with_val(p, i32::from(j / n == j % n)))
            .collect::<Vec<_>>();
        let one = vec![Float::with_val(p, 1); n];
        let zero = vec![Float::with_val(p, 0); n - 1];
        for first in 0..n {
            for second in first + 1..n {
                let mut bad = identity.clone();
                for row in 0..n {
                    bad[row * n + second] = bad[row * n + first].clone();
                }
                assert!(validate(&identity, &one, &zero, &bad, n, p).is_err());
            }
        }
        let d = (1..=n).map(|j| Float::with_val(p, j)).collect::<Vec<_>>();
        let a = (0..n * n)
            .map(|j| {
                if j / n == j % n {
                    d[j / n].clone()
                } else {
                    Float::with_val(p, 0)
                }
            })
            .collect::<Vec<_>>();
        for first in 0..n {
            for second in first + 1..n {
                let mut bad = d.clone();
                bad.swap(first, second);
                assert!(validate(&a, &bad, &zero, &identity, n, p).is_err());
            }
        }
        for shift in [-500_000_000i32, 0, 500_000_000] {
            let a = identity
                .iter()
                .map(|x| x.clone() << shift)
                .collect::<Vec<_>>();
            let d = vec![Float::with_val(p, 2) << shift; n];
            assert!(validate(&a, &d, &zero, &identity, n, p).is_err());
        }
    }
    #[test]
    fn full_transform_domains_zero_forms_and_exact_boundaries() {
        for p in [64, 128, 256] {
            let z = Float::with_val(p, 0);
            let one = Float::with_val(p, 1);
            let eps = Float::with_val(p, 1) >> p.saturating_sub(64).max(1);
            assert!(validate(
                std::slice::from_ref(&z),
                std::slice::from_ref(&z),
                &[],
                std::slice::from_ref(&one),
                1,
                p
            )
            .is_ok());
            let at = Float::with_val(p, &one + &eps);
            let outside = Float::with_val(p, &one + Float::with_val(p, &eps * 2));
            assert!(validate(
                std::slice::from_ref(&one),
                &[at],
                &[],
                std::slice::from_ref(&one),
                1,
                p
            )
            .is_ok());
            assert!(validate(
                std::slice::from_ref(&one),
                &[outside],
                &[],
                std::slice::from_ref(&one),
                1,
                p
            )
            .is_err());
            let inside = Float::with_val(p, &one + Float::with_val(p, &eps / 8));
            assert!(validate(
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &[],
                &[inside],
                1,
                p
            )
            .is_ok());
            assert!(validate(
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &[],
                &[Float::with_val(p, &one + &eps)],
                1,
                p
            )
            .is_err());
        }
        let x = Float::with_val(128, 1);
        for n in [0, 2, usize::MAX] {
            assert!(validate(
                std::slice::from_ref(&x),
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&x),
                n,
                128
            )
            .is_err());
        }
        for p in [0, 63, 1_000_001] {
            assert!(validate(
                std::slice::from_ref(&x),
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&x),
                1,
                p
            )
            .is_err());
        }
        for y in [
            Float::with_val(128, f64::NAN),
            Float::with_val(128, f64::INFINITY),
        ] {
            assert!(validate(
                std::slice::from_ref(&y),
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&x),
                1,
                128
            )
            .is_err());
            assert!(validate(
                std::slice::from_ref(&x),
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&y),
                1,
                128
            )
            .is_err());
        }
    }
}

#[cfg(test)]
mod allowance_tests {
    use super::*;
    use rug::Rational;
    #[test]
    fn source_allowance_covers_independent_closed_quadratic_spectra_and_binary_scales() {
        for p in [64, 128, 256] {
            let point = |a, b| Float::with_val(p, Rational::from((a, b)));
            let a = vec![point(6, 5), point(-12, 5), point(-12, 5), point(-1, 5)];
            let q = vec![point(3, 5), point(-4, 5), point(4, 5), point(3, 5)];
            let d = vec![Float::with_val(p, -2), Float::with_val(p, 3)];
            let e = vec![Float::with_val(p, 0)];
            let bound = bounds(&a, &d, &e, &q, 2, p).unwrap();
            assert!(
                bound.gram_defect_upper >= 0
                    && bound.normalized_residual_norm_upper >= 0
                    && bound.normalized_tridiagonal_norm_upper > 0
            );
            let work = p + 256;
            let x = I::from_float(&a[0], work).unwrap();
            let y = I::from_float(&a[1], work).unwrap();
            let z = I::from_float(&a[3], work).unwrap();
            let trace = x.add(&z);
            let difference = x.sub(&z);
            let radicand = difference
                .mul(&difference)
                .add(&y.mul(&y).mul(&I::from_i64(4, work)));
            let root = radicand.sqrt().unwrap();
            let eigenvalues = [
                trace.sub(&root).div(&I::from_i64(2, work)).unwrap(),
                trace.add(&root).div(&I::from_i64(2, work)).unwrap(),
            ];
            for (actual, target) in eigenvalues.iter().zip([-2, 3]) {
                assert!(
                    abs(&actual.sub(&I::from_i64(target, work)))
                        .unwrap()
                        .upper()
                        <= &bound.eigenvalue_allowance
                );
            }
            for shift in [-500_000_000i32, 500_000_000] {
                let scaled = |v: &[Float]| v.iter().map(|x| x.clone() << shift).collect::<Vec<_>>();
                let changed = bounds(&scaled(&a), &scaled(&d), &scaled(&e), &q, 2, p).unwrap();
                assert_eq!(
                    changed.eigenvalue_allowance,
                    bound.eigenvalue_allowance.clone() << shift
                );
                assert_eq!(changed.gram_defect_upper, bound.gram_defect_upper);
                assert_eq!(
                    changed.matrix_binary_exponent,
                    bound.matrix_binary_exponent + i64::from(shift)
                );
            }
        }
    }
    #[test]
    fn source_allowance_corrects_the_accepted_tridiagonal_gap_and_requires_symmetry() {
        let p = 128;
        let z = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        let delta = Float::with_val(p, 1) >> 80;
        let a = vec![z.clone(), z.clone(), z.clone(), one.clone()];
        let q = vec![one.clone(), z.clone(), z.clone(), one.clone()];
        let d = vec![z.clone(), Float::with_val(p, &one + &delta)];
        let b = bounds(&a, &d, std::slice::from_ref(&z), &q, 2, p).unwrap();
        assert!(b.eigenvalue_allowance >= delta);
        let corrected_gap =
            Float::with_val(p + 128, &d[1]) - Float::with_val(p + 128, &b.eigenvalue_allowance * 2);
        assert!(corrected_gap <= one);
        let exact = bounds(
            &a,
            &[z.clone(), one.clone()],
            std::slice::from_ref(&z),
            &q,
            2,
            p,
        )
        .unwrap();
        assert_eq!(exact.eigenvalue_allowance, 0);
        let mut asymmetric = a;
        asymmetric[1] = one;
        assert!(bounds(&asymmetric, &d, &[z], &q, 2, p).is_err());
    }
}
