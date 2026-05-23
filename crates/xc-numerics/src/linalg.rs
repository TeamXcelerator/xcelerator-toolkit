// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision dense linear algebra.
//!
//! - **`lu_factor` / `lu_solve`**: LU factorization with partial pivoting,
//!   followed by forward/back substitution. Parallelized via rayon over
//!   the Schur-complement update.
//! - **`inverse_iteration`**: Smallest-eigenpair finder via inverse
//!   iteration with the LU factorization cached for O(n²) per step.
//!   Optional even-symmetry projection at each step (for forced-even
//!   eigenvector extraction).
//! - **`normalize_l2` / `rayleigh_quotient`**: Standard helpers.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rug::{ops::Pow, Float};

/// Build an HP zero at the given precision. Uses an integer literal so
/// no f64 round-trip occurs.
#[inline] fn hp_zero(prec: u32) -> Float { Float::with_val(prec, 0) }

/// LU factorization with partial pivoting. The matrix `a` is stored row-major
/// with `dim` rows and columns. Returns the factored representation suitable
/// for `lu_solve`.
pub struct LuFactors {
    pub lu: Vec<Float>,
    pub perm: Vec<usize>,
}

pub fn lu_factor(a: &[Float], dim: usize) -> Result<LuFactors> {
    let mut lu: Vec<Float> = a.to_vec();
    let mut perm: Vec<usize> = (0..dim).collect();
    for k in 0..dim {
        let mut max_idx = k;
        let mut max_val = lu[k * dim + k].clone().abs();
        for i in (k + 1)..dim {
            let v = lu[i * dim + k].clone().abs();
            if v > max_val { max_val = v; max_idx = i; }
        }
        if max_idx != k {
            for j in 0..dim { lu.swap(k * dim + j, max_idx * dim + j); }
            perm.swap(k, max_idx);
        }
        let pivot = lu[k * dim + k].clone();
        if pivot.is_zero() { return Err(anyhow!("singular matrix")); }
        let factors: Vec<Float> = ((k + 1)..dim).map(|i| {
            let mut f = lu[i * dim + k].clone(); f /= &pivot; f
        }).collect();
        for (idx, i) in ((k + 1)..dim).enumerate() { lu[i * dim + k] = factors[idx].clone(); }
        let pivot_row: Vec<Float> = ((k + 1)..dim).map(|j| lu[k * dim + j].clone()).collect();
        let updates: Vec<Vec<Float>> = factors.par_iter().enumerate().map(|(idx, factor)| {
            let i = k + 1 + idx;
            ((k + 1)..dim).enumerate().map(|(j_off, j)| {
                let mut val = lu[i * dim + j].clone();
                let mut prod = pivot_row[j_off].clone(); prod *= factor;
                val -= &prod; val
            }).collect()
        }).collect();
        for (idx, i) in ((k + 1)..dim).enumerate() {
            for (j_off, j) in ((k + 1)..dim).enumerate() {
                lu[i * dim + j] = updates[idx][j_off].clone();
            }
        }
    }
    Ok(LuFactors { lu, perm })
}

/// Solve `A·x = b` given the LU factorization of `A` from `lu_factor`.
pub fn lu_solve(factors: &LuFactors, b: &[Float], dim: usize, prec: u32) -> Vec<Float> {
    let lu = &factors.lu;
    let perm = &factors.perm;
    let pb: Vec<Float> = (0..dim).map(|i| b[perm[i]].clone()).collect();
    let mut y = vec![hp_zero(prec); dim];
    for i in 0..dim {
        let mut s = pb[i].clone();
        for j in 0..i { let mut t = lu[i * dim + j].clone(); t *= &y[j]; s -= &t; }
        y[i] = s;
    }
    let mut x = vec![hp_zero(prec); dim];
    for i in (0..dim).rev() {
        let mut s = y[i].clone();
        for j in (i + 1)..dim { let mut t = lu[i * dim + j].clone(); t *= &x[j]; s -= &t; }
        s /= &lu[i * dim + i];
        x[i] = s;
    }
    x
}

/// In-place ℓ² normalization of an HP vector. Sum-of-squares is computed
/// via parallel reduction; the per-element divide is parallelized too.
pub fn normalize_l2(v: &mut [Float]) {
    if v.is_empty() { return; }
    let prec = v[0].prec();
    let norm_sq: Float = v.par_iter()
        .map(|vk| {
            let mut t = vk.clone();
            t *= vk;
            t
        })
        .reduce(|| hp_zero(prec), |mut a, b| { a += &b; a });
    let norm = norm_sq.sqrt();
    v.par_iter_mut().for_each(|vk| { *vk /= &norm; });
}

/// Rayleigh quotient `xᵀ A x` for a symmetric matrix `a` (row-major).
/// Parallelized over rows. Final reduction also parallel.
pub fn rayleigh_quotient(a: &[Float], dim: usize, xi: &[Float], prec: u32) -> Float {
    (0..dim).into_par_iter().map(|i| {
        let mut row_sum = hp_zero(prec);
        for j in 0..dim { let mut t = a[i * dim + j].clone(); t *= &xi[j]; row_sum += &t; }
        let mut contrib = row_sum; contrib *= &xi[i]; contrib
    }).reduce(|| hp_zero(prec), |mut a, b| { a += &b; a })
}

/// Inverse iteration to find the smallest-eigenpair of a symmetric
/// matrix at high precision.
///
/// `force_even = true` projects the iterate onto the even subspace
/// at each step (`xi_i ← (xi_i + xi_{n-1-i}) / 2`). This is the right
/// behavior when the construction has reflection symmetry and the
/// smallest *even* eigenvector is the object of interest, not the
/// natural smallest eigenvalue.
///
/// Returns `(eigenvalue, eigenvector)`. The eigenvector is ℓ²-normalized.
pub fn inverse_iteration(
    a: &[Float],
    dim: usize,
    prec: u32,
    max_steps: usize,
    force_even: bool,
) -> Result<(Float, Vec<Float>)> {
    let lu = lu_factor(a, dim)?;

    // Initial guess: a Gaussian-shaped vector centered at the middle index,
    // computed entirely in HP arithmetic. Each entry is independent so we
    // parallelize via into_par_iter.
    let mut xi: Vec<Float> = (0..dim).into_par_iter().map(|i| {
        let center = (dim as i64) / 2;
        let j = (i as i64) - center;
        let half = ((dim as i64) / 2).max(1);
        // x = j / (dim/2)
        let mut x = Float::with_val(prec, j);
        x /= half;
        // x_sq = x²
        let mut x_sq = x.clone();
        x_sq *= &x;
        // arg = -x² / 2 ⇒ build as 0 - x²/2
        x_sq /= 2u32;
        let mut arg = Float::with_val(prec, 0);
        arg -= &x_sq;
        // g = exp(arg)
        arg.exp()
    }).collect();
    normalize_l2(&mut xi);

    let mut mu = hp_zero(prec);
    let mut prev_mu = mu.clone();

    for step in 0..max_steps {
        let mut v = lu_solve(&lu, &xi, dim, prec);
        normalize_l2(&mut v);
        if force_even {
            // Forced-even projection: ξ_i ← (v_i + v_{n-1-i}) / 2.
            // Independent per i → parallelize.
            let xi_sym: Vec<Float> = (0..dim).into_par_iter().map(|i| {
                let mut s = v[i].clone(); s += &v[dim - 1 - i]; s /= 2u32; s
            }).collect();
            xi = xi_sym;
        } else {
            xi = v;
        }
        normalize_l2(&mut xi);
        mu = rayleigh_quotient(a, dim, &xi, prec);

        if step > 2 {
            let mut diff = mu.clone(); diff -= &prev_mu;
            let converged = if !mu.is_zero() {
                let mut r = diff.clone().abs(); r /= &mu.clone().abs();
                r < Float::with_val(prec, 2).pow(-((prec as i32) - 32))
            } else {
                diff.clone().abs() < Float::with_val(prec, 2).pow(-((prec as i32) - 32))
            };
            if converged { break; }
        }
        prev_mu = mu.clone();
    }
    Ok((mu, xi))
}


#[cfg(test)]
mod tests {
    use super::*;
    use rug::Float;
    use crate::fmt::{display_hp, relative_difference, matching_digits};

    /// Build an HP `Float` from an integer-valued seed at the given precision.
    /// Used in tests for textbook small matrices where matrix entries are
    /// integers (or short decimals); these are exact at any HP precision.
    fn hp(prec: u32, s: &str) -> Float {
        Float::with_val(prec, Float::parse(s).unwrap())
    }

    /// HP equality assertion: |a - b| / max(1, |b|) < 10^-tol_digits.
    /// Stays in HP throughout — never converts to f64.
    fn assert_hp_close(actual: &Float, expected: &Float, tol_digits: i32, msg: &str) {
        let prec = actual.prec();
        // tol = 10^-tol_digits, built as HP literal.
        let tol_str = format!("1e-{}", tol_digits);
        let tol = Float::with_val(prec, Float::parse(&tol_str).unwrap());

        // Use relative_difference where possible; for expected==0, use abs diff.
        if expected.is_zero() {
            let mut diff = actual.clone();
            diff -= expected;
            let abs_diff = diff.abs();
            assert!(abs_diff < tol,
                "{}: |actual| = {} should be < 10^-{}",
                msg, display_hp(&abs_diff, 6), tol_digits);
        } else {
            let rel = relative_difference(actual, expected).unwrap();
            assert!(rel < tol,
                "{}: actual={}, expected={}, rel diff = {} should be < 10^-{} (matching digits = {})",
                msg,
                display_hp(actual, 6),
                display_hp(expected, 6),
                display_hp(&rel, 6),
                tol_digits,
                display_hp(&matching_digits(actual, expected), 4));
        }
    }

    /// Build a small symmetric matrix and verify LU factor + solve recovers x.
    /// Test: solve [[2, 1], [1, 3]] * x = [4, 7] → x = [1, 2]. Comparison
    /// is done in HP — never converts the eigenvector to f64 for the assert.
    #[test]
    fn lu_factor_and_solve_2x2() {
        let prec = 64;
        let a = vec![
            hp(prec, "2"), hp(prec, "1"),
            hp(prec, "1"), hp(prec, "3"),
        ];
        let b = vec![hp(prec, "4"), hp(prec, "7")];
        let lu = lu_factor(&a, 2).unwrap();
        let x = lu_solve(&lu, &b, 2, prec);
        assert_hp_close(&x[0], &hp(prec, "1"), 15, "x[0]");
        assert_hp_close(&x[1], &hp(prec, "2"), 15, "x[1]");
    }

    /// Build a 3x3 symmetric matrix with known smallest eigenvalue and
    /// verify inverse iteration finds it.
    /// Matrix: diag(1, 2, 3) — smallest eigenvalue is 1.0.
    #[test]
    fn inverse_iteration_diagonal_3x3() {
        let prec = 128;
        let a = vec![
            hp(prec, "1"), hp(prec, "0"), hp(prec, "0"),
            hp(prec, "0"), hp(prec, "2"), hp(prec, "0"),
            hp(prec, "0"), hp(prec, "0"), hp(prec, "3"),
        ];
        let (mu, _v) = inverse_iteration(&a, 3, prec, 50, false).unwrap();
        assert_hp_close(&mu, &hp(prec, "1"), 10, "smallest eigenvalue");
    }

    /// Larger test: 5x5 symmetric matrix where we know the smallest
    /// eigenvalue exactly. Use diag(0.1, 1.0, 2.0, 3.0, 4.0).
    #[test]
    fn inverse_iteration_finds_small_eigenvalue() {
        let prec = 128;
        let dim = 5;
        let mut a = vec![hp(prec, "0"); dim * dim];
        let diag_vals = ["0.1", "1.0", "2.0", "3.0", "4.0"];
        for (i, v) in diag_vals.iter().enumerate() {
            a[i * dim + i] = hp(prec, v);
        }
        let (mu, v) = inverse_iteration(&a, dim, prec, 100, false).unwrap();
        assert_hp_close(&mu, &hp(prec, "0.1"), 10, "smallest eigenvalue");
        // Eigenvector should be ℓ²-normalized. Check ‖v‖² = 1 in HP.
        let mut norm_sq = hp(prec, "0");
        for vi in &v { norm_sq += vi.clone().square(); }
        assert_hp_close(&norm_sq, &hp(prec, "1"), 10, "‖v‖²");
    }

    /// normalize_l2 should produce a unit vector.
    #[test]
    fn normalize_l2_produces_unit_vector() {
        let prec = 64;
        let mut v = vec![hp(prec, "3"), hp(prec, "4")];
        normalize_l2(&mut v);
        let mut norm_sq = hp(prec, "0");
        for vi in &v { norm_sq += vi.clone().square(); }
        assert_hp_close(&norm_sq, &hp(prec, "1"), 15, "‖v‖²");
        // 3/5, 4/5
        assert_hp_close(&v[0], &hp(prec, "0.6"), 15, "v[0]");
        assert_hp_close(&v[1], &hp(prec, "0.8"), 15, "v[1]");
    }

    /// rayleigh_quotient on a known matrix and vector.
    /// xᵀ·diag(1,2,3)·x with x = (1,0,0) is 1.0.
    #[test]
    fn rayleigh_quotient_diagonal() {
        let prec = 64;
        let a = vec![
            hp(prec, "1"), hp(prec, "0"), hp(prec, "0"),
            hp(prec, "0"), hp(prec, "2"), hp(prec, "0"),
            hp(prec, "0"), hp(prec, "0"), hp(prec, "3"),
        ];
        let x = vec![hp(prec, "1"), hp(prec, "0"), hp(prec, "0")];
        let q = rayleigh_quotient(&a, 3, &x, prec);
        assert_hp_close(&q, &hp(prec, "1"), 15, "Rayleigh quotient");
    }

    /// Singular matrix should error from lu_factor.
    #[test]
    fn lu_factor_singular_errors() {
        let prec = 64;
        // [[1, 2], [2, 4]] — second row is 2× first row.
        let a = vec![
            hp(prec, "1"), hp(prec, "2"),
            hp(prec, "2"), hp(prec, "4"),
        ];
        let result = lu_factor(&a, 2);
        assert!(result.is_err(), "singular matrix should error");
    }

    /// inverse_iteration with force_even on a 4x4 matrix — verify forced-even
    /// projection converges to a valid eigenvalue with even-under-reflection
    /// symmetry (eigenvalue should be 1 or 5 of the block).
    #[test]
    fn inverse_iteration_force_even() {
        let prec = 128;
        let dim = 4;
        // Block-diagonal:
        // [[3, 2, 0, 0],
        //  [2, 3, 0, 0],
        //  [0, 0, 3, 2],
        //  [0, 0, 2, 3]]
        // Each 2×2 block has eigenvalues 1 (eigvec (1,-1)/√2) and 5 (eigvec (1,1)/√2).
        // Even-under-reflection: ξ_0 = ξ_3, ξ_1 = ξ_2.
        // (1, -1, -1, 1)/2 has ξ_0 = -ξ_3 → NOT even.
        // (1,  1,  1, 1)/2 has ξ_0 =  ξ_3 → IS even.
        let mut a = vec![hp(prec, "0"); dim * dim];
        a[0 * dim + 0] = hp(prec, "3");
        a[0 * dim + 1] = hp(prec, "2");
        a[1 * dim + 0] = hp(prec, "2");
        a[1 * dim + 1] = hp(prec, "3");
        a[2 * dim + 2] = hp(prec, "3");
        a[2 * dim + 3] = hp(prec, "2");
        a[3 * dim + 2] = hp(prec, "2");
        a[3 * dim + 3] = hp(prec, "3");
        let (mu, _v) = inverse_iteration(&a, dim, prec, 100, true).unwrap();
        // Either 1 or 5 — depends on which forced-even subspace dominates.
        // Check via HP comparison to both candidates; one must match within 1e-8.
        let one = hp(prec, "1");
        let five = hp(prec, "5");
        let mut diff_to_1 = mu.clone(); diff_to_1 -= &one; let abs_d1 = diff_to_1.abs();
        let mut diff_to_5 = mu.clone(); diff_to_5 -= &five; let abs_d5 = diff_to_5.abs();
        let tol = hp(prec, "1e-8");
        assert!(abs_d1 < tol || abs_d5 < tol,
            "forced-even smallest should be 1 or 5, got {}", display_hp(&mu, 8));
    }
}
