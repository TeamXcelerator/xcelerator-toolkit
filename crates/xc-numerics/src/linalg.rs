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

// ===========================================================================
// Tridiagonal LU factorization with partial pivoting (Thomas + pivot)
// ===========================================================================
//
// For a tridiagonal `n × n` matrix
//
//     [ d_0  c_0   0   0   ...     0    ]
//     [ a_0  d_1  c_1  0   ...     0    ]
//     [ 0    a_1  d_2  c_2 ...     0    ]
//     [ ...                              ]
//     [ 0    ...   0  a_{n-2}  d_{n-1} ]
//
// (the symmetric case has a_i = c_i = off_diag[i]; we accept the general
// form so the structure also handles non-symmetric tridiagonals like
// `T - λI + εI` where the only non-symmetric part is the diagonal shift,
// which doesn't break symmetry — but we don't rely on it.)
//
// LU factorization with partial pivoting on a tridiagonal matrix produces
// at most one column of fill-in: when row `k` is pivoted with row `k+1`,
// entry `(k, k+2)` becomes nonzero. The factored form therefore needs to
// track:
//
//   - `l[i]`     : sub-diagonal multipliers (n-1 entries, l[0..n-1])
//   - `u_d[i]`   : main diagonal of U (n entries)
//   - `u_s[i]`   : super-diagonal of U (n-1 entries, u_s[0..n-1])
//   - `u_ss[i]`  : super-super-diagonal of U (n-2 entries, u_ss[0..n-2]),
//                  populated only at pivoted rows; stays zero elsewhere
//   - `perm[i]`  : permutation in [0, n) recording row swaps
//
// Cost: O(n) ops per factor + O(n) memory. Compare to dense LU's
// O(n³) ops + O(n²) memory. At HP-1000, N=8001 the wall-time saving is
// from ~hours to ~seconds.

/// LU factorization of a tridiagonal `n × n` matrix with partial pivoting.
/// Stored as four banded vectors plus a permutation. See module docs above
/// for the data layout.
#[derive(Clone)]
pub struct TridiagLuFactors {
    /// Sub-diagonal multipliers (length `n-1`).
    pub l: Vec<Float>,
    /// Main diagonal of U (length `n`).
    pub u_d: Vec<Float>,
    /// Super-diagonal of U (length `n-1`).
    pub u_s: Vec<Float>,
    /// Super-super-diagonal of U (length `n-2`), nonzero only at pivoted rows.
    pub u_ss: Vec<Float>,
    /// Row permutation (length `n`): `perm[i]` is the original row index
    /// that ended up at position `i` after pivoting.
    pub perm: Vec<usize>,
}

/// Factorize a tridiagonal matrix at HP precision via Thomas algorithm
/// with partial pivoting. The matrix is supplied as three slices:
///
/// * `lower`: sub-diagonal entries `a_i = M[i+1, i]` for `i ∈ [0, n-1)`,
///   length `n-1`.
/// * `diag`: main diagonal `d_i = M[i, i]`, length `n`.
/// * `upper`: super-diagonal `c_i = M[i, i+1]` for `i ∈ [0, n-1)`, length `n-1`.
///
/// For symmetric tridiagonals (which is our use case in the toolkit),
/// `lower[i] == upper[i] == off_diag[i]`. The factorizer does not exploit
/// symmetry; partial pivoting can break it.
///
/// At HP, "small" pivots are detected via comparison against a precision-
/// derived threshold `2^-(prec - 16)`. A truly zero pivot returns an
/// error; a small-but-nonzero pivot proceeds (the resulting factorization
/// is well-conditioned for shift values typical of inverse iteration).
pub fn tridiag_lu_factor_hp(
    lower: &[Float],
    diag: &[Float],
    upper: &[Float],
    prec: u32,
) -> Result<TridiagLuFactors> {
    let n = diag.len();
    if n == 0 {
        return Err(anyhow!("empty matrix"));
    }
    if lower.len() != n.saturating_sub(1) {
        return Err(anyhow!(
            "lower length {} should be {} (n-1)", lower.len(), n - 1
        ));
    }
    if upper.len() != n.saturating_sub(1) {
        return Err(anyhow!(
            "upper length {} should be {} (n-1)", upper.len(), n - 1
        ));
    }

    // Working state: at each step we have a 2×3 sub-matrix to consider
    // for pivoting:
    //
    //     [ d[k]    c[k]    f[k]   ]   <- f starts as 0 (super-super-diagonal)
    //     [ a[k]    d[k+1]  c[k+1] ]
    //
    // After pivoting (if needed) and elimination, l[k] becomes the
    // multiplier and u_ss[k] = f[k] gets populated when pivoting forced
    // a column-3 swap.
    //
    // We maintain mutable copies of all three diagonals plus an
    // implicit "extra column" for the fill-in.
    let mut a: Vec<Float> = lower.to_vec();
    let mut d: Vec<Float> = diag.to_vec();
    let mut c: Vec<Float> = upper.to_vec();
    // f[k] tracks the (k, k+2) entry that pivoting can introduce. Length
    // n-2 (since the rightmost two rows have no super-super-diagonal).
    let mut f: Vec<Float> = vec![hp_zero(prec); n.saturating_sub(2)];

    let mut l: Vec<Float> = vec![hp_zero(prec); n.saturating_sub(1)];
    let mut u_ss: Vec<Float> = vec![hp_zero(prec); n.saturating_sub(2)];
    let mut perm: Vec<usize> = (0..n).collect();

    let two = Float::with_val(prec, 2);
    let small_pivot_thresh: Float = two.pow(-((prec as i32) - 16));

    for k in 0..(n.saturating_sub(1)) {
        // Compare |d[k]| (current pivot candidate) vs |a[k]| (sub-diagonal
        // candidate from row k+1). Partial pivoting picks the larger.
        let abs_d = d[k].clone().abs();
        let abs_a = a[k].clone().abs();

        if abs_a > abs_d {
            // Pivot rows k and k+1. Swap their entries in:
            //   - main diagonal: d[k] <-> sub-diagonal a[k] is part of
            //     row k+1, so the swap is (d[k] <-> a[k]) — but we do it
            //     in terms of the underlying matrix:
            //       new_row_k    = old_row_{k+1}
            //       new_row_{k+1} = old_row_k
            //   - super-diagonal: c[k] (in row k) <-> d[k+1] (which is
            //     also in row k+1's diagonal — but THAT's the pivoted
            //     row's column k slot)
            //
            // Concretely, before pivoting:
            //   row k    : [..., 0  , d[k],   c[k]  , 0     , 0, ...]
            //   row k+1  : [..., 0  , a[k],   d[k+1], c[k+1], 0, ...]
            //
            // After swap:
            //   row k    : [..., 0  , a[k],   d[k+1], c[k+1], 0, ...]
            //   row k+1  : [..., 0  , d[k],   c[k]  , 0     , 0, ...]
            //
            // Express in our state vectors:
            //   new d[k]   = a[k]      ; new c[k]   = d[k+1]    ; new f[k]   = c[k+1]
            //   new a[k]   = d[k]      ; new d[k+1] = c[k]      ; new c[k+1] = 0
            //
            // (we don't change a[k+1] or beyond; those rows aren't touched)
            let old_d_k = d[k].clone();
            let old_c_k = c[k].clone();
            let old_a_k = a[k].clone();
            let old_d_kp1 = d[k + 1].clone();
            let old_c_kp1 = if k + 1 < n - 1 { c[k + 1].clone() } else { hp_zero(prec) };

            d[k] = old_a_k;
            c[k] = old_d_kp1;
            if k < n - 2 {
                f[k] = old_c_kp1;
            }
            a[k] = old_d_k;
            d[k + 1] = old_c_k;
            if k + 1 < n - 1 {
                c[k + 1] = hp_zero(prec);
            }

            perm.swap(k, k + 1);
        }

        // Pivot must be nonzero. Small-but-finite pivots proceed (the
        // shifted matrix `T - λI + εI` ε term ensures this in practice).
        let pivot = d[k].clone();
        let abs_pivot = pivot.clone().abs();
        if abs_pivot.is_zero() {
            return Err(anyhow!(
                "tridiag LU: zero pivot at row {} (matrix is singular)", k
            ));
        }
        if abs_pivot < small_pivot_thresh {
            // Sub-precision pivot. Continue but flag could be added; for
            // shifted inverse iteration this just means slightly worse
            // conditioning, not failure. Proceed.
        }

        // Multiplier: l[k] = a[k] / pivot.
        let mut mult = a[k].clone();
        mult /= &pivot;
        l[k] = mult.clone();

        // Eliminate row k+1's column k entry. Subtract l[k] * row k from
        // row k+1:
        //   new d[k+1] = d[k+1] - l[k] * c[k]
        //   new c[k+1] = c[k+1] - l[k] * f[k]    (only if k < n-2)
        let mut update_d = c[k].clone();
        update_d *= &mult;
        d[k + 1] -= &update_d;

        if k < n - 2 && k + 1 < n - 1 {
            let mut update_c = f[k].clone();
            update_c *= &mult;
            c[k + 1] -= &update_c;
        }

        // Persist u_ss[k] = f[k] (will be 0 if no pivoting happened, else
        // populated with old c[k+1]).
        if k < n - 2 {
            u_ss[k] = f[k].clone();
        }
    }

    // After the loop, d[] holds U's main diagonal and c[] holds U's
    // super-diagonal. The factor structure is:
    //   L: identity + l[k] at position (k+1, k)
    //   U: d[k] on main, c[k] on super, u_ss[k] on super-super
    Ok(TridiagLuFactors {
        l,
        u_d: d,
        u_s: c,
        u_ss,
        perm,
    })
}

/// Solve `A·x = b` given the tridiagonal LU factorization of `A` from
/// `tridiag_lu_factor_hp`. Returns `x` of length `n`.
///
/// Forward substitution: `L·y = P·b` where `L` has `l[k]` on the
/// sub-diagonal. Back substitution: `U·x = y` where `U` has main
/// diagonal `u_d`, super-diagonal `u_s`, and super-super-diagonal `u_ss`.
pub fn tridiag_lu_solve_hp(
    factors: &TridiagLuFactors,
    b: &[Float],
    prec: u32,
) -> Result<Vec<Float>> {
    let n = factors.u_d.len();
    if b.len() != n {
        return Err(anyhow!("b length {} != n = {}", b.len(), n));
    }

    // Apply permutation: pb[i] = b[perm[i]].
    let pb: Vec<Float> = (0..n).map(|i| b[factors.perm[i]].clone()).collect();

    // Forward sub: y[0] = pb[0]; y[k] = pb[k] - l[k-1] * y[k-1].
    let mut y = vec![hp_zero(prec); n];
    y[0] = pb[0].clone();
    for k in 1..n {
        let mut s = pb[k].clone();
        let mut term = factors.l[k - 1].clone();
        term *= &y[k - 1];
        s -= &term;
        y[k] = s;
    }

    // Back sub: U·x = y.
    //   x[n-1] = y[n-1] / u_d[n-1]
    //   x[n-2] = (y[n-2] - u_s[n-2] * x[n-1]) / u_d[n-2]
    //   x[k]   = (y[k] - u_s[k] * x[k+1] - u_ss[k] * x[k+2]) / u_d[k]
    //          for k = n-3 down to 0
    let mut x = vec![hp_zero(prec); n];
    {
        let mut s = y[n - 1].clone();
        s /= &factors.u_d[n - 1];
        x[n - 1] = s;
    }
    if n >= 2 {
        let mut s = y[n - 2].clone();
        let mut term = factors.u_s[n - 2].clone();
        term *= &x[n - 1];
        s -= &term;
        s /= &factors.u_d[n - 2];
        x[n - 2] = s;
    }
    for k in (0..(n.saturating_sub(2))).rev() {
        let mut s = y[k].clone();
        let mut term = factors.u_s[k].clone();
        term *= &x[k + 1];
        s -= &term;
        let mut term2 = factors.u_ss[k].clone();
        term2 *= &x[k + 2];
        s -= &term2;
        s /= &factors.u_d[k];
        x[k] = s;
    }

    Ok(x)
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

    let iter_start = std::time::Instant::now();
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
            if converged {
                eprintln!(
                    "[HP invit] inverse iteration converged at step {}/{} on N={} (elapsed {:.1}s)",
                    step + 1, max_steps, dim, iter_start.elapsed().as_secs_f64()
                );
                break;
            }
        }
        prev_mu = mu.clone();
        // Progress: print every 25 steps. Useful to distinguish "still
        // iterating" from "wedged" on multi-hour runs at large N.
        if (step + 1) % 25 == 0 {
            eprintln!(
                "[HP invit] inverse iteration {}/{} on N={} (elapsed {:.1}s)",
                step + 1, max_steps, dim, iter_start.elapsed().as_secs_f64()
            );
        }
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

    // -----------------------------------------------------------------------
    // Tridiagonal LU tests
    // -----------------------------------------------------------------------
    //
    // Mirroring the depth of the HP eigensolver test suite (closed-form
    // structured matrices + property-based + dense-vs-banded equivalence).

    /// Helper: build a 2×3 small tridiagonal matrix and round-trip through
    /// factor + solve, confirm it matches a known analytic answer.
    #[test]
    fn tridiag_lu_solves_3x3_identity() {
        let prec = 256;
        // I_3 — pure diagonal, off-diagonals all zero.
        let lower = vec![hp(prec, "0"), hp(prec, "0")];
        let diag = vec![hp(prec, "1"), hp(prec, "1"), hp(prec, "1")];
        let upper = vec![hp(prec, "0"), hp(prec, "0")];
        let factors = tridiag_lu_factor_hp(&lower, &diag, &upper, prec).unwrap();

        let b = vec![hp(prec, "5"), hp(prec, "7"), hp(prec, "11")];
        let x = tridiag_lu_solve_hp(&factors, &b, prec).unwrap();

        // I·x = b means x == b.
        for i in 0..3 {
            assert_hp_close(&x[i], &b[i], 200, &format!("identity solve x[{}]", i));
        }
    }

    /// 4×4 strictly diagonally dominant tridiagonal: known closed-form
    /// answer for a specific b.
    #[test]
    fn tridiag_lu_solves_4x4_known_diag_dominant() {
        let prec = 256;
        // Matrix:
        //   [ 4 1 0 0 ]
        //   [ 1 4 1 0 ]
        //   [ 0 1 4 1 ]
        //   [ 0 0 1 4 ]
        let lower = vec![hp(prec, "1"), hp(prec, "1"), hp(prec, "1")];
        let diag = vec![hp(prec, "4"), hp(prec, "4"), hp(prec, "4"), hp(prec, "4")];
        let upper = vec![hp(prec, "1"), hp(prec, "1"), hp(prec, "1")];
        let factors = tridiag_lu_factor_hp(&lower, &diag, &upper, prec).unwrap();

        // Solve M·x = b for b = [1, 0, 0, 0]. The exact solution can be
        // derived; we verify by recomputing M·x and comparing to b.
        let b = vec![hp(prec, "1"), hp(prec, "0"), hp(prec, "0"), hp(prec, "0")];
        let x = tridiag_lu_solve_hp(&factors, &b, prec).unwrap();
        assert_eq!(x.len(), 4);

        // M·x = b check (in HP).
        let mut mx = vec![hp_zero(prec); 4];
        for i in 0..4 {
            let mut s = diag[i].clone(); s *= &x[i];
            if i > 0 { let mut t = lower[i - 1].clone(); t *= &x[i - 1]; s += &t; }
            if i < 3 { let mut t = upper[i].clone(); t *= &x[i + 1]; s += &t; }
            mx[i] = s;
        }
        for i in 0..4 {
            let mut diff = mx[i].clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            let tol = hp(prec, "1e-200");
            assert!(abs_diff < tol,
                "M·x[{}] - b[{}] = {} should be ≈ 0", i, i, display_hp(&abs_diff, 4));
        }
    }

    /// Pivoting test: a matrix where the natural pivot is zero or tiny,
    /// so partial pivoting is required for stability.
    ///
    ///   [ 0  2  0 ]
    ///   [ 1  3  4 ]
    ///   [ 0  5  6 ]
    ///
    /// Without pivoting, factorization fails at row 0 (zero pivot).
    /// With partial pivoting, rows 0 and 1 swap, giving a well-conditioned
    /// factorization. We solve for b = [2, 4, 11] and verify M·x = b.
    #[test]
    fn tridiag_lu_pivots_when_diagonal_is_zero() {
        let prec = 256;
        let lower = vec![hp(prec, "1"), hp(prec, "5")];
        let diag = vec![hp(prec, "0"), hp(prec, "3"), hp(prec, "6")];
        let upper = vec![hp(prec, "2"), hp(prec, "4")];

        let factors = tridiag_lu_factor_hp(&lower, &diag, &upper, prec).unwrap();

        // Verify the factorization captured the pivoting via permutation.
        // perm[0] should not be 0 (since row 0 had to be pivoted away).
        assert_ne!(factors.perm[0], 0,
            "expected row 0 to be pivoted away; perm = {:?}", factors.perm);

        // Solve M·x = b for some b.
        let b = vec![hp(prec, "2"), hp(prec, "4"), hp(prec, "11")];
        let x = tridiag_lu_solve_hp(&factors, &b, prec).unwrap();

        // M·x check.
        let mut mx = vec![hp_zero(prec); 3];
        for i in 0..3 {
            let mut s = diag[i].clone(); s *= &x[i];
            if i > 0 { let mut t = lower[i - 1].clone(); t *= &x[i - 1]; s += &t; }
            if i < 2 { let mut t = upper[i].clone(); t *= &x[i + 1]; s += &t; }
            mx[i] = s;
        }
        for i in 0..3 {
            let mut diff = mx[i].clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            let tol = hp(prec, "1e-200");
            assert!(abs_diff < tol,
                "with pivoting, M·x[{}] - b[{}] = {}", i, i, display_hp(&abs_diff, 4));
        }
    }

    /// Strang's tridiagonal: diag = [2, 2, ..., 2], off = [-1, -1, ..., -1].
    /// Eigenvalues are λ_k = 2 - 2·cos(kπ/(n+1)). We pick n=10 and solve
    /// (T - λI + εI) y = e_5 (a unit vector at the middle). Verify the
    /// banded solve matches the dense LU output to working precision.
    #[test]
    fn tridiag_lu_matches_dense_lu_on_strang() {
        let prec = 256;
        let n = 10;

        // Strang's tridiagonal.
        let lower: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();
        let diag: Vec<Float> = (0..n).map(|_| hp(prec, "2")).collect();
        let upper: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();

        // Apply a small shift λ = 0.01 to avoid the matrix being singular
        // at any eigenvalue.
        let shift = hp(prec, "0.01");
        let mut shifted_diag = diag.clone();
        for d in shifted_diag.iter_mut() { *d -= &shift; }

        // Build dense form for cross-check.
        let mut dense = vec![hp_zero(prec); n * n];
        for i in 0..n { dense[i * n + i] = shifted_diag[i].clone(); }
        for i in 0..n - 1 {
            dense[i * n + (i + 1)] = upper[i].clone();
            dense[(i + 1) * n + i] = lower[i].clone();
        }

        // Banded LU.
        let banded = tridiag_lu_factor_hp(&lower, &shifted_diag, &upper, prec).unwrap();

        // Dense LU.
        let dense_factors = lu_factor(&dense, n).unwrap();

        // Build b = e_5 (middle unit vector).
        let mut b = vec![hp_zero(prec); n];
        b[n / 2] = hp(prec, "1");

        // Solve via both.
        let x_banded = tridiag_lu_solve_hp(&banded, &b, prec).unwrap();
        let x_dense = lu_solve(&dense_factors, &b, n, prec);

        // Compare element-wise. They should agree to working precision
        // (modulo floating-point trajectory differences in the elimination
        // order — but on a strictly-diagonally-dominant matrix with no
        // pivoting needed, the order is the same and the answers should
        // be bit-close).
        for i in 0..n {
            let mut diff = x_banded[i].clone(); diff -= &x_dense[i];
            let abs_diff = diff.abs();
            // 50 digits of agreement is comfortable headroom at HP-256.
            let tol = hp(prec, "1e-50");
            assert!(abs_diff < tol,
                "banded vs dense disagreement at index {}: {} (banded={}, dense={})",
                i, display_hp(&abs_diff, 6),
                display_hp(&x_banded[i], 6),
                display_hp(&x_dense[i], 6));
        }
    }

    /// Equivalence test against a Wilkinson W11 + shift on a tridiagonal
    /// extracted by hand. Wilkinson W11 has diag = [5,4,3,2,1,0,1,2,3,4,5]
    /// and off = all 1's. We solve (W11 - λI) y = e_0 for some non-eigen
    /// shift λ = 7 (well above the largest eigenvalue ≈10.75) and verify
    /// banded vs dense agree.
    #[test]
    fn tridiag_lu_matches_dense_lu_on_wilkinson_w11_shifted() {
        let prec = 512; // a bit higher to expose any shortcoming
        let n = 11;

        // Wilkinson W11 diagonal.
        let raw_diag = ["5", "4", "3", "2", "1", "0", "1", "2", "3", "4", "5"];
        let diag: Vec<Float> = raw_diag.iter().map(|s| hp(prec, s)).collect();
        let lower: Vec<Float> = (0..n - 1).map(|_| hp(prec, "1")).collect();
        let upper: Vec<Float> = (0..n - 1).map(|_| hp(prec, "1")).collect();

        let shift = hp(prec, "7");
        let mut shifted_diag = diag.clone();
        for d in shifted_diag.iter_mut() { *d -= &shift; }

        // Dense form.
        let mut dense = vec![hp_zero(prec); n * n];
        for i in 0..n { dense[i * n + i] = shifted_diag[i].clone(); }
        for i in 0..n - 1 {
            dense[i * n + (i + 1)] = upper[i].clone();
            dense[(i + 1) * n + i] = lower[i].clone();
        }

        let banded = tridiag_lu_factor_hp(&lower, &shifted_diag, &upper, prec).unwrap();
        let dense_factors = lu_factor(&dense, n).unwrap();

        // b = first canonical basis vector.
        let mut b = vec![hp_zero(prec); n];
        b[0] = hp(prec, "1");

        let x_banded = tridiag_lu_solve_hp(&banded, &b, prec).unwrap();
        let x_dense = lu_solve(&dense_factors, &b, n, prec);

        for i in 0..n {
            let mut diff = x_banded[i].clone(); diff -= &x_dense[i];
            let abs_diff = diff.abs();
            // The W11 case may pivot, so we relax the tolerance vs the
            // bit-close case. 100 digits of agreement is still very tight.
            let tol = hp(prec, "1e-100");
            assert!(abs_diff < tol,
                "Wilkinson W11 banded vs dense at index {}: {}",
                i, display_hp(&abs_diff, 6));
        }
    }

    /// Property test: solve M·x = b on a deterministic-random tridiagonal
    /// at HP-256, then verify M·x = b within working precision.
    /// Uses a small, deterministic seed so the test is reproducible.
    #[test]
    fn tridiag_lu_property_solve_matches_b() {
        let prec = 256;
        let n = 50;

        // Deterministic-random tridiagonal: pick "random" entries from a
        // small set of integer-rationals so HP arithmetic is exact in the
        // construction. (Avoids non-deterministic test failures.)
        let mut diag = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n - 1);
        let mut upper = Vec::with_capacity(n - 1);
        for i in 0..n {
            // Diagonal entries: 5 + (i mod 4) — always positive, mild
            // diagonal dominance vs off-diagonals.
            diag.push(hp(prec, &format!("{}", 5 + (i % 4))));
        }
        for i in 0..n - 1 {
            // Off-diagonals: ±1, ±2 alternating pattern (asymmetric to
            // expose pivoting paths).
            let lv = if i % 3 == 0 { -1 } else if i % 3 == 1 { 2 } else { -2 };
            let uv = if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 2 };
            lower.push(hp(prec, &format!("{}", lv)));
            upper.push(hp(prec, &format!("{}", uv)));
        }

        let factors = tridiag_lu_factor_hp(&lower, &diag, &upper, prec).unwrap();

        // Build a non-trivial b.
        let mut b = Vec::with_capacity(n);
        for i in 0..n { b.push(hp(prec, &format!("{}", 1 + (i % 7)))); }

        let x = tridiag_lu_solve_hp(&factors, &b, prec).unwrap();
        assert_eq!(x.len(), n);

        // Verify M·x = b in HP.
        for i in 0..n {
            let mut s = diag[i].clone(); s *= &x[i];
            if i > 0 { let mut t = lower[i - 1].clone(); t *= &x[i - 1]; s += &t; }
            if i < n - 1 { let mut t = upper[i].clone(); t *= &x[i + 1]; s += &t; }
            let mut diff = s.clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            let tol = hp(prec, "1e-200");
            assert!(abs_diff < tol,
                "M·x[{}] - b[{}] = {}", i, i, display_hp(&abs_diff, 4));
        }
    }

    /// HP-1000 precision-scaling test: factorize a Strang's tridiagonal
    /// at the publication precision, solve a system, verify residual.
    /// This is the "production scenario" test — same precision Paper B
    /// runs at.
    #[test]
    fn tridiag_lu_at_hp_1000() {
        let prec = 3338; // HP-1000d
        let n = 20;

        let lower: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();
        let diag: Vec<Float> = (0..n).map(|_| hp(prec, "2")).collect();
        let upper: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();

        let factors = tridiag_lu_factor_hp(&lower, &diag, &upper, prec).unwrap();

        let mut b = vec![hp_zero(prec); n];
        b[0] = hp(prec, "1");
        b[n - 1] = hp(prec, "1");

        let x = tridiag_lu_solve_hp(&factors, &b, prec).unwrap();

        // Verify M·x = b at HP-1000.
        for i in 0..n {
            let mut s = diag[i].clone(); s *= &x[i];
            if i > 0 { let mut t = lower[i - 1].clone(); t *= &x[i - 1]; s += &t; }
            if i < n - 1 { let mut t = upper[i].clone(); t *= &x[i + 1]; s += &t; }
            let mut diff = s.clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            // At HP-1000 we should match to ~900+ digits.
            let tol = hp(prec, "1e-900");
            assert!(abs_diff < tol,
                "HP-1000 M·x[{}] - b[{}] = {}", i, i, display_hp(&abs_diff, 4));
        }
    }
}

