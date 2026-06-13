// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision dense linear algebra.
//!
//! - **`lu_factor` / `lu_solve`**: LU factorization with partial pivoting,
//!   followed by forward/back substitution. Parallelized via rayon over
//!   the Schur-complement update (factor) and the inner triangular-solve
//!   reductions (solve; see `lu_solve_with` for the serial/parallel knob).
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

// ===========================================================================
// Dense LU factorization with partial pivoting
// ===========================================================================

/// Output of `lu_factor`. Stores the in-place LU representation plus the
/// row permutation produced by partial pivoting.
///
/// The `lu` matrix is row-major with strictly lower-triangle entries
/// holding the L multipliers (unit diagonal of L is implied) and the
/// upper triangle holding U. The `perm` permutation records the row
/// swaps so that `P · A = L · U`.
pub struct LuFactors {
    /// Combined LU storage, row-major, length `dim²`. The strict lower
    /// triangle holds L (with implicit unit diagonal); the upper
    /// triangle (including the diagonal) holds U.
    pub lu: Vec<Float>,
    /// Row permutation. `perm[i]` is the original row index that ended
    /// up at position `i` after partial pivoting.
    pub perm: Vec<usize>,
}

/// LU factorization of a dense `dim × dim` matrix with partial pivoting,
/// in-place, at HP precision.
///
/// Input `a` is row-major (length `dim²`). Returns `LuFactors` suitable
/// for solving `A · x = b` via [`lu_solve`]. Returns an error if the
/// matrix is exactly singular (a zero pivot is encountered after
/// pivoting).
///
/// The Schur-complement update is parallelized across rows via rayon.
/// Cost is O(dim³) HP arithmetic ops; for the load-bearing toolkit
/// callers (`inverse_iteration`) the LU factor is computed once and
/// reused per step.
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

/// Below this row length, the inner triangular-solve reduction runs
/// serially. Rayon's task-dispatch overhead exceeds the benefit of
/// parallelizing a handful of HP multiply-subtracts; the crossover is
/// dominated by per-task overhead vs. per-op HP cost. Tuned
/// conservatively — HP ops at working precision (hundreds to thousands
/// of digits) are expensive enough that a row of ~32 already amortizes
/// the dispatch, but we leave headroom.
const PAR_SOLVE_MIN_ROW: usize = 32;

/// Solve `A·x = b` given the LU factorization of `A` from `lu_factor`.
///
/// Parallelizes the inner reduction of each triangular-solve row across
/// rayon when the row is long enough to amortize task overhead (see
/// [`lu_solve_with`] and `PAR_SOLVE_MIN_ROW`). This is the default for
/// all toolkit callers (notably `inverse_iteration`, where `lu_solve`
/// is the per-step hot path at large dimension).
pub fn lu_solve(factors: &LuFactors, b: &[Float], dim: usize, prec: u32) -> Vec<Float> {
    lu_solve_with(factors, b, dim, prec, true)
}

/// Solve `A·x = b` given the LU factorization of `A` from `lu_factor`,
/// with explicit control over whether the inner triangular-solve
/// reductions are parallelized via rayon.
///
/// The outer loops (over rows) are inherently sequential: in forward
/// substitution row `i` depends on `y[0..i]`, and in back substitution
/// row `i` depends on `x[i+1..dim]`. Only the inner sum
/// `Σ_j lu[i,j] · {y,x}[j]` is parallelizable. For each row that inner
/// sum is a reduction over up to `dim` HP multiply-subtracts; at large
/// `dim` and high precision this dominates `inverse_iteration`'s
/// per-step cost, and parallelizing it across cores gives a real
/// end-to-end speedup.
///
/// * `parallel = true` (the [`lu_solve`] default): parallelize the inner
///   reduction for rows longer than `PAR_SOLVE_MIN_ROW`; short rows stay
///   serial to avoid rayon dispatch overhead exceeding the work.
/// * `parallel = false`: fully serial. Useful for tiny matrices,
///   deterministic single-threaded benchmarking, or callers that are
///   already saturating all cores at a higher level.
///
/// The result is identical to working precision either way. HP reduction
/// order can differ between the serial and parallel paths, but the
/// difference is below the working-precision floor (the eigenvalue and
/// eigenvector reference tests confirm this tolerance).
pub fn lu_solve_with(
    factors: &LuFactors,
    b: &[Float],
    dim: usize,
    prec: u32,
    parallel: bool,
) -> Vec<Float> {
    let lu = &factors.lu;
    let perm = &factors.perm;
    let pb: Vec<Float> = (0..dim).map(|i| b[perm[i]].clone()).collect();

    // Forward substitution: solve L·y = P·b. L has implicit unit diagonal.
    let mut y = vec![hp_zero(prec); dim];
    for i in 0..dim {
        let row_len = i; // number of terms in the inner sum
        let s = if parallel && row_len >= PAR_SOLVE_MIN_ROW {
            // Parallel multiplies, then a fixed index-order fold. HP
            // addition is non-associative, so we must NOT use rayon's
            // `.reduce()` (its combine order is runtime-dependent). The
            // sequential fold over the collected terms makes the result
            // bit-identical run-to-run — required for xi cacheability.
            let terms: Vec<Float> = (0..i).into_par_iter().map(|j| {
                let mut t = lu[i * dim + j].clone(); t *= &y[j]; t
            }).collect();
            let mut sum = hp_zero(prec);
            for t in &terms { sum += t; }
            let mut s = pb[i].clone(); s -= &sum; s
        } else {
            let mut s = pb[i].clone();
            for j in 0..i { let mut t = lu[i * dim + j].clone(); t *= &y[j]; s -= &t; }
            s
        };
        y[i] = s;
    }

    // Back substitution: solve U·x = y. U has explicit diagonal.
    let mut x = vec![hp_zero(prec); dim];
    for i in (0..dim).rev() {
        let row_len = dim - 1 - i; // number of terms in the inner sum
        let mut s = if parallel && row_len >= PAR_SOLVE_MIN_ROW {
            // Parallel multiplies, then a fixed index-order fold (see the
            // forward-substitution note above): deterministic HP sum.
            let terms: Vec<Float> = ((i + 1)..dim).into_par_iter().map(|j| {
                let mut t = lu[i * dim + j].clone(); t *= &x[j]; t
            }).collect();
            let mut sum = hp_zero(prec);
            for t in &terms { sum += t; }
            let mut s = y[i].clone(); s -= &sum; s
        } else {
            let mut s = y[i].clone();
            for j in (i + 1)..dim { let mut t = lu[i * dim + j].clone(); t *= &x[j]; s -= &t; }
            s
        };
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
    // Parallel squares, then a fixed index-order fold. HP addition is
    // non-associative, so rayon's `.reduce()` (runtime-dependent combine
    // order) would make ‖v‖ — and hence the normalized v — drift in the
    // low bits run-to-run. The sequential fold keeps it bit-identical.
    let squares: Vec<Float> = v.par_iter()
        .map(|vk| {
            let mut t = vk.clone();
            t *= vk;
            t
        })
        .collect();
    let mut norm_sq = hp_zero(prec);
    for t in &squares { norm_sq += t; }
    let norm = norm_sq.sqrt();
    v.par_iter_mut().for_each(|vk| { *vk /= &norm; });
}

/// Rayleigh quotient `xᵀ A x` for a symmetric matrix `a` (row-major).
/// Per-row contributions are computed in parallel, then summed in a
/// fixed index order. The final fold is sequential (not rayon
/// `.reduce()`) because HP addition is non-associative: a runtime-
/// ordered reduction would let the low bits of the returned μ drift
/// run-to-run, which can flip the inverse-iteration convergence test
/// and change the iteration count. A deterministic μ keeps xi
/// reproducible.
pub fn rayleigh_quotient(a: &[Float], dim: usize, xi: &[Float], prec: u32) -> Float {
    let contribs: Vec<Float> = (0..dim).into_par_iter().map(|i| {
        let mut row_sum = hp_zero(prec);
        for j in 0..dim { let mut t = a[i * dim + j].clone(); t *= &xi[j]; row_sum += &t; }
        let mut contrib = row_sum; contrib *= &xi[i]; contrib
    }).collect();
    let mut total = hp_zero(prec);
    for c in &contribs { total += c; }
    total
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
///
/// # Convergence floors
///
/// The eigenvector residual `‖A·v - μ·v‖` is governed by *two* floors,
/// whichever is larger:
///
/// 1. **Rate floor**: `(λ_min / λ_next_smallest)^max_steps`. Active when
///    the iteration runs to `max_steps` without converging. Tight when
///    the smallest eigenvalue is well-separated from the next-smallest.
/// 2. **Rayleigh sqrt-floor**: When the iteration's
///    Rayleigh-quotient stability test triggers early termination
///    (eigenvalue change `|Δμ/μ| < 2^-(prec-32)`), the eigenvector
///    residual is bounded by `√(2^-(prec-32))` ≈ `10^-(0.15·prec)`.
///    This follows from Rayleigh's quadratic convergence: when the
///    eigenvalue is good to ε, the eigenvector is good to √ε.
///
/// At HP-256 the Rayleigh sqrt-floor is ~10⁻³³; at HP-1000 it's ~10⁻⁴⁹⁸.
/// To reach working-precision residual regardless of which floor is
/// active, callers should use a *shifted* inverse iteration. For
/// tridiagonal inputs see [`crate::eigen::tridiag_eigenvector_for_value_hp`]
/// (which factorizes `T - λI + ε·I` once and runs the iteration on
/// the deflated system, reaching ~10⁻⁹⁰⁰ at HP-1000 in tens of steps).
pub fn inverse_iteration(
    a: &[Float],
    dim: usize,
    prec: u32,
    max_steps: usize,
    force_even: bool,
) -> Result<(Float, Vec<Float>)> {
    inverse_iteration_from(a, dim, prec, max_steps, force_even, None)
}

/// Inverse iteration with an optional warm-start vector.
/// When `start` is `Some(v)`, uses `v` as the initial guess instead of
/// the Gaussian. When `None`, falls back to the Gaussian initial guess.
/// Warm-start from a nearby-precision cached ξ
/// dramatically reduces iteration count for P-sweep campaigns.
pub fn inverse_iteration_from(
    a: &[Float],
    dim: usize,
    prec: u32,
    max_steps: usize,
    force_even: bool,
    start: Option<Vec<Float>>,
) -> Result<(Float, Vec<Float>)> {
    let lu = lu_factor(a, dim)?;

    // Initial guess: warm-start from provided vector, or fall back to
    // Gaussian centered at the middle index.
    let mut xi: Vec<Float> = if let Some(warm) = start {
        // Re-precision the warm-start vector (it may be at a different prec)
        warm.into_iter().map(|v| {
            let s = v.to_string();
            Float::with_val(prec, Float::parse(&s).unwrap_or_else(|_| Float::parse("0").unwrap()))
        }).collect()
    } else {
        // Gaussian initial guess
        (0..dim).into_par_iter().map(|i| {
            let center = (dim as i64) / 2;
            let j = (i as i64) - center;
            let half = ((dim as i64) / 2).max(1);
            let mut x = Float::with_val(prec, j);
            x /= half;
            let mut x_sq = x.clone();
            x_sq *= &x;
            x_sq /= 2u32;
            let mut arg = Float::with_val(prec, 0);
            arg -= &x_sq;
            arg.exp()
        }).collect()
    };
    normalize_l2(&mut xi);

    let mut mu = hp_zero(prec);
    let mut prev_mu = mu.clone();

    let iter_start = std::time::Instant::now();
    for step in 0..max_steps {
        let mut v = lu_solve(&lu, &xi, dim, prec);
        normalize_l2(&mut v);
        if force_even {
            // Auto-detect natural symmetry, only project
            // when the vector is drifting odd. This avoids the overhead of
            // projecting an already-even vector, and logs when odd drift
            // is detected (debug mode only).
            //
            // Symmetry deviation: max_i |v[i] - v[dim-1-i]| / max_i |v[i]|
            // ≈ 0 → even (no projection needed)
            // ≈ 2 → odd (projection needed; log warning in debug)
            let linf: Float = v.iter().map(|x| x.clone().abs())
                .fold(hp_zero(prec), |a, b| if b > a { b } else { a });
            let odd_dev: Float = if linf.is_zero() {
                hp_zero(prec)
            } else {
                let max_asym = (0..dim).map(|i| {
                    let mut d = v[i].clone(); d -= &v[dim - 1 - i]; d.abs()
                }).fold(hp_zero(prec), |a, b| if b > a { b } else { a });
                let mut rel = max_asym; rel /= &linf; rel
            };
            // Threshold: projection needed if deviation > 0.5 (halfway between
            // even=0 and odd=2). Below threshold the vector is already even
            // enough — skip projection to save cost.
            let needs_projection = odd_dev.to_f64() > 0.5;
            if needs_projection {
                crate::hp_debug!(
                    "[HP invit] step {}: natural vector is odd (deviation={:.3}), applying even projection",
                    step + 1, odd_dev.to_f64()
                );
                let xi_sym: Vec<Float> = (0..dim).into_par_iter().map(|i| {
                    let mut s = v[i].clone(); s += &v[dim - 1 - i]; s /= 2u32; s
                }).collect();
                xi = xi_sym;
            } else {
                xi = v;
            }
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
                crate::hp_debug!(
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
            crate::hp_debug!(
                "[HP invit] inverse iteration {}/{} on N={} (elapsed {:.1}s)",
                step + 1, max_steps, dim, iter_start.elapsed().as_secs_f64()
            );
        }
    }

    // ── Part 2: Shifted inverse iteration refinement ─────────────────────
    // The Rayleigh-quotient convergence above gives μ at full precision but
    // ξ only at √(tol) ≈ half-precision. One step of shifted inverse
    // iteration at μ yields a full-precision eigenvector.
    //
    // Solves (A − μ·I)·v = ξ_old, then normalizes. The shift makes the
    // target eigenvalue appear near-zero, so one solve gives full-precision
    // convergence regardless of eigenvalue gaps.
    let shifted_a: Vec<Float> = (0..dim * dim).into_par_iter().map(|idx| {
        let i = idx / dim;
        let j = idx % dim;
        let mut val = a[idx].clone();
        if i == j {
            val -= &mu;
        }
        val
    }).collect();

    match lu_factor(&shifted_a, dim) {
        Ok(shifted_lu) => {
            let mut xi_refined = lu_solve(&shifted_lu, &xi, dim, prec);
            normalize_l2(&mut xi_refined);
            if force_even {
                let xi_sym: Vec<Float> = (0..dim).into_par_iter().map(|i| {
                    let mut s = xi_refined[i].clone();
                    s += &xi_refined[dim - 1 - i];
                    s /= 2u32;
                    s
                }).collect();
                xi_refined = xi_sym;
                normalize_l2(&mut xi_refined);
            }

            // ── Part 3: Confidence check ─────────────────────────────────
            // Recompute Rayleigh quotient on refined vector. If it differs
            // significantly from μ, the refinement picked up a different
            // eigenvalue (near-degeneracy cross-over). In that case, keep
            // the original (unrefined) ξ which was at least pointing in the
            // right direction.
            let mu_refined = rayleigh_quotient(a, dim, &xi_refined, prec);
            let mut check_diff = mu_refined.clone();
            check_diff -= &mu;
            let check_ratio = if !mu.is_zero() {
                let mut r = check_diff.abs();
                r /= &mu.clone().abs();
                r
            } else {
                check_diff.abs()
            };
            // Accept refinement only if eigenvalue didn't jump (< 1% relative change)
            let accept_tol = Float::with_val(prec, 0.01f64);
            if check_ratio < accept_tol {
                xi = xi_refined;
                mu = mu_refined;
                crate::hp_debug!(
                    "[HP invit] shifted refinement accepted (delta_mu/mu < 1%)",
                );
            } else {
                crate::hp_debug!(
                    "[HP invit] shifted refinement REJECTED: eigenvalue jumped (delta/mu = {}), keeping original ξ",
                    check_ratio.to_f64()
                );
            }
        }
        Err(_) => {
            // Shifted matrix is singular (exact eigenvalue hit) — skip
            // refinement. The unrefined ξ from inverse iteration is used.
            crate::hp_debug!(
                "[HP invit] shifted matrix singular at μ — skipping refinement (eigvec at sqrt-precision)",
            );
        }
    }

    // ── Part 1 residual check (diagnostic, not a loop — just verification) ──
    // Compute ||A·ξ − μ·ξ||∞ for logging. This lets us track eigenvector
    // quality across runs and detect regressions.
    let residual_norm: Float = {
        let residuals: Vec<Float> = (0..dim).into_par_iter().map(|i| {
            let mut av_i = hp_zero(prec);
            for j in 0..dim {
                let mut t = a[i * dim + j].clone();
                t *= &xi[j];
                av_i += &t;
            }
            let mut mu_v_i = mu.clone();
            mu_v_i *= &xi[i];
            av_i -= &mu_v_i;
            av_i.abs()
        }).collect();
        let mut max_r = hp_zero(prec);
        for r in &residuals {
            if *r > max_r { max_r = r.clone(); }
        }
        max_r
    };
    crate::hp_debug!(
        "[HP invit] final residual ||Av-μv||_∞ = {} (prec={} bits, dim={})",
        residual_norm.to_f64(), prec, dim
    );

    Ok((mu, xi))
}


// Matrix fixtures below use the row-major index convention `a[i * dim + j]`
// uniformly, including the `i = 0` / `j = 0` rows where `0 * dim` and `+ 0`
// are kept for visual alignment with their neighbors. Allow the resulting
// erasing_op / identity_op lints in this test-only module.
#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
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

    /// Near-degenerate eigenvalues: tests that shifted refinement resolves
    /// the correct eigenvector even when two eigenvalues are moderately close.
    /// Matrix: diag(1.0, 1.1, 2.0, 3.0) — gap = 0.1 (10%). The unshifted
    /// inverse iteration converges slowly (ratio 1/1.1 = 0.91 per step),
    /// but shifted refinement at μ≈1.0 should give a clean eigenvector.
    #[test]
    fn inverse_iteration_near_degenerate_eigenvalues() {
        let prec = 256;
        let dim = 4;
        let mut a = vec![hp(prec, "0"); dim * dim];
        a[0 * dim + 0] = hp(prec, "1.0");
        a[1 * dim + 1] = hp(prec, "1.1");
        a[2 * dim + 2] = hp(prec, "2.0");
        a[3 * dim + 3] = hp(prec, "3.0");
        let (mu, v) = inverse_iteration(&a, dim, prec, 200, false).unwrap();
        // Must converge to 1.0, not 1.1
        assert_hp_close(&mu, &hp(prec, "1"), 3, "near-degenerate smallest eigenvalue");
        // Eigenvector should be concentrated on index 0 (the 1.0 eigenspace)
        let v0_sq = v[0].clone().square();
        let threshold = hp(prec, "0.99");
        assert!(v0_sq > threshold,
            "eigenvector should be concentrated on index 0 for eigenvalue 1.0, got |v[0]|²={}",
            v0_sq.to_f64());
    }

    /// VERY near-degenerate (gap = 0.01): verifies the shifted refinement
    /// step helps separate eigenvalues that are close but resolvable.
    #[test]
    fn inverse_iteration_extremely_close_eigenvalues() {
        let prec = 512;
        let dim = 4;
        let mut a = vec![hp(prec, "0"); dim * dim];
        a[0 * dim + 0] = hp(prec, "1.0");
        a[1 * dim + 1] = hp(prec, "1.01"); // gap = 0.01
        a[2 * dim + 2] = hp(prec, "5.0");
        a[3 * dim + 3] = hp(prec, "10.0");
        let (mu, v) = inverse_iteration(&a, dim, prec, 200, false).unwrap();
        // Eigenvalue must be 1.0 (not 1.01).
        let mut diff = mu.clone();
        diff -= &hp(prec, "1.0");
        let abs_diff = diff.abs();
        let tol = hp(prec, "0.005"); // must be closer to 1.0 than to 1.01
        assert!(abs_diff < tol,
            "close eigenvalues: μ should be 1.0, diff = {}",
            abs_diff.to_f64());
        // Eigenvector should be concentrated on index 0
        let v0_sq = v[0].clone().square();
        let threshold = hp(prec, "0.99");
        assert!(v0_sq > threshold,
            "eigvec should point at index 0 for eigenvalue 1.0, got |v[0]|²={}",
            v0_sq.to_f64());
    }

    /// Residual quality check: after refinement, the residual ||Av−μv||
    /// should be near working precision, not the sqrt-floor.
    /// Uses a non-trivial symmetric matrix where the sqrt-floor would be
    /// visible without refinement.
    #[test]
    fn inverse_iteration_residual_below_sqrt_floor() {
        let prec = 256; // sqrt-floor at ~10^-33, working precision ~10^-66
        let dim = 6;
        // Tridiagonal 2,-1 (Strang matrix): eigenvalues are
        // 2 - 2*cos(k*π/(n+1)) for k=1..n. Smallest ~ 2*(1-cos(π/7)) ≈ 0.198.
        let mut a = vec![hp(prec, "0"); dim * dim];
        for i in 0..dim {
            a[i * dim + i] = hp(prec, "2");
            if i > 0 { a[i * dim + (i - 1)] = hp(prec, "-1"); }
            if i + 1 < dim { a[i * dim + (i + 1)] = hp(prec, "-1"); }
        }
        let (mu, v) = inverse_iteration(&a, dim, prec, 200, false).unwrap();
        // Compute residual ||Av - μv||∞
        let mut max_resid = hp(prec, "0");
        for i in 0..dim {
            let mut av_i = hp(prec, "0");
            for j in 0..dim {
                let mut t = a[i * dim + j].clone();
                t *= &v[j];
                av_i += &t;
            }
            let mut mu_v_i = mu.clone();
            mu_v_i *= &v[i];
            av_i -= &mu_v_i;
            let abs_r = av_i.abs();
            if abs_r > max_resid { max_resid = abs_r; }
        }
        // With refinement, residual should be much better than the sqrt-floor.
        // sqrt-floor at prec=256 is ~10^-33. Full precision would be ~10^-66.
        // Accept if residual < 10^-50 (significantly below sqrt-floor but
        // allowing some numerical noise).
        let tol = hp(prec, "1e-50");
        assert!(max_resid < tol,
            "residual after refinement should be below sqrt-floor; got ||Av-μv||∞ = {} (expect < 1e-50)",
            max_resid.to_f64());
    }

    /// Warm-start test: inverse_iteration_from with a warm-start converges to
    /// the same eigenpair as cold-start with the Gaussian guess.
    #[test]
    fn inverse_iteration_warm_start_matches_cold() {
        let prec = 256;
        let dim = 6;
        // Strang tridiagonal n=6
        let mut a = vec![hp(prec, "0"); dim * dim];
        for i in 0..dim {
            a[i * dim + i] = hp(prec, "2");
            if i > 0 { a[i * dim + (i-1)] = hp(prec, "-1"); }
            if i+1 < dim { a[i * dim + (i+1)] = hp(prec, "-1"); }
        }
        // Cold start
        let (mu_cold, xi_cold) = inverse_iteration(&a, dim, prec, 200, false).unwrap();
        // Warm start from the cold result (simulates a nearby-precision cache hit)
        let warm = xi_cold.clone();
        let (mu_warm, xi_warm) = inverse_iteration_from(&a, dim, prec, 200, false, Some(warm)).unwrap();

        // Eigenvalues must match to working precision
        let mut diff = mu_cold.clone(); diff -= &mu_warm;
        let abs_diff = diff.abs();
        let tol = hp(prec, "1e-50");
        assert!(abs_diff < tol,
            "warm-start eigenvalue should match cold-start; diff={}",
            abs_diff.to_f64());

        // Eigenvectors must match (up to sign)
        let dot: Float = xi_cold.iter().zip(xi_warm.iter())
            .map(|(a,b)| { let mut t=a.clone(); t*=b; t })
            .fold(hp(prec,"0"), |mut s,t| { s+=&t; s });
        // |dot| should be ≈ 1 (unit vectors)
        let dot_abs = dot.abs();
        let mut diff2 = dot_abs.clone(); diff2 -= hp(prec, "1");
        let diff2_abs = diff2.abs();
        let sign_tol = hp(prec, "1e-40");
        assert!(diff2_abs < sign_tol,
            "warm-start eigenvector should match cold-start (up to sign); |dot|-1={}",
            diff2_abs.to_f64());
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
    // normalize_l2 / rayleigh_quotient — HEAVY tests
    // -----------------------------------------------------------------------

    /// `normalize_l2` on a zero-length vector should be a no-op (no panic,
    /// no allocation). The function early-returns when `v` is empty.
    #[test]
    fn normalize_l2_empty_vector_is_noop() {
        let mut v: Vec<Float> = Vec::new();
        normalize_l2(&mut v);
        assert!(v.is_empty(), "empty vector should remain empty");
    }

    /// `normalize_l2` on a single-nonzero-element vector produces a unit
    /// vector (just sign-preserved): [3] → [1], [-7] → [-1].
    #[test]
    fn normalize_l2_single_element_preserves_sign() {
        let prec = 128;
        let mut v_pos = vec![hp(prec, "3")];
        normalize_l2(&mut v_pos);
        let one = hp(prec, "1");
        let mut diff = v_pos[0].clone(); diff -= &one;
        let abs_diff = diff.abs();
        assert!(abs_diff < hp(prec, "1e-30"),
            "[3] normalized should be [1]; got [{}]",
            display_hp(&v_pos[0], 6));

        let mut v_neg = vec![hp(prec, "-7")];
        normalize_l2(&mut v_neg);
        let neg_one = {
            let mut t = hp(prec, "1");
            t = -t;
            t
        };
        let mut diff = v_neg[0].clone(); diff -= &neg_one;
        let abs_diff = diff.abs();
        assert!(abs_diff < hp(prec, "1e-30"),
            "[-7] normalized should be [-1]; got [{}]",
            display_hp(&v_neg[0], 6));
    }

    /// Property: after `normalize_l2`, the L² norm of the result is
    /// always 1, regardless of the input vector's magnitude or length.
    /// Tests across a sweep of sizes and seeded "random" signs.
    #[test]
    fn property_normalize_l2_produces_unit_norm() {
        let prec = 256;
        let sizes = [3usize, 7, 16, 50];

        for &n in &sizes {
            for seed in 0..3 {
                // Deterministic-random vector with values in {-3, -1, 1, 3, 5}.
                let pattern = [-3i32, -1, 1, 3, 5];
                let mut v: Vec<Float> = (0..n).map(|i| {
                    let val = pattern[(i + seed) % pattern.len()];
                    hp(prec, &val.to_string())
                }).collect();

                normalize_l2(&mut v);

                // Check ‖v‖² = 1.
                let mut norm_sq = hp(prec, "0");
                for vi in &v {
                    let mut t = vi.clone();
                    t *= vi;
                    norm_sq += &t;
                }
                let mut diff = norm_sq.clone(); diff -= 1u32;
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-50");
                assert!(abs_diff < tol,
                    "n={}, seed={}: ‖v‖² should be 1, got {}",
                    n, seed, display_hp(&norm_sq, 6));
            }
        }
    }

    /// HP-1000 round-trip: build a vector at HP-1000, normalize, and
    /// verify ‖v‖² = 1 to working precision.
    #[test]
    fn normalize_l2_at_hp_1000() {
        let prec = 3338;
        let n = 100;
        let mut v: Vec<Float> = (0..n).map(|i| {
            // Linear ramp [1, 2, ..., 100] in HP.
            hp(prec, &(i + 1).to_string())
        }).collect();
        normalize_l2(&mut v);

        let mut norm_sq = hp(prec, "0");
        for vi in &v {
            let mut t = vi.clone();
            t *= vi;
            norm_sq += &t;
        }
        let mut diff = norm_sq.clone(); diff -= 1u32;
        let abs_diff = diff.abs();
        // At HP-1000 working precision, the normalize operation should
        // achieve ‖v‖² = 1 to ~working-precision floor.
        let tol = hp(prec, "1e-900");
        assert!(abs_diff < tol,
            "HP-1000 normalize_l2: ‖v‖² should be 1, got {} (diff {})",
            display_hp(&norm_sq, 8), display_hp(&abs_diff, 6));
    }

    /// `rayleigh_quotient` on a known eigenvector returns the
    /// corresponding eigenvalue. Test on Strang n=3 with the smallest
    /// eigenvector recovered via inverse iteration.
    #[test]
    fn rayleigh_quotient_returns_eigenvalue_for_eigenvector() {
        let prec = 256;
        let n = 3;
        // Strang's tridiagonal n=3 as a dense matrix.
        let mut a = vec![hp_zero(prec); n * n];
        for i in 0..n {
            a[i * n + i] = hp(prec, "2");
            if i > 0 { a[i * n + (i - 1)] = hp(prec, "-1"); }
            if i + 1 < n { a[i * n + (i + 1)] = hp(prec, "-1"); }
        }
        // Recover smallest eigenvector via inverse iteration.
        let (mu, v) = inverse_iteration(&a, n, prec, 200, false).unwrap();
        // Now compute Rayleigh quotient and compare to mu.
        let rq = rayleigh_quotient(&a, n, &v, prec);
        let mut diff = rq.clone(); diff -= &mu;
        let abs_diff = diff.abs();
        // RQ matches the iteration's μ to working precision (it's the
        // same computation modulo allocation).
        let tol = hp(prec, "1e-50");
        assert!(abs_diff < tol,
            "RQ on smallest eigenvector should match μ; RQ={}, μ={}, diff={}",
            display_hp(&rq, 8), display_hp(&mu, 8), display_hp(&abs_diff, 6));
    }

    /// Property: Rayleigh quotient is bounded above by the largest
    /// eigenvalue and below by the smallest, for any unit vector.
    /// We test by building diag(1, 2, 3, 4, 5) and confirming
    /// that the RQ for several unit vectors lies in [1, 5].
    #[test]
    fn property_rayleigh_quotient_bounded_by_spectrum() {
        let prec = 256;
        let n = 5;
        // Diagonal matrix with eigenvalues 1, 2, 3, 4, 5.
        let mut a = vec![hp_zero(prec); n * n];
        for i in 0..n {
            a[i * n + i] = hp(prec, &(i + 1).to_string());
        }

        // Test on a sweep of unit vectors. For diag(λ_1, ..., λ_n),
        // RQ = Σ λ_i x_i² / Σ x_i², which is a convex combination of
        // eigenvalues bounded by min/max λ.
        let lower = hp(prec, "1");
        let upper = hp(prec, "5");
        let tol = hp(prec, "1e-30");

        // 5 deterministic unit vectors via seeded patterns.
        for seed in 0..5 {
            let mut v: Vec<Float> = (0..n).map(|i| {
                let val = (((i + seed) * 13) % 11 + 1) as i32;
                hp(prec, &val.to_string())
            }).collect();
            normalize_l2(&mut v);
            let rq = rayleigh_quotient(&a, n, &v, prec);

            // RQ must be ≥ 1 - tol and ≤ 5 + tol.
            let mut below = lower.clone(); below -= &tol;
            let mut above = upper.clone(); above += &tol;
            assert!(rq >= below,
                "seed={}: RQ {} should be ≥ smallest eigenvalue 1",
                seed, display_hp(&rq, 6));
            assert!(rq <= above,
                "seed={}: RQ {} should be ≤ largest eigenvalue 5",
                seed, display_hp(&rq, 6));
        }
    }

    /// HP-1000 RQ test: on Strang n=10 with smallest eigenvector,
    /// Rayleigh quotient matches the known closed-form smallest
    /// eigenvalue λ_1 = 2 - 2cos(π/11) to working-precision floor.
    #[test]
    fn rayleigh_quotient_at_hp_1000() {
        let prec = 3338;
        let n = 10;
        let a = strang_dense(prec, n);
        // Use inverse_iteration to get the smallest eigenpair.
        let (mu, v) = inverse_iteration(&a, n, prec, 200, false).unwrap();
        let rq = rayleigh_quotient(&a, n, &v, prec);
        let mut diff = rq.clone(); diff -= &mu;
        let abs_diff = diff.abs();
        // RQ and μ should match very tightly. The Rayleigh-sqrt
        // convergence floor at HP-1000 (~10⁻⁴⁹⁸) governs how close v is
        // to the true eigenvector; the difference between RQ(v) and
        // μ from inverse iteration is bounded by roughly the
        // *square* of that floor (RQ has quadratic accuracy in the
        // eigenvector error). 1e-100 leaves comfortable headroom.
        let tol = hp(prec, "1e-100");
        assert!(abs_diff < tol,
            "HP-1000 RQ vs μ: |RQ - μ| = {} should be < 1e-100",
            display_hp(&abs_diff, 6));
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
            // HP-256 working precision is ~77 decimal digits. After one
            // LU factor + one solve over a 4×4 matrix, accumulated ULP
            // error is ~10^-70. The test tolerance must respect this:
            // 10^-60 leaves plenty of headroom while still catching any
            // real algorithmic error (which would be much larger).
            let tol = hp(prec, "1e-60");
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
            // HP-256 ≈ 77 decimal digits. Pivoted 3×3 LU residual is at
            // worst ~10^-70 from ULP accumulation; 10^-60 leaves headroom.
            let tol = hp(prec, "1e-60");
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
            // HP-256 ≈ 77 decimal digits. At n=50 with asymmetric off-
            // diagonals and partial pivoting, accumulated ULP error is
            // ~10^-70; 10^-50 is comfortable headroom while still
            // catching any algorithmic bug (which would manifest as
            // O(1) error, not 10^-50).
            let tol = hp(prec, "1e-50");
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

    // -----------------------------------------------------------------------
    // Dense LU — HEAVY tests
    // -----------------------------------------------------------------------
    //
    // Closed-form structured matrices + property test + HP-1000 residual,
    // mirroring the depth of the banded LU and HP eigensolver test suites.

    /// Build Strang's tridiagonal `n x n` (diag=2, off=-1) as a *dense*
    /// row-major matrix. Used as a canonical structured input for the
    /// dense LU tests below.
    fn strang_dense(prec: u32, n: usize) -> Vec<Float> {
        let mut m = vec![hp_zero(prec); n * n];
        for i in 0..n {
            m[i * n + i] = hp(prec, "2");
            if i > 0 { m[i * n + (i - 1)] = hp(prec, "-1"); }
            if i + 1 < n { m[i * n + (i + 1)] = hp(prec, "-1"); }
        }
        m
    }

    /// Build Wilkinson W11 + shift as a dense matrix for LU testing.
    /// W11 has diag = [5,4,3,2,1,0,1,2,3,4,5], off = all 1's. We add a
    /// shift of -7 to the diagonal so the matrix is well-conditioned
    /// (the largest eigenvalue ≈ 10.75; shifting by -7 keeps all
    /// eigenvalues bounded away from zero).
    fn wilkinson_w11_shifted(prec: u32) -> Vec<Float> {
        let n = 11;
        let raw = ["5", "4", "3", "2", "1", "0", "1", "2", "3", "4", "5"];
        let mut m = vec![hp_zero(prec); n * n];
        for i in 0..n {
            let mut d = hp(prec, raw[i]);
            d -= 7u32;
            m[i * n + i] = d;
            if i > 0 { m[i * n + (i - 1)] = hp(prec, "1"); }
            if i + 1 < n { m[i * n + (i + 1)] = hp(prec, "1"); }
        }
        m
    }

    /// Closed-form: dense LU of Strang n=10 should solve M·x = e_5
    /// (middle unit vector) and the recovered x should satisfy M·x = b
    /// to working precision.
    #[test]
    fn lu_factor_solves_strang_n10() {
        let prec = 256;
        let n = 10;
        let m = strang_dense(prec, n);

        let factors = lu_factor(&m, n).unwrap();

        // Solve M·x = e_{n/2}.
        let mut b = vec![hp_zero(prec); n];
        b[n / 2] = hp(prec, "1");
        let x = lu_solve(&factors, &b, n, prec);
        assert_eq!(x.len(), n);

        // Verify M·x = b.
        for i in 0..n {
            let mut s = hp_zero(prec);
            for j in 0..n {
                let mut t = m[i * n + j].clone();
                t *= &x[j];
                s += &t;
            }
            let mut diff = s.clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            // HP-256 ≈ 77 decimal digits. Dense LU on Strang n=10 with
            // partial pivoting accumulates ~10^-70 ULP; 1e-50 leaves
            // headroom while still catching algorithmic regressions.
            let tol = hp(prec, "1e-50");
            assert!(abs_diff < tol,
                "Strang n=10 dense LU: M·x[{}] - b[{}] = {}",
                i, i, display_hp(&abs_diff, 4));
        }
    }

    /// Equivalence: `lu_solve_with(parallel=true)` and
    /// `lu_solve_with(parallel=false)` must produce the same solution to
    /// working precision. Uses n=80 so rows beyond `PAR_SOLVE_MIN_ROW`
    /// (32) exercise the parallel inner-reduction branch, while early
    /// rows exercise the serial fallback within the same solve.
    #[test]
    fn lu_solve_serial_parallel_equivalence() {
        let prec = 256;
        let n = 80;
        let m = strang_dense(prec, n);
        let factors = lu_factor(&m, n).unwrap();

        // A non-trivial right-hand side: b[i] = (i mod 7) + 1, exact in HP.
        let b: Vec<Float> = (0..n).map(|i| hp(prec, &format!("{}", (i % 7) + 1))).collect();

        let x_par = lu_solve_with(&factors, &b, n, prec, true);
        let x_ser = lu_solve_with(&factors, &b, n, prec, false);

        // The two paths differ only in HP reduction order; the gap must
        // be below the working-precision floor. HP-256 ≈ 77 digits;
        // require agreement to 1e-60.
        let tol = hp(prec, "1e-60");
        for i in 0..n {
            let mut diff = x_par[i].clone(); diff -= &x_ser[i];
            let abs_diff = diff.abs();
            assert!(abs_diff < tol,
                "serial/parallel lu_solve disagree at x[{}]: |Δ| = {}",
                i, display_hp(&abs_diff, 4));
        }

        // And the default `lu_solve` must match the explicit parallel call.
        let x_default = lu_solve(&factors, &b, n, prec);
        for i in 0..n {
            let mut diff = x_default[i].clone(); diff -= &x_par[i];
            assert!(diff.abs() < tol,
                "lu_solve default disagrees with lu_solve_with(parallel=true) at x[{}]", i);
        }
    }

    /// Closed-form: dense LU on Wilkinson W11 + shift handles partial
    /// pivoting on a near-symmetric input where some pivots will be
    /// small (e.g. the row whose original diagonal was 0 - 7 = -7,
    /// which is fine, but the elimination sequence still exercises the
    /// pivoting logic).
    #[test]
    fn lu_factor_solves_wilkinson_w11_shifted() {
        let prec = 512; // a bit higher to expose any cancellation
        let n = 11;
        let m = wilkinson_w11_shifted(prec);

        let factors = lu_factor(&m, n).unwrap();

        // Solve for b = e_0 (first canonical basis vector).
        let mut b = vec![hp_zero(prec); n];
        b[0] = hp(prec, "1");
        let x = lu_solve(&factors, &b, n, prec);

        // M·x = b verification.
        for i in 0..n {
            let mut s = hp_zero(prec);
            for j in 0..n {
                let mut t = m[i * n + j].clone();
                t *= &x[j];
                s += &t;
            }
            let mut diff = s.clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            // HP-512 ≈ 154 decimal digits. Dense LU residual at this size
            // accumulates ~10^-140 ULP; 1e-100 is comfortable headroom.
            let tol = hp(prec, "1e-100");
            assert!(abs_diff < tol,
                "Wilkinson W11+shift LU: M·x[{}] - b[{}] = {}",
                i, i, display_hp(&abs_diff, 4));
        }
    }

    /// Property: solve M·x = b on deterministic-random symmetric
    /// matrices across a sweep of sizes and seeds. Verify M·x reproduces
    /// b to working precision.
    ///
    /// Uses small integer-valued entries from a deterministic LCG so HP
    /// arithmetic is exact in the construction step — the only rounding
    /// happens during LU factor + solve, which is the thing we want to
    /// measure.
    #[test]
    fn lu_property_solve_matches_b() {
        let prec = 256;
        let sizes = [3usize, 5, 8];
        let seeds_per_size = 3;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                // Deterministic-random symmetric matrix with entries in
                // a small integer range. We use a simple construction:
                // diagonal = (n + i mod 3 + 1) (always positive,
                // diagonally dominant), off-diagonal = ±1 alternating.
                let mut a = vec![hp_zero(prec); n * n];
                for i in 0..n {
                    a[i * n + i] = hp(prec, &format!("{}", n + (i % 3) + 1));
                    for j in (i + 1)..n {
                        let val = if (i + j + seed as usize).is_multiple_of(2) { 1 } else { -1 };
                        a[i * n + j] = hp(prec, &val.to_string());
                        a[j * n + i] = hp(prec, &val.to_string());
                    }
                }

                let factors = lu_factor(&a, n).unwrap();

                // Build a non-trivial b.
                let b: Vec<Float> = (0..n).map(|i| {
                    hp(prec, &format!("{}", 1 + (i % 7)))
                }).collect();

                let x = lu_solve(&factors, &b, n, prec);
                assert_eq!(x.len(), n);

                // M·x = b check.
                for i in 0..n {
                    let mut s = hp_zero(prec);
                    for j in 0..n {
                        let mut t = a[i * n + j].clone();
                        t *= &x[j];
                        s += &t;
                    }
                    let mut diff = s.clone(); diff -= &b[i];
                    let abs_diff = diff.abs();
                    let tol = hp(prec, "1e-50");
                    assert!(abs_diff < tol,
                        "n={}, seed={}: M·x[{}] - b[{}] = {}",
                        n, seed, i, i, display_hp(&abs_diff, 4));
                }
            }
        }
    }

    /// HP-1000 residual: dense LU on Strang n=20 at 3338-bit precision.
    /// Production scenario for the dense path; expect the residual to
    /// land near working-precision floor.
    #[test]
    fn lu_at_hp_1000() {
        let prec = 3338;
        let n = 20;
        let m = strang_dense(prec, n);

        let factors = lu_factor(&m, n).unwrap();

        let mut b = vec![hp_zero(prec); n];
        b[n / 2] = hp(prec, "1");
        let x = lu_solve(&factors, &b, n, prec);

        for i in 0..n {
            let mut s = hp_zero(prec);
            for j in 0..n {
                let mut t = m[i * n + j].clone();
                t *= &x[j];
                s += &t;
            }
            let mut diff = s.clone(); diff -= &b[i];
            let abs_diff = diff.abs();
            // At HP-1000 dense LU at n=20 should reach ~10^-900 residual.
            let tol = hp(prec, "1e-900");
            assert!(abs_diff < tol,
                "HP-1000 Strang n=20 LU: M·x[{}] - b[{}] = {}",
                i, i, display_hp(&abs_diff, 4));
        }
    }

    // -----------------------------------------------------------------------
    // inverse_iteration — HEAVY tests
    // -----------------------------------------------------------------------

    /// Property: `inverse_iteration` recovers the smallest eigenvalue of
    /// a deterministic-random *strongly* diagonally dominant matrix, and
    /// the returned eigenvector satisfies A·v ≈ λ·v.
    ///
    /// `inverse_iteration` is governed by *two* convergence floors that
    /// limit the achievable eigenvector residual:
    ///
    /// 1. **Rate floor**: `(λ_min / λ_next)^max_steps`. Active when the
    ///    iteration runs to `max_steps` without triggering early
    ///    termination. Tighter when the smallest eigenvalue is much
    ///    smaller than the next-smallest.
    /// 2. **Rayleigh sqrt-floor**: When the iteration's
    ///    Rayleigh-quotient stability test triggers early termination
    ///    (eigenvalue change `|Δμ/μ| < 2^-(prec-32)`), the eigenvector
    ///    residual is bounded by the *square root* of that threshold
    ///    (Rayleigh's quadratic convergence). At HP-256 this floor is
    ///    `√(2^-224)` ≈ `10^-33`; at HP-1000 it's ≈ `10^-498`.
    ///
    /// We use a strongly diagonally dominant matrix (diag grows as
    /// 1, 11, 21, ...; convergence ratio ~0.09) so the iteration
    /// converges *fast* and triggers early termination — meaning floor
    /// (2) applies. Tolerance set to `1e-25` (8 orders of headroom
    /// above the ~10⁻³³ floor; tightens enough to catch any
    /// algorithmic regression).
    ///
    /// Callers needing tighter residual than this floor should use
    /// the shifted variant in
    /// [`crate::eigen::tridiag_eigenvector_for_value_hp`], which
    /// reaches working precision (~10⁻⁹⁰⁰ at HP-1000).
    #[test]
    fn property_inverse_iteration_recovers_eigenpair() {
        let prec = 256;
        let sizes = [3usize, 5, 8];
        let seeds_per_size = 3;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                // Strongly diagonally dominant: diag[i] = 1 + 10*i. Off-
                // diagonals: small ±1 perturbations. This keeps the
                // smallest eigenvalue near 1 while the next-smallest is
                // near 10, giving an inverse-iteration convergence ratio
                // of ~0.1 — fast enough that early termination triggers
                // early (typically tens of steps).
                let mut a = vec![hp_zero(prec); n * n];
                for i in 0..n {
                    let diag_val = 1 + 10 * (i as i32);
                    a[i * n + i] = hp(prec, &diag_val.to_string());
                    for j in (i + 1)..n {
                        let val = if (i + j + seed as usize).is_multiple_of(3) { 1 }
                                  else if (i + j + seed as usize) % 3 == 1 { -1 }
                                  else { 0 };
                        a[i * n + j] = hp(prec, &val.to_string());
                        a[j * n + i] = hp(prec, &val.to_string());
                    }
                }

                let (mu, v) = inverse_iteration(&a, n, prec, 200, false).unwrap();
                assert_eq!(v.len(), n);

                // A·v - μ·v residual.
                let mut max_resid = hp_zero(prec);
                for i in 0..n {
                    let mut av_i = hp_zero(prec);
                    for j in 0..n {
                        let mut t = a[i * n + j].clone();
                        t *= &v[j];
                        av_i += &t;
                    }
                    let mut mu_v_i = mu.clone();
                    mu_v_i *= &v[i];
                    let mut diff = av_i; diff -= &mu_v_i;
                    let abs_diff = diff.abs();
                    if abs_diff > max_resid {
                        max_resid = abs_diff;
                    }
                }

                // 1e-25 sits 8 orders above the Rayleigh-sqrt floor at
                // HP-256 (~10^-33). An algorithmic regression would
                // produce O(1) error, far above this bound.
                let tol = hp(prec, "1e-25");
                assert!(max_resid < tol,
                    "n={}, seed={}: ‖A·v - μ·v‖_∞ = {} should be < 1e-25",
                    n, seed, display_hp(&max_resid, 4));

                // Eigenvector unit-normalized.
                let mut norm_sq = hp_zero(prec);
                for vi in &v {
                    let mut t = vi.clone();
                    t *= vi;
                    norm_sq += &t;
                }
                let mut nd = norm_sq.clone(); nd -= 1u32;
                let abs_nd = nd.abs();
                let norm_tol = hp(prec, "1e-50");
                assert!(abs_nd < norm_tol,
                    "n={}, seed={}: ‖v‖² should be 1, got {}",
                    n, seed, display_hp(&norm_sq, 6));
            }
        }
    }

    /// HP-1000 inverse-iteration scenario: Strang n=20, recover the
    /// smallest eigenpair and verify the residual lands at the
    /// algorithm's convergence-rate floor (not working precision).
    ///
    /// Strang's eigenvalue gap is `λ_1/λ_2 ≈ 1/4`, so unshifted inverse
    /// iteration has convergence ratio ~0.25 per step. After 200 steps
    /// the residual lands at `0.25^200 ≈ 10⁻¹²⁰` — that's the algorithm
    /// floor for this matrix, not the HP-1000 working-precision floor
    /// (~10⁻⁹⁰⁰). To reach working precision we'd need a shifted
    /// inverse iteration (which `tridiag_eigenvector_for_value_hp`
    /// does and which lands at ~10⁻⁹⁰⁰ as the eigenvector test there
    /// confirms). Tolerance set to 10⁻¹⁰⁰ — comfortably above the
    /// expected algorithm floor and well below any algorithmic
    /// regression.
    #[test]
    fn inverse_iteration_at_hp_1000() {
        let prec = 3338;
        let n = 20;
        let a = strang_dense(prec, n);

        let (mu, v) = inverse_iteration(&a, n, prec, 200, false).unwrap();
        assert_eq!(v.len(), n);

        let mut max_resid = hp_zero(prec);
        for i in 0..n {
            let mut av_i = hp_zero(prec);
            for j in 0..n {
                let mut t = a[i * n + j].clone();
                t *= &v[j];
                av_i += &t;
            }
            let mut mu_v_i = mu.clone();
            mu_v_i *= &v[i];
            let mut diff = av_i; diff -= &mu_v_i;
            let abs_diff = diff.abs();
            if abs_diff > max_resid {
                max_resid = abs_diff;
            }
        }
        let tol = hp(prec, "1e-100");
        assert!(max_resid < tol,
            "HP-1000 Strang n=20 inverse_iteration: ‖A·v - μ·v‖_∞ = {} should be < 1e-100",
            display_hp(&max_resid, 6));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Boundary-condition tests
    // ─────────────────────────────────────────────────────────────────────────

    /// `lu_factor` on a 1×1 matrix: trivial LU, solve must recover the
    /// unique solution.
    #[test]
    fn lu_factor_1x1() {
        let prec = 64;
        // [5] * x = [20] → x = 4
        let a = vec![hp(prec, "5")];
        let b = vec![hp(prec, "20")];
        let factors = lu_factor(&a, 1).expect("1×1 LU should not fail");
        let x = lu_solve(&factors, &b, 1, prec);
        assert_eq!(x.len(), 1);
        let mut diff = x[0].clone(); diff -= hp(prec, "4"); let d = diff.abs();
        assert!(d < hp(prec, "1e-15"), "1×1 solve: |x[0] - 4| = {} should be 0", d);
    }

    /// `lu_factor` on a 1×1 singular (zero) matrix should return an error.
    #[test]
    fn lu_factor_1x1_singular() {
        let prec = 64;
        let a = vec![hp(prec, "0")];
        assert!(lu_factor(&a, 1).is_err(), "1×1 zero matrix should be singular");
    }

    /// `tridiag_lu_factor_hp` on a 1×1 tridiagonal: no off-diagonals.
    #[test]
    fn tridiag_lu_1x1() {
        let prec = 64;
        let diag = vec![hp(prec, "7")];
        let lower: Vec<Float> = Vec::new();
        let upper: Vec<Float> = Vec::new();
        let factors = tridiag_lu_factor_hp(&lower, &diag, &upper, prec)
            .expect("1×1 tridiag LU should not fail");
        let b = vec![hp(prec, "14")];
        let x = tridiag_lu_solve_hp(&factors, &b, prec)
            .expect("1×1 tridiag solve should not fail");
        assert_eq!(x.len(), 1);
        let mut diff = x[0].clone(); diff -= hp(prec, "2"); let d = diff.abs();
        assert!(d < hp(prec, "1e-15"), "1×1 tridiag solve: |x[0] - 2| = {} should be 0", d);
    }

    /// `normalize_l2` on a vector of all-zeros: the norm is 0 so the
    /// function divides by zero. Confirm it does not panic (the output is
    /// NaN or Inf — not our contract to define, but no panic is the key).
    #[test]
    fn normalize_l2_all_zeros_no_panic() {
        let prec = 64;
        let mut v = vec![hp(prec, "0"), hp(prec, "0"), hp(prec, "0")];
        // Should not panic regardless of the NaN/Inf result.
        normalize_l2(&mut v);
        // Output is at least finite in the sense that v was mutated or is NaN/Inf:
        // we just confirm no panic occurred.
    }

    /// `inverse_iteration` on a 1×1 matrix: the only eigenvalue is the
    /// single diagonal entry, and the eigenvector is [1].
    #[test]
    fn inverse_iteration_1x1() {
        let prec = 128;
        let a = vec![hp(prec, "3.14159")];
        let (mu, v) = inverse_iteration(&a, 1, prec, 50, false)
            .expect("1×1 inverse_iteration should succeed");
        assert_eq!(v.len(), 1);
        // The only eigenvalue is 3.14159.
        let mut diff = mu.clone(); diff -= hp(prec, "3.14159"); let d = diff.abs();
        assert!(d < hp(prec, "1e-14"), "1×1 eigenvalue: got {}, expected 3.14159", d);
        // Eigenvector of a 1×1 is [±1]; ℓ² norm should be 1.
        let mut v0_sq = v[0].clone(); v0_sq.square_mut();
        let mut norm_diff = v0_sq; norm_diff -= hp(prec, "1");
        let nd = norm_diff.abs();
        assert!(nd < hp(prec, "1e-15"), "1×1 eigenvector should have ℓ²-norm 1");
    }

    /// `tridiag_lu_factor_hp` with mismatched slice lengths should return
    /// an error, not panic.
    #[test]
    fn tridiag_lu_length_mismatch_returns_error() {
        let prec = 64;
        // n=3 matrix but wrong off-diagonal lengths.
        let diag = vec![hp(prec, "1"), hp(prec, "2"), hp(prec, "3")];
        let bad_lower = vec![hp(prec, "0.5")]; // should be length 2
        let upper = vec![hp(prec, "0.5"), hp(prec, "0.5")];
        assert!(tridiag_lu_factor_hp(&bad_lower, &diag, &upper, prec).is_err(),
            "mismatched lower length should return Err");
    }

    /// `lu_solve` result is the inverse of `lu_factor` — round-trip on a
    /// 2×2 identity matrix should give the identity.
    #[test]
    fn lu_factor_solve_identity_2x2() {
        let prec = 64;
        let dim = 2;
        let mut a = vec![hp(prec, "0"); dim * dim];
        a[0] = hp(prec, "1"); // [0,0]
        a[dim + 1] = hp(prec, "1"); // [1,1]

        let factors = lu_factor(&a, dim).expect("identity LU");
        // Solve A·x = e_0 = [1, 0]
        let b = vec![hp(prec, "1"), hp(prec, "0")];
        let x = lu_solve(&factors, &b, dim, prec);
        assert_eq!(x.len(), 2);
        let mut d0 = x[0].clone(); d0 -= hp(prec, "1"); let d0 = d0.abs();
        let mut d1 = x[1].clone(); d1 -= hp(prec, "0"); let d1 = d1.abs();
        let tol = hp(prec, "1e-15");
        assert!(d0 < tol, "I·x = e_0: x[0] should be 1 (got diff {})", d0);
        assert!(d1 < tol, "I·x = e_0: x[1] should be 0 (got diff {})", d1);
    }
}

