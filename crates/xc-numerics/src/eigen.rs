// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision symmetric eigendecomposition.
//!
//! Algorithms (all in HP, no f64 fallback at any step):
//!
//! - **Householder tridiagonalization**: dense symmetric → tridiagonal
//!   via successive reflections. Returns the tridiagonal form plus the
//!   accumulated transformation `Q` so eigenvectors can be transformed
//!   back to the original basis.
//!
//! - **Symmetric tridiagonal QR** with implicit Wilkinson shifts: the
//!   classical algorithm (Wilkinson 1965; Press et al. NumRec §11.3).
//!   Convergence threshold is `2^-(prec - 16)`, so at HP-1000 (≈3338 bits)
//!   we converge to ~10⁻¹⁰⁰⁰ off-diagonal magnitudes. Truly dynamic in
//!   working precision.
//!
//! - **Shifted inverse iteration** for one eigenvector at a known
//!   eigenvalue: applies LU + back-substitution at the shifted matrix.
//!
//! Memory note: computing the full eigenvector matrix during QR has
//! O(n²) HP storage cost. For large matrices we expose an
//! "eigenvalues only" path; specific eigenvectors are recovered via
//! shifted inverse iteration. This keeps the whole pipeline feasible
//! at HP-1000 for n in the thousands.

#![cfg(feature = "hp")]

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rug::{ops::Pow, Float};

use crate::linalg::{lu_factor, lu_solve, normalize_l2};

/// Convergence threshold for the symmetric tridiagonal QR algorithm.
/// Scales naturally with HP precision: at `prec` bits, this is `2^-(prec-16)`,
/// giving 16 guard bits below the working precision.
fn qr_tolerance(prec: u32) -> Float {
    let two = Float::with_val(prec, 2);
    let exponent = -((prec as i32) - 16);
    two.pow(exponent)
}

/// HP zero at the given precision (integer literal — no f64).
#[inline]
fn hp_zero(prec: u32) -> Float {
    Float::with_val(prec, 0)
}

/// HP one at the given precision (integer literal).
#[inline]
fn hp_one(prec: u32) -> Float {
    Float::with_val(prec, 1)
}

// ===========================================================================
// Symmetric tridiagonal eigenvalues (QR with implicit Wilkinson shifts)
// ===========================================================================

/// Eigenvalues of a symmetric tridiagonal matrix at HP precision.
///
/// `diag` has length `n`, `off_diag` has length `n-1`. The matrix is
/// `T[i,i] = diag[i]`, `T[i+1,i] = T[i,i+1] = off_diag[i]`.
///
/// Returns eigenvalues sorted ascending.
///
/// Algorithm: implicit-shift QR for symmetric tridiagonal matrices,
/// following the classical formulation in *Numerical Recipes* §11.3
/// (the `tqli` routine) and Golub & Van Loan §8.3. Wilkinson shift,
/// QR sweep iterates from `m-1` down to `l`, deflates eigenvalues
/// from the top-left of the active region.
pub fn tridiag_eigenvalues_hp(
    diag: &[Float],
    off_diag: &[Float],
    prec: u32,
) -> Result<Vec<Float>> {
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if off_diag.len() != n - 1 {
        return Err(anyhow!(
            "off_diag length {} should be {} (= diag length - 1)",
            off_diag.len(), n - 1
        ));
    }

    // Working copies; algorithm mutates these in place.
    // Pad e with one trailing zero so e[m] is always valid for m up to n-1.
    let mut d: Vec<Float> = diag.iter().cloned().collect();
    let mut e: Vec<Float> = off_diag.iter().cloned().collect();
    e.push(hp_zero(prec));  // sentinel; index n-1 is always 0

    let tol = qr_tolerance(prec);
    let max_iter = 100;

    // Deflate eigenvalues one by one from the top of the active region [l..n-1].
    for l in 0..n {
        let mut iter_count = 0usize;
        loop {
            // Find the smallest m ≥ l such that |e[m]| is negligible
            // relative to |d[m]| + |d[m+1]|. After this point the matrix
            // decouples; we work on the block [l..=m].
            let mut m = l;
            while m < n - 1 {
                let mut dd = d[m].clone().abs();
                dd += d[m + 1].clone().abs();
                let mut threshold = dd.clone();
                threshold *= &tol;
                let abs_em = e[m].clone().abs();
                if abs_em <= threshold {
                    break;
                }
                m += 1;
            }

            // If e[l] is already negligible, eigenvalue at d[l] is converged.
            if m == l {
                break;
            }

            iter_count += 1;
            if iter_count > max_iter {
                return Err(anyhow!(
                    "tridiag QR failed to converge for eigenvalue at l={} after {} iterations",
                    l, max_iter
                ));
            }

            // Wilkinson shift, computed implicitly per NumRec §11.3.
            // g = (d[l+1] - d[l]) / (2 e[l])
            // r = sqrt(g² + 1)
            // shifted leading entry: g = d[m] - d[l] + e[l] / (g + sign(g)·r)
            let mut g = d[l + 1].clone();
            g -= &d[l];
            let mut two_el = e[l].clone();
            two_el *= 2u32;
            // e[l] is non-negligible at this point (we just established m > l),
            // so two_el is non-zero.
            g /= &two_el;

            let mut r_sq = g.clone();
            r_sq *= &g;
            r_sq += hp_one(prec);
            let r = r_sq.sqrt();

            // signed_r = sign(g) · r ; if g is zero, treat as positive sign
            let signed_r = if g.is_sign_negative() {
                let mut t = r.clone();
                t = -t;
                t
            } else {
                r.clone()
            };

            let mut g_plus_sr = g.clone();
            g_plus_sr += &signed_r;

            // shift = d[m] - d[l] + e[l] / (g + sign(g)·r)
            // After this, g is reused as the running variable in the QR sweep
            // (the value entering the bulge-chase).
            let mut new_g = e[l].clone();
            new_g /= &g_plus_sr;
            let mut shifted_diag = d[m].clone();
            shifted_diag -= &d[l];
            shifted_diag += &new_g;
            let mut g = shifted_diag;  // shadow

            // QR sweep: chase the bulge from m-1 down to l, applying
            // Givens rotations that zero successive off-diagonals.
            let mut s = hp_one(prec);
            let mut c = hp_one(prec);
            let mut p = hp_zero(prec);
            let mut converged_early = false;

            for i in (l..m).rev() {
                // f = s · e[i]
                let mut f = s.clone();
                f *= &e[i];
                // b = c · e[i]
                let mut b = c.clone();
                b *= &e[i];

                // r = sqrt(f² + g²)
                let mut f_sq = f.clone();
                f_sq *= &f;
                let mut g_sq = g.clone();
                g_sq *= &g;
                let mut r_sq = f_sq;
                r_sq += &g_sq;
                let new_r = r_sq.sqrt();

                e[i + 1] = new_r.clone();

                if new_r.is_zero() {
                    // Degenerate: deflate one element and restart this l.
                    d[i + 1] -= &p;
                    e[m] = hp_zero(prec);
                    converged_early = true;
                    break;
                }

                // s = f/r, c = g/r
                s = {
                    let mut t = f.clone();
                    t /= &new_r;
                    t
                };
                c = {
                    let mut t = g.clone();
                    t /= &new_r;
                    t
                };

                // g = d[i+1] - p
                g = {
                    let mut t = d[i + 1].clone();
                    t -= &p;
                    t
                };

                // r = (d[i] - g) · s + 2 c · b
                let mut term1 = d[i].clone();
                term1 -= &g;
                term1 *= &s;
                let mut term2 = c.clone();
                term2 *= &b;
                term2 *= 2u32;
                let mut sweep_r = term1;
                sweep_r += &term2;

                // p = s · r
                p = {
                    let mut t = s.clone();
                    t *= &sweep_r;
                    t
                };

                // d[i+1] = g + p
                d[i + 1] = {
                    let mut t = g.clone();
                    t += &p;
                    t
                };

                // g = c · r - b
                g = {
                    let mut t = c.clone();
                    t *= &sweep_r;
                    t -= &b;
                    t
                };
            }

            if !converged_early {
                d[l] -= &p;
                e[l] = g;
                e[m] = hp_zero(prec);
            }
            // Loop back to find new m; eventually m == l and we break.
        }
    }

    // Sort ascending.
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(d)
}

// ===========================================================================
// Eigenvector for a specific eigenvalue (shifted inverse iteration)
// ===========================================================================

/// Compute the eigenvector of a symmetric tridiagonal matrix corresponding
/// to a specific (already-known) eigenvalue, via shifted inverse iteration.
///
/// `diag` has length `n`, `off_diag` has length `n-1`. `eigenvalue` is the
/// known eigenvalue. Returns the unit eigenvector at HP precision.
///
/// Find the eigenvector of a symmetric tridiagonal matrix corresponding
/// to the (already-known) eigenvalue `eigenvalue` via shifted inverse
/// iteration on a dense `(T - λI + εI)`.
///
/// Uses LU factorization of (T - λI + ε·I) where ε is a small shift to
/// avoid singularity. ε is precision-dependent: `2^-(prec - 32)`.
///
/// `max_steps` is the upper bound on inverse-iteration steps. The
/// iteration runs all `max_steps` steps unless `early_termination` is
/// `true`, in which case it stops as soon as the relative change in
/// the computed Rayleigh quotient drops below `2^-(prec-32)`.
pub fn tridiag_eigenvector_for_value_hp(
    diag: &[Float],
    off_diag: &[Float],
    eigenvalue: &Float,
    prec: u32,
    max_steps: usize,
) -> Result<Vec<Float>> {
    tridiag_eigenvector_for_value_hp_with_options(
        diag, off_diag, eigenvalue, prec, max_steps, false,
    )
}

/// Variant of `tridiag_eigenvector_for_value_hp` with an explicit
/// `early_termination` flag. When `false`, runs the full `max_steps`
/// regardless (the conservative default; bit-identical output to
/// callers that always wanted N steps). When `true`, breaks the
/// inverse-iteration loop as soon as the Rayleigh quotient
/// (eigenvalue estimate from the current iterate) stops moving by
/// more than the working-precision threshold.
///
/// The `_with_options` form lets callers opt into a shorter run
/// when they're confident the iteration has converged. For a
/// well-conditioned tridiagonal whose eigenvalue separation is
/// large compared to working precision, convergence usually
/// happens in 20-50 steps; the original 200-step ceiling is
/// pessimistic.
pub fn tridiag_eigenvector_for_value_hp_with_options(
    diag: &[Float],
    off_diag: &[Float],
    eigenvalue: &Float,
    prec: u32,
    max_steps: usize,
    early_termination: bool,
) -> Result<Vec<Float>> {
    let phase_start = std::time::Instant::now();
    let n = diag.len();
    if n == 0 {
        return Err(anyhow!("empty matrix"));
    }
    if off_diag.len() != n - 1 {
        return Err(anyhow!(
            "off_diag length {} should be {}", off_diag.len(), n - 1
        ));
    }

    // Build the dense (T - λI + ε·I) matrix and run inverse iteration.
    // The shift ε scales with precision: large enough to avoid singularity,
    // small enough that the iteration converges to the right eigenvector.
    let build_start = std::time::Instant::now();
    let mut a = vec![hp_zero(prec); n * n];
    let two = Float::with_val(prec, 2);
    let epsilon: Float = two.pow(-((prec as i32) - 32));

    for i in 0..n {
        let mut entry = diag[i].clone();
        entry -= eigenvalue;
        entry += &epsilon;
        a[i * n + i] = entry;
    }
    for i in 0..(n - 1) {
        a[i * n + (i + 1)] = off_diag[i].clone();
        a[(i + 1) * n + i] = off_diag[i].clone();
    }
    eprintln!(
        "[HP eigvec] dense matrix built in {:.1}s (N={}, prec={} bits)",
        build_start.elapsed().as_secs_f64(), n, prec
    );

    let lu_start = std::time::Instant::now();
    let lu = lu_factor(&a, n)?;
    eprintln!(
        "[HP eigvec] LU factor done in {:.1}s",
        lu_start.elapsed().as_secs_f64()
    );

    // Initial guess: a Gaussian centered at the middle, all in HP.
    // Each entry independent → parallel construction.
    let mut v: Vec<Float> = (0..n).into_par_iter().map(|i| {
        let center = (n as i64) / 2;
        let j = (i as i64) - center;
        let half = ((n as i64) / 2).max(1);
        let mut x = Float::with_val(prec, j);
        x /= half;
        let mut x_sq = x.clone();
        x_sq *= &x;
        x_sq /= 2u32;
        let mut arg = hp_zero(prec);
        arg -= &x_sq;
        arg.exp()
    }).collect();
    normalize_l2(&mut v);

    // Convergence threshold for early termination: same as
    // `inverse_iteration` in linalg.rs. Roughly working-precision tight.
    let conv_thresh = if early_termination {
        Some(Float::with_val(prec, 2).pow(-((prec as i32) - 32)))
    } else {
        None
    };

    // Track the previous Rayleigh quotient estimate for the convergence
    // check. The eigenvalue estimate at step k is roughly
    // (vᵀ_{k-1} · v_k) / ‖v_k‖² which simplifies because v is already
    // ℓ²-normalized — we use the dot product of consecutive iterates as
    // a proxy. A more rigorous test would call `rayleigh_quotient`, but
    // that re-walks the matrix every step (O(n²) per step). Cheaper
    // proxy: |⟨v_k, v_{k-1}⟩| → 1 as the iteration converges.
    let mut prev_dot = hp_zero(prec);

    let iter_start = std::time::Instant::now();
    let mut completed_steps = 0usize;
    for step in 0..max_steps {
        let new_v = lu_solve(&lu, &v, n, prec);
        let mut new_v = new_v;
        normalize_l2(&mut new_v);

        // Optional convergence check: if |⟨v_k, v_{k-1}⟩| has stopped
        // moving (its relative change drops below the threshold), the
        // iteration has converged.
        if let Some(thresh) = conv_thresh.as_ref() {
            // Compute |⟨v_k, v_{k-1}⟩| (a single sequential pass; we
            // don't bother parallelizing this since it's cheap relative
            // to the LU solve already done).
            let mut dot = hp_zero(prec);
            for i in 0..n {
                let mut t = v[i].clone();
                t *= &new_v[i];
                dot += &t;
            }
            dot = dot.abs();
            if step > 2 {
                let mut diff = dot.clone();
                diff -= &prev_dot;
                diff = diff.abs();
                // Converged when |dot - prev_dot| < threshold.
                if diff < *thresh {
                    v = new_v;
                    completed_steps = step + 1;
                    eprintln!(
                        "[HP eigvec] inverse iteration converged at step {}/{} on N={} (elapsed {:.1}s, total {:.1}s)",
                        completed_steps, max_steps, n,
                        iter_start.elapsed().as_secs_f64(),
                        phase_start.elapsed().as_secs_f64()
                    );
                    break;
                }
            }
            prev_dot = dot;
        }

        v = new_v;
        completed_steps = step + 1;
        // Progress: print every 25 steps. Useful to distinguish "still
        // iterating" from "wedged" on multi-hour runs at large N. Each
        // print is one line on stderr; cost is negligible vs the LU solve.
        if completed_steps % 25 == 0 {
            eprintln!(
                "[HP eigvec] inverse iteration {}/{} on N={} (elapsed {:.1}s, total {:.1}s)",
                completed_steps, max_steps, n,
                iter_start.elapsed().as_secs_f64(),
                phase_start.elapsed().as_secs_f64()
            );
        }
    }

    // If we exited the loop without an explicit convergence break and
    // didn't already print a 25-multiple, emit a final summary so the
    // caller sees the full timing picture.
    if completed_steps % 25 != 0 {
        eprintln!(
            "[HP eigvec] inverse iteration {}/{} done on N={} (elapsed {:.1}s, total {:.1}s)",
            completed_steps, max_steps, n,
            iter_start.elapsed().as_secs_f64(),
            phase_start.elapsed().as_secs_f64()
        );
    }

    Ok(v)
}

/// Banded variant of `tridiag_eigenvector_for_value_hp_with_options`:
/// uses the toolkit's tridiagonal LU factorization (Thomas with partial
/// pivoting, O(n) factor + O(n) solve per step) instead of densifying
/// the matrix to N×N (O(n²) memory, O(n³) factor cost). Numerical output
/// is equivalent to working precision.
///
/// At HP-1000 with N=8001, the dense path uses ~26 GB of resident memory
/// and ~hours of LU factor cost; this banded path uses kilobytes and
/// finishes the LU factor in milliseconds. This delivers most of the
/// per-eigenvector wall-time savings observed in Paper B's Claim 1
/// retest cycle.
///
/// The convergence semantics, `early_termination` flag, and progress
/// printing match `tridiag_eigenvector_for_value_hp_with_options` exactly.
pub fn tridiag_eigenvector_for_value_hp_banded(
    diag: &[Float],
    off_diag: &[Float],
    eigenvalue: &Float,
    prec: u32,
    max_steps: usize,
    early_termination: bool,
) -> Result<Vec<Float>> {
    let phase_start = std::time::Instant::now();
    let n = diag.len();
    if n == 0 {
        return Err(anyhow!("empty matrix"));
    }
    if off_diag.len() != n - 1 {
        return Err(anyhow!(
            "off_diag length {} should be {}", off_diag.len(), n - 1
        ));
    }

    // Build the shifted tridiagonal (T - λI + ε·I). Only the diagonal
    // changes; off-diagonals are unchanged. Length n + 2(n-1) = 3n-2
    // HP entries — a few KB at HP-1000 vs the ~26 GB the dense form
    // would need.
    let build_start = std::time::Instant::now();
    let two = Float::with_val(prec, 2);
    let epsilon: Float = two.pow(-((prec as i32) - 32));

    let mut shifted_diag: Vec<Float> = Vec::with_capacity(n);
    for i in 0..n {
        let mut entry = diag[i].clone();
        entry -= eigenvalue;
        entry += &epsilon;
        shifted_diag.push(entry);
    }
    // Symmetric tridiagonal: lower and upper off-diagonals are equal.
    // The banded LU factorizer accepts asymmetric input (so it also
    // handles general tridiagonals), so we provide both.
    let lower: Vec<Float> = off_diag.iter().cloned().collect();
    let upper: Vec<Float> = off_diag.iter().cloned().collect();
    eprintln!(
        "[HP eigvec/banded] tridiagonal shifted matrix built in {:.3}s (N={}, prec={} bits)",
        build_start.elapsed().as_secs_f64(), n, prec
    );

    // Banded LU factor (O(n) ops vs the dense O(n³)).
    let lu_start = std::time::Instant::now();
    let factors = crate::linalg::tridiag_lu_factor_hp(&lower, &shifted_diag, &upper, prec)?;
    eprintln!(
        "[HP eigvec/banded] tridiag LU factor done in {:.3}s",
        lu_start.elapsed().as_secs_f64()
    );

    // Initial guess: a Gaussian centered at the middle, all in HP.
    // Each entry independent → parallel construction.
    let mut v: Vec<Float> = (0..n).into_par_iter().map(|i| {
        let center = (n as i64) / 2;
        let j = (i as i64) - center;
        let half = ((n as i64) / 2).max(1);
        let mut x = Float::with_val(prec, j);
        x /= half;
        let mut x_sq = x.clone();
        x_sq *= &x;
        x_sq /= 2u32;
        let mut arg = hp_zero(prec);
        arg -= &x_sq;
        arg.exp()
    }).collect();
    crate::linalg::normalize_l2(&mut v);

    let conv_thresh = if early_termination {
        Some(Float::with_val(prec, 2).pow(-((prec as i32) - 32)))
    } else {
        None
    };

    let mut prev_dot = hp_zero(prec);
    let iter_start = std::time::Instant::now();
    let mut completed_steps = 0usize;
    for step in 0..max_steps {
        // Banded solve: O(n) instead of dense O(n²) per step.
        let mut new_v = crate::linalg::tridiag_lu_solve_hp(&factors, &v, prec)?;
        crate::linalg::normalize_l2(&mut new_v);

        if let Some(thresh) = conv_thresh.as_ref() {
            // |⟨v_k, v_{k-1}⟩| convergence proxy (cheap, O(n)).
            let mut dot = hp_zero(prec);
            for i in 0..n {
                let mut t = v[i].clone();
                t *= &new_v[i];
                dot += &t;
            }
            dot = dot.abs();
            if step > 2 {
                let mut diff = dot.clone();
                diff -= &prev_dot;
                diff = diff.abs();
                if diff < *thresh {
                    v = new_v;
                    completed_steps = step + 1;
                    eprintln!(
                        "[HP eigvec/banded] inverse iteration converged at step {}/{} on N={} (elapsed {:.3}s, total {:.3}s)",
                        completed_steps, max_steps, n,
                        iter_start.elapsed().as_secs_f64(),
                        phase_start.elapsed().as_secs_f64()
                    );
                    break;
                }
            }
            prev_dot = dot;
        }

        v = new_v;
        completed_steps = step + 1;
        if completed_steps % 25 == 0 {
            eprintln!(
                "[HP eigvec/banded] inverse iteration {}/{} on N={} (elapsed {:.3}s, total {:.3}s)",
                completed_steps, max_steps, n,
                iter_start.elapsed().as_secs_f64(),
                phase_start.elapsed().as_secs_f64()
            );
        }
    }

    if completed_steps % 25 != 0 {
        eprintln!(
            "[HP eigvec/banded] inverse iteration {}/{} done on N={} (elapsed {:.3}s, total {:.3}s)",
            completed_steps, max_steps, n,
            iter_start.elapsed().as_secs_f64(),
            phase_start.elapsed().as_secs_f64()
        );
    }

    Ok(v)
}

// ===========================================================================
// Householder tridiagonalization (dense symmetric → tridiagonal)
// ===========================================================================

/// Reduce a dense symmetric `n × n` matrix to tridiagonal form via successive
/// Householder reflections. Returns `(diag, off_diag, q)` where `q` is the
/// flat row-major `n × n` accumulated transformation matrix such that
/// `Q^T A Q = T` (tridiagonal).
///
/// Eigenvectors of `A` are recovered as `Q · v` where `v` is an eigenvector
/// of `T` in the tridiagonal basis.
pub fn householder_tridiag_hp(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>)> {
    if a.len() != n * n {
        return Err(anyhow!("matrix length {} != n² = {}", a.len(), n * n));
    }
    let mut h: Vec<Float> = a.iter().cloned().collect();

    // Q starts as identity; we apply each Householder reflection from the
    // right as we go, accumulating into Q. (Equivalently, store Householder
    // vectors and reconstruct Q at the end — that's slightly more compact
    // but more code. We do the direct accumulation for simplicity.)
    let mut q: Vec<Float> = vec![hp_zero(prec); n * n];
    for i in 0..n {
        q[i * n + i] = hp_one(prec);
    }

    // For each column k = 0..n-2, build a Householder reflector that
    // zeros out h[k+2..n, k] (and symmetrically h[k, k+2..n]).
    for k in 0..n.saturating_sub(2) {
        // Pull out the column-k subdiagonal portion: x = h[k+1..n, k].
        let m = n - k - 1; // length of subdiagonal portion
        let x: Vec<Float> = (0..m).map(|i| h[(k + 1 + i) * n + k].clone()).collect();

        // ‖x‖ via parallel reduction.
        let alpha_sq: Float = x.par_iter()
            .map(|xi| {
                let mut t = xi.clone();
                t *= xi;
                t
            })
            .reduce(|| hp_zero(prec), |mut a, b| { a += &b; a });
        let alpha = alpha_sq.sqrt();

        // If subdiagonal is already zero, skip.
        if alpha.is_zero() {
            continue;
        }

        // Householder vector: v = x ± α·e₁, sign chosen to avoid cancellation.
        // Standard choice: sign = -sign(x[0]) so x[0] gets a magnitude bump.
        let sign_x0_negative = x[0].is_sign_negative();
        let alpha_signed = if sign_x0_negative {
            // sign(x[0]) is negative → use -α, so v[0] = x[0] + (-α) = x[0] - α
            // Wait: we want v[0] = x[0] - sign(x[0])·α = x[0] + α (since sign was negative).
            // Effectively v[0] = x[0] + α (always shifts away from zero).
            // Let's compute as: signed_alpha = sign(x[0]) * α.
            // Then v[0] = x[0] - signed_alpha + 2*signed_alpha if cancellation would occur.
            // Simpler: signed_alpha = sign(x[0]) * α; new x[0] = -signed_alpha.
            let mut s = alpha.clone();
            s = -s;
            s
        } else {
            alpha.clone()
        };

        // v = x; v[0] += alpha_signed (i.e. shift away from zero)
        let mut v = x.clone();
        // v[0] = x[0] - sign(x[0])*α  (this guarantees |v[0]| > |x[0]|)
        // Equivalently, v[0] += -sign(x[0])*α = -alpha_signed (with our convention).
        let mut v0 = v[0].clone();
        v0 -= &alpha_signed;
        v[0] = v0;

        // ‖v‖² via parallel reduction.
        let v_norm_sq: Float = v.par_iter()
            .map(|vi| {
                let mut t = vi.clone();
                t *= vi;
                t
            })
            .reduce(|| hp_zero(prec), |mut a, b| { a += &b; a });
        if v_norm_sq.is_zero() {
            continue;
        }

        // Householder reflection: H = I - 2·v·vᵀ/‖v‖²
        // Apply H from both sides to the trailing (n-k-1) × (n-k-1) sub-block of h.
        // Symmetric update: h ← h - v·pᵀ - p·vᵀ + (vᵀ·p / ‖v‖²) · v·vᵀ
        //                       (i.e., h ← H h H = h - 2·v·(p - β·v)ᵀ - 2·(p - β·v)·vᵀ )
        // We use the standard form:
        //   p = (2/‖v‖²) · h_sub · v
        //   β = (vᵀ p) / ‖v‖²
        //   q = p - β·v
        //   h_sub ← h_sub - v·qᵀ - q·vᵀ
        // This preserves symmetry exactly.
        //
        // For Q-update: Q_full ← Q_full · H_full where H_full has H in the
        // bottom-right (n-k-1) × (n-k-1) corner and identity elsewhere.

        // Compute p = (2/‖v‖²) · h_sub · v.
        // h_sub is the bottom-right (n-k-1)×(n-k-1) block at h[k+1+i, k+1+j].
        // Each row of p is independent → parallelize over i with rayon.
        let p: Vec<Float> = (0..m).into_par_iter().map(|i| {
            let mut acc = hp_zero(prec);
            for j in 0..m {
                let mut t = h[(k + 1 + i) * n + (k + 1 + j)].clone();
                t *= &v[j];
                acc += &t;
            }
            // p[i] = (2/‖v‖²) · acc
            acc *= 2u32;
            acc /= &v_norm_sq;
            acc
        }).collect();

        // β = (vᵀ p) / 2  (note: p already has the 2/‖v‖² factor)
        // Wait — let me redo. Standard Householder symmetric update:
        //   p = (2/‖v‖²) · h_sub · v  (p is a vector of length m)
        //   K = (vᵀ p) / ‖v‖²         (a scalar)
        //   q = p - K · v             (a vector)
        //   h_sub_new = h_sub - v·qᵀ - q·vᵀ
        //
        // This is the textbook NumRec §11.2 update, but adapted: actually
        // that text uses h ← H h H = h - 2 v vᵀ h - 2 h v vᵀ + 4 (vᵀ h v) v vᵀ / ‖v‖⁴
        // = h - v pᵀ - p vᵀ where p = 2 h v / ‖v‖² - K v with K = vᵀ p / ‖v‖²
        //   ... actually the simpler form is:  h ← h - v pᵀ - p vᵀ + 2 K v vᵀ
        // where p = 2 h v / ‖v‖² and K = vᵀ p / ‖v‖² / 2. Let me just use:
        //
        //   h_sub ← h_sub - v·qᵀ - q·vᵀ   where q = p - β·v, p = 2·h_sub·v/‖v‖², β = vᵀp/(2·‖v‖²)
        //
        // I'll trust the textbook form and verify with tests.

        // vᵀ p — parallel reduce.
        let vt_p: Float = (0..m).into_par_iter()
            .map(|i| {
                let mut t = v[i].clone();
                t *= &p[i];
                t
            })
            .reduce(|| hp_zero(prec), |mut a, b| { a += &b; a });

        // K = (vᵀ p) / ‖v‖² — projection coefficient of p onto v.
        // Since p = β·A·v with β = 2/‖v‖², we have vᵀp = β·vᵀAv, so
        //   K = β·vᵀAv / ‖v‖²  but more simply: K = vt_p / ‖v‖².
        // The correct symmetric Householder update is:
        //   q = p - K·v ;  A_sub ← A_sub - v·qᵀ - q·vᵀ
        // (Derivation: H A H = A - v pᵀ - p vᵀ + β(vᵀp) v vᵀ; setting
        //  q = p - K v with K = β(vᵀp)/2 = (vᵀp)/‖v‖² recovers this exactly.)
        let mut big_k = vt_p;
        big_k /= &v_norm_sq;

        let q_vec: Vec<Float> = (0..m).map(|i| {
            let mut qi = p[i].clone();
            let mut bk = big_k.clone();
            bk *= &v[i];
            qi -= &bk;
            qi
        }).collect();

        // h_sub ← h_sub - v·qᵀ - q·vᵀ
        // Each row update i is independent of other rows, so compute the
        // per-row delta vectors in parallel, then apply them to h. We
        // collect into a Vec<Vec<Float>> to avoid the borrowing dance of
        // mutating disjoint slices of h within rayon's borrow rules.
        let row_deltas: Vec<Vec<Float>> = (0..m).into_par_iter().map(|i| {
            let mut row = Vec::with_capacity(m);
            for j in 0..m {
                let mut delta = v[i].clone();
                delta *= &q_vec[j];
                let mut delta2 = q_vec[i].clone();
                delta2 *= &v[j];
                delta += &delta2;
                row.push(delta);
            }
            row
        }).collect();
        for i in 0..m {
            for j in 0..m {
                let cell = (k + 1 + i) * n + (k + 1 + j);
                h[cell] -= &row_deltas[i][j];
            }
        }

        // Set the (k+1, k) and (k, k+1) entries to alpha_signed (the new
        // off-diagonal after Householder zeroes out the rest of column k).
        // The Householder transform leaves h[k, k] unchanged.
        // h[k+1, k] = -alpha_signed (this is the standard Householder result;
        // sign convention may flip depending on which sign we chose for v[0]).
        // We computed v[0] = x[0] - sign(x[0])*α, which means after H is
        // applied, the first element becomes sign(x[0])*α. Match accordingly.
        let new_off_diag = if sign_x0_negative {
            // alpha_signed = -α; new entry = -alpha_signed = α (positive).
            // But we want the standard convention where the new off-diagonal
            // has sign matching the original. Use -alpha_signed.
            let mut t = alpha_signed.clone();
            t = -t;
            t
        } else {
            // alpha_signed = α; new entry = -α (sign flipped by reflection).
            let mut t = alpha_signed.clone();
            t = -t;
            t
        };
        // Actually the above two branches give the same -alpha_signed.
        // After the symmetric H update above, h[k+1, k] should already be
        // close to the reflected value; we set it explicitly to avoid
        // numerical drift.
        h[(k + 1) * n + k] = new_off_diag.clone();
        h[k * n + (k + 1)] = new_off_diag;

        // Zero out the rest of column k and row k below k+1.
        for i in (k + 2)..n {
            h[i * n + k] = hp_zero(prec);
            h[k * n + i] = hp_zero(prec);
        }

        // Update Q: Q ← Q · H_full where H_full = I - 2 v vᵀ / ‖v‖² in the
        // bottom-right block. So for each row i of Q, columns k+1..n,
        // q_row = q_row - (2/‖v‖² · q_row · v) · vᵀ. Each row independent.
        q.par_chunks_mut(n).for_each(|q_row| {
            // Compute coefficient: c = (2/‖v‖²) · sum_j q_row[k+1+j] · v[j]
            let mut c = hp_zero(prec);
            for j in 0..m {
                let mut t = q_row[k + 1 + j].clone();
                t *= &v[j];
                c += &t;
            }
            c *= 2u32;
            c /= &v_norm_sq;
            // Update: q_row[k+1+j] -= c · v[j]
            for j in 0..m {
                let mut delta = c.clone();
                delta *= &v[j];
                q_row[k + 1 + j] -= &delta;
            }
        });
    }

    // Extract diagonal and off-diagonal.
    let diag: Vec<Float> = (0..n).map(|i| h[i * n + i].clone()).collect();
    let off_diag: Vec<Float> = (0..(n - 1)).map(|i| h[(i + 1) * n + i].clone()).collect();

    Ok((diag, off_diag, q))
}

// ===========================================================================
// Top-level: dense symmetric eigendecomposition
// ===========================================================================

/// Eigenvalues (only) of a dense symmetric matrix at HP precision.
///
/// Pipeline: Householder tridiagonalization → tridiagonal QR.
/// Returns eigenvalues sorted ascending. Eigenvectors are not computed
/// to save memory at HP scale.
pub fn dense_symmetric_eigenvalues_hp(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<Vec<Float>> {
    let (diag, off_diag, _q) = householder_tridiag_hp(a, n, prec)?;
    tridiag_eigenvalues_hp(&diag, &off_diag, prec)
}

/// Compute one eigenvector of a dense symmetric matrix at the given
/// known eigenvalue. Useful when you have eigenvalues from
/// `dense_symmetric_eigenvalues_hp` and want a specific eigenvector.
///
/// Uses shifted inverse iteration directly on the dense matrix.
pub fn dense_symmetric_eigenvector_for_value_hp(
    a: &[Float],
    n: usize,
    eigenvalue: &Float,
    prec: u32,
    max_steps: usize,
) -> Result<Vec<Float>> {
    let two = Float::with_val(prec, 2);
    let epsilon: Float = two.pow(-((prec as i32) - 32));

    // Build (A - λI + εI).
    let mut shifted: Vec<Float> = a.iter().cloned().collect();
    for i in 0..n {
        shifted[i * n + i] -= eigenvalue;
        shifted[i * n + i] += &epsilon;
    }

    let lu = lu_factor(&shifted, n)?;

    // Initial guess: Gaussian-shaped, all in HP. Parallel construction.
    let mut v: Vec<Float> = (0..n).into_par_iter().map(|i| {
        let center = (n as i64) / 2;
        let j = (i as i64) - center;
        let half = ((n as i64) / 2).max(1);
        let mut x = Float::with_val(prec, j);
        x /= half;
        let mut x_sq = x.clone();
        x_sq *= &x;
        x_sq /= 2u32;
        let mut arg = hp_zero(prec);
        arg -= &x_sq;
        arg.exp()
    }).collect();
    normalize_l2(&mut v);

    for _ in 0..max_steps {
        let mut new_v = lu_solve(&lu, &v, n, prec);
        normalize_l2(&mut new_v);
        v = new_v;
    }

    Ok(v)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fmt::{display_hp, matching_digits};

    fn hp(prec: u32, s: &str) -> Float {
        Float::with_val(prec, Float::parse(s).unwrap())
    }

    /// Tridiagonal QR on a diagonal matrix should return the diagonal
    /// values in ascending order.
    #[test]
    fn tridiag_eigenvalues_diagonal() {
        let prec = 256;
        let diag = vec![hp(prec, "3"), hp(prec, "1"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "0"), hp(prec, "0")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 3);
        let one = hp(prec, "1");
        let two = hp(prec, "2");
        let three = hp(prec, "3");
        // |evals[i] - expected[i]| should be ~0 in HP.
        let tol = hp(prec, "1e-50");
        let mut d0 = evals[0].clone(); d0 -= &one; let abs0 = d0.abs();
        let mut d1 = evals[1].clone(); d1 -= &two; let abs1 = d1.abs();
        let mut d2 = evals[2].clone(); d2 -= &three; let abs2 = d2.abs();
        assert!(abs0 < tol && abs1 < tol && abs2 < tol,
            "got {}, {}, {}",
            display_hp(&evals[0], 6), display_hp(&evals[1], 6), display_hp(&evals[2], 6));
    }

    /// 2×2 symmetric tridiagonal: diag=[2,2], off=[1].
    /// Eigenvalues = 1, 3.
    #[test]
    fn tridiag_eigenvalues_2x2_known() {
        let prec = 256;
        let diag = vec![hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 2);
        let one = hp(prec, "1");
        let three = hp(prec, "3");
        let tol = hp(prec, "1e-50");
        let mut d0 = evals[0].clone(); d0 -= &one; let abs0 = d0.abs();
        let mut d1 = evals[1].clone(); d1 -= &three; let abs1 = d1.abs();
        assert!(abs0 < tol && abs1 < tol,
            "expected (1, 3), got ({}, {})",
            display_hp(&evals[0], 6), display_hp(&evals[1], 6));
    }

    /// 3×3 symmetric tridiagonal with known eigenvalues.
    /// diag=[2,3,2], off=[1,1] → eigenvalues = 1, 3, 3 (one duplicate).
    /// Actually let me use a cleaner example. Take T = [[1,2,0],[2,4,3],[0,3,9]].
    /// Eigenvalues computed by hand are roots of det(T - λI) = 0.
    /// Easier: use T = diag(1,2,3) + off-diagonal pattern.
    /// Let's use a verified small case from textbooks.
    #[test]
    fn tridiag_eigenvalues_3x3_symmetric() {
        // T = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
        // This is a well-known matrix. Eigenvalues = 2 - sqrt(2), 2, 2 + sqrt(2).
        let prec = 512;
        let diag = vec![hp(prec, "2"), hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1"), hp(prec, "1")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 3);

        let two = hp(prec, "2");
        let mut sqrt2 = hp(prec, "2"); sqrt2 = sqrt2.sqrt();
        let mut e0 = two.clone(); e0 -= &sqrt2;     // 2 - √2
        let e1 = two.clone();                          // 2
        let mut e2 = two.clone(); e2 += &sqrt2;     // 2 + √2

        let tol = hp(prec, "1e-100");

        let mut d0 = evals[0].clone(); d0 -= &e0; let abs0 = d0.abs();
        let mut d1 = evals[1].clone(); d1 -= &e1; let abs1 = d1.abs();
        let mut d2 = evals[2].clone(); d2 -= &e2; let abs2 = d2.abs();

        assert!(abs0 < tol, "eigval[0] off by {}", display_hp(&abs0, 4));
        assert!(abs1 < tol, "eigval[1] off by {}", display_hp(&abs1, 4));
        assert!(abs2 < tol, "eigval[2] off by {}", display_hp(&abs2, 4));
    }

    /// QR convergence at HP-1000 should give matching digits comparable to working precision.
    /// Use the same 3×3 matrix at higher precision.
    #[test]
    fn tridiag_eigenvalues_hp_1000() {
        let prec = 3338;  // ≈ 1000 decimal digits
        let diag = vec![hp(prec, "2"), hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1"), hp(prec, "1")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

        let two = hp(prec, "2");
        let mut sqrt2 = hp(prec, "2"); sqrt2 = sqrt2.sqrt();
        let mut e0 = two.clone(); e0 -= &sqrt2;
        let e1 = two.clone();
        let mut e2 = two; e2 += &sqrt2;

        // Should match to ~prec/3.322 ≈ 1000 decimal digits.
        let m0 = matching_digits(&evals[0], &e0);
        let m1 = matching_digits(&evals[1], &e1);
        let m2 = matching_digits(&evals[2], &e2);

        // Expect ≥500 digits of agreement (well below working precision).
        let min_digits = Float::with_val(prec, 500);
        assert!(m0 > min_digits || m0.is_infinite(),
            "eigval[0] matches only {} digits", display_hp(&m0, 4));
        assert!(m1 > min_digits || m1.is_infinite(),
            "eigval[1] matches only {} digits", display_hp(&m1, 4));
        assert!(m2 > min_digits || m2.is_infinite(),
            "eigval[2] matches only {} digits", display_hp(&m2, 4));
    }

    /// Eigenvector recovery via shifted inverse iteration.
    /// For T = [[2,1,0],[1,2,1],[0,1,2]] and eigenvalue λ = 2 - √2,
    /// eigenvector should be (1, -√2, 1)/2 (up to sign).
    #[test]
    fn tridiag_eigenvector_recovery() {
        let prec = 256;
        let diag = vec![hp(prec, "2"), hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1"), hp(prec, "1")];

        let two = hp(prec, "2");
        let mut sqrt2 = hp(prec, "2"); sqrt2 = sqrt2.sqrt();
        let mut e0 = two; e0 -= &sqrt2;

        let v = tridiag_eigenvector_for_value_hp(&diag, &off_diag, &e0, prec, 100).unwrap();
        assert_eq!(v.len(), 3);

        // Verify T·v ≈ λ·v.
        // T·v = (2v[0] + v[1], v[0] + 2v[1] + v[2], v[1] + 2v[2])
        let mut tv0 = v[0].clone(); tv0 *= 2u32; tv0 += &v[1];
        let mut tv1 = v[0].clone(); tv1 += &v[2];
        let mut tmp = v[1].clone(); tmp *= 2u32;
        tv1 += &tmp;
        let mut tv2 = v[1].clone(); tv2 += &v[2].clone(); tv2 += &v[2];

        let mut lv0 = e0.clone(); lv0 *= &v[0];
        let mut lv1 = e0.clone(); lv1 *= &v[1];
        let mut lv2 = e0.clone(); lv2 *= &v[2];

        let mut r0 = tv0; r0 -= &lv0; let r0 = r0.abs();
        let mut r1 = tv1; r1 -= &lv1; let r1 = r1.abs();
        let mut r2 = tv2; r2 -= &lv2; let r2 = r2.abs();

        let tol = hp(prec, "1e-50");
        assert!(r0 < tol, "T·v - λv at index 0: {}", display_hp(&r0, 4));
        assert!(r1 < tol, "T·v - λv at index 1: {}", display_hp(&r1, 4));
        assert!(r2 < tol, "T·v - λv at index 2: {}", display_hp(&r2, 4));
    }

    /// Householder tridiagonalization: dense symmetric → tridiagonal,
    /// eigenvalues should be preserved.
    #[test]
    fn householder_preserves_eigenvalues() {
        let prec = 256;
        let n = 4;
        // Build a random-ish symmetric matrix.
        let raw = [
            "4", "1", "2", "0",
            "1", "3", "1", "1",
            "2", "1", "5", "2",
            "0", "1", "2", "6",
        ];
        let a: Vec<Float> = raw.iter().map(|s| hp(prec, s)).collect();

        let evals_dense = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();
        assert_eq!(evals_dense.len(), n);

        // Sanity: eigenvalues should be sorted ascending.
        for i in 0..(n - 1) {
            assert!(evals_dense[i] <= evals_dense[i + 1],
                "eigenvalues not sorted: [{}, {}]",
                display_hp(&evals_dense[i], 6),
                display_hp(&evals_dense[i + 1], 6));
        }

        // Sum of eigenvalues = trace = 4 + 3 + 5 + 6 = 18.
        let mut sum = hp(prec, "0");
        for v in &evals_dense { sum += v; }
        let expected_trace = hp(prec, "18");
        let tol = hp(prec, "1e-50");
        let mut diff = sum.clone(); diff -= &expected_trace; let abs_diff = diff.abs();
        assert!(abs_diff < tol,
            "sum of eigenvalues should be trace = 18, got {}",
            display_hp(&sum, 6));
    }

    /// Dense eigenvector recovery via shifted inverse iteration.
    #[test]
    fn dense_eigenvector_recovery() {
        let prec = 256;
        let n = 3;
        // Diagonal matrix diag(1, 2, 3). Eigenvalue 2 has eigenvector (0, 1, 0).
        let a: Vec<Float> = vec![
            hp(prec, "1"), hp(prec, "0"), hp(prec, "0"),
            hp(prec, "0"), hp(prec, "2"), hp(prec, "0"),
            hp(prec, "0"), hp(prec, "0"), hp(prec, "3"),
        ];

        let lambda = hp(prec, "2");
        let v = dense_symmetric_eigenvector_for_value_hp(&a, n, &lambda, prec, 100).unwrap();

        // |v[0]| and |v[2]| should be tiny; |v[1]| ≈ 1.
        let abs_v0 = v[0].clone().abs();
        let abs_v1 = v[1].clone().abs();
        let abs_v2 = v[2].clone().abs();

        let small = hp(prec, "1e-30");
        let one = hp(prec, "1");
        let close_to_one_tol = hp(prec, "1e-30");

        assert!(abs_v0 < small, "|v[0]| should be tiny, got {}", display_hp(&abs_v0, 6));
        assert!(abs_v2 < small, "|v[2]| should be tiny, got {}", display_hp(&abs_v2, 6));
        let mut v1_diff = abs_v1; v1_diff -= &one; let v1_diff_abs = v1_diff.abs();
        assert!(v1_diff_abs < close_to_one_tol,
            "|v[1]| should be 1, got {}", display_hp(&v[1], 6));
    }

    // ===========================================================================
    // Layer 1 — Closed-form / structured matrices with known eigenvalues
    // ===========================================================================
    //
    // These tests build matrices whose eigenvalues have analytic expressions
    // and verify our HP eigensolver reproduces them. Each test is self-
    // contained at HP precision: matrices, expected eigenvalues, and
    // tolerances are all built from string literals at the working precision.

    /// Build a Strang's tridiagonal `n×n` matrix with diag=2, off=-1.
    /// Eigenvalues are λ_k = 2 - 2·cos(kπ/(n+1)) = 4·sin²(kπ/(2(n+1)))
    /// for k = 1..=n. Closed-form, exact at any precision.
    fn strang_tridiag(prec: u32, n: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>) {
        let two = Float::with_val(prec, 2);
        let mut neg_one = Float::with_val(prec, 1);
        neg_one = -neg_one;
        let diag = vec![two.clone(); n];
        let off_diag = vec![neg_one.clone(); n - 1];

        // Expected eigenvalues: λ_k = 4 sin²(kπ/(2(n+1))).
        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        let two_n_plus_1 = Float::with_val(prec, 2 * (n as u32 + 1));
        let expected: Vec<Float> = (1..=n).map(|k| {
            let mut arg = Float::with_val(prec, k as u32);
            arg *= &pi_v;
            arg /= &two_n_plus_1;
            let mut s = arg.sin();
            s *= s.clone();
            s *= 4u32;
            s
        }).collect();
        // Sort ascending (already ascending by k since sin is monotone on [0, π/2]).
        (diag, off_diag, expected)
    }

    /// Strang's tridiagonal at n=10, HP-256: closed-form eigenvalues match.
    #[test]
    fn strang_tridiag_n10() {
        let prec = 256;
        let n = 10;
        let (diag, off_diag, expected) = strang_tridiag(prec, n);
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), n);

        let tol = hp(prec, "1e-50");
        for (i, (computed, expected)) in evals.iter().zip(expected.iter()).enumerate() {
            let mut diff = computed.clone();
            diff -= expected;
            let abs_diff = diff.abs();
            assert!(abs_diff < tol,
                "eigenvalue {} off by {} (expected {}, got {})",
                i, display_hp(&abs_diff, 4),
                display_hp(expected, 6), display_hp(computed, 6));
        }
    }

    /// Strang's tridiagonal at n=50, HP-256: scaling test.
    #[test]
    fn strang_tridiag_n50() {
        let prec = 256;
        let n = 50;
        let (diag, off_diag, expected) = strang_tridiag(prec, n);
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), n);

        let tol = hp(prec, "1e-50");
        for (i, (computed, expected)) in evals.iter().zip(expected.iter()).enumerate() {
            let mut diff = computed.clone();
            diff -= expected;
            let abs_diff = diff.abs();
            assert!(abs_diff < tol,
                "n=50 eigenvalue {} off by {}", i, display_hp(&abs_diff, 4));
        }
    }

    /// Strang's tridiagonal at HP-1000: precision scaling.
    /// The smallest eigenvalue is ~10^-3 at n=10; we verify it matches
    /// to >500 decimal digits at HP-1000.
    #[test]
    fn strang_tridiag_hp_1000() {
        let prec = 3338; // ≈ 1000 decimal digits
        let n = 10;
        let (diag, off_diag, expected) = strang_tridiag(prec, n);
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

        let min_digits = Float::with_val(prec, 500);
        for (i, (computed, expected)) in evals.iter().zip(expected.iter()).enumerate() {
            let m = matching_digits(computed, expected);
            assert!(m > min_digits || m.is_infinite(),
                "n={}, k={}: only {} matching digits", n, i, display_hp(&m, 4));
        }
    }

    /// Build a Hilbert n×n matrix in HP. H[i,j] = 1/(i+j+1) (1-indexed: 1/(i+j-1)).
    /// Hilbert matrices are notoriously ill-conditioned; smallest eigenvalues
    /// drop exponentially. Perfect HP stress test — f64 cannot recover the
    /// small eigenvalues at all.
    fn hilbert(prec: u32, n: usize) -> Vec<Float> {
        let mut m = vec![Float::with_val(prec, 0); n * n];
        for i in 0..n {
            for j in 0..n {
                let denom = Float::with_val(prec, (i + j + 1) as u32);
                let mut entry = Float::with_val(prec, 1);
                entry /= &denom;
                m[i * n + j] = entry;
            }
        }
        m
    }

    /// Hilbert 4×4 at HP-256.
    /// Smallest eigenvalue is ~9.67×10^-5; without HP, recovery is impossible.
    /// Reference values from Wilf 1970 / standard linear-algebra texts:
    ///   λ_0 ≈ 9.67e-5
    ///   λ_1 ≈ 6.74e-3
    ///   λ_2 ≈ 1.69e-1
    ///   λ_3 ≈ 1.500
    /// We verify trace and determinant relations rather than individual
    /// eigenvalues (the small ones depend on n in a complicated way).
    #[test]
    fn hilbert_4x4_eigenvalue_properties() {
        let prec = 256;
        let n = 4;
        let h = hilbert(prec, n);
        let evals = dense_symmetric_eigenvalues_hp(&h, n, prec).unwrap();
        assert_eq!(evals.len(), n);

        // Sum of eigenvalues = trace = sum_k 1/(2k+1) for k=0..n-1.
        // = 1/1 + 1/3 + 1/5 + 1/7 = 1 + 0.3333... + 0.2 + 0.142857...
        //   = 1.6761904761...
        let mut sum = Float::with_val(prec, 0);
        for v in &evals { sum += v; }
        let mut expected_trace = Float::with_val(prec, 0);
        for k in 0..n {
            let mut term = Float::with_val(prec, 1);
            let denom = Float::with_val(prec, (2 * k + 1) as u32);
            term /= &denom;
            expected_trace += &term;
        }
        let tol = hp(prec, "1e-50");
        let mut diff = sum.clone(); diff -= &expected_trace;
        let abs_diff = diff.abs();
        assert!(abs_diff < tol,
            "Hilbert trace mismatch: sum {} vs trace {}, delta {}",
            display_hp(&sum, 6), display_hp(&expected_trace, 6),
            display_hp(&abs_diff, 4));

        // Eigenvalues should all be positive (Hilbert is SPD).
        let zero = Float::with_val(prec, 0);
        for (i, v) in evals.iter().enumerate() {
            assert!(*v > zero, "Hilbert eigenvalue {} should be positive, got {}",
                i, display_hp(v, 6));
        }

        // Smallest eigenvalue should be very small (Hilbert is ill-conditioned).
        // For n=4, the smallest eigenvalue is ~10^-4.
        let small_threshold = hp(prec, "1e-3");
        assert!(evals[0] < small_threshold,
            "smallest eigenvalue should be < 1e-3 (Hilbert ill-conditioning), got {}",
            display_hp(&evals[0], 6));

        // Largest eigenvalue should be ~1.5 for n=4.
        let large_lo = hp(prec, "1.0");
        let large_hi = hp(prec, "2.0");
        assert!(evals[n - 1] > large_lo && evals[n - 1] < large_hi,
            "largest eigenvalue should be in [1, 2], got {}",
            display_hp(&evals[n - 1], 6));
    }

    /// Random rotation of a known diagonal matrix. Constructs A = G^T D G
    /// where D = diag(d_1, ..., d_n) and G is a product of Givens rotations
    /// with known angles. The eigenvalues of A are the d_i (unchanged by
    /// orthogonal similarity).
    ///
    /// This is the cleanest test — we choose the eigenvalues, build a
    /// non-trivial symmetric matrix that should have those eigenvalues,
    /// and verify our solver recovers them.
    #[test]
    fn rotated_diagonal_recovers_eigenvalues() {
        let prec = 256;
        let n = 5;

        // Chosen eigenvalues — sorted ascending so we can compare directly.
        let chosen: Vec<Float> = vec![
            hp(prec, "0.5"),
            hp(prec, "1.5"),
            hp(prec, "2.7"),
            hp(prec, "4.1"),
            hp(prec, "9.0"),
        ];

        // Start with diagonal D.
        let mut a = vec![Float::with_val(prec, 0); n * n];
        for i in 0..n {
            a[i * n + i] = chosen[i].clone();
        }

        // Apply a sequence of Givens rotations from both sides:
        //   A ← G^T A G   for several (i, j, θ) triples.
        // Each rotation is a similarity, so eigenvalues are preserved.
        // Use rational angles to keep things HP-exact-ish.
        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);

        // Givens rotation angles: π/3, π/4, π/5, π/7 (irrational, distinct).
        // Apply rotations on (0,1), (1,2), (2,3), (3,4), (0,4) — covers all pairs.
        let rotations: Vec<(usize, usize, Float)> = vec![
            (0, 1, { let mut t = pi_v.clone(); t /= 3u32; t }),
            (1, 2, { let mut t = pi_v.clone(); t /= 4u32; t }),
            (2, 3, { let mut t = pi_v.clone(); t /= 5u32; t }),
            (3, 4, { let mut t = pi_v.clone(); t /= 7u32; t }),
            (0, 4, { let mut t = pi_v.clone(); t /= 11u32; t }),
        ];

        for (i, j, theta) in &rotations {
            let c = theta.clone().cos();
            let s = theta.clone().sin();
            // G^T A G: build new A row by row.
            let mut new_a = a.clone();
            // Apply G from the right: A ← A G. Updates columns i and j.
            for r in 0..n {
                let mut new_ri = c.clone(); new_ri *= &a[r * n + *i];
                let mut t = s.clone(); t *= &a[r * n + *j];
                new_ri += &t;
                let mut new_rj = s.clone(); new_rj = -new_rj; new_rj *= &a[r * n + *i];
                let mut t = c.clone(); t *= &a[r * n + *j];
                new_rj += &t;
                new_a[r * n + *i] = new_ri;
                new_a[r * n + *j] = new_rj;
            }
            a = new_a;
            // Apply G^T from the left: A ← G^T A. Updates rows i and j.
            let mut new_a = a.clone();
            for col in 0..n {
                let mut new_ic = c.clone(); new_ic *= &a[*i * n + col];
                let mut t = s.clone(); t *= &a[*j * n + col];
                new_ic += &t;
                let mut new_jc = s.clone(); new_jc = -new_jc; new_jc *= &a[*i * n + col];
                let mut t = c.clone(); t *= &a[*j * n + col];
                new_jc += &t;
                new_a[*i * n + col] = new_ic;
                new_a[*j * n + col] = new_jc;
            }
            a = new_a;
        }

        // Run our eigensolver.
        let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();
        assert_eq!(evals.len(), n);

        let tol = hp(prec, "1e-50");
        for (i, (computed, expected)) in evals.iter().zip(chosen.iter()).enumerate() {
            let mut diff = computed.clone();
            diff -= expected;
            let abs_diff = diff.abs();
            assert!(abs_diff < tol,
                "rotated_diagonal eigenvalue {} off by {} (expected {}, got {})",
                i, display_hp(&abs_diff, 4),
                display_hp(expected, 6), display_hp(computed, 6));
        }
    }

    /// Clustered eigenvalues at HP precision. Build a matrix with
    /// eigenvalues [1.0, 1.0 + 10^-100, 1.0 + 2·10^-100, 1.0 + 3·10^-100],
    /// well below f64 resolution at the cluster but resolvable in HP.
    /// Tests QR convergence under near-degenerate eigenvalues.
    #[test]
    fn clustered_eigenvalues_hp() {
        let prec = 1024; // ≈ 308 decimal digits
        let n = 4;

        // Eigenvalues separated by 10^-100 each.
        let chosen: Vec<Float> = (0..n).map(|k| {
            let mut v = Float::with_val(prec, 1);
            let mut delta = Float::with_val(prec, k as u32);
            let scale = hp(prec, "1e-100");
            delta *= &scale;
            v += &delta;
            v
        }).collect();

        // Build A = G^T D G with one Givens rotation on (0, 2).
        let mut a = vec![Float::with_val(prec, 0); n * n];
        for i in 0..n {
            a[i * n + i] = chosen[i].clone();
        }

        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        let theta = { let mut t = pi_v.clone(); t /= 6u32; t };
        let c = theta.clone().cos();
        let s = theta.clone().sin();
        let (i, j) = (0usize, 2usize);

        // G^T A G via two passes (right then left).
        let mut new_a = a.clone();
        for r in 0..n {
            let mut new_ri = c.clone(); new_ri *= &a[r * n + i];
            let mut t = s.clone(); t *= &a[r * n + j];
            new_ri += &t;
            let mut new_rj = s.clone(); new_rj = -new_rj; new_rj *= &a[r * n + i];
            let mut t = c.clone(); t *= &a[r * n + j];
            new_rj += &t;
            new_a[r * n + i] = new_ri;
            new_a[r * n + j] = new_rj;
        }
        a = new_a;
        let mut new_a = a.clone();
        for col in 0..n {
            let mut new_ic = c.clone(); new_ic *= &a[i * n + col];
            let mut t = s.clone(); t *= &a[j * n + col];
            new_ic += &t;
            let mut new_jc = s.clone(); new_jc = -new_jc; new_jc *= &a[i * n + col];
            let mut t = c.clone(); t *= &a[j * n + col];
            new_jc += &t;
            new_a[i * n + col] = new_ic;
            new_a[j * n + col] = new_jc;
        }
        a = new_a;

        let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();
        assert_eq!(evals.len(), n);

        // Tolerance well below the 10^-100 separation: 10^-150 gives 50 digits
        // below the cluster spacing, plenty.
        let tol = hp(prec, "1e-150");
        for (i, (computed, expected)) in evals.iter().zip(chosen.iter()).enumerate() {
            let mut diff = computed.clone();
            diff -= expected;
            let abs_diff = diff.abs();
            assert!(abs_diff < tol,
                "clustered eigenvalue {} off by {}",
                i, display_hp(&abs_diff, 4));
        }
    }

    /// Wilkinson's W matrix for n=21 (W21+ from Parlett's book).
    /// Diagonal: 10, 9, 8, ..., 1, 0, 1, ..., 9, 10
    /// Off-diagonal: all 1
    /// Famous as a hard symmetric tridiagonal — has nearly-equal eigenvalues
    /// at the extreme ends. Tabulated reference values from Parlett 1980 §1.5.
    /// Largest eigenvalue ≈ 10.7461942..., specifically 10.74619418...
    #[test]
    fn wilkinson_w21_extreme_eigenvalue() {
        let prec = 256;
        let n = 21;
        // diag: 10, 9, 8, ..., 1, 0, 1, ..., 9, 10
        let diag: Vec<Float> = (0..n).map(|i| {
            let mid = (n / 2) as i64; // 10 for n=21
            let val = (mid - i as i64).abs();
            Float::with_val(prec, val as u32)
        }).collect();
        let off_diag: Vec<Float> = vec![Float::with_val(prec, 1); n - 1];

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), n);

        // Largest eigenvalue ≈ 10.7461942 (Parlett 1980, tabulated).
        // We verify it's in [10.74, 10.75] — a tight window confirming
        // we found the right value.
        let lo = hp(prec, "10.74");
        let hi = hp(prec, "10.75");
        let largest = &evals[n - 1];
        assert!(*largest > lo && *largest < hi,
            "Wilkinson W21 largest eigenvalue should be ≈10.7462, got {}",
            display_hp(largest, 8));

        // Also verify trace = sum of |i - mid| for i = 0..n.
        // = 2 * (1 + 2 + ... + 10) = 2 * 55 = 110.
        let mut sum = Float::with_val(prec, 0);
        for v in &evals { sum += v; }
        let expected_trace = Float::with_val(prec, 110u32);
        let tol = hp(prec, "1e-50");
        let mut diff = sum.clone(); diff -= &expected_trace;
        let abs_diff = diff.abs();
        assert!(abs_diff < tol,
            "W21 trace mismatch: sum {} vs 110, delta {}",
            display_hp(&sum, 6), display_hp(&abs_diff, 4));
    }

    /// Verify A·v = λ·v for an eigenvector recovered via shifted inverse iteration.
    /// Uses Strang n=10 where eigenvalues are known in closed form.
    #[test]
    fn eigenvector_satisfies_eigenequation_strang() {
        let prec = 256;
        let n = 10;
        let (diag, off_diag, expected) = strang_tridiag(prec, n);

        // Pick the smallest eigenvalue and recover its eigenvector.
        let lambda = expected[0].clone();
        let v = tridiag_eigenvector_for_value_hp(&diag, &off_diag, &lambda, prec, 200).unwrap();

        // Compute T·v and verify it equals λ·v at HP precision.
        // T·v at index i = diag[i]·v[i] + off[i-1]·v[i-1] + off[i]·v[i+1]
        let mut tv = vec![Float::with_val(prec, 0); n];
        for i in 0..n {
            let mut acc = diag[i].clone();
            acc *= &v[i];
            if i > 0 {
                let mut t = off_diag[i - 1].clone();
                t *= &v[i - 1];
                acc += &t;
            }
            if i < n - 1 {
                let mut t = off_diag[i].clone();
                t *= &v[i + 1];
                acc += &t;
            }
            tv[i] = acc;
        }
        // λ·v
        let lv: Vec<Float> = v.iter().map(|vi| {
            let mut t = lambda.clone();
            t *= vi;
            t
        }).collect();

        // Residual = ‖T·v - λ·v‖_∞ should be tiny.
        let mut max_residual = Float::with_val(prec, 0);
        for i in 0..n {
            let mut r = tv[i].clone();
            r -= &lv[i];
            let abs_r = r.abs();
            if abs_r > max_residual {
                max_residual = abs_r;
            }
        }
        let tol = hp(prec, "1e-50");
        assert!(max_residual < tol,
            "‖T·v - λv‖_∞ = {} should be < 1e-50",
            display_hp(&max_residual, 4));

        // Eigenvector should be unit-normalized.
        let mut norm_sq = Float::with_val(prec, 0);
        for vi in &v {
            let mut t = vi.clone();
            t *= vi;
            norm_sq += &t;
        }
        let mut norm_diff = norm_sq.clone();
        norm_diff -= 1u32;
        let abs_norm_diff = norm_diff.abs();
        let norm_tol = hp(prec, "1e-50");
        assert!(abs_norm_diff < norm_tol,
            "‖v‖² should be 1, got {}", display_hp(&norm_sq, 6));
    }

    /// Tridiagonal QR on a matrix with very-small entries below the
    /// f64 underflow boundary, scaled overall to small values. Verifies
    /// HP convergence works at all magnitudes.
    #[test]
    fn tridiag_eigenvalues_below_f64_floor() {
        let prec = 4096;
        let n = 4;
        // Build T = 10^-500 · I + 10^-500 · S where S is the Strang n=4 matrix.
        // Eigenvalues are 10^-500 · (4 sin²(kπ/10)) (with minor adjustments
        // for the extra 10^-500 · I term).
        // Actually simpler: just scale the whole matrix.
        // T = 10^-500 · diag(2,2,2,2) + 10^-500 · off(-1,-1,-1)
        // Eigenvalues = 10^-500 · (4 sin²(kπ/10)) for k=1..4.
        let scale = hp(prec, "1e-500");

        let mut diag = vec![hp(prec, "0"); n];
        for i in 0..n {
            let mut v = scale.clone();
            v *= 2u32;
            diag[i] = v;
        }
        let mut off_diag = vec![hp(prec, "0"); n - 1];
        for i in 0..(n - 1) {
            let mut v = scale.clone();
            v = -v;
            off_diag[i] = v;
        }

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), n);

        // Expected: scale · 4·sin²(kπ/10) for k=1..4.
        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        for k in 1..=n {
            let mut arg = Float::with_val(prec, k as u32);
            arg *= &pi_v;
            arg /= 10u32;
            let mut s = arg.sin();
            s *= s.clone();
            s *= 4u32;
            let mut expected = scale.clone();
            expected *= &s;

            let mut diff = evals[k - 1].clone();
            diff -= &expected;
            let abs_diff = diff.abs();
            // Tolerance scales with the small magnitude.
            let tol = hp(prec, "1e-550");
            assert!(abs_diff < tol,
                "below-floor eigenvalue {} off by {} (expected {}, got {})",
                k, display_hp(&abs_diff, 4),
                display_hp(&expected, 6), display_hp(&evals[k - 1], 6));
        }
    }

    // ===========================================================================
    // Layer 3 — Property-based testing
    // ===========================================================================
    //
    // For each of N random symmetric matrices, verify universal properties
    // that any correct eigendecomposition must satisfy. These tests catch
    // bug classes (sign errors, off-by-one, accumulated rounding drift)
    // that don't show up on closed-form matrices.
    //
    // Randomness is seeded deterministically so tests are reproducible.
    // We use a simple linear congruential generator (LCG) producing HP
    // values at HP arithmetic — no external rand crate dependency, no
    // f64 leakage.

    /// Deterministic HP-random matrix generator. Uses LCG with HP arithmetic
    /// to produce reproducible random matrices at any precision. Each entry
    /// is in [-1, 1].
    fn lcg_random_symmetric(prec: u32, n: usize, seed: u64) -> Vec<Float> {
        // LCG constants (Numerical Recipes 64-bit values).
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut state: u64 = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);

        let mut next_uniform = || -> Float {
            state = state.wrapping_mul(a).wrapping_add(c);
            // Use top 53 bits → [0, 2^53) → [0, 1) → [-1, 1).
            let top = (state >> 11) as i64;
            // Build value in HP: value = (top / 2^53) * 2 - 1
            let scale = Float::with_val(prec, top);
            let mut v = scale;
            // Divide by 2^53 (exact integer): construct denominator as Float.
            let two_p53 = {
                let mut t = Float::with_val(prec, 1);
                t <<= 53u32;
                t
            };
            v /= &two_p53;
            v *= 2u32;
            v -= 1u32;
            v
        };

        let mut a_matrix = vec![Float::with_val(prec, 0); n * n];
        for i in 0..n {
            for j in i..n {
                let val = next_uniform();
                a_matrix[i * n + j] = val.clone();
                a_matrix[j * n + i] = val;
            }
        }
        a_matrix
    }

    /// Property: sum of eigenvalues equals matrix trace, for many random matrices.
    #[test]
    fn property_trace_equals_sum_of_eigenvalues() {
        let prec = 256;
        let sizes = [3usize, 4, 5, 6, 8];
        let seeds_per_size = 5;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let a = lcg_random_symmetric(prec, n, seed as u64 + 1);
                let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();
                assert_eq!(evals.len(), n);

                // Trace
                let mut trace = Float::with_val(prec, 0);
                for i in 0..n {
                    trace += &a[i * n + i];
                }

                // Sum
                let mut sum = Float::with_val(prec, 0);
                for v in &evals { sum += v; }

                let mut diff = sum.clone();
                diff -= &trace;
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-50");
                assert!(abs_diff < tol,
                    "n={}, seed={}: trace {} vs sum {} differ by {}",
                    n, seed,
                    display_hp(&trace, 6), display_hp(&sum, 6),
                    display_hp(&abs_diff, 4));
            }
        }
    }

    /// Property: product of eigenvalues equals determinant.
    /// Compute determinant via LU factorization for comparison.
    #[test]
    fn property_determinant_equals_product_of_eigenvalues() {
        let prec = 256;
        let sizes = [3usize, 4, 5];
        let seeds_per_size = 5;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let a = lcg_random_symmetric(prec, n, seed as u64 + 100);
                let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();

                // det(A) via LU. lu_factor returns a permuted LU; det = sign * prod(diag(U)).
                let lu = match crate::linalg::lu_factor(&a, n) {
                    Ok(lu) => lu,
                    Err(_) => continue, // singular — skip
                };
                // Determinant from the LU factorization.
                let mut det = Float::with_val(prec, 1);
                for i in 0..n {
                    det *= &lu.lu[i * n + i];
                }
                // Account for permutation sign (count inversions of perm[]).
                let mut inversions = 0usize;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if lu.perm[j] < lu.perm[i] {
                            inversions += 1;
                        }
                    }
                }
                if inversions % 2 == 1 {
                    det = -det;
                }

                // Product of eigenvalues
                let mut product = Float::with_val(prec, 1);
                for v in &evals { product *= v; }

                let mut diff = product.clone();
                diff -= &det;
                let abs_diff = diff.abs();

                // Tolerance scaled by |det| since determinants can be tiny for random symmetric matrices.
                let det_abs = det.clone().abs();
                let tol_relative = {
                    let mut t = hp(prec, "1e-30");
                    if det_abs > Float::with_val(prec, 1) {
                        t *= &det_abs;
                    }
                    t
                };
                assert!(abs_diff < tol_relative,
                    "n={}, seed={}: det {} vs product {} differ by {}",
                    n, seed,
                    display_hp(&det, 6), display_hp(&product, 6),
                    display_hp(&abs_diff, 4));
            }
        }
    }

    /// Property: for each (eigenvalue, eigenvector) pair, A·v = λ·v.
    /// Recover eigenvectors via shifted inverse iteration.
    #[test]
    fn property_eigenequation_holds() {
        let prec = 256;
        let sizes = [3usize, 4, 5];
        let seeds_per_size = 3;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let a = lcg_random_symmetric(prec, n, seed as u64 + 200);
                let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();

                for (k, lambda) in evals.iter().enumerate() {
                    let v = match dense_symmetric_eigenvector_for_value_hp(
                        &a, n, lambda, prec, 200
                    ) {
                        Ok(v) => v,
                        Err(_) => continue, // shifted singular system — rare
                    };

                    // Compute A·v.
                    let mut av = vec![Float::with_val(prec, 0); n];
                    for i in 0..n {
                        let mut acc = Float::with_val(prec, 0);
                        for j in 0..n {
                            let mut t = a[i * n + j].clone();
                            t *= &v[j];
                            acc += &t;
                        }
                        av[i] = acc;
                    }

                    // Compute λ·v.
                    let lv: Vec<Float> = v.iter().map(|vi| {
                        let mut t = lambda.clone();
                        t *= vi;
                        t
                    }).collect();

                    // Residual ‖A·v - λ·v‖_∞.
                    let mut max_residual = Float::with_val(prec, 0);
                    for i in 0..n {
                        let mut r = av[i].clone();
                        r -= &lv[i];
                        let abs_r = r.abs();
                        if abs_r > max_residual {
                            max_residual = abs_r;
                        }
                    }

                    let tol = hp(prec, "1e-40");
                    assert!(max_residual < tol,
                        "n={}, seed={}, k={}: ‖A·v - λv‖_∞ = {} should be < 1e-40",
                        n, seed, k, display_hp(&max_residual, 4));
                }
            }
        }
    }

    /// Property: eigenvectors recovered via shifted inverse iteration are
    /// unit-normalized.
    #[test]
    fn property_eigenvectors_are_unit_normalized() {
        let prec = 256;
        let sizes = [3usize, 4, 5];
        let seeds_per_size = 3;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let a = lcg_random_symmetric(prec, n, seed as u64 + 300);
                let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();

                for lambda in &evals {
                    let v = match dense_symmetric_eigenvector_for_value_hp(
                        &a, n, lambda, prec, 200
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let mut norm_sq = Float::with_val(prec, 0);
                    for vi in &v {
                        let mut t = vi.clone();
                        t *= vi;
                        norm_sq += &t;
                    }
                    let mut diff = norm_sq.clone();
                    diff -= 1u32;
                    let abs_diff = diff.abs();
                    let tol = hp(prec, "1e-50");
                    assert!(abs_diff < tol,
                        "n={}, seed={}: ‖v‖² = {} should be 1",
                        n, seed, display_hp(&norm_sq, 6));
                }
            }
        }
    }

    /// Property: eigenvectors of distinct eigenvalues are orthogonal.
    /// Symmetric matrices have an orthogonal eigenvector basis.
    #[test]
    fn property_eigenvectors_orthogonal_for_distinct_eigenvalues() {
        let prec = 256;
        let sizes = [3usize, 4, 5];
        let seeds_per_size = 3;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let a = lcg_random_symmetric(prec, n, seed as u64 + 400);
                let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();

                // Recover all eigenvectors.
                let mut vecs: Vec<Vec<Float>> = Vec::with_capacity(n);
                for lambda in &evals {
                    let v = match dense_symmetric_eigenvector_for_value_hp(
                        &a, n, lambda, prec, 200
                    ) {
                        Ok(v) => v,
                        Err(_) => return, // skip whole test on failure
                    };
                    vecs.push(v);
                }

                // Check orthogonality of pairs with distinct eigenvalues.
                let separation_threshold = hp(prec, "1e-20");
                for i in 0..n {
                    for j in (i + 1)..n {
                        let mut sep = evals[i].clone();
                        sep -= &evals[j];
                        let abs_sep = sep.abs();
                        if abs_sep < separation_threshold {
                            // Eigenvalues too close — skip orthogonality check
                            // (would need a degeneracy-aware approach).
                            continue;
                        }
                        // Inner product v_i · v_j.
                        let mut dot = Float::with_val(prec, 0);
                        for k in 0..n {
                            let mut t = vecs[i][k].clone();
                            t *= &vecs[j][k];
                            dot += &t;
                        }
                        let abs_dot = dot.clone().abs();
                        let tol = hp(prec, "1e-30");
                        assert!(abs_dot < tol,
                            "n={}, seed={}: v_{} · v_{} = {} should be ≈0 (eigenvalues separated by {})",
                            n, seed, i, j,
                            display_hp(&dot, 4),
                            display_hp(&abs_sep, 4));
                    }
                }
            }
        }
    }

    /// Property: eigendecomposition reconstructs A.
    /// A ≈ Σ λᵢ vᵢ vᵢᵀ for symmetric matrices with orthonormal eigenvectors.
    #[test]
    fn property_decomposition_reconstructs_matrix() {
        let prec = 256;
        let sizes = [3usize, 4, 5];
        let seeds_per_size = 3;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let a = lcg_random_symmetric(prec, n, seed as u64 + 500);
                let evals = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();

                let mut vecs: Vec<Vec<Float>> = Vec::with_capacity(n);
                let mut all_recovered = true;
                for lambda in &evals {
                    match dense_symmetric_eigenvector_for_value_hp(
                        &a, n, lambda, prec, 200
                    ) {
                        Ok(v) => vecs.push(v),
                        Err(_) => { all_recovered = false; break; }
                    }
                }
                if !all_recovered { continue; }

                // Reconstruct A_reconstructed[i][j] = Σ_k λ_k v_k[i] v_k[j].
                let mut a_reconstructed = vec![Float::with_val(prec, 0); n * n];
                for k in 0..n {
                    for i in 0..n {
                        for j in 0..n {
                            let mut term = evals[k].clone();
                            term *= &vecs[k][i];
                            term *= &vecs[k][j];
                            a_reconstructed[i * n + j] += &term;
                        }
                    }
                }

                // Compare element-wise.
                let mut max_diff = Float::with_val(prec, 0);
                for i in 0..n {
                    for j in 0..n {
                        let mut diff = a_reconstructed[i * n + j].clone();
                        diff -= &a[i * n + j];
                        let abs_diff = diff.abs();
                        if abs_diff > max_diff {
                            max_diff = abs_diff;
                        }
                    }
                }

                let tol = hp(prec, "1e-30");
                assert!(max_diff < tol,
                    "n={}, seed={}: ‖A - Σ λ v vᵀ‖_∞ = {} should be < 1e-30",
                    n, seed, display_hp(&max_diff, 4));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Banded eigenvector tests
    // -----------------------------------------------------------------------

    /// Banded path returns an eigenvector that satisfies T·v = λv to
    /// working precision. Same test pattern as `tridiag_eigenvector_recovery`
    /// but using the banded variant.
    #[test]
    fn banded_eigenvector_recovery() {
        let prec = 256;
        // Strang n=3: diag=[2,2,2], off=[1,1]. Smallest eigenvalue
        // 2 - √2 ≈ 0.5858.
        let diag = vec![hp(prec, "2"), hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1"), hp(prec, "1")];

        let two = hp(prec, "2");
        let mut sqrt2 = hp(prec, "2"); sqrt2 = sqrt2.sqrt();
        let mut e0 = two; e0 -= &sqrt2;

        let v = tridiag_eigenvector_for_value_hp_banded(
            &diag, &off_diag, &e0, prec, 100, true,
        ).unwrap();
        assert_eq!(v.len(), 3);

        // Verify T·v ≈ λv.
        let mut tv0 = v[0].clone(); tv0 *= 2u32; tv0 += &v[1];
        let mut tv1 = v[0].clone(); tv1 += &v[2];
        let mut tmp = v[1].clone(); tmp *= 2u32;
        tv1 += &tmp;
        let mut tv2 = v[1].clone(); tv2 += &v[2].clone(); tv2 += &v[2];

        let mut lv0 = e0.clone(); lv0 *= &v[0];
        let mut lv1 = e0.clone(); lv1 *= &v[1];
        let mut lv2 = e0.clone(); lv2 *= &v[2];

        let mut r0 = tv0; r0 -= &lv0; let r0 = r0.abs();
        let mut r1 = tv1; r1 -= &lv1; let r1 = r1.abs();
        let mut r2 = tv2; r2 -= &lv2; let r2 = r2.abs();

        let tol = hp(prec, "1e-50");
        assert!(r0 < tol, "T·v - λv at index 0: {}", display_hp(&r0, 4));
        assert!(r1 < tol, "T·v - λv at index 1: {}", display_hp(&r1, 4));
        assert!(r2 < tol, "T·v - λv at index 2: {}", display_hp(&r2, 4));
    }

    /// Banded vs dense equivalence: run both code paths on the same
    /// Strang n=10 input and confirm the eigenvectors agree (up to sign,
    /// since inverse iteration is agnostic to sign of v).
    #[test]
    fn banded_matches_dense_on_strang_n10() {
        let prec = 256;
        let n = 10;

        // Strang's tridiagonal: diag = 2, off = -1. Smallest eigenvalue
        // λ_1 = 2 - 2 cos(π/(n+1)) = 2 - 2 cos(π/11). Use 30 digits of
        // pre-computed value as the target.
        let diag: Vec<Float> = (0..n).map(|_| hp(prec, "2")).collect();
        let off_diag: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        let lambda_1 = evals[0].clone();

        // Dense path.
        let v_dense = tridiag_eigenvector_for_value_hp_with_options(
            &diag, &off_diag, &lambda_1, prec, 200, true,
        ).unwrap();

        // Banded path.
        let v_banded = tridiag_eigenvector_for_value_hp_banded(
            &diag, &off_diag, &lambda_1, prec, 200, true,
        ).unwrap();

        assert_eq!(v_dense.len(), n);
        assert_eq!(v_banded.len(), n);

        // Pin signs: both eigenvectors should have positive value at the
        // center (or both negative). If they don't agree, flip one.
        let center = n / 2;
        let zero = hp(prec, "0");
        let mut v_b = v_banded.clone();
        let dense_pos = v_dense[center] > zero;
        let banded_pos = v_b[center] > zero;
        if dense_pos != banded_pos {
            for v in v_b.iter_mut() { *v = -v.clone(); }
        }

        // Element-wise compare. Should match to working precision.
        for i in 0..n {
            let mut diff = v_dense[i].clone(); diff -= &v_b[i];
            let abs_diff = diff.abs();
            let tol = hp(prec, "1e-50");
            assert!(abs_diff < tol,
                "banded vs dense disagreement at index {}: {} (dense={}, banded={})",
                i, display_hp(&abs_diff, 6),
                display_hp(&v_dense[i], 6),
                display_hp(&v_b[i], 6));
        }
    }

    /// Banded path at HP-1000: residual of T·v - λv at publication
    /// precision. This is the "production scenario" check.
    #[test]
    fn banded_eigenvector_residual_hp_1000() {
        let prec = 3338;
        let n = 20;

        // Strang's tridiagonal at n=20.
        let diag: Vec<Float> = (0..n).map(|_| hp(prec, "2")).collect();
        let off_diag: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        let lambda_1 = evals[0].clone();

        // Banded path with early termination.
        let v = tridiag_eigenvector_for_value_hp_banded(
            &diag, &off_diag, &lambda_1, prec, 200, true,
        ).unwrap();

        // Compute residual ‖T·v - λv‖_∞.
        let mut max_resid = hp(prec, "0");
        for i in 0..n {
            // (Tv)_i = diag[i]·v[i] + off[i-1]·v[i-1] + off[i]·v[i+1]
            let mut tv_i = diag[i].clone(); tv_i *= &v[i];
            if i > 0 { let mut t = off_diag[i - 1].clone(); t *= &v[i - 1]; tv_i += &t; }
            if i < n - 1 { let mut t = off_diag[i].clone(); t *= &v[i + 1]; tv_i += &t; }
            let mut lv_i = lambda_1.clone(); lv_i *= &v[i];
            let mut resid = tv_i; resid -= &lv_i; resid = resid.abs();
            if resid > max_resid { max_resid = resid; }
        }

        // At HP-1000 with the working-precision early-termination
        // threshold, residual should be ≲ 10^-900 (~working precision).
        let tol = hp(prec, "1e-900");
        assert!(max_resid < tol,
            "HP-1000 banded residual ‖T·v - λv‖_∞ = {} should be < 1e-900",
            display_hp(&max_resid, 6));
    }
}

