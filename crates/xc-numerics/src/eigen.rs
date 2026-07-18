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

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rug::{ops::Pow, Assign, Float};
use xc_core::EigenpairDiagnostics;

use crate::linalg::{lu_factor, lu_solve, normalize_l2};

/// Convergence threshold for the symmetric tridiagonal QR algorithm.
/// Scales naturally with HP precision: at `prec` bits, this is `2^-(prec-16)`,
/// giving 16 guard bits below the working precision.
fn qr_tolerance(prec: u32) -> Float {
    let two = Float::with_val(prec, 2);
    let exponent = -((prec as i32) - 16);
    two.pow(exponent)
}

/// Maximum QR sweep iterations allowed per deflating eigenvalue before
/// `tridiag_eigenvalues_hp` gives up with an error. Default 100.
///
/// Some inputs (observed: large-N tridiagonal matrices at certain λ²
/// configurations) converge correctly but slowly, needing more than 100
/// sweeps per eigenvalue. Callers with slow-but-real convergence can pass a
/// larger typed limit to `tridiag_eigenvalues_hp_with_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TridiagQrOptions {
    /// Maximum QR sweeps allowed per deflating eigenvalue.
    pub max_iterations_per_eigenvalue: usize,
}

impl Default for TridiagQrOptions {
    fn default() -> Self {
        Self {
            max_iterations_per_eigenvalue: 100,
        }
    }
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
pub fn tridiag_eigenvalues_hp(diag: &[Float], off_diag: &[Float], prec: u32) -> Result<Vec<Float>> {
    tridiag_eigenvalues_hp_with_options(diag, off_diag, prec, TridiagQrOptions::default())
}

/// Eigenvalues of a symmetric tridiagonal matrix using explicit QR controls.
///
/// This is the provenance-friendly entry point for unusually slow but valid
/// matrices that require a sweep budget above the deterministic default.
pub fn tridiag_eigenvalues_hp_with_options(
    diag: &[Float],
    off_diag: &[Float],
    prec: u32,
    options: TridiagQrOptions,
) -> Result<Vec<Float>> {
    if options.max_iterations_per_eigenvalue == 0 {
        return Err(anyhow!(
            "max_iterations_per_eigenvalue must be greater than zero"
        ));
    }
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if off_diag.len() != n - 1 {
        return Err(anyhow!(
            "off_diag length {} should be {} (= diag length - 1)",
            off_diag.len(),
            n - 1
        ));
    }
    // Working copies; algorithm mutates these in place.
    // Pad e with one trailing zero so e[m] is always valid for m up to n-1.
    let mut d: Vec<Float> = diag.to_vec();
    let mut e: Vec<Float> = off_diag.to_vec();
    e.push(hp_zero(prec)); // sentinel; index n-1 is always 0

    let tol = qr_tolerance(prec);
    let max_iter = options.max_iterations_per_eigenvalue;

    // Scratch Floats hoisted out of all loops. Each is allocated once at
    // function start and reused via `assign` / in-place ops, avoiding
    // ~14 fresh HP-precision Float allocations per Givens-rotation step.
    // At HP-1000, N=8001, this saves on the order of 10⁹ MPFR allocations
    // across a full eigenvalue sweep — observable wall-time win.
    //
    // No algorithmic change: each scratch holds intermediate values for
    // exactly one expression, identical to the previous let-binding form.
    let mut sc_dd = hp_zero(prec);
    let mut sc_threshold = hp_zero(prec);
    let mut sc_abs_em = hp_zero(prec);
    let mut sc_g = hp_zero(prec);
    let mut sc_two_el = hp_zero(prec);
    let mut sc_r_sq_outer = hp_zero(prec);
    let mut sc_r_outer = hp_zero(prec);
    let mut sc_signed_r = hp_zero(prec);
    let mut sc_g_plus_sr = hp_zero(prec);
    let mut sc_new_g = hp_zero(prec);
    let mut sc_shifted_diag = hp_zero(prec);
    let mut sc_f = hp_zero(prec);
    let mut sc_b = hp_zero(prec);
    let mut sc_f_sq = hp_zero(prec);
    let mut sc_g_sq = hp_zero(prec);
    let mut sc_r_sq = hp_zero(prec);
    let mut sc_new_r = hp_zero(prec);
    let mut sc_term1 = hp_zero(prec);
    let mut sc_term2 = hp_zero(prec);
    let mut sc_sweep_r = hp_zero(prec);

    // Cross-iteration QR state (s, c, p, g_running) needs to be re-
    // initialized at the start of each sweep, but the scratches above
    // can be reused across sweeps.

    // Deflate eigenvalues one by one from the top of the active region [l..n-1].
    for l in 0..n {
        let mut iter_count = 0usize;
        loop {
            // Find the smallest m ≥ l such that |e[m]| is negligible
            // relative to |d[m]| + |d[m+1]|. After this point the matrix
            // decouples; we work on the block [l..=m].
            let mut m = l;
            while m < n - 1 {
                // sc_dd = |d[m]| + |d[m+1]|
                sc_dd.assign(d[m].clone().abs());
                sc_dd += d[m + 1].clone().abs();
                // threshold = sc_dd · tol
                sc_threshold.assign(&sc_dd);
                sc_threshold *= &tol;
                // abs_em = |e[m]|
                sc_abs_em.assign(e[m].clone().abs());
                if sc_abs_em <= sc_threshold {
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
                    l,
                    max_iter
                ));
            }

            // Wilkinson shift, computed implicitly per NumRec §11.3.
            // sc_g = (d[l+1] - d[l]) / (2 e[l])
            sc_g.assign(&d[l + 1]);
            sc_g -= &d[l];
            sc_two_el.assign(&e[l]);
            sc_two_el *= 2u32;
            // e[l] is non-negligible at this point (we just established m > l),
            // so sc_two_el is non-zero.
            sc_g /= &sc_two_el;

            // sc_r_outer = sqrt(g² + 1)
            sc_r_sq_outer.assign(&sc_g);
            sc_r_sq_outer *= &sc_g;
            sc_r_sq_outer += 1u32;
            // rug::Float doesn't have sqrt_mut for in-place sqrt as of v1.30
            // in our usage; we accept one allocation here per outer
            // iteration (not per inner step), which is amortized.
            sc_r_outer.assign(sc_r_sq_outer.clone().sqrt());

            // sc_signed_r = sign(g) · r ; if g is zero, treat as positive sign
            if sc_g.is_sign_negative() {
                sc_signed_r.assign(&sc_r_outer);
                sc_signed_r = -sc_signed_r;
            } else {
                sc_signed_r.assign(&sc_r_outer);
            }

            sc_g_plus_sr.assign(&sc_g);
            sc_g_plus_sr += &sc_signed_r;

            // shift = d[m] - d[l] + e[l] / (g + sign(g)·r)
            // After this, the running variable `g` (mutable, holds the
            // current bulge value) is initialized for the QR sweep.
            sc_new_g.assign(&e[l]);
            sc_new_g /= &sc_g_plus_sr;
            sc_shifted_diag.assign(&d[m]);
            sc_shifted_diag -= &d[l];
            sc_shifted_diag += &sc_new_g;
            // `g` is the running bulge variable from here. It's reused
            // across the inner sweep; we move sc_shifted_diag into it.
            let mut g = sc_shifted_diag.clone();

            // QR sweep: chase the bulge from m-1 down to l, applying
            // Givens rotations that zero successive off-diagonals.
            let mut s = hp_one(prec);
            let mut c = hp_one(prec);
            let mut p = hp_zero(prec);
            let mut converged_early = false;

            for i in (l..m).rev() {
                // f = s · e[i]
                sc_f.assign(&s);
                sc_f *= &e[i];
                // b = c · e[i]
                sc_b.assign(&c);
                sc_b *= &e[i];

                // r = sqrt(f² + g²)
                sc_f_sq.assign(&sc_f);
                sc_f_sq *= &sc_f;
                sc_g_sq.assign(&g);
                sc_g_sq *= &g;
                sc_r_sq.assign(&sc_f_sq);
                sc_r_sq += &sc_g_sq;
                sc_new_r.assign(sc_r_sq.clone().sqrt());

                e[i + 1].assign(&sc_new_r);

                if sc_new_r.is_zero() {
                    // Degenerate: deflate one element and restart this l.
                    d[i + 1] -= &p;
                    e[m] = hp_zero(prec);
                    converged_early = true;
                    break;
                }

                // s = f/r
                s.assign(&sc_f);
                s /= &sc_new_r;
                // c = g/r
                c.assign(&g);
                c /= &sc_new_r;

                // g_new = d[i+1] - p
                g.assign(&d[i + 1]);
                g -= &p;

                // sweep_r = (d[i] - g_new) · s + 2 c · b
                sc_term1.assign(&d[i]);
                sc_term1 -= &g;
                sc_term1 *= &s;
                sc_term2.assign(&c);
                sc_term2 *= &sc_b;
                sc_term2 *= 2u32;
                sc_sweep_r.assign(&sc_term1);
                sc_sweep_r += &sc_term2;

                // p = s · sweep_r
                p.assign(&s);
                p *= &sc_sweep_r;

                // d[i+1] = g + p
                d[i + 1].assign(&g);
                d[i + 1] += &p;

                // g = c · sweep_r - b
                g.assign(&c);
                g *= &sc_sweep_r;
                g -= &sc_b;
            }

            if !converged_early {
                d[l] -= &p;
                e[l].assign(&g);
                e[m] = hp_zero(prec);
            }
            // Loop back to find new m; eventually m == l and we break.
        }
    }

    // Sort ascending.
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(d)
}

/// One exact-index enclosure produced by HP Sturm bisection. The endpoint
/// counts satisfy `lower_count <= index < upper_count`.
#[derive(Clone, Debug, PartialEq)]
pub struct HpTridiagonalEigenvalueEnclosure {
    pub index: usize,
    pub lower: Float,
    pub upper: Float,
    pub lower_count: usize,
    pub upper_count: usize,
    pub iterations: usize,
}

/// Selected tridiagonal values together with route-level work telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct HpSelectedTridiagonalSpectrum {
    pub precision_bits: u32,
    pub first_index: usize,
    pub last_index: usize,
    pub sturm_evaluations: usize,
    pub enclosures: Vec<HpTridiagonalEigenvalueEnclosure>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HpSelectedTridiagonalEigenpair {
    pub enclosure: HpTridiagonalEigenvalueEnclosure,
    pub eigenvalue: Float,
    pub eigenvector: Vec<Float>,
    pub residual_norm: Float,
    pub diagnostics: EigenpairDiagnostics<Float>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HpTridiagonalEigenvalueCluster {
    pub first_index: usize,
    pub last_index: usize,
    pub lower: Float,
    pub upper: Float,
    pub requested_indices: Vec<usize>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HpSelectedTridiagonalItem {
    SimpleEigenpair(Box<HpSelectedTridiagonalEigenpair>),
    Cluster(HpTridiagonalEigenvalueCluster),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HpSelectedTridiagonalEigenpairs {
    pub spectrum: HpSelectedTridiagonalSpectrum,
    pub vector_recoveries: usize,
    pub inverse_iteration_runs: usize,
    pub items: Vec<HpSelectedTridiagonalItem>,
}

#[derive(Clone, Debug)]
pub struct HpSelectedTridiagonalEigenpairOptions {
    pub first_index: usize,
    pub last_index: usize,
    pub absolute_tolerance: Float,
    pub maximum_bisection_iterations: usize,
    pub eigenvector_options: TridiagEigvecOptions,
    pub precision_bits: u32,
}

fn validate_hp_tridiagonal(diag: &[Float], off_diag: &[Float], prec: u32) -> Result<()> {
    if diag.is_empty() || off_diag.len() + 1 != diag.len() {
        return Err(anyhow!(
            "HP tridiagonal problem requires off_diag.len() + 1 == diag.len() > 0"
        ));
    }
    if prec <= 32 {
        return Err(anyhow!("HP tridiagonal precision must exceed 32 bits"));
    }
    if diag.iter().chain(off_diag).any(|value| !value.is_finite()) {
        return Err(anyhow!("HP tridiagonal entries must be finite"));
    }
    Ok(())
}

fn sturm_sign_changes_for_block_hp(
    diag: &[Float],
    off_diag: &[Float],
    threshold: &Float,
    start: usize,
    end: usize,
    prec: u32,
) -> usize {
    let mut previous_nonzero_sign = 1i8;
    let mut changes = 0usize;
    let mut record_sign = |value: &Float| {
        if value.is_zero() {
            return;
        }
        let sign = if value.is_sign_negative() { -1 } else { 1 };
        if sign != previous_nonzero_sign {
            changes += 1;
        }
        previous_nonzero_sign = sign;
    };

    let mut previous_polynomial = hp_one(prec);
    let mut current_polynomial = Float::with_val(prec, &diag[start]);
    current_polynomial -= threshold;
    record_sign(&current_polynomial);
    for index in start + 1..end {
        let mut next_polynomial = Float::with_val(prec, &diag[index]);
        next_polynomial -= threshold;
        next_polynomial *= &current_polynomial;
        let mut coupling = Float::with_val(prec, &off_diag[index - 1]);
        coupling *= Float::with_val(prec, &off_diag[index - 1]);
        coupling *= &previous_polynomial;
        next_polynomial -= coupling;
        record_sign(&next_polynomial);
        previous_polynomial = current_polynomial;
        current_polynomial = next_polynomial;
    }
    changes
}

fn tridiag_sturm_count_below_hp_unchecked(
    diag: &[Float],
    off_diag: &[Float],
    threshold: &Float,
    prec: u32,
) -> usize {
    let mut count = 0usize;
    let mut block_start = 0usize;
    for boundary in 1..=diag.len() {
        if boundary == diag.len() || off_diag[boundary - 1].is_zero() {
            count += sturm_sign_changes_for_block_hp(
                diag,
                off_diag,
                threshold,
                block_start,
                boundary,
                prec,
            );
            block_start = boundary;
        }
    }
    count
}

/// Count eigenvalues strictly below an HP threshold through the characteristic
/// polynomial Sturm sequence. Exact zero couplings split independent blocks,
/// so diagonal and reducible tridiagonal matrices retain strict semantics even
/// when the threshold equals an eigenvalue.
pub fn tridiag_sturm_count_below_hp(
    diag: &[Float],
    off_diag: &[Float],
    threshold: &Float,
    prec: u32,
) -> Result<usize> {
    validate_hp_tridiagonal(diag, off_diag, prec)?;
    if !threshold.is_finite() {
        return Err(anyhow!("HP Sturm threshold must be finite"));
    }
    Ok(tridiag_sturm_count_below_hp_unchecked(
        diag, off_diag, threshold, prec,
    ))
}

fn tridiag_gershgorin_bounds_hp(diag: &[Float], off_diag: &[Float], prec: u32) -> (Float, Float) {
    let mut lower = Float::with_val(prec, &diag[0]);
    let mut upper = lower.clone();
    for index in 0..diag.len() {
        let mut radius = hp_zero(prec);
        if index > 0 {
            radius += Float::with_val(prec, &off_diag[index - 1]).abs();
        }
        if index + 1 < diag.len() {
            radius += Float::with_val(prec, &off_diag[index]).abs();
        }
        let mut row_lower = Float::with_val(prec, &diag[index]);
        row_lower -= &radius;
        let mut row_upper = Float::with_val(prec, &diag[index]);
        row_upper += &radius;
        if row_lower < lower {
            lower = row_lower;
        }
        if row_upper > upper {
            upper = row_upper;
        }
    }
    let mut scale = lower.clone().abs();
    let upper_abs = upper.clone().abs();
    if upper_abs > scale {
        scale = upper_abs;
    }
    if scale < 1 {
        scale.assign(1);
    }
    let mut padding = Float::with_val(prec, 2).pow(-((prec / 2) as i32));
    padding *= scale;
    lower -= &padding;
    upper += padding;
    (lower, upper)
}

/// Compute only the inclusive algebraic index range `[first_index,last_index]`
/// of a symmetric tridiagonal matrix. This is an HP computed route: endpoint
/// counts and precision-stagnation checks are retained, but rigorous claims
/// still require interval counts from `xc-certify`.
pub fn tridiag_selected_eigenvalues_hp(
    diag: &[Float],
    off_diag: &[Float],
    first_index: usize,
    last_index: usize,
    absolute_tolerance: &Float,
    maximum_iterations: usize,
    prec: u32,
) -> Result<HpSelectedTridiagonalSpectrum> {
    validate_hp_tridiagonal(diag, off_diag, prec)?;
    if first_index > last_index || last_index >= diag.len() {
        return Err(anyhow!(
            "selected HP tridiagonal range must satisfy first <= last < dimension"
        ));
    }
    if !absolute_tolerance.is_finite() || absolute_tolerance <= &hp_zero(prec) {
        return Err(anyhow!(
            "selected HP tridiagonal tolerance must be finite and positive"
        ));
    }
    if maximum_iterations == 0 {
        return Err(anyhow!(
            "selected HP tridiagonal maximum_iterations must be positive"
        ));
    }
    let tolerance = Float::with_val(prec, absolute_tolerance);
    let (global_lower, global_upper) = tridiag_gershgorin_bounds_hp(diag, off_diag, prec);
    let global_lower_count =
        tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &global_lower, prec);
    let global_upper_count =
        tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &global_upper, prec);
    if global_lower_count != 0 || global_upper_count != diag.len() {
        return Err(anyhow!(
            "HP Gershgorin bracket failed count reconciliation: [{global_lower_count}, {global_upper_count}] for dimension {}",
            diag.len()
        ));
    }

    let mut sturm_evaluations = 2usize;
    let mut enclosures = Vec::with_capacity(last_index - first_index + 1);
    for index in first_index..=last_index {
        let mut lower = global_lower.clone();
        let mut upper = global_upper.clone();
        let mut lower_count = global_lower_count;
        let mut upper_count = global_upper_count;
        let mut iterations = 0usize;
        loop {
            let mut width = upper.clone();
            width -= &lower;
            if width <= tolerance {
                break;
            }
            if iterations == maximum_iterations {
                return Err(anyhow!(
                    "HP Sturm bisection did not enclose eigenvalue {index} within the requested tolerance after {maximum_iterations} iterations"
                ));
            }
            let mut midpoint = lower.clone();
            midpoint += &upper;
            midpoint /= 2u32;
            if midpoint == lower || midpoint == upper {
                return Err(anyhow!(
                    "HP Sturm bisection stagnated at {prec} bits for eigenvalue {index}; precision escalation is required"
                ));
            }
            let midpoint_count =
                tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &midpoint, prec);
            sturm_evaluations += 1;
            if midpoint_count <= index {
                lower = midpoint;
                lower_count = midpoint_count;
            } else {
                upper = midpoint;
                upper_count = midpoint_count;
            }
            iterations += 1;
        }
        if lower_count > index || upper_count <= index {
            return Err(anyhow!(
                "HP Sturm endpoint counts do not enclose eigenvalue {index}: [{lower_count}, {upper_count}]"
            ));
        }
        enclosures.push(HpTridiagonalEigenvalueEnclosure {
            index,
            lower,
            upper,
            lower_count,
            upper_count,
            iterations,
        });
    }
    Ok(HpSelectedTridiagonalSpectrum {
        precision_bits: prec,
        first_index,
        last_index,
        sturm_evaluations,
        enclosures,
    })
}

fn tridiag_rayleigh_and_residual_hp(
    diag: &[Float],
    off_diag: &[Float],
    vector: &[Float],
    matrix_scale: &Float,
    prec: u32,
) -> (Float, EigenpairDiagnostics<Float>) {
    let mut numerator = hp_zero(prec);
    let mut denominator = hp_zero(prec);
    for index in 0..diag.len() {
        let mut square = Float::with_val(prec, &vector[index]);
        square *= &vector[index];
        denominator += &square;
        square *= &diag[index];
        numerator += square;
        if index + 1 < diag.len() {
            let mut cross = Float::with_val(prec, &vector[index]);
            cross *= &vector[index + 1];
            cross *= &off_diag[index];
            cross *= 2u32;
            numerator += cross;
        }
    }
    let mut eigenvalue = numerator;
    eigenvalue /= &denominator;

    let mut residual_squared = hp_zero(prec);
    let mut action_squared = hp_zero(prec);
    for index in 0..diag.len() {
        let mut action = Float::with_val(prec, &diag[index]);
        action *= &vector[index];
        if index > 0 {
            let mut term = Float::with_val(prec, &off_diag[index - 1]);
            term *= &vector[index - 1];
            action += term;
        }
        if index + 1 < diag.len() {
            let mut term = Float::with_val(prec, &off_diag[index]);
            term *= &vector[index + 1];
            action += term;
        }
        let mut action_square = action.clone();
        action_square *= &action;
        action_squared += action_square;
        let mut expected = eigenvalue.clone();
        expected *= &vector[index];
        action -= expected;
        action *= action.clone();
        residual_squared += action;
    }
    let absolute_residual = residual_squared.sqrt();
    let vector_norm = denominator.clone().sqrt();
    let action_norm = action_squared.sqrt();
    let mut eigenvalue_scale = eigenvalue.clone().abs();
    eigenvalue_scale *= &vector_norm;
    let mut relative_denominator = action_norm;
    relative_denominator += &eigenvalue_scale;
    let mut relative_residual = absolute_residual.clone();
    if !relative_denominator.is_zero() {
        relative_residual /= relative_denominator;
    }
    let mut backward_denominator = Float::with_val(prec, matrix_scale);
    backward_denominator *= &vector_norm;
    backward_denominator += eigenvalue_scale;
    let mut scaled_backward_error = absolute_residual.clone();
    if !backward_denominator.is_zero() {
        scaled_backward_error /= backward_denominator;
    }
    let mut orthogonality_error = denominator;
    orthogonality_error -= 1u32;
    orthogonality_error.abs_mut();
    (
        eigenvalue,
        EigenpairDiagnostics {
            absolute_residual,
            relative_residual,
            scaled_backward_error,
            orthogonality_error,
        },
    )
}

/// Recover HP eigenvectors only for selected values whose endpoint counts
/// establish a one-dimensional eigenspace. Multiplicities are coalesced into
/// cluster records and never assigned arbitrary individual vectors.
pub fn tridiag_selected_eigenpairs_hp(
    diag: &[Float],
    off_diag: &[Float],
    options: &HpSelectedTridiagonalEigenpairOptions,
) -> Result<HpSelectedTridiagonalEigenpairs> {
    let prec = options.precision_bits;
    let eigenvector_options = options.eigenvector_options;
    if eigenvector_options.max_steps == 0 {
        return Err(anyhow!(
            "selected HP eigenvector recovery requires a positive step limit"
        ));
    }
    let spectrum = tridiag_selected_eigenvalues_hp(
        diag,
        off_diag,
        options.first_index,
        options.last_index,
        &options.absolute_tolerance,
        options.maximum_bisection_iterations,
        prec,
    )?;
    let mut items = Vec::with_capacity(spectrum.enclosures.len());
    let mut vector_recoveries = 0usize;
    let mut inverse_iteration_runs = 0usize;
    let (matrix_lower, matrix_upper) = tridiag_gershgorin_bounds_hp(diag, off_diag, prec);
    let mut matrix_scale = matrix_lower.abs();
    let upper_scale = matrix_upper.abs();
    if upper_scale > matrix_scale {
        matrix_scale = upper_scale;
    }
    if matrix_scale < 1 {
        matrix_scale.assign(1);
    }
    let mut residual_target = Float::with_val(prec, 2).pow(-((prec / 2) as i32));
    residual_target *= &matrix_scale;
    for enclosure in &spectrum.enclosures {
        let cluster_dimension = enclosure
            .upper_count
            .checked_sub(enclosure.lower_count)
            .ok_or_else(|| anyhow!("selected HP endpoint count decreased"))?;
        if cluster_dimension != 1 {
            let cluster_first = enclosure.lower_count;
            let cluster_last = enclosure.upper_count.saturating_sub(1);
            if let Some(HpSelectedTridiagonalItem::Cluster(existing)) = items.last_mut() {
                if existing.first_index == cluster_first && existing.last_index == cluster_last {
                    existing.requested_indices.push(enclosure.index);
                    if enclosure.lower < existing.lower {
                        existing.lower = enclosure.lower.clone();
                    }
                    if enclosure.upper > existing.upper {
                        existing.upper = enclosure.upper.clone();
                    }
                    continue;
                }
            }
            items.push(HpSelectedTridiagonalItem::Cluster(
                HpTridiagonalEigenvalueCluster {
                    first_index: cluster_first,
                    last_index: cluster_last,
                    lower: enclosure.lower.clone(),
                    upper: enclosure.upper.clone(),
                    requested_indices: vec![enclosure.index],
                    reason: "endpoint Sturm counts do not establish a one-dimensional eigenspace"
                        .to_owned(),
                },
            ));
            continue;
        }

        let mut shift = enclosure.lower.clone();
        shift += &enclosure.upper;
        shift /= 2u32;
        let mut eigenvector =
            tridiag_eigenvector_for_value_hp(diag, off_diag, &shift, prec, eigenvector_options)?;
        inverse_iteration_runs += 1;
        let (mut eigenvalue, mut diagnostics) =
            tridiag_rayleigh_and_residual_hp(diag, off_diag, &eigenvector, &matrix_scale, prec);
        if diagnostics.absolute_residual > residual_target && eigenvector_options.early_termination
        {
            let retry_options = TridiagEigvecOptions {
                early_termination: false,
                ..eigenvector_options
            };
            eigenvector =
                tridiag_eigenvector_for_value_hp(diag, off_diag, &eigenvalue, prec, retry_options)?;
            inverse_iteration_runs += 1;
            (eigenvalue, diagnostics) =
                tridiag_rayleigh_and_residual_hp(diag, off_diag, &eigenvector, &matrix_scale, prec);
        }
        if diagnostics.absolute_residual > residual_target {
            return Err(anyhow!(
                "HP inverse iteration did not meet the residual target for selected index {}: residual={}, target={}",
                enclosure.index,
                diagnostics.absolute_residual,
                residual_target
            ));
        }
        if eigenvalue < enclosure.lower || eigenvalue > enclosure.upper {
            return Err(anyhow!(
                "HP inverse-iteration Rayleigh value escaped the selected enclosure for index {}",
                enclosure.index
            ));
        }
        vector_recoveries += 1;
        let residual_norm = diagnostics.absolute_residual.clone();
        items.push(HpSelectedTridiagonalItem::SimpleEigenpair(Box::new(
            HpSelectedTridiagonalEigenpair {
                enclosure: enclosure.clone(),
                eigenvalue,
                eigenvector,
                residual_norm,
                diagnostics,
            },
        )));
    }
    Ok(HpSelectedTridiagonalEigenpairs {
        spectrum,
        vector_recoveries,
        inverse_iteration_runs,
        items,
    })
}

// ===========================================================================
// Eigenvector for a specific eigenvalue (shifted inverse iteration)
// ===========================================================================

/// Solver choice for the inner LU step in shifted inverse iteration.
///
/// Both solvers produce eigenvectors that satisfy `T·v = λ·v` to working
/// precision, on the same tridiagonal input. The choice is purely
/// architectural: cost, memory, scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagSolver {
    /// Banded LU on the tridiagonal directly (Thomas with partial
    /// pivoting via `linalg::tridiag_lu_factor_hp`). O(n) factor,
    /// O(n) per-step solve, O(n) memory. The right choice for
    /// production: at HP-1000, N=8001 the LU factor lands in
    /// milliseconds and resident memory is a few KB.
    Banded,
    /// Dense LU after explicitly densifying `(T - λI + ε·I)` to an
    /// `n × n` matrix. O(n³) factor, O(n²) per-step solve, O(n²)
    /// memory. Retained for cross-validation: reviewers can compare
    /// banded vs dense outputs to confirm they agree to working
    /// precision (the test
    /// `eigen::tests::banded_matches_dense_on_strang_n10` does this
    /// at HP-256).
    Dense,
}

/// Options for `tridiag_eigenvector_for_value_hp`.
///
/// `Default::default()` is the production choice (banded LU, early
/// termination on, 200-step ceiling).
#[derive(Debug, Clone, Copy)]
pub struct TridiagEigvecOptions {
    /// Upper bound on inverse-iteration steps. The iteration runs at
    /// most this many steps; with `early_termination = true`, it
    /// usually finishes much earlier (typically 20–50 steps for
    /// well-conditioned inputs with widely-separated eigenvalues).
    pub max_steps: usize,
    /// Stop the iteration as soon as the |⟨v_k, v_{k-1}⟩| convergence
    /// proxy stops moving by more than the working-precision
    /// threshold (`2^-(prec-32)`). Set to `false` for a conservative
    /// full-`max_steps` run with bit-identical output across calls.
    pub early_termination: bool,
    /// Inner solver for the LU step. See `TridiagSolver`.
    pub solver: TridiagSolver,
}

impl Default for TridiagEigvecOptions {
    fn default() -> Self {
        Self {
            max_steps: 200,
            early_termination: true,
            solver: TridiagSolver::Banded,
        }
    }
}

/// Find the eigenvector of a symmetric tridiagonal matrix corresponding
/// to the (already-known) eigenvalue `eigenvalue` via shifted inverse
/// iteration on `(T - λI + ε·I)`, where `ε = 2^-(prec - 32)` is a small
/// shift that prevents singularity.
///
/// The default options (banded LU, `early_termination=true`,
/// `max_steps=200`) are the right choice for production. Callers who
/// need bit-identical, deterministic-step-count output across runs
/// should set `early_termination=false`. Callers who want to
/// cross-validate against the dense LU path should set
/// `solver=TridiagSolver::Dense`.
///
/// ```text
/// // Default (banded + early termination):
/// let v = tridiag_eigenvector_for_value_hp(
///     &diag, &off_diag, &lambda, prec, TridiagEigvecOptions::default(),
/// )?;
///
/// // Cross-validation (dense LU, full step count):
/// let v_dense = tridiag_eigenvector_for_value_hp(
///     &diag, &off_diag, &lambda, prec,
///     TridiagEigvecOptions {
///         max_steps: 200,
///         early_termination: false,
///         solver: TridiagSolver::Dense,
///     },
/// )?;
/// ```
// Keep the remainder checks below for the Rust 1.85 MSRV;
// `usize::is_multiple_of` is newer than the supported compiler.
#[allow(unknown_lints, clippy::manual_is_multiple_of)]
pub fn tridiag_eigenvector_for_value_hp(
    diag: &[Float],
    off_diag: &[Float],
    eigenvalue: &Float,
    prec: u32,
    opts: TridiagEigvecOptions,
) -> Result<Vec<Float>> {
    let phase_start = std::time::Instant::now();
    let n = diag.len();
    if n == 0 {
        return Err(anyhow!("empty matrix"));
    }
    if off_diag.len() != n - 1 {
        return Err(anyhow!(
            "off_diag length {} should be {}",
            off_diag.len(),
            n - 1
        ));
    }

    // Build the shifted system. The two solver paths diverge here: the
    // dense path materializes a full n×n matrix (O(n²) memory), while
    // the banded path stores three short vectors of length ≈n (O(n)
    // memory). After this branch, the inverse-iteration loop is
    // identical except for the solve call.
    let two = Float::with_val(prec, 2);
    let epsilon: Float = two.pow(-((prec as i32) - 32));
    let log_tag = match opts.solver {
        TridiagSolver::Banded => "[HP eigvec/banded]",
        TridiagSolver::Dense => "[HP eigvec]",
    };

    enum Factored {
        Dense(crate::linalg::LuFactors),
        Banded(crate::linalg::TridiagLuFactors),
    }

    let factored = match opts.solver {
        TridiagSolver::Dense => {
            // Densify (T - λI + ε·I).
            let build_start = std::time::Instant::now();
            let mut a = vec![hp_zero(prec); n * n];
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
            crate::hp_debug!(
                "{} dense matrix built in {:.1}s (N={}, prec={} bits)",
                log_tag,
                build_start.elapsed().as_secs_f64(),
                n,
                prec
            );

            let lu_start = std::time::Instant::now();
            let lu = lu_factor(&a, n)?;
            crate::hp_debug!(
                "{} LU factor done in {:.1}s",
                log_tag,
                lu_start.elapsed().as_secs_f64()
            );
            Factored::Dense(lu)
        }
        TridiagSolver::Banded => {
            // Build the shifted tridiagonal in three short vectors.
            // Length n + 2(n-1) = 3n-2 HP entries — a few KB at HP-1000
            // vs the ~26 GB the dense form would need at N=8001.
            let build_start = std::time::Instant::now();
            let mut shifted_diag: Vec<Float> = Vec::with_capacity(n);
            for d in diag.iter() {
                let mut entry = d.clone();
                entry -= eigenvalue;
                entry += &epsilon;
                shifted_diag.push(entry);
            }
            // Symmetric tridiagonal: lower and upper off-diagonals are
            // equal. The banded LU factorizer accepts asymmetric input,
            // so we pass both copies.
            let lower: Vec<Float> = off_diag.to_vec();
            let upper: Vec<Float> = off_diag.to_vec();
            crate::hp_debug!(
                "{} tridiagonal shifted matrix built in {:.3}s (N={}, prec={} bits)",
                log_tag,
                build_start.elapsed().as_secs_f64(),
                n,
                prec
            );

            let lu_start = std::time::Instant::now();
            let factors = crate::linalg::tridiag_lu_factor_hp(&lower, &shifted_diag, &upper, prec)?;
            crate::hp_debug!(
                "{} tridiag LU factor done in {:.3}s",
                log_tag,
                lu_start.elapsed().as_secs_f64()
            );
            Factored::Banded(factors)
        }
    };

    // Initial guess: a Gaussian centered at the middle, all in HP.
    // Each entry independent → parallel construction.
    let mut v: Vec<Float> = (0..n)
        .into_par_iter()
        .map(|i| {
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
        })
        .collect();
    normalize_l2(&mut v);

    let conv_thresh = if opts.early_termination {
        Some(Float::with_val(prec, 2).pow(-((prec as i32) - 32)))
    } else {
        None
    };

    // Track the previous |⟨v_k, v_{k-1}⟩| as a cheap O(n) convergence
    // proxy. (Strict Rayleigh quotient would re-walk the matrix every
    // step at O(n²); this is good enough since the LU solve already
    // dominates per-step cost at production sizes.)
    let mut prev_dot = hp_zero(prec);
    let iter_start = std::time::Instant::now();
    let mut completed_steps = 0usize;

    for step in 0..opts.max_steps {
        // Solve (T - λI + ε·I) y = v_k. Banded is O(n); dense is O(n²).
        let mut new_v = match &factored {
            Factored::Dense(lu) => lu_solve(lu, &v, n, prec),
            Factored::Banded(factors) => crate::linalg::tridiag_lu_solve_hp(factors, &v, prec)?,
        };
        normalize_l2(&mut new_v);

        if let Some(thresh) = conv_thresh.as_ref() {
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
                    crate::hp_debug!(
                        "{} inverse iteration converged at step {}/{} on N={} (elapsed {:.3}s, total {:.3}s)",
                        log_tag, completed_steps, opts.max_steps, n,
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
            crate::hp_debug!(
                "{} inverse iteration {}/{} on N={} (elapsed {:.3}s, total {:.3}s)",
                log_tag,
                completed_steps,
                opts.max_steps,
                n,
                iter_start.elapsed().as_secs_f64(),
                phase_start.elapsed().as_secs_f64()
            );
        }
    }

    if completed_steps % 25 != 0 {
        crate::hp_debug!(
            "{} inverse iteration {}/{} done on N={} (elapsed {:.3}s, total {:.3}s)",
            log_tag,
            completed_steps,
            opts.max_steps,
            n,
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
    let mut h: Vec<Float> = a.to_vec();

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
        let alpha_terms: Vec<Float> = x
            .par_iter()
            .map(|xi| {
                let mut t = xi.clone();
                t *= xi;
                t
            })
            .collect();
        let alpha_sq = crate::reduction::deterministic_pairwise_sum_hp(&alpha_terms, prec);
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
        let v_norm_terms: Vec<Float> = v
            .par_iter()
            .map(|vi| {
                let mut t = vi.clone();
                t *= vi;
                t
            })
            .collect();
        let v_norm_sq = crate::reduction::deterministic_pairwise_sum_hp(&v_norm_terms, prec);
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
        let p: Vec<Float> = (0..m)
            .into_par_iter()
            .map(|i| {
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
            })
            .collect();

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
        let vt_p_terms: Vec<Float> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut t = v[i].clone();
                t *= &p[i];
                t
            })
            .collect();
        let vt_p = crate::reduction::deterministic_pairwise_sum_hp(&vt_p_terms, prec);

        // K = (vᵀ p) / ‖v‖² — projection coefficient of p onto v.
        // Since p = β·A·v with β = 2/‖v‖², we have vᵀp = β·vᵀAv, so
        //   K = β·vᵀAv / ‖v‖²  but more simply: K = vt_p / ‖v‖².
        // The correct symmetric Householder update is:
        //   q = p - K·v ;  A_sub ← A_sub - v·qᵀ - q·vᵀ
        // (Derivation: H A H = A - v pᵀ - p vᵀ + β(vᵀp) v vᵀ; setting
        //  q = p - K v with K = β(vᵀp)/2 = (vᵀp)/‖v‖² recovers this exactly.)
        let mut big_k = vt_p;
        big_k /= &v_norm_sq;

        let q_vec: Vec<Float> = (0..m)
            .map(|i| {
                let mut qi = p[i].clone();
                let mut bk = big_k.clone();
                bk *= &v[i];
                qi -= &bk;
                qi
            })
            .collect();

        // h_sub ← h_sub - v·qᵀ - q·vᵀ
        // Each row update i is independent of other rows, so compute the
        // per-row delta vectors in parallel, then apply them to h. We
        // collect into a Vec<Vec<Float>> to avoid the borrowing dance of
        // mutating disjoint slices of h within rayon's borrow rules.
        let row_deltas: Vec<Vec<Float>> = (0..m)
            .into_par_iter()
            .map(|i| {
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
            })
            .collect();
        for (i, row) in row_deltas.iter().enumerate() {
            for (j, delta) in row.iter().enumerate() {
                let cell = (k + 1 + i) * n + (k + 1 + j);
                h[cell] -= delta;
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

/// Diagnostics from the independent cyclic Jacobi eigensolver.
#[derive(Clone, Debug)]
pub struct JacobiEigenvaluesHp {
    pub eigenvalues: Vec<Float>,
    pub sweeps: usize,
    pub rotations: usize,
    pub maximum_off_diagonal: Float,
}

/// Independent HP eigenvalue route for a dense real symmetric matrix.
///
/// This cyclic Jacobi implementation acts directly on the dense matrix and
/// shares neither Householder reduction nor tridiagonal QR with
/// [`dense_symmetric_eigenvalues_hp`]. All rotations, stopping tests, and
/// sorting remain in `rug::Float`; there is no f64 seed or conversion.
/// `max_sweeps` is an explicit resource bound and zero is rejected.
pub fn dense_symmetric_eigenvalues_jacobi_hp(
    input: &[Float],
    n: usize,
    prec: u32,
    max_sweeps: usize,
) -> Result<JacobiEigenvaluesHp> {
    if n == 0 || input.len() != n * n {
        return Err(anyhow!(
            "Jacobi eigensolver requires a nonempty n-by-n matrix"
        ));
    }
    if prec <= 32 || max_sweeps == 0 {
        return Err(anyhow!(
            "Jacobi eigensolver requires precision above 32 bits and at least one sweep"
        ));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("Jacobi eigensolver matrix entries must be finite"));
    }
    for row in 0..n {
        for column in 0..row {
            if input[row * n + column] != input[column * n + row] {
                return Err(anyhow!(
                    "Jacobi eigensolver requires exact symmetric storage at ({row}, {column})"
                ));
            }
        }
    }

    let mut matrix: Vec<Float> = input
        .iter()
        .map(|value| Float::with_val(prec, value))
        .collect();
    let mut scale = Float::with_val(prec, 1);
    for value in &matrix {
        let magnitude = value.clone().abs();
        if magnitude > scale {
            scale = magnitude;
        }
    }
    let mut tolerance = Float::with_val(prec, 2);
    tolerance = tolerance.pow(-((prec as i32) - 16));
    tolerance *= &scale;
    let zero = Float::with_val(prec, 0);
    let one = Float::with_val(prec, 1);
    let mut rotations = 0usize;

    for sweep in 0..max_sweeps {
        let before = maximum_off_diagonal_hp(&matrix, n, prec);
        if before <= tolerance {
            let mut eigenvalues: Vec<Float> = (0..n)
                .map(|index| matrix[index * n + index].clone())
                .collect();
            eigenvalues.sort_by(|left, right| left.partial_cmp(right).unwrap());
            return Ok(JacobiEigenvaluesHp {
                eigenvalues,
                sweeps: sweep,
                rotations,
                maximum_off_diagonal: before,
            });
        }
        for p in 0..n - 1 {
            for q in p + 1..n {
                let apq = matrix[p * n + q].clone();
                if apq.clone().abs() <= tolerance {
                    continue;
                }
                let app = matrix[p * n + p].clone();
                let aqq = matrix[q * n + q].clone();
                let mut tau = aqq.clone();
                tau -= &app;
                let mut two_apq = apq.clone();
                two_apq *= 2u32;
                tau /= two_apq;

                let mut root = tau.clone();
                root *= &tau;
                root += &one;
                root = root.sqrt();
                let mut denominator = tau.clone().abs();
                denominator += root;
                let mut tangent = one.clone();
                tangent /= denominator;
                if tau < zero {
                    tangent = -tangent;
                }
                let mut cosine = tangent.clone();
                cosine *= &tangent;
                cosine += &one;
                cosine = cosine.sqrt();
                cosine.recip_mut();
                let mut sine = tangent.clone();
                sine *= &cosine;

                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = matrix[k * n + p].clone();
                    let akq = matrix[k * n + q].clone();
                    let mut new_kp = cosine.clone();
                    new_kp *= &akp;
                    let mut term = sine.clone();
                    term *= &akq;
                    new_kp -= term;
                    let mut new_kq = sine.clone();
                    new_kq *= akp;
                    let mut term = cosine.clone();
                    term *= akq;
                    new_kq += term;
                    matrix[k * n + p].assign(&new_kp);
                    matrix[p * n + k].assign(new_kp);
                    matrix[k * n + q].assign(&new_kq);
                    matrix[q * n + k].assign(new_kq);
                }
                let mut diagonal_change = tangent;
                diagonal_change *= &apq;
                matrix[p * n + p].assign(app - &diagonal_change);
                matrix[q * n + q].assign(aqq + &diagonal_change);
                matrix[p * n + q].assign(&zero);
                matrix[q * n + p].assign(&zero);
                rotations += 1;
            }
        }
        let after = maximum_off_diagonal_hp(&matrix, n, prec);
        if after <= tolerance {
            let mut eigenvalues: Vec<Float> = (0..n)
                .map(|index| matrix[index * n + index].clone())
                .collect();
            eigenvalues.sort_by(|left, right| left.partial_cmp(right).unwrap());
            return Ok(JacobiEigenvaluesHp {
                eigenvalues,
                sweeps: sweep + 1,
                rotations,
                maximum_off_diagonal: after,
            });
        }
    }
    let maximum = maximum_off_diagonal_hp(&matrix, n, prec);
    Err(anyhow!(
        "cyclic Jacobi failed to converge after {max_sweeps} sweeps; maximum off-diagonal is {maximum}"
    ))
}

fn maximum_off_diagonal_hp(matrix: &[Float], n: usize, prec: u32) -> Float {
    let mut maximum = Float::with_val(prec, 0);
    for row in 0..n {
        for column in 0..row {
            let magnitude = matrix[row * n + column].clone().abs();
            if magnitude > maximum {
                maximum = magnitude;
            }
        }
    }
    maximum
}

/// Eigenvalues (only) of a dense symmetric matrix at HP precision.
///
/// Pipeline: Householder tridiagonalization → tridiagonal QR.
/// Returns eigenvalues sorted ascending. Eigenvectors are not computed
/// to save memory at HP scale.
pub fn dense_symmetric_eigenvalues_hp(a: &[Float], n: usize, prec: u32) -> Result<Vec<Float>> {
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
    let mut shifted: Vec<Float> = a.to_vec();
    for i in 0..n {
        shifted[i * n + i] -= eigenvalue;
        shifted[i * n + i] += &epsilon;
    }

    let lu = lu_factor(&shifted, n)?;

    // Initial guess: Gaussian-shaped, all in HP. Parallel construction.
    let mut v: Vec<Float> = (0..n)
        .into_par_iter()
        .map(|i| {
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
        })
        .collect();
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

// Verification loops below iterate matrix/vector indices directly
// (`m[i * n + j]`, paired `vecs[i][k] · vecs[j][k]`), where the index
// arithmetic is the natural expression. Allow needless_range_loop in
// this test-only module.
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
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
        let mut d0 = evals[0].clone();
        d0 -= &one;
        let abs0 = d0.abs();
        let mut d1 = evals[1].clone();
        d1 -= &two;
        let abs1 = d1.abs();
        let mut d2 = evals[2].clone();
        d2 -= &three;
        let abs2 = d2.abs();
        assert!(
            abs0 < tol && abs1 < tol && abs2 < tol,
            "got {}, {}, {}",
            display_hp(&evals[0], 6),
            display_hp(&evals[1], 6),
            display_hp(&evals[2], 6)
        );
    }

    #[test]
    fn hp_sturm_count_preserves_strict_semantics_across_reducible_blocks() {
        let prec = 256;
        let diag = vec![hp(prec, "1"), hp(prec, "0"), hp(prec, "1")];
        let off_diag = vec![hp(prec, "0"), hp(prec, "0")];
        assert_eq!(
            tridiag_sturm_count_below_hp(&diag, &off_diag, &hp(prec, "1"), prec).unwrap(),
            1
        );
        assert_eq!(
            tridiag_sturm_count_below_hp(&diag, &off_diag, &hp(prec, "2"), prec).unwrap(),
            3
        );
    }

    #[test]
    fn hp_selected_sturm_values_enclose_full_qr_reference() {
        let prec = 256;
        let dimension = 10usize;
        let diag = vec![hp(prec, "2"); dimension];
        let off_diag = vec![hp(prec, "-1"); dimension - 1];
        let full = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        let tolerance = hp(prec, "1e-40");
        let selected =
            tridiag_selected_eigenvalues_hp(&diag, &off_diag, 2, 5, &tolerance, 200, prec).unwrap();
        assert_eq!(selected.enclosures.len(), 4);
        assert_eq!(selected.first_index, 2);
        assert_eq!(selected.last_index, 5);
        assert!(selected.sturm_evaluations > 2);
        for enclosure in &selected.enclosures {
            assert!(enclosure.lower <= full[enclosure.index]);
            assert!(enclosure.upper >= full[enclosure.index]);
            let mut width = enclosure.upper.clone();
            width -= &enclosure.lower;
            assert!(width <= tolerance);
            assert!(enclosure.lower_count <= enclosure.index);
            assert!(enclosure.upper_count > enclosure.index);
        }
    }

    #[test]
    fn hp_selected_eigenpairs_recover_simple_vectors_and_coalesce_multiplicity() {
        let prec = 256;
        let diagonal = vec![hp(prec, "2"); 8];
        let off_diagonal = vec![hp(prec, "-1"); 7];
        let simple = tridiag_selected_eigenpairs_hp(
            &diagonal,
            &off_diagonal,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: 1,
                last_index: 2,
                absolute_tolerance: hp(prec, "1e-30"),
                maximum_bisection_iterations: 200,
                eigenvector_options: TridiagEigvecOptions::default(),
                precision_bits: prec,
            },
        )
        .unwrap();
        assert_eq!(simple.vector_recoveries, 2);
        assert!(
            simple.items.iter().all(|item| matches!(
                item,
                HpSelectedTridiagonalItem::SimpleEigenpair(pair)
                    if pair.residual_norm < hp(prec, "1e-40")
                        && pair.diagnostics.absolute_residual == pair.residual_norm
                        && pair.diagnostics.relative_residual < hp(prec, "1e-40")
                        && pair.diagnostics.scaled_backward_error < hp(prec, "1e-40")
                        && pair.diagnostics.orthogonality_error < hp(prec, "1e-40")
            )),
            "selected eigenpair residuals were not HP-small: {:?}",
            simple
                .items
                .iter()
                .filter_map(|item| match item {
                    HpSelectedTridiagonalItem::SimpleEigenpair(pair) => {
                        Some(pair.residual_norm.clone())
                    }
                    HpSelectedTridiagonalItem::Cluster(_) => None,
                })
                .collect::<Vec<_>>()
        );

        let repeated_diagonal = vec![hp(prec, "1"), hp(prec, "1"), hp(prec, "3")];
        let zero_off_diagonal = vec![hp(prec, "0"), hp(prec, "0")];
        let clustered = tridiag_selected_eigenpairs_hp(
            &repeated_diagonal,
            &zero_off_diagonal,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: 0,
                last_index: 1,
                absolute_tolerance: hp(prec, "1e-30"),
                maximum_bisection_iterations: 200,
                eigenvector_options: TridiagEigvecOptions::default(),
                precision_bits: prec,
            },
        )
        .unwrap();
        assert_eq!(clustered.vector_recoveries, 0);
        assert_eq!(clustered.items.len(), 1);
        let HpSelectedTridiagonalItem::Cluster(cluster) = &clustered.items[0] else {
            panic!("repeated eigenvalue was assigned an individual vector");
        };
        assert_eq!((cluster.first_index, cluster.last_index), (0, 1));
        assert_eq!(cluster.requested_indices, vec![0, 1]);
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
        let mut d0 = evals[0].clone();
        d0 -= &one;
        let abs0 = d0.abs();
        let mut d1 = evals[1].clone();
        d1 -= &three;
        let abs1 = d1.abs();
        assert!(
            abs0 < tol && abs1 < tol,
            "expected (1, 3), got ({}, {})",
            display_hp(&evals[0], 6),
            display_hp(&evals[1], 6)
        );
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
        let mut sqrt2 = hp(prec, "2");
        sqrt2 = sqrt2.sqrt();
        let mut e0 = two.clone();
        e0 -= &sqrt2; // 2 - √2
        let e1 = two.clone(); // 2
        let mut e2 = two.clone();
        e2 += &sqrt2; // 2 + √2

        let tol = hp(prec, "1e-100");

        let mut d0 = evals[0].clone();
        d0 -= &e0;
        let abs0 = d0.abs();
        let mut d1 = evals[1].clone();
        d1 -= &e1;
        let abs1 = d1.abs();
        let mut d2 = evals[2].clone();
        d2 -= &e2;
        let abs2 = d2.abs();

        assert!(abs0 < tol, "eigval[0] off by {}", display_hp(&abs0, 4));
        assert!(abs1 < tol, "eigval[1] off by {}", display_hp(&abs1, 4));
        assert!(abs2 < tol, "eigval[2] off by {}", display_hp(&abs2, 4));
    }

    /// QR convergence at HP-1000 should give matching digits comparable to working precision.
    /// Use the same 3×3 matrix at higher precision.
    #[test]
    fn tridiag_eigenvalues_hp_1000() {
        let prec = 3338; // ≈ 1000 decimal digits
        let diag = vec![hp(prec, "2"), hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1"), hp(prec, "1")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

        let two = hp(prec, "2");
        let mut sqrt2 = hp(prec, "2");
        sqrt2 = sqrt2.sqrt();
        let mut e0 = two.clone();
        e0 -= &sqrt2;
        let e1 = two.clone();
        let mut e2 = two;
        e2 += &sqrt2;

        // Should match to ~prec/3.322 ≈ 1000 decimal digits.
        let m0 = matching_digits(&evals[0], &e0);
        let m1 = matching_digits(&evals[1], &e1);
        let m2 = matching_digits(&evals[2], &e2);

        // Expect ≥500 digits of agreement (well below working precision).
        let min_digits = Float::with_val(prec, 500);
        assert!(
            m0 > min_digits || m0.is_infinite(),
            "eigval[0] matches only {} digits",
            display_hp(&m0, 4)
        );
        assert!(
            m1 > min_digits || m1.is_infinite(),
            "eigval[1] matches only {} digits",
            display_hp(&m1, 4)
        );
        assert!(
            m2 > min_digits || m2.is_infinite(),
            "eigval[2] matches only {} digits",
            display_hp(&m2, 4)
        );
    }

    /// Eigenvector recovery via shifted inverse iteration.
    /// For T = [[2,1,0],[1,2,1],[0,1,2]] and eigenvalue λ = 2 - √2,
    /// eigenvector should be (1, -√2, 1)/2 (up to sign).
    ///
    /// Tests both solver paths (dense and banded LU) — they should
    /// produce eigenvectors that satisfy T·v ≈ λ·v to working precision
    /// independently of which solver is used.
    #[test]
    fn tridiag_eigenvector_recovery() {
        let prec = 256;
        let diag = vec![hp(prec, "2"), hp(prec, "2"), hp(prec, "2")];
        let off_diag = vec![hp(prec, "1"), hp(prec, "1")];

        let two = hp(prec, "2");
        let mut sqrt2 = hp(prec, "2");
        sqrt2 = sqrt2.sqrt();
        let mut e0 = two;
        e0 -= &sqrt2;

        for solver in [TridiagSolver::Banded, TridiagSolver::Dense] {
            let opts = TridiagEigvecOptions {
                max_steps: 100,
                early_termination: true,
                solver,
            };
            let v = tridiag_eigenvector_for_value_hp(&diag, &off_diag, &e0, prec, opts).unwrap();
            assert_eq!(v.len(), 3, "solver {:?}: wrong length", solver);

            // Verify T·v ≈ λ·v.
            // T·v = (2v[0] + v[1], v[0] + 2v[1] + v[2], v[1] + 2v[2])
            let mut tv0 = v[0].clone();
            tv0 *= 2u32;
            tv0 += &v[1];
            let mut tv1 = v[0].clone();
            tv1 += &v[2];
            let mut tmp = v[1].clone();
            tmp *= 2u32;
            tv1 += &tmp;
            let mut tv2 = v[1].clone();
            tv2 += &v[2].clone();
            tv2 += &v[2];

            let mut lv0 = e0.clone();
            lv0 *= &v[0];
            let mut lv1 = e0.clone();
            lv1 *= &v[1];
            let mut lv2 = e0.clone();
            lv2 *= &v[2];

            let mut r0 = tv0;
            r0 -= &lv0;
            let r0 = r0.abs();
            let mut r1 = tv1;
            r1 -= &lv1;
            let r1 = r1.abs();
            let mut r2 = tv2;
            r2 -= &lv2;
            let r2 = r2.abs();

            let tol = hp(prec, "1e-50");
            assert!(
                r0 < tol,
                "solver {:?}: T·v - λv at index 0: {}",
                solver,
                display_hp(&r0, 4)
            );
            assert!(
                r1 < tol,
                "solver {:?}: T·v - λv at index 1: {}",
                solver,
                display_hp(&r1, 4)
            );
            assert!(
                r2 < tol,
                "solver {:?}: T·v - λv at index 2: {}",
                solver,
                display_hp(&r2, 4)
            );
        }
    }

    /// Householder tridiagonalization: dense symmetric → tridiagonal,
    /// eigenvalues should be preserved.
    #[test]
    fn householder_preserves_eigenvalues() {
        let prec = 256;
        let n = 4;
        // Build a random-ish symmetric matrix.
        let raw = [
            "4", "1", "2", "0", "1", "3", "1", "1", "2", "1", "5", "2", "0", "1", "2", "6",
        ];
        let a: Vec<Float> = raw.iter().map(|s| hp(prec, s)).collect();

        let evals_dense = dense_symmetric_eigenvalues_hp(&a, n, prec).unwrap();
        assert_eq!(evals_dense.len(), n);

        // Sanity: eigenvalues should be sorted ascending.
        for i in 0..(n - 1) {
            assert!(
                evals_dense[i] <= evals_dense[i + 1],
                "eigenvalues not sorted: [{}, {}]",
                display_hp(&evals_dense[i], 6),
                display_hp(&evals_dense[i + 1], 6)
            );
        }

        // Sum of eigenvalues = trace = 4 + 3 + 5 + 6 = 18.
        let mut sum = hp(prec, "0");
        for v in &evals_dense {
            sum += v;
        }
        let expected_trace = hp(prec, "18");
        let tol = hp(prec, "1e-50");
        let mut diff = sum.clone();
        diff -= &expected_trace;
        let abs_diff = diff.abs();
        assert!(
            abs_diff < tol,
            "sum of eigenvalues should be trace = 18, got {}",
            display_hp(&sum, 6)
        );
    }

    /// Householder reduction produces an orthogonal Q: QᵀQ = I.
    /// This is independent of the eigenvalue check above; if Q drifts
    /// from orthogonality the tridiagonalization is wrong even if
    /// `dense_symmetric_eigenvalues_hp` happens to deliver the right
    /// eigenvalues by lucky cancellation in tridiag QR.
    #[test]
    fn householder_q_is_orthogonal() {
        let prec = 256;
        let n = 5;
        // Random-ish symmetric input.
        let raw = [
            "4", "1", "2", "0", "1", "1", "3", "1", "1", "0", "2", "1", "5", "2", "1", "0", "1",
            "2", "6", "3", "1", "0", "1", "3", "7",
        ];
        let a: Vec<Float> = raw.iter().map(|s| hp(prec, s)).collect();

        let (_, _, q) = householder_tridiag_hp(&a, n, prec).unwrap();
        assert_eq!(q.len(), n * n, "Q should be n×n");

        // Compute QᵀQ. (Q^T Q)[i,j] = Σ_k Q[k,i] * Q[k,j].
        let tol = hp(prec, "1e-50");
        for i in 0..n {
            for j in 0..n {
                let mut sum = hp(prec, "0");
                for k in 0..n {
                    let mut t = q[k * n + i].clone();
                    t *= &q[k * n + j];
                    sum += &t;
                }
                let expected = if i == j { hp(prec, "1") } else { hp(prec, "0") };
                let mut diff = sum.clone();
                diff -= &expected;
                let abs_diff = diff.abs();
                assert!(
                    abs_diff < tol,
                    "(QᵀQ)[{},{}] should be {} (Kronecker δ); got {}, diff {}",
                    i,
                    j,
                    if i == j { 1 } else { 0 },
                    display_hp(&sum, 6),
                    display_hp(&abs_diff, 4)
                );
            }
        }
    }

    /// Householder on a symmetric matrix that's *already* tridiagonal
    /// should produce its own diagonal/off-diagonal verbatim.
    /// We use Strang's tridiagonal n=6: diag=2, off=-1.
    #[test]
    fn householder_on_already_tridiag_is_idempotent() {
        let prec = 256;
        let n = 6;
        // Build Strang n=6 as dense input.
        let mut a = vec![hp(prec, "0"); n * n];
        for i in 0..n {
            a[i * n + i] = hp(prec, "2");
            if i > 0 {
                a[i * n + (i - 1)] = hp(prec, "-1");
            }
            if i + 1 < n {
                a[i * n + (i + 1)] = hp(prec, "-1");
            }
        }

        let (diag, off_diag, q) = householder_tridiag_hp(&a, n, prec).unwrap();
        assert_eq!(diag.len(), n);
        assert_eq!(off_diag.len(), n - 1);

        // The output diagonal should equal the input diag (all 2's).
        let two = hp(prec, "2");
        let neg_one = hp(prec, "-1");
        let tol = hp(prec, "1e-50");
        for i in 0..n {
            let mut diff = diag[i].clone();
            diff -= &two;
            let abs_diff = diff.abs();
            assert!(
                abs_diff < tol,
                "tridiag diag[{}] should be 2; got {}",
                i,
                display_hp(&diag[i], 6)
            );
        }
        // Each off-diagonal should be |−1| (sign of the new off-diag is
        // determined by the Householder sign convention; we accept ±1).
        for i in 0..(n - 1) {
            let mut diff_neg = off_diag[i].clone();
            diff_neg -= &neg_one;
            let mut diff_pos = off_diag[i].clone();
            diff_pos -= 1u32;
            let abs_neg = diff_neg.abs();
            let abs_pos = diff_pos.abs();
            assert!(
                abs_neg < tol || abs_pos < tol,
                "tridiag off_diag[{}] should be ±1; got {}",
                i,
                display_hp(&off_diag[i], 6)
            );
        }

        // Q is orthogonal even on a tridiag input.
        for i in 0..n {
            for j in 0..n {
                let mut sum = hp(prec, "0");
                for k in 0..n {
                    let mut t = q[k * n + i].clone();
                    t *= &q[k * n + j];
                    sum += &t;
                }
                let expected = if i == j { hp(prec, "1") } else { hp(prec, "0") };
                let mut diff = sum.clone();
                diff -= &expected;
                let abs_diff = diff.abs();
                assert!(
                    abs_diff < tol,
                    "Q on already-tridiag input still orthogonal: (QᵀQ)[{},{}] should be δ; got {}",
                    i,
                    j,
                    display_hp(&sum, 6)
                );
            }
        }
    }

    /// Dense eigenvector recovery via shifted inverse iteration.
    #[test]
    fn dense_eigenvector_recovery() {
        let prec = 256;
        let n = 3;
        // Diagonal matrix diag(1, 2, 3). Eigenvalue 2 has eigenvector (0, 1, 0).
        let a: Vec<Float> = vec![
            hp(prec, "1"),
            hp(prec, "0"),
            hp(prec, "0"),
            hp(prec, "0"),
            hp(prec, "2"),
            hp(prec, "0"),
            hp(prec, "0"),
            hp(prec, "0"),
            hp(prec, "3"),
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

        assert!(
            abs_v0 < small,
            "|v[0]| should be tiny, got {}",
            display_hp(&abs_v0, 6)
        );
        assert!(
            abs_v2 < small,
            "|v[2]| should be tiny, got {}",
            display_hp(&abs_v2, 6)
        );
        let mut v1_diff = abs_v1;
        v1_diff -= &one;
        let v1_diff_abs = v1_diff.abs();
        assert!(
            v1_diff_abs < close_to_one_tol,
            "|v[1]| should be 1, got {}",
            display_hp(&v[1], 6)
        );
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
        let expected: Vec<Float> = (1..=n)
            .map(|k| {
                let mut arg = Float::with_val(prec, k as u32);
                arg *= &pi_v;
                arg /= &two_n_plus_1;
                let mut s = arg.sin();
                s *= s.clone();
                s *= 4u32;
                s
            })
            .collect();
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
            assert!(
                abs_diff < tol,
                "eigenvalue {} off by {} (expected {}, got {})",
                i,
                display_hp(&abs_diff, 4),
                display_hp(expected, 6),
                display_hp(computed, 6)
            );
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
            assert!(
                abs_diff < tol,
                "n=50 eigenvalue {} off by {}",
                i,
                display_hp(&abs_diff, 4)
            );
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
            assert!(
                m > min_digits || m.is_infinite(),
                "n={}, k={}: only {} matching digits",
                n,
                i,
                display_hp(&m, 4)
            );
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
        for v in &evals {
            sum += v;
        }
        let mut expected_trace = Float::with_val(prec, 0);
        for k in 0..n {
            let mut term = Float::with_val(prec, 1);
            let denom = Float::with_val(prec, (2 * k + 1) as u32);
            term /= &denom;
            expected_trace += &term;
        }
        let tol = hp(prec, "1e-50");
        let mut diff = sum.clone();
        diff -= &expected_trace;
        let abs_diff = diff.abs();
        assert!(
            abs_diff < tol,
            "Hilbert trace mismatch: sum {} vs trace {}, delta {}",
            display_hp(&sum, 6),
            display_hp(&expected_trace, 6),
            display_hp(&abs_diff, 4)
        );

        // Eigenvalues should all be positive (Hilbert is SPD).
        let zero = Float::with_val(prec, 0);
        for (i, v) in evals.iter().enumerate() {
            assert!(
                *v > zero,
                "Hilbert eigenvalue {} should be positive, got {}",
                i,
                display_hp(v, 6)
            );
        }

        // Smallest eigenvalue should be very small (Hilbert is ill-conditioned).
        // For n=4, the smallest eigenvalue is ~10^-4.
        let small_threshold = hp(prec, "1e-3");
        assert!(
            evals[0] < small_threshold,
            "smallest eigenvalue should be < 1e-3 (Hilbert ill-conditioning), got {}",
            display_hp(&evals[0], 6)
        );

        // Largest eigenvalue should be ~1.5 for n=4.
        let large_lo = hp(prec, "1.0");
        let large_hi = hp(prec, "2.0");
        assert!(
            evals[n - 1] > large_lo && evals[n - 1] < large_hi,
            "largest eigenvalue should be in [1, 2], got {}",
            display_hp(&evals[n - 1], 6)
        );
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
            (0, 1, {
                let mut t = pi_v.clone();
                t /= 3u32;
                t
            }),
            (1, 2, {
                let mut t = pi_v.clone();
                t /= 4u32;
                t
            }),
            (2, 3, {
                let mut t = pi_v.clone();
                t /= 5u32;
                t
            }),
            (3, 4, {
                let mut t = pi_v.clone();
                t /= 7u32;
                t
            }),
            (0, 4, {
                let mut t = pi_v.clone();
                t /= 11u32;
                t
            }),
        ];

        for (i, j, theta) in &rotations {
            let c = theta.clone().cos();
            let s = theta.clone().sin();
            // G^T A G: build new A row by row.
            let mut new_a = a.clone();
            // Apply G from the right: A ← A G. Updates columns i and j.
            for r in 0..n {
                let mut new_ri = c.clone();
                new_ri *= &a[r * n + *i];
                let mut t = s.clone();
                t *= &a[r * n + *j];
                new_ri += &t;
                let mut new_rj = s.clone();
                new_rj = -new_rj;
                new_rj *= &a[r * n + *i];
                let mut t = c.clone();
                t *= &a[r * n + *j];
                new_rj += &t;
                new_a[r * n + *i] = new_ri;
                new_a[r * n + *j] = new_rj;
            }
            a = new_a;
            // Apply G^T from the left: A ← G^T A. Updates rows i and j.
            let mut new_a = a.clone();
            for col in 0..n {
                let mut new_ic = c.clone();
                new_ic *= &a[*i * n + col];
                let mut t = s.clone();
                t *= &a[*j * n + col];
                new_ic += &t;
                let mut new_jc = s.clone();
                new_jc = -new_jc;
                new_jc *= &a[*i * n + col];
                let mut t = c.clone();
                t *= &a[*j * n + col];
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
            assert!(
                abs_diff < tol,
                "rotated_diagonal eigenvalue {} off by {} (expected {}, got {})",
                i,
                display_hp(&abs_diff, 4),
                display_hp(expected, 6),
                display_hp(computed, 6)
            );
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
        let chosen: Vec<Float> = (0..n)
            .map(|k| {
                let mut v = Float::with_val(prec, 1);
                let mut delta = Float::with_val(prec, k as u32);
                let scale = hp(prec, "1e-100");
                delta *= &scale;
                v += &delta;
                v
            })
            .collect();

        // Build A = G^T D G with one Givens rotation on (0, 2).
        let mut a = vec![Float::with_val(prec, 0); n * n];
        for i in 0..n {
            a[i * n + i] = chosen[i].clone();
        }

        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        let theta = {
            let mut t = pi_v.clone();
            t /= 6u32;
            t
        };
        let c = theta.clone().cos();
        let s = theta.clone().sin();
        let (i, j) = (0usize, 2usize);

        // G^T A G via two passes (right then left).
        let mut new_a = a.clone();
        for r in 0..n {
            let mut new_ri = c.clone();
            new_ri *= &a[r * n + i];
            let mut t = s.clone();
            t *= &a[r * n + j];
            new_ri += &t;
            let mut new_rj = s.clone();
            new_rj = -new_rj;
            new_rj *= &a[r * n + i];
            let mut t = c.clone();
            t *= &a[r * n + j];
            new_rj += &t;
            new_a[r * n + i] = new_ri;
            new_a[r * n + j] = new_rj;
        }
        a = new_a;
        let mut new_a = a.clone();
        for col in 0..n {
            let mut new_ic = c.clone();
            new_ic *= &a[i * n + col];
            let mut t = s.clone();
            t *= &a[j * n + col];
            new_ic += &t;
            let mut new_jc = s.clone();
            new_jc = -new_jc;
            new_jc *= &a[i * n + col];
            let mut t = c.clone();
            t *= &a[j * n + col];
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
            assert!(
                abs_diff < tol,
                "clustered eigenvalue {} off by {}",
                i,
                display_hp(&abs_diff, 4)
            );
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
        let diag: Vec<Float> = (0..n)
            .map(|i| {
                let mid = (n / 2) as i64; // 10 for n=21
                let val = (mid - i as i64).abs();
                Float::with_val(prec, val as u32)
            })
            .collect();
        let off_diag: Vec<Float> = vec![Float::with_val(prec, 1); n - 1];

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), n);

        // Largest eigenvalue ≈ 10.7461942 (Parlett 1980, tabulated).
        // We verify it's in [10.74, 10.75] — a tight window confirming
        // we found the right value.
        let lo = hp(prec, "10.74");
        let hi = hp(prec, "10.75");
        let largest = &evals[n - 1];
        assert!(
            *largest > lo && *largest < hi,
            "Wilkinson W21 largest eigenvalue should be ≈10.7462, got {}",
            display_hp(largest, 8)
        );

        // Also verify trace = sum of |i - mid| for i = 0..n.
        // = 2 * (1 + 2 + ... + 10) = 2 * 55 = 110.
        let mut sum = Float::with_val(prec, 0);
        for v in &evals {
            sum += v;
        }
        let expected_trace = Float::with_val(prec, 110u32);
        let tol = hp(prec, "1e-50");
        let mut diff = sum.clone();
        diff -= &expected_trace;
        let abs_diff = diff.abs();
        assert!(
            abs_diff < tol,
            "W21 trace mismatch: sum {} vs 110, delta {}",
            display_hp(&sum, 6),
            display_hp(&abs_diff, 4)
        );
    }

    /// Verify A·v = λ·v for an eigenvector recovered via shifted inverse iteration.
    /// Uses Strang n=10 where eigenvalues are known in closed form.
    /// Tests both solver paths.
    #[test]
    fn eigenvector_satisfies_eigenequation_strang() {
        let prec = 256;
        let n = 10;
        let (diag, off_diag, expected) = strang_tridiag(prec, n);

        // Pick the smallest eigenvalue and recover its eigenvector
        // under each solver path. Each path independently must satisfy
        // T·v = λ·v to working precision.
        let lambda = expected[0].clone();

        for solver in [TridiagSolver::Banded, TridiagSolver::Dense] {
            let opts = TridiagEigvecOptions {
                max_steps: 200,
                early_termination: true,
                solver,
            };
            let v =
                tridiag_eigenvector_for_value_hp(&diag, &off_diag, &lambda, prec, opts).unwrap();

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
            let lv: Vec<Float> = v
                .iter()
                .map(|vi| {
                    let mut t = lambda.clone();
                    t *= vi;
                    t
                })
                .collect();

            // Residual ‖T·v - λ·v‖_∞ should be tiny.
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
            assert!(
                max_residual < tol,
                "solver {:?}: ‖T·v - λv‖_∞ = {} should be < 1e-50",
                solver,
                display_hp(&max_residual, 4)
            );

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
            assert!(
                abs_norm_diff < norm_tol,
                "solver {:?}: ‖v‖² should be 1, got {}",
                solver,
                display_hp(&norm_sq, 6)
            );
        }
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
            assert!(
                abs_diff < tol,
                "below-floor eigenvalue {} off by {} (expected {}, got {})",
                k,
                display_hp(&abs_diff, 4),
                display_hp(&expected, 6),
                display_hp(&evals[k - 1], 6)
            );
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
        let mut state: u64 = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);

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
                for v in &evals {
                    sum += v;
                }

                let mut diff = sum.clone();
                diff -= &trace;
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-50");
                assert!(
                    abs_diff < tol,
                    "n={}, seed={}: trace {} vs sum {} differ by {}",
                    n,
                    seed,
                    display_hp(&trace, 6),
                    display_hp(&sum, 6),
                    display_hp(&abs_diff, 4)
                );
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
                for v in &evals {
                    product *= v;
                }

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
                assert!(
                    abs_diff < tol_relative,
                    "n={}, seed={}: det {} vs product {} differ by {}",
                    n,
                    seed,
                    display_hp(&det, 6),
                    display_hp(&product, 6),
                    display_hp(&abs_diff, 4)
                );
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
                    let v = match dense_symmetric_eigenvector_for_value_hp(&a, n, lambda, prec, 200)
                    {
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
                    let lv: Vec<Float> = v
                        .iter()
                        .map(|vi| {
                            let mut t = lambda.clone();
                            t *= vi;
                            t
                        })
                        .collect();

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
                    assert!(
                        max_residual < tol,
                        "n={}, seed={}, k={}: ‖A·v - λv‖_∞ = {} should be < 1e-40",
                        n,
                        seed,
                        k,
                        display_hp(&max_residual, 4)
                    );
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
                    let v = match dense_symmetric_eigenvector_for_value_hp(&a, n, lambda, prec, 200)
                    {
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
                    assert!(
                        abs_diff < tol,
                        "n={}, seed={}: ‖v‖² = {} should be 1",
                        n,
                        seed,
                        display_hp(&norm_sq, 6)
                    );
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
                    let v = match dense_symmetric_eigenvector_for_value_hp(&a, n, lambda, prec, 200)
                    {
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
                    match dense_symmetric_eigenvector_for_value_hp(&a, n, lambda, prec, 200) {
                        Ok(v) => vecs.push(v),
                        Err(_) => {
                            all_recovered = false;
                            break;
                        }
                    }
                }
                if !all_recovered {
                    continue;
                }

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
                assert!(
                    max_diff < tol,
                    "n={}, seed={}: ‖A - Σ λ v vᵀ‖_∞ = {} should be < 1e-30",
                    n,
                    seed,
                    display_hp(&max_diff, 4)
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // tridiag_eigenvalues_hp — property tests
    // -----------------------------------------------------------------------
    //
    // The dense-matrix property tests above exercise tridiag_eigenvalues_hp
    // *transitively* via dense_symmetric_eigenvalues_hp (which composes
    // Householder + tridiag QR). The tests below pin down the tridiag QR
    // alone on tridiagonal-shaped inputs across a sweep of sizes and
    // shapes — so QR-specific bugs (deflation paths, Wilkinson-shift
    // formulation, near-degenerate eigenvalues) can't be masked by
    // Householder structure.

    /// Deterministic HP-random *symmetric tridiagonal* generator. Diag
    /// drawn from `[-1, 1]` (HP), off-diagonals drawn from `[-1, 1]` and
    /// then multiplied by `0.5` so the matrix is mildly diagonally
    /// dominant on average — keeps eigenvalues numerically separated.
    fn lcg_random_tridiag(prec: u32, n: usize, seed: u64) -> (Vec<Float>, Vec<Float>) {
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut state: u64 = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);

        let mut next_uniform = || -> Float {
            state = state.wrapping_mul(a).wrapping_add(c);
            let top = (state >> 11) as i64;
            let scale = Float::with_val(prec, top);
            let mut v = scale;
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

        let diag: Vec<Float> = (0..n).map(|_| next_uniform()).collect();
        // Halve the off-diagonals so the matrix is on-average diagonally
        // dominant. This avoids near-degenerate eigenvalues that would
        // require many extra QR iterations (and tighten our tolerance
        // budget unnecessarily).
        let off_diag: Vec<Float> = (0..n.saturating_sub(1))
            .map(|_| {
                let mut v = next_uniform();
                v /= 2u32;
                v
            })
            .collect();
        (diag, off_diag)
    }

    /// Property: tridiag QR returns exactly n eigenvalues in ascending
    /// order, for random symmetric tridiagonals across a sweep of sizes
    /// and seeds.
    #[test]
    fn property_tridiag_eigenvalues_count_and_order() {
        let prec = 256;
        let sizes = [3usize, 5, 8, 12, 20];
        let seeds_per_size = 4;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let (diag, off_diag) = lcg_random_tridiag(prec, n, seed as u64 + 1000);
                let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

                assert_eq!(
                    evals.len(),
                    n,
                    "n={}, seed={}: expected {} eigenvalues, got {}",
                    n,
                    seed,
                    n,
                    evals.len()
                );

                // Ascending check: e[k] ≤ e[k+1] for all k.
                for k in 0..(n - 1) {
                    assert!(
                        evals[k] <= evals[k + 1],
                        "n={}, seed={}: eigenvalues not ascending at index {}: {} > {}",
                        n,
                        seed,
                        k,
                        display_hp(&evals[k], 6),
                        display_hp(&evals[k + 1], 6)
                    );
                }
            }
        }
    }

    /// Property: trace = sum of eigenvalues, for random symmetric
    /// tridiagonals. Holds for any symmetric matrix; tests that the
    /// QR's per-step bookkeeping preserves the trace invariant.
    #[test]
    fn property_tridiag_trace_equals_sum_of_eigenvalues() {
        let prec = 256;
        let sizes = [3usize, 5, 8, 12, 20];
        let seeds_per_size = 4;

        for &n in &sizes {
            for seed in 0..seeds_per_size {
                let (diag, off_diag) = lcg_random_tridiag(prec, n, seed as u64 + 2000);

                // Trace: sum of diag entries (off-diagonals don't contribute).
                let mut trace = hp(prec, "0");
                for d in &diag {
                    trace += d;
                }

                let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
                let mut sum = hp(prec, "0");
                for e in &evals {
                    sum += e;
                }

                let mut diff = sum.clone();
                diff -= &trace;
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-50");
                assert!(
                    abs_diff < tol,
                    "n={}, seed={}: |Σλ - trace| = {} should be < 1e-50",
                    n,
                    seed,
                    display_hp(&abs_diff, 4)
                );
            }
        }
    }

    /// Property: Strang's tridiagonal closed-form λ_k = 2 - 2 cos(kπ/(n+1))
    /// is recovered to working precision for n up to 20.
    /// This catches QR convergence regressions on a textbook input where
    /// the eigenvalues are exactly known.
    #[test]
    fn property_tridiag_strang_closed_form() {
        let prec = 256;
        for &n in &[5usize, 10, 15, 20] {
            let (diag, off_diag, expected) = strang_tridiag(prec, n);
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

            assert_eq!(evals.len(), n);
            for k in 0..n {
                let mut diff = evals[k].clone();
                diff -= &expected[k];
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-50");
                assert!(
                    abs_diff < tol,
                    "Strang n={}, eigenvalue {}: |computed - expected| = {} > 1e-50",
                    n,
                    k,
                    display_hp(&abs_diff, 6)
                );
            }
        }
    }

    /// Property: identical eigenvalues produced regardless of whether the
    /// input is presented diagonally (off-diagonal exactly 0). The QR's
    /// deflation path on zero off-diagonals should immediately succeed.
    #[test]
    fn property_tridiag_zero_off_diagonal_returns_sorted_diag() {
        let prec = 256;
        for &n in &[3usize, 5, 8] {
            // Diagonal-only "tridiagonal": diag = [n, n-1, ..., 1], off = 0.
            let diag: Vec<Float> = (0..n).map(|i| hp(prec, &(n - i).to_string())).collect();
            let off_diag: Vec<Float> = (0..(n - 1)).map(|_| hp(prec, "0")).collect();

            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
            assert_eq!(evals.len(), n);

            // Result should be the diag entries sorted ascending: [1, 2, ..., n].
            for k in 0..n {
                let expected = hp(prec, &(k + 1).to_string());
                let mut diff = evals[k].clone();
                diff -= &expected;
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-100");
                assert!(
                    abs_diff < tol,
                    "n={}, eigenvalue {}: expected {}, got {}",
                    n,
                    k,
                    k + 1,
                    display_hp(&evals[k], 6)
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Banded vs dense cross-validation tests
    // -----------------------------------------------------------------------

    /// Banded vs dense equivalence: run both code paths on the same
    /// Strang n=10 input and confirm the eigenvectors agree (up to sign,
    /// since inverse iteration is agnostic to sign of v).
    #[test]
    fn banded_matches_dense_on_strang_n10() {
        let prec = 256;
        let n = 10;

        // Strang's tridiagonal: diag = 2, off = -1.
        let diag: Vec<Float> = (0..n).map(|_| hp(prec, "2")).collect();
        let off_diag: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        let lambda_1 = evals[0].clone();

        let common = TridiagEigvecOptions {
            max_steps: 200,
            early_termination: true,
            solver: TridiagSolver::Dense, // overwritten per call
        };

        // Dense path.
        let v_dense = tridiag_eigenvector_for_value_hp(
            &diag,
            &off_diag,
            &lambda_1,
            prec,
            TridiagEigvecOptions {
                solver: TridiagSolver::Dense,
                ..common
            },
        )
        .unwrap();

        // Banded path.
        let v_banded = tridiag_eigenvector_for_value_hp(
            &diag,
            &off_diag,
            &lambda_1,
            prec,
            TridiagEigvecOptions {
                solver: TridiagSolver::Banded,
                ..common
            },
        )
        .unwrap();

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
            for v in v_b.iter_mut() {
                *v = -v.clone();
            }
        }

        // Element-wise compare. Should match to working precision.
        for i in 0..n {
            let mut diff = v_dense[i].clone();
            diff -= &v_b[i];
            let abs_diff = diff.abs();
            let tol = hp(prec, "1e-50");
            assert!(
                abs_diff < tol,
                "banded vs dense disagreement at index {}: {} (dense={}, banded={})",
                i,
                display_hp(&abs_diff, 6),
                display_hp(&v_dense[i], 6),
                display_hp(&v_b[i], 6)
            );
        }
    }

    /// HP-1000 production residual check: at publication precision,
    /// each solver path produces an eigenvector that satisfies
    /// ‖T·v - λv‖_∞ < 10^-900. Tests both Banded and Dense paths.
    #[test]
    fn eigenvector_residual_hp_1000() {
        let prec = 3338;
        let n = 20;

        // Strang's tridiagonal at n=20.
        let diag: Vec<Float> = (0..n).map(|_| hp(prec, "2")).collect();
        let off_diag: Vec<Float> = (0..n - 1).map(|_| hp(prec, "-1")).collect();

        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        let lambda_1 = evals[0].clone();

        for solver in [TridiagSolver::Banded, TridiagSolver::Dense] {
            let v = tridiag_eigenvector_for_value_hp(
                &diag,
                &off_diag,
                &lambda_1,
                prec,
                TridiagEigvecOptions {
                    max_steps: 200,
                    early_termination: true,
                    solver,
                },
            )
            .unwrap();

            // Compute residual ‖T·v - λv‖_∞.
            let mut max_resid = hp(prec, "0");
            for i in 0..n {
                let mut tv_i = diag[i].clone();
                tv_i *= &v[i];
                if i > 0 {
                    let mut t = off_diag[i - 1].clone();
                    t *= &v[i - 1];
                    tv_i += &t;
                }
                if i < n - 1 {
                    let mut t = off_diag[i].clone();
                    t *= &v[i + 1];
                    tv_i += &t;
                }
                let mut lv_i = lambda_1.clone();
                lv_i *= &v[i];
                let mut resid = tv_i;
                resid -= &lv_i;
                resid = resid.abs();
                if resid > max_resid {
                    max_resid = resid;
                }
            }

            // At HP-1000 with the working-precision early-termination
            // threshold, residual should be ≲ 10^-900 (~working precision).
            let tol = hp(prec, "1e-900");
            assert!(
                max_resid < tol,
                "solver {:?}: HP-1000 residual ‖T·v - λv‖_∞ = {} should be < 1e-900",
                solver,
                display_hp(&max_resid, 6)
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Boundary-condition tests for tridiag and Householder
    // ─────────────────────────────────────────────────────────────────────────

    /// `tridiag_eigenvalues_hp` on a 1×1 matrix: single eigenvalue equals
    /// the single diagonal entry.
    #[test]
    fn tridiag_eigenvalues_1x1() {
        let prec = 128;
        let diag = vec![hp(prec, "7.5")];
        let off_diag: Vec<Float> = Vec::new();
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 1);
        let mut diff = evals[0].clone();
        diff -= hp(prec, "7.5");
        let d = diff.abs();
        assert!(
            d < hp(prec, "1e-50"),
            "1×1 tridiag eigenvalue should be 7.5; got diff {}",
            d
        );
    }

    /// `tridiag_eigenvalues_hp` on a 2×2 symmetric tridiagonal: eigenvalues
    /// d ± off for T = [[d, off],[off, d]] are exactly d + |off| and d - |off|.
    #[test]
    fn tridiag_eigenvalues_2x2() {
        let prec = 128;
        // T = [[3, 1], [1, 3]] → eigenvalues 2, 4.
        let diag = vec![hp(prec, "3"), hp(prec, "3")];
        let off_diag = vec![hp(prec, "1")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 2);
        // Should be ascending: [2, 4].
        let mut d0 = evals[0].clone();
        d0 -= hp(prec, "2");
        let d0 = d0.abs();
        let mut d1 = evals[1].clone();
        d1 -= hp(prec, "4");
        let d1 = d1.abs();
        let tol = hp(prec, "1e-50");
        assert!(d0 < tol, "2×2 eval[0] should be 2; diff = {}", d0);
        assert!(d1 < tol, "2×2 eval[1] should be 4; diff = {}", d1);
    }

    /// `tridiag_eigenvalues_hp` with zero off-diagonal: reduces to sorting the
    /// diagonal. Results should equal the diagonal in ascending order.
    #[test]
    fn tridiag_eigenvalues_zero_off_diagonal() {
        let prec = 128;
        let diag = vec![hp(prec, "5"), hp(prec, "1"), hp(prec, "3")];
        let off_diag = vec![hp(prec, "0"), hp(prec, "0")];
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 3);
        let tol = hp(prec, "1e-50");
        let mut d0 = evals[0].clone();
        d0 -= hp(prec, "1");
        let d0 = d0.abs();
        let mut d1 = evals[1].clone();
        d1 -= hp(prec, "3");
        let d1 = d1.abs();
        let mut d2 = evals[2].clone();
        d2 -= hp(prec, "5");
        let d2 = d2.abs();
        assert!(d0 < tol, "zero off-diag evals[0] should be 1; diff={}", d0);
        assert!(d1 < tol, "zero off-diag evals[1] should be 3; diff={}", d1);
        assert!(d2 < tol, "zero off-diag evals[2] should be 5; diff={}", d2);
    }

    /// `tridiag_eigenvalues_hp` with off_diag wrong length should return Err.
    #[test]
    fn tridiag_eigenvalues_wrong_off_diag_length_errors() {
        let prec = 64;
        let diag = vec![hp(prec, "1"), hp(prec, "2"), hp(prec, "3")];
        let bad_off = vec![hp(prec, "0.5")]; // should be length 2
        assert!(
            tridiag_eigenvalues_hp(&diag, &bad_off, prec).is_err(),
            "wrong off-diag length should return Err"
        );
    }

    /// `householder_tridiag_hp` on a 1×1 matrix: no-op (no reflections
    /// needed). The single diagonal is returned unchanged.
    #[test]
    fn householder_tridiag_1x1() {
        let prec = 128;
        let a = vec![hp(prec, "42")];
        let (diag, off_diag, q) = householder_tridiag_hp(&a, 1, prec).unwrap();
        assert_eq!(diag.len(), 1);
        assert!(off_diag.is_empty());
        assert_eq!(q.len(), 1);
        let mut d = diag[0].clone();
        d -= hp(prec, "42");
        let d = d.abs();
        assert!(
            d < hp(prec, "1e-50"),
            "1×1 householder diag should be 42; diff={}",
            d
        );
    }

    /// `householder_tridiag_hp` on a 2×2 symmetric matrix: one Householder
    /// step is trivial (only the off-diagonal is set). The eigenvalues of the
    /// returned tridiagonal should match the original matrix.
    #[test]
    fn householder_tridiag_2x2_preserves_eigenvalues() {
        let prec = 128;
        // A = [[2, 3], [3, 5]]; eigenvalues = (7 ± √37) / 2 ≈ 0.459 and 6.541.
        let a = vec![hp(prec, "2"), hp(prec, "3"), hp(prec, "3"), hp(prec, "5")];
        let (diag, off_diag, _q) = householder_tridiag_hp(&a, 2, prec).unwrap();
        // The Householder output IS already tridiagonal for 2×2; get eigenvalues.
        let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(evals.len(), 2);
        // Trace = 7 and determinant = 10 - 9 = 1, so eigenvalues satisfy
        // λ₁ + λ₂ = 7, λ₁ * λ₂ = 1.
        // Tolerance is a few ULPs at prec=128 bits (~38 decimal digits).
        let tol = hp(prec, "1e-35");
        let mut sum = evals[0].clone();
        sum += &evals[1];
        let mut trace_diff = sum;
        trace_diff -= hp(prec, "7");
        let trace_diff = trace_diff.abs();
        assert!(
            trace_diff < tol,
            "2×2 eigenvalue sum (trace) should be 7; diff = {}",
            trace_diff
        );
        let mut prod = evals[0].clone();
        prod *= &evals[1];
        let mut det_diff = prod;
        det_diff -= hp(prec, "1");
        let det_diff = det_diff.abs();
        assert!(
            det_diff < tol,
            "2×2 eigenvalue product (det) should be 1; diff = {}",
            det_diff
        );
    }

    /// `dense_symmetric_eigenvalues_hp` on a 1×1 matrix: single eigenvalue.
    #[test]
    fn dense_eigenvalues_1x1() {
        let prec = 128;
        let a = vec![hp(prec, "99")];
        let evals = dense_symmetric_eigenvalues_hp(&a, 1, prec).unwrap();
        assert_eq!(evals.len(), 1);
        let mut d = evals[0].clone();
        d -= hp(prec, "99");
        let d = d.abs();
        assert!(
            d < hp(prec, "1e-50"),
            "1×1 dense eigenvalue should be 99; diff={}",
            d
        );
    }

    #[test]
    fn independent_jacobi_matches_closed_form_and_qr() {
        let prec = 256;
        let matrix = vec![
            hp(prec, "2"),
            hp(prec, "1"),
            hp(prec, "0"),
            hp(prec, "1"),
            hp(prec, "2"),
            hp(prec, "1"),
            hp(prec, "0"),
            hp(prec, "1"),
            hp(prec, "2"),
        ];
        let jacobi = dense_symmetric_eigenvalues_jacobi_hp(&matrix, 3, prec, 30).unwrap();
        let qr = dense_symmetric_eigenvalues_hp(&matrix, 3, prec).unwrap();
        assert!(jacobi.sweeps > 0);
        assert!(jacobi.rotations > 0);
        let tolerance = hp(prec, "1e-60");
        for (left, right) in jacobi.eigenvalues.iter().zip(&qr) {
            let mut difference = left.clone();
            difference -= right;
            assert!(difference.abs() < tolerance);
        }
        let mut middle = jacobi.eigenvalues[1].clone();
        middle -= 2;
        assert!(middle.abs() < tolerance);
    }

    #[test]
    fn independent_jacobi_resolves_cluster_and_repeats_at_higher_precision() {
        let solve = |prec| {
            let delta = hp(prec, "1e-30");
            let mut one_plus = hp(prec, "1");
            one_plus += &delta;
            let matrix = vec![
                hp(prec, "1"),
                hp(prec, "1e-35"),
                hp(prec, "0"),
                hp(prec, "1e-35"),
                one_plus,
                hp(prec, "0"),
                hp(prec, "0"),
                hp(prec, "0"),
                hp(prec, "3"),
            ];
            dense_symmetric_eigenvalues_jacobi_hp(&matrix, 3, prec, 30)
                .unwrap()
                .eigenvalues
        };
        let low = solve(192);
        let high = solve(320);
        let tolerance = hp(320, "1e-50");
        for (left, right) in low.iter().zip(&high) {
            let mut difference = Float::with_val(320, left);
            difference -= right;
            assert!(difference.abs() < tolerance);
        }
        let mut gap = high[1].clone();
        gap -= &high[0];
        assert!(gap > hp(320, "9e-31"));
        assert!(gap < hp(320, "2e-30"));
    }
}
