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
//!   Deflation uses `2^-prec` times the adjacent diagonal magnitudes.
//!   This is a working-precision approximation, not an eigenvalue enclosure.
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

#[path = "eigen_recovery.rs"]
mod recovery;
#[path = "eigen_sturm.rs"]
mod sturm;
pub use recovery::{
    dense_symmetric_eigenpair_at_index_hp, dense_symmetric_eigenvector_for_value_detailed_hp,
    tridiag_eigenvector_for_value_detailed_hp, HpEigenvectorRecovery, HpEigenvectorRecoveryFailure,
    DENSE_EIGENVECTOR_SEMANTICS,
};

/// Arithmetic identity for retained QR results.
pub const TRIDIAG_QR_SEMANTICS: &str = "tridiag-qr-working-unit-deflation-exponent-safe-hypot-v3";

/// Relative working-unit threshold for symmetric tridiagonal QR deflation.
fn qr_tolerance(prec: u32) -> Float {
    let two = Float::with_val(prec, 2);
    let exponent = -(prec as i32);
    two.pow(exponent)
}

/// A user work budget ended before the selected interval met its width.
/// Increasing arithmetic precision does not increase this budget.
#[derive(Debug)]
pub struct SelectedEigenvalueIterationLimit {
    pub index: usize,
    pub maximum_iterations: usize,
}
impl std::fmt::Display for SelectedEigenvalueIterationLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "selected eigenvalue {} exhausted {} bisection iterations",
            self.index, self.maximum_iterations
        )
    }
}
impl std::error::Error for SelectedEigenvalueIterationLimit {}

/// Sweep count at which QR reports slow convergence without changing the
/// arithmetic sequence or stopping the solve.
pub const TRIDIAG_QR_SLOW_SWEEP_WARNING: usize = 100;

/// Default hard QR sweep limit per deflating eigenvalue.
///
/// Large high-precision tridiagonal matrices can converge correctly but need
/// more than 100 sweeps. One hundred is therefore only a diagnostic threshold;
/// the default hard safety limit is deliberately aligned with the toolkit's
/// other slow-but-accurate high-precision iterative routes.
pub const DEFAULT_TRIDIAG_QR_MAX_ITERATIONS: usize = 2_000;

/// Maximum QR sweep iterations allowed per deflating eigenvalue before
/// `tridiag_eigenvalues_hp` gives up with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TridiagQrOptions {
    /// Maximum QR sweeps allowed per deflating eigenvalue.
    pub max_iterations_per_eigenvalue: usize,
}

impl Default for TridiagQrOptions {
    fn default() -> Self {
        Self {
            max_iterations_per_eigenvalue: DEFAULT_TRIDIAG_QR_MAX_ITERATIONS,
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
/// Returns finite eigenvalues sorted ascending, each at `prec` bits.
/// Precision must exceed 32 bits. Nonfinite inputs or nonrepresentable QR
/// intermediates return errors. The input is rounded to the requested precision;
/// computed eigenvalues require separate residual and state-selection checks.
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
    validate_hp_eigen_precision(prec)?;
    let n = diag.len();
    if n == 0 {
        if !off_diag.is_empty() {
            return Err(anyhow!(
                "an empty HP tridiagonal matrix requires no couplings"
            ));
        }
        return Ok(Vec::new());
    }
    validate_hp_tridiagonal(diag, off_diag, prec)?;
    if off_diag.len() != n - 1 {
        return Err(anyhow!(
            "off_diag length {} should be {} (= diag length - 1)",
            off_diag.len(),
            n - 1
        ));
    }
    // Working copies; algorithm mutates these in place.
    // Pad e with one trailing zero so e[m] is always valid for m up to n-1.
    let mut d: Vec<Float> = diag.iter().map(|v| Float::with_val(prec, v)).collect();
    let mut e: Vec<Float> = off_diag.iter().map(|v| Float::with_val(prec, v)).collect();
    if d.iter().chain(&e).any(|v| !v.is_finite()) {
        return Err(anyhow!("HP QR inputs overflow at the requested precision"));
    }
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
    let mut sc_r_outer = hp_zero(prec);
    let mut sc_signed_r = hp_zero(prec);
    let mut sc_g_plus_sr = hp_zero(prec);
    let mut sc_new_g = hp_zero(prec);
    let mut sc_shifted_diag = hp_zero(prec);
    let mut sc_f = hp_zero(prec);
    let mut sc_b = hp_zero(prec);
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
                if !sc_dd.is_finite() || !sc_threshold.is_finite() {
                    return Err(anyhow!(
                        "HP QR deflation arithmetic exceeds the finite exponent range"
                    ));
                }
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
                    "tridiag QR failed to converge for eigenvalue at l={} after the hard limit of {} sweeps; no incomplete eigenvalue was returned",
                    l,
                    max_iter
                ));
            }
            if iter_count == TRIDIAG_QR_SLOW_SWEEP_WARNING + 1
                && max_iter > TRIDIAG_QR_SLOW_SWEEP_WARNING
            {
                xc_core::progress_message!(
                    "[HP] tridiagonal QR slow convergence at eigenvalue l={l}: continuing beyond {TRIDIAG_QR_SLOW_SWEEP_WARNING} sweeps (hard limit={max_iter})"
                );
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
            if !sc_two_el.is_finite() || !sc_g.is_finite() {
                return Err(anyhow!(
                    "HP QR shift arithmetic exceeds the finite exponent range"
                ));
            }

            // MPFR hypot avoids overflow/underflow of separately squared operands.
            sc_r_outer.assign(sc_g.clone().hypot(&hp_one(prec)));
            if !sc_r_outer.is_finite() {
                return Err(anyhow!(
                    "HP QR shift norm exceeds the finite exponent range"
                ));
            }

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

                // Compute the finite norm without squaring exponent-extreme entries.
                sc_new_r.assign(sc_f.clone().hypot(&g));
                if !sc_new_r.is_finite()
                    || (sc_new_r.is_zero() && (!sc_f.is_zero() || !g.is_zero()))
                {
                    return Err(anyhow!(
                        "HP QR rotation norm exceeds the representable exponent range"
                    ));
                }

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
                if [&d[i + 1], &g, &p, &s, &c].iter().any(|v| !v.is_finite()) {
                    return Err(anyhow!("HP QR rotation produced a nonfinite value"));
                }
            }

            if !converged_early {
                d[l] -= &p;
                e[l].assign(&g);
                e[m] = hp_zero(prec);
            }
            // Loop back to find new m; eventually m == l and we break.
        }
    }

    if d.iter().any(|v| !v.is_finite()) {
        return Err(anyhow!("HP QR produced a nonfinite eigenvalue"));
    }
    // Sort only finite values; NaN is never treated as an ordering tie.
    d.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues"));
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
    /// Directed angle evidence for this returned vector and exact stored matrix.
    pub recovery: HpEigenvectorRecovery,
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

fn validate_hp_eigen_precision(prec: u32) -> Result<()> {
    if prec <= 32 || prec > rug::float::prec_max().min(i32::MAX as u32) {
        return Err(anyhow!(
            "HP eigen precision must exceed 32 bits and fit the supported exponent arithmetic"
        ));
    }
    Ok(())
}

fn validate_hp_tridiagonal(diag: &[Float], off_diag: &[Float], prec: u32) -> Result<()> {
    if diag.is_empty() || off_diag.len() + 1 != diag.len() {
        return Err(anyhow!(
            "HP tridiagonal problem requires off_diag.len() + 1 == diag.len() > 0"
        ));
    }
    validate_hp_eigen_precision(prec)?;
    if diag.iter().chain(off_diag).any(|value| !value.is_finite()) {
        return Err(anyhow!("HP tridiagonal entries must be finite"));
    }
    Ok(())
}

fn tridiag_sturm_count_below_hp_unchecked(
    diag: &[Float],
    off_diag: &[Float],
    threshold: &Float,
    prec: u32,
) -> Result<usize> {
    sturm::count(diag, off_diag, threshold, prec)
}

/// Count eigenvalues strictly below the exact stored HP threshold.
/// Directed MPFR determinant intervals must prove every sign or exact zero.
/// Exact zero couplings split independent blocks. The initial guard precision
/// exceeds every input precision by 32 bits. The adaptive cap is the greater
/// of source+64+2*n and 4*(source+32), limited to 1,000,000 bits.
/// Unresolved signs or nonrepresentable interval arithmetic return errors.
/// This count concerns the exact stored matrix, not matrix-assembly uncertainty.
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
    tridiag_sturm_count_below_hp_unchecked(diag, off_diag, threshold, prec)
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
/// of the exact stored symmetric tridiagonal matrix. Endpoint counts use
/// directed interval signs, and returned widths are checked with rounding up.
/// Unresolved signs, arithmetic limits or precision stagnation return errors.
/// These enclosures do not include matrix-assembly or reduction uncertainty;
/// eigenvectors and their residuals still require separate validation.
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
    crate::mpfr_interval::ensure_uniform_exponent_range()
        .map_err(|error| anyhow!(error.to_string()))?;
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
    if !global_lower.is_finite() || !global_upper.is_finite() || global_lower >= global_upper {
        return Err(anyhow!(
            "HP Gershgorin bounds exceed the finite representable range"
        ));
    }
    if !tolerance.is_finite() || tolerance <= 0 {
        return Err(anyhow!(
            "HP tolerance is not representable at working precision"
        ));
    }
    let global_lower_count =
        tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &global_lower, prec)?;
    let global_upper_count =
        tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &global_upper, prec)?;
    if global_lower_count != 0 || global_upper_count != diag.len() {
        return Err(anyhow!(
            "HP Gershgorin bracket failed count reconciliation: [{global_lower_count}, {global_upper_count}] for dimension {}",
            diag.len()
        ));
    }

    // Every algebraic index follows an independent deterministic bisection
    // over the same immutable tridiagonal input. Indexed parallel collection
    // preserves the requested order and each individual arithmetic sequence.
    let indexed = (first_index..=last_index)
        .into_par_iter()
        .map(|index| {
            let mut lower = global_lower.clone();
            let mut upper = global_upper.clone();
            let mut lower_count = global_lower_count;
            let mut upper_count = global_upper_count;
            let mut iterations = 0usize;
            let mut verified: Option<(Float, Float)> = None;
            let mut guided = false;
            loop {
                let (width, _) = Float::with_val_round(prec, &upper - &lower, rug::float::Round::Up);
                if width <= *absolute_tolerance {
                    break;
                }
                if iterations == maximum_iterations {
                    return Err(SelectedEigenvalueIterationLimit { index, maximum_iterations }.into());
                }
                if !guided && lower_count == index && upper_count == index + 1 {
                    guided = true;
                    verified = isolated_eigenvalue_guide(
                        diag,
                        off_diag,
                        index,
                        &lower,
                        &upper,
                        absolute_tolerance,
                        prec,
                    );
                }
                // Half-sum avoids overflow when both finite endpoints are large.
                let mut midpoint = lower.clone() / 2u32;
                midpoint += upper.clone() / 2u32;
                if !midpoint.is_finite() || midpoint < lower || midpoint > upper {
                    return Err(anyhow!("HP Sturm midpoint is outside the finite bracket"));
                }
                if midpoint == lower || midpoint == upper {
                    return Err(anyhow!(
                        "HP Sturm bisection stagnated at {prec} bits for eigenvalue {index}; precision escalation is required"
                    ));
                }
                // Inside the isolated bracket, a midpoint at or below a
                // verified `a` counts exactly `index`; at or above `b`,
                // exactly `index + 1`. Only midpoints in (a, b) are counted.
                let midpoint_count = match &verified {
                    Some((a, _)) if midpoint <= *a => index,
                    Some((_, b)) if midpoint >= *b => index + 1,
                    _ => tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &midpoint, prec)?,
                };
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
            Ok((
                HpTridiagonalEigenvalueEnclosure {
                    index,
                    lower,
                    upper,
                    lower_count,
                    upper_count,
                    iterations,
                },
                iterations,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let sturm_evaluations = indexed.iter().try_fold(2usize, |total, (_, evaluations)| {
        total
            .checked_add(*evaluations)
            .ok_or_else(|| anyhow!("HP Sturm evaluation count overflow"))
    })?;
    let enclosures = indexed
        .into_iter()
        .map(|(enclosure, _)| enclosure)
        .collect();
    Ok(HpSelectedTridiagonalSpectrum {
        precision_bits: prec,
        first_index,
        last_index,
        sturm_evaluations,
        enclosures,
    })
}

/// Verified points `a < b` around the isolated eigenvalue `index`, with exact
/// directed counts `count(a) == index` and `count(b) == index + 1`. A
/// safeguarded point Newton iteration on `det(T - xI)` only places them; the
/// directed counts decide. Bisection then replays midpoints outside `(a, b)`
/// without counting, so its path, enclosure, counts and iteration total are
/// those of the plain bisection. `None` leaves plain bisection unchanged.
fn isolated_eigenvalue_guide(
    diag: &[Float],
    off_diag: &[Float],
    index: usize,
    lower: &Float,
    upper: &Float,
    tolerance: &Float,
    prec: u32,
) -> Option<(Float, Float)> {
    // Resolve the guide well below the final bisection width.
    let magnitude = [lower, upper]
        .into_iter()
        .filter_map(Float::get_exp)
        .max()
        .unwrap_or(0);
    let depth = i64::from(magnitude) - i64::from(tolerance.get_exp()?);
    let work = u32::try_from(depth.max(i64::from(prec)).saturating_add(96))
        .ok()?
        .min(1_000_000);
    let spacing = Float::with_val(work, tolerance) >> 16;
    let mut low = Float::with_val(work, lower);
    let mut high = Float::with_val(work, upper);
    let mut x = Float::with_val(work, &low + &high) / 2u32;
    let mut converged = false;
    for _ in 0..256 {
        // Ratio recurrence q_i = (d_i - x) - e_{i-1}^2 / q_{i-1}: negatives
        // count eigenvalues below x, and sum(q_i'/q_i) = f'/f for f = det.
        let mut negatives = 0usize;
        let mut ratio = Float::with_val(work, 0);
        let mut q = Float::with_val(work, 0);
        let mut dq = Float::with_val(work, 0);
        let mut singular = false;
        for i in 0..diag.len() {
            let mut next = Float::with_val(work, &diag[i] - &x);
            let mut next_dq = Float::with_val(work, -1);
            if i > 0 {
                let coupling = Float::with_val(work, off_diag[i - 1].square_ref());
                if !coupling.is_zero() {
                    let mut term = Float::with_val(work, &coupling / &q);
                    next -= &term;
                    term /= &q;
                    term *= &dq;
                    next_dq += term;
                }
            }
            if next.is_zero() || !next.is_finite() || !next_dq.is_finite() {
                singular = true;
                break;
            }
            if next < 0 {
                negatives += 1;
            }
            ratio += Float::with_val(work, &next_dq / &next);
            q = next;
            dq = next_dq;
        }
        if singular {
            x = Float::with_val(work, &low + &high) / 2u32;
            continue;
        }
        if negatives <= index {
            low.assign(&x);
        } else {
            high.assign(&x);
        }
        let candidate = if ratio.is_zero() || !ratio.is_finite() {
            None
        } else {
            Some(Float::with_val(
                work,
                &x - Float::with_val(work, ratio.recip_ref()),
            ))
        };
        match candidate {
            Some(next) if next > low && next < high => {
                let step = Float::with_val(work, &next - &x).abs();
                x = next;
                if step <= spacing {
                    converged = true;
                    break;
                }
            }
            _ => {
                x = Float::with_val(work, &low + &high) / 2u32;
                if Float::with_val(work, &high - &low) <= spacing {
                    converged = true;
                    break;
                }
            }
        }
    }
    if !converged {
        return None;
    }
    for shift in [16u32, 4] {
        let offset = Float::with_val(work, tolerance) >> shift;
        let a = Float::with_val(work, &x - &offset);
        let b = Float::with_val(work, &x + &offset);
        let below = tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &a, prec).ok()?;
        let above = tridiag_sturm_count_below_hp_unchecked(diag, off_diag, &b, prec).ok()?;
        if below == index && above == index + 1 {
            return Some((a, b));
        }
    }
    None
}

// Bound ||T||_2 by ||T||_infinity for symmetric T. The search bracket's
// absolute padding is not matrix data and must not relax residual acceptance.
fn tridiag_matrix_scale_hp(diag: &[Float], off_diag: &[Float], prec: u32) -> Result<Float> {
    use rug::{float::Round, ops::AddAssignRound};
    let mut scale = hp_zero(prec);
    for i in 0..diag.len() {
        let (mut row, _) = Float::with_val_round(prec, diag[i].clone().abs(), Round::Up);
        if i > 0 {
            row.add_assign_round(off_diag[i - 1].clone().abs(), Round::Up);
        }
        if i + 1 < diag.len() {
            row.add_assign_round(off_diag[i].clone().abs(), Round::Up);
        }
        if !row.is_finite() {
            return Err(anyhow!(
                "selected-eigenpair matrix norm bound is unrepresentable"
            ));
        }
        if row > scale {
            scale = row;
        }
    }
    // Only the exactly zero matrix needs an arbitrary positive normalization.
    if scale.is_zero() {
        scale.assign(1);
    }
    Ok(scale)
}

fn tridiag_rayleigh_and_residual_hp(
    diag: &[Float],
    off_diag: &[Float],
    vector: &[Float],
    matrix_scale: &Float,
    prec: u32,
) -> Result<(Float, EigenpairDiagnostics<Float>)> {
    if !matrix_scale.is_finite() || matrix_scale <= &0 || vector.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!("invalid selected-eigenpair diagnostic domain"));
    }
    let mut denominator = hp_zero(prec);
    let mut numerator = hp_zero(prec);
    let mut actions = Vec::with_capacity(diag.len());
    for i in 0..diag.len() {
        denominator += Float::with_val(prec, &vector[i] * &vector[i]);
        let mut action = Float::with_val(prec, &diag[i] / matrix_scale) * &vector[i];
        if i > 0 {
            action += Float::with_val(prec, &off_diag[i - 1] / matrix_scale) * &vector[i - 1];
        }
        if i + 1 < diag.len() {
            action += Float::with_val(prec, &off_diag[i] / matrix_scale) * &vector[i + 1];
        }
        numerator += Float::with_val(prec, &vector[i] * &action);
        actions.push(action);
    }
    if !denominator.is_finite() || denominator <= 0 {
        return Err(anyhow!("invalid recovered vector norm"));
    }
    let scaled_value = Float::with_val(prec, numerator / &denominator);
    let eigenvalue = Float::with_val(prec, &scaled_value * matrix_scale);
    let mut residual = ScaledSquareSum::new(prec);
    let mut action_norm = ScaledSquareSum::new(prec);
    for (action, component) in actions.iter().zip(vector) {
        action_norm.add(action)?;
        residual.add(&Float::with_val(
            prec,
            action - Float::with_val(prec, &scaled_value * component),
        ))?;
    }
    let residual = residual.norm()?;
    let action_norm = action_norm.norm()?;
    let absolute_residual = Float::with_val(prec, &residual * matrix_scale);
    if !eigenvalue.is_finite()
        || !absolute_residual.is_finite()
        || (!residual.is_zero() && absolute_residual.is_zero())
        || (!scaled_value.is_zero() && eigenvalue.is_zero())
    {
        return Err(anyhow!(
            "selected-eigenpair diagnostic rescaling is unrepresentable"
        ));
    }
    let vector_norm = denominator.clone().sqrt();
    let eigenvalue_scale = Float::with_val(prec, scaled_value.abs() * &vector_norm);
    let relative_denominator = Float::with_val(prec, action_norm + &eigenvalue_scale);
    let relative_residual = if relative_denominator.is_zero() {
        residual.clone()
    } else {
        Float::with_val(prec, &residual / relative_denominator)
    };
    let backward_denominator = Float::with_val(prec, vector_norm + eigenvalue_scale);
    let scaled_backward_error = Float::with_val(prec, &residual / backward_denominator);
    let orthogonality_error = Float::with_val(prec, denominator - 1).abs();
    Ok((
        eigenvalue,
        EigenpairDiagnostics {
            absolute_residual,
            relative_residual,
            scaled_backward_error,
            orthogonality_error,
        },
    ))
}

/// Bind a computed vector to an algebraic index of the exact stored matrix.
/// For symmetric T, dist(rho, spectrum(T)) <= ||T v-rho v||_2/||v||_2.
/// Directed arithmetic encloses that residual ball; Sturm counts must isolate
/// exactly the requested index in the whole ball, including both endpoints.
/// Recover HP eigenvectors only for selected values whose endpoint counts
/// establish a one-dimensional eigenspace. Multiplicities are coalesced into
/// cluster records and never assigned arbitrary individual vectors. Residual
/// acceptance uses directed bounds on the returned vector's residual and
/// separation from all other indices. The original requested eigenvalue
/// enclosure and the separate vector-angle evidence are both retained.
/// Unresolved index or multiplicity,
/// unrepresentable bounds, or failed recovery return errors.
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
    let mut spectrum = tridiag_selected_eigenvalues_hp(
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
    let matrix_scale = tridiag_matrix_scale_hp(diag, off_diag, prec)?;
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

        let recovery = recovery::tridiagonal_at_index(
            diag,
            off_diag,
            enclosure.index,
            prec,
            eigenvector_options,
        )
        .map_err(|e| {
            e.context(format!(
                "selected index {} recovery failed",
                enclosure.index
            ))
        })?;
        let eigenvector = recovery.eigenvector.clone();
        let eigenvalue = recovery.eigenvalue.clone();
        let (_, diagnostics) =
            tridiag_rayleigh_and_residual_hp(diag, off_diag, &eigenvector, &matrix_scale, prec)?;
        spectrum.sturm_evaluations += 2;
        vector_recoveries += 1;
        inverse_iteration_runs += 1;
        let residual_norm = recovery.residual_upper_bound.clone();
        items.push(HpSelectedTridiagonalItem::SimpleEigenpair(Box::new(
            HpSelectedTridiagonalEigenpair {
                enclosure: enclosure.clone(),
                eigenvalue,
                eigenvector,
                residual_norm,
                diagnostics,
                recovery,
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
/// Both banded names use corrected interleaved pivots. Convergence and
/// residuals require verification. Historical defective arithmetic requires
/// a pinned older revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagSolver {
    /// Compatibility name for the corrected O(n) interleaved-pivot route.
    /// Its semantics ID matches BandedInterleaved; v1 arithmetic is not used.
    Banded,
    /// Corrected adjacent-pivot RHS solve, O(n) storage and work per step.
    /// Explicit name for the same corrected arithmetic as Banded.
    BandedInterleaved,
    /// Dense LU after explicitly densifying `(T - λI + ε·I)` to an
    /// `n × n` matrix. O(n³) factor, O(n²) per-step solve, O(n²)
    /// memory. Retained for cross-validation: reviewers can compare
    /// banded vs dense outputs to confirm they agree to working
    /// precision (the test
    /// `eigen::tests::banded_matches_dense_on_strang_n10` does this
    /// at HP-256).
    Dense,
}

impl TridiagSolver {
    /// Bind this value into any new retained algorithm identity.
    pub fn semantics_id(self) -> &'static str {
        match self {
            Self::Banded | Self::BandedInterleaved => {
                "tridiag-interleaved-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v10"
            }
            Self::Dense => "tridiag-dense-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v10",
        }
    }
}

/// Options for `tridiag_eigenvector_for_value_hp`.
///
/// `Default::default()` selects the corrected BandedInterleaved route.
/// New analyses must still independently
/// check the recovered vector's residual and branch eligibility.
#[derive(Debug, Clone, Copy)]
pub struct TridiagEigvecOptions {
    /// Upper bound on inverse-iteration steps. The iteration runs at
    /// most this many steps; with `early_termination = true`, it
    /// usually finishes much earlier (typically 20–50 steps for
    /// well-conditioned inputs with widely-separated eigenvalues).
    pub max_steps: usize,
    /// Stop once the directed residual/separation bound meets the working
    /// accuracy target or its explicit returned-vector rounding floor.
    /// The report states the actual angle bound; this is not a claim that all
    /// requested arithmetic bits are accurate eigenvector bits.
    /// Set false to run the full deterministic step budget.
    pub early_termination: bool,
    /// Inner solver for the LU step. See `TridiagSolver`.
    pub solver: TridiagSolver,
}

impl Default for TridiagEigvecOptions {
    fn default() -> Self {
        Self {
            max_steps: 200,
            early_termination: true,
            solver: TridiagSolver::BandedInterleaved,
        }
    }
}

/// Find the eigenvector of a symmetric tridiagonal matrix corresponding
/// to the (already-known) eigenvalue `eigenvalue` via shifted inverse
/// iteration on `(T - λI + ε·I)`, where the perturbation is
/// `2^-(prec - 32)` times the maximum matrix/target magnitude. This
/// reduces exact singularity risk; it does not guarantee convergence or prove
/// that the supplied value is an eigenvalue.
///
/// The default options use the corrected BandedInterleaved arithmetic.
/// Banded is a compatibility alias for that same corrected route. Callers who
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
    tridiag_eigenvector_with_shift_error(diag, off_diag, eigenvalue, prec, opts, None, false)
}

/// Recover a vector when the requested eigenvalue has a supplied absolute
/// uncertainty bound. The caller is responsible for establishing that bound.
/// It may identify a unique member but cannot relax the residual/angle gate.
/// Ambiguous uncertainty spanning multiple members returns an error.
pub fn tridiag_eigenvector_for_value_with_uncertainty_hp(
    diag: &[Float],
    off_diag: &[Float],
    eigenvalue: &Float,
    eigenvalue_uncertainty: &Float,
    prec: u32,
    opts: TridiagEigvecOptions,
) -> Result<Vec<Float>> {
    if !eigenvalue_uncertainty.is_finite() || eigenvalue_uncertainty < &0 {
        return Err(anyhow!(
            "eigenvalue uncertainty must be finite and nonnegative"
        ));
    }
    tridiag_eigenvector_with_shift_error(
        diag,
        off_diag,
        eigenvalue,
        prec,
        opts,
        Some(eigenvalue_uncertainty),
        false,
    )
}

fn tridiag_eigenvector_with_shift_error(
    diag: &[Float],
    off_diag: &[Float],
    eigenvalue: &Float,
    prec: u32,
    opts: TridiagEigvecOptions,
    shift_error: Option<&Float>,
    _roundoff_shift: bool,
) -> Result<Vec<Float>> {
    Ok(tridiag_eigenvector_for_value_detailed_hp(
        diag,
        off_diag,
        eigenvalue,
        shift_error,
        prec,
        opts,
    )?
    .eigenvector)
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
fn apply_householder_trailing_update_hp(
    h: &mut [Float],
    n: usize,
    k: usize,
    v: &[Float],
    q_vec: &[Float],
) {
    let m = v.len();
    h[(k + 1) * n..]
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..m {
                let mut delta = v[i].clone();
                delta *= &q_vec[j];
                let mut delta2 = q_vec[i].clone();
                delta2 *= &v[j];
                delta += &delta2;
                row[k + 1 + j] -= &delta;
            }
        });
}

/// Reduce a finite, exactly symmetric stored matrix with the scaled,
/// opposite-sign Householder algorithm. Input entries are rounded once to
/// `prec` bits before arithmetic; this general API permits explicit working
/// precision reduction. Use [`householder_tridiag_hp_stable`] to reject
/// down-rounding instead. Results are computed points, not certificates.
pub fn householder_tridiag_hp(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>)> {
    householder_tridiag_hp_impl(a, n, prec, true)
}

fn householder_tridiag_hp_impl(
    a: &[Float],
    n: usize,
    prec: u32,
    accumulate_q: bool,
) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>)> {
    validate_hp_eigen_precision(prec)?;
    if prec > 1_000_000 || n == 0 || n.checked_mul(n) != Some(a.len()) {
        return Err(anyhow!(
            "invalid Householder shape or unsupported analysis precision"
        ));
    }
    if a.iter().any(|x| !x.is_finite())
        || (0..n).any(|i| (0..i).any(|j| a[i * n + j] != a[j * n + i]))
    {
        return Err(anyhow!(
            "Householder requires finite exactly symmetric storage"
        ));
    }
    householder_tridiag_hp_route(a, n, prec, accumulate_q)
}

/// Stable route identity; distinct from historical same-sign reflector sources.
pub const STABLE_HOUSEHOLDER_SEMANTICS: &str = "householder-scaled-opposite-sign-v1";

/// Stable symmetric reduction with an explicitly accumulated orthogonal basis.
/// Uses scaled columns and the opposite-sign norm so the reflector's first
/// component does not cancel. The inputs are finite and exactly symmetric;
/// source values may be widened to analysis precision but never down-rounded.
/// This is a computed decomposition, not a certified spectrum or eigenvector.
pub fn householder_tridiag_hp_stable(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>)> {
    validate_stable_symmetric_input(a, n, prec)?;
    householder_tridiag_hp_route(a, n, prec, true)
}

/// The same stable reduction, omitting the n-by-n Q allocation and updates.
/// Its tridiagonal entries are identical to the with-Q route at fixed inputs.
pub fn dense_symmetric_tridiagonal_hp_stable(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    validate_stable_symmetric_input(a, n, prec)?;
    let (d, e, _) = householder_tridiag_hp_route(a, n, prec, false)?;
    Ok((d, e))
}

/// Stable Householder reduction followed by the existing tridiagonal QR.
/// The returned eigenvalues are computed estimates, not certified enclosures.
/// No relative-accuracy or sign guarantee is made for eigenvalues tiny compared
/// with the input norm. A positive returned point can be unresolved at this
/// source/working precision; assess retained-source precision sensitivity and
/// matrix-assembly error separately (see docs/PREFIX_CONVERGENCE.md).
pub fn dense_symmetric_eigenvalues_hp_stable(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<Vec<Float>> {
    let (d, e) = dense_symmetric_tridiagonal_hp_stable(a, n, prec)?;
    let values = tridiag_eigenvalues_hp(&d, &e, prec)?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!("nonfinite stable spectrum"));
    }
    Ok(values)
}

fn validate_stable_symmetric_input(a: &[Float], n: usize, prec: u32) -> Result<()> {
    if n == 0 || n.checked_mul(n) != Some(a.len()) || !(64..=1_000_000).contains(&prec) {
        return Err(anyhow!(
            "invalid symmetric matrix shape or analysis precision"
        ));
    }
    if a.iter().any(|x| !x.is_finite() || x.prec() > prec) {
        return Err(anyhow!(
            "stable reduction requires finite inputs without down-rounding"
        ));
    }
    for i in 0..n {
        for j in 0..i {
            if a[i * n + j] != a[j * n + i] {
                return Err(anyhow!(
                    "stable reduction requires exactly symmetric storage"
                ));
            }
        }
    }
    Ok(())
}

/// Independently evaluate a supplied reduction against the original matrix.
/// Frobenius residuals use scaled sum-of-squares; no full residual matrix is
/// allocated. Cost is O(n^3) arithmetic and O(n) extra scalar storage beyond
/// the supplied arrays. This does not rerun Householder and can check retained
/// legacy Q/T data. Results are computed diagnostics, not interval certificates.
pub fn assess_symmetric_reduction_hp(
    a: &[Float],
    diag: &[Float],
    off_diag: &[Float],
    q: &[Float],
    prec: u32,
) -> Result<xc_core::SymmetricReductionDiagnostics<Float>> {
    let n = diag.len();
    validate_stable_symmetric_input(a, n, prec)?;
    if off_diag.len() != n - 1
        || q.len() != a.len()
        || diag
            .iter()
            .chain(off_diag)
            .chain(q)
            .any(|x| !x.is_finite() || x.prec() > prec)
    {
        return Err(anyhow!("invalid supplied symmetric reduction"));
    }
    let mut residual = ScaledSquareSum::new(prec);
    let mut orthogonality = ScaledSquareSum::new(prec);
    let mut source = ScaledSquareSum::new(prec);
    let mut tridiagonal = ScaledSquareSum::new(prec);
    let mut basis = ScaledSquareSum::new(prec);
    for v in a {
        source.add(v)?;
    }
    for v in q {
        basis.add(v)?;
    }
    for v in diag {
        tridiagonal.add(v)?;
    }
    for v in off_diag {
        tridiagonal.add(v)?;
        tridiagonal.add(v)?;
    }
    // Bound temporary residual storage to 16 rows. Indexed parallel collection
    // and serial row-major norm accumulation preserve every rounding operation
    // regardless of worker count. Products use reusable, separately rounded
    // scratch values, not fused multiply-adds.
    for start in (0..n).step_by(16) {
        let rows: Vec<_> = (start..(start + 16).min(n))
            .into_par_iter()
            .map_init(
                || (hp_zero(prec), hp_zero(prec), hp_zero(prec)),
                |(aq, qtq, product), i| {
                    (0..n)
                        .map(|j| {
                            aq.assign(0);
                            qtq.assign(0);
                            for k in 0..n {
                                product.assign(&a[i * n + k] * &q[k * n + j]);
                                *aq += &*product;
                                product.assign(&q[k * n + i] * &q[k * n + j]);
                                *qtq += &*product;
                            }
                            product.assign(&q[i * n + j] * &diag[j]);
                            *aq -= &*product;
                            if j > 0 {
                                product.assign(&q[i * n + j - 1] * &off_diag[j - 1]);
                                *aq -= &*product;
                            }
                            if j + 1 < n {
                                product.assign(&q[i * n + j + 1] * &off_diag[j]);
                                *aq -= &*product;
                            }
                            if i == j {
                                *qtq -= 1;
                            }
                            (aq.clone(), qtq.clone())
                        })
                        .collect::<Vec<_>>()
                },
            )
            .collect();
        for row in rows {
            for (aq, qtq) in row {
                residual.add(&aq)?;
                orthogonality.add(&qtq)?;
            }
        }
    }
    let absolute_similarity_residual = residual.norm()?;
    let absolute_orthogonality_residual = orthogonality.norm()?;
    let source_frobenius_norm = source.norm()?;
    let tridiagonal_frobenius_norm = tridiagonal.norm()?;
    let basis_frobenius_norm = basis.norm()?;
    let mut denominator =
        Float::with_val(prec, &source_frobenius_norm + &tridiagonal_frobenius_norm);
    denominator *= &basis_frobenius_norm;
    if !denominator.is_finite() {
        return Err(anyhow!("unrepresentable diagnostic normalization"));
    }
    let mut relative_similarity_residual = absolute_similarity_residual.clone();
    if !denominator.is_zero() {
        relative_similarity_residual /= denominator;
    }
    let relative_orthogonality_residual = Float::with_val(
        prec,
        &absolute_orthogonality_residual / Float::with_val(prec, n).sqrt(),
    );
    Ok(xc_core::SymmetricReductionDiagnostics {
        absolute_similarity_residual,
        relative_similarity_residual,
        absolute_orthogonality_residual,
        relative_orthogonality_residual,
        source_frobenius_norm,
        tridiagonal_frobenius_norm,
        basis_frobenius_norm,
    })
}

// Streaming norm avoids squaring the largest/smallest raw exponent and avoids
// allocating two n-by-n diagnostic residual arrays. Reduction order is fixed.
struct ScaledSquareSum {
    scale: Float,
    sum: Float,
}
impl ScaledSquareSum {
    fn new(p: u32) -> Self {
        Self {
            scale: hp_zero(p),
            sum: hp_zero(p),
        }
    }
    fn add(&mut self, value: &Float) -> Result<()> {
        if !value.is_finite() {
            return Err(anyhow!("nonfinite reduction diagnostic"));
        }
        let value = value.clone().abs();
        if value.is_zero() {
            return Ok(());
        }
        let p = self.scale.prec();
        if value > self.scale {
            let mut ratio = Float::with_val(p, &self.scale / &value);
            ratio.square_mut();
            self.sum *= ratio;
            self.sum += 1;
            self.scale.assign(value);
        } else {
            let mut ratio = Float::with_val(p, &value / &self.scale);
            ratio.square_mut();
            self.sum += ratio;
        }
        Ok(())
    }
    fn norm(self) -> Result<Float> {
        let p = self.scale.prec();
        let value = Float::with_val(p, self.scale * self.sum.sqrt());
        if !value.is_finite() {
            return Err(anyhow!("unrepresentable Frobenius norm"));
        }
        Ok(value)
    }
}

fn householder_tridiag_hp_route(
    a: &[Float],
    n: usize,
    prec: u32,
    accumulate_q: bool,
) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>)> {
    if n == 0 || n.checked_mul(n) != Some(a.len()) {
        return Err(anyhow!("invalid Householder matrix dimension or length"));
    }
    let mut h: Vec<Float> = a.iter().map(|x| Float::with_val(prec, x)).collect();
    if h.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!(
            "Householder input precision conversion is unrepresentable"
        ));
    }

    // Q starts as identity; we apply each Householder reflection from the
    // right as we go, accumulating into Q. (Equivalently, store Householder
    // vectors and reconstruct Q at the end — that's slightly more compact
    // but more code. We do the direct accumulation for simplicity.)
    let mut q: Vec<Float> = if accumulate_q {
        vec![hp_zero(prec); n * n]
    } else {
        Vec::new()
    };
    if accumulate_q {
        for i in 0..n {
            q[i * n + i] = hp_one(prec);
        }
    }

    // For each column k = 0..n-2, build a Householder reflector that
    // zeros out h[k+2..n, k] (and symmetrically h[k, k+2..n]).
    for k in 0..n.saturating_sub(2) {
        // Pull out the column-k subdiagonal portion: x = h[k+1..n, k].
        let m = n - k - 1; // length of subdiagonal portion
        let mut x: Vec<Float> = (0..m).map(|i| h[(k + 1 + i) * n + k].clone()).collect();
        let column_scale = {
            let scale = x
                .iter()
                .map(|v| v.clone().abs())
                .max_by(Float::total_cmp)
                .unwrap();
            if !scale.is_finite() {
                return Err(anyhow!("nonfinite Householder column"));
            }
            if scale.is_zero() {
                continue;
            }
            for v in &mut x {
                *v /= &scale;
            }
            scale
        };

        // ‖x‖ via parallel reduction.
        let alpha_terms: Vec<Float> = x
            .par_iter()
            .map(|xi| {
                let mut t = xi.clone();
                t *= xi;
                t
            })
            .collect();
        let alpha_sq = crate::reduction::deterministic_pairwise_sum_hp_owned(alpha_terms, prec);
        let alpha = alpha_sq.sqrt();

        // If subdiagonal is already zero, skip.
        if alpha.is_zero() {
            continue;
        }

        // v=x+sign(x0)*norm on a scaled column. The resulting off-diagonal
        // has the opposite sign; changing only v or only T would break AQ=QT.
        let alpha_signed = if x[0].is_sign_negative() {
            -alpha.clone()
        } else {
            alpha.clone()
        };
        let mut v = x;
        v[0] += &alpha_signed;

        // ‖v‖² via parallel reduction.
        let v_norm_terms: Vec<Float> = v
            .par_iter()
            .map(|vi| {
                let mut t = vi.clone();
                t *= vi;
                t
            })
            .collect();
        let v_norm_sq = crate::reduction::deterministic_pairwise_sum_hp_owned(v_norm_terms, prec);
        if v_norm_sq.is_zero() {
            continue;
        }

        // Householder reflection: H = I - 2·v·vᵀ/‖v‖²
        // Apply H from both sides to the trailing (n-k-1) × (n-k-1) sub-block of h.
        // With p = (2/‖v‖²) h_sub v, the exact expansion is
        // H h_sub H = h_sub - v pᵀ - p vᵀ + (2 vᵀp/‖v‖²) v vᵀ.
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
                let mut t = hp_zero(prec);
                for j in 0..m {
                    let entry = &h[(k + 1 + i) * n + (k + 1 + j)];
                    if t.prec() != entry.prec() {
                        t.set_prec(entry.prec());
                    }
                    t.assign(entry);
                    t *= &v[j];
                    acc += &t;
                }
                // p[i] = (2/‖v‖²) · acc
                acc *= 2u32;
                acc /= &v_norm_sq;
                acc
            })
            .collect();

        // With p=2*A*v/(v^T*v), q=p-(v^T*p)/(v^T*v)*v gives
        // H*A*H=A-v*q^T-q*v^T.

        // vᵀ p — parallel reduce.
        let vt_p_terms: Vec<Float> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut t = v[i].clone();
                t *= &p[i];
                t
            })
            .collect();
        let vt_p = crate::reduction::deterministic_pairwise_sum_hp_owned(vt_p_terms, prec);

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
        // Each trailing row is disjoint. Update it directly to avoid an
        // additional m-by-m MPFR matrix while preserving the old cell order.
        apply_householder_trailing_update_hp(&mut h, n, k, &v, &q_vec);

        let mut new_off_diag = -alpha_signed;
        new_off_diag *= column_scale;
        if !new_off_diag.is_finite() {
            return Err(anyhow!("nonfinite Householder off-diagonal"));
        }
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
        if accumulate_q {
            q.par_chunks_mut(n).for_each(|q_row| {
                // Compute coefficient: c = (2/‖v‖²) · sum_j q_row[k+1+j] · v[j]
                let mut c = hp_zero(prec);
                let mut t = hp_zero(prec);
                for j in 0..m {
                    t.assign(&q_row[k + 1 + j]);
                    t *= &v[j];
                    c += &t;
                }
                c *= 2u32;
                c /= &v_norm_sq;
                // Update: q_row[k+1+j] -= c · v[j]
                for j in 0..m {
                    t.assign(&c);
                    t *= &v[j];
                    q_row[k + 1 + j] -= &t;
                }
            });
        }
    }

    // Extract diagonal and off-diagonal.
    let diag: Vec<Float> = (0..n).map(|i| h[i * n + i].clone()).collect();
    let off_diag: Vec<Float> = (0..(n - 1)).map(|i| h[(i + 1) * n + i].clone()).collect();

    if diag
        .iter()
        .chain(&off_diag)
        .chain(&q)
        .any(|x| !x.is_finite())
    {
        return Err(anyhow!("nonfinite stable Householder output"));
    }
    Ok((diag, off_diag, q))
}

// ===========================================================================
// Top-level: dense symmetric eigendecomposition
// ===========================================================================

#[path = "jacobi.rs"]
mod jacobi;
pub use jacobi::{
    dense_symmetric_eigendecomposition_jacobi_hp, dense_symmetric_eigenvalues_jacobi_hp,
    JacobiEigendecompositionHp, JacobiEigenvaluesHp, JACOBI_SEMANTICS,
};

/// Eigenvalues (only) of a dense symmetric matrix at HP precision.
///
/// Pipeline: scaled opposite-sign Householder tridiagonalization → tridiagonal QR.
/// Inputs are rounded once to `prec` bits. Precision must be 33..=1_000_000;
/// finite, exactly symmetric square storage is required. The returned points
/// carry no small-eigenvalue relative-accuracy or sign guarantee.
/// Returns eigenvalues sorted ascending. Eigenvectors are not computed
/// to save memory at HP scale.
pub fn dense_symmetric_eigenvalues_hp(a: &[Float], n: usize, prec: u32) -> Result<Vec<Float>> {
    let (diag, off_diag, _) = householder_tridiag_hp_impl(a, n, prec, false)?;
    tridiag_eigenvalues_hp(&diag, &off_diag, prec)
}

/// Reduce a dense symmetric HP matrix by scaled opposite-sign reflections
/// without accumulating the orthogonal basis. The returned diagonal and
/// off-diagonal are exactly those consumed by [`dense_symmetric_eigenvalues_hp`].
pub fn dense_symmetric_tridiagonal_hp(
    a: &[Float],
    n: usize,
    prec: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    let (diag, off_diag, _) = householder_tridiag_hp_impl(a, n, prec, false)?;
    Ok((diag, off_diag))
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
    Ok(
        dense_symmetric_eigenvector_for_value_detailed_hp(a, n, eigenvalue, prec, max_steps)?
            .eigenvector,
    )
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

    fn allocating_householder_trailing_update_reference(
        h: &mut [Float],
        n: usize,
        k: usize,
        v: &[Float],
        q_vec: &[Float],
    ) {
        let m = v.len();
        let row_deltas: Vec<Vec<Float>> = (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| {
                        let mut delta = v[i].clone();
                        delta *= &q_vec[j];
                        let mut delta2 = q_vec[i].clone();
                        delta2 *= &v[j];
                        delta += &delta2;
                        delta
                    })
                    .collect()
            })
            .collect();
        for (i, row) in row_deltas.iter().enumerate() {
            for (j, delta) in row.iter().enumerate() {
                h[(k + 1 + i) * n + (k + 1 + j)] -= delta;
            }
        }
    }

    #[test]
    fn in_place_householder_update_is_bit_identical_to_allocating_reference() {
        let precision = 257;
        for n in 2..=9 {
            for k in 0..n - 1 {
                let m = n - k - 1;
                let matrix = (0..n * n)
                    .map(|index| {
                        let mut value = Float::with_val(precision, index + 3);
                        value /= 11;
                        value
                    })
                    .collect::<Vec<_>>();
                let v = (0..m)
                    .map(|index| {
                        let mut value = Float::with_val(precision, index + 2);
                        value /= 7;
                        value
                    })
                    .collect::<Vec<_>>();
                let q_vec = (0..m)
                    .map(|index| {
                        let mut value = Float::with_val(precision, index + 5);
                        value /= 13;
                        value
                    })
                    .collect::<Vec<_>>();
                let mut reference = matrix.clone();
                let mut actual = matrix;
                allocating_householder_trailing_update_reference(&mut reference, n, k, &v, &q_vec);
                apply_householder_trailing_update_hp(&mut actual, n, k, &v, &q_vec);
                assert_eq!(actual, reference, "update changed at n={n}, k={k}");
            }
        }
    }

    #[test]
    fn values_only_householder_is_bit_identical_without_building_q() {
        let precision = 257;
        for n in 1..=9 {
            let matrix = (0..n * n)
                .map(|index| {
                    let row = index / n;
                    let column = index % n;
                    let low = row.min(column);
                    let high = row.max(column);
                    let mut value = Float::with_val(precision, low * 17 + high * 11 + 3);
                    value /= 19;
                    if row == column {
                        value += (n + 2) as u32;
                    }
                    value
                })
                .collect::<Vec<_>>();
            let (reference_diag, reference_off_diag, q) =
                householder_tridiag_hp_impl(&matrix, n, precision, true).unwrap();
            let (actual_diag, actual_off_diag, no_q) =
                householder_tridiag_hp_impl(&matrix, n, precision, false).unwrap();
            assert_eq!(actual_diag, reference_diag, "diagonal changed at n={n}");
            assert_eq!(
                actual_off_diag, reference_off_diag,
                "off-diagonal changed at n={n}"
            );
            assert_eq!(q.len(), n * n);
            assert!(no_q.is_empty());
        }
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

    /// The bisection before the verified-guide replay, kept as the reference.
    fn plain_bisection_reference(
        diag: &[Float],
        off_diag: &[Float],
        index: usize,
        tolerance: &Float,
        maximum_iterations: usize,
        prec: u32,
    ) -> HpTridiagonalEigenvalueEnclosure {
        let (mut lower, mut upper) = tridiag_gershgorin_bounds_hp(diag, off_diag, prec);
        let mut lower_count = 0;
        let mut upper_count = diag.len();
        let mut iterations = 0;
        loop {
            let (width, _) = Float::with_val_round(prec, &upper - &lower, rug::float::Round::Up);
            if width <= *tolerance {
                break;
            }
            assert!(iterations < maximum_iterations);
            let mut midpoint = lower.clone() / 2u32;
            midpoint += upper.clone() / 2u32;
            let count = tridiag_sturm_count_below_hp(diag, off_diag, &midpoint, prec).unwrap();
            if count <= index {
                lower = midpoint;
                lower_count = count;
            } else {
                upper = midpoint;
                upper_count = count;
            }
            iterations += 1;
        }
        HpTridiagonalEigenvalueEnclosure {
            index,
            lower,
            upper,
            lower_count,
            upper_count,
            iterations,
        }
    }

    #[test]
    fn guided_sturm_bisection_replays_the_plain_bisection_exactly() {
        let prec = 320;
        let laplacian = (vec![hp(prec, "2"); 12], vec![hp(prec, "-1"); 11]);
        // A deeply cancelled ground value: shift by a computed eigenvalue.
        let mut shifted = (
            (0..14)
                .map(|i| Float::with_val(prec, (i * 7 % 11) as u32) / 3u32)
                .collect::<Vec<_>>(),
            (0..13)
                .map(|i| Float::with_val(prec, (i * 5 % 9 + 1) as u32) / 4u32)
                .collect::<Vec<_>>(),
        );
        let ground = tridiag_eigenvalues_hp(&shifted.0, &shifted.1, prec).unwrap()[0].clone();
        for value in &mut shifted.0 {
            *value -= &ground;
        }
        // Repeated values never isolate; the guide must stay unused there.
        let repeated = (
            vec![hp(prec, "1"), hp(prec, "1"), hp(prec, "3"), hp(prec, "1")],
            vec![hp(prec, "0"), hp(prec, "0.5"), hp(prec, "0")],
        );
        for (diag, off_diag) in [&laplacian, &shifted, &repeated] {
            for bits in [40i32, 150, 300] {
                let tolerance = Float::with_val(prec, 2).pow(-bits);
                for index in 0..diag.len() {
                    let actual = tridiag_selected_eigenvalues_hp(
                        diag,
                        off_diag,
                        index,
                        index,
                        &tolerance,
                        2 * prec as usize,
                        prec,
                    )
                    .unwrap();
                    let expected = plain_bisection_reference(
                        diag,
                        off_diag,
                        index,
                        &tolerance,
                        2 * prec as usize,
                        prec,
                    );
                    assert_eq!(actual.enclosures, vec![expected.clone()], "index {index}");
                    assert_eq!(actual.sturm_evaluations, 2 + expected.iterations);
                }
            }
        }
        // The guide engages on isolated values and verifies its points.
        let (lower, upper) = tridiag_gershgorin_bounds_hp(&laplacian.0, &laplacian.1, prec);
        let tolerance = Float::with_val(prec, 2).pow(-300);
        let mut lo = lower;
        let mut hi = upper;
        while tridiag_sturm_count_below_hp(&laplacian.0, &laplacian.1, &hi, prec).unwrap() > 1 {
            let mid = Float::with_val(prec, &lo + &hi) / 2u32;
            if tridiag_sturm_count_below_hp(&laplacian.0, &laplacian.1, &mid, prec).unwrap() > 0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let (a, b) =
            isolated_eigenvalue_guide(&laplacian.0, &laplacian.1, 0, &lo, &hi, &tolerance, prec)
                .expect("isolated Laplacian ground value is guided");
        assert!(a < b && Float::with_val(prec, &b - &a) < tolerance);
        let ground_shifted = tridiag_selected_eigenvalues_hp(
            &shifted.0,
            &shifted.1,
            0,
            0,
            &tolerance,
            2 * prec as usize,
            prec,
        )
        .unwrap();
        assert!(ground_shifted.enclosures[0].lower.clone().abs() < 1e-60);
    }

    #[test]
    fn hp_selected_sturm_is_bit_identical_across_thread_counts() {
        let prec = 192;
        let dimension = 12usize;
        let diag = vec![hp(prec, "2"); dimension];
        let off_diag = vec![hp(prec, "-1"); dimension - 1];
        let tolerance = hp(prec, "1e-35");
        let run = || {
            tridiag_selected_eigenvalues_hp(&diag, &off_diag, 0, 5, &tolerance, 300, prec).unwrap()
        };
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(run);
        assert_eq!(serial, parallel);
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
        for item in &simple.items {
            if let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = item {
                // Exact rational replay of the returned vector/value, independent of point diagnostics.
                let v: Vec<_> = pair
                    .eigenvector
                    .iter()
                    .map(|x| x.to_rational().unwrap())
                    .collect();
                let lambda = pair.eigenvalue.to_rational().unwrap();
                let norm_sq = v
                    .iter()
                    .fold(rug::Rational::from(0), |sum, x| sum + x.clone() * x);
                let residual_sq = (0..v.len()).fold(rug::Rational::from(0), |sum, i| {
                    let mut action: rug::Rational = v[i].clone() * 2 - lambda.clone() * &v[i];
                    if i > 0 {
                        action -= &v[i - 1];
                    }
                    if i + 1 < v.len() {
                        action -= &v[i + 1];
                    }
                    sum + action.clone() * action
                });
                let bound = pair.residual_norm.to_rational().unwrap();
                assert!(residual_sq <= bound.clone() * bound * norm_sq);
            }
        }

        assert!(
            simple.items.iter().all(|item| matches!(
                item,
                HpSelectedTridiagonalItem::SimpleEigenpair(pair)
                    if pair.residual_norm < hp(prec, "1e-40")
                        && pair.diagnostics.absolute_residual < hp(prec, "1e-40")
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
    fn householder_q_and_tridiagonal_satisfy_similarity_relation() {
        let prec = 256;
        let n = 5;
        let raw = [
            "4", "1", "2", "0", "1", "1", "3", "1", "1", "0", "2", "1", "5", "2", "1", "0", "1",
            "2", "6", "3", "1", "0", "1", "3", "7",
        ];
        let a: Vec<Float> = raw.iter().map(|value| hp(prec, value)).collect();
        let (diagonal, off_diagonal, q) = householder_tridiag_hp(&a, n, prec).unwrap();
        let tolerance = hp(prec, "1e-60");

        // Q^T A Q = T is equivalent to A Q = Q T. Check every entry so a
        // sign mismatch in a stored off-diagonal cannot hide behind the
        // sign-invariance of the tridiagonal eigenvalues.
        for row in 0..n {
            for column in 0..n {
                let mut aq = hp(prec, "0");
                for inner in 0..n {
                    let mut term = a[row * n + inner].clone();
                    term *= &q[inner * n + column];
                    aq += term;
                }
                let mut qt = q[row * n + column].clone();
                qt *= &diagonal[column];
                if column > 0 {
                    let mut term = q[row * n + column - 1].clone();
                    term *= &off_diagonal[column - 1];
                    qt += term;
                }
                if column + 1 < n {
                    let mut term = q[row * n + column + 1].clone();
                    term *= &off_diagonal[column];
                    qt += term;
                }
                aq -= qt;
                assert!(
                    aq.abs() < tolerance,
                    "A Q = Q T failed at row {row}, column {column}"
                );
            }
        }
    }

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

        // Separate rounded left/right products differ by roundoff. Verify
        // that discrepancy before storing an exactly symmetric test matrix.
        // The 2^-240 bound is far below the 1e-50 spectral tolerance below.
        let symmetry_tolerance = Float::with_val(prec, 1) >> 240_i32;
        for i in 0..n {
            for j in (i + 1)..n {
                let discrepancy = Float::with_val(prec, &a[i * n + j] - &a[j * n + i]);
                assert!(discrepancy.abs() < symmetry_tolerance);
                let mut average = Float::with_val(prec, &a[i * n + j] + &a[j * n + i]);
                average /= 2;
                a[i * n + j] = average.clone();
                a[j * n + i] = average;
            }
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

    #[test]
    fn tridiag_qr_default_uses_slow_but_accurate_hard_limit() {
        let options = TridiagQrOptions::default();
        assert_eq!(
            options.max_iterations_per_eigenvalue,
            DEFAULT_TRIDIAG_QR_MAX_ITERATIONS
        );
        assert_eq!(DEFAULT_TRIDIAG_QR_MAX_ITERATIONS, 2_000);
        assert!(options.max_iterations_per_eigenvalue > TRIDIAG_QR_SLOW_SWEEP_WARNING);
    }

    #[test]
    fn larger_qr_budget_is_bit_identical_when_legacy_budget_already_converges() {
        let prec = 256;
        let n = 20;
        let (diag, off_diag, _) = strang_tridiag(prec, n);
        let legacy_budget = tridiag_eigenvalues_hp_with_options(
            &diag,
            &off_diag,
            prec,
            TridiagQrOptions {
                max_iterations_per_eigenvalue: 100,
            },
        )
        .unwrap();
        let accurate_default = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
        assert_eq!(accurate_default, legacy_budget);
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
    fn staged_tridiagonal_qr_is_bit_identical_to_dense_qr() {
        let prec = 256;
        let matrix = vec![
            hp(prec, "4"),
            hp(prec, "1.25"),
            hp(prec, "-0.5"),
            hp(prec, "1.25"),
            hp(prec, "3"),
            hp(prec, "0.75"),
            hp(prec, "-0.5"),
            hp(prec, "0.75"),
            hp(prec, "2"),
        ];
        let direct = dense_symmetric_eigenvalues_hp(&matrix, 3, prec).unwrap();
        let (diagonal, off_diagonal) = dense_symmetric_tridiagonal_hp(&matrix, 3, prec).unwrap();
        let staged = tridiag_eigenvalues_hp(&diagonal, &off_diagonal, prec).unwrap();
        assert_eq!(staged, direct);
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
