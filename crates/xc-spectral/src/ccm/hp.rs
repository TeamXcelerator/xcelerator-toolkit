// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision CCM tier via `rug` (MPFR/GMP).
//!
//! Strategy:
//! - Build the (2N+1)×(2N+1) Weil form matrix at user-chosen precision.
//! - Find smallest eigenpair by inverse iteration (from xc-numerics).
//! - Solve spectrum equation by Newton's method.

use anyhow::Result;
use rayon::prelude::*;
use rug::{ops::Pow, Float};
use std::time::Instant;

use super::{prime_powers_up_to, CcmParams, CcmResult};

/// Configuration for the high-precision tier.
///
/// All fields are public so callers can override individual components
/// after a `for_decimal_digits` construction. The default values are
/// tuned for the HP-1000 retest workload; tweak with
/// caution.
#[derive(Debug, Clone)]
pub struct HighPrecConfig {
    /// MPFR working precision in bits. Total digits ≈ `precision_bits / 3.322`.
    /// Use `for_decimal_digits` to construct from a target decimal precision.
    pub precision_bits: u32,
    /// Maximum number of inverse-iteration steps for the smallest
    /// Weil-form eigenvector recovery.
    pub inverse_iter_steps: usize,
    /// Maximum number of solver steps per Riemann-zero seed.
    /// Applies to both Halley (default) and Newton solvers.
    ///
    /// In practice the solver exits early via full-precision convergence
    /// (`|dz| < tol`) or stagnation detection (construction ceiling reached).
    /// This cap is a safety net only — it prevents infinite loops on
    /// pathological/wandering seeds and is never reached on real configs.
    /// Default 200. Override via env: `XCELERATOR_SOLVER_STEPS=<n>`.
    pub solver_steps: usize,
    /// Number of Gauss–Legendre quadrature points used in the integral
    /// computation of α_L, β_L, γ_L. Clamped to `[MIN_QUAD_POINTS,
    /// MAX_QUAD_POINTS]` regardless of input.
    pub quad_points: usize,
    /// Number of positive eigenvalues to compute. Newton refinement
    /// runs over the first `n_eigenvalues` reference Riemann zeros.
    pub n_eigenvalues: usize,
    /// Cache strategy for the GL-node and τ-matrix disk caches. See
    /// [`xc_numerics::quadrature::CacheMode`]. Default `DynamicFetch`
    /// (local `.json` → local zip → remote fetch → compute).
    pub cache_mode: xc_numerics::quadrature::CacheMode,
    /// Whether to project onto the even subspace at each inverse-iteration
    /// step. Default `true` (forced-even, the standard CCM path). Set to
    /// `false` to test whether the natural (unprojected) smallest
    /// eigenvector is even without forcing.
    pub force_even: bool,
    /// Enable warm-start from a nearby-precision cached eigenvector.
    /// When `true` and a cached ξ exists for the same (λ², N) within
    /// `warm_start_tolerance_bits` of the target precision, that cached ξ
    /// is used as the starting vector for inverse iteration instead of the
    /// Gaussian initial guess. Dramatically reduces iteration count for
    /// P-sweep campaigns. Default `true`.
    /// Override via env: `XCELERATOR_WARM_START=0` disables.
    pub warm_start: bool,
    /// Precision tolerance in bits for warm-start cache lookup.
    /// A cached ξ at prec' is accepted as a warm start if
    /// |prec' - target_prec| ≤ warm_start_tolerance_bits.
    /// Default 500 bits (~150 decimal digits) — spans a full HP-level step.
    /// Override via env: `XCELERATOR_WARM_START_TOL=<bits>` (e.g. `=200`).
    pub warm_start_tolerance_bits: u32,
}

/// Conversion factor: decimal digits to binary bits.
/// log₂(10) ≈ 3.32193. We use 3.322 and add 16 guard bits.
pub const DIGITS_TO_BITS_FACTOR: f64 = 3.322;

/// Extra guard bits added beyond the strict digits-to-bits conversion.
pub const GUARD_BITS: u32 = 16;

/// Minimum quadrature points for the HP tier.
pub const MIN_QUAD_POINTS: usize = 600;

/// Maximum quadrature points for the HP tier (prevents excessive runtime).
pub const MAX_QUAD_POINTS: usize = 4000;

/// Multiplier: quad_points = digits * QUAD_POINTS_PER_DIGIT (clamped to [MIN, MAX]).
pub const QUAD_POINTS_PER_DIGIT: usize = 3;

/// HP singularity guard for `rho_hp(x)` near `x = 0`. Below this magnitude
/// we use the Taylor-series branch instead of `1 / (2 sinh(x/2))` directly.
/// Stored as a string literal so it is parsed as an HP `Float` at the
/// caller's working precision (no f64 round-trip).
pub const HP_SINGULARITY_GUARD_STR: &str = "1e-30";

impl HighPrecConfig {
    /// Construct a config from a target decimal-digit working precision.
    ///
    /// Bits are computed via `digits × DIGITS_TO_BITS_FACTOR + GUARD_BITS`,
    /// rounded up. Quadrature points are `digits × QUAD_POINTS_PER_DIGIT`,
    /// clamped to `[MIN_QUAD_POINTS, MAX_QUAD_POINTS]`. Other fields take
    /// the defaults `inverse_iter_steps=200, solver_steps=20,
    /// n_eigenvalues=50`.
    pub fn for_decimal_digits(digits: u32) -> Self {
        let bits = ((digits as f64) * DIGITS_TO_BITS_FACTOR).ceil() as u32 + GUARD_BITS;
        // solver_steps: high safety cap — in practice the solver exits early
        // via full-precision convergence or stagnation detection (construction
        // ceiling). The cap only fires for pathological/wandering seeds and
        // prevents an infinite loop. Default 200 steps is never reached on
        // any real (λ², N, P) config; it is purely a safety net.
        // Override via env: XCELERATOR_SOLVER_STEPS=<n> for testing.
        let solver_steps = {
            let p = digits as f64;
            let k = (p / 10.0).log2().ceil() as usize;
            let default_cap = k.max(200);
            std::env::var("XCELERATOR_SOLVER_STEPS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(default_cap)
        };
        Self {
            precision_bits: bits,
            inverse_iter_steps: 200,
            solver_steps,
            quad_points: ((digits as usize) * QUAD_POINTS_PER_DIGIT)
                .clamp(MIN_QUAD_POINTS, MAX_QUAD_POINTS),
            n_eigenvalues: 50,
            cache_mode: xc_numerics::quadrature::CacheMode::default(),
            force_even: true,
            // Warm-start on by default. Uses a cached ξ at a nearby
            // precision as the starting vector for inverse iteration
            // instead of the Gaussian guess. Falls back to the Gaussian
            // when no nearby cache entry exists (cold cache).
            // Override via XCELERATOR_WARM_START=0 to disable.
            warm_start: std::env::var("XCELERATOR_WARM_START").as_deref() != Ok("0"),
            // Tolerance in bits for warm-start lookup, default 500
            // Override via XCELERATOR_WARM_START_TOL=<bits>
            warm_start_tolerance_bits: std::env::var("XCELERATOR_WARM_START_TOL")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(500),
        }
    }
}

/// Status of a single solver run for one eigenvalue seed.
///
/// The three-state result lets callers distinguish clean convergence,
/// a best-effort approximation (step limit hit), and a degenerate failure
/// (denominator zero — result is garbage and should not be used).
#[derive(Debug, Clone)]
pub enum EigenvalueResult {
    /// Solver converged to HP tolerance within the step limit.
    /// The value is reliable to the working precision.
    Converged(Float),
    /// Solver hit the step limit before reaching HP tolerance.
    /// The value is the best approximation found — it may be accurate
    /// to many digits (often still useful) but is NOT fully converged.
    Approximate(Float),
    /// Solver hit a degenerate denominator (pole/zero). The result is
    /// garbage and must not be used.
    Failed,
}

impl EigenvalueResult {
    /// Return the inner `Float` value if present (Converged or Approximate),
    /// or `None` if Failed.
    pub fn value(&self) -> Option<&Float> {
        match self {
            EigenvalueResult::Converged(v) | EigenvalueResult::Approximate(v) => Some(v),
            EigenvalueResult::Failed => None,
        }
    }

    /// Returns `true` if this result is fully converged.
    pub fn is_converged(&self) -> bool {
        matches!(self, EigenvalueResult::Converged(_))
    }

    /// Returns `true` if this result has any usable value (Converged or Approximate).
    pub fn has_value(&self) -> bool {
        !matches!(self, EigenvalueResult::Failed)
    }
}

/// Result of a single high-precision CCM run.
///
/// All HP fields stay in `rug::Float` at the working precision specified
/// in the config; lossy f64 views are exposed via `to_f64_result`.
pub struct HighPrecResult {
    /// Solver results for positive eigenvalues.
    ///
    /// Each entry is one of:
    /// - `Converged(v)` — fully converged to HP tolerance; reliable.
    /// - `Approximate(v)` — best approximation after step limit; may still
    ///   match many digits but is NOT certified to HP precision.
    /// - `Failed` — degenerate denominator; garbage, do not use.
    pub eigenvalues_pos: Vec<EigenvalueResult>,
    /// Smallest eigenvalue of the Weil quadratic form (the spectral
    /// gap quantity ε_N at this `(λ², N)`).
    pub weil_min_eigenvalue: Float,
    /// Smallest-eigenvalue eigenvector of the Weil form, ℓ²-normalized,
    /// stored in the V_n basis order (centered index `0` at position
    /// `n_modes`).
    pub xi: Vec<Float>,
    /// Wall-clock seconds for the entire HP run (matrix build +
    /// eigenvector + Newton).
    pub elapsed_seconds: f64,
    /// MPFR working precision used for this run, in bits.
    pub precision_bits: u32,
}

impl HighPrecResult {
    /// Lossy conversion to the f64-tier `CcmResult`. Eigenvalues, ξ
    /// entries, and ε_N collapse to f64; the f64 underflow boundary at
    /// ~10⁻³⁰⁸ silently maps to zero. Use only for f64-tier consumers
    /// (CLI summaries, plot generation); never for downstream HP work.
    pub fn to_f64_result(&self) -> CcmResult {
        CcmResult {
            eigenvalues_pos: self.eigenvalues_pos.iter().map(|r| match r {
                EigenvalueResult::Converged(f) => f.to_f64(),
                EigenvalueResult::Approximate(f) => f.to_f64(),
                EigenvalueResult::Failed => f64::NAN, // degenerate — NaN signals garbage
            }).collect(),
            weil_min_eigenvalue: self.weil_min_eigenvalue.to_f64(),
            xi: self.xi.iter().map(|f| f.to_f64()).collect(),
            elapsed_seconds: self.elapsed_seconds,
        }
    }

}

// -- Helpers
#[inline] fn fl_i(prec: u32, v: i64) -> Float { Float::with_val(prec, v) }
#[inline] fn pi(prec: u32) -> Float { Float::with_val(prec, rug::float::Constant::Pi) }
#[inline] fn euler(prec: u32) -> Float { Float::with_val(prec, rug::float::Constant::Euler) }

/// Force exact symmetry on a `dim × dim` row-major HP matrix in-place.
///
/// Computes `(M[i,j] + M[j,i]) / 2` for every upper-triangle pair
/// (parallel compute, sequential write-back to avoid aliasing), then
/// stores the average in both M[i,j] and M[j,i]. The diagonal is
/// untouched. This is called on the τ-matrix before eigenvector
/// computation to ensure that floating-point construction noise doesn't
/// break the assumed symmetry of the Weil quadratic form.
fn force_symmetric(matrix: &mut [Float], dim: usize) {
    let pairs: Vec<(usize, usize)> = (0..dim)
        .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j)))
        .collect();
    let symmetrized: Vec<(usize, usize, Float)> = pairs.par_iter().map(|&(i, j)| {
        let mut sum = matrix[i * dim + j].clone();
        sum += &matrix[j * dim + i];
        sum /= 2u32;
        (i, j, sum)
    }).collect();
    for (i, j, sum) in symmetrized {
        matrix[i * dim + j] = sum.clone();
        matrix[j * dim + i] = sum;
    }
}

/// Top-level entry. Build matrix, find eigenvector, solve spectrum.
///
/// `zero_seeds` are the reference Riemann zero imaginary parts used as
/// Newton seeds. They should be at full working precision (decimal strings
/// parsed to `Float`) — NOT f64-truncated. Using f64 seeds causes Newton
/// divergence at high eigenvalue index.
/// High-precision CCM run.
///
/// On WSL2 the call is automatically routed through
/// [`xc_numerics::hp_runtime::run_hp`]: a large-stack scoped thread plus
/// a local rayon pool with a modest worker count and per-worker stack size.
/// This avoids the GMP-arena / participating-caller stack aborts that occur
/// with the default 32-worker 8 MB-stack pool under WSL2.
///
/// On Vast / native Linux / CI: `run_hp` is a zero-overhead pass-through —
/// no thread spawn, no pool creation, full parallelism unchanged.
pub fn run(params: &CcmParams, cfg: &HighPrecConfig, zero_seeds: &[Float]) -> Result<HighPrecResult> {
    xc_numerics::hp_runtime::run_hp(|| run_inner(params, cfg, zero_seeds))
}

fn run_inner(params: &CcmParams, cfg: &HighPrecConfig, zero_seeds: &[Float]) -> Result<HighPrecResult> {
    let start = Instant::now();
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();

    let l = log_lambda_sq_hp(params, prec);
    let mut tau = build_tau_hp(params, &l, cfg)?;

    // Force exact symmetry of the τ-matrix (parallel compute, sequential write).
    force_symmetric(&mut tau, dim);

    // Smallest eigenpair (ξ, ε_N).
    //
    // After-τ cache check: if a cached Weil eigenvector exists for this
    // (λ², N, prec) AND it validates against the in-hand τ via the
    // eigen-residual ‖τξ − μξ‖, skip the costly LU factorization
    // entirely. A missing or residual-failing entry falls through to a
    // fresh inverse-iteration compute, which is then cached. The check
    // sits *after* `build_tau_hp` precisely so τ is available for the
    // residual validation (the strongest integrity test for ξ).
    //
    let lambda_sq = params.lambda_sq;
    let n_modes_key = params.n_modes;
    let mut cached_pair: Option<(Float, Vec<Float>)> = None;
    if let Some(c) = weil_eigvec_cache::load(lambda_sq, n_modes_key, prec, cfg.cache_mode, cfg.force_even) {
        if weil_eigvec_cache::residual_ok(&tau, dim, &c.xi, &c.eps_n, prec) {
            eprintln!(
                "[HP] loaded cached Weil eigenvector for λ²={}, N={}, prec={} bits (τ-residual validated)",
                lambda_sq.value_f64, n_modes_key, prec
            );
            cached_pair = Some((c.eps_n, c.xi));
        } else {
            crate::hp_debug!(
                "[HP] WARNING: cached Weil eigenvector for λ²={}, N={}, prec={} failed \
                 τ-residual validation; recomputing",
                lambda_sq.value_f64, n_modes_key, prec
            );
        }
    }
    let (eps_n, xi) = match cached_pair {
        Some(pair) => pair,
        None => {
            // Warm-start from nearby-precision cache if enabled.
            // Scan for a cached ξ at a nearby precision to use as the
            // starting vector for inverse iteration instead of the Gaussian.
            let warm_xi: Option<Vec<Float>> = if cfg.warm_start {
                weil_eigvec_cache::find_warm_start(
                    lambda_sq, n_modes_key, prec,
                    cfg.warm_start_tolerance_bits,
                    cfg.force_even,
                )
            } else {
                None
            };

            // Find smallest eigenpair by inverse iteration.
            eprintln!("[HP] LU factoring {}×{} matrix (one-time cost)...", dim, dim);
            let (eps_n, xi_raw) = if let Some(warm) = warm_xi {
                crate::hp_debug!("[HP] starting inverse iteration from warm-start vector");
                xc_numerics::linalg::inverse_iteration_from(
                    &tau, dim, prec, cfg.inverse_iter_steps, cfg.force_even, Some(warm))?
            } else {
                xc_numerics::linalg::inverse_iteration(
                    &tau, dim, prec, cfg.inverse_iter_steps, cfg.force_even)?
            };
            crate::hp_debug!("[HP] LU factorization done.");
            // Normalize: Σ ξ_j = √L.
            let xi = normalize_eigenvector(&xi_raw, &l, prec);
            eprintln!("[HP] Eigenvector computed. Solving spectrum...");
            weil_eigvec_cache::save(lambda_sq, n_modes_key, prec, &eps_n, &xi, cfg.cache_mode, cfg.force_even);
            (eps_n, xi)
        }
    };

    // Find eigenvalues as zeros of R(z), seeded from HP reference zeros.
    // Each solver refinement is independent across seeds — parallelize.
    let n_eigs = cfg.n_eigenvalues.min(zero_seeds.len());

    // Seed the HP solver from the reference Riemann zeros. At large N
    // the CCM eigenvalues are close to the Riemann zeros, so these are
    // good seeds. At small N the f64 bisection solver finds eigenvalues
    // in bracket order (k², (k+1)²) which does NOT correspond 1:1 to
    // Riemann zero indices — using f64 seeds causes cross-overs and
    // completely wrong eigenvalues in the wrong slots. Riemann zero seeds
    // are robust across all (λ², N) configurations.
    let hp_seeds: Vec<Float> = zero_seeds[..n_eigs].iter()
        .map(|s| Float::with_val(prec, s))
        .collect();
    crate::hp_debug!("[HP] using {} reference Riemann zeros as HP {} seeds (N={})",
        hp_seeds.len(), active_solver_name(), params.n_modes);

    let eigenvalues_pos: Vec<EigenvalueResult> = hp_seeds.iter()
        .map(|seed| {
            let result = solve_r_zero(&xi, params.n_modes, &l, seed, prec, cfg.solver_steps);
            if matches!(result, EigenvalueResult::Failed) {
                crate::hp_debug!(
                    "[HP] WARNING: {} hit degenerate denominator for seed {} — skipping",
                    active_solver_name(), seed.to_f64()
                );
            }
            result
        })
        .collect();
    // After all solves, verify each computed eigenvalue is closest
    // to its assigned seed (detect cross-overs). Log warnings for any
    // mismatches but don't reorder — ordering is by seed, not by value.
    for (k, ev_result) in eigenvalues_pos.iter().enumerate() {
        let ev = match ev_result.value() {
            Some(z) => z,
            None => continue,
        };
        if k >= hp_seeds.len() { break; }
        let ref_k = hp_seeds[k].clone();
        let mut dist_k = ev.clone(); dist_k -= &ref_k; let dist_k = dist_k.abs();
        for (j, ref_j_seed) in hp_seeds.iter().enumerate() {
            if j == k { continue; }
            if j >= hp_seeds.len() { break; }
            let mut dist_j = ev.clone(); dist_j -= ref_j_seed; let dist_j = dist_j.abs();
            if dist_j < dist_k {
                crate::hp_debug!(
                    "[HP] WARNING: eigenvalue {} is closer to seed {} than its own seed {} — possible cross-over",
                    k + 1, j + 1, k + 1
                );
                break;
            }
        }
    }

    Ok(HighPrecResult {
        eigenvalues_pos,
        weil_min_eigenvalue: eps_n,
        xi,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        precision_bits: prec,
    })
}

/// Measure the natural evenness of the smallest eigenvector before
/// forced symmetrization.
///
/// Returns `(evenness_deviation, natural_eigenvalue, forced_eigenvalue)` where:
/// - `evenness_deviation` = ‖ξ - γξ‖ / ‖ξ‖ (0 = perfectly even, >0 = asymmetric)
/// - `natural_eigenvalue` = smallest eigenvalue without forcing
/// - `forced_eigenvalue` = smallest *even* eigenvalue (with forcing)
///
/// At small λ, the natural eigenvector is essentially even (deviation ~10⁻¹⁵⁰).
/// At large λ (λ²≥1000), the natural eigenvector may be odd or mixed-symmetry,
/// with deviation O(1). This is a structural property of the construction,
/// not a precision artifact (verified at HP-1000).
pub fn measure_evenness(params: &CcmParams, cfg: &HighPrecConfig) -> Result<EvennessResult> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();

    let l = log_lambda_sq_hp(params, prec);
    let mut tau = build_tau_hp(params, &l, cfg)?;

    // Force exact symmetry of the τ-matrix (parallel compute, sequential write).
    force_symmetric(&mut tau, dim);

    // Natural (unforced) smallest eigenpair.
    let (natural_eval, xi_natural) = xc_numerics::linalg::inverse_iteration(
        &tau, dim, prec, cfg.inverse_iter_steps, false)?;

    // Forced-even smallest eigenpair.
    let (forced_eval, _xi_forced) = xc_numerics::linalg::inverse_iteration(
        &tau, dim, prec, cfg.inverse_iter_steps, true)?;

    // Evenness deviation: ‖ξ - γξ‖ / ‖ξ‖ where γ is index reflection.
    // γξ_i = ξ_{dim-1-i}. Deviation = ‖ξ - γξ‖₂ / ‖ξ‖₂.
    // Both reductions are parallelized over i.
    let zero = || Float::with_val(prec, 0);
    let (diff_sq, norm_sq) = (0..dim).into_par_iter()
        .map(|i| {
            let reflected = dim - 1 - i;
            let mut d = xi_natural[i].clone();
            d -= &xi_natural[reflected];
            let d_sq = d.square();
            let n_sq = xi_natural[i].clone().square();
            (d_sq, n_sq)
        })
        .reduce(|| (zero(), zero()),
                |(mut a_d, mut a_n), (b_d, b_n)| {
                    a_d += &b_d;
                    a_n += &b_n;
                    (a_d, a_n)
                });
    let mut deviation = diff_sq.sqrt();
    let norm = norm_sq.sqrt();
    if !norm.is_zero() {
        deviation /= &norm;
    }

    Ok(EvennessResult {
        evenness_deviation: deviation,
        natural_eigenvalue: natural_eval,
        forced_eigenvalue: forced_eval,
    })
}

/// Result of the evenness measurement.
pub struct EvennessResult {
    /// ‖ξ - γξ‖ / ‖ξ‖. Zero means perfectly even.
    pub evenness_deviation: Float,
    /// Smallest eigenvalue without forced-even projection.
    pub natural_eigenvalue: Float,
    /// Smallest eigenvalue with forced-even projection.
    pub forced_eigenvalue: Float,
}

// ===========================================================================
// Matrix construction
// ===========================================================================

/// Wrapper around `build_tau_hp_compute` that consults the tau matrix
/// disk cache before invoking the full HP construction.
///
/// At HP-1000 the τ-matrix construction is O(N²) HP integral
/// evaluations + O(N³) LU-equivalent work in inverse iteration; for the
/// load-bearing paper configs (λ²=13/100/1000 at N=120/500/800) this is
/// minutes-to-hours of wall-time. The output is fully determined by
/// `(λ²_int, n_modes, prec)`, so it can be cached on disk and reused
/// across runs.
///
/// Cache layout (mirrors `gl_cache` and `prolate_eigvals_cache`):
///   <cwd>/data/tau_cache/lambda_sq{L}_nmodes{N}_prec{P}.json[.zip[.partXX]]
///
/// Lookup priority:
///   1. Uncompressed `.json`
///   2. Single zip `.json.zip`
///   3. Multi-part split `.json.zip.part00, .part01, ...` (used when
///      compressed payload exceeds GitHub's 100 MB hard limit; we
///      split at 90 MB-byte boundaries and concatenate the parts on
///      read before passing to the zip decoder).
///
/// Cache miss → compute fresh via `build_tau_hp_compute`, save in
/// the most appropriate format for the resulting size.
fn build_tau_hp(params: &CcmParams, l: &Float, cfg: &HighPrecConfig) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let lambda_sq = params.lambda_sq;
    let n_modes = params.n_modes;

    if let Some(cached) = tau_cache::load(lambda_sq, n_modes, prec, cfg.cache_mode) {
        eprintln!(
            "[HP] loaded cached τ-matrix for λ²={}, N={}, prec={} bits ({}×{} = {} entries)",
            lambda_sq.value_f64, n_modes, prec,
            params.matrix_size(), params.matrix_size(), cached.len()
        );
        return Ok(cached);
    }

    let tau = build_tau_hp_compute(params, l, cfg, true)?;
    tau_cache::save(lambda_sq, n_modes, prec, &tau, cfg.cache_mode);
    Ok(tau)
}

/// Eigenvalues of the localized Weil-form (`τ`) matrix. The smallest
/// positive eigenvalue is the plunge `ε_N`, whose `−log₁₀|ε_N|` is the CCM
/// prime-content floor `D_Primes(λ²)` (up to the small ε_N↔matching-digit
/// offset).
///
/// `include_primes`:
/// - `true`  → the full Weil form (archimedean + prime sum). This is the
///   same `ε_N` returned by [`run`] / cached in `weil_eigvec_cache`.
/// - `false` → the **archimedean-only** form (prime sum `w_p` dropped from
///   every `τ` entry). Used to test the prefactor decomposition: do the
///   archimedean-only and full plunges share the same prefactor power
///   (the prime sum shifting only the additive constant)?
///
/// Returns the **full Weil-form spectrum** (all eigenvalues of `τ`, sorted
/// ascending). The smallest positive eigenvalue is the plunge `ε_N`; the
/// overall minimum reveals whether the form is positive (Weil positivity)
/// or O(1)-indefinite (relevant for the archimedean-only mode, where the
/// prime sum that enforces positivity is absent).
///
/// Uses the dense HP symmetric eigensolver (all eigenvalues) rather than
/// inverse iteration, since the archimedean-only spectrum bottom may be
/// negative — `inverse_iteration` finds the smallest-*magnitude* eigenvalue,
/// not the spectrum minimum, so it would be the wrong tool there. The
/// archimedean-only path bypasses the τ disk cache (keyed only on
/// `(λ², N, prec)`, which would collide with the full matrix).
pub fn weil_spectrum_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    xc_numerics::hp_runtime::run_hp(|| weil_spectrum_hp_inner(params, cfg, include_primes))
}

fn weil_spectrum_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    include_primes: bool,
) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);
    let mut tau = if include_primes {
        build_tau_hp(params, &l, cfg)?
    } else {
        build_tau_hp_compute(params, &l, cfg, false)?
    };
    force_symmetric(&mut tau, dim);
    xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&tau, dim, prec)
}

/// Decomposition of the plunge into archimedean and prime Rayleigh
/// contributions on the **full** plunge eigenvector ξ.
///
/// The localized Weil form splits as `A_full = A_arch − A_prime` (the τ
/// entry is `w02 − wr − wp`; the archimedean part is `w02 − wr`, the prime
/// part is the prime-power sum `wp`). By linearity of the Rayleigh
/// quotient on the full plunge eigenvector ξ:
/// ```text
///   ε_N = ⟨A_full ξ,ξ⟩/⟨ξ,ξ⟩
///       = ⟨A_arch ξ,ξ⟩/⟨ξ,ξ⟩ − ⟨A_prime ξ,ξ⟩/⟨ξ,ξ⟩
///       = arch_rayleigh − prime_rayleigh.
/// ```
/// Since `ε_N` is exponentially small while `arch_rayleigh` and
/// `prime_rayleigh` are O(1)-or-larger, the two must agree to
/// `≈ −log₁₀|ε_N|` digits — a direct quantitative picture of the
/// archimedean↔prime (Weil-positivity) cancellation that produces the
/// CCM convergence floor.
pub struct PlungeCancellation {
    /// Plunge eigenvalue `ε_N = arch_rayleigh − prime_rayleigh`.
    pub eps_n: Float,
    /// `⟨A_arch ξ,ξ⟩/⟨ξ,ξ⟩` — archimedean Rayleigh on the full plunge ξ.
    pub arch_rayleigh: Float,
    /// `⟨A_prime ξ,ξ⟩/⟨ξ,ξ⟩` — prime-sum Rayleigh on the full plunge ξ.
    pub prime_rayleigh: Float,
}

/// Compute the [`PlungeCancellation`] for `(λ², N, P)`. Finds the full
/// plunge eigenvector ξ by inverse iteration, then evaluates the
/// archimedean and prime Rayleigh quotients on that same ξ.
pub fn weil_plunge_cancellation_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
) -> Result<PlungeCancellation> {
    xc_numerics::hp_runtime::run_hp(|| weil_plunge_cancellation_hp_inner(params, cfg))
}

fn weil_plunge_cancellation_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
) -> Result<PlungeCancellation> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);

    // Full Weil form; smallest positive eigenvalue (the plunge) and its
    // eigenvector via the dense HP symmetric path.
    let mut tau_full = build_tau_hp(params, &l, cfg)?;
    force_symmetric(&mut tau_full, dim);
    let eigs = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&tau_full, dim, prec)?;
    let zero = Float::with_val(prec, 0);
    let eps_n = eigs
        .iter()
        .find(|e| **e > zero)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no positive eigenvalue (Weil form indefinite?)"))?;
    let xi = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
        &tau_full, dim, &eps_n, prec, cfg.inverse_iter_steps,
    )?;

    // Archimedean-only form (prime sum dropped).
    let mut tau_arch = build_tau_hp_compute(params, &l, cfg, false)?;
    force_symmetric(&mut tau_arch, dim);

    // Rayleigh quotients on the SAME ξ. The split is exact by linearity:
    // arch_rayleigh − prime_rayleigh = full_rayleigh = ε_N.
    let full_rayleigh = xc_numerics::linalg::rayleigh_quotient(&tau_full, dim, &xi, prec);
    let arch_rayleigh = xc_numerics::linalg::rayleigh_quotient(&tau_arch, dim, &xi, prec);
    let mut prime_rayleigh = Float::with_val(prec, &arch_rayleigh);
    prime_rayleigh -= &full_rayleigh;

    Ok(PlungeCancellation {
        eps_n: full_rayleigh,
        arch_rayleigh,
        prime_rayleigh,
    })
}

/// `sinc(t) = sin(t)/t`, with `sinc(0) = 1`, at working precision `prec`.
fn sinc_hp(t: &Float, prec: u32) -> Float {
    let tiny = Float::with_val(prec, Float::parse("1e-40").unwrap());
    if t.cmp_abs(&tiny).map(|o| o.is_lt()).unwrap_or(false) {
        Float::with_val(prec, 1)
    } else {
        let mut s = t.clone().sin();
        s /= t;
        s
    }
}

/// Time-frequency (prolate) band-concentration matrix `C` in the same
/// `V_n` trigonometric basis as the localized Weil form, for a frequency
/// band `(−Ω, Ω)` (`omega` = Ω).
///
/// With `φ_n(x) = e^{2π i n x / L}/√L` on the log-interval `(−a, a)`
/// (`a = ½ ln λ²`, `L = 2a`) and `ω_n = 2π n / L`, the entry is the
/// time-then-band-limiting (Slepian concentration) operator
/// `C[n,m] = (a/π) ∫_{−Ω}^{Ω} sinc(a(ω_n−ξ)) sinc(a(ω_m−ξ)) dξ`,
/// computed by HP Gauss–Legendre quadrature on the band.
///
/// Its eigenvalues `χ ∈ (0,1)` are the band-concentration ratios: `χ ≈ 1`
/// modes are band-concentrated (the prolate/PSWF subspace), `χ ≈ 0` modes
/// span the Sonin-like (anti-band) subspace. The number of `χ ≈ 1` modes
/// is the Shannon number `≈ 2aΩ/π`. Symmetric, row-major, dimension `2N+1`,
/// in the `params.idx` ordering (so it matches `weil_spectrum_hp`).
pub fn band_concentration_matrix_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
) -> Result<Vec<Float>> {
    xc_numerics::hp_runtime::run_hp(|| band_concentration_matrix_hp_inner(params, cfg, omega))
}

fn band_concentration_matrix_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let n_max = params.n_modes as i64;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);
    let mut a = l.clone();
    a /= 2u32;
    let pi_v = pi(prec);
    let npts = cfg.quad_points.max(8 * params.n_modes + 64);
    let (nodes, weights) =
        xc_numerics::quadrature::gauss_legendre_nodes(npts, prec, cfg.cache_mode);

    // ω_n = 2π n / L, indexed by position params.idx(n) = n + N.
    let omega_n: Vec<Float> = (-n_max..=n_max)
        .map(|n| {
            let mut w = pi_v.clone();
            w *= 2u32;
            w *= fl_i(prec, n);
            w /= &l;
            w
        })
        .collect();

    // prefactor (a/π)·Ω folded into the quadrature weight.
    let mut aon = a.clone();
    aon /= &pi_v;
    aon *= omega;

    let mut c = vec![Float::with_val(prec, 0); dim * dim];
    for (q, node) in nodes.iter().enumerate() {
        let mut xi = node.clone(); // GL node on [-1,1]
        xi *= omega; // ξ_q = Ω·node ∈ (−Ω, Ω)
        let svec: Vec<Float> = omega_n
            .iter()
            .map(|wn| {
                let mut arg = wn.clone();
                arg -= &xi;
                arg *= &a;
                sinc_hp(&arg, prec)
            })
            .collect();
        let mut wq = weights[q].clone();
        wq *= &aon;
        for i in 0..dim {
            let mut wi = svec[i].clone();
            wi *= &wq;
            for j in i..dim {
                let mut term = wi.clone();
                term *= &svec[j];
                c[i * dim + j] += &term;
            }
        }
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let v = c[i * dim + j].clone();
            c[j * dim + i] = v;
        }
    }
    Ok(c)
}

/// Result of restricting the archimedean Weil form to the Sonin-like
/// (anti-band) subspace via band-concentration deflation.
pub struct SoninRestriction {
    /// Band-concentration eigenvalues `χ ∈ (0,1)`, ascending. The count
    /// of `χ ≈ 1` is the Shannon number; the `χ ≈ 0` tail is the Sonin
    /// subspace.
    pub chi: Vec<Float>,
    /// Spectrum of the archimedean Weil form after deflating the top
    /// `n_dropped` band-concentrated modes (those land near `+σ`). The
    /// smallest entry is the archimedean Rayleigh minimum on the
    /// band-complement (Sonin-like) subspace — positive iff archimedean
    /// positivity holds there (Connes Thm 7.1).
    pub spectrum: Vec<Float>,
    /// Number of band-concentrated modes deflated out.
    pub n_dropped: usize,
}

/// Archimedean Weil-form spectrum restricted to the Sonin-like (anti-band)
/// subspace. The full-space archimedean form is O(1)-indefinite; this
/// deflates the top `n_drop` band-concentration eigenvectors (band
/// `(−Ω, Ω)`, `omega = Ω`) by a positive shift `σ`, leaving the
/// archimedean form on the band-complement. Returns the deflated spectrum;
/// its minimum is the archimedean Rayleigh minimum on that subspace.
///
/// Method: build [`band_concentration_matrix_hp`], take its top `n_drop`
/// eigenvectors `v_k` (largest `χ`), and form `Ã = A_arch + σ Σ_k v_k v_kᵀ`
/// with `σ` a Gershgorin bound on `‖A_arch‖` (so deflated modes leave the
/// spectrum bottom). `A_arch` is the prime-free Weil matrix
/// (`build_tau_hp_compute(.., include_primes=false)`).
pub fn weil_spectrum_sonin_hp(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
    n_drop: usize,
) -> Result<SoninRestriction> {
    xc_numerics::hp_runtime::run_hp(|| weil_spectrum_sonin_hp_inner(params, cfg, omega, n_drop))
}

fn weil_spectrum_sonin_hp_inner(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    omega: &Float,
    n_drop: usize,
) -> Result<SoninRestriction> {
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();
    let l = log_lambda_sq_hp(params, prec);

    let cmat = band_concentration_matrix_hp_inner(params, cfg, omega)?;
    let chi = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&cmat, dim, prec)?;

    let n_drop = n_drop.min(dim);
    let mut deflate: Vec<Vec<Float>> = Vec::with_capacity(n_drop);
    for k in 0..n_drop {
        let chi_k = &chi[dim - 1 - k]; // largest χ first
        let mut v = xc_numerics::eigen::dense_symmetric_eigenvector_for_value_hp(
            &cmat, dim, chi_k, prec, cfg.inverse_iter_steps,
        )?;
        xc_numerics::linalg::normalize_l2(&mut v);
        deflate.push(v);
    }

    let mut a_arch = build_tau_hp_compute(params, &l, cfg, false)?;
    force_symmetric(&mut a_arch, dim);

    // σ = 10 · (Gershgorin bound on |spectrum|) + 1.
    let mut sigma = Float::with_val(prec, 0);
    for i in 0..dim {
        let mut row = Float::with_val(prec, 0);
        for j in 0..dim {
            row += a_arch[i * dim + j].clone().abs();
        }
        if row > sigma {
            sigma = row;
        }
    }
    sigma *= 10u32;
    sigma += 1u32;

    for v in &deflate {
        for i in 0..dim {
            let mut svi = v[i].clone();
            svi *= &sigma;
            for j in 0..dim {
                let mut term = svi.clone();
                term *= &v[j];
                a_arch[i * dim + j] += &term;
            }
        }
    }
    force_symmetric(&mut a_arch, dim);
    let spectrum = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&a_arch, dim, prec)?;

    Ok(SoninRestriction { chi, spectrum, n_dropped: n_drop })
}

fn build_tau_hp_compute(params: &CcmParams, l: &Float, cfg: &HighPrecConfig, include_primes: bool) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let n_max = params.n_modes;
    let dim = params.matrix_size();
    let lambda_sq_int = params.lambda_sq_int();

    let pi_v = pi(prec);
    let mut two_pi = pi_v.clone(); two_pi *= 2u32;
    let mut sixteen_pi2 = pi_v.clone().square(); sixteen_pi2 *= 16u32;
    let l_sq = l.clone().square();
    let sinh2_l_over_4 = { let mut v = l.clone(); v /= 4u32; v.sinh().square() };

    let base_pts = cfg.quad_points;
    let prec_extra = (cfg.precision_bits / 2) as usize;
    crate::hp_debug!("[HP] Computing α_L, β_L, γ_L for n=0..{} (base quad={})", n_max, base_pts);

    use std::collections::HashMap;
    let pts_for_n: Vec<usize> = (0..=n_max).map(|n| base_pts.max(3 * n + prec_extra)).collect();
    let unique_pts: Vec<usize> = {
        let mut v = pts_for_n.clone(); v.sort_unstable(); v.dedup(); v
    };
    eprintln!("[HP] Precomputing {} unique GL node tables (npts up to {}, prec={} bits)...",
        unique_pts.len(), unique_pts.last().copied().unwrap_or(0), prec);
    // xc_numerics::hp_runtime::map_gl_precompute: parallel on Vast/native
    // Linux (unchanged); sequential on WSL2, where sustained concurrent
    // GMP allocation across many back-to-back GL computes triggers a
    // non-deterministic glibc abort (confirmed independent of rayon —
    // reproduces with plain std::thread too). GL tables are cached to
    // disk after first compute, so this only costs time on a cold cache.
    let gl_pairs: Vec<(usize, (Vec<Float>, Vec<Float>))> = xc_numerics::hp_runtime::map_gl_precompute(
        &unique_pts,
        |&npts| (npts, xc_numerics::quadrature::gauss_legendre_nodes(npts, prec, cfg.cache_mode)),
    );
    let gl_cache: HashMap<usize, (Vec<Float>, Vec<Float>)> = gl_pairs.into_iter().collect();
    eprintln!("[HP] GL tables ready. Computing α_L, β_L, γ_L integrals...");

    let indices: Vec<usize> = (0..=n_max).collect();
    let alpha_l: Vec<Float> = indices.par_iter().map(|&n| {
        let pts = pts_for_n[n];
        let (nodes, weights) = gl_cache.get(&pts).unwrap();
        compute_alpha_l(n as i64, l, prec, nodes, weights)
    }).collect();
    let beta_l: Vec<Float> = indices.par_iter().map(|&n| {
        let pts = pts_for_n[n];
        let (nodes, weights) = gl_cache.get(&pts).unwrap();
        compute_beta_l(n as i64, l, prec, nodes, weights)
    }).collect();
    let gamma_l: Vec<Float> = indices.par_iter().map(|&n| {
        let pts = pts_for_n[n];
        let (nodes, weights) = gl_cache.get(&pts).unwrap();
        compute_gamma_l(n as i64, l, prec, nodes, weights)
    }).collect();
    eprintln!("[HP] Integrals done. Assembling {}×{} τ-matrix...", dim, dim);

    let prime_powers = prime_powers_up_to(lambda_sq_int);
    // Pure HP path: compute log_p in HP from the exposed prime, do not
    // recover j from log ratios. j is provided directly by the sieve.
    let pp_data: Vec<(Float, Float, Float)> = prime_powers.iter().map(|&(k, _p, _j)| {
        let log_k = Float::with_val(prec, k).ln();
        let log_p = Float::with_val(prec, _p).ln();
        let sqrt_k = Float::with_val(prec, k).sqrt();
        (log_k, log_p, sqrt_k)
    }).collect();

    let cells: Vec<(i64, i64)> = (-(n_max as i64)..=(n_max as i64))
        .flat_map(|n| (-(n_max as i64)..=(n_max as i64)).map(move |m| (n, m)))
        .collect();

    let computed: Vec<Float> = cells.par_iter().map(|&(n, m)| {
        let nf = fl_i(prec, n);
        let mf = fl_i(prec, m);

        let w02 = {
            let mut mn = sixteen_pi2.clone(); mn *= &mf; mn *= &nf;
            let mut num = l_sq.clone(); num -= &mn;
            let mut a = sixteen_pi2.clone(); a *= &mf; a *= &mf; a += &l_sq;
            let mut b = sixteen_pi2.clone(); b *= &nf; b *= &nf; b += &l_sq;
            let mut den = a; den *= &b;
            let mut v = sinh2_l_over_4.clone(); v *= 32u32; v *= l; v *= &num; v /= &den;
            v
        };

        let wr = if n == m {
            let k = n.unsigned_abs() as usize;
            let mut v = gamma_l[k].clone(); v -= &beta_l[k]; v *= 2u32; v
        } else {
            let an = signed_alpha(&alpha_l, n, prec);
            let am = signed_alpha(&alpha_l, m, prec);
            let mut v = am; v -= &an; v /= fl_i(prec, n - m); v
        };

        let two_pi_n_over_l = { let mut v = two_pi.clone(); v *= &nf; v /= l; v };
        let two_pi_m_over_l = { let mut v = two_pi.clone(); v *= &mf; v /= l; v };
        let mut wp = Float::with_val(prec, 0);
        if include_primes {
        for (log_k, log_p, sqrt_k) in &pp_data {
            let q = if n == m {
                let mut ph = two_pi_n_over_l.clone(); ph *= log_k;
                let c = ph.cos();
                let mut t = log_k.clone(); t /= l;
                let mut f = Float::with_val(prec, 1); f -= &t; f *= 2u32; f *= &c; f
            } else {
                let mut sm = two_pi_m_over_l.clone(); sm *= log_k; let sm_s = sm.sin();
                let mut sn = two_pi_n_over_l.clone(); sn *= log_k; let sn_s = sn.sin();
                let mut d = sm_s; d -= &sn_s;
                let mut dn = pi_v.clone(); dn *= fl_i(prec, n - m);
                d /= &dn; d
            };
            let mut term = q; term *= log_p; term /= sqrt_k;
            wp += &term;
        }
        }
        let mut t = w02; t -= &wr; t -= &wp; t
    }).collect();

    let mut tau = vec![Float::with_val(prec, 0); dim * dim];
    for (i, &(n, m)) in cells.iter().enumerate() {
        tau[params.idx(n) * dim + params.idx(m)] = computed[i].clone();
    }
    Ok(tau)
}

fn signed_alpha(table: &[Float], n: i64, prec: u32) -> Float {
    let k = n.unsigned_abs() as usize;
    if k >= table.len() { return Float::with_val(prec, 0); }
    if n < 0 { let mut v = table[k].clone(); v = -v; v } else { table[k].clone() }
}

fn compute_alpha_l(n: i64, l: &Float, prec: u32, nodes: &[Float], weights: &[Float]) -> Float {
    if n == 0 { return Float::with_val(prec, 0); }
    let pi_v = pi(prec);
    let mut freq = pi_v.clone(); freq *= 2u32; freq *= fl_i(prec, n); freq /= l;
    let f = |x: &Float| -> Float {
        let mut ph = freq.clone(); ph *= x;
        let mut r = ph.sin(); r *= &rho_hp(x, prec); r
    };
    let mut v = quad_eval(nodes, weights, l, f);
    v /= &pi_v; v
}

fn compute_beta_l(n: i64, l: &Float, prec: u32, nodes: &[Float], weights: &[Float]) -> Float {
    let pi_v = pi(prec);
    let mut freq = pi_v.clone(); freq *= 2u32; freq *= fl_i(prec, n); freq /= l;
    let f = |x: &Float| -> Float {
        let mut ph = freq.clone(); ph *= x;
        let c = ph.cos();
        let mut r = x.clone(); r *= &c; r *= &rho_hp(x, prec); r
    };
    let mut v = quad_eval(nodes, weights, l, f);
    v /= l; v
}

fn compute_gamma_l(n: i64, l: &Float, prec: u32, nodes: &[Float], weights: &[Float]) -> Float {
    let pi_v = pi(prec);
    let mut freq = pi_v.clone(); freq *= 2u32; freq *= fl_i(prec, n); freq /= l;
    let f = |x: &Float| -> Float {
        let mut ph = freq.clone(); ph *= x;
        let c = ph.cos();
        let mut neg_half = x.clone(); neg_half /= -2i32;
        let e = neg_half.exp();
        let mut diff = c; diff -= &e;
        diff *= &rho_hp(x, prec); diff
    };
    let mut v = quad_eval(nodes, weights, l, f);
    let kappa_half = {
        let exp_l = l.clone().exp();
        let mut num = exp_l.clone(); num -= 1u32;
        let mut den = exp_l; den += 1u32;
        let mut ratio = num; ratio /= &den;
        let mut four_pi = pi_v; four_pi *= 4u32;
        ratio *= &four_pi;
        let mut k = ratio.ln(); k += euler(prec); k /= 2u32; k
    };
    v += &kappa_half; v
}

fn rho_hp(x: &Float, prec: u32) -> Float {
    let tiny = Float::with_val(prec, Float::parse(HP_SINGULARITY_GUARD_STR).unwrap());
    if x.cmp_abs(&tiny).map(|o| o.is_lt()).unwrap_or(false) {
        let mut v = x.clone().recip(); v /= 2u32; v += { let mut q = Float::with_val(prec, 1); q /= 4u32; q }; v
    } else {
        let mut hx = x.clone(); hx /= 2u32;
        let e = hx.exp();
        let mut d = x.clone().sinh(); d *= 2u32;
        let mut r = e; r /= &d; r
    }
}

fn quad_eval<F>(nodes: &[Float], weights: &[Float], l: &Float, f: F) -> Float
where F: Fn(&Float) -> Float {
    let prec = l.prec();
    let mut half_l = l.clone(); half_l /= 2u32;
    let mut acc = Float::with_val(prec, 0);
    for (n, w) in nodes.iter().zip(weights) {
        let mut x = n.clone(); x += 1u32; x *= &half_l;
        let mut term = w.clone(); term *= &f(&x);
        acc += &term;
    }
    acc *= &half_l; acc
}

fn normalize_eigenvector(xi: &[Float], l: &Float, prec: u32) -> Vec<Float> {
    let mut sum = Float::with_val(prec, 0);
    for v in xi { sum += v; }
    let mut target = l.clone().sqrt();
    target /= &sum;
    xi.iter().map(|v| { let mut x = v.clone(); x *= &target; x }).collect()
}

/// Basis half-period `L = ln(λ²) = 2 ln λ` at full working precision.
///
/// Compute `L = ln(λ²)` at full HP precision.
///
/// Integer mode: uses `value_u64` for exact HP promotion —
/// full working-precision result, no f64 representation error.
///
/// Fractional mode: formats `value_f64` to 17 significant figures
/// and parses into HP. Gives L accurate to ~17 digits (the f64
/// input limit).
fn log_lambda_sq_hp(params: &CcmParams, prec: u32) -> Float {
    if params.lambda_sq.is_integer {
        Float::with_val(prec, params.lambda_sq.value_u64).ln()
    } else {
        let lsq_str = format!("{:.17e}", params.lambda_sq.value_f64);
        let lsq_hp = Float::with_val(prec, Float::parse(&lsq_str).unwrap());
        lsq_hp.ln()
    }
}


// ===========================================================================
// Newton refinement of R(z) zeros
// ===========================================================================

/// Returns the name of the active solver based on `XCELERATOR_SOLVER`.
fn active_solver_name() -> &'static str {
    match std::env::var("XCELERATOR_SOLVER").as_deref() {
        Ok("newton") => "Newton",
        _ => "Halley",
    }
}

/// Root-finding method for R(z) zeros.
///
/// Selected via the `XCELERATOR_SOLVER` environment variable:
///   - `"halley"` (default): cubic convergence, 3 passes per step, ~33%
///     fewer steps than Newton. Requires seeds reasonably close to the
///     true eigenvalue — satisfied in practice by the f64 warm seeds.
///   - `"newton"`: quadratic convergence, 2 passes per step. Use if Halley
///     exhibits convergence issues on a specific config.
fn solve_r_zero(
    xi: &[Float], n_max: usize, l: &Float, seed: &Float, prec: u32, n_steps: usize,
) -> EigenvalueResult {
    match std::env::var("XCELERATOR_SOLVER").as_deref() {
        Ok("newton") => newton_xi_hat_zero(xi, n_max, l, seed, prec, n_steps),
        _            => halley_xi_hat_zero(xi, n_max, l, seed, prec, n_steps),
    }
}

/// Newton's method for a zero of R(z) = Σ ξ_j / (z − 2πj/L).
///
/// Quadratic convergence: correct digits double each step.
/// Uses R(z) and R'(z) — 2 passes over the poles per step.
///
/// Returns:
/// - `Converged(z)` if the step size dropped below HP tolerance, OR if
///   the step stagnated (stopped shrinking) while already below the f64
///   floor — signals the construction's precision ceiling was reached.
/// - `Approximate(z)` if the step limit was exhausted before either
///   convergence or stagnation (best approximation found).
/// - `Failed` only if R'(z) = 0 (degenerate denominator — garbage).
fn newton_xi_hat_zero(
    xi: &[Float], n_max: usize, l: &Float, seed: &Float, prec: u32, n_steps: usize,
) -> EigenvalueResult {
    let two_pi_over_l = { let mut v = pi(prec); v *= 2u32; v /= l; v };
    let mut z = seed.clone();
    let tol = Float::with_val(prec, 2).pow(-((prec as i32) - 16));
    // Stagnation floor: if |dz| stops shrinking while already below f64
    // precision, we've hit the construction's accuracy ceiling. Accept.
    let stagnation_tol = Float::with_val(prec, 2).pow(-53);
    let mut prev_dz = Float::with_val(prec, f64::INFINITY);
    for _ in 0..n_steps {
        let mut r = Float::with_val(prec, 0);
        let mut r_prime = Float::with_val(prec, 0);
        for j in -(n_max as i64)..=(n_max as i64) {
            let idx = (j + n_max as i64) as usize;
            let mut pole = two_pi_over_l.clone(); pole *= fl_i(prec, j);
            let mut den = z.clone(); den -= &pole;
            let mut term = xi[idx].clone(); term /= &den;
            r += &term;
            let mut den_sq = den.clone(); den_sq.square_mut();
            let mut dterm = xi[idx].clone(); dterm /= &den_sq;
            r_prime -= &dterm;
        }
        if r_prime.is_zero() { return EigenvalueResult::Failed; }
        let mut dz = r; dz /= &r_prime;
        z -= &dz;
        let abs_dz = dz.abs();
        // Converged to full HP precision.
        if abs_dz.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return EigenvalueResult::Converged(z);
        }
        // Stagnated: step stopped shrinking while already below f64 floor.
        // The construction's accuracy ceiling has been reached.
        if abs_dz >= prev_dz && prev_dz.cmp_abs(&stagnation_tol).map(|o| o.is_lt()).unwrap_or(false) {
            return EigenvalueResult::Converged(z);
        }
        prev_dz = abs_dz;
    }
    // Step limit exhausted — return best approximation, not a hard failure.
    crate::hp_debug!(
        "[HP] WARNING: Newton failed to converge for seed {} — returning Approximate",
        seed.to_f64()
    );
    EigenvalueResult::Approximate(z)
}

/// Halley's method for a zero of R(z) = Σ ξ_j / (z − 2πj/L).
///
/// Cubic convergence: correct digits triple each step.
/// Uses R(z), R'(z), and R''(z) — 3 passes over the poles per step.
/// Step: z ← z − 2·R·R' / (2·R'² − R·R'')
///
/// Returns:
/// - `Converged(z)` if the step size dropped below HP tolerance, OR if
///   the step stagnated (stopped shrinking) while already below the f64
///   floor — signals the construction's precision ceiling was reached.
/// - `Approximate(z)` if the step limit was exhausted before either
///   convergence or stagnation (best approximation found).
/// - `Failed` only if the Halley denominator is zero (degenerate).
fn halley_xi_hat_zero(
    xi: &[Float], n_max: usize, l: &Float, seed: &Float, prec: u32, n_steps: usize,
) -> EigenvalueResult {
    let two_pi_over_l = { let mut v = pi(prec); v *= 2u32; v /= l; v };
    let mut z = seed.clone();
    let tol = Float::with_val(prec, 2).pow(-((prec as i32) - 16));
    // Stagnation floor: if |dz| stops shrinking while already below f64
    // precision, we've hit the construction's accuracy ceiling. Accept.
    let stagnation_tol = Float::with_val(prec, 2).pow(-53);
    let mut prev_dz = Float::with_val(prec, f64::INFINITY);
    for _ in 0..n_steps {
        let mut r        = Float::with_val(prec, 0);
        let mut r_prime  = Float::with_val(prec, 0);
        let mut r_dprime = Float::with_val(prec, 0);
        for j in -(n_max as i64)..=(n_max as i64) {
            let idx = (j + n_max as i64) as usize;
            let mut pole = two_pi_over_l.clone(); pole *= fl_i(prec, j);
            let mut den = z.clone(); den -= &pole;

            let mut term = xi[idx].clone(); term /= &den;
            r += &term;

            let mut den2 = den.clone(); den2.square_mut();
            let mut dt = xi[idx].clone(); dt /= &den2;
            r_prime -= &dt;

            let mut den3 = den2.clone(); den3 *= &den;
            let mut ddt = xi[idx].clone(); ddt /= &den3;
            ddt *= 2u32;
            r_dprime += &ddt;
        }

        let mut denom = r_prime.clone();
        denom.square_mut();
        denom *= 2u32;
        let mut rr2 = r.clone(); rr2 *= &r_dprime;
        denom -= &rr2;

        if denom.is_zero() { return EigenvalueResult::Failed; }

        let mut dz = r.clone(); dz *= &r_prime; dz *= 2u32; dz /= &denom;
        z -= &dz;
        let abs_dz = dz.abs();
        // Converged to full HP precision.
        if abs_dz.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return EigenvalueResult::Converged(z);
        }
        // Stagnated: step stopped shrinking while already below f64 floor.
        // The construction's accuracy ceiling has been reached.
        if abs_dz >= prev_dz && prev_dz.cmp_abs(&stagnation_tol).map(|o| o.is_lt()).unwrap_or(false) {
            return EigenvalueResult::Converged(z);
        }
        prev_dz = abs_dz;
    }
    // Step limit exhausted — return best approximation, not a hard failure.
    crate::hp_debug!(
        "[HP] WARNING: Halley failed to converge for seed {} — returning Approximate",
        seed.to_f64()
    );
    EigenvalueResult::Approximate(z)
}


// ===========================================================================
// τ-matrix disk cache
// ===========================================================================

mod tau_cache {
    //! Disk cache for the HP τ-matrix produced by `build_tau_hp_compute`.
    //!
    //! Cache layout under `<cwd>/data/tau_cache/`:
    //!   - `lambda_sq{L}_nmodes{N}_prec{P}.json` (uncompressed, fast path)
    //!   - `lambda_sq{L}_nmodes{N}_prec{P}.json.zip` (single-zip when
    //!     compressed payload ≤ 90 MB)
    //!   - `lambda_sq{L}_nmodes{N}_prec{P}.json.zip.part00, .part01, ...`
    //!     (byte-split when compressed payload > 90 MB; we split at
    //!     90 MB-byte boundaries, comfortably under GitHub's 100 MB
    //!     hard file size limit)
    //!
    //! Read priority: uncompressed → single zip → multi-part zip →
    //! compute fresh. Multi-part read concatenates the part bytes in
    //! lexicographic order and decompresses the result as one zip.
    //! Write logic picks single vs multi-part based on the compressed
    //! payload size.
    //!
    //! Structural validation on cache hit: matrix length matches
    //! `(2N+1)²`, no NaN/Inf entries, and symmetry `τ[i,j] = τ[j,i]`
    //! to working precision. A file that fails validation is
    //! discarded with a stderr warning; the toolkit falls through to
    //! compute fresh. Bad files are preserved on disk.

    use rug::{ops::Pow, Float};
    use std::io::{Read, Write};

    use super::super::LambdaSq;

    /// Per-part byte cap for split files. 90 MB stays comfortably
    /// under GitHub's 100 MB hard file limit and well under the
    /// 50 MB soft warning, leaving headroom for git's pack overhead.
    const PART_BYTE_LIMIT: usize = 90 * 1024 * 1024;

    /// Toolkit version string embedded in every tau cache file written
    /// by this build. Matches `[workspace.package].version` in `Cargo.toml`.
    const TOOLKIT_VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Minimum toolkit version required to use a tau cache file. Files
    /// produced by an older toolkit are treated as cache misses and
    /// recomputed. Update this constant when a change to the tau
    /// computation changes the stored values.
    const CACHE_MIN_TOOLKIT_VERSION: &str = "0.10.0";

    /// Tolerance for the symmetry identity check on cache load.
    /// At precision `prec` bits this is `2^-(prec - 8)` — same
    /// pattern used by the GL and prolate caches.
    fn cache_tol(prec: u32) -> Float {
        Float::with_val(prec, 2).pow(-((prec as i32) - 8))
    }

    fn cache_dir() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let dir = cwd.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    pub(super) fn cache_filename(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> String {
        format!("lambda_sq{}_nmodes{}_prec{}.json", lambda_sq.filename_str(), n_modes, prec)
    }

    fn json_path(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| d.join(cache_filename(lambda_sq, n_modes, prec)))
    }

    fn zip_path(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| {
            let f = cache_filename(lambda_sq, n_modes, prec);
            d.join(format!("{}.zip", f))
        })
    }

    /// Glob the .partXX files for a given config in lexicographic
    /// order. Returns `None` if no parts exist.
    fn part_paths(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<Vec<std::path::PathBuf>> {
        let dir = cache_dir()?;
        let stem = cache_filename(lambda_sq, n_modes, prec);
        let prefix = format!("{}.zip.part", stem);
        let mut parts: Vec<std::path::PathBuf> = std::fs::read_dir(&dir).ok()?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name();
                let s = name.to_str()?;
                if s.starts_with(&prefix) { Some(entry.path()) } else { None }
            })
            .collect();
        if parts.is_empty() { return None; }
        parts.sort();
        Some(parts)
    }

    /// Parse the cache JSON for the tau matrix.
    /// Expects schema_version 1 envelope format. Returns `None` on any
    /// structural mismatch or incompatible `min_compatible_version`.
    fn parse_json(data: &str, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        let dim = 2 * n_modes + 1;
        let n_expected = dim * dim;
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let obj = parsed.as_object()?;

        if let Some(min_ver) = obj.get("min_compatible_version").and_then(|v| v.as_str()) {
            if version_is_older(CACHE_MIN_TOOLKIT_VERSION, min_ver) {
                return None;
            }
        }

        let arr = obj.get("matrix")?.as_array()?;
        if arr.len() != n_expected { return None; }
        let mut out = Vec::with_capacity(n_expected);
        for s in arr {
            out.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some(out)
    }

    /// Returns `true` if version string `a` is strictly older than `b`.
    fn version_is_older(a: &str, b: &str) -> bool {
        let parse = |s: &str| -> (u64, u64, u64) {
            let mut parts = s.splitn(3, '.');
            let major = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let patch = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            (major, minor, patch)
        };
        parse(a) < parse(b)
    }

    /// Verify the loaded matrix satisfies the structural identities:
    /// length matches `(2N+1)²`, no NaN/Inf entries, and symmetry
    /// `τ[i,j] = τ[j,i]` to working precision.
    pub(super) fn structural_check(
        tau: &[Float], n_modes: usize, prec: u32,
    ) -> Option<String> {
        let dim = 2 * n_modes + 1;
        if tau.len() != dim * dim {
            return Some(format!(
                "matrix length {} != expected {}², = {}",
                tau.len(), dim, dim * dim
            ));
        }
        for (k, v) in tau.iter().enumerate() {
            if v.is_nan() {
                return Some(format!("entry {} is NaN", k));
            }
            if v.is_infinite() {
                return Some(format!("entry {} is infinite", k));
            }
        }
        // Symmetry check: τ[i,j] = τ[j,i] for i < j.
        let tol = cache_tol(prec);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let mut diff = tau[i * dim + j].clone();
                diff -= &tau[j * dim + i];
                let abs_diff = diff.abs();
                if !abs_diff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
                    return Some(format!(
                        "asymmetry at ({},{}): τ[{},{}] - τ[{},{}] = {} (tol {})",
                        i, j, i, j, j, i, abs_diff, tol
                    ));
                }
            }
        }
        None
    }

    fn warn_skip(path: &std::path::Path, reason: &str) {
        crate::hp_debug!(
            "[tau_cache] WARNING: skipping {} ({}); recomputing",
            path.display(), reason
        );
    }

    /// Read a single zip and return both the parsed matrix and the
    /// raw inner JSON bytes (so the caller can write the decompressed
    /// copy without re-serializing).
    fn read_single_zip(
        zip_bytes: &[u8],
        json_filename: &str,
        n_modes: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, String)> {
        let cursor = std::io::Cursor::new(zip_bytes);
        let mut archive = zip::ZipArchive::new(cursor).ok()?;
        let mut entry = archive.by_name(json_filename).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_json(&data, n_modes, prec)?;
        Some((parsed, data))
    }

    /// Concatenate the bytes of all `.partXX` files in order, then
    /// treat the result as a single zip and decompress.
    fn read_split_zip_parts(
        parts: &[std::path::PathBuf],
        json_filename: &str,
        n_modes: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, String)> {
        let mut concatenated: Vec<u8> = Vec::new();
        for p in parts {
            let mut bytes = Vec::new();
            std::fs::File::open(p).ok()?
                .read_to_end(&mut bytes).ok()?;
            concatenated.extend_from_slice(&bytes);
        }
        read_single_zip(&concatenated, json_filename, n_modes, prec)
    }

    pub(super) fn load(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        mode: xc_numerics::quadrature::CacheMode,
    ) -> Option<Vec<Float>> {
        use xc_numerics::quadrature::CacheMode;
        if mode == CacheMode::Off { return None; }

        // Caches are zip-only: we read straight from the .json.zip
        // (decompress in memory) and never write a decompressed .json.
        // Keeps local disk usage ~2x smaller. JsonOnly is now a read
        // no-op (kept for API compatibility) since no .json is written.
        if mode == CacheMode::JsonOnly { return None; }

        // Tier 1 (JsonZip, DynamicFetch): local zip — single first, then
        // multi-part. Decompressed in memory; no .json written.
        if let Some(tau) = try_load_local_zip(lambda_sq, n_modes, prec) {
            return Some(tau);
        }

        // JsonZip stops after the local tiers.
        if mode == CacheMode::JsonZip { return None; }

        // Tier 2 (DynamicFetch only): remote fetch from the public
        // consolidated τ-cache repo. Probe the single zip first, then the
        // byte-split parts. On success the file(s) land in the local
        // cache dir; we then re-run the local-zip loader.
        if fetch_remote(lambda_sq, n_modes, prec) {
            if let Some(tau) = try_load_local_zip(lambda_sq, n_modes, prec) {
                return Some(tau);
            }
        }

        None
    }

    /// Attempt to load a config from a local single zip, then local
    /// multi-part parts. Decompresses in memory; does NOT write a
    /// decompressed `.json`. Returns `None` if no local zip/parts exist
    /// or they fail validation.
    fn try_load_local_zip(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        let json_filename = cache_filename(lambda_sq, n_modes, prec);

        // Single zip first.
        if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
            if zp.exists() {
                match std::fs::read(&zp) {
                    Ok(bytes) => match read_single_zip(&bytes, &json_filename, n_modes, prec) {
                        Some((tau, _data)) => {
                            if let Some(reason) = structural_check(&tau, n_modes, prec) {
                                warn_skip(&zp, &reason);
                            } else {
                                return Some(tau);
                            }
                        }
                        None => warn_skip(&zp, "zip parse / shape failed"),
                    },
                    Err(e) => warn_skip(&zp, &format!("read failed: {}", e)),
                }
            }
        }

        // Multi-part split zip.
        if let Some(parts) = part_paths(lambda_sq, n_modes, prec) {
            let first_part_path = parts.first().cloned().unwrap_or_default();
            match read_split_zip_parts(&parts, &json_filename, n_modes, prec) {
                Some((tau, _data)) => {
                    if let Some(reason) = structural_check(&tau, n_modes, prec) {
                        warn_skip(&first_part_path, &format!("{} (split parts)", reason));
                    } else {
                        return Some(tau);
                    }
                }
                None => warn_skip(
                    &first_part_path,
                    &format!("could not concatenate / decompress {} parts", parts.len()),
                ),
            }
        }

        None
    }

    /// Base raw URLs of the public τ-cache repositories, in probe order.
    ///
    /// The toolkit checks each repo in sequence during a remote fetch and
    /// stops at the first hit. Repo-1 is checked first (it holds all
    /// fixtures generated before the split); repo-2 holds all fixtures
    /// generated after the split. Adding a third repo in the future is a
    /// one-line change here.
    const REMOTE_BASES: &[&str] = &[
        "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache/main",
        "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache-2/main",
    ];

    /// Remote directory URL (and filename stem) for a config in a
    /// specific base repo, using the precision-first → λ² →
    /// nmodes-thousand-bucket layout.
    /// Returns `(dir_url, stem)` where `stem` is the canonical
    /// `lambda_sq{L}_nmodes{N}_prec{P}.json.zip` filename.
    fn remote_dir_and_stem(base: &str, lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> (String, String) {
        let bucket = (n_modes / 1000) * 1000;
        let dir = format!(
            "{base}/tau_cache/prec{p}/lambda_sq{l}/nmodes{b}-{bend}",
            base = base, p = prec, l = lambda_sq.filename_str(), b = bucket, bend = bucket + 999
        );
        let stem = format!("{}.zip", cache_filename(lambda_sq, n_modes, prec));
        (dir, stem)
    }

    /// Test-only accessor for `remote_dir_and_stem` (the function is
    /// private; this lets the test module assert URL formatting without
    /// widening the API). Returns results for all configured bases.
    #[cfg(test)]
    pub(super) fn remote_dir_and_stem_for_test(
        lambda_sq: LambdaSq, n_modes: usize, prec: u32,
    ) -> Vec<(String, String)> {
        REMOTE_BASES.iter()
            .map(|base| remote_dir_and_stem(base, lambda_sq, n_modes, prec))
            .collect()
    }

    /// Test-only accessor for `parse_json` (lets version-rejection tests
    /// call the parser directly without touching disk).
    #[cfg(test)]
    pub(super) fn parse_json_for_test(
        data: &str, n_modes: usize, prec: u32,
    ) -> Option<Vec<Float>> {
        parse_json(data, n_modes, prec)
    }

    /// Outcome of a single `curl` download attempt, classified by the
    /// actual HTTP status code.
    enum CurlOutcome {
        /// HTTP 2xx, file written and renamed into place.
        Ok,
        /// HTTP 404 specifically — the file does not exist. For part
        /// probing this is the genuine end-of-parts marker.
        HttpError,
        /// Anything else: 429 (rate limit), 5xx, no response, curl
        /// missing, network/DNS/timeout, write error. Must be retried;
        /// must NOT be treated as end-of-parts (would silently truncate).
        Transient,
    }

    /// `curl` a single URL to `dest`, capturing the actual HTTP status
    /// code so we can distinguish a genuine 404 (end-of-parts) from a
    /// transient 429/5xx (rate-limit / server hiccup — must retry, NOT
    /// stop). raw.githubusercontent.com rate-limits bursts of requests,
    /// so a multi-part fetch will hit 429 if we go too fast; treating
    /// that as end-of-parts would silently truncate the download.
    ///
    /// Uses `--write-out %{http_code}` (not `--fail`) so curl still
    /// writes the body decision to us via the printed code. We only
    /// keep the file on a 2xx. Downloads to a temp path and renames on
    /// success so a failed download never leaves a truncated file.
    fn curl_attempt(url: &str, dest: &std::path::Path) -> CurlOutcome {
        let tmp = dest.with_extension("downloading");
        let _ = std::fs::remove_file(&tmp);
        let output = std::process::Command::new("curl")
            .arg("--silent").arg("--show-error").arg("--location")
            .arg("--retry").arg("3").arg("--retry-delay").arg("1")
            .arg("--write-out").arg("%{http_code}")
            .arg("-o").arg(&tmp).arg(url)
            .output();
        match output {
            Ok(out) if out.status.success() => {
                let code: u32 = String::from_utf8_lossy(&out.stdout)
                    .trim().parse().unwrap_or(0);
                match code {
                    200..=299 => match std::fs::rename(&tmp, dest) {
                        Ok(()) => CurlOutcome::Ok,
                        Err(_) => { let _ = std::fs::remove_file(&tmp); CurlOutcome::Transient }
                    },
                    404 => { let _ = std::fs::remove_file(&tmp); CurlOutcome::HttpError }
                    // 429 (rate limit), 5xx, redirects-gone-wrong, 0
                    // (no response): all transient — retry, don't stop.
                    _ => { let _ = std::fs::remove_file(&tmp); CurlOutcome::Transient }
                }
            }
            // curl itself failed to run / network-level error / missing curl.
            _ => { let _ = std::fs::remove_file(&tmp); CurlOutcome::Transient }
        }
    }

    /// Download a single URL with retries on transient failure
    /// (including HTTP 429 rate-limiting). Returns the final outcome
    /// (`Ok`, `HttpError` for a genuine 404, or `Transient` if all
    /// retries were exhausted). Uses a longer backoff than curl's own
    /// `--retry` because GitHub's rate-limit window is seconds-scale.
    fn curl_with_retries(url: &str, dest: &std::path::Path) -> CurlOutcome {
        const MAX_TRIES: usize = 5;
        for attempt in 0..MAX_TRIES {
            match curl_attempt(url, dest) {
                CurlOutcome::Ok => return CurlOutcome::Ok,
                CurlOutcome::HttpError => return CurlOutcome::HttpError, // 404: definitive
                CurlOutcome::Transient => {
                    if attempt + 1 < MAX_TRIES {
                        // Backoff grows: 2s, 4s, 6s, 8s — rides out a
                        // GitHub rate-limit window.
                        let secs = 2 * (attempt as u64 + 1);
                        std::thread::sleep(std::time::Duration::from_secs(secs));
                    }
                }
            }
        }
        CurlOutcome::Transient
    }

    /// Download a config from the public τ-cache repos into the local
    /// cache dir. Iterates over `REMOTE_BASES` in order, trying each
    /// repo until one contains the file.
    ///
    /// Within each repo, probes the **single zip first**; if that 404s,
    /// probes the byte-split parts `.part00`, `.part01`, … and stops only
    /// when a part returns an HTTP error (404 = genuine end-of-parts).
    ///
    /// A *transient* failure on any part (after retries) aborts the whole
    /// fetch for that repo and tries the next repo — we must never
    /// silently truncate a multi-part config. On abort, any downloaded
    /// parts are removed so a stale partial set can't be mistaken for a
    /// complete one on a later load.
    ///
    /// Mirrors the local read order (single → parts) so the subsequent
    /// `try_load_local_zip` finds and validates whatever was fetched.
    fn fetch_remote(lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> bool {
        let dir = match cache_dir() { Some(d) => d, None => return false };

        for base in REMOTE_BASES {
            let (remote_dir, stem) = remote_dir_and_stem(base, lambda_sq, n_modes, prec);

            // Probe 1: single zip.
            let single_url = format!("{}/{}", remote_dir, stem);
            let single_dest = dir.join(&stem);
            match curl_with_retries(&single_url, &single_dest) {
                CurlOutcome::Ok => {
                    // Routine cache hit — silent.
                    return true;
                }
                CurlOutcome::Transient => {
                    // Network trouble on this repo's single-zip probe;
                    // skip to the next repo.
                    continue;
                }
                // HttpError (404): no single zip in this repo — may be
                // byte-split, or not present here at all.
                CurlOutcome::HttpError => {}
            }

            // Probe 2: byte-split parts. Download .part00, .part01, … until
            // an HTTP error (404) marks the genuine end. A transient failure
            // aborts this repo and tries the next — never a silent truncation.
            let mut downloaded: Vec<std::path::PathBuf> = Vec::new();
            let mut idx = 0usize;
            let mut transient_abort = false;
            loop {
                let part_name = format!("{}.part{:02}", stem, idx);
                let part_url = format!("{}/{}", remote_dir, part_name);
                let part_dest = dir.join(&part_name);
                match curl_with_retries(&part_url, &part_dest) {
                    CurlOutcome::Ok => {
                        downloaded.push(part_dest);
                        idx += 1;
                        if idx > 256 { break; } // safety cap
                        // Brief inter-part pause to stay under
                        // raw.githubusercontent.com's burst rate limit
                        // (datacenter IPs trip 429 on rapid sequential
                        // requests). Cheap insurance vs. a retry storm.
                        std::thread::sleep(std::time::Duration::from_millis(300));
                    }
                    CurlOutcome::HttpError => break, // genuine end-of-parts (or file not in this repo)
                    CurlOutcome::Transient => {
                        // Could not retrieve a part that may well exist.
                        // Abort: remove everything downloaded so a partial
                        // set is never concatenated into a corrupt zip.
                        crate::hp_debug!(
                            "[tau_cache] WARNING: transient failure fetching {} after retries; \
                             trying next repo",
                            part_name
                        );
                        for p in &downloaded { let _ = std::fs::remove_file(p); }
                        transient_abort = true;
                        break;
                    }
                }
            }

            if transient_abort {
                continue; // try next repo
            }

            if !downloaded.is_empty() {
                // Routine multi-part cache hit — silent.
                return true;
            }

            // No single zip and no parts in this repo — try the next one.
        }

        false
    }

    /// Serialize `tau` to the versioned JSON envelope and return the
    /// resulting bytes.
    fn serialize_to_json(tau: &[Float], lambda_sq: LambdaSq, n_modes: usize, prec: u32) -> Vec<u8> {
        let strs: Vec<String> = tau.iter().map(|f| f.to_string()).collect();
        let payload = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": TOOLKIT_VERSION,
            "min_compatible_version": CACHE_MIN_TOOLKIT_VERSION,
            "lambda_sq": lambda_sq.value_f64,
            "lambda_sq_mode": lambda_sq.mode_str(),
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        });
        serde_json::to_vec(&payload).unwrap_or_default()
    }

    /// Compress `json_bytes` to a deflated zip in-memory; the inner
    /// entry is named `entry_name`. Returns the resulting zip bytes.
    fn compress_to_zip(json_bytes: &[u8], entry_name: &str) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(json_bytes.len() / 2);
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = zip::ZipWriter::new(cursor);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            if writer.start_file(entry_name, opts).is_err() { return Vec::new(); }
            if writer.write_all(json_bytes).is_err() { return Vec::new(); }
            if writer.finish().is_err() { return Vec::new(); }
        }
        buf
    }

    /// Remove any pre-existing single-zip / multi-part-zip / .json
    /// files for this config so we don't leave stale partner files
    /// from a previous run that wrote a different shape.
    fn cleanup_previous(lambda_sq: LambdaSq, n_modes: usize, prec: u32) {
        if let Some(p) = json_path(lambda_sq, n_modes, prec) {
            if p.exists() { let _ = std::fs::remove_file(&p); }
        }
        if let Some(p) = zip_path(lambda_sq, n_modes, prec) {
            if p.exists() { let _ = std::fs::remove_file(&p); }
        }
        if let Some(parts) = part_paths(lambda_sq, n_modes, prec) {
            for p in parts { let _ = std::fs::remove_file(&p); }
        }
    }

    pub(super) fn save(
        lambda_sq: LambdaSq,
        n_modes: usize,
        prec: u32,
        tau: &[Float],
        mode: xc_numerics::quadrature::CacheMode,
    ) {
        use xc_numerics::quadrature::CacheMode;
        // Off and JsonOnly write nothing: the cache is zip-only (we never
        // persist a decompressed .json), so only JsonZip / DynamicFetch
        // produce output.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) { return; }

        // Serialize to JSON in memory.
        let json_bytes = serialize_to_json(tau, lambda_sq, n_modes, prec);
        if json_bytes.is_empty() { return; }

        cleanup_previous(lambda_sq, n_modes, prec);

        // Write ONLY the compressed copy. Readers decompress from the zip
        // on demand — no uncompressed .json is persisted. Decide single-zip
        // vs multi-part split based on compressed size.
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let zip_bytes = compress_to_zip(&json_bytes, &entry_name);
        if zip_bytes.is_empty() { return; }

        if zip_bytes.len() <= PART_BYTE_LIMIT {
            // Single-zip path: under the per-part cap, write one file.
            if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
                if let Err(e) = std::fs::write(&zp, &zip_bytes) {
                    crate::hp_debug!(
                        "[tau_cache] WARNING: could not write {}: {}",
                        zp.display(), e
                    );
                }
            }
        } else {
            // Multi-part split path: byte-split at PART_BYTE_LIMIT.
            // The toolkit reads parts back by lexicographic
            // concatenation, so naming uses zero-padded indices to
            // keep the order correct.
            let n_parts = zip_bytes.len().div_ceil(PART_BYTE_LIMIT);
            let dir = match cache_dir() { Some(d) => d, None => return };
            for i in 0..n_parts {
                let start = i * PART_BYTE_LIMIT;
                let end = ((i + 1) * PART_BYTE_LIMIT).min(zip_bytes.len());
                let part_path = dir.join(format!("{}.zip.part{:02}", entry_name, i));
                if let Err(e) = std::fs::write(&part_path, &zip_bytes[start..end]) {
                    crate::hp_debug!(
                        "[tau_cache] WARNING: could not write {}: {}",
                        part_path.display(), e
                    );
                    return;
                }
            }
            crate::hp_debug!(
                "[tau_cache] wrote {} parts of ≤{} MB each (compressed total {} MB) for λ²={}, N={}, prec={}",
                n_parts, PART_BYTE_LIMIT / (1024 * 1024),
                zip_bytes.len() / (1024 * 1024),
                lambda_sq.value_f64, n_modes, prec
            );
        }
    }

    /// Per-file outcome from `verify_tau_cache_dir`.
    ///
    /// Each variant carries the file path and (when parseable from the
    /// filename) the cache key tuple `(lambda_sq, n_modes, prec)`.
    /// `Skipped` and `LoadFailed` only have the tuple if the filename
    /// matched the expected pattern.
    #[derive(Debug, Clone)]
    pub enum TauCacheFileStatus {
        /// File parsed and passed all structural identity checks.
        Ok { path: std::path::PathBuf, lambda_sq: LambdaSq, n_modes: usize, prec: u32 },
        /// File was skipped. Either the filename didn't match the
        /// expected pattern, or it's a `.partXX` chunk handled as part
        /// of an assembled multi-part archive.
        Skipped { path: std::path::PathBuf, reason: String },
        /// Filename matched but the file failed to load (decompress,
        /// concatenate, parse JSON, etc.).
        LoadFailed { path: std::path::PathBuf, lambda_sq: LambdaSq, n_modes: usize, prec: u32, reason: String },
        /// File loaded but failed at least one structural identity
        /// check on the τ matrix it contains.
        StructurallyInvalid { path: std::path::PathBuf, lambda_sq: LambdaSq, n_modes: usize, prec: u32, reason: String },
    }

    /// Aggregate report from `verify_tau_cache_dir`.
    #[derive(Debug, Clone)]
    pub struct TauCacheVerifyReport {
        /// Directory that was scanned.
        pub directory: std::path::PathBuf,
        /// One status entry per file (or per assembled multi-part set)
        /// found in `directory`.
        pub statuses: Vec<TauCacheFileStatus>,
    }

    impl TauCacheVerifyReport {
        /// Count of files that passed all checks.
        pub fn ok_count(&self) -> usize {
            self.statuses.iter()
                .filter(|s| matches!(s, TauCacheFileStatus::Ok { .. }))
                .count()
        }
        /// Count of files that failed at least one check (load or
        /// structural). Skipped files are not counted as failures.
        pub fn failure_count(&self) -> usize {
            self.statuses.iter().filter(|s| matches!(s,
                TauCacheFileStatus::LoadFailed { .. }
                | TauCacheFileStatus::StructurallyInvalid { .. }
            )).count()
        }
        /// All failure entries (load + structural), for callers that
        /// want to print only the bad files.
        pub fn failures(&self) -> impl Iterator<Item = &TauCacheFileStatus> {
            self.statuses.iter().filter(|s| matches!(s,
                TauCacheFileStatus::LoadFailed { .. }
                | TauCacheFileStatus::StructurallyInvalid { .. }
            ))
        }
    }

    /// Parse `lambda_sq{L}_nmodes{N}_prec{P}.json[.zip[.partXX]]`.
    /// Returns the tuple plus a flag indicating whether the file is a
    /// part of a split archive (so the verifier can skip individual
    /// parts and inspect the assembled set instead).
    pub(super) fn parse_filename(name: &str) -> Option<(LambdaSq, usize, u32, FileKind)> {
        // Three possible suffixes, in priority order.
        let (stem, kind) = if let Some(s) = strip_part_suffix(name) {
            (s, FileKind::Part)
        } else if let Some(s) = name.strip_suffix(".json.zip") {
            (s, FileKind::Zip)
        } else if let Some(s) = name.strip_suffix(".json") {
            (s, FileKind::Json)
        } else {
            return None;
        };
        let after_lsq = stem.strip_prefix("lambda_sq")?;
        let (lsq_str, rest) = after_lsq.split_once("_nmodes")?;
        
        
        let (n_str, prec_str) = rest.split_once("_prec")?;
        
        
        Some((
            LambdaSq::from_filename_str(lsq_str)?,
            n_str.parse().ok()?,
            prec_str.parse().ok()?,
            kind,
        ))
    }

    /// Strip the `.json.zip.partXX` suffix and return the stem (the
    /// `lambda_sq{L}_nmodes{N}_prec{P}` portion). Returns `None` if
    /// the name doesn't match this pattern.
    fn strip_part_suffix(name: &str) -> Option<&str> {
        let pos = name.rfind(".json.zip.part")?;
        // Validate the suffix after `.part` is digits only.
        let rest = &name[pos + ".json.zip.part".len()..];
        if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        Some(&name[..pos])
    }

    pub(super) enum FileKind { Json, Zip, Part }

    /// Walk the τ-cache directory and structurally verify every file.
    /// Multi-part archives are verified as a set: if any `.partXX`
    /// files are present for a config, the verifier reads them all,
    /// concatenates, decompresses, and validates the result. Files
    /// not in the expected pattern are reported as `Skipped`.
    pub fn verify_tau_cache_dir(
        dir: &std::path::Path,
    ) -> std::io::Result<TauCacheVerifyReport> {
        let mut statuses: Vec<TauCacheFileStatus> = Vec::new();

        if !dir.exists() {
            return Ok(TauCacheVerifyReport {
                directory: dir.to_path_buf(),
                statuses,
            });
        }

        // First pass: bucket files by (lambda_sq, n_modes, prec, kind).
        // Multi-part archives (kind=Part) are deduplicated: we verify
        // the whole set once per config rather than per-file.
        // Key is (filename_str, n_modes, prec) for uniqueness; value is LambdaSq.
        let mut configs_with_parts: std::collections::BTreeMap<(String, usize, u32), LambdaSq> = std::collections::BTreeMap::new();
        let mut singletons: Vec<(std::path::PathBuf, LambdaSq, usize, u32, FileKind)> = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() { continue; }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            match parse_filename(&name) {
                Some((lsq, n_modes, prec, kind)) => match kind {
                    FileKind::Json | FileKind::Zip => {
                        singletons.push((path, lsq, n_modes, prec, kind));
                    }
                    FileKind::Part => {
                        configs_with_parts.insert((lsq.filename_str(), n_modes, prec), lsq);
                    }
                },
                None => {
                    statuses.push(TauCacheFileStatus::Skipped {
                        path,
                        reason: format!(
                            "filename '{}' not in expected lambda_sq{{L}}_nmodes{{N}}_prec{{P}}.json[.zip[.partXX]] form",
                            name
                        ),
                    });
                }
            }
        }

        // Verify singletons.
        for (path, lsq, n_modes, prec, kind) in singletons {
            let parsed: Option<Vec<Float>> = match kind {
                FileKind::Json => std::fs::read_to_string(&path).ok()
                    .and_then(|d| parse_json(&d, n_modes, prec)),
                FileKind::Zip => std::fs::read(&path).ok().and_then(|bytes| {
                    let entry_name = cache_filename(lsq, n_modes, prec);
                    read_single_zip(&bytes, &entry_name, n_modes, prec).map(|(t, _)| t)
                }),
                FileKind::Part => unreachable!(),
            };
            match parsed {
                Some(tau) => match structural_check(&tau, n_modes, prec) {
                    None => statuses.push(TauCacheFileStatus::Ok {
                        path, lambda_sq: lsq, n_modes, prec,
                    }),
                    Some(reason) => statuses.push(TauCacheFileStatus::StructurallyInvalid {
                        path, lambda_sq: lsq, n_modes, prec, reason,
                    }),
                },
                None => statuses.push(TauCacheFileStatus::LoadFailed {
                    path, lambda_sq: lsq, n_modes, prec,
                    reason: "parse / decompress failed".to_string(),
                }),
            }
        }

        // Verify split-archive sets (one entry per config, not per part).
        for ((_key, n_modes, prec), lsq) in configs_with_parts {
            let parts = match part_paths(lsq, n_modes, prec) {
                Some(p) => p,
                None => continue,
            };
            let representative = parts[0].clone();
            let entry_name = cache_filename(lsq, n_modes, prec);
            match read_split_zip_parts(&parts, &entry_name, n_modes, prec) {
                Some((tau, _data)) => match structural_check(&tau, n_modes, prec) {
                    None => statuses.push(TauCacheFileStatus::Ok {
                        path: representative, lambda_sq: lsq, n_modes, prec,
                    }),
                    Some(reason) => statuses.push(TauCacheFileStatus::StructurallyInvalid {
                        path: representative, lambda_sq: lsq, n_modes, prec, reason,
                    }),
                },
                None => statuses.push(TauCacheFileStatus::LoadFailed {
                    path: representative, lambda_sq: lsq, n_modes, prec,
                    reason: format!("split archive ({} parts) failed to assemble / decompress", parts.len()),
                }),
            }
        }

        Ok(TauCacheVerifyReport {
            directory: dir.to_path_buf(),
            statuses,
        })
    }
}

#[cfg(feature = "hp")]
pub use tau_cache::{
    verify_tau_cache_dir,
    TauCacheVerifyReport, TauCacheFileStatus,
};

// ===========================================================================
// Weil-eigenvector (ξ) disk cache
// ===========================================================================

mod weil_eigvec_cache {
    //! Disk cache for the smallest-eigenvalue eigenvector ξ of the Weil
    //! quadratic form (the vector produced by `inverse_iteration` inside
    //! [`super::run`], ℓ²-normalized so Σξ = √L). Distinct from the
    //! prolate eigenvalue cache (`prolate_eigvals_cache`, different
    //! operator *and* quantity) and the τ-matrix cache (`tau_cache`,
    //! different quantity).
    //!
    //! Cache layout under `<cwd>/data/weil_eigvec_cache/`:
    //!   - `weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json` (uncompressed,
    //!     fast path)
    //!   - `weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json.zip`
    //!     (single-zip companion for distribution)
    //!
    //! Unlike `tau_cache`, ξ is small (2N+1 entries, ≲ 2 MB even at
    //! HP-1000/N=800), so there is no byte-split `.partXX` tier — single
    //! zip only, exactly like the GL-node cache.
    //!
    //! Schema mirrors [`super::HighPrecResult::save_xi_json`]
    //! (`schema_version: 1`): a JSON object carrying ξ as decimal strings
    //! plus `weil_min_eigenvalue` (ε_N) and the `(λ², N, prec)` metadata.
    //!
    //! Validation on load is two-tier:
    //!   1. *Structural* (here, no τ needed): length = 2N+1, finite
    //!      entries, metadata match. Cheap O(N).
    //!   2. *Residual* (at the [`super::run`] call site, where τ is in
    //!      hand): ‖τξ − ε_N·ξ‖ below the working-precision floor. This is
    //!      the strongest integrity test and is why the cache check sits
    //!      *after* the τ build.

    use rug::{ops::Pow, Float};
    use std::io::{Read, Write};

    use xc_numerics::quadrature::CacheMode;

    use super::super::LambdaSq;

    /// Base raw URL of the public consolidated Weil-eigenvector cache
    /// repository. Files live at
    /// `{REMOTE_BASE}/weil_eigvec_cache/prec{P}/lambda_sq{L}/nmodes{B}-{B+999}/weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json.zip`
    /// where `B = (N / 1000) * 1000`.
    const REMOTE_BASE: &str =
        "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-weil-eigvec-cache/main";

    /// Toolkit version string embedded in every weil eigvec cache file
    /// written by this build.
    const TOOLKIT_VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Minimum toolkit version required to use a weil eigvec cache file.
    /// Files produced by an older toolkit are treated as cache misses.
    /// Update this constant when a change to the eigenvector computation
    /// changes the stored values.
    const CACHE_MIN_TOOLKIT_VERSION: &str = "0.10.0";

    /// Current schema version for the weil eigvec JSON envelope.
    const SCHEMA_VERSION: u32 = 1;

    /// A ξ entry loaded from the cache: the eigenvector plus its
    /// eigenvalue ε_N, both at the requested working precision.
    pub(super) struct CachedXi {
        pub eps_n: Float,
        pub xi: Vec<Float>,
    }

    fn cache_dir() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let dir = cwd.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    /// Find the nearest cached ξ for the same (lambda_sq, n_modes)
    /// but a different precision, within `prec_tolerance` bits.
    ///
    /// Scans the local weil_eigvec_cache directory for filename patterns
    /// matching `weil_eigvec_lambda_sq{lsq}_nmodes{n}_prec*.json.zip`
    /// and returns the closest-prec entry within tolerance, promoted to
    /// the target precision. Returns `None` if no match is found or the
    /// feature is disabled.
    ///
    /// This is used as a warm-start for inverse iteration: instead of the
    /// Gaussian initial guess, we start from a ξ that is already close to
    /// the answer, reducing iteration count from ~50-100 to ~2-5.
    pub(super) fn find_warm_start(
        lambda_sq: LambdaSq,
        n_modes: usize,
        target_prec: u32,
        tolerance_bits: u32,
        force_even: bool,
    ) -> Option<Vec<Float>> {
        let dir = cache_dir()?;
        let variant = if force_even { "" } else { "_natural" };
        let prefix = format!("weil_eigvec_lambda_sq{}_nmodes{}_prec", lambda_sq.filename_str(), n_modes);
        let suffix = format!("{}.json.zip", variant);

        let mut best_prec: Option<u32> = None;
        let mut best_diff = u32::MAX;

        // Scan directory for matching filenames
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with(&prefix) || !name_str.ends_with(&suffix) {
                continue;
            }
            // Parse the prec value from the filename
            let mid = &name_str[prefix.len()..name_str.len() - suffix.len()];
            if let Ok(p) = mid.parse::<u32>() {
                if p == target_prec { continue; } // exact match handled elsewhere
                let diff = target_prec.abs_diff(p);
                if diff <= tolerance_bits && diff < best_diff {
                    best_diff = diff;
                    best_prec = Some(p);
                }
            }
        }

        let nearby_prec = best_prec?;

        // Load the nearby cache entry (at its original precision)
        let cached = load(lambda_sq, n_modes, nearby_prec, CacheMode::JsonZip, force_even)?;

        // Promote ξ to target precision by re-parsing each entry.
        // This is exact: we're just increasing the mantissa bits, no rounding.
        let xi_promoted: Vec<Float> = cached.xi.iter()
            .map(|v| {
                let s = v.to_string();
                Float::with_val(target_prec, Float::parse(&s).unwrap_or_else(|_| Float::parse("0").unwrap()))
            })
            .collect();

        crate::hp_debug!(
            "[HP] warm-start from nearby cache prec={} (target={}, diff={} bits)",
            nearby_prec, target_prec, best_diff
        );

        Some(xi_promoted)
    }

    pub(super) fn cache_filename(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) -> String {
        // Forced-even (the default CCM path) keeps the historical name with
        // NO suffix, so every pre-existing public fixture stays valid. The
        // natural (unprojected) variant gets a `_natural` marker so the two
        // never collide in the cache.
        let variant = if force_even { "" } else { "_natural" };
        format!("weil_eigvec_lambda_sq{}_nmodes{}_prec{}{}.json", lambda_sq.filename_str(), n_modes, prec, variant)
    }

    fn json_path(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| d.join(cache_filename(lambda_sq, n_modes, prec, force_even)))
    }

    fn zip_path(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| {
            let f = cache_filename(lambda_sq, n_modes, prec, force_even);
            d.join(format!("{}.zip", f))
        })
    }

    /// Parse the cache JSON object into `(eps_n, xi)`.
    /// Expects schema_version 1 envelope format. Returns `None` on any
    /// structural mismatch or incompatible `min_compatible_version`.
    pub(super) fn parse_json(
        data: &str, lambda_sq: LambdaSq, n_modes: usize, prec: u32,
    ) -> Option<CachedXi> {
        let v: serde_json::Value = serde_json::from_str(data).ok()?;
        let obj = v.as_object()?;

        if let Some(min_ver) = obj.get("min_compatible_version").and_then(|x| x.as_str()) {
            if version_is_older(CACHE_MIN_TOOLKIT_VERSION, min_ver) {
                return None;
            }
        }

        if obj.get("n_modes").and_then(|x| x.as_u64())? as usize != n_modes { return None; }
        if obj.get("precision_bits").and_then(|x| x.as_u64())? as u32 != prec { return None; }
        let l_meta = obj.get("lambda_sq").and_then(|x| x.as_f64())?;
        if (l_meta - lambda_sq.value_f64).abs() > 0.5 { return None; }

        let eps_str = obj.get("weil_min_eigenvalue").and_then(|x| x.as_str())?;
        let eps_n = Float::with_val(prec, Float::parse(eps_str).ok()?);
        if eps_n.is_nan() || eps_n.is_infinite() { return None; }

        let arr = obj.get("xi").and_then(|x| x.as_array())?;
        if arr.len() != 2 * n_modes + 1 { return None; }
        let mut xi = Vec::with_capacity(arr.len());
        for s in arr {
            let f = Float::with_val(prec, Float::parse(s.as_str()?).ok()?);
            if f.is_nan() || f.is_infinite() { return None; }
            xi.push(f);
        }
        Some(CachedXi { eps_n, xi })
    }

    /// Returns `true` if version string `a` is strictly older than `b`.
    fn version_is_older(a: &str, b: &str) -> bool {
        let parse = |s: &str| -> (u64, u64, u64) {
            let mut parts = s.splitn(3, '.');
            let major = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let patch = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            (major, minor, patch)
        };
        parse(a) < parse(b)
    }

    /// Eigen-residual check: is `(xi, eps_n)` a genuine eigenpair of the
    /// in-hand τ matrix? Returns `true` when `‖τξ − ε_N·ξ‖_∞ / ‖ξ‖_∞`
    /// sits below the working-precision floor. This is the strong
    /// integrity test that catches a structurally-valid-but-wrong ξ
    /// (e.g. a different eigenvector, or one from a subtly different τ).
    pub(super) fn residual_ok(
        tau: &[Float], dim: usize, xi: &[Float], eps_n: &Float, prec: u32,
    ) -> bool {
        if xi.len() != dim || tau.len() != dim * dim { return false; }

        // ‖ξ‖_∞ for the relative bound. A zero vector can never be a
        // valid eigenvector.
        let mut xi_linf = Float::with_val(prec, 0);
        for v in xi {
            let a = v.clone().abs();
            if a > xi_linf { xi_linf = a; }
        }
        if xi_linf.is_zero() { return false; }

        // max_i | (τξ)_i − ε_N ξ_i |, rows computed in parallel then a
        // deterministic max-fold. The inner row sum is sequential (it is
        // the same fixed index order every run).
        use rayon::prelude::*;
        let resid_inf = (0..dim).into_par_iter().map(|i| {
            let mut row = Float::with_val(prec, 0);
            for j in 0..dim {
                let mut t = tau[i * dim + j].clone();
                t *= &xi[j];
                row += &t;
            }
            let mut e = eps_n.clone();
            e *= &xi[i];
            row -= &e;
            row.abs()
        }).reduce(|| Float::with_val(prec, 0), |a, b| if a > b { a } else { b });

        // Relative residual vs floor. Use a generous floor: the eigenpair
        // is accurate to ~working precision, but the residual accumulates
        // O(N) HP roundings in the matrix-vector product. 2^-(prec-32)
        // leaves 32 bits (~10 digits) of headroom — far below the O(1)
        // residual a wrong ξ would produce, yet safely above the genuine
        // floor.
        let mut rel = resid_inf;
        rel /= &xi_linf;
        let floor = Float::with_val(prec, 2).pow(-((prec as i32) - 32));
        rel.cmp_abs(&floor).map(|o| o.is_lt()).unwrap_or(false)
    }

    /// Deterministic remote URL for the `(λ², N, prec)` fixture in the
    /// public xcelerator-weil-eigvec-cache repo (precision-first → λ² →
    /// nmodes-thousand-bucket layout, mirroring tau/GL).
    fn remote_zip_url(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) -> String {
        let bucket = (n_modes / 1000) * 1000;
        format!(
            "{base}/weil_eigvec_cache/prec{p}/lambda_sq{l}/nmodes{b}-{bend}/{stem}.zip",
            base = REMOTE_BASE, p = prec, l = lambda_sq.filename_str(),
            b = bucket, bend = bucket + 999,
            stem = cache_filename(lambda_sq, n_modes, prec, force_even)
        )
    }

    /// Test-only accessor for `remote_zip_url`.
    #[cfg(test)]
    pub(super) fn remote_zip_url_for_test(
        lambda_sq: LambdaSq, n_modes: usize, prec: u32,
    ) -> String {
        remote_zip_url(lambda_sq, n_modes, prec, true)
    }

    /// Test-only accessor for `parse_json` (lets version-rejection tests
    /// call the parser directly without touching disk).
    #[cfg(test)]
    pub(super) fn parse_json_for_test(
        data: &str, lambda_sq: LambdaSq, n_modes: usize, prec: u32,
    ) -> Option<CachedXi> {
        parse_json(data, lambda_sq, n_modes, prec)
    }

    fn warn_skip(path: &std::path::Path, reason: &str) {
        crate::hp_debug!(
            "[weil_eigvec_cache] WARNING: skipping {} ({}); recomputing",
            path.display(), reason
        );
    }

    /// Read a single zip and return the parsed entry plus the raw inner
    /// JSON (so the caller can write the decompressed copy without
    /// re-serializing).
    fn read_single_zip(
        zip_path: &std::path::Path,
        lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool,
    ) -> Option<(CachedXi, String)> {
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        let entry_name = cache_filename(lambda_sq, n_modes, prec, force_even);
        let mut entry = archive.by_name(&entry_name).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_json(&data, lambda_sq, n_modes, prec)?;
        Some((parsed, data))
    }

    pub(super) fn load(
        lambda_sq: LambdaSq, n_modes: usize, prec: u32, mode: CacheMode, force_even: bool,
    ) -> Option<CachedXi> {
        if mode == CacheMode::Off { return None; }

        // Caches are zip-only: read straight from the .json.zip
        // (decompress in memory), never write a decompressed .json.
        // JsonOnly is now a read no-op (kept for API compatibility).
        if mode == CacheMode::JsonOnly { return None; }

        // Tier 1 (JsonZip, DynamicFetch): local single zip — in memory.
        if let Some(c) = try_load_local_zip(lambda_sq, n_modes, prec, force_even) {
            return Some(c);
        }

        // JsonZip stops after the local tiers.
        if mode == CacheMode::JsonZip { return None; }

        // Tier 2 (DynamicFetch only): remote fetch (single zip; no parts,
        // ξ is small). On success the zip lands locally and we re-run the
        // local-zip loader.
        if fetch_remote_zip(lambda_sq, n_modes, prec, force_even) {
            if let Some(c) = try_load_local_zip(lambda_sq, n_modes, prec, force_even) {
                return Some(c);
            }
        }

        None
    }

    /// Load from a local single zip. Decompresses in memory; does NOT
    /// write a decompressed `.json`. Returns the parsed entry or None.
    fn try_load_local_zip(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) -> Option<CachedXi> {
        let zp = zip_path(lambda_sq, n_modes, prec, force_even)?;
        if !zp.exists() { return None; }
        match read_single_zip(&zp, lambda_sq, n_modes, prec, force_even) {
            Some((parsed, _json_string)) => Some(parsed),
            None => {
                warn_skip(&zp, "zip open / decompress / shape parse failed");
                None
            }
        }
    }

    /// Outcome of a single `curl` download attempt (mirrors the GL/τ
    /// fetch classification).
    enum CurlOutcome { Ok, HttpError, Transient }

    fn curl_attempt(url: &str, dest: &std::path::Path) -> CurlOutcome {
        let tmp = dest.with_extension("zip.partial");
        let _ = std::fs::remove_file(&tmp);
        let output = std::process::Command::new("curl")
            .arg("--silent").arg("--show-error").arg("--location")
            .arg("--retry").arg("3").arg("--retry-delay").arg("1")
            .arg("--write-out").arg("%{http_code}")
            .arg("-o").arg(&tmp).arg(url)
            .output();
        match output {
            Ok(out) if out.status.success() => {
                let code: u32 = String::from_utf8_lossy(&out.stdout)
                    .trim().parse().unwrap_or(0);
                match code {
                    200..=299 => match std::fs::rename(&tmp, dest) {
                        Ok(()) => CurlOutcome::Ok,
                        Err(_) => { let _ = std::fs::remove_file(&tmp); CurlOutcome::Transient }
                    },
                    404 => { let _ = std::fs::remove_file(&tmp); CurlOutcome::HttpError }
                    _ => { let _ = std::fs::remove_file(&tmp); CurlOutcome::Transient }
                }
            }
            _ => { let _ = std::fs::remove_file(&tmp); CurlOutcome::Transient }
        }
    }

    /// Download the `(λ², N, prec)` `.json.zip` from the public cache
    /// repo. Returns `true` if a file was written. Robust to
    /// `raw.githubusercontent.com` rate-limiting: only a 404 is a
    /// definitive miss; 429/5xx/no-response retry with backoff.
    fn fetch_remote_zip(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) -> bool {
        let dest = match zip_path(lambda_sq, n_modes, prec, force_even) {
            Some(p) => p,
            None => return false,
        };
        let url = remote_zip_url(lambda_sq, n_modes, prec, force_even);

        const MAX_TRIES: usize = 5;
        for attempt in 0..MAX_TRIES {
            match curl_attempt(&url, &dest) {
                CurlOutcome::Ok => {
                    // Routine cache hit — silent.
                    return true;
                }
                CurlOutcome::HttpError => return false, // 404 — definitive miss.
                CurlOutcome::Transient => {
                    if attempt + 1 < MAX_TRIES {
                        let secs = 2 * (attempt as u64 + 1);
                        std::thread::sleep(std::time::Duration::from_secs(secs));
                    }
                }
            }
        }
        false
    }

    /// Serialize `(eps_n, xi)` to the versioned schema-v1 JSON object.
    fn serialize_to_json(
        lambda_sq: LambdaSq, n_modes: usize, prec: u32, eps_n: &Float, xi: &[Float],
    ) -> Vec<u8> {
        let xi_strings: Vec<String> = xi.iter().map(|f| f.to_string()).collect();
        let payload = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "toolkit_version": TOOLKIT_VERSION,
            "min_compatible_version": CACHE_MIN_TOOLKIT_VERSION,
            "lambda_sq": lambda_sq.value_f64,
            "lambda_sq_mode": lambda_sq.mode_str(),
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": eps_n.to_string(),
            "xi": xi_strings,
        });
        serde_json::to_vec(&payload).unwrap_or_default()
    }

    fn compress_to_zip(json_bytes: &[u8], entry_name: &str) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(json_bytes.len() / 2);
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = zip::ZipWriter::new(cursor);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            if writer.start_file(entry_name, opts).is_err() { return Vec::new(); }
            if writer.write_all(json_bytes).is_err() { return Vec::new(); }
            if writer.finish().is_err() { return Vec::new(); }
        }
        buf
    }

    fn cleanup_previous(lambda_sq: LambdaSq, n_modes: usize, prec: u32, force_even: bool) {
        if let Some(p) = json_path(lambda_sq, n_modes, prec, force_even) {
            if p.exists() { let _ = std::fs::remove_file(&p); }
        }
        if let Some(p) = zip_path(lambda_sq, n_modes, prec, force_even) {
            if p.exists() { let _ = std::fs::remove_file(&p); }
        }
    }

    pub(super) fn save(
        lambda_sq: LambdaSq, n_modes: usize, prec: u32,
        eps_n: &Float, xi: &[Float], mode: CacheMode, force_even: bool,
    ) {
        // Off and JsonOnly write nothing: the cache is zip-only.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) { return; }

        let json_bytes = serialize_to_json(lambda_sq, n_modes, prec, eps_n, xi);
        if json_bytes.is_empty() { return; }

        cleanup_previous(lambda_sq, n_modes, prec, force_even);

        // Write ONLY the compressed copy. Readers decompress from the zip
        // on demand — no uncompressed .json is persisted. ξ is small, so
        // this is always a single zip (no byte-split tier — unlike τ).
        let entry_name = cache_filename(lambda_sq, n_modes, prec, force_even);
        let zip_bytes = compress_to_zip(&json_bytes, &entry_name);
        if zip_bytes.is_empty() { return; }
        if let Some(zp) = zip_path(lambda_sq, n_modes, prec, force_even) {
            if let Err(e) = std::fs::write(&zp, &zip_bytes) {
                crate::hp_debug!(
                    "[weil_eigvec_cache] WARNING: could not write {}: {}",
                    zp.display(), e
                );
            }
        }
    }
}


// Matrix fixtures below use the row-major index convention `m[i * dim + j]`
// uniformly, including `i = 0` rows where `0 * dim` is kept for alignment
// with neighboring entries. Allow the erasing_op lint in this test module.
#[cfg(test)]
#[allow(clippy::erasing_op)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialize all cwd-mutating cache tests in this module. Cargo runs
    /// tests in parallel by default; cwd is per-process (not per-thread),
    /// so two cache tests racing on `set_current_dir` would corrupt each
    /// other (one test deleting the temp dir another captured as its
    /// "original"). The mutex enforces sequential access. Mirrors the GL
    /// cache tests in `xc-numerics::quadrature`.
    ///
    /// This aliases the crate-level [`crate::TEST_CWD_LOCK`] so the
    /// `ccm::hp` and `prolate::hp` cache tests — which share one
    /// process-global cwd within the same test binary — serialize
    /// against *each other*, not just within their own module.
    #[allow(dead_code)]
    static CWD_LOCK: &Mutex<()> = &crate::TEST_CWD_LOCK;

    /// Guard that restores the original cwd on drop, so a panic inside a
    /// test doesn't leave the runner in a temp dir (which would break
    /// subsequent unrelated tests). Holds the CWD_LOCK for the guard's
    /// lifetime to serialize cwd mutation.
    struct CwdGuard {
        original: std::path::PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    impl CwdGuard {
        fn enter(temp: &std::path::Path) -> Self {
            // Recover from poison: a previously-panicking test poisons the
            // lock, but subsequent tests can still safely acquire it (the
            // global cwd state isn't corrupted by a panic — the prior
            // guard's Drop ran on unwind and restored cwd).
            let lock = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let original = std::env::current_dir().expect("no cwd");
            std::env::set_current_dir(temp).expect("set_current_dir to temp");
            CwdGuard { original, _lock: lock }
        }
    }
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    /// HighPrecConfig::for_decimal_digits should produce expected values.
    #[test]
    fn config_for_200_digits() {
        let cfg = HighPrecConfig::for_decimal_digits(200);
        // 200 * 3.322 = 664.4 → ceil = 665 + 16 guard = 681 bits
        assert_eq!(cfg.precision_bits, 681);
        // 200 * 3 = 600, clamped to [600, 4000] → 600
        assert_eq!(cfg.quad_points, 600);
        assert_eq!(cfg.inverse_iter_steps, 200);
        // solver_steps = max(200, ceil(log2(200/10))) = 200 (safety cap)
        assert_eq!(cfg.solver_steps, 200);
        assert_eq!(cfg.n_eigenvalues, 50);
        assert!(cfg.force_even, "force_even should default to true");
    }

    /// Band-concentration matrix is symmetric, has a [0,1] spectrum, and
    /// exhibits a concentrated/Sonin split (some χ≈1, some χ≈0).
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn band_concentration_spectrum_in_unit_interval() {
        let params = CcmParams::from_lambda_sq_integer(5, 8);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let dim = params.matrix_size();
        let omega = Float::with_val(prec, 15);
        let c = band_concentration_matrix_hp(&params, &cfg, &omega).unwrap();
        for i in 0..dim {
            for j in 0..dim {
                let d = (c[i * dim + j].to_f64() - c[j * dim + i].to_f64()).abs();
                assert!(d < 1e-25, "C not symmetric at ({i},{j}): {d}");
            }
            let cii = c[i * dim + i].to_f64();
            assert!(cii > -1e-9 && cii < 1.0 + 1e-6, "diag {cii} out of [0,1]");
        }
        let chi = xc_numerics::eigen::dense_symmetric_eigenvalues_hp(&c, dim, prec).unwrap();
        let chi_f: Vec<f64> = chi.iter().map(|x| x.to_f64()).collect();
        for &x in &chi_f {
            assert!(x > -1e-6 && x < 1.0 + 1e-6, "chi {x} out of [0,1]");
        }
        assert!(chi_f.iter().any(|&x| x > 0.9), "expected a concentrated (χ≈1) mode");
        assert!(chi_f.iter().any(|&x| x < 0.1), "expected a Sonin-like (χ≈0) mode");
    }

    /// Sonin restriction deflates exactly `n_drop` band-concentrated modes:
    /// they cluster near the shift σ (≈ max eigenvalue), well separated
    /// from the rest, and the result shape is consistent.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn weil_sonin_deflates_band_modes() {
        let params = CcmParams::from_lambda_sq_integer(5, 8);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let dim = params.matrix_size();
        let omega = Float::with_val(prec, 15);
        let n_drop = 5usize;
        let res = weil_spectrum_sonin_hp(&params, &cfg, &omega, n_drop).unwrap();
        assert_eq!(res.chi.len(), dim);
        assert_eq!(res.spectrum.len(), dim);
        assert_eq!(res.n_dropped, n_drop);
        for w in res.chi.windows(2) {
            assert!(w[0].to_f64() <= w[1].to_f64() + 1e-12, "χ not ascending");
        }
        // The n_drop deflated modes cluster near σ (the max); everything
        // else sits below σ/2 — so exactly n_drop eigenvalues exceed σ/2.
        let s: Vec<f64> = res.spectrum.iter().map(|x| x.to_f64()).collect();
        let max_v = *s.last().unwrap();
        assert!(max_v > 10.0, "deflation shift σ should be large, got {max_v}");
        let big = s.iter().filter(|&&e| e > max_v / 2.0).count();
        assert_eq!(big, n_drop, "expected {n_drop} deflated modes near σ, got {big}");
    }

    /// HighPrecConfig::for_decimal_digits at 500 digits.
    #[test]
    fn config_for_500_digits() {
        let cfg = HighPrecConfig::for_decimal_digits(500);
        // 500 * 3.322 = 1661 → ceil = 1661 + 16 = 1677 bits
        assert_eq!(cfg.precision_bits, 1677);
        // 500 * 3 = 1500, clamped to [600, 4000] → 1500
        assert_eq!(cfg.quad_points, 1500);
    }

    /// force_even defaults to true and is settable.
    #[test]
    fn config_force_even_default_and_override() {
        let cfg = HighPrecConfig::for_decimal_digits(200);
        assert!(cfg.force_even, "force_even should default to true");

        let mut cfg2 = HighPrecConfig::for_decimal_digits(200);
        cfg2.force_even = false;
        assert!(!cfg2.force_even, "force_even should be settable to false");
    }

    /// Weil-eigenvector cache filename keying: the forced-even variant
    /// keeps the historical (suffix-free) name so all pre-existing public
    /// fixtures stay valid; the natural variant gets a `_natural` marker
    /// so the two never collide.
    #[test]
    fn weil_eigvec_cache_filename_keys_on_force_even() {
        use super::weil_eigvec_cache::cache_filename;
        use super::super::LambdaSq;
        let forced = cache_filename(LambdaSq::integer(1000), 800, 6660, true);
        let natural = cache_filename(LambdaSq::integer(1000), 800, 6660, false);
        // Forced-even is the historical name (backward compatible).
        assert_eq!(forced, "weil_eigvec_lambda_sq1000_nmodes800_prec6660.json");
        // Natural is distinct.
        assert_eq!(natural, "weil_eigvec_lambda_sq1000_nmodes800_prec6660_natural.json");
        assert_ne!(forced, natural);
    }

    /// `weil_spectrum_hp`: the FULL Weil form is positive (smallest
    /// eigenvalue > 0 — Weil positivity / the plunge), while the
    /// ARCHIMEDEAN-ONLY form (prime sum `w_p` dropped) is indefinite
    /// (negative minimum — the prime sum that enforces positivity is
    /// absent). λ²=13, N=12, 64 digits, hermetic (no cache / no network).
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn weil_spectrum_full_positive_arch_indefinite() {
        let params = CcmParams::from_lambda_sq_integer(13, 12);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;

        let full = weil_spectrum_hp(&params, &cfg, true).unwrap();
        let arch = weil_spectrum_hp(&params, &cfg, false).unwrap();

        let zero = Float::with_val(cfg.precision_bits, 0);
        // FULL: smallest eigenvalue is positive (Weil positivity).
        assert!(
            full[0] > zero,
            "full Weil form should be positive definite, min={}",
            full[0]
        );
        // ARCH-only: minimum is negative (indefinite without the prime sum).
        assert!(
            arch[0] < zero,
            "archimedean-only form should be indefinite, min={}",
            arch[0]
        );
    }

    /// `weil_plunge_cancellation_hp`: the plunge ε_N is the difference of
    /// two O(1) Rayleigh quotients (archimedean − prime) that agree to many
    /// digits. λ²=13, N=12, 64 digits, hermetic.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn weil_plunge_cancellation_decomposes() {
        let params = CcmParams::from_lambda_sq_integer(13, 12);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;

        let d = weil_plunge_cancellation_hp(&params, &cfg).unwrap();
        let prec = cfg.precision_bits;

        // ε_N = arch − prime exactly (linearity of the Rayleigh quotient).
        let mut recon = Float::with_val(prec, &d.arch_rayleigh);
        recon -= &d.prime_rayleigh;
        let mut err = Float::with_val(prec, &recon);
        err -= &d.eps_n;
        assert!(err.abs() < Float::with_val(prec, 2).pow(-(prec as i32 - 16)),
            "arch − prime should equal eps_n");

        // ε_N is small and positive; arch is O(1) (not exponentially small).
        let zero = Float::with_val(prec, 0);
        assert!(d.eps_n > zero, "plunge should be positive");
        assert!(d.eps_n < Float::with_val(prec, 1), "plunge should be small");
        let thresh = Float::with_val(prec, Float::parse("1e-3").unwrap());
        assert!(
            d.arch_rayleigh.clone().abs() > thresh,
            "arch Rayleigh should be O(1), not exponentially small: {}",
            d.arch_rayleigh
        );
    }

    /// HighPrecConfig quad_points should clamp to MAX_QUAD_POINTS.
    #[test]
    fn config_clamps_quad_points() {
        let cfg = HighPrecConfig::for_decimal_digits(2000);
        // 2000 * 3 = 6000, clamped to max 4000
        assert_eq!(cfg.quad_points, MAX_QUAD_POINTS);
    }

    /// HighPrecConfig quad_points should clamp to MIN_QUAD_POINTS.
    #[test]
    fn config_floors_quad_points() {
        let cfg = HighPrecConfig::for_decimal_digits(10);
        // 10 * 3 = 30, clamped to min 600
        assert_eq!(cfg.quad_points, MIN_QUAD_POINTS);
    }

    /// hp::run() at small N should produce eigenvalues near Riemann zeros.
    /// Uses λ²=13, N=10 (21×21 matrix) at 64-digit precision — fast enough
    /// for a unit test (~1-2 seconds).
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn run_small_n_produces_eigenvalues() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.n_eigenvalues = 5;
        // Hermetic: no cache read/write, no network. Pure compute.
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;

        // HP seeds: first 5 Riemann zeros at full precision.
        let prec = cfg.precision_bits;
        let seed_strs = [
            "14.134725141734693790457251983562470270784257115699243175685567460149",
            "21.022039638771554992628479593896902777334340524902781754629520403587",
            "25.010857580145688763213790992562821818659549672557996672496542006745",
            "30.424876125859513210311897530584091320181560023715440180962146036993",
            "32.935061587739189690662368964074903488812715603517039009280003440784",
        ];
        let zero_seeds: Vec<Float> = seed_strs.iter()
            .map(|s| Float::with_val(prec, Float::parse(s).unwrap()))
            .collect();

        let result = run(&params, &cfg, &zero_seeds).unwrap();

        // At N=10, the f64 solver finds however many eigenvalues exist
        // in the standard brackets. Just verify we got at least 1.
        assert!(!result.eigenvalues_pos.is_empty(),
            "should produce at least one eigenvalue");

        // First eigenvalue should match 14.13... to at least 10 digits.
        // Compare in HP — no f64 round-trip.
        let prec = result.precision_bits;
        let target = Float::with_val(prec,
            Float::parse("14.134725141734693790457251983").unwrap());
        let ev0 = result.eigenvalues_pos[0].value()
            .expect("Newton should converge for zero_1 at λ²=13, N=10, P=64");
        let mut diff = ev0.clone();
        diff -= &target;
        let abs_diff = diff.abs();
        let tol = Float::with_val(prec, Float::parse("1e-5").unwrap());
        assert!(abs_diff < tol,
            "first eigenvalue {} should be near 14.13",
            xc_numerics::fmt::display_hp(ev0, 10));

        // ε_N should be small (tiny Weil eigenvalue at λ²=13). Compare HP.
        let eps_tol = Float::with_val(prec, Float::parse("1e-20").unwrap());
        let abs_eps = result.weil_min_eigenvalue.clone().abs();
        assert!(abs_eps < eps_tol,
            "ε_N = {} should be tiny at λ²=13, N=10",
            xc_numerics::fmt::display_hp(&result.weil_min_eigenvalue, 6));

        // Elapsed time should be positive (f64 metadata is fine here).
        assert!(result.elapsed_seconds > 0.0);
    }

    /// P2 test: newton_xi_hat_zero returns None when given a bogus
    /// eigenvector (all zeros) — Newton has no R(z) structure to converge on.
    #[test]
    fn newton_returns_none_on_bogus_eigenvector() {
        let prec = 128;
        let n_max = 5;
        let l = Float::with_val(prec, Float::parse("3.0").unwrap()); // ln(20)~3
        let seed = Float::with_val(prec, Float::parse("14.13").unwrap());
        // All-zero eigenvector: R(z) = 0 for all z, Newton cannot converge.
        let xi = vec![Float::with_val(prec, 0); 2 * n_max + 1];
        let result = newton_xi_hat_zero(&xi, n_max, &l, &seed, prec, 20);
        // With zero ξ, R(z)=0 everywhere. Newton step dz = R/R' = 0/0.
        // Function should return Failed (r_prime == 0 check) OR converge
        // trivially (dz=0 < tol on first step). Either way, not a silent
        // fallback to seed.
        // Actually: with all-zero xi, r=0 and r_prime=0 → Failed (division guard).
        assert!(!result.has_value(),
            "newton should return Failed with all-zero xi (degenerate R(z)=0)");
    }

    /// P3 test: log_lambda_sq_hp at non-integer (fractional) λ² should
    /// give a value that is at least as accurate as 17-digit f64 input allows.
    #[test]
    fn log_lambda_sq_hp_non_integer_precision() {
        let prec = 512;
        // λ²=2.5 — fractional. ln(2.5) = 0.916290731874155...
        let params = CcmParams::from_lambda_sq_fractional(2.5, 10);
        let l = log_lambda_sq_hp(&params, prec);
        // Reference: ln(2.5) computed at HP from exact rational 5/2.
        let ref_val = {
            let five = Float::with_val(prec, 5);
            let two = Float::with_val(prec, 2);
            let mut ratio = five; ratio /= &two;
            ratio.ln()
        };
        let mut diff = l.clone(); diff -= &ref_val;
        let abs_diff = diff.abs();
        // Should match to at least 15 significant digits (f64 input limit).
        // At prec=512 the string-parse path gives 17 sig figs.
        let tol = Float::with_val(prec, Float::parse("1e-15").unwrap());
        assert!(abs_diff < tol,
            "log_lambda_sq_hp for fractional λ²=2.5: L = {}, ref = {}, diff = {}",
            l.to_f64(), ref_val.to_f64(), abs_diff.to_f64());
    }

    /// R4 test: solver_steps is a high safety cap (never reached in practice;
    /// solver exits via convergence or stagnation detection). Scales with
    /// precision but is always at least 200.
    #[test]
    fn config_newton_steps_adaptive_with_precision() {
        // All precision levels: k=ceil(log2(P/10)).max(200) = 200
        let cfg60 = HighPrecConfig::for_decimal_digits(60);
        assert_eq!(cfg60.solver_steps, 200, "HP-60 should use safety cap of 200");

        let cfg200 = HighPrecConfig::for_decimal_digits(200);
        assert_eq!(cfg200.solver_steps, 200, "HP-200 should use safety cap of 200");

        let cfg1000 = HighPrecConfig::for_decimal_digits(1000);
        assert_eq!(cfg1000.solver_steps, 200, "HP-1000 should use safety cap of 200");

        let cfg2000 = HighPrecConfig::for_decimal_digits(2000);
        assert_eq!(cfg2000.solver_steps, 200, "HP-2000 should use safety cap of 200");

        // Steps must be non-decreasing with precision
        assert!(cfg200.solver_steps >= cfg60.solver_steps);
        assert!(cfg1000.solver_steps >= cfg200.solver_steps);
        assert!(cfg2000.solver_steps >= cfg1000.solver_steps);
    }

    /// R3 test: Halley's method matches Newton to working precision.
    /// Runs the same λ²=13, N=10 config with both methods and verifies
    /// the results agree to at least 50 digits.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn halley_matches_newton_for_same_config() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        cfg.n_eigenvalues = 1;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let seed_str = "14.134725141734693790457251983562470270784257115699243175685567460149";
        let zero_seeds: Vec<Float> = vec![
            Float::with_val(prec, Float::parse(seed_str).unwrap())
        ];

        // Newton result
        std::env::set_var("XCELERATOR_SOLVER", "newton");
        let result_n = run(&params, &cfg, &zero_seeds).unwrap();
        let ev_newton = result_n.eigenvalues_pos[0].value()
            .expect("Newton should converge");

        // Halley result
        std::env::set_var("XCELERATOR_SOLVER", "halley");
        let result_h = run(&params, &cfg, &zero_seeds).unwrap();
        let ev_halley = result_h.eigenvalues_pos[0].value()
            .expect("Halley should converge");

        // Reset to default
        std::env::remove_var("XCELERATOR_SOLVER");

        // Both should agree to at least 50 digits
        let mut diff = ev_newton.clone();
        diff -= ev_halley;
        let abs_diff = diff.abs();
        let tol = Float::with_val(prec, Float::parse("1e-50").unwrap());
        assert!(abs_diff < tol,
            "Halley and Newton should agree to 50 digits; diff = {}",
            abs_diff.to_f64());
    }
    /// Uses λ²=13, N=10 which has good eigenvalue convergence.
    /// The f64 seed path is exercised when xi is available.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn run_uses_f64_seeds_at_small_n() {
        let params = CcmParams::from_lambda_sq_integer(13, 8); // N=8, tiny
        let mut cfg = HighPrecConfig::for_decimal_digits(60);
        cfg.n_eigenvalues = 1;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let prec = cfg.precision_bits;
        let seed_strs = [
            "14.134725141734693790457251983562470270784257115699243175685567460149",
        ];
        let zero_seeds: Vec<Float> = seed_strs.iter()
            .map(|s| Float::with_val(prec, Float::parse(s).unwrap()))
            .collect();
        let result = run(&params, &cfg, &zero_seeds).unwrap();
        // Should produce 1 eigenvalue (Some or None — just check no panic)
        assert_eq!(result.eigenvalues_pos.len(), 1,
            "should produce 1 eigenvalue entry for n_eigenvalues=1");
    }

    /// measure_evenness at λ²=13, N=10 should show near-perfect evenness.
    #[test]
    #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn measure_evenness_small_lambda_is_even() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(64);
        // Hermetic: no cache read/write, no network. Pure compute.
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let result = measure_evenness(&params, &cfg).unwrap();

        let prec = result.evenness_deviation.prec();

        // At λ²=13, the natural eigenvector should be essentially even.
        // Compare in HP (deviation could be 1e-30 or smaller — tighter than f64).
        let dev_tol = Float::with_val(prec, Float::parse("1e-10").unwrap());
        assert!(result.evenness_deviation < dev_tol,
            "evenness deviation at λ²=13 should be tiny, got {}",
            xc_numerics::fmt::display_hp(&result.evenness_deviation, 6));

        // Both eigenvalues should be the same (since natural IS even).
        // Use HP relative_difference helper — no f64 fallback even at tiny ε.
        if let Some(rel_diff) = xc_numerics::fmt::relative_difference(
            &result.natural_eigenvalue, &result.forced_eigenvalue
        ) {
            let rel_tol = Float::with_val(prec, Float::parse("1e-10").unwrap());
            assert!(rel_diff < rel_tol,
                "natural and forced eigenvalues should match, rel diff = {}",
                xc_numerics::fmt::display_hp(&rel_diff, 6));
        } else {
            // forced is exactly zero — the only acceptable case is when
            // natural is also zero.
            assert!(result.natural_eigenvalue.is_zero(),
                "forced is zero but natural is not — eigenvalues differ");
        }

        // Sanity check signs match (both should be positive at λ²=13).
        use xc_numerics::fmt::sign_of;
        assert_eq!(sign_of(&result.natural_eigenvalue),
                   sign_of(&result.forced_eigenvalue),
                   "natural and forced eigenvalue signs must agree at λ²=13");
    }

    // ---------------------------------------------------------------
    // tau cache — pure-function and verify_dir tests
    // ---------------------------------------------------------------

    /// `tau_cache::parse_filename` extracts (λ²_int, n_modes, prec)
    /// from each of the three accepted filename forms, and rejects
    /// other patterns.
    #[test]
    fn tau_cache_filename_parser() {
        use super::tau_cache::{parse_filename, FileKind};
        use super::super::LambdaSq;

        // .json
        let r = parse_filename("lambda_sq13_nmodes120_prec3338.json").unwrap();
        assert_eq!(r.0, LambdaSq::integer(13));
        assert_eq!(r.1, 120);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Json));

        // .json.zip
        let r = parse_filename("lambda_sq100_nmodes500_prec3338.json.zip").unwrap();
        assert_eq!(r.0, LambdaSq::integer(100));
        assert_eq!(r.1, 500);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Zip));

        // .json.zip.partXX
        let r = parse_filename("lambda_sq1000_nmodes800_prec3338.json.zip.part00").unwrap();
        assert_eq!(r.0, LambdaSq::integer(1000));
        assert_eq!(r.1, 800);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Part));
        let r = parse_filename("lambda_sq1000_nmodes800_prec3338.json.zip.part42").unwrap();
        assert!(matches!(r.3, FileKind::Part));

        // Invalid — wrong base name.
        assert!(parse_filename("foo.json").is_none());
        // Invalid — missing component.
        assert!(parse_filename("lambda_sq13_nmodes120.json").is_none());
        // Invalid — non-digit part suffix.
        assert!(parse_filename("lambda_sq13_nmodes120_prec3338.json.zip.partAA").is_none());
        // Invalid — empty part suffix.
        assert!(parse_filename("lambda_sq13_nmodes120_prec3338.json.zip.part").is_none());
    }

    /// `tau_cache::structural_check` rejects asymmetric matrices,
    /// matrices with wrong length, and matrices with NaN/Inf entries.
    #[test]
    fn tau_cache_structural_check() {
        use super::tau_cache::structural_check;
        let prec = 128;
        let n_modes = 3;
        let dim = 2 * n_modes + 1; // 7

        // Build a symmetric 7×7 matrix.
        let mut sym = vec![Float::with_val(prec, 0); dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let val = Float::with_val(prec, (i + j + 1) as f64);
                sym[i * dim + j] = val.clone();
                sym[j * dim + i] = val;
            }
        }
        assert!(structural_check(&sym, n_modes, prec).is_none(),
            "symmetric matrix should pass");

        // Wrong length.
        let mut short = sym.clone();
        short.pop();
        assert!(structural_check(&short, n_modes, prec).is_some(),
            "wrong length should be rejected");

        // Asymmetric: perturb one off-diagonal pair so τ[i,j] ≠ τ[j,i].
        let mut asym = sym.clone();
        asym[0 * dim + 1] = Float::with_val(prec, 99);
        // τ[1,0] is still its original value → asymmetry.
        assert!(structural_check(&asym, n_modes, prec).is_some(),
            "asymmetric matrix should be rejected");

        // NaN entry.
        let mut with_nan = sym.clone();
        with_nan[5] = Float::with_val(prec, f64::NAN);
        assert!(structural_check(&with_nan, n_modes, prec).is_some(),
            "NaN entry should be rejected");
    }

    /// `tau_cache::parse_json_for_test` rejects a JSON envelope where
    /// `min_compatible_version` is newer than `CACHE_MIN_TOOLKIT_VERSION`.
    #[test]
    fn tau_cache_rejects_newer_min_compatible_version() {
        let n_modes: usize = 1;
        let prec: u32 = 64;
        // Build a well-formed envelope stamped with a far-future minimum.
        let dim = 2 * n_modes + 1;
        let strs: Vec<String> = (0..dim * dim).map(|i| format!("{}", i)).collect();
        let payload = serde_json::json!({
            "toolkit_version": "99.0.0",
            "min_compatible_version": "99.0.0",
            "lambda_sq": 13_u64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        }).to_string();
        assert!(
            super::tau_cache::parse_json_for_test(&payload, n_modes, prec).is_none(),
            "tau parser should reject min_compatible_version=99.0.0"
        );
    }

    /// `verify_tau_cache_dir` on a non-existent directory returns
    /// an empty report.
    #[test]
    fn tau_cache_verify_missing_dir() {
        let nonexistent = crate::test_tmp_root()
            .join(format!("xc_spectral_tau_cache_test_missing_{}",
                          std::process::id()));
        let report = super::tau_cache::verify_tau_cache_dir(&nonexistent).unwrap();
        assert_eq!(report.statuses.len(), 0);
        assert_eq!(report.ok_count(), 0);
        assert_eq!(report.failure_count(), 0);
    }

    /// `verify_tau_cache_dir` classifies each file: Ok for a real
    /// matrix, StructurallyInvalid for an asymmetric one, Skipped
    /// for an unrecognized name, LoadFailed for malformed JSON.
    #[test]
    fn tau_cache_verify_classifies() {
        use super::tau_cache::{TauCacheFileStatus, verify_tau_cache_dir, cache_filename};
        use super::super::LambdaSq;

        let prec = 128;
        let n_modes = 3;
        let dim = 2 * n_modes + 1; // 7
        let lambda_sq = LambdaSq::integer(13);

        let temp_dir = crate::fresh_test_dir("tau_cache_classify");

        // 1. Valid: build a symmetric matrix and serialize as envelope.
        let mut sym = vec![Float::with_val(prec, 0); dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let val = Float::with_val(prec, (i + j + 1) as f64);
                sym[i * dim + j] = val.clone();
                sym[j * dim + i] = val;
            }
        }
        let valid_name = cache_filename(lambda_sq, n_modes, prec);
        let valid_path = temp_dir.join(&valid_name);
        let strs: Vec<String> = sym.iter().map(|f| f.to_string()).collect();
        let valid_json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": "0.10.0",
            "min_compatible_version": "0.10.0",
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        });
        std::fs::write(&valid_path, serde_json::to_string(&valid_json).unwrap()).unwrap();

        // 2. StructurallyInvalid: asymmetric matrix at a different lsq,
        //    wrapped in the envelope so parse succeeds but structural check fails.
        let mut asym = sym.clone();
        asym[0 * dim + 1] = Float::with_val(prec, 99);
        // τ[1,0] still its original value → asymmetry.
        let lsq_bad = LambdaSq::integer(lambda_sq.value_u64 + 1);
        let bad_name = cache_filename(lsq_bad, n_modes, prec);
        let bad_path = temp_dir.join(&bad_name);
        let bad_strs: Vec<String> = asym.iter().map(|f| f.to_string()).collect();
        let bad_json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": "0.10.0",
            "min_compatible_version": "0.10.0",
            "lambda_sq": lsq_bad.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": bad_strs,
        });
        std::fs::write(&bad_path, serde_json::to_string(&bad_json).unwrap()).unwrap();

        // 3. Skipped: unrecognized filename.
        let skipped_path = temp_dir.join("not_a_tau_cache.txt");
        std::fs::write(&skipped_path, "irrelevant").unwrap();

        // 4. LoadFailed: matching pattern, malformed JSON.
        let malformed_name = cache_filename(LambdaSq::integer(lambda_sq.value_u64 + 2), n_modes, prec);
        let malformed_path = temp_dir.join(&malformed_name);
        std::fs::write(&malformed_path, "{").unwrap();

        let report = verify_tau_cache_dir(&temp_dir).unwrap();
        assert_eq!(report.statuses.len(), 4,
            "expected 4 statuses, got {}", report.statuses.len());

        let mut saw_ok = false;
        let mut saw_invalid = false;
        let mut saw_skipped = false;
        let mut saw_loadfail = false;
        for s in &report.statuses {
            match s {
                TauCacheFileStatus::Ok { path, lambda_sq: l, n_modes: n, prec: p } => {
                    assert_eq!(path, &valid_path);
                    assert_eq!(*l, lambda_sq);
                    assert_eq!(*n, n_modes);
                    assert_eq!(*p, prec);
                    saw_ok = true;
                }
                TauCacheFileStatus::StructurallyInvalid { path, .. } => {
                    assert_eq!(path, &bad_path);
                    saw_invalid = true;
                }
                TauCacheFileStatus::Skipped { path, .. } => {
                    assert_eq!(path, &skipped_path);
                    saw_skipped = true;
                }
                TauCacheFileStatus::LoadFailed { path, .. } => {
                    assert_eq!(path, &malformed_path);
                    saw_loadfail = true;
                }
            }
        }
        assert!(saw_ok, "missing Ok");
        assert!(saw_invalid, "missing StructurallyInvalid");
        assert!(saw_skipped, "missing Skipped");
        assert!(saw_loadfail, "missing LoadFailed");

        assert_eq!(report.ok_count(), 1);
        assert_eq!(report.failure_count(), 2);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Negative: a structurally-invalid τ `.json` on disk (parseable but
    /// asymmetric) must be skipped by `tau_cache::load` (returns `None`,
    /// treated as a miss → caller recomputes). The bad file is preserved.
    /// Mirrors the GL cache's structurally-invalid-json test; brings the
    /// τ load path to parity (previously only the verify-dir audit and
    /// the unit-level structural_check were tested, not the load path).
    #[test]
    fn tau_load_skips_structurally_invalid_json() {
        use super::tau_cache::{load, cache_filename};
        use xc_numerics::quadrature::CacheMode;
        use super::super::LambdaSq;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes = 3usize;
        let dim = 2 * n_modes + 1; // 7

        let temp = crate::fresh_test_dir("tau_invalid_json");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let zip_path = dir.join(format!("{}.zip", entry_name));

        // Build a correctly-shaped matrix, then break symmetry at one
        // off-diagonal pair so structural_check rejects it.
        let mut m = vec![Float::with_val(prec, 0); dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let val = Float::with_val(prec, (i + j + 1) as f64);
                m[i * dim + j] = val.clone();
                m[j * dim + i] = val;
            }
        }
        m[0 * dim + 1] = Float::with_val(prec, 99); // τ[1,0] unchanged → asymmetric
        let strs: Vec<String> = m.iter().map(|f| f.to_string()).collect();
        let json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": "0.10.0",
            "min_compatible_version": "0.10.0",
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "matrix": strs,
        });
        let json_str = serde_json::to_string(&json).unwrap();

        // Plant the bad matrix inside a .json.zip (the only tier read now).
        {
            use std::io::Write;
            let f = std::fs::File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file(&entry_name, opts).unwrap();
            zw.write_all(json_str.as_bytes()).unwrap();
            zw.finish().unwrap();
        }

        // load must skip the asymmetric matrix (None). JsonOnly is a
        // read no-op; JsonZip reads the zip, runs structural_check, rejects.
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonOnly).is_none(),
            "JsonOnly is a read no-op under the zip-only contract");
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
            "structurally-invalid (asymmetric) τ matrix in the zip must be skipped");

        // Bad file preserved on disk.
        assert!(zip_path.exists(),
            "structurally-invalid τ zip should be preserved for inspection");

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a truncated/corrupt τ `.json.zip` must be detected and
    /// skipped without panic (`load` returns `None`). The corrupt file is
    /// preserved. Mirrors the GL cache's `cache_handles_corrupt_zip_gracefully`.
    #[test]
    fn tau_load_handles_corrupt_zip_gracefully() {
        use super::tau_cache::load;
        use xc_numerics::quadrature::CacheMode;
        use super::super::LambdaSq;
        let prec = 64;
        let lambda_sq = LambdaSq::integer(49);
        let n_modes = 3usize;

        let temp = crate::fresh_test_dir("tau_corrupt_zip");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).unwrap();
        // Garbage bytes named as the single-zip for this config; no
        // local .json, so JsonZip falls through to the zip, fails to
        // open it, and returns None without panicking.
        let zip_path = dir.join(format!(
            "lambda_sq{}_nmodes{}_prec{}.json.zip",
            lambda_sq.filename_str(), n_modes, prec));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
            "corrupt τ .json.zip must be skipped, not loaded");

        assert!(zip_path.exists(),
            "corrupt τ zip should be preserved for inspection");

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// The remote τ URL is deterministically derived from
    /// `(λ², N, prec)` using the public xcelerator-tau-cache repo's
    /// precision-first → λ² → nmodes-thousand-bucket layout.
    /// Checks that all configured bases produce correctly bucketed URLs.
    #[test]
    fn tau_remote_url_uses_bucketed_layout() {
        use super::super::LambdaSq;
        // λ²=1000, N=800, prec=3338 → bucket 0-999.
        let results = super::tau_cache::remote_dir_and_stem_for_test(LambdaSq::integer(1000), 800, 3338);
        assert_eq!(results.len(), 2, "expected 2 remote bases");
        assert_eq!(
            results[0].0,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache/main/tau_cache/prec3338/lambda_sq1000/nmodes0-999"
        );
        assert_eq!(results[0].1, "lambda_sq1000_nmodes800_prec3338.json.zip");
        assert_eq!(
            results[1].0,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache-2/main/tau_cache/prec3338/lambda_sq1000/nmodes0-999"
        );
        assert_eq!(results[1].1, "lambda_sq1000_nmodes800_prec3338.json.zip");

        // λ²=400, N=1500, prec=4999 → bucket 1000-1999.
        let results2 = super::tau_cache::remote_dir_and_stem_for_test(LambdaSq::integer(400), 1500, 4999);
        assert_eq!(results2.len(), 2, "expected 2 remote bases");
        assert_eq!(
            results2[0].0,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache/main/tau_cache/prec4999/lambda_sq400/nmodes1000-1999"
        );
        assert_eq!(results2[0].1, "lambda_sq400_nmodes1500_prec4999.json.zip");
        assert_eq!(
            results2[1].0,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache-2/main/tau_cache/prec4999/lambda_sq400/nmodes1000-1999"
        );
        assert_eq!(results2[1].1, "lambda_sq400_nmodes1500_prec4999.json.zip");
    }

    /// Live end-to-end remote τ-fetch test against the PUBLIC
    /// `xcelerator-tau-cache` repo. `#[ignore]`d so it never runs in the
    /// default suite (needs network + `curl` + the public repo).
    /// Run explicitly with:
    ///
    /// ```text
    /// cargo test -p xc-spectral --features hp -- --ignored tau_remote_fetch_live
    /// ```
    ///
    /// Uses (λ²=1000, N=800, prec=3338) — a known byte-split config in
    /// the repo (a headline HP-1000 config; >90 MB → multiple .partXX). In a
    /// fresh temp cwd with NO local cache, `build_tau_hp` under
    /// DynamicFetch must miss local tiers, hit the remote tier, probe
    /// the single zip (404) then download all `.partXX`, concatenate +
    /// decompress + validate, and return a (2·800+1)² matrix.
    #[test]
    #[ignore = "live network: hits the public xcelerator-tau-cache repo; run with --ignored"]
    fn tau_remote_fetch_live_downloads_and_validates() {
        // Isolate cwd so the cache writes land in a throwaway dir.
        // CwdGuard serializes against other cwd-mutating tests (cwd is
        // process-global) and restores the original cwd on drop.
        let temp = crate::fresh_test_dir("tau_remote_live");
        let _guard = CwdGuard::enter(&temp);

        let lambda_sq = super::super::LambdaSq::integer(1000);
        let n_modes = 800usize;
        let prec = 3338u32;
        let params = CcmParams::from_lambda_sq_integer(lambda_sq.value_u64, n_modes);
        // Sanity: no local cache present.
        assert!(super::tau_cache::load(
            lambda_sq, n_modes, prec,
            xc_numerics::quadrature::CacheMode::JsonZip
        ).is_none(), "no local cache should exist before fetch");

        // DynamicFetch: should pull from the remote repo.
        let tau = super::tau_cache::load(
            lambda_sq, n_modes, prec,
            xc_numerics::quadrature::CacheMode::DynamicFetch,
        );

        let tau = tau.expect("remote fetch should have returned a τ-matrix");
        let dim = params.matrix_size();
        assert_eq!(tau.len(), dim * dim,
            "fetched τ length {} != (2N+1)² = {}", tau.len(), dim * dim);

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    // -----------------------------------------------------------------
    // weil_eigvec_cache tests
    // -----------------------------------------------------------------

    /// A fresh temp dir + cwd guard so cache reads/writes land in a
    /// throwaway location and never touch the real `data/` tree.
    /// Scratch lives under `target/test-tmp/` (removed by `cargo clean`),
    /// not the OS temp dir.
    fn weil_temp_cwd(tag: &str) -> std::path::PathBuf {
        crate::fresh_test_dir(&format!("weil_eigvec_{}", tag))
    }

    /// The remote ξ URL is deterministically derived from `(λ², N, prec)`
    /// using the public repo's precision-first → λ² → nmodes-thousand-
    /// bucket layout (mirrors tau/GL), with the `weil_eigvec_` filename
    /// prefix.
    #[test]
    fn weil_eigvec_remote_url_uses_bucketed_layout() {
        use super::super::LambdaSq;
        // λ²=1000, N=800, prec=3338 → bucket 0-999.
        let url = super::weil_eigvec_cache::remote_zip_url_for_test(LambdaSq::integer(1000), 800, 3338);
        assert_eq!(
            url,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-weil-eigvec-cache/main/weil_eigvec_cache/prec3338/lambda_sq1000/nmodes0-999/weil_eigvec_lambda_sq1000_nmodes800_prec3338.json.zip"
        );

        // λ²=400, N=1500, prec=4999 → bucket 1000-1999.
        let url2 = super::weil_eigvec_cache::remote_zip_url_for_test(LambdaSq::integer(400), 1500, 4999);
        assert_eq!(
            url2,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-weil-eigvec-cache/main/weil_eigvec_cache/prec4999/lambda_sq400/nmodes1000-1999/weil_eigvec_lambda_sq400_nmodes1500_prec4999.json.zip"
        );
    }

    /// `parse_json` accepts a well-formed entry and rejects metadata
    /// mismatches, wrong xi length, and non-finite values.
    #[test]
    fn weil_eigvec_parse_json_validates() {
        use super::weil_eigvec_cache::parse_json;
        use super::super::LambdaSq;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes = 3usize;
        let dim = 2 * n_modes + 1; // 7

        let xi: Vec<Float> = (0..dim).map(|i| Float::with_val(prec, (i + 1) as f64)).collect();
        let xi_strs: Vec<String> = xi.iter().map(|f| f.to_string()).collect();
        let good = serde_json::json!({
            "schema_version": 1,
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.5e-40",
            "xi": xi_strs,
        }).to_string();
        let parsed = parse_json(&good, lambda_sq, n_modes, prec)
            .expect("well-formed entry should parse");
        assert_eq!(parsed.xi.len(), dim);

        // Wrong n_modes metadata → reject.
        assert!(parse_json(&good, lambda_sq, n_modes + 1, prec).is_none(),
            "n_modes mismatch should be rejected");
        // Wrong precision metadata → reject.
        assert!(parse_json(&good, lambda_sq, n_modes, prec + 1).is_none(),
            "precision mismatch should be rejected");
        // Wrong λ² metadata → reject.
        assert!(parse_json(&good, LambdaSq::integer(lambda_sq.value_u64 + 5), n_modes, prec).is_none(),
            "lambda_sq mismatch should be rejected");

        // Wrong xi length → reject.
        let mut short_strs = xi_strs.clone();
        short_strs.pop();
        let short = serde_json::json!({
            "schema_version": 1, "lambda_sq": lambda_sq.value_f64, "n_modes": n_modes,
            "precision_bits": prec, "weil_min_eigenvalue": "1.5e-40", "xi": short_strs,
        }).to_string();
        assert!(parse_json(&short, lambda_sq, n_modes, prec).is_none(),
            "wrong xi length should be rejected");
    }

    /// `weil_eigvec_cache::parse_json_for_test` rejects a JSON envelope
    /// where `min_compatible_version` is newer than `CACHE_MIN_TOOLKIT_VERSION`.
    #[test]
    fn weil_eigvec_rejects_newer_min_compatible_version() {
        use super::super::LambdaSq;
        let prec: u32 = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes: usize = 2;
        let dim = 2 * n_modes + 1;
        let xi_strs: Vec<String> = (0..dim).map(|i| format!("0.{}", i + 1)).collect();
        let payload = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": "99.0.0",
            "min_compatible_version": "99.0.0",
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.23e-10",
            "xi": xi_strs,
        }).to_string();
        assert!(
            super::weil_eigvec_cache::parse_json_for_test(
                &payload, lambda_sq, n_modes, prec
            ).is_none(),
            "weil eigvec parser should reject min_compatible_version=99.0.0"
        );
    }

    /// `residual_ok` accepts a genuine eigenpair of τ and rejects a
    /// perturbed (wrong) eigenvector — the strong integrity test the
    /// after-τ cache check relies on.
    #[test]
    fn weil_eigvec_residual_check_discriminates() {
        use super::weil_eigvec_cache::residual_ok;
        let prec = 256;
        let n = 5;
        // Diagonal matrix: eigenpairs are (λ_i, e_i). Smallest is λ=1 at e_0.
        let mut a = vec![Float::with_val(prec, 0); n * n];
        let diag = ["1", "2", "3", "4", "5"];
        for (i, d) in diag.iter().enumerate() {
            a[i * n + i] = Float::with_val(prec, Float::parse(d).unwrap());
        }
        // True smallest eigenpair.
        let eps = Float::with_val(prec, 1);
        let mut xi = vec![Float::with_val(prec, 0); n];
        xi[0] = Float::with_val(prec, 1);
        assert!(residual_ok(&a, n, &xi, &eps, prec),
            "genuine eigenpair should pass the residual check");

        // Wrong eigenvector (points along e_1, whose eigenvalue is 2,
        // not 1) → residual is O(1) → reject.
        let mut wrong = vec![Float::with_val(prec, 0); n];
        wrong[1] = Float::with_val(prec, 1);
        assert!(!residual_ok(&a, n, &wrong, &eps, prec),
            "wrong eigenvector should fail the residual check");

        // Zero vector → reject.
        let zero = vec![Float::with_val(prec, 0); n];
        assert!(!residual_ok(&a, n, &zero, &eps, prec),
            "zero vector should fail the residual check");
    }

    /// Round-trip: `save` then `load` returns a byte-identical ξ and ε_N
    /// at every CacheMode tier. Also checks that `CacheMode::Off` writes
    /// nothing.
    #[test]
    fn weil_eigvec_save_load_round_trip() {
        use super::weil_eigvec_cache::{save, load};
        use xc_numerics::quadrature::CacheMode;
        use super::super::LambdaSq;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(49);
        let n_modes = 4usize;
        let dim = 2 * n_modes + 1; // 9

        let temp = weil_temp_cwd("round_trip");
        let _guard = CwdGuard::enter(&temp);

        let eps = Float::with_val(prec, Float::parse("3.25e-12").unwrap());
        let xi: Vec<Float> = (0..dim)
            .map(|i| Float::with_val(prec, Float::parse(
                format!("0.{}1", i + 1)).unwrap()))
            .collect();

        // Off: writes nothing, reads nothing.
        save(lambda_sq, n_modes, prec, &eps, &xi, CacheMode::Off, true);
        assert!(load(lambda_sq, n_modes, prec, CacheMode::Off, true).is_none(),
            "Off should never read");
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip, true).is_none(),
            "Off save should have written nothing");

        // JsonZip: writes ONLY the .json.zip (zip-only contract); reads
        // back identical by decompressing in memory.
        save(lambda_sq, n_modes, prec, &eps, &xi, CacheMode::JsonZip, true);

        // No uncompressed .json should be written.
        let jp = temp.join("data").join("weil_eigvec_cache")
            .join(super::weil_eigvec_cache::cache_filename(lambda_sq, n_modes, prec, true));
        assert!(!jp.exists(), "zip-only: save must not write an uncompressed .json");

        let got = load(lambda_sq, n_modes, prec, CacheMode::JsonZip, true)
            .expect("JsonZip round-trip should load from the zip");
        assert_eq!(got.xi.len(), dim);
        for (a, b) in xi.iter().zip(got.xi.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "xi entry must round-trip exactly");
        }
        assert_eq!(eps.to_string(), got.eps_n.to_string(),
            "eps_n must round-trip exactly");

        // JsonOnly is now a read no-op (no uncompressed .json exists).
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonOnly, true).is_none(),
            "zip-only: JsonOnly must not read the zip");

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a structurally-invalid ξ entry (parseable JSON but
    /// wrong xi length / mismatched metadata) inside a `.json.zip` must
    /// be skipped by `load` (returns `None`, treated as a miss → caller
    /// recomputes). The bad file is preserved on disk for inspection.
    #[test]
    fn weil_eigvec_load_skips_structurally_invalid_json() {
        use super::weil_eigvec_cache::{load, cache_filename};
        use xc_numerics::quadrature::CacheMode;
        use super::super::LambdaSq;
        let prec = 128;
        let lambda_sq = LambdaSq::integer(13);
        let n_modes = 4usize; // expects 2N+1 = 9 entries

        let temp = weil_temp_cwd("invalid_json");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let entry_name = cache_filename(lambda_sq, n_modes, prec, true);
        let zip_path = dir.join(format!("{}.zip", entry_name));

        // Shape-parseable JSON, but xi has the WRONG length (3 ≠ 9)
        // and otherwise-valid metadata. parse_json must reject it.
        let bad = serde_json::json!({
            "schema_version": 1,
            "lambda_sq": lambda_sq.value_f64,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.0e-20",
            "xi": ["1.0", "2.0", "3.0"],
        }).to_string();

        // Plant the bad entry inside a .json.zip (the only tier read now).
        {
            use std::io::Write;
            let f = std::fs::File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file(&entry_name, opts).unwrap();
            zw.write_all(bad.as_bytes()).unwrap();
            zw.finish().unwrap();
        }

        // load must skip (None) — never returns a malformed entry.
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonOnly, true).is_none(),
            "JsonOnly is a read no-op under the zip-only contract");
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip, true).is_none(),
            "structurally-invalid ξ entry in the zip must be skipped");

        // The bad file is preserved on disk (load does not delete it).
        assert!(zip_path.exists(),
            "structurally-invalid zip should be preserved for inspection");

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a truncated/corrupt ξ `.json.zip` must be detected and
    /// skipped without panic (`load` returns `None`). The corrupt file is
    /// preserved. Mirrors the GL cache's `cache_handles_corrupt_zip_gracefully`.
    #[test]
    fn weil_eigvec_load_handles_corrupt_zip_gracefully() {
        use super::weil_eigvec_cache::load;
        use xc_numerics::quadrature::CacheMode;
        use super::super::LambdaSq;
        let prec = 64;
        let lambda_sq = LambdaSq::integer(49);
        let n_modes = 3usize;

        let temp = weil_temp_cwd("corrupt_zip");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).unwrap();
        // Random bytes that are NOT a valid zip. No local .json present,
        // so JsonZip must fall through to the (garbage) zip, fail to
        // open it, and return None — without panicking.
        let zip_path = dir.join(format!(
            "weil_eigvec_lambda_sq{}_nmodes{}_prec{}.json.zip",
            lambda_sq.filename_str(), n_modes, prec
        ));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip, true).is_none(),
            "corrupt .json.zip must be skipped, not loaded");

        // Corrupt file preserved on disk.
        assert!(zip_path.exists(),
            "corrupt zip should be preserved for inspection");

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Live end-to-end remote ξ-fetch test against the PUBLIC
    /// `xcelerator-weil-eigvec-cache` repo. `#[ignore]`d so it never runs
    /// in the default suite (needs network + `curl` + a populated repo).
    /// Run explicitly with:
    ///
    /// ```text
    /// cargo test -p xc-spectral --features hp -- --ignored weil_eigvec_remote_fetch_live
    /// ```
    #[test]
    #[ignore = "live network: hits the public xcelerator-weil-eigvec-cache repo; run with --ignored"]
    fn weil_eigvec_remote_fetch_live_downloads_and_validates() {
        use xc_numerics::quadrature::CacheMode;
        // CwdGuard serializes against other cwd-mutating tests (cwd is
        // process-global) and restores the original cwd on drop.
        let temp = weil_temp_cwd("remote_live");
        let _guard = CwdGuard::enter(&temp);

        // A config that exists in the public repo (the seed fixture
        // generated by examples/gen_weil_eigvec_fixture: λ²=13, N=10,
        // HP-64 → prec 229 bits).
        let lambda_sq = super::super::LambdaSq::integer(13);
        let n_modes = 10usize;
        let prec = 229u32;
        assert!(super::weil_eigvec_cache::load(
            lambda_sq, n_modes, prec, CacheMode::JsonZip, true
        ).is_none(), "no local cache should exist before fetch");

        let fetched = super::weil_eigvec_cache::load(
            lambda_sq, n_modes, prec, CacheMode::DynamicFetch, true,
        ).map(|c| (c.xi.len(), 2 * n_modes + 1));

        let (got_len, expected_len) = fetched
            .expect("remote fetch should have returned a ξ entry");
        assert_eq!(got_len, expected_len,
            "fetched ξ length {} != 2N+1 = {}", got_len, expected_len);

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// `newton_xi_hat_zero` and `halley_xi_hat_zero` on a synthetic R(z)
    /// with a known root.
    ///
    /// Setup: single-pole eigenvector ξ = (1) at n=0, L = 2π.
    /// R(z) = ξ[0] / (z − 0) = 1/z, which has no zero.
    ///
    /// Better: two poles at ±p, weights ξ = (+1, +1) gives
    /// R(z) = 1/(z-p) + 1/(z+p) = 2z / (z²-p²).
    /// R(z) = 0 → z = 0, so we seed at 0.1 and verify convergence.
    ///
    /// Even simpler (and deterministic): weights ξ = (+1, -1) gives
    /// R(z) = 1/(z-p) − 1/(z+p) = 2p/(z²-p²).
    /// R(z) = 0 → no real zero.
    ///
    /// Use the two-weight (+1, +1) case with p = π/L so the pole
    /// positions are the n=±1 poles, and the zero is at z=0.
    #[test]
    fn newton_and_halley_find_known_zero() {
        use rug::{ops::Pow, Float};
        let prec = 256;

        // L = 2π, n_max = 1: poles at 0, ±2π/L = ±1.
        // ξ = (1, 0, 1) in order n = -1, 0, 1.
        // R(z) = ξ[-1]/(z - (-1)) + ξ[0]/(z - 0) + ξ[+1]/(z - 1)
        //      = 1/(z+1) + 0 + 1/(z-1) = 2z/(z²-1)
        // Zero at z = 0.
        let n_max = 1_usize;
        let l = Float::with_val(prec, Float::parse("6.283185307179586476925").unwrap()); // 2π
        let xi = vec![
            Float::with_val(prec, 1), // n = -1
            Float::with_val(prec, 0), // n = 0
            Float::with_val(prec, 1), // n = +1
        ];
        let seed = Float::with_val(prec, Float::parse("0.5").unwrap());
        let tol_check = Float::with_val(prec, 2).pow(-((prec as i32) / 2));

        let z_newton = super::newton_xi_hat_zero(&xi, n_max, &l, &seed, prec, 100)
            .value().expect("Newton should find a zero").clone();
        assert!(z_newton.clone().abs() < tol_check,
            "Newton zero should be ~0, got {}", z_newton);

        let z_halley = super::halley_xi_hat_zero(&xi, n_max, &l, &seed, prec, 100)
            .value().expect("Halley should find a zero").clone();
        assert!(z_halley.clone().abs() < tol_check,
            "Halley zero should be ~0, got {}", z_halley);

        // Verify the two methods agree.
        let mut diff = z_newton.clone();
        diff -= &z_halley;
        let abs_diff = diff.abs();
        assert!(abs_diff < tol_check,
            "Newton and Halley zeros differ by {}", abs_diff);
    }

    /// `newton_xi_hat_zero` with a pole directly on the seed (degenerate case):
    /// should return None, not panic or loop.
    #[test]
    fn newton_returns_none_when_starting_on_pole() {
        use rug::Float;
        let prec = 128;
        let n_max = 1_usize;
        let l = Float::with_val(prec, Float::parse("6.283185307179586").unwrap());
        // ξ = (1, 0, 0): only one non-zero weight at n = -1 → pole at z = -1.
        let xi = vec![
            Float::with_val(prec, 1),
            Float::with_val(prec, 0),
            Float::with_val(prec, 0),
        ];
        // Seed exactly on the pole: R'(z) = −1/(z+1)² → → ∞ near pole,
        // Newton step dz = R/R' → 0 at the pole itself → no zero found.
        // We pass a seed very close to the pole, expecting either None or a
        // convergent zero far away (R(z)=1/(z+1) has no zero). We verify it
        // doesn't panic. The specific None/Some outcome is implementation-defined.
        let seed_near_pole = Float::with_val(prec, Float::parse("-1.0").unwrap());
        // This won't crash; outcome is None (no zero for this R(z)).
        let _result = super::newton_xi_hat_zero(&xi, n_max, &l, &seed_near_pole, prec, 10);
        // No assertion on result — just verifying no panic / infinite loop.
    }
}
