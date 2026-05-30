// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision CCM tier via `rug` (MPFR/GMP).
//!
//! Strategy:
//! - Build the (2N+1)×(2N+1) Weil form matrix at user-chosen precision.
//! - Find smallest eigenpair by inverse iteration (from xc-numerics).
//! - Solve spectrum equation by Newton's method.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rug::{ops::Pow, Float};
use std::time::Instant;

use super::{prime_powers_up_to, CcmParams, CcmResult};

/// Configuration for the high-precision tier.
///
/// All fields are public so callers can override individual components
/// after a `for_decimal_digits` construction. The default values are
/// tuned for the Paper A / Paper B HP-1000 retest workload; tweak with
/// caution.
#[derive(Debug, Clone)]
pub struct HighPrecConfig {
    /// MPFR working precision in bits. Total digits ≈ `precision_bits / 3.322`.
    /// Use `for_decimal_digits` to construct from a target decimal precision.
    pub precision_bits: u32,
    /// Maximum number of inverse-iteration steps for the smallest
    /// Weil-form eigenvector recovery.
    pub inverse_iter_steps: usize,
    /// Maximum number of Newton-refinement steps per Riemann-zero seed.
    pub newton_steps: usize,
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
    /// the defaults `inverse_iter_steps=200, newton_steps=20,
    /// n_eigenvalues=50`.
    pub fn for_decimal_digits(digits: u32) -> Self {
        let bits = ((digits as f64) * DIGITS_TO_BITS_FACTOR).ceil() as u32 + GUARD_BITS;
        Self {
            precision_bits: bits,
            inverse_iter_steps: 200,
            newton_steps: 20,
            quad_points: ((digits as usize) * QUAD_POINTS_PER_DIGIT)
                .max(MIN_QUAD_POINTS)
                .min(MAX_QUAD_POINTS),
            n_eigenvalues: 50,
            cache_mode: xc_numerics::quadrature::CacheMode::default(),
        }
    }
}

/// Result of a single high-precision CCM run.
///
/// All HP fields stay in `rug::Float` at the working precision specified
/// in the config; lossy f64 views are exposed via `to_f64_result`.
pub struct HighPrecResult {
    /// Newton-refined positive eigenvalues, sorted in the order returned
    /// by Newton (i.e. paired with the Riemann-zero seeds in the same
    /// order). Length `≤ cfg.n_eigenvalues`.
    pub eigenvalues_pos: Vec<Float>,
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
            eigenvalues_pos: self.eigenvalues_pos.iter().map(|f| f.to_f64()).collect(),
            weil_min_eigenvalue: self.weil_min_eigenvalue.to_f64(),
            xi: self.xi.iter().map(|f| f.to_f64()).collect(),
            elapsed_seconds: self.elapsed_seconds,
        }
    }

    /// Save ξ_λ to JSON as full-precision decimal strings + metadata.
    pub fn save_xi_json(&self, params: &CcmParams, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let xi_strings: Vec<String> = self.xi.iter().map(|f| f.to_string()).collect();
        let payload = serde_json::json!({
            "schema_version": 1,
            "lambda_squared": params.lambda_squared,
            "n_modes": params.n_modes,
            "precision_bits": self.precision_bits,
            "weil_min_eigenvalue": self.weil_min_eigenvalue.to_string(),
            "elapsed_seconds": self.elapsed_seconds,
            "xi": xi_strings,
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
        Ok(())
    }
}

/// A ξ_λ vector loaded from disk.
///
/// Both HP and f64 views are provided. HP code paths should use
/// `xi_hp` and `weil_min_eigenvalue_hp`; the f64 fields exist for
/// f64-tier consumers and are explicitly lossy boundaries.
pub struct LoadedXi {
    /// `λ²` recorded in the file's metadata. Stored as f64 because
    /// the paper-config values are exact integers (13, 100, 1000).
    pub lambda_squared: f64,
    /// Mode cutoff `N`. Matrix dimension is `2N+1`.
    pub n_modes: usize,
    /// MPFR working precision the file was generated at, in bits.
    /// HP loaders should be called with the same `prec` argument so the
    /// strings round-trip exactly.
    pub precision_bits: u32,
    /// HP-precision smallest Weil eigenvalue. Use this in HP paths.
    pub weil_min_eigenvalue_hp: Float,
    /// Lossy f64 view of `weil_min_eigenvalue_hp`. May underflow f64
    /// at large λ; HP callers should ignore this field.
    pub weil_min_eigenvalue: f64,
    /// Lossy f64 view of `xi_hp`. Renormalized by ‖ξ‖_∞ for f64 safety.
    pub xi_f64: Vec<f64>,
    /// HP-precision eigenvector. Use this in HP paths.
    pub xi_hp: Vec<Float>,
}

/// Load a ξ_λ JSON file written by `HighPrecResult::save_xi_json` and
/// return both HP-precision and f64-truncated views.
///
/// The file's `precision_bits` field controls how the HP fields are
/// parsed. f64 fields are derived from the HP values via `.to_f64()`
/// (lossy at extreme magnitudes; ξ values are pre-normalized by
/// `‖ξ‖_∞` to keep the f64 view inside its representable range).
pub fn load_xi_json(path: &std::path::Path) -> Result<LoadedXi> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("could not read {}: {}", path.display(), e))?;
    let v: serde_json::Value = serde_json::from_str(&data)?;
    let lambda_squared = v.get("lambda_squared").and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow!("missing lambda_squared"))?;
    let n_modes = v.get("n_modes").and_then(|x| x.as_u64())
        .ok_or_else(|| anyhow!("missing n_modes"))? as usize;
    let precision_bits = v.get("precision_bits").and_then(|x| x.as_u64())
        .ok_or_else(|| anyhow!("missing precision_bits"))? as u32;
    let weil_min_str = v.get("weil_min_eigenvalue").and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("missing weil_min_eigenvalue"))?;
    let weil_min_hp = Float::with_val(precision_bits, Float::parse(weil_min_str)
        .map_err(|e| anyhow!("parse weil_min: {}", e))?);
    let xi_strings: Vec<String> = v.get("xi").and_then(|x| x.as_array())
        .ok_or_else(|| anyhow!("missing xi array"))?
        .iter()
        .filter_map(|s| s.as_str().map(|t| t.to_string()))
        .collect();
    if xi_strings.len() != 2 * n_modes + 1 {
        anyhow::bail!("xi length {} != 2N+1 = {}", xi_strings.len(), 2 * n_modes + 1);
    }
    let xi_hp: Vec<Float> = xi_strings.iter()
        .map(|s| Float::with_val(precision_bits, Float::parse(s).unwrap()))
        .collect();

    // Renormalize by ‖ξ‖_∞ for f64 safety (HP values can overflow f64).
    let mut xi_linf_hp = Float::with_val(precision_bits, 0.0);
    for v in &xi_hp {
        let abs = v.clone().abs();
        if abs > xi_linf_hp { xi_linf_hp = abs; }
    }
    let xi_f64: Vec<f64> = if xi_linf_hp.is_zero() {
        xi_hp.iter().map(|f| f.to_f64()).collect()
    } else {
        xi_hp.iter().map(|f| { let mut t = f.clone(); t /= &xi_linf_hp; t.to_f64() }).collect()
    };
    Ok(LoadedXi {
        lambda_squared,
        n_modes,
        precision_bits,
        weil_min_eigenvalue: weil_min_hp.to_f64(),
        weil_min_eigenvalue_hp: weil_min_hp,
        xi_f64,
        xi_hp,
    })
}

// -- Helpers
#[inline] fn fl_i(prec: u32, v: i64) -> Float { Float::with_val(prec, v) }
#[inline] fn pi(prec: u32) -> Float { Float::with_val(prec, rug::float::Constant::Pi) }
#[inline] fn euler(prec: u32) -> Float { Float::with_val(prec, rug::float::Constant::Euler) }

/// Top-level entry. Build matrix, find eigenvector, solve spectrum.
///
/// `zero_seeds` are the reference Riemann zero imaginary parts used as
/// Newton seeds. They should be at full working precision (decimal strings
/// parsed to `Float`) — NOT f64-truncated. Using f64 seeds causes Newton
/// divergence at high eigenvalue index.
pub fn run(params: &CcmParams, cfg: &HighPrecConfig, zero_seeds: &[Float]) -> Result<HighPrecResult> {
    let start = Instant::now();
    let prec = cfg.precision_bits;
    let dim = params.matrix_size();

    let l = Float::with_val(prec, params.lambda_squared).ln();
    let mut tau = build_tau_hp(params, &l, cfg)?;

    // Force exact symmetry. Compute averaged upper-triangle values in
    // parallel, then write back sequentially (each pair writes two
    // distinct mirror cells, so the write-phase is sequential to avoid
    // aliasing).
    let pairs: Vec<(usize, usize)> = (0..dim)
        .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j)))
        .collect();
    let symmetrized: Vec<(usize, usize, Float)> = pairs.par_iter().map(|&(i, j)| {
        let mut sum = tau[i * dim + j].clone();
        sum += &tau[j * dim + i];
        sum /= 2u32;
        (i, j, sum)
    }).collect();
    for (i, j, sum) in symmetrized {
        tau[i * dim + j] = sum.clone();
        tau[j * dim + i] = sum;
    }

    // Smallest eigenpair (ξ, ε_N).
    //
    // After-τ cache check: if a cached Weil eigenvector exists for this
    // (λ², N, prec) AND it validates against the in-hand τ via the
    // eigen-residual ‖τξ − μξ‖, skip the costly LU factorization
    // entirely. A missing or residual-failing entry falls through to a
    // fresh inverse-iteration compute, which is then cached. The check
    // sits *after* `build_tau_hp` precisely so τ is available for the
    // residual validation (the strongest integrity test for ξ).
    let lambda_sq = params.lambda_sq_int;
    let n_modes_key = params.n_modes;
    let mut cached_pair: Option<(Float, Vec<Float>)> = None;
    if let Some(c) = weil_eigvec_cache::load(lambda_sq, n_modes_key, prec, cfg.cache_mode) {
        if weil_eigvec_cache::residual_ok(&tau, dim, &c.xi, &c.eps_n, prec) {
            eprintln!(
                "[HP] loaded cached Weil eigenvector for λ²={}, N={}, prec={} bits \
                 (skipping LU; τ-residual validated)",
                lambda_sq, n_modes_key, prec
            );
            cached_pair = Some((c.eps_n, c.xi));
        } else {
            eprintln!(
                "[HP] WARNING: cached Weil eigenvector for λ²={}, N={}, prec={} failed \
                 τ-residual validation; recomputing",
                lambda_sq, n_modes_key, prec
            );
        }
    }
    let (eps_n, xi) = match cached_pair {
        Some(pair) => pair,
        None => {
            // Find smallest eigenpair by inverse iteration (forced even).
            eprintln!("[HP] LU factoring {}×{} matrix (one-time cost)...", dim, dim);
            let (eps_n, xi_raw) = xc_numerics::linalg::inverse_iteration(
                &tau, dim, prec, cfg.inverse_iter_steps, true)?;
            eprintln!("[HP] LU factorization done.");
            // Normalize: Σ ξ_j = √L.
            let xi = normalize_eigenvector(&xi_raw, &l, prec);
            weil_eigvec_cache::save(lambda_sq, n_modes_key, prec, &eps_n, &xi, cfg.cache_mode);
            (eps_n, xi)
        }
    };

    // Find eigenvalues as zeros of R(z), seeded from HP reference zeros.
    // Each Newton refinement is independent across seeds — parallelize.
    let n_eigs = cfg.n_eigenvalues.min(zero_seeds.len());
    let eigenvalues_pos: Vec<Float> = zero_seeds[..n_eigs]
        .par_iter()
        .map(|seed| {
            let seed_hp = Float::with_val(prec, seed);
            match newton_xi_hat_zero(&xi, params.n_modes, &l, &seed_hp, prec, cfg.newton_steps) {
                Some(z) => z,
                None => seed_hp,
            }
        })
        .collect();

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

    let l = Float::with_val(prec, params.lambda_squared).ln();
    let mut tau = build_tau_hp(params, &l, cfg)?;

    // Force exact symmetry of the matrix (parallel compute, sequential write).
    let pairs: Vec<(usize, usize)> = (0..dim)
        .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j)))
        .collect();
    let symmetrized: Vec<(usize, usize, Float)> = pairs.par_iter().map(|&(i, j)| {
        let mut sum = tau[i * dim + j].clone();
        sum += &tau[j * dim + i];
        sum /= 2u32;
        (i, j, sum)
    }).collect();
    for (i, j, sum) in symmetrized {
        tau[i * dim + j] = sum.clone();
        tau[j * dim + i] = sum;
    }

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
    let lambda_sq = params.lambda_sq_int;
    let n_modes = params.n_modes;

    if let Some(cached) = tau_cache::load(lambda_sq, n_modes, prec, cfg.cache_mode) {
        eprintln!(
            "[HP] loaded cached τ-matrix for λ²={}, N={}, prec={} bits ({}×{} = {} entries)",
            lambda_sq, n_modes, prec,
            params.matrix_size(), params.matrix_size(), cached.len()
        );
        return Ok(cached);
    }

    let tau = build_tau_hp_compute(params, l, cfg)?;
    tau_cache::save(lambda_sq, n_modes, prec, &tau, cfg.cache_mode);
    Ok(tau)
}

fn build_tau_hp_compute(params: &CcmParams, l: &Float, cfg: &HighPrecConfig) -> Result<Vec<Float>> {
    let prec = cfg.precision_bits;
    let n_max = params.n_modes;
    let dim = params.matrix_size();
    let lambda_sq_int = params.lambda_sq_int;

    let pi_v = pi(prec);
    let mut two_pi = pi_v.clone(); two_pi *= 2u32;
    let mut sixteen_pi2 = pi_v.clone().square(); sixteen_pi2 *= 16u32;
    let l_sq = l.clone().square();
    let sinh2_l_over_4 = { let mut v = l.clone(); v /= 4u32; v.sinh().square() };

    let base_pts = cfg.quad_points;
    let prec_extra = (cfg.precision_bits / 2) as usize;
    eprintln!("[HP] Computing α_L, β_L, γ_L for n=0..{} (base quad={})", n_max, base_pts);

    use std::collections::HashMap;
    let pts_for_n: Vec<usize> = (0..=n_max).map(|n| base_pts.max(3 * n + prec_extra)).collect();
    let unique_pts: Vec<usize> = {
        let mut v = pts_for_n.clone(); v.sort_unstable(); v.dedup(); v
    };
    eprintln!("[HP] Precomputing {} unique GL node tables...", unique_pts.len());
    let gl_cache: HashMap<usize, (Vec<Float>, Vec<Float>)> = unique_pts
        .par_iter()
        .map(|&npts| (npts, xc_numerics::quadrature::gauss_legendre_nodes(npts, prec, cfg.cache_mode)))
        .collect();
    eprintln!("[HP] GL tables cached. Computing integrals...");

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
    eprintln!("[HP] α, β, γ done. Assembling {}×{} matrix...", dim, dim);

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

// ===========================================================================
// Newton refinement of R(z) zeros
// ===========================================================================

fn newton_xi_hat_zero(
    xi: &[Float], n_max: usize, l: &Float, seed: &Float, prec: u32, n_steps: usize,
) -> Option<Float> {
    let two_pi_over_l = { let mut v = pi(prec); v *= 2u32; v /= l; v };
    let mut z = seed.clone();
    let tol = Float::with_val(prec, 2).pow(-((prec as i32) - 16));
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
        if r_prime.is_zero() { return None; }
        let mut dz = r; dz /= &r_prime;
        z -= &dz;
        if dz.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) { break; }
    }
    Some(z)
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

    /// Per-part byte cap for split files. 90 MB stays comfortably
    /// under GitHub's 100 MB hard file limit and well under the
    /// 50 MB soft warning, leaving headroom for git's pack overhead.
    const PART_BYTE_LIMIT: usize = 90 * 1024 * 1024;

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

    pub(super) fn cache_filename(lambda_sq: u64, n_modes: usize, prec: u32) -> String {
        format!("lambda_sq{}_nmodes{}_prec{}.json", lambda_sq, n_modes, prec)
    }

    fn json_path(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| d.join(cache_filename(lambda_sq, n_modes, prec)))
    }

    fn zip_path(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| {
            let f = cache_filename(lambda_sq, n_modes, prec);
            d.join(format!("{}.zip", f))
        })
    }

    /// Glob the .partXX files for a given config in lexicographic
    /// order. Returns `None` if no parts exist.
    fn part_paths(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<Vec<std::path::PathBuf>> {
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

    /// Parse the cache JSON: a single JSON array of decimal strings,
    /// length `(2N+1)²`, each parseable at the requested precision.
    /// Returns `None` on any structural mismatch.
    fn parse_json(data: &str, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        let dim = 2 * n_modes + 1;
        let n_expected = dim * dim;
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let arr = parsed.as_array()?;
        if arr.len() != n_expected { return None; }
        let mut out = Vec::with_capacity(n_expected);
        for s in arr {
            out.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some(out)
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
        eprintln!(
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
        lambda_sq: u64,
        n_modes: usize,
        prec: u32,
        mode: xc_numerics::quadrature::CacheMode,
    ) -> Option<Vec<Float>> {
        use xc_numerics::quadrature::CacheMode;
        if mode == CacheMode::Off { return None; }

        // Tier 1 (all non-Off modes): uncompressed JSON. Fast read.
        if let Some(path) = json_path(lambda_sq, n_modes, prec) {
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(data) => match parse_json(&data, n_modes, prec) {
                        Some(tau) => {
                            if let Some(reason) = structural_check(&tau, n_modes, prec) {
                                warn_skip(&path, &reason);
                            } else {
                                return Some(tau);
                            }
                        }
                        None => warn_skip(&path, "JSON shape mismatch or unparseable"),
                    },
                    Err(e) => warn_skip(&path, &format!("read failed: {}", e)),
                }
            }
        }

        // JsonOnly stops after the uncompressed tier.
        if mode == CacheMode::JsonOnly { return None; }

        // Tier 2 (JsonZip, DynamicFetch): local zip — single first, then
        // multi-part. Both write the decompressed .json on success.
        if let Some(tau) = try_load_local_zip(lambda_sq, n_modes, prec) {
            return Some(tau);
        }

        // JsonZip stops after the local tiers.
        if mode == CacheMode::JsonZip { return None; }

        // Tier 3 (DynamicFetch only): remote fetch from the public
        // consolidated τ-cache repo. Probe the single zip first, then the
        // byte-split parts. On success the file(s) land in the local
        // cache dir; we then re-run the local-zip loader so the
        // decompressed .json is written and the same validation applies.
        if fetch_remote(lambda_sq, n_modes, prec) {
            if let Some(tau) = try_load_local_zip(lambda_sq, n_modes, prec) {
                return Some(tau);
            }
        }

        None
    }

    /// Attempt to load a config from a local single zip, then local
    /// multi-part parts (tier 2). On success writes the decompressed
    /// `.json` alongside and returns the matrix. Returns `None` if no
    /// local zip/parts exist or they fail validation.
    fn try_load_local_zip(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<Vec<Float>> {
        let json_filename = cache_filename(lambda_sq, n_modes, prec);

        // Single zip first.
        if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
            if zp.exists() {
                match std::fs::read(&zp) {
                    Ok(bytes) => match read_single_zip(&bytes, &json_filename, n_modes, prec) {
                        Some((tau, data)) => {
                            if let Some(reason) = structural_check(&tau, n_modes, prec) {
                                warn_skip(&zp, &reason);
                            } else {
                                if let Some(jp) = json_path(lambda_sq, n_modes, prec) {
                                    let _ = std::fs::write(&jp, &data);
                                }
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
                Some((tau, data)) => {
                    if let Some(reason) = structural_check(&tau, n_modes, prec) {
                        warn_skip(&first_part_path, &format!("{} (split parts)", reason));
                    } else {
                        if let Some(jp) = json_path(lambda_sq, n_modes, prec) {
                            let _ = std::fs::write(&jp, &data);
                        }
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

    /// Base raw URL of the public consolidated τ-cache repository.
    const REMOTE_BASE: &str =
        "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache/main";

    /// Remote directory (and filename stem) for a config, using the
    /// repo's precision-first → λ² → nmodes-thousand-bucket layout.
    /// Returns `(dir_url, filename)` where `filename` is the canonical
    /// `lambda_sq{L}_nmodes{N}_prec{P}.json.zip` stem.
    fn remote_dir_and_stem(lambda_sq: u64, n_modes: usize, prec: u32) -> (String, String) {
        let bucket = (n_modes / 1000) * 1000;
        let dir = format!(
            "{base}/tau_cache/prec{p}/lambda_sq{l}/nmodes{b}-{bend}",
            base = REMOTE_BASE, p = prec, l = lambda_sq, b = bucket, bend = bucket + 999
        );
        let stem = format!("{}.zip", cache_filename(lambda_sq, n_modes, prec));
        (dir, stem)
    }

    /// Test-only accessor for `remote_dir_and_stem` (the function is
    /// private; this lets the test module assert URL formatting without
    /// widening the API).
    #[cfg(test)]
    pub(super) fn remote_dir_and_stem_for_test(
        lambda_sq: u64, n_modes: usize, prec: u32,
    ) -> (String, String) {
        remote_dir_and_stem(lambda_sq, n_modes, prec)
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

    /// Download a config from the public τ-cache repo into the local
    /// cache dir. Probes the **single zip first**; if that 404s, probes
    /// the byte-split parts `.part00`, `.part01`, … and stops only when
    /// a part returns an HTTP error (404 = genuine end-of-parts).
    ///
    /// A *transient* failure on any part (after retries) aborts the whole
    /// fetch and returns `false` — we must never silently truncate a
    /// multi-part config, because concatenating a partial set produces a
    /// corrupt zip. On abort, any downloaded parts are removed so a stale
    /// partial set can't be mistaken for a complete one on a later load.
    ///
    /// Mirrors the local read order (single → parts) so the subsequent
    /// `try_load_local_zip` finds and validates whatever was fetched.
    fn fetch_remote(lambda_sq: u64, n_modes: usize, prec: u32) -> bool {
        let dir = match cache_dir() { Some(d) => d, None => return false };
        let (remote_dir, stem) = remote_dir_and_stem(lambda_sq, n_modes, prec);

        // Probe 1: single zip.
        let single_url = format!("{}/{}", remote_dir, stem);
        let single_dest = dir.join(&stem);
        match curl_with_retries(&single_url, &single_dest) {
            CurlOutcome::Ok => {
                // Routine cache hit — silent.
                return true;
            }
            CurlOutcome::Transient => {
                // Network trouble even for the single-zip probe; give up
                // (caller falls through to compute).
                return false;
            }
            // HttpError (404): no single zip — this config is byte-split.
            CurlOutcome::HttpError => {}
        }

        // Probe 2: byte-split parts. Download .part00, .part01, … until
        // an HTTP error (404) marks the genuine end. A transient failure
        // aborts and cleans up — never a silent truncation.
        let mut downloaded: Vec<std::path::PathBuf> = Vec::new();
        let mut idx = 0usize;
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
                CurlOutcome::HttpError => break, // genuine end-of-parts
                CurlOutcome::Transient => {
                    // Could not retrieve a part that may well exist.
                    // Abort: remove everything downloaded so a partial
                    // set is never concatenated into a corrupt zip.
                    eprintln!(
                        "[tau_cache] WARNING: transient failure fetching {} after retries; \
                         aborting remote fetch and recomputing",
                        part_name
                    );
                    for p in &downloaded { let _ = std::fs::remove_file(p); }
                    return false;
                }
            }
        }

        if !downloaded.is_empty() {
            // Routine multi-part cache hit — silent.
            true
        } else {
            false
        }
    }

    /// Serialize `tau` to JSON (decimal strings) and return the
    /// resulting bytes.
    fn serialize_to_json(tau: &[Float]) -> Vec<u8> {
        let strs: Vec<String> = tau.iter().map(|f| f.to_string()).collect();
        let json = serde_json::Value::Array(
            strs.into_iter().map(serde_json::Value::String).collect()
        );
        serde_json::to_vec(&json).unwrap_or_default()
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
    fn cleanup_previous(lambda_sq: u64, n_modes: usize, prec: u32) {
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
        lambda_sq: u64,
        n_modes: usize,
        prec: u32,
        tau: &[Float],
        mode: xc_numerics::quadrature::CacheMode,
    ) {
        use xc_numerics::quadrature::CacheMode;
        // Off writes nothing.
        if mode == CacheMode::Off { return; }

        // Always write the uncompressed JSON first (it's the fast-read
        // path; subsequent loads bypass zip entirely).
        let json_bytes = serialize_to_json(tau);
        if json_bytes.is_empty() { return; }

        cleanup_previous(lambda_sq, n_modes, prec);

        if let Some(jp) = json_path(lambda_sq, n_modes, prec) {
            let _ = std::fs::write(&jp, &json_bytes);
        }

        // JsonOnly writes only the uncompressed .json; no zip companion.
        if mode == CacheMode::JsonOnly { return; }

        // JsonZip / DynamicFetch: also write a compressed copy for
        // distribution. Decide single-zip vs multi-part split based on
        // compressed size.
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let zip_bytes = compress_to_zip(&json_bytes, &entry_name);
        if zip_bytes.is_empty() { return; }

        if zip_bytes.len() <= PART_BYTE_LIMIT {
            // Single-zip path: under the per-part cap, write one file.
            if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
                if let Err(e) = std::fs::write(&zp, &zip_bytes) {
                    eprintln!(
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
                    eprintln!(
                        "[tau_cache] WARNING: could not write {}: {}",
                        part_path.display(), e
                    );
                    return;
                }
            }
            eprintln!(
                "[tau_cache] wrote {} parts of ≤{} MB each (compressed total {} MB) for λ²={}, N={}, prec={}",
                n_parts, PART_BYTE_LIMIT / (1024 * 1024),
                zip_bytes.len() / (1024 * 1024),
                lambda_sq, n_modes, prec
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
        Ok { path: std::path::PathBuf, lambda_sq: u64, n_modes: usize, prec: u32 },
        /// File was skipped. Either the filename didn't match the
        /// expected pattern, or it's a `.partXX` chunk handled as part
        /// of an assembled multi-part archive.
        Skipped { path: std::path::PathBuf, reason: String },
        /// Filename matched but the file failed to load (decompress,
        /// concatenate, parse JSON, etc.).
        LoadFailed { path: std::path::PathBuf, lambda_sq: u64, n_modes: usize, prec: u32, reason: String },
        /// File loaded but failed at least one structural identity
        /// check on the τ matrix it contains.
        StructurallyInvalid { path: std::path::PathBuf, lambda_sq: u64, n_modes: usize, prec: u32, reason: String },
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
    pub(super) fn parse_filename(name: &str) -> Option<(u64, usize, u32, FileKind)> {
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
        let mut p1 = after_lsq.splitn(2, "_nmodes");
        let lsq_str = p1.next()?;
        let rest = p1.next()?;
        let mut p2 = rest.splitn(2, "_prec");
        let n_str = p2.next()?;
        let prec_str = p2.next()?;
        Some((
            lsq_str.parse().ok()?,
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
        use std::collections::BTreeSet;
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
        let mut configs_with_parts: BTreeSet<(u64, usize, u32)> = BTreeSet::new();
        let mut singletons: Vec<(std::path::PathBuf, u64, usize, u32, FileKind)> = Vec::new();

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
                        configs_with_parts.insert((lsq, n_modes, prec));
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
        for (lsq, n_modes, prec) in configs_with_parts {
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

    /// Base raw URL of the public consolidated Weil-eigenvector cache
    /// repository. Files live at
    /// `{REMOTE_BASE}/weil_eigvec_cache/prec{P}/lambda_sq{L}/nmodes{B}-{B+999}/weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json.zip`
    /// where `B = (N / 1000) * 1000`.
    const REMOTE_BASE: &str =
        "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-weil-eigvec-cache/main";

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

    pub(super) fn cache_filename(lambda_sq: u64, n_modes: usize, prec: u32) -> String {
        format!("weil_eigvec_lambda_sq{}_nmodes{}_prec{}.json", lambda_sq, n_modes, prec)
    }

    fn json_path(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| d.join(cache_filename(lambda_sq, n_modes, prec)))
    }

    fn zip_path(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<std::path::PathBuf> {
        cache_dir().map(|d| {
            let f = cache_filename(lambda_sq, n_modes, prec);
            d.join(format!("{}.zip", f))
        })
    }

    /// Parse the cache JSON object into `(eps_n, xi)`. Returns `None` on
    /// any structural mismatch: wrong xi length, metadata disagreement,
    /// unparseable HP strings, or a non-finite entry.
    pub(super) fn parse_json(
        data: &str, lambda_sq: u64, n_modes: usize, prec: u32,
    ) -> Option<CachedXi> {
        let v: serde_json::Value = serde_json::from_str(data).ok()?;

        // Metadata must match the requested key (guards against a
        // filename/content mismatch or a stale collision).
        if v.get("n_modes").and_then(|x| x.as_u64())? as usize != n_modes { return None; }
        if v.get("precision_bits").and_then(|x| x.as_u64())? as u32 != prec { return None; }
        // λ² stored as a number; the paper configs are exact integers.
        let l_meta = v.get("lambda_squared").and_then(|x| x.as_f64())?;
        if (l_meta - lambda_sq as f64).abs() > 0.5 { return None; }

        let eps_str = v.get("weil_min_eigenvalue").and_then(|x| x.as_str())?;
        let eps_n = Float::with_val(prec, Float::parse(eps_str).ok()?);
        if eps_n.is_nan() || eps_n.is_infinite() { return None; }

        let arr = v.get("xi").and_then(|x| x.as_array())?;
        if arr.len() != 2 * n_modes + 1 { return None; }
        let mut xi = Vec::with_capacity(arr.len());
        for s in arr {
            let f = Float::with_val(prec, Float::parse(s.as_str()?).ok()?);
            if f.is_nan() || f.is_infinite() { return None; }
            xi.push(f);
        }
        Some(CachedXi { eps_n, xi })
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
    fn remote_zip_url(lambda_sq: u64, n_modes: usize, prec: u32) -> String {
        let bucket = (n_modes / 1000) * 1000;
        format!(
            "{base}/weil_eigvec_cache/prec{p}/lambda_sq{l}/nmodes{b}-{bend}/{stem}.zip",
            base = REMOTE_BASE, p = prec, l = lambda_sq,
            b = bucket, bend = bucket + 999,
            stem = cache_filename(lambda_sq, n_modes, prec)
        )
    }

    /// Test-only accessor for `remote_zip_url`.
    #[cfg(test)]
    pub(super) fn remote_zip_url_for_test(
        lambda_sq: u64, n_modes: usize, prec: u32,
    ) -> String {
        remote_zip_url(lambda_sq, n_modes, prec)
    }

    fn warn_skip(path: &std::path::Path, reason: &str) {
        eprintln!(
            "[weil_eigvec_cache] WARNING: skipping {} ({}); recomputing",
            path.display(), reason
        );
    }

    /// Read a single zip and return the parsed entry plus the raw inner
    /// JSON (so the caller can write the decompressed copy without
    /// re-serializing).
    fn read_single_zip(
        zip_path: &std::path::Path,
        lambda_sq: u64, n_modes: usize, prec: u32,
    ) -> Option<(CachedXi, String)> {
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let mut entry = archive.by_name(&entry_name).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_json(&data, lambda_sq, n_modes, prec)?;
        Some((parsed, data))
    }

    pub(super) fn load(
        lambda_sq: u64, n_modes: usize, prec: u32, mode: CacheMode,
    ) -> Option<CachedXi> {
        if mode == CacheMode::Off { return None; }

        // Tier 1 (all non-Off modes): uncompressed JSON. Fast read.
        if let Some(path) = json_path(lambda_sq, n_modes, prec) {
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(data) => match parse_json(&data, lambda_sq, n_modes, prec) {
                        Some(c) => return Some(c),
                        None => warn_skip(&path, "JSON shape / metadata mismatch or unparseable"),
                    },
                    Err(e) => warn_skip(&path, &format!("read failed: {}", e)),
                }
            }
        }

        // JsonOnly stops after the uncompressed tier.
        if mode == CacheMode::JsonOnly { return None; }

        // Tier 2 (JsonZip, DynamicFetch): local single zip.
        if let Some(c) = try_load_local_zip(lambda_sq, n_modes, prec) {
            return Some(c);
        }

        // JsonZip stops after the local tiers.
        if mode == CacheMode::JsonZip { return None; }

        // Tier 3 (DynamicFetch only): remote fetch (single zip; no parts,
        // ξ is small). On success the zip lands locally and we re-run the
        // local-zip loader so the decompressed .json is written.
        if fetch_remote_zip(lambda_sq, n_modes, prec) {
            if let Some(c) = try_load_local_zip(lambda_sq, n_modes, prec) {
                return Some(c);
            }
        }

        None
    }

    /// Load from a local single zip (tier 2). On success writes the
    /// decompressed `.json` alongside and returns the parsed entry.
    fn try_load_local_zip(lambda_sq: u64, n_modes: usize, prec: u32) -> Option<CachedXi> {
        let zp = zip_path(lambda_sq, n_modes, prec)?;
        if !zp.exists() { return None; }
        match read_single_zip(&zp, lambda_sq, n_modes, prec) {
            Some((parsed, json_string)) => {
                if let Some(jp) = json_path(lambda_sq, n_modes, prec) {
                    let _ = std::fs::write(&jp, &json_string);
                }
                Some(parsed)
            }
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
    fn fetch_remote_zip(lambda_sq: u64, n_modes: usize, prec: u32) -> bool {
        let dest = match zip_path(lambda_sq, n_modes, prec) {
            Some(p) => p,
            None => return false,
        };
        let url = remote_zip_url(lambda_sq, n_modes, prec);

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

    /// Serialize `(eps_n, xi)` to the schema-versioned JSON object.
    fn serialize_to_json(
        lambda_sq: u64, n_modes: usize, prec: u32, eps_n: &Float, xi: &[Float],
    ) -> Vec<u8> {
        let xi_strings: Vec<String> = xi.iter().map(|f| f.to_string()).collect();
        let payload = serde_json::json!({
            "schema_version": 1,
            "lambda_squared": lambda_sq,
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

    fn cleanup_previous(lambda_sq: u64, n_modes: usize, prec: u32) {
        if let Some(p) = json_path(lambda_sq, n_modes, prec) {
            if p.exists() { let _ = std::fs::remove_file(&p); }
        }
        if let Some(p) = zip_path(lambda_sq, n_modes, prec) {
            if p.exists() { let _ = std::fs::remove_file(&p); }
        }
    }

    pub(super) fn save(
        lambda_sq: u64, n_modes: usize, prec: u32,
        eps_n: &Float, xi: &[Float], mode: CacheMode,
    ) {
        if mode == CacheMode::Off { return; }

        let json_bytes = serialize_to_json(lambda_sq, n_modes, prec, eps_n, xi);
        if json_bytes.is_empty() { return; }

        cleanup_previous(lambda_sq, n_modes, prec);

        // Always write the uncompressed JSON first (fast-read path).
        if let Some(jp) = json_path(lambda_sq, n_modes, prec) {
            let _ = std::fs::write(&jp, &json_bytes);
        }

        // JsonOnly writes only the uncompressed .json; no zip companion.
        if mode == CacheMode::JsonOnly { return; }

        // JsonZip / DynamicFetch: also write a compressed copy for
        // distribution. ξ is small, so this is always a single zip (no
        // byte-split tier — unlike τ).
        let entry_name = cache_filename(lambda_sq, n_modes, prec);
        let zip_bytes = compress_to_zip(&json_bytes, &entry_name);
        if zip_bytes.is_empty() { return; }
        if let Some(zp) = zip_path(lambda_sq, n_modes, prec) {
            if let Err(e) = std::fs::write(&zp, &zip_bytes) {
                eprintln!(
                    "[weil_eigvec_cache] WARNING: could not write {}: {}",
                    zp.display(), e
                );
            }
        }
    }
}


#[cfg(test)]
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
        assert_eq!(cfg.newton_steps, 20);
        assert_eq!(cfg.n_eigenvalues, 50);
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
    fn run_small_n_produces_eigenvalues() {
        let params = CcmParams::from_lambda(3.605551275463989, 10);
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

        // Should produce 5 eigenvalues.
        assert_eq!(result.eigenvalues_pos.len(), 5);

        // First eigenvalue should match 14.13... to at least 10 digits.
        // Compare in HP — no f64 round-trip.
        let prec = result.precision_bits;
        let target = Float::with_val(prec,
            Float::parse("14.134725141734693790457251983").unwrap());
        let mut diff = result.eigenvalues_pos[0].clone();
        diff -= &target;
        let abs_diff = diff.abs();
        let tol = Float::with_val(prec, Float::parse("1e-5").unwrap());
        assert!(abs_diff < tol,
            "first eigenvalue {} should be near 14.13",
            xc_numerics::fmt::display_hp(&result.eigenvalues_pos[0], 10));

        // ε_N should be small (tiny Weil eigenvalue at λ²=13). Compare HP.
        let eps_tol = Float::with_val(prec, Float::parse("1e-20").unwrap());
        let abs_eps = result.weil_min_eigenvalue.clone().abs();
        assert!(abs_eps < eps_tol,
            "ε_N = {} should be tiny at λ²=13, N=10",
            xc_numerics::fmt::display_hp(&result.weil_min_eigenvalue, 6));

        // Elapsed time should be positive (f64 metadata is fine here).
        assert!(result.elapsed_seconds > 0.0);
    }

    /// measure_evenness at λ²=13, N=10 should show near-perfect evenness.
    #[test]
    fn measure_evenness_small_lambda_is_even() {
        let params = CcmParams::from_lambda(3.605551275463989, 10);
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

        // .json
        let r = parse_filename("lambda_sq13_nmodes120_prec3338.json").unwrap();
        assert_eq!(r.0, 13);
        assert_eq!(r.1, 120);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Json));

        // .json.zip
        let r = parse_filename("lambda_sq100_nmodes500_prec3338.json.zip").unwrap();
        assert_eq!(r.0, 100);
        assert_eq!(r.1, 500);
        assert_eq!(r.2, 3338);
        assert!(matches!(r.3, FileKind::Zip));

        // .json.zip.partXX
        let r = parse_filename("lambda_sq1000_nmodes800_prec3338.json.zip.part00").unwrap();
        assert_eq!(r.0, 1000);
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

        let prec = 128;
        let n_modes = 3;
        let dim = 2 * n_modes + 1; // 7
        let lambda_sq: u64 = 13;

        let temp_dir = crate::fresh_test_dir("tau_cache_classify");

        // 1. Valid: build a symmetric matrix and serialize.
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
        let valid_json = serde_json::Value::Array(
            strs.into_iter().map(serde_json::Value::String).collect()
        );
        std::fs::write(&valid_path, serde_json::to_string(&valid_json).unwrap()).unwrap();

        // 2. StructurallyInvalid: asymmetric matrix at a different (lsq).
        let mut asym = sym.clone();
        asym[0 * dim + 1] = Float::with_val(prec, 99);
        // τ[1,0] still its original value → asymmetry.
        let bad_name = cache_filename(lambda_sq + 1, n_modes, prec);
        let bad_path = temp_dir.join(&bad_name);
        let bad_strs: Vec<String> = asym.iter().map(|f| f.to_string()).collect();
        let bad_json = serde_json::Value::Array(
            bad_strs.into_iter().map(serde_json::Value::String).collect()
        );
        std::fs::write(&bad_path, serde_json::to_string(&bad_json).unwrap()).unwrap();

        // 3. Skipped: unrecognized filename.
        let skipped_path = temp_dir.join("not_a_tau_cache.txt");
        std::fs::write(&skipped_path, "irrelevant").unwrap();

        // 4. LoadFailed: matching pattern, malformed JSON.
        let malformed_name = cache_filename(lambda_sq + 2, n_modes, prec);
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
        let prec = 128;
        let lambda_sq = 13u64;
        let n_modes = 3usize;
        let dim = 2 * n_modes + 1; // 7

        let temp = crate::fresh_test_dir("tau_invalid_json");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("tau_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(cache_filename(lambda_sq, n_modes, prec));

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
        let json = serde_json::Value::Array(
            strs.into_iter().map(serde_json::Value::String).collect());
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();

        // load must skip the asymmetric matrix (None).
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonOnly).is_none(),
            "structurally-invalid (asymmetric) τ .json must be skipped");
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
            "structurally-invalid τ .json must be skipped (no zip either)");

        // Bad file preserved on disk.
        assert!(path.exists(),
            "structurally-invalid τ file should be preserved for inspection");

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
        let prec = 64;
        let lambda_sq = 49u64;
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
            lambda_sq, n_modes, prec));
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
    #[test]
    fn tau_remote_url_uses_bucketed_layout() {
        // λ²=1000, N=800, prec=3338 → bucket 0-999.
        let (dir, stem) = super::tau_cache::remote_dir_and_stem_for_test(1000, 800, 3338);
        assert_eq!(
            dir,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache/main/tau_cache/prec3338/lambda_sq1000/nmodes0-999"
        );
        assert_eq!(stem, "lambda_sq1000_nmodes800_prec3338.json.zip");

        // λ²=400, N=1500, prec=4999 → bucket 1000-1999.
        let (dir2, stem2) = super::tau_cache::remote_dir_and_stem_for_test(400, 1500, 4999);
        assert_eq!(
            dir2,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-tau-cache/main/tau_cache/prec4999/lambda_sq400/nmodes1000-1999"
        );
        assert_eq!(stem2, "lambda_sq400_nmodes1500_prec4999.json.zip");
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
    /// the repo (Paper A headline; >90 MB → multiple .partXX). In a
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

        let lambda_sq = 1000u64;
        let n_modes = 800usize;
        let prec = 3338u32;
        let params = CcmParams::from_lambda((lambda_sq as f64).sqrt(), n_modes);
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
        // λ²=1000, N=800, prec=3338 → bucket 0-999.
        let url = super::weil_eigvec_cache::remote_zip_url_for_test(1000, 800, 3338);
        assert_eq!(
            url,
            "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-weil-eigvec-cache/main/weil_eigvec_cache/prec3338/lambda_sq1000/nmodes0-999/weil_eigvec_lambda_sq1000_nmodes800_prec3338.json.zip"
        );

        // λ²=400, N=1500, prec=4999 → bucket 1000-1999.
        let url2 = super::weil_eigvec_cache::remote_zip_url_for_test(400, 1500, 4999);
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
        let prec = 128;
        let lambda_sq = 13u64;
        let n_modes = 3usize;
        let dim = 2 * n_modes + 1; // 7

        let xi: Vec<Float> = (0..dim).map(|i| Float::with_val(prec, (i + 1) as f64)).collect();
        let xi_strs: Vec<String> = xi.iter().map(|f| f.to_string()).collect();
        let good = serde_json::json!({
            "schema_version": 1,
            "lambda_squared": lambda_sq,
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
        assert!(parse_json(&good, lambda_sq + 5, n_modes, prec).is_none(),
            "lambda_sq mismatch should be rejected");

        // Wrong xi length → reject.
        let mut short_strs = xi_strs.clone();
        short_strs.pop();
        let short = serde_json::json!({
            "schema_version": 1, "lambda_squared": lambda_sq, "n_modes": n_modes,
            "precision_bits": prec, "weil_min_eigenvalue": "1.5e-40", "xi": short_strs,
        }).to_string();
        assert!(parse_json(&short, lambda_sq, n_modes, prec).is_none(),
            "wrong xi length should be rejected");
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
        let prec = 128;
        let lambda_sq = 49u64;
        let n_modes = 4usize;
        let dim = 2 * n_modes + 1; // 9

        let temp = weil_temp_cwd("round_trip");
        let _guard = CwdGuard::enter(&temp);

        let eps = Float::with_val(prec, Float::parse("3.25e-12").unwrap());
        let xi: Vec<Float> = (0..dim)
            .map(|i| Float::with_val(prec, Float::parse(
                &format!("0.{}1", i + 1)).unwrap()))
            .collect();

        // Off: writes nothing, reads nothing.
        save(lambda_sq, n_modes, prec, &eps, &xi, CacheMode::Off);
        assert!(load(lambda_sq, n_modes, prec, CacheMode::Off).is_none(),
            "Off should never read");
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonOnly).is_none(),
            "Off save should have written nothing");

        // JsonZip: writes .json + .json.zip; reads back identical.
        save(lambda_sq, n_modes, prec, &eps, &xi, CacheMode::JsonZip);
        let got = load(lambda_sq, n_modes, prec, CacheMode::JsonZip)
            .expect("JsonZip round-trip should load");
        assert_eq!(got.xi.len(), dim);
        for (a, b) in xi.iter().zip(got.xi.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "xi entry must round-trip exactly");
        }
        assert_eq!(eps.to_string(), got.eps_n.to_string(),
            "eps_n must round-trip exactly");

        // Remove the .json so the next read must use the .zip tier.
        let jp = temp.join("data").join("weil_eigvec_cache")
            .join(super::weil_eigvec_cache::cache_filename(lambda_sq, n_modes, prec));
        std::fs::remove_file(&jp).unwrap();
        let from_zip = load(lambda_sq, n_modes, prec, CacheMode::JsonZip)
            .expect("zip-tier load should succeed after .json removed");
        for (a, b) in xi.iter().zip(from_zip.xi.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "zip-tier xi must match");
        }

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }

    /// Negative: a structurally-invalid ξ `.json` (parseable JSON but
    /// wrong xi length / mismatched metadata) must be skipped by `load`
    /// (returns `None`, treated as a miss → caller recomputes). The bad
    /// file is preserved on disk for inspection. Mirrors the GL cache's
    /// `cache_discards_structurally_invalid_json_and_recomputes`.
    #[test]
    fn weil_eigvec_load_skips_structurally_invalid_json() {
        use super::weil_eigvec_cache::{load, cache_filename};
        use xc_numerics::quadrature::CacheMode;
        let prec = 128;
        let lambda_sq = 13u64;
        let n_modes = 4usize; // expects 2N+1 = 9 entries

        let temp = weil_temp_cwd("invalid_json");
        let _guard = CwdGuard::enter(&temp);

        let dir = temp.join("data").join("weil_eigvec_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(cache_filename(lambda_sq, n_modes, prec));

        // Shape-parseable JSON, but xi has the WRONG length (3 ≠ 9)
        // and otherwise-valid metadata. parse_json must reject it.
        let bad = serde_json::json!({
            "schema_version": 1,
            "lambda_squared": lambda_sq,
            "n_modes": n_modes,
            "precision_bits": prec,
            "weil_min_eigenvalue": "1.0e-20",
            "xi": ["1.0", "2.0", "3.0"],
        }).to_string();
        std::fs::write(&path, &bad).unwrap();

        // load must skip (None) — never returns a malformed entry.
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonOnly).is_none(),
            "structurally-invalid .json must be skipped");
        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
            "structurally-invalid .json must be skipped (zip tier has nothing either)");

        // The bad file is preserved on disk (load does not delete it;
        // only a recompute+save would overwrite it).
        assert!(path.exists(),
            "structurally-invalid file should be preserved for inspection");

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
        let prec = 64;
        let lambda_sq = 49u64;
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
            lambda_sq, n_modes, prec
        ));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        assert!(load(lambda_sq, n_modes, prec, CacheMode::JsonZip).is_none(),
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
        let lambda_sq = 13u64;
        let n_modes = 10usize;
        let prec = 229u32;
        assert!(super::weil_eigvec_cache::load(
            lambda_sq, n_modes, prec, CacheMode::JsonZip
        ).is_none(), "no local cache should exist before fetch");

        let fetched = super::weil_eigvec_cache::load(
            lambda_sq, n_modes, prec, CacheMode::DynamicFetch,
        ).map(|c| (c.xi.len(), 2 * n_modes + 1));

        let (got_len, expected_len) = fetched
            .expect("remote fetch should have returned a ξ entry");
        assert_eq!(got_len, expected_len,
            "fetched ξ length {} != 2N+1 = {}", got_len, expected_len);

        drop(_guard);
        let _ = std::fs::remove_dir_all(&temp);
    }
}
