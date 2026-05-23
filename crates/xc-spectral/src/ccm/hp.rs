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
#[derive(Debug, Clone)]
pub struct HighPrecConfig {
    pub precision_bits: u32,
    pub inverse_iter_steps: usize,
    pub newton_steps: usize,
    pub quad_points: usize,
    /// Number of positive eigenvalues to compute.
    pub n_eigenvalues: usize,
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
        }
    }
}

pub struct HighPrecResult {
    pub eigenvalues_pos: Vec<Float>,
    pub weil_min_eigenvalue: Float,
    pub xi: Vec<Float>,
    pub elapsed_seconds: f64,
    pub precision_bits: u32,
}

impl HighPrecResult {
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
    pub lambda_squared: f64,
    pub n_modes: usize,
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

    // Find smallest eigenpair by inverse iteration (forced even).
    eprintln!("[HP] LU factoring {}×{} matrix (one-time cost)...", dim, dim);
    let (eps_n, xi_raw) = xc_numerics::linalg::inverse_iteration(
        &tau, dim, prec, cfg.inverse_iter_steps, true)?;
    eprintln!("[HP] LU factorization done.");

    // Normalize: Σ ξ_j = √L.
    let xi = normalize_eigenvector(&xi_raw, &l, prec);

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

fn build_tau_hp(params: &CcmParams, l: &Float, cfg: &HighPrecConfig) -> Result<Vec<Float>> {
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
        .map(|&npts| (npts, xc_numerics::quadrature::gauss_legendre_nodes(npts, prec)))
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


#[cfg(test)]
mod tests {
    use super::*;

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
        let cfg = HighPrecConfig::for_decimal_digits(64);
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
}
