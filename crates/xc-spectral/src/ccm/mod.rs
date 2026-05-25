// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Connes-Consani-Moscovici "Zeta Spectral Triples" construction.
//!
//! Implements the operator family `D_log(λ, N)` from arxiv 2511.22755
//! (27 Nov 2025), whose eigenvalues converge to the imaginary parts
//! of the non-trivial Riemann zeta zeros as `λ, N → ∞`.
//!
//! ## Pipeline
//!
//! 1. Sieve prime powers `k ≤ λ²`.
//! 2. Build the Weil quadratic form matrix `τ_{n,m}` for `n,m ∈ {-N,…,N}`.
//! 3. Eigendecompose `τ`. Take the smallest eigenvalue `ε_N` and the
//!    (forced even) eigenvector `ξ`.
//! 4. The eigenvalues of `D_log(λ, N)` are the zeros of the rational
//!    function `R(z) = Σ ξ_j / (z − 2πj/L)`.

use anyhow::{anyhow, Result};
use std::time::Instant;

#[cfg(feature = "hp")]
pub mod hp;

/// Parameters for a single CCM run. The `lambda_squared` field is f64
/// (the user's CLI input — an explicit f64 boundary). `lambda_sq_int`
/// is precomputed at construction so HP code paths can use it as `u64`
/// without doing any f64 arithmetic on `lambda_squared`.
#[derive(Debug, Clone)]
pub struct CcmParams {
    /// `λ²` as f64 — this is the user's CLI input parameter (explicit
    /// f64 boundary). HP code paths should NOT do f64 arithmetic on
    /// this field; use `lambda_sq_int` for the prime cutoff or promote
    /// to HP via `Float::with_val(prec, lambda_squared)` for other uses.
    pub lambda_squared: f64,

    /// `⌊λ² + ε⌋` precomputed as `u64` at construction. The prime
    /// cutoff for the Weil-form sum. HP code paths read this directly
    /// without any f64 arithmetic.
    pub lambda_sq_int: u64,

    /// Mode cutoff N. Matrix dimension is `2N+1`.
    pub n_modes: usize,
}

impl CcmParams {
    /// Construct from λ (not λ²) and the mode cutoff N.
    ///
    /// `lambda_squared` is computed as `lambda * lambda` at f64;
    /// `lambda_sq_int` is `⌊λ² + ε⌋` (integer-floor with a small
    /// rounding epsilon so √13² rounds to 13, not 12).
    pub fn from_lambda(lambda: f64, n_modes: usize) -> Self {
        let lambda_squared = lambda * lambda;
        // Precompute prime-cutoff integer once at construction (CLI
        // input boundary). The +EPS handles f64 imprecision when λ is
        // an exact integer-square-root: e.g. sqrt(13)² evaluates to
        // 12.999999985... at f64, which would floor to 12. After this
        // point, HP code uses `lambda_sq_int` as a u64 — no f64
        // arithmetic in HP paths.
        let lambda_sq_int = (lambda_squared + LAMBDA_SQ_ROUNDING_EPS).floor() as u64;
        Self { lambda_squared, lambda_sq_int, n_modes }
    }

    /// Recover λ from `lambda_squared` (f64).
    pub fn lambda(&self) -> f64 { self.lambda_squared.sqrt() }
    /// `L = ln(λ²) = 2 ln λ` at f64. Used as the V_n basis half-period.
    pub fn log_length(&self) -> f64 { self.lambda_squared.ln() }
    /// Matrix dimension `2N+1`.
    pub fn matrix_size(&self) -> usize { 2 * self.n_modes + 1 }

    /// Map a centered basis index `n ∈ [-N, N]` to the row/column
    /// position in the row-major matrix (`n = 0` → position `N`).
    pub fn idx(&self, n: i64) -> usize {
        (n + self.n_modes as i64) as usize
    }
}

/// Result of a single CCM run at f64 precision.
#[derive(Debug, Clone)]
pub struct CcmResult {
    /// Newton-refined positive eigenvalues at f64 precision, in the
    /// order returned by Newton (paired with the Riemann-zero seeds).
    pub eigenvalues_pos: Vec<f64>,
    /// Smallest Weil-form eigenvalue (the spectral gap quantity ε_N).
    pub weil_min_eigenvalue: f64,
    /// Smallest-eigenvalue eigenvector of the Weil form, ℓ²-normalized,
    /// in the V_n basis order.
    pub xi: Vec<f64>,
    /// Wall-clock seconds for the entire f64 run.
    pub elapsed_seconds: f64,
}

/// Enumerate prime powers `n = p^k` with `1 < n ≤ bound`.
///
/// Returns triples `(n, p, j)` where `n = p^j`. Callers are responsible
/// for computing `log p` themselves at the appropriate precision (HP via
/// `Float::with_val(prec, p).ln()`, f64 via `(p as f64).ln()`). This
/// keeps `prime_powers_up_to` precision-agnostic so HP code paths never
/// receive an f64-truncated logarithm.
pub fn prime_powers_up_to(bound: u64) -> Vec<(u64, u64, u32)> {
    if bound < 2 { return Vec::new(); }
    let n = bound as usize;
    let mut sieve = vec![true; n + 1];
    sieve[0] = false;
    if n >= 1 { sieve[1] = false; }
    let mut p = 2usize;
    while p * p <= n {
        if sieve[p] {
            let mut q = p * p;
            while q <= n { sieve[q] = false; q += p; }
        }
        p += 1;
    }
    let mut out = Vec::new();
    for p in 2..=n {
        if !sieve[p] { continue; }
        let mut q: u64 = p as u64;
        let mut j: u32 = 1;
        while q <= bound {
            out.push((q, p as u64, j));
            if q > bound / (p as u64) { break; }
            q *= p as u64;
            j += 1;
        }
    }
    out.sort_unstable_by_key(|&(n, _, _)| n);
    out
}

/// Euler-Mascheroni constant γ ≈ 0.5772. Used in the archimedean
/// (gamma-factor) contribution W_R of the Weil quadratic form.
pub const EULER_GAMMA: f64 = 0.5772156649015328606;

/// Numerical zero threshold for eigenvector sum normalization.
/// If |Σ ξ_j| < this value, the eigenvector is considered degenerate
/// and normalization is skipped with an error.
pub const EIGENVECTOR_SUM_THRESHOLD: f64 = 1e-14;

/// Epsilon for rounding λ² to the nearest integer for prime-power sieve.
/// Prevents floating-point truncation errors (e.g., 12.9999999 → 12).
pub const LAMBDA_SQ_ROUNDING_EPS: f64 = 1e-9;

/// Singularity guard for the W_R integrand near x = 0.
/// When |x| < this value, the integrand uses its Taylor expansion
/// instead of the direct formula (which has a 0/0 form at x = 0).
pub const INTEGRAND_SINGULARITY_GUARD: f64 = 1e-9;

/// Default bisection tolerance for f64 spectrum root-finding.
pub const DEFAULT_BISECT_TOL: f64 = 1e-12;

/// Default maximum bisection iterations for f64 spectrum root-finding.
pub const DEFAULT_BISECT_MAX_ITER: usize = 200;

/// f64 CPU tier — fast but limited to ~13 digits.
pub fn run_f64(params: &CcmParams) -> Result<CcmResult> {
    use nalgebra::SymmetricEigen;

    let start = Instant::now();
    let n = params.n_modes;
    let l = params.log_length();

    let tau = build_tau_f64(params, l, params.lambda_sq_int)?;

    let eig = SymmetricEigen::new(tau);
    let (eps_n, idx_min) = eig.eigenvalues.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (v, i))
        .ok_or_else(|| anyhow!("empty spectrum"))?;
    let xi_raw: Vec<f64> = eig.eigenvectors.column(idx_min).iter().copied().collect();

    // Symmetrize as even.
    let mut xi = vec![0.0_f64; 2 * n + 1];
    for j in 0..=n {
        let pos = params.idx(j as i64);
        let neg = params.idx(-(j as i64));
        let avg = 0.5 * (xi_raw[pos] + xi_raw[neg]);
        xi[pos] = avg;
        xi[neg] = avg;
    }

    // Normalize: Σ ξ_j = √L.
    let sum_xi: f64 = xi.iter().sum();
    if sum_xi.abs() < EIGENVECTOR_SUM_THRESHOLD {
        return Err(anyhow!("Sum of eigenvector components is ~0"));
    }
    let scale = l.sqrt() / sum_xi;
    for v in xi.iter_mut() { *v *= scale; }

    let eigenvalues_pos = solve_spectrum_f64(&xi, n, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER)?;

    Ok(CcmResult {
        eigenvalues_pos,
        weil_min_eigenvalue: eps_n,
        xi,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

fn build_tau_f64(params: &CcmParams, l: f64, lambda_sq_int: u64) -> Result<nalgebra::DMatrix<f64>> {
    use nalgebra::DMatrix;

    let n_max = params.n_modes;
    let dim = params.matrix_size();
    let mut tau = DMatrix::<f64>::zeros(dim, dim);

    let kappa = (4.0 * std::f64::consts::PI * (l.exp() - 1.0) / (l.exp() + 1.0)).ln()
        + EULER_GAMMA;
    let sinh2_l_over_4 = (l / 4.0).sinh().powi(2);
    let sixteen_pi2 = 16.0 * std::f64::consts::PI * std::f64::consts::PI;
    let l2 = l * l;
    let prime_powers = prime_powers_up_to(lambda_sq_int);

    for n in -(n_max as i64)..=(n_max as i64) {
        for m in -(n_max as i64)..=(n_max as i64) {
            let nf = n as f64;
            let mf = m as f64;

            // W_{0,2}
            let num = l2 - sixteen_pi2 * mf * nf;
            let den = (l2 + sixteen_pi2 * mf * mf) * (l2 + sixteen_pi2 * nf * nf);
            let w02 = 32.0 * l * sinh2_l_over_4 * num / den;

            // W_R
            let omega_0 = if n == m { 2.0 } else { 0.0 };
            let two_pi_n_over_l = 2.0 * std::f64::consts::PI * nf / l;
            let two_pi_m_over_l = 2.0 * std::f64::consts::PI * mf / l;
            let omega_f64 = |x: f64| -> f64 {
                if n == m {
                    2.0 * (1.0 - x / l) * (two_pi_n_over_l * x).cos()
                } else {
                    ((two_pi_m_over_l * x).sin() - (two_pi_n_over_l * x).sin())
                        / (std::f64::consts::PI * ((n - m) as f64))
                }
            };
            let integrand = |x: f64| -> f64 {
                let num = (x / 2.0).exp() * omega_f64(x) - omega_0;
                let den = x.exp() - (-x).exp();
                if x.abs() < INTEGRAND_SINGULARITY_GUARD {
                    let omega_prime_0 = if n == m { -omega_0 / l }
                        else { 2.0 * mf / ((n - m) as f64) };
                    omega_0 / 4.0 + omega_prime_0 / 2.0
                } else {
                    num / den
                }
            };
            let integral = xc_numerics::quadrature::gauss_legendre_64pt_f64(integrand, 0.0, l);
            let wr = (omega_0 / 2.0) * kappa + integral;

            // W_p
            let mut wp_sum = 0.0_f64;
            for &(k, p, _j) in &prime_powers {
                let log_p = (p as f64).ln();
                let y = (k as f64).ln();
                let q = if n == m {
                    2.0 * (1.0 - y / l) * (two_pi_n_over_l * y).cos()
                } else {
                    ((two_pi_m_over_l * y).sin() - (two_pi_n_over_l * y).sin())
                        / (std::f64::consts::PI * ((n - m) as f64))
                };
                wp_sum += log_p * (k as f64).powf(-0.5) * q;
            }

            tau[(params.idx(n), params.idx(m))] = w02 - wr - wp_sum;
        }
    }
    Ok(tau)
}

/// Find positive eigenvalues of `D_log(λ, N)` as zeros of the rational
/// function `R(t) = ξ_0 + 2t Σ_{j=1}^{N} ξ_j / (t − j²)`, located in
/// the intervals `(k², (k+1)²)`.
///
/// `tol` and `max_iter` control the bisection precision.
pub fn solve_spectrum_f64(xi: &[f64], n_max: usize, l: f64, tol: f64, max_iter: usize) -> Result<Vec<f64>> {
    let xi_pos: Vec<f64> = (0..=n_max).map(|j| xi[j + n_max]).collect();

    let f = |t: f64| -> f64 {
        let mut acc = xi_pos[0];
        let mut sum = 0.0_f64;
        for j in 1..=n_max {
            sum += xi_pos[j] / (t - (j as f64).powi(2));
        }
        acc += 2.0 * t * sum;
        acc
    };

    let mut roots = Vec::with_capacity(n_max);
    for k in 1..=n_max {
        let lo = (k as f64).powi(2);
        let hi = ((k + 1) as f64).powi(2);
        let eps = INTEGRAND_SINGULARITY_GUARD * (hi - lo);
        let a = lo + eps;
        let b = if k == n_max { lo + 1e6 * (lo + 1.0) } else { hi - eps };
        if let Some(t) = xc_numerics::root_finding::bisect_f64(&f, a, b, tol, max_iter) {
            roots.push((2.0 * std::f64::consts::PI / l) * t.sqrt());
        }
    }
    Ok(roots)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Prime powers up to 13 should include {2, 3, 4, 5, 7, 8, 9, 11, 13}.
    /// Each entry is (k, p, j) with k = p^j.
    #[test]
    fn prime_powers_up_to_13() {
        let pp = prime_powers_up_to(13);
        let ns: Vec<u64> = pp.iter().map(|&(n, _, _)| n).collect();
        assert!(ns.contains(&2));
        assert!(ns.contains(&3));
        assert!(ns.contains(&4)); // 2²
        assert!(ns.contains(&5));
        assert!(ns.contains(&7));
        assert!(ns.contains(&8)); // 2³
        assert!(ns.contains(&9)); // 3²
        assert!(ns.contains(&11));
        assert!(ns.contains(&13));
        assert!(!ns.contains(&6)); // not a prime power
        assert!(!ns.contains(&10));

        // Check (p, j) is correct for k = p^j.
        for &(k, p, j) in &pp {
            let mut prod: u64 = 1;
            for _ in 0..j { prod *= p; }
            assert_eq!(k, prod, "k = p^j must hold; got k={}, p={}, j={}", k, p, j);
        }
    }

    /// CcmParams basic properties.
    #[test]
    fn ccm_params_basic() {
        let p = CcmParams::from_lambda(3.605551275463989, 120);
        assert!((p.lambda_squared - 13.0).abs() < 1e-6);
        assert_eq!(p.matrix_size(), 241);
        assert_eq!(p.idx(0), 120);
        assert_eq!(p.idx(-120), 0);
        assert_eq!(p.idx(120), 240);
    }

    /// f64 tier should produce a non-empty spectrum with reasonable
    /// structure at λ²=13, N=120. Note: f64 tier uses bisection on R(t)
    /// without Newton-from-seed, so eigenvalues are Weil-form roots
    /// rather than Riemann zeros (those need the HP path).
    #[test]
    fn run_f64_produces_spectrum() {
        let params = CcmParams::from_lambda(3.605551275463989, 120);
        let result = run_f64(&params).unwrap();
        assert!(!result.eigenvalues_pos.is_empty(),
            "should produce at least one eigenvalue");
        // ε_N should be tiny at λ²=13, N=120 even at f64.
        assert!(result.weil_min_eigenvalue.abs() < 1e-10,
            "ε_N = {} should be tiny", result.weil_min_eigenvalue);
        // ξ should be normalized (Σ ξ_j = √L).
        let sum_xi: f64 = result.xi.iter().sum();
        let l = (3.605551275463989_f64 * 3.605551275463989).ln();
        let expected_sum = l.sqrt();
        assert!((sum_xi - expected_sum).abs() < 1e-3,
            "Σ ξ_j = {} should be √L = {}", sum_xi, expected_sum);
        // Eigenvalues should be sorted ascending.
        for w in result.eigenvalues_pos.windows(2) {
            assert!(w[0] < w[1], "eigenvalues should be ascending");
        }
        // Some eigenvalue should match the 2nd Riemann zero (21.02)
        // since it's outside the dense Weil-form region.
        let target = 21.022040;
        let closest_to_2nd = result.eigenvalues_pos.iter()
            .map(|&e| (e - target).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(closest_to_2nd < 0.01,
            "no eigenvalue near 21.02 (closest: {:.4e})", closest_to_2nd);
    }

    /// solve_spectrum_f64 with a synthetic ξ vector: constant ξ_0 = 1, all
    /// others zero. R(t) = 1 + 0 = 1 everywhere, so no roots exist.
    /// This tests the "no root found" path.
    #[test]
    fn solve_spectrum_constant_xi_no_roots() {
        let n_max = 5;
        let l = 13.0_f64.ln(); // L = ln(13)
        let mut xi = vec![0.0_f64; 2 * n_max + 1];
        xi[n_max] = 1.0; // only ξ_0 nonzero
        let roots = solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        // With only ξ_0 nonzero, R(t) = ξ_0 = 1 (constant), no zeros.
        assert!(roots.is_empty(), "constant R(t) should have no roots, got {:?}", roots);
    }

    /// solve_spectrum_f64 with a simple ξ that has known structure.
    /// If ξ_0 = 0 and ξ_1 = 1 (all others zero), then
    /// R(t) = 0 + 2t · [1/(t - 1)] = 2t/(t-1).
    /// This has a zero at t = 0 (outside our search range) and no zero
    /// in (1, 4), (4, 9), etc. — the function is always positive for t > 1.
    /// So we expect no roots in the standard intervals.
    #[test]
    fn solve_spectrum_single_mode() {
        let n_max = 3;
        let l = 13.0_f64.ln();
        let mut xi = vec![0.0_f64; 2 * n_max + 1];
        xi[n_max + 1] = 1.0; // ξ_1 = 1
        let roots = solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        // R(t) = 2t/(t-1) for t > 1 is always positive, so no sign changes.
        assert!(roots.is_empty(), "single-mode R(t) should have no roots in standard intervals");
    }

    /// CcmParams accessor methods.
    #[test]
    fn ccm_params_accessors() {
        let p = CcmParams::from_lambda(10.0, 50);
        assert!((p.lambda() - 10.0).abs() < 1e-10);
        assert!((p.log_length() - 100.0_f64.ln()).abs() < 1e-10);
    }

    /// prime_powers_up_to edge cases.
    #[test]
    fn prime_powers_edge_cases() {
        assert!(prime_powers_up_to(0).is_empty());
        assert!(prime_powers_up_to(1).is_empty());
        let pp2 = prime_powers_up_to(2);
        assert_eq!(pp2.len(), 1);
        assert_eq!(pp2[0].0, 2);
    }
}
