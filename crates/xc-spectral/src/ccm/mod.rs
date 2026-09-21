// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! CCM "Zeta Spectral Triples" construction.
//!
//! Implements the CCM operator family `D_log(λ, N)`.
//! (27 Nov 2025). Numerical experiments compare its finite spectra with
//! Riemann zeta zeros. Convergence as `λ, N → ∞` remains an open obligation;
//! the construction and software do not prove that limit.
//!
//! ## Pipeline
//!
//! 1. Sieve prime powers `k ≤ λ²`.
//! 2. Build the Weil quadratic form matrix `τ_{n,m}` for `n,m ∈ {-N,…,N}`.
//! 3. Solve the reduced even sector and lift its lowest eigenvector `ξ`.
//!    This selects an even Ritz state; it does not prove full-space ground selection.
//! 4. The eigenvalues of `D_log(λ, N)` are the zeros of the rational
//!    function `R(z) = Σ ξ_j / (z − 2πj/L)`.

use anyhow::{anyhow, Result};
use std::time::Instant;

#[cfg(feature = "hp")]
pub mod hp;

#[cfg(feature = "hp")]
pub mod evidence;

#[cfg(feature = "hp")]
pub mod certified_roots;

#[cfg(feature = "hp")]
pub mod sector_gap_certificate;

#[cfg(feature = "arb")]
mod arb_bridge;

#[cfg(feature = "arb")]
pub mod cutoff_free;

pub mod artifacts;
pub mod convergence;

pub mod capture;
#[cfg(feature = "hp")]
pub mod prefix;
pub mod rank_one;
pub mod reproduction;
#[cfg(feature = "hp")]
pub mod research;
pub mod window;

/// How λ² is represented and processed.
///
/// Both `value_u64` and `value_f64` are always populated. The
/// `is_integer` flag controls which path the HP computation takes:
///
/// - `is_integer = true`: uses `value_u64` for exact HP promotion
///   (`Float::with_val(prec, value_u64)`) — full working precision,
///   no representation error. This is the correct path for the publication
///   configs (13, 100, 1000, etc.).
///
/// - `is_integer = false`: uses `value_f64` formatted to 17
///   significant figures and parsed into HP via string conversion.
///   Gives ~17 digits of accuracy on L = ln(λ²). Used for
///   convergence-formula research (dense sweeps in the low-λ² region).
///
/// Having both values always available allows optimizations: the u64
/// is always used for prime sieving (`prime_powers_up_to(value_u64)`),
/// the f64 is always used for display and f64-tier computation, and
/// the bool selects the HP promotion path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaSq {
    /// Integer form: `⌊λ²⌋`. Used for prime cutoff, cache keys in
    /// integer mode, and exact HP promotion when `is_integer = true`.
    pub value_u64: u64,
    /// Float form: the full-precision f64 λ² value. Used for display,
    /// f64-tier computation, cache keys in fractional mode, and HP
    /// promotion when `is_integer = false`.
    pub value_f64: f64,
    /// `true` = integer mode (HP uses `value_u64`).
    /// `false` = fractional mode (HP uses `value_f64`).
    pub is_integer: bool,
}

impl LambdaSq {
    /// Construct in integer mode from a u64 value.
    pub fn integer(v: u64) -> Self {
        Self {
            value_u64: v,
            value_f64: v as f64,
            is_integer: true,
        }
    }

    /// Construct in fractional mode from an f64 value.
    pub fn fractional(v: f64) -> Self {
        Self {
            value_u64: v.floor() as u64,
            value_f64: v,
            is_integer: false,
        }
    }

    /// Construct in fractional mode but with an explicit u64 override
    /// (for testing: e.g. λ²=15.0 fractional with value_u64=15).
    pub fn fractional_with_int(value_f64: f64, value_u64: u64) -> Self {
        Self {
            value_u64,
            value_f64,
            is_integer: false,
        }
    }

    /// Mode string for JSON metadata: `"integer"` or `"fractional"`.
    pub fn mode_str(&self) -> &'static str {
        if self.is_integer {
            "integer"
        } else {
            "fractional"
        }
    }

    /// Canonical string for cache filenames.
    /// Integer: `"13"`, Fractional: `"12p5"` (dot replaced with `p`,
    /// trailing zeros stripped).
    pub fn filename_str(&self) -> String {
        if self.is_integer {
            format!("{}", self.value_u64)
        } else {
            let s = format!("{}", self.value_f64);
            if s.contains('.') {
                let trimmed = s.trim_end_matches('0');
                let trimmed = if trimmed.ends_with('.') {
                    format!("{}0", trimmed)
                } else {
                    trimmed.to_string()
                };
                trimmed.replace('.', "p")
            } else {
                format!("{}p0", s)
            }
        }
    }

    /// Parse a filename fragment back into a `LambdaSq`.
    /// `"13"` → integer(13), `"12p5"` → fractional(12.5).
    pub fn from_filename_str(s: &str) -> Option<Self> {
        if s.contains('p') {
            let f_str = s.replace('p', ".");
            let v: f64 = f_str.parse().ok()?;
            Some(LambdaSq::fractional(v))
        } else {
            let v: u64 = s.parse().ok()?;
            Some(LambdaSq::integer(v))
        }
    }
}

/// Parameters for a single CCM run.
#[derive(Debug, Clone)]
pub struct CcmParams {
    /// `λ²` — either exact integer or fractional f64.
    pub lambda_sq: LambdaSq,

    /// Mode cutoff N. Matrix dimension is `2N+1`.
    pub n_modes: usize,
}

impl CcmParams {
    /// Construct with an explicit integer λ² value. This is the
    /// primary constructor — pass λ² directly (e.g. 13, 100, 1000).
    pub fn from_lambda_sq_integer(lambda_sq: u64, n_modes: usize) -> Self {
        Self {
            lambda_sq: LambdaSq::integer(lambda_sq),
            n_modes,
        }
    }

    /// Construct with an explicit fractional λ² value (e.g. 12.5, 2.7).
    /// Uses the float path for HP promotion (~17 digits of accuracy on L).
    pub fn from_lambda_sq_fractional(lambda_sq: f64, n_modes: usize) -> Self {
        Self {
            lambda_sq: LambdaSq::fractional(lambda_sq),
            n_modes,
        }
    }

    /// The f64 value of λ² (for display, f64-tier computation).
    pub fn lambda_squared(&self) -> f64 {
        self.lambda_sq.value_f64
    }
    /// The integer floor of λ² — the prime cutoff for the Weil-form sum.
    pub fn lambda_sq_int(&self) -> u64 {
        self.lambda_sq.value_u64
    }
    /// `L = ln(λ²) = 2 ln λ` at f64.
    pub fn log_length(&self) -> f64 {
        self.lambda_sq.value_f64.ln()
    }
    /// Matrix dimension `2N+1`.
    pub fn matrix_size(&self) -> usize {
        2 * self.n_modes + 1
    }

    /// Map a centered basis index `n ∈ [-N, N]` to the row/column
    /// position in the row-major matrix (`n = 0` → position `N`).
    pub fn idx(&self, n: i64) -> usize {
        (n + self.n_modes as i64) as usize
    }
}

/// Result of a single CCM run at f64 precision.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CcmResult {
    /// Positive sign-change roots discovered from the even-sector state at f64
    /// precision, in ascending order. The search has no completeness guarantee.
    pub eigenvalues_pos: Vec<f64>,
    /// Lowest computed even-sector Ritz value. Binary64 does not certify its
    /// sign or its ordering against the odd sector or the continuum spectrum.
    pub weil_min_eigenvalue: f64,
    /// Selected even-sector state in centered V_n order, scaled so its
    /// component sum is sqrt(L). It is not normalized to unit ℓ² norm.
    pub xi: Vec<f64>,
    /// Wall-clock seconds for the entire f64 run.
    pub elapsed_seconds: f64,
}

impl CcmResult {
    /// Positive finite `D_log` spectral roots.
    ///
    /// These are distinct from eigenvalues of the Tau/Weil quadratic form.
    pub fn spectral_roots(&self) -> &[f64] {
        &self.eigenvalues_pos
    }
}

/// Enumerate prime powers `n = p^k` with `1 < n ≤ bound`.
///
/// Returns triples `(n, p, j)` where `n = p^j`. Callers are responsible
/// for computing `log p` themselves at the appropriate precision (HP via
/// `Float::with_val(prec, p).ln()`, f64 via `(p as f64).ln()`). This
/// keeps `prime_powers_up_to` precision-agnostic so HP code paths never
/// receive an f64-truncated logarithm.
pub fn prime_powers_up_to(bound: u64) -> Vec<(u64, u64, u32)> {
    if bound < 2 {
        return Vec::new();
    }
    let n = bound as usize;
    let mut sieve = vec![true; n + 1];
    sieve[0] = false;
    if n >= 1 {
        sieve[1] = false;
    }
    let mut p = 2usize;
    while p * p <= n {
        if sieve[p] {
            let mut q = p * p;
            while q <= n {
                sieve[q] = false;
                q += p;
            }
        }
        p += 1;
    }
    let mut out = Vec::new();
    for (p, &is_prime) in sieve.iter().enumerate().skip(2) {
        if !is_prime {
            continue;
        }
        let mut q: u64 = p as u64;
        let mut j: u32 = 1;
        while q <= bound {
            out.push((q, p as u64, j));
            if q > bound / (p as u64) {
                break;
            }
            q *= p as u64;
            j += 1;
        }
    }
    out.sort_unstable_by_key(|&(n, _, _)| n);
    out
}

/// Euler-Mascheroni constant γ ≈ 0.5772. Used in the archimedean
/// (gamma-factor) contribution W_R of the Weil quadratic form. Quoted
/// at full precision for documentation; f64 rounds to its nearest value.
#[allow(clippy::excessive_precision)]
pub const EULER_GAMMA: f64 = 0.5772156649015328606;

/// Numerical zero threshold for eigenvector sum normalization.
/// If |Σ ξ_j| < this value, the eigenvector is considered degenerate
/// and normalization is skipped with an error.
pub const EIGENVECTOR_SUM_THRESHOLD: f64 = 1e-14;

/// Singularity guard for the W_R integrand near x = 0.
/// When |x| < this value, the integrand uses its Taylor expansion
/// instead of the direct formula (which has a 0/0 form at x = 0).
pub const INTEGRAND_SINGULARITY_GUARD: f64 = 1e-9;

/// Default bisection tolerance for f64 spectrum root-finding.
pub const DEFAULT_BISECT_TOL: f64 = 1e-12;

/// Default maximum bisection iterations for f64 spectrum root-finding.
pub const DEFAULT_BISECT_MAX_ITER: usize = 200;

/// Binary64 exploratory even-sector computation. Matrix integration scales
/// with the highest Fourier mode. Near-degenerate states and tiny boundary
/// sums can still be unresolved at binary64 precision; no certified root
/// ordinal, spectral sign, or continuum ground selection is supplied.
pub fn run_f64(params: &CcmParams) -> Result<CcmResult> {
    use nalgebra::SymmetricEigen;

    let start = Instant::now();
    let n = params.n_modes;
    let l = params.log_length();

    let tau = build_tau_f64(params, l, params.lambda_sq_int())?;

    // Project the operator onto an orthonormal even basis before solving.
    // Projecting the unrestricted minimum eigenvector afterward can turn an
    // odd vector's rounding noise into a spurious normalized even state.
    let mut even = nalgebra::DMatrix::<f64>::zeros(n + 1, n + 1);
    even[(0, 0)] = tau[(n, n)];
    for i in 1..=n {
        even[(0, i)] = (tau[(n, n + i)] + tau[(n, n - i)]) / 2.0_f64.sqrt();
        even[(i, 0)] = even[(0, i)];
        for j in 1..=n {
            even[(i, j)] = 0.5
                * (tau[(n + i, n + j)]
                    + tau[(n + i, n - j)]
                    + tau[(n - i, n + j)]
                    + tau[(n - i, n - j)]);
        }
    }
    let eig = SymmetricEigen::new(even);
    let (eps_n, idx_min) = eig
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, &v)| (v, i))
        .ok_or_else(|| anyhow!("empty spectrum"))?;
    let sector_state = eig.eigenvectors.column(idx_min);
    let mut xi = vec![0.0_f64; 2 * n + 1];
    xi[n] = sector_state[0];
    for j in 1..=n {
        let value = sector_state[j] / 2.0_f64.sqrt();
        xi[n + j] = value;
        xi[n - j] = value;
    }

    // Normalize: Σ ξ_j = √L.
    let sum_xi: f64 = xi.iter().sum();
    if sum_xi.abs() < EIGENVECTOR_SUM_THRESHOLD {
        return Err(anyhow!("Sum of eigenvector components is ~0"));
    }
    let scale = l.sqrt() / sum_xi;
    for v in xi.iter_mut() {
        *v *= scale;
    }

    let eigenvalues_pos =
        solve_spectrum_f64(&xi, n, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER)?;

    Ok(CcmResult {
        eigenvalues_pos,
        weil_min_eigenvalue: eps_n,
        xi,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

fn archimedean_origin_limit_f64(n: i64, m: i64, l: f64) -> f64 {
    let omega_0 = if n == m { 2.0 } else { 0.0 };
    // Both the diagonal cosine formula and the off-diagonal sine difference
    // have omega'(0) = -2/L. Since 2*sinh(x) = 2*x + O(x^3), the limit is
    // omega(0)/4 + omega'(0)/2, independent of the off-diagonal indices.
    let omega_prime_0 = -2.0 / l;
    omega_0 / 4.0 + omega_prime_0 / 2.0
}

fn build_tau_f64(params: &CcmParams, l: f64, lambda_sq_int: u64) -> Result<nalgebra::DMatrix<f64>> {
    use nalgebra::DMatrix;

    let n_max = params.n_modes;
    if !l.is_finite()
        || l <= 0.0
        || !params.lambda_squared().is_finite()
        || params.lambda_squared() <= 1.0
        || (!params.lambda_sq.is_integer && params.lambda_squared().floor() as u64 != lambda_sq_int)
    {
        return Err(anyhow!(
            "CCM requires finite lambda-squared > 1 and a matching prime cutoff"
        ));
    }
    let dim = n_max
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| anyhow!("CCM matrix dimension overflow"))?;
    let quadrature_order = n_max
        .checked_mul(3)
        .and_then(|v| v.checked_add(32))
        .ok_or_else(|| anyhow!("CCM quadrature order overflow"))?
        .max(64);
    // Resolve one rule for the entire matrix. Fixed GL64 aliases higher modes.
    let (nodes, weights) = xc_numerics::quadrature::gl_nodes_weights_f64(quadrature_order);
    let quadrature: Vec<(f64, f64)> = nodes
        .iter()
        .zip(&weights)
        .map(|(&node, &weight)| (0.5 * l * (1.0 + node), 0.5 * l * weight))
        .collect();
    let mut tau = DMatrix::<f64>::zeros(dim, dim);

    let kappa = (4.0 * std::f64::consts::PI * (0.5 * l).tanh()).ln() + EULER_GAMMA;
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
            let integrand = |x: f64| -> f64 {
                if x == 0.0 {
                    return archimedean_origin_limit_f64(n, m, l);
                }
                if n == m {
                    let phase = two_pi_n_over_l * x;
                    let cosine = phase.cos();
                    // Rearrange exp(x/2)*(1-x/L)*cos(phase)-1 without
                    // subtracting nearly equal numbers at the origin.
                    (-2.0 * (0.5 * phase).sin().powi(2) - (x / l) * cosine
                        + (0.5 * x).exp_m1() * (1.0 - x / l) * cosine)
                        / x.sinh()
                } else {
                    let difference = std::f64::consts::PI * (nf - mf) * x / l;
                    let mean = std::f64::consts::PI * (nf + mf) * x / l;
                    let sinc = if difference == 0.0 {
                        1.0
                    } else {
                        difference.sin() / difference
                    };
                    let omega = -2.0 * (x / l) * sinc * mean.cos();
                    (0.5 * x).exp() * omega / (2.0 * x.sinh())
                }
            };
            let integral: f64 = quadrature
                .iter()
                .map(|&(x, weight)| weight * integrand(x))
                .sum();
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

/// Discover positive sign-change roots of the even-state secular function
/// `R(t) = ξ_0 + 2t Σ ξ_j/(t-j²)`, including the interval below the first pole.
///
/// This exploratory search does not prove completeness: signed residues can
/// produce multiple roots per pole gap, and the final exterior window is finite.
/// `tol` controls absolute residual or t-bracket width, not physical-ordinate
/// error or comparison accuracy against a Riemann zero. For complete root counts
/// use the HP certified discovery route.
pub fn solve_spectrum_f64(
    xi: &[f64],
    n_max: usize,
    l: f64,
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>> {
    let expected = n_max.checked_mul(2).and_then(|v| v.checked_add(1));
    if expected != Some(xi.len())
        || xi.iter().any(|x| !x.is_finite())
        || !l.is_finite()
        || l <= 0.0
        || !tol.is_finite()
        || tol < 0.0
    {
        return Err(anyhow!("invalid f64 secular state, length, or tolerance"));
    }
    if n_max == 0 {
        return Ok(Vec::new());
    }
    let xi_pos: Vec<f64> = (0..=n_max).map(|j| xi[j + n_max]).collect();

    let f = |t: f64| -> f64 {
        let mut acc = xi_pos[0];
        let mut sum = 0.0_f64;
        for (j, &xij) in xi_pos.iter().enumerate().skip(1) {
            sum += xij / (t - (j as f64).powi(2));
        }
        acc += 2.0 * t * sum;
        acc
    };

    let mut roots = Vec::with_capacity(n_max);
    for k in 0..=n_max {
        let lo = (k as f64).powi(2);
        let hi = ((k + 1) as f64).powi(2);
        let eps = INTEGRAND_SINGULARITY_GUARD * (hi - lo);
        let a = lo + eps;
        let b = if k == n_max {
            lo + 1e6 * (lo + 1.0)
        } else {
            hi - eps
        };
        if let Some(t) = xc_numerics::root_finding::bisect_f64(&f, a, b, tol, max_iter) {
            roots.push((2.0 * std::f64::consts::PI / l) * t.sqrt());
        }
    }
    Ok(roots)
}

// Reference Riemann-zero literals below are quoted at published precision
// (more digits than f64 holds); the excess is harmless on parse.
#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn archimedean_origin_limit_matches_direct_integrand_extrapolation() {
        for cutoff in [2.0_f64, 7.0, 13.0, 1200.0] {
            let l = cutoff.ln();
            for (n, m) in [
                (0, 0),
                (1, 1),
                (-3, -3),
                (1, 0),
                (2, 1),
                (3, 1),
                (3, 2),
                (0, 1),
                (-1, 1),
                (3, -2),
            ] {
                // Extrapolate the original analytic integrand, never calling
                // its origin guard or substituting the proposed limit.
                let direct = |x: f64| {
                    let a = 2.0 * std::f64::consts::PI * n as f64 / l;
                    let b = 2.0 * std::f64::consts::PI * m as f64 / l;
                    let omega = if n == m {
                        2.0 * (1.0 - x / l) * (a * x).cos()
                    } else {
                        ((b * x).sin() - (a * x).sin()) / (std::f64::consts::PI * (n - m) as f64)
                    };
                    ((x / 2.0).exp() * omega - if n == m { 2.0 } else { 0.0 }) / (2.0 * x.sinh())
                };
                let h = l * 2.0_f64.powi(-20);
                let extrapolated = 3.0 * direct(h) - 3.0 * direct(2.0 * h) + direct(3.0 * h);
                let limit = archimedean_origin_limit_f64(n, m, l);
                assert!(
                    (limit - extrapolated).abs() < 1e-8 * (1.0 + limit.abs()),
                    "C={cutoff}, n={n}, m={m}: guard={limit}, extrapolated={extrapolated}"
                );
            }
        }
    }

    #[test]
    fn archimedean_origin_guard_is_inactive_for_integer_but_reachable_for_fractional_cutoffs() {
        let (nodes, _) = xc_numerics::quadrature::gl_nodes_weights_f64(64);
        let minimum = |p: CcmParams| {
            nodes
                .iter()
                .map(|x| 0.5 * p.log_length() * (1.0 + x))
                .fold(f64::INFINITY, f64::min)
        };
        // C=2 is the smallest valid integer cutoff; mapped nodes increase with C.
        assert!(minimum(CcmParams::from_lambda_sq_integer(2, 1)) > INTEGRAND_SINGULARITY_GUARD);
        assert!(
            minimum(CcmParams::from_lambda_sq_fractional(1.00000001, 1))
                < INTEGRAND_SINGULARITY_GUARD
        );
    }

    #[test]
    fn f64_matrix_matches_independent_high_mode_and_small_length_integrals() {
        // Independently integrated defining Weil distribution at 90 decimal
        // digits with mpmath tanh-sinh, including direct pole integration.
        // Reproduction: Research 2026-09-23-full-mathematics-revalidation.
        let params = CcmParams::from_lambda_sq_integer(13, 120);
        let matrix = build_tau_f64(&params, params.log_length(), 13).unwrap();
        for (n, m, expected) in [
            (64, 64, 4.182967328799878),
            (64, 63, 0.31200771112145644),
            (120, 120, 4.636760801387921),
            (120, 0, 0.00670592132993459),
        ] {
            let actual = matrix[(params.idx(n), params.idx(m))];
            assert!(
                (actual - expected).abs() < 2e-11,
                "n={n}, m={m}: actual={actual}, reference={expected}"
            );
        }
        // The reference uses the exact binary64 cutoff input, not a rounded
        // decimal surrogate. The former fixed origin guard is inaccurate here.
        let params = CcmParams::from_lambda_sq_fractional(1.00000001, 3);
        let matrix = build_tau_f64(&params, params.log_length(), 1).unwrap();
        for (n, m, expected) in [(3, 3, 19.522062402308804), (3, 2, 0.004117773807046028)] {
            let actual = matrix[(params.idx(n), params.idx(m))];
            assert!(
                (actual - expected).abs() < 2e-11,
                "small L n={n}, m={m}: actual={actual}, reference={expected}"
            );
        }
    }

    #[test]
    fn f64_selected_even_state_has_a_small_full_matrix_residual() {
        let params = CcmParams::from_lambda_sq_integer(13, 32);
        let result = run_f64(&params).unwrap();
        let matrix = build_tau_f64(&params, params.log_length(), 13).unwrap();
        let xi = nalgebra::DVector::from_column_slice(&result.xi);
        let residual = &matrix * &xi - result.weil_min_eigenvalue * &xi;
        assert!(residual.norm() / xi.norm() < 1e-11);
        for j in 0..=params.n_modes {
            assert_eq!(result.xi[params.n_modes + j], result.xi[params.n_modes - j]);
        }
    }

    #[test]
    fn f64_secular_search_includes_the_first_positive_gap() {
        // R(t)=(3t-1)/(t-1), with the exact positive ordinate 1/sqrt(3)
        // when L=2*pi. No reference-zero input or production root oracle.
        let roots = solve_spectrum_f64(&[1.0, 1.0, 1.0], 1, 2.0 * std::f64::consts::PI, 1e-14, 200)
            .unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - 1.0 / 3.0_f64.sqrt()).abs() < 1e-13);
        assert!(solve_spectrum_f64(&[f64::NAN; 3], 1, 1.0, 1e-14, 200).is_err());
        assert!(solve_spectrum_f64(&[1.0], 1, 1.0, 1e-14, 200).is_err());
    }

    #[test]
    fn f64_ccm_rejects_invalid_cutoffs_before_assembly() {
        for cutoff in [0.0, 0.5, 1.0, f64::NAN, f64::INFINITY] {
            assert!(run_f64(&CcmParams::from_lambda_sq_fractional(cutoff, 1)).is_err());
        }
    }

    #[test]
    fn f64_ccm_result_round_trips_without_loss() {
        let result = CcmResult {
            eigenvalues_pos: vec![14.134725141734695, 21.022039638771556],
            weil_min_eigenvalue: 1.0e-200,
            xi: vec![-0.125, 0.75],
            elapsed_seconds: 1.25,
        };
        let encoded = serde_json::to_vec(&result).unwrap();
        let decoded: CcmResult = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, result);
    }

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
            for _ in 0..j {
                prod *= p;
            }
            assert_eq!(k, prod, "k = p^j must hold; got k={}, p={}, j={}", k, p, j);
        }
    }

    /// CcmParams basic properties.
    #[test]
    fn ccm_params_basic() {
        let p = CcmParams::from_lambda_sq_integer(13, 120);
        assert!((p.lambda_squared() - 13.0).abs() < 1e-12);
        assert_eq!(p.matrix_size(), 241);
        assert_eq!(p.idx(0), 120);
        assert_eq!(p.idx(-120), 0);
        assert_eq!(p.idx(120), 240);
    }

    /// f64 tier should produce a non-empty spectrum with reasonable
    /// structure at λ²=13, N=120. Note: f64 tier uses bisection on R(t)
    /// without Newton-from-seed, so eigenvalues are Weil-form roots
    /// rather than Riemann zeros (those need the HP path).
    ///
    /// Skipped in debug builds: the unoptimized nalgebra `SymmetricEigen`
    /// on a 241×241 matrix allocates a stack frame for every intermediate
    /// value, exhausting even a large stack before finishing.
    ///
    /// Skipped always (ignored): building the 241×241 τ-matrix via
    /// O(N²) GL integrations is a multi-minute test even in release mode;
    /// run explicitly with `--include-ignored` when validating the f64 path.
    #[test]
    #[ignore = "f64 τ-matrix build (58K GL integrations) — too slow for routine test runs; use --include-ignored to run explicitly"]
    #[cfg(not(debug_assertions))]
    fn run_f64_produces_spectrum() {
        let params = CcmParams::from_lambda_sq_integer(13, 120);
        let result = run_f64(&params).unwrap();
        assert!(
            !result.eigenvalues_pos.is_empty(),
            "should produce at least one eigenvalue"
        );
        // ε_N should be tiny at λ²=13, N=120 even at f64.
        assert!(
            result.weil_min_eigenvalue.abs() < 1e-10,
            "ε_N = {} should be tiny",
            result.weil_min_eigenvalue
        );
        // ξ should be normalized (Σ ξ_j = √L).
        let sum_xi: f64 = result.xi.iter().sum();
        let l = 13.0_f64.ln();
        let expected_sum = l.sqrt();
        assert!(
            (sum_xi - expected_sum).abs() < 1e-3,
            "Σ ξ_j = {} should be √L = {}",
            sum_xi,
            expected_sum
        );
        // Eigenvalues should be sorted ascending.
        for w in result.eigenvalues_pos.windows(2) {
            assert!(w[0] < w[1], "eigenvalues should be ascending");
        }
        // Some eigenvalue should match the 2nd Riemann zero (21.02)
        // since it's outside the dense Weil-form region.
        let target = 21.022040;
        let closest_to_2nd = result
            .eigenvalues_pos
            .iter()
            .map(|&e| (e - target).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest_to_2nd < 0.01,
            "no eigenvalue near 21.02 (closest: {:.4e})",
            closest_to_2nd
        );
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
        let roots =
            solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        // With only ξ_0 nonzero, R(t) = ξ_0 = 1 (constant), no zeros.
        assert!(
            roots.is_empty(),
            "constant R(t) should have no roots, got {:?}",
            roots
        );
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
        let roots =
            solve_spectrum_f64(&xi, n_max, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        // R(t) = 2t/(t-1) for t > 1 is always positive, so no sign changes.
        assert!(
            roots.is_empty(),
            "single-mode R(t) should have no roots in standard intervals"
        );
    }

    /// CcmParams accessor methods.
    #[test]
    fn ccm_params_accessors() {
        let p = CcmParams::from_lambda_sq_integer(100, 50);
        assert!((p.lambda_squared() - 100.0).abs() < 1e-12);
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

    /// prime_powers_up_to output should be sorted ascending by k.
    #[test]
    fn prime_powers_output_is_sorted() {
        let pp = prime_powers_up_to(50);
        for w in pp.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "prime_powers_up_to should be sorted; {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }

    /// prime_powers_up_to: every (k, p, j) triple satisfies k = p^j exactly.
    #[test]
    fn prime_powers_triple_invariant() {
        for &(k, p, j) in &prime_powers_up_to(100) {
            let expected: u64 = (0..j).fold(1, |acc, _| acc * p);
            assert_eq!(k, expected, "triple ({}, {}, {}): k ≠ p^j", k, p, j);
        }
    }

    /// CcmParams::from_lambda_sq_integer(0) should not panic.
    /// lambda_sq_int will be 0 and prime_powers_up_to(0) returns empty.
    #[test]
    fn ccm_params_lambda_zero() {
        let p = CcmParams::from_lambda_sq_integer(0, 10);
        assert_eq!(p.lambda_sq_int(), 0);
        assert!(prime_powers_up_to(p.lambda_sq_int()).is_empty());
    }

    /// solve_spectrum_f64 with n_max=0 (degenerate ξ = [ξ_0]) should
    /// return an empty root list without panicking — there are no poles in
    /// R(t) and the bisection loop has no intervals to search.
    #[test]
    fn solve_spectrum_n_max_zero() {
        let l = 13.0_f64.ln();
        let xi = vec![1.0_f64]; // only ξ_0
        let roots =
            solve_spectrum_f64(&xi, 0, l, DEFAULT_BISECT_TOL, DEFAULT_BISECT_MAX_ITER).unwrap();
        assert!(roots.is_empty(), "n_max=0 should produce no roots");
    }
}

#[cfg(feature = "hp")]
pub mod state_geometry;

#[cfg(feature = "hp")]
pub mod retained_evidence;

/// Additional retained-source observations and external runtime research inputs.
#[cfg(feature = "hp")]
pub mod extended_research;

/// Complete retained-run convergence measurements and optional numerical models.
#[cfg(feature = "hp")]
pub mod convergence_capture;

#[cfg(feature = "hp")]
pub mod capture_runtime;
#[cfg(feature = "hp")]
pub mod research_completion;

#[cfg(feature = "hp")]
mod transform_enclosure;

#[cfg(feature = "hp")]
pub mod research_cohort;
#[cfg(feature = "hp")]
pub mod research_prepare;
#[cfg(feature = "hp")]
mod research_target;

#[cfg(feature = "hp")]
pub mod atom_research;

#[cfg(feature = "hp")]
mod band_runtime;

#[cfg(feature = "hp")]
mod research_export;
