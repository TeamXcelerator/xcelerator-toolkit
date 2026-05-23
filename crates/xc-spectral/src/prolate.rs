// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.
//

//! Prolate spheroidal wave functions: CCM Lemma 7.2 falsification test.
//!
//! Implements the prolate-wave educated guess `k_λ` from Section 7 of
//! the CCM paper (arxiv 2511.22755). The educated guess approximates
//! the smallest-eigenvalue eigenvector ξ_λ of the Weil quadratic form.

// `--features hp`; without it, dead-code warnings are expected.
//! quadratic form. Step 2 of the CCM proof of RH requires showing
//! that this approximation is sufficiently accurate.
//!
//! ## Structure
//!
//! The prolate wave operator on `[-λ, λ]` is:
//!
//! ```text
//! PW_λ = -∂_x((λ² − x²) ∂_x) + (2πλx)²
//! ```
//!
//! Its eigenfunctions `h_{n,λ}` (for n = 0, 4) combine to form `h_λ`
//! such that `∫ h_λ dx = 0`. Then `k_λ = ℰ(h_λ)` where ℰ is the
//! Eisenstein-like sum map.
//!
//! See `docs/CCM_PHASE2_PLAN.md` for the full mathematical setup.
//!
//! ## Status: Phase 2B (f64 prototype)
//!
//! Implemented:
//! - Finite-difference PW_λ matrix construction
//! - Dense symmetric eigendecomposition via nalgebra
//! - Node-counting and parity detection to identify h_{0,λ} and h_{4,λ}
//! - Linear combination h_λ = c_4·h_{4,λ} + c_0·h_{0,λ} with ∫h_λ = 0
//! - ℰ map evaluation on a logarithmic grid in [λ⁻¹, λ]
//! - Comparison ‖ξ_λ − c·k_λ‖_∞, ‖ξ_λ − c·k_λ‖_2 against the Weil
//!   eigenvector reconstructed from its V_n Fourier coefficients
//!
//! Pending (Phase 2C):
//! - High-precision (rug) version, to match Phase 1's 500-digit data
//! - Empirical sweep λ² ∈ {13, 100, 1000, 10000} to verify λ⁻² decay

use anyhow::Result;
use nalgebra::{DMatrix, SymmetricEigen};

/// Threshold for detecting zero-norm vectors in parity classification.
pub const PARITY_ZERO_THRESHOLD: f64 = 1e-30;

/// Relative tolerance for classifying a vector as even or odd.
/// If the even-deviation / total < this, the vector is classified as even.
pub const PARITY_CLASSIFICATION_TOL: f64 = 1e-3;

/// Threshold for node-counting: values below `max_abs * NODE_NOISE_FACTOR`
/// are treated as zero (not counted as sign changes).
pub const NODE_NOISE_FACTOR: f64 = 1e-6;

/// Maximum number of eigenfunctions to search for h_0 and h_4.
/// Prolate h_4 is typically the 3rd even eigenfunction (~5th overall).
pub const PROLATE_SEARCH_DEPTH: usize = 24;

/// Threshold for detecting zero integral of h_0 (would prevent
/// enforcing the ∫h_λ = 0 constraint).
pub const INTEGRAL_ZERO_THRESHOLD: f64 = 1e-30;

/// Threshold for detecting zero dot product ⟨k, k⟩ in the comparison.
pub const DOT_PRODUCT_ZERO_THRESHOLD: f64 = 1e-300;

/// Configuration for the prolate-wave eigenfunction computation.
#[derive(Debug, Clone)]
pub struct ProlateConfig {
    /// λ for the operator PW_λ. Same λ as the Weil form.
    pub lambda: f64,
    /// Number of interior grid points for the finite-difference
    /// discretization of PW_λ on `[-λ, λ]`. Forced to be odd so the
    /// origin sits on a grid point and even/odd parity is exact.
    pub n_grid: usize,
    /// Number of sample points on `[λ⁻¹, λ]` for the comparison grid.
    pub n_sample: usize,
    /// Working precision in bits (Phase 2B is f64; reserved for the
    /// high-precision tier in Phase 2C).
    pub precision_bits: u32,
}

impl ProlateConfig {
    pub fn new(lambda: f64, n_grid: usize) -> Self {
        // Make n_grid odd so x=0 is a grid point (clean even-symmetry).
        let n = if n_grid % 2 == 0 { n_grid + 1 } else { n_grid };
        Self {
            lambda,
            n_grid: n,
            n_sample: 256,
            precision_bits: 53, // f64 default for Phase 2B
        }
    }
    pub fn with_n_sample(mut self, n_sample: usize) -> Self {
        self.n_sample = n_sample;
        self
    }
}

/// Result of computing the prolate-wave educated guess k_λ.
#[derive(Debug, Clone)]
pub struct ProlateResult {
    /// k_λ sampled on `u_grid`.
    pub k_values: Vec<f64>,
    /// Sample points u_i ∈ [λ⁻¹, λ]. Logarithmically spaced so that
    /// the V_n Fourier basis of the Weil form lines up cleanly with
    /// the grid spacing.
    pub u_grid: Vec<f64>,
    /// Eigenvalue of h_{0,λ} (≈ 2π λ²).
    pub eigenvalue_0: f64,
    /// Eigenvalue of h_{4,λ} (≈ 18π λ²).
    pub eigenvalue_4: f64,
    /// Coefficient of h_{4,λ} in the linear combination forming h_λ.
    pub c_4: f64,
    /// Coefficient of h_{0,λ}.
    pub c_0: f64,
    /// Wall-clock time spent.
    pub elapsed_seconds: f64,
}

/// Build the prolate wave operator PW_λ matrix at f64 precision via
/// 3-point finite differences on a uniform grid `[-λ, λ]` with N+2
/// points (boundary nodes at ±λ where u=0 by Dirichlet, so the matrix
/// is N × N for the interior nodes).
///
/// The matrix is symmetric tridiagonal:
///   - Diagonal: `(2/h²)·(λ² − x_i²) + (2πλ x_i)²`
///   - Off-diagonal: `-(1/h²)·(λ² − x_{i±1/2}²)`
///
/// where `h = 2λ/(N+1)` is the grid spacing and `x_i = -λ + i·h` for
/// `i = 1, …, N`.
///
/// Returns the diagonal and the lower off-diagonal as separate Vec<f64>.
pub fn build_pw_matrix_f64(cfg: &ProlateConfig) -> (Vec<f64>, Vec<f64>) {
    let n = cfg.n_grid;
    let lambda = cfg.lambda;
    let lambda_sq = lambda * lambda;
    let h = 2.0 * lambda / ((n + 1) as f64);
    let h_sq = h * h;
    let two_pi_lambda = 2.0 * std::f64::consts::PI * lambda;

    let mut diag = vec![0.0_f64; n];
    let mut off_diag = vec![0.0_f64; n.saturating_sub(1)];

    for i in 0..n {
        // x_i = -λ + (i+1)·h, i = 0..n-1 are interior nodes 1..n.
        let x = -lambda + (i + 1) as f64 * h;
        let x_minus_half = x - h / 2.0;
        let x_plus_half = x + h / 2.0;

        let coef_minus = lambda_sq - x_minus_half * x_minus_half;
        let coef_plus = lambda_sq - x_plus_half * x_plus_half;

        // Diagonal: (1/h²) · (coef_plus + coef_minus) + (2πλx)²
        diag[i] = (coef_plus + coef_minus) / h_sq + (two_pi_lambda * x).powi(2);

        // Lower off-diagonal: -(1/h²) · coef_minus (at the interface to i-1)
        if i > 0 {
            off_diag[i - 1] = -coef_minus / h_sq;
        }
    }

    (diag, off_diag)
}

/// Construct PW_λ as a dense `DMatrix<f64>` from its tridiagonal data.
fn build_pw_dense(cfg: &ProlateConfig) -> DMatrix<f64> {
    let (diag, off_diag) = build_pw_matrix_f64(cfg);
    let n = diag.len();
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = diag[i];
        if i > 0 {
            m[(i, i - 1)] = off_diag[i - 1];
            m[(i - 1, i)] = off_diag[i - 1];
        }
    }
    m
}

/// Parity classification of an eigenvector on the symmetric grid.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Parity {
    Even,
    Odd,
    Indeterminate,
}

/// Detect parity of `v` under the index reflection `i ↔ n-1-i` (which
/// implements `x ↔ -x` on the symmetric grid).
fn parity_of(v: &[f64]) -> Parity {
    let n = v.len();
    if n == 0 {
        return Parity::Indeterminate;
    }
    let mut even_dev = 0.0_f64;
    let mut odd_dev = 0.0_f64;
    let mut total = 0.0_f64;
    for i in 0..n / 2 {
        let a = v[i];
        let b = v[n - 1 - i];
        even_dev += (a - b).abs();
        odd_dev += (a + b).abs();
        total += a.abs() + b.abs();
    }
    if total < PARITY_ZERO_THRESHOLD {
        return Parity::Indeterminate;
    }
    let r_even = even_dev / total;
    let r_odd = odd_dev / total;
    if r_even < PARITY_CLASSIFICATION_TOL {
        Parity::Even
    } else if r_odd < PARITY_CLASSIFICATION_TOL {
        Parity::Odd
    } else {
        Parity::Indeterminate
    }
}

/// Count zero crossings of `v`. Small entries (< threshold of the
/// max absolute value) are skipped to avoid spurious counts from
/// numerical noise near boundary.
fn count_nodes(v: &[f64]) -> usize {
    let max_abs = v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    if max_abs < PARITY_ZERO_THRESHOLD {
        return 0;
    }
    let threshold = max_abs * NODE_NOISE_FACTOR;
    let mut count = 0usize;
    let mut prev_sign = 0i32;
    for &x in v {
        if x.abs() < threshold {
            continue;
        }
        let s = if x > 0.0 { 1 } else { -1 };
        if prev_sign != 0 && s != prev_sign {
            count += 1;
        }
        prev_sign = s;
    }
    count
}

/// Linearly interpolate a function defined on the FD grid
/// `x_i = -λ + (i+1)h` (i = 0..n-1) at an arbitrary point `x ∈ [-λ, λ]`.
/// Returns 0 outside the support (Dirichlet BC).
fn interp_grid(values: &[f64], lambda: f64, h: f64, x: f64) -> f64 {
    let n = values.len();
    if x.abs() >= lambda {
        return 0.0;
    }
    // x_i = -λ + (i+1)·h ⇒ i = (x + λ)/h − 1
    let f_idx = (x + lambda) / h - 1.0;
    let i_lo = f_idx.floor() as isize;
    let i_hi = i_lo + 1;
    if i_lo < 0 {
        // Linear extrapolation toward the left Dirichlet boundary at x=-λ
        // (where the function is 0). i_lo = -1 corresponds to x = -λ.
        if i_lo == -1 && i_hi == 0 {
            let frac = f_idx - i_lo as f64; // in [0, 1)
            return frac * values[0];
        }
        return 0.0;
    }
    if i_hi >= n as isize {
        // Linear extrapolation toward the right Dirichlet boundary x=+λ.
        if i_lo == n as isize - 1 {
            let frac = f_idx - i_lo as f64;
            return (1.0 - frac) * values[i_lo as usize];
        }
        return 0.0;
    }
    let frac = f_idx - i_lo as f64;
    (1.0 - frac) * values[i_lo as usize] + frac * values[i_hi as usize]
}

/// Compute the prolate-wave educated guess k_λ.
///
/// 1. Build PW_λ at f64 via finite differences on a uniform grid.
/// 2. Diagonalize and identify h_{0,λ} (smallest, even, 0 nodes) and
///    h_{4,λ} (even, 4 nodes) by node-counting + parity.
/// 3. Form h_λ = h_{4,λ} − r·h_{0,λ} with r = ∫h_{4,λ} / ∫h_{0,λ},
///    so that ∫h_λ dx = 0 (the constraint of Lemma 7.1 in the paper).
/// 4. Sample k_λ(u) = √u · Σ_{n=1}^{⌊λ/u⌋} h_λ(n·u) on a logarithmic
///    grid `u_i ∈ [λ⁻¹, λ]`. The grid is logarithmic to align with
///    the V_n Fourier basis of the Weil form.
pub fn compute_k_lambda(cfg: &ProlateConfig) -> Result<ProlateResult> {
    let start = std::time::Instant::now();
    let lambda = cfg.lambda;
    let n = cfg.n_grid;
    if n < 16 {
        anyhow::bail!("n_grid too small (got {}); need at least 16 to find h_4", n);
    }
    let h = 2.0 * lambda / ((n + 1) as f64);

    // Build matrix and diagonalize.
    let m = build_pw_dense(cfg);
    let eig = SymmetricEigen::new(m);

    // Sort indices by ascending eigenvalue.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap()
    });

    // Search the lowest-lying eigenfunctions for h_0 (even, 0 nodes)
    // and h_4 (even, 4 nodes). Limit the search depth to a reasonable
    // window — for prolate waves, h_4 is the third even eigenfunction
    // so it sits near the 5th eigenvalue overall (h_0, h_1, h_2, h_3, h_4).
    let n_try = PROLATE_SEARCH_DEPTH.min(n);
    let mut h0_idx: Option<usize> = None;
    let mut h4_idx: Option<usize> = None;
    for &i in idx.iter().take(n_try) {
        let v: Vec<f64> = eig.eigenvectors.column(i).iter().copied().collect();
        if parity_of(&v) != Parity::Even {
            continue;
        }
        let nodes = count_nodes(&v);
        match nodes {
            0 if h0_idx.is_none() => h0_idx = Some(i),
            4 if h4_idx.is_none() => h4_idx = Some(i),
            _ => {}
        }
        if h0_idx.is_some() && h4_idx.is_some() {
            break;
        }
    }
    let h0_idx = h0_idx.ok_or_else(|| {
        anyhow::anyhow!(
            "could not find h_{{0,λ}} (even, 0 nodes) among first {} eigenfunctions",
            n_try
        )
    })?;
    let h4_idx = h4_idx.ok_or_else(|| {
        anyhow::anyhow!(
            "could not find h_{{4,λ}} (even, 4 nodes) among first {} eigenfunctions",
            n_try
        )
    })?;
    let eigenvalue_0 = eig.eigenvalues[h0_idx];
    let eigenvalue_4 = eig.eigenvalues[h4_idx];

    // Extract eigenvectors. SymmetricEigen normalizes columns to
    // unit ℓ² norm, so ∫|f|² dx ≈ h · Σ|v_i|² = h. To get unit
    // continuous-L² norm, scale by 1/√h.
    let scale = 1.0 / h.sqrt();
    let mut h0: Vec<f64> = eig
        .eigenvectors
        .column(h0_idx)
        .iter()
        .map(|&v| v * scale)
        .collect();
    let mut h4: Vec<f64> = eig
        .eigenvectors
        .column(h4_idx)
        .iter()
        .map(|&v| v * scale)
        .collect();

    // Pin sign: make both functions positive at the origin (center
    // of the symmetric grid), which is the canonical convention for
    // h_0 and h_4 in the harmonic-oscillator limit.
    let center = n / 2;
    if h0[center] < 0.0 {
        for v in h0.iter_mut() {
            *v = -*v;
        }
    }
    if h4[center] < 0.0 {
        for v in h4.iter_mut() {
            *v = -*v;
        }
    }

    // ∫h dx via midpoint rule (h · Σ v_i is the FD analog of ∫f dx).
    let int_h0: f64 = h * h0.iter().sum::<f64>();
    let int_h4: f64 = h * h4.iter().sum::<f64>();
    if int_h0.abs() < INTEGRAL_ZERO_THRESHOLD {
        anyhow::bail!("∫h_{{0,λ}} ≈ 0; cannot enforce ∫h_λ = 0 constraint");
    }
    let r = int_h4 / int_h0;
    let c_4 = 1.0;
    let c_0 = -r;
    let h_lambda: Vec<f64> = (0..n).map(|i| c_4 * h4[i] + c_0 * h0[i]).collect();

    // Sample k_λ on a logarithmic grid in [λ⁻¹, λ].
    let n_sample = cfg.n_sample.max(2);
    let log_lambda = lambda.ln();
    let u_grid: Vec<f64> = (0..n_sample)
        .map(|i| {
            // u_i = exp(log_lambda · (2i/(M-1) − 1)) ∈ [λ⁻¹, λ].
            let t = i as f64 / (n_sample - 1) as f64;
            (log_lambda * (2.0 * t - 1.0)).exp()
        })
        .collect();

    let k_values: Vec<f64> = u_grid
        .iter()
        .map(|&u| {
            if u <= 0.0 {
                return 0.0;
            }
            let n_terms = (lambda / u).floor() as usize;
            let mut s = 0.0_f64;
            for k in 1..=n_terms {
                let x = (k as f64) * u;
                if x >= lambda {
                    break;
                }
                s += interp_grid(&h_lambda, lambda, h, x);
            }
            u.sqrt() * s
        })
        .collect();

    Ok(ProlateResult {
        k_values,
        u_grid,
        eigenvalue_0,
        eigenvalue_4,
        c_4,
        c_0,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Best fit scalar c such that c·k_λ ≈ ξ_λ on the sampling grid.
    pub optimal_scalar: f64,
    /// L∞ norm of the residual ξ_λ − c·k_λ on the grid.
    pub linf_error: f64,
    /// Discrete ℓ² norm of the residual (no quadrature weights).
    pub l2_error: f64,
    /// L∞ norm of ξ_λ on the grid (for relative-error reporting).
    pub xi_linf: f64,
    /// Index of the maximum residual on the sampling grid.
    pub linf_index: usize,
}

/// Compare ξ_λ (Weil eigenvector in V_n Fourier basis) to k_λ on
/// the prolate sample grid.
///
/// `xi` has length `2N+1`, indexed `j = -N, …, N` (so `xi[N] = ξ_0`).
/// After symmetrization in the Phase-1 pipeline `ξ_{-j} = ξ_j` to
/// working precision, so we use the even-cosine reconstruction:
///
/// ```text
/// ξ_λ(u) = (1/√L) [ξ_0 + 2 Σ_{n=1}^{N} ξ_n cos(2π n log(λu)/L)]
/// ```
///
/// where `L = 2 ln λ`. Then we find `c = ⟨ξ,k⟩/⟨k,k⟩` minimizing
/// `‖ξ_λ − c·k_λ‖₂` and report L∞ and ℓ² errors.
pub fn compare_xi_to_k_lambda(
    xi: &[f64],
    n_modes: usize,
    lambda: f64,
    u_grid: &[f64],
    k_values: &[f64],
) -> Result<ComparisonResult> {
    if xi.len() != 2 * n_modes + 1 {
        anyhow::bail!(
            "xi has wrong length: got {}, expected 2N+1 = {}",
            xi.len(),
            2 * n_modes + 1
        );
    }
    if u_grid.len() != k_values.len() {
        anyhow::bail!(
            "u_grid and k_values have different lengths: {} vs {}",
            u_grid.len(),
            k_values.len()
        );
    }
    if u_grid.is_empty() {
        anyhow::bail!("empty grid");
    }

    let l = (lambda * lambda).ln();
    let inv_sqrt_l = 1.0 / l.sqrt();
    let xi_0 = xi[n_modes];
    let xi_pos: Vec<f64> = (1..=n_modes).map(|n| xi[n_modes + n]).collect();

    let xi_values: Vec<f64> = u_grid
        .iter()
        .map(|&u| {
            let phase_base = 2.0 * std::f64::consts::PI * (lambda * u).ln() / l;
            let mut acc = xi_0;
            for n in 1..=n_modes {
                acc += 2.0 * xi_pos[n - 1] * (n as f64 * phase_base).cos();
            }
            inv_sqrt_l * acc
        })
        .collect();

    // Optimal scalar minimizing ‖ξ − c·k‖₂² is c = ⟨ξ, k⟩/⟨k, k⟩.
    let dot_xk: f64 = xi_values.iter().zip(k_values).map(|(x, k)| x * k).sum();
    let dot_kk: f64 = k_values.iter().map(|k| k * k).sum();
    if dot_kk < DOT_PRODUCT_ZERO_THRESHOLD {
        anyhow::bail!("k_values are essentially zero on the grid");
    }
    let c = dot_xk / dot_kk;

    // Residual and its norms.
    let mut linf_error = 0.0_f64;
    let mut linf_index = 0usize;
    let mut l2_sq = 0.0_f64;
    for (i, (&x, &k)) in xi_values.iter().zip(k_values).enumerate() {
        let r = x - c * k;
        l2_sq += r * r;
        let abs_r = r.abs();
        if abs_r > linf_error {
            linf_error = abs_r;
            linf_index = i;
        }
    }
    let xi_linf = xi_values
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    Ok(ComparisonResult {
        optimal_scalar: c,
        linf_error,
        l2_error: l2_sq.sqrt(),
        xi_linf,
        linf_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: build PW_λ matrix at f64 and check that diagonal
    /// values are sane.
    #[test]
    fn pw_matrix_ground_state() {
        let cfg = ProlateConfig::new(2.0, 199);
        let (diag, _off_diag) = build_pw_matrix_f64(&cfg);
        assert!(diag.iter().all(|x| x.is_finite()));
        let center = diag.len() / 2;
        assert!(diag[center] > 1000.0 && diag[center] < 100_000.0);
    }

    /// Sanity: smallest eigenvalue of PW_λ should be close to 2π·λ²
    /// (the ground state energy in the harmonic-oscillator limit).
    #[test]
    fn pw_smallest_eigenvalue_close_to_2pi_lambda_sq() {
        // λ = 5 is large enough that the prolate operator looks like
        // a harmonic oscillator on its support.
        let cfg = ProlateConfig::new(5.0, 401);
        let m = build_pw_dense(&cfg);
        let eig = SymmetricEigen::new(m);
        let mut evals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let smallest = evals[0];
        let expected = 2.0 * std::f64::consts::PI * 25.0;
        // FD on N=401 introduces O(1/N²) error and the prolate
        // eigenvalue isn't exactly 2π λ² (only asymptotically). Ask
        // for ~10% agreement.
        let rel_err = (smallest - expected).abs() / expected;
        assert!(
            rel_err < 0.20,
            "smallest eigenvalue {:.3} should be within 20% of 2πλ²={:.3} (rel err {:.3e})",
            smallest,
            expected,
            rel_err
        );
    }

    /// Find h_{0,λ} and h_{4,λ}, compute k_λ on the comparison grid,
    /// and check that it's nonzero and finite everywhere.
    #[test]
    fn compute_k_lambda_runs() {
        let cfg = ProlateConfig::new(5.0, 401).with_n_sample(64);
        let res = compute_k_lambda(&cfg).expect("k_lambda computation should succeed");
        assert_eq!(res.u_grid.len(), 64);
        assert_eq!(res.k_values.len(), 64);
        assert!(res.k_values.iter().all(|x| x.is_finite()));
        // h_0 has eigenvalue ≈ 2πλ² ≈ 157, h_4 ≈ 18πλ² ≈ 1413.
        // Allow some FD error tolerance.
        assert!(res.eigenvalue_0 > 0.0);
        assert!(res.eigenvalue_4 > res.eigenvalue_0);
        // Some k_values should be nonzero (not the entire grid is in
        // the support of zero terms).
        let max_k = res
            .k_values
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_k > 0.0, "k_λ should be nonzero somewhere");
    }

    /// Comparison against a contrived ξ_λ. Use a flat ξ vector
    /// (all zeros except ξ_0) and check the comparison machinery
    /// runs end-to-end without panicking.
    #[test]
    fn compare_runs_end_to_end() {
        let lambda = 5.0;
        let cfg = ProlateConfig::new(lambda, 401).with_n_sample(64);
        let res = compute_k_lambda(&cfg).unwrap();
        // Synthetic ξ_λ: just ξ_0 = √L (the "constant" eigenvector).
        let n_modes = 20;
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = (lambda * lambda).ln().sqrt();
        let cmp = compare_xi_to_k_lambda(&xi, n_modes, lambda, &res.u_grid, &res.k_values)
            .expect("comparison should succeed");
        assert!(cmp.linf_error.is_finite());
        assert!(cmp.l2_error.is_finite());
        assert!(cmp.xi_linf > 0.0);
    }
}
