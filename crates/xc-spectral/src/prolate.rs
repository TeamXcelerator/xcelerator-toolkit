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
//! ## Status: complete (HP version available below)
//!
//! Implemented:
//! - Finite-difference PW_λ matrix construction
//! - Dense symmetric eigendecomposition via nalgebra
//! - Node-counting and parity detection to identify h_{0,λ} and h_{4,λ}
//! - Linear combination h_λ = c_4·h_{4,λ} + c_0·h_{0,λ} with ∫h_λ = 0
//! - ℰ map evaluation on a logarithmic grid in [λ⁻¹, λ]
//! - Comparison ‖ξ_λ − c·k_λ‖_∞, ‖ξ_λ − c·k_λ‖_2 against the Weil
//!   eigenvector reconstructed from its V_n Fourier coefficients
//! - High-precision (rug) version (`prolate::hp` submodule) with
//!   truly-dynamic working precision (HP-200 through HP-5000+).

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
    /// Working precision in bits. The f64 path uses 53; the HP path
    /// (`prolate::hp`) takes precision via its own HP config.
    pub precision_bits: u32,
}

impl ProlateConfig {
    /// Construct a config with default `n_sample = 256` and
    /// `precision_bits = 53` (f64). `n_grid` is rounded up to the
    /// next odd integer so `x = 0` lands on a grid point and the
    /// even/odd parity classification is exact.
    pub fn new(lambda: f64, n_grid: usize) -> Self {
        // Make n_grid odd so x=0 is a grid point (clean even-symmetry).
        let n = if n_grid % 2 == 0 { n_grid + 1 } else { n_grid };
        Self {
            lambda,
            n_grid: n,
            n_sample: 256,
            precision_bits: 53, // f64 default
        }
    }
    /// Override the comparison-grid size.
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
fn build_pw_dense_f64(cfg: &ProlateConfig) -> DMatrix<f64> {
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
fn parity_of_f64(v: &[f64]) -> Parity {
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
fn count_nodes_f64(v: &[f64]) -> usize {
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
fn interp_grid_f64(values: &[f64], lambda: f64, h: f64, x: f64) -> f64 {
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
pub fn compute_k_lambda_f64(cfg: &ProlateConfig) -> Result<ProlateResult> {
    let start = std::time::Instant::now();
    let lambda = cfg.lambda;
    let n = cfg.n_grid;
    if n < 16 {
        anyhow::bail!("n_grid too small (got {}); need at least 16 to find h_4", n);
    }
    let h = 2.0 * lambda / ((n + 1) as f64);

    // Build matrix and diagonalize.
    let m = build_pw_dense_f64(cfg);
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
        if parity_of_f64(&v) != Parity::Even {
            continue;
        }
        let nodes = count_nodes_f64(&v);
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
                s += interp_grid_f64(&h_lambda, lambda, h, x);
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
pub fn compare_xi_to_k_lambda_f64(
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
        let m = build_pw_dense_f64(&cfg);
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
        let res = compute_k_lambda_f64(&cfg).expect("k_lambda computation should succeed");
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
        let res = compute_k_lambda_f64(&cfg).unwrap();
        // Synthetic ξ_λ: just ξ_0 = √L (the "constant" eigenvector).
        let n_modes = 20;
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = (lambda * lambda).ln().sqrt();
        let cmp = compare_xi_to_k_lambda_f64(&xi, n_modes, lambda, &res.u_grid, &res.k_values)
            .expect("comparison should succeed");
        assert!(cmp.linf_error.is_finite());
        assert!(cmp.l2_error.is_finite());
        assert!(cmp.xi_linf > 0.0);
    }
}


// ===========================================================================
// High-precision (HP) prolate-wave operator and educated-guess pipeline.
//
// Mirrors the f64 prototype above, but operates entirely in `rug::Float`
// arithmetic with truly-dynamic working precision (HP-200 through HP-5000+).
// Uses the HP eigensolver in `xc_numerics::eigen` for tridiagonal QR and
// shifted inverse iteration on the FD-discretized PW_λ matrix.
// ===========================================================================

#[cfg(feature = "hp")]
pub mod hp {
    use anyhow::Result;
    use rayon::prelude::*;
    use rug::Float;
    use xc_numerics::eigen::{
        tridiag_eigenvalues_hp, tridiag_eigenvector_for_value_hp,
        TridiagEigvecOptions,
    };
    use xc_numerics::quadrature::CacheMode;

    /// Base raw URL of the public consolidated prolate-eigenvalue cache
    /// repository. Files live at
    /// `{REMOTE_BASE}/prolate_eigvals_cache/prec{P}/lambda_sq{L}/ngrid{B}-{B+999}/lambda_sq{L}_ngrid{N}_prec{P}.json.zip`
    /// where `B = (N / 1000) * 1000`.
    const PROLATE_REMOTE_BASE: &str =
        "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-prolate-eigvals-cache/main";

    /// HP-build of the FD prolate-wave tridiagonal data.
    ///
    /// Same finite-difference formulation as `build_pw_matrix_f64`, but
    /// every arithmetic step happens in `rug::Float` at the working
    /// precision `prec`. Boundary nodes at ±λ have Dirichlet BC.
    ///
    /// The grid spacing is `h = 2λ / (N+1)`. Interior nodes are
    /// `x_i = -λ + (i+1)·h` for `i = 0..N-1`.
    ///
    /// Diagonal: `(coef_plus + coef_minus) / h² + (2π λ x_i)²` where
    /// `coef_± = λ² - x_{i±1/2}²`.
    /// Lower off-diagonal: `-coef_minus / h²`.
    pub fn build_pw_matrix(
        lambda: &Float,
        n_grid: usize,
        prec: u32,
    ) -> (Vec<Float>, Vec<Float>) {
        // n_grid forced odd so x=0 is on the grid.
        let n = if n_grid % 2 == 0 { n_grid + 1 } else { n_grid };

        let lambda_sq = {
            let mut t = lambda.clone();
            t *= lambda;
            t
        };
        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        let mut two_pi_lambda = pi_v.clone();
        two_pi_lambda *= 2u32;
        two_pi_lambda *= lambda;

        // h = 2λ / (n+1)
        let mut h = lambda.clone();
        h *= 2u32;
        let n_plus_1 = Float::with_val(prec, (n + 1) as u32);
        h /= &n_plus_1;
        let h_sq = {
            let mut t = h.clone();
            t *= &h;
            t
        };
        let half_h = {
            let mut t = h.clone();
            t /= 2u32;
            t
        };

        let mut diag: Vec<Float> = (0..n).map(|_| Float::with_val(prec, 0)).collect();
        let mut off_diag: Vec<Float> = (0..n.saturating_sub(1))
            .map(|_| Float::with_val(prec, 0))
            .collect();

        for i in 0..n {
            // x_i = -λ + (i+1)·h  computed as: x = (i+1)·h - λ
            let mut x = h.clone();
            let i_plus_1 = Float::with_val(prec, (i + 1) as u32);
            x *= &i_plus_1;
            x -= lambda;

            // x ± h/2
            let mut x_minus_half = x.clone();
            x_minus_half -= &half_h;
            let mut x_plus_half = x.clone();
            x_plus_half += &half_h;

            // coef_minus = λ² - (x - h/2)²
            let mut coef_minus = lambda_sq.clone();
            let mut tmp = x_minus_half.clone();
            tmp *= &x_minus_half;
            coef_minus -= &tmp;

            // coef_plus = λ² - (x + h/2)²
            let mut coef_plus = lambda_sq.clone();
            let mut tmp = x_plus_half.clone();
            tmp *= &x_plus_half;
            coef_plus -= &tmp;

            // diagonal term: (coef_plus + coef_minus) / h² + (2π λ x)²
            let mut sum = coef_plus.clone();
            sum += &coef_minus;
            sum /= &h_sq;
            let mut potential = two_pi_lambda.clone();
            potential *= &x;
            let mut potential_sq = potential.clone();
            potential_sq *= &potential;
            sum += &potential_sq;
            diag[i] = sum;

            // off-diagonal term: -coef_minus / h²
            if i > 0 {
                let mut off = coef_minus.clone();
                off /= &h_sq;
                off = -off;
                off_diag[i - 1] = off;
            }
        }

        (diag, off_diag)
    }

    /// Detect parity of vector `v` under index reflection `i ↔ n-1-i`.
    /// Returns Even if `‖v - γv‖ / ‖v‖ < tol`, Odd if `‖v + γv‖ / ‖v‖ < tol`,
    /// Indeterminate otherwise. All HP arithmetic.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub enum HpParity {
        /// `v(x) ≈ v(-x)` to working precision.
        Even,
        /// `v(x) ≈ -v(-x)` to working precision.
        Odd,
        /// Neither even nor odd (mixed-parity or below the zero
        /// threshold).
        Indeterminate,
    }

    /// Classify the parity of HP vector `v` under the index reflection
    /// `i ↔ n-1-i` (which corresponds to `x ↔ -x` on the symmetric FD
    /// grid). Returns `HpParity::Indeterminate` when the vector's
    /// total mass is below the zero threshold or both even/odd
    /// deviations exceed the classification tolerance.
    pub fn parity_of(v: &[Float], prec: u32) -> HpParity {
        let n = v.len();
        if n == 0 {
            return HpParity::Indeterminate;
        }
        // Build threshold: 1e-30 (HP literal).
        let zero_thresh = Float::with_val(prec, Float::parse("1e-30").unwrap());
        let class_tol = Float::with_val(prec, Float::parse("1e-3").unwrap());

        let mut even_dev = Float::with_val(prec, 0);
        let mut odd_dev = Float::with_val(prec, 0);
        let mut total = Float::with_val(prec, 0);
        for i in 0..(n / 2) {
            let a = &v[i];
            let b = &v[n - 1 - i];
            let mut diff = a.clone();
            diff -= b;
            even_dev += diff.abs();
            let mut sum = a.clone();
            sum += b;
            odd_dev += sum.abs();
            total += a.clone().abs();
            total += b.clone().abs();
        }
        if total < zero_thresh {
            return HpParity::Indeterminate;
        }
        let mut r_even = even_dev.clone();
        r_even /= &total;
        let mut r_odd = odd_dev.clone();
        r_odd /= &total;
        if r_even < class_tol {
            HpParity::Even
        } else if r_odd < class_tol {
            HpParity::Odd
        } else {
            HpParity::Indeterminate
        }
    }

    /// Count zero crossings of HP vector `v`. Values below
    /// `max_abs * 1e-6` are skipped (boundary noise).
    pub fn count_nodes(v: &[Float], prec: u32) -> usize {
        if v.is_empty() {
            return 0;
        }
        let zero_thresh = Float::with_val(prec, Float::parse("1e-30").unwrap());
        let noise_factor = Float::with_val(prec, Float::parse("1e-6").unwrap());

        // max_abs
        let mut max_abs = Float::with_val(prec, 0);
        for x in v {
            let a = x.clone().abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs < zero_thresh {
            return 0;
        }
        let mut threshold = max_abs.clone();
        threshold *= &noise_factor;

        let mut count = 0usize;
        let mut prev_sign = 0i32;
        let zero = Float::with_val(prec, 0);
        for x in v {
            let abs_x = x.clone().abs();
            if abs_x < threshold {
                continue;
            }
            let s: i32 = if *x > zero { 1 } else { -1 };
            if prev_sign != 0 && s != prev_sign {
                count += 1;
            }
            prev_sign = s;
        }
        count
    }

    /// Linearly interpolate HP function `values` defined on the FD grid
    /// `x_i = -λ + (i+1)h` at point `x ∈ [-λ, λ]`. Returns zero outside
    /// the support (Dirichlet BC).
    ///
    /// `h` is the grid spacing as an HP value.
    pub fn interp_grid(
        values: &[Float],
        lambda: &Float,
        h: &Float,
        x: &Float,
        prec: u32,
    ) -> Float {
        let n = values.len();
        let zero = Float::with_val(prec, 0);

        // If |x| >= λ, return zero.
        let abs_x = x.clone().abs();
        if abs_x >= *lambda {
            return zero;
        }

        // f_idx = (x + λ) / h - 1  (HP arithmetic)
        let mut f_idx = x.clone();
        f_idx += lambda;
        f_idx /= h;
        f_idx -= 1u32;

        // i_lo = floor(f_idx) — convert HP Float floor-result to i64 via
        // the Integer trait (requires the `integer` feature on rug, which
        // the workspace enables).
        let f_idx_floor = f_idx.clone().floor();
        let i_lo_f = f_idx_floor.clone();
        // to_integer() returns Option<Integer> (None for NaN/Inf — our
        // values are always finite).
        let i_lo_int = i_lo_f.to_integer().expect("interp f_idx must be finite");
        let i_lo: i64 = i_lo_int.to_i64().unwrap_or(0);
        let i_hi = i_lo + 1;

        // Compute fractional part: frac = f_idx - i_lo (still HP).
        let mut frac = f_idx.clone();
        frac -= &i_lo_f;

        if i_lo < 0 {
            // Linear extrapolation toward x=-λ where f=0.
            // i_lo == -1 corresponds to x = -λ (Dirichlet boundary).
            if i_lo == -1 && i_hi == 0 {
                let mut result = frac.clone();
                result *= &values[0];
                return result;
            }
            return zero;
        }
        if i_hi >= n as i64 {
            if i_lo == (n as i64) - 1 {
                let mut t = Float::with_val(prec, 1);
                t -= &frac;
                t *= &values[i_lo as usize];
                return t;
            }
            return zero;
        }
        // (1 - frac) · v[i_lo] + frac · v[i_hi]
        let mut one_minus_frac = Float::with_val(prec, 1);
        one_minus_frac -= &frac;
        let mut result = one_minus_frac;
        result *= &values[i_lo as usize];
        let mut term2 = frac.clone();
        term2 *= &values[i_hi as usize];
        result += &term2;
        result
    }

    // ===========================================================================
    // Prolate eigenvalue cache
    // ===========================================================================
    //
    // The dominant cost in `compute_k_lambda` at HP-1000 is the full
    // tridiagonal QR on PW_λ — at N=8001 prec=3338 this is ~30 minutes
    // of wall-time. The output is just the eigenvalue vector (a few MB
    // serialized at HP-1000). It's deterministic in `(λ², n_grid, prec)`
    // and reusable across runs at the same configuration.
    //
    // Cache layout (mirrors xc-numerics::quadrature::hp gl_cache):
    //   <cwd>/data/prolate_eigvals_cache/lambda_sq{LSQ}_ngrid{N}_prec{P}.json[.zip]
    //
    // Lookup priority (governed by CacheMode, default DynamicFetch):
    //   1. Uncompressed `.json` (all non-Off modes)
    //   2. Local `.json.zip` (JsonZip, DynamicFetch; auto-decompressed
    //      on first read; the decompressed `.json` is written next to
    //      the .zip for future fast reads)
    //   3. Remote `.json.zip` from the public consolidated repo
    //      (DynamicFetch only; single zip, no byte-split — the
    //      eigenvalue vector is small)
    //   4. Compute fresh; write to (1)+(2) per the mode.
    //
    // Cache key is `λ²_int = round(λ²)` — only used when the round-trip
    // is exact (i.e. λ² is an integer like 13, 100, 1000 in our paper
    // configs). Non-integer λ² silently bypasses the cache.

    /// Tolerance for structural identity checks on a loaded prolate
    /// eigenvalue cache file. Mirror of `quadrature::hp` cache check
    /// tolerance.
    fn prolate_cache_tol(prec: u32) -> Float {
        use rug::ops::Pow;
        Float::with_val(prec, 2).pow(-((prec as i32) - 8))
    }

    /// Verify a loaded eigenvalue vector satisfies the prolate-spectrum
    /// structural identities:
    ///
    ///   1. Count = `n_grid` (after rounding to odd if the caller's
    ///      n_grid was even).
    ///   2. Ascending order: `e[k] ≤ e[k+1]` for all `k`.
    ///   3. The smallest eigenvalue ≈ 2π·λ² (the asymptotic prediction
    ///      for the prolate operator's ground state). At our scales the
    ///      relative deviation is ≲ 1% — far above any cache-corruption
    ///      noise.
    ///
    /// Returns `None` if all identities hold; `Some(reason)` otherwise.
    fn prolate_cache_structural_check(
        evals: &[Float],
        n_expected: usize,
        lambda_sq: u64,
        prec: u32,
    ) -> Option<String> {
        if evals.len() != n_expected {
            return Some(format!(
                "eigenvalue count {} != expected {}", evals.len(), n_expected
            ));
        }

        // Ascending order check.
        for k in 0..(evals.len().saturating_sub(1)) {
            if evals[k] > evals[k + 1] {
                return Some(format!(
                    "not ascending at index {}: e[{}] > e[{}]",
                    k, k, k + 1
                ));
            }
        }

        // Ground state ≈ 2π·λ². Tolerance is loose: 5% relative,
        // since the FD discretization itself contributes ~0.1-1%
        // deviation from the continuum value.
        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        let mut expected_e0 = pi_v.clone();
        expected_e0 *= 2u32;
        expected_e0 *= Float::with_val(prec, lambda_sq);
        let mut diff = evals[0].clone();
        diff -= &expected_e0;
        let abs_diff = diff.abs();
        let mut tol_rel = expected_e0.clone().abs();
        tol_rel *= 5u32;
        tol_rel /= 100u32;  // 5% of expected
        if !abs_diff.cmp_abs(&tol_rel).map(|o| o.is_lt()).unwrap_or(false) {
            return Some(format!(
                "ground state {} deviates from 2πλ² ≈ {} by {} (5% tol)",
                evals[0], expected_e0, abs_diff
            ));
        }

        // Per-element finiteness/precision sanity (cheap belt-and-
        // suspenders against hot-tip values like NaN slipping through
        // the JSON parser).
        let tol = prolate_cache_tol(prec);
        for (k, e) in evals.iter().enumerate() {
            if e.is_nan() {
                return Some(format!("eigenvalue {} is NaN", k));
            }
            if e.is_infinite() {
                return Some(format!("eigenvalue {} is infinite", k));
            }
            // Suppress unused warning — tol is passed for parity with
            // future identity checks; not currently used.
            let _ = &tol;
        }

        None
    }

    /// Round-trip check: for the cache key to be valid, `λ²` must
    /// round to an exact non-negative integer to within working
    /// precision tolerance. Returns `None` if the value isn't an
    /// integer (or is negative); the cache layer treats this as
    /// "cache disabled for this caller" and falls through to compute.
    fn lambda_sq_int_for_key(lambda_sq: &Float) -> Option<u64> {
        if lambda_sq.is_sign_negative() || lambda_sq.is_zero() {
            return None;
        }
        // round(λ²) and compare back.
        let rounded = lambda_sq.clone().round();
        let mut diff = lambda_sq.clone();
        diff -= &rounded;
        let abs_diff = diff.abs();
        // Generous tolerance: 1e-10. λ² in our paper configs is
        // exactly integer (13, 100, 1000); this rounds at f64 → HP
        // round-trip noise (~1e-15 at most).
        let tol = Float::with_val(lambda_sq.prec(), rug::Float::parse("1e-10").unwrap());
        if !abs_diff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return None;
        }
        // Convert rounded HP → u64. to_integer().to_u64() handles this.
        rounded.to_integer().and_then(|i| i.to_u64())
    }

    fn prolate_cache_dir() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let dir = cwd.join("data").join("prolate_eigvals_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    fn prolate_cache_filename(lambda_sq: u64, n_grid: usize, prec: u32) -> String {
        format!("lambda_sq{}_ngrid{}_prec{}.json", lambda_sq, n_grid, prec)
    }

    fn prolate_cache_path(lambda_sq: u64, n_grid: usize, prec: u32) -> Option<std::path::PathBuf> {
        prolate_cache_dir().map(|d| d.join(prolate_cache_filename(lambda_sq, n_grid, prec)))
    }

    fn prolate_cache_zip_path(lambda_sq: u64, n_grid: usize, prec: u32) -> Option<std::path::PathBuf> {
        prolate_cache_dir().map(|d| {
            let f = prolate_cache_filename(lambda_sq, n_grid, prec);
            d.join(format!("{}.zip", f))
        })
    }

    /// Parse a prolate eigenvalue cache JSON: a single JSON array of
    /// decimal strings, length `n_expected`, each parseable at the
    /// requested precision. Returns `None` on any structural mismatch.
    fn parse_prolate_cache_json(data: &str, n_expected: usize, prec: u32) -> Option<Vec<Float>> {
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let arr = parsed.as_array()?;
        if arr.len() != n_expected { return None; }
        let mut evals = Vec::with_capacity(n_expected);
        for s in arr {
            evals.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some(evals)
    }

    fn warn_prolate_cache_skip(path: &std::path::Path, reason: &str) {
        eprintln!(
            "[prolate_cache] WARNING: skipping {} ({}); recomputing",
            path.display(), reason
        );
    }

    fn load_prolate_eigvals_from_zip(
        zip_path: &std::path::Path,
        json_filename: &str,
        n_expected: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, String)> {
        use std::io::Read;
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        let mut entry = archive.by_name(json_filename).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_prolate_cache_json(&data, n_expected, prec)?;
        Some((parsed, data))
    }

    /// Try to load prolate eigenvalues from cache for `(λ², n_grid, prec)`.
    /// Returns `None` on cache miss, parse failure, or structural
    /// validation failure (with diagnostic warning to stderr in the
    /// failure cases). The lookup depth is governed by `mode`:
    ///   - `Off`          — never read (returns `None` immediately).
    ///   - `JsonOnly`     — local `.json` only.
    ///   - `JsonZip`      — local `.json` then local `.json.zip`.
    ///   - `DynamicFetch` — the above, then a remote fetch from the
    ///                      public consolidated repo.
    fn load_prolate_eigvals_cache(
        lambda_sq: u64, n_grid: usize, prec: u32, mode: CacheMode,
    ) -> Option<Vec<Float>> {
        if mode == CacheMode::Off { return None; }

        // Caches are zip-only: read straight from the .json.zip
        // (decompress in memory), never write a decompressed .json.
        // JsonOnly is now a read no-op (kept for API compatibility).
        if mode == CacheMode::JsonOnly { return None; }

        // Tier 1 (JsonZip, DynamicFetch): local zip — in memory.
        if let Some(evals) = try_load_local_prolate_zip(lambda_sq, n_grid, prec) {
            return Some(evals);
        }

        // JsonZip stops after the local tiers.
        if mode == CacheMode::JsonZip { return None; }

        // Tier 2 (DynamicFetch only): remote fetch from the public
        // consolidated prolate-eigvals cache repo. On success the
        // `.json.zip` lands locally and we re-run the local-zip loader.
        if fetch_remote_prolate_zip(lambda_sq, n_grid, prec) {
            if let Some(evals) = try_load_local_prolate_zip(lambda_sq, n_grid, prec) {
                return Some(evals);
            }
        }

        None
    }

    /// Load from a local `.json.zip`. Decompresses in memory; does NOT
    /// write a decompressed `.json`. Returns the validated eigenvalues,
    /// or `None` if the zip is absent, corrupt, or structurally invalid.
    fn try_load_local_prolate_zip(
        lambda_sq: u64, n_grid: usize, prec: u32,
    ) -> Option<Vec<Float>> {
        let zip_path = prolate_cache_zip_path(lambda_sq, n_grid, prec)?;
        if !zip_path.exists() { return None; }
        let json_filename = prolate_cache_filename(lambda_sq, n_grid, prec);
        match load_prolate_eigvals_from_zip(&zip_path, &json_filename, n_grid, prec) {
            Some((evals, _json_string)) => {
                if let Some(reason) = prolate_cache_structural_check(
                    &evals, n_grid, lambda_sq, prec,
                ) {
                    warn_prolate_cache_skip(&zip_path, &reason);
                    None
                } else {
                    Some(evals)
                }
            }
            None => {
                warn_prolate_cache_skip(
                    &zip_path, "zip open / decompress / shape parse failed",
                );
                None
            }
        }
    }

    /// Deterministic remote URL for the `(λ², n_grid, prec)` fixture in
    /// the public xcelerator-prolate-eigvals-cache repo (precision-first
    /// → λ² → ngrid-thousand-bucket layout, mirroring tau/weil-eigvec).
    fn prolate_remote_zip_url(lambda_sq: u64, n_grid: usize, prec: u32) -> String {
        let bucket = (n_grid / 1000) * 1000;
        format!(
            "{base}/prolate_eigvals_cache/prec{p}/lambda_sq{l}/ngrid{b}-{bend}/{stem}.zip",
            base = PROLATE_REMOTE_BASE, p = prec, l = lambda_sq,
            b = bucket, bend = bucket + 999,
            stem = prolate_cache_filename(lambda_sq, n_grid, prec)
        )
    }

    /// Test-only accessor for `prolate_remote_zip_url`.
    #[cfg(test)]
    pub(super) fn prolate_remote_zip_url_for_test(
        lambda_sq: u64, n_grid: usize, prec: u32,
    ) -> String {
        prolate_remote_zip_url(lambda_sq, n_grid, prec)
    }

    /// Outcome of a single `curl` download attempt, classified by the
    /// actual HTTP status (mirrors the GL/τ/ξ fetch logic).
    enum ProlateCurlOutcome { Ok, HttpError, Transient }

    fn prolate_curl_attempt(url: &str, dest: &std::path::Path) -> ProlateCurlOutcome {
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
                        Ok(()) => ProlateCurlOutcome::Ok,
                        Err(_) => { let _ = std::fs::remove_file(&tmp); ProlateCurlOutcome::Transient }
                    },
                    404 => { let _ = std::fs::remove_file(&tmp); ProlateCurlOutcome::HttpError }
                    _ => { let _ = std::fs::remove_file(&tmp); ProlateCurlOutcome::Transient }
                }
            }
            _ => { let _ = std::fs::remove_file(&tmp); ProlateCurlOutcome::Transient }
        }
    }

    /// Download the `(λ², n_grid, prec)` `.json.zip` from the public
    /// cache repo into the local cache dir. Returns `true` if a file was
    /// written. Robust to `raw.githubusercontent.com` rate-limiting:
    /// only a 404 is a definitive miss; 429/5xx/no-response retry with
    /// backoff. Single zip only (no `.partXX` — the spectrum is small).
    fn fetch_remote_prolate_zip(lambda_sq: u64, n_grid: usize, prec: u32) -> bool {
        let dest = match prolate_cache_zip_path(lambda_sq, n_grid, prec) {
            Some(p) => p,
            None => return false,
        };
        let url = prolate_remote_zip_url(lambda_sq, n_grid, prec);

        const MAX_TRIES: usize = 5;
        for attempt in 0..MAX_TRIES {
            match prolate_curl_attempt(&url, &dest) {
                ProlateCurlOutcome::Ok => {
                    // Routine cache hit — silent.
                    return true;
                }
                ProlateCurlOutcome::HttpError => return false, // 404 — definitive miss.
                ProlateCurlOutcome::Transient => {
                    if attempt + 1 < MAX_TRIES {
                        let secs = 2 * (attempt as u64 + 1);
                        std::thread::sleep(std::time::Duration::from_secs(secs));
                    }
                }
            }
        }
        false
    }

    fn save_prolate_eigvals_cache(
        lambda_sq: u64, n_grid: usize, prec: u32, evals: &[Float], mode: CacheMode,
    ) {
        // Off and JsonOnly write nothing: the cache is zip-only.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) { return; }

        let strs: Vec<String> = evals.iter().map(|f| f.to_string()).collect();
        let json = serde_json::Value::Array(
            strs.into_iter().map(serde_json::Value::String).collect()
        );
        let json_str = match serde_json::to_string(&json) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Write ONLY a single `.json.zip`. Readers decompress on demand —
        // no uncompressed `.json` is persisted. The spectrum is small, so
        // this is always a single zip (no byte-split tier — unlike τ).
        let entry_name = prolate_cache_filename(lambda_sq, n_grid, prec);
        let zip_path = match prolate_cache_zip_path(lambda_sq, n_grid, prec) {
            Some(p) => p,
            None => return,
        };
        let mut buf: Vec<u8> = Vec::with_capacity(json_str.len() / 2);
        {
            use std::io::Write;
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = zip::ZipWriter::new(cursor);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            if writer.start_file(&entry_name, opts).is_err() { return; }
            if writer.write_all(json_str.as_bytes()).is_err() { return; }
            if writer.finish().is_err() { return; }
        }
        if let Err(e) = std::fs::write(&zip_path, &buf) {
            eprintln!(
                "[prolate_cache] WARNING: could not write {}: {}",
                zip_path.display(), e
            );
        }
    }

    /// Per-file outcome from `verify_prolate_eigvals_cache_dir`.
    ///
    /// Each variant carries the file path and (when parseable from the
    /// filename) the cache key tuple `(lambda_sq, n_grid, prec)`.
    /// `Skipped` is emitted for files whose name doesn't match the
    /// expected `lambda_sq{L}_ngrid{N}_prec{P}.json[.zip]` pattern.
    #[derive(Debug, Clone)]
    pub enum ProlateCacheFileStatus {
        /// File parsed and passed all structural identity checks.
        Ok { path: std::path::PathBuf, n_grid: usize, prec: u32, lambda_sq: u64 },
        /// File skipped because its name didn't match the expected
        /// `lambda_sq{L}_ngrid{N}_prec{P}.json[.zip]` pattern.
        Skipped { path: std::path::PathBuf, reason: String },
        /// Filename matched but the file failed to load (parse JSON,
        /// decompress zip, etc.).
        LoadFailed { path: std::path::PathBuf, n_grid: usize, prec: u32, lambda_sq: u64, reason: String },
        /// File loaded but the eigenvalue vector failed at least one
        /// of the prolate-spectrum structural identities (count, sort
        /// order, ground-state magnitude ≈ 2π·λ², per-element
        /// finiteness).
        StructurallyInvalid { path: std::path::PathBuf, n_grid: usize, prec: u32, lambda_sq: u64, reason: String },
    }

    /// Aggregate report from `verify_prolate_eigvals_cache_dir`.
    #[derive(Debug, Clone)]
    pub struct ProlateCacheVerifyReport {
        /// Directory that was scanned.
        pub directory: std::path::PathBuf,
        /// One status entry per file in `directory`.
        pub statuses: Vec<ProlateCacheFileStatus>,
    }

    impl ProlateCacheVerifyReport {
        /// Count of files that passed all checks.
        pub fn ok_count(&self) -> usize {
            self.statuses.iter().filter(|s| matches!(s, ProlateCacheFileStatus::Ok { .. })).count()
        }
        /// Count of files that failed at least one check (load or
        /// structural). Skipped files are not counted as failures.
        pub fn failure_count(&self) -> usize {
            self.statuses.iter().filter(|s| matches!(s,
                ProlateCacheFileStatus::LoadFailed { .. }
                | ProlateCacheFileStatus::StructurallyInvalid { .. }
            )).count()
        }
        /// All failure entries (load + structural), for callers that
        /// want to print only the bad files.
        pub fn failures(&self) -> impl Iterator<Item = &ProlateCacheFileStatus> {
            self.statuses.iter().filter(|s| matches!(s,
                ProlateCacheFileStatus::LoadFailed { .. }
                | ProlateCacheFileStatus::StructurallyInvalid { .. }
            ))
        }
    }

    fn parse_prolate_cache_filename(name: &str) -> Option<(u64, usize, u32)> {
        // `lambda_sq{LSQ}_ngrid{N}_prec{P}.json[.zip]`
        let stem = name
            .strip_suffix(".json.zip")
            .or_else(|| name.strip_suffix(".json"))?;
        let after_lsq = stem.strip_prefix("lambda_sq")?;
        let mut parts = after_lsq.splitn(2, "_ngrid");
        let lsq_str = parts.next()?;
        let rest = parts.next()?;
        let mut parts2 = rest.splitn(2, "_prec");
        let n_str = parts2.next()?;
        let prec_str = parts2.next()?;
        let lambda_sq: u64 = lsq_str.parse().ok()?;
        let n_grid: usize = n_str.parse().ok()?;
        let prec: u32 = prec_str.parse().ok()?;
        Some((lambda_sq, n_grid, prec))
    }

    /// Walk the prolate eigenvalue cache directory and structurally
    /// verify every `lambda_sq{L}_ngrid{N}_prec{P}.json[.zip]` file.
    /// Returns a per-file status report; does not mutate any files.
    pub fn verify_prolate_eigvals_cache_dir(
        dir: &std::path::Path,
    ) -> std::io::Result<ProlateCacheVerifyReport> {
        let mut statuses: Vec<ProlateCacheFileStatus> = Vec::new();

        if !dir.exists() {
            return Ok(ProlateCacheVerifyReport {
                directory: dir.to_path_buf(),
                statuses,
            });
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() { continue; }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };

            let (lambda_sq, n_grid, prec) = match parse_prolate_cache_filename(name) {
                Some(t) => t,
                None => {
                    statuses.push(ProlateCacheFileStatus::Skipped {
                        path: path.clone(),
                        reason: format!(
                            "filename '{}' not in expected lambda_sq{{L}}_ngrid{{N}}_prec{{P}}.json[.zip] form",
                            name
                        ),
                    });
                    continue;
                }
            };

            let parsed: Option<Vec<Float>> = if name.ends_with(".json.zip") {
                let json_filename = prolate_cache_filename(lambda_sq, n_grid, prec);
                load_prolate_eigvals_from_zip(&path, &json_filename, n_grid, prec)
                    .map(|(p, _)| p)
            } else {
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|data| parse_prolate_cache_json(&data, n_grid, prec))
            };

            let evals = match parsed {
                Some(e) => e,
                None => {
                    statuses.push(ProlateCacheFileStatus::LoadFailed {
                        path: path.clone(),
                        lambda_sq, n_grid, prec,
                        reason: "parse / decompress failed".to_string(),
                    });
                    continue;
                }
            };

            match prolate_cache_structural_check(&evals, n_grid, lambda_sq, prec) {
                None => {
                    statuses.push(ProlateCacheFileStatus::Ok {
                        path, lambda_sq, n_grid, prec
                    });
                }
                Some(reason) => {
                    statuses.push(ProlateCacheFileStatus::StructurallyInvalid {
                        path, lambda_sq, n_grid, prec, reason
                    });
                }
            }
        }

        Ok(ProlateCacheVerifyReport {
            directory: dir.to_path_buf(),
            statuses,
        })
    }

    /// HP result of `compute_k_lambda`. Stores the sample grid and k_λ
    /// values in HP.
    #[derive(Debug, Clone)]
    pub struct HpProlateResult {
        /// k_λ sampled on `u_grid` (HP).
        pub k_values: Vec<Float>,
        /// Sample points u_i ∈ [λ⁻¹, λ], logarithmically spaced (HP).
        pub u_grid: Vec<Float>,
        /// Eigenvalue of h_{0,λ} (≈ 2π λ²) in HP.
        pub eigenvalue_0: Float,
        /// Eigenvalue of h_{4,λ} (≈ 18π λ²) in HP.
        pub eigenvalue_4: Float,
        /// Coefficient of h_{4,λ} in h_λ (typically 1).
        pub c_4: Float,
        /// Coefficient of h_{0,λ} in h_λ (typically -r where r = ∫h_4 / ∫h_0).
        pub c_0: Float,
        /// Wall-clock seconds.
        pub elapsed_seconds: f64,
        /// Working precision used.
        pub precision_bits: u32,
    }

    /// HP version of `compute_k_lambda_f64`.
    ///
    /// Pipeline:
    ///   1. Build PW_λ tridiagonal at HP.
    ///   2. Compute all eigenvalues in HP via `tridiag_eigenvalues_hp`.
    ///   3. Recover lowest-lying eigenvectors via shifted inverse iteration
    ///      and identify h_0 (smallest, even, 0 nodes) and h_4 (even, 4 nodes).
    ///   4. Form h_λ = h_4 - r·h_0 with r = ∫h_4 / ∫h_0 in HP.
    ///   5. Sample k_λ on a logarithmic grid in [λ⁻¹, λ] in HP.
    ///
    /// `n_grid` is the number of FD interior points (forced odd).
    /// `n_sample` is the number of comparison-grid points.
    ///
    /// `mode` selects the prolate-eigenvalue cache strategy (see
    /// [`xc_numerics::quadrature::CacheMode`]): `Off` always recomputes
    /// the spectrum, `JsonOnly`/`JsonZip` consult the local cache,
    /// `DynamicFetch` (the recommended default) also pulls a missing
    /// spectrum from the public `xcelerator-prolate-eigvals-cache` repo.
    /// Pass `CacheMode::default()` for the standard behavior.
    pub fn compute_k_lambda(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
        mode: CacheMode,
    ) -> Result<HpProlateResult> {
        let start = std::time::Instant::now();

        // Grid forced odd.
        let n = if n_grid % 2 == 0 { n_grid + 1 } else { n_grid };
        if n < 16 {
            anyhow::bail!("n_grid too small (got {}); need at least 16 to find h_4", n);
        }

        eprintln!("[HP prolate] computing k_λ at λ²={:.6}, N={}, n_sample={}, prec={} bits",
            {
                let mut sq = lambda.clone();
                sq *= lambda;
                sq.to_f64()
            }, n, n_sample, prec);

        // Build the tridiagonal in HP.
        eprintln!("[HP prolate] building tridiagonal PW_λ on N={} grid...", n);
        let pw_start = std::time::Instant::now();
        let (diag, off_diag) = build_pw_matrix(lambda, n, prec);
        eprintln!("[HP prolate] PW_λ built in {:.1}s", pw_start.elapsed().as_secs_f64());

        // Compute h = 2λ / (N+1) for later use (continuous-L² scaling, integration).
        let mut h = lambda.clone();
        h *= 2u32;
        let n_plus_1 = Float::with_val(prec, (n + 1) as u32);
        h /= &n_plus_1;

        // Get all eigenvalues sorted ascending. Try the prolate
        // eigenvalue cache first: at HP-1000 with N=8001 the
        // tridiagonal QR is ~30 minutes; if we've computed this exact
        // (λ², n_grid, prec) before, the cache turns that into a
        // ~5-second JSON read. Cache key derives from λ²_int — only
        // active when λ² is integer-valued (our paper configs are
        // 13/100/1000); non-integer λ² silently bypasses the cache.
        let mut lambda_sq_for_key = lambda.clone();
        lambda_sq_for_key *= lambda;
        let cache_key = lambda_sq_int_for_key(&lambda_sq_for_key);

        let eigenvalues: Vec<Float> = if let Some(lambda_sq_int) = cache_key {
            if let Some(cached) = load_prolate_eigvals_cache(lambda_sq_int, n, prec, mode) {
                eprintln!(
                    "[HP prolate] loaded {} cached eigenvalues for λ²={}, N={}, prec={} bits",
                    cached.len(), lambda_sq_int, n, prec
                );
                cached
            } else {
                eprintln!("[HP prolate] computing all {} eigenvalues of PW_λ via tridiag QR...", n);
                let eig_start = std::time::Instant::now();
                let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec)?;
                eprintln!("[HP prolate] {} eigenvalues computed in {:.1}s",
                    evals.len(), eig_start.elapsed().as_secs_f64());
                save_prolate_eigvals_cache(lambda_sq_int, n, prec, &evals, mode);
                evals
            }
        } else {
            eprintln!("[HP prolate] computing all {} eigenvalues of PW_λ via tridiag QR (cache disabled: λ² not integer)...", n);
            let eig_start = std::time::Instant::now();
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec)?;
            eprintln!("[HP prolate] {} eigenvalues computed in {:.1}s",
                evals.len(), eig_start.elapsed().as_secs_f64());
            evals
        };
        if eigenvalues.len() != n {
            anyhow::bail!("eigensolver returned {} eigenvalues, expected {}",
                eigenvalues.len(), n);
        }

        // Search the lowest-lying eigenfunctions for h_0 and h_4.
        // Limit search depth: prolate h_4 is the third even eigenfunction.
        let n_try = super::PROLATE_SEARCH_DEPTH.min(n);
        eprintln!("[HP prolate] searching for h_0 (even, 0 nodes) and h_4 (even, 4 nodes) in first {} eigenvectors...", n_try);
        let search_start = std::time::Instant::now();
        let mut h0_idx: Option<usize> = None;
        let mut h4_idx: Option<usize> = None;
        let mut h0_vec: Option<Vec<Float>> = None;
        let mut h4_vec: Option<Vec<Float>> = None;

        for k in 0..n_try {
            let lambda_k = &eigenvalues[k];
            eprintln!("[HP prolate] eigenvector {}/{} (eigenvalue {})...",
                k + 1, n_try, xc_numerics::fmt::display_hp(lambda_k, 8));
            // Opt into the production defaults: banded LU (O(n) factor
            // and O(n) per-step solve via tridiag_lu_factor_hp + Thomas
            // forward/back substitution), early termination on the
            // |⟨v_k, v_{k-1}⟩| convergence proxy, 200-step ceiling.
            //
            // Prolate eigenvectors are well-conditioned (the spectrum
            // is widely-spaced and non-degenerate at small k) so the
            // iteration typically converges in 20-50 steps. Capping at
            // 200 with no convergence check was wasting hours per run
            // at HP-1000.
            //
            // Banded LU drops the per-eigenvector wall-time from hours
            // to seconds at HP-1000 with N=8001, and the memory
            // footprint from ~26 GB to a few KB, vs the dense LU
            // alternative.
            let v = match tridiag_eigenvector_for_value_hp(
                &diag, &off_diag, lambda_k, prec, TridiagEigvecOptions::default(),
            ) {
                Ok(v) => v,
                Err(_) => continue,
            };
            // Check parity.
            let parity = parity_of(&v, prec);
            let is_even = matches!(parity, HpParity::Even);
            if !is_even {
                continue;
            }
            // Count nodes.
            let nodes = count_nodes(&v, prec);
            match nodes {
                0 if h0_idx.is_none() => {
                    eprintln!("[HP prolate] found h_0 at index {}", k);
                    h0_idx = Some(k);
                    h0_vec = Some(v);
                }
                4 if h4_idx.is_none() => {
                    eprintln!("[HP prolate] found h_4 at index {}", k);
                    h4_idx = Some(k);
                    h4_vec = Some(v);
                }
                _ => {}
            }
            if h0_idx.is_some() && h4_idx.is_some() {
                break;
            }
        }
        eprintln!("[HP prolate] eigenvector search done in {:.1}s",
            search_start.elapsed().as_secs_f64());
        let h0_idx = h0_idx.ok_or_else(|| {
            anyhow::anyhow!("could not find h_{{0,λ}} (even, 0 nodes) in first {} eigenfunctions", n_try)
        })?;
        let h4_idx = h4_idx.ok_or_else(|| {
            anyhow::anyhow!("could not find h_{{4,λ}} (even, 4 nodes) in first {} eigenfunctions", n_try)
        })?;
        let mut h0_vec = h0_vec.unwrap();
        let mut h4_vec = h4_vec.unwrap();
        let eigenvalue_0 = eigenvalues[h0_idx].clone();
        let eigenvalue_4 = eigenvalues[h4_idx].clone();

        // Inverse iteration normalizes to unit ℓ² norm. To get unit
        // continuous-L² norm, scale by 1/√h.
        let h_sqrt = h.clone().sqrt();
        let mut inv_sqrt_h = Float::with_val(prec, 1);
        inv_sqrt_h /= &h_sqrt;
        for v in h0_vec.iter_mut() {
            *v *= &inv_sqrt_h;
        }
        for v in h4_vec.iter_mut() {
            *v *= &inv_sqrt_h;
        }

        // Pin sign: positive at center.
        let center = n / 2;
        let zero = Float::with_val(prec, 0);
        if h0_vec[center] < zero {
            for v in h0_vec.iter_mut() {
                *v = -v.clone();
            }
        }
        if h4_vec[center] < zero {
            for v in h4_vec.iter_mut() {
                *v = -v.clone();
            }
        }

        // ∫h via midpoint rule = h · Σ v_i.
        let mut sum_h0 = Float::with_val(prec, 0);
        for v in &h0_vec {
            sum_h0 += v;
        }
        let mut int_h0 = sum_h0.clone();
        int_h0 *= &h;

        let mut sum_h4 = Float::with_val(prec, 0);
        for v in &h4_vec {
            sum_h4 += v;
        }
        let mut int_h4 = sum_h4.clone();
        int_h4 *= &h;

        let int_zero_thresh = Float::with_val(prec, Float::parse("1e-30").unwrap());
        if int_h0.clone().abs() < int_zero_thresh {
            anyhow::bail!("∫h_{{0,λ}} ≈ 0; cannot enforce ∫h_λ = 0");
        }

        // r = ∫h_4 / ∫h_0
        let mut r = int_h4.clone();
        r /= &int_h0;
        let c_4 = Float::with_val(prec, 1);
        let mut c_0 = r.clone();
        c_0 = -c_0;

        // h_λ = c_4 · h_4 + c_0 · h_0
        let h_lambda: Vec<Float> = (0..n).map(|i| {
            let mut t = c_4.clone();
            t *= &h4_vec[i];
            let mut t2 = c_0.clone();
            t2 *= &h0_vec[i];
            t += &t2;
            t
        }).collect();

        // Build logarithmic u-grid in [λ⁻¹, λ].
        let n_sample = n_sample.max(2);
        let log_lambda = lambda.clone().ln();
        let u_grid: Vec<Float> = (0..n_sample).map(|i| {
            // t = i / (n_sample - 1), in [0, 1]
            let mut t = Float::with_val(prec, i as u32);
            let denom = Float::with_val(prec, (n_sample - 1) as u32);
            t /= &denom;
            // arg = log_lambda · (2t - 1)
            let mut arg = t.clone();
            arg *= 2u32;
            arg -= 1u32;
            arg *= &log_lambda;
            // u = exp(arg)
            arg.exp()
        }).collect();

        // Evaluate k_λ(u) = √u · Σ_{n=1}^{⌊λ/u⌋} h_λ(n·u).
        // Each grid point u is independent → parallelize via par_iter.
        eprintln!("[HP prolate] sampling k_λ on {} log-spaced grid points...", n_sample);
        let sample_start = std::time::Instant::now();
        let k_values: Vec<Float> = u_grid.par_iter().map(|u| {
            let mut ratio = lambda.clone();
            ratio /= u;
            let ratio_floor = ratio.floor();
            let n_terms_int = ratio_floor.to_integer().expect("λ/u must be finite");
            let n_terms = n_terms_int.to_u64().unwrap_or(0);

            let mut s = Float::with_val(prec, 0);
            for k in 1..=n_terms {
                let mut x = u.clone();
                let k_hp = Float::with_val(prec, k);
                x *= &k_hp;
                if x >= *lambda {
                    break;
                }
                s += &interp_grid(&h_lambda, lambda, &h, &x, prec);
            }
            // u^(1/2)
            let sqrt_u = u.clone().sqrt();
            let mut result = sqrt_u;
            result *= &s;
            result
        }).collect();
        eprintln!("[HP prolate] k_λ sampling done in {:.1}s; total compute_k_lambda elapsed {:.1}s",
            sample_start.elapsed().as_secs_f64(), start.elapsed().as_secs_f64());

        Ok(HpProlateResult {
            k_values,
            u_grid,
            eigenvalue_0,
            eigenvalue_4,
            c_4,
            c_0,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            precision_bits: prec,
        })
    }

    /// HP comparison result. All fields HP except `linf_index` (a usize).
    #[derive(Debug, Clone)]
    pub struct HpComparisonResult {
        /// Best fit scalar `c` such that `c·k_λ ≈ ξ_λ` on the sampling grid (HP).
        pub optimal_scalar: Float,
        /// L∞ norm of the residual `ξ_λ − c·k_λ` on the grid (HP).
        pub linf_error: Float,
        /// Discrete ℓ² norm of the residual (no quadrature weights), HP.
        pub l2_error: Float,
        /// L∞ norm of `ξ_λ` on the grid, used for relative-error reporting (HP).
        pub xi_linf: Float,
        /// Index of the maximum residual on the sampling grid.
        pub linf_index: usize,
    }

    /// HP comparison ‖ξ_λ − c·k_λ‖ on the prolate sample grid.
    /// `xi` has length `2N+1` and contains HP coefficients (caller is
    /// responsible for HP precision matching the rest of the pipeline).
    pub fn compare_xi_to_k_lambda(
        xi: &[Float],
        n_modes: usize,
        lambda: &Float,
        u_grid: &[Float],
        k_values: &[Float],
        prec: u32,
    ) -> Result<HpComparisonResult> {
        if xi.len() != 2 * n_modes + 1 {
            anyhow::bail!("xi has wrong length: got {}, expected 2N+1 = {}",
                xi.len(), 2 * n_modes + 1);
        }
        if u_grid.len() != k_values.len() {
            anyhow::bail!("u_grid and k_values have different lengths: {} vs {}",
                u_grid.len(), k_values.len());
        }
        if u_grid.is_empty() {
            anyhow::bail!("empty grid");
        }

        let cmp_start = std::time::Instant::now();
        eprintln!("[HP prolate] comparing ξ_λ to c·k_λ on {} grid points (N={} modes)...",
            u_grid.len(), n_modes);

        // L = ln(λ²) = 2 ln λ; inv_sqrt_l = 1/√L.
        let mut lambda_sq = lambda.clone();
        lambda_sq *= lambda;
        let l = lambda_sq.clone().ln();
        let l_sqrt = l.clone().sqrt();
        let mut inv_sqrt_l = Float::with_val(prec, 1);
        inv_sqrt_l /= &l_sqrt;

        let xi_0 = xi[n_modes].clone();
        let xi_pos: Vec<Float> = (1..=n_modes).map(|n| xi[n_modes + n].clone()).collect();

        // Reconstruct ξ_λ on u_grid.
        let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
        let mut two_pi = pi_v.clone();
        two_pi *= 2u32;

        // Reconstruct ξ_λ on u_grid. Each grid point is independent →
        // parallel evaluation across u_grid.
        let xi_values: Vec<Float> = u_grid.par_iter().map(|u| {
            // phase_base = 2π · ln(λ·u) / L
            let mut lambda_u = lambda.clone();
            lambda_u *= u;
            let log_lu = lambda_u.ln();
            let mut phase_base = two_pi.clone();
            phase_base *= &log_lu;
            phase_base /= &l;

            let mut acc = xi_0.clone();
            for n in 1..=n_modes {
                let mut arg = phase_base.clone();
                arg *= n as u32;
                let mut term = arg.cos();
                term *= 2u32;
                term *= &xi_pos[n - 1];
                acc += &term;
            }
            acc *= &inv_sqrt_l;
            acc
        }).collect();

        // Optimal c = ⟨ξ, k⟩ / ⟨k, k⟩. Parallel reductions.
        let dot_xk: Float = xi_values.par_iter().zip(k_values.par_iter())
            .map(|(x, k)| {
                let mut t = x.clone();
                t *= k;
                t
            })
            .reduce(|| Float::with_val(prec, 0), |mut a, b| { a += &b; a });
        let dot_kk: Float = k_values.par_iter()
            .map(|k| {
                let mut t = k.clone();
                t *= k;
                t
            })
            .reduce(|| Float::with_val(prec, 0), |mut a, b| { a += &b; a });

        let kk_zero_thresh = Float::with_val(prec, Float::parse("1e-300").unwrap());
        if dot_kk < kk_zero_thresh {
            anyhow::bail!("k_values essentially zero on grid");
        }
        let mut c = dot_xk;
        c /= &dot_kk;

        // Residual norms.
        let mut linf_error = Float::with_val(prec, 0);
        let mut linf_index = 0usize;
        let mut l2_sq = Float::with_val(prec, 0);
        for (i, (x, k)) in xi_values.iter().zip(k_values.iter()).enumerate() {
            // r = x - c · k
            let mut r = c.clone();
            r *= k;
            r = -r;
            r += x;
            let r_sq = {
                let mut t = r.clone();
                t *= &r;
                t
            };
            l2_sq += &r_sq;
            let abs_r = r.abs();
            if abs_r > linf_error {
                linf_error = abs_r;
                linf_index = i;
            }
        }
        let l2_error = l2_sq.sqrt();

        let mut xi_linf = Float::with_val(prec, 0);
        for x in &xi_values {
            let abs_x = x.clone().abs();
            if abs_x > xi_linf {
                xi_linf = abs_x;
            }
        }

        eprintln!("[HP prolate] compare done in {:.1}s",
            cmp_start.elapsed().as_secs_f64());
        Ok(HpComparisonResult {
            optimal_scalar: c,
            linf_error,
            l2_error,
            xi_linf,
            linf_index,
        })
    }

    // -----------------------------------------------------------------------
    // HP unit tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;
        use xc_numerics::fmt::display_hp;

        fn hp(prec: u32, s: &str) -> Float {
            Float::with_val(prec, Float::parse(s).unwrap())
        }

        /// Build PW_λ in HP and confirm the diagonal is sane (positive,
        /// finite, bounded).
        #[test]
        fn pw_matrix_ground_state_hp() {
            let prec = 256;
            let lambda = hp(prec, "2");
            let n = 199;
            let (diag, _off_diag) = build_pw_matrix(&lambda, n, prec);
            assert_eq!(diag.len(), n);
            let center = diag.len() / 2;
            // At x=0, diagonal should be (2λ²)/h² + 0 (potential is 0).
            // h = 2λ/(N+1) = 4/200 = 0.02. h² = 4e-4. 2λ²/h² = 8/4e-4 = 20000.
            // So diag[center] ≈ 20000.
            let lo = hp(prec, "1000");
            let hi = hp(prec, "100000");
            assert!(diag[center] > lo && diag[center] < hi,
                "diag[center] = {} should be in [1000, 100000]",
                display_hp(&diag[center], 6));
        }

        /// Smallest prolate eigenvalue at λ=5 should be close to 2π·25 = 157.08.
        #[test]
        fn pw_smallest_eigenvalue_hp() {
            let prec = 256;
            let lambda = hp(prec, "5");
            let n = 401;
            let (diag, off_diag) = build_pw_matrix(&lambda, n, prec);
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
            // Smallest = evals[0].
            // Expected ≈ 2π·25 ≈ 157.08.
            let expected = {
                let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
                let mut t = pi_v;
                t *= 2u32;
                t *= 25u32;
                t
            };
            // Allow 20% tolerance due to FD truncation error at N=401.
            let mut diff = evals[0].clone();
            diff -= &expected;
            let abs_diff = diff.abs();
            // tol = expected · 0.2 = expected / 5 (HP-only construction).
            let mut tol = expected.clone();
            tol /= 5u32;
            assert!(abs_diff < tol,
                "smallest eigenvalue {} should be near 2π·25 = {} (diff {})",
                display_hp(&evals[0], 6),
                display_hp(&expected, 6),
                display_hp(&abs_diff, 4));
        }

        /// End-to-end compute_k_lambda at HP. Verify result is finite,
        /// has expected shape, and eigenvalues are sane.
        #[test]
        fn compute_k_lambda_runs_hp() {
            let prec = 256;
            let lambda = hp(prec, "5");
            let res = compute_k_lambda(&lambda, 401, 64, prec, CacheMode::Off).unwrap();
            assert_eq!(res.u_grid.len(), 64);
            assert_eq!(res.k_values.len(), 64);

            // h_0 has eigenvalue ≈ 2πλ² ≈ 157, h_4 ≈ 18πλ² ≈ 1413.
            let zero = Float::with_val(prec, 0);
            assert!(res.eigenvalue_0 > zero,
                "eigenvalue_0 should be positive, got {}", display_hp(&res.eigenvalue_0, 6));
            assert!(res.eigenvalue_4 > res.eigenvalue_0,
                "eigenvalue_4 ({}) should exceed eigenvalue_0 ({})",
                display_hp(&res.eigenvalue_4, 6),
                display_hp(&res.eigenvalue_0, 6));

            // At least one k value should be nonzero.
            let mut any_nonzero = false;
            let zero_thresh = hp(prec, "1e-30");
            for k in &res.k_values {
                if k.clone().abs() > zero_thresh {
                    any_nonzero = true;
                    break;
                }
            }
            assert!(any_nonzero, "k_λ should be nonzero somewhere");
        }

        /// End-to-end compare_xi_to_k_lambda at HP with a synthetic ξ.
        #[test]
        fn compare_runs_end_to_end_hp() {
            let prec = 256;
            let lambda = hp(prec, "5");
            let res = compute_k_lambda(&lambda, 401, 64, prec, CacheMode::Off).unwrap();

            // Build synthetic ξ: only ξ_0 nonzero, equal to √L.
            let n_modes = 20;
            let mut lambda_sq = lambda.clone();
            lambda_sq *= &lambda;
            let l = lambda_sq.ln();
            let l_sqrt = l.sqrt();

            let mut xi: Vec<Float> = (0..(2 * n_modes + 1))
                .map(|_| Float::with_val(prec, 0))
                .collect();
            xi[n_modes] = l_sqrt;

            let cmp = compare_xi_to_k_lambda(&xi, n_modes, &lambda, &res.u_grid, &res.k_values, prec).unwrap();
            // All HP results should be finite (no NaN, no infinity).
            assert!(!cmp.linf_error.is_nan() && !cmp.linf_error.is_infinite());
            assert!(!cmp.l2_error.is_nan() && !cmp.l2_error.is_infinite());
            let zero = Float::with_val(prec, 0);
            assert!(cmp.xi_linf > zero,
                "xi_linf should be positive, got {}", display_hp(&cmp.xi_linf, 6));
        }

        // ---------------------------------------------------------------
        // Prolate eigenvalue cache — pure-function and verify_dir tests
        // ---------------------------------------------------------------

        /// `lambda_sq_int_for_key` accepts integer-valued λ² (within
        /// 1e-10 tolerance) and rejects non-integer values.
        #[test]
        fn cache_key_accepts_integer_lambda_sq() {
            let prec = 256;
            // Integer-valued λ² → Some(L)
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "13")), Some(13));
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "100")), Some(100));
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "1000")), Some(1000));
            // Tiny f64 round-trip noise should still parse — but we're
            // building these from exact strings so they're already tight.
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "13.0")), Some(13));
            // Non-integer rejected.
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "13.5")), None);
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "100.001")), None);
            // Negative or zero rejected.
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "0")), None);
            let mut neg = hp(prec, "13"); neg = -neg;
            assert_eq!(lambda_sq_int_for_key(&neg), None);
        }

        /// `parse_prolate_cache_filename` parses well-formed names and
        /// rejects others.
        #[test]
        fn cache_filename_parser_extracts_tuple() {
            assert_eq!(
                parse_prolate_cache_filename("lambda_sq13_ngrid4001_prec3338.json"),
                Some((13, 4001, 3338))
            );
            assert_eq!(
                parse_prolate_cache_filename("lambda_sq1000_ngrid8001_prec3338.json.zip"),
                Some((1000, 8001, 3338))
            );
            // Wrong shape → None.
            assert_eq!(parse_prolate_cache_filename("foo.json"), None);
            assert_eq!(parse_prolate_cache_filename("lambda_sq13.json"), None);
            assert_eq!(
                parse_prolate_cache_filename("lambda_sq_ngrid_prec.json"),
                None
            );
        }

        /// `prolate_cache_structural_check` accepts a real prolate
        /// spectrum and rejects perturbed versions.
        #[test]
        fn cache_structural_check_accepts_real_prolate_spectrum() {
            let prec = 256;
            let lambda = hp(prec, "5");
            let lambda_sq: u64 = 25;
            let n = 401;
            let (diag, off_diag) = build_pw_matrix(&lambda, n, prec);
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
            // Real spectrum must pass.
            assert!(prolate_cache_structural_check(&evals, n, lambda_sq, prec).is_none(),
                "real prolate spectrum should pass structural check");

            // Wrong count → reject.
            let mut short = evals.clone();
            short.pop();
            assert!(prolate_cache_structural_check(&short, n, lambda_sq, prec).is_some(),
                "wrong count should be rejected");

            // Out-of-order → reject.
            let mut shuffled = evals.clone();
            shuffled.swap(0, 5);
            assert!(prolate_cache_structural_check(&shuffled, n, lambda_sq, prec).is_some(),
                "non-ascending order should be rejected");

            // Wrong ground state magnitude → reject. Replace e_0 with
            // a value 50% off from 2π·25.
            let mut off_e0 = evals.clone();
            off_e0[0] = hp(prec, "50");  // way below 2π·25 ≈ 157
            assert!(prolate_cache_structural_check(&off_e0, n, lambda_sq, prec).is_some(),
                "ground-state magnitude check should reject 50 vs 2π·25 ≈ 157");

            // NaN entry → reject.
            let mut with_nan = evals.clone();
            with_nan[10] = Float::with_val(prec, f64::NAN);
            assert!(prolate_cache_structural_check(&with_nan, n, lambda_sq, prec).is_some(),
                "NaN entry should be rejected");
        }

        /// `verify_prolate_eigvals_cache_dir` on a non-existent directory
        /// returns an empty report, not an error.
        #[test]
        fn verify_dir_handles_missing_directory() {
            let temp_root = crate::test_tmp_root()
                .join(format!("xc_spectral_prolate_cache_test_missing_{}",
                              std::process::id()));
            let nonexistent = temp_root.join("does_not_exist");
            let report = verify_prolate_eigvals_cache_dir(&nonexistent).unwrap();
            assert_eq!(report.statuses.len(), 0);
            assert_eq!(report.ok_count(), 0);
            assert_eq!(report.failure_count(), 0);
        }

        /// `verify_prolate_eigvals_cache_dir` classifies files by kind:
        /// Ok, Skipped (unrecognized name), LoadFailed (malformed JSON),
        /// StructurallyInvalid (parses but fails identity check).
        #[test]
        fn verify_dir_classifies_files() {
            let prec = 256;
            let lambda = hp(prec, "5");
            let lambda_sq: u64 = 25;
            let n_grid: usize = 401;

            // Compute a real spectrum once.
            let (diag, off_diag) = build_pw_matrix(&lambda, n_grid, prec);
            let real_evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

            // Build an isolated temp dir under target/test-tmp.
            let temp_dir = crate::fresh_test_dir("prolate_cache_classify");

            // 1. Valid file: serialize the real spectrum.
            let valid_name = prolate_cache_filename(lambda_sq, n_grid, prec);
            let valid_path = temp_dir.join(&valid_name);
            let strs: Vec<String> = real_evals.iter().map(|f| f.to_string()).collect();
            let valid_json = serde_json::Value::Array(
                strs.into_iter().map(serde_json::Value::String).collect()
            );
            std::fs::write(&valid_path, serde_json::to_string(&valid_json).unwrap()).unwrap();

            // 2. Structurally-invalid file: reverse the spectrum so it's
            //    not ascending.
            let mut bad_evals = real_evals.clone();
            bad_evals.reverse();
            let bad_name = prolate_cache_filename(lambda_sq + 1, n_grid, prec);
            let bad_path = temp_dir.join(&bad_name);
            let bad_strs: Vec<String> = bad_evals.iter().map(|f| f.to_string()).collect();
            let bad_json = serde_json::Value::Array(
                bad_strs.into_iter().map(serde_json::Value::String).collect()
            );
            std::fs::write(&bad_path, serde_json::to_string(&bad_json).unwrap()).unwrap();

            // 3. Skipped file: unrecognized name.
            let skipped_path = temp_dir.join("not_a_prolate_cache.txt");
            std::fs::write(&skipped_path, "irrelevant").unwrap();

            // 4. LoadFailed: matching name pattern, malformed JSON.
            let malformed_name = prolate_cache_filename(lambda_sq + 2, n_grid, prec);
            let malformed_path = temp_dir.join(&malformed_name);
            std::fs::write(&malformed_path, "{").unwrap();

            let report = verify_prolate_eigvals_cache_dir(&temp_dir).unwrap();
            assert_eq!(report.statuses.len(), 4,
                "expected 4 statuses, got {}", report.statuses.len());

            let mut saw_ok = false;
            let mut saw_invalid = false;
            let mut saw_skipped = false;
            let mut saw_loadfail = false;
            for s in &report.statuses {
                match s {
                    ProlateCacheFileStatus::Ok { path, lambda_sq: l, n_grid: ng, prec: p } => {
                        assert_eq!(path, &valid_path);
                        assert_eq!(*l, lambda_sq);
                        assert_eq!(*ng, n_grid);
                        assert_eq!(*p, prec);
                        saw_ok = true;
                    }
                    ProlateCacheFileStatus::StructurallyInvalid { path, .. } => {
                        assert_eq!(path, &bad_path);
                        saw_invalid = true;
                    }
                    ProlateCacheFileStatus::Skipped { path, .. } => {
                        assert_eq!(path, &skipped_path);
                        saw_skipped = true;
                    }
                    ProlateCacheFileStatus::LoadFailed { path, .. } => {
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
            assert_eq!(report.failure_count(), 2,
                "LoadFailed + StructurallyInvalid both count; expected 2");

            // Files preserved (verify is read-only).
            assert!(valid_path.exists());
            assert!(bad_path.exists());
            assert!(skipped_path.exists());
            assert!(malformed_path.exists());

            // Cleanup.
            let _ = std::fs::remove_dir_all(&temp_dir);
        }

        // -------------------------------------------------------------
        // CacheMode / remote-fetch tests
        // -------------------------------------------------------------

        use std::sync::Mutex;

        /// Serialize cwd-mutating cache tests (cwd is process-global).
        /// Aliases the crate-level [`crate::TEST_CWD_LOCK`] so these
        /// tests serialize against the `ccm::hp` cache tests too (both
        /// run in the same xc-spectral test binary and share one
        /// process-global cwd). Mirrors the GL / ccm::hp guards.
        #[allow(dead_code)]
        static PROLATE_CWD_LOCK: &Mutex<()> = &crate::TEST_CWD_LOCK;

        struct ProlateCwdGuard {
            original: std::path::PathBuf,
            _lock: std::sync::MutexGuard<'static, ()>,
        }
        impl ProlateCwdGuard {
            fn enter(temp: &std::path::Path) -> Self {
                let lock = PROLATE_CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                let original = std::env::current_dir().expect("no cwd");
                std::env::set_current_dir(temp).expect("set_current_dir to temp");
                ProlateCwdGuard { original, _lock: lock }
            }
        }
        impl Drop for ProlateCwdGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.original);
            }
        }

        fn prolate_temp_cwd(tag: &str) -> std::path::PathBuf {
            crate::fresh_test_dir(&format!("prolate_{}", tag))
        }

        /// The remote prolate URL is deterministically derived from
        /// `(λ², ngrid, prec)` using the public repo's precision-first →
        /// λ² → ngrid-thousand-bucket layout.
        #[test]
        fn prolate_remote_url_uses_bucketed_layout() {
            // λ²=100, ngrid=4001, prec=3338 → bucket 4000-4999.
            let url = prolate_remote_zip_url_for_test(100, 4001, 3338);
            assert_eq!(
                url,
                "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-prolate-eigvals-cache/main/prolate_eigvals_cache/prec3338/lambda_sq100/ngrid4000-4999/lambda_sq100_ngrid4001_prec3338.json.zip"
            );

            // λ²=1000, ngrid=8001, prec=3338 → bucket 8000-8999.
            let url2 = prolate_remote_zip_url_for_test(1000, 8001, 3338);
            assert_eq!(
                url2,
                "https://raw.githubusercontent.com/TeamXcelerator/xcelerator-prolate-eigvals-cache/main/prolate_eigvals_cache/prec3338/lambda_sq1000/ngrid8000-8999/lambda_sq1000_ngrid8001_prec3338.json.zip"
            );
        }

        /// `save` then `load` round-trips eigenvalues at every CacheMode
        /// tier, and `CacheMode::Off` writes nothing.
        #[test]
        fn prolate_cache_save_load_round_trip() {
            let prec = 256;
            let lambda_sq = 25u64;
            let n_grid = 401usize;

            let temp = prolate_temp_cwd("round_trip");
            let _guard = ProlateCwdGuard::enter(&temp);

            // Build a real, validation-passing spectrum.
            let lambda = hp(prec, "5");
            let (diag, off_diag) = build_pw_matrix(&lambda, n_grid, prec);
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

            // Off: writes nothing, reads nothing.
            save_prolate_eigvals_cache(lambda_sq, n_grid, prec, &evals, CacheMode::Off);
            assert!(load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::Off).is_none(),
                "Off should never read");
            assert!(load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "Off save should have written nothing");

            // JsonZip: writes ONLY the .json.zip (zip-only contract);
            // reads back identical by decompressing in memory.
            save_prolate_eigvals_cache(lambda_sq, n_grid, prec, &evals, CacheMode::JsonZip);

            // No uncompressed .json should be written.
            let jp = temp.join("data").join("prolate_eigvals_cache")
                .join(prolate_cache_filename(lambda_sq, n_grid, prec));
            assert!(!jp.exists(), "zip-only: save must not write an uncompressed .json");

            let got = load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip)
                .expect("JsonZip round-trip should load from the zip");
            assert_eq!(got.len(), evals.len());
            for (a, b) in evals.iter().zip(got.iter()) {
                assert_eq!(a.to_string(), b.to_string(), "eigenvalue must round-trip exactly");
            }

            // JsonOnly is now a read no-op (no uncompressed .json exists).
            assert!(load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonOnly).is_none(),
                "zip-only: JsonOnly must not read the zip");

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }

        /// Negative: a structurally-invalid `.json` (descending order)
        /// must be skipped by `load` (returns None → recompute). Bad file
        /// preserved.
        #[test]
        fn prolate_load_skips_structurally_invalid_json() {
            let prec = 256;
            let lambda_sq = 25u64;
            let n_grid = 401usize;

            let temp = prolate_temp_cwd("invalid_json");
            let _guard = ProlateCwdGuard::enter(&temp);

            // Real spectrum reversed → not ascending → fails the check.
            let lambda = hp(prec, "5");
            let (diag, off_diag) = build_pw_matrix(&lambda, n_grid, prec);
            let mut bad = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
            bad.reverse();

            let dir = temp.join("data").join("prolate_eigvals_cache");
            std::fs::create_dir_all(&dir).unwrap();
            let entry_name = prolate_cache_filename(lambda_sq, n_grid, prec);
            let zip_path = dir.join(format!("{}.zip", entry_name));
            let strs: Vec<String> = bad.iter().map(|f| f.to_string()).collect();
            let json = serde_json::Value::Array(
                strs.into_iter().map(serde_json::Value::String).collect());
            let json_str = serde_json::to_string(&json).unwrap();

            // Plant the bad spectrum inside a .json.zip (only tier read now).
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

            assert!(load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "descending (non-ascending) spectrum in the zip must be skipped");
            assert!(zip_path.exists(),
                "structurally-invalid zip should be preserved for inspection");

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }

        /// Negative: a corrupt `.json.zip` must be detected and skipped
        /// without panic (`load` returns None). Corrupt file preserved.
        #[test]
        fn prolate_load_handles_corrupt_zip_gracefully() {
            let prec = 64;
            let lambda_sq = 25u64;
            let n_grid = 401usize;

            let temp = prolate_temp_cwd("corrupt_zip");
            let _guard = ProlateCwdGuard::enter(&temp);

            let dir = temp.join("data").join("prolate_eigvals_cache");
            std::fs::create_dir_all(&dir).unwrap();
            // Garbage bytes named as the zip; no local .json, so JsonZip
            // falls through to the zip, fails to open it, returns None.
            let zip_path = dir.join(format!(
                "lambda_sq{}_ngrid{}_prec{}.json.zip", lambda_sq, n_grid, prec));
            std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

            assert!(load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "corrupt .json.zip must be skipped, not loaded");
            assert!(zip_path.exists(),
                "corrupt zip should be preserved for inspection");

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }

        /// Live end-to-end remote prolate-fetch test against the PUBLIC
        /// `xcelerator-prolate-eigvals-cache` repo. `#[ignore]`d so it
        /// never runs in the default suite (needs network + curl + a
        /// populated repo). Run with:
        ///
        /// ```text
        /// cargo test -p xc-spectral --features hp -- --ignored prolate_remote_fetch_live
        /// ```
        #[test]
        #[ignore = "live network: hits the public xcelerator-prolate-eigvals-cache repo; run with --ignored"]
        fn prolate_remote_fetch_live_downloads_and_validates() {
            // A config that exists in the public repo (the seed fixture
            // generated by examples/gen_prolate_eigvals_fixture: λ²=25,
            // n_grid=401, HP-64 → prec 229 bits).
            let lambda_sq = 25u64;
            let n_grid = 401usize;
            let prec = 229u32;

            let temp = prolate_temp_cwd("remote_live");
            let _guard = ProlateCwdGuard::enter(&temp);

            assert!(load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "no local cache should exist before fetch");

            let fetched = load_prolate_eigvals_cache(
                lambda_sq, n_grid, prec, CacheMode::DynamicFetch);

            let evals = fetched.expect("remote fetch should have returned a spectrum");
            assert_eq!(evals.len(), n_grid,
                "fetched spectrum length {} != ngrid {}", evals.len(), n_grid);

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }
    }
}
