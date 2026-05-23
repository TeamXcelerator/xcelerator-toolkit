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
    use xc_numerics::eigen::{tridiag_eigenvalues_hp, tridiag_eigenvector_for_value_hp};

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
        Even,
        Odd,
        Indeterminate,
    }

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
    pub fn compute_k_lambda(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
    ) -> Result<HpProlateResult> {
        let start = std::time::Instant::now();

        // Grid forced odd.
        let n = if n_grid % 2 == 0 { n_grid + 1 } else { n_grid };
        if n < 16 {
            anyhow::bail!("n_grid too small (got {}); need at least 16 to find h_4", n);
        }

        // Build the tridiagonal in HP.
        let (diag, off_diag) = build_pw_matrix(lambda, n, prec);

        // Compute h = 2λ / (N+1) for later use (continuous-L² scaling, integration).
        let mut h = lambda.clone();
        h *= 2u32;
        let n_plus_1 = Float::with_val(prec, (n + 1) as u32);
        h /= &n_plus_1;

        // Get all eigenvalues sorted ascending.
        let eigenvalues = tridiag_eigenvalues_hp(&diag, &off_diag, prec)?;
        if eigenvalues.len() != n {
            anyhow::bail!("eigensolver returned {} eigenvalues, expected {}",
                eigenvalues.len(), n);
        }

        // Search the lowest-lying eigenfunctions for h_0 and h_4.
        // Limit search depth: prolate h_4 is the third even eigenfunction.
        let n_try = super::PROLATE_SEARCH_DEPTH.min(n);
        let mut h0_idx: Option<usize> = None;
        let mut h4_idx: Option<usize> = None;
        let mut h0_vec: Option<Vec<Float>> = None;
        let mut h4_vec: Option<Vec<Float>> = None;

        for k in 0..n_try {
            let lambda_k = &eigenvalues[k];
            let v = match tridiag_eigenvector_for_value_hp(&diag, &off_diag, lambda_k, prec, 200) {
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
                    h0_idx = Some(k);
                    h0_vec = Some(v);
                }
                4 if h4_idx.is_none() => {
                    h4_idx = Some(k);
                    h4_vec = Some(v);
                }
                _ => {}
            }
            if h0_idx.is_some() && h4_idx.is_some() {
                break;
            }
        }
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
        pub optimal_scalar: Float,
        pub linf_error: Float,
        pub l2_error: Float,
        pub xi_linf: Float,
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
            let res = compute_k_lambda(&lambda, 401, 64, prec).unwrap();
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
            let res = compute_k_lambda(&lambda, 401, 64, prec).unwrap();

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
    }
}
