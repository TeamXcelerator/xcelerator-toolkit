// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.
//

//! Bounded-endpoint prolate approximation and sampled prolate/Weil comparison.
//!
//! Implements the prolate-wave educated guess `k_λ` from Section 7 of
//! the CCM construction. The educated guess approximates
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
//! Public usage examples and assurance boundaries are documented in
//! `docs/RESEARCH_WORKFLOWS.md`.
//!
//! ## Scope: computed discretization (HP version available below)
//!
//! Implemented:
//! - Bounded-endpoint Legendre Galerkin approximation for ordinary entry points
//! - Explicit historical finite-difference PW_λ matrix construction
//! - Dense symmetric eigendecomposition via nalgebra
//! - Even Legendre indices 0 and 2 select full modes 0 and 4; historical FD uses
//!   node-counting and parity detection to identify h_{0,λ} and h_{4,λ}
//! - Linear combination h_λ = c_4·h_{4,λ} + c_0·h_{0,λ} with ∫h_λ = 0
//! - ℰ map evaluation on a logarithmic grid in [λ⁻¹, λ]
//! - Comparison ‖ξ_λ − c·k_λ‖_∞, ‖ξ_λ − c·k_λ‖_2 against the Weil
//!   eigenvector reconstructed from its V_n Fourier coefficients
//! - High-precision (rug) version (`prolate::hp` submodule) with
//!   dynamic working precision subject to explicit memory and work budgets.

mod comparison;
mod grid_contract;
mod legendre;
mod stencil;

use anyhow::Result;
use nalgebra::{DMatrix, SymmetricEigen};

pub fn prolate_artifact_reuse_plan() -> xc_core::ArtifactReusePlan {
    use xc_core::{ArtifactReuseNode, ArtifactReusePlan};
    let node = |kind: &str, dependencies: &[&str], invalidated_by: &[&str]| ArtifactReuseNode {
        kind: kind.to_owned(),
        independently_cacheable: true,
        dependencies: dependencies
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        invalidated_by: invalidated_by
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    };
    ArtifactReusePlan {
        schema_version: 1,
        domain: "prolate".to_owned(),
        semantics_version: "prolate-bounded-legendre-candidate-v6".to_owned(),
        artifacts: vec![
            node("basis", &[], &["lambda_squared", "basis", "truncation"]),
            node(
                "operator_components",
                &["basis"],
                &["operator_semantics", "precision_bits", "quadrature_rule"],
            ),
            node(
                "eigensystem",
                &["operator_components"],
                &["target", "solver_semantics", "normalization"],
            ),
            node(
                "reference_candidate",
                &["eigensystem"],
                &["candidate_semantics", "sampling_grid"],
            ),
            node(
                "weil_comparison",
                &["reference_candidate"],
                &["weil_state_digest", "form_digest", "truncation_bounds"],
            ),
            node(
                "deficiency_certificate",
                &["eigensystem"],
                &["certificate_policy", "enclosure_width"],
            ),
        ],
    }
}

/// Legacy compatibility constant; parity and node-count classification now
/// use relative scaling and do not apply this absolute cutoff.
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
    /// Maximum number of even Legendre coefficients for the ordinary prolate
    /// candidate. The explicit finite-Dirichlet APIs interpret this same legacy
    /// configuration field as the number of interior grid points.
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
    /// next odd integer for compatibility with the explicit finite-Dirichlet
    /// routes, where this places `x = 0` on the spatial grid. The ordinary
    /// route interprets this value as a maximum Legendre coefficient count.
    // Keep remainder arithmetic for the Rust 1.85 MSRV.
    #[allow(unknown_lints, clippy::manual_is_multiple_of)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProlateResult {
    /// Discretization and endpoint realization used by this result.
    #[serde(default = "legacy_fd_discretization")]
    pub discretization: String,
    #[serde(default)]
    pub resolution_budget: usize,
    #[serde(default)]
    pub basis_dimension: usize,
    /// Computed infinite-Legendre-operator residual divided by 1+|eigenvalue|,
    /// maximized over h0 and h4. This is not a certified continuum error bound.
    #[serde(default)]
    pub relative_operator_residual: Option<f64>,
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

fn legacy_fd_discretization() -> String {
    "centered_finite_difference_dirichlet_v1".into()
}

/// Build the prolate wave operator PW_λ matrix at f64 precision via
/// 3-point finite differences on a uniform grid `[-λ, λ]` with N+2
/// points (boundary nodes at ±λ where u=0 by Dirichlet, so the matrix
/// is N × N for the interior nodes).
///
/// The matrix is symmetric tridiagonal:
///   - Diagonal: `((λ²-x_{i-1/2}²)+(λ²-x_{i+1/2}²))/h² + (2πλ x_i)²`,
///     equivalently `(2/h²)·(λ²-x_i²) - 1/2 + (2πλ x_i)²`.
///   - Off-diagonal: `-(1/h²)·(λ² − x_{i±1/2}²)`
///
/// where `h = 2λ/(N+1)` is the grid spacing and `x_i = -λ + i·h` for
/// `i = 1, …, N`.
///
/// Returns the diagonal and the lower off-diagonal as separate `Vec<f64>`.
pub fn try_build_pw_matrix_f64(cfg: &ProlateConfig) -> Result<(Vec<f64>, Vec<f64>)> {
    stencil::build_f64(cfg)
}

/// Compatibility wrapper; invalid input panics. Use the checked variant.
pub fn build_pw_matrix_f64(cfg: &ProlateConfig) -> (Vec<f64>, Vec<f64>) {
    try_build_pw_matrix_f64(cfg).expect("valid representable native prolate grid")
}

/// Construct PW_λ as a dense `DMatrix<f64>` from its tridiagonal data.
fn build_pw_dense_f64(cfg: &ProlateConfig) -> Result<DMatrix<f64>> {
    let (diag, off) = try_build_pw_matrix_f64(cfg)?;
    let n = diag.len();
    let mut matrix = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = diag[i];
        if i > 0 {
            matrix[(i, i - 1)] = off[i - 1];
            matrix[(i - 1, i)] = off[i - 1];
        }
    }
    Ok(matrix)
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
    if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
        return Parity::Indeterminate;
    }
    let scale = v.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if scale == 0.0 {
        return Parity::Indeterminate;
    }
    let mut total = 0.0;
    let mut even = 0.0;
    let mut odd = 0.0;
    for (a, b) in v.iter().zip(v.iter().rev()) {
        let a = a / scale;
        let b = b / scale;
        total += a.abs();
        even += (a - b).abs();
        odd += (a + b).abs();
    }
    if even / total < PARITY_CLASSIFICATION_TOL {
        Parity::Even
    } else if odd / total < PARITY_CLASSIFICATION_TOL {
        Parity::Odd
    } else {
        Parity::Indeterminate
    }
}

/// Count zero crossings of `v`. Small entries (< threshold of the
/// max absolute value) are skipped to avoid spurious counts from
/// numerical noise near boundary.
fn count_nodes_f64(v: &[f64]) -> usize {
    grid_contract::nodes_f64(v)
}

/// Linearly interpolate a function defined on the FD grid
/// `x_i = -λ + (i+1)h` (i = 0..n-1) at an arbitrary point `x ∈ [-λ, λ]`.
/// Returns 0 outside the support (Dirichlet BC).
fn interp_grid_f64(values: &[f64], lambda: f64, h: f64, x: f64) -> f64 {
    if x.abs() >= lambda {
        return 0.0;
    }
    // Measure from the nearer Dirichlet endpoint. Adding lambda to a point
    // just below +lambda can round to 2*lambda and erase a nonzero value.
    let from_right = x > 0.0;
    let distance = if from_right { lambda - x } else { lambda + x };
    let position = distance / h;
    let lower = position.floor() as usize;
    let fraction = position - lower as f64;
    let sample = |index: usize| {
        if index == 0 || index > values.len() {
            0.0
        } else if from_right {
            values[values.len() - index]
        } else {
            values[index - 1]
        }
    };
    (1.0 - fraction) * sample(lower) + fraction * sample(lower + 1)
}

/// Compute the prolate-wave educated guess k_λ.
///
/// 1. Build PW_λ at f64 via finite differences on a uniform grid.
/// 2. Diagonalize and identify h_{0,λ} (smallest, even, 0 nodes) and
///    h_{4,λ} (even, 4 nodes) by node-counting + parity.
/// 3. Form h_λ = h_{4,λ} − r·h_{0,λ} with r = ∫h_{4,λ} / ∫h_{0,λ},
///    so that ∫h_λ dx = 0 (the constraint of Lemma 7.1 in the publication).
/// 4. Sample k_λ(u) = √u · Σ_{n=1}^{⌊λ/u⌋} h_λ(n·u) on a logarithmic
///    grid `u_i ∈ [λ⁻¹, λ]`. The grid is logarithmic to align with
///    the V_n Fourier basis of the Weil form.
pub fn compute_k_lambda_finite_dirichlet_f64(cfg: &ProlateConfig) -> Result<ProlateResult> {
    anyhow::ensure!(
        cfg.lambda.is_finite() && cfg.lambda > 1.0 && cfg.lambda * cfg.lambda < u64::MAX as f64,
        "prolate candidate needs lambda > 1 and a representable finite-sum cutoff"
    );
    anyhow::ensure!(
        cfg.n_sample >= 2 && cfg.n_sample <= u32::MAX as usize,
        "prolate candidate needs at least two samples"
    );

    let start = std::time::Instant::now();
    let lambda = cfg.lambda;
    let n = cfg.n_grid;
    if n < 16 {
        anyhow::bail!("n_grid too small (got {}); need at least 16 to find h_4", n);
    }
    // Admit all simultaneously retained dense storage before any allocation.
    legendre::resource_budget(n, cfg.n_sample, (lambda * lambda).ceil() as usize, 53, true)?;
    let m = build_pw_dense_f64(cfg)?;
    let h = 2.0 * lambda / ((n + 1) as f64);

    // Diagonalize the finite matrix, then pair each eigenvalue with its own
    // column; the library QR can attach tiny eigenvalues to other columns.
    let mut eig = SymmetricEigen::try_new(m.clone(), f64::EPSILON, n.saturating_mul(128))
        .ok_or_else(|| anyhow::anyhow!("prolate eigensolver did not converge"))?;

    anyhow::ensure!(
        eig.eigenvalues
            .iter()
            .chain(eig.eigenvectors.iter())
            .all(|x| x.is_finite()),
        "prolate eigensolver returned nonfinite arithmetic"
    );
    let values = xc_numerics::symmetric_f64::complete_symmetric_eigensystem_f64(
        m.as_slice(),
        n,
        eig.eigenvectors.as_mut_slice(),
    )?;
    eig.eigenvalues = nalgebra::DVector::from_vec(values);

    // Sort indices by ascending eigenvalue.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));

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
    // For this irreducible symmetric tridiagonal matrix with negative
    // off-diagonals, the ordered modes have 0 and 4 sign changes. The
    // approximate parity/node heuristic must never silently relabel a mode.
    anyhow::ensure!(
        h0_idx == idx[0] && h4_idx == idx[4],
        "prolate parity/node classification disagrees with ordered modes 0 and 4"
    );
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

    // The trapezoidal integral of the piecewise-linear interpolant is
    // exactly h * sum(v_i), since both endpoint samples are zero.
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

    anyhow::ensure!(
        k_values.iter().all(|x| x.is_finite()) && c_0.is_finite(),
        "prolate sampling produced nonfinite arithmetic"
    );
    Ok(ProlateResult {
        discretization: legacy_fd_discretization(),
        resolution_budget: cfg.n_grid,
        basis_dimension: n,
        relative_operator_residual: None,
        k_values,
        u_grid,
        eigenvalue_0,
        eigenvalue_4,
        c_4,
        c_0,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

/// Compute bounded-endpoint prolate modes by even Legendre Galerkin projection.
/// `n_grid` is a maximum coefficient budget, not a spatial grid. Failure to
/// resolve the omitted coupling returns an error. Sampling evaluates the
/// polynomial modes directly; the Eisenstein sum retains open support.
pub fn compute_k_lambda_f64(cfg: &ProlateConfig) -> Result<ProlateResult> {
    legendre::compute_f64(cfg)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComparisonResult {
    /// Best fit scalar c such that c·k_λ ≈ ξ_λ on the sampling grid.
    pub optimal_scalar: f64,
    /// L∞ norm of the residual ξ_λ − c·k_λ on the grid.
    pub linf_error: f64,
    /// Discrete ℓ² norm of the residual (no quadrature weights).
    pub l2_error: f64,
    /// L∞ norm of ξ_λ on the grid (for relative-error reporting).
    pub xi_linf: f64,
    /// Discrete ℓ² norm of ξ_λ on the grid, matching `l2_error`'s
    /// unweighted convention.
    ///
    /// Because `optimal_scalar` minimizes `‖ξ_λ − c·k_λ‖₂`, the residual is
    /// orthogonal to `k_λ` and `l2_error / xi_l2` is exactly `sin θ`, the
    /// scale-free angle between the educated guess and its target. Reported
    /// so that relative distance is recoverable from the result alone.
    pub xi_l2: f64,
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
    comparison::compare_f64(xi, n_modes, lambda, u_grid, k_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_prolate_results_round_trip_without_loss() {
        let result = ProlateResult {
            discretization: legacy_fd_discretization(),
            resolution_budget: 17,
            basis_dimension: 17,
            relative_operator_residual: None,
            k_values: vec![0.25, 0.5],
            u_grid: vec![0.5, 2.0],
            eigenvalue_0: 3.25,
            eigenvalue_4: 7.5,
            c_4: 1.0,
            c_0: -0.125,
            elapsed_seconds: 0.75,
        };
        let comparison = ComparisonResult {
            optimal_scalar: 1.25,
            linf_error: 1.0e-14,
            l2_error: 2.0e-14,
            xi_linf: 0.75,
            xi_l2: 0.875,
            linf_index: 1,
        };
        let result_json = serde_json::to_vec(&result).unwrap();
        let comparison_json = serde_json::to_vec(&comparison).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProlateResult>(&result_json).unwrap(),
            result
        );
        assert_eq!(
            serde_json::from_slice::<ComparisonResult>(&comparison_json).unwrap(),
            comparison
        );
    }

    /// Optimal scaling makes the residual orthogonal to `k_λ`, so
    /// `‖ξ‖² = ‖c·k‖² + ‖ξ − c·k‖²` and `l2_error / xi_l2` is exactly `sin θ`.
    ///
    /// This is what makes the reported distance scale-free and therefore
    /// comparable across `λ²`.
    #[test]
    fn optimal_scaling_makes_relative_l2_distance_a_sine() {
        let lambda = 5.0;
        let cfg = ProlateConfig::new(lambda, 401).with_n_sample(64);
        let res = compute_k_lambda_f64(&cfg).unwrap();

        // A ξ with several active modes, so the residual is not degenerate.
        let n_modes = 20;
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = (lambda * lambda).ln().sqrt();
        for k in 1..=4 {
            let v = 0.35_f64 / (k as f64);
            xi[n_modes + k] = v;
            xi[n_modes - k] = v;
        }
        let cmp = compare_xi_to_k_lambda_f64(&xi, n_modes, lambda, &res.u_grid, &res.k_values)
            .expect("comparison should succeed");

        assert!(cmp.xi_l2 > 0.0, "ξ must be nonzero on the grid");
        let k_norm: f64 = res.k_values.iter().map(|k| k * k).sum::<f64>().sqrt();
        let projected = cmp.optimal_scalar * k_norm;
        let pythagoras = projected * projected + cmp.l2_error * cmp.l2_error;
        assert!(
            (pythagoras - cmp.xi_l2 * cmp.xi_l2).abs() <= 1e-12 * cmp.xi_l2 * cmp.xi_l2,
            "residual is not orthogonal to k: {pythagoras} vs {}",
            cmp.xi_l2 * cmp.xi_l2
        );

        let sine = cmp.l2_error / cmp.xi_l2;
        assert!(
            (0.0..=1.0).contains(&sine),
            "relative distance must be a sine, got {sine}"
        );
        // L∞ of the residual cannot exceed the ℓ² norm on the same grid.
        assert!(cmp.linf_error <= cmp.l2_error * (1.0 + 1e-12));
    }

    #[test]
    fn prolate_reuse_plan_separates_candidate_and_certificate() {
        let plan = prolate_artifact_reuse_plan();
        plan.validate().unwrap();
        assert!(plan
            .artifacts
            .iter()
            .any(|node| node.kind == "reference_candidate"));
        assert!(plan
            .artifacts
            .iter()
            .any(|node| node.kind == "deficiency_certificate"));
    }

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
        let m = build_pw_dense_f64(&cfg).unwrap();
        let eig = SymmetricEigen::new(m);
        let mut evals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        evals.sort_by(|a, b| a.total_cmp(b));
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
        let max_k = res.k_values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
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
// The ordinary route uses the bounded even Legendre block at guarded precision;
// the explicitly named finite-Dirichlet route retains the historical FD model.
// ===========================================================================

#[cfg(feature = "hp")]
pub mod hp {
    use anyhow::Result;
    use rayon::prelude::*;
    use rug::{ops::NegAssign, Float};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use xc_cache::{
        resolve_or_compute_json_artifact, ArtifactCacheContext, ArtifactExecutionCacheRequest,
        CacheError, CacheQuality, SemanticKeyEnvelope, ToolkitVersion,
    };
    use xc_numerics::eigen::{tridiag_eigenvalues_hp, TridiagEigvecOptions};
    use xc_numerics::quadrature::CacheMode;

    use super::super::ccm::LambdaSq;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PortableProlateSpectrum {
        schema_version: u32,
        lambda: String,
        lambda_precision_bits: u32,
        grid_points: usize,
        precision_bits: u32,
        eigenvalues: Vec<String>,
    }

    enum ProlateCacheRoute<'a> {
        Standalone(CacheMode),
        Fabric(&'a ArtifactCacheContext<'a>),
    }

    fn decode_prolate_spectrum(
        artifact: &PortableProlateSpectrum,
        lambda: &Float,
        n_grid: usize,
        prec: u32,
    ) -> std::result::Result<Vec<Float>, CacheError> {
        if artifact.schema_version != 3
            || artifact.lambda != lambda.to_string()
            || artifact.lambda_precision_bits != lambda.prec()
            || artifact.grid_points != n_grid
            || artifact.precision_bits != prec
            || artifact.eigenvalues.len() != n_grid
        {
            return Err(CacheError::InvalidManifest(
                "prolate spectrum payload does not match its semantic identity".to_owned(),
            ));
        }
        let mut values = Vec::with_capacity(n_grid);
        for value in &artifact.eigenvalues {
            let parsed = Float::parse(value).map_err(|error| {
                CacheError::InvalidManifest(format!(
                    "prolate spectrum contains an invalid HP scalar: {error}"
                ))
            })?;
            let value = Float::with_val(prec, parsed);
            if value.is_nan() || value.is_infinite() {
                return Err(CacheError::InvalidManifest(
                    "prolate spectrum contains a non-finite eigenvalue".to_owned(),
                ));
            }
            if values.last().is_some_and(|previous| previous > &value) {
                return Err(CacheError::InvalidManifest(
                    "prolate spectrum eigenvalues are not sorted".to_owned(),
                ));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn prolate_spectrum_via_cache(
        lambda: &Float,
        n_grid: usize,
        prec: u32,
        diag: &[Float],
        off_diag: &[Float],
        cache: &ArtifactCacheContext<'_>,
    ) -> std::result::Result<Vec<Float>, CacheError> {
        prolate_spectrum_via_cache_model(lambda, n_grid, prec, diag, off_diag, cache, false)
    }

    // The finite-Dirichlet consumer selects these two ordered source values.
    // Validate inside cache admission so corruption becomes a miss, not a sticky
    // post-load failure. This is finite-matrix index replay, not continuum accuracy.
    fn validate_fd_selected_spectrum(
        d: &[Float],
        b: &[Float],
        values: &[Float],
        p: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            d.len() >= 5 && values.len() == d.len() && b.len() + 1 == d.len(),
            "finite-Dirichlet spectrum shape mismatch"
        );
        let work = p + 64;
        let norm = (0..d.len())
            .map(|i| {
                let mut row = Float::with_val(work, &d[i]).abs();
                if i > 0 {
                    row += Float::with_val(work, &b[i - 1]).abs();
                }
                if i < b.len() {
                    row += Float::with_val(work, &b[i]).abs();
                }
                row
            })
            .max_by(Float::total_cmp)
            .ok_or_else(|| anyhow::anyhow!("empty prolate matrix"))?;
        let radius = (norm * (8 * d.len())) >> p;
        for index in [0, 4] {
            anyhow::ensure!(
                values[index].is_finite(),
                "nonfinite finite-Dirichlet eigenvalue"
            );
            let lo = Float::with_val(work, &values[index] - &radius);
            let hi = Float::with_val(work, &values[index] + &radius);
            anyhow::ensure!(
                xc_numerics::eigen::tridiag_sturm_count_below_hp(d, b, &lo, work)? == index
                    && xc_numerics::eigen::tridiag_sturm_count_below_hp(d, b, &hi, work)?
                        == index + 1,
                "finite-Dirichlet eigenvalue does not match its source-matrix index"
            );
        }
        Ok(())
    }

    #[cfg(test)]
    mod admission_tests {
        use super::*;

        #[test]
        fn cached_indices_use_stored_precision_not_half_precision() {
            for p in [96u32, 128, 192, 257] {
                for step in [1u32, 7] {
                    let d: Vec<_> = (0..7).map(|i| Float::with_val(p, 2 + step * i)).collect();
                    let b = vec![Float::with_val(p, 0); 6];
                    validate_fd_selected_spectrum(&d, &b, &d, p).unwrap();
                    super::super::legendre::hp::validate_spectrum(&d, &b, &d, p).unwrap();
                    for index in [0, 4] {
                        for exponent in [p / 2 + 4, p - 20] {
                            let mut wrong = d.clone();
                            wrong[index] += Float::with_val(p, 1) >> exponent;
                            assert!(validate_fd_selected_spectrum(&d, &b, &wrong, p).is_err());
                            assert!(super::super::legendre::hp::validate_spectrum(
                                &d, &b, &wrong, p
                            )
                            .is_err());
                        }
                    }
                }
            }
        }

        #[test]
        fn finite_dirichlet_resource_gate_precedes_eigensolution() {
            let p = 256;
            let lambda = Float::with_val(p, 2);
            let started = std::time::Instant::now();
            let error = compute_k_lambda_finite_dirichlet(&lambda, 100001, 2, p, CacheMode::Off)
                .unwrap_err();
            assert!(error.to_string().contains("work budget"), "{error}");
            assert!(started.elapsed() < std::time::Duration::from_secs(2));
            // This documented setting is admissible under both memory and work policy.
            super::super::legendre::resource_budget(512, 64, 13, 3386, false).unwrap();
            assert!(super::super::legendre::resource_budget(100001, 2, 4, 3386, false).is_err());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prolate_spectrum_via_cache_model(
        lambda: &Float,
        n_grid: usize,
        prec: u32,
        diag: &[Float],
        off_diag: &[Float],
        cache: &ArtifactCacheContext<'_>,
        legendre: bool,
    ) -> std::result::Result<Vec<Float>, CacheError> {
        let model = if legendre {
            "prolate-bounded-legendre-even-spectrum-v1"
        } else {
            "prolate-fd-working-precision-spectrum-v0.15.1-v2"
        };
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "prolate_eigenvalue_spectrum".to_owned(),
            mathematical_semantics_version: model.to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "lambda": lambda.to_string(),
                "lambda_precision_bits": lambda.prec(),
                "grid_points": n_grid,
                "precision_bits": prec,
                "scalar_backend": "rug_mpfr",
                "discretization": if legendre { super::legendre::SPECTRUM_DISCRETIZATION } else { "centered_finite_difference_dirichlet_v1" }
            }),
            normalization: Some("ascending_eigenvalues".to_owned()),
            target: Some("prolate_wave_operator".to_owned()),
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(xc_numerics::eigen::TRIDIAG_QR_SEMANTICS.to_owned()),
        };
        let logical_key = format!(
            "prolate/exact-lambda/{}/{n_grid}/{prec}",
            semantic_key.digest()?.0
        );
        let request = ArtifactExecutionCacheRequest {
            operation: "prolate.spectrum.resolve_or_compute",
            semantic_key: &semantic_key,
            logical_key: &logical_key,
            resolver: cache.resolver,
            reference_resolver: cache.reference_resolver,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: cache.write_on_miss,
            write_visibility: cache.write_visibility,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
            minimum_reader_version: ToolkitVersion::parse(xc_cache::CLEAN_SLATE)?,
            maximum_reader_version: None,
            tags: BTreeMap::from([("domain".to_owned(), "prolate".to_owned())]),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };
        let resolved = resolve_or_compute_json_artifact(
            &request,
            || {
                let eigenvalues =
                    tridiag_eigenvalues_hp(diag, off_diag, prec).map_err(|error| {
                        CacheError::InvalidManifest(format!(
                            "prolate eigensolver failed while producing cache artifact: {error}"
                        ))
                    })?;
                Ok(PortableProlateSpectrum {
                    schema_version: 3,
                    lambda: lambda.to_string(),
                    lambda_precision_bits: lambda.prec(),
                    grid_points: n_grid,
                    precision_bits: prec,
                    eigenvalues: eigenvalues.iter().map(Float::to_string).collect(),
                })
            },
            |artifact| {
                let values = decode_prolate_spectrum(artifact, lambda, n_grid, prec)?;
                if legendre {
                    super::legendre::hp::validate_spectrum(diag, off_diag, &values, prec)
                        .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
                } else {
                    validate_fd_selected_spectrum(diag, off_diag, &values, prec)
                        .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
                }
                Ok(())
            },
        )?;
        decode_prolate_spectrum(&resolved.value, lambda, n_grid, prec)
    }

    // The standalone noninteger route uses the same exact-source payload as
    // the managed route. Never round lambda squared to reuse an integer key.
    fn exact_standalone_prolate_spectrum(
        lambda: &Float,
        n_grid: usize,
        prec: u32,
        diag: &[Float],
        off_diag: &[Float],
        cache_directory: Option<&std::path::Path>,
    ) -> Result<Vec<Float>> {
        exact_standalone_prolate_spectrum_model(
            lambda,
            n_grid,
            prec,
            diag,
            off_diag,
            cache_directory,
            false,
        )
    }

    fn exact_standalone_prolate_spectrum_model(
        lambda: &Float,
        n_grid: usize,
        prec: u32,
        diag: &[Float],
        off_diag: &[Float],
        cache_directory: Option<&std::path::Path>,
        legendre: bool,
    ) -> Result<Vec<Float>> {
        let model = if legendre {
            "prolate-bounded-legendre-even-spectrum-v1"
        } else {
            "prolate-fd-working-precision-spectrum-v0.15.1-v2"
        };
        let identity = xc_cache::ContentDigest::sha256(&serde_json::to_vec(&(
            model,
            xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
            lambda.to_string(),
            lambda.prec(),
            n_grid,
            prec,
        ))?);
        let entry_name = format!("exact_lambda_{}.json", identity.0);
        let path = cache_directory.map(|dir| dir.join(format!("{entry_name}.zip")));
        if let Some(path) = &path {
            if path.exists() {
                let read = || -> Result<Vec<Float>> {
                    use std::io::Read;
                    let mut zip = zip::ZipArchive::new(std::fs::File::open(path)?)?;
                    let limit = prolate_spectrum_json_limit(n_grid, prec.max(lambda.prec()))
                        .ok_or_else(|| anyhow::anyhow!("prolate cache size bound overflow"))?;
                    let entry = zip.by_name(&entry_name)?;
                    if entry.size() > limit {
                        anyhow::bail!("prolate cache exceeds decoded size bound");
                    }
                    let mut bytes = Vec::new();
                    entry.take(limit + 1).read_to_end(&mut bytes)?;
                    if bytes.len() as u64 > limit {
                        anyhow::bail!("prolate cache exceeds decoded size bound");
                    }
                    let artifact: PortableProlateSpectrum = serde_json::from_slice(&bytes)?;
                    let values = decode_prolate_spectrum(&artifact, lambda, n_grid, prec)?;
                    if legendre {
                        super::legendre::hp::validate_spectrum(diag, off_diag, &values, prec)?;
                    } else {
                        validate_fd_selected_spectrum(diag, off_diag, &values, prec)?;
                    }
                    Ok(values)
                };
                match read() {
                    Ok(values) => return Ok(values),
                    Err(error) => warn_prolate_cache_skip(path, &error.to_string()),
                }
            }
        }
        let eigenvalues = tridiag_eigenvalues_hp(diag, off_diag, prec)?;
        let artifact = PortableProlateSpectrum {
            schema_version: 3,
            lambda: lambda.to_string(),
            lambda_precision_bits: lambda.prec(),
            grid_points: n_grid,
            precision_bits: prec,
            eigenvalues: eigenvalues.iter().map(Float::to_string).collect(),
        };
        let values = decode_prolate_spectrum(&artifact, lambda, n_grid, prec)?;
        if legendre {
            super::legendre::hp::validate_spectrum(diag, off_diag, &values, prec)?;
        } else {
            validate_fd_selected_spectrum(diag, off_diag, &values, prec)?;
        }
        if let Some(path) = path {
            let save = || -> Result<()> {
                use std::io::Write;
                // A unique sibling prevents readers seeing a partially written zip.
                let temporary = path.with_extension(format!(
                    "zip.{}.{}.tmp",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_nanos()
                ));
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                let write = || -> Result<()> {
                    let mut writer = zip::ZipWriter::new(file);
                    writer.start_file(
                        &entry_name,
                        zip::write::SimpleFileOptions::default()
                            .compression_method(zip::CompressionMethod::Deflated)
                            .large_file(true),
                    )?;
                    writer.write_all(&serde_json::to_vec(&artifact)?)?;
                    writer.finish()?.sync_all()?;
                    std::fs::rename(&temporary, &path)?;
                    Ok(())
                };
                let result = write();
                if result.is_err() {
                    let _ = std::fs::remove_file(&temporary);
                }
                result
            };
            if let Err(error) = save() {
                warn_prolate_cache_skip(&path, &format!("write failed: {error}"));
            }
        }
        Ok(values)
    }

    /// Toolkit version string embedded in every prolate eigvals cache file
    /// written by this build.
    const PROLATE_TOOLKIT_VERSION: &str = env!("CARGO_PKG_VERSION");

    #[cfg(test)]
    pub(super) fn prolate_toolkit_version_for_test() -> &'static str {
        PROLATE_TOOLKIT_VERSION
    }

    /// Minimum toolkit version required to use a prolate eigvals cache file.
    /// Files produced by an older toolkit are treated as cache misses.
    fn prolate_effective_min_version() -> String {
        xc_cache::artifact_compatibility_policy("prolate", "prolate_eigenvalue_spectrum")
            .expect("prolate compatibility policy")
            .minimum_producer_version
            .to_string()
    }

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
    // Keep remainder arithmetic for the Rust 1.85 MSRV.
    #[allow(unknown_lints, clippy::manual_is_multiple_of)]
    pub fn try_build_pw_matrix(
        lambda: &Float,
        n_grid: usize,
        prec: u32,
    ) -> Result<(Vec<Float>, Vec<Float>)> {
        super::stencil::build_hp(lambda, n_grid, prec)
    }

    /// Compatibility wrapper; invalid input panics. Use the checked variant.
    pub fn build_pw_matrix(lambda: &Float, n_grid: usize, prec: u32) -> (Vec<Float>, Vec<Float>) {
        try_build_pw_matrix(lambda, n_grid, prec).expect("valid representable HP prolate grid")
    }

    /// Dense Galerkin forms for `PW_lambda` in a caller-supplied trial basis.
    /// Columns of `basis_vectors` live on the finite-difference interior grid
    /// and need only be linearly independent; orthogonality is not assumed.
    #[derive(Clone, Debug)]
    pub struct ProlateSubspaceFormsHp {
        pub ambient_dimension: usize,
        pub basis_dimension: usize,
        pub precision_bits: u32,
        /// `V^T PW_lambda V`, row-major.
        pub stiffness: Vec<Float>,
        /// `V^T V`, row-major. The common grid-spacing factor cancels from
        /// the generalized quotient and is omitted from both forms.
        pub gram: Vec<Float>,
    }

    fn qualify_trial_basis(basis: &[Vec<Float>], p: u32) -> Result<()> {
        // Reorthogonalize before forming Gram: otherwise rounding squares the
        // basis condition number and can manufacture a positive null pivot.
        let work = p
            .checked_add(64)
            .ok_or_else(|| anyhow::anyhow!("rank precision overflow"))?;
        let threshold = Float::with_val(work, 1) >> (p / 2);
        let mut orthogonal: Vec<Vec<Float>> = Vec::with_capacity(basis.len());
        for column in basis {
            let scale = column
                .iter()
                .map(|x| x.clone().abs())
                .max_by(Float::total_cmp)
                .unwrap();
            anyhow::ensure!(
                !scale.is_zero(),
                "prolate trial basis contains a zero vector"
            );
            let mut v = column
                .iter()
                .map(|x| Float::with_val(work, x) / &scale)
                .collect::<Vec<_>>();
            let original = Float::with_val(work, Float::dot(v.iter().zip(&v)));
            for _ in 0..2 {
                for q in &orthogonal {
                    let projection = Float::with_val(work, Float::dot(v.iter().zip(q)));
                    for (x, y) in v.iter_mut().zip(q) {
                        *x -= Float::with_val(work, &projection * y);
                    }
                }
            }
            let squared_norm = Float::with_val(work, Float::dot(v.iter().zip(&v)));
            anyhow::ensure!(squared_norm.is_finite() && squared_norm > Float::with_val(work, &original * &threshold),
                "prolate trial basis is dependent or numerically unresolved at the working precision");
            let norm = squared_norm.sqrt();
            for x in &mut v {
                *x /= &norm;
            }
            orthogonal.push(v);
        }
        Ok(())
    }

    fn qualify_gram(gram: &[Float], n: usize, p: u32) -> Result<()> {
        use rug::float::Round;
        use xc_numerics::mpfr_interval::MpfrInterval as I;
        anyhow::ensure!(
            (33..=1_000_000).contains(&p) && n > 0 && n.checked_mul(n) == Some(gram.len()),
            "invalid prolate Gram shape or precision"
        );
        anyhow::ensure!(
            gram.iter().all(Float::is_finite),
            "nonfinite prolate Gram form"
        );
        for i in 0..n {
            for j in 0..i {
                anyhow::ensure!(
                    gram[i * n + j] == gram[j * n + i],
                    "prolate Gram form must be exactly symmetric"
                );
            }
        }
        let scale_exp = gram.iter().filter_map(Float::get_exp).max().unwrap_or(0);
        let work = p + 64;
        anyhow::ensure!(
            gram.len() as u128 * (u128::from(work).div_ceil(8) + 192) * 64 <= (8u128 << 30),
            "prolate Gram qualification exceeds workspace budget"
        );
        let scaled = gram
            .iter()
            .map(|x| {
                let mut value = Float::with_val(work, x);
                value >>= scale_exp;
                anyhow::ensure!(
                    value.is_finite() && (x.is_zero() || !value.is_zero()),
                    "prolate Gram scaling exceeds exponent range"
                );
                anyhow::ensure!(
                    value
                        .get_exp()
                        .is_none_or(|e| i64::from(e).unsigned_abs() <= u64::from(work) * 4),
                    "prolate Gram dynamic range is numerically unresolved"
                );
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        let max_diagonal = (0..n)
            .map(|i| &scaled[i * n + i])
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        anyhow::ensure!(
            max_diagonal > &0,
            "prolate Gram form has no positive diagonal"
        );
        let shift = Float::with_val(work, max_diagonal) >> (p / 2);
        let intervals = scaled
            .iter()
            .enumerate()
            .map(|(k, x)| {
                if k / n == k % n {
                    I::new(
                        Float::with_val_round(work, x - &shift, Round::Down).0,
                        Float::with_val_round(work, x - &shift, Round::Up).0,
                    )
                } else {
                    I::from_float(x, work)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let rational = intervals
            .iter()
            .map(I::to_rational_interval)
            .collect::<Vec<_>>();
        match xc_certify::exact::interval_symmetric_ldlt_inertia_mpfr(&rational, n, work)? {
            xc_certify::exact::IntervalInertiaResult::Conclusive {
                positive,
                negative: 0,
                ..
            } if positive == n => Ok(()),
            _ => anyhow::bail!(
                "prolate Gram form is dependent or insufficiently separated from singularity"
            ),
        }
    }

    /// Build the generalized symmetric Ritz pair `(V^T PW V, V^T V)`.
    pub fn build_pw_subspace_forms(
        lambda: &Float,
        n_grid: usize,
        basis_vectors: &[Vec<Float>],
        precision_bits: u32,
    ) -> Result<ProlateSubspaceFormsHp> {
        if !(33..=1_000_000).contains(&precision_bits)
            || n_grid == 0
            || n_grid.is_multiple_of(2)
            || basis_vectors.is_empty()
        {
            anyhow::bail!(
                "prolate HP subspace forms require precision above 32 bits, an odd positive grid, and a nonempty basis"
            );
        }
        if !lambda.is_finite() || lambda <= &Float::with_val(precision_bits, 0) {
            anyhow::bail!("prolate HP subspace lambda must be finite and positive");
        }
        if basis_vectors
            .iter()
            .any(|vector| vector.len() != n_grid || vector.iter().any(|value| !value.is_finite()))
        {
            anyhow::bail!("every prolate trial vector must be finite and match the grid dimension");
        }
        if basis_vectors.len() > n_grid {
            anyhow::bail!("prolate trial basis has more columns than the ambient dimension");
        }

        let (diagonal, off_diagonal) = try_build_pw_matrix(lambda, n_grid, precision_bits)?;
        let basis = basis_vectors
            .iter()
            .map(|vector| {
                vector
                    .iter()
                    .map(|value| Float::with_val(precision_bits, value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        qualify_trial_basis(&basis, precision_bits)?;
        let applied = basis
            .iter()
            .map(|vector| {
                let mut output = vec![Float::with_val(precision_bits, 0); n_grid];
                for row in 0..n_grid {
                    output[row] = Float::with_val(precision_bits, &diagonal[row]);
                    output[row] *= &vector[row];
                    if row > 0 {
                        let mut term = Float::with_val(precision_bits, &off_diagonal[row - 1]);
                        term *= &vector[row - 1];
                        output[row] += term;
                    }
                    if row + 1 < n_grid {
                        let mut term = Float::with_val(precision_bits, &off_diagonal[row]);
                        term *= &vector[row + 1];
                        output[row] += term;
                    }
                }
                output
            })
            .collect::<Vec<_>>();
        let basis_dimension = basis.len();
        let entries = basis_dimension
            .checked_mul(basis_dimension)
            .ok_or_else(|| anyhow::anyhow!("prolate trial-basis shape overflows usize"))?;
        let mut stiffness = vec![Float::with_val(precision_bits, 0); entries];
        let mut gram = stiffness.clone();
        for row in 0..basis_dimension {
            for column in 0..=row {
                let gram_terms = basis[row]
                    .iter()
                    .zip(&basis[column])
                    .map(|(left, right)| {
                        let mut value = Float::with_val(precision_bits, left);
                        value *= right;
                        value
                    })
                    .collect::<Vec<_>>();
                let gram_value = xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(
                    gram_terms,
                    precision_bits,
                );
                let stiffness_terms = basis[row]
                    .iter()
                    .zip(&applied[column])
                    .map(|(left, right)| {
                        let mut value = Float::with_val(precision_bits, left);
                        value *= right;
                        value
                    })
                    .collect::<Vec<_>>();
                let stiffness_value = xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(
                    stiffness_terms,
                    precision_bits,
                );
                anyhow::ensure!(
                    gram_value.is_finite() && stiffness_value.is_finite(),
                    "prolate projected form is outside MPFR range"
                );
                let indices = [
                    row * basis_dimension + column,
                    column * basis_dimension + row,
                ];
                for index in indices {
                    gram[index] = gram_value.clone();
                    stiffness[index] = stiffness_value.clone();
                }
            }
        }
        Ok(ProlateSubspaceFormsHp {
            ambient_dimension: n_grid,
            basis_dimension,
            precision_bits,
            stiffness,
            gram,
        })
    }

    /// Solve either algebraic generalized extreme of a nonorthogonal prolate
    /// trial subspace. Precision-scaled rank/conditioning gates reject unresolved
    /// bases before the computed Cholesky solve. These are numerical admissibility
    /// checks, not certificates of continuum accuracy or exact basis rank.
    pub fn solve_pw_subspace_extreme(
        forms: &ProlateSubspaceFormsHp,
        target: xc_core::EigenTarget,
    ) -> Result<xc_solver::DenseGeneralizedEigenpairReportHp> {
        use xc_core::DecimalLiteral;
        if forms.basis_dimension > forms.ambient_dimension {
            anyhow::bail!("prolate trial basis exceeds its ambient dimension");
        }
        qualify_gram(&forms.gram, forms.basis_dimension, forms.precision_bits)?;
        let problem = xc_solver::DenseGeneralizedProblemHp::new(
            &forms.stiffness,
            &forms.gram,
            forms.basis_dimension,
        )
        .map_err(anyhow::Error::new)?;
        let tolerance = Float::with_val(forms.precision_bits, 1) >> (forms.precision_bits / 2);
        let stopping_tolerance = DecimalLiteral::new(tolerance.to_string())?;
        let mut report = xc_solver::solve_dense_generalized_whitening_hp(
            &problem,
            &xc_solver::GeneralizedExtremeConfigHp {
                target,
                precision_bits: forms.precision_bits,
                absolute_residual_tolerance: stopping_tolerance.clone(),
                scaled_backward_error_tolerance: stopping_tolerance.clone(),
                ritz_value_stability_tolerance: stopping_tolerance.clone(),
                maximum_iterations: 100,
                minimum_iterations: 1,
            },
        )
        .map_err(anyhow::Error::new)?;
        report
            .algorithm
            .push_str(";prolate_precision_scaled_ritz_stopping_v2");
        anyhow::ensure!(
            report.scaled_backward_error <= tolerance,
            "prolate Ritz pair fails the precision-scaled backward-error requirement"
        );
        Ok(report)
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
        /// Neither even nor odd, including zero or nonfinite input.
        Indeterminate,
    }

    /// Classify the parity of HP vector `v` under the index reflection
    /// `i ↔ n-1-i` (which corresponds to `x ↔ -x` on the symmetric FD
    /// grid). Returns `HpParity::Indeterminate` when the
    /// vector is zero/nonfinite or both even/odd
    /// deviations exceed the classification tolerance.
    pub fn parity_of(v: &[Float], prec: u32) -> HpParity {
        if !(32..=1_000_000).contains(&prec) || v.is_empty() || v.iter().any(|x| !x.is_finite()) {
            return HpParity::Indeterminate;
        }
        let scale = v
            .iter()
            .map(|x| x.clone().abs())
            .max_by(|a, b| a.partial_cmp(b).expect("finite entries"))
            .unwrap();
        if scale.is_zero() {
            return HpParity::Indeterminate;
        }
        let mut total = Float::with_val(prec, 0);
        let mut even = total.clone();
        let mut odd = total.clone();
        for (a, b) in v.iter().zip(v.iter().rev()) {
            let a = Float::with_val(prec, a / &scale);
            let b = Float::with_val(prec, b / &scale);
            total += a.clone().abs();
            even += Float::with_val(prec, &a - &b).abs();
            odd += Float::with_val(prec, &a + &b).abs();
        }
        even /= &total;
        odd /= total;
        let tolerance = Float::with_val(prec, Float::parse("1e-3").unwrap());
        if even < tolerance {
            HpParity::Even
        } else if odd < tolerance {
            HpParity::Odd
        } else {
            HpParity::Indeterminate
        }
    }

    /// Count zero crossings of HP vector `v`. Values below
    /// `max_abs * 1e-6` are skipped (boundary noise). Invalid input panics;
    /// use `try_count_nodes` for fallible validation.
    pub fn count_nodes(v: &[Float], prec: u32) -> usize {
        try_count_nodes(v, prec).expect("valid finite HP node-count samples and precision")
    }

    /// Count sign changes after removing samples smaller than 1e-6 of the
    /// largest magnitude. This scale-invariant heuristic is not a Sturm index.
    pub fn try_count_nodes(v: &[Float], prec: u32) -> Result<usize> {
        super::grid_contract::nodes_hp(v, prec)
    }

    /// Linearly interpolate HP function `values` defined on the FD grid
    /// `x_i = -λ + (i+1)h` at point `x ∈ [-λ, λ]`. Returns zero outside
    /// the support (Dirichlet BC).
    ///
    /// `h` is the grid spacing as an HP value. Invalid input panics;
    /// use `try_interp_grid` for fallible validation.
    pub fn interp_grid(values: &[Float], lambda: &Float, h: &Float, x: &Float, prec: u32) -> Float {
        try_interp_grid(values, lambda, h, x, prec)
            .expect("valid representable HP Dirichlet interpolation")
    }

    /// Checked linear interpolation on the uniform grid with zero endpoint
    /// values. Spacing must equal 2*lambda/(N+1) rounded at working precision.
    /// Returns zero outside support. Invalid/nonfinite data or arithmetic fail.
    pub fn try_interp_grid(
        values: &[Float],
        lambda: &Float,
        h: &Float,
        x: &Float,
        prec: u32,
    ) -> Result<Float> {
        super::grid_contract::validate_grid(values, lambda, h, prec)?;
        anyhow::ensure!(x.is_finite(), "interpolation point must be finite");
        let result = super::grid_contract::interpolate(values, lambda, x, prec);
        anyhow::ensure!(
            result.is_finite(),
            "interpolation result is outside MPFR range"
        );
        Ok(result)
    }

    // ===========================================================================
    // Prolate eigenvalue cache
    // ===========================================================================
    //
    // The dominant cost in the historical finite-Dirichlet route at HP-1000 is the full
    // tridiagonal QR on PW_λ — at N=8001 prec=3338 this is ~30 minutes
    // of wall-time. The output is just the eigenvalue vector (a few MB
    // serialized at HP-1000). It's deterministic in `(λ², n_grid, prec)`
    // and reusable across runs at the same configuration.
    //
    // Cache layout (mirrors xc-numerics::quadrature::hp gl_cache):
    //   <cache root>/prolate_eigvals_cache/lambda_sq{LSQ}_ngrid{N}_prec{P}.json[.zip]
    //
    // The standalone API reads a local `.json.zip` in memory and computes a
    // fresh spectrum on a miss. Managed remote resolution uses
    // `prolate_spectrum_via_cache`.
    //
    // The historical integer key is used only for integer source lambda.
    // Every other cutoff uses an exact-source identity; it does not bypass caching.

    /// Verify a loaded eigenvalue vector satisfies the prolate-spectrum
    /// structural identities:
    ///
    ///   1. Count = `n_grid` (after rounding to odd if the caller's
    ///      n_grid was even).
    ///   2. Ascending order: `e[k] ≤ e[k+1]` for all `k`.
    ///   3. Every value reproduces its ordered index in the stored finite
    ///      matrix at a precision-scaled radius; continuum asymptotics do not
    ///      determine whether a finite-grid spectrum is valid.
    ///
    /// Returns `None` if all identities hold; `Some(reason)` otherwise.
    fn prolate_cache_structural_check(
        evals: &[Float],
        n_expected: usize,
        lambda_sq: LambdaSq,
        prec: u32,
    ) -> Option<String> {
        if evals.len() != n_expected {
            return Some(format!(
                "eigenvalue count {} != expected {}",
                evals.len(),
                n_expected
            ));
        }

        // Ascending order check.
        for k in 0..(evals.len().saturating_sub(1)) {
            if evals[k] > evals[k + 1] {
                return Some(format!(
                    "not ascending at index {}: e[{}] > e[{}]",
                    k,
                    k,
                    k + 1
                ));
            }
        }

        if evals.iter().any(|value| !value.is_finite()) {
            return Some("nonfinite prolate eigenvalue".into());
        }
        if !(32..=1_000_000).contains(&prec)
            || n_expected == 0
            || !lambda_sq.value_f64.is_finite()
            || lambda_sq.value_f64 <= 0.0
        {
            return Some("invalid finite prolate source dimensions or cutoff".into());
        }
        if let Err(error) = super::legendre::resource_budget(n_expected, 2, 0, prec, false) {
            return Some(error.to_string());
        }
        let lambda = Float::with_val(prec, lambda_sq.value_f64).sqrt();
        let (d, b) = match try_build_pw_matrix(&lambda, n_expected, prec) {
            Ok(matrix) => matrix,
            Err(error) => return Some(error.to_string()),
        };
        if let Err(error) = super::legendre::hp::validate_spectrum(&d, &b, evals, prec) {
            return Some(error.to_string());
        }

        None
    }

    /// Integer compatibility keys require exact positive integrality. A nearby
    /// cutoff is a different source and must never alias this key.
    fn lambda_sq_int_for_key(lambda_sq: &Float) -> Option<LambdaSq> {
        if !lambda_sq.is_finite() || lambda_sq <= &0 || !lambda_sq.is_integer() {
            return None;
        }
        lambda_sq
            .to_integer()
            .and_then(|i| i.to_u64())
            .map(LambdaSq::integer)
    }

    fn prolate_cache_dir() -> Option<std::path::PathBuf> {
        let dir = crate::standalone_cache_root().join("prolate_eigvals_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    fn prolate_cache_filename(lambda_sq: LambdaSq, n_grid: usize, prec: u32) -> String {
        format!(
            "lambda_sq{}_ngrid{}_prec{}.json",
            lambda_sq.filename_str(),
            n_grid,
            prec
        )
    }

    fn prolate_cache_zip_path(
        lambda_sq: LambdaSq,
        n_grid: usize,
        prec: u32,
    ) -> Option<std::path::PathBuf> {
        prolate_cache_dir().map(|d| {
            let f = prolate_cache_filename(lambda_sq, n_grid, prec);
            d.join(format!("{}.zip", f))
        })
    }

    /// Parse a prolate eigenvalue cache JSON.
    /// Expects schema_version 1 envelope format. Returns `None` on any
    /// structural mismatch or a stale `toolkit_version`.
    fn parse_prolate_cache_json(
        data: &str,
        lambda_sq: LambdaSq,
        n_expected: usize,
        prec: u32,
    ) -> Option<Vec<Float>> {
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let obj = parsed.as_object()?;
        if !(32..=1_000_000).contains(&prec)
            || n_expected == 0
            || n_expected >= u32::MAX as usize
            || obj.get("schema_version").and_then(|v| v.as_u64()) != Some(1)
            || obj.get("lambda_sq_identity").and_then(|v| v.as_str())
                != Some(lambda_sq.filename_str().as_str())
            || obj.get("lambda_sq_mode").and_then(|v| v.as_str()) != Some(lambda_sq.mode_str())
            || obj.get("n_grid").and_then(|v| v.as_u64()) != Some(n_expected as u64)
            || obj.get("precision_bits").and_then(|v| v.as_u64()) != Some(u64::from(prec))
        {
            return None;
        }

        if obj.get("arithmetic_semantics").and_then(|v| v.as_str())
            != Some("prolate-fd-working-precision-v2")
            || obj.get("qr_arithmetic").and_then(|v| v.as_str())
                != Some(xc_numerics::eigen::TRIDIAG_QR_SEMANTICS)
        {
            return None;
        }
        let file_ver = obj.get("toolkit_version").and_then(|v| v.as_str())?;
        if prolate_version_is_older(file_ver, &prolate_effective_min_version()) {
            return None;
        }

        let arr = obj.get("eigenvalues")?.as_array()?;
        if arr.len() != n_expected {
            return None;
        }
        let mut evals = Vec::with_capacity(n_expected);
        for s in arr {
            evals.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some(evals)
    }

    #[cfg(test)]
    mod exhaustive_resumed_version_contract {
        use super::*;
        #[test]
        fn exhaustive_resumed_prolate_rejects_malformed_future_versions() {
            for version in ["999.0.bad", "999.0", "999.00.0", "999.0.0-"] {
                assert!(
                    prolate_version_is_older(version, "0.13.0"),
                    "accepted {version}"
                );
            }
            assert!(!prolate_version_is_older("0.15.1", "0.13.0"));
        }
        #[test]
        fn exhaustive_resumed_prolate_prerelease_is_below_release_floor() {
            assert!(prolate_version_is_older("0.13.0-alpha", "0.13.0"));
            assert!(!prolate_version_is_older("0.13.0", "0.13.0"));
        }
    }

    /// Reject an invalid version or one below the required producer floor.
    fn prolate_version_is_older(a: &str, b: &str) -> bool {
        match (
            xc_cache::ToolkitVersion::parse(a),
            xc_cache::ToolkitVersion::parse(b),
        ) {
            (Ok(actual), Ok(minimum)) => actual < minimum,
            _ => true,
        }
    }

    fn warn_prolate_cache_skip(path: &std::path::Path, reason: &str) {
        xc_core::progress_message!(
            "[prolate_cache] WARNING: skipping {} ({}); recomputing",
            path.display(),
            reason
        );
    }

    // Generated point decimals use fewer than p digits plus sign/exponent.
    // Allow generous metadata overhead while bounding hostile ZIP expansion.
    fn prolate_spectrum_json_limit(count: usize, precision_bits: u32) -> Option<u64> {
        (count as u64)
            .checked_add(2)?
            .checked_mul(u64::from(precision_bits).checked_add(128)?)?
            .checked_add(65_536)
            .filter(|limit| *limit < u64::MAX)
    }

    fn load_prolate_eigvals_from_zip(
        zip_path: &std::path::Path,
        json_filename: &str,
        lambda_sq: LambdaSq,
        n_expected: usize,
        prec: u32,
    ) -> Option<(Vec<Float>, String)> {
        use std::io::Read;
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        let limit = prolate_spectrum_json_limit(n_expected, prec)?;
        let entry = archive.by_name(json_filename).ok()?;
        if entry.size() > limit {
            return None;
        }
        let mut data = String::new();
        entry.take(limit + 1).read_to_string(&mut data).ok()?;
        if data.len() as u64 > limit {
            return None;
        }
        let parsed = parse_prolate_cache_json(&data, lambda_sq, n_expected, prec)?;
        Some((parsed, data))
    }

    #[cfg(test)]
    mod standalone_zip_size_contract {
        use super::*;
        #[test]
        fn bounded_zip_read_accepts_generated_shape_and_rejects_oversized_decoded_json() {
            use std::io::Write;
            let root_dir = xc_core::test_support::TestDir::new("prolate-size");
            let root = root_dir.to_path_buf();
            let lambda = LambdaSq::integer(1);
            let name = "spectrum.json";
            let json = serde_json::json!({ "schema_version":1, "toolkit_version":PROLATE_TOOLKIT_VERSION,
                "arithmetic_semantics":"prolate-fd-working-precision-v2", "qr_arithmetic":xc_numerics::eigen::TRIDIAG_QR_SEMANTICS, "lambda_sq_identity":lambda.filename_str(),
                "lambda_sq_mode":lambda.mode_str(), "n_grid":1, "precision_bits":64, "eigenvalues":["1.5"] }).to_string();
            for oversized in [false, true] {
                let path = root.join(if oversized { "large.zip" } else { "valid.zip" });
                let mut data = json.clone();
                if oversized {
                    data.push_str(
                        &" ".repeat(prolate_spectrum_json_limit(1, 64).unwrap() as usize),
                    );
                }
                assert!(parse_prolate_cache_json(&data, lambda, 1, 64).is_some());
                let mut writer = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
                writer
                    .start_file(
                        name,
                        zip::write::SimpleFileOptions::default()
                            .compression_method(zip::CompressionMethod::Deflated),
                    )
                    .unwrap();
                writer.write_all(data.as_bytes()).unwrap();
                writer.finish().unwrap();
                assert_eq!(
                    load_prolate_eigvals_from_zip(&path, name, lambda, 1, 64).is_some(),
                    !oversized
                );
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    /// Try to load prolate eigenvalues from cache for `(λ², n_grid, prec)`.
    /// Returns `None` on cache miss, parse failure, or structural
    /// validation failure (with diagnostic warning to stderr in the
    /// failure cases). The lookup depth is governed by `mode`:
    ///   - `Off`          — never read (returns `None` immediately).
    ///   - `JsonOnly`     — local `.json` only.
    ///   - `JsonZip`      — local `.json.zip`, then compute on a miss.
    ///
    /// Managed remote resolution is provided by `prolate_spectrum_via_cache`.
    fn load_prolate_eigvals_cache(
        lambda_sq: LambdaSq,
        n_grid: usize,
        prec: u32,
        mode: CacheMode,
    ) -> Option<Vec<Float>> {
        if mode == CacheMode::Off {
            return None;
        }

        // Caches are zip-only: read straight from the .json.zip
        // (decompress in memory), never write a decompressed .json.
        // JsonOnly is a read no-op because current cache files are zip-only.
        if mode == CacheMode::JsonOnly {
            return None;
        }

        // Local zip — in memory.
        if let Some(evals) = try_load_local_prolate_zip(lambda_sq, n_grid, prec) {
            return Some(evals);
        }

        None
    }

    /// Load from a local `.json.zip`. Decompresses in memory; does NOT
    /// write a decompressed `.json`. Returns the validated eigenvalues,
    /// or `None` if the zip is absent, corrupt, or structurally invalid.
    fn try_load_local_prolate_zip(
        lambda_sq: LambdaSq,
        n_grid: usize,
        prec: u32,
    ) -> Option<Vec<Float>> {
        let zip_path = prolate_cache_zip_path(lambda_sq, n_grid, prec)?;
        if !zip_path.exists() {
            return None;
        }
        let json_filename = prolate_cache_filename(lambda_sq, n_grid, prec);
        match load_prolate_eigvals_from_zip(&zip_path, &json_filename, lambda_sq, n_grid, prec) {
            Some((evals, _json_string)) => {
                if let Some(reason) =
                    prolate_cache_structural_check(&evals, n_grid, lambda_sq, prec)
                {
                    warn_prolate_cache_skip(&zip_path, &reason);
                    None
                } else {
                    Some(evals)
                }
            }
            None => {
                warn_prolate_cache_skip(&zip_path, "zip open / decompress / shape parse failed");
                None
            }
        }
    }

    fn save_prolate_eigvals_cache(
        lambda_sq: LambdaSq,
        n_grid: usize,
        prec: u32,
        evals: &[Float],
        mode: CacheMode,
    ) {
        // Off and JsonOnly write nothing: the cache is zip-only.
        if matches!(mode, CacheMode::Off | CacheMode::JsonOnly) {
            return;
        }

        let strs: Vec<String> = evals.iter().map(|f| f.to_string()).collect();
        // Versioned envelope: object with metadata + eigenvalue array.
        let json = serde_json::json!({
            "schema_version": 1,
            "toolkit_version": PROLATE_TOOLKIT_VERSION,
            "arithmetic_semantics": "prolate-fd-working-precision-v2",
            "qr_arithmetic": xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
            "lambda_sq": lambda_sq.value_f64,
            "lambda_sq_mode": lambda_sq.mode_str(),
            "lambda_sq_identity": lambda_sq.filename_str(),
            "n_grid": n_grid,
            "precision_bits": prec,
            "eigenvalues": strs,
        });
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
        // large_file(true): the `zip` crate defaults to classic
        // (non-Zip64) headers, which silently abort the write once
        // either the uncompressed or compressed size crosses 4 GiB
        // (see xc_spectral::ccm::hp::tau_cache::compress_to_zip for the
        // full writeup of this failure mode). The eigenvalue spectrum
        // is small in practice, but this keeps the write path uniform
        // and safe regardless of grid size.
        let mut buf: Vec<u8> = Vec::with_capacity(json_str.len() / 2);
        {
            use std::io::Write;
            let cursor = std::io::Cursor::new(&mut buf);
            let mut writer = zip::ZipWriter::new(cursor);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .large_file(true);
            if writer.start_file(&entry_name, opts).is_err() {
                return;
            }
            if writer.write_all(json_str.as_bytes()).is_err() {
                return;
            }
            if writer.finish().is_err() {
                return;
            }
        }
        if let Err(e) = xc_cache::atomic_replace_cache_file(&zip_path, &buf) {
            xc_core::progress_message!(
                "[prolate_cache] WARNING: could not write {}: {}",
                zip_path.display(),
                e
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
        Ok {
            path: std::path::PathBuf,
            n_grid: usize,
            prec: u32,
            lambda_sq: LambdaSq,
        },
        /// File skipped because its name didn't match the expected
        /// `lambda_sq{L}_ngrid{N}_prec{P}.json[.zip]` pattern.
        Skipped {
            path: std::path::PathBuf,
            reason: String,
        },
        /// Filename matched but the file failed to load (parse JSON,
        /// decompress zip, etc.).
        LoadFailed {
            path: std::path::PathBuf,
            n_grid: usize,
            prec: u32,
            lambda_sq: LambdaSq,
            reason: String,
        },
        /// File loaded but the eigenvalue vector failed at least one
        /// of the prolate-spectrum structural identities (count, sort
        /// order, ground-state magnitude ≈ 2π·λ², per-element
        /// finiteness).
        StructurallyInvalid {
            path: std::path::PathBuf,
            n_grid: usize,
            prec: u32,
            lambda_sq: LambdaSq,
            reason: String,
        },
    }

    /// Aggregate report from `verify_prolate_eigvals_cache_dir`.
    #[derive(Debug, Clone)]
    pub struct ProlateCacheVerifyReport {
        /// Identity of finite-matrix cache admission and replay.
        pub validation_semantics: &'static str,
        /// Directory that was scanned.
        pub directory: std::path::PathBuf,
        /// One status entry per file in `directory`.
        pub statuses: Vec<ProlateCacheFileStatus>,
    }

    impl ProlateCacheVerifyReport {
        /// Count of files that passed all checks.
        pub fn ok_count(&self) -> usize {
            self.statuses
                .iter()
                .filter(|s| matches!(s, ProlateCacheFileStatus::Ok { .. }))
                .count()
        }
        /// Count of files that failed at least one check (load or
        /// structural). Skipped files are not counted as failures.
        pub fn failure_count(&self) -> usize {
            self.statuses
                .iter()
                .filter(|s| {
                    matches!(
                        s,
                        ProlateCacheFileStatus::LoadFailed { .. }
                            | ProlateCacheFileStatus::StructurallyInvalid { .. }
                    )
                })
                .count()
        }
        /// All failure entries (load + structural), for callers that
        /// want to print only the bad files.
        pub fn failures(&self) -> impl Iterator<Item = &ProlateCacheFileStatus> {
            self.statuses.iter().filter(|s| {
                matches!(
                    s,
                    ProlateCacheFileStatus::LoadFailed { .. }
                        | ProlateCacheFileStatus::StructurallyInvalid { .. }
                )
            })
        }
    }

    fn parse_prolate_cache_filename(name: &str) -> Option<(LambdaSq, usize, u32)> {
        // `lambda_sq{LSQ}_ngrid{N}_prec{P}.json[.zip]`
        let stem = name
            .strip_suffix(".json.zip")
            .or_else(|| name.strip_suffix(".json"))?;
        let after_lsq = stem.strip_prefix("lambda_sq")?;
        let (lsq_str, rest) = after_lsq.split_once("_ngrid")?;

        let (n_str, prec_str) = rest.split_once("_prec")?;

        let lambda_sq = LambdaSq::from_filename_str(lsq_str)?;
        let n_grid: usize = n_str.parse().ok()?;
        let prec: u32 = prec_str.parse().ok()?;
        Some((lambda_sq, n_grid, prec))
    }

    /// Finite-source ordered-index replay policy for legacy cache inspection.
    pub const PROLATE_CACHE_VALIDATION_SEMANTICS: &str =
        "prolate_cache_ordered_enclosure_replay_v3";

    /// Walk the prolate eigenvalue cache directory and structurally
    /// verify every `lambda_sq{L}_ngrid{N}_prec{P}.json[.zip]` file.
    /// Returns a per-file status report; does not mutate any files.
    pub fn verify_prolate_eigvals_cache_dir(
        dir: &std::path::Path,
    ) -> std::io::Result<ProlateCacheVerifyReport> {
        let mut statuses: Vec<ProlateCacheFileStatus> = Vec::new();

        if !dir.exists() {
            return Ok(ProlateCacheVerifyReport {
                validation_semantics: PROLATE_CACHE_VALIDATION_SEMANTICS,
                directory: dir.to_path_buf(),
                statuses,
            });
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
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
                load_prolate_eigvals_from_zip(&path, &json_filename, lambda_sq, n_grid, prec)
                    .map(|(p, _)| p)
            } else {
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|data| parse_prolate_cache_json(&data, lambda_sq, n_grid, prec))
            };

            let evals = match parsed {
                Some(e) => e,
                None => {
                    statuses.push(ProlateCacheFileStatus::LoadFailed {
                        path: path.clone(),
                        lambda_sq,
                        n_grid,
                        prec,
                        reason: "parse / decompress failed".to_string(),
                    });
                    continue;
                }
            };

            match prolate_cache_structural_check(&evals, n_grid, lambda_sq, prec) {
                None => {
                    statuses.push(ProlateCacheFileStatus::Ok {
                        path,
                        lambda_sq,
                        n_grid,
                        prec,
                    });
                }
                Some(reason) => {
                    statuses.push(ProlateCacheFileStatus::StructurallyInvalid {
                        path,
                        lambda_sq,
                        n_grid,
                        prec,
                        reason,
                    });
                }
            }
        }

        Ok(ProlateCacheVerifyReport {
            validation_semantics: PROLATE_CACHE_VALIDATION_SEMANTICS,
            directory: dir.to_path_buf(),
            statuses,
        })
    }

    /// HP result of `compute_k_lambda`. Stores the sample grid and k_λ
    /// values in HP.
    #[derive(Debug, Clone)]
    pub struct HpProlateResult {
        /// Bounded Legendre or explicit historical finite-Dirichlet model.
        pub discretization: String,
        /// Caller-supplied maximum resolution budget (`n_grid`).
        pub resolution_budget: usize,
        /// Retained even Legendre coefficient count, or FD interior node count.
        pub basis_dimension: usize,
        /// Computed omitted-coupling plus finite residual diagnostic, not a
        /// certified continuum eigenfunction or sampling error bound.
        pub relative_operator_residual: Option<Float>,
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
    ///   1. Build the even bounded-endpoint Legendre tridiagonal at HP.
    ///   2. Compute all eigenvalues in HP via `tridiag_eigenvalues_hp`.
    ///   3. Recover lowest-lying eigenvectors via shifted inverse iteration
    ///      and select even-block indices 0 and 2 (h_0 and h_4).
    ///   4. Form h_λ = h_4 - r·h_0 with r = ∫h_4 / ∫h_0 in HP.
    ///   5. Sample k_λ on a logarithmic grid in [λ⁻¹, λ] in HP.
    ///
    /// `n_grid` is the maximum number of even Legendre coefficients. This
    /// legacy parameter is a resolution budget, not a spatial sampling grid.
    /// Unresolved truncation returns an error, without a Dirichlet fallback.
    /// `n_sample` is the number of comparison-grid points. The source cutoff is
    /// rounded to `prec` bits; Legendre assembly, solve, and polynomial evaluation
    /// use `prec + 64` bits, and returned samples/scalars are rounded to `prec`.
    ///
    /// `mode` selects the prolate-eigenvalue cache strategy (see
    /// [`xc_numerics::quadrature::CacheMode`]): `Off` always recomputes
    /// the spectrum and `JsonZip` consults the local compressed cache.
    /// Use `compute_k_lambda_via_cache` for managed remote resolution.
    /// Pass `CacheMode::default()` for the standard behavior.
    // Keep remainder arithmetic for the Rust 1.85 MSRV.
    #[allow(unknown_lints, clippy::manual_is_multiple_of)]
    pub fn compute_k_lambda(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
        mode: CacheMode,
    ) -> Result<HpProlateResult> {
        compute_k_lambda_legendre_inner(
            lambda,
            n_grid,
            n_sample,
            prec,
            ProlateCacheRoute::Standalone(mode),
        )
    }

    /// Computes the HP prolate comparison kernel through the common cache fabric.
    ///
    /// # Mathematical semantics
    /// Uses the bounded-endpoint even Legendre prolate-wave operator and constructs
    /// the normalized `h_0`/`h_4` comparison kernel on the requested sample grid.
    ///
    /// # Precision
    /// Legendre assembly, eigensolution, eigenvectors, and sampling use MPFR at
    /// `prec + 64` bits; returned samples and scalars are rounded to `prec`.
    /// The cached artifact contains the complete ordered spectrum.
    ///
    /// # Failure states
    /// Invalid dimensions, eigensolver failures, corrupt or incompatible cache
    /// artifacts, required-cache misses, and missing writable overlays are errors.
    ///
    /// # Assurance and validity
    /// Cached spectra are dimension checked, finite, sorted, and index-replayed
    /// against the current tridiagonal before use. Downstream eigenvectors are
    /// recomputed against that same tridiagonal.
    ///
    /// # Cache effects
    /// All lookup and persistence follows `cache`; this function has no direct
    /// filesystem layout, GitHub URL, curl process, or repository-specific policy.
    ///
    /// # Example
    /// Configure an [`ArtifactCacheContext`] with ordered overlays, then pass it
    /// here to make reuse and write behavior explicit.
    #[allow(unknown_lints, clippy::manual_is_multiple_of)]
    pub fn compute_k_lambda_via_cache(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<HpProlateResult> {
        compute_k_lambda_legendre_inner(
            lambda,
            n_grid,
            n_sample,
            prec,
            ProlateCacheRoute::Fabric(cache),
        )
    }

    /// Explicit compatibility route for the historical finite Dirichlet model.
    /// This has logarithmically slow endpoint convergence for small cutoffs and
    /// is not the ordinary bounded-endpoint prolate approximation.
    pub fn compute_k_lambda_finite_dirichlet(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
        mode: CacheMode,
    ) -> Result<HpProlateResult> {
        compute_k_lambda_inner(
            lambda,
            n_grid,
            n_sample,
            prec,
            ProlateCacheRoute::Standalone(mode),
        )
    }

    /// Managed-cache counterpart of the explicit finite-Dirichlet route.
    pub fn compute_k_lambda_finite_dirichlet_via_cache(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
        cache: &ArtifactCacheContext<'_>,
    ) -> Result<HpProlateResult> {
        compute_k_lambda_inner(
            lambda,
            n_grid,
            n_sample,
            prec,
            ProlateCacheRoute::Fabric(cache),
        )
    }

    fn compute_k_lambda_legendre_inner(
        lambda: &Float,
        budget: usize,
        n_sample: usize,
        prec: u32,
        cache_route: ProlateCacheRoute<'_>,
    ) -> Result<HpProlateResult> {
        let directory = if matches!(
            cache_route,
            ProlateCacheRoute::Standalone(CacheMode::JsonZip)
        ) {
            prolate_cache_dir()
        } else {
            None
        };
        super::legendre::hp::compute(
            lambda,
            budget,
            n_sample,
            prec,
            |d, b, p| match &cache_route {
                ProlateCacheRoute::Fabric(cache) => Ok(prolate_spectrum_via_cache_model(
                    lambda,
                    d.len(),
                    p,
                    d,
                    b,
                    cache,
                    true,
                )?),
                ProlateCacheRoute::Standalone(_) => exact_standalone_prolate_spectrum_model(
                    lambda,
                    d.len(),
                    p,
                    d,
                    b,
                    directory.as_deref(),
                    true,
                ),
            },
        )
    }

    #[allow(unknown_lints, clippy::manual_is_multiple_of)]
    fn compute_k_lambda_inner(
        lambda: &Float,
        n_grid: usize,
        n_sample: usize,
        prec: u32,
        cache_route: ProlateCacheRoute<'_>,
    ) -> Result<HpProlateResult> {
        let start = std::time::Instant::now();

        if !lambda.is_finite()
            || lambda <= &1
            || !(32..=1_000_000).contains(&prec)
            || n_grid >= u32::MAX as usize
            || n_sample < 2
            || n_sample > u32::MAX as usize
        {
            anyhow::bail!("prolate requires finite positive lambda, valid precision, grid < u32::MAX, and at least two samples");
        }
        let source_lambda = lambda;
        let working_lambda = Float::with_val(prec, lambda);
        let lambda = &working_lambda;
        anyhow::ensure!(
            lambda > &1,
            "prolate lambda must remain greater than one at working precision"
        );
        let cutoff = lambda.clone().square();
        anyhow::ensure!(
            cutoff.is_finite() && cutoff < (Float::with_val(prec, 1) << 64u32),
            "prolate finite-sum cutoff exceeds u64 range"
        );
        // Grid forced odd; the guard above also protects N+1 and u32 casts.
        let n = if n_grid % 2 == 0 { n_grid + 1 } else { n_grid };
        if n < 16 {
            anyhow::bail!("n_grid too small (got {}); need at least 16 to find h_4", n);
        }

        let terms = cutoff
            .clone()
            .ceil()
            .to_integer()
            .and_then(|x| x.to_usize())
            .ok_or_else(|| anyhow::anyhow!("prolate finite-sum work budget exceeded"))?;
        super::legendre::resource_budget(n, n_sample, terms, prec, false)?;

        xc_core::progress_message!(
            "[HP prolate] computing k_λ at λ²={}, N={}, n_sample={}, prec={} bits",
            {
                let mut sq = lambda.clone();
                sq *= lambda;
                xc_numerics::fmt::display_hp(&sq, 7)
            },
            n,
            n_sample,
            prec
        );

        // Build the tridiagonal in HP.
        xc_core::progress_message!("[HP prolate] building tridiagonal PW_λ on N={} grid...", n);
        let pw_start = std::time::Instant::now();
        let (diag, off_diag) = try_build_pw_matrix(lambda, n, prec)?;
        xc_core::progress_message!(
            "[HP prolate] PW_λ built in {:.1}s",
            pw_start.elapsed().as_secs_f64()
        );

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
        // used only when λ² is exactly integer-valued. Other cutoffs use an
        // exact-source zip identity, including rounded square roots of integers.
        let cache_key = if source_lambda.is_integer() {
            source_lambda.to_integer().and_then(|value| {
                let square = value.square();
                square
                    .to_u64()
                    .and_then(|value| lambda_sq_int_for_key(&Float::with_val(64, value)))
            })
        } else {
            None
        };

        let eigenvalues: Vec<Float> = if let ProlateCacheRoute::Fabric(cache) = &cache_route {
            // Exact source keys support every finite cutoff and enforce RequireReuse
            // even when lambda squared is not an integer.
            prolate_spectrum_via_cache(source_lambda, n, prec, &diag, &off_diag, cache)?
        } else if let Some(lambda_sq_int) = cache_key {
            if let ProlateCacheRoute::Standalone(mode) = cache_route {
                if let Some(cached) = load_prolate_eigvals_cache(lambda_sq_int, n, prec, mode)
                    .filter(|values| {
                        validate_fd_selected_spectrum(&diag, &off_diag, values, prec).is_ok()
                    })
                {
                    xc_core::progress_message!(
                        "[HP prolate] loaded {} cached eigenvalues for λ²={}, N={}, prec={} bits",
                        cached.len(),
                        lambda_sq_int.value_f64,
                        n,
                        prec
                    );
                    cached
                } else {
                    xc_core::progress_message!(
                        "[HP prolate] computing all {} eigenvalues of PW_λ via tridiag QR...",
                        n
                    );
                    let eig_start = std::time::Instant::now();
                    let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec)?;
                    xc_core::progress_message!(
                        "[HP prolate] {} eigenvalues computed in {:.1}s",
                        evals.len(),
                        eig_start.elapsed().as_secs_f64()
                    );
                    save_prolate_eigvals_cache(lambda_sq_int, n, prec, &evals, mode);
                    evals
                }
            } else {
                unreachable!("prolate cache route was exhaustively matched")
            }
        } else {
            let directory = if matches!(
                cache_route,
                ProlateCacheRoute::Standalone(CacheMode::JsonZip)
            ) {
                prolate_cache_dir()
            } else {
                None
            };
            exact_standalone_prolate_spectrum(
                source_lambda,
                n,
                prec,
                &diag,
                &off_diag,
                directory.as_deref(),
            )?
        };
        if eigenvalues.len() != n {
            anyhow::bail!(
                "eigensolver returned {} eigenvalues, expected {}",
                eigenvalues.len(),
                n
            );
        }
        validate_fd_selected_spectrum(&diag, &off_diag, &eigenvalues, prec)?;

        // Search the lowest-lying eigenfunctions for h_0 and h_4.
        // Limit search depth: prolate h_4 is the third even eigenfunction.
        let n_try = super::PROLATE_SEARCH_DEPTH.min(n);
        xc_core::progress_message!("[HP prolate] searching for h_0 (even, 0 nodes) and h_4 (even, 4 nodes) in first {} eigenvectors...", n_try);
        let search_start = std::time::Instant::now();
        let mut h0_idx: Option<usize> = None;
        let mut h4_idx: Option<usize> = None;
        let mut h0_vec: Option<Vec<Float>> = None;
        let mut h4_vec: Option<Vec<Float>> = None;
        let mut h0_value = None;
        let mut h4_value = None;

        for (k, lambda_k) in eigenvalues.iter().enumerate().take(n_try) {
            xc_core::progress_message!(
                "[HP prolate] eigenvector {}/{} (eigenvalue {})...",
                k + 1,
                n_try,
                xc_numerics::fmt::display_hp(lambda_k, 8)
            );
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
            let recovery = match xc_numerics::eigen::tridiag_eigenvector_for_value_detailed_hp(
                &diag,
                &off_diag,
                lambda_k,
                None,
                prec,
                TridiagEigvecOptions {
                    solver: xc_numerics::eigen::TridiagSolver::BandedInterleaved,
                    ..TridiagEigvecOptions::default()
                },
            ) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let v = recovery.eigenvector;
            // Check parity.
            let parity = parity_of(&v, prec);
            let is_even = matches!(parity, HpParity::Even);
            if !is_even {
                continue;
            }
            // Count nodes.
            let nodes = try_count_nodes(&v, prec)?;
            match nodes {
                0 if h0_idx.is_none() => {
                    xc_core::progress_message!("[HP prolate] found h_0 at index {}", k);
                    h0_idx = Some(k);
                    h0_vec = Some(v);
                    h0_value = Some(recovery.eigenvalue);
                }
                4 if h4_idx.is_none() => {
                    xc_core::progress_message!("[HP prolate] found h_4 at index {}", k);
                    h4_idx = Some(k);
                    h4_vec = Some(v);
                    h4_value = Some(recovery.eigenvalue);
                }
                _ => {}
            }
            if h0_idx.is_some() && h4_idx.is_some() {
                break;
            }
        }
        xc_core::progress_message!(
            "[HP prolate] eigenvector search done in {:.1}s",
            search_start.elapsed().as_secs_f64()
        );
        let h0_idx = h0_idx.ok_or_else(|| {
            anyhow::anyhow!(
                "could not find h_{{0,λ}} (even, 0 nodes) in first {} eigenfunctions",
                n_try
            )
        })?;
        let h4_idx = h4_idx.ok_or_else(|| {
            anyhow::anyhow!(
                "could not find h_{{4,λ}} (even, 4 nodes) in first {} eigenfunctions",
                n_try
            )
        })?;
        anyhow::ensure!(
            h0_idx == 0 && h4_idx == 4,
            "prolate parity/node classification disagrees with ordered modes 0 and 4"
        );
        let mut h0_vec = h0_vec.unwrap();
        let mut h4_vec = h4_vec.unwrap();
        let eigenvalue_0 = h0_value.expect("selected ground Rayleigh value");
        let eigenvalue_4 = h4_value.expect("selected fourth Rayleigh value");

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
                v.neg_assign();
            }
        }
        if h4_vec[center] < zero {
            for v in h4_vec.iter_mut() {
                v.neg_assign();
            }
        }

        // Trapezoidal integral of the piecewise-linear Dirichlet interpolant:
        // exactly h * sum(v_i), before floating-point rounding.
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
        let h_lambda: Vec<Float> = (0..n)
            .map(|i| {
                let mut t = c_4.clone();
                t *= &h4_vec[i];
                let mut t2 = c_0.clone();
                t2 *= &h0_vec[i];
                t += &t2;
                t
            })
            .collect();

        super::grid_contract::validate_grid(&h_lambda, lambda, &h, prec)?;

        // Build logarithmic u-grid in [λ⁻¹, λ].
        let n_sample = n_sample.max(2);
        let log_lambda = lambda.clone().ln();
        let u_grid: Vec<Float> = (0..n_sample)
            .map(|i| {
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
            })
            .collect();

        // Evaluate k_λ(u) = √u · Σ_{n=1}^{⌊λ/u⌋} h_λ(n·u).
        // Each grid point u is independent → parallelize via par_iter.
        xc_core::progress_message!(
            "[HP prolate] sampling k_λ on {} log-spaced grid points...",
            n_sample
        );
        let sample_start = std::time::Instant::now();
        let k_values: Vec<Float> = u_grid
            .par_iter()
            .map(|u| {
                let mut ratio = lambda.clone();
                ratio /= u;
                let ratio_floor = ratio.floor();
                let n_terms = ratio_floor
                    .to_integer()
                    .and_then(|value| value.to_u64())
                    .ok_or_else(|| {
                        anyhow::anyhow!("sampled finite-sum cutoff exceeds u64 range")
                    })?;

                let mut s = Float::with_val(prec, 0);
                for k in 1..=n_terms {
                    let mut x = u.clone();
                    let k_hp = Float::with_val(prec, k);
                    x *= &k_hp;
                    if x >= *lambda {
                        break;
                    }
                    s += super::grid_contract::interpolate(&h_lambda, lambda, &x, prec);
                }
                // u^(1/2)
                let sqrt_u = u.clone().sqrt();
                let mut result = sqrt_u;
                result *= &s;
                anyhow::ensure!(result.is_finite(), "prolate sample is outside MPFR range");
                Ok(result)
            })
            .collect::<Result<_>>()?;
        xc_core::progress_message!(
            "[HP prolate] k_λ sampling done in {:.1}s; total compute_k_lambda elapsed {:.1}s",
            sample_start.elapsed().as_secs_f64(),
            start.elapsed().as_secs_f64()
        );

        Ok(HpProlateResult {
            discretization: "prolate-finite-dirichlet-recovered-rayleigh-v3".to_owned(),
            resolution_budget: n_grid,
            basis_dimension: n,
            relative_operator_residual: None,
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
        /// Discrete ℓ² norm of `ξ_λ` on the grid, matching `l2_error`'s
        /// unweighted convention (HP).
        ///
        /// Because `optimal_scalar` minimizes `‖ξ_λ − c·k_λ‖₂`, the residual is
        /// orthogonal to `k_λ` and `l2_error / xi_l2` is exactly `sin θ`, the
        /// scale-free angle between the educated guess and its target.
        pub xi_l2: Float,
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
        super::comparison::compare_hp(xi, n_modes, lambda, u_grid, k_values, prec)
    }

    // -----------------------------------------------------------------------
    // HP unit tests
    // -----------------------------------------------------------------------

    /// End-to-end CCM -> prolate accuracy measurement.
    ///
    /// This samples the additional prolate/Weil bridge, not Lemma 7.2: how closely the prolate
    /// educated guess `k_λ` approximates the true smallest-eigenvalue Weil
    /// eigenvector `ξ_λ` at mode cutoff `N`.
    #[derive(Clone, Debug)]
    pub struct CcmProlateDistanceHp {
        pub n_grid: usize,
        pub n_sample: usize,
        pub precision_bits: u32,
        /// Precision used for the finite source, samples, and comparison before rounding.
        pub working_precision_bits: u32,
        /// Identity of the guarded finite-source comparison policy.
        pub algorithm_semantics: String,
        pub prolate_discretization: String,
        pub prolate_basis_dimension: usize,
        /// `λ²` the measurement was taken at.
        pub lambda_squared: f64,
        /// Mode cutoff `N`. The expanded `ξ_λ` has `2N+1` coefficients.
        pub n_modes: usize,
        /// Smallest even-sector Weil eigenvalue at this `(λ², N)`.
        pub eigenvalue: Float,
        /// Norms of the residual `ξ_λ − c·k_λ` on the prolate sample grid.
        pub comparison: HpComparisonResult,
        /// `‖ξ_λ − c·k_λ‖₂ / ‖ξ_λ‖₂`, that is `sin θ` between guess and target.
        ///
        /// Scale-free, so it is comparable across `λ²` and independent of the
        /// normalization the sector eigensolver happened to return.
        pub relative_l2_distance: Float,
    }

    /// Measure how accurately the prolate educated guess `k_λ` approximates the
    /// CCM Weil ground state `ξ_λ`, end to end.
    ///
    /// Computes the even-parity Weil sector ground state for `params`, expands
    /// it out of the sector basis into the full `2N+1` `V_n` layout via
    /// [`crate::ccm::hp::expand_even_sector_vector`], builds `k_λ` on the
    /// prolate grid, and compares the two.
    ///
    /// The even sector is the relevant one: `k_λ = ℰ(h_λ)` is assembled from the
    /// even prolate modes `h_0` and `h_4`, and the comparison reconstructs
    /// `ξ_λ` through an even cosine sum.
    ///
    /// This is a finite-`N`, finite-precision measurement. It does not on its
    /// own establish anything about the `N → ∞` limit.
    /// The finite source and comparison use 64 guard bits above the requested
    /// output precision; report fields are rounded once to that output precision.
    /// Guarded computation does not certify continuum or forward accuracy.
    pub fn ccm_prolate_distance_hp(
        params: &crate::ccm::CcmParams,
        cfg: &crate::ccm::hp::HighPrecConfig,
        n_grid: usize,
        n_sample: usize,
        mode: CacheMode,
    ) -> Result<CcmProlateDistanceHp> {
        let output_precision = cfg.precision_bits;
        let prec = output_precision
            .checked_add(64)
            .ok_or_else(|| anyhow::anyhow!("prolate comparison guard precision overflow"))?;
        let mut working_cfg = cfg.clone();
        working_cfg.precision_bits = prec;
        let lambda_sq = crate::ccm::hp::lambda_squared_value_hp(params, prec)?;
        if lambda_sq <= 1u32 {
            anyhow::bail!(
                "prolate comparison requires a nondegenerate interval with lambda squared > 1"
            );
        }
        let lambda = lambda_sq.sqrt();
        let gap = crate::ccm::hp::analyze_sector_gap(
            params,
            &working_cfg,
            crate::ccm::hp::MINIMUM_SECTOR_EIGENPAIRS,
        )?;
        let ground = gap
            .even
            .eigenpairs
            .first()
            .ok_or_else(|| anyhow::anyhow!("CCM even sector returned no eigenpairs"))?;
        let expected = params.n_modes + 1;
        if ground.eigenvector.len() != expected {
            anyhow::bail!(
                "CCM even sector eigenvector has dimension {}, expected {expected}",
                ground.eigenvector.len()
            );
        }
        let xi =
            crate::ccm::hp::expand_even_sector_vector(&ground.eigenvector, params.n_modes, prec);
        let k = compute_k_lambda(&lambda, n_grid, n_sample, prec, mode)?;
        let mut comparison =
            compare_xi_to_k_lambda(&xi, params.n_modes, &lambda, &k.u_grid, &k.k_values, prec)?;
        if comparison.xi_l2 == 0u32 {
            anyhow::bail!(
                "Weil state vanishes on the prolate sample grid; relative distance undefined"
            );
        }
        let relative_l2_distance = Float::with_val(
            output_precision,
            comparison.l2_error.clone() / comparison.xi_l2.clone(),
        );
        for value in [
            &mut comparison.optimal_scalar,
            &mut comparison.linf_error,
            &mut comparison.l2_error,
            &mut comparison.xi_linf,
            &mut comparison.xi_l2,
        ] {
            value.set_prec(output_precision);
        }
        Ok(CcmProlateDistanceHp {
            n_grid,
            n_sample,
            precision_bits: output_precision,
            working_precision_bits: prec,
            algorithm_semantics: "ccm_prolate_guarded_source_and_comparison_v3".into(),
            prolate_discretization: k.discretization,
            prolate_basis_dimension: k.basis_dimension,
            lambda_squared: params.lambda_squared(),
            n_modes: params.n_modes,
            eigenvalue: Float::with_val(output_precision, &ground.eigenvalue),
            comparison,
            relative_l2_distance,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use xc_numerics::fmt::display_hp;

        #[test]
        fn fd_cache_recomputes_sorted_wrong_spectrum() {
            let root_dir = xc_core::test_support::TestDir::new("remaining-fd-cache");
            let root = root_dir.to_path_buf();
            let p = 128;
            let n = 25;
            let lambda = Float::with_val(p, 2).sqrt();
            let (d, b) = build_pw_matrix(&lambda, n, p);
            let original =
                exact_standalone_prolate_spectrum_model(&lambda, n, p, &d, &b, Some(&root), false)
                    .unwrap();
            assert_eq!(
                original,
                exact_standalone_prolate_spectrum_model(&lambda, n, p, &d, &b, Some(&root), false)
                    .unwrap()
            );
            let path = std::fs::read_dir(&root)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
            let wrong = PortableProlateSpectrum {
                schema_version: 3,
                lambda: lambda.to_string(),
                lambda_precision_bits: lambda.prec(),
                grid_points: n,
                precision_bits: p,
                eigenvalues: original
                    .iter()
                    .map(|x| (Float::with_val(p, x) + 1i32).to_string())
                    .collect(),
            };
            let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
            writer
                .start_file(&name, zip::write::SimpleFileOptions::default())
                .unwrap();
            std::io::Write::write_all(&mut writer, &serde_json::to_vec(&wrong).unwrap()).unwrap();
            let bytes = writer.finish().unwrap().into_inner();
            xc_cache::atomic_replace_cache_file(&path, &bytes).unwrap();
            assert_eq!(
                original,
                exact_standalone_prolate_spectrum_model(&lambda, n, p, &d, &b, Some(&root), false)
                    .unwrap()
            );
            std::fs::remove_dir_all(root).unwrap();
        }

        #[test]
        fn legendre_cache_rejects_sorted_wrong_spectrum() {
            let root_dir = xc_core::test_support::TestDir::new("confirmed-legendre-cache");
            let root = root_dir.to_path_buf();
            let p = 128;
            let n = 24;
            let lambda = Float::with_val(p, 2).sqrt();
            let (d, b) = super::super::legendre::hp::block(&lambda, n, p);
            let original = exact_standalone_prolate_spectrum_model(
                &lambda,
                n,
                p,
                &d,
                &b[..n - 1],
                Some(&root),
                true,
            )
            .unwrap();
            assert_eq!(
                original,
                exact_standalone_prolate_spectrum_model(
                    &lambda,
                    n,
                    p,
                    &d,
                    &b[..n - 1],
                    Some(&root),
                    true
                )
                .unwrap()
            );
            let path = std::fs::read_dir(&root)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
            let wrong = PortableProlateSpectrum {
                schema_version: 3,
                lambda: lambda.to_string(),
                lambda_precision_bits: lambda.prec(),
                grid_points: n,
                precision_bits: p,
                eigenvalues: original
                    .iter()
                    .map(|x| (Float::with_val(p, x) + 1i32).to_string())
                    .collect(),
            };
            let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
            writer
                .start_file(&name, zip::write::SimpleFileOptions::default())
                .unwrap();
            std::io::Write::write_all(&mut writer, &serde_json::to_vec(&wrong).unwrap()).unwrap();
            let bytes = writer.finish().unwrap().into_inner();
            xc_cache::atomic_replace_cache_file(&path, &bytes).unwrap();
            assert_eq!(
                original,
                exact_standalone_prolate_spectrum_model(
                    &lambda,
                    n,
                    p,
                    &d,
                    &b[..n - 1],
                    Some(&root),
                    true
                )
                .unwrap()
            );
            std::fs::remove_dir_all(root).unwrap();
        }

        #[test]
        fn exact_standalone_cache_reuses_noninteger_sources_and_never_aliases_nearby_cutoffs() {
            let root_dir = xc_core::test_support::TestDir::new("exact-prolate");
            let root = root_dir.to_path_buf();
            let p = 128;
            let lambda = Float::with_val(p, Float::parse("3.123456789").unwrap());
            let n = 17;
            let (d, e) = build_pw_matrix(&lambda, n, p);
            let cold =
                exact_standalone_prolate_spectrum(&lambda, n, p, &d, &e, Some(&root)).unwrap();
            // Empty solver inputs prove the second call gets its spectrum from disk.
            let warm =
                exact_standalone_prolate_spectrum(&lambda, n, p, &d, &e, Some(&root)).unwrap();
            assert_eq!(cold, warm);
            assert!(exact_standalone_prolate_spectrum(&lambda, n, p, &[], &[], None).is_err());
            let near = Float::with_val(p, Float::parse("3.123456789000000000001").unwrap());
            assert!(exact_standalone_prolate_spectrum(&near, n, p, &[], &[], Some(&root)).is_err());
            let path = std::fs::read_dir(&root)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            std::fs::write(&path, b"broken zip").unwrap();
            assert!(
                exact_standalone_prolate_spectrum(&lambda, n, p, &[], &[], Some(&root)).is_err()
            );
            assert_eq!(
                cold,
                exact_standalone_prolate_spectrum(&lambda, n, p, &d, &e, Some(&root)).unwrap()
            );
            std::fs::remove_dir_all(root).unwrap();
        }

        #[test]
        fn prolate_spectrum_round_trips_through_common_cache_fabric() {
            use xc_cache::{
                ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheQuality, CacheResolver,
                CacheVisibility, CertificationFailurePolicy, FilesystemCacheStore,
                ManagedArtifactCacheConfig, ManagedArtifactCacheSession, ManagedRemoteCacheMode,
                ManagedRunProfile, OutputPreservationValidationReport, OutputValidationConfig,
            };

            let root_dir = xc_core::test_support::TestDir::new("prolate-fabric");
            let root = root_dir.to_path_buf();
            let reference_root = root.join("reference");
            let resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(FilesystemCacheStore::new(
                    "workstation",
                    &reference_root,
                    true,
                    CacheVisibility::Local,
                )),
            }]);
            let policy = CachePolicy {
                current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
                minimum_quality: CacheQuality::Validated,
                accepted_schema_versions: vec![1],
                allow_deprecated: false,
                allow_quarantined: false,
                allowed_visibilities: vec![CacheVisibility::Local],
            };
            let context = ArtifactCacheContext {
                resolver: Some(&resolver),
                reference_resolver: None,
                acceptance: Some(&policy),
                ordered_overlays: vec!["workstation".to_owned()],
                mode: ArtifactExecutionCacheMode::PreferReuse,
                write_on_miss: true,
                write_visibility: CacheVisibility::Local,
                requested_assurance: xc_core::AssuranceLevel::Computed,
                certification_failure_policy:
                    xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
                production_sink: None,
            };
            let precision = 128;
            let lambda = Float::with_val(precision, 2);
            let (diagonal, off_diagonal) = build_pw_matrix(&lambda, 15, precision);
            let key = &lambda;
            let first =
                prolate_spectrum_via_cache(key, 15, precision, &diagonal, &off_diagonal, &context)
                    .unwrap();
            let second =
                prolate_spectrum_via_cache(key, 15, precision, &diagonal, &off_diagonal, &context)
                    .unwrap();
            assert_eq!(first, second);
            let required = ArtifactCacheContext {
                mode: ArtifactExecutionCacheMode::RequireReuse,
                write_on_miss: false,
                ..context
            };
            let mut nearby = lambda.clone();
            nearby.next_up();
            assert!(prolate_spectrum_via_cache(
                &nearby,
                15,
                precision,
                &diagonal,
                &off_diagonal,
                &required
            )
            .is_err());
            assert_eq!(
                prolate_spectrum_via_cache(key, 15, precision, &diagonal, &off_diagonal, &required)
                    .unwrap(),
                first
            );

            let validation_root = root.join("validation");
            let session = ManagedArtifactCacheSession::with_layers_for_test(
                ManagedArtifactCacheConfig {
                    profile: ManagedRunProfile::Normal,
                    requested_assurance: xc_core::AssuranceLevel::Computed,
                    certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
                    cache_root: root.join("production"),
                    staging_root: None,
                    publication_target: xc_core::PublicationTarget::None,
                    repository_owner: "local-fixture".to_owned(),
                    remote_cache_mode: ManagedRemoteCacheMode::None,
                    cache_mode: ArtifactExecutionCacheMode::VerifyAgainstReference,
                    replace_existing_publication: false,
                    execute_remote_mutations: false,
                    output_validation: Some(OutputValidationConfig {
                        validation_root: validation_root.clone(),
                        report_root: validation_root.join("reports"),
                        reference_mode: ManagedRemoteCacheMode::Public,
                    }),
                },
                vec![CacheLayer {
                    precedence: 0,
                    store: Box::new(FilesystemCacheStore::new(
                        "validation-computed",
                        validation_root.join("computed"),
                        true,
                        CacheVisibility::Local,
                    )),
                }],
                vec![CacheLayer {
                    precedence: 0,
                    store: Box::new(FilesystemCacheStore::new(
                        "reference",
                        &reference_root,
                        false,
                        CacheVisibility::Local,
                    )),
                }],
            )
            .unwrap();
            let verified = prolate_spectrum_via_cache(
                key,
                15,
                precision,
                &diagonal,
                &off_diagonal,
                &session.context(),
            )
            .unwrap();
            assert_eq!(verified, first);
            session.finalize_publication_inventory().unwrap();
            let report: OutputPreservationValidationReport = serde_json::from_slice(
                &std::fs::read(validation_root.join("reports/latest.json")).unwrap(),
            )
            .unwrap();
            assert!(report.output_preserving);
            assert_eq!(report.totals.matched, 1);
            let _ = std::fs::remove_dir_all(root);
        }

        #[test]
        fn legacy_cache_requires_exact_identity_precision_and_arithmetic_stamp() {
            let lsq = LambdaSq::integer(25);
            let valid = serde_json::json!({
                "schema_version": 1,
                "toolkit_version": prolate_toolkit_version_for_test(),
                "arithmetic_semantics": "prolate-fd-working-precision-v2",
            "qr_arithmetic": xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
                "lambda_sq_mode": lsq.mode_str(),
                "lambda_sq_identity": lsq.filename_str(),
                "n_grid": 3,
                "precision_bits": 256,
                "eigenvalues": ["1","2","3"]
            });
            assert!(parse_prolate_cache_json(&valid.to_string(), lsq, 3, 256).is_some());
            for (field, bad) in [
                ("schema_version", serde_json::json!(2)),
                ("lambda_sq_identity", serde_json::json!("26")),
                ("lambda_sq_mode", serde_json::json!("fractional")),
                ("n_grid", serde_json::json!(5)),
                ("precision_bits", serde_json::json!(53)),
                ("arithmetic_semantics", serde_json::json!("old")),
            ] {
                let mut altered = valid.clone();
                altered[field] = bad;
                assert!(
                    parse_prolate_cache_json(&altered.to_string(), lsq, 3, 256).is_none(),
                    "{field}"
                );
                altered.as_object_mut().unwrap().remove(field);
                assert!(
                    parse_prolate_cache_json(&altered.to_string(), lsq, 3, 256).is_none(),
                    "missing {field}"
                );
            }
        }

        fn hp(prec: u32, s: &str) -> Float {
            Float::with_val(prec, Float::parse(s).unwrap())
        }

        fn dense_tridiagonal(diagonal: &[Float], off_diagonal: &[Float], prec: u32) -> Vec<Float> {
            let n = diagonal.len();
            let mut dense = vec![Float::with_val(prec, 0); n * n];
            for index in 0..n {
                dense[index * n + index] = diagonal[index].clone();
                if index + 1 < n {
                    dense[index * n + index + 1] = off_diagonal[index].clone();
                    dense[(index + 1) * n + index] = off_diagonal[index].clone();
                }
            }
            dense
        }

        #[test]
        fn structured_prolate_has_two_independent_hp_routes_and_precision_repeat() {
            let solve = |prec| {
                let lambda = hp(prec, "2");
                let (diagonal, off_diagonal) = build_pw_matrix(&lambda, 15, prec);
                let qr = tridiag_eigenvalues_hp(&diagonal, &off_diagonal, prec).unwrap();
                let dense = dense_tridiagonal(&diagonal, &off_diagonal, prec);
                let jacobi =
                    xc_numerics::eigen::dense_symmetric_eigenvalues_jacobi_hp(&dense, 15, prec, 80)
                        .unwrap();
                (qr, jacobi.eigenvalues)
            };
            let (qr_low, jacobi_low) = solve(192);
            let (qr_high, jacobi_high) = solve(320);
            let route_tolerance = hp(320, "1e-45");
            for (qr, jacobi) in qr_high.iter().zip(&jacobi_high) {
                let mut difference = qr.clone();
                difference -= jacobi;
                assert!(difference.abs() < route_tolerance);
            }
            let repeat_tolerance = hp(320, "1e-40");
            for (low, high) in qr_low.iter().zip(&qr_high) {
                let mut difference = Float::with_val(320, low);
                difference -= high;
                assert!(difference.abs() < repeat_tolerance);
            }
            for (low, high) in jacobi_low.iter().zip(&jacobi_high) {
                let mut difference = Float::with_val(320, low);
                difference -= high;
                assert!(difference.abs() < repeat_tolerance);
            }
        }

        #[test]
        fn trial_subspace_stopping_matches_precision_and_independent_quadratic() {
            use xc_core::{EigenTarget, ResultStatus};
            for p in [40u32, 64, 100, 128] {
                for (a, b, c) in [(2i32, 1i32, 5i32), (-3, 2, 7)] {
                    let forms = ProlateSubspaceFormsHp {
                        ambient_dimension: 7,
                        basis_dimension: 2,
                        precision_bits: p,
                        stiffness: [a, b, b, c]
                            .into_iter()
                            .map(|x| Float::with_val(p, x))
                            .collect(),
                        gram: [1, 0, 0, 1]
                            .into_iter()
                            .map(|x| Float::with_val(p, x))
                            .collect(),
                    };
                    // Independent characteristic polynomial of this exact 2x2 matrix.
                    let discriminant =
                        Float::with_val(p + 64, (a - c) * (a - c) + 4 * b * b).sqrt();
                    for (target, sign) in [
                        (EigenTarget::AlgebraicSmallest, -1),
                        (EigenTarget::AlgebraicLargest, 1),
                    ] {
                        let expected = (Float::with_val(p + 64, a + c)
                            + Float::with_val(p + 64, &discriminant * sign))
                            / 2u32;
                        let report = solve_pw_subspace_extreme(&forms, target).unwrap();
                        assert_eq!(report.status, ResultStatus::Converged, "p={p}");
                        assert!(report
                            .algorithm
                            .contains("prolate_precision_scaled_ritz_stopping_v2"));
                        let tolerance = Float::with_val(p + 64, 1) >> (p / 2);
                        assert!(
                            Float::with_val(p + 64, &report.eigenvalue - expected).abs()
                                < tolerance
                        );
                    }
                }
            }
        }

        #[test]
        fn finite_grid_cache_does_not_require_continuum_asymptotics() {
            for (lambda, n, p) in [
                (10u32, 17usize, 96u32),
                (10, 33, 96),
                (7, 19, 164),
                (13, 23, 128),
            ] {
                let lambda = Float::with_val(p, lambda);
                let (d, b) = try_build_pw_matrix(&lambda, n, p).unwrap();
                let values = tridiag_eigenvalues_hp(&d, &b, p).unwrap();
                let cutoff = LambdaSq::integer(
                    lambda.to_u32_saturating().unwrap() as u64
                        * lambda.to_u32_saturating().unwrap() as u64,
                );
                let rejection = prolate_cache_structural_check(&values, n, cutoff, p);
                assert!(
                    rejection.is_none(),
                    "lambda={lambda}, n={n}, p={p}: {rejection:?}"
                );
                // Independent bisection of the exact stored matrix, rather
                // than the QR producer, checks every ordered value.
                let q = p + 64;
                let tolerance = Float::with_val(q, 1) >> (p + 8);
                let independent = xc_numerics::eigen::tridiag_selected_eigenvalues_hp(
                    &d,
                    &b,
                    0,
                    n - 1,
                    &tolerance,
                    (p + 160) as usize,
                    q,
                )
                .unwrap();
                let source_norm = (0..n)
                    .map(|i| {
                        let mut row = Float::with_val(q, &d[i]).abs();
                        if i > 0 {
                            row += Float::with_val(q, &b[i - 1]).abs();
                        }
                        if i + 1 < n {
                            row += Float::with_val(q, &b[i]).abs();
                        }
                        row
                    })
                    .max_by(Float::total_cmp)
                    .unwrap();
                // Compare against the documented normwise source-precision
                // radius, using bisection rather than QR as the value oracle.
                let padding = (source_norm * (8 * n)) >> p;
                for (value, bracket) in values.iter().zip(&independent.enclosures) {
                    assert!(
                        Float::with_val(q, value) >= Float::with_val(q, &bracket.lower - &padding)
                    );
                    assert!(
                        Float::with_val(q, value) <= Float::with_val(q, &bracket.upper + &padding)
                    );
                }
                let mut bad = values.clone();
                bad[0] += Float::with_val(p, 1) >> (p / 2);
                assert!(prolate_cache_structural_check(&bad, n, cutoff, p).is_some());
            }
        }

        #[test]
        fn cache_enclosures_allow_exact_and_subscale_multiplicity() {
            let p = 192;
            for second in [
                Float::with_val(256, 1),
                Float::with_val(256, 1) + (Float::with_val(256, 1) >> 190),
            ] {
                let d = vec![Float::with_val(256, 1), second];
                let b = vec![Float::with_val(256, 0)];
                let values = d.iter().map(|v| Float::with_val(p, v)).collect::<Vec<_>>();
                // The diagonal entries are the exact, independent eigenvalues.
                super::super::legendre::hp::validate_spectrum(&d, &b, &values, p).unwrap();
                let mut wrong = values.clone();
                wrong[1] += Float::with_val(p, 1) >> 160;
                assert!(super::super::legendre::hp::validate_spectrum(&d, &b, &wrong, p).is_err());
            }
        }

        #[test]
        fn nonorthogonal_prolate_subspace_requests_both_generalized_extremes() {
            use xc_core::{EigenTarget, ResultStatus};

            let precision = 192;
            let lambda = Float::with_val(precision, 2);
            let basis = vec![
                (0..7)
                    .map(|row| Float::with_val(precision, 1 + row))
                    .collect::<Vec<_>>(),
                (0..7)
                    .map(|row| Float::with_val(precision, 2 + 2 * row))
                    .enumerate()
                    .map(|(row, mut value)| {
                        if row == 3 {
                            value += 1;
                        }
                        value
                    })
                    .collect::<Vec<_>>(),
                (0..7)
                    .map(|row| Float::with_val(precision, 1 + row * row))
                    .collect::<Vec<_>>(),
            ];
            let forms = build_pw_subspace_forms(&lambda, 7, &basis, precision).unwrap();
            assert_eq!(forms.ambient_dimension, 7);
            assert_eq!(forms.basis_dimension, 3);
            assert!(!forms.gram[1].is_zero(), "basis must be nonorthogonal");

            let smallest =
                solve_pw_subspace_extreme(&forms, EigenTarget::AlgebraicSmallest).unwrap();
            let largest = solve_pw_subspace_extreme(&forms, EigenTarget::AlgebraicLargest).unwrap();
            assert_eq!(smallest.status, ResultStatus::Converged);
            assert_eq!(largest.status, ResultStatus::Converged);
            assert!(smallest.eigenvalue < largest.eigenvalue);
            let tolerance = Float::with_val(precision, Float::parse("1e-30").unwrap());
            assert!(smallest.residual_norm <= tolerance);
            assert!(largest.residual_norm <= tolerance);
            assert!(smallest.metric_normalization_error <= tolerance);
            assert!(largest.metric_normalization_error <= tolerance);
        }

        /// Build PW_λ in HP and confirm the diagonal is sane (positive,
        /// finite, bounded).
        #[test]
        #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
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
            assert!(
                diag[center] > lo && diag[center] < hi,
                "diag[center] = {} should be in [1000, 100000]",
                display_hp(&diag[center], 6)
            );
        }

        /// Smallest prolate eigenvalue at λ=5 should be close to 2π·25 = 157.08.
        #[test]
        #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
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
            assert!(
                abs_diff < tol,
                "smallest eigenvalue {} should be near 2π·25 = {} (diff {})",
                display_hp(&evals[0], 6),
                display_hp(&expected, 6),
                display_hp(&abs_diff, 4)
            );
        }

        /// End-to-end compute_k_lambda at HP. Verify result is finite,
        /// has expected shape, and eigenvalues are sane.
        #[test]
        #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
        fn compute_k_lambda_runs_hp() {
            let prec = 256;
            let lambda = hp(prec, "5");
            let res = compute_k_lambda(&lambda, 401, 64, prec, CacheMode::Off).unwrap();
            assert_eq!(res.u_grid.len(), 64);
            assert_eq!(res.k_values.len(), 64);

            // h_0 has eigenvalue ≈ 2πλ² ≈ 157, h_4 ≈ 18πλ² ≈ 1413.
            let zero = Float::with_val(prec, 0);
            assert!(
                res.eigenvalue_0 > zero,
                "eigenvalue_0 should be positive, got {}",
                display_hp(&res.eigenvalue_0, 6)
            );
            assert!(
                res.eigenvalue_4 > res.eigenvalue_0,
                "eigenvalue_4 ({}) should exceed eigenvalue_0 ({})",
                display_hp(&res.eigenvalue_4, 6),
                display_hp(&res.eigenvalue_0, 6)
            );

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
        #[ignore = "HP matrix compute — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
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

            let cmp =
                compare_xi_to_k_lambda(&xi, n_modes, &lambda, &res.u_grid, &res.k_values, prec)
                    .unwrap();
            // All HP results should be finite (no NaN, no infinity).
            assert!(!cmp.linf_error.is_nan() && !cmp.linf_error.is_infinite());
            assert!(!cmp.l2_error.is_nan() && !cmp.l2_error.is_infinite());
            let zero = Float::with_val(prec, 0);
            assert!(
                cmp.xi_linf > zero,
                "xi_linf should be positive, got {}",
                display_hp(&cmp.xi_linf, 6)
            );
        }

        // ---------------------------------------------------------------
        // Prolate eigenvalue cache — pure-function and verify_dir tests
        // ---------------------------------------------------------------

        /// `lambda_sq_int_for_key` accepts integer-valued λ² (within
        /// exact equality) and rejects even nearby non-integer values.
        #[test]
        fn cache_key_accepts_integer_lambda_sq() {
            let prec = 256;
            // Integer-valued λ² → Some(LambdaSq::integer(L))
            assert_eq!(
                lambda_sq_int_for_key(&hp(prec, "13")),
                Some(LambdaSq::integer(13))
            );
            assert_eq!(
                lambda_sq_int_for_key(&hp(prec, "100")),
                Some(LambdaSq::integer(100))
            );
            assert_eq!(
                lambda_sq_int_for_key(&hp(prec, "1000")),
                Some(LambdaSq::integer(1000))
            );
            // Tiny f64 round-trip noise should still parse — but we're
            // building these from exact strings so they're already tight.
            assert_eq!(
                lambda_sq_int_for_key(&hp(prec, "13.0")),
                Some(LambdaSq::integer(13))
            );
            // Non-integer rejected.
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "13.5")), None);
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "13.000000000001")), None);
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "12.999999999999")), None);
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "100.001")), None);
            // Negative or zero rejected.
            assert_eq!(lambda_sq_int_for_key(&hp(prec, "0")), None);
            let mut neg = hp(prec, "13");
            neg = -neg;
            assert_eq!(lambda_sq_int_for_key(&neg), None);
        }

        /// `parse_prolate_cache_filename` parses well-formed names and
        /// rejects others.
        #[test]
        fn cache_filename_parser_extracts_tuple() {
            assert_eq!(
                parse_prolate_cache_filename("lambda_sq13_ngrid4001_prec3338.json"),
                Some((LambdaSq::integer(13), 4001, 3338))
            );
            assert_eq!(
                parse_prolate_cache_filename("lambda_sq1000_ngrid8001_prec3338.json.zip"),
                Some((LambdaSq::integer(1000), 8001, 3338))
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
            let lambda_sq = LambdaSq::integer(25);
            let n = 401;
            let (diag, off_diag) = build_pw_matrix(&lambda, n, prec);
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();
            // Real spectrum must pass.
            assert!(
                prolate_cache_structural_check(&evals, n, lambda_sq, prec).is_none(),
                "real prolate spectrum should pass structural check"
            );

            // Wrong count → reject.
            let mut short = evals.clone();
            short.pop();
            assert!(
                prolate_cache_structural_check(&short, n, lambda_sq, prec).is_some(),
                "wrong count should be rejected"
            );

            // Out-of-order → reject.
            let mut shuffled = evals.clone();
            shuffled.swap(0, 5);
            assert!(
                prolate_cache_structural_check(&shuffled, n, lambda_sq, prec).is_some(),
                "non-ascending order should be rejected"
            );

            // Wrong ground state magnitude → reject. Replace e_0 with
            // a value 50% off from 2π·25.
            let mut off_e0 = evals.clone();
            off_e0[0] = hp(prec, "50"); // way below 2π·25 ≈ 157
            assert!(
                prolate_cache_structural_check(&off_e0, n, lambda_sq, prec).is_some(),
                "ground-state magnitude check should reject 50 vs 2π·25 ≈ 157"
            );

            // NaN entry → reject.
            let mut with_nan = evals.clone();
            with_nan[10] = Float::with_val(prec, f64::NAN);
            assert!(
                prolate_cache_structural_check(&with_nan, n, lambda_sq, prec).is_some(),
                "NaN entry should be rejected"
            );
        }

        /// `verify_prolate_eigvals_cache_dir` on a non-existent directory
        /// returns an empty report, not an error.
        #[test]
        fn verify_dir_handles_missing_directory() {
            let temp_root = crate::fresh_test_dir("prolate-cache-missing");
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
            let lambda_sq = LambdaSq::integer(25);
            let n_grid: usize = 401;

            // Compute a real spectrum once.
            let (diag, off_diag) = build_pw_matrix(&lambda, n_grid, prec);
            let real_evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

            // Build an isolated, self-removing temp dir.
            let temp_dir = crate::fresh_test_dir("prolate-cache-classify");

            // 1. Valid file: serialize the real spectrum as envelope.
            let valid_name = prolate_cache_filename(lambda_sq, n_grid, prec);
            let valid_path = temp_dir.join(&valid_name);
            let strs: Vec<String> = real_evals.iter().map(|f| f.to_string()).collect();
            let valid_json = serde_json::json!({
                "schema_version": 1,
                "toolkit_version": prolate_toolkit_version_for_test(),
            "arithmetic_semantics": "prolate-fd-working-precision-v2",
            "qr_arithmetic": xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
                "lambda_sq": lambda_sq.value_f64,
                "lambda_sq_mode": lambda_sq.mode_str(),
                "lambda_sq_identity": lambda_sq.filename_str(),
                "n_grid": n_grid,
                "precision_bits": prec,
                "eigenvalues": strs,
            });
            std::fs::write(&valid_path, serde_json::to_string(&valid_json).unwrap()).unwrap();

            // 2. Structurally-invalid file: reversed spectrum (not ascending),
            //    wrapped in envelope so parse succeeds but structural check fails.
            let mut bad_evals = real_evals.clone();
            bad_evals.reverse();
            let lsq_bad = LambdaSq::integer(lambda_sq.value_u64 + 1);
            let bad_name = prolate_cache_filename(lsq_bad, n_grid, prec);
            let bad_path = temp_dir.join(&bad_name);
            let bad_strs: Vec<String> = bad_evals.iter().map(|f| f.to_string()).collect();
            let bad_json = serde_json::json!({
                "schema_version": 1,
                "toolkit_version": prolate_toolkit_version_for_test(),
            "arithmetic_semantics": "prolate-fd-working-precision-v2",
            "qr_arithmetic": xc_numerics::eigen::TRIDIAG_QR_SEMANTICS,
                "lambda_sq": lsq_bad.value_f64,
                "lambda_sq_mode": lsq_bad.mode_str(),
                "lambda_sq_identity": lsq_bad.filename_str(),
                "n_grid": n_grid,
                "precision_bits": prec,
                "eigenvalues": bad_strs,
            });
            std::fs::write(&bad_path, serde_json::to_string(&bad_json).unwrap()).unwrap();

            // 3. Skipped file: unrecognized name.
            let skipped_path = temp_dir.join("not_a_prolate_cache.txt");
            std::fs::write(&skipped_path, "irrelevant").unwrap();

            // 4. LoadFailed: matching name pattern, malformed JSON.
            let malformed_name =
                prolate_cache_filename(LambdaSq::integer(lambda_sq.value_u64 + 2), n_grid, prec);
            let malformed_path = temp_dir.join(&malformed_name);
            std::fs::write(&malformed_path, "{").unwrap();

            let report = verify_prolate_eigvals_cache_dir(&temp_dir).unwrap();
            assert_eq!(
                report.statuses.len(),
                4,
                "expected 4 statuses, got {}",
                report.statuses.len()
            );

            let mut saw_ok = false;
            let mut saw_invalid = false;
            let mut saw_skipped = false;
            let mut saw_loadfail = false;
            for s in &report.statuses {
                match s {
                    ProlateCacheFileStatus::Ok {
                        path,
                        lambda_sq: l,
                        n_grid: ng,
                        prec: p,
                    } => {
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
            assert_eq!(
                report.failure_count(),
                2,
                "LoadFailed + StructurallyInvalid both count; expected 2"
            );

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

        static PROLATE_CWD_LOCK: &std::sync::Mutex<()> = &crate::TEST_CWD_LOCK;

        struct ProlateCwdGuard {
            original: std::path::PathBuf,
            _cache_root: crate::TestCacheRoot,
            _lock: std::sync::MutexGuard<'static, ()>,
        }

        impl ProlateCwdGuard {
            fn enter(temp: &std::path::Path) -> Self {
                let lock = PROLATE_CWD_LOCK
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let original = std::env::current_dir().expect("current directory");
                std::env::set_current_dir(temp).expect("enter prolate test directory");
                Self {
                    original,
                    _cache_root: crate::TestCacheRoot::enter(temp),
                    _lock: lock,
                }
            }
        }

        impl Drop for ProlateCwdGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.original);
            }
        }

        /// A fresh, self-removing temp dir for cwd-relative cache tests.
        /// Declare it before the `ProlateCwdGuard` so the cwd is restored
        /// before the directory is removed.
        fn prolate_temp_cwd(tag: &str) -> xc_core::test_support::TestDir {
            crate::fresh_test_dir(&format!("prolate-cache-{tag}"))
        }

        /// `save` then `load` round-trips eigenvalues at every CacheMode
        /// tier, and `CacheMode::Off` writes nothing.
        #[test]
        fn prolate_cache_save_load_round_trip() {
            let prec = 256;
            let lambda_sq = LambdaSq::integer(25);
            let n_grid = 401usize;

            let temp = prolate_temp_cwd("round_trip");
            let _guard = ProlateCwdGuard::enter(&temp);

            // Build a real, validation-passing spectrum.
            let lambda = hp(prec, "5");
            let (diag, off_diag) = build_pw_matrix(&lambda, n_grid, prec);
            let evals = tridiag_eigenvalues_hp(&diag, &off_diag, prec).unwrap();

            // Off: writes nothing, reads nothing.
            save_prolate_eigvals_cache(lambda_sq, n_grid, prec, &evals, CacheMode::Off);
            assert!(
                load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::Off).is_none(),
                "Off should never read"
            );
            assert!(
                load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "Off save should have written nothing"
            );

            // JsonZip: writes ONLY the .json.zip (zip-only contract);
            // reads back identical by decompressing in memory.
            save_prolate_eigvals_cache(lambda_sq, n_grid, prec, &evals, CacheMode::JsonZip);

            // No uncompressed .json should be written.
            let jp = temp
                .join("data")
                .join("prolate_eigvals_cache")
                .join(prolate_cache_filename(lambda_sq, n_grid, prec));
            assert!(
                !jp.exists(),
                "zip-only: save must not write an uncompressed .json"
            );

            let got = load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip)
                .expect("JsonZip round-trip should load from the zip");
            assert_eq!(got.len(), evals.len());
            for (a, b) in evals.iter().zip(got.iter()) {
                assert_eq!(
                    a.to_string(),
                    b.to_string(),
                    "eigenvalue must round-trip exactly"
                );
            }

            // JsonOnly is now a read no-op (no uncompressed .json exists).
            assert!(
                load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonOnly).is_none(),
                "zip-only: JsonOnly must not read the zip"
            );

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }

        /// Negative: a structurally-invalid `.json` (descending order)
        /// must be skipped by `load` (returns None → recompute). Bad file
        /// preserved.
        #[test]
        fn prolate_load_skips_structurally_invalid_json() {
            let prec = 256;
            let lambda_sq = LambdaSq::integer(25);
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
            let json =
                serde_json::Value::Array(strs.into_iter().map(serde_json::Value::String).collect());
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

            assert!(
                load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "descending (non-ascending) spectrum in the zip must be skipped"
            );
            assert!(
                zip_path.exists(),
                "structurally-invalid zip should be preserved for inspection"
            );

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }

        /// Negative: a corrupt `.json.zip` must be detected and skipped
        /// without panic (`load` returns None). Corrupt file preserved.
        #[test]
        fn prolate_load_handles_corrupt_zip_gracefully() {
            let prec = 64;
            let lambda_sq = LambdaSq::integer(25);
            let n_grid = 401usize;

            let temp = prolate_temp_cwd("corrupt_zip");
            let _guard = ProlateCwdGuard::enter(&temp);

            let dir = temp.join("data").join("prolate_eigvals_cache");
            std::fs::create_dir_all(&dir).unwrap();
            // Garbage bytes named as the zip; no local .json, so JsonZip
            // falls through to the zip, fails to open it, returns None.
            let zip_path = dir.join(format!(
                "lambda_sq{}_ngrid{}_prec{}.json.zip",
                lambda_sq.filename_str(),
                n_grid,
                prec
            ));
            std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

            assert!(
                load_prolate_eigvals_cache(lambda_sq, n_grid, prec, CacheMode::JsonZip).is_none(),
                "corrupt .json.zip must be skipped, not loaded"
            );
            assert!(
                zip_path.exists(),
                "corrupt zip should be preserved for inspection"
            );

            drop(_guard);
            let _ = std::fs::remove_dir_all(&temp);
        }
    }
}
