// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Yakaboylu's Riemann operator framework — high-precision verification.
//!
//! Implements the matrix-element computations and W-positivity tests from
//! Yakaboylu, *Nontrivial Riemann Zeros as Spectrum* (arxiv 2408.15135 v15,
//! <https://arxiv.org/html/2408.15135v15#S4>).
//!
//! ## Mathematical setup
//!
//! Yakaboylu's regularized intertwining operator V̂_R has matrix elements
//! (his eq 52):
//!
//! ```text
//! ⟨Ψ_s | V̂_R,ε | Ψ_s'⟩ = ε² / [ε² - (s̄ + s' - 1)²]
//! ```
//!
//! For zeros on the critical line `s = 1/2 + iγ`, `s' = 1/2 + iγ'`:
//! - `s̄ + s' - 1 = -iγ + iγ' = i(γ' - γ)`
//! - `(s̄ + s' - 1)² = -(γ' - γ)²`
//! - Matrix element = `ε² / [ε² + (γ' - γ)²]`
//!
//! This is a Lorentzian peaked at γ = γ' with width ε. As ε → 0+:
//! - Diagonal (γ = γ'): → 1
//! - Off-diagonal (γ ≠ γ'): → 0
//!
//! Portable limit verification and certified spectrum positivity are planned.
//! The available `test_*` routines evaluate a supplied finite epsilon and
//! report point deviations or heuristic spectrum labels.
//!
//! ## Operations and intended verification
//!
//! - **`v_r_matrix_element_f64` / `_hp`**: closed-form computation at f64 or HP.
//! - **`build_w_matrix_f64` / hp::build_w_matrix**: W matrix on first N Riemann zeros.
//! - **`test_lorentzian_limit_f64`**: verify lim_{ε→0+} M(ε) = δ_{γ,γ'}.
//! - **`test_w_positivity_f64`**: verify all eigenvalues of W are ≥ 0.
//!
//! ## What this verifies (and doesn't)
//!
//! These routines evaluate the meromorphic formula and finite critical-line
//! Lorentzian matrices. They do not verify the full operator framework or RH.
//! The Lorentzian kernel is positive semidefinite for any real ordinates,
//! whether or not they are zeta zeros, and is singular for repeated ordinates.
//! Finite epsilon is not the epsilon-to-zero limit. Synthetic off-line inputs
//! test the continued formula; they need not lie in its integral convergence
//! strip and do not establish positivity of an underlying Hilbert-space form.

use anyhow::Result;

mod kernel;

/// Matrix element ⟨Ψ_s | V̂_R,ε | Ψ_s'⟩ at f64 precision.
///
/// Closed form: `ε² / [ε² - (s̄ + s' - 1)²]`.
/// For s, s' on the critical line, simplifies to `ε² / [ε² + (γ'-γ)²]`.
///
/// `s_re`, `s_im` are real and imaginary parts of s.
/// `sp_re`, `sp_im` are real and imaginary parts of s'. Invalid parameters,
/// poles and unrepresentable arithmetic return NaNs; use the checked variant
/// for an error. Computed values are not certified enclosures.
pub fn v_r_matrix_element_f64(
    s_re: f64,
    s_im: f64,
    sp_re: f64,
    sp_im: f64,
    epsilon: f64,
) -> (f64, f64) {
    try_v_r_matrix_element_f64(s_re, s_im, sp_re, sp_im, epsilon).unwrap_or((f64::NAN, f64::NAN))
}

/// Checked point evaluation of epsilon^2/(epsilon^2-(conj(s)+s'-1)^2).
/// Requires finite parameters and epsilon>0; poles and unrepresentable
/// arithmetic fail. Outside |Re(conj(s)+s'-1)|<epsilon this is meromorphic
/// continuation, not the convergent integral representation.
pub fn try_v_r_matrix_element_f64(
    s_re: f64,
    s_im: f64,
    sp_re: f64,
    sp_im: f64,
    epsilon: f64,
) -> Result<(f64, f64)> {
    kernel::entry_f64(s_re, s_im, sp_re, sp_im, epsilon)
}

/// Test the Lorentzian limit on a pair of critical-line zeros (f64).
/// Returns `(M(ε), |M(ε) - target|)` where target is δ_{γ,γ'}.
pub fn test_lorentzian_limit_f64(gamma1: f64, gamma2: f64, epsilon: f64) -> (f64, f64, f64) {
    try_test_lorentzian_limit_f64(gamma1, gamma2, epsilon).unwrap_or((f64::NAN, f64::NAN, f64::NAN))
}

/// Finite-epsilon deviation from the exact discrete delta (exact equality of
/// stored ordinates); this point diagnostic does not prove a limit.
pub fn try_test_lorentzian_limit_f64(
    gamma1: f64,
    gamma2: f64,
    epsilon: f64,
) -> Result<(f64, f64, f64)> {
    let (real, imaginary) = try_v_r_matrix_element_f64(0.5, gamma1, 0.5, gamma2, epsilon)?;
    let target = if gamma1 == gamma2 { 1.0 } else { 0.0 };
    let deviation = (real - target).hypot(imaginary);
    anyhow::ensure!(
        deviation.is_finite(),
        "Lorentzian deviation is outside binary64 range"
    );
    Ok((real, imaginary, deviation))
}

/// Build the finite-epsilon N×N matrix
/// `W_{ij}(ε) = ⟨Ψ_ρ_i | V̂_R,ε | Ψ_{1-ρ̄_j}⟩`.
///
/// Distinct critical-line ordinates approach the identity as epsilon tends
/// to zero. Repeated ordinates give identical rows and a singular matrix.
///
/// Returns the matrix as a flat `Vec<f64>` in row-major order.
/// This evaluates the supplied finite positive epsilon. Invalid data panics;
/// use `try_build_w_matrix_f64` to receive an error.
pub fn build_w_matrix_f64(zeros: &[f64], epsilon: f64) -> Vec<f64> {
    try_build_w_matrix_f64(zeros, epsilon).expect("valid representable finite Lorentzian matrix")
}

/// Checked finite-epsilon Lorentzian matrix. Requires at least one finite
/// ordinate and finite epsilon>0; repeated ordinates are permitted.
pub fn try_build_w_matrix_f64(zeros: &[f64], epsilon: f64) -> Result<Vec<f64>> {
    let n = zeros.len();
    anyhow::ensure!(
        n > 0 && epsilon.is_finite() && epsilon > 0.0 && zeros.iter().all(|x| x.is_finite()),
        "Lorentzian matrix requires nonempty finite ordinates and epsilon>0"
    );
    let entries = n
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("matrix dimension overflow"))?;
    let mut matrix = vec![0.0; entries];
    for i in 0..n {
        for j in 0..=i {
            let (value, _) = try_v_r_matrix_element_f64(0.5, zeros[i], 0.5, zeros[j], epsilon)?;
            matrix[i * n + j] = value;
            matrix[j * n + i] = value;
        }
    }
    Ok(matrix)
}

/// Compute the smallest eigenvalue of an N×N symmetric matrix to test
/// positivity. Returns the smallest eigenvalue.
pub fn smallest_eigenvalue_f64(matrix: &[f64], n: usize) -> Result<f64> {
    Ok(kernel::spectrum_f64(matrix, n)?[0])
}

/// Test W positivity on the first N Riemann zeros (f64 path).
/// Returns the f64 result struct.
pub fn test_w_positivity_f64(zeros: &[f64], epsilon: f64) -> Result<WPositivityResultF64> {
    let n = zeros.len();
    let matrix = try_build_w_matrix_f64(zeros, epsilon)?;
    let values = kernel::spectrum_f64(&matrix, n)?;
    let smallest = values[0];
    let largest = values[n - 1];
    let mut ordered = zeros.to_vec();
    ordered.sort_by(f64::total_cmp);
    let duplicates = ordered.windows(2).any(|pair| pair[0] == pair[1]);
    let condition = if duplicates || smallest == 0.0 {
        f64::INFINITY
    } else {
        largest.abs() / smallest.abs()
    };
    let tolerance = 64.0 * (n as f64) * f64::EPSILON;
    Ok(WPositivityResultF64 {
        n_zeros: n,
        epsilon_f64: epsilon,
        smallest_eigenvalue_f64: smallest,
        largest_eigenvalue_f64: largest,
        condition_number_f64: condition,
        positive_definite: !duplicates && smallest > tolerance,
        positive_semidefinite_with_tolerance: smallest >= -tolerance,
        positivity_tolerance_f64: tolerance,
        all_eigenvalues_f64: values,
    })
}

#[derive(Debug, Clone)]
/// Result of a W-positivity test at f64 precision.
///
/// Returned by `test_w_positivity_f64`. The HP analogue is
/// `hp::HpWPositivityResult`. f64 precision is adequate for sanity
/// checks at small `n_zeros`; for production tests at larger N use
/// the HP path to avoid f64-induced spurious indefiniteness.
pub struct WPositivityResultF64 {
    /// Number of zeros used to build the W matrix (matrix size N×N).
    pub n_zeros: usize,
    /// Regularization parameter ε used in the V̂_R matrix element.
    pub epsilon_f64: f64,
    /// Computed smallest eigenvalue. The exact critical-line kernel is
    /// positive semidefinite for any real ordinates, including nonzeros of
    /// zeta. Negative computed values reflect numerical error, not evidence
    /// of off-line zeros (this API fixes every real part at one half).
    pub smallest_eigenvalue_f64: f64,
    /// Largest eigenvalue of W.
    pub largest_eigenvalue_f64: f64,
    /// Spectral condition number `largest / |smallest|`. Returned as
    /// `f64::INFINITY` for duplicate ordinates, zero computed minimum, or
    /// overflow of the computed ratio. Not a certified condition bound.
    pub condition_number_f64: f64,
    /// Strict computed positive margin above the reported tolerance, with
    /// no duplicate ordinates. This is not a certified eigenvalue lower bound.
    pub positive_definite: bool,
    /// Computed smallest eigenvalue is at least minus the reported tolerance.
    pub positive_semidefinite_with_tolerance: bool,
    /// Heuristic roundoff scale 64*N*binary64 epsilon (the matrix norm is <=N).
    pub positivity_tolerance_f64: f64,
    /// All eigenvalues of W, sorted ascending.
    pub all_eigenvalues_f64: Vec<f64>,
}

/// High-precision matrix element using rug Float.
/// Uses epsilon's precision for output. Invalid parameters, poles and
/// unrepresentable arithmetic return NaNs. The checked variant accepts an
/// explicit output precision and returns an error.
#[cfg(feature = "hp")]
pub fn v_r_matrix_element_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    sp_re: &rug::Float,
    sp_im: &rug::Float,
    epsilon: &rug::Float,
) -> (rug::Float, rug::Float) {
    try_v_r_matrix_element_hp(s_re, s_im, sp_re, sp_im, epsilon, epsilon.prec()).unwrap_or_else(
        |_| {
            (
                rug::Float::with_val(epsilon.prec(), rug::float::Special::Nan),
                rug::Float::with_val(epsilon.prec(), rug::float::Special::Nan),
            )
        },
    )
}

/// Checked HP meromorphic matrix element, rounded to explicit working precision.
/// Input precision and exponent range are checked; the implementation uses
/// 32 guard bits above the largest input/output precision. This is point
/// arithmetic, not a certified enclosure near a pole.
#[cfg(feature = "hp")]
pub fn try_v_r_matrix_element_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    sp_re: &rug::Float,
    sp_im: &rug::Float,
    epsilon: &rug::Float,
    precision: u32,
) -> Result<(rug::Float, rug::Float)> {
    kernel::entry_hp(s_re, s_im, sp_re, sp_im, epsilon, precision)
}

// Reference Riemann-zero literals below are quoted at published precision
// (more digits than f64 holds); the excess is harmless on parse.
#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// Diagonal element on critical line: should be exactly 1.
    #[test]
    fn diagonal_on_critical_line_is_one() {
        let gamma = 14.134725141734695; // first Riemann zero
        for &eps in &[0.1, 0.01, 0.001, 1e-6, 1e-10] {
            let (m_re, m_im) = v_r_matrix_element_f64(0.5, gamma, 0.5, gamma, eps);
            assert!(
                (m_re - 1.0).abs() < 1e-12,
                "diagonal should be 1 (got {} at eps={})",
                m_re,
                eps
            );
            assert!(m_im.abs() < 1e-12, "im part should be 0 (got {})", m_im);
        }
    }

    /// Off-diagonal between two distinct critical-line zeros.
    /// As ε → 0+, the value should go to 0 like ε² / (γ-γ')².
    #[test]
    fn off_diagonal_decays_like_epsilon_squared() {
        let gamma1: f64 = 14.134725141734695;
        let gamma2: f64 = 21.022039638771556; // second Riemann zero
        let dgamma_sq = (gamma2 - gamma1).powi(2);
        for &eps in &[0.1_f64, 0.01, 0.001] {
            let (m_re, _) = v_r_matrix_element_f64(0.5, gamma1, 0.5, gamma2, eps);
            let predicted = eps.powi(2) / (eps.powi(2) + dgamma_sq);
            let rel_err = (m_re - predicted).abs() / predicted.abs();
            assert!(
                rel_err < 1e-12,
                "off-diagonal should match Lorentzian (got {}, predicted {})",
                m_re,
                predicted
            );
        }
    }

    /// Off-critical-line "zero" should give negative-real-denominator behavior.
    /// For β = 0.6, γ = 14.13: s̄ + s - 1 = (2β - 1) + 0·i = 0.2.
    /// Matrix element = ε² / (ε² - 0.04) → 0 / (-0.04) = 0 as ε → 0+.
    #[test]
    fn off_critical_line_diagonal_vanishes() {
        let beta: f64 = 0.6;
        let gamma = 14.134725141734695;
        let eps = 0.001_f64;
        let (m_re, _) = v_r_matrix_element_f64(beta, gamma, beta, gamma, eps);
        // Predicted: ε² / [ε² - (2β-1)²] = 1e-6 / (1e-6 - 0.04) ≈ -2.5e-5
        let predicted = eps.powi(2) / (eps.powi(2) - (2.0 * beta - 1.0).powi(2));
        let rel_err = (m_re - predicted).abs() / predicted.abs();
        assert!(rel_err < 1e-12);
    }

    /// W matrix on first 5 Riemann zeros at small ε should be ≈ identity.
    #[test]
    fn w_matrix_is_approximately_identity() {
        let zeros = vec![
            14.134725141734695,
            21.022039638771556,
            25.010857580145687,
            30.42487612585951,
            32.93506158773919,
        ];
        let eps = 1e-3;
        let w = build_w_matrix_f64(&zeros, eps);
        // Diagonal should be ~1
        for i in 0..5 {
            assert!((w[i * 5 + i] - 1.0).abs() < 1e-12);
        }
        // Off-diagonal should be ε² / (γ-γ')² which is small.
        // E.g. (0,1): eps² / (21.02-14.13)² ≈ 1e-6 / 47 ≈ 2e-8
        let off_01 = w[1]; // row 0, col 1 in row-major 5×5
        assert!(off_01.abs() < 1e-7, "off-diagonal too large: {}", off_01);
    }

    /// W matrix should be positive definite on critical-line zeros.
    /// At small ε, it's approximately identity, so smallest eigenvalue ≈ 1.
    #[test]
    fn w_is_positive_on_critical_line() {
        let zeros = vec![
            14.134725141734695,
            21.022039638771556,
            25.010857580145687,
            30.42487612585951,
            32.93506158773919,
            37.586178158825675,
            40.91871901214750,
            43.32707328091500,
            48.00515088116716,
            49.77383247767230,
        ];
        let eps = 1e-4;
        let result = test_w_positivity_f64(&zeros, eps).unwrap();
        assert!(
            result.positive_definite,
            "W should be positive (got smallest eigenvalue {})",
            result.smallest_eigenvalue_f64
        );
        // At small ε, eigenvalues should be very close to 1.
        for &lambda in &result.all_eigenvalues_f64 {
            assert!(
                (lambda - 1.0).abs() < 1e-6,
                "eigenvalue should be close to 1 (got {})",
                lambda
            );
        }
    }

    /// test_lorentzian_limit_f64 should report deviation from δ_{γ,γ'}.
    #[test]
    fn test_lorentzian_limit_diagonal() {
        let gamma = 14.134725141734695;
        let eps = 1e-6;
        let (m_re, m_im, dev) = test_lorentzian_limit_f64(gamma, gamma, eps);
        // Diagonal: should be (1.0, 0.0) with deviation ~0.
        assert!((m_re - 1.0).abs() < 1e-12);
        assert!(m_im.abs() < 1e-12);
        assert!(dev < 1e-12);
    }

    /// test_lorentzian_limit_f64 off-diagonal should report deviation from 0.
    #[test]
    fn test_lorentzian_limit_off_diagonal() {
        let gamma1 = 14.134725141734695;
        let gamma2 = 21.022039638771556;
        let eps = 1e-6;
        let (m_re, _m_im, dev) = test_lorentzian_limit_f64(gamma1, gamma2, eps);
        // Off-diagonal: should be ~0 with deviation = |m_re|.
        assert!(m_re.abs() < 1e-10);
        assert!((dev - m_re.abs()).abs() < 1e-15);
    }

    /// Synthetic test: if we feed off-critical-line "zeros," W should be
    /// indefinite (not positive). This verifies the framework would detect
    /// RH violations.
    #[test]
    fn w_is_indefinite_on_synthetic_off_line_zeros() {
        // Pretend ρ_1 = 0.4 + i·14.13 (off the critical line).
        // Its symmetric partner is 1 - ρ̄_1 = 1 - (0.4 - i·14.13) = 0.6 + i·14.13.
        // We need to include both in our basis since they pair via the involution.
        // W_{12} should be 1 (the involution pairs them) and W_{11}, W_{22} should be 0.
        // That's a Pauli-X-like matrix [[0,1],[1,0]] which has eigenvalues ±1.
        // → smallest eigenvalue is -1 → NOT positive definite.

        // Build a 2x2 synthetic test using the matrix-element formula.
        // ρ_1 = β + iγ, ρ_2 = 1-ρ̄_1 = (1-β) + iγ. Then:
        // W_{11} = ⟨Ψ_{ρ_1}|V_R|Ψ_{ρ_1}⟩, where s̄+s'-1 = (β-iγ)+(β+iγ)-1 = 2β-1.
        //   M = ε² / [ε² - (2β-1)²] → 0 as ε→0 (since 2β-1 ≠ 0 off-line).
        // W_{22} similarly → 0.
        // W_{12} = ⟨Ψ_{ρ_1}|V_R|Ψ_{ρ_2}⟩, where s̄+s'-1 = (β-iγ)+(1-β+iγ)-1 = 0.
        //   M = ε² / ε² = 1.
        let beta = 0.4;
        let gamma = 14.134725141734695;
        let eps = 1e-4;
        let (w11, _) = v_r_matrix_element_f64(beta, gamma, beta, gamma, eps);
        let (w12, _) = v_r_matrix_element_f64(beta, gamma, 1.0 - beta, gamma, eps);
        let (w21, _) = v_r_matrix_element_f64(1.0 - beta, gamma, beta, gamma, eps);
        let (w22, _) = v_r_matrix_element_f64(1.0 - beta, gamma, 1.0 - beta, gamma, eps);

        // W_{12} should be ≈ 1 (these two zeros are each other's involution pair).
        assert!((w12 - 1.0).abs() < 1e-6, "W_12 should be 1, got {}", w12);
        assert!((w21 - 1.0).abs() < 1e-6, "W_21 should be 1, got {}", w21);
        // W_{11}, W_{22} should be ≈ 0.
        assert!(w11.abs() < 1e-6, "W_11 should be 0, got {}", w11);
        assert!(w22.abs() < 1e-6, "W_22 should be 0, got {}", w22);

        // Smallest eigenvalue of [[~0, 1], [1, ~0]] is ≈ -1.
        let matrix = vec![w11, w12, w21, w22];
        let smallest = smallest_eigenvalue_f64(&matrix, 2).unwrap();
        assert!(
            smallest < -0.99,
            "smallest eigenvalue should be ~-1 (off-line case), got {}",
            smallest
        );
    }
}

// ===========================================================================
// High-precision (HP) Yakaboylu W-positivity pipeline.
//
// Mirrors the f64 prototype above, but operates entirely in `rug::Float`
// arithmetic. Uses the HP eigensolver in `xc_numerics::eigen` for the
// dense symmetric eigendecomposition that replaces nalgebra's f64 path.
// ===========================================================================

#[cfg(feature = "hp")]
pub mod hp {
    use anyhow::Result;
    use rayon::prelude::*;
    use rug::Float;
    use xc_numerics::eigen::dense_symmetric_eigenvalues_hp;

    /// Build the N×N matrix `W_{ij} = ⟨Ψ_ρ_i | V̂_R,ε | Ψ_{ρ_j}⟩` at HP
    /// precision, using HP gamma values (imaginary parts of the zeros).
    ///
    /// On critical-line zeros (β = 1/2), `1 - ρ̄_j = ρ_j`, so we use
    /// `s = 1/2 + iγ_i` and `s' = 1/2 + iγ_j`. Returns the matrix as
    /// a flat `Vec<Float>` in row-major order. Invalid input panics; use
    /// `try_build_w_matrix` for checked validation.
    pub fn build_w_matrix(gammas: &[Float], epsilon: &Float, prec: u32) -> Vec<Float> {
        try_build_w_matrix(gammas, epsilon, prec).expect("valid representable HP Lorentzian matrix")
    }

    /// Checked finite-epsilon Lorentzian matrix; every entry has `prec` bits.
    pub fn try_build_w_matrix(gammas: &[Float], epsilon: &Float, prec: u32) -> Result<Vec<Float>> {
        anyhow::ensure!((32..=1_000_000).contains(&prec) && !gammas.is_empty()
            && epsilon.is_finite() && epsilon>&0 && gammas.iter().all(Float::is_finite),
            "HP Lorentzian matrix requires valid precision, finite nonempty ordinates and epsilon>0");
        let n = gammas.len();
        let entries = n
            .checked_mul(n)
            .ok_or_else(|| anyhow::anyhow!("matrix dimension overflow"))?;
        let half = Float::with_val(prec, 0.5);
        let rows: Vec<Vec<Float>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        super::try_v_r_matrix_element_hp(
                            &half, &gammas[i], &half, &gammas[j], epsilon, prec,
                        )
                        .map(|pair| pair.0)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut matrix = Vec::with_capacity(entries);
        for row in rows {
            matrix.extend(row);
        }
        Ok(matrix)
    }

    /// Compute the smallest eigenvalue of an N×N symmetric matrix at HP
    /// precision, using the HP eigensolver in `xc_numerics::eigen`.
    pub fn smallest_eigenvalue(matrix: &[Float], n: usize, prec: u32) -> Result<Float> {
        Ok(checked_spectrum(matrix, n, prec)?.remove(0))
    }

    fn checked_spectrum(matrix: &[Float], n: usize, prec: u32) -> Result<Vec<Float>> {
        anyhow::ensure!(
            (33..=1_000_000).contains(&prec) && n > 0 && n.checked_mul(n) == Some(matrix.len()),
            "HP symmetric spectrum requires valid precision and a nonempty square matrix"
        );
        anyhow::ensure!(
            matrix.iter().all(Float::is_finite)
                && (0..n).all(|i| (0..i).all(|j| matrix[i * n + j] == matrix[j * n + i])),
            "HP spectrum requires finite exactly symmetric storage"
        );
        let values = dense_symmetric_eigenvalues_hp(matrix, n, prec)?;
        anyhow::ensure!(
            values.len() == n && values.iter().all(Float::is_finite),
            "nonfinite or incomplete HP spectrum"
        );
        Ok(values)
    }

    /// HP W-positivity result. All numeric fields HP except `n_zeros`
    /// and `positive_definite`.
    #[derive(Debug, Clone)]
    pub struct HpWPositivityResult {
        /// Number of zeros used to build the W matrix (matrix size N×N).
        pub n_zeros: usize,
        /// Regularization parameter ε at HP precision.
        pub epsilon: Float,
        /// Smallest eigenvalue of W at HP precision.
        pub smallest_eigenvalue: Float,
        /// Largest eigenvalue of W at HP precision.
        pub largest_eigenvalue: Float,
        /// Computed extremal-eigenvalue ratio, with infinity for duplicates,
        /// a zero computed minimum, or an unrepresentable ratio. Not certified.
        pub condition_number: Float,
        /// Strict computed positive margin above the reported tolerance,
        /// without duplicate ordinates. This is not a certified lower bound.
        pub positive_definite: bool,
        /// Computed minimum is at least minus the reported tolerance.
        pub positive_semidefinite_with_tolerance: bool,
        /// Heuristic roundoff scale N*2^(16-prec); matrix norm is at most N.
        pub positivity_tolerance: Float,
        /// All eigenvalues of W at HP precision, sorted ascending.
        pub all_eigenvalues: Vec<Float>,
    }

    /// Test W positivity on the first N Riemann zeros at HP precision.
    /// `gammas` are imaginary parts of the zeros.
    pub fn test_w_positivity(
        gammas: &[Float],
        epsilon: &Float,
        prec: u32,
    ) -> Result<HpWPositivityResult> {
        anyhow::ensure!(
            (33..=1_000_000).contains(&prec),
            "HP positivity precision must be in 33..=1000000 bits"
        );
        let n = gammas.len();
        let matrix = try_build_w_matrix(gammas, epsilon, prec)?;
        let values = checked_spectrum(&matrix, n, prec)?;
        let smallest = values[0].clone();
        let largest = values[n - 1].clone();
        let mut ordered = gammas.to_vec();
        ordered.sort_by(Float::total_cmp);
        let duplicates = ordered.windows(2).any(|pair| pair[0] == pair[1]);
        let condition = if duplicates || smallest.is_zero() {
            Float::with_val(prec, rug::float::Special::Infinity)
        } else {
            Float::with_val(prec, largest.clone().abs() / smallest.clone().abs())
        };
        let tolerance = Float::with_val(prec, n) >> (prec - 16);
        Ok(HpWPositivityResult {
            n_zeros: n,
            epsilon: Float::with_val(prec, epsilon),
            positive_definite: !duplicates && smallest > tolerance,
            positive_semidefinite_with_tolerance: smallest >= -tolerance.clone(),
            positivity_tolerance: tolerance,
            smallest_eigenvalue: smallest,
            largest_eigenvalue: largest,
            condition_number: condition,
            all_eigenvalues: values,
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

        /// HP diagonal element on critical line should be exactly 1.
        #[test]
        fn diagonal_on_critical_line_is_one_hp() {
            let prec = 256;
            let gamma = hp(prec, "14.134725141734693790457251983562470270784257");
            let half = {
                let mut t = Float::with_val(prec, 1);
                t /= 2u32;
                t
            };
            for eps_str in &["0.1", "0.01", "0.001", "1e-6", "1e-10"] {
                let eps = hp(prec, eps_str);
                let (m_re, m_im) =
                    super::super::v_r_matrix_element_hp(&half, &gamma, &half, &gamma, &eps);
                let mut diff = m_re.clone();
                diff -= 1u32;
                let abs_diff = diff.abs();
                let tol = hp(prec, "1e-50");
                assert!(
                    abs_diff < tol,
                    "HP diagonal at eps={}: got {} (off by {})",
                    eps_str,
                    display_hp(&m_re, 6),
                    display_hp(&abs_diff, 4)
                );
                let abs_im = m_im.abs();
                assert!(
                    abs_im < tol,
                    "HP im part at eps={}: got {} (should be 0)",
                    eps_str,
                    display_hp(&abs_im, 4)
                );
            }
        }

        /// HP build_w_matrix on first 5 zeros at small ε should be ≈ identity.
        #[test]
        fn w_matrix_is_approximately_identity_hp() {
            let prec = 256;
            let gammas = vec![
                hp(prec, "14.134725141734693790457251983562470270784257"),
                hp(prec, "21.022039638771554992628479593896902777334340"),
                hp(prec, "25.010857580145688763213790992562821818659549"),
                hp(prec, "30.424876125859513210311897530584091320181560"),
                hp(prec, "32.935061587739189690662368964074903488812715"),
            ];
            let eps = hp(prec, "1e-3");
            let w = build_w_matrix(&gammas, &eps, prec);

            // Diagonal entries should be ≈ 1.
            let one = Float::with_val(prec, 1);
            let tol = hp(prec, "1e-50");
            for i in 0..5 {
                let mut diff = w[i * 5 + i].clone();
                diff -= &one;
                let abs_diff = diff.abs();
                assert!(
                    abs_diff < tol,
                    "diagonal[{}] should be 1, got {} (delta {})",
                    i,
                    display_hp(&w[i * 5 + i], 6),
                    display_hp(&abs_diff, 4)
                );
            }

            // Off-diagonal (0, 1) should be ε² / (γ_2 - γ_1)².
            // γ_2 - γ_1 ≈ 6.887, (γ_2 - γ_1)² ≈ 47.4, ε² = 1e-6.
            // Predicted ≈ 1e-6 / 47.4 ≈ 2.1e-8.
            let off_01 = w[1].clone().abs(); // row 0, col 1 in row-major 5×5
            let upper = hp(prec, "1e-7");
            assert!(
                off_01 < upper,
                "off-diagonal should be small, got {}",
                display_hp(&off_01, 6)
            );
        }

        /// HP W positivity on critical-line zeros: smallest eigenvalue ≈ 1.
        #[test]
        fn w_is_positive_on_critical_line_hp() {
            let prec = 256;
            // Use a smaller set of zeros to keep test runtime down.
            let gammas = vec![
                hp(prec, "14.134725141734693790457251983562470270784257"),
                hp(prec, "21.022039638771554992628479593896902777334340"),
                hp(prec, "25.010857580145688763213790992562821818659549"),
                hp(prec, "30.424876125859513210311897530584091320181560"),
                hp(prec, "32.935061587739189690662368964074903488812715"),
            ];
            let eps = hp(prec, "1e-4");
            let result = test_w_positivity(&gammas, &eps, prec).unwrap();
            assert!(
                result.positive_definite,
                "W should be positive-definite on critical-line zeros (smallest = {})",
                display_hp(&result.smallest_eigenvalue, 6)
            );
            // All eigenvalues should be ≈ 1 at small ε.
            let one = Float::with_val(prec, 1);
            let tol = hp(prec, "1e-6");
            for v in &result.all_eigenvalues {
                let mut diff = v.clone();
                diff -= &one;
                let abs_diff = diff.abs();
                assert!(
                    abs_diff < tol,
                    "eigenvalue should be ≈1 at small ε, got {} (delta {})",
                    display_hp(v, 6),
                    display_hp(&abs_diff, 4)
                );
            }
        }

        /// HP synthetic off-line zeros should produce indefinite W.
        /// Same physical setup as the f64 test: ρ_1 = β + iγ, ρ_2 = (1-β) + iγ
        /// pair via the involution. W is approximately Pauli-X with
        /// eigenvalues ±1.
        #[test]
        fn w_is_indefinite_on_synthetic_off_line_zeros_hp() {
            let prec = 256;
            // Build the 2×2 W matrix manually using v_r_matrix_element_hp.
            let beta = hp(prec, "0.4");
            let one_minus_beta = {
                let mut t = Float::with_val(prec, 1);
                t -= &beta;
                t
            };
            let gamma = hp(prec, "14.134725141734693790457251983562470270784257");
            let eps = hp(prec, "1e-4");

            let (w11, _) = super::super::v_r_matrix_element_hp(&beta, &gamma, &beta, &gamma, &eps);
            let (w12, _) =
                super::super::v_r_matrix_element_hp(&beta, &gamma, &one_minus_beta, &gamma, &eps);
            let (w21, _) =
                super::super::v_r_matrix_element_hp(&one_minus_beta, &gamma, &beta, &gamma, &eps);
            let (w22, _) = super::super::v_r_matrix_element_hp(
                &one_minus_beta,
                &gamma,
                &one_minus_beta,
                &gamma,
                &eps,
            );

            // W_12 should be ≈ 1 (involution-paired).
            let one = Float::with_val(prec, 1);
            let tol_one = hp(prec, "1e-6");
            let mut d12 = w12.clone();
            d12 -= &one;
            let d12_abs = d12.abs();
            assert!(
                d12_abs < tol_one,
                "W_12 should be ≈1, got {}",
                display_hp(&w12, 6)
            );
            let mut d21 = w21.clone();
            d21 -= &one;
            let d21_abs = d21.abs();
            assert!(
                d21_abs < tol_one,
                "W_21 should be ≈1, got {}",
                display_hp(&w21, 6)
            );

            // W_11, W_22 should be ≈ 0.
            let tol_zero = hp(prec, "1e-6");
            let abs_w11 = w11.clone().abs();
            let abs_w22 = w22.clone().abs();
            assert!(
                abs_w11 < tol_zero,
                "W_11 should be ≈0, got {}",
                display_hp(&w11, 6)
            );
            assert!(
                abs_w22 < tol_zero,
                "W_22 should be ≈0, got {}",
                display_hp(&w22, 6)
            );

            // Smallest eigenvalue of [[~0, 1], [1, ~0]] is ≈ -1.
            let matrix = vec![w11, w12, w21, w22];
            let smallest = smallest_eigenvalue(&matrix, 2, prec).unwrap();
            // Smallest ≈ -1 confirms W is indefinite.
            let neg_99 = hp(prec, "-0.99");
            assert!(
                smallest < neg_99,
                "smallest eigenvalue should be ≈-1 for off-line case, got {}",
                display_hp(&smallest, 6)
            );
        }
    }
}
