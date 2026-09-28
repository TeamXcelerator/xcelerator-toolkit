// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.
//

//! Point quadrature and real-part crossing diagnostics for truncated and
//! eigenfunction-weighted completed eta transforms. The analytic kernel is
//! omega(t)=t*exp(t)/(1+exp(t))^2 and its full Mellin transform is
//! Gamma(s+1)*eta(s) on its domain of convergence. A crossing of Re(G) on the
//! critical line does not establish G=0. Quadrature, truncation, state accuracy,
//! and comparisons with Riemann zeros require separate validation.
//! Endpoint terms can dominate the real-part crossings, with a characteristic
//! spacing pi/log(lambda); this is not an exact general spacing law or evidence
//! for zeta zeros. The u-variable rule may converge slowly at large lambda:
//! increasing arithmetic precision alone does not control quadrature error.

mod transforms;
use transforms::*;
mod crossings;
pub use crossings::*;

/// Algorithm identity for stable eta evaluation and checked real-part scans.
pub const MELLIN_SEMANTICS_VERSION: &str = "eta-range-scaled-real-crossings-v4";

/// ω(t) = t·e^t / (1 + e^t)² — the kernel of the completed eta function.
/// Well-behaved for t > 0; decays exponentially for large t.
#[inline]
pub fn omega_f64(t: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let magnitude = t.abs();
    if magnitude == 0.0 {
        return t;
    }
    let exponential = (-magnitude).exp();
    // A subnormal exponential can lose most of its significant bits before
    // multiplication restores a normal result. Split the exponential so the
    // only possibly subnormal multiplication is the final one.
    let numerator = if exponential < f64::MIN_POSITIVE {
        let half_exponential = (-0.5 * magnitude).exp();
        (magnitude * half_exponential) * half_exponential
    } else {
        magnitude * exponential
    };
    (numerator / (1.0 + exponential).powi(2)).copysign(t)
}

/// Evaluate the truncated completed eta function at complex s = σ + it:
///
/// Λ_λ(s) = ∫_{λ⁻¹}^{λ} u^{s-1} · ω(u) du
///
/// Uses Gauss-Legendre quadrature at f64.
/// Returns (real part, imaginary part).
pub fn truncated_lambda_f64(s_re: f64, s_im: f64, lambda: f64, n_quad: usize) -> (f64, f64) {
    try_truncated_lambda_f64(s_re, s_im, lambda, n_quad).unwrap_or((f64::NAN, f64::NAN))
}

/// Checked point quadrature. Rejects invalid domains and nonfinite arithmetic.
/// This does not certify quadrature or truncation error.
pub fn try_truncated_lambda_f64(
    s_re: f64,
    s_im: f64,
    lambda: f64,
    n_quad: usize,
) -> anyhow::Result<(f64, f64)> {
    validate_transform_f64(s_re, s_im, lambda, n_quad)?;

    // Gauss-Legendre on [λ⁻¹, λ] mapped from [-1, 1].
    let a = 1.0 / lambda;
    let b = lambda;
    let mid = a.midpoint(b);
    let half = 0.5 * (b - a);

    let (nodes, weights) = xc_numerics::quadrature::try_gl_nodes_weights_f64(n_quad)?;

    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for i in 0..n_quad {
        let u = mid + half * nodes[i];
        let w = weights[i] * half;
        // u^{s-1} = u^{σ-1} · exp(i·t·ln u)
        let ln_u = u.ln();
        let common = mellin_amplitude_f64(s_re, u);
        let phase = s_im * ln_u;
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();
        let integrand_re = common * cos_phase;
        let integrand_im = common * sin_phase;
        sum_re += w * integrand_re;
        sum_im += w * integrand_im;
    }
    anyhow::ensure!(
        sum_re.is_finite() && sum_im.is_finite(),
        "nonfinite Mellin quadrature result"
    );
    Ok((sum_re, sum_im))
}

/// Evaluate the ξ_λ-weighted Mellin transform at complex s:
///
/// G(s) = ∫_{λ⁻¹}^{λ} f_λ(u) · ω(u) · u^{s-1} du
///
/// where f_λ(u) = (1/√L) Σ_n ξ_n · exp(2πi·n·log(λu)/L).
///
/// Since ξ is even (ξ_{-n} = ξ_n), f_λ is real-valued:
/// f_λ(u) = (1/√L) [ξ_0 + 2 Σ_{n=1}^N ξ_n · cos(2π·n·log(λu)/L)]
///
/// Returns (real part, imaginary part).
pub fn xi_weighted_mellin_f64(
    s_re: f64,
    s_im: f64,
    lambda: f64,
    xi: &[f64], // length 2N+1, indexed j = -N..N with xi[N] = ξ_0
    n_modes: usize,
    n_quad: usize,
) -> (f64, f64) {
    try_xi_weighted_mellin_f64(s_re, s_im, lambda, xi, n_modes, n_quad)
        .unwrap_or((f64::NAN, f64::NAN))
}

/// Checked point quadrature. Rejects invalid domains and nonfinite arithmetic.
/// This does not certify quadrature or truncation error.
/// The full 2N+1 coefficient vector must be finite and exactly even.
pub fn try_xi_weighted_mellin_f64(
    s_re: f64,
    s_im: f64,
    lambda: f64,
    xi: &[f64], // length 2N+1, indexed j = -N..N with xi[N] = ξ_0
    n_modes: usize,
    n_quad: usize,
) -> anyhow::Result<(f64, f64)> {
    validate_transform_f64(s_re, s_im, lambda, n_quad)?;
    anyhow::ensure!(lambda > 1.0, "weighted Mellin requires lambda > 1");
    let length = n_modes
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("Mellin mode dimension overflow"))?;
    anyhow::ensure!(
        xi.len() == length && n_modes <= (1u64 << 53) as usize,
        "Mellin coefficient shape or index range"
    );
    anyhow::ensure!(
        xi.iter().all(|v| v.is_finite()),
        "Mellin coefficients must be finite at the working precision"
    );
    anyhow::ensure!(
        (0..n_modes).all(|n| xi[n] == xi[length - 1 - n]),
        "Mellin cosine reconstruction requires an exactly even vector"
    );

    let l = 2.0 * lambda.ln();
    let inv_sqrt_l = 1.0 / l.sqrt();
    let a = 1.0 / lambda;
    let b = lambda;
    let mid = a.midpoint(b);
    let half = 0.5 * (b - a);

    let (nodes, weights) = xc_numerics::quadrature::try_gl_nodes_weights_f64(n_quad)?;

    let xi_0 = xi[n_modes];
    let xi_pos: Vec<f64> = (1..=n_modes).map(|n| xi[n_modes + n]).collect();

    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for i in 0..n_quad {
        let u = mid + half * nodes[i];
        let w = weights[i] * half;

        // f_λ(u) via cosine reconstruction
        let phase_base = 2.0 * std::f64::consts::PI * (lambda.ln() + u.ln()) / l;
        let mut f_val = xi_0;
        for n in 1..=n_modes {
            f_val += 2.0 * xi_pos[n - 1] * (n as f64 * phase_base).cos();
        }
        f_val *= inv_sqrt_l;

        // u^{s-1} · ω(u)
        let ln_u = u.ln();
        let common = mellin_amplitude_f64(s_re, u);
        let phase = s_im * ln_u;
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();

        let integrand_re = f_val * common * cos_phase;
        let integrand_im = f_val * common * sin_phase;
        sum_re += w * integrand_re;
        sum_im += w * integrand_im;
    }
    anyhow::ensure!(
        sum_re.is_finite() && sum_im.is_finite(),
        "nonfinite Mellin quadrature result"
    );
    Ok((sum_re, sum_im))
}

// Reference Riemann-zero literals below are quoted at published precision
// (more digits than f64 holds); the excess is harmless on parse.
#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// ω(t) should be positive for t > 0 and peak near t ≈ 1.
    #[test]
    fn omega_is_positive() {
        for &t in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
            assert!(omega_f64(t) > 0.0, "omega_f64({}) should be positive", t);
        }
    }

    /// The full Λ(s) = ∫_0^∞ t^{s-1} ω(t) dt should equal Γ(s+1)·η(s).
    /// At s = 2: Γ(3)·η(2) = 2 · π²/12 = π²/6 ≈ 1.6449.
    /// We can't integrate to ∞ at f64, but truncated at λ=50 should be close.
    #[test]
    fn truncated_lambda_at_s2_close_to_gamma3_eta2() {
        let (re, im) = truncated_lambda_f64(2.0, 0.0, 50.0, 200);
        let expected = std::f64::consts::PI.powi(2) / 6.0;
        let rel_err = (re - expected).abs() / expected;
        assert!(
            rel_err < 0.01,
            "Λ_50(2) = {} should be close to π²/6 = {} (rel err {})",
            re,
            expected,
            rel_err
        );
        assert!(im.abs() < 1e-10);
    }

    /// Λ_λ(s) should have a zero near the first Riemann zero (s = 1/2 + i·14.13)
    /// for large enough λ.
    ///
    /// Skipped in debug builds: 1000 scan points × 200-point GL — too slow
    /// in an unoptimized binary.
    ///
    /// Also ignored in routine runs: the bisection loop with 200-pt GL per
    /// eval is slow enough to time out on WSL2 even in release.
    /// Run with `--include-ignored --release` to exercise explicitly.
    #[test]
    #[ignore = "Mellin zero scan — too slow for routine test runs; run with: cargo test --release --features hp -- --include-ignored"]
    #[cfg(not(debug_assertions))]
    fn truncated_lambda_has_zero_near_first_riemann_zero() {
        let lambda = 50.0;
        let zeros = scan_critical_line_zeros_f64(
            &|sigma, t| truncated_lambda_f64(sigma, t, lambda, 200),
            10.0,
            20.0,
            1000,
        );
        // Should find at least one zero near 14.13.
        let first_riemann = 14.134725141734695;
        let closest = zeros
            .iter()
            .map(|&z| (z - first_riemann).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest < 1.0,
            "should find a zero near 14.13 (closest was {:.4} away, zeros found: {:?})",
            closest,
            zeros
        );
    }

    /// Comprehensive scan: find zeros of Λ_λ on the critical line for
    /// λ = √13 (our standard CCM config) and compare to first 10 Riemann zeros.
    ///
    /// Skipped in debug builds: 5000 scan points × 300-point GL is ~270M
    /// iterations — minutes in an unoptimized binary. Use `cargo test --release`.
    ///
    /// Also ignored in routine runs (even in release): the full bisection
    /// loop makes this many minutes even in release. Run explicitly with
    /// `--include-ignored` when validating the Mellin zero scan.
    #[test]
    #[ignore = "comprehensive Mellin zero scan — too slow for routine test runs; run with: cargo test --release --features hp -- --include-ignored"]
    #[cfg(not(debug_assertions))]
    fn truncated_lambda_zeros_vs_riemann_at_lambda_sqrt13() {
        let lambda = 13.0_f64.sqrt();
        let riemann_zeros = [
            14.134725141734695,
            21.022039638771556,
            25.010857580145687,
            30.424876125859512,
            32.935061587739189,
            37.586178158825675,
            40.918719012147500,
            43.327073280915000,
            48.005150881167160,
            49.773832477672300,
        ];
        let zeros = scan_critical_line_zeros_f64(
            &|sigma, t| truncated_lambda_f64(sigma, t, lambda, 300),
            5.0,
            55.0,
            5000,
        );
        eprintln!(
            "\nΛ_λ real-part crossings on critical line (λ = √13 ≈ {:.4}):",
            lambda
        );
        eprintln!(
            "{:>5} {:>15} {:>15} {:>12}",
            "k", "Λ_λ zero", "Riemann zero", "difference"
        );
        for (i, &rz) in riemann_zeros.iter().enumerate() {
            let closest = zeros
                .iter()
                .map(|&z| (z, (z - rz).abs()))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((z, diff)) = closest {
                eprintln!("{:>5} {:>15.6} {:>15.6} {:>12.4e}", i + 1, z, rz, diff);
            } else {
                eprintln!("{:>5} {:>15} {:>15.6} {:>12}", i + 1, "NOT FOUND", rz, "—");
            }
        }
        eprintln!(
            "Total real-part crossings found in [5, 55]: {}",
            zeros.len()
        );
        // At λ=√13, the truncation is severe (interval [0.277, 3.606]).
        // We may not find all zeros. Just check we find at least some.
        assert!(
            !zeros.is_empty(),
            "should find at least one real-part crossing"
        );
    }

    /// Same scan at λ = 10 (λ² = 100) — larger interval, should be closer.
    ///
    /// Skipped in debug builds: 3000 scan points × 300-point GL — minutes
    /// in an unoptimized binary. Use `cargo test --release`.
    ///
    /// Also ignored in routine runs (even in release): the full bisection
    /// loop makes this many minutes even in release. Run explicitly with
    /// `--include-ignored` when validating the Mellin zero scan.
    #[test]
    #[ignore = "comprehensive Mellin zero scan — too slow for routine test runs; run with: cargo test --release --features hp -- --include-ignored"]
    #[cfg(not(debug_assertions))]
    fn truncated_lambda_zeros_vs_riemann_at_lambda_10() {
        let lambda = 10.0;
        let riemann_zeros = [
            14.134725141734695,
            21.022039638771556,
            25.010857580145687,
            30.424876125859512,
            32.935061587739189,
        ];
        let zeros = scan_critical_line_zeros_f64(
            &|sigma, t| truncated_lambda_f64(sigma, t, lambda, 300),
            10.0,
            40.0,
            3000,
        );
        eprintln!("\nΛ_λ real-part crossings on critical line (λ = 10):");
        eprintln!(
            "{:>5} {:>15} {:>15} {:>12}",
            "k", "Λ_λ zero", "Riemann zero", "difference"
        );
        for (i, &rz) in riemann_zeros.iter().enumerate() {
            let closest = zeros
                .iter()
                .map(|&z| (z, (z - rz).abs()))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((z, diff)) = closest {
                eprintln!("{:>5} {:>15.6} {:>15.6} {:>12.4e}", i + 1, z, rz, diff);
            } else {
                eprintln!("{:>5} {:>15} {:>15.6} {:>12}", i + 1, "NOT FOUND", rz, "—");
            }
        }
        eprintln!(
            "Total real-part crossings found in [10, 40]: {}",
            zeros.len()
        );
        // Check the actual claimed quantity, not proximity to unrelated zeros.
        assert!(!zeros.is_empty());
        for t in zeros {
            assert!((10.0..=40.0).contains(&t));
            let (real, imaginary) = truncated_lambda_f64(0.5, t, lambda, 300);
            assert!(real.abs() <= 1e-8 * (1.0 + imaginary.abs()));
        }
    }

    /// Idea 2: ξ_λ-weighted Mellin G(s) = ∫ f_λ(u)·ω(u)·u^{s-1} du.
    /// Compare its zeros to Riemann zeros. If weighting by ξ_λ improves
    /// accuracy over the unweighted Λ_λ, that's evidence of a Mellin-side
    /// bridge.
    ///
    /// Skipped in debug builds: 5000 scan points × 300-point GL + 241×241
    /// f64 eigen — minutes in an unoptimized binary. Use `cargo test --release`.
    ///
    /// Also ignored in routine runs: same bisection cost as the other scans.
    #[test]
    #[ignore = "comprehensive Mellin zero scan — too slow for routine test runs; run with: cargo test --release --features hp -- --include-ignored"]
    #[cfg(not(debug_assertions))]
    fn xi_weighted_mellin_zeros_at_lambda_sqrt13() {
        // Run CCM at f64 to get ξ_λ.
        let n_modes = 120;
        let params = crate::ccm::CcmParams::from_lambda_sq_integer(13, n_modes);
        let result = crate::ccm::run_f64(&params).unwrap();
        let xi = &result.xi;
        // λ = √13 as f64 — only used as the Mellin integral parameter.
        let lambda = 13.0_f64.sqrt();

        let riemann_zeros = [
            14.134725141734695,
            21.022039638771556,
            25.010857580145687,
            30.424876125859512,
            32.935061587739189,
            37.586178158825675,
            40.918719012147500,
            43.327073280915000,
            48.005150881167160,
            49.773832477672300,
        ];

        // Scan zeros of G(s) on the critical line.
        let zeros = scan_critical_line_zeros_f64(
            &|sigma, t| xi_weighted_mellin_f64(sigma, t, lambda, xi, n_modes, 300),
            5.0,
            55.0,
            5000,
        );

        eprintln!("\nG(s) = ∫ f_λ·ω·u^{{s-1}} real-part crossings on critical line (λ = √13):");
        eprintln!(
            "{:>5} {:>15} {:>15} {:>12}",
            "k", "G zero", "Riemann zero", "difference"
        );
        for (i, &rz) in riemann_zeros.iter().enumerate() {
            let closest = zeros
                .iter()
                .map(|&z| (z, (z - rz).abs()))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((z, diff)) = closest {
                eprintln!("{:>5} {:>15.6} {:>15.6} {:>12.4e}", i + 1, z, rz, diff);
            } else {
                eprintln!("{:>5} {:>15} {:>15.6} {:>12}", i + 1, "NOT FOUND", rz, "—");
            }
        }
        eprintln!("Total Re(G) crossings found in [5, 55]: {}", zeros.len());
        // Just check it runs and finds some zeros.
        assert!(
            !zeros.is_empty(),
            "should find at least one real-part crossing"
        );
    }

    /// Fast test for xi_weighted_mellin_f64 with a synthetic flat ξ vector.
    /// With ξ_0 = 1 and all other ξ_n = 0, f_λ(u) = 1/√L (constant).
    /// G(s) = (1/√L) · Λ_λ(s), so G and Λ_λ should have the same zeros.
    #[test]
    fn xi_weighted_mellin_flat_xi_matches_unweighted() {
        let lambda: f64 = 5.0;
        let n_modes: usize = 10;
        let l = (lambda * lambda).ln();
        // Flat ξ: only ξ_0 = √L, rest zero. Then f_λ(u) = (1/√L)·√L = 1.
        let mut xi = vec![0.0_f64; 2 * n_modes + 1];
        xi[n_modes] = l.sqrt();

        // Evaluate at a specific point on the critical line.
        let t = 14.0;
        let (g_re, g_im) = xi_weighted_mellin_f64(0.5, t, lambda, &xi, n_modes, 100);
        let (l_re, l_im) = truncated_lambda_f64(0.5, t, lambda, 100);

        // With flat ξ, G(s) = Λ_λ(s) (the weighting is constant 1).
        let re_diff = (g_re - l_re).abs();
        let im_diff = (g_im - l_im).abs();
        assert!(
            re_diff < 1e-10,
            "G and Λ_λ should match (re diff: {:.2e})",
            re_diff
        );
        assert!(
            im_diff < 1e-10,
            "G and Λ_λ should match (im diff: {:.2e})",
            im_diff
        );
    }

    /// omega_f64(t) should peak near t = 1 and decay for large t.
    #[test]
    fn omega_peaks_near_one() {
        let peak = omega_f64(1.0);
        assert!(omega_f64(0.1) < peak);
        assert!(omega_f64(5.0) < peak);
        assert!(omega_f64(10.0) < omega_f64(5.0)); // monotone decay for t > 1
    }
}

// ===========================================================================
// High-precision Mellin computation (requires ccm-rug feature)
// ===========================================================================

/// Stable HP eta kernel at the precision of t. The exact identity using
/// exp(-|t|) avoids positive-exponential overflow and a fixed asymptotic cutoff.
/// This is a computed point value, not an error enclosure.
#[cfg(feature = "hp")]
pub fn omega_hp(t: &rug::Float) -> rug::Float {
    use rug::Float;
    let prec = t.prec();
    let Some(guard) = prec
        .checked_add(64)
        .filter(|p| *p <= rug::float::prec_max())
    else {
        return Float::with_val(prec, rug::float::Special::Nan);
    };
    if !t.is_finite() {
        return Float::with_val(prec, rug::float::Special::Nan);
    }
    if t.is_zero() {
        return t.clone();
    }
    let magnitude = Float::with_val(guard, t).abs();
    let exponential = (-magnitude.clone()).exp();
    let mut numerator =
        if exponential.is_zero() || exponential.get_exp() == Some(rug::float::exp_min()) {
            // MPFR underflow can round a nonzero exponential to its minimum
            // positive value. Split the exponential before that rounding loses
            // information; the final product alone approaches the exponent limit.
            let half_exponential = (-magnitude.clone() / 2u32).exp();
            let mut product = Float::with_val(guard, &magnitude * &half_exponential);
            product *= half_exponential;
            product
        } else {
            Float::with_val(guard, &magnitude * &exponential)
        };
    let mut denominator = exponential;
    denominator += 1u32;
    denominator.square_mut();
    numerator /= denominator;
    if t.is_sign_negative() {
        numerator = -numerator;
    }
    Float::with_val(prec, numerator)
}

/// HP version of the ξ_λ-weighted Mellin transform.
/// All arithmetic at `prec` bits. Returns (Re, Im) of G(s).
///
/// `gl_nodes` and `gl_weights` are pre-fetched GL nodes/weights for
/// `n_quad` points at `prec` bits. Callers must fetch these once before
/// any parallel section — passing them in prevents a cache-write race
/// when many rayon threads would otherwise call `gauss_legendre_nodes`
/// concurrently on a cache miss.
#[cfg(feature = "hp")]
pub fn xi_weighted_mellin_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    lambda: &rug::Float,
    xi_hp: &[rug::Float],
    n_modes: usize,
    gl_nodes: &[rug::Float],
    gl_weights: &[rug::Float],
) -> (rug::Float, rug::Float) {
    try_xi_weighted_mellin_hp(s_re, s_im, lambda, xi_hp, n_modes, gl_nodes, gl_weights)
        .unwrap_or_else(|_| {
            (
                rug::Float::with_val(lambda.prec(), rug::float::Special::Nan),
                rug::Float::with_val(lambda.prec(), rug::float::Special::Nan),
            )
        })
}

/// Checked point quadrature. Rejects invalid domains and nonfinite arithmetic.
/// This does not certify quadrature or truncation error.
/// Validates the supplied GL rule in O(n^2) arithmetic on each call.
/// The full 2N+1 coefficient vector must be finite and exactly even.
#[cfg(feature = "hp")]
pub fn try_xi_weighted_mellin_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    lambda: &rug::Float,
    xi_hp: &[rug::Float],
    n_modes: usize,
    gl_nodes: &[rug::Float],
    gl_weights: &[rug::Float],
) -> anyhow::Result<(rug::Float, rug::Float)> {
    validate_transform_hp(s_re, s_im, lambda, gl_nodes, gl_weights)?;
    anyhow::ensure!(
        lambda > &rug::Float::with_val(lambda.prec(), 1),
        "weighted Mellin requires lambda > 1"
    );
    let length = n_modes
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("Mellin mode dimension overflow"))?;
    anyhow::ensure!(
        xi_hp.len() == length && n_modes <= u32::MAX as usize,
        "Mellin coefficient shape or index range"
    );
    anyhow::ensure!(
        xi_hp
            .iter()
            .all(|v| v.is_finite() && v.prec() == lambda.prec()),
        "Mellin coefficients must be finite at the working precision"
    );
    anyhow::ensure!(
        (0..n_modes).all(|n| xi_hp[n] == xi_hp[length - 1 - n]),
        "Mellin cosine reconstruction requires an exactly even vector"
    );

    use rug::Float;

    let prec = lambda.prec();
    let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
    let two_pi = {
        let mut v = pi_v.clone();
        v *= 2u32;
        v
    };

    // L = 2 ln λ
    let l = {
        let mut v = lambda.clone().ln();
        v *= 2u32;
        v
    };
    let inv_sqrt_l = {
        let mut v = l.clone().sqrt();
        v = v.recip();
        v
    };

    let n_quad = gl_nodes.len();
    let nodes = gl_nodes;
    let weights = gl_weights;

    // Map GL nodes from [-1, 1] to [λ⁻¹, λ].
    let a = {
        let mut v = lambda.clone();
        v = v.recip();
        v
    };
    let b = lambda.clone();
    let mid = {
        let mut v = a.clone();
        v /= 2u32;
        v += Float::with_val(prec, &b / 2u32);
        v
    };
    let half_range = {
        let mut v = b.clone();
        v -= &a;
        v /= 2u32;
        v
    };

    // ξ components: ξ_0 = xi_hp[n_modes], ξ_n = xi_hp[n_modes + n]
    let xi_0 = &xi_hp[n_modes];

    let mut sum_re = Float::with_val(prec, 0);
    let mut sum_im = Float::with_val(prec, 0);

    for i in 0..n_quad {
        // u = mid + half_range * nodes[i]
        let mut u = half_range.clone();
        u *= &nodes[i];
        u += &mid;

        // quadrature weight scaled by half_range
        let mut w = weights[i].clone();
        w *= &half_range;

        // f_λ(u) = (1/√L) [ξ_0 + 2 Σ_{n=1}^N ξ_n cos(2π n log(λu)/L)]
        let log_lambda_u = {
            let mut v = lambda.clone().ln();
            v += u.clone().ln();
            v
        };
        let phase_base = {
            let mut v = two_pi.clone();
            v *= &log_lambda_u;
            v /= &l;
            v
        };
        let mut f_val = xi_0.clone();
        for n in 1..=n_modes {
            let mut phase = phase_base.clone();
            phase *= n as u32;
            let cos_val = phase.cos();
            let mut term = xi_hp[n_modes + n].clone();
            term *= &cos_val;
            term *= 2u32;
            f_val += &term;
        }
        f_val *= &inv_sqrt_l;

        let ln_u = u.clone().ln();
        let amplitude = mellin_amplitude_hp(s_re, &u);
        let phase_u = {
            let mut v = s_im.clone();
            v *= &ln_u;
            v
        };
        let cos_phase = phase_u.clone().cos();
        let sin_phase = phase_u.sin();

        // integrand = f_val * ω(u) * u^{s-1}
        let mut common = f_val;
        common *= &amplitude;

        let mut re_term = common.clone();
        re_term *= &cos_phase;
        re_term *= &w;
        sum_re += &re_term;

        let mut im_term = common;
        im_term *= &sin_phase;
        im_term *= &w;
        sum_im += &im_term;
    }

    anyhow::ensure!(
        sum_re.is_finite() && sum_im.is_finite(),
        "nonfinite Mellin quadrature result"
    );
    Ok((sum_re, sum_im))
}

/// HP version of the truncated Λ_λ (unweighted, for comparison).
///
/// `gl_nodes` and `gl_weights` are pre-fetched GL nodes/weights for
/// `n_quad` points at `prec` bits. Callers must fetch these once before
/// any parallel section — passing them in prevents a cache-write race
/// when many rayon threads would otherwise call `gauss_legendre_nodes`
/// concurrently on a cache miss.
#[cfg(feature = "hp")]
pub fn truncated_lambda_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    lambda: &rug::Float,
    gl_nodes: &[rug::Float],
    gl_weights: &[rug::Float],
) -> (rug::Float, rug::Float) {
    try_truncated_lambda_hp(s_re, s_im, lambda, gl_nodes, gl_weights).unwrap_or_else(|_| {
        (
            rug::Float::with_val(lambda.prec(), rug::float::Special::Nan),
            rug::Float::with_val(lambda.prec(), rug::float::Special::Nan),
        )
    })
}

/// Checked point quadrature. Rejects invalid domains and nonfinite arithmetic.
/// This does not certify quadrature or truncation error.
/// Validates the supplied GL rule in O(n^2) arithmetic on each call.
#[cfg(feature = "hp")]
pub fn try_truncated_lambda_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    lambda: &rug::Float,
    gl_nodes: &[rug::Float],
    gl_weights: &[rug::Float],
) -> anyhow::Result<(rug::Float, rug::Float)> {
    validate_transform_hp(s_re, s_im, lambda, gl_nodes, gl_weights)?;

    use rug::Float;

    let prec = lambda.prec();

    let n_quad = gl_nodes.len();
    let nodes = gl_nodes;
    let weights = gl_weights;

    let a = {
        let mut v = lambda.clone();
        v = v.recip();
        v
    };
    let b = lambda.clone();
    let mid = {
        let mut v = a.clone();
        v /= 2u32;
        v += Float::with_val(prec, &b / 2u32);
        v
    };
    let half_range = {
        let mut v = b.clone();
        v -= &a;
        v /= 2u32;
        v
    };

    let mut sum_re = Float::with_val(prec, 0);
    let mut sum_im = Float::with_val(prec, 0);

    for i in 0..n_quad {
        let mut u = half_range.clone();
        u *= &nodes[i];
        u += &mid;

        let mut w = weights[i].clone();
        w *= &half_range;

        let ln_u = u.clone().ln();
        let phase_u = {
            let mut v = s_im.clone();
            v *= &ln_u;
            v
        };
        let cos_phase = phase_u.clone().cos();
        let sin_phase = phase_u.sin();

        let common = mellin_amplitude_hp(s_re, &u);

        let mut re_term = common.clone();
        re_term *= &cos_phase;
        re_term *= &w;
        sum_re += &re_term;

        let mut im_term = common;
        im_term *= &sin_phase;
        im_term *= &w;
        sum_im += &im_term;
    }

    anyhow::ensure!(
        sum_re.is_finite() && sum_im.is_finite(),
        "nonfinite Mellin quadrature result"
    );
    Ok((sum_re, sum_im))
}

#[cfg(all(test, feature = "hp"))]
mod hp_tests {
    use super::*;
    use rug::Float;

    /// HP omega should match f64 omega to ~15 digits in the regime
    /// where f64 is accurate.
    #[test]
    fn omega_hp_matches_f64_in_safe_range() {
        let prec = 200;
        for &t in &[0.1f64, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0] {
            let t_hp = Float::with_val(prec, t);
            let v_hp = omega_hp(&t_hp);
            let v_f64 = omega_f64(t);
            let mut diff = v_hp.clone();
            diff -= Float::with_val(prec, v_f64);
            let abs_diff = diff.abs();
            let tol = Float::with_val(prec, Float::parse("1e-12").unwrap());
            assert!(
                abs_diff < tol,
                "omega_hp({}) = {} vs omega_f64 = {} (|diff| = {})",
                t,
                xc_numerics::fmt::display_hp(&v_hp, 20),
                v_f64,
                xc_numerics::fmt::display_hp(&abs_diff, 6)
            );
        }
    }

    /// HP omega at large t uses the asymptotic branch t·e^{-t}.
    /// Verify the truncated form matches the asymptotic form to high
    /// precision for large t (where eᵗ + 1 ≈ eᵗ).
    #[test]
    fn omega_hp_agrees_with_large_t_asymptotic_at_modest_precision() {
        let prec = 200;
        // Direct test: ω at t=499 should be ≈ 499·e^{-499} (truncated branch
        // gives the same value to high precision because eᵗ + 1 ≈ eᵗ for
        // large t).
        let t = Float::with_val(prec, 499);
        let v = omega_hp(&t);
        // Expected: 499 * e^{-499}.
        let mut neg_t = Float::with_val(prec, -499);
        neg_t = neg_t.exp();
        let mut expected = Float::with_val(prec, 499);
        expected *= &neg_t;
        // truncated branch: t·eᵗ/(1+eᵗ)² = t·e^{-t} / (1 + e^{-t})²
        // ≈ t·e^{-t} for large t.
        if let Some(rel) = xc_numerics::fmt::relative_difference(&v, &expected) {
            let tol = Float::with_val(prec, Float::parse("1e-100").unwrap());
            assert!(
                rel < tol,
                "ω(499) [truncated] should match t·e^{{-t}} closely; rel diff = {}",
                xc_numerics::fmt::display_hp(&rel, 6)
            );
        }
    }

    /// HP truncated_lambda_hp at s=2 should equal Γ(3)·η(2) = π²/6
    /// to many more digits than f64 (truncated at λ=50).
    #[test]
    #[ignore = "HP GL quadrature — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn truncated_lambda_hp_at_s2_matches_pi_sq_over_6() {
        let prec = 200;
        let s_re = Float::with_val(prec, 2);
        let s_im = Float::with_val(prec, 0);
        let lambda = Float::with_val(prec, 50);
        let (nodes, weights) = xc_numerics::quadrature::gauss_legendre_nodes(
            200,
            prec,
            xc_numerics::quadrature::CacheMode::default(),
        );
        let (re, im) = truncated_lambda_hp(&s_re, &s_im, &lambda, &nodes, &weights);
        let mut expected = Float::with_val(prec, rug::float::Constant::Pi);
        expected.square_mut();
        expected /= 6u32;
        // Same accuracy expectation as the f64 version: rel err < 1%.
        let rel = xc_numerics::fmt::relative_difference(&re, &expected).unwrap();
        let tol = Float::with_val(prec, Float::parse("0.01").unwrap());
        assert!(
            rel < tol,
            "Λ_50(2) = {} vs π²/6 = {}; rel err = {}",
            xc_numerics::fmt::display_hp(&re, 30),
            xc_numerics::fmt::display_hp(&expected, 30),
            xc_numerics::fmt::display_hp(&rel, 6)
        );
        // Imaginary part should be zero.
        assert!(im.clone().abs() < Float::with_val(prec, Float::parse("1e-50").unwrap()));
    }

    /// HP xi_weighted_mellin with a flat ξ vector should equal HP
    /// truncated_lambda (the weighting collapses to a constant).
    #[test]
    #[ignore = "HP GL quadrature — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn xi_weighted_mellin_hp_flat_xi_matches_unweighted() {
        let prec = 200;
        let lambda = Float::with_val(prec, 5);
        let n_modes: usize = 10;
        // L = 2 ln λ at HP
        let l = {
            let mut v = lambda.clone().ln();
            v *= 2u32;
            v
        };
        let sqrt_l = l.clone().sqrt();
        // Flat ξ: only ξ_0 = √L, rest zero. Then f_λ(u) = (1/√L)·√L = 1.
        let mut xi = vec![Float::with_val(prec, 0); 2 * n_modes + 1];
        xi[n_modes] = sqrt_l;

        let s_re = {
            let mut v = Float::with_val(prec, 1);
            v /= 2u32;
            v
        };
        let s_im = Float::with_val(prec, 14);

        let n_quad = 100;
        let (gl_nodes, gl_weights) = xc_numerics::quadrature::gauss_legendre_nodes(
            n_quad,
            prec,
            xc_numerics::quadrature::CacheMode::default(),
        );
        let (g_re, g_im) =
            xi_weighted_mellin_hp(&s_re, &s_im, &lambda, &xi, n_modes, &gl_nodes, &gl_weights);
        let (l_re, l_im) = truncated_lambda_hp(&s_re, &s_im, &lambda, &gl_nodes, &gl_weights);

        // With flat ξ, G(s) = Λ_λ(s).
        let mut re_diff = g_re.clone();
        re_diff -= &l_re;
        let abs_re = re_diff.abs();
        let tol = Float::with_val(prec, Float::parse("1e-50").unwrap());
        assert!(
            abs_re < tol,
            "G_re and Λ_re should match (|diff| = {})",
            xc_numerics::fmt::display_hp(&abs_re, 6)
        );
        let mut im_diff = g_im.clone();
        im_diff -= &l_im;
        assert!(im_diff.abs() < tol);
    }

    /// Check crossings of the quadrature real part, not complex or zeta zeros.
    #[test]
    #[ignore = "HP GL quadrature + parallel scan — GMP arena exhaustion in long debug test runs on WSL2; run with: RAYON_NUM_THREADS=2 cargo test --features hp -- --include-ignored --test-threads=1"]
    fn scan_real_crossings_hp_checks_the_claimed_component() {
        let prec = 100;
        let lambda = Float::with_val(prec, 50);
        let t_min = Float::with_val(prec, 10);
        let t_max = Float::with_val(prec, 20);

        // Precompute GL nodes once on the calling thread so the GL Newton
        // compute (fresh at this prec/npts combo) does not run inside
        // concurrent rayon tasks — that causes GMP allocation contention.
        let (nodes, weights) = xc_numerics::quadrature::gauss_legendre_nodes(
            200,
            prec,
            xc_numerics::quadrature::CacheMode::JsonZip,
        );

        let zeros = scan_critical_line_zeros_hp(
            &|sigma_hp, t_hp| truncated_lambda_hp(sigma_hp, t_hp, &lambda, &nodes, &weights),
            &t_min,
            &t_max,
            200,
            30,
        );

        assert!(!zeros.is_empty());
        for t in zeros {
            assert!(t >= t_min && t <= t_max);
            let (real, imaginary) =
                truncated_lambda_hp(&Float::with_val(prec, 0.5), &t, &lambda, &nodes, &weights);
            let tolerance =
                Float::with_val(prec, 1e-7) * (Float::with_val(prec, 1) + imaginary.abs());
            assert!(real.abs() <= tolerance);
        }
    }
}
