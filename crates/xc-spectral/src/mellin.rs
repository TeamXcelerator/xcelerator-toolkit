// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.
//

//! Mellin-side investigation: CCM vs naive Mellin truncation.
//!
//! Tests how much of the CCM construction's accuracy is captured by
//! integral transforms of ξ_λ. Result: not much. CCM gives 55-460
//! matching digits; naive Mellin truncation gives < 1 digit; ξ-weighted
//! Mellin gives ~2.6× over naive. The construction's power is algebraic.
//! Tests whether the CCM construction's eigenvalues relate to zeros of
//! the truncated completed eta function Λ_λ(s) = ∫_{λ⁻¹}^{λ} t^{s-1} ω(t) dt,
//! or to zeros of the ξ_λ-weighted variant G(s) = ∫ f_λ(u) ω(u) u^{s-1} du.
//!
//! ## Background
//!
//! Yakaboylu (2408.15135) shows that the full completed eta function
//! Λ(s) = ∫_0^∞ t^{s-1} ω(t) dt = Γ(s+1)·η(s) has zeros at the
//! nontrivial Riemann zeros (plus periodic eta zeros). The truncated
//! version Λ_λ(s) should approximate Λ(s) as λ → ∞, with its zeros
//! approaching the Riemann zeros.
//!
//! The question: do the zeros of Λ_λ(s) match our CCM eigenvalues?
//! If yes, we have a direct Mellin-side bridge.


/// ω(t) = t·e^t / (1 + e^t)² — the kernel of the completed eta function.
/// Well-behaved for t > 0; decays exponentially for large t.
#[inline]
pub fn omega(t: f64) -> f64 {
    if t > 500.0 {
        // For very large t: ω(t) ≈ t·e^{-t}
        return t * (-t).exp();
    }
    let et = t.exp();
    t * et / (1.0 + et).powi(2)
}

/// Evaluate the truncated completed eta function at complex s = σ + it:
///
/// Λ_λ(s) = ∫_{λ⁻¹}^{λ} u^{s-1} · ω(u) du
///
/// Uses Gauss-Legendre quadrature at f64.
/// Returns (real part, imaginary part).
pub fn truncated_lambda(
    s_re: f64, s_im: f64,
    lambda: f64,
    n_quad: usize,
) -> (f64, f64) {
    // Gauss-Legendre on [λ⁻¹, λ] mapped from [-1, 1].
    let a = 1.0 / lambda;
    let b = lambda;
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);

    let (nodes, weights) = gauss_legendre_f64(n_quad);

    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for i in 0..n_quad {
        let u = mid + half * nodes[i];
        let w = weights[i] * half;
        // u^{s-1} = u^{σ-1} · exp(i·t·ln u)
        let ln_u = u.ln();
        let u_pow_re = u.powf(s_re - 1.0); // u^{σ-1}
        let phase = s_im * ln_u;
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();
        let omega_u = omega(u);
        let integrand_re = u_pow_re * cos_phase * omega_u;
        let integrand_im = u_pow_re * sin_phase * omega_u;
        sum_re += w * integrand_re;
        sum_im += w * integrand_im;
    }
    (sum_re, sum_im)
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
pub fn xi_weighted_mellin(
    s_re: f64, s_im: f64,
    lambda: f64,
    xi: &[f64],  // length 2N+1, indexed j = -N..N with xi[N] = ξ_0
    n_modes: usize,
    n_quad: usize,
) -> (f64, f64) {
    let l = (lambda * lambda).ln();
    let inv_sqrt_l = 1.0 / l.sqrt();
    let a = 1.0 / lambda;
    let b = lambda;
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);

    let (nodes, weights) = gauss_legendre_f64(n_quad);

    let xi_0 = xi[n_modes];
    let xi_pos: Vec<f64> = (1..=n_modes).map(|n| xi[n_modes + n]).collect();

    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for i in 0..n_quad {
        let u = mid + half * nodes[i];
        let w = weights[i] * half;

        // f_λ(u) via cosine reconstruction
        let phase_base = 2.0 * std::f64::consts::PI * (lambda * u).ln() / l;
        let mut f_val = xi_0;
        for n in 1..=n_modes {
            f_val += 2.0 * xi_pos[n - 1] * (n as f64 * phase_base).cos();
        }
        f_val *= inv_sqrt_l;

        // u^{s-1} · ω(u)
        let ln_u = u.ln();
        let u_pow_re = u.powf(s_re - 1.0);
        let phase = s_im * ln_u;
        let cos_phase = phase.cos();
        let sin_phase = phase.sin();
        let omega_u = omega(u);

        let integrand_re = f_val * u_pow_re * cos_phase * omega_u;
        let integrand_im = f_val * u_pow_re * sin_phase * omega_u;
        sum_re += w * integrand_re;
        sum_im += w * integrand_im;
    }
    (sum_re, sum_im)
}

/// Find zeros of a complex function on the critical line Re(s) = 1/2
/// by scanning Im(s) and looking for sign changes in the real part.
/// Returns approximate locations of zeros.
///
/// The scan evaluations are computed in parallel via rayon.
/// `eval_fn` must be `Sync` (true for pure-math closures with no
/// shared mutable state).
pub fn scan_critical_line_zeros<F>(
    eval_fn: &F,
    t_min: f64,
    t_max: f64,
    n_scan: usize,
) -> Vec<f64>
where
    F: Fn(f64, f64) -> (f64, f64) + Sync,
{
    use rayon::prelude::*;
    let dt = (t_max - t_min) / (n_scan as f64);

    // Parallel scan: evaluate Re(eval_fn(0.5, t_i)) at each grid point.
    let re_values: Vec<f64> = (0..=n_scan)
        .into_par_iter()
        .map(|i| {
            let t = t_min + (i as f64) * dt;
            eval_fn(0.5, t).0
        })
        .collect();

    // Sequential scan for sign changes (cheap), then per-zero bisection.
    // Bisections themselves are sequential but only happen at sign changes
    // (typically O(N/log N) of them, not all N).
    let mut zeros = Vec::new();
    for i in 1..=n_scan {
        if re_values[i - 1] * re_values[i] < 0.0 {
            let prev_t = t_min + ((i - 1) as f64) * dt;
            let t = t_min + (i as f64) * dt;
            let zero_t = bisect_zero(eval_fn, prev_t, t, 50);
            zeros.push(zero_t);
        }
    }
    zeros
}

fn bisect_zero<F>(
    eval_fn: &F,
    mut a: f64, mut b: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(f64, f64) -> (f64, f64) + Sync,
{
    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        let (fa, _) = eval_fn(0.5, a);
        let (fm, _) = eval_fn(0.5, mid);
        if fa * fm < 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }
    0.5 * (a + b)
}

/// Simple Gauss-Legendre nodes and weights at f64 for moderate n.
/// Uses the standard Newton iteration on Legendre polynomials.
fn gauss_legendre_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    for k in 0..n {
        // Initial guess
        let mut x = ((4 * k + 3) as f64 * std::f64::consts::PI / (4 * n + 2) as f64).cos();
        // Newton on P_n(x)
        for _ in 0..20 {
            let (pn, pn_prime) = legendre_p_deriv(n, x);
            x -= pn / pn_prime;
        }
        let (_, pn_prime) = legendre_p_deriv(n, x);
        nodes[k] = x;
        weights[k] = 2.0 / ((1.0 - x * x) * pn_prime * pn_prime);
    }
    (nodes, weights)
}

fn legendre_p_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 { return (1.0, 0.0); }
    let mut p0 = 1.0_f64;
    let mut p1 = x;
    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
        p0 = p1;
        p1 = p_next;
    }
    let deriv = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, deriv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ω(t) should be positive for t > 0 and peak near t ≈ 1.
    #[test]
    fn omega_is_positive() {
        for &t in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
            assert!(omega(t) > 0.0, "omega({}) should be positive", t);
        }
    }

    /// The full Λ(s) = ∫_0^∞ t^{s-1} ω(t) dt should equal Γ(s+1)·η(s).
    /// At s = 2: Γ(3)·η(2) = 2 · π²/12 = π²/6 ≈ 1.6449.
    /// We can't integrate to ∞ at f64, but truncated at λ=50 should be close.
    #[test]
    fn truncated_lambda_at_s2_close_to_gamma3_eta2() {
        let (re, im) = truncated_lambda(2.0, 0.0, 50.0, 200);
        let expected = std::f64::consts::PI.powi(2) / 6.0;
        let rel_err = (re - expected).abs() / expected;
        assert!(rel_err < 0.01, "Λ_50(2) = {} should be close to π²/6 = {} (rel err {})",
            re, expected, rel_err);
        assert!(im.abs() < 1e-10);
    }

    /// Λ_λ(s) should have a zero near the first Riemann zero (s = 1/2 + i·14.13)
    /// for large enough λ.
    #[test]
    fn truncated_lambda_has_zero_near_first_riemann_zero() {
        let lambda = 50.0;
        let zeros = scan_critical_line_zeros(
            &|sigma, t| truncated_lambda(sigma, t, lambda, 200),
            10.0, 20.0, 1000,
        );
        // Should find at least one zero near 14.13.
        let first_riemann = 14.134725141734695;
        let closest = zeros.iter()
            .map(|&z| (z - first_riemann).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest < 1.0,
            "should find a zero near 14.13 (closest was {:.4} away, zeros found: {:?})",
            closest, zeros
        );
    }

    /// Comprehensive scan: find zeros of Λ_λ on the critical line for
    /// λ = √13 (our standard CCM config) and compare to first 10 Riemann zeros.
    #[test]
    fn truncated_lambda_zeros_vs_riemann_at_lambda_sqrt13() {
        let lambda = 13.0_f64.sqrt();
        let riemann_zeros = vec![
            14.134725141734695, 21.022039638771556, 25.010857580145687,
            30.424876125859512, 32.935061587739189, 37.586178158825675,
            40.918719012147500, 43.327073280915000, 48.005150881167160,
            49.773832477672300,
        ];
        let zeros = scan_critical_line_zeros(
            &|sigma, t| truncated_lambda(sigma, t, lambda, 300),
            5.0, 55.0, 5000,
        );
        eprintln!("\nΛ_λ zeros on critical line (λ = √13 ≈ {:.4}):", lambda);
        eprintln!("{:>5} {:>15} {:>15} {:>12}", "k", "Λ_λ zero", "Riemann zero", "difference");
        for (i, &rz) in riemann_zeros.iter().enumerate() {
            let closest = zeros.iter()
                .map(|&z| (z, (z - rz).abs()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((z, diff)) = closest {
                eprintln!("{:>5} {:>15.6} {:>15.6} {:>12.4e}", i + 1, z, rz, diff);
            } else {
                eprintln!("{:>5} {:>15} {:>15.6} {:>12}", i + 1, "NOT FOUND", rz, "—");
            }
        }
        eprintln!("Total zeros found in [5, 55]: {}", zeros.len());
        // At λ=√13, the truncation is severe (interval [0.277, 3.606]).
        // We may not find all zeros. Just check we find at least some.
        assert!(!zeros.is_empty(), "should find at least one zero");
    }

    /// Same scan at λ = 10 (λ² = 100) — larger interval, should be closer.
    #[test]
    fn truncated_lambda_zeros_vs_riemann_at_lambda_10() {
        let lambda = 10.0;
        let riemann_zeros = vec![
            14.134725141734695, 21.022039638771556, 25.010857580145687,
            30.424876125859512, 32.935061587739189,
        ];
        let zeros = scan_critical_line_zeros(
            &|sigma, t| truncated_lambda(sigma, t, lambda, 300),
            10.0, 40.0, 3000,
        );
        eprintln!("\nΛ_λ zeros on critical line (λ = 10):");
        eprintln!("{:>5} {:>15} {:>15} {:>12}", "k", "Λ_λ zero", "Riemann zero", "difference");
        for (i, &rz) in riemann_zeros.iter().enumerate() {
            let closest = zeros.iter()
                .map(|&z| (z, (z - rz).abs()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((z, diff)) = closest {
                eprintln!("{:>5} {:>15.6} {:>15.6} {:>12.4e}", i + 1, z, rz, diff);
            } else {
                eprintln!("{:>5} {:>15} {:>15.6} {:>12}", i + 1, "NOT FOUND", rz, "—");
            }
        }
        eprintln!("Total zeros found in [10, 40]: {}", zeros.len());
        // At λ=10, should find zeros close to Riemann zeros.
        let first_riemann = 14.134725141734695;
        let closest_to_first = zeros.iter()
            .map(|&z| (z - first_riemann).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest_to_first < 0.5,
            "should find a zero within 0.5 of 14.13 (closest was {:.4})",
            closest_to_first
        );
    }

    /// Idea 2: ξ_λ-weighted Mellin G(s) = ∫ f_λ(u)·ω(u)·u^{s-1} du.
    /// Compare its zeros to Riemann zeros. If weighting by ξ_λ improves
    /// accuracy over the unweighted Λ_λ, that's evidence of a Mellin-side
    /// bridge.
    #[test]
    fn xi_weighted_mellin_zeros_at_lambda_sqrt13() {
        // Run CCM at f64 to get ξ_λ.
        let lambda = 13.0_f64.sqrt();
        let n_modes = 120;
        let params = crate::ccm::CcmParams::from_lambda(lambda, n_modes);
        let result = crate::ccm::run_f64(&params).unwrap();
        let xi = &result.xi;

        let riemann_zeros = vec![
            14.134725141734695, 21.022039638771556, 25.010857580145687,
            30.424876125859512, 32.935061587739189, 37.586178158825675,
            40.918719012147500, 43.327073280915000, 48.005150881167160,
            49.773832477672300,
        ];

        // Scan zeros of G(s) on the critical line.
        let zeros = scan_critical_line_zeros(
            &|sigma, t| xi_weighted_mellin(sigma, t, lambda, xi, n_modes, 300),
            5.0, 55.0, 5000,
        );

        eprintln!("\nG(s) = ∫ f_λ·ω·u^{{s-1}} zeros on critical line (λ = √13):");
        eprintln!("{:>5} {:>15} {:>15} {:>12}", "k", "G zero", "Riemann zero", "difference");
        for (i, &rz) in riemann_zeros.iter().enumerate() {
            let closest = zeros.iter()
                .map(|&z| (z, (z - rz).abs()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((z, diff)) = closest {
                eprintln!("{:>5} {:>15.6} {:>15.6} {:>12.4e}", i + 1, z, rz, diff);
            } else {
                eprintln!("{:>5} {:>15} {:>15.6} {:>12}", i + 1, "NOT FOUND", rz, "—");
            }
        }
        eprintln!("Total G zeros found in [5, 55]: {}", zeros.len());
        // Just check it runs and finds some zeros.
        assert!(!zeros.is_empty(), "should find at least one zero");
    }

    /// Fast test for xi_weighted_mellin with a synthetic flat ξ vector.
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
        let (g_re, g_im) = xi_weighted_mellin(0.5, t, lambda, &xi, n_modes, 100);
        let (l_re, l_im) = truncated_lambda(0.5, t, lambda, 100);

        // With flat ξ, G(s) = Λ_λ(s) (the weighting is constant 1).
        let re_diff = (g_re - l_re).abs();
        let im_diff = (g_im - l_im).abs();
        assert!(re_diff < 1e-10, "G and Λ_λ should match (re diff: {:.2e})", re_diff);
        assert!(im_diff < 1e-10, "G and Λ_λ should match (im diff: {:.2e})", im_diff);
    }

    /// omega(t) should peak near t = 1 and decay for large t.
    #[test]
    fn omega_peaks_near_one() {
        let peak = omega(1.0);
        assert!(omega(0.1) < peak);
        assert!(omega(5.0) < peak);
        assert!(omega(10.0) < omega(5.0)); // monotone decay for t > 1
    }
}


// ===========================================================================
// High-precision Mellin computation (requires ccm-rug feature)
// ===========================================================================

/// HP version of the ξ_λ-weighted Mellin transform.
/// All arithmetic at `prec` bits. Returns (Re, Im) of G(s).
#[cfg(feature = "hp")]
pub fn xi_weighted_mellin_hp(
    s_re: &rug::Float, s_im: &rug::Float,
    lambda: &rug::Float,
    xi_hp: &[rug::Float],
    n_modes: usize,
    n_quad: usize,
) -> (rug::Float, rug::Float) {
    use rug::Float;
    use rug::ops::Pow;

    let prec = lambda.prec();
    let one = Float::with_val(prec, 1.0);
    let pi_v = Float::with_val(prec, rug::float::Constant::Pi);
    let two_pi = {
        let mut v = pi_v.clone(); v *= 2u32; v
    };

    // L = 2 ln λ
    let l = {
        let mut v = lambda.clone().ln(); v *= 2u32; v
    };
    let inv_sqrt_l = {
        let mut v = l.clone().sqrt(); v = v.recip(); v
    };

    // GL nodes and weights on [-1, 1] at HP.
    // Use the cached version from highprec if available, otherwise compute.
    let (nodes, weights) = xc_numerics::quadrature::gauss_legendre_nodes(n_quad, prec);

    // Map GL nodes from [-1, 1] to [λ⁻¹, λ].
    let a = {
        let mut v = lambda.clone(); v = v.recip(); v
    };
    let b = lambda.clone();
    let mid = {
        let mut v = a.clone(); v += &b; v /= 2u32; v
    };
    let half_range = {
        let mut v = b.clone(); v -= &a; v /= 2u32; v
    };

    // ξ components: ξ_0 = xi_hp[n_modes], ξ_n = xi_hp[n_modes + n]
    let xi_0 = &xi_hp[n_modes];

    let mut sum_re = Float::with_val(prec, 0.0);
    let mut sum_im = Float::with_val(prec, 0.0);

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
            let mut v = lambda.clone();
            v *= &u;
            v.ln()
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

        // ω(u) = u·eᵘ/(1+eᵘ)²
        let omega_u = {
            let eu = u.clone().exp();
            let mut denom = eu.clone();
            denom += &one;
            denom.square_mut();
            let mut v = u.clone();
            v *= &eu;
            v /= &denom;
            v
        };

        // u^{s-1} = u^{σ-1} · exp(i·t·ln u)
        // Real part: u^{σ-1} · cos(t·ln u)
        // Imag part: u^{σ-1} · sin(t·ln u)
        let ln_u = u.clone().ln();
        let u_pow_sigma_minus_1 = {
            let mut exp = s_re.clone();
            exp -= &one;
            u.clone().pow(exp)
        };
        let phase_u = {
            let mut v = s_im.clone();
            v *= &ln_u;
            v
        };
        let cos_phase = phase_u.clone().cos();
        let sin_phase = phase_u.sin();

        // integrand = f_val * ω(u) * u^{s-1}
        let mut common = f_val;
        common *= &omega_u;
        common *= &u_pow_sigma_minus_1;

        let mut re_term = common.clone();
        re_term *= &cos_phase;
        re_term *= &w;
        sum_re += &re_term;

        let mut im_term = common;
        im_term *= &sin_phase;
        im_term *= &w;
        sum_im += &im_term;
    }

    (sum_re, sum_im)
}

/// HP version of the truncated Λ_λ (unweighted, for comparison).
#[cfg(feature = "hp")]
pub fn truncated_lambda_hp(
    s_re: &rug::Float, s_im: &rug::Float,
    lambda: &rug::Float,
    n_quad: usize,
) -> (rug::Float, rug::Float) {
    use rug::Float;
    use rug::ops::Pow;

    let prec = lambda.prec();
    let one = Float::with_val(prec, 1.0);

    let (nodes, weights) = xc_numerics::quadrature::gauss_legendre_nodes(n_quad, prec);

    let a = { let mut v = lambda.clone(); v = v.recip(); v };
    let b = lambda.clone();
    let mid = { let mut v = a.clone(); v += &b; v /= 2u32; v };
    let half_range = { let mut v = b.clone(); v -= &a; v /= 2u32; v };

    let mut sum_re = Float::with_val(prec, 0.0);
    let mut sum_im = Float::with_val(prec, 0.0);

    for i in 0..n_quad {
        let mut u = half_range.clone();
        u *= &nodes[i];
        u += &mid;

        let mut w = weights[i].clone();
        w *= &half_range;

        // ω(u)
        let omega_u = {
            let eu = u.clone().exp();
            let mut denom = eu.clone();
            denom += &one;
            denom.square_mut();
            let mut v = u.clone();
            v *= &eu;
            v /= &denom;
            v
        };

        // u^{s-1}
        let ln_u = u.clone().ln();
        let u_pow = {
            let mut exp = s_re.clone();
            exp -= &one;
            u.clone().pow(exp)
        };
        let phase_u = { let mut v = s_im.clone(); v *= &ln_u; v };
        let cos_phase = phase_u.clone().cos();
        let sin_phase = phase_u.sin();

        let mut common = omega_u;
        common *= &u_pow;

        let mut re_term = common.clone();
        re_term *= &cos_phase;
        re_term *= &w;
        sum_re += &re_term;

        let mut im_term = common;
        im_term *= &sin_phase;
        im_term *= &w;
        sum_im += &im_term;
    }

    (sum_re, sum_im)
}
