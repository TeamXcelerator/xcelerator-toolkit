// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Suzuki screw function `g(t)` (arXiv:2606.09096), high-precision.
//!
//! ```text
//!   g(t) = −4(e^{t/2}+e^{−t/2}−2)
//!        + Σ_{n ≤ exp|t|} (Λ(n)/n)(|t|−log n)
//!        − (|t|/2)(ψ(1/4) − log π)
//!        − (1/4)(Φ(1,2,1/4) − e^{−|t|/2} Φ(e^{−2|t|},2,1/4))
//! ```
//! with `Λ` von Mangoldt, `ψ` digamma, `Φ` Hurwitz–Lerch zeta. Closed-form
//! constants (all HP, from rug): `ψ(1/4) = −γ − π/2 − 3 ln 2`,
//! `Φ(1,2,1/4) = π² + 8·G` (G = Catalan).
//!
//! This is the continuous kernel of the localized Weil-form convolution
//! operator `G_a` in Suzuki's framework. It is provided for direct study
//! of the screw function (e.g. verifying the large-`|t|` prime-sum
//! asymptotic `S(t) ≈ ½t² − γ|t|`).
//!
//! NOTE on the plunge eigenvalue `λ_a` (the CCM floor `D_Primes = −log₁₀ λ_a`):
//! it is **not** computed here by Nyström/quadrature of `g`. Resolving an
//! exponentially small eigenvalue requires HP-*exact* matrix entries, which
//! kernel quadrature cannot deliver (its `~1/n²` error swamps `λ_a ~ 10^{−500}`
//! and breaks positivity). The plunge is computed by the HP-exact Galerkin
//! Weil form in [`crate::ccm::hp::weil_min_eigenvalue_hp`] (the same operator,
//! `A_a` = Friedrichs extension of the localized Weil form, Suzuki Thm 1.1),
//! which additionally supports an archimedean-only mode (prime sum off) for
//! the prefactor decomposition test.

#![cfg(feature = "hp")]

use rug::float::Constant;
use rug::ops::Pow;
use rug::Float;

use crate::ccm::prime_powers_up_to;

/// Precomputed screw-function evaluator at a fixed precision, valid for
/// `|t| ≤ 2·a_max` (caches the von Mangoldt terms with `n ≤ e^{2 a_max}`).
pub struct ScrewKernel {
    prec: u32,
    /// `−(ψ(1/4) − log π)/2`, the coefficient of `|t|`.
    dig_coef: Float,
    /// `Φ(1,2,1/4) = π² + 8·Catalan`.
    phi1: Float,
    /// von Mangoldt support: `(n = p^k, log p)` for `n ≤ e^{2 a_max}`.
    vm: Vec<(u64, Float)>,
}

impl ScrewKernel {
    /// Build the evaluator at `prec` bits. `a_max = log λ_max`; primes up to
    /// `e^{2 a_max}` (= `λ_max²`) are enumerated for the von Mangoldt sum.
    /// (`a_max` only sizes the prime list — an integer count — so f64 is fine
    /// here; all kernel arithmetic is HP.)
    pub fn new(a_max: f64, prec: u32) -> Self {
        let pi = Float::with_val(prec, Constant::Pi);
        let ln2 = Float::with_val(prec, Constant::Log2);
        let euler = Float::with_val(prec, Constant::Euler);
        let catalan = Float::with_val(prec, Constant::Catalan);

        // ψ(1/4) = −γ − π/2 − 3 ln2
        let mut psi_quarter = -Float::with_val(prec, &euler);
        {
            let mut half_pi = Float::with_val(prec, &pi);
            half_pi /= 2u32;
            psi_quarter -= &half_pi;
        }
        {
            let mut three_ln2 = Float::with_val(prec, &ln2);
            three_ln2 *= 3u32;
            psi_quarter -= &three_ln2;
        }
        // dig_coef = −(ψ(1/4) − log π)/2
        let ln_pi = Float::with_val(prec, &pi).ln();
        let mut dig_coef = Float::with_val(prec, &psi_quarter);
        dig_coef -= &ln_pi;
        dig_coef /= -2i32;

        // Φ(1,2,1/4) = π² + 8·Catalan
        let mut phi1 = Float::with_val(prec, &pi);
        phi1 *= &pi;
        {
            let mut eight_g = Float::with_val(prec, &catalan);
            eight_g *= 8u32;
            phi1 += &eight_g;
        }

        // von Mangoldt terms n = p^k ≤ e^{2 a_max} = λ_max²
        let bound = (a_max * 2.0).exp().floor() as u64 + 1;
        let vm: Vec<(u64, Float)> = prime_powers_up_to(bound)
            .into_iter()
            .map(|(pk, p, _k)| (pk, Float::with_val(prec, p).ln()))
            .collect();

        ScrewKernel { prec, dig_coef, phi1, vm }
    }

    /// Hurwitz–Lerch `Φ(z,2,1/4) = Σ_{k≥0} z^k/(k+1/4)²` for `0 ≤ z < 1`,
    /// summed to working precision.
    fn lerch_phi2(&self, z: &Float) -> Float {
        let prec = self.prec;
        let mut sum = Float::with_val(prec, 0);
        let mut zpow = Float::with_val(prec, 1); // z^k
        let tol = Float::with_val(prec, 2).pow(-((prec as i32) + 8));
        let mut k: u64 = 0;
        let max_terms: u64 = 50_000_000;
        loop {
            let mut denom = Float::with_val(prec, k);
            denom += Float::with_val(prec, 0.25);
            denom *= Float::with_val(prec, &denom);
            let mut term = Float::with_val(prec, &zpow);
            term /= &denom;
            sum += &term;
            zpow *= z;
            k += 1;
            if term.abs() < tol && k > 4 {
                break;
            }
            if k >= max_terms {
                break;
            }
        }
        sum
    }

    /// Evaluate `g(t)` at working precision. Even in `t`.
    pub fn eval(&self, t: &Float) -> Float {
        let prec = self.prec;
        let at = Float::with_val(prec, t).abs();
        if at.is_zero() {
            return Float::with_val(prec, 0);
        }

        // arch = −4(2 cosh(|t|/2) − 2)
        let mut half = Float::with_val(prec, &at);
        half /= 2u32;
        let cosh = half.cosh();
        let mut arch = Float::with_val(prec, &cosh);
        arch *= 2u32;
        arch -= 2u32;
        arch *= -4i32;

        // prime sum: Σ_{n ≤ e^{|t|}} (log p / n)(|t| − log n)
        let limit = Float::with_val(prec, &at).exp();
        let mut psum = Float::with_val(prec, 0);
        for (n, logp) in &self.vm {
            let nf = Float::with_val(prec, *n);
            if nf > limit {
                break;
            }
            let mut term = Float::with_val(prec, logp);
            term /= &nf;
            let mut bracket = Float::with_val(prec, &at);
            bracket -= Float::with_val(prec, *n).ln();
            term *= &bracket;
            psum += &term;
        }

        // digamma linear term
        let mut dig = Float::with_val(prec, &self.dig_coef);
        dig *= &at;

        // Lerch term
        let mut z = Float::with_val(prec, &at);
        z *= -2i32;
        z = z.exp();
        let phi2 = self.lerch_phi2(&z);
        let mut em = Float::with_val(prec, &at);
        em /= -2i32;
        em = em.exp();
        let mut lerch = Float::with_val(prec, &self.phi1);
        {
            let mut t2 = Float::with_val(prec, &em);
            t2 *= &phi2;
            lerch -= &t2;
        }
        lerch /= -4i32;

        let mut g = arch;
        g += &psum;
        g += &dig;
        g += &lerch;
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp(prec: u32, s: &str) -> Float {
        Float::with_val(prec, Float::parse(s).unwrap())
    }

    #[test]
    fn screw_even_and_zero_at_origin() {
        let prec = 200;
        let k = ScrewKernel::new(2.0, prec);
        assert!(k.eval(&Float::with_val(prec, 0)).is_zero());
        let t = hp(prec, "1.3");
        let mt = hp(prec, "-1.3");
        let d = k.eval(&t) - k.eval(&mt);
        assert!(d.abs() < Float::with_val(prec, 2).pow(-150));
    }

    #[test]
    fn screw_matches_reference_values() {
        // Reference values from an independent mpmath implementation (dps 45).
        let prec = 200;
        let k = ScrewKernel::new(2.0, prec);
        let cases = [
            ("0.1", "-0.05313043381777025410005234409146544184755"),
            ("1.3", "-0.1791568325469816597486516426489133658077"),
            ("2.5", "-1.67642704153778873062948748617520315582"),
        ];
        for (t_s, ref_s) in cases {
            let g = k.eval(&hp(prec, t_s));
            let r = hp(prec, ref_s);
            let diff = (g - r).abs();
            assert!(diff < hp(prec, "1e-38"), "g({}) mismatch: {}", t_s, diff);
        }
    }
}
