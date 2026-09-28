// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Computed Suzuki screw function g=-Psi, Suzuki (2023), Eq. (1.1):
//! <https://doi.org/10.1112/jlms.12785>. Prime weights are Lambda(n)/sqrt(n).
//! The evaluator is a finite-precision point calculation, not an enclosure.
//! Its cached prime support covers |t| <= 2*a_max. Near zero a convergent
//! expansion isolates the t*log(t) cusp; the direct Lerch series is used away
//! from zero with an explicit geometric tail stopping test.

use crate::ccm::try_prime_powers_up_to;
use anyhow::{bail, Result};
use rug::float::{Constant, Round};
use rug::ops::{DivAssignRound, MulAssignRound};
use rug::Float;

/// Earlier v1 code incorrectly used Lambda(n)/n and silently truncated the
/// Lerch sum. Results from that kernel do not represent Suzuki's function.
pub const SCREW_SEMANTICS_VERSION: &str = "suzuki-sqrt-prime-cusp-v2";

pub struct ScrewKernel {
    prec: u32,
    output_prec: u32,
    maximum_argument: Float,
    dig_coef: Float,
    phi1: Float,
    cusp_coef: Float,
    vm: Vec<(u64, Float)>,
}

impl ScrewKernel {
    /// Construct the point evaluator; panics on an invalid domain.
    /// Use `try_new` for fallible construction.
    pub fn new(a_max: f64, prec: u32) -> Self {
        Self::try_new(a_max, prec).expect("valid screw-function domain required")
    }

    pub fn try_new(a_max: f64, output_prec: u32) -> Result<Self> {
        if !a_max.is_finite() || a_max < 0.0 || !(32..=1_000_000).contains(&output_prec) {
            bail!("screw kernel requires finite nonnegative support and precision 32..=1000000");
        }
        let prec = output_prec + 64;
        let maximum_argument: Float = Float::with_val(prec, a_max) * 2u32;
        let mut limit: Float = maximum_argument.clone();
        limit.exp_round(Round::Up);
        let bound = limit
            .to_integer_round(Round::Up)
            .and_then(|(n, _)| n.to_u64())
            .ok_or_else(|| anyhow::anyhow!("screw prime support exceeds u64 range"))?;
        if usize::try_from(bound)
            .ok()
            .and_then(|n| n.checked_add(1))
            .is_none()
        {
            bail!("screw prime support exceeds the platform index range");
        }
        let pi = Float::with_val(prec, Constant::Pi);
        let ln2 = Float::with_val(prec, Constant::Log2);
        let euler = Float::with_val(prec, Constant::Euler);
        let mut dig_coef = Float::with_val(prec, &pi / 2);
        dig_coef += &euler;
        dig_coef += Float::with_val(prec, ln2 * 3);
        dig_coef += pi.clone().ln();
        dig_coef /= 2;
        let mut phi1 = Float::with_val(prec, &pi * &pi);
        phi1 += Float::with_val(prec, Constant::Catalan) * 8;
        let mut cusp_coef = Float::with_val(prec, &pi * 2).ln();
        cusp_coef += euler;
        cusp_coef -= 1;
        cusp_coef /= 2;
        let vm = try_prime_powers_up_to(bound)?
            .into_iter()
            .map(|(n, p, _)| (n, Float::with_val(prec, p).ln()))
            .collect();
        Ok(Self {
            prec,
            output_prec,
            maximum_argument,
            dig_coef,
            phi1,
            cusp_coef,
            vm,
        })
    }

    /// Evaluate with explicit support, arithmetic, and convergence errors.
    pub fn try_eval(&self, t: &Float) -> Result<Float> {
        if !t.is_finite() || t.clone().abs() > self.maximum_argument {
            bail!("screw argument is nonfinite or outside the cached prime support");
        }
        let at = Float::with_val(self.prec, t).abs();
        if at.is_zero() {
            return Ok(Float::with_val(self.output_prec, 0));
        }
        let mut g = if at <= 0.5 {
            self.local_cusp(&at)?
        } else {
            self.direct_archimedean(&at)?
        };
        let limit = at.clone().exp();
        for (n, logp) in &self.vm {
            let nf = Float::with_val(self.prec, *n);
            if nf > limit {
                break;
            }
            let mut term = Float::with_val(self.prec, logp / nf.clone().sqrt());
            term *= Float::with_val(self.prec, &at - nf.ln());
            g += term;
        }
        if !g.is_finite() {
            bail!("screw arithmetic is outside the finite exponent range");
        }
        let output = Float::with_val(self.output_prec, g);
        if !output.is_finite() {
            bail!("screw output cannot be represented");
        }
        Ok(output)
    }

    /// Compatibility point API. An invalid/out-of-support/unresolved evaluation
    /// returns NaN; callers requiring an error description should use try_eval.
    pub fn eval(&self, t: &Float) -> Float {
        self.try_eval(t)
            .unwrap_or_else(|_| Float::with_val(self.output_prec, rug::float::Special::Nan))
    }

    fn pole_term(&self, t: &Float) -> Float {
        // -8(cosh(t/2)-1) = -16*sinh(t/4)^2, avoiding cancellation at zero.
        let mut value = Float::with_val(self.prec, t / 4).sinh();
        value.square_mut();
        value *= -16;
        value
    }

    fn direct_archimedean(&self, t: &Float) -> Result<Float> {
        let z = Float::with_val(self.prec, t * -2).exp();
        if !(0..1).contains(&z) {
            bail!("invalid direct Lerch argument");
        }
        let one = Float::with_val(self.prec, 1);
        let gap = Float::with_val_round(self.prec, &one - &z, Round::Down).0;
        let tolerance = Float::with_val(self.prec, 1) >> (self.output_prec + 24);
        let mut sum = Float::with_val(self.prec, 0);
        let mut power = one.clone();
        let mut power_upper = one;
        let mut converged = false;
        for k in 0..50_000_000u64 {
            let denominator = Float::with_val(self.prec, k) + Float::with_val(self.prec, 0.25);
            let denominator = denominator.square();
            sum += Float::with_val(self.prec, &power / denominator);
            power *= &z;
            power_upper.mul_assign_round(&z, Round::Up);
            let next_denominator =
                Float::with_val(self.prec, k + 1) + Float::with_val(self.prec, 0.25);
            let next_denominator = next_denominator.square();
            let mut tail = power_upper.clone();
            tail.div_assign_round(next_denominator, Round::Up);
            tail.div_assign_round(&gap, Round::Up);
            if tail <= tolerance {
                converged = true;
                break;
            }
        }
        if !converged {
            bail!("Lerch series exhausted its term budget before establishing the tail tolerance");
        }
        let exponential = Float::with_val(self.prec, t / -2).exp();
        let mut lerch = Float::with_val(self.prec, exponential * sum);
        lerch -= &self.phi1;
        lerch /= 4;
        let mut value = self.pole_term(t);
        value += Float::with_val(self.prec, t * &self.dig_coef);
        value += lerch;
        Ok(value)
    }

    fn local_cusp(&self, t: &Float) -> Result<Float> {
        // H(z)=z*exp(-z/2)/(1-exp(-2z))=sum h_n*z^n.
        // H is analytic on |z|<=1 with |H|<2 there. Cauchy's estimate gives
        // |h_n|<=2. Integrating H(t)/t twice and matching Suzuki's linear
        // term gives g_arch=t*log(t)/2+A*t+pole+sum h_n*t^(n+1)/(n*(n+1)).
        // At 0<t<=1/2, the remainder after n is bounded by
        // 2*t^(n+2)/((n+1)*(n+2)*(1-t)). This controls truncation, not all
        // rounding errors of the computed point evaluation.
        let one = Float::with_val(self.prec, 1);
        let gap = Float::with_val_round(self.prec, &one - t, Round::Down).0;
        let mut target = Float::with_val(self.prec, 1) >> (self.output_prec + 24);
        target.mul_assign_round(t, Round::Down);
        if target.is_zero() {
            bail!("local screw tolerance is outside the exponent range");
        }
        let mut power_upper = Float::with_val_round(self.prec, t * t, Round::Up).0;
        let mut terms = 0usize;
        for n in 1..=50_000usize {
            power_upper.mul_assign_round(t, Round::Up);
            let mut tail = power_upper.clone();
            tail *= 2;
            tail.div_assign_round(((n + 1) * (n + 2)) as u64, Round::Up);
            tail.div_assign_round(&gap, Round::Up);
            if tail <= target {
                terms = n;
                break;
            }
        }
        if terms == 0 {
            bail!("local screw expansion exhausted its term budget");
        }
        let mut h = vec![Float::with_val(self.prec, 0.5)];
        let mut q = vec![Float::with_val(self.prec, 2)];
        let mut numerator = Float::with_val(self.prec, 1);
        let mut power = Float::with_val(self.prec, t * t);
        let mut correction = Float::with_val(self.prec, 0);
        for n in 1..=terms {
            numerator /= -((2 * n) as i64);
            let mut qn = q[n - 1].clone();
            qn *= -2;
            qn /= (n + 1) as u64;
            q.push(qn);
            let mut hn = numerator.clone();
            for j in 1..=n {
                hn -= Float::with_val(self.prec, &q[j] * &h[n - j]);
            }
            hn /= 2;
            let mut term = Float::with_val(self.prec, &hn * &power);
            term /= (n * (n + 1)) as u64;
            correction += term;
            h.push(hn);
            power *= t;
        }
        let mut value = t.clone().ln();
        value /= 2;
        value += &self.cusp_coef;
        value *= t;
        value += self.pole_term(t);
        value += correction;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;

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
        // Suzuki Eq. (1.1), checked independently with mpmath at 110 digits.
        // The old reference implemented the same incorrect 1/n prime weight.
        let prec = 200;
        let k = ScrewKernel::new(2.0, prec);
        let cases = [
            ("0.1", "-0.05313043381777025410005234409146544184755"),
            ("1.3", "-0.038051765345468303856907782341990310938341428574467447142839022070640833355433078883929453694652"),
            ("2.5", "-0.048434868673898385098113853248465872741517281005273632964992079581229773638501528956611409456007"),
        ];
        for (t_s, ref_s) in cases {
            let g = k.eval(&hp(prec, t_s));
            let r = hp(prec, ref_s);
            let diff = (g - r).abs();
            assert!(diff < hp(prec, "1e-38"), "g({}) mismatch: {}", t_s, diff);
        }
    }
}
