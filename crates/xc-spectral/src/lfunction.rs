//! L-function specifications for the generalized CCM construction.
//!
//! This module extends the CCM construction (originally defined for the
//! Riemann zeta function) to general L-functions in the Selberg class.
//! For Phase 4, we focus on Dirichlet L-functions L(s, χ) where χ is
//! a Dirichlet character modulo q.
//!
//! ## Mathematical setup
//!
//! For a Dirichlet character χ mod q, the L-function is
//!
//! ```text
//! L(s, χ) = ∏_p (1 − χ(p) p^{-s})^{-1}
//! ```
//!
//! Its logarithmic derivative gives the twisted von Mangoldt function:
//!
//! ```text
//! −L'(s,χ)/L(s,χ) = Σ_n χ(n) Λ(n) n^{-s}
//! ```
//!
//! where Λ(n) = log p if n = p^k, else 0.
//!
//! The CCM Weil quadratic form's prime-power sum becomes
//!
//! ```text
//! W_p^χ(V_n, V_m) = Σ_{k = p^j ≤ λ²} χ(p)^j · log(p) · k^{-1/2} · q(U_n, U_m)(log k)
//! ```
//!
//! with the convention χ(p)^j = 0 if gcd(p, q) > 1 (i.e. χ(p) = 0).
//!
//! ## Stage 1 scope
//!
//! For Stage 1 we restrict to **real characters** so the matrix stays
//! real-symmetric (matches existing infrastructure). Real characters
//! mod q are exactly the Kronecker symbols (Legendre symbol mod p
//! for prime p, plus a few others). All χ(n) ∈ {-1, 0, +1}.
//!
//! Examples:
//! - χ_0 mod 1: trivial character, χ(n) = 1 for all n. Recovers ζ.
//! - χ_3: Legendre (n/3). χ(0) = 0, χ(1) = 1, χ(2) = -1.
//! - χ_4: real quadratic character mod 4. χ(1) = 1, χ(3) = -1.
//! - χ_5 (real): Legendre (n/5). χ(0)=0, χ(1)=1, χ(2)=-1, χ(3)=-1, χ(4)=1.
//! - χ_7 (real): Legendre (n/7). χ values: 0, 1, 1, -1, 1, -1, -1.

use serde::{Deserialize, Serialize};

/// Specification of a Dirichlet L-function via its character data.
///
/// The character is given by its values χ(0), χ(1), …, χ(q-1). Values
/// repeat with period q: χ(n) = χ(n mod q). For real characters all
/// values are in {-1, 0, 1}; for complex characters they're roots of
/// unity and we'd need a Float64-typed variant (Stage 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LFunctionSpec {
    /// Modulus of the character. `χ(n)` has period `modulus`.
    pub modulus: u64,
    /// Character values χ(0), χ(1), …, χ(modulus - 1).
    /// For Stage 1 (real characters) all values are in {-1, 0, 1},
    /// stored as i8 to keep things compact.
    pub chi: Vec<i8>,
    /// Parity: 0 = even (χ(-1) = +1), 1 = odd (χ(-1) = -1).
    /// Determines the gamma factor in the functional equation.
    pub parity: u8,
    /// A short human-readable label (e.g. "zeta", "chi_3", "chi_5_real").
    pub label: String,
}

impl LFunctionSpec {
    /// The trivial character mod 1 — recovers L(s, χ_0) = ζ(s).
    pub fn riemann_zeta() -> Self {
        Self {
            modulus: 1,
            chi: vec![1],
            parity: 0,
            label: "zeta".to_string(),
        }
    }

    /// The unique non-trivial character mod 3, χ_3.
    /// χ(0)=0, χ(1)=1, χ(2)=-1. Odd parity (χ(-1)=χ(2)=-1).
    pub fn chi_3() -> Self {
        Self {
            modulus: 3,
            chi: vec![0, 1, -1],
            parity: 1,
            label: "chi_3".to_string(),
        }
    }

    /// The unique non-trivial character mod 4, χ_4.
    /// χ(0)=0, χ(1)=1, χ(2)=0, χ(3)=-1. Odd.
    pub fn chi_4() -> Self {
        Self {
            modulus: 4,
            chi: vec![0, 1, 0, -1],
            parity: 1,
            label: "chi_4".to_string(),
        }
    }

    /// Real quadratic character mod 5 (Legendre). Even (χ(-1)=χ(4)=1).
    /// χ values: 0, 1, -1, -1, 1.
    pub fn chi_5_real() -> Self {
        Self {
            modulus: 5,
            chi: vec![0, 1, -1, -1, 1],
            parity: 0,
            label: "chi_5_real".to_string(),
        }
    }

    /// Legendre character mod 7. Odd (χ(-1)=χ(6)=-1).
    /// χ values: 0, 1, 1, -1, 1, -1, -1.
    pub fn chi_7() -> Self {
        Self {
            modulus: 7,
            chi: vec![0, 1, 1, -1, 1, -1, -1],
            parity: 1,
            label: "chi_7".to_string(),
        }
    }

    /// Lookup by label for CLI use.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "zeta" | "riemann" => Some(Self::riemann_zeta()),
            "chi_3" => Some(Self::chi_3()),
            "chi_4" => Some(Self::chi_4()),
            "chi_5_real" | "chi_5" => Some(Self::chi_5_real()),
            "chi_7" => Some(Self::chi_7()),
            _ => None,
        }
    }

    /// All built-in specs (for sweeps and tests).
    pub fn builtin_all() -> Vec<Self> {
        vec![
            Self::riemann_zeta(),
            Self::chi_3(),
            Self::chi_4(),
            Self::chi_5_real(),
            Self::chi_7(),
        ]
    }

    /// Evaluate χ(n). For Stage 1, returns -1/0/1 as f64.
    #[inline]
    pub fn chi_at(&self, n: u64) -> f64 {
        let idx = (n % self.modulus) as usize;
        self.chi[idx] as f64
    }

    /// Compute χ(p^j) = χ(p)^j. Returns 0 if χ(p) = 0.
    #[inline]
    pub fn chi_at_prime_power(&self, p: u64, j: u32) -> f64 {
        let chi_p = self.chi_at(p);
        if chi_p == 0.0 {
            0.0
        } else {
            // For real characters χ(p) ∈ {-1, +1}, so χ(p)^j is also ±1.
            chi_p.powi(j as i32)
        }
    }

    /// True iff χ takes only real values (Stage 1 supports only these).
    pub fn is_real(&self) -> bool {
        self.chi.iter().all(|&c| (-1..=1).contains(&c))
    }

    /// True iff χ is the trivial (principal) character — i.e. recovers ζ.
    pub fn is_trivial(&self) -> bool {
        self.modulus == 1
    }

    /// True iff χ is even (χ(-1) = +1). Determines gamma factor.
    pub fn is_even(&self) -> bool {
        self.parity == 0
    }
}

/// Enumerate prime powers `n = p^j` with `1 < n ≤ bound`, returning
/// `(n, log p, χ(p)^j)` for use in the L-function-twisted Weil form.
///
/// This is the L-function generalization of `prime_powers_up_to`.
pub fn prime_powers_up_to_chi(bound: u64, spec: &LFunctionSpec) -> Vec<(u64, f64, f64)> {
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
    for p in 2..=n {
        if !sieve[p] {
            continue;
        }
        let log_p = (p as f64).ln();
        let mut k: u64 = p as u64;
        let mut j: u32 = 1;
        while k <= bound {
            let chi_pj = spec.chi_at_prime_power(p as u64, j);
            // Only include prime powers that contribute non-zero — but
            // we keep them in for now to preserve indexing matching with
            // the zeta case. Callers can skip χ=0 entries.
            out.push((k, log_p, chi_pj));
            if k > bound / (p as u64) {
                break;
            }
            k *= p as u64;
            j += 1;
        }
    }
    out.sort_unstable_by_key(|&(n, _, _)| n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeta_is_trivial() {
        let z = LFunctionSpec::riemann_zeta();
        assert!(z.is_trivial());
        assert_eq!(z.chi_at(0), 1.0);
        assert_eq!(z.chi_at(2), 1.0);
        assert_eq!(z.chi_at(100), 1.0);
        assert_eq!(z.chi_at_prime_power(2, 5), 1.0);
    }

    #[test]
    fn chi_3_values() {
        let c = LFunctionSpec::chi_3();
        assert_eq!(c.chi_at(0), 0.0); // gcd(0, 3) > 1
        assert_eq!(c.chi_at(1), 1.0);
        assert_eq!(c.chi_at(2), -1.0);
        assert_eq!(c.chi_at(3), 0.0);
        assert_eq!(c.chi_at(4), 1.0); // 4 mod 3 = 1
        assert_eq!(c.chi_at(5), -1.0); // 5 mod 3 = 2
        assert!(!c.is_trivial());
        assert!(c.is_real());
    }

    #[test]
    fn chi_3_prime_powers() {
        let c = LFunctionSpec::chi_3();
        // p=2 gives χ(2)=-1, so χ(2^j) = (-1)^j
        assert_eq!(c.chi_at_prime_power(2, 1), -1.0);
        assert_eq!(c.chi_at_prime_power(2, 2), 1.0);
        assert_eq!(c.chi_at_prime_power(2, 3), -1.0);
        // p=3 gives χ(3)=0, so all powers are 0
        assert_eq!(c.chi_at_prime_power(3, 1), 0.0);
        assert_eq!(c.chi_at_prime_power(3, 2), 0.0);
        // p=5 gives χ(5)=χ(2)=-1
        assert_eq!(c.chi_at_prime_power(5, 1), -1.0);
        assert_eq!(c.chi_at_prime_power(5, 2), 1.0);
    }

    #[test]
    fn enumerate_prime_powers_zeta_matches_existing() {
        let zeta = LFunctionSpec::riemann_zeta();
        let pp = prime_powers_up_to_chi(13, &zeta);
        let ks: Vec<u64> = pp.iter().map(|&(k, _, _)| k).collect();
        assert_eq!(ks, vec![2, 3, 4, 5, 7, 8, 9, 11, 13]);
        // For zeta, all chi values should be 1.
        assert!(pp.iter().all(|&(_, _, c)| c == 1.0));
    }

    #[test]
    fn enumerate_prime_powers_chi_3() {
        let c = LFunctionSpec::chi_3();
        let pp = prime_powers_up_to_chi(13, &c);
        // Same prime powers as before; chi values differ.
        let pairs: Vec<(u64, f64)> = pp.iter().map(|&(k, _, x)| (k, x)).collect();
        // χ(2) = -1 ⇒ χ(2)=-1, χ(4)=1, χ(8)=-1
        // χ(3) = 0 ⇒ χ(3)=0, χ(9)=0
        // χ(5) = -1, χ(7) = 1, χ(11) = -1, χ(13) = 1
        assert_eq!(pairs[0], (2, -1.0));
        assert_eq!(pairs[1], (3, 0.0));
        assert_eq!(pairs[2], (4, 1.0));
        assert_eq!(pairs[3], (5, -1.0));
        assert_eq!(pairs[4], (7, 1.0));
        assert_eq!(pairs[5], (8, -1.0));
        assert_eq!(pairs[6], (9, 0.0));
        assert_eq!(pairs[7], (11, -1.0));
        assert_eq!(pairs[8], (13, 1.0));
    }

    #[test]
    fn lookup_by_label() {
        assert!(LFunctionSpec::from_label("zeta").is_some());
        assert!(LFunctionSpec::from_label("chi_3").is_some());
        assert!(LFunctionSpec::from_label("nonsense").is_none());
        assert_eq!(LFunctionSpec::from_label("chi_3").unwrap().label, "chi_3");
    }

    #[test]
    fn chi_4_values() {
        let c = LFunctionSpec::chi_4();
        assert_eq!(c.chi_at(0), 0.0);
        assert_eq!(c.chi_at(1), 1.0);
        assert_eq!(c.chi_at(2), 0.0);
        assert_eq!(c.chi_at(3), -1.0);
        assert!(!c.is_even()); // odd parity
        assert!(c.is_real());
    }

    #[test]
    fn chi_5_real_values() {
        let c = LFunctionSpec::chi_5_real();
        assert_eq!(c.chi_at(0), 0.0);
        assert_eq!(c.chi_at(1), 1.0);
        assert_eq!(c.chi_at(2), -1.0);
        assert_eq!(c.chi_at(3), -1.0);
        assert_eq!(c.chi_at(4), 1.0);
        assert!(c.is_even()); // even parity
    }

    #[test]
    fn chi_7_values() {
        let c = LFunctionSpec::chi_7();
        assert_eq!(c.chi_at(0), 0.0);
        assert_eq!(c.chi_at(1), 1.0);
        assert_eq!(c.chi_at(2), 1.0);
        assert_eq!(c.chi_at(3), -1.0);
        assert_eq!(c.chi_at(4), 1.0);
        assert_eq!(c.chi_at(5), -1.0);
        assert_eq!(c.chi_at(6), -1.0);
        assert!(!c.is_even()); // odd
    }

    #[test]
    fn builtin_all_returns_five() {
        let all = LFunctionSpec::builtin_all();
        assert_eq!(all.len(), 5);
        assert!(all[0].is_trivial());
        assert!(!all[1].is_trivial());
    }
}
