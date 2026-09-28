// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Validated real Dirichlet characters and prime-power twist data.
//!
//! A character modulo q is periodic, vanishes exactly on nonunits, and is
//! completely multiplicative with chi(1) = 1. This module supports real
//! characters with values in {-1, 0, 1}, including principal and imprimitive
//! characters. The modulus need not be the primitive conductor.
//!
//! For Re(s) > 1, the defining absolutely convergent identities are
//! `L(s, chi) = product_p (1 - chi(p) p^(-s))^(-1)` and
//! `-L'(s, chi)/L(s, chi) = sum_n chi(n) Lambda(n) n^(-s)`.
//! Thus a prime-power contribution at n = p^j, j >= 1, receives the exact
//! factor chi(p)^j; it vanishes when p divides q. Exponent zero instead gives
//! chi(1) = 1, including at those primes.
//!
//! These are character data and enumeration utilities. They do not assemble
//! a generalized CCM operator, conductor/gamma terms, pole corrections, or a
//! functional equation. Such an assembly must distinguish primitive conductors
//! from moduli and principal characters from the modulus-one zeta character.

use serde::{Deserialize, Serialize};

/// Validated real character data for a Dirichlet L-function.
///
/// Mathematical fields are immutable after construction. Use [`Self::new`]
/// for custom characters; deserialization applies the same validation.
#[derive(Debug, Clone, Serialize)]
pub struct LFunctionSpec {
    modulus: u64,
    chi: Vec<i8>,
    parity: u8,
    /// A descriptive label; it does not determine any mathematical property.
    pub label: String,
}

impl<'de> Deserialize<'de> for LFunctionSpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireSpec {
            modulus: u64,
            chi: Vec<i8>,
            parity: u8,
            label: String,
        }
        let raw = WireSpec::deserialize(deserializer)?;
        Self::new(raw.modulus, raw.chi, raw.parity, raw.label).map_err(serde::de::Error::custom)
    }
}

/// Validate a homomorphism on the unit group by adjoining one generator at a
/// time. Each new coset is checked once; the closing power relation ensures
/// the extension is well-defined. Nonunits are checked separately below.
fn validate_real_character(modulus: u64, chi: &[i8], parity: u8) -> anyhow::Result<()> {
    use anyhow::{ensure, Context};
    ensure!(modulus > 0, "character modulus must be positive");
    let q = usize::try_from(modulus).context("character modulus exceeds platform capacity")?;
    ensure!(
        chi.len() == q,
        "character table length must equal its modulus"
    );
    ensure!(parity <= 1, "character parity must be zero or one");
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    for (n, &value) in chi.iter().enumerate() {
        ensure!(
            (-1..=1).contains(&value),
            "character values must be -1, 0, or 1"
        );
        ensure!(
            (value != 0) == (gcd(n as u64, modulus) == 1),
            "character must vanish exactly on nonunits (residue {n})"
        );
    }
    let identity = 1 % q;
    ensure!(
        chi[identity] == 1,
        "character must take value one at the identity"
    );
    ensure!(
        chi[q - 1] == if parity == 0 { 1 } else { -1 },
        "character parity disagrees with its value at minus one"
    );
    let mut seen = Vec::new();
    seen.try_reserve_exact(q)
        .context("character validation membership allocation failed")?;
    seen.resize(q, false);
    let mut members = Vec::new();
    members
        .try_reserve_exact(q)
        .context("character validation subgroup allocation failed")?;
    members.push(identity);
    seen[identity] = true;
    let product = |a: usize, b: usize| ((a as u128 * b as u128) % modulus as u128) as usize;
    for generator in 0..q {
        if chi[generator] == 0 || seen[generator] {
            continue;
        }
        let old_len = members.len();
        let mut power = generator;
        let mut character_power = chi[generator];
        while !seen[power] {
            for i in 0..old_len {
                let h = members[i];
                let residue = product(h, power);
                ensure!(
                    chi[residue] == chi[h] * character_power,
                    "character is not multiplicative (residue {residue})"
                );
                seen[residue] = true;
                members.push(residue);
            }
            power = product(power, generator);
            character_power *= chi[generator];
        }
        ensure!(
            chi[power] == character_power,
            "character violates a generator power relation"
        );
    }
    Ok(())
}

impl LFunctionSpec {
    /// Construct a real Dirichlet character, validating its complete table.
    ///
    /// Requires a positive modulus, exactly that many entries, zeros precisely
    /// at nonunits, values +/-1 at units, complete multiplicativity, and parity
    /// agreeing with chi(-1). Validation takes O(q log q) Euclidean work and
    /// O(q) storage; queries then take constant time. This does not assert
    /// primitivity or identify the primitive conductor.
    pub fn new(modulus: u64, chi: Vec<i8>, parity: u8, label: String) -> anyhow::Result<Self> {
        validate_real_character(modulus, &chi, parity)?;
        Ok(Self {
            modulus,
            chi,
            parity,
            label,
        })
    }

    /// Period of the character; this need not equal its primitive conductor.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Exact immutable values at residues 0 through q - 1.
    pub fn values(&self) -> &[i8] {
        &self.chi
    }

    /// Zero for chi(-1) = +1, one for chi(-1) = -1.
    pub fn parity(&self) -> u8 {
        self.parity
    }

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

    /// Evaluate χ(n) returning the exact integer value `{-1, 0, +1}`.
    /// This is the precision-agnostic primitive — call it from HP code
    /// paths and convert the integer result to whatever working type
    /// you need (`Float::with_val(prec, spec.chi_at(n))`).
    #[inline]
    pub fn chi_at(&self, n: u64) -> i8 {
        let idx = (n % self.modulus) as usize;
        self.chi[idx]
    }

    /// Evaluate χ(n). For Stage 1, returns -1/0/1 as f64.
    #[inline]
    pub fn chi_at_f64(&self, n: u64) -> f64 {
        self.chi_at(n) as f64
    }

    /// Compute chi(p^j) exactly. The base need not be prime.
    /// Exponent zero returns chi(1) = 1 for every base, including zero.
    #[inline]
    // Keep remainder arithmetic for the Rust 1.85 MSRV.
    #[allow(unknown_lints, clippy::manual_is_multiple_of)]
    pub fn chi_at_prime_power(&self, p: u64, j: u32) -> i8 {
        if j == 0 {
            return 1;
        }
        let chi_p = self.chi_at(p);
        if chi_p == 0 {
            0
        } else if j % 2 == 0 {
            // χ(p) ∈ {-1, +1} squared is +1.
            1
        } else {
            chi_p
        }
    }

    /// Compute chi(p^j); returns zero for positive j when chi(p) = 0.
    #[inline]
    pub fn chi_at_prime_power_f64(&self, p: u64, j: u32) -> f64 {
        self.chi_at_prime_power(p, j) as f64
    }

    /// True iff χ takes only real values (Stage 1 supports only these).
    pub fn is_real(&self) -> bool {
        self.chi.iter().all(|&c| (-1..=1).contains(&c))
    }

    /// True exactly for the modulus-one character, which recovers zeta.
    /// Principal characters at larger moduli have missing Euler factors and
    /// therefore return false here.
    pub fn is_trivial(&self) -> bool {
        self.modulus == 1
    }

    /// True iff chi(-1) = +1. This does not itself assemble a gamma factor.
    pub fn is_even(&self) -> bool {
        self.parity == 0
    }
}

/// Enumerate prime powers `n = p^j` with `1 < n ≤ bound`, returning
/// `(n, p, j)` triples — the same shape as `ccm::prime_powers_up_to`.
///
/// **The `spec` parameter is intentionally unused.** This function enumerates
/// ALL prime powers up to `bound` regardless of the character — the χ weighting
/// is applied by the caller *after* enumeration. To get χ(p)^j at any
/// precision, call `spec.chi_at_prime_power(p, j)` on the returned triples.
///
/// This design keeps the enumerator precision-agnostic: the same `(n, p, j)`
/// triples work for both f64 (`chi_at_prime_power_f64`) and HP
/// (`Float::with_val(prec, spec.chi_at_prime_power(p, j))`).
///
/// # Panics
/// Has the same capacity/allocation failure contract as
/// [`crate::ccm::prime_powers_up_to`]. For a checked path, use
/// [`crate::ccm::try_prime_powers_up_to`] and apply this character afterward.
pub fn prime_powers_up_to_chi(bound: u64, _spec: &LFunctionSpec) -> Vec<(u64, u64, u32)> {
    crate::ccm::prime_powers_up_to(bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeta_is_trivial() {
        let z = LFunctionSpec::riemann_zeta();
        assert!(z.is_trivial());
        assert_eq!(z.chi_at_f64(0), 1.0);
        assert_eq!(z.chi_at_f64(2), 1.0);
        assert_eq!(z.chi_at_f64(100), 1.0);
        assert_eq!(z.chi_at_prime_power_f64(2, 5), 1.0);
    }

    #[test]
    fn chi_3_values() {
        let c = LFunctionSpec::chi_3();
        assert_eq!(c.chi_at_f64(0), 0.0); // gcd(0, 3) > 1
        assert_eq!(c.chi_at_f64(1), 1.0);
        assert_eq!(c.chi_at_f64(2), -1.0);
        assert_eq!(c.chi_at_f64(3), 0.0);
        assert_eq!(c.chi_at_f64(4), 1.0); // 4 mod 3 = 1
        assert_eq!(c.chi_at_f64(5), -1.0); // 5 mod 3 = 2
        assert!(!c.is_trivial());
        assert!(c.is_real());
    }

    #[test]
    fn chi_3_prime_powers() {
        let c = LFunctionSpec::chi_3();
        // p=2 gives χ(2)=-1, so χ(2^j) = (-1)^j
        assert_eq!(c.chi_at_prime_power_f64(2, 1), -1.0);
        assert_eq!(c.chi_at_prime_power_f64(2, 2), 1.0);
        assert_eq!(c.chi_at_prime_power_f64(2, 3), -1.0);
        // p=3 gives χ(3)=0, so all powers are 0
        assert_eq!(c.chi_at_prime_power_f64(3, 1), 0.0);
        assert_eq!(c.chi_at_prime_power_f64(3, 2), 0.0);
        // p=5 gives χ(5)=χ(2)=-1
        assert_eq!(c.chi_at_prime_power_f64(5, 1), -1.0);
        assert_eq!(c.chi_at_prime_power_f64(5, 2), 1.0);
    }

    #[test]
    fn enumerate_prime_powers_zeta_matches_existing() {
        let zeta = LFunctionSpec::riemann_zeta();
        let pp = prime_powers_up_to_chi(13, &zeta);
        let ks: Vec<u64> = pp.iter().map(|&(k, _, _)| k).collect();
        assert_eq!(ks, vec![2, 3, 4, 5, 7, 8, 9, 11, 13]);
        // For zeta, χ(p)^j is always 1; verify via spec.
        for &(_, p, j) in &pp {
            assert_eq!(zeta.chi_at_prime_power_f64(p, j), 1.0);
        }
    }

    #[test]
    fn enumerate_prime_powers_chi_3() {
        let c = LFunctionSpec::chi_3();
        let pp = prime_powers_up_to_chi(13, &c);
        // Each (p, j) yields a chi value via spec.chi_at_prime_power_f64.
        // χ(2) = -1 ⇒ χ(2)=-1, χ(4)=1, χ(8)=-1
        // χ(3) = 0 ⇒ χ(3)=0, χ(9)=0
        // χ(5) = -1, χ(7) = 1, χ(11) = -1, χ(13) = 1
        let mut seen: Vec<(u64, f64)> = pp
            .iter()
            .map(|&(k, p, j)| (k, c.chi_at_prime_power_f64(p, j)))
            .collect();
        seen.sort_by_key(|&(k, _)| k);
        assert_eq!(seen[0], (2, -1.0));
        assert_eq!(seen[1], (3, 0.0));
        assert_eq!(seen[2], (4, 1.0));
        assert_eq!(seen[3], (5, -1.0));
        assert_eq!(seen[4], (7, 1.0));
        assert_eq!(seen[5], (8, -1.0));
        assert_eq!(seen[6], (9, 0.0));
        assert_eq!(seen[7], (11, -1.0));
        assert_eq!(seen[8], (13, 1.0));
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
        assert_eq!(c.chi_at_f64(0), 0.0);
        assert_eq!(c.chi_at_f64(1), 1.0);
        assert_eq!(c.chi_at_f64(2), 0.0);
        assert_eq!(c.chi_at_f64(3), -1.0);
        assert!(!c.is_even()); // odd parity
        assert!(c.is_real());
    }

    #[test]
    fn chi_5_real_values() {
        let c = LFunctionSpec::chi_5_real();
        assert_eq!(c.chi_at_f64(0), 0.0);
        assert_eq!(c.chi_at_f64(1), 1.0);
        assert_eq!(c.chi_at_f64(2), -1.0);
        assert_eq!(c.chi_at_f64(3), -1.0);
        assert_eq!(c.chi_at_f64(4), 1.0);
        assert!(c.is_even()); // even parity
    }

    #[test]
    fn chi_7_values() {
        let c = LFunctionSpec::chi_7();
        assert_eq!(c.chi_at_f64(0), 0.0);
        assert_eq!(c.chi_at_f64(1), 1.0);
        assert_eq!(c.chi_at_f64(2), 1.0);
        assert_eq!(c.chi_at_f64(3), -1.0);
        assert_eq!(c.chi_at_f64(4), 1.0);
        assert_eq!(c.chi_at_f64(5), -1.0);
        assert_eq!(c.chi_at_f64(6), -1.0);
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
