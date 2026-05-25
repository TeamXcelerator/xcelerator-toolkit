// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! HP-only formatting and comparison helpers.
//!
//! These helpers exist so that callers never need to call `to_f64()` on a
//! `rug::Float` for display, sign inspection, or relative-difference
//! computation. f64 has a maximum exponent of ~10^308 and underflows below
//! ~10^-308; both bounds are routinely exceeded in HP arithmetic, where
//! values can be 10^-1000 or smaller. Going through f64 in any of these
//! paths silently destroys magnitude and sign information for HP values.
//!
//! Use these helpers wherever you would otherwise have written
//! `format!("{:e}", x.to_f64())` or `(a.to_f64() - b.to_f64()).abs()`.
//!
//! All functions stay in `rug::Float`/MPFR throughout, except the final
//! string conversion which uses MPFR's own decimal formatting via the
//! `Display` / `LowerExp` impls of `rug::Float`.

#![cfg(feature = "hp")]

use rug::Float;

/// Number of significant decimal digits used by `display_hp_short`.
pub const DISPLAY_HP_DEFAULT_DIGITS: usize = 6;

/// Format an HP value at a given number of significant decimal digits in
/// scientific notation. Returns e.g. `"1.23457e-1234"` or `"-3.00000e-59"`.
/// Zero is rendered as `"0"`. No f64 conversion happens at any step, so
/// this works for arbitrarily small or large magnitudes.
///
/// The implementation uses `rug::Float`'s `LowerExp` formatter, which uses
/// MPFR's own decimal conversion routines — there is no f64 round-trip
/// regardless of magnitude.
///
/// Note: `rug::Float`'s `LowerExp` treats the format-spec precision as
/// **total significant digits**, not "digits after the decimal point" as
/// `f64::LowerExp` does. So `format!("{:.4e}", x)` on a `rug::Float`
/// produces 4 total sig digits ("9.994e2"), whereas the same on an `f64`
/// produces 5 ("9.9945e2"). This function follows the rug convention so
/// the `sig_digits` arg always means total sig digits regardless of the
/// underlying type.
pub fn display_hp(x: &Float, sig_digits: usize) -> String {
    if x.is_zero() {
        return "0".to_string();
    }
    let total = sig_digits.max(1);
    format!("{:.*e}", total, x)
}

/// Format with the default number of significant digits.
pub fn display_hp_short(x: &Float) -> String {
    display_hp(x, DISPLAY_HP_DEFAULT_DIGITS)
}

/// Sign of an HP value. `Sign::Zero` means exactly zero; values that would
/// underflow in f64 (e.g. 10^-1000) are treated as their actual sign.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Sign {
    /// Strictly less than zero.
    Negative,
    /// Exactly zero (MPFR's `is_zero()`).
    Zero,
    /// Strictly greater than zero.
    Positive,
}

impl Sign {
    /// Lowercase English label: `"negative"`, `"zero"`, or `"positive"`.
    /// Used for diagnostic logging.
    pub fn as_str(self) -> &'static str {
        match self {
            Sign::Negative => "negative",
            Sign::Zero => "zero",
            Sign::Positive => "positive",
        }
    }
}

/// Read the sign of an HP value without going through f64. Inspects MPFR
/// metadata directly via `is_zero` and `is_sign_negative`.
pub fn sign_of(x: &Float) -> Sign {
    if x.is_zero() {
        Sign::Zero
    } else if x.is_sign_negative() {
        Sign::Negative
    } else {
        Sign::Positive
    }
}

/// Number of matching decimal digits between an HP computed value and an
/// HP reference value, computed entirely in HP arithmetic.
///
/// Returns `-log10(|computed - reference| / |reference|)` as a `Float` at
/// the same precision as `computed`. If both values are equal, returns
/// the maximum representable matching digits (precision_bits / log2(10)),
/// signaling "match to working precision". If `reference` is zero (and
/// `computed` isn't), returns `-log10(|computed|)`.
pub fn matching_digits(computed: &Float, reference: &Float) -> Float {
    let prec = computed.prec();
    let mut diff = computed.clone();
    diff -= reference;
    let abs_diff = diff.abs();

    if abs_diff.is_zero() {
        // Exact match — report a "matching to full working precision" value.
        // Matching digits ≈ precision_bits / log2(10) ≈ precision_bits / 3.322
        // We compute this in HP-friendly integer arithmetic.
        let max_decimal_digits = (prec as u64) * 1000 / 3322;  // ~prec * 0.30103
        return Float::with_val(prec, max_decimal_digits);
    }

    if reference.is_zero() {
        // Reference is zero; report -log10(|diff|).
        let log10 = abs_diff.log10();
        let mut neg = Float::with_val(prec, 0);
        neg -= &log10;
        return neg;
    }

    let abs_ref = reference.clone().abs();
    let mut rel = abs_diff;
    rel /= &abs_ref;
    let log10 = rel.log10();
    let mut neg = Float::with_val(prec, 0);
    neg -= &log10;
    neg
}

/// Relative difference `|a - b| / |b|`, computed entirely in HP arithmetic.
/// Returns `None` if `b` is exactly zero (relative difference undefined).
pub fn relative_difference(a: &Float, b: &Float) -> Option<Float> {
    if b.is_zero() {
        return None;
    }
    let mut diff = a.clone();
    diff -= b;
    let abs_diff = diff.abs();
    let abs_b = b.clone().abs();
    let mut rel = abs_diff;
    rel /= &abs_b;
    Some(rel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Float;

    fn fl(prec: u32, s: &str) -> Float {
        Float::with_val(prec, Float::parse(s).unwrap())
    }

    #[test]
    fn display_hp_zero() {
        let x = Float::with_val(256, 0);
        assert_eq!(display_hp(&x, 6), "0");
    }

    #[test]
    fn display_hp_small_negative_within_f64_range() {
        let x = fl(256, "-1.5e-50");
        let s = display_hp(&x, 4);
        assert!(s.starts_with('-'), "negative sign preserved: {}", s);
        assert!(s.contains("e-50"), "exponent preserved: {}", s);
    }

    #[test]
    fn display_hp_below_f64_underflow() {
        // 10^-500: well below f64's ~10^-308 underflow.
        let x = fl(2048, "-1.5e-500");
        let s = display_hp(&x, 4);
        assert!(s.starts_with('-'), "got: {}", s);
        assert!(s.contains("e-500") || s.contains("e-499"), "exponent preserved: {}", s);
    }

    #[test]
    fn display_hp_extreme_negative_exponent() {
        let x = fl(4096, "-2.5e-1000");
        let s = display_hp(&x, 4);
        assert!(s.starts_with('-'), "got: {}", s);
        // Allow ±1 exponent for rounding artifacts at low sig_digits.
        assert!(s.contains("e-1000") || s.contains("e-999") || s.contains("e-1001"),
            "expected exponent ~-1000, got: {}", s);
    }

    #[test]
    fn display_hp_significant_digits() {
        let x = fl(256, "3.141592653589793");
        let s = display_hp(&x, 4);
        // 4 significant digits via {:.4e}: "3.142e0".
        // Just check the leading "3.14" — exact form may vary.
        assert!(s.starts_with("3.14"), "got: {}", s);
    }

    /// Strict sig-digit count check. `display_hp(x, n)` must produce
    /// exactly `n` digits in the mantissa (1 before the decimal +
    /// `n - 1` after). This pins the previously-buggy off-by-one.
    #[test]
    fn display_hp_sig_digit_count_is_exact() {
        let x = fl(256, "3.141592653589793238462643383279502884197");
        // 4 sig digits → "3.142e0" → mantissa "3.142" → 4 digits total
        let s4 = display_hp(&x, 4);
        let mantissa4 = s4.split('e').next().unwrap().trim_start_matches('-');
        let digit_count4 = mantissa4.chars().filter(|c| c.is_ascii_digit()).count();
        assert_eq!(digit_count4, 4, "display_hp(x, 4) should produce 4 sig digits, got '{}'", s4);

        // 10 sig digits → mantissa has 10 digits.
        let s10 = display_hp(&x, 10);
        let mantissa10 = s10.split('e').next().unwrap().trim_start_matches('-');
        let digit_count10 = mantissa10.chars().filter(|c| c.is_ascii_digit()).count();
        assert_eq!(digit_count10, 10, "display_hp(x, 10) should produce 10 sig digits, got '{}'", s10);

        // 1 sig digit → mantissa "3" or "3e0" — 1 digit.
        // (rug may render this as "3e0" or "3.e0"; allow either form.)
        let s1 = display_hp(&x, 1);
        let mantissa1 = s1.split('e').next().unwrap().trim_start_matches('-');
        let digit_count1 = mantissa1.chars().filter(|c| c.is_ascii_digit()).count();
        assert!(digit_count1 == 1 || digit_count1 == 2,
            "display_hp(x, 1) should produce 1 sig digit (or 2 if rug enforces minimum), got '{}'", s1);
    }

    #[test]
    fn sign_of_positive() {
        let x = fl(64, "3.5");
        assert_eq!(sign_of(&x), Sign::Positive);
    }

    #[test]
    fn sign_of_negative() {
        let x = fl(64, "-3.5");
        assert_eq!(sign_of(&x), Sign::Negative);
    }

    #[test]
    fn sign_of_zero() {
        let x = Float::with_val(64, 0);
        assert_eq!(sign_of(&x), Sign::Zero);
    }

    #[test]
    fn sign_of_subnormal_negative_preserves_sign() {
        // A value that would underflow to -0.0 in f64 must keep its sign here.
        let x = fl(2048, "-1.0e-1000");
        assert_eq!(sign_of(&x), Sign::Negative,
            "subnormal-by-f64-standard but normal at HP must read as negative");
    }

    #[test]
    fn matching_digits_exact_match_returns_max_digits() {
        let prec = 256;
        let a = fl(prec, "1.234567890");
        let b = a.clone();
        let m = matching_digits(&a, &b);
        // For 256-bit precision, max decimal digits ≈ 256 * 0.30103 ≈ 77.
        // Returns a finite value near that.
        let s = display_hp(&m, 4);
        // Expect ~77 (e.g. "7.7e1" or "77.06" → "7.7e1").
        // Allow a generous range — the exact value depends on integer
        // arithmetic in matching_digits.
        let m_f64 = (prec as f64) * 0.30103;
        let m_int = m_f64 as u64;
        assert!(s.starts_with(&format!("{}", m_int)) ||
                s.starts_with(&format!("{}", m_int - 1)) ||
                s.starts_with(&format!("{}", m_int + 1)) ||
                s.starts_with(&format!("7.7")) ||
                s.starts_with(&format!("7.6")) ||
                s.starts_with(&format!("7.8")),
            "expected matching digits near {} (= prec/3.322), got '{}'", m_int, s);
    }

    #[test]
    fn matching_digits_below_f64_floor() {
        // Reference is O(1); diff is 10^-500. Matching digits should be ~500.
        let comp = fl(2048, "1.0");
        let mut perturbed = comp.clone();
        let eps = fl(2048, "1.0e-500");
        perturbed += &eps;
        let m = matching_digits(&perturbed, &comp);
        // Verify numerically (no f64) — must be in [490, 510].
        let lo = fl(2048, "490");
        let hi = fl(2048, "510");
        assert!(m > lo && m < hi,
            "expected matching digits ~500, got {}", display_hp(&m, 6));
    }

    #[test]
    fn matching_digits_extreme() {
        // Need precision well above 1500 decimal digits to make 10^-1500 a
        // measurable perturbation. 8192 bits ≈ 2466 decimal digits.
        let prec = 8192;
        let comp = fl(prec, "1.0");
        let mut perturbed = comp.clone();
        let eps = fl(prec, "1.0e-1500");
        perturbed += &eps;
        let m = matching_digits(&perturbed, &comp);
        // Expect ~1500 (allow [1490, 1510] for log10 + sqrt rounding).
        let lo = fl(prec, "1490");
        let hi = fl(prec, "1510");
        assert!(m > lo && m < hi,
            "expected matching digits ~1500, got {}", display_hp(&m, 6));
    }

    #[test]
    fn relative_difference_basic() {
        let a = fl(256, "1.05");
        let b = fl(256, "1.00");
        let rel = relative_difference(&a, &b).expect("nonzero b");
        // |1.05 - 1.00| / |1.00| = 0.05; verify numerically in HP.
        let target = fl(256, "0.05");
        let mut diff = rel.clone();
        diff -= &target;
        let abs_diff = diff.abs();
        let tol = fl(256, "1e-50");
        assert!(abs_diff < tol,
            "expected ~0.05, got {} (delta {})",
            display_hp(&rel, 6), display_hp(&abs_diff, 4));
    }

    #[test]
    fn relative_difference_b_zero_returns_none() {
        let a = fl(256, "1.0");
        let b = Float::with_val(256, 0);
        assert!(relative_difference(&a, &b).is_none());
    }

    #[test]
    fn relative_difference_works_below_f64() {
        // Both magnitudes below f64 underflow.
        let a = fl(2048, "1.05e-1000");
        let b = fl(2048, "1.00e-1000");
        let rel = relative_difference(&a, &b).expect("nonzero b");
        // Verify in HP — should still be ~0.05.
        let target = fl(2048, "0.05");
        let mut diff = rel.clone();
        diff -= &target;
        let abs_diff = diff.abs();
        let tol = fl(2048, "1e-50");
        assert!(abs_diff < tol,
            "expected ~0.05 below f64 floor, got {} (delta {})",
            display_hp(&rel, 6), display_hp(&abs_diff, 4));
    }
}
