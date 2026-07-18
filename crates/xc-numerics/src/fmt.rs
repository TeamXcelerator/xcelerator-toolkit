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

use rug::float::{Round, Special};
use rug::{Assign, Float, Integer};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Number of significant decimal digits used by `display_hp_short`.
pub const DISPLAY_HP_DEFAULT_DIGITS: usize = 6;

/// Exact, portable representation of one finite MPFR value.
///
/// `significand_hex * 2^binary_exponent` is the exact value. Retaining the
/// original precision makes reconstruction lossless instead of treating a
/// display-oriented decimal string as a persistence format.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableHpFloat {
    pub precision_bits: u32,
    pub significand_hex: String,
    pub binary_exponent: i32,
    pub negative_zero: bool,
}

impl PortableHpFloat {
    pub fn from_float(value: &Float) -> Result<Self, HpFormatError> {
        let (significand, binary_exponent) = value
            .to_integer_exp()
            .ok_or_else(|| HpFormatError("only finite MPFR values may be persisted".to_owned()))?;
        let portable = Self {
            precision_bits: value.prec(),
            significand_hex: significand.to_string_radix(16),
            binary_exponent,
            negative_zero: value.is_zero() && value.is_sign_negative(),
        };
        portable.validate()?;
        Ok(portable)
    }

    pub fn validate(&self) -> Result<(), HpFormatError> {
        if !(2..=1_000_000).contains(&self.precision_bits) {
            return Err(HpFormatError(
                "portable HP precision must be in 2..=1,000,000 bits".to_owned(),
            ));
        }
        let significand = Integer::from_str_radix(&self.significand_hex, 16)
            .map_err(|_| HpFormatError("portable HP significand is not base-16".to_owned()))?;
        if significand.to_string_radix(16) != self.significand_hex {
            return Err(HpFormatError(
                "portable HP significand must use canonical lowercase base-16".to_owned(),
            ));
        }
        if self.negative_zero && !significand.is_zero() {
            return Err(HpFormatError(
                "negative_zero is valid only for a zero significand".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn to_float(&self) -> Result<Float, HpFormatError> {
        self.validate()?;
        let significand = Integer::from_str_radix(&self.significand_hex, 16)
            .map_err(|_| HpFormatError("portable HP significand is not base-16".to_owned()))?;
        if significand.is_zero() {
            return Ok(Float::with_val(
                self.precision_bits,
                if self.negative_zero {
                    Special::NegZero
                } else {
                    Special::Zero
                },
            ));
        }
        let mut value = Float::new(self.precision_bits);
        value.assign(significand);
        value <<= self.binary_exponent;
        Ok(value)
    }
}

/// Exact persisted enclosure with independently retained endpoint precision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableHpInterval {
    pub lower: PortableHpFloat,
    pub upper: PortableHpFloat,
}

impl PortableHpInterval {
    pub fn from_bounds(lower: &Float, upper: &Float) -> Result<Self, HpFormatError> {
        let interval = Self {
            lower: PortableHpFloat::from_float(lower)?,
            upper: PortableHpFloat::from_float(upper)?,
        };
        interval.validate()?;
        Ok(interval)
    }

    pub fn validate(&self) -> Result<(), HpFormatError> {
        let (lower, upper) = self.to_bounds_unchecked()?;
        if lower > upper {
            return Err(HpFormatError(
                "portable HP interval lower endpoint exceeds upper endpoint".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn to_bounds(&self) -> Result<(Float, Float), HpFormatError> {
        self.validate()?;
        self.to_bounds_unchecked()
    }

    fn to_bounds_unchecked(&self) -> Result<(Float, Float), HpFormatError> {
        Ok((self.lower.to_float()?, self.upper.to_float()?))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HpDecimalRounding {
    Nearest,
    Down,
    Up,
    TowardZero,
    AwayFromZero,
}

impl HpDecimalRounding {
    fn mpfr(self) -> Round {
        match self {
            Self::Nearest => Round::Nearest,
            Self::Down => Round::Down,
            Self::Up => Round::Up,
            Self::TowardZero => Round::Zero,
            Self::AwayFromZero => Round::AwayZero,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HpDisplayNotation {
    Scientific,
    Plain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HpDisplayPolicy {
    pub significant_digits: usize,
    pub rounding: HpDecimalRounding,
    pub notation: HpDisplayNotation,
    pub trim_trailing_fractional_zeros: bool,
    pub preserve_negative_zero: bool,
    pub maximum_output_characters: usize,
}

impl HpDisplayPolicy {
    pub fn scientific(significant_digits: usize) -> Self {
        Self {
            significant_digits,
            rounding: HpDecimalRounding::Nearest,
            notation: HpDisplayNotation::Scientific,
            trim_trailing_fractional_zeros: false,
            preserve_negative_zero: true,
            maximum_output_characters: significant_digits.saturating_add(64),
        }
    }

    pub fn validate(&self) -> Result<(), HpFormatError> {
        if self.significant_digits == 0 || self.significant_digits > 1_000_000 {
            return Err(HpFormatError(
                "significant_digits must be in 1..=1,000,000".to_owned(),
            ));
        }
        if self.maximum_output_characters == 0 || self.maximum_output_characters > 10_000_000 {
            return Err(HpFormatError(
                "maximum_output_characters must be in 1..=10,000,000".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HpFormatError(String);

impl Display for HpFormatError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for HpFormatError {}

pub fn format_hp_with_policy(
    value: &Float,
    policy: &HpDisplayPolicy,
) -> Result<String, HpFormatError> {
    policy.validate()?;
    let (negative, mut digits, exponent) =
        value.to_sign_string_exp_round(10, Some(policy.significant_digits), policy.rounding.mpfr());
    let sign = if negative && (!value.is_zero() || policy.preserve_negative_zero) {
        "-"
    } else {
        ""
    };
    let mut output = if let Some(exponent) = exponent {
        match policy.notation {
            HpDisplayNotation::Scientific => {
                let first = digits.remove(0);
                let mut mantissa = if digits.is_empty() {
                    first.to_string()
                } else {
                    format!("{first}.{digits}")
                };
                if policy.trim_trailing_fractional_zeros {
                    trim_fractional_zeros(&mut mantissa);
                }
                format!("{sign}{mantissa}e{}", exponent - 1)
            }
            HpDisplayNotation::Plain => {
                let decimal_position = isize::try_from(exponent).map_err(|_| {
                    HpFormatError("decimal exponent does not fit this platform".to_owned())
                })?;
                let digit_count = isize::try_from(digits.len()).map_err(|_| {
                    HpFormatError("decimal digit count does not fit this platform".to_owned())
                })?;
                let mut magnitude = if decimal_position <= 0 {
                    let zeros = usize::try_from(-decimal_position).map_err(|_| {
                        HpFormatError("plain-format leading-zero count overflowed".to_owned())
                    })?;
                    ensure_output_size(
                        sign.len()
                            .saturating_add(2)
                            .saturating_add(zeros)
                            .saturating_add(digits.len()),
                        policy,
                    )?;
                    format!("0.{}{digits}", "0".repeat(zeros))
                } else if decimal_position >= digit_count {
                    let zeros = usize::try_from(decimal_position - digit_count).map_err(|_| {
                        HpFormatError("plain-format trailing-zero count overflowed".to_owned())
                    })?;
                    ensure_output_size(
                        sign.len()
                            .saturating_add(digits.len())
                            .saturating_add(zeros),
                        policy,
                    )?;
                    format!("{digits}{}", "0".repeat(zeros))
                } else {
                    let split = usize::try_from(decimal_position).map_err(|_| {
                        HpFormatError("plain-format decimal position overflowed".to_owned())
                    })?;
                    digits.insert(split, '.');
                    digits
                };
                if policy.trim_trailing_fractional_zeros {
                    trim_fractional_zeros(&mut magnitude);
                }
                format!("{sign}{magnitude}")
            }
        }
    } else {
        format!("{sign}{digits}")
    };
    if output.len() > policy.maximum_output_characters {
        output.clear();
        return Err(HpFormatError(format!(
            "formatted output exceeds maximum_output_characters={}",
            policy.maximum_output_characters
        )));
    }
    Ok(output)
}

fn ensure_output_size(projected: usize, policy: &HpDisplayPolicy) -> Result<(), HpFormatError> {
    if projected > policy.maximum_output_characters {
        return Err(HpFormatError(format!(
            "formatted output exceeds maximum_output_characters={}",
            policy.maximum_output_characters
        )));
    }
    Ok(())
}

fn trim_fractional_zeros(value: &mut String) {
    if !value.contains('.') {
        return;
    }
    while value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
}

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
    format_hp_with_policy(x, &HpDisplayPolicy::scientific(sig_digits.max(1)))
        .expect("the bounded scientific display policy is valid")
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
        let max_decimal_digits = (prec as u64) * 1000 / 3322; // ~prec * 0.30103
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
    fn portable_hp_values_and_intervals_round_trip_exactly() {
        let mut value = Float::with_val(256, 1);
        value += Float::with_val(256, 1) >> 200;
        let tiny = fl(4096, "-1.234567890123456789e-1000");
        let negative_zero = Float::with_val(192, Special::NegZero);

        for original in [&value, &tiny, &negative_zero] {
            let portable = PortableHpFloat::from_float(original).unwrap();
            let encoded = serde_json::to_vec(&portable).unwrap();
            let decoded: PortableHpFloat = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded, portable);
            let reconstructed = decoded.to_float().unwrap();
            assert_eq!(reconstructed.prec(), original.prec());
            assert_eq!(reconstructed, *original);
            assert_eq!(
                reconstructed.is_sign_negative(),
                original.is_sign_negative()
            );
        }

        let lower = fl(768, "3.1415926535897932384626433832795028841971");
        let upper = fl(1024, "3.1415926535897932384626433832795028841972");
        let interval = PortableHpInterval::from_bounds(&lower, &upper).unwrap();
        let encoded = serde_json::to_vec(&interval).unwrap();
        let decoded: PortableHpInterval = serde_json::from_slice(&encoded).unwrap();
        let (decoded_lower, decoded_upper) = decoded.to_bounds().unwrap();
        assert_eq!(decoded_lower, lower);
        assert_eq!(decoded_upper, upper);
        assert_eq!(decoded, interval);

        let mut result = xc_core::ResearchResult::computed(
            interval,
            xc_core::SolverProvenance::current_package("rug_mpfr"),
        );
        result.diagnostics.insert_scalar("residual", "1.0e-900");
        let encoded_result = serde_json::to_vec(&result).unwrap();
        let decoded_result: xc_core::ResearchResult<PortableHpInterval> =
            serde_json::from_slice(&encoded_result).unwrap();
        assert_eq!(decoded_result, result);
        let (saved_lower, saved_upper) = decoded_result.value.unwrap().to_bounds().unwrap();
        assert_eq!(saved_lower, lower);
        assert_eq!(saved_upper, upper);
    }

    #[test]
    fn portable_hp_values_reject_nonfinite_and_noncanonical_data() {
        assert!(PortableHpFloat::from_float(&Float::with_val(128, Special::Nan)).is_err());
        let mut invalid = PortableHpFloat::from_float(&fl(128, "1.5")).unwrap();
        invalid.significand_hex.make_ascii_uppercase();
        assert!(invalid.validate().is_err());

        let reversed = PortableHpInterval {
            lower: PortableHpFloat::from_float(&fl(128, "2")).unwrap(),
            upper: PortableHpFloat::from_float(&fl(128, "1")).unwrap(),
        };
        assert!(reversed.validate().is_err());
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
        assert!(
            s.contains("e-500") || s.contains("e-499"),
            "exponent preserved: {}",
            s
        );
    }

    #[test]
    fn display_hp_extreme_negative_exponent() {
        let x = fl(4096, "-2.5e-1000");
        let s = display_hp(&x, 4);
        assert!(s.starts_with('-'), "got: {}", s);
        // Allow ±1 exponent for rounding artifacts at low sig_digits.
        assert!(
            s.contains("e-1000") || s.contains("e-999") || s.contains("e-1001"),
            "expected exponent ~-1000, got: {}",
            s
        );
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
        assert_eq!(
            digit_count4, 4,
            "display_hp(x, 4) should produce 4 sig digits, got '{}'",
            s4
        );

        // 10 sig digits → mantissa has 10 digits.
        let s10 = display_hp(&x, 10);
        let mantissa10 = s10.split('e').next().unwrap().trim_start_matches('-');
        let digit_count10 = mantissa10.chars().filter(|c| c.is_ascii_digit()).count();
        assert_eq!(
            digit_count10, 10,
            "display_hp(x, 10) should produce 10 sig digits, got '{}'",
            s10
        );

        // 1 sig digit → mantissa "3" or "3e0" — 1 digit.
        // (rug may render this as "3e0" or "3.e0"; allow either form.)
        let s1 = display_hp(&x, 1);
        let mantissa1 = s1.split('e').next().unwrap().trim_start_matches('-');
        let digit_count1 = mantissa1.chars().filter(|c| c.is_ascii_digit()).count();
        assert!(
            digit_count1 == 1 || digit_count1 == 2,
            "display_hp(x, 1) should produce 1 sig digit (or 2 if rug enforces minimum), got '{}'",
            s1
        );
    }

    #[test]
    fn typed_display_policy_controls_decimal_rounding_without_changing_precision() {
        let value = fl(256, "23.3");
        let original_precision = value.prec();
        let mut policy = HpDisplayPolicy::scientific(2);
        policy.rounding = HpDecimalRounding::Down;
        assert_eq!(format_hp_with_policy(&value, &policy).unwrap(), "2.3e1");
        policy.rounding = HpDecimalRounding::Up;
        assert_eq!(format_hp_with_policy(&value, &policy).unwrap(), "2.4e1");
        assert_eq!(value.prec(), original_precision);

        let encoded = serde_json::to_vec(&policy).unwrap();
        let decoded: HpDisplayPolicy = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, policy);
    }

    #[test]
    fn plain_display_policy_trims_fractional_zeros_and_enforces_output_bound() {
        let mut policy = HpDisplayPolicy {
            significant_digits: 6,
            rounding: HpDecimalRounding::Nearest,
            notation: HpDisplayNotation::Plain,
            trim_trailing_fractional_zeros: true,
            preserve_negative_zero: true,
            maximum_output_characters: 64,
        };
        assert_eq!(
            format_hp_with_policy(&fl(256, "12.5"), &policy).unwrap(),
            "12.5"
        );
        policy.significant_digits = 5;
        assert_eq!(
            format_hp_with_policy(&fl(256, "3.1415926535"), &policy).unwrap(),
            "3.1416"
        );
        let tiny = fl(4096, "1e-1000");
        assert!(format_hp_with_policy(&tiny, &policy).is_err());
        policy.notation = HpDisplayNotation::Scientific;
        assert!(format_hp_with_policy(&tiny, &policy)
            .unwrap()
            .contains("e-1000"));
    }

    #[test]
    fn invalid_display_policy_fails_before_formatting() {
        let value = fl(128, "1.25");
        let mut policy = HpDisplayPolicy::scientific(0);
        assert!(format_hp_with_policy(&value, &policy).is_err());
        policy.significant_digits = 4;
        policy.maximum_output_characters = 0;
        assert!(format_hp_with_policy(&value, &policy).is_err());
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
        assert_eq!(
            sign_of(&x),
            Sign::Negative,
            "subnormal-by-f64-standard but normal at HP must read as negative"
        );
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
        let m_f64 = (prec as f64) * std::f64::consts::LOG10_2;
        let m_int = m_f64 as u64;
        assert!(
            s.starts_with(&format!("{}", m_int))
                || s.starts_with(&format!("{}", m_int - 1))
                || s.starts_with(&format!("{}", m_int + 1))
                || s.starts_with(&"7.7".to_string())
                || s.starts_with(&"7.6".to_string())
                || s.starts_with(&"7.8".to_string()),
            "expected matching digits near {} (= prec/3.322), got '{}'",
            m_int,
            s
        );
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
        assert!(
            m > lo && m < hi,
            "expected matching digits ~500, got {}",
            display_hp(&m, 6)
        );
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
        assert!(
            m > lo && m < hi,
            "expected matching digits ~1500, got {}",
            display_hp(&m, 6)
        );
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
        assert!(
            abs_diff < tol,
            "expected ~0.05, got {} (delta {})",
            display_hp(&rel, 6),
            display_hp(&abs_diff, 4)
        );
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
        assert!(
            abs_diff < tol,
            "expected ~0.05 below f64 floor, got {} (delta {})",
            display_hp(&rel, 6),
            display_hp(&abs_diff, 4)
        );
    }
}
