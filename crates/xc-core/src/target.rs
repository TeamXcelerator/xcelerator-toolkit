use crate::ConfigError;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

/// Portable finite decimal literal used by serializable problem specifications.
/// Numerical crates parse it into their active scalar backend.  Structural
/// validation and ordering do not pass through f64, so very small and very
/// large HP values remain meaningful.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct DecimalLiteral(String);

#[derive(Clone, Debug)]
struct DecimalParts {
    sign: i8,
    digits: Vec<u8>,
    scale: i64,
}

impl DecimalParts {
    fn parse(value: &str) -> Result<Self, ConfigError> {
        if value.is_empty() || value.chars().any(char::is_whitespace) {
            return Err(ConfigError::new(
                "decimal literal must be nonempty and contain no whitespace",
            ));
        }

        let bytes = value.as_bytes();
        let mut cursor = 0usize;
        let mut sign = 1i8;
        if bytes.first() == Some(&b'+') {
            cursor += 1;
        } else if bytes.first() == Some(&b'-') {
            cursor += 1;
            sign = -1;
        }
        if cursor == bytes.len() {
            return Err(ConfigError::new("decimal literal has no digits"));
        }

        let mantissa_start = cursor;
        let mut decimal_index: Option<usize> = None;
        let mut exponent_index: Option<usize> = None;
        while cursor < bytes.len() {
            match bytes[cursor] {
                b'0'..=b'9' => cursor += 1,
                b'.' if decimal_index.is_none() && exponent_index.is_none() => {
                    decimal_index = Some(cursor);
                    cursor += 1;
                }
                b'e' | b'E' if exponent_index.is_none() => {
                    exponent_index = Some(cursor);
                    break;
                }
                _ => return Err(ConfigError::new("invalid decimal literal syntax")),
            }
        }

        let mantissa_end = exponent_index.unwrap_or(bytes.len());
        let digit_count = bytes[mantissa_start..mantissa_end]
            .iter()
            .filter(|byte| byte.is_ascii_digit())
            .count();
        if digit_count == 0 {
            return Err(ConfigError::new("decimal literal has no mantissa digits"));
        }

        let mut exponent = 0i64;
        if let Some(index) = exponent_index {
            let mut pos = index + 1;
            let mut exponent_sign = 1i64;
            if bytes.get(pos) == Some(&b'+') {
                pos += 1;
            } else if bytes.get(pos) == Some(&b'-') {
                pos += 1;
                exponent_sign = -1;
            }
            if pos == bytes.len() || !bytes[pos..].iter().all(u8::is_ascii_digit) {
                return Err(ConfigError::new("invalid decimal exponent"));
            }
            let mut magnitude = 0i64;
            for byte in &bytes[pos..] {
                magnitude = magnitude
                    .checked_mul(10)
                    .and_then(|current| current.checked_add((*byte - b'0') as i64))
                    .ok_or_else(|| ConfigError::new("decimal exponent is too large"))?;
            }
            exponent = exponent_sign
                .checked_mul(magnitude)
                .ok_or_else(|| ConfigError::new("decimal exponent is too large"))?;
        }

        let fractional_digits = decimal_index
            .map(|index| mantissa_end.saturating_sub(index + 1))
            .unwrap_or(0);
        let mut digits: Vec<u8> = bytes[mantissa_start..mantissa_end]
            .iter()
            .filter(|byte| byte.is_ascii_digit())
            .map(|byte| *byte - b'0')
            .collect();

        let first_nonzero = digits.iter().position(|digit| *digit != 0);
        let Some(first_nonzero) = first_nonzero else {
            return Ok(Self {
                sign: 0,
                digits: vec![0],
                scale: 0,
            });
        };
        digits.drain(0..first_nonzero);

        // Trailing zeros may be absorbed into the base-10 scale.  This keeps
        // comparisons compact and gives numerically equivalent literals the
        // same internal value for ordering purposes.
        let mut scale = exponent
            .checked_sub(fractional_digits as i64)
            .ok_or_else(|| ConfigError::new("decimal scale is too large"))?;
        while digits.last() == Some(&0) {
            digits.pop();
            scale = scale
                .checked_add(1)
                .ok_or_else(|| ConfigError::new("decimal scale is too large"))?;
        }

        Ok(Self {
            sign,
            digits,
            scale,
        })
    }

    fn cmp_magnitude(&self, other: &Self) -> Ordering {
        debug_assert!(self.sign != 0 && other.sign != 0);
        let self_extent = self.digits.len() as i128 + self.scale as i128;
        let other_extent = other.digits.len() as i128 + other.scale as i128;
        match self_extent.cmp(&other_extent) {
            Ordering::Equal => {
                let width = self.digits.len().max(other.digits.len());
                for index in 0..width {
                    let left = self.digits.get(index).copied().unwrap_or(0);
                    let right = other.digits.get(index).copied().unwrap_or(0);
                    match left.cmp(&right) {
                        Ordering::Equal => {}
                        order => return order,
                    }
                }
                Ordering::Equal
            }
            order => order,
        }
    }

    fn cmp_numeric(&self, other: &Self) -> Ordering {
        match self.sign.cmp(&other.sign) {
            Ordering::Equal if self.sign == 0 => Ordering::Equal,
            Ordering::Equal if self.sign > 0 => self.cmp_magnitude(other),
            Ordering::Equal => self.cmp_magnitude(other).reverse(),
            order => order,
        }
    }
}

impl DecimalLiteral {
    pub fn new(value: impl Into<String>) -> Result<Self, ConfigError> {
        let value = value.into();
        DecimalParts::parse(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        DecimalParts::parse(&self.0).map(|_| ())
    }

    /// Exact comparison for finite base-10 literals.  This is suitable for
    /// configuration and certificate structure checks at arbitrary scale.
    pub fn cmp_numeric(&self, other: &Self) -> Result<Ordering, ConfigError> {
        let left = DecimalParts::parse(&self.0)?;
        let right = DecimalParts::parse(&other.0)?;
        Ok(left.cmp_numeric(&right))
    }

    /// Convenience conversion for explicitly f64-only discovery paths.
    pub fn parse_f64(&self) -> Result<f64, ConfigError> {
        self.validate()?;
        let value = self
            .0
            .parse::<f64>()
            .map_err(|error| ConfigError::new(format!("invalid f64 decimal literal: {error}")))?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(ConfigError::new(
                "decimal literal is outside the finite f64 range",
            ))
        }
    }
}

impl Display for DecimalLiteral {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Mathematical target, independent of solver implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum EigenTarget {
    AlgebraicSmallest,
    AlgebraicLargest,
    SmallestMagnitude,
    ClosestTo {
        shift: DecimalLiteral,
    },
    IndexRange {
        first: usize,
        last: usize,
    },
    Interval {
        lower: DecimalLiteral,
        upper: DecimalLiteral,
    },
}

impl EigenTarget {
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::ClosestTo { shift } => shift.validate(),
            Self::IndexRange { first, last } if first > last => Err(ConfigError::new(
                "eigenvalue index range must satisfy first <= last",
            )),
            Self::Interval { lower, upper } => {
                lower.validate()?;
                upper.validate()?;
                if lower.cmp_numeric(upper)? != Ordering::Less {
                    return Err(ConfigError::new(
                        "eigenvalue interval must have lower < upper",
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decimal(value: &str) -> DecimalLiteral {
        DecimalLiteral::new(value).unwrap()
    }

    #[test]
    fn compares_decimals_without_f64() {
        assert_eq!(
            decimal("1e-100000")
                .cmp_numeric(&decimal("2e-100000"))
                .unwrap(),
            Ordering::Less
        );
        assert_eq!(
            decimal("1.200").cmp_numeric(&decimal("1.19")).unwrap(),
            Ordering::Greater
        );
        assert_eq!(
            decimal("-1e100000")
                .cmp_numeric(&decimal("-9e99999"))
                .unwrap(),
            Ordering::Less
        );
        assert_eq!(
            decimal("+0.000e999").cmp_numeric(&decimal("-0")).unwrap(),
            Ordering::Equal
        );
    }

    #[test]
    fn rejects_invalid_decimals() {
        for value in ["", ".", "1e", "--1", "1 2", "NaN", "inf"] {
            assert!(DecimalLiteral::new(value).is_err(), "accepted {value:?}");
        }
    }

    #[test]
    fn validates_huge_interval_exactly() {
        let target = EigenTarget::Interval {
            lower: decimal("9.5e99999"),
            upper: decimal("1.01e100000"),
        };
        assert!(target.validate().is_ok());
    }
}
