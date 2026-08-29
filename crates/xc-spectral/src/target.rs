// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Runtime-supplied target profiles.
//!
//! The toolkit deliberately contains no research-target coefficients. A claim
//! runner supplies a canonical JSON specification through
//! `XC_TARGET_SPEC_FILE`. The public implementation evaluates a generic
//! Gaussian-polynomial lattice series deterministically and binds the SHA-256
//! digest of the complete specification into every target-derived artifact.
//! The specification text and coefficients are never copied into an artifact.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Environment variable naming the private target-profile specification.
pub const TARGET_SPEC_FILE_ENV: &str = "XC_TARGET_SPEC_FILE";

/// Exact scale applied to a polynomial. The algebraic form avoids freezing an
/// irrational coefficient at the decimal precision of a private JSON file.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScalarScaleSpec {
    Decimal {
        value: String,
    },
    RationalTimesSquareRoot {
        rational_numerator: i64,
        rational_denominator: u64,
        radicand_numerator: u64,
        radicand_denominator: u64,
    },
}

impl Default for ScalarScaleSpec {
    fn default() -> Self {
        Self::Decimal {
            value: "1".to_owned(),
        }
    }
}

impl ScalarScaleSpec {
    fn validate(&self, field: &str) -> Result<()> {
        match self {
            Self::Decimal { value } => {
                let parsed = value
                    .parse::<f64>()
                    .with_context(|| format!("{field} contains an invalid decimal scale"))?;
                if !parsed.is_finite() {
                    anyhow::bail!("{field} scale must be finite");
                }
            }
            Self::RationalTimesSquareRoot {
                rational_denominator,
                radicand_numerator,
                radicand_denominator,
                ..
            } => {
                if *rational_denominator == 0
                    || *radicand_numerator == 0
                    || *radicand_denominator == 0
                {
                    anyhow::bail!("{field} contains a singular algebraic scale");
                }
            }
        }
        Ok(())
    }

    fn value_f64(&self) -> Result<f64> {
        self.validate("polynomial")?;
        Ok(match self {
            Self::Decimal { value } => value.parse::<f64>()?,
            Self::RationalTimesSquareRoot {
                rational_numerator,
                rational_denominator,
                radicand_numerator,
                radicand_denominator,
            } => {
                (*rational_numerator as f64 / *rational_denominator as f64)
                    * (*radicand_numerator as f64 / *radicand_denominator as f64).sqrt()
            }
        })
    }
}

/// One generic series evaluated as
/// `sqrt(u) * sum_n exp(-pi*(n*u)^2) * (n*u)^p * P(pi*(n*u)^2)`.
///
/// Coefficients are ordered from constant term upward. When
/// `parameter_polynomial_coefficients` is nonempty, the evaluator adds
/// `parameter * Q(pi*(n*u)^2)` inside the same summand. The parameter is fixed
/// by making the complete auxiliary series vanish at `u = 1`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianPolynomialSeriesSpec {
    pub term_input_power: u32,
    pub polynomial_coefficients: Vec<String>,
    #[serde(default)]
    pub polynomial_scale: ScalarScaleSpec,
    #[serde(default)]
    pub parameter_polynomial_coefficients: Vec<String>,
    #[serde(default)]
    pub parameter_polynomial_scale: ScalarScaleSpec,
    pub minimum_terms: u32,
    pub maximum_terms: u32,
}

impl GaussianPolynomialSeriesSpec {
    fn validate(&self, field: &str) -> Result<()> {
        if self.term_input_power > 32 {
            anyhow::bail!("{field} term_input_power exceeds the supported limit");
        }
        if self.polynomial_coefficients.is_empty()
            || self.polynomial_coefficients.len() > 64
            || self.parameter_polynomial_coefficients.len() > 64
        {
            anyhow::bail!("{field} requires between 1 and 64 polynomial coefficients");
        }
        if self.minimum_terms == 0
            || self.maximum_terms < self.minimum_terms
            || self.maximum_terms > 1_000_000
        {
            anyhow::bail!("{field} contains an invalid summation range");
        }
        for coefficient in self
            .polynomial_coefficients
            .iter()
            .chain(&self.parameter_polynomial_coefficients)
        {
            let parsed = coefficient
                .parse::<f64>()
                .with_context(|| format!("{field} contains an invalid decimal coefficient"))?;
            if !parsed.is_finite() {
                anyhow::bail!("{field} coefficients must be finite");
            }
        }
        self.polynomial_scale
            .validate(&format!("{field}.polynomial_scale"))?;
        self.parameter_polynomial_scale
            .validate(&format!("{field}.parameter_polynomial_scale"))?;
        Ok(())
    }
}

/// Complete private target descriptor consumed by the generic evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetProfileSpec {
    pub schema_version: u32,
    /// Opaque, non-descriptive identifier chosen by the private research run.
    pub profile_id: String,
    pub base_series: GaussianPolynomialSeriesSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auxiliary_series: Option<GaussianPolynomialSeriesSpec>,
}

impl TargetProfileSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            anyhow::bail!("unsupported target-profile schema {}", self.schema_version);
        }
        if self.profile_id.trim().is_empty()
            || self.profile_id.len() > 128
            || !self
                .profile_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            anyhow::bail!("target profile requires an opaque identifier using [A-Za-z0-9._-]");
        }
        self.base_series.validate("base_series")?;
        if !self
            .base_series
            .parameter_polynomial_coefficients
            .is_empty()
        {
            anyhow::bail!("base_series cannot contain a solved parameter");
        }
        if let Some(auxiliary) = &self.auxiliary_series {
            auxiliary.validate("auxiliary_series")?;
            if auxiliary.parameter_polynomial_coefficients.is_empty() {
                anyhow::bail!("auxiliary_series requires parameter coefficients");
            }
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let spec: Self = serde_json::from_slice(bytes).context("invalid target-profile JSON")?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed reading target specification {}", path.display()))?;
        Self::from_json(&bytes)
    }

    pub fn from_environment() -> Result<Self> {
        let Some(path) = std::env::var_os(TARGET_SPEC_FILE_ENV) else {
            #[cfg(test)]
            {
                return Ok(testing_profile_spec());
            }
            #[cfg(not(test))]
            {
                anyhow::bail!(
                    "target-dependent work requires {TARGET_SPEC_FILE_ENV} to name a private specification"
                );
            }
        };
        if path.is_empty() {
            anyhow::bail!("{TARGET_SPEC_FILE_ENV} cannot be empty");
        }
        Self::from_file(Path::new(&path))
    }

    /// Stable identity used in semantic keys. The specification itself is not
    /// persisted outside the private runtime input.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(xc_cache::ContentDigest::sha256(&serde_json::to_vec(self)?).0)
    }
}

#[cfg(test)]
fn testing_profile_spec() -> TargetProfileSpec {
    TargetProfileSpec {
        schema_version: 1,
        profile_id: "toolkit-benign-test-profile-v1".to_owned(),
        base_series: GaussianPolynomialSeriesSpec {
            term_input_power: 0,
            polynomial_coefficients: vec!["1".to_owned()],
            polynomial_scale: ScalarScaleSpec::default(),
            parameter_polynomial_coefficients: Vec::new(),
            parameter_polynomial_scale: ScalarScaleSpec::default(),
            minimum_terms: 2,
            maximum_terms: 1000,
        },
        auxiliary_series: Some(GaussianPolynomialSeriesSpec {
            term_input_power: 0,
            polynomial_coefficients: vec!["0".to_owned(), "1".to_owned()],
            polynomial_scale: ScalarScaleSpec::default(),
            parameter_polynomial_coefficients: vec!["1".to_owned()],
            parameter_polynomial_scale: ScalarScaleSpec::default(),
            minimum_terms: 2,
            maximum_terms: 1000,
        }),
    }
}

fn polynomial_f64(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value.mul_add(x, *coefficient))
}

fn input_power_f64(input: f64, power: u32) -> f64 {
    (0..power).fold(1.0, |value, _| value * input)
}

#[derive(Clone, Debug)]
struct CompiledSeriesF64 {
    term_input_power: u32,
    polynomial_coefficients: Vec<f64>,
    polynomial_scale: f64,
    parameter_polynomial_coefficients: Vec<f64>,
    parameter_polynomial_scale: f64,
    minimum_terms: u32,
    maximum_terms: u32,
}

impl CompiledSeriesF64 {
    fn new(spec: &GaussianPolynomialSeriesSpec) -> Result<Self> {
        let parse = |value: &str| -> Result<f64> {
            let parsed = value.parse::<f64>().context("invalid target coefficient")?;
            if !parsed.is_finite() {
                anyhow::bail!("target coefficients must be finite");
            }
            Ok(parsed)
        };
        Ok(Self {
            term_input_power: spec.term_input_power,
            polynomial_coefficients: spec
                .polynomial_coefficients
                .iter()
                .map(|value| parse(value))
                .collect::<Result<_>>()?,
            polynomial_scale: spec.polynomial_scale.value_f64()?,
            parameter_polynomial_coefficients: spec
                .parameter_polynomial_coefficients
                .iter()
                .map(|value| parse(value))
                .collect::<Result<_>>()?,
            parameter_polynomial_scale: spec.parameter_polynomial_scale.value_f64()?,
            minimum_terms: spec.minimum_terms,
            maximum_terms: spec.maximum_terms,
        })
    }

    fn components(&self, u: f64) -> Result<(f64, f64)> {
        if !u.is_finite() || u <= 0.0 {
            anyhow::bail!("target evaluation requires a finite u > 0");
        }
        let mut base_sum = 0.0;
        let mut parameter_sum = 0.0;
        for n in 1..=self.maximum_terms {
            let input = f64::from(n) * u;
            let x = std::f64::consts::PI * input * input;
            let common = (-x).exp() * input_power_f64(input, self.term_input_power);
            let base_term =
                common * self.polynomial_scale * polynomial_f64(&self.polynomial_coefficients, x);
            let parameter_term = if self.parameter_polynomial_coefficients.is_empty() {
                0.0
            } else {
                common
                    * self.parameter_polynomial_scale
                    * polynomial_f64(&self.parameter_polynomial_coefficients, x)
            };
            base_sum += base_term;
            parameter_sum += parameter_term;
            let scale = base_sum
                .abs()
                .max(parameter_sum.abs())
                .max(f64::MIN_POSITIVE);
            if n >= self.minimum_terms
                && base_term.abs().max(parameter_term.abs()) <= f64::EPSILON * scale
            {
                return Ok((u.sqrt() * base_sum, u.sqrt() * parameter_sum));
            }
        }
        anyhow::bail!("target series did not converge within maximum_terms")
    }
}

/// Binary64 evaluator compiled from one private specification.
#[derive(Clone, Debug)]
pub struct TargetEvaluatorF64 {
    profile_id: String,
    definition_digest: String,
    base: CompiledSeriesF64,
    base_at_one: f64,
    auxiliary: Option<(CompiledSeriesF64, f64)>,
}

impl TargetEvaluatorF64 {
    pub fn from_spec(spec: &TargetProfileSpec) -> Result<Self> {
        spec.validate()?;
        let base = CompiledSeriesF64::new(&spec.base_series)?;
        let (base_at_one, _) = base.components(1.0)?;
        if !base_at_one.is_finite() || base_at_one == 0.0 {
            anyhow::bail!("target base series cannot normalize at u = 1");
        }
        let auxiliary = spec
            .auxiliary_series
            .as_ref()
            .map(|auxiliary| -> Result<_> {
                let compiled = CompiledSeriesF64::new(auxiliary)?;
                let (base_value, parameter_value) = compiled.components(1.0)?;
                if parameter_value == 0.0 || !parameter_value.is_finite() {
                    anyhow::bail!("auxiliary target parameter is singular at u = 1");
                }
                Ok((compiled, -base_value / parameter_value))
            })
            .transpose()?;
        Ok(Self {
            profile_id: spec.profile_id.clone(),
            definition_digest: spec.digest()?,
            base,
            base_at_one,
            auxiliary,
        })
    }

    pub fn from_environment() -> Result<Self> {
        Self::from_spec(&TargetProfileSpec::from_environment()?)
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn definition_digest(&self) -> &str {
        &self.definition_digest
    }

    pub fn value(&self, u: f64) -> f64 {
        self.base
            .components(u)
            .map(|(value, _)| value / self.base_at_one)
            .unwrap_or(f64::NAN)
    }

    pub fn auxiliary_parameter(&self) -> Option<f64> {
        self.auxiliary.as_ref().map(|(_, parameter)| *parameter)
    }

    pub fn auxiliary_value(&self, u: f64) -> Result<f64> {
        let (series, parameter) = self
            .auxiliary
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("target specification has no auxiliary profile"))?;
        let (base, coefficient) = series.components(u)?;
        Ok(base + parameter * coefficient)
    }
}

#[cfg(feature = "hp")]
pub mod hp {
    use super::{GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetProfileSpec};
    use anyhow::{Context, Result};
    use rug::{float::Constant, Float};

    const GUARD_BITS: u32 = 64;

    #[derive(Clone, Debug)]
    struct CompiledSeries {
        working_precision: u32,
        term_input_power: u32,
        polynomial_coefficients: Vec<Float>,
        polynomial_scale: Float,
        parameter_polynomial_coefficients: Vec<Float>,
        parameter_polynomial_scale: Float,
        minimum_terms: u32,
        maximum_terms: u32,
    }

    impl CompiledSeries {
        fn scale(spec: &ScalarScaleSpec, working_precision: u32) -> Result<Float> {
            spec.validate("polynomial scale")?;
            Ok(match spec {
                ScalarScaleSpec::Decimal { value } => {
                    Float::with_val(working_precision, Float::parse(value)?)
                }
                ScalarScaleSpec::RationalTimesSquareRoot {
                    rational_numerator,
                    rational_denominator,
                    radicand_numerator,
                    radicand_denominator,
                } => {
                    let mut radicand = Float::with_val(working_precision, *radicand_numerator);
                    radicand /= *radicand_denominator;
                    let mut value = radicand.sqrt();
                    value *= *rational_numerator;
                    value /= *rational_denominator;
                    value
                }
            })
        }

        fn new(spec: &GaussianPolynomialSeriesSpec, working_precision: u32) -> Result<Self> {
            let parse = |value: &str| -> Result<Float> {
                let parsed = Float::parse(value).context("invalid target coefficient")?;
                let value = Float::with_val(working_precision, parsed);
                if !value.is_finite() {
                    anyhow::bail!("target coefficients must be finite");
                }
                Ok(value)
            };
            Ok(Self {
                working_precision,
                term_input_power: spec.term_input_power,
                polynomial_coefficients: spec
                    .polynomial_coefficients
                    .iter()
                    .map(|value| parse(value))
                    .collect::<Result<_>>()?,
                polynomial_scale: Self::scale(&spec.polynomial_scale, working_precision)?,
                parameter_polynomial_coefficients: spec
                    .parameter_polynomial_coefficients
                    .iter()
                    .map(|value| parse(value))
                    .collect::<Result<_>>()?,
                parameter_polynomial_scale: Self::scale(
                    &spec.parameter_polynomial_scale,
                    working_precision,
                )?,
                minimum_terms: spec.minimum_terms,
                maximum_terms: spec.maximum_terms,
            })
        }

        fn polynomial(&self, coefficients: &[Float], x: &Float) -> Float {
            let mut value = Float::with_val(self.working_precision, 0u32);
            for coefficient in coefficients.iter().rev() {
                value *= x;
                value += coefficient;
            }
            value
        }

        fn input_power(&self, input: &Float) -> Float {
            let mut value = Float::with_val(self.working_precision, 1u32);
            for _ in 0..self.term_input_power {
                value *= input;
            }
            value
        }

        fn components(&self, u: &Float) -> (Float, Float) {
            assert!(
                u.is_finite() && u.is_sign_positive() && *u != 0u32,
                "target evaluation requires a finite u > 0"
            );
            let u = Float::with_val(self.working_precision, u);
            let pi = Float::with_val(self.working_precision, Constant::Pi);
            let threshold = Float::with_val(self.working_precision, 1u32) >> self.working_precision;
            let mut base_sum = Float::with_val(self.working_precision, 0u32);
            let mut parameter_sum = Float::with_val(self.working_precision, 0u32);
            for n in 1..=self.maximum_terms {
                let mut input = u.clone();
                input *= n;
                let mut x = input.clone().square();
                x *= &pi;
                let mut common = (-x.clone()).exp();
                common *= self.input_power(&input);
                let mut base_term = common.clone();
                base_term *= self.polynomial(&self.polynomial_coefficients, &x);
                base_term *= &self.polynomial_scale;
                let mut parameter_term = Float::with_val(self.working_precision, 0u32);
                if !self.parameter_polynomial_coefficients.is_empty() {
                    parameter_term = common;
                    parameter_term *= self.polynomial(&self.parameter_polynomial_coefficients, &x);
                    parameter_term *= &self.parameter_polynomial_scale;
                }
                let negligible = base_term.clone().abs() <= threshold
                    && parameter_term.clone().abs() <= threshold;
                base_sum += base_term;
                parameter_sum += parameter_term;
                if n >= self.minimum_terms && negligible {
                    break;
                }
            }
            let sqrt_u = u.sqrt();
            base_sum *= &sqrt_u;
            parameter_sum *= sqrt_u;
            (base_sum, parameter_sum)
        }
    }

    /// High-precision evaluator compiled once per target-dependent operation.
    #[derive(Clone, Debug)]
    pub struct TargetEvaluator {
        requested_precision: u32,
        profile_id: String,
        definition_digest: String,
        base: CompiledSeries,
        base_at_one: Float,
        auxiliary: Option<(CompiledSeries, Float)>,
    }

    impl TargetEvaluator {
        pub fn from_spec(spec: &TargetProfileSpec, precision_bits: u32) -> Result<Self> {
            spec.validate()?;
            let working = precision_bits.saturating_add(GUARD_BITS);
            let base = CompiledSeries::new(&spec.base_series, working)?;
            let one = Float::with_val(working, 1u32);
            let (base_at_one, _) = base.components(&one);
            if base_at_one == 0u32 || !base_at_one.is_finite() {
                anyhow::bail!("target base series cannot normalize at u = 1");
            }
            let auxiliary = spec
                .auxiliary_series
                .as_ref()
                .map(|auxiliary| -> Result<_> {
                    let compiled = CompiledSeries::new(auxiliary, working)?;
                    let (base_value, parameter_value) = compiled.components(&one);
                    if parameter_value == 0u32 || !parameter_value.is_finite() {
                        anyhow::bail!("auxiliary target parameter is singular at u = 1");
                    }
                    let parameter = Float::with_val(working, -base_value / parameter_value);
                    Ok((compiled, parameter))
                })
                .transpose()?;
            Ok(Self {
                requested_precision: precision_bits,
                profile_id: spec.profile_id.clone(),
                definition_digest: spec.digest()?,
                base,
                base_at_one,
                auxiliary,
            })
        }

        pub fn from_environment(precision_bits: u32) -> Result<Self> {
            Self::from_spec(&TargetProfileSpec::from_environment()?, precision_bits)
        }

        pub fn profile_id(&self) -> &str {
            &self.profile_id
        }

        pub fn definition_digest(&self) -> &str {
            &self.definition_digest
        }

        pub fn value(&self, u: &Float) -> Float {
            let mut value = self.base.components(u).0;
            value /= &self.base_at_one;
            Float::with_val(self.requested_precision, value)
        }

        pub fn auxiliary_parameter(&self) -> Option<Float> {
            self.auxiliary
                .as_ref()
                .map(|(_, parameter)| Float::with_val(self.requested_precision, parameter))
        }

        pub fn auxiliary_value(&self, u: &Float) -> Result<Float> {
            let (series, parameter) = self
                .auxiliary
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("target specification has no auxiliary profile"))?;
            let (mut base, mut coefficient) = series.components(u);
            coefficient *= parameter;
            base += coefficient;
            Ok(Float::with_val(self.requested_precision, base))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn benign_spec() -> TargetProfileSpec {
        testing_profile_spec()
    }

    #[test]
    fn generic_f64_target_normalizes_and_binds_specification() {
        let spec = benign_spec();
        let target = TargetEvaluatorF64::from_spec(&spec).unwrap();
        assert_eq!(target.value(1.0), 1.0);
        assert_eq!(target.definition_digest(), spec.digest().unwrap());
        assert!(target.auxiliary_value(1.0).unwrap().abs() < 1e-14);
    }

    #[test]
    fn malformed_private_specification_is_rejected() {
        let mut spec = benign_spec();
        spec.profile_id = "descriptive value with spaces".to_owned();
        assert!(spec.validate().is_err());
        spec = benign_spec();
        spec.base_series.maximum_terms = 0;
        assert!(spec.validate().is_err());
    }

    #[cfg(feature = "hp")]
    #[test]
    fn generic_hp_target_normalizes_and_solves_auxiliary_parameter() {
        use rug::Float;

        let target = hp::TargetEvaluator::from_spec(&benign_spec(), 256).unwrap();
        let one = Float::with_val(256, 1u32);
        assert_eq!(target.value(&one), 1u32);
        assert!(target.auxiliary_value(&one).unwrap().abs() < Float::with_val(256, 1e-70));
    }
}
