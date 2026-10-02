// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Runtime-supplied target profiles.
//!
//! The toolkit deliberately contains no research-target coefficients. A claim
//! runner supplies a canonical JSON specification through
//! `XC_TARGET_SPEC_FILE`. The public implementation evaluates a generic
//! legacy Gaussian series or an explicitly authorized external provider. The SHA-256
//! digest of the complete specification binds every target-derived artifact.
//! The specification text and coefficients are never copied into an artifact.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[path = "target_external.rs"]
mod external;
pub use external::ExternalProfileSpec;

/// Environment variable naming the private target-profile specification.
pub const TARGET_SPEC_FILE_ENV: &str = "XC_TARGET_SPEC_FILE";

/// Whether a runtime target specification is available to this process.
/// Target-derived measurements that a capture level requests automatically
/// are recorded as missing, not failed, when it is not.
pub fn runtime_target_configured() -> bool {
    cfg!(test) || std::env::var_os(TARGET_SPEC_FILE_ENV).is_some_and(|path| !path.is_empty())
}

// Decimal coefficients can become significant after normalization or polynomial
// evaluation. Do not interpret a nonzero input outside binary64 range as zero.
fn checked_decimal_f64(text: &str) -> Result<f64> {
    let value = text.parse::<f64>().context("invalid target decimal")?;
    anyhow::ensure!(value.is_finite(), "target decimal must be finite");
    anyhow::ensure!(
        value != 0.0 || xc_core::DecimalLiteral::new(text)?.canonical()?.as_str() == "0",
        "nonzero target decimal underflows binary64; use the HP evaluator"
    );
    anyhow::ensure!(
        value == 0.0 || value.is_normal(),
        "nonzero target decimal is subnormal in binary64 and loses relative precision; use the HP evaluator"
    );
    Ok(value)
}

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
                xc_core::DecimalLiteral::new(value)
                    .with_context(|| format!("{field} contains an invalid decimal scale"))?;
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
            Self::Decimal { value } => checked_decimal_f64(value)?,
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
            xc_core::DecimalLiteral::new(coefficient)
                .with_context(|| format!("{field} contains an invalid decimal coefficient"))?;
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
    /// Opaque, non-descriptive identifier chosen by the caller.
    pub profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_series: Option<GaussianPolynomialSeriesSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_profile: Option<ExternalProfileSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auxiliary_series: Option<GaussianPolynomialSeriesSpec>,
}

impl TargetProfileSpec {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.schema_version, 1 | 3) {
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
        match (
            &self.base_series,
            &self.external_profile,
            self.schema_version,
        ) {
            (Some(base), None, 1) => {
                base.validate("base_series")?;
                if !base.parameter_polynomial_coefficients.is_empty() {
                    anyhow::bail!("base_series cannot contain a solved parameter");
                }
            }
            (None, Some(source), 3) => source.validate()?,
            _ => anyhow::bail!("use schema 1 with base_series or schema 3 with external_profile"),
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
        // The external protocol is part of the measurement semantics: a cached
        // result from the uncorrelated protocol must not bypass the new checks.
        // Gaussian-series termination is part of the value semantics too.
        // Historical first-small-term results must not bypass the tail checks.
        let bytes = if self.external_profile.is_some() {
            if self.auxiliary_series.is_some() {
                serde_json::to_vec(&(
                    "external-target-provider-nonce-bound-working-input-v3",
                    "gaussian-series-exponent-floor-and-zero-tail-v5",
                    self,
                ))?
            } else {
                serde_json::to_vec(&(
                    "external-target-provider-nonce-bound-working-input-v3",
                    self,
                ))?
            }
        } else {
            serde_json::to_vec(&("gaussian-series-exponent-floor-and-zero-tail-v5", self))?
        };
        Ok(xc_cache::ContentDigest::sha256(&bytes).0)
    }
}

#[cfg(test)]
fn testing_profile_spec() -> TargetProfileSpec {
    TargetProfileSpec {
        schema_version: 1,
        profile_id: "toolkit-benign-test-profile-v1".to_owned(),
        base_series: Some(GaussianPolynomialSeriesSpec {
            term_input_power: 0,
            polynomial_coefficients: vec!["1".to_owned()],
            polynomial_scale: ScalarScaleSpec::default(),
            parameter_polynomial_coefficients: Vec::new(),
            parameter_polynomial_scale: ScalarScaleSpec::default(),
            minimum_terms: 2,
            maximum_terms: 1000,
        }),
        external_profile: None,
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
        let parse = checked_decimal_f64;
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

    fn normalized_base(spec: &GaussianPolynomialSeriesSpec) -> Result<Self> {
        let mut result = Self::new(spec)?;
        let maximum = result
            .polynomial_coefficients
            .iter()
            .map(|c| c.abs())
            .fold(0.0, f64::max);
        anyhow::ensure!(
            maximum > 0.0 && result.polynomial_scale != 0.0,
            "target base series cannot normalize a zero polynomial or scale"
        );
        // These constant factors cancel in base(u)/base(1). Remove them before
        // evaluating, so a harmless small scale cannot cause underflow.
        for coefficient in &mut result.polynomial_coefficients {
            let original = *coefficient;
            *coefficient /= maximum;
            anyhow::ensure!(
                coefficient.is_finite() && (original == 0.0 || coefficient.is_normal()),
                "nonzero target coefficient loses binary64 relative precision during normalization; use the HP evaluator"
            );
        }
        result.polynomial_scale = 1.0;
        result.parameter_polynomial_coefficients.clear();
        Ok(result)
    }

    fn term(&self, input: f64, x: f64, common: f64, coefficients: &[f64], scale: f64) -> f64 {
        if coefficients.is_empty() || scale == 0.0 || x.is_infinite() {
            return 0.0;
        }
        let polynomial = polynomial_f64(coefficients, x);
        let scaled_common = common * scale;
        let direct = scaled_common * polynomial;
        // A nonzero subnormal intermediate may already have lost most of its
        // precision before a later factor restores the final value's range.
        if (-x).exp().is_normal()
            && common.is_normal()
            && scale.is_normal()
            && scaled_common.is_normal()
            && polynomial.is_finite()
            && (direct.is_normal() || polynomial == 0.0)
        {
            return direct;
        }
        // Avoid 0*infinity when the Gaussian and polynomial factors separately
        // exceed the hardware range although their product is representable.
        let common_log = scale.abs().ln() + f64::from(self.term_input_power) * input.ln() - x;
        coefficients
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0.0)
            .map(|(k, c)| {
                (common_log + c.abs().ln() + k as f64 * x.ln()).exp() * c.signum() * scale.signum()
            })
            .sum()
    }

    /// Bound the remaining absolute monomial series in the log domain.
    /// For m >= n+1 and degree d, t_(m+1)/t_m is at most
    /// exp(d/m - pi*u^2*(2*m+1)), a decreasing function of m.
    /// Once this is <= 1/2, the entire tail is <= twice its first
    /// absolute-monomial envelope. A zero of P cannot hide later terms.
    fn tail_is_negligible(
        &self,
        u: f64,
        n: u32,
        coefficients: &[f64],
        scale: f64,
        sum: f64,
        normalization_floor: f64,
    ) -> bool {
        let Some(degree) = coefficients.iter().rposition(|c| *c != 0.0) else {
            return true;
        };
        if scale == 0.0 {
            return true;
        }
        let scale_of_sum = sum.abs().max(normalization_floor);
        // If every accumulated term rounded to zero, bound the entire series,
        // including those lost terms, before accepting a rounded zero.
        let n = if scale_of_sum == 0.0 { 0 } else { n };
        let m = f64::from(n) + 1.0;
        let degree = f64::from(self.term_input_power) + 2.0 * degree as f64;
        let decay = std::f64::consts::PI * u * u * (2.0 * m + 1.0);
        let growth = degree / m;
        if growth - decay + 64.0 * f64::EPSILON * (1.0 + growth + decay) > -std::f64::consts::LN_2 {
            return false;
        }
        let input = m * u;
        let x = std::f64::consts::PI * input * input;
        if x.is_infinite() {
            return true; // All monomial tails are below binary64 range.
        }
        let common_log = scale.abs().ln() + f64::from(self.term_input_power) * input.ln() - x;
        let logs: Vec<f64> = coefficients
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0.0)
            .map(|(k, c)| common_log + c.abs().ln() + k as f64 * x.ln())
            .collect();
        let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_tail = std::f64::consts::LN_2
            + maximum
            + logs.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln();
        if scale_of_sum == 0.0 {
            let rounding_margin = 64.0 * f64::EPSILON * (1.0 + log_tail.abs());
            log_tail + rounding_margin < f64::from_bits(1).ln() - std::f64::consts::LN_2
        } else {
            log_tail <= scale_of_sum.ln() + (f64::EPSILON / 4.0).ln()
        }
    }

    fn components(&self, u: f64) -> Result<(f64, f64)> {
        self.components_with_normalization(u, 0.0)
    }

    fn components_with_normalization(&self, u: f64, normalization: f64) -> Result<(f64, f64)> {
        if !u.is_finite() || u <= 0.0 {
            anyhow::bail!("target evaluation requires a finite u > 0");
        }
        let mut base_sum = 0.0;
        let mut parameter_sum = 0.0;
        for n in 1..=self.maximum_terms {
            let input = f64::from(n) * u;
            let x = std::f64::consts::PI * input * input;
            let common = (-x).exp() * input_power_f64(input, self.term_input_power);
            let base_term = self.term(
                input,
                x,
                common,
                &self.polynomial_coefficients,
                self.polynomial_scale,
            );
            let parameter_term = self.term(
                input,
                x,
                common,
                &self.parameter_polynomial_coefficients,
                self.parameter_polynomial_scale,
            );
            base_sum += base_term;
            parameter_sum += parameter_term;
            anyhow::ensure!(
                base_sum.is_finite() && parameter_sum.is_finite(),
                "target series produced a nonfinite partial sum"
            );
            if n >= self.minimum_terms
                && self.tail_is_negligible(
                    u,
                    n,
                    &self.polynomial_coefficients,
                    self.polynomial_scale,
                    base_sum,
                    normalization.abs() / u.sqrt(),
                )
                && self.tail_is_negligible(
                    u,
                    n,
                    &self.parameter_polynomial_coefficients,
                    self.parameter_polynomial_scale,
                    parameter_sum,
                    0.0,
                )
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
    base: Option<CompiledSeriesF64>,
    external: Option<external::CompiledF64>,
    base_at_one: f64,
    auxiliary: Option<(CompiledSeriesF64, f64)>,
}

impl TargetEvaluatorF64 {
    pub fn from_spec(spec: &TargetProfileSpec) -> Result<Self> {
        spec.validate()?;
        let base = spec
            .base_series
            .as_ref()
            .map(CompiledSeriesF64::normalized_base)
            .transpose()?;
        let external = spec
            .external_profile
            .as_ref()
            .map(external::CompiledF64::new)
            .transpose()?;
        let base_at_one = if let Some(source) = &external {
            source.raw(1.0)?
        } else {
            base.as_ref().expect("validated series").components(1.0)?.0
        };
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
                let parameter = -base_value / parameter_value;
                anyhow::ensure!(
                    parameter.is_finite() && (base_value == 0.0 || parameter != 0.0),
                    "auxiliary target parameter is not representable"
                );
                Ok((compiled, parameter))
            })
            .transpose()?;
        Ok(Self {
            profile_id: spec.profile_id.clone(),
            definition_digest: spec.digest()?,
            base,
            external,
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

    /// Reject use of a cutoff-specific source in a different configuration.
    pub fn validate_lambda(&self, lambda: f64) -> Result<()> {
        if let Some(source) = &self.external {
            source.validate_lambda(lambda)?;
        }
        Ok(())
    }

    /// Evaluate a normalized point, preserving the underlying provider error.
    pub fn try_value(&self, u: f64) -> Result<f64> {
        let result = if let Some(source) = &self.external {
            source.raw(u)
        } else {
            self.base
                .as_ref()
                .expect("validated series")
                .components_with_normalization(u, self.base_at_one)
                .map(|x| x.0)
        };
        let value = result? / self.base_at_one;
        anyhow::ensure!(
            value.is_finite(),
            "target evaluation produced a nonfinite value"
        );
        Ok(value)
    }

    /// Scalar callback compatibility; use [`Self::try_value`] to retain errors.
    pub fn value(&self, u: f64) -> f64 {
        self.try_value(u).unwrap_or(f64::NAN)
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
        let value = base + parameter * coefficient;
        anyhow::ensure!(
            value.is_finite(),
            "auxiliary target produced a nonfinite value"
        );
        Ok(value)
    }
}

#[cfg(feature = "hp")]
pub mod hp {
    use super::{GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetProfileSpec};
    use anyhow::{Context, Result};
    use rug::{
        float::{Constant, Round},
        ops::AddAssignRound,
        Float,
    };

    const GUARD_BITS: u32 = 64;

    fn checked_round(value: &Float, precision: u32, field: &str) -> Result<Float> {
        let rounded = Float::with_val(precision, value);
        anyhow::ensure!(
            rounded.is_finite() && (value == &0 || rounded != 0),
            "{field} is not representable at requested precision"
        );
        Ok(rounded)
    }

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
        fn decimal(text: &str, precision: u32) -> Result<Float> {
            let parsed = Float::parse(text).context("invalid target decimal")?;
            let (value, ordering) = Float::with_val_round(precision, parsed, Round::Nearest);
            anyhow::ensure!(
                value.is_finite(),
                "target decimal is not finite at working precision"
            );
            anyhow::ensure!(
                value != 0 || ordering == std::cmp::Ordering::Equal,
                "nonzero target decimal is below the supported exponent range"
            );
            anyhow::ensure!(
                ordering == std::cmp::Ordering::Equal
                    || value.get_exp() != Some(rug::float::exp_min()),
                "target decimal loses relative precision at the MPFR exponent floor"
            );
            Ok(value)
        }

        /// Logarithms of absolute monomials, including every multiplicative
        /// factor before exponentiation. Coefficients equal to zero are omitted.
        fn monomial_logs(
            &self,
            input: &Float,
            x: &Float,
            coefficients: &[Float],
            scale: &Float,
        ) -> Result<Vec<(Float, bool)>> {
            anyhow::ensure!(
                input.is_finite() && input > &0 && x.is_finite() && x > &0,
                "target series argument is outside the supported exponent range"
            );
            if scale == &0 {
                return Ok(Vec::new());
            }
            let mut common = scale.clone().abs().ln();
            if self.term_input_power != 0 {
                common += input.clone().ln() * self.term_input_power;
            }
            common -= x;
            let log_x = x.clone().ln();
            let mut logs = Vec::new();
            for (k, coefficient) in coefficients.iter().enumerate() {
                if coefficient == &0 {
                    continue;
                }
                let mut log = common.clone();
                log += coefficient.clone().abs().ln();
                if k != 0 {
                    log += Float::with_val(self.working_precision, &log_x * k as u32);
                }
                anyhow::ensure!(
                    log.is_finite(),
                    "target monomial logarithm is not representable"
                );
                logs.push((
                    log,
                    coefficient.is_sign_negative() != scale.is_sign_negative(),
                ));
            }
            Ok(logs)
        }

        /// Preserve ordinary arithmetic away from the exponent boundary. When
        /// its separate factors lose range, evaluate signed monomials in the
        /// log domain. Budget a smallest-positive value for every exponential
        /// rounded at the lower exponent boundary; do not silently certify zero.
        fn term(
            &self,
            input: &Float,
            x: &Float,
            coefficients: &[Float],
            scale: &Float,
            range_error: &mut Float,
        ) -> Result<Float> {
            let p = self.working_precision;
            if scale == &0 || coefficients.iter().all(|c| c == &0) {
                return Ok(Float::with_val(p, 0));
            }
            let gaussian = (-x.clone()).exp();
            let common = Float::with_val(p, &gaussian * self.input_power(input));
            let polynomial = self.polynomial(coefficients, x);
            let product = Float::with_val(p, &common * &polynomial);
            let direct = Float::with_val(p, &product * scale);
            let above_floor = |v: &Float| v.get_exp().is_some_and(|e| e > rug::float::exp_min());
            if above_floor(&gaussian)
                && common.is_finite()
                && above_floor(&common)
                && polynomial.is_finite()
                && (above_floor(&product) || polynomial == 0)
                && direct.is_finite()
                && (above_floor(&direct) || polynomial == 0)
            {
                return Ok(direct);
            }
            let mut result = Float::with_val(p, 0);
            for (log, negative) in self.monomial_logs(input, x, coefficients, scale)? {
                let mut term = log.exp();
                anyhow::ensure!(term.is_finite(), "target monomial is not representable");
                if !above_floor(&term) {
                    let mut floor = Float::with_val(p, 0);
                    floor.next_up();
                    range_error.add_assign_round(floor, Round::Up);
                }
                if negative {
                    term = -term;
                }
                result += term;
            }
            anyhow::ensure!(result.is_finite(), "target term is not representable");
            Ok(result)
        }

        fn range_error_is_negligible(&self, error: &Float, sum: &Float) -> bool {
            if error == &0 {
                return true;
            }
            if sum == &0 {
                return false;
            }
            let budget = sum.clone().abs().ln()
                - Float::with_val(self.working_precision, 2).ln() * self.working_precision;
            error.clone().ln() <= budget
        }

        fn scale(spec: &ScalarScaleSpec, working_precision: u32) -> Result<Float> {
            spec.validate("polynomial scale")?;
            Ok(match spec {
                ScalarScaleSpec::Decimal { value } => Self::decimal(value, working_precision)?,
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
            let parse = |value: &str| Self::decimal(value, working_precision);
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

        /// Absolute-monomial geometric tail, compared in the log domain so
        /// neither the tail nor its relative budget can silently underflow.
        /// These are computed estimates, not outward-rounded certificates.
        fn tail_is_negligible(
            &self,
            u: &Float,
            n: u32,
            coefficients: &[Float],
            scale: &Float,
            sum: &Float,
        ) -> Result<bool> {
            let Some(degree) = coefficients.iter().rposition(|c| c != &0) else {
                return Ok(true);
            };
            if scale == &0 {
                return Ok(true);
            }
            if sum == &0 {
                return Ok(false);
            }
            let p = self.working_precision;
            let m = n + 1;
            let pi = Float::with_val(p, Constant::Pi);
            let mut decay = u.clone().square();
            decay *= &pi;
            decay *= 2 * m + 1;
            anyhow::ensure!(
                decay.is_finite(),
                "target tail argument is outside the supported exponent range"
            );
            let mut log_ratio = Float::with_val(p, self.term_input_power + 2 * degree as u32);
            log_ratio /= m;
            log_ratio -= decay;
            // The stronger ratio <= exp(-1) leaves a margin below 1/2.
            if log_ratio > -1 {
                return Ok(false);
            }
            let input = Float::with_val(p, u * m);
            let x = input.clone().square() * pi;
            let logs = self.monomial_logs(&input, &x, coefficients, scale)?;
            let maximum = logs
                .iter()
                .map(|(v, _)| v)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("nonzero polynomial");
            let mut envelope = Float::with_val(p, 0);
            for (log, _) in &logs {
                envelope += Float::with_val(p, log - maximum).exp();
            }
            let ln2 = Float::with_val(p, 2).ln();
            let log_tail = envelope.ln() + maximum + &ln2;
            let log_budget = sum.clone().abs().ln() - ln2 * p;
            Ok(log_tail <= log_budget)
        }

        fn components(&self, u: &Float) -> Result<(Float, Float)> {
            anyhow::ensure!(
                u.is_finite() && u > &0,
                "target evaluation requires a finite u > 0"
            );
            let u = Float::with_val(self.working_precision, u);
            let pi = Float::with_val(self.working_precision, Constant::Pi);
            let mut base_sum = Float::with_val(self.working_precision, 0u32);
            let mut parameter_sum = Float::with_val(self.working_precision, 0u32);
            let mut base_range_error = Float::with_val(self.working_precision, 0);
            let mut parameter_range_error = Float::with_val(self.working_precision, 0);
            for n in 1..=self.maximum_terms {
                let mut input = u.clone();
                input *= n;
                let mut x = input.clone().square();
                x *= &pi;
                let base_term = self.term(
                    &input,
                    &x,
                    &self.polynomial_coefficients,
                    &self.polynomial_scale,
                    &mut base_range_error,
                )?;
                let parameter_term = self.term(
                    &input,
                    &x,
                    &self.parameter_polynomial_coefficients,
                    &self.parameter_polynomial_scale,
                    &mut parameter_range_error,
                )?;
                base_sum += base_term;
                parameter_sum += parameter_term;
                anyhow::ensure!(
                    base_sum.is_finite() && parameter_sum.is_finite(),
                    "target series produced a nonfinite partial sum"
                );
                if n >= self.minimum_terms
                    && self.tail_is_negligible(
                        &u,
                        n,
                        &self.polynomial_coefficients,
                        &self.polynomial_scale,
                        &base_sum,
                    )?
                    && self.tail_is_negligible(
                        &u,
                        n,
                        &self.parameter_polynomial_coefficients,
                        &self.parameter_polynomial_scale,
                        &parameter_sum,
                    )?
                    && self.range_error_is_negligible(&base_range_error, &base_sum)
                    && self.range_error_is_negligible(&parameter_range_error, &parameter_sum)
                {
                    let sqrt_u = u.sqrt();
                    base_sum *= &sqrt_u;
                    parameter_sum *= sqrt_u;
                    anyhow::ensure!(
                        base_sum.is_finite() && parameter_sum.is_finite(),
                        "target series produced a nonfinite value"
                    );
                    return Ok((base_sum, parameter_sum));
                }
            }
            anyhow::ensure!(
                self.range_error_is_negligible(&base_range_error, &base_sum)
                    && self.range_error_is_negligible(&parameter_range_error, &parameter_sum),
                "target series is outside the supported exponent range at working precision"
            );
            anyhow::bail!("target series did not converge within maximum_terms")
        }
    }

    /// High-precision evaluator compiled once per target-dependent operation.
    #[derive(Clone, Debug)]
    pub struct TargetEvaluator {
        requested_precision: u32,
        profile_id: String,
        definition_digest: String,
        base: Option<CompiledSeries>,
        external: Option<super::external::hp::Compiled>,
        base_at_one: Float,
        auxiliary: Option<(CompiledSeries, Float)>,
    }

    impl TargetEvaluator {
        pub fn from_spec(spec: &TargetProfileSpec, precision_bits: u32) -> Result<Self> {
            spec.validate()?;
            anyhow::ensure!(
                (rug::float::prec_min()..=1_000_000).contains(&precision_bits),
                "target precision must be between 1 and 1000000 bits"
            );
            let working = precision_bits
                .checked_add(GUARD_BITS)
                .context("target precision overflow")?;
            let base = spec
                .base_series
                .as_ref()
                .map(|s| CompiledSeries::new(s, working))
                .transpose()?;
            let external = spec
                .external_profile
                .as_ref()
                .map(|s| super::external::hp::Compiled::new(s, working))
                .transpose()?;
            let one = Float::with_val(working, 1u32);
            let base_at_one = if let Some(source) = &external {
                source.raw(&one)?
            } else {
                base.as_ref().expect("validated series").components(&one)?.0
            };
            if base_at_one == 0u32 || !base_at_one.is_finite() {
                anyhow::bail!("target base series cannot normalize at u = 1");
            }
            let auxiliary = spec
                .auxiliary_series
                .as_ref()
                .map(|auxiliary| -> Result<_> {
                    let compiled = CompiledSeries::new(auxiliary, working)?;
                    let (base_value, parameter_value) = compiled.components(&one)?;
                    if parameter_value == 0u32 || !parameter_value.is_finite() {
                        anyhow::bail!("auxiliary target parameter is singular at u = 1");
                    }
                    let parameter = Float::with_val(working, -base_value.clone() / parameter_value);
                    anyhow::ensure!(
                        parameter.is_finite() && (base_value == 0 || parameter != 0),
                        "auxiliary target parameter is not representable"
                    );
                    checked_round(&parameter, precision_bits, "auxiliary target parameter")?;
                    Ok((compiled, parameter))
                })
                .transpose()?;
            Ok(Self {
                requested_precision: precision_bits,
                profile_id: spec.profile_id.clone(),
                definition_digest: spec.digest()?,
                base,
                external,
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

        /// Reject use of a cutoff-specific source in a different configuration.
        pub fn validate_lambda(&self, lambda: &Float) -> Result<()> {
            if let Some(source) = &self.external {
                source.validate_lambda(lambda)?;
            }
            Ok(())
        }

        /// Evaluate a normalized point, preserving the underlying provider error.
        pub fn try_value(&self, u: &Float) -> Result<Float> {
            anyhow::ensure!(u.is_finite() && u > &0, "target requires finite u > 0");
            let mut value = if let Some(source) = &self.external {
                source.raw(u)?
            } else {
                self.base
                    .as_ref()
                    .expect("validated series")
                    .components(u)?
                    .0
            };
            let nonzero = value != 0;
            value /= &self.base_at_one;
            anyhow::ensure!(
                !nonzero || value != 0,
                "normalized target is below the supported exponent range"
            );
            anyhow::ensure!(
                value.is_finite(),
                "target evaluation produced a nonfinite value"
            );
            checked_round(&value, self.requested_precision, "target value")
        }

        /// The unnormalized target at `u = 1`, which every value is divided by.
        pub fn normalization(&self) -> Float {
            Float::with_val(self.requested_precision, &self.base_at_one)
        }

        /// Scalar callback compatibility; use [`Self::try_value`] to retain errors.
        pub fn value(&self, u: &Float) -> Float {
            self.try_value(u)
                .unwrap_or_else(|_| Float::with_val(self.requested_precision, f64::NAN))
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
            let (mut base, mut coefficient) = series.components(u)?;
            coefficient *= parameter;
            base += coefficient;
            anyhow::ensure!(
                base.is_finite(),
                "auxiliary target produced a nonfinite value"
            );
            checked_round(&base, self.requested_precision, "auxiliary target value")
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
    fn external_schema_is_unambiguous_and_content_bound() {
        let mut spec = benign_spec();
        let legacy = spec.digest().unwrap();
        let value = serde_json::to_value(&spec).unwrap();
        assert!(value.get("external_profile").is_none());
        assert_eq!(
            legacy,
            TargetProfileSpec::from_json(&serde_json::to_vec(&value).unwrap())
                .unwrap()
                .digest()
                .unwrap()
        );
        spec.schema_version = 3;
        spec.external_profile = Some(ExternalProfileSpec {
            lambda_squared: "4".into(),
            evaluation_precision_bits: 256,
            provider_sha256: "a".repeat(64),
            input: serde_json::json!({"fixture":1}),
        });
        assert!(spec.validate().is_err());
        spec.base_series = None;
        let first = spec.digest().unwrap();
        assert_ne!(
            first,
            xc_cache::ContentDigest::sha256(&serde_json::to_vec(&spec).unwrap()).0,
            "old uncorrelated-provider cache identities must not be reused"
        );
        spec.external_profile.as_mut().unwrap().input = serde_json::json!({"fixture":2});
        assert_ne!(first, spec.digest().unwrap());
        let second = spec.digest().unwrap();
        spec.external_profile.as_mut().unwrap().provider_sha256 = "b".repeat(64);
        assert_ne!(second, spec.digest().unwrap());
        spec.schema_version = 2;
        assert!(spec.validate().is_err());
    }

    #[test]
    fn malformed_private_specification_is_rejected() {
        let mut spec = benign_spec();
        spec.profile_id = "descriptive value with spaces".to_owned();
        assert!(spec.validate().is_err());
        spec = benign_spec();
        spec.base_series.as_mut().unwrap().maximum_terms = 0;
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

#[cfg(test)]
mod summation_regressions {
    use super::*;

    #[test]
    fn generic_f64_target_does_not_stop_at_a_polynomial_zero() {
        let mut spec = testing_profile_spec();
        spec.auxiliary_series = None;
        let series = spec.base_series.as_mut().unwrap();
        series.polynomial_coefficients =
            vec![(-4.0 * std::f64::consts::PI).to_string(), "1".into()];
        let constant: f64 = series.polynomial_coefficients[0].parse().unwrap();
        let direct = |u: f64| {
            u.sqrt()
                * (1..=40)
                    .map(|n| {
                        let x = std::f64::consts::PI * (f64::from(n) * u).powi(2);
                        (-x).exp() * (x + constant)
                    })
                    .sum::<f64>()
        };
        let target = TargetEvaluatorF64::from_spec(&spec).unwrap();
        assert!((target.try_value(1.1).unwrap() - direct(1.1) / direct(1.0)).abs() < 2e-15);
    }

    #[cfg(feature = "hp")]
    #[test]
    fn generic_hp_target_normalization_is_invariant_under_series_scaling() {
        use rug::Float;
        let mut spec = testing_profile_spec();
        spec.auxiliary_series = None;
        let first = hp::TargetEvaluator::from_spec(&spec, 128).unwrap();
        spec.base_series.as_mut().unwrap().polynomial_scale = ScalarScaleSpec::Decimal {
            value: "1e-100".into(),
        };
        let second = hp::TargetEvaluator::from_spec(&spec, 128).unwrap();
        let u = Float::with_val(128, Float::parse("1.1").unwrap());
        let difference = Float::with_val(
            128,
            first.try_value(&u).unwrap() - second.try_value(&u).unwrap(),
        )
        .abs();
        assert!(difference < (Float::with_val(128, 1) >> 120));
    }

    #[cfg(feature = "hp")]
    #[test]
    fn hp_target_admits_coefficients_beyond_binary64_range() {
        use rug::Float;
        let mut spec = testing_profile_spec();
        spec.auxiliary_series = None;
        let reference = hp::TargetEvaluator::from_spec(&spec, 256).unwrap();
        spec.base_series.as_mut().unwrap().polynomial_coefficients = vec!["1e400".into()];
        spec.validate().unwrap();
        assert!(TargetEvaluatorF64::from_spec(&spec).is_err());
        let scaled = hp::TargetEvaluator::from_spec(&spec, 256).unwrap();
        let u = Float::with_val(256, Float::parse("1.1").unwrap());
        let difference = Float::with_val(
            256,
            reference.try_value(&u).unwrap() - scaled.try_value(&u).unwrap(),
        )
        .abs();
        assert!(difference < (Float::with_val(256, 1) >> 235));
    }

    #[cfg(feature = "hp")]
    #[test]
    fn generic_hp_target_rejects_an_inadequate_term_budget() {
        let mut spec = testing_profile_spec();
        spec.auxiliary_series = None;
        spec.base_series.as_mut().unwrap().maximum_terms = 2;
        assert!(hp::TargetEvaluator::from_spec(&spec, 128).is_err());
    }
    #[test]
    fn generic_f64_target_handles_scaling_and_underflow_after_normalization() {
        let mut spec = testing_profile_spec();
        spec.auxiliary_series = None;
        let first = TargetEvaluatorF64::from_spec(&spec).unwrap();
        spec.base_series.as_mut().unwrap().polynomial_scale = ScalarScaleSpec::Decimal {
            value: "1e-300".into(),
        };
        spec.base_series.as_mut().unwrap().polynomial_coefficients = vec!["1e-200".into()];
        let second = TargetEvaluatorF64::from_spec(&spec).unwrap();
        for u in [1.0, 1.1, 3.0, 20.0, 100.0, 1e300] {
            let left = first.try_value(u).unwrap();
            let right = second.try_value(u).unwrap();
            assert!((left - right).abs() < 2e-15);
        }
        assert_eq!(first.try_value(100.0).unwrap(), 0.0);
    }

    #[test]
    fn generic_series_identity_rejects_first_small_term_cache_epoch() {
        let spec = testing_profile_spec();
        let legacy = xc_cache::ContentDigest::sha256(&serde_json::to_vec(&spec).unwrap()).0;
        assert_ne!(spec.digest().unwrap(), legacy);
    }
}
