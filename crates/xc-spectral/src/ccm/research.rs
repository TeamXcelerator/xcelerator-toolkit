// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Opt-in CCM research primitives. These do not change claim capture presets,
//! silently populate legacy caches, or promote point calculations to proofs.
//!
//! All numerical work stays in MPFR or exact rationals. The structured prime
//! route deliberately has a distinct identity: regrouping an exact formula
//! need not reproduce the rounding of the canonical cell-by-cell route.

use anyhow::{anyhow, bail, Result};
use rug::float::Constant;
use rug::{ops::Pow, Float, Integer, Rational};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
use xc_numerics::mpfr_interval::MpfrInterval;

#[cfg(test)]
use super::prime_powers_up_to;
#[path = "research_exact.rs"]
mod exact;
#[path = "research_prime.rs"]
mod prime_math;
pub const FINITE_DIAGNOSTIC_ARITHMETIC: &str = "exact-stored-point-research-v0.15.2-v1";

pub const RESEARCH_ASSEMBLY_SEMANTICS: &str =
    "ccm-exact-input-research-assembly-length-aware-arch-v2";
pub const AGGREGATE_PRIME_SEMANTICS: &str = "ccm-prime-divided-difference-generators-v0.15.2-v1";

/// Exact cutoff input. The active prime set is derived from the rational
/// floor, never from a rounded floating-point display value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactCutoff {
    value: Rational,
    prime_cutoff: u64,
}

impl ExactCutoff {
    /// Parse an integer, exact decimal (optionally scientific notation), or
    /// rational numerator/denominator. Input size and exponent are bounded to
    /// prevent accidental unbounded integer allocation during parsing.
    pub fn parse(literal: &str) -> Result<Self> {
        let literal = literal.trim();
        if literal.is_empty() || literal.len() > 4096 || !literal.is_ascii() {
            bail!("cutoff literal must contain 1..4096 ASCII characters");
        }
        let literal = literal.strip_prefix('+').unwrap_or(literal);
        let value = if let Some((numerator, denominator)) = literal.split_once('/') {
            if numerator.is_empty()
                || denominator.is_empty()
                || !numerator.bytes().all(|b| b.is_ascii_digit())
                || !denominator.bytes().all(|b| b.is_ascii_digit())
            {
                bail!("invalid rational cutoff");
            }
            let numerator = Integer::from_str_radix(numerator, 10)?;
            let denominator = Integer::from_str_radix(denominator, 10)?;
            if denominator <= 0 {
                bail!("cutoff denominator must be positive");
            }
            Rational::from((numerator, denominator))
        } else {
            let mut pieces = literal.split(['e', 'E']);
            let mantissa = pieces.next().ok_or_else(|| anyhow!("missing mantissa"))?;
            let exponent = match pieces.next() {
                Some(value) => value.parse::<i32>()?,
                None => 0,
            };
            if pieces.next().is_some() || exponent.unsigned_abs() > 4096 {
                bail!("invalid or excessive cutoff exponent");
            }
            let (whole, fractional) = mantissa.split_once('.').unwrap_or((mantissa, ""));
            if whole.is_empty() && fractional.is_empty() {
                bail!("empty cutoff mantissa");
            }
            if !whole
                .bytes()
                .chain(fractional.bytes())
                .all(|b| b.is_ascii_digit())
            {
                bail!("cutoff mantissa must be a nonnegative decimal");
            }
            let digits = format!("{whole}{fractional}");
            let mut numerator = Integer::from_str_radix(&digits, 10)?;
            let scale = i32::try_from(fractional.len())? - exponent;
            let denominator = if scale >= 0 {
                Integer::from(10).pow(scale as u32)
            } else {
                numerator *= Integer::from(10).pow(scale.unsigned_abs());
                Integer::from(1)
            };
            Rational::from((numerator, denominator))
        };
        Self::from_rational(value)
    }

    /// Construct from an exact rational within a 1,048,576-bit budget for
    /// each of its numerator and denominator, before division or promotion.
    pub fn from_rational(value: Rational) -> Result<Self> {
        if value.numer().significant_bits() > 1_048_576
            || value.denom().significant_bits() > 1_048_576
        {
            bail!("exact cutoff numerator or denominator exceeds the rational bit budget");
        }
        if value <= 1 {
            bail!("CCM cutoff must be greater than one");
        }
        let mut floor = value.numer().clone();
        floor /= value.denom();
        let prime_cutoff = floor
            .to_u64()
            .ok_or_else(|| anyhow!("cutoff floor exceeds u64"))?;
        Ok(Self {
            value,
            prime_cutoff,
        })
    }

    pub fn value(&self) -> &Rational {
        &self.value
    }
    pub fn prime_cutoff(&self) -> u64 {
        self.prime_cutoff
    }
    pub fn canonical(&self) -> String {
        format!("{}/{}", self.value.numer(), self.value.denom())
    }
    /// Correctly rounded log of the exact rational cutoff. Close to one,
    /// form C-1 exactly before directed log1p; never round C to one first.
    pub fn log_length(&self, precision_bits: u32) -> Result<Float> {
        use rug::float::Round;
        require_precision(precision_bits)?;
        let delta = Rational::from(&self.value - 1);
        for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
            let work = precision_bits + guard;
            let mut lower = Float::with_val_round(work, &delta, Round::Down).0;
            let mut upper = Float::with_val_round(work, &delta, Round::Up).0;
            lower.ln_1p_round(Round::Down);
            upper.ln_1p_round(Round::Up);
            let lo = Float::with_val(precision_bits, lower);
            let hi = Float::with_val(precision_bits, upper);
            if lo == hi && lo.is_finite() && lo > 0 {
                return Ok(lo);
            }
        }
        bail!("exact rational cutoff logarithm unresolved within 4096 guard bits")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimeAssemblyRoute {
    CanonicalCellSum,
    AggregateGenerators,
}

/// Explicit resource ceilings for opt-in assembly. These are operational
/// limits, not fitted mathematical constants. The implementation additionally
/// enforces an 8193-dimensional ceiling and an 8 GiB arithmetic workspace budget.
#[derive(Clone, Copy, Debug)]
pub struct ResearchAssemblyOptions {
    pub prime_route: PrimeAssemblyRoute,
    pub quadrature_order_bucket: usize,
    pub maximum_dimension: usize,
    pub maximum_prime_cutoff: u64,
}

impl Default for ResearchAssemblyOptions {
    fn default() -> Self {
        Self {
            prime_route: PrimeAssemblyRoute::CanonicalCellSum,
            quadrature_order_bucket: 1,
            maximum_dimension: 8193,
            maximum_prime_cutoff: 10_000_000,
        }
    }
}

impl ResearchAssemblyOptions {
    pub fn validate(&self, cutoff: &ExactCutoff, n_modes: usize) -> Result<usize> {
        let dimension = checked_dimension(n_modes)?;
        if self.quadrature_order_bucket == 0 || self.maximum_dimension == 0 {
            bail!("research order bucket and dimension limit must be positive");
        }
        if dimension > self.maximum_dimension
            || dimension > 8193
            || cutoff.prime_cutoff() > self.maximum_prime_cutoff
        {
            bail!("research assembly exceeds the explicit resource ceilings");
        }
        Ok(dimension)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchAssemblyIdentity {
    pub semantics: String,
    pub exact_cutoff: String,
    pub prime_cutoff: u64,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub prime_route: PrimeAssemblyRoute,
    pub quadrature_orders: Vec<usize>,
    pub assurance: String,
}

#[derive(Clone, Debug)]
pub struct ResearchMatrixHp {
    pub identity: ResearchAssemblyIdentity,
    pub entries: Vec<Float>,
}

impl ResearchMatrixHp {
    /// Bind a locally retained experiment to both its route and actual values.
    /// This digest is deliberately not a legacy managed Tau semantic key.
    pub fn content_digest(&self) -> Result<ContentDigest> {
        let values = self
            .entries
            .iter()
            .map(|v| v.to_string_radix(10, None))
            .collect::<Vec<_>>();
        Ok(ContentDigest::sha256(&serde_json::to_vec(&(
            &self.identity,
            values,
        ))?))
    }
}

fn require_precision(precision_bits: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&precision_bits) {
        bail!("research precision must be between 64 and 1000000 bits");
    }
    Ok(())
}

fn checked_dimension(n_modes: usize) -> Result<usize> {
    if n_modes > i64::MAX as usize / 4 {
        bail!("mode indices overflow signed arithmetic");
    }
    let dimension = n_modes
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow!("matrix dimension overflow"))?;
    dimension
        .checked_mul(dimension)
        .ok_or_else(|| anyhow!("matrix storage overflow"))?;
    Ok(dimension)
}

/// The ordinary order policy is bucket=1. Upward bucketing changes numerical
/// quadrature and must remain recorded in the research assembly identity.
pub fn quadrature_orders(
    n_modes: usize,
    base: usize,
    precision_bits: u32,
    bucket: usize,
) -> Result<Vec<usize>> {
    require_precision(precision_bits)?;
    if base == 0 || bucket == 0 {
        bail!("quadrature base and bucket must be positive");
    }
    checked_dimension(n_modes)?;
    (0..=n_modes)
        .map(|n| {
            let order = n
                .checked_mul(3)
                .and_then(|v| v.checked_add((precision_bits / 2) as usize))
                .ok_or_else(|| anyhow!("quadrature order overflow"))?
                .max(base);
            order
                .div_ceil(bucket)
                .checked_mul(bucket)
                .ok_or_else(|| anyhow!("bucketed order overflow"))
        })
        .collect()
}

/// Plan length-aware archimedean Gauss-Legendre orders, rounded to a bucket.
/// The nearest true pole maps to `-1 + 2*pi*i/L`. Its Bernstein ellipse
/// predicts geometric convergence; 64 guard bits and the `3*mode` term supply
/// a practical margin. This order policy is a heuristic, not an assembly-error
/// certificate. `base` remains an explicit lower bound.
pub fn quadrature_orders_for_length(
    n_modes: usize,
    base: usize,
    precision_bits: u32,
    bucket: usize,
    length: &Float,
) -> Result<Vec<usize>> {
    require_precision(precision_bits)?;
    checked_dimension(n_modes)?;
    let l = length.to_f64();
    if base == 0 || bucket == 0 || !l.is_finite() || l <= 0.0 {
        bail!("quadrature base, bucket, and finite length must be positive");
    }
    let y = 2.0 * std::f64::consts::PI / l;
    // For ellipse semimajor axis a=(y+sqrt(4+y*y))/2, avoid
    // cancellation in a-1 and acosh(a) when the pole is close to -1.
    let a_minus_one = (y + y * (y / (y.hypot(2.0) + 2.0))) / 2.0;
    let log_rho = 2.0 * (a_minus_one / 2.0).sqrt().asinh();
    let floor =
        ((f64::from(precision_bits) + 64.0) / (2.0 * log_rho / std::f64::consts::LN_2)).ceil();
    if !floor.is_finite() || !(1.0..=1_000_000.0).contains(&floor) {
        bail!("length-aware archimedean order exceeds the quadrature budget");
    }
    let oscillation = oscillation_order_table(precision_bits, length)?;
    (0..=n_modes)
        .map(|n| {
            let order = n
                .checked_mul(3)
                .and_then(|v| v.checked_add(floor as usize))
                .ok_or_else(|| anyhow!("quadrature order overflow"))?
                .max(oscillation.order(n)?)
                .max(base);
            let order = order
                .div_ceil(bucket)
                .checked_mul(bucket)
                .ok_or_else(|| anyhow!("bucketed order overflow"))?;
            if order > 1_000_000 {
                bail!("length-aware archimedean order exceeds the quadrature budget");
            }
            Ok(order)
        })
        .collect()
}

/// Oscillation-aware order requirement for one length and precision.
///
/// On the Bernstein ellipse with parameter `exp(s)`, mode `n` contributes the
/// carrier growth `cosh(pi*n*sinh(s)) <= exp(pi*n*sinh(s))`, so `m` nodes give
/// about `exp(pi*n*sinh(s) - 2*m*s)`. Any admissible `s <= s_pole` yields a
/// requirement; the smallest sampled one is used. The fixed `3n` allowance
/// covers this only when the pole ellipse is thin (larger cutoffs). Ellipse
/// geometry and `sinh` use correctly rounded MPFR, rounded once to binary64,
/// so the order is identical on every platform.
struct OscillationOrders {
    need: f64,
    samples: Vec<(f64, f64)>,
}

const OSCILLATION_SAMPLES: u32 = 400;

impl OscillationOrders {
    fn order(&self, n: usize) -> Result<usize> {
        if n == 0 {
            return Ok(0);
        }
        let pi_n = std::f64::consts::PI * n as f64;
        let required = self
            .samples
            .iter()
            .map(|(s, sinh)| (self.need + pi_n * sinh) / (2.0 * s))
            .fold(f64::INFINITY, f64::min)
            .ceil();
        if !required.is_finite() || required > 1_000_000.0 {
            bail!("length-aware archimedean order exceeds the quadrature budget");
        }
        Ok(required as usize)
    }
}

fn oscillation_order_table(precision_bits: u32, length: &Float) -> Result<OscillationOrders> {
    const BITS: u32 = 128;
    let two_pi = Float::with_val(BITS, Constant::Pi) * 2u32;
    let y = two_pi / Float::with_val(BITS, length);
    // a - 1 = (y + y^2/(sqrt(4+y^2)+2))/2 for the ellipse through -1+iy.
    let y_squared = Float::with_val(BITS, &y * &y);
    let root = Float::with_val(BITS, &y_squared + 4u32).sqrt() + 2u32;
    let a_minus_one = (y_squared / root + &y) / 2u32;
    let s_pole = (a_minus_one / 2u32).sqrt().asinh() * 2u32;
    let need = Float::with_val(BITS, Constant::Log2) * (precision_bits + 64);
    if !s_pole.is_finite() || s_pole <= 0 || !need.is_finite() {
        bail!("length-aware archimedean order requires a finite positive ellipse");
    }
    // Very wide ellipses (cutoffs near one) can overflow sinh in binary64;
    // such samples are never the minimum, so they are omitted.
    let samples = (1..=OSCILLATION_SAMPLES)
        .map(|k| {
            let s = Float::with_val(BITS, &s_pole * k) / OSCILLATION_SAMPLES;
            let sinh = Float::with_val(BITS, s.sinh_ref());
            (s.to_f64(), sinh.to_f64())
        })
        .filter(|(s, sinh)| s.is_finite() && *s > 0.0 && sinh.is_finite())
        .collect::<Vec<_>>();
    if samples.is_empty() {
        bail!("length-aware archimedean order ellipse is not representable");
    }
    Ok(OscillationOrders {
        need: need.to_f64(),
        samples,
    })
}

/// O(K*d+d^2) arithmetic and O(d) generator storage, plus the output matrix.
/// Entries are correctly rounded from directed bounds at the exact rational
/// cutoff. Unresolved rounding or resource/range exhaustion returns an error.
/// The returned matrix is still a point approximation; it does not certify the
/// eigenvalue or include error from other matrix components.
pub fn aggregate_prime_component_hp(
    cutoff: &ExactCutoff,
    n_modes: usize,
    precision_bits: u32,
    options: &ResearchAssemblyOptions,
) -> Result<Vec<Float>> {
    prime_math::evaluate(cutoff, n_modes, precision_bits, options)
}

fn validate_symmetric_matrix(
    matrix: &[Float],
    dimension: usize,
    precision_bits: u32,
) -> Result<()> {
    require_precision(precision_bits)?;
    if dimension == 0 || dimension > 257 || dimension.checked_mul(dimension) != Some(matrix.len()) {
        bail!("expected a nonempty square matrix");
    }
    if matrix
        .iter()
        .any(|v| !v.is_finite() || v.prec() < precision_bits)
    {
        bail!("matrix entries must be finite at the requested precision");
    }
    for i in 0..dimension {
        for j in 0..i {
            if matrix[i * dimension + j] != matrix[j * dimension + i] {
                bail!("exact symmetric storage is required; no silent symmetrization");
            }
        }
    }
    Ok(())
}

fn matrix_digest(matrix: &[Float], dimension: usize, p: u32) -> Result<ContentDigest> {
    let values = matrix
        .iter()
        .map(|v| v.to_string_radix(10, None))
        .collect::<Vec<_>>();
    Ok(ContentDigest::sha256(&serde_json::to_vec(&(
        dimension, p, values,
    ))?))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedSchurReport {
    pub schema_version: u32,
    pub arithmetic: String,
    pub smaller_dimension: usize,
    pub precision_bits: u32,
    pub smaller_matrix_digest: ContentDigest,
    pub larger_matrix_digest: ContentDigest,
    pub prefix_maximum_absolute_defect: String,
    pub prefix_within_tolerance: bool,
    pub requested_prefix_tolerance: String,
    pub shift: String,
    pub border_norm: String,
    pub schur_complement: String,
    pub solve_relative_residual: String,
    pub schur_uses_actual_larger_prefix: bool,
    pub assurance: String,
    /// In the generalized case: small A, small G, large A, large G.
    pub generalized_input_digests: Option<[ContentDigest; 4]>,
    /// Independent defects of A and G; the original defect field measures A-zG.
    #[serde(default)]
    pub generalized_prefix_defects: Option<[String; 2]>,
}

/// Exact stored-point Schur diagnostic for a prefix of 1..256 directions.
/// Rational workspace is limited to 64 Mbit; exhaustion is an explicit error.
/// One added direction in a shared ORTHONORMAL basis. For a nonorthonormal
/// basis, use `analyze_nested_gram_schur_hp`, which forms A-zG explicitly.
/// A measured nesting defect is retained; the Schur calculation always uses
/// the actual larger matrix's prefix, never substitutes a nearby smaller one.
pub fn analyze_nested_schur_hp(
    smaller: &[Float],
    larger: &[Float],
    smaller_dimension: usize,
    shift: &Float,
    prefix_tolerance: &Float,
    precision_bits: u32,
) -> Result<NestedSchurReport> {
    validate_symmetric_matrix(smaller, smaller_dimension, precision_bits)?;
    let bigger = smaller_dimension
        .checked_add(1)
        .ok_or_else(|| anyhow!("dimension overflow"))?;
    validate_symmetric_matrix(larger, bigger, precision_bits)?;
    let small = exact::points(smaller, precision_bits, 257 * 257)?;
    let large = exact::points(larger, precision_bits, 257 * 257)?;
    let shift_q = exact::point(shift, precision_bits)?;
    let values = exact::schur(&small, &large, smaller_dimension, &shift_q)?;
    schur_report(
        values,
        smaller_dimension,
        shift,
        prefix_tolerance,
        precision_bits,
        matrix_digest(smaller, smaller_dimension, precision_bits)?,
        matrix_digest(larger, bigger, precision_bits)?,
    )
}
fn schur_report(
    values: exact::Schur,
    n: usize,
    shift: &Float,
    tolerance: &Float,
    p: u32,
    small_digest: ContentDigest,
    large_digest: ContentDigest,
) -> Result<NestedSchurReport> {
    let tolerance_q = exact::point(tolerance, p)?;
    if tolerance_q < 0 {
        bail!("Schur prefix tolerance must be nonnegative");
    }
    Ok(NestedSchurReport {
        schema_version: 3,
        arithmetic: FINITE_DIAGNOSTIC_ARITHMETIC.into(),
        smaller_dimension: n,
        precision_bits: p,
        smaller_matrix_digest: small_digest,
        larger_matrix_digest: large_digest,
        prefix_within_tolerance: values.defect <= tolerance_q,
        prefix_maximum_absolute_defect: exact::output(&values.defect, p)?.to_string_radix(10, None),
        requested_prefix_tolerance: tolerance.to_string_radix(10, None),
        shift: shift.to_string_radix(10, None),
        border_norm: exact::sqrt(&values.border_squared, p)?.to_string_radix(10, None),
        schur_complement: exact::output(&values.value, p)?.to_string_radix(10, None),
        solve_relative_residual: "0".into(),
        schur_uses_actual_larger_prefix: true,
        assurance: "exact_stored_point_schur_correctly_rounded_not_a_positivity_certificate".into(),
        generalized_input_digests: None,
        generalized_prefix_defects: None,
    })
}

/// Generalized A-zG variant. G is checked for symmetric storage, not asserted
/// positive definite. The caller must establish its Gram interpretation.
#[allow(clippy::too_many_arguments)]
pub fn analyze_nested_gram_schur_hp(
    small_a: &[Float],
    small_g: &[Float],
    large_a: &[Float],
    large_g: &[Float],
    smaller_dimension: usize,
    shift: &Float,
    prefix_tolerance: &Float,
    p: u32,
) -> Result<NestedSchurReport> {
    validate_symmetric_matrix(small_a, smaller_dimension, p)?;
    validate_symmetric_matrix(small_g, smaller_dimension, p)?;
    let bigger = smaller_dimension
        .checked_add(1)
        .ok_or_else(|| anyhow!("dimension overflow"))?;
    validate_symmetric_matrix(large_a, bigger, p)?;
    validate_symmetric_matrix(large_g, bigger, p)?;
    let shift_q = exact::point(shift, p)?;
    let small = exact::pencil(
        &exact::points(small_a, p, 257 * 257)?,
        &exact::points(small_g, p, 257 * 257)?,
        &shift_q,
    )?;
    let large = exact::pencil(
        &exact::points(large_a, p, 257 * 257)?,
        &exact::points(large_g, p, 257 * 257)?,
        &shift_q,
    )?;
    let values = exact::schur(&small, &large, smaller_dimension, &Rational::from(0))?;
    // Digest the exact pencils as numerator/denominator pairs; rounding them
    // before the Schur solve would erase low source bits after cancellation.
    let pencil_digest = |q: &[Rational]| -> Result<ContentDigest> {
        Ok(ContentDigest::sha256(&serde_json::to_vec(&(
            FINITE_DIAGNOSTIC_ARITHMETIC,
            p,
            q.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ))?))
    };
    let mut report = schur_report(
        values,
        smaller_dimension,
        shift,
        prefix_tolerance,
        p,
        pencil_digest(&small)?,
        pencil_digest(&large)?,
    )?;
    let prefix_defect = |small: &[Float], large: &[Float]| -> Result<Rational> {
        let small = exact::points(small, p, 257 * 257)?;
        let large = exact::points(large, p, 257 * 257)?;
        let mut defect = Rational::from(0);
        for row in 0..smaller_dimension {
            for column in 0..smaller_dimension {
                defect = defect.max(
                    (large[row * bigger + column].clone()
                        - &small[row * smaller_dimension + column])
                        .abs(),
                );
            }
        }
        Ok(defect)
    };
    let defects = [
        prefix_defect(small_a, large_a)?,
        prefix_defect(small_g, large_g)?,
    ];
    let tolerance = exact::point(prefix_tolerance, p)?;
    report.prefix_within_tolerance = defects.iter().all(|d| d <= &tolerance);
    report.generalized_prefix_defects = Some([
        exact::output(&defects[0], p)?.to_string_radix(10, None),
        exact::output(&defects[1], p)?.to_string_radix(10, None),
    ]);
    report.generalized_input_digests = Some([
        matrix_digest(small_a, smaller_dimension, p)?,
        matrix_digest(small_g, smaller_dimension, p)?,
        matrix_digest(large_a, bigger, p)?,
        matrix_digest(large_g, bigger, p)?,
    ]);
    report.assurance =
        "exact_stored_point_generalized_schur_correctly_rounded_gram_positivity_not_certified"
            .into();
    Ok(report)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootTransferReport {
    pub schema_version: u32,
    pub arithmetic: String,
    pub precision_bits: u32,
    pub source_digest: ContentDigest,
    pub expansion_point: String,
    pub function_value: String,
    pub derivative: String,
    pub nearest_pole_distance: String,
    pub predicted_displacement: String,
    pub predicted_step_crosses_pole: bool,
    pub observed_displacement: Option<String>,
    pub displacement_prediction_error: Option<String>,
    pub supplied_target_relative_residual: Option<String>,
    pub assurance: String,
}

/// Correctly rounded -F_new(r_old)/F_new'(r_old) of the exact stored points.
/// Exact rational comparisons decide pole crossings and supplied displacements.
/// This local Newton prediction is not a root enclosure or a source-error bound.
pub fn analyze_root_transfer_hp(
    new_weights: &[Float],
    poles: &[Float],
    old_root: &Float,
    target_estimate: Option<&Float>,
    p: u32,
) -> Result<RootTransferReport> {
    require_precision(p)?;
    let weights = exact::points(new_weights, p, 8193)?;
    let poles_q = exact::points(poles, p, 8193)?;
    let old = exact::point(old_root, p)?;
    let source = exact::secular(&weights, &poles_q, &old)?;
    if source.derivative == 0 {
        bail!("root-transfer derivative is zero");
    }
    let prediction = -source.value.clone() / &source.derivative;
    let predicted_target = old.clone() + &prediction;
    exact::budget([&prediction, &predicted_target].into_iter())?;
    let crosses = poles_q.iter().any(|pole| {
        (old < *pole && pole <= &predicted_target) || (&predicted_target <= pole && *pole < old)
    });
    let mut observed = None;
    let mut prediction_error = None;
    let mut target_residual = None;
    if let Some(target) = target_estimate {
        let target = exact::point(target, p)?;
        let at_target = exact::secular(&weights, &poles_q, &target)?;
        let residual = if at_target.absolute_sum == 0 {
            Rational::from(0)
        } else {
            at_target.value.abs() / at_target.absolute_sum
        };
        target_residual = Some(exact::output(&residual, p)?.to_string_radix(10, None));
        let displacement = target - &old;
        observed = Some(exact::output(&displacement, p)?.to_string_radix(10, None));
        prediction_error =
            Some(exact::output(&(displacement - &prediction), p)?.to_string_radix(10, None));
    }
    let encoded_weights = new_weights
        .iter()
        .map(|v| v.to_string_radix(10, None))
        .collect::<Vec<_>>();
    let encoded_poles = poles
        .iter()
        .map(|v| v.to_string_radix(10, None))
        .collect::<Vec<_>>();
    Ok(RootTransferReport {
        schema_version: 2,
        arithmetic: FINITE_DIAGNOSTIC_ARITHMETIC.into(),
        precision_bits: p,
        source_digest: ContentDigest::sha256(&serde_json::to_vec(&(
            p,
            encoded_weights,
            encoded_poles,
        ))?),
        expansion_point: old_root.to_string_radix(10, None),
        function_value: exact::output(&source.value, p)?.to_string_radix(10, None),
        derivative: exact::output(&source.derivative, p)?.to_string_radix(10, None),
        nearest_pole_distance: exact::output(&source.nearest, p)?.to_string_radix(10, None),
        predicted_displacement: exact::output(&prediction, p)?.to_string_radix(10, None),
        predicted_step_crosses_pole: crosses,
        observed_displacement: observed,
        displacement_prediction_error: prediction_error,
        supplied_target_relative_residual: target_residual,
        assurance: "exact_stored_point_linearization_correctly_rounded_not_a_root_enclosure".into(),
    })
}

/// A panel SUPREMUM premise, not a set of sampled profile values.
#[derive(Clone, Debug)]
pub struct StripErrorPanel {
    pub left: Rational,
    pub right: Rational,
    pub supremum_absolute_error: Rational,
}

/// Conditional estimate for F(z)=integral f(x) exp(i*x*z) dx on |Im z|<=height.
/// Panel endpoints and widths use the integration coordinate x (x=log(u)
/// in a log-coordinate model); no additional measure or density is supplied.
/// The tail premise bounds integral |f-g| exp(height*|x|) dx off the panels.
/// The exact input premises
/// must bound |f-g| on EACH ENTIRE panel and the weighted tail outside their
/// union. This function rigorously encloses the algebraic bound, but it cannot
/// certify those external functional premises from point samples. An
/// unrepresentable finite enclosure returns an error, not an invalid interval.
pub fn conditional_transform_strip_bound(
    panels: &[StripErrorPanel],
    height: &Rational,
    weighted_external_tail: &Rational,
    precision_bits: u32,
) -> Result<MpfrInterval> {
    require_precision(precision_bits)?;
    if panels.is_empty() || height < &0 || weighted_external_tail < &0 {
        bail!("strip bound needs panels and nonnegative height/tail bounds");
    }
    let p = precision_bits;
    let height = MpfrInterval::from_rational(height, p);
    let mut bound = MpfrInterval::from_rational(weighted_external_tail, p);
    for (index, panel) in panels.iter().enumerate() {
        if panel.left >= panel.right
            || panel.supremum_absolute_error < 0
            || (index > 0 && panels[index - 1].right != panel.left)
        {
            bail!("panels must be a contiguous partition with nonnegative supremum bounds");
        }
        // An identically zero error contributes zero at every finite height;
        // do not manufacture 0 * an overflowing exponential.
        if panel.supremum_absolute_error == 0 {
            continue;
        }
        let extent = panel.left.clone().abs().max(panel.right.clone().abs());
        let mut width = panel.right.clone();
        width -= &panel.left;
        let panel_bound = MpfrInterval::from_rational(&panel.supremum_absolute_error, p)
            .mul(&MpfrInterval::from_rational(&width, p))
            .mul(&height.mul(&MpfrInterval::from_rational(&extent, p)).exp());
        bound = bound.add(&panel_bound);
    }
    // Nonfallible interval arithmetic propagates an invalid sentinel on range
    // failure. A successful public bound must be a valid finite enclosure.
    bound.validate()?;
    Ok(bound)
}

/// Actual Linux process high-water RSS, not an allocation-count estimate.
/// Returns None on unsupported platforms or when procfs is unavailable.
pub fn peak_resident_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
        let mut fields = line.split_whitespace();
        fields.next()?;
        let kib = fields.next()?.parse::<u64>().ok()?;
        if fields.next()? != "kB" {
            return None;
        }
        kib.checked_mul(1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hp(value: i32) -> Float {
        Float::with_val(192, value)
    }

    #[test]
    fn exact_cutoff_preserves_prime_edge_and_canonical_identity() {
        let below = ExactCutoff::parse("12.9999999999999999999999999999999999999999").unwrap();
        let above = ExactCutoff::parse("13.0000000000000000000000000000000000000001").unwrap();
        assert_eq!(below.prime_cutoff(), 12);
        assert_eq!(above.prime_cutoff(), 13);
        assert_eq!(
            ExactCutoff::parse("13.000").unwrap(),
            ExactCutoff::parse("26/2").unwrap()
        );
        assert_eq!(ExactCutoff::parse("1.3e1").unwrap().canonical(), "13/1");
        for bad in [
            "", "NaN", "inf", "1", "-2", "13/0", "13/-2", "2e99999", "2.1.0", "2e1e2",
        ] {
            assert!(ExactCutoff::parse(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn bucket_one_preserves_original_order_formula() {
        let orders = quadrature_orders(8, 100, 192, 1).unwrap();
        assert_eq!(
            orders,
            (0..=8).map(|n| 100.max(3 * n + 96)).collect::<Vec<_>>()
        );
        let bucketed = quadrature_orders(8, 100, 192, 32).unwrap();
        assert!(bucketed
            .iter()
            .zip(orders)
            .all(|(a, b)| *a >= b && *a % 32 == 0));
        assert!(quadrature_orders(8, 100, 192, 0).is_err());
    }

    #[test]
    fn structured_prime_matrix_has_exact_storage_symmetry() {
        let cutoff = ExactCutoff::parse("13").unwrap();
        let options = ResearchAssemblyOptions::default();
        let matrix = aggregate_prime_component_hp(&cutoff, 3, 192, &options).unwrap();
        validate_symmetric_matrix(&matrix, 7, 192).unwrap();
        for i in 0..7 {
            for j in 0..7 {
                assert_eq!(matrix[i * 7 + j], matrix[(6 - i) * 7 + (6 - j)]);
            }
        }
    }

    #[test]
    fn nested_schur_checks_actual_prefix_and_known_complement() {
        let small = vec![hp(2)];
        let large = vec![hp(2), hp(1), hp(1), hp(3)];
        let report = analyze_nested_schur_hp(&small, &large, 1, &hp(0), &hp(0), 192).unwrap();
        assert!(report.prefix_within_tolerance);
        let complement = Float::with_val(192, Float::parse(&report.schur_complement).unwrap());
        assert_eq!(complement, Float::with_val(192, Rational::from((5, 2))));
        let different = vec![hp(4)];
        let report = analyze_nested_schur_hp(&different, &large, 1, &hp(0), &hp(0), 192).unwrap();
        assert!(!report.prefix_within_tolerance);
        assert_eq!(
            Float::with_val(192, Float::parse(&report.schur_complement).unwrap()),
            complement
        );
        assert!(analyze_nested_schur_hp(&small, &large, 1, &hp(2), &hp(0), 192).is_err());
    }

    #[test]
    fn generalized_schur_respects_gram_not_identity() {
        let a = vec![hp(4)];
        let g = vec![hp(2)];
        let large_a = vec![hp(4), hp(0), hp(0), hp(9)];
        let large_g = vec![hp(2), hp(0), hp(0), hp(3)];
        let report =
            analyze_nested_gram_schur_hp(&a, &g, &large_a, &large_g, 1, &hp(1), &hp(0), 192)
                .unwrap();
        assert_eq!(
            Float::with_val(192, Float::parse(&report.schur_complement).unwrap()),
            6
        );
    }

    #[test]
    fn root_transfer_records_prediction_and_target_residual() {
        let weights = vec![hp(1), hp(1)];
        let poles = vec![hp(-1), hp(1)];
        let old = Float::with_val(192, Rational::from((1, 10)));
        let report = analyze_root_transfer_hp(&weights, &poles, &old, Some(&hp(0)), 192).unwrap();
        assert!(!report.predicted_step_crosses_pole);
        assert_eq!(
            Float::with_val(
                192,
                Float::parse(report.supplied_target_relative_residual.as_ref().unwrap()).unwrap()
            ),
            0
        );
        assert!(analyze_root_transfer_hp(&weights, &poles, &hp(1), None, 192).is_err());
        assert!(analyze_root_transfer_hp(&[hp(1), hp(-1)], &poles, &hp(0), None, 192).is_err());
    }

    #[test]
    fn conditional_strip_bound_requires_full_panel_coverage() {
        let panels = vec![StripErrorPanel {
            left: Rational::from(-1),
            right: Rational::from(1),
            supremum_absolute_error: Rational::from((1, 8)),
        }];
        let bound = conditional_transform_strip_bound(
            &panels,
            &Rational::from(0),
            &Rational::from((1, 4)),
            192,
        )
        .unwrap();
        assert_eq!(bound.lower(), &Float::with_val(192, Rational::from((1, 2))));
        assert_eq!(bound.upper(), bound.lower());
        let mut broken = panels.clone();
        broken.push(StripErrorPanel {
            left: Rational::from(2),
            right: Rational::from(3),
            supremum_absolute_error: Rational::from(0),
        });
        assert!(conditional_transform_strip_bound(
            &broken,
            &Rational::from(0),
            &Rational::from(0),
            192
        )
        .is_err());
    }
}

#[cfg(test)]
mod strip_range_contract_tests {
    use super::*;
    #[test]
    fn unrepresentable_positive_strip_bound_is_an_error() {
        let panel = StripErrorPanel {
            left: Rational::from(-1),
            right: Rational::from(1),
            supremum_absolute_error: Rational::from(1),
        };
        let height = Rational::from(Integer::from(1) << 64u32);
        assert!(
            conditional_transform_strip_bound(&[panel], &height, &Rational::from(0), 64,).is_err()
        );
    }
    #[test]
    fn zero_error_panel_does_not_evaluate_an_unneeded_exponential() {
        let panel = StripErrorPanel {
            left: Rational::from(-1),
            right: Rational::from(1),
            supremum_absolute_error: Rational::from(0),
        };
        let height = Rational::from(Integer::from(1) << 64u32);
        let bound =
            conditional_transform_strip_bound(&[panel], &height, &Rational::from(3), 64).unwrap();
        bound.validate().unwrap();
        assert_eq!(bound.lower(), &Float::with_val(64, 3));
        assert_eq!(bound.upper(), bound.lower());
    }
}

#[cfg(test)]
mod exhaustive_research_log {
    use super::*;
    fn directed_reference(value: &Rational, p: u32) -> Float {
        use rug::float::Round;
        let mut lo = Float::with_val_round(4096, value, Round::Down).0;
        let mut hi = Float::with_val_round(4096, value, Round::Up).0;
        lo.ln_round(Round::Down);
        hi.ln_round(Round::Up);
        let lower = Float::with_val(p, lo);
        let upper = Float::with_val(p, hi);
        assert_eq!(
            lower, upper,
            "reference interval must determine the rounding"
        );
        lower
    }
    #[test]
    fn exhaustive_research_log_preserves_cutoffs_near_one() {
        let denominator = Integer::from(1) << 200;
        let exact = Rational::from((Integer::from(&denominator + 1), denominator));
        let cutoff = ExactCutoff::from_rational(exact).unwrap();
        let actual = cutoff.log_length(64);
        assert!(
            actual.is_ok(),
            "resolvable nonzero log was lost by rounding the cutoff first: {actual:?}"
        );
        let expected = directed_reference(cutoff.value(), 64);
        assert_eq!(actual.unwrap(), expected);
    }
    #[test]
    fn exhaustive_research_log_rounds_the_exact_rational_log_once() {
        for p in [64, 128, 256] {
            for denominator in 3..=17 {
                for numerator in denominator + 1..=denominator + 20 {
                    let cutoff =
                        ExactCutoff::from_rational(Rational::from((numerator, denominator)))
                            .unwrap();
                    let reference = directed_reference(cutoff.value(), p);
                    assert_eq!(
                        cutoff.log_length(p).unwrap(),
                        reference,
                        "p={p}, cutoff={numerator}/{denominator}"
                    );
                }
            }
        }
    }
    #[test]
    fn exhaustive_research_log_precision_ceiling_precedes_allocation() {
        assert!(quadrature_orders(0, 1, 1_000_001, 1).is_err());
    }
    #[test]
    fn exhaustive_research_log_rational_input_has_a_bit_budget() {
        let denominator = Integer::from(1) << 1_048_577;
        let value = Rational::from((Integer::from(&denominator + 1), denominator));
        assert!(ExactCutoff::from_rational(value).is_err());
    }
    #[test]
    fn exhaustive_research_log_near_one_at_the_rational_budget_and_precision_domain() {
        for exponent in [200u32, 4096, 1_048_575] {
            let denominator = Integer::from(1) << exponent;
            let cutoff = ExactCutoff::from_rational(Rational::from((
                Integer::from(&denominator + 1),
                denominator,
            )))
            .unwrap();
            // delta-delta^2/2 < log(1+delta) < delta; at p=64 the
            // entire interval rounds to delta for all these exponents.
            assert_eq!(
                cutoff.log_length(64).unwrap(),
                Float::with_val(64, 1) >> exponent
            );
            assert_eq!(cutoff.prime_cutoff(), 1);
        }
        let cutoff = ExactCutoff::parse("2").unwrap();
        for p in [0, 1, 63, 1_000_001, u32::MAX] {
            assert!(cutoff.log_length(p).is_err());
        }
        assert!(require_precision(64).is_ok());
        assert!(require_precision(1_000_000).is_ok());
        assert_eq!(
            cutoff.log_length(128).unwrap(),
            Float::with_val(128, 2).ln()
        );
    }
    #[test]
    fn exhaustive_research_log_exact_representations_share_correct_rounding() {
        for group in [
            ["13.1", "131/10", "1.31e1"],
            ["2", "2/1", "2.0000"],
            ["1.000001", "1000001/1000000", "1000001e-6"],
        ] {
            let base = ExactCutoff::parse(group[0]).unwrap();
            for literal in group {
                let value = ExactCutoff::parse(literal).unwrap();
                assert_eq!(base, value);
                for p in [64, 128, 256] {
                    assert_eq!(
                        value.log_length(p).unwrap(),
                        directed_reference(value.value(), p)
                    );
                }
            }
        }
        assert_eq!(
            RESEARCH_ASSEMBLY_SEMANTICS,
            "ccm-exact-input-research-assembly-length-aware-arch-v2"
        );
        assert_eq!(
            AGGREGATE_PRIME_SEMANTICS,
            "ccm-prime-divided-difference-generators-v0.15.2-v1"
        );
    }
}

#[cfg(test)]
mod exhaustive_aggregate_prime {
    use super::*;
    fn independent(c: &ExactCutoff, n: usize, p: u32) -> Vec<Float> {
        let w = 2048;
        let l = c.log_length(w).unwrap();
        let pi = Float::with_val(w, Constant::Pi);
        let dim = 2 * n + 1;
        let mut out = vec![Float::with_val(w, 0); dim * dim];
        for (power, prime, _) in prime_powers_up_to(c.prime_cutoff()) {
            if c.value() == &Rational::from(power) {
                continue;
            }
            let ratio = Float::with_val(w, power).ln() / &l;
            let weight = Float::with_val(w, prime).ln() / Float::with_val(w, power).sqrt();
            for i in 0..dim {
                for j in 0..dim {
                    let m = i as i64 - n as i64;
                    let k = j as i64 - n as i64;
                    let x = Float::with_val(w, &pi) * 2 * &ratio;
                    let value = if m == k {
                        (Float::with_val(w, 1) - &ratio) * 2 * (Float::with_val(w, &x) * m).cos()
                    } else {
                        ((Float::with_val(w, &x) * k).sin() - (Float::with_val(w, &x) * m).sin())
                            / (Float::with_val(w, &pi) * (m - k))
                    };
                    out[i * dim + j] += value * &weight;
                }
            }
        }
        out.iter().map(|x| Float::with_val(p, x)).collect()
    }
    #[test]
    fn exhaustive_aggregate_prime_exact_first_edge_is_zero() {
        let c = ExactCutoff::parse("2").unwrap();
        let a =
            aggregate_prime_component_hp(&c, 3, 128, &ResearchAssemblyOptions::default()).unwrap();
        assert!(
            a.iter().all(Float::is_zero),
            "edge event has identically zero matrix contribution"
        );
    }
    #[test]
    fn exhaustive_aggregate_prime_rounds_final_matrix_once() {
        for p in [64, 128, 256] {
            for n in 0..=3 {
                for text in ["3", "7", "13", "17/2"] {
                    let c = ExactCutoff::parse(text).unwrap();
                    let got =
                        aggregate_prime_component_hp(&c, n, p, &ResearchAssemblyOptions::default())
                            .unwrap();
                    let want = independent(&c, n, p);
                    assert_eq!(got, want, "C={text},n={n},p={p}");
                }
            }
        }
    }
    #[test]
    fn exhaustive_aggregate_prime_near_edge_retains_nonzero() {
        let c = ExactCutoff::from_rational(
            Rational::from(2) + Rational::from((Integer::from(1), Integer::from(1) << 400)),
        )
        .unwrap();
        let got =
            aggregate_prime_component_hp(&c, 0, 128, &ResearchAssemblyOptions::default()).unwrap();
        assert_eq!(got, independent(&c, 0, 128));
        assert!(got[0] > 0);
    }
    #[test]
    fn exhaustive_aggregate_prime_resource_validation_is_bounded() {
        let o = ResearchAssemblyOptions {
            maximum_dimension: usize::MAX,
            ..Default::default()
        };
        assert!(o
            .validate(&ExactCutoff::parse("2").unwrap(), 100_000)
            .is_err());
    }
}

#[cfg(test)]
mod research_controls_tests {
    use super::*;
    #[test]
    fn generalized_prefix_checks_both_forms_before_shift_cancellation() {
        let f = |x| Float::with_val(192, x);
        let report = analyze_nested_gram_schur_hp(
            &[f(4)],
            &[f(2)],
            &[f(6), f(1), f(1), f(5)],
            &[f(4), f(0), f(0), f(1)],
            1,
            &f(1),
            &f(0),
            192,
        )
        .unwrap();
        assert!(!report.prefix_within_tolerance);
        let defects = report.generalized_prefix_defects.unwrap();
        for defect in defects {
            assert_eq!(Float::with_val(192, Float::parse(defect).unwrap()), 2);
        }
        assert_eq!(
            Float::with_val(192, Float::parse(report.schur_complement).unwrap()),
            3.5
        );
        assert_eq!(
            Float::with_val(
                192,
                Float::parse(report.prefix_maximum_absolute_defect).unwrap()
            ),
            0
        );
    }
    #[test]
    fn cutoff_rational_grammar_is_ascii_and_has_one_sign() {
        for text in ["+13/+1", "++13/1", "1_3/1", "13/0_1", "13/+1"] {
            assert!(ExactCutoff::parse(text).is_err(), "{text}");
        }
        for text in ["13/1", "+13/1", "13.0", "1.3e1"] {
            assert_eq!(
                ExactCutoff::parse(text).unwrap().value(),
                &Rational::from(13)
            );
        }
    }
    #[test]
    fn point_conversion_admits_declared_schur_and_transfer_shapes() {
        for p in [64, 192, 1024] {
            let values = vec![Float::with_val(p, 1); 257 * 257];
            assert_eq!(
                exact::points(&values, p, 257 * 257).unwrap().len(),
                values.len()
            );
        }
        let values = vec![Float::with_val(192, 1); 8193];
        assert_eq!(exact::points(&values, 192, 8193).unwrap().len(), 8193);
        let huge = Float::with_val(192, 1) << 100_000_000u32;
        assert!(exact::points(&[huge], 192, 1).is_err());
    }
}

#[cfg(test)]
mod length_order_tests {
    use super::*;
    #[test]
    fn length_order_has_independent_ellipse_geometry_and_bucket_floor() {
        // The ellipse through (-1,y) obeys a=(y+sqrt(4+y^2))/2.
        // This direct acosh computation is safe at these ordinary lengths.
        for l in [
            2.5f64,
            13.815510557964274,
            15.201804919084164,
            16.11809565095832,
        ] {
            let length = Float::with_val(128, l);
            for p in [256u32, 512] {
                let y = 2.0 * std::f64::consts::PI / l;
                let a = (y + (4.0 + y * y).sqrt()) / 2.0;
                let floor = ((f64::from(p) + 64.0) / (2.0 * a.acosh() / std::f64::consts::LN_2))
                    .ceil() as usize;
                let orders = quadrature_orders_for_length(3, 1, p, 1, &length).unwrap();
                assert_eq!(orders, (0..=3).map(|n| 3 * n + floor).collect::<Vec<_>>());
                let bucket = quadrature_orders_for_length(3, 1000, p, 32, &length).unwrap();
                assert!(bucket.iter().all(|&m| m >= 1000 && m % 32 == 0));
            }
        }
    }
}
