// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Opt-in CCM research primitives. These do not change claim capture presets,
//! silently populate legacy caches, or promote point calculations to proofs.
//!
//! All numerical work stays in MPFR or exact rationals. The structured prime
//! route deliberately has a distinct identity: regrouping an exact formula
//! need not reproduce the rounding of the canonical cell-by-cell route.

use anyhow::{anyhow, bail, Result};
use rug::{float::Constant, ops::Pow, Float, Integer, Rational};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
use xc_numerics::mpfr_interval::MpfrInterval;

use super::prime_powers_up_to;

pub const RESEARCH_ASSEMBLY_SEMANTICS: &str = "ccm-exact-input-research-assembly-v0.14.4-v1";
pub const AGGREGATE_PRIME_SEMANTICS: &str = "ccm-prime-divided-difference-generators-v0.14.4-v1";

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
            if denominator.contains('/') || numerator.is_empty() || denominator.is_empty() {
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
            if !whole.bytes().chain(fractional.bytes()).all(|b| b.is_ascii_digit()) {
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

    pub fn from_rational(value: Rational) -> Result<Self> {
        if value <= 1 {
            bail!("CCM cutoff must be greater than one");
        }
        let mut floor = value.numer().clone();
        floor /= value.denom();
        let prime_cutoff = floor.to_u64().ok_or_else(|| anyhow!("cutoff floor exceeds u64"))?;
        Ok(Self { value, prime_cutoff })
    }

    pub fn value(&self) -> &Rational { &self.value }
    pub fn prime_cutoff(&self) -> u64 { self.prime_cutoff }
    pub fn canonical(&self) -> String {
        format!("{}/{}", self.value.numer(), self.value.denom())
    }
    pub fn log_length(&self, precision_bits: u32) -> Result<Float> {
        require_precision(precision_bits)?;
        let length = Float::with_val(precision_bits, &self.value).ln();
        if !length.is_finite() || length <= 0 {
            bail!("working precision cannot resolve a positive cutoff log length");
        }
        Ok(length)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimeAssemblyRoute {
    CanonicalCellSum,
    AggregateGenerators,
}

/// Explicit resource ceilings for opt-in assembly. These are operational
/// limits, not fitted mathematical constants, and may be raised explicitly.
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
        if dimension > self.maximum_dimension || cutoff.prime_cutoff() > self.maximum_prime_cutoff {
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
        let values = self.entries.iter().map(|v| v.to_string_radix(10, None)).collect::<Vec<_>>();
        Ok(ContentDigest::sha256(&serde_json::to_vec(&(&self.identity, values))?))
    }
}

fn require_precision(precision_bits: u32) -> Result<()> {
    if !(64..=i32::MAX as u32 - 64).contains(&precision_bits) {
        bail!("research precision must be between 64 and i32::MAX-64 bits");
    }
    Ok(())
}

fn checked_dimension(n_modes: usize) -> Result<usize> {
    if n_modes > i64::MAX as usize / 4 {
        bail!("mode indices overflow signed arithmetic");
    }
    let dimension = n_modes.checked_mul(2).and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow!("matrix dimension overflow"))?;
    dimension.checked_mul(dimension).ok_or_else(|| anyhow!("matrix storage overflow"))?;
    Ok(dimension)
}

/// The ordinary order policy is bucket=1. Upward bucketing changes numerical
/// quadrature and must remain recorded in the research assembly identity.
pub fn quadrature_orders(n_modes: usize, base: usize, precision_bits: u32, bucket: usize) -> Result<Vec<usize>> {
    require_precision(precision_bits)?;
    if base == 0 || bucket == 0 { bail!("quadrature base and bucket must be positive"); }
    checked_dimension(n_modes)?;
    (0..=n_modes).map(|n| {
        let order = n.checked_mul(3).and_then(|v| v.checked_add((precision_bits / 2) as usize))
            .ok_or_else(|| anyhow!("quadrature order overflow"))?.max(base);
        order.div_ceil(bucket).checked_mul(bucket).ok_or_else(|| anyhow!("bucketed order overflow"))
    }).collect()
}

/// O(K*d+d^2) arithmetic and O(d) generator storage, plus the output matrix.
/// This is a POINT implementation, not an interval enclosure. Component-level
/// agreement alone does not certify a deeply cancelled smallest eigenvalue.
pub fn aggregate_prime_component_hp(
    cutoff: &ExactCutoff,
    n_modes: usize,
    precision_bits: u32,
    options: &ResearchAssemblyOptions,
) -> Result<Vec<Float>> {
    let dimension = options.validate(cutoff, n_modes)?;
    let length = cutoff.log_length(precision_bits)?;
    let pi = Float::with_val(precision_bits, Constant::Pi);
    let mut two_pi = pi.clone();
    two_pi *= 2;
    let prime_data = prime_powers_up_to(cutoff.prime_cutoff()).into_iter().map(|(power, prime, _)| {
        let x = Float::with_val(precision_bits, power).ln();
        let mut weight = Float::with_val(precision_bits, prime).ln();
        weight /= Float::with_val(precision_bits, power).sqrt();
        (x, weight)
    }).collect::<Vec<_>>();
    let mut sines = Vec::with_capacity(n_modes + 1);
    let mut diagonal = Vec::with_capacity(n_modes + 1);
    for n in 0..=n_modes {
        let mut sine = Float::with_val(precision_bits, 0);
        let mut diag = Float::with_val(precision_bits, 0);
        for (log_power, weight) in &prime_data {
            let mut ratio = Float::with_val(precision_bits, log_power);
            ratio /= &length;
            let mut phase = two_pi.clone();
            phase *= n as u64;
            phase *= &ratio;
            if n != 0 {
                let mut value = phase.clone().sin();
                value *= weight;
                sine += value;
            }
            let mut value = Float::with_val(precision_bits, 1);
            value -= &ratio;
            value *= 2;
            value *= phase.cos();
            value *= weight;
            diag += value;
        }
        sines.push(sine);
        diagonal.push(diag);
    }
    let signed_sine = |mode: i64| {
        let value = sines[mode.unsigned_abs() as usize].clone();
        if mode < 0 { -value } else { value }
    };
    let mut matrix = vec![Float::with_val(precision_bits, 0); dimension * dimension];
    for row in 0..dimension {
        let n = row as i64 - n_modes as i64;
        for column in row..dimension {
            let m = column as i64 - n_modes as i64;
            let value = if n == m {
                diagonal[n.unsigned_abs() as usize].clone()
            } else {
                let mut value = signed_sine(m);
                value -= signed_sine(n);
                let mut denominator = pi.clone();
                denominator *= n - m;
                value /= denominator;
                value
            };
            matrix[row * dimension + column] = value.clone();
            matrix[column * dimension + row] = value;
        }
    }
    Ok(matrix)
}

fn validate_symmetric_matrix(matrix: &[Float], dimension: usize, precision_bits: u32) -> Result<()> {
    require_precision(precision_bits)?;
    if dimension == 0 || dimension.checked_mul(dimension) != Some(matrix.len()) {
        bail!("expected a nonempty square matrix");
    }
    if matrix.iter().any(|v| !v.is_finite() || v.prec() < precision_bits) {
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

fn dot(left: &[Float], right: &[Float], p: u32) -> Float {
    let terms = left.iter().zip(right).map(|(a, b)| {
        let mut value = Float::with_val(p, a);
        value *= b;
        value
    }).collect::<Vec<_>>();
    xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(terms, p)
}

fn norm(vector: &[Float], p: u32) -> Float { dot(vector, vector, p).sqrt() }

fn matrix_digest(matrix: &[Float], dimension: usize, p: u32) -> Result<ContentDigest> {
    let values = matrix.iter().map(|v| v.to_string_radix(10, None)).collect::<Vec<_>>();
    Ok(ContentDigest::sha256(&serde_json::to_vec(&(dimension, p, values))?))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedSchurReport {
    pub schema_version: u32,
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
}

/// One added direction in a shared ORTHONORMAL basis. For a nonorthonormal
/// basis, use `analyze_nested_gram_schur_hp`, which forms A-zG explicitly.
/// A measured nesting defect is retained; the Schur calculation always uses
/// the actual larger matrix's prefix, never substitutes a nearby smaller one.
pub fn analyze_nested_schur_hp(
    smaller: &[Float], larger: &[Float], smaller_dimension: usize,
    shift: &Float, prefix_tolerance: &Float, precision_bits: u32,
) -> Result<NestedSchurReport> {
    validate_symmetric_matrix(smaller, smaller_dimension, precision_bits)?;
    let bigger = smaller_dimension.checked_add(1).ok_or_else(|| anyhow!("dimension overflow"))?;
    validate_symmetric_matrix(larger, bigger, precision_bits)?;
    if !shift.is_finite() || !prefix_tolerance.is_finite() || prefix_tolerance < &0 {
        bail!("shift and nonnegative prefix tolerance must be finite");
    }
    let n = smaller_dimension;
    let p = precision_bits;
    let mut defect = Float::with_val(p, 0);
    let mut block = Vec::with_capacity(n * n);
    let mut border = Vec::with_capacity(n);
    for i in 0..n {
        border.push(Float::with_val(p, &larger[i * bigger + n]));
        for j in 0..n {
            let mut delta = Float::with_val(p, &larger[i * bigger + j]);
            delta -= &smaller[i * n + j];
            delta.abs_mut();
            if delta > defect { defect = delta; }
            let mut value = Float::with_val(p, &larger[i * bigger + j]);
            if i == j { value -= shift; }
            block.push(value);
        }
    }
    let factors = xc_numerics::linalg::lu_factor(&block, n)?;
    let solution = xc_numerics::linalg::lu_solve(&factors, &border, n, p);
    if solution.iter().any(|v| !v.is_finite()) { bail!("nonfinite Schur solve"); }
    let residual = (0..n).map(|i| {
        let mut value = dot(&block[i*n..(i+1)*n], &solution, p);
        value -= &border[i];
        value
    }).collect::<Vec<_>>();
    let border_norm = norm(&border, p);
    let mut denominator = norm(&block, p);
    denominator *= norm(&solution, p);
    denominator += &border_norm;
    let mut relative = norm(&residual, p);
    if !denominator.is_zero() { relative /= denominator; }
    let mut schur = Float::with_val(p, &larger[n * bigger + n]);
    schur -= shift;
    schur -= dot(&border, &solution, p);
    Ok(NestedSchurReport {
        schema_version: 1, smaller_dimension: n, precision_bits: p,
        smaller_matrix_digest: matrix_digest(smaller, n, p)?,
        larger_matrix_digest: matrix_digest(larger, bigger, p)?,
        prefix_within_tolerance: &defect <= prefix_tolerance,
        prefix_maximum_absolute_defect: defect.to_string_radix(10, None),
        requested_prefix_tolerance: prefix_tolerance.to_string_radix(10, None),
        shift: shift.to_string_radix(10, None),
        border_norm: border_norm.to_string_radix(10, None),
        schur_complement: schur.to_string_radix(10, None),
        solve_relative_residual: relative.to_string_radix(10, None),
        schur_uses_actual_larger_prefix: true,
        assurance: "computed_diagnostic_not_a_positivity_certificate".to_owned(),
    })
}

/// Generalized A-zG variant. G is checked for symmetric storage, not asserted
/// positive definite. The caller must establish its Gram interpretation.
pub fn analyze_nested_gram_schur_hp(
    small_a: &[Float], small_g: &[Float], large_a: &[Float], large_g: &[Float],
    smaller_dimension: usize, shift: &Float, prefix_tolerance: &Float, p: u32,
) -> Result<NestedSchurReport> {
    validate_symmetric_matrix(small_a, smaller_dimension, p)?;
    validate_symmetric_matrix(small_g, smaller_dimension, p)?;
    let bigger = smaller_dimension.checked_add(1).ok_or_else(|| anyhow!("dimension overflow"))?;
    validate_symmetric_matrix(large_a, bigger, p)?;
    validate_symmetric_matrix(large_g, bigger, p)?;
    if !shift.is_finite() { bail!("nonfinite generalized shift"); }
    let pencil = |a: &[Float], g: &[Float]| a.iter().zip(g).map(|(a, g)| {
        let mut value = Float::with_val(p, g);
        value *= shift;
        value = -value;
        value += a;
        value
    }).collect::<Vec<_>>();
    let small = pencil(small_a, small_g);
    let large = pencil(large_a, large_g);
    let zero = Float::with_val(p, 0);
    let mut report = analyze_nested_schur_hp(&small, &large, smaller_dimension, &zero, prefix_tolerance, p)?;
    report.shift = shift.to_string_radix(10, None);
    report.assurance = "computed_generalized_pencil_diagnostic_gram_positivity_not_certified".to_owned();
    Ok(report)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootTransferReport {
    pub schema_version: u32,
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

fn secular_value_derivative(weights: &[Float], poles: &[Float], point: &Float, p: u32)
    -> Result<(Float, Float, Float, Float)>
{
    let mut terms = Vec::with_capacity(weights.len());
    let mut derivatives = Vec::with_capacity(weights.len());
    let mut nearest: Option<Float> = None;
    let mut magnitude_sum = Float::with_val(p, 0);
    for (weight, pole) in weights.iter().zip(poles) {
        let mut denominator = Float::with_val(p, point);
        denominator -= pole;
        if denominator.is_zero() { bail!("secular evaluation encountered a pole"); }
        let distance = denominator.clone().abs();
        if nearest.as_ref().is_none_or(|old| &distance < old) { nearest = Some(distance); }
        let mut term = Float::with_val(p, weight);
        term /= &denominator;
        magnitude_sum += term.clone().abs();
        let mut derivative = -term.clone();
        derivative /= denominator;
        terms.push(term);
        derivatives.push(derivative);
    }
    Ok((
        xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(terms, p),
        xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(derivatives, p),
        nearest.ok_or_else(|| anyhow!("empty secular source"))?, magnitude_sum,
    ))
}

/// Evaluate -F_new(r_old)/F_new'(r_old) and compare with a supplied target
/// estimate. No reference zero is used to produce the prediction. A supplied
/// target is checked by residual but is NOT thereby promoted to a certificate.
pub fn analyze_root_transfer_hp(
    new_weights: &[Float], poles: &[Float], old_root: &Float,
    target_estimate: Option<&Float>, p: u32,
) -> Result<RootTransferReport> {
    require_precision(p)?;
    if new_weights.is_empty() || new_weights.len() != poles.len()
        || new_weights.iter().chain(poles).any(|v| !v.is_finite() || v.prec() < p)
        || !old_root.is_finite() || target_estimate.is_some_and(|v| !v.is_finite())
    { bail!("incompatible or nonfinite root-transfer inputs"); }
    if poles.windows(2).any(|pair| pair[0] >= pair[1]) { bail!("poles must be strictly ordered"); }
    let (value, derivative, nearest, _) = secular_value_derivative(new_weights, poles, old_root, p)?;
    if derivative.is_zero() { bail!("root-transfer derivative is unresolved or zero"); }
    let mut prediction = -value.clone();
    prediction /= &derivative;
    if !prediction.is_finite() { bail!("nonfinite root-transfer prediction"); }
    let mut predicted_target = Float::with_val(p, old_root);
    predicted_target += &prediction;
    let crosses = poles.iter().any(|pole| {
        (old_root < pole && pole <= &predicted_target) || (&predicted_target <= pole && pole < old_root)
    });
    let mut observed = None;
    let mut prediction_error = None;
    let mut target_residual = None;
    if let Some(target) = target_estimate {
        let (mut residual, _, _, scale) = secular_value_derivative(new_weights, poles, target, p)?;
        residual.abs_mut();
        if !scale.is_zero() { residual /= scale; }
        target_residual = Some(residual.to_string_radix(10, None));
        let mut displacement = Float::with_val(p, target);
        displacement -= old_root;
        observed = Some(displacement.to_string_radix(10, None));
        displacement -= &prediction;
        prediction_error = Some(displacement.to_string_radix(10, None));
    }
    let encoded_weights = new_weights.iter().map(|v| v.to_string_radix(10, None)).collect::<Vec<_>>();
    let encoded_poles = poles.iter().map(|v| v.to_string_radix(10, None)).collect::<Vec<_>>();
    Ok(RootTransferReport {
        schema_version: 1, precision_bits: p,
        source_digest: ContentDigest::sha256(&serde_json::to_vec(&(p, encoded_weights, encoded_poles))?),
        expansion_point: old_root.to_string_radix(10, None),
        function_value: value.to_string_radix(10, None),
        derivative: derivative.to_string_radix(10, None),
        nearest_pole_distance: nearest.to_string_radix(10, None),
        predicted_displacement: prediction.to_string_radix(10, None),
        predicted_step_crosses_pole: crosses,
        observed_displacement: observed, displacement_prediction_error: prediction_error,
        supplied_target_relative_residual: target_residual,
        assurance: "computed_local_linearization_not_a_root_enclosure".to_owned(),
    })
}

/// A panel SUPREMUM premise, not a set of sampled profile values.
#[derive(Clone, Debug)]
pub struct StripErrorPanel {
    pub left: Rational,
    pub right: Rational,
    pub supremum_absolute_error: Rational,
}

/// Conditional transform estimate on |Im z|<=height. The exact input premises
/// must bound |f-g| on EACH ENTIRE panel and the weighted tail outside their
/// union. This function rigorously encloses the algebraic bound, but it cannot
/// certify those external functional premises from point samples.
pub fn conditional_transform_strip_bound(
    panels: &[StripErrorPanel], height: &Rational, weighted_external_tail: &Rational,
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
        if panel.left >= panel.right || panel.supremum_absolute_error < 0
            || (index > 0 && panels[index - 1].right != panel.left)
        { bail!("panels must be a contiguous partition with nonnegative supremum bounds"); }
        let extent = panel.left.clone().abs().max(panel.right.clone().abs());
        let mut width = panel.right.clone();
        width -= &panel.left;
        let panel_bound = MpfrInterval::from_rational(&panel.supremum_absolute_error, p)
            .mul(&MpfrInterval::from_rational(&width, p))
            .mul(&height.mul(&MpfrInterval::from_rational(&extent, p)).exp());
        bound = bound.add(&panel_bound);
    }
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
        if fields.next()? != "kB" { return None; }
        kib.checked_mul(1024)
    }
    #[cfg(not(target_os = "linux"))]
    { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hp(value: i32) -> Float { Float::with_val(192, value) }

    #[test]
    fn exact_cutoff_preserves_prime_edge_and_canonical_identity() {
        let below = ExactCutoff::parse("12.9999999999999999999999999999999999999999").unwrap();
        let above = ExactCutoff::parse("13.0000000000000000000000000000000000000001").unwrap();
        assert_eq!(below.prime_cutoff(), 12);
        assert_eq!(above.prime_cutoff(), 13);
        assert_eq!(ExactCutoff::parse("13.000").unwrap(), ExactCutoff::parse("26/2").unwrap());
        assert_eq!(ExactCutoff::parse("1.3e1").unwrap().canonical(), "13/1");
        for bad in ["", "NaN", "inf", "1", "-2", "13/0", "13/-2", "2e99999", "2.1.0", "2e1e2"] {
            assert!(ExactCutoff::parse(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn bucket_one_preserves_original_order_formula() {
        let orders = quadrature_orders(8, 100, 192, 1).unwrap();
        assert_eq!(orders, (0..=8).map(|n| 100.max(3*n+96)).collect::<Vec<_>>());
        let bucketed = quadrature_orders(8, 100, 192, 32).unwrap();
        assert!(bucketed.iter().zip(orders).all(|(a,b)| *a >= b && *a % 32 == 0));
        assert!(quadrature_orders(8, 100, 192, 0).is_err());
    }

    #[test]
    fn structured_prime_matrix_has_exact_storage_symmetry() {
        let cutoff = ExactCutoff::parse("13").unwrap();
        let options = ResearchAssemblyOptions::default();
        let matrix = aggregate_prime_component_hp(&cutoff, 3, 192, &options).unwrap();
        validate_symmetric_matrix(&matrix, 7, 192).unwrap();
        for i in 0..7 { for j in 0..7 {
            assert_eq!(matrix[i*7+j], matrix[(6-i)*7+(6-j)]);
        }}
    }

    #[test]
    fn nested_schur_checks_actual_prefix_and_known_complement() {
        let small = vec![hp(2)];
        let large = vec![hp(2), hp(1), hp(1), hp(3)];
        let report = analyze_nested_schur_hp(&small, &large, 1, &hp(0), &hp(0), 192).unwrap();
        assert!(report.prefix_within_tolerance);
        let complement = Float::with_val(192, Float::parse(&report.schur_complement).unwrap());
        assert_eq!(complement, Float::with_val(192, Rational::from((5,2))));
        let different = vec![hp(4)];
        let report = analyze_nested_schur_hp(&different, &large, 1, &hp(0), &hp(0), 192).unwrap();
        assert!(!report.prefix_within_tolerance);
        assert_eq!(Float::with_val(192, Float::parse(&report.schur_complement).unwrap()), complement);
        assert!(analyze_nested_schur_hp(&small, &large, 1, &hp(2), &hp(0), 192).is_err());
    }

    #[test]
    fn generalized_schur_respects_gram_not_identity() {
        let a = vec![hp(4)];
        let g = vec![hp(2)];
        let large_a = vec![hp(4),hp(0),hp(0),hp(9)];
        let large_g = vec![hp(2),hp(0),hp(0),hp(3)];
        let report = analyze_nested_gram_schur_hp(&a,&g,&large_a,&large_g,1,&hp(1),&hp(0),192).unwrap();
        assert_eq!(Float::with_val(192, Float::parse(&report.schur_complement).unwrap()), 6);
    }

    #[test]
    fn root_transfer_records_prediction_and_target_residual() {
        let weights = vec![hp(1), hp(1)];
        let poles = vec![hp(-1), hp(1)];
        let old = Float::with_val(192, Rational::from((1,10)));
        let report = analyze_root_transfer_hp(&weights, &poles, &old, Some(&hp(0)), 192).unwrap();
        assert!(!report.predicted_step_crosses_pole);
        assert_eq!(Float::with_val(192, Float::parse(report.supplied_target_relative_residual.as_ref().unwrap()).unwrap()), 0);
        assert!(analyze_root_transfer_hp(&weights, &poles, &hp(1), None, 192).is_err());
        assert!(analyze_root_transfer_hp(&[hp(1),hp(-1)], &poles, &hp(0), None, 192).is_err());
    }

    #[test]
    fn conditional_strip_bound_requires_full_panel_coverage() {
        let panels = vec![StripErrorPanel { left: Rational::from(-1), right: Rational::from(1), supremum_absolute_error: Rational::from((1,8)) }];
        let bound = conditional_transform_strip_bound(&panels, &Rational::from(0), &Rational::from((1,4)), 192).unwrap();
        assert_eq!(bound.lower(), &Float::with_val(192, Rational::from((1,2))));
        assert_eq!(bound.upper(), bound.lower());
        let mut broken = panels.clone();
        broken.push(StripErrorPanel { left: Rational::from(2), right: Rational::from(3), supremum_absolute_error: Rational::from(0) });
        assert!(conditional_transform_strip_bound(&broken, &Rational::from(0), &Rational::from(0), 192).is_err());
    }
}
