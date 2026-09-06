// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Fixed-order positive-definite prefix diagnostics and checked decimal exports.
//! These analyze a supplied point matrix, never regularize it, and are not
//! certificates of matrix assembly, positivity, or eigenvalue error.
//!
//! Generated implementation assistance for the owner-authorized CCM extension
//! implementation. No external algorithm implementation is copied. The block
//! inverse identities are documented in `docs/CCM_PREFIX_ANALYSIS.md`.
//! Existing Toolkit deterministic reduction and decimal-width primitives are reused.

use crate::reduction::{deterministic_pairwise_sum_hp_owned, roundtrip_decimal_digits};
use anyhow::{bail, Result};
use rug::{ops::Pow, Float};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn sum(values: Vec<Float>, p: u32) -> Float {
    deterministic_pairwise_sum_hp_owned(values, p)
}

fn dot(a: &[Float], b: &[Float], p: u32) -> Float {
    assert_eq!(a.len(), b.len());
    sum(
        a.iter()
            .zip(b)
            .map(|(a, b)| {
                let mut v = Float::with_val(p, a);
                v *= b;
                v
            })
            .collect(),
        p,
    )
}

pub fn lossless_decimal(v: &Float) -> String {
    v.to_string_radix(10, Some(roundtrip_decimal_digits(v.prec())))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixRow {
    pub dimension: usize,
    pub sigma: String,
    pub innovation_mass: String,
    pub inverse_trace_increment: String,
    pub inverse_trace: String,
    pub inverse_square_trace: String,
    pub pivot_cancellation_scale: String,
    pub effective_inverse_rank: String,
    pub newest_inverse_trace_fraction: String,
    pub smallest_eigenvalue_lower_estimate: String,
    pub smallest_eigenvalue_upper_estimate: String,
    pub eigenvalue_depth_lower_estimate: String,
    pub eigenvalue_depth_upper_estimate: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixStop {
    pub attempted_dimension: usize,
    pub reason: String,
    pub pivot: String,
    pub scale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixAnalysisReport {
    pub semantics: String,
    pub precision_bits: u32,
    pub pivot_margin_bits: u32,
    pub source_precision_bits: Vec<u32>,
    pub requested_dimension: usize,
    pub rows: Vec<PrefixRow>,
    /// Each innovation has last coefficient +1, not unit norm.
    pub checkpoint_innovations: BTreeMap<usize, Vec<String>>,
    pub stopped: Option<PrefixStop>,
    pub assurance: String,
}

/// Computes the ladder of the supplied symmetric point matrix, in its exact
/// supplied order, without pivoting, regularization, or source recomputation.
/// The pivot margin is a declared numerical screening policy, NOT a proof
/// error bound. A positive computed pivot does not certify the input matrix.
///
/// Total arithmetic O(d^3), live working storage O(d^2). Both the lower
/// factor and the inverse-transpose columns use packed triangular storage.
/// No eigensolve or complete inverse matrix is needed at any prefix.
/// The operation is serial with a fixed reduction tree. The dimension ceiling
/// of 8193 is an operational safeguard (quadratic MPFR memory), not a formula
/// constant. Callers needing a larger ladder must explicitly extend and qualify
/// this limit rather than allocate unbounded storage from a malformed request.
pub fn analyze_prefixes(
    matrix: &[Float],
    d: usize,
    p: u32,
    pivot_margin_bits: u32,
    checkpoints: &[usize],
) -> Result<PrefixAnalysisReport> {
    if d == 0
        || d.checked_mul(d) != Some(matrix.len())
        || d > 8193
        || p < 64
        || p > i32::MAX as u32 - 64
        || pivot_margin_bits >= p
    {
        bail!("invalid dimension, precision, or pivot policy");
    }
    // Bound additional factor/innovation storage before allocating. This is
    // a conservative operational estimate, not an RSS measurement or a
    // mathematical constant; larger jobs require a separately qualified API.
    let estimated_bytes = 2 * (d as u128) * (d as u128) * (u128::from(p).div_ceil(8) + 96);
    if estimated_bytes > 16 * 1024_u128.pow(3) {
        bail!("prefix analysis exceeds the 16-GiB estimated working-storage limit");
    }
    if matrix.iter().any(|v| !v.is_finite() || v.prec() > p) {
        bail!("entries must be finite; analysis must not down-round the source");
    }
    if checkpoints.iter().any(|&k| k == 0 || k > d) || checkpoints.windows(2).any(|w| w[0] >= w[1])
    {
        bail!("checkpoint dimensions must be strictly increasing within the source");
    }
    for i in 0..d {
        for j in 0..i {
            if matrix[i * d + j] != matrix[j * d + i] {
                bail!("exact symmetric storage required");
            }
        }
    }
    let mut report = PrefixAnalysisReport {
        semantics: "prefix-spd-unpivoted-ldlt-innovation-gram-v1".into(),
        precision_bits: p,
        pivot_margin_bits,
        source_precision_bits: {
            let mut bits: Vec<u32> = matrix.iter().map(Float::prec).collect();
            bits.sort_unstable();
            bits.dedup();
            bits
        },
        requested_dimension: d,
        rows: Vec::new(),
        checkpoint_innovations: BTreeMap::new(),
        stopped: None,
        assurance: "computed_point_diagnostic_not_a_certificate".into(),
    };
    let mut lower: Vec<Vec<Float>> = Vec::with_capacity(d);
    let mut diagonal: Vec<Float> = Vec::with_capacity(d);
    // Column j of L^{-T} is v_j, supported on indices 0..=j.
    let mut innovations: Vec<Vec<Float>> = Vec::with_capacity(d);
    let mut t1 = Float::with_val(p, 0);
    let mut t2 = Float::with_val(p, 0);
    for j in 0..d {
        lower.push(vec![Float::with_val(p, 0); j + 1]);
        for k in 0..j {
            let correction = sum(
                (0..k)
                    .map(|i| {
                        let mut value = lower[j][i].clone();
                        value *= &diagonal[i];
                        value *= &lower[k][i];
                        value
                    })
                    .collect(),
                p,
            );
            let mut value = Float::with_val(p, &matrix[j * d + k]);
            value -= correction;
            value /= &diagonal[k];
            if !value.is_finite() {
                bail!("nonfinite factor entry at ({j},{k})");
            }
            lower[j][k] = value;
        }
        let corrections: Vec<Float> = (0..j)
            .map(|i| {
                let mut value = lower[j][i].clone().square();
                value *= &diagonal[i];
                value
            })
            .collect();
        let mut scale = Float::with_val(p, &matrix[j * d + j]).abs();
        scale += sum(corrections.iter().map(|v| v.clone().abs()).collect(), p);
        let mut pivot = Float::with_val(p, &matrix[j * d + j]);
        pivot -= sum(corrections, p);
        let mut floor = scale.clone();
        floor *= Float::with_val(p, 2).pow(-((p - pivot_margin_bits) as i32));
        let reason = if !pivot.is_finite() || !scale.is_finite() {
            Some("nonfinite_computed_pivot")
        } else if pivot <= 0 {
            Some("nonpositive_computed_pivot_not_a_definiteness_proof")
        } else if pivot <= floor {
            Some("insufficient_computed_pivot_margin")
        } else {
            None
        };
        if let Some(reason) = reason {
            report.stopped = Some(PrefixStop {
                attempted_dimension: j + 1,
                reason: reason.into(),
                pivot: lossless_decimal(&pivot),
                scale: lossless_decimal(&scale),
            });
            break;
        }
        lower[j][j] = Float::with_val(p, 1);
        let mut innovation = vec![Float::with_val(p, 0); j + 1];
        innovation[j] = Float::with_val(p, 1);
        for i in (0..j).rev() {
            let value = sum(
                (i + 1..=j)
                    .map(|k| {
                        let mut value = lower[k][i].clone();
                        value *= &innovation[k];
                        value
                    })
                    .collect(),
                p,
            );
            innovation[i] = -value;
        }
        let mass = dot(&innovation, &innovation, p);
        let mut increment = mass.clone();
        increment /= &pivot;
        let cross = sum(
            innovations
                .iter()
                .enumerate()
                .map(|(i, old)| {
                    let mut term = dot(old, &innovation[..=i], p).square();
                    term /= &diagonal[i];
                    term /= &pivot;
                    term
                })
                .collect(),
            p,
        );
        t1 += &increment;
        let mut delta_t2 = cross;
        delta_t2 *= 2;
        delta_t2 += increment.clone().square();
        t2 += delta_t2;
        if [&mass, &increment, &t1, &t2]
            .iter()
            .any(|v| !v.is_finite() || **v <= 0)
        {
            bail!("nonfinite diagnostic at prefix {}", j + 1);
        }
        let mut effective_rank = t1.clone().square();
        effective_rank /= &t2;
        let mut fraction = increment.clone();
        fraction /= &t1;
        let mut lower_estimate = Float::with_val(p, 1);
        lower_estimate /= t2.clone().sqrt();
        let mut upper_estimate = t1.clone();
        upper_estimate /= &t2;
        let depth_lower = -upper_estimate.clone().log10();
        let depth_upper = -lower_estimate.clone().log10();
        if [&effective_rank, &fraction, &lower_estimate, &upper_estimate]
            .iter()
            .any(|v| !v.is_finite() || **v <= 0)
            || !depth_lower.is_finite()
            || !depth_upper.is_finite()
        {
            bail!(
                "unresolved or nonfinite moment-derived estimate at prefix {}",
                j + 1
            );
        }
        report.rows.push(PrefixRow {
            dimension: j + 1,
            sigma: lossless_decimal(&pivot),
            innovation_mass: lossless_decimal(&mass),
            inverse_trace_increment: lossless_decimal(&increment),
            inverse_trace: lossless_decimal(&t1),
            inverse_square_trace: lossless_decimal(&t2),
            pivot_cancellation_scale: lossless_decimal(&scale),
            effective_inverse_rank: lossless_decimal(&effective_rank),
            newest_inverse_trace_fraction: lossless_decimal(&fraction),
            smallest_eigenvalue_lower_estimate: lossless_decimal(&lower_estimate),
            smallest_eigenvalue_upper_estimate: lossless_decimal(&upper_estimate),
            eigenvalue_depth_lower_estimate: lossless_decimal(&depth_lower),
            eigenvalue_depth_upper_estimate: lossless_decimal(&depth_upper),
        });
        if checkpoints.binary_search(&(j + 1)).is_ok() {
            report
                .checkpoint_innovations
                .insert(j + 1, innovation.iter().map(lossless_decimal).collect());
        }
        diagonal.push(pivot);
        innovations.push(innovation);
    }
    Ok(report)
}

/// Decimal export is read back at the SAME arithmetic precision. The returned
/// bytes have not been accepted until the requested identity check passes.
/// Checks receive the actual decoded values, not the unrounded originals.
/// Candidate widths and their limit are explicit, so the result is deterministic.
pub fn checked_decimal_export<F>(
    values: &[Float],
    digits_schedule: &[usize],
    check: F,
) -> Result<(usize, Vec<String>)>
where
    F: Fn(&[Float]) -> Result<bool>,
{
    if values.is_empty()
        || values.iter().any(|v| !v.is_finite())
        || digits_schedule.is_empty()
        || digits_schedule[0] == 0
        || digits_schedule.last().is_some_and(|&n| n > 1_000_000)
        || digits_schedule.windows(2).any(|w| w[0] >= w[1])
    {
        bail!("invalid decimal export input or schedule");
    }
    if !check(values)? {
        bail!("the source values already fail the requested export identity");
    }
    for &digits in digits_schedule {
        let encoded: Vec<String> = values
            .iter()
            .map(|v| v.to_string_radix(10, Some(digits)))
            .collect();
        let decoded: Vec<Float> = values
            .iter()
            .zip(&encoded)
            .map(|(v, s)| Ok(Float::with_val(v.prec(), Float::parse(s)?)))
            .collect::<Result<_>>()?;
        if check(&decoded)? {
            return Ok((digits, encoded));
        }
    }
    bail!("export identity remains unresolved at the maximum declared width")
}
