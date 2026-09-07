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

use crate::reduction::{
    deterministic_pairwise_sum_hp_owned, roundtrip_decimal_digits, PairwiseScratch,
};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{Assign, Float};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub use xc_core::PrefixDiagnosticPolicy;
#[path = "prefix_factor.rs"]
mod factor;
#[path = "prefix_moments.rs"]
mod moments;
pub use moments::{ThirdInverseMoment, TwoModeMomentFit, TwoModeMomentStatus};
#[cfg(test)]
#[path = "prefix_reference.rs"]
mod reference;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub third_inverse_moment: Option<ThirdInverseMoment>,
    pub pivot_cancellation_scale: String,
    /// Absent in historical v1 reports; never interpret absence as zero loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub innovation_cancellation: Option<InnovationCancellation>,
    pub effective_inverse_rank: String,
    pub newest_inverse_trace_fraction: String,
    pub smallest_eigenvalue_lower_estimate: String,
    pub smallest_eigenvalue_upper_estimate: String,
    /// Width proxy U/L-1; approximates lambda_1/lambda_2 only with a
    /// dominant second inverse mode. Absent for dimension one or unresolved width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap_ratio_estimate: Option<String>,
    /// Two-mode asymptotic model L*(1+(U/L-1)^2/2), never a bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smallest_eigenvalue_second_order_estimate: Option<String>,
    pub eigenvalue_depth_lower_estimate: String,
    pub eigenvalue_depth_upper_estimate: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InnovationCancellation {
    /// Component with the largest computed absolute-term-sum / result ratio.
    /// None when no solve component exists (dimension one).
    pub worst_component: Option<usize>,
    pub absolute_term_sum: String,
    pub absolute_result: String,
    /// None for zero result with nonzero terms. The ratio for 0/0 is one
    /// (no cancellation in an identically zero inner sum).
    pub ratio: Option<String>,
    pub decimal_digits_lost: Option<String>,
    pub zero_result_with_nonzero_terms: bool,
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
    #[serde(
        default,
        skip_serializing_if = "PrefixDiagnosticPolicy::is_legacy_default"
    )]
    pub diagnostic_policy: PrefixDiagnosticPolicy,
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

struct SolvedInnovation {
    values: Vec<Float>,
    cancellation: Option<InnovationCancellation>,
}
fn solve_innovation(
    lower: &[Vec<Float>],
    j: usize,
    p: u32,
    capture_cancellation: bool,
    scratch: &mut PairwiseScratch,
) -> Result<SolvedInnovation> {
    let mut innovation = vec![Float::with_val(p, 0); j + 1];
    innovation[j] = Float::with_val(p, 1);
    let mut cancellation_ratio = Float::with_val(p, 1);
    let mut cancellation_sum = Float::with_val(p, 0);
    let mut cancellation_result = Float::with_val(p, 0);
    let mut worst_component = None;
    let mut zero_result_with_nonzero_terms = false;
    for i in (0..j).rev() {
        let leaf = |offset: usize, value: &mut Float| {
            let k = i + 1 + offset;
            value.assign(&lower[k][i]);
            *value *= &innovation[k];
        };
        let (value, absolute_sum) = if capture_cancellation {
            scratch.sum_by_with_absolute(j - i, false, leaf)
        } else {
            let value = scratch.sum_by(j - i, leaf);
            if !value.is_finite() {
                bail!("nonfinite innovation at prefix {}", j + 1);
            }
            innovation[i] = -value;
            continue;
        };
        let absolute_result = value.clone().abs();
        if !absolute_sum.is_finite() || !absolute_result.is_finite() {
            bail!(
                "nonfinite innovation solve or cancellation scale at prefix {}",
                j + 1
            );
        }
        if absolute_result.is_zero() && !absolute_sum.is_zero() {
            if !zero_result_with_nonzero_terms {
                worst_component = Some(i);
                cancellation_sum.assign(&absolute_sum);
                cancellation_result.assign(&absolute_result);
            }
            zero_result_with_nonzero_terms = true;
        } else if !zero_result_with_nonzero_terms {
            let ratio = if absolute_result.is_zero() {
                Float::with_val(p, 1)
            } else {
                absolute_sum.clone() / &absolute_result
            };
            if worst_component.is_none() || ratio > cancellation_ratio {
                worst_component = Some(i);
                cancellation_ratio.assign(&ratio);
                cancellation_sum.assign(&absolute_sum);
                cancellation_result.assign(&absolute_result);
            }
        }
        innovation[i] = -value;
    }
    Ok(SolvedInnovation {
        values: innovation,
        cancellation: capture_cancellation.then(|| InnovationCancellation {
            worst_component,
            absolute_term_sum: lossless_decimal(&cancellation_sum),
            absolute_result: lossless_decimal(&cancellation_result),
            ratio: (!zero_result_with_nonzero_terms).then(|| lossless_decimal(&cancellation_ratio)),
            decimal_digits_lost: (!zero_result_with_nonzero_terms)
                .then(|| lossless_decimal(&cancellation_ratio.clone().log10())),
            zero_result_with_nonzero_terms,
        }),
    })
}

/// Computes the ladder of the supplied symmetric point matrix, in its exact
/// supplied order, without pivoting, regularization, or source recomputation.
/// The pivot margin is a declared numerical screening policy, NOT a proof
/// error bound. A positive computed pivot does not certify the input matrix.
///
/// Total arithmetic O(d^3), live working storage O(d^2). Both the lower
/// factor and the inverse-transpose columns use packed triangular storage.
/// Factorization, independent innovation solves and moment accumulation run in
/// separate phases. A stopped factorization still returns its accepted prefixes.
/// No eigensolve or complete inverse matrix is needed at any prefix.
/// Pivots and triangular-solve dependencies remain ordered. Independent lower-
/// factor entries within a column, innovation columns and Gram cross terms run
/// in parallel on sufficiently large inputs, retaining every inner tree and
/// separately rounded operation. Smaller or single-worker inputs use row-order
/// factorization to avoid scheduling overhead.
/// The dimension ceiling
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
    analyze_prefixes_with_policy(
        matrix,
        d,
        p,
        pivot_margin_bits,
        checkpoints,
        &PrefixDiagnosticPolicy::default(),
    )
}

/// Analyze with explicitly selected additional data. The third moment adds a
/// retained Gram triangle and another cubic accumulation; it is not free work.
pub fn analyze_prefixes_with_policy(
    matrix: &[Float],
    d: usize,
    p: u32,
    pivot_margin_bits: u32,
    checkpoints: &[usize],
    policy: &PrefixDiagnosticPolicy,
) -> Result<PrefixAnalysisReport> {
    analyze_prefixes_impl(
        matrix,
        d,
        p,
        pivot_margin_bits,
        checkpoints,
        policy,
        d >= 128 && rayon::current_num_threads() > 1,
    )
}

fn analyze_prefixes_impl(
    matrix: &[Float],
    d: usize,
    p: u32,
    pivot_margin_bits: u32,
    checkpoints: &[usize],
    policy: &PrefixDiagnosticPolicy,
    column_factor: bool,
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
    let estimated_bytes = (if policy.third_inverse_moment { 3 } else { 2 })
        * (d as u128)
        * (d as u128)
        * (u128::from(p).div_ceil(8) + 96);
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
        semantics: if policy.is_legacy_default() {
            "prefix-spd-unpivoted-ldlt-innovation-gram-v2"
        } else {
            "prefix-spd-unpivoted-ldlt-innovation-gram-v3"
        }
        .into(),
        diagnostic_policy: *policy,
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
    let factor = if column_factor {
        factor::column_major(matrix, d, p, pivot_margin_bits)?
    } else {
        factor::row_major(matrix, d, p, pivot_margin_bits)?
    };
    let factor::Factorization {
        lower,
        diagonal,
        scales,
        stopped,
    } = factor;
    report.stopped = stopped;
    // All L^{-T} columns are independent once the accepted factor prefix exists.
    // Indexed collection and ordered error inspection preserve source row order.
    let results: Vec<_> = (0..diagonal.len())
        .into_par_iter()
        .map_init(
            || PairwiseScratch::new(p),
            |local, j| solve_innovation(&lower, j, p, policy.innovation_cancellation, local),
        )
        .collect();
    let innovations = results.into_iter().collect::<Result<Vec<_>>>()?;
    drop(lower);
    let mut t1 = Float::with_val(p, 0);
    let mut t2 = Float::with_val(p, 0);
    let mut scratch = PairwiseScratch::new(p);
    let mut dot_scratch = PairwiseScratch::new(p);
    // Fixed indexed blocks amortize scheduling and retain each block's MPFR
    // allocations across prefixes. Worker count never changes the sum tree.
    const CROSS_BLOCK: usize = 32;
    let parallel_cross = rayon::current_num_threads() > 1;
    let mut cross_leaves = vec![Float::with_val(p, 0); d];
    let mut cross_scratch: Vec<_> = (0..d.div_ceil(CROSS_BLOCK))
        .map(|_| PairwiseScratch::new(p))
        .collect();
    let mut cube = policy
        .third_inverse_moment
        .then(|| moments::CubeAccumulator::new(p, &diagonal));
    for j in 0..diagonal.len() {
        let pivot = &diagonal[j];
        let scale = &scales[j];
        let innovation = &innovations[j].values;
        let mass = dot(innovation, innovation, p);
        let mut increment = mass.clone();
        increment /= pivot;
        let mut gram_row = cube.as_ref().map(|_| vec![Float::with_val(p, 0); j + 1]);
        let cross_term = |i: usize,
                          term: &mut Float,
                          normalized: Option<&mut Float>,
                          dot_scratch: &mut PairwiseScratch| {
            term.assign(dot_scratch.sum_by(i + 1, |k, product| {
                product.assign(&innovations[i].values[k]);
                *product *= &innovation[k];
            }));
            if let Some(normalized) = normalized {
                normalized.assign(&*term);
                let roots = &cube.as_ref().expect("requested Gram row").pivot_roots;
                *normalized /= &roots[i];
                *normalized /= &roots[j];
            }
            term.square_mut();
            *term /= &diagonal[i];
            *term /= pivot;
        };
        if parallel_cross && j >= 4 * CROSS_BLOCK {
            if let Some(row) = &mut gram_row {
                cross_leaves[..j]
                    .par_chunks_mut(CROSS_BLOCK)
                    .zip(row[..j].par_chunks_mut(CROSS_BLOCK))
                    .zip(cross_scratch.par_iter_mut())
                    .enumerate()
                    .for_each(|(block, ((terms, normalized), local))| {
                        for (offset, (term, g)) in terms.iter_mut().zip(normalized).enumerate() {
                            cross_term(block * CROSS_BLOCK + offset, term, Some(g), local);
                        }
                    });
            } else {
                cross_leaves[..j]
                    .par_chunks_mut(CROSS_BLOCK)
                    .zip(cross_scratch.par_iter_mut())
                    .enumerate()
                    .for_each(|(block, (terms, local))| {
                        for (offset, term) in terms.iter_mut().enumerate() {
                            cross_term(block * CROSS_BLOCK + offset, term, None, local);
                        }
                    });
            }
        } else {
            for (i, term) in cross_leaves[..j].iter_mut().enumerate() {
                cross_term(
                    i,
                    term,
                    gram_row.as_mut().map(|r| &mut r[i]),
                    &mut dot_scratch,
                );
            }
        }
        let cross = scratch.sum_by(j, |i, term| term.assign(&cross_leaves[i]));
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
        let mut width = upper_estimate.clone() / &lower_estimate;
        width -= 1;
        let model_resolved = width.is_finite() && width >= 0;
        let second_order = (j == 0 || model_resolved).then(|| {
            if j == 0 {
                return lossless_decimal(&lower_estimate);
            }
            let mut correction = width.clone().square();
            correction /= 2;
            correction += 1;
            correction *= &lower_estimate;
            lossless_decimal(&correction)
        });
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
        let third_inverse_moment = if let (Some(cube), Some(mut row)) = (&mut cube, gram_row) {
            row[j].assign(&increment);
            Some(cube.append(row, &t1, &t2)?)
        } else {
            None
        };
        report.rows.push(PrefixRow {
            third_inverse_moment,
            dimension: j + 1,
            sigma: lossless_decimal(pivot),
            innovation_mass: lossless_decimal(&mass),
            inverse_trace_increment: lossless_decimal(&increment),
            inverse_trace: lossless_decimal(&t1),
            inverse_square_trace: lossless_decimal(&t2),
            pivot_cancellation_scale: lossless_decimal(scale),
            innovation_cancellation: innovations[j].cancellation.clone(),
            effective_inverse_rank: lossless_decimal(&effective_rank),
            newest_inverse_trace_fraction: lossless_decimal(&fraction),
            smallest_eigenvalue_lower_estimate: lossless_decimal(&lower_estimate),
            smallest_eigenvalue_upper_estimate: lossless_decimal(&upper_estimate),
            gap_ratio_estimate: (j > 0 && model_resolved).then(|| lossless_decimal(&width)),
            smallest_eigenvalue_second_order_estimate: second_order,
            eigenvalue_depth_lower_estimate: lossless_decimal(&depth_lower),
            eigenvalue_depth_upper_estimate: lossless_decimal(&depth_upper),
        });
        if checkpoints.binary_search(&(j + 1)).is_ok() {
            report
                .checkpoint_innovations
                .insert(j + 1, innovation.iter().map(lossless_decimal).collect());
        }
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

#[cfg(test)]
mod phase_tests {
    use super::*;
    fn fixture(d: usize, p: u32) -> Vec<Float> {
        (0..d * d)
            .map(|index| {
                let (i, j) = (index / d, index % d);
                let value = if i == j {
                    (16 * d + i) as i32
                } else {
                    let magnitude = ((i + j) % 7 + 1) as i32;
                    if (i + j) % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                };
                Float::with_val(p, value)
            })
            .collect()
    }
    #[test]
    fn phased_reports_match_the_frozen_interleaved_algorithm() {
        for p in [64, 128, 256, 512, 2722] {
            let d = 137;
            let matrix = fixture(d, p);
            let expected = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| {
                    reference::interleaved_reference(&matrix, d, p, 32, &[1, 127, 137], true)
                        .unwrap()
                });
            for workers in [1, 2, 4] {
                let actual = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap()
                    .install(|| analyze_prefixes(&matrix, d, p, 32, &[1, 127, 137]).unwrap());
                assert_eq!(
                    serde_json::to_vec(&expected).unwrap(),
                    serde_json::to_vec(&actual).unwrap()
                );
            }
        }
        for last in [0, -1] {
            let matrix = [2, 1, 0, 1, 2, 0, 0, 0, last].map(|v| Float::with_val(128, v));
            let old =
                reference::interleaved_reference(&matrix, 3, 128, 32, &[1, 2, 3], true).unwrap();
            let new = analyze_prefixes(&matrix, 3, 128, 32, &[1, 2, 3]).unwrap();
            assert_eq!(old, new);
        }
    }
    #[test]
    fn column_factor_preserves_pivot_stops_and_failure_order() {
        for p in [64, 256, 2722] {
            let d = 137;
            for stopped_row in [0, 16, 127, 136] {
                let mut matrix = fixture(d, p);
                matrix[stopped_row * d + stopped_row] = Float::with_val(p, 0);
                let expected = analyze_prefixes_impl(
                    &matrix,
                    d,
                    p,
                    32,
                    &[1, 127, 137],
                    &PrefixDiagnosticPolicy::full(),
                    false,
                )
                .unwrap();
                for workers in [1, 4] {
                    let actual = rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap()
                        .install(|| {
                            analyze_prefixes_impl(
                                &matrix,
                                d,
                                p,
                                32,
                                &[1, 127, 137],
                                &PrefixDiagnosticPolicy::full(),
                                true,
                            )
                            .unwrap()
                        });
                    assert_eq!(expected, actual);
                    assert_eq!(actual.rows.len(), stopped_row);
                }
            }
        }
        // A future factor overflow must not pre-empt an earlier pivot stop.
        // If that future row is reached, report the same first failing entry.
        let p = 128;
        let tiny = Float::with_val(p, 1) >> 100_i32;
        let huge = Float::with_val(p, 1) << 1_073_741_773_i32;
        assert!(tiny.is_finite() && tiny > 0 && huge.is_finite());
        assert!((huge.clone() / &tiny).is_infinite());
        for middle in [0, 1] {
            let mut matrix = vec![Float::with_val(p, 0); 9];
            matrix[0] = tiny.clone();
            matrix[4] = Float::with_val(p, middle);
            matrix[2] = huge.clone();
            matrix[6] = huge.clone();
            matrix[8] = Float::with_val(p, 1);
            let run = |columns| {
                analyze_prefixes_impl(
                    &matrix,
                    3,
                    p,
                    32,
                    &[1, 2, 3],
                    &PrefixDiagnosticPolicy::default(),
                    columns,
                )
            };
            if middle == 0 {
                let row = run(false).unwrap();
                let column = run(true).unwrap();
                assert_eq!(row, column);
                assert_eq!(column.rows.len(), 1);
            } else {
                assert_eq!(
                    run(false).unwrap_err().to_string(),
                    "nonfinite factor entry at (2,0)"
                );
                assert_eq!(
                    run(true).unwrap_err().to_string(),
                    "nonfinite factor entry at (2,0)"
                );
            }
        }
        let mut margin_matrix = [1, 1, 1, 1].map(|v| Float::with_val(p, v));
        margin_matrix[3] += Float::with_val(p, 1) >> 112_i32;
        let run = |columns| {
            analyze_prefixes_impl(
                &margin_matrix,
                2,
                p,
                32,
                &[1, 2],
                &PrefixDiagnosticPolicy::default(),
                columns,
            )
            .unwrap()
        };
        assert_eq!(run(false), run(true));
        assert_eq!(
            run(true).stopped.unwrap().reason,
            "insufficient_computed_pivot_margin"
        );
    }

    #[test]
    #[ignore = "explicit paired row/column factor scheduling performance measurement"]
    fn qualify_prefix_column_factor() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (d, p) in [(640, 256), (640, 2722), (1024, 256)] {
            let matrix = fixture(d, p);
            let run = |columns| {
                pool.install(|| {
                    let start = std::time::Instant::now();
                    let report = analyze_prefixes_impl(
                        &matrix,
                        d,
                        p,
                        32,
                        &[],
                        &PrefixDiagnosticPolicy::default(),
                        columns,
                    )
                    .unwrap();
                    (start.elapsed().as_secs_f64(), report)
                })
            };
            let (row_seconds, row) = run(false);
            let (column_seconds, column) = run(true);
            let identity =
                serde_json::to_vec(&row).unwrap() == serde_json::to_vec(&column).unwrap();
            assert!(identity);
            println!(
                "PREFIX_COLUMN_BENCH {}",
                serde_json::json!({
                    "dimension":d,"precision_bits":p,"workers":4,
                    "fixture":"signed_dense_strictly_diagonally_dominant",
                    "row_factor_seconds":row_seconds,"column_factor_seconds":column_seconds,
                    "report_byte_identity":identity,
                    "scope":"one paired measurement; identical phased algorithm except factor scheduling; third moment disabled"
                })
            );
        }
    }

    #[test]
    #[ignore = "explicit paired scheduling and cancellation performance measurement"]
    fn qualify_prefix_phase_split() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (d, p) in [(640, 256), (640, 2722), (1024, 256)] {
            let matrix = fixture(d, p);
            let (old_seconds, old) = pool.install(|| {
                let start = std::time::Instant::now();
                let report =
                    reference::interleaved_reference(&matrix, d, p, 32, &[], true).unwrap();
                (start.elapsed().as_secs_f64(), report)
            });
            let (new_seconds, new) = pool.install(|| {
                let start = std::time::Instant::now();
                let report = analyze_prefixes(&matrix, d, p, 32, &[]).unwrap();
                (start.elapsed().as_secs_f64(), report)
            });
            let identity = serde_json::to_vec(&old).unwrap() == serde_json::to_vec(&new).unwrap();
            assert!(identity);
            let (without_seconds, without) = pool.install(|| {
                let start = std::time::Instant::now();
                let report = analyze_prefixes_with_policy(
                    &matrix,
                    d,
                    p,
                    32,
                    &[],
                    &PrefixDiagnosticPolicy {
                        third_inverse_moment: false,
                        innovation_cancellation: false,
                    },
                )
                .unwrap();
                (start.elapsed().as_secs_f64(), report)
            });
            let mut scalar_reference = new.clone();
            scalar_reference.semantics = without.semantics.clone();
            scalar_reference.diagnostic_policy = without.diagnostic_policy;
            for row in &mut scalar_reference.rows {
                row.innovation_cancellation = None;
            }
            let omission_identity = scalar_reference == without;
            assert!(omission_identity);
            println!(
                "PREFIX_PHASE_BENCH {}",
                serde_json::json!({
                    "dimension":d,"precision_bits":p,"workers":4,
                    "fixture":"signed_dense_strictly_diagonally_dominant",
                    "interleaved_seconds":old_seconds,"phased_seconds":new_seconds,
                    "phased_without_cancellation_seconds":without_seconds,
                    "report_byte_identity":identity,"other_fields_identical_without_cancellation":omission_identity,
                    "scope":"one paired measurement; identical source and arithmetic; third moment disabled"
                })
            );
        }
    }
}
