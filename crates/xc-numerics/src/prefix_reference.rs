// Frozen interleaved arithmetic from v0.15.0 commit 178dc50.
// Test-only reference for the phase scheduling rewrite; optional fields defaulted.
use super::*;
use rug::ops::Pow;

pub(super) fn interleaved_reference(
    matrix: &[Float],
    d: usize,
    p: u32,
    pivot_margin_bits: u32,
    checkpoints: &[usize],
    parallel_inner: bool,
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
        semantics: "prefix-spd-unpivoted-ldlt-innovation-gram-v2".into(),
        diagnostic_policy: PrefixDiagnosticPolicy::default(),
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
    for j in 0..d {
        lower.push(vec![Float::with_val(p, 0); j + 1]);
        for k in 0..j {
            let leaf = |i: usize, value: &mut Float| {
                value.assign(&lower[j][i]);
                *value *= &diagonal[i];
                *value *= &lower[k][i];
            };
            let correction = if parallel_inner {
                scratch.sum_by_indexed(k, leaf)
            } else {
                scratch.sum_by(k, leaf)
            };
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
        let mut cancellation_ratio = Float::with_val(p, 1);
        let mut cancellation_sum = Float::with_val(p, 0);
        let mut cancellation_result = Float::with_val(p, 0);
        let mut worst_component = None;
        let mut zero_result_with_nonzero_terms = false;
        for i in (0..j).rev() {
            let (value, absolute_sum) =
                scratch.sum_by_with_absolute(j - i, parallel_inner, |offset, value| {
                    let k = i + 1 + offset;
                    value.assign(&lower[k][i]);
                    *value *= &innovation[k];
                });
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
        let mass = dot(&innovation, &innovation, p);
        let mut increment = mass.clone();
        increment /= &pivot;
        let cross_term = |i: usize, term: &mut Float, dot_scratch: &mut PairwiseScratch| {
            term.assign(dot_scratch.sum_by(i + 1, |k, product| {
                product.assign(&innovations[i][k]);
                *product *= &innovation[k];
            }));
            term.square_mut();
            *term /= &diagonal[i];
            *term /= &pivot;
        };
        let cross = if parallel_cross && j >= 4 * CROSS_BLOCK {
            cross_leaves[..j]
                .par_chunks_mut(CROSS_BLOCK)
                .zip(cross_scratch.par_iter_mut())
                .enumerate()
                .for_each(|(block, (terms, local))| {
                    for (offset, term) in terms.iter_mut().enumerate() {
                        cross_term(block * CROSS_BLOCK + offset, term, local);
                    }
                });
            scratch.sum_by(j, |i, term| term.assign(&cross_leaves[i]))
        } else {
            scratch.sum_by(j, |i, term| cross_term(i, term, &mut dot_scratch))
        };
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
        report.rows.push(PrefixRow {
            third_inverse_moment: None,
            dimension: j + 1,
            sigma: lossless_decimal(&pivot),
            innovation_mass: lossless_decimal(&mass),
            inverse_trace_increment: lossless_decimal(&increment),
            inverse_trace: lossless_decimal(&t1),
            inverse_square_trace: lossless_decimal(&t2),
            pivot_cancellation_scale: lossless_decimal(&scale),
            innovation_cancellation: Some(InnovationCancellation {
                worst_component,
                absolute_term_sum: lossless_decimal(&cancellation_sum),
                absolute_result: lossless_decimal(&cancellation_result),
                ratio: (!zero_result_with_nonzero_terms)
                    .then(|| lossless_decimal(&cancellation_ratio)),
                decimal_digits_lost: (!zero_result_with_nonzero_terms)
                    .then(|| lossless_decimal(&cancellation_ratio.clone().log10())),
                zero_result_with_nonzero_terms,
            }),
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
        diagonal.push(pivot);
        innovations.push(innovation);
    }
    Ok(report)
}
