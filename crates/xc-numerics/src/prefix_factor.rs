//! Arithmetic-preserving schedules for unpivoted prefix factorization.
use super::{lossless_decimal, sum, PairwiseScratch, PrefixStop};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{ops::Pow, Assign, Float};

pub(super) struct Factorization {
    pub lower: Vec<Vec<Float>>,
    pub diagonal: Vec<Float>,
    pub scales: Vec<Float>,
    pub stopped: Option<PrefixStop>,
}

// Retained row schedule also avoids parallel scheduling overhead on one worker.
pub(super) fn row_major(
    matrix: &[Float],
    d: usize,
    p: u32,
    pivot_margin_bits: u32,
) -> Result<Factorization> {
    let mut lower: Vec<Vec<Float>> = Vec::with_capacity(d);
    let mut diagonal: Vec<Float> = Vec::with_capacity(d);
    let mut scales: Vec<Float> = Vec::with_capacity(d);
    let mut scratch = PairwiseScratch::new(p);
    let mut stopped = None;
    for j in 0..d {
        lower.push(vec![Float::with_val(p, 0); j + 1]);
        for k in 0..j {
            let leaf = |i: usize, value: &mut Float| {
                value.assign(&lower[j][i]);
                *value *= &diagonal[i];
                *value *= &lower[k][i];
            };
            let correction = scratch.sum_by_indexed(k, leaf);
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
            stopped = Some(PrefixStop {
                attempted_dimension: j + 1,
                reason: reason.into(),
                pivot: lossless_decimal(&pivot),
                scale: lossless_decimal(&scale),
            });
            break;
        }
        lower[j][j] = Float::with_val(p, 1);
        diagonal.push(pivot);
        scales.push(scale);
    }
    Ok(Factorization {
        lower,
        diagonal,
        scales,
        stopped,
    })
}

pub(super) fn column_major(
    matrix: &[Float],
    d: usize,
    p: u32,
    pivot_margin_bits: u32,
) -> Result<Factorization> {
    // Reserve packed row capacity, but construct each MPFR value only when its
    // column is reached. No dense zero-filled MPFR factor is materialized.
    let mut lower: Vec<Vec<Float>> = (0..d).map(|j| Vec::with_capacity(j + 1)).collect();
    let mut diagonal: Vec<Float> = Vec::with_capacity(d);
    let mut scales: Vec<Float> = Vec::with_capacity(d);
    const ROW_BLOCK: usize = 16;
    let mut workers: Vec<_> = (0..d.div_ceil(ROW_BLOCK))
        .map(|_| PairwiseScratch::new(p))
        .collect();
    let mut scratch = PairwiseScratch::new(p);
    let mut stopped = None;
    for k in 0..d {
        // Entries in future rows are speculative. Inspect failures only when
        // their row is reached, preserving the original first failure and any
        // earlier pivot stop. In particular, an unused overflow cannot discard
        // an otherwise valid accepted prefix.
        if let Some(i) = lower[k].iter().position(|value| !value.is_finite()) {
            bail!("nonfinite factor entry at ({k},{i})");
        }
        let corrections: Vec<Float> = (0..k)
            .map(|i| {
                let mut value = lower[k][i].clone().square();
                value *= &diagonal[i];
                value
            })
            .collect();
        let mut scale = Float::with_val(p, &matrix[k * d + k]).abs();
        scale += sum(corrections.iter().map(|v| v.clone().abs()).collect(), p);
        let mut pivot = Float::with_val(p, &matrix[k * d + k]);
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
            stopped = Some(PrefixStop {
                attempted_dimension: k + 1,
                reason: reason.into(),
                pivot: lossless_decimal(&pivot),
                scale: lossless_decimal(&scale),
            });
            break;
        }
        diagonal.push(pivot);
        scales.push(scale);
        lower[k].push(Float::with_val(p, 1));
        let (accepted, pending) = lower.split_at_mut(k + 1);
        let pivot_row = &accepted[k];
        let entry =
            |j: usize, row: &mut Vec<Float>, local: &mut PairwiseScratch, parallel_inner: bool| {
                let leaf = |i: usize, value: &mut Float| {
                    value.assign(&row[i]);
                    *value *= &diagonal[i];
                    *value *= &pivot_row[i];
                };
                let correction = if parallel_inner {
                    local.sum_by_indexed(k, leaf)
                } else {
                    local.sum_by(k, leaf)
                };
                let mut value = Float::with_val(p, &matrix[j * d + k]);
                value -= correction;
                value /= &diagonal[k];
                row.push(value);
            };
        if rayon::current_num_threads() > 1 && k >= 16 && pending.len() >= 2 * ROW_BLOCK {
            pending
                .par_chunks_mut(ROW_BLOCK)
                .zip(workers.par_iter_mut())
                .enumerate()
                .for_each(|(block, (rows, local))| {
                    for (offset, row) in rows.iter_mut().enumerate() {
                        entry(k + 1 + block * ROW_BLOCK + offset, row, local, false);
                    }
                });
        } else {
            for (offset, row) in pending.iter_mut().enumerate() {
                entry(k + 1 + offset, row, &mut scratch, true);
            }
        }
    }
    lower.truncate(diagonal.len());
    Ok(Factorization {
        lower,
        diagonal,
        scales,
        stopped,
    })
}
