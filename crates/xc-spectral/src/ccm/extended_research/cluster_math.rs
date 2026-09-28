//! Guarded finite Euclidean cluster geometry; no branch-identity certificate.
use super::{point, solve_gram_checked};
use anyhow::{bail, Result};
use rug::Float;

#[derive(Debug)]
pub(super) struct Measurement {
    pub(super) gram: Vec<Float>,
    pub(super) overlaps: Vec<Float>,
    pub(super) leakage: Option<Float>,
    pub(super) minimum_pivot: Option<Float>,
    pub(super) previous: Vec<Vec<Float>>,
    pub(super) arithmetic_precision: u32,
}
fn validate(v: &[Float], requested: u32) -> Result<()> {
    if v.is_empty()
        || v.len().is_multiple_of(2)
        || v.len() > 16385
        || v.iter().any(|x| !x.is_finite() || x.prec() > requested)
        || v.iter().all(Float::is_zero)
    {
        bail!("invalid nonzero finite cluster vector or source precision");
    }
    Ok(())
}
fn unit_padded(v: &[Float], len: usize, p: u32) -> Result<Vec<Float>> {
    let unit = point::unit(v, p)?;
    let offset = (len - v.len()) / 2;
    let mut out = vec![Float::with_val(p, 0); len];
    for (i, x) in unit.into_iter().enumerate() {
        out[offset + i] = x;
    }
    Ok(out)
}
fn calculate(
    source: &[Float],
    current: &[Vec<Float>],
    previous: &[Vec<Float>],
    requested: u32,
    p: u32,
) -> Result<Measurement> {
    if !(64..=1_000_000).contains(&requested)
        || current.is_empty()
        || current.len() > 32
        || previous.len() > 32
    {
        bail!("invalid finite cluster shape or precision");
    }
    validate(source, requested)?;
    for v in current.iter().chain(previous) {
        validate(v, requested)?;
    }
    let len = std::iter::once(source.len())
        .chain(current.iter().chain(previous).map(Vec::len))
        .max()
        .unwrap();
    let source = unit_padded(source, len, p)?;
    let current = current
        .iter()
        .map(|v| unit_padded(v, len, p))
        .collect::<Result<Vec<_>>>()?;
    let previous = previous
        .iter()
        .map(|v| unit_padded(v, len, p))
        .collect::<Result<Vec<_>>>()?;
    let mut gram = Vec::with_capacity(current.len() * current.len());
    let mut overlaps = Vec::with_capacity(current.len());
    for a in &current {
        overlaps.push(point::dot(a, &source, p)?);
        for b in &current {
            gram.push(point::dot(a, b, p)?);
        }
    }
    let solved = solve_gram_checked(&gram, &overlaps, p)?;
    let (leakage, pivot) = if let Some((coefficients, pivot)) = solved {
        let mut norm = Float::with_val(p, 0);
        for i in 0..len {
            let mut terms = vec![source[i].clone()];
            for (v, a) in current.iter().zip(&coefficients) {
                terms.push(-point::product(&[&v[i], a], 2 * p)?);
            }
            norm.hypot_mut(&point::sum(&terms, p)?);
        }
        let squared = point::product(&[&norm, &norm], p)?;
        (
            Some(point::output(&squared, requested)?),
            Some(point::output(&pivot, requested)?),
        )
    } else {
        (None, None)
    };
    let mut comparisons = Vec::with_capacity(current.len());
    for a in &current {
        comparisons.push(
            previous
                .iter()
                .map(|b| point::output(&point::dot(a, b, p)?, requested))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok(Measurement {
        gram: gram
            .iter()
            .map(|x| point::output(x, requested))
            .collect::<Result<Vec<_>>>()?,
        overlaps: overlaps
            .iter()
            .map(|x| point::output(x, requested))
            .collect::<Result<Vec<_>>>()?,
        leakage,
        minimum_pivot: pivot,
        previous: comparisons,
        arithmetic_precision: p,
    })
}

/// Precision policy for a point Gram solve, not an interval rank certificate.
/// For exact SPD unit-column G, trace(G)=b and det(G) is the product of LDL
/// pivots. A minimum pivot d gives cond_2(G)<=b^b/d^b. The computed pivot is
/// used as a conservative precision-selection proxy; it is not a certified d.
pub(super) fn measure(
    source: &[Float],
    current: &[Vec<Float>],
    previous: &[Vec<Float>],
    requested: u32,
) -> Result<Measurement> {
    let mut last = None;
    for guard in [64u32, 128, 256, 512, 1024, 2048, 4096] {
        let p = requested
            .checked_add(guard)
            .ok_or_else(|| anyhow::anyhow!("cluster precision overflow"))?;
        let mut result = calculate(source, current, previous, requested, p)?;
        if let Some(pivot) = &result.minimum_pivot {
            let exponent = pivot
                .get_exp()
                .ok_or_else(|| anyhow::anyhow!("zero resolved Gram pivot"))?;
            let b = current.len() as u32;
            let log_b = u32::BITS - (b - 1).leading_zeros();
            let pivot_bits = u32::try_from((1i64 - i64::from(exponent)).max(0))?;
            let needed = 64u64 + u64::from(b) * u64::from(pivot_bits + log_b);
            if u64::from(guard) >= needed {
                return Ok(result);
            }
        }
        result.leakage = None;
        result.minimum_pivot = None;
        last = Some(result);
    }
    Ok(last.unwrap())
}
