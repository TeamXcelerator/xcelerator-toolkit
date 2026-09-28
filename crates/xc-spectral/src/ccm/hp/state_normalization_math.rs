//! Correct rounding of xi_i * sqrt(L) / sum(xi) for exact stored inputs.
//! This arithmetic does not certify the source eigenstate or its construction.
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) const ARITHMETIC: &str = "directed_scaled_exact_sum_sqrt_l_state_v2";

pub(super) fn evaluate(xi: &[Float], l: &Float, p: u32) -> Result<Vec<Float>> {
    if !(64..=1_000_000).contains(&p)
        || xi.is_empty()
        || xi.len() > 16385
        || !l.is_finite()
        || l <= &0
        || l.prec() > p
        || xi.iter().any(|v| !v.is_finite() || v.prec() > p)
        || xi.len() as u128 * (u128::from(p + 4096).div_ceil(8) + 96) * 12 > (8u128 << 30)
    {
        bail!("invalid eigenstate normalization source, precision, shape or workspace");
    }
    let vector_exponent = xi
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero eigenstate cannot be normalized"))?;
    let l_exponent = i64::from(l.get_exp().unwrap()).div_euclid(2) * 2;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p + guard;
        let normalized = xi
            .iter()
            .map(|v| scale_float(v, -i64::from(vector_exponent), work))
            .collect::<Result<Vec<_>>>()?;
        // MPFR's sum encloses the exact sum once, preserving cancellation and
        // avoiding dependence on input order or overflowing partial sums.
        let sum = I::new(
            Float::with_val_round(work, Float::sum(normalized.iter()), Round::Down).0,
            Float::with_val_round(work, Float::sum(normalized.iter()), Round::Up).0,
        )?;
        if sum.lower().is_zero() && sum.upper().is_zero() {
            bail!("eigenstate has zero normalization sum");
        }
        if sum.contains_zero() {
            continue;
        }
        let target = I::point(scale_float(l, -l_exponent, work)?).sqrt()?;
        let mut out = Vec::with_capacity(xi.len());
        for value in &normalized {
            let result = scale(
                &I::point(value.clone()).mul(&target).div(&sum)?,
                l_exponent / 2,
            )?;
            let lo = Float::with_val(p, result.lower());
            let hi = Float::with_val(p, result.upper());
            if lo != hi || !lo.is_finite() || (lo.is_zero() && !value.is_zero()) {
                break;
            }
            out.push(lo);
        }
        if out.len() == xi.len() {
            return Ok(out);
        }
    }
    bail!("eigenstate normalization unresolved within 4096 guard bits")
}
