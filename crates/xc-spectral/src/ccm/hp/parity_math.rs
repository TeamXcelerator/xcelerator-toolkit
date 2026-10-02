//! Correct rounding for finite orthonormal reflection-basis transforms.
//! Only the exact stored source points are enclosed; no assembly error claim.
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) const ARITHMETIC: &str = "directed_stored_point_orthonormal_parity_v2";

/// Round 2^binary_scale sum(sign_i value_i) / (sqrt(2) if requested).
pub(super) fn linear(
    terms: &[(&Float, i32)],
    binary_scale: i64,
    sqrt_divisor: bool,
    p: u32,
) -> Result<Float> {
    if !(rug::float::prec_min()..=1_000_000).contains(&p)
        || terms.is_empty()
        || terms.len() > 4
        || !(-1..=0).contains(&binary_scale)
        || terms
            .iter()
            .any(|(v, sign)| !v.is_finite() || v.prec() > 1_000_000 || ![-1, 1].contains(sign))
    {
        bail!("invalid parity transform source, precision or shape");
    }
    let source_precision = terms.iter().map(|(v, _)| v.prec()).max().unwrap().max(p);
    let e = terms
        .iter()
        .filter_map(|(v, _)| v.get_exp())
        .max()
        .map_or(0, i64::from);
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = source_precision + guard;
        let values = terms
            .iter()
            .map(|(v, sign)| {
                let value = scale_float(v, -e, work)?;
                Ok(if *sign < 0 { -value } else { value })
            })
            .collect::<Result<Vec<_>>>()?;
        let sum = I::new(
            Float::with_val_round(work, Float::sum(values.iter()), Round::Down).0,
            Float::with_val_round(work, Float::sum(values.iter()), Round::Up).0,
        )?;
        let quotient = if sqrt_divisor {
            sum.div(&I::from_i64(2, work).sqrt()?)?
        } else {
            sum
        };
        let output = scale(&quotient, e + binary_scale)?;
        let lo = Float::with_val(p, output.lower());
        let hi = Float::with_val(p, output.upper());
        if lo == hi
            && lo.is_finite()
            && (!lo.is_zero() || (output.lower().is_zero() && output.upper().is_zero()))
        {
            return Ok(lo);
        }
    }
    bail!("parity transform rounding unresolved within 4096 guard bits")
}

pub(super) struct Projection<'a> {
    tau: &'a [Float],
    n: usize,
    full: usize,
    p: u32,
    odd: bool,
}
impl<'a> Projection<'a> {
    pub(super) fn new(tau: &'a [Float], n: usize, p: u32, odd: bool) -> Result<Self> {
        let full = n
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("parity source dimension overflow"))?;
        if !(64..=1_000_000).contains(&p)
            || full.checked_mul(full) != Some(tau.len())
            || !super::matrix_is_exactly_symmetric(tau, full)
            || tau.iter().any(|v| v.prec() > 1_000_000)
        {
            bail!("parity projection requires a finite symmetric source and supported precision");
        }
        let maximum_precision = tau.iter().map(Float::prec).max().unwrap().max(p);
        if tau.len() as u128 * (u128::from(maximum_precision + 4096).div_ceil(8) + 96) * 2
            > super::source_working_budget()?
        {
            bail!("parity projection exceeds numerical workspace budget");
        }
        Ok(Self {
            tau,
            n,
            full,
            p,
            odd,
        })
    }
    pub(super) fn dimension(&self) -> usize {
        self.n + usize::from(!self.odd)
    }
    pub(super) fn entry(&self, i: usize, j: usize) -> Result<Float> {
        let d = self.dimension();
        if i >= d || j >= d {
            bail!("parity entry index is out of range");
        }
        let at = |row: usize, col: usize| &self.tau[row * self.full + col];
        let n = self.n;
        if !self.odd {
            if i == 0 && j == 0 {
                return linear(&[(at(n, n), 1)], 0, false, self.p);
            }
            if i == 0 || j == 0 {
                let k = i.max(j);
                return linear(&[(at(n, n - k), 1), (at(n, n + k), 1)], 0, true, self.p);
            }
        }
        let k = i + usize::from(self.odd);
        let j = j + usize::from(self.odd);
        let sign = if self.odd { -1 } else { 1 };
        linear(
            &[
                (at(n - k, n - j), 1),
                (at(n - k, n + j), sign),
                (at(n + k, n - j), sign),
                (at(n + k, n + j), 1),
            ],
            -1,
            false,
            self.p,
        )
    }
    pub(super) fn matrix(&self) -> Result<Vec<Float>> {
        let d = self.dimension();
        let mut out = vec![Float::with_val(self.p, 0); d * d];
        for i in 0..d {
            for j in i..d {
                let value = self.entry(i, j)?;
                out[i * d + j] = value.clone();
                out[j * d + i] = value;
            }
        }
        Ok(out)
    }
}
