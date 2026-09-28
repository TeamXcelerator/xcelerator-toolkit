//! Finite stored-point quadratic diagnostics with directed arithmetic.
//! Excludes source construction, operator modeling error and ground selection.
use super::{
    coeffs, scalar, ExternalResearchInputs, OperatorComponent, RetainedMatrix, RetainedState,
};
use crate::ccm::retained_evidence::{
    finite_math::{abs, narrow, scale, scale_float},
    point,
};
use anyhow::{bail, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) struct Measurement {
    pub values: BTreeMap<String, I>,
    pub energies: Vec<I>,
    pub arithmetic_precision: u32,
    pub zero_trial: bool,
}
pub(super) struct Operator {
    diagonal: Vec<Float>,
    dense: Vec<Float>,
    rank: Vec<(Float, Vec<Float>)>,
}
impl Operator {
    pub(super) fn parse(c: &OperatorComponent, p: u32) -> Result<Self> {
        Ok(Self {
            diagonal: coeffs(&c.diagonal, p)?,
            dense: coeffs(&c.dense, p)?,
            rank: c
                .rank_one
                .iter()
                .map(|r| Ok((scalar(&r.weight, p)?, coeffs(&r.vector, p)?)))
                .collect::<Result<_>>()?,
        })
    }
    pub(super) fn exponent(&self) -> i64 {
        let direct = self
            .diagonal
            .iter()
            .chain(&self.dense)
            .filter_map(Float::get_exp)
            .map(i64::from);
        let rank = self.rank.iter().filter_map(|(w, q)| {
            Some(
                i64::from(w.get_exp()?) + 2 * i64::from(q.iter().filter_map(Float::get_exp).max()?),
            )
        });
        direct.chain(rank).max().unwrap_or(0)
    }
    pub(super) fn action(&self, x: &[I], e: i64, p: u32) -> Result<Vec<I>> {
        let n = x.len();
        let mut out = vec![I::from_i64(0, p); n];
        if !self.dense.is_empty() {
            for (row, value) in self.dense.chunks_exact(n).zip(&mut out) {
                for (a, v) in row.iter().zip(x) {
                    *value = value.add(&stored(a, -e, p)?.mul(v));
                }
            }
        }
        for ((value, a), v) in out.iter_mut().zip(&self.diagonal).zip(x) {
            *value = value.add(&stored(a, -e, p)?.mul(v));
        }
        for (w, q) in &self.rank {
            let qe = i64::from(q.iter().filter_map(Float::get_exp).max().unwrap_or(0));
            let q = q
                .iter()
                .map(|a| stored(a, -qe, p))
                .collect::<Result<Vec<_>>>()?;
            let factor = dot(&q, x, p).mul(&stored(w, 2 * qe - e, p)?);
            for (value, q) in out.iter_mut().zip(q) {
                *value = value.add(&q.mul(&factor));
            }
        }
        Ok(out)
    }
}
pub(super) fn stored(x: &Float, e: i64, p: u32) -> Result<I> {
    Ok(I::from_float(&scale_float(x, e, p)?, p)?)
}
pub(super) fn vector(x: &[Float], p: u32) -> Result<Option<Vec<I>>> {
    let Some(e) = x.iter().filter_map(Float::get_exp).max() else {
        return Ok(None);
    };
    x.iter()
        .map(|x| stored(x, -i64::from(e), p))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}
pub(super) fn dot(x: &[I], y: &[I], p: u32) -> I {
    x.iter()
        .zip(y)
        .fold(I::from_i64(0, p), |a, (x, y)| a.add(&x.mul(y)))
}
pub(super) fn norm2(x: &[I], p: u32) -> I {
    x.iter().fold(I::from_i64(0, p), |a, x| a.add(&x.square()))
}
fn dense_action(a: &[Float], x: &[I], e: i64, p: u32) -> Result<Vec<I>> {
    a.chunks_exact(x.len())
        .map(|row| {
            let mut result = I::from_i64(0, p);
            for (a, x) in row.iter().zip(x) {
                result = result.add(&stored(a, -e, p)?.mul(x));
            }
            Ok(result)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)] // Keep stored input families and output/guard precision explicit.
fn calculate(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    input: &ExternalResearchInputs,
    operators: &[Operator],
    compact: &[Vec<Float>],
    trial: Option<&[Float]>,
    e: i64,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let x = vector(&s.coefficients, p)?.ok_or_else(|| anyhow::anyhow!("zero energy source"))?;
    let x2 = norm2(&x, p);
    let norm = x2.sqrt()?;
    let ax = dense_action(&m.entries, &x, e, p)?;
    let total = dot(&x, &ax, p).div(&x2)?;
    let mut energies = Vec::new();
    let mut combined = vec![I::from_i64(0, p); x.len()];
    let residual;
    let sum;
    let closure;
    if !operators.is_empty() {
        for op in operators {
            let a = op.action(&x, e, p)?;
            energies.push(dot(&x, &a, p).div(&x2)?);
            for (sum, a) in combined.iter_mut().zip(a) {
                *sum = sum.add(&a);
            }
        }
        let delta = combined
            .iter()
            .zip(&ax)
            .map(|(a, b)| a.sub(b))
            .collect::<Vec<_>>();
        residual = norm2(&delta, p).sqrt()?.div(&norm)?;
        sum = dot(&x, &combined, p).div(&x2)?;
        closure = dot(&x, &delta, p).div(&x2)?;
    } else {
        let sign = I::from_i64(i64::from(point::orientation(&s.coefficients, p)), p);
        for a in compact {
            let a = a
                .iter()
                .map(|a| stored(a, -e, p))
                .collect::<Result<Vec<_>>>()?;
            energies.push(dot(&x, &a, p).mul(&sign).div(&norm)?);
            for (sum, a) in combined.iter_mut().zip(a) {
                *sum = sum.add(&a);
            }
        }
        let delta = combined
            .iter()
            .zip(&ax)
            .map(|(a, b)| Ok(a.sub(&b.mul(&sign).div(&norm)?)))
            .collect::<Result<Vec<_>>>()?;
        residual = norm2(&delta, p).sqrt()?;
        sum = dot(&x, &combined, p).mul(&sign).div(&norm)?;
        closure = dot(&x, &delta, p).mul(&sign).div(&norm)?;
    }
    let mut absolute = I::from_i64(0, p);
    for energy in &energies {
        absolute = absolute.add(&abs(energy)?);
    }
    let mut values = BTreeMap::new();
    for (name, value) in [
        ("total_tau_energy", total.clone()),
        ("sum_component_energy", sum.clone()),
        ("sum_absolute_component_energy", absolute),
        ("energy_closure_defect", closure),
        ("operator_action_closure_norm", residual),
    ] {
        values.insert(name.into(), scale(&value, e)?);
    }
    let eigen = I::from_float(&scalar(&s.eigenvalue, s.precision)?, p)?;
    values.insert("retained_weil_energy".into(), eigen.clone());
    let mut zero_trial = false;
    if let Some(trial) = trial {
        if let Some(q) = vector(trial, p)? {
            let aq = dense_action(&m.entries, &q, e, p)?;
            values.insert(
                "finite_projected_trial_energy".into(),
                scale(&dot(&q, &aq, p).div(&norm2(&q, p))?, e)?,
            );
        } else {
            zero_trial = true;
        }
    }
    if let Some(d) = &input.deficit {
        let d = I::from_float(&scalar(d, input.precision_bits)?, p)?;
        values.insert("reference_deficit".into(), d.clone());
        if d.is_strictly_positive() {
            values.insert("signed_weil_over_deficit".into(), eigen.div(&d)?);
            if let Some(trial) = values.get("finite_projected_trial_energy").cloned() {
                values.insert("signed_trial_over_deficit".into(), trial.div(&d)?);
            }
        }
    }
    if !eigen.contains_zero() {
        if let Some(trial) = values.get("finite_projected_trial_energy").cloned() {
            values.insert("signed_trial_over_weil".into(), trial.div(&eigen)?);
        }
    }
    let energies = energies
        .iter()
        .map(|x| scale(x, e))
        .collect::<Result<Vec<_>>>()?;
    for value in values.values().chain(&energies) {
        value.validate()?;
        let magnitude = value
            .lower()
            .clone()
            .abs()
            .max(&value.upper().clone().abs());
        // Relative width for resolved nonzero values; absolute width around zero.
        let natural = if value.contains_zero() {
            Float::with_val(p, 1)
        } else {
            magnitude
        };
        if !narrow(std::slice::from_ref(value), &natural, requested)? {
            return Ok(None);
        }
    }
    Ok(Some(Measurement {
        values,
        energies,
        arithmetic_precision: p,
        zero_trial,
    }))
}

pub(super) fn measure(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    input: &ExternalResearchInputs,
    requested: u32,
) -> Result<Option<Measurement>> {
    if requested < s.precision.max(m.precision).max(input.precision_bits) {
        bail!("energy source precision reduction")
    }
    let operators = input
        .components
        .iter()
        .map(|c| Operator::parse(c, input.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let compact = if operators.is_empty() {
        input
            .run_once
            .as_ref()
            .map(|r| {
                r.component_actions
                    .iter()
                    .map(|c| coeffs(&c.action, input.precision_bits))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let trial = input
        .target
        .as_ref()
        .and_then(|t| t.trial_coefficients.as_ref())
        .map(|q| coeffs(q, input.precision_bits))
        .transpose()?;
    let e = m
        .entries
        .iter()
        .filter_map(Float::get_exp)
        .map(i64::from)
        .chain(operators.iter().map(Operator::exponent))
        .chain(
            compact
                .iter()
                .flatten()
                .filter_map(Float::get_exp)
                .map(i64::from),
        )
        .max()
        .unwrap_or(0);
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(result) = calculate(
            s,
            m,
            input,
            &operators,
            &compact,
            trial.as_deref(),
            e,
            requested,
            requested + guard,
        )? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}
