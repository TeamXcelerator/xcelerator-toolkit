//! Directed arithmetic for a finite stored-point projected resolvent.
//! Response ratios remain conditional; root proximity is only a stated rule.
use super::{
    coeffs,
    energy_math::{dot, norm2, stored, vector, Operator},
    scalar, ExternalResearchInputs, RetainedMatrix, RetainedState,
};
use crate::ccm::retained_evidence::{
    finite_math::{abs, decimal, narrow, scale},
    point,
};
use anyhow::{bail, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) struct Measurement {
    pub values: BTreeMap<String, I>,
    pub precision: u32,
    pub root_rule_met: bool,
    pub denominator_resolved: bool,
    pub has_actions: bool,
    pub displacement_rule_met: bool,
}
pub(super) enum Outcome {
    Measured(Measurement),
    Carrier,
    GuardExhausted,
}

#[allow(clippy::too_many_arguments)] // Distinguish original points from work precision.
fn calculate(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    t: &Float,
    eigen: &Float,
    operators: &[(String, Operator)],
    compact: &[(String, Vec<Float>)],
    e: i64,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let x =
        vector(&s.coefficients, p)?.ok_or_else(|| anyhow::anyhow!("zero directional source"))?;
    let q = norm2(&x, p);
    let q2 = q.square();
    let q3 = q2.mul(&q);
    let sqrtq = q.sqrt()?;
    let sign = I::from_i64(i64::from(point::orientation(&s.coefficients, p)), p);
    let ti = I::from_float(t, p)?;
    // C is the exact decimal model parameter, while t is a stored source point.
    let tau = ti
        .mul(&decimal(&s.cutoff, p)?.ln()?)
        .div(&I::pi(p).mul(&I::from_i64(2, p)))?
        .square();
    let mut u = Vec::with_capacity(x.len());
    for (idx, a) in x.iter().enumerate() {
        let j = u64::try_from(idx.abs_diff(s.modes))?;
        let denominator = tau.sub(&I::from_u64(
            j.checked_mul(j)
                .ok_or_else(|| anyhow::anyhow!("mode square overflow"))?,
            p,
        ));
        if denominator.contains_zero() {
            return Ok(None);
        }
        u.push(a.div(&denominator)?);
    }
    let h = dot(&x, &u, p);
    let z = u
        .iter()
        .zip(&x)
        .map(|(u, x)| q.mul(u).sub(&x.mul(&h)))
        .collect::<Vec<_>>();
    // Build A-EI before applying it: scalar shifts cancel at the entry level.
    let ei = stored(eigen, -e, p)?;
    let mut az = Vec::with_capacity(z.len());
    for (i, row) in m.entries.chunks_exact(z.len()).enumerate() {
        let mut total = I::from_i64(0, p);
        for (j, (a, z)) in row.iter().zip(&z).enumerate() {
            let mut a = stored(a, -e, p)?;
            if i == j {
                a = a.sub(&ei);
            }
            total = total.add(&a.mul(z));
        }
        az.push(total);
    }
    let k = scale(&dot(&z, &az, p).div(&q3)?, e)?;
    let direction2 = norm2(&z, p).div(&q3)?;
    let sum = u.iter().fold(I::from_i64(0, p), |a, b| a.add(b));
    let absolute = u.iter().try_fold(I::from_i64(0, p), |a, b| {
        Ok::<_, anyhow::Error>(a.add(&abs(b)?))
    })?;
    let root = sum.mul(&sign).div(&sqrtq)?;
    let root_scale = absolute.div(&sqrtq)?;
    let tolerance = scale(
        &root_scale.add(&I::from_i64(1, p)),
        -i64::from(s.precision.saturating_sub(32)),
    )?;
    let root_rule_met = abs(&root)?.upper() <= tolerance.lower();
    let denominator_resolved = !k.contains_zero();
    // z=Q P R x and w=Q P R 1. The root-velocity identity requires
    // (A-E)z parallel to w; a root and a simple minimum alone do not imply it.
    // Both az and its projection retain the same matrix power-of-two scaling.
    let w = x
        .iter()
        .enumerate()
        .map(|(idx, x)| {
            let j = idx.abs_diff(s.modes) as u64;
            Ok(q.div(&tau.sub(&I::from_u64(j * j, p)))?.sub(&x.mul(&sum)))
        })
        .collect::<Result<Vec<_>>>()?;
    let az2 = norm2(&az, p);
    let w2 = norm2(&w, p);
    let displacement_tolerance = scale(&I::from_i64(1, p), -i64::from(s.precision / 2))?;
    let displacement_defect =
        if az2.lower() > &Float::with_val(p, 0) && w2.lower() > &Float::with_val(p, 0) {
            let c = dot(&w, &az, p).div(&w2)?;
            let difference = az
                .iter()
                .zip(&w)
                .map(|(a, w)| a.sub(&c.mul(w)))
                .collect::<Vec<_>>();
            Some(norm2(&difference, p).div(&az2)?.sqrt()?)
        } else {
            None
        };
    let displacement_rule_met = displacement_defect
        .as_ref()
        .is_some_and(|d| d.upper() <= displacement_tolerance.lower());
    let mut values = BTreeMap::new();
    for (name, value) in [
        ("t", ti),
        ("tau", tau),
        ("directional_energy", k.clone()),
        ("direction_norm_squared", direction2),
        // Exact identity z^T x=Q*x^T R x-(x^T x)*(x^T R x)=0.
        ("orthogonality_defect", I::from_i64(0, p)),
        ("rational_root_condition", root),
        ("root_condition_tolerance", tolerance),
        ("displacement_defect_tolerance", displacement_tolerance),
    ] {
        values.insert(name.to_string(), value);
    }
    if let Some(defect) = displacement_defect {
        values.insert("displacement_defect".into(), defect);
    }
    for (label, op) in operators {
        let action = op.action(&x, e, p)?;
        let forcing = scale(&dot(&z, &action, p).div(&q2)?, e)?;
        if denominator_resolved && root_rule_met && displacement_rule_met {
            values.insert(
                format!("conditional_tau_response_{label}"),
                forcing.neg().div(&k)?,
            );
        }
        values.insert(format!("forcing_{label}"), forcing);
    }
    for (label, action) in compact {
        let action = action
            .iter()
            .map(|a| stored(a, -e, p))
            .collect::<Result<Vec<_>>>()?;
        let forcing = scale(&dot(&z, &action, p).mul(&sign).div(&q.mul(&sqrtq))?, e)?;
        if denominator_resolved && root_rule_met && displacement_rule_met {
            values.insert(
                format!("conditional_tau_response_{label}"),
                forcing.neg().div(&k)?,
            );
        }
        values.insert(format!("forcing_{label}"), forcing);
    }
    for value in values.values() {
        value.validate()?;
        let natural = if value.contains_zero() {
            Float::with_val(p, 1)
        } else {
            value
                .lower()
                .clone()
                .abs()
                .max(&value.upper().clone().abs())
        };
        if !narrow(std::slice::from_ref(value), &natural, requested)? {
            return Ok(None);
        }
    }
    Ok(Some(Measurement {
        values,
        precision: p,
        root_rule_met,
        denominator_resolved,
        displacement_rule_met,
        has_actions: !operators.is_empty() || !compact.is_empty(),
    }))
}

pub(super) fn measure(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    input: Option<&ExternalResearchInputs>,
    root: &str,
    root_precision: u32,
    requested: u32,
) -> Result<Outcome> {
    if requested
        < s.precision
            .max(m.precision)
            .max(root_precision)
            .max(input.map_or(64, |i| i.precision_bits))
    {
        bail!("directional source precision reduction");
    }
    let t = scalar(root, root_precision)?;
    if t.is_zero() {
        return Ok(Outcome::Carrier);
    }
    let eigen = scalar(&s.eigenvalue, s.precision)?;
    let mut operators = Vec::new();
    let mut compact = Vec::new();
    if let Some(i) = input {
        for op in &i.perturbations {
            operators.push((op.label.clone(), Operator::parse(op, i.precision_bits)?));
        }
        if let Some(run) = &i.run_once {
            for a in &run.derivative_actions {
                if operators.iter().any(|(label, _)| label == &a.label)
                    || compact.iter().any(|(label, _)| label == &a.label)
                {
                    bail!("duplicate derivative action label");
                }
                compact.push((a.label.clone(), coeffs(&a.action, i.precision_bits)?));
            }
        }
    }
    let e = m
        .entries
        .iter()
        .chain(std::iter::once(&eigen))
        .filter_map(Float::get_exp)
        .map(i64::from)
        .chain(operators.iter().map(|(_, op)| op.exponent()))
        .chain(
            compact
                .iter()
                .flat_map(|(_, a)| a)
                .filter_map(Float::get_exp)
                .map(i64::from),
        )
        .max()
        .unwrap_or(0);
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(value) = calculate(
            s,
            m,
            &t,
            &eigen,
            &operators,
            &compact,
            e,
            requested,
            requested + guard,
        )? {
            return Ok(Outcome::Measured(value));
        }
    }
    Ok(Outcome::GuardExhausted)
}
