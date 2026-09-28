//! Conditional finite block expressions, with directed arithmetic only.
//! The caller's block/hypothesis claims are not authenticated here.
use super::{scalar, EnergyAllowance};
use crate::ccm::retained_evidence::finite_math::{narrow, scale, scale_float};
use anyhow::Result;
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;
pub(super) struct Measurement {
    pub values: BTreeMap<String, I>,
    pub precision: u32,
    pub valid_margin: bool,
    pub zero_trial: bool,
    pub below: Option<bool>,
}
fn stored(x: &Float, e: i64, p: u32) -> Result<I> {
    Ok(I::from_float(&scale_float(x, e, p)?, p)?)
}
fn calculate(
    points: &[Float; 4],
    requested: u32,
    p: u32,
    last: bool,
) -> Result<Option<Measurement>> {
    let [u, b, mu, h] = points;
    let de = u
        .get_exp()
        .into_iter()
        .chain(mu.get_exp())
        .map(i64::from)
        .max()
        .unwrap_or(0);
    let us = stored(u, -de, p)?;
    let mus = stored(mu, -de, p)?;
    let d = mus.sub(&us);
    let valid_margin = mu > u;
    let zero_trial = u.is_zero();
    let mut below = None;
    let mut values = BTreeMap::new();
    for (name, value) in [
        ("upper_trial_energy", stored(u, 0, p)?),
        ("low_block_lower_bound", stored(b, 0, p)?),
        ("high_block_lower_bound", stored(mu, 0, p)?),
        ("cross_block_norm_bound", stored(h, 0, p)?),
        ("denominator", scale(&d, de)?),
    ] {
        values.insert(name.to_string(), value);
    }
    if valid_margin {
        if !d.is_strictly_positive() {
            return Ok(None);
        }
        let he = i64::from(h.get_exp().unwrap_or(0));
        let hs = stored(h, -he, p)?;
        let energy_base = hs.square().div(&d)?;
        let energy = scale(&energy_base, 2 * he - de)?;
        let vector = scale(&hs.div(&d)?, he - de)?;
        let magnitude = u.clone().abs();
        let magnitude_i = stored(&magnitude, 0, p)?;
        if !zero_trial {
            let ue = i64::from(magnitude.get_exp().unwrap());
            let ratio = scale(
                &energy_base.div(&stored(&magnitude, -ue, p)?)?,
                2 * he - de - ue,
            )?;
            if energy.upper() < magnitude_i.lower() {
                below = Some(true);
            } else if energy.lower() >= magnitude_i.upper() {
                below = Some(false);
            } else if !last {
                return Ok(None);
            }
            values.insert("allowance_to_trial_energy_magnitude".into(), ratio);
            // 1 certifies the strict arithmetic comparison for stored points;
            // 0 also covers an explicitly unresolved comparison, never a claim
            // of equality/greater-than unless scale_comparison_resolved is 1.
            values.insert(
                "allowance_below_trial_energy_magnitude".into(),
                I::from_i64(i64::from(below == Some(true)), p),
            );
            values.insert(
                "scale_comparison_resolved".into(),
                I::from_i64(i64::from(below.is_some()), p),
            );
        }
        values.insert("trial_energy_magnitude".into(), magnitude_i);
        values.insert("conditional_energy_allowance".into(), energy);
        values.insert("conditional_vector_allowance".into(), vector);
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
        valid_margin,
        zero_trial,
        below,
    }))
}
pub(super) fn measure(
    a: &EnergyAllowance,
    input_precision: u32,
    requested: u32,
) -> Result<Option<Measurement>> {
    let points = [
        scalar(&a.upper_trial_energy, input_precision)?,
        scalar(&a.low_block_lower_bound, input_precision)?,
        scalar(&a.high_block_lower_bound, input_precision)?,
        scalar(&a.cross_block_norm_bound, input_precision)?,
    ];
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(value) = calculate(&points, requested, requested + guard, guard == 4096)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}
