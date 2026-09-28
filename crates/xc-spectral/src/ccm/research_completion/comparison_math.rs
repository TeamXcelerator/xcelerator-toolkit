//! finite comparison expressions on original stored source points.
use crate::ccm::{
    convergence_capture::ComparisonState,
    retained_evidence::{
        finite_math::{abs, decimal, narrow, scale, scale_float},
        scalar, RetainedMatrix,
    },
    state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;
pub(super) struct Measurement {
    pub values: BTreeMap<String, I>,
    pub precision: u32,
    pub zero_comparison: bool,
}
fn vector(x: &[Float], p: u32) -> Result<Option<Vec<I>>> {
    let Some(e) = x.iter().filter_map(Float::get_exp).max() else {
        return Ok(None);
    };
    x.iter()
        .map(|x| Ok(I::from_float(&scale_float(x, -i64::from(e), p)?, p)?))
        .collect::<Result<_>>()
        .map(Some)
}
fn norm2(x: &[I], p: u32) -> I {
    x.iter().fold(I::from_i64(0, p), |a, x| a.add(&x.square()))
}
fn is_narrow(x: &I, requested: u32) -> Result<bool> {
    x.validate()?;
    let natural = if x.contains_zero() {
        Float::with_val(x.precision(), 1)
    } else {
        x.lower().clone().abs().max(&x.upper().clone().abs())
    };
    narrow(std::slice::from_ref(x), &natural, requested)
}
fn calculate(
    s: &RetainedState,
    a: &ComparisonState,
    m: Option<&RetainedMatrix<'_>>,
    same_basis: bool,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let eigen = scalar(&a.eigenvalue, a.precision_bits)?;
    let ei = I::from_float(&eigen, p)?;
    let mut values = BTreeMap::from([
        ("comparison_C".into(), decimal(&a.lambda_squared, p)?),
        ("comparison_N".into(), I::from_u64(a.n_modes as u64, p)),
        (
            "comparison_P".into(),
            I::from_u64(a.precision_bits.into(), p),
        ),
        (
            "signed_energy_difference".into(),
            I::from_float(&scalar(&s.eigenvalue, s.precision)?, p)?.sub(&ei),
        ),
    ]);
    let raw = a
        .coefficients
        .iter()
        .map(|v| scalar(v, a.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let y = vector(&raw, p)?;
    let zero_comparison = y.is_none();
    if same_basis && a.n_modes <= s.modes {
        if let Some(y) = y {
            let x = vector(&s.coefficients, p)?
                .ok_or_else(|| anyhow::anyhow!("zero retained comparison source"))?;
            let qx = norm2(&x, p);
            let qy = norm2(&y, p);
            let offset = s.modes - a.n_modes;
            let mut dot = I::from_i64(0, p);
            for (x, y) in x[offset..offset + y.len()].iter().zip(&y) {
                dot = dot.add(&x.mul(y));
            }
            values.insert(
                "absolute_unit_overlap".into(),
                abs(&dot)?.div(&qx.mul(&qy).sqrt()?)?,
            );
            if let Some(m) = m {
                let n = x.len();
                let shifted = |i: usize, j: usize| -> Result<I> {
                    let entry = I::from_float(&m.entries[i * n + j], p)?;
                    Ok(if i == j { entry.sub(&ei) } else { entry })
                };
                // Remove the common eigenvalue shift before selecting an exponent. This
                // preserves small off-diagonal forcing next to an exactly canceled B*I.
                // Two passes avoid allocating a second full interval matrix.
                let mut exponent = 0i64;
                let mut present = false;
                for i in 0..n {
                    for (k, original) in raw.iter().enumerate() {
                        if original.is_zero() {
                            continue;
                        }
                        let entry = shifted(i, offset + k)?;
                        entry.validate()?;
                        for e in [entry.lower(), entry.upper()]
                            .iter()
                            .filter_map(|v| v.get_exp())
                        {
                            exponent = if present {
                                exponent.max(i64::from(e))
                            } else {
                                i64::from(e)
                            };
                            present = true;
                        }
                    }
                }
                let mut low = I::from_i64(0, p);
                let mut high = low.clone();
                for i in 0..n {
                    let mut action = I::from_i64(0, p);
                    for (k, (y, original)) in y.iter().zip(&raw).enumerate() {
                        if original.is_zero() {
                            continue;
                        }
                        action = action.add(&scale(&shifted(i, offset + k)?, -exponent)?.mul(y));
                    }
                    if (offset..offset + y.len()).contains(&i) {
                        low = low.add(&action.square());
                    } else {
                        high = high.add(&action.square());
                    }
                }
                values.insert(
                    "independent_small_state_low_residual_squared".into(),
                    scale(&low.div(&qy)?, 2 * exponent)?,
                );
                values.insert(
                    "independent_small_state_high_forcing_squared".into(),
                    scale(&high.div(&qy)?, 2 * exponent)?,
                );
            }
        }
    }
    for value in values.values() {
        if !is_narrow(value, requested)? {
            return Ok(None);
        }
    }
    Ok(Some(Measurement {
        values,
        precision: p,
        zero_comparison,
    }))
}
pub(super) fn measure_at(
    s: &RetainedState,
    a: &ComparisonState,
    m: Option<&RetainedMatrix<'_>>,
    same_basis: bool,
    requested: u32,
    guard: u32,
) -> Result<Option<Measurement>> {
    crate::ccm::retained_evidence::precision(requested)?;
    if requested < s.precision.max(a.precision_bits)
        || ![64, 128, 256, 512, 1024, 2048, 4096].contains(&guard)
    {
        bail!("comparison precision/guard below stored input contract");
    }
    calculate(s, a, m, same_basis, requested, requested + guard)
}
