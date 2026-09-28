//! conditional finite-support L2 transport with directed arithmetic.
use super::{
    ExtendedAnalysis, ExtensionOptions, ExternalResearchInputs, RetainedRoots, RetainedState,
};
use crate::ccm::{
    extended_research::{put, report, row, save_arithmetic_enclosure, unresolved},
    retained_evidence::{
        finite_math::{abs, decimal, narrow, scale},
        scalar, transform_math,
    },
};
use anyhow::{bail, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;
struct Measurement {
    values: BTreeMap<String, I>,
    precision: u32,
}
fn narrow_enough(value: &I, p: u32) -> Result<bool> {
    value.validate()?;
    let natural = if value.contains_zero() {
        Float::with_val(value.precision(), 1)
    } else {
        value
            .lower()
            .clone()
            .abs()
            .max(&value.upper().clone().abs())
    };
    narrow(std::slice::from_ref(value), &natural, p)
}
fn measure(
    s: &RetainedState,
    t: &Float,
    error: Option<&Float>,
    requested: u32,
    origin: bool,
) -> Result<Option<Measurement>> {
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let Some(m) = transform_math::measure_at_root(s, t, requested, guard)? else {
            continue;
        };
        let p = m.precision;
        let mut values = if origin {
            BTreeMap::from([
                ("transform_origin".into(), m.value.clone()),
                ("origin_absolute_terms".into(), m.absolute_terms.clone()),
            ])
        } else {
            BTreeMap::from([
                ("t".into(), I::from_float(t, p)?),
                ("value".into(), m.value.clone()),
                ("derivative".into(), m.derivative.clone()),
                ("absolute_value_terms".into(), m.absolute_terms.clone()),
                (
                    "absolute_derivative_terms".into(),
                    m.absolute_derivative_terms.clone(),
                ),
            ])
        };
        if let Some(error) = error {
            let error = I::from_float(error, p)?;
            if error.lower() < &0 {
                bail!("negative source L2 error");
            }
            let l = decimal(&s.cutoff, p)?.ln()?;
            if !l.is_strictly_positive() {
                continue;
            }
            let e = i64::from(
                l.upper()
                    .get_exp()
                    .ok_or_else(|| anyhow::anyhow!("nonpositive observation support"))?,
            )
            .div_euclid(2)
                * 2;
            let reduced = scale(&l, -e)?;
            let root = reduced.sqrt()?;
            let value_norm = scale(&root, e / 2)?;
            let derivative_norm = scale(
                &reduced.mul(&root).div(&I::from_i64(12, p).sqrt()?)?,
                3 * (e / 2),
            )?;
            let value_error = error.mul(&value_norm);
            let derivative_error = error.mul(&derivative_norm);
            values.insert("declared_unit_state_l2_error".into(), error);
            if origin {
                values.insert("declared_origin_error".into(), value_error.clone());
                values.insert(
                    "conditional_origin_lower_margin".into(),
                    abs(&m.value)?.sub(&value_error),
                );
            } else {
                values.insert("conditional_value_error".into(), value_error);
                values.insert(
                    "conditional_derivative_error".into(),
                    derivative_error.clone(),
                );
                values.insert(
                    "conditional_slope_lower_margin".into(),
                    abs(&m.derivative)?.sub(&derivative_error),
                );
            }
        }
        let mut accepted = true;
        for value in values.values() {
            if !narrow_enough(value, requested)? {
                accepted = false;
                break;
            }
        }
        if accepted {
            return Ok(Some(Measurement {
                values,
                precision: p,
            }));
        }
    }
    Ok(None)
}
fn save(map: &mut BTreeMap<String, String>, measured: Measurement, p: u32) -> Result<()> {
    for (name, value) in measured.values {
        save_arithmetic_enclosure(map, &name, &value, p)?;
        let side = if name.ends_with("_lower_margin") {
            Some("lower")
        } else if [
            "declared_origin_error",
            "declared_unit_state_l2_error",
            "conditional_value_error",
            "conditional_derivative_error",
        ]
        .contains(&name.as_str())
        {
            Some("upper")
        } else {
            None
        };
        if let Some(side) = side {
            map.insert(name.clone(), map[&format!("{name}_{side}")].clone());
        }
    }
    put(
        map,
        "arithmetic_precision_bits",
        &Float::with_val(p, measured.precision),
    );
    Ok(())
}
pub(super) fn analyze(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("observable_budget", s, o);
    let p = o.working_precision_bits;
    let scratch = (8 * s.coefficients.len() as u64 + 256) * (u64::from(p + 4096).div_ceil(8) + 64);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            out,
            "observation maximum-guard scratch exceeds explicit working-byte budget",
        ));
    }
    let uncertainty = input
        .and_then(|i| i.run_once.as_ref())
        .and_then(|i| i.uncertainty.as_ref());
    let error = uncertainty
        .map(|u| {
            crate::ccm::retained_evidence::finite_math::decimal_upper(&u.unit_state_l2_error, p)
        })
        .transpose()?;
    let Some(origin) = measure(s, &Float::with_val(p, 0), error.as_ref(), p, true)? else {
        return Ok(unresolved(
            out,
            "origin/error transport arithmetic unresolved within 4096 guard bits",
        ));
    };
    save(&mut out.values, origin, p)?;
    if let Some(roots) = roots {
        for point in &roots.dataset.points {
            let mut rr = row(point.ordinal, "retained_window_ordinal");
            if let Some(t) = &point.value {
                let t = scalar(t, roots.dataset.precision_bits)?;
                if let Some(measured) = measure(s, &t, error.as_ref(), p, false)? {
                    save(&mut rr.values, measured, p)?;
                    if error.is_none() {
                        rr.outcome = "channels_resolved_budget_unassessed".into();
                        rr.notes.push("source error unavailable; arithmetic enclosures do not establish source accuracy".into());
                    } else {
                        rr.notes.push("conditional on external finite-support L2 error bound; source certificate and hypotheses retained as input, not certified here; no root isolation or ordinal certificate inferred".into());
                    }
                } else {
                    rr.outcome = "cancellation_limited".into();
                    rr.notes.push(
                        "observation/error arithmetic unresolved within 4096 guard bits".into(),
                    );
                }
            } else {
                rr.outcome = "missing_input".into();
            }
            out.rows.push(rr);
        }
    }
    out.convention="unit_L2_dx finite-support source uncertainty; exp(-itx) at retained roots; directed Cauchy-Schwarz value and derivative transport; allowance upper and margin lower endpoints; retained-window ordinals remain distinct from zeta ordinals; external hypotheses conditional".into();
    Ok(out)
}
