//! Directed finite-section expressions on original stored points.
use crate::ccm::{
    convergence_capture::ComparisonState,
    extended_research::{
        missing, put, report, row, save_arithmetic_enclosure, unresolved, ExtendedAnalysis,
        ExtensionOptions, ExternalResearchInputs,
    },
    retained_evidence::{
        finite_math::{narrow, scale, scale_float},
        scalar, RetainedMatrix,
    },
    state_geometry::RetainedState,
};
use anyhow::{bail, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;
type Values = BTreeMap<String, I>;
struct Measurement {
    rows: Vec<Values>,
    values: Values,
    precision: u32,
    zero_comparison: bool,
}
fn vector(raw: &[Float], p: u32) -> Result<Option<Vec<I>>> {
    let Some(e) = raw.iter().filter_map(Float::get_exp).max() else {
        return Ok(None);
    };
    raw.iter()
        .map(|x| Ok(I::from_float(&scale_float(x, -i64::from(e), p)?, p)?))
        .collect::<Result<_>>()
        .map(Some)
}
fn norm2(x: &[I], p: u32) -> I {
    x.iter()
        .fold(I::from_i64(0, p), |sum, x| sum.add(&x.square()))
}
fn narrow_values(values: &Values, requested: u32) -> Result<bool> {
    for value in values.values() {
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
        if !narrow(std::slice::from_ref(value), &natural, requested)? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn update_exponent(value: &I, maximum: &mut Option<i64>) -> Result<()> {
    value.validate()?;
    for exponent in [value.lower(), value.upper()]
        .iter()
        .filter_map(|v| v.get_exp())
    {
        *maximum = Some(maximum.map_or(i64::from(exponent), |old| old.max(i64::from(exponent))));
    }
    Ok(())
}
fn comparison(
    c: &ComparisonState,
    m: &RetainedMatrix<'_>,
    modes: usize,
    p: u32,
) -> Result<(Values, bool)> {
    let n = 2 * modes + 1;
    let d = c.coefficients.len();
    let offset = modes - c.n_modes;
    let delta = |i: usize, j: usize| -> Result<I> {
        Ok(
            I::from_float(&m.entries[(i + offset) * n + j + offset], p)?.sub(&I::from_float(
                &scalar(&c.matrix[i * d + j], c.precision_bits)?,
                p,
            )?),
        )
    };
    let mut exponent = None;
    for i in 0..d {
        for j in 0..d {
            update_exponent(&delta(i, j)?, &mut exponent)?
        }
    }
    let exponent = exponent.unwrap_or(0);
    let raw = c
        .coefficients
        .iter()
        .map(|v| scalar(v, c.precision_bits))
        .collect::<Result<Vec<_>>>()?;
    let x = vector(&raw, p)?;
    let mut frobenius = I::from_i64(0, p);
    let mut signed = frobenius.clone();
    for i in 0..d {
        for j in 0..d {
            let a = scale(&delta(i, j)?, -exponent)?;
            frobenius = frobenius.add(&a.square());
            if let Some(x) = &x {
                signed = signed.add(&x[i].mul(&a).mul(&x[j]))
            }
        }
    }
    let mut values = Values::from([
        (
            "comparison_block_frobenius_difference".into(),
            scale(&frobenius.sqrt()?, exponent)?,
        ),
        (
            "comparison_precision_bits".into(),
            I::from_u64(c.precision_bits.into(), p),
        ),
    ]);
    if let Some(x) = &x {
        values.insert(
            "comparison_state_signed_block_defect".into(),
            scale(&signed.div(&norm2(x, p))?, exponent)?,
        );
    }
    Ok((values, x.is_none()))
}
fn calculate(
    s: &RetainedState,
    m: &RetainedMatrix<'_>,
    c: Option<&ComparisonState>,
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let x =
        vector(&s.coefficients, p)?.ok_or_else(|| anyhow::anyhow!("zero finite-section source"))?;
    let n = x.len();
    let q = norm2(&x, p);
    let eigen = I::from_float(&scalar(&s.eigenvalue, s.precision)?, p)?;
    let shifted = |i: usize, j: usize| -> Result<I> {
        let entry = I::from_float(&m.entries[i * n + j], p)?;
        Ok(if i == j { entry.sub(&eigen) } else { entry })
    };
    let matrix_exponent = m
        .entries
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let mut residual_exponent = None;
    for i in 0..n {
        for j in 0..n {
            if !s.coefficients[j].is_zero() {
                update_exponent(&shifted(i, j)?, &mut residual_exponent)?
            }
        }
    }
    let residual_exponent = residual_exponent.unwrap_or(0);
    let mut energy_action = vec![I::from_i64(0, p); n];
    let mut residual_action = energy_action.clone();
    let mut retained = I::from_i64(0, p);
    let mut rows = Vec::with_capacity(s.modes + 1);
    // Each matrix column enters once. All symmetric prefixes remain O(n^2).
    for k in 0..=s.modes {
        let columns = if k == 0 {
            vec![s.modes]
        } else {
            vec![s.modes - k, s.modes + k]
        };
        for j in columns {
            retained = retained.add(&x[j].square());
            if s.coefficients[j].is_zero() {
                continue;
            }
            for i in 0..n {
                energy_action[i] = energy_action[i].add(
                    &I::from_float(&scale_float(&m.entries[i * n + j], -matrix_exponent, p)?, p)?
                        .mul(&x[j]),
                );
                residual_action[i] =
                    residual_action[i].add(&scale(&shifted(i, j)?, -residual_exponent)?.mul(&x[j]));
            }
        }
        let lo = s.modes - k;
        let hi = s.modes + k;
        let mut low = I::from_i64(0, p);
        let mut high = low.clone();
        let mut energy = low.clone();
        let mut omitted = low.clone();
        for i in 0..n {
            if (lo..=hi).contains(&i) {
                low = low.add(&residual_action[i].square());
                energy = energy.add(&x[i].mul(&energy_action[i]));
            } else {
                high = high.add(&residual_action[i].square());
                omitted = omitted.add(&x[i].square());
            }
        }
        let values = Values::from([
            ("n_modes".into(), I::from_u64(k as u64, p)),
            ("retained_mass".into(), retained.div(&q)?),
            ("omitted_mass".into(), omitted.div(&q)?),
            (
                "low_residual_squared".into(),
                scale(&low.div(&q)?, 2 * residual_exponent)?,
            ),
            (
                "high_forcing_squared".into(),
                scale(&high.div(&q)?, 2 * residual_exponent)?,
            ),
            (
                "truncated_energy".into(),
                scale(&energy.div(&q)?, matrix_exponent)?,
            ),
        ]);
        if !narrow_values(&values, requested)? {
            return Ok(None);
        }
        rows.push(values);
    }
    let (values, zero_comparison) = if let Some(c) = c {
        comparison(c, m, s.modes, p)?
    } else {
        (Values::new(), false)
    };
    if !narrow_values(&values, requested)? {
        return Ok(None);
    }
    Ok(Some(Measurement {
        rows,
        values,
        precision: p,
        zero_comparison,
    }))
}
fn save(
    map: &mut BTreeMap<String, String>,
    values: Values,
    precision: u32,
    requested: u32,
) -> Result<()> {
    for (name, value) in values {
        save_arithmetic_enclosure(map, &name, &value, requested)?
    }
    put(
        map,
        "arithmetic_precision_bits",
        &Float::with_val(requested, precision),
    );
    Ok(())
}
pub(super) fn analyze(
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("finite_section_transfer", s, o);
    let Some(m) = m else {
        return Ok(missing(out, "retained Tau required"));
    };
    let requested = o.working_precision_bits;
    let c = input
        .and_then(|i| i.run_once.as_ref())
        .and_then(|i| i.comparison.as_ref());
    if let Some(c) = c {
        if xc_core::DecimalLiteral::new(&c.lambda_squared)?.canonical()?
            != xc_core::DecimalLiteral::new(&s.cutoff)?.canonical()?
        {
            bail!("comparison cutoff differs")
        }
        if c.precision_bits > requested {
            return Ok(unresolved(
                out,
                "comparison precision exceeds requested working precision",
            ));
        }
    }
    let scratch =
        (48 * s.coefficients.len() as u64 + 512) * (u64::from(requested + 4096).div_ceil(8) + 64);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            out,
            "finite-section maximum-guard scratch exceeds explicit working-byte budget",
        ));
    }
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let Some(measured) = calculate(s, m, c, requested, requested + guard)? else {
            continue;
        };
        for (k, values) in measured.rows.into_iter().enumerate() {
            let mut rr = row(k + 1, "projection_of_retained_full_state");
            save(&mut rr.values, values, measured.precision, requested)?;
            out.rows.push(rr);
        }
        save(
            &mut out.values,
            measured.values,
            measured.precision,
            requested,
        )?;
        if measured.zero_comparison {
            out.outcome = "partial_unresolved".into();
            out.reason=Some("comparison coefficient vector is zero; signed defect withheld; prefix and block-norm measurements retained".into());
        } else if c.is_none() {
            out.reason=Some("independently assembled comparison configuration unavailable; every parent-derived prefix retained".into());
        }
        out.convention="all symmetric prefixes of one retained Tau and projections of its state; homogeneous stored-point arithmetic enclosures; no independent prefix eigenstate or convergence claim; optional comparison block retains its own point precision".into();
        return Ok(out);
    }
    Ok(unresolved(out,"finite-section arithmetic unresolved within 4096 guard bits; no prefix measurements claimed"))
}
