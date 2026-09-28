//! Directed differences of independently supplied actions on one unit source.
use crate::ccm::{
    convergence_capture::OperatorAction,
    extended_research::{
        missing, put, report, row, save_arithmetic_enclosure, unresolved, ExtendedAnalysis,
        ExtensionOptions, ExternalResearchInputs,
    },
    retained_evidence::{
        finite_math::{narrow, normalized, scale, scale_float},
        point, scalar,
    },
    state_geometry::RetainedState,
};
use anyhow::Result;
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;
fn calculate(
    s: &RetainedState,
    a: &OperatorAction,
    b: &OperatorAction,
    input_precision: u32,
    requested: u32,
    p: u32,
) -> Result<Option<(I, I)>> {
    let exponent = s
        .coefficients
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero consistency source"))?;
    let x = s
        .coefficients
        .iter()
        .map(|x| Ok(I::from_float(&scale_float(x, -i64::from(exponent), p)?, p)?))
        .collect::<Result<Vec<_>>>()?;
    let q = x.iter().fold(I::from_i64(0, p), |q, x| q.add(&x.square()));
    let delta = a
        .action
        .iter()
        .zip(&b.action)
        .map(|(a, b)| {
            Ok(I::from_float(&scalar(a, input_precision)?, p)?
                .sub(&I::from_float(&scalar(b, input_precision)?, p)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let (delta, e) = normalized(&delta)?;
    let mut norm2 = I::from_i64(0, p);
    let mut signed = norm2.clone();
    for (delta, x) in delta.iter().zip(&x) {
        norm2 = norm2.add(&delta.square());
        signed = signed.add(&delta.mul(x));
    }
    let norm = scale(&norm2.sqrt()?, e)?;
    let signed = scale(
        &signed
            .mul(&I::from_i64(
                point::orientation(&s.coefficients, p).into(),
                p,
            ))
            .div(&q.sqrt()?)?,
        e,
    )?;
    for value in [&norm, &signed] {
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
    Ok(Some((norm, signed)))
}
pub(super) fn analyze(
    s: &RetainedState,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("consistency", s, o);
    let Some(input) = input else {
        return Ok(missing(out, "independent component actions required"));
    };
    let Some(run) = &input.run_once else {
        return Ok(missing(out, "independent component actions required"));
    };
    let Some(completion) = &run.completion else {
        return Ok(missing(out, "independent component actions required"));
    };
    let requested = o.working_precision_bits;
    let scratch =
        (12 * s.coefficients.len() as u64 + 256) * (u64::from(requested + 4096).div_ceil(8) + 64);
    if o.maximum_working_bytes.is_some_and(|limit| scratch > limit) {
        return Ok(unresolved(
            out,
            "consistency maximum-guard scratch exceeds explicit working-byte budget",
        ));
    }
    for action in &completion.independent_actions {
        let mut rr = row(out.rows.len() + 1, action.label.clone());
        if let Some(other) = run
            .component_actions
            .iter()
            .find(|a| a.label == "tau_prime_reconstructed" && action.label == "tau_prime_direct")
        {
            let mut accepted = false;
            for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
                if let Some((norm, signed)) = calculate(
                    s,
                    action,
                    other,
                    input.precision_bits,
                    requested,
                    requested + guard,
                )? {
                    save_arithmetic_enclosure(
                        &mut rr.values,
                        "action_difference_norm",
                        &norm,
                        requested,
                    )?;
                    save_arithmetic_enclosure(
                        &mut rr.values,
                        "signed_energy_difference",
                        &signed,
                        requested,
                    )?;
                    put(
                        &mut rr.values,
                        "arithmetic_precision_bits",
                        &Float::with_val(requested, requested + guard),
                    );
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                rr.outcome = "cancellation_limited".into();
                rr.notes
                    .push("action difference unresolved within 4096 guard bits".into());
            }
            rr.notes.push(format!("independent source {}; reconstructed source {}; common signed unit-state convention is an external premise; {}",action.source_digest.0,other.source_digest.0,action.convention));
        } else {
            rr.outcome = "missing_input".into();
            rr.notes
                .push("no identified direct/reconstructed action pair".into());
        }
        out.rows.push(rr);
    }
    if out.rows.is_empty() {
        return Ok(missing(out,"direct retained prime component unavailable; algebraic closure is not independent validation"));
    }
    out.convention="direct retained component action versus algebraic reconstruction on the same signed unit source; directed stored-point differences; common normalization and source independence remain external premises".into();
    Ok(out)
}
