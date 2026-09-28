//! Transport of a finite directional response to physical Mellin coordinates.
//! The arithmetic is enclosed; the displacement/root hypotheses stay conditional.
use super::super::{
    extended_research::{
        missing, report, save_arithmetic_enclosure, unresolved, AnalysisRow, ExtendedAnalysis,
        ExtensionOptions, ExternalResearchInputs,
    },
    retained_evidence::{
        finite_math::{abs, decimal, narrow},
        scalar, RetainedRoots,
    },
    state_geometry::RetainedState,
};
use anyhow::{bail, Context, Result};
use rug::Float;
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::MpfrInterval as I;

fn saved(row: &AnalysisRow, name: &str, p: u32) -> Result<I> {
    // The reused directional producer serializes its interval endpoints outward.
    let lower = decimal(
        row.values
            .get(&format!("{name}_lower"))
            .context("directional lower endpoint absent")?,
        p,
    )?;
    let upper = decimal(
        row.values
            .get(&format!("{name}_upper"))
            .context("directional upper endpoint absent")?,
        p,
    )?;
    Ok(I::new(lower.lower().clone(), upper.upper().clone())?)
}
fn support_log(c: &str, requested: u32) -> Result<(I, u32)> {
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let p = requested + guard;
        let l = decimal(c, p)?.ln()?;
        if l.is_strictly_positive() && narrow(std::slice::from_ref(&l), l.lower(), requested)? {
            return Ok((l, p));
        }
    }
    bail!("transport exact-cutoff logarithm unresolved within 4096 guard bits")
}

fn calculate(
    s: &RetainedState,
    row: &AnalysisRow,
    root: Option<(&str, u32)>,
    input: Option<&ExternalResearchInputs>,
    requested: u32,
) -> Result<(BTreeMap<String, I>, u32, Option<String>)> {
    let once = input.and_then(|i| i.run_once.as_ref());
    let source_precision = input.map_or(s.precision, |i| i.precision_bits);
    let velocity = once.and_then(|i| i.log_cutoff_velocity.as_ref());
    let physical = row
        .values
        .contains_key("conditional_tau_response_tau_total")
        && velocity.is_some()
        && root.is_some();
    let (l, p) = if physical {
        let (l, p) = support_log(&s.cutoff, requested)?;
        (Some(l), p)
    } else {
        (None, requested + 64)
    };
    let point = |x: &str| -> Result<I> { Ok(I::from_float(&scalar(x, source_precision)?, p)?) };
    let mut values = BTreeMap::new();
    let mut sum = I::from_i64(0, p);
    let mut absolute = I::from_i64(0, p);
    for action in once.map(|i| i.derivative_actions.as_slice()).unwrap_or(&[]) {
        if action.label == "tau_total" {
            continue;
        }
        let name = format!("forcing_{}", action.label);
        if row.values.contains_key(&name) {
            let f = saved(row, &name, p)?;
            sum = sum.add(&f);
            absolute = absolute.add(&abs(&f)?);
        }
    }
    values.insert("component_forcing_sum".into(), sum.clone());
    values.insert("absolute_component_forcing_sum".into(), absolute);
    if row.values.contains_key("forcing_tau_total") {
        values.insert(
            "forcing_closure_defect".into(),
            saved(row, "forcing_tau_total", p)?.sub(&sum),
        );
    }
    let original_root = root.map(|(t, q)| scalar(t, q)).transpose()?;
    if let (Some(l), Some(t), Some(velocity)) = (&l, &original_root, velocity) {
        let t = I::from_float(t, p)?;
        let dl = point(velocity)?;
        let support = t.neg().mul(&dl).div(l)?;
        let response = saved(row, "conditional_tau_response_tau_total", p)?;
        // tau=(t*log(C)/(2*pi))^2, so dt/dtau=t/(2*tau).
        // This identity avoids forming log(C)^2 before a compensating division.
        let motion = response
            .mul(&t)
            .div(&saved(row, "tau", p)?.mul(&I::from_i64(2, p)))?;
        values.insert(
            "conditional_total_physical_velocity".into(),
            support.add(&motion),
        );
        values.insert("support_motion".into(), support);
        values.insert("operator_motion".into(), motion);
    }
    let mut note = None;
    if let Some(check) = once
        .and_then(|i| i.completion.as_ref())
        .and_then(|c| c.response_checks.iter().find(|c| c.ordinal == row.ordinal))
    {
        let matches=check.branch=="production_shifted_secular" && check.coordinate=="mellin_t" && check.derivative_parameter=="u=log(lambda_squared)" && check.activation_convention=="analytic_right_continuous_active_prime_set; tau=pole-archimedean-prime; total roots include d(2*pi*n/u)/du=-2*pi*n/u^2" && original_root.as_ref().is_some_and(|t| scalar(&check.t,source_precision).is_ok_and(|v|v==*t));
        if matches {
            if let Some(v) = &check.support_velocity {
                values.insert("retained_secular_pole_motion".into(), point(v)?);
            }
            if let Some(v) = &check.total_velocity {
                values.insert("retained_total_velocity".into(), point(v)?);
            }
            if let (Some(a), Some(b), Some(total)) = (
                &check.fixed_velocity,
                &check.support_velocity,
                &check.total_velocity,
            ) {
                values.insert(
                    "retained_transport_additivity_defect".into(),
                    point(a)?.add(&point(b)?).sub(&point(total)?),
                );
            }
            note=Some(format!("retained shifted-secular response source {}; conditional unshifted directional formula is a distinct branch; cross-formula equality not asserted",check.source_digest.0));
        } else {
            note=Some("retained response comparison withheld: coordinate, parameter, activation convention or stored evaluation point differs".into());
        }
    }
    for value in values.values() {
        value.validate()?;
    }
    Ok((values, p, note))
}

pub(super) fn analyze(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    input: Option<&ExternalResearchInputs>,
    directional: Option<&ExtendedAnalysis>,
) -> Result<ExtendedAnalysis> {
    let mut out = report("root_transport", s, o);
    out.assurance="finite_stored_point_arithmetic_enclosures; inherited directional intervals; conditional root/displacement hypotheses; no source-error or root certificate".into();
    let (Some(directional), Some(roots)) = (directional, roots) else {
        return Ok(missing(out, "retained Tau and roots required"));
    };
    let p = o.working_precision_bits;
    if o.maximum_working_bytes
        .is_some_and(|limit| 256 * (u64::from(p + 4096).div_ceil(8) + 64) > limit)
    {
        return Ok(unresolved(
            out,
            "transport maximum-guard scratch estimate exceeds working-byte budget",
        ));
    }
    out.rows = directional.rows.clone();
    out.outcome = directional.outcome.clone();
    out.reason = directional.reason.clone();
    let points = roots
        .dataset
        .points
        .iter()
        .map(|v| (v.ordinal, v.value.as_deref()))
        .collect::<BTreeMap<_, _>>();
    for row in &mut out.rows {
        let root = points
            .get(&row.ordinal)
            .copied()
            .flatten()
            .map(|t| (t, roots.dataset.precision_bits));
        match calculate(s, row, root, input, p) {
            Ok((values, work, note)) => {
                let mut output = row.values.clone();
                for (name, value) in values {
                    save_arithmetic_enclosure(&mut output, &name, &value, p)?;
                }
                super::super::extended_research::put(
                    &mut output,
                    "transport_arithmetic_precision_bits",
                    &Float::with_val(p, work),
                );
                row.values = output;
                if let Some(note) = note {
                    row.notes.push(note);
                }
            }
            Err(error) => {
                row.outcome = "cancellation_limited".into();
                row.notes.push(format!(
                    "transport arithmetic unresolved; directional measurements retained: {error}"
                ));
                out.outcome = "partial_unresolved".into();
                out.reason = Some(
                    "one or more transport calculations unresolved; directional evidence retained"
                        .into(),
                );
            }
        }
    }
    out.convention="fixed Fourier dimension; declared stored log-cutoff velocities and retained root points; exact decimal cutoff; outward transport of the directional intervals; conditional support-plus-operator response; shifted-secular comparison remains a separate branch".into();
    Ok(out)
}
