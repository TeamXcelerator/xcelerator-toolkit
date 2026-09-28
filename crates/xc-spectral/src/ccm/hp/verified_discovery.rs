//! Complete distinct movable-root enumeration in an exact requested window.
//! Exact rational coefficients bind every nonzero residue and distinct pole.
//! Unresolved multiplicity, output rounding or resource limits return errors.
use crate::ccm::certified_roots::boundary;
use anyhow::{bail, Result};
use rug::{Float, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;

fn coefficient_budget(values: &[Rational]) -> Result<()> {
    let bits: u128 = values
        .iter()
        .map(|x| {
            u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits())
        })
        .sum();
    if bits > 67_108_864 || bits.div_ceil(8) * (values.len() as u128 + 32) > 8u128 << 30 {
        bail!("exact discovery polynomial exceeds the coefficient workspace budget");
    }
    Ok(())
}
fn divide(values: &[Rational], root: &Rational) -> Result<Vec<Rational>> {
    let degree = values.len() - 1;
    let mut out = vec![Rational::from(0); degree];
    out[degree - 1] = values[degree].clone();
    for j in (1..degree).rev() {
        out[j - 1] = values[j].clone() + Rational::from(root * &out[j]);
    }
    if values[0].clone() + Rational::from(root * &out[0]) != 0 {
        bail!("exact discovery synthetic division left a remainder");
    }
    coefficient_budget(&out)?;
    Ok(out)
}
fn numerator(poles: &[Float], weights: &[Float]) -> Result<Vec<Rational>> {
    let active = poles
        .iter()
        .zip(weights)
        .filter(|(_, w)| !w.is_zero())
        .map(|(p, w)| {
            (
                p.to_rational().expect("validated pole"),
                w.to_rational().expect("validated residue"),
            )
        })
        .collect::<Vec<_>>();
    if active.is_empty() {
        bail!("the secular point source is identically zero");
    }
    let mut denominator = vec![Rational::from(1)];
    for (pole, _) in &active {
        let mut next = vec![Rational::from(0); denominator.len() + 1];
        for (j, value) in denominator.iter().enumerate() {
            next[j] -= Rational::from(value * pole);
            next[j + 1] += value;
        }
        coefficient_budget(&next)?;
        denominator = next;
    }
    let mut out = vec![Rational::from(0); active.len()];
    for (pole, weight) in &active {
        for (value, coefficient) in out.iter_mut().zip(divide(&denominator, pole)?) {
            *value += coefficient * weight;
        }
        coefficient_budget(&out)?;
    }
    while out.len() > 1 && out.last().is_some_and(|x| x == &0) {
        out.pop();
    }
    if out.iter().all(|x| x == &0) {
        bail!("the secular point source is identically zero");
    }
    Ok(out)
}

#[cfg(any(test, not(feature = "arb")))]
fn exact_isolation(
    coefficients: &[Rational],
    lower: &Rational,
    upper: &Rational,
    work: u32,
) -> Result<Vec<I>> {
    use rug::Integer;
    // The portable rational fallback is bounded; larger computations require
    // the FLINT/Arb implementation, never a silent return to sampled discovery.
    if coefficients.len() > 129 || work > 16_384 {
        bail!("large exact discovery requires the Arb feature");
    }
    let width = (upper.clone() - lower) / Rational::from(Integer::from(1) << work);
    xc_numerics::interval::exact_sturm_isolate_roots(
        coefficients,
        lower.clone(),
        upper.clone(),
        width,
        work as usize * 4,
    )?
    .iter()
    .map(|v| {
        let lo = I::from_rational(v.lower(), work);
        let hi = I::from_rational(v.upper(), work);
        Ok(I::new(lo.lower().clone(), hi.upper().clone())?)
    })
    .collect()
}
fn isolate(
    coefficients: &[Rational],
    lower: &Rational,
    upper: &Rational,
    work: u32,
) -> Result<Vec<I>> {
    #[cfg(feature = "arb")]
    {
        let (roots, square_free) = crate::ccm::arb_bridge::rational_polynomial_real_roots(
            coefficients,
            lower,
            upper,
            work,
        )?;
        if !square_free {
            bail!("exact discovery has unresolved repeated roots");
        }
        Ok(roots)
    }
    #[cfg(not(feature = "arb"))]
    exact_isolation(coefficients, lower, upper, work)
}

pub(super) fn roots(
    poles: &[Float],
    weights: &[Float],
    lower: &Float,
    upper: &Float,
    p: u32,
) -> Result<Vec<Float>> {
    if !lower.is_finite() || !upper.is_finite() {
        bail!("nonfinite discovery window");
    }
    boundary::rational_budget([lower, upper].into_iter(), poles.len())?;
    roots_rational(
        poles,
        weights,
        &lower.to_rational().expect("finite bound"),
        &upper.to_rational().expect("finite bound"),
        p,
        false,
        false,
    )
}

fn roots_rational(
    poles: &[Float],
    weights: &[Float],
    lower: &Rational,
    upper: &Rational,
    p: u32,
    closed_lower: bool,
    closed_upper: bool,
) -> Result<Vec<Float>> {
    if !(64..=1_000_000).contains(&p)
        || poles.is_empty()
        || poles.len() != weights.len()
        || lower >= upper
    {
        bail!(
            "exact discovery requires compatible nonempty source shape, supported precision and an ordered finite window"
        );
    }
    let source_precision = poles
        .iter()
        .chain(weights)
        .map(Float::prec)
        .max()
        .unwrap()
        .max(p);
    boundary::source_budget(poles.len(), source_precision)?;
    boundary::rational_budget(poles.iter().chain(weights), poles.len())?;
    coefficient_budget(&[lower.clone(), upper.clone()])?;
    let argument_bits = [lower, upper]
        .into_iter()
        .map(|q| {
            u128::from(q.numer().significant_bits()) + u128::from(q.denom().significant_bits())
        })
        .sum::<u128>();
    if argument_bits * poles.len() as u128 * 2 > 67_108_864 {
        bail!("discovery window exceeds exact evaluation workspace");
    }
    if poles.windows(2).any(|x| x[0] >= x[1]) {
        bail!("exact discovery poles must be distinct and increasing");
    }
    let original = numerator(poles, weights)?;
    let mut coefficients = original.clone();
    let mut endpoints = Vec::new();
    // Endpoint membership is decided exactly before point rounding or root
    // isolation. The interior polynomial has its boundary factors removed.
    for (bound, included) in [(lower, closed_lower), (upper, closed_upper)] {
        let mut multiplicity = 0;
        while coefficients.len() > 1
            && xc_numerics::interval::evaluate_real_polynomial_exact(&coefficients, bound)? == 0
        {
            coefficients = divide(&coefficients, bound)?;
            multiplicity += 1;
        }
        if included && multiplicity > 1 {
            bail!("included discovery boundary has an unresolved repeated root");
        }
        if included && multiplicity == 1 {
            endpoints.push(bound.clone());
        }
    }
    if coefficients.len() == 1 && endpoints.is_empty() {
        return Ok(vec![]);
    }
    for guard in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p.saturating_add(guard).min(1_000_000);
        let mut intervals = if coefficients.len() > 1 {
            match isolate(&coefficients, lower, upper, work) {
                Ok(intervals) => intervals,
                Err(error) => {
                    #[cfg(feature = "arb")]
                    if crate::ccm::arb_bridge::isolation_needs_more_precision(&error) {
                        continue;
                    }
                    return Err(error);
                }
            }
        } else {
            vec![]
        };
        intervals.extend(endpoints.iter().map(|q| I::from_rational(q, work)));
        intervals.sort_by(|a, b| {
            a.lower()
                .partial_cmp(b.lower())
                .expect("finite isolating endpoint")
        });
        if intervals.windows(2).any(|x| x[0].upper() >= x[1].lower()) {
            continue;
        }
        let mut values = Vec::with_capacity(intervals.len());
        for interval in &intervals {
            interval.validate()?;
            let lo = Float::with_val(p, interval.lower());
            let hi = Float::with_val(p, interval.upper());
            if lo != hi
                || !lo.is_finite()
                || (lo.is_zero()
                    && xc_numerics::interval::evaluate_real_polynomial_exact(
                        &original,
                        &Rational::from(0),
                    )? != 0)
            {
                break;
            }
            if poles.contains(&lo) || values.last().is_some_and(|previous| previous >= &lo) {
                bail!(
                    "isolated roots cannot be represented as distinct non-pole points at the requested precision"
                );
            }
            values.push(lo);
        }
        if values.len() == intervals.len() {
            return Ok(values);
        }
    }
    bail!("exact discovery seed rounding unresolved within 4096 guard bits")
}

/// Positive heights use (lower, upper]; signed symmetric heights use [-H,H].
/// Exact rational window classification precedes all seed point rounding.
pub(super) fn height_plan(
    params: &super::CcmParams,
    l: &Float,
    xi: &[Float],
    target: &super::ZeroTarget,
    options: super::IndependentRootDiscoveryOptions,
    p: u32,
) -> Result<super::IndependentRootDiscoveryPlan> {
    use super::{
        IndependentRootDiscoveryOptions, IndependentRootDiscoveryPlan, IndependentRootDomain,
        RootWindowSemantics, ZeroTarget,
    };
    let spacing = boundary::rounded_spacing(l, p)?;
    let maximum = spacing.clone() * params.n_modes;
    if !maximum.is_finite() || maximum <= 0 {
        bail!("height window has no positive finite CCM reach");
    }
    boundary::rational_budget(std::iter::once(&maximum), xi.len())?;
    let maximum = maximum.to_rational().expect("finite reach");
    let signed = options.domain == IndependentRootDomain::Signed;
    let (lower, upper) = match target {
        ZeroTarget::HeightWindow { lower, upper } if !signed => {
            (boundary::decimal(lower)?, boundary::decimal(upper)?)
        }
        ZeroTarget::SymmetricHeightWindow { height } => {
            let upper = boundary::decimal(height)?;
            (
                if signed {
                    -upper.clone()
                } else {
                    Rational::from(0)
                },
                upper,
            )
        }
        _ => bail!("invalid exact height discovery target"),
    };
    if upper <= 0
        || lower >= upper
        || upper > maximum
        || (!signed && lower < 0)
        || (matches!(target, ZeroTarget::HeightWindow { .. }) && lower <= 0)
    {
        bail!("exact height window lies outside the finite CCM reach");
    }
    let poles = super::secular_poles(&spacing, params.n_modes, p);
    let zero = Rational::from(0);
    let values = roots_rational(
        &poles,
        xi,
        if signed { &lower } else { &zero },
        &upper,
        p,
        signed,
        true,
    )?;
    let first = if signed || lower == 0 {
        0
    } else {
        roots_rational(&poles, xi, &zero, &lower, p, false, true)?.len()
    };
    if first > values.len() {
        bail!("exact height-prefix counts are inconsistent");
    }
    let count = values.len() - first;
    if count == 0 && !options.allow_incomplete {
        bail!("exact height window contains no movable roots");
    }
    if options == IndependentRootDiscoveryOptions::default() {
        Ok(IndependentRootDiscoveryPlan {
            artifact_first_root_index: first + 1,
            artifact_seeds: values[first..].to_vec(),
            selected_positions: (0..count).collect(),
            result_first_root_index: first + 1,
            request_semantics: RootWindowSemantics::strict_positive(count),
        })
    } else {
        Ok(IndependentRootDiscoveryPlan {
            artifact_first_root_index: 1,
            selected_positions: (first..values.len()).collect(),
            artifact_seeds: values,
            result_first_root_index: if signed { 1 } else { first + 1 },
            request_semantics: RootWindowSemantics::advanced(
                options.domain,
                count,
                options.allow_incomplete,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portable_sturm_fallback_agrees_with_independent_factored_roots() {
        let roots = exact_isolation(
            &[
                Rational::from((3, 8)),
                Rational::from((-5, 4)),
                Rational::from(1),
            ],
            &Rational::from(0),
            &Rational::from(1),
            96,
        )
        .unwrap();
        assert_eq!(roots.len(), 2);
        for (interval, q) in roots
            .iter()
            .zip([Rational::from((1, 2)), Rational::from((3, 4))])
        {
            assert!(
                interval.lower().to_rational().unwrap() <= q
                    && interval.upper().to_rational().unwrap() >= q
            );
        }
    }
}
