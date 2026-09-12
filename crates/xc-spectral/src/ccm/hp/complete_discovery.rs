//! Complete positive movable-root discovery for an exact even point source.
//!
//! Counts all positive roots of the rational numerator, including beyond the
//! largest retained pole. The pole points are the same rounded MPFR values
//! consumed by refinement; no ideal-lattice replacement or reference data.
use super::*;
#[cfg(feature = "arb")]
use rug::Rational;

pub(super) fn plan(
    params: &CcmParams,
    l: &Float,
    xi: &[Float],
    target: &ZeroTarget,
    options: IndependentRootDiscoveryOptions,
    precision: u32,
) -> Result<IndependentRootDiscoveryPlan> {
    if options.domain != IndependentRootDomain::Positive {
        bail!("complete movable-root discovery supports the positive domain only");
    }
    let values = positive_roots(params.n_modes, l, xi, precision)?;
    let (requested_count, selected_positions, first_index) = match target {
        ZeroTarget::FirstK { count } if *count > 0 => (
            *count,
            (0..(*count).min(values.len())).collect::<Vec<_>>(),
            1,
        ),
        ZeroTarget::IndexRange { first, last } if *first > 0 && first <= last => (
            last - first + 1,
            ((*first - 1).min(values.len())..(*last).min(values.len())).collect(),
            *first,
        ),
        ZeroTarget::HeightWindow { lower, upper } => {
            let lower = Float::with_val(precision, Float::parse(lower)?);
            let upper = Float::with_val(precision, Float::parse(upper)?);
            if lower <= 0 || lower >= upper {
                bail!("complete positive height window requires 0 < lower < upper");
            }
            let first = values.partition_point(|v| v <= &lower);
            let last = values.partition_point(|v| v <= &upper);
            (last - first, (first..last).collect(), first + 1)
        }
        _ => bail!(
            "complete positive discovery requires FirstK, IndexRange or a positive HeightWindow"
        ),
    };
    if selected_positions.len() < requested_count || selected_positions.is_empty() {
        let reason = format!(
            "complete source-only isolation found {} positive movable roots; request needs {} roots starting at index {} ({} available in the requested window)",
            values.len(), requested_count, first_index, selected_positions.len()
        );
        if !options.allow_incomplete {
            bail!("{reason}");
        }
        eprintln!("[HP] incomplete root window: {reason}; preserving available evidence");
    }
    eprintln!("[HP] complete positive movable-root discovery: {} roots isolated from the retained point source", values.len());
    Ok(IndependentRootDiscoveryPlan {
        artifact_first_root_index: 1,
        artifact_seeds: values,
        selected_positions,
        result_first_root_index: first_index,
        request_semantics: RootWindowSemantics {
            domain: IndependentRootDomain::Positive,
            requested_count,
            allow_incomplete: options.allow_incomplete,
            artifact_format: RootArtifactFormat::CompleteV10,
        },
    })
}

#[cfg(not(feature = "arb"))]
fn positive_roots(_n: usize, _l: &Float, _xi: &[Float], _precision: u32) -> Result<Vec<Float>> {
    bail!("complete movable-root discovery requires the xc-spectral arb feature and FLINT; rebuild the application with its documented Arb feature")
}

#[cfg(feature = "arb")]
fn positive_roots(n: usize, l: &Float, xi: &[Float], precision: u32) -> Result<Vec<Float>> {
    let mut spacing = pi(precision);
    spacing *= 2;
    spacing /= l;
    let poles = secular_poles(&spacing, n, precision);
    roots_from_points(&poles, xi, precision)
}

#[cfg(feature = "arb")]
fn numerator(poles: &[Float], weights: &[Float]) -> Result<Vec<Rational>> {
    if poles.len() < 3 || poles.len().is_multiple_of(2) || poles.len() != weights.len() {
        bail!("complete even discovery requires matching odd-sized pole and weight arrays");
    }
    let n = poles.len() / 2;
    if poles[n] != 0
        || poles.windows(2).any(|p| p[0] >= p[1])
        || poles.iter().chain(weights).any(|x| !x.is_finite())
        || (0..n).any(|j| poles[j] != -poles[2 * n - j].clone() || weights[j] != weights[2 * n - j])
    {
        bail!("complete movable-root discovery requires exact even weights and symmetric distinct pole points; no parity projection is applied");
    }
    let active = (n + 1..poles.len())
        .filter(|&j| weights[j] != 0)
        .map(|j| {
            let mut square = poles[j].to_rational().expect("finite pole");
            square.square_mut();
            (square, weights[j].to_rational().expect("finite weight"))
        })
        .collect::<Vec<_>>();
    let mut denominator = vec![Rational::from(1)];
    for (pole, _) in &active {
        let mut next = vec![Rational::from(0); denominator.len() + 1];
        for (j, a) in denominator.iter().enumerate() {
            next[j] -= Rational::from(a * pole);
            next[j + 1] += a;
        }
        denominator = next;
    }
    let central = weights[n].to_rational().expect("finite central weight");
    let mut result = denominator
        .iter()
        .map(|x| Rational::from(x * &central))
        .collect::<Vec<_>>();
    for (pole, weight) in &active {
        let degree = denominator.len() - 1;
        let mut quotient = vec![Rational::from(0); degree];
        quotient[degree - 1] = denominator[degree].clone();
        for j in (1..degree).rev() {
            quotient[j - 1] = denominator[j].clone() + Rational::from(pole * &quotient[j]);
        }
        if denominator[0].clone() + Rational::from(pole * &quotient[0]) != 0 {
            bail!("exact secular numerator division failed");
        }
        for (j, a) in quotient.iter().enumerate() {
            result[j + 1] += Rational::from(a * weight) * 2;
        }
    }
    while result.len() > 1 && result.last().is_some_and(|x| x == &0) {
        result.pop();
    }
    if result.iter().all(|x| x == &0) {
        bail!("the retained secular function is identically zero");
    }
    // Multiplication by t introduces zero factors when the central residue is
    // absent. They are outside the strictly positive movable-root domain.
    while result.len() > 1 && result[0] == 0 {
        result.remove(0);
    }
    Ok(result)
}

#[cfg(feature = "arb")]
fn roots_from_points(poles: &[Float], weights: &[Float], precision: u32) -> Result<Vec<Float>> {
    let coefficients = numerator(poles, weights)?;
    if coefficients.len() == 1 {
        return Ok(vec![]);
    }
    let leading = coefficients
        .last()
        .expect("nonempty numerator")
        .clone()
        .abs();
    let maximum = coefficients[..coefficients.len() - 1]
        .iter()
        .map(|x| x.clone().abs() / &leading)
        .max()
        .expect("nonconstant numerator");
    // Cauchy's strict bound covers every complex root, without reference
    // heights and without treating the largest pole as a root boundary.
    let upper = maximum + 2;
    let (mut squares, square_free) = super::super::arb_bridge::rational_polynomial_real_roots(
        &coefficients,
        &Rational::from(0),
        &upper,
        precision,
    )?;
    if !square_free {
        bail!("complete movable-root count has unresolved repeated roots");
    }
    squares.sort_by(|a, b| a.lower().partial_cmp(b.lower()).expect("finite interval"));
    if squares.windows(2).any(|w| w[0].upper() >= w[1].lower()) {
        bail!("complete movable-root isolation returned overlapping intervals");
    }
    let mut roots = Vec::with_capacity(squares.len());
    for square in squares {
        let interval = square.sqrt()?;
        let value = interval.midpoint_point().lower().clone();
        if value <= 0 || roots.last().is_some_and(|previous| previous >= &value) {
            bail!("complete movable-root seeds cannot be represented as distinct positive points at the source precision");
        }
        roots.push(value);
    }
    Ok(roots)
}

#[cfg(all(test, feature = "arb"))]
mod tests {
    use super::*;
    fn source(weights: &[f64]) -> (Vec<Float>, Vec<Float>) {
        let n = weights.len() / 2;
        (
            (-(n as i64)..=n as i64)
                .map(|j| Float::with_val(256, j))
                .collect(),
            weights.iter().map(|w| Float::with_val(256, w)).collect(),
        )
    }
    #[test]
    fn complete_discovery_finds_both_roots_above_last_pole() {
        // Q(s)=(s-9)(s-16); neither t=3 nor t=4 lies below the last pole 2.
        let (p, w) = source(&[2.5, -20., 36., -20., 2.5]);
        let r = roots_from_points(&p, &w, 256).unwrap();
        assert_eq!(r.len(), 2);
        assert!((r[0].clone() - 3i32).abs() < Float::with_val(256, 2).pow(-240));
        assert!((r[1].clone() - 4i32).abs() < Float::with_val(256, 2).pow(-240));
    }
    #[test]
    fn complete_discovery_handles_missing_residues_and_no_positive_roots() {
        let (p, w) = source(&[0., 0., 1., 0., 0.]);
        assert!(roots_from_points(&p, &w, 256).unwrap().is_empty());
        let (p, w) = source(&[0., 1., 0., 1., 0.]);
        assert!(roots_from_points(&p, &w, 256).unwrap().is_empty());
        let (p, w) = source(&[0., -1., 4., -1., 0.]);
        let r = roots_from_points(&p, &w, 256).unwrap();
        assert_eq!(r.len(), 1);
        assert!((r[0].clone().square() - 2i32).abs() < Float::with_val(256, 2).pow(-240));
    }
    #[test]
    fn complete_discovery_rejects_parity_and_repeated_root_ambiguity() {
        let (p, mut w) = source(&[2.5, -20., 36., -20., 2.5]);
        w[0] += 1;
        assert!(roots_from_points(&p, &w, 256).is_err());
        // Q(s)=24*(s-9)^2; exact integer residues preserve multiplicity.
        let (p, w) = source(&[25., -256., 486., -256., 25.]);
        assert!(roots_from_points(&p, &w, 256).is_err());
    }
}

/// Refined values must retain the ordering of the complete isolated roots.
/// Failed/stagnated outcomes stay visible; no root can jump into another
/// isolated root's nearest-seed cell and still acquire that root's ordinal.
pub(super) fn validate_assignment(roots: &[EigenvalueResult], seeds: &[Float]) -> Result<()> {
    if roots.len() != seeds.len() {
        bail!("complete root count changed during refinement");
    }
    for (j, root) in roots.iter().enumerate() {
        let Some(value) = root.value() else {
            continue;
        };
        let distance = (value.clone() - &seeds[j]).abs();
        for neighbor in [j.checked_sub(1), (j + 1 < seeds.len()).then_some(j + 1)]
            .into_iter()
            .flatten()
        {
            if distance >= (value.clone() - &seeds[neighbor]).abs() {
                bail!("complete root ordinal {} crossed its isolated starting-point cell during refinement",j+1);
            }
        }
    }
    Ok(())
}
