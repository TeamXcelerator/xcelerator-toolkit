//! Complete positive movable-root discovery for an exact even point source.
//!
//! Counts all positive roots of the rational numerator, including beyond the
//! largest retained pole. The pole points are the same rounded MPFR values
//! consumed by refinement; no ideal-lattice replacement or reference data.
use super::*;
use crate::ccm::certified_roots::boundary;
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
    let discovery = positive_roots(params.n_modes, l, xi, precision)?;
    let values = &discovery.values;
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
            let lower = boundary::decimal(lower)?;
            let upper = boundary::decimal(upper)?;
            if lower <= 0 || lower >= upper {
                bail!("complete positive height window requires 0 < lower < upper");
            }
            let first = discovery.partition_at_most(&lower)?;
            let last = discovery.partition_at_most(&upper)?;
            if first > last || last > values.len() {
                bail!("exact complete-window partition contradicts the isolated root count");
            }
            (last - first, (first..last).collect(), first + 1)
        }
        _ => bail!(
            "complete positive discovery requires FirstK, IndexRange or a positive HeightWindow"
        ),
    };
    if selected_positions.len() < requested_count || selected_positions.is_empty() {
        let reason = format!(
            "complete source-only isolation found {} positive movable roots; request needs {} roots starting at index {} ({} available in the requested window)",
            values.len(),
            requested_count,
            first_index,
            selected_positions.len()
        );
        if !options.allow_incomplete {
            bail!("{reason}");
        }
        eprintln!("[HP] incomplete root window: {reason}; preserving available evidence");
    }
    eprintln!(
        "[HP] complete positive movable-root discovery: {} roots isolated from the retained point source",
        values.len()
    );
    Ok(IndependentRootDiscoveryPlan {
        artifact_first_root_index: 1,
        artifact_seeds: discovery.values,
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

struct CompleteDiscovery {
    values: Vec<Float>,
    #[cfg(feature = "arb")]
    coefficients: Vec<Rational>,
    #[cfg(feature = "arb")]
    squares: Vec<xc_numerics::mpfr_interval::MpfrInterval>,
}

impl CompleteDiscovery {
    #[cfg(not(feature = "arb"))]
    fn partition_at_most(&self, _bound: &rug::Rational) -> Result<usize> {
        bail!("complete movable-root discovery requires the Arb feature")
    }

    #[cfg(feature = "arb")]
    fn partition_at_most(&self, bound: &Rational) -> Result<usize> {
        let square = Rational::from(bound * bound);
        boundary::rational_budget(
            self.squares.iter().flat_map(|x| [x.lower(), x.upper()]),
            self.squares.len(),
        )?;
        let mut count = 0;
        for interval in &self.squares {
            let lower = interval
                .lower()
                .to_rational()
                .expect("finite isolated endpoint");
            let upper = interval
                .upper()
                .to_rational()
                .expect("finite isolated endpoint");
            if square < lower {
                break;
            }
            if square >= upper {
                count += 1;
                continue;
            }
            // Each disjoint interval contains one simple root of the exact
            // square-free numerator. Inside it, an exact sign comparison
            // determines the side of that root, even when both requested
            // decimal endpoints round to the same MPFR point.
            let coefficient_bits: u128 = self
                .coefficients
                .iter()
                .map(|x| {
                    u128::from(x.numer().significant_bits())
                        + u128::from(x.denom().significant_bits())
                })
                .sum();
            let argument_bits = u128::from(square.numer().significant_bits())
                + u128::from(square.denom().significant_bits())
                + u128::from(lower.numer().significant_bits())
                + u128::from(lower.denom().significant_bits());
            if coefficient_bits + argument_bits * self.coefficients.len() as u128 * 2 > 67_108_864 {
                bail!("exact height-boundary comparison exceeds the rational workspace budget");
            }
            let at_lower =
                xc_numerics::interval::evaluate_real_polynomial_exact(&self.coefficients, &lower)?;
            let at_bound =
                xc_numerics::interval::evaluate_real_polynomial_exact(&self.coefficients, &square)?;
            if at_lower == 0 || at_bound == 0 || (at_lower > 0) != (at_bound > 0) {
                count += 1;
            }
            break;
        }
        Ok(count)
    }
}

#[cfg(not(feature = "arb"))]
fn positive_roots(
    _n: usize,
    _l: &Float,
    _xi: &[Float],
    _precision: u32,
) -> Result<CompleteDiscovery> {
    bail!(
        "complete movable-root discovery requires the xc-spectral arb feature and FLINT; rebuild the application with its documented Arb feature"
    )
}

#[cfg(feature = "arb")]
fn positive_roots(n: usize, l: &Float, xi: &[Float], precision: u32) -> Result<CompleteDiscovery> {
    if !(64..=1_000_000).contains(&precision) || !l.is_finite() || l <= &0 || l.prec() > 1_000_000 {
        bail!("complete discovery requires a finite positive length and supported precision");
    }
    boundary::shape(n, xi.len(), precision)?;
    // Exact height partitioning must use the same retained pole points as
    // refinement and certificate replay, including the rounding of 2*pi/L.
    let spacing = boundary::rounded_spacing(l, precision)?;
    let poles = secular_poles(&spacing, n, precision);
    discover_points(&poles, xi, precision)
}

#[cfg(feature = "arb")]
fn numerator(poles: &[Float], weights: &[Float]) -> Result<Vec<Rational>> {
    if poles.is_empty() || poles.len().is_multiple_of(2) || poles.len() != weights.len() {
        bail!("complete even discovery requires matching odd-sized pole and weight arrays");
    }
    let precision = poles.iter().chain(weights).map(Float::prec).max().unwrap();
    boundary::source_budget(poles.len(), precision)?;
    boundary::rational_budget(poles.iter().chain(weights), poles.len())?;
    let n = poles.len() / 2;
    if poles[n] != 0
        || poles.windows(2).any(|p| p[0] >= p[1])
        || poles.iter().chain(weights).any(|x| !x.is_finite())
        || (0..n).any(|j| poles[j] != -poles[2 * n - j].clone() || weights[j] != weights[2 * n - j])
    {
        bail!(
            "complete movable-root discovery requires exact even weights and symmetric distinct pole points; no parity projection is applied"
        );
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
fn discover_points(
    poles: &[Float],
    weights: &[Float],
    precision: u32,
) -> Result<CompleteDiscovery> {
    if !(64..=1_000_000).contains(&precision) {
        bail!("unsupported complete-discovery output precision");
    }
    let coefficients = numerator(poles, weights)?;
    if coefficients.len() == 1 {
        return Ok(CompleteDiscovery {
            values: vec![],
            coefficients,
            squares: vec![],
        });
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
        precision.saturating_add(32).min(1_000_000),
    )?;
    if !square_free {
        bail!("complete movable-root count has unresolved repeated roots");
    }
    squares.sort_by(|a, b| a.lower().partial_cmp(b.lower()).expect("finite interval"));
    if squares.windows(2).any(|w| w[0].upper() >= w[1].lower()) {
        bail!("complete movable-root isolation returned overlapping intervals");
    }
    let mut roots = Vec::with_capacity(squares.len());
    for square in &squares {
        let interval = square.sqrt()?;
        let value = Float::with_val(precision, interval.midpoint_point().lower());
        if !value.is_finite()
            || value <= 0
            || roots.last().is_some_and(|previous| previous >= &value)
        {
            bail!(
                "complete movable-root seeds cannot be represented as distinct positive points at the source precision"
            );
        }
        roots.push(value);
    }
    Ok(CompleteDiscovery {
        values: roots,
        coefficients,
        squares,
    })
}

#[cfg(all(test, feature = "arb"))]
fn roots_from_points(poles: &[Float], weights: &[Float], precision: u32) -> Result<Vec<Float>> {
    discover_points(poles, weights, precision).map(|result| result.values)
}

#[cfg(all(test, feature = "arb"))]
mod tests {
    use super::*;

    #[test]
    fn complete_discovery_uses_the_public_certificate_pole_source() {
        use crate::ccm::certified_roots::CertifiedSecularFunction;
        for (precision, cutoff) in [(64, 2), (128, 31), (256, 2)] {
            let params = CcmParams::from_lambda_sq_integer(cutoff, 1);
            let length = log_lambda_sq_hp(&params, precision).unwrap();
            let weights = [1, -4, 1].map(|x| Float::with_val(precision, x));
            let certificate_source =
                CertifiedSecularFunction::from_integer_ccm_state(cutoff, 1, &weights, precision)
                    .unwrap();
            let pole = &certificate_source.poles()[2];
            assert_eq!(pole.lower(), pole.upper());
            let mut square = pole.lower().to_rational().unwrap();
            square.square_mut();
            // For residues (1,-4,1) at (-s,0,s), multiplication by
            // t*(t^2-s^2) gives the exact numerator 4*s^2 - 2*t^2.
            let expected = vec![square * 4, Rational::from(-2)];
            let discovered = positive_roots(1, &length, &weights, precision).unwrap();
            assert_eq!(discovered.coefficients, expected);
            assert_eq!(discovered.values.len(), 1);
        }
    }

    #[test]
    fn complete_height_selection_matches_the_shared_exact_source() {
        let precision = 64;
        let params = CcmParams::from_lambda_sq_integer(2, 1);
        let length = log_lambda_sq_hp(&params, precision).unwrap();
        let weights = [1, -4, 1].map(|x| Float::with_val(precision, x));
        // Independent rational witness in fresh_ccm_spacing_audit.rs proves
        // sqrt(2)*shared_spacing < upper < sqrt(2)*historical_staged_spacing.
        let target = ZeroTarget::HeightWindow {
            lower: "1".into(),
            upper: "12.8194503642625241484".into(),
        };
        let selected = plan(
            &params,
            &length,
            &weights,
            &target,
            IndependentRootDiscoveryOptions::complete_positive(true),
            precision,
        )
        .unwrap();
        assert_eq!(selected.selected_positions, vec![0]);
        assert_eq!(selected.result_first_root_index, 1);
    }

    fn source(weights: &[f64]) -> (Vec<Float>, Vec<Float>) {
        let n = weights.len() / 2;
        (
            (-(n as i64)..=n as i64)
                .map(|j| Float::with_val(256, j))
                .collect(),
            weights.iter().map(|w| Float::with_val(256, w)).collect(),
        )
    }

    fn half_root_plan(
        lower: String,
        upper: String,
        p: u32,
    ) -> Result<IndependentRootDiscoveryPlan> {
        let params = CcmParams::from_lambda_sq_integer(13, 1);
        let length = Float::with_val(p, rug::float::Constant::Pi) * 2;
        let weights = [3, 2, 3].map(|x| Float::with_val(p, x));
        plan(
            &params,
            &length,
            &weights,
            &ZeroTarget::HeightWindow { lower, upper },
            IndependentRootDiscoveryOptions::complete_positive(true),
            p,
        )
    }

    #[test]
    fn exhaustive_complete_boundary_exact_endpoint_inclusion_and_invalid_literals() {
        for p in [64, 128, 256] {
            assert_eq!(
                half_root_plan("0.1".into(), "0.5".into(), p)
                    .unwrap()
                    .selected_positions,
                vec![0]
            );
            assert!(half_root_plan("0.5".into(), "0.9".into(), p)
                .unwrap()
                .selected_positions
                .is_empty());
            assert_eq!(
                half_root_plan("+1e-1".into(), "+5e-1".into(), p)
                    .unwrap()
                    .selected_positions,
                vec![0]
            );
        }
        for (lower, upper) in [
            ("NaN", "1"),
            ("0", "1"),
            ("-1", "1"),
            ("0.5", "0.5"),
            ("1", "0.5"),
            ("0.1", "inf"),
            ("0.1", "1e1000001"),
        ] {
            assert!(half_root_plan(lower.into(), upper.into(), 128).is_err());
        }
    }
    #[test]
    fn exhaustive_complete_boundary_irrational_root_side_matches_exact_square() {
        use rug::{float::Round, ops::Pow, Integer};
        let denominator = Integer::from(10).pow(100);
        let x = (Float::with_val(1024, 1) / 3u32).sqrt() * Float::with_val(1024, &denominator);
        let integer = x.to_integer_round(Round::Down).unwrap().0;
        let lo = Rational::from((integer.clone(), denominator.clone()));
        let hi = Rational::from((Integer::from(&integer + 1), denominator));
        assert!(Rational::from(&lo * &lo) < Rational::from((1, 3)));
        assert!(Rational::from(&hi * &hi) > Rational::from((1, 3)));
        let lower = format!("{integer}e-100");
        let upper = format!("{}e-100", Integer::from(&integer + 1));
        let params = CcmParams::from_lambda_sq_integer(13, 1);
        for sign in [-1, 1] {
            let p = 128;
            let length = Float::with_val(p, rug::float::Constant::Pi) * 2;
            let weights = [sign; 3].map(|x| Float::with_val(p, x));
            let run = |a: String, b: String| {
                plan(
                    &params,
                    &length,
                    &weights,
                    &ZeroTarget::HeightWindow { lower: a, upper: b },
                    IndependentRootDiscoveryOptions::complete_positive(true),
                    p,
                )
                .unwrap()
            };
            assert_eq!(
                run(lower.clone(), upper.clone()).selected_positions,
                vec![0]
            );
            assert!(run("0.1".into(), lower.clone())
                .selected_positions
                .is_empty());
            assert!(run(upper.clone(), "0.9".into())
                .selected_positions
                .is_empty());
        }
    }
    #[test]
    fn exhaustive_complete_boundary_preflights_large_exponents_and_source_domain() {
        let poles = [-1, 0, 1].map(|x| Float::with_val(128, x));
        let huge = Float::with_val(128, 1) << 70_000_000u32;
        assert!(numerator(&poles, &[huge.clone(), huge.clone(), huge]).is_err());
        let weights = [3, 2, 3].map(|x| Float::with_val(128, x));
        for p in [0, 1, 63, 1_000_001, u32::MAX] {
            assert!(discover_points(&poles, &weights, p).is_err());
        }
        for length in [
            Float::with_val(128, 0),
            Float::with_val(128, -1),
            Float::with_val(128, rug::float::Special::Nan),
            Float::with_val(128, rug::float::Special::Infinity),
        ] {
            assert!(positive_roots(1, &length, &weights, 128).is_err());
        }
        assert!(positive_roots(usize::MAX, &Float::with_val(128, 1), &weights, 128).is_err());
        assert!(positive_roots(2, &Float::with_val(128, 1), &weights, 128).is_err());
    }
    #[test]
    fn exhaustive_complete_boundary_partition_matches_known_polynomial_roots() {
        let (poles, weights) = source(&[2.5, -20., 36., -20., 2.5]);
        let discovered = discover_points(&poles, &weights, 128).unwrap();
        // Reduced polynomial roots are exactly 9 and 16, hence heights 3 and 4.
        for (numerator, denominator, expected) in
            [(1, 2, 0), (3, 1, 1), (7, 2, 1), (4, 1, 2), (9, 2, 2)]
        {
            assert_eq!(
                discovered
                    .partition_at_most(&Rational::from((numerator, denominator)))
                    .unwrap(),
                expected
            );
        }
        assert_eq!(
            discovered
                .values
                .iter()
                .map(Float::prec)
                .collect::<Vec<_>>(),
            vec![128, 128]
        );
    }
    #[test]
    fn exhaustive_complete_boundary_excludes_root_above_exact_upper() {
        let upper = format!("0.4{}", "9".repeat(150));
        let result = half_root_plan("0.1".into(), upper, 128).unwrap();
        assert!(
            result.selected_positions.is_empty(),
            "root 1/2 lies above the exact decimal upper endpoint"
        );
    }
    #[test]
    fn exhaustive_complete_boundary_narrow_decimal_window_is_not_collapsed() {
        let result = half_root_plan(
            format!("0.4{}", "9".repeat(150)),
            format!("0.5{}1", "0".repeat(149)),
            128,
        );
        assert!(
            result.is_ok(),
            "distinct exact boundaries collapsed after MPFR rounding: {result:?}"
        );
        assert_eq!(result.unwrap().selected_positions, vec![0]);
    }
    #[test]
    fn exhaustive_complete_boundary_supports_64_bit_source() {
        let result = half_root_plan("0.1".into(), "0.9".into(), 64);
        assert!(
            result.is_ok(),
            "supported 64-bit source rejected by internal isolation precision: {result:?}"
        );
        assert_eq!(result.unwrap().selected_positions, vec![0]);
    }
    #[test]
    fn exhaustive_complete_boundary_zero_mode_has_no_movable_root() {
        let params = CcmParams::from_lambda_sq_integer(13, 0);
        let result = plan(
            &params,
            &Float::with_val(128, 2),
            &[Float::with_val(128, 1)],
            &ZeroTarget::FirstK { count: 1 },
            IndependentRootDiscoveryOptions::complete_positive(true),
            128,
        );
        assert!(
            result.is_ok(),
            "nonzero one-pole source has an empty movable-root set: {result:?}"
        );
        assert!(result.unwrap().artifact_seeds.is_empty());
    }
    #[test]
    fn exhaustive_complete_boundary_rejects_zero_precision_without_panicking() {
        let params = CcmParams::from_lambda_sq_integer(13, 1);
        let weights = [3, 2, 3].map(|x| Float::with_val(128, x));
        let result = std::panic::catch_unwind(|| {
            plan(
                &params,
                &Float::with_val(128, 2),
                &weights,
                &ZeroTarget::FirstK { count: 1 },
                IndependentRootDiscoveryOptions::complete_positive(true),
                0,
            )
        });
        assert!(result.is_ok_and(|value| value.is_err()));
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

/// Check nearest-seed cells against the supplied ordered seeds only. End cells
/// are unbounded and a single seed supplies no positional bound. This consistency
/// check is not root isolation, ordinal proof, or completeness evidence; those
/// claims require the separate exact-source discovery/certificate route.
pub(super) fn validate_assignment(roots: &[EigenvalueResult], seeds: &[Float]) -> Result<()> {
    if roots.len() != seeds.len() || seeds.windows(2).any(|s| s[0] >= s[1]) {
        bail!("complete root assignment needs matching counts and strictly increasing seeds");
    }
    boundary::rational_budget(
        seeds
            .iter()
            .chain(roots.iter().filter_map(EigenvalueResult::value)),
        seeds.len(),
    )?;
    let seeds = seeds
        .iter()
        .map(|v| v.to_rational().expect("validated finite seed"))
        .collect::<Vec<_>>();
    for (j, root) in roots.iter().enumerate() {
        let Some(value) = root.value() else {
            continue;
        };
        let value = value.to_rational().expect("validated finite root");
        let distance = (value.clone() - &seeds[j]).abs();
        for neighbor in [j.checked_sub(1), (j + 1 < seeds.len()).then_some(j + 1)]
            .into_iter()
            .flatten()
        {
            if distance >= (value.clone() - &seeds[neighbor]).abs() {
                bail!(
                    "complete root ordinal {} crossed its exact nearest-seed cell during refinement",
                    j + 1
                );
            }
        }
    }
    Ok(())
}
