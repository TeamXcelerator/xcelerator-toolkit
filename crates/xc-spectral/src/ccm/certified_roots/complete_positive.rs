//! Explicit whole-positive certification of one unchanged stored point source.
//!
//! This route is separate from the legacy pole-span certificate. An exact Cauchy
//! coefficient bound covers every numerator zero, including zeros beyond the
//! largest pole. FLINT/Arb counts/isolation and replayable interval Newton then
//! certify all positive movable roots before selecting the requested ordinals.
use super::*;

/// Certify a positive ordinal prefix/range of the exact retained CCM source,
/// including roots beyond its largest pole. The certificate's complete window
/// retains every positive movable root; selected ordinals never change the count.
/// No fixed carrier roots, analytic assembly error, or continuum claim is included.
/// Existing pole-span APIs and their artifact identities are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn certify_production_complete_positive_ccm_roots_at_cutoff(
    weights: &[Float],
    cutoff: &str,
    modes: usize,
    target: &IndependentCcmRootTarget,
    precision_bits: u32,
    isolation_bits: u32,
    interval_newton: &IntervalNewtonOptions,
) -> Result<ProductionIndependentCcmRootCertificate> {
    let source = independent_source(weights, cutoff, modes, precision_bits, isolation_bits)?;
    certify_with_source(
        &source,
        weights,
        cutoff,
        modes,
        target,
        precision_bits,
        isolation_bits,
        interval_newton,
    )
}

/// For a nonconstant P(x)=a_d*x^d+...+a_0, every complex zero satisfies
/// |x| < 1+max_{j<d}|a_j/a_d|. Enlarge the exact rational bound beyond every pole
/// as well, so a nonzero residue prevents the boundary from being a removable
/// denominator artifact. Leading zero coefficients are excluded before division.
fn positive_bound(source: &CertifiedSecularFunction) -> Result<Rational> {
    let (poles, coefficients) = source.exact_numerator_data()?;
    let degree = coefficients
        .iter()
        .rposition(|coefficient| coefficient != &0)
        .context("complete-positive source numerator is identically zero")?;
    if degree == 0 {
        bail!("complete-positive source numerator has no roots");
    }
    // Preserve the existing exact-numerator admission and bound the extra exact
    // ratios before division. These are scalar scratch, never a second polynomial.
    let maximum_coefficient_bits = coefficients
        .iter()
        .map(|coefficient| {
            u128::from(coefficient.numer().significant_bits())
                + u128::from(coefficient.denom().significant_bits())
        })
        .max()
        .unwrap_or(0);
    let maximum_pole_bits = poles
        .iter()
        .map(|pole| {
            u128::from(pole.numer().significant_bits())
                + u128::from(pole.denom().significant_bits())
        })
        .max()
        .unwrap_or(0);
    // A ratio's numerator/denominator widths sum to no more than the two
    // coefficient widths. Allow the usual scalar-arithmetic scratch factor,
    // including comparisons and +1, without imposing a new polynomial cap.
    let scratch_bytes = (2 * maximum_coefficient_bits)
        .max(maximum_pole_bits)
        .checked_add(2)
        .context("Cauchy scalar bit-width overflow")?
        .div_ceil(8)
        .checked_add(96)
        .context("Cauchy scalar byte-width overflow")?
        .checked_mul(64)
        .context("Cauchy scalar scratch overflow")?;
    let _reservation = xc_core::RunResources::current()
        .as_ref()
        .map(|resources| {
            resources.try_reserve_working(
                u64::try_from(scratch_bytes)
                    .map_err(|_| xc_core::ConfigError::new("Cauchy scalar scratch exceeds u64"))?,
            )
        })
        .transpose()?;
    let leading = coefficients[degree].clone().abs();
    let mut maximum = Rational::from(0);
    for coefficient in &coefficients[..degree] {
        let ratio = coefficient.clone().abs() / &leading;
        if ratio > maximum {
            maximum = ratio;
        }
    }
    let mut bound = maximum + 1;
    for pole in poles {
        let outside_pole = pole.clone().abs() + 1;
        if outside_pole > bound {
            bound = outside_pole;
        }
    }
    Ok(bound)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn certify_with_source(
    source: &CertifiedSecularFunction,
    weights: &[Float],
    cutoff: &str,
    modes: usize,
    target: &IndependentCcmRootTarget,
    precision_bits: u32,
    isolation_bits: u32,
    interval_newton: &IntervalNewtonOptions,
) -> Result<ProductionIndependentCcmRootCertificate> {
    let (first, last) = match target {
        IndependentCcmRootTarget::Prefix { count } if *count > 0 => (1, *count),
        IndependentCcmRootTarget::IndexRange { first, last } if *first > 0 && first <= last => {
            (*first, *last)
        }
        _ => bail!(
            "complete-positive certification requires a nonempty positive ordinal prefix/range"
        ),
    };
    let lower = Rational::from(0);
    let upper = positive_bound(source)?;
    let window = certify_tight_window(source, &lower, &upper, isolation_bits, interval_newton)?;
    if window.count.certified_root_count != window.roots.len()
        || !window.reconciliation.complete
        || !window.count.square_free
    {
        bail!("complete-positive finite source count did not reconcile");
    }
    if last > window.roots.len() {
        bail!(
            "complete finite source has {} positive movable roots; requested ordinal {last}",
            window.roots.len()
        );
    }
    let source_weights = weights
        .iter()
        .map(|weight| serialize_float(&Float::with_val(precision_bits, weight), precision_bits))
        .collect::<Vec<_>>();
    let integer_cutoff_c = cutoff_floor(cutoff)?;
    Ok(ProductionIndependentCcmRootCertificate {
        schema_version: 2,
        integer_cutoff_c,
        lambda_squared_decimal: (cutoff != integer_cutoff_c.to_string()).then(|| cutoff.to_owned()),
        modes,
        precision_bits,
        isolation_bits,
        interval_newton: interval_newton.clone(),
        target: target.clone(),
        lower_bound: lower.to_string(),
        upper_bound: upper.to_string(),
        positive_roots_before_window: 0,
        selected_root_offset: first - 1,
        selected_root_count: last - first + 1,
        first_selected_positive_index: Some(first),
        last_selected_positive_index: Some(last),
        source_weights_digest: weights_digest(&source_weights)?,
        source_certification_scope: FiniteSourceCertificationScope::ExactStoredPointSource,
        source_weights,
        window,
        reference_seeds_used: false,
        discovery_method: COMPLETE_POSITIVE_DISCOVERY_METHOD.to_owned(),
        count_method: COMPLETE_POSITIVE_COUNT_METHOD.to_owned(),
        finite_model_statement: "certified independently indexed positive movable roots of one exact stored finite CCM secular source; exact Cauchy coefficient bound covers all numerator roots; fixed carrier zeros and analytic assembly uncertainty are not included; no external reference data used".to_owned(),
    })
}

/// Retain the Arb isolation scale rather than inflating every candidate by the
/// requested maximum width. Exact pole/window clearances bound the padding;
/// promotion recomputes isolation from the same exact numerator when necessary.
fn tight_candidate_box(
    source: &CertifiedSecularFunction,
    candidate: &MpfrInterval,
    lower_bound: &Rational,
    upper_bound: &Rational,
    work: u32,
) -> Result<MpfrInterval> {
    let a = candidate
        .lower()
        .to_rational()
        .context("finite candidate lower")?;
    let b = candidate
        .upper()
        .to_rational()
        .context("finite candidate upper")?;
    if a <= *lower_bound || b >= *upper_bound || a > b {
        bail!("Arb candidate does not establish strict window membership");
    }
    let mut clearance = a.clone() - lower_bound;
    clearance = clearance.min(upper_bound.clone() - &b);
    for pole in &source.poles {
        let point = pole.lower().to_rational().context("finite point pole")?;
        if point >= a && point <= b {
            bail!("Arb candidate overlaps a retained pole at this proof precision");
        }
        clearance = clearance.min(if point < a {
            a.clone() - point
        } else {
            point - &b
        });
    }
    let exponent = candidate
        .lower()
        .get_exp()
        .unwrap_or(0)
        .max(candidate.upper().get_exp().unwrap_or(0));
    let shift: i32 = (i64::from(exponent) - i64::from(work) + 2)
        .try_into()
        .context("candidate padding exponent exceeds i32")?;
    let ulps = (Float::with_val(work, 1) << shift)
        .to_rational()
        .context("finite proof padding")?;
    let maximum_padding: Rational = clearance / 4;
    let mut padding = (b.clone() - &a).max(ulps).min(maximum_padding.clone());
    // A very tight Arb box can be narrower than the cancellation error in a
    // directed secular evaluation. Enlarge only from its directed Newton image,
    // with exact separation caps, then require the ordinary replayable witness.
    for _ in 0..4 {
        let lower = Float::with_val_round(work, a.clone() - &padding, Round::Down).0;
        let upper = Float::with_val_round(work, b.clone() + &padding, Round::Up).0;
        let box_ = MpfrInterval::new(lower, upper)?;
        if source.overlaps_pole(&box_)
            || box_.lower().to_rational().unwrap() <= *lower_bound
            || box_.upper().to_rational().unwrap() >= *upper_bound
        {
            bail!("outward-rounded candidate box lacks strict pole/window clearance");
        }
        let derivative = source.derivative_interval(&box_)?;
        if derivative.contains_zero() {
            bail!("candidate derivative contains zero at this proof precision");
        }
        let midpoint = box_.midpoint_point();
        let image = midpoint.sub(&source.evaluate_interval(&midpoint)?.div(&derivative)?);
        if image.is_interior_subset_of(&box_) {
            return Ok(box_);
        }
        let needed = (a.clone() - image.lower().to_rational().context("finite Newton image")?)
            .max(image.upper().to_rational().context("finite Newton image")? - &b)
            .max(padding.clone())
            * 2;
        if needed > maximum_padding {
            bail!("conditioned Newton box exceeds exact pole/root/window separation at this proof precision");
        }
        padding = needed;
    }
    bail!("conditioned candidate box has not achieved directed Newton self-inclusion")
}

pub(super) fn certify_tight_window(
    source: &CertifiedSecularFunction,
    lower_bound: &Rational,
    upper_bound: &Rational,
    isolation_bits: u32,
    options: &IntervalNewtonOptions,
) -> Result<SecularWindowCertificate> {
    boundary::isolation(isolation_bits, source.precision_bits())?;
    options.validate(source.precision_bits())?;
    let (_, numerator) = source.exact_numerator_data()?;
    let isolate = |work: u32| -> Result<Vec<MpfrInterval>> {
        let (mut candidates, square_free) = crate::ccm::arb_bridge::rational_polynomial_real_roots(
            numerator,
            lower_bound,
            upper_bound,
            work,
        )?;
        if !square_free {
            bail!("complete-positive isolation requires a square-free polynomial");
        }
        candidates.sort_by(|a, b| a.lower().partial_cmp(b.lower()).unwrap());
        Ok(candidates)
    };
    let base = source.precision_bits();
    xc_core::progress_message!(
        "[HP] complete-positive certificate: exact count and isolation at {base} bits started"
    );
    let mut initial_pair = None;
    let mut initial_failure = String::new();
    for guard in [0_u32, 64, 256, 1024, 4096] {
        let work = base.saturating_add(guard).min(1_000_000);
        let attempt = (|| -> Result<_> {
            let candidates = isolate(work)?;
            let count = source
                .with_precision(work)?
                .exact_flint_numerator_count_in_window(lower_bound, upper_bound)?;
            Ok((work, candidates, count))
        })();
        match attempt {
            Ok(pair) => {
                initial_pair = Some(pair);
                break;
            }
            Err(error) if crate::ccm::arb_bridge::isolation_needs_more_precision(&error) => {
                initial_failure = format!("precision {work}: {error}");
                xc_core::progress_message!("[HP] tight finite-root certificate: isolation unresolved at {work} bits; retaining exact source for next proof precision");
            }
            Err(error) => return Err(error),
        }
    }
    let (initial_work, initial, count) = initial_pair.context(format!(
        "finite-root isolation unresolved within proof budget: {initial_failure}"
    ))?;
    if !count.square_free || count.certified_root_count != initial.len() {
        bail!("complete-positive exact count and isolation did not reconcile");
    }
    let total = initial.len();
    let mut isolated = std::collections::BTreeMap::from([(initial_work, initial)]);
    let target_width = Float::with_val(base, 2).pow(-(isolation_bits as i32));
    let mut roots = Vec::with_capacity(total);
    for index in 0..total {
        let mut failure = String::new();
        let mut accepted = None;
        for guard in [0_u32, 64, 256, 1024, 4096] {
            let work = base.saturating_add(guard).min(1_000_000);
            if work < initial_work {
                continue;
            }
            if let std::collections::btree_map::Entry::Vacant(entry) = isolated.entry(work) {
                xc_core::progress_message!("[HP] complete-positive certificate: root {}/{total}, refining exact-source isolation at {work} bits (guard={guard})", index + 1);
                let candidates = match isolate(work) {
                    Ok(candidates) => candidates,
                    Err(error)
                        if crate::ccm::arb_bridge::isolation_needs_more_precision(&error) =>
                    {
                        failure = format!("proof precision {work}: {error}");
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                if candidates.len() != total {
                    bail!("promoted complete-positive isolation changed the exact root count");
                }
                entry.insert(candidates);
            }
            let candidate = &isolated[&work][index];
            if candidate.width() > target_width {
                failure = format!(
                    "proof precision {work}: Arb candidate exceeds the requested isolation width"
                );
                continue;
            }
            let promoted = source.with_precision(work)?;
            let separated_lower = if index == 0 {
                lower_bound.clone()
            } else {
                lower_bound.clone().max(
                    isolated[&work][index - 1]
                        .upper()
                        .to_rational()
                        .context("finite preceding root box")?,
                )
            };
            let separated_upper = if index + 1 == total {
                upper_bound.clone()
            } else {
                upper_bound.clone().min(
                    isolated[&work][index + 1]
                        .lower()
                        .to_rational()
                        .context("finite following root box")?,
                )
            };
            let box_ = match tight_candidate_box(
                &promoted,
                candidate,
                &separated_lower,
                &separated_upper,
                work,
            ) {
                Ok(box_) => box_,
                Err(error) => {
                    failure = format!("proof precision {work}: {error}");
                    continue;
                }
            };
            let root = promoted.isolate(&box_, options)?;
            if root.status != IntervalRootStatus::CertifiedUnique {
                failure = format!(
                    "proof precision {work}: interval-Newton uniqueness remains {:?}",
                    root.status
                );
                continue;
            }
            let (lower, upper) = exact_root_bounds(&root)?;
            if lower <= *lower_bound || upper >= *upper_bound {
                failure = format!("proof precision {work}: certified enclosure does not establish strict open-window membership");
                continue;
            }
            if upper - lower > target_width.to_rational().unwrap() {
                failure = format!("proof precision {work}: certified enclosure exceeds the requested isolation width");
                continue;
            }
            accepted = Some(root);
            break;
        }
        roots.push(accepted.with_context(|| {
            format!(
                "complete-positive root {} proof unresolved: {failure}",
                index + 1
            )
        })?);
        if index == 0 || (index + 1) % 10 == 0 || index + 1 == total {
            xc_core::progress_message!(
                "[HP] complete-positive certificate: {}/{total} root witnesses complete",
                index + 1
            );
        }
    }
    let reconciliation = reconcile_complete_window(&roots, &count)?;
    if !reconciliation.complete {
        bail!(
            "complete-positive witnesses did not reconcile: {}",
            reconciliation.reason
        );
    }
    Ok(SecularWindowCertificate {
        roots,
        count,
        reconciliation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_core::DecimalLiteral;

    fn options() -> IntervalNewtonOptions {
        IntervalNewtonOptions {
            width_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 100,
        }
    }

    #[test]
    fn outer_positive_root_is_counted_and_replayed_without_reference_data() {
        let p = 192;
        // R(z)=2z/(z^2-a^2)-3/z has the sole positive root sqrt(3)*a,
        // strictly beyond the largest pole a.
        let weights = [1, -3, 1].map(|w| Float::with_val(p, w));
        let target = IndependentCcmRootTarget::Prefix { count: 1 };
        assert!(certify_production_independent_ccm_roots_at_cutoff(
            &weights,
            "13",
            1,
            &target,
            p,
            64,
            &options()
        )
        .is_err());
        let certificate = certify_production_complete_positive_ccm_roots_at_cutoff(
            &weights,
            "13",
            1,
            &target,
            p,
            64,
            &options(),
        )
        .unwrap();
        let source = independent_source(&weights, "13", 1, p, 64).unwrap();
        assert!(
            exact_root_bounds(&certificate.window.roots[0]).unwrap().0
                > source.poles[2].upper().to_rational().unwrap()
        );
        assert_eq!(certificate.window.count.certified_root_count, 1);
        let (_, numerator) = source.exact_numerator_data().unwrap();
        let oracle = exact_sturm_root_count(
            numerator,
            Rational::from(0),
            positive_bound(&source).unwrap(),
        )
        .unwrap();
        assert_eq!(oracle.distinct_real_roots, 1);
        validate_production_independent_ccm_root_certificate_structure(&certificate).unwrap();
        verify_production_independent_ccm_root_certificate(&certificate).unwrap();
        let mut wrong_count = certificate.clone();
        wrong_count.window.count.certified_root_count = 0;
        assert!(verify_production_independent_ccm_root_certificate(&wrong_count).is_err());
        let mut wrong_bound = certificate;
        wrong_bound.upper_bound = "1000000".into();
        assert!(verify_production_independent_ccm_root_certificate(&wrong_bound).is_err());
    }

    #[test]
    fn mixed_inner_outer_prefix_keeps_total_count_and_rejects_shortfall() {
        let p = 192;
        // With poles -2a,-a,0,a,2a, numerator is
        // 16*(z^2-a^2/4)*(z^2-9a^2). All coefficients are exact dyadics.
        let weights = [-12.5, 16.0, 9.0, 16.0, -12.5].map(|w| Float::with_val(p, w));
        let all = certify_production_complete_positive_ccm_roots_at_cutoff(
            &weights,
            "13",
            2,
            &IndependentCcmRootTarget::Prefix { count: 2 },
            p,
            64,
            &options(),
        )
        .unwrap();
        assert_eq!(all.window.roots.len(), 2);
        let second = certify_production_complete_positive_ccm_roots_at_cutoff(
            &weights,
            "13",
            2,
            &IndependentCcmRootTarget::IndexRange { first: 2, last: 2 },
            p,
            64,
            &options(),
        )
        .unwrap();
        assert_eq!(second.window.count.certified_root_count, 2);
        assert_eq!(second.selected_root_offset, 1);
        assert_eq!(second.try_selected_roots().unwrap(), &all.window.roots[1..]);
        verify_production_independent_ccm_root_certificate(&second).unwrap();
        assert!(certify_production_complete_positive_ccm_roots_at_cutoff(
            &weights,
            "13",
            2,
            &IndependentCcmRootTarget::Prefix { count: 3 },
            p,
            64,
            &options()
        )
        .is_err());
        let mut missing_inner = second;
        missing_inner.window.roots.remove(0);
        assert!(verify_production_independent_ccm_root_certificate(&missing_inner).is_err());
    }

    #[test]
    fn removable_poles_origin_and_repeated_root_are_refused() {
        let p = 192;
        let target = IndependentCcmRootTarget::Prefix { count: 1 };
        // A zero residue makes its pole removable; the unreduced numerator then
        // contains that pole as a factor. In particular w_0=0 puts a factor at
        // the origin. The source constructor refuses these cases before any
        // root count; a fixed carrier/pole cannot become a movable root.
        for values in [[1, 0, 1], [0, -3, 1], [1, -3, 0], [0, 0, 0]] {
            let weights = values.map(|w| Float::with_val(p, w));
            let error = certify_production_complete_positive_ccm_roots_at_cutoff(
                &weights,
                "13",
                1,
                &target,
                p,
                64,
                &options(),
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("residue intervals must exclude zero"));
        }
        // With poles -a,0,a these exact weights give (z-2a)^2.
        // Repeated roots do not support the required simple-root witness.
        let repeated = [4.5, -4.0, 0.5].map(|w| Float::with_val(p, w));
        assert!(certify_production_complete_positive_ccm_roots_at_cutoff(
            &repeated,
            "13",
            1,
            &target,
            p,
            64,
            &options()
        )
        .is_err());
        // Leading cancellation leaves only -3*a*z-a^2, whose sole root is
        // negative. A positive request must fail with an actual zero count.
        let negative = [1, 1, -2].map(|w| Float::with_val(p, w));
        let error = certify_production_complete_positive_ccm_roots_at_cutoff(
            &negative,
            "13",
            1,
            &target,
            p,
            64,
            &options(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("has 0 positive movable roots"));
    }

    #[test]
    fn stored_source_precision_and_scalar_scratch_are_bound() {
        let p = 192;
        let weights = [1, -3, 1].map(|w| Float::with_val(p, w));
        let target = IndependentCcmRootTarget::Prefix { count: 1 };
        let certificate = certify_production_complete_positive_ccm_roots_at_cutoff(
            &weights,
            "13",
            1,
            &target,
            p,
            64,
            &options(),
        )
        .unwrap();
        assert_eq!(
            certificate.source_weights_digest,
            production_ccm_source_weights_digest(&weights, p).unwrap()
        );
        // Values not exactly representable at the declared source precision
        // become outward intervals and cannot masquerade as exact point data.
        let mut higher = weights.each_ref().map(|w| Float::with_val(p + 64, w));
        higher[0] += Float::with_val(p + 64, 2).pow(-((p + 8) as i32));
        assert!(certify_production_complete_positive_ccm_roots_at_cutoff(
            &higher,
            "13",
            1,
            &target,
            p,
            64,
            &options()
        )
        .is_err());
        let source = independent_source(&weights, "13", 1, p, 64).unwrap();
        // Admit the already-built numerator first, then check that the new
        // bound obeys a caller's live scratch budget and releases ownership.
        source.exact_numerator_data().unwrap();
        let resources = xc_core::RunResources::new(1).unwrap();
        let _scope = resources.enter();
        assert!(positive_bound(&source).is_err());
    }

    #[test]
    fn tight_witness_preserves_close_roots_and_near_pole_clearance() {
        let p = 512;
        let delta = Float::with_val(p, 2).pow(-160);
        // Two simple roots at 2a and (2+delta)a. The legacy 2^-96 padding
        // contains both roots and cannot establish a unique-root witness.
        let mut left = Float::with_val(p, &delta * 1.5);
        left += 4.5;
        let mut center = Float::with_val(p, &delta * -2);
        center -= 4;
        let mut right = Float::with_val(p, &delta / 2);
        right += 0.5;
        let clustered = [left, center, right];
        let source = independent_source(&clustered, "13", 1, p, 96).unwrap();
        let upper = positive_bound(&source).unwrap();
        assert!(source
            .certify_flint_numerator_rational_window(&Rational::from(0), &upper, 96, &options())
            .is_err());
        let certificate = certify_production_complete_positive_ccm_roots_at_cutoff(
            &clustered,
            "13",
            1,
            &IndependentCcmRootTarget::Prefix { count: 2 },
            p,
            96,
            &options(),
        )
        .unwrap();
        assert_eq!(certificate.window.roots.len(), 2);
        verify_production_independent_ccm_root_certificate(&certificate).unwrap();
        // A tiny nonzero residue gives a genuine root just outside the last
        // pole. It must remain distinct from the fixed carrier/pole itself.
        let near_pole = [Float::with_val(p, 1), Float::with_val(p, -3), delta];
        let source = independent_source(&near_pole, "13", 1, p, 96).unwrap();
        assert!(source
            .certify_flint_numerator_rational_window(
                &Rational::from(0),
                &positive_bound(&source).unwrap(),
                96,
                &options()
            )
            .is_err());
        let certificate = certify_production_complete_positive_ccm_roots_at_cutoff(
            &near_pole,
            "13",
            1,
            &IndependentCcmRootTarget::Prefix { count: 1 },
            p,
            96,
            &options(),
        )
        .unwrap();
        let (lower, _) = exact_root_bounds(&certificate.window.roots[0]).unwrap();
        assert!(lower > source.poles[2].upper().to_rational().unwrap());
        verify_production_independent_ccm_root_certificate(&certificate).unwrap();
    }
}
