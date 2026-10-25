//! Complete distinct movable-root enumeration in an exact requested window.
//! Exact rational coefficients bind every nonzero residue and distinct pole.
//! Unresolved multiplicity, output rounding or resource limits return errors.
use crate::ccm::certified_roots::boundary;
use anyhow::{bail, Result};
#[cfg(any(test, not(feature = "arb")))]
use rug::Integer;
use rug::{Float, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;

fn coefficient_bits(values: &[Rational]) -> u128 {
    values
        .iter()
        .map(|x| {
            u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits())
        })
        .sum()
}

/// One admission covers construction, exact endpoint arithmetic, and every
/// native isolation retry. The reservation must outlive the rational vectors.
pub(super) struct DiscoveryWorkspace {
    coefficient_bits: u128,
    pub(super) bytes: u128,
    _reservation: Option<xc_core::MemoryReservation>,
}

fn dyadic_denominator_bits(value: &Float) -> u128 {
    let Some(significand) = value.get_significand() else {
        return 0; // Finiteness was validated; the remaining case is zero.
    };
    (i64::from(significand.significant_bits())
        - i64::from(significand.find_one(0).expect("nonzero significand"))
        - i64::from(value.get_exp().expect("normal source point")))
    .max(0) as u128
}

fn reserve_workspace(bytes: u128) -> Result<Option<xc_core::MemoryReservation>> {
    let maximum = super::source_working_budget()?;
    if bytes > maximum {
        bail!(
            "exact discovery exceeds the coefficient workspace budget: estimated_bytes={bytes}, maximum_working_bytes={maximum}"
        );
    }
    xc_core::RunResources::current()
        .map(|resources| {
            let bytes = u64::try_from(bytes)?;
            Ok(resources.try_reserve_working(bytes)?)
        })
        .transpose()
}

impl DiscoveryWorkspace {
    pub(super) fn estimate(
        poles: &[Float],
        weights: &[Float],
        lower: &Rational,
        upper: &Rational,
        precision: u32,
    ) -> Result<Self> {
        // This validates finiteness and the existing per-point exponent guard
        // before converting any MPFR source point to a rational number.
        let points = boundary::rational_point_vector_workspace(poles.iter().chain(weights))?;
        let mut count = 0u128;
        let mut pole_denominator = 0u128;
        let mut pole_magnitude = 0u128;
        let mut weight_denominator = 0u128;
        let mut weight_magnitude = 0u128;
        for (pole, weight) in poles.iter().zip(weights).filter(|(_, w)| !w.is_zero()) {
            count += 1;
            let pole_exp = i64::from(pole.get_exp().unwrap_or(0));
            if !pole.is_zero() {
                pole_denominator += dyadic_denominator_bits(pole);
                pole_magnitude += pole_exp.max(0) as u128;
            }
            let weight_exp = i64::from(weight.get_exp().unwrap_or(0));
            weight_denominator = weight_denominator.max(dyadic_denominator_bits(weight));
            weight_magnitude = weight_magnitude.max(weight_exp.max(0) as u128);
        }
        if count == 0 {
            bail!("the secular point source is identically zero");
        }
        let count_bits = u128::from(u128::BITS - count.leading_zeros());
        let argument_bits = [lower, upper]
            .into_iter()
            .map(|q| {
                u128::from(q.numer().significant_bits()) + u128::from(q.denom().significant_bits())
            })
            .sum::<u128>();
        let argument_magnitude = [lower, upper]
            .into_iter()
            .map(|q| {
                (i64::from(q.numer().significant_bits()) - i64::from(q.denom().significant_bits())
                    + 1)
                .max(0) as u128
            })
            .max()
            .unwrap();
        // Every denominator coefficient divides 2^sum(d_i). Every numerator
        // coefficient divides 2^(sum(d_i)+max(e_i)), since all sources are
        // dyadic. FLINT's common denominator is this power of two, NOT the
        // product of the denominators of all polynomial coefficients.
        // prod(1+|pole|) bounds denominator coefficient magnitudes; the sum
        // of at most count weighted quotients adds weight and count bits.
        let denominator = pole_denominator + weight_denominator;
        let magnitude = pole_magnitude + count + weight_magnitude + count_bits + 4;
        // Exact division by a rational root preserves a dyadic polynomial's
        // common denominator (Gauss's lemma after clearing that denominator).
        // For s removed roots, the reciprocal product's kth coefficient is
        // bounded by binomial(s+k-1,k)*R^k, where R=max(1,|lower|,|upper|).
        // Here s+k <= degree, so all partial quotients have coefficient
        // magnitude at most (degree+1)*2^degree*R^degree times the original
        // coefficient bound. This avoids charging quadratic degree growth
        // for synthetic divisions that are exact.
        let endpoint_growth = count_bits + count * (1 + argument_magnitude);
        let scalar_bits = 2 * denominator + magnitude + endpoint_growth + 8;
        let coefficient_bits = (count + 1) * scalar_bits;
        // Exact Horner evaluation at a non-dyadic endpoint additionally needs
        // up to degree copies of its numerator/denominator bits in a scalar.
        let evaluation_bits = scalar_bits + 2 * count * argument_bits;
        // Arb uses a bounded number of polynomial/root vectors, not a retained
        // degree-long rational Sturm sequence. Its square-free test uses a
        // derivative and modular GCD for degree >= 5; complex_roots retains
        // two ball polynomials, a deflated integer polynomial, and root
        // vectors; find_roots uses 4*degree+1 approximation entries. Sixty-four
        // polynomial equivalents are a conservative policy allowance for
        // Rust/native overlap, decimal conversion scratch, and native
        // arithmetic/precision growth. The exact numerator's depth-first
        // product tree retains sibling pairs with at most a geometric sum
        // of degrees; a merge holds two child pairs, three products and its
        // parent pair. Its integer coefficients have no more bits than this
        // common-denominator bound, so it fits within the same allowance.
        // The portable Sturm route keeps its separate degree-dependent
        // allowance and existing size limits.
        // This is cooperative workspace admission, not an OS RSS guarantee.
        #[cfg(feature = "arb")]
        let polynomial_copies = 64;
        #[cfg(not(feature = "arb"))]
        let polynomial_copies = count + 32;
        let bytes = (coefficient_bits.div_ceil(8) + evaluation_bits.div_ceil(8) + 96 * (count + 1))
            * polynomial_copies
            + points
            + (count + 1) * (u128::from(precision).saturating_add(4096).div_ceil(8) + 96) * 8;
        Ok(Self {
            coefficient_bits,
            bytes,
            _reservation: None,
        })
    }

    pub(super) fn admit(mut self) -> Result<Self> {
        self._reservation = reserve_workspace(self.bytes)?;
        Ok(self)
    }

    pub(super) fn check(&self, values: &[Rational]) -> Result<()> {
        if coefficient_bits(values) > self.coefficient_bits {
            bail!("exact discovery coefficient growth exceeds its admitted workspace estimate");
        }
        Ok(())
    }
}

fn divide(
    values: &[Rational],
    root: &Rational,
    workspace: &DiscoveryWorkspace,
) -> Result<Vec<Rational>> {
    let degree = values.len() - 1;
    let mut out = vec![Rational::from(0); degree];
    out[degree - 1] = values[degree].clone();
    for j in (1..degree).rev() {
        out[j - 1] = values[j].clone() + Rational::from(root * &out[j]);
    }
    if values[0].clone() + Rational::from(root * &out[0]) != 0 {
        bail!("exact discovery synthetic division left a remainder");
    }
    workspace.check(&out)?;
    Ok(out)
}
#[cfg(test)]
fn numerator_sequential_oracle(
    poles: &[Float],
    weights: &[Float],
    workspace: &DiscoveryWorkspace,
) -> Result<Vec<Rational>> {
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
        workspace.check(&next)?;
        denominator = next;
    }
    let mut out = vec![Rational::from(0); active.len()];
    for (pole, weight) in &active {
        for (value, coefficient) in out.iter_mut().zip(divide(&denominator, pole, workspace)?) {
            *value += coefficient * weight;
        }
        workspace.check(&out)?;
    }
    while out.len() > 1 && out.last().is_some_and(|x| x == &0) {
        out.pop();
    }
    if out.iter().all(|x| x == &0) {
        bail!("the secular point source is identically zero");
    }
    Ok(out)
}

/// D = product(x - pole), N = sum(weight * D/(x - pole)). Clearing every
/// pole denominator and one common weight denominator makes all tree
/// arithmetic integral: (D1,N1),(D2,N2) -> (D1*D2,N1*D2+D1*N2).
/// The common weight denominator occurs once, including at every leaf.
#[cfg(any(test, not(feature = "arb")))]
fn numerator_integer_tree(
    poles: &[Float],
    weights: &[Float],
    weight_denominator_bits: usize,
) -> Result<Vec<Rational>> {
    struct Pair {
        denominator: Vec<Integer>,
        numerator: Vec<Integer>,
        pole_denominator_bits: usize,
    }
    fn multiply(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
        let mut out = vec![Integer::from(0); a.len() + b.len() - 1];
        for (j, x) in a.iter().enumerate().filter(|(_, x)| !x.is_zero()) {
            for (k, y) in b.iter().enumerate().filter(|(_, y)| !y.is_zero()) {
                out[j + k] += Integer::from(x * y);
            }
        }
        while out.len() > 1 && out.last().is_some_and(Integer::is_zero) {
            out.pop();
        }
        out
    }
    fn tree(points: &[(&Float, &Float)], weight_bits: usize, need_denominator: bool) -> Pair {
        if points.len() == 1 {
            let pole = points[0].0.to_rational().expect("validated pole");
            let weight = points[0].1.to_rational().expect("validated residue");
            let pole_bits = pole.denom().significant_bits() as usize - 1;
            let residue_bits = weight.denom().significant_bits() as usize - 1;
            return Pair {
                denominator: if need_denominator {
                    vec![-pole.numer().clone(), pole.denom().clone()]
                } else {
                    vec![]
                },
                numerator: vec![weight.numer().clone() << (pole_bits + weight_bits - residue_bits)],
                pole_denominator_bits: pole_bits,
            };
        }
        let middle = points.len() / 2;
        let left = tree(&points[..middle], weight_bits, true);
        let right = tree(&points[middle..], weight_bits, true);
        let denominator = if need_denominator {
            multiply(&left.denominator, &right.denominator)
        } else {
            vec![]
        };
        let mut numerator = multiply(&left.numerator, &right.denominator);
        let other = multiply(&left.denominator, &right.numerator);
        numerator.resize_with(numerator.len().max(other.len()), || Integer::from(0));
        for (coefficient, value) in numerator.iter_mut().zip(other) {
            *coefficient += value;
        }
        Pair {
            denominator,
            numerator,
            pole_denominator_bits: left.pole_denominator_bits + right.pole_denominator_bits,
        }
    }
    let active = poles
        .iter()
        .zip(weights)
        .filter(|(_, weight)| !weight.is_zero())
        .collect::<Vec<_>>();
    if active.is_empty() {
        bail!("the secular point source is identically zero");
    }
    // The root denominator itself is never consumed by root discovery.
    let result = tree(&active, weight_denominator_bits, false);
    let denominator = Integer::from(1) << (result.pole_denominator_bits + weight_denominator_bits);
    Ok(result
        .numerator
        .into_iter()
        .map(|value| Rational::from((value, denominator.clone())))
        .collect())
}

pub(super) fn numerator(
    poles: &[Float],
    weights: &[Float],
    workspace: &DiscoveryWorkspace,
) -> Result<Vec<Rational>> {
    let weight_bits = weights
        .iter()
        .filter(|weight| !weight.is_zero())
        .map(dyadic_denominator_bits)
        .max()
        .ok_or_else(|| anyhow::anyhow!("the secular point source is identically zero"))?;
    #[cfg(feature = "arb")]
    let mut out = {
        let pole_bits = poles
            .iter()
            .zip(weights)
            .filter(|(_, weight)| !weight.is_zero())
            .map(|(pole, _)| dyadic_denominator_bits(pole))
            .sum::<u128>();
        crate::ccm::arb_bridge::exact_secular_numerator(
            poles,
            weights,
            weight_bits,
            pole_bits + weight_bits,
        )?
    };
    #[cfg(not(feature = "arb"))]
    let mut out = numerator_integer_tree(poles, weights, usize::try_from(weight_bits)?)?;
    while out.len() > 1 && out.last().is_some_and(|x| x == &0) {
        out.pop();
    }
    if out.is_empty() || out.iter().all(|x| x == &0) {
        bail!("the secular point source is identically zero");
    }
    workspace.check(&out)?;
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
    let count =
        xc_numerics::interval::exact_sturm_root_count(coefficients, lower.clone(), upper.clone())?;
    if !count.square_free {
        bail!("exact discovery has unresolved repeated roots");
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
    let window_bytes = boundary::rational_point_vector_workspace([lower, upper].into_iter())?;
    let _window_reservation = reserve_workspace(window_bytes)?;
    let rational_lower = lower.to_rational().expect("finite bound");
    let rational_upper = upper.to_rational().expect("finite bound");
    roots_rational(
        poles,
        weights,
        &rational_lower,
        &rational_upper,
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
    if poles.windows(2).any(|x| x[0] >= x[1]) {
        bail!("exact discovery poles must be distinct and increasing");
    }
    let workspace = DiscoveryWorkspace::estimate(poles, weights, lower, upper, p)?.admit()?;
    let original = {
        let _stage = crate::ccm::capture_runtime::Stage::new(format!(
            "exact root discovery numerator ({} poles, {p} bits, {} admitted bytes)",
            poles.len(),
            workspace.bytes
        ));
        numerator(poles, weights, &workspace)?
    };
    let mut coefficients = original.clone();
    let mut endpoints = Vec::new();
    // Endpoint membership is decided exactly before point rounding or root
    // isolation. The interior polynomial has its boundary factors removed.
    for (bound, included) in [(lower, closed_lower), (upper, closed_upper)] {
        let mut multiplicity = 0;
        while coefficients.len() > 1
            && xc_numerics::interval::evaluate_real_polynomial_exact(&coefficients, bound)? == 0
        {
            coefficients = divide(&coefficients, bound, &workspace)?;
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
            let _stage = crate::ccm::capture_runtime::Stage::new(format!(
                "exact root discovery isolation (degree {}, {work} bits)",
                coefficients.len() - 1
            ));
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

    fn assert_numerator_matches_oracle(poles: &[Float], weights: &[Float], p: u32) {
        let workspace = DiscoveryWorkspace::estimate(
            poles,
            weights,
            &Rational::from(-17),
            &Rational::from(17),
            p,
        )
        .unwrap()
        .admit()
        .unwrap();
        let expected = numerator_sequential_oracle(poles, weights, &workspace).unwrap();
        let actual = numerator(poles, weights, &workspace).unwrap();
        let weight_bits = weights
            .iter()
            .filter(|weight| !weight.is_zero())
            .map(dyadic_denominator_bits)
            .max()
            .unwrap();
        let mut portable =
            numerator_integer_tree(poles, weights, usize::try_from(weight_bits).unwrap()).unwrap();
        while portable.len() > 1 && portable.last().is_some_and(|value| value == &0) {
            portable.pop();
        }
        assert_eq!(actual, expected);
        assert_eq!(portable, expected);
        // Canonical coefficient text feeds downstream exact polynomial
        // serialization, so preserve its byte identity as well as its value.
        assert_eq!(
            actual.iter().map(ToString::to_string).collect::<Vec<_>>(),
            expected.iter().map(ToString::to_string).collect::<Vec<_>>()
        );
    }

    #[test]
    fn discovery_integer_numerator_matches_sequential_dyadic_coefficients() {
        let p = 192;
        for count in [1, 2, 3, 8, 17, 32] {
            let poles = (0..count)
                .map(|j| Float::with_val(p, 2 * j - count) >> (j % 7))
                .collect::<Vec<_>>();
            let weights = (0..count)
                .map(|j| Float::with_val(p, (13 * j + 3) % 19 - 9) >> (j % 11))
                .collect::<Vec<_>>();
            assert_numerator_matches_oracle(&poles, &weights, p);
        }
    }

    #[test]
    fn discovery_integer_numerator_preserves_degree_cancellation_and_zero_weights() {
        let p = 128;
        let poles = [-2, -1, 0, 1, 2].map(|value| Float::with_val(p, value));
        for weights in [[1, 0, 0, 0, -1], [1, -4, 6, -4, 1], [0, 0, 3, 0, 0]] {
            let weights = weights.map(|value| Float::with_val(p, value));
            assert_numerator_matches_oracle(&poles, &weights, p);
        }
        // Repeated poles are refused by roots_rational; exercise full
        // cancellation at the private construction boundary independently.
        let repeated = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let weights = [Float::with_val(p, 3), Float::with_val(p, -3)];
        let workspace = DiscoveryWorkspace::estimate(
            &repeated,
            &weights,
            &Rational::from(0),
            &Rational::from(2),
            p,
        )
        .unwrap()
        .admit()
        .unwrap();
        let old_error = numerator_sequential_oracle(&repeated, &weights, &workspace).unwrap_err();
        let new_error = numerator(&repeated, &weights, &workspace).unwrap_err();
        assert_eq!(new_error.to_string(), old_error.to_string());
        let zero = [Float::with_val(p, 0), Float::with_val(p, 0)];
        assert_eq!(
            numerator(&repeated, &zero, &workspace)
                .unwrap_err()
                .to_string(),
            numerator_sequential_oracle(&repeated, &zero, &workspace)
                .unwrap_err()
                .to_string()
        );
    }

    #[test]
    fn discovery_integer_numerator_preserves_wide_dyadic_exponents() {
        let p = 160;
        let poles = [-2048, -61, -1, 0, 79, 2048]
            .into_iter()
            .enumerate()
            .map(|(j, exponent)| Float::with_val(p, 2 * j + 1) << exponent)
            .collect::<Vec<_>>();
        let weights = [1536, -1024, 17, -17, 0, -1536]
            .into_iter()
            .enumerate()
            .map(|(j, exponent)| Float::with_val(p, j as i32 - 3) << exponent)
            .collect::<Vec<_>>();
        assert_numerator_matches_oracle(&poles, &weights, p);
    }

    #[test]
    #[ignore = "explicit bounded numerator timing; no root isolation or matrix assembly"]
    fn discovery_integer_numerator_hp1000_bounded_comparison() {
        let p = 3386;
        for modes in [32, 64] {
            let spacing = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
            let poles = super::super::secular_poles(&spacing, modes, p);
            let weights = poles
                .iter()
                .enumerate()
                .map(|(j, _)| Float::with_val(p, (j % 7 + 1) as u32) >> (j % 5))
                .collect::<Vec<_>>();
            let workspace = DiscoveryWorkspace::estimate(
                &poles,
                &weights,
                &Rational::from(0),
                &Rational::from(modes + 1),
                p,
            )
            .unwrap()
            .admit()
            .unwrap();
            assert!(workspace.bytes < 16u128 << 30);
            let start = std::time::Instant::now();
            let expected = numerator_sequential_oracle(&poles, &weights, &workspace).unwrap();
            let sequential_seconds = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let actual = numerator(&poles, &weights, &workspace).unwrap();
            let tree_seconds = start.elapsed().as_secs_f64();
            assert_eq!(actual, expected);
            eprintln!(
                "exact_numerator_comparison poles={} precision={p} coefficient_bits={} admitted_bytes={} sequential_seconds={sequential_seconds:.6} tree_seconds={tree_seconds:.6}",
                poles.len(), coefficient_bits(&actual), workspace.bytes,
            );
        }
    }

    #[cfg(feature = "arb")]
    #[test]
    fn discovery_arb_workspace_admits_campaign_dimensions_within_384_gib() {
        for (modes, p) in [(120, 3386), (500, 3386), (800, 6708)] {
            let spacing = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
            let poles = super::super::secular_poles(&spacing, modes, p);
            let weights = vec![spacing.clone(); poles.len()];
            // Estimate only: no rational source conversion, polynomial
            // construction, native root isolation, or workspace reservation.
            let estimate = DiscoveryWorkspace::estimate(
                &poles,
                &weights,
                &Rational::from(0),
                &Rational::from(modes + 1),
                p,
            )
            .unwrap();
            assert!(
                estimate.bytes < 384u128 << 30,
                "N={modes}, p={p}: estimate={} bytes",
                estimate.bytes
            );
        }
    }

    #[test]
    fn discovery_dyadic_denominator_estimate_uses_the_stored_value() {
        for p in [64, 3386] {
            for exponent in [-1000, 0, 1000] {
                for value in [0, 1, -3] {
                    let point: Float = Float::with_val(p, value) << exponent;
                    let denominator = point.to_rational().unwrap().denom().significant_bits() - 1;
                    assert_eq!(dyadic_denominator_bits(&point), u128::from(denominator));
                }
            }
            let point = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
            assert_eq!(dyadic_denominator_bits(&point), u128::from(p - 1));
        }
    }

    #[test]
    fn discovery_workspace_obeys_run_budget_and_releases_after_success_or_failure() {
        let p = 128;
        let poles = [-1, 0, 1].map(|x| Float::with_val(p, x));
        let weights = [(21, 16), (-3, 8), (1, 16)].map(|q| Float::with_val(p, Rational::from(q)));
        let lower = Rational::from(0);
        let upper = Rational::from(1);
        let bytes = DiscoveryWorkspace::estimate(&poles, &weights, &lower, &upper, p)
            .unwrap()
            .bytes as u64;
        let small = xc_core::RunResources::new(bytes - 1).unwrap();
        {
            let _scope = small.enter();
            let error =
                roots_rational(&poles, &weights, &lower, &upper, p, false, false).unwrap_err();
            assert!(error.to_string().contains("workspace budget"));
            assert_eq!(small.usage().peak_bytes, 0);
        }
        let large = xc_core::RunResources::new(bytes + 128).unwrap();
        let _scope = large.enter();
        let retained = large.try_reserve_retained(128).unwrap();
        let values = roots_rational(&poles, &weights, &lower, &upper, p, false, false).unwrap();
        assert_eq!(values, [Float::with_val(p, 0.5), Float::with_val(p, 0.75)]);
        assert_eq!(large.usage().working_bytes, 0);
        assert_eq!(large.usage().peak_bytes, bytes + 128);
        let pressure = large.try_reserve_retained(1).unwrap();
        assert!(roots_rational(&poles, &weights, &lower, &upper, p, false, false).is_err());
        assert_eq!(large.usage().working_bytes, 0);
        drop(pressure);
        drop(retained);

        let repeated = [(9, 8), (-1, 4), (1, 8)].map(|q| Float::with_val(p, Rational::from(q)));
        assert!(roots_rational(&poles, &repeated, &lower, &upper, p, false, false).is_err());
        assert_eq!(large.usage().working_bytes, 0);
        assert_eq!(large.usage().retained_bytes, 0);
    }

    #[test]
    fn discovery_workspace_preserves_exact_nondyadic_boundary_deflation() {
        let p = 128;
        let poles = [-1, 0, 1].map(|x| Float::with_val(p, x));
        // Numerator (3*x - 1)*(x - 1/2), with exact dyadic residues.
        let weights = [(3, 1), (-1, 2), (1, 2)].map(|q| Float::with_val(p, Rational::from(q)));
        let lower = Rational::from((1, 3));
        let upper = Rational::from(1);
        let bytes = DiscoveryWorkspace::estimate(&poles, &weights, &lower, &upper, p)
            .unwrap()
            .bytes as u64;
        let resources = xc_core::RunResources::new(bytes).unwrap();
        let _scope = resources.enter();
        assert_eq!(
            roots_rational(&poles, &weights, &lower, &upper, p, false, false).unwrap(),
            [Float::with_val(p, 0.5)]
        );
        assert_eq!(
            roots_rational(&poles, &weights, &lower, &upper, p, true, false).unwrap(),
            [Float::with_val(p, &lower), Float::with_val(p, 0.5)]
        );
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    #[ignore = "explicit release-mode N=120/HP-1000 exact coefficient qualification; no matrix assembly"]
    fn discovery_workspace_n120_hp1000_crosses_the_old_coefficient_limit() {
        let p = 3386;
        let spacing = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
        let poles = super::super::secular_poles(&spacing, 120, p);
        let weights = vec![Float::with_val(p, 1); poles.len()];
        let lower = Rational::from(0);
        let upper = Rational::from(121);
        let estimate = DiscoveryWorkspace::estimate(&poles, &weights, &lower, &upper, p).unwrap();
        assert!(
            estimate.bytes < 64u128 << 30,
            "dyadic common denominator was overcharged"
        );
        let small = xc_core::RunResources::new(1 << 20).unwrap();
        {
            let _scope = small.enter();
            assert!(estimate.admit().is_err());
            assert_eq!(small.usage().peak_bytes, 0);
        }
        let large = xc_core::RunResources::new(384u64 << 30).unwrap();
        let _scope = large.enter();
        {
            let workspace = DiscoveryWorkspace::estimate(&poles, &weights, &lower, &upper, p)
                .unwrap()
                .admit()
                .unwrap();
            let coefficients = numerator(&poles, &weights, &workspace).unwrap();
            let bits = coefficient_bits(&coefficients);
            assert!(
                bits > 67_108_864,
                "fixture did not cross the former cap: {bits}"
            );
            assert_eq!(coefficients.len(), 241);
            assert_eq!(coefficients.last(), Some(&Rational::from(241)));
            assert!(coefficients
                .iter()
                .skip(1)
                .step_by(2)
                .all(|value| value == &0));
            assert_eq!(large.usage().working_bytes, workspace.bytes as u64);
        }
        assert_eq!(large.usage().working_bytes, 0);
    }

    #[test]
    fn portable_sturm_fallback_rejects_repeated_roots() {
        // The exact polynomial is (x - 65/128)^2.
        let error = exact_isolation(
            &[
                Rational::from((4225, 16384)),
                Rational::from((-65, 64)),
                Rational::from(1),
            ],
            &Rational::from(0),
            &Rational::from(1),
            96,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unresolved repeated roots"));
    }

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
