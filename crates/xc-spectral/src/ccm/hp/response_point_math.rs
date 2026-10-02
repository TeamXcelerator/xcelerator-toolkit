//! Correct rounding of response dot products and Euclidean/Frobenius norms.
//! Exact stored points only; this does not bound source or linear-solve error.
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;
const GUARDS: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

fn precision<'a>(values: impl Iterator<Item = &'a Float>, p: u32, count: usize) -> Result<u32> {
    if !(64..=1_000_000).contains(&p) {
        bail!("unsupported response point precision");
    }
    let mut working = p;
    for value in values {
        if !value.is_finite() || value.prec() > 1_000_128 {
            bail!("response point source is nonfinite or exceeds supported precision");
        }
        working = working.max(value.prec());
    }
    if count as u128 * (u128::from(working + 4096).div_ceil(8) + 96) * 32
        > super::source_working_budget()?
    {
        bail!("response point arithmetic exceeds the workspace budget");
    }
    Ok(working)
}
// Dense shifted entries, their squared intervals and the scaled sum's
// aligned intervals are the three simultaneous interval arrays in the
// bordered path. Source points remain live; vector/scalar work is separate.
fn dense_workspace_bytes(
    entries: usize,
    vectors: usize,
    source_precision: u32,
    work: u32,
    working_floats: u128,
) -> u128 {
    let source_point = u128::from(source_precision).div_ceil(8) + 96;
    let work_point = u128::from(work).div_ceil(8) + 96;
    entries as u128 * source_point
        + (entries as u128 * working_floats + vectors as u128 * 32 + 128) * work_point
}
fn dense_workspace(
    entries: usize,
    vectors: usize,
    source_precision: u32,
    work: u32,
    working_floats: u128,
) -> Result<()> {
    if dense_workspace_bytes(entries, vectors, source_precision, work, working_floats)
        > super::source_working_budget()?
    {
        bail!(
            "response dense point arithmetic exceeds the declared workspace at {work} working bits"
        );
    }
    Ok(())
}
fn sum(values: &[I], p: u32) -> Result<I> {
    for value in values {
        value.validate()?;
    }
    Ok(I::new(
        Float::with_val_round(p, Float::sum(values.iter().map(I::lower)), Round::Down).0,
        Float::with_val_round(p, Float::sum(values.iter().map(I::upper)), Round::Up).0,
    )?)
}
fn rounded(value: &I, p: u32) -> Result<Option<Float>> {
    value.validate()?;
    let lo = Float::with_val(p, value.lower());
    let hi = Float::with_val(p, value.upper());
    if lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (value.lower().is_zero() && value.upper().is_zero()))
    {
        Ok(Some(lo))
    } else {
        Ok(None)
    }
}
fn norm_enclosure(values: &[I], p: u32) -> Result<I> {
    for value in values {
        value.validate()?;
    }
    let exponent = values
        .iter()
        .flat_map(|v| [v.lower(), v.upper()])
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let squares = values
        .iter()
        .map(|value| scale(value, -exponent).map(|x| x.square()))
        .collect::<Result<Vec<_>>>()?;
    scale(&sum(&squares, p)?.sqrt()?, exponent)
}
pub(super) fn dot(left: &[Float], right: &[Float], p: u32) -> Result<Float> {
    if left.len() != right.len() || left.len() > 16385 {
        bail!("response dot product shape is invalid");
    }
    let base = precision(left.iter().chain(right), p, left.len())?;
    let le = left
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let re = right
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    for guard in GUARDS {
        let work = base + guard;
        let products = left
            .iter()
            .zip(right)
            .map(|(a, b)| -> Result<I> {
                Ok(I::point(scale_float(a, -le, work)?).mul(&I::point(scale_float(b, -re, work)?)))
            })
            .collect::<Result<Vec<_>>>()?;
        let value = scale(&sum(&products, work)?, le + re)?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!("response dot product rounding unresolved within 4096 guard bits")
}
pub(super) fn norm(values: &[Float], p: u32) -> Result<Float> {
    if values.len() > 16385 {
        bail!("response vector norm shape exceeds the supported limit");
    }
    let base = precision(values.iter(), p, values.len())?;
    for guard in GUARDS {
        let work = base + guard;
        let points = values
            .iter()
            .map(|v| I::point(Float::with_val(work, v)))
            .collect::<Vec<_>>();
        let value = norm_enclosure(&points, work)?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!("response norm rounding unresolved within 4096 guard bits")
}
pub(super) fn shifted_norm(
    matrix: &[Float],
    eigenvalue: &Float,
    dimension: usize,
    p: u32,
) -> Result<Float> {
    if dimension == 0 || dimension > 16385 || dimension.checked_mul(dimension) != Some(matrix.len())
    {
        bail!("shifted response norm requires a nonempty square matrix");
    }
    let base = precision(
        matrix.iter().chain(std::iter::once(eigenvalue)),
        p,
        dimension + 1,
    )?;
    for guard in GUARDS {
        let work = base + guard;
        dense_workspace(matrix.len(), dimension + 1, base, work, 4)?;
        let eigenvalue = I::point(Float::with_val(work, eigenvalue));
        // Subtract the common diagonal before choosing the norm's scale so a
        // huge common shift cannot erase tiny off-diagonal feedback.
        let shifted = matrix
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let value = I::point(Float::with_val(work, value));
                if index / dimension == index % dimension {
                    value.sub(&eigenvalue)
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let value = norm_enclosure(&shifted, work)?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!("shifted response norm rounding unresolved within 4096 guard bits")
}

// Keep binary exponents separate so residual and denominator products need not
// fit in MPFR's exponent range. Every mantissa operation remains directed.
struct Scaled {
    value: I,
    exponent: i64,
}
impl Scaled {
    fn new(value: I, exponent: i64) -> Result<Self> {
        value.validate()?;
        let shift = [value.lower(), value.upper()]
            .into_iter()
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        Ok(Self {
            value: scale(&value, -shift)?,
            exponent: exponent + shift,
        })
    }
    fn point(value: &Float, p: u32) -> Result<Self> {
        let exponent = value.get_exp().map_or(0, i64::from);
        Ok(Self {
            value: I::point(scale_float(value, -exponent, p)?),
            exponent,
        })
    }
    fn zero(&self) -> bool {
        self.value.lower().is_zero() && self.value.upper().is_zero()
    }
    fn neg(&self) -> Self {
        Self {
            value: self.value.neg(),
            exponent: self.exponent,
        }
    }
    fn mul(&self, other: &Self) -> Result<Self> {
        Self::new(self.value.mul(&other.value), self.exponent + other.exponent)
    }
    fn div(&self, other: &Self) -> Result<Self> {
        Self::new(
            self.value.div(&other.value)?,
            self.exponent - other.exponent,
        )
    }
    fn sqrt(&self) -> Result<Self> {
        let half = self.exponent.div_euclid(2);
        Self::new(scale(&self.value, self.exponent - 2 * half)?.sqrt()?, half)
    }
    fn sum(values: &[Self], p: u32) -> Result<Self> {
        let exponent = values
            .iter()
            .filter(|v| !v.zero())
            .map(|v| v.exponent)
            .max()
            .unwrap_or(0);
        let parts = values
            .iter()
            .map(|v| scale(&v.value, v.exponent - exponent))
            .collect::<Result<Vec<_>>>()?;
        Self::new(sum(&parts, p)?, exponent)
    }
    fn norm(values: &[Self], p: u32) -> Result<Self> {
        let squares = values
            .iter()
            .map(|v| Self::new(v.value.square(), 2 * v.exponent))
            .collect::<Result<Vec<_>>>()?;
        let sum = Self::sum(&squares, p)?;
        let half = sum.exponent.div_euclid(2);
        Self::new(scale(&sum.value, sum.exponent - 2 * half)?.sqrt()?, half)
    }
    fn materialize(&self) -> Result<I> {
        scale(&self.value, self.exponent)
    }
}

/// An upper bound for the exact stored bordered residual divided by
/// ||A-lambda I||_F ||v||_2 + ||f||_2 + |mu|. The border row is u^T v.
/// This verifies that diagnostic, not continuum accuracy or forward solve error.
#[allow(clippy::too_many_arguments)]
pub(super) fn bordered_residual(
    matrix: &[Float],
    eigenvalue: &Float,
    unit: &[Float],
    forcing: &[Float],
    response: &[Float],
    multiplier: &Float,
    supplied_norm: &Float,
    p: u32,
) -> Result<Float> {
    let n = unit.len();
    if n == 0
        || n > 16385
        || n.checked_mul(n) != Some(matrix.len())
        || forcing.len() != n
        || response.len() != n
    {
        bail!("bordered response residual requires compatible nonempty dimensions");
    }
    let base = precision(
        matrix
            .iter()
            .chain(unit)
            .chain(forcing)
            .chain(response)
            .chain([eigenvalue, multiplier, supplied_norm]),
        p,
        3 * n + 3,
    )?;
    if supplied_norm < &0 {
        bail!("bordered response matrix norm is negative");
    }
    for guard in GUARDS {
        let work = base + guard;
        dense_workspace(matrix.len(), 3 * n + 3, base, work, 6)?;
        let points = |values: &[Float]| {
            values
                .iter()
                .map(|v| Scaled::point(v, work))
                .collect::<Result<Vec<_>>>()
        };
        let unit = points(unit)?;
        let response = points(response)?;
        let forcing = points(forcing)?;
        let eigenvalue = Scaled::point(eigenvalue, work)?;
        let multiplier = Scaled::point(multiplier, work)?;
        let shifted = matrix
            .iter()
            .enumerate()
            .map(|(index, v)| {
                let value = Scaled::point(v, work)?;
                if index / n == index % n {
                    Scaled::sum(&[value, eigenvalue.neg()], work)
                } else {
                    Ok(value)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let matrix_norm = Scaled::norm(&shifted, work)?;
        // A caller's cached point norm cannot inflate the acceptance denominator.
        let Some(replayed_norm) = rounded(&matrix_norm.materialize()?, p)? else {
            continue;
        };
        if replayed_norm != *supplied_norm {
            bail!("bordered response matrix norm does not match the exact stored source");
        }
        let mut residual = Vec::with_capacity(n + 1);
        for (row, coefficients) in shifted.chunks_exact(n).enumerate() {
            let mut terms = coefficients
                .iter()
                .zip(&response)
                .map(|(x, y)| x.mul(y))
                .collect::<Result<Vec<_>>>()?;
            terms.push(unit[row].mul(&multiplier)?);
            terms.push(Scaled::new(
                forcing[row].value.clone(),
                forcing[row].exponent,
            )?);
            residual.push(Scaled::sum(&terms, work)?);
        }
        residual.push(Scaled::sum(
            &unit
                .iter()
                .zip(&response)
                .map(|(x, y)| x.mul(y))
                .collect::<Result<Vec<_>>>()?,
            work,
        )?);
        let numerator = Scaled::norm(&residual, work)?;
        let denominator = Scaled::sum(
            &[
                matrix_norm.mul(&Scaled::norm(&response, work)?)?,
                Scaled::norm(&forcing, work)?,
                Scaled::new(
                    crate::ccm::retained_evidence::finite_math::abs(&multiplier.value)?,
                    multiplier.exponent,
                )?,
            ],
            work,
        )?;
        let ratio = if denominator.zero() {
            numerator.materialize()?
        } else {
            if !denominator.value.is_strictly_positive() {
                continue;
            }
            Scaled::new(
                numerator.value.div(&denominator.value)?,
                numerator.exponent - denominator.exponent,
            )?
            .materialize()?
        };
        let natural = ratio.upper().clone().max(&Float::with_val(work, 1));
        if !crate::ccm::retained_evidence::finite_math::narrow(
            std::slice::from_ref(&ratio),
            &natural,
            p,
        )? {
            continue;
        }
        let bound = Float::with_val_round(p, ratio.upper(), Round::Up).0;
        if !bound.is_finite() || bound < 0 || (bound.is_zero() && !ratio.upper().is_zero()) {
            bail!("bordered response residual bound exceeds the output range");
        }
        return Ok(bound);
    }
    bail!("bordered response residual bound unresolved within 4096 guard bits")
}

/// Correctly rounded a_i - u_i * eigenvalue_response for exact stored points.
pub(super) fn projected_forcing(
    action: &[Float],
    unit: &[Float],
    eigenvalue_response: &Float,
    p: u32,
) -> Result<Vec<Float>> {
    if action.len() != unit.len() || action.len() > 16385 {
        bail!("response forcing projection shape is invalid");
    }
    let base = precision(
        action
            .iter()
            .chain(unit)
            .chain(std::iter::once(eigenvalue_response)),
        p,
        2 * action.len() + 1,
    )?;
    for guard in GUARDS {
        let work = base + guard;
        let eigenvalue_response = Scaled::point(eigenvalue_response, work)?;
        let mut result = Vec::with_capacity(action.len());
        for (a, u) in action.iter().zip(unit) {
            let projection = Scaled::point(u, work)?.mul(&eigenvalue_response)?;
            let value =
                Scaled::sum(&[Scaled::point(a, work)?, projection.neg()], work)?.materialize()?;
            let Some(point) = rounded(&value, p)? else {
                break;
            };
            result.push(point);
        }
        if result.len() == action.len() {
            return Ok(result);
        }
    }
    bail!("response forcing projection rounding unresolved within 4096 guard bits")
}

/// MPFR rounds the complete exact stored sum once, including cancellation.
pub(super) fn point_sum(values: &[Float], p: u32) -> Result<Float> {
    if values.len() > 16385 {
        bail!("response state sum shape exceeds the supported limit");
    }
    precision(values.iter(), p, values.len())?;
    let (value, direction) = Float::with_val_round(p, Float::sum(values.iter()), Round::Nearest);
    if !value.is_finite() || (value.is_zero() && direction != std::cmp::Ordering::Equal) {
        bail!("response state sum exceeds the output range");
    }
    Ok(value)
}

/// Correctly rounded (target_velocity - scale * sum(response)) / sum(unit).
/// All sums and scalar operands refer to exact stored points. The supplied
/// rounded unit sum is replayed; it is not substituted for the exact denominator.
pub(super) fn normalization_tangent(
    unit: &[Float],
    response: &[Float],
    target_velocity: &Float,
    normalization_scale: &Float,
    supplied_sum: &Float,
    p: u32,
) -> Result<Float> {
    if unit.is_empty() || unit.len() > 16385 || response.len() != unit.len() {
        bail!("response normalization tangent shape is invalid");
    }
    let base = precision(
        unit.iter()
            .chain(response)
            .chain([target_velocity, normalization_scale, supplied_sum]),
        p,
        2 * unit.len() + 3,
    )?;
    if *supplied_sum != point_sum(unit, p)? {
        bail!("response normalization sum does not match the stored state");
    }
    for guard in GUARDS {
        let work = base + guard;
        let total = |v: &[Float]| {
            Scaled::sum(
                &v.iter()
                    .map(|x| Scaled::point(x, work))
                    .collect::<Result<Vec<_>>>()?,
                work,
            )
        };
        let denominator = total(unit)?;
        if denominator.zero() {
            bail!("response normalization has zero state sum");
        }
        if denominator.value.contains_zero() {
            continue;
        }
        let gauge = Scaled::point(normalization_scale, work)?.mul(&total(response)?)?;
        let numerator = Scaled::sum(&[Scaled::point(target_velocity, work)?, gauge.neg()], work)?;
        let result = Scaled::new(
            numerator.value.div(&denominator.value)?,
            numerator.exponent - denominator.exponent,
        )?
        .materialize()?;
        if let Some(point) = rounded(&result, p)? {
            return Ok(point);
        }
    }
    bail!("response normalization tangent rounding unresolved within 4096 guard bits")
}

/// Correctly rounded components xi_i / sqrt(sum_j xi_j^2), for exact stored xi.
/// The returned rounded vector need not have an exactly unit rational norm.
pub(super) fn unit_state(xi: &[Float], p: u32) -> Result<Vec<Float>> {
    if xi.is_empty() || xi.len() > 16385 {
        bail!("response unit-state shape is invalid");
    }
    let base = precision(xi.iter(), p, xi.len())?;
    if xi.iter().all(Float::is_zero) {
        bail!("response unit state is zero");
    }
    for guard in GUARDS {
        let work = base + guard;
        let points = xi
            .iter()
            .map(|v| Scaled::point(v, work))
            .collect::<Result<Vec<_>>>()?;
        let denominator = Scaled::norm(&points, work)?;
        if denominator.value.contains_zero() {
            continue;
        }
        let mut out = Vec::with_capacity(xi.len());
        for point in &points {
            let value = point.div(&denominator)?.materialize()?;
            if let Some(point) = rounded(&value, p)? {
                out.push(point);
            } else {
                break;
            }
        }
        if out.len() == xi.len() {
            return Ok(out);
        }
    }
    bail!("response unit-state rounding unresolved within 4096 guard bits")
}

/// Correctly rounded sqrt(L) / exact_sum(unit). L and unit are stored points;
/// the separately rounded norm/sum must not be substituted into this formula.
pub(super) fn normalization_scale(unit: &[Float], l: &Float, p: u32) -> Result<Float> {
    if unit.is_empty() || unit.len() > 16385 {
        bail!("response normalization scale shape is invalid");
    }
    let base = precision(unit.iter().chain(std::iter::once(l)), p, unit.len() + 1)?;
    if l <= &0 {
        bail!("response normalization cutoff must be positive");
    }
    for guard in GUARDS {
        let work = base + guard;
        let denominator = Scaled::sum(
            &unit
                .iter()
                .map(|v| Scaled::point(v, work))
                .collect::<Result<Vec<_>>>()?,
            work,
        )?;
        if denominator.zero() {
            bail!("response normalization has zero state sum");
        }
        if denominator.value.contains_zero() {
            continue;
        }
        let value = Scaled::point(l, work)?
            .sqrt()?
            .div(&denominator)?
            .materialize()?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!("response normalization scale rounding unresolved within 4096 guard bits")
}

/// Correctly rounded 1 / (2 sqrt(L)) for the exact stored positive cutoff L.
pub(super) fn target_velocity(l: &Float, p: u32) -> Result<Float> {
    let base = precision(std::iter::once(l), p, 1)?;
    if l <= &0 {
        bail!("response normalization cutoff must be positive");
    }
    for guard in GUARDS {
        let work = base + guard;
        let mut denominator = Scaled::point(l, work)?.sqrt()?;
        denominator.exponent += 1;
        let value = Scaled::point(&Float::with_val(work, 1), work)?
            .div(&denominator)?
            .materialize()?;
        if let Some(value) = rounded(&value, p)? {
            return Ok(value);
        }
    }
    bail!("response target velocity rounding unresolved within 4096 guard bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::{float::Special, Rational};

    #[test]
    fn response_normalization_exact_rational_oracles() {
        for p in [64, 128, 256] {
            for i in -9..=9 {
                for j in -5..=5 {
                    let f = |n, d| Float::with_val(320, Rational::from((n, d)));
                    let unit = [f(i, 7), f(1, 3)];
                    let response = [f(j, 11), f(i - j, 13)];
                    let target = f(i + 1, 17);
                    let scale = f(j + 2, 19);
                    let sum = unit
                        .iter()
                        .fold(Rational::from(0), |a, x| a + x.to_rational().unwrap());
                    assert_ne!(sum, 0);
                    let response_sum = response
                        .iter()
                        .fold(Rational::from(0), |a, x| a + x.to_rational().unwrap());
                    let expected = Float::with_val(
                        p,
                        (target.to_rational().unwrap()
                            - scale.to_rational().unwrap() * response_sum)
                            / &sum,
                    );
                    let stored = point_sum(&unit, p).unwrap();
                    assert_eq!(stored, Float::with_val(p, sum));
                    assert_eq!(
                        normalization_tangent(&unit, &response, &target, &scale, &stored, p)
                            .unwrap(),
                        expected,
                        "p={p},i={i},j={j}"
                    );
                }
            }
        }
    }
    #[test]
    fn response_sum_cancels_unrepresentable_partial_sum_and_preserves_source_bits() {
        let p = 128;
        let huge = Float::with_val(p, 0.75) << rug::float::exp_max();
        assert_eq!(
            point_sum(&[huge.clone(), huge.clone(), -huge.clone()], p).unwrap(),
            huge
        );
        let one = Float::with_val(320, 1);
        let delta = one.clone() >> 250u32;
        assert_eq!(
            point_sum(&[Float::with_val(320, &one + &delta), -one], 64).unwrap(),
            delta
        );
        assert_eq!(point_sum(&[], 64).unwrap(), 0);
    }
    #[test]
    fn response_normalization_validates_all_sources_and_shape() {
        let p = 128;
        let one = Float::with_val(p, 1);
        for slot in 0..5 {
            let mut values = vec![one.clone(); 5];
            values[slot] = Float::with_val(p, Special::Nan);
            assert!(normalization_tangent(
                &values[0..1],
                &values[1..2],
                &values[2],
                &values[3],
                &values[4],
                p
            )
            .is_err());
        }
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(point_sum(&[], p).is_err());
            assert!(normalization_tangent(
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &one,
                &one,
                &one,
                p
            )
            .is_err());
        }
        assert!(
            normalization_tangent(std::slice::from_ref(&one), &[], &one, &one, &one, p).is_err()
        );
        assert!(normalization_tangent(&[], &[], &one, &one, &one, p).is_err());
        assert!(point_sum(&vec![one.clone(); 16386], p).is_err());
        assert!(point_sum(&[Float::with_val(1_000_129, 1)], p).is_err());
        let huge = Float::with_val(p, 0.75) << rug::float::exp_max();
        assert!(point_sum(&[huge.clone(), huge], p).is_err());
        let minimum = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        let mut adjacent = minimum.clone();
        adjacent.next_up();
        assert!(point_sum(&[adjacent, -minimum], p).is_err());
    }
    #[test]
    fn response_projection_exact_rational_oracles() {
        for p in [64, 128, 256] {
            for i in -9..=9 {
                for j in -5..=5 {
                    let f = |n, d| Float::with_val(320, Rational::from((n, d)));
                    let action = [f(i, 7), f(j, 17)];
                    let unit = [f(j, 11), f(i, 13)];
                    let eigenvalue = f(i - j, 19);
                    let expected = action
                        .iter()
                        .zip(&unit)
                        .map(|(a, u)| {
                            Float::with_val(
                                p,
                                a.to_rational().unwrap()
                                    - u.to_rational().unwrap() * eigenvalue.to_rational().unwrap(),
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        projected_forcing(&action, &unit, &eigenvalue, p).unwrap(),
                        expected
                    );
                }
            }
        }
    }
    #[test]
    fn response_projection_invalid_precision_range_and_empty_domain() {
        let one = Float::with_val(128, 1);
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(projected_forcing(&[], &[], &one, p).is_err());
        }
        assert!(projected_forcing(&[], &[], &one, 64).unwrap().is_empty());
        for special in [Special::Nan, Special::Infinity, Special::NegInfinity] {
            let bad = Float::with_val(128, special);
            assert!(projected_forcing(
                std::slice::from_ref(&bad),
                std::slice::from_ref(&one),
                &one,
                128
            )
            .is_err());
            assert!(projected_forcing(
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &bad,
                128
            )
            .is_err());
        }
        let huge = one.clone() << 700_000_000u32;
        let tiny = one.clone() >> 700_000_000u32;
        assert!(projected_forcing(
            &[Float::with_val(128, 0)],
            std::slice::from_ref(&huge),
            &huge,
            128
        )
        .is_err());
        assert!(projected_forcing(
            &[Float::with_val(128, 0)],
            std::slice::from_ref(&tiny),
            &tiny,
            128
        )
        .is_err());
    }
    #[test]
    fn bordered_residual_rational_oracles() {
        fn square_sum(v: &[Rational]) -> Rational {
            v.iter()
                .fold(Rational::from(0), |acc, x| acc + Rational::from(x * x))
        }
        fn norm(v: &[Rational]) -> I {
            I::from_rational(&square_sum(v), 4096).sqrt().unwrap()
        }
        for p in [64, 128, 256] {
            for i in -3..=3 {
                for j in -3..=3 {
                    for exponent in [-100i64, 0, 100] {
                        let f = |n, d| Float::with_val(320, Rational::from((n, d)));
                        let scale = |x: Float| scale_float(&x, exponent, 320).unwrap();
                        let matrix = [
                            scale(f(i, 3)),
                            scale(f(j, 7)),
                            scale(f(j, 7)),
                            scale(f(i + 1, 11)),
                        ];
                        let eigenvalue = scale(f(1, 5));
                        let unit = [f(3, 5), f(4, 5)];
                        let response = [f(i + 2, 13), f(j - 1, 17)];
                        let forcing = [scale(f(j + 2, 19)), scale(f(i - 3, 23))];
                        let multiplier = scale(f(i - j, 29));
                        let source_norm = shifted_norm(&matrix, &eigenvalue, 2, p).unwrap();
                        let actual = bordered_residual(
                            &matrix,
                            &eigenvalue,
                            &unit,
                            &forcing,
                            &response,
                            &multiplier,
                            &source_norm,
                            p,
                        )
                        .unwrap();
                        let rationals = |v: &[Float]| {
                            v.iter()
                                .map(|x| x.to_rational().unwrap())
                                .collect::<Vec<_>>()
                        };
                        let mut a = rationals(&matrix);
                        let eigenvalue = eigenvalue.to_rational().unwrap();
                        a[0] -= &eigenvalue;
                        a[3] -= &eigenvalue;
                        let u = rationals(&unit);
                        let v = rationals(&response);
                        let f = rationals(&forcing);
                        let m = multiplier.to_rational().unwrap();
                        let residual = [
                            Rational::from(&a[0] * &v[0])
                                + Rational::from(&a[1] * &v[1])
                                + Rational::from(&u[0] * &m)
                                + &f[0],
                            Rational::from(&a[2] * &v[0])
                                + Rational::from(&a[3] * &v[1])
                                + Rational::from(&u[1] * &m)
                                + &f[1],
                            Rational::from(&u[0] * &v[0]) + Rational::from(&u[1] * &v[1]),
                        ];
                        let denominator = norm(&a)
                            .mul(&norm(&v))
                            .add(&norm(&f))
                            .add(&I::from_rational(&m.abs(), 4096));
                        let oracle = norm(&residual).div(&denominator).unwrap();
                        let expected = Float::with_val_round(p, oracle.upper(), Round::Up).0;
                        assert_eq!(
                            expected,
                            Float::with_val_round(p, oracle.lower(), Round::Up).0,
                            "oracle must resolve target rounding"
                        );
                        let mut slack = expected.clone();
                        slack.next_up();
                        assert!(
                            actual >= expected && actual <= slack,
                            "p={p},i={i},j={j},scale={exponent}: bound {actual}, exact upper {expected}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn bordered_residual_cancellation_beyond_product_range() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let huge = Float::with_val(p, 1) << 700_000_000u32;
        let matrix = [huge.clone(), huge.clone(), huge.clone(), huge.clone()];
        let unit = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let response = [huge.clone(), -huge.clone()];
        let supplied = Float::with_val(p, &huge * 2);
        assert_eq!(
            bordered_residual(
                &matrix,
                &zero,
                &unit,
                &[zero.clone(), zero.clone()],
                &response,
                &zero,
                &supplied,
                p
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn bordered_residual_zero_denominator_and_exact_solve() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        assert_eq!(
            bordered_residual(
                std::slice::from_ref(&zero),
                &zero,
                std::slice::from_ref(&one),
                std::slice::from_ref(&zero),
                std::slice::from_ref(&one),
                &zero,
                &zero,
                p
            )
            .unwrap(),
            1
        );
        assert_eq!(
            bordered_residual(
                std::slice::from_ref(&zero),
                &zero,
                &[one],
                &[Float::with_val(p, -1)],
                std::slice::from_ref(&zero),
                &Float::with_val(p, 1),
                &zero,
                p
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn bordered_residual_validates_every_scalar_and_dimension() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        for slot in 0..7 {
            let mut data = vec![one.clone(); 7];
            data[slot] = Float::with_val(p, Special::Nan);
            assert!(bordered_residual(
                &data[0..1],
                &data[1],
                &data[2..3],
                &data[3..4],
                &data[4..5],
                &data[5],
                &data[6],
                p
            )
            .is_err());
        }
        assert!(bordered_residual(
            std::slice::from_ref(&one),
            &zero,
            std::slice::from_ref(&one),
            &[],
            std::slice::from_ref(&one),
            &zero,
            &one,
            p
        )
        .is_err());
        assert!(bordered_residual(
            std::slice::from_ref(&one),
            &zero,
            std::slice::from_ref(&one),
            std::slice::from_ref(&zero),
            &[],
            &zero,
            &one,
            p
        )
        .is_err());
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(bordered_residual(
                std::slice::from_ref(&one),
                &zero,
                std::slice::from_ref(&one),
                std::slice::from_ref(&zero),
                std::slice::from_ref(&one),
                &zero,
                &one,
                p
            )
            .is_err());
        }
        assert!(bordered_residual(
            std::slice::from_ref(&one),
            &zero,
            std::slice::from_ref(&one),
            std::slice::from_ref(&zero),
            std::slice::from_ref(&one),
            &zero,
            &Float::with_val(p, -1),
            p
        )
        .is_err());
    }

    #[test]
    fn response_point_exact_dot_rational_oracles() {
        for p in [64, 128, 256] {
            for i in -9..=9 {
                for j in -5..=5 {
                    let a = [
                        Float::with_val(320, Rational::from((i, 7))),
                        Float::with_val(320, Rational::from((j, 11))),
                        Float::with_val(320, 1),
                    ];
                    let b = [
                        Float::with_val(320, Rational::from((j, 3))),
                        Float::with_val(320, Rational::from((i, 13))),
                        Float::with_val(320, -1),
                    ];
                    let exact = a.iter().zip(&b).fold(Rational::from(0), |acc, (x, y)| {
                        acc + x.to_rational().unwrap() * y.to_rational().unwrap()
                    });
                    assert_eq!(
                        dot(&a, &b, p).unwrap(),
                        Float::with_val(p, exact),
                        "p={p},i={i},j={j}"
                    );
                }
            }
        }
    }
    #[test]
    fn response_point_scaled_pythagorean_norms() {
        for p in [64, 128, 256] {
            for exponent in [-700_000_000i64, -100, 0, 100, 700_000_000] {
                let unit = scale_float(&Float::with_val(p, 1), exponent, p).unwrap();
                let values = [
                    Float::with_val(p, &unit * -3),
                    Float::with_val(p, &unit * 4),
                ];
                assert_eq!(norm(&values, p).unwrap(), Float::with_val(p, &unit * 5));
                assert_eq!(
                    dot(&values, &[Float::with_val(p, 4), Float::with_val(p, 3)], p).unwrap(),
                    0
                );
            }
        }
    }
    #[test]
    fn response_point_empty_vectors_and_zero_matrix() {
        assert_eq!(norm(&[], 64).unwrap(), 0);
        assert_eq!(dot(&[], &[], 64).unwrap(), 0);
        assert_eq!(
            shifted_norm(
                &vec![Float::with_val(128, 0); 9],
                &Float::with_val(128, 0),
                3,
                64
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn response_point_domain_errors_are_explicit() {
        let zero = Float::with_val(64, 0);
        let one = Float::with_val(64, 1);
        for p in [0, 1, 63, 1_000_001, u32::MAX] {
            assert!(norm(&[], p).is_err());
            assert!(dot(&[], &[], p).is_err());
            assert!(shifted_norm(std::slice::from_ref(&one), &zero, 1, p).is_err());
        }
        for special in [Special::Nan, Special::Infinity, Special::NegInfinity] {
            let value = Float::with_val(64, special);
            assert!(norm(std::slice::from_ref(&value), 64).is_err());
            assert!(dot(std::slice::from_ref(&value), std::slice::from_ref(&one), 64).is_err());
            assert!(shifted_norm(std::slice::from_ref(&one), &value, 1, 64).is_err());
        }
        for dimension in [0, 2, 16386, usize::MAX] {
            assert!(shifted_norm(std::slice::from_ref(&one), &zero, dimension, 64).is_err());
        }
        assert!(norm(&vec![zero.clone(); 16386], 64).is_err());
        let overprec = Float::with_val(1_000_129, 1);
        assert!(norm(std::slice::from_ref(&overprec), 64).is_err());
        assert!(precision(std::iter::empty(), 1_000_000, usize::MAX).is_err());
    }
    #[test]
    fn response_point_range_failures_cannot_be_finite_zero() {
        let huge = Float::with_val(128, 1) << 700_000_000u32;
        let tiny = Float::with_val(128, 1) >> 700_000_000u32;
        assert!(dot(
            std::slice::from_ref(&huge),
            std::slice::from_ref(&huge),
            128
        )
        .is_err());
        assert!(dot(
            std::slice::from_ref(&tiny),
            std::slice::from_ref(&tiny),
            128
        )
        .is_err());
        assert!(norm(&[huge, tiny], 128).is_err()); // exact rescaling cannot retain both components
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn documented_response_shape_uses_actual_dense_buffer_lifetimes() {
        for (n, p) in [(661usize, 131u32), (801, 6708), (1001, 3386)] {
            for guard in GUARDS {
                assert!(dense_workspace_bytes(n * n, 3 * n + 3, p, p + guard, 6) < (8u128 << 30));
            }
        }
        assert!(dense_workspace(16385 * 16385, 3 * 16385 + 3, 1_000_000, 1_004_096, 6).is_err());
        for p in [131u32, 6708] {
            let matrix = [2, 0, 0, 0, 3, 0, 0, 0, 6].map(|x| Float::with_val(p, x));
            let zero = Float::with_val(p, 0);
            assert_eq!(shifted_norm(&matrix, &zero, 3, p).unwrap(), 7);
            let unit = [0, 1, 0].map(|x| Float::with_val(p, x));
            let vector = [0, 0, 0].map(|x| Float::with_val(p, x));
            assert_eq!(
                bordered_residual(
                    &matrix,
                    &zero,
                    &unit,
                    &vector,
                    &vector,
                    &zero,
                    &Float::with_val(p, 7),
                    p
                )
                .unwrap(),
                0
            );
        }
    }
}
