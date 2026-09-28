//! Correctly rounded conditioning of the exact stored finite secular source.
//! Coordinate and weight scales are removed before division. These point
//! diagnostics do not certify the source assembly or an infinite operator.
use crate::ccm::retained_evidence::finite_math::{abs, scale, scale_float};
use anyhow::{bail, Result};
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) fn evaluate(
    weights: &[Float],
    poles: &[Float],
    z: &Float,
    p: u32,
) -> Result<(Float, Float, Float)> {
    if !(64..=1_000_000).contains(&p)
        || weights.is_empty()
        || weights.len() != poles.len()
        || weights
            .iter()
            .chain(poles)
            .chain(std::iter::once(z))
            .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid conditioning source points, precision or shape");
    }
    let coordinates = poles
        .iter()
        .chain(std::iter::once(z))
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let amplitudes = weights
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map(i64::from)
        .ok_or_else(|| anyhow::anyhow!("conditioning source has zero derivative"))?;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = p + guard;
        if (weights.len() as u64)
            .saturating_mul(12)
            .saturating_add(128)
            .saturating_mul(u64::from(work).div_ceil(8) + 64)
            > (8u64 << 30)
        {
            bail!("conditioning exceeds numerical workspace budget");
        }
        let z = I::point(scale_float(z, -coordinates, work)?);
        let mut magnitude = I::from_i64(0, work);
        let mut derivative = I::from_i64(0, work);
        for (weight, pole) in weights.iter().zip(poles) {
            let denominator = z.sub(&I::point(scale_float(pole, -coordinates, work)?));
            if denominator.contains_zero() {
                bail!("conditioning root coincides with a pole or its separation is unresolved");
            }
            let term = I::point(scale_float(weight, -amplitudes, work)?).div(&denominator)?;
            magnitude = magnitude.add(&abs(&term)?);
            derivative = derivative.sub(&term.div(&denominator)?);
        }
        derivative.validate()?;
        if derivative.lower().is_zero() && derivative.upper().is_zero() {
            bail!("conditioning source has zero derivative");
        }
        if derivative.contains_zero() {
            continue;
        }
        let reciprocal = I::from_i64(1, work).div(&derivative)?;
        let values = [
            scale(&magnitude, amplitudes - coordinates)?,
            scale(&derivative, amplitudes - 2 * coordinates)?,
            scale(&reciprocal, 2 * coordinates - amplitudes)?,
        ];
        let mut rounded = Vec::with_capacity(3);
        for value in values {
            value.validate()?;
            let lower = Float::with_val(p, value.lower());
            let upper = Float::with_val(p, value.upper());
            if lower != upper || !lower.is_finite() || lower.is_zero() {
                break;
            }
            rounded.push(lower);
        }
        if rounded.len() == 3 {
            return Ok((rounded.remove(0), rounded.remove(0), rounded.remove(0)));
        }
    }
    bail!("conditioning rounding unresolved within 4096 guard bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;
    #[test]
    fn exact_rational_conditioning_across_precisions_signs_and_scales() {
        for p in [64, 128, 256, 512] {
            for case in 0..12 {
                let raw = [case - 6, case + 1, 3 - case];
                let distances = [
                    Rational::from((5, 4)),
                    Rational::from((1, 4)),
                    Rational::from((-3, 4)),
                ];
                let mut mag = Rational::new();
                let mut derivative = Rational::new();
                for (w, d) in raw.iter().zip(&distances) {
                    let term = Rational::from(*w) / d;
                    mag += term.clone().abs();
                    derivative -= term / d;
                }
                if derivative == 0 {
                    continue;
                }
                for exponent in [-700_000_000i32, 0, 700_000_000] {
                    let weights = raw
                        .iter()
                        .map(|w| Float::with_val(p, *w) << exponent)
                        .collect::<Vec<_>>();
                    let poles = [-1, 0, 1]
                        .iter()
                        .map(|x| Float::with_val(p, *x) << exponent)
                        .collect::<Vec<_>>();
                    let z = Float::with_val(p, 0.25) << exponent;
                    let (m, d, r) = evaluate(&weights, &poles, &z, p).unwrap();
                    assert_eq!(m, Float::with_val(p, &mag));
                    assert_eq!(d, Float::with_val(p, &derivative) >> exponent);
                    assert_eq!(
                        r,
                        Float::with_val(p, derivative.clone().recip()) << exponent
                    );
                }
            }
        }
    }
    #[test]
    fn singular_and_unrepresentable_conditioning_is_explicit() {
        let p = 128;
        let zero = Float::with_val(p, 0);
        let one = Float::with_val(p, 1);
        assert!(evaluate(
            std::slice::from_ref(&zero),
            std::slice::from_ref(&zero),
            &one,
            p
        )
        .is_err());
        assert!(evaluate(
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            &one,
            p
        )
        .is_err());
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        assert!(evaluate(&[tiny], &[zero], &one, p).is_err());
    }
}
