//! Correctly rounded implicit root motion of exact stored point data.
//! Directed, reversibly scaled arithmetic excludes source/assembly error.
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;
const GUARDS: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

pub(super) fn validate(values: &[Float], p: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&p)
        || values.is_empty()
        || values.len() > 16385
        || values.iter().any(|x| !x.is_finite() || x.prec() > p)
        || values.len() as u128 * (u128::from(p + 4096).div_ceil(8) + 96) * 24 > (8u128 << 30)
    {
        bail!("invalid root-response precision, finite source, shape or workspace");
    }
    Ok(())
}
fn exponent(values: &[Float]) -> Option<i64> {
    values
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map(i64::from)
}

pub(super) struct Geometry {
    denominators: Option<Vec<I>>,
    derivative: I,
    root: I,
    coordinates: i64,
    amplitudes: i64,
}
impl Geometry {
    pub(super) fn prepare(
        weights: &[Float],
        poles: &[Float],
        root: &Float,
        p: u32,
        guard: u32,
        retain: bool,
    ) -> Result<Self> {
        validate(weights, p)?;
        validate(poles, p)?;
        validate(std::slice::from_ref(root), p)?;
        if weights.len() != poles.len() || !GUARDS.contains(&guard) {
            bail!("invalid root-response geometry");
        }
        let work = p + guard;
        let coordinates = poles
            .iter()
            .chain(std::iter::once(root))
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        let amplitudes = exponent(weights)
            .ok_or_else(|| anyhow::anyhow!("root response has a zero secular derivative"))?;
        let root = I::point(scale_float(root, -coordinates, work)?);
        let mut out = Self {
            denominators: retain.then(|| Vec::with_capacity(poles.len())),
            derivative: I::from_i64(0, work),
            root,
            coordinates,
            amplitudes,
        };
        for (weight, pole) in weights.iter().zip(poles) {
            let denominator = out.separation(pole)?;
            let term = I::point(scale_float(weight, -amplitudes, work)?)
                .div(&denominator)?
                .div(&denominator)?;
            out.derivative = out.derivative.add(&term);
            if let Some(stored) = &mut out.denominators {
                stored.push(denominator);
            }
        }
        out.derivative.validate()?;
        if out.derivative.lower().is_zero() && out.derivative.upper().is_zero() {
            bail!("root response has a zero secular derivative");
        }
        Ok(out)
    }
    fn separation(&self, pole: &Float) -> Result<I> {
        let d = self.root.sub(&I::point(scale_float(
            pole,
            -self.coordinates,
            self.root.precision(),
        )?));
        d.validate()?;
        if d.contains_zero() {
            bail!("root response encountered a secular pole or unresolved pole separation");
        }
        Ok(d)
    }
    fn response(
        &self,
        weights: &[Float],
        tangent: &[Float],
        poles: &[Float],
        motions: Option<&[Float]>,
        p: u32,
    ) -> Result<Option<Float>> {
        if self.derivative.contains_zero() {
            return Ok(None);
        }
        let work = self.root.precision();
        let tangent_exponent = exponent(tangent);
        let motion_exponent = motions.and_then(exponent);
        let fixed_exponent = tangent_exponent.map(|e| e - self.amplitudes + self.coordinates);
        let output_exponent = fixed_exponent
            .into_iter()
            .chain(motion_exponent)
            .max()
            .unwrap_or(0);
        let mut fixed = I::from_i64(0, work);
        let mut moving = I::from_i64(0, work);
        for (index, pole) in poles.iter().enumerate() {
            let temporary;
            let denominator = if let Some(stored) = &self.denominators {
                &stored[index]
            } else {
                temporary = self.separation(pole)?;
                &temporary
            };
            if let Some(e) = tangent_exponent {
                fixed =
                    fixed.add(&I::point(scale_float(&tangent[index], -e, work)?).div(denominator)?);
            }
            if let (Some(velocities), Some(e)) = (motions, motion_exponent) {
                let term = I::point(scale_float(&weights[index], -self.amplitudes, work)?)
                    .mul(&I::point(scale_float(&velocities[index], -e, work)?))
                    .div(denominator)?
                    .div(denominator)?;
                moving = moving.add(&term);
            }
        }
        let mut source = I::from_i64(0, work);
        if let Some(e) = fixed_exponent {
            source = source.add(&scale(&fixed, e - output_exponent)?);
        }
        if let Some(e) = motion_exponent {
            source = source.add(&scale(&moving, e - output_exponent)?);
        }
        let response = scale(&source.div(&self.derivative)?, output_exponent)?;
        response.validate()?;
        let lo = Float::with_val(p, response.lower());
        let hi = Float::with_val(p, response.upper());
        if lo == hi
            && lo.is_finite()
            && (!lo.is_zero() || (response.lower().is_zero() && response.upper().is_zero()))
        {
            return Ok(Some(lo));
        }
        Ok(None)
    }
}

pub(super) fn evaluate(
    weights: &[Float],
    tangent: &[Float],
    poles: &[Float],
    motions: Option<&[Float]>,
    root: &Float,
    p: u32,
    prepared: Option<&Geometry>,
) -> Result<Float> {
    validate(weights, p)?;
    validate(tangent, p)?;
    validate(poles, p)?;
    validate(std::slice::from_ref(root), p)?;
    if weights.len() != poles.len() || tangent.len() != poles.len() {
        bail!("root-response source dimensions differ");
    }
    if let Some(motions) = motions {
        validate(motions, p)?;
        if motions.len() != poles.len() {
            bail!("root-response pole-motion dimensions differ");
        }
    }
    for guard in GUARDS {
        if guard == 64 {
            if let Some(g) = prepared {
                if let Some(value) = g.response(weights, tangent, poles, motions, p)? {
                    return Ok(value);
                }
                continue;
            }
        }
        let g = Geometry::prepare(weights, poles, root, p, guard, false)?;
        if let Some(value) = g.response(weights, tangent, poles, motions, p)? {
            return Ok(value);
        }
    }
    bail!("root-response rounding or nonzero derivative unresolved within 4096 guard bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;
    #[test]
    fn exhaustive_response_range_independent_rational_oracles() {
        let mut checked = 0;
        for p in [64, 128, 256] {
            for case in 0i32..12 {
                let w = [case - 4, case + 1, 2 - case];
                let t = [case + 3, 1 - case, case - 7];
                let v = [case - 2, 3 - case, case + 1];
                let positions: [i32; 3] = [-2, -1, 1];
                let z = Rational::from((1, 4));
                let mut derivative = Rational::new();
                let mut fixed = Rational::new();
                let mut moving = Rational::new();
                for i in 0..3 {
                    let d = z.clone() - positions[i];
                    derivative += Rational::from(w[i]) / d.clone().square();
                    fixed += Rational::from(t[i]) / &d;
                    moving += Rational::from(w[i] * v[i]) / d.square();
                }
                assert_ne!(derivative, 0);
                let expected_fixed = Float::with_val(p, fixed.clone() / &derivative);
                let expected_total = Float::with_val(p, (fixed + moving) / &derivative);
                for c in [-700_000_000i32, 0, 700_000_000] {
                    for amplitude in [-700_000_000i32, 0, 700_000_000] {
                        let weights = w.map(|x| Float::with_val(p, x) << amplitude);
                        let tangent = t.map(|x| Float::with_val(p, x) << amplitude);
                        let poles = positions.map(|x| Float::with_val(p, x) << c);
                        let motions = v.map(|x| Float::with_val(p, x) << c);
                        let root = Float::with_val(p, &z) << c;
                        assert_eq!(
                            evaluate(&weights, &tangent, &poles, None, &root, p, None).unwrap(),
                            expected_fixed.clone() << c
                        );
                        assert_eq!(
                            evaluate(&weights, &tangent, &poles, Some(&motions), &root, p, None)
                                .unwrap(),
                            expected_total.clone() << c
                        );
                        for retain in [false, true] {
                            let geometry =
                                Geometry::prepare(&weights, &poles, &root, p, 64, retain).unwrap();
                            assert_eq!(
                                evaluate(
                                    &weights,
                                    &tangent,
                                    &poles,
                                    None,
                                    &root,
                                    p,
                                    Some(&geometry)
                                )
                                .unwrap(),
                                expected_fixed.clone() << c
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 324);
    }
    #[test]
    fn exhaustive_response_range_zero_motion_and_source_contracts() {
        let p = 128;
        let w = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let zeros = vec![Float::with_val(p, 0); 2];
        let poles = [Float::with_val(p, -1), Float::with_val(p, 1)];
        let root = Float::with_val(p, 0);
        assert_eq!(
            evaluate(&w, &zeros, &poles, Some(&zeros), &root, p, None).unwrap(),
            0
        );
        assert!(evaluate(&w[..1], &zeros, &poles, None, &root, p, None).is_err());
        assert!(evaluate(&w, &zeros, &poles, Some(&zeros[..1]), &root, p, None).is_err());
        assert!(evaluate(&w, &zeros, &poles, None, &poles[0], p, None).is_err());
        assert!(evaluate(&zeros, &zeros, &poles, None, &root, p, None).is_err());
        let cancelling = [Float::with_val(p, 1), Float::with_val(p, -1)];
        assert!(evaluate(&cancelling, &zeros, &poles, None, &root, p, None).is_err());
        for bad in [0, 63, 1_000_001, u32::MAX] {
            assert!(validate(&w, bad).is_err());
        }
        assert!(validate(&vec![Float::with_val(p, 0); 16386], p).is_err());
    }
}
