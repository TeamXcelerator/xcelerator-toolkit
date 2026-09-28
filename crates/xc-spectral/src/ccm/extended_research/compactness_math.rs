//! Arithmetic-checked point measurements of a finite stored Fourier state.
//! Internal enclosures exclude source-construction and continuum errors.
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

#[derive(Clone, Debug)]
pub(super) struct WeightedNorm {
    pub(super) rate: Float,
    pub(super) value: Float,
    pub(super) absolute_terms: Float,
}
#[derive(Clone, Debug)]
pub(super) struct Measurement {
    pub(super) origin: Float,
    pub(super) second: Float,
    pub(super) fourth: Float,
    pub(super) sigma: Option<Float>,
    pub(super) weighted: Vec<WeightedNorm>,
    pub(super) arithmetic_precision: u32,
}
fn validate(v: &[Float], requested: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&requested)
        || v.is_empty()
        || v.len().is_multiple_of(2)
        || v.len() > 16385
        || v.iter().any(|x| !x.is_finite() || x.prec() > 1_000_000)
        || v.iter().all(Float::is_zero)
    {
        bail!("unsupported finite Fourier normalization input");
    }
    Ok(())
}
fn scale(x: &Float, e: i64, p: u32) -> Result<Float> {
    let mut out = Float::with_val(p, x);
    if x.is_zero() {
        return Ok(out);
    }
    let amount = u32::try_from(e.unsigned_abs())?;
    if e >= 0 {
        out <<= amount;
    } else {
        out >>= amount;
    }
    if !out.is_finite() || out.is_zero() {
        bail!("Fourier normalization exceeds exponent range");
    }
    let mut reverse = out.clone();
    if e >= 0 {
        reverse >>= amount;
    } else {
        reverse <<= amount;
    }
    if reverse != *x {
        bail!("Fourier binary scaling loses a source component");
    }
    Ok(out)
}
fn scaled(v: &[Float], p: u32) -> Result<Vec<Float>> {
    let exponent = v.iter().filter_map(Float::get_exp).max().unwrap();
    v.iter()
        .map(|x| scale(x, -i64::from(exponent), p))
        .collect()
}
fn orientation(v: &[Float], p: u32) -> i32 {
    let n = v.len() / 2;
    let terms = v
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if i.abs_diff(n).is_multiple_of(2) {
                x.clone()
            } else {
                -x.clone()
            }
        })
        .collect::<Vec<_>>();
    let (c, dir) = Float::with_val_round(p, Float::sum(terms.iter()), Round::Nearest);
    if c < 0 || (c.is_zero() && dir == std::cmp::Ordering::Greater) {
        return -1;
    }
    if c > 0 || (c.is_zero() && dir == std::cmp::Ordering::Less) {
        return 1;
    }
    let largest = v
        .iter()
        .max_by(|a, b| (*a).clone().abs().total_cmp(&(*b).clone().abs()))
        .unwrap();
    if largest < &0 {
        -1
    } else {
        1
    }
}
/// Checked, homogeneous Euclidean unit vector at requested point precision.
pub(super) fn unit(v: &[Float], requested: u32) -> Result<Vec<Float>> {
    validate(v, requested)?;
    let p = v.iter().map(Float::prec).fold(requested, u32::max) + 64;
    let values = scaled(v, p)?;
    let sign = orientation(v, p);
    let mut norm = Float::with_val(p, 0);
    for x in &values {
        norm.hypot_mut(x);
    }
    if norm <= 0 || !norm.is_finite() {
        bail!("unresolved source norm");
    }
    values
        .iter()
        .map(|x| {
            let value = crate::ccm::retained_evidence::point::quotient(x, &norm, p)? * sign;
            let out = Float::with_val(requested, &value);
            if !out.is_finite() || (!x.is_zero() && out.is_zero()) {
                bail!("unit coefficient exceeds output exponent range");
            }
            Ok(out)
        })
        .collect()
}
fn decimal(s: &str, p: u32) -> Result<I> {
    let (lo, _) = Float::with_val_round(p, Float::parse(s)?, Round::Down);
    let (hi, _) = Float::with_val_round(p, Float::parse(s)?, Round::Up);
    Ok(I::new(lo, hi)?)
}
fn expm1(x: &I) -> Result<I> {
    x.validate()?;
    let mut lo = x.lower().clone();
    let mut hi = x.upper().clone();
    lo.exp_m1_round(Round::Down);
    hi.exp_m1_round(Round::Up);
    Ok(I::new(lo, hi)?)
}
fn abs(x: &I) -> Result<I> {
    x.validate()?;
    let a = x.lower().clone().abs();
    let b = x.upper().clone().abs();
    let hi = a.clone().max(&b);
    let lo = if x.contains_zero() {
        Float::with_val(x.precision(), 0)
    } else {
        a.min(&b)
    };
    Ok(I::new(lo, hi)?)
}
// Monotonic round-to-nearest maps a whole valid enclosure to the same point.
// A successful nonzero interval is never silently replaced by zero.
fn resolved(x: &I, requested: u32) -> Result<Option<Float>> {
    x.validate()?;
    let lo = Float::with_val(requested, x.lower());
    let hi = Float::with_val(requested, x.upper());
    if !lo.is_finite() || !hi.is_finite() {
        bail!("compactness output exceeds exponent range");
    }
    if lo != hi || (lo.is_zero() && (x.lower() != &0 || x.upper() != &0)) {
        return Ok(None);
    }
    Ok(Some(lo))
}
fn calculate(
    v: &[Float],
    cutoff: &str,
    rates: &[String],
    requested: u32,
    p: u32,
) -> Result<Option<Measurement>> {
    let zero = I::from_i64(0, p);
    let one = I::from_i64(1, p);
    let two = I::from_i64(2, p);
    let l = decimal(cutoff, p)?.ln()?;
    if !l.is_strictly_positive() {
        bail!("Fourier support must have C>1");
    }
    let values = scaled(v, p)?;
    let xi = values
        .iter()
        .map(|x| Ok(I::from_float(x, p)?))
        .collect::<Result<Vec<_>>>()?;
    let mut norm2 = zero.clone();
    for x in &xi {
        norm2 = norm2.add(&x.square());
    }
    if !norm2.is_strictly_positive() {
        bail!("source norm enclosure is unresolved");
    }
    let sign = I::from_i64(i64::from(orientation(v, p)), p);
    let normalization = sign.div(&norm2.mul(&l).sqrt()?)?;
    let n = v.len() / 2;
    let pi2 = I::pi(p).square();
    let l2 = l.square();
    let l3 = l2.mul(&l);
    let l5 = l3.mul(&l2);
    let origin = xi[n].mul(&l).mul(&normalization);
    let mut m2 = xi[n].mul(&l3).div(&I::from_i64(12, p))?;
    let mut m4 = xi[n].mul(&l5).div(&I::from_i64(80, p))?;
    for j in 1..=n {
        let paired = xi[n + j].add(&xi[n - j]);
        let q2 = pi2.mul(&I::from_u64((j * j) as u64, p));
        m2 = m2.add(&paired.mul(&l3).div(&two.mul(&q2))?);
        let kernel = one
            .div(&I::from_i64(4, p).mul(&q2))?
            .sub(&I::from_i64(3, p).div(&two.mul(&q2.square()))?);
        m4 = m4.add(&paired.mul(&l5).mul(&kernel));
    }
    m2 = m2.mul(&normalization);
    m4 = m4.mul(&normalization);
    let second = zero.sub(&m2);
    let Some(origin_point) = resolved(&origin, requested)? else {
        return Ok(None);
    };
    let Some(second_point) = resolved(&second, requested)? else {
        return Ok(None);
    };
    let Some(fourth_point) = resolved(&m4, requested)? else {
        return Ok(None);
    };
    let sigma = if values[n].is_zero() {
        None
    } else {
        let value = m2.div(&two.mul(&origin))?;
        let Some(point) = resolved(&value, requested)? else {
            return Ok(None);
        };
        Some(point)
    };
    let mut correlations = Vec::with_capacity(v.len());
    correlations.push(one.clone());
    for lag in 1..v.len() {
        let mut c = zero.clone();
        for (a, b) in xi[lag..].iter().zip(&xi[..xi.len() - lag]) {
            c = c.add(&a.mul(b));
        }
        correlations.push(c.div(&norm2)?);
    }
    let mut weighted = Vec::with_capacity(rates.len());
    for rate in rates {
        let a = decimal(rate, p)?;
        if a.lower() < &0 || a.upper() > &100 {
            bail!("compactness rate must lie in [0,100]");
        }
        let rate_point = Float::with_val(requested, Float::parse(rate)?);
        if a.lower() == &0 && a.upper() == &0 {
            weighted.push(WeightedNorm {
                rate: rate_point,
                value: Float::with_val(requested, 1),
                absolute_terms: Float::with_val(requested, 1),
            });
            continue;
        }
        let z = a.mul(&l);
        z.validate()?;
        let em1 = expm1(&z)?;
        let k0 = if z.lower() == &0 {
            // phi_1(z)=integral_0^1 exp(t*z)dt is between 1 and exp(z).
            let upper = I::from_float(z.upper(), p)?.exp();
            upper.validate()?;
            I::new(Float::with_val(p, 1), upper.upper().clone())?
        } else {
            em1.div(&z)?
        };
        let mut total = k0.clone();
        let mut absolute = k0;
        let b = two.mul(&a);
        let b2 = b.square();
        for (lag, c) in correlations.iter().enumerate().skip(1) {
            let omega = I::pi(p).mul(&I::from_u64((2 * lag) as u64, p)).div(&l)?;
            let numerator = if lag.is_multiple_of(2) {
                em1.clone()
            } else {
                em1.add(&two)
            };
            let kernel = two
                .mul(&b)
                .mul(&numerator)
                .div(&l.mul(&b2.add(&omega.square())))?;
            let term = two.mul(c).mul(&kernel);
            total = total.add(&term);
            absolute = absolute.add(&abs(&term)?);
        }
        total.validate()?;
        if total.upper() < &1 {
            bail!("weighted norm enclosure contradicts unit normalization");
        }
        let total = I::new(
            total.lower().clone().max(&Float::with_val(p, 1)),
            total.upper().clone(),
        )?;
        let Some(value) = resolved(&total, requested)? else {
            return Ok(None);
        };
        let Some(absolute_terms) = resolved(&absolute, requested)? else {
            return Ok(None);
        };
        weighted.push(WeightedNorm {
            rate: rate_point,
            value,
            absolute_terms,
        });
    }
    Ok(Some(Measurement {
        origin: origin_point,
        second: second_point,
        fourth: fourth_point,
        sigma,
        weighted,
        arithmetic_precision: p,
    }))
}
/// Finite stored-state point values accepted only when directed enclosures agree
/// after rounding to requested precision. Source/construction errors are separate.
pub(super) fn measure(
    v: &[Float],
    cutoff: &str,
    rates: &[String],
    requested: u32,
) -> Result<Measurement> {
    validate(v, requested)?;
    if rates.len() > 8 {
        bail!("too many compactness rates");
    }
    for rate in rates {
        let point = Float::with_val(requested, Float::parse(rate)?);
        if !point.is_finite()
            || !(0..=100).contains(&point)
            || (point.is_zero() && xc_core::DecimalLiteral::new(rate)?.canonical()?.as_str() != "0")
        {
            bail!("compactness rate is outside the supported finite point domain");
        }
    }
    let base = v.iter().map(Float::prec).fold(requested, u32::max);
    for guard in [64u32, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(result) = calculate(v, cutoff, rates, requested, base + guard)? {
            return Ok(result);
        }
    }
    bail!("finite compactness arithmetic unresolved within 4096 guard bits")
}

#[cfg(test)]
mod exhaustive_resumed_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_unit_rejects_partial_normalization_underflow() {
        let p = 128;
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        let small = tiny * 2u32;
        let values = [
            small.clone(),
            Float::with_val(p, 1.5),
            Float::with_val(p, 1.5),
            Float::with_val(p, 1.5),
            small,
        ];
        assert!(unit(&values, p).is_err());
    }
}
