//! Checked projective normalization and physical-unit conversion.
use super::RankOneError;

fn invalid(message: &str) -> RankOneError {
    RankOneError::InvalidState(message.to_owned())
}

/// Preserve the supplied dyadics under a common exact power-of-two scale.
pub(in crate::ccm) fn exact_scaled_state_f64(state: &[f64]) -> Result<Vec<f64>, RankOneError> {
    use num_rational::BigRational as Q;
    use num_traits::Zero;
    if state.is_empty() || state.iter().any(|v| !v.is_finite()) {
        return Err(invalid("state coefficients must be finite and nonempty"));
    }
    let maximum = state.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return Err(invalid("state must have nonzero eta pairing"));
    }
    let exponent = (((maximum.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022);
    let scale = 2.0_f64.powi(exponent);
    let scale_q = Q::from_float(scale).expect("finite nonzero dyadic scale");
    let scaled = state.iter().map(|value| value / scale).collect::<Vec<_>>();
    for (&value, &original) in scaled.iter().zip(state) {
        if !value.is_finite()
            || Q::from_float(value).expect("finite scaled coefficient") * &scale_q
                != Q::from_float(original).expect("finite source coefficient")
        {
            return Err(invalid(
                "state dynamic range cannot be preserved during binary scaling",
            ));
        }
    }
    let sum: Q = scaled.iter().map(|v| Q::from_float(*v).unwrap()).sum();
    if sum.is_zero() {
        return Err(invalid("state eta pairing is zero"));
    }
    Ok(scaled)
}

pub(in crate::ccm) fn normalized_state_f64(state: &[f64]) -> Result<Vec<f64>, RankOneError> {
    use num_rational::BigRational as Q;
    use num_traits::ToPrimitive;
    let scaled = exact_scaled_state_f64(state)?;
    let pairing: Q = scaled.iter().map(|v| Q::from_float(*v).unwrap()).sum();
    scaled
        .iter()
        .map(|source| {
            let exact = Q::from_float(*source).expect("finite coefficient") / &pairing;
            let value = exact
                .to_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| invalid("normalized state is outside binary64 range"))?;
            if (value == 0.0 && *source != 0.0)
                || (value.is_subnormal() && Q::from_float(value).unwrap() != exact)
            {
                return Err(invalid(
                    "normalized state loses precision below the normal binary64 range",
                ));
            }
            Ok(value)
        })
        .collect()
}

pub(in crate::ccm) fn ordinate_f64(value: f64, length: f64) -> Result<f64, RankOneError> {
    if value == 0.0 {
        return Ok(value);
    }
    let factor = 2.0 * std::f64::consts::PI;
    let direct = (factor / length) * value;
    if direct.is_finite() && direct != 0.0 {
        return Ok(direct);
    }
    let magnitude = (value.abs().ln() + factor.ln() - length.ln()).exp();
    if !magnitude.is_finite() || magnitude == 0.0 {
        return Err(invalid("physical ordinate is outside binary64 range"));
    }
    Ok(magnitude.copysign(value))
}

#[cfg(feature = "hp")]
pub(super) fn normalized_state_hp(
    state: &[rug::Float],
    working: u32,
) -> Result<Vec<rug::Float>, RankOneError> {
    use rug::Float;
    let shift = state
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| invalid("state must have nonzero eta pairing"))?
        - 1;
    let scaled = state
        .iter()
        .map(|value| Float::with_val(working, value) >> shift)
        .collect::<Vec<_>>();
    if scaled
        .iter()
        .zip(state)
        .any(|(v, source)| !v.is_finite() || (v.clone() << shift) != *source)
    {
        return Err(invalid(
            "state dynamic range cannot be preserved during binary scaling",
        ));
    }
    let mut partials: Vec<Float> = Vec::new();
    for value in &scaled {
        let mut x = value.clone();
        let mut out = 0;
        for index in 0..partials.len() {
            let mut y = partials[index].clone();
            if x.clone().abs() < y.clone().abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = Float::with_val(working, &x + &y);
            let lo = y - Float::with_val(working, &hi - &x);
            if !lo.is_zero() {
                partials[out] = lo;
                out += 1;
            }
            x = hi;
        }
        partials.truncate(out);
        partials.push(x);
    }
    let sum = partials
        .iter()
        .rev()
        .fold(Float::with_val(working, 0), |sum, value| sum + value);
    if !sum.is_finite() || sum.is_zero() {
        return Err(invalid("state eta pairing is zero or unrepresentable"));
    }
    let normalized = scaled
        .iter()
        .map(|v| Float::with_val(working, v / &sum))
        .collect::<Vec<_>>();
    if normalized
        .iter()
        .zip(state)
        .any(|(v, source)| !v.is_finite() || (v.is_zero() && !source.is_zero()))
    {
        return Err(invalid("normalized state is outside the MPFR range"));
    }
    for (value, source) in normalized.iter().zip(&scaled) {
        if value.get_exp() == Some(rug::float::exp_min())
            && Float::with_val_round(working, source / &sum, rug::float::Round::Zero)
                .0
                .is_zero()
        {
            return Err(invalid(
                "normalized state loses precision below the MPFR exponent range",
            ));
        }
    }
    Ok(normalized)
}

#[cfg(feature = "hp")]
pub(super) fn ordinate_hp(
    value: &rug::Float,
    length: &rug::Float,
    working: u32,
    output: u32,
) -> Result<rug::Float, RankOneError> {
    use rug::{float::Constant, Float};
    if value.is_zero() {
        return Ok(Float::with_val(output, 0));
    }
    let factor = Float::with_val(working, Constant::Pi) * 2u32;
    let direct = Float::with_val(working, &factor / length) * value;
    let candidate = if direct.is_finite() && !direct.is_zero() {
        direct
    } else {
        let logarithm = Float::with_val(working, value).abs().ln() + factor.ln()
            - Float::with_val(working, length).ln();
        let magnitude = logarithm.exp();
        if value < &0 {
            -magnitude
        } else {
            magnitude
        }
    };
    let result = Float::with_val(output, candidate);
    if !result.is_finite() || result.is_zero() {
        return Err(invalid("physical ordinate is outside the MPFR range"));
    }
    Ok(result)
}

#[cfg(test)]
mod exhaustive_resumed_normalization_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_native_normalization_rejects_partial_underflow() {
        let tiny = f64::from_bits(1);
        for state in [[4.0, -4.0_f64.next_down(), 3.0 * tiny], [tiny, 1.5, 0.0]] {
            assert!(
                normalized_state_f64(&state).is_err(),
                "normalization accepted lost nonzero coefficient bits: {state:?}"
            );
        }
        assert!(normalized_state_f64(&[4.0, -4.0_f64.next_down(), 4.0 * tiny]).is_ok());
    }
    #[test]
    #[cfg(feature = "hp")]
    fn exhaustive_resumed_hp_normalization_rejects_partial_underflow() {
        use rug::Float;
        let p = 128;
        let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
        let four = Float::with_val(p, 4);
        let mut below = four.clone();
        below.next_down();
        for state in [
            [four.clone(), -below.clone(), tiny.clone() * 3u32],
            [tiny.clone(), Float::with_val(p, 1.5), Float::with_val(p, 0)],
        ] {
            assert!(
                normalized_state_hp(&state, p + 64).is_err(),
                "normalization accepted a partially underflowed coefficient"
            );
        }
        assert!(normalized_state_hp(&[four, -below, tiny * 4u32], p + 64).is_ok());
    }
}
