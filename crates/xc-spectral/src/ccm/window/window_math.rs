//! Exact stored-point secular sums and rational finite reach planning.
use super::WindowError;
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive, Zero};

fn error(message: &str) -> WindowError {
    WindowError::EvaluationFailed(message.into())
}

fn point(x: f64) -> Result<Q, WindowError> {
    Q::from_float(x).ok_or_else(|| error("finite binary64 point required"))
}

/// Round the exact stored-data rational sum once, after all cancellation.
/// Nonzero results that round to zero and nonfinite outputs are unresolved.
pub(super) fn secular(
    poles: &[f64],
    weights: &[f64],
    x: f64,
    derivative: bool,
) -> Result<f64, WindowError> {
    if poles.is_empty() || poles.len() != weights.len() || poles.len() > 1_000_000 {
        return Err(error("invalid secular dimensions"));
    }
    let x = point(x)?;
    let mut terms = Vec::with_capacity(poles.len());
    let mut total_bits = 0u64;
    for (&pole, &weight) in poles.iter().zip(weights) {
        let difference = &x - point(pole)?;
        if difference.is_zero() {
            return Err(error("evaluation point coincides with a pole"));
        }
        let weight = point(weight)?;
        let term = if derivative {
            -weight / (&difference * &difference)
        } else {
            weight / difference
        };
        total_bits = total_bits
            .saturating_add(term.numer().bits())
            .saturating_add(term.denom().bits());
        if total_bits > 67_108_864 {
            return Err(error("exact secular terms exceed aggregate bit budget"));
        }
        terms.push(term);
    }
    // Balanced exact addition avoids repeatedly taking a gcd between one
    // tiny term and an ever-growing full denominator. No rounding occurs.
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        for pair in terms.chunks(2) {
            let sum = if pair.len() == 2 {
                &pair[0] + &pair[1]
            } else {
                pair[0].clone()
            };
            if sum.numer().bits() > 4_194_304 || sum.denom().bits() > 4_194_304 {
                return Err(error("exact secular sum exceeds rational bit budget"));
            }
            next.push(sum);
        }
        terms = next;
    }
    let sum = terms.pop().expect("validated nonempty secular input");
    let value = sum
        .to_f64()
        .ok_or_else(|| error("unrepresentable secular result"))?;
    if !value.is_finite() || (value == 0.0 && !sum.is_zero()) {
        return Err(error(
            "nonzero secular result underflows or exceeds binary64 range",
        ));
    }
    Ok(value)
}

/// Compare exact stored-point distances without overflowing a difference or
/// multiplying a subnormal tolerance in binary64.
pub(super) fn within_distance(
    a: f64,
    b: f64,
    tolerance: f64,
    multiple: u32,
) -> Result<bool, WindowError> {
    if tolerance <= 0.0 || multiple == 0 {
        return Err(error("invalid distance tolerance"));
    }
    Ok((point(a)? - point(b)?).abs() <= point(tolerance)? * Q::from_integer(multiple.into()))
}

/// Mixed absolute/relative width: exact comparison on the retained binary64 points.
pub(super) fn within_scaled_distance(a: f64, b: f64, tolerance: f64) -> Result<bool, WindowError> {
    if tolerance <= 0.0 {
        return Err(error("invalid distance tolerance"));
    }
    let a = point(a)?;
    let b = point(b)?;
    let scale = a.abs().max(b.abs()).max(Q::from_integer(1.into()));
    Ok((a - b).abs() <= point(tolerance)? * scale)
}

// For 0<=t<=1/3, log((1+t)/(1-t)) is the positive atanh series.
// The remaining denominators are at least 2N+1, giving a geometric upper tail.
fn log_series(t: Q, n: u32) -> (Q, Q) {
    let square = &t * &t;
    let mut term = t;
    let mut sum = Q::zero();
    for k in 0..n {
        sum += &term / Q::from_integer((2 * k + 1).into());
        term *= &square;
    }
    sum *= Q::from_integer(2.into());
    let tail = term * Q::from_integer(2.into())
        / (Q::from_integer((2 * n + 1).into()) * (Q::from_integer(1.into()) - square));
    (sum.clone(), sum + tail)
}

fn log_bounds(value: f64, n: u32) -> Result<(Q, Q), WindowError> {
    if !value.is_finite() || value <= 1.0 {
        return Err(error(
            "logarithm planning requires a finite value greater than one",
        ));
    }
    let exponent = ((value.to_bits() >> 52) & 0x7ff) as u32 - 1023;
    let mantissa = f64::from_bits((value.to_bits() & ((1u64 << 52) - 1)) | (1023u64 << 52));
    let m = point(mantissa)?;
    let one = Q::from_integer(1.into());
    let (lo, hi) = log_series((&m - &one) / (&m + &one), n);
    if exponent == 0 {
        return Ok((lo, hi));
    }
    let (two_lo, two_hi) = log_series(Q::new(1.into(), 3.into()), n);
    let e = Q::from_integer(exponent.into());
    Ok((lo + two_lo * &e, hi + two_hi * e))
}

// Machin's identity pi=16 atan(1/5)-4 atan(1/239), enclosed by
// alternating-series remainders. The even number of terms ends below atan.
fn pi_bounds() -> (Q, Q) {
    fn atan(d: u32) -> (Q, Q) {
        let t = Q::new(1.into(), d.into());
        let square = &t * &t;
        let mut term = t;
        let mut sum = Q::zero();
        for k in 0..32u32 {
            let value = &term / Q::from_integer((2 * k + 1).into());
            if k % 2 == 0 {
                sum += value;
            } else {
                sum -= value;
            }
            term *= &square;
        }
        (sum.clone(), sum + term / Q::from_integer(65.into()))
    }
    let (a, b) = atan(5);
    let (c, d) = atan(239);
    (
        a * Q::from_integer(16.into()) - d * Q::from_integer(4.into()),
        b * Q::from_integer(16.into()) - c * Q::from_integer(4.into()),
    )
}

/// Resolve ceil(height*ln(C)/(2*pi)) from exact rational enclosures.
/// An unresolved integer boundary or unrepresentable count returns an error.
pub(super) fn minimum_modes(c: f64, height: f64) -> Result<usize, WindowError> {
    if !c.is_finite() || c <= 1.0 || !height.is_finite() || height < 0.0 {
        return Err(error("invalid finite mode-reach inputs"));
    }
    if height == 0.0 {
        return Ok(0);
    }
    let h = point(height)?;
    let (pi_lo, pi_hi) = pi_bounds();
    let two = Q::from_integer(2.into());
    for n in [32, 64, 128] {
        let (lo, hi) = log_bounds(c, n)?;
        let lower = (&h * lo / (&two * &pi_hi)).ceil().to_integer();
        let upper = (&h * hi / (&two * &pi_lo)).ceil().to_integer();
        let count = upper
            .to_usize()
            .ok_or_else(|| error("minimum mode count does not fit this platform"))?;
        if lower == upper {
            return Ok(count);
        }
    }
    Err(error(
        "minimum mode count unresolved at an integer boundary",
    ))
}

pub(super) fn precision_bits(target: u32, guard: u32) -> Result<u64, WindowError> {
    let digits = u64::from(target) + u64::from(guard);
    let d = Q::from_integer(digits.into());
    for n in [32, 64, 128] {
        let (ten_lo, ten_hi) = log_bounds(10.0, n)?;
        let (two_lo, two_hi) = log_bounds(2.0, n)?;
        let lower = (&d * ten_lo / two_hi).ceil().to_integer();
        let upper = (&d * ten_hi / two_lo).ceil().to_integer();
        if lower == upper {
            return upper
                .to_u64()
                .map(|bits| bits.max(64))
                .ok_or_else(|| error("precision recommendation exceeds u64"));
        }
    }
    Err(error(
        "precision recommendation unresolved at an integer boundary",
    ))
}

/// Exact decimal spelling of a positive finite binary64 tolerance.
pub(super) fn tolerance_decimal(tolerance: f64) -> Result<String, WindowError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(error("invalid duplicate tolerance"));
    }
    let q = point(tolerance)?;
    let power = q.denom().bits() - 1;
    let mut numerator = q.numer().clone();
    for _ in 0..power {
        numerator *= 5u32;
    }
    Ok(format!("{numerator}e-{power}"))
}

#[cfg(test)]
mod tests {
    use super as window_math;
    #[test]
    fn exact_secular_equations_match_independent_python_fraction_rounding() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/window_math_oracle.json"
        ))
        .unwrap();
        let mut count = 0;
        for row in data["secular"].as_array().unwrap() {
            let points = |name: &str| {
                row[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            };
            let p = points("poles");
            let w = points("weights");
            let x = row["x"].as_f64().unwrap();
            let derivative = row["derivative"].as_bool().unwrap();
            let got = window_math::secular(&p, &w, x, derivative);
            if let Some(bits) = row["expected_bits"].as_u64() {
                assert_eq!(got.unwrap().to_bits(), bits, "case {count}");
            } else {
                assert!(got.is_err(), "case {count}");
            }
            count += 1;
        }
        assert_eq!(count, 677);
    }
    #[test]
    fn rational_planning_bounds_match_independent_high_precision_transcendentals() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/window_math_oracle.json"
        ))
        .unwrap();
        for row in data["planning"].as_array().unwrap() {
            let got = window_math::minimum_modes(
                row["c"].as_f64().unwrap(),
                row["height"].as_f64().unwrap(),
            );
            if let Some(n) = row["minimum"].as_u64() {
                assert_eq!(got.unwrap(), usize::try_from(n).unwrap());
            } else {
                assert!(got.is_err());
            }
        }
        for row in data["precision"].as_array().unwrap() {
            assert_eq!(
                window_math::precision_bits(
                    row["target"].as_u64().unwrap() as u32,
                    row["guard"].as_u64().unwrap() as u32
                )
                .unwrap(),
                row["bits"].as_u64().unwrap()
            );
        }
    }
}
