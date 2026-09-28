//! Exact secular signs for the rank-one update of stored binary64 data.
//! Floating coordinates are only accepted after an exact bracket-width check.
use super::{
    RankOneSecularEnclosureF64, RankOneSecularSpectrumF64, RankOneSecularToleranceF64, SolverError,
};
use num_bigint::{BigInt, BigUint, Sign};
use std::cmp::Ordering;

fn breakdown(message: impl Into<String>) -> SolverError {
    SolverError::NumericalBreakdown(message.into())
}

fn dyadic(value: f64) -> (BigInt, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mantissa, power) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), exponent - 1023 - 52)
    };
    if mantissa == 0 {
        return (BigInt::from(0), 0);
    }
    let mantissa = BigInt::from(mantissa);
    (if bits >> 63 == 0 { mantissa } else { -mantissa }, power)
}

fn compare_positive_float_ratio(
    value: f64,
    numerator: &BigUint,
    denominator: &BigUint,
) -> Ordering {
    let (mantissa, power) = dyadic(value);
    let product = mantissa.magnitude() * denominator;
    if power >= 0 {
        (product << (power as usize)).cmp(numerator)
    } else {
        product.cmp(&(numerator << ((-power) as usize)))
    }
}

/// The floor/ceiling binary64 values of a nonnegative exact rational.
/// Positive bit patterns have the same order as finite positive floats.
fn positive_ratio_bounds(numerator: &BigUint, denominator: &BigUint) -> Option<(f64, f64)> {
    if numerator == &BigUint::from(0u8) {
        return Some((0.0, 0.0));
    }
    if compare_positive_float_ratio(f64::MAX, numerator, denominator) == Ordering::Less {
        return None;
    }
    let mut lo = 0u64;
    let mut hi = f64::MAX.to_bits();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if compare_positive_float_ratio(f64::from_bits(mid), numerator, denominator)
            == Ordering::Less
        {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let ceiling = f64::from_bits(lo);
    let floor = if compare_positive_float_ratio(ceiling, numerator, denominator) == Ordering::Equal
    {
        ceiling
    } else {
        f64::from_bits(lo - 1)
    };
    Some((floor, ceiling))
}

fn dyadic_bounds(mantissa: &BigInt, power: i32) -> Option<(f64, f64)> {
    let mut numerator = mantissa.magnitude().clone();
    let mut denominator = BigUint::from(1u8);
    if power >= 0 {
        numerator <<= power as usize;
    } else {
        denominator <<= (-power) as usize;
    }
    let (lo, hi) = positive_ratio_bounds(&numerator, &denominator)?;
    Some(if mantissa.sign() == Sign::Minus {
        (-hi, -lo)
    } else {
        (lo, hi)
    })
}

struct Value {
    numerator: BigInt,
    denominator: BigInt,
}
impl Value {
    fn sign(&self) -> Sign {
        self.numerator.sign()
    }
    fn absolute_upper_bound(&self) -> Result<f64, SolverError> {
        positive_ratio_bounds(self.numerator.magnitude(), self.denominator.magnitude())
            .map(|(_, upper)| upper)
            .ok_or_else(|| breakdown("secular residual has no finite binary64 upper bound"))
    }
}

struct ExactProblem<'a> {
    diagonal_values: &'a [f64],
    diagonal: Vec<BigInt>,
    weights: Vec<BigInt>,
    power: i32,
    positive: bool,
}
impl<'a> ExactProblem<'a> {
    fn new(diagonal: &'a [f64], vector: &[f64], alpha: f64) -> Self {
        let (am, ap) = dyadic(alpha);
        let diagonal_parts = diagonal.iter().map(|&v| dyadic(v)).collect::<Vec<_>>();
        let weights = vector
            .iter()
            .map(|&v| {
                let (m, p) = dyadic(v);
                (&am * &m * &m, ap + 2 * p)
            })
            .collect::<Vec<_>>();
        let power = diagonal_parts
            .iter()
            .chain(&weights)
            .map(|(_, p)| *p)
            .min()
            .expect("validated nonempty problem");
        Self {
            diagonal_values: diagonal,
            diagonal: diagonal_parts
                .into_iter()
                .map(|(m, p)| m << ((p - power) as usize))
                .collect(),
            weights: weights
                .into_iter()
                .map(|(m, p)| m << ((p - power) as usize))
                .collect(),
            power,
            positive: alpha > 0.0,
        }
    }
    fn is_pole(&self, value: f64) -> bool {
        self.diagonal_values.contains(&value)
    }
    fn evaluate(&self, value: f64) -> Result<Value, SolverError> {
        if !value.is_finite() {
            return Err(breakdown("secular coordinate must be finite"));
        }
        let (xm, xp) = dyadic(value);
        let power = self.power.min(xp);
        let shift = (self.power - power) as usize;
        let x = xm << ((xp - power) as usize);
        let mut numerator = BigInt::from(1);
        let mut denominator = BigInt::from(1);
        for (pole, weight) in self.diagonal.iter().zip(&self.weights) {
            let difference = (pole << shift) - &x;
            if difference.sign() == Sign::NoSign {
                return Err(breakdown("secular evaluation reached a pole"));
            }
            numerator = &numerator * &difference + (weight << shift) * &denominator;
            denominator *= difference;
        }
        if denominator.sign() == Sign::Minus {
            numerator = -numerator;
            denominator = -denominator;
        }
        Ok(Value {
            numerator,
            denominator,
        })
    }
    fn outer_bound(&self) -> f64 {
        let index = if self.positive {
            self.diagonal.len() - 1
        } else {
            0
        };
        let weight_sum = self.weights.iter().fold(BigInt::from(0), |sum, v| sum + v);
        let bound = &self.diagonal[index] + weight_sum;
        match dyadic_bounds(&bound, self.power) {
            Some((lo, hi)) => {
                if self.positive {
                    hi
                } else {
                    lo
                }
            }
            None => {
                if self.positive {
                    f64::MAX
                } else {
                    -f64::MAX
                }
            }
        }
    }
}

struct Bracket {
    lower: f64,
    upper: f64,
    lower_pole: bool,
    upper_pole: bool,
}
fn midpoint(lower: f64, upper: f64) -> f64 {
    if lower.is_sign_negative() != upper.is_sign_negative() {
        lower / 2.0 + upper / 2.0
    } else {
        lower + (upper - lower) / 2.0
    }
}
fn width_accepted(bracket: &Bracket, point: f64, tolerance: RankOneSecularToleranceF64) -> bool {
    super::exact_sturm_f64::difference_at_most_scaled(
        bracket.lower,
        bracket.upper,
        tolerance.absolute_width,
        1.0,
    ) || super::exact_sturm_f64::difference_at_most_scaled(
        bracket.lower,
        bracket.upper,
        tolerance.relative_width,
        point.abs(),
    )
}

fn enclosure(bracket: &Bracket, point: f64) -> RankOneSecularEnclosureF64 {
    let (lo, lp) = dyadic(bracket.lower);
    let (hi, hp) = dyadic(bracket.upper);
    let power = lp.min(hp);
    let width = (hi << ((hp - power) as usize)) - (lo << ((lp - power) as usize));
    let absolute_width_upper_bound = dyadic_bounds(&width, power).map(|(_, upper)| upper);
    let (pm, pp) = dyadic(point.abs());
    let relative_width_upper_bound = if width.sign() == Sign::NoSign {
        Some(0.0)
    } else if pm.sign() == Sign::NoSign {
        None
    } else {
        let mut numerator = width.magnitude().clone();
        let mut denominator = pm.magnitude().clone();
        if power >= pp {
            numerator <<= (power - pp) as usize;
        } else {
            denominator <<= (pp - power) as usize;
        }
        positive_ratio_bounds(&numerator, &denominator).map(|(_, upper)| upper)
    };
    RankOneSecularEnclosureF64 {
        lower: bracket.lower,
        upper: bracket.upper,
        lower_is_pole: bracket.lower_pole,
        upper_is_pole: bracket.upper_pole,
        absolute_width_upper_bound,
        relative_width_upper_bound,
    }
}
fn exact_enclosure(point: f64) -> RankOneSecularEnclosureF64 {
    enclosure(
        &Bracket {
            lower: point,
            upper: point,
            lower_pole: false,
            upper_pole: false,
        },
        point,
    )
}

fn solve_bracket(
    problem: &ExactProblem<'_>,
    mut bracket: Bracket,
    tolerance: RankOneSecularToleranceF64,
    maximum_iterations: usize,
) -> Result<(f64, f64, usize, RankOneSecularEnclosureF64), SolverError> {
    if !bracket.lower.is_finite() || !bracket.upper.is_finite() || bracket.lower >= bracket.upper {
        return Err(breakdown(
            "rank-one root has no ordered finite binary64 bracket",
        ));
    }
    let lower_sign = if problem.positive {
        Sign::Minus
    } else {
        Sign::Plus
    };
    // At a pole the one-sided limit supplies the exact sign analytically.
    for (point, is_pole, is_lower) in [
        (bracket.lower, bracket.lower_pole, true),
        (bracket.upper, bracket.upper_pole, false),
    ] {
        if is_pole {
            continue;
        }
        let value = problem.evaluate(point)?;
        if value.sign() == Sign::NoSign {
            return Ok((point, 0.0, 0, exact_enclosure(point)));
        }
        if (value.sign() == lower_sign) != is_lower {
            return Err(breakdown(
                "rank-one root cannot be bracketed within finite binary64 range",
            ));
        }
    }
    for iteration in 1..=maximum_iterations {
        let point = midpoint(bracket.lower, bracket.upper);
        if point <= bracket.lower || point >= bracket.upper || problem.is_pole(point) {
            let endpoint = if !bracket.lower_pole {
                bracket.lower
            } else if !bracket.upper_pole {
                bracket.upper
            } else {
                return Err(breakdown(
                    "binary64 has no nonpole coordinate inside this root bracket",
                ));
            };
            if width_accepted(&bracket, endpoint, tolerance) {
                return Ok((
                    endpoint,
                    problem.evaluate(endpoint)?.absolute_upper_bound()?,
                    iteration - 1,
                    enclosure(&bracket, endpoint),
                ));
            }
            return Err(breakdown(
                "binary64 cannot resolve the requested rank-one bracket width",
            ));
        }
        let value = problem.evaluate(point)?;
        if value.sign() == Sign::NoSign {
            return Ok((point, 0.0, iteration, exact_enclosure(point)));
        }
        if width_accepted(&bracket, point, tolerance) {
            return Ok((
                point,
                value.absolute_upper_bound()?,
                iteration,
                enclosure(&bracket, point),
            ));
        }
        if value.sign() == lower_sign {
            bracket.lower = point;
            bracket.lower_pole = false;
        } else {
            bracket.upper = point;
            bracket.upper_pole = false;
        }
        if width_accepted(&bracket, point, tolerance) {
            return Ok((
                point,
                value.absolute_upper_bound()?,
                iteration,
                enclosure(&bracket, point),
            ));
        }
    }
    Err(breakdown(format!("rank-one bisection exhausted {maximum_iterations} iterations before reaching the requested width")))
}

pub(super) fn solve(
    diagonal: &[f64],
    vector: &[f64],
    alpha: f64,
    tolerance: RankOneSecularToleranceF64,
    maximum_iterations: usize,
) -> Result<RankOneSecularSpectrumF64, SolverError> {
    diagonal
        .len()
        .checked_mul(maximum_iterations)
        .ok_or_else(|| breakdown("rank-one iteration count would overflow"))?;
    let problem = ExactProblem::new(diagonal, vector, alpha);
    let outer = problem.outer_bound();
    let mut brackets = Vec::with_capacity(diagonal.len());
    if alpha < 0.0 {
        brackets.push(Bracket {
            lower: outer,
            upper: diagonal[0],
            lower_pole: false,
            upper_pole: true,
        });
    }
    for pair in diagonal.windows(2) {
        brackets.push(Bracket {
            lower: pair[0],
            upper: pair[1],
            lower_pole: true,
            upper_pole: true,
        });
    }
    if alpha > 0.0 {
        brackets.push(Bracket {
            lower: *diagonal.last().expect("validated diagonal"),
            upper: outer,
            lower_pole: true,
            upper_pole: false,
        });
    }
    let mut eigenvalues = Vec::with_capacity(diagonal.len());
    let mut residuals = Vec::with_capacity(diagonal.len());
    let mut enclosures = Vec::with_capacity(diagonal.len());
    let mut total_iterations = 0usize;
    for bracket in brackets {
        let (root, residual, iterations, enclosure) =
            solve_bracket(&problem, bracket, tolerance, maximum_iterations)?;
        eigenvalues.push(root);
        residuals.push(residual);
        enclosures.push(enclosure);
        total_iterations = total_iterations
            .checked_add(iterations)
            .ok_or_else(|| breakdown("rank-one iteration count overflows"))?;
    }
    if eigenvalues.windows(2).any(|v| v[0] >= v[1]) {
        return Err(breakdown(
            "binary64 rank-one root representatives are not strictly ordered",
        ));
    }
    Ok(RankOneSecularSpectrumF64{eigenvalues,residuals,enclosures,tolerance_policy:Some(tolerance),bisection_iterations:total_iterations,algorithm:"diagonal_rank_one_explicit_width_enclosures_f64_v3".into(),assumptions:vec!["stored binary64 inputs are treated as exact dyadic rationals".into(),"strictly ordered diagonal entries and nonzero update components give complete interlacing".into(),"every returned point is an exact secular zero or meets the exact absolute/relative bracket-width tolerance".into(),"residuals are finite upward-rounded bounds on the absolute stored-data secular residual".into(),"finite matrix evidence does not certify a continuum operator or scientific identification".into()]})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_ratio_bounds_match_independent_fraction_oracle() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/rank_one_ratio_oracle.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let numerator = case["numerator"]
                .as_str()
                .unwrap()
                .parse::<BigUint>()
                .unwrap();
            let denominator = case["denominator"]
                .as_str()
                .unwrap()
                .parse::<BigUint>()
                .unwrap();
            let result = positive_ratio_bounds(&numerator, &denominator);
            if case["bounds_bits"].is_null() {
                assert!(result.is_none());
            } else {
                let (lo, hi) = result.unwrap();
                let expected = case["bounds_bits"].as_array().unwrap();
                assert_eq!(lo.to_bits(), expected[0].as_u64().unwrap());
                assert_eq!(hi.to_bits(), expected[1].as_u64().unwrap());
            }
        }
    }
}
