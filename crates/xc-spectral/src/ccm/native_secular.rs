//! Checked point discovery for the even CCM secular equation.
//!
//! Search multiple positive roots per pole gap using bounded binary64 subdivision. With `p_j = j²`,
//! `R(t) = A + Σ c_j/(t-p_j)` where `A = ξ_0 + 2Σξ_j` and `c_j = 2ξ_j p_j`.
//! A pole gap or the exterior window can hold any number of roots, so each
//! window is subdivided using root-exclusion and monotonicity estimates.
//! These binary64 exclusion margins are not directed interval certificates.
//! Accepted signs and zeros use bounded exact rational evaluation of the stored
//! original binary64 source under an exact common dyadic scale (at most 4096 active poles). This does not certify
//! completeness of heuristic exclusions. Interior pieces use the centered form
//! `1/(x-p) = Σ_{k<K} (-1)^k (x-m)^k/(m-p)^(k+1) + (-1)^K (x-m)^K/((m-p)^K (x-p))`,
//! whose exact remainder stays small although the terms cancel strongly.
//! Pieces touching a pole use the monotonicity of each term away from its pole.
use anyhow::{anyhow, Result};
use xc_root::{RealFunctionF64, RootBracketF64, RootError, RootStoppingF64};

/// Order of the centered form. The remainder carries `(r/|m-p|)^ORDER`, which
/// must dominate cancellation among the partial-fraction terms.
const ORDER: usize = 16;
const MAX_EXACT_WORKSPACE_BITS: u64 = 8_000_000;

// Bound inputs before combined rational arithmetic: the numerator/denominator
// bit count of a sum, product, or quotient is at most the combined input bit
// count plus two carry bits. Exhaustion refuses the hint and falls back to
// subdivision, whose existing finite work budget remains authoritative.
fn exact_workspace_available(values: &[&num_rational::BigRational]) -> bool {
    values
        .iter()
        .try_fold(2_u64, |bits, value| {
            bits.checked_add(value.numer().bits())?
                .checked_add(value.denom().bits())
        })
        .is_some_and(|bits| bits <= MAX_EXACT_WORKSPACE_BITS)
}

struct EvenSecular<'a> {
    positive: &'a [f64],
    /// Evaluate `S(t) = R(t)/t`, which has the same positive roots but not
    /// the removable root at `t = 0` present when `ξ_0 = 0`.
    reduced: bool,
}
impl RealFunctionF64 for EvenSecular<'_> {
    fn evaluate(&self, t: f64) -> std::result::Result<f64, RootError> {
        use num_rational::BigRational as Q;
        use num_traits::{Signed, Zero};
        let tq = Q::from_float(t)
            .ok_or_else(|| RootError::Evaluation("nonfinite secular argument".into()))?;
        let mut sum = Q::zero();
        for (j, value) in self
            .positive
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, v)| **v != 0.0)
        {
            let denominator = &tq - Q::from_integer((j * j).into());
            if denominator.is_zero() {
                return Err(RootError::PoleCollision(
                    "secular point equals an active pole".into(),
                ));
            }
            sum += Q::from_float(*value).expect("finite validated weight") / denominator;
            if sum.numer().bits() + sum.denom().bits() > MAX_EXACT_WORKSPACE_BITS {
                return Err(RootError::NonConvergence(
                    "native secular exact-sign workspace exhausted".into(),
                ));
            }
        }
        let value = if self.reduced {
            sum * Q::from_integer(2.into())
        } else {
            Q::from_float(self.positive[0]).expect("finite validated center")
                + tq * sum * Q::from_integer(2.into())
        };
        // This API is consumed only for signs and exact zeros. Returning a
        // unit sign prevents overflow and prevents the smallest subnormal
        // from satisfying the bisection residual shortcut as a nonzero value.
        Ok(if value.is_zero() {
            0.0
        } else if value.is_negative() {
            -1.0
        } else {
            1.0
        })
    }
}

/// The same function as `constant + Σ residue_j/(t - pole_j)`, used only to
/// classify pieces. `rounding` scales every magnitude into an error margin.
struct PartialFractions {
    constant: f64,
    constant_error: f64,
    poles: Vec<f64>,
    residues: Vec<f64>,
    rounding: f64,
    exact_constant: num_rational::BigRational,
    exact_residues: Vec<num_rational::BigRational>,
}

#[derive(Clone, Copy, PartialEq)]
enum End {
    Regular,
    Pole(usize),
}

#[derive(Clone, Copy)]
struct Piece {
    lower: f64,
    upper: f64,
    left: End,
    right: End,
}

enum Verdict {
    NoRoot,
    Monotone,
    Unknown,
}

impl PartialFractions {
    /// A floating hint may guide work, but only an exact source-bound
    /// centered-form test may exclude a piece or assert monotonicity.
    fn exact_interior_hint(&self, piece: &Piece, monotone: bool) -> bool {
        use num_rational::BigRational as Q;
        use num_traits::{Signed, Zero};
        let two = Q::from_integer(2.into());
        let one = Q::from_integer(1.into());
        let lower = Q::from_float(piece.lower).expect("finite endpoint");
        let upper = Q::from_float(piece.upper).expect("finite endpoint");
        let middle = (&lower + &upper) / &two;
        let radius = (&upper - &lower) / &two;
        let mut coefficients = vec![Q::zero(); ORDER];
        let mut remainder = Q::zero();
        let mut slope_remainder = Q::zero();
        for (&pole, residue) in self.poles.iter().zip(&self.exact_residues) {
            let delta = &middle - Q::from_float(pole).expect("exact integer pole");
            if delta.is_zero() {
                return false;
            }
            let ratio = &radius / &delta;
            let size = ratio.abs();
            if size >= &one / &two {
                return false;
            }
            let base = residue / delta;
            let mut power = one.clone();
            for coefficient in &mut coefficients {
                if !exact_workspace_available(&[coefficient, &base, &power, &ratio]) {
                    return false;
                }
                *coefficient += &base * &power;
                power *= -&ratio;
            }
            if !exact_workspace_available(&[&base, &power, &one, &size]) {
                return false;
            }
            let tail = (&base * &power).abs() / (&one - &size);
            let order = Q::from_integer(ORDER.into());
            if !exact_workspace_available(&[
                &slope_remainder,
                &remainder,
                &tail,
                &order,
                &size,
                &one,
                &size,
            ]) {
                return false;
            }
            slope_remainder += &tail * (Q::from_integer(ORDER.into()) + &size / (&one - &size));
            remainder += tail;
        }
        if !exact_workspace_available(&[&coefficients[0], &self.exact_constant]) {
            return false;
        }
        coefficients[0] += &self.exact_constant;
        let mut spread = Q::zero();
        for (k, c) in coefficients
            .iter()
            .enumerate()
            .skip(if monotone { 2 } else { 1 })
        {
            let multiplier = Q::from_integer((if monotone { k } else { 1 }).into());
            if !exact_workspace_available(&[&spread, &multiplier, c]) {
                return false;
            }
            spread += multiplier * c.abs();
        }
        let remainder = if monotone { slope_remainder } else { remainder };
        let leading = &coefficients[usize::from(monotone)];
        exact_workspace_available(&[leading, &spread, &remainder])
            && leading.abs() > spread + remainder
    }

    fn exact_pole_exclusion(&self, piece: &Piece) -> bool {
        use num_rational::BigRational as Q;
        use num_traits::{Signed, Zero};
        let mut lower = Some(self.exact_constant.clone());
        let mut upper = Some(self.exact_constant.clone());
        for (index, (&pole, residue)) in self.poles.iter().zip(&self.exact_residues).enumerate() {
            let pole = Q::from_float(pole).expect("exact integer pole");
            let (lo, hi) = if piece.left == End::Pole(index) {
                let v = residue / (Q::from_float(piece.upper).unwrap() - &pole);
                if residue.is_positive() {
                    (Some(v), None)
                } else {
                    (None, Some(v))
                }
            } else if piece.right == End::Pole(index) {
                let v = residue / (Q::from_float(piece.lower).unwrap() - &pole);
                if residue.is_positive() {
                    (None, Some(v))
                } else {
                    (Some(v), None)
                }
            } else {
                let a = residue / (Q::from_float(piece.lower).unwrap() - &pole);
                let b = residue / (Q::from_float(piece.upper).unwrap() - &pole);
                (Some(a.clone().min(b.clone())), Some(a.max(b)))
            };
            if lower
                .as_ref()
                .zip(lo.as_ref())
                .is_some_and(|(a, b)| !exact_workspace_available(&[a, b]))
                || upper
                    .as_ref()
                    .zip(hi.as_ref())
                    .is_some_and(|(a, b)| !exact_workspace_available(&[a, b]))
            {
                return false;
            }
            lower = lower.zip(lo).map(|(a, b)| a + b);
            upper = upper.zip(hi).map(|(a, b)| a + b);
        }
        lower.is_some_and(|v| v > Q::zero()) || upper.is_some_and(|v| v < Q::zero())
    }

    fn exact_exterior_exclusion(&self, t: f64) -> bool {
        use num_rational::BigRational as Q;
        use num_traits::{Signed, Zero};
        let t = Q::from_float(t).expect("finite exterior endpoint");
        let one = Q::from_integer(1.into());
        let mut deviation = Q::zero();
        for k in 0..=ORDER {
            let mut moment = Q::zero();
            let mut tail = Q::zero();
            for (&pole, residue) in self.poles.iter().zip(&self.exact_residues) {
                let ratio = Q::from_float(pole).unwrap() / &t;
                if ratio >= one {
                    return false;
                }
                let mut term = residue / &t;
                for _ in 0..k {
                    if !exact_workspace_available(&[&term, &ratio]) {
                        return false;
                    }
                    term *= &ratio;
                }
                if k == ORDER {
                    if !exact_workspace_available(&[&tail, &term, &one, &ratio]) {
                        return false;
                    }
                    tail += term.abs() / (&one - &ratio);
                } else {
                    if !exact_workspace_available(&[&moment, &term]) {
                        return false;
                    }
                    moment += term;
                }
            }
            let contribution = if k == ORDER { tail } else { moment.abs() };
            if !exact_workspace_available(&[&deviation, &contribution]) {
                return false;
            }
            deviation += contribution;
        }
        self.exact_constant.abs() > deviation
    }

    /// Classify a closed piece containing no pole. Coefficients are scaled by
    /// powers of `r/(m-p)` with magnitude below 1/2, so nothing overflows.
    fn classify_interior(&self, piece: &Piece) -> Verdict {
        let middle = piece.lower.midpoint(piece.upper);
        let radius = 0.5 * piece.upper - 0.5 * piece.lower;
        let mut coefficients = [0.0_f64; ORDER];
        let mut magnitudes = [0.0_f64; ORDER];
        let (mut remainder, mut slope_remainder) = (0.0_f64, 0.0_f64);
        for (&pole, &residue) in self.poles.iter().zip(&self.residues) {
            let delta = middle - pole;
            let ratio = radius / delta;
            let size = ratio.abs();
            if !size.is_finite() || size >= 0.5 {
                return Verdict::Unknown;
            }
            let base = residue / delta;
            let mut power = 1.0;
            for k in 0..ORDER {
                coefficients[k] += base * power;
                magnitudes[k] += (base * power).abs();
                power *= -ratio;
            }
            let tail = (base * power).abs() / (1.0 - size);
            remainder += tail;
            slope_remainder += tail * (ORDER as f64 + size / (1.0 - size));
        }
        coefficients[0] += self.constant;
        let noise = |k: usize| self.rounding * magnitudes[k];
        let value_spread = (1..ORDER)
            .map(|k| coefficients[k].abs() + noise(k))
            .sum::<f64>()
            + noise(0)
            + self.constant_error
            + remainder;
        if coefficients[0].abs() > value_spread * (1.0 + self.rounding) {
            return if self.exact_interior_hint(piece, false) {
                Verdict::NoRoot
            } else {
                Verdict::Unknown
            };
        }
        let slope_spread = (2..ORDER)
            .map(|k| k as f64 * (coefficients[k].abs() + noise(k)))
            .sum::<f64>()
            + noise(1)
            + slope_remainder;
        if coefficients[1].abs() > slope_spread * (1.0 + self.rounding) {
            return if self.exact_interior_hint(piece, true) {
                Verdict::Monotone
            } else {
                Verdict::Unknown
            };
        }
        Verdict::Unknown
    }

    /// Estimate root exclusion for a piece with one pole endpoint. Away from its pole each
    /// term is monotone; the pole term is unbounded with the residue's sign on
    /// the right of its pole and the opposite sign on the left.
    fn excludes_pole_piece(&self, piece: &Piece) -> bool {
        let (mut lower, mut upper) = (self.constant, self.constant);
        let mut magnitude = 0.0_f64;
        for (index, (&pole, &residue)) in self.poles.iter().zip(&self.residues).enumerate() {
            if piece.left == End::Pole(index) {
                let value = residue / (piece.upper - pole);
                magnitude += value.abs();
                if residue > 0.0 {
                    lower += value;
                    upper = f64::INFINITY;
                } else {
                    upper += value;
                    lower = f64::NEG_INFINITY;
                }
            } else if piece.right == End::Pole(index) {
                let value = residue / (piece.lower - pole);
                magnitude += value.abs();
                if residue > 0.0 {
                    upper += value;
                    lower = f64::NEG_INFINITY;
                } else {
                    lower += value;
                    upper = f64::INFINITY;
                }
            } else {
                let a = residue / (piece.lower - pole);
                let b = residue / (piece.upper - pole);
                lower += a.min(b);
                upper += a.max(b);
                magnitude += a.abs().max(b.abs());
            }
        }
        let margin = self.rounding * magnitude + self.constant_error;
        (lower - margin > 0.0 || upper + margin < 0.0) && self.exact_pole_exclusion(piece)
    }

    /// A finite `T` beyond the last pole with no root in `[T, ∞)`. For `t ≥ T`,
    /// `R(t) - A = Σ_{k<K} Σ_j c_j p_j^k / t^(k+1) + Σ_j c_j p_j^K / (t^K (t-p_j))`,
    /// and every term's magnitude decreases in `t`.
    fn exterior_bound(&self, last_pole: f64) -> Option<f64> {
        let mut t = 2.0 * last_pole;
        while t.is_finite() {
            let mut deviation = self.constant_error;
            for k in 0..=ORDER {
                let (mut moment, mut magnitude) = (0.0_f64, 0.0_f64);
                for (&pole, &residue) in self.poles.iter().zip(&self.residues) {
                    let ratio = pole / t;
                    let term = residue / t * ratio.powi(k as i32);
                    if k == ORDER {
                        magnitude += term.abs() / (1.0 - ratio);
                    } else {
                        moment += term;
                        magnitude += term.abs();
                    }
                }
                deviation += if k == ORDER {
                    magnitude
                } else {
                    moment.abs() + self.rounding * magnitude
                };
            }
            if self.constant.abs() > deviation * (1.0 + self.rounding)
                && self.exact_exterior_exclusion(t)
            {
                return Some(t);
            }
            t *= 2.0;
        }
        None
    }
}

fn opposite(a: f64, b: f64) -> bool {
    a != 0.0 && b != 0.0 && a.is_sign_positive() != b.is_sign_positive()
}

fn search_window(
    function: &EvenSecular<'_>,
    classes: &PartialFractions,
    window: Piece,
    stopping: &RootStoppingF64,
    budget: &mut usize,
    roots: &mut Vec<f64>,
) -> Result<()> {
    enum Work {
        Piece(Piece),
        Root(f64),
    }
    let mut stack = vec![Work::Piece(window)];
    while let Some(work) = stack.pop() {
        let piece = match work {
            Work::Root(t) => {
                roots.push(t);
                continue;
            }
            Work::Piece(piece) => piece,
        };
        *budget = budget
            .checked_sub(1)
            .ok_or_else(|| anyhow!("CCM point discovery exhausted its subdivision budget"))?;
        let middle = piece.lower.midpoint(piece.upper);
        let splittable = middle > piece.lower && middle < piece.upper;
        match (piece.left, piece.right) {
            (End::Pole(_), End::Pole(_)) => {}
            (End::Pole(_), End::Regular) | (End::Regular, End::Pole(_)) => {
                if classes.excludes_pole_piece(&piece) {
                    continue;
                }
                if !splittable {
                    return Err(anyhow!(
                        "CCM root/pole separation is unresolved at binary64 spacing"
                    ));
                }
            }
            (End::Regular, End::Regular) => match classes.classify_interior(&piece) {
                Verdict::NoRoot => continue,
                Verdict::Monotone => {
                    let lower = function.evaluate(piece.lower)?;
                    let upper = function.evaluate(piece.upper)?;
                    if opposite(lower, upper) {
                        let bracket = RootBracketF64 {
                            lower: piece.lower,
                            upper: piece.upper,
                        };
                        roots.push(xc_root::bisect_f64(function, bracket, stopping)?.midpoint);
                    }
                    continue;
                }
                Verdict::Unknown => {
                    let limit = stopping
                        .absolute_x_tolerance
                        .max(stopping.relative_x_tolerance * middle.abs().max(1.0));
                    if !splittable || piece.upper - piece.lower <= limit {
                        return Err(anyhow!(
                            "CCM point discovery cannot separate roots near t={middle:e} in binary64"
                        ));
                    }
                }
            },
        }
        let value = function.evaluate(middle)?;
        stack.push(Work::Piece(Piece {
            lower: middle,
            upper: piece.upper,
            left: End::Regular,
            right: piece.right,
        }));
        if value == 0.0 {
            stack.push(Work::Root(middle));
        }
        stack.push(Work::Piece(Piece {
            lower: piece.lower,
            upper: middle,
            left: piece.left,
            right: End::Regular,
        }));
    }
    Ok(())
}

pub(super) fn solve(
    xi: &[f64],
    n: usize,
    length: f64,
    tolerance: f64,
    maximum_iterations: usize,
) -> Result<Vec<f64>> {
    let expected = n.checked_mul(2).and_then(|v| v.checked_add(1));
    if expected != Some(xi.len())
        || n > 1 << 26
        || xi.iter().any(|v| !v.is_finite())
        || !xi.iter().eq(xi.iter().rev())
        || !length.is_finite()
        || length <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || maximum_iterations == 0
    {
        return Err(anyhow!("CCM point discovery requires a finite even state with at most 2^26 modes, positive length/tolerance and positive iteration budget"));
    }
    // The CCM quotient requires nonzero eta pairing. Its common state scale
    // cancels exactly in the mathematical roots. This exact binary scaling
    // also rejects zero pairings and unrepresentable coefficient loss.
    let state = super::rank_one::arithmetic::exact_scaled_state_f64(xi)?;
    let positive = &state[n..];
    let active = (1..=n).filter(|&j| positive[j] != 0.0).collect::<Vec<_>>();
    if active.len() > 4096 {
        return Err(anyhow!(
            "native exact-sign discovery supports at most 4096 active poles"
        ));
    }
    if active.is_empty() {
        // R(t) = ξ_0, the nonzero source pairing: no roots.
        return Ok(Vec::new());
    }
    let rounding = 4.0 * (n as f64 + 4.0) * f64::EPSILON;
    let poles = active
        .iter()
        .map(|&j| (j as f64).powi(2))
        .collect::<Vec<_>>();
    let terms = std::iter::once(positive[0]).chain(active.iter().map(|&j| 2.0 * positive[j]));
    use num_rational::BigRational as Q;
    use num_traits::ToPrimitive;
    let limit_q: Q = std::iter::once(Q::from_float(positive[0]).unwrap())
        .chain(
            active
                .iter()
                .map(|&j| Q::from_float(positive[j]).unwrap() * Q::from_integer(2.into())),
        )
        .sum();
    let limit = limit_q
        .to_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| anyhow!("CCM source pairing exceeds binary64 range"))?;
    let full = PartialFractions {
        constant: limit,
        constant_error: 2.0 * f64::EPSILON * limit.abs()
            + f64::from_bits(1)
            + rounding * f64::EPSILON * terms.map(f64::abs).sum::<f64>(),
        residues: active
            .iter()
            .map(|&j| 2.0 * positive[j] * (j as f64).powi(2))
            .collect(),
        poles: poles.clone(),
        rounding,
        exact_constant: limit_q,
        exact_residues: active
            .iter()
            .map(|&j| Q::from_float(positive[j]).unwrap() * Q::from_integer((2 * j * j).into()))
            .collect(),
    };
    let reduced = PartialFractions {
        constant: 0.0,
        constant_error: 0.0,
        residues: active.iter().map(|&j| 2.0 * positive[j]).collect(),
        poles,
        rounding,
        exact_constant: Q::from_integer(0.into()),
        exact_residues: active
            .iter()
            .map(|&j| Q::from_float(positive[j]).unwrap() * Q::from_integer(2.into()))
            .collect(),
    };
    let last_pole = *full.poles.last().expect("active poles are nonempty");
    let exterior = full
        .exterior_bound(last_pole)
        .ok_or_else(|| anyhow!("CCM secular exterior window cannot be bounded in binary64"))?;
    let stopping = RootStoppingF64 {
        absolute_x_tolerance: tolerance,
        relative_x_tolerance: tolerance,
        // No scale-dependent nonzero residual shortcut: permit only zero
        // or the smallest representable point residual, otherwise use width.
        residual_tolerance: f64::from_bits(1),
        maximum_iterations,
    };
    let mut budget = maximum_iterations
        .saturating_mul(64)
        .saturating_mul(n.saturating_add(1));
    let mut roots = Vec::with_capacity(n);
    let count = full.poles.len();
    for index in 0..=count {
        let window = Piece {
            lower: if index == 0 {
                0.0
            } else {
                full.poles[index - 1]
            },
            upper: if index == count {
                exterior
            } else {
                full.poles[index]
            },
            left: if index == 0 {
                End::Regular
            } else {
                End::Pole(index - 1)
            },
            right: if index == count {
                End::Regular
            } else {
                End::Pole(index)
            },
        };
        let reduce = index == 0 && positive[0] == 0.0;
        let function = EvenSecular {
            positive,
            reduced: reduce,
        };
        let classes = if reduce { &reduced } else { &full };
        search_window(
            &function,
            classes,
            window,
            &stopping,
            &mut budget,
            &mut roots,
        )?;
    }
    // The numerator after clearing active poles has degree at most their count.
    if roots.len() > active.len() {
        return Err(anyhow!(
            "native secular roots exceed the exact numerator degree"
        ));
    }
    let mut ordinates = Vec::with_capacity(roots.len());
    for root in roots {
        let ordinate = super::rank_one::arithmetic::ordinate_f64(root.sqrt(), length)?;
        if !ordinate.is_finite() || ordinate <= 0.0 {
            return Err(anyhow!(
                "CCM discovered ordinate must be finite and positive"
            ));
        }
        if ordinates
            .last()
            .is_some_and(|previous| *previous >= ordinate)
        {
            return Err(anyhow!(
                "CCM point discovery cannot order two roots in binary64"
            ));
        }
        ordinates.push(ordinate);
    }
    Ok(ordinates)
}

#[cfg(test)]
mod native_signs_tests {
    use super::*;
    use num_rational::BigRational as Q;
    #[test]
    fn cancellation_fixture_is_checked_against_original_dyadic_roots() {
        let xi = [
            0x4047000000000000,
            0xbff9d2aa52ef6999,
            0x4027fcbf5942a7f7,
            0xbff9d2aa52ef6999,
            0x4047000000000000,
        ]
        .map(f64::from_bits);
        let original = &xi;
        let [c, b, a] = [original[2], original[3], original[4]].map(|x| Q::from_float(x).unwrap());
        let two = Q::from_integer(2.into());
        let four = Q::from_integer(4.into());
        let aa = &c + &two * &b + &two * &a;
        let bb = -Q::from_integer(5.into()) * &c - Q::from_integer(8.into()) * &b - &two * &a;
        let cc = &four * &c;
        assert!(
            &bb * &bb - &four * &aa * &cc > Q::from_integer(0.into()),
            "exact original numerator has two real roots"
        );
        for (tol, steps) in [(1e-12, 200), (1e-13, 400)] {
            match solve(&xi, 2, 2.0 * std::f64::consts::PI, tol, steps) {
                Ok(roots) => {
                    assert_eq!(roots.len(), 2);
                    for (root, exact_t) in
                        roots.iter().zip([0.6899999957237214, 0.6900000042762785])
                    {
                        assert!((root * root - exact_t).abs() < 2e-12);
                    }
                }
                Err(error) => assert!(
                    error.to_string().contains("separate")
                        || error.to_string().contains("unresolved")
                ),
            }
        }
    }
    #[test]
    fn well_separated_same_sign_source_retains_its_two_roots() {
        let roots = solve(&[1.0; 5], 2, 2.0 * std::f64::consts::PI, 1e-12, 200).unwrap();
        assert_eq!(roots.len(), 2);
        assert!(roots[0] > 0.0 && roots[0] < 1.0);
        assert!(roots[1] > 1.0 && roots[1] < 2.0);
    }
}

#[cfg(test)]
mod original_dyadic_sign_range_controls {
    use super::*;
    #[test]
    fn exact_nonzero_sign_is_never_a_residual_shortcut_or_overflow() {
        let tiny = [0.0, f64::from_bits(1)];
        assert_eq!(
            EvenSecular {
                positive: &tiny,
                reduced: true
            }
            .evaluate(1e308)
            .unwrap(),
            1.0
        );
        let huge = [f64::MAX, f64::MAX];
        assert_eq!(
            EvenSecular {
                positive: &huge,
                reduced: false
            }
            .evaluate(2.0)
            .unwrap(),
            1.0
        );
        let zero = [0.0, 0.0];
        assert_eq!(
            EvenSecular {
                positive: &zero,
                reduced: false
            }
            .evaluate(0.5)
            .unwrap(),
            0.0
        );
    }
    #[test]
    fn deliberately_wrong_floating_exclusion_cannot_drop_an_exact_root() {
        use num_rational::BigRational as Q;
        let classes = PartialFractions {
            constant: 1.0,
            constant_error: 0.0,
            poles: vec![1.0],
            residues: vec![0.0],
            rounding: 0.0,
            exact_constant: Q::from_integer(1.into()),
            exact_residues: vec![Q::from_integer((-1).into())],
        };
        let piece = Piece {
            lower: 1.75,
            upper: 2.25,
            left: End::Regular,
            right: End::Regular,
        };
        assert!(matches!(
            classes.classify_interior(&piece),
            Verdict::Unknown
        ));
        // Exact R(t)=1-1/(t-1) vanishes at t=2, including the exterior.
        assert!(!classes.exact_exterior_exclusion(1.75));
        assert!(classes.exact_exterior_exclusion(4.0));
        let pole_piece = Piece {
            lower: 1.0,
            upper: 2.25,
            left: End::Pole(0),
            right: End::Regular,
        };
        assert!(!classes.excludes_pole_piece(&pole_piece));
    }

    #[test]
    fn rational_hint_workspace_exhaustion_refuses_admission() {
        use num_rational::BigRational as Q;
        let value = Q::from_integer(2.into()).pow(512);
        assert!(exact_workspace_available(&[&value, &value]));
        let values = vec![&value; 16_000];
        assert!(!exact_workspace_available(&values));
    }
}
