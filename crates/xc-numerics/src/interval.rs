//! Exact-rational interval and rectangular complex-ball arithmetic.
//!
//! This backend is deliberately conservative and validation-oriented.  Every
//! endpoint is an arbitrary-precision rational, so its enclosures are exact;
//! operations return a wider interval rather than make a floating-point sign
//! decision.  Faster MPFR/Arb backends may implement the same contracts.

use rug::{Integer, Rational};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntervalError {
    Invalid(String),
    DivisionByZeroInterval,
    Inconclusive(String),
}

impl Display for IntervalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid interval problem: {message}"),
            Self::DivisionByZeroInterval => {
                f.write_str("interval division denominator contains zero")
            }
            Self::Inconclusive(message) => write!(f, "interval proof is inconclusive: {message}"),
        }
    }
}

impl Error for IntervalError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalInterval {
    lower: Rational,
    upper: Rational,
}

impl RationalInterval {
    pub fn new(lower: Rational, upper: Rational) -> Result<Self, IntervalError> {
        if lower > upper {
            return Err(IntervalError::Invalid(
                "lower endpoint exceeds upper endpoint".to_owned(),
            ));
        }
        Ok(Self { lower, upper })
    }

    pub fn point(value: Rational) -> Self {
        Self {
            lower: value.clone(),
            upper: value,
        }
    }

    pub fn hull(left: Rational, right: Rational) -> Self {
        if left <= right {
            Self {
                lower: left,
                upper: right,
            }
        } else {
            Self {
                lower: right,
                upper: left,
            }
        }
    }

    pub fn lower(&self) -> &Rational {
        &self.lower
    }

    pub fn upper(&self) -> &Rational {
        &self.upper
    }

    pub fn is_point(&self) -> bool {
        self.lower == self.upper
    }

    pub fn contains(&self, value: &Rational) -> bool {
        self.lower <= *value && *value <= self.upper
    }

    pub fn contains_zero(&self) -> bool {
        self.lower <= 0 && self.upper >= 0
    }

    pub fn is_strictly_positive(&self) -> bool {
        self.lower > 0
    }

    pub fn is_strictly_negative(&self) -> bool {
        self.upper < 0
    }

    pub fn width(&self) -> Rational {
        let mut width = self.upper.clone();
        width -= &self.lower;
        width
    }

    pub fn midpoint(&self) -> Rational {
        let mut midpoint = self.lower.clone();
        midpoint += &self.upper;
        midpoint /= 2;
        midpoint
    }

    /// Enclose the non-negative square root on a dyadic grid.
    ///
    /// The returned endpoints are exact rationals with denominator
    /// `2^fraction_bits`.  Integer square roots are taken only after outward
    /// rounding the scaled rational endpoints, so this operation does not
    /// depend on a floating-point square-root implementation.
    pub fn sqrt_nonnegative(&self, fraction_bits: u32) -> Result<Self, IntervalError> {
        if self.lower < 0 {
            return Err(IntervalError::Invalid(
                "square-root interval has a negative lower endpoint".to_owned(),
            ));
        }

        fn scaled_floor(value: &Rational, bits: u32) -> Integer {
            let mut numerator = value.numer().clone();
            numerator <<= 2 * bits;
            numerator / value.denom()
        }

        fn scaled_ceil(value: &Rational, bits: u32) -> Integer {
            let mut numerator = value.numer().clone();
            numerator <<= 2 * bits;
            let denominator = value.denom();
            let mut quotient = numerator.clone() / denominator;
            let mut reconstructed = quotient.clone();
            reconstructed *= denominator;
            if reconstructed < numerator {
                quotient += 1;
            }
            quotient
        }

        let lower_scaled = scaled_floor(&self.lower, fraction_bits);
        let upper_scaled = scaled_ceil(&self.upper, fraction_bits);
        let lower_root = lower_scaled.sqrt();
        let mut upper_root = upper_scaled.clone().sqrt();
        let mut upper_square = upper_root.clone();
        upper_square *= &upper_root;
        if upper_square < upper_scaled {
            upper_root += 1;
        }
        let denominator = Integer::from(1) << fraction_bits;
        Self::new(
            Rational::from((lower_root, denominator.clone())),
            Rational::from((upper_root, denominator)),
        )
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            lower: rational_add(&self.lower, &other.lower),
            upper: rational_add(&self.upper, &other.upper),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            lower: rational_sub(&self.lower, &other.upper),
            upper: rational_sub(&self.upper, &other.lower),
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            lower: -self.upper.clone(),
            upper: -self.lower.clone(),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [
            rational_mul(&self.lower, &other.lower),
            rational_mul(&self.lower, &other.upper),
            rational_mul(&self.upper, &other.lower),
            rational_mul(&self.upper, &other.upper),
        ];
        let lower = products.iter().min().expect("four products").clone();
        let upper = products.iter().max().expect("four products").clone();
        Self { lower, upper }
    }

    pub fn square(&self) -> Self {
        if self.contains_zero() {
            let lower = Rational::from((0, 1));
            let upper = rational_mul(
                &rational_abs(&self.lower).max(rational_abs(&self.upper)),
                &rational_abs(&self.lower).max(rational_abs(&self.upper)),
            );
            Self { lower, upper }
        } else {
            self.mul(self)
        }
    }

    pub fn reciprocal(&self) -> Result<Self, IntervalError> {
        if self.contains_zero() {
            return Err(IntervalError::DivisionByZeroInterval);
        }
        let left = rational_div(&Rational::from((1, 1)), &self.lower);
        let right = rational_div(&Rational::from((1, 1)), &self.upper);
        Ok(Self::hull(left, right))
    }

    pub fn div(&self, other: &Self) -> Result<Self, IntervalError> {
        Ok(self.mul(&other.reciprocal()?))
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let lower = self.lower.clone().max(other.lower.clone());
        let upper = self.upper.clone().min(other.upper.clone());
        (lower <= upper).then_some(Self { lower, upper })
    }

    pub fn is_strictly_inside(&self, other: &Self) -> bool {
        other.lower < self.lower && self.upper < other.upper
    }
}

fn rational_add(left: &Rational, right: &Rational) -> Rational {
    let mut value = left.clone();
    value += right;
    value
}

fn rational_sub(left: &Rational, right: &Rational) -> Rational {
    let mut value = left.clone();
    value -= right;
    value
}

fn rational_mul(left: &Rational, right: &Rational) -> Rational {
    let mut value = left.clone();
    value *= right;
    value
}

fn rational_div(left: &Rational, right: &Rational) -> Rational {
    let mut value = left.clone();
    value /= right;
    value
}

fn rational_abs(value: &Rational) -> Rational {
    if value < &0 {
        -value.clone()
    } else {
        value.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComplexRational {
    pub real: Rational,
    pub imaginary: Rational,
}

impl ComplexRational {
    pub fn zero() -> Self {
        Self {
            real: Rational::from((0, 1)),
            imaginary: Rational::from((0, 1)),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            real: rational_add(&self.real, &other.real),
            imaginary: rational_add(&self.imaginary, &other.imaginary),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            real: rational_sub(
                &rational_mul(&self.real, &other.real),
                &rational_mul(&self.imaginary, &other.imaginary),
            ),
            imaginary: rational_add(
                &rational_mul(&self.real, &other.imaginary),
                &rational_mul(&self.imaginary, &other.real),
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComplexRationalBall {
    pub real: RationalInterval,
    pub imaginary: RationalInterval,
}

impl ComplexRationalBall {
    pub fn point(value: ComplexRational) -> Self {
        Self {
            real: RationalInterval::point(value.real),
            imaginary: RationalInterval::point(value.imaginary),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            real: self.real.add(&other.real),
            imaginary: self.imaginary.add(&other.imaginary),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let ac = self.real.mul(&other.real);
        let bd = self.imaginary.mul(&other.imaginary);
        let ad = self.real.mul(&other.imaginary);
        let bc = self.imaginary.mul(&other.real);
        Self {
            real: ac.sub(&bd),
            imaginary: ad.add(&bc),
        }
    }

    pub fn modulus_squared(&self) -> RationalInterval {
        self.real.square().add(&self.imaginary.square())
    }

    pub fn excludes_zero(&self) -> bool {
        !self.real.contains_zero() || !self.imaginary.contains_zero()
    }
}

pub fn evaluate_real_polynomial_interval(
    coefficients_ascending: &[Rational],
    argument: &RationalInterval,
) -> Result<RationalInterval, IntervalError> {
    let Some(highest) = coefficients_ascending.last() else {
        return Err(IntervalError::Invalid(
            "polynomial must contain at least one coefficient".to_owned(),
        ));
    };
    let mut value = RationalInterval::point(highest.clone());
    for coefficient in coefficients_ascending.iter().rev().skip(1) {
        value = value
            .mul(argument)
            .add(&RationalInterval::point(coefficient.clone()));
    }
    Ok(value)
}

pub fn evaluate_real_polynomial_exact(
    coefficients_ascending: &[Rational],
    argument: &Rational,
) -> Result<Rational, IntervalError> {
    let Some(highest) = coefficients_ascending.last() else {
        return Err(IntervalError::Invalid(
            "polynomial must contain at least one coefficient".to_owned(),
        ));
    };
    let mut value = highest.clone();
    for coefficient in coefficients_ascending.iter().rev().skip(1) {
        value *= argument;
        value += coefficient;
    }
    Ok(value)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactSturmCount {
    pub lower: Rational,
    pub upper: Rational,
    pub distinct_real_roots: usize,
    pub variations_at_lower: usize,
    pub variations_at_upper: usize,
    pub square_free: bool,
    pub sequence_length: usize,
}

fn trim_polynomial(mut polynomial: Vec<Rational>) -> Vec<Rational> {
    while polynomial.len() > 1 && polynomial.last().is_some_and(|value| value == &0) {
        polynomial.pop();
    }
    polynomial
}

/// Scale a nonzero polynomial by a strictly positive rational so its leading
/// coefficient has absolute value one. Positive scaling preserves every sign
/// used by Sturm variation while preventing avoidable numerator/denominator
/// growth in the rational Euclidean sequence.
fn normalize_polynomial_positive(polynomial: Vec<Rational>) -> Vec<Rational> {
    let polynomial = trim_polynomial(polynomial);
    let leading = polynomial.last().expect("trimmed polynomial is nonempty");
    if leading == &0 {
        return polynomial;
    }
    let scale = if leading < &0 {
        -leading.clone()
    } else {
        leading.clone()
    };
    polynomial
        .into_iter()
        .map(|mut coefficient| {
            coefficient /= &scale;
            coefficient
        })
        .collect()
}

fn polynomial_derivative(polynomial: &[Rational]) -> Vec<Rational> {
    if polynomial.len() <= 1 {
        return vec![Rational::from((0, 1))];
    }
    polynomial
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| {
            let mut value = coefficient.clone();
            value *= power;
            value
        })
        .collect()
}

fn polynomial_remainder(
    dividend: &[Rational],
    divisor: &[Rational],
) -> Result<Vec<Rational>, IntervalError> {
    let divisor = trim_polynomial(divisor.to_vec());
    if divisor.len() == 1 && divisor[0] == 0 {
        return Err(IntervalError::Invalid(
            "polynomial division by zero polynomial".to_owned(),
        ));
    }
    let mut remainder = trim_polynomial(dividend.to_vec());
    while !(remainder.len() == 1 && remainder[0] == 0) && remainder.len() >= divisor.len() {
        let shift = remainder.len() - divisor.len();
        let mut factor = remainder.last().expect("nonempty polynomial").clone();
        factor /= divisor.last().expect("nonempty divisor");
        for (index, coefficient) in divisor.iter().enumerate() {
            let mut term = coefficient.clone();
            term *= &factor;
            remainder[shift + index] -= term;
        }
        // The current dividend may be rescaled by any positive rational
        // without changing the final remainder except by that same positive
        // factor. Normalize after every eliminated degree so large exact
        // inputs do not accumulate enormous intermediate fractions before
        // the outer Sturm-sequence normalization gets a chance to run.
        remainder = normalize_polynomial_positive(remainder);
    }
    Ok(remainder)
}

fn sturm_variations(sequence: &[Vec<Rational>], point: &Rational) -> Result<usize, IntervalError> {
    let mut previous_sign = None;
    let mut variations = 0usize;
    for polynomial in sequence {
        let value = evaluate_real_polynomial_exact(polynomial, point)?;
        if value == 0 {
            continue;
        }
        let sign = value > 0;
        if previous_sign.is_some_and(|previous| previous != sign) {
            variations += 1;
        }
        previous_sign = Some(sign);
    }
    Ok(variations)
}

/// Count distinct real roots in an open rational interval using an exact
/// Sturm sequence. Endpoint roots are rejected so the interval semantics are
/// unambiguous. `square_free=false` reports unresolved multiplicity rather
/// than silently treating a distinct-root count as a multiplicity count.
pub fn exact_sturm_root_count(
    coefficients_ascending: &[Rational],
    lower: Rational,
    upper: Rational,
) -> Result<ExactSturmCount, IntervalError> {
    if coefficients_ascending.len() < 2 || lower >= upper {
        return Err(IntervalError::Invalid(
            "Sturm counting requires a nonconstant polynomial and lower < upper".to_owned(),
        ));
    }
    let polynomial = normalize_polynomial_positive(coefficients_ascending.to_vec());
    if polynomial.len() < 2 {
        return Err(IntervalError::Invalid(
            "Sturm counting polynomial becomes constant after trimming".to_owned(),
        ));
    }
    if evaluate_real_polynomial_exact(&polynomial, &lower)? == 0
        || evaluate_real_polynomial_exact(&polynomial, &upper)? == 0
    {
        return Err(IntervalError::Inconclusive(
            "Sturm counting interval has a root on its boundary".to_owned(),
        ));
    }
    let mut sequence = vec![
        polynomial.clone(),
        normalize_polynomial_positive(polynomial_derivative(&polynomial)),
    ];
    loop {
        let remainder =
            polynomial_remainder(&sequence[sequence.len() - 2], &sequence[sequence.len() - 1])?;
        if remainder.len() == 1 && remainder[0] == 0 {
            break;
        }
        sequence.push(normalize_polynomial_positive(
            remainder.into_iter().map(|value| -value).collect(),
        ));
    }
    let variations_at_lower = sturm_variations(&sequence, &lower)?;
    let variations_at_upper = sturm_variations(&sequence, &upper)?;
    if variations_at_lower < variations_at_upper {
        return Err(IntervalError::Invalid(
            "Sturm variation count increased across an ordered interval".to_owned(),
        ));
    }
    let square_free = sequence.last().is_some_and(|last| last.len() == 1);
    Ok(ExactSturmCount {
        lower,
        upper,
        distinct_real_roots: variations_at_lower - variations_at_upper,
        variations_at_lower,
        variations_at_upper,
        square_free,
        sequence_length: sequence.len(),
    })
}

/// Isolate every distinct real root in an open rational interval by exact
/// Sturm subdivision. The returned intervals are ordered, disjoint, contain
/// one distinct root each, and have width at most `target_width`.
pub fn exact_sturm_isolate_roots(
    coefficients_ascending: &[Rational],
    lower: Rational,
    upper: Rational,
    target_width: Rational,
    maximum_depth: usize,
) -> Result<Vec<RationalInterval>, IntervalError> {
    if target_width <= 0 || maximum_depth == 0 {
        return Err(IntervalError::Invalid(
            "Sturm isolation requires positive target width and subdivision depth".to_owned(),
        ));
    }
    let initial = exact_sturm_root_count(coefficients_ascending, lower.clone(), upper.clone())?;
    if !initial.square_free {
        return Err(IntervalError::Inconclusive(
            "Sturm isolation polynomial has unresolved repeated roots".to_owned(),
        ));
    }
    let mut pending = vec![(lower, upper, initial.distinct_real_roots, 0usize)];
    let mut isolated = Vec::with_capacity(initial.distinct_real_roots);
    while let Some((left, right, count, depth)) = pending.pop() {
        if count == 0 {
            continue;
        }
        let mut width = right.clone();
        width -= &left;
        if count == 1 && width <= target_width {
            isolated.push(RationalInterval::new(left, right)?);
            continue;
        }
        if depth >= maximum_depth {
            return Err(IntervalError::Inconclusive(format!(
                "Sturm isolation reached depth {maximum_depth} with {count} unresolved roots"
            )));
        }
        let mut split = None;
        for denominator in 2..=19 {
            let mut offset = width.clone();
            offset /= denominator;
            let mut candidate = left.clone();
            candidate += offset;
            if evaluate_real_polynomial_exact(coefficients_ascending, &candidate)? != 0 {
                split = Some(candidate);
                break;
            }
        }
        let split = split.ok_or_else(|| {
            IntervalError::Inconclusive(
                "could not choose a non-root rational Sturm subdivision point".to_owned(),
            )
        })?;
        let left_count =
            exact_sturm_root_count(coefficients_ascending, left.clone(), split.clone())?
                .distinct_real_roots;
        let right_count =
            exact_sturm_root_count(coefficients_ascending, split.clone(), right.clone())?
                .distinct_real_roots;
        if left_count + right_count != count {
            return Err(IntervalError::Invalid(
                "Sturm subdivision count did not conserve the parent count".to_owned(),
            ));
        }
        pending.push((split.clone(), right, right_count, depth + 1));
        pending.push((left, split, left_count, depth + 1));
    }
    isolated.sort_by(|left, right| left.lower().cmp(right.lower()));
    Ok(isolated)
}

pub fn evaluate_complex_polynomial_ball(
    coefficients_ascending: &[ComplexRational],
    argument: &ComplexRationalBall,
) -> Result<ComplexRationalBall, IntervalError> {
    let Some(highest) = coefficients_ascending.last() else {
        return Err(IntervalError::Invalid(
            "complex polynomial must contain at least one coefficient".to_owned(),
        ));
    };
    let mut value = ComplexRationalBall::point(highest.clone());
    for coefficient in coefficients_ascending.iter().rev().skip(1) {
        value = value
            .mul(argument)
            .add(&ComplexRationalBall::point(coefficient.clone()));
    }
    Ok(value)
}

pub fn evaluate_complex_polynomial_exact(
    coefficients_ascending: &[ComplexRational],
    argument: &ComplexRational,
) -> Result<ComplexRational, IntervalError> {
    let Some(highest) = coefficients_ascending.last() else {
        return Err(IntervalError::Invalid(
            "complex polynomial must contain at least one coefficient".to_owned(),
        ));
    };
    let mut value = highest.clone();
    for coefficient in coefficients_ascending.iter().rev().skip(1) {
        value = value.mul(argument).add(coefficient);
    }
    Ok(value)
}

pub fn differentiate_complex_polynomial(
    coefficients_ascending: &[ComplexRational],
) -> Result<Vec<ComplexRational>, IntervalError> {
    if coefficients_ascending.is_empty() {
        return Err(IntervalError::Invalid(
            "complex polynomial must contain at least one coefficient".to_owned(),
        ));
    }
    if coefficients_ascending.len() == 1 {
        return Ok(vec![ComplexRational::zero()]);
    }
    Ok(coefficients_ascending
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| {
            let factor = Rational::from(power);
            ComplexRational {
                real: rational_mul(&coefficient.real, &factor),
                imaginary: rational_mul(&coefficient.imaginary, &factor),
            }
        })
        .collect())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalContourRectangle {
    pub real_lower: Rational,
    pub real_upper: Rational,
    pub imaginary_lower: Rational,
    pub imaginary_upper: Rational,
}

impl RationalContourRectangle {
    pub fn new(
        real_lower: Rational,
        real_upper: Rational,
        imaginary_lower: Rational,
        imaginary_upper: Rational,
    ) -> Result<Self, IntervalError> {
        if real_lower >= real_upper || imaginary_lower >= imaginary_upper {
            return Err(IntervalError::Invalid(
                "contour rectangle must have strictly ordered real and imaginary bounds".to_owned(),
            ));
        }
        Ok(Self {
            real_lower,
            real_upper,
            imaginary_lower,
            imaginary_upper,
        })
    }

    fn counterclockwise_vertices(&self) -> Vec<ComplexRational> {
        vec![
            ComplexRational {
                real: self.real_lower.clone(),
                imaginary: self.imaginary_lower.clone(),
            },
            ComplexRational {
                real: self.real_upper.clone(),
                imaginary: self.imaginary_lower.clone(),
            },
            ComplexRational {
                real: self.real_upper.clone(),
                imaginary: self.imaginary_upper.clone(),
            },
            ComplexRational {
                real: self.real_lower.clone(),
                imaginary: self.imaginary_upper.clone(),
            },
            ComplexRational {
                real: self.real_lower.clone(),
                imaginary: self.imaginary_lower.clone(),
            },
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialContourCell {
    pub domain_start: ComplexRational,
    pub domain_end: ComplexRational,
    pub image_start: ComplexRational,
    pub image_end: ComplexRational,
    pub image_enclosure: ComplexRationalBall,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialContourCount {
    pub rectangle: RationalContourRectangle,
    pub zero_count: usize,
    pub winding_number: i64,
    pub accepted_contour_cells: usize,
    pub maximum_subdivision_depth: usize,
    pub boundary_excludes_zero: bool,
    pub rigorous: bool,
    pub cells: Vec<PolynomialContourCell>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalFunctionContourCount {
    pub numerator: PolynomialContourCount,
    pub denominator: PolynomialContourCount,
    pub zeros_minus_poles: i64,
    pub rigorous: bool,
}

fn complex_segment_ball(start: &ComplexRational, end: &ComplexRational) -> ComplexRationalBall {
    ComplexRationalBall {
        real: RationalInterval::hull(start.real.clone(), end.real.clone()),
        imaginary: RationalInterval::hull(start.imaginary.clone(), end.imaginary.clone()),
    }
}

fn complex_midpoint(start: &ComplexRational, end: &ComplexRational) -> ComplexRational {
    let mut real = start.real.clone();
    real += &end.real;
    real /= 2;
    let mut imaginary = start.imaginary.clone();
    imaginary += &end.imaginary;
    imaginary /= 2;
    ComplexRational { real, imaginary }
}

fn enclose_polynomial_contour_segment(
    coefficients_ascending: &[ComplexRational],
    start: &ComplexRational,
    end: &ComplexRational,
    depth: usize,
    maximum_depth: usize,
    cells: &mut Vec<PolynomialContourCell>,
    vertices: &mut Vec<ComplexRational>,
) -> Result<(), IntervalError> {
    let enclosure = evaluate_complex_polynomial_ball(
        coefficients_ascending,
        &complex_segment_ball(start, end),
    )?;
    if enclosure.excludes_zero() {
        cells.push(PolynomialContourCell {
            domain_start: start.clone(),
            domain_end: end.clone(),
            image_start: evaluate_complex_polynomial_exact(coefficients_ascending, start)?,
            image_end: evaluate_complex_polynomial_exact(coefficients_ascending, end)?,
            image_enclosure: enclosure,
        });
        vertices.push(end.clone());
        return Ok(());
    }
    if depth >= maximum_depth {
        return Err(IntervalError::Inconclusive(format!(
            "polynomial contour image still contains zero after {maximum_depth} subdivisions"
        )));
    }
    let midpoint = complex_midpoint(start, end);
    enclose_polynomial_contour_segment(
        coefficients_ascending,
        start,
        &midpoint,
        depth + 1,
        maximum_depth,
        cells,
        vertices,
    )?;
    enclose_polynomial_contour_segment(
        coefficients_ascending,
        &midpoint,
        end,
        depth + 1,
        maximum_depth,
        cells,
        vertices,
    )
}

fn exact_polygon_winding_about_zero(vertices: &[ComplexRational]) -> Result<i64, IntervalError> {
    if vertices.len() < 2 || vertices.first() != vertices.last() {
        return Err(IntervalError::Invalid(
            "winding polygon must be nonempty and closed".to_owned(),
        ));
    }
    let mut winding = 0i64;
    for edge in vertices.windows(2) {
        let start = &edge[0];
        let end = &edge[1];
        if (start.real == 0 && start.imaginary == 0) || (end.real == 0 && end.imaginary == 0) {
            return Err(IntervalError::Inconclusive(
                "polynomial has a zero at a contour subdivision vertex".to_owned(),
            ));
        }
        let cross = rational_sub(
            &rational_mul(&start.real, &end.imaginary),
            &rational_mul(&end.real, &start.imaginary),
        );
        if start.imaginary <= 0 && end.imaginary > 0 && cross > 0 {
            winding += 1;
        } else if start.imaginary > 0 && end.imaginary <= 0 && cross < 0 {
            winding -= 1;
        }
    }
    Ok(winding)
}

/// Count all polynomial zeros inside a counterclockwise rational rectangle.
///
/// Each adaptively subdivided contour segment is evaluated with exact
/// rectangular complex-ball arithmetic. Once its image box excludes zero,
/// convexity gives a zero-avoiding homotopy to the chord joining its exact
/// endpoint images. The winding of the resulting rational polygon is then
/// computed using exact ray crossings. Boundary zeros therefore yield
/// `Inconclusive`, never an invented count.
pub fn certify_polynomial_zero_count_on_rectangle(
    coefficients_ascending: &[ComplexRational],
    rectangle: RationalContourRectangle,
    maximum_subdivision_depth: usize,
) -> Result<PolynomialContourCount, IntervalError> {
    if coefficients_ascending.is_empty()
        || coefficients_ascending
            .iter()
            .all(|coefficient| coefficient.real == 0 && coefficient.imaginary == 0)
        || maximum_subdivision_depth == 0
    {
        return Err(IntervalError::Invalid(
            "contour count requires a nonzero polynomial and positive subdivision depth".to_owned(),
        ));
    }
    let contour = rectangle.counterclockwise_vertices();
    let mut cells = Vec::new();
    let mut argument_vertices = vec![evaluate_complex_polynomial_exact(
        coefficients_ascending,
        &contour[0],
    )?];
    for edge in contour.windows(2) {
        let mut domain_vertices = Vec::new();
        enclose_polynomial_contour_segment(
            coefficients_ascending,
            &edge[0],
            &edge[1],
            0,
            maximum_subdivision_depth,
            &mut cells,
            &mut domain_vertices,
        )?;
        for vertex in domain_vertices {
            argument_vertices.push(evaluate_complex_polynomial_exact(
                coefficients_ascending,
                &vertex,
            )?);
        }
    }
    let winding_number = exact_polygon_winding_about_zero(&argument_vertices)?;
    let zero_count = usize::try_from(winding_number).map_err(|_| {
        IntervalError::Invalid(
            "counterclockwise polynomial contour produced a negative winding".to_owned(),
        )
    })?;
    Ok(PolynomialContourCount {
        rectangle,
        zero_count,
        winding_number,
        accepted_contour_cells: cells.len(),
        maximum_subdivision_depth,
        boundary_excludes_zero: true,
        rigorous: true,
        cells,
    })
}

/// Apply the argument principle to an exact rational function. The result is
/// the number of numerator zeros minus denominator zeros (poles), counted
/// with multiplicity. Both contour images are certified independently.
pub fn certify_rational_function_argument_count_on_rectangle(
    numerator_ascending: &[ComplexRational],
    denominator_ascending: &[ComplexRational],
    rectangle: RationalContourRectangle,
    maximum_subdivision_depth: usize,
) -> Result<RationalFunctionContourCount, IntervalError> {
    let numerator = certify_polynomial_zero_count_on_rectangle(
        numerator_ascending,
        rectangle.clone(),
        maximum_subdivision_depth,
    )?;
    let denominator = certify_polynomial_zero_count_on_rectangle(
        denominator_ascending,
        rectangle,
        maximum_subdivision_depth,
    )?;
    Ok(RationalFunctionContourCount {
        zeros_minus_poles: numerator.winding_number - denominator.winding_number,
        numerator,
        denominator,
        rigorous: true,
    })
}

pub fn interval_dot(
    left: &[RationalInterval],
    right: &[RationalInterval],
) -> Result<RationalInterval, IntervalError> {
    if left.len() != right.len() {
        return Err(IntervalError::Invalid(
            "interval dot-product dimensions differ".to_owned(),
        ));
    }
    Ok(left.iter().zip(right).fold(
        RationalInterval::point(Rational::from((0, 1))),
        |sum, (a, b)| sum.add(&a.mul(b)),
    ))
}

pub fn interval_matvec(
    matrix_row_major: &[RationalInterval],
    dimension: usize,
    vector: &[RationalInterval],
) -> Result<Vec<RationalInterval>, IntervalError> {
    if dimension == 0
        || matrix_row_major.len() != dimension.saturating_mul(dimension)
        || vector.len() != dimension
    {
        return Err(IntervalError::Invalid(
            "interval matrix-vector dimensions are inconsistent".to_owned(),
        ));
    }
    (0..dimension)
        .map(|row| {
            interval_dot(
                &matrix_row_major[row * dimension..(row + 1) * dimension],
                vector,
            )
        })
        .collect()
}

pub fn interval_quadratic_form(
    matrix_row_major: &[RationalInterval],
    dimension: usize,
    vector: &[RationalInterval],
) -> Result<RationalInterval, IntervalError> {
    interval_dot(
        vector,
        &interval_matvec(matrix_row_major, dimension, vector)?,
    )
}

pub fn interval_rayleigh_quotient(
    numerator_matrix: &[RationalInterval],
    denominator_matrix: &[RationalInterval],
    dimension: usize,
    vector: &[RationalInterval],
) -> Result<RationalInterval, IntervalError> {
    let numerator = interval_quadratic_form(numerator_matrix, dimension, vector)?;
    let denominator = interval_quadratic_form(denominator_matrix, dimension, vector)?;
    if !denominator.is_strictly_positive() {
        return Err(IntervalError::Inconclusive(
            "Rayleigh denominator is not proven strictly positive".to_owned(),
        ));
    }
    numerator.div(&denominator)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialRootIsolation {
    pub interval: RationalInterval,
    pub derivative_enclosure: RationalInterval,
    pub value_at_lower: Rational,
    pub value_at_upper: Rational,
    pub unique: bool,
}

/// Certify one simple real polynomial root by exact endpoint signs and an
/// interval derivative that excludes zero.  Failure to prove either property
/// is reported as inconclusive, never as an invented count.
pub fn certify_simple_polynomial_root(
    coefficients_ascending: &[Rational],
    interval: RationalInterval,
) -> Result<PolynomialRootIsolation, IntervalError> {
    if coefficients_ascending.len() < 2 || interval.is_point() {
        return Err(IntervalError::Invalid(
            "root isolation requires a nonconstant polynomial and nonzero-width interval"
                .to_owned(),
        ));
    }
    let derivative = coefficients_ascending
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| {
            let mut value = coefficient.clone();
            value *= power;
            value
        })
        .collect::<Vec<_>>();
    let derivative_enclosure = evaluate_real_polynomial_interval(&derivative, &interval)?;
    if derivative_enclosure.contains_zero() {
        return Err(IntervalError::Inconclusive(
            "derivative enclosure contains zero".to_owned(),
        ));
    }
    let value_at_lower = evaluate_real_polynomial_exact(coefficients_ascending, interval.lower())?;
    let value_at_upper = evaluate_real_polynomial_exact(coefficients_ascending, interval.upper())?;
    if value_at_lower == 0 || value_at_upper == 0 {
        return Err(IntervalError::Inconclusive(
            "a root lies on an interval endpoint; submit a strict isolating interval".to_owned(),
        ));
    }
    if (value_at_lower > 0) == (value_at_upper > 0) {
        return Err(IntervalError::Inconclusive(
            "endpoint signs do not prove existence".to_owned(),
        ));
    }
    Ok(PolynomialRootIsolation {
        interval,
        derivative_enclosure,
        value_at_lower,
        value_at_upper,
        unique: true,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoucheCircleCount {
    pub radius: Rational,
    pub zero_count: usize,
    pub dominant_power: usize,
    pub dominant_lower_bound: Rational,
    pub remainder_upper_bound: Rational,
    pub rigorous: bool,
}

/// Count zeros of a complex polynomial in `|z| < radius` using a rigorous
/// one-term Rouché dominance test.  The rectangular coefficient magnitude
/// bounds are conservative (`max(|re|,|im|)` below and `|re|+|im|` above).
pub fn certify_polynomial_zero_count_on_circle(
    coefficients_ascending: &[ComplexRational],
    radius: Rational,
) -> Result<RoucheCircleCount, IntervalError> {
    if coefficients_ascending.is_empty() || radius <= 0 {
        return Err(IntervalError::Invalid(
            "Rouche count requires coefficients and a positive radius".to_owned(),
        ));
    }
    let mut powers = Vec::with_capacity(coefficients_ascending.len());
    let mut power = Rational::from((1, 1));
    for _ in coefficients_ascending {
        powers.push(power.clone());
        power *= &radius;
    }
    for (dominant_power, coefficient) in coefficients_ascending.iter().enumerate() {
        let lower_abs = rational_abs(&coefficient.real).max(rational_abs(&coefficient.imaginary));
        let dominant_lower_bound = rational_mul(&lower_abs, &powers[dominant_power]);
        let mut remainder_upper_bound = Rational::from((0, 1));
        for (power_index, other) in coefficients_ascending.iter().enumerate() {
            if power_index == dominant_power {
                continue;
            }
            let upper_abs =
                rational_add(&rational_abs(&other.real), &rational_abs(&other.imaginary));
            remainder_upper_bound += rational_mul(&upper_abs, &powers[power_index]);
        }
        if dominant_lower_bound > remainder_upper_bound {
            return Ok(RoucheCircleCount {
                radius,
                zero_count: dominant_power,
                dominant_power,
                dominant_lower_bound,
                remainder_upper_bound,
                rigorous: true,
            });
        }
    }
    Err(IntervalError::Inconclusive(
        "no single polynomial term rigorously dominates on the contour".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(numerator: i32, denominator: i32) -> Rational {
        Rational::from((numerator, denominator))
    }

    #[test]
    fn rational_interval_arithmetic_contains_exact_values() {
        let a = RationalInterval::new(q(1, 3), q(1, 2)).unwrap();
        let b = RationalInterval::new(q(-2, 1), q(3, 1)).unwrap();
        let product = a.mul(&b);
        assert!(product.contains(&q(-1, 1)));
        assert!(product.contains(&q(3, 2)));
        assert_eq!(a.square(), RationalInterval::new(q(1, 9), q(1, 4)).unwrap());
        assert_eq!(b.square(), RationalInterval::new(q(0, 1), q(9, 1)).unwrap());
        assert_eq!(
            a.div(&RationalInterval::point(q(2, 1))).unwrap(),
            RationalInterval::new(q(1, 6), q(1, 4)).unwrap()
        );
        assert_eq!(b.reciprocal(), Err(IntervalError::DivisionByZeroInterval));
    }

    #[test]
    fn dyadic_square_root_enclosure_rounds_outward_exactly() {
        let quarter = RationalInterval::point(q(1, 4));
        assert_eq!(
            quarter.sqrt_nonnegative(16).unwrap(),
            RationalInterval::point(q(1, 2))
        );

        let two = RationalInterval::point(q(2, 1));
        let enclosure = two.sqrt_nonnegative(12).unwrap();
        assert!(enclosure.square().contains(&q(2, 1)));
        assert_eq!(enclosure.width(), q(1, 4096));

        assert!(RationalInterval::point(q(-1, 1))
            .sqrt_nonnegative(8)
            .is_err());
    }

    #[test]
    fn matrix_vector_and_rayleigh_enclosures_are_rigorous() {
        let matrix = [q(2, 1), q(1, 1), q(1, 1), q(3, 1)]
            .into_iter()
            .map(RationalInterval::point)
            .collect::<Vec<_>>();
        let identity = [q(1, 1), q(0, 1), q(0, 1), q(1, 1)]
            .into_iter()
            .map(RationalInterval::point)
            .collect::<Vec<_>>();
        let vector = [q(1, 1), q(2, 1)]
            .into_iter()
            .map(RationalInterval::point)
            .collect::<Vec<_>>();
        let quotient = interval_rayleigh_quotient(&matrix, &identity, 2, &vector).unwrap();
        assert!(quotient.is_point());
        assert_eq!(quotient.lower(), &q(18, 5));
    }

    #[test]
    fn simple_polynomial_root_is_certified_and_multiple_root_is_inconclusive() {
        // x^2 - 2 on [1, 3/2].
        let certificate = certify_simple_polynomial_root(
            &[q(-2, 1), q(0, 1), q(1, 1)],
            RationalInterval::new(q(1, 1), q(3, 2)).unwrap(),
        )
        .unwrap();
        assert!(certificate.unique);
        assert!(certificate.derivative_enclosure.is_strictly_positive());

        // x^2 on [-1,1] has a root, but the derivative enclosure contains 0.
        let inconclusive = certify_simple_polynomial_root(
            &[q(0, 1), q(0, 1), q(1, 1)],
            RationalInterval::new(q(-1, 1), q(1, 1)).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(inconclusive, IntervalError::Inconclusive(_)));
    }

    #[test]
    fn complex_ball_polynomial_and_rouche_count_are_rigorous() {
        // z^3 + 1/10 has exactly three zeros in the unit disk because |z^3| > 1/10.
        let coefficients = vec![
            ComplexRational {
                real: q(1, 10),
                imaginary: q(0, 1),
            },
            ComplexRational::zero(),
            ComplexRational::zero(),
            ComplexRational {
                real: q(1, 1),
                imaginary: q(0, 1),
            },
        ];
        let count = certify_polynomial_zero_count_on_circle(&coefficients, q(1, 1)).unwrap();
        assert_eq!(count.zero_count, 3);
        assert!(count.rigorous);

        let argument = ComplexRationalBall {
            real: RationalInterval::new(q(0, 1), q(1, 10)).unwrap(),
            imaginary: RationalInterval::new(q(0, 1), q(1, 10)).unwrap(),
        };
        let enclosure = evaluate_complex_polynomial_ball(&coefficients, &argument).unwrap();
        assert!(enclosure.real.contains(&q(1, 10)));
        assert!(enclosure.imaginary.contains(&q(0, 1)));
    }

    #[test]
    fn exact_rectangular_argument_counts_entire_and_meromorphic_functions() {
        // z^2 + 1 has both zeros inside [-2,2] x [-2,2].
        let numerator = vec![
            ComplexRational {
                real: q(1, 1),
                imaginary: q(0, 1),
            },
            ComplexRational::zero(),
            ComplexRational {
                real: q(1, 1),
                imaginary: q(0, 1),
            },
        ];
        let rectangle =
            RationalContourRectangle::new(q(-2, 1), q(2, 1), q(-2, 1), q(2, 1)).unwrap();
        let entire =
            certify_polynomial_zero_count_on_rectangle(&numerator, rectangle.clone(), 20).unwrap();
        assert_eq!(entire.zero_count, 2);
        assert_eq!(entire.winding_number, 2);
        assert!(entire.boundary_excludes_zero && entire.rigorous);

        // (z^2 + 1) / z has two zeros and one pole in the same contour.
        let denominator = vec![
            ComplexRational::zero(),
            ComplexRational {
                real: q(1, 1),
                imaginary: q(0, 1),
            },
        ];
        let meromorphic = certify_rational_function_argument_count_on_rectangle(
            &numerator,
            &denominator,
            rectangle,
            20,
        )
        .unwrap();
        assert_eq!(meromorphic.numerator.zero_count, 2);
        assert_eq!(meromorphic.denominator.zero_count, 1);
        assert_eq!(meromorphic.zeros_minus_poles, 1);
    }

    #[test]
    fn contour_root_on_boundary_is_inconclusive() {
        // z - 1 vanishes at the midpoint of the right edge.
        let polynomial = vec![
            ComplexRational {
                real: q(-1, 1),
                imaginary: q(0, 1),
            },
            ComplexRational {
                real: q(1, 1),
                imaginary: q(0, 1),
            },
        ];
        let rectangle =
            RationalContourRectangle::new(q(-1, 1), q(1, 1), q(-1, 1), q(1, 1)).unwrap();
        assert!(matches!(
            certify_polynomial_zero_count_on_rectangle(&polynomial, rectangle, 12),
            Err(IntervalError::Inconclusive(_))
        ));
    }

    #[test]
    fn exact_sturm_sequence_counts_distinct_roots_and_reports_multiplicity() {
        // x^3 - x has roots -1, 0, 1.
        let count =
            exact_sturm_root_count(&[q(0, 1), q(-1, 1), q(0, 1), q(1, 1)], q(-2, 1), q(2, 1))
                .unwrap();
        assert_eq!(count.distinct_real_roots, 3);
        assert!(count.square_free);

        // (x-1)^2 has one distinct root but unresolved multiplicity.
        let repeated =
            exact_sturm_root_count(&[q(1, 1), q(-2, 1), q(1, 1)], q(0, 1), q(2, 1)).unwrap();
        assert_eq!(repeated.distinct_real_roots, 1);
        assert!(!repeated.square_free);

        let isolated = exact_sturm_isolate_roots(
            &[q(0, 1), q(-1, 1), q(0, 1), q(1, 1)],
            q(-2, 1),
            q(2, 1),
            q(1, 100),
            32,
        )
        .unwrap();
        assert_eq!(isolated.len(), 3);
        assert!(isolated
            .windows(2)
            .all(|pair| pair[0].upper() < pair[1].lower()));
        assert!(isolated
            .iter()
            .all(|interval| interval.width() <= q(1, 100)));
    }
}
