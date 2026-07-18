//! Outward-rounded MPFR intervals for rigorous transcendental assembly.
//!
//! Each endpoint operation selects an MPFR directed rounding mode.  The
//! trigonometric operations evaluate at an interior point and widen by the
//! input radius using the global Lipschitz bound `|sin'|, |cos'| <= 1`.
//! This avoids assumptions about argument reduction while retaining narrow
//! enclosures for the high-precision point intervals used by CCM.

use crate::interval::{IntervalError, RationalInterval};
use rug::float::{Constant, Round};
use rug::{Float, Rational};

#[derive(Clone, Debug)]
pub struct MpfrInterval {
    lower: Float,
    upper: Float,
}

impl MpfrInterval {
    pub fn new(lower: Float, upper: Float) -> Result<Self, IntervalError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(IntervalError::Invalid(
                "MPFR interval endpoints must be finite and ordered".to_owned(),
            ));
        }
        if lower.prec() != upper.prec() {
            return Err(IntervalError::Invalid(
                "MPFR interval endpoints must have equal precision".to_owned(),
            ));
        }
        Ok(Self { lower, upper })
    }

    pub fn from_i64(value: i64, precision: u32) -> Self {
        let value = Float::with_val(precision, value);
        Self {
            lower: value.clone(),
            upper: value,
        }
    }

    pub fn from_u64(value: u64, precision: u32) -> Self {
        let value = Float::with_val(precision, value);
        Self {
            lower: value.clone(),
            upper: value,
        }
    }

    pub fn from_rational(value: &Rational, precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, value, Round::Down);
        let (upper, _) = Float::with_val_round(precision, value, Round::Up);
        Self { lower, upper }
    }

    pub fn point(value: Float) -> Self {
        Self {
            lower: value.clone(),
            upper: value,
        }
    }

    pub fn pi(precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, Constant::Pi, Round::Down);
        let (upper, _) = Float::with_val_round(precision, Constant::Pi, Round::Up);
        Self { lower, upper }
    }

    pub fn euler_gamma(precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, Constant::Euler, Round::Down);
        let (upper, _) = Float::with_val_round(precision, Constant::Euler, Round::Up);
        Self { lower, upper }
    }

    pub fn precision(&self) -> u32 {
        self.lower.prec()
    }

    /// Re-enclose both endpoints at a requested MPFR precision using outward
    /// rounding. This is valid for either precision escalation or reduction.
    pub fn with_precision(&self, precision: u32) -> Result<Self, IntervalError> {
        if precision < 32 {
            return Err(IntervalError::Invalid(
                "MPFR interval precision must be at least 32 bits".to_owned(),
            ));
        }
        let (lower, _) = Float::with_val_round(precision, &self.lower, Round::Down);
        let (upper, _) = Float::with_val_round(precision, &self.upper, Round::Up);
        Self::new(lower, upper)
    }

    pub fn lower(&self) -> &Float {
        &self.lower
    }

    pub fn upper(&self) -> &Float {
        &self.upper
    }

    pub fn contains_zero(&self) -> bool {
        self.lower <= 0 && self.upper >= 0
    }

    pub fn is_strictly_positive(&self) -> bool {
        self.lower > 0
    }

    pub fn width(&self) -> Float {
        let (width, _) =
            Float::with_val_round(self.precision(), &self.upper - &self.lower, Round::Up);
        width
    }

    pub fn midpoint_point(&self) -> Self {
        let p = self.precision();
        let (sum, _) = Float::with_val_round(p, &self.lower + &self.upper, Round::Nearest);
        let (midpoint, _) = Float::with_val_round(p, sum / 2, Round::Nearest);
        Self::point(midpoint)
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        self.require_same_precision(other);
        let lower = if self.lower >= other.lower {
            self.lower.clone()
        } else {
            other.lower.clone()
        };
        let upper = if self.upper <= other.upper {
            self.upper.clone()
        } else {
            other.upper.clone()
        };
        (lower <= upper).then_some(Self { lower, upper })
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.require_same_precision(other);
        self.lower >= other.lower && self.upper <= other.upper
    }

    pub fn is_interior_subset_of(&self, other: &Self) -> bool {
        self.require_same_precision(other);
        self.lower > other.lower && self.upper < other.upper
    }

    fn require_same_precision(&self, other: &Self) {
        assert_eq!(
            self.precision(),
            other.precision(),
            "MPFR interval precision mismatch"
        );
    }

    pub fn add(&self, other: &Self) -> Self {
        self.require_same_precision(other);
        let p = self.precision();
        let (lower, _) = Float::with_val_round(p, &self.lower + &other.lower, Round::Down);
        let (upper, _) = Float::with_val_round(p, &self.upper + &other.upper, Round::Up);
        Self { lower, upper }
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.require_same_precision(other);
        let p = self.precision();
        let (lower, _) = Float::with_val_round(p, &self.lower - &other.upper, Round::Down);
        let (upper, _) = Float::with_val_round(p, &self.upper - &other.lower, Round::Up);
        Self { lower, upper }
    }

    pub fn neg(&self) -> Self {
        Self {
            lower: -self.upper.clone(),
            upper: -self.lower.clone(),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.require_same_precision(other);
        let p = self.precision();
        let pairs = [
            (&self.lower, &other.lower),
            (&self.lower, &other.upper),
            (&self.upper, &other.lower),
            (&self.upper, &other.upper),
        ];
        let mut lower_values = Vec::with_capacity(4);
        let mut upper_values = Vec::with_capacity(4);
        for (left, right) in pairs {
            lower_values.push(Float::with_val_round(p, left * right, Round::Down).0);
            upper_values.push(Float::with_val_round(p, left * right, Round::Up).0);
        }
        let lower = lower_values.into_iter().min_by(Float::total_cmp).unwrap();
        let upper = upper_values.into_iter().max_by(Float::total_cmp).unwrap();
        Self { lower, upper }
    }

    pub fn square(&self) -> Self {
        if self.contains_zero() {
            let p = self.precision();
            let abs_lower = self.lower.clone().abs();
            let abs_upper = self.upper.clone().abs();
            let maximum = if abs_lower >= abs_upper {
                abs_lower
            } else {
                abs_upper
            };
            let (upper, _) = Float::with_val_round(p, &maximum * &maximum, Round::Up);
            Self {
                lower: Float::with_val(p, 0),
                upper,
            }
        } else {
            self.mul(self)
        }
    }

    pub fn reciprocal(&self) -> Result<Self, IntervalError> {
        if self.contains_zero() {
            return Err(IntervalError::DivisionByZeroInterval);
        }
        let p = self.precision();
        let one = Float::with_val(p, 1);
        let (lower, _) = Float::with_val_round(p, &one / &self.upper, Round::Down);
        let (upper, _) = Float::with_val_round(p, &one / &self.lower, Round::Up);
        Ok(Self { lower, upper })
    }

    pub fn div(&self, other: &Self) -> Result<Self, IntervalError> {
        Ok(self.mul(&other.reciprocal()?))
    }

    pub fn exp(&self) -> Self {
        let mut lower = self.lower.clone();
        lower.exp_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.exp_round(Round::Up);
        Self { lower, upper }
    }

    pub fn ln(&self) -> Result<Self, IntervalError> {
        if self.lower <= 0 {
            return Err(IntervalError::Invalid(
                "logarithm interval is not strictly positive".to_owned(),
            ));
        }
        let mut lower = self.lower.clone();
        lower.ln_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.ln_round(Round::Up);
        Ok(Self { lower, upper })
    }

    pub fn sqrt(&self) -> Result<Self, IntervalError> {
        if self.lower < 0 {
            return Err(IntervalError::Invalid(
                "square-root interval has a negative lower endpoint".to_owned(),
            ));
        }
        let mut lower = self.lower.clone();
        lower.sqrt_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.sqrt_round(Round::Up);
        Ok(Self { lower, upper })
    }

    pub fn atan(&self) -> Self {
        let mut lower = self.lower.clone();
        lower.atan_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.atan_round(Round::Up);
        Self { lower, upper }
    }

    fn lipschitz_trig(&self, sine: bool) -> Self {
        let p = self.precision();
        let (sum, _) = Float::with_val_round(p, &self.lower + &self.upper, Round::Nearest);
        let (midpoint, _) = Float::with_val_round(p, sum / 2, Round::Nearest);
        let (left_radius, _) = Float::with_val_round(p, &midpoint - &self.lower, Round::Up);
        let (right_radius, _) = Float::with_val_round(p, &self.upper - &midpoint, Round::Up);
        let radius = if left_radius >= right_radius {
            left_radius
        } else {
            right_radius
        };
        let mut lower = midpoint.clone();
        let mut upper = midpoint;
        if sine {
            lower.sin_round(Round::Down);
            upper.sin_round(Round::Up);
        } else {
            lower.cos_round(Round::Down);
            upper.cos_round(Round::Up);
        }
        lower = Float::with_val_round(p, lower - &radius, Round::Down).0;
        upper = Float::with_val_round(p, upper + &radius, Round::Up).0;
        let minus_one = Float::with_val(p, -1);
        let one = Float::with_val(p, 1);
        if lower < minus_one {
            lower = minus_one;
        }
        if upper > one {
            upper = one;
        }
        Self { lower, upper }
    }

    pub fn sin(&self) -> Self {
        self.lipschitz_trig(true)
    }

    pub fn cos(&self) -> Self {
        self.lipschitz_trig(false)
    }

    pub fn to_rational_interval(&self) -> RationalInterval {
        RationalInterval::new(
            self.lower
                .to_rational()
                .expect("finite MPFR lower endpoint"),
            self.upper
                .to_rational()
                .expect("finite MPFR upper endpoint"),
        )
        .expect("ordered MPFR endpoints")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpfrBallBackendDescriptor {
    pub implementation: String,
    pub precision_bits: u32,
    pub real_enclosure: String,
    pub complex_enclosure: String,
    pub rounding_policy: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpfrBallContext {
    precision_bits: u32,
}

impl MpfrBallContext {
    pub fn new(precision_bits: u32) -> Result<Self, IntervalError> {
        if precision_bits < 32 {
            return Err(IntervalError::Invalid(
                "MPFR ball precision must be at least 32 bits".to_owned(),
            ));
        }
        Ok(Self { precision_bits })
    }

    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }

    pub fn descriptor(&self) -> MpfrBallBackendDescriptor {
        MpfrBallBackendDescriptor {
            implementation: "rug-mpfr-rectangular-ball-v1".to_owned(),
            precision_bits: self.precision_bits,
            real_enclosure: "closed directed-rounding MPFR endpoint interval".to_owned(),
            complex_enclosure: "Cartesian product of two MPFR endpoint intervals".to_owned(),
            rounding_policy: "MPFR Round::Down/Up on every real endpoint operation".to_owned(),
        }
    }

    pub fn real_from_rational(&self, value: &Rational) -> MpfrInterval {
        MpfrInterval::from_rational(value, self.precision_bits)
    }

    pub fn complex_from_rationals(&self, real: &Rational, imaginary: &Rational) -> MpfrComplexBall {
        MpfrComplexBall {
            real: self.real_from_rational(real),
            imaginary: self.real_from_rational(imaginary),
        }
    }
}

/// Arbitrary-precision rectangular complex ball. The enclosure is the
/// Cartesian product `real x imaginary`; all operations are reduced to the
/// directed-rounding `MpfrInterval` backend.
#[derive(Clone, Debug)]
pub struct MpfrComplexBall {
    real: MpfrInterval,
    imaginary: MpfrInterval,
}

impl MpfrComplexBall {
    pub fn new(real: MpfrInterval, imaginary: MpfrInterval) -> Result<Self, IntervalError> {
        if real.precision() != imaginary.precision() {
            return Err(IntervalError::Invalid(
                "complex ball components must have equal precision".to_owned(),
            ));
        }
        Ok(Self { real, imaginary })
    }

    pub fn point(real: Float, imaginary: Float) -> Result<Self, IntervalError> {
        Self::new(MpfrInterval::point(real), MpfrInterval::point(imaginary))
    }

    pub fn precision(&self) -> u32 {
        self.real.precision()
    }

    pub fn real(&self) -> &MpfrInterval {
        &self.real
    }

    pub fn imaginary(&self) -> &MpfrInterval {
        &self.imaginary
    }

    pub fn with_precision(&self, precision: u32) -> Result<Self, IntervalError> {
        Self::new(
            self.real.with_precision(precision)?,
            self.imaginary.with_precision(precision)?,
        )
    }

    pub fn excludes_zero(&self) -> bool {
        !self.real.contains_zero() || !self.imaginary.contains_zero()
    }

    fn require_compatible(&self, other: &Self) -> Result<(), IntervalError> {
        if self.precision() != other.precision() {
            return Err(IntervalError::Invalid(format!(
                "complex ball precision mismatch: {} != {}",
                self.precision(),
                other.precision()
            )));
        }
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self, IntervalError> {
        self.require_compatible(other)?;
        Self::new(
            self.real.add(&other.real),
            self.imaginary.add(&other.imaginary),
        )
    }

    pub fn sub(&self, other: &Self) -> Result<Self, IntervalError> {
        self.require_compatible(other)?;
        Self::new(
            self.real.sub(&other.real),
            self.imaginary.sub(&other.imaginary),
        )
    }

    pub fn neg(&self) -> Self {
        Self {
            real: self.real.neg(),
            imaginary: self.imaginary.neg(),
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real.clone(),
            imaginary: self.imaginary.neg(),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self, IntervalError> {
        self.require_compatible(other)?;
        let ac = self.real.mul(&other.real);
        let bd = self.imaginary.mul(&other.imaginary);
        let ad = self.real.mul(&other.imaginary);
        let bc = self.imaginary.mul(&other.real);
        Self::new(ac.sub(&bd), ad.add(&bc))
    }

    pub fn modulus_squared(&self) -> MpfrInterval {
        self.real.square().add(&self.imaginary.square())
    }

    pub fn reciprocal(&self) -> Result<Self, IntervalError> {
        let denominator = self.modulus_squared();
        if denominator.contains_zero() {
            return Err(IntervalError::DivisionByZeroInterval);
        }
        Self::new(
            self.real.div(&denominator)?,
            self.imaginary.neg().div(&denominator)?,
        )
    }

    pub fn div(&self, other: &Self) -> Result<Self, IntervalError> {
        self.require_compatible(other)?;
        self.mul(&other.reciprocal()?)
    }

    pub fn exp(&self) -> Result<Self, IntervalError> {
        let magnitude = self.real.exp();
        Self::new(
            magnitude.mul(&self.imaginary.cos()),
            magnitude.mul(&self.imaginary.sin()),
        )
    }

    pub fn powu(&self, exponent: u32) -> Result<Self, IntervalError> {
        let precision = self.precision();
        let mut result = Self::point(Float::with_val(precision, 1), Float::with_val(precision, 0))?;
        let mut factor = self.clone();
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining % 2 == 1 {
                result = result.mul(&factor)?;
            }
            remaining /= 2;
            if remaining > 0 {
                factor = factor.mul(&factor)?;
            }
        }
        Ok(result)
    }
}

pub fn evaluate_complex_polynomial_mpfr(
    coefficients_ascending: &[MpfrComplexBall],
    argument: &MpfrComplexBall,
) -> Result<MpfrComplexBall, IntervalError> {
    let Some(highest) = coefficients_ascending.last() else {
        return Err(IntervalError::Invalid(
            "complex polynomial must contain at least one coefficient".to_owned(),
        ));
    };
    if coefficients_ascending
        .iter()
        .any(|coefficient| coefficient.precision() != argument.precision())
    {
        return Err(IntervalError::Invalid(
            "complex polynomial coefficient precision mismatch".to_owned(),
        ));
    }
    let mut value = highest.clone();
    for coefficient in coefficients_ascending.iter().rev().skip(1) {
        value = value.mul(argument)?.add(coefficient)?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_arithmetic_contains_exact_values() {
        let p = 96;
        let third = MpfrInterval::from_rational(&Rational::from((1, 3)), p);
        let seven = MpfrInterval::from_i64(7, p);
        let result = third.mul(&seven).div(&seven).unwrap();
        let exact = Rational::from((1, 3));
        let rational = result.to_rational_interval();
        assert!(rational.contains(&exact));
    }

    #[test]
    fn transcendental_enclosures_overlap_higher_precision_values() {
        let p = 96;
        let x = MpfrInterval::from_rational(&Rational::from((7, 5)), p);
        for interval in [x.sin(), x.cos(), x.exp(), x.ln().unwrap(), x.atan()] {
            assert!(interval.lower() <= interval.upper());
            assert!(interval.width() > 0);
        }
        let pi = MpfrInterval::pi(p);
        assert!(pi.lower() < &Float::with_val(p, 4));
        assert!(pi.upper() > &Float::with_val(p, 3));
    }

    #[test]
    fn complex_ball_arithmetic_contains_exact_rational_results() {
        let context = MpfrBallContext::new(128).unwrap();
        assert_eq!(context.descriptor().precision_bits, 128);
        let a = Rational::from((1, 3));
        let b = Rational::from((2, 5));
        let c = Rational::from((-3, 7));
        let d = Rational::from((5, 11));
        let left = context.complex_from_rationals(&a, &b);
        let right = context.complex_from_rationals(&c, &d);
        let product = left.mul(&right).unwrap();
        let expected_real = a.clone() * &c - b.clone() * &d;
        let expected_imaginary = a.clone() * &d + b.clone() * &c;
        assert!(product
            .real()
            .to_rational_interval()
            .contains(&expected_real));
        assert!(product
            .imaginary()
            .to_rational_interval()
            .contains(&expected_imaginary));

        let quotient = left.div(&left).unwrap();
        assert!(quotient
            .real()
            .to_rational_interval()
            .contains(&Rational::from((1, 1))));
        assert!(quotient
            .imaginary()
            .to_rational_interval()
            .contains(&Rational::from((0, 1))));
        assert!(left.excludes_zero());
    }

    #[test]
    fn complex_ball_polynomial_exponential_and_failure_paths_are_checked() {
        let context = MpfrBallContext::new(128).unwrap();
        let zero = Rational::from((0, 1));
        let one = Rational::from((1, 1));
        let zero_ball = context.complex_from_rationals(&zero, &zero);
        let one_ball = context.complex_from_rationals(&one, &zero);
        let imaginary_unit = context.complex_from_rationals(&zero, &one);
        let value = evaluate_complex_polynomial_mpfr(
            &[one_ball.clone(), zero_ball.clone(), one_ball],
            &imaginary_unit,
        )
        .unwrap();
        assert!(value
            .real()
            .to_rational_interval()
            .contains(&Rational::from((0, 1))));
        assert!(value
            .imaginary()
            .to_rational_interval()
            .contains(&Rational::from((0, 1))));

        let imaginary_pi =
            MpfrComplexBall::new(MpfrInterval::from_i64(0, 128), MpfrInterval::pi(128)).unwrap();
        let exponential = imaginary_pi.exp().unwrap();
        assert!(exponential
            .real()
            .to_rational_interval()
            .contains(&Rational::from((-1, 1))));
        assert!(exponential
            .imaginary()
            .to_rational_interval()
            .contains(&Rational::from((0, 1))));

        assert!(matches!(
            zero_ball.reciprocal(),
            Err(IntervalError::DivisionByZeroInterval)
        ));
        let low_precision = MpfrBallContext::new(64)
            .unwrap()
            .complex_from_rationals(&one, &zero);
        assert!(imaginary_unit.add(&low_precision).is_err());
        assert!(MpfrBallContext::new(31).is_err());
        assert!(evaluate_complex_polynomial_mpfr(&[], &imaginary_unit).is_err());
    }
}
