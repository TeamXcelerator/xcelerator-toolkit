//! Outward-rounded MPFR intervals for rigorous transcendental assembly.
//!
//! Each endpoint operation selects an MPFR directed rounding mode.  The
//! trigonometric operations evaluate at an interior point and widen by the
//! input radius using the global Lipschitz bound `|sin'|, |cos'| <= 1`.
//! This avoids assumptions about argument reduction while retaining narrow
//! enclosures for the high-precision point intervals used by CCM.
//!
//! Finite intervals are the supported domain. Infallible arithmetic preserves
//! the existing API and returns an invalid NaN interval if a finite enclosure
//! cannot be represented or operand precisions differ. Such a value must never establish a proof: call
//! `validate()` at certificate boundaries. Sign/subset predicates fail closed,
//! and fallible arithmetic returns an error for invalid operands or results.
//! Constructors with a raw precision argument follow MPFR and panic outside its
//! supported precision range; `from_float` is the checked alternative.
//! `with_precision` intentionally requires at least 32 bits, the ball-backend
//! policy, while exact construction can represent lower MPFR precisions.
//! `point`, `midpoint_point`, `to_rational_interval`, and compatibility `intersection` require valid
//! inputs and can panic; use checked construction/intersection at boundaries.

use crate::interval::{IntervalError, RationalInterval};
use rug::float::{Constant, Round};
use rug::{Float, Rational};
use std::cmp::Ordering;

/// Fail before parallel MPFR work unless every worker of the current Rayon
/// pool shares the caller's exponent range. MPFR exponent bounds are
/// thread-local, so arithmetic on a worker with other bounds could round or
/// underflow differently and make results depend on scheduling. The check
/// visits every worker, so its own outcome does not depend on scheduling.
pub fn ensure_uniform_exponent_range() -> Result<(), IntervalError> {
    let caller = (rug::float::exp_min(), rug::float::exp_max());
    let workers = rayon::broadcast(|_| (rug::float::exp_min(), rug::float::exp_max()));
    if workers.iter().any(|worker| *worker != caller) {
        return Err(IntervalError::Inconclusive(
            "parallel MPFR workers do not share the caller's exponent range".into(),
        ));
    }
    Ok(())
}

/// Downward and upward correctly rounded values of the correctly rounded MPFR
/// function `f` at one stored point, bitwise equal to two directed calls.
/// One evaluation suffices when the downward result settles the upward one;
/// see `upward_from_downward`.
pub fn directed_point_pair(x: &Float, f: impl Fn(&mut Float, Round) -> Ordering) -> (Float, Float) {
    let mut lower = x.clone();
    let ternary = f(&mut lower, Round::Down);
    let upper = upward_from_downward(x, &lower, ternary, f);
    (lower, upper)
}

/// The upward rounding of `f(x)` from its downward rounding `lower` and that
/// call's ternary. Correct rounding makes an exact result equal in either
/// direction and an inexact downward result the predecessor of the upward
/// one. Zeros, results at the exponent extremes and every other case evaluate
/// the upward rounding directly.
fn upward_from_downward(
    x: &Float,
    lower: &Float,
    ternary: Ordering,
    f: impl Fn(&mut Float, Round) -> Ordering,
) -> Float {
    if lower.is_normal() {
        match ternary {
            Ordering::Equal => return lower.clone(),
            Ordering::Less
                if lower
                    .get_exp()
                    .is_some_and(|exponent| exponent > rug::float::exp_min()) =>
            {
                let mut upper = lower.clone();
                upper.next_up();
                if upper.is_finite() {
                    return upper;
                }
            }
            _ => {}
        }
    }
    let mut upper = x.clone();
    f(&mut upper, Round::Up);
    upper
}

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

    /// Check the finite, ordered, equal-precision enclosure contract.
    pub fn validate(&self) -> Result<(), IntervalError> {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || self.lower > self.upper
            || self.lower.prec() != self.upper.prec()
        {
            return Err(IntervalError::Invalid(
                "invalid or unrepresentable MPFR interval".into(),
            ));
        }
        Ok(())
    }

    fn invalid(precision: u32) -> Self {
        let nan = Float::with_val(precision, rug::float::Special::Nan);
        Self {
            lower: nan.clone(),
            upper: nan,
        }
    }

    fn arithmetic_result(lower: Float, upper: Float) -> Self {
        let precision = lower.prec();
        Self::new(lower, upper).unwrap_or_else(|_| Self::invalid(precision))
    }

    /// Enclose the exact stored Float when changing precision, including reduction.
    pub fn from_float(value: &Float, precision: u32) -> Result<Self, IntervalError> {
        if !value.is_finite()
            || !(rug::float::prec_min()..=rug::float::prec_max()).contains(&precision)
        {
            return Err(IntervalError::Invalid(
                "finite Float and supported precision required".into(),
            ));
        }
        let (lower, _) = Float::with_val_round(precision, value, Round::Down);
        let (upper, _) = Float::with_val_round(precision, value, Round::Up);
        Self::new(lower, upper)
    }

    /// Enclose the exact integer even when its significand exceeds `precision`.
    pub fn from_i64(value: i64, precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, value, Round::Down);
        let (upper, _) = Float::with_val_round(precision, value, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    /// Enclose the exact integer even when its significand exceeds `precision`.
    pub fn from_u64(value: u64, precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, value, Round::Down);
        let (upper, _) = Float::with_val_round(precision, value, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    pub fn from_rational(value: &Rational, precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, value, Round::Down);
        let (upper, _) = Float::with_val_round(precision, value, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    /// A singleton containing an exact stored finite Float.
    ///
    /// Panics for a nonfinite value; use `from_float` for fallible construction.
    pub fn point(value: Float) -> Self {
        Self::new(value.clone(), value).expect("finite MPFR point required")
    }

    pub fn pi(precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, Constant::Pi, Round::Down);
        let (upper, _) = Float::with_val_round(precision, Constant::Pi, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    pub fn euler_gamma(precision: u32) -> Self {
        let (lower, _) = Float::with_val_round(precision, Constant::Euler, Round::Down);
        let (upper, _) = Float::with_val_round(precision, Constant::Euler, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    pub fn precision(&self) -> u32 {
        self.lower.prec()
    }

    /// Re-enclose both endpoints at a requested MPFR precision using outward
    /// rounding. This is valid for either precision escalation or reduction.
    pub fn with_precision(&self, precision: u32) -> Result<Self, IntervalError> {
        self.validate()?;
        if !(32..=rug::float::prec_max()).contains(&precision) {
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
        // An invalid value cannot prove exclusion of zero.
        self.validate().is_err() || (self.lower <= 0 && self.upper >= 0)
    }

    pub fn is_strictly_positive(&self) -> bool {
        self.validate().is_ok() && self.lower > 0
    }

    pub fn width(&self) -> Float {
        let (width, _) =
            Float::with_val_round(self.precision(), &self.upper - &self.lower, Round::Up);
        width
    }

    pub fn midpoint_point(&self) -> Self {
        let p = self.precision();
        self.validate().expect("finite interval midpoint required");
        // Same-sign subtraction cannot overflow. Opposite-sign half-sums
        // cannot overflow, and clamping handles exponent-floor rounding.
        let midpoint = if self.lower.is_sign_negative() == self.upper.is_sign_negative() {
            let difference = Float::with_val(p, &self.upper - &self.lower);
            Float::with_val(p, &self.lower + difference / 2)
        } else {
            Float::with_val(p, self.lower.clone() / 2 + self.upper.clone() / 2)
        };
        let midpoint = if midpoint < self.lower {
            self.lower.clone()
        } else if midpoint > self.upper {
            self.upper.clone()
        } else {
            midpoint
        };
        Self::point(midpoint)
    }

    /// Panics on invalid operands; use `try_intersection` at proof boundaries.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        self.try_intersection(other)
            .expect("valid MPFR intersection operands required")
    }

    /// Distinguish invalid arithmetic from a proved empty intersection.
    pub fn try_intersection(&self, other: &Self) -> Result<Option<Self>, IntervalError> {
        self.validate()?;
        other.validate()?;
        if self.precision() != other.precision() {
            return Err(IntervalError::Invalid(
                "MPFR interval precision mismatch".into(),
            ));
        }
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
        Ok((lower <= upper).then_some(Self { lower, upper }))
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.precision() == other.precision()
            && self.validate().is_ok()
            && other.validate().is_ok()
            && self.lower >= other.lower
            && self.upper <= other.upper
    }

    pub fn is_interior_subset_of(&self, other: &Self) -> bool {
        self.precision() == other.precision()
            && self.validate().is_ok()
            && other.validate().is_ok()
            && self.lower > other.lower
            && self.upper < other.upper
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.precision() != other.precision()
            || self.validate().is_err()
            || other.validate().is_err()
        {
            return Self::invalid(self.precision());
        }
        let p = self.precision();
        let (lower, _) = Float::with_val_round(p, &self.lower + &other.lower, Round::Down);
        let (upper, _) = Float::with_val_round(p, &self.upper + &other.upper, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self.precision() != other.precision()
            || self.validate().is_err()
            || other.validate().is_err()
        {
            return Self::invalid(self.precision());
        }
        let p = self.precision();
        let (lower, _) = Float::with_val_round(p, &self.lower - &other.upper, Round::Down);
        let (upper, _) = Float::with_val_round(p, &self.upper - &other.lower, Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    pub fn neg(&self) -> Self {
        Self {
            lower: -self.upper.clone(),
            upper: -self.lower.clone(),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.precision() != other.precision()
            || self.validate().is_err()
            || other.validate().is_err()
        {
            return Self::invalid(self.precision());
        }
        let p = self.precision();
        // A point operand repeats its corner products. Rounded products of
        // bitwise-equal operands are bitwise equal, so the distinct corners
        // give the same extrema; signed zeros keep both endpoints.
        let (left, left_count) = self.corners();
        let (right, right_count) = other.corners();
        let mut lower: Option<Float> = None;
        let mut upper: Option<Float> = None;
        for &x in &left[..left_count] {
            for &y in &right[..right_count] {
                let down = Float::with_val_round(p, x * y, Round::Down).0;
                let up = Float::with_val_round(p, x * y, Round::Up).0;
                if lower.as_ref().is_none_or(|v| down.total_cmp(v).is_lt()) {
                    lower = Some(down);
                }
                if upper.as_ref().is_none_or(|v| up.total_cmp(v).is_gt()) {
                    upper = Some(up);
                }
            }
        }
        Self::arithmetic_result(lower.unwrap(), upper.unwrap())
    }

    /// The distinct endpoints: one for a bitwise point, including zero sign.
    fn corners(&self) -> ([&Float; 2], usize) {
        let point = self.lower == self.upper
            && self.lower.is_sign_negative() == self.upper.is_sign_negative();
        ([&self.lower, &self.upper], if point { 1 } else { 2 })
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
            Self::arithmetic_result(Float::with_val(p, 0), upper)
        } else {
            self.mul(self)
        }
    }

    pub fn reciprocal(&self) -> Result<Self, IntervalError> {
        self.validate()?;
        if self.contains_zero() {
            return Err(IntervalError::DivisionByZeroInterval);
        }
        let p = self.precision();
        let one = Float::with_val(p, 1);
        let (lower, _) = Float::with_val_round(p, &one / &self.upper, Round::Down);
        let (upper, _) = Float::with_val_round(p, &one / &self.lower, Round::Up);
        Self::new(lower, upper)
    }

    pub fn div(&self, other: &Self) -> Result<Self, IntervalError> {
        self.validate()?;
        other.validate()?;
        if self.precision() != other.precision() {
            return Err(IntervalError::Invalid(
                "MPFR interval precision mismatch".into(),
            ));
        }
        let result = self.mul(&other.reciprocal()?);
        result.validate()?;
        Ok(result)
    }

    pub fn exp(&self) -> Self {
        let mut lower = self.lower.clone();
        lower.exp_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.exp_round(Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    pub fn ln(&self) -> Result<Self, IntervalError> {
        self.validate()?;
        if self.lower <= 0 {
            return Err(IntervalError::Invalid(
                "logarithm interval is not strictly positive".to_owned(),
            ));
        }
        let mut lower = self.lower.clone();
        lower.ln_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.ln_round(Round::Up);
        Self::new(lower, upper)
    }

    pub fn sqrt(&self) -> Result<Self, IntervalError> {
        self.validate()?;
        if self.lower < 0 {
            return Err(IntervalError::Invalid(
                "square-root interval has a negative lower endpoint".to_owned(),
            ));
        }
        let mut lower = self.lower.clone();
        lower.sqrt_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.sqrt_round(Round::Up);
        Self::new(lower, upper)
    }

    pub fn atan(&self) -> Self {
        let mut lower = self.lower.clone();
        lower.atan_round(Round::Down);
        let mut upper = self.upper.clone();
        upper.atan_round(Round::Up);
        Self::arithmetic_result(lower, upper)
    }

    /// Midpoint and radius of the Lipschitz trigonometric enclosure; `None`
    /// when the radius already covers the full range of sine and cosine.
    fn trig_center(&self) -> Option<(Float, Float)> {
        let p = self.precision();
        let midpoint = self.midpoint_point().lower;
        let (left_radius, _) = Float::with_val_round(p, &midpoint - &self.lower, Round::Up);
        let (right_radius, _) = Float::with_val_round(p, &self.upper - &midpoint, Round::Up);
        let radius = if left_radius >= right_radius {
            left_radius
        } else {
            right_radius
        };
        // A radius of two already covers the full range of either function.
        // Avoid expensive argument reduction when it cannot improve this bound.
        (radius < 2).then_some((midpoint, radius))
    }

    fn trig_widened(lower: Float, upper: Float, radius: &Float) -> Self {
        let p = lower.prec();
        // Borrow both operands: an owned left operand is rounded to nearest in
        // place before the directed conversion can apply.
        let mut lower = Float::with_val_round(p, &lower - radius, Round::Down).0;
        let mut upper = Float::with_val_round(p, &upper + radius, Round::Up).0;
        let minus_one = Float::with_val(p, -1);
        let one = Float::with_val(p, 1);
        if lower < minus_one {
            lower = minus_one;
        }
        if upper > one {
            upper = one;
        }
        Self::arithmetic_result(lower, upper)
    }

    fn lipschitz_trig(&self, sine: bool) -> Self {
        let p = self.precision();
        if self.validate().is_err() {
            return Self::invalid(p);
        }
        let Some((midpoint, radius)) = self.trig_center() else {
            return Self::arithmetic_result(Float::with_val(p, -1), Float::with_val(p, 1));
        };
        let (lower, upper) = if sine {
            directed_point_pair(&midpoint, |x, round| x.sin_round(round))
        } else {
            directed_point_pair(&midpoint, |x, round| x.cos_round(round))
        };
        Self::trig_widened(lower, upper, &radius)
    }

    /// `(self.sin(), self.cos())`, sharing one midpoint evaluation.
    pub fn sin_cos(&self) -> (Self, Self) {
        let p = self.precision();
        if self.validate().is_err() {
            return (Self::invalid(p), Self::invalid(p));
        }
        let Some((midpoint, radius)) = self.trig_center() else {
            let full = Self::arithmetic_result(Float::with_val(p, -1), Float::with_val(p, 1));
            return (full.clone(), full);
        };
        // MPFR rounds each sin_cos output correctly, with its own ternary.
        let mut sin_lower = midpoint.clone();
        let mut cos_lower = Float::new(p);
        let (sin_ternary, cos_ternary) = sin_lower.sin_cos_round(&mut cos_lower, Round::Down);
        let sin_upper = upward_from_downward(&midpoint, &sin_lower, sin_ternary, |x, round| {
            x.sin_round(round)
        });
        let cos_upper = upward_from_downward(&midpoint, &cos_lower, cos_ternary, |x, round| {
            x.cos_round(round)
        });
        (
            Self::trig_widened(sin_lower, sin_upper, &radius),
            Self::trig_widened(cos_lower, cos_upper, &radius),
        )
    }

    pub fn sin(&self) -> Self {
        self.lipschitz_trig(true)
    }

    pub fn cos(&self) -> Self {
        self.lipschitz_trig(false)
    }

    /// Compatibility conversion; panics on invalid endpoints. Use the checked
    /// variant at arithmetic and artifact boundaries.
    pub fn to_rational_interval(&self) -> RationalInterval {
        self.try_to_rational_interval()
            .expect("valid finite MPFR endpoints required")
    }
    /// Convert finite, ordered, equal-precision stored endpoints exactly.
    pub fn try_to_rational_interval(&self) -> Result<RationalInterval, IntervalError> {
        self.validate()?;
        let lower = self
            .lower
            .to_rational()
            .ok_or_else(|| IntervalError::Invalid("nonfinite MPFR lower endpoint".into()))?;
        let upper = self
            .upper
            .to_rational()
            .ok_or_else(|| IntervalError::Invalid("nonfinite MPFR upper endpoint".into()))?;
        RationalInterval::new(lower, upper)
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
        if !(32..=rug::float::prec_max()).contains(&precision_bits) {
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
    pub fn validate(&self) -> Result<(), IntervalError> {
        self.real.validate()?;
        self.imaginary.validate()?;
        if self.real.precision() != self.imaginary.precision() {
            return Err(IntervalError::Invalid(
                "complex ball component precision mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn new(real: MpfrInterval, imaginary: MpfrInterval) -> Result<Self, IntervalError> {
        real.validate()?;
        imaginary.validate()?;
        if real.precision() != imaginary.precision() {
            return Err(IntervalError::Invalid(
                "complex ball components must have equal precision".to_owned(),
            ));
        }
        Ok(Self { real, imaginary })
    }

    pub fn point(real: Float, imaginary: Float) -> Result<Self, IntervalError> {
        let real = MpfrInterval::from_float(&real, real.prec())?;
        let imaginary = MpfrInterval::from_float(&imaginary, imaginary.prec())?;
        Self::new(real, imaginary)
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
        self.validate().is_ok() && (!self.real.contains_zero() || !self.imaginary.contains_zero())
    }

    fn require_compatible(&self, other: &Self) -> Result<(), IntervalError> {
        self.validate()?;
        other.validate()?;
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
        self.validate()?;
        let denominator = self.modulus_squared();
        denominator.validate()?;
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
        self.validate()?;
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
    argument.validate()?;
    for coefficient in coefficients_ascending {
        coefficient.validate()?;
    }
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
    use rayon::prelude::*;

    // Frozen eight-product multiplication preceding the point fast paths.
    fn reference_mul(a: &MpfrInterval, b: &MpfrInterval) -> MpfrInterval {
        if a.precision() != b.precision() || a.validate().is_err() || b.validate().is_err() {
            return MpfrInterval::invalid(a.precision());
        }
        let p = a.precision();
        let pairs = [
            (&a.lower, &b.lower),
            (&a.lower, &b.upper),
            (&a.upper, &b.lower),
            (&a.upper, &b.upper),
        ];
        let mut lower_values = Vec::with_capacity(4);
        let mut upper_values = Vec::with_capacity(4);
        for (left, right) in pairs {
            lower_values.push(Float::with_val_round(p, left * right, Round::Down).0);
            upper_values.push(Float::with_val_round(p, left * right, Round::Up).0);
        }
        let lower = lower_values.into_iter().min_by(Float::total_cmp).unwrap();
        let upper = upper_values.into_iter().max_by(Float::total_cmp).unwrap();
        MpfrInterval::arithmetic_result(lower, upper)
    }

    // Frozen two-evaluation trigonometric enclosure preceding the shared
    // directed pair.
    fn reference_trig(x: &MpfrInterval, sine: bool) -> MpfrInterval {
        let p = x.precision();
        if x.validate().is_err() {
            return MpfrInterval::invalid(p);
        }
        let midpoint = x.midpoint_point().lower;
        let (left_radius, _) = Float::with_val_round(p, &midpoint - &x.lower, Round::Up);
        let (right_radius, _) = Float::with_val_round(p, &x.upper - &midpoint, Round::Up);
        let radius = if left_radius >= right_radius {
            left_radius
        } else {
            right_radius
        };
        if radius >= 2 {
            return MpfrInterval::arithmetic_result(Float::with_val(p, -1), Float::with_val(p, 1));
        }
        let mut lower = midpoint.clone();
        let mut upper = midpoint;
        if sine {
            lower.sin_round(Round::Down);
            upper.sin_round(Round::Up);
        } else {
            lower.cos_round(Round::Down);
            upper.cos_round(Round::Up);
        }
        lower = Float::with_val_round(p, &lower - &radius, Round::Down).0;
        upper = Float::with_val_round(p, &upper + &radius, Round::Up).0;
        let minus_one = Float::with_val(p, -1);
        let one = Float::with_val(p, 1);
        if lower < minus_one {
            lower = minus_one;
        }
        if upper > one {
            upper = one;
        }
        MpfrInterval::arithmetic_result(lower, upper)
    }

    // Bitwise: precision, value and the sign of zero (NaN by sign).
    fn same_float(a: &Float, b: &Float) -> bool {
        a.prec() == b.prec()
            && if a.is_nan() || b.is_nan() {
                a.is_nan() && b.is_nan() && a.is_sign_negative() == b.is_sign_negative()
            } else {
                a.total_cmp(b).is_eq()
            }
    }

    fn same_interval(a: &MpfrInterval, b: &MpfrInterval) -> bool {
        same_float(&a.lower, &b.lower) && same_float(&a.upper, &b.upper)
    }

    // Deterministic finite sample: random significand, sign and exponent,
    // with a few fixed special values at the start of every stream.
    fn sample(p: u32, index: u64, exponent_span: i32) -> Float {
        let mut state = index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(p);
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let half_pi = Float::with_val(p, Constant::Pi) >> 1;
        let minimum = Float::with_val(p, rug::float::Special::Zero).next_up_value();
        match index {
            0 => return Float::with_val(p, 0),
            1 => return -Float::with_val(p, 0),
            2 => return Float::with_val(p, 1),
            3 => return half_pi,
            4 => return -half_pi * 3u32,
            5 => return minimum,
            6 => return -minimum,
            7 => return Float::with_val(p, 1) >> 100_000,
            8 => return Float::with_val(p, Constant::Pi) * 1_000_000u32,
            _ => {}
        }
        let limbs = (p as usize).div_ceil(64);
        let digits = (0..limbs).map(|_| next()).collect::<Vec<_>>();
        let significand = rug::Integer::from_digits(&digits, rug::integer::Order::Lsf);
        let exponent = (next() % (2 * exponent_span as u64 + 1)) as i32 - exponent_span;
        let value = (Float::with_val(p, significand) >> (64 * limbs as i32)) << exponent;
        let value = if next() % 7 == 0 {
            // Exactly representable small dyadics and near multiples of pi/2.
            if next() % 2 == 0 {
                Float::with_val(p, (next() % 64) as i64 - 32) >> (next() % 8) as u32
            } else {
                (Float::with_val(p, Constant::Pi) * ((next() % 64) as i64 - 32)) >> 1
            }
        } else {
            value
        };
        if next() % 2 == 0 {
            -value
        } else {
            value
        }
    }

    trait NextUpValue {
        fn next_up_value(self) -> Self;
    }
    impl NextUpValue for Float {
        fn next_up_value(mut self) -> Self {
            self.next_up();
            self
        }
    }

    #[test]
    fn point_multiplication_matches_all_four_corners_bitwise() {
        for p in [32, 128, 3386, 9000] {
            let zero = Float::with_val(p, 0);
            let minimum = zero.clone().next_up_value();
            let maximum = Float::with_val(p, 1) << (rug::float::exp_max() - 1);
            let mut values = vec![
                zero.clone(),
                -zero,
                minimum.clone(),
                -minimum.clone(),
                minimum.next_up_value(),
                maximum.clone(),
                -maximum.clone(),
                Float::with_val(p, 1) << (rug::float::exp_max() / 2),
                Float::with_val(p, 1) >> (-(rug::float::exp_min() / 2)),
            ];
            for k in [-3i32, -1, 1, 2, 7] {
                values.push(Float::with_val(p, k) / 3u32);
                values.push((Float::with_val(p, k) / 7u32) << 4000);
                values.push((Float::with_val(p, k) / 11u32) >> 4000);
            }
            let mut intervals = Vec::new();
            for a in &values {
                for b in &values {
                    // Ordered pairs only; [+0, -0] is a valid interval.
                    if a <= b {
                        intervals.push(MpfrInterval {
                            lower: a.clone(),
                            upper: b.clone(),
                        });
                    }
                }
            }
            intervals.push(MpfrInterval::from_i64(1, p + 1));
            intervals.push(MpfrInterval::invalid(p));
            intervals.par_iter().for_each(|left| {
                for right in &intervals {
                    let actual = left.mul(right);
                    let expected = reference_mul(left, right);
                    assert!(
                        same_interval(&actual, &expected),
                        "{left:?} * {right:?}: {actual:?} != {expected:?}"
                    );
                }
            });
        }
    }

    #[test]
    fn single_evaluation_directed_functions_match_two_directed_calls() {
        type Function = fn(&mut Float, Round) -> Ordering;
        let functions: [(Function, i32); 4] = [
            (|x, round| x.sin_round(round), 64),
            (|x, round| x.cos_round(round), 64),
            (|x, round| x.sinh_round(round), 40),
            (|x, round| x.exp_m1_round(round), 40),
        ];
        for p in [64, 128, 512, 3386, 9000] {
            let samples: u64 = 100_000;
            (0..samples).into_par_iter().for_each(|index| {
                let x = sample(p, index, 64);
                for (number, (f, span)) in functions.iter().enumerate() {
                    if number >= 2 && x.get_exp().is_some_and(|e| e > *span) {
                        continue;
                    }
                    let mut lower = x.clone();
                    f(&mut lower, Round::Down);
                    let mut upper = x.clone();
                    f(&mut upper, Round::Up);
                    let (actual_lower, actual_upper) = directed_point_pair(&x, f);
                    assert!(
                        same_float(&actual_lower, &lower) && same_float(&actual_upper, &upper),
                        "function {number} at {x}"
                    );
                }
                let mut sin = x.clone();
                let mut cos = Float::new(p);
                sin.sin_cos_round(&mut cos, Round::Down);
                let mut sin_lower = x.clone();
                sin_lower.sin_round(Round::Down);
                let mut cos_lower = x.clone();
                cos_lower.cos_round(Round::Down);
                assert!(same_float(&sin, &sin_lower) && same_float(&cos, &cos_lower));
                let width = Float::with_val(p, 1) >> ((index % 8) as u32 * (p / 8));
                for interval in [
                    MpfrInterval::point(x.clone()),
                    MpfrInterval {
                        upper: Float::with_val_round(p, &x + &width, Round::Up).0,
                        lower: x.clone(),
                    },
                ] {
                    let expected = (
                        reference_trig(&interval, true),
                        reference_trig(&interval, false),
                    );
                    let (sin, cos) = interval.sin_cos();
                    assert!(same_interval(&interval.sin(), &expected.0), "sin at {x}");
                    assert!(same_interval(&interval.cos(), &expected.1), "cos at {x}");
                    assert!(same_interval(&sin, &expected.0) && same_interval(&cos, &expected.1));
                }
            });
        }
        let wide = MpfrInterval {
            lower: Float::with_val(128, -3),
            upper: Float::with_val(128, 3),
        };
        let (sin, cos) = wide.sin_cos();
        assert!(same_interval(&sin, &reference_trig(&wide, true)));
        assert!(same_interval(&cos, &reference_trig(&wide, false)));
        let invalid = MpfrInterval::invalid(128);
        assert!(same_interval(
            &invalid.sin_cos().0,
            &reference_trig(&invalid, true)
        ));
    }

    #[test]
    fn invalid_complex_components_cannot_establish_zero_exclusion() {
        // Reproduce the invalid component produced by an infallible rational
        // constructor outside MPFR's finite range without allocating a giant
        // integer. Either invalid component invalidates the rectangle.
        let invalid = MpfrInterval::invalid(64);
        let positive = MpfrInterval::from_i64(1, 64);
        for (real, imaginary) in [(invalid.clone(), positive.clone()), (positive, invalid)] {
            let ball = MpfrComplexBall { real, imaginary };
            assert!(ball.validate().is_err());
            assert!(!ball.excludes_zero());
        }
    }

    #[test]
    fn integer_constructors_enclose_exact_inputs_at_every_supported_precision() {
        // Exact rational comparisons are independent of MPFR's rounded point.
        for precision in [2, 16, 32, 53, 63, 64, 96] {
            for value in [0_u64, 1, 13, (1_u64 << 32) + 1, u64::MAX] {
                let interval = MpfrInterval::from_u64(value, precision).to_rational_interval();
                assert!(
                    interval.contains(&Rational::from(value)),
                    "u64={value}, p={precision}"
                );
            }
            for value in [i64::MIN, i64::MIN + 1, -4_294_967_297, -13, 0, 13, i64::MAX] {
                let interval = MpfrInterval::from_i64(value, precision).to_rational_interval();
                assert!(
                    interval.contains(&Rational::from(value)),
                    "i64={value}, p={precision}"
                );
            }
        }
    }

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

#[cfg(test)]
mod exponent_range_tests {
    #[test]
    fn default_pools_share_the_caller_exponent_range() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| super::ensure_uniform_exponent_range().unwrap());
        }
    }
}
