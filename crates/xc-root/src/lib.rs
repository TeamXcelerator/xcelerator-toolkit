// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Reusable root discovery, refinement, and verification contracts.
//!
//! The f64 implementation is an explicit reference/discovery layer. Certified
//! interval Newton, Krawczyk, and contour-count backends remain separate
//! assurance milestones and may not be silently replaced by these routines.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use xc_certify::DecimalInterval;
use xc_core::CancellationToken;

#[cfg(feature = "hp")]
pub use xc_numerics::interval::{
    PolynomialContourCell, PolynomialContourCount, RationalContourRectangle,
    RationalFunctionContourCount,
};

/// Certified argument-principle count for an exact finite entire polynomial.
#[cfg(feature = "hp")]
pub fn certify_entire_polynomial_contour(
    coefficients_ascending: &[xc_numerics::interval::ComplexRational],
    rectangle: RationalContourRectangle,
    maximum_subdivision_depth: usize,
) -> Result<PolynomialContourCount, RootError> {
    xc_numerics::interval::certify_polynomial_zero_count_on_rectangle(
        coefficients_ascending,
        rectangle,
        maximum_subdivision_depth,
    )
    .map_err(|error| RootError::Evaluation(error.to_string()))
}

/// Certified argument-principle count for an exact meromorphic rational
/// function. Numerator zeros and denominator poles are enclosed and counted
/// separately before the signed count is formed.
#[cfg(feature = "hp")]
pub fn certify_meromorphic_rational_contour(
    numerator_ascending: &[xc_numerics::interval::ComplexRational],
    denominator_ascending: &[xc_numerics::interval::ComplexRational],
    rectangle: RationalContourRectangle,
    maximum_subdivision_depth: usize,
) -> Result<RationalFunctionContourCount, RootError> {
    xc_numerics::interval::certify_rational_function_argument_count_on_rectangle(
        numerator_ascending,
        denominator_ascending,
        rectangle,
        maximum_subdivision_depth,
    )
    .map_err(|error| RootError::Evaluation(error.to_string()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RootError {
    InvalidConfiguration(String),
    Evaluation(String),
    NoBracket(String),
    NonConvergence(String),
    PoleCollision(String),
    Cancelled(String),
}

impl Display for RootError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid root configuration: {message}")
            }
            Self::Evaluation(message) => write!(f, "function evaluation failed: {message}"),
            Self::NoBracket(message) => write!(f, "root is not bracketed: {message}"),
            Self::NonConvergence(message) => write!(f, "root solver did not converge: {message}"),
            Self::PoleCollision(message) => write!(f, "root step collided with a pole: {message}"),
            Self::Cancelled(message) => write!(f, "root operation cancelled: {message}"),
        }
    }
}

impl Error for RootError {}

pub trait RealFunctionF64: Send + Sync {
    fn evaluate(&self, x: f64) -> Result<f64, RootError>;

    fn derivative(&self, _x: f64) -> Result<f64, RootError> {
        Err(RootError::InvalidConfiguration(
            "derivative is not available".to_owned(),
        ))
    }
}

pub trait MeromorphicFunctionF64: RealFunctionF64 {
    /// Strictly increasing real poles relevant to the discovery window.
    fn real_poles(&self) -> &[f64];
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootBracketF64 {
    pub lower: f64,
    pub upper: f64,
}

impl RootBracketF64 {
    pub fn validate(&self) -> Result<(), RootError> {
        if !self.lower.is_finite() || !self.upper.is_finite() || self.lower >= self.upper {
            return Err(RootError::InvalidConfiguration(
                "root bracket must have finite lower < upper".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootApproximationStatus {
    Discovered,
    Refined,
    Approximate,
    CrossChecked,
    Certified,
    Duplicate,
    Unresolved,
    Inconclusive,
    TooCloseToPole,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootApproximationF64 {
    pub midpoint: f64,
    pub bracket: RootBracketF64,
    pub residual: f64,
    pub derivative_magnitude: Option<f64>,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub derivative_evaluations: usize,
    pub status: RootApproximationStatus,
    pub method: String,
}

impl RootApproximationF64 {
    pub fn decimal_interval(&self) -> DecimalInterval {
        DecimalInterval {
            lower: format!("{:.17e}", self.bracket.lower),
            upper: format!("{:.17e}", self.bracket.upper),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootStoppingF64 {
    pub absolute_x_tolerance: f64,
    pub relative_x_tolerance: f64,
    pub residual_tolerance: f64,
    pub maximum_iterations: usize,
}

impl Default for RootStoppingF64 {
    fn default() -> Self {
        Self {
            absolute_x_tolerance: 1e-14,
            relative_x_tolerance: 1e-14,
            residual_tolerance: 1e-14,
            maximum_iterations: 200,
        }
    }
}

impl RootStoppingF64 {
    pub fn validate(&self) -> Result<(), RootError> {
        for (name, value) in [
            ("absolute_x_tolerance", self.absolute_x_tolerance),
            ("relative_x_tolerance", self.relative_x_tolerance),
            ("residual_tolerance", self.residual_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(RootError::InvalidConfiguration(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        if self.maximum_iterations == 0 {
            return Err(RootError::InvalidConfiguration(
                "maximum_iterations must be positive".to_owned(),
            ));
        }
        Ok(())
    }

    fn x_converged(&self, lower: f64, upper: f64, midpoint: f64) -> bool {
        (upper - lower).abs()
            <= self
                .absolute_x_tolerance
                .max(self.relative_x_tolerance * midpoint.abs().max(1.0))
    }
}

fn checked_evaluate<F>(function: &F, x: f64, evaluations: &mut usize) -> Result<f64, RootError>
where
    F: RealFunctionF64 + ?Sized,
{
    *evaluations += 1;
    let value = function.evaluate(x)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RootError::Evaluation(format!(
            "function returned non-finite value at {x:e}"
        )))
    }
}

pub fn bisect_f64<F>(
    function: &F,
    bracket: RootBracketF64,
    stopping: &RootStoppingF64,
) -> Result<RootApproximationF64, RootError>
where
    F: RealFunctionF64 + ?Sized,
{
    bisect_f64_controlled(function, bracket, stopping, &CancellationToken::new())
}

pub fn bisect_f64_controlled<F>(
    function: &F,
    bracket: RootBracketF64,
    stopping: &RootStoppingF64,
    cancellation: &CancellationToken,
) -> Result<RootApproximationF64, RootError>
where
    F: RealFunctionF64 + ?Sized,
{
    check_cancellation(cancellation)?;
    bracket.validate()?;
    stopping.validate()?;
    let mut evaluations = 0usize;
    let mut lower = bracket.lower;
    let mut upper = bracket.upper;
    let mut f_lower = checked_evaluate(function, lower, &mut evaluations)?;
    let f_upper = checked_evaluate(function, upper, &mut evaluations)?;
    if f_lower == 0.0 {
        return Ok(RootApproximationF64 {
            midpoint: lower,
            bracket: RootBracketF64 {
                lower,
                upper: lower,
            },
            residual: 0.0,
            derivative_magnitude: None,
            iterations: 0,
            function_evaluations: evaluations,
            derivative_evaluations: 0,
            status: RootApproximationStatus::Refined,
            method: "bisection_f64".to_owned(),
        });
    }
    if f_upper == 0.0 {
        return Ok(RootApproximationF64 {
            midpoint: upper,
            bracket: RootBracketF64 {
                lower: upper,
                upper,
            },
            residual: 0.0,
            derivative_magnitude: None,
            iterations: 0,
            function_evaluations: evaluations,
            derivative_evaluations: 0,
            status: RootApproximationStatus::Refined,
            method: "bisection_f64".to_owned(),
        });
    }
    if f_lower.is_sign_positive() == f_upper.is_sign_positive() {
        return Err(RootError::NoBracket(format!(
            "endpoint signs agree on [{lower:e}, {upper:e}]"
        )));
    }

    for iteration in 1..=stopping.maximum_iterations {
        check_cancellation(cancellation)?;
        let midpoint = lower + 0.5 * (upper - lower);
        let f_midpoint = checked_evaluate(function, midpoint, &mut evaluations)?;
        if f_midpoint.abs() <= stopping.residual_tolerance
            || stopping.x_converged(lower, upper, midpoint)
        {
            let derivative = function.derivative(midpoint).ok();
            return Ok(RootApproximationF64 {
                midpoint,
                bracket: RootBracketF64 { lower, upper },
                residual: f_midpoint.abs(),
                derivative_magnitude: derivative.map(f64::abs),
                iterations: iteration,
                function_evaluations: evaluations,
                derivative_evaluations: usize::from(derivative.is_some()),
                status: RootApproximationStatus::Refined,
                method: "bisection_f64".to_owned(),
            });
        }
        if f_lower.is_sign_positive() != f_midpoint.is_sign_positive() {
            upper = midpoint;
        } else {
            lower = midpoint;
            f_lower = f_midpoint;
        }
    }
    Err(RootError::NonConvergence(format!(
        "bisection exceeded {} iterations",
        stopping.maximum_iterations
    )))
}

/// Refines one bracketed real root with Newton steps safeguarded by bisection.
///
/// # Mathematical semantics
/// The bracket must contain a sign change. Newton steps are accepted only when
/// safe; otherwise interval bisection preserves containment of a root.
///
/// # Precision
/// Evaluation, derivative evaluation, and stopping tests use binary64. This
/// entry point never promotes itself to an HP or interval-certified route.
///
/// # Failure states
/// Invalid brackets, non-finite evaluations, derivative failures, and exhausted
/// iteration limits return `RootError`; no unconverged midpoint is presented as
/// a successful root.
///
/// # Assurance and validity
/// Success is a finite binary64 approximation with residual and bracket-width
/// evidence. It is not an interval proof of uniqueness or a continuum claim.
///
/// # Cache effects
/// The refinement performs no cache reads, writes, or publication.
///
/// # Example
/// Compiled example: `crates/xc-root/examples/bracketed_root.rs`.
pub fn safeguarded_newton_f64(
    function: &dyn RealFunctionF64,
    bracket: RootBracketF64,
    initial: f64,
    stopping: &RootStoppingF64,
) -> Result<RootApproximationF64, RootError> {
    safeguarded_newton_f64_controlled(
        function,
        bracket,
        initial,
        stopping,
        &CancellationToken::new(),
    )
}

pub fn safeguarded_newton_f64_controlled(
    function: &dyn RealFunctionF64,
    bracket: RootBracketF64,
    initial: f64,
    stopping: &RootStoppingF64,
    cancellation: &CancellationToken,
) -> Result<RootApproximationF64, RootError> {
    check_cancellation(cancellation)?;
    bracket.validate()?;
    stopping.validate()?;
    if !initial.is_finite() || initial < bracket.lower || initial > bracket.upper {
        return Err(RootError::InvalidConfiguration(
            "initial Newton point must lie inside the bracket".to_owned(),
        ));
    }
    let mut function_evaluations = 0usize;
    let mut derivative_evaluations = 0usize;
    let mut lower = bracket.lower;
    let mut upper = bracket.upper;
    let mut f_lower = checked_evaluate(function, lower, &mut function_evaluations)?;
    let f_upper = checked_evaluate(function, upper, &mut function_evaluations)?;
    if f_lower.is_sign_positive() == f_upper.is_sign_positive() {
        return Err(RootError::NoBracket(
            "safeguarded Newton requires opposite endpoint signs".to_owned(),
        ));
    }
    let mut x = initial;
    for iteration in 1..=stopping.maximum_iterations {
        check_cancellation(cancellation)?;
        let fx = checked_evaluate(function, x, &mut function_evaluations)?;
        if fx.abs() <= stopping.residual_tolerance || stopping.x_converged(lower, upper, x) {
            let derivative = function.derivative(x).ok();
            derivative_evaluations += usize::from(derivative.is_some());
            return Ok(RootApproximationF64 {
                midpoint: x,
                bracket: RootBracketF64 { lower, upper },
                residual: fx.abs(),
                derivative_magnitude: derivative.map(f64::abs),
                iterations: iteration,
                function_evaluations,
                derivative_evaluations,
                status: RootApproximationStatus::Refined,
                method: "safeguarded_newton_f64".to_owned(),
            });
        }
        if f_lower.is_sign_positive() != fx.is_sign_positive() {
            upper = x;
        } else {
            lower = x;
            f_lower = fx;
        }
        derivative_evaluations += 1;
        let derivative = function.derivative(x)?;
        let candidate = if derivative != 0.0 && derivative.is_finite() {
            x - fx / derivative
        } else {
            f64::NAN
        };
        x = if candidate.is_finite() && candidate > lower && candidate < upper {
            candidate
        } else {
            lower + 0.5 * (upper - lower)
        };
    }
    Err(RootError::NonConvergence(format!(
        "safeguarded Newton exceeded {} iterations",
        stopping.maximum_iterations
    )))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoleAwareDiscoveryOptionsF64 {
    pub subdivisions_per_interval: usize,
    pub pole_margin_fraction: f64,
    pub stopping: RootStoppingF64,
    pub duplicate_tolerance: f64,
}

impl Default for PoleAwareDiscoveryOptionsF64 {
    fn default() -> Self {
        Self {
            subdivisions_per_interval: 16,
            pole_margin_fraction: 1e-10,
            stopping: RootStoppingF64::default(),
            duplicate_tolerance: 1e-12,
        }
    }
}

pub fn discover_pole_aware_sign_changes_f64(
    function: &dyn MeromorphicFunctionF64,
    lower: f64,
    upper: f64,
    options: &PoleAwareDiscoveryOptionsF64,
) -> Result<Vec<RootApproximationF64>, RootError> {
    discover_pole_aware_sign_changes_f64_controlled(
        function,
        lower,
        upper,
        options,
        &CancellationToken::new(),
    )
}

pub fn discover_pole_aware_sign_changes_f64_controlled(
    function: &dyn MeromorphicFunctionF64,
    lower: f64,
    upper: f64,
    options: &PoleAwareDiscoveryOptionsF64,
    cancellation: &CancellationToken,
) -> Result<Vec<RootApproximationF64>, RootError> {
    check_cancellation(cancellation)?;
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(RootError::InvalidConfiguration(
            "discovery range must have finite lower < upper".to_owned(),
        ));
    }
    if options.subdivisions_per_interval == 0
        || !(0.0..0.1).contains(&options.pole_margin_fraction)
        || !options.duplicate_tolerance.is_finite()
        || options.duplicate_tolerance <= 0.0
    {
        return Err(RootError::InvalidConfiguration(
            "invalid pole-aware discovery options".to_owned(),
        ));
    }
    options.stopping.validate()?;
    let poles = function.real_poles();
    if poles.iter().any(|value| !value.is_finite())
        || poles.windows(2).any(|window| window[0] >= window[1])
    {
        return Err(RootError::InvalidConfiguration(
            "real poles must be finite and strictly increasing".to_owned(),
        ));
    }
    let mut boundaries = vec![lower];
    boundaries.extend(
        poles
            .iter()
            .copied()
            .filter(|pole| *pole > lower && *pole < upper),
    );
    boundaries.push(upper);

    let mut roots = Vec::new();
    for window in boundaries.windows(2) {
        check_cancellation(cancellation)?;
        let width = window[1] - window[0];
        let margin = (options.pole_margin_fraction * width)
            .max(f64::EPSILON * window[0].abs().max(window[1].abs()).max(1.0));
        let interval_lower = window[0] + margin;
        let interval_upper = window[1] - margin;
        if interval_lower >= interval_upper {
            continue;
        }
        let mut x0 = interval_lower;
        let mut f0 = function.evaluate(x0)?;
        for step in 1..=options.subdivisions_per_interval {
            check_cancellation(cancellation)?;
            let x1 = interval_lower
                + (interval_upper - interval_lower) * step as f64
                    / options.subdivisions_per_interval as f64;
            let f1 = function.evaluate(x1)?;
            if f0 == 0.0 || f1 == 0.0 || f0.is_sign_positive() != f1.is_sign_positive() {
                let bracket = RootBracketF64 {
                    lower: x0,
                    upper: x1,
                };
                let root =
                    bisect_f64_controlled(function, bracket, &options.stopping, cancellation)?;
                if roots.iter().all(|prior: &RootApproximationF64| {
                    (prior.midpoint - root.midpoint).abs() > options.duplicate_tolerance
                }) {
                    roots.push(root);
                }
            }
            x0 = x1;
            f0 = f1;
        }
    }
    roots.sort_by(|left, right| left.midpoint.total_cmp(&right.midpoint));
    Ok(roots)
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<(), RootError> {
    cancellation
        .check()
        .map_err(|error| RootError::Cancelled(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_core::CancellationReason;

    struct Cubic;

    impl RealFunctionF64 for Cubic {
        fn evaluate(&self, x: f64) -> Result<f64, RootError> {
            Ok(x * x * x - 2.0)
        }

        fn derivative(&self, x: f64) -> Result<f64, RootError> {
            Ok(3.0 * x * x)
        }
    }

    struct SimpleMeromorphic {
        poles: Vec<f64>,
    }

    impl RealFunctionF64 for SimpleMeromorphic {
        fn evaluate(&self, x: f64) -> Result<f64, RootError> {
            Ok(1.0 / (x + 1.0) + 1.0 / (x - 1.0))
        }

        fn derivative(&self, x: f64) -> Result<f64, RootError> {
            Ok(-1.0 / (x + 1.0).powi(2) - 1.0 / (x - 1.0).powi(2))
        }
    }

    impl MeromorphicFunctionF64 for SimpleMeromorphic {
        fn real_poles(&self) -> &[f64] {
            &self.poles
        }
    }

    #[test]
    fn safeguarded_newton_refines_bracketed_root() {
        let result = safeguarded_newton_f64(
            &Cubic,
            RootBracketF64 {
                lower: 1.0,
                upper: 2.0,
            },
            1.5,
            &RootStoppingF64::default(),
        )
        .unwrap();
        assert!((result.midpoint - 2.0_f64.cbrt()).abs() < 1e-12);
    }

    #[test]
    fn pole_aware_discovery_does_not_cross_poles() {
        let function = SimpleMeromorphic {
            poles: vec![-1.0, 1.0],
        };
        let roots = discover_pole_aware_sign_changes_f64(
            &function,
            -0.9,
            0.9,
            &PoleAwareDiscoveryOptionsF64::default(),
        )
        .unwrap();
        assert_eq!(roots.len(), 1);
        assert!(roots[0].midpoint.abs() < 1e-12);
    }

    #[test]
    fn controlled_refinement_honors_cancellation_before_evaluation() {
        let cancellation = CancellationToken::new();
        cancellation.cancel(CancellationReason::UserRequested);
        let result = bisect_f64_controlled(
            &Cubic,
            RootBracketF64 {
                lower: 1.0,
                upper: 2.0,
            },
            &RootStoppingF64::default(),
            &cancellation,
        );
        assert!(matches!(result, Err(RootError::Cancelled(_))));
    }

    #[test]
    fn refinement_rejects_degenerate_interval_and_zero_iterations() {
        let degenerate = bisect_f64(
            &Cubic,
            RootBracketF64 {
                lower: 1.0,
                upper: 1.0,
            },
            &RootStoppingF64::default(),
        );
        assert!(matches!(
            degenerate,
            Err(RootError::InvalidConfiguration(_))
        ));

        let stopping = RootStoppingF64 {
            maximum_iterations: 0,
            ..RootStoppingF64::default()
        };
        let zero_iterations = bisect_f64(
            &Cubic,
            RootBracketF64 {
                lower: 1.0,
                upper: 2.0,
            },
            &stopping,
        );
        assert!(matches!(
            zero_iterations,
            Err(RootError::InvalidConfiguration(_))
        ));
    }
}

// ===========================================================================
// Arbitrary-precision point root refinement
// ===========================================================================

#[cfg(feature = "hp")]
pub trait RealFunctionHp: Send + Sync {
    fn evaluate(&self, x: &rug::Float, precision_bits: u32) -> Result<rug::Float, RootError>;

    fn derivative(&self, _x: &rug::Float, _precision_bits: u32) -> Result<rug::Float, RootError> {
        Err(RootError::InvalidConfiguration(
            "HP derivative is not available".to_owned(),
        ))
    }
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootStoppingHp {
    pub x_tolerance: xc_core::DecimalLiteral,
    pub residual_tolerance: xc_core::DecimalLiteral,
    pub maximum_iterations: usize,
}

#[cfg(feature = "hp")]
impl RootStoppingHp {
    pub fn validate(&self) -> Result<(), RootError> {
        let zero = xc_core::DecimalLiteral::new("0")
            .map_err(|error| RootError::InvalidConfiguration(error.to_string()))?;
        for (name, value) in [
            ("x_tolerance", &self.x_tolerance),
            ("residual_tolerance", &self.residual_tolerance),
        ] {
            value
                .validate()
                .map_err(|error| RootError::InvalidConfiguration(error.to_string()))?;
            if value
                .cmp_numeric(&zero)
                .map_err(|error| RootError::InvalidConfiguration(error.to_string()))?
                != std::cmp::Ordering::Greater
            {
                return Err(RootError::InvalidConfiguration(format!(
                    "{name} must be strictly positive"
                )));
            }
        }
        if self.maximum_iterations == 0 {
            return Err(RootError::InvalidConfiguration(
                "maximum_iterations must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootApproximationHp {
    pub midpoint: String,
    pub lower: String,
    pub upper: String,
    pub residual: String,
    pub derivative_magnitude: Option<String>,
    pub precision_bits: u32,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub derivative_evaluations: usize,
    pub status: RootApproximationStatus,
    pub method: String,
}

#[cfg(feature = "hp")]
fn parse_hp_decimal(
    value: &xc_core::DecimalLiteral,
    precision_bits: u32,
) -> Result<rug::Float, RootError> {
    let parsed = rug::Float::parse(value.as_str()).map_err(|error| {
        RootError::InvalidConfiguration(format!(
            "failed to parse HP decimal {:?}: {error}",
            value.as_str()
        ))
    })?;
    Ok(rug::Float::with_val(precision_bits, parsed))
}

#[cfg(feature = "hp")]
fn hp_string(value: &rug::Float, precision_bits: u32) -> String {
    // Exact-round-trip width. The bare ceiling of bits*log10(2) used before
    // v0.14.0 was one digit short, so a persisted root decoded up to one ulp
    // away from the computed root and replay validation of the residual
    // failed on every reuse.
    let digits = xc_numerics::reduction::roundtrip_decimal_digits(precision_bits).max(32);
    value.to_string_radix(10, Some(digits))
}

#[cfg(feature = "hp")]
fn hp_same_nonzero_sign(left: &rug::Float, right: &rug::Float) -> bool {
    !left.is_zero() && !right.is_zero() && left.is_sign_negative() == right.is_sign_negative()
}

#[cfg(feature = "hp")]
pub fn bisect_hp(
    function: &dyn RealFunctionHp,
    lower: &rug::Float,
    upper: &rug::Float,
    precision_bits: u32,
    stopping: &RootStoppingHp,
) -> Result<RootApproximationHp, RootError> {
    bisect_hp_controlled(
        function,
        lower,
        upper,
        precision_bits,
        stopping,
        &CancellationToken::new(),
    )
}

#[cfg(feature = "hp")]
pub fn bisect_hp_controlled(
    function: &dyn RealFunctionHp,
    lower: &rug::Float,
    upper: &rug::Float,
    precision_bits: u32,
    stopping: &RootStoppingHp,
    cancellation: &CancellationToken,
) -> Result<RootApproximationHp, RootError> {
    use rug::Float;
    check_cancellation(cancellation)?;
    stopping.validate()?;
    if precision_bits < 32 || lower >= upper {
        return Err(RootError::InvalidConfiguration(
            "HP bisection requires precision >= 32 and lower < upper".to_owned(),
        ));
    }
    let x_tolerance = parse_hp_decimal(&stopping.x_tolerance, precision_bits)?;
    let residual_tolerance = parse_hp_decimal(&stopping.residual_tolerance, precision_bits)?;
    let mut function_evaluations = 0usize;
    let mut lower = Float::with_val(precision_bits, lower);
    let mut upper = Float::with_val(precision_bits, upper);
    let mut f_lower = function.evaluate(&lower, precision_bits)?;
    function_evaluations += 1;
    let f_upper = function.evaluate(&upper, precision_bits)?;
    function_evaluations += 1;
    if f_lower == 0 {
        return Ok(RootApproximationHp {
            midpoint: hp_string(&lower, precision_bits),
            lower: hp_string(&lower, precision_bits),
            upper: hp_string(&lower, precision_bits),
            residual: "0".to_owned(),
            derivative_magnitude: None,
            precision_bits,
            iterations: 0,
            function_evaluations,
            derivative_evaluations: 0,
            status: RootApproximationStatus::Refined,
            method: "bisection_hp".to_owned(),
        });
    }
    if f_upper == 0 {
        return Ok(RootApproximationHp {
            midpoint: hp_string(&upper, precision_bits),
            lower: hp_string(&upper, precision_bits),
            upper: hp_string(&upper, precision_bits),
            residual: "0".to_owned(),
            derivative_magnitude: None,
            precision_bits,
            iterations: 0,
            function_evaluations,
            derivative_evaluations: 0,
            status: RootApproximationStatus::Refined,
            method: "bisection_hp".to_owned(),
        });
    }
    if hp_same_nonzero_sign(&f_lower, &f_upper) {
        return Err(RootError::NoBracket(
            "HP bisection endpoint signs agree".to_owned(),
        ));
    }

    for iteration in 1..=stopping.maximum_iterations {
        check_cancellation(cancellation)?;
        let mut midpoint = lower.clone();
        midpoint += &upper;
        midpoint /= 2;
        let f_midpoint = function.evaluate(&midpoint, precision_bits)?;
        function_evaluations += 1;
        let mut residual = f_midpoint.clone();
        residual.abs_mut();
        let mut width = upper.clone();
        width -= &lower;
        width.abs_mut();
        if residual <= residual_tolerance || width <= x_tolerance {
            let derivative = function.derivative(&midpoint, precision_bits).ok();
            let derivative_magnitude = derivative.as_ref().map(|value| {
                let mut magnitude = value.clone();
                magnitude.abs_mut();
                hp_string(&magnitude, precision_bits)
            });
            return Ok(RootApproximationHp {
                midpoint: hp_string(&midpoint, precision_bits),
                lower: hp_string(&lower, precision_bits),
                upper: hp_string(&upper, precision_bits),
                residual: hp_string(&residual, precision_bits),
                derivative_magnitude,
                precision_bits,
                iterations: iteration,
                function_evaluations,
                derivative_evaluations: usize::from(derivative.is_some()),
                status: RootApproximationStatus::Refined,
                method: "bisection_hp".to_owned(),
            });
        }
        if !hp_same_nonzero_sign(&f_lower, &f_midpoint) {
            upper = midpoint;
        } else {
            lower = midpoint;
            f_lower = f_midpoint;
        }
    }
    Err(RootError::NonConvergence(format!(
        "HP bisection exceeded {} iterations",
        stopping.maximum_iterations
    )))
}

#[cfg(feature = "hp")]
pub fn safeguarded_newton_hp(
    function: &dyn RealFunctionHp,
    lower: &rug::Float,
    upper: &rug::Float,
    initial: &rug::Float,
    precision_bits: u32,
    stopping: &RootStoppingHp,
) -> Result<RootApproximationHp, RootError> {
    safeguarded_newton_hp_controlled(
        function,
        lower,
        upper,
        initial,
        precision_bits,
        stopping,
        &CancellationToken::new(),
    )
}

#[cfg(feature = "hp")]
pub fn safeguarded_newton_hp_controlled(
    function: &dyn RealFunctionHp,
    lower: &rug::Float,
    upper: &rug::Float,
    initial: &rug::Float,
    precision_bits: u32,
    stopping: &RootStoppingHp,
    cancellation: &CancellationToken,
) -> Result<RootApproximationHp, RootError> {
    use rug::Float;
    check_cancellation(cancellation)?;
    stopping.validate()?;
    if precision_bits < 32 || lower >= upper || initial < lower || initial > upper {
        return Err(RootError::InvalidConfiguration(
            "HP safeguarded Newton requires precision >= 32 and an initial point inside lower < upper"
                .to_owned(),
        ));
    }
    let x_tolerance = parse_hp_decimal(&stopping.x_tolerance, precision_bits)?;
    let residual_tolerance = parse_hp_decimal(&stopping.residual_tolerance, precision_bits)?;
    let mut lower = Float::with_val(precision_bits, lower);
    let mut upper = Float::with_val(precision_bits, upper);
    let mut x = Float::with_val(precision_bits, initial);
    let mut function_evaluations = 0usize;
    let mut derivative_evaluations = 0usize;
    let mut f_lower = function.evaluate(&lower, precision_bits)?;
    function_evaluations += 1;
    let f_upper = function.evaluate(&upper, precision_bits)?;
    function_evaluations += 1;
    if hp_same_nonzero_sign(&f_lower, &f_upper) {
        return Err(RootError::NoBracket(
            "HP safeguarded Newton endpoint signs agree".to_owned(),
        ));
    }

    for iteration in 1..=stopping.maximum_iterations {
        check_cancellation(cancellation)?;
        let fx = function.evaluate(&x, precision_bits)?;
        function_evaluations += 1;
        let mut residual = fx.clone();
        residual.abs_mut();
        let mut width = upper.clone();
        width -= &lower;
        width.abs_mut();
        if residual <= residual_tolerance || width <= x_tolerance {
            let derivative = function.derivative(&x, precision_bits).ok();
            derivative_evaluations += usize::from(derivative.is_some());
            let derivative_magnitude = derivative.as_ref().map(|value| {
                let mut magnitude = value.clone();
                magnitude.abs_mut();
                hp_string(&magnitude, precision_bits)
            });
            return Ok(RootApproximationHp {
                midpoint: hp_string(&x, precision_bits),
                lower: hp_string(&lower, precision_bits),
                upper: hp_string(&upper, precision_bits),
                residual: hp_string(&residual, precision_bits),
                derivative_magnitude,
                precision_bits,
                iterations: iteration,
                function_evaluations,
                derivative_evaluations,
                status: RootApproximationStatus::Refined,
                method: "safeguarded_newton_hp".to_owned(),
            });
        }
        if !hp_same_nonzero_sign(&f_lower, &fx) {
            upper = x.clone();
        } else {
            lower = x.clone();
            f_lower = fx.clone();
        }
        let derivative = function.derivative(&x, precision_bits)?;
        derivative_evaluations += 1;
        let mut candidate = x.clone();
        let valid_derivative = derivative != 0;
        if valid_derivative {
            let mut step = fx;
            step /= derivative;
            candidate -= step;
        }
        if !valid_derivative || candidate <= lower || candidate >= upper {
            candidate = lower.clone();
            candidate += &upper;
            candidate /= 2;
        }
        x = candidate;
    }
    Err(RootError::NonConvergence(format!(
        "HP safeguarded Newton exceeded {} iterations",
        stopping.maximum_iterations
    )))
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HpRootCrossCheck {
    pub bisection: RootApproximationHp,
    pub safeguarded_newton: RootApproximationHp,
    pub agreement_tolerance: String,
    pub observed_midpoint_difference: String,
    pub accepted: bool,
    pub independence_established: bool,
    pub independence_rationale: String,
}

#[cfg(feature = "hp")]
fn parse_hp_output(value: &str, precision_bits: u32) -> Result<rug::Float, RootError> {
    let parsed = rug::Float::parse(value).map_err(|error| {
        RootError::Evaluation(format!("failed to parse HP root output {value:?}: {error}"))
    })?;
    Ok(rug::Float::with_val(precision_bits, parsed))
}

/// Cross-check one simple bracketed root through two independently executed
/// HP refinement algorithms. Bisection uses only ordered signs; safeguarded
/// Newton separately evaluates derivatives and may fall back to its own
/// bisection step. Neither route receives the other route's result as a seed,
/// stopping decision, or intermediate value.
#[cfg(feature = "hp")]
pub fn cross_check_simple_root_hp(
    function: &dyn RealFunctionHp,
    lower: &rug::Float,
    upper: &rug::Float,
    initial: &rug::Float,
    precision_bits: u32,
    stopping: &RootStoppingHp,
    agreement_tolerance: &xc_core::DecimalLiteral,
) -> Result<HpRootCrossCheck, RootError> {
    let tolerance = parse_hp_decimal(agreement_tolerance, precision_bits)?;
    if tolerance <= 0 {
        return Err(RootError::InvalidConfiguration(
            "HP root cross-check agreement tolerance must be positive".to_owned(),
        ));
    }
    let bisection = bisect_hp(function, lower, upper, precision_bits, stopping)?;
    let safeguarded_newton =
        safeguarded_newton_hp(function, lower, upper, initial, precision_bits, stopping)?;
    let bisection_midpoint = parse_hp_output(&bisection.midpoint, precision_bits)?;
    let newton_midpoint = parse_hp_output(&safeguarded_newton.midpoint, precision_bits)?;
    let mut difference = bisection_midpoint;
    difference -= newton_midpoint;
    difference.abs_mut();
    let independence_established = safeguarded_newton.derivative_evaluations > 0;
    let accepted = difference <= tolerance && independence_established;
    Ok(HpRootCrossCheck {
        bisection,
        safeguarded_newton,
        agreement_tolerance: hp_string(&tolerance, precision_bits),
        observed_midpoint_difference: hp_string(&difference, precision_bits),
        accepted,
        independence_established,
        independence_rationale: "HP sign-only bisection and derivative-based safeguarded Newton execute independently from the original bracket and share no decisive iterates"
            .to_owned(),
    })
}

#[cfg(feature = "hp")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HpRefinementMethod {
    Bisection,
    SafeguardedNewton,
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HpRootRefinementTask {
    pub task_id: String,
    pub lower: xc_core::DecimalLiteral,
    pub upper: xc_core::DecimalLiteral,
    pub initial: Option<xc_core::DecimalLiteral>,
    pub precision_bits: u32,
    pub stopping: RootStoppingHp,
    pub method: HpRefinementMethod,
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HpRootRefinementOutcome {
    pub input_ordinal: usize,
    pub task_id: String,
    pub requested_precision_bits: u32,
    pub approximation: RootApproximationHp,
}

/// Refine independent HP roots in parallel while returning results in exact
/// input order. Each task owns its precision and stopping policy; difficult
/// roots therefore do not raise their neighbors' precision. Errors are also
/// resolved in input order, independent of worker completion order.
#[cfg(feature = "hp")]
pub fn refine_roots_parallel_hp(
    function: &dyn RealFunctionHp,
    tasks: &[HpRootRefinementTask],
    cancellation: &CancellationToken,
) -> Result<Vec<HpRootRefinementOutcome>, RootError> {
    use rayon::prelude::*;
    use std::collections::BTreeSet;

    if tasks.is_empty() {
        return Err(RootError::InvalidConfiguration(
            "parallel HP root refinement requires at least one task".to_owned(),
        ));
    }
    let mut identifiers = BTreeSet::new();
    for task in tasks {
        if task.task_id.trim().is_empty() || !identifiers.insert(task.task_id.clone()) {
            return Err(RootError::InvalidConfiguration(
                "parallel HP root task identifiers must be nonempty and unique".to_owned(),
            ));
        }
        task.stopping.validate()?;
    }
    check_cancellation(cancellation)?;
    let attempts = tasks
        .par_iter()
        .enumerate()
        .map(|(input_ordinal, task)| {
            check_cancellation(cancellation)?;
            let lower = parse_hp_decimal(&task.lower, task.precision_bits)?;
            let upper = parse_hp_decimal(&task.upper, task.precision_bits)?;
            let approximation = match task.method {
                HpRefinementMethod::Bisection => bisect_hp_controlled(
                    function,
                    &lower,
                    &upper,
                    task.precision_bits,
                    &task.stopping,
                    cancellation,
                )?,
                HpRefinementMethod::SafeguardedNewton => {
                    let initial = task.initial.as_ref().ok_or_else(|| {
                        RootError::InvalidConfiguration(format!(
                            "parallel HP Newton task {:?} lacks an initial point",
                            task.task_id
                        ))
                    })?;
                    let initial = parse_hp_decimal(initial, task.precision_bits)?;
                    safeguarded_newton_hp_controlled(
                        function,
                        &lower,
                        &upper,
                        &initial,
                        task.precision_bits,
                        &task.stopping,
                        cancellation,
                    )?
                }
            };
            Ok(HpRootRefinementOutcome {
                input_ordinal,
                task_id: task.task_id.clone(),
                requested_precision_bits: task.precision_bits,
                approximation,
            })
        })
        .collect::<Vec<Result<HpRootRefinementOutcome, RootError>>>();
    attempts.into_iter().collect()
}

// ===========================================================================
// Certified arbitrary-precision interval Newton isolation
// ===========================================================================

#[cfg(feature = "hp")]
pub trait RealIntervalFunctionHp: Send + Sync {
    fn evaluate_interval(
        &self,
        argument: &xc_numerics::mpfr_interval::MpfrInterval,
    ) -> Result<xc_numerics::mpfr_interval::MpfrInterval, RootError>;

    fn derivative_interval(
        &self,
        argument: &xc_numerics::mpfr_interval::MpfrInterval,
    ) -> Result<xc_numerics::mpfr_interval::MpfrInterval, RootError>;
}

#[cfg(feature = "hp")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalRootStatus {
    CertifiedUnique,
    ExcludedNoRoot,
    Inconclusive,
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntervalRootCertificate {
    pub lower: String,
    pub upper: String,
    pub precision_bits: u32,
    pub iterations: usize,
    pub status: IntervalRootStatus,
    pub uniqueness_witnessed: bool,
    pub reason: String,
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntervalNewtonOptions {
    pub width_tolerance: xc_core::DecimalLiteral,
    pub maximum_iterations: usize,
}

#[cfg(feature = "hp")]
impl IntervalNewtonOptions {
    pub fn validate(&self, precision_bits: u32) -> Result<rug::Float, RootError> {
        if self.maximum_iterations == 0 {
            return Err(RootError::InvalidConfiguration(
                "interval Newton maximum_iterations must be positive".to_owned(),
            ));
        }
        let tolerance = parse_hp_decimal(&self.width_tolerance, precision_bits)?;
        if tolerance <= 0 {
            return Err(RootError::InvalidConfiguration(
                "interval Newton width tolerance must be positive".to_owned(),
            ));
        }
        Ok(tolerance)
    }
}

#[cfg(feature = "hp")]
fn interval_root_certificate(
    interval: &xc_numerics::mpfr_interval::MpfrInterval,
    iterations: usize,
    status: IntervalRootStatus,
    uniqueness_witnessed: bool,
    reason: impl Into<String>,
) -> IntervalRootCertificate {
    IntervalRootCertificate {
        lower: hp_string(interval.lower(), interval.precision()),
        upper: hp_string(interval.upper(), interval.precision()),
        precision_bits: interval.precision(),
        iterations,
        status,
        uniqueness_witnessed,
        reason: reason.into(),
    }
}

#[cfg(feature = "hp")]
pub fn interval_newton_hp(
    function: &dyn RealIntervalFunctionHp,
    initial: &xc_numerics::mpfr_interval::MpfrInterval,
    options: &IntervalNewtonOptions,
) -> Result<IntervalRootCertificate, RootError> {
    let tolerance = options.validate(initial.precision())?;
    let mut current = initial.clone();
    let mut uniqueness_witnessed = false;
    for iteration in 1..=options.maximum_iterations {
        let derivative = function.derivative_interval(&current)?;
        if derivative.contains_zero() {
            return Ok(interval_root_certificate(
                &current,
                iteration,
                IntervalRootStatus::Inconclusive,
                uniqueness_witnessed,
                "derivative enclosure contains zero",
            ));
        }
        let midpoint = current.midpoint_point();
        let midpoint_value = function.evaluate_interval(&midpoint)?;
        let newton_image = midpoint.sub(
            &midpoint_value
                .div(&derivative)
                .map_err(|error| RootError::Evaluation(error.to_string()))?,
        );
        if newton_image.is_interior_subset_of(&current) {
            uniqueness_witnessed = true;
        }
        let Some(next) = current.intersection(&newton_image) else {
            return Ok(interval_root_certificate(
                &current,
                iteration,
                IntervalRootStatus::ExcludedNoRoot,
                false,
                "interval Newton image is disjoint from the candidate interval",
            ));
        };
        current = next;
        if uniqueness_witnessed && current.width() <= tolerance {
            return Ok(interval_root_certificate(
                &current,
                iteration,
                IntervalRootStatus::CertifiedUnique,
                true,
                "interval Newton image lies strictly inside the candidate interval",
            ));
        }
    }
    Ok(interval_root_certificate(
        &current,
        options.maximum_iterations,
        IntervalRootStatus::Inconclusive,
        uniqueness_witnessed,
        "interval Newton iteration limit reached before the width target",
    ))
}

#[cfg(all(test, feature = "hp"))]
mod hp_tests {
    use super::*;
    use rug::{Float, Rational};

    /// Every persisted decimal scalar must decode to the exact bits it was
    /// printed from. The pre-v0.14.0 width, `ceil(bits * log10(2))` with no
    /// guard digit, is one digit short of unique recovery: a stored root
    /// decoded up to one ulp away from the computed root, and cancellation-
    /// dominated replay validation then failed on every cache reuse.
    #[test]
    fn hp_string_round_trips_exactly_at_claim_precision() {
        let prec = 3386_u32; // HP-1000: the precision of the paper's claims
        let old_width = ((f64::from(prec)) * std::f64::consts::LOG10_2).ceil() as usize;

        let mut old_width_failures = 0_usize;
        for k in 2_u32..202 {
            // Deterministic full-mantissa values across wildly different
            // scales, including the e-994 residual regime where the defect
            // was observed.
            let base = Float::with_val(prec, k).sqrt();
            for scale in [-994_i32, -59, 0, 300] {
                let value = base.clone() * Float::with_val(prec, Float::i_exp(1, scale * 10 / 3));
                let encoded = hp_string(&value, prec);
                let decoded = Float::with_val(
                    prec,
                    Float::parse(&encoded).expect("hp_string output must parse"),
                );
                assert_eq!(
                    decoded,
                    value,
                    "hp_string lost bits for sqrt({k}) at scale 2^{}",
                    scale * 10 / 3
                );

                let truncated = value.to_string_radix(10, Some(old_width));
                let decoded_old = Float::with_val(prec, Float::parse(&truncated).expect("parses"));
                if decoded_old != value {
                    old_width_failures += 1;
                }
            }
        }
        // The old width must demonstrably lose values, or this test would
        // not have caught the defect it documents.
        assert!(
            old_width_failures > 0,
            "the pre-fix width unexpectedly round-tripped every sample"
        );
    }

    struct SqrtTwo;

    impl RealIntervalFunctionHp for SqrtTwo {
        fn evaluate_interval(
            &self,
            x: &xc_numerics::mpfr_interval::MpfrInterval,
        ) -> Result<xc_numerics::mpfr_interval::MpfrInterval, RootError> {
            Ok(x.square()
                .sub(&xc_numerics::mpfr_interval::MpfrInterval::from_i64(
                    2,
                    x.precision(),
                )))
        }

        fn derivative_interval(
            &self,
            x: &xc_numerics::mpfr_interval::MpfrInterval,
        ) -> Result<xc_numerics::mpfr_interval::MpfrInterval, RootError> {
            Ok(x.mul(&xc_numerics::mpfr_interval::MpfrInterval::from_i64(
                2,
                x.precision(),
            )))
        }
    }

    impl RealFunctionHp for SqrtTwo {
        fn evaluate(&self, x: &Float, precision_bits: u32) -> Result<Float, RootError> {
            let mut value = Float::with_val(precision_bits, x);
            value *= x;
            value -= 2;
            Ok(value)
        }

        fn derivative(&self, x: &Float, precision_bits: u32) -> Result<Float, RootError> {
            let mut value = Float::with_val(precision_bits, x);
            value *= 2;
            Ok(value)
        }
    }

    #[test]
    fn hp_newton_resolves_deep_tolerance_without_f64() {
        let precision = 512;
        let stopping = RootStoppingHp {
            x_tolerance: xc_core::DecimalLiteral::new("1e-100").unwrap(),
            residual_tolerance: xc_core::DecimalLiteral::new("1e-100").unwrap(),
            maximum_iterations: 300,
        };
        let result = safeguarded_newton_hp(
            &SqrtTwo,
            &Float::with_val(precision, 1),
            &Float::with_val(precision, 2),
            &Float::with_val(precision, 1.5),
            precision,
            &stopping,
        )
        .unwrap();
        assert!(result.residual.starts_with("0") || result.residual.contains("e-"));
    }

    #[test]
    fn hp_bisection_and_newton_cross_check_without_shared_iterates() {
        let precision = 256;
        let stopping = RootStoppingHp {
            x_tolerance: xc_core::DecimalLiteral::new("1e-55").unwrap(),
            residual_tolerance: xc_core::DecimalLiteral::new("1e-55").unwrap(),
            maximum_iterations: 400,
        };
        let report = cross_check_simple_root_hp(
            &SqrtTwo,
            &Float::with_val(precision, 1),
            &Float::with_val(precision, 2),
            &Float::with_val(precision, 1.5),
            precision,
            &stopping,
            &xc_core::DecimalLiteral::new("1e-50").unwrap(),
        )
        .unwrap();
        assert!(report.accepted, "{:?}", report);
        assert!(report.independence_established);
        assert_eq!(report.bisection.method, "bisection_hp");
        assert_eq!(report.safeguarded_newton.method, "safeguarded_newton_hp");
        assert!(report.independence_rationale.contains("share no decisive"));
    }

    #[test]
    fn interval_newton_certifies_unique_root_without_point_sign_guesses() {
        use xc_numerics::mpfr_interval::MpfrInterval;

        let precision = 192;
        let initial =
            MpfrInterval::new(Float::with_val(precision, 1), Float::with_val(precision, 2))
                .unwrap();
        let certificate = interval_newton_hp(
            &SqrtTwo,
            &initial,
            &IntervalNewtonOptions {
                width_tolerance: xc_core::DecimalLiteral::new("1e-40").unwrap(),
                maximum_iterations: 20,
            },
        )
        .unwrap();
        assert_eq!(certificate.status, IntervalRootStatus::CertifiedUnique);
        assert!(certificate.uniqueness_witnessed);
    }

    struct ShiftedLinear;

    impl RealIntervalFunctionHp for ShiftedLinear {
        fn evaluate_interval(
            &self,
            x: &xc_numerics::mpfr_interval::MpfrInterval,
        ) -> Result<xc_numerics::mpfr_interval::MpfrInterval, RootError> {
            Ok(x.add(&xc_numerics::mpfr_interval::MpfrInterval::from_i64(
                2,
                x.precision(),
            )))
        }

        fn derivative_interval(
            &self,
            x: &xc_numerics::mpfr_interval::MpfrInterval,
        ) -> Result<xc_numerics::mpfr_interval::MpfrInterval, RootError> {
            Ok(xc_numerics::mpfr_interval::MpfrInterval::from_i64(
                1,
                x.precision(),
            ))
        }
    }

    #[test]
    fn interval_newton_excludes_rootless_candidate() {
        use xc_numerics::mpfr_interval::MpfrInterval;

        let precision = 128;
        let initial =
            MpfrInterval::new(Float::with_val(precision, 0), Float::with_val(precision, 1))
                .unwrap();
        let certificate = interval_newton_hp(
            &ShiftedLinear,
            &initial,
            &IntervalNewtonOptions {
                width_tolerance: xc_core::DecimalLiteral::new("1e-20").unwrap(),
                maximum_iterations: 4,
            },
        )
        .unwrap();
        assert_eq!(certificate.status, IntervalRootStatus::ExcludedNoRoot);
    }

    struct CubicRoots;

    impl RealFunctionHp for CubicRoots {
        fn evaluate(&self, x: &Float, precision_bits: u32) -> Result<Float, RootError> {
            let mut square = Float::with_val(precision_bits, x);
            square *= x;
            square -= 1;
            square *= x;
            Ok(square)
        }

        fn derivative(&self, x: &Float, precision_bits: u32) -> Result<Float, RootError> {
            let mut derivative = Float::with_val(precision_bits, x);
            derivative *= x;
            derivative *= 3;
            derivative -= 1;
            Ok(derivative)
        }
    }

    #[test]
    fn parallel_hp_batch_preserves_input_order_and_per_root_precision() {
        let decimal = |value| xc_core::DecimalLiteral::new(value).unwrap();
        let stopping = RootStoppingHp {
            x_tolerance: decimal("1e-30"),
            residual_tolerance: decimal("1e-30"),
            maximum_iterations: 300,
        };
        let tasks = vec![
            HpRootRefinementTask {
                task_id: "right".to_owned(),
                lower: decimal("0.5"),
                upper: decimal("1.5"),
                initial: Some(decimal("0.8")),
                precision_bits: 192,
                stopping: stopping.clone(),
                method: HpRefinementMethod::SafeguardedNewton,
            },
            HpRootRefinementTask {
                task_id: "left".to_owned(),
                lower: decimal("-1.5"),
                upper: decimal("-0.5"),
                initial: None,
                precision_bits: 128,
                stopping: stopping.clone(),
                method: HpRefinementMethod::Bisection,
            },
            HpRootRefinementTask {
                task_id: "center".to_owned(),
                lower: decimal("-0.5"),
                upper: decimal("0.5"),
                initial: Some(decimal("0.1")),
                precision_bits: 256,
                stopping,
                method: HpRefinementMethod::SafeguardedNewton,
            },
        ];
        let outcomes =
            refine_roots_parallel_hp(&CubicRoots, &tasks, &CancellationToken::new()).unwrap();
        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["right", "left", "center"]
        );
        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.requested_precision_bits)
                .collect::<Vec<_>>(),
            vec![192, 128, 256]
        );
        assert!(outcomes.iter().enumerate().all(|(index, outcome)| {
            outcome.input_ordinal == index
                && outcome.approximation.precision_bits == outcome.requested_precision_bits
        }));
    }

    #[test]
    fn root_api_exposes_certified_entire_and_meromorphic_contour_counts() {
        use xc_numerics::interval::ComplexRational;

        let q = |numerator, denominator| Rational::from((numerator, denominator));
        let real = |value| ComplexRational {
            real: q(value, 1),
            imaginary: q(0, 1),
        };
        let numerator = vec![real(1), real(0), real(1)]; // z^2 + 1
        let denominator = vec![real(0), real(1)]; // z
        let rectangle =
            RationalContourRectangle::new(q(-2, 1), q(2, 1), q(-2, 1), q(2, 1)).unwrap();
        let entire = certify_entire_polynomial_contour(&numerator, rectangle.clone(), 20).unwrap();
        assert_eq!(entire.zero_count, 2);
        assert!(entire
            .cells
            .iter()
            .all(|cell| cell.image_enclosure.excludes_zero()));
        let meromorphic =
            certify_meromorphic_rational_contour(&numerator, &denominator, rectangle, 20).unwrap();
        assert_eq!(meromorphic.zeros_minus_poles, 1);
    }
}
