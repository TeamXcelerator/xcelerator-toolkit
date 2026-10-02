// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Uniform-grid integration on `[a, b]`.
//!
//! Deterministic Riemann-family rules on an equally spaced grid, in the
//! integration variable itself or in its logarithm. These exist alongside
//! Gauss--Legendre (see [`crate::quadrature`]) because cross-checks may need
//! to reproduce another implementation's quadrature convention exactly, not
//! merely converge to the same limit:
//!
//! For sufficiently smooth transformed integrands:
//!
//! - left/right Riemann sums carry an `O(h)` error term proportional to
//!   `g(hi) - g(lo)` for the integrand in the grid variable (negative
//!   for left and positive for right sums, with coefficient `h/2`);
//! - midpoint and trapezoid rules carry `O(h²)` error;
//! - a grid uniform in `log u` and a grid uniform in `u` are different rules
//!   with different finite-step values.
//!
//! Every entry point therefore takes the scheme and the grid variable as
//! explicit arguments, and callers are expected to record both alongside any
//! reported number.
//!
//! # Cache effects
//!
//! No function in this module performs cache lookup, persistence, or
//! publication.

use anyhow::Result;

/// Arithmetic identity: exact endpoints and rounded interior points confined to the domain.
pub const UNIFORM_GRID_SEMANTICS: &str = "uniform-grid-relative-log-span-pinned-endpoints-v3";

/// Which uniform-grid rule to apply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UniformGridScheme {
    /// Sample at the left edge of each cell: `h·Σ f(a + i·h)`, `i = 0…S−1`.
    LeftRiemann,
    /// Sample at the right edge of each cell: `h·Σ f(a + i·h)`, `i = 1…S`.
    RightRiemann,
    /// Sample at cell centers: `h·Σ f(a + (i + ½)·h)`.
    Midpoint,
    /// Trapezoid rule: `h·(½f(a) + interior + ½f(b))`.
    Trapezoid,
}

impl UniformGridScheme {
    /// Stable identifier for recording the convention next to results.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LeftRiemann => "left_riemann",
            Self::RightRiemann => "right_riemann",
            Self::Midpoint => "midpoint",
            Self::Trapezoid => "trapezoid",
        }
    }
}

/// Which variable the grid is uniform in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridVariable {
    /// Equal steps in `u` itself.
    U,
    /// Equal steps in `ln u`; the integral is transformed as
    /// `∫ f(u) du = ∫ f(eᵗ) eᵗ dt` over `[ln a, ln b]`. Requires `a > 0`.
    LogU,
}

impl GridVariable {
    /// Stable identifier for recording the convention next to results.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::U => "uniform_u",
            Self::LogU => "uniform_log_u",
        }
    }
}

fn validate_bounds(a: f64, b: f64, steps: usize, variable: GridVariable) -> Result<()> {
    if !a.is_finite() || !b.is_finite() {
        anyhow::bail!("integration bounds must be finite (got [{a}, {b}])");
    }
    if b <= a {
        anyhow::bail!("integration requires b > a (got [{a}, {b}])");
    }
    if steps == 0 {
        anyhow::bail!("integration requires at least one grid step");
    }
    if variable == GridVariable::LogU && a <= 0.0 {
        anyhow::bail!("a log-u grid requires a > 0 (got a = {a})");
    }
    Ok(())
}

fn finite_grid_value(value: f64) -> Result<f64> {
    if !value.is_finite() {
        anyhow::bail!("uniform-grid evaluation or accumulation is nonfinite");
    }
    Ok(value)
}

fn uniform_sum_f64<F: Fn(f64) -> f64>(
    g: F,
    lo: f64,
    hi: f64,
    steps: usize,
    scheme: UniformGridScheme,
) -> Result<f64> {
    // Integer and half-integer grid offsets must be exactly representable.
    if steps as u128 > (1u128 << 52) {
        anyhow::bail!("binary64 grid offsets require steps <= 2^52");
    }
    let h = (hi - lo) / steps as f64;
    if !lo.is_finite() || !hi.is_finite() || hi <= lo || !h.is_finite() || h <= 0.0 {
        anyhow::bail!("uniform grid is not representable at binary64 precision");
    }
    let sample = |x: f64| -> Result<f64> {
        finite_grid_value(x)?;
        finite_grid_value(g(x.clamp(lo, hi)))
    };
    let sum_samples = |start: usize, end: usize, offset: f64| -> Result<f64> {
        (start..end).try_fold(-0.0, |sum, i| {
            finite_grid_value(sum + sample(lo + (i as f64 + offset) * h)?)
        })
    };
    let sum = match scheme {
        UniformGridScheme::LeftRiemann => sum_samples(0, steps, 0.0)?,
        UniformGridScheme::RightRiemann => (1..=steps).try_fold(-0.0, |sum, i| {
            finite_grid_value(sum + sample(if i == steps { hi } else { lo + i as f64 * h })?)
        })?,
        UniformGridScheme::Midpoint => sum_samples(0, steps, 0.5)?,
        UniformGridScheme::Trapezoid => {
            let interior = sum_samples(1, steps, 0.0)?;
            finite_grid_value(0.5 * finite_grid_value(sample(lo)? + sample(hi)?)? + interior)?
        }
    };
    finite_grid_value(sum * h)
}

/// Integrate `f` over `[a, b]` on a uniform grid of `steps` cells at binary64.
///
/// The scheme and grid variable are the caller's stated convention; record
/// both (e.g. via [`UniformGridScheme::as_str`]) with any reported value.
/// Nonfinite samples/arithmetic and unrepresentable grid widths return errors.
/// This computed finite sum does not certify discretization or callback error.
pub fn uniform_grid_integral_f64<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    steps: usize,
    scheme: UniformGridScheme,
    variable: GridVariable,
) -> Result<f64> {
    validate_bounds(a, b, steps, variable)?;
    match variable {
        GridVariable::U => uniform_sum_f64(f, a, b, steps, scheme),
        GridVariable::LogU => {
            let lo = 0.0;
            let relative = (b - a) / a;
            let hi = if relative.is_finite() {
                relative.ln_1p()
            } else {
                b.ln() - a.ln()
            };
            let g = |t: f64| {
                let u = if t == lo {
                    a
                } else if t == hi {
                    b
                } else {
                    let growth = t.exp();
                    let point = if growth.is_finite() {
                        a * growth
                    } else {
                        (a.ln() + t).exp()
                    };
                    point.clamp(a, b)
                };
                if !u.is_finite() || u <= 0.0 {
                    return f64::NAN;
                }
                f(u) * u
            };
            uniform_sum_f64(g, lo, hi, steps, scheme)
        }
    }
}

#[cfg(feature = "hp")]
pub mod hp {
    //! High-precision uniform-grid integration via rug/MPFR.

    use super::{GridVariable, UniformGridScheme};
    use anyhow::{bail, Result};
    use rug::Float;

    const GUARD_BITS: u32 = 32;

    fn grid_point(lo: &Float, h: &Float, index: usize, midpoint: bool, working: u32) -> Float {
        let mut offset = Float::with_val(working, index);
        if midpoint {
            // Exact dyadic half; never convert the integer index through f64.
            offset += Float::with_val(working, 1) / 2;
        }
        offset *= h;
        offset += lo;
        offset
    }

    fn finite(value: Float) -> Result<Float> {
        if !value.is_finite() {
            bail!("HP uniform-grid evaluation or accumulation is nonfinite");
        }
        Ok(value)
    }

    fn uniform_sum<F: Fn(&Float) -> Float>(
        g: F,
        lo: &Float,
        hi: &Float,
        steps: usize,
        scheme: UniformGridScheme,
        working: u32,
    ) -> Result<Float> {
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            bail!("HP grid bounds collapse or are nonfinite at working precision");
        }
        let mut h = Float::with_val(working, hi - lo);
        h /= steps;
        if !h.is_finite() || h <= 0 {
            bail!("HP grid spacing is nonpositive or nonfinite");
        }
        let sample = |x: &Float| -> Result<Float> {
            if !x.is_finite() {
                bail!("HP grid sample point is nonfinite");
            }
            let bounded = x.clone().max(lo).min(hi);
            finite(Float::with_val(working, g(&bounded)))
        };
        let mut sum = Float::with_val(working, 0);
        let (start, end, midpoint) = match scheme {
            UniformGridScheme::LeftRiemann => (0, steps, false),
            UniformGridScheme::Midpoint => (0, steps, true),
            UniformGridScheme::Trapezoid => {
                let mut edges = sample(lo)?;
                edges += sample(hi)?;
                edges = finite(edges)?;
                edges /= 2;
                sum += edges;
                (1, steps, false)
            }
            UniformGridScheme::RightRiemann => {
                for i in 1..=steps {
                    let point = if i == steps {
                        hi.clone()
                    } else {
                        grid_point(lo, &h, i, false, working)
                    };
                    sum += sample(&point)?;
                    if !sum.is_finite() {
                        bail!("HP uniform-grid accumulation is nonfinite");
                    }
                }
                return finite(sum * h);
            }
        };
        for i in start..end {
            sum += sample(&grid_point(lo, &h, i, midpoint, working))?;
            if !sum.is_finite() {
                bail!("HP uniform-grid accumulation is nonfinite");
            }
        }
        finite(sum * h)
    }

    /// Compute the chosen finite quadrature sum at the maximum of `prec + 32`
    /// and the endpoint precisions, then
    /// round to `prec`. Bounds and grid indices never pass through binary64.
    ///
    /// Bounds must be finite and ordered, with positive lower bound for a
    /// log grid. Invalid precision, unresolved grid bounds, and nonfinite
    /// samples or arithmetic return errors. This is a computed quadrature
    /// value: discretization error and callback accuracy require separate bounds.
    pub fn uniform_grid_integral<F: Fn(&Float) -> Float>(
        f: F,
        a: &Float,
        b: &Float,
        steps: usize,
        scheme: UniformGridScheme,
        variable: GridVariable,
        prec: u32,
    ) -> Result<Float> {
        if !a.is_finite() || !b.is_finite() || b <= a || steps == 0 {
            bail!("HP integration requires finite bounds b > a and at least one step");
        }
        if variable == GridVariable::LogU && a <= &0 {
            bail!("an HP log-u grid requires a > 0");
        }
        let working = prec
            .checked_add(GUARD_BITS)
            .filter(|&p| p <= rug::float::prec_max());
        if prec < rug::float::prec_min() || working.is_none() {
            bail!("HP integration precision is outside the supported range");
        }
        let working = working
            .expect("validated working precision")
            .max(a.prec())
            .max(b.prec());
        if usize::BITS - steps.leading_zeros() + 1 > working {
            bail!("HP working precision cannot represent the requested grid offsets");
        }
        let value = match variable {
            GridVariable::U => uniform_sum(
                f,
                &Float::with_val(working, a),
                &Float::with_val(working, b),
                steps,
                scheme,
                working,
            )?,
            GridVariable::LogU => {
                let lo = Float::with_val(working, 0);
                let mut relative = Float::with_val(working, b - a);
                relative /= a;
                let hi = if relative.is_finite() {
                    relative.ln_1p()
                } else {
                    Float::with_val(working, b).ln() - Float::with_val(working, a).ln()
                };
                let g = |t: &Float| {
                    let u = if t == &lo {
                        Float::with_val(working, a)
                    } else if t == &hi {
                        Float::with_val(working, b)
                    } else {
                        let growth = t.clone().exp();
                        let point = if growth.is_finite() {
                            Float::with_val(working, a) * growth
                        } else {
                            (Float::with_val(working, a).ln() + t).exp()
                        };
                        point.max(a).min(b)
                    };
                    if !u.is_finite() || u <= 0 {
                        return Float::with_val(working, rug::float::Special::Nan);
                    }
                    Float::with_val(working, f(&u)) * u
                };
                uniform_sum(g, &lo, &hi, steps, scheme, working)?
            }
        };
        finite(Float::with_val(prec, value))
    }

    #[cfg(test)]
    mod exact_index_tests {
        use super::*;
        #[test]
        #[cfg(target_pointer_width = "64")]
        fn high_grid_indices_keep_their_integer_and_half_integer_bits() {
            let p = 128;
            let i = (1usize << 54) + 1;
            let point = grid_point(&Float::with_val(p, 0), &Float::with_val(p, 1), i, true, p);
            let expected = Float::with_val(p, 2 * i + 1) / 2;
            assert_eq!(point, expected);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `∫₁² 3u² du = 7` exactly; every scheme must converge to it, and the
    /// error orders must be the ones the module documents: halving `h` halves
    /// a left-Riemann error (`O(h)`) and quarters a trapezoid error (`O(h²)`).
    /// Those orders are the mathematical reason the scheme choice must travel
    /// with reported numbers.
    #[test]
    fn schemes_converge_at_their_documented_orders() {
        let f = |u: f64| 3.0 * u * u;
        let exact = 7.0_f64;
        for scheme in [
            UniformGridScheme::LeftRiemann,
            UniformGridScheme::RightRiemann,
            UniformGridScheme::Midpoint,
            UniformGridScheme::Trapezoid,
        ] {
            let coarse =
                uniform_grid_integral_f64(f, 1.0, 2.0, 1_000, scheme, GridVariable::U).unwrap();
            let fine =
                uniform_grid_integral_f64(f, 1.0, 2.0, 2_000, scheme, GridVariable::U).unwrap();
            let ratio = (coarse - exact).abs() / (fine - exact).abs();
            let expected_ratio = match scheme {
                UniformGridScheme::LeftRiemann | UniformGridScheme::RightRiemann => 2.0,
                UniformGridScheme::Midpoint | UniformGridScheme::Trapezoid => 4.0,
            };
            assert!(
                (ratio - expected_ratio).abs() < 0.25,
                "{}: error ratio {ratio}, expected ~{expected_ratio}",
                scheme.as_str()
            );
        }
    }

    /// A u-grid and a log-u-grid are different rules at finite step but must
    /// agree in the limit; with ample steps they match to quadrature accuracy.
    #[test]
    fn log_grid_and_u_grid_agree_on_a_smooth_integrand() {
        let f = |u: f64| (-u).exp();
        let on_u = uniform_grid_integral_f64(
            f,
            1.0,
            4.0,
            50_000,
            UniformGridScheme::Trapezoid,
            GridVariable::U,
        )
        .unwrap();
        let on_log = uniform_grid_integral_f64(
            f,
            1.0,
            4.0,
            50_000,
            UniformGridScheme::Trapezoid,
            GridVariable::LogU,
        )
        .unwrap();
        let exact = (-1.0_f64).exp() - (-4.0_f64).exp();
        assert!((on_u - exact).abs() < 1e-9);
        assert!((on_log - exact).abs() < 1e-9);
        assert!((on_u - on_log).abs() < 1e-9);
    }

    #[test]
    fn invalid_requests_are_rejected() {
        let f = |_: f64| 1.0;
        assert!(uniform_grid_integral_f64(
            f,
            2.0,
            1.0,
            10,
            UniformGridScheme::Midpoint,
            GridVariable::U
        )
        .is_err());
        assert!(uniform_grid_integral_f64(
            f,
            1.0,
            2.0,
            0,
            UniformGridScheme::Midpoint,
            GridVariable::U
        )
        .is_err());
        assert!(uniform_grid_integral_f64(
            f,
            -1.0,
            2.0,
            10,
            UniformGridScheme::Midpoint,
            GridVariable::LogU
        )
        .is_err());
        assert!(uniform_grid_integral_f64(
            f,
            f64::NAN,
            2.0,
            10,
            UniformGridScheme::Midpoint,
            GridVariable::U
        )
        .is_err());
    }

    #[cfg(feature = "hp")]
    mod hp_tests {
        use super::super::hp;
        use super::*;
        use rug::Float;

        /// `∫₁² u^{−1/2} du = 2(√2 − 1)`: the HP trapezoid value must land on
        /// the closed form within its `O(h²)` budget, on both grids.
        #[test]
        fn hp_trapezoid_matches_a_closed_form() {
            let prec = 256;
            let a = Float::with_val(prec, 1u32);
            let b = Float::with_val(prec, 2u32);
            let exact = (Float::with_val(prec, 2u32).sqrt() - 1u32) * 2u32;
            for variable in [GridVariable::U, GridVariable::LogU] {
                let got = hp::uniform_grid_integral(
                    |u: &Float| u.clone().recip().sqrt(),
                    &a,
                    &b,
                    10_000,
                    UniformGridScheme::Trapezoid,
                    variable,
                    prec,
                )
                .unwrap();
                let error = Float::with_val(prec, &got - &exact).abs();
                assert!(error < 1e-8, "{}: error {error:?}", variable.as_str());
            }
        }

        /// The HP and f64 paths implement the same rule: at matching inputs
        /// they must agree to f64 accuracy.
        #[test]
        fn hp_and_f64_paths_agree() {
            let prec = 128;
            let a = Float::with_val(prec, 1u32);
            let b = Float::with_val(prec, 3u32);
            let hp_value = hp::uniform_grid_integral(
                |u: &Float| u.clone().square(),
                &a,
                &b,
                5_000,
                UniformGridScheme::LeftRiemann,
                GridVariable::U,
                prec,
            )
            .unwrap()
            .to_f64();
            let f64_value = uniform_grid_integral_f64(
                |u| u * u,
                1.0,
                3.0,
                5_000,
                UniformGridScheme::LeftRiemann,
                GridVariable::U,
            )
            .unwrap();
            assert!((hp_value - f64_value).abs() < 1e-12);
        }
    }
}
