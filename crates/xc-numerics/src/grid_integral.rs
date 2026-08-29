// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Uniform-grid integration on `[a, b]`.
//!
//! Deterministic Riemann-family rules on an equally spaced grid, in the
//! integration variable itself or in its logarithm. These exist alongside
//! Gauss--Legendre (see [`crate::quadrature`]) because collaborative
//! cross-checks must be able to reproduce a partner's quadrature convention
//! exactly, not merely converge to the same limit:
//!
//! - left/right Riemann sums carry an `O(h)` error term proportional to
//!   `F(b) − F(a)`;
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

fn uniform_sum_f64<F: Fn(f64) -> f64>(
    g: F,
    lo: f64,
    hi: f64,
    steps: usize,
    scheme: UniformGridScheme,
) -> f64 {
    let h = (hi - lo) / steps as f64;
    let sum = match scheme {
        UniformGridScheme::LeftRiemann => (0..steps).map(|i| g(lo + i as f64 * h)).sum::<f64>(),
        UniformGridScheme::RightRiemann => (1..=steps).map(|i| g(lo + i as f64 * h)).sum::<f64>(),
        UniformGridScheme::Midpoint => (0..steps)
            .map(|i| g(lo + (i as f64 + 0.5) * h))
            .sum::<f64>(),
        UniformGridScheme::Trapezoid => {
            let interior = (1..steps).map(|i| g(lo + i as f64 * h)).sum::<f64>();
            0.5 * (g(lo) + g(hi)) + interior
        }
    };
    sum * h
}

/// Integrate `f` over `[a, b]` on a uniform grid of `steps` cells at binary64.
///
/// The scheme and grid variable are the caller's stated convention; record
/// both (e.g. via [`UniformGridScheme::as_str`]) with any reported value.
pub fn uniform_grid_integral_f64<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    steps: usize,
    scheme: UniformGridScheme,
    variable: GridVariable,
) -> Result<f64> {
    validate_bounds(a, b, steps, variable)?;
    Ok(match variable {
        GridVariable::U => uniform_sum_f64(f, a, b, steps, scheme),
        GridVariable::LogU => {
            let g = |t: f64| {
                let u = t.exp();
                f(u) * u
            };
            uniform_sum_f64(g, a.ln(), b.ln(), steps, scheme)
        }
    })
}

#[cfg(feature = "hp")]
pub mod hp {
    //! High-precision uniform-grid integration via rug/MPFR.

    use super::{validate_bounds, GridVariable, UniformGridScheme};
    use anyhow::Result;
    use rug::Float;

    /// Accumulation guard bits above the requested precision.
    const GUARD_BITS: u32 = 32;

    fn uniform_sum<F: Fn(&Float) -> Float>(
        g: F,
        lo: &Float,
        hi: &Float,
        steps: usize,
        scheme: UniformGridScheme,
        working: u32,
    ) -> Float {
        let mut h = Float::with_val(working, hi - lo);
        h /= steps as u32;
        let point = |i: f64| {
            let mut x = h.clone();
            x *= Float::with_val(working, i);
            x += lo;
            x
        };
        let mut sum = Float::with_val(working, 0u32);
        match scheme {
            UniformGridScheme::LeftRiemann => {
                for i in 0..steps {
                    sum += g(&point(i as f64));
                }
            }
            UniformGridScheme::RightRiemann => {
                for i in 1..=steps {
                    sum += g(&point(i as f64));
                }
            }
            UniformGridScheme::Midpoint => {
                for i in 0..steps {
                    sum += g(&point(i as f64 + 0.5));
                }
            }
            UniformGridScheme::Trapezoid => {
                let mut edges = g(&Float::with_val(working, lo));
                edges += g(&Float::with_val(working, hi));
                edges /= 2u32;
                sum += edges;
                for i in 1..steps {
                    sum += g(&point(i as f64));
                }
            }
        }
        sum * h
    }

    /// Integrate `f` over `[a, b]` on a uniform grid of `steps` cells at
    /// `prec` bits (accumulated at `prec + 32`).
    ///
    /// The scheme and grid variable are the caller's stated convention; record
    /// both with any reported value.
    pub fn uniform_grid_integral<F: Fn(&Float) -> Float>(
        f: F,
        a: &Float,
        b: &Float,
        steps: usize,
        scheme: UniformGridScheme,
        variable: GridVariable,
        prec: u32,
    ) -> Result<Float> {
        validate_bounds(a.to_f64(), b.to_f64(), steps, variable)?;
        let working = prec.saturating_add(GUARD_BITS);
        let value = match variable {
            GridVariable::U => {
                let a = Float::with_val(working, a);
                let b = Float::with_val(working, b);
                uniform_sum(f, &a, &b, steps, scheme, working)
            }
            GridVariable::LogU => {
                let lo = Float::with_val(working, a).ln();
                let hi = Float::with_val(working, b).ln();
                let g = |t: &Float| {
                    let u = t.clone().exp();
                    f(&u) * u
                };
                uniform_sum(g, &lo, &hi, steps, scheme, working)
            }
        };
        Ok(Float::with_val(prec, value))
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
