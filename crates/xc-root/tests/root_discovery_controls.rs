use xc_root::*;
struct Close {
    center: f64,
    offset: f64,
}
impl RealFunctionF64 for Close {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok((x - (self.center - self.offset)) * (x - (self.center + self.offset)))
    }
}
impl MeromorphicFunctionF64 for Close {
    fn real_poles(&self) -> &[f64] {
        &[]
    }
}
struct HiddenPole {
    center: f64,
    residue: f64,
    scale: f64,
}
impl RealFunctionF64 for HiddenPole {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        let d = x - self.center;
        Ok(self.scale * d.powi(3) + self.residue / d)
    }
    fn derivative(&self, x: f64) -> Result<f64, RootError> {
        let d = x - self.center;
        Ok(3. * self.scale * d * d - self.residue / (d * d))
    }
}
#[test]
fn disjoint_sign_brackets_keep_close_distinct_roots() {
    for (center, offset) in [(0.5, 4e-13), (0.25, 2e-13), (0.75, 3e-13)] {
        let function = Close { center, offset };
        let options = PoleAwareDiscoveryOptionsF64 {
            stopping: RootStoppingF64 {
                absolute_x_tolerance: 1e-15,
                relative_x_tolerance: 1e-18,
                residual_tolerance: 1e-300,
                maximum_iterations: 200,
            },
            ..PoleAwareDiscoveryOptionsF64::default()
        };
        let roots = discover_pole_aware_sign_changes_f64(&function, 0., 1., &options).unwrap();
        assert_eq!(roots.len(), 2, "center={center},roots={roots:?}");
        assert!(roots[0].bracket.upper < roots[1].bracket.lower);
        for (root, exact) in roots.iter().zip([center - offset, center + offset]) {
            assert!((root.midpoint - exact).abs() < 1e-15);
        }
    }
}
#[test]
fn local_guard_rejects_small_residue_pole_hidden_by_large_endpoint_values() {
    for (center, residue, scale) in [(0.3, 1e-12, 1e10), (0.61, 1e-15, 1e12), (0.43, 1e-10, 1e8)] {
        let function = HiddenPole {
            center,
            residue,
            scale,
        };
        let b = RootBracketF64 {
            lower: 0.,
            upper: 1.,
        };
        let bisection = bisect_f64(&function, b, &RootStoppingF64::default());
        assert!(
            bisection.is_err(),
            "center={center}, bisection={bisection:?}"
        );
        let newton = safeguarded_newton_f64(&function, b, 0.8, &RootStoppingF64::default());
        assert!(newton.is_err(), "center={center}, newton={newton:?}");
    }
}

struct Smooth {
    center: f64,
    width: f64,
}
impl RealFunctionF64 for Smooth {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        let d = x - self.center;
        Ok(100.0 * d * (-(d / self.width).powi(2)).exp())
    }
    fn derivative(&self, x: f64) -> Result<f64, RootError> {
        let d = x - self.center;
        Ok(100.0 * (-(d / self.width).powi(2)).exp() * (1.0 - 2.0 * (d / self.width).powi(2)))
    }
}
#[test]
fn smooth_roots_remain_position_refined_with_stationary_small_endpoint_residuals() {
    for (center, width) in [(0.3, 0.05), (0.63, 0.12), (0.41, 0.09)] {
        let function = Smooth { center, width };
        let b = RootBracketF64 {
            lower: 0.0,
            upper: 1.0,
        };
        for result in [
            bisect_f64(&function, b, &RootStoppingF64::default()),
            safeguarded_newton_f64(&function, b, 0.8, &RootStoppingF64::default()),
        ] {
            let root = result.unwrap();
            assert!(
                (root.midpoint - center).abs() < 1e-12,
                "center={center}, root={root:?}"
            );
            assert_ne!(
                root.convergence_criterion,
                RootConvergenceCriterion::Residual
            );
        }
    }
}
#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{ops::Pow, Float};
    impl RealFunctionHp for HiddenPole {
        fn evaluate(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            let d = Float::with_val(p, x) - Float::with_val(p, self.center);
            Ok(Float::with_val(p, self.scale) * d.clone().pow(3)
                + Float::with_val(p, self.residue) / d)
        }
        fn derivative(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            let d = Float::with_val(p, x) - Float::with_val(p, self.center);
            Ok(
                Float::with_val(p, 3) * Float::with_val(p, self.scale) * d.clone().square()
                    - Float::with_val(p, self.residue) / d.square(),
            )
        }
    }
    impl RealFunctionHp for Smooth {
        fn evaluate(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            let d = Float::with_val(p, x) - Float::with_val(p, self.center);
            let scaled = d.clone() / Float::with_val(p, self.width);
            Ok(Float::with_val(p, 100) * d * (-scaled.square()).exp())
        }
        fn derivative(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            let d = Float::with_val(p, x) - Float::with_val(p, self.center);
            let square = (d / Float::with_val(p, self.width)).square();
            Ok(Float::with_val(p, 100)
                * (-square.clone()).exp()
                * (Float::with_val(p, 1) - Float::with_val(p, 2) * square))
        }
    }
    #[test]
    fn hp_guard_matches_native_root_free_poles_and_smooth_known_roots() {
        let p = 192;
        let (lo, hi, x) = (
            Float::with_val(p, 0),
            Float::with_val(p, 1),
            Float::with_val(p, 0.8),
        );
        let stopping = RootStoppingHp {
            x_tolerance: xc_core::DecimalLiteral::new("1e-30").unwrap(),
            residual_tolerance: xc_core::DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 240,
        };
        for (center, residue, scale) in
            [(0.3, 1e-12, 1e10), (0.61, 1e-15, 1e12), (0.43, 1e-10, 1e8)]
        {
            let function = HiddenPole {
                center,
                residue,
                scale,
            };
            for result in [
                bisect_hp(&function, &lo, &hi, p, &stopping),
                safeguarded_newton_hp(&function, &lo, &hi, &x, p, &stopping),
            ] {
                assert!(result.is_err(), "center={center}, result={result:?}");
            }
        }
        for (center, width) in [(0.3, 0.05), (0.63, 0.12), (0.41, 0.09)] {
            let function = Smooth { center, width };
            for result in [
                bisect_hp(&function, &lo, &hi, p, &stopping),
                safeguarded_newton_hp(&function, &lo, &hi, &x, p, &stopping),
            ] {
                let root = result.unwrap();
                let got = Float::with_val(p, Float::parse(&root.midpoint).unwrap());
                assert!(
                    (got - Float::with_val(p, center)).abs()
                        < Float::with_val(p, Float::parse("1e-29").unwrap()),
                    "center={center}, root={root:?}"
                );
                assert_ne!(
                    root.convergence_criterion,
                    RootConvergenceCriterion::Residual
                );
            }
        }
    }
}
