use xc_root::*;
struct Damped {
    center: f64,
    width: f64,
}
impl RealFunctionF64 for Damped {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        let d = x - self.center;
        Ok(100.0 * d * (-(d / self.width).powi(2)).exp())
    }
    fn derivative(&self, x: f64) -> Result<f64, RootError> {
        let d = x - self.center;
        Ok(100.0 * (-(d / self.width).powi(2)).exp() * (1.0 - 2.0 * (d / self.width).powi(2)))
    }
}
struct Cubic {
    center: f64,
    scale: f64,
}
impl RealFunctionF64 for Cubic {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(self.scale * (x - self.center).powi(3))
    }
    fn derivative(&self, x: f64) -> Result<f64, RootError> {
        Ok(3.0 * self.scale * (x - self.center).powi(2))
    }
}
#[test]
fn smooth_roots_are_position_refined_despite_tiny_tail_or_absolute_residual() {
    for function in [
        &Damped {
            center: 0.3,
            width: 0.05,
        } as &dyn RealFunctionF64,
        &Damped {
            center: 0.63,
            width: 0.12,
        },
        &Cubic {
            center: 0.3,
            scale: 1.0,
        },
        &Cubic {
            center: 0.61,
            scale: 1e-20,
        },
    ] {
        let center = if function.evaluate(0.3).unwrap() == 0.0 {
            0.3
        } else if function.evaluate(0.63).unwrap() == 0.0 {
            0.63
        } else {
            0.61
        };
        let bracket = RootBracketF64 {
            lower: 0.0,
            upper: 1.0,
        };
        for result in [
            bisect_f64(function, bracket, &RootStoppingF64::default()).unwrap(),
            safeguarded_newton_f64(function, bracket, 0.9, &RootStoppingF64::default()).unwrap(),
        ] {
            assert!(
                (result.midpoint - center).abs() <= 2e-12,
                "wrong position {} for {center}",
                result.midpoint
            );
            assert_ne!(
                result.convergence_criterion,
                RootConvergenceCriterion::Residual
            );
        }
    }
}
#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{ops::Pow, Float};
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    struct Square;
    impl RealIntervalFunctionHp for Square {
        fn evaluate_interval(&self, x: &I) -> Result<I, RootError> {
            Ok(x.square().sub(&I::from_i64(2, x.precision())))
        }
        fn derivative_interval(&self, x: &I) -> Result<I, RootError> {
            Ok(x.mul(&I::from_i64(2, x.precision())))
        }
    }
    #[test]
    fn unique_root_certificate_replays_prior_witness_and_exact_published_bounds() {
        for p in [79, 128, 257] {
            let initial = I::new(Float::with_val(p, 1), Float::with_val(p, 2)).unwrap();
            let cert = interval_newton_hp(
                &Square,
                &initial,
                &IntervalNewtonOptions {
                    width_tolerance: xc_core::DecimalLiteral::new("1e-18").unwrap(),
                    maximum_iterations: 32,
                },
            )
            .unwrap();
            assert_eq!(cert.status, IntervalRootStatus::CertifiedUnique);
            assert_eq!(cert.schema_version, 2);
            assert!(verify_interval_root_certificate_hp(&Square, &cert).unwrap());
            let lower = Float::with_val(1024, Float::parse(&cert.lower).unwrap());
            let upper = Float::with_val(1024, Float::parse(&cert.upper).unwrap());
            let root = Float::with_val(1024, 2).sqrt();
            assert!(lower < root && root < upper);
            let mut bad = cert.clone();
            bad.uniqueness_witness.as_mut().unwrap().contraction_steps += 1;
            assert!(!verify_interval_root_certificate_hp(&Square, &bad).unwrap());
            let mut bad = cert.clone();
            bad.lower = Float::with_val(p, 2).to_string();
            assert!(!verify_interval_root_certificate_hp(&Square, &bad).unwrap_or(false));
            let mut bad = cert;
            bad.uniqueness_witness = None;
            assert!(!verify_interval_root_certificate_hp(&Square, &bad).unwrap());
        }
    }
    struct StoredLinear(Float);
    impl RealIntervalFunctionHp for StoredLinear {
        fn evaluate_interval(&self, x: &I) -> Result<I, RootError> {
            Ok(x.sub(&I::from_float(&self.0, x.precision()).unwrap()))
        }
        fn derivative_interval(&self, x: &I) -> Result<I, RootError> {
            Ok(I::from_i64(1, x.precision()))
        }
    }
    #[test]
    fn outward_decimal_singleton_contains_the_exact_stored_nondecimal_root() {
        for p in [67, 131, 193] {
            let root = Float::with_val(p, 2).sqrt();
            let f = StoredLinear(root.clone());
            let initial = I::new(Float::with_val(p, 1), Float::with_val(p, 2)).unwrap();
            let cert = interval_newton_hp(
                &f,
                &initial,
                &IntervalNewtonOptions {
                    width_tolerance: xc_core::DecimalLiteral::new("1e-30").unwrap(),
                    maximum_iterations: 8,
                },
            )
            .unwrap();
            assert_eq!(cert.status, IntervalRootStatus::CertifiedUnique);
            assert!(verify_interval_root_certificate_hp(&f, &cert).unwrap());
            let lower = Float::with_val(2048, Float::parse(&cert.lower).unwrap());
            let upper = Float::with_val(2048, Float::parse(&cert.upper).unwrap());
            assert!(lower <= root && root <= upper);
            assert!((upper - lower) < Float::with_val(2048, 2).pow(-(p as i32)));
        }
    }
}
