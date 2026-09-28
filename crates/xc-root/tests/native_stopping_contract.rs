use xc_root::{
    bisect_f64, safeguarded_newton_f64, RealFunctionF64, RootBracketF64, RootError, RootStoppingF64,
};
struct Linear;
impl RealFunctionF64 for Linear {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(x - 0.75)
    }
    fn derivative(&self, _: f64) -> Result<f64, RootError> {
        Ok(1.0)
    }
}
fn stopping() -> RootStoppingF64 {
    RootStoppingF64 {
        absolute_x_tolerance: 1.0,
        relative_x_tolerance: 1e-30,
        residual_tolerance: 1e-30,
        maximum_iterations: 1,
    }
}
#[test]
fn rounded_width_does_not_meet_exact_absolute_tolerance() {
    let bracket = RootBracketF64 {
        lower: -2.0f64.powi(-54),
        upper: 1.0,
    };
    // Exact width is 1 + 2^-54 > 1, and |f(midpoint)| = 1/4.
    let b = bisect_f64(&Linear, bracket, &stopping());
    assert!(matches!(b, Err(RootError::NonConvergence(_))), "{b:?}");
}
#[test]
fn newton_rounded_width_does_not_meet_exact_absolute_tolerance() {
    let bracket = RootBracketF64 {
        lower: -2.0f64.powi(-54),
        upper: 1.0,
    };
    let n = safeguarded_newton_f64(&Linear, bracket, 0.5, &stopping());
    assert!(matches!(n, Err(RootError::NonConvergence(_))), "{n:?}");
}

struct ShiftedLinear;
impl RealFunctionF64 for ShiftedLinear {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(x - 5.25)
    }
    fn derivative(&self, _: f64) -> Result<f64, RootError> {
        Ok(1.0)
    }
}
#[test]
fn rounded_relative_product_does_not_relax_the_tolerance() {
    let bracket = RootBracketF64 {
        lower: 4.25,
        upper: 5.75,
    };
    // Stored binary64 0.3 is below 3/10; its exact product with 5 is < 1.5.
    let limits = RootStoppingF64 {
        absolute_x_tolerance: 1e-30,
        relative_x_tolerance: 0.3,
        ..stopping()
    };
    let b = bisect_f64(&ShiftedLinear, bracket, &limits);
    assert!(matches!(b, Err(RootError::NonConvergence(_))), "{b:?}");
    let n = safeguarded_newton_f64(&ShiftedLinear, bracket, 5.0, &limits);
    assert!(matches!(n, Err(RootError::NonConvergence(_))), "{n:?}");
}
#[test]
fn exact_width_boundary_is_inclusive() {
    let bracket = RootBracketF64 {
        lower: 0.0,
        upper: 1.0,
    };
    assert!(bisect_f64(&Linear, bracket, &stopping()).is_ok());
    assert!(safeguarded_newton_f64(&Linear, bracket, 0.5, &stopping()).is_ok());
}

#[test]
fn discovery_does_not_merge_roots_just_beyond_duplicate_tolerance() {
    use xc_root::{
        discover_pole_aware_sign_changes_f64, MeromorphicFunctionF64, PoleAwareDiscoveryOptionsF64,
    };
    struct Polynomial;
    impl RealFunctionF64 for Polynomial {
        fn evaluate(&self, x: f64) -> Result<f64, RootError> {
            Ok((x + 2.0f64.powi(-54)) * (x - 1.0))
        }
    }
    impl MeromorphicFunctionF64 for Polynomial {
        fn real_poles(&self) -> &[f64] {
            &[]
        }
    }
    let options = PoleAwareDiscoveryOptionsF64 {
        subdivisions_per_interval: 8,
        pole_margin_fraction: 0.0,
        duplicate_tolerance: 1.0,
        stopping: RootStoppingF64 {
            absolute_x_tolerance: f64::from_bits(1),
            relative_x_tolerance: f64::from_bits(1),
            residual_tolerance: f64::from_bits(1),
            maximum_iterations: 500,
        },
    };
    let roots = discover_pole_aware_sign_changes_f64(&Polynomial, -2.0, 2.0, &options).unwrap();
    assert_eq!(roots.len(), 2, "{roots:?}");
    assert_eq!(roots[0].midpoint, -2.0f64.powi(-54));
    assert_eq!(roots[1].midpoint, 1.0);
}
