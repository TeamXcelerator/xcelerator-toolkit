#![cfg(feature = "hp")]
use rug::Float;
use xc_core::DecimalLiteral;
use xc_root::{bisect_hp, cross_check_simple_root_hp, RealFunctionHp, RootError, RootStoppingHp};
struct Linear {
    root: Float,
}
impl RealFunctionHp for Linear {
    fn evaluate(&self, x: &Float, p: u32) -> Result<Float, RootError> {
        let mut value = Float::with_val(p, x / &self.root);
        value -= 1;
        Ok(value)
    }
    fn derivative(&self, _: &Float, p: u32) -> Result<Float, RootError> {
        Ok(Float::with_val(p, 1) / &self.root)
    }
}
fn stopping() -> RootStoppingHp {
    RootStoppingHp {
        x_tolerance: DecimalLiteral::new("1e-20").unwrap(),
        residual_tolerance: DecimalLiteral::new("1e-20").unwrap(),
        maximum_iterations: 50,
    }
}
#[test]
fn a_final_derivative_diagnostic_is_not_an_independent_newton_step() {
    let p = 128;
    let f = Linear {
        root: Float::with_val(p, 1),
    };
    let r = cross_check_simple_root_hp(
        &f,
        &Float::with_val(p, 0),
        &Float::with_val(p, 2),
        &Float::with_val(p, 1),
        p,
        &stopping(),
        &DecimalLiteral::new("1e-10").unwrap(),
    )
    .unwrap();
    assert!(
        !r.independence_established,
        "a terminal derivative was incorrectly counted as independent root refinement"
    );
    assert!(!r.accepted);
}
#[test]
fn point_bisection_handles_a_finite_bracket_with_overflowing_endpoint_sum() {
    let p = 128;
    let lower = Float::with_val(p, 1) << (rug::float::exp_max() - 2);
    let upper = Float::with_val(p, &lower * 3);
    let root = Float::with_val(p, &lower * 2);
    let result = bisect_hp(&Linear { root }, &lower, &upper, p, &stopping());
    assert!(result.is_ok(), "{result:?}");
}
#[test]
fn unrepresentable_decimal_tolerances_are_errors() {
    let p = 128;
    let f = Linear {
        root: Float::with_val(p, 1),
    };
    for decimal in ["1e1000000000", "1e-1000000000"] {
        let mut s = stopping();
        s.x_tolerance = DecimalLiteral::new(decimal).unwrap();
        assert!(
            bisect_hp(&f, &Float::with_val(p, 0), &Float::with_val(p, 3), p, &s).is_err(),
            "{decimal}"
        );
    }
}
