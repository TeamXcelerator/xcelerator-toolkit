#![cfg(feature = "hp")]
use rug::Float;
use xc_core::DecimalLiteral;
use xc_root::{bisect_hp, safeguarded_newton_hp, RealFunctionHp, RootError, RootStoppingHp};

struct Linear {
    value_bits: u32,
    derivative_bits: u32,
}
impl RealFunctionHp for Linear {
    fn evaluate(&self, x: &Float, _: u32) -> Result<Float, RootError> {
        Ok(Float::with_val(self.value_bits, x) - 1u32)
    }
    fn derivative(&self, _: &Float, _: u32) -> Result<Float, RootError> {
        Ok(Float::with_val(self.derivative_bits, 1))
    }
}
fn stopping() -> RootStoppingHp {
    RootStoppingHp {
        x_tolerance: DecimalLiteral::new("1e-30").unwrap(),
        residual_tolerance: DecimalLiteral::new("1e-30").unwrap(),
        maximum_iterations: 100,
    }
}
#[test]
fn hp_bisection_rejects_lower_precision_function_values() {
    let f = Linear {
        value_bits: 32,
        derivative_bits: 128,
    };
    let result = bisect_hp(
        &f,
        &Float::with_val(128, 0),
        &Float::with_val(128, 2),
        128,
        &stopping(),
    );
    assert!(
        matches!(result, Err(RootError::Evaluation(_))),
        "{result:?}"
    );
}
#[test]
fn hp_newton_rejects_lower_precision_derivatives() {
    let f = Linear {
        value_bits: 128,
        derivative_bits: 32,
    };
    let result = safeguarded_newton_hp(
        &f,
        &Float::with_val(128, 0),
        &Float::with_val(128, 2),
        &Float::with_val(128, 0.5),
        128,
        &stopping(),
    );
    assert!(
        matches!(result, Err(RootError::Evaluation(_))),
        "{result:?}"
    );
}
#[test]
fn callbacks_at_requested_or_guard_precision_refine_the_exact_linear_root() {
    for bits in [128, 192] {
        let f = Linear {
            value_bits: bits,
            derivative_bits: bits,
        };
        let report = safeguarded_newton_hp(
            &f,
            &Float::with_val(128, 0),
            &Float::with_val(128, 2),
            &Float::with_val(128, 0.5),
            128,
            &stopping(),
        )
        .unwrap();
        assert_eq!(
            Float::with_val(128, Float::parse(&report.midpoint).unwrap()),
            1
        );
        assert_eq!(report.newton_steps, 1);
    }
}
