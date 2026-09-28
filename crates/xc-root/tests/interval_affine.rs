//! Manufactured affine roots; exact rational root checked against decoded endpoints.
#![cfg(feature = "hp")]
use rug::{Float, Rational};
use xc_core::DecimalLiteral;
use xc_numerics::mpfr_interval::MpfrInterval;
use xc_root::{
    interval_newton_hp, IntervalNewtonOptions, IntervalRootStatus, RealIntervalFunctionHp,
    RootError,
};
struct Affine {
    exponent: i32,
}
impl RealIntervalFunctionHp for Affine {
    fn evaluate_interval(&self, x: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        let p = x.precision();
        let scale = MpfrInterval::point(Float::with_val(p, 1) << self.exponent);
        Ok(x.mul(&MpfrInterval::from_i64(3, p))
            .sub(&MpfrInterval::from_i64(1, p))
            .mul(&scale))
    }
    fn derivative_interval(&self, x: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        Ok(MpfrInterval::point(
            Float::with_val(x.precision(), 3) << self.exponent,
        ))
    }
}
#[test]
fn certified_affine_roots_preserve_exact_one_third_at_extreme_scales() {
    let expected = Rational::from((1, 3));
    for p in [32, 79, 192] {
        for exponent in [-1000, 0, 1000] {
            let initial = MpfrInterval::new(Float::with_val(p, 0), Float::with_val(p, 1)).unwrap();
            let result = interval_newton_hp(
                &Affine { exponent },
                &initial,
                &IntervalNewtonOptions {
                    width_tolerance: DecimalLiteral::new("1e-8").unwrap(),
                    maximum_iterations: 8,
                },
            )
            .unwrap();
            assert_eq!(result.status, IntervalRootStatus::CertifiedUnique);
            let lower = Float::with_val(p, Float::parse(&result.lower).unwrap())
                .to_rational()
                .unwrap();
            let upper = Float::with_val(p, Float::parse(&result.upper).unwrap())
                .to_rational()
                .unwrap();
            assert!(
                lower <= expected && expected <= upper,
                "p={p}, exponent={exponent}"
            );
        }
    }
}
