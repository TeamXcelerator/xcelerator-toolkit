#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval;
use xc_root::{
    interval_newton_hp, IntervalNewtonOptions, IntervalRootStatus, RealIntervalFunctionHp,
    RootError,
};

struct Broken {
    derivative: bool,
    mismatch: bool,
}
impl RealIntervalFunctionHp for Broken {
    fn evaluate_interval(&self, x: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        if self.derivative {
            return Ok(x.clone());
        }
        if self.mismatch {
            return Ok(MpfrInterval::from_i64(1, x.precision() + 32));
        }
        let huge = Float::with_val(x.precision(), 1) << (rug::float::exp_max() - 1) as u32;
        Ok(MpfrInterval::point(huge).square().sin())
    }
    fn derivative_interval(&self, x: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        if self.derivative {
            return Broken {
                derivative: false,
                mismatch: self.mismatch,
            }
            .evaluate_interval(x);
        }
        Ok(MpfrInterval::from_i64(1, x.precision()))
    }
}
#[test]
fn invalid_or_mismatched_callback_enclosures_return_errors() {
    let p = 128;
    let candidate = MpfrInterval::new(Float::with_val(p, -1), Float::with_val(p, 1)).unwrap();
    let options = IntervalNewtonOptions {
        width_tolerance: xc_core::DecimalLiteral::new("1e-20").unwrap(),
        maximum_iterations: 5,
    };
    for derivative in [false, true] {
        for mismatch in [false, true] {
            assert!(interval_newton_hp(
                &Broken {
                    derivative,
                    mismatch
                },
                &candidate,
                &options
            )
            .is_err());
        }
    }
    for text in ["1e400000000", "1e-400000000"] {
        let bad = IntervalNewtonOptions {
            width_tolerance: xc_core::DecimalLiteral::new(text).unwrap(),
            ..options.clone()
        };
        assert!(bad.validate(p).is_err());
    }
}

struct Linear;
impl RealIntervalFunctionHp for Linear {
    fn evaluate_interval(&self, x: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        Ok(x.sub(&MpfrInterval::from_i64(2, x.precision())))
    }
    fn derivative_interval(&self, x: &MpfrInterval) -> Result<MpfrInterval, RootError> {
        Ok(MpfrInterval::from_i64(1, x.precision()))
    }
}
#[test]
fn valid_newton_still_distinguishes_existence_and_exclusion() {
    let p = 128;
    let options = IntervalNewtonOptions {
        width_tolerance: xc_core::DecimalLiteral::new("1e-20").unwrap(),
        maximum_iterations: 5,
    };
    for (a, b, status) in [
        (1, 3, IntervalRootStatus::CertifiedUnique),
        (4, 5, IntervalRootStatus::ExcludedNoRoot),
    ] {
        let candidate = MpfrInterval::new(Float::with_val(p, a), Float::with_val(p, b)).unwrap();
        assert_eq!(
            interval_newton_hp(&Linear, &candidate, &options)
                .unwrap()
                .status,
            status
        );
    }
}
