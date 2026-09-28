use xc_root::*;

struct Pole;
impl RealFunctionF64 for Pole {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(1.0 / (x - 0.3))
    }
    fn derivative(&self, x: f64) -> Result<f64, RootError> {
        Ok(-1.0 / (x - 0.3).powi(2))
    }
}
struct ScaledLinear;
impl RealFunctionF64 for ScaledLinear {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(1e-20 * (x - 0.7))
    }
    fn derivative(&self, _: f64) -> Result<f64, RootError> {
        Ok(1e-20)
    }
}
#[test]
fn point_stops_distinguish_residual_from_location_and_reject_divergent_pole() {
    let b = RootBracketF64 {
        lower: 0.0,
        upper: 1.0,
    };
    let o = RootStoppingF64::default();
    assert!(matches!(
        bisect_f64(&Pole, b, &o),
        Err(RootError::NonConvergence(_))
    ));
    assert!(matches!(
        safeguarded_newton_f64(&Pole, b, 0.5, &o),
        Err(RootError::NonConvergence(_))
    ));
    for v in [
        bisect_f64(&ScaledLinear, b, &o).unwrap(),
        safeguarded_newton_f64(&ScaledLinear, b, 0.5, &o).unwrap(),
    ] {
        assert!((v.midpoint - 0.7).abs() <= 1e-12);
        assert!(v.bracket.lower <= 0.7 && v.bracket.upper >= 0.7);
    }
}
#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::Float;
    impl RealFunctionHp for Pole {
        fn evaluate(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            let d = Float::with_val(p, x) - Float::with_val(p, Float::parse("0.3").unwrap());
            Ok(Float::with_val(p, 1) / d)
        }
        fn derivative(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            let y = RealFunctionHp::evaluate(self, x, p)?;
            Ok(-y.square())
        }
    }
    impl RealFunctionHp for ScaledLinear {
        fn evaluate(&self, x: &Float, p: u32) -> Result<Float, RootError> {
            Ok(
                (Float::with_val(p, x) - Float::with_val(p, Float::parse("0.7").unwrap()))
                    * Float::with_val(p, Float::parse("1e-40").unwrap()),
            )
        }
    }
    #[test]
    fn hp_point_stops_preserve_the_same_mathematical_distinction() {
        let o = RootStoppingHp {
            x_tolerance: xc_core::DecimalLiteral::new("1e-30").unwrap(),
            residual_tolerance: xc_core::DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 200,
        };
        let (a, b, x) = (
            Float::with_val(192, 0),
            Float::with_val(192, 1),
            Float::with_val(192, 0.5),
        );
        assert!(matches!(
            bisect_hp(&Pole, &a, &b, 192, &o),
            Err(RootError::NonConvergence(_))
        ));
        assert!(matches!(
            safeguarded_newton_hp(&Pole, &a, &b, &x, 192, &o),
            Err(RootError::NonConvergence(_))
        ));
        let v = bisect_hp(&ScaledLinear, &a, &b, 192, &o).unwrap();
        let exact = Float::with_val(192, Float::parse("0.7").unwrap());
        let got = Float::with_val(192, Float::parse(&v.midpoint).unwrap());
        assert!((got - exact).abs() < Float::with_val(192, Float::parse("1e-29").unwrap()));
    }
}
