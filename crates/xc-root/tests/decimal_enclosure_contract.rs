use xc_core::DecimalLiteral;
use xc_root::{RootApproximationF64, RootApproximationStatus, RootBracketF64};
fn approximation(value: f64) -> RootApproximationF64 {
    RootApproximationF64 {
        midpoint: value,
        bracket: RootBracketF64 {
            lower: value,
            upper: value,
        },
        residual: 0.,
        derivative_magnitude: None,
        iterations: 0,
        function_evaluations: 0,
        derivative_evaluations: 0,
        status: RootApproximationStatus::Refined,
        convergence_criterion: xc_root::RootConvergenceCriterion::Unspecified,
        method: "exact stored singleton".into(),
    }
}
#[test]
fn decimal_export_contains_the_exact_binary_root_bracket() {
    // IEEE binary64 0.1 is the exact rational 3602879701896397/2^55.
    for (value, exact) in [
        (
            0.1,
            "0.1000000000000000055511151231257827021181583404541015625",
        ),
        (
            -0.1,
            "-0.1000000000000000055511151231257827021181583404541015625",
        ),
    ] {
        let interval = approximation(value).decimal_interval();
        let exact = DecimalLiteral::new(exact).unwrap();
        assert!(
            DecimalLiteral::new(interval.lower)
                .unwrap()
                .cmp_numeric(&exact)
                .unwrap()
                != std::cmp::Ordering::Greater
        );
        assert!(
            DecimalLiteral::new(interval.upper)
                .unwrap()
                .cmp_numeric(&exact)
                .unwrap()
                != std::cmp::Ordering::Less
        );
    }
}
#[cfg(feature = "hp")]
#[test]
fn decimal_exports_preserve_extreme_endpoints_against_exact_mpfr_values() {
    use rug::Float;
    for value in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::MAX,
        0.1,
        -0.1,
        -f64::MAX,
    ] {
        let interval = approximation(value).decimal_interval();
        let source = Float::with_val(4096, value);
        let lower = Float::with_val(4096, Float::parse(&interval.lower).unwrap());
        let upper = Float::with_val(4096, Float::parse(&interval.upper).unwrap());
        assert!(lower <= source && upper >= source);
    }
}

#[test]
fn invalid_export_brackets_return_an_error() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(approximation(value).try_decimal_interval().is_err());
    }
    let mut value = approximation(1.);
    value.bracket.lower = 2.;
    assert!(value.try_decimal_interval().is_err());
}
