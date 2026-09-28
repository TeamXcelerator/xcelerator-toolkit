use xc_numerics::grid_integral::{uniform_grid_integral_f64, GridVariable, UniformGridScheme};

const SCHEMES: [UniformGridScheme; 4] = [
    UniformGridScheme::LeftRiemann,
    UniformGridScheme::RightRiemann,
    UniformGridScheme::Midpoint,
    UniformGridScheme::Trapezoid,
];

#[test]
fn quadratic_finite_sums_match_exact_rational_values() {
    // Four cells on [0,1]. These are exact finite-rule values, not the
    // continuum integral 1/3: sums of integer squares and half-integer squares.
    for (scheme, numerator) in SCHEMES.into_iter().zip([14., 30., 21., 22.]) {
        let value =
            uniform_grid_integral_f64(|u| u * u, 0., 1., 4, scheme, GridVariable::U).unwrap();
        assert_eq!(value, numerator / 64.);
    }
}

#[test]
fn nonfinite_samples_and_unrepresentable_widths_are_errors() {
    for scheme in SCHEMES {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                uniform_grid_integral_f64(|_| bad, 0., 1., 2, scheme, GridVariable::U).is_err()
            );
        }
        assert!(
            uniform_grid_integral_f64(|_| 1., -1e308, 1e308, 1, scheme, GridVariable::U).is_err()
        );
    }
}

#[cfg(feature = "hp")]
mod hp_tests {
    use super::*;
    use rug::{float::Special, Float, Rational};
    use xc_numerics::grid_integral::hp::uniform_grid_integral;

    #[test]
    fn distinct_hp_bounds_do_not_pass_through_binary64() {
        let a = Float::with_val(256, 1);
        let delta = Float::with_val(256, 1) >> 100;
        let b = Float::with_val(256, &a + &delta);
        for scheme in SCHEMES {
            assert_eq!(
                uniform_grid_integral(
                    |x| Float::with_val(x.prec(), 1),
                    &a,
                    &b,
                    4,
                    scheme,
                    GridVariable::U,
                    256
                )
                .unwrap(),
                delta
            );
        }
    }

    #[test]
    fn hp_finite_exponent_range_and_quadratic_rules_are_preserved() {
        for scale in ["1e400", "1e-400"] {
            let a = Float::with_val(256, Float::parse(scale).unwrap());
            let b = Float::with_val(256, &a * 2);
            for scheme in SCHEMES {
                assert_eq!(
                    uniform_grid_integral(
                        |x| Float::with_val(x.prec(), 1),
                        &a,
                        &b,
                        4,
                        scheme,
                        GridVariable::U,
                        256
                    )
                    .unwrap(),
                    a
                );
            }
        }
        for (scheme, numerator) in SCHEMES.into_iter().zip([14, 30, 21, 22]) {
            let result = uniform_grid_integral(
                |x| x.clone().square(),
                &Float::with_val(128, 0),
                &Float::with_val(128, 1),
                4,
                scheme,
                GridVariable::U,
                128,
            )
            .unwrap();
            assert_eq!(
                result,
                Float::with_val(128, Rational::from((numerator, 64)))
            );
        }
    }

    #[test]
    fn invalid_precision_and_nonfinite_hp_samples_are_errors() {
        let a = Float::with_val(128, 1);
        let b = Float::with_val(128, 2);
        let minimum = rug::float::prec_min();
        let constant = uniform_grid_integral(
            |x| Float::with_val(x.prec(), 1),
            &a,
            &b,
            4,
            UniformGridScheme::Midpoint,
            GridVariable::U,
            minimum,
        )
        .unwrap();
        assert_eq!(constant, 1);
        assert_eq!(constant.prec(), minimum);
        for precision in [rug::float::prec_min() - 1, rug::float::prec_max(), u32::MAX] {
            assert!(uniform_grid_integral(
                |x| x.clone(),
                &a,
                &b,
                4,
                UniformGridScheme::Midpoint,
                GridVariable::U,
                precision
            )
            .is_err());
        }
        for bad in [Special::Nan, Special::Infinity, Special::NegInfinity] {
            assert!(uniform_grid_integral(
                |x| Float::with_val(x.prec(), bad),
                &a,
                &b,
                4,
                UniformGridScheme::Midpoint,
                GridVariable::U,
                128
            )
            .is_err());
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn step_count_above_u32_keeps_the_first_midpoint_correct() {
        struct StopAfterFirst;
        let point = std::cell::RefCell::new(None);
        let steps = (1usize << 32) + 1;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = uniform_grid_integral(
                |x| {
                    *point.borrow_mut() = Some(x.clone());
                    std::panic::panic_any(StopAfterFirst)
                },
                &Float::with_val(256, 0),
                &Float::with_val(256, 1),
                steps,
                UniformGridScheme::Midpoint,
                GridVariable::U,
                256,
            );
        }));
        assert!(outcome.unwrap_err().is::<StopAfterFirst>());
        let expected = Float::with_val(288, Rational::from((1, 2 * steps)));
        assert_eq!(point.into_inner().unwrap(), expected);
    }
}
