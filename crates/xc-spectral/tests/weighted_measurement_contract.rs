use xc_numerics::grid_integral::{GridVariable, UniformGridScheme};
use xc_spectral::distance::{
    weighted_alpha_distance_f64, weighted_alpha_norm_f64, WeightedIntegrationRule,
    WeilEigenfunctionF64,
};

fn rule(case: &serde_json::Value) -> WeightedIntegrationRule {
    let variable = if case["variable"] == "u" {
        GridVariable::U
    } else {
        GridVariable::LogU
    };
    match case["rule"].as_str().unwrap() {
        "gl1" => WeightedIntegrationRule::GaussLegendre {
            points: 1,
            variable,
        },
        "gl2" => WeightedIntegrationRule::GaussLegendre {
            points: 2,
            variable,
        },
        "gl3" => WeightedIntegrationRule::GaussLegendre {
            points: 3,
            variable,
        },
        name => WeightedIntegrationRule::UniformGrid {
            scheme: match name {
                "left" => UniformGridScheme::LeftRiemann,
                "right" => UniformGridScheme::RightRiemann,
                "midpoint" => UniformGridScheme::Midpoint,
                "trapezoid" => UniformGridScheme::Trapezoid,
                _ => panic!("fixture rule"),
            },
            variable,
            steps: 4,
        },
    }
}
fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/weighted_measurement_oracle.json")).unwrap()
}
#[test]
fn native_weighted_rules_match_independent_finite_sums() {
    for case in oracle()["cases"].as_array().unwrap() {
        let lam = case["lam"].as_str().unwrap().parse().unwrap();
        let alpha = case["alpha"].as_str().unwrap().parse().unwrap();
        let constant = case["function"] == "constant";
        let got = weighted_alpha_distance_f64(
            |u| if constant { 1. } else { 3. * u * u + 1. },
            |u| if constant { 0. } else { u + 1. },
            lam,
            alpha,
            rule(case),
        )
        .unwrap()
        .value;
        let expected = case["value"].as_str().unwrap().parse::<f64>().unwrap();
        assert!((got / expected - 1.).abs() < 5e-13, "{case}: {got}");
    }
}
#[test]
fn native_weighted_range_recovery_and_explicit_failures() {
    let rule = WeightedIntegrationRule::GaussLegendre {
        points: 1,
        variable: GridVariable::U,
    };
    let got = weighted_alpha_norm_f64(|_| 1e300, 2., 2000., rule)
        .unwrap()
        .value;
    // Independently calculate a ratio in two representable stages.
    let expected = (1e150 * 1.5_f64.powf(-1000.)).powi(2);
    assert!((got / expected - 1.).abs() < 3e-13 && got > 0.);
    let recovered = weighted_alpha_distance_f64(|_| f64::MAX, |_| -f64::MAX, 2., 1000., rule)
        .unwrap()
        .value;
    let expected = (f64::MAX * 1.5_f64.powf(-1000.)) * 2.;
    assert!((recovered / expected - 1.).abs() < 3e-13);
    assert_eq!(
        weighted_alpha_distance_f64(|_| 1., |_| 1., 2., -2000., rule)
            .unwrap()
            .value,
        0.
    );
    assert!(weighted_alpha_norm_f64(|_| 1., 2., 2000., rule).is_err());
    assert!(weighted_alpha_distance_f64(|_| f64::NAN, |_| f64::NAN, 2., 0., rule).is_err());
    let invalid_rule = WeightedIntegrationRule::GaussLegendre {
        points: usize::MAX,
        variable: GridVariable::U,
    };
    assert!(weighted_alpha_norm_f64(
        |_| panic!("invalid order must fail before callback"),
        2.,
        0.,
        invalid_rule
    )
    .is_err());
    let minimum = f64::from_bits(1);
    assert!(weighted_alpha_norm_f64(|_| minimum, 1.125, 0., rule).is_err());
}
#[test]
fn normalized_native_cosine_sum_is_invariant_under_extreme_common_scales() {
    for exponent in [-1070, -1000, -600, 0, 900, 1022] {
        let scale = if exponent == -1070 {
            f64::from_bits(16)
        } else {
            2.0_f64.powi(exponent)
        };
        let f = WeilEigenfunctionF64::from_v_basis(&[scale, 3. * scale, scale], 1, 2.).unwrap();
        let replay =
            WeilEigenfunctionF64::from_normalized_coefficients(&f.normalized_coefficients(), 2.)
                .unwrap();
        for (u, expected) in [(1., 1.), (2.0_f64.sqrt(), 3.), (2., 5.)] {
            assert!(
                (f.eval(u) - expected).abs() < 2e-13,
                "exponent={exponent},u={u}"
            );
            assert!((replay.eval(u) - expected).abs() < 2e-13);
        }
    }
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::Float;
    use xc_spectral::distance::hp::{
        weighted_alpha_distance, weighted_alpha_norm, WeilEigenfunction,
    };
    fn parse(s: &str, p: u32) -> Float {
        Float::with_val(p, Float::parse(s).unwrap())
    }
    #[test]
    fn hp_weighted_rules_match_independent_finite_sums_with_low_precision_callbacks() {
        let p = 256;
        for case in oracle()["cases"].as_array().unwrap() {
            let lam = parse(case["lam"].as_str().unwrap(), p);
            let alpha = parse(case["alpha"].as_str().unwrap(), p);
            let constant = case["function"] == "constant";
            let got = weighted_alpha_distance(
                |u| {
                    if constant {
                        Float::with_val(53, 1)
                    } else {
                        Float::with_val(u.prec(), u * u) * 3u32 + 1u32
                    }
                },
                |u| {
                    if constant {
                        Float::with_val(53, 0)
                    } else {
                        u.clone() + 1u32
                    }
                },
                &lam,
                &alpha,
                rule(case),
                p,
            )
            .unwrap()
            .value;
            let expected = parse(case["value"].as_str().unwrap(), p);
            let error = (Float::with_val(p, &got - &expected) / expected).abs();
            assert!(error < Float::with_val(p, 1) >> 230, "{case}: {error}");
            assert_eq!(got.prec(), p);
        }
    }
    #[test]
    fn hp_weighted_range_precision_and_domain_contracts() {
        let p = 256;
        let lam = Float::with_val(p, 2);
        let rule = WeightedIntegrationRule::GaussLegendre {
            points: 1,
            variable: GridVariable::U,
        };
        let got = weighted_alpha_norm(
            |_| Float::with_val(53, 1),
            &lam,
            &Float::with_val(p, 0.5),
            rule,
            p,
        )
        .unwrap()
        .value;
        let expected = (Float::with_val(512, 2) / 3u32).sqrt();
        assert!(Float::with_val(512, &got - &expected).abs() < Float::with_val(512, 1) >> 250);
        let large = Float::with_val(p, 1) << 900_000_000u32;
        let got = weighted_alpha_norm(
            |_| large.clone(),
            &lam,
            &Float::with_val(p, 2_000_000_000u32),
            rule,
            p,
        )
        .unwrap()
        .value;
        let mut log_reference = Float::with_val(512, 2).ln() * 900_000_000u32;
        log_reference -= Float::with_val(512, 1.5).ln() * 2_000_000_000u32;
        let expected = log_reference.exp();
        assert!(
            (Float::with_val(512, &got / &expected) - 1u32).abs() < Float::with_val(512, 1) >> 240
        );
        assert!(!got.is_zero());
        for bad in [0, 1, 31, 1_000_001, u32::MAX] {
            assert!(weighted_alpha_norm(
                |_| panic!("invalid precision must fail before callback"),
                &lam,
                &Float::with_val(p, 0),
                rule,
                bad
            )
            .is_err());
        }
        let invalid_rule = WeightedIntegrationRule::GaussLegendre {
            points: usize::MAX,
            variable: GridVariable::U,
        };
        assert!(weighted_alpha_norm(
            |_| panic!("invalid order must fail before callback"),
            &lam,
            &Float::with_val(p, 0),
            invalid_rule,
            p
        )
        .is_err());
        let collapsed_lambda = Float::with_val(512, 1) + (Float::with_val(512, 1) >> 400u32);
        assert!(weighted_alpha_norm(
            |_| panic!("collapsed bounds must fail before callback"),
            &collapsed_lambda,
            &Float::with_val(p, 0),
            rule,
            32
        )
        .is_err());
        let huge_alpha = Float::with_val(p, -2_000_000_000i32);
        assert!(weighted_alpha_distance(
            |_| Float::with_val(53, 1),
            |_| Float::with_val(53, 1),
            &lam,
            &huge_alpha,
            rule,
            p
        )
        .unwrap()
        .value
        .is_zero());
    }
    #[test]
    fn hp_difference_at_exponent_floor_uses_exact_binary_rescaling() {
        let p = 256;
        let emin = rug::float::exp_min();
        let left = Float::with_val(p, 1.5) << emin;
        let right: Float = (Float::with_val(p, 1.5) + (Float::with_val(p, 1) >> 255u32)) << emin;
        assert_ne!(left, right);
        let rule = WeightedIntegrationRule::GaussLegendre {
            points: 1,
            variable: GridVariable::U,
        };
        let value = weighted_alpha_distance(
            |_| left.clone(),
            |_| right.clone(),
            &Float::with_val(p, 2),
            &Float::with_val(p, -512),
            rule,
            p,
        )
        .unwrap()
        .value;
        let log_reference =
            Float::with_val(512, 2).ln() * (emin - 255) + Float::with_val(512, 1.5).ln() * 512u32;
        let expected = log_reference.exp();
        let error = (Float::with_val(512, &value / &expected) - 1u32).abs();
        assert!(
            error < Float::with_val(512, 1) >> 240,
            "relative error={error}"
        );
    }

    #[test]
    fn hp_uniform_trapezoid_promotes_endpoint_values_before_averaging() {
        let p = 256;
        let value = xc_numerics::grid_integral::hp::uniform_grid_integral(
            |u| {
                let mut value = Float::with_val(53, 1);
                if u > &1 {
                    value += Float::with_val(53, 1) >> 52;
                }
                value
            },
            &Float::with_val(p, 1),
            &Float::with_val(p, 2),
            1,
            UniformGridScheme::Trapezoid,
            GridVariable::U,
            p,
        )
        .unwrap();
        assert_eq!(value, Float::with_val(p, 1) + (Float::with_val(p, 1) >> 53));
    }
    #[test]
    fn hp_normalized_cosine_sum_is_invariant_under_extreme_common_scales() {
        let p = 256;
        let lam = Float::with_val(p, 2);
        for exponent in [
            rug::float::exp_min() + 8,
            -1000,
            0,
            1000,
            rug::float::exp_max() - 2,
        ] {
            let scale = Float::with_val(p, 1) << exponent;
            let f = WeilEigenfunction::from_v_basis(
                &[scale.clone(), scale.clone() * 3u32, scale],
                1,
                &lam,
                p,
            )
            .unwrap();
            let replay = WeilEigenfunction::from_normalized_coefficients(
                &f.normalized_coefficients(),
                &lam,
                p,
            )
            .unwrap();
            for (u, expected) in [
                (Float::with_val(p, 1), 1u32),
                (lam.clone().sqrt(), 3),
                (lam.clone(), 5),
            ] {
                assert!(
                    (f.eval(&u) - expected).abs() < Float::with_val(p, 1) >> 240,
                    "exponent={exponent}"
                );
                assert!((replay.eval(&u) - expected).abs() < Float::with_val(p, 1) >> 240);
            }
        }
    }
}
