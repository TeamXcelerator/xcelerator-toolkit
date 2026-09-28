use xc_numerics::grid_integral::{uniform_grid_integral_f64, GridVariable, UniformGridScheme};

#[test]
fn native_grid_samples_remain_inside_exact_declared_endpoints() {
    let a = 1.9431114014114512;
    let b = 6.124251869373814;
    uniform_grid_integral_f64(
        |u| {
            assert!(u >= a && u <= b);
            (b - u).sqrt()
        },
        a,
        b,
        34,
        UniformGridScheme::RightRiemann,
        GridVariable::U,
    )
    .unwrap();
    let a = 2.766374348818396;
    let b = 7.6675950788258564;
    for scheme in [
        UniformGridScheme::LeftRiemann,
        UniformGridScheme::RightRiemann,
        UniformGridScheme::Trapezoid,
    ] {
        uniform_grid_integral_f64(
            |u| {
                assert!(u >= a && u <= b);
                (u - a).sqrt()
            },
            a,
            b,
            34,
            scheme,
            GridVariable::LogU,
        )
        .unwrap();
    }
}

#[test]
fn native_symmetric_contract_ignores_unread_upper_triangle() {
    let a = [2., 1., f64::NAN, 2.];
    let mut q = [1., 0., 0., 1.];
    let values =
        xc_numerics::symmetric_f64::complete_symmetric_eigensystem_f64(&a, 2, &mut q).unwrap();
    assert!((values[0] - 1.).abs() < 1e-14 && (values[1] - 3.).abs() < 1e-14);
    assert!(
        xc_numerics::symmetric_f64::complete_symmetric_eigensystem_f64(
            &[2., f64::NAN, 1., 2.],
            2,
            &mut q
        )
        .is_err()
    );
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{ops::Pow, Float, Rational};
    use xc_numerics::{eigen::*, linalg, mpfr_interval::MpfrInterval as I};
    fn f(p: u32, n: i32) -> Float {
        Float::with_val(p, n)
    }
    #[test]
    fn hp_grid_pins_right_endpoint_and_log_endpoints() {
        let p = 53;
        let a = Float::with_val(p, Float::parse("0.3").unwrap());
        let b = Float::with_val(p, 10) / 3;
        for variable in [GridVariable::U, GridVariable::LogU] {
            for scheme in [
                UniformGridScheme::LeftRiemann,
                UniformGridScheme::RightRiemann,
                UniformGridScheme::Trapezoid,
            ] {
                xc_numerics::grid_integral::hp::uniform_grid_integral(
                    |u| {
                        assert!(u >= &a && u <= &b);
                        Float::with_val(u.prec(), &b - u).sqrt()
                    },
                    &a,
                    &b,
                    19,
                    scheme,
                    variable,
                    p,
                )
                .unwrap();
            }
        }
    }
    #[test]
    fn close_pair_is_not_deflated_at_65536_working_units() {
        for p in [64, 128, 256] {
            let e = f(p, 2).pow(17 - p as i32);
            let d = vec![f(p, 1); 2];
            let expected = [
                Float::with_val(p, &d[0] - &e),
                Float::with_val(p, &d[0] + &e),
            ];
            let qr = tridiag_eigenvalues_hp(&d, std::slice::from_ref(&e), p).unwrap();
            let a = vec![d[0].clone(), e.clone(), e, d[0].clone()];
            let jacobi =
                xc_numerics::eigen::dense_symmetric_eigenvalues_jacobi_hp(&a, 2, p, 100).unwrap();
            for values in [&qr, &jacobi.eigenvalues] {
                for (actual, reference) in values.iter().zip(&expected) {
                    assert!(
                        Float::with_val(p, actual - reference).abs() <= f(p, 2).pow(3 - p as i32)
                    );
                }
            }
        }
    }
    #[test]
    fn default_selected_recovery_sees_the_odd_double_well_state() {
        let p = 128;
        let n = 21;
        let d = (0..n)
            .map(|i| f(p, if (9..=11).contains(&i) { 32 } else { 2 }))
            .collect::<Vec<_>>();
        let e = vec![f(p, -1); n - 1];
        let result = tridiag_selected_eigenpairs_hp(
            &d,
            &e,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: 1,
                last_index: 1,
                absolute_tolerance: Float::with_val(p, Float::parse("1.5e-7").unwrap()),
                maximum_bisection_iterations: 200,
                eigenvector_options: TridiagEigvecOptions::default(),
                precision_bits: p,
            },
        )
        .unwrap();
        let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &result.items[0] else {
            panic!("simple odd state expected")
        };
        assert!(pair.residual_norm < f(p, 2).pow(-100));
        for i in 0..n {
            assert!(
                Float::with_val(p, &pair.eigenvector[i] + &pair.eigenvector[n - 1 - i]).abs()
                    < f(p, 2).pow(-80)
            );
        }
    }
    #[test]
    fn shifted_recovery_is_scale_equivariant_on_exact_two_by_two() {
        let p = 128;
        for k in [-120, 0, 33, 100] {
            let scale: Float = f(p, 2).pow(k);
            let d = vec![Float::with_val(p, &scale * 2); 2];
            let e = vec![scale.clone()];
            let value = Float::with_val(p, &scale * 3);
            for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
                let v = tridiag_eigenvector_for_value_hp(
                    &d,
                    &e,
                    &value,
                    p,
                    TridiagEigvecOptions {
                        solver,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(Float::with_val(p, &v[0] - &v[1]).abs() < f(p, 2).pow(-100));
            }
        }
    }
    #[test]
    fn sturm_dimension_guard_recovers_exact_counts_for_long_chains() {
        // Independent integer determinant recurrences at rational thresholds
        // give 1 and 113 strict-below roots, respectively, for these n=1000 chains.
        for (p, diagonal, off, threshold, expected) in [
            (256, 2, -1, Rational::from((1, 100000)), 1),
            (128, 0, 1, Rational::from((-15, 8)), 113),
        ] {
            let n = 1000;
            let d = vec![f(p, diagonal); n];
            let e = vec![f(p, off); n - 1];
            let t = Float::with_val(p, &threshold);
            assert_eq!(
                tridiag_sturm_count_below_hp(&d, &e, &t, p).unwrap(),
                expected
            );
        }
    }
    #[test]
    fn dense_inverse_iteration_uses_matrix_scaled_residual_stopping() {
        let p = 128;
        for k in [-120, 0, 100] {
            let scale: Float = f(p, 2).pow(k);
            // Exact SPD eigenpairs: (1,(1,-1)), (3,(1,1)). The asymmetric
            // start overlaps both; only the smaller eigenspace is acceptable.
            let a = [2, 1, 1, 2].map(|x| Float::with_val(p, &scale * x));
            let output = linalg::inverse_iteration_from_detailed(
                &a,
                2,
                p,
                300,
                false,
                Some(vec![f(p, 1), f(p, 0)]),
            )
            .unwrap();
            let relative =
                Float::with_val(p, &output.diagnostics.final_relative_residual_norm / &scale);
            assert!(relative < f(p, 2).pow(-100));
            assert!(
                Float::with_val(p, &output.eigenvector[0] + &output.eigenvector[1]).abs()
                    < f(p, 2).pow(-100)
            );
            assert!(output.diagnostics.unshifted_steps <= 300);
        }
    }
    #[test]
    fn mixed_precision_intervals_fail_closed_and_exact_encodings_are_nonunique() {
        let a = I::from_i64(1, 64);
        let b = I::from_i64(1, 128);
        assert!(a.add(&b).validate().is_err());
        assert!(a.sub(&b).validate().is_err());
        assert!(a.mul(&b).validate().is_err());
        assert!(!a.is_subset_of(&b));
        assert!(!a.is_interior_subset_of(&b));
        use xc_numerics::fmt::PortableHpFloat;
        let x = PortableHpFloat {
            precision_bits: 64,
            significand_hex: "2".into(),
            binary_exponent: 0,
            negative_zero: false,
        };
        let y = PortableHpFloat {
            significand_hex: "1".into(),
            binary_exponent: 1,
            ..x.clone()
        };
        assert_ne!(x, y);
        assert_eq!(x.to_float().unwrap(), y.to_float().unwrap());
        assert_eq!(
            xc_numerics::fmt::display_hp(&Float::with_val(64, rug::float::Special::NegZero), 0),
            "-0"
        );
    }
    #[test]
    fn repeated_root_outside_window_preserves_distinct_root_isolation() {
        // (x-10)^2(x-1) = x^3 -21x^2 +120x -100.
        let coefficients = [-100, 120, -21, 1].map(Rational::from);
        let count =
            xc_numerics::interval::exact_sturm_root_count(&coefficients, 0.into(), 2.into())
                .unwrap();
        assert_eq!(count.distinct_real_roots, 1);
        let intervals = xc_numerics::interval::exact_sturm_isolate_roots(
            &coefficients,
            0.into(),
            2.into(),
            Rational::from((1, 100)),
            100,
        )
        .unwrap();
        assert_eq!(intervals.len(), 1);
        assert!(intervals[0].contains(&Rational::from(1)));
        assert!(intervals[0].width() <= Rational::from((1, 100)));
    }
}
