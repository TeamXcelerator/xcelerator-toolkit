use xc_numerics::grid_integral::{uniform_grid_integral_f64, GridVariable, UniformGridScheme};
#[test]
fn native_log_grid_has_relative_span_accuracy_and_gaussian_tails_are_admitted() {
    for (a, b) in [
        (1e6, 1e6 + 1.),
        (2., 2. + 2f64.powi(-30)),
        (7., 7. + 2f64.powi(-28)),
    ] {
        let h = ((b - a) / a).ln_1p() / 64.;
        let left = a * h * ((b - a) / a) / h.exp_m1();
        for (scheme, reference) in [
            (UniformGridScheme::LeftRiemann, left),
            (UniformGridScheme::RightRiemann, left * h.exp()),
            (UniformGridScheme::Trapezoid, left * (1. + h.exp()) / 2.),
        ] {
            let got =
                uniform_grid_integral_f64(|_| 1., a, b, 64, scheme, GridVariable::LogU).unwrap();
            assert!(
                ((got - reference) / reference).abs() < 8. * f64::EPSILON,
                "{a} {b} {got} {reference}"
            );
        }
    }
    for (width, n) in [(30., 128), (32., 256), (40., 512)] {
        let got = xc_numerics::quadrature::try_gauss_legendre_npt_f64(
            |x| (-x * x).exp(),
            -width,
            width,
            n,
        )
        .unwrap();
        assert!(got > 1.7 && got < 1.9);
    }
}
#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::{ops::Pow, Float, Rational};
    use xc_numerics::{fmt, interval::*, linalg, prefix};
    fn f(p: u32, x: i32) -> Float {
        Float::with_val(p, x)
    }
    #[test]
    fn diagnostics_keep_precision_at_exponent_floor_and_cap_source_digits() {
        for p in [8, 16, 24, 53, 128, 256, 512] {
            let k = (p / 2).min(50) as i32;
            let mut reference = f(p, 1);
            reference <<= rug::float::exp_min();
            let mut computed = f(p, 1) + f(p, 2).pow(-k);
            computed <<= rug::float::exp_min();
            assert!(Float::with_val(p, &computed - &reference).is_zero());
            let expected = f(p + 128, 2).log10() * k;
            let got = fmt::matching_digits(&computed, &reference);
            assert!(
                (Float::with_val(p + 128, &got) - &expected).abs()
                    < f(p + 128, 8)
                        * f(p + 128, 2).pow(-(p as i32))
                        * expected.clone().abs().max(&f(p + 128, 1)),
                "p={p} got={got} ref={expected}"
            );
            let ratio = fmt::relative_difference(&computed, &reference).unwrap();
            assert_eq!(ratio, f(p, 2).pow(-k));
            let cap = fmt::matching_digits(&reference, &reference);
            assert!(got <= cap);
            let unit = f(p, 1);
            let mut neighbor = unit.clone();
            neighbor.next_up();
            assert!(fmt::matching_digits(&neighbor, &unit) <= fmt::matching_digits(&unit, &unit));
        }
        let a = f(128, 1);
        let b = Float::with_val(256, &a) + f(256, 2).pow(-220);
        assert!(fmt::matching_digits(&a, &b) <= fmt::matching_digits(&a, &a));
    }
    #[test]
    fn hp_relative_log_grid_matches_independent_geometric_sum() {
        for (p, a, m, k) in [(64, 3, 11, 60i32), (128, 5, 7, 100), (192, 11, 13, 160)] {
            let a = f(p, a);
            let b = a.clone() + f(p, m) * f(p, 2).pow(-k);
            let w = p + 256;
            let h: Float = (Float::with_val(w, &b - &a) / &a).ln_1p() / 64;
            let left = Float::with_val(w, &a) * &h * (Float::with_val(w, &b - &a) / &a)
                / h.clone().exp_m1();
            for (scheme, reference) in [
                (UniformGridScheme::LeftRiemann, left.clone()),
                (
                    UniformGridScheme::RightRiemann,
                    left.clone() * h.clone().exp(),
                ),
                (UniformGridScheme::Trapezoid, left * (h.exp() + 1i32) / 2),
            ] {
                let got = xc_numerics::grid_integral::hp::uniform_grid_integral(
                    |u| f(u.prec(), 1),
                    &a,
                    &b,
                    64,
                    scheme,
                    GridVariable::LogU,
                    p,
                )
                .unwrap();
                let error = (Float::with_val(w, &got) - &reference).abs() / reference;
                assert!(
                    error < f(w, 4) * f(w, 2).pow(-(p as i32)),
                    "p={p} error={error}"
                );
            }
        }
    }
    #[test]
    fn native_quadrature_equals_exact_stored_triple_sum() {
        use xc_numerics::quadrature::{try_gauss_legendre_npt_f64, try_gl_nodes_weights_f64};
        for (a, b, n, kind) in [
            (-30., 30., 128, 0),
            (-40., 40., 256, 0),
            (-1e150, 1e150, 64, 1),
            (-9., 13., 97, 2),
        ] {
            let sample = |x: f64| match kind {
                0 => (-x * x).exp(),
                1 => 1e-300,
                _ => x * x * x - 3. * x,
            };
            let (nodes, weights) = try_gl_nodes_weights_f64(n).unwrap();
            let half = (b - a) * 0.5;
            let mid = (a + b) * 0.5;
            let mut exact = Rational::new();
            for (x, w) in nodes.iter().zip(&weights) {
                let y = sample(mid + half * x);
                exact += Rational::from_f64(*w).unwrap()
                    * Rational::from_f64(y).unwrap()
                    * Rational::from_f64(half).unwrap();
            }
            let expected = Float::with_val(8192, &exact).to_f64();
            let got = try_gauss_legendre_npt_f64(sample, a, b, n).unwrap();
            assert_eq!(got.to_bits(), expected.to_bits(), "kind={kind} n={n}");
        }
    }
    #[test]
    fn repeated_polynomials_preserve_distinct_roots_and_contour_work_is_bounded() {
        // (x-2)^2*(x^2-2), repeated roots both outside and inside the interval.
        let coefficients = [-8, 8, 2, -4, 1].map(Rational::from);
        for (lo, hi, expected) in [(-2, 2, 2), (-3, 3, 3)] {
            // Avoid a root on an endpoint for the open-interval contract.
            let hi = if hi == 2 {
                Rational::from((19, 10))
            } else {
                Rational::from(hi)
            };
            let roots = exact_sturm_isolate_roots(
                &coefficients,
                Rational::from(lo),
                hi,
                Rational::from((1, 1000)),
                64,
            )
            .unwrap();
            assert_eq!(roots.len(), expected);
            for interval in roots {
                assert_eq!(
                    exact_sturm_root_count(
                        &coefficients,
                        interval.lower().clone(),
                        interval.upper().clone()
                    )
                    .unwrap()
                    .distinct_real_roots,
                    1
                );
            }
        }
        let coefficients = [
            ComplexRational {
                real: 0.into(),
                imaginary: 0.into(),
            },
            ComplexRational {
                real: 1.into(),
                imaginary: 0.into(),
            },
        ];
        let rectangle =
            RationalContourRectangle::new((-1).into(), 1.into(), (-1).into(), 1.into()).unwrap();
        let good = certify_polynomial_zero_count_on_rectangle(&coefficients, rectangle.clone(), 12)
            .unwrap();
        assert_eq!(good.zero_count, 1);
        let budget = ContourWorkBudget {
            maximum_accepted_cells: 2,
            maximum_segment_evaluations: 2,
            maximum_depth: 12,
        };
        assert!(matches!(
            certify_polynomial_zero_count_on_rectangle_with_budget(
                &coefficients,
                rectangle,
                12,
                &budget
            ),
            Err(IntervalError::Inconclusive(_))
        ));
    }
    #[test]
    fn inverse_iteration_unforced_start_reaches_odd_member_and_tuple_rejects_partial() {
        let p = 192;
        // Reflection-odd [1,0,-1] eigenvalue1; even eigenvalues3 and4.
        let a = [2, 0, 1, 0, 4, 0, 1, 0, 2].map(|x| f(p, x));
        let (value, v) = linalg::inverse_iteration(&a, 3, p, 200, false).unwrap();
        assert!((value - 1i32).abs() < f(p, 2).pow(-80));
        assert!((v[0].clone() + &v[2]).abs() < f(p, 2).pow(-70));
        let d = [f(p, 1), f(p, 0), f(p, 0), f(p, 1) + f(p, 2).pow(-30)];
        assert!(linalg::inverse_iteration(&d, 2, p, 1, false).is_err());
        let mixed = [f(64, 2), f(128, 0), f(64, 0), f(128, 3)];
        assert!(linalg::lu_factor(&mixed, 2).is_err());
        let lu = linalg::lu_factor_at_precision(&mixed, 2, 128).unwrap();
        assert!(linalg::try_lu_solve(&lu, &[f(192, 1), f(192, 1)], 2, 192).is_err());
        let x = linalg::try_lu_solve(&lu, &[f(128, 1), f(128, 1)], 2, 128).unwrap();
        assert_eq!(x[0], f(128, 1) / 2);
    }
    #[test]
    fn equal_two_mode_spectra_never_report_outside_and_fits_expose_conditioning() {
        for p in [64, 71, 128, 191, 256, 509, 638] {
            for c in ["1", "3", "0.1", "7e-30", "1e25", "2"] {
                let c = Float::with_val(p, Float::parse(c).unwrap());
                let a = vec![c.clone(), f(p, 0), f(p, 0), c];
                let r = prefix::analyze_prefixes_with_policy(
                    &a,
                    2,
                    p,
                    8,
                    &[],
                    &prefix::PrefixDiagnosticPolicy::full(),
                )
                .unwrap();
                let fit = &r.rows[1]
                    .third_inverse_moment
                    .as_ref()
                    .unwrap()
                    .two_mode_fit;
                assert_ne!(
                    fit.status,
                    prefix::TwoModeMomentStatus::OutsideTwoModeRange,
                    "p={p} matrix={a:?}"
                );
            }
        }
        for ratio in [2, 8, 100] {
            let p = 192;
            let a = [f(p, 1), f(p, 0), f(p, 0), f(p, ratio)];
            let r = prefix::analyze_prefixes_with_policy(
                &a,
                2,
                p,
                8,
                &[],
                &prefix::PrefixDiagnosticPolicy::full(),
            )
            .unwrap();
            let fit = &r.rows[1]
                .third_inverse_moment
                .as_ref()
                .unwrap()
                .two_mode_fit;
            assert_eq!(fit.status, prefix::TwoModeMomentStatus::Resolved);
            assert!(fit.conditioning_indicator.is_some());
        }
        let p = 128;
        let a = [
            f(p, 1),
            f(p, 0),
            f(p, 0),
            f(p, 0),
            f(p, 1),
            f(p, 0),
            f(p, 0),
            f(p, 0),
            f(p, 1),
        ];
        let r = prefix::analyze_prefixes_with_policy(
            &a,
            3,
            p,
            8,
            &[],
            &prefix::PrefixDiagnosticPolicy::full(),
        )
        .unwrap();
        assert_eq!(
            r.rows[2]
                .third_inverse_moment
                .as_ref()
                .unwrap()
                .two_mode_fit
                .status,
            prefix::TwoModeMomentStatus::OutsideTwoModeRange
        );
    }
    #[test]
    fn failed_runtime_does_not_commit_a_successful_policy_record() {
        let mut provenance = xc_core::SolverProvenance::current_package("rug_mpfr");
        let before = serde_json::to_value(&provenance).unwrap();
        let policy =
            xc_core::HpRuntimePolicy::safe_capped(1, 2 * 1024 * 1024, "manufactured regression")
                .unwrap();
        let result =
            xc_numerics::hp_runtime::run_hp_with_provenance(&policy, 1, &mut provenance, || {
                panic!("manufactured failure")
            });
        assert!(result.is_err());
        assert_eq!(serde_json::to_value(&provenance).unwrap(), before);
    }

    #[test]
    fn original_dense_inverse_iteration_witnesses_keep_the_odd_ground_and_reject_partial_pairs() {
        let p = 128;
        let vectors = [
            [1, 1, 0, 1, 1],
            [1, -1, 0, -1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 0, -1, -1],
            [1, -1, 0, 1, -1],
        ];
        let eigenvalues = [
            Rational::from(1),
            100.into(),
            200.into(),
            Rational::from((3, 4)),
            5.into(),
        ];
        let mut exact = vec![Rational::new(); 25];
        for (v, value) in vectors.iter().zip(&eigenvalues) {
            let norm: i32 = v.iter().map(|x| x * x).sum();
            for i in 0..5 {
                for j in 0..5 {
                    exact[i * 5 + j] += value.clone() * (v[i] * v[j]) / norm;
                }
            }
        }
        let matrix: Vec<_> = exact.iter().map(|x| Float::with_val(p, x)).collect();
        let (value, vector) = linalg::inverse_iteration(&matrix, 5, p, 400, false).unwrap();
        assert!((value - Float::with_val(p, Rational::from((3, 4)))).abs() < f(p, 2).pow(-90));
        let target = &vectors[3];
        let projection: Float = vector
            .iter()
            .zip(target)
            .fold(f(512, 0), |sum, (x, y)| sum + Float::with_val(512, x) * *y)
            / 4;
        let mut square_error = f(512, 0);
        for (x, y) in vector.iter().zip(target) {
            square_error += (Float::with_val(512, x) - projection.clone() * *y).square();
        }
        assert!(square_error.sqrt() < f(512, 2).pow(-90));

        let rows = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]];
        let eigenvalues = [f(p, 1), f(p, 1) + f(p, 2).pow(-8), f(p, 3), f(p, 5)];
        let mut matrix = vec![f(p, 0); 16];
        for (v, value) in rows.iter().zip(eigenvalues) {
            for i in 0..4 {
                for j in 0..4 {
                    matrix[i * 4 + j] += value.clone() * (v[i] * v[j]) / 4;
                }
            }
        }
        for steps in [50, 400] {
            if let Ok((_, vector)) = linalg::inverse_iteration(&matrix, 4, p, steps, false) {
                let projection: Float = vector.iter().fold(f(512, 0), |sum, x| sum + x) / 4;
                let error = vector
                    .iter()
                    .fold(f(512, 0), |sum, x| {
                        sum + (Float::with_val(512, x) - &projection).square()
                    })
                    .sqrt();
                assert!(error < Float::with_val(512, Float::parse("1e-30").unwrap()));
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn original_hilbert_source_is_refactored_before_high_precision_solve() {
        for n in [4usize, 6] {
            let source: Vec<_> = (0..n)
                .flat_map(|i| {
                    (0..n).map(move |j| Float::with_val(53, Rational::from((1, i + j + 1))))
                })
                .collect();
            let factors = linalg::lu_factor(&source, n).unwrap();
            let rhs = vec![f(256, 1); n];
            assert!(linalg::try_lu_solve(&factors, &rhs, n, 256).is_err());
            // Exact Gaussian elimination on the stored dyadic matrix is an
            // independent oracle, with no MPFR factorization or substitution.
            let mut augmented: Vec<Vec<Rational>> = source
                .chunks(n)
                .map(|row| {
                    let mut values: Vec<_> = row.iter().map(|x| x.to_rational().unwrap()).collect();
                    values.push(Rational::from(1));
                    values
                })
                .collect();
            for k in 0..n {
                let pivot = augmented[k][k].clone();
                assert_ne!(pivot, 0);
                for j in k..=n {
                    augmented[k][j] /= &pivot;
                }
                let pivot_row = augmented[k].clone();
                for i in 0..n {
                    if i != k {
                        let multiplier = augmented[i][k].clone();
                        for j in k..=n {
                            augmented[i][j] -= multiplier.clone() * &pivot_row[j];
                        }
                    }
                }
            }
            let factors = linalg::lu_factor_at_precision(&source, n, 256).unwrap();
            let got = linalg::try_lu_solve(&factors, &rhs, n, 256).unwrap();
            for (x, row) in got.iter().zip(&augmented) {
                let expected = Float::with_val(512, &row[n]);
                let relative = (Float::with_val(512, x) - &expected).abs() / expected.abs();
                assert!(relative < f(512, 2).pow(-210), "n={n} relative={relative}");
            }
        }
    }

    #[test]
    fn original_contour_boundary_witnesses_exhaust_explicit_work_instead_of_counting() {
        let rectangle =
            RationalContourRectangle::new(0.into(), 1.into(), 0.into(), 1.into()).unwrap();
        for (coefficients, depth) in [(vec![-1, 3], 3200), (vec![-1, 9, -27, 27], 40)] {
            // The independently known root 1/3 lies on the contour; the
            // triple-root polynomial is exactly (3z-1)^3.
            let coefficients: Vec<_> = coefficients
                .into_iter()
                .map(|x| ComplexRational {
                    real: Rational::from(x),
                    imaginary: Rational::new(),
                })
                .collect();
            let budget = ContourWorkBudget {
                maximum_accepted_cells: 128,
                maximum_segment_evaluations: 256,
                maximum_depth: 4096,
            };
            let result = certify_polynomial_zero_count_on_rectangle_with_budget(
                &coefficients,
                rectangle.clone(),
                depth,
                &budget,
            );
            assert!(
                matches!(result, Err(IntervalError::Inconclusive(ref reason)) if reason.contains("work budget")),
                "{result:?}"
            );
        }
    }

    #[test]
    fn ill_conditioned_original_two_mode_witness_is_flagged_or_has_a_replayable_indicator() {
        for (p, value) in [
            (128, "1e16"),
            (128, "1e18"),
            (128, "1e19"),
            (128, "3e19"),
            (256, "1e19"),
        ] {
            let second = Float::with_val(p, Float::parse(value).unwrap());
            let matrix = [f(p, 1), f(p, 0), f(p, 0), second.clone()];
            let report = prefix::analyze_prefixes_with_policy(
                &matrix,
                2,
                p,
                8,
                &[],
                &prefix::PrefixDiagnosticPolicy::full(),
            )
            .unwrap();
            let fit = &report.rows[1]
                .third_inverse_moment
                .as_ref()
                .unwrap()
                .two_mode_fit;
            if fit.status == prefix::TwoModeMomentStatus::Resolved {
                let ratio = Float::with_val(
                    512,
                    Float::parse(fit.gap_ratio_estimate.as_ref().unwrap()).unwrap(),
                );
                let expected = f(512, 2).pow(-(p as i32)) / ratio.square();
                let indicator = Float::with_val(
                    512,
                    Float::parse(fit.conditioning_indicator.as_ref().unwrap()).unwrap(),
                );
                assert!(
                    (indicator.clone() - &expected).abs() / expected
                        < f(512, 2).pow(-((p - 8) as i32))
                );
                let estimate = Float::with_val(
                    512,
                    Float::parse(fit.second_eigenvalue_estimate.as_ref().unwrap()).unwrap(),
                );
                let actual_error = (estimate - &second).abs() / &second;
                if actual_error > Float::with_val(512, Float::parse("0.01").unwrap()) {
                    assert!(indicator > Float::with_val(512, Float::parse("0.01").unwrap()));
                }
            } else {
                assert!(matches!(
                    fit.status,
                    prefix::TwoModeMomentStatus::UnresolvedSeparation
                        | prefix::TwoModeMomentStatus::UnresolvedArithmetic
                ));
            }
        }
    }
}
