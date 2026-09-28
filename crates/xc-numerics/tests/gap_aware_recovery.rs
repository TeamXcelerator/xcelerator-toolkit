#![cfg(feature = "hp")]
use rug::{ops::Pow, Float};
use xc_numerics::eigen::*;
fn f(p: u32, x: i32) -> Float {
    Float::with_val(p, x)
}
fn analytic_sine(v: &[Float], a: i32, b: i32) -> Float {
    let p = 1024;
    let mut norm = f(p, 0);
    for x in v {
        norm += Float::with_val(p, x * x);
    }
    let mut perpendicular = Float::with_val(p, &v[0] * b);
    perpendicular -= Float::with_val(p, &v[1] * a);
    let mut square = Float::with_val(p, &perpendicular * &perpendicular);
    square /= a * a + b * b;
    for x in &v[2..] {
        square += Float::with_val(p, x * x);
    }
    square /= norm;
    square.sqrt()
}
#[test]
fn close_analytic_pairs_bind_actual_vector_angle_in_all_routes() {
    // First case is an integer rotation; others are independent rotations/gaps.
    for (a, b, exponent, p) in [
        (21, 20, -122, 128),
        (5, 12, -100, 128),
        (9, 40, -210, 256),
        (17, 8, -75, 128),
    ] {
        let y = f(p, 2).pow(exponent);
        let x = Float::with_val(p, 0.5);
        let mut d0 = Float::with_val(p, &y * (b * b));
        d0 += &x;
        let mut d1 = Float::with_val(p, &y * (a * a));
        d1 += &x;
        let off = Float::with_val(p, &y * (-a * b));
        let d = vec![d0, d1, f(p, -1), f(p, 2)];
        let e = vec![off, f(p, 0), f(p, 0)];
        let mut dense = vec![f(p, 0); 16];
        for i in 0..4 {
            dense[4 * i + i] = d[i].clone();
        }
        dense[1] = e[0].clone();
        dense[4] = e[0].clone();
        for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
            let report = tridiag_eigenvector_for_value_detailed_hp(
                &d,
                &e,
                &x,
                None,
                p,
                TridiagEigvecOptions {
                    solver,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(report.index, 1);
            let actual = analytic_sine(&report.eigenvector, a, b);
            assert!(
                actual
                    <= Float::with_val(1024, &report.sine_angle_upper_bound) + f(1024, 2).pow(-900),
                "analytic angle exceeds directed bound"
            );
            assert!(
                actual < f(1024, 2).pow(-((p / 3) as i32)),
                "resolved pair remained inaccurate: {actual}"
            );
        }

        let selected = tridiag_selected_eigenpairs_hp(
            &d,
            &e,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: 1,
                last_index: 1,
                absolute_tolerance: f(p, 2).pow(-(p as i32)),
                maximum_bisection_iterations: 800,
                eigenvector_options: Default::default(),
                precision_bits: p,
            },
        )
        .unwrap();
        let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &selected.items[0] else {
            panic!("analytic simple member was lost")
        };
        let actual = analytic_sine(&pair.eigenvector, a, b);
        assert!(
            actual
                <= Float::with_val(1024, &pair.recovery.sine_angle_upper_bound)
                    + f(1024, 2).pow(-900)
        );
        assert!(actual < f(1024, 2).pow(-((p / 3) as i32)));
        let report = dense_symmetric_eigenpair_at_index_hp(&dense, 4, 1, p, 200).unwrap();
        let actual = analytic_sine(&report.eigenvector, a, b);
        assert!(
            actual <= Float::with_val(1024, &report.sine_angle_upper_bound) + f(1024, 2).pow(-900)
        );
        assert!(actual < f(1024, 2).pow(-((p / 3) as i32)));
    }
}
#[test]
fn dense_and_selected_recovery_do_not_hide_reflection_odd_zero_member() {
    for exponent in [-54, -58, -43] {
        let p = 128;
        let b: Float = f(p, 2).pow(exponent);
        let d = vec![f(p, 0), f(p, -1), f(p, 0)];
        let e = vec![b.clone(), b.clone()];
        let a = vec![
            f(p, 0),
            b.clone(),
            f(p, 0),
            b.clone(),
            f(p, -1),
            b.clone(),
            f(p, 0),
            b,
            f(p, 0),
        ];
        let dense =
            dense_symmetric_eigenvector_for_value_detailed_hp(&a, 3, &f(p, 0), p, 200).unwrap();
        let selected = tridiag_selected_eigenpairs_hp(
            &d,
            &e,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: 1,
                last_index: 1,
                absolute_tolerance: f(p, 2).pow(-160),
                maximum_bisection_iterations: 600,
                eigenvector_options: Default::default(),
                precision_bits: p,
            },
        )
        .unwrap();
        let HpSelectedTridiagonalItem::SimpleEigenpair(selected) = &selected.items[0] else {
            panic!("simple exact member was lost")
        };
        for report in [&dense, &selected.recovery] {
            assert_eq!(report.index, 1);
            let mut even = Float::with_val(512, &report.eigenvector[0] + &report.eigenvector[2]);
            even.abs_mut();
            assert!(even < f(512, 2).pow(-110));
            assert!(report.eigenvector[1].clone().abs() < f(p, 2).pow(-110));
            assert!(report.sine_angle_upper_bound < f(report.working_precision_bits, 2).pow(-8));
        }
    }
}
#[test]
fn exact_multiplicity_and_wrong_member_fail_with_typed_unresolved_status() {
    let p = 128;
    let a = vec![f(p, 1), f(p, 0), f(p, 0), f(p, 1)];
    let error = dense_symmetric_eigenpair_at_index_hp(&a, 2, 0, p, 200).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<HpEigenvectorRecoveryFailure>(),
        Some(HpEigenvectorRecoveryFailure::UnresolvedEigenspace(_))
    ));
    let a = vec![f(p, 1), f(p, 0), f(p, 0), f(p, 2)];
    assert!(dense_symmetric_eigenvector_for_value_hp(&a, 2, &f(p, 10), p, 200).is_err());
}

#[test]
fn exact_zero_endpoint_eigenvalues_remain_available_with_angle_evidence() {
    for (n, p) in [(9, 64), (11, 64), (21, 64), (7, 96), (13, 128)] {
        let d = vec![f(p, 0); n];
        let e = vec![f(p, 1); n - 1];
        let result = tridiag_selected_eigenpairs_hp(
            &d,
            &e,
            &HpSelectedTridiagonalEigenpairOptions {
                first_index: (n - 1) / 2,
                last_index: (n - 1) / 2,
                absolute_tolerance: f(p, 2).pow(-((p / 2) as i32)),
                maximum_bisection_iterations: 512,
                eigenvector_options: Default::default(),
                precision_bits: p,
            },
        )
        .unwrap();
        let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &result.items[0] else {
            panic!("exact simple zero became a cluster")
        };
        assert!(pair.enclosure.lower <= 0 && pair.enclosure.upper >= 0);
        let w = 512;
        let mut projection = f(w, 0);
        let mut norm = f(w, 0);
        for (i, x) in pair.eigenvector.iter().enumerate() {
            norm += Float::with_val(w, x * x);
            if i % 2 == 0 {
                if i % 4 == 0 {
                    projection += x;
                } else {
                    projection -= x;
                }
            }
        }
        projection /= n.div_ceil(2);
        let mut error = f(w, 0);
        for (i, x) in pair.eigenvector.iter().enumerate() {
            let expected = if i % 2 == 1 {
                f(w, 0)
            } else if i % 4 == 0 {
                projection.clone()
            } else {
                -projection.clone()
            };
            error += Float::with_val(w, x - expected).square();
        }
        let sine = (error / norm).sqrt();
        assert!(
            sine <= Float::with_val(w, &pair.recovery.sine_angle_upper_bound) + f(w, 2).pow(-400)
        );
        assert!(
            sine < f(w, 2).pow(-((p / 3) as i32)),
            "n={n} p={p} sine={sine}"
        );
    }
}
