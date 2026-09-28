#![cfg(feature = "hp")]
use rug::{float::Constant, Float};
use xc_numerics::eigen::*;

#[test]
fn requested_precision_rounding_matches_explicit_source_conversion() {
    let p = 64;
    // This high-precision coefficient actually loses a nonzero bit at p.
    let off: Float = Float::with_val(256, 1) + (Float::with_val(256, 1) >> 100);
    let rounded = Float::with_val(p, &off);
    assert_ne!(off, rounded);
    let d = vec![Float::with_val(256, 2); 2];
    let value = Float::with_val(p, 1);
    for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
        let opts = TridiagEigvecOptions {
            solver,
            max_steps: 32,
            early_termination: false,
        };
        let actual = tridiag_eigenvector_for_value_detailed_hp(
            &d,
            std::slice::from_ref(&off),
            &value,
            None,
            p,
            opts,
        )
        .unwrap();
        let expected = tridiag_eigenvector_for_value_detailed_hp(
            &vec![Float::with_val(p, 2); 2],
            std::slice::from_ref(&rounded),
            &value,
            None,
            p,
            opts,
        )
        .unwrap();
        assert_eq!(actual, expected);
        // Exact eigen-equation for the rounded [[2,1],[1,2]] working source.
        let x = actual.eigenvector[0].to_rational().unwrap();
        let y = actual.eigenvector[1].to_rational().unwrap();
        assert_eq!(x + y, 0);
    }
    let a = [d[0].clone(), off.clone(), off, d[1].clone()];
    let actual = dense_symmetric_eigenpair_at_index_hp(&a, 2, 0, p, 32).unwrap();
    let rounded = [2, 1, 1, 2].map(|x| Float::with_val(p, x));
    let expected = dense_symmetric_eigenpair_at_index_hp(&rounded, 2, 0, p, 32).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn exponent_extreme_qr_matches_closed_form_strang_spectrum() {
    let p = 128;
    let n = 5;
    for exponent in [-700_000_000i32, 0, 700_000_000] {
        let d = vec![Float::with_val(p, 2) << exponent; n];
        let e = vec![Float::with_val(p, -1) << exponent; n - 1];
        let values = tridiag_eigenvalues_hp(&d, &e, p).unwrap();
        for (i, value) in values.iter().enumerate() {
            let angle = Float::with_val(512, Constant::Pi) * (i + 1) / (n + 1);
            let expected: Float = Float::with_val(512, 2) - angle.cos() * 2;
            let actual: Float = Float::with_val(512, value) >> exponent;
            assert!((actual - expected).abs() < Float::with_val(512, 1) >> 110);
        }
    }
}

#[test]
fn exponent_extreme_recovery_counts_preserve_exact_quadratic_eigenspace() {
    let p = 128;
    let lambda: Float = (Float::with_val(512, 5) - Float::with_val(512, 5).sqrt()) / 2;
    let slope: Float = Float::with_val(512, &lambda) - 2;
    for exponent in [-700_000_000i32, 0, 700_000_000] {
        let d = [2, 3].map(|x| Float::with_val(p, x) << exponent);
        let e = [Float::with_val(p, 1) << exponent];
        let value = Float::with_val(p, &lambda) << exponent;
        for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
            let result = tridiag_eigenvector_for_value_detailed_hp(
                &d,
                &e,
                &value,
                None,
                p,
                TridiagEigvecOptions {
                    solver,
                    max_steps: 32,
                    early_termination: true,
                },
            )
            .unwrap();
            let actual = Float::with_val(512, &result.eigenvector[1] / &result.eigenvector[0]);
            assert!((actual - &slope).abs() < Float::with_val(512, 1) >> 110);
            assert!(
                Float::with_val(512, &result.residual_upper_bound) >> exponent
                    < Float::with_val(512, 1) >> 110
            );
            assert!(result.sine_angle_upper_bound <= result.angle_acceptance_bound);
        }
        let a = [d[0].clone(), e[0].clone(), e[0].clone(), d[1].clone()];
        let result = dense_symmetric_eigenpair_at_index_hp(&a, 2, 0, p, 32).unwrap();
        let actual = Float::with_val(512, &result.eigenvector[1] / &result.eigenvector[0]);
        assert!((actual - &slope).abs() < Float::with_val(512, 1) >> 110);
    }
}

#[test]
fn exponent_extreme_recovery_counts_preserve_exact_projector_with_repeated_neighbors() {
    let p = 128;
    // A=I-vv^T/(v^Tv), v=(0,0,1,1). Exact spectrum {0,1,1,1}.
    for exponent in [-500_000_000i32, 0, 500_000_000] {
        let d = [1, 1, 0, 0].map(|x| Float::with_val(p, x) << exponent);
        let d = [
            d[0].clone(),
            d[1].clone(),
            Float::with_val(p, 1) << (exponent - 1),
            Float::with_val(p, 1) << (exponent - 1),
        ];
        let e = [
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            -Float::with_val(p, 1) << (exponent - 1),
        ];
        for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
            let result = tridiag_eigenvector_for_value_detailed_hp(
                &d,
                &e,
                &Float::with_val(p, 0),
                None,
                p,
                TridiagEigvecOptions {
                    solver,
                    max_steps: 32,
                    early_termination: true,
                },
            )
            .unwrap();
            let v = &result.eigenvector;
            assert!(v[0].clone().abs() < Float::with_val(p, 1) >> 110);
            assert!(v[1].clone().abs() < Float::with_val(p, 1) >> 110);
            assert!(Float::with_val(p, &v[2] - &v[3]).abs() < Float::with_val(p, 1) >> 110);
            assert_eq!(result.index, 0);
            assert!(result.sine_angle_upper_bound <= result.angle_acceptance_bound);
        }
    }
}
