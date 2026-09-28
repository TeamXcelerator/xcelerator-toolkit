#![cfg(feature = "hp")]
use rug::{float::Special, Float};
use xc_numerics::{eigen::*, linalg::*};
#[test]
fn normalization_handles_finite_extreme_exponents_and_rejects_zero() {
    let p = 128;
    for entry in [
        Float::with_val(p, 1) << (rug::float::exp_max() - 1) as u32,
        Float::with_val(p, 1) << (rug::float::exp_min() + 10),
    ] {
        let mut vector = vec![entry.clone(), entry];
        try_normalize_l2(&mut vector).unwrap();
        let expected = Float::with_val(p, 2).sqrt().recip();
        for value in vector {
            assert!((value - &expected).abs() < (Float::with_val(p, 1) >> 120_i32));
        }
    }
    assert!(try_normalize_l2(&mut [Float::with_val(p, 0)]).is_err());
    assert!(try_normalize_l2(&mut [Float::with_val(p, Special::Nan)]).is_err());
}
#[test]
fn tiny_spectrum_is_not_overwhelmed_by_the_inverse_iteration_shift() {
    let p = 128;
    let scale = Float::with_val(p, 1) >> 400_i32;
    let d = [scale.clone(), Float::with_val(p, &scale * 2)];
    for solver in [TridiagSolver::Dense, TridiagSolver::BandedInterleaved] {
        let vector = tridiag_eigenvector_for_value_hp(
            &d,
            &[Float::with_val(p, 0)],
            &scale,
            p,
            TridiagEigvecOptions {
                max_steps: 40,
                early_termination: false,
                solver,
            },
        )
        .unwrap();
        assert!(vector[1].clone().abs() < (Float::with_val(p, 1) >> 100_i32));
        assert!((vector[0].clone().abs() - 1_i32).abs() < (Float::with_val(p, 1) >> 100_i32));
    }
}
#[test]
fn lu_and_eigenvector_domains_fail_explicitly() {
    let p = 128;
    assert!(lu_factor(&[Float::with_val(p, Special::Nan)], 1).is_err());
    assert!(lu_factor(&[Float::with_val(p, 1)], 2).is_err());
    for solver in [TridiagSolver::Dense, TridiagSolver::BandedInterleaved] {
        let options = TridiagEigvecOptions {
            max_steps: 40,
            early_termination: false,
            solver,
        };
        assert!(tridiag_eigenvector_for_value_hp(
            &[Float::with_val(p, 1)],
            &[],
            &Float::with_val(p, Special::Nan),
            p,
            options
        )
        .is_err());
        assert!(tridiag_eigenvector_for_value_hp(
            &[Float::with_val(p, 1)],
            &[],
            &Float::with_val(p, 1),
            0,
            options
        )
        .is_err());
    }
    assert_eq!(
        TridiagEigvecOptions::default().solver,
        TridiagSolver::BandedInterleaved
    );
}

#[test]
fn public_lu_records_are_checked_before_solving() {
    let p = 128;
    let one = Float::with_val(p, 1);
    let nan = Float::with_val(p, Special::Nan);
    assert!(tridiag_lu_factor_hp(&[], &[nan], &[], p).is_err());
    assert!(tridiag_lu_factor_hp(&[], &[Float::with_val(p, 0)], &[], p).is_err());
    let mut factors = lu_factor(
        &[
            one.clone(),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            one.clone(),
        ],
        2,
    )
    .unwrap();
    factors.perm = vec![0, 0];
    assert!(try_lu_solve(&factors, &[one.clone(), one.clone()], 2, p).is_err());
    factors.perm = vec![0, 1];
    assert!(try_lu_solve(&factors, std::slice::from_ref(&one), 2, p).is_err());
    assert!(try_lu_solve(&factors, &[one.clone(), one.clone()], 2, 31).is_err());
    let result = try_lu_solve(&factors, &[one.clone(), one.clone()], 2, p).unwrap();
    assert_eq!(result, vec![one.clone(), one]);
}
