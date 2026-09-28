#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::linalg::*;

fn f(x: i32) -> Float {
    Float::with_val(128, x)
}

#[test]
fn default_tridiagonal_solve_recovers_exact_solution_after_a_later_pivot() {
    // A*[1,1,1]=[3,3,2]. This requires a pivot AFTER the first elimination.
    let factors =
        tridiag_lu_factor_hp(&[f(1), f(1)], &[f(2), f(1), f(1)], &[f(1), f(1)], 128).unwrap();
    assert_eq!(
        tridiag_lu_solve_hp(&factors, &[f(3), f(3), f(2)], 128).unwrap(),
        vec![f(1), f(1), f(1)]
    );
}

#[test]
fn forced_even_iteration_projects_even_small_parity_errors() {
    let a = vec![f(1), f(0), f(0), f(1)];
    for steps in [1, 4, 10] {
        for pair in [[3, 4], [7, 8], [15, 16], [31, 32]] {
            let result = inverse_iteration_from_detailed(
                &a,
                2,
                128,
                steps,
                true,
                Some(pair.map(f).to_vec()),
            )
            .unwrap();
            assert_eq!(
                result.eigenvector[0], result.eigenvector[1],
                "force_even was ignored for {pair:?} at {steps} steps"
            );
            assert!((result.eigenvalue - f(1)).abs() < f(1) >> 100u32);
        }
    }
}

#[test]
fn inverse_iteration_rejects_non_symmetric_and_non_invariant_even_requests() {
    assert!(inverse_iteration(&[f(2), f(1), f(0), f(1)], 2, 128, 8, false).is_err());
    // Reflection does not preserve eigenspaces of diag(1,2). P A^-1 P
    // is not the inverse of the even compression P A P.
    assert!(inverse_iteration(&[f(1), f(0), f(0), f(2)], 2, 128, 8, true).is_err());
}

#[test]
fn retained_inverse_iteration_rejects_overflowing_dimensions_without_panicking() {
    let factors = LuFactors {
        lu: vec![],
        perm: vec![],
    };
    assert!(inverse_iteration_from_factors_detailed(
        &[],
        &factors,
        usize::MAX,
        128,
        4,
        false,
        None
    )
    .is_err());
}
