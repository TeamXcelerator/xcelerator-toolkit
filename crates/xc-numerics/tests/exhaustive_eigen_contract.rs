#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::eigen::{tridiag_eigenvector_for_value_hp, TridiagEigvecOptions, TridiagSolver};

fn recover(stored_bits: u32, solver: TridiagSolver) -> anyhow::Result<Vec<Float>> {
    tridiag_eigenvector_for_value_hp(
        &vec![Float::with_val(64, 2); 2],
        &[Float::with_val(stored_bits, 1)],
        &Float::with_val(64, 1),
        64,
        TridiagEigvecOptions {
            max_steps: 32,
            early_termination: false,
            solver,
        },
    )
}

#[test]
fn exact_low_precision_couplings_do_not_lower_the_requested_solve_precision() {
    for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
        let reference = recover(64, solver).unwrap();
        let actual = recover(4, solver).unwrap();
        assert_eq!(
            actual, reference,
            "same exact coefficients and requested precision, solver={solver:?}"
        );
        // Independent exact eigen-equation: [[2,1],[1,2]] at lambda=1 requires x+y=0.
        let residual = Float::with_val(128, &actual[0] + &actual[1]).abs();
        assert!(residual <= Float::with_val(128, 1) >> 48);
    }
}
#[test]
fn exact_high_precision_couplings_are_converted_to_the_requested_solve_precision() {
    for solver in [TridiagSolver::BandedInterleaved, TridiagSolver::Dense] {
        let reference = recover(64, solver).unwrap();
        let actual =
            recover(128, solver).unwrap_or_else(|error| panic!("solver={solver:?}: {error}"));
        assert_eq!(actual, reference);
        assert!(actual.iter().all(|x| x.prec() == 64));
    }
}
