#![cfg(feature = "hp-reference")]

use rug::Float;
use xc_core::{DecimalLiteral, EigenTarget};
use xc_operator::DenseSymmetricHp;
use xc_solver::{
    ShiftInvertFactorizationDescriptorHp, ShiftInvertKrylovConfigHp, ShiftInvertKrylovSolverHp,
    ShiftInvertSolveHp, SolverError,
};

struct NonfiniteInverse {
    value: f64,
    output_precision: u32,
}
impl ShiftInvertSolveHp for NonfiniteInverse {
    fn descriptor(&self) -> ShiftInvertFactorizationDescriptorHp {
        ShiftInvertFactorizationDescriptorHp {
            id: "audit_nonfinite_inverse".into(),
            dimension: 3,
            shift: DecimalLiteral::new("0").unwrap(),
            factorization_precision_bits: 128,
            exact_shifted_solve: true,
            approximation_error_bound: None,
        }
    }
    fn solve_shifted(&self, _: &[Float], output: &mut [Float], p: u32) -> Result<(), SolverError> {
        for x in output {
            *x = Float::with_val(p.min(self.output_precision), self.value);
        }
        Ok(())
    }
}

#[test]
fn nonfinite_shifted_action_must_not_be_silently_replaced_by_coordinates() {
    let p = 128;
    let values = [1, 0, 0, 0, 2, 0, 0, 0, 3]
        .into_iter()
        .map(|x| Float::with_val(p, x))
        .collect();
    let operator =
        DenseSymmetricHp::new("audit_diag123", 3, values, p, &Float::with_val(p, 0)).unwrap();
    let tolerance = || DecimalLiteral::new("1e-20").unwrap();
    let config = ShiftInvertKrylovConfigHp {
        target: EigenTarget::SmallestMagnitude,
        precision_bits: p,
        requested_eigenpairs: 1,
        guard_eigenpairs: 1,
        maximum_subspace_dimension: 3,
        maximum_restarts: 3,
        minimum_restarts: 2,
        maximum_projected_sweeps: 100,
        absolute_residual_tolerance: tolerance(),
        scaled_backward_error_tolerance: tolerance(),
        ritz_value_stability_tolerance: tolerance(),
        boundary_cluster_tolerance: tolerance(),
    };
    let initial = vec![vec![
        Float::with_val(p, 1),
        Float::with_val(p, 0),
        Float::with_val(p, 0),
    ]];
    for (value, output_precision) in [
        (f64::NAN, p),
        (f64::INFINITY, p),
        (f64::NEG_INFINITY, p),
        (1.0, 32),
    ] {
        let callback = NonfiniteInverse {
            value,
            output_precision,
        };
        let outcome = ShiftInvertKrylovSolverHp
            .solve_with_initial_basis(&operator, &callback, &config, &initial);
        eprintln!("invalid callback outcome: {outcome:?}");
        assert!(
            outcome.is_err(),
            "an invalid shifted-solve callback was silently bypassed"
        );
    }
}
