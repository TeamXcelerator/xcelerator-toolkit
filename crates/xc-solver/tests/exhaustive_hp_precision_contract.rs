#![cfg(feature = "hp-reference")]
use rug::Float;
use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionPolicy, Reproducibility, SolverConfig,
    StoppingPolicy, Subspace,
};
use xc_solver::{solve_dense_reference_hp, DenseSymmetricProblemHp};
#[test]
fn dense_hp_diagnostics_use_requested_precision_for_stored_exact_entries() {
    let config = SolverConfig {
        target: EigenTarget::AlgebraicSmallest,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(128),
        stopping: StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-30").unwrap(),
            scaled_backward_error: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 30,
            minimum_iterations: 2,
        },
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: vec![],
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    };
    let mut residuals = Vec::new();
    for storage in [4, 128] {
        let matrix: Vec<_> = [2, 1, 1, 2]
            .into_iter()
            .map(|v| Float::with_val(storage, v))
            .collect();
        let report =
            solve_dense_reference_hp(&DenseSymmetricProblemHp::new(&matrix, 2).unwrap(), &config)
                .unwrap();
        let residual = Float::with_val(128, Float::parse(&report.residual_norm).unwrap());
        residuals.push(residual);
    }
    assert!(
        residuals[0] < Float::with_val(128, Float::parse("1e-30").unwrap()),
        "low-storage residual: {:?}; working-storage residual: {:?}",
        residuals[0],
        residuals[1]
    );
    assert_eq!(residuals[0], residuals[1]);
}
