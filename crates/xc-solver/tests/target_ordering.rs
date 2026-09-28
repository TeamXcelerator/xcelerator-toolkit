//! Fresh 2026-09-28 adversarial target-selection checks. Independent oracle:
//! the diagonal spectrum is exact; the projector's largest eigenvalue is at
//! least its (2,2) entry by the Rayleigh variational principle.
use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionPolicy, Reproducibility, ResultStatus,
    SolverConfig, StoppingPolicy, Subspace,
};
use xc_operator::{DenseSymmetricF64, DiagonalF64};
use xc_solver::{
    BlockExtremeConfigF64, BlockSubspaceIterationF64, EigenSolverF64, LanczosSolverF64,
    ShiftedPowerSolverF64, SymmetricProblemF64,
};

fn config() -> SolverConfig {
    SolverConfig {
        target: EigenTarget::AlgebraicLargest,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        stopping: StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-12").unwrap(),
            scaled_backward_error: DecimalLiteral::new("1e-12").unwrap(),
            maximum_iterations: 100,
            minimum_iterations: 1,
        },
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: vec![],
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    }
}

#[test]
fn native_single_vector_routes_must_disclose_unestablished_global_target() {
    // Reproduce the fixed public solver's deterministic seed. Rounding of the
    // projector does not invalidate the oracle: lambda_max >= A[1,1] > 0.79.
    let mut seed = [1.0_f64 - 3.0e-4, 0.5_f64 - 2.0e-4];
    let norm = (seed[0] * seed[0] + seed[1] * seed[1]).sqrt();
    seed.iter_mut().for_each(|x| *x /= norm);
    let [a, b] = seed;
    let matrix = vec![b * b, -a * b, -a * b, a * a];
    assert!(matrix[3] > 0.79);
    let op = DenseSymmetricF64::new("orthogonal-projector", 2, matrix, 0.0).unwrap();
    let problem = SymmetricProblemF64::new(&op);
    let power = ShiftedPowerSolverF64.solve(&problem, &config()).unwrap();
    let lanczos = LanczosSolverF64::default()
        .solve(&problem, &config())
        .unwrap();
    eprintln!("power: value={} residual={} status={:?}; lanczos: value={} residual={} status={:?}; independent lower bound={}",
        power.eigenvalue, power.residual_norm, power.status,
        lanczos.eigenvalue, lanczos.residual_norm, lanczos.status, a*a);
    for report in [power, lanczos] {
        // The independent counterexample remains: residual convergence can
        // occur at the wrong global extreme. The report must disclose that.
        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.eigenvalue < 0.79);
        assert!(
            !report.global_target_ordering_established,
            "global ordering claimed for a seed-orthogonal wrong extreme: {report:?}"
        );
        let mut legacy = serde_json::to_value(&report).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("global_target_ordering_established");
        let reopened: xc_solver::EigenpairReportF64 = serde_json::from_value(legacy).unwrap();
        assert!(!reopened.global_target_ordering_established);
    }
}

#[test]
fn native_block_target_boundary_must_not_certify_an_unseen_extreme() {
    let op = DiagonalF64::new("known-spectrum", vec![1.0, 2.0, 3.0]).unwrap();
    let cfg = BlockExtremeConfigF64 {
        target: EigenTarget::AlgebraicLargest,
        requested_count: 1,
        block_size: 2,
        absolute_residual_tolerance: 1e-12,
        scaled_backward_error_tolerance: 1e-12,
        ritz_value_stability_tolerance: 1e-12,
        cluster_absolute_tolerance: 0.0,
        cluster_relative_tolerance: 1e-12,
        maximum_iterations: 10,
        minimum_iterations: 2,
    };
    let initial = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let report = BlockSubspaceIterationF64::default()
        .solve_with_initial_subspace(&SymmetricProblemF64::new(&op), &cfg, &initial)
        .unwrap();
    let value = report.invariant_subspaces[0].ritz_values[0];
    eprintln!(
        "block: value={value} residual={} status={:?} boundary={}; exact largest=3",
        report.maximum_residual_norm, report.status, report.target_boundary_separation_established
    );
    assert_eq!(value, 2.0);
    assert!(report.ritz_boundary_separation_established);
    assert!(!report.global_target_ordering_established);
    assert!(
        !report.target_boundary_separation_established || value == 3.0,
        "Target boundary claimed despite missing exact largest eigenvalue: {report:?}"
    );
    let full = BlockExtremeConfigF64 {
        block_size: 3,
        ..cfg
    };
    let full_report = BlockSubspaceIterationF64::default()
        .solve(&SymmetricProblemF64::new(&op), &full)
        .unwrap();
    assert!(full_report.global_target_ordering_established);
    assert!(full_report.target_boundary_separation_established);
    assert!((full_report.invariant_subspaces[0].ritz_values[0] - 3.0).abs() < 1e-12);
}
