use xc_core::EigenTarget;
use xc_operator::DiagonalF64;
use xc_solver::{BlockExtremeConfigF64, BlockSubspaceIterationF64, SymmetricProblemF64};
fn selected_count(diagonal: Vec<f64>, absolute: f64, relative: f64) -> usize {
    let operator = DiagonalF64::new("exact-cluster-boundary", diagonal).unwrap();
    let config = BlockExtremeConfigF64 {
        target: EigenTarget::AlgebraicLargest,
        requested_count: 1,
        block_size: 2,
        absolute_residual_tolerance: 1e-20,
        scaled_backward_error_tolerance: 1e-20,
        ritz_value_stability_tolerance: 1e-20,
        cluster_absolute_tolerance: absolute,
        cluster_relative_tolerance: relative,
        maximum_iterations: 3,
        minimum_iterations: 2,
    };
    let report = BlockSubspaceIterationF64::default()
        .solve_with_initial_subspace(
            &SymmetricProblemF64::new(&operator),
            &config,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
    report.returned_count
}
#[test]
fn rounded_gap_does_not_merge_distinct_clusters() {
    assert_eq!(selected_count(vec![-2.0f64.powi(-54), 1.0], 1.0, 0.0), 1);
}
#[test]
fn rounded_relative_product_does_not_expand_cluster_tolerance() {
    assert_eq!(selected_count(vec![3.5, 5.0], 0.0, 0.3), 1);
}
#[test]
fn exact_cluster_boundary_remains_inclusive() {
    assert_eq!(selected_count(vec![0.0, 1.0], 1.0, 0.0), 2);
}
