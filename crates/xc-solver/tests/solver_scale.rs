use xc_core::{EigenTarget, ResultStatus};
use xc_operator::DiagonalF64;
use xc_solver::{BlockExtremeConfigF64, BlockSubspaceIterationF64, SymmetricProblemF64};

fn config(scale: f64, target: EigenTarget) -> BlockExtremeConfigF64 {
    BlockExtremeConfigF64 {
        target,
        requested_count: 1,
        block_size: 2,
        absolute_residual_tolerance: scale * 1e-12,
        scaled_backward_error_tolerance: 1e-12,
        ritz_value_stability_tolerance: 1e-12,
        cluster_absolute_tolerance: 0.0,
        cluster_relative_tolerance: 1e-10,
        maximum_iterations: 600,
        minimum_iterations: 2,
    }
}

#[test]
fn block_residual_does_not_vanish_when_its_square_underflows() {
    let scale = 2.0f64.powi(-600);
    let operator =
        DiagonalF64::new("tiny", vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale]).unwrap();
    let mut cfg = config(scale, EigenTarget::AlgebraicLargest);
    cfg.maximum_iterations = 2;
    let report = BlockSubspaceIterationF64::default()
        .solve(&SymmetricProblemF64::new(&operator), &cfg)
        .unwrap();
    let cluster = &report.invariant_subspaces[0];
    let value = cluster.ritz_values[0] / scale;
    // Independent residual in dimensionless coordinates; none of these squares
    // are remotely near underflow, and the exact spectrum is {1, 2, 3, 4}.
    let expected = cluster.basis[0]
        .iter()
        .enumerate()
        .map(|(i, x)| ((i as f64 + 1.0 - value) * x).powi(2))
        .sum::<f64>()
        .sqrt();
    eprintln!("status={:?} lambda/scale={value:e} actual_residual/scale={expected:e} reported_residual/scale={:e}", report.status, report.maximum_residual_norm/scale);
    assert!(expected > 1e-3);
    assert!((report.maximum_residual_norm / scale - expected).abs() < 1e-13);
    assert!((cluster.residual_frobenius_norm / scale - expected).abs() < 1e-13);
    assert_ne!(report.status, ResultStatus::Converged);
}

#[test]
fn block_extremes_and_diagnostics_are_scale_covariant() {
    for scale in [
        2.0f64.powi(-1000),
        2.0f64.powi(-600),
        1.0,
        2.0f64.powi(600),
        f64::MAX * 0.1875,
    ] {
        let exponent = scale.log2();
        let operator =
            DiagonalF64::new("scaled", vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale]).unwrap();
        for (target, expected) in [
            (EigenTarget::AlgebraicLargest, 4.0),
            (EigenTarget::AlgebraicSmallest, 1.0),
        ] {
            let report = BlockSubspaceIterationF64::default()
                .solve(&SymmetricProblemF64::new(&operator), &config(scale, target))
                .unwrap();
            assert_eq!(
                report.status,
                ResultStatus::Converged,
                "exponent {exponent}: {report:?}"
            );
            let cluster = &report.invariant_subspaces[0];
            let value = cluster.ritz_values[0] / scale;
            assert!(
                (value - expected).abs() < 1e-11,
                "exponent {exponent}: {value} != {expected}"
            );
            let residual = cluster.basis[0]
                .iter()
                .enumerate()
                .map(|(i, x)| ((i as f64 + 1.0 - value) * x).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual < 1e-10, "exponent {exponent}: residual {residual}");
            assert!((report.maximum_residual_norm / scale - residual).abs() < 1e-13);
            assert!(report.maximum_scaled_backward_error.is_finite());
        }
    }
}

fn single_config(scale: f64, target: EigenTarget) -> xc_core::SolverConfig {
    xc_core::SolverConfig {
        target,
        subspace: xc_core::Subspace::Full,
        assurance: xc_core::AssuranceLevel::Computed,
        precision: xc_core::PrecisionPolicy::fixed(53),
        stopping: xc_core::StoppingPolicy {
            absolute_residual: xc_core::DecimalLiteral::new(format!("{:e}", scale * 1e-12))
                .unwrap(),
            scaled_backward_error: xc_core::DecimalLiteral::new("1e-12").unwrap(),
            maximum_iterations: 600,
            minimum_iterations: 2,
        },
        reproducibility: xc_core::Reproducibility::Deterministic,
        algorithm_preferences: Vec::new(),
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    }
}

fn check_single_extremes(solver: &dyn xc_solver::EigenSolverF64) {
    for exponent in [-600, 0, 600] {
        let scale = 2.0f64.powi(exponent);
        let operator = DiagonalF64::new(
            "single-scaled",
            vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale],
        )
        .unwrap();
        for (target, expected) in [
            (EigenTarget::AlgebraicLargest, 4.0),
            (EigenTarget::AlgebraicSmallest, 1.0),
        ] {
            let report = solver
                .solve(
                    &SymmetricProblemF64::new(&operator),
                    &single_config(scale, target),
                )
                .unwrap();
            assert_eq!(
                report.status,
                ResultStatus::Converged,
                "exponent {exponent}: {report:?}"
            );
            assert!((report.eigenvalue / scale - expected).abs() < 1e-10);
            let residual = report
                .eigenvector
                .iter()
                .enumerate()
                .map(|(i, x)| ((i as f64 + 1.0 - report.eigenvalue / scale) * x).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual < 1e-10);
            assert!((residual - report.residual_norm / scale).abs() < 1e-13);
        }
    }
}

#[test]
fn shifted_power_extremes_are_scale_covariant() {
    check_single_extremes(&xc_solver::ShiftedPowerSolverF64);
}

#[test]
fn lanczos_extremes_are_scale_covariant() {
    check_single_extremes(&xc_solver::LanczosSolverF64::default());
}

#[test]
fn rotated_indefinite_spectrum_is_correct_across_scales() {
    // H/2 is exactly orthogonal; H diag(-3,-1,2,5) H^T/4 has this exact spectrum.
    let h = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    let eigenvalues = [-3.0, -1.0, 2.0, 5.0];
    let mut matrix = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            matrix[4 * i + j] = (0..4)
                .map(|k| h[i][k] * eigenvalues[k] * h[j][k] / 4.0)
                .sum();
        }
    }
    for exponent in [-600, 0, 600] {
        let scale = 2.0f64.powi(exponent);
        let operator = xc_operator::DenseSymmetricF64::new(
            "hadamard-spectrum",
            4,
            matrix.iter().map(|v| v * scale).collect(),
            0.0,
        )
        .unwrap();
        for (target, expected) in [
            (EigenTarget::AlgebraicLargest, 5.0),
            (EigenTarget::AlgebraicSmallest, -3.0),
        ] {
            let report = BlockSubspaceIterationF64::default()
                .solve(
                    &SymmetricProblemF64::new(&operator),
                    &config(scale, target.clone()),
                )
                .unwrap();
            assert_eq!(report.status, ResultStatus::Converged);
            assert!(
                (report.invariant_subspaces[0].ritz_values[0] / scale - expected).abs() < 1e-10
            );
            for solver in [
                &xc_solver::ShiftedPowerSolverF64 as &dyn xc_solver::EigenSolverF64,
                &xc_solver::LanczosSolverF64::default(),
            ] {
                let report = solver
                    .solve(
                        &SymmetricProblemF64::new(&operator),
                        &single_config(scale, target.clone()),
                    )
                    .unwrap();
                assert_eq!(report.status, ResultStatus::Converged);
                assert!((report.eigenvalue / scale - expected).abs() < 1e-10);
                let residual = (0..4)
                    .map(|i| {
                        let ax: f64 = (0..4)
                            .map(|j| matrix[4 * i + j] * report.eigenvector[j])
                            .sum();
                        (ax - report.eigenvalue / scale * report.eigenvector[i]).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();
                assert!(residual < 1e-10);
                assert!((report.residual_norm / scale - residual).abs() < 1e-13);
            }
        }
    }
}

#[test]
fn block_clustering_does_not_merge_an_overflowing_gap() {
    let large = f64::MAX * 0.75;
    let operator = DiagonalF64::new("large-gap", vec![-large, large]).unwrap();
    let mut cfg = config(1.0, EigenTarget::AlgebraicLargest);
    // In exact arithmetic, gap=2*large > 1.5*large; both intermediates
    // overflow in binary64, but the comparison itself is well defined.
    cfg.cluster_relative_tolerance = 1.5;
    let report = BlockSubspaceIterationF64::default()
        .solve_with_initial_subspace(
            &SymmetricProblemF64::new(&operator),
            &cfg,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
    assert_eq!(report.returned_count, 1);
    assert_eq!(report.invariant_subspaces[0].ritz_values[0], large);
    assert_eq!(report.status, ResultStatus::Converged);
}
