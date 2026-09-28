use xc_core::{EigenTarget, ResultStatus};
use xc_operator::{
    DiagonalF64, GeneralizedEigenProblem, LinearOperator, OperatorError, OperatorMetadata,
    PositiveDefiniteMetric, SymmetricOperator,
};
use xc_solver::*;
struct Metric(DiagonalF64);
impl LinearOperator<f64> for Metric {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        self.0.apply(x, y)
    }
    fn metadata(&self) -> OperatorMetadata {
        self.0.metadata()
    }
    fn norm_bound(&self) -> Option<f64> {
        self.0.norm_bound()
    }
}
impl SymmetricOperator<f64> for Metric {}
impl PositiveDefiniteMetric<f64> for Metric {}
fn config() -> GeneralizedExtremeConfigF64 {
    GeneralizedExtremeConfigF64 {
        target: EigenTarget::AlgebraicLargest,
        absolute_residual_tolerance: 1e-14,
        scaled_backward_error_tolerance: 1e-14,
        ritz_value_stability_tolerance: 1e-14,
        maximum_iterations: 20,
        minimum_iterations: 2,
    }
}
#[test]
fn dense_native_revalidates_public_shape() {
    for dimension in [0, 2, usize::MAX] {
        let problem = DenseGeneralizedProblemF64 {
            operator: &[],
            metric: &[],
            dimension,
        };
        let result = std::panic::catch_unwind(|| {
            DenseGeneralizedReferenceSolverF64::default()
                .solve(&problem, &EigenTarget::AlgebraicLargest)
        });
        assert!(
            result.is_ok(),
            "fallible API panicked for dimension {dimension}"
        );
        assert!(result.unwrap().is_err());
    }
}
#[test]
fn dense_native_requires_exact_symmetric_storage() {
    assert!(DenseGeneralizedProblemF64::new(&[2., 1., 0., 3.], &[1., 0., 0., 1.], 2, 2.).is_err());
}
#[test]
fn dense_native_rejects_unrepresentable_whitening() {
    let a = [1e308];
    let b = [1e-308];
    let problem = DenseGeneralizedProblemF64::new(&a, &b, 1, 0.).unwrap();
    assert!(DenseGeneralizedReferenceSolverF64::default()
        .solve(&problem, &EigenTarget::AlgebraicLargest)
        .is_err());
}
#[test]
fn native_exact_warm_start_performs_a_real_projected_stability_update() {
    let a = DiagonalF64::new("diag", vec![1., 2.]).unwrap();
    let b = Metric(DiagonalF64::new("I", vec![1., 1.]).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
    let r = MatrixFreeLobpcgF64
        .solve_with_initial_vector(&problem, &config(), &[0., 1.])
        .unwrap();
    assert_eq!(r.status, ResultStatus::Converged);
    assert_eq!(r.eigenvalue, 2.);
    assert_eq!(r.relative_residual, 0.);
    assert_eq!(r.operator_applications, 4);
    assert_eq!(r.metric_applications, 4);
    assert_eq!(r.projected_factorizations, 1);
    assert_eq!(r.iterations, 2);
    assert!(r.ritz_value_stability_observed);
    let mut legacy = serde_json::to_value(&r).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("ritz_value_stability_observed");
    let decoded: MatrixFreeGeneralizedEigenpairReportF64 = serde_json::from_value(legacy).unwrap();
    assert!(!decoded.ritz_value_stability_observed);
}
#[test]
fn native_generalized_relative_residual_preserves_subnormal_scale() {
    let scale = f64::from_bits(1u64 << 34); // exactly 2^-1040
    let a = DiagonalF64::new("tiny", vec![scale, 2. * scale]).unwrap();
    let b = Metric(DiagonalF64::new("I", vec![1., 1.]).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
    let mut cfg = config();
    cfg.minimum_iterations = 0;
    cfg.maximum_iterations = 1;
    cfg.absolute_residual_tolerance = f64::from_bits(1);
    let r = MatrixFreeLobpcgF64
        .solve_with_initial_vector(&problem, &cfg, &[1., 1.])
        .unwrap();
    assert!(
        (r.relative_residual - 0.16227766016837933).abs() < 1e-9,
        "relative residual {}",
        r.relative_residual
    );
}
struct NonfiniteOperator;
impl LinearOperator<f64> for NonfiniteOperator {
    fn dimension(&self) -> usize {
        1
    }
    fn apply(&self, _: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        y.fill(f64::NAN);
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        DiagonalF64::new("bad callback", vec![1.])
            .unwrap()
            .metadata()
    }
}
impl SymmetricOperator<f64> for NonfiniteOperator {}
#[test]
fn native_generalized_rejects_nonfinite_callback_even_at_iteration_limit() {
    let b = Metric(DiagonalF64::new("I", vec![1.]).unwrap());
    let problem = GeneralizedEigenProblem::new(&NonfiniteOperator, &b).unwrap();
    let mut cfg = config();
    cfg.maximum_iterations = 1;
    cfg.minimum_iterations = 0;
    assert!(MatrixFreeLobpcgF64.solve(&problem, &cfg).is_err());
}

#[test]
fn native_residual_convergence_does_not_claim_missing_global_ordering() {
    let a = DiagonalF64::new("diag", vec![1., 2.]).unwrap();
    let b = Metric(DiagonalF64::new("I", vec![1., 1.]).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
    let report = MatrixFreeLobpcgF64
        .solve_with_initial_vector(&problem, &config(), &[1., 0.])
        .unwrap();
    // Preserve the original two-dimensional counterexample. The required real
    // complement update now spans the whole space and reaches its true maximum.
    assert_eq!(report.status, ResultStatus::Converged);
    assert_eq!(report.eigenvalue, 2.);
    assert!(report.target_ordering_established_by_full_space_projection);
    assert!(report.ritz_value_stability_observed);
    assert!(report.projected_factorizations >= 2);
    assert_eq!(report.residual_norm, 0.);
    assert_eq!(report.eigenvector[0], 0.);
    assert_eq!(report.eigenvector[1].abs(), 1.);

    // In four dimensions the first real complement finds 4, and the next
    // coordinate has value 2. Stability can then be observed while the exact
    // largest value 8 is still outside the two-dimensional trial space.
    for scale in [1.0, 2.0] {
        let spectrum = [scale, 4.0 * scale, 2.0 * scale, 8.0 * scale];
        let a = DiagonalF64::new("hidden diagonal extreme", spectrum.to_vec()).unwrap();
        let b = Metric(DiagonalF64::new("I", vec![1.; 4]).unwrap());
        let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
        let partial = MatrixFreeLobpcgF64
            .solve_with_initial_vector(&problem, &config(), &[1., 0., 0., 0.])
            .unwrap();
        assert_eq!(partial.status, ResultStatus::Converged);
        assert_eq!(partial.eigenvalue, 4.0 * scale);
        assert!(partial.eigenvalue < spectrum[3]);
        assert!(!partial.target_ordering_established_by_full_space_projection);
        assert!(partial.ritz_value_stability_observed);
        let mut residual = 0.0f64;
        let mut norm_sq = 0.0;
        for (entry, x) in spectrum.iter().zip(&partial.eigenvector) {
            residual = residual.hypot(entry * x - partial.eigenvalue * x);
            norm_sq += x * x;
        }
        assert_eq!(residual, 0.0);
        assert_eq!(norm_sq, 1.0);
    }
}
