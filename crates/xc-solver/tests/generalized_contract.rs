#![cfg(feature = "hp-reference")]
use rug::{float::Special, Float, Rational};
use std::sync::atomic::{AtomicUsize, Ordering};
use xc_core::{DecimalLiteral as D, EigenTarget, ResultStatus};
use xc_operator::{
    DenseSymmetricHp, GeneralizedEigenProblem, LinearOperator, OperatorError, OperatorMetadata,
    PositiveDefiniteMetric, SymmetricOperator,
};
use xc_solver::*;
struct Metric(DenseSymmetricHp);
impl LinearOperator<Float> for Metric {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        self.0.apply(x, y)
    }
    fn metadata(&self) -> OperatorMetadata {
        self.0.metadata()
    }
    fn norm_bound(&self) -> Option<Float> {
        self.0.norm_bound()
    }
}
impl SymmetricOperator<Float> for Metric {}
impl PositiveDefiniteMetric<Float> for Metric {}
struct Counted<'a, O: ?Sized> {
    inner: &'a O,
    calls: AtomicUsize,
}
impl<O: LinearOperator<Float> + ?Sized> LinearOperator<Float> for Counted<'_, O> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply(x, y)
    }
    fn metadata(&self) -> OperatorMetadata {
        self.inner.metadata()
    }
    fn norm_bound(&self) -> Option<Float> {
        self.inner.norm_bound()
    }
}
impl<O: SymmetricOperator<Float> + ?Sized> SymmetricOperator<Float> for Counted<'_, O> {}
impl<O: PositiveDefiniteMetric<Float> + ?Sized> PositiveDefiniteMetric<Float> for Counted<'_, O> {}

fn config() -> GeneralizedExtremeConfigHp {
    GeneralizedExtremeConfigHp {
        target: EigenTarget::AlgebraicLargest,
        precision_bits: 256,
        absolute_residual_tolerance: D::new("1e-40").unwrap(),
        scaled_backward_error_tolerance: D::new("1e-40").unwrap(),
        ritz_value_stability_tolerance: D::new("1e-40").unwrap(),
        maximum_iterations: 200,
        minimum_iterations: 2,
    }
}
fn fixture() -> (
    DenseSymmetricHp,
    Metric,
    MatrixFreeGeneralizedEigenpairReportHp,
    DenseGeneralizedEigenpairReportHp,
) {
    let p = 256;
    let z = Float::with_val(p, 0);
    let vals = |a: &[i32]| a.iter().map(|x| Float::with_val(p, *x)).collect::<Vec<_>>();
    let data = vals(&[3, 1, 1, 3]);
    let b = vals(&[2, 1, 1, 2]);
    let a = DenseSymmetricHp::new("A", 2, data.clone(), p, &z).unwrap();
    let metric = Metric(DenseSymmetricHp::new("B", 2, b.clone(), p, &z).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &metric).unwrap();
    let mf = MatrixFreeGeneralizedRayleighRitzHp
        .solve(&problem, &config())
        .unwrap();
    let dense = solve_dense_generalized_whitening_hp(
        &DenseGeneralizedProblemHp::new(&data, &b, 2).unwrap(),
        &config(),
    )
    .unwrap();
    (a, metric, mf, dense)
}
#[test]
fn generalized_crosscheck_rejects_nonfinite_and_rounded_boundary_reports() {
    let p = 256;
    let (a, metric, mf, dense) = fixture();
    let problem = GeneralizedEigenProblem::new(&a, &metric).unwrap();
    let tol = HpCrossCheckTolerance {
        eigenvalue_absolute: D::new("1").unwrap(),
        one_minus_overlap_squared: D::new("1e-20").unwrap(),
    };
    assert!(cross_check_generalized_hp_reports(&problem, &mf, &dense, &tol).is_ok());
    let mut nan = dense.clone();
    nan.eigenvalue = Float::with_val(p, Special::Nan);
    assert!(cross_check_generalized_hp_reports(&problem, &mf, &nan, &tol).is_err());
    let mut left = mf.clone();
    left.eigenvalue = Float::with_val(p, 1);
    let mut right = dense.clone();
    right.eigenvalue = -(Float::with_val(p, 1) >> 400u32);
    assert!(cross_check_generalized_hp_reports(&problem, &left, &right, &tol).is_err());
    let mut bad = dense.clone();
    bad.residual_norm = Float::with_val(p, Special::Nan);
    assert!(cross_check_generalized_hp_reports(&problem, &mf, &bad, &tol).is_err());
    let mut bad = dense.clone();
    bad.algorithm = mf.algorithm.clone();
    assert!(cross_check_generalized_hp_reports(&problem, &mf, &bad, &tol).is_err());
    let mut bad = dense;
    bad.eigenvector.fill(Float::with_val(p, 0));
    assert!(cross_check_generalized_hp_reports(&problem, &mf, &bad, &tol).is_err());
}
#[test]
fn generalized_dense_revalidates_public_shape_and_precision() {
    let cfg = config();
    let empty = DenseGeneralizedProblemHp {
        operator: &[],
        metric: &[],
        dimension: 0,
    };
    assert!(solve_dense_generalized_whitening_hp(&empty, &cfg).is_err());
    let bad = DenseGeneralizedProblemHp {
        operator: &[],
        metric: &[],
        dimension: usize::MAX,
    };
    assert!(solve_dense_generalized_whitening_hp(&bad, &cfg).is_err());
    let one = [Float::with_val(256, 1)];
    let valid = DenseGeneralizedProblemHp::new(&one, &one, 1).unwrap();
    let mut invalid = cfg;
    invalid.precision_bits = u32::MAX;
    assert!(solve_dense_generalized_whitening_hp(&valid, &invalid).is_err());
}
#[test]
fn generalized_projected_spectrum_is_relative_to_the_actual_matrix_scale() {
    let p = 256;
    let zero = Float::with_val(p, 0);
    let one = Float::with_val(p, 1);
    for exponent in [-20000i32, -400, 0, 400, 20000] {
        let scale: Float = Float::with_val(p, 1) << exponent;
        let data = vec![-scale.clone(), scale.clone(), scale.clone(), scale.clone()];
        let a = DenseSymmetricHp::new("scaled", 2, data, p, &zero).unwrap();
        let metric = Metric(
            DenseSymmetricHp::new(
                "I",
                2,
                vec![one.clone(), zero.clone(), zero.clone(), one.clone()],
                p,
                &zero,
            )
            .unwrap(),
        );
        let problem = GeneralizedEigenProblem::new(&a, &metric).unwrap();
        let config = BlockGeneralizedConfigHp {
            target: EigenTarget::AlgebraicSmallest,
            precision_bits: p,
            requested_eigenpairs: 2,
            guard_eigenpairs: 0,
            absolute_residual_tolerance: D::new("1e-10000").unwrap(),
            scaled_backward_error_tolerance: D::new("1e-40").unwrap(),
            ritz_value_stability_tolerance: D::new("1e-40").unwrap(),
            boundary_cluster_tolerance: D::new("1e-10000").unwrap(),
            maximum_iterations: 3,
            minimum_iterations: 1,
            maximum_projected_sweeps: 100,
        };
        let report = MatrixFreeBlockGeneralizedLobpcgHp
            .solve(&problem, &config)
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert_eq!(report.retained_eigenpairs.len(), 2);
        let root = Float::with_val(p, 2).sqrt();
        for (pair, expected) in report.retained_eigenpairs.iter().zip([-root.clone(), root]) {
            assert!(
                (Float::with_val(p, &pair.eigenvalue / &scale) - expected).abs()
                    < (one.clone() >> 180u32)
            );
        }
    }
}

#[test]
fn exact_generalized_warm_start_has_valid_diagnostics_and_real_work_counts() {
    let p = 256;
    let z = Float::with_val(p, 0);
    let one = Float::with_val(p, 1);
    let metric = Metric(
        DenseSymmetricHp::new(
            "I",
            2,
            vec![one.clone(), z.clone(), z.clone(), one.clone()],
            p,
            &z,
        )
        .unwrap(),
    );
    for largest_value in [0, 2] {
        let a = DenseSymmetricHp::new(
            "diagonal",
            2,
            vec![
                z.clone(),
                z.clone(),
                z.clone(),
                Float::with_val(p, largest_value),
            ],
            p,
            &z,
        )
        .unwrap();
        let counted_a = Counted {
            inner: &a,
            calls: AtomicUsize::new(0),
        };
        let counted_b = Counted {
            inner: &metric,
            calls: AtomicUsize::new(0),
        };
        let problem = GeneralizedEigenProblem::new(&counted_a, &counted_b).unwrap();
        let report = MatrixFreeGeneralizedRayleighRitzHp
            .solve_with_initial_vector(&problem, &config(), &[z.clone(), one.clone()])
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert_eq!(report.eigenvalue, largest_value);
        assert_eq!(report.relative_residual, 0);
        assert_eq!(report.iterations, 2);
        // Initialization and both fresh-image diagnostic iterations are actual
        // callback work, even though this exact HP stationary pair needs no factor.
        assert_eq!(
            report.operator_applications,
            counted_a.calls.load(Ordering::Relaxed)
        );
        assert_eq!(
            report.metric_applications,
            counted_b.calls.load(Ordering::Relaxed)
        );
        assert_eq!(report.operator_applications, 1 + report.iterations);
        assert_eq!(report.metric_applications, 1 + report.iterations);
        assert_eq!(report.projected_factorizations, 0);
        let lambda = report.eigenvalue.to_rational().unwrap();
        let x: Vec<_> = report
            .eigenvector
            .iter()
            .map(|v| v.to_rational().unwrap())
            .collect();
        let mut residual_sq = Rational::new();
        let mut metric_norm = Rational::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..2 {
            let value = if i == 0 { 0 } else { largest_value };
            let residual = (Rational::from(value) - &lambda) * &x[i];
            residual_sq += rug::Rational::from(&residual * &residual);
            metric_norm += rug::Rational::from(&x[i] * &x[i]);
        }
        assert_eq!(residual_sq, 0);
        assert_eq!(metric_norm, 1);
    }
}

#[test]
fn hp_factories_reject_invalid_precision_and_underflowing_shift() {
    let matrix = [Float::with_val(256, 1)];
    let zero = D::new("0").unwrap();
    let factory = DenseShiftInvertFactoryHp::new("I", 1, &matrix, zero.clone(), 256).unwrap();
    for p in [0, 1, 32, u32::MAX] {
        assert!(factory.factor_at_precision(p).is_err());
        assert!(DenseShiftInvertFactorizationHp::factor("I", 1, &matrix, zero.clone(), p).is_err());
    }
    assert!(DenseShiftInvertFactorizationHp::factor(
        "I",
        1,
        &matrix,
        D::new("1e-999999999").unwrap(),
        256
    )
    .is_err());
}

#[test]
fn nonzero_preconditioner_bound_cannot_underflow_into_exactness() {
    let mut d = BlockPreconditionerDescriptorHp {
        id: "probe".into(),
        changes_only_convergence: true,
        approximation_error_bound: Some(D::new("1e-999999999").unwrap()),
    };
    assert!(d.validate(256).is_err());
    d.approximation_error_bound = None;
    assert!(d.validate(0).is_err());
    assert!(d.validate(u32::MAX).is_err());
    assert!(d.validate(256).is_ok());
}

#[test]
fn crosscheck_detects_global_target_missed_by_invariant_warm_start() {
    let p = 256;
    let z = Float::with_val(p, 0);
    let one = Float::with_val(p, 1);
    let data = vec![one.clone(), z.clone(), z.clone(), Float::with_val(p, 2)];
    let b = vec![one.clone(), z.clone(), z.clone(), one.clone()];
    let a = DenseSymmetricHp::new("A", 2, data.clone(), p, &z).unwrap();
    let metric = Metric(DenseSymmetricHp::new("I", 2, b.clone(), p, &z).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &metric).unwrap();
    let mf = MatrixFreeGeneralizedRayleighRitzHp
        .solve_with_initial_vector(&problem, &config(), &[one, z])
        .unwrap();
    assert_eq!(mf.status, ResultStatus::Converged);
    assert_eq!(mf.eigenvalue, 1);
    assert!(!mf.target_ordering_established_by_full_space_projection);
    let dense = solve_dense_generalized_whitening_hp(
        &DenseGeneralizedProblemHp::new(&data, &b, 2).unwrap(),
        &config(),
    )
    .unwrap();
    assert_eq!(dense.eigenvalue, 2);
    let tol = HpCrossCheckTolerance {
        eigenvalue_absolute: D::new("1e-20").unwrap(),
        one_minus_overlap_squared: D::new("1e-20").unwrap(),
    };
    assert!(cross_check_generalized_hp_reports(&problem, &mf, &dense, &tol).is_err());
}
#[test]
fn ritz_value_stability_is_measured_between_distinct_iterates() {
    // The loose residual test first passes while the Ritz value still moves by
    // about the squared residual. The tight stability tolerance must then be
    // met by successive iterates; re-observing one iterate always gave zero.
    let p = 256;
    let n = 40;
    let z = Float::with_val(p, 0);
    let mut data = vec![z.clone(); n * n];
    let mut identity = data.clone();
    for k in 0..n {
        data[k * n + k] = Float::with_val(p, if k + 1 == n { 100 } else { k + 1 });
        identity[k * n + k] = Float::with_val(p, 1);
    }
    let a = DenseSymmetricHp::new("A", n, data, p, &z).unwrap();
    let metric = Metric(DenseSymmetricHp::new("I", n, identity, p, &z).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &metric).unwrap();
    let error = |value: &Float| Float::with_val(p, value - 100u32).abs();
    let bound = Float::with_val(p, 1e-50);
    let single = GeneralizedExtremeConfigHp {
        target: EigenTarget::AlgebraicLargest,
        precision_bits: p,
        absolute_residual_tolerance: D::new("1e-100000").unwrap(),
        scaled_backward_error_tolerance: D::new("1e-12").unwrap(),
        ritz_value_stability_tolerance: D::new("1e-60").unwrap(),
        maximum_iterations: 2000,
        minimum_iterations: 2,
    };
    let report = MatrixFreeGeneralizedRayleighRitzHp
        .solve(&problem, &single)
        .unwrap();
    assert_eq!(report.status, ResultStatus::Converged);
    assert!(error(&report.eigenvalue) < bound, "{}", report.eigenvalue);
    let block = BlockGeneralizedConfigHp {
        target: EigenTarget::AlgebraicLargest,
        precision_bits: p,
        requested_eigenpairs: 2,
        guard_eigenpairs: 1,
        absolute_residual_tolerance: D::new("1e-100000").unwrap(),
        scaled_backward_error_tolerance: D::new("1e-12").unwrap(),
        ritz_value_stability_tolerance: D::new("1e-60").unwrap(),
        boundary_cluster_tolerance: D::new("1e-30").unwrap(),
        maximum_iterations: 2000,
        minimum_iterations: 2,
        maximum_projected_sweeps: 200,
    };
    let report = MatrixFreeBlockGeneralizedLobpcgHp
        .solve(&problem, &block)
        .unwrap();
    assert_eq!(report.status, ResultStatus::Converged);
    for (pair, exact) in report.retained_eigenpairs.iter().zip([100, 39]) {
        let difference = Float::with_val(p, &pair.eigenvalue - exact).abs();
        assert!(difference < bound, "{} vs {exact}", pair.eigenvalue);
    }
}
