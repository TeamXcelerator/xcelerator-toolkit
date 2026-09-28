use xc_core::{EigenTarget, ResultStatus};
use xc_operator::{
    DiagonalF64, GeneralizedEigenProblem, LinearOperator, OperatorError, OperatorMetadata,
    PositiveDefiniteMetric, SymmetricOperator,
};
use xc_solver::{GeneralizedExtremeConfigF64, MatrixFreeLobpcgF64};
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
}
impl SymmetricOperator<f64> for Metric {}
impl PositiveDefiniteMetric<f64> for Metric {}
#[test]
fn exact_extreme_and_nonextreme_warm_starts_receive_actual_stability_updates() {
    let a = DiagonalF64::new("A", vec![1., 4., 9., 16.]).unwrap();
    let b = Metric(DiagonalF64::new("B", vec![1., 2., 3., 4.]).unwrap());
    let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
    let cfg = GeneralizedExtremeConfigF64 {
        target: EigenTarget::AlgebraicLargest,
        absolute_residual_tolerance: 1e-12,
        scaled_backward_error_tolerance: 1e-12,
        ritz_value_stability_tolerance: 1e-12,
        maximum_iterations: 50,
        minimum_iterations: 5,
    };
    for start in [[0., 1., 0., 0.], [0., 0., 0., 1.]] {
        let r = MatrixFreeLobpcgF64
            .solve_with_initial_vector(&problem, &cfg, &start)
            .unwrap();
        assert_eq!(r.status, ResultStatus::Converged);
        assert!(r.iterations >= 5);
        assert!(r.projected_factorizations >= 4);
        assert!(r.ritz_value_stability_observed);
        assert!(r.ritz_value_stability <= 1e-12);
        assert!((r.eigenvalue - 4.).abs() < 1e-12); // Exact generalized spectrum 1,2,3,4.
        let mut residual = 0.0f64;
        let mut metric_norm = 0.0;
        for (i, x) in r.eigenvector.iter().enumerate() {
            residual =
                residual.hypot(((i + 1) * (i + 1)) as f64 * x - r.eigenvalue * (i + 1) as f64 * x);
            metric_norm += (i + 1) as f64 * x * x;
        }
        assert!(residual < 1e-12);
        assert!((metric_norm - 1.).abs() < 1e-12);
    }
    let mut once = cfg;
    once.maximum_iterations = 1;
    once.minimum_iterations = 1;
    let r = MatrixFreeLobpcgF64
        .solve_with_initial_vector(&problem, &once, &[0., 0., 0., 1.])
        .unwrap();
    assert_eq!(r.status, ResultStatus::Approximate);
    assert!(!r.ritz_value_stability_observed);
    assert_eq!(r.projected_factorizations, 0);
}
#[cfg(feature = "hp-reference")]
mod exact_width {
    use rug::Rational;
    use xc_solver::{
        diagonal_rank_one_spectrum_f64, diagonal_rank_one_spectrum_with_tolerances_f64,
        RankOneSecularToleranceF64,
    };
    fn q(x: f64) -> Rational {
        Rational::from_f64(x).unwrap()
    }
    fn polynomial(d: &[f64], u: &[f64], alpha: f64, x: f64) -> Rational {
        let x = q(x);
        let mut p = Rational::from(1);
        for v in d {
            p *= q(*v) - &x;
        }
        for (i, ui) in u.iter().enumerate() {
            let mut term = q(alpha) * q(*ui) * q(*ui);
            for (j, dj) in d.iter().enumerate() {
                if i != j {
                    term *= q(*dj) - &x;
                }
            }
            p += term;
        }
        p
    }
    #[test]
    fn rank_one_relative_policy_encloses_tiny_and_ordinary_roots_independently() {
        for exponent in [-200, -300] {
            let d = [0., 1.];
            let u = [2.0f64.powi(exponent), 1.];
            let tolerance = RankOneSecularToleranceF64 {
                absolute_width: 0.,
                relative_width: 1e-12,
            };
            let report =
                diagonal_rank_one_spectrum_with_tolerances_f64(&d, &u, 1., tolerance, 900).unwrap();
            assert_eq!(report.tolerance_policy, Some(tolerance));
            assert_eq!(report.enclosures.len(), 2);
            for (root, b) in report.eigenvalues.iter().zip(&report.enclosures) {
                let width = q(b.upper) - q(b.lower);
                assert!(width <= q(tolerance.relative_width) * q(root.abs()));
                assert!(polynomial(&d, &u, 1., b.lower) * polynomial(&d, &u, 1., b.upper) <= 0);
                assert!(q(b.absolute_width_upper_bound.unwrap()) >= width);
                assert!(q(b.relative_width_upper_bound.unwrap()) * q(root.abs()) >= width);
            }
            // Independently, lambda_small solves lambda^2-(2+u0^2)lambda+u0^2=0.
            let tiny_square = q(u[0]) * q(u[0]);
            let low = &report.enclosures[0];
            assert!(q(low.lower) > tiny_square.clone() / 3);
            assert!(q(low.upper) < tiny_square);
            let legacy = diagonal_rank_one_spectrum_f64(&d, &u, 1., 1e-12, 200).unwrap();
            assert_eq!(legacy.tolerance_policy.unwrap().absolute_width, 1e-12);
            assert!(legacy.enclosures[0].relative_width_upper_bound.unwrap() > 1e-3);
        }
        let d = [-2., 1., 4.];
        let u = [1., 0.5, 2.];
        let tolerance = RankOneSecularToleranceF64 {
            absolute_width: 1e-15,
            relative_width: 1e-12,
        };
        let report =
            diagonal_rank_one_spectrum_with_tolerances_f64(&d, &u, 0.75, tolerance, 300).unwrap();
        for b in report.enclosures {
            assert!(polynomial(&d, &u, 0.75, b.lower) * polynomial(&d, &u, 0.75, b.upper) <= 0);
        }
    }
}

#[test]
fn rank_one_values_plan_executes_explicit_policy_without_generic_eigenpair_dispatch() {
    use xc_core::{AssuranceLevel, PrecisionPolicy};
    use xc_operator::MatrixStructure;
    use xc_solver::*;
    let policy = RankOneSecularToleranceF64 {
        absolute_width: 0.0,
        relative_width: 1e-12,
    };
    for (diagonal, vector, alpha, expected) in [
        (
            vec![1., 3.],
            vec![1., 1.],
            -0.5,
            vec![2. - (1.25f64).sqrt() - 0.5, 2. + (1.25f64).sqrt() - 0.5],
        ),
        (
            vec![0., 2.],
            vec![1., 1.],
            1.,
            vec![2. - 2f64.sqrt(), 2. + 2f64.sqrt()],
        ),
    ] {
        let plan =
            plan_diagonal_rank_one_spectrum_f64(&diagonal, &vector, alpha, policy, 300).unwrap();
        assert_eq!(plan.tolerance(), policy);
        let report = plan.execute().unwrap();
        // Closed-form quadratic eigenvalues, independent of secular bisection.
        for (actual, wanted) in report.eigenvalues.iter().zip(expected) {
            assert!((actual - wanted).abs() < 2e-12 * wanted.abs());
        }
        let encoded = serde_json::to_value(&plan).unwrap();
        let restored: RankOneSecularPlanF64 = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(restored.execute().unwrap().eigenvalues, report.eigenvalues);
        let mut invalid = encoded;
        invalid["vector"] = serde_json::json!([0., 1.]);
        let malformed: RankOneSecularPlanF64 = serde_json::from_value(invalid).unwrap();
        assert!(malformed.execute().is_err());
    }
    let catalog = installed_solver_capability_catalog();
    let scalar = catalog
        .solvers
        .iter()
        .find(|r| r.id == "diagonal_rank_one_secular")
        .unwrap();
    assert!(!scalar.delivers_eigenvectors);
    assert!(scalar
        .operator_representations
        .contains("diagonal_rank_one"));
    let plan = plan_symmetric_eigenproblem(&SolverPlannerInput {
        structure: MatrixStructure::RankOneUpdate,
        dimension: 4,
        target: EigenTarget::AlgebraicLargest,
        requested_eigenpairs: 1,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        matrix_materialized: false,
        generalized: false,
    })
    .unwrap();
    assert_ne!(plan.primary.id(), scalar.id);
}

#[cfg(feature = "hp-reference")]
#[test]
fn dense_reference_precision_floor_has_independent_stored_data_residual_controls() {
    use rug::{Float, Rational};
    use xc_core::{
        AssuranceLevel, DecimalLiteral, PrecisionPolicy, Reproducibility, SolverConfig,
        StoppingPolicy, Subspace, TerminationReason,
    };
    use xc_solver::{solve_dense_reference_hp, DenseSymmetricProblemHp};
    let hilbert: Vec<_> = (0..9)
        .map(|ij| Float::with_val(64, Rational::from((1, ij / 3 + ij % 3 + 1))))
        .collect();
    let extra: Vec<_> = [-3, 1, 0, 1, 2, 1, 0, 1, 5]
        .into_iter()
        .map(|x| Float::with_val(64, x))
        .collect();
    let parse = |s: &str, bits| Float::with_val(bits, Float::parse(s).unwrap());
    for matrix in [hilbert, extra] {
        let problem = DenseSymmetricProblemHp::new(&matrix, 3).unwrap();
        let mut config = SolverConfig {
            target: EigenTarget::AlgebraicLargest,
            subspace: Subspace::Full,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(64),
            stopping: StoppingPolicy {
                absolute_residual: DecimalLiteral::new("1e-60").unwrap(),
                scaled_backward_error: DecimalLiteral::new("1e-60").unwrap(),
                maximum_iterations: 100,
                minimum_iterations: 1,
            },
            reproducibility: Reproducibility::Deterministic,
            algorithm_preferences: vec![],
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        };
        let low = solve_dense_reference_hp(&problem, &config).unwrap();
        assert_eq!(low.status, ResultStatus::Approximate);
        assert_eq!(low.termination, TerminationReason::MaximumPrecision);
        let lambda = parse(&low.eigenvalue, 64).to_rational().unwrap();
        let x: Vec<_> = low
            .eigenvector
            .iter()
            .map(|v| parse(v, 64).to_rational().unwrap())
            .collect();
        let mut residual_sq = Rational::new();
        for i in 0..3 {
            let mut residual = -Rational::from(&lambda * &x[i]);
            for j in 0..3 {
                residual += matrix[3 * i + j].to_rational().unwrap() * &x[j];
            }
            residual_sq += Rational::from(&residual * &residual);
        }
        // Exact rational evaluation of the returned stored-data pair is above
        // the requested floor yet far below ordinary candidate accuracy.
        assert!(Float::with_val(512, &residual_sq) > parse("1e-120", 512));
        assert!(Float::with_val(512, &residual_sq) < parse("1e-32", 512));
        config.precision = PrecisionPolicy::fixed(256);
        let high = solve_dense_reference_hp(&problem, &config).unwrap();
        assert_eq!(high.status, ResultStatus::Converged);
        assert!(parse(&high.residual_norm, 512) < parse("1e-60", 512));
        assert!(
            (parse(&high.eigenvalue, 512) - parse(&low.eigenvalue, 512)).abs()
                < parse("1e-17", 512)
        );
    }
}

#[test]
fn rank_one_catalog_preflight_accepts_values_and_rejects_vector_requirements() {
    use xc_core::*;
    use xc_solver::installed_solver_capability_catalog;
    let catalog = installed_solver_capability_catalog();
    let mut request = PreflightRequest {
        effective_config_digest: ConfigDigest("a".repeat(64)),
        platform: "linux".into(),
        scalar_backend: "f64".into(),
        precision_bits: 53,
        operator_representation: "diagonal_rank_one".into(),
        target_kind: "full_spectrum".into(),
        generalized: false,
        require_eigenvectors: false,
        requested_eigenpairs: Some(2),
        primary_solver: "diagonal_rank_one_secular".into(),
        independent_solver: None,
        requested_assurance: AssuranceLevel::Computed,
        certification_route: None,
        certification_claim: None,
        complex_claim: false,
        checkpoint_requested: false,
        cache_mode: CacheAccessMode::Disabled,
        cache_policy_digest: None,
        cache_validation_mode: None,
        authenticated_principal: None,
        publication: PublicationPreflightRequest::default(),
        resources: ResourcePolicy::default(),
        estimate: ResourceEstimate::default(),
    };
    let report = catalog.preflight(&request);
    assert!(report.accepted, "{:?}", report.failures);
    request.require_eigenvectors = true;
    assert!(catalog
        .preflight(&request)
        .failures
        .iter()
        .any(|f| f.code == PreflightFailureCode::UnsupportedEigenvectors));
    request.require_eigenvectors = false;
    request.operator_representation = "rank_one_update".into();
    assert!(catalog
        .preflight(&request)
        .failures
        .iter()
        .any(|f| f.code == PreflightFailureCode::UnsupportedOperatorRepresentation));
}
