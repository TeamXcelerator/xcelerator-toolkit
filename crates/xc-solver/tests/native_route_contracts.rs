use std::sync::atomic::{AtomicUsize, Ordering};
use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionPolicy, Reproducibility, ResultStatus,
    SolverConfig, StoppingPolicy, Subspace,
};
use xc_operator::{MatrixFreeSymmetricF64, MatrixStructure};
use xc_solver::{
    cross_check_f64, installed_solver_capability_catalog, plan_symmetric_eigenproblem,
    DenseReferenceSolverF64, EigenSolverF64, EigenpairReportF64, LanczosSolverF64, SolverError,
    SolverPlannerInput, SolverRoute, SymmetricProblemF64,
};

fn config() -> SolverConfig {
    SolverConfig {
        target: EigenTarget::AlgebraicLargest,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        stopping: StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-300").unwrap(),
            scaled_backward_error: DecimalLiteral::new("1e-7").unwrap(),
            maximum_iterations: 20,
            minimum_iterations: 1,
        },
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: vec![],
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    }
}
fn matrix(bound: f64) -> MatrixFreeSymmetricF64 {
    MatrixFreeSymmetricF64::exact("independent-integer-three", 3, Some(bound), |x, y| {
        y[0] = 4.0 * x[0] + x[1];
        y[1] = x[0] + 3.0 * x[1] + x[2];
        y[2] = x[1] + 2.0 * x[2];
        Ok(())
    })
    .unwrap()
}
#[test]
fn loose_norm_bounds_do_not_relax_native_convergence_or_agreement() {
    // Characteristic polynomial (lambda-3)(lambda^2-6lambda+6).
    let expected = 3.0 + 3.0f64.sqrt();
    for bound in [5.0, 1e8, 1e30] {
        let a = matrix(bound);
        let problem = SymmetricProblemF64::new(&a);
        let report = LanczosSolverF64::default()
            .solve(&problem, &config())
            .unwrap();
        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.iterations > 1);
        assert!((report.eigenvalue - expected).abs() < 1e-12);
        let x = &report.eigenvector;
        let ax = [
            4.0 * x[0] + x[1],
            x[0] + 3.0 * x[1] + x[2],
            x[1] + 2.0 * x[2],
        ];
        let norm = |v: &[f64]| v.iter().fold(0.0f64, |n, x| n.hypot(*x));
        let residual: Vec<_> = ax
            .iter()
            .zip(x)
            .map(|(a, x)| a - report.eigenvalue * x)
            .collect();
        let independent_ratio = norm(&residual) / (norm(&ax) + report.eigenvalue.abs() * norm(x));
        assert!(independent_ratio < 1e-7);
        assert!((report.scaled_backward_error - independent_ratio).abs() < 1e-15);
        let checked = cross_check_f64(
            &LanczosSolverF64::default(),
            &DenseReferenceSolverF64::default(),
            &problem,
            &config(),
            1e-7,
        )
        .unwrap();
        assert_eq!(checked.accepted.assurance, AssuranceLevel::CrossChecked);
    }
}
static CALLS: AtomicUsize = AtomicUsize::new(0);
struct Spoof;
impl EigenSolverF64 for Spoof {
    fn name(&self) -> &'static str {
        DenseReferenceSolverF64::default().name()
    }
    fn solve(
        &self,
        p: &SymmetricProblemF64<'_>,
        c: &SolverConfig,
    ) -> Result<EigenpairReportF64, SolverError> {
        CALLS.fetch_add(1, Ordering::SeqCst);
        let mut r = LanczosSolverF64::default().solve(p, c)?;
        r.algorithm = self.name().into();
        r.global_target_ordering_established = true;
        Ok(r)
    }
}
#[test]
fn report_names_cannot_spoof_execution_independence() {
    let a = matrix(5.0);
    assert!(matches!(
        cross_check_f64(
            &Spoof,
            &LanczosSolverF64::default(),
            &SymmetricProblemF64::new(&a),
            &config(),
            1e-7
        ),
        Err(SolverError::CrossCheckDisagreement(_))
    ));
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
}
#[test]
fn plans_deliver_vectors_and_requested_count_or_explicitly_reject() {
    let catalog = installed_solver_capability_catalog();
    for structure in [
        MatrixStructure::Dense,
        MatrixStructure::Tridiagonal,
        MatrixStructure::Diagonal,
        MatrixStructure::MatrixFree,
    ] {
        for count in [1, 3, 6] {
            let input = SolverPlannerInput {
                structure: structure.clone(),
                dimension: 6,
                target: EigenTarget::AlgebraicLargest,
                requested_eigenpairs: count,
                assurance: AssuranceLevel::Computed,
                precision: PrecisionPolicy::fixed(53),
                matrix_materialized: true,
                generalized: false,
            };
            let plan = plan_symmetric_eigenproblem(&input).unwrap();
            let capability = catalog
                .solvers
                .iter()
                .find(|c| c.id == plan.primary.id())
                .unwrap();
            assert!(capability.delivers_eigenvectors);
            assert!(capability
                .maximum_eigenpairs
                .is_none_or(|maximum| count <= maximum));
            if structure == MatrixStructure::Tridiagonal {
                assert_ne!(plan.primary, SolverRoute::TridiagonalSturmSelected);
            }
        }
    }
    let input = SolverPlannerInput {
        structure: MatrixStructure::Tridiagonal,
        dimension: 6,
        target: EigenTarget::IndexRange { first: 1, last: 3 },
        requested_eigenpairs: 3,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        matrix_materialized: true,
        generalized: false,
    };
    assert!(matches!(
        plan_symmetric_eigenproblem(&input),
        Err(SolverError::UnsupportedTarget(_))
    ));
    let generalized = SolverPlannerInput {
        structure: MatrixStructure::Banded { lower: 1, upper: 1 },
        target: EigenTarget::AlgebraicLargest,
        requested_eigenpairs: 1,
        generalized: true,
        ..input
    };
    assert_eq!(
        plan_symmetric_eigenproblem(&generalized).unwrap().primary,
        SolverRoute::MatrixFreeGeneralizedLobpcg
    );
}
