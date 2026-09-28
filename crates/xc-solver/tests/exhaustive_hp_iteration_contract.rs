#![cfg(feature = "hp-reference")]
use rug::{ops::Pow, Float};
use xc_core::{DecimalLiteral, EigenTarget, ResultStatus, TerminationReason};
use xc_operator::{
    DenseSymmetricHp, GeneralizedEigenProblem, LinearOperator, OperatorError, OperatorMetadata,
    PositiveDefiniteMetric, SymmetricOperator,
};
use xc_solver::*;
fn lit(s: &str) -> DecimalLiteral {
    DecimalLiteral::new(s).unwrap()
}
fn dense(n: usize, scale: i32) -> (DenseSymmetricHp, Vec<Float>) {
    let data: Vec<_> = (0..n * n)
        .map(|i| {
            Float::with_val(128, if i / n == i % n { i / n + 1 } else { 0 })
                * Float::with_val(128, 2).pow(scale)
        })
        .collect();
    (
        DenseSymmetricHp::new(
            "iteration-contract",
            n,
            data.clone(),
            128,
            &Float::with_val(128, 0),
        )
        .unwrap(),
        data,
    )
}
fn block_config() -> BlockShiftInvertConfigHp {
    BlockShiftInvertConfigHp {
        target: EigenTarget::SmallestMagnitude,
        precision_bits: 128,
        requested_eigenpairs: 1,
        guard_eigenpairs: 1,
        absolute_residual_tolerance: lit("1e6"),
        scaled_backward_error_tolerance: lit("1e-100"),
        ritz_value_stability_tolerance: lit("1e6"),
        boundary_cluster_tolerance: lit("1e-10"),
        maximum_iterations: 3,
        minimum_iterations: 2,
        maximum_projected_sweeps: 100,
    }
}
fn krylov_config() -> ShiftInvertKrylovConfigHp {
    ShiftInvertKrylovConfigHp {
        target: EigenTarget::SmallestMagnitude,
        precision_bits: 128,
        requested_eigenpairs: 1,
        guard_eigenpairs: 1,
        maximum_subspace_dimension: 3,
        maximum_restarts: 3,
        minimum_restarts: 2,
        maximum_projected_sweeps: 100,
        absolute_residual_tolerance: lit("1e6"),
        scaled_backward_error_tolerance: lit("1e-100"),
        ritz_value_stability_tolerance: lit("1e6"),
        boundary_cluster_tolerance: lit("1e-10"),
    }
}
#[test]
fn shifted_inverse_rank_is_independent_of_matrix_scale() {
    let (op, data) = dense(2, 200);
    let factor = DenseShiftInvertFactorizationHp::factor("scale", 2, &data, lit("0"), 128).unwrap();
    let mut c = block_config();
    c.scaled_backward_error_tolerance = lit("1e-25");
    c.ritz_value_stability_tolerance = lit("1e40");
    let result = BlockShiftInvertSolverHp.solve(&op, &factor, &c).unwrap();
    assert_eq!(result.status, ResultStatus::Converged);
    let mut relative =
        result.retained_eigenpairs[0].eigenvalue.clone() / Float::with_val(128, 2).pow(200i32);
    relative -= 1;
    assert!(relative.abs() < Float::with_val(128, 2).pow(-80));
}
#[test]
fn initial_basis_cannot_exceed_krylov_subspace_limit() {
    let (op, data) = dense(4, 0);
    let factor = DenseShiftInvertFactorizationHp::factor("basis", 4, &data, lit("0"), 128).unwrap();
    let basis: Vec<Vec<_>> = (0..4)
        .map(|i| {
            (0..4)
                .map(|j| Float::with_val(128, usize::from(i == j)))
                .collect()
        })
        .collect();
    assert!(ShiftInvertKrylovSolverHp
        .solve_with_initial_basis(&op, &factor, &krylov_config(), &basis)
        .is_err());
}
#[test]
fn block_shift_invert_reports_actual_residual_stopping_condition() {
    let (op, data) = dense(4, 0);
    let factor = DenseShiftInvertFactorizationHp::factor("stop", 4, &data, lit("0"), 128).unwrap();
    let report = BlockShiftInvertSolverHp
        .solve(&op, &factor, &block_config())
        .unwrap();
    assert_eq!(report.status, ResultStatus::Converged);
    assert!(
        report.retained_eigenpairs[0].scaled_backward_error
            > Float::with_val(128, Float::parse("1e-100").unwrap())
    );
    assert_eq!(report.termination, TerminationReason::ResidualTolerance);
}
#[test]
fn shift_invert_krylov_reports_actual_residual_stopping_condition() {
    let (op, data) = dense(4, 0);
    let factor = DenseShiftInvertFactorizationHp::factor("stop", 4, &data, lit("0"), 128).unwrap();
    let report = ShiftInvertKrylovSolverHp
        .solve(&op, &factor, &krylov_config())
        .unwrap();
    assert_eq!(report.status, ResultStatus::Converged);
    assert!(
        report.retained_eigenpairs[0].scaled_backward_error
            > Float::with_val(128, Float::parse("1e-100").unwrap())
    );
    assert_eq!(report.termination, TerminationReason::ResidualTolerance);
}
#[test]
fn thick_restart_reports_actual_residual_stopping_condition() {
    let (op, _) = dense(4, 0);
    let c = ThickRestartLanczosConfigHp {
        target: EigenTarget::AlgebraicSmallest,
        precision_bits: 128,
        requested_eigenpairs: 1,
        guard_eigenpairs: 1,
        maximum_subspace_dimension: 3,
        maximum_restarts: 3,
        minimum_restarts: 2,
        maximum_projected_sweeps: 100,
        absolute_residual_tolerance: lit("1e6"),
        scaled_backward_error_tolerance: lit("1e-100"),
        ritz_value_stability_tolerance: lit("1e6"),
        boundary_cluster_tolerance: lit("1e-10"),
    };
    let report = ThickRestartLanczosHp.solve(&op, &c).unwrap();
    assert_eq!(report.status, ResultStatus::Converged);
    assert!(
        report.retained_eigenpairs[0].scaled_backward_error
            > Float::with_val(128, Float::parse("1e-100").unwrap())
    );
    assert_eq!(report.termination, TerminationReason::ResidualTolerance);
}
struct Identity(usize);
impl LinearOperator<Float> for Identity {
    fn dimension(&self) -> usize {
        self.0
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        y.clone_from_slice(x);
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        OperatorMetadata::new(
            "identity",
            self.0,
            xc_operator::MatrixStructure::Diagonal,
            "rug_mpfr",
        )
    }
}
impl SymmetricOperator<Float> for Identity {}
impl PositiveDefiniteMetric<Float> for Identity {}
#[test]
fn generalized_block_reports_actual_residual_stopping_condition() {
    let (op, _) = dense(4, 0);
    let metric = Identity(4);
    let problem = GeneralizedEigenProblem::new(&op, &metric).unwrap();
    let c = BlockGeneralizedConfigHp {
        target: EigenTarget::AlgebraicSmallest,
        precision_bits: 128,
        requested_eigenpairs: 2,
        guard_eigenpairs: 1,
        absolute_residual_tolerance: lit("1e6"),
        scaled_backward_error_tolerance: lit("1e-100"),
        ritz_value_stability_tolerance: lit("1e6"),
        boundary_cluster_tolerance: lit("1e-10"),
        maximum_iterations: 3,
        minimum_iterations: 2,
        maximum_projected_sweeps: 100,
    };
    let report = MatrixFreeBlockGeneralizedLobpcgHp
        .solve(&problem, &c)
        .unwrap();
    assert_eq!(report.status, ResultStatus::Converged);
    assert!(
        report.retained_eigenpairs[0].scaled_backward_error
            > Float::with_val(128, Float::parse("1e-100").unwrap())
    );
    assert_eq!(report.termination, TerminationReason::ResidualTolerance);
}
