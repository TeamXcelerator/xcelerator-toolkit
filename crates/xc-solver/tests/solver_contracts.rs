#![cfg(feature = "hp-reference")]
use rug::Float;
use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionEscalation, PrecisionPolicy, ResultStatus,
};
use xc_operator::{DenseSymmetricHp, MatrixStructure};
use xc_solver::*;
fn literal(s: &str) -> DecimalLiteral {
    DecimalLiteral::new(s).unwrap()
}

#[test]
fn bisection_work_limit_does_not_trigger_precision_escalation() {
    let p = 256;
    let d = vec![Float::with_val(p, 2); 8];
    let e = vec![Float::with_val(p, -1); 7];
    let problem = TridiagonalProblemHp::new(&d, &e).unwrap();
    let result = solve_tridiagonal_selected_eigenpairs_adaptive_hp(
        &problem,
        &HpAdaptiveSelectedTridiagonalOptions {
            first_index: 1,
            last_index: 1,
            absolute_tolerance: literal("1e-30"),
            maximum_bisection_iterations: 1,
            eigenvector_options: TridiagEigvecOptions::default(),
            precision: PrecisionPolicy {
                initial_bits: 64,
                maximum_bits: p,
                guard_bits: 0,
                escalation: PrecisionEscalation::AddBits(64),
            },
        },
    )
    .unwrap();
    let HpAdaptiveSelectedTridiagonalResult::Inconclusive {
        attempts, reason, ..
    } = result
    else {
        panic!("budget must be inconclusive")
    };
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, ResultStatus::Inconclusive);
    assert!(reason.contains("budget exhausted"));
}

#[test]
fn exact_decimal_interval_midpoint_is_admitted_and_guards_are_not_requested_residuals() {
    for p in [64, 128, 192] {
        let d = ["0.15", "0.27", "0.8"];
        let mut a = vec![Float::with_val(p, 0); 9];
        for (i, value) in d.iter().enumerate() {
            a[3 * i + i] = Float::with_val(p, Float::parse(value).unwrap());
        }
        let operator =
            DenseSymmetricHp::new("decimal interval", 3, a.clone(), p, &Float::with_val(p, 0))
                .unwrap();
        let factor =
            DenseShiftInvertFactorizationHp::factor("decimal midpoint", 3, &a, literal("0.2"), p)
                .unwrap();
        let config = BlockShiftInvertConfigHp {
            target: EigenTarget::Interval {
                lower: literal("0.1"),
                upper: literal("0.3"),
            },
            precision_bits: p,
            requested_eigenpairs: 1,
            guard_eigenpairs: 2,
            absolute_residual_tolerance: literal("1e-12"),
            scaled_backward_error_tolerance: literal("1e-12"),
            ritz_value_stability_tolerance: literal("1e-12"),
            boundary_cluster_tolerance: literal("1e-10"),
            maximum_iterations: 20,
            minimum_iterations: 2,
            maximum_projected_sweeps: 100,
        };
        let result = BlockShiftInvertSolverHp
            .solve(&operator, &factor, &config)
            .unwrap();
        assert_eq!(result.status, ResultStatus::Converged);
        assert!(
            Float::with_val(
                p,
                &result.retained_eigenpairs[0].eigenvalue
                    - Float::with_val(p, Float::parse("0.15").unwrap())
            )
            .abs()
                < Float::with_val(p, Float::parse("1e-12").unwrap())
        );
    }
}

#[test]
fn hp_planner_accounts_for_headers_limbs_and_dense_workspace() {
    for p in [64, 256, 1024] {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 1000,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(p),
            matrix_materialized: true,
            generalized: false,
        };
        let plan = plan_symmetric_eigenproblem(&input).unwrap();
        let scalar = std::mem::size_of::<Float>() as u64 + u64::from(p).div_ceil(64) * 8;
        let resident = plan.resource_estimate.resident_memory_bytes.unwrap();
        let temporary = plan.resource_estimate.temporary_memory_bytes.unwrap();
        assert!(
            resident >= 1000 * scalar,
            "resident omitted live MPFR headers"
        );
        if plan.requires_materialization {
            assert!(
                resident + temporary >= 4_000_000 * scalar,
                "dense work omitted matrices"
            );
        }
    }
}
