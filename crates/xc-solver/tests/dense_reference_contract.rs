use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionPolicy, Reproducibility, ResultStatus,
    SolverConfig, StoppingPolicy, Subspace,
};
use xc_operator::{DenseSymmetricF64, DiagonalF64};
use xc_solver::{DenseReferenceSolverF64, EigenSolverF64, SymmetricProblemF64};

fn config(target: EigenTarget) -> SolverConfig {
    SolverConfig {
        target,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        stopping: StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-300").unwrap(),
            scaled_backward_error: DecimalLiteral::new("1e-12").unwrap(),
            maximum_iterations: 100,
            minimum_iterations: 2,
        },
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: Vec::new(),
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    }
}

#[test]
fn dense_reference_rejects_invalid_symmetry_tolerances() {
    let op = DiagonalF64::new("control", vec![1., 2.]).unwrap();
    for symmetry_tolerance in [f64::NAN, f64::INFINITY, -1.] {
        let solver = DenseReferenceSolverF64 {
            symmetry_tolerance,
            ..Default::default()
        };
        assert!(solver
            .solve(
                &SymmetricProblemF64::new(&op),
                &config(EigenTarget::AlgebraicLargest)
            )
            .is_err());
    }
}

#[test]
fn dense_reference_preserves_representable_extreme_off_diagonal_spectra() {
    for scale in [2f64.powi(-1000), 1., f64::MAX * 0.75] {
        let op =
            DenseSymmetricF64::new("exact two by two", 2, vec![0., scale, scale, 0.], 0.).unwrap();
        for (target, sign) in [
            (EigenTarget::AlgebraicLargest, 1.),
            (EigenTarget::AlgebraicSmallest, -1.),
        ] {
            let result = DenseReferenceSolverF64::default()
                .solve(&SymmetricProblemF64::new(&op), &config(target))
                .unwrap();
            assert_eq!(result.status, ResultStatus::Converged);
            assert!((result.eigenvalue / scale - sign).abs() < 1e-14);
            let x = &result.eigenvector;
            let residual = (x[1] - sign * x[0]).hypot(x[0] - sign * x[1]);
            assert!(residual < 1e-14);
            assert!(result.scaled_backward_error.is_finite());
        }
    }
}

#[test]
fn native_routes_reject_unapplied_subspaces_and_high_precision_requests() {
    let op = DenseSymmetricF64::new("even=1 odd=3", 2, vec![2., -1., -1., 2.], 0.).unwrap();
    let solvers: Vec<Box<dyn EigenSolverF64>> = vec![
        Box::new(DenseReferenceSolverF64::default()),
        Box::new(xc_solver::LanczosSolverF64::default()),
        Box::new(xc_solver::ShiftedPowerSolverF64),
    ];
    for solver in solvers {
        for subspace in [
            Subspace::EvenReflection,
            Subspace::OddReflection,
            Subspace::ExplicitBasis {
                ambient_dimension: 2,
                reduced_dimension: 1,
                basis_id: "even".into(),
            },
            Subspace::Projector {
                ambient_dimension: 2,
                reduced_dimension: 1,
                projector_id: "even".into(),
            },
        ] {
            let mut cfg = config(EigenTarget::AlgebraicLargest);
            cfg.subspace = subspace;
            assert!(
                solver.solve(&SymmetricProblemF64::new(&op), &cfg).is_err(),
                "{} silently ignored {:?}",
                solver.name(),
                cfg.subspace
            );
        }
        for bits in [54, 64, 128] {
            let mut cfg = config(EigenTarget::AlgebraicLargest);
            cfg.precision = PrecisionPolicy::fixed(bits);
            assert!(
                solver.solve(&SymmetricProblemF64::new(&op), &cfg).is_err(),
                "{} silently accepted {bits} bits",
                solver.name()
            );
        }
    }
}

#[test]
fn planner_uses_significand_precision_and_includes_guard_bits() {
    use xc_solver::{
        installed_solver_capability_catalog, plan_symmetric_eigenproblem, SolverPlannerInput,
        SolverRoute,
    };
    let catalog = installed_solver_capability_catalog();
    assert_eq!(
        catalog
            .scalar_backends
            .iter()
            .find(|x| x.id == "f64")
            .unwrap()
            .maximum_precision_bits,
        Some(53)
    );
    for bits in [53, 54, 63, 64, 65, 128] {
        let input = SolverPlannerInput {
            structure: xc_operator::MatrixStructure::Dense,
            dimension: 2,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(bits),
            matrix_materialized: true,
            generalized: false,
        };
        let plan = plan_symmetric_eigenproblem(&input).unwrap();
        assert_eq!(
            plan.primary,
            if bits <= 53 {
                SolverRoute::DenseFullSpectrumReference
            } else {
                SolverRoute::HpThickRestartLanczos
            }
        );
    }
    let input = SolverPlannerInput {
        structure: xc_operator::MatrixStructure::Dense,
        dimension: 2,
        target: EigenTarget::AlgebraicLargest,
        requested_eigenpairs: 1,
        assurance: AssuranceLevel::CrossChecked,
        precision: PrecisionPolicy {
            initial_bits: 48,
            maximum_bits: 128,
            guard_bits: 16,
            escalation: xc_core::PrecisionEscalation::AddBits(32),
        },
        matrix_materialized: true,
        generalized: false,
    };
    let plan = plan_symmetric_eigenproblem(&input).unwrap();
    assert_eq!(plan.primary, SolverRoute::HpThickRestartLanczos);
    assert_eq!(plan.precision_schedule_bits, vec![64, 96]);
}

#[cfg(feature = "hp-reference")]
#[test]
fn dense_hp_reference_applies_requested_guard_precision() {
    use rug::Float;
    use xc_solver::{solve_dense_reference_hp, DenseSymmetricProblemHp};
    let values = [3, 1, 1, 3].map(|v| Float::with_val(128, v));
    let problem = DenseSymmetricProblemHp::new(&values, 2).unwrap();
    let mut cfg = config(EigenTarget::AlgebraicLargest);
    cfg.precision = PrecisionPolicy {
        initial_bits: 64,
        maximum_bits: 128,
        guard_bits: 32,
        escalation: xc_core::PrecisionEscalation::Fixed,
    };
    let report = solve_dense_reference_hp(&problem, &cfg).unwrap();
    assert_eq!(report.precision_bits, 96);
    let value = Float::with_val(96, Float::parse(&report.eigenvalue).unwrap());
    assert!(Float::with_val(96, value - 4).abs() < Float::with_val(96, 1) >> 75u32);
    // The source integers are unchanged by requested-precision conversion.
    // The exact 2x2 spectrum is {2,4}; evaluate returned stored decimals with
    // exact rationals rather than accepting the reported residual alone.
    let lambda = Float::with_val(96, Float::parse(&report.eigenvalue).unwrap())
        .to_rational()
        .unwrap();
    let x: Vec<_> = report
        .eigenvector
        .iter()
        .map(|v| {
            Float::with_val(96, Float::parse(v).unwrap())
                .to_rational()
                .unwrap()
        })
        .collect();
    let mut residual_sq = rug::Rational::new();
    let mut norm_sq = rug::Rational::new();
    for i in 0..2 {
        let mut residual = (rug::Rational::from(3) - &lambda) * &x[i];
        residual += &x[1 - i];
        residual_sq += rug::Rational::from(&residual * &residual);
        norm_sq += rug::Rational::from(&x[i] * &x[i]);
    }
    let error = rug::Rational::from((1, rug::Integer::from(1) << 75));
    assert!((lambda - rug::Rational::from(4)).abs() < error);
    assert!((norm_sq - rug::Rational::from(1)).abs() < error);
    assert!((x[0].clone() - &x[1]).abs() < error);
    assert!(residual_sq < error.clone() * error);
    assert_eq!(report.status, ResultStatus::Converged);
}

#[test]
fn dense_reference_rejects_nonfinite_actions_before_decomposition() {
    use xc_operator::{LinearOperator, OperatorError, OperatorMetadata, SymmetricOperator};
    struct Invalid;
    impl LinearOperator<f64> for Invalid {
        fn dimension(&self) -> usize {
            3
        }
        fn apply(&self, _: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
            y.fill(f64::NAN);
            Ok(())
        }
        fn metadata(&self) -> OperatorMetadata {
            OperatorMetadata::new(
                "nonfinite callback",
                3,
                xc_operator::MatrixStructure::MatrixFree,
                "f64",
            )
        }
    }
    impl SymmetricOperator<f64> for Invalid {}
    assert!(DenseReferenceSolverF64::default()
        .solve(
            &SymmetricProblemF64::new(&Invalid),
            &config(EigenTarget::AlgebraicLargest)
        )
        .is_err());
}

#[test]
fn dense_reference_pairs_tiny_eigenvalues_with_their_own_vectors() {
    // The library QR left this 2x2 tail unrotated but relabeled its values,
    // so the smallest target returned 2e-10 with a converged residual.
    let op = DenseSymmetricF64::new(
        "tiny tail",
        3,
        vec![1., 0., 0., 0., 1e-10, 1e-20, 0., 1e-20, 2e-10],
        0.,
    )
    .unwrap();
    let result = DenseReferenceSolverF64::default()
        .solve(
            &SymmetricProblemF64::new(&op),
            &config(EigenTarget::AlgebraicSmallest),
        )
        .unwrap();
    assert_eq!(result.status, ResultStatus::Converged);
    assert!(
        (result.eigenvalue - 1e-10).abs() < 1e-24,
        "{}",
        result.eigenvalue
    );
    assert!((result.eigenvector[1].abs() - 1.0).abs() < 1e-9);
}
