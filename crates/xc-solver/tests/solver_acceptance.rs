//! Independent exact manufactured spectra and acceptance-contract controls.
use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionPolicy, Reproducibility, SolverConfig,
    StoppingPolicy, Subspace,
};
use xc_operator::{
    DiagonalF64, LinearOperator, MatrixStructure, OperatorError, OperatorMetadata,
    RankOneUpdateF64, SymmetricOperator, TridiagonalF64,
};
use xc_solver::*;
fn config() -> SolverConfig {
    SolverConfig {
        target: EigenTarget::AlgebraicLargest,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        stopping: StoppingPolicy {
            absolute_residual: DecimalLiteral::new("1e-12").unwrap(),
            scaled_backward_error: DecimalLiteral::new("1e-12").unwrap(),
            maximum_iterations: 200,
            minimum_iterations: 2,
        },
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: vec![],
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    }
}
#[test]
fn common_seed_wrong_extreme_cannot_earn_cross_checked() {
    let mut seed: Vec<_> = (0..8)
        .map(|i| 1.0 / (i + 1) as f64 + ((i % 7) as f64 - 3.0) * 1e-4)
        .collect();
    let length = seed.iter().map(|x| x * x).sum::<f64>().sqrt();
    seed.iter_mut().for_each(|x| *x /= length);
    // Exact rank-one formula gives eigenvalues 1 and 5. Both seeded solvers
    // stay near the 1-eigenvector; this is not evidence for the requested 5.
    let base = DiagonalF64::new("5I", vec![5.0; 8]).unwrap();
    let op = RankOneUpdateF64::new(&base, -4.0, seed).unwrap();
    let problem = SymmetricProblemF64::new(&op);
    assert!(cross_check_f64(
        &LanczosSolverF64::default(),
        &ShiftedPowerSolverF64,
        &problem,
        &config(),
        1e-8
    )
    .is_err());
    let a = SolverRoute::LanczosExtremeReference.evidence(53, Some(2));
    let b = SolverRoute::ShiftedPowerExtremeReference.evidence(53, Some(2));
    assert!(a.seed.is_some());
    assert_eq!(a.seed, b.seed);
    assert!(!a
        .decisive_intermediates
        .is_disjoint(&b.decisive_intermediates));
    let assessment = xc_core::assess_route_independence(
        &a,
        &b,
        &xc_core::IndependenceDeclaration {
            intended_claim: "algebraic maximum".into(),
            rationale: "different iterations".into(),
            accepted_shared_inputs: Default::default(),
        },
    );
    assert!(!assessment.independent);
}

struct RawAction {
    a: [f64; 4],
}
impl LinearOperator<f64> for RawAction {
    fn dimension(&self) -> usize {
        2
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        y[0] = self.a[0] * x[0] + self.a[1] * x[1];
        y[1] = self.a[2] * x[0] + self.a[3] * x[1];
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        OperatorMetadata::new("raw", 2, MatrixStructure::MatrixFree, "f64")
    }
}
impl SymmetricOperator<f64> for RawAction {}
#[test]
fn dense_symmetry_admission_is_relative_at_small_and_large_scales() {
    for scale in [1e-100, 1.0, 1e100] {
        let bad = RawAction {
            a: [0.0, scale, 0.0, 0.0],
        };
        assert!(DenseReferenceSolverF64::default()
            .solve(&SymmetricProblemF64::new(&bad), &config())
            .is_err());
        // Deliberately introduce only one ulp of relative application asymmetry.
        let good = RawAction {
            a: [
                2.0 * scale,
                scale,
                scale * (1.0 + f64::EPSILON),
                3.0 * scale,
            ],
        };
        assert!(DenseReferenceSolverF64::default()
            .solve(&SymmetricProblemF64::new(&good), &config())
            .is_ok());
    }
}
#[test]
fn planner_does_not_offer_unimplemented_native_targets_or_cross_checks() {
    let mut input = SolverPlannerInput {
        structure: MatrixStructure::MatrixFree,
        dimension: 8,
        target: EigenTarget::SmallestMagnitude,
        requested_eigenpairs: 1,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::fixed(53),
        matrix_materialized: false,
        generalized: false,
    };
    assert!(plan_symmetric_eigenproblem(&input).is_err());
    input.target = EigenTarget::AlgebraicLargest;
    input.assurance = AssuranceLevel::CrossChecked;
    let plan = plan_symmetric_eigenproblem(&input).unwrap();
    assert!(plan.independent_crosscheck.is_none());
    input.structure = MatrixStructure::Tridiagonal;
    let tridiagonal_plan = plan_symmetric_eigenproblem(&input).unwrap();
    assert!(tridiagonal_plan.independent_crosscheck.is_none());
    let catalog = installed_solver_capability_catalog();
    let capability = catalog
        .solvers
        .iter()
        .find(|route| route.id == tridiagonal_plan.primary.id())
        .unwrap();
    assert!(capability.delivers_eigenvectors);
    assert!(capability
        .maximum_eigenpairs
        .is_none_or(|count| count >= input.requested_eigenpairs));
    // Uniform tridiagonals have the independently known spectrum
    // d + 2 b cos(k*pi/(n+1)); execute the planned vector-producing route.
    for (diagonal, off_diagonal) in [(2.0, -1.0), (3.0, 0.5)] {
        let n = input.dimension;
        let operator = TridiagonalF64::new(
            "analytic tridiagonal",
            vec![diagonal; n],
            vec![off_diagonal; n - 1],
        )
        .unwrap();
        let execution = match tridiagonal_plan.primary {
            SolverRoute::LanczosExtremeReference => LanczosSolverF64::default()
                .solve(&SymmetricProblemF64::new(&operator), &config())
                .unwrap(),
            route => panic!("unexpected native tridiagonal eigenpair route: {route:?}"),
        };
        let expected =
            diagonal + 2.0 * off_diagonal.abs() * (std::f64::consts::PI / (n + 1) as f64).cos();
        assert_eq!(execution.status, xc_core::ResultStatus::Converged);
        assert_eq!(execution.assurance, AssuranceLevel::Computed);
        assert_eq!(execution.eigenvector.len(), n);
        assert!((execution.eigenvalue - expected).abs() < 1e-10);
        let mut residual = 0.0f64;
        for i in 0..n {
            let mut image = diagonal * execution.eigenvector[i];
            if i > 0 {
                image += off_diagonal * execution.eigenvector[i - 1];
            }
            if i + 1 < n {
                image += off_diagonal * execution.eigenvector[i + 1];
            }
            residual = residual.hypot(image - execution.eigenvalue * execution.eigenvector[i]);
        }
        assert!(residual < 1e-10);
        let norm_sq: f64 = execution.eigenvector.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }
    assert!(!installed_solver_capability_catalog()
        .solvers
        .iter()
        .any(|r| r.id == SolverRoute::TridiagonalFullSpectrumReference.id()));
    input.structure = MatrixStructure::Dense;
    input.matrix_materialized = true;
    input.requested_eigenpairs = 3;
    assert_eq!(
        plan_symmetric_eigenproblem(&input).unwrap().primary,
        SolverRoute::BlockSubspaceExtremeReference
    );
}

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
impl xc_operator::PositiveDefiniteMetric<f64> for Metric {}
#[test]
fn minimum_lobpcg_iterations_cannot_be_frozen_vector_recounts() {
    let a = DiagonalF64::new("a", (1..=8).map(f64::from).collect()).unwrap();
    let b = Metric(DiagonalF64::new("b", (1..=8).rev().map(f64::from).collect()).unwrap());
    let problem = xc_operator::GeneralizedEigenProblem::new(&a, &b).unwrap();
    for minimum_iterations in [1, 78] {
        let c = GeneralizedExtremeConfigF64 {
            target: EigenTarget::AlgebraicLargest,
            absolute_residual_tolerance: 1e-9,
            scaled_backward_error_tolerance: 1e-12,
            ritz_value_stability_tolerance: 1e-12,
            maximum_iterations: 100,
            minimum_iterations,
        };
        match MatrixFreeLobpcgF64.solve(&problem, &c) {
            Ok(r) => {
                assert!(
                    r.operator_applications >= r.iterations,
                    "frozen evaluations counted as updates: {r:?}"
                );
                if r.iterations < minimum_iterations {
                    assert_eq!(r.residual_norm, 0.0);
                }
                assert!((r.eigenvalue - 8.0).abs() < 1e-8);
            }
            // Rank exhaustion after actual steps is an honest explicit failure,
            // unlike fabricated stability from re-evaluating a frozen vector.
            Err(SolverError::NumericalBreakdown(message)) => {
                assert!(message.contains("rank") || message.contains("metric norm"))
            }
            Err(error) => panic!("unexpected error: {error}"),
        }
    }
}

#[cfg(feature = "hp-reference")]
mod hp {
    use super::*;
    use rug::{Assign, Float, Rational};
    use xc_operator::DenseSymmetricHp;
    fn verify_count_proof(
        evidence: &BoundaryCountEvidenceHp,
        reported: bool,
        spectrum: &[Float],
        requested: usize,
    ) {
        if let BoundaryCountEvidenceHp::Verified {
            lower_shift,
            upper_shift,
            count,
        } = evidence
        {
            assert!(reported);
            assert_eq!(*count, requested);
            // Exact manufactured spectra determine these separated boundary
            // counts. The margin also covers the rounded 1/26 hidden block.
            let margin = Float::with_val(512, 1) >> 240u32;
            for value in spectrum {
                assert!((Float::with_val(512, value) - upper_shift).abs() > margin);
                if let Some(lower) = lower_shift {
                    assert!((Float::with_val(512, value) - lower).abs() > margin);
                }
            }
            let independent = spectrum
                .iter()
                .filter(|value| {
                    *value < upper_shift && lower_shift.as_ref().is_none_or(|lower| *value > lower)
                })
                .count();
            assert_eq!(independent, *count);
        } else {
            assert!(!reported);
        }
    }
    fn verify_stored_pair(matrix: &[Float], vector: &[Float], value: &Float) {
        let n = vector.len();
        let x: Vec<_> = vector.iter().map(|v| v.to_rational().unwrap()).collect();
        let lambda = value.to_rational().unwrap();
        let mut residual_sq = Rational::new();
        let mut norm_sq = Rational::new();
        for i in 0..n {
            let mut residual = -Rational::from(&lambda * &x[i]);
            for j in 0..n {
                residual += matrix[i * n + j].to_rational().unwrap() * &x[j];
            }
            residual_sq += Rational::from(&residual * &residual);
            norm_sq += Rational::from(&x[i] * &x[i]);
        }
        let tolerance = Float::with_val(512, Float::parse("1e-28").unwrap());
        assert!(Float::with_val(512, residual_sq).sqrt() < tolerance);
        assert!(Float::with_val(512, norm_sq - 1).abs() < tolerance);
    }
    fn literal(s: &str) -> DecimalLiteral {
        DecimalLiteral::new(s).unwrap()
    }
    fn configs(requested: usize) -> (ThickRestartLanczosConfigHp, ShiftInvertKrylovConfigHp) {
        let thick = ThickRestartLanczosConfigHp {
            target: EigenTarget::AlgebraicSmallest,
            precision_bits: 256,
            requested_eigenpairs: requested,
            guard_eigenpairs: 1,
            maximum_subspace_dimension: 4,
            maximum_restarts: 400,
            minimum_restarts: 2,
            maximum_projected_sweeps: 256,
            absolute_residual_tolerance: literal("1e-40"),
            scaled_backward_error_tolerance: literal("1e-40"),
            ritz_value_stability_tolerance: literal("1e-40"),
            boundary_cluster_tolerance: literal("1e-30"),
        };
        let krylov = ShiftInvertKrylovConfigHp {
            target: EigenTarget::SmallestMagnitude,
            precision_bits: 256,
            requested_eigenpairs: requested,
            guard_eigenpairs: 1,
            maximum_subspace_dimension: 4,
            maximum_restarts: 400,
            minimum_restarts: 2,
            maximum_projected_sweeps: 256,
            absolute_residual_tolerance: literal("1e-40"),
            scaled_backward_error_tolerance: literal("1e-40"),
            ritz_value_stability_tolerance: literal("1e-40"),
            boundary_cluster_tolerance: literal("1e-30"),
        };
        (thick, krylov)
    }
    fn diagonal(values: &[i32]) -> Vec<Float> {
        let n = values.len();
        let mut a = vec![Float::with_val(256, 0); n * n];
        for (i, v) in values.iter().enumerate() {
            a[i * n + i].assign(*v);
        }
        a
    }
    #[test]
    fn scalar_krylov_checks_a_fresh_complement_before_accepting_multiplicity() {
        // Exact dyadic reflection blocks: spectrum [1,1,2,3,4,5].
        let integers = [
            7, -1, -3, 1, 0, 0, -1, 7, 1, -3, 0, 0, -3, 1, 7, -1, 0, 0, 1, -3, -1, 7, 0, 0, 0, 0,
            0, 0, 18, -2, 0, 0, 0, 0, -2, 18,
        ];
        let rotated: Vec<Float> = integers
            .iter()
            .map(|v| Float::with_val(256, *v) / 4)
            .collect();
        for a in [diagonal(&[1, 1, 2, 3, 4, 5]), rotated] {
            let op =
                DenseSymmetricHp::new("multiplicity", 6, a.clone(), 256, &Float::with_val(256, 0))
                    .unwrap();
            let factor =
                DenseShiftInvertFactorizationHp::factor("factor", 6, &a, literal("0"), 256)
                    .unwrap();
            for requested in [1, 2] {
                let (t, k) = configs(requested);
                let tr = ThickRestartLanczosHp.solve(&op, &t).unwrap();
                let kr = ShiftInvertKrylovSolverHp.solve(&op, &factor, &k).unwrap();
                let exact_spectrum: Vec<_> = [1, 1, 2, 3, 4, 5]
                    .into_iter()
                    .map(|value| Float::with_val(256, value))
                    .collect();
                verify_count_proof(
                    &tr.boundary_count_evidence,
                    tr.global_target_ordering_established,
                    &exact_spectrum,
                    requested,
                );
                verify_count_proof(
                    &kr.boundary_count_evidence,
                    kr.global_target_ordering_established,
                    &exact_spectrum,
                    requested,
                );
                if requested == 1 {
                    assert_eq!(tr.status, xc_core::ResultStatus::UnresolvedCluster);
                    assert_eq!(kr.status, xc_core::ResultStatus::UnresolvedCluster);
                } else {
                    assert_eq!(tr.status, xc_core::ResultStatus::Converged);
                    assert_eq!(kr.status, xc_core::ResultStatus::Converged);
                    for pair in tr.retained_eigenpairs.iter().take(2) {
                        verify_stored_pair(&a, &pair.eigenvector, &pair.eigenvalue);
                    }
                    for pair in kr.retained_eigenpairs.iter().take(2) {
                        verify_stored_pair(&a, &pair.eigenvector, &pair.eigenvalue);
                    }
                    for value in tr
                        .retained_eigenpairs
                        .iter()
                        .take(2)
                        .map(|x| &x.eigenvalue)
                        .chain(kr.retained_eigenpairs.iter().take(2).map(|x| &x.eigenvalue))
                    {
                        assert!(
                            (value.clone() - 1i32).abs()
                                < Float::with_val(256, Float::parse("1e-30").unwrap())
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn a_fresh_coordinate_still_does_not_prove_global_target_completeness() {
        // q=(1,1/2,1/3,...) and the first fresh coordinate e0 are both
        // orthogonal to w=(0,2,-3,0,0,0). On coordinates1,2 the
        // exact block is3I-(5/26)ww^T; other eigenvalues are1,2,4,5.
        let w = [0, 2, -3, 0, 0, 0];
        let diagonal = [1, 3, 3, 2, 4, 5];
        let a: Vec<Float> = (0..6)
            .flat_map(|i| {
                (0..6).map(move |j| {
                    let mut x = Float::with_val(
                        256,
                        (if i == j { 26 * diagonal[i] } else { 0 }) - 5 * w[i] * w[j],
                    );
                    x /= 26;
                    x
                })
            })
            .collect();
        let op =
            DenseSymmetricHp::new("hidden", 6, a.clone(), 256, &Float::with_val(256, 0)).unwrap();
        let factor =
            DenseShiftInvertFactorizationHp::factor("hidden-factor", 6, &a, literal("0"), 256)
                .unwrap();
        let (t, k) = configs(1);
        let tr = ThickRestartLanczosHp.solve(&op, &t).unwrap();
        let kr = ShiftInvertKrylovSolverHp.solve(&op, &factor, &k).unwrap();
        let exact_spectrum = ["0.5", "1", "2", "3", "4", "5"]
            .map(|value| Float::with_val(256, Float::parse(value).unwrap()));
        verify_count_proof(
            &tr.boundary_count_evidence,
            tr.global_target_ordering_established,
            &exact_spectrum,
            1,
        );
        verify_count_proof(
            &kr.boundary_count_evidence,
            kr.global_target_ordering_established,
            &exact_spectrum,
            1,
        );
        for (converged, value, vector) in [
            (
                tr.status == xc_core::ResultStatus::Converged,
                &tr.retained_eigenpairs[0].eigenvalue,
                &tr.retained_eigenpairs[0].eigenvector,
            ),
            (
                kr.status == xc_core::ResultStatus::Converged,
                &kr.retained_eigenpairs[0].eigenvalue,
                &kr.retained_eigenpairs[0].eigenvector,
            ),
        ] {
            if converged {
                assert!(
                    (value.clone() - &exact_spectrum[0]).abs()
                        < Float::with_val(256, Float::parse("1e-28").unwrap())
                );
                verify_stored_pair(&a, vector, value);
            }
        }
    }
    #[test]
    fn actual_full_space_projection_records_ordering_evidence() {
        let a = diagonal(&[1, 2, 3, 4, 5, 6]);
        let op =
            DenseSymmetricHp::new("complete", 6, a.clone(), 256, &Float::with_val(256, 0)).unwrap();
        let factor =
            DenseShiftInvertFactorizationHp::factor("complete-factor", 6, &a, literal("0"), 256)
                .unwrap();
        let (mut t, mut k) = configs(1);
        t.maximum_subspace_dimension = 6;
        k.maximum_subspace_dimension = 6;
        let tr = ThickRestartLanczosHp.solve(&op, &t).unwrap();
        let kr = ShiftInvertKrylovSolverHp.solve(&op, &factor, &k).unwrap();
        assert!(tr.global_target_ordering_established);
        assert!(kr.global_target_ordering_established);
        assert!(
            (tr.retained_eigenpairs[0].eigenvalue.clone() - 1i32).abs()
                < Float::with_val(256, 1e-30)
        );
        assert!(
            (kr.retained_eigenpairs[0].eigenvalue.clone() - 1i32).abs()
                < Float::with_val(256, 1e-30)
        );
    }

    #[test]
    fn renamed_hp_report_clone_is_agreement_not_independent_assurance() {
        let a = diagonal(&[1, 2]);
        let problem = DenseSymmetricProblemHp::new(&a, 2).unwrap();
        let mut c = config();
        c.precision = PrecisionPolicy::fixed(256);
        let original = solve_dense_reference_hp(&problem, &c).unwrap();
        let mut clone = original.clone();
        clone.algorithm = "renamed-copy".into();
        let compared = cross_check_hp_reports(
            &original,
            &clone,
            HpCrossCheckTolerance {
                eigenvalue_absolute: literal("1e-30"),
                one_minus_overlap_squared: literal("1e-30"),
            },
        )
        .unwrap();
        assert_eq!(compared.accepted.assurance, AssuranceLevel::Computed);
    }
}
