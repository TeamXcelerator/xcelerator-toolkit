#![cfg(feature = "hp-reference")]
use rug::{ops::Pow, Float, Integer, Rational};
use xc_core::{
    DecimalLiteral, EigenTarget, PrecisionEscalation, PrecisionPolicy, ResultStatus,
    TerminationReason,
};
use xc_operator::{
    DenseSymmetricHp, GeneralizedEigenProblem, LinearOperator, OperatorError, OperatorMetadata,
    PositiveDefiniteMetric, SpectralInertia, SymmetricOperator,
};
use xc_solver::*;
fn d(s: &str) -> DecimalLiteral {
    DecimalLiteral::new(s).unwrap()
}
fn config(target: EigenTarget) -> BlockShiftInvertConfigHp {
    BlockShiftInvertConfigHp {
        target,
        precision_bits: 256,
        requested_eigenpairs: 1,
        guard_eigenpairs: 2,
        absolute_residual_tolerance: d("1e-40"),
        scaled_backward_error_tolerance: d("1e-40"),
        ritz_value_stability_tolerance: d("1e-40"),
        boundary_cluster_tolerance: d("1e-35"),
        maximum_iterations: 400,
        minimum_iterations: 2,
        maximum_projected_sweeps: 100,
    }
}
fn dense(entries: &[i64], n: usize) -> DenseSymmetricHp {
    DenseSymmetricHp::new(
        "exact-integer-oracle",
        n,
        entries.iter().map(|x| Float::with_val(320, *x)).collect(),
        320,
        &Float::with_val(320, 0),
    )
    .unwrap()
}
fn hadamard_matrix(n: usize) -> Vec<i64> {
    let spectrum = [1, 3, 4, 6, 8, 10, 12, 15];
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n)
                .map(|k| {
                    let sign = if ((i & k).count_ones() + (j & k).count_ones()) % 2 == 0 {
                        1
                    } else {
                        -1
                    };
                    sign * spectrum[k]
                })
                .sum()
        })
        .collect()
}
#[test]
fn near_shift_keeps_independent_guard_directions_and_physical_residuals() {
    for n in [4, 8] {
        let entries = hadamard_matrix(n);
        let a = dense(&entries, n);
        for suffix in [
            "000000000000000000000000000001",
            "000000000000000000000000000000000000000000000000000000000001",
        ] {
            let shift = d(&format!("{}.{}", 3 * n, suffix));
            let cfg = config(EigenTarget::ClosestTo {
                shift: shift.clone(),
            });
            let matrix: Vec<_> = entries.iter().map(|x| Float::with_val(320, *x)).collect();
            let lu = DenseShiftInvertFactorizationHp::factor("near-shift", n, &matrix, shift, 256)
                .unwrap();
            let report = BlockShiftInvertSolverHp.solve(&a, &lu, &cfg).unwrap();
            assert_eq!(
                report.status,
                ResultStatus::Converged,
                "n={n}, {}",
                report.iterations
            );
            assert_eq!(report.factorizations, 0);
            assert!(
                Float::with_val(256, &report.retained_eigenpairs[0].eigenvalue - 3 * n).abs()
                    < Float::with_val(256, 1) >> 100
            );
            for pair in &report.retained_eigenpairs {
                let x: Vec<_> = pair
                    .eigenvector
                    .iter()
                    .map(|v| v.to_rational().unwrap())
                    .collect();
                let lambda = pair.eigenvalue.to_rational().unwrap();
                let mut residual_sq = Rational::new();
                let mut image_sq = Rational::new();
                for i in 0..n {
                    let mut ax = Rational::new();
                    for j in 0..n {
                        ax += Rational::from(&x[j] * entries[i * n + j]);
                    }
                    let r = ax.clone() - Rational::from(&lambda * &x[i]);
                    residual_sq += Rational::from(&r * &r);
                    image_sq += Rational::from(&ax * &ax);
                }
                assert!(
                    residual_sq <= image_sq / Rational::from(Integer::from(10).pow(76)),
                    "independent exact source residual"
                );
            }
        }
    }
}
struct Diagonal {
    entries: Vec<Float>,
    counts: bool,
}
impl LinearOperator<Float> for Diagonal {
    fn dimension(&self) -> usize {
        self.entries.len()
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        for (i, v) in y.iter_mut().enumerate() {
            *v = Float::with_val(320, &self.entries[i] * &x[i]);
        }
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        OperatorMetadata::new(
            "analytic-diagonal",
            self.entries.len(),
            xc_operator::MatrixStructure::Diagonal,
            "rug_mpfr",
        )
    }
}
impl SymmetricOperator<Float> for Diagonal {
    fn spectral_inertia_at(&self, shift: &Float) -> Result<Option<SpectralInertia>, OperatorError> {
        Ok(self.counts.then(|| SpectralInertia {
            below: self.entries.iter().filter(|x| *x < shift).count(),
            equal: self.entries.iter().filter(|x| *x == shift).count(),
            above: self.entries.iter().filter(|x| *x > shift).count(),
        }))
    }
}
impl PositiveDefiniteMetric<Float> for Diagonal {}
fn diag(values: &[i32], counts: bool) -> Diagonal {
    Diagonal {
        entries: values.iter().map(|x| Float::with_val(320, *x)).collect(),
        counts,
    }
}
#[test]
fn interval_capacity_and_closed_endpoints_use_source_counts() {
    let a = diag(&[-1, 0, 2, 5], true);
    let mut matrix = vec![Float::with_val(320, 0); 16];
    for i in 0..4 {
        matrix[5 * i] = a.entries[i].clone();
    }
    for (lower, upper, shift, count) in [
        ("-0.5", "1", "0.25", 1),
        ("0.3", "0.4", "0.35", 0),
        ("0", "2", "1", 2),
    ] {
        let lu =
            DenseShiftInvertFactorizationHp::factor("interval", 4, &matrix, d(shift), 256).unwrap();
        let mut cfg = config(EigenTarget::Interval {
            lower: d(lower),
            upper: d(upper),
        });
        cfg.requested_eigenpairs = 3;
        cfg.guard_eigenpairs = 1;
        let report = BlockShiftInvertSolverHp.solve(&a, &lu, &cfg).unwrap();
        assert_eq!(
            report.status,
            ResultStatus::Converged,
            "{lower},{upper}: {:?}",
            report.interval_count_evidence
        );
        assert_eq!(report.selected_eigenpairs, count);
        assert_eq!(report.requested_eigenpairs, 3);
        assert!(report
            .interval_count_evidence
            .as_ref()
            .unwrap()
            .establishes_requested_count());
        if count == 0 {
            assert_eq!(report.termination, TerminationReason::EmptySelection);
        }
    }
    let unknown = diag(&[-1, 0, 2, 5], false);
    let lu =
        DenseShiftInvertFactorizationHp::factor("unknown", 4, &matrix, d("0.25"), 256).unwrap();
    let mut cfg = config(EigenTarget::Interval {
        lower: d("-0.5"),
        upper: d("1"),
    });
    cfg.requested_eigenpairs = 3;
    cfg.guard_eigenpairs = 1;
    let report = BlockShiftInvertSolverHp.solve(&unknown, &lu, &cfg).unwrap();
    assert_eq!(report.status, ResultStatus::UnresolvedEigenspace);
}
fn policy() -> PrecisionPolicy {
    PrecisionPolicy {
        initial_bits: 64,
        maximum_bits: 256,
        guard_bits: 0,
        escalation: PrecisionEscalation::AddBits(192),
    }
}
#[test]
fn failed_metric_and_singular_factor_do_not_claim_insufficient_precision() {
    let a = diag(&[2, 3, 5], true);
    for values in [[-1, -1, -1], [1, -100, 1]] {
        let b = diag(&values, false);
        let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
        let options = AdaptiveGeneralizedExtremeOptionsHp {
            target: EigenTarget::AlgebraicSmallest,
            absolute_residual_tolerance: d("1e-30"),
            scaled_backward_error_tolerance: d("1e-30"),
            ritz_value_stability_tolerance: d("1e-30"),
            maximum_iterations: 10,
            minimum_iterations: 2,
            precision: policy(),
        };
        let AdaptiveGeneralizedExtremeResultHp::Inconclusive {
            attempts, reason, ..
        } = solve_matrix_free_generalized_adaptive_hp(&problem, &options).unwrap()
        else {
            panic!("invalid metric accepted")
        };
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].status, ResultStatus::Failed);
        assert!(!reason.contains("maximum precision"));
    }
    let mut matrix = vec![Float::with_val(320, 0); 9];
    for i in 0..3 {
        matrix[4 * i] = a.entries[i].clone();
    }
    let factory =
        DenseShiftInvertFactoryHp::new("exact-singular", 3, &matrix, d("3"), 320).unwrap();
    let cfg = AdaptiveBlockShiftInvertOptionsHp {
        target: EigenTarget::ClosestTo { shift: d("3") },
        requested_eigenpairs: 1,
        guard_eigenpairs: 1,
        absolute_residual_tolerance: d("1e-30"),
        scaled_backward_error_tolerance: d("1e-30"),
        ritz_value_stability_tolerance: d("1e-30"),
        boundary_cluster_tolerance: d("1e-20"),
        maximum_iterations: 10,
        minimum_iterations: 2,
        maximum_projected_sweeps: 50,
        precision: policy(),
    };
    let AdaptiveBlockShiftInvertResultHp::Inconclusive { attempts, .. } =
        solve_block_shift_invert_adaptive_hp(&a, &factory, &cfg).unwrap()
    else {
        panic!("singular solve accepted")
    };
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, ResultStatus::Failed);
    assert_eq!(attempts[0].factorizations, 0);
}
#[test]
fn scale_evidence_reports_the_preserved_absolute_or_backward_criteria() {
    for scale in ["1e-120", "1e120"] {
        let scalar = Float::with_val(320, Float::parse(scale).unwrap());
        let a = Diagonal {
            entries: vec![scalar.clone(), Float::with_val(320, &scalar * 2)],
            counts: false,
        };
        let b = diag(&[1, 1], false);
        let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
        let cfg = GeneralizedExtremeConfigHp {
            target: EigenTarget::AlgebraicLargest,
            precision_bits: 256,
            absolute_residual_tolerance: d("1e-35"),
            scaled_backward_error_tolerance: d("1e-45"),
            ritz_value_stability_tolerance: d("1e-40"),
            maximum_iterations: 1,
            minimum_iterations: 1,
        };
        let report = MatrixFreeGeneralizedRayleighRitzHp
            .solve(&problem, &cfg)
            .unwrap();
        let evidence = &report.stopping_evidence;
        assert_eq!(
            evidence.absolute_residual_passed,
            report.residual_norm <= evidence.absolute_residual_tolerance
        );
        assert_eq!(
            evidence.scaled_backward_error_passed,
            report.scaled_backward_error <= evidence.scaled_backward_error_tolerance
        );
        if scale == "1e-120" {
            assert!(evidence.absolute_residual_passed);
            assert!(!evidence.scaled_backward_error_passed);
        }
        let x: Vec<_> = report
            .eigenvector
            .iter()
            .map(|v| Float::with_val(512, v))
            .collect();
        let mut norm_ax = Float::with_val(512, 0);
        let mut norm_x = Float::with_val(512, 0);
        let mut residual = Float::with_val(512, 0);
        #[allow(clippy::needless_range_loop)]
        for i in 0..2 {
            let ax = Float::with_val(512, &a.entries[i] * &x[i]);
            norm_ax += Float::with_val(512, &ax * &ax);
            norm_x += Float::with_val(512, &x[i] * &x[i]);
            let r = ax - Float::with_val(512, &report.eigenvalue * &x[i]);
            residual += Float::with_val(512, &r * &r);
        }
        let denominator =
            norm_ax.sqrt() + Float::with_val(512, report.eigenvalue.clone().abs()) * norm_x.sqrt();
        let difference = (Float::with_val(512, &evidence.image_norm_denominator) - &denominator)
            .abs()
            / &denominator;
        assert!(difference < Float::with_val(512, 1) >> 240);
        let independent = residual.sqrt() / denominator;
        assert!(
            (Float::with_val(512, &report.scaled_backward_error) - independent).abs()
                < Float::with_val(512, 1) >> 240
        );
    }
}
