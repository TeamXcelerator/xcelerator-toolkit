#![cfg(feature = "hp-reference")]
use rug::{Float, Rational};
use xc_core::{DecimalLiteral, EigenTarget, ResultStatus};
use xc_operator::{
    DenseSymmetricHp, LinearOperator, OperatorError, OperatorMetadata, SpectralInertia,
    SymmetricOperator,
};
use xc_solver::*;
const P: u32 = 192;
fn lit(s: &str) -> DecimalLiteral {
    DecimalLiteral::new(s).unwrap()
}
fn thick(requested: usize, m: usize) -> ThickRestartLanczosConfigHp {
    ThickRestartLanczosConfigHp {
        target: EigenTarget::AlgebraicSmallest,
        precision_bits: P,
        requested_eigenpairs: requested,
        guard_eigenpairs: 1,
        maximum_subspace_dimension: m,
        maximum_restarts: 180,
        minimum_restarts: 2,
        maximum_projected_sweeps: 128,
        absolute_residual_tolerance: lit("1e-35"),
        scaled_backward_error_tolerance: lit("1e-35"),
        ritz_value_stability_tolerance: lit("1e-30"),
        boundary_cluster_tolerance: lit("1e-20"),
    }
}
fn shifted(requested: usize, m: usize) -> ShiftInvertKrylovConfigHp {
    ShiftInvertKrylovConfigHp {
        target: EigenTarget::SmallestMagnitude,
        precision_bits: P,
        requested_eigenpairs: requested,
        guard_eigenpairs: 1,
        maximum_subspace_dimension: m,
        maximum_restarts: 180,
        minimum_restarts: 2,
        maximum_projected_sweeps: 128,
        absolute_residual_tolerance: lit("1e-35"),
        scaled_backward_error_tolerance: lit("1e-35"),
        ritz_value_stability_tolerance: lit("1e-30"),
        boundary_cluster_tolerance: lit("1e-20"),
    }
}
fn dense(matrix: &[i32], n: usize) -> DenseSymmetricHp {
    DenseSymmetricHp::new(
        "exact integer multiplicity fixture",
        n,
        matrix.iter().map(|v| Float::with_val(P, *v)).collect(),
        P,
        &Float::with_val(P, 0),
    )
    .unwrap()
}
// Independent exact-rational LDL count; these fixtures have no zero leading pivot.
fn exact_count(matrix: &[i32], n: usize, shift: Rational) -> usize {
    let mut a: Vec<Rational> = matrix.iter().map(|v| Rational::from(*v)).collect();
    for i in 0..n {
        a[i * n + i] -= &shift;
    }
    let mut negative = 0;
    for k in 0..n {
        let pivot = a[k * n + k].clone();
        assert_ne!(pivot, 0);
        negative += usize::from(pivot < 0);
        for i in k + 1..n {
            for j in i..n {
                let correction = Rational::from(&a[i * n + k] * &a[j * n + k]) / &pivot;
                a[i * n + j] -= &correction;
                a[j * n + i] = a[i * n + j].clone();
            }
        }
    }
    negative
}
fn hadamard_repeated() -> Vec<i32> {
    let spectrum = [1, 1, 2, 3, 5, 7, 9, 11];
    // H H^T=8I, so H diag(spectrum) H^T has exact eigenvalues 8*spectrum.
    (0usize..8)
        .flat_map(|i| {
            (0usize..8).map(move |j| {
                (0usize..8)
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
        })
        .collect()
}
fn grid() -> Vec<i32> {
    let v = [3, 1, 4, 1, 5, 9];
    let mut a = vec![0; 36 * 36];
    for i in 0..6 {
        for j in 0..6 {
            let k = 6 * i + j;
            a[k * 36 + k] = 4 + v[i] + v[j];
            for (r, c) in [
                (i.wrapping_sub(1), j),
                (i + 1, j),
                (i, j.wrapping_sub(1)),
                (i, j + 1),
            ] {
                if r < 6 && c < 6 {
                    a[k * 36 + 6 * r + c] = -1;
                }
            }
        }
    }
    a
}
#[test]
fn multiplicity_every_admitted_small_subspace_has_a_source_count_gate() {
    let rotated = hadamard_repeated();
    assert_eq!(exact_count(&rotated, 8, Rational::from(9)), 2);
    let natural = grid();
    assert_eq!(exact_count(&natural, 36, Rational::from((47388, 10000))), 1);
    assert_eq!(exact_count(&natural, 36, Rational::from((49877, 10000))), 3);
    for (matrix, n, cut) in [(rotated, 8, 1), (natural, 36, 2)] {
        let op = dense(&matrix, n);
        let factor = DenseShiftInvertFactorizationHp::factor(
            "counted zero shift",
            n,
            op.data(),
            lit("0"),
            P,
        )
        .unwrap();
        for m in 3..=7 {
            let requested = cut;
            if m <= requested + 1 {
                assert!(matches!(
                    ThickRestartLanczosHp.solve(&op, &thick(requested, m)),
                    Err(SolverError::InvalidConfiguration(_))
                ));
                assert!(matches!(
                    ShiftInvertKrylovSolverHp.solve(&op, &factor, &shifted(requested, m)),
                    Err(SolverError::InvalidConfiguration(_))
                ));
                continue;
            }
            let a = ThickRestartLanczosHp
                .solve(&op, &thick(requested, m))
                .unwrap();
            let b = ShiftInvertKrylovSolverHp
                .solve(&op, &factor, &shifted(requested, m))
                .unwrap();
            eprintln!(
                "n={n} requested={requested} m={m}: thick {:?} {:?}, shifted {:?} {:?}",
                a.status, a.boundary_count_evidence, b.status, b.boundary_count_evidence
            );
            assert_ne!(
                a.status,
                ResultStatus::Converged,
                "a requested boundary cuts an exact multiplicity"
            );
            assert_ne!(
                b.status,
                ResultStatus::Converged,
                "a requested boundary cuts an exact multiplicity"
            );
            assert!(!a.retained_eigenpairs.is_empty() || a.boundary_cluster.is_some());
            assert!(!b.retained_eigenpairs.is_empty());
        }
    }
}
struct Analytic {
    matrix: DenseSymmetricHp,
    diagonal: Vec<i32>,
    mode: u8,
}
impl LinearOperator<Float> for Analytic {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        self.matrix.apply(x, y)
    }
    fn metadata(&self) -> OperatorMetadata {
        self.matrix.metadata()
    }
}
impl SymmetricOperator<Float> for Analytic {
    fn spectral_inertia_at(&self, shift: &Float) -> Result<Option<SpectralInertia>, OperatorError> {
        if self.mode == 0 {
            return Ok(None);
        }
        let mut count = SpectralInertia {
            below: 0,
            equal: 0,
            above: 0,
        };
        for x in &self.diagonal {
            match Float::with_val(P, *x).partial_cmp(shift).unwrap() {
                std::cmp::Ordering::Less => count.below += 1,
                std::cmp::Ordering::Equal => count.equal += 1,
                std::cmp::Ordering::Greater => count.above += 1,
            }
        }
        if self.mode == 2 {
            count.above += 1;
        }
        Ok(Some(count))
    }
}
#[test]
fn count_capability_accepts_analytic_proof_and_rejects_unavailable_or_bad_totals() {
    for mode in 0..=2 {
        let values = vec![1, 3, 5, 7];
        let mut matrix = vec![0; 16];
        for (i, v) in values.iter().enumerate() {
            matrix[i * 4 + i] = *v
        }
        let op = Analytic {
            matrix: dense(&matrix, 4),
            diagonal: values,
            mode,
        };
        let result = ThickRestartLanczosHp.solve(&op, &thick(1, 3));
        if mode == 2 {
            assert!(matches!(result, Err(SolverError::NumericalBreakdown(_))));
            continue;
        }
        let report = result.unwrap();
        if mode == 0 {
            assert_eq!(report.status, ResultStatus::UnresolvedEigenspace);
            assert!(!report.global_target_ordering_established);
        } else {
            assert_eq!(report.status, ResultStatus::Converged);
            assert!(report.global_target_ordering_established);
            assert!(
                (report.retained_eigenpairs[0].eigenvalue.clone() - 1u32).abs()
                    < Float::with_val(P, 1e-25)
            );
        }
    }
}

fn interior_config(m: usize, shift: &str) -> ShiftInvertKrylovConfigHp {
    ShiftInvertKrylovConfigHp {
        target: EigenTarget::ClosestTo { shift: lit(shift) },
        maximum_restarts: 400,
        ..shifted(1, m)
    }
}
#[test]
fn interior_inverse_projection_recovers_requested_and_guard_at_every_small_subspace() {
    let spectrum = [1i32, 3, 4, 5, 6, 7, 8, 9];
    let h: Vec<i32> = (0usize..8)
        .flat_map(|i| (0usize..8).map(move |j| if (i & j).count_ones() % 2 == 0 { 1 } else { -1 }))
        .collect();
    let n: i32 = (1..=8).map(|x| x * x).sum();
    let reflector: Vec<i32> = (0..8)
        .flat_map(|i| {
            (0..8).map(move |j| {
                if i == j {
                    n - 2 * (i + 1) * (j + 1)
                } else {
                    -2 * (i + 1) * (j + 1)
                }
            })
        })
        .collect();
    for (basis, scale) in [(h, 8), (reflector, n * n)] {
        for i in 0..8 {
            for j in 0..8 {
                let dot: i32 = (0..8).map(|k| basis[i * 8 + k] * basis[j * 8 + k]).sum();
                assert_eq!(dot, if i == j { scale } else { 0 });
            }
        }
        let matrix: Vec<i32> = (0..8)
            .flat_map(|i| {
                let b = &basis;
                (0..8).map(move |j| {
                    (0..8)
                        .map(|k| b[i * 8 + k] * spectrum[k] * b[j * 8 + k])
                        .sum()
                })
            })
            .collect();
        let op = dense(&matrix, 8);
        let shift = (3 * scale + 1).to_string();
        let factor = DenseShiftInvertFactorizationHp::factor(
            "manufactured interior shift",
            8,
            op.data(),
            lit(&shift),
            P,
        )
        .unwrap();
        for m in 3..=7 {
            let r = ShiftInvertKrylovSolverHp
                .solve(&op, &factor, &interior_config(m, &shift))
                .unwrap();
            eprintln!(
                "interior scale={scale} m={m} status={:?} restarts={}",
                r.status, r.restarts
            );
            assert_eq!(r.status, ResultStatus::Converged);
            for (pair, expected) in r.retained_eigenpairs.iter().zip([3 * scale, 4 * scale]) {
                assert!((pair.eigenvalue.clone() - expected).abs() < Float::with_val(P, 1e-20));
            }
            assert!(r.boundary_count_evidence.establishes_requested_count());
        }
    }
}
