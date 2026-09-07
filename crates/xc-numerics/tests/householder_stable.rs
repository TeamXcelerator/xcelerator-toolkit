#![cfg(feature = "hp")]
use rug::{ops::Pow, Float};
use xc_numerics::eigen::*;
fn f(p: u32, x: i32) -> Float {
    Float::with_val(p, x)
}
fn tolerance(p: u32) -> Float {
    f(p, 2).pow(-(p as i32 - 8))
}
fn matrix(p: u32) -> Vec<Float> {
    [
        4, 1, 2, -1, 0, 1, 5, -1, 2, 1, 2, -1, 6, 0, -2, -1, 2, 0, 7, 1, 0, 1, -2, 1, 8,
    ]
    .iter()
    .map(|&x| f(p, x))
    .collect()
}
#[test]
fn near_tridiagonal_cancellation_is_detected_and_corrected() {
    for (p, exponent) in [(64_u32, 40_i32), (128, 80), (256, 150)] {
        for sign in [-1, 1] {
            let b = f(p, 2).pow(-exponent);
            let a = vec![
                f(p, 0),
                f(p, sign),
                b.clone(),
                f(p, sign),
                f(p, 0),
                f(p, 0),
                b.clone(),
                f(p, 0),
                f(p, 2),
            ];
            let (d, e, q) = householder_tridiag_hp(&a, 3, p).unwrap();
            let old = assess_symmetric_reduction_hp(&a, &d, &e, &q, p).unwrap();
            assert!(old.absolute_similarity_residual > b / 2);
            assert!(old.absolute_orthogonality_residual < tolerance(p));
            let (d, e, q) = householder_tridiag_hp_stable(&a, 3, p).unwrap();
            let corrected = assess_symmetric_reduction_hp(&a, &d, &e, &q, p).unwrap();
            assert!(corrected.relative_similarity_residual < tolerance(p));
            assert!(corrected.relative_orthogonality_residual < tolerance(p));
        }
    }
}
#[test]
fn stable_values_agree_with_independent_jacobi() {
    let p = 256;
    let a = matrix(p);
    let actual = dense_symmetric_eigenvalues_hp_stable(&a, 5, p).unwrap();
    let reference = dense_symmetric_eigenvalues_jacobi_hp(&a, 5, p, 100)
        .unwrap()
        .eigenvalues;
    for (a, b) in actual.iter().zip(reference) {
        assert!(Float::with_val(p, a - b).abs() < f(p, 2).pow(-200));
    }
}
#[test]
fn omitted_q_preserves_t_and_thread_counts_preserve_output() {
    let p = 128;
    let a = matrix(p);
    let mut outputs = Vec::new();
    let mut diagnostics = Vec::new();
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = pool.install(|| householder_tridiag_hp_stable(&a, 5, p).unwrap());
        let (d, e) = pool.install(|| dense_symmetric_tridiagonal_hp_stable(&a, 5, p).unwrap());
        assert_eq!((&result.0, &result.1), (&d, &e));
        diagnostics.push(pool.install(|| {
            assess_symmetric_reduction_hp(&a, &result.0, &result.1, &result.2, p).unwrap()
        }));
        outputs.push(result);
    }
    assert!(outputs.windows(2).all(|x| x[0] == x[1]));
    assert!(diagnostics.windows(2).all(|x| x[0] == x[1]));
}
#[test]
fn recovered_vectors_satisfy_the_original_matrix_equation() {
    let p = 256;
    let n = 5;
    let a = matrix(p);
    let (d, e, q) = householder_tridiag_hp_stable(&a, n, p).unwrap();
    let values = tridiag_eigenvalues_hp(&d, &e, p).unwrap();
    for lambda in values {
        let v = tridiag_eigenvector_for_value_hp(
            &d,
            &e,
            &lambda,
            p,
            TridiagEigvecOptions {
                solver: TridiagSolver::BandedInterleaved,
                max_steps: 30,
                early_termination: false,
            },
        )
        .unwrap();
        let original = (0..n)
            .map(|i| {
                let mut sum = f(p, 0);
                for j in 0..n {
                    sum += Float::with_val(p, &q[i * n + j] * &v[j]);
                }
                sum
            })
            .collect::<Vec<_>>();
        for (i, x) in original.iter().enumerate() {
            let mut residual = Float::with_val(p, -Float::with_val(p, &lambda * x));
            for j in 0..n {
                residual += Float::with_val(p, &a[i * n + j] * &original[j]);
            }
            assert!(residual.abs() < f(p, 2).pow(-180));
        }
    }
}
#[test]
fn scaled_inputs_preserve_relative_checks_and_eigenvalue_scaling() {
    let p = 192;
    let a = matrix(p);
    let reference = dense_symmetric_eigenvalues_hp_stable(&a, 5, p).unwrap();
    for exponent in [-4000, 4000] {
        let scale = f(p, 2).pow(exponent);
        let scaled = a
            .iter()
            .map(|x| Float::with_val(p, x * &scale))
            .collect::<Vec<_>>();
        let (d, e, q) = householder_tridiag_hp_stable(&scaled, 5, p).unwrap();
        let checks = assess_symmetric_reduction_hp(&scaled, &d, &e, &q, p).unwrap();
        assert!(checks.relative_similarity_residual < tolerance(p));
        let values = tridiag_eigenvalues_hp(&d, &e, p).unwrap();
        for (value, expected) in values.iter().zip(&reference) {
            let unscaled = Float::with_val(p, value / &scale);
            assert!(Float::with_val(p, unscaled - expected).abs() < f(p, 2).pow(-140));
        }
    }
}
#[test]
fn diagonal_singleton_and_invalid_input_contracts() {
    let p = 128;
    let (d, e, q) = householder_tridiag_hp_stable(&[f(p, 7)], 1, p).unwrap();
    assert_eq!(d, vec![f(p, 7)]);
    assert!(e.is_empty());
    assert_eq!(q, vec![f(p, 1)]);
    assert!(householder_tridiag_hp_stable(&[], 0, p).is_err());
    assert!(householder_tridiag_hp(&[], 0, p).is_err());
    assert!(householder_tridiag_hp_stable(&[], usize::MAX, p).is_err());
    assert!(householder_tridiag_hp_stable(&[f(p, 1)], 1, 64).is_err());
    assert!(householder_tridiag_hp_stable(&[f(p, 1)], 1, 32).is_err());
    let mut a = matrix(p);
    a[1] += 1;
    assert!(householder_tridiag_hp_stable(&a, 5, p).is_err());
    a = matrix(p);
    a[0] = Float::with_val(p, rug::float::Special::Nan);
    assert!(householder_tridiag_hp_stable(&a, 5, p).is_err());
    let a = matrix(p);
    let (d, e, mut q) = householder_tridiag_hp_stable(&a, 5, p).unwrap();
    q.pop();
    assert!(assess_symmetric_reduction_hp(&a, &d, &e, &q, p).is_err());
}
#[test]
fn legacy_tridiagonal_source_retains_its_exact_identity_basis() {
    let p = 128;
    let a = vec![
        f(p, 2),
        f(p, 1),
        f(p, 0),
        f(p, 1),
        f(p, 3),
        f(p, -2),
        f(p, 0),
        f(p, -2),
        f(p, 5),
    ];
    let (d, e, q) = householder_tridiag_hp(&a, 3, p).unwrap();
    assert_eq!(d, vec![f(p, 2), f(p, 3), f(p, 5)]);
    assert_eq!(e, vec![f(p, 1), f(p, -2)]);
    assert_eq!(
        q,
        (0..9)
            .map(|i| f(p, i32::from(i % 4 == 0)))
            .collect::<Vec<_>>()
    );
}
