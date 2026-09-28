#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::eigen::{
    dense_symmetric_eigendecomposition_jacobi_hp, dense_symmetric_eigenvalues_jacobi_hp,
};

#[test]
fn jacobi_respects_matrix_scale_and_exact_two_by_two_eigenvalues() {
    for p in [64, 128, 256] {
        for e in [-20000_i32, -400, 0, 400, 20000] {
            let scale = Float::with_val(p, 1) << e;
            let matrix = vec![-scale.clone(), scale.clone(), scale.clone(), scale.clone()];
            let result = dense_symmetric_eigenvalues_jacobi_hp(&matrix, 2, p, 20).unwrap();
            let expected = Float::with_val(p, 2).sqrt();
            let error = Float::with_val(
                p,
                Float::with_val(p, &result.eigenvalues[1] / &scale) - expected,
            )
            .abs();
            assert!(
                error < (Float::with_val(p, 1) >> (p - 8)),
                "p={p}, e={e}, error={error}"
            );
            assert!(result.rotations > 0);
        }
    }
}

#[test]
fn jacobi_vectors_obey_aq_qd_and_orthonormality_including_repeated_values() {
    let p = 128;
    for entries in [
        [2, 1, 0, 1, 2, 1, 0, 1, 2],
        [3, 0, 0, 0, 3, 0, 0, 0, 3],
        [2, 1, 1, 1, 2, 1, 1, 1, 2],
    ] {
        let a: Vec<_> = entries.iter().map(|x| Float::with_val(p, *x)).collect();
        let result = dense_symmetric_eigendecomposition_jacobi_hp(&a, 3, p, 40).unwrap();
        let q = &result.eigenvectors;
        let tolerance = Float::with_val(p, 1) >> 100_u32;
        for i in 0..3 {
            for j in 0..3 {
                let mut aq = Float::with_val(p, 0);
                let mut gram = Float::with_val(p, 0);
                for k in 0..3 {
                    aq += Float::with_val(p, &a[i * 3 + k] * &q[k * 3 + j]);
                    gram += Float::with_val(p, &q[k * 3 + i] * &q[k * 3 + j]);
                }
                aq -= Float::with_val(p, &q[i * 3 + j] * &result.eigenvalues[j]);
                gram -= usize::from(i == j);
                assert!(aq.abs() < tolerance);
                assert!(gram.abs() < tolerance);
            }
        }
    }
}

#[test]
fn jacobi_rejects_overflowed_outputs_and_invalid_domains() {
    let p = 128;
    let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 1);
    let mut huge = huge;
    huge *= Float::with_val(p, 1.5);
    let a = vec![-huge.clone(), huge.clone(), huge.clone(), huge];
    assert!(dense_symmetric_eigenvalues_jacobi_hp(&a, 2, p, 20).is_err());
    assert!(dense_symmetric_eigenvalues_jacobi_hp(&[], usize::MAX, p, 20).is_err());
    let a = vec![Float::with_val(p, 1)];
    assert!(dense_symmetric_eigenvalues_jacobi_hp(&a, 1, 32, 20).is_err());
    assert!(dense_symmetric_eigenvalues_jacobi_hp(&a, 1, p, 0).is_err());
}
