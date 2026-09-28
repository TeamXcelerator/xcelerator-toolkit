#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::eigen::{
    dense_symmetric_eigenvector_for_value_hp, tridiag_eigenvector_for_value_hp,
    TridiagEigvecOptions,
};

/// [[0, g/2, 0], [g/2, g, 0], [0, 0, 1]] has eigenvalues g(1 +/- sqrt2)/2 with
/// vectors along (1, 1 +/- sqrt2, 0). With g = 2^-(p-32), a shift 2^-(p-32)
/// below the upper value lies nearer the lower one, and the lower vector was
/// returned for the upper value with an accepted residual.
#[test]
fn recovered_eigenvectors_belong_to_the_requested_eigenvalue() {
    for p in [96, 128, 256] {
        let g = Float::with_val(p, 1) >> (p - 32);
        let half = Float::with_val(p, &g / 2u32);
        let zero = Float::with_val(p, 0);
        let root_two = Float::with_val(4 * p, 2u32).sqrt();
        for sign in [1i32, -1] {
            let factor = Float::with_val(4 * p, 1) + Float::with_val(4 * p, &root_two * sign);
            let value = Float::with_val(p, Float::with_val(4 * p, &factor * &g) / 2u32);
            let aligned = |v: &[Float]| {
                let norm = Float::with_val(4 * p, 1) + Float::with_val(4 * p, &factor * &factor);
                let dot = Float::with_val(4 * p, &v[0]) + Float::with_val(4 * p, &factor * &v[1]);
                let length = v.iter().fold(Float::with_val(4 * p, 0), |sum, x| {
                    sum + Float::with_val(4 * p, x * x)
                });
                let cosine = dot.abs() / (norm * length).sqrt();
                assert!(
                    Float::with_val(4 * p, 1) - cosine < Float::with_val(4 * p, 1) >> 60u32,
                    "p={p}, sign={sign}"
                );
            };
            let tridiagonal = tridiag_eigenvector_for_value_hp(
                &[zero.clone(), g.clone(), Float::with_val(p, 1)],
                &[half.clone(), zero.clone()],
                &value,
                p,
                TridiagEigvecOptions::default(),
            )
            .unwrap();
            aligned(&tridiagonal);
            let matrix = [
                zero.clone(),
                half.clone(),
                zero.clone(),
                half.clone(),
                g.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                Float::with_val(p, 1),
            ];
            let dense =
                dense_symmetric_eigenvector_for_value_hp(&matrix, 3, &value, p, 100).unwrap();
            aligned(&dense);
        }
    }
}

#[test]
fn retained_solver_identities_require_pairing_checks() {
    use xc_numerics::eigen::TridiagSolver;
    assert_eq!(
        TridiagSolver::Banded.semantics_id(),
        "tridiag-interleaved-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v10"
    );
    assert_eq!(
        TridiagSolver::BandedInterleaved.semantics_id(),
        TridiagSolver::Banded.semantics_id()
    );
    assert_eq!(
        TridiagSolver::Dense.semantics_id(),
        "tridiag-dense-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v10"
    );
}

#[test]
fn rounded_selected_values_carry_their_input_uncertainty() {
    use xc_numerics::eigen::tridiag_eigenvector_for_value_with_uncertainty_hp;
    let p = 128;
    let reference = (Float::with_val(512, 5) - Float::with_val(512, 5).sqrt()) / 2;
    let requested = Float::with_val(p, Float::with_val(64, &reference));
    let uncertainty = Float::with_val(p, 1e-18);
    let diag = [Float::with_val(p, 2), Float::with_val(p, 3)];
    let off = [Float::with_val(p, 1)];
    assert!(tridiag_eigenvector_for_value_hp(
        &diag,
        &off,
        &requested,
        p,
        TridiagEigvecOptions::default()
    )
    .is_err());
    let vector = tridiag_eigenvector_for_value_with_uncertainty_hp(
        &diag,
        &off,
        &requested,
        &uncertainty,
        p,
        TridiagEigvecOptions::default(),
    )
    .unwrap();
    let slope = Float::with_val(512, &vector[1] / &vector[0]);
    let expected = Float::with_val(512, &reference) - 2u32;
    assert!((slope - expected).abs() < Float::with_val(512, 1e-30));
    for invalid in [
        Float::with_val(p, -1),
        Float::with_val(p, f64::NAN),
        Float::with_val(p, f64::INFINITY),
    ] {
        assert!(tridiag_eigenvector_for_value_with_uncertainty_hp(
            &diag,
            &off,
            &requested,
            &invalid,
            p,
            TridiagEigvecOptions::default(),
        )
        .is_err());
    }
}
