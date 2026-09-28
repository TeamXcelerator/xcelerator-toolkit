#![cfg(feature = "hp")]
use rug::Rational;
use xc_operator::{
    DenseSymmetricF64, DiagonalF64, LinearOperator, PackedSymmetricF64, RankOneUpdateF64,
    SymmetricBandedF64, TridiagonalF64,
};

#[test]
fn binary64_bounds_dominate_exact_stored_matrix_row_sums() {
    let tiny = 2_f64.powi(-54);
    let entries = vec![1., tiny, tiny, tiny, 1., tiny, tiny, tiny, 1.];
    let dense = DenseSymmetricF64::new("dyadic", 3, entries.clone(), 0.).unwrap();
    let packed = PackedSymmetricF64::new("dyadic", 3, vec![1., tiny, 1., tiny, tiny, 1.]).unwrap();
    let banded =
        SymmetricBandedF64::new("dyadic", vec![vec![1.; 3], vec![tiny; 2], vec![tiny]]).unwrap();
    // Every row sums to 1+2^-53; the all-ones vector proves that this
    // exact row sum is also an eigenvalue and the exact spectral norm.
    let exact = Rational::from_f64(1.).unwrap() + Rational::from_f64(tiny).unwrap() * 2;
    for bound in [dense.norm_bound(), packed.norm_bound(), banded.norm_bound()] {
        assert!(Rational::from_f64(bound.unwrap()).unwrap() >= exact);
    }
    let tri = TridiagonalF64::new("tri", vec![1.; 3], vec![tiny; 2]).unwrap();
    assert!(Rational::from_f64(tri.norm_bound().unwrap()).unwrap() >= exact);
    // Coordinate-vector action reads one exact column without cancellation.
    for operator in [&dense as &dyn LinearOperator<f64>, &packed, &banded] {
        let mut actual = [0.; 3];
        operator.apply(&[0., 1., 0.], &mut actual).unwrap();
        assert_eq!(actual, [tiny, 1., tiny]);
    }
}

#[test]
fn rank_one_bound_encloses_exact_nonzero_subnormal_products() {
    let zero = DiagonalF64::new("zero", vec![0.; 3]).unwrap();
    for vector in [
        vec![1., 2_f64.powi(-30), 2_f64.powi(-500)],
        vec![2_f64.powi(-537); 3],
    ] {
        let norm_sq = vector
            .iter()
            .map(|x| {
                let x = Rational::from_f64(*x).unwrap();
                Rational::from(&x * &x)
            })
            .fold(Rational::new(), |a, b| a + b);
        let updated = RankOneUpdateF64::new(&zero, 1., vector).unwrap();
        assert!(Rational::from_f64(updated.norm_bound().unwrap()).unwrap() >= norm_sq);
    }
}
