#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::eigen::*;

fn star(p: u32, shift: i32) -> Vec<Float> {
    [0, 1, 1, 1, 0, 0, 1, 0, 0]
        .iter()
        .map(|&x| {
            let mut v = Float::with_val(p, x);
            v <<= shift;
            v
        })
        .collect()
}

#[test]
fn public_default_preserves_couplings_when_raw_squares_underflow() {
    let p = 128;
    let a = star(p, -600_000_000);
    let (d, e, q) = householder_tridiag_hp(&a, 3, p).unwrap();
    // Check the original equation after exact binary rescaling. The historical
    // algorithm returned Q=I and dropped A[0,2], giving a residual of exactly1.
    let scaled: Vec<_> = a.iter().map(|x| x.clone() << 600_000_000).collect();
    let diagonal: Vec<_> = d.iter().map(|x| x.clone() << 600_000_000).collect();
    let off: Vec<_> = e.iter().map(|x| x.clone() << 600_000_000).collect();
    let residual = assess_symmetric_reduction_hp(&scaled, &diagonal, &off, &q, p).unwrap();
    assert!(residual.absolute_similarity_residual < Float::with_val(p, 1) >> 100);
    assert!(residual.absolute_orthogonality_residual < Float::with_val(p, 1) >> 100);
    assert_eq!((d, e, q), householder_tridiag_hp_stable(&a, 3, p).unwrap());
}

#[test]
fn default_spectra_match_analytic_star_under_global_scaling() {
    let p = 128;
    let root = Float::with_val(p, 2).sqrt();
    for shift in [-500_000_000, 0, 500_000_000] {
        let values = dense_symmetric_eigenvalues_hp(&star(p, shift), 3, p).unwrap();
        for (value, expected) in
            values
                .into_iter()
                .zip([-root.clone(), Float::with_val(p, 0), root.clone()])
        {
            let normalized = value >> shift;
            assert!(Float::with_val(p, normalized - expected).abs() < Float::with_val(p, 1) >> 95);
        }
    }
}

#[test]
fn default_working_precision_is_explicit_and_strict_route_rejects_loss() {
    let mut a = star(256, 0);
    a[0] = Float::with_val(256, 1);
    a[0] += Float::with_val(256, 1) >> 90;
    let rounded: Vec<_> = a.iter().map(|x| Float::with_val(64, x)).collect();
    let actual = householder_tridiag_hp(&a, 3, 64).unwrap();
    assert_eq!(
        actual,
        householder_tridiag_hp_stable(&rounded, 3, 64).unwrap()
    );
    assert!(actual
        .0
        .iter()
        .chain(&actual.1)
        .chain(&actual.2)
        .all(|x| x.prec() == 64));
    assert!(householder_tridiag_hp_stable(&a, 3, 64).is_err());
    assert!(householder_tridiag_hp(&a, 3, 33).is_ok());
}

#[test]
fn default_reduction_rejects_invalid_domains_before_arithmetic() {
    let a = star(128, 0);
    for p in [0, 32, 1_000_001] {
        assert!(householder_tridiag_hp(&a, 3, p).is_err());
    }
    assert!(householder_tridiag_hp(&a, usize::MAX, 128).is_err());
    assert!(householder_tridiag_hp(&[], 0, 128).is_err());
    let mut bad = a.clone();
    bad[1] += Float::with_val(128, 1) >> 100;
    // Source asymmetry cannot be hidden by rounding to the requested precision.
    assert!(householder_tridiag_hp(&bad, 3, 64).is_err());
    bad = a;
    bad[0] = Float::with_val(128, rug::float::Special::Nan);
    assert!(householder_tridiag_hp(&bad, 3, 128).is_err());
}
