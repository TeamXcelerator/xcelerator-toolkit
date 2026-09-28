#![cfg(feature = "hp")]
use rug::Float;
use xc_core::EigenTarget;
use xc_spectral::prolate::hp::{
    build_pw_subspace_forms, solve_pw_subspace_extreme, ProlateSubspaceFormsHp,
};

#[test]
fn prolate_rejects_dependent_and_unresolved_bases() {
    let p = 256;
    let lambda = Float::with_val(p, 2).sqrt();
    let v0 = [-7, -1, 2, 6, -2, -6, -3, 0, -1].map(|x| Float::with_val(p, x) / 8);
    let v1 = [1, 7, 8, -3, 1, -8, 1, -7, 4].map(|x| Float::with_val(p, x) / 8);
    for bits in [0, 124, 127, 130, 180] {
        let mut v2 = v0
            .iter()
            .zip(&v1)
            .map(|(a, b)| Float::with_val(p, a + b))
            .collect::<Vec<_>>();
        if bits != 0 {
            v2[7] += Float::with_val(p, 1) >> bits;
        }
        assert!(build_pw_subspace_forms(&lambda, 9, &[v0.to_vec(), v1.to_vec(), v2], p).is_err());
    }
    let overcomplete = vec![
        vec![Float::with_val(p, 1) / 7],
        vec![Float::with_val(p, 11) / 3],
    ];
    assert!(build_pw_subspace_forms(&lambda, 1, &overcomplete, p).is_err());
    let forms = build_pw_subspace_forms(&lambda, 9, &[v0.to_vec(), v1.to_vec()], p).unwrap();
    let report = solve_pw_subspace_extreme(&forms, EigenTarget::AlgebraicSmallest).unwrap();
    let reference = Float::with_val(p, Float::parse("57.3908369988043521826973628016").unwrap());
    assert!(
        Float::with_val(p, &report.eigenvalue - reference).abs()
            < Float::with_val(p, 1) / 1_000_000_000_000u64
    );
}

#[test]
fn prolate_public_gram_cannot_bypass_rank_gate() {
    let p = 256;
    let forms = ProlateSubspaceFormsHp {
        ambient_dimension: 3,
        basis_dimension: 2,
        precision_bits: p,
        stiffness: [1, 0, 0, 1].map(|x| Float::with_val(p, x)).to_vec(),
        gram: vec![
            Float::with_val(p, 1),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 1) >> 200,
        ],
    };
    assert!(solve_pw_subspace_extreme(&forms, EigenTarget::AlgebraicSmallest).is_err());
}
