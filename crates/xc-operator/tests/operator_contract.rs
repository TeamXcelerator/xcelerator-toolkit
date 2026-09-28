use xc_operator::{DenseSymmetricF64, DiagonalF64, LinearOperator, TridiagonalF64};
#[test]
fn a_norm_upper_bound_cannot_round_below_the_exact_largest_eigenvalue() {
    let small = 2_f64.powi(-54);
    let dense = DenseSymmetricF64::new("tiny coupling", 2, vec![1., small, small, 1.], 0.).unwrap();
    let tri = TridiagonalF64::new("tiny coupling", vec![1., 1.], vec![small]).unwrap();
    // Exact largest eigenvalue is 1+2^-54, strictly larger than one.
    assert!(dense.norm_bound().unwrap() > 1.);
    assert!(tri.norm_bound().unwrap() > 1.);
}
#[test]
fn symmetric_operators_require_symmetric_storage_and_finite_actions() {
    assert!(DenseSymmetricF64::new("not symmetric", 2, vec![1., 0.01, 0., 1.], 0.1).is_err());
    let d = DiagonalF64::new("overflow", vec![f64::MAX]).unwrap();
    assert!(d.apply(&[2.], &mut [0.]).is_err());
    assert!(d.apply(&[f64::NAN], &mut [0.]).is_err());
}
#[cfg(feature = "hp")]
#[test]
fn hp_norm_and_input_contracts_are_mathematical() {
    use rug::{float::Special, Float};
    use xc_operator::{DenseSymmetricHp, TridiagonalHp};
    let p = 128;
    let small = Float::with_val(p, 1) >> 130_u32;
    let a = vec![
        Float::with_val(p, 1),
        small.clone(),
        small.clone(),
        Float::with_val(p, 1),
    ];
    let dense = DenseSymmetricHp::new("tiny coupling", 2, a, p, &Float::with_val(p, 0)).unwrap();
    let tri = TridiagonalHp::new(
        "tiny coupling",
        vec![Float::with_val(p, 1); 2],
        vec![small],
        p,
    )
    .unwrap();
    assert!(dense.norm_bound().unwrap() > 1);
    assert!(tri.norm_bound().unwrap() > 1);
    assert!(DenseSymmetricHp::new(
        "invalid",
        1,
        vec![Float::with_val(p, Special::Nan)],
        p,
        &Float::with_val(p, 0)
    )
    .is_err());
    assert!(
        TridiagonalHp::new("invalid", vec![Float::with_val(p, Special::Nan)], vec![], p).is_err()
    );
    assert!(dense
        .apply(
            &vec![Float::with_val(p, Special::Nan); 2],
            &mut vec![Float::with_val(p, 0); 2]
        )
        .is_err());
}
