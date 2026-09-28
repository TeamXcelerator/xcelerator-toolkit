use xc_spectral::ccm::CcmParams;
#[test]
fn exhaustive_ccm_index_rejects_values_outside_the_centered_basis() {
    let parameters = CcmParams::from_lambda_sq_integer(13, 1);
    for index in [-2, 2, i64::MIN, i64::MAX] {
        assert!(
            std::panic::catch_unwind(|| parameters.idx(index)).is_err(),
            "accepted index {index}"
        );
    }
    assert_eq!(
        (parameters.idx(-1), parameters.idx(0), parameters.idx(1)),
        (0, 1, 2)
    );
}
#[test]
fn exhaustive_ccm_matrix_size_rejects_overflow_in_every_profile() {
    for modes in [usize::MAX, usize::MAX / 2 + 1] {
        let parameters = CcmParams::from_lambda_sq_integer(13, modes);
        assert!(std::panic::catch_unwind(|| parameters.matrix_size()).is_err());
    }
}

#[test]
fn exhaustive_ccm_index_preserves_extreme_representable_shapes() {
    let modes = usize::MAX / 2;
    let parameters = CcmParams::from_lambda_sq_integer(13, modes);
    assert_eq!(parameters.matrix_size(), usize::MAX);
    let extreme = i64::try_from(modes).unwrap();
    assert_eq!(parameters.idx(-extreme), 0);
    assert_eq!(parameters.idx(0), modes);
    assert_eq!(parameters.idx(extreme), usize::MAX - 1);
}
