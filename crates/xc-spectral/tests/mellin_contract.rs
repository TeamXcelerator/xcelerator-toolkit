use xc_spectral::mellin::*;
#[test]
fn native_eta_kernel_preserves_small_finite_values() {
    for t in [350.0f64, 400.0, 500.0, 501.0] {
        let expected = t * (-t).exp();
        assert!((omega_f64(t) / expected - 1.0).abs() < 1e-13);
        assert_eq!(omega_f64(-t), -omega_f64(t));
    }
    assert!(omega_f64(746.0) > 0.0);
    assert!(omega_f64(f64::NAN).is_nan());
}
#[test]
fn native_scans_preserve_midpoint_sampled_and_tiny_crossings() {
    for scale in [1.0, 1e-200] {
        let evaluate = |_: f64, t: f64| ((t - 1.0) * scale, 1.0);
        for count in [1, 2, 3] {
            let roots =
                try_scan_critical_line_real_crossings_f64(&evaluate, 0.0, 2.0, count).unwrap();
            assert_eq!(roots.len(), 1);
            assert!((roots[0] - 1.0).abs() < 1e-14);
        }
    }
    assert!(
        try_scan_critical_line_real_crossings_f64(&|_, _| (0.0, f64::NAN), 0.0, 2.0, 1).is_err()
    );
    assert!(try_scan_critical_line_real_crossings_f64(&|_, t| (t, 0.0), 0.0, 2.0, 0).is_err());
}
#[cfg(feature = "hp")]
#[test]
fn hp_eta_retains_precision_above_the_old_cutoff() {
    use rug::Float;
    let p = 2048;
    let t = Float::with_val(p, 501);
    let e = t.clone().exp();
    let mut denominator = e.clone();
    denominator += 1u32;
    denominator.square_mut();
    let mut expected = Float::with_val(p, &t * &e);
    expected /= denominator;
    let mut relative = omega_hp(&t) - &expected;
    relative /= expected;
    relative.abs_mut();
    assert!(relative < (Float::with_val(p, 1) >> 1900u32));
}
#[cfg(feature = "hp")]
#[test]
fn hp_scans_preserve_exact_and_extreme_brackets() {
    use rug::Float;
    let p = 128;
    let evaluate = |_: &Float, t: &Float| (Float::with_val(p, t - 1u32), Float::with_val(p, 1));
    for count in [1, 2, 3] {
        let roots = try_scan_critical_line_real_crossings_hp(
            &evaluate,
            &Float::with_val(p, 0),
            &Float::with_val(p, 2),
            count,
            150,
        )
        .unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0].clone() - 1u32).abs() < (Float::with_val(p, 1) >> 100u32));
    }
    let huge: Float = Float::with_val(p, 1) << (rug::float::exp_max() - 1) as u32;
    let roots = try_scan_critical_line_real_crossings_hp(
        &|_, t| (t.clone(), Float::with_val(p, 1)),
        &(-huge.clone()),
        &huge,
        1,
        100,
    )
    .unwrap();
    assert_eq!(roots.len(), 1);
    assert!(roots[0].is_zero());
}
