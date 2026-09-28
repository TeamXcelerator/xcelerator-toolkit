use xc_spectral::ccm::window::{compare_ordered_roots_to_references, CcmRootRecord, RootStatus};

#[test]
#[cfg(feature = "hp")]
fn decimal_root_comparison_must_not_erase_a_nonzero_stored_displacement() {
    // The exact decimal is 1 + 2^-60, so its error relative to the EXACT
    // binary64 reference 1 is 2^-60. This does not test true-zeta accuracy.
    let mut roots = vec![CcmRootRecord {
        positive_index: Some(1),
        midpoint: "1.000000000000000000867361737988403547205962240695953369140625".into(),
        enclosure: None,
        residual_bound: None,
        derivative_magnitude: None,
        conditioning: None,
        isolation_distance: None,
        nearest_left_pole: None,
        nearest_right_pole: None,
        precision_bits: 128,
        precision_history_bits: vec![128],
        certified_digits: None,
        status: RootStatus::Refined,
        discovery_method: "manufactured exact dyadic point".into(),
        crosscheck_method: None,
        reference_comparison_digits: None,
    }];
    compare_ordered_roots_to_references(&mut roots, &[1.0]).unwrap();
    let digits = roots[0].reference_comparison_digits.unwrap();
    let expected = 60.0 * std::f64::consts::LOG10_2;
    assert!(
        digits.is_finite() && (digits - expected).abs() < 1e-12,
        "nonzero exact point displacement has {expected} digits, got {digits}"
    );
}

fn ordinary_root(midpoint: &str, precision_bits: u32) -> CcmRootRecord {
    CcmRootRecord {
        positive_index: Some(1),
        midpoint: midpoint.into(),
        enclosure: None,
        residual_bound: None,
        derivative_magnitude: None,
        conditioning: None,
        isolation_distance: None,
        nearest_left_pole: None,
        nearest_right_pole: None,
        precision_bits,
        precision_history_bits: vec![precision_bits],
        certified_digits: None,
        status: RootStatus::Refined,
        discovery_method: "manufactured point".into(),
        crosscheck_method: None,
        reference_comparison_digits: None,
    }
}

#[test]
fn comparison_preserves_exact_stored_equality_and_rejects_invalid_inputs() {
    let mut equal = [ordinary_root("1", 53)];
    compare_ordered_roots_to_references(&mut equal, &[1.0]).unwrap();
    assert_eq!(equal[0].reference_comparison_digits, Some(f64::INFINITY));
    let mut distinct = [ordinary_root("1.125", 53)];
    compare_ordered_roots_to_references(&mut distinct, &[1.0]).unwrap();
    assert!(
        (distinct[0].reference_comparison_digits.unwrap() - 3.0 * std::f64::consts::LOG10_2).abs()
            < 1e-14
    );
    for (midpoint, p) in [("NaN", 128), ("inf", 128), ("1", 0), ("1", 1_000_001)] {
        assert!(
            compare_ordered_roots_to_references(&mut [ordinary_root(midpoint, p)], &[1.0]).is_err()
        );
    }
}

#[test]
#[cfg(feature = "hp")]
fn comparison_logs_the_displacement_before_binary64_conversion() {
    let mut roots = [ordinary_root("1e-400", 256)];
    compare_ordered_roots_to_references(&mut roots, &[f64::from_bits(1)]).unwrap();
    // The reference is exactly2^-1074; source is much smaller. The error is
    // nonzero, although an intermediate binary64 parse of source would be zero.
    assert!(
        (roots[0].reference_comparison_digits.unwrap() - 1074.0 * std::f64::consts::LOG10_2).abs()
            < 1e-12
    );
}

#[test]
#[cfg(not(feature = "hp"))]
fn native_comparison_explicitly_rejects_high_precision_records() {
    let mut roots = [ordinary_root(
        "1.000000000000000000867361737988403547205962240695953369140625",
        128,
    )];
    let error = compare_ordered_roots_to_references(&mut roots, &[1.0]).unwrap_err();
    assert!(error.to_string().contains("hp feature"));
    assert!(roots[0].reference_comparison_digits.is_none());
}
