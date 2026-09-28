use xc_solver::TridiagonalProblemF64;

#[test]
fn small_positive_pivot_preserves_negative_eigenvalue_and_bracket() {
    let problem = TridiagonalProblemF64::new(&[1e-20, 1.], &[1e-9]).unwrap();
    assert_eq!(problem.sturm_count_below(0.).unwrap(), 1);
    let (lo, hi) = problem.bisect_index(0, 1e-23, 256).unwrap();
    assert!(hi < 0. && lo < hi);
    assert!((1e-20 - lo) * (1. - lo) - 1e-18 > 0.);
    assert!((1e-20 - hi) * (1. - hi) - 1e-18 < 0.);
}

#[test]
fn exact_zero_pivots_and_eigenvalue_thresholds_keep_strict_counts() {
    let problem = TridiagonalProblemF64::new(&[0., 0.], &[1.]).unwrap();
    for (threshold, expected) in [(-1., 0), (0., 1), (1., 1), (1.0_f64.next_up(), 2)] {
        assert_eq!(problem.sturm_count_below(threshold).unwrap(), expected);
    }
    let repeated = TridiagonalProblemF64::new(&[-1., 0., 0., 2.], &[0., 0., 0.]).unwrap();
    assert_eq!(repeated.sturm_count_below(0.).unwrap(), 1);
    assert_eq!(repeated.sturm_count_below(f64::from_bits(1)).unwrap(), 3);
    assert_eq!(
        TridiagonalProblemF64::new(&[0.; 4], &[1.; 3])
            .unwrap()
            .sturm_count_below(0.)
            .unwrap(),
        2
    );
}

#[test]
fn subnormal_couplings_and_large_finite_entries_do_not_overflow_the_count() {
    for off in [f64::from_bits(1), 1e-200, 1e200, f64::MAX] {
        assert_eq!(
            TridiagonalProblemF64::new(&[0., 0.], &[off])
                .unwrap()
                .sturm_count_below(0.)
                .unwrap(),
            1
        );
    }
    let problem = TridiagonalProblemF64::new(&[-1e308, 1e308], &[0.]).unwrap();
    for (index, expected) in [(0, -1e308), (1, 1e308)] {
        let (lo, hi) = problem.bisect_index(index, 1e294, 100).unwrap();
        assert!(lo.is_finite() && hi.is_finite() && lo <= expected && expected <= hi);
    }
}

#[test]
fn invalid_public_fields_and_unresolvable_brackets_fail_explicitly() {
    let malformed = TridiagonalProblemF64 {
        diagonal: &[0., 1.],
        off_diagonal: &[],
    };
    assert!(malformed.sturm_count_below(0.).is_err());
    assert!(malformed.bisect_index(0, 1e-8, 100).is_err());
    let one = TridiagonalProblemF64::new(&[1.], &[]).unwrap();
    assert!(one.bisect_index(0, f64::MIN_POSITIVE, 100).is_err());
    let huge = TridiagonalProblemF64::new(&[f64::MAX, f64::MAX], &[f64::MAX]).unwrap();
    assert!(huge.bisect_index(1, 1e-8, 100).is_err());
}

#[test]
fn rounded_subtraction_cannot_hide_an_excess_bracket_width() {
    let diagonal = [-1., 2.0_f64.powi(-100)];
    let problem = TridiagonalProblemF64::new(&diagonal, &[0.]).unwrap();
    let tolerance = 1.0_f64.next_up();
    for index in 0..2 {
        let (lo, hi) = problem.bisect_index(index, tolerance, 100).unwrap();
        // The old initial bracket straddled zero, with -lo == tolerance and
        // hi > 0: its exact width exceeded tolerance despite rounded equality.
        assert!(hi - lo < tolerance);
    }
}
