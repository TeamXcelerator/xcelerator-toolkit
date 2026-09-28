#![cfg(feature = "hp")]

use rug::Rational;
use std::sync::mpsc;
use std::time::Duration;
use xc_certify::exact::{
    build_portable_selected_eigenvalue_certificate, certify_interval_matrix_eigenvalues_below,
    certify_interval_matrix_eigenvalues_in_open_interval, certify_selected_interval_eigenvalue,
    verify_portable_selected_eigenvalue_certificate,
};
use xc_certify::{
    IntervalEigenvalueCountResult as Count, SelectedEigenvalueEnclosureResult as Selected,
};
use xc_numerics::interval::RationalInterval;

fn point(value: i32) -> RationalInterval {
    RationalInterval::point(value.into())
}

fn promptly_inconclusive(action: impl FnOnce() -> Count + Send + 'static) {
    // A regression in release indexing must fail this test, not hang the suite.
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(action());
    });
    let result = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("invalid shape must be rejected before iteration or indexing");
    assert!(matches!(result, Count::Inconclusive { .. }));
}

#[test]
fn invalid_count_dimensions_are_rejected_before_index_arithmetic() {
    for n in [0, 2, usize::MAX, usize::MAX / 2 + 1] {
        promptly_inconclusive(move || {
            certify_interval_matrix_eigenvalues_below(&[point(3)], n, 1.into())
        });
        promptly_inconclusive(move || {
            certify_interval_matrix_eigenvalues_in_open_interval(&[point(3)], n, 1.into(), 2.into())
        });
    }
    promptly_inconclusive(|| certify_interval_matrix_eigenvalues_below(&[], 1, 0.into()));
}

#[test]
fn strict_count_boundaries_and_interval_uncertainty_keep_their_meaning() {
    // J_3 - 2I has eigenvalues -2, -2, 1. Entry errors <= 1/1000
    // have spectral norm <= 3/1000 and cannot cross the separated thresholds.
    for uncertain in [false, true] {
        let radius = if uncertain {
            Rational::from((1, 1000))
        } else {
            Rational::from(0)
        };
        let matrix = [-1, 1, 1, 1, -1, 1, 1, 1, -1]
            .into_iter()
            .map(|v| {
                RationalInterval::new(Rational::from(v) - &radius, Rational::from(v) + &radius)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for (threshold, expected) in [(-3, 0), (0, 2), (2, 3)] {
            let Count::Conclusive { certificate } =
                certify_interval_matrix_eigenvalues_below(&matrix, 3, threshold.into())
            else {
                panic!("separated threshold should resolve");
            };
            assert_eq!(certificate.eigenvalue_count, expected);
        }
        for threshold in [-2, 1] {
            assert!(matches!(
                certify_interval_matrix_eigenvalues_below(&matrix, 3, threshold.into()),
                Count::Inconclusive { .. }
            ));
        }
        let Count::Conclusive { certificate } =
            certify_interval_matrix_eigenvalues_in_open_interval(&matrix, 3, (-3).into(), 0.into())
        else {
            panic!("separated open interval should resolve");
        };
        assert_eq!(certificate.eigenvalue_count, 2);
    }
}

#[test]
fn selected_indices_retain_clusters_and_portable_boundary_proofs() {
    let matrix = [-1, 1, 1, 1, -1, 1, 1, 1, -1]
        .into_iter()
        .map(point)
        .collect::<Vec<_>>();
    for index in 0..3 {
        let eigenvalue = if index < 2 { -2 } else { 1 };
        let lower = Rational::from(eigenvalue) - Rational::from((1, 3));
        let upper = Rational::from(eigenvalue) + Rational::from((2, 3));
        let Selected::Conclusive { certificate } = certify_selected_interval_eigenvalue(
            &matrix,
            3,
            index,
            lower,
            upper,
            Rational::from((1, 64)),
            20,
        ) else {
            panic!("known selected cluster should resolve");
        };
        assert_eq!(
            (
                certificate.first_enclosed_index,
                certificate.last_enclosed_index
            ),
            if index < 2 { (0, 1) } else { (2, 2) }
        );
        assert_eq!(certificate.simple, index == 2);
        let portable =
            build_portable_selected_eigenvalue_certificate(&matrix, &certificate).unwrap();
        assert!(verify_portable_selected_eigenvalue_certificate(&portable).valid);
        let mut changed = portable;
        changed.enclosure.lower_boundary.count_below += 1;
        assert!(!verify_portable_selected_eigenvalue_certificate(&changed).valid);
    }
}

#[test]
fn invalid_selection_controls_fail_and_exact_midpoint_hits_certify() {
    let matrix = vec![point(3)];
    for (n, index, lower, upper, width, steps) in [
        (0, 0, 2, 4, 1, 20),
        (2, 0, 2, 4, 1, 20),
        (1, 1, 2, 4, 1, 20),
        (1, 0, 3, 3, 1, 20),
        (1, 0, 4, 2, 1, 20),
        (1, 0, 2, 4, 0, 20),
        (1, 0, 2, 4, -1, 20),
        (1, 0, 2, 4, 1, 0),
        (1, 0, 4, 5, 1, 20),
    ] {
        assert!(matches!(
            certify_selected_interval_eigenvalue(
                &matrix,
                n,
                index,
                lower.into(),
                upper.into(),
                width.into(),
                steps
            ),
            Selected::Inconclusive { .. }
        ));
    }
    // The exact diagonal spectrum {3} remains isolatable when the midpoint is a root.
    let Selected::Conclusive { certificate } = certify_selected_interval_eigenvalue(
        &matrix,
        1,
        0,
        2.into(),
        4.into(),
        Rational::from((1, 64)),
        20,
    ) else {
        panic!("exact midpoint must retain the known isolated eigenvalue");
    };
    let rational = |x: &xc_certify::ExactRationalRecord| -> Rational {
        format!("{}/{}", x.numerator, x.denominator)
            .parse()
            .unwrap()
    };
    let lower = rational(&certificate.lower);
    let upper = rational(&certificate.upper);
    assert!(lower < 3 && upper > 3);
    assert!(upper - lower <= Rational::from((1, 64)));
    assert!(certificate.simple);
    let portable = build_portable_selected_eigenvalue_certificate(&matrix, &certificate).unwrap();
    assert!(verify_portable_selected_eigenvalue_certificate(&portable).valid);
    assert!(matches!(
        certify_selected_interval_eigenvalue(
            &matrix,
            1,
            0,
            Rational::from((7, 3)),
            Rational::from((10, 3)),
            Rational::from((1, 1024)),
            1
        ),
        Selected::Inconclusive { .. }
    ));
}
