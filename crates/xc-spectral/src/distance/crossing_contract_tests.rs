use super::*;
use std::cell::Cell;
#[cfg(feature = "hp")]
use std::cell::RefCell;

#[test]
fn native_crossings_keep_opposite_signs_across_zeros() {
    let target = crate::target::TargetEvaluatorF64::from_environment().unwrap();
    for (signs, expected) in [
        (vec![-1., 0., 1.], vec![(2., 4.)]),
        (vec![0., -1., 0., 1., 0., -1.], vec![(3., 5.), (5., 7.)]),
        (vec![1., 0., 1.], vec![]),
        (vec![0., 0., 0.], vec![]),
    ] {
        let next = Cell::new(0);
        let report = target_crossings_f64(
            |u| {
                let i = next.get();
                next.set(i + 1);
                target.try_value(u).unwrap() + signs[i]
            },
            signs.len() as f64 + 1.,
            signs.len(),
            GridVariable::U,
        )
        .unwrap();
        assert_eq!(report.brackets, expected);
        assert_eq!(
            report.initial_sign,
            signs
                .iter()
                .copied()
                .find(|x| *x != 0.)
                .map_or(0, |x| if x < 0. { -1 } else { 1 })
        );
    }
}

#[test]
fn native_crossings_reject_nonfinite_samples_and_collapsed_grid() {
    for variable in [GridVariable::U, GridVariable::LogU] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let calls = Cell::new(0);
            assert!(target_crossings_f64(
                |_| {
                    calls.set(calls.get() + 1);
                    bad
                },
                4.,
                3,
                variable
            )
            .is_err());
            assert_eq!(calls.get(), 1);
        }
        assert!(
            target_crossings_f64(|_| 1., f64::from_bits(1f64.to_bits() + 1), 3, variable).is_err()
        );
    }
}

#[cfg(feature = "hp")]
#[test]
fn hp_crossings_keep_opposite_signs_across_zeros() {
    use rug::Float;
    let p = 128;
    let target = crate::target::hp::TargetEvaluator::from_environment(p + 64).unwrap();
    for (signs, expected) in [
        (vec![-1, 0, 1], vec![(2, 4)]),
        (vec![0, -1, 0, 1, 0, -1], vec![(3, 5), (5, 7)]),
        (vec![1, 0, 1], vec![]),
        (vec![0, 0, 0], vec![]),
    ] {
        let next = Cell::new(0);
        let report = hp::target_crossings(
            |u| {
                let i = next.get();
                next.set(i + 1);
                target.try_value(u).unwrap() + signs[i]
            },
            &Float::with_val(p, signs.len() + 1),
            signs.len(),
            GridVariable::U,
            p,
        )
        .unwrap();
        assert_eq!(report.brackets.len(), expected.len());
        for ((left, right), (a, b)) in report.brackets.iter().zip(expected) {
            assert_eq!(left, &a);
            assert_eq!(right, &b);
        }
    }
}

#[cfg(feature = "hp")]
#[test]
fn hp_crossings_validate_samples_precision_and_full_width_counts() {
    use rug::Float;
    let p = 128;
    let lambda = Float::with_val(p, 4);
    for variable in [GridVariable::U, GridVariable::LogU] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let calls = Cell::new(0);
            assert!(hp::target_crossings(
                |_| {
                    calls.set(calls.get() + 1);
                    Float::with_val(p, bad)
                },
                &lambda,
                3,
                variable,
                p
            )
            .is_err());
            assert_eq!(calls.get(), 1);
        }
    }
    for bits in [0, 999937, 1_000_001, u32::MAX] {
        let calls = Cell::new(0);
        assert!(hp::target_crossings(
            |_| {
                calls.set(calls.get() + 1);
                Float::with_val(p, 1)
            },
            &lambda,
            3,
            GridVariable::U,
            bits
        )
        .is_err());
        assert_eq!(calls.get(), 0);
    }
    if let Ok(samples) = usize::try_from(u64::from(u32::MAX) + 2) {
        let first = RefCell::new(None);
        assert!(hp::target_crossings(
            |u| {
                *first.borrow_mut() = Some(u.clone());
                Float::with_val(p, f64::NAN)
            },
            &lambda,
            samples,
            GridVariable::U,
            p
        )
        .is_err());
        let first = first.into_inner().expect("first sample evaluated");
        assert!(first > 1);
        assert!(first < Float::with_val(p, 1.001));
    }
}
