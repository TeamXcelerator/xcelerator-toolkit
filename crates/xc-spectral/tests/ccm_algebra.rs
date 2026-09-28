#![cfg(feature = "hp")]

use rug::{Float, Rational};
use xc_core::DecimalLiteral;
use xc_numerics::mpfr_interval::MpfrInterval;
use xc_root::{IntervalNewtonOptions, IntervalRootStatus};
use xc_spectral::ccm::certified_roots::{reconcile_complete_window, CertifiedSecularFunction};
use xc_spectral::ccm::rank_one::FiniteRankOneOperatorF64;

// Keep the independent elimination oracle in explicit row/column notation.
#[allow(clippy::needless_range_loop)]
fn determinant(mut a: Vec<Vec<Rational>>) -> Rational {
    let mut value = Rational::from(1);
    for k in 0..a.len() {
        let Some(pivot) = (k..a.len()).find(|&j| a[j][k] != 0) else {
            return Rational::from(0);
        };
        if pivot != k {
            a.swap(pivot, k);
            value = -value;
        }
        let diagonal = a[k][k].clone();
        value *= &diagonal;
        for i in k + 1..a.len() {
            let ratio = a[i][k].clone() / &diagonal;
            for j in k + 1..a.len() {
                let delta = ratio.clone() * &a[k][j];
                a[i][j] -= delta;
            }
        }
    }
    value
}

#[test]
fn quotient_characteristic_matches_independent_exact_partial_fraction_identity() {
    // Every normalized entry is dyadic. Thus the public binary64 quotient is
    // exact here, and rational Gaussian elimination checks its characteristic
    // polynomial without using either numerical spectral/root solver.
    for state in [
        [1, 1, 4, 1, 1],
        [2, -1, 4, 1, 2],
        [0, 1, 2, 4, 1],
        [-4, 8, -1, 2, 3],
    ] {
        let floating = state.map(f64::from);
        let operator = FiniteRankOneOperatorF64::from_state(&floating).unwrap();
        let q = operator.quotient_matrix();
        let sum: i32 = state.iter().sum();
        assert_eq!(sum, 8);
        for numerator in -10..=10 {
            let z = Rational::from((numerator, 3));
            if (-2..=2).any(|d| z == d) {
                continue;
            }
            let matrix = (0..4)
                .map(|row| {
                    (0..4)
                        .map(|column| {
                            let entry =
                                Float::with_val(53, q[(row, column)]).to_rational().unwrap();
                            (if row == column {
                                z.clone()
                            } else {
                                Rational::from(0)
                            }) - entry
                        })
                        .collect()
                })
                .collect();
            let mut product = Rational::from(1);
            let mut secular = Rational::from(0);
            for (j, weight) in state.iter().enumerate() {
                let denominator = z.clone() - (j as i32 - 2);
                product *= &denominator;
                secular += Rational::from((*weight, sum)) / denominator;
            }
            assert_eq!(determinant(matrix), product * secular);
        }
    }
}

#[test]
fn mixed_sign_same_gap_roots_have_exact_count_and_generic_reconciliation_reason() {
    let p = 192;
    let poles = [-3, -1, 2].map(|v| Float::with_val(p, v));
    let weights = [18, -5, 2].map(|v| Float::with_val(p, v));
    // By direct common-denominator expansion, R(z) =
    // 15*z*(z-1) / ((z+3)*(z+1)*(z-2)). Both roots are in ONE pole gap.
    let source = CertifiedSecularFunction::from_point_data(&poles, &weights, p).unwrap();
    let entire = source.normalized_finite_entire_function().unwrap();
    let expected = [0, -1, 1].map(Rational::from);
    assert_eq!(entire.coefficients().len(), expected.len());
    for (actual, expected) in entire.coefficients().iter().zip(expected) {
        assert_eq!(actual.real, expected);
        assert_eq!(actual.imaginary, 0);
    }
    let count = source.exact_numerator_count_between_poles(1, 2).unwrap();
    assert_eq!(count.certified_root_count, 2);
    assert_eq!(count.open_intervals_counted, 1);
    let options = IntervalNewtonOptions {
        width_tolerance: DecimalLiteral::new("1e-30").unwrap(),
        maximum_iterations: 100,
    };
    let roots = [0, 1].map(|center| {
        let lower = Float::with_val(p, Rational::from((center * 16 - 1, 16)));
        let upper = Float::with_val(p, Rational::from((center * 16 + 1, 16)));
        let root = source
            .isolate(&MpfrInterval::new(lower, upper).unwrap(), &options)
            .unwrap();
        assert_eq!(root.status, IntervalRootStatus::CertifiedUnique);
        root
    });
    let reconciliation = reconcile_complete_window(&roots, &count).unwrap();
    assert!(reconciliation.complete);
    assert_eq!(reconciliation.isolated_root_count, 2);
    assert!(
        !reconciliation
            .reason
            .contains("every monotonic pole interval has one"),
        "mixed-sign two-roots-per-gap proof was mislabeled: {}",
        reconciliation.reason
    );
}
