#![cfg(feature = "hp")]
use rug::Rational;
use xc_certify::{
    exact::*, PortableSelectedEigenvalueCertificate, SelectedEigenvalueEnclosureResult,
};
use xc_numerics::interval::RationalInterval as I;
fn enclosure(
    result: SelectedEigenvalueEnclosureResult,
) -> xc_certify::ExactSelectedEigenvalueEnclosure {
    match result {
        SelectedEigenvalueEnclosureResult::Conclusive { certificate } => *certificate,
        other => panic!("expected independent count proof: {other:?}"),
    }
}
#[test]
fn mpfr_selected_interval_proofs_match_independent_hadamard_eigenvalues() {
    for n in [4usize, 8, 16] {
        // H H^T=n I, so H diag(1,...,n) H^T/n has the exact integer spectrum.
        let eps: Rational = Rational::from((1, 1)) >> 100;
        let mut matrix = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let mut entry = 0i64;
                for k in 0..n {
                    let sign = if ((i & k).count_ones() + (j & k).count_ones()) % 2 == 0 {
                        1
                    } else {
                        -1
                    };
                    entry += sign * (k + 1) as i64;
                }
                let point = Rational::from((entry, n));
                matrix.push(I::new(point.clone() - &eps, point + &eps).unwrap());
            }
        }
        for index in [0, n / 2, n - 1] {
            let eigenvalue = Rational::from(index + 1);
            eprintln!("Hadamard interval selection n={n}, index={index}, p=256, radius=2^-100, width=2^-60");
            let cert = enclosure(certify_selected_interval_eigenvalue_mpfr(
                &matrix,
                n,
                index,
                eigenvalue.clone() - Rational::from((1, 2)),
                eigenvalue.clone() + Rational::from((1, 2)),
                Rational::from(1) >> 60,
                200,
                256,
            ));
            assert!(cert.simple);
            assert_eq!(cert.inertia_precision_bits, Some(256));
            let portable = build_portable_selected_eigenvalue_certificate(&matrix, &cert).unwrap();
            assert_eq!(portable.schema_version, 2);
            assert!(
                verify_portable_selected_eigenvalue_certificate(&portable)
                    .mathematical_claim_verified
            );
            let parse = |r: &xc_certify::ExactRationalRecord| {
                Rational::from((
                    r.numerator.parse::<rug::Integer>().unwrap(),
                    r.denominator.parse::<rug::Integer>().unwrap(),
                ))
            };
            assert!(parse(&cert.lower) < eigenvalue && parse(&cert.upper) > eigenvalue);
            let mut mutations = Vec::new();
            let mut bad = portable.clone();
            bad.enclosure.inertia_precision_bits = Some(32);
            mutations.push(bad);
            let mut bad = portable.clone();
            bad.schema_version = 3;
            mutations.push(bad);
            let mut bad = portable.clone();
            bad.enclosure.requested_index = (index + 1) % n;
            mutations.push(bad);
            let mut bad = portable.clone();
            bad.enclosure.lower.numerator = "1000".into();
            mutations.push(bad);
            let mut bad = portable.clone();
            bad.enclosure.lower_boundary.count_below += 1;
            mutations.push(bad);
            let mut bad = portable.clone();
            bad.enclosure.lower_boundary.pivot_enclosures[0]
                .lower
                .numerator = "123456789".into();
            mutations.push(bad);
            let mut bad = portable.clone();
            bad.matrix_row_major[0].lower.numerator = "123456789".into();
            mutations.push(bad);
            for bad in mutations {
                assert!(!verify_portable_selected_eigenvalue_certificate(&bad).valid);
            }
        }
    }
}
#[test]
fn exact_midpoint_eigenvalue_uses_an_independently_proven_interior_split() {
    for root in [
        Rational::from((1, 2)),
        Rational::from((3, 8)),
        Rational::from((7, 16)),
    ] {
        let matrix = vec![
            I::point(root.clone()),
            I::point(Rational::from(0)),
            I::point(Rational::from(0)),
            I::point(Rational::from(3)),
        ];
        let cert = enclosure(certify_selected_interval_eigenvalue(
            &matrix,
            2,
            0,
            root.clone() - Rational::from((1, 4)),
            root.clone() + Rational::from((1, 4)),
            Rational::from(1) >> 20,
            200,
        ));
        let portable = build_portable_selected_eigenvalue_certificate(&matrix, &cert).unwrap();
        assert_eq!(portable.schema_version, 3);
        assert!(verify_portable_selected_eigenvalue_certificate(&portable).valid);
    }
}
#[test]
fn recorded_exact_proof_replays_despite_new_producer_growth_budget() {
    // Exact diagonal pivots have a large exponent, but no elimination growth.
    // MPFR can generate exactly the same dyadic evidence at small precision.
    let tiny: Rational = Rational::from(1) >> 300_000;
    let matrix = vec![
        I::point(-tiny.clone()),
        I::point(Rational::from(0)),
        I::point(Rational::from(0)),
        I::point(tiny.clone()),
    ];
    assert!(matches!(
        certify_selected_interval_eigenvalue(
            &matrix,
            2,
            0,
            -2 * tiny.clone(),
            Rational::from(0),
            2 * tiny.clone(),
            4
        ),
        SelectedEigenvalueEnclosureResult::Inconclusive { .. }
    ));
    let mut cert = enclosure(certify_selected_interval_eigenvalue_mpfr(
        &matrix,
        2,
        0,
        -2 * tiny.clone(),
        Rational::from(0),
        2 * tiny.clone(),
        4,
        64,
    ));
    cert.inertia_precision_bits = None;
    // Frozen first-signed historical order, derived exactly from the diagonal
    // source. New stable pivot selection intentionally has a different order.
    let point = |x: Rational| {
        let value = xc_certify::ExactRationalRecord {
            numerator: x.numer().to_string(),
            denominator: x.denom().to_string(),
        };
        xc_certify::ExactRationalIntervalRecord {
            lower: value.clone(),
            upper: value,
        }
    };
    cert.lower_boundary.pivot_enclosures = vec![point(tiny.clone()), point(3 * tiny.clone())];
    cert.upper_boundary.pivot_enclosures = vec![point(-tiny.clone()), point(tiny)];
    let portable = PortableSelectedEigenvalueCertificate {
        schema_version: 1,
        matrix_row_major: matrix
            .iter()
            .map(|x| xc_certify::ExactRationalIntervalRecord {
                lower: xc_certify::ExactRationalRecord {
                    numerator: x.lower().numer().to_string(),
                    denominator: x.lower().denom().to_string(),
                },
                upper: xc_certify::ExactRationalRecord {
                    numerator: x.upper().numer().to_string(),
                    denominator: x.upper().denom().to_string(),
                },
            })
            .collect(),
        enclosure: cert,
    };
    let replay = verify_portable_selected_eigenvalue_certificate(&portable);
    assert!(replay.valid, "{:?}", replay.errors);
    let mut bad = portable;
    bad.enclosure.lower_boundary.count_below = 1;
    assert!(!verify_portable_selected_eigenvalue_certificate(&bad).valid);
}

#[test]
fn nontrivial_interval_box_count_agrees_with_all_exact_rational_corners() {
    let n = 3;
    let center = [
        Rational::from(1),
        Rational::from((1, 8)),
        Rational::from((1, 16)),
        Rational::from(2),
        Rational::from((1, 8)),
        Rational::from(3),
    ];
    let positions = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let eps = Rational::from((1, 32));
    let mut matrix = vec![I::point(Rational::from(0)); 9];
    for ((i, j), x) in positions.iter().zip(&center) {
        let entry = I::new(x.clone() - &eps, x.clone() + &eps).unwrap();
        matrix[i * n + j] = entry.clone();
        matrix[j * n + i] = entry;
    }
    let proof =
        certify_interval_matrix_eigenvalues_below_mpfr(&matrix, n, Rational::from((3, 2)), 128);
    let xc_certify::IntervalEigenvalueCountResult::Conclusive { certificate } = proof else {
        panic!("box should separate shift")
    };
    for mask in 0..64 {
        let mut a = vec![Rational::from(0); 9];
        for (k, ((i, j), x)) in positions.iter().zip(&center).enumerate() {
            let entry = x.clone()
                + if mask & (1 << k) == 0 {
                    -eps.clone()
                } else {
                    eps.clone()
                };
            a[i * n + j] = entry.clone();
            a[j * n + i] = entry;
        }
        for i in 0..n {
            a[i * n + i] -= Rational::from((3, 2));
        }
        let mut negative = 0;
        for k in 0..n {
            let pivot = a[k * n + k].clone();
            assert_ne!(pivot, 0);
            negative += usize::from(pivot < 0);
            for i in k + 1..n {
                for j in i..n {
                    let correction = a[i * n + k].clone() * a[j * n + k].clone() / &pivot;
                    a[i * n + j] -= correction;
                    a[j * n + i] = a[i * n + j].clone();
                }
            }
        }
        assert_eq!(
            negative, certificate.eigenvalue_count,
            "exact corner {mask}"
        );
    }
}
