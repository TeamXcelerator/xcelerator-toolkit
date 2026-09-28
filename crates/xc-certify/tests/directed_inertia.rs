#![cfg(feature = "hp")]

use rug::Rational;
use std::collections::BTreeMap;
use xc_cache::ContentDigest;
use xc_certify::exact::{
    build_portable_interval_inertia_certificate, build_portable_interval_inertia_certificate_mpfr,
    interval_record, interval_symmetric_ldlt_inertia_mpfr,
    verify_portable_interval_inertia_certificate, IntervalInertiaResult,
};
use xc_certify::PortableIntervalInertiaCertificate;
use xc_numerics::interval::RationalInterval;

fn matrix(values: &[i32]) -> Vec<RationalInterval> {
    values
        .iter()
        .map(|&v| RationalInterval::point(Rational::from(v)))
        .collect()
}

fn proof(values: &[RationalInterval], n: usize, p: u32) -> PortableIntervalInertiaCertificate {
    build_portable_interval_inertia_certificate_mpfr(
        values,
        n,
        p,
        "independent-test-fixture",
        ContentDigest::sha256(b"fixture"),
        BTreeMap::from([("fixture".into(), "independent".into())]),
        vec![],
    )
    .unwrap()
}

#[test]
fn directed_inertia_preserves_known_spectra_across_scales_and_uncertainty() {
    // Spectra are {1,3}, {-3,-1}, {-1,1}, and {-1,-1,2}.
    // Entry errors <= 1/100 give ||E||_2 <= n/100 < min |lambda|,
    // so every symmetric member of each interval family has the same inertia.
    for (values, n, counts) in [
        (vec![2, 1, 1, 2], 2, (2, 0)),
        (vec![-2, -1, -1, -2], 2, (0, 2)),
        (vec![0, 1, 1, 0], 2, (1, 1)),
        (vec![0, 1, 1, 1, 0, 1, 1, 1, 0], 3, (1, 2)),
    ] {
        for uncertain in [false, true] {
            for p in [64, 128, 256] {
                for shift in [-5000, 0, 5000] {
                    let radius = if uncertain {
                        Rational::from((1, 100))
                    } else {
                        Rational::from(0)
                    };
                    let entries = values
                        .iter()
                        .map(|&v| {
                            RationalInterval::new(
                                (Rational::from(v) - &radius) << shift,
                                (Rational::from(v) + &radius) << shift,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let IntervalInertiaResult::Conclusive {
                        positive,
                        negative,
                        pivot_enclosures,
                    } = interval_symmetric_ldlt_inertia_mpfr(&entries, n, p).unwrap()
                    else {
                        panic!("known nonsingular family remained unresolved");
                    };
                    assert_eq!((positive, negative), counts);
                    assert_eq!(pivot_enclosures.len(), n);
                    assert_eq!(
                        pivot_enclosures
                            .iter()
                            .filter(|v| v.is_strictly_positive())
                            .count(),
                        positive
                    );
                    assert_eq!(
                        pivot_enclosures
                            .iter()
                            .filter(|v| v.is_strictly_negative())
                            .count(),
                        negative
                    );
                }
            }
        }
    }
}

#[test]
fn singular_and_unresolved_signs_never_become_positive_proofs() {
    for entries in [
        matrix(&[1, 1, 1, 1]),
        vec![RationalInterval::new((-1).into(), 1.into()).unwrap(); 4],
    ] {
        for p in [32, 128, 256] {
            let result = proof(&entries, 2, p);
            assert!(result.zero_or_unresolved > 0);
            assert_eq!(
                result.positive + result.negative + result.zero_or_unresolved,
                2
            );
            assert!(verify_portable_interval_inertia_certificate(&result).valid);
        }
    }
    let mut near = matrix(&[1, 1, 1, 1]);
    near[3] = RationalInterval::point(Rational::from(1) + (Rational::from(1) >> 200));
    assert!(proof(&near, 2, 32).zero_or_unresolved > 0);
    let resolved = proof(&near, 2, 256);
    assert_eq!(
        (
            resolved.positive,
            resolved.negative,
            resolved.zero_or_unresolved
        ),
        (2, 0, 0)
    );
}

#[test]
fn exact_current_and_historical_schema_and_directed_schema_have_distinct_replay() {
    let entries = matrix(&[3, 1, 1, 2]);
    let current = build_portable_interval_inertia_certificate(
        &entries,
        2,
        128,
        "independent-test-fixture",
        ContentDigest::sha256(b"fixture"),
        BTreeMap::from([("fixture".into(), "independent".into())]),
        vec![],
    )
    .unwrap();
    assert_eq!(current.schema_version, 3);
    // This positive 1x1-pivot example has unchanged exact historical evidence.
    let mut legacy = current.clone();
    legacy.schema_version = 1;
    legacy.refresh_certificate_id().unwrap();
    assert_eq!(
        legacy.pivot_enclosures[1],
        interval_record(&RationalInterval::point(Rational::from((5, 3))))
    );
    let directed = proof(&entries, 2, 128);
    assert_eq!(directed.schema_version, 2);
    assert_ne!(directed.pivot_enclosures[1], legacy.pivot_enclosures[1]);
    for certificate in [current, legacy, directed] {
        let decoded = serde_json::from_slice(&serde_json::to_vec(&certificate).unwrap()).unwrap();
        assert!(verify_portable_interval_inertia_certificate(&decoded).mathematical_claim_verified);
    }
}

#[test]
fn refreshed_hashes_cannot_hide_arithmetic_or_schema_tampering() {
    let good = proof(&matrix(&[3, 1, 1, 2]), 2, 128);
    for mutation in 0..10 {
        let mut changed = good.clone();
        match mutation {
            0 => {
                changed.positive = 1;
                changed.negative = 1;
            }
            1 => changed.schema_version = 1,
            2 => changed.schema_version = 3,
            3 => changed.precision_bits = 64,
            4 => changed.precision_bits = 0,
            5 => changed.precision_bits = 1_000_001,
            6 => changed.matrix_row_major[0] = interval_record(&matrix(&[4])[0]),
            7 => changed.matrix_row_major[1] = interval_record(&matrix(&[2])[0]),
            8 => changed.pivot_enclosures[1] = interval_record(&matrix(&[2])[0]),
            _ => {
                changed.pivot_enclosures.pop();
            }
        }
        changed.matrix_digest = changed.computed_matrix_digest().unwrap();
        changed.refresh_certificate_id().unwrap();
        assert!(
            !verify_portable_interval_inertia_certificate(&changed).valid,
            "mutation {mutation}"
        );
    }
    let mut stale = good;
    stale.certificate_id = ContentDigest::sha256(b"stale");
    assert!(!verify_portable_interval_inertia_certificate(&stale).valid);
}

#[test]
fn directed_inertia_rejects_invalid_dimensions_precision_and_symmetry() {
    for p in [0, 31, 1_000_001] {
        assert!(interval_symmetric_ldlt_inertia_mpfr(&matrix(&[1]), 1, p).is_err());
    }
    for (entries, n) in [
        (vec![], 0),
        (matrix(&[1]), 2),
        (matrix(&[1]), usize::MAX),
        (matrix(&[1, 1, 0, 1]), 2),
    ] {
        assert!(interval_symmetric_ldlt_inertia_mpfr(&entries, n, 128).is_err());
    }
}
