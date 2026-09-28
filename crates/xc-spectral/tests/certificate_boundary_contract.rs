#![cfg(feature = "hp")]
use rug::Float;
use xc_root::{IntervalRootCertificate, IntervalRootStatus};
use xc_spectral::ccm::certified_roots::{
    reconcile_complete_window, ReferenceZeroDataset, SecularCountCertificate,
};

fn root(lower: &Float, upper: &Float) -> IntervalRootCertificate {
    IntervalRootCertificate {
        schema_version: 1,
        endpoint_encoding: xc_root::IntervalRootEndpointEncoding::StoredBinaryNearestV1,
        uniqueness_witness: None,
        lower: lower.to_string_radix(10, Some(100)),
        upper: upper.to_string_radix(10, Some(100)),
        precision_bits: lower.prec(),
        iterations: 1,
        status: IntervalRootStatus::CertifiedUnique,
        uniqueness_witnessed: true,
        reason: "synthetic structural fixture; no source truth asserted".into(),
    }
}
fn count(n: usize) -> SecularCountCertificate {
    SecularCountCertificate {
        pole_count: n + 1,
        open_intervals_counted: n,
        certified_root_count: n,
        residue_sign: "positive".into(),
        method: "synthetic count premise".into(),
        square_free: true,
    }
}
#[test]
fn reference_dataset_rejects_nonfinite_ordinates() {
    for text in ["NaN", "inf", "+inf", "1e1000000000"] {
        let result = ReferenceZeroDataset::new("probe", "synthetic", "1", 128, vec![text.into()]);
        assert!(result.is_err(), "accepted {text}: {result:?}");
    }
}
#[test]
fn reconciliation_rejects_invalid_enclosures() {
    for (lo, hi) in [("NaN", "NaN"), ("1", "inf"), ("2", "1")] {
        let mut r = root(&Float::with_val(128, 1), &Float::with_val(128, 2));
        r.lower = lo.into();
        r.upper = hi.into();
        let result = reconcile_complete_window(&[r], &count(1));
        assert!(result.is_err(), "accepted [{lo},{hi}]: {result:?}");
    }
}
#[test]
fn reconciliation_decodes_each_bound_at_its_retained_precision() {
    let mut exercised = 0;
    for k in 2..40 {
        let shared = Float::with_val(80, k).sqrt();
        let text = shared.to_string_radix(10, Some(32));
        let reread = Float::with_val(256, Float::parse(&text).unwrap());
        if reread >= shared {
            continue;
        }
        exercised += 1;
        let mut left = root(&Float::with_val(80, &shared - 1), &shared);
        left.upper = text;
        let right = root(
            &Float::with_val(256, &shared),
            &Float::with_val(256, &shared + 1),
        );
        let result = reconcile_complete_window(&[left, right], &count(2)).unwrap();
        assert!(
            !result.ordered_and_disjoint && !result.complete,
            "touching exact stored bounds were separated by re-parsing: {result:?}"
        );
    }
    assert!(exercised > 0);
}

#[test]
fn certificate_readers_reject_unsupported_precision_without_panicking() {
    for precision in [0, 31, 1_000_001, u32::MAX] {
        assert!(
            ReferenceZeroDataset::new("probe", "synthetic", "1", precision, vec!["1".into()])
                .is_err()
        );
        let mut r = root(&Float::with_val(128, 1), &Float::with_val(128, 2));
        r.precision_bits = precision;
        assert!(reconcile_complete_window(&[r], &count(1)).is_err());
    }
}
#[test]
fn reference_comparison_keeps_exact_stored_endpoint_membership_and_versions_replay() {
    use xc_core::DecimalLiteral;
    use xc_root::IntervalNewtonOptions;
    use xc_spectral::ccm::certified_roots::{
        certify_first_positive_ccm_roots, compare_first_positive_roots_to_references,
        verify_post_discovery_reference_comparison, FirstPositiveCcmRootOptions,
    };
    let p = 128;
    let options = FirstPositiveCcmRootOptions {
        integer_cutoff_c: 13,
        modes: 3,
        requested_roots: 3,
        precision_bits: p,
        pole_interval_bisection_steps: 16,
        interval_newton: IntervalNewtonOptions {
            width_tolerance: DecimalLiteral::new("1e-18").unwrap(),
            maximum_iterations: 30,
        },
    };
    let certificate =
        certify_first_positive_ccm_roots(&vec![Float::with_val(p, 1); 7], &options).unwrap();
    for upper in [false, true] {
        let ordinates = certificate
            .roots
            .iter()
            .map(|r| {
                let text = if upper { &r.upper } else { &r.lower };
                Float::with_val(r.precision_bits, Float::parse(text).unwrap())
                    .to_string_radix(10, Some(200))
            })
            .collect();
        let refs = ReferenceZeroDataset::new(
            "exact-endpoints",
            "synthetic endpoint controls",
            "1",
            512,
            ordinates,
        )
        .unwrap();
        let comparison = compare_first_positive_roots_to_references(&certificate, &refs).unwrap();
        assert_eq!(comparison.schema_version, 2);
        assert!(comparison
            .records
            .iter()
            .all(|r| r.reference_inside_enclosure));
        verify_post_discovery_reference_comparison(&comparison, &certificate, &refs).unwrap();
        let mut legacy = comparison;
        legacy.schema_version = 1;
        assert!(verify_post_discovery_reference_comparison(&legacy, &certificate, &refs).is_err());
    }
}
