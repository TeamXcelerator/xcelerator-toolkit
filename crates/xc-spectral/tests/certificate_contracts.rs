#![cfg(feature = "hp")]

use rug::Float;
use xc_root::IntervalNewtonOptions;
use xc_spectral::ccm::certified_roots::*;

fn options() -> IntervalNewtonOptions {
    IntervalNewtonOptions {
        maximum_iterations: 80,
        width_tolerance: xc_core::DecimalLiteral::new("1e-20").unwrap(),
    }
}

const OLD_REASON: &str = "every monotonic pole interval has one disjoint certified unique root";

#[test]
fn legacy_certificate_annotation_preserves_math_checks() {
    let p = 128;
    let mut certificate = certify_first_positive_ccm_roots(
        &vec![Float::with_val(p, 1); 7],
        &FirstPositiveCcmRootOptions {
            integer_cutoff_c: 13,
            modes: 3,
            requested_roots: 3,
            precision_bits: p,
            pole_interval_bisection_steps: 16,
            interval_newton: options(),
        },
    )
    .unwrap();
    verify_first_positive_ccm_root_certificate(&certificate).unwrap();
    for root in &certificate.roots {
        let lower = Float::with_val(p, Float::parse(&root.lower).unwrap());
        let upper = Float::with_val(p, Float::parse(&root.upper).unwrap());
        assert!(
            Float::with_val_round(p, &upper - &lower, rug::float::Round::Up).0
                <= options().validate(p).unwrap()
        );
    }
    certificate.reconciliation.reason = OLD_REASON.into();
    verify_first_positive_ccm_root_certificate(&certificate).unwrap();
    let legacy = certificate.clone();
    certificate.reconciliation.independently_counted_roots += 1;
    assert!(verify_first_positive_ccm_root_certificate(&certificate).is_err());
    certificate = legacy.clone();
    certificate.roots[0].lower = "0".into();
    assert!(verify_first_positive_ccm_root_certificate(&certificate).is_err());
    certificate = legacy.clone();
    // The historical reconciliation annotation does not alter the current
    // outward-endpoint root schema or its stored uniqueness witness. Replacing
    // its enclosure by a point must fail that witness replay.
    assert_eq!(certificate.roots[0].schema_version, 2);
    assert_eq!(
        certificate.roots[0].endpoint_encoding,
        xc_root::IntervalRootEndpointEncoding::OutwardDecimalV2
    );
    certificate.roots[0].upper = certificate.roots[0].lower.clone();
    let error = verify_first_positive_ccm_root_certificate(&certificate).unwrap_err();
    assert!(error
        .to_string()
        .contains("finite secular root witness does not replay"));
    certificate = legacy;
    certificate.reconciliation.reason = "unrecognized assertion".into();
    assert!(verify_first_positive_ccm_root_certificate(&certificate).is_err());
}

#[test]
fn pole_gap_rejects_three_root_mixed_sign_witness() {
    let p = 128;
    let poles = [0, 1, 2, 3].map(|x| Float::with_val(p, x));
    // Numerator is exactly 3(x-19/16)(x-21/16)(x-15/8).
    let weights = [5985, -315, -429, 7047].map(|x| Float::with_val(p, x) / 4096u32);
    let source = CertifiedSecularFunction::from_point_data(&poles, &weights, p).unwrap();
    assert_eq!(
        source
            .exact_numerator_count_between_poles(1, 2)
            .unwrap()
            .certified_root_count,
        3
    );
    assert!(source.certify_pole_interval(1, 16, &options()).is_err());
    for sign in [-1, 1] {
        let weights = [sign; 4].map(|x| Float::with_val(p, x));
        let source = CertifiedSecularFunction::from_point_data(&poles, &weights, p).unwrap();
        assert!(
            source
                .certify_pole_interval(1, 16, &options())
                .unwrap()
                .uniqueness_witnessed
        );
    }
}

#[cfg(feature = "arb")]
#[test]
fn production_certificate_legacy_annotations() {
    let p = 128;
    let weights = vec![Float::with_val(p, 1); 7];
    let mut prefix =
        certify_production_first_positive_ccm_roots(&weights, 13, 3, 2, p, 48, &options()).unwrap();
    prefix.window.reconciliation.reason = OLD_REASON.into();
    verify_production_first_positive_ccm_root_certificate(&prefix).unwrap();
    prefix.window.reconciliation.ordered_and_disjoint = false;
    assert!(verify_production_first_positive_ccm_root_certificate(&prefix).is_err());
    let mut independent = certify_production_independent_ccm_roots(
        &weights,
        13,
        3,
        &IndependentCcmRootTarget::Prefix { count: 2 },
        p,
        48,
        &options(),
    )
    .unwrap();
    independent.window.reconciliation.reason = OLD_REASON.into();
    verify_production_independent_ccm_root_certificate(&independent).unwrap();
    independent.selected_root_count += 1;
    assert!(verify_production_independent_ccm_root_certificate(&independent).is_err());
}
