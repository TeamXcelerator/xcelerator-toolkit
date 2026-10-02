//! Rational witnesses that independently demonstrate why staged pi rounding
//! cannot define the source.
//! Production discovery agreement is exercised in complete_discovery's tests.
#![cfg(feature = "hp")]

use rug::{float::Constant, ops::Pow, Float, Rational};
use xc_spectral::ccm::certified_roots::CertifiedSecularFunction;

#[test]
fn fresh_audit_complete_discovery_spacing_differs_from_public_certificate_source() {
    for (precision, cutoff) in [(64, 2), (128, 31), (256, 2)] {
        let weights = [1, -4, 1].map(|v| Float::with_val(precision, v));
        let source =
            CertifiedSecularFunction::from_integer_ccm_state(cutoff, 1, &weights, precision)
                .unwrap();
        let shared = source.poles()[2].lower();
        assert_eq!(shared, source.poles()[2].upper());

        // Historical operation sequence before the CCM-01 repair.
        let length = Float::with_val(precision, cutoff).ln();
        let mut staged = Float::with_val(precision, Constant::Pi);
        staged *= 2;
        staged /= &length;
        assert_ne!(
            &staged, shared,
            "audit discrepancy at p={precision}, c={cutoff}"
        );
    }
}

#[test]
fn fresh_audit_exact_height_boundary_classifies_different_roots() {
    let p = 64;
    let weights = [1, -4, 1].map(|v| Float::with_val(p, v));
    let source = CertifiedSecularFunction::from_integer_ccm_state(2, 1, &weights, p).unwrap();
    let actual_spacing = source.poles()[2].lower().to_rational().unwrap();
    let mut discovery_spacing = Float::with_val(p, Constant::Pi);
    discovery_spacing *= 2;
    discovery_spacing /= Float::with_val(p, 2).ln();
    let discovery_spacing = discovery_spacing.to_rational().unwrap();
    // A finite decimal strictly between sqrt(2)*the two exact spacing points.
    let boundary = Rational::from((
        rug::Integer::from_str_radix("128194503642625241484", 10).unwrap(),
        rug::Integer::from(10).pow(19),
    ));
    let b2 = boundary.square();
    assert!(actual_spacing.square() * 2 < b2);
    assert!(discovery_spacing.square() * 2 > b2);
}
