#![cfg(feature = "hp")]
use rug::{float::Special, Float};
use xc_spectral::ccm::certified_roots::CertifiedSecularFunction;
#[test]
fn reduced_precision_secular_sources_enclose_the_stored_function() {
    let p = 128;
    let source = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 100_i32);
    let poles = [source.clone(), Float::with_val(p, 3)];
    let weights = [Float::with_val(p, 1), Float::with_val(p, 1)];
    let state = CertifiedSecularFunction::from_point_data(&poles, &weights, 64).unwrap();
    assert!(state.poles()[0].lower() <= &source && state.poles()[0].upper() >= &source);
    // A non-singleton enclosure is not an exact point polynomial.
    assert!(state.normalized_finite_entire_function().is_err());
    assert_eq!(state.monotone_count().unwrap().certified_root_count, 1);
    for value in [Special::Nan, Special::Infinity, Special::NegInfinity] {
        assert!(CertifiedSecularFunction::from_point_data(
            &[Float::with_val(p, value), Float::with_val(p, 3)],
            &weights,
            p
        )
        .is_err());
        assert!(CertifiedSecularFunction::from_point_data(
            &poles,
            &[Float::with_val(p, value), Float::with_val(p, 1)],
            p
        )
        .is_err());
    }
    assert!(CertifiedSecularFunction::from_point_data(&poles, &weights, 0).is_err());
}
