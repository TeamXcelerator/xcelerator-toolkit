#![cfg(feature = "hp")]
use rug::Float;
use xc_spectral::ccm::{
    convergence::{analyze_prime_floor, PrimeFloorPoint},
    research::{analyze_nested_schur_hp, analyze_root_transfer_hp},
};

#[test]
fn root_transfer_rejects_unrepresentable_derivative_instead_of_zero_step() {
    let p = 128;
    let root = Float::with_val(p, 1) >> 600_000_000u32;
    let result = analyze_root_transfer_hp(
        &[Float::with_val(p, 1)],
        &[Float::with_val(p, 0)],
        &root,
        None,
        p,
    );
    assert!(
        result.is_err(),
        "unrepresentable derivative produced an accepted report: {result:?}"
    );
}

#[test]
fn prime_floor_rejects_overflowing_modulation() {
    let point = PrimeFloorPoint {
        cutoff: "12".into(),
        cutoff_is_prime: false,
        prime_power_count: 8,
        largest_prime_power: 11,
        smooth_leading_digits: -f64::MAX,
        measured_digits: f64::MAX,
        arithmetic_modulation_digits: 0.0,
    };
    assert!(analyze_prime_floor(vec![point.clone(), point]).is_err());
}

#[test]
fn analytic_schur_and_root_transfer_values() {
    let p = 128;
    let f = |x| Float::with_val(p, x);
    // [[2,1],[1,3]] has Schur complement 3-1/2=5/2.
    let schur =
        analyze_nested_schur_hp(&[f(2)], &[f(2), f(1), f(1), f(3)], 1, &f(0), &f(0), p).unwrap();
    assert_eq!(
        Float::with_val(p, Float::parse(&schur.schur_complement).unwrap()),
        Float::with_val(p, 2.5)
    );
    assert!(schur.prefix_within_tolerance);
    // F=1/r, F'=-1/r^2, so -F/F'=r at r=2.
    let report = analyze_root_transfer_hp(&[f(1)], &[f(0)], &f(2), None, p).unwrap();
    assert_eq!(
        Float::with_val(p, Float::parse(&report.predicted_displacement).unwrap()),
        f(2)
    );
}
