#![cfg(feature = "hp")]
use xc_variational::maynard::{MkMonomialReference, MkSymmetricReference, MultiIndex};
#[test]
fn entry_indices_outside_the_declared_degree_return_an_error() {
    let r = MkMonomialReference::new(2, 1).unwrap();
    let bad = MultiIndex(vec![3, 0]);
    assert!(r.i_entry(&bad, &bad).is_err());
    assert!(r.j_entry(0, &bad, &bad).is_err());
}
#[test]
fn native_mk_actions_reject_nonfinite_inputs() {
    let r = MkMonomialReference::new(2, 1).unwrap();
    assert!(r
        .apply_i_f64(&vec![f64::NAN; r.dimension()], &mut vec![0.; r.dimension()])
        .is_err());
    let r = MkSymmetricReference::new(2, 1).unwrap();
    assert!(r
        .apply_j_total_f64(
            &vec![f64::INFINITY; r.dimension()],
            &mut vec![0.; r.dimension()]
        )
        .is_err());
}
