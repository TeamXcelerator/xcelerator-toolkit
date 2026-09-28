#![cfg(feature = "hp")]
use rug::Float;
use xc_numerics::linalg::{inverse_iteration_detailed, rayleigh_quotient};

#[test]
fn quadratic_form_multiplies_exact_low_precision_storage_at_requested_precision() {
    let p = 128;
    let a = [1, 0, 0, 1].map(|x| Float::with_val(32, x));
    let x = [Float::with_val(p, 1) / 3u32, Float::with_val(p, 2) / 3u32];
    let expected = Float::with_val(256, &x[0] * &x[0]) + Float::with_val(256, &x[1] * &x[1]);
    let actual = rayleigh_quotient(&a, 2, &x, p);
    let error = Float::with_val(256, actual - expected).abs();
    assert!(error < Float::with_val(256, 1) >> 120u32, "error={error}");
}

#[test]
fn inverse_iteration_promotes_exact_stored_matrix_before_arithmetic() {
    let p = 160;
    // Exact eigenvalues (5 +- sqrt(5))/2, independent of a storage precision
    // capable of representing the four integer entries.
    let a = [3, 1, 1, 2].map(|x| Float::with_val(32, x));
    let expected = (Float::with_val(256, 5) - Float::with_val(256, 5).sqrt()) / 2u32;
    let result = inverse_iteration_detailed(&a, 2, p, 160, false).unwrap();
    let error = Float::with_val(256, result.eigenvalue - expected).abs();
    assert!(error < Float::with_val(256, 1) >> 120u32, "error={error}");
    assert!(result.diagnostics.final_relative_residual_norm < Float::with_val(p, 1) >> 110u32);
}

#[test]
fn retained_low_precision_factors_cannot_be_padded_to_a_higher_precision_solve() {
    use xc_numerics::linalg::{inverse_iteration_from_factors_detailed, lu_factor};
    let a = [3, 1, 1, 2].map(|x| Float::with_val(32, x));
    let factors = lu_factor(&a, 2).unwrap();
    assert!(
        inverse_iteration_from_factors_detailed(&a, &factors, 2, 160, 160, false, None).is_err()
    );
    let high = a.map(|x| Float::with_val(160, x));
    let factors = lu_factor(&high, 2).unwrap();
    assert!(
        inverse_iteration_from_factors_detailed(&high, &factors, 2, 160, 160, false, None).is_ok()
    );
}
