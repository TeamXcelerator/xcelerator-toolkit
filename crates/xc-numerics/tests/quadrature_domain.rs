use xc_numerics::quadrature::{
    gauss_legendre_npt_f64, try_gauss_legendre_npt_f64, try_gl_nodes_weights_f64,
};
#[test]
fn finite_extreme_bounds_do_not_overflow_the_affine_mapping() {
    let actual = try_gauss_legendre_npt_f64(|x| 1.0 / x, 1e308, 1.1e308, 16).unwrap();
    assert!((actual - (1.1_f64).ln()).abs() < 1e-14);
    let actual = try_gauss_legendre_npt_f64(|_| 1e-308, -1e308, 1e308, 16).unwrap();
    assert!((actual - 2.0).abs() < 1e-14);
    assert!(
        (try_gauss_legendre_npt_f64(|_| 1e-308, 1e308, -1e308, 16).unwrap() + 2.0).abs() < 1e-14
    );
}
#[test]
fn quadrature_domain_failures_are_explicit() {
    assert!(try_gl_nodes_weights_f64(0).is_err());
    assert!(try_gl_nodes_weights_f64(usize::MAX).is_err());
    assert!(gauss_legendre_npt_f64(|_| 1.0, 0.0, 1.0, 0).is_nan());
    assert!(try_gauss_legendre_npt_f64(|_| f64::NAN, 0.0, 1.0, 8).is_err());
    assert!(try_gauss_legendre_npt_f64(|_| 1.0, 0.0, f64::INFINITY, 8).is_err());
}
#[test]
fn gauss_legendre_integrates_all_monomials_through_its_exactness_degree() {
    for n in [1, 2, 3, 4, 8, 16, 32] {
        for power in 0..2 * n {
            let actual =
                try_gauss_legendre_npt_f64(|x| x.powi(power as i32), -1.0, 1.0, n).unwrap();
            let expected = if power % 2 == 0 {
                2.0 / (power + 1) as f64
            } else {
                0.0
            };
            assert!(
                (actual - expected).abs() < 2e-13,
                "n={n}, power={power}, actual={actual}"
            );
        }
    }
}

#[test]
fn an_underflowed_weighted_sample_cannot_silently_erase_a_finite_integral() {
    // The analytic integral of the constant is 2*bound*sample. Multiply
    // sample by bound first so this independent oracle stays representable.
    // The original least-subnormal / 1e308 / order16 case must now succeed.
    for sample_bits in [1, 2, 7] {
        for bound in [1e280, 1e308] {
            for n in [8, 16, 32] {
                let sample = f64::from_bits(sample_bits);
                let expected = 2.0 * (sample * bound);
                let actual = try_gauss_legendre_npt_f64(|_| sample, -bound, bound, n).unwrap();
                assert!(actual > 0.0);
                assert!((actual - expected).abs() <= expected * (64.0 * f64::EPSILON));
                let reverse = try_gauss_legendre_npt_f64(|_| sample, bound, -bound, n).unwrap();
                assert_eq!(reverse, -actual);
            }
        }
    }
}
