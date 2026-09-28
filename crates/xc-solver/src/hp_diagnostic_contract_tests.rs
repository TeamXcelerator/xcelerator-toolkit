use super::*;
use rug::Float;

#[test]
fn finite_singleton_norm_survives_exponent_extremes() {
    let p = 256;
    for exponent in [
        rug::float::exp_min() / 2 - 100,
        rug::float::exp_max() / 2 + 100,
    ] {
        let value: Float = Float::with_val(p, 1) << exponent;
        assert!(value.is_finite() && !value.is_zero());
        assert_eq!(hp_norm(std::slice::from_ref(&value), p), value);
    }
}

#[test]
fn residual_diagnostics_reject_overflow_and_nonzero_ratio_underflow() {
    let p = 256;
    let z = Float::with_val(p, 0);
    let one = Float::with_val(p, 1);
    let enormous: Float = one.clone() << (rug::float::exp_max() - 2);
    let overflow = hp_residual_measures(
        std::slice::from_ref(&one),
        std::slice::from_ref(&one),
        std::slice::from_ref(&enormous),
        &enormous,
        p,
    );
    assert!(overflow.is_err());
    let tiny: Float = one.clone() << (rug::float::exp_min() + 2);
    assert!(hp_residual_measures(
        &[tiny],
        std::slice::from_ref(&enormous),
        std::slice::from_ref(&one),
        &enormous,
        p
    )
    .is_err());
    assert_eq!(
        hp_residual_measures(
            std::slice::from_ref(&z),
            std::slice::from_ref(&z),
            &[one],
            &z,
            p
        )
        .unwrap()
        .1,
        z
    );
}

#[test]
fn generalized_projected_diagonal_and_repeated_roots_are_preserved() {
    let p = 256;
    let z = Float::with_val(p, 0);
    let one = Float::with_val(p, 1);
    for exponent in [-1000, -400, 0, 400, 1000] {
        let small: Float = one.clone() << exponent;
        for largest in [false, true] {
            let (value, x, y) = hp_generalized::projected_generalized_extreme_2x2(
                &small, &z, &one, &one, &z, &one, largest, p,
            )
            .unwrap();
            let expected = if largest {
                small.clone().max(&one)
            } else {
                small.clone().min(&one)
            };
            assert_eq!(value, expected);
            assert!(!x.is_zero() || !y.is_zero());
        }
    }
    // A=B has the exactly repeated generalized eigenvalue 1.
    let two = Float::with_val(p, 2);
    let three = Float::with_val(p, 3);
    for largest in [false, true] {
        let (value, x, y) = hp_generalized::projected_generalized_extreme_2x2(
            &two, &one, &three, &two, &one, &three, largest, p,
        )
        .unwrap();
        assert!((value - one.clone()).abs() < (one.clone() >> 200u32));
        let bx = Float::with_val(p, &two * &x) + &y;
        let by = Float::with_val(p, &three * &y) + &x;
        let norm = Float::with_val(p, &x * bx) + Float::with_val(p, &y * by);
        assert!((norm - one.clone()).abs() < (one.clone() >> 200u32));
    }
}

#[test]
fn generalized_projection_matches_exact_congruence_pencils() {
    // For S=[[1,k],[1,k+1]], det S=1. Thus
    // A=S^T diag(d0,d1) S, B=S^T S have exactly eigenvalues d0,d1.
    // The stored dyadic matrices below are exact at 768 bits; the 400-bit
    // spectral separation therefore does not disappear in input rounding.
    let p = 768;
    let one = Float::with_val(p, 1);
    let spectra = [
        (Float::with_val(p, -3), one.clone()),
        (Float::with_val(p, 0), Float::with_val(p, 5)),
        (one.clone(), Float::with_val(p, 2)),
        (one.clone(), one.clone()),
        (one.clone() >> 400u32, one.clone()),
        (Float::with_val(p, -5), Float::with_val(p, -2)),
    ];
    let tolerance = one.clone() >> 600u32;
    for k in -8i32..=8 {
        let b00 = Float::with_val(p, 2);
        let b01 = Float::with_val(p, 2 * k + 1);
        let b11 = Float::with_val(p, k * k + (k + 1) * (k + 1));
        for (d0, d1) in &spectra {
            for exponent in [-20000i32, -400, 0, 400, 20000] {
                let scale: Float = one.clone() << exponent;
                let a00 = Float::with_val(p, d0 + d1) * &scale;
                let a01 = (Float::with_val(p, d0 * k) + Float::with_val(p, d1 * (k + 1))) * &scale;
                let a11 = (Float::with_val(p, d0 * (k * k))
                    + Float::with_val(p, d1 * ((k + 1) * (k + 1))))
                    * &scale;
                for largest in [false, true] {
                    let (value, x, y) = hp_generalized::projected_generalized_extreme_2x2(
                        &a00, &a01, &a11, &b00, &b01, &b11, largest, p,
                    )
                    .unwrap();
                    let expected = if largest { d1 } else { d0 };
                    assert!(
                        (Float::with_val(p, &value / &scale) - expected).abs() < tolerance,
                        "k={k}, exponent={exponent}, largest={largest}"
                    );
                    // S x has Euclidean norm one, independently of the solver's
                    // metric normalization calculation.
                    let sx0 = x.clone() + Float::with_val(p, &y * k);
                    let sx1 = x.clone() + Float::with_val(p, &y * (k + 1));
                    let length = Float::with_val(p, &sx0 * &sx0) + Float::with_val(p, &sx1 * &sx1);
                    assert!((length - 1u32).abs() < tolerance);
                    if d0 != d1 {
                        let leakage = if largest { sx0 } else { sx1 };
                        // A tiny gap is absent in these distinct-root cases;
                        // the known S^-1 eigenvectors give an independent check.
                        assert!(leakage.abs() < tolerance);
                    }
                }
            }
        }
    }
}

#[test]
fn acceptance_and_cluster_thresholds_use_opposite_conservative_directions() {
    let below =
        xc_core::DecimalLiteral::new("0.999999999999999999999999999999999999999999").unwrap();
    let above =
        xc_core::DecimalLiteral::new("1.000000000000000000000000000000000000000001").unwrap();
    assert!(hp_positive_threshold(&below, 64, "acceptance", rug::float::Round::Down).unwrap() < 1);
    assert!(hp_positive_threshold(&above, 64, "cluster", rug::float::Round::Up).unwrap() > 1);
    // Shifts remain nearest-rounded point parameters; bounds do not change them.
    assert_eq!(hp_parse_literal(&below, 64).unwrap(), 1);
    assert_eq!(hp_parse_literal(&above, 64).unwrap(), 1);
}
