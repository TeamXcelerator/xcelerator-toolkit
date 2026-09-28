#![cfg(feature = "hp")]
use rug::{float::Special, Float, Rational};
use xc_numerics::mpfr_interval::{MpfrBallContext, MpfrComplexBall, MpfrInterval};

#[test]
fn invalid_arithmetic_cannot_prove_a_sign_or_empty_intersection() {
    let p = 128;
    let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 1) as u32;
    let bad = MpfrInterval::point(huge).square();
    let good = MpfrInterval::from_i64(1, p);
    assert!(bad.validate().is_err());
    assert!(bad.contains_zero());
    assert!(!bad.is_strictly_positive());
    assert!(!bad.is_subset_of(&good));
    assert!(!bad.is_interior_subset_of(&good));
    assert!(good.try_intersection(&bad).is_err());
    assert!(bad.reciprocal().is_err());
    assert!(bad.ln().is_err());
    assert!(bad.sqrt().is_err());
    assert!(good.div(&bad).is_err());
    assert!(bad.with_precision(256).is_err());
    for result in [
        bad.add(&good),
        bad.sub(&good),
        bad.mul(&good),
        bad.sin(),
        bad.cos(),
        bad.exp(),
        bad.atan(),
    ] {
        assert!(result.validate().is_err());
    }
    assert!(MpfrComplexBall::new(bad, good.clone()).is_err());
    for special in [Special::Nan, Special::Infinity, Special::NegInfinity] {
        assert!(
            MpfrComplexBall::point(Float::with_val(p, special), Float::with_val(p, 0)).is_err()
        );
        assert!(MpfrInterval::from_float(&Float::with_val(p, special), p).is_err());
    }
    assert!(MpfrBallContext::new(31).is_err());
    assert!(good.div(&MpfrInterval::from_i64(1, 64)).is_err());
}

#[test]
fn finite_midpoints_and_trigonometric_ranges_survive_extreme_exponents() {
    let p = 128;
    let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 1) as u32;
    let tiny = Float::with_val(p, 1) << (rug::float::exp_min() - 1);
    for point in [huge.clone(), -huge.clone(), tiny.clone(), -tiny.clone()] {
        assert_eq!(
            MpfrInterval::point(point.clone()).midpoint_point().lower(),
            &point
        );
    }
    // A billion-bit argument reduction is not needed to test the midpoint.
    // Large but practical exact points exercise directed transcendental output.
    let large = Float::with_val(p, 1) << 20000_i32;
    for point in [large.clone(), -large, tiny.clone(), -tiny] {
        let x = MpfrInterval::point(point.clone());
        for (enclosure, reference) in [(x.sin(), point.clone().sin()), (x.cos(), point.cos())] {
            enclosure.validate().unwrap();
            assert!(enclosure.lower() <= &reference && &reference <= enclosure.upper());
            assert!(enclosure.lower() >= &-1 && enclosure.upper() <= &1);
        }
    }
    for (a, b) in [
        (Float::with_val(p, &huge / 2), huge.clone()),
        (-huge.clone(), huge),
    ] {
        let x = MpfrInterval::new(a.clone(), b.clone()).unwrap();
        let midpoint = x.midpoint_point();
        assert!(midpoint.lower() >= &a && midpoint.upper() <= &b);
    }
}

#[test]
fn trigonometric_enclosures_contain_values_at_rational_multiples_of_pi() {
    // Radius widening once rounded to nearest, so cos(pi/36) at 128 bits and
    // cos(6*pi/79) at 64 bits excluded the true value by about one ulp.
    // References use exact j/n at 2048 bits, far beyond every tested width.
    let reference_pi = Float::with_val(2048, rug::float::Constant::Pi);
    for p in [64, 128, 192] {
        let pi = MpfrInterval::pi(p);
        for n in 2..=80_i64 {
            for j in 1..n {
                let x = pi
                    .mul(&MpfrInterval::from_i64(j, p))
                    .div(&MpfrInterval::from_i64(n, p))
                    .unwrap();
                let angle = Float::with_val(2048, &reference_pi * j) / n;
                for (enclosure, reference) in
                    [(x.sin(), angle.clone().sin()), (x.cos(), angle.cos())]
                {
                    enclosure.validate().unwrap();
                    assert!(
                        enclosure.lower() <= &reference && &reference <= enclosure.upper(),
                        "p={p}, angle={j}*pi/{n}"
                    );
                }
            }
        }
    }
}

#[test]
fn precision_changes_preserve_exact_stored_values() {
    let source = Float::with_val(160, 1) + (Float::with_val(160, 1) >> 100_i32);
    for p in [1, 2, 32, 64, 128, 256] {
        for source in [source.clone(), -source.clone()] {
            let enclosure = MpfrInterval::from_float(&source, p).unwrap();
            let exact = source.to_rational().unwrap();
            assert!(enclosure.to_rational_interval().contains(&exact));
        }
    }
}

#[test]
fn interval_arithmetic_encloses_exact_rational_corner_values() {
    // Independent exact point arithmetic checks all corners, not rounded
    // midpoint agreement. Includes negative, positive, and straddling intervals.
    for p in [2, 8, 32, 64, 128] {
        for k in -12..12 {
            let a = Rational::from((k, 7));
            let b = Rational::from((k + 3, 7));
            let c = Rational::from((2 * k - 1, 11));
            let d = Rational::from((2 * k + 4, 11));
            let x = MpfrInterval::new(
                MpfrInterval::from_rational(&a, p).lower().clone(),
                MpfrInterval::from_rational(&b, p).upper().clone(),
            )
            .unwrap();
            let y = MpfrInterval::new(
                MpfrInterval::from_rational(&c, p).lower().clone(),
                MpfrInterval::from_rational(&d, p).upper().clone(),
            )
            .unwrap();
            for left in [&a, &b] {
                for right in [&c, &d] {
                    for (actual, exact) in [
                        (x.add(&y), left.clone() + right),
                        (x.sub(&y), left.clone() - right),
                        (x.mul(&y), left.clone() * right),
                    ] {
                        assert!(actual.to_rational_interval().contains(&exact));
                    }
                    if !y.contains_zero() {
                        assert!(x
                            .div(&y)
                            .unwrap()
                            .to_rational_interval()
                            .contains(&(left.clone() / right)));
                    }
                }
            }
            let square = x.square().to_rational_interval();
            for t in [&a, &b] {
                assert!(square.contains(&(t.clone() * t)));
            }
            if a <= 0 && b >= 0 {
                assert!(square.contains(&Rational::from(0)));
            }
        }
    }
}
