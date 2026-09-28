#![cfg(feature = "hp")]
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval;

#[test]
fn trigonometric_widening_rounds_outward_after_the_radius_operation() {
    for p in [4, 8, 16, 32, 64] {
        for numerator in -200..=200 {
            let lo = Float::with_val(p, numerator) / 37u32;
            let mut hi = lo.clone();
            for _ in 0..3 {
                hi.next_up();
            }
            let interval = MpfrInterval::new(lo.clone(), hi.clone()).unwrap();
            let midpoint = interval.midpoint_point().lower().clone();
            let r1 = Float::with_val_round(p, &midpoint - &lo, Round::Up).0;
            let r2 = Float::with_val_round(p, &hi - &midpoint, Round::Up).0;
            let radius = if r1 >= r2 { r1 } else { r2 };
            if radius >= 2 {
                continue;
            }
            for sine in [false, true] {
                let mut low = midpoint.clone();
                let mut high = midpoint.clone();
                if sine {
                    low.sin_round(Round::Down);
                    high.sin_round(Round::Up);
                } else {
                    low.cos_round(Round::Down);
                    high.cos_round(Round::Up);
                }
                let mut expected_low = Float::with_val_round(p, &low - &radius, Round::Down).0;
                let mut expected_high = Float::with_val_round(p, &high + &radius, Round::Up).0;
                if expected_low < -1 {
                    expected_low = Float::with_val(p, -1);
                }
                if expected_high > 1 {
                    expected_high = Float::with_val(p, 1);
                }
                let actual = if sine { interval.sin() } else { interval.cos() };
                assert!(actual.lower() <= &expected_low && actual.upper() >= &expected_high,
                    "inward Lipschitz widening: p={p}, numerator={numerator}, sine={sine}, interval={interval:?}, actual={actual:?}, expected=[{expected_low}, {expected_high}]");
            }
        }
    }
}

#[test]
fn trigonometric_intervals_contain_high_precision_endpoint_evaluations() {
    for p in [2, 3, 4, 8, 16, 32, 64] {
        for numerator in -512..=512 {
            let lo = Float::with_val(p, numerator) / 113u32;
            for count in [1, 3, 9] {
                let mut hi = lo.clone();
                for _ in 0..count {
                    hi.next_up();
                }
                let interval = MpfrInterval::new(lo.clone(), hi.clone()).unwrap();
                for sine in [false, true] {
                    let actual = if sine { interval.sin() } else { interval.cos() };
                    for point in [&lo, &hi] {
                        let mut low = Float::with_val(256, point);
                        let mut high = low.clone();
                        if sine {
                            low.sin_round(Round::Down);
                            high.sin_round(Round::Up);
                        } else {
                            low.cos_round(Round::Down);
                            high.cos_round(Round::Up);
                        }
                        assert!(actual.lower() <= &low && actual.upper() >= &high,
                            "endpoint excluded: p={p}, numerator={numerator}, count={count}, sine={sine}, interval={interval:?}, actual={actual:?}, reference=[{low}, {high}]");
                    }
                }
            }
        }
    }
}
#[test]
fn trigonometric_intervals_enclose_endpoints_near_zeros_at_research_precisions() {
    use rug::float::Constant;
    for p in [32, 53, 64, 96, 256] {
        for multiple in -8..=8 {
            let center = Float::with_val(512, Constant::Pi) * multiple / 2u32;
            for offset in -8i32..=8 {
                let mut lo = Float::with_val(p, &center);
                for _ in 0..offset.unsigned_abs() {
                    if offset < 0 {
                        lo.next_down();
                    } else {
                        lo.next_up();
                    }
                }
                for count in [1, 2, 3, 9] {
                    let mut hi = lo.clone();
                    for _ in 0..count {
                        hi.next_up();
                    }
                    let interval = MpfrInterval::new(lo.clone(), hi.clone()).unwrap();
                    for sine in [false, true] {
                        let actual = if sine { interval.sin() } else { interval.cos() };
                        for point in [&lo, &hi] {
                            let mut low = Float::with_val(512, point);
                            let mut high = low.clone();
                            if sine {
                                low.sin_round(Round::Down);
                                high.sin_round(Round::Up);
                            } else {
                                low.cos_round(Round::Down);
                                high.cos_round(Round::Up);
                            }
                            assert!(actual.lower() <= &low && actual.upper() >= &high,
                                "endpoint excluded: p={p}, multiple={multiple}, offset={offset}, count={count}, sine={sine}, interval={interval:?}, actual={actual:?}, reference=[{low}, {high}]");
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn sine_interval_near_minus_four_pi_contains_exact_rational_taylor_enclosures() {
    use rug::{float::Constant, Rational};
    let mut lo = Float::with_val(32, Float::with_val(512, Constant::Pi) * -4i32);
    for _ in 0..5 {
        lo.next_down();
    }
    let mut hi = lo.clone();
    for _ in 0..9 {
        hi.next_up();
    }
    let actual = MpfrInterval::new(lo.clone(), hi.clone())
        .unwrap()
        .sin()
        .to_rational_interval();
    for endpoint in [lo, hi] {
        let x = endpoint.to_rational().unwrap();
        let square = x.clone() * &x;
        let mut term = x.clone();
        let mut sum = Rational::from(0);
        for k in 0..64u32 {
            sum += &term;
            term *= -square.clone();
            term /= (2 * k + 2) * (2 * k + 3);
        }
        // Degree-128 Taylor polynomial (the even coefficient is zero).
        // The real sine derivative has magnitude <= 1, so |x|^129/129!
        // bounds the remainder without using an MPFR trigonometric oracle.
        let radius = term.abs();
        let exact_lower = sum.clone() - &radius;
        let exact_upper = sum + radius;
        assert!(
            actual.lower() <= &exact_lower && actual.upper() >= &exact_upper,
            "sine endpoint's exact rational Taylor enclosure was excluded at {endpoint}"
        );
    }
}
