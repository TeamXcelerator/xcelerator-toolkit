//! Independent exact 2x2 trace/determinant oracle for every admitted interval family.
#![cfg(feature = "hp")]
use rug::Rational;
use xc_certify::exact::{
    interval_symmetric_ldlt_inertia, interval_symmetric_ldlt_inertia_mpfr, IntervalInertiaResult,
};
use xc_numerics::interval::RationalInterval;

fn exact_counts(a: &Rational, b: &Rational, c: &Rational) -> (usize, usize) {
    let determinant = a.clone() * c - b.clone() * b;
    let trace = a.clone() + c;
    if determinant < 0 {
        (1, 1)
    } else if determinant > 0 {
        if trace > 0 {
            (2, 0)
        } else {
            (0, 2)
        }
    } else if trace > 0 {
        (1, 0)
    } else if trace < 0 {
        (0, 1)
    } else {
        (0, 0)
    }
}
#[test]
fn interval_inertia_matches_exact_quadratic_signs_on_independent_family_members() {
    let mut conclusive = 0;
    let mut sampled = 0;
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for radius in [Rational::from(0), Rational::from((1, 8))] {
                    let intervals: Vec<_> = [a, b, b, c]
                        .into_iter()
                        .map(|x| {
                            RationalInterval::new(
                                Rational::from(x) - &radius,
                                Rational::from(x) + &radius,
                            )
                            .unwrap()
                        })
                        .collect();
                    let reports = [
                        interval_symmetric_ldlt_inertia(&intervals, 2).unwrap(),
                        interval_symmetric_ldlt_inertia_mpfr(&intervals, 2, 32).unwrap(),
                        interval_symmetric_ldlt_inertia_mpfr(&intervals, 2, 127).unwrap(),
                    ];
                    for report in reports {
                        if let IntervalInertiaResult::Conclusive {
                            positive, negative, ..
                        } = report
                        {
                            conclusive += 1;
                            for da in [-1, 0, 1] {
                                for db in [-1, 0, 1] {
                                    for dc in [-1, 0, 1] {
                                        let av = Rational::from(a) + radius.clone() * da;
                                        let bv = Rational::from(b) + radius.clone() * db;
                                        let cv = Rational::from(c) + radius.clone() * dc;
                                        assert_eq!((positive,negative),exact_counts(&av,&bv,&cv),"family {a},{b},{c}, radius {radius}; sample {da},{db},{dc}");
                                        sampled += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(conclusive > 100);
    eprintln!("independent quadratic-sign oracle: {conclusive} conclusive family reports, {sampled} exact member comparisons");
}
