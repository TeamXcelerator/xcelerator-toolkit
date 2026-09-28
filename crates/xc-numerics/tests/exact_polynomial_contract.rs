#![cfg(feature = "hp")]
use rug::Rational;
use xc_numerics::interval::*;
fn q(s: &str) -> Rational {
    s.parse().unwrap()
}
fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/exact_polynomial_oracle.json")).unwrap()
}
fn complex_coefficients(case: &serde_json::Value) -> Vec<ComplexRational> {
    case["coefficients"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| ComplexRational {
            real: q(v[0].as_str().unwrap()),
            imaginary: q(v[1].as_str().unwrap()),
        })
        .collect()
}
#[test]
fn exact_sturm_counts_and_isolation_match_independent_factorizations() {
    for case in oracle()["sturm"].as_array().unwrap() {
        let coefficients: Vec<Rational> = case["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| q(v.as_str().unwrap()))
            .collect();
        let lower = q(case["lower"].as_str().unwrap());
        let upper = q(case["upper"].as_str().unwrap());
        let result = exact_sturm_root_count(&coefficients, lower.clone(), upper.clone()).unwrap();
        let roots: Vec<Rational> = case["roots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| q(v.as_str().unwrap()))
            .collect();
        assert_eq!(result.distinct_real_roots, roots.len());
        assert_eq!(result.square_free, case["square_free"].as_bool().unwrap());
        for root in &roots {
            let lo = root.clone() - q("1/64");
            let hi = root.clone() + q("1/64");
            assert_eq!(
                exact_sturm_root_count(&coefficients, lo, hi)
                    .unwrap()
                    .distinct_real_roots,
                1
            );
            assert!(exact_sturm_root_count(&coefficients, root.clone(), upper.clone()).is_err());
        }
        let isolated = exact_sturm_isolate_roots(&coefficients, lower, upper, q("1/32"), 64);
        {
            let intervals = isolated.unwrap();
            assert_eq!(intervals.len(), roots.len());
            for (interval, root) in intervals.iter().zip(&roots) {
                assert!(interval.contains(root));
                assert!(interval.width() <= q("1/32"));
            }
            for pair in intervals.windows(2) {
                assert!(pair[0].upper() <= pair[1].lower());
            }
        }
    }
}
#[test]
fn exact_contours_match_factored_complex_roots_with_multiplicity() {
    for case in oracle()["contours"].as_array().unwrap() {
        let coefficients = complex_coefficients(case);
        let bounds: Vec<Rational> = case["rectangle"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| q(v.as_str().unwrap()))
            .collect();
        let rectangle = RationalContourRectangle::new(
            bounds[0].clone(),
            bounds[1].clone(),
            bounds[2].clone(),
            bounds[3].clone(),
        )
        .unwrap();
        let result =
            certify_polynomial_zero_count_on_rectangle(&coefficients, rectangle, 32).unwrap();
        assert!(result.rigorous && result.boundary_excludes_zero);
        assert_eq!(result.zero_count, case["count"].as_u64().unwrap() as usize);
        assert!(result
            .cells
            .iter()
            .all(|cell| cell.image_enclosure.excludes_zero()));
    }
}
#[test]
fn exact_interval_sqrt_bounds_are_outward_and_on_the_requested_grid() {
    for numerator in 0..50 {
        for denominator in 1..10 {
            for bits in [0u32, 1, 7, 53, 256] {
                let value = Rational::from((numerator, denominator));
                let result = RationalInterval::point(value.clone())
                    .sqrt_nonnegative(bits)
                    .unwrap();
                let square = |x: &Rational| x.clone() * x;
                let step = Rational::from((1, rug::Integer::from(1) << bits));
                assert!(square(result.lower()) <= value && square(result.upper()) >= value);
                assert!(result.width() <= step);
                assert!(square(&(result.lower().clone() + &step)) > value);
                if result.upper() > &0 {
                    assert!(square(&(result.upper().clone() - step)) < value);
                }
            }
        }
    }
    assert!(RationalInterval::point(q("1"))
        .sqrt_nonnegative(u32::MAX)
        .is_err());
}
#[test]
fn invalid_public_rectangles_and_boundary_roots_fail_before_certification() {
    let one = vec![ComplexRational {
        real: q("1"),
        imaginary: q("0"),
    }];
    for (a, b, c, d) in [
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (1, 0, 1, 0),
    ] {
        let rectangle = RationalContourRectangle {
            real_lower: a.into(),
            real_upper: b.into(),
            imaginary_lower: c.into(),
            imaginary_upper: d.into(),
        };
        assert!(certify_polynomial_zero_count_on_rectangle(&one, rectangle, 8).is_err());
    }
    let polynomial = vec![ComplexRational::zero(), one[0].clone()];
    let rectangle = RationalContourRectangle::new(q("0"), q("1"), q("0"), q("1")).unwrap();
    assert!(
        certify_polynomial_zero_count_on_rectangle(&polynomial, rectangle, usize::MAX).is_err()
    );
}
#[test]
fn rouche_and_argument_difference_agree_with_exact_known_roots() {
    for degree in 1..9 {
        let mut coefficients = vec![ComplexRational::zero(); degree + 1];
        coefficients[0] = ComplexRational {
            real: q("1/8"),
            imaginary: q("0"),
        };
        coefficients[degree] = ComplexRational {
            real: q("1"),
            imaginary: q("1"),
        };
        assert_eq!(
            certify_polynomial_zero_count_on_circle(&coefficients, q("1"))
                .unwrap()
                .zero_count,
            degree
        );
        coefficients[0].real = q("8");
        assert_eq!(
            certify_polynomial_zero_count_on_circle(&coefficients, q("1"))
                .unwrap()
                .zero_count,
            0
        );
    }
    let linear = vec![
        ComplexRational::zero(),
        ComplexRational {
            real: q("1"),
            imaginary: q("0"),
        },
    ];
    let rectangle = RationalContourRectangle::new(q("-1"), q("1"), q("-1"), q("1")).unwrap();
    let result =
        certify_rational_function_argument_count_on_rectangle(&linear, &linear, rectangle, 16)
            .unwrap();
    assert!(result.rigorous);
    assert_eq!(result.zeros_minus_poles, 0);
}
