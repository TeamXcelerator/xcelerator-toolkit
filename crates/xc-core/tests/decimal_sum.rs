use std::cmp::Ordering;
use xc_core::DecimalLiteral as D;
#[test]
fn exact_sparse_decimal_comparison_handles_signs_carries_and_huge_exponents() {
    for (a, b, c, expected) in [
        ("10", "9", "1", Ordering::Equal),
        ("-10", "-9", "-1", Ordering::Equal),
        ("0", "1e1000000000", "-1e1000000000", Ordering::Equal),
        ("1e1000000000", "1e1000000000", "1", Ordering::Less),
        ("1e1000000000", "1e1000000000", "-1", Ordering::Greater),
        ("0", "1e-1000000000", "0", Ordering::Less),
        (
            "1",
            "0.999999999999999999999999999999",
            "1e-30",
            Ordering::Equal,
        ),
        (
            "1",
            "0.999999999999999999999999999999",
            "1e-29",
            Ordering::Less,
        ),
    ] {
        assert_eq!(
            D::new(a)
                .unwrap()
                .cmp_sum(&D::new(b).unwrap(), &D::new(c).unwrap())
                .unwrap(),
            expected
        );
    }
    for a in -17_i64..18 {
        for b in -17_i64..18 {
            for c in -17_i64..18 {
                assert_eq!(
                    D::new(a.to_string())
                        .unwrap()
                        .cmp_sum(
                            &D::new(b.to_string()).unwrap(),
                            &D::new(c.to_string()).unwrap()
                        )
                        .unwrap(),
                    a.cmp(&(b + c))
                );
            }
        }
    }
}
