use xc_operator::{DiagonalF64, LinearOperator, RankOneUpdateF64};

#[test]
fn rank_one_action_preserves_representable_products_across_exponent_ranges() {
    let base = DiagonalF64::new("zero", vec![0.]).unwrap();
    // Exact powers of two give an independent integer-exponent oracle.
    for (a, v, x) in [
        (-1000, -100, 1000),
        (1000, -200, -1000),
        (-1000, 700, -500),
        (1000, -700, 500),
    ] {
        let operator = RankOneUpdateF64::new(&base, 2f64.powi(a), vec![2f64.powi(v)]).unwrap();
        let mut result = [0.];
        operator.apply(&[2f64.powi(x)], &mut result).unwrap();
        assert_eq!(result[0], 2f64.powi(a + 2 * v + x), "exponents {a},{v},{x}");
    }
}

#[test]
fn zero_rank_one_weight_is_the_base_action_even_for_large_vectors() {
    let base = DiagonalF64::new("identity", vec![1., 1.]).unwrap();
    let operator = RankOneUpdateF64::new(&base, 0., vec![f64::MAX, f64::MAX]).unwrap();
    let mut result = [0.; 2];
    operator.apply(&[2., -1.], &mut result).unwrap();
    assert_eq!(result, [2., -1.]);
}

#[test]
fn rescaled_rank_one_parameterizations_give_the_same_operator() {
    let base = DiagonalF64::new("zero", vec![0.; 3]).unwrap();
    // alpha*v*(v dot x), v=(1,-2,3), x=(2,1,-2) gives (-6,12,-18).
    for e in [-500, -200, 0, 200, 500] {
        let scale = 2f64.powi(e);
        let operator = RankOneUpdateF64::new(
            &base,
            2f64.powi(-2 * e),
            vec![scale, -2. * scale, 3. * scale],
        )
        .unwrap();
        let mut result = [0.; 3];
        operator.apply(&[2., 1., -2.], &mut result).unwrap();
        assert_eq!(result, [-6., 12., -18.]);
    }
}

#[test]
fn one_dimensional_signed_actions_match_exact_binary_exponent_arithmetic() {
    fn power(e: i32) -> f64 {
        if e >= -1022 {
            f64::from_bits(((e + 1023) as u64) << 52)
        } else {
            f64::from_bits(1u64 << ((e + 1074) as u32))
        }
    }
    let base = DiagonalF64::new("zero", vec![0.]).unwrap();
    let mut cases = 0;
    for a in [-1074, -1000, -400, 0, 400, 1000, 1023] {
        for v in [-1074, -1000, -400, -20, 0, 400, 1000, 1023] {
            for x in [-1074, -1000, -400, -34, 0, 400, 1000, 1023] {
                let e = a + 2 * v + x;
                if !(-1074..=1023).contains(&e) {
                    continue;
                }
                for sign in [-1., 1.] {
                    let operator =
                        RankOneUpdateF64::new(&base, sign * power(a), vec![-power(v)]).unwrap();
                    let mut y = [0.];
                    operator.apply(&[-power(x)], &mut y).unwrap();
                    assert_eq!(y[0], -sign * power(e), "exponents {a},{v},{x}");
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 200);
}

#[test]
fn full_internal_dynamic_range_returns_the_correctly_rounded_exact_action() {
    let base = DiagonalF64::new("zero", vec![0.; 2]).unwrap();
    let operator = RankOneUpdateF64::new(&base, 1., vec![f64::MAX, f64::from_bits(1)]).unwrap();
    let mut output = [0.; 2];
    operator.apply(&[0., 1.], &mut output).unwrap();
    // dot = 2^-1074. MAX * dot = (2^53-1)*2^-103 is exactly
    // representable (exponent -51 with all 52 fractional bits set),
    // while the second component 2^-2148 rounds to zero.
    let first = f64::from_bits(((1023u64 - 51) << 52) | ((1u64 << 52) - 1));
    assert_eq!(output, [first, 0.]);
}
