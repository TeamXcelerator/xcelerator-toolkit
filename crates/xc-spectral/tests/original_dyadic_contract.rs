use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive, Zero};
use xc_spectral::ccm::{rank_one::FiniteRankOneOperatorF64, solve_spectrum_f64};

fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/original_dyadic_oracle.json")).unwrap()
}
fn positive(case: &serde_json::Value) -> Vec<f64> {
    case["positive_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| f64::from_bits(u64::from_str_radix(h.as_str().unwrap(), 16).unwrap()))
        .collect()
}
fn full(positive: &[f64]) -> Vec<f64> {
    positive
        .iter()
        .skip(1)
        .rev()
        .chain(positive)
        .copied()
        .collect()
}
// Independent polynomial evaluation over Q, rather than the production
// meromorphic accumulation or its projective normalization.
fn numerator(positive: &[f64], t: &Q) -> Q {
    let n = positive.len() - 1;
    let factor = |j: usize| t - Q::from_integer((j * j).into());
    let product = |omitted: usize| {
        (1..=n)
            .filter(|j| *j != omitted)
            .map(factor)
            .fold(Q::from_integer(1.into()), |a, b| a * b)
    };
    let mut value = Q::from_float(positive[0]).unwrap() * product(0);
    for (j, weight) in positive.iter().enumerate().skip(1) {
        value += Q::from_integer(2.into()) * t * Q::from_float(*weight).unwrap() * product(j);
    }
    value
}

#[test]
fn original_supplied_dyadics_match_independent_qq_root_isolation() {
    for case in oracle()["cases"].as_array().unwrap() {
        let pos = positive(case);
        for interval in case["roots"].as_array().unwrap() {
            let lo: Q = interval["lower"].as_str().unwrap().parse().unwrap();
            let hi: Q = interval["upper"].as_str().unwrap().parse().unwrap();
            let a = numerator(&pos, &lo);
            let b = numerator(&pos, &hi);
            assert!(a.is_zero() || b.is_zero() || a.is_negative() != b.is_negative());
            let bound = Q::from_integer(2.into()).pow(-180);
            assert!(hi - lo < bound);
        }
        for scale in [1.0, -1.0, 2.0_f64.powi(60), 2.0_f64.powi(-60)] {
            let state = full(&pos).iter().map(|x| x * scale).collect::<Vec<_>>();
            let result = solve_spectrum_f64(
                &state,
                pos.len() - 1,
                2.0 * std::f64::consts::PI,
                1e-14,
                600,
            );
            // Retain the preexisting cancellation fixture's explicit unresolved
            // alternative: these roots are especially close, not impossible.
            if case["name"] == "retained_cancellation" {
                if let Err(error) = &result {
                    let message = error.to_string();
                    assert!(message.contains("separate") || message.contains("unresolved"));
                    continue;
                }
            }
            let roots = result.unwrap_or_else(|e| panic!("{} scale {scale}: {e}", case["name"]));
            let expected = case["roots"].as_array().unwrap();
            assert_eq!(roots.len(), expected.len(), "{}", case["name"]);
            for (ordinate, exact) in roots.iter().zip(expected) {
                let expected_t = exact["root_t"].as_f64().unwrap();
                assert!(
                    (ordinate * ordinate - expected_t).abs() <= 4e-13 * expected_t.abs().max(1.0),
                    "{}: {} versus {}",
                    case["name"],
                    ordinate * ordinate,
                    expected_t
                );
            }
        }
    }
}

#[test]
fn independent_quotient_entries_bind_to_original_rational_ratios() {
    for case in oracle()["cases"].as_array().unwrap() {
        let state = full(&positive(case));
        let operator = FiniteRankOneOperatorF64::from_state(&state).unwrap();
        let pairing: Q = state.iter().map(|v| Q::from_float(*v).unwrap()).sum();
        let pivot = operator.quotient_pivot();
        let n = state.len() / 2;
        let indices = (0..state.len()).filter(|i| *i != pivot).collect::<Vec<_>>();
        for (r, &i) in indices.iter().enumerate() {
            for (c, &j) in indices.iter().enumerate() {
                let diagonal = if i == j { i as isize - n as isize } else { 0 };
                let difference = i as isize - pivot as isize;
                let exact = Q::from_integer(diagonal.into())
                    - Q::from_integer(difference.into()) * Q::from_float(state[i]).unwrap()
                        / &pairing;
                assert_eq!(
                    operator.quotient_matrix()[(r, c)].to_bits(),
                    exact.to_f64().unwrap().to_bits()
                );
            }
        }
    }
}

#[test]
fn matrix_route_remains_independent_on_well_conditioned_source() {
    let state = [1.0; 5];
    let matrix = FiniteRankOneOperatorF64::from_state(&state)
        .unwrap()
        .spectrum(1e-12)
        .unwrap();
    let roots = solve_spectrum_f64(&state, 2, 2.0 * std::f64::consts::PI, 1e-14, 600).unwrap();
    let positive = matrix.into_iter().filter(|v| *v > 0.0).collect::<Vec<_>>();
    assert_eq!(positive.len(), roots.len());
    for (a, b) in positive.iter().zip(roots) {
        assert!((a - b).abs() < 1e-12);
    }
}
