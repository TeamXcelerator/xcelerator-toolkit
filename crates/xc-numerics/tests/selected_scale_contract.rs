#![cfg(feature = "hp")]
use rug::{Float, Rational};
use xc_numerics::eigen::*;

fn options(p: u32, last: usize, tolerance: Float) -> HpSelectedTridiagonalEigenpairOptions {
    HpSelectedTridiagonalEigenpairOptions {
        first_index: last,
        last_index: last,
        absolute_tolerance: tolerance,
        maximum_bisection_iterations: 1600,
        eigenvector_options: TridiagEigvecOptions {
            solver: TridiagSolver::BandedInterleaved,
            max_steps: 20,
            early_termination: false,
        },
        precision_bits: p,
    }
}

#[test]
fn padded_search_bracket_does_not_excuse_an_inaccurate_tiny_state() {
    let p = 128;
    let scale: Float = Float::with_val(p, 1) >> 400_i32;
    let diagonal = [-scale.clone(), scale];
    let off = [Float::with_val(p, 0)];
    // Historically accepted lambda/s ~= 0.462 and relative residual ~= 0.607.
    let result = tridiag_selected_eigenpairs_hp(
        &diagonal,
        &off,
        &options(p, 1, Float::with_val(p, 1) >> 70_i32),
    )
    .unwrap();
    let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &result.items[0] else {
        panic!("exact tiny simple state must recover");
    };
    assert_eq!(pair.eigenvalue, diagonal[1]);
    // Exact rational residual of the original diagonal source, independent
    // of the solver's interval evidence. Inverse iteration need not produce
    // a bitwise zero component (the recovered value is about 1e-1885).
    let x = pair.eigenvector[0].to_rational().unwrap();
    let y = pair.eigenvector[1].to_rational().unwrap();
    let lambda = pair.eigenvalue.to_rational().unwrap();
    let r0 = (diagonal[0].to_rational().unwrap() - &lambda) * &x;
    let r1 = (diagonal[1].to_rational().unwrap() - &lambda) * &y;
    let bound = (Float::with_val(p, &diagonal[1]) >> 100u32)
        .to_rational()
        .unwrap();
    assert!(r0.clone() * r0 + r1.clone() * r1 <= bound.clone() * bound);
    let angle_bound = Rational::from((1, rug::Integer::from(1) << 100u32));
    assert!(x.clone().abs() <= angle_bound);
    assert_eq!(pair.eigenvector[1].clone().abs(), 1);
    assert!(pair.residual_norm <= Float::with_val(p, &diagonal[1]) >> 100u32);
}

#[test]
fn resolved_scaled_states_satisfy_the_known_source_eigenspace() {
    for p in [64, 128, 256] {
        for exponent in [-400_i32, 0, 400] {
            let scale: Float = Float::with_val(p, 1) << exponent;
            let diagonal = [-scale.clone(), scale.clone()];
            let off = [Float::with_val(p, 0)];
            for index in 0..2 {
                let result = tridiag_selected_eigenpairs_hp(
                    &diagonal,
                    &off,
                    &options(p, index, scale.clone() >> ((p / 2 + 8) as i32)),
                )
                .unwrap();
                let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &result.items[0] else {
                    panic!("distinct resolved values must have individual states");
                };
                let expected = if index == 0 { -1 } else { 1 };
                let value = Float::with_val(p, &pair.eigenvalue / &scale);
                assert!(
                    Float::with_val(p, value - expected).abs()
                        < Float::with_val(p, 1) >> (p / 2 - 4)
                );
                // The opposite coordinate measures error in the selected eigenspace.
                assert!(
                    pair.eigenvector[1 - index].clone().abs()
                        < Float::with_val(p, 1) >> (p / 2 - 4)
                );
                assert!(
                    Float::with_val(p, &pair.residual_norm / &scale)
                        < Float::with_val(p, 1) >> (p / 2 - 4)
                );
            }
        }
    }
}

#[test]
fn repeated_and_zero_scaled_values_remain_clusters() {
    let p = 128;
    for exponent in [-400_i32, 0, 400] {
        let scale: Float = Float::with_val(p, 1) << exponent;
        for value in [0, 1] {
            let diagonal = vec![Float::with_val(p, value) * &scale; 2];
            let mut request = options(p, 1, scale.clone() >> 70_i32);
            request.first_index = 0;
            let result =
                tridiag_selected_eigenpairs_hp(&diagonal, &[Float::with_val(p, 0)], &request)
                    .unwrap();
            assert_eq!(result.vector_recoveries, 0);
            assert!(matches!(
                &result.items[..],
                [HpSelectedTridiagonalItem::Cluster(_)]
            ));
        }
    }
}
