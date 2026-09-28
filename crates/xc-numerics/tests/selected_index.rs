#![cfg(feature = "hp")]
//! Exact parity/index oracle, independently derived from a five-node Jacobi chain.
use rug::{ops::Pow, Float};
use xc_numerics::eigen::*;

#[test]
fn closed_endpoint_neighbor_cannot_be_returned_as_selected_simple_pair() {
    for p in [128, 256] {
        let a = Float::with_val(p, 2).pow(-40i32);
        let d = vec![Float::with_val(p, 0); 5];
        let e = vec![
            a.clone(),
            Float::with_val(p, 1),
            Float::with_val(p, 1),
            a.clone(),
        ];
        // Reflection-odd spectrum is {-a,+a}; reflection-even is
        // {-sqrt(2+a*a),0,+sqrt(2+a*a)}. Index 1 belongs to the odd sector.
        for early_termination in [false, true] {
            let result = tridiag_selected_eigenpairs_hp(
                &d,
                &e,
                &HpSelectedTridiagonalEigenpairOptions {
                    first_index: 1,
                    last_index: 1,
                    absolute_tolerance: Float::with_val(p, &a * 1.5),
                    maximum_bisection_iterations: 512,
                    eigenvector_options: TridiagEigvecOptions {
                        early_termination,
                        ..Default::default()
                    },
                    precision_bits: p,
                },
            );
            match result {
                Err(error) => assert!(
                    error.to_string().contains("index") || error.to_string().contains("enclosure"),
                    "selected index 1 failed at {p} bits, early_termination={early_termination}: {error:#}"
                ),
                Ok(report) => {
                    if let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &report.items[0] {
                        let mut error = pair.eigenvalue.clone();
                        error += &a;
                        assert!(error.abs() < Float::with_val(p, &a / 1000));
                        let reflection_error =
                            Float::with_val(p, &pair.eigenvector[0] + &pair.eigenvector[4]).abs();
                        assert!(reflection_error < Float::with_val(p, 2).pow(-60i32));
                    }
                }
            }
        }
    }
}

#[test]
fn separated_selected_pairs_remain_available_with_bound_indices() {
    let p = 128;
    let d = [-3, 1, 4].map(|x| Float::with_val(p, x));
    let e = [Float::with_val(p, 0), Float::with_val(p, 0)];
    let report = tridiag_selected_eigenpairs_hp(
        &d,
        &e,
        &HpSelectedTridiagonalEigenpairOptions {
            first_index: 0,
            last_index: 2,
            absolute_tolerance: Float::with_val(p, 2).pow(-80i32),
            maximum_bisection_iterations: 256,
            eigenvector_options: Default::default(),
            precision_bits: p,
        },
    )
    .unwrap();
    assert_eq!(report.items.len(), 3);
    for (k, item) in report.items.iter().enumerate() {
        let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = item else {
            panic!("separated diagonal value became a cluster")
        };
        assert!((pair.eigenvalue.clone() - &d[k]).abs() < Float::with_val(p, 2).pow(-60i32));
        assert!(pair.eigenvector[k].clone().abs() > Float::with_val(p, 0.99));
    }
}
