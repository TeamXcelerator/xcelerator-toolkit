// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.
#![cfg(feature = "hp")]

use rug::{float::Special, Float};
use serde_json::Value;
use xc_numerics::linalg::try_normalize_l2;

fn parse(p: u32, s: &str) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}

#[test]
fn exact_dyadic_vectors_match_independent_references_under_extreme_signed_scaling() {
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/l2_normalization_oracle.json")).unwrap();
    for p in [64, 128, 256] {
        for row in oracle["cases"].as_array().unwrap() {
            let terms = row["terms"].as_array().unwrap();
            for shift in [-700_000_000i32, -536_870_913, 0, 536_870_913, 700_000_000] {
                for sign in [-1, 1] {
                    let mut vector: Vec<Float> = terms
                        .iter()
                        .map(|term| {
                            let mut x = Float::with_val(p, term[0].as_i64().unwrap() * sign);
                            x <<= term[1].as_i64().unwrap() as i32 + shift;
                            x
                        })
                        .collect();
                    try_normalize_l2(&mut vector).unwrap();
                    let mut tolerance = Float::with_val(p + 64, terms.len());
                    tolerance <<= 8 - (p as i32);
                    for (actual, expected) in vector.iter().zip(row["unit"].as_array().unwrap()) {
                        let reference = parse(p + 64, expected.as_str().unwrap()) * sign;
                        let error = Float::with_val(p + 64, actual) - reference;
                        assert!(
                            error.abs() <= tolerance,
                            "p={p}, shift={shift}, sign={sign}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mixed_precision_and_boundary_squared_underflow_do_not_return_nonunit_vectors() {
    let mut mixed = vec![Float::with_val(2, 3), Float::with_val(256, 4)];
    try_normalize_l2(&mut mixed).unwrap();
    assert!(mixed.iter().all(|v| v.prec() == 256));
    let tolerance = parse(256, "1e-75");
    assert!((mixed[0].clone() - parse(256, "0.6")).abs() < tolerance);
    assert!((mixed[1].clone() - parse(256, "0.8")).abs() < tolerance);
    let mut boundary = Float::with_val(128, 1.75);
    boundary >>= 536_870_913u32;
    let mut vector = vec![boundary.clone(), boundary];
    try_normalize_l2(&mut vector).unwrap();
    let reference=parse(256,"0.7071067811865475244008443621048490392848359376884740365883398689953662392310535194251937");
    for value in vector {
        assert!((Float::with_val(256, value) - &reference).abs() < parse(256, "1e-37"));
    }
}

#[test]
fn range_failures_preserve_input_and_ordinary_uniform_results_preserve_rounding() {
    let mut unrepresentable = vec![parse(128, "1e200000000"), parse(128, "1e-200000000")];
    let original = unrepresentable.clone();
    assert!(try_normalize_l2(&mut unrepresentable).is_err());
    assert_eq!(unrepresentable, original);
    let mut zeros = vec![Float::with_val(128, 0); 3];
    let original = zeros.clone();
    assert!(try_normalize_l2(&mut zeros).is_err());
    assert_eq!(zeros, original);
    let mut invalid = vec![
        Float::with_val(128, Special::Infinity),
        Float::with_val(128, 1),
    ];
    let original = invalid.clone();
    assert!(try_normalize_l2(&mut invalid).is_err());
    assert_eq!(invalid, original);
    try_normalize_l2(&mut []).unwrap();
    // Independently preserve the prior unscaled operation order on an ordinary
    // domain. This checks compatibility, in addition to the reference test above.
    for p in [64, 128, 256] {
        for n in 1usize..=65 {
            let mut values: Vec<Float> = (0..n)
                .map(|i| Float::with_val(p, ((i * 17 + 3) % 29) as i32 - 14))
                .collect();
            let mut previous = values.clone();
            let squares = previous.iter().map(|v| v.clone().square()).collect();
            let norm =
                xc_numerics::reduction::deterministic_pairwise_sum_hp_owned(squares, p).sqrt();
            for value in &mut previous {
                *value /= &norm;
            }
            try_normalize_l2(&mut values).unwrap();
            assert_eq!(values, previous, "p={p},n={n}");
        }
    }
}
