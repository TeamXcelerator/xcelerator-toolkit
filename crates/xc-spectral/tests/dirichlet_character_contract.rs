// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

use serde_json::{json, Value};
use xc_spectral::lfunction::LFunctionSpec;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/dirichlet_character_oracle.json")).unwrap()
}

#[test]
fn exhaustive_small_characters_match_independent_all_pairs_oracle() {
    for row in fixture()["cases"].as_array().unwrap() {
        let q = row["q"].as_u64().unwrap();
        let chi: Vec<i8> = serde_json::from_value(row["values"].clone()).unwrap();
        let parity = row["parity"].as_u64().unwrap() as u8;
        let expected = row["valid"].as_bool().unwrap();
        let direct = LFunctionSpec::new(q, chi.clone(), parity, "custom".into());
        assert_eq!(direct.is_ok(), expected, "{row}");
        let wire = json!({"modulus":q,"chi":chi,"parity":parity,"label":"custom"});
        let decoded = serde_json::from_value::<LFunctionSpec>(wire.clone());
        assert_eq!(decoded.is_ok(), expected, "{row}");
        if let Ok(spec) = direct {
            assert_eq!(spec.modulus(), q);
            assert_eq!(spec.values(), chi);
            assert_eq!(spec.parity(), parity);
            assert!(spec.is_real());
            assert_eq!(spec.is_even(), parity == 0);
            assert_eq!(spec.is_trivial(), q == 1);
            assert_eq!(serde_json::to_value(spec).unwrap(), wire);
        }
    }
}

#[test]
fn prime_power_values_match_independent_modular_exponentiation() {
    for row in fixture()["powers"].as_array().unwrap() {
        let q = row[0].as_u64().unwrap();
        let chi: Vec<i8> = serde_json::from_value(row[1].clone()).unwrap();
        let parity = u8::from(chi[(q - 1) as usize] == -1);
        let spec = LFunctionSpec::new(q, chi, parity, "custom".into()).unwrap();
        let base = row[2].as_u64().unwrap();
        let exponent = row[3].as_u64().unwrap() as u32;
        let expected = row[4].as_i64().unwrap() as i8;
        assert_eq!(spec.chi_at_prime_power(base, exponent), expected, "{row}");
        assert_eq!(
            spec.chi_at_prime_power_f64(base, exponent),
            f64::from(expected)
        );
    }
    for spec in LFunctionSpec::builtin_all() {
        let checked = LFunctionSpec::new(
            spec.modulus(),
            spec.values().to_vec(),
            spec.parity(),
            spec.label.clone(),
        )
        .unwrap();
        for n in 0..=1024 {
            assert_eq!(spec.chi_at(n), checked.chi_at(n));
            assert_eq!(spec.chi_at_f64(n), f64::from(checked.chi_at(n)));
        }
    }
}

#[test]
fn malformed_characters_and_parity_are_rejected_at_all_construction_boundaries() {
    let cases = [
        (0, vec![], 0),
        (1, vec![], 0),
        (1, vec![0], 0),
        (1, vec![-1], 0),
        (1, vec![1], 1),
        (1, vec![1], 2),
        (3, vec![0, 1], 1),
        (3, vec![0, 1, -1, 0], 1),
        (3, vec![0, 1, -1], 0),
        (3, vec![0, 1, 2], 0),
        (3, vec![1, 1, 1], 0),
        (3, vec![0, 0, -1], 1),
        (5, vec![0, 1, 1, 1, -1], 1),
        (u64::MAX, vec![1], 0),
    ];
    for (q, chi, parity) in cases {
        assert!(LFunctionSpec::new(q, chi.clone(), parity, "bad".into()).is_err());
        let wire = json!({"modulus":q,"chi":chi,"parity":parity,"label":"bad"});
        assert!(serde_json::from_value::<LFunctionSpec>(wire).is_err());
    }
    // A principal character above modulus one is valid but does not give zeta.
    let principal = LFunctionSpec::new(6, vec![0, 1, 0, 0, 0, 1], 0, "principal".into()).unwrap();
    assert!(!principal.is_trivial());
    assert_eq!(principal.chi_at_prime_power(3, 0), 1);
    assert_eq!(principal.chi_at_prime_power(3, 1), 0);
}
