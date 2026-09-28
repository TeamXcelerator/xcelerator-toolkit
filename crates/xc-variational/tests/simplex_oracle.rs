#![cfg(feature = "hp")]
use rug::Rational;
use xc_variational::maynard::{maynard_2015_m5_candidate, MkMonomialReference, MultiIndex};

#[test]
fn exact_maynard_forms_match_independent_nested_integration() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/independent-simplex-oracle.json")).unwrap();
    let rational =
        |value: &serde_json::Value| Rational::from_str_radix(value.as_str().unwrap(), 10).unwrap();
    let mut comparisons = 0;
    for entry in fixture["entries"].as_array().unwrap() {
        let k = entry["k"].as_u64().unwrap() as usize;
        let reference = MkMonomialReference::new(k, 2).unwrap();
        let index = |name: &str| {
            MultiIndex(
                entry[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect(),
            )
        };
        let a = index("a");
        let b = index("b");
        assert_eq!(reference.i_entry(&a, &b).unwrap(), rational(&entry["i"]));
        comparisons += 1;
        for axis in 0..k {
            assert_eq!(
                reference.j_entry(axis, &a, &b).unwrap(),
                rational(&entry["j"][axis])
            );
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, fixture["entry_comparisons"].as_u64().unwrap());
    let reference = MkMonomialReference::new(5, 3).unwrap();
    let witness = maynard_2015_m5_candidate(&reference).unwrap();
    assert_eq!(
        reference.quadratic_i(&witness).unwrap(),
        rational(&fixture["witness"]["i"])
    );
    assert_eq!(
        reference.quadratic_j_total(&witness).unwrap(),
        rational(&fixture["witness"]["j"])
    );
    assert_eq!(
        reference.rayleigh_quotient(&witness).unwrap(),
        rational(&fixture["witness"]["ratio"])
    );
    eprintln!("{comparisons} independently integrated I/J entries and published M5 witness agree");
}
