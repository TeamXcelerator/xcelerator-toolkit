use std::path::PathBuf;
fn fixture(label: &str, value: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("xc-zero-audit-{}-{label}.json", std::process::id()));
    std::fs::write(&p, serde_json::to_vec(&vec![value]).unwrap()).unwrap();
    p
}
#[test]
fn binary64_loader_rejects_nonfinite_reference_numbers() {
    for (i, value) in ["NaN", "inf", "1e999"].into_iter().enumerate() {
        let path = fixture(&format!("f64-{i}"), value);
        assert!(xc_zeta::zeros::first_n_f64(&path, 1).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
#[cfg(feature = "hp")]
#[test]
fn hp_loader_rejects_nonfinite_numbers_and_invalid_precision() {
    for (i, value) in ["NaN", "inf", "1e999999999"].into_iter().enumerate() {
        let path = fixture(&format!("hp-{i}"), value);
        assert!(xc_zeta::zeros::first_n_hp(&path, 1, 128).is_err());
        std::fs::remove_file(path).unwrap();
    }
    let path = fixture("precision", "14.125");
    for p in [0, 1_000_001, u32::MAX] {
        let result = std::panic::catch_unwind(|| xc_zeta::zeros::first_n_hp(&path, 1, p));
        assert!(matches!(result, Ok(Err(_))));
    }
    assert_eq!(
        xc_zeta::zeros::first_n_hp(&path, 1, 128).unwrap()[0],
        14.125
    );
    std::fs::remove_file(path).unwrap();
}
