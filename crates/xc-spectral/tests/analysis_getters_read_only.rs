//! Analysis getters must not persist cache artifacts.
//!
//! This test runs in its own process (one integration-test binary) with a
//! private managed cache root, so no concurrently running test can write into
//! the directory it inventories.
#![cfg(feature = "hp")]

use rug::Float;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use xc_numerics::grid_integral::{GridVariable, UniformGridScheme};
use xc_spectral::ccm::hp::HighPrecConfig;
use xc_spectral::ccm::CcmParams;
use xc_spectral::distance::hp::{ccm_discretization_distance_hp, ccm_distance_to_target_hp};
use xc_spectral::distance::WeightedIntegrationRule;

fn inventory(root: &Path, result: &mut BTreeMap<PathBuf, (u64, Option<SystemTime>)>) {
    if !root.exists() {
        return;
    }
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_symlink() {
            continue;
        }
        let metadata = entry.metadata().unwrap();
        if metadata.is_dir() {
            inventory(&entry.path(), result);
        } else {
            result.insert(entry.path(), (metadata.len(), metadata.modified().ok()));
        }
    }
}

#[test]
fn analysis_getters_leave_persistent_cache_unchanged() {
    let root = std::env::temp_dir().join(format!(
        "xc-analysis-getters-read-only-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    // Safe: this binary contains only this test, so no other thread reads the
    // environment concurrently.
    std::env::set_var("XC_CACHE_ROOT", &root);
    // A manufactured public target, stored outside the inventoried cache root.
    let spec = std::env::temp_dir().join(format!(
        "xc-analysis-getters-target-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &spec,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1, "profile_id": "analysis-getters-manufactured-v1",
            "base_series": {"term_input_power": 0, "polynomial_coefficients": ["1"],
                "minimum_terms": 1, "maximum_terms": 1000},
            "auxiliary_series": {"term_input_power": 0, "polynomial_coefficients": ["0", "1"],
                "parameter_polynomial_coefficients": ["1"], "minimum_terms": 1, "maximum_terms": 1000}
        }))
        .unwrap(),
    )
    .unwrap();
    std::env::set_var("XC_TARGET_SPEC_FILE", &spec);
    let policy = xc_cache::ManagedArtifactCacheConfig::from_environment()
        .unwrap()
        .unwrap();
    assert_eq!(policy.cache_root, root);
    let mut before = BTreeMap::new();
    inventory(&policy.cache_root, &mut before);
    let mut cfg = HighPrecConfig::for_decimal_digits(20);
    cfg.quad_points = 600;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let alpha = Float::with_val(cfg.precision_bits, 0.5);
    let rule = WeightedIntegrationRule::UniformGrid {
        scheme: UniformGridScheme::Trapezoid,
        variable: GridVariable::U,
        steps: 16,
    };
    for (cutoff, n, m) in [(13u64, 3usize, 4usize), (2, 2, 3), (3, 3, 4)] {
        let first = CcmParams::from_lambda_sq_integer(cutoff, n);
        let second = CcmParams::from_lambda_sq_integer(cutoff, m);
        let target = ccm_distance_to_target_hp(&first, &cfg, &alpha, rule).unwrap();
        assert!(target.distances[0].value.is_finite());
        let self_distance =
            ccm_discretization_distance_hp(&first, &first, &cfg, &alpha, &[rule]).unwrap();
        assert_eq!(self_distance.distances[0].value, 0);
        let between =
            ccm_discretization_distance_hp(&first, &second, &cfg, &alpha, &[rule]).unwrap();
        assert!(between.distances[0].value.is_finite());
        let mut after = BTreeMap::new();
        inventory(&policy.cache_root, &mut after);
        assert_eq!(after, before, "analysis getter persisted cache artifacts");
    }
    assert!(before.is_empty(), "private cache root must start empty");
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_file(&spec);
}
