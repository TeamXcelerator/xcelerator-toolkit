#![cfg(feature = "hp")]
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::{
    ArtifactKey, ArtifactManifest, CacheObjectRef, CacheQuality, CacheVisibility, ContentDigest,
    ToolkitVersion,
};
use xc_spectral::ccm::state_geometry::*;
fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(kind, "synthetic-geometry-fixture", kind.as_bytes()).unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    (manifest, bytes)
}

fn fixture(coefficients: &[&str]) -> (ArtifactManifest, Vec<u8>) {
    source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"eigenvalue":"0.125","eigenvector":coefficients}),
    )
}
fn retained(m: &ArtifactManifest, b: &[u8]) -> RetainedState {
    RetainedState::from_payload(m, b, std::slice::from_ref(&m.content_digest)).unwrap()
}
fn hp(s: &str, p: u32) -> rug::Float {
    rug::Float::with_val(p, rug::Float::parse(s).unwrap())
}
fn close(actual: &str, expected: &str) {
    let a = hp(actual, 384);
    let e = hp(expected, 384);
    assert!(
        (a - e.clone()).abs() <= hp("1e-65", 384) * (1 + e.abs()),
        "actual={actual} expected={expected}"
    );
}

#[test]
fn retained_geometry_matches_independent_complex_fourier_sums() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/state_geometry_oracle.json")).unwrap();
    for case in oracle["cases"].as_array().unwrap() {
        for schema in [3, 4, 5] {
            let coefficients = case["coefficients"].as_array().unwrap();
            let (m, bytes) = source(
                "ccm_weil_eigenpair",
                json!({"schema_version":schema,"lambda_squared":case["cutoff"],"n_modes":(coefficients.len()-1)/2,"precision_bits":256,"eigenvalue":"0","eigenvector":coefficients}),
            );
            let state = retained(&m, &bytes);
            let report =
                analyze_state_geometry(&state, &GeometryOptions::for_source(&state)).unwrap();
            assert_eq!(
                report.orientation,
                case["orientation"].as_i64().unwrap() as i32
            );
            let value = serde_json::to_value(&report).unwrap();
            for field in [
                "coefficient_norm",
                "raw_center",
                "unit_l2_center",
                "coefficient_evenness_defect",
                "unit_l2_signed_mass",
            ] {
                if let Some(reference) = case[field].as_str() {
                    close(value[field].as_str().unwrap(), reference);
                } else {
                    assert!(value[field].is_null());
                }
            }
            for grid in ["coarse", "refined"] {
                for field in [
                    "l2_mass",
                    "sampled_real_minimum",
                    "sampled_negative_part_l1",
                ] {
                    if let Some(reference) = case[grid][field].as_str() {
                        close(value[grid][field].as_str().unwrap(), reference);
                    } else {
                        assert!(value[grid][field].is_null());
                    }
                }
                for field in ["spatial_moments", "outer_shell_masses"] {
                    for (a, e) in value[grid][field]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(case[grid][field].as_array().unwrap())
                    {
                        close(a.as_str().unwrap(), e.as_str().unwrap());
                    }
                }
                close(value[grid]["l2_mass"].as_str().unwrap(), "1");
                close(value[grid]["spatial_moments"][0].as_str().unwrap(), "0");
            }
        }
    }
}

#[test]
fn exponent_boundaries_do_not_change_symmetry_or_erase_the_defect() {
    let (m, bytes) = fixture(&["1e-400000000", "1", "0"]);
    assert!(
        RetainedState::from_payload(&m, &bytes, std::slice::from_ref(&m.content_digest)).is_err()
    );
    let (m, bytes) = fixture(&["1e-200000000", "1", "0"]);
    let state = retained(&m, &bytes);
    let report = analyze_state_geometry(&state, &GeometryOptions::for_source(&state)).unwrap();
    assert_eq!(
        report.physical_sign_status,
        "unavailable_nonreal_fourier_state"
    );
    let actual = hp(&report.coefficient_evenness_defect, 256);
    let expected =
        rug::Float::with_val(256, hp("1e-200000000", 128)) * rug::Float::with_val(256, 2).sqrt();
    assert!(actual > 0);
    assert!((actual / expected - 1_i32).abs() < hp("1e-34", 256));
    for coefficient in [
        "1e-200000000",
        "-1e-200000000",
        "1e200000000",
        "-1e200000000",
    ] {
        let (m, bytes) = fixture(&[coefficient; 3]);
        let state = retained(&m, &bytes);
        let report = analyze_state_geometry(&state, &GeometryOptions::for_source(&state)).unwrap();
        let expected =
            rug::Float::with_val(256, 1) / (rug::Float::with_val(256, 9).ln() * 3_i32).sqrt();
        assert!((hp(&report.unit_l2_center, 256) - expected).abs() < hp("1e-34", 256));
        assert_eq!(hp(&report.coefficient_evenness_defect, 128), 0);
        assert!((hp(&report.refined.l2_mass, 256) - 1_i32).abs() < hp("1e-34", 256));
    }
}

#[test]
fn exact_center_sum_preserves_orientation_under_cancellation() {
    let (m, bytes) = fixture(&["1e100", "1e100", "-1", "1e100", "1e100"]);
    let state = retained(&m, &bytes);
    let report = analyze_state_geometry(&state, &GeometryOptions::for_source(&state)).unwrap();
    assert_eq!(hp(&report.raw_center, 128), -1);
    assert_eq!(report.orientation, -1);
    assert!(hp(&report.unit_l2_center, 128) > 0);
}
