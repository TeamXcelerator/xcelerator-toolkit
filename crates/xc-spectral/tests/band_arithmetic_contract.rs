#![cfg(feature = "hp")]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, state_geometry::RetainedState};
fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    (
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(kind, "synthetic-extension-fixture", kind.as_bytes()).unwrap(),
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
        },
        bytes,
    )
}
fn context() -> ArtifactCacheContext<'static> {
    ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: None,
        ordered_overlays: vec!["disabled".into()],
        mode: ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    }
}
fn state(c: &str, co: &[String], p: u32) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":p,"force_even":false,"eigenvalue":"3","eigenvector":co}),
    );
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
    )
}

use xc_numerics::prefix::lossless_decimal as dec;
fn point(t: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(t).unwrap())
}
fn input(
    m: &ArtifactManifest,
    p: u32,
    atoms: serde_json::Value,
    degree: usize,
    cutoffs: serde_json::Value,
) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":0,"precision_bits":p,"convention_id":"signed band arithmetic oracle","definition_digest":ContentDigest::sha256(b"signed band arithmetic oracle"),"approximation_scope":"finite atoms","run_once":{"completion":{"band":{"degree":degree,"coordinate":"declared finite coordinate","definition_digest":ContentDigest::sha256(b"signed band arithmetic model"),"atoms":atoms,"coverage":"finite test atoms","hypotheses":["finite supplied functional"],"borrowed_inputs":[],"input_energy":null,"scoring_roots":[]},"atom_analysis":{"maximum_atoms":4096,"maximum_input_bytes":1000000,"cutoffs":cutoffs}}}})).unwrap()
}
fn capture(s: &RetainedState, i: &ExternalResearchInputs) -> ExtendedAnalysis {
    let record = capture_extended(
        "band_reconstruction",
        s,
        None,
        None,
        Some(i),
        &ExtensionOptions::for_source(s),
        &[],
        &context(),
    )
    .unwrap()
    .value;
    assert_eq!(
        record.request["signed_band_arithmetic"],
        "declared_points_normalized_recurrence_exact_contractions_v1"
    );
    record.data
}
fn close(actual: &str, expected: &str, p: u32, shift: i32) {
    let a = point(actual, p) >> shift;
    let expected = point(expected, p);
    let error = (a - &expected).abs();
    let limit = expected.clone().abs() * (Float::with_val(p, 1) >> (p - 40));
    assert!(
        error <= limit,
        "actual {actual}, normalized expected {expected}, error {error}, limit {limit}"
    );
}
#[test]
fn signed_bands_match_independent_monic_polynomial_oracle_at_binary_scales() {
    let corpus: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/band_oracle.json")).unwrap();
    let mut reports = 0;
    let mut comparisons = 0;
    for q in [64, 128, 256] {
        let (s, m) = state("9", &["1".into()], q);
        let p = q + 64;
        for model in corpus["models"].as_array().unwrap() {
            for shift in [-300_000_000i32, 0, 300_000_000] {
                let degree = model["degree"].as_u64().unwrap() as usize;
                let mut atoms = model["atoms"].clone();
                for atom in atoms.as_array_mut().unwrap() {
                    atom["coordinate"] = json!(dec(&(point(
                        atom["coordinate"].as_str().unwrap(),
                        q
                    ) << shift)));
                    atom["signed_weight"] = json!(dec(&(point(
                        atom["signed_weight"].as_str().unwrap(),
                        q
                    ) >> (2 * shift))));
                }
                let r = capture(&s, &input(&m, q, atoms, degree, json!([])));
                assert_eq!(r.outcome, "point_measurement", "{:?}", r.reason);
                close(
                    &r.values["signed_mass"],
                    model["signed_mass"].as_str().unwrap(),
                    p,
                    -2 * shift,
                );
                comparisons += 1;
                for (j, root) in model["roots"].as_array().unwrap().iter().enumerate() {
                    close(
                        &r.rows[j].values["model_band_root"],
                        root.as_str().unwrap(),
                        p,
                        shift,
                    );
                    comparisons += 1;
                }
                for (j, name) in ["one", "two", "three"].iter().enumerate() {
                    close(
                        &r.values[&format!("band_inverse_moment_{name}")],
                        model["inverse_moments"][j].as_str().unwrap(),
                        p,
                        -((j + 1) as i32) * shift,
                    );
                    comparisons += 1;
                }
                for j in 0..degree {
                    for (field, oracle, power) in [
                        ("jacobi_diagonal", "diagonal", 1),
                        ("next_signed_norm_squared", "next_signed_norm", 2),
                        ("next_absolute_norm_squared", "next_absolute_norm", 2),
                    ] {
                        close(
                            &r.rows[j].values[field],
                            model[oracle][j].as_str().unwrap(),
                            p,
                            power * shift,
                        );
                        comparisons += 1;
                    }
                    if j > 0 {
                        close(
                            &r.rows[j].values["jacobi_off_diagonal_previous"],
                            model["previous_beta"][j - 1].as_str().unwrap(),
                            p,
                            shift,
                        );
                        comparisons += 1;
                    }
                    let leakage =
                        point(&r.rows[j].values["reorthogonalization_correction"], p) >> shift;
                    assert!(leakage.abs() <= Float::with_val(p, 64) >> (p - 40));
                    comparisons += 1;
                }
                reports += 1;
            }
        }
    }
    assert_eq!(reports, 324);
    assert_eq!(comparisons, 5832);
}
#[test]
fn band_original_points_and_cutoff_echoes_are_preserved() {
    let (s, m) = state("9", &["1".into()], 128);
    let mut reports = Vec::new();
    for alias in ["1", "1.0000000000000000000000000000000000000001"] {
        let mut i = input(
            &m,
            128,
            json!([{"coordinate":alias,"signed_weight":alias,"family":"zero"},{"coordinate":"2","signed_weight":"1","family":"zero"}]),
            1,
            json!([alias]),
        );
        let band = i
            .run_once
            .as_mut()
            .unwrap()
            .completion
            .as_mut()
            .unwrap()
            .band
            .as_mut()
            .unwrap();
        band.scoring_roots = vec![alias.into()];
        band.input_energy = Some(alias.into());
        let r = capture(&s, &i);
        assert_eq!(r.outcome, "point_measurement");
        assert_eq!(point(&r.rows[0].values["model_band_root"], 192), 1.5);
        assert_eq!(
            point(&r.rows[0].values["scoring_root_difference"], 192),
            0.5
        );
        assert_eq!(point(&r.values["borrowed_input_energy"], 192), 1);
        assert_eq!(point(&r.rows[1].values["zero_atom_cutoff"], 192), 1);
        reports.push(r);
    }
    assert_eq!(reports[0].values, reports[1].values);
    assert_eq!(reports[0].rows[0].values, reports[1].rows[0].values);
    assert_eq!(reports[0].rows[1].values, reports[1].rows[1].values);
}
#[test]
fn exact_signed_cancellation_recovers_the_positive_functional() {
    let (s, m) = state("9", &["1".into()], 128);
    let atoms = json!([{"coordinate":"1","signed_weight":"1e1000","family":"zero"},{"coordinate":"2","signed_weight":"1","family":"zero"},{"coordinate":"1","signed_weight":"-1e1000","family":"zero"}]);
    let r = capture(&s, &input(&m, 128, atoms, 1, json!([])));
    assert_eq!(r.outcome, "point_measurement");
    assert_eq!(point(&r.values["signed_mass"], 192), 1);
    assert_eq!(point(&r.rows[0].values["model_band_root"], 192), 2);
    assert_eq!(point(&r.values["band_inverse_moment_three"], 192), 0.125);
}
#[test]
fn zero_roots_indefinite_models_and_budgets_remain_unresolved() {
    let (s, m) = state("9", &["1".into()], 128);
    for atoms in [
        json!([{"coordinate":"-1","signed_weight":"1","family":"zero"},{"coordinate":"1","signed_weight":"1","family":"zero"}]),
        json!([{"coordinate":"1","signed_weight":"-1","family":"zero"}]),
    ] {
        let r = capture(&s, &input(&m, 128, atoms, 1, json!([])));
        assert_ne!(r.outcome, "point_measurement");
        assert!(!r.values.contains_key("band_inverse_moment_one"));
    }
    let i = input(
        &m,
        128,
        json!([{"coordinate":"1","signed_weight":"1","family":"zero"}]),
        1,
        json!([]),
    );
    let mut o = ExtensionOptions::for_source(&s);
    o.maximum_working_bytes = Some(1);
    let r = capture_extended(
        "band_reconstruction",
        &s,
        None,
        None,
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_ne!(r.outcome, "point_measurement");
}

#[test]
fn normalized_recurrence_preserves_extreme_finite_singleton_roots() {
    let (s, m) = state("9", &["1".into()], 128);
    for (x, w) in [
        ("1e-200000000", "1e300000000"),
        ("1e200000000", "1e-300000000"),
    ] {
        let r = capture(
            &s,
            &input(
                &m,
                128,
                json!([{"coordinate":x,"signed_weight":w,"family":"zero"}]),
                1,
                json!([]),
            ),
        );
        let root = point(&r.rows[0].values["model_band_root"], 192);
        assert!(root > 0);
        let expected = Float::with_val(192, point(x, 128));
        let relative = (root / &expected - 1u32).abs();
        assert!(relative < Float::with_val(192, 1) >> 170);
        assert_eq!(r.outcome, "partial_unresolved");
        assert!(!r.values.contains_key("band_inverse_moment_three"));
    }
}
