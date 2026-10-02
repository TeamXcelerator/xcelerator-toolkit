#![cfg(feature = "hp")]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, retained_evidence::*, state_geometry::RetainedState};
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
use xc_spectral::ccm::convergence_capture::RunOnceInputs;
fn state(
    c: &str,
    co: &[String],
    e: &str,
    entries: Option<&[String]>,
) -> (
    RetainedState,
    ArtifactManifest,
    Option<RetainedMatrix<'static>>,
) {
    let n = co.len() / 2;
    let mut matrix = None;
    let (mut m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":n,"precision_bits":128,"force_even":false,"eigenvalue":e,"eigenvector":co}),
    );
    if let Some(entries) = entries {
        let (mm, mb) = source(
            "ccm_tau_matrix",
            json!({"schema_version":2,"lambda_squared":c,"n_modes":n,"precision_bits":128,"entries":entries}),
        );
        m.dependencies.push(DependencyRef {
            key: mm.key.clone(),
            content_digest: mm.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        matrix = Some(
            RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest))
                .unwrap(),
        );
    }
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
        matrix,
    )
}
fn input(m: &ArtifactManifest, c: &str, n: usize) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":c,"n_modes":n,"precision_bits":128,"convention_id":"independent comparison fixture","definition_digest":ContentDigest::sha256(b"independent finite algebra"),"approximation_scope":"synthetic stored points; no scientific selection assertion"})).unwrap()
}
fn hp(s: &str) -> Float {
    Float::with_val(768, Float::parse(s).unwrap())
}
fn within(v: &BTreeMap<String, String>, name: &str, expected: &str, p: u32) {
    let expected = hp(expected);
    let lo = hp(&v[&format!("{name}_lower")]);
    let hi = hp(&v[&format!("{name}_upper")]);
    let slack = (Float::with_val(768, 1) + expected.clone().abs()) * hp("1e-155");
    assert!(
        Float::with_val(768, &lo) - &slack <= expected
            && expected <= Float::with_val(768, &hi) + &slack,
        "{name}: [{lo},{hi}] excludes {expected}"
    );
    assert!(
        (hp(&v[name]) - &expected).abs() <= (Float::with_val(768, 1) + expected.abs()) >> (p - 6),
        "inaccurate {name}"
    );
}
fn strings(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().into())
        .collect()
}
fn run(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    input: &ExternalResearchInputs,
    p: u32,
) -> ExtendedAnalysis {
    capture_extended(
        id,
        s,
        m,
        None,
        Some(input),
        &ExtensionOptions {
            working_precision_bits: p,
            ..ExtensionOptions::for_source(s)
        },
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
}
fn column_scale(v: &[String], shift: i32) -> Vec<String> {
    v.iter()
        .map(|v| {
            let mut x = Float::with_val(128, Float::parse(v).unwrap());
            if shift >= 0 {
                x <<= shift as u32
            } else {
                x >>= shift.unsigned_abs()
            }
            xc_numerics::prefix::lossless_decimal(&x)
        })
        .collect()
}
#[test]
fn independent_rational_projector_and_high_precision_coordinate_oracle() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/cluster_operator_oracle.json")).unwrap();
    let mut checks = 0;
    for case in reference["cases"].as_array().unwrap() {
        let entries = strings(&case["matrix"]);
        let co = strings(&case["source"]);
        let e = case["energy"].as_str().unwrap();
        for p in [128, 192, 256] {
            for shift in [-400, 0, 400] {
                let (s, sm, m) = state("9", &column_scale(&co, shift), e, Some(&entries));
                let mut i = input(&sm, "9", co.len() / 2);
                i.run_once = Some(RunOnceInputs {
                    reference_vectors: case["columns"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(k, v)| {
                            column_scale(&strings(v), if k % 2 == 0 { shift } else { -shift })
                        })
                        .collect(),
                    ..Default::default()
                });
                let r = run("operator_cluster", &s, m.as_ref(), &i, p);
                assert_eq!(
                    r.outcome, "point_measurement",
                    "case {} at {p}: {:?}",
                    case["id"], r.reason
                );
                assert!(r.assurance.contains("no full-declared-span"));
                within(
                    &r.values,
                    "source_leakage_squared",
                    case["source_leakage_squared"].as_str().unwrap(),
                    p,
                );
                checks += 1;
                assert_eq!(r.rows.len(), case["rows"].as_array().unwrap().len());
                for (actual, expected) in r.rows.iter().zip(case["rows"].as_array().unwrap()) {
                    for name in [
                        "compressed_operator",
                        "coupling_gram",
                        "signed_complement_feedback",
                        "effective_operator",
                    ] {
                        within(&actual.values, name, expected[name].as_str().unwrap(), p);
                        checks += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checks, 5184);
}
#[test]
fn common_diagonal_shift_does_not_contaminate_coupling_or_feedback() {
    for exponent in [0u32, 400, 10000] {
        let b = if exponent == 0 {
            "3".to_string()
        } else {
            (rug::Integer::from(1) << exponent).to_string()
        };
        for energy in [
            b.clone(),
            if exponent == 0 {
                "3.0000000000000000000000000000000000000001".into()
            } else {
                b.clone()
            },
        ] {
            let a = vec![
                b.clone(),
                "1".into(),
                "2".into(),
                "1".into(),
                b.clone(),
                "1".into(),
                "2".into(),
                "1".into(),
                b.clone(),
            ];
            let (s, sm, m) = state(
                "9",
                &["1".into(), "1".into(), "1".into()],
                &energy,
                Some(&a),
            );
            let mut i = input(&sm, "9", 1);
            i.run_once = Some(RunOnceInputs {
                reference_vectors: vec![vec!["1".into(); 3]],
                ..Default::default()
            });
            for p in [128, 192, 256] {
                let r = run("operator_cluster", &s, m.as_ref(), &i, p);
                assert_eq!(r.outcome, "point_measurement");
                within(
                    &r.rows[0].values,
                    "coupling_gram",
                    &(Float::with_val(768, 2) / 9u32).to_string(),
                    p,
                );
                within(
                    &r.rows[0].values,
                    "signed_complement_feedback",
                    &(Float::with_val(768, -1) / 3u32).to_string(),
                    p,
                );
                assert_eq!(
                    Float::with_val(p, Float::parse(&r.values["retained_energy_shift"]).unwrap()),
                    Float::with_val(p, Float::parse(&b).unwrap())
                );
            }
        }
    }
}
#[test]
fn original_column_points_and_threshold_decisions_are_explicit() {
    let a = vec!["3", "1", "2", "1", "3", "1", "2", "1", "3"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let (s, sm, m) = state("9", &["1".into(), "1".into(), "1".into()], "3", Some(&a));
    let mut i = input(&sm, "9", 1);
    for (power, dimension) in [(80, 2), (96, 1), (100, 1)] {
        let delta = xc_numerics::prefix::lossless_decimal(&(Float::with_val(128, 1) >> power));
        i.run_once = Some(RunOnceInputs {
            reference_vectors: vec![
                vec!["1".into(), "0".into(), "0".into()],
                vec!["1".into(), delta, "0".into()],
            ],
            ..Default::default()
        });
        let r = run("operator_cluster", &s, m.as_ref(), &i, 192);
        assert_eq!(hp(&r.values["subspace_dimension"]), dimension);
        assert!(r.convention.contains("not a full-declared-span"));
    }
    i.run_once = Some(RunOnceInputs {
        reference_vectors: vec![vec!["1".into(); 3]],
        ..Default::default()
    });
    let first = run("operator_cluster", &s, m.as_ref(), &i, 192);
    i.run_once.as_mut().unwrap().reference_vectors[0][1] =
        "1.0000000000000000000000000000000000000001".into();
    let alias = run("operator_cluster", &s, m.as_ref(), &i, 192);
    assert_eq!(first.values, alias.values);
    assert_eq!(
        serde_json::to_value(first.rows).unwrap(),
        serde_json::to_value(alias.rows).unwrap()
    );
    // Lower working precision is rejected; an admitted higher-precision point is retained.
    i.precision_bits = 512;
    let rejected = capture_extended(
        "operator_cluster",
        &s,
        m.as_ref(),
        None,
        Some(&i),
        &ExtensionOptions {
            working_precision_bits: 128,
            ..ExtensionOptions::for_source(&s)
        },
        &[],
        &context(),
    );
    assert!(rejected.is_err());
    let mixed = run("operator_cluster", &s, m.as_ref(), &i, 512);
    assert_eq!(mixed.outcome, "point_measurement");
}
#[test]
fn missing_zero_singular_and_resource_cases_remain_qualified() {
    let a = vec!["3", "0", "0", "0", "3", "0", "0", "0", "3"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let (s, sm, m) = state("9", &["1".into(), "1".into(), "1".into()], "3", Some(&a));
    let mut i = input(&sm, "9", 1);
    i.run_once = Some(RunOnceInputs {
        reference_vectors: vec![vec!["0".into(), "1".into(), "0".into()]],
        ..Default::default()
    });
    let r = run("operator_cluster", &s, m.as_ref(), &i, 192);
    assert_eq!(r.outcome, "partial_unresolved");
    assert_eq!(hp(&r.rows[0].values["coupling_gram"]), 0);
    assert!(!r.rows[0].values.contains_key("signed_complement_feedback"));
    let r = run("operator_cluster", &s, None, &i, 192);
    assert_eq!(r.outcome, "missing_input");
    i.run_once.as_mut().unwrap().reference_vectors[0] = vec!["0".into(); 3];
    let r = run("operator_cluster", &s, m.as_ref(), &i, 192);
    assert_eq!(r.outcome, "unresolved");
    assert!(r.rows.is_empty());
    let o = ExtensionOptions {
        maximum_working_bytes: Some(1),
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "operator_cluster",
        &s,
        m.as_ref(),
        None,
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.values.is_empty());
    // With enough room for vectors but not dense interval factorization, retain core bounds.
    let n = 51;
    let co = vec!["1".into(); n];
    let mut a = vec!["0".into(); n * n];
    for k in 0..n {
        a[k * n + k] = "4".into();
    }
    let (s, sm, m) = state("9", &co, "3", Some(&a));
    let mut i = input(&sm, "9", n / 2);
    i.run_once = Some(RunOnceInputs {
        reference_vectors: vec![co],
        ..Default::default()
    });
    let o = ExtensionOptions {
        working_precision_bits: 192,
        maximum_working_bytes: Some(5_000_000),
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "operator_cluster",
        &s,
        m.as_ref(),
        None,
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "partial_unresolved");
    assert!(r.reason.unwrap().contains("budget"));
    assert!(r.rows[0].values.contains_key("coupling_gram_lower"));
}
