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
            producer_toolkit_version: ToolkitVersion::parse("0.15.1").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.15.1").unwrap(),
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
use xc_spectral::ccm::{
    convergence_capture::{ComparisonState, OperatorAction, RunOnceInputs},
    research_completion::CompletionInputs,
};
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
        json!({"schema_version":3,"lambda_squared":c,"n_modes":n,"precision_bits":128,"force_even":true,"eigenvalue":e,"eigenvector":co}),
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
fn scaled(v: &[String], e: i32) -> Vec<String> {
    v.iter()
        .map(|v| {
            let mut v = Float::with_val(128, Float::parse(v).unwrap());
            if e >= 0 {
                v <<= e as u32;
                v = -v;
            } else {
                v >>= e.unsigned_abs();
            }
            v.to_string()
        })
        .collect()
}
fn action(label: &str, values: Vec<String>) -> OperatorAction {
    OperatorAction {
        label: label.into(),
        source_digest: ContentDigest::sha256(label.as_bytes()),
        action: values,
        convention: "same signed center-oriented unit source".into(),
    }
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
fn comparison(coefficients: Vec<String>, matrix: Vec<String>) -> ComparisonState {
    ComparisonState {
        source_digest: ContentDigest::sha256(b"comparison"),
        matrix_digest: ContentDigest::sha256(b"comparison matrix"),
        lambda_squared: "9.0".into(),
        n_modes: coefficients.len() / 2,
        precision_bits: 128,
        coefficients,
        matrix,
        eigenvalue: "0".into(),
        assembly_policy: "same supplied Fourier basis".into(),
    }
}
#[test]
fn all_prefixes_block_defects_and_action_differences_match_independent_exact_algebra() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/transfer_consistency_oracle.json")).unwrap();
    let mut count = 0;
    for case in oracle["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for exponent in [0, 400, -400] {
                let co = scaled(&strings(&case["coefficients"]), exponent);
                let entries = strings(&case["matrix"]);
                let (s, sm, m) = state(
                    "9",
                    &co,
                    case["eigenvalue"].as_str().unwrap(),
                    Some(&entries),
                );
                let mut i = input(&sm, "9", co.len() / 2);
                i.run_once = Some(RunOnceInputs {
                    comparison: Some(comparison(
                        scaled(&strings(&case["comparison_coefficients"]), -exponent),
                        strings(&case["comparison_matrix"]),
                    )),
                    component_actions: vec![action(
                        "tau_prime_reconstructed",
                        strings(&case["reconstructed_action"]),
                    )],
                    completion: Some(CompletionInputs {
                        independent_actions: vec![action(
                            "tau_prime_direct",
                            strings(&case["direct_action"]),
                        )],
                        ..Default::default()
                    }),
                    ..Default::default()
                });
                let report = run("finite_section_transfer", &s, m.as_ref(), &i, p);
                assert_eq!(report.rows.len(), co.len() / 2 + 1);
                for (k, expected) in case["prefixes"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(report.rows[k].ordinal, k + 1);
                    assert_eq!(report.rows[k].outcome, "point_measurement");
                    within(&report.rows[k].values, "n_modes", &k.to_string(), p);
                    for (name, value) in expected.as_object().unwrap() {
                        within(&report.rows[k].values, name, value.as_str().unwrap(), p);
                        count += 1;
                    }
                }
                for (name, value) in case["comparison"].as_object().unwrap() {
                    within(&report.values, name, value.as_str().unwrap(), p);
                    count += 1;
                }
                let report = run("consistency", &s, None, &i, p);
                assert_eq!(report.rows[0].outcome, "point_measurement");
                for (name, value) in case["consistency"].as_object().unwrap() {
                    within(&report.rows[0].values, name, value.as_str().unwrap(), p);
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 6966);
}
#[test]
fn finite_transfer_preserves_common_shift_residuals_and_scale_invariance() {
    let co = vec!["1".into(); 3];
    for exponent in [400u32, 10000] {
        let value = Float::with_val(128, 1) << exponent;
        for big in [value.to_string(), value.to_integer().unwrap().to_string()] {
            let entries = (0..3)
                .flat_map(|i| {
                    (0..3)
                        .map(|j| if i == j { big.clone() } else { "1".into() })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let (s, sm, m) = state("9", &co, &big, Some(&entries));
            let i = input(&sm, "9", 1);
            let r = run("finite_section_transfer", &s, m.as_ref(), &i, 192);
            within(&r.rows[1].values, "low_residual_squared", "4", 192);
            within(&r.rows[1].values, "omitted_mass", "0", 192);
            assert_eq!(r.rows[1].values["omitted_mass"], "0");
        }
    }
    let entries = vec!["3", "1", "1", "1", "3", "1", "1", "1", "3"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let (s, sm, m) = state("9", &co, "3", Some(&entries));
    for value in ["1", "1e-200000000", "1e200000000"] {
        let mut i = input(&sm, "9", 1);
        i.run_once = Some(RunOnceInputs {
            comparison: Some(comparison(vec![value.into(); 3], vec!["0".into(); 9])),
            ..Default::default()
        });
        let r = run("finite_section_transfer", &s, m.as_ref(), &i, 192);
        within(&r.values, "comparison_state_signed_block_defect", "5", 192);
    }
}
#[test]
fn consistency_preserves_original_points_and_representable_tiny_norms() {
    let (s, sm, _) = state("9", &["1".into(), "1".into(), "1".into()], "3", None);
    for value in ["1", "1.0000000000000000000000000000000000000001"] {
        let mut i = input(&sm, "9", 1);
        i.run_once = Some(RunOnceInputs {
            component_actions: vec![action(
                "tau_prime_reconstructed",
                vec!["0".into(), "1".into(), "0".into()],
            )],
            completion: Some(CompletionInputs {
                independent_actions: vec![action(
                    "tau_prime_direct",
                    vec!["0".into(), value.into(), "0".into()],
                )],
                ..Default::default()
            }),
            ..Default::default()
        });
        let r = run("consistency", &s, None, &i, 192);
        assert_eq!(hp(&r.rows[0].values["action_difference_norm"]), 0);
        assert_eq!(hp(&r.rows[0].values["signed_energy_difference"]), 0);
    }
    for value in ["1e-200000000", "1e200000000"] {
        let mut i = input(&sm, "9", 1);
        i.run_once = Some(RunOnceInputs {
            component_actions: vec![action("tau_prime_reconstructed", vec!["0".into(); 3])],
            completion: Some(CompletionInputs {
                independent_actions: vec![action(
                    "tau_prime_direct",
                    vec!["0".into(), value.into(), "0".into()],
                )],
                ..Default::default()
            }),
            ..Default::default()
        });
        let r = run("consistency", &s, None, &i, 192);
        let expected = Float::with_val(768, Float::with_val(128, Float::parse(value).unwrap()));
        let values = &r.rows[0].values;
        assert!(
            hp(&values["action_difference_norm_lower"]) <= expected
                && expected <= hp(&values["action_difference_norm_upper"])
        );
        assert!(hp(&values["action_difference_norm"]) > 0);
        assert!(
            (hp(&values["action_difference_norm"]) / expected - 1u32).abs()
                < (Float::with_val(768, 1) >> 185)
        );
    }
}
#[test]
fn operator_cluster_declared_span_is_invariant_under_independent_column_scales() {
    let entries = vec!["3", "1", "1", "1", "3", "1", "1", "1", "3"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let (s, sm, m) = state(
        "9",
        &["1".into(), "1".into(), "1".into()],
        "3",
        Some(&entries),
    );
    for value in [
        "1".to_string(),
        (Float::with_val(128, 1) >> 100u32).to_string(),
        "1e-200000000".into(),
        "1e200000000".into(),
    ] {
        let mut i = input(&sm, "9", 1);
        i.run_once = Some(RunOnceInputs {
            reference_vectors: vec![vec!["0".into(), value, "0".into()]],
            ..Default::default()
        });
        let r = run("operator_cluster", &s, m.as_ref(), &i, 192);
        assert_eq!(r.outcome, "point_measurement");
        assert_eq!(hp(&r.values["subspace_dimension"]), 1);
        for (name, value) in [
            ("compressed_operator", 3),
            ("coupling_gram", 2),
            ("signed_complement_feedback", 2),
            ("effective_operator", 1),
        ] {
            assert_eq!(hp(&r.rows[0].values[name]), value);
        }
    }
}
#[test]
fn finite_transfer_and_consistency_keep_zero_missing_and_budget_cases_explicit() {
    let entries = vec!["3", "1", "1", "1", "3", "1", "1", "1", "3"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let (s, sm, m) = state(
        "9",
        &["1".into(), "1".into(), "1".into()],
        "3",
        Some(&entries),
    );
    let mut i = input(&sm, "9", 1);
    i.run_once = Some(RunOnceInputs {
        comparison: Some(comparison(vec!["0".into(); 3], vec!["0".into(); 9])),
        component_actions: vec![action("tau_prime_reconstructed", vec!["0".into(); 3])],
        completion: Some(CompletionInputs {
            independent_actions: vec![action("tau_prime_direct", vec!["1".into(); 3])],
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("finite_section_transfer", &s, m.as_ref(), &i, 192);
    assert_eq!(r.outcome, "partial_unresolved");
    assert!(!r
        .values
        .contains_key("comparison_state_signed_block_defect"));
    assert!(r
        .values
        .contains_key("comparison_block_frobenius_difference"));
    assert_eq!(r.rows.len(), 2);
    for id in ["finite_section_transfer", "consistency"] {
        let o = ExtensionOptions {
            maximum_working_bytes: Some(1),
            ..ExtensionOptions::for_source(&s)
        };
        let r = capture_extended(id, &s, m.as_ref(), None, Some(&i), &o, &[], &context())
            .unwrap()
            .value
            .data;
        assert_eq!(r.outcome, "unresolved");
        assert!(r.values.is_empty() && r.rows.is_empty());
    }
    let r = run("finite_section_transfer", &s, None, &i, 192);
    assert_eq!(r.outcome, "missing_input");
    i.run_once.as_mut().unwrap().component_actions.clear();
    let r = run("consistency", &s, None, &i, 192);
    assert_eq!(r.rows[0].outcome, "missing_input");
    assert!(r.rows[0].values.is_empty());
    let mut cmp = comparison(vec!["1".into(); 3], entries);
    cmp.matrix[0] = "3.0000000000000000000000000000000000000001".into();
    i.run_once.as_mut().unwrap().comparison = Some(cmp);
    let r = run("finite_section_transfer", &s, m.as_ref(), &i, 192);
    within(&r.values, "comparison_block_frobenius_difference", "0", 192);
    within(&r.values, "comparison_state_signed_block_defect", "0", 192);
    let (s, sm, m) = state("9", &["-2".into()], "3", Some(&["5".into()]));
    let i = input(&sm, "9", 0);
    let r = run("finite_section_transfer", &s, m.as_ref(), &i, 128);
    assert_eq!(r.rows.len(), 1);
    within(&r.rows[0].values, "low_residual_squared", "4", 128);
    within(&r.rows[0].values, "truncated_energy", "5", 128);
    within(&r.rows[0].values, "omitted_mass", "0", 128);
    let big = Float::with_val(128, 1) << 10000u32;
    let a = big.to_integer().unwrap().to_string();
    let b = (-big * 2u32).to_integer().unwrap().to_string();
    let matrix = vec![
        a.clone(),
        "1".into(),
        "1".into(),
        "1".into(),
        b,
        "1".into(),
        "1".into(),
        "1".into(),
        a,
    ];
    let (s, sm, m) = state(
        "9",
        &["1".into(), "1".into(), "1".into()],
        "0",
        Some(&matrix),
    );
    let i = input(&sm, "9", 1);
    let r = run("finite_section_transfer", &s, m.as_ref(), &i, 128);
    assert_eq!(r.outcome, "unresolved");
    assert!(r.rows.is_empty() && r.values.is_empty());
    assert!(r.reason.unwrap().contains("4096"));
}
#[test]
fn action_consistency_keeps_higher_precision_external_points_distinct() {
    let (s, sm, _) = state(
        "9",
        &["0".into(), "1".into(), "0".into()],
        "3.0000000000000000000000000000000000000001",
        None,
    );
    let mut i = input(&sm, "9", 1);
    i.precision_bits = 256;
    let value = "1.000000000000000000000000000000000000000000000000000000000001";
    i.run_once = Some(RunOnceInputs {
        component_actions: vec![action(
            "tau_prime_reconstructed",
            vec!["0".into(), "1".into(), "0".into()],
        )],
        completion: Some(CompletionInputs {
            independent_actions: vec![action(
                "tau_prime_direct",
                vec!["0".into(), value.into(), "0".into()],
            )],
            ..Default::default()
        }),
        ..Default::default()
    });
    let expected = Float::with_val(768, Float::with_val(256, Float::parse(value).unwrap())) - 1u32;
    assert!(expected > 0);
    let r = run("consistency", &s, None, &i, 256);
    for name in ["action_difference_norm", "signed_energy_difference"] {
        within(&r.rows[0].values, name, &expected.to_string(), 256);
        assert!(hp(&r.rows[0].values[name]) > 0);
    }
}
#[test]
fn cluster_checkpoints_bind_declared_column_precision() {
    if let Ok(bits) = std::env::var("XC_TEST_CLUSTER_POINT_PRECISION") {
        let entries = vec!["3", "1", "1", "1", "3", "1", "1", "1", "3"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let (s, sm, m) = state(
            "9",
            &["1".into(), "1".into(), "1".into()],
            "3",
            Some(&entries),
        );
        let mut i = input(&sm, "9", 1);
        i.precision_bits = bits.parse().unwrap();
        i.run_once = Some(RunOnceInputs {
            reference_vectors: vec![vec![
                "1".into(),
                "1.0000000000000000000000000000000000000001".into(),
                "1".into(),
            ]],
            ..Default::default()
        });
        let r = run("operator_cluster", &s, m.as_ref(), &i, 256);
        assert_eq!(r.outcome, "point_measurement");
        std::fs::write(
            std::env::var_os("XC_TEST_CLUSTER_RESULT").unwrap(),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        return;
    }
    let root = std::env::temp_dir().join(format!(
        "xc-cluster-point-identity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let warm = root.join("warm");
    let cold = root.join("cold");
    let execute = |bits: &str, dir: &std::path::Path, name: &str| {
        let output = root.join(name);
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "cluster_checkpoints_bind_declared_column_precision",
                "--nocapture",
            ])
            .env("XC_TEST_CLUSTER_POINT_PRECISION", bits)
            .env("XC_RESEARCH_CHECKPOINT_DIR", dir)
            .env("XC_TEST_CLUSTER_RESULT", &output)
            .env("XC_CACHE_REMOTE", "none")
            .env("XC_PUBLISH_EXECUTE", "false")
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        std::fs::read(output).unwrap()
    };
    execute("128", &warm, "first.json");
    let before = std::fs::read_dir(&warm).unwrap().count();
    assert!(before > 0);
    let warmed = execute("256", &warm, "warm.json");
    let after = std::fs::read_dir(&warm).unwrap().count();
    assert!(
        after > before,
        "changing original point precision must change the cluster checkpoint identity"
    );
    let fresh = execute("256", &cold, "cold.json");
    assert_eq!(
        warmed, fresh,
        "warm checkpoint result differs from fresh calculation on the same declared points"
    );
    std::fs::remove_dir_all(root).unwrap();
}
