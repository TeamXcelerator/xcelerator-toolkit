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
fn fixture() -> (RetainedState, ArtifactManifest, RetainedMatrix<'static>) {
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["4","0","0","0","3","0","0","0","4"]}),
    );
    let (mut m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    m.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    (s, m, matrix)
}
fn inputs(m: &ArtifactManifest) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":128,"convention_id":"synthetic test values","definition_digest":ContentDigest::sha256(b"synthetic definition"),"approximation_scope":"finite test points"})).unwrap()
}
fn run(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    i: Option<&ExternalResearchInputs>,
) -> ExtendedAnalysis {
    capture_extended(
        id,
        s,
        m,
        None,
        i,
        &ExtensionOptions::for_source(s),
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
}

#[test]
fn arithmetic_energy_original_counterexamples_are_repaired() {
    for case in [
        "component_cancellation",
        "declared_component_point",
        "tiny_trial_norm",
    ] {
        let (s, manifest, matrix) = fixture();
        let mut input = inputs(&manifest);
        let big = (Float::with_val(128, 1) << 400u32).to_string();
        let diagonals = if case == "component_cancellation" {
            vec![
                vec!["0".to_string(), big.clone(), "0".into()],
                vec!["4".into(), "3".into(), "4".into()],
                vec!["0".into(), format!("-{big}"), "0".into()],
            ]
        } else if case == "declared_component_point" {
            vec![vec![
                "4".into(),
                "3.0000000000000000000000000000000000000001".into(),
                "4".into(),
            ]]
        } else {
            vec![vec!["4".into(), "3".into(), "4".into()]]
        };
        for (j, diagonal) in diagonals.into_iter().enumerate() {
            let label = format!("component_{j}");
            input.components.push(OperatorComponent {
                label: label.clone(),
                source_digest: ContentDigest::sha256(label.as_bytes()),
                diagonal,
                dense: vec![],
                rank_one: vec![],
            });
        }
        input.components_are_complete = true;
        if case == "tiny_trial_norm" {
            input.target = Some(SampledReference {
                definition_digest: ContentDigest::sha256(b"scaled constant trial"),
                evaluation_policy: "explicit finite uniform points".into(),
                approximation_scope: "finite trial".into(),
                intervals: 8,
                values: vec!["1".into(); 9],
                basis_values: vec![],
                fixed_second_component: None,
                raw_normalizer: "1".into(),
                trial_coefficients: Some(vec!["0".into(), "1e-200000000".into(), "0".into()]),
            });
        }
        let report = run("arithmetic_energy", &s, Some(&matrix), Some(&input));
        assert_eq!(report.outcome, "point_measurement");
        assert_eq!(number(&report.values["sum_component_energy"]), 3.0);
        assert_eq!(number(&report.values["energy_closure_defect"]), 0.0);
        assert_eq!(number(&report.values["operator_action_closure_norm"]), 0.0);
        if case == "tiny_trial_norm" {
            assert_eq!(number(&report.values["finite_projected_trial_energy"]), 3.0);
        }
    }
}

fn number(s: &str) -> f64 {
    Float::with_val(512, Float::parse(s).unwrap()).to_f64()
}
fn within(map: &BTreeMap<String, String>, name: &str, expected: &str, p: u32) {
    let reference = Float::with_val(512, Float::parse(expected).unwrap());
    let low = Float::with_val(512, Float::parse(&map[&format!("{name}_lower")]).unwrap());
    let high = Float::with_val(512, Float::parse(&map[&format!("{name}_upper")]).unwrap());
    assert!(
        low <= reference && reference <= high,
        "{name}, p={p}: [{low},{high}] excludes {reference}"
    );
    let actual = Float::with_val(512, Float::parse(&map[name]).unwrap());
    let tolerance = (Float::with_val(512, 1) + reference.clone().abs()) >> (p - 8);
    assert!(
        (actual - reference).abs() <= tolerance,
        "midpoint {name} failed p={p}"
    );
}
#[test]
fn public_energy_matches_independent_exact_matrix_quotients_and_residuals() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/arithmetic_energy_oracle.json")).unwrap();
    let mut compared = 0;
    for case in oracle["cases"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for (sign, shift) in [(1i32, 0i32), (1, 400), (-1, -400)] {
                let scaled = |values: &serde_json::Value| {
                    values
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| {
                            let mut x =
                                Float::with_val(128, Float::parse(v.as_str().unwrap()).unwrap());
                            x *= sign;
                            if shift >= 0 {
                                x <<= shift as u32;
                            } else {
                                x >>= shift.unsigned_abs();
                            }
                            x.to_string()
                        })
                        .collect::<Vec<_>>()
                };
                let n = case["N"].as_u64().unwrap() as usize;
                let (mm, mb) = source(
                    "ccm_tau_matrix",
                    json!({"schema_version":2,"lambda_squared":"9","n_modes":n,"precision_bits":128,"entries":case["matrix"]}),
                );
                let (mut sm, sb) = source(
                    "ccm_weil_eigenpair",
                    json!({"schema_version":3,"lambda_squared":"9","n_modes":n,"precision_bits":128,"force_even":true,"eigenvalue":case["eigenvalue"],"eigenvector":scaled(&case["coefficients"])}),
                );
                sm.dependencies.push(DependencyRef {
                    key: mm.key.clone(),
                    content_digest: mm.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                });
                let state =
                    RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
                        .unwrap();
                let matrix = RetainedMatrix::from_payload(
                    &mm,
                    &mb,
                    std::slice::from_ref(&mm.content_digest),
                )
                .unwrap();
                let components = case["components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(k, c)| {
                        let mut c = c.clone();
                        c["label"] = json!(format!("component_{k}"));
                        c["source_digest"] =
                            json!(ContentDigest::sha256(format!("component_{k}").as_bytes()));
                        c
                    })
                    .collect::<Vec<_>>();
                let input:ExternalResearchInputs=serde_json::from_value(json!({"schema_version":1,"source_eigenpair":sm.content_digest,"lambda_squared":"9","n_modes":n,"precision_bits":128,"convention_id":"exact rational finite matrix fixture","definition_digest":ContentDigest::sha256(b"independent quadratic form oracle"),"approximation_scope":"finite stored points","components":components,"components_are_complete":case["components_are_complete"],"deficit":case["deficit"],"deficit_kind":"exact_source","target":{"definition_digest":ContentDigest::sha256(b"independent finite trial"),"evaluation_policy":"uniform samples","approximation_scope":"finite trial","intervals":8,"values":vec!["1";9],"basis_values":[],"fixed_second_component":null,"raw_normalizer":"1","trial_coefficients":scaled(&case["trial_coefficients"])}})).unwrap();
                let mut options = ExtensionOptions::for_source(&state);
                options.working_precision_bits = p;
                let report = capture_extended(
                    "arithmetic_energy",
                    &state,
                    Some(&matrix),
                    None,
                    Some(&input),
                    &options,
                    &[],
                    &context(),
                )
                .unwrap();
                assert_eq!(
                    report.value.data.outcome, "point_measurement",
                    "case {} p={p}: {:?}",
                    case["case"], report.value.data.reason
                );
                for (name, expected) in case["expected"].as_object().unwrap() {
                    within(
                        &report.value.data.values,
                        name,
                        expected.as_str().unwrap(),
                        p,
                    );
                    compared += 1;
                }
                for (row, expected) in report
                    .value
                    .data
                    .rows
                    .iter()
                    .zip(case["component_energies"].as_array().unwrap())
                {
                    within(&row.values, "energy", expected.as_str().unwrap(), p);
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 3465);
}
#[test]
fn arithmetic_energy_limits_zero_trials_and_compact_actions_are_explicit() {
    let (s, manifest, matrix) = fixture();
    let mut input = inputs(&manifest);
    input.components.push(OperatorComponent {
        label: "tau".into(),
        source_digest: ContentDigest::sha256(b"tau"),
        diagonal: vec!["4".into(), "3".into(), "4".into()],
        dense: vec![],
        rank_one: vec![],
    });
    let mut options = ExtensionOptions::for_source(&s);
    options.maximum_working_bytes = Some(1);
    let result = capture_extended(
        "arithmetic_energy",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(result.value.data.outcome, "unresolved");
    assert!(result.value.data.values.is_empty());
    options.maximum_working_bytes = None;
    input.target = Some(SampledReference {
        definition_digest: ContentDigest::sha256(b"zero trial"),
        evaluation_policy: "finite samples".into(),
        approximation_scope: "exact zero trial".into(),
        intervals: 8,
        values: vec!["1".into(); 9],
        basis_values: vec![],
        fixed_second_component: None,
        raw_normalizer: "1".into(),
        trial_coefficients: Some(vec!["0".into(); 3]),
    });
    let result = capture_extended(
        "arithmetic_energy",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(result.value.data.outcome, "partial_unresolved");
    assert!(result.value.data.reason.unwrap().contains("zero trial"));
    input.target = None;
    let big = (Float::with_val(128, 1) << 10000u32).to_string();
    for (label, value) in [("large", big.clone()), ("negative", format!("-{big}"))] {
        input.components.push(OperatorComponent {
            label: label.into(),
            source_digest: ContentDigest::sha256(label.as_bytes()),
            diagonal: vec![value; 3],
            dense: vec![],
            rank_one: vec![],
        });
    }
    // Order large, small, negative large forces unresolved interval cancellation.
    input.components.swap(0, 1);
    let result = capture_extended(
        "arithmetic_energy",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(result.value.data.outcome, "unresolved");
    assert!(result.value.data.reason.unwrap().contains("4096"));
    input.components.clear();
    input.run_once=Some(serde_json::from_value(json!({"component_actions":[{"label":"retained_action","source_digest":ContentDigest::sha256(b"exact action"),"action":["0","3","0"],"convention":"center-oriented unit coefficient state"}]})).unwrap());
    let result = capture_extended(
        "arithmetic_energy",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(result.value.data.outcome, "point_measurement");
    assert_eq!(number(&result.value.data.values["total_tau_energy"]), 3.0);
    assert_eq!(
        number(&result.value.data.values["operator_action_closure_norm"]),
        0.0
    );
}
