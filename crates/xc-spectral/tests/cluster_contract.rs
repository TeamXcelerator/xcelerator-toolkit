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
fn f(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn fixture(vector: Vec<String>) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":vector.len()/2,"precision_bits":128,"force_even":false,"eigenvalue":"3","eigenvector":vector}),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    (s, m)
}
fn inputs(
    m: &ArtifactManifest,
    n: usize,
    current: Vec<Vec<String>>,
    previous: Vec<Vec<String>>,
) -> ExternalResearchInputs {
    let mut i:ExternalResearchInputs=serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":n,"precision_bits":128,"convention_id":"independent finite cluster fixture","definition_digest":ContentDigest::sha256(b"exact rational cluster references"),"approximation_scope":"finite stored coefficient points only"})).unwrap();
    let make = |v: Vec<String>| ClusterVector {
        source_digest: ContentDigest::sha256(&serde_json::to_vec(&v).unwrap()),
        n_modes: v.len() / 2,
        precision_bits: 128,
        eigenvalue: "3".into(),
        coefficients: v,
        assembly_policy: "synthetic finite Fourier fixture".into(),
    };
    i.cluster = current.into_iter().map(make).collect();
    i.previous_cluster = previous.into_iter().map(make).collect();
    i
}
fn run(s: &RetainedState, i: &ExternalResearchInputs, o: &ExtensionOptions) -> ExtendedAnalysis {
    let r = capture_extended(
        "spectral_cluster",
        s,
        None,
        None,
        Some(i),
        o,
        &[],
        &context(),
    )
    .unwrap()
    .value;
    assert_eq!(
        r.request["cluster_arithmetic"],
        "stored_points_scaled_unit_checked_gram_v3"
    );
    r.data
}
#[test]
fn public_cluster_geometry_matches_independent_rational_projections_across_scales() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/cluster_oracle.json")).unwrap();
    let mut comparisons = 0;
    for row in fixture["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for (scale, sign) in [
                (0i32, 1i32),
                (0, -1),
                (600_000_000, 1),
                (-600_000_000, 1),
                (-600_000_000, -1),
            ] {
                let vec = |a: &serde_json::Value, e: i32, s: i32| {
                    a.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| {
                            let mut x = f(x.as_str().unwrap(), 128) * s;
                            if e >= 0 {
                                x <<= e as u32;
                            } else {
                                x >>= e.unsigned_abs();
                            }
                            x.to_string_radix(10, None)
                        })
                        .collect::<Vec<_>>()
                };
                let vector = vec(&row["source"], scale, sign);
                let n = vector.len() / 2;
                let (s, m) = self::fixture(vector);
                let current = row["current"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(j, a)| {
                        vec(
                            a,
                            if j % 2 == 0 { -scale } else { scale },
                            if j % 2 == 0 { 1 } else { -1 },
                        )
                    })
                    .collect::<Vec<_>>();
                let count = current.len();
                let previous = row["previous"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(j, a)| {
                        vec(
                            a,
                            if j % 2 == 0 { scale } else { -scale },
                            if j % 2 == 0 { -1 } else { 1 },
                        )
                    })
                    .collect();
                let i = inputs(&m, n, current, previous);
                let mut o = ExtensionOptions::for_source(&s);
                o.working_precision_bits = p;
                let r = run(&s, &i, &o);
                assert_eq!(r.outcome, "point_measurement");
                let near = |actual: &str, reference: &str, sgn: i32| {
                    let expected = f(reference, p) * sgn;
                    let error = (f(actual, p) - &expected).abs();
                    let tolerance = (Float::with_val(p, expected.abs()) + 1u32) >> (p - 40);
                    assert!(
                        error <= tolerance,
                        "p={p}, case {}, {} versus {}",
                        row["case"],
                        actual,
                        reference
                    );
                };
                near(
                    &r.values["source_cluster_leakage_squared"],
                    row["leakage"].as_str().unwrap(),
                    1,
                );
                for j in 0..count {
                    let signj = if j % 2 == 0 { 1 } else { -1 };
                    near(
                        &r.values[&format!("source_overlap_{j}")],
                        row["overlaps"][j].as_str().unwrap(),
                        sign * signj,
                    );
                    for k in 0..count {
                        near(
                            &r.values[&format!("gram_{j}_{k}")],
                            row["gram"][j * count + k].as_str().unwrap(),
                            signj * if k % 2 == 0 { 1 } else { -1 },
                        );
                    }
                    for (k, reference) in row["previous_overlaps"][j]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        near(
                            &r.rows[j].values[&format!("previous_overlap_{k}")],
                            reference.as_str().unwrap(),
                            signj * if k % 2 == 0 { -1 } else { 1 },
                        );
                    }
                }
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 600);
}
#[test]
fn cluster_extreme_scale_and_squared_leakage_counterexamples_are_repaired() {
    for h in ["1", "-1", "1e200000000", "1e-200000000"] {
        let (s, m) = fixture([h, "0", "0"].map(String::from).to_vec());
        let i = inputs(
            &m,
            1,
            vec![["0", "1", "0"].map(String::from).to_vec()],
            vec![],
        );
        let o = ExtensionOptions::for_source(&s);
        let r = run(&s, &i, &o);
        assert_eq!(f(&r.values["source_cluster_leakage_squared"], 192), 1);
        let (s, m) = fixture(["0", "1", "0"].map(String::from).to_vec());
        let i = inputs(
            &m,
            1,
            vec![["0", "1", "0"].map(String::from).to_vec()],
            vec![["0", h, "0"].map(String::from).to_vec()],
        );
        let r = run(&s, &i, &ExtensionOptions::for_source(&s));
        assert_eq!(f(&r.rows[0].values["largest_previous_overlap"], 192), 1);
    }
    let (s, m) = fixture(["1e-200000000", "1", "0"].map(String::from).to_vec());
    let i = inputs(
        &m,
        1,
        vec![["0", "1", "0"].map(String::from).to_vec()],
        vec![],
    );
    assert!(capture_extended(
        "spectral_cluster",
        &s,
        None,
        None,
        Some(&i),
        &ExtensionOptions::for_source(&s),
        &[],
        &context()
    )
    .is_err());
}
#[test]
fn cluster_source_precision_rank_ties_and_budgets_are_explicit() {
    let (s, m) = fixture(["0", "1", "0"].map(String::from).to_vec());
    let vector = ["0", "1", "0"].map(String::from).to_vec();
    let mut i = inputs(
        &m,
        1,
        vec![vector.clone()],
        vec![vector.clone(), ["0", "-1", "0"].map(String::from).to_vec()],
    );
    i.cluster[0].precision_bits = 64;
    i.cluster[0].eigenvalue = "0.1".into();
    i.cluster_boundary_eigenvalues = Some([
        "1".into(),
        "1.0000000000000000000000000000000000000001".into(),
    ]);
    let o = ExtensionOptions::for_source(&s);
    let r = run(&s, &i, &o);
    assert_eq!(
        f(&r.rows[0].values["eigenvalue"], 192),
        Float::with_val(192, f("0.1", 64))
    );
    assert_eq!(f(&r.values["declared_boundary_gap"], 192), 0);
    assert_eq!(f(&r.rows[0].values["match_margin"], 192), 0);
    assert_eq!(f(&r.rows[0].values["best_previous_index"], 192), 0);
    assert!(r.rows[0].notes.iter().any(|s| s.contains("overlap tie")));
    i.cluster.push(i.cluster[0].clone());
    let r = run(&s, &i, &o);
    assert_eq!(r.outcome, "rank_or_precision_unresolved");
    assert!(!r.values.contains_key("source_cluster_leakage_squared"));
    let mut o = o;
    o.maximum_working_bytes = Some(1);
    assert!(run(&s, &i, &o).reason.unwrap().contains("scratch estimate"));
}

#[test]
fn cluster_precision_escalates_for_nearly_parallel_columns_and_reports_exhaustion() {
    let (s, m) = fixture(["1", "2", "3"].map(String::from).to_vec());
    let mut escalated = false;
    for p in [128, 192, 256] {
        for exponent in [20u32, 60, 80, 100, 110] {
            for slope in [1, 3] {
                let tiny = Float::with_val(128, 1) >> exponent;
                let basis = vec![
                    vec![
                        "1".into(),
                        slope.to_string(),
                        tiny.to_string_radix(10, None),
                    ],
                    vec![
                        "2".into(),
                        (2 * slope).to_string(),
                        (tiny * 3u32).to_string_radix(10, None),
                    ],
                ];
                let i = inputs(&m, 1, basis, vec![]);
                let mut o = ExtensionOptions::for_source(&s);
                o.working_precision_bits = p;
                let r = run(&s, &i, &o);
                assert_eq!(r.outcome, "point_measurement");
                escalated |= f(&r.values["arithmetic_precision_bits"], p) > p + 64;
                let expected = Float::with_val(p, 1) / (if slope == 1 { 28u32 } else { 140u32 });
                let error = (f(&r.values["source_cluster_leakage_squared"], p) - expected).abs();
                assert!(
                    error <= Float::with_val(p, 1) >> (p - 40),
                    "p={p}, exponent={exponent}, slope={slope}, error={error}"
                );
            }
        }
    }
    assert!(escalated);
    let tiny = Float::with_val(128, 1) >> 2500u32;
    let i = inputs(
        &m,
        1,
        vec![
            vec!["1".into(), "1".into(), tiny.to_string_radix(10, None)],
            vec![
                "2".into(),
                "2".into(),
                (tiny * 3u32).to_string_radix(10, None),
            ],
        ],
        vec![],
    );
    let r = run(&s, &i, &ExtensionOptions::for_source(&s));
    assert_eq!(r.outcome, "rank_or_precision_unresolved");
    assert!(!r.values.contains_key("source_cluster_leakage_squared"));
    assert_eq!(f(&r.values["arithmetic_precision_bits"], 192), 192 + 4096);
}
