#![cfg(feature = "hp")]
//! Extended research producers and atom research against independent oracles.
//!
//! Expected values come from `fixtures/directional_response/oracles.json`, produced
//! by mpmath quadrature of the finite function itself and exact rationals. The
//! oracle shares no code with the toolkit; it shares only the
//! documented conventions (STATE_GEOMETRY.md, EXTENDED_RESEARCH.md, ATOM_RESEARCH.md).
//! Fixtures and immutable-source hashes are retained under fixtures/directional_response.
use rug::Float;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, retained_evidence::*, state_geometry::RetainedState};

const ORACLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/directional_response/oracles.json"
);
const P: u32 = 512;

fn oracle() -> Value {
    serde_json::from_str(&std::fs::read_to_string(ORACLE).expect("directional oracle fixture"))
        .unwrap()
}
fn fl(s: &str) -> Float {
    Float::with_val(P, Float::parse(s).unwrap_or_else(|e| panic!("{s}: {e}")))
}

fn close(actual: &str, expected: &str, rel: f64, what: &str) {
    let a = fl(actual);
    let e = fl(expected);
    // Pure relative error for nonzero expectations (absolute only for an exact zero).
    let scale = if e.is_zero() {
        Float::with_val(P, 1)
    } else {
        Float::with_val(P, e.clone().abs())
    };
    let err = Float::with_val(P, &a - &e).abs() / scale;
    assert!(
        err <= rel,
        "{what}: actual {actual} expected {expected} relative error {}",
        err.to_f64()
    );
}

fn source(kind: &str, value: Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    (
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(kind, "r15-audit-fixture", &bytes).unwrap(),
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
fn dep(m: &ArtifactManifest) -> DependencyRef {
    DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    }
}
struct Fixture {
    state: RetainedState,
    manifest: ArtifactManifest,
    matrix: Option<RetainedMatrix<'static>>,
}
fn fixture(xi: &[&str], eigen: &str, entries: Option<Vec<String>>) -> Fixture {
    let modes = xi.len() / 2;
    let matrix = entries.map(|e| {
        source(
            "ccm_tau_matrix",
            json!({"schema_version":2,"lambda_squared":"9","n_modes":modes,"precision_bits":128,"entries":e}),
        )
    });
    let (mut m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":modes,"precision_bits":128,"force_even":true,"eigenvalue":eigen,"eigenvector":xi}),
    );
    if let Some((mm, _)) = &matrix {
        m.dependencies.push(dep(mm));
    }
    let state =
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let matrix = matrix.map(|(mm, mb)| {
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap()
    });
    Fixture {
        state,
        manifest: m,
        matrix,
    }
}
fn roots_for(fx: &Fixture, values: &[&str]) -> RetainedRoots {
    let (mut sec, secbytes) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":2,"precision_bits":128,"force_even":true,"eigenpair_content_digest":fx.manifest.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(dep(&fx.manifest));
    let outcomes: Vec<Value> = values
        .iter()
        .map(|v| json!({"status":"converged","details":{"value":v}}))
        .collect();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":2,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":outcomes}),
    );
    rm.dependencies.push(dep(&sec));
    RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &secbytes,
        &fx.state,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap()
}
fn base_inputs(m: &ArtifactManifest) -> Value {
    json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":2,"precision_bits":128,"convention_id":"r15 manufactured values","definition_digest":ContentDigest::sha256(b"r15 manufactured definition"),"approximation_scope":"finite manufactured points"})
}
fn inputs(v: Value) -> ExternalResearchInputs {
    serde_json::from_value(v).unwrap()
}
fn options(fx: &Fixture) -> ExtensionOptions {
    ExtensionOptions::for_source(&fx.state)
}
fn run(
    id: &str,
    fx: &Fixture,
    roots: Option<&RetainedRoots>,
    i: Option<&ExternalResearchInputs>,
    o: &ExtensionOptions,
) -> Result<ExtendedAnalysis, String> {
    capture_extended(
        id,
        &fx.state,
        fx.matrix.as_ref(),
        roots,
        i,
        o,
        &[],
        &context(),
    )
    .map(|r| r.value.data)
    .map_err(|e| format!("{e:#}"))
}
fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn non_displacement_matrix_withholds_root_velocity() {
    let o = oracle();
    let d = &o["directional"];
    let fx = fixture(
        &["1", "1", "2", "1", "1"],
        "0.5",
        Some(strs(&d["A_entries"])),
    );
    let root_t = strs(&d["root_t"]);
    let roots = roots_for(&fx, &[&root_t[0], &root_t[1], "3", "0"]);
    let mut v = base_inputs(&fx.manifest);
    v["perturbations"] = json!([{"label":"B","source_digest":ContentDigest::sha256(b"B"),"diagonal":["1","0","0","0","1"],"dense":[],"rank_one":[]}]);
    v["run_once"] = json!({"derivative_actions":[{"label":"tau_total","source_digest":ContentDigest::sha256(b"Bv"),"action":d["tau_total_action"],"convention":"B applied to oriented unit state"}]});
    let i = inputs(v);
    let mut opt = options(&fx);
    opt.maximum_directional_rows = 4;
    let r = run("directional_response", &fx, Some(&roots), Some(&i), &opt).unwrap();
    for row in &r.rows[..2] {
        assert!(!row.values.contains_key("conditional_tau_response_B"));
        assert_eq!(row.outcome, "channels_resolved_budget_unassessed");
        assert!(row.values.contains_key("displacement_defect"));
        assert!(
            fl(&row.values["displacement_defect_lower"])
                > fl(&row.values["displacement_defect_tolerance_upper"])
        );
    }
}
#[test]
fn genuine_ccm_displacement_keeps_root_velocity() {
    let m: Value = serde_json::from_str(
        &std::fs::read_to_string(ORACLE.replace("oracles.json", "ccm_matrices.json")).unwrap(),
    )
    .unwrap();
    let x: Value = serde_json::from_str(
        &std::fs::read_to_string(ORACLE.replace("oracles.json", "ccm_producer_inputs.json"))
            .unwrap(),
    )
    .unwrap();
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"10","n_modes":6,"precision_bits":256,"entries":m["center"]["entries"]}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"10","n_modes":6,"precision_bits":256,"force_even":true,"eigenvalue":x["E"],"eigenvector":x["v"]}),
    );
    sm.dependencies.push(dep(&mm));
    let state =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let (mut sec, secbytes) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"10","n_modes":6,"precision_bits":256,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(dep(&sm));
    let outcomes: Vec<Value> = x["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| json!({"status":"converged","details":{"value":r["t"]}}))
        .collect();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"10","n_modes":6,"precision_bits":256,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":outcomes}),
    );
    rm.dependencies.push(dep(&sec));
    let roots = RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &secbytes,
        &state,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap();
    let fx = Fixture {
        state,
        manifest: sm.clone(),
        matrix: Some(matrix),
    };
    let i: ExternalResearchInputs = serde_json::from_value(json!({"schema_version":1,"source_eigenpair":sm.content_digest,"lambda_squared":"10","n_modes":6,"precision_bits":256,"convention_id":"r15 genuine ccm","definition_digest":ContentDigest::sha256(b"r15"),"approximation_scope":"finite",
        "perturbations":[{"label":"B","source_digest":ContentDigest::sha256(b"B"),"diagonal":[],"dense":x["B"],"rank_one":[]}]})).unwrap();
    let mut o = options(&fx);
    o.maximum_directional_rows = 6;
    let r = run("directional_response", &fx, Some(&roots), Some(&i), &o).unwrap();
    for (row, e) in r.rows.iter().zip(x["rows"].as_array().unwrap()) {
        println!(
            "genuine CCM root {} outcome {} response {:?} exact {} rule |root| {:?} tol {:?}",
            row.ordinal,
            row.outcome,
            row.values.get("conditional_tau_response_B"),
            e["exact_first_order"],
            row.values.get("rational_root_condition"),
            row.values.get("root_condition_tolerance")
        );
        close(
            &row.values["conditional_tau_response_B"],
            e["exact_first_order"].as_str().unwrap(),
            1e-30,
            "genuine CCM response",
        );
    }
}
#[test]
fn zero_moment_returns_unresolved_report() {
    let fx = fixture(&["-2", "0.5", "0", "0.5", "-2"], "1", None);
    let r = run("compactness", &fx, None, None, &options(&fx));
    match &r {
        Ok(r) => println!(
            "exact-zero compactness outcome {} reason {:?} values {:?}",
            r.outcome, r.reason, r.values
        ),
        Err(e) => println!("exact-zero compactness ERROR: {e}"),
    }
    assert_eq!(r.unwrap().outcome, "unresolved");
}
