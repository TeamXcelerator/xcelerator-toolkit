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
fn state(cutoff: &str, coefficients: Vec<String>, p: u32) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({
            "schema_version":3,"lambda_squared":cutoff,"n_modes":coefficients.len()/2,
            "precision_bits":p,"force_even":false,"eigenvalue":"3","eigenvector":coefficients
        }),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    (s, m)
}
fn f(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn run(s: &RetainedState, o: &ExtensionOptions) -> ExtendedAnalysis {
    let report = capture_extended("compactness", s, None, None, None, o, &[], &context())
        .unwrap()
        .value;
    assert_eq!(
        report.request["compactness_arithmetic"],
        "directed_enclosure_agreed_rounding_or_unresolved_v2"
    );
    assert_eq!(
        report.request["source_unit_arithmetic"],
        "binary_scaled_hypot_checked_range_v2"
    );
    report.data
}
#[test]
fn public_compactness_matches_independent_integrals_across_precisions_and_scales() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/compactness_oracle.json")).unwrap();
    let mut measurements = 0;
    let mut escalated = false;
    for row in fixture["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for (e, sign) in [
                (0i32, 1i32),
                (0, -1),
                (600_000_000, 1),
                (-600_000_000, 1),
                (-600_000_000, -1),
            ] {
                let coefficients = row["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        let mut x = f(x.as_str().unwrap(), 128) * sign;
                        if e >= 0 {
                            x <<= e as u32;
                        } else {
                            x >>= e.unsigned_abs();
                        }
                        x.to_string_radix(10, None)
                    })
                    .collect();
                let (s, _) = state(row["cutoff"].as_str().unwrap(), coefficients, 128);
                let mut o = ExtensionOptions::for_source(&s);
                o.working_precision_bits = p;
                o.exponential_rates = row["rates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().to_owned())
                    .collect();
                let r = run(&s, &o);
                escalated |= f(&r.values["arithmetic_precision_bits"], p) > p + 64;
                for (actual, expected) in [
                    ("transform_origin", "origin"),
                    ("transform_second_derivative", "second"),
                    ("transform_fourth_derivative", "fourth"),
                ] {
                    assert_eq!(
                        f(&r.values[actual], p),
                        f(row[expected].as_str().unwrap(), p),
                        "{actual}, p={p}, profile={}",
                        row["profile"]
                    );
                }
                if let Some(sigma) = row["sigma"].as_str() {
                    assert_eq!(f(&r.values["sigma"], p), f(sigma, p));
                    assert_eq!(r.outcome, "point_measurement");
                } else {
                    assert!(!r.values.contains_key("sigma"));
                    assert_eq!(r.outcome, "partial_unresolved");
                    assert!(r.reason.as_ref().unwrap().contains("exactly zero"));
                }
                for (i, weighted) in r.rows.iter().enumerate() {
                    let value = f(&weighted.values["analytic_integral"], p);
                    assert_eq!(value, f(row["weighted"][i].as_str().unwrap(), p));
                    assert!(value >= 1);
                    assert!(f(&weighted.values["sum_absolute_terms"], p) >= value);
                    assert_eq!(
                        f(&weighted.values["rate"], p),
                        f(&o.exponential_rates[i], p)
                    );
                }
                measurements += 1;
            }
        }
    }
    assert_eq!(measurements, 135);
    assert!(escalated);
}
#[test]
fn compactness_handles_constant_limits_exact_cancellation_and_explicit_limits() {
    for coefficient in ["1", "-5", "1e200000000", "1e-200000000"] {
        let (s, _) = state("9", vec!["0".into(), coefficient.into(), "0".into()], 128);
        let mut o = ExtensionOptions::for_source(&s);
        o.exponential_rates = ["0", "1e-100", "1e-200000000"].map(String::from).to_vec();
        let r = run(&s, &o);
        assert_eq!(
            f(&r.values["transform_origin"], 192),
            Float::with_val(192, Float::with_val(448, 9).ln().sqrt())
        );
        for row in r.rows {
            assert_eq!(f(&row.values["analytic_integral"], 192), 1);
        }
        o.maximum_working_bytes = Some(1);
        let r = run(&s, &o);
        assert!(r.reason.unwrap().contains("scratch estimate"));
        assert!(r.rows.is_empty());
        o.maximum_working_bytes = None;
        for rate in ["-1", "101", "NaN", "inf", "1e-400000000"] {
            o.exponential_rates = vec![rate.into()];
            assert!(
                capture_extended("compactness", &s, None, None, None, &o, &[], &context()).is_err()
            );
        }
    }
    let (s, _) = state(
        "9",
        ["1e100", "-1", "-2e100", "-1", "1e100"]
            .map(String::from)
            .to_vec(),
        128,
    );
    let mut o = ExtensionOptions::for_source(&s);
    o.exponential_rates = vec!["0".into()];
    let r = run(&s, &o);
    assert!(f(&r.values["transform_origin"], 192) < 0);
    assert_eq!(f(&r.rows[0].values["analytic_integral"], 192), 1);
    let (s, _) = state("9", vec!["1".into()], 1_000_000);
    assert_eq!(
        ExtensionOptions::for_source(&s).working_precision_bits,
        1_000_000
    );
}
#[test]
fn shared_source_normalization_preserves_complete_component_energy_at_extreme_scale() {
    for coefficient in ["1", "-5", "1e200000000", "1e-200000000"] {
        let (mm, mb) = source(
            "ccm_tau_matrix",
            json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["3","0","0","0","3","0","0","0","3"]}),
        );
        let (mut sm, sb) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0",coefficient,"0"]}),
        );
        sm.dependencies.push(DependencyRef {
            key: mm.key.clone(),
            content_digest: mm.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        });
        let s = RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
            .unwrap();
        let matrix =
            RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest))
                .unwrap();
        let mut input:ExternalResearchInputs=serde_json::from_value(json!({"schema_version":1,"source_eigenpair":sm.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":128,"convention_id":"synthetic complete operator fixture","definition_digest":ContentDigest::sha256(b"compactness shared normalization fixture"),"approximation_scope":"finite test points"})).unwrap();
        for (label, value) in [("positive", "5"), ("negative", "-2")] {
            input.components.push(OperatorComponent {
                label: label.into(),
                source_digest: ContentDigest::sha256(label.as_bytes()),
                diagonal: vec![value.into(); 3],
                dense: vec![],
                rank_one: vec![],
            });
        }
        input.components_are_complete = true;
        let r = capture_extended(
            "arithmetic_energy",
            &s,
            Some(&matrix),
            None,
            Some(&input),
            &ExtensionOptions::for_source(&s),
            &[],
            &context(),
        )
        .unwrap()
        .value
        .data;
        for (name, value) in [
            ("sum_component_energy", 3),
            ("sum_absolute_component_energy", 7),
            ("energy_closure_defect", 0),
        ] {
            assert_eq!(
                f(&r.values[name], 192),
                value,
                "{name}, scale {coefficient}"
            );
        }
    }
}
#[test]
fn working_budget_is_not_identity_and_budget_limited_results_are_not_retained() {
    let root = xc_core::test_support::TestDir::new("budget-identity");
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(ZipJsonFilesystemCacheStore::new(
            "local",
            &*root,
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let cache = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let (s, m) = state("9", vec!["0".into(), "1".into(), "0".into()], 128);
    let sources = [m];
    let mut o = ExtensionOptions::for_source(&s);
    o.exponential_rates = vec!["0".into()];
    let capture = |o: &ExtensionOptions| {
        capture_extended("compactness", &s, None, None, None, o, &sources, &cache).unwrap()
    };

    // A budget-limited result is returned but not retained.
    o.maximum_working_bytes = Some(1);
    let limited = capture(&o);
    assert!(limited
        .value
        .data
        .reason
        .as_deref()
        .unwrap()
        .contains("working-byte budget"));
    assert!(limited.produced_manifest.is_none() && limited.reused_manifest.is_none());

    // A larger budget computes and retains the complete result.
    o.maximum_working_bytes = Some(1 << 40);
    let complete = capture(&o);
    assert!(!complete.value.data.rows.is_empty());
    let produced = complete
        .produced_manifest
        .expect("complete result is retained");

    // A different budget reuses the same complete result.
    o.maximum_working_bytes = Some(1 << 41);
    let reused = capture(&o);
    assert_eq!(
        reused.reused_manifest.expect("reused").content_digest,
        produced.content_digest
    );
    assert_eq!(reused.value.data.rows.len(), complete.value.data.rows.len());
}
