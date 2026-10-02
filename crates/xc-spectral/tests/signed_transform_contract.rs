#![cfg(feature = "hp")]

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

fn parsed(text: &str, p: u32) -> rug::Float {
    rug::Float::with_val(p, rug::Float::parse(text).unwrap())
}
fn input_for(
    sm: &ArtifactManifest,
    n: usize,
    cutoff: &str,
    precision: u32,
    jets: serde_json::Value,
) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":sm.content_digest,"lambda_squared":cutoff,"n_modes":n,"precision_bits":precision,"convention_id":"finite signed-transform test","definition_digest":ContentDigest::sha256(b"signed-transform independent fixture"),"approximation_scope":"finite exact stored input points","reference_jets":jets})).unwrap()
}
fn jet(v: &str) -> serde_json::Value {
    json!({"value":v,"derivative":v})
}
fn reference(
    window: &str,
    full: &str,
    tail: &str,
    fitted: Vec<serde_json::Value>,
) -> serde_json::Value {
    json!({"ordinal":1,"t":"0","reference_window":jet(window),"reference_full":jet(full),"exterior_tail":jet(tail),"endpoint_tail_part":null,"fitted_interior_parts":fitted,"source_value_error":null,"tail_value_error":null,"source_derivative_error":null,"root_separation_radius":null})
}

#[test]
fn declared_points_and_exact_cancellation_are_preserved() {
    let (sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    for p in [128, 192, 256] {
        let mut o = ExtensionOptions::for_source(&s);
        o.working_precision_bits = p;
        for (name, r) in [
            (
                "points",
                reference(
                    "1.0000000000000000000000000000000000000001",
                    "1",
                    "0",
                    vec![],
                ),
            ),
            ("closure", reference("1e100", "1", "-1e100", vec![])),
            (
                "fitted",
                reference("0", "0", "0", vec![jet("1e100"), jet("1"), jet("-1e100")]),
            ),
        ] {
            let input = input_for(&sm, 1, "9", 128, json!([r]));
            let report = capture_extended(
                "signed_transform",
                &s,
                None,
                None,
                Some(&input),
                &o,
                &[],
                &context(),
            )
            .unwrap();
            let v = &report.value.data.rows[0].values;
            if name == "points" {
                assert_eq!(parsed(&v["value_reference_window"], p), 1);
                assert_eq!(parsed(&v["value_reference_closure_defect"], p), 0);
            } else if name == "closure" {
                assert_eq!(parsed(&v["value_reference_closure_defect"], p), 1);
                assert_eq!(parsed(&v["derivative_reference_closure_defect"], p), 1);
                assert_eq!(v["value_signed_total"], v["value_actual"]);
            } else {
                assert_eq!(parsed(&v["derivative_unfitted_interior"], p), -1);
                let expected = parsed(&v["value_actual"], p) - 1u32;
                assert_eq!(parsed(&v["value_unfitted_interior"], p), expected);
            }
        }
    }
}

#[test]
fn direct_fourier_quadrature_and_exact_external_algebra_match() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/signed_transform_oracle.json")).unwrap();
    let mut checked = 0;
    for row in fixture["rows"].as_array().unwrap() {
        let n = row["n_modes"].as_u64().unwrap() as usize;
        let cutoff = row["lambda_squared"].as_str().unwrap();
        for p in [128, 192, 256] {
            for (sign, scale) in [(1i32, 0i32), (1, 400), (-1, -400)] {
                let coefficients: Vec<String> = row["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        let mut value = parsed(v.as_str().unwrap(), 128) * sign;
                        if scale >= 0 {
                            value <<= scale as u32;
                        } else {
                            value >>= (-scale) as u32;
                        }
                        value.to_string_radix(10, None)
                    })
                    .collect();
                let (sm, sb) = source(
                    "ccm_weil_eigenpair",
                    json!({"schema_version":3,"lambda_squared":cutoff,"n_modes":n,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":coefficients}),
                );
                let s =
                    RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
                        .unwrap();
                let parts = row["fitted"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| jet(v.as_str().unwrap()))
                    .collect();
                let mut r = reference(
                    row["window"].as_str().unwrap(),
                    row["full"].as_str().unwrap(),
                    row["tail"].as_str().unwrap(),
                    parts,
                );
                r["t"] = row["t"].clone();
                r["endpoint_tail_part"] = jet(row["endpoint"].as_str().unwrap());
                // 128 bits preserves all exact dyadic external inputs, including
                // the 2^80 +/- small terms. Output precision cannot redefine them.
                let input = input_for(&sm, n, cutoff, 128, json!([r]));
                let mut o = ExtensionOptions::for_source(&s);
                o.working_precision_bits = p;
                let report = capture_extended(
                    "signed_transform",
                    &s,
                    None,
                    None,
                    Some(&input),
                    &o,
                    &[],
                    &context(),
                )
                .unwrap();
                let values = &report.value.data.rows[0].values;
                assert_eq!(
                    parsed(&report.value.data.values["arithmetic_precision_bits"], 64),
                    p
                );
                for (name, expected) in row["expected"].as_object().unwrap() {
                    let expected = parsed(expected.as_str().unwrap(), 768);
                    let actual = parsed(&values[name], 768);
                    let error = (actual - &expected).abs();
                    let bound = (expected.abs() + 1u32) >> (p - 12);
                    assert!(
                        error <= bound,
                        "profile={} p={p} scale={scale} {name} error={error}",
                        row["profile"]
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 48 * 9 * 12);
}

#[test]
fn external_precision_and_output_budget_are_enforced() {
    let (sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let input = input_for(&sm, 1, "9", 512, json!([reference("0", "0", "0", vec![])]));
    let mut options = ExtensionOptions::for_source(&s);
    let refusal = capture_extended(
        "signed_transform",
        &s,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    );
    assert!(refusal
        .err()
        .expect("external precision reduction must fail")
        .to_string()
        .contains("working precision below external source precision"));
    options.working_precision_bits = 512;
    options.maximum_estimated_output_bytes = 18_000;
    let limited = capture_extended(
        "signed_transform",
        &s,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(limited.value.data.outcome, "unresolved");
    assert!(limited.value.data.rows.is_empty());
    options.maximum_estimated_output_bytes = 30_000;
    let resolved = capture_extended(
        "signed_transform",
        &s,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(resolved.value.data.outcome, "point_measurement");
    assert_eq!(resolved.value.data.rows.len(), 1);
    assert_eq!(
        parsed(
            &resolved.value.data.values["normalization_precision_bits"],
            64
        ),
        576
    );
}
#[test]
fn center_one_origin_uses_the_exact_stored_center() {
    let mut failures = 0;
    for bits in [40u32, 60, 100, 120, 350, 1000] {
        let eps = (rug::Float::with_val(128, 1) >> bits).to_string_radix(10, None);
        let (sm, sb) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":3,"lambda_squared":"9","n_modes":3,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["1","3","2",eps,"2","3","1"]}),
        );
        let s = RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
            .unwrap();
        let input = input_for(&sm, 3, "9", 128, json!([reference("0", "0", "0", vec![])]));
        for p in [192u32, 256] {
            let mut options = ExtensionOptions::for_source(&s);
            options.working_precision_bits = p;
            let measured = capture_extended(
                "signed_transform",
                &s,
                None,
                None,
                Some(&input),
                &options,
                &[],
                &context(),
            )
            .unwrap();
            assert_eq!(measured.value.data.outcome, "point_measurement");
            // c = -2+6-4+eps = eps and the Fourier integral at zero is L*eps.
            // Their quotient is exactly L, independent of eps and L2 scaling.
            let actual = parsed(&measured.value.data.rows[0].values["value_actual"], 768);
            let expected = rug::Float::with_val(768, 9).ln();
            let error = (actual - &expected).abs();
            let bound = (expected.abs() + 1u32) >> (p - 16);
            if error > bound {
                failures += 1;
                println!("center-one origin bits={bits}, requested={p}, absolute_error={error}, allowed={bound}");
            }
        }
    }
    assert_eq!(failures, 0, "exact origin identity failed");
}

#[test]
fn zero_stored_center_remains_explicitly_unresolved() {
    let (sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":3,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["1","3","2","0","2","3","1"]}),
    );
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let input = input_for(&sm, 3, "9", 128, json!([reference("0", "0", "0", vec![])]));
    let report = capture_extended(
        "signed_transform",
        &s,
        None,
        None,
        Some(&input),
        &ExtensionOptions::for_source(&s),
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(report.value.data.outcome, "unresolved");
    assert!(report.value.data.rows.is_empty());
    assert!(report
        .value
        .data
        .reason
        .as_ref()
        .unwrap()
        .contains("nonzero exact stored center"));
}

#[test]
fn exhaustive_resumed_signed_constant_origin_uses_exact_cutoff() {
    use rug::Float;
    for cutoff in ["9", "1.000000000000000000000000000001"] {
        let (sm, sb) = source(
            "ccm_weil_eigenpair",
            json!({"schema_version":3,"lambda_squared":cutoff,"n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":["0","1","0"]}),
        );
        let state = RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest))
            .unwrap();
        let input = input_for(
            &sm,
            1,
            cutoff,
            128,
            json!([reference("0", "0", "0", vec![])]),
        );
        let mut options = ExtensionOptions::for_source(&state);
        options.working_precision_bits = 128;
        let report = capture_extended(
            "signed_transform",
            &state,
            None,
            None,
            Some(&input),
            &options,
            &[],
            &context(),
        )
        .unwrap()
        .value
        .data;
        let actual = parsed(&report.rows[0].values["value_actual"], 1024);
        // The center-one constant function has transform at zero exactly log(C).
        let expected = parsed(cutoff, 1024).ln();
        let relative = Float::with_val(1024, &actual - &expected).abs() / &expected;
        assert!(
            relative < Float::with_val(1024, 1) >> 124u32,
            "C={cutoff}: relative error {relative}"
        );
    }
}
#[test]
fn exhaustive_resumed_resolution_encloses_exact_tolerance() {
    let (sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":["0","1","0"]}),
    );
    let state =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let input = input_for(&sm, 1, "9", 128, json!([reference("0", "0", "0", vec![])]));
    let mut options = ExtensionOptions::for_source(&state);
    options.working_precision_bits = 128;
    options.relative_tolerance =
        "0.4999999999999999999999999999999999999999999999999999999999999999".into();
    let report = capture_extended(
        "resolution_budget",
        &state,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    let exact = parsed(&options.relative_tolerance, 1024);
    let lo = parsed(&report.values["relative_tolerance_lower"], 1024);
    let hi = parsed(&report.values["relative_tolerance_upper"], 1024);
    assert!(
        lo <= exact && exact <= hi,
        "reported tolerance bounds do not enclose the supplied decimal"
    );
}
#[test]
fn exhaustive_resumed_geometry_rejects_partial_normalization_underflow() {
    use rug::Float;
    use xc_spectral::ccm::state_geometry::{analyze_state_geometry, GeometryOptions};
    let tiny = Float::with_val(128, 1) << (rug::float::exp_min() - 1);
    let coefficient = (tiny * 2u32).to_string();
    let (sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":2,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":[coefficient,"1.5","1.5","1.5",coefficient]}),
    );
    let state =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    assert!(analyze_state_geometry(&state, &GeometryOptions::for_source(&state)).is_err());
}
