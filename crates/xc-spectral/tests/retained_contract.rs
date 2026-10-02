#![cfg(feature = "hp")]
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{retained_evidence::*, state_geometry::RetainedState};
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
fn reference(coefficients: &[&str]) -> ReferenceSpec {
    ReferenceSpec {
        schema_version: 1,
        definition: "synthetic Fourier test function".into(),
        lambda_squared: "9".into(),
        precision_bits: 128,
        coefficients: coefficients.iter().map(|s| s.to_string()).collect(),
        approximation_scope: "exact finite test function; serialized coefficients are points"
            .into(),
    }
}
fn retained_reference(spec: &ReferenceSpec) -> RetainedReference {
    let value = capture_reference(spec, &context()).unwrap().value;
    let (m, b) = source("ccm_reference_source", serde_json::to_value(value).unwrap());
    RetainedReference::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap()
}
use rug::Float;
fn dataset_at(t: &str, p: u32) -> RetainedDataset {
    let spec = DatasetSpec {
        schema_version: 1,
        role: "evaluation_points".into(),
        attribution: "independent integral test".into(),
        coordinate: "mellin_t".into(),
        precision_bits: p,
        points: vec![EvaluationPoint {
            ordinal: 1,
            value: Some(t.into()),
            source_status: "supplied".into(),
        }],
    };
    let value = capture_dataset(&spec, &context()).unwrap().value;
    let (m, b) = source(
        "research_reference_dataset",
        serde_json::to_value(value).unwrap(),
    );
    RetainedDataset::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap()
}
fn transform_row(coefficients: &[String], cutoff: &str, t: &str, p: u32) -> TransformRow {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":cutoff,"n_modes":(coefficients.len()-1)/2,"precision_bits":128,"force_even":false,"eigenvalue":"1","eigenvector":coefficients}),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    capture_transforms_at_dataset(
        &s,
        &dataset_at(t, p),
        &TransformOptions {
            working_precision_bits: p,
            maximum_rows: 1,
            maximum_estimated_output_bytes: 1000000,
        },
        &context(),
    )
    .unwrap()
    .value
    .data
    .rows
    .remove(0)
}
fn float(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn close(actual: &str, expected: &str, p: u32) {
    let a = float(actual, p + 64);
    let e = float(expected, p + 64);
    let tolerance = (Float::with_val(p + 64, 1) + e.clone().abs()) >> (p - 24);
    assert!(
        (Float::with_val(p + 64, &a) - &e).abs() <= tolerance,
        "p={p}: {a} versus {e}"
    );
}
#[test]
fn defining_integral_and_derivative_match_independent_quadrature_under_binary_scaling() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/retained_transform_oracle.json")).unwrap();
    let mut scalars = 0;
    for row in fixture["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for (exponent, sign) in [
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
                        let mut x = float(x.as_str().unwrap(), 128) * sign;
                        if exponent >= 0 {
                            x <<= exponent as u32;
                        } else {
                            x >>= (-exponent) as u32;
                        }
                        x.to_string()
                    })
                    .collect::<Vec<_>>();
                let r = transform_row(
                    &coefficients,
                    row["cutoff"].as_str().unwrap(),
                    row["t"].as_str().unwrap(),
                    p,
                );
                close(r.value.as_ref().unwrap(), row["value"].as_str().unwrap(), p);
                close(
                    r.derivative.as_ref().unwrap(),
                    row["derivative"].as_str().unwrap(),
                    p,
                );
                scalars += 2;
            }
        }
    }
    assert_eq!(scalars, 2160);
}
#[test]
fn derivative_survives_square_underflow_and_exact_center_controls_orientation() {
    let p = 192;
    let t = "1e-200000000";
    let r = transform_row(&["0".into(), "1".into(), "0".into()], "9", t, p);
    let derivative = float(r.derivative.as_ref().unwrap(), p);
    assert!(derivative < 0);
    let l = Float::with_val(p + 64, 9).ln();
    let expected = -(l.clone().sqrt() * &l * &l / 12u32) * float(t, p + 64);
    let relative = (Float::with_val(p + 64, &derivative) / &expected - 1u32).abs();
    assert!(relative < Float::with_val(p + 64, 1) >> (p - 16));
    let r = transform_row(
        &["1e100", "-1", "-2e100", "-1", "1e100"].map(String::from),
        "9",
        "0",
        p,
    );
    assert!(float(r.value.as_ref().unwrap(), p) < 0);
    // The large coefficient's MPFR rounding preserves the exact doubling.
    let expected = -(Float::with_val(p + 64, 9).ln() * 2u32 / 3u32).sqrt();
    close(r.value.as_ref().unwrap(), &expected.to_string(), p);
}
#[test]
fn strict_scalar_admission_and_authenticated_definitions_preserve_identity() {
    assert!(capture_reference(&reference(&["1e-400000000", "1", "0"]), &context()).is_err());
    let mut spec = dataset_at("0", 128).spec().clone();
    spec.points[0].value = Some("1e-400000000".into());
    assert!(capture_dataset(&spec, &context()).is_err());
    let input = reference(&["1", "3", "1"]);
    let reference = retained_reference(&input);
    let mut copy = reference.spec().clone();
    copy.coefficients[1] = "4".into();
    assert_eq!(reference.spec(), &input);
    let first = capture_reference(&input, &context()).unwrap().value;
    let second = capture_reference(&copy, &context()).unwrap().value;
    assert_ne!(first.request, second.request);
    assert_ne!(first.data.definition_digest, second.data.definition_digest);
    assert_eq!(first.semantics, SEMANTICS);
    let mut legacy = serde_json::to_value(first).unwrap();
    legacy["semantics"] = json!("ccm-retained-research-observations-v1");
    let (m, b) = source("ccm_reference_source", legacy.clone());
    assert!(
        RetainedReference::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).is_ok()
    );
    legacy["data"]["spec"]["coefficients"][0] = json!("1e-400000000");
    let (m, b) = source("ccm_reference_source", legacy);
    assert!(
        RetainedReference::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).is_err()
    );
    let (m, b) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":0,"precision_bits":128,"entries":["1e-400000000"]}),
    );
    assert!(RetainedMatrix::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).is_err());
}
#[test]
fn options_at_maximum_admitted_precision_remain_supported() {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":0,"precision_bits":1_000_000,"force_even":false,"eigenvalue":"1","eigenvector":["1"]}),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    assert_eq!(
        TransformOptions::for_source(&s).working_precision_bits,
        1_000_000
    );
    assert_eq!(
        TransformOptions::for_dataset(&s, &dataset_at("0", 128)).working_precision_bits,
        1_000_000
    );
}
fn shifted(s: &str, e: i32) -> String {
    let mut x = float(s, 128);
    if e >= 0 {
        x <<= e as u32;
    } else {
        x >>= e.unsigned_abs();
    }
    x.to_string()
}
fn energy_report(
    entries: &[String],
    coefficients: &[String],
    eigen: &str,
) -> anyhow::Result<EnergyData> {
    let (matrix, bytes) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"entries":entries}),
    );
    let m = RetainedMatrix::from_payload(
        &matrix,
        &bytes,
        std::slice::from_ref(&matrix.content_digest),
    )?;
    let (mut manifest, bytes) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"force_even":false,"eigenvalue":eigen,"eigenvector":coefficients}),
    );
    manifest.dependencies.push(DependencyRef {
        key: matrix.key.clone(),
        content_digest: matrix.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let state = RetainedState::from_payload(
        &manifest,
        &bytes,
        std::slice::from_ref(&manifest.content_digest),
    )?;
    Ok(capture_operator_energy(&state, &m, &context())?.value.data)
}
#[test]
fn retained_energy_matches_exact_matrix_algebra_at_extreme_scales() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/retained_energy_oracle.json")).unwrap();
    let p = 160;
    let mut checks = 0;
    for row in fixture["rows"].as_array().unwrap() {
        for (matrix_exp, vector_exp) in [
            (0i32, 0i32),
            (500_000_000, 0),
            (-500_000_000, 0),
            (0, 100_000_000),
            (0, -100_000_000),
            (-500_000_000, 100_000_000),
        ] {
            let entries = row["matrix"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| shifted(x.as_str().unwrap(), matrix_exp))
                .collect::<Vec<_>>();
            let coefficients = row["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| shifted(x.as_str().unwrap(), vector_exp))
                .collect::<Vec<_>>();
            let eigen = shifted(row["eigenvalue"].as_str().unwrap(), matrix_exp);
            let r = energy_report(&entries, &coefficients, &eigen).unwrap();
            let values = serde_json::to_value(&r).unwrap();
            for (field, power) in [
                ("coefficient_norm_squared", 2 * vector_exp),
                ("rayleigh_quotient", matrix_exp),
                ("eigenvalue_defect", matrix_exp),
                ("sum_absolute_energy_terms", matrix_exp + 2 * vector_exp),
                (
                    "relative_residual",
                    if row["eigenvalue"] == "0" {
                        matrix_exp
                    } else {
                        0
                    },
                ),
            ] {
                let mut value = float(values[field].as_str().unwrap(), p + 64);
                if power >= 0 {
                    value >>= power as u32;
                } else {
                    value <<= power.unsigned_abs();
                }
                close(
                    &value.to_string(),
                    row["values"][field].as_str().unwrap(),
                    p,
                );
                checks += 1;
            }
            match row["cancellation_digits"].as_str() {
                Some(c) => close(r.cancellation_digits.as_ref().unwrap(), c, p),
                None => assert!(r.cancellation_digits.is_none()),
            }
        }
    }
    assert_eq!(checks, 1200);
}
#[test]
fn tiny_energy_residual_is_nonzero_and_unrepresentable_raw_fields_fail() {
    let a = [
        "1e-200000000",
        "0",
        "0",
        "0",
        "2e-200000000",
        "0",
        "0",
        "0",
        "1e-200000000",
    ]
    .map(String::from);
    let r = energy_report(&a, &["1", "1", "1"].map(String::from), "1e-200000000").unwrap();
    close(
        &r.relative_residual,
        &(Float::with_val(224, 1) / Float::with_val(224, 3).sqrt()).to_string(),
        160,
    );
    assert!(energy_report(&["1".into()], &["1e-200000000".into()], "1").is_err());
    assert!(energy_report(&["1".into()], &["1e200000000".into()], "1").is_err());
}
#[test]
fn exact_energy_accumulator_preserves_cancellation_across_large_exponent_spans() {
    let huge = Float::with_val(128, 1) << 1000u32;
    let a = vec![
        huge.to_string(),
        "0".into(),
        "0".into(),
        "0".into(),
        "1".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        (-huge).to_string(),
    ];
    let r = energy_report(&a, &["1", "1", "1"].map(String::from), "0").unwrap();
    close(
        &r.rayleigh_quotient,
        &(Float::with_val(224, 1) / 3u32).to_string(),
        160,
    );
    for eigen in ["0", "1"] {
        let r = energy_report(&["0".into()], &["1".into()], eigen).unwrap();
        assert_eq!(float(&r.rayleigh_quotient, 192), 0);
        assert_eq!(float(&r.sum_absolute_energy_terms, 192), 0);
        assert_eq!(
            float(&r.relative_residual, 192),
            if eigen == "0" { 0 } else { 1 }
        );
    }
}
