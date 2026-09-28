#![cfg(feature = "hp")]
use rug::{float::Constant, Float};
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
#[allow(clippy::too_many_arguments)] // Keep source and root admission dimensions explicit.
fn roots(
    s: &RetainedState,
    m: &ArtifactManifest,
    c: &str,
    n: usize,
    source_p: u32,
    root_p: u32,
    points: &[Option<String>],
) -> RetainedRoots {
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":c,"n_modes":n,"precision_bits":source_p,"force_even":false,"eigenpair_content_digest":m.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let outcomes = points
        .iter()
        .map(|v| {
            if let Some(v) = v {
                json!({"status":"converged","details":{"value":v}})
            } else {
                json!({"status":"failed","details":{"error":"synthetic missing point"}})
            }
        })
        .collect::<Vec<_>>();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":c,"n_modes":n,"precision_bits":root_p,"force_even":false,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":outcomes}),
    );
    rm.dependencies.push(DependencyRef {
        key: sec.key.clone(),
        content_digest: sec.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &sb,
        s,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap()
}
fn run(s: &RetainedState, roots: Option<&RetainedRoots>, p: u32) -> ExtendedAnalysis {
    capture_extended(
        "complex_transform",
        s,
        None,
        roots,
        None,
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
fn hp(s: &str) -> Float {
    Float::with_val(1024, Float::parse(s).unwrap())
}
fn own(s: &str, p: u32) -> Float {
    Float::with_val(1024, Float::with_val(p, Float::parse(s).unwrap()))
}
fn within(v: &BTreeMap<String, String>, name: &str, expected: &str, p: u32) {
    let expected = hp(expected);
    let lo = hp(&v[&format!("{name}_lower")]);
    let hi = hp(&v[&format!("{name}_upper")]);
    let natural = Float::with_val(1024, 1) + expected.clone().abs();
    let slack = Float::with_val(1024, &natural) * hp("1e-170");
    assert!(
        Float::with_val(1024, &lo) - &slack <= expected
            && expected <= Float::with_val(1024, &hi) + &slack,
        "{name}: [{lo},{hi}] excludes {expected}"
    );
    assert!(
        (hp(&v[name]) - expected).abs() <= natural >> (p - 6),
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
fn scaled(v: &[String], shift: i32) -> Vec<String> {
    v.iter()
        .map(|v| {
            let mut x = Float::with_val(128, Float::parse(v).unwrap());
            if shift >= 0 {
                x <<= shift as u32;
                x = -x
            } else {
                x >>= shift.unsigned_abs()
            }
            xc_numerics::prefix::lossless_decimal(&x)
        })
        .collect()
}
fn sample<'a>(
    report: &'a ExtendedAnalysis,
    ordinal: usize,
    imag: &str,
) -> &'a BTreeMap<String, String> {
    &report
        .rows
        .iter()
        .find(|r| {
            r.values
                .get("input_ordinal")
                .is_some_and(|v| hp(v) == ordinal)
                && r.values.get("z_im").is_some_and(|v| hp(v) == hp(imag))
        })
        .unwrap()
        .values
}
#[test]
fn complex_samples_match_independent_defining_fourier_integrals() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/complex_transform_minus_oracle.json")).unwrap();
    let mut checks = 0;
    for profile in 0..6 {
        let rows = oracle["rows"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v["profile"] == profile)
            .collect::<Vec<_>>();
        let first = rows[0];
        let c = first["cutoff"].as_str().unwrap();
        let co = strings(&first["coefficients"]);
        let n = co.len() / 2;
        let points = (1..=3)
            .map(|ordinal| {
                Some(
                    rows.iter().find(|v| v["ordinal"] == ordinal).unwrap()["t"]
                        .as_str()
                        .unwrap()
                        .into(),
                )
            })
            .collect::<Vec<_>>();
        for p in [128, 192, 256] {
            for shift in [-600, 0, 600] {
                let (s, sm) = state(c, &scaled(&co, shift), 128);
                let roots = roots(&s, &sm, c, n, 128, 128, &points);
                let report = run(&s, Some(&roots), p);
                assert_eq!(report.rows.len(), 85);
                assert!(report.assurance.contains("samples do not certify"));
                for row in &rows {
                    let values = sample(
                        &report,
                        row["ordinal"].as_u64().unwrap() as usize,
                        row["imaginary"].as_str().unwrap(),
                    );
                    for (name, value) in row["expected"].as_object().unwrap() {
                        within(
                            if name == "normalization_anchor" {
                                &report.values
                            } else {
                                values
                            },
                            name,
                            value.as_str().unwrap(),
                            p,
                        );
                        checks += 1;
                    }
                }
                assert_eq!(
                    report.rows[20].values, report.rows[84].values,
                    "contour must close exactly"
                );
            }
        }
    }
    assert_eq!(checks, 7290);
}
#[test]
fn tiny_real_derivative_and_log_derivative_remain_nonzero_under_source_scaling() {
    let t = Float::with_val(128, Float::parse("1e-200000000").unwrap());
    let high_t = Float::with_val(1024, &t);
    let l = Float::with_val(1024, 9).ln();
    let expected = -l.clone().square() * l.clone().sqrt() / 12u32;
    let expected_log = -l.square() / 12u32;
    for c0 in ["1", "-1", "1e-200000000", "1e200000000"] {
        let (s, sm) = state("9", &["0".into(), c0.into(), "0".into()], 128);
        let roots = roots(
            &s,
            &sm,
            "9",
            1,
            128,
            128,
            &[Some(xc_numerics::prefix::lossless_decimal(&t))],
        );
        for p in [128, 192, 256] {
            let r = run(&s, Some(&roots), p);
            let v = sample(&r, 1, "0");
            assert!(hp(&v["derivative_re_upper"]) < 0);
            for (name, expected) in [
                ("derivative_re", &expected),
                ("log_derivative_re", &expected_log),
            ] {
                let ratio = own(&v[name], p) / &high_t;
                assert!(
                    (ratio - expected).abs() < Float::with_val(1024, expected).abs() >> (p - 6),
                    "{name} lost the representable leading term"
                );
            }
            assert_eq!(own(&v["z_re"], p), high_t);
        }
    }
}
#[test]
fn stored_ordinate_aliases_and_exact_cutoff_are_preserved() {
    let (s, sm) = state("9", &["0".into(), "1".into(), "0".into()], 128);
    let a = roots(&s, &sm, "9", 1, 128, 128, &[Some("1".into())]);
    let b = roots(
        &s,
        &sm,
        "9",
        1,
        128,
        128,
        &[Some("1.0000000000000000000000000000000000000001".into())],
    );
    let a = run(&s, Some(&a), 192);
    let b = run(&s, Some(&b), 192);
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
    let cutoff = include_str!("fixtures/resolution_near_one_cutoff.txt").trim();
    let p = 1536;
    let (s, sm) = state(cutoff, &["0".into(), "1".into(), "0".into()], 512);
    let c0 = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 250u32);
    let t = Float::with_val(512, Float::with_val(p, Constant::Pi) * 2u32 / c0.ln());
    let roots = roots(
        &s,
        &sm,
        cutoff,
        1,
        512,
        512,
        &[Some(xc_numerics::prefix::lossless_decimal(&t))],
    );
    let r = run(&s, Some(&roots), 512);
    let v = sample(&r, 1, "0");
    let l = Float::with_val(p, Float::parse(cutoff).unwrap()).ln();
    let q = Float::with_val(p, &t) * &l / 2u32;
    let expected = l.clone().sqrt() * q.clone().sin() / &q;
    let derivative = l.clone().sqrt() * l / 2u32 * (q.clone().cos() - q.clone().sin() / &q) / q;
    for (name, value) in [("value_re", expected), ("derivative_re", derivative)] {
        let low = Float::with_val(p, Float::parse(&v[&format!("{name}_lower")]).unwrap());
        let high = Float::with_val(p, Float::parse(&v[&format!("{name}_upper")]).unwrap());
        assert!(low <= value && value <= high, "cutoff changed {name}");
    }
    assert!(hp(&v["value_re"]) < hp("-1e-137"));
    assert!(hp(&v["arithmetic_precision_bits"]) > 576);
}
#[test]
fn missing_points_zero_anchor_and_resource_limits_are_explicit() {
    let (s, sm) = state("9", &["1".into(), "0".into(), "1".into()], 128);
    let roots = roots(
        &s,
        &sm,
        "9",
        1,
        128,
        128,
        &[Some("1".into()), None, Some("2".into())],
    );
    let r = run(&s, Some(&roots), 192);
    assert_eq!(r.outcome, "partial_unresolved");
    assert_eq!(r.rows.len(), 85);
    let missing = r
        .rows
        .iter()
        .filter(|r| r.outcome == "missing_input")
        .collect::<Vec<_>>();
    assert_eq!(missing.len(), 5);
    for row in missing {
        assert_eq!(hp(&row.values["input_ordinal"]), 2);
        assert!(!row.values.contains_key("z_re") && !row.values.contains_key("value_re"));
    }
    for row in r.rows.iter().filter(|r| r.values.contains_key("value_re")) {
        assert!(!row.values.contains_key("normalized_re"));
        assert_eq!(hp(&row.values["normalization_denominator_resolved"]), 0);
    }
    let o = ExtensionOptions {
        maximum_working_bytes: Some(1),
        ..ExtensionOptions::for_source(&s)
    };
    let r = capture_extended(
        "complex_transform",
        &s,
        None,
        Some(&roots),
        None,
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.values.is_empty() && r.rows.is_empty());
    let (s, _) = state("9", &["1".into()], 128);
    let r = run(&s, None, 128);
    assert_eq!(r.rows.len(), 70);
    assert_eq!(r.outcome, "point_measurement");
}
#[test]
fn ratio_range_failure_keeps_defined_finite_transform_and_derivative() {
    let (s, _) = state(
        "1e200000000",
        &["1".into(), "1e-323200000".into(), "1".into()],
        128,
    );
    let r = run(&s, None, 192);
    let v = sample(&r, 0, "1");
    assert!(v.contains_key("value_re_lower") && v.contains_key("derivative_im_upper"));
    assert_eq!(hp(&v["normalization_denominator_resolved"]), 0);
    assert!(!v.contains_key("normalized_re"));
}
#[test]
fn complex_checkpoint_cold_warm_and_worker_counts_agree() {
    if std::env::var_os("XC_TEST_COMPLEX_CHILD").is_some() {
        let (s, sm) = state("9", &["1".into(), "3".into(), "2".into()], 128);
        let roots = roots(
            &s,
            &sm,
            "9",
            1,
            128,
            128,
            &[Some("1".into()), None, Some("1e-200000000".into())],
        );
        let r = run(&s, Some(&roots), 192);
        std::fs::write(
            std::env::var_os("XC_TEST_COMPLEX_OUTPUT").unwrap(),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        return;
    }
    let root = std::env::temp_dir().join(format!(
        "xc-complex-checkpoint-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let execute = |dir: &str, workers: &str, name: &str| {
        let path = root.join(name);
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "complex_checkpoint_cold_warm_and_worker_counts_agree",
                "--nocapture",
            ])
            .env("XC_TEST_COMPLEX_CHILD", "1")
            .env("XC_TEST_COMPLEX_OUTPUT", &path)
            .env("XC_RESEARCH_CHECKPOINT_DIR", root.join(dir))
            .env("RAYON_NUM_THREADS", workers)
            .env("XC_CACHE_REMOTE", "none")
            .env("XC_PUBLISH_EXECUTE", "false")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read(path).unwrap()
    };
    let cold = execute("warm", "1", "cold.json");
    assert!(std::fs::read_dir(root.join("warm")).unwrap().count() > 0);
    let warm = execute("warm", "4", "warm.json");
    let fresh = execute("fresh", "4", "fresh.json");
    assert_eq!(cold, warm);
    assert_eq!(warm, fresh);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn retained_root_complex_channels_and_contour_follow_independent_minus_integrals() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/minus_fourier_oracle.json",)).unwrap();
    for tag in [
        "original_five_component",
        "unrelated_nonparity",
        "unrelated_even",
    ] {
        let rows = oracle["rows"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["case"] == tag)
            .collect::<Vec<_>>();
        let first = rows[0];
        let cutoff = first["cutoff"].as_str().unwrap();
        let coefficients = strings(&first["coefficients"]);
        let (state, manifest) = state(cutoff, &coefficients, 128);
        let roots = roots(
            &state,
            &manifest,
            cutoff,
            coefficients.len() / 2,
            128,
            128,
            &[Some(first["t"].as_str().unwrap().into())],
        );
        for p in [128, 192, 256] {
            let report = run(&state, Some(&roots), p);
            assert!(report.convention.contains("exp(-i*z*x)"));
            within(
                &report.values,
                "normalization_anchor",
                first["normalization_anchor"].as_str().unwrap(),
                p,
            );
            for reference in &rows {
                let values = if reference["kind"] == "contour" {
                    &report.rows[10
                        + 16 * reference["side"].as_u64().unwrap() as usize
                        + reference["step"].as_u64().unwrap() as usize]
                        .values
                } else {
                    sample(
                        &report,
                        reference["ordinal"].as_u64().unwrap() as usize,
                        reference["imaginary"].as_str().unwrap(),
                    )
                };
                for (name, expected) in reference["expected"].as_object().unwrap() {
                    within(values, name, expected.as_str().unwrap(), p);
                }
            }
        }
    }
}

#[cfg(feature = "arb")]
#[test]
fn retained_root_arb_values_derivatives_and_contour_match_minus_integrals() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/minus_fourier_oracle.json",)).unwrap();
    let contains = |values: &BTreeMap<String, String>, name: &str, expected: &str| {
        let exact = hp(expected);
        let lo = hp(&values[&format!("{name}_lower")]);
        let hi = hp(&values[&format!("{name}_upper")]);
        let slack = (Float::with_val(1024, 1) + exact.clone().abs()) * hp("1e-140");
        assert!(
            Float::with_val(1024, &lo) - &slack <= exact
                && exact <= Float::with_val(1024, &hi) + &slack,
            "{name}: [{lo},{hi}] excludes {exact}"
        );
    };
    for tag in [
        "original_five_component",
        "unrelated_nonparity",
        "unrelated_even",
    ] {
        let rows = oracle["rows"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["case"] == tag)
            .collect::<Vec<_>>();
        let first = rows[0];
        let cutoff = first["cutoff"].as_str().unwrap();
        let coefficients = strings(&first["coefficients"]);
        let (state, manifest) = state(cutoff, &coefficients, 128);
        let roots = roots(
            &state,
            &manifest,
            cutoff,
            coefficients.len() / 2,
            128,
            128,
            &[Some(first["t"].as_str().unwrap().into())],
        );
        let record = capture_extended(
            "transform_enclosure",
            &state,
            None,
            Some(&roots),
            None,
            &ExtensionOptions {
                working_precision_bits: 192,
                ..ExtensionOptions::for_source(&state)
            },
            &[],
            &context(),
        )
        .unwrap()
        .value;
        assert!(record.data.convention.contains("exp(-i*z*x)"));
        assert_eq!(
            record.request["enclosure_fourier_semantics"],
            "retained_fourier_minus_sign_at_original_coordinates_v1"
        );
        let root = record
            .data
            .rows
            .iter()
            .find(|r| r.label == "retained_root_enclosure")
            .unwrap();
        let reference = rows
            .iter()
            .find(|r| r["kind"] == "root" && r["imaginary"] == "0")
            .unwrap();
        for (actual, expected) in [
            ("value_real", "value_re"),
            ("value_imaginary", "value_im"),
            ("derivative_real", "derivative_re"),
            ("derivative_imaginary", "derivative_im"),
            ("normalized_real", "normalized_re"),
            ("normalized_imaginary", "normalized_im"),
        ] {
            contains(
                &root.values,
                actual,
                reference["expected"][expected].as_str().unwrap(),
            );
        }
        for reference in rows.iter().filter(|r| r["kind"] == "contour") {
            let side = reference["side"].as_u64().unwrap();
            let a = Float::with_val(1024, reference["step"].as_u64().unwrap()) / 16;
            let right: Float = own(first["t"].as_str().unwrap(), 128) + 1;
            let width = right.clone() + 1;
            let (re, im) = match side {
                0 => (
                    Float::with_val(1024, -1) + width * a,
                    Float::with_val(1024, -1),
                ),
                1 => (right, Float::with_val(1024, -1) + a * 2),
                2 => (right - width * a, Float::with_val(1024, 1)),
                _ => (Float::with_val(1024, -1), Float::with_val(1024, 1) - a * 2),
            };
            let segment = record
                .data
                .rows
                .iter()
                .find(|r| {
                    r.label == "contour_segment"
                        && hp(&r.values["start_re"]).min(&hp(&r.values["end_re"])) <= re
                        && re <= hp(&r.values["start_re"]).max(&hp(&r.values["end_re"]))
                        && hp(&r.values["start_im"]).min(&hp(&r.values["end_im"])) <= im
                        && im <= hp(&r.values["start_im"]).max(&hp(&r.values["end_im"]))
                        && r.values.contains_key("value_real_lower")
                })
                .unwrap();
            for (actual, expected) in [
                ("value_real", "value_re"),
                ("value_imaginary", "value_im"),
                ("derivative_real", "derivative_re"),
                ("derivative_imaginary", "derivative_im"),
            ] {
                contains(
                    &segment.values,
                    actual,
                    reference["expected"][expected].as_str().unwrap(),
                );
            }
        }
    }
}
