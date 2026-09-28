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

use rug::{Integer, Rational};
use xc_numerics::prefix::lossless_decimal as dec;
use xc_spectral::ccm::research_completion::prepare_tail_form;

fn point(text: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(text).unwrap())
}
fn rational(v: &serde_json::Value, p: u32) -> Float {
    Float::with_val(
        p,
        Rational::from((
            v["numerator"].as_str().unwrap().parse::<Integer>().unwrap(),
            v["denominator"]
                .as_str()
                .unwrap()
                .parse::<Integer>()
                .unwrap(),
        )),
    )
}
fn atom(x: String, w: String, family: &str, ordinal: usize) -> WeightedAtom {
    WeightedAtom {
        ordinal,
        coordinate: x,
        weight: w,
        family: family.into(),
        partition: "finite fixture".into(),
    }
}
fn input(m: &ArtifactManifest, p: u32, form: serde_json::Value) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":0,"precision_bits":p,"convention_id":"finite pencil oracle","definition_digest":ContentDigest::sha256(b"finite pencil oracle"),"approximation_scope":"finite fixture","run_once":{"tail_form":form}})).unwrap()
}
fn form(
    finite: serde_json::Value,
    tail: serde_json::Value,
    gram: serde_json::Value,
    n: usize,
) -> serde_json::Value {
    json!({"definition_digest":ContentDigest::sha256(b"finite pencil"),"dimension":n,"finite_zero_form":finite,"tail_correction":tail,"lattice_gram":gram,"coverage":"finite","hypotheses":[]})
}
fn capture(s: &RetainedState, i: &ExternalResearchInputs, p: u32) -> ExtendedAnalysis {
    let mut o = ExtensionOptions::for_source(s);
    o.working_precision_bits = p;
    let record = capture_extended("tail_operator", s, None, None, Some(i), &o, &[], &context())
        .unwrap()
        .value;
    assert_eq!(
        record.semantics,
        xc_spectral::ccm::retained_evidence::SEMANTICS
    );
    assert_eq!(
        record.request["tail_model_arithmetic"],
        "declared_points_exact_dyadic_recipe_forms_v2"
    );
    record.data
}

#[test]
fn exact_weighted_polynomial_forms_match_fraction_oracle_at_extreme_scales() {
    let all: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/tail_model_oracle.json")).unwrap();
    let mut comparisons = 0;
    for case in all["forms"].as_array().unwrap() {
        for p in [64, 128, 256] {
            for shift in [-500_000_000i32, 0, 500_000_000] {
                let basis = case["basis"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| dec(&(point(v.as_str().unwrap(), p) << shift)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let atoms = case["atoms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(j, a)| {
                        atom(
                            a["coordinate"].as_str().unwrap().into(),
                            dec(&(point(a["weight"].as_str().unwrap(), p) >> (2 * shift))),
                            if a["zero_family"].as_bool().unwrap() {
                                "zero"
                            } else {
                                "lattice"
                            },
                            j + 1,
                        )
                    })
                    .collect::<Vec<_>>();
                let f = prepare_tail_form(
                    ContentDigest::sha256(b"fraction oracle"),
                    &basis,
                    &atoms,
                    None,
                    "finite",
                    &[],
                    p,
                )
                .unwrap();
                for (actual, key) in [(&f.finite_zero_form, "zero"), (&f.lattice_gram, "lattice")] {
                    for (a, e) in actual.iter().zip(case[key].as_array().unwrap()) {
                        assert_eq!(point(a, p), rational(e, p), "p={p}, shift={shift}");
                        comparisons += 1;
                    }
                }
            }
        }
    }
    assert_eq!(comparisons, 8640);
}

#[test]
fn direct_tail_model_decodes_form_error_and_energy_at_own_precision() {
    let (s, m) = state("9", &["1".into()], 128);
    let mut reports = Vec::new();
    for value in ["1", "1.0000000000000000000000000000000000000001"] {
        let mut f = form(json!([value]), json!(["0"]), json!(["1"]), 1);
        f["tail_operator_error"] = json!(value);
        let i = input(&m, 128, f);
        reports.push(capture(&s, &i, 192));
    }
    // Matrix/eigenvalue strings are stored p-bit points. The declared error is
    // instead an exact decimal upper bound and may not be rounded down to one.
    let error = xc_core::DecimalLiteral::new("1.0000000000000000000000000000000000000001").unwrap();
    let reported =
        xc_core::DecimalLiteral::new(&reports[1].values["declared_tail_form_error"]).unwrap();
    assert!(!reported.cmp_numeric(&error).unwrap().is_lt());
    assert!(point(&reports[1].values["conditional_energy_error_budget"], 192) > 2);
    let mut first = reports[0].values.clone();
    let mut second = reports[1].values.clone();
    for name in [
        "declared_tail_form_error",
        "conditional_energy_error_budget",
    ] {
        first.remove(name);
        second.remove(name);
    }
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_value(&reports[0].rows).unwrap(),
        serde_json::to_value(&reports[1].rows).unwrap()
    );
    assert_eq!(point(&reports[0].values["model_energy"], 192), 2);
    assert_eq!(
        point(&reports[0].values["conditional_energy_error_budget"], 192),
        2
    );
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":0,"precision_bits":128,"force_even":false,"eigenvalue":"3.0000000000000000000000000000000000000001","eigenvector":["1"]}),
    );
    let alias =
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let i = input(&m, 128, form(json!(["1"]), json!(["0"]), json!(["1"]), 1));
    assert_eq!(
        point(
            &capture(&alias, &i, 192).values["retained_energy_for_scoring_only"],
            192
        ),
        3
    );
}

#[test]
fn tail_recipe_cutoffs_coefficients_and_weights_obey_source_precision() {
    let (s, m) = state("9", &["1".into()], 128);
    let mut reports = Vec::new();
    for value in ["1", "1.0000000000000000000000000000000000000001"] {
        let mut i = input(&m, 128, serde_json::Value::Null);
        i.atoms = vec![
            atom(value.into(), value.into(), "zero", 1),
            atom("1".into(), "1".into(), "lattice", 1),
        ];
        i.atom_coordinate = Some("declared coordinate".into());
        i.atom_coverage = Some("finite".into());
        i.run_once.as_mut().unwrap().completion=Some(serde_json::from_value(json!({"atom_analysis":{"maximum_atoms":2,"maximum_input_bytes":4096,"cutoffs":["1"],"tail_recipe":{"basis_polynomials":[[value]],"tail_correction":["0"],"hypotheses":[]}}})).unwrap());
        reports.push(capture(&s, &i, 192));
    }
    assert_eq!(reports[0].values, reports[1].values);
    assert_eq!(
        serde_json::to_value(&reports[0].rows).unwrap(),
        serde_json::to_value(&reports[1].rows).unwrap()
    );
    for row in &reports[0].rows {
        if row.label == "tail_model_cutoff" {
            assert_eq!(point(&row.values["model_energy"], 192), 2);
        }
    }
}

#[test]
fn tiny_weighted_product_is_nonzero_after_compensation() {
    let p = 128;
    let c = point("1e-200000000", p);
    let w = point("1e200000000", p);
    let f = prepare_tail_form(
        ContentDigest::sha256(b"tiny product"),
        &[vec![dec(&c)]],
        &[atom("1".into(), dec(&w), "zero", 1)],
        None,
        "finite",
        &[],
        p,
    )
    .unwrap();
    let exact = Float::with_val(1024, &c) * Float::with_val(1024, &w) * Float::with_val(1024, &c);
    assert_ne!(exact, 0);
    assert_eq!(point(&f.finite_zero_form[0], p), Float::with_val(p, exact));
}

#[test]
fn exact_form_preserves_cancellation_and_reports_resource_and_range_failures() {
    let p = 128;
    let digest = ContentDigest::sha256(b"exact cancellation");
    let atoms = vec![
        atom("1".into(), "1e200000000".into(), "zero", 1),
        atom("1".into(), "-1e200000000".into(), "zero", 2),
        atom("1".into(), "1".into(), "zero", 3),
    ];
    let f = prepare_tail_form(
        digest.clone(),
        &[vec!["1".into()]],
        &atoms,
        None,
        "finite",
        &[],
        p,
    )
    .unwrap();
    assert_eq!(point(&f.finite_zero_form[0], p), 1);
    let huge = dec(&(Float::with_val(p, 1) << 10_000_000u32));
    let err = prepare_tail_form(
        digest.clone(),
        &[vec!["1".into(), "1".into()]],
        &[atom(huge, "1".into(), "zero", 1)],
        None,
        "finite",
        &[],
        p,
    )
    .unwrap_err();
    assert!(err.to_string().contains("span budget"));
    let huge = dec(&(Float::with_val(p, 1) << 1_000_000_000u32));
    let err = prepare_tail_form(
        digest,
        &[vec![huge]],
        &[atom("1".into(), "1".into(), "zero", 1)],
        None,
        "finite",
        &[],
        p,
    )
    .unwrap_err();
    assert!(err.to_string().contains("exponent range"));
}

fn close(actual: &Float, expected: &Float, p: u32, message: &str) {
    let error = (Float::with_val(p, actual) - expected).abs();
    let scale = expected.clone().abs().max(&Float::with_val(p, 1));
    assert!(
        error <= scale * (Float::with_val(p, 1) >> (p / 3)),
        "{message}: actual {actual}, expected {expected}, error {error}"
    );
}

#[test]
fn generalized_pencils_match_exact_congruence_oracle() {
    let all: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/tail_model_oracle.json")).unwrap();
    let mut reports = 0;
    for (source_p, p) in [(64, 128), (128, 192), (256, 320)] {
        let (s, m) = state("9", &["1".into()], source_p);
        for (index, c) in all["pencils"].as_array().unwrap().iter().enumerate() {
            for shift in [-500_000_000i32, 0, 500_000_000] {
                let n = c["dimension"].as_u64().unwrap() as usize;
                let values = |key: &str| {
                    json!(c[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| dec(&(point(x.as_str().unwrap(), source_p) << shift)))
                        .collect::<Vec<_>>())
                };
                let i = input(
                    &m,
                    source_p,
                    form(values("finite"), values("tail"), values("gram"), n),
                );
                let out = capture(&s, &i, p);
                assert_eq!(
                    out.outcome, "point_measurement",
                    "case {index}, shift {shift}: {:?}",
                    out.reason
                );
                for (field, key) in [
                    ("model_energy", "model_energy"),
                    ("model_energy_without_tail", "without_tail"),
                    ("tail_energy_lift", "tail_lift"),
                ] {
                    close(
                        &point(&out.values[field], p),
                        &rational(&c[key], p),
                        p,
                        field,
                    );
                }
                for (row, e) in out.rows.iter().zip(c["energies"].as_array().unwrap()) {
                    close(
                        &point(&row.values["energy"], p),
                        &rational(e, p),
                        p,
                        "spectrum",
                    );
                }
                let actual_norm =
                    point(&out.values["inverse_cholesky_frobenius_norm_squared"], p) << shift;
                close(
                    &actual_norm,
                    &rational(&c["inverse_cholesky_frobenius_squared"], p),
                    p,
                    "inverse Cholesky",
                );
                let actual = (0..n)
                    .map(|j| {
                        point(&out.values[&format!("model_vector_coefficient_{j}")], p)
                            << (shift / 2)
                    })
                    .collect::<Vec<_>>();
                let expected = c["vector"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| rational(v, p))
                    .collect::<Vec<_>>();
                let first = expected.iter().position(|v| v != &0).unwrap();
                let sign = if actual[first].is_sign_negative() == expected[first].is_sign_negative()
                {
                    1
                } else {
                    -1
                };
                for (a, e) in actual.iter().zip(&expected) {
                    close(
                        &Float::with_val(p, a),
                        &(e.clone() * sign),
                        p,
                        "G-unit vector",
                    );
                }
                close(
                    &point(&out.values["model_vector_lattice_norm_squared"], p),
                    &Float::with_val(p, 1),
                    p,
                    "lattice norm",
                );
                reports += 1;
            }
        }
    }
    assert_eq!(reports, 432);
}

#[test]
fn singular_gram_and_non_symmetric_tail_are_not_accepted_as_solved_models() {
    let (s, m) = state("9", &["1".into()], 128);
    let i = input(
        &m,
        128,
        form(
            json!(["1", "0", "0", "1"]),
            json!(["0", "0", "0", "0"]),
            json!(["1", "1", "1", "1"]),
            2,
        ),
    );
    assert_ne!(capture(&s, &i, 192).outcome, "point_measurement");
    let err = prepare_tail_form(
        ContentDigest::sha256(b"asymmetric"),
        &[vec!["1".into()], vec!["0".into(), "1".into()]],
        &[],
        Some(&["0".into(), "1".into(), "0".into(), "0".into()]),
        "finite",
        &[],
        128,
    )
    .unwrap_err();
    assert!(err.to_string().contains("not symmetric"));
}
#[test]
fn conditional_tail_budget_is_an_upper_bound() {
    let (s, m) = state("9", &["1".into()], 128);
    for g in 2..=32 {
        let mut f = form(json!(["1"]), json!(["0"]), json!([g.to_string()]), 1);
        f["tail_operator_error"] = json!("1");
        let out = capture(&s, &input(&m, 128, f), 192);
        let bound = point(&out.values["conditional_energy_error_budget"], 192);
        let exact = Rational::from((2, g));
        assert!(
            bound.to_rational().unwrap() >= exact,
            "underestimated error budget for Gram=[{g}]: {bound}"
        );
    }
}
