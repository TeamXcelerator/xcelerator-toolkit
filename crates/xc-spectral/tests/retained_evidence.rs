#![cfg(feature = "hp")]
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{retained_evidence::*, state_geometry::RetainedState};
#[path = "common/published_sources.rs"]
mod published_sources;
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
        producer_toolkit_version: ToolkitVersion::parse("0.14.3").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
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
fn state(coefficients: &[&str], eigenvalue: &str) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"force_even":true,"eigenvalue":eigenvalue,"eigenvector":coefficients}),
    );
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
    )
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
fn number(s: &str) -> f64 {
    rug::Float::with_val(256, rug::Float::parse(s).unwrap()).to_f64()
}
#[test]
fn finite_fourier_projection_preserves_nonorthogonal_coefficients() {
    let (s, _) = state(&["1", "3", "1"], "3");
    let reference = retained_reference(&reference(&["0", "1", "0"]));
    let b0 = retained_reference(&self::reference(&["0", "1", "0"]));
    let b1 = retained_reference(&self::reference(&["1", "1", "1"]));
    let options = ProjectionOptions {
        working_precision_bits: 192,
        normalization: "center_one".into(),
        fixed_second_component: Some("-0.75".into()),
    };
    let r = capture_projection(&s, &reference, &[b0.clone(), b1], &options, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.outcome, "point_measurement");
    let a = r.coefficients.unwrap();
    assert!((number(&a[0]) - 1.0).abs() < 1e-14);
    assert!((number(&a[1]) - 1.0).abs() < 1e-14);
    assert!((number(r.b2.as_ref().unwrap()) - 1.75).abs() < 1e-14);
    assert!(number(r.fit_residual_norm_squared.as_ref().unwrap()) < 1e-50);
    let r = capture_projection(&s, &reference, &[b0.clone(), b0], &options, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.outcome, "rank_or_precision_unresolved");
    assert!(r.coefficients.is_none());
    let (odd, _) = state(&["-1", "0", "1"], "3");
    let r = capture_projection(
        &odd,
        &reference,
        &[],
        &ProjectionOptions {
            fixed_second_component: None,
            ..options
        },
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "normalization_unresolved");
    assert!(r.difference_norm_squared.is_none());
}
#[test]
fn indexed_transform_matches_closed_form_and_preserves_missing_rows() {
    use rug::{float::Constant, Float};
    let (s, _) = state(&["0", "-5", "0"], "3");
    let p = 256;
    let l = Float::with_val(p, 9).ln();
    let pi = Float::with_val(p, Constant::Pi);
    let t = Float::with_val(p, &pi) / &l;
    let spec = DatasetSpec {
        schema_version: 1,
        role: "evaluation_points".into(),
        attribution: "analytic constant-function test".into(),
        coordinate: "mellin_t".into(),
        precision_bits: p,
        points: vec![
            EvaluationPoint {
                ordinal: 1,
                value: Some("0".into()),
                source_status: "supplied".into(),
            },
            EvaluationPoint {
                ordinal: 2,
                value: Some(t.to_string()),
                source_status: "supplied".into(),
            },
            EvaluationPoint {
                ordinal: 3,
                value: None,
                source_status: "missing".into(),
            },
            EvaluationPoint {
                ordinal: 4,
                value: Some((t * 2u32).to_string()),
                source_status: "carrier".into(),
            },
        ],
    };
    let value = capture_dataset(&spec, &context()).unwrap().value;
    let (m, b) = source(
        "research_reference_dataset",
        serde_json::to_value(value).unwrap(),
    );
    let d = RetainedDataset::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let o = TransformOptions {
        working_precision_bits: p,
        maximum_rows: 4,
        maximum_estimated_output_bytes: 1024 * 1024,
    };
    let result = capture_transforms_at_dataset(&s, &d, &o, &context()).unwrap();
    let r = &result.value.data;
    let lf = 9f64.ln();
    assert!((number(r.rows[0].value.as_ref().unwrap()) - lf.sqrt()).abs() < 1e-14);
    assert_eq!(r.rows[0].outcome, "unresolved_derivative");
    assert!(r.rows[0].newton_correction.is_none());
    assert!(
        (number(r.rows[1].value.as_ref().unwrap()) - 2.0 * lf.sqrt() / std::f64::consts::PI).abs()
            < 1e-14
    );
    assert!(
        (number(r.rows[1].derivative.as_ref().unwrap())
            + 2.0 * lf.powf(1.5) / std::f64::consts::PI.powi(2))
        .abs()
            < 1e-14
    );
    assert_eq!(r.rows[2].outcome, "missing_input");
    assert!(r.rows[2].value.is_none());
    assert!(number(r.rows[3].value.as_ref().unwrap()).abs() < 1e-60);
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| capture_transforms_at_dataset(&s, &d, &o, &context()).unwrap());
    assert_eq!(
        serde_json::to_value(result.value).unwrap(),
        serde_json::to_value(four.value).unwrap()
    );
    assert!(capture_transforms_at_dataset(
        &s,
        &d,
        &TransformOptions {
            maximum_rows: 3,
            ..o
        },
        &context()
    )
    .is_err());
}
#[test]
fn energy_requires_exact_ancestry_and_matches_diagonal_operator() {
    let (matrix, bytes) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["7","0","0","0","3","0","0","0","5"]}),
    );
    let m = RetainedMatrix::from_payload(
        &matrix,
        &bytes,
        std::slice::from_ref(&matrix.content_digest),
    )
    .unwrap();
    let (mut em, eb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","-5","0"]}),
    );
    let s =
        RetainedState::from_payload(&em, &eb, std::slice::from_ref(&em.content_digest)).unwrap();
    assert!(capture_operator_energy(&s, &m, &context()).is_err());
    em.dependencies.push(DependencyRef {
        key: matrix.key.clone(),
        content_digest: matrix.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s =
        RetainedState::from_payload(&em, &eb, std::slice::from_ref(&em.content_digest)).unwrap();
    let r = capture_operator_energy(&s, &m, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(number(&r.rayleigh_quotient), 3.0);
    assert_eq!(number(&r.eigenvalue_defect), 0.0);
    assert_eq!(number(&r.relative_residual), 0.0);
}

#[test]
fn published_energy_traces_canonical_factor_ancestry_and_rejects_tampering() {
    use published_sources::published;
    let (matrix, bytes) = source(
        "ccm_tau_matrix",
        json!({
            "schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "entries":["7","0","0","0","3","0","0","0","5"]
        }),
    );
    let matrix = published(matrix, &[]);
    let (factor, _) = source("ccm_factorization", json!({"fixture":"factor"}));
    let mut factor = published(factor, &[&matrix]);
    let (em, eb) = source(
        "ccm_weil_eigenpair",
        json!({
            "schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "force_even":true,"eigenvalue":"3","eigenvector":["0","-5","0"]
        }),
    );
    let em = published(em, &[&factor]);
    // Identity-based lookup gives this dependency a different logical address.
    factor.key.logical_key = "published/identity/factor".into();
    let s =
        RetainedState::from_payload(&em, &eb, std::slice::from_ref(&em.content_digest)).unwrap();
    let m = RetainedMatrix::from_payload(
        &matrix,
        &bytes,
        std::slice::from_ref(&matrix.content_digest),
    )
    .unwrap();
    assert!(capture_operator_energy(&s, &m, &context()).is_err());
    let result =
        capture_operator_energy_with_ancestry(&s, &m, &[factor.clone()], &context()).unwrap();
    assert_eq!(number(&result.value.data.rayleigh_quotient), 3.0);
    assert_eq!(number(&result.value.data.relative_residual), 0.0);
    let mut canonical: CanonicalArtifactManifest =
        serde_json::from_str(&factor.tags[REMOTE_CANONICAL_MANIFEST_TAG]).unwrap();
    canonical.canonical_payload.dependencies.clear();
    canonical.payload_digest = canonical.canonical_payload.digest().unwrap();
    factor.tags.insert(
        REMOTE_CANONICAL_MANIFEST_TAG.into(),
        serde_json::to_string(&canonical).unwrap(),
    );
    assert!(capture_operator_energy_with_ancestry(&s, &m, &[factor], &context()).is_err());
}

#[test]
fn published_root_sources_validate_exact_canonical_edges_with_empty_local_lists() {
    use published_sources::published;
    let (sm, eb) = source(
        "ccm_weil_eigenpair",
        json!({
            "schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]
        }),
    );
    let sm = published(sm, &[]);
    let s =
        RetainedState::from_payload(&sm, &eb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let (sec, sb) = source(
        "ccm_secular_source",
        json!({
            "schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,
            "normalization":"sum_xi_equals_sqrt_log_lambda_squared","eigenpair_content_digest":sm.content_digest.0
        }),
    );
    let sec = published(sec, &[&sm]);
    let (rm, rb) = source(
        "ccm_root_refinement",
        json!({
            "schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,
            "force_even":true,"first_root_index":1,"discovery_mode":"reference_seeded_audit",
            "reference_seeds_used":true,"completeness":"complete",
            "outcomes":[{"status":"converged","details":{"value":"14"}}]
        }),
    );
    let rm = published(rm, &[&sec]);
    let allowed = [rm.content_digest.clone(), sec.content_digest.clone()];
    let roots = RetainedRoots::from_payload(&rm, &rb, &sec, &sb, &s, &allowed).unwrap();
    assert_eq!(
        capture_root_window(&roots, &context())
            .unwrap()
            .value
            .data
            .points
            .len(),
        1
    );
    assert!(capture_transforms_at_roots(
        &s,
        &roots,
        &TransformOptions::for_roots(&s, &roots),
        &context()
    )
    .is_ok());
    let wrong = published(sec, &[]);
    assert!(RetainedRoots::from_payload(&rm, &rb, &wrong, &sb, &s, &allowed).is_err());
}
#[test]
fn stabilization_is_a_frozen_finite_rule_and_imports_are_not_replayed() {
    let (a, _) = state(&["1"], "1");
    let (b, _) = state(&["0", "1", "0"], "1.005");
    let (c, _) = state(&["0", "0", "1", "0", "0"], "1.006");
    let o = StabilizationOptions {
        working_precision_bits: 160,
        relative_tolerance: "0.01".into(),
        consecutive_steps: 2,
    };
    let r = capture_stabilization(&[a, b, c], &o, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(r.outcome, "finite_rule_met");
    assert!(r.qualification.contains("not_N_infinity"));
    let observation = ExternalObservation {
        schema_version: 1,
        original_utf8: "Unverified external point result: 1e-900".into(),
        attribution: "synthetic test".into(),
        definition: "external observation".into(),
        hypotheses: vec!["not checked".into()],
        borrowed_inputs: vec![],
        limitations: vec!["not replayed".into()],
    };
    let r = capture_external_observation(&observation, &context())
        .unwrap()
        .value
        .data;
    assert!(r.validation.contains("not_replayed"));
    assert_eq!(
        r.original_digest,
        ContentDigest::sha256(observation.original_utf8.as_bytes())
    );
    let mut public = context();
    public.write_visibility = CacheVisibility::Public;
    assert!(capture_external_observation(&observation, &public).is_err());
    assert!(capture_reference(&reference(&["1"]), &public).is_err());
}

#[test]
fn root_windows_bind_sources_keep_failures_and_adaptive_precision() {
    let (s, sm) = state(&["0", "1", "0"], "3");
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"normalization":"sum_xi_equals_sqrt_log_lambda_squared","eigenpair_content_digest":sm.content_digest.0}),
    );
    let dep = |m: &ArtifactManifest| DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    };
    sec.dependencies.push(dep(&sm));
    let payload = json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":7,"discovery_mode":"reference_seeded_audit","reference_seeds_used":true,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":"14","adaptive_precision":{"target_precision_bits":256,"evaluation_precision_bits":320,"verification_precision_bits":384}}},{"status":"failed","details":{"reason":"missing"}},{"status":"approximate","details":{"value":"21"}}]});
    let (mut rm, rb) = source("ccm_root_refinement", payload.clone());
    rm.dependencies.push(dep(&sec));
    let approved = vec![
        rm.content_digest.clone(),
        sec.content_digest.clone(),
        sm.content_digest.clone(),
    ];
    let r = RetainedRoots::from_payload(&rm, &rb, &sec, &sb, &s, &approved).unwrap();
    let options = TransformOptions::for_roots(&s, &r);
    assert_eq!(options.working_precision_bits, 448);
    let report = capture_root_window(&r, &context()).unwrap().value.data;
    assert_eq!(
        report.points.iter().map(|r| r.ordinal).collect::<Vec<_>>(),
        vec![7, 8, 9]
    );
    assert_eq!(report.missing_count, 1);
    assert_eq!(report.source_acquisition["reference_seeds_used"], true);
    let transforms = capture_transforms_at_roots(&s, &r, &options, &context())
        .unwrap()
        .value
        .data;
    assert_eq!(transforms.rows[1].outcome, "missing_input");
    assert!(capture_transforms_at_roots(
        &s,
        &r,
        &TransformOptions {
            maximum_estimated_output_bytes: 1,
            ..options
        },
        &context()
    )
    .is_err());
    let mut altered = rb.clone();
    altered[0] = b' ';
    assert!(RetainedRoots::from_payload(&rm, &altered, &sec, &sb, &s, &approved).is_err());
    // Fixed-point schema 6 must carry the same exact source as its dependency.
    for (digest, accepted) in [
        (Some(sec.content_digest.0.clone()), true),
        (Some(ContentDigest::sha256(b"different source").0), false),
        (None, false),
    ] {
        let mut fixed = payload.clone();
        fixed["schema_version"] = json!(6);
        if let Some(digest) = digest {
            fixed["secular_source_content_digest"] = json!(digest);
        }
        let (mut manifest, bytes) = source("ccm_root_refinement", fixed);
        manifest.dependencies.push(dep(&sec));
        let mut allowed = approved.clone();
        allowed.push(manifest.content_digest.clone());
        assert_eq!(
            RetainedRoots::from_payload(&manifest, &bytes, &sec, &sb, &s, &allowed).is_ok(),
            accepted
        );
    }
    let mut bad = payload;
    bad["schema_version"] = json!(99);
    let (mut bm, bb) = source("ccm_root_refinement", bad);
    bm.dependencies.push(dep(&sec));
    let mut allowed = approved.clone();
    allowed.push(bm.content_digest.clone());
    assert!(RetainedRoots::from_payload(&bm, &bb, &sec, &sb, &s, &allowed).is_err());
    sec.dependencies.clear();
    assert!(RetainedRoots::from_payload(&rm, &rb, &sec, &sb, &s, &approved).is_err());
}
#[test]
fn nonzero_carrier_mode_has_correct_removable_value_and_derivative() {
    use rug::{float::Constant, Float};
    let (s, _) = state(&["1", "0", "1"], "3");
    let p = 256;
    let l = Float::with_val(p, 9).ln();
    let t = Float::with_val(p, Constant::Pi) * 2u32 / l;
    let spec = DatasetSpec {
        schema_version: 1,
        role: "evaluation_points".into(),
        attribution: "cosine closed form".into(),
        coordinate: "mellin_t".into(),
        precision_bits: p,
        points: vec![EvaluationPoint {
            ordinal: 1,
            value: Some(t.to_string()),
            source_status: "supplied".into(),
        }],
    };
    let value = capture_dataset(&spec, &context()).unwrap().value;
    let (m, b) = source(
        "research_reference_dataset",
        serde_json::to_value(value).unwrap(),
    );
    let d = RetainedDataset::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let r =
        capture_transforms_at_dataset(&s, &d, &TransformOptions::for_dataset(&s, &d), &context())
            .unwrap()
            .value
            .data;
    let lf = 9f64.ln();
    assert!((number(r.rows[0].value.as_ref().unwrap()) - (lf / 2.0).sqrt()).abs() < 1e-14);
    assert!(
        (number(r.rows[0].derivative.as_ref().unwrap())
            - lf.powf(1.5) / (4.0 * 2f64.sqrt() * std::f64::consts::PI))
            .abs()
            < 1e-14
    );
}
