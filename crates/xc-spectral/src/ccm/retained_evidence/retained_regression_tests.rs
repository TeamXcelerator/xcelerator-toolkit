use super::*;
fn state() -> RetainedState {
    capacity_state(192, &["1", "-4", "2"])
}
fn capacity_state(p: u32, coefficients: &[&str]) -> RetainedState {
    let bytes=serde_json::to_vec(&json!({"schema_version":3,"lambda_squared":"10","n_modes":1,"precision_bits":p,"force_even":false,"eigenvalue":"1","eigenvector":coefficients})).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(
            "ccm_weil_eigenpair",
            "remaining-mixed-state",
            b"remaining-mixed-state",
        )
        .unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    RetainedState::from_payload(&manifest, &bytes, &[digest]).unwrap()
}

#[test]
fn capacity_transform_guard_policy_is_bounded_and_canonical() {
    for text in ["4096", "8192", "1048576"] {
        assert_eq!(
            transform_math::parse_maximum_guard_bits(text)
                .unwrap()
                .to_string(),
            text
        );
    }
    for text in [
        "", " 8192", "+8192", "08192", "4097", "2048", "2097152", "NaN",
    ] {
        assert!(
            transform_math::parse_maximum_guard_bits(text).is_err(),
            "{text}"
        );
    }
    let miss: Result<()> = Err(CacheError::NotFound("missing".into()).into());
    assert!(transform::retry_extended(
        &miss,
        false,
        ArtifactExecutionCacheMode::RequireReuse
    ));
    assert!(!transform::retry_extended(
        &miss,
        false,
        ArtifactExecutionCacheMode::PreferReuse
    ));
    let corrupt: Result<()> = Err(CacheError::InvalidManifest("bad".into()).into());
    assert!(!transform::retry_extended(
        &corrupt,
        false,
        ArtifactExecutionCacheMode::RequireReuse
    ));
}

#[test]
fn capacity_transform_extends_only_exhausted_arithmetic_and_preserves_points() {
    let source = capacity_state(5000, &["0", "1", "0"]);
    let p = source.precision + 64;
    let work = p * 2 + 128;
    let t = Float::with_val(
        source.precision,
        Float::with_val(work, rug::float::Constant::Pi) * 2u32 / Float::with_val(work, 10).ln(),
    );
    for root in [false, true] {
        let old = transform::terms_with_guard_limit(&source, &t, p, 4096, root).unwrap_err();
        assert!(old.is::<transform::GuardExhausted>(), "{old:#}");
        let extended = transform::terms_with_guard_limit(&source, &t, p, 8192, root).unwrap();
        assert!(extended.0.is_finite() && extended.1.is_finite());
        assert!(!extended.0.is_zero());
    }
    let automatic = transform_terms(&source, &t, p).unwrap();
    let extended = transform::terms_with_guard_limit(&source, &t, p, 8192, false).unwrap();
    assert_eq!(dec(&automatic.0), dec(&extended.0));
    let t = Float::with_val(source.precision, 1);
    let old = transform::terms_with_guard_limit(&source, &t, p, 4096, false).unwrap();
    let extended = transform::terms_with_guard_limit(&source, &t, p, 8192, false).unwrap();
    assert_eq!(
        [dec(&old.0), dec(&old.1), dec(&old.2), dec(&old.3)],
        [
            dec(&extended.0),
            dec(&extended.1),
            dec(&extended.2),
            dec(&extended.3)
        ]
    );
}

#[test]
fn capacity_transform_budget_refusal_is_not_arithmetic_exhaustion() {
    let source = state();
    let resources = xc_core::RunResources::new(1).unwrap();
    let _scope = resources.enter();
    let error =
        transform::terms_with_guard_limit(&source, &Float::with_val(192, 1), 256, 8192, false)
            .unwrap_err();
    assert!(!error.is::<transform::GuardExhausted>());
    assert!(error.to_string().contains("memory"), "{error}");
    assert_eq!(resources.usage().working_bytes, 0);
}

#[test]
fn capacity_transform_child_admission_does_not_wait_on_parent() {
    let resources = xc_core::RunResources::new(1_000_000).unwrap();
    let retained_parent_work = resources.reserve_working(999_999).unwrap();
    let child_resources = resources.clone();
    let (send, receive) = std::sync::mpsc::channel();
    let child = std::thread::spawn(move || {
        let source = state();
        let _scope = child_resources.enter();
        let result =
            transform::terms_with_guard_limit(&source, &Float::with_val(192, 1), 256, 8192, false);
        send.send(result.map_err(|error| error.to_string()))
            .unwrap();
    });
    let result = receive.recv_timeout(std::time::Duration::from_secs(2));
    // Release even on an unexpected wait so a regression cannot strand a test.
    drop(retained_parent_work);
    child.join().unwrap();
    let error = result
        .expect("nested transform admission waited for its parent")
        .unwrap_err();
    assert!(error.contains("memory"), "{error}");
    assert_eq!(resources.usage().working_bytes, 0);
}

#[test]
fn capacity_indexed_transform_preserves_legacy_bytes_and_reuses_extended_identity() {
    let root = crate::fresh_test_dir("capacity-transform-cache");
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "capacity-local",
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
    let context = |mode| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["capacity-local".into()],
        mode,
        write_on_miss: mode.writes_computed_artifacts(),
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let source = capacity_state(5000, &["0", "1", "0"]);
    let options = TransformOptions::for_source(&source);
    let mut dataset = DatasetSpec {
        schema_version: 1,
        role: "evaluation_points".into(),
        attribution: "finite transform fixture".into(),
        coordinate: "mellin_t".into(),
        precision_bits: source.precision,
        points: vec![EvaluationPoint {
            ordinal: 1,
            value: Some("1".into()),
            source_status: "stored_point".into(),
        }],
    };
    let exhausted = std::cell::Cell::new(false);
    let old = transforms_with_guard_limit(
        &source,
        &dataset,
        &options,
        &[],
        false,
        &context(ArtifactExecutionCacheMode::PreferReuse),
        4096,
        &exhausted,
    )
    .unwrap();
    let warm = transforms(
        &source,
        &dataset,
        &options,
        &[],
        false,
        &context(ArtifactExecutionCacheMode::RequireReuse),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&old.value).unwrap(),
        serde_json::to_vec(&warm.value).unwrap()
    );
    assert_eq!(
        old.produced_manifest.as_ref().unwrap().key,
        warm.reused_manifest.as_ref().unwrap().key
    );
    let work = 11000;
    let t = Float::with_val(
        source.precision,
        Float::with_val(work, rug::float::Constant::Pi) * 2u32 / Float::with_val(work, 10).ln(),
    );
    dataset.points[0].value = Some(dec(&t));
    let cold = transforms(
        &source,
        &dataset,
        &options,
        &[],
        false,
        &context(ArtifactExecutionCacheMode::PreferReuse),
    )
    .unwrap();
    assert_eq!(cold.value.request["maximum_transform_guard_bits"], 8192);
    let warm = transforms(
        &source,
        &dataset,
        &options,
        &[],
        false,
        &context(ArtifactExecutionCacheMode::RequireReuse),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&cold.value).unwrap(),
        serde_json::to_vec(&warm.value).unwrap()
    );
    assert_eq!(
        cold.produced_manifest.as_ref().unwrap().key,
        warm.reused_manifest.as_ref().unwrap().key
    );

    use crate::ccm::extended_research::{
        capture_extended, ExtensionOptions, ExternalResearchInputs,
    };
    let mut input: ExternalResearchInputs = serde_json::from_value(json!({
        "schema_version":1,"source_eigenpair":source.manifest.content_digest,
        "lambda_squared":source.cutoff,"n_modes":source.modes,"precision_bits":source.precision,
        "convention_id":"finite fixture","definition_digest":ContentDigest::sha256(b"capacity-transform"),
        "approximation_scope":"finite stored points",
        "reference_jets":[{"ordinal":1,"t":"1",
            "reference_window":{"value":"0","derivative":"0"},
            "reference_full":{"value":"0","derivative":"0"},
            "exterior_tail":{"value":"0","derivative":"0"},
            "endpoint_tail_part":null,"fitted_interior_parts":[],
            "source_value_error":null,"tail_value_error":null,
            "source_derivative_error":null,"root_separation_radius":null}]
    })).unwrap();
    let options = ExtensionOptions::for_source(&source);
    let resolved = capture_extended(
        "signed_transform",
        &source,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(ArtifactExecutionCacheMode::PreferReuse),
    )
    .unwrap();
    assert_eq!(resolved.value.request["maximum_transform_guard_bits"], 4096);
    let warm = capture_extended(
        "signed_transform",
        &source,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(ArtifactExecutionCacheMode::RequireReuse),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&resolved.value).unwrap(),
        serde_json::to_vec(&warm.value).unwrap()
    );
    input.reference_jets[0].t = dec(&t);
    let cold = capture_extended(
        "signed_transform",
        &source,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(ArtifactExecutionCacheMode::PreferReuse),
    )
    .unwrap();
    assert_eq!(cold.value.request["maximum_transform_guard_bits"], 8192);
    let warm = capture_extended(
        "signed_transform",
        &source,
        None,
        None,
        Some(&input),
        &options,
        &[],
        &context(ArtifactExecutionCacheMode::RequireReuse),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&cold.value).unwrap(),
        serde_json::to_vec(&warm.value).unwrap()
    );
    assert_eq!(
        cold.produced_manifest.as_ref().unwrap().key,
        warm.reused_manifest.as_ref().unwrap().key
    );
}
#[test]
fn mixed_parity_transform_uses_the_root_convention() {
    let s = state();
    let p = 192;
    let work = 384;
    // Independent algebra: residues [1,-4,2] at [-w,0,w] give
    // numerator -t^2+w*t+4w^2, root w*(1+sqrt(17))/2.
    let l = Float::with_val(work, 10).ln();
    let w = Float::with_val(work, rug::float::Constant::Pi) * 2u32 / &l;
    let t = Float::with_val(p, &w * (Float::with_val(work, 17).sqrt() + 1u32) / 2u32);
    let root = transform_math::measure_root(&s, &t, p).unwrap().unwrap();
    let plus = transform_math::measure(&s, &t, p).unwrap().unwrap();
    assert!(root.value.lower().clone().abs() < Float::with_val(root.precision, 1) >> 180u32);
    assert!(root.value.upper().clone().abs() < Float::with_val(root.precision, 1) >> 180u32);
    assert!(plus.value.lower().clone().abs() > Float::with_val(plus.precision, 0.01));
    // Independent closed-form integral derivative at the retained t point.
    let t = Float::with_val(work, &t);
    let half = Float::with_val(work, &l / 2u32);
    let phase = Float::with_val(work, &t * &half);
    let sine = phase.clone().sin();
    let cosine = phase.clone().cos();
    let mut derivative = Float::with_val(work, 0);
    for (j, c) in [(-1, 1), (0, -4), (1, 2)] {
        let q = Float::with_val(
            work,
            &phase - Float::with_val(work, rug::float::Constant::Pi) * j,
        );
        derivative += (Float::with_val(work, &cosine / &q)
            - Float::with_val(work, &sine / Float::with_val(work, &q * &q)))
            * c;
    }
    derivative *= half;
    derivative *= l.sqrt();
    derivative /= Float::with_val(work, 21).sqrt();
    derivative = -derivative; // center coefficient fixes orientation negative
    let actual = Float::with_val(work, root.derivative.midpoint_point().lower());
    assert!((actual - derivative).abs() < Float::with_val(work, 1) >> 180u32);
}
#[test]
fn aliases_share_numeric_options_and_cutoffs() {
    assert!(equal_cutoff("13", "13.0").unwrap());
    assert_eq!(
        canonical_decimal("0.10").unwrap(),
        canonical_decimal("1e-1").unwrap()
    );
    assert!(!equal_cutoff("13", "13.000000000000000001").unwrap());
    assert_eq!(
        energy::residual_normalization("0", 128).unwrap(),
        "absolute_residual_per_coefficient_l2_norm"
    );
    assert_eq!(
        energy::residual_normalization("1", 128).unwrap(),
        "relative_to_abs_eigenvalue_and_coefficient_l2_norm"
    );
}
