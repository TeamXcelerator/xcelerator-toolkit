use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crate::ccm::published_source_fixtures as published_sources;

fn manifest(kind: &str, params: Value, payload: &[u8]) -> ArtifactManifest {
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: "source-admission-fixture-v1".into(),
        resolved_mathematical_parameters: params,
        normalization: Some(
            if kind == "ccm_tau_matrix" {
                "symmetric_row_major"
            } else {
                "sum_xi_equals_sqrt_log_lambda_squared"
            }
            .into(),
        ),
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let digest = ContentDigest::sha256(payload);
    ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey {
            kind: kind.into(),
            logical_key: format!("fixture/{kind}"),
            parameters_digest: semantic.digest().unwrap(),
        },
        content_digest: digest.clone(),
        size_bytes: payload.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: payload.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.17.0").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.17.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::from([(
            SEMANTIC_KEY_MANIFEST_TAG.into(),
            serde_json::to_string(&semantic).unwrap(),
        )]),
        provenance_digest: None,
    }
}

fn rewrite(m: &mut ArtifactManifest, edit: impl FnOnce(&mut Value)) {
    let mut s: SemanticKeyEnvelope =
        serde_json::from_str(&m.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
    edit(&mut s.resolved_mathematical_parameters);
    m.key.parameters_digest = s.digest().unwrap();
    m.tags.insert(
        SEMANTIC_KEY_MANIFEST_TAG.into(),
        serde_json::to_string(&s).unwrap(),
    );
    m.tags.remove(REMOTE_CANONICAL_MANIFEST_TAG);
    m.provenance_digest = None;
}

fn fixture(legacy: bool) -> (ArtifactManifest, ArtifactManifest, Vec<u8>, Vec<u8>) {
    let matrix_bytes = serde_json::to_vec(
        &json!({"schema_version":2,"lambda_squared":"13","n_modes":1,"precision_bits":128,
        "entries":["1","0","0","0","2","0","0","0","3"]}),
    )
    .unwrap();
    let tau = published_sources::published(
        manifest(
            "ccm_tau_matrix",
            json!({"lambda_squared":"13","n_modes":1,"precision_bits":128}),
            &matrix_bytes,
        ),
        &[],
    );
    let identity = if legacy {
        serde_json::to_value(DependencyRef {
            key: tau.key.clone(),
            content_digest: tau.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        })
        .unwrap()
    } else {
        serde_json::to_value(numerical_source_identity(
            &tau,
            "tau",
            "stored-tau-matrix-v1",
        ))
        .unwrap()
    };
    let state_bytes = serde_json::to_vec(
        &json!({"schema_version":3,"lambda_squared":"13","n_modes":1,"precision_bits":128,
        "force_even":true,"eigenvalue":"2","eigenvector":["0","1","0"]}),
    )
    .unwrap();
    let state = published_sources::published(
        manifest(
            "ccm_weil_eigenpair",
            json!({"lambda_squared":"13","n_modes":1,"precision_bits":128,"tau_source_identity":identity}),
            &state_bytes,
        ),
        &[&tau],
    );
    (state, tau, state_bytes, matrix_bytes)
}

#[test]
fn value_identity_admits_published_variants_without_weakening_provenance() {
    for legacy in [false, true] {
        let (state, tau, _, _) = fixture(legacy);
        for visibility in [
            CacheVisibility::Local,
            CacheVisibility::Public,
            CacheVisibility::Private,
        ] {
            let mut variant = tau.clone();
            variant.visibility = visibility;
            variant.producer_toolkit_version = ToolkitVersion::parse("0.17.1").unwrap();
            variant.key.logical_key = "adopted/alias".into();
            let variant = published_sources::published(variant, &[]);
            assert!(!manifest_depends_on(&state, &variant).unwrap());
            assert!(declared_tau_matches(&state, &variant, None).unwrap());
            assert!(!manifest_depends_on(&state, &variant).unwrap());
        }
        // A source predating explicit numerical parent declarations continues
        // to require exact historical ancestry.
        let mut historical = state.clone();
        rewrite(&mut historical, |p| {
            p.as_object_mut().unwrap().remove("tau_source_identity");
        });
        let historical = published_sources::published(historical, &[&tau]);
        assert!(!declared_tau_matches(&historical, &tau, None).unwrap());
        assert!(manifest_depends_on(&historical, &tau).unwrap());
    }
}

#[test]
fn numerical_admission_rejects_changed_values_semantics_dimensions_and_versions() {
    let (state, tau, _, _) = fixture(false);
    for (field, value) in [
        ("role", json!("other")),
        ("meaning", json!("different-codec")),
        ("kind", json!("ccm_weil_eigenpair")),
        (
            "content_digest",
            json!(ContentDigest::sha256(b"different values")),
        ),
    ] {
        let mut wrong = state.clone();
        rewrite(&mut wrong, |p| p["tau_source_identity"][field] = value);
        assert!(declared_tau_matches(&wrong, &tau, None).is_err());
    }
    for (field, value) in [
        ("lambda_squared", json!("17")),
        ("n_modes", json!(2)),
        ("precision_bits", json!(256)),
    ] {
        let mut wrong = tau.clone();
        rewrite(&mut wrong, |p| p[field] = value);
        assert!(declared_tau_matches(&state, &wrong, None).is_err());
    }
    let mut wrong = tau.clone();
    wrong.minimum_reader_version = ToolkitVersion::parse("999.0.0").unwrap();
    assert!(declared_tau_matches(&state, &wrong, None).is_err());
    let mut wrong = tau.clone();
    wrong.quality = CacheQuality::Quarantined;
    assert!(declared_tau_matches(&state, &wrong, None).is_err());
    let mut wrong = state.clone();
    wrong
        .tags
        .insert(SEMANTIC_KEY_MANIFEST_TAG.into(), "{}".into());
    assert!(declared_tau_matches(&wrong, &tau, None).is_err());
    let mut policy = policy();
    policy.allowed_visibilities = vec![CacheVisibility::Private];
    assert!(declared_tau_matches(&state, &tau, Some(&policy)).is_err());
}

#[test]
fn secular_admission_preserves_consumed_state_key_without_requiring_producer_identity() {
    let (state, _, _, _) = fixture(false);
    let sec = published_sources::published(
        manifest(
            "ccm_secular_source",
            json!({
                "lambda_squared":"13", "n_modes":1, "precision_bits":128,
                "eigenpair_content_digest":state.content_digest,
                "eigenpair_semantic_digest":state.key.parameters_digest
            }),
            b"secular fixture",
        ),
        &[&state],
    );
    let mut variant = state.clone();
    variant.producer_toolkit_version = ToolkitVersion::parse("0.17.1").unwrap();
    let variant = published_sources::published(variant, &[]);
    assert!(!manifest_depends_on(&sec, &variant).unwrap());
    assert!(declared_secular_matches(&sec, &variant, None).unwrap());
    for field in ["eigenpair_content_digest", "eigenpair_semantic_digest"] {
        let mut bad = sec.clone();
        rewrite(&mut bad, |p| {
            p[field] = json!(ContentDigest::sha256(b"wrong source"))
        });
        assert!(declared_secular_matches(&bad, &state, None).is_err());
        let mut missing = sec.clone();
        rewrite(&mut missing, |p| {
            p.as_object_mut().unwrap().remove(field);
        });
        assert!(declared_secular_matches(&missing, &state, None).is_err());
    }
    let mut bad = variant;
    bad.maximum_reader_version = Some(ToolkitVersion::parse("0.1.0").unwrap());
    assert!(declared_secular_matches(&sec, &bad, None).is_err());
}

fn policy() -> CachePolicy {
    CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    }
}

#[test]
fn direct_state_and_matrix_decoders_bind_declared_payload_shape() {
    use crate::ccm::{retained_evidence::RetainedMatrix, state_geometry::RetainedState};

    let (state, tau, state_bytes, matrix) = fixture(false);
    for (field, value) in [
        ("lambda_squared", json!("17")),
        ("n_modes", json!(2)),
        ("precision_bits", json!(256)),
    ] {
        let mut bad_state = state.clone();
        rewrite(&mut bad_state, |p| p[field] = value.clone());
        let err = RetainedState::from_payload(
            &bad_state,
            &state_bytes,
            std::slice::from_ref(&bad_state.content_digest),
        )
        .expect_err("state declaration must bind the payload shape");
        assert!(err.to_string().contains(field), "{err}");
        let mut bad_matrix = tau.clone();
        rewrite(&mut bad_matrix, |p| p[field] = value);
        let err = RetainedMatrix::from_payload(
            &bad_matrix,
            &matrix,
            std::slice::from_ref(&bad_matrix.content_digest),
        )
        .expect_err("matrix declaration must bind the payload shape");
        assert!(err.to_string().contains(field), "{err}");
    }
}

#[test]
fn direct_readers_honor_team_visibility_and_canonical_reader_windows() {
    use crate::ccm::retained_evidence::RetainedMatrix;

    let (_, tau, _, bytes) = fixture(false);
    let allowed = std::slice::from_ref(&tau.content_digest);
    let mut team = tau.clone();
    team.visibility = CacheVisibility::Team;
    let mut team_only = policy();
    team_only.allowed_visibilities = vec![CacheVisibility::Team];
    RetainedMatrix::from_payload(&team, &bytes, allowed).unwrap();
    RetainedMatrix::from_payload_with_policy(&team, &bytes, allowed, &team_only).unwrap();
    assert!(RetainedMatrix::from_payload_with_policy(&team, &bytes, allowed, &policy()).is_err());
    for (minimum, maximum) in [("999.0.0", None), ("0.16.0", Some("0.17.1"))] {
        let mut inconsistent = tau.clone();
        let mut canonical = retained_canonical_manifest(&tau).unwrap().unwrap();
        canonical.minimum_reader_version = ToolkitVersion::parse(minimum).unwrap();
        canonical.maximum_reader_version = maximum.map(|v| ToolkitVersion::parse(v).unwrap());
        inconsistent.provenance_digest = Some(canonical.digest().unwrap());
        inconsistent.tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.into(),
            serde_json::to_string(&canonical).unwrap(),
        );
        let error = policy().assess(&inconsistent).reasons.join("; ");
        assert!(
            error.contains("widens the canonical reader window"),
            "{error}"
        );
        assert!(RetainedMatrix::from_payload(&inconsistent, &bytes, allowed).is_err());
    }
    let mut tighter = tau.clone();
    tighter.minimum_reader_version = policy().current_toolkit_version;
    tighter.maximum_reader_version = Some(tighter.minimum_reader_version.clone());
    assert!(policy().accepts(&tighter));
    tighter.producer_toolkit_version = ToolkitVersion::parse("0.17.3").unwrap();
    assert!(!policy().accepts(&tighter));
}

#[test]
fn published_parent_selection_applies_policy_before_selecting_aliases() {
    let (state, tau, _, _) = fixture(false);
    let p = policy();
    let mut cache = ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: Some(&p),
        ordered_overlays: vec![],
        mode: ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let mut bad = tau.clone();
    bad.visibility = CacheVisibility::Private;
    let error = resolve_manifest_sources(&state, std::slice::from_ref(&bad), &cache)
        .unwrap_err()
        .to_string();
    assert!(error.contains("visibility"), "{error}");
    assert_eq!(
        resolve_manifest_sources(&state, &[bad.clone(), tau.clone()], &cache).unwrap(),
        vec![tau.clone()]
    );
    let opens = Arc::new(AtomicUsize::new(0));
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(ReadStore {
            values: vec![(bad, vec![]), (tau.clone(), vec![])],
            opens: opens.clone(),
        }),
    }]);
    cache.resolver = Some(&resolver);
    assert_eq!(
        resolve_manifest_sources(&state, &[], &cache).unwrap(),
        vec![tau]
    );
    assert_eq!(opens.load(Ordering::SeqCst), 0);
}

#[test]
fn direct_readers_apply_current_and_caller_policies_before_payload_access() {
    use crate::ccm::{
        prefix::{RetainedEvenEigenpair, RetainedEvenMatrix},
        retained_evidence::{RetainedDataset, RetainedMatrix, RetainedReference, RetainedRoots},
        state_geometry::RetainedState,
    };
    struct NoIo;
    impl std::io::Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("payload read before admission")
        }
    }
    impl std::io::Seek for NoIo {
        fn seek(&mut self, _: std::io::SeekFrom) -> std::io::Result<u64> {
            panic!("payload seek before admission")
        }
    }
    let (state, tau, state_bytes, matrix_bytes) = fixture(false);
    let decoded = RetainedState::from_payload(
        &state,
        &state_bytes,
        std::slice::from_ref(&state.content_digest),
    )
    .unwrap();
    for maximum in [false, true] {
        let mut bad = tau.clone();
        if maximum {
            bad.maximum_reader_version = Some(ToolkitVersion::parse("0.17.1").unwrap());
        } else {
            bad.minimum_reader_version = ToolkitVersion::parse("999.0.0").unwrap();
        }
        let allowed = std::slice::from_ref(&bad.content_digest);
        let mut bad_root = manifest("ccm_root_refinement", json!({}), b"unread");
        bad_root.minimum_reader_version = bad.minimum_reader_version.clone();
        bad_root.maximum_reader_version = bad.maximum_reader_version.clone();
        let errors = [
            RetainedMatrix::from_payload(&bad, &matrix_bytes, allowed).err(),
            RetainedMatrix::from_reader(&bad, &mut NoIo, allowed).err(),
            RetainedState::from_payload(&bad, b"unread", allowed).err(),
            RetainedReference::from_payload(&bad, b"unread", allowed).err(),
            RetainedDataset::from_payload(&bad, b"unread", allowed).err(),
            RetainedRoots::from_payload(&bad_root, b"unread", &bad, b"unread", &decoded, allowed)
                .err(),
            RetainedEvenMatrix::from_payload(&bad, b"unread", allowed).err(),
            RetainedEvenEigenpair::from_payload(&bad, b"unread", allowed).err(),
        ];
        for error in errors {
            let error = error.unwrap().to_string();
            assert!(
                error.contains(if maximum {
                    "maximum reader"
                } else {
                    "minimum reader"
                }),
                "{error}"
            );
        }
    }
    let mut p = policy();
    p.allowed_visibilities = vec![CacheVisibility::Private];
    let allowed = std::slice::from_ref(&tau.content_digest);
    let resources = xc_core::RunResources::new(32 << 20).unwrap();
    let errors = [
        RetainedMatrix::from_payload_with_policy(&tau, &matrix_bytes, allowed, &p).err(),
        RetainedMatrix::from_reader_with_resources_and_policy(
            &tau, &mut NoIo, allowed, &resources, &p,
        )
        .err(),
        RetainedState::from_payload_with_policy(
            &state,
            &state_bytes,
            std::slice::from_ref(&state.content_digest),
            &p,
        )
        .err(),
        RetainedReference::from_payload_with_policy(&tau, b"unread", allowed, &p).err(),
        RetainedDataset::from_payload_with_policy(&tau, b"unread", allowed, &p).err(),
        RetainedRoots::from_payload_with_policy(
            &tau, b"unread", &tau, b"unread", &decoded, allowed, &p,
        )
        .err(),
        RetainedEvenMatrix::from_payload_with_policy(&tau, b"unread", allowed, &p).err(),
        RetainedEvenEigenpair::from_payload_with_policy(&tau, b"unread", allowed, &p).err(),
    ];
    for error in errors {
        assert!(error.unwrap().to_string().contains("visibility"));
    }
    RetainedMatrix::from_payload_with_policy(&tau, &matrix_bytes, allowed, &policy()).unwrap();
}

#[test]
fn root_secular_admission_binds_values_and_acquisition_without_producer_metadata() {
    let secular = published_sources::published(
        manifest(
            "ccm_secular_source",
            json!({
                "lambda_squared":"13", "n_modes":1, "precision_bits":128,"force_even":true
            }),
            b"secular fixture",
        ),
        &[],
    );
    for seeded in [false, true] {
        let root = published_sources::published(
            manifest(
                if seeded {
                    "ccm_root_refinement"
                } else {
                    "ccm_root_discovery_window"
                },
                json!({
                    "lambda_squared":"13", "n_modes":1, "precision_bits":128,"force_even":true,
                    "discovery_mode":if seeded {"reference_seeded_refinement"} else {"independent"},
                    "reference_seeds_used":seeded,"secular_source_content_digest":secular.content_digest
                }),
                b"root fixture",
            ),
            &[&secular],
        );
        let mut variant = secular.clone();
        variant.producer_toolkit_version = ToolkitVersion::parse("0.17.2").unwrap();
        let variant = published_sources::published(variant, &[]);
        assert!(!manifest_depends_on(&root, &variant).unwrap());
        assert!(declared_root_secular_matches(&root, &variant, None).unwrap());
        for (field, value) in [
            ("reference_seeds_used", json!(!seeded)),
            ("discovery_mode", json!("unknown")),
            (
                "secular_source_content_digest",
                json!(ContentDigest::sha256(b"foreign")),
            ),
            ("secular_source_content_digest", json!(null)),
            ("precision_bits", json!(256)),
            ("force_even", json!(false)),
        ] {
            let mut bad = root.clone();
            rewrite(&mut bad, |p| p[field] = value);
            assert!(
                declared_root_secular_matches(&bad, &secular, None).is_err(),
                "{field}"
            );
        }
        let mut legacy = root.clone();
        rewrite(&mut legacy, |p| {
            p.as_object_mut()
                .unwrap()
                .remove("secular_source_content_digest");
        });
        let legacy = published_sources::published(legacy, &[&secular]);
        assert!(!declared_root_secular_matches(&legacy, &secular, None).unwrap());
        assert!(manifest_depends_on(&legacy, &secular).unwrap());
        assert!(!manifest_depends_on(&legacy, &variant).unwrap());
    }
}
struct ReadStore {
    values: Vec<(ArtifactManifest, Vec<u8>)>,
    opens: Arc<AtomicUsize>,
}
impl CacheStore for ReadStore {
    fn name(&self) -> &str {
        "fixture"
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        CacheVisibility::Local
    }
    fn put(
        &self,
        _: &ArtifactDraft,
        _: &[u8],
    ) -> std::result::Result<ArtifactManifest, CacheError> {
        panic!("read-only fixture")
    }
    fn candidates(
        &self,
        key: &ArtifactKey,
    ) -> std::result::Result<Vec<ArtifactManifest>, CacheError> {
        Ok(self
            .values
            .iter()
            .filter(|(m, _)| &m.key == key)
            .map(|(m, _)| m.clone())
            .collect())
    }
    fn identity_candidates(
        &self,
        id: &PayloadDependencyIdentity,
    ) -> std::result::Result<Vec<ArtifactManifest>, CacheError> {
        Ok(self
            .values
            .iter()
            .filter(|(m, _)| {
                m.key.parameters_digest == id.semantic_digest
                    && m.provenance_digest.as_ref() == Some(&id.manifest_digest)
            })
            .map(|(m, _)| m.clone())
            .collect())
    }
    fn read_payload_to(
        &self,
        m: &ArtifactManifest,
        writer: &mut dyn std::io::Write,
    ) -> std::result::Result<(), CacheError> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        let (_, bytes) = self
            .values
            .iter()
            .find(|(v, _)| v.key == m.key && v.content_digest == m.content_digest)
            .unwrap();
        writer.write_all(bytes)?;
        Ok(())
    }
}

#[test]
fn retained_recovery_accepts_equivalent_tau_and_rejects_bad_join_before_any_payload_read() {
    use crate::ccm::convergence_capture::finite_capture::RetainedFiniteSources;
    for bad in [false, true] {
        let (mut state, mut tau, state_bytes, matrix_bytes) = fixture(false);
        tau.producer_toolkit_version = ToolkitVersion::parse("0.17.1").unwrap();
        let tau = published_sources::published(tau, &[]);
        assert!(!manifest_depends_on(&state, &tau).unwrap());
        if bad {
            rewrite(&mut state, |p| {
                p["tau_source_identity"]["content_digest"] =
                    json!(ContentDigest::sha256(b"wrong matrix"))
            });
        }
        let opens = Arc::new(AtomicUsize::new(0));
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(ReadStore {
                values: vec![(state.clone(), state_bytes), (tau.clone(), matrix_bytes)],
                opens: opens.clone(),
            }),
        }]);
        let policy = policy();
        let cache = ArtifactCacheContext {
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["fixture".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: false,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        if !bad {
            // Even an error in a later optional root dependency must be found
            // before reading the otherwise valid state and large matrix.
            let root = manifest(
                "ccm_root_refinement",
                json!({
                    "lambda_squared":"13", "n_modes":1, "precision_bits":128
                }),
                b"unread root",
            );
            let incomplete = RetainedFiniteSources::from_manifests(
                &[state.clone(), tau.clone(), root],
                true,
                &cache,
            );
            assert!(incomplete
                .err()
                .unwrap()
                .to_string()
                .contains("secular source manifest missing"));
            assert_eq!(opens.load(Ordering::SeqCst), 0);
        }
        let result = RetainedFiniteSources::from_manifests(&[state, tau], true, &cache);
        if bad {
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("Tau payload differs"));
            assert_eq!(opens.load(Ordering::SeqCst), 0);
        } else {
            result.unwrap();
            assert_eq!(opens.load(Ordering::SeqCst), 2);
        }
    }
}
