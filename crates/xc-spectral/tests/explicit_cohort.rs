#![cfg(feature = "hp")]
use serde_json::{json, Value};
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::explicit_cohort::*;

struct NoMatrixReads(FilesystemCacheStore);
impl CacheStore for NoMatrixReads {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn writable(&self) -> bool {
        self.0.writable()
    }
    fn visibility(&self) -> CacheVisibility {
        self.0.visibility()
    }
    fn put(&self, draft: &ArtifactDraft, payload: &[u8]) -> Result<ArtifactManifest, CacheError> {
        self.0.put(draft, payload)
    }
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        self.0.candidates(key)
    }
    fn read_payload_to(
        &self,
        m: &ArtifactManifest,
        writer: &mut dyn std::io::Write,
    ) -> Result<(), CacheError> {
        if m.key.kind == "ccm_tau_matrix" {
            return Err(CacheError::InvalidManifest(
                "test forbids every dense matrix payload read".into(),
            ));
        }
        self.0.read_payload_to(m, writer)
    }
}

fn semantic(kind: &str, parameters: Value) -> SemanticKeyEnvelope {
    SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: "explicit-cohort-fixture-v1".into(),
        resolved_mathematical_parameters: parameters,
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
    }
}
fn put(
    store: &dyn CacheStore,
    kind: &str,
    label: &str,
    parameters: Value,
    payload: Value,
    parents: &[&ArtifactManifest],
) -> ArtifactManifest {
    let s = semantic(kind, parameters);
    let key = ArtifactKey {
        kind: kind.into(),
        logical_key: label.into(),
        parameters_digest: s.digest().unwrap(),
    };
    let mut dependencies = parents
        .iter()
        .map(|m| DependencyRef {
            key: m.key.clone(),
            content_digest: m.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        })
        .collect::<Vec<_>>();
    dependencies.sort_by(|a, b| {
        (
            &a.key.kind,
            &a.key.logical_key,
            &a.key.parameters_digest,
            &a.content_digest,
        )
            .cmp(&(
                &b.key.kind,
                &b.key.logical_key,
                &b.key.parameters_digest,
                &b.content_digest,
            ))
    });
    store
        .put(
            &ArtifactDraft {
                schema_version: 1,
                key,
                producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies,
                tags: BTreeMap::from([(
                    SEMANTIC_KEY_MANIFEST_TAG.into(),
                    serde_json::to_string(&s).unwrap(),
                )]),
                provenance_digest: None,
            },
            &serde_json::to_vec(&payload).unwrap(),
        )
        .unwrap()
}
fn fixture(
    store: &dyn CacheStore,
    label: &str,
    n: usize,
    eigenvalue: &str,
    q: usize,
    coefficients: Vec<&str>,
) -> Vec<ArtifactManifest> {
    let dim = 2 * n + 1;
    let mut entries = vec!["0"; dim * dim];
    for i in 0..dim {
        entries[i * dim + i] = "2";
    }
    let tau = put(
        store,
        "ccm_tau_matrix",
        &format!("{label}/tau"),
        json!({"lambda_squared":"13","n_modes":n,"precision_bits":128,"resolved_quadrature_orders":vec![q;n+1],"quadrature_policy":"fixed","scalar_backend":"rug_mpfr"}),
        json!({"schema_version":2,"lambda_squared":"13","n_modes":n,"precision_bits":128,"entries":entries}),
        &[],
    );
    let state = put(
        store,
        "ccm_weil_eigenpair",
        &format!("{label}/state"),
        json!({"lambda_squared":"13","n_modes":n,"precision_bits":128,"force_even":true,"normalization":"sum_xi_equals_sqrt_log_lambda_squared","tau_source_identity":DependencyRef {key:tau.key.clone(),content_digest:tau.content_digest.clone(),required_quality:CacheQuality::Validated}}),
        json!({"schema_version":3,"lambda_squared":"13","n_modes":n,"precision_bits":128,"force_even":true,"eigenvalue":eigenvalue,"eigenvector":coefficients}),
        &[&tau],
    );
    vec![tau, state]
}
fn roots(
    store: &dyn CacheStore,
    source: &mut Vec<ArtifactManifest>,
    first: usize,
    values: Vec<Option<&str>>,
    seeded: bool,
) {
    let state = source.last().unwrap().clone();
    let state_semantic: SemanticKeyEnvelope =
        serde_json::from_str(&state.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
    let n = state_semantic.resolved_mathematical_parameters["n_modes"].clone();
    let sec = put(
        store,
        "ccm_secular_source",
        &format!("{}/secular", state.key.logical_key),
        json!({"lambda_squared":"13","n_modes":n,"precision_bits":128}),
        json!({"schema_version":2,"lambda_squared":"13","n_modes":n,"precision_bits":128,"force_even":true,"normalization":"sum_xi_equals_sqrt_log_lambda_squared","eigenpair_content_digest":state.content_digest.0,"eigenpair_semantic_digest":state.key.parameters_digest.0}),
        &[&state],
    );
    let acquisition = if seeded {
        "reference_seeded_refinement"
    } else {
        "independent"
    };
    let outcomes = values
        .into_iter()
        .map(|v| match v {
            Some(v) => json!({"status":"converged","details":{"value":v}}),
            None => json!({"status":"failed","details":{}}),
        })
        .collect::<Vec<_>>();
    let root = put(
        store,
        if seeded {
            "ccm_root_refinement"
        } else {
            "ccm_root_discovery_window"
        },
        &format!("{}/roots", state.key.logical_key),
        json!({"lambda_squared":"13","n_modes":n,"precision_bits":128,"first_root_index":first,"force_even":true,"reference_seeds_used":seeded,"discovery_mode":acquisition,"secular_source_content_digest":sec.content_digest.0}),
        json!({"schema_version":6,"lambda_squared":"13","n_modes":n,"precision_bits":128,"first_root_index":first,"force_even":true,"reference_seeds_used":seeded,"discovery_mode":acquisition,"completeness":"requested_window","secular_source_content_digest":sec.content_digest.0,"outcomes":outcomes}),
        &[&sec],
    );
    source.extend([sec, root]);
}
fn plan(a: &[ArtifactManifest], b: &[ArtifactManifest], residual: bool) -> FrozenCohortPlan {
    FrozenCohortPlan {
        schema_version: 1,
        members: vec![
            freeze_member("left", a).unwrap(),
            freeze_member("right", b).unwrap(),
        ],
        pairs: vec![CohortPairRequest {
            left: "left".into(),
            right: "right".into(),
            root_ordinals: vec![],
            embedded_residual: residual,
        }],
    }
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
fn context<'a>(resolver: &'a CacheResolver, policy: &'a CachePolicy) -> ArtifactCacheContext<'a> {
    ArtifactCacheContext {
        resolver: Some(resolver),
        reference_resolver: None,
        acceptance: Some(policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    }
}
#[test]
fn explicit_cohort_lightweight_replay_and_aliases_open_no_matrices() {
    let dir = xc_core::test_support::TestDir::new("explicit-cohort-light");
    let store = FilesystemCacheStore::new("local", dir.to_path_buf(), true, CacheVisibility::Local);
    let a = fixture(&store, "a", 1, "0", 8, vec!["0", "1", "0"]);
    let b = fixture(&store, "b", 2, "0.5", 8, vec!["0", "0", "-2", "0", "0"]);
    let mut manifests = a.clone();
    manifests.extend(b.clone());
    let p = plan(&a, &b, false);
    let resources = xc_core::RunResources::new(64 << 20).unwrap();
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(NoMatrixReads(store)),
    }]);
    let policy = policy();
    let cache = context(&resolver, &policy);
    let first = analyze_frozen_cohort(&p, &manifests, &cache, &resources, &dir).unwrap();
    assert!(
        first.measurements.execution_complete,
        "{:?}",
        first.operations.failures
    );
    assert_eq!(first.operations.dense_matrix_payload_opens, 0);
    assert_eq!(
        first.measurements.pairs[0]["eigenvalue_change"]["absolute"],
        "1/2"
    );
    assert_eq!(
        first.measurements.pairs[0]["eigenvalue_change"]["relative_denominator_zero"],
        true
    );
    assert_eq!(
        first.measurements.pairs[0]["state"]["normalized_absolute_overlap"]["lower"],
        "1"
    );
    assert_eq!(
        first.measurements.pairs[0]["state"]["sign_aligned_distance_squared"]["upper"],
        "0"
    );
    assert!(first.artifact.is_some());
    let mut renamed = p.clone();
    renamed.members[0].id = "different-local-label".into();
    renamed.pairs[0].left = renamed.members[0].id.clone();
    manifests.reverse();
    let replay = analyze_frozen_cohort(&renamed, &manifests, &cache, &resources, &dir).unwrap();
    assert_eq!(replay.cache_status, "reused");
    assert_eq!(replay.operations.decoded_payload_bytes_opened, 0);
    assert_eq!(
        serde_json::to_vec(&first.measurements).unwrap(),
        serde_json::to_vec(&replay.measurements).unwrap()
    );
}
#[test]
fn explicit_cohort_optional_residual_is_directed_and_source_bound() {
    let dir = xc_core::test_support::TestDir::new("explicit-cohort-residual");
    let store = FilesystemCacheStore::new("local", dir.to_path_buf(), true, CacheVisibility::Local);
    let a = fixture(&store, "a", 1, "1", 8, vec!["0", "3", "0"]);
    let b = fixture(&store, "b", 2, "2", 8, vec!["0", "0", "1", "0", "0"]);
    let mut manifests = a.clone();
    manifests.extend(b.clone());
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(store),
    }]);
    let policy = policy();
    let cache = context(&resolver, &policy);
    let r = analyze_frozen_cohort(
        &plan(&a, &b, true),
        &manifests,
        &cache,
        &xc_core::RunResources::new(64 << 20).unwrap(),
        &dir,
    )
    .unwrap();
    assert!(
        r.measurements.execution_complete,
        "{:?}",
        r.operations.failures
    );
    assert_eq!(r.operations.dense_matrix_payload_opens, 1);
    let low = &r.measurements.pairs[0]["embedded_residual"]["low_mode_residual_squared"];
    assert!(rug::Rational::from_str_radix(low["lower"].as_str().unwrap(), 10).unwrap() <= 1);
    assert!(rug::Rational::from_str_radix(low["upper"].as_str().unwrap(), 10).unwrap() >= 1);
    assert_eq!(
        r.measurements.pairs[0]["embedded_residual"]["high_mode_forcing_squared"]["upper"],
        "0"
    );
}
#[test]
fn explicit_cohort_invalid_or_missing_sources_are_not_replaced_or_persisted() {
    let dir = xc_core::test_support::TestDir::new("explicit-cohort-missing");
    let store = FilesystemCacheStore::new("local", dir.to_path_buf(), true, CacheVisibility::Local);
    let a = fixture(&store, "a", 1, "1", 8, vec!["0", "1", "0"]);
    let b = fixture(&store, "b", 2, "2", 16, vec!["0", "0", "1", "0", "0"]);
    let p = plan(&a, &b, false);
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(store),
    }]);
    let policy = policy();
    let cache = context(&resolver, &policy);
    let r = analyze_frozen_cohort(
        &p,
        &a,
        &cache,
        &xc_core::RunResources::new(64 << 20).unwrap(),
        &dir,
    )
    .unwrap();
    assert!(!r.measurements.execution_complete);
    assert!(r.artifact.is_none());
    assert_eq!(r.operations.decoded_payload_bytes_opened, 0);
    let mut duplicate = p.clone();
    duplicate.pairs.push(duplicate.pairs[0].clone());
    assert!(duplicate.validate().is_err());
    duplicate = p.clone();
    duplicate.members[1].id = duplicate.members[0].id.clone();
    assert!(duplicate.validate().is_err());
    let mut wrong = a.clone();
    wrong[0].tags.insert(
        SEMANTIC_KEY_MANIFEST_TAG.into(),
        serde_json::to_string(&semantic(
            "ccm_tau_matrix",
            json!({"resolved_quadrature_orders":[999]}),
        ))
        .unwrap(),
    );
    assert!(SourceIdentity::from_manifest(&wrong[0]).is_err());
    let mut both = a.clone();
    both.extend(b);
    assert!(freeze_member("ambiguous", &both).is_err());
}

#[test]
fn explicit_cohort_root_coverage_keeps_missing_ordinals_and_acquisition() {
    let dir = xc_core::test_support::TestDir::new("explicit-cohort-roots");
    let store = FilesystemCacheStore::new("local", dir.to_path_buf(), true, CacheVisibility::Local);
    let mut a = fixture(&store, "a", 1, "1", 8, vec!["0", "1", "0"]);
    let mut b = fixture(&store, "b", 2, "2", 8, vec!["0", "0", "1", "0", "0"]);
    roots(&store, &mut a, 1, vec![Some("1"), Some("2")], true);
    roots(&store, &mut b, 2, vec![Some("4"), None], false);
    let mut p = plan(&a, &b, false);
    p.pairs[0].root_ordinals = vec![1, 2, 3];
    let mut manifests = a.clone();
    manifests.extend(b);
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(store),
    }]);
    let policy = policy();
    let cache = context(&resolver, &policy);
    let r = analyze_frozen_cohort(
        &p,
        &manifests,
        &cache,
        &xc_core::RunResources::new(64 << 20).unwrap(),
        &dir,
    )
    .unwrap();
    assert!(
        r.measurements.execution_complete,
        "{:?}",
        r.operations.failures
    );
    assert!(!r.measurements.requested_coverage_complete);
    assert_eq!(r.operations.dense_matrix_payload_opens, 0);
    let rows = r.measurements.pairs[0]["roots"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["status"], "unresolved");
    assert_eq!(rows[1]["change"]["signed"], "2");
    assert_eq!(rows[2]["right_status"], "failed");
    assert_eq!(
        r.measurements.members[0]["root_acquisition"]["reference_seeds_used"],
        true
    );
    assert_eq!(
        r.measurements.members[1]["root_acquisition"]["reference_seeds_used"],
        false
    );
    p.pairs[0].root_ordinals = vec![0];
    assert!(p.validate().is_err());
}

#[test]
fn explicit_cohort_authentic_seeded_and_independent_producer_policy_layouts() {
    use rug::Float;
    use xc_spectral::ccm::{
        hp::{capture_run::RetainedCcmRun, HighPrecConfig, IndependentRootDiscoveryOptions},
        window::ZeroTarget,
        CcmParams,
    };
    let dir = xc_core::test_support::TestDir::new("explicit-real-root-policy");
    let policy = policy();
    let resources = xc_core::RunResources::new(256 << 20).unwrap();
    let _scope = resources.enter();
    let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
    for adaptive in [false, true] {
        for seeded in [false, true] {
            let mut inventories = Vec::new();
            {
                let resolver = CacheResolver::new(vec![CacheLayer {
                    precedence: 0,
                    store: Box::new(FilesystemCacheStore::new(
                        "local",
                        dir.to_path_buf(),
                        true,
                        CacheVisibility::Local,
                    )),
                }]);
                let cache = context(&resolver, &policy);
                for n in [4, 6] {
                    let mut cfg = HighPrecConfig {
                        precision_bits: 192,
                        n_eigenvalues: 1,
                        ..HighPrecConfig::for_decimal_digits(40)
                    };
                    if adaptive {
                        cfg = cfg.with_adaptive_root_precision();
                    }
                    let params = CcmParams::from_lambda_sq_integer(13, n);
                    let run = if seeded {
                        let seeds = xc_zeta::zeros::bundled_first_n_strings(1)
                            .unwrap()
                            .iter()
                            .map(|s| Float::with_val(cfg.precision_bits, Float::parse(s).unwrap()))
                            .collect::<Vec<_>>();
                        RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap()
                    } else {
                        RetainedCcmRun::independent(
                            &params,
                            &cfg,
                            &ZeroTarget::FirstK { count: 1 },
                            IndependentRootDiscoveryOptions {
                                allow_incomplete: true,
                                ..IndependentRootDiscoveryOptions::default()
                            },
                            &cache,
                        )
                        .unwrap()
                    };
                    inventories.push(run.primary_sources());
                }
            }
            let mut frozen = plan(&inventories[0], &inventories[1], false);
            frozen.pairs[0].root_ordinals = vec![1];
            let manifests = inventories.into_iter().flatten().collect::<Vec<_>>();
            let store =
                FilesystemCacheStore::new("local", dir.to_path_buf(), true, CacheVisibility::Local);
            let root_manifest = manifests
                .iter()
                .find(|m| {
                    m.key.kind
                        == if seeded {
                            "ccm_root_refinement"
                        } else {
                            "ccm_root_discovery_window"
                        }
                })
                .unwrap();
            let root_identity = SourceIdentity::from_manifest(root_manifest).unwrap();
            let root_member_index = frozen
                .members
                .iter()
                .position(|m| m.roots.as_ref() == Some(&root_identity))
                .unwrap();
            let raw: Value =
                serde_json::from_slice(&store.read_payload(root_manifest).unwrap()).unwrap();
            let semantic: SemanticKeyEnvelope =
                serde_json::from_str(&root_manifest.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
            assert_eq!(raw["schema_version"], if adaptive { 5 } else { 6 });
            if adaptive {
                assert_eq!(raw["root_precision_policy"], "adaptive");
                assert_eq!(
                    semantic.resolved_mathematical_parameters["root_precision_policy"],
                    "adaptive_v1"
                );
            } else {
                assert!(raw.get("root_precision_policy").is_none());
                assert!(semantic
                    .resolved_mathematical_parameters
                    .get("root_precision_policy")
                    .is_none());
            }
            let resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(NoMatrixReads(store)),
            }]);
            let cache = context(&resolver, &policy);
            let first =
                analyze_frozen_cohort(&frozen, &manifests, &cache, &resources, &dir).unwrap();
            assert!(
                first.measurements.execution_complete,
                "adaptive={adaptive} seeded={seeded}: {:?}",
                first.operations.failures
            );
            if seeded {
                assert!(first.measurements.requested_coverage_complete);
            } else {
                // The actual N=4 independent finite window has no positive
                // root. Preserve that absence rather than substituting seeds.
                assert!(!first.measurements.requested_coverage_complete);
                assert_eq!(
                    first.measurements.pairs[0]["roots"][0]["status"],
                    "unresolved"
                );
            }
            assert_eq!(first.operations.dense_matrix_payload_opens, 0);
            assert_eq!(
                first.measurements.members[root_member_index]["root_acquisition"]
                    ["reference_seeds_used"],
                seeded
            );
            let replay =
                analyze_frozen_cohort(&frozen, &manifests, &cache, &resources, &dir).unwrap();
            assert_eq!(replay.cache_status, "reused");
            assert_eq!(replay.operations.decoded_payload_bytes_opened, 0);
            // Authenticate deliberately inconsistent payload bytes under their own
            // new digest: the consumer must still reject the policy mismatch.
            let mut wrong = raw.clone();
            if adaptive {
                wrong["root_precision_policy"] = json!("adaptive_v1");
            } else {
                wrong["target_precision_bits"] = json!(1);
            }
            let store =
                FilesystemCacheStore::new("local", dir.to_path_buf(), true, CacheVisibility::Local);
            let bad = store
                .put(
                    &ArtifactDraft {
                        schema_version: 1,
                        key: root_manifest.key.clone(),
                        producer_toolkit_version: root_manifest.producer_toolkit_version.clone(),
                        minimum_reader_version: root_manifest.minimum_reader_version.clone(),
                        maximum_reader_version: None,
                        quality: root_manifest.quality,
                        visibility: root_manifest.visibility,
                        immutable: true,
                        dependencies: root_manifest.dependencies.clone(),
                        tags: root_manifest.tags.clone(),
                        provenance_digest: None,
                    },
                    &serde_json::to_vec(&wrong).unwrap(),
                )
                .unwrap();
            let mut bad_plan = frozen.clone();
            bad_plan.members[root_member_index].roots =
                Some(SourceIdentity::from_manifest(&bad).unwrap());
            let mut bad_inventory = manifests.clone();
            bad_inventory.push(bad);
            let reject_resolver = CacheResolver::new(vec![CacheLayer {
                precedence: 0,
                store: Box::new(NoMatrixReads(store)),
            }]);
            let reject_cache = context(&reject_resolver, &policy);
            let rejected =
                analyze_frozen_cohort(&bad_plan, &bad_inventory, &reject_cache, &resources, &dir)
                    .unwrap();
            assert!(!rejected.measurements.execution_complete);
            assert!(rejected.artifact.is_none());
            assert!(
                rejected
                    .operations
                    .failures
                    .iter()
                    .any(|reason| reason.contains(if adaptive {
                        "adaptive root payload policy mismatch"
                    } else {
                        "unbound adaptive target_precision_bits"
                    })),
                "{:?}",
                rejected.operations.failures
            );
        }
    }
}
