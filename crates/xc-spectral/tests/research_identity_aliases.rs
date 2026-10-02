#![cfg(feature = "hp")]
//! The same parent bytes reach a research child under different store forms:
//! as computed (producer tags, real logical key), as resolved from a remote
//! shard by identity (`closure/...` key, no local dependency list) and as a
//! locally adopted copy (`xc.cached_from`). Each form must reach the same child.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::extended_research::{capture_extended, ExtensionOptions};
use xc_spectral::ccm::prefix::*;
use xc_spectral::ccm::state_geometry::RetainedState;
#[path = "common/published_sources.rs"]
mod published_sources;

fn source(kind: &str, value: Value, label: &str) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: "alias-fixture-v1".into(),
        resolved_mathematical_parameters: json!({"p":128,"label":label}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey {
            kind: kind.into(),
            logical_key: format!("actual/{kind}"),
            parameters_digest: semantic.digest().unwrap(),
        },
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
        tags: BTreeMap::from([
            ("xc_storage_encoding".into(), "zip-json-entry-v1".into()),
            ("domain".into(), "ccm".into()),
            (
                SEMANTIC_KEY_MANIFEST_TAG.into(),
                serde_json::to_string(&semantic).unwrap(),
            ),
        ]),
        provenance_digest: None,
    };
    (manifest, bytes)
}

/// The remote adapter form: synthetic key, private lane, no dependency list,
/// a different tag map.
fn remote(m: &ArtifactManifest) -> ArtifactManifest {
    let mut a = m.clone();
    a.key.logical_key = format!("closure/{}", &m.key.parameters_digest.0[..16]);
    a.visibility = CacheVisibility::Private;
    a.dependencies.clear();
    a.created_unix_seconds = 100;
    a.tags = BTreeMap::from([("xc.semantic_key.v1".into(), "{\"remote\":true}".into())]);
    a
}

/// A locally adopted copy of a remote artifact.
fn adopted(m: &ArtifactManifest) -> ArtifactManifest {
    let mut a = remote(m);
    a.tags
        .insert("xc.cached_from".into(), "github-private".into());
    a
}

struct Cache {
    _dir: xc_core::test_support::TestDir,
    resolver: CacheResolver,
    policy: CachePolicy,
}
impl Cache {
    fn new(name: &str) -> Self {
        let dir = xc_core::test_support::TestDir::new(name);
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "workstation",
                dir.to_path_buf(),
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
        Self {
            _dir: dir,
            resolver,
            policy,
        }
    }
    fn context(&self, mode: ArtifactExecutionCacheMode) -> ArtifactCacheContext<'_> {
        ArtifactCacheContext {
            resolver: Some(&self.resolver),
            reference_resolver: None,
            acceptance: Some(&self.policy),
            ordered_overlays: vec!["workstation".into()],
            mode,
            write_on_miss: mode != ArtifactExecutionCacheMode::RequireReuse,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        }
    }
}
const PREFER: ArtifactExecutionCacheMode = ArtifactExecutionCacheMode::PreferReuse;
const REQUIRE: ArtifactExecutionCacheMode = ArtifactExecutionCacheMode::RequireReuse;

fn child<T>(result: &ArtifactExecutionCacheResult<T>) -> &ArtifactManifest {
    result
        .produced_manifest
        .as_ref()
        .or(result.reused_manifest.as_ref())
        .unwrap()
}

fn state_source() -> (ArtifactManifest, Vec<u8>) {
    source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"13","n_modes":1,"precision_bits":128,
            "eigenvalue":"1","eigenvector":["0","1","0"],"force_even":true,"parity_policy":"even_sector"}),
        "stable",
    )
}
fn matrix_source() -> (ArtifactManifest, Vec<u8>) {
    source(
        "ccm_even_sector_matrix",
        json!({"schema_version":1,"lambda_squared":"13","n_modes":1,"precision_bits":128,
            "dimension":2,"entries":["1","0","0","2"]}),
        "stable",
    )
}
fn prefix_options() -> PrefixAnalysisOptions {
    PrefixAnalysisOptions {
        working_precision_bits: 128,
        pivot_margin_bits: 32,
        checkpoint_dimensions: vec![2],
        export_significant_digits: vec![40, 48],
        export_relative_tolerance: "1e-30".into(),
        diagnostics: PrefixDiagnosticPolicy::default(),
    }
}

#[test]
fn extended_diagnostics_reach_one_child_from_producer_remote_and_adopted_states() {
    let cache = Cache::new("alias-extended");
    let (sm, sb) = state_source();
    let state = |m: &ArtifactManifest| {
        RetainedState::from_payload(m, &sb, std::slice::from_ref(&m.content_digest)).unwrap()
    };
    let producer = state(&sm);
    let options = ExtensionOptions::for_source(&producer);
    let first = capture_extended(
        "compactness",
        &producer,
        None,
        None,
        None,
        &options,
        &[],
        &cache.context(PREFER),
    )
    .unwrap();
    assert!(first.produced_manifest.is_some());
    for form in [remote(&sm), adopted(&sm)] {
        let reused = capture_extended(
            "compactness",
            &state(&form),
            None,
            None,
            None,
            &options,
            &[],
            &cache.context(REQUIRE),
        )
        .unwrap();
        assert!(reused.reused_manifest.is_some());
        assert_eq!(child(&reused).key, child(&first).key);
        assert_eq!(child(&reused).content_digest, child(&first).content_digest);
    }
    // A genuinely different state identity is a different child.
    let (other, other_bytes) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"13","n_modes":1,"precision_bits":128,
            "eigenvalue":"1","eigenvector":["0","1","0"],"force_even":true,"parity_policy":"even_sector"}),
        "other-semantics",
    );
    let other = RetainedState::from_payload(
        &other,
        &other_bytes,
        std::slice::from_ref(&other.content_digest),
    )
    .unwrap();
    assert!(capture_extended(
        "compactness",
        &other,
        None,
        None,
        None,
        &options,
        &[],
        &cache.context(REQUIRE)
    )
    .is_err());
}

#[test]
fn prefix_and_reduction_children_accept_equivalent_parent_aliases_in_both_directions() {
    let (mm, mb) = matrix_source();
    let (sm, sb) = state_source();
    let matrix = |m: &ArtifactManifest| {
        RetainedEvenMatrix::from_payload(m, &mb, std::slice::from_ref(&m.content_digest)).unwrap()
    };
    let pair = |m: &ArtifactManifest| {
        RetainedEvenEigenpair::from_payload(m, &sb, std::slice::from_ref(&m.content_digest))
            .unwrap()
    };
    let forms = |m: &ArtifactManifest, s: &ArtifactManifest, alias: bool| {
        if alias {
            (remote(m), remote(s))
        } else {
            (m.clone(), s.clone())
        }
    };
    for computed_as_alias in [false, true] {
        let cache = Cache::new(if computed_as_alias {
            "alias-prefix-from-remote"
        } else {
            "alias-prefix-from-producer"
        });
        let (m0, s0) = forms(&mm, &sm, computed_as_alias);
        let prefix = analyze_retained_prefixes_via_cache(
            &matrix(&m0),
            &prefix_options(),
            &[pair(&s0)],
            &cache.context(PREFER),
        )
        .unwrap();
        let reduction = check_retained_reduction_via_cache(
            &matrix(&m0),
            128,
            2,
            "1e-25",
            &cache.context(PREFER),
        )
        .unwrap();
        // The child manifest keeps the parent keys it was computed with.
        assert!(child(&prefix)
            .dependencies
            .iter()
            .any(|d| d.key.logical_key == m0.key.logical_key));
        let (m1, s1) = forms(&mm, &sm, !computed_as_alias);
        for mode in [PREFER, REQUIRE] {
            let hit = analyze_retained_prefixes_via_cache(
                &matrix(&m1),
                &prefix_options(),
                &[pair(&s1)],
                &cache.context(mode),
            )
            .unwrap();
            assert!(hit.reused_manifest.is_some());
            assert_eq!(child(&hit).key, child(&prefix).key);
            assert_eq!(hit.value, prefix.value);
            let hit = check_retained_reduction_via_cache(
                &matrix(&m1),
                128,
                2,
                "1e-25",
                &cache.context(mode),
            )
            .unwrap();
            assert!(hit.reused_manifest.is_some());
            assert_eq!(child(&hit).key, child(&reduction).key);
        }
        // A missing parent is a different child, not a match.
        assert!(analyze_retained_prefixes_via_cache(
            &matrix(&m1),
            &prefix_options(),
            &[],
            &cache.context(REQUIRE)
        )
        .is_err());
        // Changed parent content under the same semantic key is a different child.
        let (changed, changed_bytes) = source(
            "ccm_even_sector_matrix",
            json!({"schema_version":1,"lambda_squared":"13","n_modes":1,"precision_bits":128,
                "dimension":2,"entries":["1","0","0","3"]}),
            "stable",
        );
        assert_eq!(changed.key.parameters_digest, mm.key.parameters_digest);
        let changed = RetainedEvenMatrix::from_payload(
            &changed,
            &changed_bytes,
            std::slice::from_ref(&changed.content_digest),
        )
        .unwrap();
        assert!(check_retained_reduction_via_cache(
            &changed,
            128,
            2,
            "1e-25",
            &cache.context(REQUIRE)
        )
        .is_err());
    }
}

/// Store a published (canonical) child, as a shard adapter presents it.
fn publish(
    store: &FilesystemCacheStore,
    manifest: &ArtifactManifest,
    payload: &[u8],
    parents: &[&ArtifactManifest],
) {
    let p = published_sources::published(manifest.clone(), parents);
    store
        .put(
            &ArtifactDraft {
                schema_version: p.schema_version,
                key: p.key,
                producer_toolkit_version: p.producer_toolkit_version,
                minimum_reader_version: p.minimum_reader_version,
                maximum_reader_version: p.maximum_reader_version,
                quality: p.quality,
                visibility: p.visibility,
                immutable: p.immutable,
                dependencies: p.dependencies,
                tags: p.tags,
                provenance_digest: p.provenance_digest,
            },
            payload,
        )
        .unwrap();
}

#[test]
fn published_children_reuse_proven_parents_and_recompute_for_locally_computed_ones() {
    let (mm, mb) = matrix_source();
    let (sm, sb) = state_source();
    let matrix = |m: &ArtifactManifest| {
        RetainedEvenMatrix::from_payload(m, &mb, std::slice::from_ref(&m.content_digest)).unwrap()
    };
    let pair = |m: &ArtifactManifest| {
        RetainedEvenEigenpair::from_payload(m, &sb, std::slice::from_ref(&m.content_digest))
            .unwrap()
    };
    // Published parents keep the local semantic key, so the child identity is shared.
    let pm = published_sources::published(mm.clone(), &[]);
    let ps = published_sources::published(sm.clone(), &[]);
    assert_eq!(pm.key.parameters_digest, mm.key.parameters_digest);
    let producer = Cache::new("mixed-producer");
    let prefix = analyze_retained_prefixes_via_cache(
        &matrix(&pm),
        &prefix_options(),
        &[pair(&ps)],
        &producer.context(PREFER),
    )
    .unwrap();
    let reduction = check_retained_reduction_via_cache(
        &matrix(&pm),
        128,
        2,
        "1e-25",
        &producer.context(PREFER),
    )
    .unwrap();
    let remote = Cache::new("mixed-remote");
    let store = FilesystemCacheStore::new(
        "workstation",
        remote._dir.to_path_buf(),
        true,
        CacheVisibility::Local,
    );
    publish(
        &store,
        child(&prefix),
        &serde_json::to_vec(&prefix.value).unwrap(),
        &[&pm, &ps],
    );
    publish(
        &store,
        child(&reduction),
        &serde_json::to_vec(&reduction.value).unwrap(),
        &[&pm],
    );
    let run_prefix = |m: &ArtifactManifest, s: &ArtifactManifest, mode| {
        analyze_retained_prefixes_via_cache(
            &matrix(m),
            &prefix_options(),
            &[pair(s)],
            &remote.context(mode),
        )
    };
    let run_reduction = |m: &ArtifactManifest, mode| {
        check_retained_reduction_via_cache(&matrix(m), 128, 2, "1e-25", &remote.context(mode))
    };
    // Proven canonical parents, under their published or another label, reuse.
    let mut relabelled = pm.clone();
    relabelled.key.logical_key = format!("closure/{}", &pm.key.parameters_digest.0[..16]);
    for m in [&pm, &relabelled] {
        let hit = run_prefix(m, &ps, REQUIRE).unwrap();
        assert!(hit.reused_manifest.is_some());
        assert_eq!(hit.value, prefix.value);
        assert!(run_reduction(m, REQUIRE).unwrap().reused_manifest.is_some());
    }
    // Conflicting published ancestry for the same bytes is rejected, not recomputed.
    let conflicting = published_sources::published(mm.clone(), &[&pm]);
    for mode in [PREFER, REQUIRE] {
        assert!(run_prefix(&conflicting, &ps, mode).is_err());
        assert!(run_reduction(&conflicting, mode).is_err());
    }
    // Parents computed on this workstation cannot prove the published child's
    // canonical ancestry: required reuse misses, ordinary reuse recomputes.
    assert!(run_prefix(&mm, &sm, REQUIRE).is_err());
    assert!(run_reduction(&mm, REQUIRE).is_err());
    let fresh = run_prefix(&mm, &sm, PREFER).unwrap();
    assert!(fresh.produced_manifest.is_some());
    assert_eq!(fresh.value, prefix.value);
    let fresh = run_reduction(&mm, PREFER).unwrap();
    assert!(fresh.produced_manifest.is_some());
    assert_eq!(
        serde_json::to_value(&fresh.value).unwrap(),
        serde_json::to_value(&reduction.value).unwrap()
    );
}

fn store_source(store: &FilesystemCacheStore, m: &ArtifactManifest, bytes: &[u8]) {
    store
        .put(
            &ArtifactDraft {
                schema_version: m.schema_version,
                key: m.key.clone(),
                producer_toolkit_version: m.producer_toolkit_version.clone(),
                minimum_reader_version: m.minimum_reader_version.clone(),
                maximum_reader_version: m.maximum_reader_version.clone(),
                quality: m.quality,
                visibility: m.visibility,
                immutable: m.immutable,
                dependencies: m.dependencies.clone(),
                tags: m.tags.clone(),
                provenance_digest: m.provenance_digest.clone(),
            },
            bytes,
        )
        .unwrap();
}

/// With publication staging on, an adopted published child whose canonical
/// parents are unavailable must still fall back to recomputation from the
/// current, locally computed parents, and stage that current closure.
fn unprovable_child_recomputes_and_stages(canonical_staging: bool) {
    let (mm, mb) = matrix_source();
    let (mut sm, sb) = state_source();
    // The continuation index reads the eigenpair's semantic fields.
    let mut semantic: SemanticKeyEnvelope =
        serde_json::from_str(&sm.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
    semantic.resolved_mathematical_parameters = json!({"lambda_squared":"13","n_modes":1,
        "precision_bits":128,"force_even":true,"label":"state"});
    sm.key.parameters_digest = semantic.digest().unwrap();
    sm.tags.insert(
        SEMANTIC_KEY_MANIFEST_TAG.into(),
        serde_json::to_string(&semantic).unwrap(),
    );
    let matrix = |m: &ArtifactManifest| {
        RetainedEvenMatrix::from_payload(m, &mb, std::slice::from_ref(&m.content_digest)).unwrap()
    };
    let pair = |m: &ArtifactManifest| {
        RetainedEvenEigenpair::from_payload(m, &sb, std::slice::from_ref(&m.content_digest))
            .unwrap()
    };
    let pm = published_sources::published(mm.clone(), &[]);
    let ps = published_sources::published(sm.clone(), &[]);
    let producer = Cache::new("staging-producer");
    let prefix = analyze_retained_prefixes_via_cache(
        &matrix(&pm),
        &prefix_options(),
        &[pair(&ps)],
        &producer.context(PREFER),
    )
    .unwrap();
    let reduction = check_retained_reduction_via_cache(
        &matrix(&pm),
        128,
        2,
        "1e-25",
        &producer.context(PREFER),
    )
    .unwrap();
    // The consumer holds the adopted published children and locally computed
    // parents, but none of the children's canonical parents.
    let consumer = Cache::new(if canonical_staging {
        "staging-consumer-canonical"
    } else {
        "staging-consumer-directory"
    });
    let store = FilesystemCacheStore::new(
        "workstation",
        consumer._dir.to_path_buf(),
        true,
        CacheVisibility::Local,
    );
    store_source(&store, &mm, &mb);
    store_source(&store, &sm, &sb);
    publish(
        &store,
        child(&prefix),
        &serde_json::to_vec(&prefix.value).unwrap(),
        &[&pm, &ps],
    );
    publish(
        &store,
        child(&reduction),
        &serde_json::to_vec(&reduction.value).unwrap(),
        &[&pm],
    );
    let sink: Box<dyn ArtifactProductionSink> = if canonical_staging {
        Box::new(
            CanonicalStagingProductionSink::new(
                consumer._dir.join("staging"),
                TransportPolicy::default(),
                xc_core::ResourcePolicy::default(),
                xc_core::CancellationToken::new(),
            )
            .unwrap(),
        )
    } else {
        Box::new(DirectoryArtifactProductionSink::new(consumer._dir.join("queue")).unwrap())
    };
    // Required reuse first: while the cache holds only the unprovable published
    // children, it must report the unprovable parents. Ordinary reuse then
    // recomputes, after which a valid local child may serve later requests.
    for mode in [REQUIRE, PREFER] {
        let mut context = consumer.context(mode);
        context.production_sink = Some(sink.as_ref());
        let fresh_prefix = analyze_retained_prefixes_via_cache(
            &matrix(&mm),
            &prefix_options(),
            &[pair(&sm)],
            &context,
        );
        let fresh_reduction =
            check_retained_reduction_via_cache(&matrix(&mm), 128, 2, "1e-25", &context);
        if mode == REQUIRE {
            for error in [
                fresh_prefix.err().unwrap().to_string(),
                fresh_reduction.err().unwrap().to_string(),
            ] {
                assert!(
                    error.contains("cannot be established for these locally computed parents"),
                    "{error}"
                );
            }
            continue;
        }
        let fresh_prefix = fresh_prefix.unwrap();
        let fresh_reduction = fresh_reduction.unwrap();
        assert!(fresh_prefix.produced_manifest.is_some());
        assert!(fresh_reduction.produced_manifest.is_some());
        assert_eq!(fresh_prefix.value, prefix.value);
        assert_eq!(
            serde_json::to_value(&fresh_reduction.value).unwrap(),
            serde_json::to_value(&reduction.value).unwrap()
        );
        // The staged closure is the recomputed children and their current parents.
        for staged in [child(&fresh_prefix), child(&fresh_reduction), &mm, &sm] {
            assert!(
                sink.contains_artifact(&staged.key, &staged.content_digest)
                    .unwrap(),
                "{} was not staged",
                staged.key.kind
            );
        }
    }
}

#[test]
fn unprovable_published_children_recompute_and_stage_with_a_directory_sink() {
    unprovable_child_recomputes_and_stages(false);
}

#[test]
fn unprovable_published_children_recompute_and_stage_with_canonical_staging() {
    unprovable_child_recomputes_and_stages(true);
}
