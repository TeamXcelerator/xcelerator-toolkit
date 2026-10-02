use std::collections::BTreeMap;
use xc_cache::*;
use xc_core::{CancellationToken, ResourcePolicy};

fn record(kind: &str) -> ProducedArtifactRecord {
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: "fresh-audit-dependency-quality-v1".into(),
        resolved_mathematical_parameters: serde_json::json!({"precision_bits": 53}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let payload = b"{\"fixture\":1}".to_vec();
    let digest = ContentDigest::sha256(&payload);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey {
            kind: kind.into(),
            logical_key: kind.into(),
            parameters_digest: semantic_key.digest().unwrap(),
        },
        content_digest: digest.clone(),
        size_bytes: payload.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: payload.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_reader_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    ProducedArtifactRecord {
        operation: "audit.fixture".into(),
        semantic_key,
        logical_key: kind.into(),
        manifest,
        achieved_assurance: ArtifactAssuranceState::Computed,
        assurance_evidence_digests: vec![],
        payload,
    }
}

#[test]
fn canonical_staging_must_enforce_each_declared_dependency_quality() {
    let scratch = xc_core::test_support::TestDir::new("fresh-staging-quality");
    let root = scratch.join("root");
    let parent = record("ccm_tau_matrix");
    let mut child = record("ccm_factorization");
    child.manifest.dependencies.push(DependencyRef {
        key: parent.manifest.key.clone(),
        content_digest: parent.manifest.content_digest.clone(),
        required_quality: CacheQuality::Certified,
    });
    parent.manifest.validate().unwrap();
    child.manifest.validate().unwrap();
    let transport = TransportPolicy::default();
    let resources = ResourcePolicy::default();
    let cancellation = CancellationToken::new();
    let draft = stage_produced_artifact(
        &parent,
        &root.join("direct"),
        &transport,
        &resources,
        &cancellation,
    )
    .unwrap();
    assert_eq!(draft.source_quality, Some(CacheQuality::Validated));
    let direct = stage_produced_artifact_with_dependencies(
        &child,
        &[draft],
        &root.join("direct"),
        &transport,
        &resources,
        &cancellation,
    );
    let sink =
        CanonicalStagingProductionSink::new(root.join("sink"), transport, resources, cancellation)
            .unwrap();
    sink.record(parent).unwrap();
    let observed = sink.record(child);
    let accepted = (direct.is_ok(), observed.is_ok());
    std::fs::remove_dir_all(&root).unwrap();
    eprintln!(
        "Validated parent supplied for Certified dependency; direct accepted={}, sink accepted={}",
        accepted.0, accepted.1
    );
    assert_eq!(
        accepted,
        (false, false),
        "canonical staging erased a stronger required_quality without enforcing it"
    );
}

#[test]
fn qualified_dependency_is_selected_and_quality_upgrades_are_retained() {
    let scratch = xc_core::test_support::TestDir::new("fresh-staging-quality-se");
    let root = scratch.join("root");
    let low = record("ccm_tau_matrix");
    let mut high = low.clone();
    high.manifest.quality = CacheQuality::Certified;
    let mut child = record("ccm_factorization");
    child.manifest.dependencies.push(DependencyRef {
        key: low.manifest.key.clone(),
        content_digest: low.manifest.content_digest.clone(),
        required_quality: CacheQuality::Certified,
    });
    let transport = TransportPolicy::default();
    let resources = ResourcePolicy::default();
    let cancellation = CancellationToken::new();
    let low_draft = stage_produced_artifact(
        &low,
        &root.join("direct"),
        &transport,
        &resources,
        &cancellation,
    )
    .unwrap();
    let high_draft = stage_produced_artifact(
        &high,
        &root.join("direct"),
        &transport,
        &resources,
        &cancellation,
    )
    .unwrap();
    let mut unknown = low_draft.clone();
    unknown.source_quality = None;
    assert!(stage_produced_artifact_with_dependencies(
        &child,
        &[unknown],
        &root.join("unknown"),
        &transport,
        &resources,
        &cancellation
    )
    .is_err());
    stage_produced_artifact_with_dependencies(
        &child,
        &[low_draft, high_draft],
        &root.join("direct"),
        &transport,
        &resources,
        &cancellation,
    )
    .unwrap();
    let sink =
        CanonicalStagingProductionSink::new(root.join("sink"), transport, resources, cancellation)
            .unwrap();
    sink.record(low).unwrap();
    let mut lower_requirement = child.clone();
    lower_requirement.manifest.dependencies[0].required_quality = CacheQuality::Validated;
    sink.record(lower_requirement).unwrap();
    // The same source key/content already exists: dedup must still enforce
    // the stronger current requirement instead of returning the older draft.
    assert!(sink.record(child.clone()).is_err());
    sink.record(high).unwrap();
    sink.record(child).unwrap();
    assert!(sink
        .drafts()
        .unwrap()
        .iter()
        .any(|draft| draft.source_quality == Some(CacheQuality::Certified)));
    drop(sink);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn incomplete_staging_without_commit_marker_can_be_reopened_and_rebuilt() {
    let scratch = xc_core::test_support::TestDir::new("fresh-staging-durability");
    let root = scratch.join("root");
    let record = record("ccm_tau_matrix");
    let sink = CanonicalStagingProductionSink::new(
        root.clone(),
        TransportPolicy::default(),
        ResourcePolicy::default(),
        CancellationToken::new(),
    )
    .unwrap();
    sink.record(record.clone()).unwrap();
    let drafts = sink.drafts().unwrap();
    let directory = drafts[0].staged_parts_root.parent().unwrap().to_owned();
    drop(sink);
    // Model interruption before the atomic marker rename: completed parts plus
    // an incomplete private sibling, but no public commit marker.
    std::fs::remove_file(directory.join("draft.json")).unwrap();
    std::fs::write(
        directory.join(".replace-interrupted.tmp"),
        b"{\"schema_version\":",
    )
    .unwrap();
    let reopened = CanonicalStagingProductionSink::new(
        root.clone(),
        TransportPolicy::default(),
        ResourcePolicy::default(),
        CancellationToken::new(),
    )
    .unwrap();
    assert!(reopened.drafts().unwrap().is_empty());
    reopened.record(record).unwrap();
    assert_eq!(reopened.drafts().unwrap().len(), 1);
    assert!(!directory.join(".replace-interrupted.tmp").exists());
    let _: CanonicalProductionDraft =
        serde_json::from_slice(&std::fs::read(directory.join("draft.json")).unwrap()).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn dependency_quality_distinguishes_publication_and_rejected_dispositions() {
    let scratch = xc_core::test_support::TestDir::new("fresh-staging-dispositio");
    let root = scratch.join("root");
    for (index, (source, required, accepted)) in [
        (CacheQuality::Validated, CacheQuality::Published, false),
        (CacheQuality::Published, CacheQuality::Certified, false),
        (CacheQuality::Quarantined, CacheQuality::Published, false),
        (CacheQuality::Deprecated, CacheQuality::Published, false),
        (CacheQuality::Published, CacheQuality::Published, true),
        (CacheQuality::Certified, CacheQuality::Validated, true),
    ]
    .into_iter()
    .enumerate()
    {
        let mut parent = record("ccm_tau_matrix");
        parent.manifest.quality = source;
        let mut child = record("ccm_factorization");
        child.manifest.dependencies.push(DependencyRef {
            key: parent.manifest.key.clone(),
            content_digest: parent.manifest.content_digest.clone(),
            required_quality: required,
        });
        let directory = root.join(index.to_string());
        let sink = CanonicalStagingProductionSink::new(
            directory,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        sink.record(parent).unwrap();
        assert_eq!(
            sink.record(child).is_ok(),
            accepted,
            "{source:?} for {required:?}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn typed_execution_records_resolved_dependency_quality_upgrade_in_both_storage_routes() {
    let scratch = xc_core::test_support::TestDir::new("fresh-closure-upgrade");
    let root = scratch.join("root");
    for encoded in [false, true] {
        let directory = root.join(if encoded { "encoded" } else { "decoded" });
        let low = record("ccm_tau_matrix");
        let child = record("ccm_factorization");
        let mut tags = low.manifest.tags.clone();
        tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.into(),
            serde_json::to_string(&low.semantic_key).unwrap(),
        );
        let high = ArtifactDraft {
            schema_version: 1,
            key: low.manifest.key.clone(),
            producer_toolkit_version: low.manifest.producer_toolkit_version.clone(),
            minimum_reader_version: low.manifest.minimum_reader_version.clone(),
            maximum_reader_version: None,
            quality: CacheQuality::Certified,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags,
            provenance_digest: None,
        };
        let store: Box<dyn CacheStore> = if encoded {
            Box::new(ZipJsonFilesystemCacheStore::new(
                "local",
                directory.join("cache"),
                true,
                CacheVisibility::Local,
            ))
        } else {
            Box::new(FilesystemCacheStore::new(
                "local",
                directory.join("cache"),
                true,
                CacheVisibility::Local,
            ))
        };
        let parent = store.put(&high, &low.payload).unwrap();
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store,
        }]);
        let policy = CachePolicy {
            current_toolkit_version: low.manifest.producer_toolkit_version.clone(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let sink = CanonicalStagingProductionSink::new(
            directory.join("staging"),
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        sink.record(low).unwrap();
        let request = ArtifactExecutionCacheRequest {
            operation: "audit.closure.upgrade",
            semantic_key: &child.semantic_key,
            logical_key: &child.logical_key,
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["local".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: child.manifest.producer_toolkit_version.clone(),
            minimum_reader_version: child.manifest.minimum_reader_version.clone(),
            maximum_reader_version: None,
            tags: BTreeMap::new(),
            provenance_digest: None,
            production_sink: Some(&sink),
        };
        let outcome = resolve_or_compute_json_artifact_with_dependencies(
            &request,
            || {
                Ok((
                    serde_json::json!({"fixture":1}),
                    vec![DependencyRef {
                        key: parent.key.clone(),
                        content_digest: parent.content_digest.clone(),
                        required_quality: CacheQuality::Certified,
                    }],
                ))
            },
            |_| Ok(()),
        );
        assert!(
            outcome.is_ok(),
            "encoded={encoded}; resolved stronger dependency was not staged: {:?}",
            outcome.err()
        );
        let drafts = sink.drafts().unwrap();
        assert!(drafts
            .iter()
            .any(|draft| draft.source_artifact_key == parent.key
                && draft.source_quality == Some(CacheQuality::Certified)));
        assert!(drafts
            .iter()
            .any(|draft| draft.source_artifact_key == child.manifest.key));
    }
    std::fs::remove_dir_all(root).unwrap();
}
