fn alias_dependency(draft: &CanonicalProductionDraft) -> crate::PayloadDependencyIdentity {
    crate::PayloadDependencyIdentity {
        artifact_family: draft.family.clone(),
        semantic_digest: draft.manifest.semantic_digest.clone(),
        manifest_digest: draft.manifest.digest().unwrap(),
        payload_digest: draft.manifest.payload_digest.clone(),
    }
}

fn alias_dependencies(
    mut draft: CanonicalProductionDraft,
    parents: &[&CanonicalProductionDraft],
) -> CanonicalProductionDraft {
    let mut dependencies = parents
        .iter()
        .map(|p| alias_dependency(p))
        .collect::<Vec<_>>();
    dependencies.sort_by_key(|d| {
        (
            d.artifact_family.clone(),
            d.semantic_digest.clone(),
            d.manifest_digest.clone(),
            d.payload_digest.clone(),
        )
    });
    draft.manifest.canonical_payload.dependencies = dependencies;
    draft.manifest.payload_digest = draft.manifest.canonical_payload.digest().unwrap();
    draft.encoding.canonical_payload_digest = draft.manifest.payload_digest.clone();
    draft.manifest.transport_digests = vec![draft.encoding.digest().unwrap()];
    draft.manifest.validate().unwrap();
    draft
}

#[test]
fn destination_aliases_publish_once_with_closed_dependencies_in_both_lanes() {
    let root = std::env::temp_dir().join(format!("xc-publication-aliases-{}", std::process::id()));
    let neutral = fixture_draft_with_n(&root, 2);
    let mut private = neutral.clone();
    private.manifest = target_manifest(&private, PublicationDestination::Private).unwrap();
    private.source_operation = "cache.dependency.closure".to_owned();
    let mut public = private.clone();
    public.manifest = target_manifest(&public, PublicationDestination::Public).unwrap();

    // The same leaf is reached by a local key, a public source and a private
    // receipt's exact historical dependency. Its children converge only after
    // recursively remapping their different parent manifest identities.
    let left = alias_dependencies(fixture_draft_with_n(&root, 3), &[&neutral]);
    let right = alias_dependencies(left.clone(), &[&private]);
    let top = alias_dependencies(fixture_draft_with_n(&root, 4), &[&left, &right]);
    let drafts = vec![top, right, public, left, private, neutral.clone(), neutral];
    for destination in [
        PublicationDestination::Private,
        PublicationDestination::Public,
    ] {
        let mut expected = None;
        for input in [drafts.clone(), drafts.iter().cloned().rev().collect()] {
            let remapped = remap_destination_drafts_with_existing(&input, destination, |_| {
                panic!("the complete staged closure needs no remote dependency lookup")
            })
            .unwrap();
            assert_eq!(
                remapped.len(),
                3,
                "one destination artifact per canonical identity"
            );
            let identities = remapped.iter().map(alias_dependency).collect::<Vec<_>>();
            for draft in &remapped {
                draft.manifest.validate().unwrap();
                assert_eq!(
                    draft.encoding.canonical_payload_digest,
                    draft.manifest.payload_digest
                );
                assert_eq!(
                    draft.manifest.transport_digests,
                    vec![draft.encoding.digest().unwrap()]
                );
                assert!(draft
                    .manifest
                    .canonical_payload
                    .dependencies
                    .iter()
                    .all(|d| identities.contains(d)));
                assert!(draft.manifest.canonical_payload.dependencies.len() <= 1);
            }
            let leaf = remapped
                .iter()
                .find(|d| d.manifest.canonical_payload.dependencies.is_empty())
                .unwrap();
            assert_ne!(
                leaf.source_operation, "cache.dependency.closure",
                "retain the directly observed source role"
            );
            let tree = remapped
                .iter()
                .map(|d| (d.manifest.digest().unwrap(), d.manifest.clone()))
                .collect::<BTreeMap<_, _>>();
            if let Some(expected) = &expected {
                assert_eq!(&tree, expected);
            }
            expected = Some(tree);

            // Pass the result into the real destination index/batch verifier.
            // A second publication of an already indexed closure is a no-op.
            let refs = remapped.iter().collect::<Vec<_>>();
            let remote = FilesystemMemoryRemote::new(root.clone());
            let repository = repository_url("example-org", "ccm-matrices", destination);
            remote.insert_repository(
                repository.clone(),
                "head".to_owned(),
                published_destination_tree(&refs, destination),
            );
            let selection = select_missing_destination_drafts(
                &remote,
                &repository,
                "head",
                "ccm-matrices",
                destination,
                &refs,
                &CancellationToken::new(),
            )
            .unwrap();
            assert!(selection.pending.is_empty());
            assert_eq!(selection.already_present, 3);
        }
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn destination_aliases_preserve_assurance_requirements_and_evidence() {
    let root = std::env::temp_dir().join(format!(
        "xc-publication-alias-assurance-{}",
        std::process::id()
    ));
    let strong = fixture_draft(&root);
    let mut weak = strong.clone();
    weak.achieved_assurance = ArtifactAssuranceState::Computed;
    weak.required_assurance = None;
    weak.assurance_evidence_digests = vec![ContentDigest::sha256(b"validation-evidence")];
    let expected = strong
        .assurance_evidence_digests
        .iter()
        .chain(&weak.assurance_evidence_digests)
        .cloned()
        .collect::<BTreeSet<_>>();
    for input in [vec![strong.clone(), weak.clone()], vec![weak, strong]] {
        let remapped = remap_destination_drafts(&input, PublicationDestination::Private).unwrap();
        assert_eq!(remapped.len(), 1);
        assert_eq!(
            remapped[0].achieved_assurance,
            ArtifactAssuranceState::Certified
        );
        assert_eq!(
            remapped[0].required_assurance,
            Some(ArtifactAssuranceState::Certified)
        );
        assert_eq!(
            remapped[0]
                .assurance_evidence_digests
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn destination_aliases_do_not_hide_unbound_transport() {
    let root = std::env::temp_dir().join(format!(
        "xc-publication-alias-forged-{}",
        std::process::id()
    ));
    let valid = fixture_draft(&root);
    let mut forged = valid.clone();
    forged.encoding.canonical_payload_digest = ContentDigest::sha256(b"other-payload");
    for input in [vec![valid.clone(), forged.clone()], vec![forged, valid]] {
        let error = remap_destination_drafts(&input, PublicationDestination::Private).unwrap_err();
        assert!(error.to_string().contains("transport binding"), "{error}");
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn destination_aliases_preserve_distinct_historical_closures() {
    let root = std::env::temp_dir().join(format!(
        "xc-publication-alias-history-{}",
        std::process::id()
    ));
    let first = fixture_draft_with_n(&root, 2);
    let second = fixture_draft_with_n(&root, 3);
    let child = alias_dependencies(fixture_draft_with_n(&root, 4), &[&first]);
    let different = alias_dependencies(child.clone(), &[&second]);
    assert_eq!(
        child.manifest.semantic_digest,
        different.manifest.semantic_digest
    );
    assert_eq!(
        child.manifest.canonical_payload.ordered_items,
        different.manifest.canonical_payload.ordered_items
    );
    let remapped = remap_destination_drafts(
        &[child, different, first, second],
        PublicationDestination::Private,
    )
    .unwrap();
    assert_eq!(
        remapped.len(),
        4,
        "shared semantic key and item bytes do not establish equivalence"
    );
    assert_eq!(
        remapped
            .iter()
            .map(|d| d.manifest.digest().unwrap())
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn destination_aliases_from_reopened_production_staging_remap_successfully() {
    use crate::ArtifactProductionSink;
    let root = std::env::temp_dir().join(format!(
        "xc-publication-alias-staging-{}",
        std::process::id()
    ));
    let fixture = fixture_draft(&root.join("fixture"));
    let payload = b"{\"entries\":[\"1\",\"0\",\"0\",\"1\"]}".to_vec();
    let semantic_key = fixture.manifest.semantic_key.clone();
    let key = crate::ArtifactKey {
        kind: semantic_key.artifact_kind.clone(),
        logical_key: "ccm/tau/alias-fixture".to_owned(),
        parameters_digest: semantic_key.digest().unwrap(),
    };
    let record = crate::ProducedArtifactRecord {
        operation: "ccm.tau.resolve_or_compute".to_owned(),
        logical_key: key.logical_key.clone(),
        semantic_key,
        manifest: crate::ArtifactManifest {
            schema_version: 1,
            key,
            content_digest: ContentDigest::sha256(&payload),
            size_bytes: payload.len() as u64,
            objects: vec![crate::CacheObjectRef {
                content_digest: ContentDigest::sha256(&payload),
                size_bytes: payload.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.15.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            quality: crate::CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: Vec::new(),
            tags: BTreeMap::new(),
            provenance_digest: None,
        },
        achieved_assurance: ArtifactAssuranceState::Computed,
        assurance_evidence_digests: Vec::new(),
        payload,
    };
    let staging = root.join("staging");
    let open = || {
        crate::CanonicalStagingProductionSink::new(
            &staging,
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap()
    };
    let sink = open();
    sink.record(record.clone()).unwrap();
    let neutral = sink.drafts().unwrap().remove(0);
    for destination in [
        PublicationDestination::Private,
        PublicationDestination::Public,
    ] {
        let manifest = target_manifest(&neutral, destination).unwrap();
        let mut reused = record.clone();
        reused.operation = "cache.dependency.closure".to_owned();
        reused.manifest.provenance_digest = Some(manifest.digest().unwrap());
        reused.manifest.tags.insert(
            crate::REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&manifest).unwrap(),
        );
        sink.record(reused).unwrap();
    }
    assert_eq!(
        sink.drafts().unwrap().len(),
        3,
        "staging preserves exact source identities"
    );
    drop(sink);
    let reopened = open();
    let drafts = reopened.drafts().unwrap();
    assert_eq!(drafts.len(), 3);
    for destination in [
        PublicationDestination::Private,
        PublicationDestination::Public,
    ] {
        let remapped = remap_destination_drafts(&drafts, destination).unwrap();
        assert_eq!(remapped.len(), 1);
        assert_eq!(
            remapped[0].manifest,
            target_manifest(&neutral, destination).unwrap()
        );
        assert_eq!(
            remapped[0].encoding, neutral.encoding,
            "archive and part identities are preserved"
        );
        assert_eq!(remapped[0].source_operation, record.operation);
    }
    drop(reopened);
    fs::remove_dir_all(root).unwrap();
}
