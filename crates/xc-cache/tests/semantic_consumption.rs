use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use xc_cache::*;
use xc_core::{AssuranceLevel, CancellationToken, PublicationAuthorityMode};

struct MemoryRemote {
    heads: BTreeMap<(String, String), String>,
    documents: BTreeMap<(String, String, String), Vec<u8>>,
}

impl RemoteGitStore for MemoryRemote {
    fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError> {
        self.heads
            .get(&(repository.to_owned(), branch.to_owned()))
            .cloned()
            .ok_or_else(|| CacheError::NotFound(format!("{repository}:{branch}")))
    }
    fn immutable_path_digest(
        &self,
        _: &str,
        _: &str,
        _: &str,
    ) -> Result<Option<ContentDigest>, CacheError> {
        Ok(None)
    }
    fn read_committed_path(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        maximum_bytes: u64,
        _cancellation: &CancellationToken,
        writer: &mut dyn Write,
    ) -> Result<RemoteReadReport, CacheError> {
        let bytes = self
            .documents
            .get(&(repository.to_owned(), revision.to_owned(), path.to_owned()))
            .ok_or_else(|| CacheError::NotFound(format!("{revision}:{path}")))?;
        if bytes.len() as u64 > maximum_bytes {
            return Err(CacheError::ResourceLimit(path.to_owned()));
        }
        writer.write_all(bytes)?;
        Ok(RemoteReadReport {
            repository_path: path.to_owned(),
            revision: revision.to_owned(),
            size_bytes: bytes.len() as u64,
            content_digest: ContentDigest::sha256(bytes),
        })
    }
    fn compare_and_swap_commit(
        &self,
        _: &RemoteCommitRequest,
    ) -> Result<CompareAndSwapResult, CacheError> {
        panic!("resolution must not mutate")
    }
    fn verify_committed_part(&self, _: &str, _: &str, _: &TransportPart) -> Result<(), CacheError> {
        panic!("metadata resolution must not fetch parts")
    }
}

/// Canonical JSON bytes (sorted keys, compact) through serde_json::Value,
/// whose map is a BTreeMap in this build; equal to the toolkit's canonical
/// form for these float-free records.
fn canonical<T: serde_json_ser::S>(value: &T) -> Vec<u8> {
    value.bytes()
}
mod serde_json_ser {
    pub trait S {
        fn bytes(&self) -> Vec<u8>;
    }
    macro_rules! impl_s {
        ($($t:ty),*) => {$(impl S for $t {
            fn bytes(&self) -> Vec<u8> {
                serde_json::to_vec(&serde_json::to_value(self).unwrap()).unwrap()
            }
        })*};
    }
    use xc_cache::*;
    impl_s!(
        CanonicalArtifactManifest,
        TransportEncodingRecord,
        ShardIndexPartition,
        PublicationReceipt,
        TopologyRegistry
    );
}

struct Artifact {
    semantic_key: SemanticKeyEnvelope,
    semantic_digest: ContentDigest,
    manifest: CanonicalArtifactManifest,
    manifest_digest: ContentDigest,
    payload_digest: ContentDigest,
    encoding: TransportEncodingRecord,
    transport_digest: ContentDigest,
    transaction_id: String,
    assurance: ArtifactAssuranceState,
}

fn artifact(
    kind: &str,
    precision_bits: Option<u64>,
    backend: &str,
    dependencies: Vec<PayloadDependencyIdentity>,
    assurance: ArtifactAssuranceState,
    seed: &str,
) -> Artifact {
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.to_owned(),
        mathematical_semantics_version: "r24-v1".to_owned(),
        resolved_mathematical_parameters: json!({"seed": seed}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    let semantic_digest = semantic_key.digest().unwrap();
    let canonical_payload = CanonicalPayloadEnvelope {
        schema_version: 1,
        scalar_backend: backend.to_owned(),
        precision_bits,
        scalar_representation: "canonical-json-utf8-v1".to_owned(),
        dimensions: vec![],
        endianness: "not-applicable".to_owned(),
        special_value_encoding: "decimal-string-or-json-number-v1".to_owned(),
        ordered_items: vec![LogicalPayloadItem {
            normalized_path: "payload.json".to_owned(),
            content_digest: ContentDigest::sha256(seed.as_bytes()),
            size_bytes: seed.len() as u64,
        }],
        dependencies,
    };
    let payload_digest = canonical_payload.digest().unwrap();
    let part_digest = ContentDigest::sha256(format!("package-{seed}").as_bytes());
    let encoding = TransportEncodingRecord {
        schema_version: 1,
        canonical_payload_digest: payload_digest.clone(),
        encoder_profile: DETERMINISTIC_ZIP64_PROFILE_V1.to_owned(),
        package_size_bytes: 15,
        package_digest: part_digest.clone(),
        ordered_parts: vec![TransportPart {
            sequence: 0,
            repository_path: format!("objects/{}.part", part_digest.0),
            size_bytes: 15,
            content_digest: part_digest,
        }],
        reconstruction: "concatenate".to_owned(),
    };
    let transport_digest = encoding.digest().unwrap();
    let manifest = CanonicalArtifactManifest {
        schema_version: 1,
        artifact_family: "ccm".to_owned(),
        semantic_key: semantic_key.clone(),
        semantic_digest: semantic_digest.clone(),
        canonical_payload,
        payload_digest: payload_digest.clone(),
        transport_digests: vec![transport_digest.clone()],
        // As production staging does: digest of the resolved parameters.
        resolved_mathematical_configuration_digest: ContentDigest::sha256(
            &serde_json::to_vec(&json!({"seed": seed})).unwrap(),
        ),
        producer_toolkit_version: ToolkitVersion::parse("0.15.2").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.15.2").unwrap(),
        maximum_reader_version: None,
        requested_assurance: AssuranceLevel::Computed,
        claim_scope: "r24 manufactured fabric".to_owned(),
        assumptions: vec![],
    };
    let manifest_digest = manifest.digest().unwrap();
    Artifact {
        transaction_id: ContentDigest::sha256(format!("tx-{seed}").as_bytes()).0,
        semantic_key,
        semantic_digest,
        manifest,
        manifest_digest,
        payload_digest,
        encoding,
        transport_digest,
        assurance,
    }
}

fn identity(a: &Artifact) -> PayloadDependencyIdentity {
    PayloadDependencyIdentity {
        artifact_family: "ccm".to_owned(),
        semantic_digest: a.semantic_digest.clone(),
        manifest_digest: a.manifest_digest.clone(),
        payload_digest: a.payload_digest.clone(),
    }
}

struct Fabric {
    remote: MemoryRemote,
    overlays: Vec<RemoteResolverOverlay>,
    policy_digest: ContentDigest,
}

fn fabric(artifacts: &[&Artifact]) -> Fabric {
    let topology_repository = "example-org/topology".to_owned();
    let topology_revision = "a".repeat(40);
    let shard_revision = "b".repeat(40);
    let policy_digest = ContentDigest::sha256(b"publication-policy");
    let topology = TopologyRegistry {
        schema_version: 1,
        generation: 3,
        previous_registry_digest: Some(ContentDigest::sha256(b"generation-2")),
        policy_digest: policy_digest.clone(),
        trust_anchor_ids: vec!["release-key".to_owned()],
        family_routes: vec![ArtifactFamilyRoute {
            family: "ccm".to_owned(),
            visibility: CacheVisibility::Private,
            ordered_shards: vec![TopologyShardRoute {
                shard_id: "private-001".to_owned(),
                endpoint_id: "private-001".to_owned(),
                sequence: 1,
                status: TopologyShardStatus::Writable,
                successor_shard_id: None,
            }],
        }],
    };
    let network = CacheNetworkRegistry {
        schema_version: 1,
        repositories: vec![GitHubRepositoryEndpoint {
            shard_id: "private-001".to_owned(),
            owner: "example-org".to_owned(),
            repository: "restricted-cache".to_owned(),
            branch: "main".to_owned(),
            visibility: CacheVisibility::Private,
            enabled_for_read: true,
            enabled_for_write: true,
            clone_via_ssh: false,
        }],
    };
    let shard = network.repositories[0].preferred_clone_url();
    let mut documents = BTreeMap::new();
    documents.insert(
        (
            topology_repository.clone(),
            topology_revision.clone(),
            "registry/topology.json".to_owned(),
        ),
        canonical(&topology),
    );
    let mut partitions: BTreeMap<String, Vec<ShardIndexEntry>> = BTreeMap::new();
    for a in artifacts {
        partitions
            .entry(a.semantic_digest.0[..2].to_owned())
            .or_default()
            .push(ShardIndexEntry {
                semantic_digest: a.semantic_digest.clone(),
                canonical_payload_digest: a.payload_digest.clone(),
                manifest_digest: a.manifest_digest.clone(),
                achieved_assurance: a.assurance,
                disposition: ArtifactDisposition::Active,
                producer_toolkit_version: ToolkitVersion::parse("0.15.2").unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.15.2").unwrap(),
                transport_digests: vec![a.transport_digest.clone()],
                publication_transaction_id: a.transaction_id.clone(),
            });
    }
    let mut index_digests = BTreeMap::new();
    for (prefix, entries) in partitions {
        let index = ShardIndexPartition::rebuild("ccm", prefix.clone(), entries).unwrap();
        let bytes = canonical(&index);
        let path = format!("indexes/ccm/{prefix}.json");
        index_digests.insert(path.clone(), ContentDigest::sha256(&bytes));
        documents.insert((shard.clone(), shard_revision.clone(), path), bytes);
    }
    for a in artifacts {
        let manifest_path = format!(
            "manifests/{}/{}.json",
            &a.semantic_digest.0[..2],
            a.manifest_digest.0
        );
        let encoding_path = format!(
            "encodings/{}/{}.json",
            &a.payload_digest.0[..2],
            a.transport_digest.0
        );
        let index_path = format!("indexes/ccm/{}.json", &a.semantic_digest.0[..2]);
        let receipt = PublicationReceipt {
            schema_version: 1,
            transaction_id: a.transaction_id.clone(),
            idempotency_key: ContentDigest(a.transaction_id.clone()),
            destination: PublicationDestination::Private,
            principal: "test-owner".to_owned(),
            authorized_repository: "example-org/restricted-cache".to_owned(),
            repository_permission_evidence_digest: ContentDigest::sha256(b"permission"),
            shard_id: "private-001".to_owned(),
            branch: "main".to_owned(),
            semantic_digest: a.semantic_digest.clone(),
            canonical_payload_digest: a.payload_digest.clone(),
            manifest_digest: a.manifest_digest.clone(),
            transport_digest: a.transport_digest.clone(),
            policy_digest: policy_digest.clone(),
            policy_id: "fixture-owner-policy".to_owned(),
            authority_mode: PublicationAuthorityMode::OwnerDirect,
            validation_evidence_digests: vec![ContentDigest::sha256(b"validator-evidence")],
            contributor_authorization_digest: None,
            reviewer_approvals: Vec::new(),
            payload_commit_ids: vec!["payload-commit".to_owned()],
            payload_batch_record_commit_ids: Vec::new(),
            payload_batch_record_digests: BTreeMap::new(),
            metadata_commit_id: "metadata-commit".to_owned(),
            metadata_file_digests: BTreeMap::from([
                (manifest_path.clone(), a.manifest_digest.clone()),
                (encoding_path.clone(), a.transport_digest.clone()),
            ]),
            discoverability_subject_digests: BTreeMap::from([(
                index_path.clone(),
                index_digests[&index_path].clone(),
            )]),
            remote_verification_results: vec![RemoteCommitVerificationResult {
                phase: "immutable_metadata".to_owned(),
                sequence: 0,
                commit_id: "metadata-commit".to_owned(),
                verified: true,
                content_digests: vec![a.manifest_digest.clone(), a.transport_digest.clone()],
            }],
            verified_at_unix_seconds: 100,
        };
        receipt.validate().unwrap();
        documents.insert(
            (shard.clone(), shard_revision.clone(), manifest_path),
            canonical(&a.manifest),
        );
        documents.insert(
            (shard.clone(), shard_revision.clone(), encoding_path),
            canonical(&a.encoding),
        );
        documents.insert(
            (
                shard.clone(),
                shard_revision.clone(),
                format!("transactions/{}/private/receipt.json", a.transaction_id),
            ),
            canonical(&receipt),
        );
    }
    let root = |role, repository: &str, revision: &str| {
        let protection = ProtectedBranchStatement {
            schema_version: 1,
            repository: repository.to_owned(),
            branch: "main".to_owned(),
            observed_revision: revision.to_owned(),
            force_pushes_prohibited: true,
            pushes_restricted: true,
            observed_at_unix_seconds: 100,
            valid_until_unix_seconds: 300,
            trust_anchor_id: "release-key".to_owned(),
        };
        TrustedRepositoryRoot {
            role,
            repository: repository.to_owned(),
            owner: "example-org".to_owned(),
            branch: "main".to_owned(),
            revision_policy: TrustedRevisionPolicy::Exact {
                revision: revision.to_owned(),
            },
            branch_protection_digest: protection.digest().unwrap(),
            branch_protection: protection,
        }
    };
    let overlays = vec![RemoteResolverOverlay {
        name: "private".to_owned(),
        visibility: CacheVisibility::Private,
        topology_source: RemoteTopologySource {
            repository: topology_repository.clone(),
            ..RemoteTopologySource::default()
        },
        topology_trust: TopologyTrustPolicy {
            minimum_generation: 3,
            pinned_registry_digest: Some(topology.digest().unwrap()),
            required_trust_anchor: Some("release-key".to_owned()),
        },
        fabric_trust: RemoteFabricTrustPolicy {
            schema_version: 1,
            approved_trust_anchor_ids: ["release-key".to_owned()].into_iter().collect(),
            approved_policy_digests: [policy_digest.clone()].into_iter().collect(),
            repositories: vec![
                root(
                    TrustedRepositoryRole::Registry,
                    &topology_repository,
                    &topology_revision,
                ),
                root(
                    TrustedRepositoryRole::Shard,
                    "example-org/restricted-cache",
                    &shard_revision,
                ),
            ],
        },
        network,
    }];
    Fabric {
        remote: MemoryRemote {
            heads: BTreeMap::from([
                ((topology_repository, "main".to_owned()), topology_revision),
                ((shard, "main".to_owned()), shard_revision),
            ]),
            documents,
        },
        overlays,
        policy_digest,
    }
}

fn query(f: &Fabric, key: &SemanticKeyEnvelope) -> RemoteSemanticQuery {
    RemoteSemanticQuery {
        family: "ccm".to_owned(),
        semantic_key: key.clone(),
        minimum_assurance: ArtifactAssuranceState::Computed,
        allowed_scalar_backends: BTreeSet::new(),
        minimum_precision_bits: None,
        required_configuration_digest: None,
        required_provenance_evidence_digests: BTreeSet::new(),
        current_toolkit_version: ToolkitVersion::parse("0.15.2").unwrap(),
        accepted_publication_policy_digests: [f.policy_digest.clone()].into_iter().collect(),
        allow_deprecated: false,
        evaluation_unix_seconds: 200,
        maximum_topology_bytes: 1 << 20,
        maximum_index_bytes: 1 << 20,
        maximum_manifest_bytes: 1 << 20,
        maximum_encoding_bytes: 1 << 20,
        maximum_receipt_bytes: 1 << 20,
        maximum_revocation_partition_bytes: 1 << 20,
        maximum_dependency_depth: 8,
        maximum_dependency_count: 100,
    }
}

fn resolve(f: &Fabric, q: &RemoteSemanticQuery) -> SemanticResolutionReport {
    resolve_remote_semantic_artifact(&f.remote, &CancellationToken::new(), q, &f.overlays).unwrap()
}

#[test]
fn consumption_minima_bind_the_selected_artifact_and_exact_dependencies() {
    let leaf = artifact(
        "ccm_r2_leaf",
        None,
        "canonical_json",
        vec![],
        ArtifactAssuranceState::Computed,
        "leaf",
    );
    let middle = artifact(
        "ccm_r2_middle",
        Some(64),
        "ieee754",
        vec![identity(&leaf)],
        ArtifactAssuranceState::Computed,
        "middle",
    );
    for extra_level in [false, true] {
        let deps = if extra_level {
            vec![identity(&middle), identity(&leaf)]
        } else {
            vec![identity(&leaf)]
        };
        let child = artifact(
            "ccm_r2_selected",
            Some(256),
            "rug_mpfr",
            deps,
            ArtifactAssuranceState::CrossChecked,
            "selected",
        );
        let f = fabric(&[&leaf, &middle, &child]);
        for mode in 0..5 {
            let mut q = query(&f, &child.semantic_key);
            if mode == 0 || mode == 4 {
                q.required_configuration_digest = Some(
                    child
                        .manifest
                        .resolved_mathematical_configuration_digest
                        .clone(),
                );
            }
            if mode == 1 || mode == 4 {
                q.minimum_assurance = ArtifactAssuranceState::CrossChecked;
            }
            if mode == 2 || mode == 4 {
                q.minimum_precision_bits = Some(128);
            }
            if mode == 3 || mode == 4 {
                q.allowed_scalar_backends = ["rug_mpfr".to_owned()].into_iter().collect();
            }
            let report = resolve(&f, &q);
            let selected = report
                .selected
                .expect("root minima do not apply to exact dependencies");
            assert_eq!(selected.manifest.payload_digest, child.payload_digest);
            assert_eq!(selected.dependencies.len(), if extra_level { 2 } else { 1 });
            assert_eq!(
                selected.dependencies[0].manifest.payload_digest,
                if extra_level {
                    middle.payload_digest.clone()
                } else {
                    leaf.payload_digest.clone()
                }
            );
            if extra_level {
                assert_eq!(
                    selected.dependencies[0].dependencies[0]
                        .manifest
                        .payload_digest,
                    leaf.payload_digest
                );
            }
            // Each policy still rejects a root that fails it.
            if mode == 0 {
                q.required_configuration_digest = Some(
                    leaf.manifest
                        .resolved_mathematical_configuration_digest
                        .clone(),
                );
            }
            if mode == 1 {
                q.minimum_assurance = ArtifactAssuranceState::Certified;
            }
            if mode == 2 {
                q.minimum_precision_bits = Some(257);
            }
            if mode == 3 {
                q.allowed_scalar_backends = ["canonical_json".to_owned()].into_iter().collect();
            }
            if mode < 4 {
                assert!(resolve(&f, &q).selected.is_none());
            }
        }
        // A wrong dependency payload is never accepted, even though dependency
        // consumption minima are intentionally weaker than the root minima.
        let mut wrong = identity(&leaf);
        wrong.payload_digest = ContentDigest::sha256(b"wrong dependency payload");
        let bad = artifact(
            "ccm_r2_bad",
            Some(256),
            "rug_mpfr",
            vec![wrong],
            ArtifactAssuranceState::CrossChecked,
            "bad",
        );
        let f = fabric(&[&leaf, &bad]);
        assert!(resolve(&f, &query(&f, &bad.semantic_key))
            .selected
            .is_none());
        let mut missing = fabric(&[&leaf, &child]);
        missing
            .remote
            .documents
            .retain(|(_, _, path), _| !path.ends_with(&format!("{}.json", leaf.manifest_digest.0)));
        assert!(resolve(&missing, &query(&missing, &child.semantic_key))
            .selected
            .is_none());
    }
}
