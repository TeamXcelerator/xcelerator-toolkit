//! Explicit resource limits and publication-only recovery of retained artifacts.
use super::*;
use xc_core::ResourcePolicy;

pub(super) fn validate_resources(resources: &ResourcePolicy) -> Result<(), CacheError> {
    for (name, limit) in [
        ("memory", resources.maximum_memory_bytes),
        ("temporary disk", resources.maximum_temporary_disk_bytes),
        ("permanent disk", resources.maximum_permanent_disk_bytes),
        ("transfer", resources.maximum_transfer_bytes),
        ("CPU seconds", resources.maximum_cpu_seconds),
        ("wall seconds", resources.maximum_wall_seconds),
        ("threads", resources.maximum_threads.map(|n| n as u64)),
        ("checkpoint interval", resources.checkpoint_interval_seconds),
    ] {
        if limit == Some(0) {
            return Err(CacheError::InvalidManifest(format!(
                "resource policy {name} limit must be positive or null"
            )));
        }
    }
    Ok(())
}

fn read_resource_policy(path: &Path) -> Result<ResourcePolicy, CacheError> {
    // Bound configuration input independently of the policy being loaded.
    let mut bytes = Vec::new();
    fs::File::open(path)?.take(65_537).read_to_end(&mut bytes)?;
    if bytes.len() > 65_536 {
        return Err(CacheError::ResourceLimit(
            "resource policy file exceeds 64 KiB".to_owned(),
        ));
    }
    let resources: ResourcePolicy = serde_json::from_slice(&bytes)?;
    validate_resources(&resources)?;
    Ok(resources)
}

/// Load `XC_RESOURCE_POLICY_FILE`, or keep the normal workstation defaults.
/// The JSON uses the complete `xc_core::ResourcePolicy` schema. Unknown fields,
/// oversized files and zero limits are rejected before cache/network setup.
/// These are resource ceilings, not changes to numerical precision or identities.
pub fn managed_resource_policy_from_environment() -> Result<ResourcePolicy, CacheError> {
    match std::env::var_os("XC_RESOURCE_POLICY_FILE") {
        Some(path) if path.is_empty() => Err(CacheError::InvalidManifest(
            "XC_RESOURCE_POLICY_FILE must not be empty".to_owned(),
        )),
        Some(path) => read_resource_policy(Path::new(&path)),
        None => Ok(ResourcePolicy::default()),
    }
}

impl ManagedArtifactCacheSession {
    /// Resolved ceilings used for remote reads, staging and publication.
    pub fn resources(&self) -> &ResourcePolicy {
        &self.resources
    }

    /// Stage an exact retained artifact and its dependency closure without
    /// executing a numerical producer or domain validator. Existing cache
    /// quality is accepted under the normal policy; no assurance is upgraded.
    /// Encoded hashes and canonical logical payload bindings are verified by
    /// the normal staging path using bounded streaming memory. Missing,
    /// incompatible or unprofiled objects fail; there is no compute fallback.
    /// Remote publication remains a separate, explicitly configured finalization.
    pub fn stage_cached_artifact(&self, artifact: &crate::DependencyRef) -> Result<(), CacheError> {
        if !artifact.key.parameters_digest.validate() || !artifact.content_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "recovery requires exact valid artifact digests".to_owned(),
            ));
        }
        let sink = self.production_sink.as_ref().ok_or_else(|| {
            CacheError::InvalidTransition(
                "publication recovery requires staging and a reuse-capable cache mode".to_owned(),
            )
        })?;
        eprintln!(
            "publication recovery: verifying retained encoded object for {}",
            artifact.key.kind
        );
        let resolved = self.resolver.resolve_exact_encoded(
            &artifact.key, &artifact.content_digest, artifact.required_quality, &self.policy,
        )?.ok_or_else(|| CacheError::NotFound(
            "exact recovery artifact has no supported verified encoded representation; no computation attempted".to_owned()))?;
        manifest_semantic_key(&resolved.manifest)?;
        eprintln!(
            "publication recovery: verified {} encoded bytes; staging exact dependencies",
            resolved.encoded.size_bytes
        );
        let mut visiting = BTreeSet::new();
        emit_dependency_closure(
            &self.resolver,
            &self.policy,
            sink,
            &resolved.manifest,
            &mut visiting,
        )?;
        emit_retained_canonical_closure(
            &self.resolver,
            &self.policy,
            sink,
            &resolved.manifest,
            &mut visiting,
        )?;
        eprintln!(
            "publication recovery: staging retained payload and verifying its logical digest"
        );
        record_encoded_dependency(sink, "cache.publication.recover", resolved)?;
        crate::atomic_replace(
            &sink.staging_root().join("resource-policy.json"),
            &serde_json::to_vec_pretty(&self.resources)?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheStore, DependencyRef, ZipJsonFilesystemCacheStore};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!(
            "xc-recovery-{}-{}-{}",
            unique,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }
    fn config(root: &Path) -> ManagedArtifactCacheConfig {
        ManagedArtifactCacheConfig {
            profile: ManagedRunProfile::Author,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            cache_root: root.join("cache"),
            staging_root: Some(root.join("staging")),
            publication_target: xc_core::PublicationTarget::None,
            repository_owner: "fixture".to_owned(),
            remote_cache_mode: ManagedRemoteCacheMode::None,
            cache_mode: ArtifactExecutionCacheMode::PreferReuse,
            replace_existing_publication: false,
            execute_remote_mutations: false,
            output_validation: None,
        }
    }
    fn semantic(kind: &str) -> SemanticKeyEnvelope {
        SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: kind.to_owned(),
            mathematical_semantics_version: "recovery-test-v1".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({"test":1}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        }
    }
    fn artifact_key(logical: &str, semantic: &SemanticKeyEnvelope) -> ArtifactKey {
        ArtifactKey {
            kind: semantic.artifact_kind.clone(),
            logical_key: logical.to_owned(),
            parameters_digest: semantic.digest().unwrap(),
        }
    }
    fn produce(root: &Path, kind: &str, dependencies: Vec<DependencyRef>) -> ArtifactManifest {
        let key = semantic(kind);
        ZipJsonFilesystemCacheStore::new(
            "workstation",
            root.join("cache"),
            true,
            CacheVisibility::Local,
        )
        .put(
            &ArtifactDraft {
                schema_version: 1,
                key: artifact_key(kind, &key),
                producer_toolkit_version: crate::current_toolkit_version().unwrap(),
                minimum_reader_version: ToolkitVersion::parse("0.15.0").unwrap(),
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies,
                tags: BTreeMap::from([(
                    SEMANTIC_KEY_MANIFEST_TAG.to_owned(),
                    serde_json::to_string(&key).unwrap(),
                )]),
                provenance_digest: None,
            },
            b"{\"result\":[1,2,3]}",
        )
        .unwrap()
    }
    fn dep(m: &ArtifactManifest) -> DependencyRef {
        DependencyRef {
            key: m.key.clone(),
            content_digest: m.content_digest.clone(),
            required_quality: CacheQuality::Validated,
        }
    }
    #[test]
    fn publication_recovery_keeps_produced_payload_after_limit_and_stages_without_compute() {
        let root = temp();
        let source = produce(&root, "ccm_prime_component", vec![]);
        let key = semantic("ccm_tau_matrix");
        let limited = ResourcePolicy {
            maximum_transfer_bytes: Some(1),
            ..ResourcePolicy::default()
        };
        let failed =
            ManagedArtifactCacheSession::new_with_resources(config(&root), limited).unwrap();
        let cache = failed.context();
        let calls = AtomicUsize::new(0);
        let req = ArtifactExecutionCacheRequest {
            operation: "test.recovery",
            semantic_key: &key,
            logical_key: "recoverable",
            resolver: cache.resolver,
            reference_resolver: None,
            acceptance: cache.acceptance,
            ordered_overlays: cache.ordered_overlays.clone(),
            mode: cache.mode,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: crate::current_toolkit_version().unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.15.0").unwrap(),
            maximum_reader_version: None,
            tags: BTreeMap::new(),
            provenance_digest: None,
            production_sink: cache.production_sink,
        };
        let error = resolve_or_compute_json_artifact_with_dependencies(
            &req,
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok((vec![11, 22, 33], vec![dep(&source)]))
            },
            |_| Ok(()),
        )
        .err()
        .unwrap();
        assert!(matches!(error, CacheError::ResourceLimit(_)), "{error}");
        let artifact = DependencyRef {
            key: artifact_key("recoverable", &key),
            content_digest: ContentDigest::sha256(b"[11,22,33]"),
            required_quality: CacheQuality::Validated,
        };
        drop(failed);
        let session = ManagedArtifactCacheSession::new_with_resources(
            config(&root),
            ResourcePolicy::default(),
        )
        .unwrap();
        session.stage_cached_artifact(&artifact).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(session.staged_drafts().unwrap().len(), 2);
        assert!(session.staged_drafts().unwrap().iter().any(|d| d
            .manifest
            .canonical_payload
            .ordered_items[0]
            .content_digest
            == artifact.content_digest));
        session.finalize_publication_inventory().unwrap();
        drop(session);
        let reopened = ManagedArtifactCacheSession::new(config(&root)).unwrap();
        reopened.stage_cached_artifact(&artifact).unwrap();
        assert_eq!(reopened.staged_drafts().unwrap().len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn recovery_rejects_corrupt_encoded_objects_and_missing_exact_payloads() {
        let root = temp();
        let manifest = produce(&root, "ccm_tau_matrix", vec![]);
        let session = ManagedArtifactCacheSession::new(config(&root)).unwrap();
        let mut wrong = dep(&manifest);
        wrong.content_digest = ContentDigest("c".repeat(64));
        assert!(session.stage_cached_artifact(&wrong).is_err());
        let o = &manifest.objects[0];
        let path = root
            .join("cache/objects/sha256")
            .join(&o.content_digest.0[..2])
            .join(&o.content_digest.0);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(session.stage_cached_artifact(&dep(&manifest)).is_err());
        assert!(session.staged_drafts().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn recovery_rejects_missing_dependencies_without_publishing_a_root() {
        let root = temp();
        let absent = DependencyRef {
            key: artifact_key("absent", &semantic("ccm_prime_component")),
            content_digest: ContentDigest("d".repeat(64)),
            required_quality: CacheQuality::Validated,
        };
        let m = produce(&root, "ccm_tau_matrix", vec![absent]);
        let session = ManagedArtifactCacheSession::new(config(&root)).unwrap();
        assert!(session.stage_cached_artifact(&dep(&m)).is_err());
        assert!(session.staged_drafts().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn resource_policy_file_is_exact_bounded_and_rejects_zero_limits() {
        let root = temp();
        let path = root.join("policy.json");
        let expected = ResourcePolicy {
            maximum_transfer_bytes: Some(64 * 1024 * 1024 * 1024),
            ..ResourcePolicy::default()
        };
        fs::write(&path, serde_json::to_vec(&expected).unwrap()).unwrap();
        assert_eq!(read_resource_policy(&path).unwrap(), expected);
        let mut invalid = serde_json::to_value(&expected).unwrap();
        invalid["typo"] = true.into();
        fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(read_resource_policy(&path).is_err());
        fs::write(&path, vec![b' '; 65_537]).unwrap();
        assert!(read_resource_policy(&path).is_err());
        assert!(ManagedArtifactCacheSession::new_with_resources(
            config(&root),
            ResourcePolicy {
                maximum_transfer_bytes: Some(0),
                ..expected
            }
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
