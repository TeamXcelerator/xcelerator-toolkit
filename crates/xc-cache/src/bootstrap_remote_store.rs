//! Selective reads from live public and authenticated private registries.

use crate::*;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapRegistry {
    schema_version: u32,
    repository: String,
    default_branch: String,
    families: Vec<BootstrapFamily>,
    #[serde(default)]
    separate_visibility_inventory: Option<bool>,
    #[serde(default)]
    visibility: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapFamily {
    family: String,
    current_writable_shard: String,
    metadata: String,
}

/// Read-only GitHub layer routed by one visibility-specific bootstrap registry.
pub struct GitHubBootstrapCacheStore {
    name: String,
    owner: String,
    root: PathBuf,
    visibility: CacheVisibility,
    required: bool,
    remote: GitCliRemoteStore,
    resolved: Mutex<HashMap<ContentDigest, ResolvedRemoteArtifact>>,
    registry_snapshot: Mutex<Option<BootstrapRegistry>>,
    shard_revisions: Mutex<HashMap<String, String>>,
}

impl GitHubBootstrapCacheStore {
    pub fn public(owner: impl Into<String>, root: impl Into<PathBuf>) -> Result<Self, CacheError> {
        Self::new(owner, root, CacheVisibility::Public, false)
    }

    pub fn private(owner: impl Into<String>, root: impl Into<PathBuf>) -> Result<Self, CacheError> {
        Self::new(owner, root, CacheVisibility::Private, true)
    }

    fn new(
        owner: impl Into<String>,
        root: impl Into<PathBuf>,
        visibility: CacheVisibility,
        required: bool,
    ) -> Result<Self, CacheError> {
        let owner = owner.into();
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            name: format!("github-{}", visibility_name(visibility)),
            owner,
            visibility,
            required,
            remote: GitCliRemoteStore::new(
                root.join("git"),
                root.join("parts"),
                "Xcelerator cache reader",
                "cache-reader@localhost",
            )?,
            root,
            resolved: Mutex::new(HashMap::new()),
            registry_snapshot: Mutex::new(None),
            shard_revisions: Mutex::new(HashMap::new()),
        })
    }

    /// Verify that the bootstrap registry can be reached before a numerical
    /// run starts.  Remote cache reads are otherwise lazy: a cold high
    /// precision run can spend hours computing before its first Git fetch
    /// discovers an expired PAT or a repository permission problem.
    pub fn preflight(&self) -> Result<(), CacheError> {
        self.registry_snapshot().map(|_| ())
    }

    fn registry_snapshot(&self) -> Result<BootstrapRegistry, CacheError> {
        if let Some(registry) = self
            .registry_snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .cloned()
        {
            return Ok(registry);
        }
        let visibility_name = visibility_name(self.visibility);
        let registry_id = format!("{}/xcelerator-cache-{visibility_name}-registry", self.owner);
        let repository = format!("https://github.com/{registry_id}.git");
        let revision = self.remote.read_ref(&repository, "main")?;
        let (registry, _): (BootstrapRegistry, _) =
            self.read_json(&repository, &revision, "registry.json")?;
        if registry.schema_version != 1
            || registry.repository != registry_id
            || registry.default_branch != "main"
            || registry
                .visibility
                .as_deref()
                .is_some_and(|declared| declared != visibility_name)
            || registry
                .separate_visibility_inventory
                .is_some_and(|separate| !separate)
        {
            return Err(CacheError::InvalidManifest(format!(
                "{visibility_name} bootstrap registry identity is invalid"
            )));
        }
        *self
            .registry_snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(registry.clone());
        Ok(registry)
    }

    fn shard_revision(&self, repository: &str) -> Result<String, CacheError> {
        if let Some(revision) = self
            .shard_revisions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(repository)
            .cloned()
        {
            return Ok(revision);
        }
        let revision = self.remote.read_ref(repository, "main")?;
        self.shard_revisions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(repository.to_owned(), revision.clone());
        Ok(revision)
    }

    fn read_json<T: serde::de::DeserializeOwned>(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
    ) -> Result<(T, RemoteReadReport), CacheError> {
        let mut bytes = Vec::new();
        let report = self.remote.read_committed_path(
            repository,
            revision,
            path,
            16 * 1024 * 1024,
            &CancellationToken::new(),
            &mut bytes,
        )?;
        Ok((serde_json::from_slice(&bytes)?, report))
    }

    fn resolve(&self, key: &ArtifactKey) -> Result<Option<ResolvedRemoteArtifact>, CacheError> {
        let Some(family) = family_for_artifact_kind(&key.kind) else {
            return Ok(None);
        };
        let visibility_name = visibility_name(self.visibility);
        let registry = self.registry_snapshot()?;
        let route = registry
            .families
            .iter()
            .find(|route| route.family == family)
            .ok_or_else(|| CacheError::NotFound(format!("public family {family}")))?;
        if route.metadata.trim().is_empty()
            || !route.current_writable_shard.starts_with(&format!(
                "{}/xcelerator-cache-{visibility_name}-{family}-",
                self.owner
            ))
        {
            return Err(CacheError::InvalidManifest(format!(
                "{visibility_name} route for {family} is invalid"
            )));
        }
        let repository = format!("https://github.com/{}.git", route.current_writable_shard);
        let revision = self.shard_revision(&repository)?;
        let prefix = &key.parameters_digest.0[..2];
        let index_path = format!("indexes/{family}/{prefix}.json");
        let (index, index_source): (ShardIndexPartition, _) =
            match self.read_json(&repository, &revision, &index_path) {
                Ok(value) => value,
                Err(CacheError::NotFound(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
        index.validate()?;
        let current = crate::current_toolkit_version()?;
        let Some(entry) = index
            .lookup(&key.parameters_digest)
            .filter(|entry| entry.disposition == ArtifactDisposition::Active)
            .filter(|entry| entry.achieved_assurance.mathematical().is_some())
            .filter(|entry| entry.minimum_reader_version <= current)
            .max_by_key(|entry| (entry.achieved_assurance, entry.manifest_digest.clone()))
            .cloned()
        else {
            return Ok(None);
        };
        let manifest_path = format!("manifests/{prefix}/{}.json", entry.manifest_digest.0);
        let (manifest, manifest_source): (CanonicalArtifactManifest, _) =
            self.read_json(&repository, &revision, &manifest_path)?;
        manifest.validate()?;
        let transport_digest = entry.transport_digests.first().ok_or_else(|| {
            CacheError::InvalidManifest(format!(
                "{visibility_name} index has no transport encoding"
            ))
        })?;
        let payload_prefix = &entry.canonical_payload_digest.0[..2];
        let encoding_path = format!("encodings/{payload_prefix}/{}.json", transport_digest.0);
        let (encoding, encoding_source): (TransportEncodingRecord, _) =
            self.read_json(&repository, &revision, &encoding_path)?;
        encoding.validate()?;
        let expected_destination = match self.visibility {
            CacheVisibility::Private => PublicationDestination::Private,
            CacheVisibility::Public => PublicationDestination::Public,
            _ => unreachable!("GitHub bootstrap stores are private or public"),
        };
        let batch_path = format!(
            "transactions/batches/{}/{visibility_name}.json",
            entry.publication_transaction_id,
        );
        let (batch, receipt_source): (RepositoryPublicationBatch, _) =
            self.read_json(&repository, &revision, &batch_path)?;
        validate_bootstrap_batch(
            &batch,
            &receipt_source,
            expected_destination,
            family,
            &route.current_writable_shard,
            &key.parameters_digest,
            &entry,
            &manifest_path,
            transport_digest,
        )?;
        let resolved = ResolvedRemoteArtifact {
            family: family.to_owned(),
            semantic_digest: key.parameters_digest.clone(),
            overlay: self.name.clone(),
            visibility: self.visibility,
            shard_id: route.current_writable_shard.clone(),
            authorized_repository: route.current_writable_shard.clone(),
            repository,
            revision,
            index: entry,
            manifest,
            encoding,
            receipt: crate::RemotePublicationEvidence::RepositoryBatch(Box::new(batch)),
            index_source,
            manifest_source,
            encoding_source,
            receipt_source,
            dependencies: Vec::new(),
        };
        // The materializer repeats all payload/manifest/receipt identity checks.
        Ok(Some(resolved))
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_bootstrap_batch(
    batch: &RepositoryPublicationBatch,
    source: &RemoteReadReport,
    expected_destination: PublicationDestination,
    family: &str,
    authorized_repository: &str,
    semantic_digest: &ContentDigest,
    entry: &ShardIndexEntry,
    manifest_path: &str,
    transport_digest: &ContentDigest,
) -> Result<(), CacheError> {
    batch.validate()?;
    if source.content_digest != batch.digest()? {
        return Err(CacheError::InvalidManifest(
            "repository batch bytes are not canonical".to_owned(),
        ));
    }
    let artifact = batch
        .artifacts
        .iter()
        .find(|artifact| {
            &artifact.semantic_digest == semantic_digest
                && artifact.manifest_digest == entry.manifest_digest
        })
        .ok_or_else(|| {
            CacheError::InvalidManifest(
                "repository batch does not contain the selected bootstrap artifact".to_owned(),
            )
        })?;
    if batch.batch_id.0 != entry.publication_transaction_id
        || batch.destination != expected_destination
        || batch.family != family
        || batch.authorized_repository != authorized_repository
        || batch.branch != "main"
        || artifact.canonical_payload_digest != entry.canonical_payload_digest
        || artifact.transport_digest != *transport_digest
        || artifact.manifest_path != manifest_path
        || artifact.achieved_assurance != entry.achieved_assurance
        || artifact.producer_toolkit_version != entry.producer_toolkit_version
    {
        return Err(CacheError::InvalidManifest(
            "repository batch does not prove this bootstrap index entry".to_owned(),
        ));
    }
    Ok(())
}

impl CacheStore for GitHubBootstrapCacheStore {
    fn name(&self) -> &str {
        &self.name
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        self.visibility
    }
    fn put(&self, _draft: &ArtifactDraft, _payload: &[u8]) -> Result<ArtifactManifest, CacheError> {
        Err(CacheError::ReadOnlyLayer(self.name.clone()))
    }
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        // Remote reuse is an optimization for normal consumers. An offline,
        // unavailable, or invalid public source must safely degrade to a fresh
        // local computation rather than make the research script unusable.
        let resolved = if self.required {
            self.resolve(key)?
        } else {
            self.resolve(key).unwrap_or(None)
        };
        let Some(resolved) = resolved else {
            return Ok(Vec::new());
        };
        let ordered_items = &resolved.manifest.canonical_payload.ordered_items;
        if ordered_items.len() != 1 || ordered_items[0].normalized_path != "payload.json" {
            return Err(CacheError::InvalidManifest(
                "managed JSON artifact must contain exactly one payload.json item".to_owned(),
            ));
        }
        let item = &ordered_items[0];
        let assurance = resolved.index.achieved_assurance;
        let quality = match assurance {
            ArtifactAssuranceState::Certified => CacheQuality::Certified,
            ArtifactAssuranceState::CrossChecked => CacheQuality::CrossChecked,
            _ => CacheQuality::Validated,
        };
        let mut tags = BTreeMap::new();
        tags.insert(
            SEMANTIC_KEY_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&resolved.manifest.semantic_key)?,
        );
        tags.insert(
            REMOTE_CANONICAL_MANIFEST_TAG.to_owned(),
            serde_json::to_string(&resolved.manifest)?,
        );
        let adapter_manifest = ArtifactManifest {
            schema_version: 1,
            key: key.clone(),
            content_digest: item.content_digest.clone(),
            size_bytes: item.size_bytes,
            objects: vec![CacheObjectRef {
                content_digest: item.content_digest.clone(),
                size_bytes: item.size_bytes,
            }],
            created_unix_seconds: resolved.receipt.verified_at_unix_seconds(),
            producer_toolkit_version: resolved.manifest.producer_toolkit_version.clone(),
            minimum_reader_version: resolved.manifest.minimum_reader_version.clone(),
            maximum_reader_version: resolved.manifest.maximum_reader_version.clone(),
            quality,
            visibility: self.visibility,
            immutable: true,
            dependencies: Vec::new(),
            tags,
            provenance_digest: Some(resolved.manifest.digest()?),
        };
        self.resolved
            .lock()
            .map_err(|_| CacheError::Io("public cache lock poisoned".to_owned()))?
            .insert(adapter_manifest.content_digest.clone(), resolved);
        Ok(vec![adapter_manifest])
    }
    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        let resolved = self
            .resolved
            .lock()
            .map_err(|_| CacheError::Io("public cache lock poisoned".to_owned()))?
            .get(&manifest.content_digest)
            .cloned()
            .ok_or_else(|| CacheError::NotFound(manifest.content_digest.to_string()))?;
        let package = self
            .root
            .join("packages")
            .join(format!("{}.zip", resolved.encoding.digest()?.0));
        if let Some(parent) = package.parent() {
            fs::create_dir_all(parent)?;
        }
        let started = Instant::now();
        let report = materialize_resolved_remote_artifact_to_writer(
            &self.remote,
            &resolved,
            &self.root.join("parts"),
            &package,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            writer,
        )?;
        if resolved.encoding.package_size_bytes >= 64 * 1024 * 1024 {
            eprintln!(
                "  cache transport: {} {:.1} MB, {} parts ({} downloaded, {} reused), verified and decoded once in {:.3}s",
                resolved.family,
                resolved.encoding.package_size_bytes as f64 / 1_000_000.0,
                resolved.encoding.ordered_parts.len(),
                report
                    .part_fetch
                    .as_ref()
                    .map_or(0, |parts| parts.downloaded_sequences.len()),
                report
                    .part_fetch
                    .as_ref()
                    .map_or(resolved.encoding.ordered_parts.len(), |parts| {
                        parts.reused_sequences.len()
                    }),
                started.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    fn verified_encoded_payload(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<Option<VerifiedEncodedPayload>, CacheError> {
        let resolved = self
            .resolved
            .lock()
            .map_err(|_| CacheError::Io("public cache lock poisoned".to_owned()))?
            .get(&manifest.content_digest)
            .cloned()
            .ok_or_else(|| CacheError::NotFound(manifest.content_digest.to_string()))?;
        let path = self
            .root
            .join("packages")
            .join(format!("{}.zip", resolved.encoding.digest()?.0));
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(VerifiedEncodedPayload {
            path,
            encoding: "zip-json-entry-v1".to_owned(),
            content_digest: resolved.encoding.package_digest,
            size_bytes: resolved.encoding.package_size_bytes,
        }))
    }
}

fn visibility_name(visibility: CacheVisibility) -> &'static str {
    match visibility {
        CacheVisibility::Public => "public",
        CacheVisibility::Private => "private",
        _ => unreachable!("GitHub bootstrap stores are private or public"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BootstrapBatchFixture {
        batch: RepositoryPublicationBatch,
        entry: ShardIndexEntry,
        source: RemoteReadReport,
        semantic_digest: ContentDigest,
        manifest_path: String,
        transport_digest: ContentDigest,
    }

    fn bootstrap_batch_fixture() -> BootstrapBatchFixture {
        let semantic_digest = ContentDigest::sha256(b"bootstrap-semantic");
        let payload_digest = ContentDigest::sha256(b"bootstrap-payload");
        let manifest_digest = ContentDigest::sha256(b"bootstrap-manifest");
        let transport_digest = ContentDigest::sha256(b"bootstrap-transport");
        let manifest_path = format!(
            "manifests/{}/{}.json",
            &semantic_digest.0[..2],
            manifest_digest.0
        );
        let version = ToolkitVersion::parse("0.13.0").unwrap();
        let batch = RepositoryPublicationBatch::new(
            PublicationDestination::Private,
            "ccm-matrices",
            "fixture-author",
            "TeamXcelerator/xcelerator-cache-private-ccm-matrices-0001",
            "main",
            ContentDigest::sha256(b"bootstrap-policy"),
            Some(1),
            vec![RepositoryBatchArtifact {
                semantic_digest: semantic_digest.clone(),
                canonical_payload_digest: payload_digest.clone(),
                manifest_digest: manifest_digest.clone(),
                transport_digest: transport_digest.clone(),
                manifest_path: manifest_path.clone(),
                achieved_assurance: ArtifactAssuranceState::Computed,
                producer_toolkit_version: version.clone(),
                provenance_evidence_digests: vec![ContentDigest::sha256(b"bootstrap-provenance")],
            }],
            1,
        )
        .unwrap();
        let entry = ShardIndexEntry {
            semantic_digest: semantic_digest.clone(),
            canonical_payload_digest: payload_digest,
            manifest_digest,
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: version.clone(),
            minimum_reader_version: version,
            transport_digests: vec![transport_digest.clone()],
            publication_transaction_id: batch.batch_id.0.clone(),
        };
        let source = RemoteReadReport {
            repository_path: batch.repository_path(),
            revision: "a".repeat(40),
            size_bytes: 1,
            content_digest: batch.digest().unwrap(),
        };
        BootstrapBatchFixture {
            batch,
            entry,
            source,
            semantic_digest,
            manifest_path,
            transport_digest,
        }
    }

    #[test]
    fn current_repository_batch_proves_bootstrap_index_entry() {
        let fixture = bootstrap_batch_fixture();
        validate_bootstrap_batch(
            &fixture.batch,
            &fixture.source,
            PublicationDestination::Private,
            "ccm-matrices",
            "TeamXcelerator/xcelerator-cache-private-ccm-matrices-0001",
            &fixture.semantic_digest,
            &fixture.entry,
            &fixture.manifest_path,
            &fixture.transport_digest,
        )
        .unwrap();
    }

    #[test]
    fn bootstrap_batch_rejects_unproven_index_entry() {
        let mut fixture = bootstrap_batch_fixture();
        fixture.entry.canonical_payload_digest = ContentDigest::sha256(b"wrong-payload");
        let error = validate_bootstrap_batch(
            &fixture.batch,
            &fixture.source,
            PublicationDestination::Private,
            "ccm-matrices",
            "TeamXcelerator/xcelerator-cache-private-ccm-matrices-0001",
            &fixture.semantic_digest,
            &fixture.entry,
            &fixture.manifest_path,
            &fixture.transport_digest,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not prove this bootstrap index entry"));
    }

    #[test]
    #[ignore = "read-only live GitHub consumer acceptance"]
    fn claim_1a_artifacts_resolve_from_the_public_bootstrap_fabric() {
        let root = std::env::temp_dir().join("xc-public-consumer-acceptance");
        let store = GitHubBootstrapCacheStore::public("TeamXcelerator", root).unwrap();
        assert_claim_1a_resolves(&store, CacheVisibility::Public);
    }

    #[test]
    #[ignore = "authenticated read-only live GitHub acceptance"]
    fn claim_1a_artifacts_resolve_from_the_private_bootstrap_fabric() {
        let root = std::env::temp_dir().join("xc-private-consumer-acceptance");
        let store = GitHubBootstrapCacheStore::private("TeamXcelerator", root).unwrap();
        assert_claim_1a_resolves(&store, CacheVisibility::Private);
    }

    fn assert_claim_1a_resolves(
        store: &GitHubBootstrapCacheStore,
        expected_visibility: CacheVisibility,
    ) {
        for (kind, digest) in [
            (
                "gauss_legendre_rule",
                "a23310358f5f5e1f83db0cef44bd27d2a26ff1bdbffd94f02292bdba64f8b3fa",
            ),
            (
                "ccm_tau_matrix",
                "a1e6432315f7e7a3af1d85c03957c5ce641cd9e00599dffec0d692e1d29a1408",
            ),
            (
                "ccm_weil_eigenpair",
                "861e1a56ba4c38b9046d9a49b903b9bf98878c6d2a3b6f9ad7ce8a23c8ce3c63",
            ),
        ] {
            let key = ArtifactKey {
                kind: kind.to_owned(),
                logical_key: "live-claim-1a".to_owned(),
                parameters_digest: ContentDigest(digest.to_owned()),
            };
            let resolved = store.resolve(&key).unwrap().expect("published artifact");
            assert_eq!(resolved.manifest.semantic_digest, key.parameters_digest);
            assert_eq!(resolved.visibility, expected_visibility);
            if kind == "ccm_weil_eigenpair" {
                let manifest = store.candidates(&key).unwrap().pop().unwrap();
                let payload = store.read_payload(&manifest).unwrap();
                let json: serde_json::Value = serde_json::from_slice(&payload).unwrap();
                assert_eq!(json["n_modes"], 120);
            }
        }
    }
}
