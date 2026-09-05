//! Selective reads from live public and authenticated private registries.

use crate::*;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xc_core::{CancellationToken, ResourcePolicy};

type HistoricalBatchInventory = Vec<(RepositoryPublicationBatch, RemoteReadReport)>;
type HistoricalBatchCache = HashMap<(String, String), HistoricalBatchInventory>;
type MetadataDocumentKey = (String, String, String);

#[derive(Default)]
struct MetadataDocumentCache {
    retained_bytes: u64,
    documents: HashMap<MetadataDocumentKey, (Arc<[u8]>, RemoteReadReport)>,
}

fn regular_file_has_exact_size(
    path: &std::path::Path,
    size_bytes: u64,
) -> Result<bool, CacheError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() == size_bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// Return only transport paths that are not already represented by a local
/// complete package or retained part. Content is still verified by the normal
/// materialization path before use; this is only a network-preparation filter.
fn missing_local_transport_paths(
    root: &std::path::Path,
    encoding: &TransportEncodingRecord,
) -> Result<Vec<RemotePathPrefetch>, CacheError> {
    encoding.validate()?;
    let package_path = root
        .join("packages")
        .join(format!("{}.zip", encoding.digest()?.0));
    if regular_file_has_exact_size(&package_path, encoding.package_size_bytes)? {
        return Ok(Vec::new());
    }
    let parts_root = root.join("parts");
    let mut missing = Vec::new();
    for part in &encoding.ordered_parts {
        let part_path = part
            .repository_path
            .split('/')
            .fold(parts_root.clone(), |path, component| path.join(component));
        if !regular_file_has_exact_size(&part_path, part.size_bytes)? {
            missing.push(RemotePathPrefetch {
                repository_path: part.repository_path.clone(),
                maximum_bytes: part.size_bytes,
            });
        }
    }
    Ok(missing)
}

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
    verified_transports: Mutex<HashMap<ContentDigest, VerifiedTransportParts>>,
    discovered_keys: Mutex<HashMap<(String, String), Vec<ArtifactKey>>>,
    continuation_inventory_lock: Mutex<()>,
    registry_snapshot: Mutex<Option<BootstrapRegistry>>,
    family_topologies: Mutex<HashMap<String, crate::bootstrap_topology::BootstrapFamilyTopology>>,
    shard_revisions: Mutex<HashMap<String, String>>,
    historical_batches: Mutex<HistoricalBatchCache>,
    metadata_documents: Mutex<MetadataDocumentCache>,
}

impl GitHubBootstrapCacheStore {
    pub fn public(owner: impl Into<String>, root: impl Into<PathBuf>) -> Result<Self, CacheError> {
        Self::new(owner, root, CacheVisibility::Public, false)
    }

    pub fn public_required(
        owner: impl Into<String>,
        root: impl Into<PathBuf>,
    ) -> Result<Self, CacheError> {
        Self::new(owner, root, CacheVisibility::Public, true)
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
            verified_transports: Mutex::new(HashMap::new()),
            discovered_keys: Mutex::new(HashMap::new()),
            continuation_inventory_lock: Mutex::new(()),
            registry_snapshot: Mutex::new(None),
            family_topologies: Mutex::new(HashMap::new()),
            shard_revisions: Mutex::new(HashMap::new()),
            historical_batches: Mutex::new(HashMap::new()),
            metadata_documents: Mutex::new(MetadataDocumentCache::default()),
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
            || registry.families.is_empty()
            || registry.families.iter().any(|route| {
                route.family.trim().is_empty()
                    || route.current_writable_shard.trim().is_empty()
                    || route.metadata.trim().is_empty()
            })
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

    fn family_topology(
        &self,
        family: &str,
    ) -> Result<crate::bootstrap_topology::BootstrapFamilyTopology, CacheError> {
        if let Some(topology) = self
            .family_topologies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(family)
            .cloned()
        {
            return Ok(topology);
        }
        let topology = crate::bootstrap_topology::read_bootstrap_family_topology(
            &self.remote,
            &self.owner,
            self.visibility,
            family,
            &CancellationToken::new(),
        )?;
        self.family_topologies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(family.to_owned(), topology.clone());
        Ok(topology)
    }

    fn read_json<T: serde::de::DeserializeOwned>(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
    ) -> Result<(T, RemoteReadReport), CacheError> {
        const MAXIMUM_RETAINED_METADATA_BYTES: u64 = 64 * 1024 * 1024;
        let cache_key = (repository.to_owned(), revision.to_owned(), path.to_owned());
        let cached = {
            let cache = self
                .metadata_documents
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.documents.get(&cache_key).cloned()
        };
        if let Some((bytes, report)) = cached {
            return Ok((serde_json::from_slice(bytes.as_ref())?, report));
        }
        let mut bytes = Vec::new();
        let report = self.remote.read_committed_path(
            repository,
            revision,
            path,
            16 * 1024 * 1024,
            &CancellationToken::new(),
            &mut bytes,
        )?;
        let value = serde_json::from_slice(&bytes)?;
        let retained_bytes = bytes.len() as u64;
        let mut cache = self
            .metadata_documents
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !cache.documents.contains_key(&cache_key)
            && cache.retained_bytes.saturating_add(retained_bytes)
                <= MAXIMUM_RETAINED_METADATA_BYTES
        {
            cache.retained_bytes = cache.retained_bytes.saturating_add(retained_bytes);
            cache
                .documents
                .insert(cache_key, (Arc::from(bytes), report.clone()));
        }
        Ok((value, report))
    }

    /// Read and validate the immutable publication-batch inventory once for a
    /// particular shard revision. Historical dependency identities cannot be
    /// recovered from the active semantic index after a newer artifact has
    /// superseded them, but their canonical manifests and publication proofs
    /// remain committed in the shard.
    fn historical_batches(
        &self,
        repository: &str,
        revision: &str,
    ) -> Result<HistoricalBatchInventory, CacheError> {
        let cache_key = (repository.to_owned(), revision.to_owned());
        if let Some(batches) = self
            .historical_batches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&cache_key)
            .cloned()
        {
            return Ok(batches);
        }

        let visibility_name = visibility_name(self.visibility);
        let listing = self.remote.list_committed_paths(
            repository,
            revision,
            "transactions/batches",
            100_000,
            16 * 1024 * 1024,
            &CancellationToken::new(),
        )?;
        let suffix = format!("/{visibility_name}.json");
        let mut batches = Vec::new();
        let mut total_batch_bytes = 0u64;
        for path in listing.paths.iter().filter(|path| path.ends_with(&suffix)) {
            let (batch, source): (RepositoryPublicationBatch, _) =
                self.read_json(repository, revision, path)?;
            total_batch_bytes = total_batch_bytes
                .checked_add(source.size_bytes)
                .ok_or_else(|| {
                    CacheError::ResourceLimit(
                        "historical repository-batch bytes exceed u64".to_owned(),
                    )
                })?;
            if total_batch_bytes > 256 * 1024 * 1024 {
                return Err(CacheError::ResourceLimit(
                    "historical repository-batch inventory exceeds 256 MiB".to_owned(),
                ));
            }
            batch.validate()?;
            if source.repository_path != *path || source.content_digest != batch.digest()? {
                return Err(CacheError::InvalidManifest(format!(
                    "historical repository batch {path:?} is not canonical"
                )));
            }
            batches.push((batch, source));
        }
        if batches.is_empty() {
            return Err(CacheError::InvalidManifest(format!(
                "shard {repository} has no {visibility_name} publication batches"
            )));
        }
        self.historical_batches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, batches.clone());
        Ok(batches)
    }

    fn resolve(&self, key: &ArtifactKey) -> Result<Option<ResolvedRemoteArtifact>, CacheError> {
        if self.visibility == CacheVisibility::Public && artifact_kind_is_private_only(&key.kind) {
            return Ok(None);
        }
        let Some(family) = family_for_artifact_kind(&key.kind) else {
            return Ok(None);
        };
        let topology = self.family_topology(family)?;
        let prefix = &key.parameters_digest.0[..2];
        let index_path = format!("indexes/{family}/{prefix}.json");
        let current = crate::current_toolkit_version()?;
        for shard in &topology.readable_shards {
            let repository = shard.repository_url.clone();
            let revision = self.shard_revision(&repository)?;
            let (index, index_source): (ShardIndexPartition, _) =
                match self.read_json(&repository, &revision, &index_path) {
                    Ok(value) => value,
                    Err(CacheError::NotFound(_)) => continue,
                    Err(error) => return Err(error),
                };
            index.validate()?;
            if index.family != family {
                return Err(CacheError::InvalidManifest(format!(
                    "shard index {index_path:?} belongs to {:?}, not {family:?}",
                    index.family
                )));
            }
            let entries_for_identity = index.lookup(&key.parameters_digest).collect::<Vec<_>>();
            let Some(entry) = entries_for_identity
                .iter()
                .copied()
                .filter(|entry| entry.disposition == ArtifactDisposition::Active)
                .filter(|entry| entry.achieved_assurance.mathematical().is_some())
                .filter(|entry| entry.minimum_reader_version <= current)
                .max_by_key(|entry| (entry.achieved_assurance, entry.manifest_digest.clone()))
                .cloned()
            else {
                // Once a newer shard names this semantic identity, its live
                // policy disposition shadows older shards. Falling back to a
                // predecessor here could silently undo a revocation, reader
                // floor, or assurance downgrade recorded during rollover.
                if !entries_for_identity.is_empty() {
                    return Ok(None);
                }
                continue;
            };
            return self.materialize_indexed_entry(
                repository,
                revision,
                family,
                shard,
                &key.parameters_digest,
                entry,
                index_source,
            );
        }
        Ok(None)
    }

    /// Resolve an artifact by its exact published dependency identity.
    ///
    /// The lookup mirrors [`Self::resolve`]: the family route comes from the
    /// registry snapshot and the shard index partition is addressed by the
    /// identity's semantic digest. Where key-based resolution selects the best
    /// admissible entry, this selects the one whose canonical manifest digest
    /// equals the identity's, then verifies the fetched manifest against every
    /// identity field.
    fn resolve_identity(
        &self,
        identity: &crate::PayloadDependencyIdentity,
    ) -> Result<Option<ResolvedRemoteArtifact>, CacheError> {
        identity.validate()?;
        let family = identity.artifact_family.as_str();
        let topology = self.family_topology(family)?;
        let Some(prefix) = identity.semantic_digest.0.get(..2) else {
            return Err(CacheError::InvalidManifest(
                "dependency identity semantic digest is too short to address an index".to_owned(),
            ));
        };
        let index_path = format!("indexes/{family}/{prefix}.json");
        let current = crate::current_toolkit_version()?;
        for shard in &topology.readable_shards {
            let repository = shard.repository_url.clone();
            let revision = self.shard_revision(&repository)?;
            let indexed =
                match self.read_json::<ShardIndexPartition>(&repository, &revision, &index_path) {
                    Ok((index, source)) => {
                        index.validate()?;
                        if index.family != family {
                            return Err(CacheError::InvalidManifest(format!(
                                "shard index {index_path:?} belongs to the wrong family"
                            )));
                        }
                        Some((index, source))
                    }
                    Err(CacheError::NotFound(_)) => None,
                    Err(error) => return Err(error),
                };
            let active_entry = indexed.as_ref().and_then(|(index, source)| {
                index
                    .lookup(&identity.semantic_digest)
                    .find(|entry| {
                        entry.disposition == ArtifactDisposition::Active
                            && entry.manifest_digest == identity.manifest_digest
                            && entry.minimum_reader_version <= current
                    })
                    .cloned()
                    .map(|entry| (entry, source.clone()))
            });
            let (entry, index_source) = if let Some(active_entry) = active_entry {
                active_entry
            } else {
                let manifest_path =
                    bootstrap_manifest_path(&identity.semantic_digest, &identity.manifest_digest);
                let (manifest, manifest_source): (CanonicalArtifactManifest, _) =
                    match self.read_json(&repository, &revision, &manifest_path) {
                        Ok(value) => value,
                        Err(CacheError::NotFound(_)) => continue,
                        Err(error) => return Err(error),
                    };
                manifest.validate()?;
                if manifest_source.content_digest != identity.manifest_digest
                    || manifest.digest()? != identity.manifest_digest
                    || manifest.artifact_family != family
                    || manifest.semantic_digest != identity.semantic_digest
                    || manifest.payload_digest != identity.payload_digest
                    || manifest.minimum_reader_version > current
                {
                    return Err(CacheError::InvalidManifest(format!(
                    "historical manifest for {family}/{} does not reproduce the requested dependency identity",
                    identity.semantic_digest.0
                )));
                }
                let expected_destination = match self.visibility {
                    CacheVisibility::Private => PublicationDestination::Private,
                    CacheVisibility::Public => PublicationDestination::Public,
                    _ => unreachable!("GitHub bootstrap stores are private or public"),
                };
                let batches = self.historical_batches(&repository, &revision)?;
                let Some((batch, batch_source, artifact)) = historical_publication_for_identity(
                    &batches,
                    expected_destination,
                    family,
                    &shard.authorized_repository,
                    identity,
                    &manifest_path,
                ) else {
                    return Err(CacheError::InvalidManifest(format!(
                    "historical manifest for {family}/{} has no canonical publication-batch proof",
                    identity.semantic_digest.0
                )));
                };
                (
                    ShardIndexEntry {
                        semantic_digest: identity.semantic_digest.clone(),
                        canonical_payload_digest: identity.payload_digest.clone(),
                        manifest_digest: identity.manifest_digest.clone(),
                        achieved_assurance: artifact.achieved_assurance,
                        disposition: ArtifactDisposition::Active,
                        producer_toolkit_version: artifact.producer_toolkit_version.clone(),
                        minimum_reader_version: manifest.minimum_reader_version,
                        transport_digests: vec![artifact.transport_digest.clone()],
                        publication_transaction_id: batch.batch_id.0.clone(),
                    },
                    batch_source.clone(),
                )
            };
            let resolved = self.materialize_indexed_entry(
                repository,
                revision,
                family,
                shard,
                &identity.semantic_digest,
                entry,
                index_source,
            )?;
            let Some(resolved) = resolved else {
                continue;
            };
            if resolved.manifest.semantic_digest != identity.semantic_digest
                || resolved.manifest.digest()? != identity.manifest_digest
                || resolved.manifest.payload_digest != identity.payload_digest
            {
                return Err(CacheError::InvalidManifest(format!(
                    "shard entry for {}/{} does not reproduce the requested dependency identity",
                    family, identity.semantic_digest.0
                )));
            }
            return Ok(Some(resolved));
        }
        Ok(None)
    }

    /// Fetch, validate, and assemble one indexed shard entry. Shared by the
    /// key-based and identity-based resolution paths; every payload, manifest,
    /// and receipt identity check is common to both.
    #[allow(clippy::too_many_arguments)]
    fn materialize_indexed_entry(
        &self,
        repository: String,
        revision: String,
        family: &str,
        shard: &crate::bootstrap_topology::BootstrapShard,
        semantic_digest: &ContentDigest,
        entry: ShardIndexEntry,
        index_source: RemoteReadReport,
    ) -> Result<Option<ResolvedRemoteArtifact>, CacheError> {
        let visibility_name = visibility_name(self.visibility);
        let manifest_path = bootstrap_manifest_path(semantic_digest, &entry.manifest_digest);
        let (manifest, manifest_source): (CanonicalArtifactManifest, _) =
            self.read_json(&repository, &revision, &manifest_path)?;
        manifest.validate()?;
        // Identity-based closure resolution does not know the artifact kind
        // until the canonical manifest is opened. Apply the same hard public
        // boundary here as the key-based path, including for historical shard
        // entries that predate the restriction.
        if self.visibility == CacheVisibility::Public
            && artifact_kind_is_private_only(&manifest.semantic_key.artifact_kind)
        {
            return Ok(None);
        }
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
            &shard.authorized_repository,
            semantic_digest,
            &entry,
            &manifest_path,
            transport_digest,
        )?;
        let resolved = ResolvedRemoteArtifact {
            family: family.to_owned(),
            semantic_digest: semantic_digest.clone(),
            overlay: self.name.clone(),
            visibility: self.visibility,
            shard_id: shard.shard_id.clone(),
            authorized_repository: shard.authorized_repository.clone(),
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
        Ok(Some(resolved))
    }

    fn local_continuation_inventory_path(
        &self,
        query: &CcmEigenpairContinuationQuery,
    ) -> Result<PathBuf, CacheError> {
        Ok(query
            .repository_path()?
            .split('/')
            .fold(self.root.join("derived"), |path, component| {
                path.join(component)
            }))
    }

    fn record_local_continuation_entry(
        &self,
        logical_key: &str,
        manifest: &CanonicalArtifactManifest,
        manifest_digest: ContentDigest,
        assurance: ArtifactAssuranceState,
        disposition: ArtifactDisposition,
    ) -> Result<(), CacheError> {
        let Some((query, addition)) = ccm_eigenpair_continuation_entry(
            &manifest.semantic_key,
            logical_key,
            manifest_digest,
            assurance,
            disposition,
            manifest.producer_toolkit_version.clone(),
            manifest.minimum_reader_version.clone(),
        )?
        else {
            return Ok(());
        };
        let path = self.local_continuation_inventory_path(&query)?;
        let _guard = self
            .continuation_inventory_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries = if path.exists() {
            let existing: CcmEigenpairContinuationIndex =
                serde_json::from_slice(&fs::read(&path)?)?;
            existing.validate()?;
            existing.entries
        } else {
            Vec::new()
        };
        entries.retain(|entry| entry.semantic_digest != addition.semantic_digest);
        entries.push(addition);
        let inventory = CcmEigenpairContinuationIndex::rebuild(&query, entries)?;
        crate::atomic_replace(&path, &serde_json::to_vec_pretty(&inventory)?)
    }
}

fn historical_publication_for_identity<'a>(
    batches: &'a [(RepositoryPublicationBatch, RemoteReadReport)],
    expected_destination: PublicationDestination,
    family: &str,
    authorized_repository: &str,
    identity: &crate::PayloadDependencyIdentity,
    manifest_path: &str,
) -> Option<(
    &'a RepositoryPublicationBatch,
    &'a RemoteReadReport,
    &'a RepositoryBatchArtifact,
)> {
    batches.iter().find_map(|(batch, source)| {
        if batch.destination != expected_destination
            || batch.family != family
            || batch.authorized_repository != authorized_repository
            || batch.branch != "main"
        {
            return None;
        }
        batch
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.semantic_digest == identity.semantic_digest
                    && artifact.manifest_digest == identity.manifest_digest
                    && artifact.canonical_payload_digest == identity.payload_digest
                    && artifact.manifest_path == manifest_path
            })
            .map(|artifact| (batch, source, artifact))
    })
}

fn bootstrap_manifest_path(
    semantic_digest: &ContentDigest,
    manifest_digest: &ContentDigest,
) -> String {
    format!(
        "manifests/{}/{}.json",
        &semantic_digest.0[..2],
        manifest_digest.0
    )
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

impl GitHubBootstrapCacheStore {
    /// Build the local adapter manifest for a resolved shard artifact.
    ///
    /// Shared by key-based and identity-based candidate construction. The
    /// full canonical manifest rides along in a tag so publication staging
    /// and closure walks can recover identity dependencies; the adapter's own
    /// key-based dependency list is deliberately empty because canonical
    /// manifests reference dependencies by identity, not by key.
    fn adapter_manifest_for(
        &self,
        key: ArtifactKey,
        resolved: &ResolvedRemoteArtifact,
    ) -> Result<ArtifactManifest, CacheError> {
        let ordered_items = &resolved.manifest.canonical_payload.ordered_items;
        if ordered_items.len() != 1 || ordered_items[0].normalized_path != "payload.json" {
            return Err(CacheError::InvalidManifest(
                "managed JSON artifact must contain exactly one payload.json item".to_owned(),
            ));
        }
        let item = &ordered_items[0];
        let quality = match resolved.index.achieved_assurance {
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
        Ok(ArtifactManifest {
            schema_version: 1,
            key,
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
        })
    }

    /// The key under which resolved remote state is retained: the canonical
    /// manifest digest that every adapter carries as its provenance. The
    /// logical payload digest is not sufficient, because the schema permits
    /// identical payload bytes under different dependency closures, and
    /// those are different artifacts with different transports.
    fn resolved_key(manifest: &ArtifactManifest) -> Result<ContentDigest, CacheError> {
        manifest.provenance_digest.clone().ok_or_else(|| {
            CacheError::InvalidManifest(format!(
                "remote adapter manifest for {} / {} carries no canonical manifest digest",
                manifest.key.kind, manifest.key.logical_key
            ))
        })
    }

    fn remember_resolved(
        &self,
        adapter_manifest: &ArtifactManifest,
        resolved: ResolvedRemoteArtifact,
    ) -> Result<(), CacheError> {
        let key = Self::resolved_key(adapter_manifest)?;
        if key != resolved.manifest.digest()? {
            return Err(CacheError::InvalidManifest(
                "remote adapter manifest does not name the resolved canonical manifest".to_owned(),
            ));
        }
        self.resolved
            .lock()
            .map_err(|_| CacheError::Io("public cache lock poisoned".to_owned()))?
            .insert(key, resolved);
        Ok(())
    }
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
    fn identity_candidates(
        &self,
        identity: &crate::PayloadDependencyIdentity,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        // Same degradation contract as key-based candidates: an optional
        // remote that is offline or invalid yields no candidates rather than
        // failing the run.
        let resolved = if self.required {
            self.resolve_identity(identity)?
        } else {
            self.resolve_identity(identity).unwrap_or(None)
        };
        let Some(resolved) = resolved else {
            return Ok(Vec::new());
        };
        // Canonical manifests carry no logical key, and none is recoverable
        // from an identity-addressed shard. The synthetic key is local-only
        // provenance: publication addresses artifacts by identity, so it never
        // reaches a destination path.
        let key = ArtifactKey {
            kind: resolved.manifest.semantic_key.artifact_kind.clone(),
            logical_key: format!(
                "closure/{}",
                identity
                    .semantic_digest
                    .0
                    .get(..16)
                    .unwrap_or(&identity.semantic_digest.0)
            ),
            parameters_digest: identity.semantic_digest.clone(),
        };
        let adapter_manifest = self.adapter_manifest_for(key, &resolved)?;
        self.remember_resolved(&adapter_manifest, resolved)?;
        Ok(vec![adapter_manifest])
    }

    fn prefetch_manifests(&self, manifests: &[ArtifactManifest]) -> Result<(), CacheError> {
        if manifests.is_empty() {
            return Ok(());
        }
        let resolved = self
            .resolved
            .lock()
            .map_err(|_| CacheError::Io("public cache lock poisoned".to_owned()))?;
        let mut batches = BTreeMap::<(String, String), BTreeMap<String, u64>>::new();
        let mut artifact_count = 0usize;
        for manifest in manifests {
            let key = Self::resolved_key(manifest)?;
            let artifact = resolved
                .get(&key)
                .ok_or_else(|| CacheError::NotFound(key.to_string()))?;
            let missing_paths = missing_local_transport_paths(&self.root, &artifact.encoding)?;
            if missing_paths.is_empty() {
                continue;
            }
            artifact_count = artifact_count.saturating_add(1);
            let paths = batches
                .entry((artifact.repository.clone(), artifact.revision.clone()))
                .or_default();
            for part in missing_paths {
                paths
                    .entry(part.repository_path)
                    .and_modify(|maximum| *maximum = (*maximum).max(part.maximum_bytes))
                    .or_insert(part.maximum_bytes);
            }
        }
        drop(resolved);
        let cancellation = CancellationToken::new();
        let repository_count = batches.len();
        let prepared_path_count = batches.values().map(BTreeMap::len).sum::<usize>();
        let prepared_bytes = batches
            .values()
            .flat_map(BTreeMap::values)
            .fold(0u64, |total, bytes| total.saturating_add(*bytes));
        let batches = batches
            .into_iter()
            .map(|((repository, revision), paths)| {
                let paths = paths
                    .into_iter()
                    .map(|(repository_path, maximum_bytes)| RemotePathPrefetch {
                        repository_path,
                        maximum_bytes,
                    })
                    .collect::<Vec<_>>();
                (repository, revision, paths)
            })
            .collect::<Vec<_>>();
        let concurrency = std::env::var("XC_CACHE_PREFETCH_CONCURRENCY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 8);
        let started = Instant::now();
        let performance = xc_core::performance_stage_with("cache.remote.prefetch", || {
            xc_core::PerformanceStageMetadata {
                operation: Some("dependency-closure".to_owned()),
                cache_disposition: Some("transport-prefetch".to_owned()),
                scheduling: Some(format!(
                    "bounded-repository-batches;workers={concurrency};repositories={repository_count};paths={prepared_path_count}"
                )),
                ..xc_core::PerformanceStageMetadata::default()
            }
        });
        for batch in batches.chunks(concurrency) {
            std::thread::scope(|scope| {
                let workers = batch
                    .iter()
                    .map(|(repository, revision, paths)| {
                        let cancellation = &cancellation;
                        scope.spawn(move || {
                            self.remote.prefetch_committed_paths(
                                repository,
                                revision,
                                paths,
                                cancellation,
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                for worker in workers {
                    worker.join().map_err(|_| {
                        CacheError::Io("cache prefetch worker panicked".to_owned())
                    })??;
                }
                Ok::<(), CacheError>(())
            })?;
        }
        drop(performance);
        if prepared_bytes >= 64 * 1024 * 1024 {
            eprintln!(
                "  cache prefetch: {} artifacts, {:.1} MB, {} immutable paths across {} repositories in {:.3}s (workers={})",
                artifact_count,
                prepared_bytes as f64 / 1_000_000.0,
                prepared_path_count,
                repository_count,
                started.elapsed().as_secs_f64(),
                concurrency
            );
        }
        Ok(())
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
        let adapter_manifest = self.adapter_manifest_for(key.clone(), &resolved)?;
        let assurance = resolved.index.achieved_assurance;
        self.record_local_continuation_entry(
            &key.logical_key,
            &resolved.manifest,
            adapter_manifest
                .provenance_digest
                .clone()
                .expect("remote adapter records canonical manifest identity"),
            assurance,
            resolved.index.disposition,
        )?;
        self.remember_resolved(&adapter_manifest, resolved)?;
        Ok(vec![adapter_manifest])
    }

    fn matching_keys(
        &self,
        _kind: &str,
        _logical_key_prefix: &str,
        _maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        // Hash-partitioned shard indexes do not contain logical keys. A
        // generic prefix search would therefore require opening every
        // canonical manifest and is intentionally unsupported for remote
        // stores. Purpose-built secondary indexes provide bounded discovery.
        Ok(Vec::new())
    }

    fn ccm_eigenpair_continuation_keys(
        &self,
        query: &CcmEigenpairContinuationQuery,
        maximum_keys: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        if maximum_keys == 0 {
            return Ok(Vec::new());
        }
        query.validate()?;
        let family = family_for_artifact_kind(CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND)
            .expect("CCM eigenpairs have a managed shard family");
        let inventory_path = query.repository_path()?;
        let cache_identity = (
            CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND.to_owned(),
            format!("{inventory_path}#lt-{}", query.maximum_n_modes),
        );
        if let Some(cached) = self
            .discovered_keys
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&cache_identity)
            .cloned()
        {
            return Ok(cached.into_iter().take(maximum_keys).collect());
        }
        let result = (|| {
            let topology = self.family_topology(family)?;
            let current = crate::current_toolkit_version()?;
            let mut entries =
                if let Ok(bytes) = fs::read(self.local_continuation_inventory_path(query)?) {
                    let inventory: CcmEigenpairContinuationIndex = serde_json::from_slice(&bytes)?;
                    inventory.validate()?;
                    inventory.entries
                } else {
                    Vec::new()
                };
            // Historical shards are merged oldest-to-newest so a current
            // writable shard wins if the same semantic identity was carried
            // forward during rollover.
            for shard in topology.readable_shards.iter().rev() {
                let revision = self.shard_revision(&shard.repository_url)?;
                match self.read_json::<CcmEigenpairContinuationIndex>(
                    &shard.repository_url,
                    &revision,
                    &inventory_path,
                ) {
                    Ok((inventory, _)) => {
                        inventory.validate()?;
                        for entry in inventory.entries {
                            entries
                                .retain(|current| current.semantic_digest != entry.semantic_digest);
                            entries.push(entry);
                        }
                    }
                    Err(CacheError::NotFound(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            let inventory = CcmEigenpairContinuationIndex::rebuild(query, entries)?;
            let keys = inventory.query(query, &current, maximum_keys)?;
            self.discovered_keys
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(cache_identity, keys.clone());
            Ok(keys)
        })();
        if self.required {
            result
        } else {
            Ok(result.unwrap_or_default())
        }
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
            .get(&Self::resolved_key(manifest)?)
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
        {
            // A reconstruction verified every part in this process. A package
            // reuse did not touch the parts, so they are offered unverified
            // and staging hashes them before linking.
            let identity = Self::resolved_key(manifest)?;
            let parts_verified = report.part_fetch.is_some();
            let mut verified_transports = self
                .verified_transports
                .lock()
                .map_err(|_| CacheError::Io("verified transport lock poisoned".to_owned()))?;
            let already_verified = verified_transports
                .get(&identity)
                .is_some_and(|existing| existing.parts_verified);
            if parts_verified || !already_verified {
                verified_transports.insert(
                    identity,
                    VerifiedTransportParts {
                        encoding: resolved.encoding.clone(),
                        parts_root: self.root.join("parts"),
                        parts_verified,
                    },
                );
            }
        }
        if resolved.encoding.package_size_bytes >= 64 * 1024 * 1024 {
            eprintln!(
                "  cache transport: {} {:.1} MB, {} parts ({} downloaded, {} reused), fetch {:.3}s, reconstruct {:.3}s, verify/decode {:.3}s, total {:.3}s",
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
                report.part_fetch_elapsed_millis as f64 / 1_000.0,
                report.package_reconstruction_elapsed_millis as f64 / 1_000.0,
                report.payload_verification_elapsed_millis as f64 / 1_000.0,
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
            .get(&Self::resolved_key(manifest)?)
            .cloned()
            .ok_or_else(|| CacheError::NotFound(manifest.content_digest.to_string()))?;
        let path = self
            .root
            .join("packages")
            .join(format!("{}.zip", resolved.encoding.digest()?.0));
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() != resolved.encoding.package_size_bytes
        {
            return Err(CacheError::InvalidManifest(format!(
                "retained package has invalid filesystem metadata: {}",
                path.display()
            )));
        }
        Ok(Some(VerifiedEncodedPayload {
            path,
            encoding: "zip-json-entry-v1".to_owned(),
            encoder_profile: Some(resolved.encoding.encoder_profile.clone()),
            content_digest: resolved.encoding.package_digest,
            size_bytes: resolved.encoding.package_size_bytes,
        }))
    }

    fn verified_transport_parts(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<Option<VerifiedTransportParts>, CacheError> {
        let identity = Self::resolved_key(manifest)?;
        if let Some(verified) = self
            .verified_transports
            .lock()
            .map_err(|_| CacheError::Io("verified transport lock poisoned".to_owned()))?
            .get(&identity)
        {
            return Ok(Some(verified.clone()));
        }
        // No materialization ran for this manifest in this process (for
        // example a metadata-only closure resolution). Offer the retained
        // part store unverified; staging hashes each part before linking and
        // falls back to the verified package when a part is absent or wrong.
        // The entry is keyed by the exact canonical manifest, so two
        // artifacts with identical payload bytes and different closures can
        // never exchange encodings here.
        Ok(self
            .resolved
            .lock()
            .map_err(|_| CacheError::Io("public cache lock poisoned".to_owned()))?
            .get(&identity)
            .map(|resolved| VerifiedTransportParts {
                encoding: resolved.encoding.clone(),
                parts_root: self.root.join("parts"),
                parts_verified: false,
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

    /// A resolved remote artifact whose payload bytes are fixed and whose
    /// canonical closure is the caller's. Two calls with different closures
    /// therefore describe two different published artifacts that share every
    /// payload byte, every ZIP byte, and every split part, but not their
    /// canonical manifests or transport encodings.
    fn resolved_fixture(
        root: &std::path::Path,
        label: &str,
        dependencies: Vec<PayloadDependencyIdentity>,
    ) -> ResolvedRemoteArtifact {
        let payload = br#"{"same":"bytes under two closures"}"#;
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "json".to_owned(),
            precision_bits: None,
            scalar_representation: "json".to_owned(),
            dimensions: vec![payload.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "payload.json".to_owned(),
                content_digest: ContentDigest::sha256(payload),
                size_bytes: payload.len() as u64,
            }],
            dependencies,
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let package_path = root.join(format!("{label}.zip"));
        package_canonical_payload_bytes_zip64(
            &canonical_payload,
            "payload.json",
            payload,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let encoded = fs::read(&package_path).unwrap();
        let encoding = stream_split_encoded(
            &mut encoded.as_slice(),
            payload_digest.clone(),
            DETERMINISTIC_ZIP64_PROFILE_V1,
            &TransportPolicy {
                maximum_file_bytes_exclusive: 1_000,
                split_part_bytes: 32,
                maximum_batch_payload_bytes: 1_000,
                maximum_pending_batches: 1,
            },
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            |_, _| Ok(()),
        )
        .unwrap();
        let transport_digest = encoding.digest().unwrap();
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_tau_matrix".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({"case": "same-bytes"}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm-matrices".to_owned(),
            semantic_key,
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![transport_digest.clone()],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.14.1").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.14.1").unwrap(),
            maximum_reader_version: None,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            claim_scope: "same-bytes fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let transaction_id = ContentDigest::sha256(label.as_bytes()).0;
        let index = ShardIndexEntry {
            semantic_digest: semantic_digest.clone(),
            canonical_payload_digest: payload_digest.clone(),
            manifest_digest: manifest_digest.clone(),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: ToolkitVersion::parse("0.14.1").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.14.1").unwrap(),
            transport_digests: vec![transport_digest.clone()],
            publication_transaction_id: transaction_id.clone(),
        };
        let receipt = PublicationReceipt {
            schema_version: 1,
            transaction_id: transaction_id.clone(),
            idempotency_key: ContentDigest(transaction_id),
            destination: PublicationDestination::Public,
            principal: "fixture".to_owned(),
            authorized_repository: "team/shard".to_owned(),
            repository_permission_evidence_digest: ContentDigest::sha256(b"permission"),
            shard_id: "fixture-001".to_owned(),
            branch: "main".to_owned(),
            semantic_digest: semantic_digest.clone(),
            canonical_payload_digest: payload_digest,
            manifest_digest: manifest_digest.clone(),
            transport_digest: transport_digest.clone(),
            policy_digest: ContentDigest::sha256(b"policy"),
            policy_id: "fixture-owner-policy".to_owned(),
            authority_mode: xc_core::PublicationAuthorityMode::OwnerDirect,
            validation_evidence_digests: vec![ContentDigest::sha256(b"validation")],
            contributor_authorization_digest: None,
            reviewer_approvals: Vec::new(),
            payload_commit_ids: vec!["payload-commit".to_owned()],
            payload_batch_record_commit_ids: Vec::new(),
            payload_batch_record_digests: BTreeMap::new(),
            metadata_commit_id: "metadata-commit".to_owned(),
            metadata_file_digests: BTreeMap::from([(
                "manifests/fixture.json".to_owned(),
                manifest_digest,
            )]),
            discoverability_subject_digests: BTreeMap::from([(
                "indexes/fixture/00.json".to_owned(),
                ContentDigest::sha256(b"index"),
            )]),
            remote_verification_results: vec![RemoteCommitVerificationResult {
                phase: "immutable_metadata".to_owned(),
                sequence: 0,
                commit_id: "metadata-commit".to_owned(),
                verified: true,
                content_digests: vec![transport_digest],
            }],
            verified_at_unix_seconds: 1,
        };
        let source = RemoteReadReport {
            repository_path: "fixture.json".to_owned(),
            revision: "a".repeat(40),
            size_bytes: 1,
            content_digest: ContentDigest::sha256(b"fixture"),
        };
        ResolvedRemoteArtifact {
            family: "ccm-matrices".to_owned(),
            semantic_digest,
            overlay: "public".to_owned(),
            visibility: CacheVisibility::Public,
            shard_id: "fixture-001".to_owned(),
            authorized_repository: "team/shard".to_owned(),
            repository: "team/shard".to_owned(),
            revision: source.revision.clone(),
            index,
            manifest,
            encoding,
            receipt: RemotePublicationEvidence::ArtifactReceipt(Box::new(receipt)),
            index_source: source.clone(),
            manifest_source: source.clone(),
            encoding_source: source.clone(),
            receipt_source: source,
            dependencies: Vec::new(),
        }
    }

    #[test]
    fn same_byte_artifacts_with_different_closures_keep_their_own_transports() {
        let root = std::env::temp_dir().join(format!(
            "xc-bootstrap-same-bytes-identity-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let store = GitHubBootstrapCacheStore::public("fixture-owner", root.join("store")).unwrap();

        let artifact_a = resolved_fixture(&root, "a", Vec::new());
        let artifact_b = resolved_fixture(
            &root,
            "b",
            vec![PayloadDependencyIdentity {
                artifact_family: "ccm-matrices".to_owned(),
                semantic_digest: ContentDigest::sha256(b"dependency semantic"),
                manifest_digest: ContentDigest::sha256(b"dependency manifest"),
                payload_digest: ContentDigest::sha256(b"dependency payload"),
            }],
        );
        assert_eq!(
            artifact_a.manifest.canonical_payload.ordered_items,
            artifact_b.manifest.canonical_payload.ordered_items,
            "identical payload bytes"
        );
        assert_eq!(
            artifact_a.encoding.ordered_parts, artifact_b.encoding.ordered_parts,
            "identical split parts"
        );
        assert_ne!(
            artifact_a.manifest.digest().unwrap(),
            artifact_b.manifest.digest().unwrap(),
            "different canonical manifests"
        );
        assert_ne!(
            artifact_a.encoding, artifact_b.encoding,
            "different transport encodings"
        );

        let key = ArtifactKey {
            kind: "ccm_tau_matrix".to_owned(),
            logical_key: "closure/same-bytes".to_owned(),
            parameters_digest: artifact_a.semantic_digest.clone(),
        };
        let adapter_a = store
            .adapter_manifest_for(key.clone(), &artifact_a)
            .unwrap();
        let adapter_b = store.adapter_manifest_for(key, &artifact_b).unwrap();
        assert_eq!(
            adapter_a.content_digest, adapter_b.content_digest,
            "the logical payload digest cannot tell them apart"
        );
        assert_ne!(adapter_a.provenance_digest, adapter_b.provenance_digest);

        // Resolve A, then B, then ask for A's transport parts.
        store
            .remember_resolved(&adapter_a, artifact_a.clone())
            .unwrap();
        store
            .remember_resolved(&adapter_b, artifact_b.clone())
            .unwrap();
        let parts_a = store.verified_transport_parts(&adapter_a).unwrap().unwrap();
        assert_eq!(parts_a.encoding, artifact_a.encoding);
        let parts_b = store.verified_transport_parts(&adapter_b).unwrap().unwrap();
        assert_eq!(parts_b.encoding, artifact_b.encoding);
        // The encoded descriptor is addressed the same way; A's package is
        // the one named by A's encoding.
        let package_a = store
            .root
            .join("packages")
            .join(format!("{}.zip", artifact_a.encoding.digest().unwrap().0));
        fs::create_dir_all(package_a.parent().unwrap()).unwrap();
        fs::write(
            &package_a,
            vec![0u8; artifact_a.encoding.package_size_bytes as usize],
        )
        .unwrap();
        assert_eq!(
            store
                .verified_encoded_payload(&adapter_a)
                .unwrap()
                .unwrap()
                .content_digest,
            artifact_a.encoding.package_digest
        );
        assert!(
            store
                .verified_encoded_payload(&adapter_b)
                .unwrap()
                .is_none(),
            "B's package was never materialized, so B has no encoded descriptor"
        );

        // An adapter that carries no canonical manifest digest is refused
        // rather than answered from the payload digest.
        let mut anonymous = adapter_a.clone();
        anonymous.provenance_digest = None;
        assert!(matches!(
            store.verified_transport_parts(&anonymous),
            Err(CacheError::InvalidManifest(_))
        ));
        assert!(matches!(
            store.remember_resolved(&anonymous, artifact_a.clone()),
            Err(CacheError::InvalidManifest(_))
        ));

        // Concurrent variant: writers keep re-resolving A and B in
        // alternation while readers ask for each one's parts. Every answer
        // must belong to the artifact that was asked about.
        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| {
                    for _ in 0..200 {
                        store
                            .remember_resolved(&adapter_a, artifact_a.clone())
                            .unwrap();
                        store
                            .remember_resolved(&adapter_b, artifact_b.clone())
                            .unwrap();
                    }
                });
                scope.spawn(|| {
                    for _ in 0..200 {
                        assert_eq!(
                            store
                                .verified_transport_parts(&adapter_a)
                                .unwrap()
                                .unwrap()
                                .encoding,
                            artifact_a.encoding
                        );
                        assert_eq!(
                            store
                                .verified_transport_parts(&adapter_b)
                                .unwrap()
                                .unwrap()
                                .encoding,
                            artifact_b.encoding
                        );
                    }
                });
            }
        });
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dependency_prefetch_skips_locally_available_packages_and_parts() {
        let root = std::env::temp_dir().join(format!(
            "xc-bootstrap-local-prefetch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let package_bytes = b"abcdef";
        let encoding = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: ContentDigest::sha256(b"canonical payload"),
            encoder_profile: DETERMINISTIC_ZIP64_PROFILE_V1.to_owned(),
            package_size_bytes: package_bytes.len() as u64,
            package_digest: ContentDigest::sha256(package_bytes),
            ordered_parts: vec![
                TransportPart {
                    sequence: 0,
                    repository_path: "objects/aa/part-00000".to_owned(),
                    size_bytes: 3,
                    content_digest: ContentDigest::sha256(b"abc"),
                },
                TransportPart {
                    sequence: 1,
                    repository_path: "objects/bb/part-00001".to_owned(),
                    size_bytes: 3,
                    content_digest: ContentDigest::sha256(b"def"),
                },
            ],
            reconstruction: "concatenate ordered parts".to_owned(),
        };
        assert_eq!(
            missing_local_transport_paths(&root, &encoding)
                .unwrap()
                .len(),
            2
        );

        let first_part = root.join("parts/objects/aa/part-00000");
        fs::create_dir_all(first_part.parent().unwrap()).unwrap();
        fs::write(&first_part, b"abc").unwrap();
        let missing = missing_local_transport_paths(&root, &encoding).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].repository_path, "objects/bb/part-00001");

        let package_path = root
            .join("packages")
            .join(format!("{}.zip", encoding.digest().unwrap().0));
        fs::create_dir_all(package_path.parent().unwrap()).unwrap();
        fs::write(package_path, package_bytes).unwrap();
        assert!(missing_local_transport_paths(&root, &encoding)
            .unwrap()
            .is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bootstrap_manifest_paths_are_partitioned_by_semantic_digest() {
        let semantic_digest = ContentDigest(format!("01{}", "a".repeat(62)));
        let manifest_digest = ContentDigest(format!("78{}", "b".repeat(62)));
        assert_eq!(
            bootstrap_manifest_path(&semantic_digest, &manifest_digest),
            format!("manifests/01/{}.json", manifest_digest.0)
        );
    }

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
    fn historical_batch_resolves_identity_no_longer_named_by_active_index() {
        let fixture = bootstrap_batch_fixture();
        let identity = PayloadDependencyIdentity {
            artifact_family: "ccm-matrices".to_owned(),
            semantic_digest: fixture.semantic_digest.clone(),
            manifest_digest: fixture.entry.manifest_digest.clone(),
            payload_digest: fixture.entry.canonical_payload_digest.clone(),
        };
        let batches = vec![(fixture.batch.clone(), fixture.source.clone())];
        let (batch, source, artifact) = historical_publication_for_identity(
            &batches,
            PublicationDestination::Private,
            "ccm-matrices",
            "TeamXcelerator/xcelerator-cache-private-ccm-matrices-0001",
            &identity,
            &fixture.manifest_path,
        )
        .expect("the immutable publication proof remains usable after index replacement");
        assert_eq!(batch.batch_id.0, fixture.entry.publication_transaction_id);
        assert_eq!(source, &fixture.source);
        assert_eq!(artifact.manifest_digest, identity.manifest_digest);

        let mut wrong_identity = identity;
        wrong_identity.payload_digest = ContentDigest::sha256(b"replacement-payload");
        assert!(historical_publication_for_identity(
            &batches,
            PublicationDestination::Private,
            "ccm-matrices",
            "TeamXcelerator/xcelerator-cache-private-ccm-matrices-0001",
            &wrong_identity,
            &fixture.manifest_path,
        )
        .is_none());
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

    #[test]
    #[ignore = "authenticated V1-corpus staging acceptance against the private shard"]
    fn claim_1a_v1_artifact_stages_for_republication() {
        let root = std::env::temp_dir().join("xc-private-v1-republication-acceptance");
        let _ = fs::remove_dir_all(&root);
        let store =
            GitHubBootstrapCacheStore::private("TeamXcelerator", root.join("remote")).unwrap();
        let key = ArtifactKey {
            kind: "gauss_legendre_rule".to_owned(),
            logical_key: "live-claim-1a".to_owned(),
            parameters_digest: ContentDigest(
                "a23310358f5f5e1f83db0cef44bd27d2a26ff1bdbffd94f02292bdba64f8b3fa".to_owned(),
            ),
        };
        let resolved = store.resolve(&key).unwrap().expect("published V1 artifact");
        assert!(resolved.manifest.canonical_payload.dependencies.is_empty());
        assert_eq!(
            resolved.encoding.encoder_profile,
            DETERMINISTIC_ZIP64_PROFILE_V1
        );
        let resolved_manifest_digest = resolved.manifest.digest().unwrap();
        let adapter = store
            .candidates(&key)
            .unwrap()
            .into_iter()
            .find(|candidate| {
                candidate.provenance_digest.as_ref() == Some(&resolved_manifest_digest)
            })
            .expect("identity-bound adapter manifest");
        let (payload, encoded) = store.read_payload_and_encoded(&adapter).unwrap();
        let encoded = encoded.expect("verified retained package");
        let transport = store
            .verified_transport_parts(&adapter)
            .unwrap()
            .expect("retained V1 parts");
        let sink = CanonicalStagingProductionSink::new(
            root.join("staging"),
            TransportPolicy::default(),
            ResourcePolicy::default(),
            CancellationToken::new(),
        )
        .unwrap();
        ArtifactProductionSink::record_with_verified_transport(
            &sink,
            ProducedArtifactRecord {
                operation: "cache.acceptance.v1-republication".to_owned(),
                semantic_key: resolved.manifest.semantic_key.clone(),
                logical_key: adapter.key.logical_key.clone(),
                manifest: adapter,
                achieved_assurance: resolved.index.achieved_assurance,
                assurance_evidence_digests: Vec::new(),
                payload,
            },
            &encoded,
            &transport,
        )
        .unwrap();
        let staged = sink.drafts().unwrap();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].manifest, resolved.manifest);
        assert_eq!(staged[0].encoding, resolved.encoding);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore = "authenticated historical-identity acceptance against the private shard"]
    fn superseded_claim_1a_matrix_resolves_by_exact_historical_identity() {
        let root = std::env::temp_dir().join("xc-private-historical-identity-acceptance");
        let store = GitHubBootstrapCacheStore::private("TeamXcelerator", root).unwrap();
        let identity = PayloadDependencyIdentity {
            artifact_family: "ccm-matrices".to_owned(),
            semantic_digest: ContentDigest(
                "c08e50d3f8af0b8e6c782f0c5d0b92391790a6603196b06e9fa3edc402f0e2dd".to_owned(),
            ),
            manifest_digest: ContentDigest(
                "3e5e36a0eaff6d81fd6122d3467b3e7b8c1a286160b48c5341fcf49c5b026ce8".to_owned(),
            ),
            payload_digest: ContentDigest(
                "cdcdd08618047bcccf4667800d6dbc3eb6cffe41e34fd4cc5fd3029f9a1e2175".to_owned(),
            ),
        };
        let resolved = store
            .resolve_identity(&identity)
            .unwrap()
            .expect("historical matrix remains published");
        assert_eq!(
            resolved.manifest.digest().unwrap(),
            identity.manifest_digest
        );
        assert_eq!(resolved.manifest.payload_digest, identity.payload_digest);

        let candidates = store.identity_candidates(&identity).unwrap();
        assert_eq!(candidates.len(), 1);
        let payload = store.read_payload(&candidates[0]).unwrap();
        assert_eq!(
            ContentDigest::sha256(&payload),
            candidates[0].content_digest
        );
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
