//! Selective reads from live public and authenticated private registries.

use crate::*;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Deserialize)]
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

#[derive(Deserialize)]
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
        })
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
        let registry_id = format!("{}/xcelerator-cache-{visibility_name}-registry", self.owner);
        let registry_repository = format!("https://github.com/{registry_id}.git");
        let registry_revision = self.remote.read_ref(&registry_repository, "main")?;
        let (registry, _): (BootstrapRegistry, _) =
            self.read_json(&registry_repository, &registry_revision, "registry.json")?;
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
        let revision = self.remote.read_ref(&repository, "main")?;
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
        let receipt_path = format!(
            "transactions/{}/{visibility_name}/receipt.json",
            entry.publication_transaction_id,
        );
        let (receipt, receipt_source): (PublicationReceipt, _) =
            self.read_json(&repository, &revision, &receipt_path)?;
        receipt.validate()?;
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
            receipt: crate::RemotePublicationEvidence::ArtifactReceipt(Box::new(receipt)),
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
        let item = resolved
            .manifest
            .canonical_payload
            .ordered_items
            .iter()
            .find(|item| item.normalized_path == "payload.json")
            .ok_or_else(|| {
                CacheError::InvalidManifest("managed JSON artifact has no payload.json".to_owned())
            })?;
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
            producer_toolkit_version: ToolkitVersion::parse("0.13.0")?,
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
        materialize_resolved_remote_artifact(
            &self.remote,
            &resolved,
            &self.root.join("parts"),
            &package,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )?;
        let file = fs::File::open(package)?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|error| CacheError::Io(error.to_string()))?;
        let mut payload = archive
            .by_name("payload.json")
            .map_err(|error| CacheError::Io(error.to_string()))?;
        let mut bytes = Vec::new();
        payload.read_to_end(&mut bytes)?;
        let actual = ContentDigest::sha256(&bytes);
        if actual != manifest.content_digest || bytes.len() as u64 != manifest.size_bytes {
            return Err(CacheError::DigestMismatch {
                expected: manifest.content_digest.to_string(),
                actual: actual.to_string(),
            });
        }
        writer.write_all(&bytes)?;
        Ok(())
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
                "67e1ed84a3f813fee4cbb7b1b69931d61e0a98c8ebb050c8ccfba9bd77e47650",
            ),
            (
                "ccm_tau_matrix",
                "8b6c9bacb85a971541dc029a08912cb7b19e1a67fbc9c5bea5b725b59e8e6442",
            ),
            (
                "ccm_weil_eigenpair",
                "81115ac35d4a59ad70ac4982a03ea4024a01e22f5ed8b0d8e625cfd5b7a12b6e",
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
