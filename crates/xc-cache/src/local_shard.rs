//! Bounded, offline loading of an explicitly selected canonical shard manifest.
//! Verifies the selected local snapshot; does not acquire dependencies, check
//! remote freshness, replay certificates, or publish anything.

use crate::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalShardReadOptions {
    /// Temporary package storage outside the source shard.
    pub scratch_directory: PathBuf,
    pub maximum_payload_bytes: u64,
    pub maximum_package_bytes: u64,
}

pub struct LocalShardJson {
    pub manifest: ArtifactManifest,
    pub payload: Vec<u8>,
    pub canonical_manifest_digest: ContentDigest,
    pub transport_digest: ContentDigest,
}

fn invalid(message: impl Into<String>) -> CacheError {
    CacheError::InvalidManifest(message.into())
}

fn confined_file(root: &Path, relative: &str) -> Result<PathBuf, CacheError> {
    if !crate::protocol::normalized_relative_path(relative) {
        return Err(invalid("unsafe local shard path"));
    }
    let path = root.join(relative).canonicalize()?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(invalid(
            "local shard path escapes its root or is not a file",
        ));
    }
    Ok(path)
}

fn metadata<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, CacheError> {
    Ok(metadata_with_digest::<T>(path)?.0)
}

fn metadata_with_digest<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<(T, ContentDigest), CacheError> {
    const LIMIT: u64 = 16 * 1024 * 1024;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Err(CacheError::ResourceLimit(
            "local shard metadata exceeds 16 MiB".into(),
        ));
    }
    Ok((
        serde_json::from_slice(&bytes)?,
        ContentDigest::sha256(&bytes),
    ))
}

static SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new(base: &Path) -> Result<Self, CacheError> {
        fs::create_dir_all(base)?;
        let base = base.canonicalize()?;
        for _ in 0..100 {
            let path = base.join(format!(
                "xc-local-shard-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err(invalid(
            "cannot allocate unique local shard scratch directory",
        ))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        // Only our named package and our now-empty directory; never recursive.
        let _ = fs::remove_file(self.0.join("package.zip"));
        let _ = fs::remove_dir(&self.0);
    }
}
struct BoundedPayload {
    bytes: Vec<u8>,
    maximum: usize,
}
impl Write for BoundedPayload {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "decoded payload exceeds its declared byte budget",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(std::io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Read `SHARD/manifests/SEMANTIC_PREFIX/MANIFEST_DIGEST.json` using the
/// existing ordered-part reconstruction and canonical ZIP verifier. The
/// manifest must be active in the local index. Returned assurance is the
/// index's declared grade, not a newly replayed certificate. The caller must
/// still approve the returned logical payload digest for its experiment.
pub fn read_local_shard_json(
    manifest_path: &Path,
    options: &LocalShardReadOptions,
) -> Result<LocalShardJson, CacheError> {
    if options.maximum_payload_bytes == 0 || options.maximum_package_bytes == 0 {
        return Err(invalid("local shard reads require positive byte budgets"));
    }
    let path = manifest_path.canonicalize()?;
    let root = path
        .ancestors()
        .nth(3)
        .ok_or_else(|| invalid("manifest must be inside a canonical shard layout"))?
        .to_path_buf();
    let (canonical, raw_manifest_digest): (CanonicalArtifactManifest, ContentDigest) =
        metadata_with_digest(&path)?;
    canonical.validate()?;
    let manifest_digest = canonical.digest()?;
    if raw_manifest_digest != manifest_digest {
        return Err(invalid(
            "local manifest bytes are not canonical or their digest differs",
        ));
    }
    let expected = format!(
        "manifests/{}/{}.json",
        &canonical.semantic_digest.0[..2],
        manifest_digest.0
    );
    if confined_file(&root, &expected)? != path {
        return Err(invalid("canonical manifest path does not bind its digest"));
    }
    #[derive(Deserialize)]
    struct Descriptor {
        schema_version: u32,
        family: String,
        visibility: CacheVisibility,
        immutable_objects: bool,
        artifact_kinds: Vec<String>,
    }
    let descriptor: Descriptor = metadata(&confined_file(&root, "cache-repository.json")?)?;
    if descriptor.schema_version != 1
        || descriptor.family != canonical.artifact_family
        || !descriptor.immutable_objects
        || !descriptor
            .artifact_kinds
            .contains(&canonical.semantic_key.artifact_kind)
        || !matches!(
            descriptor.visibility,
            CacheVisibility::Public | CacheVisibility::Private
        )
    {
        return Err(invalid(
            "local shard descriptor does not admit this manifest",
        ));
    }
    if descriptor.visibility == CacheVisibility::Public
        && !artifact_semantics_admitted_to_destination(
            &canonical.semantic_key,
            PublicationDestination::Public,
        )
    {
        return Err(invalid(
            "artifact semantics are not eligible for the public shard",
        ));
    }
    let current = ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?;
    if current < canonical.minimum_reader_version
        || canonical
            .maximum_reader_version
            .as_ref()
            .is_some_and(|v| &current > v)
    {
        return Err(invalid(
            "local shard artifact is incompatible with this reader version",
        ));
    }
    let index: ShardIndexPartition = metadata(&confined_file(
        &root,
        &format!(
            "indexes/{}/{}.json",
            canonical.artifact_family,
            &canonical.semantic_digest.0[..2]
        ),
    )?)?;
    index.validate()?;
    let entry = index
        .entries
        .iter()
        .find(|e| e.manifest_digest == manifest_digest)
        .ok_or_else(|| invalid("selected manifest is absent from the local shard index"))?;
    if index.family != canonical.artifact_family
        || index.semantic_prefix != canonical.semantic_digest.0[..2]
        || entry.disposition != ArtifactDisposition::Active
        || entry.semantic_digest != canonical.semantic_digest
        || entry.canonical_payload_digest != canonical.payload_digest
        || entry.transport_digests != canonical.transport_digests
        || entry.producer_toolkit_version != canonical.producer_toolkit_version
        || entry.minimum_reader_version != canonical.minimum_reader_version
        || entry.achieved_assurance.mathematical().is_none()
    {
        return Err(invalid(
            "local shard index does not admit the selected manifest",
        ));
    }
    let items = &canonical.canonical_payload.ordered_items;
    if items.len() != 1 || items[0].normalized_path != "payload.json" {
        return Err(invalid(
            "local JSON source must contain exactly one payload.json",
        ));
    }
    let item = &items[0];
    if item.size_bytes > options.maximum_payload_bytes {
        return Err(CacheError::ResourceLimit(
            "logical payload exceeds the explicit local read budget".into(),
        ));
    }
    let transport_digest = entry
        .transport_digests
        .first()
        .ok_or_else(|| invalid("missing transport identity"))?
        .clone();
    let (encoding, raw_transport_digest): (TransportEncodingRecord, ContentDigest) =
        metadata_with_digest(&confined_file(
            &root,
            &format!(
                "encodings/{}/{}.json",
                &canonical.payload_digest.0[..2],
                transport_digest.0
            ),
        )?)?;
    encoding.validate()?;
    if raw_transport_digest != transport_digest
        || encoding.digest()? != transport_digest
        || encoding.canonical_payload_digest != canonical.payload_digest
    {
        return Err(invalid(
            "local encoding identity does not match the canonical manifest",
        ));
    }
    if encoding.package_size_bytes > options.maximum_package_bytes {
        return Err(CacheError::ResourceLimit(
            "transport package exceeds the explicit local read budget".into(),
        ));
    }
    for part in &encoding.ordered_parts {
        confined_file(&root, &part.repository_path)?;
    }
    if options
        .scratch_directory
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return Err(invalid(
            "scratch_directory must be normalized without parent components",
        ));
    }
    let existing = options
        .scratch_directory
        .ancestors()
        .find(|p| p.exists())
        .ok_or_else(|| invalid("scratch directory has no existing ancestor"))?;
    if existing.canonicalize()?.starts_with(&root) {
        return Err(invalid(
            "scratch directory must be outside the source shard",
        ));
    }
    fs::create_dir_all(&options.scratch_directory)?;
    if options.scratch_directory.canonicalize()?.starts_with(&root) {
        return Err(invalid(
            "scratch directory must be outside the source shard",
        ));
    }
    let scratch = Scratch::new(&options.scratch_directory)?;
    let resources = ResourcePolicy {
        maximum_memory_bytes: Some(options.maximum_payload_bytes),
        maximum_temporary_disk_bytes: Some(options.maximum_package_bytes),
        maximum_transfer_bytes: Some(options.maximum_package_bytes),
        ..ResourcePolicy::default()
    };
    let cancellation = CancellationToken::default();
    let package = scratch.0.join("package.zip");
    reconstruct_transport_package(&encoding, &root, &package, &resources, &cancellation)?;
    let maximum = usize::try_from(item.size_bytes)
        .map_err(|_| CacheError::ResourceLimit("payload size does not fit this platform".into()))?;
    let mut payload = BoundedPayload {
        bytes: Vec::new(),
        maximum,
    };
    crate::packaging::verify_canonical_payload_zip64_to_writer(
        &canonical.canonical_payload,
        &encoding,
        &package,
        &cancellation,
        false,
        &mut payload,
    )?;
    let quality = match entry.achieved_assurance {
        ArtifactAssuranceState::Certified => CacheQuality::Certified,
        ArtifactAssuranceState::CrossChecked => CacheQuality::CrossChecked,
        _ => CacheQuality::Validated,
    };
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey {
            kind: canonical.semantic_key.artifact_kind.clone(),
            logical_key: format!(
                "local-shard/{}/{}",
                canonical.artifact_family, canonical.semantic_digest.0
            ),
            parameters_digest: canonical.semantic_digest.clone(),
        },
        content_digest: item.content_digest.clone(),
        size_bytes: item.size_bytes,
        objects: vec![CacheObjectRef {
            content_digest: item.content_digest.clone(),
            size_bytes: item.size_bytes,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: canonical.producer_toolkit_version.clone(),
        minimum_reader_version: canonical.minimum_reader_version.clone(),
        maximum_reader_version: canonical.maximum_reader_version.clone(),
        quality,
        visibility: descriptor.visibility,
        immutable: true,
        dependencies: Vec::new(),
        tags: BTreeMap::from([
            (
                SEMANTIC_KEY_MANIFEST_TAG.into(),
                serde_json::to_string(&canonical.semantic_key)?,
            ),
            (
                REMOTE_CANONICAL_MANIFEST_TAG.into(),
                serde_json::to_string(&canonical)?,
            ),
        ]),
        provenance_digest: Some(manifest_digest.clone()),
    };
    manifest.validate()?;
    Ok(LocalShardJson {
        manifest,
        payload: payload.bytes,
        canonical_manifest_digest: manifest_digest,
        transport_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture {
        base: PathBuf,
        manifest_path: PathBuf,
        index_path: PathBuf,
        part_path: PathBuf,
        payload: Vec<u8>,
        options: LocalShardReadOptions,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let temporary = std::env::temp_dir().canonicalize().unwrap();
            let target = self.base.canonicalize().unwrap();
            assert!(target.starts_with(&temporary) && target != temporary);
            assert!(target
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("xc-local-shard-test-"));
            fs::remove_dir_all(target).unwrap();
        }
    }
    fn write_json(path: &Path, value: &impl Serialize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, crate::protocol::canonical_json_bytes(value).unwrap()).unwrap();
    }
    fn fixture(visibility: CacheVisibility, assurance: ArtifactAssuranceState) -> Fixture {
        let base = std::env::temp_dir().join(format!(
            "xc-local-shard-test-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&base).unwrap();
        let root = base.join("shard");
        fs::create_dir(&root).unwrap();
        let payload=serde_json::to_vec(&json!({"schema_version":1,"lambda_squared":"13","dimension":2,"n_modes":1,"precision_bits":256,"entries":["2","0","0","3"]})).unwrap();
        let envelope = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "mpfr".into(),
            precision_bits: Some(256),
            scalar_representation: "decimal_roundtrip".into(),
            dimensions: vec![2, 2],
            endianness: "not-applicable".into(),
            special_value_encoding: "none".into(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "payload.json".into(),
                content_digest: ContentDigest::sha256(&payload),
                size_bytes: payload.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let package = base.join("original.zip");
        let resources = ResourcePolicy::default();
        let cancel = CancellationToken::default();
        package_canonical_payload_bytes_zip64(
            &envelope,
            "payload.json",
            &payload,
            &package,
            &resources,
            &cancel,
        )
        .unwrap();
        let encoding = stream_split_encoded(
            &mut fs::File::open(&package).unwrap(),
            envelope.digest().unwrap(),
            DETERMINISTIC_ZIP64_PROFILE_V2,
            &TransportPolicy {
                split_part_bytes: 64,
                ..TransportPolicy::default()
            },
            &resources,
            &cancel,
            |part, bytes| {
                let path = root.join(&part.repository_path);
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(path, bytes)?;
                Ok(())
            },
        )
        .unwrap();
        assert!(encoding.ordered_parts.len() > 1);
        let semantic = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_even_sector_matrix".into(),
            mathematical_semantics_version: "test-local-shard-v1".into(),
            resolved_mathematical_parameters: json!({"cutoff":"13"}),
            normalization: None,
            target: None,
            subspace: Some("even".into()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let version = ToolkitVersion::parse("0.15.0").unwrap();
        let canonical = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm-matrices".into(),
            semantic_digest: semantic.digest().unwrap(),
            semantic_key: semantic,
            canonical_payload: envelope.clone(),
            payload_digest: envelope.digest().unwrap(),
            transport_digests: vec![encoding.digest().unwrap()],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"configuration"),
            producer_toolkit_version: version.clone(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            claim_scope: "synthetic finite fixture".into(),
            assumptions: Vec::new(),
        };
        let entry = ShardIndexEntry {
            semantic_digest: canonical.semantic_digest.clone(),
            canonical_payload_digest: canonical.payload_digest.clone(),
            manifest_digest: canonical.digest().unwrap(),
            achieved_assurance: assurance,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: version,
            minimum_reader_version: canonical.minimum_reader_version.clone(),
            transport_digests: canonical.transport_digests.clone(),
            publication_transaction_id: "a".repeat(64),
        };
        let index = ShardIndexPartition::rebuild(
            "ccm-matrices",
            &canonical.semantic_digest.0[..2],
            vec![entry],
        )
        .unwrap();
        let index_path = root.join(format!(
            "indexes/ccm-matrices/{}.json",
            &canonical.semantic_digest.0[..2]
        ));
        write_json(&index_path, &index);
        let manifest_path = root.join(format!(
            "manifests/{}/{}.json",
            &canonical.semantic_digest.0[..2],
            canonical.digest().unwrap().0
        ));
        write_json(&manifest_path, &canonical);
        write_json(
            &root.join(format!(
                "encodings/{}/{}.json",
                &canonical.payload_digest.0[..2],
                encoding.digest().unwrap().0
            )),
            &encoding,
        );
        write_json(
            &root.join("cache-repository.json"),
            &json!({"schema_version":1,"family":"ccm-matrices","visibility":visibility,"immutable_objects":true,"artifact_kinds":["ccm_even_sector_matrix"]}),
        );
        Fixture {
            base: base.clone(),
            manifest_path,
            index_path,
            part_path: root.join(&encoding.ordered_parts[0].repository_path),
            payload,
            options: LocalShardReadOptions {
                scratch_directory: base.join("scratch"),
                maximum_payload_bytes: 1024 * 1024,
                maximum_package_bytes: 1024 * 1024,
            },
        }
    }

    #[test]
    fn reconstructs_split_sources_and_preserves_visibility_grade_and_provenance() {
        for visibility in [CacheVisibility::Private, CacheVisibility::Public] {
            for (assurance, quality) in [
                (ArtifactAssuranceState::Computed, CacheQuality::Validated),
                (
                    ArtifactAssuranceState::CrossChecked,
                    CacheQuality::CrossChecked,
                ),
                (ArtifactAssuranceState::Certified, CacheQuality::Certified),
            ] {
                let f = fixture(visibility, assurance);
                let read = read_local_shard_json(&f.manifest_path, &f.options).unwrap();
                assert_eq!(read.payload, f.payload);
                assert_eq!(
                    read.manifest.content_digest,
                    ContentDigest::sha256(&f.payload)
                );
                assert_eq!(read.manifest.visibility, visibility);
                assert_eq!(read.manifest.quality, quality);
                assert_eq!(
                    read.manifest.provenance_digest,
                    Some(read.canonical_manifest_digest)
                );
                assert!(read
                    .manifest
                    .tags
                    .contains_key(REMOTE_CANONICAL_MANIFEST_TAG));
                assert_eq!(
                    fs::read_dir(&f.options.scratch_directory).unwrap().count(),
                    0
                );
            }
        }
    }
    #[test]
    fn rejects_tampered_parts_stale_disposition_and_uncomputed_sources() {
        let f = fixture(CacheVisibility::Private, ArtifactAssuranceState::Computed);
        let manifest_bytes = fs::read(&f.manifest_path).unwrap();
        let mut reformatted = manifest_bytes.clone();
        reformatted.push(b'\n');
        fs::write(&f.manifest_path, reformatted).unwrap();
        assert!(read_local_shard_json(&f.manifest_path, &f.options).is_err());
        fs::write(&f.manifest_path, manifest_bytes).unwrap();
        let canonical: CanonicalArtifactManifest = metadata(&f.manifest_path).unwrap();
        let encoding_path = f.base.join(format!(
            "shard/encodings/{}/{}.json",
            &canonical.payload_digest.0[..2],
            canonical.transport_digests[0].0
        ));
        let encoding_bytes = fs::read(&encoding_path).unwrap();
        let mut reformatted = encoding_bytes.clone();
        reformatted.push(b'\n');
        fs::write(&encoding_path, reformatted).unwrap();
        assert!(read_local_shard_json(&f.manifest_path, &f.options).is_err());
        fs::write(&encoding_path, encoding_bytes).unwrap();
        let original = fs::read(&f.part_path).unwrap();
        fs::write(&f.part_path, b"corrupt").unwrap();
        assert!(read_local_shard_json(&f.manifest_path, &f.options).is_err());
        fs::write(&f.part_path, original).unwrap();
        let mut index: ShardIndexPartition = metadata(&f.index_path).unwrap();
        for disposition in [
            ArtifactDisposition::Deprecated,
            ArtifactDisposition::Quarantined,
            ArtifactDisposition::Revoked,
        ] {
            index.entries[0].disposition = disposition;
            write_json(&f.index_path, &index);
            assert!(read_local_shard_json(&f.manifest_path, &f.options).is_err());
        }
        index.entries[0].disposition = ArtifactDisposition::Active;
        index.entries[0].achieved_assurance = ArtifactAssuranceState::StructurallyValidated;
        write_json(&f.index_path, &index);
        assert!(read_local_shard_json(&f.manifest_path, &f.options).is_err());
    }
    #[test]
    fn enforces_resource_limits_and_never_creates_scratch_inside_a_source_shard() {
        let f = fixture(CacheVisibility::Private, ArtifactAssuranceState::Computed);
        for options in [
            LocalShardReadOptions {
                maximum_payload_bytes: 1,
                ..f.options.clone()
            },
            LocalShardReadOptions {
                maximum_package_bytes: 1,
                ..f.options.clone()
            },
        ] {
            assert!(matches!(
                read_local_shard_json(&f.manifest_path, &options),
                Err(CacheError::ResourceLimit(_))
            ));
        }
        let forbidden = f.base.join("shard/new-scratch");
        assert!(read_local_shard_json(
            &f.manifest_path,
            &LocalShardReadOptions {
                scratch_directory: forbidden.clone(),
                ..f.options.clone()
            }
        )
        .is_err());
        assert!(!forbidden.exists());
        let mut bounded = BoundedPayload {
            bytes: Vec::new(),
            maximum: 1,
        };
        assert!(bounded.write_all(b"too large").is_err());
        assert!(bounded.bytes.is_empty());
    }
}
