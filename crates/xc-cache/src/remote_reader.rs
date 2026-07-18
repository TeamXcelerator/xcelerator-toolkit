//! Selective, bounded Git shard reads without a persistent full checkout.

use crate::protocol::normalized_relative_path;
use crate::{
    CacheError, CacheVisibility, ContentDigest, RemoteGitStore, RemoteReadReport,
    ShardIndexPartition, TopologyRegistry, TopologyTrustPolicy, TransportEncodingRecord,
};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xc_core::{CancellationToken, ResourcePolicy};

static DOWNLOAD_TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RemoteDocument<T> {
    pub source: RemoteReadReport,
    pub value: T,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TrustedTopologyDocument {
    pub source: RemoteReadReport,
    pub topology_digest: ContentDigest,
    pub registry: TopologyRegistry,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PartFetchReport {
    pub revision: String,
    pub downloaded_sequences: Vec<u64>,
    pub reused_sequences: Vec<u64>,
    pub downloaded_bytes: u64,
    pub verified_bytes: u64,
}

pub struct RemoteShardReader<'a> {
    remote: &'a dyn RemoteGitStore,
    maximum_metadata_bytes: u64,
}

impl<'a> RemoteShardReader<'a> {
    pub fn new(
        remote: &'a dyn RemoteGitStore,
        maximum_metadata_bytes: u64,
    ) -> Result<Self, CacheError> {
        if maximum_metadata_bytes == 0 {
            return Err(CacheError::ResourceLimit(
                "remote metadata limit must be positive".to_owned(),
            ));
        }
        Ok(Self {
            remote,
            maximum_metadata_bytes,
        })
    }

    pub fn read_json<T: DeserializeOwned>(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        cancellation: &CancellationToken,
    ) -> Result<RemoteDocument<T>, CacheError> {
        let mut bytes = Vec::new();
        let source = self.remote.read_committed_path(
            repository,
            revision,
            path,
            self.maximum_metadata_bytes,
            cancellation,
            &mut bytes,
        )?;
        let value = serde_json::from_slice(&bytes)?;
        Ok(RemoteDocument { source, value })
    }

    pub fn load_trusted_topology(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        trust: &TopologyTrustPolicy,
        cancellation: &CancellationToken,
    ) -> Result<TrustedTopologyDocument, CacheError> {
        let document: RemoteDocument<TopologyRegistry> =
            self.read_json(repository, revision, path, cancellation)?;
        let topology_digest = trust.verify(&document.value)?;
        Ok(TrustedTopologyDocument {
            source: document.source,
            topology_digest,
            registry: document.value,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_index_partition(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        family: &str,
        semantic_prefix: &str,
        cancellation: &CancellationToken,
    ) -> Result<RemoteDocument<ShardIndexPartition>, CacheError> {
        let document: RemoteDocument<ShardIndexPartition> =
            self.read_json(repository, revision, path, cancellation)?;
        document.value.validate()?;
        if document.value.family != family || document.value.semantic_prefix != semantic_prefix {
            return Err(CacheError::InvalidManifest(format!(
                "remote shard index {path:?} does not match family/prefix"
            )));
        }
        Ok(document)
    }

    pub fn load_transport_encoding(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        expected_digest: &ContentDigest,
        cancellation: &CancellationToken,
    ) -> Result<RemoteDocument<TransportEncodingRecord>, CacheError> {
        let document: RemoteDocument<TransportEncodingRecord> =
            self.read_json(repository, revision, path, cancellation)?;
        let actual = document.value.digest()?;
        if &actual != expected_digest {
            return Err(CacheError::DigestMismatch {
                expected: expected_digest.to_string(),
                actual: actual.to_string(),
            });
        }
        Ok(document)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fetch_transport_parts(
        &self,
        repository: &str,
        revision: &str,
        record: &TransportEncodingRecord,
        destination_root: &Path,
        resources: &ResourcePolicy,
        cancellation: &CancellationToken,
    ) -> Result<PartFetchReport, CacheError> {
        record.validate()?;
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let mut report = PartFetchReport {
            revision: revision.to_owned(),
            downloaded_sequences: Vec::new(),
            reused_sequences: Vec::new(),
            downloaded_bytes: 0,
            verified_bytes: 0,
        };
        for part in &record.ordered_parts {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let destination = resolve_part_path(destination_root, &part.repository_path)?;
            if destination.exists() {
                verify_local_part(
                    &destination,
                    part.size_bytes,
                    &part.content_digest,
                    resources,
                    cancellation,
                )?;
                report.reused_sequences.push(part.sequence);
                report.verified_bytes = report.verified_bytes.saturating_add(part.size_bytes);
                continue;
            }
            let projected = report.downloaded_bytes.saturating_add(part.size_bytes);
            for (description, maximum) in [
                ("transfer", resources.maximum_transfer_bytes),
                ("permanent-disk", resources.maximum_permanent_disk_bytes),
            ] {
                if maximum.is_some_and(|maximum| projected > maximum) {
                    return Err(CacheError::ResourceLimit(format!(
                        "selective part fetch exceeds {description} budget"
                    )));
                }
            }
            if resources
                .maximum_temporary_disk_bytes
                .is_some_and(|maximum| part.size_bytes > maximum)
            {
                return Err(CacheError::ResourceLimit(
                    "one downloaded part exceeds temporary-disk budget".to_owned(),
                ));
            }
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            let (temporary_path, mut temporary) = create_download_file(&destination)?;
            let download = self.remote.read_committed_path(
                repository,
                revision,
                &part.repository_path,
                part.size_bytes,
                cancellation,
                &mut temporary,
            );
            let source = match download {
                Ok(source) => source,
                Err(error) => {
                    drop(temporary);
                    let _ = fs::remove_file(&temporary_path);
                    return Err(error);
                }
            };
            temporary.sync_all()?;
            drop(temporary);
            if source.size_bytes != part.size_bytes
                || source.content_digest != part.content_digest
                || source.repository_path != part.repository_path
                || source.revision != revision
            {
                let _ = fs::remove_file(&temporary_path);
                return Err(CacheError::DigestMismatch {
                    expected: format!("{} ({} bytes)", part.content_digest, part.size_bytes),
                    actual: format!("{} ({} bytes)", source.content_digest, source.size_bytes),
                });
            }
            match fs::hard_link(&temporary_path, &destination) {
                Ok(()) => {
                    let _ = fs::remove_file(&temporary_path);
                    report.downloaded_sequences.push(part.sequence);
                    report.downloaded_bytes = projected;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    let _ = fs::remove_file(&temporary_path);
                    verify_local_part(
                        &destination,
                        part.size_bytes,
                        &part.content_digest,
                        resources,
                        cancellation,
                    )?;
                    report.reused_sequences.push(part.sequence);
                }
                Err(error) => {
                    let _ = fs::remove_file(&temporary_path);
                    return Err(error.into());
                }
            }
            report.verified_bytes = report.verified_bytes.saturating_add(part.size_bytes);
        }
        Ok(report)
    }
}

pub fn resolve_topology_family(
    topology: &TopologyRegistry,
    family: &str,
    visibility: CacheVisibility,
) -> Result<crate::ArtifactFamilyRoute, CacheError> {
    topology.validate()?;
    topology.route(family, visibility).cloned().ok_or_else(|| {
        CacheError::NotFound(format!(
            "topology route for family {family:?} and {visibility:?} visibility"
        ))
    })
}

fn resolve_part_path(root: &Path, repository_path: &str) -> Result<PathBuf, CacheError> {
    if !normalized_relative_path(repository_path) {
        return Err(CacheError::InvalidManifest(format!(
            "remote part path {repository_path:?} is unsafe"
        )));
    }
    Ok(repository_path
        .split('/')
        .fold(root.to_owned(), |path, component| path.join(component)))
}

fn verify_local_part(
    path: &Path,
    expected_size: u64,
    expected_digest: &ContentDigest,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CacheError::InvalidManifest(format!(
            "local part {} is not a regular file",
            path.display()
        )));
    }
    let buffer_size = resources
        .maximum_memory_bytes
        .unwrap_or(1024 * 1024)
        .clamp(1, 1024 * 1024) as usize;
    let mut buffer = vec![0u8; buffer_size];
    let mut input = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        size = size.saturating_add(count as u64);
        hasher.update(&buffer[..count]);
    }
    let digest = ContentDigest(format!("{:x}", hasher.finalize()));
    if size != expected_size || &digest != expected_digest {
        return Err(CacheError::DigestMismatch {
            expected: format!("{expected_digest} ({expected_size} bytes)"),
            actual: format!("{digest} ({size} bytes)"),
        });
    }
    Ok(())
}

fn create_download_file(destination: &Path) -> Result<(PathBuf, File), CacheError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let name = destination
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("part"))
        .to_string_lossy();
    for _ in 0..128 {
        let sequence = DOWNLOAD_TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{name}.xc-download-{}-{sequence}",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(CacheError::Io(
        "could not allocate a unique part download file".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArtifactFamilyRoute, CompareAndSwapResult, RemoteCommitRequest, TopologyShardRoute,
        TopologyShardStatus, TransportPart,
    };
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    struct MemoryRemote {
        revision: String,
        paths: BTreeMap<String, Vec<u8>>,
        reads: AtomicUsize,
    }

    impl RemoteGitStore for MemoryRemote {
        fn read_ref(&self, _repository: &str, _branch: &str) -> Result<String, CacheError> {
            Ok(self.revision.clone())
        }

        fn immutable_path_digest(
            &self,
            _repository: &str,
            _revision: &str,
            path: &str,
        ) -> Result<Option<ContentDigest>, CacheError> {
            Ok(self
                .paths
                .get(path)
                .map(|bytes| ContentDigest::sha256(bytes)))
        }

        fn read_committed_path(
            &self,
            _repository: &str,
            revision: &str,
            path: &str,
            maximum_bytes: u64,
            cancellation: &CancellationToken,
            writer: &mut dyn Write,
        ) -> Result<RemoteReadReport, CacheError> {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            if revision != self.revision {
                return Err(CacheError::NotFound(revision.to_owned()));
            }
            let bytes = self
                .paths
                .get(path)
                .ok_or_else(|| CacheError::NotFound(path.to_owned()))?;
            if bytes.len() as u64 > maximum_bytes {
                return Err(CacheError::ResourceLimit(path.to_owned()));
            }
            self.reads.fetch_add(1, AtomicOrdering::Relaxed);
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
            _request: &RemoteCommitRequest,
        ) -> Result<CompareAndSwapResult, CacheError> {
            Err(CacheError::ReadOnlyLayer("memory-remote".to_owned()))
        }

        fn verify_committed_part(
            &self,
            _repository: &str,
            _revision: &str,
            _part: &TransportPart,
        ) -> Result<(), CacheError> {
            Ok(())
        }
    }

    fn temporary_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "xc-cache-remote-reader-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn trusted_topology_is_fetched_as_one_bounded_document() {
        let topology = TopologyRegistry {
            schema_version: 1,
            generation: 3,
            previous_registry_digest: None,
            policy_digest: ContentDigest::sha256(b"policy"),
            trust_anchor_ids: vec!["release-key".to_owned()],
            family_routes: vec![ArtifactFamilyRoute {
                family: "ccm".to_owned(),
                visibility: CacheVisibility::Public,
                ordered_shards: vec![TopologyShardRoute {
                    shard_id: "public-001".to_owned(),
                    endpoint_id: "public-001".to_owned(),
                    sequence: 1,
                    status: TopologyShardStatus::Writable,
                    successor_shard_id: None,
                }],
            }],
        };
        let bytes = serde_json::to_vec(&topology).unwrap();
        let remote = MemoryRemote {
            revision: "a".repeat(40),
            paths: BTreeMap::from([("registry.json".to_owned(), bytes)]),
            reads: AtomicUsize::new(0),
        };
        let reader = RemoteShardReader::new(&remote, 1024 * 1024).unwrap();
        let document = reader
            .load_trusted_topology(
                "team/registry",
                &remote.revision,
                "registry.json",
                &TopologyTrustPolicy {
                    minimum_generation: 3,
                    pinned_registry_digest: Some(topology.digest().unwrap()),
                    required_trust_anchor: Some("release-key".to_owned()),
                },
                &CancellationToken::new(),
            )
            .unwrap();
        let route =
            resolve_topology_family(&document.registry, "ccm", CacheVisibility::Public).unwrap();
        assert_eq!(route.ordered_shards[0].shard_id, "public-001");
        assert_eq!(remote.reads.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn part_fetch_resumes_from_verified_local_objects() {
        let revision = "b".repeat(40);
        let values = [b"first-part".as_slice(), b"second-part".as_slice()];
        let parts: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(sequence, bytes)| {
                let digest = ContentDigest::sha256(bytes);
                TransportPart {
                    sequence: sequence as u64,
                    repository_path: format!("objects/{digest}.part"),
                    size_bytes: bytes.len() as u64,
                    content_digest: digest,
                }
            })
            .collect();
        let record = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: ContentDigest::sha256(b"payload"),
            encoder_profile: "fixture-v1".to_owned(),
            package_size_bytes: values.iter().map(|value| value.len() as u64).sum(),
            package_digest: ContentDigest::sha256_chunks(values),
            ordered_parts: parts.clone(),
            reconstruction: "concatenate".to_owned(),
        };
        let remote = MemoryRemote {
            revision: revision.clone(),
            paths: parts
                .iter()
                .zip(values)
                .map(|(part, bytes)| (part.repository_path.clone(), bytes.to_vec()))
                .collect(),
            reads: AtomicUsize::new(0),
        };
        let root = temporary_root("resume");
        let _ = fs::remove_dir_all(&root);
        let reader = RemoteShardReader::new(&remote, 1024).unwrap();
        let first = reader
            .fetch_transport_parts(
                "team/shard",
                &revision,
                &record,
                &root,
                &ResourcePolicy::default(),
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(first.downloaded_sequences, vec![0, 1]);
        let reads_after_first = remote.reads.load(AtomicOrdering::Relaxed);
        let second = reader
            .fetch_transport_parts(
                "team/shard",
                &revision,
                &record,
                &root,
                &ResourcePolicy::default(),
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(second.reused_sequences, vec![0, 1]);
        assert_eq!(
            remote.reads.load(AtomicOrdering::Relaxed),
            reads_after_first
        );
        let _ = fs::remove_dir_all(root);
    }
}
