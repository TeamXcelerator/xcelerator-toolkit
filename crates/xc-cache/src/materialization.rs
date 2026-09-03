//! End-to-end selective remote retrieval and decoded payload verification.

use crate::protocol::normalized_relative_path;
use crate::remote_reader::{quarantine_corrupt_reused_parts, QuarantineCleanup};
use crate::{
    reconstruct_transport_package, verify_canonical_payload_zip64,
    verify_canonical_payload_zip64_to_writer, CacheError, ContentDigest,
    DeterministicPackageReport, PartFetchReport, RemoteGitStore, RemoteShardReader,
    ResolvedRemoteArtifact, VerifiedPackageReport,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RemoteArtifactMaterializationReport {
    pub schema_version: u32,
    pub artifact_family: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub canonical_payload_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub repository: String,
    pub revision: String,
    pub package_path: PathBuf,
    pub projected_new_local_bytes: u64,
    pub reused_verified_package: bool,
    pub part_fetch: Option<PartFetchReport>,
    pub package: DeterministicPackageReport,
    pub verification: VerifiedPackageReport,
    #[serde(skip)]
    pub part_fetch_elapsed_millis: u64,
    #[serde(skip)]
    pub package_reconstruction_elapsed_millis: u64,
    #[serde(skip)]
    pub payload_verification_elapsed_millis: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MaterializationTimings {
    part_fetch_millis: u64,
    package_reconstruction_millis: u64,
    payload_verification_millis: u64,
}

fn elapsed_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn performance_metadata(
    artifact: &ResolvedRemoteArtifact,
    cache_disposition: &str,
) -> xc_core::PerformanceStageMetadata {
    xc_core::PerformanceStageMetadata {
        operation: Some(artifact.family.clone()),
        cache_disposition: Some(cache_disposition.to_owned()),
        ..xc_core::PerformanceStageMetadata::default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RemoteArtifactClosureMaterializationReport {
    pub schema_version: u32,
    pub root_semantic_digest: ContentDigest,
    pub root_manifest_digest: ContentDigest,
    pub dependency_count: usize,
    pub dependency_closure_fully_validated: bool,
    pub artifacts_dependency_first: Vec<RemoteArtifactMaterializationReport>,
}

pub fn materialize_resolved_remote_artifact_closure(
    remote: &dyn RemoteGitStore,
    root: &ResolvedRemoteArtifact,
    parts_root: &Path,
    dependency_packages_root: &Path,
    root_package_path: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<RemoteArtifactClosureMaterializationReport, CacheError> {
    if dependency_packages_root.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "full closure materialization requires a dependency package root".to_owned(),
        ));
    }
    let mut ordered = Vec::new();
    let mut seen = BTreeSet::new();
    let mut active = BTreeSet::new();
    collect_dependency_first(root, &mut seen, &mut active, &mut ordered)?;
    let root_manifest_digest = root.manifest.digest()?;
    let mut reports = Vec::with_capacity(ordered.len());
    for artifact in ordered {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let manifest_digest = artifact.manifest.digest()?;
        let package_path = if artifact.semantic_digest == root.semantic_digest
            && manifest_digest == root_manifest_digest
        {
            root_package_path.to_owned()
        } else {
            dependency_packages_root
                .join(&artifact.semantic_digest.0)
                .join(format!("{}.zip", manifest_digest.0))
        };
        if let Some(parent) = package_path.parent() {
            fs::create_dir_all(parent)?;
        }
        reports.push(materialize_resolved_remote_artifact(
            remote,
            artifact,
            parts_root,
            &package_path,
            resources,
            cancellation,
        )?);
    }
    Ok(RemoteArtifactClosureMaterializationReport {
        schema_version: 1,
        root_semantic_digest: root.semantic_digest.clone(),
        root_manifest_digest,
        dependency_count: reports.len().saturating_sub(1),
        dependency_closure_fully_validated: true,
        artifacts_dependency_first: reports,
    })
}

fn collect_dependency_first<'a>(
    artifact: &'a ResolvedRemoteArtifact,
    seen: &mut BTreeSet<(ContentDigest, ContentDigest)>,
    active: &mut BTreeSet<(ContentDigest, ContentDigest)>,
    ordered: &mut Vec<&'a ResolvedRemoteArtifact>,
) -> Result<(), CacheError> {
    let identity = (
        artifact.semantic_digest.clone(),
        artifact.manifest.digest()?,
    );
    if seen.contains(&identity) {
        return Ok(());
    }
    if !active.insert(identity.clone()) {
        return Err(CacheError::InvalidManifest(
            "resolved dependency materialization graph contains a cycle".to_owned(),
        ));
    }
    if artifact.manifest.canonical_payload.dependencies.len() != artifact.dependencies.len() {
        return Err(CacheError::InvalidManifest(format!(
            "resolved dependency count does not match the canonical manifest for {}",
            artifact.semantic_digest.0
        )));
    }
    for declared in &artifact.manifest.canonical_payload.dependencies {
        let mut matching = artifact.dependencies.iter().filter(|dependency| {
            dependency.family == declared.artifact_family
                && dependency.semantic_digest == declared.semantic_digest
                && dependency.manifest.payload_digest == declared.payload_digest
                && dependency.manifest.digest().ok().as_ref() == Some(&declared.manifest_digest)
        });
        if matching.next().is_none() || matching.next().is_some() {
            return Err(CacheError::InvalidManifest(format!(
                "resolved dependency graph does not exactly match declared dependency {}/{}",
                declared.artifact_family, declared.semantic_digest.0
            )));
        }
    }
    for dependency in &artifact.dependencies {
        collect_dependency_first(dependency, seen, active, ordered)?;
    }
    active.remove(&identity);
    seen.insert(identity);
    ordered.push(artifact);
    Ok(())
}

/// Materialize one already-resolved artifact without cloning its shard.
///
/// Verified parts are retained under `parts_root` for resumable reuse. The
/// reconstructed package becomes visible only after its transport identity is
/// complete, and a newly reconstructed package is removed if decoded logical
/// payload validation fails.
pub fn materialize_resolved_remote_artifact(
    remote: &dyn RemoteGitStore,
    artifact: &ResolvedRemoteArtifact,
    parts_root: &Path,
    package_path: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<RemoteArtifactMaterializationReport, CacheError> {
    materialize_resolved_remote_artifact_inner(
        remote,
        artifact,
        parts_root,
        package_path,
        resources,
        cancellation,
        None,
    )
}

pub(crate) fn materialize_resolved_remote_artifact_to_writer(
    remote: &dyn RemoteGitStore,
    artifact: &ResolvedRemoteArtifact,
    parts_root: &Path,
    package_path: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    writer: &mut dyn Write,
) -> Result<RemoteArtifactMaterializationReport, CacheError> {
    materialize_resolved_remote_artifact_inner(
        remote,
        artifact,
        parts_root,
        package_path,
        resources,
        cancellation,
        Some(writer),
    )
}

#[allow(clippy::too_many_arguments)]
fn materialize_resolved_remote_artifact_inner(
    remote: &dyn RemoteGitStore,
    artifact: &ResolvedRemoteArtifact,
    parts_root: &Path,
    package_path: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
    decoded_writer: Option<&mut dyn Write>,
) -> Result<RemoteArtifactMaterializationReport, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    artifact.manifest.validate()?;
    artifact.encoding.validate()?;
    artifact.receipt.validate()?;
    if parts_root.as_os_str().is_empty() || package_path.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "remote materialization requires explicit part-store and package paths".to_owned(),
        ));
    }
    let manifest_digest = artifact.manifest.digest()?;
    let transport_digest = artifact.encoding.digest()?;
    if manifest_digest != artifact.index.manifest_digest
        || artifact.manifest.artifact_family != artifact.family
        || artifact.manifest.semantic_digest != artifact.semantic_digest
        || artifact.manifest.payload_digest != artifact.index.canonical_payload_digest
        || artifact.encoding.canonical_payload_digest != artifact.manifest.payload_digest
        || !artifact
            .manifest
            .transport_digests
            .contains(&transport_digest)
        || artifact
            .receipt
            .artifact(&artifact.semantic_digest, &manifest_digest)
            .is_none_or(|proof| {
                proof.manifest_digest != manifest_digest
                    || proof.transport_digest != transport_digest
                    || proof.canonical_payload_digest != artifact.manifest.payload_digest
            })
    {
        return Err(CacheError::InvalidManifest(
            "resolved artifact identities are inconsistent before materialization".to_owned(),
        ));
    }

    let package_metadata = match fs::symlink_metadata(package_path) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let package_exists = package_metadata.is_some();
    let projected_new_local_bytes = if package_exists {
        0
    } else {
        projected_new_local_bytes(
            &artifact.encoding,
            parts_root,
            artifact.encoding.package_size_bytes,
        )?
    };
    if resources
        .maximum_permanent_disk_bytes
        .is_some_and(|maximum| projected_new_local_bytes > maximum)
    {
        return Err(CacheError::ResourceLimit(format!(
            "remote materialization projects {projected_new_local_bytes} new local bytes above the permanent-disk budget"
        )));
    }

    if package_exists {
        let _performance =
            xc_core::performance_stage_with("cache.remote.payload_verification", || {
                performance_metadata(artifact, "package-reuse")
            });
        let verification_started = Instant::now();
        let verification = match decoded_writer {
            Some(writer) => verify_canonical_payload_zip64_to_writer(
                &artifact.manifest.canonical_payload,
                &artifact.encoding,
                package_path,
                cancellation,
                false,
                writer,
            ),
            None => verify_canonical_payload_zip64(
                &artifact.manifest.canonical_payload,
                &artifact.encoding,
                package_path,
                cancellation,
            ),
        };
        let verification = match verification {
            Ok(verification) => verification,
            Err(error) => {
                // Only remove an ordinary retained file. Symlinks and other
                // special filesystem entries remain fail-closed and are never
                // followed or silently replaced. Removing a corrupt regular
                // package lets the next attempt reconstruct it from verified
                // parts instead of failing forever on the same bytes.
                let proves_corruption = matches!(
                    &error,
                    CacheError::DigestMismatch { .. } | CacheError::InvalidManifest(_)
                );
                if proves_corruption
                    && package_metadata.as_ref().is_some_and(|metadata| {
                        metadata.is_file() && !metadata.file_type().is_symlink()
                    })
                {
                    fs::remove_file(package_path)?;
                }
                return Err(error);
            }
        };
        return report(
            artifact,
            projected_new_local_bytes,
            true,
            None,
            DeterministicPackageReport {
                canonical_payload_digest: verification.canonical_payload_digest.clone(),
                encoder_profile: artifact.encoding.encoder_profile.clone(),
                package_size_bytes: verification.package_size_bytes,
                package_digest: verification.package_digest.clone(),
                package_path: package_path.to_owned(),
            },
            verification,
            MaterializationTimings {
                payload_verification_millis: elapsed_millis(verification_started.elapsed()),
                ..MaterializationTimings::default()
            },
        );
    }

    fs::create_dir_all(parts_root)?;
    let reader = RemoteShardReader::new(remote, 1)?;
    let mut part_fetch_millis = 0u64;
    let mut package_reconstruction_millis = 0u64;
    let mut repaired_reused_parts = false;
    let mut quarantined_reused_parts = QuarantineCleanup::default();
    let (mut part_fetch, package) = loop {
        let performance = xc_core::performance_stage_with("cache.remote.part_fetch", || {
            performance_metadata(artifact, "remote-or-part-reuse")
        });
        let part_fetch_started = Instant::now();
        let part_fetch = reader.fetch_transport_parts_for_reconstruction(
            &artifact.repository,
            &artifact.revision,
            &artifact.encoding,
            parts_root,
            resources,
            cancellation,
        )?;
        part_fetch_millis =
            part_fetch_millis.saturating_add(elapsed_millis(part_fetch_started.elapsed()));
        drop(performance);
        let performance =
            xc_core::performance_stage_with("cache.remote.package_reconstruction", || {
                performance_metadata(artifact, "part-reuse")
            });
        let package_started = Instant::now();
        let reconstructed = reconstruct_transport_package(
            &artifact.encoding,
            parts_root,
            package_path,
            resources,
            cancellation,
        );
        package_reconstruction_millis =
            package_reconstruction_millis.saturating_add(elapsed_millis(package_started.elapsed()));
        drop(performance);
        match reconstructed {
            Ok(package) => break (part_fetch, package),
            // Reused parts are digest-checked during the reconstruction copy.
            // A corrupt one is quarantined and fetched again exactly once;
            // a second failure, or a failure with no corrupt reused part,
            // is reported unchanged.
            Err(CacheError::DigestMismatch { expected, actual })
                if !repaired_reused_parts && !part_fetch.reused_sequences.is_empty() =>
            {
                let quarantined = quarantine_corrupt_reused_parts(
                    &artifact.encoding,
                    parts_root,
                    &part_fetch.reused_sequences,
                    resources,
                    cancellation,
                )?;
                if quarantined.is_empty() {
                    return Err(CacheError::DigestMismatch { expected, actual });
                }
                eprintln!(
                    "  cache transport: {} reused part(s) failed verification and were quarantined; fetching them again",
                    quarantined.len()
                );
                quarantined_reused_parts.absorb(quarantined);
                repaired_reused_parts = true;
            }
            Err(error) => return Err(error),
        }
    };
    // Downloaded parts were verified while streaming and reused parts were
    // verified during the one-pass reconstruction above. At this boundary the
    // report can truthfully account for the whole package as verified.
    part_fetch.verified_bytes = artifact.encoding.package_size_bytes;
    let performance = xc_core::performance_stage_with("cache.remote.payload_verification", || {
        performance_metadata(artifact, "reconstructed")
    });
    let verification_started = Instant::now();
    let verified = match decoded_writer {
        Some(writer) => verify_canonical_payload_zip64_to_writer(
            &artifact.manifest.canonical_payload,
            &artifact.encoding,
            package_path,
            cancellation,
            true,
            writer,
        ),
        None => {
            let mut sink = std::io::sink();
            verify_canonical_payload_zip64_to_writer(
                &artifact.manifest.canonical_payload,
                &artifact.encoding,
                package_path,
                cancellation,
                true,
                &mut sink,
            )
        }
    };
    let verification = match verified {
        Ok(verification) => verification,
        Err(error) => {
            if matches!(
                &error,
                CacheError::DigestMismatch { .. } | CacheError::InvalidManifest(_)
            ) {
                let _ = fs::remove_file(package_path);
            }
            return Err(error);
        }
    };
    let payload_verification_millis = elapsed_millis(verification_started.elapsed());
    drop(performance);
    quarantined_reused_parts.cleanup();
    report(
        artifact,
        projected_new_local_bytes,
        false,
        Some(part_fetch),
        package,
        verification,
        MaterializationTimings {
            part_fetch_millis,
            package_reconstruction_millis,
            payload_verification_millis,
        },
    )
}

fn report(
    artifact: &ResolvedRemoteArtifact,
    projected_new_local_bytes: u64,
    reused_verified_package: bool,
    part_fetch: Option<PartFetchReport>,
    package: DeterministicPackageReport,
    verification: VerifiedPackageReport,
    timings: MaterializationTimings,
) -> Result<RemoteArtifactMaterializationReport, CacheError> {
    let manifest_digest = artifact.manifest.digest()?;
    let transport_digest = artifact.encoding.digest()?;
    let package_path = package.package_path.clone();
    Ok(RemoteArtifactMaterializationReport {
        schema_version: 1,
        artifact_family: artifact.family.clone(),
        semantic_digest: artifact.semantic_digest.clone(),
        manifest_digest,
        canonical_payload_digest: artifact.manifest.payload_digest.clone(),
        transport_digest,
        repository: artifact.repository.clone(),
        revision: artifact.revision.clone(),
        package_path,
        projected_new_local_bytes,
        reused_verified_package,
        part_fetch,
        package,
        verification,
        part_fetch_elapsed_millis: timings.part_fetch_millis,
        package_reconstruction_elapsed_millis: timings.package_reconstruction_millis,
        payload_verification_elapsed_millis: timings.payload_verification_millis,
    })
}

fn projected_new_local_bytes(
    encoding: &crate::TransportEncodingRecord,
    parts_root: &Path,
    package_bytes: u64,
) -> Result<u64, CacheError> {
    let mut projected = package_bytes;
    for part in &encoding.ordered_parts {
        if !normalized_relative_path(&part.repository_path) {
            return Err(CacheError::InvalidManifest(format!(
                "transport part path {:?} is unsafe",
                part.repository_path
            )));
        }
        let path = part
            .repository_path
            .split('/')
            .fold(parts_root.to_owned(), |path, component| {
                path.join(component)
            });
        if !path.exists() {
            projected = projected.checked_add(part.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit(
                    "projected remote materialization size exceeds u64".to_owned(),
                )
            })?;
        }
    }
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        package_canonical_payload_zip64, stream_split_encoded, ArtifactAssuranceState,
        ArtifactDisposition, CacheVisibility, CanonicalArtifactManifest, CanonicalPayloadEnvelope,
        CompareAndSwapResult, LogicalPayloadItem, PayloadFileSource, PublicationDestination,
        PublicationReceipt, RemoteCommitRequest, RemoteReadReport, SemanticKeyEnvelope,
        ShardIndexEntry, ToolkitVersion, TransportPart, TransportPolicy,
        DETERMINISTIC_ZIP64_PROFILE_V1,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use xc_core::{AssuranceLevel, CancellationReason, PublicationAuthorityMode};

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
            self.reads.fetch_add(1, Ordering::Relaxed);
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

    fn temporary_root() -> PathBuf {
        std::env::temp_dir().join(format!("xc-cache-materialization-{}", std::process::id()))
    }

    #[test]
    fn selective_materialization_reuses_parts_and_verified_package() {
        let root = temporary_root();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source_path = root.join("source.bin");
        let logical_bytes = b"canonical decoded payload";
        fs::write(&source_path, logical_bytes).unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "bytes".to_owned(),
            dimensions: vec![logical_bytes.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(logical_bytes),
                size_bytes: logical_bytes.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let encoded_path = root.join("encoded.zip");
        package_canonical_payload_zip64(
            &canonical_payload,
            &[PayloadFileSource {
                normalized_path: "value.bin".to_owned(),
                source_path,
            }],
            &encoded_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let encoded = fs::read(encoded_path).unwrap();
        let mut remote_paths = BTreeMap::new();
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
            |part, bytes| {
                remote_paths.insert(part.repository_path.clone(), bytes.to_vec());
                Ok(())
            },
        )
        .unwrap();
        let transport_digest = encoding.digest().unwrap();
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "materialization_fixture".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: json!({"case": 1}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "fixture".to_owned(),
            semantic_key,
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![transport_digest.clone()],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "materialization fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let transaction_id = ContentDigest::sha256(b"transaction").0;
        let index = ShardIndexEntry {
            semantic_digest: semantic_digest.clone(),
            canonical_payload_digest: payload_digest.clone(),
            manifest_digest: manifest_digest.clone(),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
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
            authority_mode: PublicationAuthorityMode::OwnerDirect,
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
            remote_verification_results: vec![crate::RemoteCommitVerificationResult {
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
        let artifact = ResolvedRemoteArtifact {
            family: "fixture".to_owned(),
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
            receipt: crate::RemotePublicationEvidence::ArtifactReceipt(Box::new(receipt)),
            index_source: source.clone(),
            manifest_source: source.clone(),
            encoding_source: source.clone(),
            receipt_source: source,
            dependencies: Vec::new(),
        };
        let mut remote = MemoryRemote {
            revision: artifact.revision.clone(),
            paths: remote_paths,
            reads: AtomicUsize::new(0),
        };
        let parts_root = root.join("parts");
        let package_path = root.join("materialized.zip");
        let first = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(!first.reused_verified_package);
        let serialized = serde_json::to_value(&first).unwrap();
        for operational_only in [
            "part_fetch_elapsed_millis",
            "package_reconstruction_elapsed_millis",
            "payload_verification_elapsed_millis",
        ] {
            assert!(
                serialized.get(operational_only).is_none(),
                "schema-v1 JSON must not gain timing field {operational_only}"
            );
        }
        assert_eq!(
            first
                .part_fetch
                .as_ref()
                .unwrap()
                .downloaded_sequences
                .len(),
            artifact.encoding.ordered_parts.len()
        );
        let reads = remote.reads.load(Ordering::Relaxed);
        let second = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(second.reused_verified_package);
        assert!(second.part_fetch.is_none());
        assert_eq!(remote.reads.load(Ordering::Relaxed), reads);
        assert_eq!(
            second.verification.logical_size_bytes,
            logical_bytes.len() as u64
        );

        // Cancellation is not evidence of corruption. A Ctrl-C during a warm
        // verification must leave the valid immutable package available for
        // the next run.
        let cancelled = CancellationToken::new();
        cancelled.cancel(CancellationReason::UserRequested);
        let error = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &cancelled,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::Cancelled(_)));
        assert!(package_path.is_file());

        // A corrupt retained package is removed after the failing
        // verification. The next call can then rebuild it from its verified
        // retained parts without downloading them again.
        let mut corrupt_package = fs::read(&package_path).unwrap();
        corrupt_package[0] ^= 1;
        fs::write(&package_path, corrupt_package).unwrap();
        let reads_before_package_repair = remote.reads.load(Ordering::Relaxed);
        let error = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::DigestMismatch { .. }));
        assert!(
            !package_path.exists(),
            "the corrupt ordinary package must not wedge later attempts"
        );
        let rebuilt = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(!rebuilt.reused_verified_package);
        assert_eq!(
            remote.reads.load(Ordering::Relaxed),
            reads_before_package_repair,
            "retained verified parts repair the package without network I/O"
        );

        // The same rule applies when cancellation arrives during validation
        // immediately after a package was reconstructed from verified parts.
        // The package is complete and valid, so the next run may reuse it.
        struct CancellingWriter<'a>(&'a CancellationToken);
        impl std::io::Write for CancellingWriter<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.cancel(CancellationReason::UserRequested);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        fs::remove_file(&package_path).unwrap();
        let cancellation = CancellationToken::new();
        let mut writer = CancellingWriter(&cancellation);
        let error = materialize_resolved_remote_artifact_to_writer(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &cancellation,
            &mut writer,
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::Cancelled(_)));
        assert!(
            package_path.is_file(),
            "cancellation after reconstruction must retain the complete package"
        );

        let mut dependency = artifact.clone();
        dependency
            .manifest
            .semantic_key
            .resolved_mathematical_parameters = json!({"case": "dependency"});
        dependency.semantic_digest = dependency.manifest.semantic_key.digest().unwrap();
        dependency.manifest.semantic_digest = dependency.semantic_digest.clone();
        dependency.dependencies.clear();
        let dependency_identity = crate::PayloadDependencyIdentity {
            artifact_family: dependency.family.clone(),
            semantic_digest: dependency.semantic_digest.clone(),
            manifest_digest: dependency.manifest.digest().unwrap(),
            payload_digest: dependency.manifest.payload_digest.clone(),
        };
        let mut root_artifact = artifact.clone();
        root_artifact.manifest.canonical_payload.dependencies = vec![dependency_identity];
        root_artifact.manifest.payload_digest =
            root_artifact.manifest.canonical_payload.digest().unwrap();
        root_artifact.dependencies = vec![dependency];
        let mut ordered = Vec::new();
        collect_dependency_first(
            &root_artifact,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
            &mut ordered,
        )
        .unwrap();
        assert_eq!(ordered.len(), 2);
        assert_eq!(
            ordered[0].semantic_digest,
            root_artifact.dependencies[0].semantic_digest
        );
        assert_eq!(ordered[1].semantic_digest, root_artifact.semantic_digest);

        root_artifact.dependencies[0].family = "substituted-family".to_owned();
        let error = collect_dependency_first(
            &root_artifact,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::InvalidManifest(_)));

        // Reused parts deliberately skip a preliminary full-file hash. Their
        // digests are still checked while bytes are copied into the package.
        // A corrupt reused part is quarantined and fetched again exactly
        // once, so the run succeeds with a freshly downloaded copy instead
        // of failing identically on every attempt.
        fs::remove_file(&package_path).unwrap();
        let corrupt_part = &artifact.encoding.ordered_parts[0];
        let corrupt_path = corrupt_part
            .repository_path
            .split('/')
            .fold(parts_root.clone(), |path, component| path.join(component));
        let mut corrupt_bytes = fs::read(&corrupt_path).unwrap();
        corrupt_bytes[0] ^= 1;
        fs::write(&corrupt_path, &corrupt_bytes).unwrap();
        let reads_before_repair = remote.reads.load(Ordering::Relaxed);
        let repaired = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(package_path.exists());
        assert_eq!(
            remote.reads.load(Ordering::Relaxed),
            reads_before_repair + 1,
            "exactly the corrupt part is downloaded again"
        );
        assert_eq!(
            repaired.part_fetch.as_ref().unwrap().downloaded_sequences,
            vec![corrupt_part.sequence]
        );
        assert_eq!(
            fs::read(&corrupt_path).unwrap(),
            remote.paths[&corrupt_part.repository_path],
            "the repaired part is the remote's bytes"
        );
        let quarantined = fs::read_dir(corrupt_path.parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
            .count();
        assert_eq!(
            quarantined, 0,
            "a successfully repaired part must not leave unaccounted quarantine bytes"
        );

        // A retained part of the wrong size is corruption too. It is
        // quarantined at fetch time and downloaded again, before any
        // reconstruction is attempted.
        fs::remove_file(&package_path).unwrap();
        let truncated_part = &artifact.encoding.ordered_parts[1];
        let truncated_path = truncated_part
            .repository_path
            .split('/')
            .fold(parts_root.clone(), |path, component| path.join(component));
        let intact = fs::read(&truncated_path).unwrap();
        fs::write(&truncated_path, &intact[..intact.len() - 1]).unwrap();
        let reads_before_size_repair = remote.reads.load(Ordering::Relaxed);
        let repaired = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(package_path.exists());
        assert_eq!(
            remote.reads.load(Ordering::Relaxed),
            reads_before_size_repair + 1
        );
        assert_eq!(
            repaired.part_fetch.as_ref().unwrap().downloaded_sequences,
            vec![truncated_part.sequence]
        );
        assert_eq!(fs::read(&truncated_path).unwrap(), intact);
        let quarantined_after_size_repair = fs::read_dir(truncated_path.parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
            .count();
        assert_eq!(quarantined_after_size_repair, 0);

        // When the replacement download is corrupt as well, the run fails
        // closed without leaving a visible package.
        fs::remove_file(&package_path).unwrap();
        fs::write(&corrupt_path, &corrupt_bytes).unwrap();
        remote
            .paths
            .insert(corrupt_part.repository_path.clone(), corrupt_bytes);
        let error = materialize_resolved_remote_artifact(
            &remote,
            &artifact,
            &parts_root,
            &package_path,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CacheError::DigestMismatch { .. }));
        assert!(!package_path.exists());
        assert_eq!(
            fs::read_dir(corrupt_path.parent().unwrap())
                .unwrap()
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
                .count(),
            0,
            "failed repair must not leak quarantine files outside accounting"
        );
        let _ = fs::remove_dir_all(root);
    }
}
