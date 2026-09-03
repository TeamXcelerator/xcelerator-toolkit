//! Deterministic ZIP/ZIP64 packaging for canonical logical payloads.

use crate::protocol::{normalized_relative_path, CanonicalPayloadEnvelope, LogicalPayloadItem};
use crate::{
    stream_split_encoded, CacheError, ContentDigest, TransportEncodingRecord, TransportPart,
    TransportPolicy,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xc_core::{CancellationToken, ResourcePolicy};
use zip::result::ZipError;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, DateTime, ZipWriter};

/// Deterministic profile used by the historical and current single-entry
/// canonical-payload encoder. That encoder has always requested ZIP64 metadata
/// for its one entry, so retaining V1 preserves every published artifact
/// transport identity. Historical multi-item writers also used this label but
/// were entry-size-aware; those already-published transports remain readable.
pub const DETERMINISTIC_ZIP64_PROFILE_V1: &str = concat!(
    "xc-zip64-deflate-v1;zip-rs=2.4.2;flate2=1.1.9;miniz_oxide=0.8.9;",
    "level=6;mtime=1980-01-01T00:00:00;mode=0644;order=utf8-bytewise"
);

/// Deterministic profile for newly created multi-item packages. Every entry
/// uses ZIP64 metadata, including small entries, removing the historical
/// multi-item writer ambiguity without relabeling unchanged single-entry
/// artifact packages.
pub const DETERMINISTIC_ZIP64_PROFILE_V2: &str = concat!(
    "xc-zip64-deflate-v2;zip-rs=2.4.2;flate2=1.1.9;miniz_oxide=0.8.9;",
    "level=6;mtime=1980-01-01T00:00:00;mode=0644;order=utf8-bytewise;zip64=always"
);

/// Current profile for the single-entry canonical artifact and workstation
/// object encoders. Kept as the original public name for API compatibility.
pub const CURRENT_DETERMINISTIC_ZIP64_PROFILE: &str = DETERMINISTIC_ZIP64_PROFILE_V1;

/// Current profile for file-backed, potentially multi-item packages.
pub const CURRENT_MULTI_ITEM_ZIP64_PROFILE: &str = DETERMINISTIC_ZIP64_PROFILE_V2;

const COPY_BUFFER_BYTES: u64 = 1024 * 1024;
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadFileSource {
    pub normalized_path: String,
    pub source_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DeterministicPackageReport {
    pub canonical_payload_digest: ContentDigest,
    pub encoder_profile: String,
    pub package_size_bytes: u64,
    pub package_digest: ContentDigest,
    /// Local operational state; this path is not part of artifact identity.
    pub package_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VerifiedPackageReport {
    pub canonical_payload_digest: ContentDigest,
    pub package_digest: ContentDigest,
    pub package_size_bytes: u64,
    pub logical_size_bytes: u64,
    pub item_count: usize,
}

pub struct StreamingPackageSplitRequest<'a> {
    pub envelope: &'a CanonicalPayloadEnvelope,
    pub sources: &'a [PayloadFileSource],
    pub temporary_archive_path: &'a Path,
    pub transport_policy: &'a TransportPolicy,
    pub resources: &'a ResourcePolicy,
    pub cancellation: &'a CancellationToken,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingPackageSplitReport {
    pub schema_version: u32,
    pub encoding: TransportEncodingRecord,
    pub temporary_archive_bytes: u64,
    pub maximum_in_flight_part_bytes: u64,
    pub maximum_in_flight_parts: usize,
    pub complete_archive_retained: bool,
    pub split_parts_retained_by_packager: bool,
    pub cleanup_complete: bool,
}

/// Encode one deterministic archive, synchronously hand each bounded part to
/// an upload/checkpoint sink, and remove the archive on every exit path.
/// The packager never retains split parts, so callers can publish directly
/// with at most one in-flight part instead of materializing a second full copy.
pub fn package_and_stream_split_zip64<F>(
    request: StreamingPackageSplitRequest<'_>,
    mut sink: F,
) -> Result<StreamingPackageSplitReport, CacheError>
where
    F: FnMut(&TransportPart, &[u8]) -> Result<(), CacheError>,
{
    if request.temporary_archive_path.as_os_str().is_empty()
        || request.temporary_archive_path.exists()
    {
        return Err(CacheError::InvalidManifest(
            "streaming package archive path must be explicit and absent".to_owned(),
        ));
    }
    let result = (|| {
        let package = package_canonical_payload_zip64(
            request.envelope,
            request.sources,
            request.temporary_archive_path,
            request.resources,
            request.cancellation,
        )?;
        let mut archive = BufReader::new(File::open(request.temporary_archive_path)?);
        let encoding = stream_split_encoded(
            &mut archive,
            package.canonical_payload_digest,
            package.encoder_profile,
            request.transport_policy,
            request.resources,
            request.cancellation,
            |part, bytes| {
                request
                    .cancellation
                    .check()
                    .map_err(|error| CacheError::Cancelled(error.to_string()))?;
                sink(part, bytes)
            },
        )?;
        Ok(StreamingPackageSplitReport {
            schema_version: 1,
            temporary_archive_bytes: package.package_size_bytes,
            maximum_in_flight_part_bytes: request.transport_policy.split_part_bytes,
            maximum_in_flight_parts: 1,
            encoding,
            complete_archive_retained: false,
            split_parts_retained_by_packager: false,
            cleanup_complete: true,
        })
    })();
    let cleanup = match fs::remove_file(request.temporary_archive_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(CacheError::Io(format!(
            "could not remove streaming archive {}: {error}",
            request.temporary_archive_path.display()
        ))),
    };
    match (result, cleanup) {
        (Ok(mut report), Ok(())) => {
            report.cleanup_complete = !request.temporary_archive_path.exists();
            Ok(report)
        }
        (_, Err(error)) => Err(error),
        (Err(error), Ok(())) => Err(error),
    }
}

/// Package the files named by a canonical payload envelope into an immutable,
/// atomically visible ZIP/ZIP64 file. Source bytes are hashed while they are
/// encoded and must match the envelope exactly.
pub fn package_canonical_payload_zip64(
    envelope: &CanonicalPayloadEnvelope,
    sources: &[PayloadFileSource],
    destination: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    envelope.validate()?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if destination.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "package destination must be explicit".to_owned(),
        ));
    }
    if destination.exists() {
        return Err(CacheError::InvalidManifest(format!(
            "immutable package destination already exists: {}",
            destination.display()
        )));
    }
    let conservative_package_bytes = conservative_zip_upper_bound(envelope)?;
    if resources
        .maximum_temporary_disk_bytes
        .is_some_and(|maximum| conservative_package_bytes > maximum)
    {
        return Err(CacheError::ResourceLimit(format!(
            "deterministic package may require {conservative_package_bytes} temporary bytes"
        )));
    }

    let resolved_sources = resolve_sources(envelope, sources)?;
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let (temporary_path, output) = create_temporary_file(destination)?;
    let result = package_inner(
        envelope,
        &resolved_sources,
        destination,
        &temporary_path,
        output,
        resources,
        cancellation,
    );
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

/// The one single-entry ZIP encoder behind every byte that can be published.
///
/// The workstation ZIP object store and the deterministic publication
/// packager both call this, so a verified local object is byte-for-byte the
/// package that publication would otherwise re-encode, and staging may split
/// it directly. Any byte-affecting change here must also change
/// `CURRENT_DETERMINISTIC_ZIP64_PROFILE`.
pub(crate) fn write_deterministic_zip_entry<W: Write + Seek>(
    output: W,
    normalized_path: &str,
    bytes: &[u8],
) -> Result<W, CacheError> {
    let mut writer = ZipWriter::new(output);
    let options = deterministic_options();
    writer
        .start_file(normalized_path, options)
        .map_err(|error| CacheError::Io(error.to_string()))?;
    writer.write_all(bytes)?;
    writer
        .finish()
        .map_err(|error| CacheError::Io(error.to_string()))
}

/// Deterministically package one already-canonical logical byte stream without
/// first materializing it as an uncompressed filesystem file.
pub fn package_canonical_payload_bytes_zip64(
    envelope: &CanonicalPayloadEnvelope,
    normalized_path: &str,
    bytes: &[u8],
    destination: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    envelope.validate()?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if envelope.ordered_items.len() != 1
        || envelope.ordered_items[0].normalized_path != normalized_path
        || envelope.ordered_items[0].content_digest != ContentDigest::sha256(bytes)
        || envelope.ordered_items[0].size_bytes != bytes.len() as u64
        || destination.as_os_str().is_empty()
        || destination.exists()
    {
        return Err(CacheError::InvalidManifest(
            "canonical byte package source does not match its payload envelope".to_owned(),
        ));
    }
    let conservative_package_bytes = conservative_zip_upper_bound(envelope)?;
    if resources
        .maximum_temporary_disk_bytes
        .is_some_and(|maximum| conservative_package_bytes > maximum)
    {
        return Err(CacheError::ResourceLimit(format!(
            "deterministic package may require {conservative_package_bytes} temporary bytes"
        )));
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let (temporary_path, output) = create_temporary_file(destination)?;
    let result = (|| {
        let mut output = write_deterministic_zip_entry(output, normalized_path, bytes)?;
        output.flush()?;
        output.sync_all()?;
        drop(output);
        let package_size_bytes = fs::metadata(&temporary_path)?.len();
        let mut digest_buffer = vec![0u8; COPY_BUFFER_BYTES as usize];
        let package_digest = digest_file(&temporary_path, cancellation, &mut digest_buffer)?;
        publish_immutable_package(
            &temporary_path,
            destination,
            &package_digest,
            package_size_bytes,
            cancellation,
        )?;
        Ok(DeterministicPackageReport {
            canonical_payload_digest: envelope.digest()?,
            encoder_profile: CURRENT_DETERMINISTIC_ZIP64_PROFILE.to_owned(),
            package_size_bytes,
            package_digest,
            package_path: destination.to_path_buf(),
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

/// Reconstruct a package only from the record's authoritative ordered part
/// list. Each part and the whole package are verified before the destination
/// becomes visible.
pub fn reconstruct_transport_package(
    record: &crate::TransportEncodingRecord,
    parts_root: &Path,
    destination: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    reconstruct_transport_package_inner(record, parts_root, destination, resources, cancellation)
}

fn reconstruct_transport_package_inner(
    record: &crate::TransportEncodingRecord,
    parts_root: &Path,
    destination: &Path,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    record.validate()?;
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    if destination.as_os_str().is_empty() {
        return Err(CacheError::InvalidManifest(
            "reconstructed package destination must be explicit".to_owned(),
        ));
    }
    match fs::symlink_metadata(destination) {
        Ok(_) => return verified_existing_transport_package(record, destination, cancellation),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    for (description, maximum) in [
        ("temporary-disk", resources.maximum_temporary_disk_bytes),
        ("network-transfer", resources.maximum_transfer_bytes),
    ] {
        if maximum.is_some_and(|maximum| record.package_size_bytes > maximum) {
            return Err(CacheError::ResourceLimit(format!(
                "package reconstruction requires {} bytes above the {description} budget",
                record.package_size_bytes
            )));
        }
    }

    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let (temporary_path, output) = create_temporary_file(destination)?;
    let result = reconstruct_inner(
        record,
        parts_root,
        destination,
        &temporary_path,
        output,
        resources,
        cancellation,
    );
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_inner(
    record: &crate::TransportEncodingRecord,
    parts_root: &Path,
    destination: &Path,
    temporary_path: &Path,
    output: File,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    let buffer_bytes = resources
        .maximum_memory_bytes
        .unwrap_or(COPY_BUFFER_BYTES)
        .clamp(1, COPY_BUFFER_BYTES);
    let buffer_bytes = usize::try_from(buffer_bytes).map_err(|_| {
        CacheError::ResourceLimit("reconstruction buffer does not fit this platform".to_owned())
    })?;
    let mut buffer = vec![0u8; buffer_bytes];
    let mut output = ControlledFile::new(output, resources.maximum_temporary_disk_bytes);
    let mut package_hasher = Sha256::new();
    let mut package_size = 0u64;

    for part in &record.ordered_parts {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let path = resolve_part_path(parts_root, &part.repository_path)?;
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                CacheError::NotFound(format!("transport part {}", part.repository_path))
            } else {
                error.into()
            }
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(CacheError::InvalidManifest(format!(
                "transport part {:?} must be a regular non-symlink file",
                part.repository_path
            )));
        }
        if metadata.len() != part.size_bytes {
            return Err(CacheError::DigestMismatch {
                expected: format!("{} bytes", part.size_bytes),
                actual: format!("{} bytes", metadata.len()),
            });
        }
        let mut input = BufReader::new(File::open(&path)?);
        let mut part_hasher = Sha256::new();
        let mut part_size = 0u64;
        loop {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            part_size = part_size.saturating_add(read as u64);
            part_hasher.update(&buffer[..read]);
            package_hasher.update(&buffer[..read]);
            output
                .write_all(&buffer[..read])
                .map_err(|error| map_io_error(error, cancellation))?;
        }
        let part_digest = ContentDigest(format!("{:x}", part_hasher.finalize()));
        if part_size != part.size_bytes || part_digest != part.content_digest {
            return Err(CacheError::DigestMismatch {
                expected: format!("{} ({} bytes)", part.content_digest, part.size_bytes),
                actual: format!("{part_digest} ({part_size} bytes)"),
            });
        }
        package_size = package_size.saturating_add(part_size);
    }
    output.flush()?;
    output.inner.sync_all()?;
    drop(output);
    let package_digest = ContentDigest(format!("{:x}", package_hasher.finalize()));
    if package_size != record.package_size_bytes || package_digest != record.package_digest {
        return Err(CacheError::DigestMismatch {
            expected: format!(
                "{} ({} bytes)",
                record.package_digest, record.package_size_bytes
            ),
            actual: format!("{package_digest} ({package_size} bytes)"),
        });
    }
    match fs::hard_link(temporary_path, destination) {
        Ok(()) => {
            let _ = fs::remove_file(temporary_path);
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(temporary_path);
            return verified_existing_transport_package(record, destination, cancellation);
        }
        Err(error) => return Err(error.into()),
    }
    Ok(DeterministicPackageReport {
        canonical_payload_digest: record.canonical_payload_digest.clone(),
        encoder_profile: record.encoder_profile.clone(),
        package_size_bytes: package_size,
        package_digest,
        package_path: destination.to_owned(),
    })
}

/// Verify the package transport identity and every decoded logical item.
pub fn verify_canonical_payload_zip64(
    envelope: &CanonicalPayloadEnvelope,
    record: &crate::TransportEncodingRecord,
    package_path: &Path,
    cancellation: &CancellationToken,
) -> Result<VerifiedPackageReport, CacheError> {
    verify_canonical_payload_zip64_inner(envelope, record, package_path, cancellation, false, None)
}

pub(crate) fn verify_canonical_payload_zip64_to_writer(
    envelope: &CanonicalPayloadEnvelope,
    record: &crate::TransportEncodingRecord,
    package_path: &Path,
    cancellation: &CancellationToken,
    package_digest_preverified: bool,
    writer: &mut dyn Write,
) -> Result<VerifiedPackageReport, CacheError> {
    verify_canonical_payload_zip64_inner(
        envelope,
        record,
        package_path,
        cancellation,
        package_digest_preverified,
        Some(writer),
    )
}

fn verify_canonical_payload_zip64_inner(
    envelope: &CanonicalPayloadEnvelope,
    record: &crate::TransportEncodingRecord,
    package_path: &Path,
    cancellation: &CancellationToken,
    package_digest_preverified: bool,
    mut decoded_writer: Option<&mut dyn Write>,
) -> Result<VerifiedPackageReport, CacheError> {
    envelope.validate()?;
    record.validate()?;
    if !matches!(
        record.encoder_profile.as_str(),
        DETERMINISTIC_ZIP64_PROFILE_V1 | DETERMINISTIC_ZIP64_PROFILE_V2
    ) {
        return Err(CacheError::InvalidManifest(format!(
            "unsupported deterministic ZIP profile {:?}",
            record.encoder_profile
        )));
    }
    let canonical_payload_digest = envelope.digest()?;
    if canonical_payload_digest != record.canonical_payload_digest {
        return Err(CacheError::DigestMismatch {
            expected: record.canonical_payload_digest.to_string(),
            actual: canonical_payload_digest.to_string(),
        });
    }
    let metadata = fs::symlink_metadata(package_path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != record.package_size_bytes
    {
        return Err(CacheError::InvalidManifest(
            "verified package must be a regular file with the recorded size".to_owned(),
        ));
    }
    let mut digest_buffer = vec![0u8; COPY_BUFFER_BYTES as usize];
    let package_digest = if package_digest_preverified {
        record.package_digest.clone()
    } else {
        let digest = digest_file(package_path, cancellation, &mut digest_buffer)?;
        if digest != record.package_digest {
            return Err(CacheError::DigestMismatch {
                expected: record.package_digest.to_string(),
                actual: digest.to_string(),
            });
        }
        digest
    };

    let input = File::open(package_path)?;
    let mut archive =
        zip::ZipArchive::new(input).map_err(|error| map_zip_error(error, cancellation))?;
    if archive.len() != envelope.ordered_items.len() {
        return Err(CacheError::InvalidManifest(format!(
            "ZIP contains {} entries but canonical payload declares {}",
            archive.len(),
            envelope.ordered_items.len()
        )));
    }
    let fixed_time =
        DateTime::from_date_and_time(1980, 1, 1, 0, 0, 0).expect("the fixed ZIP epoch is valid");
    for (index, item) in envelope.ordered_items.iter().enumerate() {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let mut entry = archive
            .by_index(index)
            .map_err(|error| map_zip_error(error, cancellation))?;
        if entry.name() != item.normalized_path
            || entry.size() != item.size_bytes
            || entry.compression() != CompressionMethod::Deflated
            || entry.last_modified() != Some(fixed_time)
            || entry.unix_mode() != Some(0o100644)
            || (record.encoder_profile == DETERMINISTIC_ZIP64_PROFILE_V2
                && !zip_local_header_uses_zip64(package_path, entry.header_start())?)
        {
            return Err(CacheError::InvalidManifest(format!(
                "ZIP entry {index} metadata does not match canonical profile"
            )));
        }
        let mut hasher = Sha256::new();
        let mut size = 0u64;
        loop {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let read = entry.read(&mut digest_buffer)?;
            if read == 0 {
                break;
            }
            size = size.saturating_add(read as u64);
            hasher.update(&digest_buffer[..read]);
            if let Some(writer) = decoded_writer.as_deref_mut() {
                writer.write_all(&digest_buffer[..read])?;
            }
        }
        let digest = ContentDigest(format!("{:x}", hasher.finalize()));
        if size != item.size_bytes || digest != item.content_digest {
            return Err(CacheError::DigestMismatch {
                expected: format!("{} ({} bytes)", item.content_digest, item.size_bytes),
                actual: format!("{digest} ({size} bytes)"),
            });
        }
    }
    Ok(VerifiedPackageReport {
        canonical_payload_digest,
        package_digest,
        package_size_bytes: record.package_size_bytes,
        logical_size_bytes: envelope.logical_size_bytes(),
        item_count: envelope.ordered_items.len(),
    })
}

fn zip_local_header_uses_zip64(path: &Path, header_start: u64) -> Result<bool, CacheError> {
    let mut input = File::open(path)?;
    input.seek(SeekFrom::Start(header_start))?;
    let mut header = [0u8; 30];
    input.read_exact(&mut header)?;
    if header[0..4] != [0x50, 0x4b, 0x03, 0x04] {
        return Ok(false);
    }
    let version_needed = u16::from_le_bytes([header[4], header[5]]);
    let compressed_size = u32::from_le_bytes([header[18], header[19], header[20], header[21]]);
    let uncompressed_size = u32::from_le_bytes([header[22], header[23], header[24], header[25]]);
    let name_size = u16::from_le_bytes([header[26], header[27]]);
    let extra_size = u16::from_le_bytes([header[28], header[29]]) as usize;
    input.seek(SeekFrom::Current(i64::from(name_size)))?;
    let mut extra = vec![0u8; extra_size];
    input.read_exact(&mut extra)?;
    let mut raw = extra.as_slice();
    while raw.len() >= 4 {
        let id = u16::from_le_bytes([raw[0], raw[1]]);
        let size = u16::from_le_bytes([raw[2], raw[3]]) as usize;
        raw = &raw[4..];
        if size > raw.len() {
            return Ok(false);
        }
        if id == 0x0001 {
            return Ok(version_needed >= 45
                && compressed_size == u32::MAX
                && uncompressed_size == u32::MAX);
        }
        raw = &raw[size..];
    }
    Ok(false)
}

fn resolve_part_path(root: &Path, repository_path: &str) -> Result<PathBuf, CacheError> {
    if !normalized_relative_path(repository_path) {
        return Err(CacheError::InvalidManifest(format!(
            "transport part path {repository_path:?} is unsafe"
        )));
    }
    Ok(repository_path
        .split('/')
        .fold(root.to_owned(), |path, component| path.join(component)))
}

fn resolve_sources<'a>(
    envelope: &'a CanonicalPayloadEnvelope,
    sources: &'a [PayloadFileSource],
) -> Result<Vec<(&'a LogicalPayloadItem, &'a Path)>, CacheError> {
    let mut by_path = BTreeMap::new();
    for source in sources {
        if !normalized_relative_path(&source.normalized_path)
            || by_path
                .insert(
                    source.normalized_path.as_str(),
                    source.source_path.as_path(),
                )
                .is_some()
        {
            return Err(CacheError::InvalidManifest(format!(
                "invalid or duplicate package source path {:?}",
                source.normalized_path
            )));
        }
    }
    if by_path.len() != envelope.ordered_items.len() {
        return Err(CacheError::InvalidManifest(
            "package sources must exactly match canonical payload items".to_owned(),
        ));
    }
    envelope
        .ordered_items
        .iter()
        .map(|item| {
            let source = by_path.get(item.normalized_path.as_str()).ok_or_else(|| {
                CacheError::InvalidManifest(format!(
                    "canonical payload item {:?} has no package source",
                    item.normalized_path
                ))
            })?;
            let metadata = fs::symlink_metadata(source)?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(CacheError::InvalidManifest(format!(
                    "package source for {:?} must be a regular non-symlink file",
                    item.normalized_path
                )));
            }
            Ok((item, *source))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn package_inner(
    envelope: &CanonicalPayloadEnvelope,
    sources: &[(&LogicalPayloadItem, &Path)],
    destination: &Path,
    temporary_path: &Path,
    output: File,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    let controlled = ControlledFile::new(output, resources.maximum_temporary_disk_bytes);
    let mut archive = ZipWriter::new(controlled);
    let buffer_bytes = resources
        .maximum_memory_bytes
        .unwrap_or(COPY_BUFFER_BYTES)
        .clamp(1, COPY_BUFFER_BYTES);
    let buffer_bytes = usize::try_from(buffer_bytes).map_err(|_| {
        CacheError::ResourceLimit("package buffer does not fit this platform".to_owned())
    })?;
    let mut buffer = vec![0u8; buffer_bytes];

    for (item, source_path) in sources {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let options = deterministic_options();
        archive
            .start_file(&item.normalized_path, options)
            .map_err(|error| map_zip_error(error, cancellation))?;
        let source = File::open(source_path)?;
        let mut source = BufReader::new(source);
        let mut item_hasher = Sha256::new();
        let mut item_size = 0u64;
        loop {
            cancellation
                .check()
                .map_err(|error| CacheError::Cancelled(error.to_string()))?;
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            item_size = item_size.checked_add(read as u64).ok_or_else(|| {
                CacheError::ResourceLimit("logical payload item exceeds u64".to_owned())
            })?;
            item_hasher.update(&buffer[..read]);
            archive
                .write_all(&buffer[..read])
                .map_err(|error| map_io_error(error, cancellation))?;
        }
        let actual_digest = ContentDigest(format!("{:x}", item_hasher.finalize()));
        if item_size != item.size_bytes || actual_digest != item.content_digest {
            return Err(CacheError::DigestMismatch {
                expected: format!("{} ({} bytes)", item.content_digest, item.size_bytes),
                actual: format!("{actual_digest} ({item_size} bytes)"),
            });
        }
    }

    let controlled = archive
        .finish()
        .map_err(|error| map_zip_error(error, cancellation))?;
    controlled.inner.sync_all()?;
    let package_size_bytes = controlled.length;
    drop(controlled);
    let package_digest = digest_file(temporary_path, cancellation, &mut buffer)?;

    // Hard linking creates the immutable destination only if it is still
    // absent, avoiding a concurrent overwrite. The temporary name is in the
    // same directory so this is an atomic same-filesystem operation.
    publish_immutable_package(
        temporary_path,
        destination,
        &package_digest,
        package_size_bytes,
        cancellation,
    )?;

    let encoder_profile = if envelope.ordered_items.len() == 1 {
        CURRENT_DETERMINISTIC_ZIP64_PROFILE
    } else {
        CURRENT_MULTI_ITEM_ZIP64_PROFILE
    };
    Ok(DeterministicPackageReport {
        canonical_payload_digest: envelope.digest()?,
        encoder_profile: encoder_profile.to_owned(),
        package_size_bytes,
        package_digest,
        package_path: destination.to_owned(),
    })
}

fn deterministic_options() -> SimpleFileOptions {
    SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(6))
        .last_modified_time(
            DateTime::from_date_and_time(1980, 1, 1, 0, 0, 0)
                .expect("the fixed ZIP epoch is valid"),
        )
        .unix_permissions(0o644)
        .large_file(true)
}

fn publish_immutable_package(
    temporary_path: &Path,
    destination: &Path,
    expected_digest: &ContentDigest,
    expected_size: u64,
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    match fs::hard_link(temporary_path, destination) {
        Ok(()) => {
            let _ = fs::remove_file(temporary_path);
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = fs::symlink_metadata(destination)?;
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() != expected_size
            {
                let _ = fs::remove_file(temporary_path);
                return Err(CacheError::InvalidManifest(
                    "existing immutable package has invalid filesystem metadata".to_owned(),
                ));
            }
            let mut buffer = vec![0u8; COPY_BUFFER_BYTES as usize];
            let digest = digest_file(destination, cancellation, &mut buffer)?;
            let size = fs::symlink_metadata(destination)?.len();
            let _ = fs::remove_file(temporary_path);
            if &digest != expected_digest || size != expected_size {
                return Err(CacheError::DigestMismatch {
                    expected: format!("{expected_digest} ({expected_size} bytes)"),
                    actual: format!("{digest} ({size} bytes)"),
                });
            }
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn verified_existing_transport_package(
    record: &TransportEncodingRecord,
    destination: &Path,
    cancellation: &CancellationToken,
) -> Result<DeterministicPackageReport, CacheError> {
    let metadata = fs::symlink_metadata(destination)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != record.package_size_bytes
    {
        return Err(CacheError::InvalidManifest(
            "existing reconstructed package has invalid filesystem metadata".to_owned(),
        ));
    }
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES as usize];
    let digest = digest_file(destination, cancellation, &mut buffer)?;
    let size = metadata.len();
    if digest != record.package_digest || size != record.package_size_bytes {
        return Err(CacheError::DigestMismatch {
            expected: format!(
                "{} ({} bytes)",
                record.package_digest, record.package_size_bytes
            ),
            actual: format!("{digest} ({size} bytes)"),
        });
    }
    Ok(DeterministicPackageReport {
        canonical_payload_digest: record.canonical_payload_digest.clone(),
        encoder_profile: record.encoder_profile.clone(),
        package_size_bytes: size,
        package_digest: digest,
        package_path: destination.to_owned(),
    })
}

fn conservative_zip_upper_bound(envelope: &CanonicalPayloadEnvelope) -> Result<u64, CacheError> {
    // Raw DEFLATE can expand incompressible inputs by a small block overhead.
    // Six bytes per 16 KiB plus fixed ZIP/ZIP64 headers is deliberately more
    // conservative than the encoder's normal bound.
    envelope
        .ordered_items
        .iter()
        .try_fold(1024u64, |total, item| {
            let blocks = item.size_bytes.saturating_add(16_383) / 16_384;
            let deflate_bound = item
                .size_bytes
                .checked_add(blocks.saturating_mul(6))
                .and_then(|value| value.checked_add(256))
                .and_then(|value| {
                    value.checked_add((item.normalized_path.len() as u64).saturating_mul(2))
                })
                .ok_or_else(|| {
                    CacheError::ResourceLimit("deterministic package bound exceeds u64".to_owned())
                })?;
            total.checked_add(deflate_bound).ok_or_else(|| {
                CacheError::ResourceLimit("deterministic package bound exceeds u64".to_owned())
            })
        })
}

fn digest_file(
    path: &Path,
    cancellation: &CancellationToken,
    buffer: &mut [u8],
) -> Result<ContentDigest, CacheError> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let read = input.read(buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(ContentDigest(format!("{:x}", hasher.finalize())))
}

fn create_temporary_file(destination: &Path) -> Result<(PathBuf, File), CacheError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let name = destination
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("payload.zip"))
        .to_string_lossy();
    for _ in 0..128 {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{name}.xc-partial-{}-{sequence}",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(CacheError::Io(
        "could not allocate a unique package staging file".to_owned(),
    ))
}

fn map_zip_error(error: ZipError, cancellation: &CancellationToken) -> CacheError {
    match error {
        ZipError::Io(error) => map_io_error(error, cancellation),
        other => CacheError::InvalidManifest(format!("deterministic ZIP encoding failed: {other}")),
    }
}

fn map_io_error(error: io::Error, cancellation: &CancellationToken) -> CacheError {
    if cancellation.is_cancelled() {
        let reason = cancellation
            .check()
            .err()
            .map_or_else(|| error.to_string(), |reason| reason.to_string());
        CacheError::Cancelled(reason)
    } else if error.kind() == io::ErrorKind::StorageFull {
        CacheError::ResourceLimit("deterministic package exceeds temporary-disk budget".to_owned())
    } else {
        CacheError::Io(error.to_string())
    }
}

struct ControlledFile {
    inner: File,
    maximum_bytes: Option<u64>,
    position: u64,
    length: u64,
}

impl ControlledFile {
    fn new(inner: File, maximum_bytes: Option<u64>) -> Self {
        Self {
            inner,
            maximum_bytes,
            position: 0,
            length: 0,
        }
    }

    fn check(&self, end: u64) -> io::Result<()> {
        if self.maximum_bytes.is_some_and(|maximum| end > maximum) {
            return Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "temporary-disk budget reached",
            ));
        }
        Ok(())
    }
}

impl Write for ControlledFile {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::StorageFull, "package size overflow"))?;
        self.check(end)?;
        let written = self.inner.write(bytes)?;
        self.position = self.position.saturating_add(written as u64);
        self.length = self.length.max(self.position);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.check(self.length)?;
        self.inner.flush()
    }
}

impl Seek for ControlledFile {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let position = self.inner.seek(position)?;
        self.check(position)?;
        self.position = position;
        Ok(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PayloadDependencyIdentity;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};
    use xc_core::CancellationReason;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_root(name: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "xc-cache-packaging-{name}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn fixture(root: &Path) -> (CanonicalPayloadEnvelope, Vec<PayloadFileSource>) {
        let alpha = b"alpha payload\n";
        let beta = b"beta payload with repeated repeated repeated bytes\n";
        let alpha_path = root.join("alpha.bin");
        let beta_path = root.join("beta.bin");
        fs::write(&alpha_path, alpha).unwrap();
        fs::write(&beta_path, beta).unwrap();
        let envelope = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "opaque-bytes-v1".to_owned(),
            dimensions: vec![2],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![
                LogicalPayloadItem {
                    normalized_path: "data/alpha.bin".to_owned(),
                    content_digest: ContentDigest::sha256(alpha),
                    size_bytes: alpha.len() as u64,
                },
                LogicalPayloadItem {
                    normalized_path: "data/beta.bin".to_owned(),
                    content_digest: ContentDigest::sha256(beta),
                    size_bytes: beta.len() as u64,
                },
            ],
            dependencies: Vec::<PayloadDependencyIdentity>::new(),
        };
        let sources = vec![
            PayloadFileSource {
                normalized_path: "data/beta.bin".to_owned(),
                source_path: beta_path,
            },
            PayloadFileSource {
                normalized_path: "data/alpha.bin".to_owned(),
                source_path: alpha_path,
            },
        ];
        (envelope, sources)
    }

    #[test]
    fn deterministic_zip_has_fixed_bytes_and_metadata() {
        let root = test_root("deterministic");
        let (envelope, sources) = fixture(&root);
        let first = root.join("first.zip");
        let second = root.join("second.zip");
        let cancellation = CancellationToken::new();
        let first_report = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &first,
            &ResourcePolicy::default(),
            &cancellation,
        )
        .unwrap();
        let second_report = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &second,
            &ResourcePolicy::default(),
            &cancellation,
        )
        .unwrap();
        assert_eq!(first_report.package_digest, second_report.package_digest);
        assert_eq!(
            first_report.encoder_profile,
            CURRENT_MULTI_ITEM_ZIP64_PROFILE
        );
        assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());

        let bytes = fs::read(&first).unwrap();
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        assert_eq!(archive.len(), 2);
        let alpha = archive.by_index(0).unwrap();
        assert_eq!(alpha.name(), "data/alpha.bin");
        assert_eq!(alpha.unix_mode(), Some(0o100644));
        assert_eq!(alpha.last_modified().unwrap().year(), 1980);
        drop(alpha);
        assert_eq!(archive.by_index(1).unwrap().name(), "data/beta.bin");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn v2_profile_requires_zip64_metadata_on_every_entry() {
        let root = test_root("v2-requires-zip64");
        let (envelope, sources) = fixture(&root);
        let package = root.join("legacy-multi.zip");
        let output = File::create(&package).unwrap();
        let mut archive = ZipWriter::new(output);
        for item in &envelope.ordered_items {
            let source = sources
                .iter()
                .find(|source| source.normalized_path == item.normalized_path)
                .unwrap();
            archive
                .start_file(
                    &item.normalized_path,
                    deterministic_options().large_file(false),
                )
                .unwrap();
            archive
                .write_all(&fs::read(&source.source_path).unwrap())
                .unwrap();
        }
        archive.finish().unwrap();

        let mut input = File::open(&package).unwrap();
        let record = crate::stream_split_encoded(
            &mut input,
            envelope.digest().unwrap(),
            DETERMINISTIC_ZIP64_PROFILE_V2,
            &crate::TransportPolicy::default(),
            &ResourcePolicy::default(),
            &CancellationToken::new(),
            |_, _| Ok(()),
        )
        .unwrap();
        assert!(matches!(
            verify_canonical_payload_zip64(&envelope, &record, &package, &CancellationToken::new()),
            Err(CacheError::InvalidManifest(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fused_packaging_retains_only_one_in_flight_part_and_cleans_archive() {
        let root = test_root("fused-stream");
        let (envelope, sources) = fixture(&root);
        let archive = root.join("transient.zip");
        let policy = TransportPolicy {
            maximum_file_bytes_exclusive: 64,
            split_part_bytes: 32,
            maximum_batch_payload_bytes: 64,
            maximum_pending_batches: 2,
        };
        let mut checkpoints = Vec::new();
        let report = package_and_stream_split_zip64(
            StreamingPackageSplitRequest {
                envelope: &envelope,
                sources: &sources,
                temporary_archive_path: &archive,
                transport_policy: &policy,
                resources: &ResourcePolicy::default(),
                cancellation: &CancellationToken::new(),
            },
            |part, bytes| {
                assert_eq!(ContentDigest::sha256(bytes), part.content_digest);
                checkpoints.push((part.sequence, part.content_digest.clone(), bytes.len()));
                Ok(())
            },
        )
        .unwrap();
        assert!(!archive.exists());
        assert!(report.cleanup_complete);
        assert_eq!(report.maximum_in_flight_parts, 1);
        assert!(!report.complete_archive_retained);
        assert!(!report.split_parts_retained_by_packager);
        assert_eq!(checkpoints.len(), report.encoding.ordered_parts.len());
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.2 as u64)
                .sum::<u64>(),
            report.encoding.package_size_bytes
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fused_packaging_cancellation_removes_archive_after_checkpointed_part() {
        let root = test_root("fused-cancel");
        let (envelope, sources) = fixture(&root);
        let archive = root.join("transient.zip");
        let cancellation = CancellationToken::new();
        let sink_cancellation = cancellation.clone();
        let mut checkpointed = Vec::new();
        let result = package_and_stream_split_zip64(
            StreamingPackageSplitRequest {
                envelope: &envelope,
                sources: &sources,
                temporary_archive_path: &archive,
                transport_policy: &TransportPolicy {
                    maximum_file_bytes_exclusive: 32,
                    split_part_bytes: 16,
                    maximum_batch_payload_bytes: 32,
                    maximum_pending_batches: 1,
                },
                resources: &ResourcePolicy::default(),
                cancellation: &cancellation,
            },
            |part, _| {
                checkpointed.push(part.clone());
                sink_cancellation.cancel(CancellationReason::UserRequested);
                Ok(())
            },
        );
        assert!(matches!(result, Err(CacheError::Cancelled(_))));
        assert_eq!(checkpointed.len(), 1);
        assert!(!archive.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ordered_parts_reconstruct_and_verify_canonical_payload() {
        let root = test_root("reconstruct");
        let (envelope, sources) = fixture(&root);
        let package = root.join("source.zip");
        let cancellation = CancellationToken::new();
        let report = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &package,
            &ResourcePolicy::default(),
            &cancellation,
        )
        .unwrap();
        let parts_root = root.join("parts");
        let policy = crate::TransportPolicy {
            maximum_file_bytes_exclusive: 64,
            split_part_bytes: 32,
            maximum_batch_payload_bytes: 64,
            maximum_pending_batches: 1,
        };
        let mut package_input = File::open(&package).unwrap();
        let record = crate::stream_split_encoded(
            &mut package_input,
            report.canonical_payload_digest.clone(),
            report.encoder_profile.clone(),
            &policy,
            &ResourcePolicy::default(),
            &cancellation,
            |part, bytes| {
                let path = resolve_part_path(&parts_root, &part.repository_path)?;
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(path, bytes)?;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(record.package_digest, report.package_digest);

        let reconstructed = root.join("reconstructed.zip");
        let reconstructed_report = reconstruct_transport_package(
            &record,
            &parts_root,
            &reconstructed,
            &ResourcePolicy::default(),
            &cancellation,
        )
        .unwrap();
        assert_eq!(reconstructed_report.package_digest, report.package_digest);
        let verified =
            verify_canonical_payload_zip64(&envelope, &record, &reconstructed, &cancellation)
                .unwrap();
        assert_eq!(verified.item_count, 2);
        assert_eq!(verified.logical_size_bytes, envelope.logical_size_bytes());

        // Two processes can finish the same content-addressed reconstruction
        // together. The hard-link loser verifies and reuses the winner rather
        // than treating the already-visible immutable package as an error.
        fs::remove_file(&reconstructed).unwrap();
        let barrier = std::sync::Barrier::new(2);
        let (first, second) = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                barrier.wait();
                reconstruct_transport_package(
                    &record,
                    &parts_root,
                    &reconstructed,
                    &ResourcePolicy::default(),
                    &CancellationToken::new(),
                )
            });
            let second = scope.spawn(|| {
                barrier.wait();
                reconstruct_transport_package(
                    &record,
                    &parts_root,
                    &reconstructed,
                    &ResourcePolicy::default(),
                    &CancellationToken::new(),
                )
            });
            (
                first.join().unwrap().unwrap(),
                second.join().unwrap().unwrap(),
            )
        });
        assert_eq!(first.package_digest, report.package_digest);
        assert_eq!(second.package_digest, report.package_digest);
        verify_canonical_payload_zip64(&envelope, &record, &reconstructed, &cancellation).unwrap();

        let corrupt_part =
            resolve_part_path(&parts_root, &record.ordered_parts[0].repository_path).unwrap();
        fs::write(
            corrupt_part,
            vec![0u8; record.ordered_parts[0].size_bytes as usize],
        )
        .unwrap();
        let rejected = reconstruct_transport_package(
            &record,
            &parts_root,
            &root.join("rejected.zip"),
            &ResourcePolicy::default(),
            &cancellation,
        );
        assert!(matches!(rejected, Err(CacheError::DigestMismatch { .. })));
        assert!(!root.join("rejected.zip").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_digest_mismatch_never_exposes_package() {
        let root = test_root("mismatch");
        let (envelope, sources) = fixture(&root);
        fs::write(&sources[0].source_path, b"tampered").unwrap();
        let destination = root.join("payload.zip");
        let result = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &destination,
            &ResourcePolicy::default(),
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(CacheError::DigestMismatch { .. })));
        assert!(!destination.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disk_budget_and_cancellation_leave_no_visible_package() {
        let root = test_root("limits");
        let (envelope, sources) = fixture(&root);
        let limited_destination = root.join("limited.zip");
        let resources = ResourcePolicy {
            maximum_temporary_disk_bytes: Some(8),
            ..ResourcePolicy::default()
        };
        let result = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &limited_destination,
            &resources,
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(CacheError::ResourceLimit(_))));
        assert!(!limited_destination.exists());

        let cancelled_destination = root.join("cancelled.zip");
        let cancellation = CancellationToken::new();
        cancellation.cancel(CancellationReason::UserRequested);
        let result = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &cancelled_destination,
            &ResourcePolicy::default(),
            &cancellation,
        );
        assert!(matches!(result, Err(CacheError::Cancelled(_))));
        assert!(!cancelled_destination.exists());

        let deadline_destination = root.join("deadline.zip");
        let deadline_resources = ResourcePolicy {
            maximum_wall_seconds: Some(0),
            ..ResourcePolicy::default()
        };
        let deadline = CancellationToken::for_policy(&deadline_resources);
        let result = package_canonical_payload_zip64(
            &envelope,
            &sources,
            &deadline_destination,
            &deadline_resources,
            &deadline,
        );
        assert!(matches!(result, Err(CacheError::Cancelled(_))));
        assert!(format!("{}", deadline.check().unwrap_err()).contains("WallTime"));
        assert!(!deadline_destination.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
