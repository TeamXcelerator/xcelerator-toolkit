//! Authenticate both compact payloads before coalescing producer-only differences.
use super::*;
use std::io::{Cursor, Read};

/// A linked numerical backend can verify a compact proof against its request
/// and return its mathematical digest. Cache-only builds fail closed.
pub struct CompactPublicationVerifier(
    pub fn(&crate::SemanticKeyEnvelope, &[u8]) -> Result<ContentDigest, CacheError>,
);
inventory::collect!(CompactPublicationVerifier);

fn invalid(message: &str) -> CacheError {
    CacheError::InvalidManifest(message.into())
}

fn certificate_bytes(
    manifest: &crate::CanonicalArtifactManifest,
    encoding: &crate::TransportEncodingRecord,
    memory_limit: Option<u64>,
    mut read_part: impl FnMut(&TransportPart, &mut Vec<u8>) -> Result<(), CacheError>,
) -> Result<Vec<u8>, CacheError> {
    encoding.validate()?;
    let [item] = manifest.canonical_payload.ordered_items.as_slice() else {
        return Err(invalid("expected one compact certificate payload"));
    };
    if item.normalized_path != "payload.json"
        || item
            .size_bytes
            .checked_add(encoding.package_size_bytes)
            .is_none_or(|bytes| memory_limit.is_some_and(|limit| bytes > limit))
        || encoding.canonical_payload_digest != manifest.payload_digest
        || !manifest.transport_digests.contains(&encoding.digest()?)
    {
        return Err(invalid(
            "invalid compact certificate transport binding or size",
        ));
    }
    let reserve = |bytes: u64| -> Result<Vec<u8>, CacheError> {
        let mut buf = Vec::new();
        buf.try_reserve_exact(
            usize::try_from(bytes).map_err(|e| CacheError::ResourceLimit(e.to_string()))?,
        )
        .map_err(|e| CacheError::ResourceLimit(e.to_string()))?;
        Ok(buf)
    };
    let mut archive = reserve(encoding.package_size_bytes)?;
    for part in &encoding.ordered_parts {
        let offset = archive.len();
        if part.size_bytes > encoding.package_size_bytes.saturating_sub(offset as u64) {
            return Err(invalid("compact certificate part exceeds package length"));
        }
        read_part(part, &mut archive)?;
        let bytes = &archive[offset..];
        if bytes.len() as u64 != part.size_bytes
            || ContentDigest::sha256(bytes) != part.content_digest
        {
            return Err(invalid("compact certificate part digest mismatch"));
        }
    }
    if archive.len() as u64 != encoding.package_size_bytes
        || ContentDigest::sha256(&archive) != encoding.package_digest
    {
        return Err(invalid("compact certificate package digest mismatch"));
    }
    let mut zip =
        zip::ZipArchive::new(Cursor::new(archive)).map_err(|e| invalid(&e.to_string()))?;
    if zip.len() != 1 {
        return Err(invalid("unexpected compact certificate package members"));
    }
    let member = zip.by_index(0).map_err(|e| invalid(&e.to_string()))?;
    if member.name() != item.normalized_path || member.size() != item.size_bytes {
        return Err(invalid("unexpected compact certificate member"));
    }
    let mut bytes = reserve(item.size_bytes)?;
    member.take(item.size_bytes + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != item.size_bytes || ContentDigest::sha256(&bytes) != item.content_digest
    {
        return Err(invalid("compact certificate payload digest mismatch"));
    }
    Ok(bytes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn equivalent(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    existing: &crate::CanonicalArtifactManifest,
    transport: &ContentDigest,
    draft: &CanonicalProductionDraft,
    cancellation: &xc_core::CancellationToken,
    memo: Option<&producer_admission::ProducerProofCache>,
) -> Result<bool, CacheError> {
    if existing.schema_version != 2
        || draft.manifest.schema_version != 2
        || existing.semantic_key.artifact_kind != "ccm_sector_gap_certificate"
        || existing.semantic_key.resolved_mathematical_parameters["representation"]
            != "compact_analytic_replay_v1"
    {
        return Ok(false);
    }
    // Only the content of the single member can differ. Preserve all numerical
    // envelope, configuration, policy, reader and assurance admission checks.
    if !same_publication_contract(existing, &draft.manifest) {
        return Ok(false);
    }
    let mut retained = existing.canonical_payload.clone();
    let mut staged = draft.manifest.canonical_payload.clone();
    retained.dependencies.clear();
    staged.dependencies.clear();
    retained.ordered_items.clear();
    staged.ordered_items.clear();
    if retained != staged {
        return Ok(false);
    }
    let mut verifiers = inventory::iter::<CompactPublicationVerifier>.into_iter();
    let verifier = verifiers.next().ok_or_else(|| {
        invalid("compact publication equivalence requires the spectral Arb replay verifier")
    })?;
    if verifiers.next().is_some() {
        return Err(invalid("ambiguous compact proof verifier"));
    }
    let digest = |manifest: &crate::CanonicalArtifactManifest, bytes: &[u8]| {
        let key = (
            manifest.semantic_digest.clone(),
            ContentDigest::sha256(bytes),
        );
        if let Some(cache) = memo {
            if let Some(proof) = cache.compact_proof(&key) {
                return Ok(proof);
            }
        }
        let proof = (verifier.0)(&manifest.semantic_key, bytes)?;
        if let Some(cache) = memo {
            cache.remember_compact_proof(key, proof.clone());
        }
        Ok::<_, CacheError>(proof)
    };
    let reader = crate::RemoteShardReader::new(remote, 16 * 1024 * 1024)?;
    let memory_limit = memo
        .map(|cache| cache.memory_limit)
        .unwrap_or_else(|| xc_core::ResourcePolicy::default().maximum_memory_bytes);
    let path = format!(
        "encodings/{}/{}.json",
        &existing.payload_digest.0[..2],
        transport
    );
    let rec = reader
        .load_transport_encoding(repository, revision, &path, transport, cancellation)?
        .value;
    let prior = certificate_bytes(existing, &rec, memory_limit, |part, bytes| {
        remote.read_committed_path(
            repository,
            revision,
            &part.repository_path,
            part.size_bytes,
            cancellation,
            bytes,
        )?;
        Ok(())
    })?;
    let prior_proof = digest(existing, &prior)?;
    drop(prior);
    let fresh = certificate_bytes(
        &draft.manifest,
        &draft.encoding,
        memory_limit,
        |part, bytes| {
            File::open(draft.staged_parts_root.join(&part.repository_path))?
                .take(part.size_bytes + 1)
                .read_to_end(bytes)?;
            Ok(())
        },
    )?;
    Ok(prior_proof == digest(&draft.manifest, &fresh)?)
}
