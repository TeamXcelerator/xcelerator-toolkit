//! Bounded session-local manifests, payloads, and assurance evidence.
//! No disk paths or publication transports are created by this store.
use crate::{
    ArtifactAssuranceAttestation, ArtifactAssuranceRequirement, ArtifactDraft, ArtifactKey,
    ArtifactManifest, ArtifactProductionAssessment, ArtifactProductionSink, CacheError,
    CacheObjectRef, CacheStore, CacheVisibility, ContentDigest, ProducedArtifactRecord,
};
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard};

const MAXIMUM_RECORDS: usize = 8192;

#[derive(Default)]
struct State {
    artifacts: Vec<(ArtifactManifest, Vec<u8>)>,
    produced: Vec<ProducedArtifactRecord>,
    attestations: Vec<ArtifactAssuranceAttestation>,
    requirements: Vec<ArtifactAssuranceRequirement>,
    evidence: Vec<(ContentDigest, Vec<u8>)>,
    accounted_bytes: u64,
}

/// A bounded in-memory cache and evidence sink for one managed computation.
/// Clones share the same budget and records. Dropping the session releases all
/// records; no staging, inventories, or encoded filesystem objects are retained.
#[derive(Clone)]
pub struct EphemeralCacheStore {
    state: Arc<Mutex<State>>,
    maximum_bytes: u64,
}

impl EphemeralCacheStore {
    pub fn new(maximum_bytes: u64) -> Result<Self, CacheError> {
        if maximum_bytes == 0 {
            return Err(CacheError::ResourceLimit(
                "ephemeral memory budget is zero".to_owned(),
            ));
        }
        Ok(Self {
            state: Arc::new(Mutex::new(State::default())),
            maximum_bytes,
        })
    }
    fn state(&self) -> Result<MutexGuard<'_, State>, CacheError> {
        self.state
            .lock()
            .map_err(|_| CacheError::InvalidTransition("ephemeral cache lock poisoned".to_owned()))
    }
    fn reserve(&self, state: &mut State, bytes: u64, count: usize) -> Result<(), CacheError> {
        let next = state.accounted_bytes.checked_add(bytes).ok_or_else(|| {
            CacheError::ResourceLimit("ephemeral memory accounting overflow".to_owned())
        })?;
        if next > self.maximum_bytes || count >= MAXIMUM_RECORDS {
            return Err(CacheError::ResourceLimit(
                "ephemeral cache budget exhausted".to_owned(),
            ));
        }
        state.accounted_bytes = next;
        Ok(())
    }
    fn metadata_bytes<T: serde::Serialize>(value: &T) -> Result<u64, CacheError> {
        // Reserve two serialized sizes plus a per-record allocation allowance
        // for owned strings, collection capacities, and metadata structure.
        let bytes = serde_json::to_vec(value)?.len() as u64;
        bytes
            .checked_mul(2)
            .and_then(|v| v.checked_add(1024))
            .ok_or_else(|| {
                CacheError::ResourceLimit("ephemeral metadata accounting overflow".to_owned())
            })
    }
    fn check_evidence(state: &State, digests: &[ContentDigest]) -> Result<(), CacheError> {
        if digests
            .iter()
            .any(|digest| !state.evidence.iter().any(|(d, _)| d == digest))
        {
            return Err(CacheError::InvalidManifest(
                "ephemeral assurance references missing evidence".to_owned(),
            ));
        }
        Ok(())
    }
    /// Bytes charged to the shared resident-record budget.
    pub fn accounted_bytes(&self) -> Result<u64, CacheError> {
        Ok(self.state()?.accounted_bytes)
    }
}

impl CacheStore for EphemeralCacheStore {
    fn name(&self) -> &str {
        "ephemeral"
    }
    fn writable(&self) -> bool {
        true
    }
    fn visibility(&self) -> CacheVisibility {
        CacheVisibility::Local
    }
    fn put(&self, draft: &ArtifactDraft, payload: &[u8]) -> Result<ArtifactManifest, CacheError> {
        if draft.visibility != CacheVisibility::Local {
            return Err(CacheError::InvalidManifest(
                "ephemeral drafts must be local".to_owned(),
            ));
        }
        let digest = ContentDigest::sha256(payload);
        let size = payload.len() as u64;
        let manifest = ArtifactManifest {
            schema_version: draft.schema_version,
            key: draft.key.clone(),
            content_digest: digest.clone(),
            size_bytes: size,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: size,
            }],
            created_unix_seconds: 0,
            producer_toolkit_version: draft.producer_toolkit_version.clone(),
            minimum_reader_version: draft.minimum_reader_version.clone(),
            maximum_reader_version: draft.maximum_reader_version.clone(),
            quality: draft.quality,
            visibility: draft.visibility,
            immutable: draft.immutable,
            dependencies: draft.dependencies.clone(),
            tags: draft.tags.clone(),
            provenance_digest: draft.provenance_digest.clone(),
        };
        manifest.validate()?;
        let bytes = Self::metadata_bytes(&manifest)?
            .checked_add(size)
            .ok_or_else(|| {
                CacheError::ResourceLimit("ephemeral payload accounting overflow".to_owned())
            })?;
        let mut state = self.state()?;
        if state.artifacts.iter().any(|(m, _)| m == &manifest) {
            return Ok(manifest);
        }
        let count = state.artifacts.len();
        self.reserve(&mut state, bytes, count)?;
        state.artifacts.push((manifest.clone(), payload.to_vec()));
        Ok(manifest)
    }
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        key.validate()?;
        Ok(self
            .state()?
            .artifacts
            .iter()
            .filter(|(m, _)| &m.key == key)
            .map(|(m, _)| m.clone())
            .collect())
    }
    fn matching_keys(
        &self,
        kind: &str,
        prefix: &str,
        maximum: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        let state = self.state()?;
        let mut keys = Vec::new();
        for (m, _) in &state.artifacts {
            if m.key.kind == kind && m.key.logical_key.starts_with(prefix) && !keys.contains(&m.key)
            {
                if keys.len() >= maximum {
                    break;
                }
                keys.push(m.key.clone());
            }
        }
        Ok(keys)
    }
    fn read_payload(&self, manifest: &ArtifactManifest) -> Result<Vec<u8>, CacheError> {
        manifest.validate()?;
        // Never reserve from caller-supplied size metadata before confirming
        // exact membership. Only already bounded retained payloads are copied.
        let state = self.state()?;
        let (_, payload) = state
            .artifacts
            .iter()
            .find(|(m, _)| m == manifest)
            .ok_or_else(|| CacheError::NotFound("ephemeral manifest".to_owned()))?;
        if payload.len() as u64 != manifest.size_bytes
            || ContentDigest::sha256(payload) != manifest.content_digest
        {
            return Err(CacheError::InvalidManifest(
                "ephemeral payload binding failed".to_owned(),
            ));
        }
        Ok(payload.clone())
    }
    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        manifest.validate()?;
        let state = self.state()?;
        let (_, payload) = state
            .artifacts
            .iter()
            .find(|(m, _)| m == manifest)
            .ok_or_else(|| CacheError::NotFound("ephemeral manifest".to_owned()))?;
        if payload.len() as u64 != manifest.size_bytes
            || ContentDigest::sha256(payload) != manifest.content_digest
        {
            return Err(CacheError::InvalidManifest(
                "ephemeral payload binding failed".to_owned(),
            ));
        }
        writer.write_all(payload)?;
        Ok(())
    }
}

impl ArtifactProductionSink for EphemeralCacheStore {
    fn record(&self, mut artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
        artifact.manifest.validate()?;
        if ContentDigest::sha256(&artifact.payload) != artifact.manifest.content_digest
            || artifact.payload.len() as u64 != artifact.manifest.size_bytes
        {
            return Err(CacheError::InvalidManifest(
                "ephemeral production payload binding failed".to_owned(),
            ));
        }
        let assessment = ArtifactProductionAssessment {
            achieved_assurance: artifact.achieved_assurance,
            evidence_digests: artifact.assurance_evidence_digests.clone(),
        };
        assessment.validate()?;
        // CacheStore owns fresh payloads. The sink retains evidence and record
        // metadata only; warm payloads remain in the read-only overlay.
        artifact.payload.clear();
        artifact.payload.shrink_to_fit();
        let bytes = Self::metadata_bytes(&artifact)?;
        let mut state = self.state()?;
        Self::check_evidence(&state, &assessment.evidence_digests)?;
        if state.produced.iter().any(|p| p == &artifact) {
            return Ok(());
        }
        let count = state.produced.len();
        self.reserve(&mut state, bytes, count)?;
        state.produced.push(artifact);
        Ok(())
    }
    fn contains_artifact(
        &self,
        key: &ArtifactKey,
        digest: &ContentDigest,
    ) -> Result<bool, CacheError> {
        Ok(self
            .state()?
            .produced
            .iter()
            .any(|p| &p.manifest.key == key && &p.manifest.content_digest == digest))
    }
    fn retained_assurance(
        &self,
        key: &ArtifactKey,
        digest: &ContentDigest,
    ) -> Result<Option<ArtifactProductionAssessment>, CacheError> {
        let state = self.state()?;
        if let Some(a) = state
            .attestations
            .iter()
            .rev()
            .find(|a| &a.artifact_key == key && &a.content_digest == digest)
        {
            return Ok(Some(ArtifactProductionAssessment {
                achieved_assurance: a.achieved_assurance,
                evidence_digests: a.evidence_digests.clone(),
            }));
        }
        Ok(state
            .produced
            .iter()
            .rev()
            .find(|p| &p.manifest.key == key && &p.manifest.content_digest == digest)
            .map(|p| ArtifactProductionAssessment {
                achieved_assurance: p.achieved_assurance,
                evidence_digests: p.assurance_evidence_digests.clone(),
            }))
    }
    fn record_assurance_requirement(
        &self,
        requirement: ArtifactAssuranceRequirement,
    ) -> Result<(), CacheError> {
        requirement.validate()?;
        let bytes = Self::metadata_bytes(&requirement)?;
        let mut state = self.state()?;
        if state.requirements.contains(&requirement) {
            return Ok(());
        }
        let count = state.requirements.len();
        self.reserve(&mut state, bytes, count)?;
        state.requirements.push(requirement);
        Ok(())
    }
    fn record_assurance(
        &self,
        attestation: ArtifactAssuranceAttestation,
    ) -> Result<(), CacheError> {
        attestation.validate()?;
        let bytes = Self::metadata_bytes(&attestation)?;
        let mut state = self.state()?;
        Self::check_evidence(&state, &attestation.evidence_digests)?;
        if state.attestations.contains(&attestation) {
            return Ok(());
        }
        let count = state.attestations.len();
        self.reserve(&mut state, bytes, count)?;
        state.attestations.push(attestation);
        Ok(())
    }
    fn record_evidence(&self, kind: &str, bytes: &[u8]) -> Result<ContentDigest, CacheError> {
        if kind.is_empty()
            || kind.len() > 128
            || !kind
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
        {
            return Err(CacheError::InvalidManifest(
                "ephemeral evidence kind is invalid".to_owned(),
            ));
        }
        let digest = ContentDigest::sha256(bytes);
        let mut state = self.state()?;
        if state.evidence.iter().any(|(d, _)| d == &digest) {
            return Ok(digest);
        }
        let count = state.evidence.len();
        self.reserve(&mut state, (bytes.len() as u64).saturating_add(1024), count)?;
        state.evidence.push((digest.clone(), bytes.to_vec()));
        Ok(digest)
    }
}

/// A nonwriting persistent overlay that rejects oversized logical/encoded
/// reads before the inner store reserves payload or archive buffers.
pub(crate) struct ReadOnlyBudgetedStore {
    inner: Box<dyn CacheStore>,
    maximum_read_bytes: u64,
}
impl ReadOnlyBudgetedStore {
    pub(crate) fn new(inner: Box<dyn CacheStore>, maximum_read_bytes: u64) -> Self {
        Self {
            inner,
            maximum_read_bytes,
        }
    }
    fn check_read(&self, manifest: &ArtifactManifest) -> Result<(), CacheError> {
        manifest.validate()?;
        let encoded = manifest.objects.iter().try_fold(0u64, |sum, object| {
            sum.checked_add(object.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit("read-only encoded size overflow".to_owned())
            })
        })?;
        let bytes = manifest.size_bytes.checked_add(encoded).ok_or_else(|| {
            CacheError::ResourceLimit("read-only payload size overflow".to_owned())
        })?;
        if bytes > self.maximum_read_bytes {
            return Err(CacheError::ResourceLimit(
                "read-only overlay payload and encoded object exceed the session read budget"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}
impl CacheStore for ReadOnlyBudgetedStore {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        self.inner.visibility()
    }
    fn put(&self, _: &ArtifactDraft, _: &[u8]) -> Result<ArtifactManifest, CacheError> {
        Err(CacheError::ReadOnlyLayer(self.name().to_owned()))
    }
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        self.inner.candidates(key)
    }
    fn matching_keys(
        &self,
        kind: &str,
        prefix: &str,
        maximum: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        self.inner.matching_keys(kind, prefix, maximum)
    }
    fn identity_candidates(
        &self,
        identity: &crate::PayloadDependencyIdentity,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        self.inner.identity_candidates(identity)
    }
    fn ccm_eigenpair_continuation_keys(
        &self,
        query: &crate::CcmEigenpairContinuationQuery,
        maximum: usize,
    ) -> Result<Vec<ArtifactKey>, CacheError> {
        self.inner.ccm_eigenpair_continuation_keys(query, maximum)
    }
    fn read_payload(&self, manifest: &ArtifactManifest) -> Result<Vec<u8>, CacheError> {
        self.check_read(manifest)?;
        self.inner.read_payload(manifest)
    }
    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        self.check_read(manifest)?;
        self.inner.read_payload_to(manifest, writer)
    }
    fn verified_encoded_payload(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<Option<crate::VerifiedEncodedPayload>, CacheError> {
        self.check_read(manifest)?;
        self.inner.verified_encoded_payload(manifest)
    }
    fn verified_transport_parts(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<Option<crate::VerifiedTransportParts>, CacheError> {
        self.check_read(manifest)?;
        self.inner.verified_transport_parts(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn draft() -> ArtifactDraft {
        ArtifactDraft {
            schema_version: 1,
            key: ArtifactKey::new("fixture", "exact", b"independent-input").unwrap(),
            producer_toolkit_version: crate::current_toolkit_version().unwrap(),
            minimum_reader_version: crate::current_toolkit_version().unwrap(),
            maximum_reader_version: None,
            quality: crate::CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: Vec::new(),
            tags: Default::default(),
            provenance_digest: None,
        }
    }
    #[test]
    fn ephemeral_budget_is_transactional_and_payload_is_content_bound() {
        let store = EphemeralCacheStore::new(16_384).unwrap();
        let d = draft();
        let manifest = store.put(&d, b"exact bytes").unwrap();
        assert_eq!(
            manifest.content_digest,
            ContentDigest::sha256(b"exact bytes")
        );
        assert_eq!(store.read_payload(&manifest).unwrap(), b"exact bytes");
        let before = store.accounted_bytes().unwrap();
        let mut too_large = d.clone();
        too_large.key.logical_key = "over-budget".into();
        assert!(matches!(
            store.put(&too_large, &[1; 16_384]),
            Err(CacheError::ResourceLimit(_))
        ));
        assert_eq!(store.accounted_bytes().unwrap(), before);
        assert!(store.candidates(&too_large.key).unwrap().is_empty());
        assert_eq!(store.read_payload(&manifest).unwrap(), b"exact bytes");
        let mut false_manifest = manifest.clone();
        false_manifest.content_digest = ContentDigest::sha256(b"forged");
        assert!(store.read_payload(&false_manifest).is_err());
        // A syntactically valid huge manifest must fail before reserving its
        // advertised length or invoking any writer operation.
        let mut huge = manifest.clone();
        huge.size_bytes = u64::MAX;
        huge.objects[0].size_bytes = u64::MAX;
        huge.validate().unwrap();
        assert!(matches!(
            store.read_payload(&huge),
            Err(CacheError::NotFound(_))
        ));
        struct NoWrite;
        impl Write for NoWrite {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                panic!("forged read invoked writer")
            }
            fn flush(&mut self) -> std::io::Result<()> {
                panic!("forged read flushed writer")
            }
        }
        assert!(matches!(
            store.read_payload_to(&huge, &mut NoWrite),
            Err(CacheError::NotFound(_))
        ));
        assert_eq!(store.accounted_bytes().unwrap(), before);
    }
    #[test]
    fn read_only_overlay_enforces_read_limit_and_preserves_errors() {
        let memory = EphemeralCacheStore::new(16_384).unwrap();
        let manifest = memory.put(&draft(), b"exact bytes").unwrap();
        let readonly = ReadOnlyBudgetedStore::new(Box::new(memory.clone()), 21);
        // Exact stored payload size is 11 bytes; logical plus encoded is 22.
        assert!(matches!(
            readonly.read_payload(&manifest),
            Err(CacheError::ResourceLimit(_))
        ));
        let mut output = Vec::new();
        assert!(matches!(
            readonly.read_payload_to(&manifest, &mut output),
            Err(CacheError::ResourceLimit(_))
        ));
        assert!(output.is_empty());
        let admitted = ReadOnlyBudgetedStore::new(Box::new(memory), 22);
        assert_eq!(admitted.read_payload(&manifest).unwrap(), b"exact bytes");
        let mut absent = manifest.clone();
        absent.key.logical_key = "absent".into();
        assert!(matches!(
            admitted.read_payload(&absent),
            Err(CacheError::NotFound(_))
        ));
        let mut invalid = manifest;
        invalid.content_digest.0 = "invalid".into();
        assert!(matches!(
            admitted.read_payload(&invalid),
            Err(CacheError::InvalidManifest(_))
        ));
        assert!(!admitted.writable());
        assert!(matches!(
            admitted.put(&draft(), b"forbidden"),
            Err(CacheError::ReadOnlyLayer(_))
        ));
    }
    #[test]
    fn ephemeral_assurance_requires_actual_retained_evidence() {
        let store = EphemeralCacheStore::new(65_536).unwrap();
        let manifest = store.put(&draft(), b"exact bytes").unwrap();
        let digest = ContentDigest::sha256(b"independent certificate");
        let attestation = ArtifactAssuranceAttestation {
            schema_version: 1,
            artifact_key: manifest.key.clone(),
            content_digest: manifest.content_digest.clone(),
            achieved_assurance: crate::ArtifactAssuranceState::Certified,
            evidence_digests: vec![digest.clone()],
        };
        let before = store.accounted_bytes().unwrap();
        assert!(store.record_assurance(attestation.clone()).is_err());
        assert_eq!(store.accounted_bytes().unwrap(), before);
        assert_eq!(
            store
                .record_evidence("certificate", b"independent certificate")
                .unwrap(),
            digest
        );
        store.record_assurance(attestation).unwrap();
        assert_eq!(
            store
                .retained_assurance(&manifest.key, &manifest.content_digest)
                .unwrap()
                .unwrap()
                .achieved_assurance,
            crate::ArtifactAssuranceState::Certified
        );
    }
}
