//! Explicit schema-2 wire types. Schema-1 bytes and identity rules stay frozen.
//!
//! The in-memory manifest adapter retains the original numerical API. On the
//! wire, exact dependency ancestry belongs to the manifest, never to the v2
//! logical payload. ZIP encoding profiles are independent of these schemas.
use crate::{
    CacheError, CanonicalArtifactManifest, CanonicalPayloadEnvelope, ContentDigest,
    LogicalPayloadItem, PayloadDependencyIdentity, SemanticKeyEnvelope, ToolkitVersion,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use xc_core::AssuranceLevel;

/// Operational manifests carry this authenticated metadata until converted to
/// the canonical schema-2 wire model. Absence means legacy format, never an
/// instruction to reinterpret the hash of an existing artifact.
pub const CONTENT_MODEL_V2_MINIMUM_READER: &str = "0.17.1";

pub const MANIFEST_PROVENANCE_V2_TAG: &str = "xc.manifest_provenance.v2";

pub fn local_manifest_provenance_v2(
    manifest: &crate::ArtifactManifest,
) -> Result<Option<ManifestProvenanceV2>, CacheError> {
    manifest
        .tags
        .get(MANIFEST_PROVENANCE_V2_TAG)
        .map(|text| {
            let provenance: ManifestProvenanceV2 = serde_json::from_str(text)?;
            provenance.validate()?;
            if crate::artifact_kind_is_private_only(&manifest.key.kind)
                && provenance.disclosure == ArtifactDisclosure::PublicInputs
            {
                return Err(CacheError::InvalidManifest(
                    "private-only producer cannot assert public inputs".into(),
                ));
            }
            Ok(provenance)
        })
        .transpose()
}

/// Read producer metadata from the authenticated local or adopted manifest.
pub fn manifest_producer_annotations(
    manifest: &crate::ArtifactManifest,
) -> Result<std::collections::BTreeMap<String, String>, CacheError> {
    manifest.validate()?;
    if let Some(provenance) = local_manifest_provenance_v2(manifest)? {
        return Ok(provenance.producer_annotations);
    }
    Ok(crate::retained_canonical_manifest(manifest)?
        .and_then(|manifest| manifest.provenance)
        .map(|provenance| provenance.producer_annotations)
        .unwrap_or_default())
}

/// Classify inputs from authenticated producer metadata, never their storage
/// visibility. Legacy retained-source policy remains conservative.
pub fn local_artifact_disclosure(
    manifest: &crate::ArtifactManifest,
) -> Result<ArtifactDisclosure, CacheError> {
    manifest.validate()?;
    if crate::artifact_kind_is_private_only(&manifest.key.kind) {
        return Ok(ArtifactDisclosure::PrivateOnly);
    }
    let local = local_manifest_provenance_v2(manifest)?.map(|p| p.disclosure);
    if let Some(canonical) = crate::retained_canonical_manifest(manifest)? {
        // An adapter may restrict reuse further. It cannot grant permission
        // that the retained canonical producer did not grant.
        if !crate::manifest_admitted_to_destination(
            &canonical,
            crate::PublicationDestination::Public,
        ) {
            return Ok(ArtifactDisclosure::PrivateOnly);
        }
        return Ok(local.unwrap_or(ArtifactDisclosure::PublicInputs));
    }
    if let Some(disclosure) = local {
        return Ok(disclosure);
    }
    if let Some(encoded) = manifest.tags.get(crate::SEMANTIC_KEY_MANIFEST_TAG) {
        let semantic: SemanticKeyEnvelope = serde_json::from_str(encoded)?;
        if semantic.artifact_kind != manifest.key.kind
            || semantic.digest()? != manifest.key.parameters_digest
        {
            return Err(CacheError::InvalidManifest(
                "source disclosure semantics do not bind its key".into(),
            ));
        }
        return Ok(
            if crate::artifact_semantics_admitted_to_destination(
                &semantic,
                crate::PublicationDestination::Public,
            ) {
                ArtifactDisclosure::PublicInputs
            } else {
                ArtifactDisclosure::Unclassified
            },
        );
    }
    Ok(ArtifactDisclosure::Unclassified)
}

pub fn provenance_for_numerical_inputs(
    kind: &str,
    sources: &[crate::ArtifactManifest],
    mut producer_receipts: Vec<ContentDigest>,
) -> Result<ManifestProvenanceV2, CacheError> {
    let mut disclosure = if crate::artifact_kind_is_private_only(kind) {
        ArtifactDisclosure::PrivateOnly
    } else {
        ArtifactDisclosure::PublicInputs
    };
    for source in sources {
        match local_artifact_disclosure(source)? {
            ArtifactDisclosure::PrivateOnly => disclosure = ArtifactDisclosure::PrivateOnly,
            ArtifactDisclosure::Unclassified if disclosure != ArtifactDisclosure::PrivateOnly => {
                disclosure = ArtifactDisclosure::Unclassified
            }
            _ => {}
        }
    }
    producer_receipts.sort();
    producer_receipts.dedup();
    let provenance = ManifestProvenanceV2 {
        producer_annotations: std::collections::BTreeMap::new(),
        disclosure,
        producer_receipts,
    };
    provenance.validate()?;
    Ok(provenance)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDisclosure {
    /// Independently public mathematical inputs and provenance.
    PublicInputs,
    /// Includes target-related or otherwise restricted inputs/evidence.
    PrivateOnly,
    /// No independently validated disclosure classification is available.
    Unclassified,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestProvenanceV2 {
    pub disclosure: ArtifactDisclosure,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub producer_annotations: std::collections::BTreeMap<String, String>,
    /// Exact evidence/producer records; these are not numerical input hashes.
    pub producer_receipts: Vec<ContentDigest>,
}

impl ManifestProvenanceV2 {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self
            .producer_receipts
            .iter()
            .any(|digest| !digest.validate())
            || self
                .producer_receipts
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(CacheError::InvalidManifest(
                "producer receipts must be valid, unique and ordered".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalPayloadEnvelopeV2 {
    pub schema_version: u32,
    pub scalar_backend: String,
    pub precision_bits: Option<u64>,
    pub scalar_representation: String,
    pub dimensions: Vec<u64>,
    pub endianness: String,
    pub special_value_encoding: String,
    pub ordered_items: Vec<LogicalPayloadItem>,
}

impl LogicalPayloadEnvelopeV2 {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        crate::canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 2 {
            return Err(CacheError::InvalidManifest(
                "unsupported logical payload schema".into(),
            ));
        }
        self.clone().into_internal(Vec::new()).validate()
    }

    fn into_internal(
        self,
        dependencies: Vec<PayloadDependencyIdentity>,
    ) -> CanonicalPayloadEnvelope {
        CanonicalPayloadEnvelope {
            schema_version: self.schema_version,
            scalar_backend: self.scalar_backend,
            precision_bits: self.precision_bits,
            scalar_representation: self.scalar_representation,
            dimensions: self.dimensions,
            endianness: self.endianness,
            special_value_encoding: self.special_value_encoding,
            ordered_items: self.ordered_items,
            dependencies,
        }
    }
}

impl TryFrom<&CanonicalPayloadEnvelope> for LogicalPayloadEnvelopeV2 {
    type Error = CacheError;
    fn try_from(value: &CanonicalPayloadEnvelope) -> Result<Self, Self::Error> {
        value.validate()?;
        if value.schema_version != 2 {
            return Err(CacheError::InvalidManifest(
                "legacy payload requires its original hash rules".into(),
            ));
        }
        Ok(Self {
            schema_version: 2,
            scalar_backend: value.scalar_backend.clone(),
            precision_bits: value.precision_bits,
            scalar_representation: value.scalar_representation.clone(),
            dimensions: value.dimensions.clone(),
            endianness: value.endianness.clone(),
            special_value_encoding: value.special_value_encoding.clone(),
            ordered_items: value.ordered_items.clone(),
        })
    }
}

// The legacy wire type deliberately has no optional schema-2 fields. A v1
// object cannot smuggle new policy metadata into an old producer's identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalArtifactManifestV1 {
    pub schema_version: u32,
    pub artifact_family: String,
    pub semantic_key: SemanticKeyEnvelope,
    pub semantic_digest: ContentDigest,
    pub canonical_payload: CanonicalPayloadEnvelope,
    pub payload_digest: ContentDigest,
    pub transport_digests: Vec<ContentDigest>,
    pub resolved_mathematical_configuration_digest: ContentDigest,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub maximum_reader_version: Option<ToolkitVersion>,
    pub requested_assurance: AssuranceLevel,
    pub claim_scope: String,
    pub assumptions: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalArtifactManifestV2 {
    pub schema_version: u32,
    pub artifact_family: String,
    pub semantic_key: SemanticKeyEnvelope,
    pub semantic_digest: ContentDigest,
    pub canonical_payload: LogicalPayloadEnvelopeV2,
    pub payload_digest: ContentDigest,
    pub dependencies: Vec<PayloadDependencyIdentity>,
    pub transport_digests: Vec<ContentDigest>,
    pub resolved_mathematical_configuration_digest: ContentDigest,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub maximum_reader_version: Option<ToolkitVersion>,
    pub requested_assurance: AssuranceLevel,
    pub claim_scope: String,
    pub assumptions: Vec<String>,
    pub provenance: ManifestProvenanceV2,
}

impl Serialize for CanonicalArtifactManifest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.schema_version {
            1 if self.provenance.is_none() && self.canonical_payload.schema_version == 1 => {
                CanonicalArtifactManifestV1 {
                    schema_version: 1,
                    artifact_family: self.artifact_family.clone(),
                    semantic_key: self.semantic_key.clone(),
                    semantic_digest: self.semantic_digest.clone(),
                    canonical_payload: self.canonical_payload.clone(),
                    payload_digest: self.payload_digest.clone(),
                    transport_digests: self.transport_digests.clone(),
                    resolved_mathematical_configuration_digest: self
                        .resolved_mathematical_configuration_digest
                        .clone(),
                    producer_toolkit_version: self.producer_toolkit_version.clone(),
                    minimum_reader_version: self.minimum_reader_version.clone(),
                    maximum_reader_version: self.maximum_reader_version.clone(),
                    requested_assurance: self.requested_assurance,
                    claim_scope: self.claim_scope.clone(),
                    assumptions: self.assumptions.clone(),
                }
                .serialize(serializer)
            }
            2 => CanonicalArtifactManifestV2::try_from(self)
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
            _ => Err(serde::ser::Error::custom(
                "unsupported or inconsistent canonical manifest schema",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for CanonicalArtifactManifest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let decoded = match value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
        {
            Some(1) => {
                let old: CanonicalArtifactManifestV1 =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                if old.canonical_payload.schema_version != 1 {
                    return Err(serde::de::Error::custom(
                        "legacy manifest requires legacy payload",
                    ));
                }
                Self {
                    schema_version: 1,
                    artifact_family: old.artifact_family,
                    semantic_key: old.semantic_key,
                    semantic_digest: old.semantic_digest,
                    canonical_payload: old.canonical_payload,
                    payload_digest: old.payload_digest,
                    transport_digests: old.transport_digests,
                    resolved_mathematical_configuration_digest: old
                        .resolved_mathematical_configuration_digest,
                    producer_toolkit_version: old.producer_toolkit_version,
                    minimum_reader_version: old.minimum_reader_version,
                    maximum_reader_version: old.maximum_reader_version,
                    requested_assurance: old.requested_assurance,
                    claim_scope: old.claim_scope,
                    assumptions: old.assumptions,
                    provenance: None,
                }
            }
            Some(2) => {
                let new: CanonicalArtifactManifestV2 =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                new.try_into().map_err(serde::de::Error::custom)?
            }
            _ => {
                return Err(serde::de::Error::custom(
                    "unsupported canonical manifest schema",
                ))
            }
        };
        Ok(decoded)
    }
}

impl TryFrom<&CanonicalArtifactManifest> for CanonicalArtifactManifestV2 {
    type Error = CacheError;
    fn try_from(value: &CanonicalArtifactManifest) -> Result<Self, Self::Error> {
        if value.schema_version != 2 {
            return Err(CacheError::InvalidManifest(
                "legacy manifest requires its original wire format".into(),
            ));
        }
        let provenance = value.provenance.clone().ok_or_else(|| {
            CacheError::InvalidManifest("schema 2 requires provenance admission metadata".into())
        })?;
        provenance.validate()?;
        Ok(Self {
            schema_version: 2,
            artifact_family: value.artifact_family.clone(),
            semantic_key: value.semantic_key.clone(),
            semantic_digest: value.semantic_digest.clone(),
            canonical_payload: LogicalPayloadEnvelopeV2::try_from(&value.canonical_payload)?,
            payload_digest: value.payload_digest.clone(),
            dependencies: value.canonical_payload.dependencies.clone(),
            transport_digests: value.transport_digests.clone(),
            resolved_mathematical_configuration_digest: value
                .resolved_mathematical_configuration_digest
                .clone(),
            producer_toolkit_version: value.producer_toolkit_version.clone(),
            minimum_reader_version: value.minimum_reader_version.clone(),
            maximum_reader_version: value.maximum_reader_version.clone(),
            requested_assurance: value.requested_assurance,
            claim_scope: value.claim_scope.clone(),
            assumptions: value.assumptions.clone(),
            provenance,
        })
    }
}

impl TryFrom<CanonicalArtifactManifestV2> for CanonicalArtifactManifest {
    type Error = CacheError;
    fn try_from(value: CanonicalArtifactManifestV2) -> Result<Self, Self::Error> {
        if value.schema_version != 2 || value.canonical_payload.schema_version != 2 {
            return Err(CacheError::InvalidManifest(
                "schema 2 requires a schema-2 logical payload".into(),
            ));
        }
        value.canonical_payload.validate()?;
        value.provenance.validate()?;
        Ok(Self {
            schema_version: 2,
            artifact_family: value.artifact_family,
            semantic_key: value.semantic_key,
            semantic_digest: value.semantic_digest,
            canonical_payload: value.canonical_payload.into_internal(value.dependencies),
            payload_digest: value.payload_digest,
            transport_digests: value.transport_digests,
            resolved_mathematical_configuration_digest: value
                .resolved_mathematical_configuration_digest,
            producer_toolkit_version: value.producer_toolkit_version,
            minimum_reader_version: value.minimum_reader_version,
            maximum_reader_version: value.maximum_reader_version,
            requested_assurance: value.requested_assurance,
            claim_scope: value.claim_scope,
            assumptions: value.assumptions,
            provenance: Some(value.provenance),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    const LEGACY: &[u8] = include_bytes!("../tests/fixtures/canonical-manifest-v1.json");

    fn legacy() -> CanonicalArtifactManifest {
        serde_json::from_slice(LEGACY).unwrap()
    }

    pub(crate) fn current_fixture() -> CanonicalArtifactManifest {
        let mut manifest = legacy();
        manifest.schema_version = 2;
        manifest.canonical_payload.schema_version = 2;
        manifest.payload_digest = manifest.canonical_payload.digest().unwrap();
        manifest.provenance = Some(ManifestProvenanceV2 {
            producer_annotations: std::collections::BTreeMap::new(),
            disclosure: ArtifactDisclosure::PublicInputs,
            producer_receipts: Vec::new(),
        });
        manifest
    }

    #[test]
    fn disclosure_inheritance_preserves_canonical_restrictions() {
        use crate::{ArtifactKey, ArtifactManifest, CacheObjectRef, CacheQuality, CacheVisibility};
        let mut canonical = current_fixture();
        canonical.provenance.as_mut().unwrap().disclosure = ArtifactDisclosure::PrivateOnly;
        canonical.canonical_payload.ordered_items[0].normalized_path = "payload.json".into();
        canonical.payload_digest = canonical.canonical_payload.digest().unwrap();
        let logical = &canonical.canonical_payload.ordered_items[0];
        let mut adapter = ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey {
                kind: canonical.semantic_key.artifact_kind.clone(),
                logical_key: "synthetic/source".into(),
                parameters_digest: canonical.semantic_digest.clone(),
            },
            content_digest: logical.content_digest.clone(),
            size_bytes: logical.size_bytes,
            objects: vec![CacheObjectRef {
                content_digest: logical.content_digest.clone(),
                size_bytes: logical.size_bytes,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: canonical.producer_toolkit_version.clone(),
            minimum_reader_version: canonical.minimum_reader_version.clone(),
            maximum_reader_version: canonical.maximum_reader_version.clone(),
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags: std::collections::BTreeMap::from([
                (
                    crate::SEMANTIC_KEY_MANIFEST_TAG.into(),
                    serde_json::to_string(&canonical.semantic_key).unwrap(),
                ),
                (
                    crate::REMOTE_CANONICAL_MANIFEST_TAG.into(),
                    serde_json::to_string(&canonical).unwrap(),
                ),
            ]),
            provenance_digest: Some(canonical.digest().unwrap()),
        };
        let local = current_fixture().provenance.unwrap();
        adapter.tags.insert(
            MANIFEST_PROVENANCE_V2_TAG.into(),
            serde_json::to_string(&local).unwrap(),
        );
        let derived =
            provenance_for_numerical_inputs("ccm_tau_matrix", &[adapter], vec![]).unwrap();
        assert_eq!(derived.disclosure, ArtifactDisclosure::PrivateOnly);
    }

    #[test]
    fn legacy_manifest_roundtrips_without_a_byte_or_hash_change() {
        let manifest = legacy();
        assert_eq!(crate::canonical_json_bytes(&manifest).unwrap(), LEGACY);
        assert_eq!(manifest.digest().unwrap(), ContentDigest::sha256(LEGACY));
        // A schema-1 decoder cannot accept schema-2 bytes even when both
        // producers happen to have the same Toolkit release version.
        assert!(serde_json::from_slice::<CanonicalArtifactManifestV1>(
            &crate::canonical_json_bytes(&current_fixture()).unwrap()
        )
        .is_err());
    }

    #[test]
    fn content_identity_excludes_exact_ancestry_and_producer_receipts() {
        let mut first = current_fixture();
        first
            .canonical_payload
            .dependencies
            .push(PayloadDependencyIdentity {
                artifact_family: "ccm-matrices".into(),
                semantic_digest: ContentDigest::sha256(b"parent-request"),
                manifest_digest: ContentDigest::sha256(b"parent-producer-one"),
                payload_digest: ContentDigest::sha256(b"parent-values"),
            });
        let mut second = first.clone();
        second.producer_toolkit_version = ToolkitVersion::parse("0.17.1").unwrap();
        second.canonical_payload.dependencies[0].manifest_digest =
            ContentDigest::sha256(b"parent-producer-two");
        second
            .provenance
            .as_mut()
            .unwrap()
            .producer_receipts
            .push(ContentDigest::sha256(b"new build receipt"));
        assert_eq!(first.semantic_digest, second.semantic_digest);
        assert_eq!(
            first.canonical_payload.digest().unwrap(),
            second.canonical_payload.digest().unwrap()
        );
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
        let wire = serde_json::to_value(&second).unwrap();
        assert!(wire["canonical_payload"].get("dependencies").is_none());
        assert_eq!(wire["dependencies"].as_array().unwrap().len(), 1);
        let decoded: CanonicalArtifactManifest = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded, second);
        second.canonical_payload.ordered_items[0].content_digest =
            ContentDigest::sha256(b"changed values");
        assert_ne!(
            first.canonical_payload.digest().unwrap(),
            second.canonical_payload.digest().unwrap()
        );
        // Legacy ancestry remains part of its historical payload identity.
        first.schema_version = 1;
        first.canonical_payload.schema_version = 1;
        second.canonical_payload = first.canonical_payload.clone();
        second.canonical_payload.dependencies[0].manifest_digest =
            ContentDigest::sha256(b"different legacy ancestor");
        assert_ne!(
            first.canonical_payload.digest().unwrap(),
            second.canonical_payload.digest().unwrap()
        );
    }

    #[test]
    fn unknown_and_mixed_schemas_are_rejected() {
        let original = serde_json::to_value(current_fixture()).unwrap();
        for schema in [0, 1, 3, u32::MAX] {
            let mut invalid = original.clone();
            invalid["schema_version"] = schema.into();
            assert!(serde_json::from_value::<CanonicalArtifactManifest>(invalid).is_err());
        }
        let mut invalid = original;
        invalid["canonical_payload"]["schema_version"] = 1.into();
        assert!(serde_json::from_value::<CanonicalArtifactManifest>(invalid).is_err());
    }
}
