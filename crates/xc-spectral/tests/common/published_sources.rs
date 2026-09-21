use std::collections::BTreeMap;
use xc_cache::*;

// Reproduce the metadata returned by a shard: exact canonical identities in
// tags, with an empty key-based dependency list on the adapter.
pub fn published(mut m: ArtifactManifest, parents: &[&ArtifactManifest]) -> ArtifactManifest {
    let semantic = m
        .tags
        .get(SEMANTIC_KEY_MANIFEST_TAG)
        .map(|s| serde_json::from_str::<SemanticKeyEnvelope>(s).unwrap())
        .unwrap_or_else(|| SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: m.key.kind.clone(),
            mathematical_semantics_version: "published-source-regression-v1".into(),
            resolved_mathematical_parameters: serde_json::json!({"fixture":m.key.logical_key}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        });
    let mut dependencies = parents
        .iter()
        .map(|p| {
            let c: CanonicalArtifactManifest =
                serde_json::from_str(&p.tags[REMOTE_CANONICAL_MANIFEST_TAG]).unwrap();
            PayloadDependencyIdentity {
                artifact_family: c.artifact_family.clone(),
                semantic_digest: c.semantic_digest.clone(),
                manifest_digest: c.digest().unwrap(),
                payload_digest: c.payload_digest.clone(),
            }
        })
        .collect::<Vec<_>>();
    dependencies.sort_by_key(|d| {
        (
            d.artifact_family.clone(),
            d.semantic_digest.clone(),
            d.manifest_digest.clone(),
            d.payload_digest.clone(),
        )
    });
    let payload = CanonicalPayloadEnvelope {
        schema_version: 1,
        scalar_backend: "canonical_json".into(),
        precision_bits: Some(128),
        scalar_representation: "canonical-json-utf8-v1".into(),
        dimensions: vec![],
        endianness: "not-applicable".into(),
        special_value_encoding: "decimal-string-or-json-number-v1".into(),
        ordered_items: vec![LogicalPayloadItem {
            normalized_path: "payload.json".into(),
            content_digest: m.content_digest.clone(),
            size_bytes: m.size_bytes,
        }],
        dependencies,
    };
    let canonical = CanonicalArtifactManifest {
        schema_version: 1,
        artifact_family: family_for_artifact_kind(&m.key.kind).unwrap().into(),
        semantic_digest: semantic.digest().unwrap(),
        semantic_key: semantic.clone(),
        payload_digest: payload.digest().unwrap(),
        canonical_payload: payload,
        transport_digests: vec![ContentDigest::sha256(b"fixture transport")],
        resolved_mathematical_configuration_digest: ContentDigest::sha256(b"fixture configuration"),
        producer_toolkit_version: m.producer_toolkit_version.clone(),
        minimum_reader_version: m.minimum_reader_version.clone(),
        maximum_reader_version: m.maximum_reader_version.clone(),
        requested_assurance: xc_core::AssuranceLevel::Computed,
        claim_scope: "synthetic published-source regression".into(),
        assumptions: vec![],
    };
    canonical.validate().unwrap();
    m.key.parameters_digest = canonical.semantic_digest.clone();
    m.provenance_digest = Some(canonical.digest().unwrap());
    m.tags.insert(
        SEMANTIC_KEY_MANIFEST_TAG.into(),
        serde_json::to_string(&semantic).unwrap(),
    );
    m.tags.insert(
        REMOTE_CANONICAL_MANIFEST_TAG.into(),
        serde_json::to_string(&canonical).unwrap(),
    );
    m.dependencies.clear();
    m
}
