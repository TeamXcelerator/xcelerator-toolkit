//! Uniform, artifact-specific validation before cache reuse.

use crate::{
    protocol::canonical_digest, CacheError, CanonicalArtifactManifest, ContentDigest,
    LogicalPayloadItem, PayloadDependencyIdentity, ToolkitVersion,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactValidationFacet {
    Schema,
    Structure,
    Invariants,
    NumericMetadata,
    Dependencies,
    Hashes,
    Compatibility,
}

impl ArtifactValidationFacet {
    pub const ALL: [Self; 7] = [
        Self::Schema,
        Self::Structure,
        Self::Invariants,
        Self::NumericMetadata,
        Self::Dependencies,
        Self::Hashes,
        Self::Compatibility,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactValidationFacetEvidence {
    pub facet: ArtifactValidationFacet,
    pub passed: bool,
    pub reason: String,
    pub evidence_digests: Vec<ContentDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactValidationReport {
    pub schema_version: u32,
    pub validator_id: String,
    pub artifact_family: String,
    pub artifact_kind: String,
    pub reusable: bool,
    pub facets: Vec<ArtifactValidationFacetEvidence>,
}

pub type ArtifactInvariantValidator =
    fn(&CanonicalArtifactManifest) -> Result<Vec<ContentDigest>, String>;

#[derive(Clone)]
pub struct ArtifactValidatorRegistration {
    pub validator_id: String,
    pub artifact_family: String,
    pub artifact_kind: String,
    pub supported_manifest_schemas: BTreeSet<u32>,
    pub supported_payload_schemas: BTreeSet<u32>,
    pub allowed_scalar_backends: BTreeSet<String>,
    pub minimum_precision_bits: Option<u64>,
    pub invariant_validator: ArtifactInvariantValidator,
}

pub struct ArtifactValidationRequest<'a> {
    pub manifest: &'a CanonicalArtifactManifest,
    /// Dependency identities actually resolved for this reuse attempt.
    pub resolved_dependencies: &'a [PayloadDependencyIdentity],
    /// Logical items whose decoded bytes were checked against these records.
    pub verified_logical_items: &'a [LogicalPayloadItem],
    /// Numeric metadata observed by the artifact decoder.
    pub observed_scalar_backend: &'a str,
    pub observed_precision_bits: Option<u64>,
    pub observed_dimensions: &'a [u64],
    pub reader_version: &'a ToolkitVersion,
}

#[derive(Default)]
pub struct ArtifactValidatorRegistry {
    validators: BTreeMap<(String, String), ArtifactValidatorRegistration>,
}

impl ArtifactValidatorRegistry {
    pub fn register(
        &mut self,
        registration: ArtifactValidatorRegistration,
    ) -> Result<(), CacheError> {
        if registration.validator_id.trim().is_empty()
            || registration.artifact_family.trim().is_empty()
            || registration.artifact_kind.trim().is_empty()
            || registration.supported_manifest_schemas.is_empty()
            || registration.supported_manifest_schemas.contains(&0)
            || registration.supported_payload_schemas.is_empty()
            || registration.supported_payload_schemas.contains(&0)
            || registration
                .allowed_scalar_backends
                .iter()
                .any(|backend| backend.trim().is_empty())
            || registration.minimum_precision_bits == Some(0)
        {
            return Err(CacheError::InvalidManifest(
                "artifact validator registration is incomplete".to_owned(),
            ));
        }
        let key = (
            registration.artifact_family.clone(),
            registration.artifact_kind.clone(),
        );
        if self.validators.insert(key.clone(), registration).is_some() {
            return Err(CacheError::InvalidManifest(format!(
                "duplicate validator for artifact family {:?} kind {:?}",
                key.0, key.1
            )));
        }
        Ok(())
    }

    pub fn validate_coverage(
        &self,
        declared_artifact_types: &BTreeSet<(String, String)>,
    ) -> Result<(), CacheError> {
        if let Some((family, kind)) = declared_artifact_types
            .iter()
            .find(|artifact_type| !self.validators.contains_key(*artifact_type))
        {
            return Err(CacheError::InvalidManifest(format!(
                "artifact family {family:?} kind {kind:?} has no registered validator"
            )));
        }
        Ok(())
    }

    pub fn validate_for_reuse(
        &self,
        request: &ArtifactValidationRequest<'_>,
    ) -> Result<ArtifactValidationReport, CacheError> {
        let family = request.manifest.artifact_family.as_str();
        let kind = request.manifest.semantic_key.artifact_kind.as_str();
        let validator = self
            .validators
            .get(&(family.to_owned(), kind.to_owned()))
            .ok_or_else(|| {
                CacheError::InvalidManifest(format!(
                    "artifact family {family:?} kind {kind:?} has no registered validator"
                ))
            })?;
        let mut facets = Vec::with_capacity(ArtifactValidationFacet::ALL.len());
        let mut record = |facet, passed, reason: String, evidence_digests| {
            facets.push(ArtifactValidationFacetEvidence {
                facet,
                passed,
                reason,
                evidence_digests,
            });
        };

        let schema_ok = validator
            .supported_manifest_schemas
            .contains(&request.manifest.schema_version)
            && validator
                .supported_payload_schemas
                .contains(&request.manifest.canonical_payload.schema_version);
        record(
            ArtifactValidationFacet::Schema,
            schema_ok,
            if schema_ok {
                "manifest and payload schemas are supported".to_owned()
            } else {
                format!(
                    "unsupported manifest schema {} or payload schema {}",
                    request.manifest.schema_version,
                    request.manifest.canonical_payload.schema_version
                )
            },
            Vec::new(),
        );

        match request.manifest.validate() {
            Ok(()) => record(
                ArtifactValidationFacet::Structure,
                true,
                "canonical manifest structure is valid".to_owned(),
                vec![request.manifest.digest()?],
            ),
            Err(error) => record(
                ArtifactValidationFacet::Structure,
                false,
                error.to_string(),
                Vec::new(),
            ),
        }

        match (validator.invariant_validator)(request.manifest) {
            Ok(evidence)
                if !evidence.is_empty() && evidence.iter().all(ContentDigest::validate) =>
            {
                record(
                    ArtifactValidationFacet::Invariants,
                    true,
                    "artifact-specific mathematical invariants passed".to_owned(),
                    evidence,
                );
            }
            Ok(_) => record(
                ArtifactValidationFacet::Invariants,
                false,
                "artifact-specific invariant validator returned no valid evidence".to_owned(),
                Vec::new(),
            ),
            Err(reason) => record(
                ArtifactValidationFacet::Invariants,
                false,
                reason,
                Vec::new(),
            ),
        }

        let payload = &request.manifest.canonical_payload;
        let numeric_ok = request.observed_scalar_backend == payload.scalar_backend
            && request.observed_precision_bits == payload.precision_bits
            && request.observed_dimensions == payload.dimensions
            && (validator.allowed_scalar_backends.is_empty()
                || validator
                    .allowed_scalar_backends
                    .contains(request.observed_scalar_backend))
            && validator
                .minimum_precision_bits
                .is_none_or(|minimum| request.observed_precision_bits.unwrap_or(0) >= minimum);
        record(
            ArtifactValidationFacet::NumericMetadata,
            numeric_ok,
            if numeric_ok {
                "decoded numeric metadata matches the manifest and validator policy".to_owned()
            } else {
                "decoded backend, precision, or dimensions do not match the manifest and validator policy"
                    .to_owned()
            },
            Vec::new(),
        );

        let dependencies_ok = request.resolved_dependencies == payload.dependencies;
        record(
            ArtifactValidationFacet::Dependencies,
            dependencies_ok,
            if dependencies_ok {
                "exact dependency identities are resolved in canonical order".to_owned()
            } else {
                "resolved dependency identities differ from the canonical payload".to_owned()
            },
            request
                .resolved_dependencies
                .iter()
                .map(|dependency| dependency.manifest_digest.clone())
                .collect(),
        );

        let hashes_ok = request.verified_logical_items == payload.ordered_items;
        record(
            ArtifactValidationFacet::Hashes,
            hashes_ok,
            if hashes_ok {
                "decoded logical item sizes and content hashes match".to_owned()
            } else {
                "decoded logical item sizes or content hashes differ from the canonical payload"
                    .to_owned()
            },
            request
                .verified_logical_items
                .iter()
                .map(|item| item.content_digest.clone())
                .collect(),
        );

        let compatibility_ok = request.reader_version >= &request.manifest.minimum_reader_version
            && request
                .manifest
                .maximum_reader_version
                .as_ref()
                .is_none_or(|maximum| request.reader_version <= maximum);
        record(
            ArtifactValidationFacet::Compatibility,
            compatibility_ok,
            if compatibility_ok {
                "reader version is inside the manifest compatibility range".to_owned()
            } else {
                format!(
                    "reader version {} is outside the manifest compatibility range",
                    request.reader_version
                )
            },
            Vec::new(),
        );

        let reusable = facets.iter().all(|facet| facet.passed);
        let report = ArtifactValidationReport {
            schema_version: 1,
            validator_id: validator.validator_id.clone(),
            artifact_family: family.to_owned(),
            artifact_kind: kind.to_owned(),
            reusable,
            facets,
        };
        canonical_digest(&report)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalPayloadEnvelope, SemanticKeyEnvelope};
    use serde_json::json;
    use xc_core::AssuranceLevel;

    fn invariant(manifest: &CanonicalArtifactManifest) -> Result<Vec<ContentDigest>, String> {
        (manifest.claim_scope == "fixture theorem")
            .then(|| vec![ContentDigest::sha256(b"fixture-invariant-evidence")])
            .ok_or_else(|| "fixture theorem invariant failed".to_owned())
    }

    fn manifest() -> CanonicalArtifactManifest {
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "fixture_matrix".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: json!({"rank": 2}),
            normalization: Some("canonical".to_owned()),
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "mpfr".to_owned(),
            precision_bits: Some(256),
            scalar_representation: "canonical-binary".to_owned(),
            dimensions: vec![2, 2],
            endianness: "big".to_owned(),
            special_value_encoding: "none".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "matrix.bin".to_owned(),
                content_digest: ContentDigest::sha256(b"matrix"),
                size_bytes: 6,
            }],
            dependencies: Vec::new(),
        };
        CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "fixture".to_owned(),
            semantic_digest: semantic_key.digest().unwrap(),
            payload_digest: payload.digest().unwrap(),
            semantic_key,
            canonical_payload: payload,
            transport_digests: vec![ContentDigest::sha256(b"transport")],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"configuration"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: Some(ToolkitVersion::parse("0.13.9").unwrap()),
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "fixture theorem".to_owned(),
            assumptions: Vec::new(),
        }
    }

    fn registry() -> ArtifactValidatorRegistry {
        let mut registry = ArtifactValidatorRegistry::default();
        registry
            .register(ArtifactValidatorRegistration {
                validator_id: "fixture-matrix-v1".to_owned(),
                artifact_family: "fixture".to_owned(),
                artifact_kind: "fixture_matrix".to_owned(),
                supported_manifest_schemas: BTreeSet::from([1]),
                supported_payload_schemas: BTreeSet::from([1]),
                allowed_scalar_backends: BTreeSet::from(["mpfr".to_owned()]),
                minimum_precision_bits: Some(256),
                invariant_validator: invariant,
            })
            .unwrap();
        registry
    }

    #[test]
    fn every_validation_facet_must_pass_before_reuse() {
        let manifest = manifest();
        let report = registry()
            .validate_for_reuse(&ArtifactValidationRequest {
                resolved_dependencies: &manifest.canonical_payload.dependencies,
                verified_logical_items: &manifest.canonical_payload.ordered_items,
                observed_scalar_backend: "mpfr",
                observed_precision_bits: Some(256),
                observed_dimensions: &[2, 2],
                reader_version: &ToolkitVersion::parse("0.13.0").unwrap(),
                manifest: &manifest,
            })
            .unwrap();
        assert!(report.reusable);
        assert_eq!(report.facets.len(), ArtifactValidationFacet::ALL.len());
        assert!(report.facets.iter().all(|facet| facet.passed));
    }

    #[test]
    fn invalid_numeric_metadata_and_incompatible_reader_have_specific_reasons() {
        let manifest = manifest();
        let report = registry()
            .validate_for_reuse(&ArtifactValidationRequest {
                resolved_dependencies: &manifest.canonical_payload.dependencies,
                verified_logical_items: &manifest.canonical_payload.ordered_items,
                observed_scalar_backend: "binary64",
                observed_precision_bits: Some(53),
                observed_dimensions: &[4],
                reader_version: &ToolkitVersion::parse("0.14.0").unwrap(),
                manifest: &manifest,
            })
            .unwrap();
        assert!(!report.reusable);
        assert!(report.facets.iter().any(|facet| {
            facet.facet == ArtifactValidationFacet::NumericMetadata
                && !facet.passed
                && facet.reason.contains("backend, precision, or dimensions")
        }));
        assert!(report.facets.iter().any(|facet| {
            facet.facet == ArtifactValidationFacet::Compatibility
                && !facet.passed
                && facet.reason.contains("0.14.0")
        }));
    }

    #[test]
    fn registry_rejects_an_artifact_type_without_a_validator() {
        let missing = BTreeSet::from([("mk".to_owned(), "operator".to_owned())]);
        let error = registry().validate_coverage(&missing).unwrap_err();
        assert!(error.to_string().contains("has no registered validator"));
    }
}
