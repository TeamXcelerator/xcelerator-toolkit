//! Canonical cache identities and independent artifact state axes.

use crate::{CacheError, ContentDigest, ToolkitVersion};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use xc_core::AssuranceLevel;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticKeyEnvelope {
    pub schema_version: u32,
    pub artifact_kind: String,
    pub mathematical_semantics_version: String,
    pub resolved_mathematical_parameters: Value,
    pub normalization: Option<String>,
    pub target: Option<String>,
    pub subspace: Option<String>,
    pub source_data_identities: BTreeMap<String, ContentDigest>,
    /// Present only when the algorithm changes mathematical meaning.
    pub algorithm_semantics: Option<String>,
}

impl SemanticKeyEnvelope {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.artifact_kind.trim().is_empty()
            || self.mathematical_semantics_version.trim().is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "semantic identity requires a schema, artifact kind, and semantics version"
                    .to_owned(),
            ));
        }
        validate_digest_map(&self.source_data_identities, "source data")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalPayloadItem {
    pub normalized_path: String,
    pub content_digest: ContentDigest,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadDependencyIdentity {
    pub artifact_family: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub payload_digest: ContentDigest,
}

impl PayloadDependencyIdentity {
    /// Reject degenerate identities before any path construction or digest
    /// slicing. Identities cross trust boundaries (they arrive inside retained
    /// canonical manifests), so a malformed digest must surface as an invalid
    /// manifest, never as a panic in a slice or a strange repository path.
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.artifact_family.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "dependency identity requires a nonempty artifact family".to_owned(),
            ));
        }
        for (field, digest) in [
            ("semantic", &self.semantic_digest),
            ("manifest", &self.manifest_digest),
            ("payload", &self.payload_digest),
        ] {
            if !digest.validate() {
                return Err(CacheError::InvalidManifest(format!(
                    "dependency identity {field} digest is not a canonical sha-256 digest"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalPayloadEnvelope {
    pub schema_version: u32,
    pub scalar_backend: String,
    pub precision_bits: Option<u64>,
    pub scalar_representation: String,
    pub dimensions: Vec<u64>,
    pub endianness: String,
    pub special_value_encoding: String,
    pub ordered_items: Vec<LogicalPayloadItem>,
    pub dependencies: Vec<PayloadDependencyIdentity>,
}

impl CanonicalPayloadEnvelope {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.scalar_backend.trim().is_empty()
            || self.precision_bits == Some(0)
            || self.scalar_representation.trim().is_empty()
            || self.endianness.trim().is_empty()
            || self.special_value_encoding.trim().is_empty()
            || self.ordered_items.is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "canonical payload identity is incomplete".to_owned(),
            ));
        }
        let mut previous_path: Option<&str> = None;
        for item in &self.ordered_items {
            if !normalized_relative_path(&item.normalized_path)
                || !item.content_digest.validate()
                || previous_path.is_some_and(|previous| previous >= item.normalized_path.as_str())
            {
                return Err(CacheError::InvalidManifest(format!(
                    "invalid, duplicate, or non-canonical payload path {:?}",
                    item.normalized_path
                )));
            }
            previous_path = Some(&item.normalized_path);
        }
        let mut previous_dependency: Option<(
            &str,
            &ContentDigest,
            &ContentDigest,
            &ContentDigest,
        )> = None;
        for dependency in &self.dependencies {
            if dependency.artifact_family.trim().is_empty()
                || !dependency.semantic_digest.validate()
                || !dependency.manifest_digest.validate()
                || !dependency.payload_digest.validate()
            {
                return Err(CacheError::InvalidManifest(
                    "payload dependency contains an invalid digest".to_owned(),
                ));
            }
            let identity = (
                dependency.artifact_family.as_str(),
                &dependency.semantic_digest,
                &dependency.manifest_digest,
                &dependency.payload_digest,
            );
            if previous_dependency.is_some_and(|previous| previous >= identity) {
                return Err(CacheError::InvalidManifest(
                    "payload dependencies must be unique and canonically ordered".to_owned(),
                ));
            }
            previous_dependency = Some(identity);
        }
        Ok(())
    }

    pub fn logical_size_bytes(&self) -> u64 {
        self.ordered_items
            .iter()
            .fold(0u64, |total, item| total.saturating_add(item.size_bytes))
    }
}

/// Immutable identity-bearing manifest. Event time, actor, location,
/// disposition, and publication status live in linked state/attestation data.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalArtifactManifest {
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

impl CanonicalArtifactManifest {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.artifact_family.trim().is_empty()
            || self.claim_scope.trim().is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "canonical manifest requires a schema, artifact family, and claim scope".to_owned(),
            ));
        }
        let semantic_digest = self.semantic_key.digest()?;
        let payload_digest = self.canonical_payload.digest()?;
        if semantic_digest != self.semantic_digest || payload_digest != self.payload_digest {
            return Err(CacheError::InvalidManifest(
                "canonical manifest identity records do not match their declared digests"
                    .to_owned(),
            ));
        }
        for digest in [
            &self.semantic_digest,
            &self.payload_digest,
            &self.resolved_mathematical_configuration_digest,
        ]
        .into_iter()
        .chain(self.transport_digests.iter())
        {
            if !digest.validate() {
                return Err(CacheError::InvalidManifest(
                    "canonical manifest contains an invalid digest".to_owned(),
                ));
            }
        }
        if self.transport_digests.is_empty()
            || self
                .transport_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(CacheError::InvalidManifest(
                "canonical manifest transport identities are empty, duplicated, or unordered"
                    .to_owned(),
            ));
        }
        if self
            .maximum_reader_version
            .as_ref()
            .is_some_and(|maximum| maximum < &self.minimum_reader_version)
        {
            return Err(CacheError::InvalidManifest(
                "maximum reader version precedes minimum reader version".to_owned(),
            ));
        }
        crate::artifact_compatibility_policy(
            &self.artifact_family,
            &self.semantic_key.artifact_kind,
        )?
        .validate_manifest_versions(
            self.schema_version,
            &self.producer_toolkit_version,
            &self.minimum_reader_version,
            self.maximum_reader_version.as_ref(),
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactCompletionState {
    Planned,
    InProgress,
    Partial,
    Complete,
    Failed,
    Abandoned,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactAssuranceState {
    Unchecked,
    StructurallyValidated,
    Computed,
    CrossChecked,
    Certified,
}

impl ArtifactAssuranceState {
    pub fn mathematical(self) -> Option<AssuranceLevel> {
        match self {
            Self::Unchecked | Self::StructurallyValidated => None,
            Self::Computed => Some(AssuranceLevel::Computed),
            Self::CrossChecked => Some(AssuranceLevel::CrossChecked),
            Self::Certified => Some(AssuranceLevel::Certified),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDisposition {
    Active,
    Deprecated,
    Quarantined,
    Revoked,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactLocationKind {
    Process,
    Workstation,
    ProjectPrivate,
    TeamPrivate,
    Public,
    ExportBundle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactLocation {
    pub kind: ArtifactLocationKind,
    pub locator: String,
    pub verified: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationDestination {
    Private,
    Public,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationTargetState {
    Planned,
    Uploading,
    BatchVerified,
    RemoteVerified,
    ReceiptComplete,
    Failed,
    Abandoned,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssuranceTransition {
    pub from: ArtifactAssuranceState,
    pub to: ArtifactAssuranceState,
    pub evidence_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactState {
    pub completion: ArtifactCompletionState,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub locations: Vec<ArtifactLocation>,
    pub publication: BTreeMap<PublicationDestination, PublicationTargetState>,
    pub assurance_history: Vec<AssuranceTransition>,
}

impl Default for ArtifactState {
    fn default() -> Self {
        Self {
            completion: ArtifactCompletionState::Planned,
            achieved_assurance: ArtifactAssuranceState::Unchecked,
            disposition: ArtifactDisposition::Active,
            locations: Vec::new(),
            publication: BTreeMap::new(),
            assurance_history: Vec::new(),
        }
    }
}

impl ArtifactState {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.achieved_assurance == ArtifactAssuranceState::Certified
            && self.completion != ArtifactCompletionState::Complete
        {
            return Err(CacheError::InvalidManifest(
                "Certified assurance requires a complete artifact".to_owned(),
            ));
        }
        for location in &self.locations {
            if location.locator.trim().is_empty() {
                return Err(CacheError::InvalidManifest(
                    "artifact location must have a nonempty locator".to_owned(),
                ));
            }
        }
        for state in self.publication.values() {
            if matches!(
                state,
                PublicationTargetState::Uploading
                    | PublicationTargetState::BatchVerified
                    | PublicationTargetState::RemoteVerified
                    | PublicationTargetState::ReceiptComplete
            ) && self.completion != ArtifactCompletionState::Complete
            {
                return Err(CacheError::InvalidManifest(
                    "publication may not advance before artifact completion".to_owned(),
                ));
            }
            if *state == PublicationTargetState::ReceiptComplete
                && self.disposition != ArtifactDisposition::Active
            {
                return Err(CacheError::InvalidManifest(
                    "a non-active artifact may not acquire a completed publication receipt"
                        .to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn record_location(&mut self, location: ArtifactLocation) -> Result<(), CacheError> {
        if location.locator.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "artifact location must have a nonempty locator".to_owned(),
            ));
        }
        if !self.locations.contains(&location) {
            self.locations.push(location);
        }
        Ok(())
    }

    pub fn promote_assurance(
        &mut self,
        next: ArtifactAssuranceState,
        evidence_digest: ContentDigest,
    ) -> Result<(), CacheError> {
        if !evidence_digest.validate() {
            return Err(CacheError::InvalidTransition(
                "assurance transition evidence digest is invalid".to_owned(),
            ));
        }
        if next <= self.achieved_assurance {
            return Err(CacheError::InvalidTransition(format!(
                "assurance must increase from {:?}, got {:?}",
                self.achieved_assurance, next
            )));
        }
        if next == ArtifactAssuranceState::Certified
            && self.completion != ArtifactCompletionState::Complete
        {
            return Err(CacheError::InvalidTransition(
                "Certified assurance requires a complete artifact".to_owned(),
            ));
        }
        let from = self.achieved_assurance;
        self.achieved_assurance = next;
        self.assurance_history.push(AssuranceTransition {
            from,
            to: next,
            evidence_digest,
        });
        Ok(())
    }

    pub fn transition_publication(
        &mut self,
        destination: PublicationDestination,
        next: PublicationTargetState,
    ) -> Result<(), CacheError> {
        let current = self.publication.get(&destination).copied();
        let allowed = match (current, next) {
            (None, PublicationTargetState::Planned)
            | (Some(PublicationTargetState::Planned), PublicationTargetState::Uploading)
            | (Some(PublicationTargetState::Uploading), PublicationTargetState::BatchVerified)
            | (Some(PublicationTargetState::BatchVerified), PublicationTargetState::Uploading)
            | (
                Some(PublicationTargetState::BatchVerified),
                PublicationTargetState::RemoteVerified,
            )
            | (
                Some(PublicationTargetState::RemoteVerified),
                PublicationTargetState::ReceiptComplete,
            )
            | (Some(PublicationTargetState::Failed), PublicationTargetState::Planned) => true,
            (Some(current), next) if current == next => true,
            (_, PublicationTargetState::Failed | PublicationTargetState::Abandoned) => true,
            _ => false,
        };
        if !allowed {
            return Err(CacheError::InvalidTransition(format!(
                "publication {:?} may not move from {:?} to {:?}",
                destination, current, next
            )));
        }
        let previous_assurance = self.achieved_assurance;
        self.publication.insert(destination, next);
        if let Err(error) = self.validate() {
            match current {
                Some(current) => {
                    self.publication.insert(destination, current);
                }
                None => {
                    self.publication.remove(&destination);
                }
            }
            return Err(error);
        }
        debug_assert_eq!(self.achieved_assurance, previous_assurance);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttestationKind {
    Validation,
    Certification,
    Publication,
    Revocation,
    Receipt,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationEnvelope {
    pub schema_version: u32,
    pub kind: AttestationKind,
    pub subject_digest: ContentDigest,
    pub actor: String,
    pub policy_digest: ContentDigest,
    pub execution_fingerprint_digest: ContentDigest,
    pub producer_toolkit_version: ToolkitVersion,
    pub dependency_versions: BTreeMap<String, String>,
    pub source_revision: String,
    pub event_unix_seconds: u64,
    pub location: Option<String>,
    pub evidence_digests: Vec<ContentDigest>,
}

impl AttestationEnvelope {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        if self.schema_version == 0
            || self.actor.trim().is_empty()
            || self.source_revision.trim().is_empty()
            || !self.subject_digest.validate()
            || !self.policy_digest.validate()
            || !self.execution_fingerprint_digest.validate()
            || self
                .producer_toolkit_version
                .prerelease
                .as_ref()
                .is_some_and(|value| value.trim().is_empty())
            || self.dependency_versions.is_empty()
            || self
                .dependency_versions
                .iter()
                .any(|(name, version)| name.trim().is_empty() || version.trim().is_empty())
            || self
                .evidence_digests
                .iter()
                .any(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "attestation identity is incomplete".to_owned(),
            ));
        }
        canonical_digest(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkedArtifactRole {
    Certificate,
    ReferenceDataset,
    ValidationReport,
    PublicationReadyExport,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactPreservationClass {
    ExploratoryOutput,
    ReferenceDataset,
    GeneratedCertificate,
    FrozenPublication,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkedArtifactReference {
    pub role: LinkedArtifactRole,
    pub preservation_class: ArtifactPreservationClass,
    pub artifact_family: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub payload_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactLinkSet {
    pub schema_version: u32,
    pub subject_semantic_digest: ContentDigest,
    pub subject_payload_digest: ContentDigest,
    pub links: Vec<LinkedArtifactReference>,
}

impl ArtifactLinkSet {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        if self.schema_version != 1
            || !self.subject_semantic_digest.validate()
            || !self.subject_payload_digest.validate()
            || self.links.is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "artifact link set requires an exact subject and linked artifacts".to_owned(),
            ));
        }
        let mut previous = None;
        for link in &self.links {
            if link.artifact_family.trim().is_empty()
                || !link.semantic_digest.validate()
                || !link.manifest_digest.validate()
                || !link.payload_digest.validate()
            {
                return Err(CacheError::InvalidManifest(
                    "linked artifact identity is incomplete".to_owned(),
                ));
            }
            let compatible = matches!(
                (link.role, link.preservation_class),
                (
                    LinkedArtifactRole::Certificate,
                    ArtifactPreservationClass::GeneratedCertificate
                ) | (
                    LinkedArtifactRole::ReferenceDataset,
                    ArtifactPreservationClass::ReferenceDataset
                ) | (
                    LinkedArtifactRole::PublicationReadyExport,
                    ArtifactPreservationClass::FrozenPublication
                ) | (
                    LinkedArtifactRole::ValidationReport,
                    ArtifactPreservationClass::ExploratoryOutput
                        | ArtifactPreservationClass::FrozenPublication
                )
            );
            if !compatible {
                return Err(CacheError::InvalidManifest(
                    "linked artifact role is incompatible with its preservation class".to_owned(),
                ));
            }
            let current = (
                link.role,
                link.preservation_class,
                &link.semantic_digest,
                &link.manifest_digest,
                &link.payload_digest,
            );
            if previous.is_some_and(|previous| previous >= current) {
                return Err(CacheError::InvalidManifest(
                    "artifact links are duplicated or not canonically ordered".to_owned(),
                ));
            }
            previous = Some(current);
        }
        canonical_digest(self)
    }
}

pub(crate) fn canonical_digest<T: Serialize>(value: &T) -> Result<ContentDigest, CacheError> {
    Ok(ContentDigest::sha256(&canonical_json_bytes(value)?))
}

pub(crate) fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, CacheError> {
    xc_core::validate_secret_free(value, "canonical cache record")
        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
    let value = serde_json::to_value(value)?;
    let mut canonical = String::new();
    write_canonical_json(&value, &mut canonical)?;
    Ok(canonical.into_bytes())
}

fn write_canonical_json(value: &Value, output: &mut String) -> Result<(), CacheError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => output.push_str(&serde_json::to_string(value)?),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push(':');
                write_canonical_json(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn validate_digest_map(
    values: &BTreeMap<String, ContentDigest>,
    description: &str,
) -> Result<(), CacheError> {
    if values
        .iter()
        .any(|(name, digest)| name.trim().is_empty() || !digest.validate())
    {
        return Err(CacheError::InvalidManifest(format!(
            "{description} identities contain an invalid name or digest"
        )));
    }
    Ok(())
}

pub(crate) fn normalized_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.starts_with('\\')
        && !path.contains('\\')
        && !path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic(parameters: Value) -> SemanticKeyEnvelope {
        SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_matrix".to_owned(),
            mathematical_semantics_version: "ccm-form-v2".to_owned(),
            resolved_mathematical_parameters: parameters,
            normalization: Some("weil".to_owned()),
            target: None,
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        }
    }

    #[test]
    fn semantic_digest_is_independent_of_json_object_order() {
        let first = semantic(serde_json::json!({"n": 120, "lambda": "13"}));
        let second = semantic(serde_json::json!({"lambda": "13", "n": 120}));
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn canonical_cache_records_reject_credential_fields_without_echoing_values() {
        let record = semantic(serde_json::json!({"access_token": "do-not-echo"}));
        let error = record.digest().unwrap_err().to_string();
        assert!(error.contains("credential-bearing field"));
        assert!(!error.contains("do-not-echo"));
    }

    #[test]
    fn canonical_payload_items_require_bytewise_path_order() {
        let mut envelope = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "opaque-bytes-v1".to_owned(),
            dimensions: vec![2],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![
                LogicalPayloadItem {
                    normalized_path: "b.bin".to_owned(),
                    content_digest: ContentDigest::sha256(b"b"),
                    size_bytes: 1,
                },
                LogicalPayloadItem {
                    normalized_path: "a.bin".to_owned(),
                    content_digest: ContentDigest::sha256(b"a"),
                    size_bytes: 1,
                },
            ],
            dependencies: Vec::new(),
        };
        assert!(envelope.validate().is_err());
        envelope.ordered_items.reverse();
        assert!(envelope.validate().is_ok());
    }

    #[test]
    fn canonical_payload_dependencies_require_identity_order() {
        let dependency = |family: &str| PayloadDependencyIdentity {
            artifact_family: family.to_owned(),
            semantic_digest: ContentDigest::sha256(family.as_bytes()),
            manifest_digest: ContentDigest::sha256(b"manifest"),
            payload_digest: ContentDigest::sha256(b"payload"),
        };
        let mut envelope = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "opaque-bytes-v1".to_owned(),
            dimensions: vec![1],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(b"value"),
                size_bytes: 5,
            }],
            dependencies: vec![dependency("zeta"), dependency("alpha")],
        };
        assert!(envelope.validate().is_err());
        envelope.dependencies.reverse();
        assert!(envelope.validate().is_ok());
        envelope.dependencies.push(dependency("zeta"));
        assert!(envelope.validate().is_err());
    }

    #[test]
    fn canonical_manifest_is_self_contained_and_rejects_digest_drift() {
        let semantic_key = semantic(serde_json::json!({"n": 120}));
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "opaque-bytes-v1".to_owned(),
            dimensions: vec![1],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(b"value"),
                size_bytes: 5,
            }],
            dependencies: Vec::new(),
        };
        let mut manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm".to_owned(),
            semantic_digest: semantic_key.digest().unwrap(),
            semantic_key,
            payload_digest: canonical_payload.digest().unwrap(),
            canonical_payload,
            transport_digests: vec![ContentDigest::sha256(b"transport")],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"config"),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "self-contained fixture".to_owned(),
            assumptions: Vec::new(),
        };
        assert!(manifest.validate().is_ok());
        manifest.payload_digest = ContentDigest::sha256(b"other-payload");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn run_provenance_changes_attestation_without_fragmenting_canonical_identity() {
        let semantic_key = semantic(serde_json::json!({"n": 120}));
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "mpfr".to_owned(),
            precision_bits: Some(256),
            scalar_representation: "mpfr-256-bit-v1".to_owned(),
            dimensions: vec![1],
            endianness: "big".to_owned(),
            special_value_encoding: "mpfr-canonical-v1".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "value.bin".to_owned(),
                content_digest: ContentDigest::sha256(b"value"),
                size_bytes: 5,
            }],
            dependencies: Vec::new(),
        };
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm".to_owned(),
            semantic_digest: semantic_key.digest().unwrap(),
            semantic_key,
            payload_digest: canonical_payload.digest().unwrap(),
            canonical_payload,
            transport_digests: vec![ContentDigest::sha256(b"transport")],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(
                b"mathematical-configuration",
            ),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "canonical value".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let attestation = |actor: &str, revision: &str, time| AttestationEnvelope {
            schema_version: 1,
            kind: AttestationKind::Validation,
            subject_digest: manifest_digest.clone(),
            actor: actor.to_owned(),
            policy_digest: ContentDigest::sha256(b"validation-policy"),
            execution_fingerprint_digest: ContentDigest::sha256(actor.as_bytes()),
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            dependency_versions: BTreeMap::from([("rug".to_owned(), "1.27.0".to_owned())]),
            source_revision: revision.to_owned(),
            event_unix_seconds: time,
            location: Some(format!("runner/{actor}")),
            evidence_digests: vec![ContentDigest::sha256(revision.as_bytes())],
        };
        let first = attestation("runner-a", "revision-a", 10);
        let second = attestation("runner-b", "revision-b", 20);
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
        assert_eq!(manifest.digest().unwrap(), manifest_digest);
        assert_eq!(first.subject_digest, second.subject_digest);
    }

    #[test]
    fn reports_and_certificates_are_independent_linked_artifacts() {
        let subject_semantic_digest = ContentDigest::sha256(b"numerical-semantic");
        let subject_payload_digest = ContentDigest::sha256(b"numerical-payload");
        let link = |role, preservation_class, seed: &[u8]| LinkedArtifactReference {
            role,
            preservation_class,
            artifact_family: "research_evidence".to_owned(),
            semantic_digest: ContentDigest::sha256(&[seed, b"semantic"].concat()),
            manifest_digest: ContentDigest::sha256(&[seed, b"manifest"].concat()),
            payload_digest: ContentDigest::sha256(&[seed, b"payload"].concat()),
        };
        let mut links = vec![
            link(
                LinkedArtifactRole::Certificate,
                ArtifactPreservationClass::GeneratedCertificate,
                b"certificate",
            ),
            link(
                LinkedArtifactRole::ReferenceDataset,
                ArtifactPreservationClass::ReferenceDataset,
                b"reference",
            ),
            link(
                LinkedArtifactRole::ValidationReport,
                ArtifactPreservationClass::ExploratoryOutput,
                b"validation-a",
            ),
            link(
                LinkedArtifactRole::ValidationReport,
                ArtifactPreservationClass::FrozenPublication,
                b"validation-b",
            ),
            link(
                LinkedArtifactRole::PublicationReadyExport,
                ArtifactPreservationClass::FrozenPublication,
                b"publication",
            ),
        ];
        links.sort_by(|left, right| {
            (
                left.role,
                left.preservation_class,
                &left.semantic_digest,
                &left.manifest_digest,
                &left.payload_digest,
            )
                .cmp(&(
                    right.role,
                    right.preservation_class,
                    &right.semantic_digest,
                    &right.manifest_digest,
                    &right.payload_digest,
                ))
        });
        let set = ArtifactLinkSet {
            schema_version: 1,
            subject_semantic_digest: subject_semantic_digest.clone(),
            subject_payload_digest: subject_payload_digest.clone(),
            links,
        };
        assert!(set.digest().unwrap().validate());
        assert_eq!(set.links.len(), 5);
        assert_eq!(set.subject_semantic_digest, subject_semantic_digest);
        assert_eq!(set.subject_payload_digest, subject_payload_digest);

        let mut relabelled = set;
        relabelled.links[0].preservation_class = ArtifactPreservationClass::ExploratoryOutput;
        assert!(relabelled.digest().is_err());
    }

    #[test]
    fn locations_and_publication_never_change_assurance() {
        let mut state = ArtifactState {
            completion: ArtifactCompletionState::Complete,
            achieved_assurance: ArtifactAssuranceState::Computed,
            ..ArtifactState::default()
        };
        state
            .record_location(ArtifactLocation {
                kind: ArtifactLocationKind::Workstation,
                locator: "objects/abc".to_owned(),
                verified: true,
            })
            .unwrap();
        state
            .transition_publication(
                PublicationDestination::Private,
                PublicationTargetState::Planned,
            )
            .unwrap();
        state
            .transition_publication(
                PublicationDestination::Private,
                PublicationTargetState::Uploading,
            )
            .unwrap();
        assert_eq!(state.achieved_assurance, ArtifactAssuranceState::Computed);
    }

    #[test]
    fn publication_targets_advance_independently() {
        let mut state = ArtifactState {
            completion: ArtifactCompletionState::Complete,
            achieved_assurance: ArtifactAssuranceState::CrossChecked,
            ..ArtifactState::default()
        };
        for destination in [
            PublicationDestination::Private,
            PublicationDestination::Public,
        ] {
            state
                .transition_publication(destination, PublicationTargetState::Planned)
                .unwrap();
        }
        state
            .transition_publication(
                PublicationDestination::Private,
                PublicationTargetState::Uploading,
            )
            .unwrap();
        state
            .transition_publication(
                PublicationDestination::Public,
                PublicationTargetState::Failed,
            )
            .unwrap();
        assert_eq!(
            state.publication[&PublicationDestination::Private],
            PublicationTargetState::Uploading
        );
        assert_eq!(
            state.publication[&PublicationDestination::Public],
            PublicationTargetState::Failed
        );
    }

    #[test]
    fn certified_promotion_requires_complete_artifact_and_evidence() {
        let mut state = ArtifactState::default();
        assert!(state
            .promote_assurance(
                ArtifactAssuranceState::Certified,
                ContentDigest::sha256(b"certificate")
            )
            .is_err());
        state.completion = ArtifactCompletionState::Complete;
        state
            .promote_assurance(
                ArtifactAssuranceState::Certified,
                ContentDigest::sha256(b"certificate"),
            )
            .unwrap();
    }
}
