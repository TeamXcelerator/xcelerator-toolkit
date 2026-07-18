//! Scholarly release-archive planning and DOI receipt verification.
//!
//! The toolkit validates the complete immutable inventory before an external
//! archive deposit. A DOI is accepted only through a receipt that covers the
//! exact manifest; this module never invents or predicts an external DOI.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveAuthor {
    pub given_names: String,
    pub family_names: String,
    pub name_suffix: Option<String>,
    pub orcid: String,
}

impl ArchiveAuthor {
    pub fn validate(&self) -> Result<(), ScholarlyArchiveError> {
        if self.given_names.trim().is_empty()
            || self.family_names.trim().is_empty()
            || !valid_orcid(&self.orcid)
        {
            return Err(ScholarlyArchiveError::InvalidMetadata(
                "archive authors require names and a canonical ORCID URL".to_owned(),
            ));
        }
        Ok(())
    }

    fn zenodo_name(&self) -> String {
        let suffix = self
            .name_suffix
            .as_ref()
            .filter(|suffix| !suffix.trim().is_empty())
            .map_or_else(String::new, |suffix| format!(", {suffix}"));
        format!("{}, {}{suffix}", self.family_names, self.given_names)
    }
}

/// Citation metadata embedded in an immutable generated research artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactCitationMetadata {
    pub schema_version: u32,
    pub artifact_title: String,
    pub artifact_type: String,
    pub authors: Vec<ArchiveAuthor>,
    pub software_title: String,
    pub software_version: String,
    pub repository: String,
    pub preferred_citation: String,
    pub software_doi: Option<String>,
    pub artifact_doi: Option<String>,
}

impl ArtifactCitationMetadata {
    pub fn validate(&self) -> Result<(), ScholarlyArchiveError> {
        if self.schema_version != 1
            || self.artifact_title.trim().is_empty()
            || self.artifact_type.trim().is_empty()
            || self.authors.is_empty()
            || self.software_title.trim().is_empty()
            || self.software_version.trim().is_empty()
            || !self.repository.starts_with("https://")
            || self.preferred_citation.trim().is_empty()
            || self
                .software_doi
                .as_deref()
                .is_some_and(|doi| !valid_doi(doi))
            || self
                .artifact_doi
                .as_deref()
                .is_some_and(|doi| !valid_doi(doi))
        {
            return Err(ScholarlyArchiveError::InvalidMetadata(
                "artifact citation metadata is incomplete or malformed".to_owned(),
            ));
        }
        for author in &self.authors {
            author.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScholarlyReleaseMetadata {
    pub title: String,
    pub version: String,
    pub release_date: String,
    pub abstract_text: String,
    pub license_identifier: String,
    pub repository: String,
    pub authors: Vec<ArchiveAuthor>,
    pub keywords: Vec<String>,
    pub preferred_citation: String,
}

impl ScholarlyReleaseMetadata {
    pub fn validate(&self) -> Result<(), ScholarlyArchiveError> {
        if self.title.trim().is_empty()
            || self.version.trim().is_empty()
            || !valid_date(&self.release_date)
            || self.abstract_text.trim().is_empty()
            || self.license_identifier.trim().is_empty()
            || !self.repository.starts_with("https://")
            || self.preferred_citation.trim().is_empty()
            || self.authors.is_empty()
            || self.keywords.is_empty()
            || self
                .keywords
                .iter()
                .any(|keyword| keyword.trim().is_empty())
        {
            return Err(ScholarlyArchiveError::InvalidMetadata(
                "release metadata is incomplete or malformed".to_owned(),
            ));
        }
        for author in &self.authors {
            author.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScholarlyArchiveArtifactRole {
    TaggedSource,
    DependencyLock,
    Requirements,
    TechnicalDesign,
    ReleaseNotes,
    Traceability,
    CitationMetadata,
    License,
    ReproducibilityManifest,
    TrustSnapshot,
    CertificateBundle,
    EssentialReferenceArtifact,
}

impl ScholarlyArchiveArtifactRole {
    const REQUIRED: [Self; 12] = [
        Self::TaggedSource,
        Self::DependencyLock,
        Self::Requirements,
        Self::TechnicalDesign,
        Self::ReleaseNotes,
        Self::Traceability,
        Self::CitationMetadata,
        Self::License,
        Self::ReproducibilityManifest,
        Self::TrustSnapshot,
        Self::CertificateBundle,
        Self::EssentialReferenceArtifact,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScholarlyArchiveArtifact {
    pub path: String,
    pub role: ScholarlyArchiveArtifactRole,
    pub media_type: String,
    pub sha256: String,
    pub byte_length: u64,
}

impl ScholarlyArchiveArtifact {
    fn validate(&self) -> Result<(), ScholarlyArchiveError> {
        if !normalized_relative_path(&self.path)
            || self.media_type.trim().is_empty()
            || !valid_digest(&self.sha256)
            || self.byte_length == 0
        {
            return Err(ScholarlyArchiveError::InvalidManifest(format!(
                "invalid scholarly archive artifact {:?}",
                self.path
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScholarlyArchiveManifest {
    pub schema_version: u32,
    pub release: ScholarlyReleaseMetadata,
    pub tag: String,
    pub source_revision: String,
    pub dependency_lock_digest: String,
    pub requirements_digest: String,
    pub technical_design_digest: String,
    pub trust_snapshot_digest: String,
    pub artifacts: Vec<ScholarlyArchiveArtifact>,
    pub created_at_utc: String,
    pub finite_claim_statement: String,
}

impl ScholarlyArchiveManifest {
    pub fn validate(&self) -> Result<(), ScholarlyArchiveError> {
        self.release.validate()?;
        if self.schema_version != 1
            || self.tag != format!("v{}", self.release.version)
            || !valid_revision(&self.source_revision)
            || !valid_digest(&self.dependency_lock_digest)
            || !valid_digest(&self.requirements_digest)
            || !valid_digest(&self.technical_design_digest)
            || !valid_digest(&self.trust_snapshot_digest)
            || !valid_timestamp(&self.created_at_utc)
            || self.finite_claim_statement.trim().is_empty()
        {
            return Err(ScholarlyArchiveError::InvalidManifest(
                "archive manifest identity, provenance, or claim scope is invalid".to_owned(),
            ));
        }
        let mut paths = BTreeSet::new();
        let mut roles = BTreeSet::new();
        let mut previous_path = None;
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !paths.insert(artifact.path.as_str()) {
                return Err(ScholarlyArchiveError::InvalidManifest(
                    "archive artifact paths must be unique".to_owned(),
                ));
            }
            roles.insert(artifact.role);
            if previous_path.is_some_and(|previous| previous >= artifact.path.as_str()) {
                return Err(ScholarlyArchiveError::InvalidManifest(
                    "archive artifacts must be ordered strictly by path".to_owned(),
                ));
            }
            previous_path = Some(artifact.path.as_str());
        }
        if ScholarlyArchiveArtifactRole::REQUIRED
            .iter()
            .any(|role| !roles.contains(role))
        {
            return Err(ScholarlyArchiveError::InvalidManifest(
                "archive manifest lacks a required scholarly-release artifact role".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ScholarlyArchiveError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| ScholarlyArchiveError::Serialization(error.to_string()))?;
        Ok(sha256(&bytes))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZenodoCreator {
    pub name: String,
    pub orcid: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZenodoDepositMetadata {
    pub title: String,
    pub upload_type: String,
    pub publication_date: String,
    pub description: String,
    pub creators: Vec<ZenodoCreator>,
    pub version: String,
    pub license: String,
    pub keywords: Vec<String>,
    pub related_identifiers: Vec<String>,
    pub notes: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScholarlyArchivePlan {
    pub schema_version: u32,
    pub manifest: ScholarlyArchiveManifest,
    pub manifest_digest: String,
    pub zenodo_metadata: ZenodoDepositMetadata,
    pub external_deposit_required: bool,
}

pub fn build_scholarly_archive_plan(
    manifest: ScholarlyArchiveManifest,
) -> Result<ScholarlyArchivePlan, ScholarlyArchiveError> {
    let manifest_digest = manifest.digest()?;
    let release = &manifest.release;
    let zenodo_metadata = ZenodoDepositMetadata {
        title: release.title.clone(),
        upload_type: "software".to_owned(),
        publication_date: release.release_date.clone(),
        description: release.abstract_text.clone(),
        creators: release
            .authors
            .iter()
            .map(|author| ZenodoCreator {
                name: author.zenodo_name(),
                orcid: author.orcid.clone(),
            })
            .collect(),
        version: release.version.clone(),
        license: release.license_identifier.clone(),
        keywords: release.keywords.clone(),
        related_identifiers: vec![release.repository.clone()],
        notes: release.preferred_citation.clone(),
    };
    Ok(ScholarlyArchivePlan {
        schema_version: 1,
        manifest,
        manifest_digest,
        zenodo_metadata,
        external_deposit_required: true,
    })
}

pub fn verify_scholarly_archive_inventory<'a, I>(
    manifest: &ScholarlyArchiveManifest,
    files: I,
) -> Result<(), ScholarlyArchiveError>
where
    I: IntoIterator<Item = (&'a str, &'a [u8])>,
{
    manifest.validate()?;
    let mut supplied = BTreeMap::new();
    for (path, bytes) in files {
        if supplied.insert(path, bytes).is_some() {
            return Err(ScholarlyArchiveError::InventoryMismatch(format!(
                "duplicate supplied archive path {path:?}"
            )));
        }
    }
    if supplied.len() != manifest.artifacts.len() {
        return Err(ScholarlyArchiveError::InventoryMismatch(
            "supplied archive inventory cardinality differs from the manifest".to_owned(),
        ));
    }
    for artifact in &manifest.artifacts {
        let bytes = supplied.get(artifact.path.as_str()).ok_or_else(|| {
            ScholarlyArchiveError::InventoryMismatch(format!(
                "archive inventory lacks {:?}",
                artifact.path
            ))
        })?;
        if bytes.len() as u64 != artifact.byte_length || sha256(bytes) != artifact.sha256 {
            return Err(ScholarlyArchiveError::InventoryMismatch(format!(
                "archive inventory bytes do not match {:?}",
                artifact.path
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveDepositObject {
    pub path: String,
    pub sha256: String,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveDepositReceipt {
    pub schema_version: u32,
    pub manifest_digest: String,
    pub tag: String,
    pub archive_provider: String,
    pub record_id: String,
    pub provider_record_url: String,
    pub provider_evidence_sha256: String,
    pub doi: String,
    pub deposited_at_utc: String,
    pub verified_at_utc: String,
    pub immutable: bool,
    pub objects: Vec<ArchiveDepositObject>,
}

pub fn verify_archive_deposit_receipt(
    plan: &ScholarlyArchivePlan,
    receipt: &ArchiveDepositReceipt,
) -> Result<(), ScholarlyArchiveError> {
    if plan.schema_version != 1
        || plan.manifest_digest != plan.manifest.digest()?
        || !plan.external_deposit_required
        || receipt.schema_version != 1
        || receipt.manifest_digest != plan.manifest_digest
        || receipt.tag != plan.manifest.tag
        || receipt.archive_provider.trim().is_empty()
        || receipt.record_id.trim().is_empty()
        || !receipt.provider_record_url.starts_with("https://")
        || !valid_digest(&receipt.provider_evidence_sha256)
        || !valid_doi(&receipt.doi)
        || !valid_timestamp(&receipt.deposited_at_utc)
        || !valid_timestamp(&receipt.verified_at_utc)
        || receipt.deposited_at_utc < plan.manifest.created_at_utc
        || receipt.verified_at_utc < receipt.deposited_at_utc
        || !receipt.immutable
        || receipt.objects.len() != plan.manifest.artifacts.len()
    {
        return Err(ScholarlyArchiveError::InvalidReceipt(
            "archive deposit receipt identity or external DOI evidence is invalid".to_owned(),
        ));
    }
    for (artifact, object) in plan.manifest.artifacts.iter().zip(&receipt.objects) {
        if artifact.path != object.path
            || artifact.sha256 != object.sha256
            || artifact.byte_length != object.byte_length
        {
            return Err(ScholarlyArchiveError::InvalidReceipt(
                "archive deposit receipt does not cover the exact ordered manifest".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Verify the complete post-deposit evidence without trusting a provider label.
///
/// The caller supplies the revision resolved from the local immutable tag, the
/// raw provider response bytes, and the exact local deposit inventory. This
/// function binds those independent observations to the typed plan and receipt.
pub fn verify_archive_deposit_evidence<'a, I>(
    plan: &ScholarlyArchivePlan,
    receipt: &ArchiveDepositReceipt,
    resolved_tag_revision: &str,
    provider_evidence: &[u8],
    files: I,
) -> Result<(), ScholarlyArchiveError>
where
    I: IntoIterator<Item = (&'a str, &'a [u8])>,
{
    if resolved_tag_revision != plan.manifest.source_revision {
        return Err(ScholarlyArchiveError::InvalidReceipt(
            "release tag does not resolve to the archived source revision".to_owned(),
        ));
    }
    if sha256(provider_evidence) != receipt.provider_evidence_sha256 {
        return Err(ScholarlyArchiveError::InvalidReceipt(
            "raw provider evidence does not match the receipt digest".to_owned(),
        ));
    }
    verify_scholarly_archive_inventory(&plan.manifest, files)?;
    verify_archive_deposit_receipt(plan, receipt)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScholarlyArchiveError {
    InvalidMetadata(String),
    InvalidManifest(String),
    InventoryMismatch(String),
    InvalidReceipt(String),
    Serialization(String),
}

impl Display for ScholarlyArchiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMetadata(message) => {
                write!(formatter, "invalid release metadata: {message}")
            }
            Self::InvalidManifest(message) => {
                write!(formatter, "invalid archive manifest: {message}")
            }
            Self::InventoryMismatch(message) => {
                write!(formatter, "archive inventory mismatch: {message}")
            }
            Self::InvalidReceipt(message) => {
                write!(formatter, "invalid archive receipt: {message}")
            }
            Self::Serialization(message) => {
                write!(formatter, "archive serialization failed: {message}")
            }
        }
    }
}

impl Error for ScholarlyArchiveError {}

fn normalized_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_date(value: &str) -> bool {
    value.len() == 10
        && value.as_bytes()[4] == b'-'
        && value.as_bytes()[7] == b'-'
        && value
            .bytes()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
}

fn valid_timestamp(value: &str) -> bool {
    value.len() >= 20
        && value.ends_with('Z')
        && valid_date(&value[..10])
        && value.as_bytes()[10] == b'T'
}

fn valid_orcid(value: &str) -> bool {
    let Some(identifier) = value.strip_prefix("https://orcid.org/") else {
        return false;
    };
    identifier.len() == 19
        && identifier.as_bytes()[4] == b'-'
        && identifier.as_bytes()[9] == b'-'
        && identifier.as_bytes()[14] == b'-'
        && identifier.bytes().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 9 | 14) || byte.is_ascii_digit() || (index == 18 && byte == b'X')
        })
}

fn valid_doi(value: &str) -> bool {
    value.starts_with("10.")
        && value
            .split_once('/')
            .is_some_and(|(registrant, suffix)| registrant.len() > 3 && !suffix.trim().is_empty())
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (ScholarlyArchiveManifest, Vec<(String, Vec<u8>)>) {
        let mut entries = [
            (
                "artifacts/certificates.json",
                ScholarlyArchiveArtifactRole::CertificateBundle,
            ),
            (
                "artifacts/reference.json",
                ScholarlyArchiveArtifactRole::EssentialReferenceArtifact,
            ),
            (
                "CITATION.cff",
                ScholarlyArchiveArtifactRole::CitationMetadata,
            ),
            ("Cargo.lock", ScholarlyArchiveArtifactRole::DependencyLock),
            (
                "docs/RELEASE_NOTES.md",
                ScholarlyArchiveArtifactRole::ReleaseNotes,
            ),
            (
                "docs/REQUIREMENTS.md",
                ScholarlyArchiveArtifactRole::Requirements,
            ),
            (
                "docs/TECHNICAL_DESIGN.md",
                ScholarlyArchiveArtifactRole::TechnicalDesign,
            ),
            (
                "docs/TRACEABILITY.json",
                ScholarlyArchiveArtifactRole::Traceability,
            ),
            ("LICENSE", ScholarlyArchiveArtifactRole::License),
            (
                "manifests/reproducibility.json",
                ScholarlyArchiveArtifactRole::ReproducibilityManifest,
            ),
            (
                "manifests/trust.json",
                ScholarlyArchiveArtifactRole::TrustSnapshot,
            ),
            (
                "source/xcelerator-toolkit-v0.13.0.tar.gz",
                ScholarlyArchiveArtifactRole::TaggedSource,
            ),
        ];
        entries.sort_by_key(|(path, _)| *path);
        let files = entries
            .iter()
            .map(|(path, role)| (path.to_string(), format!("{role:?}:{path}").into_bytes()))
            .collect::<Vec<_>>();
        let artifacts = entries
            .iter()
            .zip(&files)
            .map(|((path, role), (_, bytes))| ScholarlyArchiveArtifact {
                path: path.to_string(),
                role: *role,
                media_type: "application/octet-stream".to_owned(),
                sha256: sha256(bytes),
                byte_length: bytes.len() as u64,
            })
            .collect();
        let manifest = ScholarlyArchiveManifest {
            schema_version: 1,
            release: ScholarlyReleaseMetadata {
                title: "Xcelerator Toolkit".to_owned(),
                version: "0.13.0".to_owned(),
                release_date: "2026-07-16".to_owned(),
                abstract_text: "Certified and high-precision numerical research software.".to_owned(),
                license_identifier: "other".to_owned(),
                repository: "https://github.com/TeamXcelerator/xcelerator-toolkit".to_owned(),
                authors: vec![ArchiveAuthor {
                    given_names: "Ronnie".to_owned(),
                    family_names: "Andrews".to_owned(),
                    name_suffix: Some("Jr.".to_owned()),
                    orcid: "https://orcid.org/0009-0003-9724-3104".to_owned(),
                }],
                keywords: vec!["certified numerics".to_owned()],
                preferred_citation: "Andrews, Xcelerator Toolkit v0.13.0 (2026).".to_owned(),
            },
            tag: "v0.13.0".to_owned(),
            source_revision: "9".repeat(40),
            dependency_lock_digest: "a".repeat(64),
            requirements_digest: "b".repeat(64),
            technical_design_digest: "c".repeat(64),
            trust_snapshot_digest: "d".repeat(64),
            artifacts,
            created_at_utc: "2026-07-16T12:00:00Z".to_owned(),
            finite_claim_statement: "Finite archived results retain their recorded assurance and do not imply continuum claims."
                .to_owned(),
        };
        (manifest, files)
    }

    #[test]
    fn archive_plan_binds_inventory_and_generates_deposit_metadata() {
        let (manifest, files) = fixture();
        verify_scholarly_archive_inventory(
            &manifest,
            files
                .iter()
                .map(|(path, bytes)| (path.as_str(), bytes.as_slice())),
        )
        .unwrap();
        let plan = build_scholarly_archive_plan(manifest).unwrap();
        assert!(plan.external_deposit_required);
        assert_eq!(plan.zenodo_metadata.upload_type, "software");
        assert_eq!(
            plan.zenodo_metadata.creators[0].name,
            "Andrews, Ronnie, Jr."
        );

        let provider_evidence = b"immutable Zenodo API response";
        let receipt = ArchiveDepositReceipt {
            schema_version: 1,
            manifest_digest: plan.manifest_digest.clone(),
            tag: plan.manifest.tag.clone(),
            archive_provider: "Zenodo".to_owned(),
            record_id: "1234567".to_owned(),
            provider_record_url: "https://zenodo.org/records/1234567".to_owned(),
            provider_evidence_sha256: sha256(provider_evidence),
            doi: "10.5281/zenodo.1234567".to_owned(),
            deposited_at_utc: "2026-07-16T13:00:00Z".to_owned(),
            verified_at_utc: "2026-07-16T13:05:00Z".to_owned(),
            immutable: true,
            objects: plan
                .manifest
                .artifacts
                .iter()
                .map(|artifact| ArchiveDepositObject {
                    path: artifact.path.clone(),
                    sha256: artifact.sha256.clone(),
                    byte_length: artifact.byte_length,
                })
                .collect(),
        };
        verify_archive_deposit_receipt(&plan, &receipt).unwrap();
        verify_archive_deposit_evidence(
            &plan,
            &receipt,
            &plan.manifest.source_revision,
            provider_evidence,
            files
                .iter()
                .map(|(path, bytes)| (path.as_str(), bytes.as_slice())),
        )
        .unwrap();

        assert!(verify_archive_deposit_evidence(
            &plan,
            &receipt,
            &"8".repeat(40),
            provider_evidence,
            files
                .iter()
                .map(|(path, bytes)| (path.as_str(), bytes.as_slice())),
        )
        .is_err());
        assert!(verify_archive_deposit_evidence(
            &plan,
            &receipt,
            &plan.manifest.source_revision,
            b"forged provider response",
            files
                .iter()
                .map(|(path, bytes)| (path.as_str(), bytes.as_slice())),
        )
        .is_err());

        let mut tampered = receipt;
        tampered.objects[0].sha256 = "e".repeat(64);
        assert!(verify_archive_deposit_receipt(&plan, &tampered).is_err());

        let mut forged = build_scholarly_archive_plan(fixture().0).unwrap();
        forged.manifest.created_at_utc = "2026-07-16T14:00:00Z".to_owned();
        forged.manifest_digest = forged.manifest.digest().unwrap();
        let mut early_receipt = tampered;
        early_receipt.manifest_digest = forged.manifest_digest.clone();
        early_receipt.objects[0].sha256 = forged.manifest.artifacts[0].sha256.clone();
        assert!(verify_archive_deposit_receipt(&forged, &early_receipt).is_err());
    }

    #[test]
    fn archive_inventory_and_required_roles_fail_closed() {
        let (mut manifest, mut files) = fixture();
        files[0].1.push(0);
        assert!(verify_scholarly_archive_inventory(
            &manifest,
            files
                .iter()
                .map(|(path, bytes)| (path.as_str(), bytes.as_slice())),
        )
        .is_err());

        manifest.artifacts.remove(0);
        assert!(manifest.validate().is_err());
    }
}
