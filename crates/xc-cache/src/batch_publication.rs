//! Repository-level publication records for byte-batched Git pushes.

use crate::protocol::{canonical_digest, normalized_relative_path};
use crate::{
    ArtifactAssuranceState, CacheError, ContentDigest, PublicationDestination, PublicationReceipt,
    ToolkitVersion,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryBatchArtifact {
    pub semantic_digest: ContentDigest,
    pub canonical_payload_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub manifest_path: String,
    pub achieved_assurance: ArtifactAssuranceState,
    pub producer_toolkit_version: ToolkitVersion,
    pub provenance_evidence_digests: Vec<ContentDigest>,
}

impl RepositoryBatchArtifact {
    pub fn validate(&self) -> Result<(), CacheError> {
        if [
            &self.semantic_digest,
            &self.canonical_payload_digest,
            &self.manifest_digest,
            &self.transport_digest,
        ]
        .into_iter()
        .any(|digest| !digest.validate())
            || !normalized_relative_path(&self.manifest_path)
            || self
                .provenance_evidence_digests
                .iter()
                .any(|digest| !digest.validate())
            || self
                .provenance_evidence_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(CacheError::InvalidManifest(
                "repository batch contains an invalid artifact identity".to_owned(),
            ));
        }
        ToolkitVersion::parse(&self.producer_toolkit_version.to_string())?;
        Ok(())
    }
}

#[derive(Serialize)]
struct RepositoryBatchIdentity<'a> {
    schema_version: u32,
    destination: PublicationDestination,
    family: &'a str,
    authorized_repository: &'a str,
    branch: &'a str,
    policy_digest: &'a ContentDigest,
    artifacts: &'a [RepositoryBatchArtifact],
}

/// One immutable audit record for every logical artifact carried by the same
/// repository publication. It deliberately excludes PAT/session evidence and
/// Git commit IDs, allowing the record, indexes, ledger, manifests, and payload
/// files to be committed atomically in the same byte-batched push.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryPublicationBatch {
    pub schema_version: u32,
    pub batch_id: ContentDigest,
    pub destination: PublicationDestination,
    pub family: String,
    pub principal: String,
    pub authorized_repository: String,
    pub branch: String,
    pub policy_digest: ContentDigest,
    pub artifacts: Vec<RepositoryBatchArtifact>,
    pub created_unix_seconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemotePublicationEvidence {
    ArtifactReceipt(Box<PublicationReceipt>),
    RepositoryBatch(Box<RepositoryPublicationBatch>),
}

impl RemotePublicationEvidence {
    pub fn validate(&self) -> Result<(), CacheError> {
        match self {
            Self::ArtifactReceipt(receipt) => receipt.validate(),
            Self::RepositoryBatch(batch) => batch.validate(),
        }
    }

    pub fn verified_at_unix_seconds(&self) -> u64 {
        match self {
            Self::ArtifactReceipt(receipt) => receipt.verified_at_unix_seconds,
            Self::RepositoryBatch(batch) => batch.created_unix_seconds,
        }
    }

    pub fn artifact(
        &self,
        semantic_digest: &ContentDigest,
        manifest_digest: &ContentDigest,
    ) -> Option<RepositoryBatchArtifact> {
        match self {
            Self::ArtifactReceipt(receipt)
                if &receipt.semantic_digest == semantic_digest
                    && &receipt.manifest_digest == manifest_digest =>
            {
                Some(RepositoryBatchArtifact {
                    semantic_digest: receipt.semantic_digest.clone(),
                    canonical_payload_digest: receipt.canonical_payload_digest.clone(),
                    manifest_digest: receipt.manifest_digest.clone(),
                    transport_digest: receipt.transport_digest.clone(),
                    manifest_path: receipt
                        .metadata_file_digests
                        .keys()
                        .find(|path| path.starts_with("manifests/"))
                        .cloned()
                        .unwrap_or_default(),
                    achieved_assurance: ArtifactAssuranceState::Computed,
                    producer_toolkit_version: ToolkitVersion::parse("0.13.0")
                        .expect("current toolkit version is valid"),
                    provenance_evidence_digests: Vec::new(),
                })
            }
            Self::RepositoryBatch(batch) => batch
                .artifacts
                .iter()
                .find(|artifact| {
                    &artifact.semantic_digest == semantic_digest
                        && &artifact.manifest_digest == manifest_digest
                })
                .cloned(),
            _ => None,
        }
    }
}

impl RepositoryPublicationBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        destination: PublicationDestination,
        family: impl Into<String>,
        principal: impl Into<String>,
        authorized_repository: impl Into<String>,
        branch: impl Into<String>,
        policy_digest: ContentDigest,
        mut artifacts: Vec<RepositoryBatchArtifact>,
        created_unix_seconds: u64,
    ) -> Result<Self, CacheError> {
        artifacts.sort_by(|left, right| {
            left.semantic_digest
                .cmp(&right.semantic_digest)
                .then_with(|| left.manifest_digest.cmp(&right.manifest_digest))
        });
        let family = family.into();
        let authorized_repository = authorized_repository.into();
        let branch = branch.into();
        let batch_id = canonical_digest(&RepositoryBatchIdentity {
            schema_version: 1,
            destination,
            family: &family,
            authorized_repository: &authorized_repository,
            branch: &branch,
            policy_digest: &policy_digest,
            artifacts: &artifacts,
        })?;
        let batch = Self {
            schema_version: 1,
            batch_id,
            destination,
            family,
            principal: principal.into(),
            authorized_repository,
            branch,
            policy_digest,
            artifacts,
            created_unix_seconds,
        };
        batch.validate()?;
        Ok(batch)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.family.trim().is_empty()
            || self.principal.trim().is_empty()
            || self.authorized_repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || !self.policy_digest.validate()
            || self.artifacts.is_empty()
            || self.created_unix_seconds == 0
        {
            return Err(CacheError::InvalidManifest(
                "repository publication batch is incomplete".to_owned(),
            ));
        }
        for artifact in &self.artifacts {
            artifact.validate()?;
        }
        let identities = self
            .artifacts
            .iter()
            .map(|artifact| (&artifact.semantic_digest, &artifact.manifest_digest))
            .collect::<BTreeSet<_>>();
        if identities.len() != self.artifacts.len()
            || self.artifacts.windows(2).any(|pair| {
                (&pair[0].semantic_digest, &pair[0].manifest_digest)
                    >= (&pair[1].semantic_digest, &pair[1].manifest_digest)
            })
        {
            return Err(CacheError::InvalidManifest(
                "repository publication batch artifacts are duplicated or unordered".to_owned(),
            ));
        }
        let expected = canonical_digest(&RepositoryBatchIdentity {
            schema_version: self.schema_version,
            destination: self.destination,
            family: &self.family,
            authorized_repository: &self.authorized_repository,
            branch: &self.branch,
            policy_digest: &self.policy_digest,
            artifacts: &self.artifacts,
        })?;
        if self.batch_id != expected {
            return Err(CacheError::DigestMismatch {
                expected: expected.to_string(),
                actual: self.batch_id.to_string(),
            });
        }
        Ok(())
    }

    pub fn repository_path(&self) -> String {
        format!(
            "transactions/batches/{}/{}.json",
            self.batch_id,
            match self.destination {
                PublicationDestination::Private => "private",
                PublicationDestination::Public => "public",
            }
        )
    }

    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_identity_ignores_volatile_authorization_sessions() {
        let artifact = RepositoryBatchArtifact {
            semantic_digest: ContentDigest::sha256(b"semantic"),
            canonical_payload_digest: ContentDigest::sha256(b"payload"),
            manifest_digest: ContentDigest::sha256(b"manifest"),
            transport_digest: ContentDigest::sha256(b"transport"),
            manifest_path: "manifests/aa/manifest.json".to_owned(),
            achieved_assurance: ArtifactAssuranceState::Computed,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            provenance_evidence_digests: Vec::new(),
        };
        let first = RepositoryPublicationBatch::new(
            PublicationDestination::Private,
            "quadrature",
            "author-one",
            "owner/private-quadrature-0001",
            "main",
            ContentDigest::sha256(b"policy"),
            vec![artifact.clone()],
            10,
        )
        .unwrap();
        let later_session = RepositoryPublicationBatch::new(
            PublicationDestination::Private,
            "quadrature",
            "author-two",
            "owner/private-quadrature-0001",
            "main",
            ContentDigest::sha256(b"policy"),
            vec![artifact],
            20,
        )
        .unwrap();
        assert_eq!(first.batch_id, later_session.batch_id);
    }

    #[test]
    fn batch_proof_selects_the_requested_artifact() {
        let artifact = RepositoryBatchArtifact {
            semantic_digest: ContentDigest::sha256(b"semantic"),
            canonical_payload_digest: ContentDigest::sha256(b"payload"),
            manifest_digest: ContentDigest::sha256(b"manifest"),
            transport_digest: ContentDigest::sha256(b"transport"),
            manifest_path: "manifests/aa/manifest.json".to_owned(),
            achieved_assurance: ArtifactAssuranceState::Computed,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            provenance_evidence_digests: Vec::new(),
        };
        let batch = RepositoryPublicationBatch::new(
            PublicationDestination::Public,
            "ccm-matrices",
            "author",
            "owner/public-ccm-matrices-0001",
            "main",
            ContentDigest::sha256(b"policy"),
            vec![artifact.clone()],
            10,
        )
        .unwrap();
        let evidence = RemotePublicationEvidence::RepositoryBatch(Box::new(batch));
        assert_eq!(
            evidence
                .artifact(&artifact.semantic_digest, &artifact.manifest_digest)
                .unwrap(),
            artifact
        );
        assert!(evidence
            .artifact(&ContentDigest::sha256(b"other"), &artifact.manifest_digest)
            .is_none());
    }
}
