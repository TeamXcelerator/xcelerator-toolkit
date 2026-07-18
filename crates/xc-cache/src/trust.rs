//! Explicit remote fabric trust roots and branch-rewrite protection.

use crate::{canonical_digest, CacheError, ContentDigest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustedRepositoryRole {
    Registry,
    Shard,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TrustedRevisionPolicy {
    Exact {
        revision: String,
    },
    AttestedDescendant {
        statement: RevisionAncestryStatement,
        ancestry_attestation_digest: ContentDigest,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionAncestryStatement {
    pub schema_version: u32,
    pub repository: String,
    pub branch: String,
    pub trusted_ancestor: String,
    pub observed_descendant: String,
    pub trust_anchor_id: String,
}

impl RevisionAncestryStatement {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        if self.schema_version != 1
            || self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || !commit_id(&self.trusted_ancestor)
            || !commit_id(&self.observed_descendant)
            || self.trust_anchor_id.trim().is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "revision ancestry statement is incomplete".to_owned(),
            ));
        }
        canonical_digest(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedBranchStatement {
    pub schema_version: u32,
    pub repository: String,
    pub branch: String,
    pub observed_revision: String,
    pub force_pushes_prohibited: bool,
    pub pushes_restricted: bool,
    pub observed_at_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
    pub trust_anchor_id: String,
}

impl ProtectedBranchStatement {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.repository.trim().is_empty()
            || self.branch.trim().is_empty()
            || !commit_id(&self.observed_revision)
            || !self.force_pushes_prohibited
            || !self.pushes_restricted
            || self.observed_at_unix_seconds > self.valid_until_unix_seconds
            || self.trust_anchor_id.trim().is_empty()
        {
            return Err(CacheError::InvalidManifest(
                "protected branch statement is incomplete or permits an unsafe rewrite".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedRepositoryRoot {
    pub role: TrustedRepositoryRole,
    pub repository: String,
    pub owner: String,
    pub branch: String,
    pub revision_policy: TrustedRevisionPolicy,
    pub branch_protection: ProtectedBranchStatement,
    pub branch_protection_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteFabricTrustPolicy {
    pub schema_version: u32,
    pub approved_trust_anchor_ids: BTreeSet<String>,
    pub approved_policy_digests: BTreeSet<ContentDigest>,
    pub repositories: Vec<TrustedRepositoryRoot>,
}

impl RemoteFabricTrustPolicy {
    pub fn verify_repository(
        &self,
        role: TrustedRepositoryRole,
        repository: &str,
        branch: &str,
        revision: &str,
        evaluation_unix_seconds: u64,
    ) -> Result<(), CacheError> {
        self.validate()?;
        let root = self
            .repositories
            .iter()
            .find(|root| {
                root.role == role && root.repository == repository && root.branch == branch
            })
            .ok_or_else(|| {
                CacheError::PermissionDenied(format!(
                    "repository {repository}:{branch} is outside the explicit trust roots"
                ))
            })?;
        if repository_owner(repository) != Some(root.owner.as_str()) {
            return Err(CacheError::PermissionDenied(
                "trusted owner does not match the canonical repository identity".to_owned(),
            ));
        }
        match &root.revision_policy {
            TrustedRevisionPolicy::Exact { revision: trusted } if trusted == revision => {}
            TrustedRevisionPolicy::AttestedDescendant {
                statement,
                ancestry_attestation_digest,
            } if statement.repository == repository
                && statement.branch == branch
                && statement.observed_descendant == revision
                && statement.digest()? == *ancestry_attestation_digest
                && self
                    .approved_trust_anchor_ids
                    .contains(&statement.trust_anchor_id) => {}
            _ => {
                return Err(CacheError::PermissionDenied(
                    "remote revision does not satisfy its exact or attested-ancestry policy"
                        .to_owned(),
                ));
            }
        }
        let protection = &root.branch_protection;
        if protection.repository != repository
            || protection.branch != branch
            || protection.observed_revision != revision
            || evaluation_unix_seconds < protection.observed_at_unix_seconds
            || evaluation_unix_seconds > protection.valid_until_unix_seconds
            || !self
                .approved_trust_anchor_ids
                .contains(&protection.trust_anchor_id)
            || protection.digest()? != root.branch_protection_digest
        {
            return Err(CacheError::PermissionDenied(
                "remote head lacks current digest-bound branch-protection evidence".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn verify_policy_digest(&self, digest: &ContentDigest) -> Result<(), CacheError> {
        self.validate()?;
        if !self.approved_policy_digests.contains(digest) {
            return Err(CacheError::PermissionDenied(format!(
                "remote policy digest {} is outside the trust root",
                digest.0
            )));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.approved_trust_anchor_ids.is_empty()
            || self.approved_policy_digests.is_empty()
            || self.repositories.is_empty()
            || self
                .approved_trust_anchor_ids
                .iter()
                .any(|anchor| anchor.trim().is_empty())
            || self
                .approved_policy_digests
                .iter()
                .any(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "remote fabric trust policy is incomplete".to_owned(),
            ));
        }
        let mut identities = BTreeSet::new();
        for root in &self.repositories {
            root.branch_protection.validate()?;
            if root.repository.trim().is_empty()
                || root.owner.trim().is_empty()
                || root.branch.trim().is_empty()
                || !root.branch_protection_digest.validate()
                || !identities.insert((root.role, root.repository.as_str(), root.branch.as_str()))
            {
                return Err(CacheError::InvalidManifest(
                    "remote repository trust roots are invalid or duplicated".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

fn repository_owner(repository: &str) -> Option<&str> {
    let (owner, name) = repository.split_once('/')?;
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        None
    } else {
        Some(owner)
    }
}

fn commit_id(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> RemoteFabricTrustPolicy {
        let revision = "a".repeat(40);
        let protection = ProtectedBranchStatement {
            schema_version: 1,
            repository: "example-org/public-shard".to_owned(),
            branch: "main".to_owned(),
            observed_revision: revision.clone(),
            force_pushes_prohibited: true,
            pushes_restricted: true,
            observed_at_unix_seconds: 100,
            valid_until_unix_seconds: 200,
            trust_anchor_id: "release-key".to_owned(),
        };
        RemoteFabricTrustPolicy {
            schema_version: 1,
            approved_trust_anchor_ids: ["release-key".to_owned()].into_iter().collect(),
            approved_policy_digests: [ContentDigest::sha256(b"policy")].into_iter().collect(),
            repositories: vec![TrustedRepositoryRoot {
                role: TrustedRepositoryRole::Shard,
                repository: protection.repository.clone(),
                owner: "example-org".to_owned(),
                branch: protection.branch.clone(),
                revision_policy: TrustedRevisionPolicy::Exact { revision },
                branch_protection_digest: protection.digest().unwrap(),
                branch_protection: protection,
            }],
        }
    }

    #[test]
    fn exact_protected_head_is_accepted_and_rewrite_is_rejected() {
        let policy = policy();
        policy
            .verify_repository(
                TrustedRepositoryRole::Shard,
                "example-org/public-shard",
                "main",
                &"a".repeat(40),
                150,
            )
            .unwrap();
        assert!(policy
            .verify_repository(
                TrustedRepositoryRole::Shard,
                "example-org/public-shard",
                "main",
                &"b".repeat(40),
                150,
            )
            .is_err());
        assert!(policy
            .verify_repository(
                TrustedRepositoryRole::Shard,
                "Attacker/cache-public",
                "main",
                &"a".repeat(40),
                150,
            )
            .is_err());
    }

    #[test]
    fn expired_protection_and_unapproved_policy_fail_closed() {
        let policy = policy();
        assert!(policy
            .verify_repository(
                TrustedRepositoryRole::Shard,
                "example-org/public-shard",
                "main",
                &"a".repeat(40),
                201,
            )
            .is_err());
        assert!(policy
            .verify_policy_digest(&ContentDigest::sha256(b"other-policy"))
            .is_err());
    }
}
