//! Evidence-driven validation/promotion and owner-direct publication policy.

use crate::{
    canonical_digest, ArtifactAssuranceState, ArtifactCompletionState, ArtifactDisposition,
    AuthenticatedGitHubSession, CacheError, ContentDigest, PublicationDestination,
    RepositoryPermissionEvidence,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use xc_core::{PublicationAuthority, PublicationAuthorityMode, PublicationTarget};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidatorEvidence {
    pub validator_id: String,
    pub passed: bool,
    pub evidence_digest: ContentDigest,
    pub establishes_assurance: Option<ArtifactAssuranceState>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationReviewEvidence {
    pub reviewer_principal: String,
    pub approved: bool,
    pub pull_request_number: u64,
    pub reviewed_head_revision: String,
    pub evidence_digest: ContentDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContributorAuthorizationKind {
    WrittenAuthorization,
    ContributorAgreement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributorPublicationEvidence {
    pub contributor_principal: String,
    pub authorizing_principal: String,
    pub authorization_kind: ContributorAuthorizationKind,
    pub authorized_destinations: BTreeSet<PublicationDestination>,
    pub authorized_repositories: BTreeSet<String>,
    pub authorization_evidence_digest: ContentDigest,
    pub source_repository: String,
    pub source_branch: String,
    pub contribution_head_revision: String,
    pub pull_request_number: u64,
    pub pull_request_evidence_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicSanitizerProfile {
    /// Exact dotted leaf paths admitted to a public manifest.
    pub allowed_leaf_fields: BTreeSet<String>,
    pub allowed_source_identifiers: BTreeSet<String>,
    pub allowed_repository_names: BTreeSet<String>,
    pub prohibited_key_fragments: BTreeSet<String>,
    pub prohibited_value_fragments: BTreeSet<String>,
}

impl Default for PublicSanitizerProfile {
    fn default() -> Self {
        Self {
            allowed_leaf_fields: BTreeSet::new(),
            allowed_source_identifiers: BTreeSet::new(),
            allowed_repository_names: BTreeSet::new(),
            prohibited_key_fragments: [
                "authorization",
                "password",
                "private_key",
                "secret",
                "token",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            prohibited_value_fragments: [
                "begin private key",
                "github_pat_",
                "ghp_",
                "gho_",
                "ghs_",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PublicSanitizationReport {
    pub accepted: bool,
    pub inspected_leaf_count: usize,
    pub reasons: Vec<String>,
}

impl PublicSanitizerProfile {
    pub fn inspect(
        &self,
        metadata: &BTreeMap<String, Value>,
        repository: &str,
    ) -> PublicSanitizationReport {
        let mut leaves = Vec::new();
        for (key, value) in metadata {
            collect_leaves(key, value, &mut leaves);
        }
        let mut reasons = Vec::new();
        if !self.allowed_repository_names.contains(repository) {
            reasons.push(format!(
                "public repository {repository:?} is not on the sanitizer allowlist"
            ));
        }
        for (path, value) in &leaves {
            let lower_path = path.to_ascii_lowercase();
            if !self.allowed_leaf_fields.contains(path) {
                reasons.push(format!("public field {path:?} is not allowlisted"));
            }
            if self
                .prohibited_key_fragments
                .iter()
                .any(|fragment| lower_path.contains(fragment))
            {
                reasons.push(format!("public field name {path:?} is sensitive"));
            }
            if let Value::String(text) = value {
                let lower = text.to_ascii_lowercase();
                if self
                    .prohibited_value_fragments
                    .iter()
                    .any(|fragment| lower.contains(fragment))
                {
                    reasons.push(format!("public field {path:?} contains a secret marker"));
                }
                if looks_like_local_path(text) {
                    reasons.push(format!(
                        "public field {path:?} contains a local filesystem path"
                    ));
                }
                if path.ends_with("source_identifier")
                    && !self.allowed_source_identifiers.contains(text)
                {
                    reasons.push(format!(
                        "source identifier in {path:?} is not publicly allowlisted"
                    ));
                }
                if path.ends_with("repository") && !self.allowed_repository_names.contains(text) {
                    reasons.push(format!(
                        "repository name in {path:?} is not publicly allowlisted"
                    ));
                }
            }
        }
        reasons.sort();
        reasons.dedup();
        PublicSanitizationReport {
            accepted: reasons.is_empty(),
            inspected_leaf_count: leaves.len(),
            reasons,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePublicationPolicy {
    pub policy_id: String,
    pub owner_principals: BTreeSet<String>,
    pub allow_owner_direct: bool,
    pub minimum_assurance: BTreeMap<PublicationDestination, ArtifactAssuranceState>,
    pub required_validators: BTreeMap<PublicationDestination, BTreeSet<String>>,
    pub minimum_unique_contributor_reviews: usize,
    pub sanitizer: PublicSanitizerProfile,
}

impl CachePublicationPolicy {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        if self.policy_id.trim().is_empty()
            || self
                .owner_principals
                .iter()
                .any(|owner| owner.trim().is_empty())
            || self
                .required_validators
                .values()
                .flatten()
                .any(|validator| validator.trim().is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "cache publication policy contains an empty identity".to_owned(),
            ));
        }
        canonical_digest(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachePublicationCandidate {
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub payload_digest: ContentDigest,
    pub completion: ArtifactCompletionState,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub validator_evidence: Vec<ValidatorEvidence>,
    pub public_metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPublicationAuthorizationRequest {
    pub destination: PublicationDestination,
    pub repository: String,
    pub authority: PublicationAuthority,
    pub contributor: Option<ContributorPublicationEvidence>,
    pub reviews: Vec<PublicationReviewEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PublicationAuthorizationDecision {
    pub authorized: bool,
    pub owner_direct: bool,
    pub requires_pull_request: bool,
    pub policy_digest: ContentDigest,
    pub contributor_authorization_digest: Option<ContentDigest>,
    pub completed_validators: Vec<String>,
    pub reasons: Vec<String>,
    pub sanitizer: Option<PublicSanitizationReport>,
    pub repository_permission: RepositoryPermissionEvidence,
}

pub fn authorize_target_publication(
    policy: &CachePublicationPolicy,
    candidate: &CachePublicationCandidate,
    request: &TargetPublicationAuthorizationRequest,
    authenticated_session: &AuthenticatedGitHubSession,
) -> Result<PublicationAuthorizationDecision, CacheError> {
    let policy_digest = policy.digest()?;
    let mut reasons = Vec::new();
    if let Err(error) =
        authenticated_session.require_write_for(&request.authority.principal, &request.repository)
    {
        reasons.push(error.to_string());
    }
    for digest in [
        &candidate.semantic_digest,
        &candidate.manifest_digest,
        &candidate.payload_digest,
    ] {
        if !digest.validate() {
            reasons.push("publication candidate contains an invalid identity digest".to_owned());
        }
    }
    if candidate.completion != ArtifactCompletionState::Complete {
        reasons.push("only complete artifacts may be published".to_owned());
    }
    if candidate.disposition != ArtifactDisposition::Active {
        reasons.push(format!(
            "artifact disposition {:?} is not publishable",
            candidate.disposition
        ));
    }
    if policy
        .minimum_assurance
        .get(&request.destination)
        .is_some_and(|minimum| candidate.achieved_assurance < *minimum)
    {
        reasons.push(format!(
            "achieved assurance {:?} is below the {:?} target minimum",
            candidate.achieved_assurance, request.destination
        ));
    }

    let requested_target = match request.destination {
        PublicationDestination::Private => PublicationTarget::Private,
        PublicationDestination::Public => PublicationTarget::Public,
    };
    if request.authority.principal.trim().is_empty()
        || request.authority.policy_digest != policy_digest.0
    {
        reasons.push("publication authority does not identify this resolved policy".to_owned());
    }
    if !request
        .authority
        .allowed_targets
        .contains(&requested_target)
        && !request
            .authority
            .allowed_targets
            .contains(&PublicationTarget::Both)
    {
        reasons.push("publication authority does not allow this target".to_owned());
    }
    if !request
        .authority
        .allowed_repositories
        .contains(&request.repository)
    {
        reasons.push("publication authority does not allow this repository".to_owned());
    }

    let completed_validators: BTreeSet<_> = candidate
        .validator_evidence
        .iter()
        .filter(|evidence| evidence.passed && evidence.evidence_digest.validate())
        .map(|evidence| evidence.validator_id.clone())
        .collect();
    for required in policy
        .required_validators
        .get(&request.destination)
        .into_iter()
        .flatten()
    {
        if !completed_validators.contains(required) {
            reasons.push(format!("required validator {required:?} has not passed"));
        }
    }

    // Assurance is evidence-derived: an owner identity can remove a manual
    // review step, but it can never replace missing mathematical evidence.
    let evidence_assurance = candidate
        .validator_evidence
        .iter()
        .filter(|evidence| evidence.passed && evidence.evidence_digest.validate())
        .filter_map(|evidence| evidence.establishes_assurance)
        .max()
        .unwrap_or(ArtifactAssuranceState::Unchecked);
    if candidate.achieved_assurance > evidence_assurance {
        reasons.push(format!(
            "achieved assurance {:?} exceeds validator evidence {:?}",
            candidate.achieved_assurance, evidence_assurance
        ));
    }

    let owner_direct = request.authority.mode == PublicationAuthorityMode::OwnerDirect
        && policy.allow_owner_direct
        && policy
            .owner_principals
            .contains(&request.authority.principal);
    let requires_pull_request = !owner_direct;
    if request.authority.mode == PublicationAuthorityMode::OwnerDirect && !owner_direct {
        reasons.push("principal is not authorized for owner-direct publication".to_owned());
    }
    let mut contributor_authorization_digest = None;
    if requires_pull_request {
        let contributor = request.contributor.as_ref();
        if let Some(contributor) = contributor {
            contributor_authorization_digest =
                Some(contributor.authorization_evidence_digest.clone());
            if contributor.contributor_principal.trim().is_empty()
                || !policy
                    .owner_principals
                    .contains(&contributor.authorizing_principal)
                || !contributor
                    .authorized_destinations
                    .contains(&request.destination)
                || !contributor
                    .authorized_repositories
                    .contains(&request.repository)
                || !contributor.authorization_evidence_digest.validate()
            {
                reasons.push(
                    "contributor lacks explicit written authorization or an approved contributor agreement for this target"
                        .to_owned(),
                );
            }
            let valid_source = contributor.source_repository.contains('/')
                && !contributor
                    .source_repository
                    .chars()
                    .any(char::is_whitespace)
                && !contributor.source_branch.trim().is_empty()
                && !contributor.source_branch.chars().any(char::is_whitespace)
                && (contributor.source_repository != request.repository
                    || contributor.source_branch != "main");
            if !valid_source
                || contributor.pull_request_number == 0
                || !valid_git_revision(&contributor.contribution_head_revision)
                || !contributor.pull_request_evidence_digest.validate()
            {
                reasons.push(
                    "contributor publication requires a valid fork or contribution branch and digest-bound pull request"
                        .to_owned(),
                );
            }
        } else {
            reasons.push(
                "contributor publication requires explicit authorization and pull-request evidence"
                    .to_owned(),
            );
        }
        let approvals: BTreeSet<_> = request
            .reviews
            .iter()
            .filter(|review| {
                review.approved
                    && !review.reviewer_principal.trim().is_empty()
                    && review.evidence_digest.validate()
                    && contributor.is_some_and(|contributor| {
                        review.reviewer_principal != contributor.contributor_principal
                            && review.pull_request_number == contributor.pull_request_number
                            && review.reviewed_head_revision
                                == contributor.contribution_head_revision
                    })
            })
            .map(|review| review.reviewer_principal.as_str())
            .collect();
        if approvals.len() < policy.minimum_unique_contributor_reviews {
            reasons.push(format!(
                "contributor publication has {} valid unique reviews, requires {}",
                approvals.len(),
                policy.minimum_unique_contributor_reviews
            ));
        }
    }

    let sanitizer = (request.destination == PublicationDestination::Public).then(|| {
        policy
            .sanitizer
            .inspect(&candidate.public_metadata, &request.repository)
    });
    if let Some(sanitizer) = &sanitizer {
        reasons.extend(sanitizer.reasons.iter().cloned());
    }
    reasons.sort();
    reasons.dedup();
    Ok(PublicationAuthorizationDecision {
        authorized: reasons.is_empty(),
        owner_direct,
        requires_pull_request,
        policy_digest,
        contributor_authorization_digest,
        completed_validators: completed_validators.into_iter().collect(),
        reasons,
        sanitizer,
        repository_permission: authenticated_session.evidence().clone(),
    })
}

fn valid_git_revision(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn collect_leaves<'a>(prefix: &str, value: &'a Value, output: &mut Vec<(String, &'a Value)>) {
    match value {
        Value::Object(values) => {
            for (key, value) in values {
                collect_leaves(&format!("{prefix}.{key}"), value, output);
            }
        }
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                collect_leaves(&format!("{prefix}[{index}]"), value, output);
            }
        }
        _ => output.push((prefix.to_owned(), value)),
    }
}

fn looks_like_local_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    (bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/'))
        || value.starts_with("/home/")
        || value.starts_with("/Users/")
        || value.starts_with("\\\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CachePublicationPolicy {
        let sanitizer = PublicSanitizerProfile {
            allowed_leaf_fields: [
                "artifact.kind".to_owned(),
                "artifact.source_identifier".to_owned(),
                "artifact.repository".to_owned(),
            ]
            .into_iter()
            .collect(),
            allowed_source_identifiers: ["public-fixture-v1".to_owned()].into_iter().collect(),
            allowed_repository_names: ["example-org/public-shard".to_owned()]
                .into_iter()
                .collect(),
            ..PublicSanitizerProfile::default()
        };
        CachePublicationPolicy {
            policy_id: "cache-publication-v1".to_owned(),
            owner_principals: ["test-owner".to_owned()].into_iter().collect(),
            allow_owner_direct: true,
            minimum_assurance: BTreeMap::from([
                (
                    PublicationDestination::Private,
                    ArtifactAssuranceState::Computed,
                ),
                (
                    PublicationDestination::Public,
                    ArtifactAssuranceState::CrossChecked,
                ),
            ]),
            required_validators: BTreeMap::from([
                (
                    PublicationDestination::Private,
                    ["manifest".to_owned()].into_iter().collect(),
                ),
                (
                    PublicationDestination::Public,
                    ["manifest".to_owned(), "public-sanitizer".to_owned()]
                        .into_iter()
                        .collect(),
                ),
            ]),
            minimum_unique_contributor_reviews: 1,
            sanitizer,
        }
    }

    fn candidate() -> CachePublicationCandidate {
        CachePublicationCandidate {
            semantic_digest: ContentDigest::sha256(b"semantic"),
            manifest_digest: ContentDigest::sha256(b"manifest"),
            payload_digest: ContentDigest::sha256(b"payload"),
            completion: ArtifactCompletionState::Complete,
            achieved_assurance: ArtifactAssuranceState::CrossChecked,
            disposition: ArtifactDisposition::Active,
            validator_evidence: vec![
                ValidatorEvidence {
                    validator_id: "manifest".to_owned(),
                    passed: true,
                    evidence_digest: ContentDigest::sha256(b"manifest-validation"),
                    establishes_assurance: Some(ArtifactAssuranceState::CrossChecked),
                },
                ValidatorEvidence {
                    validator_id: "public-sanitizer".to_owned(),
                    passed: true,
                    evidence_digest: ContentDigest::sha256(b"sanitization"),
                    establishes_assurance: None,
                },
            ],
            public_metadata: BTreeMap::from([(
                "artifact".to_owned(),
                serde_json::json!({
                    "kind": "ccm_matrix",
                    "source_identifier": "public-fixture-v1",
                    "repository": "example-org/public-shard"
                }),
            )]),
        }
    }

    fn authority(
        policy: &CachePublicationPolicy,
        mode: PublicationAuthorityMode,
        repository: &str,
    ) -> PublicationAuthority {
        PublicationAuthority {
            principal: "test-owner".to_owned(),
            mode,
            allowed_targets: [PublicationTarget::Both].into_iter().collect(),
            allowed_repositories: [repository.to_owned()].into_iter().collect(),
            policy_digest: policy.digest().unwrap().0,
        }
    }

    fn session(repository: &str) -> AuthenticatedGitHubSession {
        AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            repository,
            crate::RepositoryPermission::Write,
        )
    }

    #[test]
    fn owner_direct_private_publication_needs_evidence_not_manual_review() {
        let policy = policy();
        let request = TargetPublicationAuthorizationRequest {
            destination: PublicationDestination::Private,
            repository: "example-org/restricted-cache".to_owned(),
            authority: authority(
                &policy,
                PublicationAuthorityMode::OwnerDirect,
                "example-org/restricted-cache",
            ),
            contributor: None,
            reviews: Vec::new(),
        };
        let decision = authorize_target_publication(
            &policy,
            &candidate(),
            &request,
            &session("example-org/restricted-cache"),
        )
        .unwrap();
        assert!(decision.authorized, "{:?}", decision.reasons);
        assert!(decision.owner_direct);
        assert!(!decision.requires_pull_request);
    }

    #[test]
    fn owner_identity_never_replaces_missing_validator() {
        let policy = policy();
        let mut candidate = candidate();
        candidate.validator_evidence.clear();
        let request = TargetPublicationAuthorizationRequest {
            destination: PublicationDestination::Private,
            repository: "example-org/restricted-cache".to_owned(),
            authority: authority(
                &policy,
                PublicationAuthorityMode::OwnerDirect,
                "example-org/restricted-cache",
            ),
            contributor: None,
            reviews: Vec::new(),
        };
        let decision = authorize_target_publication(
            &policy,
            &candidate,
            &request,
            &session("example-org/restricted-cache"),
        )
        .unwrap();
        assert!(!decision.authorized);
        assert!(decision
            .reasons
            .iter()
            .any(|reason| reason.contains("validator")));
    }

    #[test]
    fn public_sanitizer_is_allowlist_based_and_detects_secrets() {
        let policy = policy();
        let mut candidate = candidate();
        candidate.public_metadata.insert(
            "token".to_owned(),
            Value::String("github_pat_not-public".to_owned()),
        );
        let request = TargetPublicationAuthorizationRequest {
            destination: PublicationDestination::Public,
            repository: "example-org/public-shard".to_owned(),
            authority: authority(
                &policy,
                PublicationAuthorityMode::OwnerDirect,
                "example-org/public-shard",
            ),
            contributor: None,
            reviews: Vec::new(),
        };
        let decision = authorize_target_publication(
            &policy,
            &candidate,
            &request,
            &session("example-org/public-shard"),
        )
        .unwrap();
        assert!(!decision.authorized);
        assert!(decision.sanitizer.unwrap().reasons.len() >= 2);
    }

    #[test]
    fn contributor_mode_requires_explicit_authorization_and_review() {
        let policy = policy();
        let request = TargetPublicationAuthorizationRequest {
            destination: PublicationDestination::Private,
            repository: "example-org/restricted-cache".to_owned(),
            authority: authority(
                &policy,
                PublicationAuthorityMode::ContributorReviewed,
                "example-org/restricted-cache",
            ),
            contributor: None,
            reviews: Vec::new(),
        };
        let decision = authorize_target_publication(
            &policy,
            &candidate(),
            &request,
            &session("example-org/restricted-cache"),
        )
        .unwrap();
        assert!(!decision.authorized);
        assert!(decision.requires_pull_request);
        assert!(decision
            .reasons
            .iter()
            .any(|reason| reason.contains("explicit authorization")));
        assert!(decision
            .reasons
            .iter()
            .any(|reason| reason.contains("valid unique reviews")));
    }

    #[test]
    fn reviewed_contributor_publication_uses_the_same_validation_and_sanitizer_gates() {
        let policy = policy();
        let request = TargetPublicationAuthorizationRequest {
            destination: PublicationDestination::Public,
            repository: "example-org/public-shard".to_owned(),
            authority: authority(
                &policy,
                PublicationAuthorityMode::ContributorReviewed,
                "example-org/public-shard",
            ),
            contributor: Some(ContributorPublicationEvidence {
                contributor_principal: "authorized-contributor".to_owned(),
                authorizing_principal: "test-owner".to_owned(),
                authorization_kind: ContributorAuthorizationKind::ContributorAgreement,
                authorized_destinations: [PublicationDestination::Public].into_iter().collect(),
                authorized_repositories: ["example-org/public-shard".to_owned()]
                    .into_iter()
                    .collect(),
                authorization_evidence_digest: ContentDigest::sha256(b"contributor-agreement"),
                source_repository: "authorized-contributor/cache-public".to_owned(),
                source_branch: "contribution/fixture".to_owned(),
                contribution_head_revision: "c".repeat(40),
                pull_request_number: 42,
                pull_request_evidence_digest: ContentDigest::sha256(b"pull-request-42"),
            }),
            reviews: vec![PublicationReviewEvidence {
                reviewer_principal: "approved-reviewer".to_owned(),
                approved: true,
                pull_request_number: 42,
                reviewed_head_revision: "c".repeat(40),
                evidence_digest: ContentDigest::sha256(b"review-evidence"),
            }],
        };
        let decision = authorize_target_publication(
            &policy,
            &candidate(),
            &request,
            &session("example-org/public-shard"),
        )
        .unwrap();
        assert!(decision.authorized, "{:?}", decision.reasons);
        assert!(!decision.owner_direct);
        assert!(decision.requires_pull_request);
        assert_eq!(
            decision.contributor_authorization_digest,
            Some(ContentDigest::sha256(b"contributor-agreement"))
        );
        assert!(decision.sanitizer.is_some_and(|report| report.accepted));
    }

    #[test]
    fn contributor_audit_evidence_cannot_omit_reviewer_approvals() {
        let audit = crate::TargetPublicationAuditEvidence {
            policy_id: "cache-publication-v1".to_owned(),
            authority_mode: PublicationAuthorityMode::ContributorReviewed,
            validation_evidence_digests: vec![ContentDigest::sha256(b"validation")],
            contributor_authorization_digest: Some(ContentDigest::sha256(b"authorization")),
            reviewer_approvals: Vec::new(),
        };
        assert!(audit
            .validate()
            .unwrap_err()
            .to_string()
            .contains("authority mode"));
    }
}
