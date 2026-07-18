//! Replayable evidence for owner/contributor live GitHub publication acceptance.

use crate::{
    canonical_digest, verify_large_corpus_acceptance_report, CacheError, ContentDigest,
    LargeCorpusAcceptanceReport, LargeCorpusShardObservation, PublicationDestination,
    PublicationReceipt, TopologyRegistry,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use xc_core::PublicationAuthorityMode;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LivePublicationScenario {
    OwnerPrivate,
    OwnerPublic,
    OwnerDual,
    ContributorReviewedPublic,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivePublicationObservation {
    pub scenario: LivePublicationScenario,
    pub receipts: Vec<PublicationReceipt>,
    pub peak_local_bytes: u64,
    pub persistent_full_clone_bytes: u64,
    pub public_sanitization_evidence_digest: Option<ContentDigest>,
    pub github_api_observation_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoMutationControlObservation {
    pub execute_remote_mutations: bool,
    pub remote_mutation_requests: u64,
    pub repository_heads_before: BTreeMap<String, String>,
    pub repository_heads_after: BTreeMap<String, String>,
    pub github_api_observation_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveGitHubPublicationAcceptanceRecord {
    pub schema_version: u32,
    pub toolkit_version: String,
    pub source_revision: String,
    pub required_validation_evidence_digest: ContentDigest,
    pub observations: Vec<LivePublicationObservation>,
    pub no_mutation_control: NoMutationControlObservation,
    pub finite_scope_statement: String,
    pub evidence_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveGitHubLargeCorpusAcceptanceRecord {
    pub schema_version: u32,
    pub toolkit_version: String,
    pub source_revision: String,
    pub github_repositories: BTreeSet<String>,
    pub topology: TopologyRegistry,
    pub shard_observations: Vec<LargeCorpusShardObservation>,
    pub minimum_corpus_logical_bytes: u64,
    pub report: LargeCorpusAcceptanceReport,
    pub github_measurement_digest: ContentDigest,
    pub measured_on_live_github: bool,
    pub finite_scope_statement: String,
    pub evidence_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "acceptance_kind", content = "record", rename_all = "snake_case")]
pub enum LiveGitHubAcceptanceArtifact {
    Publication(Box<LiveGitHubPublicationAcceptanceRecord>),
    LargeCorpus(Box<LiveGitHubLargeCorpusAcceptanceRecord>),
}

#[derive(Serialize)]
struct EvidenceEnvelope<'a> {
    schema_version: u32,
    toolkit_version: &'a str,
    source_revision: &'a str,
    required_validation_evidence_digest: &'a ContentDigest,
    observations: &'a [LivePublicationObservation],
    no_mutation_control: &'a NoMutationControlObservation,
    finite_scope_statement: &'a str,
}

#[derive(Serialize)]
struct LargeCorpusLiveEvidenceEnvelope<'a> {
    schema_version: u32,
    toolkit_version: &'a str,
    source_revision: &'a str,
    github_repositories: &'a BTreeSet<String>,
    topology: &'a TopologyRegistry,
    shard_observations: &'a [LargeCorpusShardObservation],
    minimum_corpus_logical_bytes: u64,
    report: &'a LargeCorpusAcceptanceReport,
    github_measurement_digest: &'a ContentDigest,
    measured_on_live_github: bool,
    finite_scope_statement: &'a str,
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_repository(value: &str) -> bool {
    let mut pieces = value.split('/');
    pieces.next().is_some_and(|piece| !piece.is_empty())
        && pieces.next().is_some_and(|piece| !piece.is_empty())
        && pieces.next().is_none()
        && !value.chars().any(char::is_whitespace)
}

impl LivePublicationObservation {
    fn validate(&self, required_validation: &ContentDigest) -> Result<(), CacheError> {
        let (destinations, authority) = match self.scenario {
            LivePublicationScenario::OwnerPrivate => (
                BTreeSet::from([PublicationDestination::Private]),
                PublicationAuthorityMode::OwnerDirect,
            ),
            LivePublicationScenario::OwnerPublic => (
                BTreeSet::from([PublicationDestination::Public]),
                PublicationAuthorityMode::OwnerDirect,
            ),
            LivePublicationScenario::OwnerDual => (
                BTreeSet::from([
                    PublicationDestination::Private,
                    PublicationDestination::Public,
                ]),
                PublicationAuthorityMode::OwnerDirect,
            ),
            LivePublicationScenario::ContributorReviewedPublic => (
                BTreeSet::from([PublicationDestination::Public]),
                PublicationAuthorityMode::ContributorReviewed,
            ),
        };
        if self.receipts.len() != destinations.len()
            || self.peak_local_bytes == 0
            || self.persistent_full_clone_bytes != 0
            || !self.github_api_observation_digest.validate()
            || (destinations.contains(&PublicationDestination::Public)
                != self
                    .public_sanitization_evidence_digest
                    .as_ref()
                    .is_some_and(ContentDigest::validate))
        {
            return Err(CacheError::InvalidManifest(format!(
                "live GitHub scenario {:?} has incomplete storage, sanitizer, or API evidence",
                self.scenario
            )));
        }
        let mut observed_destinations = BTreeSet::new();
        let mut payload_digest = None;
        for receipt in &self.receipts {
            receipt.validate()?;
            if receipt.authority_mode != authority
                || !valid_repository(&receipt.authorized_repository)
                || !receipt
                    .validation_evidence_digests
                    .contains(required_validation)
                || receipt
                    .payload_commit_ids
                    .iter()
                    .chain(receipt.payload_batch_record_commit_ids.iter())
                    .chain(std::iter::once(&receipt.metadata_commit_id))
                    .any(|revision| !valid_revision(revision))
                || receipt
                    .remote_verification_results
                    .iter()
                    .any(|result| !valid_revision(&result.commit_id))
                || !observed_destinations.insert(receipt.destination)
            {
                return Err(CacheError::InvalidManifest(format!(
                    "live GitHub scenario {:?} contains an invalid receipt",
                    self.scenario
                )));
            }
            if let Some(expected) = &payload_digest {
                if expected != &receipt.canonical_payload_digest {
                    return Err(CacheError::InvalidManifest(
                        "dual publication receipts do not bind one canonical payload".to_owned(),
                    ));
                }
            } else {
                payload_digest = Some(receipt.canonical_payload_digest.clone());
            }
        }
        if observed_destinations != destinations {
            return Err(CacheError::InvalidManifest(format!(
                "live GitHub scenario {:?} has the wrong targets",
                self.scenario
            )));
        }
        Ok(())
    }
}

impl NoMutationControlObservation {
    fn validate(&self) -> Result<(), CacheError> {
        if self.execute_remote_mutations
            || self.remote_mutation_requests != 0
            || self.repository_heads_before.is_empty()
            || self.repository_heads_before != self.repository_heads_after
            || !self.github_api_observation_digest.validate()
            || self
                .repository_heads_before
                .iter()
                .any(|(repository, revision)| {
                    !valid_repository(repository) || !valid_revision(revision)
                })
        {
            return Err(CacheError::InvalidManifest(
                "no-mutation control did not preserve every observed GitHub ref".to_owned(),
            ));
        }
        Ok(())
    }
}

impl LiveGitHubPublicationAcceptanceRecord {
    pub fn refresh_evidence_digest(&mut self) -> Result<(), CacheError> {
        self.evidence_digest = canonical_digest(&self.envelope())?;
        Ok(())
    }

    fn envelope(&self) -> EvidenceEnvelope<'_> {
        EvidenceEnvelope {
            schema_version: self.schema_version,
            toolkit_version: &self.toolkit_version,
            source_revision: &self.source_revision,
            required_validation_evidence_digest: &self.required_validation_evidence_digest,
            observations: &self.observations,
            no_mutation_control: &self.no_mutation_control,
            finite_scope_statement: &self.finite_scope_statement,
        }
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.toolkit_version != "0.13.0"
            || !valid_revision(&self.source_revision)
            || !self.required_validation_evidence_digest.validate()
            || self.finite_scope_statement
                != "finite live GitHub publication workflow acceptance; no mathematical assurance promotion"
            || !self.evidence_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "live GitHub acceptance identity is incomplete".to_owned(),
            ));
        }
        let expected = BTreeSet::from([
            LivePublicationScenario::OwnerPrivate,
            LivePublicationScenario::OwnerPublic,
            LivePublicationScenario::OwnerDual,
            LivePublicationScenario::ContributorReviewedPublic,
        ]);
        let mut actual = BTreeSet::new();
        let mut owner_principal = None;
        for observation in &self.observations {
            if !actual.insert(observation.scenario) {
                return Err(CacheError::InvalidManifest(
                    "live GitHub acceptance repeats a publication scenario".to_owned(),
                ));
            }
            observation.validate(&self.required_validation_evidence_digest)?;
            if observation.scenario != LivePublicationScenario::ContributorReviewedPublic {
                for receipt in &observation.receipts {
                    if let Some(expected_principal) = &owner_principal {
                        if expected_principal != &receipt.principal {
                            return Err(CacheError::InvalidManifest(
                                "owner-direct scenarios used different authenticated principals"
                                    .to_owned(),
                            ));
                        }
                    } else {
                        owner_principal = Some(receipt.principal.clone());
                    }
                }
            }
        }
        if actual != expected {
            return Err(CacheError::InvalidManifest(
                "live GitHub acceptance does not cover all four publication scenarios".to_owned(),
            ));
        }
        self.no_mutation_control.validate()?;
        if canonical_digest(&self.envelope())? != self.evidence_digest {
            return Err(CacheError::DigestMismatch {
                expected: self.evidence_digest.0.clone(),
                actual: canonical_digest(&self.envelope())?.0,
            });
        }
        Ok(())
    }
}

pub fn verify_live_github_publication_acceptance(
    record: &LiveGitHubPublicationAcceptanceRecord,
) -> Result<(), CacheError> {
    record.validate()
}

impl LiveGitHubLargeCorpusAcceptanceRecord {
    fn envelope(&self) -> LargeCorpusLiveEvidenceEnvelope<'_> {
        LargeCorpusLiveEvidenceEnvelope {
            schema_version: self.schema_version,
            toolkit_version: &self.toolkit_version,
            source_revision: &self.source_revision,
            github_repositories: &self.github_repositories,
            topology: &self.topology,
            shard_observations: &self.shard_observations,
            minimum_corpus_logical_bytes: self.minimum_corpus_logical_bytes,
            report: &self.report,
            github_measurement_digest: &self.github_measurement_digest,
            measured_on_live_github: self.measured_on_live_github,
            finite_scope_statement: &self.finite_scope_statement,
        }
    }

    pub fn refresh_evidence_digest(&mut self) -> Result<(), CacheError> {
        self.evidence_digest = canonical_digest(&self.envelope())?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.toolkit_version != "0.13.0"
            || !valid_revision(&self.source_revision)
            || self.github_repositories.len() < 2
            || self.github_repositories.iter().any(|repository| !valid_repository(repository))
            || !self.github_measurement_digest.validate()
            || !self.measured_on_live_github
            || self.finite_scope_statement
                != "finite measured live GitHub large-corpus acceptance; no mathematical assurance promotion"
            || !self.report.accepted
            || !self.evidence_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "live GitHub large-corpus acceptance identity is incomplete".to_owned(),
            ));
        }
        verify_large_corpus_acceptance_report(
            &self.report,
            &self.topology,
            &self.shard_observations,
            self.minimum_corpus_logical_bytes,
        )?;
        let expected = canonical_digest(&self.envelope())?;
        if expected != self.evidence_digest {
            return Err(CacheError::DigestMismatch {
                expected: self.evidence_digest.0.clone(),
                actual: expected.0,
            });
        }
        Ok(())
    }
}

pub fn verify_live_github_acceptance(
    artifact: &LiveGitHubAcceptanceArtifact,
) -> Result<(), CacheError> {
    match artifact {
        LiveGitHubAcceptanceArtifact::Publication(record) => record.validate(),
        LiveGitHubAcceptanceArtifact::LargeCorpus(record) => record.validate(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PublicationReviewEvidence, RemoteCommitVerificationResult};

    fn digest(value: &str) -> ContentDigest {
        ContentDigest::sha256(value.as_bytes())
    }

    fn revision(character: char) -> String {
        std::iter::repeat_n(character, 40).collect()
    }

    fn receipt(
        destination: PublicationDestination,
        authority: PublicationAuthorityMode,
        repository: &str,
        validation: &ContentDigest,
        payload: &ContentDigest,
    ) -> PublicationReceipt {
        let idempotency = digest(&format!("{repository}-{destination:?}-{authority:?}"));
        let contributor = (authority == PublicationAuthorityMode::ContributorReviewed)
            .then(|| digest("contributor-authorization"));
        let reviews = if contributor.is_some() {
            vec![PublicationReviewEvidence {
                reviewer_principal: "owner-reviewer".to_owned(),
                approved: true,
                pull_request_number: 17,
                reviewed_head_revision: revision('a'),
                evidence_digest: digest("review"),
            }]
        } else {
            Vec::new()
        };
        PublicationReceipt {
            schema_version: 1,
            transaction_id: idempotency.0.clone(),
            idempotency_key: idempotency,
            destination,
            principal: if contributor.is_some() {
                "controlled-publisher".to_owned()
            } else {
                "repository-owner".to_owned()
            },
            authorized_repository: repository.to_owned(),
            repository_permission_evidence_digest: digest("permission"),
            shard_id: format!("{repository}-001"),
            branch: "main".to_owned(),
            semantic_digest: digest("semantic"),
            canonical_payload_digest: payload.clone(),
            manifest_digest: digest("manifest"),
            transport_digest: digest("transport"),
            policy_digest: digest("policy"),
            policy_id: "release-publication-policy".to_owned(),
            authority_mode: authority,
            validation_evidence_digests: vec![validation.clone()],
            contributor_authorization_digest: contributor,
            reviewer_approvals: reviews,
            payload_commit_ids: vec![revision('b')],
            payload_batch_record_commit_ids: vec![revision('c')],
            payload_batch_record_digests: BTreeMap::from([(
                "transactions/batch.json".to_owned(),
                digest("batch-record"),
            )]),
            metadata_commit_id: revision('d'),
            metadata_file_digests: BTreeMap::from([(
                "manifests/artifact.json".to_owned(),
                digest("metadata"),
            )]),
            discoverability_subject_digests: BTreeMap::from([(
                "index/family.json".to_owned(),
                digest("index"),
            )]),
            remote_verification_results: vec![RemoteCommitVerificationResult {
                phase: "payload_batch".to_owned(),
                sequence: 0,
                commit_id: revision('b'),
                verified: true,
                content_digests: vec![digest("payload-part")],
            }],
            verified_at_unix_seconds: 1_750_000_000,
        }
    }

    fn observation(
        scenario: LivePublicationScenario,
        receipts: Vec<PublicationReceipt>,
    ) -> LivePublicationObservation {
        let has_public = receipts
            .iter()
            .any(|receipt| receipt.destination == PublicationDestination::Public);
        LivePublicationObservation {
            scenario,
            receipts,
            peak_local_bytes: 8 * 1024 * 1024,
            persistent_full_clone_bytes: 0,
            public_sanitization_evidence_digest: has_public.then(|| digest("sanitizer")),
            github_api_observation_digest: digest("github-api"),
        }
    }

    fn record() -> LiveGitHubPublicationAcceptanceRecord {
        let validation = digest("common-validator-evidence");
        let payload = digest("canonical-payload");
        let mut record = LiveGitHubPublicationAcceptanceRecord {
            schema_version: 1,
            toolkit_version: "0.13.0".to_owned(),
            source_revision: revision('e'),
            required_validation_evidence_digest: validation.clone(),
            observations: vec![
                observation(
                    LivePublicationScenario::OwnerPrivate,
                    vec![receipt(
                        PublicationDestination::Private,
                        PublicationAuthorityMode::OwnerDirect,
                        "example-org/restricted-cache",
                        &validation,
                        &payload,
                    )],
                ),
                observation(
                    LivePublicationScenario::OwnerPublic,
                    vec![receipt(
                        PublicationDestination::Public,
                        PublicationAuthorityMode::OwnerDirect,
                        "example-org/public-shard",
                        &validation,
                        &payload,
                    )],
                ),
                observation(
                    LivePublicationScenario::OwnerDual,
                    vec![
                        receipt(
                            PublicationDestination::Private,
                            PublicationAuthorityMode::OwnerDirect,
                            "example-org/restricted-cache",
                            &validation,
                            &payload,
                        ),
                        receipt(
                            PublicationDestination::Public,
                            PublicationAuthorityMode::OwnerDirect,
                            "example-org/public-shard",
                            &validation,
                            &payload,
                        ),
                    ],
                ),
                observation(
                    LivePublicationScenario::ContributorReviewedPublic,
                    vec![receipt(
                        PublicationDestination::Public,
                        PublicationAuthorityMode::ContributorReviewed,
                        "example-org/public-shard",
                        &validation,
                        &payload,
                    )],
                ),
            ],
            no_mutation_control: NoMutationControlObservation {
                execute_remote_mutations: false,
                remote_mutation_requests: 0,
                repository_heads_before: BTreeMap::from([(
                    "example-org/public-shard".to_owned(),
                    revision('f'),
                )]),
                repository_heads_after: BTreeMap::from([(
                    "example-org/public-shard".to_owned(),
                    revision('f'),
                )]),
                github_api_observation_digest: digest("no-mutation-api"),
            },
            finite_scope_statement: "finite live GitHub publication workflow acceptance; no mathematical assurance promotion".to_owned(),
            evidence_digest: ContentDigest("0".repeat(64)),
        };
        record.refresh_evidence_digest().unwrap();
        record
    }

    #[test]
    fn live_record_covers_all_scenarios_and_replays_exact_receipts() {
        let record = record();
        verify_live_github_publication_acceptance(&record).unwrap();
        let encoded = serde_json::to_vec(&record).unwrap();
        let decoded: LiveGitHubPublicationAcceptanceRecord =
            serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, record);
        verify_live_github_publication_acceptance(&decoded).unwrap();
    }

    #[test]
    fn live_record_rejects_missing_scenario_ref_drift_and_tampering() {
        let mut missing = record();
        missing.observations.pop();
        missing.refresh_evidence_digest().unwrap();
        assert!(missing.validate().is_err());

        let mut mutated = record();
        mutated
            .no_mutation_control
            .repository_heads_after
            .insert("example-org/public-shard".to_owned(), revision('1'));
        mutated.refresh_evidence_digest().unwrap();
        assert!(mutated.validate().is_err());

        let mut tampered = record();
        tampered.observations[0].receipts[0].principal = "different-owner".to_owned();
        assert!(tampered.validate().is_err());
    }
}
