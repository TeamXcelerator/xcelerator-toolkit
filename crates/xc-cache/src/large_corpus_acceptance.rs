//! Integrated, bounded evidence for the ACC-012 large-corpus milestone.
//!
//! Scale fixtures use measured or simulated byte counters rather than
//! allocating a publication-scale payload during ordinary tests. The report
//! composes evidence emitted by the real topology, audit, retrieval,
//! publication, and receipt paths and applies the production byte limits.

use crate::{
    canonical_digest, CacheError, CacheVisibility, ContentDigest, TopologyRegistry,
    TopologyShardStatus, GITHUB_MAX_PUBLICATION_BATCH_BYTES, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LargeCorpusShardObservation {
    pub shard_id: String,
    pub revision: String,
    pub index_entry_count: u64,
    pub complete_artifact_count: u64,
    pub publication_receipt_count: u64,
    pub logical_payload_bytes: u64,
    pub unique_payload_bytes: u64,
    pub reachable_payload_bytes: u64,
    pub metadata_bytes: u64,
    pub history_and_emergency_reserve_bytes: u64,
    pub abandoned_reachable_bytes: u64,
    pub accounted_bytes: u64,
    pub remaining_capacity_bytes: u64,
    pub durable_history_complete: bool,
    pub audit_error_count: u64,
}

impl LargeCorpusShardObservation {
    fn validate(&self) -> Result<(), CacheError> {
        let expected_accounted = self
            .reachable_payload_bytes
            .saturating_add(self.metadata_bytes)
            .saturating_add(self.history_and_emergency_reserve_bytes)
            .saturating_add(self.abandoned_reachable_bytes);
        if self.shard_id.trim().is_empty()
            || self.revision.trim().is_empty()
            || self.index_entry_count == 0
            || self.complete_artifact_count == 0
            || self.publication_receipt_count == 0
            || self.logical_payload_bytes == 0
            || self.unique_payload_bytes == 0
            || self.unique_payload_bytes > self.reachable_payload_bytes
            || expected_accounted != self.accounted_bytes
            || self.accounted_bytes > GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES
            || self.remaining_capacity_bytes
                != GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES - self.accounted_bytes
            || !self.durable_history_complete
            || self.audit_error_count != 0
        {
            return Err(CacheError::InvalidManifest(format!(
                "large-corpus shard observation {:?} is incomplete or inconsistent",
                self.shard_id
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectiveRetrievalObservation {
    pub shard_id: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub transport_digest: ContentDigest,
    pub receipt_digest: ContentDigest,
    pub selected_logical_payload_bytes: u64,
    pub metadata_bytes_read: u64,
    pub payload_bytes_downloaded: u64,
    pub reused_verified_part_bytes: u64,
    pub peak_local_bytes: u64,
    pub persistent_full_clone_bytes: u64,
    pub decoded_payload_verified: bool,
}

impl SelectiveRetrievalObservation {
    fn validate(&self, corpus_logical_bytes: u64) -> Result<(), CacheError> {
        if self.shard_id.trim().is_empty()
            || [
                &self.semantic_digest,
                &self.manifest_digest,
                &self.transport_digest,
                &self.receipt_digest,
            ]
            .into_iter()
            .any(|digest| !digest.validate())
            || self.selected_logical_payload_bytes == 0
            || self.selected_logical_payload_bytes >= corpus_logical_bytes
            || self.payload_bytes_downloaded
                > self
                    .selected_logical_payload_bytes
                    .saturating_add(self.metadata_bytes_read)
            || self.reused_verified_part_bytes > self.selected_logical_payload_bytes
            || self.peak_local_bytes >= corpus_logical_bytes
            || self.persistent_full_clone_bytes != 0
            || !self.decoded_payload_verified
        {
            return Err(CacheError::InvalidManifest(
                "selective retrieval did not prove bounded no-clone decoded access".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LargeCorpusPublicationObservation {
    pub shard_id: String,
    pub semantic_digest: ContentDigest,
    pub receipt_digest: ContentDigest,
    pub newly_committed_payload_bytes: u64,
    pub newly_committed_metadata_bytes: u64,
    pub projected_history_bytes: u64,
    pub batch_payload_bytes: Vec<u64>,
    pub maximum_part_bytes: u64,
    pub accounted_before_bytes: u64,
    pub accounted_after_bytes: u64,
    pub persistent_full_clone_bytes: u64,
    pub remote_payload_verified: bool,
    pub remote_receipt_verified: bool,
}

impl LargeCorpusPublicationObservation {
    fn projected_addition(&self) -> u64 {
        self.newly_committed_payload_bytes
            .saturating_add(self.newly_committed_metadata_bytes)
            .saturating_add(self.projected_history_bytes)
    }

    fn validate(&self) -> Result<(), CacheError> {
        let batch_total = self
            .batch_payload_bytes
            .iter()
            .try_fold(0_u64, |sum, bytes| {
                if *bytes == 0 || *bytes > GITHUB_MAX_PUBLICATION_BATCH_BYTES {
                    return Err(CacheError::ResourceLimit(format!(
                        "publication batch {bytes} exceeds the 1,000,000,000-byte limit"
                    )));
                }
                sum.checked_add(*bytes).ok_or_else(|| {
                    CacheError::ResourceLimit("publication batch total exceeds u64".to_owned())
                })
            })?;
        if self.shard_id.trim().is_empty()
            || !self.semantic_digest.validate()
            || !self.receipt_digest.validate()
            || self.newly_committed_payload_bytes == 0
            || batch_total != self.newly_committed_payload_bytes
            || self.maximum_part_bytes == 0
            || self.maximum_part_bytes >= 100_000_000
            || self.accounted_after_bytes
                != self
                    .accounted_before_bytes
                    .saturating_add(self.projected_addition())
            || self.accounted_after_bytes > GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES
            || self.persistent_full_clone_bytes != 0
            || !self.remote_payload_verified
            || !self.remote_receipt_verified
        {
            return Err(CacheError::InvalidManifest(
                "large-corpus publication evidence violates identity, byte, capacity, no-clone, or verification rules"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LargeCorpusShardSummary {
    pub shard_id: String,
    pub artifact_inventory: u64,
    pub publication_receipts: u64,
    pub logical_payload_bytes: u64,
    pub unique_payload_bytes: u64,
    pub reachable_payload_bytes: u64,
    pub history_and_overhead_reserve_bytes: u64,
    pub remaining_capacity_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LargeCorpusAcceptanceReport {
    pub schema_version: u32,
    pub family: String,
    pub visibility: CacheVisibility,
    pub topology_digest: ContentDigest,
    pub topology_contains_artifact_inventory: bool,
    pub corpus_logical_payload_bytes: u64,
    pub sum_shard_unique_payload_bytes: u64,
    pub shard_summaries: Vec<LargeCorpusShardSummary>,
    pub retrieval: SelectiveRetrievalObservation,
    pub publication: LargeCorpusPublicationObservation,
    pub rollover_was_required: bool,
    pub accepted: bool,
    pub finite_scale_statement: String,
    pub evidence_digest: ContentDigest,
}

#[derive(Serialize)]
struct LargeCorpusEvidenceEnvelope<'a> {
    schema_version: u32,
    family: &'a str,
    visibility: CacheVisibility,
    topology_digest: &'a ContentDigest,
    observations: &'a [LargeCorpusShardObservation],
    retrieval: &'a SelectiveRetrievalObservation,
    publication: &'a LargeCorpusPublicationObservation,
    minimum_corpus_logical_bytes: u64,
}

/// Reconcile one topology-only registry, all routed shard audits, one
/// selective retrieval, and one no-checkout publication into ACC-012 evidence.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_large_corpus_acceptance(
    topology: &TopologyRegistry,
    family: &str,
    visibility: CacheVisibility,
    observations: &[LargeCorpusShardObservation],
    retrieval: &SelectiveRetrievalObservation,
    publication: &LargeCorpusPublicationObservation,
    minimum_corpus_logical_bytes: u64,
) -> Result<LargeCorpusAcceptanceReport, CacheError> {
    topology.validate()?;
    let topology_digest = topology.digest()?;
    let route = topology.route(family, visibility).ok_or_else(|| {
        CacheError::InvalidManifest(
            "large-corpus evidence has no matching family topology route".to_owned(),
        )
    })?;
    if family.trim().is_empty() || minimum_corpus_logical_bytes == 0 {
        return Err(CacheError::InvalidManifest(
            "large-corpus family and scale threshold are required".to_owned(),
        ));
    }
    let route_ids = route
        .ordered_shards
        .iter()
        .map(|shard| shard.shard_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut by_shard = BTreeMap::new();
    for observation in observations {
        observation.validate()?;
        if !route_ids.contains(observation.shard_id.as_str())
            || by_shard
                .insert(observation.shard_id.as_str(), observation)
                .is_some()
        {
            return Err(CacheError::InvalidManifest(
                "large-corpus shard observations are duplicated or outside the route".to_owned(),
            ));
        }
    }
    if by_shard.len() != route_ids.len() {
        return Err(CacheError::InvalidManifest(
            "large-corpus evidence must audit every routed shard".to_owned(),
        ));
    }
    let corpus_logical_payload_bytes = observations
        .iter()
        .map(|observation| observation.logical_payload_bytes)
        .sum::<u64>();
    let sum_shard_unique_payload_bytes = observations
        .iter()
        .map(|observation| observation.unique_payload_bytes)
        .sum::<u64>();
    if corpus_logical_payload_bytes < minimum_corpus_logical_bytes {
        return Err(CacheError::ResourceLimit(format!(
            "observed corpus has {corpus_logical_payload_bytes} logical bytes, below required {minimum_corpus_logical_bytes}"
        )));
    }
    retrieval.validate(corpus_logical_payload_bytes)?;
    publication.validate()?;
    if !route_ids.contains(retrieval.shard_id.as_str())
        || !route_ids.contains(publication.shard_id.as_str())
    {
        return Err(CacheError::InvalidManifest(
            "retrieval or publication selected a shard outside the topology route".to_owned(),
        ));
    }
    let selected_route = route
        .ordered_shards
        .iter()
        .find(|shard| shard.shard_id == publication.shard_id)
        .expect("publication shard membership was checked");
    if selected_route.status != TopologyShardStatus::Writable {
        return Err(CacheError::NoWritableShard(
            "large-corpus publication selected a nonwritable shard".to_owned(),
        ));
    }
    let addition = publication.projected_addition();
    let admissible = route
        .ordered_shards
        .iter()
        .filter(|shard| shard.status == TopologyShardStatus::Writable)
        .filter_map(|shard| {
            let observed = by_shard[shard.shard_id.as_str()];
            observed
                .accounted_bytes
                .checked_add(addition)
                .filter(|projected| *projected <= GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES)
                .map(|_| observed)
        })
        .max_by_key(|observation| observation.accounted_bytes)
        .ok_or_else(|| CacheError::NoWritableShard("no shard admits the publication".to_owned()))?;
    if admissible.shard_id != publication.shard_id
        || by_shard[publication.shard_id.as_str()].accounted_bytes
            != publication.accounted_before_bytes
    {
        return Err(CacheError::InvalidManifest(
            "publication did not select the fullest admissible shard-local ledger".to_owned(),
        ));
    }
    let rollover_was_required = route
        .ordered_shards
        .iter()
        .take_while(|shard| shard.shard_id != publication.shard_id)
        .filter(|shard| shard.status == TopologyShardStatus::Writable)
        .any(|shard| {
            by_shard[shard.shard_id.as_str()]
                .accounted_bytes
                .saturating_add(addition)
                > GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES
        });
    let shard_summaries = route
        .ordered_shards
        .iter()
        .map(|shard| {
            let observed = by_shard[shard.shard_id.as_str()];
            LargeCorpusShardSummary {
                shard_id: shard.shard_id.clone(),
                artifact_inventory: observed.complete_artifact_count,
                publication_receipts: observed.publication_receipt_count,
                logical_payload_bytes: observed.logical_payload_bytes,
                unique_payload_bytes: observed.unique_payload_bytes,
                reachable_payload_bytes: observed.reachable_payload_bytes,
                history_and_overhead_reserve_bytes: observed
                    .history_and_emergency_reserve_bytes
                    .saturating_add(observed.metadata_bytes)
                    .saturating_add(observed.abandoned_reachable_bytes),
                remaining_capacity_bytes: observed.remaining_capacity_bytes,
            }
        })
        .collect::<Vec<_>>();
    let evidence_digest = canonical_digest(&LargeCorpusEvidenceEnvelope {
        schema_version: 1,
        family,
        visibility,
        topology_digest: &topology_digest,
        observations,
        retrieval,
        publication,
        minimum_corpus_logical_bytes,
    })?;
    Ok(LargeCorpusAcceptanceReport {
        schema_version: 1,
        family: family.to_owned(),
        visibility,
        topology_digest,
        topology_contains_artifact_inventory: false,
        corpus_logical_payload_bytes,
        sum_shard_unique_payload_bytes,
        shard_summaries,
        retrieval: retrieval.clone(),
        publication: publication.clone(),
        rollover_was_required,
        accepted: true,
        finite_scale_statement: "finite measured or simulated large-corpus acceptance evidence; production GitHub execution remains separately provenance-labelled"
            .to_owned(),
        evidence_digest,
    })
}

pub fn verify_large_corpus_acceptance_report(
    report: &LargeCorpusAcceptanceReport,
    topology: &TopologyRegistry,
    observations: &[LargeCorpusShardObservation],
    minimum_corpus_logical_bytes: u64,
) -> Result<(), CacheError> {
    let replay = evaluate_large_corpus_acceptance(
        topology,
        &report.family,
        report.visibility,
        observations,
        &report.retrieval,
        &report.publication,
        minimum_corpus_logical_bytes,
    )?;
    if &replay != report {
        return Err(CacheError::InvalidManifest(
            "large-corpus acceptance report does not match replayed evidence".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        verify_live_github_acceptance, ArtifactFamilyRoute, LiveGitHubAcceptanceArtifact,
        LiveGitHubLargeCorpusAcceptanceRecord, TopologyShardRoute,
    };

    fn topology() -> TopologyRegistry {
        TopologyRegistry {
            schema_version: 1,
            generation: 7,
            previous_registry_digest: Some(ContentDigest::sha256(b"generation-6")),
            policy_digest: ContentDigest::sha256(b"large-corpus-policy"),
            trust_anchor_ids: vec!["release-key".to_owned()],
            family_routes: vec![ArtifactFamilyRoute {
                family: "large-corpus".to_owned(),
                visibility: CacheVisibility::Private,
                ordered_shards: vec![
                    TopologyShardRoute {
                        shard_id: "restricted-shard-001".to_owned(),
                        endpoint_id: "restricted-shard-001".to_owned(),
                        sequence: 1,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: Some("restricted-shard-002".to_owned()),
                    },
                    TopologyShardRoute {
                        shard_id: "restricted-shard-002".to_owned(),
                        endpoint_id: "restricted-shard-002".to_owned(),
                        sequence: 2,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: None,
                    },
                ],
            }],
        }
    }

    fn observation(
        shard_id: &str,
        logical: u64,
        unique: u64,
        reachable: u64,
        metadata: u64,
        reserve: u64,
    ) -> LargeCorpusShardObservation {
        let accounted = reachable + metadata + reserve;
        LargeCorpusShardObservation {
            shard_id: shard_id.to_owned(),
            revision: ContentDigest::sha256(shard_id.as_bytes()).0,
            index_entry_count: 100,
            complete_artifact_count: 100,
            publication_receipt_count: 100,
            logical_payload_bytes: logical,
            unique_payload_bytes: unique,
            reachable_payload_bytes: reachable,
            metadata_bytes: metadata,
            history_and_emergency_reserve_bytes: reserve,
            abandoned_reachable_bytes: 0,
            accounted_bytes: accounted,
            remaining_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES - accounted,
            durable_history_complete: true,
            audit_error_count: 0,
        }
    }

    #[test]
    fn large_corpus_report_proves_selective_access_and_capacity_rollover() {
        let observations = vec![
            observation(
                "restricted-shard-001",
                62_000_000_000,
                58_000_000_000,
                98_000_000_000,
                100_000_000,
                100_000_000,
            ),
            observation(
                "restricted-shard-002",
                5_000_000_000,
                4_000_000_000,
                60_000_000_000,
                1_000_000_000,
                1_000_000_000,
            ),
        ];
        let retrieval = SelectiveRetrievalObservation {
            shard_id: "restricted-shard-001".to_owned(),
            semantic_digest: ContentDigest::sha256(b"selected-configuration"),
            manifest_digest: ContentDigest::sha256(b"selected-manifest"),
            transport_digest: ContentDigest::sha256(b"selected-transport"),
            receipt_digest: ContentDigest::sha256(b"selected-receipt"),
            selected_logical_payload_bytes: 750_000_000,
            metadata_bytes_read: 2_000_000,
            payload_bytes_downloaded: 740_000_000,
            reused_verified_part_bytes: 10_000_000,
            peak_local_bytes: 1_500_000_000,
            persistent_full_clone_bytes: 0,
            decoded_payload_verified: true,
        };
        let publication = LargeCorpusPublicationObservation {
            shard_id: "restricted-shard-002".to_owned(),
            semantic_digest: ContentDigest::sha256(b"new-configuration"),
            receipt_digest: ContentDigest::sha256(b"new-receipt"),
            newly_committed_payload_bytes: 1_800_000_000,
            newly_committed_metadata_bytes: 10_000_000,
            projected_history_bytes: 20_000_000,
            batch_payload_bytes: vec![1_000_000_000, 800_000_000],
            maximum_part_bytes: 90 * 1024 * 1024,
            accounted_before_bytes: 62_000_000_000,
            accounted_after_bytes: 63_830_000_000,
            persistent_full_clone_bytes: 0,
            remote_payload_verified: true,
            remote_receipt_verified: true,
        };
        let report = evaluate_large_corpus_acceptance(
            &topology(),
            "large-corpus",
            CacheVisibility::Private,
            &observations,
            &retrieval,
            &publication,
            62_000_000_000,
        )
        .unwrap();
        assert!(report.accepted);
        assert!(report.rollover_was_required);
        assert!(!report.topology_contains_artifact_inventory);
        assert_eq!(report.shard_summaries.len(), 2);
        assert_eq!(report.publication.batch_payload_bytes[0], 1_000_000_000);
        let encoded = serde_json::to_vec(&report).unwrap();
        let decoded: LargeCorpusAcceptanceReport = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, report);
        verify_large_corpus_acceptance_report(&decoded, &topology(), &observations, 62_000_000_000)
            .unwrap();

        let mut live = LiveGitHubLargeCorpusAcceptanceRecord {
            schema_version: 1,
            toolkit_version: "0.13.0".to_owned(),
            source_revision: "e".repeat(40),
            github_repositories: BTreeSet::from([
                "example-org/registry".to_owned(),
                "example-org/restricted-cache-001".to_owned(),
                "example-org/restricted-cache-002".to_owned(),
            ]),
            topology: topology(),
            shard_observations: observations.clone(),
            minimum_corpus_logical_bytes: 62_000_000_000,
            report: decoded.clone(),
            github_measurement_digest: ContentDigest::sha256(b"github-api-measurements"),
            measured_on_live_github: true,
            finite_scope_statement: "finite measured live GitHub large-corpus acceptance; no mathematical assurance promotion".to_owned(),
            evidence_digest: ContentDigest("0".repeat(64)),
        };
        live.refresh_evidence_digest().unwrap();
        verify_live_github_acceptance(&LiveGitHubAcceptanceArtifact::LargeCorpus(Box::new(
            live.clone(),
        )))
        .unwrap();
        live.measured_on_live_github = false;
        live.refresh_evidence_digest().unwrap();
        assert!(live.validate().is_err());

        let mut tampered = decoded;
        tampered.publication.remote_receipt_verified = false;
        assert!(verify_large_corpus_acceptance_report(
            &tampered,
            &topology(),
            &observations,
            62_000_000_000,
        )
        .is_err());
    }

    #[test]
    fn large_corpus_report_rejects_oversized_batch_and_fake_no_clone_claim() {
        let observations = vec![
            observation(
                "restricted-shard-001",
                62_000_000_000,
                58_000_000_000,
                98_000_000_000,
                100_000_000,
                100_000_000,
            ),
            observation(
                "restricted-shard-002",
                5_000_000_000,
                4_000_000_000,
                60_000_000_000,
                1_000_000_000,
                1_000_000_000,
            ),
        ];
        let mut retrieval = SelectiveRetrievalObservation {
            shard_id: "restricted-shard-001".to_owned(),
            semantic_digest: ContentDigest::sha256(b"selected-configuration"),
            manifest_digest: ContentDigest::sha256(b"selected-manifest"),
            transport_digest: ContentDigest::sha256(b"selected-transport"),
            receipt_digest: ContentDigest::sha256(b"selected-receipt"),
            selected_logical_payload_bytes: 750_000_000,
            metadata_bytes_read: 2_000_000,
            payload_bytes_downloaded: 740_000_000,
            reused_verified_part_bytes: 10_000_000,
            peak_local_bytes: 1_500_000_000,
            persistent_full_clone_bytes: 1,
            decoded_payload_verified: true,
        };
        let mut publication = LargeCorpusPublicationObservation {
            shard_id: "restricted-shard-002".to_owned(),
            semantic_digest: ContentDigest::sha256(b"new-configuration"),
            receipt_digest: ContentDigest::sha256(b"new-receipt"),
            newly_committed_payload_bytes: 1_800_000_000,
            newly_committed_metadata_bytes: 10_000_000,
            projected_history_bytes: 20_000_000,
            batch_payload_bytes: vec![1_000_000_001, 799_999_999],
            maximum_part_bytes: 90 * 1024 * 1024,
            accounted_before_bytes: 62_000_000_000,
            accounted_after_bytes: 63_830_000_000,
            persistent_full_clone_bytes: 0,
            remote_payload_verified: true,
            remote_receipt_verified: true,
        };
        assert!(evaluate_large_corpus_acceptance(
            &topology(),
            "large-corpus",
            CacheVisibility::Private,
            &observations,
            &retrieval,
            &publication,
            62_000_000_000,
        )
        .is_err());

        retrieval.persistent_full_clone_bytes = 0;
        publication.batch_payload_bytes = vec![1_000_000_000, 800_000_000];
        publication.persistent_full_clone_bytes = 1;
        assert!(evaluate_large_corpus_acceptance(
            &topology(),
            "large-corpus",
            CacheVisibility::Private,
            &observations,
            &retrieval,
            &publication,
            62_000_000_000,
        )
        .is_err());
    }
}
