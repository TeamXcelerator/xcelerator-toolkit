//! Family-only topology and shard-local derived indexes/capacity ledgers.

use crate::{
    canonical_digest, ArtifactAssuranceState, ArtifactDisposition, CacheError, CacheVisibility,
    ContentDigest, ToolkitVersion, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyShardStatus {
    Writable,
    ReadOnly,
    Sealed,
    IncidentHold,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyShardRoute {
    pub shard_id: String,
    /// Resolves through the separately managed network endpoint registry.
    pub endpoint_id: String,
    pub sequence: u32,
    pub status: TopologyShardStatus,
    pub successor_shard_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFamilyRoute {
    pub family: String,
    pub visibility: CacheVisibility,
    pub ordered_shards: Vec<TopologyShardRoute>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyRegistry {
    pub schema_version: u32,
    pub generation: u64,
    pub previous_registry_digest: Option<ContentDigest>,
    pub policy_digest: ContentDigest,
    pub trust_anchor_ids: Vec<String>,
    pub family_routes: Vec<ArtifactFamilyRoute>,
}

impl TopologyRegistry {
    pub fn digest(&self) -> Result<ContentDigest, CacheError> {
        self.validate()?;
        canonical_digest(self)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0 || !self.policy_digest.validate() {
            return Err(CacheError::InvalidManifest(
                "topology registry requires a schema and policy digest".to_owned(),
            ));
        }
        if self.generation > 0
            && self
                .previous_registry_digest
                .as_ref()
                .is_some_and(|digest| !digest.validate())
        {
            return Err(CacheError::InvalidManifest(
                "topology registry has an invalid predecessor digest".to_owned(),
            ));
        }
        if self
            .trust_anchor_ids
            .iter()
            .any(|anchor| anchor.trim().is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "topology trust anchor identifiers must be nonempty".to_owned(),
            ));
        }
        let mut route_keys = BTreeSet::new();
        let mut global_shards = BTreeMap::<&str, (&str, CacheVisibility)>::new();
        for route in &self.family_routes {
            if route.family.trim().is_empty()
                || route.ordered_shards.is_empty()
                || !route_keys.insert((route.family.as_str(), route.visibility))
            {
                return Err(CacheError::InvalidManifest(
                    "topology family routes must be nonempty and unique by family/visibility"
                        .to_owned(),
                ));
            }
            let mut sequences = BTreeSet::new();
            let mut shard_ids = BTreeSet::new();
            for shard in &route.ordered_shards {
                if shard.shard_id.trim().is_empty()
                    || shard.endpoint_id.trim().is_empty()
                    || !sequences.insert(shard.sequence)
                    || !shard_ids.insert(shard.shard_id.as_str())
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "route {:?} contains an invalid or duplicate shard",
                        route.family
                    )));
                }
                if let Some((family, visibility)) =
                    global_shards.insert(&shard.shard_id, (&route.family, route.visibility))
                {
                    if family != route.family || visibility != route.visibility {
                        return Err(CacheError::InvalidManifest(format!(
                            "shard {:?} is assigned to multiple family routes",
                            shard.shard_id
                        )));
                    }
                }
            }
            for shard in &route.ordered_shards {
                if shard
                    .successor_shard_id
                    .as_ref()
                    .is_some_and(|successor| !shard_ids.contains(successor.as_str()))
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "successor for shard {:?} is outside its family route",
                        shard.shard_id
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn route(&self, family: &str, visibility: CacheVisibility) -> Option<&ArtifactFamilyRoute> {
        self.family_routes
            .iter()
            .find(|route| route.family == family && route.visibility == visibility)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyTrustPolicy {
    pub minimum_generation: u64,
    pub pinned_registry_digest: Option<ContentDigest>,
    pub required_trust_anchor: Option<String>,
}

impl TopologyTrustPolicy {
    pub fn verify(&self, registry: &TopologyRegistry) -> Result<ContentDigest, CacheError> {
        let digest = registry.digest()?;
        if registry.generation < self.minimum_generation {
            return Err(CacheError::InvalidManifest(format!(
                "topology generation {} is below trusted minimum {}",
                registry.generation, self.minimum_generation
            )));
        }
        if self
            .pinned_registry_digest
            .as_ref()
            .is_some_and(|pinned| pinned != &digest)
        {
            return Err(CacheError::DigestMismatch {
                expected: self
                    .pinned_registry_digest
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
                actual: digest.to_string(),
            });
        }
        if self.required_trust_anchor.as_ref().is_some_and(|required| {
            !registry
                .trust_anchor_ids
                .iter()
                .any(|anchor| anchor == required)
        }) {
            return Err(CacheError::InvalidManifest(
                "topology registry does not carry the required trust anchor".to_owned(),
            ));
        }
        Ok(digest)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardIndexEntry {
    pub semantic_digest: ContentDigest,
    pub canonical_payload_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
    pub transport_digests: Vec<ContentDigest>,
    /// Stable transaction whose atomic discoverability commit introduced this
    /// index entry and its receipt.
    pub publication_transaction_id: String,
}

impl ShardIndexEntry {
    pub fn validate(&self) -> Result<(), CacheError> {
        if [
            &self.semantic_digest,
            &self.canonical_payload_digest,
            &self.manifest_digest,
        ]
        .into_iter()
        .chain(self.transport_digests.iter())
        .any(|digest| !digest.validate())
            || self.publication_transaction_id.len() != 64
            || !self
                .publication_transaction_id
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CacheError::InvalidManifest(
                "shard index entry contains an invalid digest".to_owned(),
            ));
        }
        if self.transport_digests.is_empty()
            || self
                .transport_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(CacheError::InvalidManifest(
                "shard index transport identities are empty, duplicated, or unordered".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevocationScope {
    Semantic,
    Payload,
    Manifest,
    Transport,
    Attestation,
    Repository,
    Policy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevocationRecord {
    pub schema_version: u32,
    pub scope: RevocationScope,
    pub identity_digest: ContentDigest,
    pub reason: String,
    pub effective_unix_seconds: u64,
    pub replacement_digest: Option<ContentDigest>,
    pub incident_reference: Option<String>,
    pub authorizing_evidence_digest: ContentDigest,
}

impl RevocationRecord {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || !self.identity_digest.validate()
            || self.reason.trim().is_empty()
            || self.reason.len() > 4_096
            || self
                .replacement_digest
                .as_ref()
                .is_some_and(|digest| !digest.validate())
            || !self.authorizing_evidence_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "revocation record identity or evidence is invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevocationIndexPartition {
    pub schema_version: u32,
    pub identity_prefix: String,
    pub records: Vec<RevocationRecord>,
}

impl RevocationIndexPartition {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.identity_prefix.len() != 2
            || !self
                .identity_prefix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CacheError::InvalidManifest(
                "revocation partition identity is invalid".to_owned(),
            ));
        }
        let mut previous: Option<(RevocationScope, &ContentDigest)> = None;
        for record in &self.records {
            record.validate()?;
            if !record.identity_digest.0.starts_with(&self.identity_prefix) {
                return Err(CacheError::InvalidManifest(
                    "revocation is stored in the wrong identity partition".to_owned(),
                ));
            }
            let current = (record.scope, &record.identity_digest);
            if previous.is_some_and(|previous| previous >= current) {
                return Err(CacheError::InvalidManifest(
                    "revocation partition is duplicated or not canonically ordered".to_owned(),
                ));
            }
            previous = Some(current);
        }
        Ok(())
    }

    pub fn active(
        &self,
        scope: RevocationScope,
        identity: &ContentDigest,
        evaluation_unix_seconds: u64,
    ) -> Option<&RevocationRecord> {
        self.records.iter().find(|record| {
            record.scope == scope
                && &record.identity_digest == identity
                && record.effective_unix_seconds <= evaluation_unix_seconds
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardIndexPartition {
    pub schema_version: u32,
    pub family: String,
    pub semantic_prefix: String,
    pub entries: Vec<ShardIndexEntry>,
}

impl ShardIndexPartition {
    pub fn rebuild(
        family: impl Into<String>,
        semantic_prefix: impl Into<String>,
        mut entries: Vec<ShardIndexEntry>,
    ) -> Result<Self, CacheError> {
        let family = family.into();
        let semantic_prefix = semantic_prefix.into();
        entries.sort_by(|left, right| {
            (&left.semantic_digest, &left.manifest_digest)
                .cmp(&(&right.semantic_digest, &right.manifest_digest))
        });
        let partition = Self {
            schema_version: 1,
            family,
            semantic_prefix,
            entries,
        };
        partition.validate()?;
        Ok(partition)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.family.trim().is_empty()
            || self.semantic_prefix.is_empty()
            || self.semantic_prefix.len() > 64
            || !self
                .semantic_prefix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CacheError::InvalidManifest(
                "shard index partition identity is invalid".to_owned(),
            ));
        }
        let mut manifest_ids = BTreeSet::new();
        let mut previous: Option<(&ContentDigest, &ContentDigest)> = None;
        for entry in &self.entries {
            entry.validate()?;
            if !entry.semantic_digest.0.starts_with(&self.semantic_prefix)
                || !manifest_ids.insert(&entry.manifest_digest)
            {
                return Err(CacheError::InvalidManifest(
                    "shard index entry is in the wrong partition or duplicated".to_owned(),
                ));
            }
            let current = (&entry.semantic_digest, &entry.manifest_digest);
            if previous.is_some_and(|previous| previous > current) {
                return Err(CacheError::InvalidManifest(
                    "shard index entries are not canonically ordered".to_owned(),
                ));
            }
            previous = Some(current);
        }
        Ok(())
    }

    pub fn lookup<'a>(
        &'a self,
        semantic_digest: &'a ContentDigest,
    ) -> impl Iterator<Item = &'a ShardIndexEntry> + 'a {
        self.entries
            .iter()
            .filter(move |entry| &entry.semantic_digest == semantic_digest)
    }

    /// Reject a publication that would make an older producer compete with an
    /// already discoverable newer result for the same semantic identity.
    pub fn ensure_monotonic_producer(
        &self,
        semantic_digest: &ContentDigest,
        incoming: &ToolkitVersion,
    ) -> Result<(), CacheError> {
        if let Some(newer) = self.entries.iter().find(|entry| {
            &entry.semantic_digest == semantic_digest
                && &entry.producer_toolkit_version > incoming
                && !matches!(
                    entry.disposition,
                    ArtifactDisposition::Revoked | ArtifactDisposition::Quarantined
                )
        }) {
            return Err(CacheError::PermissionDenied(format!(
                "publication downgrade rejected: producer toolkit {incoming} cannot supersede active toolkit {} for semantic identity {semantic_digest}",
                newer.producer_toolkit_version
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityLedger {
    pub schema_version: u32,
    pub shard_id: String,
    pub hard_capacity_bytes: u64,
    pub warning_reserve_bytes: u64,
    pub first_seen_immutable_payload_bytes: u64,
    pub manifest_index_receipt_bytes: u64,
    pub estimated_history_bytes: u64,
    pub emergency_reserve_bytes: u64,
    pub abandoned_reachable_bytes: u64,
    pub last_reconciled_commit: String,
    pub reconciliation_digest: ContentDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapacityAdmission {
    pub accepted: bool,
    pub current_accounted_bytes: u64,
    pub projected_accounted_bytes: u64,
    pub hard_capacity_bytes: u64,
    pub remaining_after_bytes: u64,
    pub warning_reserve_reached: bool,
    pub reasons: Vec<String>,
}

impl CapacityLedger {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version == 0
            || self.shard_id.trim().is_empty()
            || self.last_reconciled_commit.trim().is_empty()
            || !self.reconciliation_digest.validate()
            || self.hard_capacity_bytes == 0
            || self.hard_capacity_bytes > GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES
            || self.warning_reserve_bytes > self.hard_capacity_bytes
            || self.emergency_reserve_bytes > self.hard_capacity_bytes
        {
            return Err(CacheError::InvalidManifest(
                "capacity ledger identity or limits are invalid".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn accounted_bytes(&self) -> u64 {
        self.first_seen_immutable_payload_bytes
            .saturating_add(self.manifest_index_receipt_bytes)
            .saturating_add(self.estimated_history_bytes)
            .saturating_add(self.emergency_reserve_bytes)
            .saturating_add(self.abandoned_reachable_bytes)
    }

    pub fn assess_addition(
        &self,
        unique_payload_bytes: u64,
        metadata_bytes: u64,
        projected_history_bytes: u64,
    ) -> Result<CapacityAdmission, CacheError> {
        self.validate()?;
        let current = self.accounted_bytes();
        let projected = current
            .saturating_add(unique_payload_bytes)
            .saturating_add(metadata_bytes)
            .saturating_add(projected_history_bytes);
        let accepted = projected <= self.hard_capacity_bytes;
        let remaining = self.hard_capacity_bytes.saturating_sub(projected);
        Ok(CapacityAdmission {
            accepted,
            current_accounted_bytes: current,
            projected_accounted_bytes: projected,
            hard_capacity_bytes: self.hard_capacity_bytes,
            remaining_after_bytes: remaining,
            warning_reserve_reached: remaining < self.warning_reserve_bytes,
            reasons: if accepted {
                Vec::new()
            } else {
                vec![format!(
                    "projected shard bytes {projected} exceed hard capacity {}",
                    self.hard_capacity_bytes
                )]
            },
        })
    }
}

pub fn select_write_shard_from_local_ledgers<'a>(
    topology: &'a TopologyRegistry,
    ledgers: &'a BTreeMap<String, CapacityLedger>,
    family: &str,
    visibility: CacheVisibility,
    unique_payload_bytes: u64,
    metadata_bytes: u64,
    projected_history_bytes: u64,
) -> Result<(&'a TopologyShardRoute, CapacityAdmission), CacheError> {
    topology.validate()?;
    let route = topology.route(family, visibility).ok_or_else(|| {
        CacheError::NoWritableShard(format!(
            "no topology route for family={family:?}, visibility={visibility:?}"
        ))
    })?;
    let mut accepted = Vec::new();
    for shard in &route.ordered_shards {
        if shard.status != TopologyShardStatus::Writable {
            continue;
        }
        let ledger = ledgers.get(&shard.shard_id).ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "shard {:?} has no fetched capacity ledger",
                shard.shard_id
            ))
        })?;
        if ledger.shard_id != shard.shard_id {
            return Err(CacheError::InvalidManifest(format!(
                "capacity ledger {:?} does not match topology shard {:?}",
                ledger.shard_id, shard.shard_id
            )));
        }
        let admission = ledger.assess_addition(
            unique_payload_bytes,
            metadata_bytes,
            projected_history_bytes,
        )?;
        if admission.accepted {
            accepted.push((shard, admission));
        }
    }
    // Fill the most-used admissible shard first without putting its changing
    // byte counts into the top-level topology registry.
    accepted
        .into_iter()
        .max_by_key(|(_, admission)| admission.projected_accounted_bytes)
        .ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "no shard-local ledger admits {unique_payload_bytes} payload bytes"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> TopologyRegistry {
        TopologyRegistry {
            schema_version: 1,
            generation: 4,
            previous_registry_digest: Some(ContentDigest::sha256(b"generation-3")),
            policy_digest: ContentDigest::sha256(b"topology-policy"),
            trust_anchor_ids: vec!["team-xcelerator-release-key".to_owned()],
            family_routes: vec![ArtifactFamilyRoute {
                family: "ccm".to_owned(),
                visibility: CacheVisibility::Private,
                ordered_shards: vec![
                    TopologyShardRoute {
                        shard_id: "ccm-private-001".to_owned(),
                        endpoint_id: "ccm-private-001".to_owned(),
                        sequence: 1,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: Some("ccm-private-002".to_owned()),
                    },
                    TopologyShardRoute {
                        shard_id: "ccm-private-002".to_owned(),
                        endpoint_id: "ccm-private-002".to_owned(),
                        sequence: 2,
                        status: TopologyShardStatus::Writable,
                        successor_shard_id: None,
                    },
                ],
            }],
        }
    }

    fn ledger(id: &str, payload: u64) -> CapacityLedger {
        CapacityLedger {
            schema_version: 1,
            shard_id: id.to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 5_000_000_000,
            first_seen_immutable_payload_bytes: payload,
            manifest_index_receipt_bytes: 1_000,
            estimated_history_bytes: 2_000,
            emergency_reserve_bytes: 1_000_000_000,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: "abc123".to_owned(),
            reconciliation_digest: ContentDigest::sha256(id.as_bytes()),
        }
    }

    #[test]
    fn topology_contains_family_routes_not_artifact_entries() {
        let encoded = serde_json::to_string(&topology()).unwrap();
        assert!(encoded.contains("family_routes"));
        assert!(!encoded.contains("semantic_digest"));
        assert!(!encoded.contains("object_digest"));
    }

    #[test]
    fn anti_rollback_policy_rejects_old_generation() {
        let policy = TopologyTrustPolicy {
            minimum_generation: 5,
            pinned_registry_digest: None,
            required_trust_anchor: Some("team-xcelerator-release-key".to_owned()),
        };
        assert!(policy.verify(&topology()).is_err());
    }

    #[test]
    fn shard_selection_reads_shard_local_ledgers() {
        let ledgers = BTreeMap::from([
            (
                "ccm-private-001".to_owned(),
                ledger("ccm-private-001", 60_000_000_000),
            ),
            ("ccm-private-002".to_owned(), ledger("ccm-private-002", 0)),
        ]);
        let topology = topology();
        let (selected, admission) = select_write_shard_from_local_ledgers(
            &topology,
            &ledgers,
            "ccm",
            CacheVisibility::Private,
            30_000_000_000,
            1_000,
            1_000,
        )
        .unwrap();
        assert_eq!(selected.shard_id, "ccm-private-001");
        assert!(admission.accepted);
    }

    fn entry(seed: &[u8]) -> ShardIndexEntry {
        let semantic_digest = ContentDigest::sha256(seed);
        ShardIndexEntry {
            semantic_digest,
            canonical_payload_digest: ContentDigest::sha256(&[seed, b"payload"].concat()),
            manifest_digest: ContentDigest::sha256(&[seed, b"manifest"].concat()),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
            transport_digests: vec![ContentDigest::sha256(&[seed, b"transport"].concat())],
            publication_transaction_id: ContentDigest::sha256(&[seed, b"transaction"].concat()).0,
        }
    }

    #[test]
    fn shard_index_rebuild_is_partitioned_and_canonically_sorted() {
        let first = entry(b"first");
        let mut second = entry(b"second");
        // Put both fixtures in the same simulated prefix partition.
        second
            .semantic_digest
            .0
            .replace_range(..2, &first.semantic_digest.0[..2]);
        let prefix = first.semantic_digest.0[..2].to_owned();
        let partition = ShardIndexPartition::rebuild("ccm", prefix, vec![second, first]).unwrap();
        assert!(partition.entries.windows(2).all(|pair| {
            (&pair[0].semantic_digest, &pair[0].manifest_digest)
                <= (&pair[1].semantic_digest, &pair[1].manifest_digest)
        }));
    }
}
