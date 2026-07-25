//! Family-only topology and shard-local derived indexes/capacity ledgers.

use crate::{
    canonical_digest, ArtifactAssuranceState, ArtifactDisposition, CacheError, CacheVisibility,
    ContentDigest, SemanticKeyEnvelope, ToolkitVersion, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND: &str = "ccm_weil_eigenpair";

/// Compatibility coordinates for locating a lower-N CCM eigenstate without
/// enumerating hash-partitioned manifests.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmEigenpairContinuationQuery {
    pub lambda_squared: String,
    pub maximum_n_modes: usize,
    pub precision_bits: u32,
    pub force_even: bool,
}

impl CcmEigenpairContinuationQuery {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.lambda_squared.trim().is_empty()
            || self.maximum_n_modes == 0
            || self.precision_bits == 0
        {
            return Err(CacheError::InvalidManifest(
                "CCM eigenpair continuation query is incomplete".to_owned(),
            ));
        }
        Ok(())
    }

    fn index_identity(&self) -> CcmEigenpairContinuationIndexIdentity {
        CcmEigenpairContinuationIndexIdentity {
            schema_version: 1,
            lambda_squared: self.lambda_squared.clone(),
            precision_bits: self.precision_bits,
            force_even: self.force_even,
        }
    }

    pub fn repository_path(&self) -> Result<String, CacheError> {
        self.validate()?;
        let digest = canonical_digest(&self.index_identity())?;
        Ok(format!("inventories/ccm-weil-eigenpair/{}.json", digest.0))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CcmEigenpairContinuationIndexIdentity {
    schema_version: u32,
    lambda_squared: String,
    precision_bits: u32,
    force_even: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmEigenpairContinuationEntry {
    pub n_modes: usize,
    pub eigenstate_route: String,
    pub logical_key: String,
    pub semantic_digest: ContentDigest,
    pub manifest_digest: ContentDigest,
    pub achieved_assurance: ArtifactAssuranceState,
    pub disposition: ArtifactDisposition,
    pub producer_toolkit_version: ToolkitVersion,
    pub minimum_reader_version: ToolkitVersion,
}

impl CcmEigenpairContinuationEntry {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.n_modes == 0
            || !matches!(
                self.eigenstate_route.as_str(),
                "legacy_inverse_iteration" | "shift_invert_krylov"
            )
            || self.logical_key.trim().is_empty()
            || !self.semantic_digest.validate()
            || !self.manifest_digest.validate()
        {
            return Err(CacheError::InvalidManifest(
                "CCM eigenpair continuation entry is invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Small derived index stored once per (lambda-squared, precision, sector).
/// It is mutable shard metadata; immutable artifact identities remain in the
/// canonical manifest and primary semantic-digest index.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmEigenpairContinuationIndex {
    pub schema_version: u32,
    pub lambda_squared: String,
    pub precision_bits: u32,
    pub force_even: bool,
    pub entries: Vec<CcmEigenpairContinuationEntry>,
}

impl CcmEigenpairContinuationIndex {
    pub fn rebuild(
        query: &CcmEigenpairContinuationQuery,
        mut entries: Vec<CcmEigenpairContinuationEntry>,
    ) -> Result<Self, CacheError> {
        query.validate()?;
        entries.sort_by(|left, right| {
            left.n_modes
                .cmp(&right.n_modes)
                .then_with(|| left.eigenstate_route.cmp(&right.eigenstate_route))
                .then_with(|| left.semantic_digest.cmp(&right.semantic_digest))
        });
        let index = Self {
            schema_version: 1,
            lambda_squared: query.lambda_squared.clone(),
            precision_bits: query.precision_bits,
            force_even: query.force_even,
            entries,
        };
        index.validate()?;
        Ok(index)
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        let query = CcmEigenpairContinuationQuery {
            lambda_squared: self.lambda_squared.clone(),
            maximum_n_modes: usize::MAX,
            precision_bits: self.precision_bits,
            force_even: self.force_even,
        };
        if self.schema_version != 1 {
            return Err(CacheError::InvalidManifest(
                "unsupported CCM eigenpair continuation index schema".to_owned(),
            ));
        }
        query.validate()?;
        let mut previous = None;
        for entry in &self.entries {
            entry.validate()?;
            let identity = (
                entry.n_modes,
                entry.eigenstate_route.as_str(),
                &entry.semantic_digest,
            );
            if previous.is_some_and(|previous| previous >= identity) {
                return Err(CacheError::InvalidManifest(
                    "CCM eigenpair continuation entries are duplicated or unordered".to_owned(),
                ));
            }
            previous = Some(identity);
        }
        Ok(())
    }

    pub fn query(
        &self,
        query: &CcmEigenpairContinuationQuery,
        current_toolkit_version: &ToolkitVersion,
        maximum_keys: usize,
    ) -> Result<Vec<crate::ArtifactKey>, CacheError> {
        self.validate()?;
        query.validate()?;
        if self.lambda_squared != query.lambda_squared
            || self.precision_bits != query.precision_bits
            || self.force_even != query.force_even
        {
            return Err(CacheError::InvalidManifest(
                "CCM eigenpair continuation index has the wrong compatibility identity".to_owned(),
            ));
        }
        let mut entries = self
            .entries
            .iter()
            .filter(|entry| {
                entry.n_modes < query.maximum_n_modes
                    && entry.disposition == ArtifactDisposition::Active
                    && entry.achieved_assurance.mathematical().is_some()
                    && &entry.minimum_reader_version <= current_toolkit_version
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            right
                .n_modes
                .cmp(&left.n_modes)
                .then_with(|| left.eigenstate_route.cmp(&right.eigenstate_route))
                .then_with(|| left.semantic_digest.cmp(&right.semantic_digest))
        });
        entries.truncate(maximum_keys);
        Ok(entries
            .into_iter()
            .map(|entry| crate::ArtifactKey {
                kind: CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND.to_owned(),
                logical_key: entry.logical_key.clone(),
                parameters_digest: entry.semantic_digest.clone(),
            })
            .collect())
    }
}

pub fn ccm_eigenpair_continuation_entry(
    semantic_key: &SemanticKeyEnvelope,
    logical_key: &str,
    manifest_digest: ContentDigest,
    achieved_assurance: ArtifactAssuranceState,
    disposition: ArtifactDisposition,
    producer_toolkit_version: ToolkitVersion,
    minimum_reader_version: ToolkitVersion,
) -> Result<Option<(CcmEigenpairContinuationQuery, CcmEigenpairContinuationEntry)>, CacheError> {
    if semantic_key.artifact_kind != CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND {
        return Ok(None);
    }
    semantic_key.validate()?;
    let parameters = semantic_key
        .resolved_mathematical_parameters
        .as_object()
        .ok_or_else(|| {
            CacheError::InvalidManifest(
                "CCM eigenpair semantic parameters must be an object".to_owned(),
            )
        })?;
    let lambda_squared = parameters
        .get("lambda_squared")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            CacheError::InvalidManifest(
                "CCM eigenpair semantic key lacks lambda-squared".to_owned(),
            )
        })?
        .to_owned();
    let n_modes = parameters
        .get("n_modes")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            CacheError::InvalidManifest("CCM eigenpair semantic key lacks N".to_owned())
        })?;
    let precision_bits = parameters
        .get("precision_bits")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            CacheError::InvalidManifest("CCM eigenpair semantic key lacks precision".to_owned())
        })?;
    let force_even = parameters
        .get("force_even")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            CacheError::InvalidManifest(
                "CCM eigenpair semantic key lacks sector identity".to_owned(),
            )
        })?;
    let eigenstate_route = parameters
        .get("eigenstate_route")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("legacy_inverse_iteration")
        .to_owned();
    let semantic_digest = semantic_key.digest()?;
    let query = CcmEigenpairContinuationQuery {
        lambda_squared,
        maximum_n_modes: n_modes.saturating_add(1),
        precision_bits,
        force_even,
    };
    let entry = CcmEigenpairContinuationEntry {
        n_modes,
        eigenstate_route,
        logical_key: logical_key.to_owned(),
        semantic_digest,
        manifest_digest,
        achieved_assurance,
        disposition,
        producer_toolkit_version,
        minimum_reader_version,
    };
    query.validate()?;
    entry.validate()?;
    Ok(Some((query, entry)))
}

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

    #[test]
    fn continuation_inventory_returns_nearest_compatible_lower_n() {
        let query = CcmEigenpairContinuationQuery {
            lambda_squared: "13".to_owned(),
            maximum_n_modes: 30,
            precision_bits: 729,
            force_even: true,
        };
        let make_entry = |n_modes: usize, route: &str| CcmEigenpairContinuationEntry {
            n_modes,
            eigenstate_route: route.to_owned(),
            logical_key: format!("ccm/weil-eigenpair/13/{n_modes}/729/even/{route}"),
            semantic_digest: ContentDigest::sha256(
                format!("semantic-{n_modes}-{route}").as_bytes(),
            ),
            manifest_digest: ContentDigest::sha256(
                format!("manifest-{n_modes}-{route}").as_bytes(),
            ),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
            producer_toolkit_version: ToolkitVersion::parse("0.13.2").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
        };
        let index = CcmEigenpairContinuationIndex::rebuild(
            &query,
            vec![
                make_entry(10, "shift_invert_krylov"),
                make_entry(20, "shift_invert_krylov"),
                make_entry(30, "shift_invert_krylov"),
            ],
        )
        .unwrap();
        let keys = index
            .query(&query, &ToolkitVersion::parse("0.13.2").unwrap(), 8)
            .unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys[0].logical_key.contains("/20/"));
        assert!(keys[1].logical_key.contains("/10/"));
        assert_eq!(
            query.repository_path().unwrap(),
            CcmEigenpairContinuationQuery {
                maximum_n_modes: 999,
                ..query
            }
            .repository_path()
            .unwrap()
        );
    }

    #[test]
    fn continuation_entry_is_derived_without_reopening_a_manifest() {
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: CCM_EIGENPAIR_CONTINUATION_ARTIFACT_KIND.to_owned(),
            mathematical_semantics_version: "fixture".to_owned(),
            resolved_mathematical_parameters: serde_json::json!({
                "lambda_squared": "13",
                "n_modes": 20,
                "precision_bits": 729,
                "force_even": true,
                "eigenstate_route": "shift_invert_krylov"
            }),
            normalization: Some("fixture".to_owned()),
            target: Some("smallest_weil_form_eigenpair".to_owned()),
            subspace: Some("even".to_owned()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some("fixture".to_owned()),
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let (query, entry) = ccm_eigenpair_continuation_entry(
            &semantic_key,
            "ccm/weil-eigenpair/13/20/729/even/shift_invert_krylov",
            ContentDigest::sha256(b"manifest"),
            ArtifactAssuranceState::Computed,
            ArtifactDisposition::Active,
            ToolkitVersion::parse("0.13.2").unwrap(),
            ToolkitVersion::parse("0.13.0").unwrap(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(query.lambda_squared, "13");
        assert_eq!(query.precision_bits, 729);
        assert_eq!(entry.n_modes, 20);
        assert_eq!(entry.semantic_digest, semantic_digest);
    }
}
