//! Privacy-aware, backend-local deduplication planning.

use crate::{CacheError, CacheVisibility, ContentDigest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DedupObject {
    pub digest: ContentDigest,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DedupBackendInventory {
    pub backend_id: String,
    pub repository_id: String,
    pub visibility: CacheVisibility,
    pub backend_local_reuse_supported: bool,
    pub objects: BTreeMap<ContentDigest, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalPayloadObservation {
    pub canonical_payload_digest: ContentDigest,
    pub repository_id: String,
    pub visibility: CacheVisibility,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeduplicationPlanningRequest {
    pub schema_version: u32,
    pub operation_visibility: CacheVisibility,
    pub canonical_payload_digest: ContentDigest,
    pub objects: Vec<DedupObject>,
    pub target: DedupBackendInventory,
    /// Inventories already authorized and observed by the caller. Planning
    /// performs no discovery or cross-repository probing.
    pub observed_other_repositories: Vec<DedupBackendInventory>,
    pub logical_payload_observations: Vec<LogicalPayloadObservation>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeduplicationPlan {
    pub schema_version: u32,
    pub canonical_payload_digest: ContentDigest,
    pub target_backend_id: String,
    pub target_repository_id: String,
    pub logical_payload_reuse_observed: bool,
    pub logical_reuse_repository_count: usize,
    pub backend_local_physical_reuse_objects: usize,
    pub backend_local_physical_reuse_bytes: u64,
    pub cross_repository_copy_objects: usize,
    pub cross_repository_copy_bytes: u64,
    pub new_upload_objects: usize,
    pub new_upload_bytes: u64,
    pub physical_deduplication_claim_limited_to_target_repository: bool,
    pub private_store_probes_performed: u64,
}

pub fn plan_deduplication(
    request: &DeduplicationPlanningRequest,
) -> Result<DeduplicationPlan, CacheError> {
    validate_request(request)?;
    let mut local_objects = 0usize;
    let mut local_bytes = 0u64;
    let mut copied_objects = 0usize;
    let mut copied_bytes = 0u64;
    let mut uploaded_objects = 0usize;
    let mut uploaded_bytes = 0u64;
    for object in &request.objects {
        if request.target.backend_local_reuse_supported
            && request.target.objects.get(&object.digest) == Some(&object.size_bytes)
        {
            local_objects += 1;
            local_bytes = checked_add(local_bytes, object.size_bytes)?;
        } else if request
            .observed_other_repositories
            .iter()
            .any(|inventory| inventory.objects.get(&object.digest) == Some(&object.size_bytes))
        {
            copied_objects += 1;
            copied_bytes = checked_add(copied_bytes, object.size_bytes)?;
        } else {
            uploaded_objects += 1;
            uploaded_bytes = checked_add(uploaded_bytes, object.size_bytes)?;
        }
    }
    let logical_repositories = request
        .logical_payload_observations
        .iter()
        .filter(|observation| {
            observation.canonical_payload_digest == request.canonical_payload_digest
        })
        .map(|observation| observation.repository_id.as_str())
        .collect::<BTreeSet<_>>();
    Ok(DeduplicationPlan {
        schema_version: 1,
        canonical_payload_digest: request.canonical_payload_digest.clone(),
        target_backend_id: request.target.backend_id.clone(),
        target_repository_id: request.target.repository_id.clone(),
        logical_payload_reuse_observed: !logical_repositories.is_empty(),
        logical_reuse_repository_count: logical_repositories.len(),
        backend_local_physical_reuse_objects: local_objects,
        backend_local_physical_reuse_bytes: local_bytes,
        cross_repository_copy_objects: copied_objects,
        cross_repository_copy_bytes: copied_bytes,
        new_upload_objects: uploaded_objects,
        new_upload_bytes: uploaded_bytes,
        physical_deduplication_claim_limited_to_target_repository: true,
        private_store_probes_performed: 0,
    })
}

fn validate_request(request: &DeduplicationPlanningRequest) -> Result<(), CacheError> {
    if request.schema_version != 1
        || !request.canonical_payload_digest.validate()
        || request.objects.is_empty()
        || request.target.backend_id.trim().is_empty()
        || request.target.repository_id.trim().is_empty()
    {
        return Err(CacheError::InvalidManifest(
            "deduplication planning identity is incomplete".to_owned(),
        ));
    }
    if request.operation_visibility == CacheVisibility::Public
        && (request.target.visibility != CacheVisibility::Public
            || request
                .observed_other_repositories
                .iter()
                .any(|inventory| inventory.visibility != CacheVisibility::Public)
            || request
                .logical_payload_observations
                .iter()
                .any(|observation| observation.visibility != CacheVisibility::Public))
    {
        return Err(CacheError::PermissionDenied(
            "public deduplication planning cannot receive private or local observations".to_owned(),
        ));
    }
    let mut object_digests = BTreeSet::new();
    for object in &request.objects {
        if !object.digest.validate()
            || object.size_bytes == 0
            || !object_digests.insert(object.digest.clone())
        {
            return Err(CacheError::InvalidManifest(
                "deduplication objects must have unique digests and positive sizes".to_owned(),
            ));
        }
    }
    let mut repositories = BTreeSet::from([request.target.repository_id.as_str()]);
    for inventory in &request.observed_other_repositories {
        if inventory.backend_id.trim().is_empty()
            || inventory.repository_id.trim().is_empty()
            || !repositories.insert(inventory.repository_id.as_str())
            || inventory
                .objects
                .iter()
                .any(|(digest, size)| !digest.validate() || *size == 0)
        {
            return Err(CacheError::InvalidManifest(
                "deduplication repository inventories are invalid or duplicated".to_owned(),
            ));
        }
    }
    if request
        .target
        .objects
        .iter()
        .any(|(digest, size)| !digest.validate() || *size == 0)
        || request
            .logical_payload_observations
            .iter()
            .any(|observation| {
                !observation.canonical_payload_digest.validate()
                    || observation.repository_id.trim().is_empty()
            })
    {
        return Err(CacheError::InvalidManifest(
            "deduplication observations contain an invalid identity".to_owned(),
        ));
    }
    Ok(())
}

fn checked_add(total: u64, value: u64) -> Result<u64, CacheError> {
    total.checked_add(value).ok_or_else(|| {
        CacheError::ResourceLimit("deduplication byte accounting exceeds u64".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory(repository: &str, visibility: CacheVisibility) -> DedupBackendInventory {
        DedupBackendInventory {
            backend_id: "github_git".to_owned(),
            repository_id: repository.to_owned(),
            visibility,
            backend_local_reuse_supported: true,
            objects: BTreeMap::new(),
        }
    }

    #[test]
    fn report_separates_logical_local_copy_and_upload_reuse() {
        let local = ContentDigest::sha256(b"local");
        let copied = ContentDigest::sha256(b"copied");
        let uploaded = ContentDigest::sha256(b"uploaded");
        let payload = ContentDigest::sha256(b"payload");
        let mut target = inventory("team/target", CacheVisibility::Private);
        target.objects.insert(local.clone(), 10);
        let mut other = inventory("team/other", CacheVisibility::Private);
        other.objects.insert(copied.clone(), 20);
        let plan = plan_deduplication(&DeduplicationPlanningRequest {
            schema_version: 1,
            operation_visibility: CacheVisibility::Private,
            canonical_payload_digest: payload.clone(),
            objects: vec![
                DedupObject {
                    digest: local,
                    size_bytes: 10,
                },
                DedupObject {
                    digest: copied,
                    size_bytes: 20,
                },
                DedupObject {
                    digest: uploaded,
                    size_bytes: 30,
                },
            ],
            target,
            observed_other_repositories: vec![other],
            logical_payload_observations: vec![LogicalPayloadObservation {
                canonical_payload_digest: payload,
                repository_id: "project/third".to_owned(),
                visibility: CacheVisibility::Private,
            }],
        })
        .unwrap();
        assert!(plan.logical_payload_reuse_observed);
        assert_eq!(plan.backend_local_physical_reuse_bytes, 10);
        assert_eq!(plan.cross_repository_copy_bytes, 20);
        assert_eq!(plan.new_upload_bytes, 30);
        assert!(plan.physical_deduplication_claim_limited_to_target_repository);
    }

    #[test]
    fn public_plan_rejects_private_existence_observations() {
        let object = DedupObject {
            digest: ContentDigest::sha256(b"object"),
            size_bytes: 10,
        };
        let result = plan_deduplication(&DeduplicationPlanningRequest {
            schema_version: 1,
            operation_visibility: CacheVisibility::Public,
            canonical_payload_digest: ContentDigest::sha256(b"payload"),
            objects: vec![object],
            target: inventory("team/public", CacheVisibility::Public),
            observed_other_repositories: vec![inventory("team/private", CacheVisibility::Private)],
            logical_payload_observations: Vec::new(),
        });
        assert!(matches!(result, Err(CacheError::PermissionDenied(_))));
    }
}
