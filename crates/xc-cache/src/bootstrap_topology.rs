//! Validation and resolution of the visibility-specific bootstrap registry.
//!
//! The deployed cache uses a compact registry document that points to one
//! family document, which in turn lists every shard descriptor.  Keeping this
//! reader shared prevents the managed publisher and the GitHub cache reader
//! from disagreeing after a shard rollover.

use crate::{CacheError, CacheVisibility, RemoteGitStore, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES};
use serde::Deserialize;
use std::collections::BTreeSet;
use xc_core::CancellationToken;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapRegistry {
    schema_version: u32,
    repository: String,
    default_branch: String,
    families: Vec<BootstrapRegistryFamily>,
    #[serde(default)]
    separate_visibility_inventory: Option<bool>,
    #[serde(default)]
    visibility: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapRegistryFamily {
    family: String,
    current_writable_shard: String,
    metadata: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapFamilyDocument {
    schema_version: u32,
    family: String,
    display_name: String,
    description: String,
    visibility: String,
    artifact_kinds: Vec<String>,
    /// Optional rollover route understood by readers that can search more
    /// than one shard. The legacy field remains pinned to the predecessor so
    /// released single-shard readers can continue to resolve its contents.
    #[serde(default)]
    active_writable_shard: Option<String>,
    current_writable_shard: String,
    shards: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapShardDocument {
    schema_version: u32,
    repository: String,
    family: String,
    visibility: String,
    shard: u32,
    default_branch: String,
    immutable_objects: bool,
    writable: bool,
    capacity_policy: BootstrapShardCapacityPolicy,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapShardCapacityPolicy {
    maximum_reachable_payload_bytes: u64,
    rollover_repository_pattern: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BootstrapShard {
    pub(crate) authorized_repository: String,
    pub(crate) repository_url: String,
    pub(crate) shard_id: String,
    pub(crate) sequence: u32,
    pub(crate) writable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BootstrapFamilyTopology {
    pub(crate) family: String,
    pub(crate) visibility: CacheVisibility,
    pub(crate) current_writable: BootstrapShard,
    /// Current writable shard first, then historical shards newest-to-oldest.
    pub(crate) readable_shards: Vec<BootstrapShard>,
}

fn visibility_name(visibility: CacheVisibility) -> Result<&'static str, CacheError> {
    match visibility {
        CacheVisibility::Private => Ok("private"),
        CacheVisibility::Public => Ok("public"),
        _ => Err(CacheError::InvalidManifest(
            "bootstrap registries are defined only for private or public visibility".to_owned(),
        )),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    path: &str,
    cancellation: &CancellationToken,
) -> Result<T, CacheError> {
    let mut bytes = Vec::new();
    remote.read_committed_path(
        repository,
        revision,
        path,
        4 * 1024 * 1024,
        cancellation,
        &mut bytes,
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) fn read_bootstrap_family_topology(
    remote: &dyn RemoteGitStore,
    owner: &str,
    visibility: CacheVisibility,
    family: &str,
    cancellation: &CancellationToken,
) -> Result<BootstrapFamilyTopology, CacheError> {
    let visibility_name = visibility_name(visibility)?;
    if owner.trim().is_empty()
        || owner.contains('/')
        || family.trim().is_empty()
        || family.contains('/')
    {
        return Err(CacheError::InvalidManifest(
            "bootstrap owner or family is invalid".to_owned(),
        ));
    }
    let registry_id = format!("{owner}/xcelerator-cache-{visibility_name}-registry");
    let registry_repository = format!("https://github.com/{registry_id}.git");
    let registry_revision = remote.read_ref(&registry_repository, "main")?;
    let registry: BootstrapRegistry = read_json(
        remote,
        &registry_repository,
        &registry_revision,
        "registry.json",
        cancellation,
    )?;
    if registry.schema_version != 1
        || registry.repository != registry_id
        || registry.default_branch != "main"
        || registry.families.is_empty()
        || registry
            .separate_visibility_inventory
            .is_some_and(|separate| !separate)
        || registry
            .visibility
            .as_deref()
            .is_some_and(|declared| declared != visibility_name)
    {
        return Err(CacheError::InvalidManifest(format!(
            "{visibility_name} bootstrap registry identity is invalid"
        )));
    }
    let mut registry_families = BTreeSet::new();
    if registry.families.iter().any(|candidate| {
        candidate.family.trim().is_empty()
            || candidate.family.contains('/')
            || candidate.metadata != format!("families/{}.json", candidate.family)
            || !candidate.current_writable_shard.starts_with(&format!(
                "{owner}/xcelerator-cache-{visibility_name}-{}-",
                candidate.family
            ))
            || !registry_families.insert(candidate.family.clone())
    }) {
        return Err(CacheError::InvalidManifest(format!(
            "{visibility_name} bootstrap registry contains an invalid family route"
        )));
    }
    let route = registry
        .families
        .iter()
        .find(|route| route.family == family)
        .ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "{visibility_name} registry has no route for family {family:?}"
            ))
        })?;
    let family_document: BootstrapFamilyDocument = read_json(
        remote,
        &registry_repository,
        &registry_revision,
        &route.metadata,
        cancellation,
    )?;
    if family_document.schema_version != 1
        || family_document.family != family
        || family_document.visibility != visibility_name
        || family_document.current_writable_shard != route.current_writable_shard
        || family_document.display_name.trim().is_empty()
        || family_document.description.trim().is_empty()
        || family_document.artifact_kinds.is_empty()
        || family_document
            .artifact_kinds
            .iter()
            .any(|kind| kind.trim().is_empty())
        || family_document
            .artifact_kinds
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != family_document.artifact_kinds.len()
        || family_document.shards.is_empty()
    {
        return Err(CacheError::InvalidManifest(format!(
            "{visibility_name} family document for {family:?} is invalid"
        )));
    }

    let active_writable_shard = family_document
        .active_writable_shard
        .as_deref()
        .unwrap_or(&family_document.current_writable_shard);
    if !active_writable_shard.starts_with(&format!(
        "{owner}/xcelerator-cache-{visibility_name}-{family}-"
    )) {
        return Err(CacheError::InvalidManifest(format!(
            "{visibility_name} family {family:?} has an invalid active writable shard"
        )));
    }
    let expected_pattern = format!("xcelerator-cache-{visibility_name}-{family}-{{shard:04d}}");
    let mut seen_paths = BTreeSet::new();
    let mut seen_repositories = BTreeSet::new();
    let mut shards = Vec::with_capacity(family_document.shards.len());
    for (index, path) in family_document.shards.iter().enumerate() {
        if !seen_paths.insert(path.clone()) {
            return Err(CacheError::InvalidManifest(format!(
                "{visibility_name} family {family:?} repeats shard metadata {path:?}"
            )));
        }
        let descriptor: BootstrapShardDocument = read_json(
            remote,
            &registry_repository,
            &registry_revision,
            path,
            cancellation,
        )?;
        let sequence = u32::try_from(index + 1).map_err(|_| {
            CacheError::ResourceLimit("bootstrap shard sequence exceeds u32".to_owned())
        })?;
        let repository_name = format!("xcelerator-cache-{visibility_name}-{family}-{sequence:04}");
        let authorized_repository = format!("{owner}/{repository_name}");
        let expected_path = format!("shards/{repository_name}.json");
        let should_be_writable = authorized_repository == active_writable_shard;
        if path != &expected_path
            || descriptor.schema_version != 1
            || descriptor.repository != authorized_repository
            || descriptor.family != family
            || descriptor.visibility != visibility_name
            || descriptor.shard != sequence
            || descriptor.default_branch != "main"
            || !descriptor.immutable_objects
            || descriptor.writable != should_be_writable
            || descriptor.capacity_policy.maximum_reachable_payload_bytes
                != GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES
            || descriptor.capacity_policy.rollover_repository_pattern != expected_pattern
            || !seen_repositories.insert(authorized_repository.clone())
        {
            return Err(CacheError::InvalidManifest(format!(
                "{visibility_name} shard descriptor {path:?} for {family:?} is invalid"
            )));
        }
        shards.push(BootstrapShard {
            repository_url: format!("https://github.com/{authorized_repository}.git"),
            shard_id: format!("{visibility_name}-{family}-{sequence:04}"),
            authorized_repository,
            sequence,
            writable: descriptor.writable,
        });
    }
    let current_writable = shards
        .iter()
        .find(|shard| shard.writable)
        .cloned()
        .ok_or_else(|| {
            CacheError::NoWritableShard(format!(
                "{visibility_name} family {family:?} has no writable shard"
            ))
        })?;
    if shards.iter().filter(|shard| shard.writable).count() != 1
        || current_writable.authorized_repository != active_writable_shard
        || current_writable.sequence as usize != shards.len()
    {
        return Err(CacheError::InvalidManifest(format!(
            "{visibility_name} family {family:?} does not name its final shard as the sole writable shard"
        )));
    }
    let mut readable_shards = vec![current_writable.clone()];
    readable_shards.extend(shards.into_iter().rev().filter(|shard| !shard.writable));
    Ok(BootstrapFamilyTopology {
        family: family.to_owned(),
        visibility,
        current_writable,
        readable_shards,
    })
}
