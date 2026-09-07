//! Local metadata preflight; no remote access or payload writes.
use serde::Deserialize;
use xc_cache::{audit_artifact_kind_registration, CacheVisibility};

#[derive(Deserialize)]
struct Family {
    schema_version: u32,
    family: String,
    visibility: CacheVisibility,
    artifact_kinds: Vec<String>,
    current_writable_shard: String,
    active_writable_shard: Option<String>,
}
#[derive(Deserialize)]
struct Shard {
    schema_version: u32,
    family: String,
    visibility: CacheVisibility,
    artifact_kinds: Vec<String>,
    repository: String,
    default_branch: String,
    immutable_objects: bool,
    writable: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: audit_kind_registration FAMILY.json CACHE-REPOSITORY.json".into());
    }
    let family: Family = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let shard: Shard = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let active = family
        .active_writable_shard
        .as_ref()
        .unwrap_or(&family.current_writable_shard);
    if family.schema_version != 1
        || shard.schema_version != 1
        || family.family != shard.family
        || family.visibility != shard.visibility
        || active != &shard.repository
        || shard.default_branch != "main"
        || !shard.immutable_objects
        || !shard.writable
    {
        return Err("family and active writable shard identities do not agree".into());
    }
    let report = audit_artifact_kind_registration(
        &family.family,
        family.visibility,
        &family.artifact_kinds,
        &shard.artifact_kinds,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.is_ready() {
        std::process::exit(1);
    }
    Ok(())
}
