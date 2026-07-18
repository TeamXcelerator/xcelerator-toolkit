use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use xc_cache::{
    load_queued_produced_artifact, stage_produced_artifact_with_dependencies,
    CanonicalProductionDraft, ProducedArtifactRecord, TransportPolicy,
};
use xc_core::{CancellationToken, ResourcePolicy};

fn record_paths(root: &Path, output: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            record_paths(&path, output)?;
        } else if path.file_name().is_some_and(|name| name == "record.json") {
            output.push(path);
        }
    }
    Ok(())
}

fn load_record(path: &Path) -> Result<ProducedArtifactRecord, Box<dyn Error>> {
    match load_queued_produced_artifact(path) {
        Ok(record) => Ok(record),
        Err(_) => Ok(serde_json::from_slice(&fs::read(path)?)?),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() != 2 {
        return Err("usage: stage_production_queue QUEUE_ROOT STAGING_ROOT".into());
    }
    let queue_root = PathBuf::from(&arguments[0]);
    let staging_root = PathBuf::from(&arguments[1]);
    if !queue_root.is_dir() || staging_root.exists() {
        return Err("queue root must exist and staging root must be absent".into());
    }
    let mut paths = Vec::new();
    record_paths(&queue_root, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err("production queue contains no records".into());
    }
    let cancellation = CancellationToken::new();
    let mut pending = paths
        .iter()
        .map(|path| load_record(path))
        .collect::<Result<Vec<_>, _>>()?;
    let mut drafts = Vec::<CanonicalProductionDraft>::new();
    while !pending.is_empty() {
        let ready = pending.iter().position(|record| {
            record.manifest.dependencies.iter().all(|dependency| {
                drafts.iter().any(|draft| {
                    draft.source_artifact_key == dependency.key
                        && draft.source_content_digest == dependency.content_digest
                })
            })
        });
        let Some(index) = ready else {
            return Err("production queue has a missing or cyclic dependency".into());
        };
        let record = pending.remove(index);
        drafts.push(stage_produced_artifact_with_dependencies(
            &record,
            &drafts,
            &staging_root,
            &TransportPolicy::default(),
            &ResourcePolicy::default(),
            &cancellation,
        )?);
    }
    println!("{}", serde_json::to_string_pretty(&drafts)?);
    Ok(())
}
