//! Verify a completed scholarly deposit against Git, local bytes, and provider evidence.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use xc_core::{verify_archive_deposit_evidence, ArchiveDepositReceipt, ScholarlyArchivePlan};

fn collect_paths(
    root: &Path,
    directory: &Path,
    output: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(format!(
                "archive inventory contains symlink {}",
                entry.path().display()
            )
            .into());
        }
        if file_type.is_dir() {
            collect_paths(root, &entry.path(), output)?;
        } else if file_type.is_file() {
            output.push(
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() != 4 {
        return Err(
            "usage: verify_archive_deposit PLAN.json RECEIPT.json INVENTORY_ROOT PROVIDER_EVIDENCE"
                .into(),
        );
    }
    let plan: ScholarlyArchivePlan = serde_json::from_slice(&fs::read(&arguments[0])?)?;
    let receipt: ArchiveDepositReceipt = serde_json::from_slice(&fs::read(&arguments[1])?)?;
    let inventory_root = PathBuf::from(&arguments[2]);
    let provider_evidence = fs::read(&arguments[3])?;
    let revision = Command::new("git")
        .args(["rev-parse", &format!("{}^{{commit}}", plan.manifest.tag)])
        .output()?;
    if !revision.status.success() {
        return Err("release tag cannot be resolved".into());
    }
    let resolved_revision = String::from_utf8(revision.stdout)?;
    let mut observed_paths = Vec::new();
    collect_paths(&inventory_root, &inventory_root, &mut observed_paths)?;
    observed_paths.sort();
    let expected_paths = plan
        .manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    if observed_paths != expected_paths {
        return Err("local archive inventory paths differ from the exact manifest".into());
    }
    let files = expected_paths
        .iter()
        .map(|path| fs::read(inventory_root.join(path)).map(|bytes| (path.clone(), bytes)))
        .collect::<Result<Vec<_>, _>>()?;
    verify_archive_deposit_evidence(
        &plan,
        &receipt,
        resolved_revision.trim(),
        &provider_evidence,
        files
            .iter()
            .map(|(path, bytes)| (path.as_str(), bytes.as_slice())),
    )?;
    println!(
        "verified archive deposit {} ({}) for {} at {}",
        receipt.record_id, receipt.doi, receipt.tag, plan.manifest.source_revision
    );
    Ok(())
}
