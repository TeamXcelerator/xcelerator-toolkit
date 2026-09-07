// Retained-evidence workflow. Publication requires explicit managed-cache configuration.
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use serde::{Deserialize, Serialize};
    use std::{
        io::Write,
        path::{Path, PathBuf},
    };
    use xc_core::*;
    use xc_numerics::hypothesis::{
        evaluate_hypothesis_packet, persist_hypothesis_evaluation, HypothesisEvaluationPacket,
    };
    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct Entry {
        metadata: ObservationMetadata,
        payload: PathBuf,
    }
    fn write_new(path: &str, bytes: &[u8]) -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if (args.len() == 2 || args.len() == 3) && args[0] == "replay" {
        let allow = args
            .get(2)
            .is_some_and(|s| s == "--allow-protected-validation");
        if args.len() == 3 && !allow {
            bail!("unknown replay option");
        }
        let packet: HypothesisEvaluationPacket = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        packet.validate(allow)?;
        println!(
            "Verified {:?}; complete={}",
            packet.score.verdict, packet.score.complete
        );
        return Ok(());
    }
    if (args.len() == 3 || args.len() == 4) && args[0] == "store" {
        let allow = args
            .get(3)
            .is_some_and(|s| s == "--allow-protected-validation");
        if args.len() == 4 && !allow {
            bail!("unknown store option");
        }
        let packet: HypothesisEvaluationPacket = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let sources: Vec<xc_cache::ArtifactManifest> =
            serde_json::from_slice(&std::fs::read(&args[2])?)?;
        let session = xc_cache::ManagedArtifactCacheSession::from_environment()?
            .context("managed cache is unavailable")?;
        let result = persist_hypothesis_evaluation(&packet, &sources, allow, &session.context())?;
        session.finalize_publication_inventory()?;
        let manifest = result
            .produced_manifest
            .or(result.reused_manifest)
            .context("record was not persisted")?;
        println!("Stored {} {}", manifest.key.kind, manifest.content_digest.0);
        return Ok(());
    }
    if args.len() == 3 && args[0] == "freeze" {
        let spec: HypothesisSpec = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let frozen = spec.freeze()?;
        write_new(&args[2], &serde_json::to_vec_pretty(&frozen)?)?;
        println!(
            "Frozen {}. Identity does not establish prior blindness.",
            frozen.digest()
        );
        return Ok(());
    }
    if !(args.len() == 6 || args.len() == 7) || args[0] != "score" {
        bail!("usage: research_score freeze SPEC.json NEW_FROZEN.json | score FROZEN.json INVENTORY.json PARTITION BITS NEW_PACKET.json [--allow-protected-validation] | replay PACKET.json [--allow-protected-validation] | store PACKET.json SOURCES.json [--allow-protected-validation]");
    }
    let allow = args
        .get(6)
        .is_some_and(|s| s == "--allow-protected-validation");
    if args.len() == 7 && !allow {
        bail!("unknown option");
    }
    let partition = match args[3].as_str() {
        "calibration" => DatasetPartition::Calibration,
        "development" => DatasetPartition::Development,
        "replication" => DatasetPartition::Replication,
        "protected_validation" => DatasetPartition::ProtectedValidation,
        _ => bail!("unknown partition"),
    };
    let frozen: FrozenHypothesis = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let inventory_path = Path::new(&args[2]).canonicalize()?;
    let entries: Vec<Entry> = serde_json::from_slice(&std::fs::read(&inventory_path)?)?;
    let base = inventory_path.parent().context("inventory has no parent")?;
    let metadata = entries
        .iter()
        .map(|e| e.metadata.clone())
        .collect::<Vec<_>>();
    let packet = evaluate_hypothesis_packet(
        &frozen,
        &metadata,
        partition,
        allow,
        args[4].parse()?,
        |m| {
            let entry = entries
                .iter()
                .find(|e| &e.metadata == m)
                .context("selected metadata missing")?;
            // This is the ONLY observation-file read, behind the toolkit's gates.
            Ok(std::fs::read(base.join(&entry.payload))?)
        },
    )?;
    validate_secret_free(&packet, "research packet")?;
    write_new(&args[5], &serde_json::to_vec_pretty(&packet)?)?;
    println!("Research packet written; inspect scientific verdict and completeness separately.");
    Ok(())
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("research_score requires --features hp");
    std::process::exit(2);
}
