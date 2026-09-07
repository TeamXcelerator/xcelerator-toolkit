// Owner-authorized generated-code assistance. See docs/CCM_PREFIX_ANALYSIS.md.
// Reads explicitly named retained files or local shard manifests; no network or publication.
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use serde::Deserialize;
    use std::{io::Write, path::PathBuf};
    use xc_cache::{read_local_shard_json, ArtifactManifest, ContentDigest, LocalShardReadOptions};
    use xc_spectral::ccm::prefix::{
        analyze_retained_prefixes, check_prefix_nesting, PrefixAnalysisOptions,
        RetainedEvenEigenpair, RetainedEvenMatrix,
    };
    #[derive(Deserialize)]
    #[serde(untagged, deny_unknown_fields)]
    enum Source {
        Runtime { manifest: PathBuf, payload: PathBuf },
        Shard { shard_manifest: PathBuf },
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        matrix: Source,
        eigenpairs: Vec<Source>,
        approved_payload_digests: Vec<ContentDigest>,
        options: PrefixAnalysisOptions,
        #[serde(default)]
        nesting_matrices: Vec<Source>,
        /// Required when any source uses a canonical shard manifest.
        local_shard_read: Option<LocalShardReadOptions>,
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("usage: ccm_prefix_retained REQUEST.json NEW_OUTPUT.json (retained sources only)");
    }
    let request_path = PathBuf::from(&args[0]).canonicalize()?;
    let request: Request = serde_json::from_slice(&std::fs::read(&request_path)?)?;
    let base = request_path.parent().context("request has no parent")?;
    let load = |s: &Source| -> anyhow::Result<(ArtifactManifest, Vec<u8>)> {
        match s {
            Source::Runtime { manifest, payload } => {
                let manifest = serde_json::from_slice(&std::fs::read(base.join(manifest))?)?;
                Ok((manifest, std::fs::read(base.join(payload))?))
            }
            Source::Shard { shard_manifest } => {
                let mut options=request.local_shard_read.clone().context("shard sources require explicit local_shard_read byte budgets and scratch_directory")?;
                options.scratch_directory = base.join(&options.scratch_directory);
                let loaded = read_local_shard_json(&base.join(shard_manifest), &options)?;
                Ok((loaded.manifest, loaded.payload))
            }
        }
    };
    let (m, b) = load(&request.matrix)?;
    let matrix = RetainedEvenMatrix::from_payload(&m, &b, &request.approved_payload_digests)?;
    let eigenpairs = request
        .eigenpairs
        .iter()
        .map(|s| {
            let (m, b) = load(s)?;
            RetainedEvenEigenpair::from_payload(&m, &b, &request.approved_payload_digests)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    // Compare sources before cubic work. A mismatch is recorded, never repaired.
    let nesting_checks = request
        .nesting_matrices
        .iter()
        .map(|source| -> anyhow::Result<_> {
            let (manifest, bytes) = load(source)?;
            let smaller = RetainedEvenMatrix::from_payload(
                &manifest,
                &bytes,
                &request.approved_payload_digests,
            )?;
            check_prefix_nesting(&smaller, &matrix)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut report = analyze_retained_prefixes(&matrix, &request.options, &eigenpairs)?;
    report.nesting_checks = nesting_checks;
    let encoded = serde_json::to_vec_pretty(&report)?;
    // Never overwrite an existing source artifact, report, or request file.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    eprintln!("Retained-source diagnostics written. Computed evidence, not a certificate; preserve source confidentiality.");
    Ok(())
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("ccm_prefix_retained requires --features hp");
    std::process::exit(2);
}
