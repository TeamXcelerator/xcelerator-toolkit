// Owner-authorized generated-code assistance. See docs/CCM_PREFIX_ANALYSIS.md.
// Reads explicitly named retained files only; no cache discovery or publication.
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use serde::Deserialize;
    use std::{io::Write, path::PathBuf};
    use xc_cache::{ArtifactManifest, ContentDigest};
    use xc_spectral::ccm::prefix::{
        analyze_retained_prefixes, PrefixAnalysisOptions, RetainedEvenEigenpair, RetainedEvenMatrix,
    };
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Source {
        manifest: PathBuf,
        payload: PathBuf,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        matrix: Source,
        eigenpairs: Vec<Source>,
        approved_payload_digests: Vec<ContentDigest>,
        options: PrefixAnalysisOptions,
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("usage: ccm_prefix_retained REQUEST.json NEW_OUTPUT.json (retained sources only)");
    }
    let request_path = PathBuf::from(&args[0]).canonicalize()?;
    let request: Request = serde_json::from_slice(&std::fs::read(&request_path)?)?;
    let base = request_path.parent().context("request has no parent")?;
    let load = |s: &Source| -> anyhow::Result<(ArtifactManifest, Vec<u8>)> {
        let manifest = serde_json::from_slice(&std::fs::read(base.join(&s.manifest))?)?;
        Ok((manifest, std::fs::read(base.join(&s.payload))?))
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
    let report = analyze_retained_prefixes(&matrix, &request.options, &eigenpairs)?;
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
