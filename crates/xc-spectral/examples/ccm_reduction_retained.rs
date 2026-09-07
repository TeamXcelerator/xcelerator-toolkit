// Check explicitly supplied retained matrices; no discovery or publication.
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use serde::Deserialize;
    use std::{io::Write, path::PathBuf};
    use xc_cache::{ArtifactManifest, ContentDigest};
    use xc_spectral::ccm::prefix::{check_retained_reduction, RetainedEvenMatrix};

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
        approved_payload_digests: Vec<ContentDigest>,
        working_precision_bits: u32,
        maximum_dimension: usize,
        relative_tolerance: String,
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("usage: ccm_reduction_retained REQUEST.json NEW_OUTPUT.json (retained sources only)");
    }
    if PathBuf::from(&args[1]).exists() {
        bail!("output already exists; refusing to replace an earlier result");
    }
    let request_path = PathBuf::from(&args[0]).canonicalize()?;
    let request: Request = serde_json::from_slice(&std::fs::read(&request_path)?)?;
    let base = request_path.parent().context("request has no parent")?;
    let manifest: ArtifactManifest =
        serde_json::from_slice(&std::fs::read(base.join(&request.matrix.manifest))?)?;
    let bytes = std::fs::read(base.join(&request.matrix.payload))?;
    let matrix =
        RetainedEvenMatrix::from_payload(&manifest, &bytes, &request.approved_payload_digests)?;
    let report = check_retained_reduction(
        &matrix,
        request.working_precision_bits,
        request.maximum_dimension,
        &request.relative_tolerance,
    )?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    if !report.checks_passed {
        bail!("computed reduction checks exceed the requested tolerance; report retained");
    }
    eprintln!("Retained reduction checks passed at the requested tolerance. Computed diagnostics, not a certificate.");
    Ok(())
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("ccm_reduction_retained requires --features hp");
    std::process::exit(2);
}
