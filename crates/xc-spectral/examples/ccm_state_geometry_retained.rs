//! Build one managed geometry child from explicitly selected retained files.
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use serde::Deserialize;
    use std::{io::Write, path::PathBuf};
    use xc_cache::*;
    use xc_spectral::ccm::state_geometry::*;
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        manifest: PathBuf,
        payload: PathBuf,
        approved_payload_digests: Vec<ContentDigest>,
        cache_root: PathBuf,
        options: Option<GeometryOptions>,
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("usage: ccm_state_geometry_retained REQUEST.json NEW_OUTPUT.json");
    }
    if PathBuf::from(&args[1]).exists() {
        bail!("output already exists; refusing to replace retained evidence");
    }
    let input = PathBuf::from(&args[0]).canonicalize()?;
    let base = input.parent().context("request has no parent")?;
    let request: Request = serde_json::from_slice(&std::fs::read(&input)?)?;
    let manifest: ArtifactManifest =
        serde_json::from_slice(&std::fs::read(base.join(request.manifest))?)?;
    let bytes = std::fs::read(base.join(request.payload))?;
    let source = RetainedState::from_payload(&manifest, &bytes, &request.approved_payload_digests)?;
    let options = request
        .options
        .unwrap_or_else(|| GeometryOptions::for_source(&source));
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "geometry-retained",
            base.join(request.cache_root),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let context = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["geometry-retained".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let result = xc_numerics::hp_runtime::run_hp(|| {
        analyze_state_geometry_via_cache(&source, &options, &context)
    })?;
    let child = result
        .produced_manifest
        .as_ref()
        .or(result.reused_manifest.as_ref())
        .context("managed child manifest missing")?;
    let output = serde_json::json!({"manifest":child,"report":result.value});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    file.write_all(&serde_json::to_vec_pretty(&output)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    eprintln!("State geometry retained with exact source dependency. Computed diagnostics; inspect physical_sign_status and grid refinement before interpretation.");
    Ok(())
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("ccm_state_geometry_retained requires --features hp");
    std::process::exit(2);
}
