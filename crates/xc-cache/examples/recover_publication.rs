//! Stage and optionally publish an exact retained artifact without numerical replay.
use std::{error::Error, fs::File, io::Read, path::Path};
use xc_cache::{DependencyRef, ManagedArtifactCacheConfig, ManagedArtifactCacheSession};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help") {
        println!("Usage: recover_publication ARTIFACT.json [--execute]\nARTIFACT.json is an exact DependencyRef. Default: local staging only.\nSet XC_CACHE_ROOT, XC_PUBLISH_STAGING_ROOT and XC_RESOURCE_POLICY_FILE.\n--execute requires XC_PUBLISH_EXECUTE=true and an explicit publication target.");
        return Ok(());
    }
    if args.len() > 2 || (args.len() == 2 && args[1] != "--execute") {
        return Err("invalid recovery arguments".into());
    }
    let mut bytes = Vec::new();
    File::open(Path::new(&args[0]))?
        .take(65_537)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 65_536 {
        return Err("recovery request exceeds 64 KiB".into());
    }
    let artifact: DependencyRef = serde_json::from_slice(&bytes)?;
    let mut config =
        ManagedArtifactCacheConfig::from_environment()?.ok_or("managed cache required")?;
    let execute = args.len() == 2;
    if execute
        && (!config.execute_remote_mutations
            || config.publication_target == xc_core::PublicationTarget::None)
    {
        return Err(
            "--execute requires explicit XC_PUBLISH_EXECUTE=true and publication target".into(),
        );
    }
    if config.replace_existing_publication {
        return Err(
            "recovery requires XC_PUBLISH_REPLACE=false to preserve historical artifacts".into(),
        );
    }
    config.execute_remote_mutations = execute;
    config.cache_mode = xc_cache::ArtifactExecutionCacheMode::RequireReuse;
    let session = ManagedArtifactCacheSession::new_with_resources(
        config,
        xc_cache::managed_resource_policy_from_environment()?,
    )?;
    session.stage_cached_artifact(&artifact)?;
    let inventory = session
        .finalize_publication_inventory()?
        .ok_or("missing publication inventory")?;
    println!("[PASS] exact retained artifact staged; numerical computation not invoked");
    println!("Inventory: {}", inventory.display());
    println!("Remote execution requested: {execute}");
    Ok(())
}
