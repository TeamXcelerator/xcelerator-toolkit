use std::path::PathBuf;
use xc_core::ExecutionFingerprint;
use xc_spectral::ccm::reproduction::{reproduce_saved_ccm_f64_observation, SavedCcmF64Observation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let saved_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: ccm_reproduce <saved-observation.json> <current-fingerprint.json>")?,
    );
    let fingerprint_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: ccm_reproduce <saved-observation.json> <current-fingerprint.json>")?,
    );
    if arguments.next().is_some() {
        return Err("ccm_reproduce accepts exactly two paths".into());
    }
    let saved: SavedCcmF64Observation = serde_json::from_slice(&std::fs::read(saved_path)?)?;
    let current: ExecutionFingerprint = serde_json::from_slice(&std::fs::read(fingerprint_path)?)?;
    let report = reproduce_saved_ccm_f64_observation(&saved, &current)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
