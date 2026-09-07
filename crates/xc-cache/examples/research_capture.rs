//! Synthetic example of managed outcome accounting; no numerical source acquisition.
use xc_cache::{
    capture_and_persist, CaptureFailure, CapturedDiagnostic, ManagedArtifactCacheSession,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session =
        ManagedArtifactCacheSession::from_environment()?.ok_or("managed cache unavailable")?;
    let plan = serde_json::json!({"example":"synthetic outcome accounting","normalization":"declared fixture units"});
    let record = capture_and_persist(
        &plan,
        vec!["measurement".into(), "missing_input".into()],
        |id| match id {
            "measurement" => CapturedDiagnostic::new(
                &serde_json::json!({"value":"-0.25","numerically_resolved":false}),
                vec![],
            )
            .map_err(|_| CaptureFailure::Failed {
                reason: "fixture serialization failed".into(),
            }),
            _ => Err(CaptureFailure::Missing {
                reason: "synthetic example has no second source".into(),
            }),
        },
        &session.context(),
    )?;
    session.finalize_publication_inventory()?;
    println!("{}", serde_json::to_string_pretty(&record.value)?);
    Ok(())
}
