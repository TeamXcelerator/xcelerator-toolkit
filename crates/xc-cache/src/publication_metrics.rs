//! Local operational telemetry, excluded from scientific and publication identities.
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Default)]
pub(crate) struct PublicationMetrics {
    file: Mutex<Option<File>>,
}
impl PublicationMetrics {
    pub fn enable(&self, directory: &Path) -> std::io::Result<()> {
        fs::create_dir_all(directory)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = directory.join(format!("attempt-{}-{stamp}.jsonl", std::process::id()));
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        *self.file.lock().unwrap_or_else(|e| e.into_inner()) = Some(file);
        self.record(
            "attempt_started",
            Duration::ZERO,
            json!({"scope":"local operational timings; scheduled bytes are not wire bytes"}),
        );
        Ok(())
    }
    pub fn record(&self, phase: &str, elapsed: Duration, details: Value) {
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let Some(writer) = file.as_mut() else { return };
        let value = json!({"schema_version":1,"phase":phase,"elapsed_seconds":elapsed.as_secs_f64(),"details":details});
        let result = (|| -> std::io::Result<()> {
            serde_json::to_writer(&mut *writer, &value)?;
            writer.write_all(b"\n")?;
            writer.flush()
        })();
        if result.is_err() {
            eprintln!(
                "publication telemetry unavailable; numerical and publication checks continue"
            );
            *file = None;
        }
    }
}

impl Drop for PublicationMetrics {
    fn drop(&mut self) {
        self.record(
            "attempt_closed",
            Duration::ZERO,
            json!({"completion":"consult attempt_finished and canonical publication report"}),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn telemetry_retains_attempts_and_failed_stages_without_identity_data() {
        let root = std::env::temp_dir().join(format!(
            "xc-telemetry-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let m = PublicationMetrics::default();
        m.enable(&root).unwrap();
        m.record(
            "push",
            Duration::from_millis(12),
            json!({"success":false,"scheduled_bytes":100}),
        );
        m.enable(&root).unwrap();
        drop(m);
        let files = fs::read_dir(&root)
            .unwrap()
            .map(|p| p.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 2);
        let rows = files
            .iter()
            .flat_map(|p| {
                fs::read_to_string(p)
                    .unwrap()
                    .lines()
                    .map(|s| serde_json::from_str::<Value>(s).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 4);
        assert!(rows
            .iter()
            .any(|r| r["phase"] == "push" && r["details"]["success"] == false));
        fs::remove_dir_all(root).unwrap();
    }
}
