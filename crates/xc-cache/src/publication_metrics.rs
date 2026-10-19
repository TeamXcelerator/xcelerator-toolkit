//! Local operational telemetry, excluded from scientific and publication identities.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
struct AttemptWriter {
    file: File,
    started: Instant,
}

#[derive(Debug, Default)]
pub(crate) struct PublicationMetrics {
    file: Mutex<Option<AttemptWriter>>,
}
impl PublicationMetrics {
    pub fn enable(&self, directory: &Path) -> std::io::Result<()> {
        // A retained transport may start another dependency wave. Close the
        // previous attempt even if creating the next output file fails.
        self.close();
        fs::create_dir_all(directory)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = directory.join(format!("attempt-{}-{stamp}.jsonl", std::process::id()));
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        *self.file.lock().unwrap_or_else(|e| e.into_inner()) = Some(AttemptWriter {
            file,
            started: Instant::now(),
        });
        self.record(
            "attempt_started",
            Duration::ZERO,
            json!({"scope":"local operational timings; scheduled bytes are not wire bytes"}),
        );
        Ok(())
    }
    pub(crate) fn close(&self) {
        self.record(
            "attempt_closed",
            Duration::ZERO,
            json!({"completion":"consult attempt_outcome and canonical publication report"}),
        );
        *self.file.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    pub fn record(&self, phase: &str, elapsed: Duration, details: Value) {
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let Some(attempt) = file.as_mut() else { return };
        let value = json!({"schema_version":1,"phase":phase,"elapsed_seconds":elapsed.as_secs_f64(),"observed_seconds_since_attempt_start":attempt.started.elapsed().as_secs_f64(),"details":details});
        let writer = &mut attempt.file;
        let result = (|| -> std::io::Result<()> {
            serde_json::to_writer(&mut *writer, &value)?;
            writer.write_all(b"\n")?;
            writer.flush()
        })();
        if result.is_err() {
            xc_core::progress_message!(
                "publication telemetry unavailable; numerical and publication checks continue"
            );
            *file = None;
        }
    }
}

impl Drop for PublicationMetrics {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct PhaseTimingSummary {
    event_count: u64,
    elapsed_seconds: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AttemptSummary {
    relative_path: String,
    outcome: String,
    outcome_evidence: String,
    /// Available only from the whole-attempt timer, never a phase sum.
    wall_seconds: Option<f64>,
    last_observed_seconds: Option<f64>,
    closed: bool,
    malformed_records: u64,
    unreadable_or_oversized: bool,
    scan_budget_exceeded: bool,
    phase_timings: BTreeMap<String, PhaseTimingSummary>,
    counters: BTreeMap<String, u64>,
}

fn operational_phase(phase: &str) -> bool {
    matches!(
        phase,
        "preparation"
            | "lock_wait"
            | "metadata_preparation"
            | "staged_verification"
            | "git_import"
            | "git_push"
            | "destination_verification"
            | "batch_checked"
            | "batch_reused"
            | "batch_committed"
            | "attempt_finished"
    )
}

fn increment_counter(summary: &mut AttemptSummary, name: &str, value: u64) {
    let total = summary.counters.entry(name.to_owned()).or_default();
    *total = total.saturating_add(value);
}

fn summarize_attempt(
    path: &Path,
    journal_root: &Path,
    remaining_bytes: &mut u64,
) -> AttemptSummary {
    const MAXIMUM_ATTEMPT_BYTES: u64 = 16 * 1024 * 1024;
    const MAXIMUM_RECORD_BYTES: usize = 1024 * 1024;
    let mut summary = AttemptSummary {
        relative_path: path
            .strip_prefix(journal_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/"),
        outcome: "unknown".to_owned(),
        outcome_evidence: "missing".to_owned(),
        ..AttemptSummary::default()
    };
    if *remaining_bytes == 0 {
        summary.scan_budget_exceeded = true;
        return summary;
    }
    let maximum_read = MAXIMUM_ATTEMPT_BYTES.min(*remaining_bytes);
    let input = (|| -> std::io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(maximum_read + 1)
            .read_to_end(&mut bytes)?;
        Ok(bytes)
    })();
    let bytes = match input {
        Ok(bytes) if bytes.len() as u64 <= maximum_read => {
            *remaining_bytes -= bytes.len() as u64;
            bytes
        }
        _ => {
            summary.scan_budget_exceeded = maximum_read < MAXIMUM_ATTEMPT_BYTES;
            *remaining_bytes = remaining_bytes.saturating_sub(maximum_read + 1);
            summary.unreadable_or_oversized = true;
            return summary;
        }
    };
    let Ok(input) = std::str::from_utf8(&bytes) else {
        summary.unreadable_or_oversized = true;
        return summary;
    };
    for line in input.lines() {
        let parsed = (|| {
            if line.len() > MAXIMUM_RECORD_BYTES {
                return None;
            }
            let row = serde_json::from_str::<Value>(line).ok()?;
            let phase = row["phase"].as_str()?;
            let elapsed = row["elapsed_seconds"].as_f64()?;
            if row["schema_version"] != 1
                || !elapsed.is_finite()
                || !(0.0..=1_000_000_000.0).contains(&elapsed)
            {
                return None;
            }
            Some((phase.to_owned(), elapsed, row))
        })();
        let Some((phase, elapsed, row)) = parsed else {
            summary.malformed_records += 1;
            continue;
        };
        if let Some(observed) = row["observed_seconds_since_attempt_start"]
            .as_f64()
            .filter(|value| value.is_finite() && (0.0..=1_000_000_000.0).contains(value))
        {
            summary.last_observed_seconds =
                Some(summary.last_observed_seconds.unwrap_or(0.0).max(observed));
        }
        if phase == "attempt_closed" {
            summary.closed = true;
        }
        if (phase == "attempt_outcome"
            || (phase == "attempt_finished" && summary.outcome_evidence != "attempt_outcome"))
            && row["details"]["success"].is_boolean()
        {
            summary.outcome = if row["details"]["success"] == true {
                "success"
            } else {
                "failure"
            }
            .to_owned();
            summary.outcome_evidence = phase.clone();
            if phase == "attempt_outcome" {
                summary.wall_seconds = Some(elapsed);
            }
        }
        if operational_phase(&phase) {
            let timing = summary.phase_timings.entry(phase.clone()).or_default();
            timing.event_count += 1;
            timing.elapsed_seconds += elapsed;
        }
        // Keep observations from different layers separate: the same bytes can
        // appear in batch selection, local import and destination validation.
        let fields: &[(&str, &str)] = match phase.as_str() {
            "candidate_selection" => &[
                ("already_present", "candidates_already_present"),
                ("pending", "candidates_pending"),
            ],
            "batch_checked" => &[
                ("reused_bytes", "batch_checked_reused_bytes"),
                ("scheduled_bytes", "batch_checked_scheduled_bytes"),
            ],
            "git_import" => &[
                ("scheduled_bytes", "git_import_scheduled_bytes"),
                ("files", "git_import_files"),
            ],
            "staged_verification" => &[("bytes", "staged_verification_observed_bytes")],
            "destination_verification" => &[(
                "reused_verified_bytes",
                "destination_verification_reused_bytes",
            )],
            _ => &[],
        };
        for (field, counter) in fields {
            if let Some(value) = row["details"][field].as_u64() {
                increment_counter(&mut summary, counter, value);
            }
        }
        if phase == "batch_reused" {
            increment_counter(&mut summary, "batch_reused_events", 1);
        }
        if phase == "batch_committed" {
            increment_counter(&mut summary, "batch_committed_events", 1);
        }
        if matches!(
            phase.as_str(),
            "git_push" | "git_import" | "staged_verification" | "lock_wait"
        ) && row["details"]["success"] == false
        {
            increment_counter(&mut summary, &format!("{phase}_failure_events"), 1);
        }
    }
    // An incomplete last line may be a crash or concurrent snapshot. Even a
    // parseable prefix without its terminating newline is not a durable event.
    if !input.is_empty() && !input.ends_with('\n') {
        summary.malformed_records += 1;
    }
    if summary.malformed_records > 0 {
        summary.outcome = "unknown".to_owned();
        summary.outcome_evidence = "malformed_telemetry".to_owned();
        summary.wall_seconds = None;
    }
    summary
}

/// A bounded, best-effort snapshot beside the canonical publication records.
/// Nothing in this document is read by admission, identity or recovery logic.
pub(crate) fn write_operational_summary(
    journal_root: &Path,
    invocation_wall: Duration,
    invocation_succeeded: bool,
) -> Result<PathBuf, crate::CacheError> {
    const MAXIMUM_ATTEMPTS: usize = 1024;
    const MAXIMUM_SCAN_ENTRIES: usize = 16_384;
    const MAXIMUM_TOTAL_BYTES: u64 = 64 * 1024 * 1024;
    let mut files = Vec::new();
    let mut remaining = MAXIMUM_SCAN_ENTRIES;
    let mut incomplete_scan = false;
    let mut pending = vec![(journal_root.join("family-batches"), 0usize)];
    while let Some((directory, depth)) = pending.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => {
                incomplete_scan = true;
                continue;
            }
        };
        for entry in entries {
            if remaining == 0 || files.len() >= MAXIMUM_ATTEMPTS {
                incomplete_scan = true;
                break;
            }
            remaining -= 1;
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    incomplete_scan = true;
                    continue;
                }
            };
            let kind = match entry.file_type() {
                Ok(kind) => kind,
                Err(_) => {
                    incomplete_scan = true;
                    continue;
                }
            };
            // Never follow directory links. Only the known family/destination/
            // publication-metrics layout is telemetry, not arbitrary JSONL.
            if kind.is_dir()
                && depth < 3
                && (depth != 2 || entry.file_name() == "publication-metrics")
            {
                pending.push((entry.path(), depth + 1));
            }
            if kind.is_file() && depth == 3 {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("attempt-") && name.ends_with(".jsonl") {
                    files.push(entry.path());
                }
            }
        }
    }
    files.sort();
    let mut remaining_bytes = MAXIMUM_TOTAL_BYTES;
    let attempts = files
        .iter()
        .map(|path| summarize_attempt(path, journal_root, &mut remaining_bytes))
        .collect::<Vec<_>>();
    incomplete_scan |= attempts.iter().any(|attempt| attempt.scan_budget_exceeded);
    let mut phases = BTreeMap::<String, PhaseTimingSummary>::new();
    let mut counters = BTreeMap::<String, u64>::new();
    for attempt in &attempts {
        for (phase, observed) in &attempt.phase_timings {
            let total = phases.entry(phase.clone()).or_default();
            total.event_count += observed.event_count;
            total.elapsed_seconds += observed.elapsed_seconds;
        }
        for (name, value) in &attempt.counters {
            let total = counters.entry(name.clone()).or_default();
            *total = total.saturating_add(*value);
        }
    }
    let report = json!({
        "schema_version": 1,
        "scope": "local operational snapshot; canonical publication reports determine durable publication",
        "timing_semantics": "phase totals are inclusive, nested and may overlap across workers; never sum them as wall time",
        "byte_semantics": "counters are separate layer observations, may overlap or repeat, and never measure wire bytes",
        "latest_invocation": {"outcome":if invocation_succeeded {"success"} else {"failure"}, "wall_seconds":invocation_wall.as_secs_f64()},
        "attempt_count": attempts.len(),
        "successful_attempts": attempts.iter().filter(|attempt| attempt.outcome == "success").count(),
        "failed_attempts": attempts.iter().filter(|attempt| attempt.outcome == "failure").count(),
        "unknown_outcome_attempts": attempts.iter().filter(|attempt| attempt.outcome == "unknown").count(),
        "incomplete_scan": incomplete_scan,
        "phase_timings": phases,
        "counters": counters,
        "attempts": attempts,
    });
    let path = journal_root.join("publication-operational-summary.json");
    crate::atomic_replace(&path, &serde_json::to_vec_pretty(&report)?)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operational_summary_keeps_nested_timings_bytes_and_outcomes_separate() {
        let root = crate::test_support::TestDir::new("publication-summary");
        let metrics_path = root.join("family-batches/family/private/publication-metrics");
        let metrics = PublicationMetrics::default();
        metrics.enable(&metrics_path).unwrap();
        metrics.record("preparation", Duration::from_secs(10), json!({}));
        metrics.record(
            "git_import",
            Duration::from_secs(8),
            json!({"scheduled_bytes":100,"files":1,"success":true}),
        );
        metrics.record("lock_wait", Duration::from_secs(2), json!({}));
        metrics.record(
            "candidate_selection",
            Duration::ZERO,
            json!({"already_present":3,"pending":2}),
        );
        metrics.record(
            "batch_checked",
            Duration::from_secs(4),
            json!({"reused_bytes":20,"scheduled_bytes":100}),
        );
        metrics.record("git_push", Duration::from_secs(3), json!({"success":false}));
        metrics.record(
            "attempt_outcome",
            Duration::from_secs(17),
            json!({"success":false}),
        );
        metrics.close();
        metrics.enable(&metrics_path).unwrap();
        metrics.close(); // Closure alone is not success.
        let path = write_operational_summary(&root, Duration::from_secs(19), false).unwrap();
        let summary: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(summary["latest_invocation"]["wall_seconds"], 19.0);
        assert_eq!(summary["failed_attempts"], 1);
        assert_eq!(summary["unknown_outcome_attempts"], 1);
        assert_eq!(
            summary["phase_timings"]["preparation"]["elapsed_seconds"],
            10.0
        );
        assert_eq!(
            summary["phase_timings"]["git_import"]["elapsed_seconds"],
            8.0
        );
        assert_eq!(summary["counters"]["git_import_scheduled_bytes"], 100);
        assert_eq!(summary["counters"]["batch_checked_scheduled_bytes"], 100);
        assert_eq!(summary["counters"]["candidates_already_present"], 3);
        assert_eq!(summary["counters"]["git_push_failure_events"], 1);
        assert!(summary.get("wire_bytes").is_none());
        assert!(summary.get("total_phase_wall_seconds").is_none());
    }

    #[test]
    fn operational_summary_marks_truncation_and_oversize_unknown_without_modifying_jsonl() {
        let root = crate::test_support::TestDir::new("publication-summary-partial");
        let logs = root.join("family-batches/family/private/publication-metrics");
        fs::create_dir_all(&logs).unwrap();
        let partial = logs.join("attempt-partial.jsonl");
        let bytes = b"{\"schema_version\":1,\"phase\":\"attempt_outcome\",\"elapsed_seconds\":2,\"details\":{\"success\":true}}\n{\"phase\":";
        fs::write(&partial, bytes).unwrap();
        File::create(logs.join("attempt-oversize.jsonl"))
            .unwrap()
            .set_len(16 * 1024 * 1024 + 1)
            .unwrap();
        let path = write_operational_summary(&root, Duration::from_secs(3), true).unwrap();
        let summary: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(summary["successful_attempts"], 0);
        assert_eq!(summary["unknown_outcome_attempts"], 2);
        assert_eq!(fs::read(partial).unwrap(), bytes);
    }
    #[test]
    fn telemetry_retains_attempts_and_failed_stages_without_identity_data() {
        let scratch = crate::test_support::TestDir::new("telemetry");
        let root = scratch.join("root");
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
        assert_eq!(rows.len(), 5);
        assert_eq!(
            rows.iter()
                .filter(|row| row["phase"] == "attempt_closed")
                .count(),
            2
        );
        assert!(rows
            .iter()
            .any(|r| r["phase"] == "push" && r["details"]["success"] == false));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_attempt_reenable_closes_previous_file_without_appending_new_events() {
        let root = crate::test_support::TestDir::new("telemetry-reenable");
        let metrics = PublicationMetrics::default();
        metrics.enable(&root.join("logs")).unwrap();
        let path = fs::read_dir(root.join("logs"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let blocker = root.join("ordinary-file");
        fs::write(&blocker, b"not a directory").unwrap();
        assert!(metrics.enable(&blocker).is_err());
        let closed = fs::read(&path).unwrap();
        metrics.record("push", Duration::ZERO, json!({"success":true}));
        drop(metrics);
        assert_eq!(fs::read(path).unwrap(), closed);
        let rows = std::str::from_utf8(&closed)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["phase"], "attempt_closed");
    }
}
