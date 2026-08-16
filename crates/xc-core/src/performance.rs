// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Opt-in process-wide performance tracing for research runs.
//!
//! The trace is operational evidence only. It is never part of artifact
//! identity, payloads, manifests, or publication inputs.

use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Environment variable containing the JSON performance-report path.
pub const PERFORMANCE_REPORT_ENV: &str = "XC_PERF_REPORT";

/// Scheduling and problem-shape context attached to one measured stage.
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceStageMetadata {
    pub operation: Option<String>,
    pub requested_table_order: Option<usize>,
    pub matrix_dimension: Option<usize>,
    pub precision_bits: Option<u32>,
    pub rayon_workers: Option<usize>,
    pub hp_runtime_mode: Option<String>,
    pub cache_disposition: Option<String>,
    pub retained_hp_entries: Option<usize>,
    pub unique_table_orders: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub table_orders: Vec<usize>,
    pub scheduling: Option<String>,
}

impl PerformanceStageMetadata {
    pub fn matrix(dimension: usize, precision_bits: u32, rayon_workers: usize) -> Self {
        Self {
            matrix_dimension: Some(dimension),
            precision_bits: Some(precision_bits),
            rayon_workers: Some(rayon_workers),
            ..Self::default()
        }
    }

    pub fn gl_batch(
        table_orders: Vec<usize>,
        precision_bits: u32,
        rayon_workers: usize,
        scheduling: impl Into<String>,
    ) -> Self {
        Self {
            precision_bits: Some(precision_bits),
            rayon_workers: Some(rayon_workers),
            unique_table_orders: Some(table_orders.len()),
            table_orders,
            scheduling: Some(scheduling.into()),
            ..Self::default()
        }
    }
}

/// Aggregated measurements for one stage and one exact metadata tuple.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceStageRecord {
    pub stage: String,
    pub metadata: PerformanceStageMetadata,
    pub invocations: u64,
    pub total_elapsed_ns: u64,
    pub minimum_elapsed_ns: u64,
    pub maximum_elapsed_ns: u64,
}

/// Cumulative process report written to `XC_PERF_REPORT`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceRunReport {
    pub schema_version: u32,
    pub toolkit_version: String,
    pub process_id: u32,
    pub started_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub records: Vec<PerformanceStageRecord>,
}

impl PerformanceRunReport {
    fn new() -> Self {
        let now = unix_millis();
        Self {
            schema_version: 1,
            toolkit_version: env!("CARGO_PKG_VERSION").to_owned(),
            process_id: std::process::id(),
            started_unix_ms: now,
            updated_unix_ms: now,
            records: Vec::new(),
        }
    }

    fn record(&mut self, stage: &str, metadata: PerformanceStageMetadata, elapsed: Duration) {
        let elapsed_ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
        if let Some(record) = self
            .records
            .iter_mut()
            .find(|record| record.stage == stage && record.metadata == metadata)
        {
            record.invocations = record.invocations.saturating_add(1);
            record.total_elapsed_ns = record.total_elapsed_ns.saturating_add(elapsed_ns);
            record.minimum_elapsed_ns = record.minimum_elapsed_ns.min(elapsed_ns);
            record.maximum_elapsed_ns = record.maximum_elapsed_ns.max(elapsed_ns);
        } else {
            self.records.push(PerformanceStageRecord {
                stage: stage.to_owned(),
                metadata,
                invocations: 1,
                total_elapsed_ns: elapsed_ns,
                minimum_elapsed_ns: elapsed_ns,
                maximum_elapsed_ns: elapsed_ns,
            });
        }
        self.records.sort_by(|left, right| {
            left.stage
                .cmp(&right.stage)
                .then_with(|| left.metadata.cmp(&right.metadata))
        });
        self.updated_unix_ms = unix_millis();
    }
}

struct ActivePerformanceRun {
    report_path: PathBuf,
    report: PerformanceRunReport,
    open_top_level_stages: usize,
}

#[derive(Default)]
struct PerformanceRecorder {
    active: Option<ActivePerformanceRun>,
}

static PERFORMANCE_RECORDER: OnceLock<Mutex<PerformanceRecorder>> = OnceLock::new();

/// RAII timer for one performance stage.
///
/// Dropping the guard records the elapsed duration even when the measured
/// operation returns early with an error.
#[must_use]
pub struct PerformanceStageGuard {
    stage: &'static str,
    metadata: PerformanceStageMetadata,
    started: Instant,
    report_path: Option<PathBuf>,
    top_level: bool,
}

impl PerformanceStageGuard {
    pub fn set_cache_disposition(&mut self, disposition: impl Into<String>) {
        if self.report_path.is_some() {
            self.metadata.cache_disposition = Some(disposition.into());
        }
    }
}

impl Drop for PerformanceStageGuard {
    fn drop(&mut self) {
        let Some(report_path) = self.report_path.take() else {
            return;
        };
        if let Err(error) = finish_stage(
            &report_path,
            self.stage,
            self.metadata.clone(),
            self.started.elapsed(),
            self.top_level,
        ) {
            eprintln!(
                "performance report: failed to record stage {} at {}: {error}",
                self.stage,
                report_path.display()
            );
        }
    }
}

/// Start a child stage. Its measurement is included in the next top-level
/// snapshot but does not perform report I/O when it closes.
pub fn performance_stage(
    stage: &'static str,
    metadata: PerformanceStageMetadata,
) -> PerformanceStageGuard {
    start_stage(stage, || metadata, false)
}

/// Lazily start a child stage. The metadata closure is not evaluated when
/// performance reporting is disabled.
pub fn performance_stage_with<F>(stage: &'static str, metadata: F) -> PerformanceStageGuard
where
    F: FnOnce() -> PerformanceStageMetadata,
{
    start_stage(stage, metadata, false)
}

/// Start a top-level stage. Closing the last open top-level stage safely
/// refreshes the cumulative process report.
pub fn performance_top_level_stage(
    stage: &'static str,
    metadata: PerformanceStageMetadata,
) -> PerformanceStageGuard {
    start_stage(stage, || metadata, true)
}

/// Lazily start a top-level stage. The metadata closure is not evaluated when
/// performance reporting is disabled.
pub fn performance_top_level_stage_with<F>(
    stage: &'static str,
    metadata: F,
) -> PerformanceStageGuard
where
    F: FnOnce() -> PerformanceStageMetadata,
{
    start_stage(stage, metadata, true)
}

/// Explicit-path variant used by concurrency tests and embedding applications
/// that have already resolved operational configuration.
#[doc(hidden)]
pub fn performance_stage_at_path(
    report_path: &Path,
    stage: &'static str,
    metadata: PerformanceStageMetadata,
) -> PerformanceStageGuard {
    start_stage_with_path(stage, || metadata, false, Some(report_path.to_owned()))
}

/// Explicit-path top-level variant; see [`performance_stage_at_path`].
#[doc(hidden)]
pub fn performance_top_level_stage_at_path(
    report_path: &Path,
    stage: &'static str,
    metadata: PerformanceStageMetadata,
) -> PerformanceStageGuard {
    start_stage_with_path(stage, || metadata, true, Some(report_path.to_owned()))
}

fn start_stage<F>(stage: &'static str, metadata: F, top_level: bool) -> PerformanceStageGuard
where
    F: FnOnce() -> PerformanceStageMetadata,
{
    start_stage_with_path(stage, metadata, top_level, configured_report_path())
}

fn start_stage_with_path<F>(
    stage: &'static str,
    metadata: F,
    top_level: bool,
    mut report_path: Option<PathBuf>,
) -> PerformanceStageGuard
where
    F: FnOnce() -> PerformanceStageMetadata,
{
    if let Some(path) = report_path.clone() {
        let mut recorder = lock_recorder();
        let replace_active = recorder
            .active
            .as_ref()
            .is_none_or(|active| active.report_path != path && active.open_top_level_stages == 0);
        if replace_active {
            recorder.active = Some(ActivePerformanceRun {
                report_path: path.clone(),
                report: PerformanceRunReport::new(),
                open_top_level_stages: 0,
            });
        } else if recorder
            .active
            .as_ref()
            .is_some_and(|active| active.report_path != path)
        {
            eprintln!(
                "performance report: ignored concurrent report path {} while {} is active",
                path.display(),
                recorder
                    .active
                    .as_ref()
                    .expect("active report checked above")
                    .report_path
                    .display()
            );
            report_path = None;
        }
        if top_level && report_path.is_some() {
            let active = recorder
                .active
                .as_mut()
                .expect("enabled report path has an active recorder");
            active.open_top_level_stages = active.open_top_level_stages.saturating_add(1);
        }
    }
    let metadata = if report_path.is_some() {
        metadata()
    } else {
        PerformanceStageMetadata::default()
    };
    PerformanceStageGuard {
        stage,
        metadata,
        started: Instant::now(),
        report_path,
        top_level,
    }
}

fn finish_stage(
    report_path: &Path,
    stage: &str,
    metadata: PerformanceStageMetadata,
    elapsed: Duration,
    top_level: bool,
) -> Result<(), String> {
    let mut recorder = lock_recorder();
    let active = recorder
        .active
        .as_mut()
        .filter(|active| active.report_path == report_path)
        .ok_or_else(|| "active performance run changed before stage completion".to_owned())?;
    active.report.record(stage, metadata, elapsed);
    if top_level {
        active.open_top_level_stages = active
            .open_top_level_stages
            .checked_sub(1)
            .ok_or_else(|| "performance top-level stage accounting underflow".to_owned())?;
    }
    if top_level && active.open_top_level_stages == 0 {
        persist_report(report_path, &active.report)?;
    }
    Ok(())
}

fn configured_report_path() -> Option<PathBuf> {
    std::env::var_os(PERFORMANCE_REPORT_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn lock_recorder() -> MutexGuard<'static, PerformanceRecorder> {
    PERFORMANCE_RECORDER
        .get_or_init(|| Mutex::new(PerformanceRecorder::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn persist_report(path: &Path, report: &PerformanceRunReport) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
    replace_file(path, &bytes)
}

fn replace_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("performance-report.json");
    let temporary = parent.join(format!(
        ".{file_name}.{}-{}.tmp",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        drop(file);
        #[cfg(windows)]
        if path.exists() {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
        fs::rename(&temporary, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
fn reset_recorder_for_test() {
    lock_recorder().active = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());

    fn report_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "xc-performance-{label}-{}-{}.performance.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn read_report(path: &Path) -> PerformanceRunReport {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn disabled_recorder_creates_no_file() {
        let _environment = ENVIRONMENT_LOCK.lock().unwrap();
        reset_recorder_for_test();
        std::env::remove_var(PERFORMANCE_REPORT_ENV);
        let path = report_path("disabled");
        {
            let _stage =
                performance_top_level_stage("test.disabled", PerformanceStageMetadata::default());
        }
        assert!(!path.exists());
    }

    #[test]
    fn disabled_lazy_recorder_does_not_build_metadata() {
        let _environment = ENVIRONMENT_LOCK.lock().unwrap();
        reset_recorder_for_test();
        std::env::remove_var(PERFORMANCE_REPORT_ENV);
        let mut evaluated = false;
        {
            let _stage = performance_top_level_stage_with("test.disabled_lazy", || {
                evaluated = true;
                PerformanceStageMetadata::default()
            });
        }
        assert!(!evaluated);
    }

    #[test]
    fn nested_top_level_stages_snapshot_only_after_the_outer_stage_closes() {
        let _environment = ENVIRONMENT_LOCK.lock().unwrap();
        reset_recorder_for_test();
        let path = report_path("nested");
        let outer = performance_top_level_stage_at_path(
            &path,
            "test.outer",
            PerformanceStageMetadata::default(),
        );
        {
            let _inner = performance_top_level_stage_at_path(
                &path,
                "test.inner",
                PerformanceStageMetadata::default(),
            );
        }
        assert!(!path.exists());
        drop(outer);

        let report = read_report(&path);
        assert!(report
            .records
            .iter()
            .any(|record| record.stage == "test.inner"));
        assert!(report
            .records
            .iter()
            .any(|record| record.stage == "test.outer"));

        reset_recorder_for_test();
        let _ = fs::remove_file(path);
    }

    #[test]
    fn error_drop_persists_and_later_stages_accumulate() {
        let _environment = ENVIRONMENT_LOCK.lock().unwrap();
        reset_recorder_for_test();
        let path = report_path("cumulative");
        std::env::set_var(PERFORMANCE_REPORT_ENV, &path);

        let fail = || -> Result<(), &'static str> {
            let _stage = performance_top_level_stage(
                "test.first_session",
                PerformanceStageMetadata::matrix(7, 256, 2),
            );
            Err("deliberate")
        };
        assert_eq!(fail(), Err("deliberate"));
        {
            let _child =
                performance_stage("test.worker_child", PerformanceStageMetadata::default());
        }
        {
            let _stage = performance_top_level_stage(
                "test.second_session",
                PerformanceStageMetadata::default(),
            );
        }

        let report = read_report(&path);
        assert_eq!(report.schema_version, 1);
        assert!(report
            .records
            .iter()
            .any(|record| record.stage == "test.first_session"));
        assert!(report
            .records
            .iter()
            .any(|record| record.stage == "test.worker_child"));
        assert!(report
            .records
            .iter()
            .any(|record| record.stage == "test.second_session"));

        std::env::remove_var(PERFORMANCE_REPORT_ENV);
        reset_recorder_for_test();
        let _ = fs::remove_file(path);
    }
}
