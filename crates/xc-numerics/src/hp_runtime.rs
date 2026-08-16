// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Explicit high-precision runtime scheduling policy.
//!
//! Numerical entry points default to full parallel execution. Safe mode is
//! never inferred from the platform or process environment: callers resolve an
//! [`HpRuntimePolicy`], record it in provenance, and
//! invoke [`run_hp_with_policy`]. The policy changes scheduling and stack
//! resources only; it cannot change mathematical configuration or assurance.

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{Display, Formatter};
pub use xc_core::{HpRuntimeMode, HpRuntimePolicy};

thread_local! {
    static ACTIVE_POLICY: RefCell<Option<HpRuntimePolicy>> = const { RefCell::new(None) };
}

// Root construction is quadratic in table order and increasingly expensive
// with MPFR precision. Keep the opt-in schedule conservative until native-
// Linux qualification supplies a better measured boundary.
const GL_ROOT_PARALLEL_MIN_WORK: u128 = 512 * 512 * 256;

/// Explicit root schedule resolved by the owning precompute thread.
///
/// Its representation is private so callers cannot manufacture a parallel
/// schedule that bypasses [`plan_gl_precompute`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlRootSchedule {
    min_task_len: Option<usize>,
}

impl GlRootSchedule {
    pub(crate) const fn serial() -> Self {
        Self { min_task_len: None }
    }

    fn parallel(min_task_len: usize) -> Self {
        Self {
            min_task_len: Some(min_task_len.max(1)),
        }
    }

    pub fn is_parallel(self) -> bool {
        self.min_task_len.is_some()
    }

    pub fn label(self) -> &'static str {
        if self.is_parallel() {
            "root_parallel"
        } else {
            "root_serial"
        }
    }

    #[cfg(feature = "hp")]
    pub(crate) fn parallel_min_task_len(self) -> Option<usize> {
        self.min_task_len
    }

    #[cfg(all(test, feature = "hp"))]
    pub(crate) fn parallel_for_test(min_task_len: usize) -> Self {
        Self::parallel(min_task_len)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GlPrecomputeKind {
    ParallelTables,
    PlannedRoots {
        precision_bits: u32,
        rayon_workers: usize,
    },
    Serial,
}

/// One-level-only scheduling plan for a batch of Gauss--Legendre tables.
/// Its private representation prevents callers from bypassing policy
/// validation and the no-nesting planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlPrecomputePlan {
    kind: GlPrecomputeKind,
}

impl GlPrecomputePlan {
    /// Stable diagnostic label used in performance reports.
    pub fn label(self) -> &'static str {
        match self.kind {
            GlPrecomputeKind::ParallelTables => "table_parallel_root_serial",
            GlPrecomputeKind::PlannedRoots { .. } => "table_serial_planned_roots",
            GlPrecomputeKind::Serial => "table_serial_root_serial",
        }
    }

    /// Root schedule for one table. This uses only immutable plan state and
    /// never re-reads the thread-local runtime policy on a Rayon worker.
    pub fn root_schedule(self, table_order: usize) -> GlRootSchedule {
        let GlPrecomputeKind::PlannedRoots {
            precision_bits,
            rayon_workers,
        } = self.kind
        else {
            return GlRootSchedule::serial();
        };
        if gl_root_work(table_order, precision_bits) < GL_ROOT_PARALLEL_MIN_WORK {
            return GlRootSchedule::serial();
        }
        let target_tasks = rayon_workers.saturating_mul(4).max(1);
        let min_task_len = (table_order / target_tasks).max(8).min(table_order);
        GlRootSchedule::parallel(min_task_len)
    }
}

fn gl_root_work(table_order: usize, precision_bits: u32) -> u128 {
    let order = table_order as u128;
    order
        .saturating_mul(order)
        .saturating_mul(u128::from(precision_bits))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HpRuntimeError {
    message: String,
}

impl Display for HpRuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HpRuntimeError {}

/// Default full-parallel execution. This compatibility wrapper performs no
/// environment reads and introduces no platform-dependent policy.
pub fn run_hp<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    f()
}

/// Execute HP work under one explicit, validated scheduling policy.
pub fn run_hp_with_policy<F, R>(policy: &HpRuntimePolicy, f: F) -> Result<R, HpRuntimeError>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    policy.validate().map_err(|error| HpRuntimeError {
        message: error.to_string(),
    })?;
    if policy.parallel_gl_roots && !gl_root_parallel_platform_supported() {
        return Err(HpRuntimeError {
            message: "GL root parallelism requires native Linux and is disabled on WSL, Windows, and macOS because concurrent GMP allocation has not been qualified there"
                .to_owned(),
        });
    }
    match policy.mode {
        HpRuntimeMode::FullParallel => Ok(with_active_policy(policy, f)),
        HpRuntimeMode::SafeCapped => {
            let threads = policy.worker_threads.expect("validated safe policy");
            let stack = policy.stack_bytes.expect("validated safe policy");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(stack)
                .build()
                .map_err(|error| HpRuntimeError {
                    message: format!("safe-mode HP rayon pool build failed: {error}"),
                })?;
            std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .stack_size(stack)
                    .spawn_scoped(scope, || pool.install(|| with_active_policy(policy, f)))
                    .map_err(|error| HpRuntimeError {
                        message: format!("safe-mode HP outer thread spawn failed: {error}"),
                    })?
                    .join()
                    .map_err(|_| HpRuntimeError {
                        message: "safe-mode HP outer thread panicked".to_owned(),
                    })
            })
        }
    }
}

fn gl_root_parallel_platform_supported() -> bool {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WSL_INTEROP").is_some()
            || std::env::var_os("WSL_DISTRO_NAME").is_some()
        {
            return false;
        }
        for path in ["/proc/sys/kernel/osrelease", "/proc/version"] {
            if std::fs::read_to_string(path).is_ok_and(|value| {
                let value = value.to_ascii_lowercase();
                value.contains("microsoft") || value.contains("wsl")
            }) {
                return false;
            }
        }
        true
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Bind the exact runtime policy into provenance before any HP work begins,
/// then execute under that same policy.
pub fn run_hp_with_provenance<F, R>(
    policy: &HpRuntimePolicy,
    full_parallel_threads: usize,
    provenance: &mut xc_core::SolverProvenance,
    f: F,
) -> Result<R, HpRuntimeError>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    provenance
        .record_hp_runtime_policy(policy.clone(), full_parallel_threads)
        .map_err(|error| HpRuntimeError {
            message: error.to_string(),
        })?;
    run_hp_with_policy(policy, f)
}

fn with_active_policy<F, R>(policy: &HpRuntimePolicy, f: F) -> R
where
    F: FnOnce() -> R,
{
    struct RestorePolicy(Option<HpRuntimePolicy>);
    impl Drop for RestorePolicy {
        fn drop(&mut self) {
            ACTIVE_POLICY.with(|active| {
                active.replace(self.0.take());
            });
        }
    }
    let previous = ACTIVE_POLICY.with(|active| active.replace(Some(policy.clone())));
    let _restore = RestorePolicy(previous);
    f()
}

/// Map precomputation tasks according to the active explicit policy. Calls
/// outside [`run_hp_with_policy`] use the documented full-parallel default.
pub fn map_gl_precompute<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    let sequential = ACTIVE_POLICY.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|policy| policy.sequential_precompute)
    });
    if sequential {
        items.iter().map(&f).collect()
    } else {
        items.par_iter().map(f).collect()
    }
}

/// Resolve the whole Gauss--Legendre batch schedule once on its owning thread.
///
/// An active policy is thread-local and is intentionally read only here,
/// before any Rayon dispatch. The returned plan must be passed explicitly to
/// numerical work executed on workers.
pub fn plan_gl_precompute(table_orders: &[usize], precision_bits: u32) -> GlPrecomputePlan {
    let policy = ACTIVE_POLICY.with(|active| active.borrow().clone());
    if policy
        .as_ref()
        .is_some_and(|value| value.sequential_precompute)
    {
        return GlPrecomputePlan {
            kind: GlPrecomputeKind::Serial,
        };
    }

    let workers = rayon::current_num_threads().max(1);
    let roots_requested = policy.as_ref().is_some_and(|value| value.parallel_gl_roots);
    let underfilled = table_orders.len() < workers;
    let has_qualifying_table = table_orders
        .iter()
        .any(|&order| gl_root_work(order, precision_bits) >= GL_ROOT_PARALLEL_MIN_WORK);
    if roots_requested && underfilled && has_qualifying_table {
        GlPrecomputePlan {
            kind: GlPrecomputeKind::PlannedRoots {
                precision_bits,
                rayon_workers: workers,
            },
        }
    } else {
        GlPrecomputePlan {
            kind: GlPrecomputeKind::ParallelTables,
        }
    }
}

/// Execute one planned GL batch without nesting table- and root-level Rayon.
pub fn map_gl_precompute_planned<U, F>(
    table_orders: &[usize],
    plan: GlPrecomputePlan,
    f: F,
) -> Vec<U>
where
    U: Send,
    F: Fn(usize, GlRootSchedule) -> U + Sync + Send,
{
    use rayon::prelude::*;
    match plan.kind {
        GlPrecomputeKind::ParallelTables => table_orders
            .par_iter()
            .map(|&order| f(order, GlRootSchedule::serial()))
            .collect(),
        GlPrecomputeKind::PlannedRoots { .. } | GlPrecomputeKind::Serial => table_orders
            .iter()
            .map(|&order| f(order, plan.root_schedule(order)))
            .collect(),
    }
}

/// Active explicit HP runtime mode for process performance diagnostics.
pub fn active_runtime_mode_label() -> &'static str {
    ACTIVE_POLICY.with(
        |active| match active.borrow().as_ref().map(|policy| policy.mode) {
            Some(HpRuntimeMode::SafeCapped) => "safe_capped",
            Some(HpRuntimeMode::FullParallel) => "full_parallel",
            None => "default_full_parallel",
        },
    )
}

/// True only inside an explicitly selected safe-capped execution scope.
pub fn safe_mode() -> bool {
    ACTIVE_POLICY.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|policy| policy.mode == HpRuntimeMode::SafeCapped)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn explicit_safe_policy_changes_scheduling_not_mathematics() {
        let inputs = vec![5u64, 2, 9, 1, 7];
        let full = run_hp_with_policy(&HpRuntimePolicy::default(), || {
            assert!(!safe_mode());
            map_gl_precompute(&inputs, |value| value * value)
        })
        .unwrap();
        let safe_policy =
            HpRuntimePolicy::safe_capped(2, 8 * 1024 * 1024, "test-platform-hp-instability")
                .unwrap();
        let safe = run_hp_with_policy(&safe_policy, || {
            assert!(safe_mode());
            assert_eq!(rayon::current_num_threads(), 2);
            map_gl_precompute(&inputs, |value| value * value)
        })
        .unwrap();
        assert_eq!(safe, full);
        assert_eq!(safe, vec![25, 4, 81, 1, 49]);
    }

    #[test]
    fn default_wrapper_has_no_implicit_safe_mode() {
        assert_eq!(run_hp(|| 42), 42);
        assert!(!safe_mode());
    }

    #[test]
    fn absent_and_default_policies_keep_roots_serial() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(
                plan_gl_precompute(&[3000], 3386).label(),
                "table_parallel_root_serial"
            );
            run_hp_with_policy(&HpRuntimePolicy::default(), || {
                assert_eq!(
                    plan_gl_precompute(&[3000], 3386).label(),
                    "table_parallel_root_serial"
                );
            })
            .unwrap();
        });
    }

    #[test]
    fn planned_root_schedule_is_resolved_on_owner_and_passed_to_workers() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let policy = HpRuntimePolicy {
            parallel_gl_roots: true,
            ..HpRuntimePolicy::default()
        };
        pool.install(|| {
            with_active_policy(&policy, || {
                let underfilled = plan_gl_precompute(&[511, 512, 3000], 256);
                assert_eq!(underfilled.label(), "table_serial_planned_roots");
                let schedules =
                    map_gl_precompute_planned(&[511, 512, 3000], underfilled, |order, schedule| {
                        (order, schedule)
                    });
                assert_eq!(schedules[0], (511, GlRootSchedule::serial()));
                assert!(schedules[1].1.is_parallel());
                assert!(schedules[2].1.is_parallel());

                // Enough tables occupy the pool, so schedules observed inside
                // Rayon workers must remain explicitly root-serial.
                let filled = plan_gl_precompute(&[1000, 1001, 1002, 1003], 3386);
                assert_eq!(filled.label(), "table_parallel_root_serial");
                let observed =
                    map_gl_precompute_planned(&[1000, 1001, 1002, 1003], filled, |_, schedule| {
                        schedule
                    });
                assert!(observed
                    .iter()
                    .all(|schedule| *schedule == GlRootSchedule::serial()));
            });
        });
    }

    #[test]
    fn root_parallel_policy_fails_closed_outside_native_linux() {
        let policy = HpRuntimePolicy {
            parallel_gl_roots: true,
            ..HpRuntimePolicy::default()
        };
        let result = run_hp_with_policy(&policy, || ());
        assert_eq!(result.is_ok(), gl_root_parallel_platform_supported());
        if let Err(error) = result {
            assert!(error.to_string().contains("requires native Linux"));
        }
    }

    #[test]
    fn sequential_precompute_policy_is_serial_at_both_levels() {
        let policy =
            HpRuntimePolicy::safe_capped(2, 8 * 1024 * 1024, "test-platform-instability").unwrap();
        run_hp_with_policy(&policy, || {
            let plan = plan_gl_precompute(&[3000], 3386);
            assert_eq!(plan.label(), "table_serial_root_serial");
            assert_eq!(plan.root_schedule(3000), GlRootSchedule::serial());
        })
        .unwrap();
    }

    #[test]
    fn execution_helper_records_the_exact_safe_policy_before_work() {
        let policy =
            HpRuntimePolicy::safe_capped(2, 8 * 1024 * 1024, "test-platform-hp-instability")
                .unwrap();
        let mut provenance = xc_core::SolverProvenance::current_package("rug_mpfr");
        let result = run_hp_with_provenance(&policy, 8, &mut provenance, || 21 * 2).unwrap();
        assert_eq!(result, 42);
        assert_eq!(provenance.hp_runtime_policy, Some(policy));
        assert_eq!(provenance.thread_count, Some(2));
    }

    #[test]
    fn process_performance_report_includes_safe_thread_and_rayon_workers() {
        let report_path = std::env::temp_dir().join(format!(
            "xc-safe-performance-{}-{}.performance.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let policy =
            HpRuntimePolicy::safe_capped(2, 8 * 1024 * 1024, "test-platform-hp-instability")
                .unwrap();

        run_hp_with_policy(&policy, || {
            let _top = xc_core::performance_top_level_stage_at_path(
                &report_path,
                "test.safe_capped_outer",
                xc_core::PerformanceStageMetadata::default(),
            );
            rayon::join(
                || {
                    let _worker = xc_core::performance_stage_at_path(
                        &report_path,
                        "test.safe_capped_worker_a",
                        xc_core::PerformanceStageMetadata::default(),
                    );
                },
                || {
                    let _worker = xc_core::performance_stage_at_path(
                        &report_path,
                        "test.safe_capped_worker_b",
                        xc_core::PerformanceStageMetadata::default(),
                    );
                },
            );
        })
        .unwrap();

        let report: xc_core::PerformanceRunReport =
            serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
        for expected in [
            "test.safe_capped_outer",
            "test.safe_capped_worker_a",
            "test.safe_capped_worker_b",
        ] {
            assert!(
                report.records.iter().any(|record| record.stage == expected),
                "missing {expected} from process report"
            );
        }

        let _ = fs::remove_file(report_path);
    }
}
