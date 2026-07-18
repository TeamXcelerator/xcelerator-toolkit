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
use xc_core::{HpRuntimeMode, HpRuntimePolicy};

thread_local! {
    static ACTIVE_POLICY: RefCell<Option<HpRuntimePolicy>> = const { RefCell::new(None) };
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
}
