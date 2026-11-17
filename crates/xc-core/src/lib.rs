// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Stable, domain-independent contracts for Xcelerator Toolkit v0.13.0.
//!
//! This crate deliberately contains no numerical algorithms.  It defines the
//! typed language shared by operators, solvers, caches, certificates, and
//! domain modules so that a mathematical problem is not confused with the
//! algorithm used to solve it.

mod analytic_context;
mod archive;
mod artifact_plan;
mod assurance;
mod cache_location;
mod capability;
mod config;
mod config_resolution;
mod diagnostics;
mod error;
pub mod finite_json;
mod hex;
mod hypothesis;
mod optimization;
mod performance;
mod precision;
mod progress;
mod provenance;
mod publication;
mod research;
mod resource;
mod run_resources;
mod secret;
mod status;
mod subspace;
mod target;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use analytic_context::*;
pub use archive::*;
pub use artifact_plan::*;
pub use assurance::*;
pub use cache_location::*;
pub use capability::*;
pub use config::*;
pub use config_resolution::*;
pub use diagnostics::*;
pub use error::*;
pub use hex::{hex_lower, sha256_json};
pub use hypothesis::*;
pub use optimization::*;
pub use performance::*;
pub use precision::*;
pub use progress::*;
pub use provenance::*;
pub use publication::*;
pub use research::*;
pub use resource::*;
pub use run_resources::*;
pub use secret::*;
pub use status::*;
pub use subspace::*;
pub use target::*;

/// Install a complete sibling file atomically, retrying transient Windows sharing failures.
pub fn rename_replacement(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    retry_file_operation(|| std::fs::rename(source, destination))
}

/// Remove a caller-owned temporary file after readers with transient sharing
/// restrictions have released it.
pub fn remove_temporary_file(path: &std::path::Path) -> std::io::Result<()> {
    retry_file_operation(|| std::fs::remove_file(path))
}

fn retry_file_operation(mut operation: impl FnMut() -> std::io::Result<()>) -> std::io::Result<()> {
    #[cfg(windows)]
    for delay_ms in [5, 10, 20, 40, 80, 160, 320, 640, 1280] {
        match operation() {
            Err(e) if matches!(e.raw_os_error(), Some(5 | 32 | 33)) => {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
            result => return result,
        }
    }
    operation()
}

// Worker pools and spawned threads can restore the same intermediate files.
static CHECKPOINT_READ_BARRIERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// An explicit refresh or verification bypasses checkpoints on all worker threads.
pub fn execution_checkpoint_reads_enabled() -> bool {
    CHECKPOINT_READ_BARRIERS.load(std::sync::atomic::Ordering::Acquire) == 0
}

/// Overlapping refreshes keep the barrier active until their last callback returns.
pub fn with_execution_checkpoint_reads<T>(allowed: bool, compute: impl FnOnce() -> T) -> T {
    if allowed {
        return compute();
    }
    struct Barrier;
    impl Drop for Barrier {
        fn drop(&mut self) {
            CHECKPOINT_READ_BARRIERS.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
    }
    CHECKPOINT_READ_BARRIERS.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let _barrier = Barrier;
    compute()
}
