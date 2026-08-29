// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Spectral methods for analytic number theory.
//!
//! Building blocks for spectral approaches to the Riemann Hypothesis
//! and related problems:
//!
//! - **`ccm`**: 2025 CCM construction. Weil
//!   quadratic form on the V_n basis, smallest-eigenvector computation,
//!   rational-function root extraction. f64 + HP tiers.
//! - **`prolate`**: Prolate-wave operator PW_λ on `[-λ, λ]`,
//!   Sturm-Liouville eigenfunctions, the ℰ map, comparison against
//!   ξ_λ for Lemma 7.2-style tests. HP eigenvalue spectrum is cached
//!   to disk for re-runs at the same `(λ², n_grid, prec)`.
//! - **`mellin`**: Truncated completed eta function `Λ_λ(s)` and
//!   `ξ`-weighted variants on the critical line, with parallelized
//!   zero scanners.
//! - **`yakaboylu`**: Yakaboylu's Hilbert–Pólya framework. The
//!   `V_R(s, s')` matrix element, W-positivity tests on the critical
//!   line, indefiniteness on synthetic off-line zeros. f64 + HP
//!   tiers.
//! - **`lfunction`**: Dirichlet L-function character specs (`χ₃, χ₄,
//!   χ₅, χ₇`) and twisted prime-power enumeration. Used to extend
//!   the CCM construction to non-trivial L-functions.
//!
//! The HP tier is gated behind the `hp` feature.

pub mod ccm;
pub mod deviation;
pub mod distance;
pub mod lfunction;
pub mod mellin;
pub mod prolate;
#[cfg(feature = "hp")]
pub mod screw;
pub mod target;
pub mod yakaboylu;

pub use xc_cache::OutputValidationClaim;

/// Debug logging macro controlled by [`set_hp_debug_logging`].
///
/// Use this instead of bare `eprintln!` for diagnostic/progress messages.
/// Disabled by default; the atomic check is inexpensive and does not alter
/// numerical decisions.
///
/// Usage: `hp_debug!("message {}", value);`
static HP_DEBUG_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Explicitly enables or disables diagnostic HP progress logging.
pub fn set_hp_debug_logging(enabled: bool) {
    HP_DEBUG_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[doc(hidden)]
pub fn hp_debug_enabled() -> bool {
    HP_DEBUG_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[macro_export]
macro_rules! hp_debug {
    ($($arg:tt)*) => {
        if $crate::hp_debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

/// Crate-wide lock serializing all cwd-mutating cache tests.
///
/// Cargo runs tests in parallel within a single test binary, and the
/// current working directory is process-global (not per-thread). The
/// `ccm::hp` and `prolate::hp` cache tests both `set_current_dir` into
/// a throwaway temp dir to isolate their `<cwd>/data/*_cache/` writes;
/// if they used separate mutexes they would race each other (one test
/// deleting the temp dir another captured as its "original"). A single
/// crate-level lock guarantees mutual exclusion across both modules.
///
/// Gated on the `hp` feature: the only consumers are the HP-gated
/// cache tests, so without `hp` this would be unused.
#[cfg(all(test, feature = "hp"))]
pub(crate) static TEST_CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Root for throwaway test directories, under the workspace `target/`
/// dir (not the OS temp dir). Keeping test scratch inside `target/`
/// means it is contained in the repo's build area and removed by
/// `cargo clean`, rather than scattering directories in `/tmp` or
/// `%TEMP%`. Resolved from `CARGO_MANIFEST_DIR` at compile time, so it
/// is correct regardless of the process's runtime cwd.
///
/// Gated on the `hp` feature: only the HP-gated cache tests use it.
#[cfg(all(test, feature = "hp"))]
pub(crate) fn test_tmp_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("test-tmp")
}

/// Make a fresh, unique throwaway directory under [`test_tmp_root`].
/// The `tag` plus a process-id + nanosecond suffix avoids clashes when
/// tests run in parallel or are re-run rapidly.
///
/// Gated on the `hp` feature: only the HP-gated cache tests use it.
#[cfg(all(test, feature = "hp"))]
pub(crate) fn fresh_test_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = test_tmp_root().join(format!("{}_{}_{}", tag, std::process::id(), nanos));
    std::fs::create_dir_all(&dir).expect("create test tmp dir");
    dir
}
