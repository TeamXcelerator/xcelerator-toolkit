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
//!   zero scanners. Complex-zero validation is planned; the available scans
//!   return real-part crossings of the finite truncated transform.
//! - **`yakaboylu`**: Yakaboylu's Hilbert–Pólya framework. The
//!   `V_R(s, s')` matrix element, W-positivity tests on the critical
//!   line, indefiniteness on synthetic off-line zeros. f64 + HP
//!   tiers.
//! - **`lfunction`**: Dirichlet L-function character specs (`χ₃, χ₄,
//!   χ₅, χ₇`) and twisted prime-power enumeration. A generalized CCM
//!   assembly for nontrivial L-functions is planned, not implemented.
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

/// Lets exported macros reach `xc-core` without a caller dependency on it.
#[doc(hidden)]
pub use xc_core as __xc_core;

#[macro_export]
macro_rules! hp_debug {
    ($($arg:tt)*) => {
        if $crate::hp_debug_enabled() {
            $crate::__xc_core::progress_message!($($arg)*);
        }
    };
}

/// Crate-wide lock serializing all cwd-mutating cache tests.
///
/// Cargo runs tests in parallel within a single test binary, and the
/// current working directory is process-global (not per-thread). The
/// `ccm::hp` and `prolate::hp` cache tests both `set_current_dir` into
/// a throwaway temp dir and redirect the standalone cache root there;
/// if they used separate mutexes they would race each other (one test
/// deleting the temp dir another captured as its "original"). A single
/// crate-level lock guarantees mutual exclusion across both modules.
///
/// Gated on the `hp` feature: the only consumers are the HP-gated
/// cache tests, so without `hp` this would be unused.
#[cfg(all(test, feature = "hp"))]
pub(crate) static TEST_CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Test-only override of [`standalone_cache_root`], set by the cache-test
/// guards while they hold [`TEST_CWD_LOCK`].
#[cfg(all(test, feature = "hp"))]
pub(crate) static TEST_CACHE_ROOT: std::sync::Mutex<Option<std::path::PathBuf>> =
    std::sync::Mutex::new(None);

/// Root of the standalone HP caches (`tau_cache`, `weil_eigvec_cache`,
/// `prolate_eigvals_cache`): `$XC_CACHE_ROOT` when set, else the per-user
/// cache root shared with the managed cache. The working directory is never
/// used, so runs started inside a checkout do not write into it.
#[cfg(feature = "hp")]
pub(crate) fn standalone_cache_root() -> std::path::PathBuf {
    #[cfg(test)]
    if let Some(root) = TEST_CACHE_ROOT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
    {
        return root;
    }
    xc_core::configured_cache_root()
}

/// Points [`standalone_cache_root`] at `<dir>/data` for a cache test and
/// restores the previous value on drop.
#[cfg(all(test, feature = "hp"))]
pub(crate) struct TestCacheRoot(Option<std::path::PathBuf>);

#[cfg(all(test, feature = "hp"))]
impl TestCacheRoot {
    pub(crate) fn enter(dir: &std::path::Path) -> Self {
        let mut slot = TEST_CACHE_ROOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self(slot.replace(dir.join("data")))
    }
}

#[cfg(all(test, feature = "hp"))]
impl Drop for TestCacheRoot {
    fn drop(&mut self) {
        *TEST_CACHE_ROOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = self.0.take();
    }
}

/// Make a fresh, unique throwaway directory for a test.
///
/// The directory lives under `<OS temp>/xc-test/`, never inside the
/// checkout, and is removed when the returned guard is dropped (including
/// when the test panics). Keep the guard alive for the whole test.
///
/// Gated on the `hp` feature: only the HP-gated cache tests use it.
#[cfg(all(test, feature = "hp"))]
pub(crate) fn fresh_test_dir(tag: &str) -> xc_core::test_support::TestDir {
    xc_core::test_support::TestDir::new(tag)
}
