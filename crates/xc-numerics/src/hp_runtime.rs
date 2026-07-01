// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! WSL-aware HP execution context.
//!
//! On WSL2 the default rayon pool (32 workers, 8 MB stack) aborts during
//! GMP-heavy compute via two independent mechanisms:
//!
//! 1. **Thread-count limit** (LU / inverse-iteration): ~32 concurrent
//!    GMP workers exhaust the WSL `vm.max_map_count` / glibc-arena limit.
//!    Fixed by capping worker count to ~4.
//! 2. **Participating-caller stack** (large-dim LU): `rayon::pool.install`
//!    makes the calling thread run tasks; the default 8 MB main-thread
//!    stack overflows at dim ≥ ~240. Fixed by running inside a large-stack
//!    `std::thread`.
//! 3. **Worker stack depth** (GL-node Newton): each rayon worker computing
//!    a GL table needs more stack than the 8 MB default. Fixed by setting
//!    a large `stack_size` on the pool.
//!
//! On Vast (native Linux) and CI: **zero overhead** — `run_hp` returns
//! immediately if WSL is not detected, with no thread spawn and no pool
//! creation.
//!
//! ## Env overrides (WSL only)
//!
//! - `XC_HP_THREADS` — rayon worker count (default 4)
//! - `XC_HP_STACK_MB` — worker AND outer-thread stack in MiB (default 256)
//!
//! ## Usage
//!
//! Wrap the top-level HP entry point once:
//!
//! ```rust,ignore
//! pub fn run(params: &CcmParams, cfg: &HighPrecConfig, seeds: &[Float])
//!     -> Result<HighPrecResult>
//! {
//!     xc_numerics::hp_runtime::run_hp(|| run_inner(params, cfg, seeds))
//! }
//! ```
//!
//! All rayon calls within `run_inner` — including nested `par_iter` inside
//! LU, GL precompute, and eigenvector routines — automatically use the
//! WSL-safe local pool because `rayon::ThreadPool::install` routes all
//! work spawned within the closure through that pool.

use std::sync::OnceLock;

static GLOBAL_POOL_INIT: OnceLock<()> = OnceLock::new();
// On WSL, a single long-lived local pool prevents arena exhaustion from
// creating a new pool on every run_hp() call (old pool threads linger briefly).
static WSL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Configure the global rayon pool for WSL2 on first call (no-op elsewhere).
/// Caps workers to `XC_HP_THREADS` (default 4) so all rayon users in the
/// process — including f64 scans in mellin.rs and other non-HP paths — stay
/// within the WSL arena limit.
///
/// Also called by [`run_hp`]; exposed as `pub` so that pure-f64 rayon code
/// (e.g. `scan_critical_line_zeros_f64`) can also trigger the cap before
/// the global pool is first used, even when no HP work precedes it.
pub fn init_global_pool_for_wsl() {
    GLOBAL_POOL_INIT.get_or_init(|| {
        if !is_wsl() { return; }
        let threads = wsl_threads();
        let stack = wsl_stack_bytes();
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(stack)
            .build_global()
        {
            Ok(()) => eprintln!(
                "[xc-hp] WSL2: global rayon pool = {} workers, {}MB stack",
                threads, stack >> 20
            ),
            Err(_) => {
                // Already configured by the caller or a previous call —
                // silently fine. Our local pool still enforces the settings
                // for all HP work spawned via run_hp().
            }
        }
    });
}

/// Default worker count for the WSL HP rayon pool.
const WSL_HP_THREADS: usize = 2;

/// Default worker AND outer-thread stack in MiB for WSL.
const WSL_HP_STACK_MB: usize = 256;

/// Returns `true` when running under WSL2 (or WSL1).
/// Reads `/proc/version` once; any read error is treated as non-WSL.
pub fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|s| s.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn wsl_threads() -> usize {
    std::env::var("XC_HP_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(WSL_HP_THREADS)
}

fn wsl_stack_bytes() -> usize {
    let mb = std::env::var("XC_HP_STACK_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(WSL_HP_STACK_MB);
    mb << 20
}

/// Run `f` in a WSL-safe HP execution context.
///
/// **Non-WSL (Vast, native Linux, CI):** calls `f()` directly — no thread
/// spawn, no pool creation, zero overhead. Parallelism is unchanged.
///
/// **WSL2:** spawns a large-stack `std::thread` (scoped, so borrows in
/// `f` are valid) and routes all rayon work within `f` through a local
/// pool with a modest worker count and large per-worker stack. Both the
/// outer (caller) thread and the pool workers therefore have adequate
/// stack for GMP-heavy dense compute.
///
/// Because a *local* pool is used (not `build_global`), this does not
/// mutate global rayon state and is safe to call from test harnesses,
/// examples, and concurrent callers.
pub fn run_hp<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    // On first call, cap the global rayon pool on WSL so ALL rayon users
    // in the process stay within the WSL arena/thread limit.
    init_global_pool_for_wsl();

    if !is_wsl() {
        return f();
    }

    let stack = wsl_stack_bytes();

    // On WSL, use a single long-lived pool (created once, then reused) so
    // that successive run_hp() calls don't spawn new pools whose lingering
    // threads accumulate arena slots and exhaust the WSL limit.
    let pool = WSL_POOL.get_or_init(|| {
        let threads = wsl_threads();
        eprintln!(
            "[xc-hp] WSL2: HP pool = {} workers, {}MB stack (shared, long-lived)",
            threads, stack >> 20
        );
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(stack)
            .build()
            .expect("WSL2 HP rayon pool build failed")
    });

    // Run on a large-stack scoped thread so the participating caller also
    // has adequate stack for large-dim LU. `spawn_scoped` borrows `f` and
    // `pool` from the surrounding frame — no 'static requirement.
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn_scoped(scope, || pool.install(f))
            .expect("WSL2 HP outer thread spawn failed")
            .join()
            .expect("WSL2 HP outer thread panicked")
    })
}
