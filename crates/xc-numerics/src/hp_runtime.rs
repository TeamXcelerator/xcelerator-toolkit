// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Optional safe-mode HP execution context (opt-in; default is full,
//! uncapped parallelism identical to Vast/native Linux on every platform,
//! including WSL2).
//!
//! ## Summary
//!
//! By default, `run_hp(f)` just calls `f()` directly and `map_gl_precompute`
//! always runs in parallel on rayon's default global pool. This is true on
//! every platform — Vast, native Linux, macOS, CI, and WSL2. **Zero
//! overhead, zero configuration, full parallelism everywhere.**
//!
//! Set `XC_HP_SAFE_MODE=1` to opt into a capped-pool / sequential-GL
//! execution context instead. This exists as a documented escape hatch for
//! anyone who hits HP-compute instability on a constrained or unusual
//! environment — see "History" below for why it existed as the default in
//! v0.11.2–v0.11.4, and why it no longer is.
//!
//! ## History
//!
//! v0.11.2 through v0.11.4 auto-detected WSL2 (`/proc/version` containing
//! "microsoft") and *unconditionally* routed HP compute through a capped
//! rayon pool (`nproc/8` workers, clamped to 2–4) plus sequential GL
//! precompute, because early testing found that rayon's default pool
//! (sized to all logical cores) reliably aborted the process (`exit 1`, no
//! panic, no backtrace — a glibc-level `abort()`) on one 32-core WSL2 test
//! machine during dense HP linear algebra (LU factorization / inverse
//! iteration) at matrix dimension ≳ 240. That investigation is preserved
//! below since the mechanism findings (thread-count sensitivity, not
//! rayon-specific, not stack/arena-size sensitive) may still matter if the
//! abort resurfaces.
//!
//! **2026-07-02: the abort was retested and no longer reproduces**, on the
//! same machine, with no WSL update and no environment change in between.
//! Both the exact originally-documented aborting configuration (λ²=20,
//! N=140, dim 281) and a substantially harder one (λ²=30, N=230, dim 461)
//! completed cleanly, repeatedly (8/8 runs total: 2 at dim 281, 6 at dim
//! 461), running the *exact* code path this module now uses by default —
//! full uncapped rayon global pool, parallel GL precompute, zero capping.
//! Given that, unconditionally capping WSL2 by default is no longer
//! justified: it imposes a real performance cost (2–4 workers instead of
//! all cores) for a failure mode that no longer reproduces. The capped path
//! is kept as an opt-in (`XC_HP_SAFE_MODE=1`) rather than removed outright,
//! in case the original abort was environment-specific in a way that
//! recurs on a different machine or a different WSL2 build.
//!
//! ## What was confirmed during the original investigation (kept for
//! reference; applies only when `XC_HP_SAFE_MODE=1` is set)
//!
//! - **The abort was real and reproducible at the time**: with rayon's
//!   default pool (`nproc` workers, 32 on the test machine) doing
//!   dense-matrix HP linear algebra at matrix dimension ≳ 240, the process
//!   aborted (`exit 1`, no Rust panic, no backtrace — a glibc-level
//!   `abort()`, not a Rust-level failure).
//! - **It was NOT rayon-specific.** The identical workload reproduced with
//!   plain `std::thread::spawn` (zero rayon) at a high enough combined
//!   thread count.
//! - **It was thread-count sensitive, not stack-size or arena-limit
//!   sensitive.** Sweeps varying `MALLOC_ARENA_MAX`, worker stack size
//!   (8 MB–2 GB), and outer-thread stack size found no effect. Only
//!   reducing the combined number of concurrently active HP-compute
//!   threads prevented the abort (≤8 combined was reliable; ≥16 was not,
//!   on that one 32-core machine).
//! - **`pool.install` (pool-member execution) was required, not just a
//!   large stack**, to reach a reliable capped configuration.
//! - **GL (Gauss-Legendre node) compute at high npts/precision is SLOW,
//!   not unstable.** An earlier working theory that high-volume GL compute
//!   was itself an abort source (tracked internally as "DEFECT-00y") was a
//!   **false positive** from short-timeout polling of genuinely
//!   multi-minute cold-cache compute — there is no confirmed GL-compute
//!   instability distinct from the thread-count-sensitive abort above.
//!
//! ## Safe-mode design (`XC_HP_SAFE_MODE=1`)
//!
//! - `init_hp_pool()` caps rayon's *global* pool to a small worker count.
//!   This matters because idle default-sized pools still count toward the
//!   combined active-thread budget once any HP work starts, and pure-f64
//!   rayon callers (e.g. the Mellin critical-line scanner) use the global
//!   pool directly.
//! - `run_hp(f)` runs `f` inside a large-stack `spawn_scoped` thread with
//!   `pool.install` routing all nested `par_iter` calls through one
//!   dedicated, reused pool (not a fresh pool per call).
//! - `map_gl_precompute(items, f)` runs `f` over `items` sequentially
//!   instead of in parallel, as additional headroom under the thread-count
//!   budget.
//! - Worker count: `nproc/8`, clamped to `[2, 4]`. Overridable via
//!   `XC_HP_THREADS`.
//! - Stack size: 256 MiB default (worker and outer thread). Overridable via
//!   `XC_HP_STACK_MB`.
//!
//! ## Env vars
//!
//! - `XC_HP_SAFE_MODE=1` — opt into the capped-pool / sequential-GL
//!   execution context described above, on any platform. Default: unset
//!   (full uncapped parallelism, identical to pre-v0.11.2 behavior).
//! - `XC_HP_THREADS` — safe-mode worker count per pool (default: `nproc/8`
//!   clamped to `[2, 4]`). No effect unless `XC_HP_SAFE_MODE=1`.
//! - `XC_HP_STACK_MB` — safe-mode worker AND outer-thread stack size in
//!   MiB (default: 256). No effect unless `XC_HP_SAFE_MODE=1`.

use std::sync::OnceLock;

// ----- constants -----------------------------------------------------------

/// Default per-worker AND outer-thread stack size for safe mode, in MiB.
const SAFE_MODE_STACK_MB: usize = 256;

// ----- pool state (safe mode only) ------------------------------------------

static GLOBAL_CAP_INIT: OnceLock<()> = OnceLock::new();
static SAFE_MODE_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Cap the global rayon pool when safe mode is enabled (no-op otherwise;
/// idempotent; safe to call multiple times or from multiple entry points —
/// only runs once).
///
/// Also called internally by [`run_hp`]. Exposed as `pub` so pure-f64
/// rayon callers that never call `run_hp` (e.g. the Mellin critical-line
/// scanner) can still ensure the global pool is capped before their own
/// `par_iter` use, when safe mode is on.
pub fn init_hp_pool() {
    GLOBAL_CAP_INIT.get_or_init(|| {
        if !safe_mode() { return; }
        let threads = safe_mode_threads();
        let stack = safe_mode_stack_bytes();
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(stack)
            .build_global()
        {
            Ok(()) => eprintln!(
                "[xc-hp] XC_HP_SAFE_MODE: global rayon pool capped to {threads} workers, {}MB stack",
                stack >> 20
            ),
            Err(_) => { /* global pool already configured elsewhere; fine */ }
        }
    });
}

fn safe_mode_pool() -> &'static rayon::ThreadPool {
    SAFE_MODE_POOL.get_or_init(|| {
        let threads = safe_mode_threads();
        let stack = safe_mode_stack_bytes();
        eprintln!(
            "[xc-hp] XC_HP_SAFE_MODE: dedicated HP pool = {threads} workers, {}MB stack",
            stack >> 20
        );
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(stack)
            .build()
            .expect("safe-mode HP rayon pool build failed")
    })
}

// ----- public API ------------------------------------------------------------

/// Run `f` in an HP execution context.
///
/// **Default (`XC_HP_SAFE_MODE` unset):** calls `f()` directly. Zero
/// overhead; rayon's default global pool (all cores) handles parallelism
/// exactly as it would without this wrapper. Identical on every platform,
/// including WSL2.
///
/// **`XC_HP_SAFE_MODE=1`:** runs `f` inside a large-stack `spawn_scoped`
/// thread, with `pool.install` routing every nested `par_iter` call in `f`
/// through a single dedicated, capped HP pool. See the module docs for why
/// this exists and when you might want it.
pub fn run_hp<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    init_hp_pool();

    if !safe_mode() {
        return f();
    }

    let pool = safe_mode_pool();
    let stack = safe_mode_stack_bytes();

    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn_scoped(scope, || pool.install(f))
            .expect("safe-mode HP outer thread spawn failed")
            .join()
            .expect("safe-mode HP outer thread panicked")
    })
}

/// Map `items` through `f`: in parallel (global pool) by default; serially
/// when `XC_HP_SAFE_MODE=1`, as additional headroom under the safe-mode
/// thread-count budget.
///
/// `f` must be `Sync` even on the serial branch, to keep one signature
/// for both paths.
pub fn map_gl_precompute<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    use rayon::prelude::*;
    if safe_mode() {
        items.iter().map(|x| f(x)).collect()
    } else {
        items.par_iter().map(|x| f(x)).collect()
    }
}

/// Returns `true` when `XC_HP_SAFE_MODE=1` is set. This is the only gate in
/// this module — there is no automatic platform detection. Default (unset,
/// or any other value) is `false`, meaning full uncapped parallelism on
/// every platform.
pub fn safe_mode() -> bool {
    std::env::var("XC_HP_SAFE_MODE").as_deref() == Ok("1")
}

// ----- helpers ---------------------------------------------------------------

fn safe_mode_threads() -> usize {
    if let Some(n) = std::env::var("XC_HP_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return n.max(1);
    }
    // nproc/8, clamped to [2, 4]. See module docs: combined budget (this
    // value × 2, since both the global-cap pool and the dedicated HP pool
    // use it) was reliable at 8 and not at 16 on the original 32-core WSL2
    // test machine. Not exhaustively bracketed between those two points.
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (nproc / 8).clamp(2, 4)
}

fn safe_mode_stack_bytes() -> usize {
    std::env::var("XC_HP_STACK_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(SAFE_MODE_STACK_MB)
        << 20
}
