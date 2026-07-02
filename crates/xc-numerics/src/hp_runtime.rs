// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Auto-configured HP execution context for WSL2 compatibility.
//!
//! ## Summary
//!
//! On non-WSL (Vast, native Linux, macOS, CI): `run_hp` calls `f()` directly.
//! Rayon's default global pool runs on all cores at full speed. **Zero
//! overhead, zero configuration change, identical to pre-v0.11.2
//! behavior.** Nothing in this module alters Vast/Linux performance —
//! every function here starts with an `is_wsl()` check and is a no-op when
//! it returns `false`.
//!
//! On WSL2: HP (GMP/MPFR) compute is routed through a small, capped rayon
//! setup instead of the default. This is necessary because the default
//! (rayon's global pool sized to all logical cores) reliably aborts the
//! process on WSL2 during HP-heavy work — see "What we confirmed" below.
//!
//! ## What we confirmed (empirically, this session)
//!
//! - **The abort is real and reproducible**: with rayon's default pool
//!   (`nproc` workers, 32 on a 32-core WSL2 box) doing dense-matrix HP
//!   linear algebra (LU factorization / inverse iteration) at matrix
//!   dimension ≳ 240, the process aborts (`exit 1`, no Rust panic, no
//!   backtrace — a glibc-level `abort()`, not a Rust-level failure).
//! - **It is NOT rayon-specific.** The identical workload reproduced with
//!   plain `std::thread::spawn` (zero rayon) at a high enough combined
//!   thread count. The mechanism is WSL2's kernel-level handling of
//!   concurrent GMP/glibc memory allocation across many threads, not a
//!   rayon defect. (Switching to a different threading/async library would
//!   not avoid it, since all of them ultimately issue the same glibc calls
//!   on the same kernel.)
//! - **It is thread-count sensitive, not stack-size or arena-limit
//!   sensitive.** Systematic sweeps varying `MALLOC_ARENA_MAX`, worker
//!   stack size (8 MB–2 GB), and outer-thread stack size found NO effect.
//!   Only reducing the *combined* number of concurrently active
//!   HP-compute threads reliably prevented the abort. On the 32-core WSL2
//!   box used for testing, ≤8 combined concurrently active threads was
//!   reliable; ≥16 was not.
//! - **`pool.install` (pool-member execution) is required, not just a
//!   large stack.** Calling `par_iter` from a thread that merely
//!   *participates* in a pool via work-stealing (the default behavior of
//!   calling `par_iter` from an arbitrary thread) behaves differently from
//!   calling it inside `pool.install(...)`, which makes the calling thread
//!   a full pool member. The pool-member path was the reliable one.
//! - **Two coexisting pools (an old, since-corrected design of this
//!   module) made the abort *more* likely**, presumably because it doubles
//!   the combined concurrently-active-thread count for the same nominal
//!   per-pool worker setting. The current design uses exactly one capped
//!   global pool plus one dedicated pool reused via `pool.install`, with a
//!   conservative combined worker budget.
//! - **GL (Gauss-Legendre node) compute at high npts/precision is SLOW on
//!   WSL2, not unstable.** An earlier working theory held that
//!   high-volume sequential/parallel GL compute was itself an additional
//!   abort source (tracked internally as "DEFECT-00y"). Direct,
//!   patient testing (waiting for full completion rather than polling with
//!   a short timeout) showed this was a **false positive**: a cold-cache
//!   GL precompute set at npts≈1200, prec≈1000 bits genuinely takes many
//!   minutes on WSL2 (single-digit minutes per table at the largest sizes),
//!   and short timeouts during manual testing were misread as crashes.
//!   There is no confirmed GL-compute instability distinct from the
//!   general thread-count-sensitive abort above.
//!
//! ## Design
//!
//! - `init_hp_pool()` caps rayon's *global* pool to a small worker count on
//!   WSL2 (no-op elsewhere). This matters because idle default-sized pools
//!   still count toward the combined active-thread budget once any HP work
//!   starts, and pure-f64 rayon callers (e.g. the Mellin critical-line
//!   scanner) use the global pool directly.
//! - `run_hp(f)` is the entry point wrapped around every public HP
//!   function. On WSL2 it runs `f` inside a large-stack `spawn_scoped`
//!   thread with `pool.install` routing all nested `par_iter` calls
//!   through one dedicated, reused pool (not a fresh pool per call — pool
//!   creation/teardown churn was avoided by storing it in a `OnceLock`).
//! - `map_gl_precompute(items, f)` runs `f` over `items` sequentially on
//!   WSL2, in parallel (via the global pool) elsewhere. GL tables are a
//!   one-time cost per `(npts, prec)` — cached to disk after first compute
//!   — so this trades a slower cold-cache WSL2 precompute for one fewer
//!   source of concurrent HP-thread activity. Given the "GL is slow, not
//!   unstable" finding above, this specific function's WSL2 benefit is
//!   unconfirmed; it is currently kept as defense-in-depth headroom under
//!   the general thread-count budget, at the cost of a slower cold-cache
//!   GL precompute on WSL2 (parallel across up to 4 workers vs. serial).
//!
//! ## Worker count: `nproc / 8`, clamped to `[2, 4]`
//!
//! Both the capped global pool and the dedicated HP pool run this many
//! workers each on WSL2 (combined budget = 2× this value). `nproc/8`
//! (max 4) gives a combined budget of 8 on a 32-core machine, which was
//! reliable in testing; 16 was not. This is a conservative, not
//! exhaustively-optimized, choice — see `XC_HP_THREADS` below to
//! experiment with a higher value on a specific machine.
//!
//! ## Env overrides (WSL2 only; no effect elsewhere)
//!
//! - `XC_HP_THREADS` — worker count per pool (default: `nproc/8` clamped
//!   to `[2, 4]`). Raising this is untested territory — the abort
//!   threshold was only bracketed to "somewhere between 8 and 16 combined
//!   workers" on one 32-core test machine, not precisely pinned, and may
//!   differ on other WSL2 configurations.
//! - `XC_HP_STACK_MB` — worker AND outer-thread stack size in MiB
//!   (default: 256). Stack size was not found to affect the abort in
//!   testing, but is kept configurable in case future workloads (e.g.
//!   larger matrices) need more.
//!
//! ## Vast / native Linux
//!
//! `is_wsl()` reads `/proc/version` once; on Vast it returns `false`, and
//! every public function in this module is then a zero-cost passthrough.

use std::sync::OnceLock;

// ----- constants -----------------------------------------------------------

/// Default per-worker AND outer-thread stack size for WSL2, in MiB.
const WSL_STACK_MB: usize = 256;

// ----- pool state (WSL2 only) -----------------------------------------------

static GLOBAL_CAP_INIT: OnceLock<()> = OnceLock::new();
static WSL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Cap the global rayon pool on WSL2 (no-op on non-WSL; idempotent; safe to
/// call multiple times or from multiple entry points — only runs once).
///
/// Also called internally by [`run_hp`]. Exposed as `pub` so pure-f64
/// rayon callers that never call `run_hp` (e.g. the Mellin critical-line
/// scanner) can still ensure the global pool is capped before their own
/// `par_iter` use.
pub fn init_hp_pool() {
    GLOBAL_CAP_INIT.get_or_init(|| {
        if !is_wsl() { return; }
        let threads = wsl_threads();
        let stack = wsl_stack_bytes();
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(stack)
            .build_global()
        {
            Ok(()) => eprintln!(
                "[xc-hp] WSL2: global rayon pool capped to {threads} workers, {}MB stack",
                stack >> 20
            ),
            Err(_) => { /* global pool already configured elsewhere; fine */ }
        }
    });
}

fn wsl_pool() -> &'static rayon::ThreadPool {
    WSL_POOL.get_or_init(|| {
        let threads = wsl_threads();
        let stack = wsl_stack_bytes();
        eprintln!(
            "[xc-hp] WSL2: dedicated HP pool = {threads} workers, {}MB stack",
            stack >> 20
        );
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(stack)
            .build()
            .expect("WSL2 HP rayon pool build failed")
    })
}

// ----- public API ------------------------------------------------------------

/// Run `f` in an HP-safe execution context.
///
/// **Non-WSL (Vast, native Linux, CI):** calls `f()` directly. Zero
/// overhead; rayon's default global pool (all cores) handles parallelism
/// exactly as it would without this wrapper.
///
/// **WSL2:** runs `f` inside a large-stack `spawn_scoped` thread, with
/// `pool.install` routing every nested `par_iter` call in `f` through the
/// single dedicated HP pool. Making the calling thread a pool member (via
/// `pool.install`, not plain external `par_iter`) was necessary to reach
/// a reliable configuration in testing — see the module docs.
pub fn run_hp<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    init_hp_pool();

    if !is_wsl() {
        return f();
    }

    let pool = wsl_pool();
    let stack = wsl_stack_bytes();

    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn_scoped(scope, || pool.install(f))
            .expect("WSL2 HP outer thread spawn failed")
            .join()
            .expect("WSL2 HP outer thread panicked")
    })
}

/// Map `items` through `f`: in parallel (global pool) on non-WSL, serially
/// on WSL2. See the module docs' "GL compute at high npts/precision" note —
/// this is defense-in-depth headroom under the WSL2 thread-count budget,
/// not a confirmed fix for a distinct instability. On WSL2 it trades a
/// slower cold-cache precompute (serial instead of parallel across up to 4
/// workers) for one fewer source of concurrent HP-thread activity.
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
    if is_wsl() {
        items.iter().map(|x| f(x)).collect()
    } else {
        items.par_iter().map(|x| f(x)).collect()
    }
}

/// Returns `true` when running under WSL2 (or WSL1).
/// Reads `/proc/version` once per call; a read error is treated as non-WSL
/// (i.e. behaves as native Linux — the safe default, since every WSL-only
/// behavior in this module is an extra restriction, never a relaxation).
pub fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|s| s.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

// ----- helpers ---------------------------------------------------------------

fn wsl_threads() -> usize {
    if let Some(n) = std::env::var("XC_HP_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return n.max(1);
    }
    // nproc/8, clamped to [2, 4]. See module docs: combined budget (this
    // value × 2, since both the global-cap pool and the dedicated HP pool
    // use it) was reliable at 8 and not at 16 on a 32-core WSL2 test
    // machine. Not exhaustively bracketed between those two points.
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (nproc / 8).clamp(2, 4)
}

fn wsl_stack_bytes() -> usize {
    std::env::var("XC_HP_STACK_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(WSL_STACK_MB)
        << 20
}
