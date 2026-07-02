// Regression-check / diagnostic harness for the WSL2 HP abort fixed in
// v0.11.2/v0.11.3 (see xc_numerics::hp_runtime module docs for the full
// writeup of what was confirmed and why). Fresh compute (CacheMode::Off)
// so the LU / inverse-iteration path is always exercised.
//
// With no env vars, this exercises the toolkit's built-in WSL2 handling
// (xc_numerics::hp_runtime::run_hp) exactly as the production ccm::hp::run
// entry point does — i.e. it should just pass on WSL2 with no configuration:
//
//   cargo run --release --features hp --example wsl_abort_repro
//
// The POOL_*/OUTER_STACK_MB knobs below bypass the toolkit's own WSL
// handling and build an ad hoc pool/thread instead — useful for
// re-diagnosing a *new* WSL2 issue (varying worker count / stack size by
// hand) without touching hp_runtime.rs itself:
//
//   LSQ=20 NM=140 DIG=210 POOL_THREADS=4 POOL_STACK_MB=512 \
//     ./target/release/examples/wsl_abort_repro
//
// Env knobs:
//   LSQ, NM, DIG         : config (default 5 / 40 / 90)
//   POOL_THREADS         : ad hoc rayon worker count (0/unset = skip, use hp_runtime)
//   POOL_STACK_MB        : ad hoc rayon worker stack in MiB (0/unset = std default)
//   OUTER_STACK_MB       : big-stack std::thread wrapping the whole run (0 = none)

#[cfg(feature = "hp")]
fn run_config() {
    use xc_spectral::ccm::{CcmParams, hp::{run, HighPrecConfig}};
    use xc_numerics::quadrature::CacheMode;
    let getenv = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let lsq = getenv("LSQ", 5);
    let n = getenv("NM", 40) as usize;
    let digits = getenv("DIG", 90) as u32;
    let params = CcmParams::from_lambda_sq_integer(lsq, n);
    let mut cfg = HighPrecConfig::for_decimal_digits(digits);
    cfg.cache_mode = CacheMode::Off;
    cfg.n_eigenvalues = 0;
    eprintln!("[repro] λ²={lsq} N={n} dim={} digits={} prec={} bits",
        2 * n + 1, digits, cfg.precision_bits);
    let res = run(&params, &cfg, &[]).expect("HP run");
    println!("OK dim={} log10|eps_N|={:.3}",
        2 * n + 1, res.weil_min_eigenvalue.clone().abs().log10().to_f64());
}

#[cfg(feature = "hp")]
fn main() {
    let uenv = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    let pool_threads = uenv("POOL_THREADS");
    let pool_stack_mb = uenv("POOL_STACK_MB");
    let outer_stack_mb = uenv("OUTER_STACK_MB");
    eprintln!("[repro] pool_threads={pool_threads} pool_stack_mb={pool_stack_mb} outer_stack_mb={outer_stack_mb}");

    let exec = move || {
        if pool_threads > 0 || pool_stack_mb > 0 {
            let mut b = rayon::ThreadPoolBuilder::new();
            if pool_threads > 0 { b = b.num_threads(pool_threads); }
            if pool_stack_mb > 0 { b = b.stack_size(pool_stack_mb << 20); }
            let pool = b.build().expect("pool build");
            pool.install(run_config);
        } else {
            run_config();
        }
    };

    if outer_stack_mb > 0 {
        std::thread::Builder::new()
            .stack_size(outer_stack_mb << 20)
            .spawn(exec).expect("spawn outer")
            .join().expect("outer join");
    } else {
        exec();
    }
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("requires --features hp");
    std::process::exit(1);
}
