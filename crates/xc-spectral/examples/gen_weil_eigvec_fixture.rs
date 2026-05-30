// Generate a genuine Weil-eigenvector (ξ) cache fixture by running the
// real HP CCM pipeline. Because the eigenvector comes straight out of
// `inverse_iteration`, the saved file is guaranteed to pass the
// toolkit's residual validation on load.
//
// Usage (from the toolkit root):
//
//   cargo run --release --features hp --example gen_weil_eigvec_fixture -- <lambda_sq> <n_modes> <digits>
//
// Example: λ²=13, N=10, HP-64
//
//   cargo run --release --features hp --example gen_weil_eigvec_fixture -- 13 10 64
//
// The fixture lands in `./data/weil_eigvec_cache/` as both an
// uncompressed `.json` and a single `.json.zip` (CacheMode::JsonZip,
// local-only — no network). Harvest the `.json.zip` for the public repo.

#[cfg(feature = "hp")]
fn main() {
    use xc_spectral::ccm::{CcmParams, hp::{run, HighPrecConfig}};
    use xc_numerics::quadrature::CacheMode;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <lambda_sq> <n_modes> <digits>", args[0]);
        std::process::exit(2);
    }
    let lambda_sq: u64 = args[1].parse().expect("lambda_sq must be a u64");
    let n_modes: usize = args[2].parse().expect("n_modes must be a usize");
    let digits: u32 = args[3].parse().expect("digits must be a u32");

    let params = CcmParams::from_lambda((lambda_sq as f64).sqrt(), n_modes);
    let mut cfg = HighPrecConfig::for_decimal_digits(digits);
    // Local-only cache: write .json + .json.zip, never touch the network.
    cfg.cache_mode = CacheMode::JsonZip;
    // No Newton seeds needed — we only want ξ and ε_N for the fixture.
    cfg.n_eigenvalues = 0;

    eprintln!(
        "[gen] running CCM HP pipeline for λ²={}, N={}, digits={} (prec={} bits)...",
        lambda_sq, n_modes, digits, cfg.precision_bits
    );
    let result = run(&params, &cfg, &[]).expect("HP run should succeed");
    eprintln!(
        "[gen] done in {:.1}s. ε_N (smallest Weil eigenvalue) computed; ξ has {} entries.",
        result.elapsed_seconds, result.xi.len()
    );
    eprintln!(
        "[gen] fixture written under ./data/weil_eigvec_cache/ \
         (weil_eigvec_lambda_sq{}_nmodes{}_prec{}.json[.zip])",
        lambda_sq, n_modes, cfg.precision_bits
    );
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("this example requires the `hp` feature: \
               cargo run --features hp --example gen_weil_eigvec_fixture -- <lambda_sq> <n_modes> <digits>");
    std::process::exit(1);
}
