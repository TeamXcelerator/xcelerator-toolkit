// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

// Generate a genuine prolate-eigenvalue cache fixture by running the
// real HP prolate pipeline (`compute_k_lambda`). The eigenvalue
// spectrum comes straight out of `tridiag_eigenvalues_hp` and is saved
// by the toolkit's own cache writer, so the resulting file is
// guaranteed to pass the structural validation on load (count = N,
// ascending order, ground state ≈ 2π·λ²).
//
// Usage (from the toolkit root):
//
//   cargo run --release --features hp --example gen_prolate_eigvals_fixture -- <lambda_sq> <n_grid> <digits>
//
// Example: λ²=25, N=401, HP-64
//
//   cargo run --release --features hp --example gen_prolate_eigvals_fixture -- 25 401 64
//
// The fixture lands in `./data/prolate_eigvals_cache/` as both an
// uncompressed `.json` and a single `.json.zip` (CacheMode::JsonZip,
// local-only — no network). Harvest the `.json.zip` for the public repo.
//
// Note: λ² must be a positive integer (the cache key is only active for
// integer λ²); the example computes λ = √(λ²) in HP at the working
// precision so no f64 round-trip enters the operator build.

#[cfg(feature = "hp")]
fn main() {
    use xc_spectral::prolate::hp::compute_k_lambda;
    use xc_numerics::quadrature::CacheMode;
    use rug::Float;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <lambda_sq> <n_grid> <digits>", args[0]);
        std::process::exit(2);
    }
    let lambda_sq: u64 = args[1].parse().expect("lambda_sq must be a u64");
    let n_grid: usize = args[2].parse().expect("n_grid must be a usize");
    let digits: u32 = args[3].parse().expect("digits must be a u32");

    // Bits from digits, matching the toolkit's HP config convention
    // (digits * 3.322 + 16 guard bits), then a touch extra rounding.
    let prec: u32 = ((digits as f64) * 3.322).ceil() as u32 + 16;

    // λ = √(λ²) in HP at the working precision (no f64 round-trip).
    let lambda = Float::with_val(prec, lambda_sq).sqrt();

    // A small comparison-grid count is fine — we only need the cache
    // write, which happens during the eigenvalue stage regardless.
    let n_sample = 64usize;

    eprintln!(
        "[gen] running prolate pipeline for λ²={}, n_grid={}, digits={} (prec={} bits)...",
        lambda_sq, n_grid, digits, prec
    );
    // JsonZip: write .json + .json.zip locally, never touch the network.
    let res = compute_k_lambda(&lambda, n_grid, n_sample, prec, CacheMode::JsonZip)
        .expect("prolate compute should succeed");
    eprintln!(
        "[gen] done in {:.1}s. spectrum cached; h_0 eigenvalue ≈ {}.",
        res.elapsed_seconds,
        xc_numerics::fmt::display_hp(&res.eigenvalue_0, 8)
    );
    // n_grid is forced odd inside compute_k_lambda; report the actual N.
    let n_actual = if n_grid.is_multiple_of(2) { n_grid + 1 } else { n_grid };
    eprintln!(
        "[gen] fixture written under ./data/prolate_eigvals_cache/ \
         (lambda_sq{}_ngrid{}_prec{}.json[.zip])",
        lambda_sq, n_actual, prec
    );
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("this example requires the `hp` feature: \
               cargo run --features hp --example gen_prolate_eigvals_fixture -- <lambda_sq> <n_grid> <digits>");
    std::process::exit(1);
}
