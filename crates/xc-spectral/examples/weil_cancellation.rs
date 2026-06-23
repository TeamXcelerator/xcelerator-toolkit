// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Quantify the archimedean↔prime cancellation that produces the CCM
//! convergence floor.
//!
//! On the full plunge eigenvector ξ, the plunge decomposes as
//!   ε_N = ⟨A_arch ξ,ξ⟩/⟨ξ,ξ⟩ − ⟨A_prime ξ,ξ⟩/⟨ξ,ξ⟩ = arch − prime.
//! Both pieces are O(1)-or-larger; their difference is exponentially small,
//! so they agree to ≈ −log₁₀ε_N digits. Prints arch, prime, ε_N, the floor,
//! and the cancellation depth; across λ² this gives how arch/prime scale.
//!
//! HP-gated. Run on a high-precision Linux host (HP needs GMP/MPFR).
//!
//! Usage:
//!   cargo run --release -p xc-spectral --features hp --example weil_cancellation -- <lambda_sq> <n_modes> <digits>

#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use rug::Float;
    use xc_spectral::ccm::hp::{weil_plunge_cancellation_hp, HighPrecConfig};
    use xc_spectral::ccm::CcmParams;

    let args: Vec<String> = std::env::args().collect();
    let lambda_sq: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(13);
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(48);
    let digits: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(120);

    let params = CcmParams::from_lambda_sq_integer(lambda_sq, n);
    let mut cfg = HighPrecConfig::for_decimal_digits(digits);
    // Hermetic: pure compute, no cache read/write or network.
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let prec = cfg.precision_bits;

    let t0 = std::time::Instant::now();
    let d = weil_plunge_cancellation_hp(&params, &cfg)?;
    let dt = t0.elapsed().as_secs_f64();

    // cancellation depth = -log10|eps_n / arch|  (digits the two pieces agree to)
    let mut ratio = Float::with_val(prec, &d.eps_n);
    ratio /= &d.arch_rayleigh;
    let depth = -Float::with_val(prec, ratio.abs()).log10();
    let floor = -Float::with_val(prec, d.eps_n.clone().abs()).log10();

    println!(
        "lambda_sq={} N={} dim={} prec={}b ({:.1}s)",
        lambda_sq, n, 2 * n + 1, prec, dt
    );
    println!("  arch_rayleigh  = {:.14e}", d.arch_rayleigh);
    println!("  prime_rayleigh = {:.14e}", d.prime_rayleigh);
    println!("  eps_N          = {:.8e}", d.eps_n);
    println!("  D_Primes = -log10|eps_N|        = {:.6}", floor);
    println!("  cancellation depth (arch==prime) = {:.6}", depth);
    Ok(())
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("build with --features hp");
}
