// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Archimedean Weil form on the Sonin-like (anti-band) subspace.
//!
//! The full-space archimedean (prime-free) Weil form is O(1)-indefinite.
//! Connes Thm 7.1 gives archimedean positivity on the Sonin space (the
//! orthocomplement of the band-concentrated / prolate subspace). This
//! example builds the time-frequency band-concentration operator
//! (`band_concentration_matrix_hp`), deflates its top `n_drop`
//! band-concentrated modes, and reports the archimedean spectrum on the
//! band-complement: whether it becomes positive there, and how its
//! smallest eigenvalue scales in λ².
//!
//! The band is set by a Shannon number `c_S` (the count of strongly
//! band-concentrated modes): Ω = π·c_S/(2a), a = ½ ln λ².
//!
//! HP-gated. Run on a high-precision Linux host (HP needs GMP/MPFR).
//!
//! Usage:
//!   cargo run --release -p xc-spectral --features hp --example weil_sonin -- <lambda_sq> <n_modes> <digits> <shannon_c> <n_drop>

#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use rug::Float;
    use xc_spectral::ccm::hp::{weil_spectrum_sonin_hp, HighPrecConfig};
    use xc_spectral::ccm::CcmParams;

    let args: Vec<String> = std::env::args().collect();
    let lambda_sq: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(13);
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(48);
    let digits: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(120);
    let shannon_c: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(4.0 * lambda_sq as f64);
    let n_drop: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(shannon_c.ceil() as usize);

    let params = CcmParams::from_lambda_sq_integer(lambda_sq, n);
    let mut cfg = HighPrecConfig::for_decimal_digits(digits);
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off; // hermetic
    let prec = cfg.precision_bits;

    // Ω = π·c_S/(2a), a = ½ ln λ².
    let a = Float::with_val(prec, lambda_sq).ln() / 2u32;
    let mut omega = Float::with_val(prec, rug::float::Constant::Pi);
    omega *= Float::with_val(prec, shannon_c);
    omega /= &a;
    omega /= 2u32;

    let t0 = std::time::Instant::now();
    let res = weil_spectrum_sonin_hp(&params, &cfg, &omega, n_drop)?;
    let dt = t0.elapsed().as_secs_f64();

    // χ transition: count χ > 1/2 (≈ Shannon number).
    let half = Float::with_val(prec, 0.5);
    let n_conc = res.chi.iter().filter(|c| **c > half).count();

    // smallest eigenvalue of the deflated (band-complement) archimedean form.
    let eps_min = res.spectrum.first().cloned().unwrap();
    let zero = Float::with_val(prec, 0);
    let smallest_pos = res.spectrum.iter().find(|e| **e > zero).cloned();

    let k = 4.0 * std::f64::consts::PI / std::f64::consts::LN_10;
    println!(
        "lambda_sq={} N={} dim={} prec={}b shannon_c={:.2} n_drop={} ({:.1}s)",
        lambda_sq, n, 2 * n + 1, prec, shannon_c, n_drop, dt
    );
    println!("  band-concentrated modes (chi>1/2) = {}", n_conc);
    println!("  restricted spectrum min           = {:.8e}", eps_min);
    match smallest_pos {
        Some(e) => {
            let dfloor = -Float::with_val(prec, e.clone().abs()).log10();
            println!("  smallest positive eigenvalue      = {:.8e}", e);
            println!("  -log10(smallest positive)         = {:.6}", dfloor);
            println!("  (4pi/ln10)*lambda_sq              = {:.6}", k * lambda_sq as f64);
        }
        None => println!("  (no positive eigenvalue: form still indefinite on this subspace)"),
    }
    Ok(())
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("build with --features hp");
}
