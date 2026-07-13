// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Archimedean-only ground eigenvector of the localized Weil form
//! (prime-power sum dropped, `include_primes = false`).
//!
//! Emits the plunge eigenvalue and the full eigenvector ξ (coefficients
//! ordered `j = −N..+N`) as full-precision decimal strings, so the
//! archimedean-only coefficient tail can be fit — settling whether π/4 is
//! the real archimedean rate (pure exponential, no prime roughening).
//!
//! HP-gated. Run on a high-precision Linux host (HP needs GMP/MPFR).
//!
//! Usage:
//!   cargo run --release -p xc-spectral --features hp --example weil_arch_eigvec -- <lambda_sq> <n_modes> <digits>
//!
//! Output: JSON on stdout with fields lambda_sq, n_modes, precision_bits,
//! weil_min_eigenvalue (decimal string), prime_rayleigh (decimal string:
//! ρ_p = ⟨A_prime ψ, ψ⟩/‖ψ‖² on the even vector), xi (array of 2N+1 decimal
//! strings).

#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use xc_numerics::fmt::display_hp;
    use xc_spectral::ccm::hp::{weil_arch_eigvec_hp, HighPrecConfig};
    use xc_spectral::ccm::CcmParams;

    let args: Vec<String> = std::env::args().collect();
    let lambda_sq: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(13);
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(200);
    let digits: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1345);

    let params = CcmParams::from_lambda_sq_integer(lambda_sq, n);
    let mut cfg = HighPrecConfig::for_decimal_digits(digits);
    // Hermetic: pure compute, no cache read/write or network (same as
    // weil_cancellation).
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let prec = cfg.precision_bits;

    // Full-precision significant decimal digits for this working precision.
    let sig = ((prec as f64) * std::f64::consts::LOG10_2).floor() as usize;

    let (eps, rho_p, xi) = weil_arch_eigvec_hp(&params, &cfg)?;
    assert_eq!(xi.len(), 2 * n + 1, "xi length must be 2N+1");

    // JSON to stdout. xi ordered j = −N..+N (params.idx order, position 0 =
    // mode −N). display_hp emits full-precision scientific-notation decimal
    // strings (no characters needing JSON escaping).
    let xi_json: Vec<String> = xi.iter().map(|c| format!("\"{}\"", display_hp(c, sig))).collect();
    println!("{{");
    println!("  \"lambda_sq\": {:?},", lambda_sq as f64);
    println!("  \"n_modes\": {},", n);
    println!("  \"precision_bits\": {},", prec);
    println!("  \"weil_min_eigenvalue\": \"{}\",", display_hp(&eps, sig));
    println!("  \"prime_rayleigh\": \"{}\",", display_hp(&rho_p, sig));
    println!("  \"xi\": [{}]", xi_json.join(","));
    println!("}}");
    Ok(())
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("build with --features hp");
}
