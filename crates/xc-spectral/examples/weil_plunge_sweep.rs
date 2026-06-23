// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Weil-form plunge prefactor: full vs archimedean-only.
//!
//! Computes the smallest eigenvalues of the localized Weil-form `τ`-matrix
//! (= Suzuki's `A_a`, the CCM construction's operator) in two modes:
//!   - FULL  (archimedean + prime sum)   → `ε_N`, the CCM floor `−log₁₀ ε_N`
//!   - ARCH  (prime sum dropped, `w_p=0`) → archimedean-only plunge
//!
//! Purpose: test the §13 prefactor decomposition. If both share the same
//! prefactor power (slope of `−log₁₀ ε_N` vs `log₁₀ λ²`) and differ only in
//! the additive constant, that confirms the prime sum cannot shift the
//! power (`P = 9/2` archimedean). It also resolves the §13.4 question: is
//! the archimedean-only form positive (a plunge) or O(1)-indefinite?
//!
//! Usage:
//!   cargo run --release -p xc-spectral --features hp --example weil_plunge_sweep -- <lambda_sq> <n_modes> <digits>

#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use rug::Float;
    use xc_spectral::ccm::hp::{weil_spectrum_hp, HighPrecConfig};
    use xc_spectral::ccm::CcmParams;

    let args: Vec<String> = std::env::args().collect();
    let lambda_sq: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(13);
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(48);
    let digits: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(120);

    let params = CcmParams::from_lambda_sq_integer(lambda_sq, n);
    let cfg = HighPrecConfig::for_decimal_digits(digits);
    let prec = cfg.precision_bits;

    let report = |label: &str, eigs: &[Float]| {
        let zero = Float::with_val(prec, 0);
        let min = &eigs[0];
        let smallest_pos = eigs.iter().find(|e| **e > zero);
        println!("[{}] dim={} prec={}b", label, 2 * n + 1, prec);
        print!("   4 smallest:");
        for e in eigs.iter().take(4) {
            print!(" {:.4e}", e);
        }
        println!();
        println!("   min eig        = {:.6e}", min);
        match smallest_pos {
            Some(sp) => {
                let d = -Float::with_val(prec, sp).log10();
                println!("   smallest pos   = {:.6e}", sp);
                println!("   D_op=-log10(+) = {:.8}", d);
            }
            None => println!("   (no positive eigenvalue)"),
        }
    };

    println!("=== lambda_sq={} N={} ===", lambda_sq, n);
    let t0 = std::time::Instant::now();
    let full = weil_spectrum_hp(&params, &cfg, true)?;
    report("FULL", &full);
    println!("   ({:.1}s)", t0.elapsed().as_secs_f64());

    let t1 = std::time::Instant::now();
    let arch = weil_spectrum_hp(&params, &cfg, false)?;
    report("ARCH-only", &arch);
    println!("   ({:.1}s)", t1.elapsed().as_secs_f64());

    Ok(())
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("build with --features hp");
}
