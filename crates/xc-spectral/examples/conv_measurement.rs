// Local WSL fix proof + conv(λ²) = D_eig − D_match measurement.
//
// For each (λ², N, P) config below, runs the full HP CCM pipeline via
// ccm::hp::run (exercising the WSL2 fix end-to-end), then reports:
//
//   D_match   = -log10(|ν₁ − γ₁| / |γ₁|)   [measured matching digits]
//   D_eig     = K·λ² − (9/2)·log₁₀(λ²) − C0  [derived eigenvalue floor]
//   conv      = D_eig − D_match               [the conversion gap]
//   ε_N       = smallest Weil eigenvalue       [plunge, for context]
//
// D_eig constants (from paper C / Fuchs / Connes):
//   K   = 4π/ln10  = 5.4575...
//   9/2 = 4.5      (Fuchs prolate index)
//   C0  = log10(2^14 · √2 · π^5 / 3) = 6.3736...
//
// The question being probed: does conv track ½·log₁₀(λ²) + const
// (conversion hypothesis, P_m = 5 asymptotically) or approach a constant
// (curvature hypothesis, P_m → 4.5)?  Over λ²=5→30 the slope difference
// is only ~0.4 digits in conv, so we need clean saturated-floor data.
//
// Usage (via WSL2; no env vars needed — the WSL fix is automatic):
//   cargo run --release --features hp --example conv_measurement

#[cfg(feature = "hp")]
fn main() {
    use xc_spectral::ccm::{CcmParams, hp::{run, HighPrecConfig}};
    use xc_numerics::quadrature::CacheMode;
    use rug::Float;

    // γ₁ to 65 digits (well above our measurement depths).
    let gamma1_str = "14.134725141734693790457251983562470270784257115699243175685567";

    // D_eig closed-form constants.
    let k = 4.0 * std::f64::consts::PI / std::f64::consts::LN_10; // 5.4575...
    let c0 = ((2.0_f64.powi(14) * 2.0_f64.sqrt() * std::f64::consts::PI.powi(5)) / 3.0).log10();
    let deig = |lsq: f64| -> f64 {
        k * lsq - (9.0 / 2.0) * lsq.log10() - c0
    };

    // Configs: (lambda_sq, n_modes, digits).
    // N is chosen ~3–4× Shannon number = λ²·ln(λ²) so the ramp is saturated.
    // P is chosen ~1.4× D_eig so the precision ceiling doesn't bind.
    let configs: &[(u64, usize, u32)] = &[
        (5,  40,  70),   // Shannon≈8,  D_eig≈27,  D_match≈14
        (8,  60, 110),   // Shannon≈17, D_eig≈44,  D_match≈30
        (13, 100, 160),  // Shannon≈33, D_eig≈71,  D_match≈56
        (20, 150, 220),  // Shannon≈60, D_eig≈96,  D_match≈87
        (30, 230, 310),  // Shannon≈102,D_eig≈147, D_match≈137
    ];

    println!("\n# conv(λ²) = D_eig − D_match measurement (WSL2 fix proof)");
    println!("#");
    println!("# λ²     N   P  D_match   D_eig   conv     ε_N(log10)  ½log₁₀(λ²)+3.0");
    println!("# ─────────────────────────────────────────────────────────────────────");

    for &(lsq, n, digits) in configs {
        let params = CcmParams::from_lambda_sq_integer(lsq, n);
        let mut cfg = HighPrecConfig::for_decimal_digits(digits);
        cfg.cache_mode = CacheMode::DynamicFetch; // use public cache if available
        cfg.n_eigenvalues = 1;
        let prec = cfg.precision_bits;

        let gamma1: Float = Float::with_val(prec,
            Float::parse(gamma1_str).expect("γ₁ parse"));
        let zero_seeds = vec![gamma1.clone()];

        eprintln!("[run] λ²={lsq} N={n} digits={digits} ...");
        let res = run(&params, &cfg, &zero_seeds).expect("HP run");

        // First eigenvalue (ν₁).
        let nu1 = match res.eigenvalues_pos.first()
            .and_then(|ev| ev.value()) {
            Some(v) => v.clone(),
            None => {
                println!("{lsq:>4} {n:>4} {digits:>4}  [NO EIGENVALUE CONVERGED]");
                continue;
            }
        };

        // D_match = −log₁₀(|ν₁ − γ₁| / |γ₁|)
        let mut err = nu1.clone(); err -= &gamma1;
        let mut rel_err = err.abs(); rel_err /= gamma1.clone().abs();
        let d_match = if rel_err.is_zero() {
            f64::INFINITY
        } else {
            -rel_err.log10().to_f64()
        };

        // D_eig from formula.
        let d_eig_val = deig(lsq as f64);

        // conv = D_eig − D_match.
        let conv_val = d_eig_val - d_match;

        // ε_N.
        let log10_eps = res.weil_min_eigenvalue.clone().abs().log10().to_f64();

        // Prediction from conversion hypothesis (½·log₁₀(λ²) + 3.0).
        let conv_pred = 0.5 * (lsq as f64).log10() + 3.0;

        println!("{lsq:>4} {n:>4} {digits:>4}  {d_match:8.3}  {d_eig_val:7.3}  {conv_val:7.3}   {log10_eps:10.3}  {conv_pred:7.3}");
    }

    println!("\n# Key: conv tracks ½·log₁₀(λ²)+3 ⟹ P_m=5 (conversion hypothesis)");
    println!("#      conv approaching constant ⟹ curvature hypothesis (P_m→4.5)");
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("requires --features hp (run via WSL2):\n  \
               cargo run --release --features hp --example conv_measurement");
    std::process::exit(1);
}
