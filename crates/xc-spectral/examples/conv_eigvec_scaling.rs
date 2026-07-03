// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

// Measurement for the `conv` eigenvalue->zero conversion derivation
// (research/convergence/CONV_CONVERSION_DERIVATION.md §8.5).
//
// For a set of small λ², compute the HP ground eigenvector ξ (normalized
// Σξ_j = √L) and report the λ-scaling of the quantities that feed the
// conversion factor A = e_match/ε^∞:
//   - ‖ξ‖² = Σ ξ_j²                          (norm under the Σξ_j=√L convention)
//   - participation M_eff = (Σ|ξ_j|)² / Σξ_j²  (normalization-invariant mode count)
//   - support radius j99/j999 and bandwidth z = 2π j/L (99% / 99.9% of the ℓ² mass)
//   - |ξ̂'(γ₁)|                                (slope of ξ̂ at the first zero)
//
// The question: does the effective mode support / bandwidth carry a power
// of λ (edge T=2λ hypothesis ⇒ z* ~ λ, ‖ξ‖² ~ λ^{±1}) or only log(λ²)?
// That power sets P_m (5 vs 4.5 vs 5.5).
//
// Usage (via WSL2, per the local_build_env rule):
//   cargo run --release --features hp --example conv_eigvec_scaling

#[cfg(feature = "hp")]
fn main() {
    use xc_spectral::ccm::{CcmParams, hp::{run, HighPrecConfig}};
    use xc_numerics::quadrature::CacheMode;
    use rug::Float;
    use rug::float::Constant;

    // (lambda_sq, n_modes, digits). N ≳ 1.7·λ²lnλ² (past support);
    // digits ≳ 1.5·D_eig ≈ 1.5·(4π/ln10)λ² so the ground state is resolved.
    let configs = [
        (5u64, 40usize, 90u32),
        (8, 60, 120),
        (13, 90, 160),
        (20, 140, 210),
        (30, 200, 280),
    ];

    let gamma1_str = "14.134725141734693790457251983562470270784257115699";

    println!("# lambda_sq L lambda N digits negLog10_epsN sum_xi sqrtL sum_abs norm2 M_eff j99 z99 j999 z999 log10_absXihatPrimeG1");
    for &(lsq, n, digits) in &configs {
        let params = CcmParams::from_lambda_sq_integer(lsq, n);
        let mut cfg = HighPrecConfig::for_decimal_digits(digits);
        cfg.cache_mode = CacheMode::Off; // fresh, hermetic compute
        cfg.n_eigenvalues = 0;           // only want ξ + ε_N
        let prec = cfg.precision_bits;

        eprintln!("[run] λ²={lsq} N={n} digits={digits} (prec={prec} bits) ...");
        let res = run(&params, &cfg, &[]).expect("HP run should succeed");
        let xi = &res.xi; // Vec<Float>, length 2n+1, normalized Σξ_j=√L
        let nmax = n as i64;
        let idx = |j: i64| -> usize { (j + nmax) as usize };

        let pi = Float::with_val(prec, Constant::Pi);
        let l = Float::with_val(prec, lsq).ln(); // L = ln(λ²)
        let sqrt_l = l.clone().sqrt();

        // Sums.
        let mut sum_xi = Float::with_val(prec, 0);
        let mut sum_abs = Float::with_val(prec, 0);
        let mut norm2 = Float::with_val(prec, 0);
        for x in xi.iter() {
            sum_xi += x;
            sum_abs += x.clone().abs();
            let mut sq = x.clone(); sq *= x; norm2 += &sq;
        }
        // Participation M_eff = sum_abs^2 / norm2.
        let mut m_eff = sum_abs.clone(); m_eff *= &sum_abs; m_eff /= &norm2;

        // Cumulative ℓ² mass outward from j=0; find 99% / 99.9% radii.
        let mut t99 = norm2.clone(); t99 *= 0.99;
        let mut t999 = norm2.clone(); t999 *= 0.999;
        let mut cum = { let x = &xi[idx(0)]; let mut s = x.clone(); s *= x; s };
        let (mut j99, mut j999) = (nmax, nmax);
        let (mut hit99, mut hit999) = (false, false);
        for j in 1..=nmax {
            for &jj in &[j, -j] {
                let x = &xi[idx(jj)];
                let mut sq = x.clone(); sq *= x; cum += &sq;
            }
            if !hit99 && cum >= t99 { j99 = j; hit99 = true; }
            if !hit999 && cum >= t999 { j999 = j; hit999 = true; }
        }
        let z_of = |j: i64| -> f64 {
            let mut z = pi.clone(); z *= 2; z *= j; z /= &l; z.to_f64()
        };

        // |ξ̂'(γ₁)|:  ξ̂(z)=2L^{-1/2} sin(zL/2) g(z), g(z)=Σ ξ_j/(z−2πj/L).
        // ξ̂'(z)=2L^{-1/2}[ (L/2)cos(zL/2) g + sin(zL/2) g' ], g'=−Σ ξ_j/(z−2πj/L)².
        let z = Float::with_val(prec, Float::parse(gamma1_str).unwrap());
        let mut g = Float::with_val(prec, 0);
        let mut gp = Float::with_val(prec, 0);
        for j in -nmax..=nmax {
            let mut lattice = pi.clone(); lattice *= 2; lattice *= j; lattice /= &l;
            let mut d = z.clone(); d -= &lattice;
            let xij = &xi[idx(j)];
            let mut term = xij.clone(); term /= &d; g += &term;
            let mut d2 = d.clone(); d2 *= &d;
            let mut term2 = xij.clone(); term2 /= &d2; gp -= &term2;
        }
        let mut arg = z.clone(); arg *= &l; arg /= 2; // zL/2
        let sin_a = arg.clone().sin();
        let cos_a = arg.clone().cos();
        let inv_sqrt_l = Float::with_val(prec, 1) / &sqrt_l;
        // (L/2)cos·g
        let mut ta = cos_a.clone(); ta *= &l; ta /= 2; ta *= &g;
        // sin·g'
        let mut tb = sin_a.clone(); tb *= &gp;
        let mut xihat_p = ta + tb; xihat_p *= 2; xihat_p *= &inv_sqrt_l;
        let log10_abs_xihat_p = xihat_p.clone().abs().log10().to_f64();

        // ε_N floor in digits.
        let neg_log10_eps = {
            let e = res.weil_min_eigenvalue.clone().abs();
            let lg = e.log10(); (-lg).to_f64()
        };

        let lambda = (lsq as f64).sqrt();
        println!(
            "{} {:.6} {:.6} {} {} {:.4} {:.6} {:.6} {:.6} {:.6e} {:.4} {} {:.4} {} {:.4} {:.4}",
            lsq, l.to_f64(), lambda, n, digits, neg_log10_eps,
            sum_xi.to_f64(), sqrt_l.to_f64(), sum_abs.to_f64(),
            norm2.to_f64(), m_eff.to_f64(),
            j99, z_of(j99), j999, z_of(j999), log10_abs_xihat_p
        );
    }
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("this example requires the `hp` feature (run via WSL2):\n  \
               cargo run --release --features hp --example conv_eigvec_scaling");
    std::process::exit(1);
}
