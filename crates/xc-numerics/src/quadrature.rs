// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Gauss-Legendre quadrature at f64 and high precision.
//!
//! The HP nodes/weights are computed via Newton iteration on Legendre
//! polynomials and cached to disk under `~/.cache/ccm_gl/` so they're
//! reused across runs at the same `(n_pts, precision_bits)`.

/// 64-point Gauss-Legendre quadrature on [a, b] at f64 precision.
/// For a configurable number of points, use `gauss_legendre_n`.
pub fn gauss_legendre_64<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    gauss_legendre_n(f, a, b, 64)
}

/// N-point Gauss-Legendre quadrature on [a, b] at f64 precision.
pub fn gauss_legendre_n<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = gl_nodes_weights(n);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0_f64;
    for i in 0..n {
        let x = mid + half * nodes[i];
        sum += weights[i] * f(x);
    }
    sum * half
}

fn gl_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    for k in 0..n {
        let mut x = ((4 * k + 3) as f64 * std::f64::consts::PI / (4 * n + 2) as f64).cos();
        for _ in 0..20 {
            let (pn, pn_prime) = legendre_p_deriv_f64(n, x);
            x -= pn / pn_prime;
        }
        let (_, pn_prime) = legendre_p_deriv_f64(n, x);
        nodes[k] = x;
        weights[k] = 2.0 / ((1.0 - x * x) * pn_prime * pn_prime);
    }
    (nodes, weights)
}

fn legendre_p_deriv_f64(n: usize, x: f64) -> (f64, f64) {
    if n == 0 { return (1.0, 0.0); }
    let mut p0 = 1.0_f64;
    let mut p1 = x;
    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
        p0 = p1;
        p1 = p_next;
    }
    let deriv = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, deriv)
}

// ===========================================================================
// High-precision Gauss-Legendre nodes and weights (with disk cache)
// ===========================================================================

#[cfg(feature = "hp")]
mod hp {
    use rug::{ops::Pow, Float};

    #[inline] fn fl(prec: u32, v: f64) -> Float { Float::with_val(prec, v) }
    #[inline] fn fl_i(prec: u32, v: i64) -> Float { Float::with_val(prec, v) }
    #[inline] fn pi(prec: u32) -> Float { Float::with_val(prec, rug::float::Constant::Pi) }

    /// Compute (or load from disk cache) Gauss-Legendre nodes and weights
    /// at the given precision in bits, for `n` points on `[-1, 1]`.
    pub fn gauss_legendre_nodes(n: usize, prec: u32) -> (Vec<Float>, Vec<Float>) {
        if let Some(cached) = load_gl_cache(n, prec) { return cached; }
        let result = gauss_legendre_compute(n, prec);
        save_gl_cache(n, prec, &result.0, &result.1);
        result
    }

    fn gl_cache_path(n: usize, prec: u32) -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let dir = std::path::PathBuf::from(home).join(".cache/ccm_gl");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir.join(format!("prec{}_npts{}.json", prec, n)))
    }

    fn load_gl_cache(n: usize, prec: u32) -> Option<(Vec<Float>, Vec<Float>)> {
        let path = gl_cache_path(n, prec)?;
        if !path.exists() { return None; }
        let data = std::fs::read_to_string(&path).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&data).ok()?;
        let arr = parsed.as_array()?;
        if arr.len() != 2 { return None; }
        let nodes_arr = arr[0].as_array()?;
        let weights_arr = arr[1].as_array()?;
        if nodes_arr.len() != n || weights_arr.len() != n { return None; }
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for s in nodes_arr { nodes.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?)); }
        for s in weights_arr { weights.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?)); }
        Some((nodes, weights))
    }

    fn save_gl_cache(n: usize, prec: u32, nodes: &[Float], weights: &[Float]) {
        if let Some(path) = gl_cache_path(n, prec) {
            let ns: Vec<String> = nodes.iter().map(|f| f.to_string()).collect();
            let ws: Vec<String> = weights.iter().map(|f| f.to_string()).collect();
            let json = serde_json::json!([ns, ws]);
            if let Ok(s) = serde_json::to_string(&json) { let _ = std::fs::write(path, s); }
        }
    }

    fn gauss_legendre_compute(n: usize, prec: u32) -> (Vec<Float>, Vec<Float>) {
        let pi_v = pi(prec);
        let one = fl(prec, 1.0);
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let four_n_plus_two = fl_i(prec, (4 * n + 2) as i64);
        for k in 1..=n {
            let mut phi = pi_v.clone();
            phi *= fl_i(prec, (4 * k - 1) as i64);
            phi /= &four_n_plus_two;
            let mut x = phi.cos();
            let eps_threshold = fl(prec, 2.0).pow(-((prec as i32) - 8));
            for _ in 0..50 {
                let (pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
                let mut dx = pn; dx /= &pn_prime;
                x -= &dx;
                if dx.cmp_abs(&eps_threshold).map(|o| o.is_lt()).unwrap_or(false) { break; }
            }
            let (_pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
            let one_minus_x2 = { let mut v = one.clone(); v -= x.clone().square(); v };
            let mut den = one_minus_x2; den *= &pn_prime.square();
            let mut w = fl(prec, 2.0); w /= &den;
            nodes.push(x);
            weights.push(w);
        }
        let mut combined: Vec<(Float, Float)> = nodes.into_iter().zip(weights).collect();
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let nodes: Vec<Float> = combined.iter().map(|p| p.0.clone()).collect();
        let weights: Vec<Float> = combined.iter().map(|p| p.1.clone()).collect();
        (nodes, weights)
    }

    fn legendre_p_and_deriv(n: usize, x: &Float, prec: u32) -> (Float, Float) {
        let one = fl(prec, 1.0);
        if n == 0 { return (one, fl(prec, 0.0)); }
        let mut p0 = fl(prec, 1.0);
        let mut p1 = x.clone();
        if n == 1 { return (p1, one); }
        for k in 1..n {
            let kf = k as i64;
            let mut t1 = x.clone(); t1 *= &p1; t1 *= fl_i(prec, 2 * kf + 1);
            let mut t2 = p0.clone(); t2 *= fl_i(prec, kf);
            let mut p_next = t1; p_next -= &t2; p_next /= fl_i(prec, kf + 1);
            p0 = p1; p1 = p_next;
        }
        let nf = n as i64;
        let mut numer = x.clone(); numer *= &p1; numer -= &p0; numer *= fl_i(prec, nf);
        let mut denom = x.clone().square(); denom -= 1u32;
        let mut deriv = numer; deriv /= &denom;
        (p1, deriv)
    }
}

#[cfg(feature = "hp")]
pub use hp::gauss_legendre_nodes;


#[cfg(test)]
mod tests {
    use super::*;

    /// GL-64 should integrate x² on [0, 1] exactly (polynomial degree < 2*64).
    #[test]
    fn gl64_integrates_x_squared() {
        let result = gauss_legendre_64(|x| x * x, 0.0, 1.0);
        let expected = 1.0 / 3.0;
        let rel_err = (result - expected).abs() / expected;
        assert!(rel_err < 1e-14, "GL-64 x² integral: got {}, expected {}, rel err {:.2e}", result, expected, rel_err);
    }

    /// GL-64 should integrate sin(x) on [0, π] to high accuracy.
    #[test]
    fn gl64_integrates_sin() {
        let result = gauss_legendre_64(|x| x.sin(), 0.0, std::f64::consts::PI);
        let expected = 2.0;
        let rel_err = (result - expected).abs() / expected;
        assert!(rel_err < 1e-13, "GL-64 sin integral: got {}, expected {}, rel err {:.2e}", result, expected, rel_err);
    }

    /// GL-64 should integrate exp(-x²) on [-5, 5] ≈ √π.
    #[test]
    fn gl64_integrates_gaussian() {
        let result = gauss_legendre_64(|x| (-x * x).exp(), -5.0, 5.0);
        let expected = std::f64::consts::PI.sqrt();
        let rel_err = (result - expected).abs() / expected;
        assert!(rel_err < 1e-10, "GL-64 Gaussian integral: got {}, expected {}, rel err {:.2e}", result, expected, rel_err);
    }
}
