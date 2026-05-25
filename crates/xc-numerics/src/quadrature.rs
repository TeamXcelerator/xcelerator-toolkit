// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Gauss-Legendre quadrature at f64 and high precision.
//!
//! The HP nodes/weights are computed via Newton iteration on Legendre
//! polynomials and cached to disk under `<cwd>/data/gl_cache/` so they're
//! reused across runs at the same `(n_pts, precision_bits)`.
//!
//! Cache lookup order (HP path):
//! 1. `<cwd>/data/gl_cache/prec{prec}_npts{n}.json` (uncompressed)
//! 2. `<cwd>/data/gl_cache/prec{prec}_npts{n}.json.zip` (zip archive
//!    containing one entry of the same name without `.zip`).
//!    Auto-decompressed on first read; the result is also written
//!    out as the uncompressed `.json` so future reads in the same
//!    process and on the same machine hit path (1) directly.
//! 3. Compute fresh via Newton iteration; cache result to (1).
//!
//! New computes always write to the uncompressed `.json` form. Zip
//! files are read-only from the toolkit's perspective; they're
//! produced offline (e.g. by checking compressed cache fixtures into
//! the paper repository to avoid cold-start cost on fresh machines).

/// 64-point Gauss-Legendre quadrature on [a, b] at f64 precision.
/// For a configurable number of points, use `gauss_legendre_npt_f64`.
pub fn gauss_legendre_64pt_f64<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    gauss_legendre_npt_f64(f, a, b, 64)
}

/// N-point Gauss-Legendre quadrature on [a, b] at f64 precision.
pub fn gauss_legendre_npt_f64<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = gl_nodes_weights_f64(n);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0_f64;
    for i in 0..n {
        let x = mid + half * nodes[i];
        sum += weights[i] * f(x);
    }
    sum * half
}

fn gl_nodes_weights_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
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

    #[inline] fn fl_i(prec: u32, v: i64) -> Float { Float::with_val(prec, v) }
    #[inline] fn pi(prec: u32) -> Float { Float::with_val(prec, rug::float::Constant::Pi) }

    /// Compute (or load from disk cache) Gauss-Legendre nodes and weights
    /// at the given precision in bits, for `n` points on `[-1, 1]`.
    ///
    /// See module docs for the cache lookup order.
    pub fn gauss_legendre_nodes(n: usize, prec: u32) -> (Vec<Float>, Vec<Float>) {
        if let Some(cached) = load_gl_cache(n, prec) { return cached; }
        let result = gauss_legendre_compute(n, prec);
        save_gl_cache(n, prec, &result.0, &result.1);
        result
    }

    /// Cache directory: `<cwd>/data/gl_cache`. Created on demand so
    /// fresh checkouts work without manual setup.
    fn gl_cache_dir() -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let dir = cwd.join("data").join("gl_cache");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    /// Path to the uncompressed cache file for `(n, prec)`.
    fn gl_cache_path(n: usize, prec: u32) -> Option<std::path::PathBuf> {
        gl_cache_dir().map(|d| d.join(format!("prec{}_npts{}.json", prec, n)))
    }

    /// Path to the zip-compressed cache file for `(n, prec)`. The zip
    /// archive is expected to contain a single entry whose name is the
    /// uncompressed JSON filename (no `.zip` suffix).
    fn gl_cache_zip_path(n: usize, prec: u32) -> Option<std::path::PathBuf> {
        gl_cache_dir().map(|d| d.join(format!("prec{}_npts{}.json.zip", prec, n)))
    }

    /// Parse a 2-element JSON array of decimal strings into HP node and
    /// weight vectors. Returns `None` on any structural mismatch
    /// (precision tag missing, length mismatch, malformed numbers).
    fn parse_gl_json(data: &str, n: usize, prec: u32) -> Option<(Vec<Float>, Vec<Float>)> {
        let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
        let arr = parsed.as_array()?;
        if arr.len() != 2 { return None; }
        let nodes_arr = arr[0].as_array()?;
        let weights_arr = arr[1].as_array()?;
        if nodes_arr.len() != n || weights_arr.len() != n { return None; }
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for s in nodes_arr {
            nodes.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        for s in weights_arr {
            weights.push(Float::with_val(prec, Float::parse(s.as_str()?).ok()?));
        }
        Some((nodes, weights))
    }

    fn load_gl_cache(n: usize, prec: u32) -> Option<(Vec<Float>, Vec<Float>)> {
        // Path 1: uncompressed JSON. Fast read.
        if let Some(path) = gl_cache_path(n, prec) {
            if path.exists() {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Some(parsed) = parse_gl_json(&data, n, prec) {
                        return Some(parsed);
                    }
                }
            }
        }

        // Path 2: zip-compressed JSON. Decompress, parse, and write the
        // decompressed JSON next to the .zip so subsequent reads in the
        // same process and on the same machine hit path 1 directly.
        if let Some(zip_path) = gl_cache_zip_path(n, prec) {
            if zip_path.exists() {
                if let Some((parsed, json_string)) = load_from_zip(&zip_path, n, prec) {
                    // Write the decompressed copy. Best-effort: errors are
                    // logged but don't block returning the parsed result,
                    // since we already have it in memory.
                    if let Some(json_path) = gl_cache_path(n, prec) {
                        let _ = std::fs::write(&json_path, &json_string);
                    }
                    return Some(parsed);
                }
            }
        }

        None
    }

    /// Read a zip cache file. Expects the archive to contain exactly one
    /// entry whose name matches the uncompressed JSON filename
    /// (`prec{prec}_npts{n}.json`).
    ///
    /// Returns the parsed `(nodes, weights)` plus the raw JSON string,
    /// so the caller can write the decompressed copy to disk without
    /// re-serializing.
    fn load_from_zip(
        zip_path: &std::path::Path,
        n: usize,
        prec: u32,
    ) -> Option<((Vec<Float>, Vec<Float>), String)> {
        use std::io::Read;
        let file = std::fs::File::open(zip_path).ok()?;
        let mut archive = zip::ZipArchive::new(file).ok()?;
        let entry_name = format!("prec{}_npts{}.json", prec, n);
        let mut entry = archive.by_name(&entry_name).ok()?;
        let mut data = String::new();
        entry.read_to_string(&mut data).ok()?;
        let parsed = parse_gl_json(&data, n, prec)?;
        Some((parsed, data))
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
        let one = Float::with_val(prec, 1);
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let four_n_plus_two = fl_i(prec, (4 * n + 2) as i64);
        for k in 1..=n {
            let mut phi = pi_v.clone();
            phi *= fl_i(prec, (4 * k - 1) as i64);
            phi /= &four_n_plus_two;
            let mut x = phi.cos();
            let eps_threshold = Float::with_val(prec, 2).pow(-((prec as i32) - 8));
            for _ in 0..50 {
                let (pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
                let mut dx = pn; dx /= &pn_prime;
                x -= &dx;
                if dx.cmp_abs(&eps_threshold).map(|o| o.is_lt()).unwrap_or(false) { break; }
            }
            let (_pn, pn_prime) = legendre_p_and_deriv(n, &x, prec);
            let one_minus_x2 = { let mut v = one.clone(); v -= x.clone().square(); v };
            let mut den = one_minus_x2; den *= &pn_prime.square();
            let mut w = Float::with_val(prec, 2); w /= &den;
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
        let one = Float::with_val(prec, 1);
        if n == 0 { return (one, Float::with_val(prec, 0)); }
        let mut p0 = Float::with_val(prec, 1);
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
        let result = gauss_legendre_64pt_f64(|x| x * x, 0.0, 1.0);
        let expected = 1.0 / 3.0;
        let rel_err = (result - expected).abs() / expected;
        assert!(rel_err < 1e-14, "GL-64 x² integral: got {}, expected {}, rel err {:.2e}", result, expected, rel_err);
    }

    /// GL-64 should integrate sin(x) on [0, π] to high accuracy.
    #[test]
    fn gl64_integrates_sin() {
        let result = gauss_legendre_64pt_f64(|x| x.sin(), 0.0, std::f64::consts::PI);
        let expected = 2.0;
        let rel_err = (result - expected).abs() / expected;
        assert!(rel_err < 1e-13, "GL-64 sin integral: got {}, expected {}, rel err {:.2e}", result, expected, rel_err);
    }

    /// GL-64 should integrate exp(-x²) on [-5, 5] ≈ √π.
    #[test]
    fn gl64_integrates_gaussian() {
        let result = gauss_legendre_64pt_f64(|x| (-x * x).exp(), -5.0, 5.0);
        let expected = std::f64::consts::PI.sqrt();
        let rel_err = (result - expected).abs() / expected;
        assert!(rel_err < 1e-10, "GL-64 Gaussian integral: got {}, expected {}, rel err {:.2e}", result, expected, rel_err);
    }
}

#[cfg(all(test, feature = "hp"))]
mod hp_cache_tests {
    //! Tests for the HP GL cache lookup logic introduced in v0.4.1.
    //!
    //! These tests exercise the cwd-relative cache directory and the
    //! `.json` / `.json.zip` lookup priority. To avoid polluting the
    //! caller's `data/gl_cache/` directory, each test runs in an
    //! isolated temp directory via `std::env::set_current_dir`.
    //!
    //! Because `set_current_dir` mutates global process state, these
    //! tests must run sequentially. We use a single mutex to serialize
    //! them and restore the original cwd on drop.

    use super::*;
    use rug::Float;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Serialize all cwd-mutating tests in this module. Cargo runs
    /// tests in parallel by default; cwd is per-process (not
    /// per-thread), so two cache tests racing would corrupt each
    /// other. The mutex enforces sequential access.
    static CWD_LOCK: Mutex<()> = Mutex::new(());

    /// Guard that restores the original cwd when dropped, so a panic
    /// inside a test doesn't leave the test runner in a temp dir
    /// (which would break subsequent unrelated tests).
    struct CwdGuard {
        original: PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    impl CwdGuard {
        fn enter(temp: &std::path::Path) -> Self {
            let lock = CWD_LOCK.lock().expect("cwd lock poisoned");
            let original = std::env::current_dir().expect("no cwd");
            std::env::set_current_dir(temp).expect("set_current_dir to temp");
            CwdGuard { original, _lock: lock }
        }
    }
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    /// Make a fresh, unique temp directory under the OS temp.
    fn fresh_temp_dir(tag: &str) -> std::path::PathBuf {
        // Use a tag plus a process+nanosecond suffix to avoid clashes
        // when tests are re-run rapidly.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let dir = std::env::temp_dir()
            .join(format!("xc_numerics_gl_cache_test_{}_{}_{}", tag, pid, nanos));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Write a 2-element JSON array of decimal-string node/weight
    /// values for the given (n, prec). The values are deterministic
    /// fakes — not actual GL nodes, but well-formed enough to round
    /// trip through `parse_gl_json`.
    fn fake_gl_json(n: usize) -> String {
        let nodes: Vec<String> = (0..n).map(|i| format!("0.{:010}", i)).collect();
        let weights: Vec<String> = (0..n).map(|i| format!("0.{:010}", n + i)).collect();
        serde_json::json!([nodes, weights]).to_string()
    }

    /// Round-trip helper: compute or load nodes/weights, then verify
    /// that we got back exactly `n` of each at the requested
    /// precision, with no NaN.
    fn assert_well_formed(n: usize, prec: u32, nodes: &[Float], weights: &[Float]) {
        assert_eq!(nodes.len(), n, "node count");
        assert_eq!(weights.len(), n, "weight count");
        for (i, x) in nodes.iter().enumerate() {
            assert_eq!(x.prec(), prec, "node {} precision", i);
            assert!(!x.is_nan(), "node {} is NaN", i);
        }
        for (i, w) in weights.iter().enumerate() {
            assert_eq!(w.prec(), prec, "weight {} precision", i);
            assert!(!w.is_nan(), "weight {} is NaN", i);
        }
    }

    /// Lookup priority test: when both `.json` and `.json.zip` exist
    /// for the same `(n, prec)`, the toolkit must prefer the
    /// uncompressed `.json` (the fast path).
    #[test]
    fn cache_prefers_uncompressed_json_over_zip() {
        let temp = fresh_temp_dir("prefers_json");
        let _guard = CwdGuard::enter(&temp);

        let n = 4;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Fake .json with one set of values.
        let json_payload = fake_gl_json(n);
        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        std::fs::write(&json_path, &json_payload).unwrap();

        // Fake .json.zip with DIFFERENT values (to detect if zip path
        // is taken). The .zip should be ignored because the .json wins.
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        let zip_file = std::fs::File::create(&zip_path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(zip_file);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip_writer
            .start_file(format!("prec{}_npts{}.json", prec, n), opts)
            .unwrap();
        let other_payload = serde_json::json!([
            vec!["9.99"; n], vec!["8.88"; n]
        ]).to_string();
        zip_writer.write_all(other_payload.as_bytes()).unwrap();
        zip_writer.finish().unwrap();

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // The first node should match the .json file's value
        // (`0.0000000000`), not the .zip's (`9.99`).
        let first_node = nodes[0].to_f64();
        assert!(first_node.abs() < 1e-9,
            "expected uncompressed-json value (~0), got {}", first_node);
    }

    /// Zip fallback test: when only `.json.zip` exists, the toolkit
    /// must read it, decompress, and also write the decompressed
    /// `.json` next to it for future reads.
    #[test]
    fn cache_reads_zip_and_writes_decompressed_json() {
        let temp = fresh_temp_dir("zip_fallback");
        let _guard = CwdGuard::enter(&temp);

        let n = 5;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Fake .json.zip with a known payload; no uncompressed .json.
        let payload = fake_gl_json(n);
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        let zip_file = std::fs::File::create(&zip_path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(zip_file);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip_writer
            .start_file(format!("prec{}_npts{}.json", prec, n), opts)
            .unwrap();
        zip_writer.write_all(payload.as_bytes()).unwrap();
        zip_writer.finish().unwrap();

        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        assert!(!json_path.exists(), "uncompressed .json should not exist before read");

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // After read, the .json must exist (decompressed copy written
        // for future fast reads).
        assert!(json_path.exists(),
            "uncompressed .json should be written next to .zip after first read");
        // And its contents should be the original payload.
        let written = std::fs::read_to_string(&json_path).unwrap();
        assert_eq!(written, payload,
            "decompressed copy must match original zipped payload");
    }

    /// Compute-and-cache test: when neither `.json` nor `.json.zip`
    /// exists, the toolkit must compute fresh and write the result
    /// to the uncompressed `.json` path.
    #[test]
    fn cache_computes_fresh_and_writes_json() {
        let temp = fresh_temp_dir("compute_fresh");
        let _guard = CwdGuard::enter(&temp);

        let n = 8;
        let prec: u32 = 128;
        // Note: we deliberately do NOT create data/gl_cache/ ahead of
        // time — the toolkit must create the directory on demand.

        let json_path = temp.join("data").join("gl_cache")
            .join(format!("prec{}_npts{}.json", prec, n));
        assert!(!json_path.exists(), "cache file should not exist before compute");

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        assert!(json_path.exists(),
            "fresh compute should write to <cwd>/data/gl_cache/...");

        // Sanity: cached value should round-trip. Re-reading should
        // not recompute (fast path), and we should get back the same
        // nodes/weights bit-for-bit.
        let (nodes2, weights2) = hp::gauss_legendre_nodes(n, prec);
        assert_eq!(nodes.len(), nodes2.len());
        for (a, b) in nodes.iter().zip(nodes2.iter()) {
            // Compare via string form to avoid HP equality subtleties:
            // re-parsed Float should have the same string representation.
            assert_eq!(a.to_string(), b.to_string(), "node round-trip");
        }
        for (a, b) in weights.iter().zip(weights2.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "weight round-trip");
        }
    }

    /// Sanity check that the legitimate computed GL nodes integrate a
    /// known polynomial correctly. This is a sanity wrapper around
    /// `gauss_legendre_compute` (no cache involved if the test runs
    /// in a fresh temp cwd).
    #[test]
    fn fresh_compute_integrates_x_squared() {
        let temp = fresh_temp_dir("integrate_x2");
        let _guard = CwdGuard::enter(&temp);

        // Use small n, modest precision: enough to validate, fast to run.
        let n = 16;
        let prec: u32 = 128;
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // ∫_{-1}^{1} x² dx = 2/3.
        let mut sum = Float::with_val(prec, 0);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            let mut term = x.clone();
            term.square_mut();
            term *= w;
            sum += &term;
        }
        let two_thirds = {
            let mut v = Float::with_val(prec, 2);
            v /= 3u32;
            v
        };
        let mut diff = sum.clone();
        diff -= &two_thirds;
        let abs_err = diff.abs();
        // GL-16 nails any polynomial of degree ≤ 31 to working precision.
        let tol = Float::with_val(prec, rug::Float::parse("1e-30").unwrap());
        assert!(abs_err.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "GL-{} integral of x² should match 2/3 to working precision; abs err = {}",
            n, abs_err);
    }
}
