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

    /// Tolerance for structural identity checks on a loaded cache file.
    /// At precision `prec` bits, this is `2^-(prec - 8)` — 8 guard bits
    /// below working precision, which gives ~10⁻⁷⁵ at HP-256 and
    /// ~10⁻³⁰⁰ at HP-1000. Far above any plausible correct-cache rounding
    /// drift, well below any value corruption that would meaningfully
    /// affect downstream integration.
    fn cache_structural_tol(prec: u32) -> Float {
        Float::with_val(prec, 2).pow(-((prec as i32) - 8))
    }

    /// Verify the three classical structural identities of a Gauss-
    /// Legendre node/weight pair on `[-1, 1]`:
    ///
    ///   1. Σ w_i = 2 (length of [-1, 1])
    ///   2. Σ x_i · w_i = 0 (first moment of an even weight function)
    ///   3. Antisymmetry: nodes[i] + nodes[n-1-i] = 0,
    ///      weights[i] - weights[n-1-i] = 0
    ///
    /// Returns `None` if all three identities hold within
    /// `cache_structural_tol(prec)`. Returns `Some(reason)` describing
    /// the first identity that fails, with both magnitudes in the
    /// reason string for diagnostic purposes.
    ///
    /// Used by `load_gl_cache` to discard structurally-broken cache
    /// files (e.g. wrong precision, value corruption, accidental edit)
    /// before they pollute downstream HP integration.
    fn cache_structural_check(
        nodes: &[Float],
        weights: &[Float],
        prec: u32,
    ) -> Option<String> {
        let n = nodes.len();
        if weights.len() != n {
            return Some(format!(
                "weight count {} != node count {}", weights.len(), n
            ));
        }
        let tol = cache_structural_tol(prec);

        // Identity 1: Σ w_i = 2.
        let mut wsum = Float::with_val(prec, 0);
        for w in weights { wsum += w; }
        let mut wdiff = wsum.clone(); wdiff -= 2u32;
        let abs_wdiff = wdiff.abs();
        if !abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return Some(format!(
                "Σ weights deviates from 2 by {} (tol {})",
                abs_wdiff, tol
            ));
        }

        // Identity 2: Σ x_i · w_i = 0.
        let mut moment = Float::with_val(prec, 0);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            let mut t = x.clone();
            t *= w;
            moment += &t;
        }
        let abs_moment = moment.abs();
        if !abs_moment.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
            return Some(format!(
                "first moment Σ x_i w_i deviates from 0 by {} (tol {})",
                abs_moment, tol
            ));
        }

        // Identity 3: antisymmetry of nodes; mirror symmetry of weights.
        for i in 0..(n / 2) {
            let mut node_sum = nodes[i].clone();
            node_sum += &nodes[n - 1 - i];
            let abs_ns = node_sum.abs();
            if !abs_ns.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
                return Some(format!(
                    "antisymmetry: nodes[{}] + nodes[{}] = {} (tol {})",
                    i, n - 1 - i, abs_ns, tol
                ));
            }
            let mut wm = weights[i].clone();
            wm -= &weights[n - 1 - i];
            let abs_wm = wm.abs();
            if !abs_wm.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false) {
                return Some(format!(
                    "weight mirror: weights[{}] - weights[{}] = {} (tol {})",
                    i, n - 1 - i, abs_wm, tol
                ));
            }
        }

        None
    }

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

    /// Print a diagnostic warning to stderr when a cache file exists
    /// but fails to load (corrupt JSON, malformed zip, structural
    /// identity violation). Includes the file path and a short reason
    /// so reviewers can identify and remediate corrupt fixtures
    /// without silently triggering a multi-minute Newton recompute.
    fn warn_cache_skip(path: &std::path::Path, reason: &str) {
        eprintln!(
            "[gl_cache] WARNING: skipping {} ({}); recomputing",
            path.display(), reason
        );
    }

    fn load_gl_cache(n: usize, prec: u32) -> Option<(Vec<Float>, Vec<Float>)> {
        // Path 1: uncompressed JSON. Fast read.
        if let Some(path) = gl_cache_path(n, prec) {
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(data) => match parse_gl_json(&data, n, prec) {
                        Some((nodes, weights)) => {
                            // Structural identity check on the loaded
                            // values. A wrong-precision file or a value-
                            // corrupted file can pass `parse_gl_json`
                            // (which only checks shape), but it won't
                            // pass the GL structural identities.
                            if let Some(reason) =
                                cache_structural_check(&nodes, &weights, prec)
                            {
                                warn_cache_skip(&path, &reason);
                            } else {
                                return Some((nodes, weights));
                            }
                        }
                        None => {
                            warn_cache_skip(&path, "JSON shape mismatch or unparseable");
                        }
                    },
                    Err(e) => {
                        warn_cache_skip(&path, &format!("read failed: {}", e));
                    }
                }
            }
        }

        // Path 2: zip-compressed JSON. Decompress, parse, and write the
        // decompressed JSON next to the .zip so subsequent reads in the
        // same process and on the same machine hit path 1 directly.
        if let Some(zip_path) = gl_cache_zip_path(n, prec) {
            if zip_path.exists() {
                match load_from_zip(&zip_path, n, prec) {
                    Some((parsed, json_string)) => {
                        // Structural check on the zip-loaded values too.
                        if let Some(reason) =
                            cache_structural_check(&parsed.0, &parsed.1, prec)
                        {
                            warn_cache_skip(&zip_path, &reason);
                        } else {
                            // Write the decompressed copy. Best-effort: errors
                            // are non-fatal since we already have the parsed
                            // result in memory.
                            if let Some(json_path) = gl_cache_path(n, prec) {
                                let _ = std::fs::write(&json_path, &json_string);
                            }
                            return Some(parsed);
                        }
                    }
                    None => {
                        warn_cache_skip(
                            &zip_path,
                            "zip open / decompress / shape parse failed",
                        );
                    }
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
        // Always write the uncompressed .json first — it's the fast
        // next-read path. Then also write a deflated .json.zip
        // alongside for distribution (smaller commits, fewer git
        // bytes). Mirrors the tau_cache pattern. The toolkit's read
        // path already prefers .json over .json.zip on subsequent
        // reads, so the .zip is purely a distribution artifact.
        if let Some(path) = gl_cache_path(n, prec) {
            let ns: Vec<String> = nodes.iter().map(|f| f.to_string()).collect();
            let ws: Vec<String> = weights.iter().map(|f| f.to_string()).collect();
            let json = serde_json::json!([ns, ws]);
            let json_str = match serde_json::to_string(&json) {
                Ok(s) => s,
                Err(_) => return,
            };
            if std::fs::write(&path, &json_str).is_err() { return; }

            // Compress to .json.zip alongside. Failure on the zip
            // write is non-fatal (the .json above is the canonical
            // cache; the .zip is opportunistic).
            use std::io::Write;
            let entry_name = format!("prec{}_npts{}.json", prec, n);
            let zip_filename = format!("{}.zip", entry_name);
            let zip_path = match path.parent() {
                Some(p) => p.join(&zip_filename),
                None => return,
            };
            let mut buf: Vec<u8> = Vec::with_capacity(json_str.len() / 2);
            {
                let cursor = std::io::Cursor::new(&mut buf);
                let mut writer = zip::ZipWriter::new(cursor);
                let opts: zip::write::SimpleFileOptions =
                    zip::write::SimpleFileOptions::default()
                        .compression_method(zip::CompressionMethod::Deflated);
                if writer.start_file(&entry_name, opts).is_err() { return; }
                if writer.write_all(json_str.as_bytes()).is_err() { return; }
                if writer.finish().is_err() { return; }
            }
            let _ = std::fs::write(&zip_path, &buf);
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

    // ===========================================================================
    // Public cache-verification API
    // ===========================================================================

    /// Per-file outcome from `verify_gl_cache_dir`.
    #[derive(Debug, Clone)]
    pub enum CacheFileStatus {
        /// File loaded and passed all structural identity checks.
        Ok { path: std::path::PathBuf, n: usize, prec: u32 },
        /// File path was not in the expected `prec{P}_npts{N}.json[.zip]`
        /// pattern; skipped.
        Skipped { path: std::path::PathBuf, reason: String },
        /// File was in the expected pattern but failed to load (malformed
        /// JSON, malformed zip, IO error).
        LoadFailed { path: std::path::PathBuf, n: usize, prec: u32, reason: String },
        /// File loaded successfully but failed at least one of the GL
        /// structural identities (Σw=2, Σx·w=0, antisymmetry).
        StructurallyInvalid { path: std::path::PathBuf, n: usize, prec: u32, reason: String },
    }

    /// Aggregate report from `verify_gl_cache_dir`.
    #[derive(Debug, Clone)]
    pub struct CacheVerifyReport {
        /// Directory that was scanned.
        pub directory: std::path::PathBuf,
        /// One status entry per file found in `directory` (in arbitrary
        /// filesystem order). Files outside the expected naming pattern
        /// are reported as `CacheFileStatus::Skipped`.
        pub statuses: Vec<CacheFileStatus>,
    }

    impl CacheVerifyReport {
        /// Count of files that passed all checks.
        pub fn ok_count(&self) -> usize {
            self.statuses.iter().filter(|s| matches!(s, CacheFileStatus::Ok { .. })).count()
        }
        /// Count of files that failed at least one check (load or
        /// structural). Skipped files are not counted as failures.
        pub fn failure_count(&self) -> usize {
            self.statuses.iter().filter(|s| {
                matches!(s,
                    CacheFileStatus::LoadFailed { .. }
                    | CacheFileStatus::StructurallyInvalid { .. }
                )
            }).count()
        }
        /// All failure entries (load + structural), for callers that
        /// want to print only the bad files.
        pub fn failures(&self) -> impl Iterator<Item = &CacheFileStatus> {
            self.statuses.iter().filter(|s| {
                matches!(s,
                    CacheFileStatus::LoadFailed { .. }
                    | CacheFileStatus::StructurallyInvalid { .. }
                )
            })
        }
    }

    /// Parse a cache filename of the form `prec{P}_npts{N}.json` or
    /// `prec{P}_npts{N}.json.zip`, returning `(N, P)`. Returns `None`
    /// for any other filename pattern.
    fn parse_cache_filename(name: &str) -> Option<(usize, u32)> {
        let stem = name
            .strip_suffix(".json.zip")
            .or_else(|| name.strip_suffix(".json"))?;
        // stem now: "prec{P}_npts{N}"
        let after_prec = stem.strip_prefix("prec")?;
        let mut parts = after_prec.splitn(2, "_npts");
        let prec_str = parts.next()?;
        let n_str = parts.next()?;
        let prec: u32 = prec_str.parse().ok()?;
        let n: usize = n_str.parse().ok()?;
        Some((n, prec))
    }

    /// Walk the given cache directory and structurally verify every
    /// `prec{P}_npts{N}.json[.zip]` file in it. Returns a per-file
    /// status report; does not mutate any files (corrupt files are
    /// not deleted).
    ///
    /// Use this from a CLI wrapper to audit a cache directory before
    /// a long HP run, e.g.:
    ///
    /// ```ignore
    /// let report = verify_gl_cache_dir(std::path::Path::new("data/gl_cache"))?;
    /// for failure in report.failures() { eprintln!("{:?}", failure); }
    /// if report.failure_count() > 0 { std::process::exit(1); }
    /// ```
    ///
    /// The verification runs sequentially per file. At HP-3338 with
    /// thousands of cache files this can take several seconds; for
    /// production use callers may want to prune the directory first
    /// to the precisions they actually plan to use.
    pub fn verify_gl_cache_dir(
        dir: &std::path::Path,
    ) -> std::io::Result<CacheVerifyReport> {
        let mut statuses: Vec<CacheFileStatus> = Vec::new();

        if !dir.exists() {
            return Ok(CacheVerifyReport {
                directory: dir.to_path_buf(),
                statuses,
            });
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() { continue; }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };

            let (n, prec) = match parse_cache_filename(name) {
                Some(np) => np,
                None => {
                    statuses.push(CacheFileStatus::Skipped {
                        path: path.clone(),
                        reason: format!(
                            "filename '{}' not in expected prec{{P}}_npts{{N}}.json[.zip] form",
                            name
                        ),
                    });
                    continue;
                }
            };

            // Load the file.
            let parsed: Option<(Vec<Float>, Vec<Float>)> = if name.ends_with(".json.zip") {
                load_from_zip(&path, n, prec).map(|(p, _)| p)
            } else {
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|data| parse_gl_json(&data, n, prec))
            };

            let (nodes, weights) = match parsed {
                Some(p) => p,
                None => {
                    statuses.push(CacheFileStatus::LoadFailed {
                        path: path.clone(),
                        n, prec,
                        reason: "parse / decompress failed".to_string(),
                    });
                    continue;
                }
            };

            // Structural identity check.
            match cache_structural_check(&nodes, &weights, prec) {
                None => {
                    statuses.push(CacheFileStatus::Ok { path, n, prec });
                }
                Some(reason) => {
                    statuses.push(CacheFileStatus::StructurallyInvalid {
                        path, n, prec, reason
                    });
                }
            }
        }

        Ok(CacheVerifyReport {
            directory: dir.to_path_buf(),
            statuses,
        })
    }
}

#[cfg(feature = "hp")]
pub use hp::{
    gauss_legendre_nodes,
    verify_gl_cache_dir, CacheVerifyReport, CacheFileStatus,
};


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

    /// `gauss_legendre_npt_f64` at N=8 should integrate any polynomial
    /// of degree < 2N = 16 exactly. We test x^15 on [0, 1]:
    /// ∫₀¹ x¹⁵ dx = 1/16.
    #[test]
    fn gl_npt_integrates_polynomial_exactly_n8() {
        let result = gauss_legendre_npt_f64(|x| x.powi(15), 0.0, 1.0, 8);
        let expected = 1.0 / 16.0;
        let abs_err = (result - expected).abs();
        // GL with N nodes is exact (modulo f64 rounding) for polynomials
        // of degree ≤ 2N-1.
        assert!(abs_err < 1e-14,
            "GL-8 ∫x¹⁵ on [0,1]: got {}, expected {}, abs err {:.2e}",
            result, expected, abs_err);
    }

    /// `gauss_legendre_npt_f64` at N=4 should integrate ∫₀¹ x⁷ = 1/8
    /// exactly (degree 7 < 2·4 = 8).
    #[test]
    fn gl_npt_n4_polynomial_exact() {
        let result = gauss_legendre_npt_f64(|x| x.powi(7), 0.0, 1.0, 4);
        let expected = 1.0 / 8.0;
        let abs_err = (result - expected).abs();
        assert!(abs_err < 1e-14,
            "GL-4 ∫x⁷ on [0,1]: got {}, expected {}, abs err {:.2e}",
            result, expected, abs_err);
    }

    /// `gauss_legendre_npt_f64` should fail to integrate degree 2N exactly:
    /// at N=4 (max exact degree 7), degree 8 has nonzero error.
    /// We don't need the error to be a specific value — just that it's
    /// detectably nonzero (vs the polynomial-exact case above).
    #[test]
    fn gl_npt_above_exact_degree_has_error() {
        // ∫₀¹ x⁸ = 1/9
        let result = gauss_legendre_npt_f64(|x| x.powi(8), 0.0, 1.0, 4);
        let expected = 1.0 / 9.0;
        let abs_err = (result - expected).abs();
        // Error at N=4 for degree 2N=8 is O(10⁻⁵) for x⁸ on [0,1].
        // We just check it's larger than the polynomial-exact case (~1e-14).
        assert!(abs_err > 1e-7,
            "GL-4 ∫x⁸ should have appreciable error (degree exceeds 2N-1); got {:.2e}",
            abs_err);
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
            // Recover from poison: a previously-panicking test will have
            // poisoned the lock, but subsequent tests can still safely
            // acquire it (the global cwd state isn't corrupted by a test
            // panic — the prior test's CwdGuard::drop ran on unwind and
            // restored the original cwd). Without this recovery, one
            // test panic cascades into all subsequent tests panicking
            // on "cwd lock poisoned" instead of running.
            let lock = CWD_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
    /// Generate a structurally-valid synthetic GL cache JSON for use in
    /// cache-priority and zip-fallback tests, where the *content* of
    /// the cache file isn't important (we're testing the cache lookup
    /// machinery, not GL correctness). The values are not real GL
    /// nodes/weights, but they satisfy the three structural identities
    /// `cache_structural_check` enforces:
    ///
    ///   - antisymmetric nodes:  nodes[i] + nodes[n-1-i] = 0
    ///   - mirror weights:       weights[i] = weights[n-1-i]
    ///   - sum of weights = 2,   first moment = 0
    ///
    /// **Exactness matters.** The structural validator runs at HP
    /// precision and computes Σw exactly. If the f64-formatted weight
    /// strings round-trip through HP with ULP error (e.g. "4e-01" for
    /// 2/5, which is 0.4 + 2e-17 in binary), Σw is not exactly 2 and
    /// the validator (correctly) rejects the fixture. We side-step this
    /// by picking weights from `{1/4, 1/2, 3/4, 1/8, ...}` — exact
    /// powers-of-2 fractions that round-trip through binary HP exactly.
    /// For each `n`, the weight pattern is chosen so the sum is
    /// exactly 2 in binary arithmetic.
    ///
    /// `epsilon` controls the smallest absolute node value, so two
    /// synthetic fixtures with different epsilons are distinguishable
    /// (used by the cache-priority test).
    fn synthetic_valid_gl_json(n: usize, epsilon: f64) -> String {
        // Weight pattern: pick mirror-symmetric weights summing to 2
        // using only exact-binary fractions (1/2, 1/4, 3/4, 3/8, etc.).
        // For arbitrary n, use a uniform "{1/2 each pair}" structure
        // when n is even and `2/n` is exact (power of 2), and a
        // "split-the-middle" structure otherwise.
        //
        // The simplest universal pattern: weights = [w, w, ..., w_center, ..., w, w]
        // with paired w = 1/2 and w_center = 2 - n_pairs (for odd n only).
        //
        // For n even with an even count of pairs:
        //   each weight = 2/n. Exact only if n ∈ {1, 2, 4, 8, 16}.
        //
        // We instead use a deterministic recipe that's exact for any n:
        //   - For n = 1: [2.0]   (degenerate; rarely used in tests)
        //   - For n = 2: [1.0, 1.0]
        //   - For n ≥ 3: pair the first n-2 elements with weight 1/2 each
        //                (sum = (n-2)/2 from these), and split the
        //                remaining (2 - (n-2)/2) = (6 - n)/2 = ?
        //
        // Cleaner recipe: assign weight 0.5 to indices {0, n-1, 1, n-2}
        // (the outer two pairs) and the remainder uniformly to the rest.
        // But "the rest" weight = (2 - 4*0.5) / (n-4) = 0/(n-4) = 0 for
        // n > 4 — which makes interior weights zero (technically
        // structurally valid: zero is symmetric and contributes nothing).
        //
        // Use this: outer two pairs each weighted 0.5 (mirror); all
        // interior weights = 0. Σw = 4 × 0.5 = 2 exactly. Σx·w on the
        // antisymmetric outer four cancels exactly. The interior zero
        // weights contribute 0 to all moments. Structurally valid for
        // any n ≥ 4.
        //
        // For n < 4: special-case. n = 2: both 1.0; n = 3: outer pair
        // 0.5 each, center 1.0.
        let weights_exact: Vec<&'static str> = match n {
            0 => Vec::new(),
            1 => vec!["2"],
            2 => vec!["1", "1"],
            3 => vec!["5e-1", "1", "5e-1"],
            _ => {
                // n ≥ 4: outer two pairs at 0.5, interior zero.
                let mut w = vec!["0"; n];
                w[0] = "5e-1";
                w[n - 1] = "5e-1";
                w[1] = "5e-1";
                w[n - 2] = "5e-1";
                w
            }
        };

        // Nodes: antisymmetric. We use simple f64 arithmetic to build
        // them, then format the strings; mirror-symmetry of the
        // pre/post-image ensures that even when individual values have
        // ULP error in their f64 representation, the antisymmetric sum
        // (nodes[i] + nodes[n-1-i]) of two strings parsed independently
        // at HP gives exactly 0 (since one parsed value is the negation
        // of the other, character-for-character).
        let nodes: Vec<String> = (0..n).map(|i| {
            if 2 * i + 1 < n {
                // Lower half.
                let v = epsilon + 0.1 * (i as f64);
                format!("-{:.20e}", v)
            } else if 2 * i + 1 == n {
                // Center (only for odd n).
                "0".to_string()
            } else {
                // Upper half: mirror of lower half.
                let j = n - 1 - i;
                let v = epsilon + 0.1 * (j as f64);
                format!("{:.20e}", v)
            }
        }).collect();

        // Weights are already exact decimal strings.
        let weights: Vec<String> = weights_exact.iter().map(|s| s.to_string()).collect();

        serde_json::json!([nodes, weights]).to_string()
    }

    /// Old fake-data generator. Retained for tests that specifically
    /// want a *structurally-invalid* payload to exercise the rejection
    /// path; the cache-priority and zip-fallback tests no longer use
    /// it because the structural validator now rejects it.
    #[allow(dead_code)]
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
    ///
    /// Both fixtures are structurally-valid (so neither is rejected by
    /// the cache structural validator); they differ only in their
    /// "epsilon" pin so we can tell which one was loaded.
    #[test]
    fn cache_prefers_uncompressed_json_over_zip() {
        let temp = fresh_temp_dir("prefers_json");
        let _guard = CwdGuard::enter(&temp);

        let n = 4;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // .json fixture: epsilon = 0.01 (so smallest |node| ≈ 0.01).
        let json_payload = synthetic_valid_gl_json(n, 0.01);
        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        std::fs::write(&json_path, &json_payload).unwrap();

        // .zip fixture: epsilon = 0.5 (so smallest |node| ≈ 0.5).
        // The .zip should be ignored because the .json wins.
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        let zip_file = std::fs::File::create(&zip_path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(zip_file);
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip_writer
            .start_file(format!("prec{}_npts{}.json", prec, n), opts)
            .unwrap();
        let zip_payload = synthetic_valid_gl_json(n, 0.5);
        zip_writer.write_all(zip_payload.as_bytes()).unwrap();
        zip_writer.finish().unwrap();

        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // The smallest |node| should match the .json file's epsilon
        // (~0.01), not the .zip's (~0.5).
        let mut smallest_abs = f64::INFINITY;
        for x in &nodes {
            let a = x.clone().abs().to_f64();
            if a < smallest_abs { smallest_abs = a; }
        }
        assert!(smallest_abs < 0.05,
            "expected smallest |node| ~0.01 from .json fixture, got {}",
            smallest_abs);
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

        // Structurally-valid .json.zip with a known payload; no
        // uncompressed .json yet.
        let payload = synthetic_valid_gl_json(n, 0.05);
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

    /// v0.7.0 contract: fresh compute writes both `.json` AND
    /// `.json.zip` alongside. The `.json` is the canonical fast-read
    /// path; the `.zip` is a distribution artifact for paper repos
    /// to commit.
    #[test]
    fn cache_fresh_compute_writes_json_and_zip() {
        let temp = fresh_temp_dir("compute_writes_both");
        let _guard = CwdGuard::enter(&temp);

        let n = 8;
        let prec: u32 = 128;

        let json_path = temp.join("data").join("gl_cache")
            .join(format!("prec{}_npts{}.json", prec, n));
        let zip_path = temp.join("data").join("gl_cache")
            .join(format!("prec{}_npts{}.json.zip", prec, n));
        assert!(!json_path.exists(), ".json should not exist before compute");
        assert!(!zip_path.exists(), ".json.zip should not exist before compute");

        // Trigger fresh compute.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // Both files must exist after compute.
        assert!(json_path.exists(),
            "fresh compute should write {} (canonical fast-read path)",
            json_path.display());
        assert!(zip_path.exists(),
            "fresh compute should write {} (distribution artifact)",
            zip_path.display());

        // Sanity: the .zip must be smaller than the .json (compression
        // ratio > 1) and must round-trip back to the same data when
        // loaded via the existing zip-read path.
        let json_size = std::fs::metadata(&json_path).unwrap().len();
        let zip_size = std::fs::metadata(&zip_path).unwrap().len();
        assert!(zip_size < json_size,
            ".json.zip ({} bytes) should be smaller than .json ({} bytes)",
            zip_size, json_size);

        // Round-trip via the zip path: delete the .json so the loader
        // falls through to the .zip, then verify nodes/weights match.
        std::fs::remove_file(&json_path).unwrap();
        let (nodes2, weights2) = hp::gauss_legendre_nodes(n, prec);
        assert_eq!(nodes.len(), nodes2.len());
        for (a, b) in nodes.iter().zip(nodes2.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "node round-trip via zip");
        }
        for (a, b) in weights.iter().zip(weights2.iter()) {
            assert_eq!(a.to_string(), b.to_string(), "weight round-trip via zip");
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

    /// HP GL nodes/weights satisfy classical structural identities:
    ///
    /// 1. **Symmetry**: nodes[i] = -nodes[n-1-i], weights[i] = weights[n-1-i]
    ///    on the symmetric interval [-1, 1].
    /// 2. **Sum of weights**: Σ w_i = 2 (= length of [-1, 1]).
    /// 3. **First moment**: Σ x_i · w_i = 0 (by symmetry of the
    ///    weight function on [-1, 1]).
    ///
    /// All three should hold to working precision regardless of `n`.
    #[test]
    fn hp_gl_nodes_satisfy_symmetry_and_moments() {
        let temp = fresh_temp_dir("symmetry_moments");
        let _guard = CwdGuard::enter(&temp);

        let n: usize = 12;
        let prec: u32 = 256;
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // 1. Symmetry: nodes[i] + nodes[n-1-i] = 0.
        let tol = Float::with_val(prec, rug::Float::parse("1e-50").unwrap());
        for i in 0..n / 2 {
            let mut sum = nodes[i].clone();
            sum += &nodes[n - 1 - i];
            let abs_sum = sum.abs();
            assert!(abs_sum.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
                "node symmetry: nodes[{}] + nodes[{}] should be 0; got {}",
                i, n - 1 - i, abs_sum);
            // Weights mirror too.
            let mut wdiff = weights[i].clone();
            wdiff -= &weights[n - 1 - i];
            let abs_wdiff = wdiff.abs();
            assert!(abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
                "weight symmetry: weights[{}] - weights[{}] should be 0; got {}",
                i, n - 1 - i, abs_wdiff);
        }

        // 2. Sum of weights = 2.
        let mut wsum = Float::with_val(prec, 0);
        for w in &weights { wsum += w; }
        let mut wdiff = wsum.clone();
        wdiff -= 2u32;
        let abs_wdiff = wdiff.abs();
        assert!(abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "Σ weights should be 2; got {} (diff {})", wsum, abs_wdiff);

        // 3. First moment: Σ x_i · w_i = 0.
        let mut moment = Float::with_val(prec, 0);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            let mut t = x.clone();
            t *= w;
            moment += &t;
        }
        let abs_moment = moment.abs();
        assert!(abs_moment.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "Σ x_i w_i should be 0; got {}", abs_moment);
    }

    /// A structurally-invalid `.json` (shape OK, but values violate
    /// Σ w_i = 2 and antisymmetry) must be discarded by the
    /// structural validator, falling through to fresh compute.
    /// The bad file is preserved on disk (not deleted).
    #[test]
    fn cache_discards_structurally_invalid_json_and_recomputes() {
        let temp = fresh_temp_dir("structurally_invalid");
        let _guard = CwdGuard::enter(&temp);

        let n = 6;
        let prec: u32 = 128;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // The legacy fake_gl_json produces shape-valid but
        // structurally-invalid data (nodes = [0.0000000000, ...],
        // weights = [0.0000000006, ...]; sum of weights ≠ 2,
        // nodes not antisymmetric).
        let bad_payload = fake_gl_json(n);
        let json_path = cache_dir.join(format!("prec{}_npts{}.json", prec, n));
        std::fs::write(&json_path, &bad_payload).unwrap();

        // Loading should silently fall through to fresh compute. The
        // returned values must be REAL GL nodes/weights (which pass the
        // structural check), not the bad payload's values.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // Sanity: real GL nodes are antisymmetric on [-1, 1] and
        // weights sum to 2. The bad payload had neither property.
        let mut wsum = Float::with_val(prec, 0);
        for w in &weights { wsum += w; }
        let mut wdiff = wsum.clone(); wdiff -= 2u32;
        let abs_wdiff = wdiff.abs();
        let tol = Float::with_val(prec, rug::Float::parse("1e-30").unwrap());
        assert!(abs_wdiff.cmp_abs(&tol).map(|o| o.is_lt()).unwrap_or(false),
            "fresh compute should produce Σw=2; got Σw={}", wsum);

        // The bad file is preserved (not deleted).
        assert!(json_path.exists(),
            "structurally-invalid file should be preserved on disk for the user to inspect");

        // After load, the toolkit also wrote the freshly-computed
        // result over the bad file at the canonical .json path
        // (because save_gl_cache always writes after fresh compute).
        // Reading the .json now should give the *real* GL data.
        let now_on_disk = std::fs::read_to_string(&json_path).unwrap();
        assert_ne!(now_on_disk, bad_payload,
            "fresh compute should have overwritten the bad payload with valid GL data");
    }

    /// A truncated/corrupt `.json.zip` must be detected and skipped
    /// without panic, with the cache falling through to fresh compute.
    #[test]
    fn cache_handles_corrupt_zip_gracefully() {
        let temp = fresh_temp_dir("corrupt_zip");
        let _guard = CwdGuard::enter(&temp);

        let n = 4;
        let prec: u32 = 64;
        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // Write a "zip" that's actually random garbage. zip::ZipArchive::new
        // should reject it; our loader handles that and falls through.
        let zip_path = cache_dir.join(format!("prec{}_npts{}.json.zip", prec, n));
        std::fs::write(&zip_path, b"not a zip file at all -- random bytes").unwrap();

        // Should not panic. Should fall through to fresh compute.
        let (nodes, weights) = hp::gauss_legendre_nodes(n, prec);
        assert_well_formed(n, prec, &nodes, &weights);

        // The corrupt file is preserved on disk.
        assert!(zip_path.exists(),
            "corrupt zip should be preserved on disk for the user to inspect");
    }

    /// `verify_gl_cache_dir` reports OK for valid cache files and
    /// `StructurallyInvalid` for corrupt ones, without modifying any files.
    #[test]
    fn verify_gl_cache_dir_reports_per_file_status() {
        use hp::CacheFileStatus;

        let temp = fresh_temp_dir("verify_dir");
        let _guard = CwdGuard::enter(&temp);

        let cache_dir = temp.join("data").join("gl_cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // 1. Valid file (synthetic-but-structurally-valid).
        let valid_path = cache_dir.join("prec64_npts4.json");
        std::fs::write(&valid_path, synthetic_valid_gl_json(4, 0.1)).unwrap();

        // 2. Structurally-invalid file (legacy fake_gl_json shape-valid,
        //    structurally bogus).
        let bad_path = cache_dir.join("prec64_npts5.json");
        std::fs::write(&bad_path, fake_gl_json(5)).unwrap();

        // 3. Unrecognized filename — should be reported as Skipped.
        let skipped_path = cache_dir.join("not_a_cache_file.txt");
        std::fs::write(&skipped_path, "irrelevant").unwrap();

        // 4. File matching the pattern but malformed JSON.
        let malformed_path = cache_dir.join("prec64_npts3.json");
        std::fs::write(&malformed_path, "{").unwrap();

        let report = hp::verify_gl_cache_dir(&cache_dir).unwrap();
        assert_eq!(report.directory, cache_dir);
        // 4 entries; one OK, one StructurallyInvalid, one Skipped, one LoadFailed.
        assert_eq!(report.statuses.len(), 4,
            "expected 4 statuses (one per file); got {}", report.statuses.len());

        let mut saw_ok = false;
        let mut saw_invalid = false;
        let mut saw_skipped = false;
        let mut saw_loadfail = false;
        for s in &report.statuses {
            match s {
                CacheFileStatus::Ok { path, n, prec } => {
                    assert_eq!(path, &valid_path);
                    assert_eq!(*n, 4);
                    assert_eq!(*prec, 64);
                    saw_ok = true;
                }
                CacheFileStatus::StructurallyInvalid { path, n, prec, .. } => {
                    assert_eq!(path, &bad_path);
                    assert_eq!(*n, 5);
                    assert_eq!(*prec, 64);
                    saw_invalid = true;
                }
                CacheFileStatus::Skipped { path, .. } => {
                    assert_eq!(path, &skipped_path);
                    saw_skipped = true;
                }
                CacheFileStatus::LoadFailed { path, n, prec, .. } => {
                    assert_eq!(path, &malformed_path);
                    assert_eq!(*n, 3);
                    assert_eq!(*prec, 64);
                    saw_loadfail = true;
                }
            }
        }
        assert!(saw_ok, "missing Ok status");
        assert!(saw_invalid, "missing StructurallyInvalid status");
        assert!(saw_skipped, "missing Skipped status");
        assert!(saw_loadfail, "missing LoadFailed status");

        assert_eq!(report.ok_count(), 1);
        assert_eq!(report.failure_count(), 2,
            "LoadFailed + StructurallyInvalid both count as failures; expected 2");

        // Files preserved (verify_gl_cache_dir is read-only).
        assert!(valid_path.exists());
        assert!(bad_path.exists());
        assert!(skipped_path.exists());
        assert!(malformed_path.exists());
    }

    /// `verify_gl_cache_dir` on a non-existent directory returns an
    /// empty report, not an error.
    #[test]
    fn verify_gl_cache_dir_handles_missing_directory() {
        let temp = fresh_temp_dir("verify_missing");
        let _guard = CwdGuard::enter(&temp);

        let nonexistent = temp.join("does_not_exist");
        let report = hp::verify_gl_cache_dir(&nonexistent).unwrap();
        assert_eq!(report.statuses.len(), 0);
        assert_eq!(report.ok_count(), 0);
        assert_eq!(report.failure_count(), 0);
    }
}
