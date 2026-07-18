// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Reference Riemann zeta zero loaders.
//!
//! The canonical bundled reference is a JSON array containing the first 1,000
//! positive ordinates as 2,500-digit decimal strings. Consumers either
//! use the strings directly (HP path) or parse to f64 (bound checks
//! and other ε-O(1) computations).
//!
//! Callers may use the toolkit-owned bundled reference or explicitly load a
//! different file. The bundled bytes make verification independent of any
//! research-project repository layout.

use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;

/// Stable logical name of the toolkit-owned high-precision zero table.
pub const BUNDLED_ZETA_ZEROS_RESOURCE: &str = "xc-zeta/data/zeta_zeros_1000x2500.json";

/// Exact bytes of the first 1,000 positive ordinates at 2,500 decimal digits.
///
/// The ordinates were computed with rigorous Arb interval arithmetic. Their
/// leading 1,000 digits were independently cross-checked against Odlyzko's
/// standard tabulation; that tabulation is validation evidence, not the source
/// of these 2,500-digit values.
pub const BUNDLED_ZETA_ZEROS_JSON: &[u8] = include_bytes!("../data/zeta_zeros_1000x2500.json");

/// Loads the first `n` decimal strings from the toolkit-owned reference table.
pub fn bundled_first_n_strings(n: usize) -> Result<Vec<String>> {
    first_n_strings_from_bytes(BUNDLED_ZETA_ZEROS_JSON, n, BUNDLED_ZETA_ZEROS_RESOURCE)
}

fn first_n_strings_from_bytes(bytes: &[u8], n: usize, source: &str) -> Result<Vec<String>> {
    let zeros: Vec<String> = serde_json::from_slice(bytes)?;
    if zeros.len() < n {
        return Err(anyhow!(
            "Need {} reference zeros; {} contains {}.",
            n,
            source,
            zeros.len()
        ));
    }
    Ok(zeros[..n].to_vec())
}

/// Loads the first `n` reference-zero imaginary parts without numeric conversion.
///
/// # Mathematical semantics
/// Returns the ordered decimal records from a caller-selected reference dataset;
/// the values are comparison inputs, not discovered roots or proof evidence.
///
/// # Precision
/// Decimal text is preserved exactly. Parsing into binary64 or an HP scalar is
/// available only through separate explicit entry points.
///
/// # Failure states
/// A missing or unreadable file, invalid JSON, or a dataset shorter than `n`
/// returns an error and no partial prefix.
///
/// # Assurance and validity
/// Loading proves neither provenance nor correctness of the dataset. Callers
/// must bind and verify its digest before using it as trusted comparison data.
///
/// # Cache effects
/// Reads only the exact local path supplied by the caller and performs no cache
/// lookup, download, persistence, or publication.
///
/// # Example
/// Compiled example: `crates/xc-zeta/examples/reference_zeros.rs`.
pub fn first_n_strings(path: &Path, n: usize) -> Result<Vec<String>> {
    if !path.exists() {
        return Err(anyhow!("Reference zeros file {} not found", path.display()));
    }
    let data = fs::read(path)?;
    first_n_strings_from_bytes(&data, n, &path.display().to_string())
}

/// Load the first n zero imaginary parts truncated to f64.
pub fn first_n_f64(path: &Path, n: usize) -> Result<Vec<f64>> {
    let strings = first_n_strings(path, n)?;
    let mut out = Vec::with_capacity(strings.len());
    for s in strings {
        let v: f64 = s
            .parse()
            .map_err(|e| anyhow!("Failed to parse zero {:?}: {}", s, e))?;
        out.push(v);
    }
    Ok(out)
}

/// Load the first n zero imaginary parts as `rug::Float` at the given
/// precision (in bits).
#[cfg(feature = "hp")]
pub fn first_n_hp(path: &Path, n: usize, prec: u32) -> Result<Vec<rug::Float>> {
    let strings = first_n_strings(path, n)?;
    let mut out = Vec::with_capacity(strings.len());
    for s in strings {
        let parsed =
            rug::Float::parse(&s).map_err(|e| anyhow!("Failed to parse zero {:?}: {}", s, e))?;
        out.push(rug::Float::with_val(prec, parsed));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)] // reference zeros quoted at published precision
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a temp file with 3 zeros and verify loading works.
    #[test]
    fn load_zeros_from_file() {
        // Scratch under target/test-tmp (removed by cargo clean), not
        // the OS temp dir. Resolved from CARGO_MANIFEST_DIR so it is
        // correct regardless of the process's runtime cwd.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("xc_zeta_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_zeros.json");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"["14.134725141734693790457251983562470270784257", "21.022039638771554992628479593896902777334340", "25.010857580145688763213790992562821818659549"]"#).unwrap();

        let strings = first_n_strings(&path, 2).unwrap();
        assert_eq!(strings.len(), 2);
        assert!(strings[0].starts_with("14.134725"));

        let f64s = first_n_f64(&path, 3).unwrap();
        assert_eq!(f64s.len(), 3);
        assert!((f64s[0] - 14.134725141734694).abs() < 1e-12);
        assert!((f64s[2] - 25.010857580145689).abs() < 1e-12);

        // Requesting more than available should error.
        let err = first_n_f64(&path, 10);
        assert!(err.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bundled_table_has_expected_shape() {
        let zeros = bundled_first_n_strings(1_000).unwrap();
        assert_eq!(zeros.len(), 1_000);
        assert!(zeros.iter().all(|zero| zero.len() == 2_501));
        assert!(zeros[0].starts_with("14.134725141734693790457251983562"));
    }

    /// `first_n_hp` loads zeros as rug::Float at high precision.
    /// Verify the first zero matches the string value to working precision.
    #[cfg(feature = "hp")]
    #[test]
    fn first_n_hp_loads_at_working_precision() {
        use rug::{ops::Pow, Float};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("xc_zeta_hp_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_zeros_hp.json");
        let mut f = std::fs::File::create(&path).unwrap();
        // Write one zero with full 43-digit precision.
        writeln!(f, r#"["14.134725141734693790457251983562470270784257"]"#).unwrap();

        let prec = 256;
        let zeros = first_n_hp(&path, 1, prec).unwrap();
        assert_eq!(zeros.len(), 1);
        // Parse the reference string at the same precision and compare.
        let reference = Float::with_val(
            prec,
            Float::parse("14.134725141734693790457251983562470270784257").unwrap(),
        );
        let mut diff = zeros[0].clone();
        diff -= &reference;
        let abs_diff = diff.abs();
        let two = Float::with_val(prec, 2);
        let tol = two.pow(-(prec as i32 - 16));
        assert!(
            abs_diff < tol,
            "first_n_hp should parse to working precision; abs_diff = {}",
            abs_diff
        );

        // Requesting more than available should error.
        let err = first_n_hp(&path, 5, prec);
        assert!(err.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `first_n_strings` on a non-existent file should return an error.
    #[test]
    fn load_zeros_missing_file_returns_error() {
        let path = std::path::Path::new("/does/not/exist/zeros.json");
        assert!(first_n_strings(path, 1).is_err());
        assert!(first_n_f64(path, 1).is_err());
    }
}
