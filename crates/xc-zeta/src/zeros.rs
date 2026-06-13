// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Reference Riemann zeta zero loaders.
//!
//! The canonical reference is a JSON array of decimal strings (one per
//! zero, at HP precision — typically 1000 digits). Consumers either
//! use the strings directly (HP path) or parse to f64 (bound checks
//! and other ε-O(1) computations).
//!
//! Path resolution: callers pass the file path explicitly so this
//! crate is independent of any specific repo layout.

use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;

/// Load the first n zero imaginary parts as decimal strings (HP-precision).
pub fn first_n_strings(path: &Path, n: usize) -> Result<Vec<String>> {
    if !path.exists() {
        return Err(anyhow!(
            "Reference zeros file {} not found",
            path.display()
        ));
    }
    let data = fs::read_to_string(path)?;
    let zeros: Vec<String> = serde_json::from_str(&data)?;
    if zeros.len() < n {
        return Err(anyhow!(
            "Need {} reference zeros; {} contains {}.",
            n, path.display(), zeros.len()
        ));
    }
    Ok(zeros[..n].to_vec())
}

/// Load the first n zero imaginary parts truncated to f64.
pub fn first_n_f64(path: &Path, n: usize) -> Result<Vec<f64>> {
    let strings = first_n_strings(path, n)?;
    let mut out = Vec::with_capacity(strings.len());
    for s in strings {
        let v: f64 = s.parse()
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
        let parsed = rug::Float::parse(&s)
            .map_err(|e| anyhow!("Failed to parse zero {:?}: {}", s, e))?;
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
            .join("..").join("..").join("target").join("test-tmp")
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

    /// `first_n_hp` loads zeros as rug::Float at high precision.
    /// Verify the first zero matches the string value to working precision.
    #[cfg(feature = "hp")]
    #[test]
    fn first_n_hp_loads_at_working_precision() {
        use rug::{Float, ops::Pow};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..").join("target").join("test-tmp")
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
        let reference = Float::with_val(prec,
            Float::parse("14.134725141734693790457251983562470270784257").unwrap());
        let mut diff = zeros[0].clone();
        diff -= &reference;
        let abs_diff = diff.abs();
        let two = Float::with_val(prec, 2);
        let tol = two.pow(-(prec as i32 - 16));
        assert!(abs_diff < tol,
            "first_n_hp should parse to working precision; abs_diff = {}",
            abs_diff);

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
