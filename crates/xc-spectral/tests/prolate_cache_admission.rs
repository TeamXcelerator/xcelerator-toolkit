#![cfg(feature = "hp")]

use rug::{float::Round, Float};
use serde_json::json;
use std::path::{Path, PathBuf};
use xc_numerics::eigen::{tridiag_eigenvalues_hp, TRIDIAG_QR_SEMANTICS};
use xc_numerics::quadrature::CacheMode;
use xc_spectral::ccm::LambdaSq;
use xc_spectral::prolate::hp::{
    compute_k_lambda, compute_k_lambda_finite_dirichlet, try_build_pw_matrix,
};

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Minimal single-entry stored ZIP.
fn write_stored_zip(path: &Path, entry: &str, data: &[u8]) {
    let crc = crc32(data);
    let name = entry.as_bytes();
    let mut out = Vec::new();
    let le16 = |v: &mut Vec<u8>, x: u16| v.extend_from_slice(&x.to_le_bytes());
    let le32 = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    le32(&mut out, 0x0403_4b50);
    le16(&mut out, 20);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 0x21);
    le32(&mut out, crc);
    le32(&mut out, data.len() as u32);
    le32(&mut out, data.len() as u32);
    le16(&mut out, name.len() as u16);
    le16(&mut out, 0);
    out.extend_from_slice(name);
    out.extend_from_slice(data);
    let cd_offset = out.len() as u32;
    le32(&mut out, 0x0201_4b50);
    le16(&mut out, 20);
    le16(&mut out, 20);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 0x21);
    le32(&mut out, crc);
    le32(&mut out, data.len() as u32);
    le32(&mut out, data.len() as u32);
    le16(&mut out, name.len() as u16);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le32(&mut out, 0);
    le32(&mut out, 0);
    out.extend_from_slice(name);
    let cd_size = out.len() as u32 - cd_offset;
    le32(&mut out, 0x0605_4b50);
    le16(&mut out, 0);
    le16(&mut out, 0);
    le16(&mut out, 1);
    le16(&mut out, 1);
    le32(&mut out, cd_size);
    le32(&mut out, cd_offset);
    le16(&mut out, 0);
    std::fs::write(path, out).unwrap();
}

/// Even Legendre block exactly as documented in PROLATE_NUMERICAL_MODEL.md
/// (diagonal j(j+1)+c^2(a_j^2+a_{j-1}^2), coupling c^2 a_j a_{j+1}), c=2*pi*lambda^2.
/// Built independently from the documented formula, not from toolkit code.
fn legendre_block(lambda: &Float, n: usize, p: u32) -> (Vec<Float>, Vec<Float>) {
    let c = Float::with_val(p, rug::float::Constant::Pi) * 2u32 * lambda * lambda;
    let c2 = Float::with_val(p, &c * &c);
    let a = |j: i64| -> Float {
        if j < 0 {
            return Float::with_val(p, 0);
        }
        Float::with_val(p, j + 1) / Float::with_val(p, (2 * j + 1) * (2 * j + 3)).sqrt()
    };
    let mut d = Vec::new();
    let mut b = Vec::new();
    for i in 0..n {
        let j = 2 * i as i64;
        let diag = Float::with_val(p, j * (j + 1))
            + Float::with_val(p, &c2 * (a(j).square() + a(j - 1).square()));
        d.push(diag);
        if i + 1 < n {
            b.push(Float::with_val(p, &c2 * (a(j) * a(j + 1))));
        }
    }
    (d, b)
}

fn spectrum_entry(model: &str, lambda: &Float, n: usize, prec: u32) -> String {
    let identity = xc_cache::ContentDigest::sha256(
        &serde_json::to_vec(&(
            model,
            TRIDIAG_QR_SEMANTICS,
            lambda.to_string(),
            lambda.prec(),
            n,
            prec,
        ))
        .unwrap(),
    );
    format!("exact_lambda_{}.json", identity.0)
}

fn write_spectrum(
    dir: &Path,
    model: &str,
    lambda: &Float,
    n: usize,
    prec: u32,
    values: &[Float],
) -> PathBuf {
    let entry = spectrum_entry(model, lambda, n, prec);
    let payload = json!({
        "schema_version": 3, "lambda": lambda.to_string(), "lambda_precision_bits": lambda.prec(),
        "grid_points": n, "precision_bits": prec,
        "eigenvalues": values.iter().map(Float::to_string).collect::<Vec<_>>(),
    });
    let path = dir.join(format!("{entry}.zip"));
    write_stored_zip(&path, &entry, &serde_json::to_vec(&payload).unwrap());
    path
}

fn perturbed(values: &[Float], index: usize, relative_exponent: i32, prec: u32) -> Vec<Float> {
    let mut out = values.to_vec();
    let mut delta = Float::with_val(prec, &out[index]);
    delta <<= relative_exponent; // multiply by 2^relative_exponent (negative)
    let (v, _) = Float::with_val_round(prec, &out[index] + &delta, Round::Nearest);
    out[index] = v;
    out
}

fn rel(a: &Float, b: &Float) -> String {
    let d = Float::with_val(a.prec().max(b.prec()), a - b).abs() / b.clone().abs();
    d.to_string_radix(10, Some(4))
}

#[test]
fn corrupted_spectra_become_recoverable_misses_and_return_fresh_pairs() {
    let root_dir = xc_core::test_support::TestDir::new("r2-repair-prolate-cache");
    let root = root_dir.to_path_buf();
    let original_cwd = std::env::current_dir().unwrap();
    // Restore the cwd before `root_dir` is removed, including on panic:
    // Windows cannot delete the current directory.
    struct RestoreCwd(std::path::PathBuf);
    impl Drop for RestoreCwd {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }
    let _restore_cwd = RestoreCwd(original_cwd.clone());
    std::env::set_current_dir(&root).unwrap();
    // The only test in this binary, so setting the process environment is safe.
    std::env::set_var("XC_CACHE_ROOT", root.join("data"));
    let dir = root.join("data").join("prolate_eigvals_cache");

    // ---------------- ordinary bounded Legendre route ----------------
    let p = 128u32;
    let work = p + 64;
    let lambda = Float::with_val(p, 2).sqrt();
    let baseline = compute_k_lambda(&lambda, 512, 4, p, CacheMode::Off).unwrap();
    let n = baseline.basis_dimension;
    println!(
        "LEG baseline n={n} eig0={} residual={}",
        baseline.eigenvalue_0.to_string_radix(10, Some(45)),
        baseline
            .relative_operator_residual
            .as_ref()
            .unwrap()
            .to_string_radix(10, Some(4))
    );
    std::fs::create_dir_all(&dir).unwrap();
    let lam_work = Float::with_val(p, &lambda);
    let (d, b) = legendre_block(&lam_work, n, work);
    let exact = tridiag_eigenvalues_hp(&d, &b, work).unwrap();
    let model = "prolate-bounded-legendre-even-spectrum-v1";
    // Relative perturbations of cached eigenvalue 0 (all inside the index-replay
    // radius 2^-(work/2) except the last):
    //   2^-(work-8): below the downstream eigenvector pairing tolerance;
    //   2^-(p-6):    inside the documented residual gate 2^-(p-12);
    //   2^-(work/2+4): inside the index radius only;
    //   2^-8:        outside the index radius.
    for (label, exponent) in [
        ("below_pairing_tolerance", -((work as i32) - 8)),
        ("inside_residual_gate", -((p as i32) - 6)),
        ("inside_index_radius_only", -((work as i32) / 2 + 4)),
        ("outside_index_radius", -8),
    ] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        let tampered = perturbed(&exact, 0, exponent, work);
        write_spectrum(&dir, model, &lambda, n, work, &tampered);
        let first = compute_k_lambda(&lambda, 512, 4, p, CacheMode::JsonZip);
        let second = compute_k_lambda(&lambda, 512, 4, p, CacheMode::JsonZip);
        match (&first, &second) {
            (Ok(got), Ok(_)) => println!(
                "LEG {label}: Ok basis_dimension {} (baseline {n}); eig0 rel diff to baseline {}; k[1] rel diff {}",
                got.basis_dimension,
                rel(&got.eigenvalue_0, &baseline.eigenvalue_0),
                rel(&got.k_values[1], &baseline.k_values[1]),
            ),
            _ => println!(
                "LEG {label}: first call {:?}; second call {:?}",
                first.as_ref().map(|r| r.basis_dimension).map_err(|e| e.to_string()),
                second.as_ref().map(|r| r.basis_dimension).map_err(|e| e.to_string()),
            ),
        }
        let got = first.unwrap();
        assert_eq!(got.basis_dimension, n, "{label}");
        assert_eq!(got.eigenvalue_0, baseline.eigenvalue_0, "{label}");
        assert_eq!(got.k_values, baseline.k_values, "{label}");
        assert_eq!(
            second.unwrap().eigenvalue_0,
            baseline.eigenvalue_0,
            "warm {label}"
        );
    }
    // The same request without the cache succeeds.
    assert!(compute_k_lambda(&lambda, 512, 4, p, CacheMode::Off).is_ok());

    // ---------------- historical finite-Dirichlet route ----------------
    let nfd = 31usize;
    let fd_base = compute_k_lambda_finite_dirichlet(&lambda, nfd, 4, p, CacheMode::Off).unwrap();
    let lam_p = Float::with_val(p, &lambda);
    let (fd_d, fd_b) = try_build_pw_matrix(&lam_p, nfd, p).unwrap();
    let fd_exact = tridiag_eigenvalues_hp(&fd_d, &fd_b, p).unwrap();
    let fd_model = "prolate-fd-working-precision-spectrum-v0.15.1-v2";
    for (label, exponent) in [
        ("fd_inside_index_radius", -((p as i32) / 2 + 4)),
        ("fd_outside_radius", -((p as i32) / 2 - 8)),
    ] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        let tampered = perturbed(&fd_exact, 0, exponent, p);
        write_spectrum(&dir, fd_model, &lambda, nfd, p, &tampered);
        let first = compute_k_lambda_finite_dirichlet(&lambda, nfd, 4, p, CacheMode::JsonZip);
        let second = compute_k_lambda_finite_dirichlet(&lambda, nfd, 4, p, CacheMode::JsonZip);
        println!(
            "FD {label}: first {:?}; second {:?}",
            first
                .as_ref()
                .map(|r| rel(&r.eigenvalue_0, &fd_base.eigenvalue_0))
                .map_err(|e| e.to_string()),
            second
                .as_ref()
                .map(|r| rel(&r.eigenvalue_0, &fd_base.eigenvalue_0))
                .map_err(|e| e.to_string()),
        );
        assert_eq!(first.unwrap().eigenvalue_0, fd_base.eigenvalue_0, "{label}");
        assert_eq!(
            second.unwrap().eigenvalue_0,
            fd_base.eigenvalue_0,
            "warm {label}"
        );
    }

    // ---------------- legacy integer-key FD cache (lambda = 2) ----------------
    let lambda_int = Float::with_val(p, 2);
    let fd_int_base =
        compute_k_lambda_finite_dirichlet(&lambda_int, nfd, 4, p, CacheMode::Off).unwrap();
    let (di, bi) = try_build_pw_matrix(&lambda_int, nfd, p).unwrap();
    let int_exact = tridiag_eigenvalues_hp(&di, &bi, p).unwrap();
    let key = LambdaSq::integer(4);
    let name = format!(
        "lambda_sq{}_ngrid{}_prec{}.json",
        key.filename_str(),
        nfd,
        p
    );
    for (label, exponent, version) in [
        ("legacy_inside_radius", -((p as i32) / 2 + 4), "0.13.0"),
        ("legacy_old_version", -((p as i32) / 2 + 4), "0.12.9"),
    ] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        let tampered = perturbed(&int_exact, 0, exponent, p);
        let payload = json!({
            "schema_version": 1, "toolkit_version": version,
            "arithmetic_semantics": "prolate-fd-working-precision-v2", "qr_arithmetic": TRIDIAG_QR_SEMANTICS,
            "lambda_sq": 4.0, "lambda_sq_mode": key.mode_str(), "lambda_sq_identity": key.filename_str(),
            "n_grid": nfd, "precision_bits": p,
            "eigenvalues": tampered.iter().map(Float::to_string).collect::<Vec<_>>(),
        });
        write_stored_zip(
            &dir.join(format!("{name}.zip")),
            &name,
            &serde_json::to_vec(&payload).unwrap(),
        );
        let got = compute_k_lambda_finite_dirichlet(&lambda_int, nfd, 4, p, CacheMode::JsonZip);
        println!(
            "FD {label}: {:?}",
            got.as_ref()
                .map(|r| rel(&r.eigenvalue_0, &fd_int_base.eigenvalue_0))
                .map_err(|e| e.to_string()),
        );
        assert_eq!(
            got.unwrap().eigenvalue_0,
            fd_int_base.eigenvalue_0,
            "{label}"
        );
    }

    // Additional cutoffs/precisions were not selected by the finding.
    for (cutoff, p) in [(3u32, 96u32), (5, 164)] {
        let lambda = Float::with_val(p, cutoff).sqrt();
        let base = compute_k_lambda_finite_dirichlet(&lambda, 19, 3, p, CacheMode::Off).unwrap();
        let (d, b) = try_build_pw_matrix(&lambda, 19, p).unwrap();
        let values = tridiag_eigenvalues_hp(&d, &b, p).unwrap();
        let wrong = perturbed(&values, 4, -((p as i32) / 2 + 7), p);
        write_spectrum(
            &dir,
            "prolate-fd-working-precision-spectrum-v0.15.1-v2",
            &lambda,
            19,
            p,
            &wrong,
        );
        let got = compute_k_lambda_finite_dirichlet(&lambda, 19, 3, p, CacheMode::JsonZip).unwrap();
        assert_eq!(got.eigenvalue_4, base.eigenvalue_4);
        assert_eq!(got.k_values, base.k_values);
    }
    std::env::set_current_dir(original_cwd).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}
