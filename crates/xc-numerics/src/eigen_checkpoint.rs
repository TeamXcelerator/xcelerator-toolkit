// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Private execution checkpoints, never published as mathematical evidence.
//! Only completed QR sweeps are saved. The exact binary executable, native
//! versions, exponent range, inputs, precision and sweep limit bind every file.
use super::TridiagQrOptions;
use anyhow::{ensure, Context, Result};
use rug::{Assign, Float};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{Duration, Instant},
};

const MAGIC: &[u8; 8] = b"XCQR0001";
static SERIAL: AtomicU64 = AtomicU64::new(0);
thread_local! { static ALLOWED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) }; }

pub(super) fn without<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ALLOWED.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(ALLOWED.with(|v| v.replace(false)));
    f()
}

pub(super) struct Checkpoint {
    path: PathBuf,
    identity: [u8; 32],
    maximum_bytes: u64,
    interval: Duration,
    saved: Instant,
    _scratch: Option<xc_core::MemoryReservation>,
}

fn hash_executable() -> Result<[u8; 32]> {
    static DIGEST: OnceLock<std::result::Result<[u8; 32], String>> = OnceLock::new();
    DIGEST
        .get_or_init(|| {
            (|| -> Result<_> {
                let mut file = File::open(std::env::current_exe()?)?;
                let mut hash = Sha256::new();
                let mut buffer = [0; 65536];
                loop {
                    let n = file.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    hash.update(&buffer[..n]);
                }
                Ok(hash.finalize().into())
            })()
            .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn native_versions() -> Result<Vec<u8>> {
    // These are the same linked MPFR/GMP symbols used by Rug. The strings are
    // immutable, NUL-terminated library version constants, not owned buffers.
    unsafe extern "C" {
        fn mpfr_get_version() -> *const std::ffi::c_char;
        static __gmp_version: *const std::ffi::c_char;
    }
    unsafe {
        let mpfr = mpfr_get_version();
        let gmp = __gmp_version;
        ensure!(
            !mpfr.is_null() && !gmp.is_null(),
            "native version unavailable"
        );
        let mut result = std::ffi::CStr::from_ptr(mpfr).to_bytes_with_nul().to_vec();
        result.extend_from_slice(std::ffi::CStr::from_ptr(gmp).to_bytes_with_nul());
        Ok(result)
    }
}

fn integer_env(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(text) => text.parse().with_context(|| format!("invalid {name}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(e.into()),
    }
}

impl Checkpoint {
    pub(super) fn configured(
        diag: &[Float],
        off: &[Float],
        p: u32,
        options: TridiagQrOptions,
    ) -> Option<Self> {
        if !ALLOWED.with(std::cell::Cell::get)
            || matches!(
                std::env::var("XC_CACHE_MODE").as_deref(),
                Ok("refresh" | "verify" | "verify_against_reference" | "verify-against-reference")
            )
        {
            return None;
        }
        let result = (|| -> Result<Option<Self>> {
            let seconds = integer_env("XC_QR_CHECKPOINT_SECONDS", 120)?;
            if seconds == 0 {
                return Ok(None);
            }
            let directory = std::env::var_os("XC_RESEARCH_CHECKPOINT_DIR")
                .map(PathBuf::from)
                .or_else(|| {
                    xc_cache::ManagedArtifactCacheConfig::from_environment()
                        .ok()
                        .flatten()
                        .map(|c| c.cache_root.join("research-checkpoints"))
                });
            let Some(directory) = directory else {
                return Ok(None);
            };
            let cap = integer_env("XC_RESEARCH_CHECKPOINT_BYTES", 8 << 30)?;
            Ok(Some(Self::new(
                &directory,
                diag,
                off,
                p,
                options,
                cap,
                Duration::from_secs(seconds),
            )?))
        })();
        match result {
            Ok(value) => value,
            Err(e) => {
                xc_core::progress_message!(
                    "[HP QR] checkpoint unavailable: {e}; computing normally"
                );
                None
            }
        }
    }

    fn new(
        directory: &Path,
        diag: &[Float],
        off: &[Float],
        p: u32,
        options: TridiagQrOptions,
        cap: u64,
        interval: Duration,
    ) -> Result<Self> {
        let token = u64::from(p).div_ceil(4) + 64;
        let maximum_bytes = (diag.len() as u64)
            .checked_mul(2)
            .and_then(|n| n.checked_mul(token + 8))
            .and_then(|n| n.checked_add(128))
            .context("QR checkpoint size overflow")?;
        ensure!(maximum_bytes <= cap, "QR checkpoint exceeds byte budget");
        let scratch = xc_core::RunResources::current()
            .map(|r| r.try_reserve_working(65536 + 4 * token))
            .transpose()?;
        let mut hash = Sha256::new();
        hash.update(MAGIC);
        hash.update(hash_executable()?);
        hash.update(native_versions()?);
        hash.update(rug::float::exp_min().to_le_bytes());
        hash.update(rug::float::exp_max().to_le_bytes());
        hash.update(p.to_le_bytes());
        hash.update((options.max_iterations_per_eigenvalue as u64).to_le_bytes());
        hash.update((diag.len() as u64).to_le_bytes());
        for value in diag.iter().chain(off) {
            hash.update(value.prec().to_le_bytes());
            let text = value.to_string_radix(16, None);
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        let identity: [u8; 32] = hash.finalize().into();
        let name = identity
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Ok(Self {
            path: directory.join("qr").join(format!("{name}.state")),
            identity,
            maximum_bytes,
            interval,
            saved: Instant::now(),
            _scratch: scratch,
        })
    }

    /// Restores only after authenticating the complete file. A rejected file may
    /// have written into the scratch arrays; callers restore their original inputs.
    pub(super) fn load(
        &self,
        d: &mut [Float],
        e: &mut [Float],
        p: u32,
        options: TridiagQrOptions,
    ) -> Option<(usize, usize)> {
        if !self.path.exists() {
            return None;
        }
        let result = self.read(d, e, p, options);
        match result {
            Ok(position) => {
                xc_core::progress_message!(
                    "[HP QR] resumed dimension={}, deflated={}, sweeps={}",
                    d.len(),
                    position.0,
                    position.1
                );
                Some(position)
            }
            Err(error) => {
                xc_core::progress_message!("[HP QR] checkpoint rejected: {error}; restarting QR");
                None
            }
        }
    }

    fn read(
        &self,
        d: &mut [Float],
        e: &mut [Float],
        p: u32,
        options: TridiagQrOptions,
    ) -> Result<(usize, usize)> {
        let file = File::open(&self.path)?;
        ensure!(
            file.metadata()?.len() <= self.maximum_bytes,
            "checkpoint oversized"
        );
        let mut input = BufReader::new(file.take(self.maximum_bytes + 1));
        let mut hash = Sha256::new();
        fn bytes<const N: usize>(input: &mut impl Read, hash: &mut Sha256) -> Result<[u8; N]> {
            let mut b = [0; N];
            input.read_exact(&mut b)?;
            hash.update(b);
            Ok(b)
        }
        ensure!(
            &bytes::<8>(&mut input, &mut hash)? == MAGIC,
            "checkpoint format"
        );
        ensure!(
            bytes::<32>(&mut input, &mut hash)? == self.identity,
            "checkpoint identity"
        );
        let l = usize::try_from(u64::from_le_bytes(bytes(&mut input, &mut hash)?))?;
        let sweeps = usize::try_from(u64::from_le_bytes(bytes(&mut input, &mut hash)?))?;
        ensure!(
            l <= d.len()
                && sweeps <= options.max_iterations_per_eigenvalue
                && (l < d.len() || sweeps == 0),
            "checkpoint position"
        );
        for v in d.iter_mut().chain(e.iter_mut()) {
            let length = u64::from_le_bytes(bytes(&mut input, &mut hash)?);
            ensure!(
                length <= u64::from(p).div_ceil(4) + 64,
                "checkpoint scalar oversized"
            );
            let mut text = vec![0; usize::try_from(length)?];
            input.read_exact(&mut text)?;
            hash.update(&text);
            let value = Float::with_val(p, Float::parse_radix(&text, 16)?);
            ensure!(
                value.is_finite() && value.to_string_radix(16, None).as_bytes() == text,
                "checkpoint scalar is not canonical/exact"
            );
            v.assign(value);
        }
        ensure!(e.last().is_some_and(Float::is_zero), "checkpoint sentinel");
        let mut seal = [0; 32];
        input.read_exact(&mut seal)?;
        ensure!(
            <[u8; 32]>::from(hash.finalize()) == seal,
            "checkpoint digest"
        );
        ensure!(input.read(&mut [0; 1])? == 0, "checkpoint trailing data");
        Ok((l, sweeps))
    }

    pub(super) fn save_due(
        &mut self,
        l: usize,
        sweeps: usize,
        d: &[Float],
        e: &[Float],
        complete: bool,
    ) {
        if !complete && self.saved.elapsed() < self.interval {
            return;
        }
        // Avoid writing checkpoints for short solves that do not need recovery.
        if complete && self.saved.elapsed() < self.interval && !self.path.exists() {
            return;
        }
        if let Err(error) = self.write(l, sweeps, d, e) {
            xc_core::progress_message!(
                "[HP QR] checkpoint write unavailable: {error}; numerical calculation continues"
            );
        }
        self.saved = Instant::now();
    }

    fn write(&self, l: usize, sweeps: usize, d: &[Float], e: &[Float]) -> Result<()> {
        fs::create_dir_all(self.path.parent().context("checkpoint parent")?)?;
        let temp = self.path.with_extension(format!(
            "{}-{}.tmp",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| -> Result<()> {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            let mut writer = BufWriter::new(file);
            let mut hash = Sha256::new();
            let mut length = 32u64; // final digest
            let mut put = |bytes: &[u8]| -> Result<()> {
                length = length
                    .checked_add(bytes.len() as u64)
                    .context("checkpoint size overflow")?;
                ensure!(length <= self.maximum_bytes, "checkpoint byte limit");
                writer.write_all(bytes)?;
                hash.update(bytes);
                Ok(())
            };
            put(MAGIC)?;
            put(&self.identity)?;
            put(&(l as u64).to_le_bytes())?;
            put(&(sweeps as u64).to_le_bytes())?;
            for v in d.iter().chain(e) {
                ensure!(v.is_finite(), "nonfinite QR checkpoint state");
                let text = v.to_string_radix(16, None);
                put(&(text.len() as u64).to_le_bytes())?;
                put(text.as_bytes())?;
            }
            writer.write_all(&hash.finalize())?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
            drop(writer);
            // Atomic replacement leaves the preceding complete file intact if
            // writing is interrupted. Concurrent identical solves may overwrite
            // progress, but every accepted state follows the same arithmetic.
            fs::rename(&temp, &self.path)?;
            #[cfg(unix)]
            File::open(self.path.parent().context("checkpoint parent")?)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eigen::tridiag_qr_resume;

    fn matrix() -> (Vec<Float>, Vec<Float>) {
        let d = (0..19).map(|i| Float::with_val(160, i * i + 1)).collect();
        let mut e: Vec<_> = (0..18).map(|i| Float::with_val(160, i + 2)).collect();
        e.push(Float::with_val(160, -0.0));
        (d, e)
    }
    fn bits(values: &[Float]) -> Vec<String> {
        values.iter().map(|v| v.to_string_radix(16, None)).collect()
    }
    #[test]
    fn independent_verification_scope_restores_policy_even_on_unwind() {
        assert!(ALLOWED.with(std::cell::Cell::get));
        let result = std::panic::catch_unwind(|| {
            without(|| {
                assert!(!ALLOWED.with(std::cell::Cell::get));
                without(|| assert!(!ALLOWED.with(std::cell::Cell::get)));
                panic!("probe unwind");
            })
        });
        assert!(result.is_err());
        assert!(ALLOWED.with(std::cell::Cell::get));
    }
    #[test]
    fn interruption_at_each_boundary_resumes_identical_full_spectrum() {
        let root = xc_core::test_support::TestDir::new("qr-resume-boundaries");
        let (d, e) = matrix();
        let options = TridiagQrOptions::default();
        let c =
            Checkpoint::new(&root, &d, &e[..18], 160, options, 1 << 20, Duration::ZERO).unwrap();
        let mut boundaries = Vec::new();
        let reference =
            tridiag_qr_resume(d.clone(), e.clone(), 160, options, (0, 0), |l, s, d, e| {
                boundaries.push((l, s, d.to_vec(), e.to_vec()));
                Ok(())
            })
            .unwrap();
        assert!(boundaries.iter().any(|b| b.1 > 1));
        for (l, sweeps, saved_d, saved_e) in boundaries {
            c.write(l, sweeps, &saved_d, &saved_e).unwrap();
            let (mut rd, mut re) = matrix();
            let position = c.read(&mut rd, &mut re, 160, options).unwrap();
            assert_eq!(position, (l, sweeps));
            assert_eq!(bits(&rd), bits(&saved_d));
            assert_eq!(bits(&re), bits(&saved_e));
            let resumed =
                tridiag_qr_resume(rd, re, 160, options, position, |_, _, _, _| Ok(())).unwrap();
            assert_eq!(bits(&resumed), bits(&reference));
        }
    }

    #[test]
    fn sweep_budget_survives_resume_and_bad_or_stale_files_are_rejected() {
        let root = xc_core::test_support::TestDir::new("qr-resume-integrity");
        let (d, e) = matrix();
        let options = TridiagQrOptions {
            max_iterations_per_eigenvalue: 1,
        };
        let c =
            Checkpoint::new(&root, &d, &e[..18], 160, options, 1 << 20, Duration::ZERO).unwrap();
        let interrupted =
            tridiag_qr_resume(d.clone(), e.clone(), 160, options, (0, 0), |l, s, d, e| {
                c.write(l, s, d, e)?;
                anyhow::bail!("simulated interruption")
            });
        assert!(interrupted
            .unwrap_err()
            .to_string()
            .contains("interruption"));
        let (mut rd, mut re) = matrix();
        let position = c.read(&mut rd, &mut re, 160, options).unwrap();
        let error =
            tridiag_qr_resume(rd, re, 160, options, position, |_, _, _, _| Ok(())).unwrap_err();
        assert!(error.to_string().contains("hard limit of 1 sweeps"));
        let mut changed = d.clone();
        changed[0] += 1;
        let other = Checkpoint::new(
            &root,
            &changed,
            &e[..18],
            160,
            options,
            1 << 20,
            Duration::ZERO,
        )
        .unwrap();
        assert_ne!(c.identity, other.identity);
        let other = Checkpoint::new(
            &root,
            &d,
            &e[..18],
            160,
            TridiagQrOptions::default(),
            1 << 20,
            Duration::ZERO,
        )
        .unwrap();
        assert_ne!(c.identity, other.identity);
        assert!(Checkpoint::new(&root, &d, &e[..18], 160, options, 10, Duration::ZERO).is_err());
        let original = fs::read(&c.path).unwrap();
        for length in [0, 40, original.len() - 1] {
            fs::write(&c.path, &original[..length]).unwrap();
            let (mut rd, mut re) = matrix();
            assert!(c.read(&mut rd, &mut re, 160, options).is_err());
        }
        let mut altered = original.clone();
        *altered.last_mut().unwrap() ^= 1;
        fs::write(&c.path, altered).unwrap();
        let (mut rd, mut re) = matrix();
        assert!(c.read(&mut rd, &mut re, 160, options).is_err());
        fs::write(&c.path, &original).unwrap();
        // An unfinished staging file cannot displace the last committed state.
        fs::write(c.path.with_extension("abandoned.tmp"), b"partial").unwrap();
        assert!(c.read(&mut rd, &mut re, 160, options).is_ok());
    }
}
