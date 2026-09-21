//! Local, content-bound recovery and progress. Runtime paths and elapsed times
//! never enter mathematical identities or published numerical payloads.
use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::mpsc,
    time::Instant,
};
use xc_cache::ContentDigest;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureResourcePolicy {
    pub maximum_working_bytes: u64,
    pub maximum_output_bytes: u64,
    pub maximum_checkpoint_bytes: u64,
    pub root_block_rows: usize,
}
impl Default for CaptureResourcePolicy {
    fn default() -> Self {
        Self {
            maximum_working_bytes: 8 << 30,
            maximum_output_bytes: 8 << 30,
            maximum_checkpoint_bytes: 8 << 30,
            root_block_rows: 128,
        }
    }
}
impl CaptureResourcePolicy {
    /// Explicit limits; no assumption that every process owns all host memory.
    pub fn from_environment() -> Result<Self> {
        let mut p = Self::default();
        for (name, value) in [
            ("XC_RESEARCH_WORKING_BYTES", &mut p.maximum_working_bytes),
            ("XC_RESEARCH_OUTPUT_BYTES", &mut p.maximum_output_bytes),
            (
                "XC_RESEARCH_CHECKPOINT_BYTES",
                &mut p.maximum_checkpoint_bytes,
            ),
        ] {
            if let Ok(s) = std::env::var(name) {
                *value = s.parse().with_context(|| name)?;
            }
            if *value == 0 {
                bail!("{name} must be positive");
            }
        }
        if let Ok(s) = std::env::var("XC_RESEARCH_ROOT_BLOCK_ROWS") {
            p.root_block_rows = s.parse()?;
        }
        if p.root_block_rows == 0 || p.root_block_rows > 4096 {
            bail!("root block size must be 1..4096");
        }
        Ok(p)
    }
}

pub(crate) struct Stage {
    stop: mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    label: String,
    start: Instant,
}
impl Stage {
    pub(crate) fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        eprintln!("research stage {label}: started");
        let (stop, rx) = mpsc::channel();
        let name = label.clone();
        let thread = std::thread::spawn(move || {
            let start = Instant::now();
            while rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .is_err_and(|e| e == mpsc::RecvTimeoutError::Timeout)
            {
                eprintln!(
                    "research stage {name}: active, {:.1}s elapsed",
                    start.elapsed().as_secs_f64()
                );
            }
        });
        Self {
            stop,
            thread: Some(thread),
            label,
            start: Instant::now(),
        }
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
        eprintln!(
            "research stage {}: ended after {:.3}s",
            self.label,
            self.start.elapsed().as_secs_f64()
        );
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Seal {
    schema_version: u32,
    identity: ContentDigest,
    payload: ContentDigest,
    bytes: u64,
}

// Checkpoint keys may contain full-precision coordinates. Keep progress readable
// without changing the exact key used for storage and validation.
fn checkpoint_label(label: &str) -> String {
    if label.len() <= 96 && !label.chars().any(char::is_control) {
        label.to_owned()
    } else {
        format!("sha256:{}", ContentDigest::sha256(label.as_bytes()).0)
    }
}

// Stage heartbeats remain visible. Individual checkpoint I/O is diagnostic
// detail; enabling it never changes checkpoint identity or validation.
fn checkpoint_logging() -> bool {
    std::env::var("XC_RESEARCH_CHECKPOINT_LOG").is_ok_and(|v| v == "detail")
}

fn identity_digest<T: Serialize>(value: &T) -> Result<ContentDigest> {
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.update(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut w = HashWriter(Sha256::new());
    serde_json::to_writer(
        &mut w,
        &(
            env!("CARGO_PKG_VERSION"),
            env!("XC_CHECKPOINT_BUILD_ID"),
            cfg!(feature = "arb"),
            value,
        ),
    )?;
    Ok(ContentDigest(format!("{:x}", w.0.finalize())))
}
/// Checkpoints are trusted local execution data, integrity-checked on reuse.
/// They are not portable mathematical certificates. Source digests bind every key.
pub(crate) struct Checkpoints {
    directory: Option<PathBuf>,
    identity: ContentDigest,
    max_bytes: u64,
    verbose: bool,
}
impl Checkpoints {
    pub(crate) fn new<T: Serialize>(identity: &T) -> Result<Self> {
        let policy = CaptureResourcePolicy::from_environment()?;
        Ok(Self {
            directory: std::env::var_os("XC_RESEARCH_CHECKPOINT_DIR")
                .map(PathBuf::from)
                .or_else(|| {
                    xc_cache::ManagedArtifactCacheConfig::from_environment()
                        .ok()
                        .flatten()
                        .map(|c| c.cache_root.join("research-checkpoints"))
                }),
            identity: identity_digest(identity)?,
            max_bytes: policy.maximum_checkpoint_bytes,
            verbose: checkpoint_logging(),
        })
    }
    pub(crate) fn enabled(&self) -> bool {
        self.directory.is_some()
    }
    pub(crate) fn local<T: Serialize>(identity: &T, directory: PathBuf) -> Result<Self> {
        Ok(Self {
            directory: Some(directory),
            identity: identity_digest(identity)?,
            max_bytes: CaptureResourcePolicy::from_environment()?.maximum_checkpoint_bytes,
            verbose: checkpoint_logging(),
        })
    }
    fn path(&self, label: &str) -> Option<PathBuf> {
        self.directory.as_ref().map(|d| {
            d.join(&self.identity.0)
                .join(ContentDigest::sha256(label.as_bytes()).0)
        })
    }
    pub(crate) fn load<T: DeserializeOwned>(&self, label: &str) -> Result<Option<T>> {
        let Some(path) = self.path(label) else {
            return Ok(None);
        };
        let seal_path = path.with_extension("seal.json");
        if !path.exists() || !seal_path.exists() {
            return Ok(None);
        }
        let display_label = checkpoint_label(label);
        let result = (|| -> Result<T> {
            if fs::metadata(&seal_path)?.len() > 4096 {
                bail!("checkpoint seal oversized");
            }
            let seal: Seal = serde_json::from_reader(BufReader::new(File::open(&seal_path)?))?;
            if seal.schema_version != 1
                || seal.identity != self.identity
                || seal.bytes > self.max_bytes
                || fs::metadata(&path)?.len() != seal.bytes
            {
                bail!("checkpoint binding or size mismatch");
            }
            let _stage = self
                .verbose
                .then(|| Stage::new(format!("checkpoint read/validation {display_label}")));
            let mut reader = BufReader::with_capacity(
                1024 * 1024,
                DigestReader {
                    reader: File::open(&path)?,
                    digest: Sha256::new(),
                    bytes: 0,
                    maximum: seal.bytes,
                },
            );
            let value = serde_json::from_reader(&mut reader)?;
            let reader = reader.into_inner();
            if reader.bytes != seal.bytes
                || format!("{:x}", reader.digest.finalize()) != seal.payload.0
            {
                bail!("checkpoint hash mismatch");
            }
            Ok(value)
        })();
        match result {
            Ok(value) => {
                if self.verbose {
                    eprintln!("research checkpoint {display_label}: reused");
                }
                Ok(Some(value))
            }
            Err(e) => {
                eprintln!(
                    "research checkpoint {display_label}: rejected ({e}); recomputing diagnostic stage"
                );
                Ok(None)
            }
        }
    }
    pub(crate) fn save<T: Serialize>(&self, label: &str, value: &T) -> Result<()> {
        let Some(path) = self.path(label) else {
            return Ok(());
        };
        fs::create_dir_all(path.parent().expect("checkpoint parent"))?;
        // Unique temporary names also allow independent diagnostics/processes.
        let suffix = format!(
            "{}.{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let temp = path.with_extension(format!("{suffix}.tmp"));
        let _stage = self
            .verbose
            .then(|| Stage::new(format!("checkpoint write {}", checkpoint_label(label))));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let mut writer = DigestWriter {
            writer: BufWriter::new(file),
            digest: Sha256::new(),
            bytes: 0,
            max: self.max_bytes,
        };
        let result = serde_json::to_writer(&mut writer, value);
        if let Err(e) = result {
            drop(writer);
            let _ = fs::remove_file(&temp);
            return Err(e.into());
        }
        writer.flush()?;
        writer.writer.get_ref().sync_all()?;
        let seal = Seal {
            schema_version: 1,
            identity: self.identity.clone(),
            payload: ContentDigest(format!("{:x}", writer.digest.finalize())),
            bytes: writer.bytes,
        };
        drop(writer.writer);
        // A pre-existing valid checkpoint wins; interrupted checkpoints lack a seal.
        if path.exists() {
            fs::remove_file(&path)?;
        }
        fs::rename(&temp, &path)?;
        let seal_tmp = path.with_extension(format!("{suffix}.seal.tmp"));
        fs::write(&seal_tmp, serde_json::to_vec(&seal)?)?;
        let seal_path = path.with_extension("seal.json");
        if seal_path.exists() {
            fs::remove_file(&seal_path)?;
        }
        fs::rename(seal_tmp, seal_path)?;
        Ok(())
    }
}
struct DigestReader<R> {
    reader: R,
    digest: Sha256,
    bytes: u64,
    maximum: u64,
}
impl<R: Read> Read for DigestReader<R> {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(b)?;
        self.bytes = self.bytes.saturating_add(n as u64);
        if self.bytes > self.maximum {
            return Err(std::io::Error::other("checkpoint grew while reading"));
        }
        self.digest.update(&b[..n]);
        Ok(n)
    }
}
struct DigestWriter<W> {
    writer: W,
    digest: Sha256,
    bytes: u64,
    max: u64,
}
impl<W: Write> Write for DigestWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.bytes.saturating_add(bytes.len() as u64) > self.max {
            return Err(std::io::Error::other("checkpoint byte budget exceeded"));
        }
        let n = self.writer.write(bytes)?;
        self.digest.update(&bytes[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

pub(crate) fn row_blocks<F>(
    store: &Checkpoints,
    count: usize,
    compute: F,
) -> Result<Vec<super::extended_research::AnalysisRow>>
where
    F: Fn(usize) -> Result<super::extended_research::AnalysisRow> + Sync,
{
    use rayon::prelude::*;
    let block = CaptureResourcePolicy::from_environment()?.root_block_rows;
    let mut result = Vec::with_capacity(count);
    for start in (0..count).step_by(block) {
        let end = (start + block).min(count);
        let key = format!("rows-{start}-{end}");
        let rows = if let Some(rows) = store
            .load::<Vec<super::extended_research::AnalysisRow>>(&key)?
            .filter(|rows| rows.len() == end - start)
        {
            rows
        } else {
            let _stage = Stage::new(format!("compute {key}/{count}"));
            let rows = (start..end)
                .into_par_iter()
                .map(&compute)
                .collect::<Result<Vec<_>>>()?;
            if let Err(e) = store.save(&key, &rows) {
                eprintln!("research checkpoint unavailable: {e}; numerical result retained");
            }
            rows
        };
        result.extend(rows);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoints_reject_legacy_content_only_identity() {
        let root = std::env::temp_dir().join(format!("xc-build-bound-{}", std::process::id()));
        let key = ("legacy", "same inputs");
        let legacy = Checkpoints {
            directory: Some(root.clone()),
            identity: ContentDigest::sha256(&serde_json::to_vec(&key).unwrap()),
            max_bytes: 1024,
            verbose: false,
        };
        legacy.save("block", &vec![1, 2, 3]).unwrap();
        let current = Checkpoints::local(&key, root.clone()).unwrap();
        assert_ne!(legacy.identity, current.identity);
        assert!(current.load::<Vec<u32>>("block").unwrap().is_none());
        assert_eq!(env!("XC_CHECKPOINT_BUILD_ID").len(), 64);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn checkpoint_reuse_binds_sources_and_rejects_corruption_and_resource_excess() {
        let root = std::env::temp_dir().join(format!(
            "xc-checkpoint-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = Checkpoints {
            directory: Some(root.clone()),
            identity: ContentDigest::sha256(b"source A and policy"),
            max_bytes: 1024,
            verbose: false,
        };
        store
            .save("block", &vec!["1.234567890123456789012345678901234567890"])
            .unwrap();
        let a: Vec<String> = store.load("block").unwrap().unwrap();
        assert_eq!(a[0], "1.234567890123456789012345678901234567890");
        let other = Checkpoints {
            directory: Some(root.clone()),
            identity: ContentDigest::sha256(b"source B and policy"),
            max_bytes: 1024,
            verbose: false,
        };
        assert!(other.load::<Vec<String>>("block").unwrap().is_none());
        let path = store.path("block").unwrap();
        std::fs::write(path, b"[]").unwrap();
        assert!(store.load::<Vec<String>>("block").unwrap().is_none());
        assert!(store.save("too-large", &"x".repeat(1025)).is_err());
        assert!(store.load::<String>("too-large").unwrap().is_none());
        std::fs::remove_dir_all(root).unwrap();
    }
}
