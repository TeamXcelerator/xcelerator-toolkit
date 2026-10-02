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

/// Largest serialized research input a capture admits: prepared external
/// inputs, finite-diagnostic and energy requests, and retained trial
/// components. A fixed constant, so admission never changes an identity; the
/// declared working budget separately decides whether the work may run, and
/// work over that budget is reported as unavailable instead of truncated.
pub const RESEARCH_INPUT_MAXIMUM_BYTES: u64 = 4 << 30;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
        xc_core::progress_message!("research stage {label}: started");
        let (stop, rx) = mpsc::channel();
        let name = label.clone();
        // Heartbeats join any message capture of the stage's own thread.
        let capture = xc_core::current_message_capture();
        let thread = std::thread::spawn(move || {
            xc_core::with_message_capture(capture, || {
                let start = Instant::now();
                while rx
                    .recv_timeout(std::time::Duration::from_secs(30))
                    .is_err_and(|e| e == mpsc::RecvTimeoutError::Timeout)
                {
                    xc_core::progress_message!(
                        "research stage {name}: active, {:.1}s elapsed",
                        start.elapsed().as_secs_f64()
                    );
                }
            })
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
        xc_core::progress_message!(
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
                    xc_core::progress_message!("research checkpoint {display_label}: reused");
                }
                Ok(Some(value))
            }
            Err(e) => {
                xc_core::progress_message!(
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
        // The newly computed checkpoint replaces the old one. An interrupted
        // replacement has no matching seal and is rejected by the reader.
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
                xc_core::progress_message!(
                    "research checkpoint unavailable: {e}; numerical result retained"
                );
            }
            rows
        };
        result.extend(rows);
    }
    Ok(result)
}

type LookAheadJob<T> = Box<dyn FnOnce() -> T + Send>;
type LookAheadOutcome<T> = std::thread::Result<(T, Vec<String>)>;

enum LookAheadSlot<T> {
    Queued(LookAheadJob<T>),
    Running,
    Done(LookAheadOutcome<T>),
    /// Taken, released to its owner, or never runnable here.
    Closed,
}

struct LookAheadState<K, T> {
    slots: Vec<(String, K, LookAheadSlot<T>)>,
    stopped: bool,
}

struct LookAheadShared<K, T> {
    state: std::sync::Mutex<LookAheadState<K, T>>,
    changed: std::sync::Condvar,
}

impl<K, T> LookAheadShared<K, T> {
    fn lock(&self) -> std::sync::MutexGuard<'_, LookAheadState<K, T>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
std::thread_local! {
    static LOOKAHEAD_RESULTS_TAKEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Pure computations started ahead of their turn on one background thread,
/// in the order given. A job must be a deterministic function of the inputs
/// recorded in its key, read no cache and write no file. Its owner claims the
/// result where the serial code would compute it, after checking that the
/// key equals the inputs it would use there; on any mismatch, or for a job
/// that has not started, the owner computes inline as before. Everything
/// observable stays with the owner: cache access, staging and errors happen
/// at the serial position, and the job's progress messages are held and
/// delivered there in order. A job's panic resumes on the owner's thread when
/// it claims that job.
///
/// Jobs run in the configuration of their owner: on the global Rayon pool,
/// or, for an owner inside another pool, on a private pool with the same
/// number of workers (waiting on the lane never occupies a worker the lane
/// needs); an active full-parallel HP policy is installed for them; and a job
/// runs only where the MPFR exponent range equals the owner's. The lane does
/// not start under a safe-capped policy. Arb/FLINT keep per-thread constant
/// caches, so jobs must not reach Arb.
pub(crate) struct LookAhead<K, T> {
    shared: std::sync::Arc<LookAheadShared<K, T>>,
    range: (i32, i32),
    thread: Option<std::thread::JoinHandle<()>>,
}

impl<K: Send + 'static, T: Send + 'static> LookAhead<K, T> {
    pub(crate) fn start(jobs: Vec<(String, K, LookAheadJob<T>)>) -> Option<Self> {
        if jobs.is_empty() {
            return None;
        }
        let policy = xc_numerics::hp_runtime::active_policy();
        if policy
            .as_ref()
            .is_some_and(|p| p.mode != xc_numerics::hp_runtime::HpRuntimeMode::FullParallel)
        {
            return None;
        }
        let pool = match rayon::current_thread_index() {
            Some(_) => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(rayon::current_num_threads())
                    .thread_name(|i| format!("xc-capture-lookahead-{i}"))
                    .build()
                    .ok()?,
            ),
            None => None,
        };
        let range = (rug::float::exp_min(), rug::float::exp_max());
        let shared = std::sync::Arc::new(LookAheadShared {
            state: std::sync::Mutex::new(LookAheadState {
                slots: jobs
                    .into_iter()
                    .map(|(id, key, job)| (id, key, LookAheadSlot::Queued(job)))
                    .collect(),
                stopped: false,
            }),
            changed: std::sync::Condvar::new(),
        });
        let worker = std::sync::Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("xc-capture-lookahead".into())
            .stack_size(64 << 20)
            .spawn(move || {
                // One job with held messages, under the owner's policy and
                // exponent range; `None` releases it to the owner.
                let execute = |job: LookAheadJob<T>, capture: &xc_core::MessageCapture| {
                    xc_core::with_message_capture(Some(capture.clone()), || {
                        if (rug::float::exp_min(), rug::float::exp_max()) != range {
                            return None;
                        }
                        match &policy {
                            Some(policy) => {
                                xc_numerics::hp_runtime::run_hp_with_policy(policy, job).ok()
                            }
                            None => Some(job()),
                        }
                    })
                };
                loop {
                    let job = {
                        let mut state = worker.lock();
                        let runnable = !state.stopped;
                        let next = state
                            .slots
                            .iter_mut()
                            .find(|(_, _, slot)| matches!(slot, LookAheadSlot::Queued(_)));
                        match next {
                            Some((_, _, slot)) if runnable => {
                                match std::mem::replace(slot, LookAheadSlot::Running) {
                                    LookAheadSlot::Queued(job) => job,
                                    _ => unreachable!("selected a queued job"),
                                }
                            }
                            _ => {
                                for (_, _, slot) in &mut state.slots {
                                    if matches!(slot, LookAheadSlot::Queued(_)) {
                                        *slot = LookAheadSlot::Closed;
                                    }
                                }
                                worker.changed.notify_all();
                                return;
                            }
                        }
                    };
                    let capture = xc_core::MessageCapture::default();
                    let outcome =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &pool {
                            Some(pool) => pool.install(|| execute(job, &capture)),
                            None => execute(job, &capture),
                        }));
                    let mut state = worker.lock();
                    if let Some((_, _, slot)) = state
                        .slots
                        .iter_mut()
                        .find(|(_, _, slot)| matches!(slot, LookAheadSlot::Running))
                    {
                        *slot = match outcome {
                            Ok(Some(value)) => LookAheadSlot::Done(Ok((value, capture.take()))),
                            Ok(None) => LookAheadSlot::Closed,
                            Err(panic) => LookAheadSlot::Done(Err(panic)),
                        };
                    }
                    worker.changed.notify_all();
                }
            })
            .ok()?;
        Some(Self {
            shared,
            range,
            thread: Some(thread),
        })
    }

    /// Take the result of `id` when `matches` accepts the inputs it was
    /// computed from, waiting while it runs; deliver its held messages here.
    /// `None` means compute inline. A job not yet started is withdrawn.
    pub(crate) fn claim(&self, id: &str, matches: impl FnOnce(&K) -> bool) -> Option<T>
    where
        K: Clone,
    {
        let usable = (rug::float::exp_min(), rug::float::exp_max()) == self.range;
        // Compare inputs outside the lock; a key never changes.
        let key = self
            .shared
            .lock()
            .slots
            .iter()
            .find(|(name, _, _)| name == id)
            .map(|(_, key, _)| key.clone())?;
        let accepted = usable && matches(&key);
        let mut state = self.shared.lock();
        let outcome = loop {
            let (_, _, slot) = state.slots.iter_mut().find(|(name, _, _)| name == id)?;
            if !accepted {
                if matches!(slot, LookAheadSlot::Queued(_)) {
                    *slot = LookAheadSlot::Closed;
                }
                return None;
            }
            match std::mem::replace(slot, LookAheadSlot::Closed) {
                LookAheadSlot::Running => {
                    *slot = LookAheadSlot::Running;
                    state = self
                        .shared
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                LookAheadSlot::Done(outcome) => break outcome,
                LookAheadSlot::Queued(_) | LookAheadSlot::Closed => return None,
            }
        };
        drop(state);
        match outcome {
            Ok((value, messages)) => {
                for message in messages {
                    xc_core::emit_message(message);
                }
                #[cfg(test)]
                LOOKAHEAD_RESULTS_TAKEN.with(|taken| taken.set(taken.get() + 1));
                Some(value)
            }
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}

#[cfg(test)]
impl<K, T> LookAhead<K, T> {
    /// Wait until no job is queued or running (tests only).
    pub(crate) fn wait_idle(&self) {
        let mut state = self.shared.lock();
        while state
            .slots
            .iter()
            .any(|(_, _, slot)| matches!(slot, LookAheadSlot::Queued(_) | LookAheadSlot::Running))
        {
            state = self
                .shared
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

impl<K, T> Drop for LookAhead<K, T> {
    fn drop(&mut self) {
        self.shared.lock().stopped = true;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Results taken from look-ahead lanes on this thread (tests only).
#[cfg(test)]
pub(crate) fn lookahead_results_taken() -> usize {
    LOOKAHEAD_RESULTS_TAKEN.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    type Jobs<T> = Vec<(String, u32, LookAheadJob<T>)>;

    // A job that reports its start, then waits for the test to release it.
    fn gated<T: Send + 'static>(
        value: impl FnOnce() -> T + Send + 'static,
    ) -> (LookAheadJob<T>, mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (started, on_start) = mpsc::channel();
        let (release, gate) = mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        let job: LookAheadJob<T> = Box::new(move || {
            started.send(()).unwrap();
            gate.lock().unwrap().recv().unwrap();
            value()
        });
        (job, on_start, release)
    }

    #[test]
    fn lookahead_delivers_results_and_messages_at_the_claim_in_order() {
        let ran = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (first, started, release) = gated(|| {
            xc_core::progress_message!("first job message");
            let _stage = Stage::new("lookahead test stage");
            10
        });
        let counted = |value| -> LookAheadJob<u32> {
            let ran = Arc::clone(&ran);
            Box::new(move || {
                ran.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                value
            })
        };
        let jobs: Jobs<u32> = vec![
            ("a".into(), 1, first),
            ("b".into(), 2, counted(20)),
            ("c".into(), 3, counted(30)),
        ];
        let lane = LookAhead::start(jobs).unwrap();
        started.recv().unwrap();
        // Not started: withdrawn and computed inline by the owner.
        assert_eq!(lane.claim("c", |key| *key == 3), None);
        // Inputs differ from the recorded key: withdrawn as well.
        assert_eq!(lane.claim("b", |key| *key == 99), None);
        assert_eq!(lane.claim("unknown", |_| true), None);
        // The owner waits for a running job, which is released meanwhile.
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            release.send(()).unwrap();
        });
        let capture = xc_core::MessageCapture::default();
        let taken = lookahead_results_taken();
        let first = xc_core::with_message_capture(Some(capture.clone()), || {
            lane.claim("a", |key| *key == 1)
        });
        releaser.join().unwrap();
        assert_eq!(first, Some(10));
        assert_eq!(lookahead_results_taken(), taken + 1);
        let messages = capture.take();
        assert_eq!(messages.len(), 3, "{messages:?}");
        assert_eq!(messages[0], "first job message");
        assert_eq!(messages[1], "research stage lookahead test stage: started");
        assert!(messages[2].starts_with("research stage lookahead test stage: ended after "));
        // A taken result is gone, and withdrawn jobs never ran.
        assert_eq!(lane.claim("a", |_| true), None);
        drop(lane);
        assert_eq!(ran.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn lookahead_panics_resume_at_the_claim_and_jobs_keep_the_owner_configuration() {
        let (job, started, release) = gated(|| -> u32 { panic!("look-ahead job failure") });
        let lane = LookAhead::start(vec![("p".to_owned(), 0, job)]).unwrap();
        started.recv().unwrap();
        release.send(()).unwrap();
        let panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lane.claim("p", |_| true)))
                .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"look-ahead job failure")
        );
        // Started run to completion and claimed: (workers, policy, held message).
        let observe = || {
            let (job, started, release) = gated(|| {
                xc_core::progress_message!("from the job");
                (
                    rayon::current_num_threads(),
                    xc_numerics::hp_runtime::active_policy(),
                )
            });
            let lane = LookAhead::start(vec![("o".to_owned(), 0, job)])?;
            started.recv().unwrap();
            release.send(()).unwrap();
            let capture = xc_core::MessageCapture::default();
            let observed =
                xc_core::with_message_capture(Some(capture.clone()), || lane.claim("o", |_| true));
            assert_eq!(capture.take(), ["from the job"]);
            observed
        };
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(observe), Some((threads, None)));
        }
        let full = xc_numerics::hp_runtime::HpRuntimePolicy::default();
        let observed = xc_numerics::hp_runtime::run_hp_with_policy(&full, observe).unwrap();
        assert_eq!(observed.map(|(_, policy)| policy), Some(Some(full)));
        let safe =
            xc_numerics::hp_runtime::HpRuntimePolicy::safe_capped(2, 8 << 20, "test").unwrap();
        xc_numerics::hp_runtime::run_hp_with_policy(&safe, || {
            let jobs: Jobs<u32> = vec![("x".into(), 0, Box::new(|| 1))];
            assert!(LookAhead::start(jobs).is_none());
        })
        .unwrap();
    }

    #[test]
    fn checkpoints_reject_legacy_content_only_identity() {
        let root_dir = xc_core::test_support::TestDir::new("build-bound");
        let root = root_dir.to_path_buf();
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
        let root_dir = xc_core::test_support::TestDir::new("checkpoint-test");
        let root = root_dir.to_path_buf();
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
