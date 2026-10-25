// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// Data for a caller-authorized target provider. The Toolkit does not interpret
/// `input` or provide the implementation of the caller's research function.
/// Precision-bearing numbers in `input` must be decimal strings. JSON integer
/// values fitting i64/u64 are preserved exactly; floating JSON numbers are
/// rejected before hashing or transmission because their digits may be lost.
/// No executable path or command can be specified by this data object.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalProfileSpec {
    pub lambda_squared: String,
    pub evaluation_precision_bits: u32,
    pub provider_sha256: String,
    pub input: serde_json::Value,
}
impl ExternalProfileSpec {
    pub(super) fn validate(&self) -> Result<()> {
        let c = xc_core::DecimalLiteral::new(&self.lambda_squared)
            .context("invalid external cutoff")?;
        anyhow::ensure!(
            c.cmp_numeric(&xc_core::DecimalLiteral::new("1")?)?.is_gt()
                && (64..=1_000_000).contains(&self.evaluation_precision_bits),
            "invalid external target policy"
        );
        anyhow::ensure!(
            self.provider_sha256.len() == 64
                && self.provider_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid provider digest"
        );
        validate_exact_input(&self.input)?;
        Ok(())
    }
}

fn validate_exact_input(value: &serde_json::Value) -> Result<()> {
    match value {
        serde_json::Value::Number(number) => anyhow::ensure!(
            number.is_i64() || number.is_u64(),
            "external target input numbers must be exact i64/u64 integers; encode decimals and larger integers as strings"
        ),
        serde_json::Value::Array(values) => {
            for value in values {
                validate_exact_input(value)?;
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                validate_exact_input(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Own cleanup from the moment of spawn, including failures during setup.
#[derive(Debug)]
struct ProviderChild(Child);

impl Drop for ProviderChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[derive(Debug)]
struct Connection {
    child: ProviderChild,
    requests: mpsc::Sender<Vec<u8>>,
    writes: mpsc::Receiver<std::io::Result<()>>,
    replies: mpsc::Receiver<std::result::Result<serde_json::Value, String>>,
    timeout: Duration,
    failed: bool,
    next_request_id: u64,
    precision_bits: u32,
    _executable_file: File,
}

struct RequestFrame {
    bytes: Vec<u8>,
    maximum: usize,
}

struct InputSize {
    bytes: u64,
    maximum: u64,
}

impl Write for InputSize {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|length| *length <= self.maximum)
            .ok_or_else(|| {
                std::io::Error::other("target input exceeds its configured byte limit")
            })?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn admit_input(input: &serde_json::Value, maximum: u64) -> Result<()> {
    // Count without constructing a second large buffer or starting a provider.
    serde_json::to_writer(&mut InputSize { bytes: 0, maximum }, input)?;
    Ok(())
}

impl Write for RequestFrame {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let length = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("target request length overflows"))?;
        if length > self.maximum {
            return Err(std::io::Error::other(
                "target request exceeds its configured byte limit",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| std::io::Error::other("cannot reserve target request memory"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn request_frame(request: &serde_json::Value, maximum: u64) -> Result<Vec<u8>> {
    let mut frame = RequestFrame {
        bytes: Vec::new(),
        maximum: usize::try_from(maximum).context("target request exceeds address space")?,
    };
    serde_json::to_writer(&mut frame, request)?;
    frame.write_all(b"\n")?;
    Ok(frame.bytes)
}

impl Connection {
    fn exchange(&mut self, message: &serde_json::Value) -> Result<serde_json::Value> {
        anyhow::ensure!(
            !self.failed,
            "external target provider is unavailable after an earlier failure"
        );
        let started = Instant::now();
        let result = (|| {
            anyhow::ensure!(
                matches!(self.replies.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "external provider sent an unsolicited response or disconnected"
            );
            let mut entropy = [0u8; 32];
            getrandom::fill(&mut entropy)
                .map_err(|_| anyhow::anyhow!("target request entropy unavailable"))?;
            let nonce: String = entropy.iter().map(|byte| format!("{byte:02x}")).collect();
            let id = self.next_request_id;
            self.next_request_id = id
                .checked_add(1)
                .context("target request counter exhausted")?;
            let mut request = message.clone();
            let fields = request.as_object_mut().context("invalid target request")?;
            fields.insert("request_id".into(), id.into());
            fields.insert("protocol_version".into(), 2.into());
            fields.insert("request_nonce".into(), nonce.clone().into());
            fields.insert("precision_bits".into(), self.precision_bits.into());
            let frame = request_frame(&request, super::input::target_request_maximum_bytes()?)?;
            let remaining = || {
                self.timeout
                    .checked_sub(started.elapsed())
                    .context("external target provider timed out")
            };
            remaining()?;
            self.requests
                .send(frame)
                .context("external target writer disconnected")?;
            self.writes
                .recv_timeout(remaining()?)
                .context("external target request timed out or disconnected")?
                .context("cannot write external target request")?;
            let reply = self
                .replies
                .recv_timeout(remaining()?)
                .map_err(|_| anyhow::anyhow!("external target provider timed out or disconnected"))?
                .map_err(|_| {
                    anyhow::anyhow!("external target provider returned an invalid response")
                })?;
            validate_reply(
                &reply,
                id,
                self.precision_bits,
                &nonce,
                message["operation"]
                    .as_str()
                    .context("invalid target operation")?,
            )?;
            Ok(reply)
        })();
        if result.is_err() {
            self.failed = true;
            let _ = self.child.0.kill();
        }
        result
    }
}

fn request_writer(
    mut input: BufWriter<ChildStdin>,
) -> (mpsc::Sender<Vec<u8>>, mpsc::Receiver<std::io::Result<()>>) {
    let (requests, frames) = mpsc::channel::<Vec<u8>>();
    let (completed, writes) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        while let Ok(frame) = frames.recv() {
            let result = input.write_all(&frame).and_then(|()| input.flush());
            let failed = result.is_err();
            if completed.send(result).is_err() || failed {
                break;
            }
        }
    });
    (requests, writes)
}

// Precision is a provider declaration, not proof of accuracy. In particular,
// exact decimal values such as 0.5 must not fail a digit-count heuristic.
fn validate_reply(
    reply: &serde_json::Value,
    id: u64,
    bits: u32,
    nonce: &str,
    operation: &str,
) -> Result<()> {
    anyhow::ensure!(
        reply["protocol_version"].as_u64() == Some(2),
        "target provider protocol version mismatch: expected protocol 2 (with request_nonce)"
    );
    anyhow::ensure!(
        reply["request_nonce"].as_str() == Some(nonce),
        "target provider request nonce mismatch"
    );
    anyhow::ensure!(
        reply["request_id"].as_u64() == Some(id),
        "target provider request/reply mismatch"
    );
    anyhow::ensure!(
        reply["operation"].as_str() == Some(operation),
        "target provider operation mismatch"
    );
    anyhow::ensure!(
        reply["precision_bits"].as_u64() == Some(u64::from(bits)),
        "target provider working precision mismatch"
    );
    anyhow::ensure!(
        reply.get("error").is_none(),
        "external target provider rejected the request"
    );
    match operation {
        "initialize" => anyhow::ensure!(
            reply["ready"].as_bool() == Some(true),
            "external target provider did not initialize"
        ),
        "evaluate" => anyhow::ensure!(
            reply["value"].as_str().is_some(),
            "external target provider omitted its value"
        ),
        _ => anyhow::bail!("invalid target operation"),
    }
    Ok(())
}

fn verified_executable(path: &Path, digest: &str) -> Result<File> {
    anyhow::ensure!(
        path.is_absolute(),
        "target provider must be an absolute executable path"
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Keep the verified image open without write/delete sharing while used.
        options.share_mode(1);
    }
    let mut file = options
        .open(path)
        .context("cannot open authorized target provider")?;
    anyhow::ensure!(
        file.metadata()?.is_file(),
        "target provider must be a regular file"
    );
    let mut bytes = Vec::new();
    (&mut file)
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= 256 * 1024 * 1024,
        "target provider exceeds executable size limit"
    );
    anyhow::ensure!(
        xc_cache::ContentDigest::sha256(&bytes).0 == digest,
        "external target provider executable digest mismatch"
    );
    Ok(file)
}

fn diagnostic_stderr() -> Result<Stdio> {
    let Some(directory) = std::env::var_os("XC_TARGET_PROVIDER_STDERR_DIR") else {
        return Ok(Stdio::null());
    };
    let directory = Path::new(&directory);
    anyhow::ensure!(
        directory.is_absolute() && directory.is_dir(),
        "target provider stderr directory must be an existing absolute directory"
    );
    static NEXT_LOG: AtomicU64 = AtomicU64::new(0);
    for _ in 0..1024 {
        let name = format!(
            "provider-{}-{}.stderr.log",
            std::process::id(),
            NEXT_LOG.fetch_add(1, Ordering::Relaxed)
        );
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(directory.join(name)) {
            Ok(file) => return Ok(Stdio::from(file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => anyhow::bail!("cannot create private target provider diagnostic log"),
        }
    }
    anyhow::bail!("target provider diagnostic log name budget exhausted")
}

/// Replies of providers the operator declared pure, shared by every evaluator
/// in this process. Bounded by entries and by a logical byte budget (the
/// lengths of the retained key and value strings, not their allocations); an
/// evicted or absent entry is asked again.
const PURE_REPLY_MEMO_ENTRIES: usize = 1 << 20;
const PURE_REPLY_MEMO_BYTES: usize = 512 << 20;

static PURE_REPLIES: std::sync::OnceLock<Mutex<PureReplies>> = std::sync::OnceLock::new();

#[derive(Default)]
struct PureReplies {
    values: std::collections::HashMap<String, String>,
    order: std::collections::VecDeque<String>,
    bytes: usize,
}

impl PureReplies {
    /// Remember a reply within `entries` and `bytes`, evicting the oldest
    /// first. An entry larger than the whole byte budget is not remembered,
    /// and nothing is evicted for it.
    fn remember(&mut self, key: String, value: &str, entries: usize, bytes: usize) {
        let size = 2 * key.len() + value.len();
        if size > bytes || self.values.contains_key(&key) {
            return;
        }
        while self.order.len() >= entries || self.bytes + size > bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.values.remove(&oldest) {
                self.bytes -= 2 * oldest.len() + evicted.len();
            }
        }
        self.order.push_back(key.clone());
        self.values.insert(key, value.to_owned());
        self.bytes += size;
    }
}

/// Memo-key prefix for one pure provider, target input and precision: a
/// fixed-size digest, so no key carries the (possibly megabyte) input.
fn pure_reply_identity(spec: &ExternalProfileSpec, bits: u32) -> Result<String> {
    let identity =
        xc_cache::ContentDigest::sha256_json(&(&spec.provider_sha256, &spec.input, bits))?;
    Ok(format!("{}\0", identity.0))
}

/// Whether the operator declared the authorized provider pure: the same
/// target input, precision and point always yield the same reply. Only then
/// may replies be reused within the process.
pub(crate) fn provider_declared_pure() -> Result<bool> {
    match std::env::var("XC_TARGET_PROVIDER_PURE") {
        Ok(value) if value == "1" => Ok(true),
        Ok(value) if value.is_empty() || value == "0" => Ok(false),
        Ok(_) => anyhow::bail!("XC_TARGET_PROVIDER_PURE must be 0 or 1"),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Debug)]
struct Provider {
    connection: Arc<Mutex<Connection>>,
    /// Digest of the exact provider, target input and precision of these
    /// replies when the provider is declared pure; `None` asks the provider
    /// every time. A digest, not the input itself: at high precision the input
    /// is megabytes and prefixes every memo key.
    pure_identity: Option<String>,
}
impl Provider {
    fn start(spec: &ExternalProfileSpec, bits: u32) -> Result<Self> {
        spec.validate()?;
        admit_input(&spec.input, super::input::target_maximum_bytes()?)?;
        anyhow::ensure!(
            bits <= spec.evaluation_precision_bits,
            "external target requires regeneration at the requested precision"
        );
        let pure_identity = provider_declared_pure()?
            .then(|| pure_reply_identity(spec, bits))
            .transpose()?;
        let path = std::env::var_os("XC_TARGET_PROVIDER_EXECUTABLE").context(
            "external target requires explicit XC_TARGET_PROVIDER_EXECUTABLE authorization",
        )?;
        anyhow::ensure!(!path.is_empty(), "empty target provider path");
        let path = std::path::Path::new(&path);
        let executable_file = verified_executable(path, &spec.provider_sha256)?;
        let timeout = std::env::var("XC_TARGET_PROVIDER_TIMEOUT_SECONDS")
            .ok()
            .map(|s| s.parse::<u64>())
            .transpose()?
            .unwrap_or(300);
        anyhow::ensure!(
            (1..=3600).contains(&timeout),
            "invalid target provider timeout"
        );
        let mut command = Command::new(path);
        command
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(diagnostic_stderr()?);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = ProviderChild(
            command
                .spawn()
                .context("cannot start authorized target provider")?,
        );
        // Detect ordinary replacement across spawn. This is not an atomic
        // execution guarantee against a hostile local writer on every platform.
        verified_executable(path, &spec.provider_sha256)?;
        let input = BufWriter::new(
            child
                .0
                .stdin
                .take()
                .context("target provider stdin unavailable")?,
        );
        let (requests, writes) = request_writer(input);
        let output = child
            .0
            .stdout
            .take()
            .context("target provider stdout unavailable")?;
        let (sender, replies) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            loop {
                let mut line = Vec::new();
                // Bound the response before allocation growth; no input/response is logged.
                let result = std::io::Read::by_ref(&mut reader)
                    .take(1_048_577)
                    .read_until(b'\n', &mut line);
                let value = match result {
                    Ok(n) if n > 0 && n <= 1_048_576 && line.last() == Some(&b'\n') => {
                        serde_json::from_slice(&line).map_err(|_| "invalid response".to_owned())
                    }
                    _ => Err("invalid response".to_owned()),
                };
                let failed = value.is_err();
                if sender.send(value).is_err() || failed {
                    break;
                }
            }
        });
        let provider = Self {
            connection: Arc::new(Mutex::new(Connection {
                child,
                requests,
                writes,
                replies,
                timeout: Duration::from_secs(timeout),
                failed: false,
                next_request_id: 1,
                precision_bits: bits,
                _executable_file: executable_file,
            })),
            pure_identity,
        };
        provider
            .connection
            .lock()
            .map_err(|_| anyhow::anyhow!("target provider lock failed"))?
            .exchange(&serde_json::json!({"operation":"initialize","input":spec.input}))?;
        Ok(provider)
    }
    fn value<T>(&self, u: &str, parse: impl FnOnce(&str) -> Result<T>) -> Result<T> {
        let key = self
            .pure_identity
            .as_ref()
            .map(|identity| format!("{identity}{u}"));
        if let Some(key) = &key {
            if let Some(value) = PURE_REPLIES
                .get()
                .and_then(|memo| memo.lock().ok()?.values.get(key).cloned())
            {
                return parse(&value);
            }
        }
        let reply = self
            .connection
            .lock()
            .map_err(|_| anyhow::anyhow!("target provider lock failed"))?
            .exchange(&serde_json::json!({"operation":"evaluate","u":u}))?;
        let value = reply
            .get("value")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .context("external target provider omitted its value")?;
        // A string-shaped reply can still be invalid for the scalar backend.
        // Only validated replies are remembered; a failing point is asked again.
        let parsed = parse(&value)?;
        if let Some(key) = key {
            let memo = PURE_REPLIES.get_or_init(|| Mutex::new(PureReplies::default()));
            if let Ok(mut memo) = memo.lock() {
                memo.remember(key, &value, PURE_REPLY_MEMO_ENTRIES, PURE_REPLY_MEMO_BYTES);
            }
        }
        Ok(parsed)
    }
}
fn parse_scalar_f64(text: &str) -> Result<f64> {
    xc_core::DecimalLiteral::new(text)?;
    let value = text.parse::<f64>().context("invalid provider scalar")?;
    anyhow::ensure!(
        value == 0.0 || value.is_normal(),
        "provider scalar is subnormal or outside the native range; use the HP evaluator"
    );
    anyhow::ensure!(
        value.is_finite(),
        "external target provider returned a nonfinite value"
    );
    anyhow::ensure!(
        value != 0.0 || xc_core::DecimalLiteral::new(text)?.canonical()?.as_str() == "0",
        "external target provider scalar underflows the native exponent range"
    );
    Ok(value)
}

#[derive(Clone, Debug)]
pub(super) struct CompiledF64 {
    provider: Provider,
    lambda: f64,
}
impl CompiledF64 {
    pub(super) fn new(spec: &ExternalProfileSpec) -> Result<Self> {
        let cutoff = spec.lambda_squared.parse::<f64>()?;
        anyhow::ensure!(
            cutoff.is_finite() && cutoff > 1.0,
            "external cutoff is outside the native range; use HP"
        );
        Ok(Self {
            provider: Provider::start(spec, 64)?,
            lambda: cutoff.sqrt(),
        })
    }
    pub(super) fn validate_lambda(&self, lambda: f64) -> Result<()> {
        anyhow::ensure!(
            lambda.is_finite() && (lambda - self.lambda).abs() <= 8.0 * f64::EPSILON * self.lambda,
            "runtime target cutoff does not match the configuration"
        );
        Ok(())
    }
    pub(super) fn raw(&self, u: f64) -> Result<f64> {
        anyhow::ensure!(u.is_finite() && u > 0.0, "target requires finite u > 0");
        self.provider.value(&u.to_string(), parse_scalar_f64)
    }
}
#[cfg(feature = "hp")]
pub(super) mod hp {
    use super::*;
    use rug::Float;
    pub(super) fn parse_scalar(text: &str, bits: u32) -> Result<Float> {
        xc_core::DecimalLiteral::new(text)?;
        let (value, rounding) = Float::with_val_round(
            bits,
            Float::parse(text).map_err(|_| anyhow::anyhow!("invalid provider scalar"))?,
            rug::float::Round::Nearest,
        );
        anyhow::ensure!(
            rounding == std::cmp::Ordering::Equal || value.get_exp() != Some(rug::float::exp_min()),
            "provider scalar loses relative precision at the MPFR exponent floor"
        );
        anyhow::ensure!(
            value.is_finite(),
            "external target provider returned a nonfinite value"
        );
        anyhow::ensure!(
            !value.is_zero() || xc_core::DecimalLiteral::new(text)?.canonical()?.as_str() == "0",
            "external target provider scalar underflows the working exponent range"
        );
        Ok(value)
    }
    #[derive(Clone, Debug)]
    pub(crate) struct Compiled {
        provider: Provider,
        lambda: Float,
        bits: u32,
    }
    impl Compiled {
        pub(crate) fn new(spec: &ExternalProfileSpec, bits: u32) -> Result<Self> {
            let cutoff = parse_scalar(&spec.lambda_squared, bits)?;
            anyhow::ensure!(
                cutoff > 1,
                "external cutoff is unresolved above one at working precision"
            );
            let lambda = cutoff.sqrt();
            anyhow::ensure!(
                lambda.is_finite() && lambda > 1,
                "external support is unresolved at working precision"
            );
            Ok(Self {
                provider: Provider::start(spec, bits)?,
                lambda,
                bits,
            })
        }
        pub(crate) fn validate_lambda(&self, lambda: &Float) -> Result<()> {
            let tolerance = Float::with_val(self.bits, &self.lambda)
                >> lambda.prec().min(self.bits).saturating_sub(4);
            anyhow::ensure!(
                lambda.is_finite()
                    && Float::with_val(self.bits, lambda - &self.lambda).abs() <= tolerance,
                "runtime target cutoff does not match the configuration"
            );
            Ok(())
        }
        pub(crate) fn raw(&self, u: &Float) -> Result<Float> {
            anyhow::ensure!(u.is_finite() && u > &0, "target requires finite u > 0");
            let provider_u = Float::with_val(self.bits.max(u.prec()), u);
            self.provider
                .value(&provider_u.to_string_radix(10, None), |text| {
                    parse_scalar(text, self.bits)
                })
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn reply(id: u64) -> serde_json::Value {
        serde_json::json!({"protocol_version":2,"request_nonce":"test-nonce","request_id":id,"operation":"evaluate","precision_bits":256,"value":"0.5"})
    }
    #[test]
    fn request_frames_keep_exact_json_and_enforce_the_complete_frame_budget() {
        let request = serde_json::json!({"operation":"initialize","input":["0.125","1e-2000"]});
        let mut expected = serde_json::to_vec(&request).unwrap();
        expected.push(b'\n');
        assert_eq!(
            request_frame(&request, expected.len() as u64).unwrap(),
            expected
        );
        assert!(request_frame(&request, expected.len() as u64 - 1).is_err());
        assert!(request_frame(&request, 0).is_err());
        let input = &request["input"];
        let length = serde_json::to_vec(input).unwrap().len() as u64;
        admit_input(input, length).unwrap();
        assert!(admit_input(input, length - 1).is_err());
    }
    #[test]
    fn pure_reply_keys_carry_a_digest_not_the_target_input() {
        let spec = |coefficient: &str| ExternalProfileSpec {
            lambda_squared: "100".into(),
            evaluation_precision_bits: 4096,
            provider_sha256: "ab".repeat(32),
            input: serde_json::json!({"even_legendre_coefficients": vec![coefficient; 4000]}),
        };
        let large = spec(&"1".repeat(1000));
        let identity = pure_reply_identity(&large, 3706).unwrap();
        // 64 hex digits and a separator, however large the input.
        assert_eq!(identity.len(), 65);
        assert_ne!(identity, pure_reply_identity(&large, 3707).unwrap());
        assert_ne!(identity, pure_reply_identity(&spec("2"), 3706).unwrap());
        assert_eq!(identity, pure_reply_identity(&large, 3706).unwrap());
    }
    #[test]
    fn pure_reply_memo_stays_within_its_entry_and_byte_budgets() {
        let mut memo = PureReplies::default();
        let consistent = |memo: &PureReplies| {
            assert_eq!(memo.order.len(), memo.values.len());
            let bytes: usize = memo
                .values
                .iter()
                .map(|(key, value)| 2 * key.len() + value.len())
                .sum();
            assert_eq!(memo.bytes, bytes);
        };
        // 2*4 + 2 = 10 bytes per entry; budget 25 bytes, 3 entries.
        for key in ["key1", "key2"] {
            memo.remember(key.into(), "vv", 3, 25);
        }
        consistent(&memo);
        assert_eq!(memo.order, ["key1", "key2"]);
        memo.remember("key3".into(), "vv", 3, 25);
        consistent(&memo);
        assert_eq!(
            memo.order,
            ["key2", "key3"],
            "byte budget evicts the oldest"
        );
        assert!(memo.bytes <= 25);
        // An entry larger than the whole budget is not kept and evicts nothing.
        memo.remember("huge".into(), &"v".repeat(30), 3, 25);
        consistent(&memo);
        assert_eq!(memo.order, ["key2", "key3"]);
        // A repeated key keeps its first reply.
        memo.remember("key3".into(), "ww", 3, 25);
        assert_eq!(memo.values["key3"], "vv");
        // The entry limit evicts the oldest as well.
        for key in ["a", "b", "c"] {
            memo.remember(key.into(), "", 3, 1000);
        }
        consistent(&memo);
        assert_eq!(memo.order, ["a", "b", "c"]);
    }
    #[test]
    fn duplicate_or_uncorrelated_reply_is_rejected() {
        let first = reply(1);
        validate_reply(&first, 1, 256, "test-nonce", "evaluate").unwrap();
        assert!(validate_reply(&first, 2, 256, "test-nonce", "evaluate").is_err());
        for id in [
            serde_json::Value::Null,
            serde_json::json!("2"),
            serde_json::json!(3),
        ] {
            let mut wrong = reply(2);
            wrong["request_id"] = id;
            assert!(validate_reply(&wrong, 2, 256, "test-nonce", "evaluate").is_err());
        }
    }
    #[test]
    fn precision_is_declared_not_inferred_from_decimal_length() {
        validate_reply(&reply(1), 1, 256, "test-nonce", "evaluate").unwrap();
        for precision in [
            serde_json::Value::Null,
            serde_json::json!(53),
            serde_json::json!(512),
        ] {
            let mut wrong = reply(1);
            wrong["precision_bits"] = precision;
            assert!(validate_reply(&wrong, 1, 256, "test-nonce", "evaluate").is_err());
        }
    }
    #[test]
    fn operation_version_error_and_missing_value_fail_closed() {
        for (key, value) in [
            ("operation", serde_json::json!("initialize")),
            ("protocol_version", serde_json::json!(0)),
            ("request_nonce", serde_json::json!("prefetched-prediction")),
            ("value", serde_json::Value::Null),
            ("error", serde_json::json!("private detail")),
        ] {
            let mut wrong = reply(1);
            wrong[key] = value;
            let error = validate_reply(&wrong, 1, 256, "test-nonce", "evaluate").unwrap_err();
            assert!(!error.to_string().contains("private detail"));
        }
    }
    #[test]
    fn protocol_version_errors_never_render_response_values() {
        for version in [
            serde_json::json!("private-version-marker"),
            serde_json::json!({"private-version-marker": "private-response-marker"}),
            serde_json::json!(["private-response-marker"]),
            serde_json::Value::Null,
            serde_json::json!(3),
        ] {
            let mut wrong = reply(1);
            wrong["protocol_version"] = version;
            let error = validate_reply(&wrong, 1, 256, "test-nonce", "evaluate")
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                "target provider protocol version mismatch: expected protocol 2 (with request_nonce)"
            );
        }
        let mut missing = reply(1);
        missing.as_object_mut().unwrap().remove("protocol_version");
        assert!(validate_reply(&missing, 1, 256, "test-nonce", "evaluate").is_err());
    }
    #[test]
    fn external_policy_rejects_invalid_cutoff_precision_and_digest() {
        let mut s = ExternalProfileSpec {
            lambda_squared: "4".into(),
            evaluation_precision_bits: 256,
            provider_sha256: "a".repeat(64),
            input: serde_json::json!({"fixture":true}),
        };
        s.validate().unwrap();
        s.lambda_squared = "NaN".into();
        assert!(s.validate().is_err());
        s.lambda_squared = "4".into();
        s.evaluation_precision_bits = 0;
        assert!(s.validate().is_err());
        s.evaluation_precision_bits = 256;
        s.provider_sha256 = "a".into();
        assert!(s.validate().is_err());
    }
}

#[cfg(test)]
mod exhaustive_resumed_scalar_contract {
    use super::*;
    #[test]
    fn exhaustive_resumed_native_provider_rejects_nonzero_underflow() {
        assert!(parse_scalar_f64("1e-400").is_err());
        assert!(parse_scalar_f64("-1e-400").is_err());
        assert_eq!(parse_scalar_f64("-0e-400").unwrap(), 0.);
        assert!(parse_scalar_f64("4.9406564584124654e-324").is_err());
        assert_eq!(parse_scalar_f64("0.5").unwrap(), 0.5);
    }
    #[cfg(feature = "hp")]
    #[test]
    fn exhaustive_resumed_hp_provider_rejects_nonzero_underflow() {
        assert!(hp::parse_scalar("1e-400000000", 128).is_err());
        assert!(hp::parse_scalar("-1e-400000000", 128).is_err());
        assert_eq!(hp::parse_scalar("0e-400000000", 128).unwrap(), 0);
        assert_eq!(hp::parse_scalar("0.5", 128).unwrap(), 0.5);
        assert!(hp::parse_scalar(" 0.5", 128).is_err());
        let mut floor = rug::Float::with_val(128, 0);
        floor.next_up();
        assert!(hp::parse_scalar(&floor.to_string_radix(10, None), 128).is_err());
    }
}

#[cfg(test)]
#[path = "target_external_deadline_tests.rs"]
mod deadline_tests;
