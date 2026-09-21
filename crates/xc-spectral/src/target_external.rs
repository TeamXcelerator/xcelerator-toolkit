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
use std::time::Duration;

/// Data for a caller-authorized target provider. The Toolkit does not interpret
/// `input` or provide the implementation of the caller's research function.
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
        let c = self
            .lambda_squared
            .parse::<f64>()
            .context("invalid external cutoff")?;
        anyhow::ensure!(
            c.is_finite() && c > 1.0 && (64..=1_000_000).contains(&self.evaluation_precision_bits),
            "invalid external target policy"
        );
        anyhow::ensure!(
            self.provider_sha256.len() == 64
                && self.provider_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid provider digest"
        );
        Ok(())
    }
}

#[derive(Debug)]
struct Connection {
    child: Child,
    input: BufWriter<ChildStdin>,
    replies: mpsc::Receiver<std::result::Result<serde_json::Value, String>>,
    timeout: Duration,
    failed: bool,
    next_request_id: u64,
    precision_bits: u32,
    _executable_file: File,
}
impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Connection {
    fn exchange(&mut self, message: &serde_json::Value) -> Result<serde_json::Value> {
        anyhow::ensure!(
            !self.failed,
            "external target provider is unavailable after an earlier failure"
        );
        let result = (|| {
            let id = self.next_request_id;
            self.next_request_id = id
                .checked_add(1)
                .context("target request counter exhausted")?;
            let mut request = message.clone();
            let fields = request.as_object_mut().context("invalid target request")?;
            fields.insert("request_id".into(), id.into());
            fields.insert("protocol_version".into(), 1.into());
            fields.insert("precision_bits".into(), self.precision_bits.into());
            serde_json::to_writer(&mut self.input, &request)
                .context("cannot write external target request")?;
            writeln!(self.input)?;
            self.input.flush()?;
            let reply = self
                .replies
                .recv_timeout(self.timeout)
                .map_err(|_| anyhow::anyhow!("external target provider timed out or disconnected"))?
                .map_err(|_| {
                    anyhow::anyhow!("external target provider returned an invalid response")
                })?;
            validate_reply(
                &reply,
                id,
                self.precision_bits,
                message["operation"]
                    .as_str()
                    .context("invalid target operation")?,
            )?;
            Ok(reply)
        })();
        if result.is_err() {
            self.failed = true;
            let _ = self.child.kill();
        }
        result
    }
}

// Precision is a provider declaration, not proof of accuracy. In particular,
// exact decimal values such as 0.5 must not fail a digit-count heuristic.
fn validate_reply(reply: &serde_json::Value, id: u64, bits: u32, operation: &str) -> Result<()> {
    anyhow::ensure!(
        reply["protocol_version"].as_u64() == Some(1),
        "target provider protocol version mismatch"
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

#[derive(Clone, Debug)]
struct Provider(Arc<Mutex<Connection>>);
impl Provider {
    fn start(spec: &ExternalProfileSpec, bits: u32) -> Result<Self> {
        spec.validate()?;
        anyhow::ensure!(
            bits <= spec.evaluation_precision_bits,
            "external target requires regeneration at the requested precision"
        );
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
        let mut child = command
            .spawn()
            .context("cannot start authorized target provider")?;
        // Detect ordinary replacement across spawn. This is not an atomic
        // execution guarantee against a hostile local writer on every platform.
        if let Err(error) = verified_executable(path, &spec.provider_sha256) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        let input = BufWriter::new(
            child
                .stdin
                .take()
                .context("target provider stdin unavailable")?,
        );
        let output = child
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
        let provider = Self(Arc::new(Mutex::new(Connection {
            child,
            input,
            replies,
            timeout: Duration::from_secs(timeout),
            failed: false,
            next_request_id: 1,
            precision_bits: bits,
            _executable_file: executable_file,
        })));
        provider
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("target provider lock failed"))?
            .exchange(&serde_json::json!({"operation":"initialize","input":spec.input}))?;
        Ok(provider)
    }
    fn value(&self, u: &str) -> Result<String> {
        let reply = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("target provider lock failed"))?
            .exchange(&serde_json::json!({"operation":"evaluate","u":u}))?;
        reply
            .get("value")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .context("external target provider omitted its value")
    }
}
#[derive(Clone, Debug)]
pub(super) struct CompiledF64 {
    provider: Provider,
    lambda: f64,
}
impl CompiledF64 {
    pub(super) fn new(spec: &ExternalProfileSpec) -> Result<Self> {
        Ok(Self {
            provider: Provider::start(spec, 64)?,
            lambda: spec.lambda_squared.parse::<f64>()?.sqrt(),
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
        let value = self
            .provider
            .value(&u.to_string())?
            .parse::<f64>()
            .context("invalid provider scalar")?;
        anyhow::ensure!(
            value.is_finite(),
            "external target provider returned a nonfinite value"
        );
        Ok(value)
    }
}
#[cfg(feature = "hp")]
pub(super) mod hp {
    use super::*;
    use rug::Float;
    #[derive(Clone, Debug)]
    pub(crate) struct Compiled {
        provider: Provider,
        lambda: Float,
        bits: u32,
    }
    impl Compiled {
        pub(crate) fn new(spec: &ExternalProfileSpec, bits: u32) -> Result<Self> {
            Ok(Self {
                provider: Provider::start(spec, bits)?,
                lambda: Float::with_val(bits, Float::parse(&spec.lambda_squared)?).sqrt(),
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
            let text = self.provider.value(&u.to_string_radix(10, None))?;
            let value = Float::with_val(
                self.bits,
                Float::parse(&text).map_err(|_| anyhow::anyhow!("invalid provider scalar"))?,
            );
            anyhow::ensure!(
                value.is_finite(),
                "external target provider returned a nonfinite value"
            );
            Ok(value)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn reply(id: u64) -> serde_json::Value {
        serde_json::json!({"protocol_version":1,"request_id":id,"operation":"evaluate","precision_bits":256,"value":"0.5"})
    }
    #[test]
    fn duplicate_or_uncorrelated_reply_is_rejected() {
        let first = reply(1);
        validate_reply(&first, 1, 256, "evaluate").unwrap();
        assert!(validate_reply(&first, 2, 256, "evaluate").is_err());
        for id in [
            serde_json::Value::Null,
            serde_json::json!("2"),
            serde_json::json!(3),
        ] {
            let mut wrong = reply(2);
            wrong["request_id"] = id;
            assert!(validate_reply(&wrong, 2, 256, "evaluate").is_err());
        }
    }
    #[test]
    fn precision_is_declared_not_inferred_from_decimal_length() {
        validate_reply(&reply(1), 1, 256, "evaluate").unwrap();
        for precision in [
            serde_json::Value::Null,
            serde_json::json!(53),
            serde_json::json!(512),
        ] {
            let mut wrong = reply(1);
            wrong["precision_bits"] = precision;
            assert!(validate_reply(&wrong, 1, 256, "evaluate").is_err());
        }
    }
    #[test]
    fn operation_version_error_and_missing_value_fail_closed() {
        for (key, value) in [
            ("operation", serde_json::json!("initialize")),
            ("protocol_version", serde_json::json!(0)),
            ("value", serde_json::Value::Null),
            ("error", serde_json::json!("private detail")),
        ] {
            let mut wrong = reply(1);
            wrong[key] = value;
            let error = validate_reply(&wrong, 1, 256, "evaluate").unwrap_err();
            assert!(!error.to_string().contains("private detail"));
        }
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
