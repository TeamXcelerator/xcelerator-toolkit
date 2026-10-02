//! Live GitHub principal and repository-permission preflight.
//!
//! Environment PATs are preferred for ephemeral compute hosts and are passed
//! to the configured Git credential provider over stdin so the API probe and
//! later Git transport use the same secret. Credentials are never placed in
//! evidence, diagnostics, command-line arguments, or child-process environment
//! variables.

use crate::{canonical_digest, CacheError, ContentDigest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GITHUB_API_ORIGIN: &str = "https://api.github.com";
const GITHUB_CREDENTIAL_HOST: &str = "github.com";
const GITHUB_TOKEN_ENV: &str = "GITHUB_TOKEN";
const GH_TOKEN_ENV: &str = "GH_TOKEN";
const PROVIDER_ID: &str = "git-credential+github-api-curl-v1";
/// Retries after the first API attempt; a retry starts only while
/// [`GITHUB_API_RETRY_WINDOW`] has not elapsed since the first attempt began.
const GITHUB_API_RETRIES: u32 = 4;
const GITHUB_API_RETRY_WINDOW: Duration = Duration::from_secs(120);
/// Marks curl's status trailer after the response body (`--write-out`).
const GITHUB_API_STATUS_MARKER: &str = "\nxc-github-api-status:";
/// GitHub's minimum wait after a rate-limit response that states no time.
const GITHUB_RATE_LIMIT_MINIMUM_WAIT: Duration = Duration::from_secs(60);
pub const AUTHORITY_PROBE_MAX_AGE_SECONDS: u64 = 300;
pub const STALE_AUTHORITY_PROBE_MESSAGE: &str =
    "GitHub permission probe is stale and must be repeated before mutation";
static CREDENTIAL_PROVIDER_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositoryPermission {
    None,
    Read,
    Triage,
    Write,
    Maintain,
    Admin,
}

impl RepositoryPermission {
    pub fn permits_write(self) -> bool {
        self >= Self::Write
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositoryVisibility {
    Public,
    Private,
    Internal,
}

/// Redacted, serializable evidence retained in plans and receipts. It contains
/// no credential material. Remote mutation must additionally hold the opaque
/// `AuthenticatedGitHubSession` produced by a live probe.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryPermissionEvidence {
    pub provider_id: String,
    pub principal: String,
    pub repository: String,
    pub visibility: RepositoryVisibility,
    pub permission: RepositoryPermission,
    pub verified_at_unix_seconds: u64,
    pub evidence_digest: ContentDigest,
}

#[derive(Serialize)]
struct PermissionEvidenceMaterial<'a> {
    provider_id: &'a str,
    principal: &'a str,
    repository: &'a str,
    visibility: RepositoryVisibility,
    permission: RepositoryPermission,
    verified_at_unix_seconds: u64,
}

impl RepositoryPermissionEvidence {
    fn new(
        principal: String,
        repository: String,
        visibility: RepositoryVisibility,
        permission: RepositoryPermission,
    ) -> Result<Self, CacheError> {
        let verified_at_unix_seconds = unix_time_now()?;
        let material = PermissionEvidenceMaterial {
            provider_id: PROVIDER_ID,
            principal: &principal,
            repository: &repository,
            visibility,
            permission,
            verified_at_unix_seconds,
        };
        let evidence_digest = canonical_digest(&material)?;
        Ok(Self {
            provider_id: PROVIDER_ID.to_owned(),
            principal,
            repository,
            visibility,
            permission,
            verified_at_unix_seconds,
            evidence_digest,
        })
    }

    pub fn validate(&self) -> Result<(), CacheError> {
        if self.provider_id != PROVIDER_ID
            || self.principal.trim().is_empty()
            || !valid_repository_name(&self.repository)
            || !self.evidence_digest.validate()
        {
            return Err(CacheError::Authentication(
                "repository permission evidence is incomplete".to_owned(),
            ));
        }
        let material = PermissionEvidenceMaterial {
            provider_id: &self.provider_id,
            principal: &self.principal,
            repository: &self.repository,
            visibility: self.visibility,
            permission: self.permission,
            verified_at_unix_seconds: self.verified_at_unix_seconds,
        };
        let digest = canonical_digest(&material)?;
        if digest != self.evidence_digest {
            return Err(CacheError::Authentication(
                "repository permission evidence digest does not match".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Opaque in-process proof that authentication was performed by the live
/// provider. This type cannot be deserialized or constructed by callers.
#[derive(Clone, Debug)]
pub struct AuthenticatedGitHubSession {
    evidence: RepositoryPermissionEvidence,
}

impl AuthenticatedGitHubSession {
    pub fn evidence(&self) -> &RepositoryPermissionEvidence {
        &self.evidence
    }

    pub fn require_write_for(&self, principal: &str, repository: &str) -> Result<(), CacheError> {
        self.evidence.validate()?;
        let now = unix_time_now()?;
        if self.evidence.verified_at_unix_seconds > now.saturating_add(30)
            || now.saturating_sub(self.evidence.verified_at_unix_seconds)
                > AUTHORITY_PROBE_MAX_AGE_SECONDS
        {
            return Err(CacheError::Authentication(
                STALE_AUTHORITY_PROBE_MESSAGE.to_owned(),
            ));
        }
        if self.evidence.principal != principal {
            return Err(CacheError::Authentication(format!(
                "authenticated principal {:?} does not match authority principal {principal:?}",
                self.evidence.principal
            )));
        }
        if !self.evidence.repository.eq_ignore_ascii_case(repository) {
            return Err(CacheError::PermissionDenied(format!(
                "permission was verified for {:?}, not {repository:?}",
                self.evidence.repository
            )));
        }
        if !self.evidence.permission.permits_write() {
            return Err(CacheError::PermissionDenied(format!(
                "principal {principal:?} has {:?} permission for {repository:?}",
                self.evidence.permission
            )));
        }
        Ok(())
    }

    /// Whether this permission evidence should be refreshed before beginning
    /// another potentially long publication unit.
    pub fn requires_refresh(&self, safety_margin_seconds: u64) -> Result<bool, CacheError> {
        self.evidence.validate()?;
        let now = unix_time_now()?;
        if self.evidence.verified_at_unix_seconds > now.saturating_add(30) {
            return Ok(true);
        }
        let age = now.saturating_sub(self.evidence.verified_at_unix_seconds);
        Ok(age.saturating_add(safety_margin_seconds) >= AUTHORITY_PROBE_MAX_AGE_SECONDS)
    }

    #[cfg(test)]
    pub(crate) fn verified_for_test(
        principal: &str,
        repository: &str,
        permission: RepositoryPermission,
    ) -> Self {
        Self {
            evidence: RepositoryPermissionEvidence::new(
                principal.to_owned(),
                repository.to_owned(),
                RepositoryVisibility::Private,
                permission,
            )
            .unwrap(),
        }
    }

    #[cfg(test)]
    pub(crate) fn verified_for_test_with_age(
        principal: &str,
        repository: &str,
        permission: RepositoryPermission,
        age_seconds: u64,
    ) -> Self {
        let mut evidence = RepositoryPermissionEvidence::new(
            principal.to_owned(),
            repository.to_owned(),
            RepositoryVisibility::Private,
            permission,
        )
        .unwrap();
        evidence.verified_at_unix_seconds = evidence
            .verified_at_unix_seconds
            .saturating_sub(age_seconds);
        evidence.evidence_digest = canonical_digest(&PermissionEvidenceMaterial {
            provider_id: &evidence.provider_id,
            principal: &evidence.principal,
            repository: &evidence.repository,
            visibility: evidence.visibility,
            permission: evidence.permission,
            verified_at_unix_seconds: evidence.verified_at_unix_seconds,
        })
        .unwrap();
        Self { evidence }
    }
}

#[derive(Clone, Debug)]
pub struct GitHubCredentialApiProbe {
    git_executable: OsString,
    curl_executable: OsString,
    /// First backoff delay between API attempts; doubles per retry.
    retry_delay: Duration,
}

impl Default for GitHubCredentialApiProbe {
    fn default() -> Self {
        Self {
            git_executable: OsString::from("git"),
            curl_executable: OsString::from(if cfg!(windows) { "curl.exe" } else { "curl" }),
            retry_delay: Duration::from_secs(1),
        }
    }
}

impl GitHubCredentialApiProbe {
    pub fn with_git_executable(mut self, executable: impl Into<OsString>) -> Self {
        self.git_executable = executable.into();
        self
    }

    pub fn with_curl_executable(mut self, executable: impl Into<OsString>) -> Self {
        self.curl_executable = executable.into();
        self
    }

    /// Make the resolved GitHub credential available to non-interactive Git
    /// transport before any private repository read.
    ///
    /// An environment PAT is authoritative and is approved through the
    /// configured credential provider over stdin. This must run before the
    /// bootstrap cache performs its first `git` read; publication permission
    /// probes happen too late for a cold private-cache fetch after a machine
    /// restart has cleared an in-memory credential helper.
    pub fn prepare_git_transport(&self) -> Result<(), CacheError> {
        self.resolved_credential().map(|_| ())
    }

    /// Resolve the current GitHub principal and its effective permission for
    /// exactly one owner/repository target.
    pub fn probe_repository(
        &self,
        repository: &str,
    ) -> Result<AuthenticatedGitHubSession, CacheError> {
        if !valid_repository_name(repository) {
            return Err(CacheError::InvalidManifest(
                "GitHub repository must be an owner/name pair".to_owned(),
            ));
        }
        let credential = self.resolved_credential()?;
        let user = self.github_get("/user", &credential.secret)?;
        let principal = required_string(&user, "login", "authenticated GitHub user")?;
        let repository_response =
            self.github_get(&format!("/repos/{repository}"), &credential.secret)?;
        let resolved_repository = required_string(
            &repository_response,
            "full_name",
            "GitHub repository permission response",
        )?;
        if !resolved_repository.eq_ignore_ascii_case(repository) {
            return Err(CacheError::PermissionDenied(format!(
                "GitHub resolved repository {resolved_repository:?}, expected {repository:?}"
            )));
        }
        let visibility = parse_visibility(&repository_response)?;
        let permission = parse_permission(&repository_response)?;
        let evidence = RepositoryPermissionEvidence::new(
            principal,
            resolved_repository,
            visibility,
            permission,
        )?;
        let session = AuthenticatedGitHubSession { evidence };
        session.require_write_for(session.evidence().principal.as_str(), repository)?;
        Ok(session)
    }

    fn resolved_credential(&self) -> Result<GitCredential, CacheError> {
        let _guard = CREDENTIAL_PROVIDER_LOCK.lock().map_err(|_| {
            CacheError::Authentication("Git credential provider lock is poisoned".to_owned())
        })?;
        if let Some(mut credential) = environment_credential()? {
            // Reuse the provider's existing username key when present so an
            // explicit environment PAT replaces a stale host credential
            // instead of creating an ambiguous second entry.
            if let Ok(existing) = self.git_credential() {
                credential.username = existing.username;
            }
            self.approve_git_credential(&credential)?;
            let retained = self.git_credential().map_err(|_| {
                CacheError::Authentication(
                    "the configured Git credential provider did not retain the environment PAT for Git transport"
                        .to_owned(),
                )
            })?;
            if retained.secret != credential.secret {
                return Err(CacheError::Authentication(
                    "the configured Git credential provider returned a different credential after accepting the environment PAT"
                        .to_owned(),
                ));
            }
            return Ok(credential);
        }
        self.git_credential()
    }

    fn approve_git_credential(&self, credential: &GitCredential) -> Result<(), CacheError> {
        let input = format!(
            "protocol=https\nhost={GITHUB_CREDENTIAL_HOST}\nusername={}\npassword={}\n\n",
            credential.username, credential.secret
        );
        let output = run_with_input(
            &self.git_executable,
            [OsString::from("credential"), OsString::from("approve")],
            input.as_bytes(),
            &[
                (OsStr::new("GIT_TERMINAL_PROMPT"), OsStr::new("0")),
                (OsStr::new("GCM_INTERACTIVE"), OsStr::new("Never")),
            ],
        )?;
        if !output.status.success() {
            return Err(CacheError::Authentication(
                "the configured Git credential provider rejected the environment PAT".to_owned(),
            ));
        }
        Ok(())
    }

    fn git_credential(&self) -> Result<GitCredential, CacheError> {
        let input = format!("protocol=https\nhost={GITHUB_CREDENTIAL_HOST}\n\n");
        let output = run_with_input(
            &self.git_executable,
            [OsString::from("credential"), OsString::from("fill")],
            input.as_bytes(),
            &[
                (OsStr::new("GIT_TERMINAL_PROMPT"), OsStr::new("0")),
                (OsStr::new("GCM_INTERACTIVE"), OsStr::new("Never")),
            ],
        )?;
        if !output.status.success() {
            return Err(CacheError::Authentication(
                "the configured Git credential provider did not return GitHub credentials"
                    .to_owned(),
            ));
        }
        parse_git_credential(&output.stdout)
    }

    /// A read-only GitHub API GET. Each attempt is a separate curl process with
    /// its own captured output, so only the final response is decoded.
    /// Transport failures, HTTP 408, 429, 500, 502, 503 and 504, and 403 rate
    /// limits are retried with exponential backoff (or the server's stated
    /// wait); other HTTP errors fail at once.
    fn github_get(&self, path: &str, secret: &str) -> Result<Value, CacheError> {
        if !path.starts_with('/') || path.contains(['\r', '\n']) {
            return Err(CacheError::InvalidManifest(
                "GitHub API path is invalid".to_owned(),
            ));
        }
        if secret.is_empty()
            || secret
                .bytes()
                .any(|byte| !byte.is_ascii_graphic() || matches!(byte, b'"' | b'\\'))
        {
            return Err(CacheError::Authentication(
                "credential provider returned an unsupported secret format".to_owned(),
            ));
        }
        let started = Instant::now();
        let mut delay = self.retry_delay;
        let mut attempts = 0;
        loop {
            attempts += 1;
            let failure = match self.github_get_once(path, secret)? {
                ApiAttempt::Response(value) => return Ok(value),
                ApiAttempt::Retry { reason, wait } => {
                    let wait = wait.unwrap_or(delay);
                    if attempts <= GITHUB_API_RETRIES
                        && started.elapsed() + wait <= GITHUB_API_RETRY_WINDOW
                    {
                        std::thread::sleep(wait);
                        delay = delay.saturating_mul(2);
                        continue;
                    }
                    reason
                }
                ApiAttempt::Fail(reason) => reason,
            };
            let tries = if attempts == 1 {
                "1 attempt".to_owned()
            } else {
                format!("{attempts} attempts")
            };
            return Err(CacheError::Authentication(format!(
                "GitHub API authentication or permission probe failed for {path} ({failure}; {tries})"
            )));
        }
    }

    fn github_get_once(&self, path: &str, secret: &str) -> Result<ApiAttempt, CacheError> {
        let config = format!(
            concat!(
                "url = \"{GITHUB_API_ORIGIN}{path}\"\n",
                "request = \"GET\"\n",
                "header = \"Accept: application/vnd.github+json\"\n",
                "header = \"Authorization: Bearer {secret}\"\n",
                "header = \"X-GitHub-Api-Version: 2022-11-28\"\n",
                "silent\nshow-error\n"
            ),
            GITHUB_API_ORIGIN = GITHUB_API_ORIGIN,
            path = path,
            secret = secret,
        );
        let output = run_with_input(
            &self.curl_executable,
            [
                OsString::from("--proto"),
                OsString::from("=https"),
                OsString::from("--tlsv1.2"),
                OsString::from("--connect-timeout"),
                OsString::from("15"),
                OsString::from("--max-time"),
                OsString::from("30"),
                OsString::from("--max-filesize"),
                OsString::from("1048576"),
                OsString::from("--write-out"),
                OsString::from(format!(
                    "{GITHUB_API_STATUS_MARKER}%{{http_code}} %header{{retry-after}}|%header{{x-ratelimit-remaining}}|%header{{x-ratelimit-reset}}\n"
                )),
                OsString::from("--config"),
                OsString::from("-"),
            ],
            config.as_bytes(),
            &[],
        )?;
        Ok(classify_api_attempt(
            output.status.code(),
            &output.stdout,
            path,
            SystemTime::now(),
        ))
    }
}

/// One GitHub API attempt's outcome.
#[derive(Debug)]
enum ApiAttempt {
    Response(Value),
    /// A transient failure; `wait` is the server's stated delay, if any.
    Retry {
        reason: String,
        wait: Option<Duration>,
    },
    Fail(String),
}

fn classify_api_attempt(
    exit: Option<i32>,
    stdout: &[u8],
    path: &str,
    now: SystemTime,
) -> ApiAttempt {
    match exit {
        Some(0) => {}
        // Resolve, connect, timeout, TLS, empty reply, send and receive
        // failures, partial transfers and HTTP/2 stream errors.
        Some(code @ (5 | 6 | 7 | 16 | 18 | 28 | 35 | 52 | 55 | 56 | 92)) => {
            return ApiAttempt::Retry {
                reason: format!("curl exit {code}"),
                wait: None,
            };
        }
        Some(code) => return ApiAttempt::Fail(format!("curl exit {code}")),
        None => return ApiAttempt::Fail("curl terminated without exit code".to_owned()),
    }
    let marker = GITHUB_API_STATUS_MARKER.as_bytes();
    let Some(split) = stdout
        .windows(marker.len())
        .rposition(|window| window == marker)
    else {
        return ApiAttempt::Fail("curl reported no HTTP status".to_owned());
    };
    let (body, trailer) = (&stdout[..split], &stdout[split + marker.len()..]);
    let trailer = String::from_utf8_lossy(trailer);
    let (status, headers) = trailer
        .trim_end()
        .split_once(' ')
        .unwrap_or((trailer.trim_end(), ""));
    let mut headers = headers.split('|').map(str::trim);
    let (retry_after, remaining, reset) = (
        headers.next().and_then(|v| v.parse::<u64>().ok()),
        headers.next().unwrap_or(""),
        headers.next().and_then(|v| v.parse::<u64>().ok()),
    );
    let Ok(status) = status.parse::<u16>() else {
        return ApiAttempt::Fail("curl reported no HTTP status".to_owned());
    };
    match status {
        200..=299 => match serde_json::from_slice(body) {
            Ok(value) => ApiAttempt::Response(value),
            Err(_) => ApiAttempt::Fail(format!("invalid JSON response for {path}")),
        },
        408 | 500 | 502 | 503 | 504 => ApiAttempt::Retry {
            reason: format!("HTTP {status}"),
            wait: retry_after.map(Duration::from_secs),
        },
        // GitHub signals primary and secondary rate limits with 429, or with
        // 403 and these headers or a rate-limit message; any other 403 is a
        // denial. Its rules: honor Retry-After; with no requests remaining,
        // wait for the reset; otherwise wait at least a minute.
        429 | 403
            if status == 429
                || retry_after.is_some()
                || remaining == "0"
                || String::from_utf8_lossy(body)
                    .to_ascii_lowercase()
                    .contains("rate limit") =>
        {
            let reset = reset.filter(|_| remaining == "0");
            let (wait, until) = match (retry_after, reset) {
                (Some(seconds), _) => (Duration::from_secs(seconds), String::new()),
                (None, Some(reset)) => {
                    let now = now
                        .duration_since(UNIX_EPOCH)
                        .map_or(0, |elapsed| elapsed.as_secs());
                    (
                        Duration::from_secs(reset.saturating_sub(now)),
                        format!(" until {reset}"),
                    )
                }
                (None, None) => (GITHUB_RATE_LIMIT_MINIMUM_WAIT, String::new()),
            };
            ApiAttempt::Retry {
                reason: format!("HTTP {status} rate limited{until}"),
                wait: Some(wait),
            }
        }
        _ => ApiAttempt::Fail(format!("HTTP {status}")),
    }
}

struct GitCredential {
    #[allow(dead_code)]
    username: String,
    secret: String,
}

fn environment_credential() -> Result<Option<GitCredential>, CacheError> {
    environment_credential_from_values(
        std::env::var_os(GITHUB_TOKEN_ENV),
        std::env::var_os(GH_TOKEN_ENV),
    )
}

fn environment_credential_from_values(
    github_token: Option<OsString>,
    gh_token: Option<OsString>,
) -> Result<Option<GitCredential>, CacheError> {
    for (name, value) in [(GITHUB_TOKEN_ENV, github_token), (GH_TOKEN_ENV, gh_token)] {
        let Some(value) = value else { continue };
        let secret = value.into_string().map_err(|_| {
            CacheError::Authentication(format!("{name} contains non-Unicode credential data"))
        })?;
        if secret.is_empty() {
            continue;
        }
        if secret
            .bytes()
            .any(|byte| !byte.is_ascii_graphic() || matches!(byte, b'"' | b'\\'))
        {
            return Err(CacheError::Authentication(format!(
                "{name} contains an unsupported credential format"
            )));
        }
        return Ok(Some(GitCredential {
            username: "x-access-token".to_owned(),
            secret,
        }));
    }
    Ok(None)
}

fn parse_git_credential(bytes: &[u8]) -> Result<GitCredential, CacheError> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        CacheError::Authentication("credential provider returned non-UTF-8 data".to_owned())
    })?;
    let mut username = None;
    let mut secret = None;
    for line in text.lines() {
        if let Some((name, value)) = line.split_once('=') {
            match name {
                "username" => username = Some(value.to_owned()),
                "password" => secret = Some(value.to_owned()),
                _ => {}
            }
        }
    }
    let username = username.filter(|value| !value.is_empty()).ok_or_else(|| {
        CacheError::Authentication("credential provider returned no username".to_owned())
    })?;
    let secret = secret.filter(|value| !value.is_empty()).ok_or_else(|| {
        CacheError::Authentication("credential provider returned no secret".to_owned())
    })?;
    Ok(GitCredential { username, secret })
}

fn unix_time_now() -> Result<u64, CacheError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| CacheError::Authentication("system clock precedes the Unix epoch".to_owned()))
}

fn valid_repository_name(repository: &str) -> bool {
    let mut parts = repository.split('/');
    let valid_part = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(owner), Some(name), None) if valid_part(owner) && valid_part(name)
    )
}

fn required_string(value: &Value, field: &str, context: &str) -> Result<String, CacheError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            CacheError::Authentication(format!("{context} omitted required field {field:?}"))
        })
}

fn parse_visibility(value: &Value) -> Result<RepositoryVisibility, CacheError> {
    match value.get("visibility").and_then(Value::as_str) {
        Some("public") => Ok(RepositoryVisibility::Public),
        Some("private") => Ok(RepositoryVisibility::Private),
        Some("internal") => Ok(RepositoryVisibility::Internal),
        _ => Err(CacheError::Authentication(
            "GitHub repository response omitted a recognized visibility".to_owned(),
        )),
    }
}

fn parse_permission(value: &Value) -> Result<RepositoryPermission, CacheError> {
    let permissions = value
        .get("permissions")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            CacheError::PermissionDenied(
                "GitHub repository response omitted effective permissions".to_owned(),
            )
        })?;
    let enabled = |name: &str| {
        permissions
            .get(name)
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    Ok(if enabled("admin") {
        RepositoryPermission::Admin
    } else if enabled("maintain") {
        RepositoryPermission::Maintain
    } else if enabled("push") {
        RepositoryPermission::Write
    } else if enabled("triage") {
        RepositoryPermission::Triage
    } else if enabled("pull") {
        RepositoryPermission::Read
    } else {
        RepositoryPermission::None
    })
}

fn run_with_input<const N: usize>(
    executable: &OsStr,
    arguments: [OsString; N],
    input: &[u8],
    environment: &[(&OsStr, &OsStr)],
) -> Result<std::process::Output, CacheError> {
    let mut command = Command::new(executable);
    command
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .envs(environment.iter().copied())
        .env_remove(GITHUB_TOKEN_ENV)
        .env_remove(GH_TOKEN_ENV);
    let mut child = command.spawn().map_err(|error| {
        CacheError::Authentication(format!(
            "could not start external credential provider: {error}"
        ))
    })?;
    child
        .stdin
        .take()
        .ok_or_else(|| {
            CacheError::Authentication("credential provider stdin unavailable".to_owned())
        })?
        .write_all(input)?;
    child.wait_with_output().map_err(CacheError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn credential_parser_never_requires_logging_the_secret() {
        let credential = parse_git_credential(
            b"protocol=https\nhost=github.com\nusername=owner\npassword=secret=value\n",
        )
        .unwrap();
        assert_eq!(credential.username, "owner");
        assert_eq!(credential.secret, "secret=value");
    }

    #[test]
    fn environment_pat_precedence_is_github_then_gh() {
        let credential = environment_credential_from_values(
            Some(OsString::from("github-secret")),
            Some(OsString::from("gh-secret")),
        )
        .unwrap()
        .unwrap();
        assert_eq!(credential.username, "x-access-token");
        assert_eq!(credential.secret, "github-secret");

        let fallback = environment_credential_from_values(
            Some(OsString::from("")),
            Some(OsString::from("gh-secret")),
        )
        .unwrap()
        .unwrap();
        assert_eq!(fallback.secret, "gh-secret");
    }

    #[test]
    fn environment_pat_rejects_control_characters_before_credential_approval() {
        let error = match environment_credential_from_values(
            Some(OsString::from("secret\nsecond-line")),
            None,
        ) {
            Err(error) => error,
            Ok(_) => panic!("control characters must be rejected"),
        };
        assert!(matches!(error, CacheError::Authentication(_)));
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn permission_parser_uses_effective_write_capability() {
        let response = json!({
            "permissions": {
                "admin": false,
                "maintain": false,
                "push": true,
                "triage": true,
                "pull": true
            }
        });
        assert_eq!(
            parse_permission(&response).unwrap(),
            RepositoryPermission::Write
        );
    }

    #[test]
    fn opaque_session_rejects_read_only_and_wrong_principal() {
        let read_only = AuthenticatedGitHubSession::verified_for_test(
            "owner",
            "example-org/restricted-cache",
            RepositoryPermission::Read,
        );
        assert!(matches!(
            read_only.require_write_for("owner", "example-org/restricted-cache"),
            Err(CacheError::PermissionDenied(_))
        ));
        let writable = AuthenticatedGitHubSession::verified_for_test(
            "owner",
            "example-org/restricted-cache",
            RepositoryPermission::Write,
        );
        assert!(matches!(
            writable.require_write_for("someone-else", "example-org/restricted-cache"),
            Err(CacheError::Authentication(_))
        ));
    }

    #[test]
    fn refresh_margin_marks_a_session_before_its_hard_expiry() {
        let session = AuthenticatedGitHubSession::verified_for_test(
            "owner",
            "example-org/cache",
            RepositoryPermission::Write,
        );
        assert!(!session.requires_refresh(0).unwrap());
        assert!(session
            .requires_refresh(AUTHORITY_PROBE_MAX_AGE_SECONDS)
            .unwrap());
    }

    /// curl's per-attempt output: the body, then the status trailer.
    fn response(
        body: &str,
        status: &str,
        retry_after: &str,
        remaining: &str,
        reset: &str,
    ) -> String {
        format!("{body}{GITHUB_API_STATUS_MARKER}{status} {retry_after}|{remaining}|{reset}\n")
    }

    #[test]
    fn api_attempts_classify_status_rate_limits_and_transport_failures() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let ok = response(r#"{"login":"owner"}"#, "200", "", "4999", "2000");
        assert!(matches!(
            classify_api_attempt(Some(0), ok.as_bytes(), "/user", now),
            ApiAttempt::Response(value) if value["login"] == "owner"
        ));
        for status in ["408", "500", "502", "503", "504"] {
            let failed = response(r#"{"message":"temporary"}"#, status, "", "", "");
            assert!(matches!(
                classify_api_attempt(Some(0), failed.as_bytes(), "/user", now),
                ApiAttempt::Retry { wait: None, .. }
            ));
        }
        let limited = response(r#"{"message":"rate"}"#, "403", "", "0", "1030");
        assert!(matches!(
            classify_api_attempt(Some(0), limited.as_bytes(), "/user", now),
            ApiAttempt::Retry { wait: Some(wait), .. } if wait == Duration::from_secs(30)
        ));
        let secondary = response(r#"{"message":"slow down"}"#, "403", "7", "12", "");
        assert!(matches!(
            classify_api_attempt(Some(0), secondary.as_bytes(), "/user", now),
            ApiAttempt::Retry { wait: Some(wait), .. } if wait == Duration::from_secs(7)
        ));
        // 429 and marked 403s follow GitHub's rate-limit waits.
        for status in ["429", "403"] {
            let until_reset = response(r#"{"message":"limit"}"#, status, "", "0", "1030");
            assert!(matches!(
                classify_api_attempt(Some(0), until_reset.as_bytes(), "/user", now),
                ApiAttempt::Retry { wait: Some(wait), reason }
                    if wait == Duration::from_secs(30) && reason == format!("HTTP {status} rate limited until 1030")
            ));
            let stated = response(r#"{"message":"limit"}"#, status, "7", "0", "1030");
            assert!(matches!(
                classify_api_attempt(Some(0), stated.as_bytes(), "/user", now),
                ApiAttempt::Retry { wait: Some(wait), .. } if wait == Duration::from_secs(7)
            ));
        }
        for (status, message) in [
            ("429", "too many"),
            ("403", "You have exceeded a secondary rate limit"),
        ] {
            let untimed = response(
                &format!(r#"{{"message":"{message}"}}"#),
                status,
                "",
                "12",
                "1030",
            );
            assert!(matches!(
                classify_api_attempt(Some(0), untimed.as_bytes(), "/user", now),
                ApiAttempt::Retry { wait: Some(wait), .. } if wait == GITHUB_RATE_LIMIT_MINIMUM_WAIT
            ));
        }
        for status in ["401", "403", "404"] {
            let denied = response(r#"{"message":"no"}"#, status, "", "4999", "2000");
            assert!(matches!(
                classify_api_attempt(Some(0), denied.as_bytes(), "/user", now),
                ApiAttempt::Fail(reason) if reason == format!("HTTP {status}")
            ));
        }
        for code in [6, 7, 28, 35, 56] {
            assert!(matches!(
                classify_api_attempt(Some(code), br#"{"log"#, "/user", now),
                ApiAttempt::Retry { .. }
            ));
        }
        assert!(matches!(
            classify_api_attempt(Some(63), b"", "/user", now),
            ApiAttempt::Fail(reason) if reason == "curl exit 63"
        ));
        assert!(matches!(
            classify_api_attempt(Some(0), b"no trailer", "/user", now),
            ApiAttempt::Fail(_)
        ));
    }

    /// A fake curl that serves the next scripted response per process, as
    /// real curl output arrives per attempt.
    #[cfg(unix)]
    fn scripted_curl(
        label: &str,
        attempts: &[(String, i32)],
    ) -> (crate::test_support::TestDir, GitHubCredentialApiProbe) {
        use std::os::unix::fs::PermissionsExt;
        let scratch = crate::test_support::TestDir::new(label);
        for (index, (output, exit)) in attempts.iter().enumerate() {
            std::fs::write(scratch.join(format!("out.{}", index + 1)), output).unwrap();
            std::fs::write(
                scratch.join(format!("exit.{}", index + 1)),
                exit.to_string(),
            )
            .unwrap();
        }
        let curl = scratch.join("curl");
        std::fs::write(
            &curl,
            concat!(
                "#!/bin/sh\n[ -n \"$XC_FAKE_CURL_READY\" ] && exit 0\n",
                "cat > /dev/null\nd=$(dirname \"$0\")\n",
                "n=$(( $(cat \"$d/count\" 2>/dev/null || echo 0) + 1 ))\necho $n > \"$d/count\"\n",
                "printf '%s\\n' \"$@\" > \"$d/args.$n\"\n",
                "cat \"$d/out.$n\"\nexit $(cat \"$d/exit.$n\")\n"
            ),
        )
        .unwrap();
        std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o755)).unwrap();
        // A test thread forking while this script was open for writing can
        // briefly hold it open (ETXTBSY); wait until it executes.
        for attempt in 0.. {
            match Command::new(&curl).env("XC_FAKE_CURL_READY", "1").status() {
                Err(error) if error.raw_os_error() == Some(26) && attempt < 500 => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                result => {
                    assert!(result.unwrap().success());
                    break;
                }
            }
        }
        let probe = GitHubCredentialApiProbe {
            retry_delay: Duration::ZERO,
            ..GitHubCredentialApiProbe::default().with_curl_executable(curl)
        };
        (scratch, probe)
    }

    #[cfg(unix)]
    fn attempts(scratch: &crate::test_support::TestDir) -> String {
        std::fs::read_to_string(scratch.join("count"))
            .unwrap()
            .trim()
            .to_owned()
    }

    #[cfg(unix)]
    #[test]
    fn api_probe_decodes_only_the_final_response_after_transient_failures() {
        let ok = (
            response(r#"{"login":"owner"}"#, "200", "", "4999", "2000"),
            0,
        );
        for (label, first) in [
            (
                "probe-503-body",
                (response(r#"{"message":"temporary"}"#, "503", "", "", ""), 0),
            ),
            (
                "probe-429-body",
                (response(r#"{"message":"slow"}"#, "429", "0", "", ""), 0),
            ),
            ("probe-partial-timeout", (r#"{"login":"par"#.to_owned(), 28)),
            (
                "probe-rate-limited",
                (response(r#"{"message":"rate"}"#, "403", "0", "0", ""), 0),
            ),
        ] {
            let (scratch, probe) = scripted_curl(label, &[first, ok.clone()]);
            let value = probe.github_get("/user", "secret").unwrap();
            assert_eq!(value["login"], "owner", "{label}");
            assert_eq!(attempts(&scratch), "2", "{label}");
            let arguments = std::fs::read_to_string(scratch.join("args.2")).unwrap();
            assert!(!arguments.contains("secret") && !arguments.contains("--retry"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn api_probe_fails_denials_at_once_and_bounds_transient_retries() {
        let denied = (
            response(r#"{"message":"Not Found"}"#, "404", "", "4999", "2000"),
            0,
        );
        let (scratch, probe) = scripted_curl("probe-denied", &[denied]);
        let error = probe.github_get("/user", "secret").unwrap_err().to_string();
        assert!(
            error.ends_with("probe failed for /user (HTTP 404; 1 attempt)"),
            "{error}"
        );
        assert_eq!(attempts(&scratch), "1");

        let busy = (response(r#"{"message":"busy"}"#, "503", "", "", ""), 0);
        let (scratch, probe) = scripted_curl("probe-busy", &vec![busy; 6]);
        let error = probe.github_get("/user", "secret").unwrap_err().to_string();
        assert!(error.ends_with("(HTTP 503; 5 attempts)"), "{error}");
        assert!(!error.contains("secret"));
        assert_eq!(attempts(&scratch), "5");

        // A rate limit that resets beyond the retry window fails now, with
        // one request, for 403 and 429 alike.
        let reset = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3_600;
        let ok = (
            response(r#"{"login":"owner"}"#, "200", "", "4999", "2000"),
            0,
        );
        for status in ["403", "429"] {
            let limited = (
                response(r#"{"message":"rate"}"#, status, "", "0", &reset.to_string()),
                0,
            );
            let (scratch, probe) = scripted_curl(
                &format!("probe-rate-reset-{status}"),
                &[limited, ok.clone()],
            );
            let error = probe.github_get("/user", "secret").unwrap_err().to_string();
            assert!(
                error.contains(&format!(
                    "HTTP {status} rate limited until {reset}; 1 attempt"
                )),
                "{error}"
            );
            assert_eq!(attempts(&scratch), "1");
        }
    }

    #[cfg(unix)]
    #[test]
    fn api_probe_waits_for_a_429_reset_within_the_retry_window() {
        let ok = (
            response(r#"{"login":"owner"}"#, "200", "", "4999", "2000"),
            0,
        );
        let reset = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 2;
        let limited = (
            response(r#"{"message":"rate"}"#, "429", "", "0", &reset.to_string()),
            0,
        );
        let (scratch, probe) = scripted_curl("probe-429-in-window", &[limited, ok]);
        let started = Instant::now();
        let value = probe.github_get("/user", "secret").unwrap();
        assert_eq!(value["login"], "owner");
        assert_eq!(attempts(&scratch), "2");
        // The retry waited for the reset, not the zero test backoff.
        assert!(
            started.elapsed() >= Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }
}
