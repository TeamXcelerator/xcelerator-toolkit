//! Live GitHub principal and repository-permission preflight.
//!
//! Credentials remain in the external Git credential provider. They are used
//! only long enough to query GitHub's allowlisted API endpoints and are never
//! placed in evidence, diagnostics, command-line arguments, or child-process
//! environment variables.

use crate::{canonical_digest, CacheError, ContentDigest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const GITHUB_API_ORIGIN: &str = "https://api.github.com";
const GITHUB_CREDENTIAL_HOST: &str = "github.com";
const PROVIDER_ID: &str = "git-credential+github-api-curl-v1";
pub const AUTHORITY_PROBE_MAX_AGE_SECONDS: u64 = 300;

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
#[derive(Debug)]
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
                "GitHub permission probe is stale and must be repeated before mutation".to_owned(),
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
}

#[derive(Clone, Debug)]
pub struct GitHubCredentialApiProbe {
    git_executable: OsString,
    curl_executable: OsString,
}

impl Default for GitHubCredentialApiProbe {
    fn default() -> Self {
        Self {
            git_executable: OsString::from("git"),
            curl_executable: OsString::from(if cfg!(windows) { "curl.exe" } else { "curl" }),
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
        let credential = self.git_credential()?;
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
        let config = format!(
            concat!(
                "url = \"{GITHUB_API_ORIGIN}{path}\"\n",
                "request = \"GET\"\n",
                "header = \"Accept: application/vnd.github+json\"\n",
                "header = \"Authorization: Bearer {secret}\"\n",
                "header = \"X-GitHub-Api-Version: 2022-11-28\"\n",
                "silent\nshow-error\nfail-with-body\n"
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
                OsString::from("--config"),
                OsString::from("-"),
            ],
            config.as_bytes(),
            &[],
        )?;
        if !output.status.success() {
            return Err(CacheError::Authentication(format!(
                "GitHub API authentication or permission probe failed for {path}"
            )));
        }
        serde_json::from_slice(&output.stdout).map_err(|_| {
            CacheError::Authentication(format!(
                "GitHub API returned an invalid permission response for {path}"
            ))
        })
    }
}

struct GitCredential {
    #[allow(dead_code)]
    username: String,
    secret: String,
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
        .envs(environment.iter().copied());
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
}
