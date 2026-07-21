//! Direct Git smart-protocol transport using bounded temporary bare state.

use crate::{
    CacheError, CompareAndSwapResult, ContentDigest, RemoteCommitRequest, RemoteGitStore,
    TransportPart,
};
use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Debug)]
pub struct GitCliRemoteStore {
    git_executable: OsString,
    temporary_root: PathBuf,
    staged_parts_roots: Vec<PathBuf>,
    author_name: String,
    author_email: String,
    resources: ResourcePolicy,
    operation_lock: Mutex<()>,
}

impl GitCliRemoteStore {
    pub fn new(
        temporary_root: impl Into<PathBuf>,
        staged_parts_root: impl Into<PathBuf>,
        author_name: impl Into<String>,
        author_email: impl Into<String>,
    ) -> Result<Self, CacheError> {
        let store = Self {
            git_executable: OsString::from("git"),
            temporary_root: temporary_root.into(),
            staged_parts_roots: vec![staged_parts_root.into()],
            author_name: author_name.into(),
            author_email: author_email.into(),
            resources: ResourcePolicy::default(),
            operation_lock: Mutex::new(()),
        };
        if store.author_name.trim().is_empty() || store.author_email.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "Git publication author name and email must be explicit".to_owned(),
            ));
        }
        fs::create_dir_all(&store.temporary_root)?;
        fs::create_dir_all(&store.staged_parts_roots[0])?;
        Ok(store)
    }

    /// Add immutable staging roots to one transport session. Managed
    /// publication uses this to send several artifacts to the same shard
    /// without copying their potentially large encoded parts into a combined
    /// temporary directory.
    pub fn with_additional_staged_parts_roots(
        mut self,
        roots: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, CacheError> {
        for root in roots {
            fs::create_dir_all(&root)?;
            if !self.staged_parts_roots.contains(&root) {
                self.staged_parts_roots.push(root);
            }
        }
        Ok(self)
    }

    pub fn with_git_executable(mut self, executable: impl Into<OsString>) -> Self {
        self.git_executable = executable.into();
        self
    }

    pub fn with_resource_policy(mut self, resources: ResourcePolicy) -> Self {
        self.resources = resources;
        self
    }

    /// Explicit cleanup after a receipt-complete or deliberately abandoned
    /// transaction. No persistent clone is kept by the transport itself.
    pub fn cleanup_session(&self, repository: &str) -> Result<(), CacheError> {
        validate_repository(repository)?;
        let session = self.session_path(repository);
        if session.exists() {
            fs::remove_dir_all(session)?;
        }
        Ok(())
    }

    fn session_path(&self, repository: &str) -> PathBuf {
        self.temporary_root
            .join(ContentDigest::sha256(repository.as_bytes()).0)
            .join("remote.git")
    }

    fn ensure_session(&self, repository: &str) -> Result<PathBuf, CacheError> {
        validate_repository(repository)?;
        let session = self.session_path(repository);
        if !session.join("HEAD").exists() {
            if let Some(parent) = session.parent() {
                fs::create_dir_all(parent)?;
            }
            run_git(
                &self.git_executable,
                None,
                [
                    OsString::from("init"),
                    OsString::from("--bare"),
                    session.as_os_str().to_owned(),
                ],
                &[],
            )?;
            run_git(
                &self.git_executable,
                Some(&session),
                [
                    OsString::from("remote"),
                    OsString::from("add"),
                    OsString::from("origin"),
                    OsString::from(repository),
                ],
                &[],
            )?;
            run_git(
                &self.git_executable,
                Some(&session),
                [
                    OsString::from("config"),
                    OsString::from("remote.origin.promisor"),
                    OsString::from("true"),
                ],
                &[],
            )?;
            run_git(
                &self.git_executable,
                Some(&session),
                [
                    OsString::from("config"),
                    OsString::from("remote.origin.partialclonefilter"),
                    OsString::from("blob:none"),
                ],
                &[],
            )?;
        }
        Ok(session)
    }

    fn fetch_revision(&self, repository: &str, revision: &str) -> Result<PathBuf, CacheError> {
        validate_revision(revision)?;
        let session = self.ensure_session(repository)?;
        // A multi-artifact managed publication advances one shard through a
        // chain of locally created commits. After a successful push, the next
        // expected head is already present in this bare session; avoid a
        // redundant network fetch while retaining the independent remote-head
        // compare-and-swap immediately before every mutation.
        let local = run_git_allow_failure(
            &self.git_executable,
            Some(&session),
            [
                OsString::from("cat-file"),
                OsString::from("-e"),
                OsString::from(format!("{revision}^{{commit}}")),
            ],
            &[],
        )?;
        if local.status.success() {
            return Ok(session);
        }
        run_git_network(
            &self.git_executable,
            Some(&session),
            [
                OsString::from("fetch"),
                OsString::from("--no-tags"),
                OsString::from("--depth=1"),
                OsString::from("--filter=blob:none"),
                OsString::from("origin"),
                OsString::from(revision),
            ],
            &[],
        )?;
        Ok(session)
    }

    fn staged_part_path(&self, part: &TransportPart) -> Result<PathBuf, CacheError> {
        validate_relative_git_path(&part.repository_path)?;
        let components = Path::new(&part.repository_path)
            .components()
            .map(|component| match component {
                Component::Normal(component) => Ok(component.to_owned()),
                _ => Err(CacheError::InvalidManifest(
                    "transport part path is not a normalized relative path".to_owned(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut observed = None;
        for root in self.staged_parts_roots.iter().rev() {
            let mut path = root.clone();
            path.extend(&components);
            if path.is_file() {
                let (digest, size) = digest_file(&path)?;
                if digest == part.content_digest && size == part.size_bytes {
                    return Ok(path);
                }
                observed = Some((digest, size));
            }
        }
        if let Some((digest, size)) = observed {
            return Err(CacheError::DigestMismatch {
                expected: format!("{} ({} bytes)", part.content_digest, part.size_bytes),
                actual: format!("{digest} ({size} bytes)"),
            });
        }
        Err(CacheError::NotFound(format!(
            "staged transport part {:?} in {} configured roots",
            part.repository_path,
            self.staged_parts_roots.len()
        )))
    }

    fn hash_staged_blob(&self, session: &Path, part: &TransportPart) -> Result<String, CacheError> {
        let path = self.staged_part_path(part)?;
        let output = run_git(
            &self.git_executable,
            Some(session),
            [
                OsString::from("hash-object"),
                OsString::from("-w"),
                path.as_os_str().to_owned(),
            ],
            &[],
        )?;
        parse_single_line(&output.stdout, "git hash-object")
    }

    fn commit_tree(
        &self,
        session: &Path,
        request: &RemoteCommitRequest,
    ) -> Result<String, CacheError> {
        let index_name = format!(
            "publication-index-{}",
            ContentDigest::sha256(
                format!("{}:{}", request.expected_head, request.message).as_bytes()
            )
        );
        let index_path = session.parent().unwrap_or(session).join(index_name);
        let index_environment = [(OsStr::new("GIT_INDEX_FILE"), index_path.as_os_str())];
        let result = (|| {
            run_git(
                &self.git_executable,
                Some(session),
                [
                    OsString::from("read-tree"),
                    OsString::from(&request.expected_head),
                ],
                &index_environment,
            )?;
            for part in &request.parts {
                let oid = self.hash_staged_blob(session, part)?;
                run_git(
                    &self.git_executable,
                    Some(session),
                    [
                        OsString::from("update-index"),
                        OsString::from("--add"),
                        OsString::from("--cacheinfo"),
                        OsString::from("100644"),
                        OsString::from(oid),
                        OsString::from(&part.repository_path),
                    ],
                    &index_environment,
                )?;
            }
            if !request.delete_paths.is_empty() {
                let mut removals = Vec::new();
                for path in &request.delete_paths {
                    removals.extend_from_slice(
                        format!("0 0000000000000000000000000000000000000000\t{path}\0").as_bytes(),
                    );
                }
                run_git_with_input(
                    &self.git_executable,
                    Some(session),
                    [
                        OsString::from("update-index"),
                        OsString::from("-z"),
                        OsString::from("--index-info"),
                    ],
                    &index_environment,
                    &removals,
                )?;
            }
            let tree = run_git(
                &self.git_executable,
                Some(session),
                [OsString::from("write-tree")],
                &index_environment,
            )?;
            let tree = parse_single_line(&tree.stdout, "git write-tree")?;
            let author_environment = [
                (OsStr::new("GIT_AUTHOR_NAME"), OsStr::new(&self.author_name)),
                (
                    OsStr::new("GIT_AUTHOR_EMAIL"),
                    OsStr::new(&self.author_email),
                ),
                (
                    OsStr::new("GIT_COMMITTER_NAME"),
                    OsStr::new(&self.author_name),
                ),
                (
                    OsStr::new("GIT_COMMITTER_EMAIL"),
                    OsStr::new(&self.author_email),
                ),
            ];
            let commit = run_git(
                &self.git_executable,
                Some(session),
                [
                    OsString::from("commit-tree"),
                    OsString::from(tree),
                    OsString::from("-p"),
                    OsString::from(&request.expected_head),
                    OsString::from("-m"),
                    OsString::from(&request.message),
                ],
                &author_environment,
            )?;
            parse_single_line(&commit.stdout, "git commit-tree")
        })();
        let _ = fs::remove_file(index_path);
        result
    }
}

impl RemoteGitStore for GitCliRemoteStore {
    fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError> {
        validate_repository(repository)?;
        validate_branch(branch)?;
        let output = run_git_network(
            &self.git_executable,
            None,
            [
                OsString::from("ls-remote"),
                OsString::from("--heads"),
                OsString::from(repository),
                OsString::from(format!("refs/heads/{branch}")),
            ],
            &[],
        )?;
        let line = String::from_utf8(output.stdout)
            .map_err(|error| CacheError::Io(format!("git ls-remote returned non-UTF8: {error}")))?;
        let revision = line.split_whitespace().next().ok_or_else(|| {
            CacheError::NotFound(format!("branch {branch:?} in repository {repository:?}"))
        })?;
        validate_revision(revision)?;
        Ok(revision.to_owned())
    }

    fn immutable_path_digest(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
    ) -> Result<Option<ContentDigest>, CacheError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        validate_relative_git_path(path)?;
        let session = self.fetch_revision(repository, revision)?;
        let listing = run_git(
            &self.git_executable,
            Some(&session),
            [
                OsString::from("ls-tree"),
                OsString::from(revision),
                OsString::from("--"),
                OsString::from(path),
            ],
            &[],
        )?;
        if listing.stdout.is_empty() {
            return Ok(None);
        }
        let specification = format!("{revision}:{path}");
        let mut child = Command::new(&self.git_executable)
            .arg("-C")
            .arg(&session)
            .arg("show")
            .arg(&specification)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| CacheError::Io(format!("failed to launch git show: {error}")))?;
        let mut stdout = child.stdout.take().ok_or_else(|| {
            CacheError::Io("git show did not provide a readable stream".to_owned())
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let count = stdout.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let status = child.wait()?;
        if !status.success() {
            return Err(CacheError::Io(format!(
                "git show failed with status {status} for {path:?}"
            )));
        }
        Ok(Some(ContentDigest(format!("{:x}", hasher.finalize()))))
    }

    fn read_committed_path(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        maximum_bytes: u64,
        cancellation: &CancellationToken,
        writer: &mut dyn Write,
    ) -> Result<crate::RemoteReadReport, CacheError> {
        if maximum_bytes == 0 {
            return Err(CacheError::ResourceLimit(
                "remote path read limit must be positive".to_owned(),
            ));
        }
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        validate_relative_git_path(path)?;
        let session = self.fetch_revision(repository, revision)?;
        let listing = run_git(
            &self.git_executable,
            Some(&session),
            [
                OsString::from("ls-tree"),
                OsString::from(revision),
                OsString::from("--"),
                OsString::from(path),
            ],
            &[],
        )?;
        if listing.stdout.is_empty() {
            return Err(CacheError::NotFound(format!(
                "remote path {path:?} at {revision}"
            )));
        }
        let effective_maximum = self
            .resources
            .maximum_transfer_bytes
            .map_or(maximum_bytes, |resource_maximum| {
                maximum_bytes.min(resource_maximum)
            });
        if effective_maximum == 0 {
            return Err(CacheError::ResourceLimit(
                "remote path read exceeds the zero transfer budget".to_owned(),
            ));
        }
        let specification = format!("{revision}:{path}");
        let mut child = Command::new(&self.git_executable)
            .arg("-C")
            .arg(&session)
            .arg("show")
            .arg(&specification)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| CacheError::Io(format!("failed to launch git show: {error}")))?;
        let result = (|| {
            let mut stdout = child.stdout.take().ok_or_else(|| {
                CacheError::Io("git show did not provide a readable stream".to_owned())
            })?;
            let buffer_size = effective_maximum.saturating_add(1).min(1024 * 1024) as usize;
            let mut buffer = vec![0u8; buffer_size];
            let mut hasher = Sha256::new();
            let mut size_bytes = 0u64;
            loop {
                cancellation
                    .check()
                    .map_err(|error| CacheError::Cancelled(error.to_string()))?;
                let count = stdout.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                size_bytes = size_bytes.saturating_add(count as u64);
                if size_bytes > effective_maximum {
                    return Err(CacheError::ResourceLimit(format!(
                        "remote path {path:?} exceeds {effective_maximum} bytes"
                    )));
                }
                hasher.update(&buffer[..count]);
                writer.write_all(&buffer[..count])?;
            }
            Ok((
                size_bytes,
                ContentDigest(format!("{:x}", hasher.finalize())),
            ))
        })();
        if result.is_err() {
            let _ = child.kill();
        }
        let status = child.wait()?;
        let (size_bytes, content_digest) = result?;
        if !status.success() {
            return Err(CacheError::Io(format!(
                "git show failed with status {status} for {path:?}"
            )));
        }
        Ok(crate::RemoteReadReport {
            repository_path: path.to_owned(),
            revision: revision.to_owned(),
            size_bytes,
            content_digest,
        })
    }

    fn list_committed_paths(
        &self,
        repository: &str,
        revision: &str,
        prefix: &str,
        maximum_paths: u64,
        maximum_total_path_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<crate::RemotePathListReport, CacheError> {
        if maximum_paths == 0 || maximum_total_path_bytes == 0 {
            return Err(CacheError::ResourceLimit(
                "remote tree-enumeration bounds must be positive".to_owned(),
            ));
        }
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        validate_relative_git_path(prefix)?;
        let session = self.fetch_revision(repository, revision)?;
        let mut child = Command::new(&self.git_executable)
            .arg("-C")
            .arg(&session)
            .args(["ls-tree", "-r", "-z", "--name-only"])
            .arg(revision)
            .arg("--")
            .arg(prefix)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| CacheError::Io(format!("failed to launch git ls-tree: {error}")))?;
        let result = (|| {
            let mut stdout = child.stdout.take().ok_or_else(|| {
                CacheError::Io("git ls-tree did not provide a readable stream".to_owned())
            })?;
            let mut buffer = [0u8; 64 * 1024];
            let mut current_path = Vec::new();
            let mut paths = Vec::new();
            let mut total_path_bytes = 0u64;
            loop {
                cancellation
                    .check()
                    .map_err(|error| CacheError::Cancelled(error.to_string()))?;
                let count = stdout.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                for byte in &buffer[..count] {
                    if *byte != 0 {
                        current_path.push(*byte);
                        if current_path.len() as u64 > maximum_total_path_bytes {
                            return Err(CacheError::ResourceLimit(
                                "one remote tree path exceeds the path-byte budget".to_owned(),
                            ));
                        }
                        continue;
                    }
                    let path =
                        String::from_utf8(std::mem::take(&mut current_path)).map_err(|error| {
                            CacheError::InvalidManifest(format!(
                                "remote tree contains a non-UTF8 path: {error}"
                            ))
                        })?;
                    validate_relative_git_path(&path)?;
                    if path != prefix && !path.starts_with(&format!("{prefix}/")) {
                        return Err(CacheError::InvalidManifest(format!(
                            "remote tree returned path {path:?} outside prefix {prefix:?}"
                        )));
                    }
                    total_path_bytes =
                        total_path_bytes
                            .checked_add(path.len() as u64)
                            .ok_or_else(|| {
                                CacheError::ResourceLimit(
                                    "remote tree path bytes exceed u64".to_owned(),
                                )
                            })?;
                    paths.push(path);
                    if paths.len() as u64 > maximum_paths
                        || total_path_bytes > maximum_total_path_bytes
                    {
                        return Err(CacheError::ResourceLimit(format!(
                            "remote tree prefix {prefix:?} exceeds its enumeration bounds"
                        )));
                    }
                }
            }
            if !current_path.is_empty() {
                return Err(CacheError::InvalidManifest(
                    "remote tree path stream is not NUL-terminated".to_owned(),
                ));
            }
            if paths.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(CacheError::InvalidManifest(
                    "remote tree paths are duplicated or not canonical".to_owned(),
                ));
            }
            Ok(crate::RemotePathListReport {
                prefix: prefix.to_owned(),
                revision: revision.to_owned(),
                paths,
                total_path_bytes,
            })
        })();
        if result.is_err() {
            let _ = child.kill();
        }
        let status = child.wait()?;
        let report = result?;
        if !status.success() {
            return Err(CacheError::Io(format!(
                "git ls-tree failed with status {status} for prefix {prefix:?}"
            )));
        }
        Ok(report)
    }

    fn compare_and_swap_commit(
        &self,
        request: &RemoteCommitRequest,
    ) -> Result<CompareAndSwapResult, CacheError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        validate_repository(&request.repository)?;
        validate_branch(&request.branch)?;
        validate_revision(&request.expected_head)?;
        let batch_bytes = request.validate_limits()?;
        if self
            .resources
            .maximum_temporary_disk_bytes
            .is_some_and(|maximum| batch_bytes > maximum)
        {
            return Err(CacheError::ResourceLimit(format!(
                "publication batch {batch_bytes} exceeds temporary-disk budget"
            )));
        }
        if self
            .resources
            .maximum_transfer_bytes
            .is_some_and(|maximum| batch_bytes > maximum)
        {
            return Err(CacheError::ResourceLimit(format!(
                "publication batch {batch_bytes} exceeds transfer budget"
            )));
        }
        for part in &request.parts {
            validate_relative_git_path(&part.repository_path)?;
        }
        for path in &request.delete_paths {
            validate_relative_git_path(path)?;
        }
        let current_head = self.read_ref(&request.repository, &request.branch)?;
        if current_head != request.expected_head {
            return Ok(CompareAndSwapResult::RefConflict { current_head });
        }
        let session = self.fetch_revision(&request.repository, &request.expected_head)?;
        let commit_id = self.commit_tree(&session, request)?;
        let push = run_git_allow_failure(
            &self.git_executable,
            Some(&session),
            [
                OsString::from("push"),
                OsString::from("origin"),
                OsString::from(format!("{commit_id}:refs/heads/{}", request.branch)),
            ],
            &[],
        )?;
        if push.status.success() {
            Ok(CompareAndSwapResult::Committed { commit_id })
        } else {
            let current_head = self.read_ref(&request.repository, &request.branch)?;
            if current_head != request.expected_head {
                Ok(CompareAndSwapResult::RefConflict { current_head })
            } else {
                Err(command_failure("git push", &push))
            }
        }
    }

    fn verify_committed_part(
        &self,
        repository: &str,
        revision: &str,
        part: &TransportPart,
    ) -> Result<(), CacheError> {
        match self.immutable_path_digest(repository, revision, &part.repository_path)? {
            Some(actual) if actual == part.content_digest => Ok(()),
            Some(actual) => Err(CacheError::DigestMismatch {
                expected: part.content_digest.to_string(),
                actual: actual.to_string(),
            }),
            None => Err(CacheError::NotFound(format!(
                "verified remote part {:?} at {revision}",
                part.repository_path
            ))),
        }
    }
}

fn run_git<I>(
    executable: &OsStr,
    directory: Option<&Path>,
    arguments: I,
    environment: &[(&OsStr, &OsStr)],
) -> Result<Output, CacheError>
where
    I: IntoIterator<Item = OsString>,
{
    let output = run_git_allow_failure(executable, directory, arguments, environment)?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(command_failure("git", &output))
    }
}

fn run_git_network<I>(
    executable: &OsStr,
    directory: Option<&Path>,
    arguments: I,
    environment: &[(&OsStr, &OsStr)],
) -> Result<Output, CacheError>
where
    I: IntoIterator<Item = OsString>,
{
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    const MAX_ATTEMPTS: u32 = 5;
    for attempt in 1..=MAX_ATTEMPTS {
        let output = run_git_allow_failure(
            executable,
            directory,
            arguments.iter().cloned(),
            environment,
        )?;
        if output.status.success() {
            return Ok(output);
        }
        if attempt == MAX_ATTEMPTS || !transient_git_failure(&output) {
            return Err(command_failure("git", &output));
        }
        thread::sleep(Duration::from_secs(1 << (attempt - 1)));
    }
    unreachable!("bounded Git retry loop always returns")
}

fn transient_git_failure(output: &Output) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr).to_ascii_lowercase();
    [
        "connection reset",
        "connection was reset",
        "connection timed out",
        "operation timed out",
        "the remote end hung up unexpectedly",
        "unexpected disconnect",
        "early eof",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "requested url returned error: 500",
        "requested url returned error: 502",
        "requested url returned error: 503",
        "requested url returned error: 504",
    ]
    .iter()
    .any(|pattern| stderr.contains(pattern))
}

fn run_git_allow_failure<I>(
    executable: &OsStr,
    directory: Option<&Path>,
    arguments: I,
    environment: &[(&OsStr, &OsStr)],
) -> Result<Output, CacheError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut command = Command::new(executable);
    // Public cache reads must never block an unattended consumer run on an
    // interactive credential prompt. Configured credential helpers and PATs
    // continue to work for explicit authenticated operations.
    command.env("GIT_TERMINAL_PROMPT", "0");
    if let Some(directory) = directory {
        command.arg("-C").arg(directory);
    }
    command.args(arguments);
    for (key, value) in environment {
        command.env(key, value);
    }
    command
        .output()
        .map_err(|error| CacheError::Io(format!("failed to launch git: {error}")))
}

fn run_git_with_input<I>(
    executable: &OsStr,
    directory: Option<&Path>,
    arguments: I,
    environment: &[(&OsStr, &OsStr)],
    input: &[u8],
) -> Result<Output, CacheError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut command = Command::new(executable);
    command.env("GIT_TERMINAL_PROMPT", "0");
    if let Some(directory) = directory {
        command.arg("-C").arg(directory);
    }
    command.args(arguments);
    for (key, value) in environment {
        command.env(key, value);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| CacheError::Io(format!("failed to launch git: {error}")))?;
    child
        .stdin
        .take()
        .ok_or_else(|| CacheError::Io("git stdin was unavailable".to_owned()))?
        .write_all(input)?;
    let output = child.wait_with_output()?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(command_failure("git", &output))
    }
}

fn command_failure(operation: &str, output: &Output) -> CacheError {
    let stderr = String::from_utf8_lossy(&output.stderr);
    CacheError::Io(format!(
        "{operation} failed with status {}: {}",
        output.status,
        stderr.trim()
    ))
}

fn parse_single_line(bytes: &[u8], operation: &str) -> Result<String, CacheError> {
    let value = std::str::from_utf8(bytes)
        .map_err(|error| CacheError::Io(format!("{operation} returned non-UTF8: {error}")))?
        .trim();
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        return Err(CacheError::Io(format!(
            "{operation} returned an invalid object identity"
        )));
    }
    Ok(value.to_owned())
}

fn validate_repository(repository: &str) -> Result<(), CacheError> {
    if repository.trim().is_empty()
        || repository.starts_with('-')
        || repository.chars().any(|character| character.is_control())
    {
        return Err(CacheError::InvalidManifest(
            "Git repository identity is empty or unsafe".to_owned(),
        ));
    }
    Ok(())
}

fn validate_branch(branch: &str) -> Result<(), CacheError> {
    if branch.trim().is_empty()
        || branch.starts_with('-')
        || branch.starts_with('/')
        || branch.ends_with('/')
        || branch.ends_with('.')
        || branch.contains("..")
        || branch.contains("@{")
        || branch
            .chars()
            .any(|character| character.is_control() || " ~^:?*[\\".contains(character))
    {
        return Err(CacheError::InvalidManifest(format!(
            "Git branch {branch:?} is unsafe"
        )));
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), CacheError> {
    if !matches!(revision.len(), 40 | 64) || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CacheError::InvalidManifest(
            "Git revision must be a full hexadecimal object id".to_owned(),
        ));
    }
    Ok(())
}

fn validate_relative_git_path(path: &str) -> Result<(), CacheError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.starts_with('\\')
        || path.contains('\\')
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(CacheError::InvalidManifest(format!(
            "Git path {path:?} is not a normalized relative path"
        )));
    }
    Ok(())
}

fn digest_file(path: &Path) -> Result<(ContentDigest, u64), CacheError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        size = size
            .checked_add(count as u64)
            .ok_or_else(|| CacheError::ResourceLimit("staged part exceeds u64".to_owned()))?;
    }
    Ok((ContentDigest(format!("{:x}", hasher.finalize())), size))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_root(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("{name}-{}", std::process::id()))
    }

    fn test_git(directory: Option<&Path>, arguments: &[&str]) -> bool {
        let mut command = Command::new("git");
        if let Some(directory) = directory {
            command.arg("-C").arg(directory);
        }
        command.args(arguments);
        command.stdout(Stdio::null()).stderr(Stdio::null());
        command.status().is_ok_and(|status| status.success())
    }

    #[test]
    fn multiple_staging_roots_select_the_identity_matching_file() {
        let root = temporary_root("git-cli-multiple-staging-roots");
        let _ = fs::remove_dir_all(&root);
        let first = root.join("first");
        let second = root.join("second");
        let relative = Path::new("indexes/family/partition.json");
        fs::create_dir_all(first.join(relative).parent().unwrap()).unwrap();
        fs::create_dir_all(second.join(relative).parent().unwrap()).unwrap();
        fs::write(first.join(relative), b"older sidecar").unwrap();
        fs::write(second.join(relative), b"current sidecar").unwrap();
        let part = TransportPart {
            sequence: 0,
            repository_path: relative.to_string_lossy().replace('\\', "/"),
            size_bytes: b"current sidecar".len() as u64,
            content_digest: ContentDigest::sha256(b"current sidecar"),
        };
        let store = GitCliRemoteStore::new(
            root.join("transport"),
            &first,
            "Test Publisher",
            "test@example.invalid",
        )
        .unwrap()
        .with_additional_staged_parts_roots(vec![second.clone()])
        .unwrap();
        assert_eq!(
            store.staged_part_path(&part).unwrap(),
            second.join(relative)
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn direct_git_transport_commits_without_a_checkout_of_the_remote() {
        if !test_git(None, &["--version"]) {
            return;
        }
        let root = temporary_root("git-cli-transport");
        let _ = fs::remove_dir_all(&root);
        let remote = root.join("remote.git");
        let seed = root.join("seed");
        let temporary = root.join("transport");
        let staging = root.join("staging");
        fs::create_dir_all(&root).unwrap();
        assert!(test_git(
            None,
            &["init", "--bare", remote.to_str().unwrap()]
        ));
        assert!(test_git(None, &["init", seed.to_str().unwrap()]));
        assert!(test_git(
            Some(&seed),
            &["config", "user.name", "Test Publisher"]
        ));
        assert!(test_git(
            Some(&seed),
            &["config", "user.email", "test@example.invalid"]
        ));
        fs::write(seed.join("README.md"), b"seed\n").unwrap();
        assert!(test_git(Some(&seed), &["add", "README.md"]));
        assert!(test_git(Some(&seed), &["commit", "-m", "seed"]));
        assert!(test_git(
            Some(&seed),
            &["remote", "add", "origin", remote.to_str().unwrap()]
        ));
        assert!(test_git(
            Some(&seed),
            &["push", "origin", "HEAD:refs/heads/main"]
        ));

        let payload = b"direct cache payload";
        let digest = ContentDigest::sha256(payload);
        let repository_path = format!("objects/sha256/{}/{}.part", &digest.0[..2], digest.0);
        let local_path = staging.join(Path::new(&repository_path));
        fs::create_dir_all(local_path.parent().unwrap()).unwrap();
        fs::write(&local_path, payload).unwrap();
        let part = TransportPart {
            sequence: 0,
            repository_path,
            size_bytes: payload.len() as u64,
            content_digest: digest,
        };
        let store = GitCliRemoteStore::new(
            &temporary,
            &staging,
            "Test Publisher",
            "test@example.invalid",
        )
        .unwrap();
        let repository = remote.to_string_lossy().to_string();
        let expected_head = store.read_ref(&repository, "main").unwrap();
        let result = store
            .compare_and_swap_commit(&RemoteCommitRequest {
                repository: repository.clone(),
                branch: "main".to_owned(),
                expected_head,
                message: "publish test part".to_owned(),
                parts: vec![part.clone()],
                delete_paths: Vec::new(),
            })
            .unwrap();
        let CompareAndSwapResult::Committed { commit_id } = result else {
            panic!("local test publication unexpectedly conflicted");
        };
        store
            .verify_committed_part(&repository, &commit_id, &part)
            .unwrap();
        let mut read_back = Vec::new();
        let read = store
            .read_committed_path(
                &repository,
                &commit_id,
                &part.repository_path,
                part.size_bytes,
                &CancellationToken::new(),
                &mut read_back,
            )
            .unwrap();
        assert_eq!(read_back, payload);
        assert_eq!(read.size_bytes, payload.len() as u64);
        assert_eq!(read.content_digest, part.content_digest);
        let listing = store
            .list_committed_paths(
                &repository,
                &commit_id,
                "objects",
                10,
                4_096,
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(listing.paths, vec![part.repository_path.clone()]);
        let deletion = store
            .compare_and_swap_commit(&RemoteCommitRequest {
                repository: repository.clone(),
                branch: "main".to_owned(),
                expected_head: commit_id,
                message: "prune test part".to_owned(),
                parts: Vec::new(),
                delete_paths: vec![part.repository_path.clone()],
            })
            .unwrap();
        let CompareAndSwapResult::Committed { commit_id } = deletion else {
            panic!("local test deletion unexpectedly conflicted");
        };
        assert!(store
            .immutable_path_digest(&repository, &commit_id, &part.repository_path)
            .unwrap()
            .is_none());
        assert!(!temporary.join("working-tree").exists());
        store.cleanup_session(&repository).unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unsafe_ref_and_path_inputs_are_rejected() {
        assert!(validate_branch("--force").is_err());
        assert!(validate_branch("main..evil").is_err());
        assert!(validate_relative_git_path("../secret").is_err());
        assert!(validate_relative_git_path("C:\\secret").is_err());
    }
}
