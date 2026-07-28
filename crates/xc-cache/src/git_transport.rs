//! Direct Git smart-protocol transport using bounded temporary bare state.

use crate::{
    AtomicCompareAndSwapResult, AtomicRemoteCommitRequest, CacheError, CompareAndSwapResult,
    ContentDigest, CreateRefResult, RemoteCommitRequest, RemoteGitStore, RemoteRefCreationRequest,
    TransportPart,
};
use fs2::{available_space, total_space};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug)]
struct GitTreeLeaf {
    mode: String,
    object_type: String,
    object_id: String,
}

#[derive(Default)]
struct FlatGitTreeNode {
    children: BTreeMap<String, String>,
    leaves: BTreeMap<String, GitTreeLeaf>,
}

#[derive(Debug)]
pub struct GitCliRemoteStore {
    git_executable: OsString,
    temporary_root: PathBuf,
    staged_parts_roots: Vec<PathBuf>,
    author_name: String,
    author_email: String,
    resources: ResourcePolicy,
    operation_lock: Mutex<()>,
    prepared_blob_oids: Mutex<BTreeMap<(PathBuf, ContentDigest, u64), String>>,
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
            prepared_blob_oids: Mutex::new(BTreeMap::new()),
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
            fs::remove_dir_all(&session)?;
        }
        self.prepared_blob_oids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|(prepared_session, _, _), _| prepared_session != &session);
        Ok(())
    }

    /// Remove every ephemeral bare repository owned by this transport store.
    ///
    /// A store's temporary root is never shared with durable artifact staging.
    /// This is therefore safe after success, failure, or an abandoned preflight.
    pub fn cleanup_all_sessions(&self) -> Result<(), CacheError> {
        if self.temporary_root.exists() {
            fs::remove_dir_all(&self.temporary_root)?;
        }
        self.prepared_blob_oids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        Ok(())
    }

    /// Validate and insert staged blobs into the local Git object database
    /// before a remote publication lease is acquired. Commit construction can
    /// then reuse the prepared object IDs without rereading large payload
    /// parts while the destination repository is locked.
    pub fn prepare_staged_parts(
        &self,
        repository: &str,
        parts: &[TransportPart],
    ) -> Result<(), CacheError> {
        validate_repository(repository)?;
        let additional_bytes = parts.iter().try_fold(0u64, |total, part| {
            total.checked_add(part.size_bytes).ok_or_else(|| {
                CacheError::ResourceLimit(
                    "prepared Git transport parts exceed the u64 byte range".to_owned(),
                )
            })
        })?;
        self.enforce_transport_disk_budget(additional_bytes)?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = self.ensure_session(repository)?;
        self.hash_staged_blobs(&session, parts)?;
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
        }
        Ok(session)
    }

    fn enforce_transport_disk_budget(&self, additional_bytes: u64) -> Result<(), CacheError> {
        fs::create_dir_all(&self.temporary_root)?;
        let current_bytes = directory_size_bytes(&self.temporary_root)?;
        let projected_bytes = current_bytes.checked_add(additional_bytes).ok_or_else(|| {
            CacheError::ResourceLimit("Git transport disk projection exceeds u64".to_owned())
        })?;
        if self
            .resources
            .maximum_temporary_disk_bytes
            .is_some_and(|maximum| projected_bytes > maximum)
        {
            return Err(CacheError::ResourceLimit(format!(
                "Git transport requires at least {projected_bytes} temporary bytes \
                 ({current_bytes} existing + {additional_bytes} pending), exceeding the configured limit"
            )));
        }
        let available = available_space(&self.temporary_root)?;
        let total = total_space(&self.temporary_root)?;
        let reserve = (total / 20).clamp(2 * 1024 * 1024 * 1024, 20 * 1024 * 1024 * 1024);
        let required = additional_bytes.checked_add(reserve).ok_or_else(|| {
            CacheError::ResourceLimit("Git transport free-space requirement exceeds u64".to_owned())
        })?;
        if available < required {
            return Err(CacheError::ResourceLimit(format!(
                "Git transport requires {additional_bytes} additional bytes plus a {reserve}-byte \
                 filesystem reserve, but only {available} bytes are available"
            )));
        }
        Ok(())
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
        self.enforce_transport_disk_budget(0)?;
        run_git_network(
            &self.git_executable,
            Some(&session),
            [
                OsString::from("fetch"),
                OsString::from("--no-tags"),
                OsString::from("--filter=blob:none"),
                OsString::from("origin"),
                OsString::from(revision),
            ],
            &[],
        )?;
        self.disable_implicit_lazy_fetch(&session)?;
        Ok(session)
    }

    fn disable_implicit_lazy_fetch(&self, session: &Path) -> Result<(), CacheError> {
        for key in ["remote.origin.promisor", "remote.origin.partialclonefilter"] {
            let output = run_git_allow_failure(
                &self.git_executable,
                Some(session),
                [
                    OsString::from("config"),
                    OsString::from("--unset-all"),
                    OsString::from(key),
                ],
                &[],
            )?;
            // Git config returns 5 when the requested key was already absent.
            if !output.status.success() && output.status.code() != Some(5) {
                return Err(CacheError::Io(format!(
                    "git config could not disable implicit lazy fetch for {key}: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                )));
            }
        }
        Ok(())
    }

    fn ensure_blob_available(
        &self,
        session: &Path,
        _repository: &str,
        object_id: &str,
        maximum_bytes: Option<u64>,
    ) -> Result<(), CacheError> {
        validate_revision(object_id)?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let local = run_git_allow_failure(
            &self.git_executable,
            Some(session),
            [
                OsString::from("cat-file"),
                OsString::from("-e"),
                OsString::from(format!("{object_id}^{{blob}}")),
            ],
            &[],
        )?;
        if local.status.success() {
            return Ok(());
        }
        if let Some(maximum) = maximum_bytes {
            self.enforce_transport_disk_budget(maximum)?;
        } else {
            self.enforce_transport_disk_budget(0)?;
        }
        // Fetch exactly the missing blob object. A blob has no referenced
        // children, so this cannot recursively hydrate historical payloads.
        run_git_network(
            &self.git_executable,
            Some(session),
            [
                OsString::from("fetch"),
                OsString::from("--no-tags"),
                OsString::from("origin"),
                OsString::from(object_id),
            ],
            &[],
        )?;
        let verified = run_git_allow_failure(
            &self.git_executable,
            Some(session),
            [
                OsString::from("cat-file"),
                OsString::from("-e"),
                OsString::from(format!("{object_id}^{{blob}}")),
            ],
            &[],
        )?;
        if !verified.status.success() {
            return Err(command_failure(
                "git cat-file after explicit blob fetch",
                &verified,
            ));
        }
        Ok(())
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

    fn hash_staged_blobs(
        &self,
        session: &Path,
        parts: &[TransportPart],
    ) -> Result<Vec<String>, CacheError> {
        const MAXIMUM_PATHS_PER_INVOCATION: usize = 256;
        const MAXIMUM_ARGUMENT_BYTES_PER_INVOCATION: usize = 24 * 1024;

        let session = session.to_path_buf();
        let keys = parts
            .iter()
            .map(|part| {
                validate_relative_git_path(&part.repository_path)?;
                Ok((
                    session.clone(),
                    part.content_digest.clone(),
                    part.size_bytes,
                ))
            })
            .collect::<Result<Vec<_>, CacheError>>()?;
        let mut resolved = BTreeMap::new();
        {
            let cache = self
                .prepared_blob_oids
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for key in &keys {
                if let Some(object_id) = cache.get(key) {
                    resolved.insert(key.clone(), object_id.clone());
                }
            }
        }
        let mut missing = BTreeMap::new();
        for (part, key) in parts.iter().zip(&keys) {
            if resolved.contains_key(key) || missing.contains_key(key) {
                continue;
            }
            // Resolving the staged path performs the required SHA-256 and
            // length validation before Git is allowed to ingest the bytes.
            missing.insert(key.clone(), self.staged_part_path(part)?);
        }

        let missing = missing.into_iter().collect::<Vec<_>>();
        let mut start = 0;
        while start < missing.len() {
            let mut end = start;
            let mut argument_bytes = 0usize;
            while end < missing.len() && end - start < MAXIMUM_PATHS_PER_INVOCATION {
                let path_bytes = missing[end].1.as_os_str().to_string_lossy().len();
                if end > start
                    && argument_bytes.saturating_add(path_bytes)
                        > MAXIMUM_ARGUMENT_BYTES_PER_INVOCATION
                {
                    break;
                }
                argument_bytes = argument_bytes.saturating_add(path_bytes);
                end += 1;
            }
            let mut arguments = vec![
                OsString::from("hash-object"),
                OsString::from("-w"),
                OsString::from("--"),
            ];
            arguments.extend(
                missing[start..end]
                    .iter()
                    .map(|(_, path)| path.as_os_str().to_owned()),
            );
            let output = run_git(&self.git_executable, Some(&session), arguments, &[])?;
            let object_ids = parse_object_id_lines(
                &output.stdout,
                "git hash-object",
                missing[start..end].len(),
            )?;
            for ((key, _), object_id) in missing[start..end].iter().zip(object_ids) {
                resolved.insert(key.clone(), object_id);
            }
            start = end;
        }
        let object_ids = keys
            .iter()
            .map(|key| {
                resolved.get(key).cloned().ok_or_else(|| {
                    CacheError::InvalidTransition(
                        "prepared Git blob object identity was not recorded".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut cache = self
            .prepared_blob_oids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for key in keys {
            if let Some(object_id) = resolved.get(&key) {
                cache.insert(key, object_id.clone());
            }
        }
        Ok(object_ids)
    }

    fn update_index_parts(
        &self,
        session: &Path,
        parts: &[TransportPart],
        object_ids: &[String],
        delete_paths: &[String],
        environment: &[(&OsStr, &OsStr)],
    ) -> Result<(), CacheError> {
        if parts.len() != object_ids.len() {
            return Err(CacheError::InvalidTransition(
                "Git index update has a different number of parts and object identities".to_owned(),
            ));
        }
        if parts.is_empty() && delete_paths.is_empty() {
            return Ok(());
        }
        let mut input = Vec::new();
        for (part, object_id) in parts.iter().zip(object_ids) {
            validate_relative_git_path(&part.repository_path)?;
            input.extend_from_slice(format!("100644 {object_id}\t").as_bytes());
            input.extend_from_slice(part.repository_path.as_bytes());
            input.push(0);
        }
        for path in delete_paths {
            validate_relative_git_path(path)?;
            input.extend_from_slice(b"0 0000000000000000000000000000000000000000\t");
            input.extend_from_slice(path.as_bytes());
            input.push(0);
        }
        run_git_with_input(
            &self.git_executable,
            Some(session),
            [
                OsString::from("update-index"),
                OsString::from("-z"),
                OsString::from("--index-info"),
            ],
            environment,
            &input,
        )?;
        Ok(())
    }

    fn read_tree_leaves(
        &self,
        session: &Path,
        revision: &str,
    ) -> Result<BTreeMap<String, GitTreeLeaf>, CacheError> {
        let output = run_git(
            &self.git_executable,
            Some(session),
            [
                OsString::from("ls-tree"),
                OsString::from("-r"),
                OsString::from("-z"),
                OsString::from(revision),
            ],
            &[],
        )?;
        let mut leaves = BTreeMap::new();
        for record in output.stdout.split(|byte| *byte == 0) {
            if record.is_empty() {
                continue;
            }
            let record = std::str::from_utf8(record).map_err(|error| {
                CacheError::InvalidManifest(format!(
                    "Git tree contains a non-UTF-8 repository path: {error}"
                ))
            })?;
            let (metadata, path) = record.split_once('\t').ok_or_else(|| {
                CacheError::InvalidManifest(
                    "git ls-tree returned a malformed tree entry".to_owned(),
                )
            })?;
            let mut metadata = metadata.split_whitespace();
            let mode = metadata.next().ok_or_else(|| {
                CacheError::InvalidManifest("Git tree entry is missing its mode".to_owned())
            })?;
            let object_type = metadata.next().ok_or_else(|| {
                CacheError::InvalidManifest("Git tree entry is missing its object type".to_owned())
            })?;
            let object_id = metadata.next().ok_or_else(|| {
                CacheError::InvalidManifest("Git tree entry is missing its object id".to_owned())
            })?;
            if metadata.next().is_some() {
                return Err(CacheError::InvalidManifest(
                    "Git tree entry contains unexpected metadata".to_owned(),
                ));
            }
            validate_revision(object_id)?;
            validate_relative_git_path(path)?;
            leaves.insert(
                path.to_owned(),
                GitTreeLeaf {
                    mode: mode.to_owned(),
                    object_type: object_type.to_owned(),
                    object_id: object_id.to_owned(),
                },
            );
        }
        Ok(leaves)
    }

    fn write_tree_from_leaves(
        &self,
        session: &Path,
        leaves: BTreeMap<String, GitTreeLeaf>,
    ) -> Result<String, CacheError> {
        let mut nodes = BTreeMap::<String, FlatGitTreeNode>::new();
        nodes.entry(String::new()).or_default();
        for (path, leaf) in leaves {
            let components = path.split('/').collect::<Vec<_>>();
            let (name, directories) = components.split_last().ok_or_else(|| {
                CacheError::InvalidManifest("Git tree leaf has an empty path".to_owned())
            })?;
            let mut parent = String::new();
            for directory in directories {
                if nodes
                    .get(&parent)
                    .is_some_and(|node| node.leaves.contains_key(*directory))
                {
                    return Err(CacheError::InvalidManifest(format!(
                        "Git tree path {path:?} conflicts with a file ancestor"
                    )));
                }
                let child = if parent.is_empty() {
                    (*directory).to_owned()
                } else {
                    format!("{parent}/{directory}")
                };
                nodes
                    .entry(parent.clone())
                    .or_default()
                    .children
                    .insert((*directory).to_owned(), child.clone());
                nodes.entry(child.clone()).or_default();
                parent = child;
            }
            if nodes
                .get(&parent)
                .is_some_and(|node| node.children.contains_key(*name))
            {
                return Err(CacheError::InvalidManifest(format!(
                    "Git tree path {path:?} conflicts with a directory"
                )));
            }
            nodes
                .entry(parent)
                .or_default()
                .leaves
                .insert((*name).to_owned(), leaf);
        }

        let maximum_depth = nodes
            .keys()
            .map(|path| {
                if path.is_empty() {
                    0
                } else {
                    path.bytes().filter(|byte| *byte == b'/').count() + 1
                }
            })
            .max()
            .unwrap_or(0);
        let mut tree_ids = BTreeMap::<String, String>::new();
        for depth in (0..=maximum_depth).rev() {
            let paths = nodes
                .keys()
                .filter(|path| {
                    let path_depth = if path.is_empty() {
                        0
                    } else {
                        path.bytes().filter(|byte| *byte == b'/').count() + 1
                    };
                    path_depth == depth
                })
                .cloned()
                .collect::<Vec<_>>();
            let mut trees = Vec::with_capacity(paths.len());
            for path in &paths {
                let node = nodes.get(path).ok_or_else(|| {
                    CacheError::InvalidTransition(
                        "flattened Git tree node disappeared during construction".to_owned(),
                    )
                })?;
                let mut entries = node.leaves.clone();
                for (name, child_path) in &node.children {
                    let object_id = tree_ids.get(child_path).ok_or_else(|| {
                        CacheError::InvalidTransition(format!(
                            "child Git tree {child_path:?} was not constructed before {path:?}"
                        ))
                    })?;
                    entries.insert(
                        name.clone(),
                        GitTreeLeaf {
                            mode: "040000".to_owned(),
                            object_type: "tree".to_owned(),
                            object_id: object_id.clone(),
                        },
                    );
                }
                trees.push(entries);
            }
            let object_ids = self.write_tree_batch(session, &trees)?;
            for (path, object_id) in paths.into_iter().zip(object_ids) {
                tree_ids.insert(path, object_id);
            }
        }
        tree_ids.remove("").ok_or_else(|| {
            CacheError::InvalidTransition("root Git tree was not constructed".to_owned())
        })
    }

    fn write_tree_batch(
        &self,
        session: &Path,
        trees: &[BTreeMap<String, GitTreeLeaf>],
    ) -> Result<Vec<String>, CacheError> {
        if trees.is_empty() {
            return Ok(Vec::new());
        }
        let mut input = Vec::new();
        for (tree_index, tree) in trees.iter().enumerate() {
            for (name, entry) in tree {
                input.extend_from_slice(
                    format!("{} {} {}\t", entry.mode, entry.object_type, entry.object_id)
                        .as_bytes(),
                );
                input.extend_from_slice(name.as_bytes());
                input.push(0);
            }
            if tree_index + 1 < trees.len() {
                input.push(0);
            }
        }
        let mut arguments = vec![
            OsString::from("mktree"),
            OsString::from("--missing"),
            OsString::from("-z"),
        ];
        if trees.len() > 1 {
            arguments.push(OsString::from("--batch"));
        }
        let output =
            run_git_with_input(&self.git_executable, Some(session), arguments, &[], &input)?;
        parse_object_id_lines(&output.stdout, "git mktree", trees.len())
    }

    fn commit_tree(
        &self,
        session: &Path,
        request: &RemoteCommitRequest,
    ) -> Result<String, CacheError> {
        let mut leaves = self.read_tree_leaves(session, &request.expected_head)?;
        for path in &request.delete_paths {
            validate_relative_git_path(path)?;
            leaves.remove(path);
        }
        let object_ids = self.hash_staged_blobs(session, &request.parts)?;
        for (part, object_id) in request.parts.iter().zip(object_ids) {
            leaves.insert(
                part.repository_path.clone(),
                GitTreeLeaf {
                    mode: "100644".to_owned(),
                    object_type: "blob".to_owned(),
                    object_id,
                },
            );
        }
        let tree = self.write_tree_from_leaves(session, leaves)?;
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
    }

    fn commit_root_tree(
        &self,
        session: &Path,
        request: &RemoteRefCreationRequest,
    ) -> Result<String, CacheError> {
        let index_name = format!(
            "publication-root-index-{}",
            ContentDigest::sha256(format!("{}:{}", request.branch, request.message).as_bytes())
        );
        let index_path = session.parent().unwrap_or(session).join(index_name);
        let index_environment = [(OsStr::new("GIT_INDEX_FILE"), index_path.as_os_str())];
        let result = (|| {
            run_git(
                &self.git_executable,
                Some(session),
                [OsString::from("read-tree"), OsString::from("--empty")],
                &index_environment,
            )?;
            let object_ids = self.hash_staged_blobs(session, &request.parts)?;
            self.update_index_parts(
                session,
                &request.parts,
                &object_ids,
                &[],
                &index_environment,
            )?;
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
        validate_relative_git_path(path)?;
        // Session initialization and revision fetch mutate the shared bare
        // repository and remain serialized. Once the commit is present, blob
        // reads are immutable and may proceed concurrently so split transport
        // parts do not download one after another.
        let session = {
            let _guard = self
                .operation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.fetch_revision(repository, revision)?
        };
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
        let object_id = parse_ls_tree_blob_object(&listing.stdout, path)?;
        self.ensure_blob_available(&session, repository, &object_id, None)?;
        let mut child = Command::new(&self.git_executable)
            .arg("-C")
            .arg(&session)
            .args(["cat-file", "blob"])
            .arg(&object_id)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| CacheError::Io(format!("failed to launch git cat-file: {error}")))?;
        let mut stdout = child.stdout.take().ok_or_else(|| {
            CacheError::Io("git cat-file did not provide a readable stream".to_owned())
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
                "git cat-file failed with status {status} for {path:?}"
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
        validate_relative_git_path(path)?;
        let session = {
            let _guard = self
                .operation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.fetch_revision(repository, revision)?
        };
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
        let object_id = parse_ls_tree_blob_object(&listing.stdout, path)?;
        self.ensure_blob_available(&session, repository, &object_id, Some(effective_maximum))?;
        let mut child = Command::new(&self.git_executable)
            .arg("-C")
            .arg(&session)
            .args(["cat-file", "blob"])
            .arg(&object_id)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| CacheError::Io(format!("failed to launch git cat-file: {error}")))?;
        let result = (|| {
            let mut stdout = child.stdout.take().ok_or_else(|| {
                CacheError::Io("git cat-file did not provide a readable stream".to_owned())
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
                "git cat-file failed with status {status} for {path:?}"
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

    fn create_ref_commit_if_absent(
        &self,
        request: &RemoteRefCreationRequest,
    ) -> Result<CreateRefResult, CacheError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        validate_repository(&request.repository)?;
        validate_branch(&request.branch)?;
        let batch_bytes = request.validate_limits()?;
        if self
            .resources
            .maximum_temporary_disk_bytes
            .is_some_and(|maximum| batch_bytes > maximum)
            || self
                .resources
                .maximum_transfer_bytes
                .is_some_and(|maximum| batch_bytes > maximum)
        {
            return Err(CacheError::ResourceLimit(
                "coordination ref creation exceeds the configured resource budget".to_owned(),
            ));
        }
        for part in &request.parts {
            validate_relative_git_path(&part.repository_path)?;
        }
        match self.read_ref(&request.repository, &request.branch) {
            Ok(current_head) => return Ok(CreateRefResult::RefExists { current_head }),
            Err(CacheError::NotFound(_)) => {}
            Err(error) => return Err(error),
        }
        let session = self.ensure_session(&request.repository)?;
        let commit_id = self.commit_root_tree(&session, request)?;
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
            Ok(CreateRefResult::Created { commit_id })
        } else {
            match self.read_ref(&request.repository, &request.branch) {
                Ok(current_head) => Ok(CreateRefResult::RefExists { current_head }),
                Err(CacheError::NotFound(_)) => Err(command_failure("git push", &push)),
                Err(error) => Err(error),
            }
        }
    }

    fn compare_and_swap_commits_atomically(
        &self,
        request: &AtomicRemoteCommitRequest,
    ) -> Result<AtomicCompareAndSwapResult, CacheError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        request.validate_limits()?;
        validate_repository(&request.repository)?;
        let transfer_bytes = request.commits.iter().try_fold(0u64, |total, commit| {
            total.checked_add(commit.validate_limits()?).ok_or_else(|| {
                CacheError::ResourceLimit("atomic publication transfer exceeds u64".to_owned())
            })
        })?;
        if self
            .resources
            .maximum_temporary_disk_bytes
            .is_some_and(|maximum| transfer_bytes > maximum)
        {
            return Err(CacheError::ResourceLimit(format!(
                "atomic publication {transfer_bytes} exceeds temporary-disk budget"
            )));
        }
        if self
            .resources
            .maximum_transfer_bytes
            .is_some_and(|maximum| transfer_bytes > maximum)
        {
            return Err(CacheError::ResourceLimit(format!(
                "atomic publication {transfer_bytes} exceeds transfer budget"
            )));
        }
        for commit in &request.commits {
            validate_branch(&commit.branch)?;
            validate_revision(&commit.expected_head)?;
            for part in &commit.parts {
                validate_relative_git_path(&part.repository_path)?;
            }
            for path in &commit.delete_paths {
                validate_relative_git_path(path)?;
            }
        }
        let mut current_heads = BTreeMap::new();
        for commit in &request.commits {
            let current = self.read_ref(&request.repository, &commit.branch)?;
            current_heads.insert(commit.branch.clone(), current);
        }
        if request.commits.iter().any(|commit| {
            current_heads
                .get(&commit.branch)
                .is_none_or(|head| head != &commit.expected_head)
        }) {
            return Ok(AtomicCompareAndSwapResult::RefConflict { current_heads });
        }
        let session = self.ensure_session(&request.repository)?;
        let mut commit_ids = BTreeMap::new();
        for commit in &request.commits {
            self.fetch_revision(&request.repository, &commit.expected_head)?;
            let commit_id = self.commit_tree(&session, commit)?;
            commit_ids.insert(commit.branch.clone(), commit_id);
        }
        let mut arguments = vec![
            OsString::from("push"),
            OsString::from("--atomic"),
            OsString::from("origin"),
        ];
        arguments.extend(request.commits.iter().map(|commit| {
            OsString::from(format!(
                "{}:refs/heads/{}",
                commit_ids[&commit.branch], commit.branch
            ))
        }));
        let push = run_git_allow_failure(&self.git_executable, Some(&session), arguments, &[])?;
        if push.status.success() {
            return Ok(AtomicCompareAndSwapResult::Committed { commit_ids });
        }
        let mut observed = BTreeMap::new();
        for commit in &request.commits {
            observed.insert(
                commit.branch.clone(),
                self.read_ref(&request.repository, &commit.branch)?,
            );
        }
        if request.commits.iter().any(|commit| {
            observed
                .get(&commit.branch)
                .is_none_or(|head| head != &commit.expected_head)
        }) {
            Ok(AtomicCompareAndSwapResult::RefConflict {
                current_heads: observed,
            })
        } else {
            Err(command_failure("git push --atomic", &push))
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

fn parse_object_id_lines(
    bytes: &[u8],
    operation: &str,
    expected_count: usize,
) -> Result<Vec<String>, CacheError> {
    let output = std::str::from_utf8(bytes)
        .map_err(|error| CacheError::Io(format!("{operation} returned non-UTF8: {error}")))?;
    let values = output
        .lines()
        .map(|line| {
            let value = line.trim();
            if value.is_empty() || value.chars().any(char::is_whitespace) {
                return Err(CacheError::Io(format!(
                    "{operation} returned an invalid object identity"
                )));
            }
            Ok(value.to_owned())
        })
        .collect::<Result<Vec<_>, CacheError>>()?;
    if values.len() != expected_count {
        return Err(CacheError::Io(format!(
            "{operation} returned {} object identities for {expected_count} paths",
            values.len()
        )));
    }
    Ok(values)
}

fn parse_ls_tree_blob_object(bytes: &[u8], expected_path: &str) -> Result<String, CacheError> {
    let output = std::str::from_utf8(bytes)
        .map_err(|error| CacheError::Io(format!("git ls-tree returned non-UTF8: {error}")))?;
    let mut lines = output.lines();
    let line = lines.next().ok_or_else(|| {
        CacheError::NotFound(format!("git ls-tree did not return {expected_path:?}"))
    })?;
    if lines.next().is_some() {
        return Err(CacheError::InvalidManifest(format!(
            "git ls-tree returned multiple entries for exact path {expected_path:?}"
        )));
    }
    let (identity, path) = line.split_once('\t').ok_or_else(|| {
        CacheError::InvalidManifest("git ls-tree entry has no path separator".to_owned())
    })?;
    if path != expected_path {
        return Err(CacheError::InvalidManifest(format!(
            "git ls-tree returned path {path:?} instead of {expected_path:?}"
        )));
    }
    let fields = identity.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 3 || fields[1] != "blob" {
        return Err(CacheError::InvalidManifest(format!(
            "git ls-tree path {expected_path:?} is not a regular blob"
        )));
    }
    validate_revision(fields[2])?;
    Ok(fields[2].to_owned())
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

fn directory_size_bytes(path: &Path) -> Result<u64, CacheError> {
    if !path.exists() {
        return Ok(0);
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Ok(0);
    }
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        total = total
            .checked_add(directory_size_bytes(&entry.path())?)
            .ok_or_else(|| {
                CacheError::ResourceLimit("Git transport directory size exceeds u64".to_owned())
            })?;
    }
    Ok(total)
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

    fn test_git_stdout(directory: Option<&Path>, arguments: &[&str]) -> String {
        let mut command = Command::new("git");
        if let Some(directory) = directory {
            command.arg("-C").arg(directory);
        }
        let output = command.args(arguments).output().unwrap();
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            arguments,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    fn local_file_url(path: &Path) -> String {
        let canonical = fs::canonicalize(path)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let canonical = canonical.strip_prefix("//?/").unwrap_or(&canonical);
        if cfg!(windows) {
            format!("file:///{canonical}")
        } else {
            format!("file://{canonical}")
        }
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
    fn filtered_transport_does_not_hydrate_unchanged_historical_blobs() {
        if !test_git(None, &["--version"]) {
            return;
        }
        let root = temporary_root("git-cli-no-historical-hydration");
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
        assert!(test_git(
            Some(&remote),
            &["config", "uploadpack.allowFilter", "true"]
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

        // Use deterministic high-entropy bytes so Git cannot compress this
        // into a deceptively small pack if the blob is accidentally hydrated.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut historical = vec![0u8; 8 * 1024 * 1024];
        for byte in &mut historical {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        fs::write(seed.join("historical-large.bin"), &historical).unwrap();
        fs::write(seed.join("README.md"), b"seed\n").unwrap();
        assert!(test_git(Some(&seed), &["add", "."]));
        assert!(test_git(Some(&seed), &["commit", "-m", "seed"]));
        let historical_blob =
            test_git_stdout(Some(&seed), &["rev-parse", "HEAD:historical-large.bin"]);
        assert!(test_git(
            Some(&seed),
            &["remote", "add", "origin", remote.to_str().unwrap()]
        ));
        assert!(test_git(
            Some(&seed),
            &["push", "origin", "HEAD:refs/heads/main"]
        ));

        let payload = b"new small cache payload";
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
        let repository = local_file_url(&remote);
        let expected_head = store.read_ref(&repository, "main").unwrap();
        let result = store
            .compare_and_swap_commit(&RemoteCommitRequest {
                repository: repository.clone(),
                branch: "main".to_owned(),
                expected_head,
                message: "publish without hydrating history".to_owned(),
                parts: vec![part],
                delete_paths: Vec::new(),
            })
            .unwrap();
        let CompareAndSwapResult::Committed { commit_id } = result else {
            panic!("local test publication unexpectedly conflicted");
        };

        let session = store.session_path(&repository);
        let mut readme = Vec::new();
        store
            .read_committed_path(
                &repository,
                &commit_id,
                "README.md",
                1024,
                &CancellationToken::new(),
                &mut readme,
            )
            .unwrap();
        assert_eq!(readme, b"seed\n");
        assert!(
            !test_git(
                Some(&session),
                &["cat-file", "-e", &format!("{historical_blob}^{{blob}}")]
            ),
            "unchanged historical payload was hydrated into the transport workspace"
        );
        assert!(
            !test_git(
                Some(&session),
                &["config", "--get", "remote.origin.promisor"]
            ),
            "transport session was incorrectly configured for implicit lazy fetch"
        );
        let tree = test_git_stdout(
            Some(&session),
            &["ls-tree", "-r", "--name-only", &commit_id],
        );
        assert!(tree.lines().any(|path| path == "historical-large.bin"));
        assert!(tree.lines().any(|path| path.ends_with(".part")));

        store.cleanup_session(&repository).unwrap();
        assert!(!session.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prepared_blobs_are_not_reread_during_commit_construction() {
        if !test_git(None, &["--version"]) {
            return;
        }
        let root = temporary_root("git-cli-prepared-blobs");
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

        let payloads = [
            b"payload prepared before publication lock".as_slice(),
            b"second payload in the same hash-object batch".as_slice(),
            b"third payload in the same index-info stream".as_slice(),
        ];
        let mut parts = Vec::new();
        let mut local_paths = Vec::new();
        for (sequence, payload) in payloads.iter().enumerate() {
            let digest = ContentDigest::sha256(payload);
            let repository_path = format!("objects/sha256/{}/{}.part", &digest.0[..2], digest.0);
            let local_path = staging.join(Path::new(&repository_path));
            fs::create_dir_all(local_path.parent().unwrap()).unwrap();
            fs::write(&local_path, payload).unwrap();
            parts.push(TransportPart {
                sequence: sequence as u64,
                repository_path,
                size_bytes: payload.len() as u64,
                content_digest: digest,
            });
            local_paths.push(local_path);
        }
        let store = GitCliRemoteStore::new(
            &temporary,
            &staging,
            "Test Publisher",
            "test@example.invalid",
        )
        .unwrap();
        let repository = remote.to_string_lossy().to_string();

        store.prepare_staged_parts(&repository, &parts).unwrap();
        for local_path in &local_paths {
            fs::remove_file(local_path).unwrap();
        }

        let expected_head = store.read_ref(&repository, "main").unwrap();
        let result = store
            .compare_and_swap_commit(&RemoteCommitRequest {
                repository: repository.clone(),
                branch: "main".to_owned(),
                expected_head,
                message: "publish prehashed part".to_owned(),
                parts: parts.clone(),
                delete_paths: Vec::new(),
            })
            .unwrap();
        let CompareAndSwapResult::Committed { commit_id } = result else {
            panic!("local test publication unexpectedly conflicted");
        };
        for part in &parts {
            store
                .verify_committed_part(&repository, &commit_id, part)
                .unwrap();
        }
        store.cleanup_session(&repository).unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn private_batch_and_coordination_ref_advance_atomically() {
        if !test_git(None, &["--version"]) {
            return;
        }
        let root = temporary_root("git-cli-atomic-private-publication");
        let _ = fs::remove_dir_all(&root);
        let remote = root.join("remote.git");
        let seed = root.join("seed");
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

        let repository = remote.to_string_lossy().to_string();
        let store = GitCliRemoteStore::new(
            root.join("transport"),
            &staging,
            "Test Publisher",
            "test@example.invalid",
        )
        .unwrap();
        let stage = |path: &str, bytes: &[u8]| {
            let target = path
                .split('/')
                .fold(staging.clone(), |current, part| current.join(part));
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::write(target, bytes).unwrap();
            TransportPart {
                sequence: 0,
                repository_path: path.to_owned(),
                size_bytes: bytes.len() as u64,
                content_digest: ContentDigest::sha256(bytes),
            }
        };
        let state = stage("coordination/state.json", b"{\"generation\":1}\n");
        let lock = stage("coordination/publication-lock.json", b"{\"owner\":\"a\"}\n");
        let created = store
            .create_ref_commit_if_absent(&RemoteRefCreationRequest {
                repository: repository.clone(),
                branch: "xcelerator-coordination".to_owned(),
                message: "initialize coordination".to_owned(),
                parts: vec![state, lock],
            })
            .unwrap();
        let CreateRefResult::Created {
            commit_id: coordination_head,
        } = created
        else {
            panic!("coordination branch unexpectedly existed");
        };
        assert!(matches!(
            store
                .create_ref_commit_if_absent(&RemoteRefCreationRequest {
                    repository: repository.clone(),
                    branch: "xcelerator-coordination".to_owned(),
                    message: "competing initialization".to_owned(),
                    parts: vec![stage("coordination/state.json", b"other\n")],
                })
                .unwrap(),
            CreateRefResult::RefExists { .. }
        ));

        let main_head = store.read_ref(&repository, "main").unwrap();
        let payload = stage("objects/test.part", b"private payload");
        let renewed_lock = stage(
            "coordination/publication-lock.json",
            b"{\"owner\":\"a\",\"heartbeat\":2}\n",
        );
        let atomic = store
            .compare_and_swap_commits_atomically(&AtomicRemoteCommitRequest {
                repository: repository.clone(),
                commits: vec![
                    RemoteCommitRequest {
                        repository: repository.clone(),
                        branch: "main".to_owned(),
                        expected_head: main_head.clone(),
                        message: "publish private batch".to_owned(),
                        parts: vec![payload.clone()],
                        delete_paths: Vec::new(),
                    },
                    RemoteCommitRequest {
                        repository: repository.clone(),
                        branch: "xcelerator-coordination".to_owned(),
                        expected_head: coordination_head,
                        message: "renew private lease".to_owned(),
                        parts: vec![renewed_lock],
                        delete_paths: Vec::new(),
                    },
                ],
            })
            .unwrap();
        let AtomicCompareAndSwapResult::Committed { commit_ids } = atomic else {
            panic!("atomic private publication unexpectedly conflicted");
        };
        let new_main = commit_ids["main"].clone();
        let new_coordination = commit_ids["xcelerator-coordination"].clone();
        store
            .verify_committed_part(&repository, &new_main, &payload)
            .unwrap();

        let contender_lock = stage(
            "coordination/publication-lock.json",
            b"{\"owner\":\"b\",\"generation\":2}\n",
        );
        let takeover = store
            .compare_and_swap_commit(&RemoteCommitRequest {
                repository: repository.clone(),
                branch: "xcelerator-coordination".to_owned(),
                expected_head: new_coordination.clone(),
                message: "take over expired lease".to_owned(),
                parts: vec![contender_lock],
                delete_paths: Vec::new(),
            })
            .unwrap();
        assert!(matches!(takeover, CompareAndSwapResult::Committed { .. }));
        let late_payload = stage("objects/late.part", b"stale writer");
        let stale_renewal = stage(
            "coordination/publication-lock.json",
            b"{\"owner\":\"a\",\"heartbeat\":3}\n",
        );
        let stale = store
            .compare_and_swap_commits_atomically(&AtomicRemoteCommitRequest {
                repository: repository.clone(),
                commits: vec![
                    RemoteCommitRequest {
                        repository: repository.clone(),
                        branch: "main".to_owned(),
                        expected_head: new_main.clone(),
                        message: "stale private batch".to_owned(),
                        parts: vec![late_payload],
                        delete_paths: Vec::new(),
                    },
                    RemoteCommitRequest {
                        repository: repository.clone(),
                        branch: "xcelerator-coordination".to_owned(),
                        expected_head: new_coordination,
                        message: "stale lease renewal".to_owned(),
                        parts: vec![stale_renewal],
                        delete_paths: Vec::new(),
                    },
                ],
            })
            .unwrap();
        assert!(matches!(
            stale,
            AtomicCompareAndSwapResult::RefConflict { .. }
        ));
        assert_eq!(store.read_ref(&repository, "main").unwrap(), new_main);
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
