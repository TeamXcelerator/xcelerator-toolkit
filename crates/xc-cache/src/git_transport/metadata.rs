//! Bounded immutable metadata reuse and batched local Git blob reads.
use super::*;

impl GitCliRemoteStore {
    fn metadata_budget(&self) -> u64 {
        (self
            .resources
            .maximum_memory_bytes
            .unwrap_or(256 * 1024 * 1024)
            / 8)
        .min(32 * 1024 * 1024)
    }

    pub(super) fn remember_metadata_absence(&self, repository: &str, revision: &str, path: &str) {
        if !is_cache_metadata_path(path) || self.metadata_budget() < 64 * 1024 {
            return;
        }
        let mut memo = self
            .metadata_reads
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if memo.entries.len() + memo.absent.len() >= 4096 {
            *memo = MetadataReadMemo::default();
        }
        memo.absent
            .insert((repository.into(), revision.into(), path.into()));
    }

    pub(super) fn metadata_prefetch_is_cached(
        &self,
        repository: &str,
        revision: &str,
        paths: &BTreeMap<String, u64>,
        allow_missing: bool,
    ) -> Result<bool, CacheError> {
        self.require_usable_session(&self.session_path(repository))?;
        let memo = self
            .metadata_reads
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut unique = BTreeMap::new();
        for (path, maximum) in paths {
            let key = (repository.into(), revision.into(), path.clone());
            if let Some((bytes, digest)) = memo.entries.get(&key) {
                if bytes.len() as u64 > *maximum || self.resources.maximum_memory_bytes == Some(0) {
                    return Err(CacheError::ResourceLimit(
                        "cached prefetched metadata exceeds its read limit".into(),
                    ));
                }
                unique.insert(digest, bytes.len() as u64);
            } else if memo.absent.contains(&key) {
                if !allow_missing {
                    return Err(CacheError::NotFound(format!(
                        "remote path {path:?} at {revision}"
                    )));
                }
            } else {
                return Ok(false);
            }
        }
        let total = unique
            .values()
            .try_fold(0u64, |sum, n| sum.checked_add(*n))
            .ok_or_else(|| CacheError::ResourceLimit("cached prefetch size overflows".into()))?;
        if self
            .resources
            .maximum_transfer_bytes
            .is_some_and(|max| total > max)
        {
            return Err(CacheError::ResourceLimit(
                "cached prefetch exceeds transfer budget".into(),
            ));
        }
        Ok(true)
    }

    pub(super) fn prefetch_metadata_bytes(
        &self,
        session: &Path,
        repository: &str,
        revision: &str,
        paths: &BTreeMap<String, u64>,
        absent: &BTreeSet<String>,
        cancellation: &CancellationToken,
    ) -> Result<(), CacheError> {
        let budget = self.metadata_budget();
        if budget < 64 * 1024 {
            return Ok(());
        }
        let mut pending = Vec::new();
        let mut bytes = 0u64;
        for path in paths
            .keys()
            .filter(|p| is_cache_metadata_path(p) && !absent.contains(*p))
        {
            let key = (repository.into(), revision.into(), path.clone());
            if self
                .metadata_reads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entries
                .contains_key(&key)
            {
                continue;
            }
            let Some(object) = self.cached_blob_object(repository, revision, path) else {
                continue;
            };
            let Some(size) = object.size_bytes.filter(|s| *s <= budget.min(1024 * 1024)) else {
                continue;
            };
            if !pending.is_empty()
                && (pending.len() >= 256 || bytes + size > budget.min(4 * 1024 * 1024))
            {
                self.read_metadata_batch(session, repository, revision, &pending, cancellation)?;
                pending.clear();
                bytes = 0;
            }
            bytes += size;
            pending.push((path.clone(), object.object_id, size));
        }
        if !pending.is_empty() {
            self.read_metadata_batch(session, repository, revision, &pending, cancellation)?;
        }
        Ok(())
    }

    fn read_metadata_batch(
        &self,
        session: &Path,
        repository: &str,
        revision: &str,
        objects: &[(String, String, u64)],
        cancellation: &CancellationToken,
    ) -> Result<(), CacheError> {
        let started = Instant::now();
        let mut child = git_command(&self.git_executable)
            .arg("--no-replace-objects")
            .arg("-C")
            .arg(session)
            .arg("--git-dir=.")
            .args(["cat-file", "--batch"])
            .env("GIT_NO_LAZY_FETCH", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| CacheError::Io(format!("failed to launch metadata batch reader: {e}")))?;
        let result = (|| {
            let mut input = child
                .stdin
                .take()
                .ok_or_else(|| CacheError::Io("metadata batch has no stdin".into()))?;
            let mut output = BufReader::new(
                child
                    .stdout
                    .take()
                    .ok_or_else(|| CacheError::Io("metadata batch has no stdout".into()))?,
            );
            let mut documents = Vec::new();
            for (path, object, size) in objects {
                cancellation
                    .check()
                    .map_err(|e| CacheError::Cancelled(e.to_string()))?;
                validate_revision(object)?;
                writeln!(input, "{object}")?;
                input.flush()?;
                let mut header = Vec::new();
                output.by_ref().take(256).read_until(b'\n', &mut header)?;
                let header = std::str::from_utf8(&header).map_err(|_| {
                    CacheError::InvalidManifest("metadata batch header is not UTF-8".into())
                })?;
                let expected = format!("{object} blob {size}\n");
                if header != expected {
                    return Err(CacheError::InvalidManifest(
                        "metadata batch identity or size mismatch".into(),
                    ));
                }
                let mut bytes = vec![0u8; *size as usize];
                for chunk in bytes.chunks_mut(64 * 1024) {
                    cancellation
                        .check()
                        .map_err(|e| CacheError::Cancelled(e.to_string()))?;
                    output.read_exact(chunk)?;
                }
                let mut newline = [0];
                output.read_exact(&mut newline)?;
                if newline != *b"\n" {
                    return Err(CacheError::InvalidManifest(
                        "metadata batch has no blob terminator".into(),
                    ));
                }
                let digest = ContentDigest::sha256(&bytes);
                documents.push((path.clone(), bytes, digest));
            }
            Ok::<_, CacheError>(documents)
        })();
        if result.is_err() {
            let _ = child.kill();
        }
        let status = child.wait()?;
        let documents = result?;
        if !status.success() {
            return Err(CacheError::Io(format!(
                "metadata batch reader failed: {status}"
            )));
        }
        let mut memo = self
            .metadata_reads
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (path, bytes, digest) in documents {
            if memo.entries.len() + memo.absent.len() >= 4096
                || memo.bytes.saturating_add(bytes.len() as u64) > self.metadata_budget()
            {
                *memo = MetadataReadMemo::default();
            }
            let key = (repository.into(), revision.into(), path);
            memo.absent.remove(&key);
            let size = bytes.len() as u64;
            if let Some((old, _)) = memo.entries.insert(key, (Arc::new(bytes), digest)) {
                memo.bytes = memo.bytes.saturating_sub(old.len() as u64);
            }
            memo.bytes += size;
        }
        drop(memo);
        self.publication_event(
            "metadata_batch_read",
            started.elapsed(),
            serde_json::json!({"documents":objects.len(),"success":true}),
        );
        Ok(())
    }
}

pub(super) fn local_blob_sizes(
    git: &OsStr,
    session: &Path,
    objects: &[String],
) -> Result<BTreeMap<String, u64>, CacheError> {
    let mut sizes = BTreeMap::new();
    for batch in objects.chunks(4096) {
        let mut input = Vec::new();
        for object in batch {
            validate_revision(object)?;
            input.extend_from_slice(object.as_bytes());
            input.push(b'\n');
        }
        let output = run_git_with_input(
            git,
            Some(session),
            [OsString::from("cat-file"), OsString::from("--batch-check")],
            &[(OsStr::new("GIT_NO_LAZY_FETCH"), OsStr::new("1"))],
            &input,
        )?;
        let text = std::str::from_utf8(&output.stdout)
            .map_err(|_| CacheError::InvalidManifest("blob-size batch is not UTF-8".into()))?;
        let lines = text.lines().collect::<Vec<_>>();
        if lines.len() != batch.len() {
            return Err(CacheError::InvalidManifest(
                "blob-size batch has an unexpected row count".into(),
            ));
        }
        for (line, object) in lines.iter().zip(batch) {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() == 2 && fields[0] == object && fields[1] == "missing" {
                continue;
            }
            if fields.len() != 3 || fields[0] != object || fields[1] != "blob" {
                return Err(CacheError::InvalidManifest(
                    "blob-size batch identity mismatch".into(),
                ));
            }
            let size = fields[2]
                .parse()
                .map_err(|_| CacheError::InvalidManifest("invalid blob size".into()))?;
            sizes.insert(object.clone(), size);
        }
    }
    Ok(sizes)
}
