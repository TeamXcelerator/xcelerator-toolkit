//! Consistent, short-lived read views. Mutation always uses the original store.
use super::*;
use std::sync::Mutex;

pub(super) struct ReadSnapshot<'a> {
    remote: &'a dyn RemoteGitStore,
    heads: Mutex<BTreeMap<(String, String), String>>,
}

impl<'a> ReadSnapshot<'a> {
    pub(super) fn new(remote: &'a dyn RemoteGitStore) -> Self {
        Self {
            remote,
            heads: Mutex::new(BTreeMap::new()),
        }
    }

    pub(super) fn with_head(
        remote: &'a dyn RemoteGitStore,
        repository: &str,
        revision: &str,
    ) -> Self {
        Self {
            remote,
            heads: Mutex::new(BTreeMap::from([(
                (repository.into(), "main".into()),
                revision.into(),
            )])),
        }
    }

    pub(super) fn observed_heads(&self) -> BTreeMap<(String, String), String> {
        self.heads.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl RemoteGitStore for ReadSnapshot<'_> {
    fn read_ref(&self, repository: &str, branch: &str) -> Result<String, CacheError> {
        let mut heads = self.heads.lock().unwrap_or_else(|e| e.into_inner());
        let key = (repository.into(), branch.into());
        if let Some(head) = heads.get(&key) {
            return Ok(head.clone());
        }
        let head = self.remote.read_ref(repository, branch)?;
        heads.insert(key, head.clone());
        Ok(head)
    }
    fn immutable_path_digest(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
    ) -> Result<Option<ContentDigest>, CacheError> {
        self.remote
            .immutable_path_digest(repository, revision, path)
    }
    fn read_committed_path(
        &self,
        repository: &str,
        revision: &str,
        path: &str,
        maximum_bytes: u64,
        cancellation: &xc_core::CancellationToken,
        writer: &mut dyn std::io::Write,
    ) -> Result<crate::RemoteReadReport, CacheError> {
        self.remote.read_committed_path(
            repository,
            revision,
            path,
            maximum_bytes,
            cancellation,
            writer,
        )
    }
    fn prefetch_committed_paths(
        &self,
        repository: &str,
        revision: &str,
        paths: &[crate::RemotePathPrefetch],
        cancellation: &xc_core::CancellationToken,
    ) -> Result<(), CacheError> {
        self.remote
            .prefetch_committed_paths(repository, revision, paths, cancellation)
    }
    fn prefetch_present_committed_paths(
        &self,
        repository: &str,
        revision: &str,
        paths: &[crate::RemotePathPrefetch],
        cancellation: &xc_core::CancellationToken,
    ) -> Result<(), CacheError> {
        self.remote
            .prefetch_present_committed_paths(repository, revision, paths, cancellation)
    }
    fn list_committed_paths(
        &self,
        repository: &str,
        revision: &str,
        prefix: &str,
        maximum_paths: u64,
        maximum_total_path_bytes: u64,
        cancellation: &xc_core::CancellationToken,
    ) -> Result<crate::RemotePathListReport, CacheError> {
        self.remote.list_committed_paths(
            repository,
            revision,
            prefix,
            maximum_paths,
            maximum_total_path_bytes,
            cancellation,
        )
    }
    fn compare_and_swap_commit(
        &self,
        _: &crate::RemoteCommitRequest,
    ) -> Result<crate::CompareAndSwapResult, CacheError> {
        Err(CacheError::ReadOnlyLayer(
            "publication read snapshots cannot mutate repositories".into(),
        ))
    }
    fn verify_committed_part(
        &self,
        repository: &str,
        revision: &str,
        part: &TransportPart,
    ) -> Result<(), CacheError> {
        self.remote
            .verify_committed_part(repository, revision, part)
    }
}
