//! Private-shard publication leases carried by an isolated repository branch.
//!
//! The coordination branch is deliberately independent of the cache branch.
//! Every cache batch renews the lease in the same atomic Git push, so a stale
//! publisher cannot advance `main` after another publisher takes over.

use crate::{
    AtomicCompareAndSwapResult, AtomicRemoteCommitRequest, CacheError, CompareAndSwapResult,
    ContentDigest, CreateRefResult, RemoteCommitRequest, RemoteGitStore, RemoteRefCreationRequest,
    RemoteShardReader, TransportPart,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::Path;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use xc_core::CancellationToken;

pub const PRIVATE_COORDINATION_BRANCH: &str = "xcelerator-coordination";
pub const PRIVATE_COORDINATION_STATE_PATH: &str = "coordination/state.json";
pub const PRIVATE_PUBLICATION_LOCK_PATH: &str = "coordination/publication-lock.json";
const MAXIMUM_COORDINATION_DOCUMENT_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrivatePublicationCoordinationState {
    pub schema_version: u32,
    pub fencing_generation: u64,
    pub last_completed_transaction: Option<String>,
    pub last_released_at_unix_seconds: Option<u64>,
}

impl PrivatePublicationCoordinationState {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1 || self.fencing_generation == 0 {
            return Err(CacheError::InvalidManifest(
                "private publication coordination state is invalid".to_owned(),
            ));
        }
        if self
            .last_completed_transaction
            .as_ref()
            .is_some_and(|transaction| transaction.trim().is_empty())
        {
            return Err(CacheError::InvalidManifest(
                "private coordination completed transaction is empty".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrivatePublicationLock {
    pub schema_version: u32,
    pub owner_run_id: String,
    pub publication_transaction_id: String,
    pub github_principal: String,
    pub toolkit_version: String,
    pub instance_fingerprint: ContentDigest,
    pub process_id: u32,
    pub fencing_generation: u64,
    pub observed_main_head: String,
    pub acquired_at_unix_seconds: u64,
    pub heartbeat_at_unix_seconds: u64,
    pub lease_expires_at_unix_seconds: u64,
}

impl PrivatePublicationLock {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.schema_version != 1
            || self.owner_run_id.trim().is_empty()
            || self.publication_transaction_id.trim().is_empty()
            || self.github_principal.trim().is_empty()
            || self.toolkit_version.trim().is_empty()
            || !self.instance_fingerprint.validate()
            || self.process_id == 0
            || self.fencing_generation == 0
            || !valid_revision(&self.observed_main_head)
            || self.acquired_at_unix_seconds == 0
            || self.heartbeat_at_unix_seconds < self.acquired_at_unix_seconds
            || self.lease_expires_at_unix_seconds <= self.heartbeat_at_unix_seconds
        {
            return Err(CacheError::InvalidManifest(
                "private publication lock record is invalid".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn expired_at(&self, now_unix_seconds: u64, clock_skew_grace_seconds: u64) -> bool {
        now_unix_seconds
            >= self
                .lease_expires_at_unix_seconds
                .saturating_add(clock_skew_grace_seconds)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivatePublicationLeasePolicy {
    pub lease_seconds: u64,
    pub clock_skew_grace_seconds: u64,
    pub initial_poll_seconds: u64,
    pub maximum_poll_seconds: u64,
    pub maximum_wait_seconds: u64,
}

impl Default for PrivatePublicationLeasePolicy {
    fn default() -> Self {
        Self {
            // One large Git push can be slow on rented compute. Batch pushes
            // renew this lease atomically; two hours prevents routine uploads
            // from becoming accidental takeovers.
            lease_seconds: 2 * 60 * 60,
            clock_skew_grace_seconds: 60,
            initial_poll_seconds: 5,
            maximum_poll_seconds: 60,
            maximum_wait_seconds: 24 * 60 * 60,
        }
    }
}

impl PrivatePublicationLeasePolicy {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.lease_seconds < 60
            || self.clock_skew_grace_seconds > self.lease_seconds
            || self.initial_poll_seconds == 0
            || self.maximum_poll_seconds < self.initial_poll_seconds
            || self.maximum_wait_seconds < self.initial_poll_seconds
        {
            return Err(CacheError::InvalidManifest(
                "private publication lease policy is invalid".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivatePublicationLeaseOwner {
    pub owner_run_id: String,
    pub github_principal: String,
    pub instance_fingerprint: ContentDigest,
    pub process_id: u32,
}

impl PrivatePublicationLeaseOwner {
    pub fn for_current_process(
        repository: &str,
        github_principal: &str,
        event_unix_seconds: u64,
    ) -> Result<Self, CacheError> {
        if repository.trim().is_empty() || github_principal.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "private publication lease owner is incomplete".to_owned(),
            ));
        }
        let process_id = std::process::id();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| CacheError::InvalidTransition(error.to_string()))?
            .as_nanos();
        let instance = std::env::var("XC_PUBLICATION_INSTANCE_LABEL")
            .or_else(|_| std::env::var("HOSTNAME"))
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unlabelled-instance".to_owned());
        let instance_fingerprint =
            ContentDigest::sha256(format!("{repository}\0{instance}").as_bytes());
        let owner_run_id = ContentDigest::sha256(
            format!(
                "{repository}\0{github_principal}\0{process_id}\0{event_unix_seconds}\0{nonce}"
            )
            .as_bytes(),
        )
        .0;
        Ok(Self {
            owner_run_id,
            github_principal: github_principal.to_owned(),
            instance_fingerprint,
            process_id,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivatePublicationLease {
    pub repository: String,
    pub main_branch: String,
    pub coordination_branch: String,
    pub coordination_head: String,
    pub state: PrivatePublicationCoordinationState,
    pub lock: PrivatePublicationLock,
}

impl PrivatePublicationLease {
    pub fn set_transaction_id(&mut self, transaction_id: impl Into<String>) {
        self.lock.publication_transaction_id = transaction_id.into();
    }
}

fn valid_revision(revision: &str) -> bool {
    matches!(revision.len(), 40 | 64) && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn now_unix_seconds() -> Result<u64, CacheError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| CacheError::InvalidTransition(error.to_string()))?
        .as_secs()
        .max(1))
}

fn stage_json<T: Serialize>(
    staging_root: &Path,
    repository_path: &str,
    value: &T,
) -> Result<TransportPart, CacheError> {
    let bytes = crate::protocol::canonical_json_bytes(value)?;
    let target = repository_path
        .split('/')
        .fold(staging_root.to_path_buf(), |path, component| {
            path.join(component)
        });
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&target, &bytes)?;
    Ok(TransportPart {
        sequence: 0,
        repository_path: repository_path.to_owned(),
        size_bytes: bytes.len() as u64,
        content_digest: ContentDigest::sha256(&bytes),
    })
}

fn read_json<T: DeserializeOwned>(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    path: &str,
    cancellation: &CancellationToken,
) -> Result<T, CacheError> {
    Ok(
        RemoteShardReader::new(remote, MAXIMUM_COORDINATION_DOCUMENT_BYTES)?
            .read_json::<T>(repository, revision, path, cancellation)?
            .value,
    )
}

fn read_optional_json<T: DeserializeOwned>(
    remote: &dyn RemoteGitStore,
    repository: &str,
    revision: &str,
    path: &str,
    cancellation: &CancellationToken,
) -> Result<Option<T>, CacheError> {
    if remote
        .immutable_path_digest(repository, revision, path)?
        .is_none()
    {
        return Ok(None);
    }
    read_json(remote, repository, revision, path, cancellation).map(Some)
}

fn lock_record(
    owner: &PrivatePublicationLeaseOwner,
    transaction_id: &str,
    generation: u64,
    main_head: &str,
    acquired_at: u64,
    now: u64,
    policy: &PrivatePublicationLeasePolicy,
) -> Result<PrivatePublicationLock, CacheError> {
    let record = PrivatePublicationLock {
        schema_version: 1,
        owner_run_id: owner.owner_run_id.clone(),
        publication_transaction_id: transaction_id.to_owned(),
        github_principal: owner.github_principal.clone(),
        toolkit_version: env!("CARGO_PKG_VERSION").to_owned(),
        instance_fingerprint: owner.instance_fingerprint.clone(),
        process_id: owner.process_id,
        fencing_generation: generation,
        observed_main_head: main_head.to_owned(),
        acquired_at_unix_seconds: acquired_at,
        heartbeat_at_unix_seconds: now,
        lease_expires_at_unix_seconds: now.saturating_add(policy.lease_seconds),
    };
    record.validate()?;
    Ok(record)
}

fn sleep_with_cancellation(
    seconds: u64,
    cancellation: &CancellationToken,
) -> Result<(), CacheError> {
    for _ in 0..seconds {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}

fn jittered_poll_seconds(
    base_seconds: u64,
    maximum_seconds: u64,
    owner_run_id: &str,
    generation: u64,
) -> u64 {
    let width = (base_seconds / 4).max(1);
    let seed = owner_run_id.bytes().fold(generation, |value, byte| {
        value.wrapping_mul(33) ^ byte as u64
    });
    base_seconds
        .saturating_add(seed % width.saturating_add(1))
        .min(maximum_seconds)
}

#[allow(clippy::too_many_arguments)]
pub fn acquire_private_publication_lease(
    remote: &dyn RemoteGitStore,
    repository: &str,
    main_branch: &str,
    owner: &PrivatePublicationLeaseOwner,
    transaction_hint: &str,
    staging_root: &Path,
    cancellation: &CancellationToken,
    policy: &PrivatePublicationLeasePolicy,
) -> Result<PrivatePublicationLease, CacheError> {
    policy.validate()?;
    if repository.trim().is_empty()
        || main_branch.trim().is_empty()
        || transaction_hint.trim().is_empty()
    {
        return Err(CacheError::InvalidManifest(
            "private publication lease target is incomplete".to_owned(),
        ));
    }
    let started = now_unix_seconds()?;
    let mut poll_seconds = policy.initial_poll_seconds;
    let mut last_status = 0u64;
    loop {
        cancellation
            .check()
            .map_err(|error| CacheError::Cancelled(error.to_string()))?;
        let main_head = remote.read_ref(repository, main_branch)?;
        let now = now_unix_seconds()?;
        match remote.read_ref(repository, PRIVATE_COORDINATION_BRANCH) {
            Err(CacheError::NotFound(_)) => {
                let state = PrivatePublicationCoordinationState {
                    schema_version: 1,
                    fencing_generation: 1,
                    last_completed_transaction: None,
                    last_released_at_unix_seconds: None,
                };
                let lock = lock_record(
                    owner,
                    transaction_hint,
                    state.fencing_generation,
                    &main_head,
                    now,
                    now,
                    policy,
                )?;
                let parts = vec![
                    stage_json(staging_root, PRIVATE_COORDINATION_STATE_PATH, &state)?,
                    stage_json(staging_root, PRIVATE_PUBLICATION_LOCK_PATH, &lock)?,
                ];
                match remote.create_ref_commit_if_absent(&RemoteRefCreationRequest {
                    repository: repository.to_owned(),
                    branch: PRIVATE_COORDINATION_BRANCH.to_owned(),
                    message: "initialize private publication coordination lease".to_owned(),
                    parts,
                })? {
                    CreateRefResult::Created { commit_id } => {
                        eprintln!(
                            "private publication lock acquired: generation=1 repository={repository}"
                        );
                        return Ok(PrivatePublicationLease {
                            repository: repository.to_owned(),
                            main_branch: main_branch.to_owned(),
                            coordination_branch: PRIVATE_COORDINATION_BRANCH.to_owned(),
                            coordination_head: commit_id,
                            state,
                            lock,
                        });
                    }
                    CreateRefResult::RefExists { .. } => continue,
                }
            }
            Err(error) => return Err(error),
            Ok(coordination_head) => {
                let mut state: PrivatePublicationCoordinationState = read_json(
                    remote,
                    repository,
                    &coordination_head,
                    PRIVATE_COORDINATION_STATE_PATH,
                    cancellation,
                )?;
                state.validate()?;
                let existing: Option<PrivatePublicationLock> = read_optional_json(
                    remote,
                    repository,
                    &coordination_head,
                    PRIVATE_PUBLICATION_LOCK_PATH,
                    cancellation,
                )?;
                if let Some(existing) = &existing {
                    existing.validate()?;
                    if existing.fencing_generation != state.fencing_generation {
                        return Err(CacheError::InvalidManifest(
                            "private lock generation does not match coordination state".to_owned(),
                        ));
                    }
                }
                let available = existing
                    .as_ref()
                    .is_none_or(|lock| lock.expired_at(now, policy.clock_skew_grace_seconds));
                if available {
                    state.fencing_generation =
                        state.fencing_generation.checked_add(1).ok_or_else(|| {
                            CacheError::ResourceLimit(
                                "private publication fencing generation exhausted".to_owned(),
                            )
                        })?;
                    let lock = lock_record(
                        owner,
                        transaction_hint,
                        state.fencing_generation,
                        &main_head,
                        now,
                        now,
                        policy,
                    )?;
                    let parts = vec![
                        stage_json(staging_root, PRIVATE_COORDINATION_STATE_PATH, &state)?,
                        stage_json(staging_root, PRIVATE_PUBLICATION_LOCK_PATH, &lock)?,
                    ];
                    let request = RemoteCommitRequest {
                        repository: repository.to_owned(),
                        branch: PRIVATE_COORDINATION_BRANCH.to_owned(),
                        expected_head: coordination_head,
                        message: format!(
                            "acquire private publication lease generation {}",
                            state.fencing_generation
                        ),
                        parts,
                        delete_paths: Vec::new(),
                    };
                    match remote.compare_and_swap_commit(&request)? {
                        CompareAndSwapResult::Committed { commit_id } => {
                            eprintln!(
                                "private publication lock acquired: generation={} repository={repository}",
                                state.fencing_generation
                            );
                            return Ok(PrivatePublicationLease {
                                repository: repository.to_owned(),
                                main_branch: main_branch.to_owned(),
                                coordination_branch: PRIVATE_COORDINATION_BRANCH.to_owned(),
                                coordination_head: commit_id,
                                state,
                                lock,
                            });
                        }
                        CompareAndSwapResult::RefConflict { .. } => continue,
                    }
                }
                if now.saturating_sub(started) >= policy.maximum_wait_seconds {
                    return Err(CacheError::ResourceLimit(format!(
                        "timed out waiting {} seconds for private publication lock in {repository}",
                        policy.maximum_wait_seconds
                    )));
                }
                if now.saturating_sub(last_status) >= 60 {
                    let existing = existing.expect("unavailable lock exists");
                    eprintln!(
                        "private publication lock held: principal={} run={} generation={} lease_remaining={}s; waiting",
                        existing.github_principal,
                        &existing.owner_run_id[..12.min(existing.owner_run_id.len())],
                        existing.fencing_generation,
                        existing.lease_expires_at_unix_seconds.saturating_sub(now)
                    );
                    last_status = now;
                }
                sleep_with_cancellation(
                    jittered_poll_seconds(
                        poll_seconds,
                        policy.maximum_poll_seconds,
                        &owner.owner_run_id,
                        state.fencing_generation,
                    ),
                    cancellation,
                )?;
                poll_seconds = poll_seconds
                    .saturating_mul(2)
                    .min(policy.maximum_poll_seconds);
            }
        }
    }
}

pub fn prepare_private_lease_renewal(
    lease: &PrivatePublicationLease,
    observed_main_head: &str,
    staging_root: &Path,
    policy: &PrivatePublicationLeasePolicy,
) -> Result<(RemoteCommitRequest, PrivatePublicationLock), CacheError> {
    policy.validate()?;
    if !valid_revision(observed_main_head) {
        return Err(CacheError::InvalidManifest(
            "private lease renewal observed an invalid main head".to_owned(),
        ));
    }
    let now = now_unix_seconds()?;
    let mut renewed = lease.lock.clone();
    renewed.observed_main_head = observed_main_head.to_owned();
    renewed.heartbeat_at_unix_seconds = now;
    renewed.lease_expires_at_unix_seconds = now.saturating_add(policy.lease_seconds);
    renewed.validate()?;
    let part = stage_json(staging_root, PRIVATE_PUBLICATION_LOCK_PATH, &renewed)?;
    Ok((
        RemoteCommitRequest {
            repository: lease.repository.clone(),
            branch: lease.coordination_branch.clone(),
            expected_head: lease.coordination_head.clone(),
            message: format!(
                "renew private publication lease generation {}",
                renewed.fencing_generation
            ),
            parts: vec![part],
            delete_paths: Vec::new(),
        },
        renewed,
    ))
}

pub fn renew_private_publication_lease(
    remote: &dyn RemoteGitStore,
    lease: &mut PrivatePublicationLease,
    observed_main_head: &str,
    staging_root: &Path,
    policy: &PrivatePublicationLeasePolicy,
) -> Result<(), CacheError> {
    let (request, renewed) =
        prepare_private_lease_renewal(lease, observed_main_head, staging_root, policy)?;
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::Committed { commit_id } => {
            lease.coordination_head = commit_id;
            lease.lock = renewed;
            Ok(())
        }
        CompareAndSwapResult::RefConflict { current_head } => Err(CacheError::InvalidTransition(
            format!("private publication lease was lost at coordination head {current_head}"),
        )),
    }
}

pub fn commit_private_batch_atomically(
    remote: &dyn RemoteGitStore,
    lease: &mut PrivatePublicationLease,
    main_request: RemoteCommitRequest,
    staging_root: &Path,
    policy: &PrivatePublicationLeasePolicy,
) -> Result<String, CacheError> {
    if main_request.repository != lease.repository
        || main_request.branch != lease.main_branch
        || main_request.expected_head != lease.lock.observed_main_head
    {
        return Err(CacheError::InvalidTransition(
            "private publication batch does not match its lease target and observed head"
                .to_owned(),
        ));
    }
    let (coordination_request, renewed) =
        prepare_private_lease_renewal(lease, &main_request.expected_head, staging_root, policy)?;
    let request = AtomicRemoteCommitRequest {
        repository: lease.repository.clone(),
        commits: vec![main_request, coordination_request],
    };
    match remote.compare_and_swap_commits_atomically(&request)? {
        AtomicCompareAndSwapResult::Committed { commit_ids } => {
            let main_head = commit_ids.get(&lease.main_branch).cloned().ok_or_else(|| {
                CacheError::InvalidTransition(
                    "atomic private publication omitted the cache commit".to_owned(),
                )
            })?;
            let coordination_head = commit_ids
                .get(&lease.coordination_branch)
                .cloned()
                .ok_or_else(|| {
                    CacheError::InvalidTransition(
                        "atomic private publication omitted the lease renewal".to_owned(),
                    )
                })?;
            lease.coordination_head = coordination_head;
            lease.lock = renewed;
            lease.lock.observed_main_head = main_head.clone();
            Ok(main_head)
        }
        AtomicCompareAndSwapResult::RefConflict { current_heads } => {
            Err(CacheError::InvalidTransition(format!(
                "private publication lost its lease or shard head: {current_heads:?}"
            )))
        }
    }
}

pub fn release_private_publication_lease(
    remote: &dyn RemoteGitStore,
    lease: &PrivatePublicationLease,
    staging_root: &Path,
    completed: bool,
) -> Result<(), CacheError> {
    let now = now_unix_seconds()?;
    let mut state = lease.state.clone();
    if completed {
        state.last_completed_transaction = Some(lease.lock.publication_transaction_id.clone());
    }
    state.last_released_at_unix_seconds = Some(now);
    state.validate()?;
    let state_part = stage_json(staging_root, PRIVATE_COORDINATION_STATE_PATH, &state)?;
    let request = RemoteCommitRequest {
        repository: lease.repository.clone(),
        branch: lease.coordination_branch.clone(),
        expected_head: lease.coordination_head.clone(),
        message: format!(
            "release private publication lease generation {}",
            lease.lock.fencing_generation
        ),
        parts: vec![state_part],
        delete_paths: vec![PRIVATE_PUBLICATION_LOCK_PATH.to_owned()],
    };
    match remote.compare_and_swap_commit(&request)? {
        CompareAndSwapResult::Committed { .. } => {
            eprintln!(
                "private publication lock released: generation={} repository={}",
                lease.lock.fencing_generation, lease.repository
            );
            Ok(())
        }
        CompareAndSwapResult::RefConflict { current_head } => Err(CacheError::InvalidTransition(
            format!("refused to release a superseded private publication lease at {current_head}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GitCliRemoteStore;
    use std::fs;
    use std::process::{Command, Stdio};

    fn temporary_root(name: &str) -> std::path::PathBuf {
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
    fn lock_expiration_includes_clock_skew_grace() {
        let lock = PrivatePublicationLock {
            schema_version: 1,
            owner_run_id: "run".to_owned(),
            publication_transaction_id: "transaction".to_owned(),
            github_principal: "principal".to_owned(),
            toolkit_version: "0.13.0".to_owned(),
            instance_fingerprint: ContentDigest::sha256(b"instance"),
            process_id: 1,
            fencing_generation: 3,
            observed_main_head: "a".repeat(40),
            acquired_at_unix_seconds: 10,
            heartbeat_at_unix_seconds: 20,
            lease_expires_at_unix_seconds: 100,
        };
        lock.validate().unwrap();
        assert!(!lock.expired_at(159, 60));
        assert!(lock.expired_at(160, 60));
    }

    #[test]
    fn state_rejects_empty_completed_transaction() {
        let state = PrivatePublicationCoordinationState {
            schema_version: 1,
            fencing_generation: 1,
            last_completed_transaction: Some(String::new()),
            last_released_at_unix_seconds: Some(1),
        };
        assert!(state.validate().is_err());
    }

    #[test]
    fn wait_jitter_is_bounded_and_owner_specific() {
        let first = jittered_poll_seconds(20, 60, "owner-a", 4);
        let second = jittered_poll_seconds(20, 60, "owner-b", 4);
        assert!((20..=25).contains(&first));
        assert!((20..=25).contains(&second));
        assert_ne!(first, second);
        assert_eq!(jittered_poll_seconds(60, 60, "owner-a", 4), 60);
    }

    #[test]
    fn private_lease_lifecycle_fences_cache_batches_and_increments_generation() {
        if !test_git(None, &["--version"]) {
            return;
        }
        let root = temporary_root("private-publication-lease-lifecycle");
        let _ = fs::remove_dir_all(&root);
        let remote_path = root.join("remote.git");
        let seed = root.join("seed");
        let staging = root.join("staging");
        fs::create_dir_all(&root).unwrap();
        assert!(test_git(
            None,
            &["init", "--bare", remote_path.to_str().unwrap()]
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
            &["remote", "add", "origin", remote_path.to_str().unwrap()]
        ));
        assert!(test_git(
            Some(&seed),
            &["push", "origin", "HEAD:refs/heads/main"]
        ));
        let repository = remote_path.to_string_lossy().to_string();
        let store = GitCliRemoteStore::new(
            root.join("transport"),
            &staging,
            "Test Publisher",
            "test@example.invalid",
        )
        .unwrap();
        let owner = PrivatePublicationLeaseOwner {
            owner_run_id: "owner-run-a".to_owned(),
            github_principal: "publisher".to_owned(),
            instance_fingerprint: ContentDigest::sha256(b"instance-a"),
            process_id: 1,
        };
        let cancellation = CancellationToken::new();
        let policy = PrivatePublicationLeasePolicy::default();
        let mut lease = acquire_private_publication_lease(
            &store,
            &repository,
            "main",
            &owner,
            "transaction-a",
            &staging,
            &cancellation,
            &policy,
        )
        .unwrap();
        assert_eq!(lease.lock.fencing_generation, 1);
        let payload_path = staging.join("objects").join("payload.part");
        fs::create_dir_all(payload_path.parent().unwrap()).unwrap();
        fs::write(&payload_path, b"payload").unwrap();
        let payload = TransportPart {
            sequence: 0,
            repository_path: "objects/payload.part".to_owned(),
            size_bytes: 7,
            content_digest: ContentDigest::sha256(b"payload"),
        };
        let main_head = store.read_ref(&repository, "main").unwrap();
        assert_eq!(main_head, lease.lock.observed_main_head);
        let committed = commit_private_batch_atomically(
            &store,
            &mut lease,
            RemoteCommitRequest {
                repository: repository.clone(),
                branch: "main".to_owned(),
                expected_head: main_head,
                message: "publish fenced payload".to_owned(),
                parts: vec![payload.clone()],
                delete_paths: Vec::new(),
            },
            &staging,
            &policy,
        )
        .unwrap();
        store
            .verify_committed_part(&repository, &committed, &payload)
            .unwrap();
        release_private_publication_lease(&store, &lease, &staging, true).unwrap();
        let released_head = store
            .read_ref(&repository, PRIVATE_COORDINATION_BRANCH)
            .unwrap();
        assert!(store
            .immutable_path_digest(&repository, &released_head, PRIVATE_PUBLICATION_LOCK_PATH)
            .unwrap()
            .is_none());

        let second_owner = PrivatePublicationLeaseOwner {
            owner_run_id: "owner-run-b".to_owned(),
            github_principal: "publisher".to_owned(),
            instance_fingerprint: ContentDigest::sha256(b"instance-b"),
            process_id: 2,
        };
        let second = acquire_private_publication_lease(
            &store,
            &repository,
            "main",
            &second_owner,
            "transaction-b",
            &staging,
            &cancellation,
            &policy,
        )
        .unwrap();
        assert_eq!(second.lock.fencing_generation, 2);
        release_private_publication_lease(&store, &second, &staging, false).unwrap();
        let _ = fs::remove_dir_all(root);
    }
}
