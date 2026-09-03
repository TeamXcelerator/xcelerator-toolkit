//! One-step, resumable orchestration of a complete publication target.

use crate::{
    execute_next_finalization_step, execute_next_payload_batch,
    stage_immutable_publication_metadata, stage_publication_discoverability,
    AuthenticatedGitHubSession, CacheError, CapacityLedger, DiscoverabilityStageReport,
    ImmutableMetadataStageReport, PublicSanitizerProfile, PublicationDestination,
    PublicationFinalizationOutcome, PublicationFinalizationPolicy, PublicationJournalStore,
    PublicationMetadataBundle, PublicationStepOutcome, PublicationTargetState,
    PublicationTransactionJournal, RemoteDocument, RemoteGitStore, RemoteShardReader,
    ShardIndexPartition, DEFAULT_CAPACITY_LEDGER_PATH,
};
use std::path::Path;
use xc_core::{CancellationToken, ResourcePolicy};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicationAdvanceOutcome {
    FailedStateRestored { state: PublicationTargetState },
    Payload(PublicationStepOutcome),
    ImmutableMetadataStaged(ImmutableMetadataStageReport),
    Finalization(PublicationFinalizationOutcome),
    DiscoverabilityStaged(DiscoverabilityStageReport),
    Complete,
}

pub struct PublicationTargetExecution<'a> {
    pub staging_root: &'a Path,
    pub resources: &'a ResourcePolicy,
    pub finalization_policy: &'a PublicationFinalizationPolicy,
    pub bundle: &'a PublicationMetadataBundle,
    pub public_sanitizer: Option<&'a PublicSanitizerProfile>,
    pub maximum_index_bytes: u64,
    pub receipt_verified_at_unix_seconds: u64,
    pub replace_existing_semantic: bool,
}

pub struct PublicationTransactionExecution<'a> {
    pub private: Option<PublicationTargetExecution<'a>>,
    pub public: Option<PublicationTargetExecution<'a>>,
}

impl PublicationTransactionExecution<'_> {
    fn target(
        &self,
        destination: PublicationDestination,
    ) -> Option<&PublicationTargetExecution<'_>> {
        match destination {
            PublicationDestination::Private => self.private.as_ref(),
            PublicationDestination::Public => self.public.as_ref(),
        }
    }
}

pub struct PublicationTransactionSessions<'a> {
    pub private: Option<&'a AuthenticatedGitHubSession>,
    pub public: Option<&'a AuthenticatedGitHubSession>,
}

impl PublicationTransactionSessions<'_> {
    fn target(&self, destination: PublicationDestination) -> Option<&AuthenticatedGitHubSession> {
        match destination {
            PublicationDestination::Private => self.private,
            PublicationDestination::Public => self.public,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicationTransactionAdvanceOutcome {
    TargetAdvanced {
        destination: PublicationDestination,
        outcome: PublicationAdvanceOutcome,
    },
    Complete,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationTransactionCompletion {
    pub transaction_id: String,
    pub steps: Vec<PublicationTransactionAdvanceOutcome>,
    pub final_journal_digest: crate::ContentDigest,
}

fn next_incomplete_destination(
    journal: &PublicationTransactionJournal,
) -> Option<PublicationDestination> {
    [
        PublicationDestination::Private,
        PublicationDestination::Public,
    ]
    .into_iter()
    .find(|destination| {
        journal.targets.get(destination).is_some_and(|target| {
            target.state != PublicationTargetState::ReceiptComplete
                && target.state != PublicationTargetState::Abandoned
        })
    })
}

/// Advance the next incomplete target in deterministic private-then-public
/// order. Target-specific sessions and public sanitization prevent a dual
/// publication from accidentally reusing private authority or metadata.
#[allow(clippy::too_many_arguments)]
pub fn advance_publication_transaction(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    sessions: &PublicationTransactionSessions<'_>,
    journal: &mut PublicationTransactionJournal,
    execution: &PublicationTransactionExecution<'_>,
) -> Result<PublicationTransactionAdvanceOutcome, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    let Some(destination) = next_incomplete_destination(journal) else {
        return if journal.complete() {
            Ok(PublicationTransactionAdvanceOutcome::Complete)
        } else {
            Err(CacheError::InvalidTransition(
                "publication transaction has no resumable target and is not complete".to_owned(),
            ))
        };
    };
    let session = sessions.target(destination).ok_or_else(|| {
        CacheError::Authentication(format!(
            "publication transaction lacks a current {destination:?} authenticated session"
        ))
    })?;
    let target_execution = execution.target(destination).ok_or_else(|| {
        CacheError::InvalidManifest(format!(
            "publication transaction lacks {destination:?} execution policy"
        ))
    })?;
    let outcome = advance_publication_target(
        remote,
        checkpoints,
        cancellation,
        session,
        journal,
        destination,
        target_execution,
    )?;
    Ok(PublicationTransactionAdvanceOutcome::TargetAdvanced {
        destination,
        outcome,
    })
}

/// Execute an explicitly authorized publication transaction to receipt
/// completion, with a caller-supplied step ceiling for bounded automation.
#[allow(clippy::too_many_arguments)]
pub fn complete_publication_transaction(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    sessions: &PublicationTransactionSessions<'_>,
    journal: &mut PublicationTransactionJournal,
    execution: &PublicationTransactionExecution<'_>,
    maximum_steps: usize,
) -> Result<PublicationTransactionCompletion, CacheError> {
    if maximum_steps == 0 {
        return Err(CacheError::ResourceLimit(
            "automatic publication requires a positive step ceiling".to_owned(),
        ));
    }
    let mut steps = Vec::new();
    for _ in 0..maximum_steps {
        let outcome = advance_publication_transaction(
            remote,
            checkpoints,
            cancellation,
            sessions,
            journal,
            execution,
        )?;
        let complete = outcome == PublicationTransactionAdvanceOutcome::Complete;
        steps.push(outcome);
        if complete {
            return Ok(PublicationTransactionCompletion {
                transaction_id: journal.transaction_id.clone(),
                steps,
                final_journal_digest: journal.digest()?,
            });
        }
    }
    Err(CacheError::ResourceLimit(format!(
        "publication transaction {} did not complete within {maximum_steps} steps",
        journal.transaction_id
    )))
}

/// Advance one target by at most one local planning phase or one remote
/// mutation. Repeated calls are the normal execution and recovery interface.
/// Every remote mutation independently rechecks live permission and capacity.
#[allow(clippy::too_many_arguments)]
pub fn advance_publication_target(
    remote: &dyn RemoteGitStore,
    checkpoints: &PublicationJournalStore,
    cancellation: &CancellationToken,
    authenticated_session: &AuthenticatedGitHubSession,
    journal: &mut PublicationTransactionJournal,
    destination: PublicationDestination,
    execution: &PublicationTargetExecution<'_>,
) -> Result<PublicationAdvanceOutcome, CacheError> {
    cancellation
        .check()
        .map_err(|error| CacheError::Cancelled(error.to_string()))?;
    journal.validate()?;
    let target = journal.targets.get(&destination).ok_or_else(|| {
        CacheError::InvalidTransition(format!(
            "transaction has no {destination:?} publication target"
        ))
    })?;
    if target.state == PublicationTargetState::ReceiptComplete {
        return Ok(PublicationAdvanceOutcome::Complete);
    }
    if target.state == PublicationTargetState::Abandoned {
        return Err(CacheError::InvalidTransition(format!(
            "abandoned {destination:?} publication target cannot be resumed"
        )));
    }
    if target.state == PublicationTargetState::Failed {
        let target = journal
            .targets
            .get_mut(&destination)
            .expect("target existence was checked");
        target.restore_failed_resume_state()?;
        let state = target.state;
        checkpoints.save(journal)?;
        return Ok(PublicationAdvanceOutcome::FailedStateRestored { state });
    }
    if target
        .batches
        .iter()
        .any(|batch| batch.state != crate::PublicationBatchState::RemoteVerified)
    {
        let outcome = execute_next_payload_batch(
            remote,
            checkpoints,
            cancellation,
            authenticated_session,
            execution.finalization_policy,
            execution.staging_root,
            execution.resources,
            journal,
            destination,
        )?;
        return Ok(PublicationAdvanceOutcome::Payload(outcome));
    }
    if target.metadata_commit.is_none() {
        let report = stage_immutable_publication_metadata(
            execution.staging_root,
            execution.resources,
            cancellation,
            journal,
            destination,
            execution.bundle,
            execution.public_sanitizer,
        )?;
        checkpoints.save(journal)?;
        return Ok(PublicationAdvanceOutcome::ImmutableMetadataStaged(report));
    }
    if target
        .metadata_commit
        .as_ref()
        .is_some_and(|commit| commit.state != crate::PublicationCommitState::RemoteVerified)
    {
        return Ok(PublicationAdvanceOutcome::Finalization(
            execute_next_finalization_step(
                remote,
                checkpoints,
                cancellation,
                authenticated_session,
                execution.finalization_policy,
                journal,
                destination,
            )?,
        ));
    }
    if target.discoverability_commit.is_none() {
        if execution.maximum_index_bytes == 0 {
            return Err(CacheError::ResourceLimit(
                "maximum shard-index read size must be positive".to_owned(),
            ));
        }
        let repository = target.repository.clone();
        let revision = target.expected_head.clone();
        let semantic_prefix = &journal.semantic_digest.0[..2];
        let index_path = format!("indexes/{}/{semantic_prefix}.json", execution.bundle.family);
        let index_reader = RemoteShardReader::new(remote, execution.maximum_index_bytes)?;
        let index: RemoteDocument<ShardIndexPartition> =
            index_reader.read_json(&repository, &revision, &index_path, cancellation)?;
        let ledger_reader = RemoteShardReader::new(
            remote,
            execution.finalization_policy.maximum_capacity_ledger_bytes,
        )?;
        let ledger: RemoteDocument<CapacityLedger> = ledger_reader.read_json(
            &repository,
            &revision,
            DEFAULT_CAPACITY_LEDGER_PATH,
            cancellation,
        )?;
        let report = stage_publication_discoverability(
            execution.staging_root,
            execution.resources,
            cancellation,
            journal,
            destination,
            execution.bundle,
            &index,
            &ledger,
            execution.receipt_verified_at_unix_seconds,
            execution.finalization_policy.projected_history_bytes,
            execution.replace_existing_semantic,
        )?;
        checkpoints.save(journal)?;
        return Ok(PublicationAdvanceOutcome::DiscoverabilityStaged(report));
    }
    Ok(PublicationAdvanceOutcome::Finalization(
        execute_next_finalization_step(
            remote,
            checkpoints,
            cancellation,
            authenticated_session,
            execution.finalization_policy,
            journal,
            destination,
        )?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical_json_bytes;
    use crate::{
        plan_publication_batches, ArtifactAssuranceState, ArtifactDisposition, AttestationEnvelope,
        AttestationKind, CanonicalArtifactManifest, CanonicalPayloadEnvelope, ContentDigest,
        GitCliRemoteStore, LogicalPayloadItem, PublicationMetadataBundle, PublicationReceipt,
        RepositoryPermission, SemanticKeyEnvelope, ShardIndexPartition, TransportEncodingRecord,
        TransportPart, TransportPolicy, ValidatorEvidence, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use xc_core::{AssuranceLevel, PublicationTarget};

    fn test_git(directory: Option<&Path>, arguments: &[&str]) -> bool {
        let mut command = Command::new("git");
        if let Some(directory) = directory {
            command.arg("-C").arg(directory);
        }
        command.args(arguments);
        command.stdout(Stdio::null()).stderr(Stdio::null());
        command.status().is_ok_and(|status| status.success())
    }

    fn temporary_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("test-tmp")
            .join(format!("publication-orchestrator-{}", std::process::id()))
    }

    #[test]
    fn one_step_driver_completes_a_real_no_checkout_git_transaction() {
        if !test_git(None, &["--version"]) {
            return;
        }
        let root = temporary_root();
        let _ = fs::remove_dir_all(&root);
        let remote_path = root.join("remote.git");
        let seed = root.join("seed");
        let transport_root = root.join("transport");
        let staging_root = root.join("staging");
        let checkpoints_root = root.join("journals");
        fs::create_dir_all(&root).unwrap();

        let payload_bytes = b"orchestrated-payload-part";
        let semantic_key = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "orchestrated_fixture".to_owned(),
            mathematical_semantics_version: "fixture-v1".to_owned(),
            resolved_mathematical_parameters: json!({"case": "orchestrator"}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let semantic_digest = semantic_key.digest().unwrap();
        let canonical_payload = CanonicalPayloadEnvelope {
            schema_version: 1,
            scalar_backend: "opaque".to_owned(),
            precision_bits: None,
            scalar_representation: "bytes".to_owned(),
            dimensions: vec![payload_bytes.len() as u64],
            endianness: "not-applicable".to_owned(),
            special_value_encoding: "not-applicable".to_owned(),
            ordered_items: vec![LogicalPayloadItem {
                normalized_path: "fixture.bin".to_owned(),
                content_digest: ContentDigest::sha256(payload_bytes),
                size_bytes: payload_bytes.len() as u64,
            }],
            dependencies: Vec::new(),
        };
        let payload_digest = canonical_payload.digest().unwrap();
        let part_digest = ContentDigest::sha256(payload_bytes);
        let part = TransportPart {
            sequence: 0,
            repository_path: format!(
                "objects/sha256/{}/{}.part",
                &part_digest.0[..2],
                part_digest.0
            ),
            size_bytes: payload_bytes.len() as u64,
            content_digest: part_digest,
        };
        let encoding = TransportEncodingRecord {
            schema_version: 1,
            canonical_payload_digest: payload_digest.clone(),
            encoder_profile: crate::DETERMINISTIC_ZIP64_PROFILE_V1.to_owned(),
            package_size_bytes: part.size_bytes,
            package_digest: part.content_digest.clone(),
            ordered_parts: vec![part.clone()],
            reconstruction: "concatenate".to_owned(),
        };
        let transport_digest = encoding.digest().unwrap();
        let manifest = CanonicalArtifactManifest {
            schema_version: 1,
            artifact_family: "ccm".to_owned(),
            semantic_key,
            semantic_digest: semantic_digest.clone(),
            canonical_payload,
            payload_digest: payload_digest.clone(),
            transport_digests: vec![transport_digest],
            resolved_mathematical_configuration_digest: ContentDigest::sha256(b"configuration"),
            producer_toolkit_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            minimum_reader_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
            maximum_reader_version: None,
            requested_assurance: AssuranceLevel::Computed,
            claim_scope: "orchestrator fixture".to_owned(),
            assumptions: Vec::new(),
        };
        let manifest_digest = manifest.digest().unwrap();
        let evidence_digest = ContentDigest::sha256(b"validator-evidence");
        let bundle = PublicationMetadataBundle {
            schema_version: 1,
            family: "ccm".to_owned(),
            manifest,
            encoding,
            validation_attestations: vec![AttestationEnvelope {
                schema_version: 1,
                kind: AttestationKind::Validation,
                subject_digest: manifest_digest.clone(),
                actor: "fixture-validator".to_owned(),
                policy_digest: ContentDigest::sha256(b"policy"),
                execution_fingerprint_digest: ContentDigest::sha256(b"fingerprint"),
                producer_toolkit_version: crate::ToolkitVersion::parse("0.13.0").unwrap(),
                dependency_versions: BTreeMap::from([("xc-cache".to_owned(), "0.13.0".to_owned())]),
                source_revision: "toolkit-revision".to_owned(),
                event_unix_seconds: 1,
                location: None,
                evidence_digests: vec![evidence_digest.clone()],
            }],
            validator_evidence: vec![ValidatorEvidence {
                validator_id: "fixture-validator".to_owned(),
                passed: true,
                evidence_digest,
                establishes_assurance: Some(ArtifactAssuranceState::Computed),
            }],
            target_metadata: BTreeMap::new(),
            achieved_assurance: ArtifactAssuranceState::Computed,
            disposition: ArtifactDisposition::Active,
        };

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
        let prefix = &semantic_digest.0[..2];
        let index_path = seed.join("indexes").join("ccm");
        fs::create_dir_all(&index_path).unwrap();
        let index = ShardIndexPartition::rebuild("ccm", prefix, Vec::new()).unwrap();
        fs::write(
            index_path.join(format!("{prefix}.json")),
            canonical_json_bytes(&index).unwrap(),
        )
        .unwrap();
        let ledger_path = seed.join("ledger");
        fs::create_dir_all(&ledger_path).unwrap();
        let ledger = CapacityLedger {
            schema_version: 1,
            shard_id: "private-001".to_owned(),
            hard_capacity_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            warning_reserve_bytes: 1_000_000,
            first_seen_immutable_payload_bytes: 0,
            manifest_index_receipt_bytes: 0,
            estimated_history_bytes: 0,
            emergency_reserve_bytes: 0,
            abandoned_reachable_bytes: 0,
            last_reconciled_commit: "initial".to_owned(),
            reconciliation_digest: ContentDigest::sha256(b"initial-ledger"),
        };
        fs::write(
            ledger_path.join("capacity.json"),
            canonical_json_bytes(&ledger).unwrap(),
        )
        .unwrap();
        assert!(test_git(Some(&seed), &["add", "."]));
        assert!(test_git(Some(&seed), &["commit", "-m", "initialize shard"]));
        assert!(test_git(
            Some(&seed),
            &["remote", "add", "origin", remote_path.to_str().unwrap()]
        ));
        assert!(test_git(
            Some(&seed),
            &["push", "origin", "HEAD:refs/heads/main"]
        ));

        let payload_staging_path = part
            .repository_path
            .split('/')
            .fold(staging_root.clone(), |path, component| path.join(component));
        fs::create_dir_all(payload_staging_path.parent().unwrap()).unwrap();
        fs::write(&payload_staging_path, payload_bytes).unwrap();
        let remote = GitCliRemoteStore::new(
            &transport_root,
            &staging_root,
            "Test Publisher",
            "test@example.invalid",
        )
        .unwrap();
        let repository = remote_path.to_string_lossy().to_string();
        let expected_head = remote.read_ref(&repository, "main").unwrap();
        let session = AuthenticatedGitHubSession::verified_for_test(
            "test-owner",
            "team/private",
            RepositoryPermission::Write,
        );
        let batches =
            plan_publication_batches(std::slice::from_ref(&part), &TransportPolicy::default())
                .unwrap();
        let mut journal = PublicationTransactionJournal::new(
            semantic_digest.clone(),
            BTreeMap::from([(PublicationDestination::Private, manifest_digest.clone())]),
            payload_digest,
            ContentDigest::sha256(b"policy"),
            PublicationTarget::Private,
            BTreeMap::from([(PublicationDestination::Private, repository.clone())]),
            BTreeMap::from([(PublicationDestination::Private, "team/private".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, session.evidence().clone())]),
            BTreeMap::from([(PublicationDestination::Private, "private-001".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, "main".to_owned())]),
            BTreeMap::from([(PublicationDestination::Private, expected_head)]),
            &batches,
        )
        .unwrap();
        crate::attach_test_owner_audit_evidence(&mut journal, PublicationDestination::Private);
        let checkpoints = PublicationJournalStore::new(&checkpoints_root);
        checkpoints.save(&journal).unwrap();
        let resources = ResourcePolicy::default();
        let finalization_policy = PublicationFinalizationPolicy {
            projected_history_bytes: 256,
            ..PublicationFinalizationPolicy::default()
        };
        let execution = PublicationTransactionExecution {
            private: Some(PublicationTargetExecution {
                staging_root: &staging_root,
                resources: &resources,
                finalization_policy: &finalization_policy,
                bundle: &bundle,
                public_sanitizer: None,
                maximum_index_bytes: 4 * 1024 * 1024,
                receipt_verified_at_unix_seconds: 123,
                replace_existing_semantic: false,
            }),
            public: None,
        };
        let cancellation = CancellationToken::new();
        let sessions = PublicationTransactionSessions {
            private: Some(&session),
            public: None,
        };
        let missing_sessions = PublicationTransactionSessions {
            private: None,
            public: None,
        };
        assert!(matches!(
            advance_publication_transaction(
                &remote,
                &checkpoints,
                &cancellation,
                &missing_sessions,
                &mut journal,
                &execution,
            ),
            Err(CacheError::Authentication(_))
        ));
        assert!(matches!(
            complete_publication_transaction(
                &remote,
                &checkpoints,
                &cancellation,
                &sessions,
                &mut journal,
                &execution,
                0,
            ),
            Err(CacheError::ResourceLimit(_))
        ));
        let completion = complete_publication_transaction(
            &remote,
            &checkpoints,
            &cancellation,
            &sessions,
            &mut journal,
            &execution,
            16,
        )
        .unwrap();
        let saw_payload = completion.steps.iter().any(|step| {
            matches!(
                step,
                PublicationTransactionAdvanceOutcome::TargetAdvanced {
                    outcome: PublicationAdvanceOutcome::Payload(_),
                    ..
                }
            )
        });
        let saw_metadata = completion.steps.iter().any(|step| {
            matches!(
                step,
                PublicationTransactionAdvanceOutcome::TargetAdvanced {
                    outcome: PublicationAdvanceOutcome::ImmutableMetadataStaged(_),
                    ..
                }
            )
        });
        let saw_discoverability = completion.steps.iter().any(|step| {
            matches!(
                step,
                PublicationTransactionAdvanceOutcome::TargetAdvanced {
                    outcome: PublicationAdvanceOutcome::DiscoverabilityStaged(_),
                    ..
                }
            )
        });
        assert!(journal.complete());
        assert_eq!(completion.transaction_id, journal.transaction_id);
        assert_eq!(completion.final_journal_digest, journal.digest().unwrap());
        assert!(saw_payload && saw_metadata && saw_discoverability);
        assert_eq!(
            checkpoints.load_latest(&journal.transaction_id).unwrap(),
            journal
        );
        let final_head = remote.read_ref(&repository, "main").unwrap();
        let receipt_path = format!(
            "transactions/{}/private/receipt.json",
            journal.transaction_id
        );
        let mut receipt_bytes = Vec::new();
        remote
            .read_committed_path(
                &repository,
                &final_head,
                &receipt_path,
                4 * 1024 * 1024,
                &cancellation,
                &mut receipt_bytes,
            )
            .unwrap();
        let receipt: PublicationReceipt = serde_json::from_slice(&receipt_bytes).unwrap();
        receipt
            .validate_for_transaction(&journal, PublicationDestination::Private)
            .unwrap();
        let reader = RemoteShardReader::new(&remote, 4 * 1024 * 1024).unwrap();
        let final_index: RemoteDocument<ShardIndexPartition> = reader
            .read_json(
                &repository,
                &final_head,
                &format!("indexes/ccm/{prefix}.json"),
                &cancellation,
            )
            .unwrap();
        assert_eq!(
            final_index
                .value
                .lookup(&semantic_digest)
                .next()
                .unwrap()
                .manifest_digest,
            manifest_digest
        );
        assert!(!transport_root.join("working-tree").exists());
        remote.cleanup_session(&repository).unwrap();
        let _ = fs::remove_dir_all(root);
    }
}
