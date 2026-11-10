//! Artifact admission before public payload or metadata mutations.
//!
//! A journal authenticates a transaction, not permission to disclose its
//! artifact. Admission is reconstructed on every step, including resume.
use crate::{
    CacheError, CanonicalArtifactManifest, ContentDigest, PublicSanitizerProfile,
    PublicationDestination, PublicationMetadataBundle, PublicationTransactionJournal,
};

/// Checked admission for one exact transaction state and destination.
/// Cannot be deserialized or constructed by setting an eligibility flag.
#[derive(Debug)]
pub struct PublicationAdmission {
    destination: PublicationDestination,
    journal_digest: ContentDigest,
}

impl PublicationAdmission {
    pub(crate) fn check(
        &self,
        journal: &PublicationTransactionJournal,
        destination: PublicationDestination,
    ) -> Result<(), CacheError> {
        if self.destination != destination || self.journal_digest != journal.digest()? {
            return Err(CacheError::PermissionDenied(
                "publication admission does not bind this transaction state and destination".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn require_artifact_destination(
    manifest: &CanonicalArtifactManifest,
    destination: PublicationDestination,
) -> Result<(), CacheError> {
    if !manifest_admitted_to_destination(manifest, destination) {
        // Do not include private input or provenance values in the reason.
        return Err(CacheError::PermissionDenied("withheld_private_only".into()));
    }
    manifest.validate()
}

/// Classification comes from the selected authenticated producer manifest,
/// independently of numerical identity or the cache lane holding its bytes.
pub fn manifest_admitted_to_destination(
    manifest: &CanonicalArtifactManifest,
    destination: PublicationDestination,
) -> bool {
    if destination == PublicationDestination::Private {
        return true;
    }
    let Ok(contract) = crate::artifact_identity_contract(&manifest.semantic_key.artifact_kind)
    else {
        return false;
    };
    if contract.family != manifest.artifact_family {
        return false;
    }
    if crate::artifact_kind_is_private_only(&manifest.semantic_key.artifact_kind) {
        return false;
    }
    match (manifest.schema_version, &manifest.provenance) {
        (1, None) => {
            crate::artifact_semantics_admitted_to_destination(&manifest.semantic_key, destination)
        }
        (2, Some(provenance)) => provenance.disclosure == crate::ArtifactDisclosure::PublicInputs,
        _ => false,
    }
}

/// Validate the complete selected public ancestry, never substituting a
/// same-content producer. Returns the exact reachable closure in digest order.
pub fn public_admission_closure(
    root: &CanonicalArtifactManifest,
    available: &[CanonicalArtifactManifest],
) -> Result<Vec<CanonicalArtifactManifest>, CacheError> {
    use std::collections::{BTreeMap, BTreeSet};
    let reader = crate::current_toolkit_version()?;
    let mut by_digest = BTreeMap::new();
    for manifest in available {
        by_digest.insert(manifest.digest()?, manifest);
    }
    let mut visited = BTreeSet::new();
    let mut pending = vec![root];
    let root_digest = root.digest()?;
    while let Some(manifest) = pending.pop() {
        let digest = manifest.digest()?;
        if !visited.insert(digest) {
            continue;
        }
        require_artifact_destination(manifest, PublicationDestination::Public)?;
        if reader < manifest.minimum_reader_version
            || manifest
                .maximum_reader_version
                .as_ref()
                .is_some_and(|maximum| &reader > maximum)
        {
            return Err(CacheError::PermissionDenied(
                "public ancestry is incompatible with this reader".into(),
            ));
        }
        for dependency in &manifest.canonical_payload.dependencies {
            let parent = by_digest.get(&dependency.manifest_digest).ok_or_else(|| {
                CacheError::PermissionDenied(
                    "public admission requires the exact authenticated dependency closure".into(),
                )
            })?;
            if parent.artifact_family != dependency.artifact_family
                || parent.semantic_digest != dependency.semantic_digest
                || parent.payload_digest != dependency.payload_digest
            {
                return Err(CacheError::InvalidManifest(
                    "public dependency admission identity mismatch".into(),
                ));
            }
            pending.push(parent);
        }
    }
    Ok(visited
        .into_iter()
        .filter(|digest| digest != &root_digest)
        .map(|digest| by_digest[&digest].clone())
        .collect())
}

pub(crate) fn require_private_unadmitted(
    destination: PublicationDestination,
) -> Result<(), CacheError> {
    if destination == PublicationDestination::Public {
        return Err(CacheError::PermissionDenied(
            "public writes require artifact-bound publication admission".into(),
        ));
    }
    Ok(())
}

/// Validate the artifact, metadata policy and every scheduled payload part
/// before an artifact publication API may write to a public repository.
/// A changed/resumed journal must obtain a new permit under current policy.
pub fn admit_publication(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    public_sanitizer: Option<&PublicSanitizerProfile>,
) -> Result<PublicationAdmission, CacheError> {
    admit_publication_inner(journal, destination, bundle, public_sanitizer, None)
}

/// Admission with staged discovery documents available for an exact resume.
/// The legacy entry point fails closed for a public discovery plan.
pub fn admit_publication_with_staging(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    public_sanitizer: Option<&PublicSanitizerProfile>,
    staging_root: &std::path::Path,
) -> Result<PublicationAdmission, CacheError> {
    admit_publication_inner(
        journal,
        destination,
        bundle,
        public_sanitizer,
        Some(staging_root),
    )
}

fn admit_publication_inner(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    public_sanitizer: Option<&PublicSanitizerProfile>,
    staging_root: Option<&std::path::Path>,
) -> Result<PublicationAdmission, CacheError> {
    require_artifact_destination(&bundle.manifest, destination)?;
    if destination == PublicationDestination::Public {
        public_admission_closure(&bundle.manifest, &bundle.required_manifests)?;
    }
    journal.validate()?;
    crate::publication_staging::validate_bundle(journal, destination, bundle)?;
    crate::publication_staging::validate_public_documents(
        journal,
        destination,
        bundle,
        public_sanitizer,
    )?;
    let target = journal
        .targets
        .get(&destination)
        .ok_or_else(|| CacheError::InvalidManifest("publication admission has no target".into()))?;
    for part in target.batches.iter().flat_map(|batch| &batch.plan.parts) {
        if !bundle.encoding.ordered_parts.iter().any(|allowed| {
            allowed.repository_path == part.repository_path
                && allowed.content_digest == part.content_digest
                && allowed.size_bytes == part.size_bytes
        }) {
            return Err(CacheError::PermissionDenied(
                "publication payload part is outside the admitted artifact".into(),
            ));
        }
    }
    if let Some(metadata) = &target.metadata_commit {
        if metadata.files
            != crate::publication_staging::immutable_metadata_file_plan(
                journal,
                destination,
                bundle,
            )?
        {
            return Err(CacheError::PermissionDenied(
                "metadata plan is outside the admitted artifact".into(),
            ));
        }
    }
    for (sequence, batch) in target.batches.iter().enumerate() {
        if let Some(plan) = &batch.record_commit {
            let record = crate::build_payload_batch_record(journal, destination, sequence)?;
            let bytes = crate::protocol::canonical_json_bytes(&record)?;
            let expected = crate::TransportPart {
                sequence: 0,
                repository_path: record.repository_path(),
                content_digest: ContentDigest::sha256(&bytes),
                size_bytes: bytes.len() as u64,
            };
            if plan.files != [expected] {
                return Err(CacheError::PermissionDenied(
                    "batch record plan is outside the admitted artifact".into(),
                ));
            }
        }
    }
    if destination == PublicationDestination::Public && target.discoverability_commit.is_some() {
        let root = staging_root.ok_or_else(|| {
            CacheError::PermissionDenied(
                "public discovery admission requires staged documents".into(),
            )
        })?;
        validate_discovery_documents(journal, destination, bundle, root)?;
    }
    Ok(PublicationAdmission {
        destination,
        journal_digest: journal.digest()?,
    })
}

fn validate_discovery_documents(
    journal: &PublicationTransactionJournal,
    destination: PublicationDestination,
    bundle: &PublicationMetadataBundle,
    root: &std::path::Path,
) -> Result<(), CacheError> {
    use std::collections::BTreeMap;
    let target = &journal.targets[&destination];
    let plan = target.discoverability_commit.as_ref().unwrap();
    let index_path = format!(
        "indexes/{}/{}.json",
        bundle.family,
        &journal.semantic_digest.0[..2]
    );
    let receipt_path = format!(
        "transactions/{}/public/receipt.json",
        journal.transaction_id
    );
    if plan.files.len() != 3
        || ![
            &index_path,
            &receipt_path,
            &crate::DEFAULT_CAPACITY_LEDGER_PATH.to_owned(),
        ]
        .iter()
        .all(|path| plan.files.iter().any(|file| &file.repository_path == *path))
    {
        return Err(CacheError::PermissionDenied(
            "unexpected public discovery document".into(),
        ));
    }
    let read = |path: &str| -> Result<Vec<u8>, CacheError> {
        let file = plan
            .files
            .iter()
            .find(|file| file.repository_path == path)
            .unwrap();
        // Discovery metadata is bounded independently of numerical payloads.
        if file.size_bytes > 64 * 1024 * 1024 {
            return Err(CacheError::ResourceLimit(
                "discovery admission metadata exceeds 64 MiB".into(),
            ));
        }
        let base = root.canonicalize()?;
        let resolved = root.join(path).canonicalize()?;
        if !resolved.starts_with(&base) || std::fs::metadata(&resolved)?.len() != file.size_bytes {
            return Err(CacheError::PermissionDenied(
                "discovery staging path or length mismatch".into(),
            ));
        }
        let bytes = std::fs::read(resolved)?;
        if ContentDigest::sha256(&bytes) != file.content_digest {
            return Err(CacheError::PermissionDenied(
                "discovery document digest mismatch".into(),
            ));
        }
        Ok(bytes)
    };
    let index_bytes = read(&index_path)?;
    let index: crate::ShardIndexPartition = serde_json::from_slice(&index_bytes)?;
    index.validate()?;
    let manifest_digest = bundle.manifest.digest()?;
    let transport_digest = bundle.encoding.digest()?;
    if index.family != bundle.family
        || index.semantic_prefix != journal.semantic_digest.0[..2]
        || !index.entries.iter().any(|entry| {
            entry.semantic_digest == journal.semantic_digest
                && entry.manifest_digest == manifest_digest
                && entry.canonical_payload_digest == journal.payload_digest
                && entry.producer_toolkit_version == bundle.manifest.producer_toolkit_version
                && entry.minimum_reader_version == bundle.manifest.minimum_reader_version
                && entry.publication_transaction_id == journal.transaction_id
                && entry.transport_digests == [transport_digest.clone()]
        })
    {
        return Err(CacheError::PermissionDenied(
            "discovery index does not bind the admitted artifact".into(),
        ));
    }
    let receipt_bytes = read(&receipt_path)?;
    let receipt: crate::PublicationReceipt = serde_json::from_slice(&receipt_bytes)?;
    receipt.validate_for_transaction(journal, destination)?;
    if receipt.transport_digest != bundle.encoding.digest()?
        || plan.receipt_digest.as_ref() != Some(&receipt.digest()?)
        || receipt.discoverability_subject_digests
            != BTreeMap::from([(index_path, ContentDigest::sha256(&index_bytes))])
    {
        return Err(CacheError::PermissionDenied(
            "discovery receipt identity mismatch".into(),
        ));
    }
    let ledger_bytes = read(crate::DEFAULT_CAPACITY_LEDGER_PATH)?;
    let ledger: crate::CapacityLedger = serde_json::from_slice(&ledger_bytes)?;
    ledger.validate()?;
    if ledger.shard_id != target.shard_id
        || crate::protocol::canonical_json_bytes(&index)? != index_bytes
        || crate::protocol::canonical_json_bytes(&receipt)? != receipt_bytes
        || crate::protocol::canonical_json_bytes(&ledger)? != ledger_bytes
    {
        return Err(CacheError::PermissionDenied(
            "discovery documents contain unadmitted fields or encoding".into(),
        ));
    }
    xc_core::validate_secret_free(&(&index, &receipt, &ledger), "public discovery documents")
        .map_err(|_| {
            CacheError::PermissionDenied("public discovery documents failed sanitization".into())
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent_binding(parent: &CanonicalArtifactManifest) -> crate::PayloadDependencyIdentity {
        crate::PayloadDependencyIdentity {
            artifact_family: parent.artifact_family.clone(),
            semantic_digest: parent.semantic_digest.clone(),
            manifest_digest: parent.digest().unwrap(),
            payload_digest: parent.payload_digest.clone(),
        }
    }

    #[test]
    fn public_ancestry_requires_exact_admissible_producer_and_disclosure() {
        let parent = crate::content_model::tests::current_fixture();
        let mut child = parent.clone();
        child
            .canonical_payload
            .dependencies
            .push(parent_binding(&parent));
        assert_eq!(
            public_admission_closure(&child, std::slice::from_ref(&parent)).unwrap(),
            vec![parent.clone()]
        );
        assert!(public_admission_closure(&child, &[]).is_err());
        let mut different_producer = parent.clone();
        different_producer.producer_toolkit_version =
            crate::ToolkitVersion::parse("0.17.1").unwrap();
        assert_eq!(different_producer.payload_digest, parent.payload_digest);
        assert!(public_admission_closure(&child, &[different_producer]).is_err());
        for classification in [
            crate::ArtifactDisclosure::PrivateOnly,
            crate::ArtifactDisclosure::Unclassified,
        ] {
            let mut restricted = parent.clone();
            restricted.provenance.as_mut().unwrap().disclosure = classification;
            child.canonical_payload.dependencies = vec![parent_binding(&restricted)];
            assert!(public_admission_closure(&child, &[restricted]).is_err());
        }
        let mut target = parent.clone();
        target.artifact_family = "prolate".into();
        target.semantic_key.artifact_kind = "ccm_reference_source".into();
        target.semantic_digest = target.semantic_key.digest().unwrap();
        target.provenance.as_mut().unwrap().disclosure = crate::ArtifactDisclosure::PrivateOnly;
        child.canonical_payload.dependencies = vec![parent_binding(&target)];
        assert!(public_admission_closure(&child, &[target]).is_err());
        // Classification changes neither the numerical request nor values.
        let mut restricted = parent.clone();
        restricted.provenance.as_mut().unwrap().disclosure = crate::ArtifactDisclosure::PrivateOnly;
        assert_eq!(restricted.semantic_digest, parent.semantic_digest);
        assert_eq!(restricted.payload_digest, parent.payload_digest);
        assert!(!manifest_admitted_to_destination(
            &restricted,
            PublicationDestination::Public
        ));
    }

    #[test]
    fn permit_is_invalid_after_any_journal_change_or_destination_change() {
        let (mut journal, bundle) =
            crate::publication_staging::tests::fixture(PublicationDestination::Private);
        let permit =
            admit_publication(&journal, PublicationDestination::Private, &bundle, None).unwrap();
        assert!(permit
            .check(&journal, PublicationDestination::Private)
            .is_ok());
        assert!(permit
            .check(&journal, PublicationDestination::Public)
            .is_err());
        journal.policy_digest = ContentDigest::sha256(b"altered policy");
        assert!(permit
            .check(&journal, PublicationDestination::Private)
            .is_err());
    }

    #[test]
    fn private_root_and_private_ancestor_make_zero_public_calls_even_on_resume() {
        use crate::*;
        for private_root in [true, false] {
            let (mut journal, mut bundle) =
                crate::publication_staging::tests::fixture(PublicationDestination::Public);
            // The fixture is already past verified payload batching: this also
            // covers a resumed journal, whose earlier state is not permission.
            if private_root {
                bundle.manifest.semantic_key.artifact_kind = "ccm_reference_source".into();
                bundle.manifest.artifact_family = "ccm-evidence".into();
                bundle.manifest.semantic_digest = bundle.manifest.semantic_key.digest().unwrap();
            } else {
                let mut parent = crate::content_model::tests::current_fixture();
                parent.provenance.as_mut().unwrap().disclosure = ArtifactDisclosure::PrivateOnly;
                bundle.manifest.canonical_payload.dependencies = vec![parent_binding(&parent)];
                bundle.manifest.payload_digest =
                    bundle.manifest.canonical_payload.digest().unwrap();
                bundle.required_manifests = vec![parent];
            }
            let remote = crate::publisher::tests::FakeRemote::new();
            let root = crate::test_support::temporary_root("public-private-resume");
            let resources = xc_core::ResourcePolicy::default();
            let policy = PublicationFinalizationPolicy::default();
            let execution = PublicationTargetExecution {
                staging_root: &root,
                resources: &resources,
                finalization_policy: &policy,
                bundle: &bundle,
                public_sanitizer: None,
                maximum_index_bytes: 1024 * 1024,
                receipt_verified_at_unix_seconds: 1,
                replace_existing_semantic: false,
            };
            let before = journal.clone();
            let result = advance_publication_target(
                &remote,
                &PublicationJournalStore::new(root.path()),
                &xc_core::CancellationToken::new(),
                &AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    "team/public",
                    RepositoryPermission::Write,
                ),
                &mut journal,
                PublicationDestination::Public,
                &execution,
            );
            assert!(matches!(result, Err(CacheError::PermissionDenied(_))));
            assert_eq!(*remote.calls.lock().unwrap(), 0);
            assert_eq!(journal, before);
        }
    }
    #[test]
    fn resumed_public_metadata_and_batch_plans_cannot_add_or_replace_documents() {
        use crate::*;
        for attack in 0..3 {
            let (mut journal, bundle) =
                crate::publication_staging::tests::fixture(PublicationDestination::Public);
            let repository = journal.targets[&PublicationDestination::Public]
                .authorized_repository
                .clone();
            let sanitizer = PublicSanitizerProfile {
                allowed_leaf_fields: crate::managed_publication::public_leaf_allowlist(
                    &repository,
                    &bundle.manifest,
                    &bundle.encoding,
                    &bundle.validation_attestations[0],
                    &bundle.validator_evidence[0],
                )
                .unwrap(),
                allowed_repository_names: [repository.clone()].into_iter().collect(),
                ..PublicSanitizerProfile::default()
            };
            let root = crate::test_support::temporary_root("public-plan-tampering");
            let resources = xc_core::ResourcePolicy::default();
            let cancellation = xc_core::CancellationToken::new();
            stage_immutable_publication_metadata(
                &root,
                &resources,
                &cancellation,
                &mut journal,
                PublicationDestination::Public,
                &bundle,
                Some(&sanitizer),
            )
            .unwrap();
            admit_publication_with_staging(
                &journal,
                PublicationDestination::Public,
                &bundle,
                Some(&sanitizer),
                &root,
            )
            .unwrap();
            let target = journal
                .targets
                .get_mut(&PublicationDestination::Public)
                .unwrap();
            match attack {
                0 => {
                    target.metadata_commit.as_mut().unwrap().files[0].content_digest =
                        ContentDigest::sha256(b"private replacement")
                }
                1 => {
                    target.metadata_commit.as_mut().unwrap().files[0].repository_path =
                        "manifests/extra-private.json".into()
                }
                _ => {
                    target.batches[0].record_commit.as_mut().unwrap().files[0].content_digest =
                        ContentDigest::sha256(b"private batch")
                }
            }
            let remote = crate::publisher::tests::FakeRemote::new();
            let finalization_policy = PublicationFinalizationPolicy::default();
            let execution = PublicationTargetExecution {
                staging_root: &root,
                resources: &resources,
                finalization_policy: &finalization_policy,
                bundle: &bundle,
                public_sanitizer: Some(&sanitizer),
                maximum_index_bytes: 1024 * 1024,
                receipt_verified_at_unix_seconds: 1,
                replace_existing_semantic: false,
            };
            let before = journal.clone();
            assert!(advance_publication_target(
                &remote,
                &PublicationJournalStore::new(root.path()),
                &cancellation,
                &AuthenticatedGitHubSession::verified_for_test(
                    "test-owner",
                    &repository,
                    RepositoryPermission::Write
                ),
                &mut journal,
                PublicationDestination::Public,
                &execution
            )
            .is_err());
            assert_eq!(*remote.calls.lock().unwrap(), 0);
            assert_eq!(journal, before);
        }
    }

    #[test]
    fn unknown_kind_is_never_public_admission_evidence() {
        let mut manifest = crate::content_model::tests::current_fixture();
        manifest.schema_version = 1;
        manifest.canonical_payload.schema_version = 1;
        manifest.provenance = None;
        manifest.semantic_key.artifact_kind = "unknown_public_producer".into();
        manifest.semantic_digest = manifest.semantic_key.digest().unwrap();
        manifest.payload_digest = manifest.canonical_payload.digest().unwrap();
        assert!(!manifest_admitted_to_destination(
            &manifest,
            PublicationDestination::Public
        ));
    }
}
