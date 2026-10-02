//! Fresh public API regressions for cache assurance and dependency graph contracts.
use std::collections::BTreeMap;
use std::io::Write;
use xc_cache::*;

fn key(name: &str) -> ArtifactKey {
    ArtifactKey::new("fixture", name, name.as_bytes()).unwrap()
}
fn manifest(
    name: &str,
    quality: CacheQuality,
    dependencies: Vec<DependencyRef>,
) -> ArtifactManifest {
    let digest = ContentDigest::sha256(name.as_bytes());
    ArtifactManifest {
        schema_version: 1,
        key: key(name),
        content_digest: digest.clone(),
        size_bytes: name.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: name.len() as u64,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: ToolkitVersion::parse("0.18.2").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
        maximum_reader_version: None,
        quality,
        visibility: CacheVisibility::Public,
        immutable: true,
        dependencies,
        tags: BTreeMap::new(),
        provenance_digest: None,
    }
}
fn policy(minimum_quality: CacheQuality) -> CachePolicy {
    CachePolicy {
        current_toolkit_version: ToolkitVersion::parse("0.18.2").unwrap(),
        minimum_quality,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Public],
    }
}

#[test]
fn publication_without_certification_must_not_satisfy_a_certified_read() {
    let promotion = CachePromotionRequest {
        source_manifest_digest: ContentDigest::sha256(b"computed source manifest"),
        source_quality: CacheQuality::Validated,
        target_quality: CacheQuality::Published,
        source_visibility: CacheVisibility::Public,
        target_visibility: CacheVisibility::Public,
        reviews: vec![],
    };
    CachePromotionPolicy {
        minimum_unique_approvals: 0,
        require_certified_before_publication: false,
        allow_private_to_public: false,
    }
    .validate_request(&promotion)
    .unwrap();
    let published = manifest("computed-only", CacheQuality::Published, vec![]);
    published.validate().unwrap();
    assert!(!policy(CacheQuality::Certified).accepts(&published),
        "the compatibility cache ranks publication above certification despite no certificate evidence");
}

struct MetadataStore(Vec<ArtifactManifest>);
impl CacheStore for MetadataStore {
    fn name(&self) -> &str {
        "fresh-metadata-fixture"
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        CacheVisibility::Public
    }
    fn put(&self, _: &ArtifactDraft, _: &[u8]) -> Result<ArtifactManifest, CacheError> {
        Err(CacheError::ReadOnlyLayer(self.name().into()))
    }
    fn candidates(&self, k: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        Ok(self.0.iter().filter(|m| &m.key == k).cloned().collect())
    }
    fn read_payload_to(&self, _: &ArtifactManifest, _: &mut dyn Write) -> Result<(), CacheError> {
        panic!("metadata closure validation must not read payloads")
    }
}

#[test]
fn dependency_closure_must_reject_a_cycle() {
    let dependency = |name: &str| DependencyRef {
        key: key(name),
        content_digest: ContentDigest::sha256(name.as_bytes()),
        required_quality: CacheQuality::Validated,
    };
    let a = manifest("a", CacheQuality::Validated, vec![dependency("b")]);
    let b = manifest("b", CacheQuality::Validated, vec![dependency("a")]);
    a.validate().unwrap();
    b.validate().unwrap();
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(MetadataStore(vec![a.clone(), b])),
    }]);
    let result = resolver.validate_dependency_closure(&a, &policy(CacheQuality::Validated));
    assert!(
        result.is_err(),
        "cyclic dependency provenance returned a successful closure: {result:?}"
    );
}

#[test]
fn publication_plan_must_bind_capacity_to_the_actual_repository() {
    let capacity = CacheRepositoryRegistry {
        schema_version: 1,
        shards: vec![RepositoryShard {
            id: "same-id".into(),
            repository: "owner/measured-empty".into(),
            visibility: CacheVisibility::Public,
            artifact_kinds: vec!["fixture".into()],
            reachable_payload_bytes: 0,
            estimated_history_bytes: 0,
            safe_payload_limit_bytes: 1000,
            writable: true,
        }],
    };
    let network = CacheNetworkRegistry {
        schema_version: 1,
        repositories: vec![GitHubRepositoryEndpoint {
            shard_id: "same-id".into(),
            owner: "owner".into(),
            repository: "different-unmeasured".into(),
            branch: "main".into(),
            visibility: CacheVisibility::Public,
            enabled_for_read: true,
            enabled_for_write: true,
            clone_via_ssh: false,
        }],
    };
    let result = plan_github_publication(
        &capacity,
        &network,
        "fixture",
        CacheVisibility::Public,
        900,
        0,
        false,
    );
    assert!(
        result.is_err(),
        "capacity of one repository authorized a different destination: {result:?}"
    );
}

#[test]
fn git_metadata_fetch_must_not_succeed_after_exceeding_its_disk_limit() {
    use std::{fs, path::Path, process::Command};
    fn git(at: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .current_dir(at)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "local fixture git failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().into()
    }
    fn bytes(path: &Path) -> u64 {
        fs::read_dir(path)
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                if e.file_type().unwrap().is_dir() {
                    bytes(&e.path())
                } else {
                    e.metadata().unwrap().len()
                }
            })
            .sum()
    }
    // This is a new, exclusively owned local fixture. No existing checkout,
    // external remote, credentials, push, or GitHub workflow is used.
    let scratch = xc_core::test_support::TestDir::new("fresh-fetch-budget");
    let root = scratch.join("root");
    fs::create_dir(&root).unwrap();
    let source = root.join("source");
    fs::create_dir(&source).unwrap();
    git(&source, &["init", "--initial-branch=main"]);
    fs::write(source.join("tiny.json"), b"{}").unwrap();
    let mut state = 123456789u64;
    let data: Vec<u8> = (0..2 * 1024 * 1024)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    fs::write(source.join("unrequested.bin"), data).unwrap();
    git(&source, &["add", "."]);
    git(
        &source,
        &[
            "-c",
            "user.name=FreshAudit",
            "-c",
            "user.email=audit@example.invalid",
            "commit",
            "-m",
            "isolated fixture",
        ],
    );
    let revision = git(&source, &["rev-parse", "HEAD"]);
    let temporary = root.join("transport");
    let limit = 1024 * 1024;
    let store = GitCliRemoteStore::new(
        &temporary,
        root.join("staged"),
        "FreshAudit",
        "audit@example.invalid",
    )
    .unwrap()
    .with_resource_policy(xc_core::ResourcePolicy {
        maximum_temporary_disk_bytes: Some(limit),
        maximum_transfer_bytes: Some(1024),
        ..xc_core::ResourcePolicy::default()
    });
    let mut output = Vec::new();
    let result = store.read_committed_path(
        &source.to_string_lossy(),
        &revision,
        "tiny.json",
        1024,
        &xc_core::CancellationToken::default(),
        &mut output,
    );
    let retained = bytes(&temporary);
    // Cleanup only the newly created fixture, including when the assertion fails.
    drop(store);
    fs::remove_dir_all(&root).unwrap();
    assert!(result.is_err() || retained <= limit,
        "successful tiny read retained {retained} Git bytes above its {limit}-byte disk ceiling: {result:?}");
}

#[test]
fn dependency_minima_must_not_erase_an_incomparable_publication_policy() {
    let parent = manifest("certified-parent", CacheQuality::Certified, vec![]);
    let root = manifest(
        "published-root",
        CacheQuality::Published,
        vec![DependencyRef {
            key: parent.key.clone(),
            content_digest: parent.content_digest.clone(),
            required_quality: CacheQuality::Certified,
        }],
    );
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(MetadataStore(vec![parent])),
    }]);
    assert!(resolver
        .validate_dependency_closure(&root, &policy(CacheQuality::Published))
        .is_err());
}

#[test]
fn supplied_sources_must_satisfy_the_original_acceptance_minimum() {
    let parent = manifest("certified-parent", CacheQuality::Certified, vec![]);
    let root = manifest(
        "root",
        CacheQuality::Validated,
        vec![DependencyRef {
            key: parent.key.clone(),
            content_digest: parent.content_digest.clone(),
            required_quality: CacheQuality::Certified,
        }],
    );
    let published_policy = policy(CacheQuality::Published);
    let certified_policy = policy(CacheQuality::Certified);
    let mut context = ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: Some(&published_policy),
        ordered_overlays: vec!["fixture".into()],
        mode: ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let parents = [parent];
    assert!(resolve_manifest_sources(&root, &parents, &context).is_err());
    context.acceptance = Some(&certified_policy);
    assert_eq!(
        resolve_manifest_sources(&root, &parents, &context)
            .unwrap()
            .len(),
        1
    );
}
