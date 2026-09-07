//! Exercise managed receipt reuse with the actual GitHub adapter constructor.
//! Fixture transport is replaced with in-memory I/O after adapter construction;
//! source payload reads are forbidden to check metadata-only validation.
use super::*;
use serde_json::json;

struct ReceiptStore {
    root: ArtifactManifest,
    source: Option<ArtifactManifest>,
    payload: Vec<u8>,
}
impl CacheStore for ReceiptStore {
    fn name(&self) -> &str {
        "github-private-fixture"
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        CacheVisibility::Private
    }
    fn put(&self, _: &ArtifactDraft, _: &[u8]) -> Result<ArtifactManifest, CacheError> {
        panic!("receipt reuse must not write or recompute")
    }
    fn candidates(&self, key: &ArtifactKey) -> Result<Vec<ArtifactManifest>, CacheError> {
        Ok(if key == &self.root.key {
            vec![self.root.clone()]
        } else {
            vec![]
        })
    }
    fn identity_candidates(
        &self,
        _: &PayloadDependencyIdentity,
    ) -> Result<Vec<ArtifactManifest>, CacheError> {
        Ok(self.source.clone().into_iter().collect())
    }
    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn Write,
    ) -> Result<(), CacheError> {
        assert_eq!(
            manifest.key.kind, CAPTURE_RECEIPT_KIND,
            "source payload must not be read"
        );
        writer.write_all(&self.payload)?;
        Ok(())
    }
}

struct Fixture {
    directory: PathBuf,
    record: CaptureArtifact,
    root: ArtifactManifest,
    source: ArtifactManifest,
    source_identity: PayloadDependencyIdentity,
    payload: Vec<u8>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

fn policy() -> CachePolicy {
    CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Private, CacheVisibility::Local],
    }
}
fn context<'a>(
    resolver: &'a CacheResolver,
    policy: &'a CachePolicy,
    mode: ArtifactExecutionCacheMode,
) -> ArtifactCacheContext<'a> {
    ArtifactCacheContext {
        resolver: Some(resolver),
        reference_resolver: None,
        acceptance: Some(policy),
        ordered_overlays: vec!["receipt-fixture".into()],
        mode,
        write_on_miss: !mode.requires_reuse(),
        write_visibility: CacheVisibility::Private,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    }
}
fn rewrite_canonical(
    manifest: &mut ArtifactManifest,
    edit: impl FnOnce(&mut CanonicalArtifactManifest),
) {
    let mut canonical: CanonicalArtifactManifest =
        serde_json::from_str(&manifest.tags[REMOTE_CANONICAL_MANIFEST_TAG]).unwrap();
    edit(&mut canonical);
    canonical.payload_digest = canonical.canonical_payload.digest().unwrap();
    manifest.provenance_digest = Some(canonical.digest().unwrap());
    manifest.tags.insert(
        REMOTE_CANONICAL_MANIFEST_TAG.into(),
        serde_json::to_string(&canonical).unwrap(),
    );
}

impl Fixture {
    fn new(aliases: bool) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "xc-receipt-adapter-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        let adapter =
            GitHubBootstrapCacheStore::public("fixture-owner", directory.join("remote")).unwrap();
        let mut source_artifact = super::tests::resolved_fixture(&directory, "source", vec![]);
        let source_key = ArtifactKey {
            kind: "ccm_tau_matrix".into(),
            logical_key: "fixture/source".into(),
            parameters_digest: source_artifact.semantic_digest.clone(),
        };
        let mut source = adapter
            .adapter_manifest_for(source_key, &source_artifact)
            .unwrap();
        source.visibility = CacheVisibility::Private;
        let source_identity = PayloadDependencyIdentity {
            artifact_family: source_artifact.manifest.artifact_family.clone(),
            semantic_digest: source_artifact.manifest.semantic_digest.clone(),
            manifest_digest: source_artifact.manifest.digest().unwrap(),
            payload_digest: source_artifact.manifest.payload_digest.clone(),
        };
        let mut sources = vec![source.clone()];
        if aliases {
            let mut alias = source.clone();
            alias.key.logical_key = "fixture/alias".into();
            sources.push(alias);
        }
        let record = collect_capture(
            &json!({"fixture":"remote-receipt"}),
            vec!["source".into(), "unavailable".into()],
            |id| {
                if id == "source" {
                    CapturedDiagnostic::new(&json!(42), sources.clone())
                        .map_err(CaptureFailure::failed)
                } else {
                    Err(CaptureFailure::Missing {
                        reason: "retained missing result".into(),
                    })
                }
            },
        )
        .unwrap();
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "cold",
                directory.join("cold"),
                true,
                CacheVisibility::Private,
            )),
        }]);
        let cold = persist_capture(
            &record,
            &context(
                &resolver,
                &policy(),
                ArtifactExecutionCacheMode::PreferReuse,
            ),
        )
        .unwrap()
        .produced_manifest
        .unwrap();
        let payload = serde_json::to_vec(&record).unwrap();
        assert_eq!(ContentDigest::sha256(&payload), cold.content_digest);
        let semantic: SemanticKeyEnvelope =
            serde_json::from_str(&cold.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
        source_artifact.manifest.semantic_key = semantic.clone();
        source_artifact.manifest.semantic_digest = semantic.digest().unwrap();
        source_artifact.manifest.artifact_family = "ccm-evidence".into();
        source_artifact.manifest.producer_toolkit_version = cold.producer_toolkit_version.clone();
        source_artifact.manifest.minimum_reader_version = cold.minimum_reader_version.clone();
        source_artifact.manifest.maximum_reader_version = cold.maximum_reader_version.clone();
        source_artifact
            .manifest
            .resolved_mathematical_configuration_digest = ContentDigest(
            xc_core::research_digest(&semantic.resolved_mathematical_parameters)
                .unwrap()
                .0,
        );
        source_artifact.manifest.canonical_payload.dimensions = vec![payload.len() as u64];
        source_artifact.manifest.canonical_payload.ordered_items[0].content_digest =
            cold.content_digest;
        source_artifact.manifest.canonical_payload.ordered_items[0].size_bytes =
            payload.len() as u64;
        source_artifact.manifest.canonical_payload.dependencies = vec![source_identity.clone()];
        source_artifact.manifest.payload_digest =
            source_artifact.manifest.canonical_payload.digest().unwrap();
        let mut root = adapter
            .adapter_manifest_for(cold.key, &source_artifact)
            .unwrap();
        root.visibility = CacheVisibility::Private;
        assert!(root.dependencies.is_empty());
        assert!(!record.source_dependencies.is_empty());
        Self {
            directory,
            record,
            root,
            source,
            source_identity,
            payload,
        }
    }
    fn reuse(
        &self,
        root: ArtifactManifest,
        source: Option<ArtifactManifest>,
    ) -> Result<ArtifactExecutionCacheResult<CaptureArtifact>, CacheError> {
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(ReceiptStore {
                root,
                source,
                payload: self.payload.clone(),
            }),
        }]);
        persist_capture(
            &self.record,
            &context(
                &resolver,
                &policy(),
                ArtifactExecutionCacheMode::RequireReuse,
            ),
        )
    }
}

#[test]
fn published_receipt_reuse_validates_canonical_sources_without_payload_reads() {
    for aliases in [false, true] {
        let fixture = Fixture::new(aliases);
        let result = fixture
            .reuse(fixture.root.clone(), Some(fixture.source.clone()))
            .unwrap();
        assert_eq!(result.value, fixture.record);
        assert!(result.produced_manifest.is_none());
        // The same canonical adapter may already have been adopted locally.
        let mut local = fixture.root.clone();
        local.visibility = CacheVisibility::Local;
        assert_eq!(
            fixture
                .reuse(local, Some(fixture.source.clone()))
                .unwrap()
                .value,
            fixture.record
        );
    }
}

#[test]
fn published_receipt_reuse_rejects_missing_mismatched_and_weak_sources() {
    let fixture = Fixture::new(false);
    assert!(fixture.reuse(fixture.root.clone(), None).is_err());
    let mut weak = fixture.source.clone();
    weak.quality = CacheQuality::Staged;
    assert!(fixture.reuse(fixture.root.clone(), Some(weak)).is_err());
    let mut local = fixture.root.clone();
    local.tags.remove(REMOTE_CANONICAL_MANIFEST_TAG);
    assert!(fixture
        .reuse(local, Some(fixture.source.clone()))
        .err()
        .unwrap()
        .to_string()
        .contains("dependency closure mismatch"));
    let mut missing = fixture.root.clone();
    rewrite_canonical(&mut missing, |m| m.canonical_payload.dependencies.clear());
    assert!(fixture
        .reuse(missing, Some(fixture.source.clone()))
        .is_err());
    let mut wrong = fixture.root.clone();
    rewrite_canonical(&mut wrong, |m| {
        m.canonical_payload.dependencies[0].manifest_digest =
            ContentDigest::sha256(b"wrong manifest")
    });
    assert!(fixture.reuse(wrong, Some(fixture.source.clone())).is_err());
    let mut unbound = fixture.root.clone();
    unbound.provenance_digest = None;
    assert!(fixture
        .reuse(unbound, Some(fixture.source.clone()))
        .is_err());
    let mut wrong_source = fixture.source.clone();
    wrong_source.content_digest = ContentDigest::sha256(b"different bytes");
    wrong_source.objects[0].content_digest = wrong_source.content_digest.clone();
    let wrong_digest = wrong_source.content_digest.clone();
    rewrite_canonical(&mut wrong_source, |m| {
        m.canonical_payload.ordered_items[0].content_digest = wrong_digest
    });
    let mut wrong_root = fixture.root.clone();
    let mut identity = fixture.source_identity.clone();
    let source_canonical: CanonicalArtifactManifest =
        serde_json::from_str(&wrong_source.tags[REMOTE_CANONICAL_MANIFEST_TAG]).unwrap();
    identity.manifest_digest = source_canonical.digest().unwrap();
    identity.payload_digest = source_canonical.payload_digest;
    rewrite_canonical(&mut wrong_root, |m| {
        m.canonical_payload.dependencies = vec![identity]
    });
    assert!(fixture
        .reuse(wrong_root, Some(wrong_source))
        .err()
        .unwrap()
        .to_string()
        .contains("canonical source identity mismatch"));
}
