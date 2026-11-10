use super::*;
use std::sync::Mutex;
use xc_cache::{
    ArtifactAssuranceAttestation, ArtifactAssuranceRequirement, ArtifactDraft,
    ArtifactExecutionCacheMode, ArtifactProductionSink, CacheLayer, CachePolicy, CacheResolver,
    CacheStore, CacheVisibility, ProducedArtifactRecord, ZipJsonFilesystemCacheStore,
};

#[derive(Default)]
struct Sink {
    records: Mutex<Vec<ProducedArtifactRecord>>,
    evidence: Mutex<Vec<Vec<u8>>>,
}
impl ArtifactProductionSink for Sink {
    fn record(&self, artifact: ProducedArtifactRecord) -> Result<(), CacheError> {
        self.records.lock().unwrap().push(artifact);
        Ok(())
    }
    fn record_assurance_requirement(
        &self,
        _: ArtifactAssuranceRequirement,
    ) -> Result<(), CacheError> {
        Ok(())
    }
    fn record_assurance(&self, _: ArtifactAssuranceAttestation) -> Result<(), CacheError> {
        Ok(())
    }
    fn record_evidence(&self, _: &str, bytes: &[u8]) -> Result<ContentDigest, CacheError> {
        self.evidence.lock().unwrap().push(bytes.to_vec());
        Ok(ContentDigest::sha256(bytes))
    }
}

fn guide(store: &ZipJsonFilesystemCacheStore, kind: &str, label: &str) -> ArtifactManifest {
    let key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: kind.into(),
        mathematical_semantics_version: "test-guide-v1".into(),
        resolved_mathematical_parameters: serde_json::json!({"fixture": label}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: Some("test-guide-v1".into()),
    };
    let version = ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap();
    store
        .put(
            &ArtifactDraft {
                schema_version: 1,
                key: xc_cache::ArtifactKey {
                    kind: kind.into(),
                    logical_key: label.into(),
                    parameters_digest: key.digest().unwrap(),
                },
                producer_toolkit_version: version.clone(),
                minimum_reader_version: version,
                maximum_reader_version: None,
                quality: CacheQuality::Validated,
                visibility: CacheVisibility::Local,
                immutable: true,
                dependencies: vec![],
                tags: BTreeMap::from([(
                    xc_cache::SEMANTIC_KEY_MANIFEST_TAG.into(),
                    serde_json::to_string(&key).unwrap(),
                )]),
                provenance_digest: None,
            },
            &serde_json::to_vec(&serde_json::json!({"fixture":label})).unwrap(),
        )
        .unwrap()
}

#[test]
fn compact_managed_producer_stages_only_compact_evidence_and_reuses_exact_payload() {
    let scratch = xc_core::test_support::TestDir::new("compact-sector-cache");
    let store = ZipJsonFilesystemCacheStore::new(
        "local",
        scratch.to_path_buf(),
        true,
        CacheVisibility::Local,
    );
    let even = guide(&store, "ccm_sector_spectrum", "even-fixture");
    let odd = guide(&store, "ccm_sector_spectrum", "odd-fixture");
    let gap = guide(&store, "ccm_sector_gap", "gap-fixture");
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(store),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let sink = Sink::default();
    let mut cache = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: Some(&sink),
    };
    let run = |cache: &ArtifactCacheContext<'_>| {
        resolve_or_compute(
            &crate::ccm::CcmParams::from_lambda_sq_integer(5, 2),
            256,
            &Float::with_val(256, 1),
            &Float::with_val(256, 2),
            &Float::with_val(256, 3),
            &even,
            &odd,
            &gap,
            CcmSectorGapCertificationOptions::default(),
            ReconstructionSourceIdentity::current("test-retained-source"),
            cache,
        )
        .unwrap()
    };
    let first = run(&cache);
    let manifest = first.produced_manifest.as_ref().unwrap();
    assert_eq!(
        manifest.minimum_reader_version,
        ToolkitVersion::parse("0.17.1").unwrap()
    );
    assert!(manifest
        .key
        .logical_key
        .starts_with("ccm/sector-gap-replay-certificate/"));
    assert!(xc_cache::json_payload_matches_manifest(manifest, &first.value).unwrap());
    assert_eq!(manifest.dependencies.len(), 3);
    let bytes = serde_json::to_vec(&first.value).unwrap();
    let evidence = sink.evidence.lock().unwrap();
    assert_eq!(*evidence, vec![bytes.clone()]);
    drop(evidence);
    let records = sink.records.lock().unwrap();
    let certificate_records: Vec<_> = records
        .iter()
        .filter(|r| r.manifest.key.kind == "ccm_sector_gap_certificate")
        .collect();
    assert_eq!(certificate_records.len(), 1);
    assert_eq!(certificate_records[0].payload, bytes);
    assert_eq!(
        certificate_records[0].achieved_assurance,
        ArtifactAssuranceState::Certified
    );
    assert!(serde_json::from_slice::<PortableCcmSectorGapCertificate>(&bytes).is_err());
    drop(records);
    cache.mode = ArtifactExecutionCacheMode::RequireReuse;
    cache.write_on_miss = false;
    let second = run(&cache);
    assert_eq!(first.value, second.value);
    assert_eq!(
        manifest.content_digest,
        second.reused_manifest.as_ref().unwrap().content_digest
    );
    cache.requested_assurance = xc_core::AssuranceLevel::Certified;
    let verified_reuse = run(&cache);
    assert_eq!(first.value, verified_reuse.value);
    // Producer provenance is retained historical evidence, not a reason to
    // redo expensive compatible mathematics after an implementation rebuild.
    let newer_source = resolve_or_compute(
        &crate::ccm::CcmParams::from_lambda_sq_integer(5, 2),
        256,
        &Float::with_val(256, 1),
        &Float::with_val(256, 2),
        &Float::with_val(256, 3),
        &even,
        &odd,
        &gap,
        CcmSectorGapCertificationOptions::default(),
        ReconstructionSourceIdentity {
            toolkit_revision: "another-compatible-build".into(),
            source_tree_digest: ContentDigest::sha256(b"another-compatible-build"),
            build_identity: "another-compatible-build".into(),
        },
        &cache,
    )
    .unwrap();
    assert_eq!(first.value, newer_source.value);
    assert_eq!(
        manifest.content_digest,
        newer_source.reused_manifest.unwrap().content_digest
    );
    assert_eq!(
        sink.evidence.lock().unwrap().len(),
        1,
        "reuse must not produce a second full certificate"
    );
}
