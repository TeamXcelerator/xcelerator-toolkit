use super::*;
use xc_cache::{
    ArtifactExecutionCacheMode, CacheLayer, CacheObjectRef, CachePolicy, CacheResolver,
    CacheVisibility, CertificationFailurePolicy, EphemeralCacheStore, SEMANTIC_KEY_MANIFEST_TAG,
};

fn replace_semantic(m: &mut ArtifactManifest, s: &SemanticKeyEnvelope) {
    m.key.parameters_digest = s.digest().unwrap();
    m.tags.insert(
        SEMANTIC_KEY_MANIFEST_TAG.into(),
        serde_json::to_string(s).unwrap(),
    );
}

#[test]
fn current_producer_recovers_roots_with_numerical_and_legacy_tau_identities() {
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(EphemeralCacheStore::new(64 << 20).unwrap()),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let cache = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["ephemeral".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let params = CcmParams::from_lambda_sq_integer(13, 16);
    let mut cfg = HighPrecConfig::for_decimal_digits(40).with_adaptive_root_precision();
    cfg.n_eigenvalues = 1;
    cfg.quad_points = 32;
    let dataset = xc_zeta::zeros::bundled_dataset_identity().unwrap();
    let seeds = vec![Float::with_val(
        cfg.precision_bits,
        Float::parse(&xc_zeta::zeros::bundled_first_n_strings(1).unwrap()[0]).unwrap(),
    )];
    let run =
        capture_run::RetainedCcmRun::seeded(&params, &cfg, 1, &seeds, &dataset, &cache).unwrap();
    let original = PortableHighPrecResult::from_runtime(run.primary()).unwrap();
    let sources = run.primary_sources();
    let get = |kind| sources.iter().find(|m| m.key.kind == kind).unwrap().clone();
    let tau = get("ccm_tau_matrix");
    let ep = get("ccm_weil_eigenpair");
    let sec = get("ccm_secular_source");
    let ep_bytes = resolver
        .read_exact_payload(&ep.key, &ep.content_digest, &policy)
        .unwrap();
    let sec_bytes = resolver
        .read_exact_payload(&sec.key, &sec.content_digest, &policy)
        .unwrap();
    for legacy in [false, true] {
        let mut ep = ep.clone();
        let mut sec = sec.clone();
        let mut sec_bytes = sec_bytes.clone();
        if legacy {
            let mut s: SemanticKeyEnvelope =
                serde_json::from_str(&ep.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
            s.resolved_mathematical_parameters["tau_source_identity"] =
                serde_json::to_value(DependencyRef {
                    key: tau.key.clone(),
                    content_digest: tau.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                })
                .unwrap();
            replace_semantic(&mut ep, &s);
            let mut s: SemanticKeyEnvelope =
                serde_json::from_str(&sec.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
            s.resolved_mathematical_parameters["eigenpair_semantic_digest"] =
                serde_json::json!(ep.key.parameters_digest);
            replace_semantic(&mut sec, &s);
            let mut payload: serde_json::Value = serde_json::from_slice(&sec_bytes).unwrap();
            payload["eigenpair_semantic_digest"] = serde_json::json!(ep.key.parameters_digest);
            sec_bytes = serde_json::to_vec(&payload).unwrap();
            sec.content_digest = ContentDigest::sha256(&sec_bytes);
            sec.size_bytes = sec_bytes.len() as u64;
            sec.objects = vec![CacheObjectRef {
                content_digest: sec.content_digest.clone(),
                size_bytes: sec.size_bytes,
            }];
            sec.dependencies = vec![DependencyRef {
                key: ep.key.clone(),
                content_digest: ep.content_digest.clone(),
                required_quality: CacheQuality::Validated,
            }];
        }
        let allowed = vec![
            tau.content_digest.clone(),
            ep.content_digest.clone(),
            sec.content_digest.clone(),
        ];
        let recover = |tau: &ArtifactManifest, ep: &ArtifactManifest, bytes: &[u8]| {
            RetainedSeededRootSource::from_payloads(
                &params, &cfg, &original, tau, ep, bytes, &sec, &sec_bytes, &allowed,
            )
        };
        let recovered = recover(&tau, &ep, &ep_bytes).unwrap();
        let mut denied = policy.clone();
        denied.allowed_visibilities = vec![CacheVisibility::Private];
        let error = RetainedSeededRootSource::from_payloads_with_policy(
            &params, &cfg, &original, &tau, &ep, &ep_bytes, &sec, &sec_bytes, &allowed, &denied,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("visibility"), "{error}");
        assert_eq!(recovered.original().xi, original.xi);
        if !legacy {
            let (roots, _) = recovered.refine(1, &seeds, &dataset, &cache).unwrap();
            assert_eq!(roots.len(), 1);
            assert!(matches!(roots[0], EigenvalueResult::Converged(_)));
        }
        let mut variant = tau.clone();
        variant.key.logical_key = "adopted/tau".into();
        variant.producer_toolkit_version = ToolkitVersion::parse("0.17.1").unwrap();
        recover(&variant, &ep, &ep_bytes).unwrap();
        variant.minimum_reader_version = ToolkitVersion::parse("999.0.0").unwrap();
        assert!(recover(&variant, &ep, &ep_bytes).is_err());
        let mut changed = ep_bytes.clone();
        changed.push(b' ');
        assert!(recover(&tau, &ep, &changed).is_err());
        let mut changed = ep.clone();
        let mut s: SemanticKeyEnvelope =
            serde_json::from_str(&changed.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
        s.resolved_mathematical_parameters["precision_bits"] =
            serde_json::json!(cfg.precision_bits + 1);
        replace_semantic(&mut changed, &s);
        assert!(recover(&tau, &changed, &ep_bytes).is_err());
    }
}
