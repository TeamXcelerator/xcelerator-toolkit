// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

use super::*;
use std::sync::{Arc, Mutex};
use xc_cache::{
    ArtifactDraft, CacheLayer, CachePolicy, CacheResolver, CacheStore, CacheVisibility,
};

#[test]
fn dense_lu_primary_keys_follow_actual_sector_dimension() {
    let source = super::tests::conditioning_test_manifest("ccm_tau_matrix", "lu-policy-source");
    for (modes, parity, dimension) in [
        (1, CcmParityPolicy::EvenSector, 2),
        (24, CcmParityPolicy::EvenSector, 25),
        (127, CcmParityPolicy::Natural, 255),
        (128, CcmParityPolicy::Natural, 257),
        (128, CcmParityPolicy::AdaptiveEven, 257),
        (254, CcmParityPolicy::EvenSector, 255),
        (255, CcmParityPolicy::EvenSector, 256),
    ] {
        let params = CcmParams::from_lambda_sq_integer(13, modes);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        cfg.set_parity_policy(parity);
        let (eigenpair, _) = weil_eigenpair_cache_identity(&params, &cfg, &source).unwrap();
        let (factor, _) = factorization_cache_identity(
            &params,
            &cfg,
            &source,
            if parity == CcmParityPolicy::EvenSector {
                "even"
            } else {
                "full"
            },
        )
        .unwrap();
        let expected = xc_numerics::linalg::dense_lu_arithmetic(dimension);
        assert_eq!(
            expected,
            Some("dense_lu_partial_pivot_dimension_panel512_exact_schur_v1")
        );
        for semantic in [eigenpair, factor] {
            assert_eq!(
                semantic
                    .resolved_mathematical_parameters
                    .get("dense_lu_arithmetic")
                    .and_then(serde_json::Value::as_str),
                expected
            );
            let mut legacy = semantic.clone();
            legacy
                .resolved_mathematical_parameters
                .as_object_mut()
                .unwrap()
                .remove("dense_lu_arithmetic");
            assert_ne!(semantic.digest().unwrap(), legacy.digest().unwrap());
        }
        if parity == CcmParityPolicy::EvenSector {
            cfg.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
            let (key, _) = weil_eigenpair_cache_identity(&params, &cfg, &source).unwrap();
            assert_eq!(
                key.resolved_mathematical_parameters
                    .get("dense_lu_arithmetic")
                    .and_then(serde_json::Value::as_str),
                expected
            );
        }
    }
}

struct PrimaryLookup {
    payload: Vec<u8>,
    seen: Arc<Mutex<Vec<ArtifactKey>>>,
    tau: Arc<Mutex<Option<ArtifactManifest>>>,
}
impl CacheStore for PrimaryLookup {
    fn name(&self) -> &str {
        "primary-lookup-test"
    }
    fn writable(&self) -> bool {
        false
    }
    fn visibility(&self) -> CacheVisibility {
        CacheVisibility::Local
    }
    fn put(
        &self,
        _: &ArtifactDraft,
        _: &[u8],
    ) -> std::result::Result<ArtifactManifest, CacheError> {
        panic!("RequireReuse primary lookup must not write or compute")
    }
    fn candidates(
        &self,
        key: &ArtifactKey,
    ) -> std::result::Result<Vec<ArtifactManifest>, CacheError> {
        self.seen.lock().unwrap().push(key.clone());
        if key.kind != "ccm_tau_matrix" {
            return Ok(Vec::new());
        }
        // Deliberately synthetic retained Tau: this test exercises public
        // admission and exact lookup selection, never a numerical producer.
        let mut manifest =
            super::tests::conditioning_test_manifest("ccm_tau_matrix", "lookup-only");
        manifest.key = key.clone();
        manifest.content_digest = ContentDigest::sha256(&self.payload);
        manifest.size_bytes = self.payload.len() as u64;
        manifest.objects = vec![xc_cache::CacheObjectRef {
            content_digest: manifest.content_digest.clone(),
            size_bytes: manifest.size_bytes,
        }];
        *self.tau.lock().unwrap() = Some(manifest.clone());
        Ok(vec![manifest])
    }
    fn read_payload_to(
        &self,
        manifest: &ArtifactManifest,
        writer: &mut dyn std::io::Write,
    ) -> std::result::Result<(), CacheError> {
        assert_eq!(manifest.key.kind, "ccm_tau_matrix");
        writer
            .write_all(&self.payload)
            .map_err(|error| CacheError::InvalidManifest(error.to_string()))
    }
}

#[test]
fn dense_lu_primary_public_require_reuse_binds_before_lookup() {
    for modes in [2usize, 255] {
        let params = CcmParams::from_lambda_sq_integer(13, modes);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.precision_bits = 128;
        cfg.n_eigenvalues = 0;
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        cfg.set_parity_policy(CcmParityPolicy::EvenSector);
        let dimension = params.matrix_size();
        let payload = serde_json::to_vec(&PortableTauMatrix {
            schema_version: 2,
            lambda_squared: "13".into(),
            n_modes: modes,
            precision_bits: 128,
            entries: (0..dimension * dimension)
                .map(|i| {
                    if i / dimension == i % dimension {
                        "1"
                    } else {
                        "0"
                    }
                    .to_owned()
                })
                .collect(),
        })
        .unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tau = Arc::new(Mutex::new(None));
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(PrimaryLookup {
                payload,
                seen: seen.clone(),
                tau: tau.clone(),
            }),
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
            acceptance: Some(&policy),
            mode: xc_cache::ArtifactExecutionCacheMode::RequireReuse,
            write_on_miss: false,
            ordered_overlays: vec!["primary-lookup-test".into()],
            reference_resolver: None,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        assert!(run_via_cache(&params, &cfg, &cache).is_err());
        let tau = tau
            .lock()
            .unwrap()
            .clone()
            .expect("public run must admit the retained Tau");
        let (semantic, logical) = weil_eigenpair_cache_identity(&params, &cfg, &tau).unwrap();
        let key = ArtifactKey {
            kind: semantic.artifact_kind.clone(),
            logical_key: logical,
            parameters_digest: semantic.digest().unwrap(),
        };
        let seen = seen.lock().unwrap();
        assert!(
            seen.contains(&key),
            "public run did not reach the exact eigenpair lookup: {seen:?}"
        );
        assert!(!seen.iter().any(|key| key.kind == "ccm_factorization"));
        let mut legacy = semantic;
        legacy
            .resolved_mathematical_parameters
            .as_object_mut()
            .unwrap()
            .remove("dense_lu_arithmetic");
        assert!(!seen.iter().any(|key| key.kind == "ccm_weil_eigenpair"
            && key.parameters_digest == legacy.digest().unwrap()));
    }
}

#[test]
fn dense_lu_primary_identity_is_independent_of_worker_count() {
    let source = super::tests::conditioning_test_manifest("ccm_tau_matrix", "lu-workers");
    let params = CcmParams::from_lambda_sq_integer(13, 24);
    let mut cfg = HighPrecConfig::for_decimal_digits(30);
    cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
    let identity = || {
        let eigenpair = weil_eigenpair_cache_identity(&params, &cfg, &source).unwrap();
        let factors = factorization_cache_identity(&params, &cfg, &source, "even").unwrap();
        serde_json::to_vec(&(eigenpair, factors)).unwrap()
    };
    let expected = identity();
    for workers in [1, 2] {
        let actual = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(identity);
        assert_eq!(actual, expected);
    }
}

#[test]
fn dense_lu_execution_failure_is_not_an_auto_krylov_refusal() {
    let execution = anyhow::anyhow!("synthetic accelerator admission failure")
        .context(xc_numerics::linalg::DenseLuExecutionFailure);
    let mapped = krylov_polishing_error(execution);
    assert!(matches!(&mapped, CacheError::InvalidManifest(message)
        if message.contains("synthetic accelerator admission failure")));
    assert!(!is_retryable_auto_krylov_failure(&anyhow::Error::new(
        mapped
    )));

    let numerical = krylov_polishing_error(anyhow::anyhow!("unresolved numerical target"));
    assert!(is_retryable_auto_krylov_failure(&anyhow::Error::new(
        numerical
    )));
}
