#![cfg(feature = "hp")]
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::{
    ArtifactKey, ArtifactManifest, CacheObjectRef, CacheQuality, CacheVisibility, ContentDigest,
    ToolkitVersion,
};
use xc_spectral::ccm::state_geometry::*;
fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(kind, "synthetic-geometry-fixture", kind.as_bytes()).unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest,
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.14.3").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    (manifest, bytes)
}

fn fixture(coefficients: &[&str]) -> (ArtifactManifest, Vec<u8>) {
    source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":(coefficients.len()-1)/2,"precision_bits":128,"eigenvalue":"0.125","eigenvector":coefficients}),
    )
}
fn retained(m: &ArtifactManifest, b: &[u8]) -> RetainedState {
    RetainedState::from_payload(m, b, std::slice::from_ref(&m.content_digest)).unwrap()
}
fn number(s: &str) -> f64 {
    rug::Float::with_val(128, rug::Float::parse(s).unwrap()).to_f64()
}
#[test]
fn geometry_constant_matches_analytic_norm_center_mass_moments_and_shells() {
    let (m, b) = fixture(&["0", "-5", "0"]);
    let state = retained(&m, &b);
    let mut options = GeometryOptions::for_source(&state);
    options.base_intervals = 512;
    let r = analyze_state_geometry(&state, &options).unwrap();
    let l = 9f64.ln();
    assert!(number(&r.refined.spatial_moments[0]).abs() < 1e-14);
    assert_eq!(number(&r.coefficient_norm), 5.0);
    assert_eq!(number(&r.raw_center), -5.0);
    assert_eq!(r.orientation, -1);
    assert!((number(&r.unit_l2_center) - 1.0 / l.sqrt()).abs() < 1e-14);
    assert!((number(r.unit_l2_signed_mass.as_ref().unwrap()) - l.sqrt()).abs() < 1e-14);
    assert!((number(&r.refined.l2_mass) - 1.0).abs() < 1e-14);
    assert!((number(&r.refined.spatial_moments[1]) - l * l / 12.0).abs() < 1e-6);
    for (v, expected) in r.refined.outer_shell_masses.iter().zip([0.75, 0.5, 0.25]) {
        assert!((number(v) - expected).abs() < 1e-14);
    }
    assert_eq!(
        number(r.refined.sampled_negative_part_l1.as_ref().unwrap()),
        0.0
    );
    assert_eq!(r.source, m.content_digest);
}
#[test]
fn geometry_cosine_and_odd_state_do_not_confuse_sign_with_energy() {
    let (m, b) = fixture(&["1", "0", "1"]);
    let s = retained(&m, &b);
    let mut o = GeometryOptions::for_source(&s);
    o.base_intervals = 1024;
    let r = analyze_state_geometry(&s, &o).unwrap();
    let l = 9f64.ln();
    assert!(
        (number(r.refined.sampled_real_minimum.as_ref().unwrap()) + (2.0 / l).sqrt()).abs() < 1e-14
    );
    assert!(
        (number(r.refined.sampled_negative_part_l1.as_ref().unwrap())
            - (2.0 * l).sqrt() / std::f64::consts::PI)
            .abs()
            < 1e-6
    );
    let (m, b) = fixture(&["-1", "0", "1"]);
    let s = retained(&m, &b);
    let r = analyze_state_geometry(&s, &o).unwrap();
    assert_eq!(r.physical_sign_status, "unavailable_nonreal_fourier_state");
    assert!(r.refined.sampled_real_minimum.is_none());
    assert!(r.unit_l2_signed_mass.is_none());
    assert!((number(&r.refined.l2_mass) - 1.0).abs() < 1e-14);
    assert_eq!(number(&r.raw_center), 0.0);
}
#[test]
fn geometry_authentication_quality_budget_and_worker_count() {
    let (mut m, b) = fixture(&["1", "3", "1"]);
    assert!(RetainedState::from_payload(&m, &b, &[]).is_err());
    let mut corrupt = b.clone();
    corrupt.push(b' ');
    assert!(
        RetainedState::from_payload(&m, &corrupt, std::slice::from_ref(&m.content_digest)).is_err()
    );
    for q in [
        CacheQuality::Validated,
        CacheQuality::Certified,
        CacheQuality::CrossChecked,
    ] {
        m.quality = q;
        let s = retained(&m, &b);
        let o = GeometryOptions::for_source(&s);
        let compute = |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| serde_json::to_vec(&analyze_state_geometry(&s, &o).unwrap()).unwrap())
        };
        assert_eq!(compute(1), compute(4));
        let mut bad = o.clone();
        bad.working_precision_bits = 64;
        assert!(analyze_state_geometry(&s, &bad).is_err());
        bad = o;
        bad.base_intervals = 8;
        assert!(analyze_state_geometry(&s, &bad).is_err());
    }
}
#[test]
fn geometry_managed_reuse_is_source_bound_and_publication_fails_closed() {
    use xc_cache::{
        ArtifactCacheContext, ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver,
        FilesystemCacheStore,
    };
    let root = std::env::temp_dir().join(format!(
        "ccm-geometry-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "geometry-test",
            root.clone(),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let context = |mode, write_on_miss| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["geometry-test".into()],
        mode,
        write_on_miss,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let (m, b) = fixture(&["1", "3", "1"]);
    let s = retained(&m, &b);
    let o = GeometryOptions::for_source(&s);
    let cold = analyze_state_geometry_via_cache(
        &s,
        &o,
        &context(ArtifactExecutionCacheMode::PreferReuse, true),
    )
    .unwrap();
    let warm = analyze_state_geometry_via_cache(
        &s,
        &o,
        &context(ArtifactExecutionCacheMode::RequireReuse, false),
    )
    .unwrap();
    assert!(cold.produced_manifest.is_some());
    assert!(warm.reused_manifest.is_some());
    assert_eq!(
        serde_json::to_value(&cold.value).unwrap(),
        serde_json::to_value(&warm.value).unwrap()
    );
    assert_eq!(
        cold.produced_manifest.unwrap().dependencies[0].content_digest,
        m.content_digest
    );
    let mut variant = o.clone();
    variant.base_intervals *= 2;
    assert!(analyze_state_geometry_via_cache(
        &s,
        &variant,
        &context(ArtifactExecutionCacheMode::RequireReuse, false)
    )
    .is_err());
    let (m, b) = fixture(&["1", "4", "1"]);
    let other = retained(&m, &b);
    assert!(analyze_state_geometry_via_cache(
        &other,
        &o,
        &context(ArtifactExecutionCacheMode::RequireReuse, false)
    )
    .is_err());
    let mut public = context(ArtifactExecutionCacheMode::Disabled, false);
    public.write_visibility = CacheVisibility::Public;
    assert!(analyze_state_geometry_via_cache(&s, &o, &public).is_err());
    let mut cert = context(ArtifactExecutionCacheMode::Disabled, false);
    cert.requested_assurance = xc_core::AssuranceLevel::Certified;
    assert!(analyze_state_geometry_via_cache(&s, &o, &cert).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn geometry_public_routing_requires_authenticated_parent_flag() {
    use xc_cache::{
        artifact_semantics_admitted_to_destination, PublicationDestination, SemanticKeyEnvelope,
    };
    let mut semantic = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: ARTIFACT_KIND.into(),
        mathematical_semantics_version: SEMANTICS.into(),
        resolved_mathematical_parameters: json!({}),
        normalization: None,
        target: None,
        subspace: None,
        source_data_identities: BTreeMap::new(),
        algorithm_semantics: None,
    };
    assert!(!artifact_semantics_admitted_to_destination(
        &semantic,
        PublicationDestination::Public
    ));
    assert!(artifact_semantics_admitted_to_destination(
        &semantic,
        PublicationDestination::Private
    ));
    semantic.resolved_mathematical_parameters = json!({"source_parents_are_public":false});
    assert!(!artifact_semantics_admitted_to_destination(
        &semantic,
        PublicationDestination::Public
    ));
    semantic.resolved_mathematical_parameters = json!({"source_parents_are_public":true});
    assert!(artifact_semantics_admitted_to_destination(
        &semantic,
        PublicationDestination::Public
    ));
}
