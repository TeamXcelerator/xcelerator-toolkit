use super::*;
use xc_cache::{ArtifactKey, CacheObjectRef};

fn floats() -> Vec<Float> {
    [4, 1, 0, 1, 5, 1, 0, 1, 6]
        .map(|x| Float::with_val(128, x))
        .to_vec()
}
fn runtime_bytes(values: &[Float]) -> u64 {
    64 + values
        .iter()
        .map(|v| 64 + u64::from(v.prec()).div_ceil(64) * 8)
        .sum::<u64>()
}
fn source(values: &[Float]) -> (ArtifactManifest, Vec<u8>) {
    let resources = RunResources::new(1 << 30).unwrap();
    let bytes = xc_core::finite_json::to_vec(&Payload {
        schema_version: 1,
        lambda_squared: "13",
        n_modes: 2,
        precision_bits: values[0].prec(),
        dimension: 3,
        entries: Entries {
            values,
            resources: &resources,
        },
    })
    .unwrap();
    let digest = ContentDigest::sha256(&bytes);
    (
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(
                "ccm_even_sector_matrix",
                "shared-retained-fixture",
                b"fixture",
            )
            .unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags: BTreeMap::new(),
            provenance_digest: None,
        },
        bytes,
    )
}
fn accept(
    m: &ArtifactManifest,
    values: SharedValidated<Vec<Float>>,
    resources: &RunResources,
) -> Result<RetainedEvenMatrix> {
    RetainedEvenMatrix::from_shared_source(
        m,
        "13",
        2,
        128,
        values,
        std::slice::from_ref(&m.content_digest),
        resources,
    )
}

#[test]
fn shared_retained_matrix_preserves_exact_values_reports_and_owner_lifetime() {
    let values = floats();
    let (manifest, bytes) = source(&values);
    let retained = runtime_bytes(&values);
    let resources = RunResources::new(SERIALIZATION_WORKING_BYTES + retained).unwrap();
    let scope = resources.enter();
    let shared = SharedValidated::new(values, retained).unwrap();
    let legacy = RetainedEvenMatrix::from_payload(
        &manifest,
        &bytes,
        std::slice::from_ref(&manifest.content_digest),
    )
    .unwrap();
    let matrix = accept(&manifest, shared.clone(), &resources).unwrap();
    assert_eq!(matrix.entries(), legacy.entries());
    assert_eq!(matrix.manifest(), legacy.manifest());
    assert_eq!(matrix.entries().as_ptr(), shared.as_ptr());
    assert_eq!(
        resources.usage().peak_bytes,
        SERIALIZATION_WORKING_BYTES + retained
    );
    assert_eq!(resources.usage().working_bytes, 0);
    drop(scope);
    let options = PrefixAnalysisOptions {
        working_precision_bits: 128,
        pivot_margin_bits: 32,
        checkpoint_dimensions: vec![2, 3],
        export_significant_digits: vec![20, 30],
        export_relative_tolerance: "1e-15".into(),
        diagnostics: PrefixDiagnosticPolicy::default(),
    };
    assert_eq!(
        xc_core::finite_json::to_vec(&analyze_retained_prefixes(&matrix, &options, &[]).unwrap())
            .unwrap(),
        xc_core::finite_json::to_vec(&analyze_retained_prefixes(&legacy, &options, &[]).unwrap())
            .unwrap()
    );
    assert_eq!(
        xc_core::finite_json::to_vec(&check_retained_reduction(&matrix, 128, 3, "1e-15").unwrap())
            .unwrap(),
        xc_core::finite_json::to_vec(&check_retained_reduction(&legacy, 128, 3, "1e-15").unwrap())
            .unwrap()
    );
    let clone = matrix.clone();
    drop(shared);
    drop(matrix);
    assert_eq!(resources.usage().retained_bytes, retained);
    drop(clone);
    assert_eq!(resources.usage().retained_bytes, 0);
}

#[test]
fn shared_retained_matrix_reserves_scratch_before_replay_and_releases_on_failure() {
    let values = floats();
    let (manifest, _) = source(&values);
    let retained = runtime_bytes(&values);
    let resources = RunResources::new(SERIALIZATION_WORKING_BYTES + retained + 100).unwrap();
    let _scope = resources.enter();
    let shared = SharedValidated::new(values, retained).unwrap();
    let unrelated = resources.try_reserve_retained(101).unwrap();
    assert!(accept(&manifest, shared.clone(), &resources).is_err());
    assert_eq!(resources.usage().working_bytes, 0);
    drop(unrelated);
    let unrelated = resources.try_reserve_retained(100).unwrap();
    assert!(accept(&manifest, shared.clone(), &resources).is_ok());
    let mut bad = manifest.clone();
    bad.content_digest = ContentDigest::sha256(b"other");
    assert!(accept(&bad, shared.clone(), &resources).is_err());
    assert_eq!(resources.usage().working_bytes, 0);
    drop(unrelated);
    drop(shared);
    assert_eq!(resources.usage().retained_bytes, 0);
}

#[test]
fn shared_retained_matrix_refuses_uncharged_foreign_and_undercharged_owners() {
    let values = floats();
    let (manifest, _) = source(&values);
    let bytes = runtime_bytes(&values);
    let uncharged = SharedValidated::new(values.clone(), bytes).unwrap();
    let resources = RunResources::new(1 << 30).unwrap();
    let foreign = RunResources::new(1 << 30).unwrap();
    let _scope = resources.enter();
    let shared = SharedValidated::new(values.clone(), bytes).unwrap();
    assert!(!uncharged.is_charged_to(&resources));
    assert!(shared.is_charged_to(&resources));
    assert!(!shared.is_charged_to(&foreign));
    assert!(accept(&manifest, uncharged, &resources).is_err());
    assert!(accept(&manifest, shared, &foreign).is_err());
    let undercharged = SharedValidated::new(values, bytes - 1).unwrap();
    assert!(accept(&manifest, undercharged, &resources).is_err());
    assert_eq!(resources.usage().retained_bytes, 0);
}

#[test]
fn shared_retained_matrix_refuses_metadata_identity_scalar_and_symmetry_changes() {
    let values = floats();
    let (manifest, _) = source(&values);
    let resources = RunResources::new(8 << 30).unwrap();
    let _scope = resources.enter();
    let shared = SharedValidated::new(values.clone(), runtime_bytes(&values)).unwrap();
    for change in 0..8 {
        let mut m = manifest.clone();
        match change {
            0 => m.size_bytes -= 1,
            1 => m.size_bytes += 1,
            2 => m.content_digest = ContentDigest::sha256(b"other"),
            3 => m.immutable = false,
            4 => m.key.kind = "ccm_tau_matrix".into(),
            5 => m.quality = CacheQuality::Staged,
            6 => m.size_bytes = LEGACY_MAXIMUM_ENCODED_BYTES + 1,
            _ => m.size_bytes = resources.maximum_bytes() + 1,
        }
        m.objects[0].size_bytes = m.size_bytes;
        m.objects[0].content_digest = m.content_digest.clone();
        assert!(
            accept(&m, shared.clone(), &resources).is_err(),
            "mutation {change}"
        );
    }
    assert!(RetainedEvenMatrix::from_shared_source(
        &manifest,
        "13",
        2,
        128,
        shared.clone(),
        &[],
        &resources
    )
    .is_err());
    for (cutoff, modes, precision) in [
        ("14", 2, 128),
        ("13", 3, 128),
        ("13", 2, 129),
        ("13", usize::MAX, 128),
    ] {
        assert!(RetainedEvenMatrix::from_shared_source(
            &manifest,
            cutoff,
            modes,
            precision,
            shared.clone(),
            std::slice::from_ref(&manifest.content_digest),
            &resources
        )
        .is_err());
    }
    let mut asymmetric = values.clone();
    asymmetric[1] += 1;
    let (m, _) = source(&asymmetric);
    let changed = SharedValidated::new(asymmetric, runtime_bytes(&values)).unwrap();
    assert!(accept(&m, changed, &resources)
        .unwrap_err()
        .to_string()
        .contains("exactly symmetric"));
    let mut nonfinite = values;
    nonfinite[0] = Float::with_val(128, rug::float::Special::Nan);
    let changed = SharedValidated::new(nonfinite, shared.retained_bytes()).unwrap();
    assert!(accept(&manifest, changed, &resources)
        .unwrap_err()
        .to_string()
        .contains("finite scalars"));
    assert_eq!(resources.usage().working_bytes, 0);
}

#[test]
fn shared_retained_matrix_does_not_relabel_noncanonical_source_aliases() {
    let values = floats();
    let (mut manifest, bytes) = source(&values);
    let alias = String::from_utf8(bytes).unwrap();
    // Whitespace is semantically harmless to the legacy decoder, but it is not
    // the exact producer encoding and must never get a new identity silently.
    let alias = format!(" {alias}").into_bytes();
    manifest.content_digest = ContentDigest::sha256(&alias);
    manifest.size_bytes = alias.len() as u64;
    manifest.objects[0].size_bytes = manifest.size_bytes;
    manifest.objects[0].content_digest = manifest.content_digest.clone();
    let resources = RunResources::new(1 << 30).unwrap();
    let _scope = resources.enter();
    let legacy = RetainedEvenMatrix::from_payload(
        &manifest,
        &alias,
        std::slice::from_ref(&manifest.content_digest),
    )
    .unwrap();
    let values = legacy.entries().to_vec();
    let shared = SharedValidated::new(values.clone(), runtime_bytes(&values)).unwrap();
    assert!(accept(&manifest, shared, &resources).is_err());
}

#[test]
fn shared_retained_matrix_honors_cancellation_during_digest_writes() {
    let resources = RunResources::new(1 << 30).unwrap();
    let mut writer = HashWriter {
        hash: Sha256::new(),
        length: 0,
        maximum: 100,
        resources: &resources,
    };
    writer.write_all(b"first").unwrap();
    resources.cancel();
    assert!(writer.write_all(b"second").is_err());
    assert_eq!(writer.length, 5);
    let values = floats();
    let (manifest, _) = source(&values);
    let shared = SharedValidated::new(values.clone(), runtime_bytes(&values)).unwrap();
    assert!(accept(&manifest, shared, &resources)
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
}
#[test]
fn shared_retained_matrix_preserves_signed_zero_and_claim_scale_precision() {
    for p in [64, 4217, 1_000_000] {
        let mut values: Vec<_> = [4, 1, 0, 1, 5, 1, 0, 1, 6]
            .map(|x| Float::with_val(p, x))
            .to_vec();
        values[2] = Float::with_val(p, rug::float::Special::NegZero);
        values[6] = Float::with_val(p, rug::float::Special::NegZero);
        let (manifest, bytes) = source(&values);
        let retained = runtime_bytes(&values);
        let resources = RunResources::new(SERIALIZATION_WORKING_BYTES + retained).unwrap();
        let scope = resources.enter();
        let shared = SharedValidated::new(values, retained).unwrap();
        let handoff = RetainedEvenMatrix::from_shared_source(
            &manifest,
            "13",
            2,
            p,
            shared,
            std::slice::from_ref(&manifest.content_digest),
            &resources,
        )
        .unwrap();
        drop(scope);
        let legacy = RetainedEvenMatrix::from_payload(
            &manifest,
            &bytes,
            std::slice::from_ref(&manifest.content_digest),
        )
        .unwrap();
        for (a, b) in handoff.entries().iter().zip(legacy.entries()) {
            assert_eq!(a, b);
            assert_eq!(a.prec(), b.prec());
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
        assert_eq!(resources.usage().working_bytes, 0);
        drop(handoff);
        assert_eq!(resources.usage().retained_bytes, 0);
    }
}
#[test]
fn shared_retained_matrix_checks_exponent_before_formatter_allocation() {
    let values = floats();
    let (manifest, _) = source(&values);
    let resources = RunResources::new(1 << 30).unwrap();
    let _scope = resources.enter();
    for shift in [-1_000_001, -100_001, 100_001, 1_000_001] {
        let mut changed = values.clone();
        changed[0] = Float::with_val(128, 1) << shift;
        assert!(changed[0].is_finite() && !changed[0].is_zero());
        let shared = SharedValidated::new(changed, runtime_bytes(&values)).unwrap();
        assert!(accept(&manifest, shared, &resources)
            .unwrap_err()
            .to_string()
            .contains("bounded formatting domain"));
        assert_eq!(resources.usage().working_bytes, 0);
    }
    for shift in [-100_000, 100_000] {
        let mut changed = values.clone();
        changed[0] = Float::with_val(128, 1) << shift;
        let (manifest, bytes) = source(&changed);
        let shared = SharedValidated::new(changed, runtime_bytes(&values)).unwrap();
        let handoff = accept(&manifest, shared, &resources).unwrap();
        let legacy = RetainedEvenMatrix::from_payload(
            &manifest,
            &bytes,
            std::slice::from_ref(&manifest.content_digest),
        )
        .unwrap();
        assert_eq!(handoff.entries(), legacy.entries());
    }
}
