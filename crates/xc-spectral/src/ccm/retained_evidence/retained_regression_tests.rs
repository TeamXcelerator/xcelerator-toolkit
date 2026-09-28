use super::*;
fn state() -> RetainedState {
    let p = 192;
    let bytes=serde_json::to_vec(&json!({"schema_version":3,"lambda_squared":"10","n_modes":1,"precision_bits":p,"force_even":false,"eigenvalue":"1","eigenvector":["1","-4","2"]})).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(
            "ccm_weil_eigenpair",
            "remaining-mixed-state",
            b"remaining-mixed-state",
        )
        .unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 0,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.13.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    RetainedState::from_payload(&manifest, &bytes, &[digest]).unwrap()
}
#[test]
fn mixed_parity_transform_uses_the_root_convention() {
    let s = state();
    let p = 192;
    let work = 384;
    // Independent algebra: residues [1,-4,2] at [-w,0,w] give
    // numerator -t^2+w*t+4w^2, root w*(1+sqrt(17))/2.
    let l = Float::with_val(work, 10).ln();
    let w = Float::with_val(work, rug::float::Constant::Pi) * 2u32 / &l;
    let t = Float::with_val(p, &w * (Float::with_val(work, 17).sqrt() + 1u32) / 2u32);
    let root = transform_math::measure_root(&s, &t, p).unwrap().unwrap();
    let plus = transform_math::measure(&s, &t, p).unwrap().unwrap();
    assert!(root.value.lower().clone().abs() < Float::with_val(root.precision, 1) >> 180u32);
    assert!(root.value.upper().clone().abs() < Float::with_val(root.precision, 1) >> 180u32);
    assert!(plus.value.lower().clone().abs() > Float::with_val(plus.precision, 0.01));
    // Independent closed-form integral derivative at the retained t point.
    let t = Float::with_val(work, &t);
    let half = Float::with_val(work, &l / 2u32);
    let phase = Float::with_val(work, &t * &half);
    let sine = phase.clone().sin();
    let cosine = phase.clone().cos();
    let mut derivative = Float::with_val(work, 0);
    for (j, c) in [(-1, 1), (0, -4), (1, 2)] {
        let q = Float::with_val(
            work,
            &phase - Float::with_val(work, rug::float::Constant::Pi) * j,
        );
        derivative += (Float::with_val(work, &cosine / &q)
            - Float::with_val(work, &sine / Float::with_val(work, &q * &q)))
            * c;
    }
    derivative *= half;
    derivative *= l.sqrt();
    derivative /= Float::with_val(work, 21).sqrt();
    derivative = -derivative; // center coefficient fixes orientation negative
    let actual = Float::with_val(work, root.derivative.midpoint_point().lower());
    assert!((actual - derivative).abs() < Float::with_val(work, 1) >> 180u32);
}
#[test]
fn aliases_share_numeric_options_and_cutoffs() {
    assert!(equal_cutoff("13", "13.0").unwrap());
    assert_eq!(
        canonical_decimal("0.10").unwrap(),
        canonical_decimal("1e-1").unwrap()
    );
    assert!(!equal_cutoff("13", "13.000000000000000001").unwrap());
    assert_eq!(
        energy::residual_normalization("0", 128).unwrap(),
        "absolute_residual_per_coefficient_l2_norm"
    );
    assert_eq!(
        energy::residual_normalization("1", 128).unwrap(),
        "relative_to_abs_eigenvalue_and_coefficient_l2_norm"
    );
}
