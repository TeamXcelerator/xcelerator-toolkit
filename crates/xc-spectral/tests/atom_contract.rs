#![cfg(feature = "hp")]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, state_geometry::RetainedState};
fn source(kind: &str, value: serde_json::Value) -> (ArtifactManifest, Vec<u8>) {
    let bytes = serde_json::to_vec(&value).unwrap();
    let digest = ContentDigest::sha256(&bytes);
    (
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new(kind, "synthetic-extension-fixture", kind.as_bytes()).unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 1,
            producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
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
fn context() -> ArtifactCacheContext<'static> {
    ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: None,
        ordered_overlays: vec!["disabled".into()],
        mode: ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    }
}
fn f(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn q(v: &serde_json::Value, p: u32) -> Float {
    let n = rug::Integer::from_str_radix(v["n"].as_str().unwrap(), 10).unwrap();
    let d = rug::Integer::from_str_radix(v["d"].as_str().unwrap(), 10).unwrap();
    Float::with_val(p, rug::Rational::from((n, d)))
}
fn fixture() -> (RetainedState, ArtifactManifest) {
    let (sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    (s, sm)
}
fn inputs(
    sm: &ArtifactManifest,
    atoms: Vec<(String, String)>,
    cutoffs: Vec<String>,
    z: &str,
) -> ExternalResearchInputs {
    let mut input:ExternalResearchInputs=serde_json::from_value(json!({"schema_version":1,"source_eigenpair":sm.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":128,"convention_id":"synthetic finite atom fixture","definition_digest":ContentDigest::sha256(b"independent atom fraction references"),"approximation_scope":"finite stored points only"})).unwrap();
    input.atom_coordinate = Some("synthetic nonnegative coordinate".into());
    input.atom_coverage = Some("complete finite synthetic table".into());
    input.tail_checkpoints = cutoffs;
    input.atoms = atoms
        .into_iter()
        .enumerate()
        .map(|(j, (x, w))| WeightedAtom {
            ordinal: j + 1,
            coordinate: x,
            weight: w,
            family: "lattice".into(),
            partition: "fixture".into(),
        })
        .collect();
    input.run_once=Some(serde_json::from_value(json!({"completion":{"atom_analysis":{"maximum_atoms":100,"maximum_input_bytes":1000000,"evaluations":[{"ordinal":1,"coordinate":z,"label":"independent fixture","exclude":[]}],"cutoffs":[]}}})).unwrap());
    input
}
fn run(s: &RetainedState, i: &ExternalResearchInputs, o: &ExtensionOptions) -> ExtendedAnalysis {
    let value = capture_extended("weighted_tail", s, None, None, Some(i), o, &[], &context())
        .unwrap()
        .value;
    assert_eq!(
        value.request["atom_arithmetic"],
        "stored_points_exact_mass_directed_moments_v2"
    );
    value.data
}
#[test]
fn public_atom_reports_match_exact_fraction_sums_at_three_precisions_and_scales() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/atom_oracle.json")).unwrap();
    let (s, sm) = self::fixture();
    let mut prefixes = 0;
    let mut kernels = 0;
    for row in fixture["rows"].as_array().unwrap() {
        for p in [128, 192, 256] {
            for e in [-400, 0, 400] {
                let atoms = row["atoms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|a| {
                        let mut w = q(&a["w"], 128);
                        if e >= 0 {
                            w <<= e as u32;
                        } else {
                            w >>= (-e) as u32;
                        }
                        (
                            q(&a["x"], 128).to_string_radix(10, None),
                            w.to_string_radix(10, None),
                        )
                    })
                    .collect();
                let expected = |v: &serde_json::Value| {
                    let mut x = q(v, p);
                    if e >= 0 {
                        x <<= e as u32;
                    } else {
                        x >>= (-e) as u32;
                    }
                    x
                };
                let refs = row["tails"].as_array().unwrap();
                let cutoffs = refs
                    .iter()
                    .map(|r| q(&r["cutoff"], 128).to_string_radix(10, None))
                    .collect();
                let i = inputs(
                    &sm,
                    atoms,
                    cutoffs,
                    &q(&row["evaluation"], 128).to_string_radix(10, None),
                );
                let mut o = ExtensionOptions::for_source(&s);
                o.working_precision_bits = p;
                let r = run(&s, &i, &o);
                assert_eq!(r.outcome, "point_measurement");
                let tails = r
                    .rows
                    .iter()
                    .filter(|r| r.label == "lattice/fixture")
                    .collect::<Vec<_>>();
                assert_eq!(tails.len(), refs.len());
                for (a, b) in tails.iter().zip(refs) {
                    for (key, expected_key) in [
                        ("included_mass", "mass"),
                        ("included_absolute_mass", "absolute_mass"),
                        ("remaining_supplied_mass", "remaining"),
                    ] {
                        assert_eq!(f(&a.values[key], p), expected(&b[expected_key]));
                    }
                    assert_eq!(
                        f(&a.values["included_count"], p),
                        b["count"].as_u64().unwrap()
                    );
                    for j in 0..3 {
                        assert_eq!(
                            f(&a.values[&format!("weighted_inverse_moment_{}", j + 1)], p),
                            expected(&b["moments"][j])
                        );
                    }
                    prefixes += 1;
                }
                let k = r
                    .rows
                    .iter()
                    .find(|r| r.label == "signed_atom_kernel")
                    .unwrap();
                for j in 0..3 {
                    assert_eq!(
                        f(&k.values[&format!("signed_kernel_{}", j + 1)], p),
                        expected(&row["kernels"][j])
                    );
                    assert_eq!(
                        f(&k.values[&format!("absolute_kernel_{}", j + 1)], p),
                        expected(&row["absolute_kernels"][j])
                    );
                }
                assert_eq!(
                    f(&k.values["closest_included_atom_distance"], p),
                    q(&row["closest"], p)
                );
                kernels += 1;
            }
        }
    }
    assert_eq!(prefixes, 1800);
    assert_eq!(kernels, 360);
}
#[test]
fn atom_counterexamples_preserve_cancellation_suffix_and_source_point_coincidence() {
    let (s, sm) = fixture();
    let o = ExtensionOptions::for_source(&s);
    let atomvec = |a: Vec<(&str, &str)>| a.into_iter().map(|(x, w)| (x.into(), w.into())).collect();
    let i = inputs(
        &sm,
        atomvec(vec![("1", "1e100"), ("1", "1"), ("1", "-1e100")]),
        vec!["1".into()],
        "0",
    );
    let r = run(&s, &i, &o);
    assert_eq!(f(&r.rows[0].values["included_mass"], 192), 1);
    for j in 1..=3 {
        assert_eq!(
            f(
                &r.rows[0].values[&format!("weighted_inverse_moment_{j}")],
                192
            ),
            1
        );
        assert_eq!(f(&r.rows[1].values[&format!("signed_kernel_{j}")], 192), 1);
    }
    let i = inputs(
        &sm,
        atomvec(vec![("1", "1e100"), ("2", "1")]),
        vec!["1".into(), "2".into()],
        "0",
    );
    let r = run(&s, &i, &o);
    assert_eq!(f(&r.rows[0].values["remaining_supplied_mass"], 192), 1);
    let i = inputs(
        &sm,
        atomvec(vec![("1e200000000", "1")]),
        vec!["1e200000000".into()],
        "0",
    );
    let r = run(&s, &i, &o);
    assert_eq!(r.outcome, "partial_unresolved");
    assert_eq!(f(&r.rows[0].values["included_mass"], 192), 1);
    assert!(!r.rows[0].values.contains_key("weighted_inverse_moment_2"));
    assert!(!r.rows[1].values.contains_key("signed_kernel_2"));
    let i = inputs(
        &sm,
        atomvec(vec![("1.0000000000000000000000000000000000000001", "1")]),
        vec!["2".into()],
        "1",
    );
    let r = run(&s, &i, &o);
    assert_eq!(r.rows[1].outcome, "unresolved_denominator");
    assert!(!r.rows[1].values.contains_key("signed_kernel_1"));
    assert!(r.rows[1].notes.iter().any(|n| n.contains("coincident")));
    let mut limited = o.clone();
    limited.maximum_working_bytes = Some(1);
    let r = run(&s, &i, &limited);
    assert!(r.reason.as_ref().unwrap().contains("scratch estimate"));
}
#[test]
fn atom_origin_exclusions_empty_sums_and_default_checkpoints_keep_their_meaning() {
    let (s, sm) = fixture();
    let o = ExtensionOptions::for_source(&s);
    let mut i = inputs(
        &sm,
        vec![
            ("0".into(), "1".into()),
            ("1".into(), "2".into()),
            ("1".into(), "-1".into()),
            ("3".into(), "1".into()),
        ],
        vec![],
        "0",
    );
    let r = run(&s, &i, &o);
    assert_eq!(r.outcome, "partial_unresolved");
    let tails = r
        .rows
        .iter()
        .filter(|r| r.label == "lattice/fixture")
        .collect::<Vec<_>>();
    assert_eq!(tails.len(), 3);
    assert_eq!(f(&tails[2].values["cutoff"], 192), 3);
    assert_eq!(f(&tails[2].values["included_count"], 192), 4);
    assert!(!tails[2].values.contains_key("weighted_inverse_moment_1"));
    let policy = &mut i
        .run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .atom_analysis
        .as_mut()
        .unwrap()
        .evaluations[0];
    policy.exclude = (1..=4)
        .map(|ordinal| xc_spectral::ccm::atom_research::AtomKey {
            family: "lattice".into(),
            partition: "fixture".into(),
            ordinal,
        })
        .collect();
    let r = run(&s, &i, &o);
    let k = r.rows.last().unwrap();
    assert_eq!(k.outcome, "point_measurement");
    for j in 1..=3 {
        assert_eq!(f(&k.values[&format!("signed_kernel_{j}")], 192), 0);
    }
    i.run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .atom_analysis
        .as_mut()
        .unwrap()
        .evaluations[0]
        .exclude[0]
        .ordinal = 99;
    let r = run(&s, &i, &o);
    assert_eq!(r.rows.last().unwrap().outcome, "missing_input");
}

#[test]
fn atom_evaluation_coordinate_echo_preserves_the_source_point_at_report_precision() {
    let (s, sm) = fixture();
    let i = inputs(&sm, vec![("2".into(), "1".into())], vec!["2".into()], "0.1");
    let o = ExtensionOptions::for_source(&s);
    let r = run(&s, &i, &o);
    let expected = Float::with_val(o.working_precision_bits, f("0.1", i.precision_bits));
    assert_eq!(
        f(
            &r.rows.last().unwrap().values["evaluation_coordinate"],
            o.working_precision_bits
        ),
        expected
    );
}
