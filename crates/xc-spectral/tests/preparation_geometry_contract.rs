#![cfg(feature = "hp")]
use rug::Float;
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, retained_evidence::*, state_geometry::RetainedState};
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
fn state(c: &str, co: &[String], p: u32) -> (RetainedState, ArtifactManifest) {
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":c,"n_modes":co.len()/2,"precision_bits":p,"force_even":false,"eigenvalue":"3","eigenvector":co}),
    );
    (
        RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap(),
        m,
    )
}
fn roots(
    s: &RetainedState,
    m: &ArtifactManifest,
    c: &str,
    n: usize,
    source_p: u32,
    root_p: u32,
    points: &[Option<String>],
) -> RetainedRoots {
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":c,"n_modes":n,"precision_bits":source_p,"force_even":false,"eigenpair_content_digest":m.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: m.key.clone(),
        content_digest: m.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let outcomes = points
        .iter()
        .map(|v| {
            if let Some(v) = v {
                json!({"status":"converged","details":{"value":v}})
            } else {
                json!({"status":"failed","details":{"error":"synthetic missing point"}})
            }
        })
        .collect::<Vec<_>>();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":c,"n_modes":n,"precision_bits":root_p,"force_even":false,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":outcomes}),
    );
    rm.dependencies.push(DependencyRef {
        key: sec.key.clone(),
        content_digest: sec.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &sb,
        s,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap()
}

fn number(s: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/preparation_geometry_oracle.json")).unwrap()
}
fn check(actual: &str, expected: &str, p: u32, natural: &Float) {
    let a = number(actual, 2048);
    let e = number(expected, 2048);
    let scale = e.clone().abs().max(natural);
    let error = (a - e).abs();
    let limit = Float::with_val(2048, scale) >> (p - 16);
    assert!(
        error <= limit,
        "p={p}: actual={actual}, expected={expected}, error={error}, limit={limit}"
    );
}
fn preparation(
    c: &str,
    p: u32,
    co: &[String],
) -> xc_spectral::ccm::research_completion::ReferencePreparation {
    serde_json::from_value(json!({"schema_version":1,"lambda_squared":c,"precision_bits":p,
      "definition_digest":ContentDigest::sha256(b"finite preparation fixture"),"approximation_scope":"finite test fixture",
      "finite_reference":{"schema_version":1,"reference":{"schema_version":1,"definition":"finite Fourier fixture","lambda_squared":c,"precision_bits":p,"coefficients":co,"approximation_scope":"finite"},"basis":[],"projection":{"working_precision_bits":p,"normalization":"center_one"}}
    })).unwrap()
}
#[test]
fn geometry_matches_independent_discrete_fourier_sums_under_signed_scaling() {
    use xc_spectral::ccm::state_geometry::*;
    let corpus = oracle();
    let one = Float::with_val(2048, 1);
    for row in corpus["geometry"].as_array().unwrap() {
        let c = row["cutoff"].as_str().unwrap();
        for p in [128, 192, 256] {
            for exponent in [-1000, 0, 1000] {
                let factor = Float::with_val(p, if exponent < 0 { -1 } else { 1 }) << exponent;
                let co: Vec<String> = row["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        xc_numerics::prefix::lossless_decimal(
                            &(number(v.as_str().unwrap(), p) * &factor),
                        )
                    })
                    .collect();
                let (s, _) = state(c, &co, p);
                let r = analyze_state_geometry(
                    &s,
                    &GeometryOptions {
                        working_precision_bits: p,
                        base_intervals: row["base_intervals"].as_u64().unwrap() as usize,
                    },
                )
                .unwrap();
                assert_eq!(
                    r.orientation,
                    row["orientation"].as_i64().unwrap() as i32 * if exponent < 0 { -1 } else { 1 }
                );
                for (a, key) in [
                    (&r.unit_l2_center, "unit_l2_center"),
                    (
                        &r.coefficient_evenness_defect,
                        "coefficient_evenness_defect",
                    ),
                ] {
                    check(a, row[key].as_str().unwrap(), p, &one);
                }
                if let Some(mass) = r.unit_l2_signed_mass.as_ref() {
                    check(mass, row["unit_l2_signed_mass"].as_str().unwrap(), p, &one);
                }
                for (actual, wanted) in [&r.coarse, &r.refined]
                    .iter()
                    .zip(row["grids"].as_array().unwrap())
                {
                    check(
                        &actual.l2_mass,
                        wanted["l2_mass"].as_str().unwrap(),
                        p,
                        &one,
                    );
                    for (a, e) in actual
                        .spatial_moments
                        .iter()
                        .zip(wanted["spatial_moments"].as_array().unwrap())
                    {
                        check(a, e.as_str().unwrap(), p, &one);
                    }
                    assert_eq!(number(&actual.spatial_moments[0], p), 0);
                    for (a, e) in actual
                        .outer_shell_masses
                        .iter()
                        .zip(wanted["outer_shell_masses"].as_array().unwrap())
                    {
                        check(a, e.as_str().unwrap(), p, &one);
                    }
                    for (a, key) in [
                        (&actual.sampled_real_minimum, "sampled_real_minimum"),
                        (&actual.sampled_negative_part_l1, "sampled_negative_part_l1"),
                    ] {
                        if let Some(a) = a {
                            check(a, wanted[key].as_str().unwrap(), p, &one);
                        } else {
                            assert!(wanted[key].is_null());
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn geometry_uses_exact_cutoff_before_logarithm() {
    use xc_spectral::ccm::state_geometry::*;
    let all = oracle();
    let case = &all["near_one"];
    let c = case["cutoff"].as_str().unwrap();
    let (s, _) = state(c, &["1".into()], 512);
    let report = analyze_state_geometry(
        &s,
        &GeometryOptions {
            working_precision_bits: 512,
            base_intervals: 8,
        },
    )
    .unwrap();
    for (actual, key) in [
        (&report.unit_l2_center, "center"),
        (report.unit_l2_signed_mass.as_ref().unwrap(), "mass"),
        (&report.coarse.spatial_moments[1], "second_moment"),
        (&report.coarse.spatial_moments[2], "fourth_moment"),
    ] {
        check(actual, case[key].as_str().unwrap(), 512, &Float::new(2048));
    }
    assert!(report.semantics.ends_with("-v4"));
}
#[test]
fn finite_reference_samples_and_jets_match_independent_defining_integrals() {
    let all = oracle();
    let one = Float::with_val(2048, 1);
    for case in all["references"].as_array().unwrap() {
        let c = case["cutoff"].as_str().unwrap();
        let co: Vec<String> = case["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().into())
            .collect();
        for source_p in [128, 192, 256] {
            let (s, m) = state(c, &["1".into()], source_p);
            let points: Vec<_> = case["jets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Some(v["t"].as_str().unwrap().into()))
                .collect();
            let roots = roots(&s, &m, c, 0, source_p, source_p, &points);
            let prep = preparation(c, source_p, &co);
            let input = prep.prepare(&s, Some(&roots)).unwrap();
            let p = input.precision_bits;
            let target = input.target.unwrap();
            assert_eq!(target.intervals, 256);
            for (node, wanted) in case["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["samples"].as_array().unwrap())
            {
                check(
                    &target.values[node.as_u64().unwrap() as usize],
                    wanted.as_str().unwrap(),
                    p,
                    &one,
                );
            }
            for (actual, wanted) in input
                .reference_jets
                .iter()
                .zip(case["jets"].as_array().unwrap())
            {
                check(
                    &actual.reference_full.value,
                    wanted["value"].as_str().unwrap(),
                    p,
                    &one,
                );
                check(
                    &actual.reference_full.derivative,
                    wanted["derivative"].as_str().unwrap(),
                    p,
                    &one,
                );
                assert_eq!(actual.reference_window, actual.reference_full);
                assert_eq!(actual.exterior_tail.value, "0");
                assert_eq!(actual.exterior_tail.derivative, "0");
            }
        }
    }
}
#[test]
fn reference_and_root_aliases_keep_their_original_points() {
    let alias = "1.0000000000000000000000000000000000000001";
    let (s, m) = state("9", &["0".into(), "1".into(), "0".into()], 128);
    let mut outputs = vec![];
    for a in ["1", alias] {
        let r = roots(&s, &m, "9", 1, 128, 128, &[Some(a.into())]);
        let prep = preparation("9", 128, &[a.into(), "3".into(), "1".into()]);
        outputs.push(prep.prepare(&s, Some(&r)).unwrap());
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(
        outputs[0].convention_id,
        "external_reference_preparation_v2"
    );
}
#[test]
fn independent_reference_and_basis_precision_raise_working_precision() {
    let (s, _) = state("9", &["1".into()], 128);
    let mut prep = preparation("9", 128, &["1".into()]);
    let f = prep.finite_reference.as_mut().unwrap();
    f.reference.precision_bits = 256;
    let higher = Float::with_val(256, 1) + (Float::with_val(256, 1) >> 200);
    f.reference.coefficients = vec![xc_numerics::prefix::lossless_decimal(&higher)];
    let mut basis = f.reference.clone();
    basis.precision_bits = 384;
    let basis_value = Float::with_val(384, 1) + (Float::with_val(384, 1) >> 300);
    basis.coefficients = vec![xc_numerics::prefix::lossless_decimal(&basis_value)];
    f.basis.push(basis);
    let input = prep.prepare(&s, None).unwrap();
    assert_eq!(input.precision_bits, 448);
    let target = input.target.unwrap();
    assert_eq!(number(&target.raw_normalizer, 448), higher);
    for value in target.basis_values[0].iter() {
        assert_eq!(number(value, 448), basis_value);
    }
}
#[test]
fn preparation_rejects_complex_basis_and_foreign_roots() {
    let (s, _) = state("9", &["1".into()], 128);
    let mut prep = preparation("9", 128, &["1".into()]);
    let f = prep.finite_reference.as_mut().unwrap();
    let mut basis = f.reference.clone();
    basis.coefficients = vec!["1".into(), "0".into(), "0".into()];
    f.basis.push(basis);
    assert!(prep
        .prepare(&s, None)
        .unwrap_err()
        .to_string()
        .contains("even basis"));
    prep.finite_reference.as_mut().unwrap().basis.clear();
    let (other, m) = state("9", &["2".into()], 128);
    let r = roots(&other, &m, "9", 0, 128, 128, &[Some("1".into())]);
    assert!(prep
        .prepare(&s, Some(&r))
        .unwrap_err()
        .to_string()
        .contains("another state"));
}
#[test]
fn every_nested_inherited_point_preserves_aliases_without_rewriting_metadata() {
    use xc_spectral::ccm::research_completion::ReferencePreparation;
    let (s, _) = state("9", &["0".into(), "1".into(), "0".into()], 128);
    let digest = ContentDigest::sha256(b"nested preparation fixture");
    let make = |a: &str| -> ReferencePreparation {
        serde_json::from_value(json!({"schema_version":1,"lambda_squared":"9","precision_bits":128,"definition_digest":digest,"approximation_scope":"1.0000000000000000000000000000000000000001",
      "sampled_reference":{"definition_digest":digest,"evaluation_policy":"supplied","approximation_scope":"finite","intervals":8,"values":vec![a;9],"basis_values":[vec![a;9],vec![a;9]],"fixed_second_component":a,"raw_normalizer":a,"trial_coefficients":[a,a,a]},
      "weighted_atoms":[{"ordinal":1,"coordinate":a,"weight":a,"family":"zero","partition":"finite"}],"atom_coordinate":"z","atom_coverage":"finite",
      "tail_form":{"definition_digest":digest,"dimension":1,"finite_zero_form":[a],"tail_correction":[a],"lattice_gram":[a],"tail_operator_error":a,"coverage":"finite","hypotheses":["explicit finite model"]},
      "completion":{
        "atom_analysis":{"maximum_atoms":10,"maximum_input_bytes":100000,"evaluations":[{"ordinal":1,"coordinate":a,"label":"supplied"}],"cutoffs":[a],"tail_recipe":{"basis_polynomials":[[a]],"tail_correction":[a],"hypotheses":["finite"]}},
        "independent_actions":[{"label":"direct","source_digest":digest,"action":[a,a,a],"convention":"unit state"}],
        "response_checks":[{"ordinal":1,"t":a,"source_digest":digest,"branch":"finite","coordinate":"t","derivative_parameter":"s","activation_convention":"fixed","fixed_velocity":a,"support_velocity":a,"total_velocity":a}],
        "band":{"degree":1,"coordinate":"z","definition_digest":digest,"atoms":[{"coordinate":a,"signed_weight":a,"family":"zero"}],"coverage":"finite","hypotheses":["finite"],"borrowed_inputs":[],"input_energy":a,"scoring_roots":[a]},
        "contour":{"left":"-1","right":a,"bottom":"-1","top":a,"maximum_depth":8,"maximum_segments":32}
      }
    })).unwrap()
    };
    let baseline = make("1").prepare(&s, None).unwrap();
    let mut aliased = make("1.0000000000000000000000000000000000000001")
        .prepare(&s, None)
        .unwrap();
    // Point aliases remain equal, but an exact declared error premise must
    // remain an upper bound rather than collapse to its nearest binary point.
    let declared = "1.0000000000000000000000000000000000000001";
    use rug::{ops::Pow, Integer, Rational};
    let exact = Rational::from((
        Integer::from_str_radix(&declared.replace('.', ""), 10).unwrap(),
        Integer::from(10).pow((declared.len() - 2) as u32),
    ));
    let bound = aliased
        .run_once
        .as_mut()
        .unwrap()
        .tail_form
        .as_mut()
        .unwrap()
        .tail_operator_error
        .as_mut()
        .unwrap();
    let upper = Float::with_val(
        aliased.precision_bits,
        Float::parse(bound.as_str()).unwrap(),
    );
    assert!(upper >= exact && upper > 1);
    *bound = baseline
        .run_once
        .as_ref()
        .unwrap()
        .tail_form
        .as_ref()
        .unwrap()
        .tail_operator_error
        .as_ref()
        .unwrap()
        .clone();
    assert_eq!(baseline, aliased);
    assert_eq!(
        baseline.approximation_scope,
        "1.0000000000000000000000000000000000000001"
    );
}
#[test]
fn top_level_tail_recipe_preserves_original_points_before_assembly() {
    let (s, _) = state("9", &["1".into()], 128);
    let mut prep = preparation("9", 128, &["1".into()]);
    prep.finite_reference = None;
    prep.atom_coverage = Some("finite".into());
    prep.atom_coordinate = Some("z".into());
    prep.weighted_atoms = vec![WeightedAtom {
        ordinal: 1,
        coordinate: "1".into(),
        weight: "1".into(),
        family: "zero".into(),
        partition: "finite".into(),
    }];
    let mut results = vec![];
    for a in ["1", "1.0000000000000000000000000000000000000001"] {
        prep.tail_recipe = Some(xc_spectral::ccm::research_completion::TailFormRecipe {
            basis_polynomials: vec![vec![a.into()]],
            tail_correction: Some(vec![a.into()]),
            hypotheses: vec!["finite".into()],
        });
        results.push(prep.prepare(&s, None).unwrap());
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn geometry_reports_cutoff_guard_exhaustion_explicitly() {
    use xc_spectral::ccm::state_geometry::*;
    let cutoff = Float::with_val(8192, 1) + (Float::with_val(8192, 1) >> 6000);
    let cutoff = xc_numerics::prefix::lossless_decimal(&cutoff);
    let (s, _) = state(&cutoff, &["1".into()], 8192);
    let error = analyze_state_geometry(&s, &GeometryOptions::for_source(&s)).unwrap_err();
    assert!(error
        .to_string()
        .contains("unresolved within 4096 guard bits"));
}
#[test]
fn independently_tagged_comparison_points_are_not_reinterpreted() {
    use xc_spectral::ccm::{convergence_capture::ComparisonState, research_completion::*};
    let (s, _) = state("9", &["1".into()], 128);
    let mut prep = preparation("9", 128, &["1".into()]);
    let value = xc_numerics::prefix::lossless_decimal(
        &(Float::with_val(512, 1) + (Float::with_val(512, 1) >> 400)),
    );
    let digest = ContentDigest::sha256(b"independent high precision comparison");
    let comparison = ComparisonSnapshot {
        state: ComparisonState {
            source_digest: digest.clone(),
            matrix_digest: digest,
            lambda_squared: "9".into(),
            n_modes: 0,
            precision_bits: 512,
            coefficients: vec![value.clone()],
            matrix: vec![],
            eigenvalue: value.clone(),
            assembly_policy: "finite".into(),
        },
        selection_policy: "finite".into(),
        assembly_policy: "finite".into(),
        quadrature_policy: "supplied".into(),
        root_branch: "finite".into(),
        root_coordinate: "mellin_t".into(),
        roots: vec![EvaluationPoint {
            ordinal: 1,
            value: Some(value),
            source_status: "supplied".into(),
        }],
    };
    prep.completion = Some(CompletionInputs {
        comparisons: vec![comparison.clone()],
        ..Default::default()
    });
    let input = prep.prepare(&s, None).unwrap();
    assert_eq!(input.precision_bits, 192);
    assert_eq!(
        input.run_once.unwrap().completion.unwrap().comparisons,
        vec![comparison]
    );
}
#[test]
fn finite_reference_tiny_center_does_not_square_an_overflowing_norm() {
    let (s, m) = state("9", &["1".into()], 128);
    let roots = roots(&s, &m, "9", 0, 128, 128, &[Some("0".into())]);
    // The exact center is epsilon; the center-one constant coefficient is one.
    // The integral is therefore exactly log(9), despite huge nonconstant terms.
    let prep = preparation(
        "9",
        128,
        &[
            "1".into(),
            "1".into(),
            "1e-200000000".into(),
            "1".into(),
            "1".into(),
        ],
    );
    let input = prep.prepare(&s, Some(&roots)).unwrap();
    assert_eq!(
        number(
            &input.target.as_ref().unwrap().values[0],
            input.precision_bits
        ),
        1
    );
    let expected = Float::with_val(2048, 9).ln();
    check(
        &input.reference_jets[0].reference_full.value,
        &expected.to_string(),
        input.precision_bits,
        &expected,
    );
    assert_eq!(
        number(
            &input.reference_jets[0].reference_full.derivative,
            input.precision_bits
        ),
        0
    );
}
