#![cfg(feature = "hp")]
mod energy_extension_tests {
    use super::*;
    use rug::Rational;
    use xc_solver::trial_energy::ExactBounds as B;
    use xc_spectral::ccm::convergence_capture::finite_capture::energy_extensions::capture_projection;
    use xc_spectral::ccm::convergence_capture::finite_capture::{self as f, energy_extensions::*};
    fn bound(s: &str) -> DeclaredBound {
        DeclaredBound {
            upper: s.into(),
            provenance: "synthetic exact model".into(),
            scope: "declared object only".into(),
        }
    }
    fn projection() -> ProjectionRequest {
        ProjectionRequest {
            definition_digest: ContentDigest::sha256(b"generic polynomial"),
            operator_id: "test form".into(),
            period: "2".into(),
            basis_id: "orthonormal_periodic_fourier".into(),
            domain: "interval_h1".into(),
            coefficients: complex_vector("polynomial", &["0", "1", "0"], &["0", "0", "0"]),
            retained_modes: 0,
            remainder: FourierRemainder::ExactPolynomial,
            source_l2_error: bound("0"),
            source_h1_error: bound("0"),
            continuity: Some(FormContinuity {
                operator_id: "test form".into(),
                period: "2".into(),
                basis_id: "orthonormal_periodic_fourier".into(),
                domain: "interval_h1".into(),
                bound: bound("2"),
            }),
            trial_absolute_energy: Some(bound("3")),
            precision_bits: 128,
        }
    }
    fn rat(v: &serde_json::Value) -> Rational {
        Rational::from_str_radix(v.as_str().unwrap(), 10).unwrap()
    }
    fn contains(v: &serde_json::Value, s: &str) -> bool {
        let x = Rational::from_str_radix(s, 10).unwrap();
        rat(&v["lower"]) <= x && x <= rat(&v["upper"])
    }
    #[test]
    fn projection_energy_keeps_interval_traces_and_missing_tail_distinct() {
        let mut p = projection();
        let r = analyze_projection(&p).unwrap();
        assert_eq!(r["absolute_rayleigh_upper"], "3");
        assert_eq!(r["energy_difference_upper"], "0");
        assert!(rat(&r["endpoint_traces"]["real"]["lower"]) > 0);
        p.remainder = FourierRemainder::Unknown;
        let r = analyze_projection(&p).unwrap();
        assert_eq!(r["status"], "unresolved");
        assert_eq!(r["observed_tail_l2_squared"]["upper"], "0");
        assert!(r.get("energy_difference_upper").is_none());
        p.remainder = FourierRemainder::ExactPolynomial;
        p.source_l2_error = bound("1");
        assert_eq!(
            analyze_projection(&p).unwrap()["quotient_status"],
            "unresolved_norm_floor"
        );
        p.continuity.as_mut().unwrap().period = "3".into();
        assert!(analyze_projection(&p).is_err());
    }
    #[test]
    fn projection_high_frequency_and_tighter_declared_tails_change_the_bound() {
        let mut p = projection();
        p.period = "1/10000".into();
        p.continuity.as_mut().unwrap().period = p.period.clone();
        p.coefficients.real[2] = B::point("1/10000");
        let r = analyze_projection(&p).unwrap();
        assert!(rat(&r["observed_tail_l2_squared"]["upper"]) < 1);
        assert!(rat(&r["observed_tail_h1_squared"]["lower"]) > 30);
        p = projection();
        p.remainder = FourierRemainder::Declared {
            l2: bound("1/10"),
            h1: bound("1/5"),
        };
        let loose = analyze_projection(&p).unwrap();
        p.remainder = FourierRemainder::Declared {
            l2: bound("1/100"),
            h1: bound("1/50"),
        };
        let tight = analyze_projection(&p).unwrap();
        assert!(rat(&tight["energy_difference_upper"]) < rat(&loose["energy_difference_upper"]));
        assert!(rat(&tight["absolute_rayleigh_upper"]) < rat(&loose["absolute_rayleigh_upper"]));
    }
    fn component(label: &str, sign: &str, diagonal: &[&str]) -> SignedComponent {
        SignedComponent::from_data(ComponentData {
            label: label.into(),
            signed_weight: sign.into(),
            diagonal: diagonal.iter().map(|x| B::point(*x)).collect(),
            upper_triangle: vec![],
            rank_one: vec![],
        })
        .unwrap()
    }
    fn component_input(
        m: &ArtifactManifest,
        _matrix: &RetainedMatrix<'_>,
    ) -> ExternalResearchInputs {
        let mut i = inputs(m);
        i.finite_diagnostics = Some(f::Inputs {
            scope: "synthetic signed components".into(),
            component_energy: Some(ComponentRequest {
                matrix_digest: m.dependencies[0].content_digest.clone(),
                basis_id: "centered_full_V_fourier".into(),
                components: vec![
                    component("positive", "1", &["10", "3", "10"]),
                    component("subtracted", "-1", &["6", "0", "6"]),
                ],
                assembly_operator_norm_error: None,
            }),
            ..Default::default()
        });
        i
    }
    fn component_run(
        i: &ExternalResearchInputs,
        s: &RetainedState,
        m: &RetainedMatrix<'_>,
    ) -> serde_json::Value {
        f::capture(
            "trial_vector_energy",
            s,
            Some(m),
            None,
            Some(i),
            None,
            &[],
            &context(),
        )
        .unwrap()
        .value
        .data
        .result["component_energy"]
            .clone()
    }
    #[test]
    fn component_closure_checks_all_vectors_and_authenticates_signs() {
        let (s, sm, m) = fixture();
        let mut i = component_input(&sm, &m);
        let r = component_run(&i, &s, &m);
        assert_eq!(r["operator_closure"]["status"], "exact_stored_equality");
        // Components stream concurrently; the result is independent of threads.
        for threads in [1, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(|| component_run(&i, &s, &m)), r);
        }
        let x = i
            .finite_diagnostics
            .as_mut()
            .unwrap()
            .component_energy
            .as_mut()
            .unwrap();
        x.components.pop();
        let r = component_run(&i, &s, &m);
        // The retained center vector still has exactly matching energy.
        assert_eq!(r["stages"][0]["total_minus_component_sum"]["lower"], "0");
        assert_eq!(r["operator_closure"]["status"], "refuted");
        assert_eq!(r["operator_closure"]["spectral_norm_upper"], "6");
        let x = i
            .finite_diagnostics
            .as_mut()
            .unwrap()
            .component_energy
            .as_mut()
            .unwrap();
        x.components
            .push(component("wrong sign", "1", &["6", "0", "6"]));
        assert_eq!(
            component_run(&i, &s, &m)["operator_closure"]["spectral_norm_upper"],
            "12"
        );
        i.finite_diagnostics
            .as_mut()
            .unwrap()
            .component_energy
            .as_mut()
            .unwrap()
            .components[0]
            .data
            .signed_weight = "-1".into();
        assert!(f::capture(
            "trial_vector_energy",
            &s,
            Some(&m),
            None,
            Some(&i),
            None,
            &[],
            &context()
        )
        .is_err());
    }
    #[test]
    fn signed_components_follow_complex_stages_and_pairings() {
        let (s, sm, m) = fixture();
        let mut i = component_input(&sm, &m);
        i.finite_diagnostics.as_mut().unwrap().complex_trials = Some(f::ComplexTrialSeries {
            basis_id: "centered_full_V_fourier".into(),
            matrix_digest: sm.dependencies[0].content_digest.clone(),
            scope: "complex synthetic".into(),
            baseline: complex_vector("first", &["1", "0", "0"], &["0", "1", "0"]),
            corrections: vec![complex_vector(
                "change",
                &["-1", "1", "0"],
                &["0", "0", "1"],
            )],
            functional: None,
            provenance: BTreeMap::new(),
        });
        let r = component_run(&i, &s, &m);
        assert_eq!(r["stages"][0]["signed_sum"]["lower"], "7");
        assert_eq!(r["stages"][1]["signed_sum"]["lower"], "10");
        assert_eq!(r["stages"][1]["total_minus_component_sum"]["upper"], "0");
        assert_eq!(
            r["components"][0]["report"]["energy_pairings"][1]["lower"],
            "-10"
        );
        assert_eq!(
            r["components"][1]["report"]["energy_pairings"][1]["lower"],
            "6"
        );
    }
    #[test]
    fn component_rank_one_and_interval_closure_do_not_overstate_equality() {
        let (s, sm, m) = fixture();
        let mut i = component_input(&sm, &m);
        let remainder = ComponentData {
            label: "remainder".into(),
            signed_weight: "1".into(),
            diagonal: vec![],
            upper_triangle: ["3", "-1", "0", "2", "0", "4"]
                .into_iter()
                .map(B::point)
                .collect(),
            rank_one: vec![],
        };
        let rank = ComponentData {
            label: "rank one".into(),
            signed_weight: "1".into(),
            diagonal: vec![],
            upper_triangle: vec![],
            rank_one: vec![RankOne {
                weight: B::point("1"),
                vector: ["1", "1", "0"].into_iter().map(B::point).collect(),
            }],
        };
        i.finite_diagnostics
            .as_mut()
            .unwrap()
            .component_energy
            .as_mut()
            .unwrap()
            .components = vec![
            SignedComponent::from_data(rank).unwrap(),
            SignedComponent::from_data(remainder.clone()).unwrap(),
        ];
        assert_eq!(
            component_run(&i, &s, &m)["operator_closure"]["status"],
            "exact_stored_equality"
        );
        let mut uncertain = remainder;
        uncertain.upper_triangle[1] = B {
            lower: "-11/10".into(),
            upper: "-9/10".into(),
        };
        i.finite_diagnostics
            .as_mut()
            .unwrap()
            .component_energy
            .as_mut()
            .unwrap()
            .components[1] = SignedComponent::from_data(uncertain).unwrap();
        let r = component_run(&i, &s, &m);
        assert_eq!(r["operator_closure"]["status"], "enclosed_not_proved_equal");
        assert_eq!(r["operator_closure"]["spectral_norm_upper"], "1/10");
    }
    fn profile(label: &str, knots: &[(&str, &str)]) -> LogProfile {
        LogProfile {
            label: label.into(),
            knots: knots
                .iter()
                .map(|(x, y)| Knot {
                    coordinate: (*x).into(),
                    value: (*y).into(),
                })
                .collect(),
        }
    }
    fn continuous() -> ContinuousRequest {
        let k = profile("compact tent", &[("-1/4", "0"), ("0", "1"), ("1/4", "0")]);
        ContinuousRequest {
            definition_digest: ContentDigest::sha256(b"synthetic compact functions"),
            profile_a: k.clone(),
            profile_b: k.clone(),
            profile_sum: profile("sum", &[("-1/4", "0"), ("0", "2"), ("1/4", "0")]),
            scope: ProfileScope::ExactInterpolant,
            summands: vec![],
            integration_windows: vec![],
            unrepresented_sum_l2: None,
            overlap: None,
            translation_shifts: vec!["1/8".into(), "3/8".into()],
            integration_cells: 128,
            maximum_prime_power: 8,
            precision_bits: 128,
        }
    }
    #[test]
    fn continuous_energy_is_independent_and_preserves_correlation_identity() {
        let mut resolved = continuous();
        resolved.integration_cells = 2048;
        let r = analyze_continuous(&resolved).unwrap();
        // Forms, prime terms and cells are evaluated concurrently and folded
        // in order; the result is independent of the thread count.
        for threads in [1, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(|| analyze_continuous(&resolved).unwrap()), r);
        }
        assert_eq!(r["profile_a"]["squared_norm"], "1/6");
        assert_eq!(r["profile_a"]["derivative_squared_norm"], "8");
        assert_eq!(r["profile_a"]["prime"]["lower"], "0");
        assert!(contains(&r["difference_energy"], "0"));
        // Independent closed-form correlation plus high-precision quadrature oracle.
        assert!(contains(
            &r["profile_a"]["energy"],
            "12933191472510148103423655415665/1000000000000000000000000000000000"
        ));
        assert_eq!(r["decomposition_l2_squared"], "0");
        assert_eq!(r["translations"][0]["correlation"], "23/192");
        assert_eq!(
            r["translations"][0]["translated_difference_squared_norm"],
            "3/32"
        );
        assert_eq!(r["translations"][1]["shifted_support"][1], "-1/8");
        assert!(!contains(&r["sum_with_a"], "0"));
        let mut c = resolved;
        c.profile_b = profile("zero", &[("-1/4", "0"), ("1/4", "0")]);
        c.profile_sum = c.profile_a.clone();
        let r = analyze_continuous(&c).unwrap();
        assert!(!contains(&r["difference_energy"], "0"));
        assert_eq!(r["profile_b"]["rayleigh"], serde_json::Value::Null);
    }
    #[test]
    fn continuous_enclosures_refine_but_do_not_invent_source_tails() {
        let mut c = continuous();
        c.integration_cells = 32;
        let a = analyze_continuous(&c).unwrap();
        c.integration_cells = 256;
        c.scope = ProfileScope::UnresolvedSource;
        let b = analyze_continuous(&c).unwrap();
        assert!(rat(&b["profile_a"]["absolute_width"]) < rat(&a["profile_a"]["absolute_width"]));
        assert_eq!(b["source_global_status"], "unresolved");
        c.scope = ProfileScope::DeclaredEnergyErrors {
            profile_a: Box::new(bound("1/10")),
            profile_b: Box::new(bound("1/5")),
            profile_sum: Box::new(bound("1/2")),
        };
        let declared = analyze_continuous(&c).unwrap();
        assert_eq!(
            declared["source_global_status"],
            "conditional_on_declared_energy_error_ledgers"
        );
        assert_eq!(
            rat(&b["profile_a"]["energy"]["lower"])
                - rat(&declared["profile_a"]["conditional_source_energy"]["lower"]),
            Rational::from((1, 10))
        );
        c.profile_a.knots[0].value = "1".into();
        assert!(analyze_continuous(&c).is_err());
        c = continuous();
        c.profile_a.knots[0].coordinate = "-8".into();
        assert!(analyze_continuous(&c).is_err());
    }
    #[test]
    fn separate_summands_keep_endpoint_atoms_without_spurious_connections() {
        let mut c = continuous();
        c.summands = vec![
            profile("left", &[("0", "1"), ("1", "1")]),
            profile("right", &[("2", "1"), ("3", "1")]),
        ];
        c.integration_windows = vec![IntegrationWindow {
            label: "edge strip".into(),
            lower: "3/4".into(),
            upper: "5/4".into(),
        }];
        c.unrepresented_sum_l2 = Some(bound("1/10"));
        let r = analyze_continuous(&c).unwrap();
        assert_eq!(r["summands"]["diagonal_sum"], "2");
        assert_eq!(r["summands"]["sum_norm_squared"], "2");
        assert_eq!(r["summands"]["aggregate_off_diagonal"], "0");
        assert_eq!(r["summands"]["endpoint_atom_variations"][0], "2");
        assert_eq!(
            r["summands"]["windows"][0]["first_summand_squared_norm"],
            "1/4"
        );
        assert_eq!(r["summands"]["tail_status"], "conditional");
        c.summands[1] = c.summands[0].clone();
        let r = analyze_continuous(&c).unwrap();
        assert_eq!(r["summands"]["sum_norm_squared"], "4");
        assert_eq!(r["summands"]["aggregate_off_diagonal"], "2");
    }
    #[test]
    fn independent_poisson_route_retains_pole_correction_and_explicit_tail() {
        let mut o = AdditiveOverlap {
            source: profile("reflected tent samples", &[("0", "1"), ("1", "0")]),
            coordinates: vec!["1/2".into(), "2".into()],
            fourier_terms: 16,
        };
        let a = analyze_overlap(&o, 128).unwrap();
        assert_eq!(a["zero_correction_constraints_hold"], false);
        for row in a["rows"].as_array().unwrap() {
            assert_eq!(row["overlap_consistent"], true);
        }
        assert!(!contains(&a["rows"][0]["pole_correction"], "0"));
        o.fourier_terms = 64;
        let b = analyze_overlap(&o, 192).unwrap();
        assert!(
            rat(&b["rows"][0]["omitted_fourier_terms_upper"])
                < rat(&a["rows"][0]["omitted_fourier_terms_upper"])
        );
        o.source = profile(
            "zero constraints",
            &[("0", "0"), ("1", "1"), ("2", "-1"), ("3", "0")],
        );
        let c = analyze_overlap(&o, 128).unwrap();
        assert_eq!(c["zero_constraints_hold"], true);
        for row in c["rows"].as_array().unwrap() {
            assert_eq!(row["overlap_consistent"], true);
        }
    }
    #[test]
    fn extension_capture_is_private_replayable_and_joins_existing_groups() {
        let dir = xc_core::test_support::TestDir::new("energy-extension-replay");
        let resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(ZipJsonFilesystemCacheStore::new(
                "local",
                dir.to_path_buf(),
                true,
                CacheVisibility::Local,
            )),
        }]);
        let policy = CachePolicy {
            current_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let cache = ArtifactCacheContext {
            resolver: Some(&resolver),
            acceptance: Some(&policy),
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            ordered_overlays: vec!["local".into()],
            ..context()
        };
        let p = projection();
        let c = continuous();
        let first = capture_projection(&p, &cache).unwrap();
        let replay = capture_projection(&p, &cache).unwrap();
        assert!(first.produced_manifest.is_some() && replay.reused_manifest.is_some());
        assert_eq!(first.value.data, replay.value.data);
        let first = capture_continuous(&c, &cache).unwrap();
        let replay = capture_continuous(&c, &cache).unwrap();
        assert!(first.produced_manifest.is_some() && replay.reused_manifest.is_some());
        assert_eq!(first.value.data, replay.value.data);
        let mut public = context();
        public.write_visibility = CacheVisibility::Public;
        assert!(capture_projection(&p, &public).is_err());
        assert!(capture_continuous(&c, &public).is_err());
        let (s, sm, m) = fixture();
        let mut i = component_input(&sm, &m);
        i.finite_diagnostics.as_mut().unwrap().projection_energy = Some(p);
        i.finite_diagnostics.as_mut().unwrap().continuous_energy = Some(c);
        for (group, field) in [
            ("finite_tail_bound", "projection_energy"),
            ("trial_vector_energy", "continuous_energy"),
        ] {
            let a = f::capture(group, &s, Some(&m), None, Some(&i), None, &[], &cache).unwrap();
            assert!(a.value.data.result[field].is_object());
            assert!(a.value.data.rows.iter().any(|r| r.label == field));
            let b = f::capture(group, &s, Some(&m), None, Some(&i), None, &[], &cache).unwrap();
            assert!(b.reused_manifest.is_some());
        }
        let combined = f::capture(
            "trial_vector_energy",
            &s,
            None,
            None,
            Some(&i),
            None,
            &[],
            &cache,
        )
        .unwrap();
        assert!(combined
            .value
            .data
            .rows
            .iter()
            .any(|r| r.label == "primary_analysis" && r.outcome == "missing_input"));
        assert!(combined
            .value
            .data
            .rows
            .iter()
            .any(|r| r.label == "continuous_energy"));

        i.finite_diagnostics.as_mut().unwrap().energy_distance = Some(f::EnergyDistancePremises {
            matrix_digest: sm.dependencies[0].content_digest.clone(),
            basis_id: "centered_full_V_fourier".into(),
            metric_id: "identity-coefficient-metric-v1".into(),
            ground_enclosure: B {
                lower: "100000".into(),
                upper: "100001".into(),
            },
            complementary_eigenvalue_lower: "200000".into(),
            full_space_simple_ground: true,
            provenance: "deliberately refuted synthetic premise".into(),
        });
        let retained = f::capture(
            "trial_vector_energy",
            &s,
            Some(&m),
            None,
            Some(&i),
            None,
            &[],
            &cache,
        )
        .unwrap();
        assert!(retained.value.data.result["report"]["stages"].is_array());
        assert!(retained
            .value
            .data
            .rows
            .iter()
            .any(|r| r.label == "energy_distance" && r.outcome == "premise_not_verified"));

        let mut execution = f::ExecutionContext {
            certificate_error: Some("attempt at path one".into()),
            ..Default::default()
        };
        let first = f::capture_with_context(
            "finite_root_budget",
            &s,
            Some(&m),
            None,
            None,
            None,
            &[],
            &cache,
            &execution,
        )
        .unwrap();
        execution.certificate_error = Some("attempt at path two".into());
        let replay = f::capture_with_context(
            "finite_root_budget",
            &s,
            Some(&m),
            None,
            None,
            None,
            &[],
            &cache,
            &execution,
        )
        .unwrap();
        assert!(replay.reused_manifest.is_some());
        assert_eq!(first.value.data, replay.value.data);
    }
}
use rug::{float::Constant, Float};
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
fn fixture() -> (RetainedState, ArtifactManifest, RetainedMatrix<'static>) {
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["4","0","0","0","3","0","0","0","4"]}),
    );
    let (mut m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    m.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    (s, m, matrix)
}
fn inputs(m: &ArtifactManifest) -> ExternalResearchInputs {
    serde_json::from_value(json!({"schema_version":1,"source_eigenpair":m.content_digest,"lambda_squared":"9","n_modes":1,"precision_bits":128,"convention_id":"synthetic test values","definition_digest":ContentDigest::sha256(b"synthetic definition"),"approximation_scope":"finite test points"})).unwrap()
}
fn run(
    id: &str,
    s: &RetainedState,
    m: Option<&RetainedMatrix<'_>>,
    i: Option<&ExternalResearchInputs>,
) -> ExtendedAnalysis {
    capture_extended(
        id,
        s,
        m,
        None,
        i,
        &ExtensionOptions::for_source(s),
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
}
fn number(s: &str) -> f64 {
    Float::with_val(256, Float::parse(s).unwrap()).to_f64()
}
fn near(actual: &str, expected: f64, tol: f64) {
    assert!(
        (number(actual) - expected).abs() < tol,
        "{actual} versus {expected}"
    );
}
#[test]
fn compactness_origin_jet_matches_constant_and_cosine_integrals() {
    let (s, _, _) = fixture();
    let r = run("compactness", &s, None, None);
    let l = 9f64.ln();
    near(&r.values["transform_origin"], l.sqrt(), 1e-14);
    near(
        &r.values["transform_second_derivative"],
        -l.powf(2.5) / 12.,
        1e-13,
    );
    near(
        &r.values["transform_fourth_derivative"],
        l.powf(4.5) / 80.,
        1e-13,
    );
    near(&r.values["sigma"], l * l / 24., 1e-14);
    for row in &r.rows {
        let a = number(&row.values["rate"]);
        let exact = ((a * l).exp() - 1.) / (a * l);
        near(&row.values["analytic_integral"], exact, 1e-12);
    }
    let (m, b) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["-1","0","-1"]}),
    );
    let s = RetainedState::from_payload(&m, &b, std::slice::from_ref(&m.content_digest)).unwrap();
    let r = run("compactness", &s, None, None);
    assert!(!r.values.contains_key("sigma"));
    near(
        &r.values["transform_second_derivative"],
        l.powf(2.5) / (2f64.sqrt() * std::f64::consts::PI.powi(2)),
        1e-13,
    );
}
#[test]
fn weighted_nonorthogonal_projection_retains_raw_b2_convention() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    let p = 192;
    let half = Float::with_val(p, 9).ln() / 2u32;
    let n = 32usize;
    let mut values = vec![];
    let mut basis = vec![vec![], vec![]];
    for j in 0..=n {
        let x: Float = Float::with_val(p, &half) * j / n;
        values.push((Float::with_val(p, 1) - Float::with_val(p, &x) / 10u32).to_string());
        basis[0].push((Float::with_val(p, 1) + &x).to_string());
        basis[1].push((Float::with_val(p, 1) - &x).to_string());
    }
    i.target = Some(SampledReference {
        definition_digest: ContentDigest::sha256(b"linear numerical fixture"),
        evaluation_policy: "explicit uniform x points".into(),
        approximation_scope: "point samples".into(),
        intervals: n,
        values,
        basis_values: basis,
        fixed_second_component: Some("0.5".into()),
        raw_normalizer: "1".into(),
        trial_coefficients: None,
    });
    let r = run("weighted_reference_projection", &s, None, Some(&i));
    near(&r.values["a_0"], 0.05, 1e-14);
    near(&r.values["a_1"], -0.05, 1e-14);
    near(&r.values["b2"], -0.075, 1e-14);
    near(&r.values["b_effective"], -1., 1e-13);
    assert!(number(&r.values["fit_residual_norm_squared"]) < 1e-70);
    let t = i.target.as_mut().unwrap();
    t.basis_values[1] = t.basis_values[0].clone();
    assert_eq!(
        run("weighted_reference_projection", &s, None, Some(&i)).outcome,
        "rank_or_precision_unresolved"
    );
}
#[test]
fn signed_reference_channels_keep_cancellation_and_closure_separate() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    let l = Float::with_val(192, 9).ln();
    i.reference_jets.push(ReferenceJet {
        matched_root_ordinal: None,
        ordinal: 200,
        t: "0".into(),
        reference_window: Jet {
            value: (Float::with_val(192, &l) - Float::with_val(192, Float::parse("0.1").unwrap()))
                .to_string(),
            derivative: "0".into(),
        },
        reference_full: Jet {
            value: (Float::with_val(192, &l) + Float::with_val(192, Float::parse("0.1").unwrap()))
                .to_string(),
            derivative: "0".into(),
        },
        exterior_tail: Jet {
            value: "0.2".into(),
            derivative: "0".into(),
        },
        endpoint_tail_part: Some(Jet {
            value: "0.03".into(),
            derivative: "0".into(),
        }),
        fitted_interior_parts: vec![Jet {
            value: "0.08".into(),
            derivative: "0".into(),
        }],
        error_normalization: None,
        source_value_error: None,
        tail_value_error: None,
        source_derivative_error: None,
        root_separation_radius: None,
    });
    let r = run("signed_transform", &s, None, Some(&i));
    let v = &r.rows[0].values;
    assert_eq!(r.rows[0].ordinal, 200);
    near(&v["value_signed_interior"], 0.1, 1e-14);
    near(&v["value_signed_total"], -0.1, 1e-14);
    near(&v["value_unfitted_interior"], 0.02, 1e-14);
    near(&v["value_remaining_tail"], 0.17, 1e-14);
    // These three decimal strings were built at 192 bits but the input
    // explicitly declares 128-bit points. Their independently rounded closure
    // is nonzero; compare against those points, not the pre-rounding formula.
    let point = |text: &str| {
        Float::with_val(
            512,
            Float::with_val(i.precision_bits, Float::parse(text).unwrap()),
        )
    };
    let jet = &i.reference_jets[0];
    let expected = point(&jet.reference_full.value)
        - point(&jet.reference_window.value)
        - point(&jet.exterior_tail.value);
    assert_ne!(expected, 0);
    // Report decimals round-trip at the declared report precision. Decode that
    // point before exact promotion; the decimal spelling is not the dyadic.
    assert_eq!(
        Float::with_val(
            512,
            Float::with_val(
                r.working_precision_bits,
                Float::parse(&v["value_reference_closure_defect"]).unwrap()
            )
        ),
        expected
    );
}
#[test]
fn arithmetic_split_closes_and_preserves_signed_energy_ratios() {
    let (s, m, matrix) = fixture();
    let mut i = inputs(&m);
    for (label, value) in [("positive", "5"), ("negative", "-2")] {
        i.components.push(OperatorComponent {
            label: label.into(),
            source_digest: ContentDigest::sha256(label.as_bytes()),
            diagonal: vec![value.into(); 3],
            dense: vec![],
            rank_one: vec![],
        });
    }
    i.components_are_complete = true;
    i.deficit = Some("0.5".into());
    i.deficit_kind = Some("fuchs_approximation".into());
    let r = run("arithmetic_energy", &s, Some(&matrix), Some(&i));
    near(&r.values["sum_component_energy"], 3., 1e-14);
    near(&r.values["sum_absolute_component_energy"], 7., 1e-14);
    near(&r.values["energy_closure_defect"], 0., 1e-14);
    near(&r.values["signed_weil_over_deficit"], 6., 1e-14);
    i.components[0].diagonal[1] = "6".into();
    near(
        &run("arithmetic_energy", &s, Some(&matrix), Some(&i)).values["energy_closure_defect"],
        1.,
        1e-14,
    );
}
#[test]
fn weighted_tail_keeps_lattice_sign_and_declared_coverage() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.atom_coordinate = Some("test positive coordinate".into());
    i.atom_coverage = Some("finite supplied table, no omitted-tail bound".into());
    i.tail_checkpoints = vec!["1".into(), "3".into()];
    for (ordinal, x, w, family) in [
        (1, "1", "2", "zero"),
        (2, "2", "4", "zero"),
        (1, "3", "-1", "lattice"),
    ] {
        i.atoms.push(WeightedAtom {
            ordinal,
            coordinate: x.into(),
            weight: w.into(),
            family: family.into(),
            partition: "declared_band".into(),
        });
    }
    let r = run("weighted_tail", &s, None, Some(&i));
    let z = r
        .rows
        .iter()
        .find(|r| {
            r.label == "zero/declared_band"
                && r.values["cutoff"]
                    == "3.0000000000000000000000000000000000000000000000000000000000"
        })
        .or_else(|| {
            r.rows
                .iter()
                .rev()
                .find(|r| r.label == "zero/declared_band")
        })
        .unwrap();
    near(&z.values["included_mass"], 6., 1e-14);
    near(&z.values["weighted_inverse_moment_1"], 4., 1e-14);
    let l = r
        .rows
        .iter()
        .rev()
        .find(|r| r.label == "lattice/declared_band")
        .unwrap();
    near(&l.values["included_mass"], -1., 1e-14);
}
#[test]
fn cluster_projection_handles_nonorthogonal_vectors_and_cross_n_embedding() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.cluster = vec![
        ClusterVector {
            source_digest: ContentDigest::sha256(b"v1"),
            n_modes: 1,
            precision_bits: 128,
            eigenvalue: "3".into(),
            coefficients: vec!["0".into(), "2".into(), "0".into()],
            assembly_policy: "fixture".into(),
        },
        ClusterVector {
            source_digest: ContentDigest::sha256(b"v2"),
            n_modes: 1,
            precision_bits: 128,
            eigenvalue: "4".into(),
            coefficients: vec!["1".into(), "1".into(), "1".into()],
            assembly_policy: "fixture".into(),
        },
    ];
    i.previous_cluster = vec![ClusterVector {
        n_modes: 2,
        coefficients: vec!["0".into(), "0".into(), "1".into(), "0".into(), "0".into()],
        ..i.cluster[0].clone()
    }];
    let r = run("spectral_cluster", &s, None, Some(&i));
    assert!(number(&r.values["source_cluster_leakage_squared"]) < 1e-80);
    near(&r.rows[0].values["largest_previous_overlap"], 1., 1e-14);
}
#[test]
fn energy_allowance_never_divides_by_invalid_margin() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.energy_allowance = Some(EnergyAllowance {
        upper_trial_energy: "2".into(),
        low_block_lower_bound: "1".into(),
        high_block_lower_bound: "6".into(),
        cross_block_norm_bound: "3".into(),
        hypothesis_record_digest: ContentDigest::sha256(b"conditional fixture"),
        hypotheses: vec!["supplied point bounds, not certified".into()],
    });
    let r = run("energy_allowance", &s, None, Some(&i));
    near(&r.values["conditional_energy_allowance"], 2.25, 1e-14);
    near(&r.values["conditional_vector_allowance"], 0.75, 1e-14);
    near(
        &r.values["allowance_to_trial_energy_magnitude"],
        1.125,
        1e-14,
    );
    assert_eq!(
        number(&r.values["allowance_below_trial_energy_magnitude"]),
        0.
    );
    assert!(r.reason.as_ref().unwrap().contains("non-informative"));
    i.energy_allowance.as_mut().unwrap().high_block_lower_bound = "2".into();
    let r = run("energy_allowance", &s, None, Some(&i));
    assert_eq!(r.outcome, "sufficient_bound_unavailable");
    assert!(!r.values.contains_key("conditional_energy_allowance"));
    assert!(!r.values.contains_key("allowance_to_trial_energy_magnitude"));
}

#[test]
fn energy_allowance_scale_comparison_is_explicit_and_handles_zero_and_negative_trials() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    for (trial, cross, expected_ratio, below) in [
        ("1e-59", "2", Some(1e59), false),
        ("2", "1", Some(0.125), true),
        ("-2", "1", Some(0.125), true),
        ("2", "0", Some(0.), true),
        ("0", "1", None, false),
        ("0", "0", None, false),
    ] {
        let high: Float = Float::with_val(256, Float::parse(trial).unwrap()) + 4;
        i.energy_allowance = Some(EnergyAllowance {
            upper_trial_energy: trial.into(),
            low_block_lower_bound: "-3".into(),
            high_block_lower_bound: high.to_string_radix(10, None),
            cross_block_norm_bound: cross.into(),
            hypothesis_record_digest: ContentDigest::sha256(b"scale fixture"),
            hypotheses: vec!["finite point fixture".into()],
        });
        let r = run("energy_allowance", &s, None, Some(&i));
        assert_eq!(r.outcome, "conditional_bound_expression");
        assert!(r.convention.contains("not a certificate"));
        if let Some(expected) = expected_ratio {
            near(
                &r.values["allowance_to_trial_energy_magnitude"],
                expected,
                expected.abs() * 1e-14 + 1e-14,
            );
            assert_eq!(
                number(&r.values["allowance_below_trial_energy_magnitude"]),
                f64::from(u32::from(below))
            );
            assert_eq!(
                r.reason.as_ref().unwrap().contains("non-informative"),
                !below
            );
        } else {
            assert!(!r.values.contains_key("allowance_to_trial_energy_magnitude"));
            assert!(!r
                .values
                .contains_key("allowance_below_trial_energy_magnitude"));
            assert!(r.reason.as_ref().unwrap().contains("trial energy is zero"));
        }
    }
}
#[test]
fn missing_inputs_foreign_sources_and_resource_limits_are_explicit() {
    let (s, m, _) = fixture();
    for id in [
        "weighted_reference_projection",
        "signed_transform",
        "weighted_tail",
        "spectral_cluster",
        "energy_allowance",
    ] {
        let r = run(id, &s, None, None);
        assert_eq!(r.outcome, "missing_input");
        assert!(r.reason.is_some());
    }
    let mut i = inputs(&m);
    i.source_eigenpair = ContentDigest::sha256(b"wrong state");
    assert!(capture_extended(
        "weighted_tail",
        &s,
        None,
        None,
        Some(&i),
        &ExtensionOptions::for_source(&s),
        &[],
        &context()
    )
    .is_err());
    let mut o = ExtensionOptions::for_source(&s);
    o.working_precision_bits = 64;
    assert!(capture_extended("compactness", &s, None, None, None, &o, &[], &context()).is_err());
}
#[test]
fn extended_analysis_is_deterministic_across_worker_counts() {
    let (s, _, _) = fixture();
    let run_pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| serde_json::to_value(run("compactness", &s, None, None)).unwrap())
    };
    assert_eq!(run_pool(1), run_pool(4));
}
#[test]
fn resolution_refuses_to_invent_source_error_or_an_n_threshold() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    let t = (Float::with_val(192, Constant::Pi) * 2u32 / Float::with_val(192, 9).ln()).to_string();
    i.reference_jets.push(ReferenceJet {
        matched_root_ordinal: None,
        ordinal: 1,
        t,
        reference_window: Jet {
            value: "0".into(),
            derivative: "0".into(),
        },
        reference_full: Jet {
            value: "0".into(),
            derivative: "0".into(),
        },
        exterior_tail: Jet {
            value: "0".into(),
            derivative: "0".into(),
        },
        endpoint_tail_part: None,
        fitted_interior_parts: vec![],
        error_normalization: None,
        source_value_error: None,
        tail_value_error: None,
        source_derivative_error: None,
        root_separation_radius: None,
    });
    let r = run("resolution_budget", &s, None, Some(&i));
    assert_ne!(r.rows[0].outcome, "conditional_budget_met");
    near(&r.values["conditional_contiguous_prefix"], 0., 1e-14);
    assert!(!r.rows[0]
        .values
        .contains_key("spacing_normalized_displacement"));
    let mut next = i.reference_jets[0].clone();
    next.ordinal = 2;
    next.t = (Float::with_val(192, Float::parse(&next.t).unwrap()) + 2u32).to_string();
    i.reference_jets.push(next);
    let r = run("resolution_budget", &s, None, Some(&i));
    near(&r.rows[0].values["reference_neighbor_spacing"], 2., 1e-12);
    assert!(r.rows[0]
        .values
        .contains_key("spacing_normalized_newton_correction"));
    assert!(!r.rows[0]
        .values
        .contains_key("spacing_normalized_displacement"));
    i.reference_jets[0].matched_root_ordinal = Some(1);
    let r = run("resolution_budget", &s, None, Some(&i));
    assert!(r.rows[0]
        .notes
        .iter()
        .any(|s| s.contains("join unavailable")));
}

#[test]
fn directional_energy_matches_independent_projected_resolvent_calculation() {
    let p = 192;
    let co = ["1", "3", "1"];
    let norm = 11f64;
    let entries = (0..3)
        .flat_map(|a| {
            (0..3).map(move |b| {
                (Float::with_val(p, if a == b { 2 } else { 0 })
                    - Float::with_val(
                        p,
                        co[a].parse::<u32>().unwrap() * co[b].parse::<u32>().unwrap(),
                    ) / 11u32)
                    .to_string()
            })
        })
        .collect::<Vec<_>>();
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":entries}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":co}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let state =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let (mut sec, secbytes) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: sm.key.clone(),
        content_digest: sm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let t = Float::with_val(p, Constant::Pi) * 2u32 * (Float::with_val(p, 3) / 5u32).sqrt()
        / Float::with_val(p, 9).ln();
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":1,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":t.to_string()}},{"status":"converged","details":{"value":"0"}}]}),
    );
    rm.dependencies.push(DependencyRef {
        key: sec.key.clone(),
        content_digest: sec.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let roots = RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &secbytes,
        &state,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap();
    let mut i = inputs(&sm);
    i.perturbations.push(OperatorComponent {
        label: "diagonal".into(),
        source_digest: ContentDigest::sha256(b"D"),
        diagonal: vec!["1".into(), "0".into(), "1".into()],
        dense: vec![],
        rank_one: vec![],
    });
    let o = ExtensionOptions::for_source(&state);
    let result = capture_extended(
        "directional_response",
        &state,
        Some(&matrix),
        Some(&roots),
        Some(&i),
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    let v = [1f64 / norm.sqrt(), 3f64 / norm.sqrt(), 1f64 / norm.sqrt()];
    let rv = [v[0] / (-0.4), v[1] / 0.6, v[2] / (-0.4)];
    let dot = v.iter().zip(rv).map(|(a, b)| a * b).sum::<f64>();
    let x = [rv[0] - dot * v[0], rv[1] - dot * v[1], rv[2] - dot * v[2]];
    let k = x.iter().map(|x| x * x).sum::<f64>();
    near(&result.rows[0].values["directional_energy"], k, 1e-12);
    near(
        &result.rows[0].values["conditional_tau_response_diagonal"],
        -(x[0] * v[0] + x[2] * v[2]) / k,
        1e-13,
    );
    assert_eq!(result.rows[1].outcome, "carrier_or_unresolved");
    let action = vec![v[0].to_string(), "0".into(), v[2].to_string()];
    i.run_once=Some(serde_json::from_value(json!({"derivative_actions":[
        {"label":"tau_total","source_digest":ContentDigest::sha256(b"total fixture"),"action":action,"convention":"dTau/dlogC synthetic diagonal"},
        {"label":"tau_fixture","source_digest":ContentDigest::sha256(b"component fixture"),"action":action,"convention":"single complete synthetic component"}],"log_cutoff_velocity":"1"})).unwrap());
    let limited = ExtensionOptions {
        maximum_directional_rows: 1,
        ..o.clone()
    };
    let transported = capture_extended(
        "root_transport",
        &state,
        Some(&matrix),
        Some(&roots),
        Some(&i),
        &limited,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(transported.rows.len(), 2);
    assert_eq!(transported.rows[1].outcome, "carrier_or_unresolved"); // Not omitted by the legacy one-row work budget.
    let t = t.to_f64();
    let tau_velocity = -(x[0] * v[0] + x[2] * v[2]) / k;
    near(
        &transported.rows[0].values["conditional_total_physical_velocity"],
        2.0 * std::f64::consts::PI.powi(2) * tau_velocity / (t * 9f64.ln().powi(2)) - t / 9f64.ln(),
        1e-12,
    );
    near(
        &transported.rows[0].values["forcing_closure_defect"],
        0.0,
        1e-14,
    );
    let o = ExtensionOptions {
        maximum_directional_rows: 1,
        ..o
    };
    let result = capture_extended(
        "directional_response",
        &state,
        Some(&matrix),
        Some(&roots),
        None,
        &o,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.rows[1].outcome, "budget_limited");
}
#[test]
fn qualified_missing_record_never_becomes_a_completed_measurement() {
    let (s, _, _) = fixture();
    let result = capture_extended(
        "weighted_reference_projection",
        &s,
        None,
        None,
        None,
        &ExtensionOptions::for_source(&s),
        &[],
        &context(),
    )
    .unwrap();
    let d = CapturedDiagnostic::new(&result.value, vec![]).unwrap();
    assert!(matches!(d.qualified(), Err(CaptureFailure::Missing { .. })));
}

#[test]
fn complex_transform_matches_the_entire_constant_integral() {
    let (s, _, _) = fixture();
    let r = run("complex_transform", &s, None, None);
    let l = 9f64.ln();
    assert_eq!(r.rows.len(), 70);
    near(&r.rows[2].values["value_re"], l.sqrt(), 1e-14);
    near(
        &r.rows[4].values["value_re"],
        2.0 * (l / 2.0).sinh() / l.sqrt(),
        1e-14,
    );
    near(&r.rows[4].values["value_im"], 0.0, 1e-14);
    near(
        &r.rows[4].values["derivative_im"],
        -2.0 * ((l / 2.0) * (l / 2.0).cosh() - (l / 2.0).sinh()) / l.sqrt(),
        1e-14,
    );
    assert!(r.convention.contains("not contour certificates"));
    assert_eq!(
        r.rows
            .iter()
            .filter(|r| r.label == "contour_sample")
            .count(),
        65
    );
    assert_eq!(r.rows[5].values, r.rows[69].values);
}
#[test]
fn every_parent_prefix_and_independent_comparison_remain_distinct() {
    let (s, sm, m) = fixture();
    let r = run("finite_section_transfer", &s, Some(&m), None);
    assert_eq!(r.rows.len(), 2);
    for row in &r.rows {
        near(&row.values["retained_mass"], 1.0, 1e-14);
        near(&row.values["high_forcing_squared"], 0.0, 1e-14);
        near(&row.values["truncated_energy"], 3.0, 1e-14);
    }
    let mut i = inputs(&sm);
    i.run_once=Some(serde_json::from_value(json!({"comparison":{"source_digest":ContentDigest::sha256(b"comparison-state"),"matrix_digest":ContentDigest::sha256(b"comparison-matrix"),"lambda_squared":"9","n_modes":0,"precision_bits":128,"coefficients":["1"],"matrix":["2"],"eigenvalue":"2","assembly_policy":"independent synthetic comparison"}})).unwrap());
    let r = run("finite_section_transfer", &s, Some(&m), Some(&i));
    near(
        &r.values["comparison_block_frobenius_difference"],
        1.0,
        1e-14,
    );
    near(
        &r.values["comparison_state_signed_block_defect"],
        1.0,
        1e-14,
    );
}
#[test]
fn cluster_feedback_matches_a_two_by_two_schur_complement() {
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,"entries":["4","0","1","0","3","0","1","0","6"]}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"3","eigenvector":["0","1","0"]}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let m =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let r = run("operator_cluster", &s, Some(&m), None);
    assert_eq!(r.outcome, "point_measurement");
    near(&r.rows[3].values["compressed_operator"], 6.0, 1e-14);
    near(&r.rows[3].values["signed_complement_feedback"], 1.0, 1e-14);
    near(&r.rows[3].values["effective_operator"], 5.0, 1e-14);
}
#[test]
fn fixed_tail_model_solves_without_borrowing_retained_energy() {
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    i.run_once=Some(serde_json::from_value(json!({"tail_form":{"definition_digest":ContentDigest::sha256(b"fixed synthetic basis"),"dimension":2,"finite_zero_form":["1","0","0","3"],"tail_correction":["0.5","0.25","0.25","1"],"lattice_gram":["2","0","0","1"],"tail_operator_error":null,"coverage":"synthetic finite atoms; no infinite tail","hypotheses":[]}})).unwrap());
    let r = run("tail_operator", &s, None, Some(&i));
    near(
        &r.values["model_energy"],
        4.75 - (3.25f64 * 3.25 + 0.125).sqrt(),
        1e-12,
    );
    near(&r.values["model_energy_without_tail"], 1.0, 1e-12);
    near(&r.values["lattice_gram_pivot_ratio"], 2.0, 1e-12);
    near(&r.values["model_vector_lattice_norm_squared"], 1.0, 1e-12);
    assert!(number(&r.values["model_vector_relative_residual"]) < 1e-20);
    near(
        &r.values["tail_energy_lift"],
        number(&r.values["model_energy"]) - 1.0,
        1e-12,
    );
    assert!(
        (number(&r.values["model_vector_zero_energy"])
            + number(&r.values["model_vector_tail_energy"])
            - number(&r.values["model_energy"]))
        .abs()
            < 1e-12
    );
    let (mut other, bytes) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"999","eigenvector":["0","1","0"]}),
    );
    other.dependencies = sm.dependencies.clone();
    let otherstate =
        RetainedState::from_payload(&other, &bytes, std::slice::from_ref(&other.content_digest))
            .unwrap();
    i.source_eigenpair = other.content_digest;
    let b = run("tail_operator", &otherstate, None, Some(&i));
    assert_eq!(r.values["model_energy"], b.values["model_energy"]);
    let mut scaled = i.clone();
    let form = scaled
        .run_once
        .as_mut()
        .unwrap()
        .tail_form
        .as_mut()
        .unwrap();
    for x in form
        .finite_zero_form
        .iter_mut()
        .chain(&mut form.tail_correction)
    {
        *x = (Float::with_val(192, Float::parse(x.as_str()).unwrap())
            * Float::with_val(192, Float::parse("1e-100").unwrap()))
        .to_string();
    }
    let tiny = run("tail_operator", &otherstate, None, Some(&scaled));
    // Scaling the pencil must never turn a poor relative vector into a qualified one.
    let residual = number(&tiny.values["model_vector_relative_residual"]);
    assert!(residual < 1e-25 || tiny.outcome == "partial_unresolved");
    assert!(tiny.values.contains_key("model_energy_without_tail"));

    i.run_once
        .as_mut()
        .unwrap()
        .tail_form
        .as_mut()
        .unwrap()
        .lattice_gram[0] = "0".into();
    assert_eq!(
        run("tail_operator", &otherstate, None, Some(&i)).outcome,
        "unresolved"
    );
}
#[test]
fn missing_error_source_does_not_become_a_certificate() {
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    let a = run("observable_budget", &s, None, None);
    assert!(!a.values.contains_key("conditional_origin_lower_margin"));
    i.run_once=Some(serde_json::from_value(json!({"uncertainty":{"unit_state_l2_error":"0.01","source_certificate_digest":ContentDigest::sha256(b"synthetic conditional source"),"hypotheses":["fixture allowance only"]}})).unwrap());
    let b = run("observable_budget", &s, None, Some(&i));
    near(
        &b.values["conditional_origin_lower_margin"],
        0.99 * 9f64.ln().sqrt(),
        1e-14,
    );
}

#[test]
fn cluster_tiny_workspace_limit_withholds_unfunded_vector_arithmetic() {
    let (s, _, m) = fixture();
    let mut options = ExtensionOptions::for_source(&s);
    options.maximum_working_bytes = Some(1);
    let r = capture_extended(
        "operator_cluster",
        &s,
        Some(&m),
        None,
        None,
        &options,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.reason.unwrap().contains("working-byte budget"));
    assert!(r.values.is_empty() && r.rows.is_empty());
}

#[test]
fn signed_band_recurrence_matches_discrete_gaussian_nodes_and_keeps_failure() {
    use xc_spectral::ccm::research_completion::*;
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    let model = SignedBandModel {
        degree: 2,
        coordinate: "synthetic x".into(),
        definition_digest: ContentDigest::sha256(b"three equal atoms"),
        atoms: ["-1", "0", "1"]
            .iter()
            .map(|x| BandAtom {
                coordinate: (*x).into(),
                signed_weight: "1".into(),
                family: "synthetic".into(),
            })
            .collect(),
        coverage: "complete finite three-atom measure".into(),
        hypotheses: vec!["finite positive measure".into()],
        borrowed_inputs: vec![],
        input_energy: None,
        scoring_roots: vec![],
    };
    let once = xc_spectral::ccm::convergence_capture::RunOnceInputs {
        completion: Some(CompletionInputs {
            band: Some(model),
            ..Default::default()
        }),
        ..Default::default()
    };
    i.run_once = Some(once);
    let result = run("band_reconstruction", &s, None, Some(&i));
    assert_eq!(result.outcome, "point_measurement");
    let mut roots = result
        .rows
        .iter()
        .map(|r| number(&r.values["model_band_root"]))
        .collect::<Vec<_>>();
    roots.sort_by(f64::total_cmp);
    assert!((roots[0] + (2f64 / 3.).sqrt()).abs() < 1e-12);
    assert!((roots[1] - (2f64 / 3.).sqrt()).abs() < 1e-12);
    near(&result.values["band_inverse_moment_one"], 0.0, 1e-12);
    near(&result.values["band_inverse_moment_two"], 3.0, 1e-12);
    near(&result.values["band_inverse_moment_three"], 0.0, 1e-12);
    let m = i
        .run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .band
        .as_mut()
        .unwrap();
    m.atoms[1].signed_weight = "-3".into();
    assert_eq!(
        run("band_reconstruction", &s, None, Some(&i)).outcome,
        "unresolved"
    );
}
#[test]
fn independently_prepared_prime_action_exposes_an_error_hidden_by_reconstruction() {
    use xc_spectral::ccm::{convergence_capture::*, research_completion::*};
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    let action = |label: &str, value: &str| OperatorAction {
        label: label.into(),
        source_digest: ContentDigest::sha256(label.as_bytes()),
        action: vec!["0".into(), value.into(), "0".into()],
        convention: "same unit state".into(),
    };
    i.run_once = Some(RunOnceInputs {
        component_actions: vec![action("tau_prime_reconstructed", "2")],
        completion: Some(CompletionInputs {
            independent_actions: vec![action("tau_prime_direct", "3")],
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("consistency", &s, None, Some(&i));
    near(&r.rows[0].values["signed_energy_difference"], 1., 1e-14);
    near(&r.rows[0].values["action_difference_norm"], 1., 1e-14);
}
#[test]
fn source_independent_reference_preparation_generates_only_a_finite_target() {
    use xc_spectral::ccm::research_completion::*;
    let (s, _, _) = fixture();
    let reference = ReferenceSpec {
        schema_version: 1,
        definition: "finite constant".into(),
        lambda_squared: "9".into(),
        precision_bits: 128,
        coefficients: vec!["1".into()],
        approximation_scope: "finite compact support".into(),
    };
    let prep = ReferencePreparation {
        schema_version: 1,
        finite_reference: Some(ResearchInputs {
            schema_version: 1,
            reference,
            basis: vec![],
            projection: ProjectionOptions {
                working_precision_bits: 192,
                normalization: "center_one".into(),
                fixed_second_component: None,
            },
        }),
        sampled_reference: None,
        lambda_squared: "9".into(),
        precision_bits: 128,
        definition_digest: ContentDigest::sha256(b"external constant"),
        approximation_scope: "finite compact support".into(),
        weighted_atoms: vec![],
        atom_coordinate: None,
        atom_coverage: None,
        tail_form: None,
        tail_recipe: None,
        completion: None,
    };
    let input = prep.prepare(&s, None).unwrap();
    assert!(input
        .target
        .as_ref()
        .unwrap()
        .values
        .iter()
        .all(|x| number(x) == 1.));
    assert!(input.reference_jets.is_empty());
    let r = run("weighted_reference_projection", &s, None, Some(&input));
    near(&r.values["weighted_l2_squared"], 0., 1e-14);
}
#[test]
fn output_budget_exhaustion_is_retained_as_an_explicit_diagnostic_outcome() {
    let (s, _, _) = fixture();
    let mut options = ExtensionOptions::for_source(&s);
    options.maximum_estimated_output_bytes = 1;
    let r = capture_extended(
        "compactness",
        &s,
        None,
        None,
        None,
        &options,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.outcome, "unresolved");
    assert!(r.reason.unwrap().contains("resource budget"));
}
#[test]
#[cfg(feature = "arb")]
fn certified_contour_counts_constant_profile_roots_and_never_certifies_exhausted_boundary() {
    use xc_spectral::ccm::{convergence_capture::*, research_completion::*};
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    i.run_once = Some(RunOnceInputs {
        completion: Some(CompletionInputs {
            contour: Some(ContourPolicy {
                left: "-1".into(),
                right: "7".into(),
                bottom: "-0.5".into(),
                top: "0.5".into(),
                maximum_depth: 20,
                maximum_segments: 2048,
            }),
            ..Default::default()
        }),
        ..Default::default()
    });
    let result = run("transform_enclosure", &s, None, Some(&i));
    assert_eq!(
        result.outcome, "certified_finite_enclosure",
        "{:?}",
        result.reason
    );
    near(&result.values["certified_finite_zero_count"], 2., 1e-14);
    assert!(result
        .convention
        .contains("no eigenstate/assembly accuracy claim"));
    let c = i
        .run_once
        .as_mut()
        .unwrap()
        .completion
        .as_mut()
        .unwrap()
        .contour
        .as_mut()
        .unwrap();
    c.bottom = "0".into();
    c.maximum_depth = 0;
    c.maximum_segments = 4;
    let result = run("transform_enclosure", &s, None, Some(&i));
    assert_eq!(result.outcome, "partial_unresolved");
    assert!(!result.values.contains_key("certified_finite_zero_count"));
}

#[test]
fn independent_cohort_measures_forcing_and_withholds_unmatched_roots() {
    use xc_spectral::ccm::{convergence_capture::*, research_completion::*};
    let (s, sm, m) = fixture();
    let mut i = inputs(&sm);
    let deferred = run("configuration_comparison", &s, Some(&m), Some(&i));
    assert_eq!(deferred.outcome, "awaiting_cohort");
    assert!(deferred.rows.is_empty());
    assert!(deferred.reason.unwrap().contains("afterward"));
    i.run_once = Some(RunOnceInputs {
        completion: Some(CompletionInputs {
            comparisons: vec![ComparisonSnapshot {
                state: ComparisonState {
                    source_digest: ContentDigest::sha256(b"small state"),
                    matrix_digest: ContentDigest::sha256(b"small matrix"),
                    lambda_squared: "9".into(),
                    n_modes: 0,
                    precision_bits: 128,
                    eigenvalue: "2".into(),
                    coefficients: vec!["1".into()],
                    matrix: vec!["2".into()],
                    assembly_policy: "synthetic".into(),
                },
                selection_policy: "even ground".into(),
                assembly_policy: "synthetic".into(),
                quadrature_policy: "exact".into(),
                root_branch: "different branch".into(),
                root_coordinate: "mellin_t".into(),
                roots: vec![],
            }],
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("configuration_comparison", &s, Some(&m), Some(&i));
    near(&r.rows[0].values["absolute_unit_overlap"], 1., 1e-14);
    near(
        &r.rows[0].values["independent_small_state_low_residual_squared"],
        1.,
        1e-14,
    );
    near(
        &r.rows[0].values["independent_small_state_high_forcing_squared"],
        0.,
        1e-14,
    );
    assert!(r.rows[0]
        .notes
        .iter()
        .any(|n| n.contains("ordinal differences withheld")));
}
#[test]
fn finite_atom_tail_builder_keeps_omitted_tail_qualification() {
    use xc_spectral::ccm::research_completion::*;
    let atoms: Vec<WeightedAtom> = serde_json::from_value(json!([
        {"ordinal":1,"coordinate":"-1","weight":"2","family":"zero","partition":"finite"},
        {"ordinal":2,"coordinate":"1","weight":"2","family":"zero","partition":"finite"},
        {"ordinal":1,"coordinate":"0","weight":"1","family":"lattice","partition":"finite"}
    ]))
    .unwrap();
    let form = prepare_tail_form(
        ContentDigest::sha256(b"finite test"),
        &[vec!["1".into()], vec!["0".into(), "1".into()]],
        &atoms,
        None,
        "three finite atoms",
        &[],
        128,
    )
    .unwrap();
    near(&form.finite_zero_form[0], 4., 1e-14);
    near(&form.finite_zero_form[3], 4., 1e-14);
    near(&form.lattice_gram[0], 1., 1e-14);
    assert!(form
        .hypotheses
        .iter()
        .any(|h| h.contains("unknown, not proved zero")));
    assert!(prepare_tail_form(
        ContentDigest::sha256(b"finite test"),
        &[vec!["1".into()], vec!["0".into(), "1".into()]],
        &atoms,
        Some(&["0".into(), "1".into(), "2".into(), "0".into()]),
        "three finite atoms",
        &[],
        128
    )
    .is_err());
}

#[test]
fn spacing_displacement_requires_an_explicit_join_and_ordered_reference() {
    let (state, sm, _) = fixture();
    let (mut sec, sb) = source(
        "ccm_secular_source",
        json!({"schema_version":1,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenpair_content_digest":sm.content_digest,"normalization":"sum_xi_equals_sqrt_log_lambda_squared"}),
    );
    sec.dependencies.push(DependencyRef {
        key: sm.key.clone(),
        content_digest: sm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let (mut rm, rb) = source(
        "ccm_root_discovery_window",
        json!({"schema_version":5,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"first_root_index":3,"discovery_mode":"independent","reference_seeds_used":false,"completeness":"partial","outcomes":[{"status":"converged","details":{"value":"12"}}]}),
    );
    rm.dependencies.push(DependencyRef {
        key: sec.key.clone(),
        content_digest: sec.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let roots = RetainedRoots::from_payload(
        &rm,
        &rb,
        &sec,
        &sb,
        &state,
        &[rm.content_digest.clone(), sec.content_digest.clone()],
    )
    .unwrap();
    let mut i = inputs(&sm);
    for (ordinal, t) in [(10, "11"), (11, "13")] {
        i.reference_jets.push(ReferenceJet {
            matched_root_ordinal: if ordinal == 10 { Some(3) } else { None },
            ordinal,
            t: t.into(),
            reference_window: Jet {
                value: "0".into(),
                derivative: "0".into(),
            },
            reference_full: Jet {
                value: "0".into(),
                derivative: "0".into(),
            },
            exterior_tail: Jet {
                value: "0".into(),
                derivative: "0".into(),
            },
            endpoint_tail_part: None,
            fitted_interior_parts: vec![],
            error_normalization: None,
            source_value_error: None,
            tail_value_error: None,
            source_derivative_error: None,
            root_separation_radius: None,
        });
    }
    let evaluate = |i: &ExternalResearchInputs| {
        capture_extended(
            "resolution_budget",
            &state,
            None,
            Some(&roots),
            Some(i),
            &ExtensionOptions::for_source(&state),
            &[],
            &context(),
        )
        .unwrap()
        .value
        .data
    };
    let r = evaluate(&i);
    near(
        &r.rows[0].values["spacing_normalized_displacement"],
        0.5,
        1e-12,
    );
    i.reference_jets[0].matched_root_ordinal = None;
    assert!(!evaluate(&i).rows[0]
        .values
        .contains_key("matched_root_reference_displacement"));
    i.reference_jets[0].matched_root_ordinal = Some(3);
    i.reference_jets[1].t = "11".into();
    assert!(!evaluate(&i).rows[0]
        .values
        .contains_key("spacing_normalized_displacement"));
}

#[test]
fn signed_atom_sums_keep_exclusion_singularity_and_fixed_cutoff_ladders() {
    use xc_spectral::ccm::research_completion::*;
    let (s, sm, _) = fixture();
    let mut i = inputs(&sm);
    i.atom_coordinate = Some("synthetic x".into());
    i.atom_coverage = Some("finite fixture only".into());
    i.atoms = serde_json::from_value(serde_json::json!([
        {"ordinal":1,"coordinate":"1","weight":"2","family":"zero","partition":"band"},
        {"ordinal":2,"coordinate":"2","weight":"3","family":"zero","partition":"band"},
        {"ordinal":1,"coordinate":"3","weight":"1","family":"lattice","partition":"band"}
    ]))
    .unwrap();
    let policy=serde_json::from_value(serde_json::json!({"maximum_atoms":300000,"maximum_input_bytes":1000000,"evaluations":[
        {"ordinal":1,"coordinate":"3","label":"pole retained"},
        {"ordinal":2,"coordinate":"3","label":"self atom excluded","exclude":[{"family":"lattice","partition":"band","ordinal":1}]}
    ],"cutoffs":["1","2"],"tail_recipe":{"basis_polynomials":[["1"]],"tail_correction":null,"hypotheses":["finite fixture"]}})).unwrap();
    let band = SignedBandModel {
        degree: 1,
        coordinate: "synthetic x".into(),
        definition_digest: ContentDigest::sha256(b"finite band"),
        atoms: vec![
            BandAtom {
                coordinate: "1".into(),
                signed_weight: "2".into(),
                family: "zero".into(),
            },
            BandAtom {
                coordinate: "2".into(),
                signed_weight: "3".into(),
                family: "zero".into(),
            },
        ],
        coverage: "finite fixture".into(),
        hypotheses: vec!["finite positive measure".into()],
        borrowed_inputs: vec![],
        input_energy: None,
        scoring_roots: vec![],
    };
    i.run_once = Some(xc_spectral::ccm::convergence_capture::RunOnceInputs {
        completion: Some(CompletionInputs {
            atom_analysis: Some(policy),
            band: Some(band),
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = run("weighted_tail", &s, None, Some(&i));
    let zero = r
        .rows
        .iter()
        .find(|r| {
            r.label == "signed_atom_kernel"
                && r.ordinal == 2
                && r.notes.iter().any(|n| n.contains("family zero"))
        })
        .unwrap();
    assert_eq!(number(&zero.values["signed_kernel_1"]), -4.0);
    assert_eq!(number(&zero.values["signed_kernel_2"]), 3.5);
    assert_eq!(number(&zero.values["signed_kernel_3"]), -3.25);
    assert!(r.rows.iter().any(|r| r.label == "signed_atom_kernel"
        && r.ordinal == 1
        && r.outcome == "unresolved_denominator"
        && !r.values.contains_key("signed_kernel_1")));
    let band = run("band_reconstruction", &s, None, Some(&i));
    assert_eq!(
        number(&band.values["supplied_zero_extent_covers_model_band"]),
        1.0
    );
    let cuts = band
        .rows
        .iter()
        .filter(|r| r.label == "band_cutoff")
        .collect::<Vec<_>>();
    assert_eq!(cuts.len(), 2);
    assert_eq!(number(&cuts[0].values["first_model_band_root"]), 1.0);
    assert!((number(&cuts[1].values["first_model_band_root"]) - 1.6).abs() < 1e-14);
    let tail = run("tail_operator", &s, None, Some(&i));
    let cuts = tail
        .rows
        .iter()
        .filter(|r| r.label == "tail_model_cutoff")
        .collect::<Vec<_>>();
    assert_eq!(cuts.len(), 2);
    assert_eq!(number(&cuts[0].values["model_energy"]), 4.0);
    assert_eq!(number(&cuts[1].values["model_energy"]), 10.0);
}

#[test]
fn band_scoring_requires_a_complete_ordered_coordinate_match() {
    use xc_spectral::ccm::research_completion::*;
    let mut c: CompletionInputs = serde_json::from_value(serde_json::json!({"band": {
        "degree":2,"coordinate":"synthetic x","definition_digest":ContentDigest::sha256(b"audit scoring"),
        "atoms":[{"coordinate":"1","signed_weight":"1","family":"zero"},{"coordinate":"2","signed_weight":"1","family":"zero"},{"coordinate":"3","signed_weight":"1","family":"zero"}],
        "coverage":"finite fixture","hypotheses":["finite positive functional"],"borrowed_inputs":[],"input_energy":null,"scoring_roots":[]
    }})).unwrap();
    assert!(c.validate(3, 192).is_ok());
    for bad in [
        vec!["1"],
        vec!["2", "1"],
        vec!["1", "1"],
        vec!["1", "2", "3"],
    ] {
        c.band.as_mut().unwrap().scoring_roots = bad.into_iter().map(str::to_owned).collect();
        assert!(c.validate(3, 192).is_err());
    }
    c.band.as_mut().unwrap().scoring_roots = vec!["1".into(), "2".into()];
    assert!(c.validate(3, 192).is_ok());
}

#[test]
fn capability_dependent_diagnostics_do_not_reuse_other_or_unspecified_backends() {
    let root_dir = xc_core::test_support::TestDir::new("ccm-feature-identity");
    let root = root_dir.to_path_buf();
    let (state, _, _) = fixture();
    let options = ExtensionOptions::for_source(&state);
    for id in ["transform_enclosure", "band_reconstruction"] {
        let initial =
            capture_extended(id, &state, None, None, None, &options, &[], &context()).unwrap();
        assert_eq!(
            initial.value.request["arb_available"],
            json!(cfg!(feature = "arb"))
        );
        let producer = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(FilesystemCacheStore::new(
                "feature-test",
                root.join(id),
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
        let mut cache = ArtifactCacheContext {
            resolver: Some(&producer),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["feature-test".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        };
        // Produce an authentic envelope in an isolated cache, then populate a
        // different cache with only the opposite-capability and legacy requests.
        let current =
            capture_extended(id, &state, None, None, None, &options, &[], &cache).unwrap();
        let manifest = current.produced_manifest.unwrap();
        let foreign = FilesystemCacheStore::new(
            "foreign",
            root.join(format!("{id}-foreign")),
            true,
            CacheVisibility::Local,
        );
        for capability in [Some(!cfg!(feature = "arb")), None] {
            let mut value = serde_json::to_value(&current.value).unwrap();
            match capability {
                Some(available) => {
                    value["request"]["arb_available"] = json!(available);
                }
                None => {
                    value["request"]
                        .as_object_mut()
                        .unwrap()
                        .remove("arb_available");
                }
            }
            let mut semantic: SemanticKeyEnvelope =
                serde_json::from_str(&manifest.tags[SEMANTIC_KEY_MANIFEST_TAG]).unwrap();
            semantic.resolved_mathematical_parameters["request"] = value["request"].clone();
            let logical = format!(
                "ccm/research/{}/{}",
                manifest.key.kind,
                ContentDigest::sha256(&serde_json::to_vec(&semantic).unwrap()).0
            );
            let mut tags = manifest.tags.clone();
            tags.insert(
                SEMANTIC_KEY_MANIFEST_TAG.into(),
                serde_json::to_string(&semantic).unwrap(),
            );
            foreign
                .put(
                    &ArtifactDraft {
                        schema_version: manifest.schema_version,
                        key: ArtifactKey {
                            kind: manifest.key.kind.clone(),
                            logical_key: logical,
                            parameters_digest: semantic.digest().unwrap(),
                        },
                        producer_toolkit_version: manifest.producer_toolkit_version.clone(),
                        minimum_reader_version: manifest.minimum_reader_version.clone(),
                        maximum_reader_version: None,
                        quality: manifest.quality,
                        visibility: manifest.visibility,
                        immutable: true,
                        dependencies: manifest.dependencies.clone(),
                        tags,
                        provenance_digest: None,
                    },
                    &serde_json::to_vec(&value).unwrap(),
                )
                .unwrap();
        }
        let foreign_resolver = CacheResolver::new(vec![CacheLayer {
            precedence: 0,
            store: Box::new(foreign),
        }]);
        cache.resolver = Some(&foreign_resolver);
        cache.mode = ArtifactExecutionCacheMode::RequireReuse;
        cache.write_on_miss = false;
        assert!(capture_extended(id, &state, None, None, None, &options, &[], &cache).is_err());
        cache.resolver = Some(&producer);
        let warm = capture_extended(id, &state, None, None, None, &options, &[], &cache).unwrap();
        assert!(warm.reused_manifest.is_some());
        assert_eq!(
            serde_json::to_value(warm.value).unwrap(),
            serde_json::to_value(current.value).unwrap()
        );
    }
    // This is a unique test-owned child of the platform temporary directory.
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn weighted_tail_reports_origin_mass_and_keeps_nonzero_inverse_moments() {
    let (s, m, _) = fixture();
    let mut i = inputs(&m);
    i.atom_coordinate = Some("test lattice coordinate".into());
    i.atom_coverage = Some("finite supplied table, no omitted-tail bound".into());
    i.tail_checkpoints = vec!["0".into(), "1".into(), "4".into()];
    for (ordinal, x, w, partition) in [
        (1, "0", "3", "origin;ordinal=mode+1"),
        (2, "1", "2", "nonzero_modes;ordinal=mode+1"),
        (3, "4", "1", "nonzero_modes;ordinal=mode+1"),
    ] {
        i.atoms.push(WeightedAtom {
            ordinal,
            coordinate: x.into(),
            weight: w.into(),
            family: "lattice".into(),
            partition: partition.into(),
        });
    }
    let r = run("weighted_tail", &s, None, Some(&i));
    let origin = r
        .rows
        .iter()
        .filter(|row| row.label == "lattice/origin;ordinal=mode+1")
        .collect::<Vec<_>>();
    let nonzero = r
        .rows
        .iter()
        .filter(|row| row.label == "lattice/nonzero_modes;ordinal=mode+1")
        .collect::<Vec<_>>();
    assert!(!origin.is_empty() && !nonzero.is_empty());
    for row in &origin {
        assert_eq!(row.outcome, "point_measurement");
        assert!(row.values.contains_key("included_mass"));
        assert!(!row.values.contains_key("weighted_inverse_moment_1"));
        assert!(row
            .notes
            .iter()
            .any(|n| n.contains("do not apply at the origin")));
    }
    let last = nonzero.last().unwrap();
    assert_eq!(last.outcome, "point_measurement");
    assert!(last.values.contains_key("weighted_inverse_moment_1"));
}
#[test]
fn automatic_finite_recipes_keep_fixed_proposals_and_need_no_future_results() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, sm, matrix) = fixture();
    let mut input = inputs(&sm);
    input.target = Some(SampledReference {
        definition_digest: ContentDigest::sha256(b"constant finite test reference"),
        evaluation_policy: "constant exact test points".into(),
        approximation_scope: "finite constant function only".into(),
        intervals: 8,
        values: vec!["1".into(); 9],
        basis_values: vec![],
        fixed_second_component: None,
        raw_normalizer: "1".into(),
        trial_coefficients: Some(vec!["0".into(), "1".into(), "0".into()]),
    });
    let analyze = |id, input| {
        f::capture(id, &s, Some(&matrix), None, input, None, &[], &context())
            .unwrap()
            .value
            .data
    };
    let cluster = analyze("spectral_cluster_bound", None);
    // The fixed window [-3,9] contains both even eigenvalues 3 and 4.
    // It must not silently shrink until a rank-one certificate succeeds.
    assert_eq!(cluster.rows[0].outcome, "unresolved");
    for id in ["finite_tail_bound", "directional_error_bound"] {
        assert_eq!(
            analyze(id, None).rows[0].outcome,
            "certified_finite_enclosure",
            "{id}"
        );
    }
    let fit = analyze("constrained_l1_fit", Some(&input));
    assert_eq!(fit.result["fit"]["witness"]["coefficients"][0], "1");
    assert_eq!(fit.result["basis_dimension"], 1);
    assert_eq!(fit.result["equality_constraints"], 0);
    let profile = analyze("continuous_l1_bound", Some(&input));
    assert_eq!(profile.rows.len(), 2);
    for row in profile.rows {
        assert_eq!(row.outcome, "certified_finite_enclosure");
        assert_eq!(
            row.result["report"]["output"]["output"]["weighted_l1"]["upper"], "0",
            "{row:?}"
        );
    }
    assert_eq!(
        analyze("dimension_precision_budget", Some(&input)).outcome,
        "awaiting_cohort"
    );
    input.source_eigenpair = ContentDigest::sha256(b"another source");
    assert!(f::capture(
        "continuous_l1_bound",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        None,
        &[],
        &context()
    )
    .is_err());
}

#[test]
fn ultra_finite_interfaces_execute_every_supplied_problem_and_keep_absence_records() {
    use xc_solver::convergence as c;
    use xc_spectral::ccm::{
        capture::{CcmCapturePlan, FINITE_DIAGNOSTICS},
        convergence_capture::finite_capture as f,
    };
    let (s, m, matrix) = fixture();
    let plan = CcmCapturePlan::ultra(2, 2).unwrap();
    assert!(FINITE_DIAGNOSTICS.iter().all(|id| plan
        .receipt()
        .unwrap()
        .outcomes()
        .contains_key(*id)));
    let missing = capture_and_persist(
        &plan,
        FINITE_DIAGNOSTICS.iter().map(|s| s.to_string()).collect(),
        |id| {
            CapturedDiagnostic::from_cached(
                f::capture(id, &s, Some(&matrix), None, None, None, &[], &context()).unwrap(),
            )
            .map_err(CaptureFailure::failed)
        },
        &context(),
    )
    .unwrap()
    .value;
    assert_eq!(missing.measurements.len(), FINITE_DIAGNOSTICS.len());
    assert_eq!(
        missing.coverage()["constrained_l1_fit"].outcome,
        "awaiting_source"
    );
    assert_eq!(
        missing.coverage()["trial_vector_energy"].outcome,
        "certified_finite_enclosure"
    );
    assert_ne!(
        missing.coverage()["indexed_prolate_comparison"].outcome,
        "certified_finite_enclosure"
    );
    let bounds = |s: &str| c::ExactBounds::point(s);
    let form = |entries: Vec<&str>| c::TrialForm {
        source_id: "explicit-independent-test-matrix".into(),
        basis_id: "test".into(),
        normalization_id: "euclidean".into(),
        dimension: 2,
        entries: entries.into_iter().map(bounds).collect(),
    };
    let root = c::RootProblem {
        source_id: "test-rational-function".into(),
        branch_id: "test".into(),
        requested_index: 1,
        weights: vec![bounds("1"), bounds("1")],
        poles: vec![bounds("-1"), bounds("1")],
        bracket: c::ExactBounds {
            lower: "-1/4".into(),
            upper: "1/4".into(),
        },
        center: "0".into(),
        value_error_upper: "0".into(),
        derivative_error_upper: "0".into(),
        taylor_order: 8,
    };
    let mut input = inputs(&m);
    let mut extra = f::Inputs {
        scope: "synthetic independent supplied premises; no physical operator assertion".into(),
        ..Default::default()
    };
    extra.l1 = Some(xc_solver::weighted_l1::WeightedL1Problem {
        schema_version: 1,
        basis_id: "test".into(),
        target_id: "test-target".into(),
        normalization_id: "raw".into(),
        quadrature_id: "finite-grid".into(),
        basis: vec![vec!["1".into()], vec!["1".into()], vec!["1".into()]],
        target: vec!["0".into(), "1".into(), "9".into()],
        weights: vec!["1".into(); 3],
        constraints: vec![],
        rhs: vec![],
    });
    extra.trials = Some(f::TrialSeries {
        basis_id: "centered_full_V_fourier".into(),
        scope: "synthetic full-coordinate trial with two independent corrections".into(),
        baseline: f::Vector {
            label: "base".into(),
            coefficients: vec![bounds("0"), bounds("1"), bounds("0")],
        },
        corrections: vec![f::Vector {
            label: "correction".into(),
            coefficients: vec![bounds("1"), bounds("0"), bounds("0")],
        }],
        provenance: BTreeMap::from([("domain".into(), "finite source window".into())]),
    });
    extra.convergence = BTreeMap::from([
        ("finite_root_budget".into(), c::Problem::Root(root.clone())),
        (
            "directional_error_bound".into(),
            c::Problem::Directional(c::DirectionalProblem {
                matrix: form(vec!["2", "0", "0", "3"]),
                rhs: vec![bounds("2"), bounds("3")],
                approximate_solution: vec![bounds("0"), bounds("0")],
                functional: vec![bounds("1"), bounds("1")],
                approximate_dual: vec![bounds("1/2"), bounds("1/3")],
                coercivity_lower: "1".into(),
            }),
        ),
        (
            "finite_tail_bound".into(),
            c::Problem::TailBlock(c::TailBlockProblem {
                matrix: form(vec!["2", "1", "1", "4"]),
                retained_dimension: 1,
                shift: bounds("0"),
                gap_lower: "3".into(),
            }),
        ),
        (
            "dimension_precision_budget".into(),
            c::Problem::Budget(c::BudgetProblem {
                branch_id: "test".into(),
                target_absolute_error: "1/100".into(),
                candidates: vec![c::BudgetCandidate {
                    modes: 2,
                    precision_bits: 128,
                    root,
                }],
                truncation: None,
                cutoff: None,
                reference_assisted: false,
            }),
        ),
        (
            "normalization_error_bound".into(),
            c::Problem::Normalization(c::NormalizationProblem {
                source_id: "test".into(),
                raw_norm_error_upper: "1/10".into(),
                reference_norm_upper: "2".into(),
                source_normalizer: bounds("2"),
                reference_normalizer: bounds("3"),
                normalizer_difference_upper: "1".into(),
            }),
        ),
        (
            "continuous_l1_bound".into(),
            c::Problem::Profile(c::ProfileProblem {
                source_id: "test".into(),
                validation_grid_id: "validation".into(),
                training_grid_id: Some("training".into()),
                cells: vec![c::ProfileCell {
                    left: "0".into(),
                    right: "1".into(),
                    left_residual: bounds("1"),
                    right_residual: bounds("1"),
                    weight: bounds("1"),
                    second_derivative_upper: "0".into(),
                    additional_sup_error_upper: "0".into(),
                }],
            }),
        ),
        (
            "spectral_cluster_bound".into(),
            c::Problem::Cluster(c::ClusterProblem {
                matrix: form(vec!["1", "0", "0", "3"]),
                spectral_window: c::ExactBounds {
                    lower: "0".into(),
                    upper: "2".into(),
                },
                columns: vec![vec![bounds("1"), bounds("0")]],
                shifts: vec!["1".into()],
                gram_lower: "1/2".into(),
                previous_columns: None,
            }),
        ),
    ]);
    input.finite_diagnostics = Some(extra);
    input.validate().unwrap();
    let all = capture_and_persist(
        &plan,
        FINITE_DIAGNOSTICS.iter().map(|s| s.to_string()).collect(),
        |id| {
            CapturedDiagnostic::from_cached(
                f::capture(
                    id,
                    &s,
                    Some(&matrix),
                    None,
                    Some(&input),
                    None,
                    &[],
                    &context(),
                )
                .unwrap(),
            )
            .map_err(CaptureFailure::failed)
        },
        &context(),
    )
    .unwrap()
    .value;
    all.validate().unwrap();
    assert_eq!(all.measurements.len(), FINITE_DIAGNOSTICS.len());
    for id in FINITE_DIAGNOSTICS {
        assert_eq!(
            all.measurements[*id].value["data"]["outcome"], "computed",
            "{id}"
        );
    }
    let data = &all.measurements["trial_vector_energy"].value["data"]["result"];
    assert_eq!(
        data["report"]["energy"]["total"],
        json!({"lower":"7","upper":"7"})
    );
    assert_eq!(
        data["report"]["norm"]["total"],
        json!({"lower":"2","upper":"2"})
    );
    assert_eq!(
        data["report"]["stages"][1]["measurement"]["normalized"]["rayleigh_quotient"],
        json!({"lower":"7/2","upper":"7/2"})
    );
    assert_eq!(data["report"]["stages"].as_array().unwrap().len(), 2);
    assert_eq!(
        data["report"]["energy_pairings"].as_array().unwrap().len(),
        4
    );
    assert_eq!(
        all.measurements["constrained_l1_fit"].value["data"]["result"]["witness"]["coefficients"],
        json!(["1"])
    );
    // Numerical incomplete budgets remain qualified despite successful acquisition.
    assert_ne!(
        all.coverage()["dimension_precision_budget"].outcome,
        "certified_finite_enclosure"
    );
    let mut changed = input.clone();
    let mut auto = input.clone();
    auto.finite_diagnostics
        .as_mut()
        .unwrap()
        .convergence
        .remove("normalization_error_bound");
    auto.target=Some(serde_json::from_value(json!({"definition_digest":ContentDigest::sha256(b"finite projection"),"evaluation_policy":"finite test","approximation_scope":"finite coefficients only","intervals":8,"values":vec!["2";9],"basis_values":[],"fixed_second_component":null,"raw_normalizer":"2","trial_coefficients":["0","2","0"]})).unwrap());
    let normalized = f::capture(
        "normalization_error_bound",
        &s,
        None,
        None,
        Some(&auto),
        None,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(normalized.value.data.result["squared_difference"], "0");
    let extra = auto.finite_diagnostics.as_mut().unwrap();
    extra.prolate_references.push(f::ProlateReference {
        full_index: 4,
        source_id: "synthetic-reference".into(),
        log_singular_value_deficit: "-10".into(),
        precision_bits: 128,
        quadrature_order: 32,
        scope: "synthetic numerical reference".into(),
        weil: Some(f::WeilReference {
            source_id: "synthetic-weil-reference".into(),
            sector: "even".into(),
            zero_based_index: 0,
            sector_dimension: 2,
            log_absolute_eigenvalue: "-9".into(),
        }),
    });
    let indexed = f::capture(
        "indexed_prolate_comparison",
        &s,
        None,
        None,
        Some(&auto),
        None,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(indexed.value.data.rows.len(), 4);
    assert_eq!(
        Float::with_val(
            128,
            Float::parse(
                indexed.value.data.rows[3].result["log_abs_weil_over_numerical_deficit"]
                    .as_str()
                    .unwrap()
            )
            .unwrap()
        ),
        1
    );
    changed
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .not_applicable
        .insert("continuous_l1_bound".into(), "finite grid only".into());
    assert!(changed.validate().is_err());
    changed
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .convergence
        .remove("continuous_l1_bound");
    let na = f::capture(
        "continuous_l1_bound",
        &s,
        None,
        None,
        Some(&changed),
        None,
        &[],
        &context(),
    )
    .unwrap();
    assert_eq!(na.value.data.outcome, "not_applicable");
    assert_ne!(
        na.value.request,
        all.measurements["continuous_l1_bound"].value["request"]
    );
    changed.source_eigenpair = ContentDigest::sha256(b"foreign state");
    assert!(f::capture(
        "continuous_l1_bound",
        &s,
        None,
        None,
        Some(&changed),
        None,
        &[],
        &context()
    )
    .is_err());
    let mut public = context();
    public.write_visibility = CacheVisibility::Public;
    assert!(f::capture(
        "trial_vector_energy",
        &s,
        Some(&matrix),
        None,
        None,
        None,
        &[],
        &public
    )
    .is_err());
}

#[test]
fn finite_capture_sign_and_reference_joins_are_explicit() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, m, _) = fixture();
    let mut input = inputs(&m);
    input.finite_diagnostics = Some(f::Inputs {
        scope: "synthetic".into(),
        ..Default::default()
    });
    let r = f::ProlateReference {
        full_index: 4,
        source_id: "synthetic-deficit".into(),
        log_singular_value_deficit: "-10".into(),
        precision_bits: 128,
        quadrature_order: 32,
        scope: "synthetic point".into(),
        weil: Some(f::WeilReference {
            source_id: "synthetic-spectrum".into(),
            sector: "even".into(),
            zero_based_index: 0,
            sector_dimension: 2,
            log_absolute_eigenvalue: "-9".into(),
        }),
    };
    input
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .prolate_references
        .push(r);
    let result = f::capture(
        "indexed_prolate_comparison",
        &s,
        None,
        None,
        Some(&input),
        None,
        &[],
        &context(),
    )
    .unwrap();
    // Independent elementary expression for full mode 4. D=-log(deficit).
    assert_eq!(result.value.data.result["convention"], "D=-log(deficit)");
    assert_eq!(result.value.data.result["logarithm"], "natural");
    assert_eq!(
        result.value.request["indexed_prolate_convention"],
        result.value.data.result
    );
    let log_asym = ((16384.0 / 3.0) * 2f64.sqrt() * std::f64::consts::PI.powi(5)).ln()
        - 36.0 * std::f64::consts::PI
        + 4.5 * 9f64.ln();
    let expected = 10.0 - (-log_asym);
    assert!(expected < 0.0);
    near(
        result.value.data.rows[3].result["D_numerical_minus_D_asymptotic"]
            .as_str()
            .unwrap(),
        expected,
        1e-12,
    );
    for (sector, index, dimension, mode) in [
        ("odd", 0, 2, 4),
        ("full", 0, 3, 4),
        ("even", 999, 2, 4),
        ("even", 0, 0, 4),
        ("even", 0, 2, 6),
    ] {
        let mut bad = input.clone();
        let r = &mut bad.finite_diagnostics.as_mut().unwrap().prolate_references[0];
        r.full_index = mode;
        let w = r.weil.as_mut().unwrap();
        w.sector = sector.into();
        w.zero_based_index = index;
        w.sector_dimension = dimension;
        assert!(bad.validate().is_err());
    }
}

#[test]
fn finite_capture_final_norm_controls_assurance() {
    use xc_solver::trial_energy::ExactBounds;
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, m, matrix) = fixture();
    for bounds in [
        ExactBounds::point("0"),
        ExactBounds {
            lower: "-1".into(),
            upper: "1".into(),
        },
    ] {
        let mut input = inputs(&m);
        input.finite_diagnostics = Some(f::Inputs {
            scope: "synthetic".into(),
            trials: Some(f::TrialSeries {
                basis_id: "centered_full_V_fourier".into(),
                scope: "finite".into(),
                baseline: f::Vector {
                    label: "zero-or-uncertain".into(),
                    coefficients: vec![bounds; 3],
                },
                corrections: vec![],
                provenance: BTreeMap::new(),
            }),
            ..Default::default()
        });
        for id in ["trial_vector_energy", "trial_vector_parity"] {
            let r = f::capture(
                id,
                &s,
                Some(&matrix),
                None,
                Some(&input),
                None,
                &[],
                &context(),
            )
            .unwrap();
            assert_eq!(r.value.data.rows[0].outcome, "unresolved");
            let stages = r.value.data.result["report"]["stages"].as_array().unwrap();
            assert_eq!(
                stages.last().unwrap()["measurement"]["status"],
                "norm_not_separated_from_zero"
            );
            assert_ne!(
                NumericalCoverage::from_value(&serde_json::to_value(&r.value).unwrap()).outcome,
                "certified_finite_enclosure"
            );
        }
    }
}

#[test]
fn finite_capture_center_normalization_removes_scale_and_sign_artifacts() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, m, matrix) = fixture();
    for target in [vec!["0", "-2", "0"], vec!["1", "3", "0"]] {
        let mut input = inputs(&m);
        input.target=Some(serde_json::from_value(json!({"definition_digest":ContentDigest::sha256(b"test target"),"evaluation_policy":"finite test","approximation_scope":"finite coefficients only","intervals":8,"values":vec!["2";9],"basis_values":[],"fixed_second_component":null,"raw_normalizer":"2","trial_coefficients":target})).unwrap());
        let r = f::capture(
            "normalization_error_bound",
            &s,
            None,
            None,
            Some(&input),
            None,
            &[],
            &context(),
        )
        .unwrap();
        let e = f::capture(
            "trial_vector_energy",
            &s,
            Some(&matrix),
            None,
            Some(&input),
            None,
            &[],
            &context(),
        )
        .unwrap();
        if target[1] == "-2" {
            assert_eq!(r.value.data.result["squared_difference"], "0");
            assert_eq!(
                r.value.data.result["normalized_norm_error"],
                json!({"lower":"0","upper":"0"})
            );
            assert!(
                e.value.data.result["trial_series"]["corrections"][0]["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|b| b["lower"] == "0" && b["upper"] == "0")
            );
        } else {
            // x=(0,1,0), y=(1,3,0); centers 1 and 2. Squared distance=1/2.
            assert_eq!(r.value.data.result["squared_difference"], "1/2");
            let norm = &r.value.data.result["normalized_norm_error"];
            let lo = rug::Rational::from_str_radix(norm["lower"].as_str().unwrap(), 10).unwrap();
            let hi = rug::Rational::from_str_radix(norm["upper"].as_str().unwrap(), 10).unwrap();
            assert!(rug::Rational::from(&lo * &lo) <= rug::Rational::from((1, 2)));
            assert!(rug::Rational::from(&hi * &hi) >= rug::Rational::from((1, 2)));
            assert!(hi - lo < rug::Rational::from((1, rug::Integer::from(1) << 120)));
        }
        assert_eq!(
            e.value.data.result["trial_series"]["provenance"]["source_center_normalizer"],
            "1"
        );
    }
}

#[test]
fn finite_capture_resource_limits_do_not_poison_warm_replay() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, _, matrix) = fixture();
    let dir = xc_core::test_support::TestDir::new("finite-resource-replay");
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(ZipJsonFilesystemCacheStore::new(
            "local",
            dir.to_path_buf(),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let cache = ArtifactCacheContext {
        resolver: Some(&resolver),
        acceptance: Some(&policy),
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        ordered_overlays: vec!["local".into()],
        ..context()
    };
    let run = |execution: &f::ExecutionContext| {
        f::capture_with_context(
            "trial_vector_energy",
            &s,
            Some(&matrix),
            None,
            None,
            None,
            &[],
            &cache,
            execution,
        )
        .unwrap()
    };
    let mut limit = f::ExecutionContext::default();
    limit.policy.maximum_working_bytes = 1000;
    let blocked = run(&limit);
    assert_eq!(blocked.value.data.outcome, "blocked");
    assert!(blocked.produced_manifest.is_none() && blocked.reused_manifest.is_none());
    limit.policy.maximum_working_bytes = 96 << 30;
    limit.policy.maximum_output_bytes = 1;
    let output = run(&limit);
    assert_eq!(output.value.data.outcome, "blocked");
    assert!(output.produced_manifest.is_none() && output.reused_manifest.is_none());
    limit.policy.maximum_output_bytes = 8 << 30;
    let computed = run(&limit);
    assert!(computed.produced_manifest.is_some());
    limit.policy.maximum_working_bytes = 8 << 30;
    let replay = run(&limit);
    assert!(replay.reused_manifest.is_some());
    assert_eq!(
        serde_json::to_value(&computed.value).unwrap(),
        serde_json::to_value(&replay.value).unwrap()
    );
    assert_eq!(blocked.value.request, computed.value.request);
    // Even a small execution cap can reuse already-completed content.
    limit.policy.maximum_working_bytes = 1000;
    assert!(run(&limit).reused_manifest.is_some());
    let mut profiles = profile_request();
    let cold = f::capture_target_profiles(&profiles, &cache).unwrap();
    let warm = f::capture_target_profiles(&profiles, &cache).unwrap();
    assert!(cold.produced_manifest.is_some() && warm.reused_manifest.is_some());
    assert_eq!(cold.value.data, warm.value.data);
    profiles.profiles[0].real[2] = xc_solver::trial_energy::ExactBounds::point("2");
    let changed = f::capture_target_profiles(&profiles, &cache).unwrap();
    assert_ne!(
        cold.produced_manifest.unwrap().key,
        changed.produced_manifest.unwrap().key
    );
}

#[test]
fn finite_capture_refuted_premise_is_evidence_but_bad_shape_is_error() {
    use xc_solver::convergence as c;
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, m, _) = fixture();
    let mut input = inputs(&m);
    let b = c::ExactBounds::point;
    let mut p = c::DirectionalProblem {
        matrix: c::TrialForm {
            source_id: "finite".into(),
            basis_id: "test".into(),
            normalization_id: "unit".into(),
            dimension: 1,
            entries: vec![b("2")],
        },
        rhs: vec![b("1")],
        approximate_solution: vec![b("0")],
        functional: vec![b("1")],
        approximate_dual: vec![b("0")],
        coercivity_lower: "5".into(),
    };
    input.finite_diagnostics = Some(f::Inputs {
        scope: "synthetic".into(),
        convergence: BTreeMap::from([(
            "directional_error_bound".into(),
            c::Problem::Directional(p.clone()),
        )]),
        ..Default::default()
    });
    let run = |i: &ExternalResearchInputs| {
        f::capture(
            "directional_error_bound",
            &s,
            None,
            None,
            Some(i),
            None,
            &[],
            &context(),
        )
    };
    let r = run(&input).unwrap();
    assert_eq!(r.value.data.outcome, "computed");
    assert_eq!(r.value.data.rows[0].outcome, "premise_not_verified");
    assert!(r
        .value
        .data
        .reason
        .unwrap()
        .contains("finite form is not positive definite"));
    p.rhs.clear();
    input
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .convergence
        .insert("directional_error_bound".into(), c::Problem::Directional(p));
    assert!(run(&input).is_err());
}

fn complex_vector(
    label: &str,
    real: &[&str],
    imaginary: &[&str],
) -> xc_spectral::ccm::convergence_capture::finite_capture::ComplexVector {
    use xc_solver::trial_energy::ExactBounds as B;
    xc_spectral::ccm::convergence_capture::finite_capture::ComplexVector {
        label: label.into(),
        real: real.iter().map(|v| B::point(*v)).collect(),
        imaginary: imaginary.iter().map(|v| B::point(*v)).collect(),
    }
}
fn profile_request() -> xc_spectral::ccm::convergence_capture::finite_capture::ProfileRequest {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let target = complex_vector("reference", &["0", "1", "0"], &["0", "0", "0"]);
    f::ProfileRequest {
        basis_id: "centered_full_V_fourier".into(),
        lambda_squared: "4".into(),
        scope: "synthetic finite functions".into(),
        definition_digest: ContentDigest::sha256(b"finite synthetic profile"),
        profiles: vec![
            complex_vector("candidate", &["0", "1", "1"], &["0", "0", "0"]),
            target.clone(),
        ],
        functional: f::BoundedFunctional {
            id: "synthetic inner product".into(),
            reference: target,
            denominator: f::FunctionalDenominator::ReferenceSquaredNorm,
        },
        intervals_per_half: 8,
        precision_bits: 128,
    }
}
#[test]
fn functional_normalization_is_not_a_least_squares_fit_and_survives_zero_center() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let request = profile_request();
    let result = f::analyze_profiles(&request).unwrap();
    assert_eq!(result["normalizers"][0]["functional"]["real"]["lower"], "1");
    assert_eq!(result["pairs"][0]["status"], "unresolved"); // center 1-1=0
    let pair = &result["pairs"][1];
    let upper = rug::Rational::from_str_radix(
        result["functional_weighted_l1_dual_bounds"]["full_window_upper"]
            .as_str()
            .unwrap(),
        10,
    )
    .unwrap()
    .to_f64();
    assert!((upper - 2.0f64.sqrt() / 4.0f64.ln()).abs() < 1e-14);
    assert_eq!(
        result["normalizers"][0]["reciprocal_even_point_coefficients"],
        false
    );
    assert_eq!(pair["normalization"], "bounded_functional");
    assert_eq!(pair["coefficient_l2_difference"]["lower"], "1");
    assert_eq!(pair["coefficient_l2_difference"]["upper"], "1");
    // Least-squares rescaling would produce squared error 1/2, not 1.
    assert_eq!(pair["halves"].as_array().unwrap().len(), 2);
    assert_eq!(
        pair["halves"][0]["signed_residual_samples"]
            .as_array()
            .unwrap()
            .len(),
        9
    );
    // The residual is a unit-modulus Fourier mode. Its integral on [1,2]
    // is exactly 2(sqrt(2)-1); directed cell ranges must enclose it.
    let bound = &pair["halves"][1]["continuous_weighted_l1_enclosure"];
    let r = |s: &serde_json::Value| {
        rug::Rational::from_str_radix(s.as_str().unwrap(), 10)
            .unwrap()
            .to_f64()
    };
    let exact = 2.0 * (2.0f64.sqrt() - 1.0);
    assert!(r(&bound["lower"]) <= exact && r(&bound["upper"]) >= exact);
    let mut rotated = request.clone();
    rotated.profiles[0] = complex_vector("candidate", &["0", "0", "0"], &["0", "3", "3"]);
    let rotated = f::analyze_profiles(&rotated).unwrap();
    assert_eq!(
        rotated["pairs"][1]["coefficient_l2_difference"],
        pair["coefficient_l2_difference"]
    );
    assert_eq!(rotated["pairs"][1]["halves"], pair["halves"]);
}
#[test]
fn profile_analysis_is_bit_identical_across_thread_counts() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let mut request = profile_request();
    request.intervals_per_half = 64;
    // Several pairs per normalization, evaluated concurrently, keep their order.
    request.profiles.push(complex_vector(
        "third",
        &["1/3", "2", "-1"],
        &["0", "1/5", "0"],
    ));
    request.profiles.push(complex_vector(
        "fourth",
        &["-1", "1", "1/7"],
        &["1/2", "0", "-1/9"],
    ));
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| f::analyze_profiles(&request).unwrap())
    };
    let serial = run(1);
    for _ in 0..10 {
        assert_eq!(run(4), serial);
    }
}
#[test]
fn target_only_capture_replays_without_an_eigenstate_and_blocks_publication() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let mut request = profile_request();
    let result = f::capture_target_profiles(&request, &context()).unwrap();
    assert!(result.value.source_dependencies.is_empty());
    assert_eq!(result.value.data, f::analyze_profiles(&request).unwrap());
    let mut public = context();
    public.write_visibility = CacheVisibility::Public;
    assert!(f::capture_target_profiles(&request, &public).is_err());
    request.functional.denominator = f::FunctionalDenominator::Declared {
        value: xc_solver::trial_energy::ExactBounds {
            lower: "0".into(),
            upper: "1".into(),
        },
        provenance: "synthetic exterior allowance".into(),
        scope: "declared full-domain denominator".into(),
    };
    let unresolved = f::analyze_profiles(&request).unwrap();
    assert_eq!(unresolved["pairs"][1]["status"], "unresolved");
    assert!(unresolved["functional_operator_norm_bound"].is_null());
}
#[test]
fn complex_capture_preserves_hermitian_energy_cancellation_and_parity() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, m, matrix) = fixture();
    let mut input = inputs(&m);
    // diag(4,3,4): Q(x+iy)=3+8=11; Q(x+y) would add
    // unrelated bilinear terms on a general matrix.
    input.finite_diagnostics = Some(f::Inputs {
        scope: "synthetic complex trial".into(),
        complex_trials: Some(f::ComplexTrialSeries {
            basis_id: "centered_full_V_fourier".into(),
            matrix_digest: m.dependencies[0].content_digest.clone(),
            scope: "finite coefficients".into(),
            baseline: complex_vector("baseline", &["0", "1", "0"], &["-1", "0", "1"]),
            corrections: vec![],
            functional: Some(profile_request().functional),
            provenance: BTreeMap::new(),
        }),
        ..Default::default()
    });
    let capture = |id: &str, i: &ExternalResearchInputs| {
        f::capture(id, &s, Some(&matrix), None, Some(i), None, &[], &context())
    };
    let energy = capture("trial_vector_energy", &input).unwrap().value.data;
    let report = &energy.result["report"];
    assert_eq!(report["stages"][0]["measurement"]["energy"]["lower"], "11");
    assert_eq!(
        report["stages"][0]["measurement"]["squared_norm"]["lower"],
        "3"
    );
    assert_eq!(
        energy.result["functional_energy"]["rows"][0]["energy_over_functional_squared"]["lower"],
        "11"
    );
    let parity = capture("trial_vector_parity", &input)
        .unwrap()
        .value
        .data
        .result;
    assert_eq!(parity["report"]["parts"][0]["energy"]["lower"], "3");
    assert_eq!(parity["report"]["parts"][1]["energy"]["lower"], "8");
    assert_eq!(parity["report"]["energy_pairings"][1]["lower"], "0");
    assert_eq!(
        parity["report"]["stages"][1]["measurement"]["energy"]["lower"],
        "11"
    );
    input
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .complex_trials
        .as_mut()
        .unwrap()
        .corrections
        .push(complex_vector(
            "correction",
            &["0", "1", "0"],
            &["1", "0", "-1"],
        ));
    let corrected = capture("trial_vector_energy", &input)
        .unwrap()
        .value
        .data
        .result;
    assert_eq!(corrected["report"]["energy_pairings"][1]["lower"], "-5");
    assert_eq!(
        corrected["report"]["stages"][1]["measurement"]["energy"]["lower"],
        "12"
    );
    let trial = input
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .complex_trials
        .as_mut()
        .unwrap();
    trial.baseline = complex_vector(
        "overlapping_real_imaginary",
        &["0", "1", "0"],
        &["0", "1", "0"],
    );
    trial.corrections.clear();
    let overlap = capture("trial_vector_energy", &input)
        .unwrap()
        .value
        .data
        .result;
    assert_eq!(
        overlap["report"]["stages"][0]["measurement"]["energy"]["lower"],
        "6"
    ); // Q(x+y) would incorrectly be 12.
    input
        .finite_diagnostics
        .as_mut()
        .unwrap()
        .complex_trials
        .as_mut()
        .unwrap()
        .matrix_digest = ContentDigest::sha256(b"wrong matrix");
    assert!(capture("trial_vector_energy", &input).is_err());
}
#[test]
fn even_projection_can_reduce_energy_and_increase_rayleigh_quotient() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,
        "entries":["1.5","0","-0.5","0","10","0","-0.5","0","1.5"]}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":["1","0","1"]}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let mut input = inputs(&sm);
    input.finite_diagnostics = Some(f::Inputs {
        scope: "ordered finite spectrum 1 < 2 < 10".into(),
        complex_trials: Some(f::ComplexTrialSeries {
            basis_id: "centered_full_V_fourier".into(),
            matrix_digest: mm.content_digest,
            scope: "finite complex vector".into(),
            baseline: complex_vector("trial", &["0", "1", "0"], &["-1", "0", "1"]),
            corrections: vec![],
            functional: None,
            provenance: BTreeMap::new(),
        }),
        ..Default::default()
    });
    let r = f::capture(
        "trial_vector_parity",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        None,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
    .result;
    assert_eq!(
        r["report"]["stages"][0]["measurement"]["energy"]["lower"],
        "10"
    );
    assert_eq!(
        r["report"]["stages"][1]["measurement"]["energy"]["lower"],
        "14"
    );
    assert_eq!(
        r["report"]["stages"][0]["measurement"]["normalized"]["rayleigh_quotient"]["lower"],
        "10"
    );
    assert_eq!(
        r["report"]["stages"][1]["measurement"]["normalized"]["rayleigh_quotient"]["lower"],
        "14/3"
    );
}

#[test]
fn tiny_odd_norm_can_carry_material_energy() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let diagonal = format!("{}.5", 1u128 << 119);
    let off_diagonal = format!("-{}.5", (1u128 << 119) - 1);
    let (mm, mb) = source(
        "ccm_tau_matrix",
        json!({"schema_version":2,"lambda_squared":"9","n_modes":1,"precision_bits":128,
        "entries":[diagonal,"0",off_diagonal,"0","1","0",off_diagonal,"0",diagonal]}),
    );
    let (mut sm, sb) = source(
        "ccm_weil_eigenpair",
        json!({"schema_version":3,"lambda_squared":"9","n_modes":1,"precision_bits":128,"force_even":true,"eigenvalue":"1","eigenvector":["0","1","0"]}),
    );
    sm.dependencies.push(DependencyRef {
        key: mm.key.clone(),
        content_digest: mm.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    });
    let s =
        RetainedState::from_payload(&sm, &sb, std::slice::from_ref(&sm.content_digest)).unwrap();
    let matrix =
        RetainedMatrix::from_payload(&mm, &mb, std::slice::from_ref(&mm.content_digest)).unwrap();
    let mut input = inputs(&sm);
    input.finite_diagnostics = Some(f::Inputs {
        scope: "synthetic tiny odd component".into(),
        complex_trials: Some(f::ComplexTrialSeries {
            basis_id: "centered_full_V_fourier".into(),
            matrix_digest: mm.content_digest,
            scope: "finite complex vector".into(),
            baseline: complex_vector(
                "trial",
                &["0", "1", "0"],
                &["-1/1152921504606846976", "0", "1/1152921504606846976"],
            ),
            corrections: vec![],
            functional: None,
            provenance: BTreeMap::new(),
        }),
        ..Default::default()
    });
    let r = f::capture(
        "trial_vector_parity",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        None,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data
    .result;
    assert_eq!(r["report"]["parts"][1]["energy"]["lower"], "2");
    assert_eq!(
        r["report"]["parts"][1]["squared_norm"]["lower"],
        format!("1/{}", 1u128 << 119)
    );
    assert_eq!(
        r["parity_summary"]["odd_signed_energy_fraction"]["lower"],
        "2/3"
    );
}

#[test]
fn zero_center_withholds_the_correction_but_keeps_raw_and_functional_energy() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let (s, m, matrix) = fixture();
    let mut input = inputs(&m);
    input.target = Some(SampledReference {
        definition_digest: ContentDigest::sha256(b"zero-center synthetic reference"),
        evaluation_policy: "stored finite projection".into(),
        approximation_scope: "synthetic".into(),
        intervals: 8,
        values: vec!["0".into(); 9],
        basis_values: vec![],
        fixed_second_component: None,
        raw_normalizer: "1".into(),
        trial_coefficients: Some(vec!["0".into(), "1".into(), "1".into()]),
    });
    let r = f::capture(
        "trial_vector_energy",
        &s,
        Some(&matrix),
        None,
        Some(&input),
        None,
        &[],
        &context(),
    )
    .unwrap()
    .value
    .data;
    assert_eq!(r.rows[0].outcome, "certified_finite_enclosure");
    assert_eq!(r.result["report"]["stages"].as_array().unwrap().len(), 1);
    assert_eq!(
        r.result["report"]["stages"][0]["measurement"]["energy"]["lower"],
        "3"
    );
    assert!(
        r.result["trial_series"]["provenance"]["automatic_target_correction_withheld"].is_string()
    );
    assert_eq!(
        r.result["functional_energy"]["rows"][0]["energy_over_functional_squared"]["lower"],
        "12"
    );
}
fn refinement_cohort() -> xc_spectral::ccm::convergence_capture::finite_capture::RefinementCohort {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    f::RefinementCohort {
        axis: f::RefinementAxis::Dimension,
        scope: "synthetic increasing dimension".into(),
        relative_tolerance: "0.01".into(),
        observations: ["100", "100.5", "101"]
            .iter()
            .enumerate()
            .map(|(j, value)| f::RefinementObservation {
                source: ContentDigest::sha256(format!("source{j}").as_bytes()),
                matrix: ContentDigest::sha256(format!("matrix{j}").as_bytes()),
                configuration: f::RefinementConfiguration {
                    observable: f::RefinementObservable::Eigenvalue,
                    domain: f::ComparisonDomain::CoefficientSpace,
                    lambda_squared: "4".into(),
                    branch: f::SpectralBranch::EvenGround,
                    external_index: 1,
                    index_origin: f::IndexOrigin::One,
                    n_modes: j + 1,
                    precision_bits: 128,
                    operator_quadrature_orders: vec![64],
                    projection_quadrature_order: 64,
                    representation_order: 32,
                    guard_bits: 32,
                    certificate_precision_bits: 256,
                    basis_id: "centered_full_V_fourier".into(),
                    metric_id: "identity-coefficient-metric-v1".into(),
                    operator_recipe: ContentDigest::sha256(b"fixed recipe"),
                    target_definition: None,
                    normalizer_id: "unit coefficient norm".into(),
                    trial_recipe: None,
                },
                validity: f::ObservationValidity::Accepted,
                value: (*value).into(),
                enclosure: Some(xc_solver::trial_energy::ExactBounds::point(*value)),
            })
            .collect(),
    }
}
#[test]
fn refinement_gates_use_unrounded_adjacent_distinct_steps_and_typed_joins() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let original = refinement_cohort();
    let result = f::analyze_refinement_cohort(&original).unwrap();
    assert_eq!(
        f::capture_refinement_cohort(&original, &context())
            .unwrap()
            .value
            .data,
        result
    );
    assert_eq!(result["operationally_stabilized"], true);
    assert_eq!(result["enclosed_changes_below_tolerance"], true);
    assert_eq!(result["distinct_dimension_precision_configurations"], 3);
    let mut c = original.clone();
    c.observations.push(c.observations[2].clone());
    assert_eq!(
        f::analyze_refinement_cohort(&c).unwrap()["operationally_stabilized"],
        false
    );
    c = original.clone();
    c.observations[1].validity = f::ObservationValidity::Unresolved;
    assert_eq!(
        f::analyze_refinement_cohort(&c).unwrap()["operationally_stabilized"],
        false
    );
    c = original.clone();
    for (o, value) in c.observations.iter_mut().zip(["98", "99", "100"]) {
        o.value = value.into();
        o.enclosure = Some(xc_solver::trial_energy::ExactBounds::point(value));
    }
    let equality = f::analyze_refinement_cohort(&c).unwrap();
    assert_eq!(equality["rows"][2]["relative_change"], "1/100");
    assert_eq!(equality["operationally_stabilized"], false);
    c = original.clone();
    c.observations[2].configuration.branch = f::SpectralBranch::OddGround;
    assert!(f::analyze_refinement_cohort(&c).is_err());
    c = original.clone();
    c.observations[2].configuration.projection_quadrature_order = 128;
    assert!(f::analyze_refinement_cohort(&c).is_err());
    c = original.clone();
    c.observations[2].configuration.observable = f::RefinementObservable::SquaredNorm;
    assert!(f::analyze_refinement_cohort(&c).is_err());
    c = original.clone();
    c.observations[2].enclosure = Some(xc_solver::trial_energy::ExactBounds {
        lower: "-101".into(),
        upper: "101".into(),
    });
    let interval = f::analyze_refinement_cohort(&c).unwrap();
    assert_eq!(interval["operationally_stabilized"], true);
    assert_eq!(interval["enclosed_changes_below_tolerance"], false);
    assert!(interval["rows"][2]["relative_change_enclosure"].is_null());
    c = original.clone();
    c.axis = f::RefinementAxis::OperatorQuadrature;
    for (j, o) in c.observations.iter_mut().enumerate() {
        o.configuration.n_modes = 1;
        o.configuration.operator_quadrature_orders = vec![64 * (j + 1)];
    }
    assert_eq!(
        f::analyze_refinement_cohort(&c).unwrap()["distinct_dimension_precision_configurations"],
        1
    );
}
#[test]
fn energy_distance_uses_absolute_complement_floor_and_refuses_inconsistency() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let mut p = f::EnergyDistancePremises {
        matrix_digest: ContentDigest::sha256(b"finite matrix"),
        basis_id: "centered_full_V_fourier".into(),
        metric_id: "identity-coefficient-metric-v1".into(),
        ground_enclosure: xc_solver::trial_energy::ExactBounds {
            lower: "1".into(),
            upper: "2".into(),
        },
        complementary_eigenvalue_lower: "5".into(),
        full_space_simple_ground: true,
        provenance: "synthetic premises".into(),
    };
    assert_eq!(
        f::energy_distance_bound(&p, "3", 128).unwrap()["squared_sine_upper"],
        "1/2"
    );
    assert_eq!(
        f::energy_distance_bound(&p, "0", 128).unwrap()["status"],
        "premise_not_verified"
    );
    p.full_space_simple_ground = false;
    assert_eq!(
        f::energy_distance_bound(&p, "3", 128).unwrap()["status"],
        "unresolved"
    );
}
#[test]
fn matched_spectral_triples_reject_unmatched_matrices_and_index_conventions() {
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    let seed = refinement_cohort().observations[0].clone();
    let mut triple = [seed.clone(), seed.clone(), seed];
    for (j, (o, branch)) in triple
        .iter_mut()
        .zip([
            f::SpectralBranch::EvenGround,
            f::SpectralBranch::OddGround,
            f::SpectralBranch::EvenFirstExcited,
        ])
        .enumerate()
    {
        o.configuration.branch = branch;
        o.configuration.external_index = if j == 2 { 2 } else { 1 };
        o.value = (j + 1).to_string();
        o.enclosure = Some(xc_solver::trial_energy::ExactBounds::point(&o.value));
        o.source = ContentDigest::sha256(format!("branch{j}").as_bytes());
    }
    let r = f::analyze_matched_spectral_triple(&triple).unwrap();
    assert_eq!(r["enclosed_even0_lt_odd0_lt_even1"], true);
    triple[2].configuration.index_origin = f::IndexOrigin::Zero;
    assert!(f::analyze_matched_spectral_triple(&triple).is_err());
    triple[2].configuration.external_index = 1;
    assert!(f::analyze_matched_spectral_triple(&triple).is_ok());
    triple[2].matrix = ContentDigest::sha256(b"other dimension");
    assert!(f::analyze_matched_spectral_triple(&triple).is_err());
}

#[test]
fn high_precision_complex_profile_normalization_encloses_exact_midpoints() {
    use rug::{Integer, Rational as R};
    use xc_solver::trial_energy::ExactBounds as B;
    use xc_spectral::ccm::convergence_capture::finite_capture as f;
    // Exact accumulation before dyadic enclosure exceeds 70,000 denominator bits.
    let tiny = R::from((Integer::from(1), Integer::from(1) << 7000));
    let width: R = tiny.clone() / 16i32;
    let raw = [tiny.clone(), R::from(1), tiny.clone() * 2i32];
    let imaginary = [tiny.clone(), R::from(0), -tiny.clone()];
    let center: R = R::from(1) - tiny.clone() * 3i32;
    let reference: Vec<R> = raw.iter().map(|x| x.clone() / &center).collect();
    let interval = |x: &R| B {
        lower: (x.clone() - &width).to_string(),
        upper: (x.clone() + &width).to_string(),
    };
    let mut request = profile_request();
    request.precision_bits = 7014;
    request.profiles = vec![
        f::ComplexVector {
            label: "interval candidate".into(),
            real: raw.iter().map(interval).collect(),
            imaginary: imaginary.iter().map(interval).collect(),
        },
        f::ComplexVector {
            label: "point target".into(),
            real: raw.iter().map(|x| B::point(x.to_string())).collect(),
            imaginary: vec![B::point("0"); 3],
        },
    ];
    request.functional.reference = f::ComplexVector {
        label: "center-one reference".into(),
        real: reference.iter().map(|x| B::point(x.to_string())).collect(),
        imaginary: vec![B::point("0"); 3],
    };
    let result = f::analyze_profiles(&request).unwrap();
    let norm2 = |v: &[(R, R)]| {
        v.iter().fold(R::from(0), |sum, (r, i)| {
            sum + r.clone() * r + i.clone() * i
        })
    };
    let x: Vec<_> = raw.iter().cloned().zip(imaginary).collect();
    let y: Vec<_> = raw.iter().cloned().map(|r| (r, R::from(0))).collect();
    let denominator = reference
        .iter()
        .fold(R::from(0), |sum, r| sum + r.clone() * r);
    let divide = |z: &(R, R), a: &(R, R)| {
        let d = a.0.clone() * &a.0 + a.1.clone() * &a.1;
        (
            (z.0.clone() * &a.0 + z.1.clone() * &a.1) / &d,
            (z.1.clone() * &a.0 - z.0.clone() * &a.1) / &d,
        )
    };
    for pair in result["pairs"].as_array().unwrap() {
        assert_eq!(pair["status"], "finite_enclosure");
        let normalizer = |v: &[(R, R)]| {
            v.iter()
                .enumerate()
                .fold((R::from(0), R::from(0)), |s, (j, z)| {
                    let weight = if pair["normalization"] == "center" {
                        R::from(if j == 1 { 1 } else { -1 })
                    } else {
                        reference[j].clone() / &denominator
                    };
                    (s.0 + z.0.clone() * &weight, s.1 + z.1.clone() * weight)
                })
        };
        let (a, b) = (normalizer(&x), normalizer(&y));
        let delta: Vec<_> = x
            .iter()
            .zip(&y)
            .map(|(x, y)| {
                let (x, y) = (divide(x, &a), divide(y, &b));
                (x.0 - y.0, x.1 - y.1)
            })
            .collect();
        let exact = norm2(&delta);
        let bound = &pair["coefficient_l2_difference"];
        let lower = R::from_str_radix(bound["lower"].as_str().unwrap(), 10).unwrap();
        let upper = R::from_str_radix(bound["upper"].as_str().unwrap(), 10).unwrap();
        assert!(lower >= 0);
        assert!(lower.clone() * lower <= exact);
        assert!(upper.clone() * upper >= exact);
    }
}
