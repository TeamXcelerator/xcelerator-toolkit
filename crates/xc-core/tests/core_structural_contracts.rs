use std::collections::BTreeSet;
use xc_core::*;

fn route(id: &str, family: &str, formulation: &str, implementation: &str) -> RouteEvidence {
    RouteEvidence {
        route_id: id.into(),
        algorithm_family: family.into(),
        formulation: formulation.into(),
        implementation_id: implementation.into(),
        decisive_intermediates: BTreeSet::new(),
        precision_bits: Some(256),
        seed: None,
        thread_count: Some(1),
        evidence_digest: None,
    }
}

#[test]
fn self_comparison_and_spelling_variants_cannot_supply_independence() {
    let declaration = IndependenceDeclaration {
        intended_claim: "ordered finite spectrum".into(),
        rationale: "different mathematical formulations and implementations".into(),
        accepted_shared_inputs: BTreeSet::new(),
    };
    let a = route(
        "qr",
        "householder_qr",
        "matrix_eigensolve",
        "implementation_a",
    );
    let b = route(
        "sturm",
        "sign_recurrence",
        "threshold_count",
        "implementation_b",
    );
    let assessment = assess_route_independence(&a, &b, &declaration);
    assert!(assessment.independent);
    let mut evidence = AssuranceEvidence {
        computation_valid: true,
        independence: Some(assessment),
        comparison: Some(RouteComparisonEvidence {
            intended_claim: declaration.intended_claim.clone(),
            left_result_digest: "a".repeat(64),
            right_result_digest: "b".repeat(64),
            comparison_rule: "equal exact integer inertia counts at all isolating endpoints".into(),
            agreement_accepted: true,
        }),
        ..Default::default()
    };
    assert_eq!(
        evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
        Some(AssuranceLevel::CrossChecked)
    );
    for (left, right) in [
        ("a".repeat(64), "a".repeat(64)),
        ("A".repeat(64), "b".repeat(64)),
        ("a".repeat(64), "B".repeat(64)),
        ("x".repeat(64), "b".repeat(64)),
    ] {
        let comparison = evidence.comparison.as_mut().unwrap();
        comparison.left_result_digest = left;
        comparison.right_result_digest = right;
        assert_eq!(
            evaluate_assurance(AssuranceLevel::CrossChecked, true, &evidence).achieved,
            Some(AssuranceLevel::Computed)
        );
    }
    for altered in [
        route(
            "different",
            " Householder_QR ",
            "matrix_eigensolve\t",
            "other",
        ),
        route("different", "other", "other", " IMPLEMENTATION_A\n"),
        route(" QR ", "other", "other", "other"),
    ] {
        assert!(!assess_route_independence(&a, &altered, &declaration).independent);
    }
    let mut c = a.clone();
    let mut d = b.clone();
    c.decisive_intermediates.insert(" Factorization_1 ".into());
    d.decisive_intermediates.insert("factorization_1".into());
    assert!(!assess_route_independence(&c, &d, &declaration).independent);
}

fn base() -> SolverConfig {
    SolverConfig {
        target: EigenTarget::AlgebraicSmallest,
        subspace: Subspace::Full,
        assurance: AssuranceLevel::Computed,
        precision: PrecisionPolicy::default(),
        stopping: StoppingPolicy::default(),
        reproducibility: Reproducibility::Deterministic,
        algorithm_preferences: vec!["reference".into()],
        allow_lower_precision_seed: false,
        allow_randomized_seed: false,
    }
}
fn layer(source: ConfigSource, value: serde_json::Value) -> ConfigurationLayer {
    ConfigurationLayer {
        source,
        name: format!("{source:?}"),
        value,
        override_allowlist: None,
    }
}

#[test]
fn enum_replacements_match_direct_typed_requests_and_remove_stale_attribution() {
    let d = |x: &str| DecimalLiteral::new(x).unwrap();
    let variants = vec![
        EigenTarget::AlgebraicSmallest,
        EigenTarget::AlgebraicLargest,
        EigenTarget::SmallestMagnitude,
        EigenTarget::ClosestTo { shift: d("0.125") },
        EigenTarget::IndexRange { first: 1, last: 3 },
        EigenTarget::Interval {
            lower: d("1e-30"),
            upper: d("2e-30"),
        },
    ];
    for before in &variants {
        for after in &variants {
            let expected = SolverConfig {
                target: after.clone(),
                ..base()
            };
            let resolved: EffectiveConfiguration<SolverConfig> = resolve_configuration([
                ConfigurationLayer::from_serializable(ConfigSource::BuiltIn, "defaults", &base())
                    .unwrap(),
                layer(ConfigSource::Project, serde_json::json!({"target":before})),
                layer(
                    ConfigSource::CommandLine,
                    serde_json::json!({"target":after}),
                ),
            ])
            .unwrap();
            assert_eq!(resolved.resolved, expected);
            for item in &resolved.overrides {
                assert!(item.path.starts_with("target."));
                assert_eq!(item.source, ConfigSource::CommandLine);
                let key = item.path.strip_prefix("target.").unwrap();
                assert_eq!(
                    item.effective_value,
                    serde_json::to_value(after).unwrap()[key]
                );
            }
            let direct: EffectiveConfiguration<SolverConfig> =
                resolve_configuration([ConfigurationLayer::from_serializable(
                    ConfigSource::BuiltIn,
                    "direct",
                    &expected,
                )
                .unwrap()])
                .unwrap();
            assert_eq!(resolved.digest, direct.digest);
        }
    }
    let escalation = [
        PrecisionEscalation::Fixed,
        PrecisionEscalation::AddBits(16),
        PrecisionEscalation::Multiply {
            numerator: 3,
            denominator: 2,
        },
    ];
    for before in escalation {
        for after in escalation {
            let result: EffectiveConfiguration<SolverConfig> = resolve_configuration([
                ConfigurationLayer::from_serializable(ConfigSource::BuiltIn, "defaults", &base())
                    .unwrap(),
                layer(
                    ConfigSource::Project,
                    serde_json::json!({"precision":{"escalation":before}}),
                ),
                layer(
                    ConfigSource::CommandLine,
                    serde_json::json!({"precision":{"escalation":after}}),
                ),
            ])
            .unwrap();
            assert_eq!(result.resolved.precision.escalation, after);
            if after == PrecisionEscalation::Fixed {
                assert!(!result
                    .resolved_paths
                    .keys()
                    .any(|p| p.starts_with("precision.escalation.value")));
            }
        }
    }
    let result: EffectiveConfiguration<SolverConfig> = resolve_configuration([
        ConfigurationLayer::from_serializable(ConfigSource::BuiltIn, "defaults", &base()).unwrap(),
        layer(
            ConfigSource::Project,
            serde_json::json!({"target":{"target":"interval","lower":"1","upper":"2"}}),
        ),
        layer(
            ConfigSource::CommandLine,
            serde_json::json!({"target":{"upper":"3"}}),
        ),
    ])
    .unwrap();
    assert_eq!(
        result.resolved.target,
        EigenTarget::Interval {
            lower: d("1"),
            upper: d("3")
        }
    );
    assert_eq!(result.resolved_paths["target.lower"], ConfigSource::Project);
}

#[test]
fn subspace_variant_replacement_matches_independent_direct_requests() {
    let variants = [
        Subspace::Full,
        Subspace::EvenReflection,
        Subspace::OddReflection,
        Subspace::ExplicitBasis {
            ambient_dimension: 7,
            reduced_dimension: 3,
            basis_id: "manufactured-basis".into(),
        },
        Subspace::Projector {
            ambient_dimension: 11,
            reduced_dimension: 4,
            projector_id: "manufactured-projector".into(),
        },
    ];
    for before in &variants {
        for after in &variants {
            let expected = SolverConfig {
                subspace: after.clone(),
                ..base()
            };
            let result: EffectiveConfiguration<SolverConfig> = resolve_configuration([
                ConfigurationLayer::from_serializable(ConfigSource::BuiltIn, "defaults", &base())
                    .unwrap(),
                layer(
                    ConfigSource::Project,
                    serde_json::json!({"subspace":before}),
                ),
                layer(
                    ConfigSource::CommandLine,
                    serde_json::json!({"subspace":after}),
                ),
            ])
            .unwrap();
            assert_eq!(result.resolved, expected);
            let direct: EffectiveConfiguration<SolverConfig> =
                resolve_configuration([ConfigurationLayer::from_serializable(
                    ConfigSource::BuiltIn,
                    "direct",
                    &expected,
                )
                .unwrap()])
                .unwrap();
            assert_eq!(result.digest, direct.digest);
            let object = serde_json::to_value(after).unwrap();
            for path in result
                .resolved_paths
                .keys()
                .filter(|path| path.starts_with("subspace."))
            {
                assert!(
                    object
                        .get(path.strip_prefix("subspace.").unwrap())
                        .is_some(),
                    "stale field {path}"
                );
            }
        }
    }
}

#[test]
fn ordinary_mode_fields_still_merge_without_declared_enum_semantics() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Ordinary {
        mode: String,
        knob: u32,
    }
    impl ValidateResolvedConfig for Ordinary {
        fn validate_resolved(&self) -> Result<(), ConfigError> {
            Ok(())
        }
    }
    let result: EffectiveConfiguration<Ordinary> = resolve_configuration([
        layer(
            ConfigSource::BuiltIn,
            serde_json::json!({"mode":"a","knob":17}),
        ),
        layer(ConfigSource::Project, serde_json::json!({"mode":"b"})),
    ])
    .unwrap();
    assert_eq!(result.resolved.mode, "b");
    assert_eq!(result.resolved.knob, 17);
}

#[test]
fn decimal_subnormal_conversion_rejects_relative_precision_loss() {
    for text in [
        "2.4703282292062328e-324",
        "2.5e-324",
        "3e-324",
        "7.4e-324",
        "1e-320",
        "1e-310",
        "-1e-320",
        "1e-400",
        "1e400",
    ] {
        assert!(
            DecimalLiteral::new(text).unwrap().parse_f64().is_err(),
            "{text}"
        );
    }
    for value in [
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.0,
        -1.0,
        f64::MAX,
        0.0,
        -0.0,
    ] {
        assert_eq!(
            DecimalLiteral::from_f64_exact(value)
                .unwrap()
                .parse_f64()
                .unwrap(),
            value
        );
    }
}
