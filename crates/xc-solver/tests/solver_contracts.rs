#![cfg(feature = "hp-reference")]
use rug::Float;
use xc_core::{
    AssuranceLevel, DecimalLiteral, EigenTarget, PrecisionEscalation, PrecisionPolicy, ResultStatus,
};
use xc_operator::{DenseSymmetricHp, MatrixStructure};
use xc_solver::*;
fn literal(s: &str) -> DecimalLiteral {
    DecimalLiteral::new(s).unwrap()
}

#[test]
fn bisection_work_limit_does_not_trigger_precision_escalation() {
    let p = 256;
    let d = vec![Float::with_val(p, 2); 8];
    let e = vec![Float::with_val(p, -1); 7];
    let problem = TridiagonalProblemHp::new(&d, &e).unwrap();
    let result = solve_tridiagonal_selected_eigenpairs_adaptive_hp(
        &problem,
        &HpAdaptiveSelectedTridiagonalOptions {
            first_index: 1,
            last_index: 1,
            absolute_tolerance: literal("1e-30"),
            maximum_bisection_iterations: 1,
            eigenvector_options: TridiagEigvecOptions::default(),
            precision: PrecisionPolicy {
                initial_bits: 64,
                maximum_bits: p,
                guard_bits: 0,
                escalation: PrecisionEscalation::AddBits(64),
            },
        },
    )
    .unwrap();
    let HpAdaptiveSelectedTridiagonalResult::Inconclusive {
        attempts, reason, ..
    } = result
    else {
        panic!("budget must be inconclusive")
    };
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, ResultStatus::Inconclusive);
    assert!(reason.contains("budget exhausted"));
}

#[test]
fn exact_decimal_interval_midpoint_is_admitted_and_guards_are_not_requested_residuals() {
    for p in [64, 128, 192] {
        let d = ["0.15", "0.27", "0.8"];
        let mut a = vec![Float::with_val(p, 0); 9];
        for (i, value) in d.iter().enumerate() {
            a[3 * i + i] = Float::with_val(p, Float::parse(value).unwrap());
        }
        let operator =
            DenseSymmetricHp::new("decimal interval", 3, a.clone(), p, &Float::with_val(p, 0))
                .unwrap();
        let factor =
            DenseShiftInvertFactorizationHp::factor("decimal midpoint", 3, &a, literal("0.2"), p)
                .unwrap();
        let config = BlockShiftInvertConfigHp {
            target: EigenTarget::Interval {
                lower: literal("0.1"),
                upper: literal("0.3"),
            },
            precision_bits: p,
            requested_eigenpairs: 1,
            guard_eigenpairs: 2,
            absolute_residual_tolerance: literal("1e-12"),
            scaled_backward_error_tolerance: literal("1e-12"),
            ritz_value_stability_tolerance: literal("1e-12"),
            boundary_cluster_tolerance: literal("1e-10"),
            maximum_iterations: 20,
            minimum_iterations: 2,
            maximum_projected_sweeps: 100,
        };
        let result = BlockShiftInvertSolverHp
            .solve(&operator, &factor, &config)
            .unwrap();
        assert_eq!(result.status, ResultStatus::Converged);
        assert!(
            Float::with_val(
                p,
                &result.retained_eigenpairs[0].eigenvalue
                    - Float::with_val(p, Float::parse("0.15").unwrap())
            )
            .abs()
                < Float::with_val(p, Float::parse("1e-12").unwrap())
        );
    }
}

#[test]
fn hp_planner_accounts_for_headers_limbs_and_dense_workspace() {
    for p in [64, 256, 1024] {
        let input = SolverPlannerInput {
            structure: MatrixStructure::Dense,
            dimension: 1000,
            target: EigenTarget::AlgebraicLargest,
            requested_eigenpairs: 1,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::fixed(p),
            matrix_materialized: true,
            generalized: false,
        };
        let plan = plan_symmetric_eigenproblem(&input).unwrap();
        let scalar = std::mem::size_of::<Float>() as u64 + u64::from(p).div_ceil(64) * 8;
        let resident = plan.resource_estimate.resident_memory_bytes.unwrap();
        let temporary = plan.resource_estimate.temporary_memory_bytes.unwrap();
        assert!(
            resident >= 1000 * scalar,
            "resident omitted live MPFR headers"
        );
        if plan.requires_materialization {
            assert!(
                resident + temporary >= 4_000_000 * scalar,
                "dense work omitted matrices"
            );
        }
    }
}

mod constrained_weighted_l1 {
    use xc_core::CancellationToken;
    use xc_solver::weighted_l1::*;
    use xc_solver::SolverError;

    fn vec_s(values: &[&str]) -> Vec<String> {
        values.iter().map(|x| (*x).into()).collect()
    }
    fn example() -> WeightedL1Problem {
        WeightedL1Problem {
            schema_version: 1,
            basis_id: "affine-columns".into(),
            target_id: "three-point-target".into(),
            normalization_id: "intercept-one".into(),
            quadrature_id: "weights-1-3-1".into(),
            basis: vec![vec_s(&["1", "0"]), vec_s(&["1", "1"]), vec_s(&["1", "2"])],
            target: vec_s(&["1", "0", "10"]),
            weights: vec_s(&["1", "3", "1"]),
            constraints: vec![vec_s(&["1", "0"])],
            rhs: vec_s(&["1"]),
        }
    }
    fn fit(p: &WeightedL1Problem) -> WeightedL1Fit {
        fit_weighted_l1(p, &WeightedL1Options::default(), &CancellationToken::new()).unwrap()
    }

    #[test]
    fn signed_optimum_matches_independently_known_primal_and_dual() {
        let p = example();
        let r = fit(&p);
        assert_eq!(r.witness.coefficients, vec_s(&["1", "-1"]));
        assert_eq!(r.verification.objective_lower, "11");
        assert_eq!(r.verification.objective_upper, "11");
        assert_eq!(r.verification.status, WeightedL1Status::ExactFiniteOptimum);
        assert!(r.verification.accepted);
        let hand = WeightedL1Witness {
            coefficients: vec_s(&["1", "-1"]),
            dual_z: vec_s(&["0", "-2", "1"]),
            dual_nu: vec_s(&["-1"]),
        };
        assert_eq!(
            verify_weighted_l1_witness(
                &p,
                &WeightedL1Options::default(),
                &hand,
                &CancellationToken::new()
            )
            .unwrap(),
            r.verification
        );
        let serialized = serde_json::to_vec(&r).unwrap();
        let replay: WeightedL1Fit = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(
            verify_weighted_l1_fit(
                &p,
                &WeightedL1Options::default(),
                &replay,
                &CancellationToken::new()
            )
            .unwrap(),
            r.verification
        );
    }

    #[test]
    fn median_negative_targets_nonunique_and_redundant_basis() {
        let mut p = example();
        p.basis = vec![vec_s(&["1"]); 3];
        p.target = vec_s(&["-100", "-1", "0"]);
        p.constraints.clear();
        p.rhs.clear();
        let r = fit(&p);
        assert_eq!(r.witness.coefficients, vec_s(&["-1"]));
        assert_eq!(r.verification.objective_upper, "100");
        // Nonunique minimizers are legitimate; no coefficient uniqueness claim.
        p.basis = vec![vec_s(&["1", "2"]); 2];
        p.target = vec_s(&["0", "2"]);
        p.weights = vec_s(&["1", "1"]);
        assert_eq!(fit(&p).verification.objective_upper, "2");
    }

    #[test]
    fn equality_phase_handles_signs_redundancy_zero_rows_and_inconsistency() {
        let mut p = example();
        p.constraints = vec![vec_s(&["-1", "0"]), vec_s(&["2", "0"]), vec_s(&["0", "0"])];
        p.rhs = vec_s(&["-1", "2", "0"]);
        assert_eq!(fit(&p).verification.objective_upper, "11");
        p.rhs[1] = "3".into();
        assert!(matches!(
            fit_weighted_l1(&p, &WeightedL1Options::default(), &CancellationToken::new()),
            Err(SolverError::InvalidConfiguration(_))
        ));
        p.constraints = vec![vec_s(&["0", "0"])];
        p.rhs = vec_s(&["1"]);
        assert!(
            fit_weighted_l1(&p, &WeightedL1Options::default(), &CancellationToken::new()).is_err()
        );
    }

    #[test]
    fn exact_arithmetic_retains_sub_binary64_differences_and_tiny_objectives() {
        let mut p = example();
        p.basis = vec![vec_s(&["1"]); 3];
        p.constraints.clear();
        p.rhs.clear();
        let epsilon = format!("1/1{}", "0".repeat(1000));
        let twice = format!("2/1{}", "0".repeat(1000));
        p.target = vec_s(&["0", &epsilon, &twice]);
        p.weights = vec_s(&["1", "3", "1"]);
        let r = fit(&p);
        assert_eq!(r.witness.coefficients, vec_s(&[&epsilon]));
        assert_eq!(
            rug::Rational::from(rug::Rational::parse(&r.verification.objective_upper).unwrap()),
            rug::Rational::from(rug::Rational::parse(&twice).unwrap())
        );
        // Distinct decimal columns round to the same binary64 value. Exact
        // equalities determine a=1,b=-1 and the sample residual is 1e-100.
        p.basis = vec![vec_s(&["1", "1"])];
        p.weights = vec_s(&["1"]);
        p.target = vec_s(&["1e-100"]);
        let near = format!("1.{}1", "0".repeat(99));
        p.constraints = vec![vec_s(&["1", "1"]), vec_s(&[&near, "1"])];
        p.rhs = vec_s(&["0", "1e-100"]);
        let r = fit(&p);
        assert_eq!(r.witness.coefficients, vec_s(&["1", "-1"]));
        assert_eq!(r.verification.optimality_gap, "0");
        assert_ne!(r.verification.objective_upper, "0");
    }

    #[test]
    fn rational_decimal_inputs_and_basis_rescaling_preserve_objective() {
        let p = example();
        let original = fit(&p);
        let mut rescaled = p.clone();
        rescaled.basis = vec![
            vec_s(&["2e0", "0"]),
            vec_s(&["2", "0.25"]),
            vec_s(&["2", "1/2"]),
        ];
        rescaled.constraints = vec![vec_s(&["2.0", "0"])];
        let result = fit(&rescaled);
        assert_eq!(result.verification, original.verification);
        assert_eq!(result.witness.coefficients, vec_s(&["1/2", "-4"]));
        assert_ne!(result.problem_sha256, original.problem_sha256);
    }

    #[test]
    fn invalid_data_and_preallocation_resource_limits_fail_closed() {
        let p = example();
        let options = WeightedL1Options::default();
        let token = CancellationToken::new();
        for scalar in [
            "NaN",
            "inf",
            "1e1000000000",
            "1/0",
            "1/-2",
            "1/2/3",
            " 1",
            "1 2",
        ] {
            let mut bad = p.clone();
            bad.target[0] = scalar.into();
            assert!(fit_weighted_l1(&bad, &options, &token).is_err(), "{scalar}");
        }
        for weight in ["0", "-1"] {
            let mut bad = p.clone();
            bad.weights[0] = weight.into();
            assert!(fit_weighted_l1(&bad, &options, &token).is_err());
        }
        let mut bad = p.clone();
        bad.basis[0].pop();
        assert!(fit_weighted_l1(&bad, &options, &token).is_err());
        let mut bad = p.clone();
        bad.target_id.clear();
        assert!(fit_weighted_l1(&bad, &options, &token).is_err());
        let mut limited = options.clone();
        limited.maximum_pivots = 0;
        assert!(matches!(
            fit_weighted_l1(&p, &limited, &token),
            Err(SolverError::IterationBudgetExhausted(_))
        ));
        limited = options.clone();
        limited.maximum_tableau_cells = 1;
        assert!(matches!(
            fit_weighted_l1(&p, &limited, &token),
            Err(SolverError::IterationBudgetExhausted(_))
        ));
        limited = options;
        limited.maximum_rational_bits = 64;
        bad = p.clone();
        bad.target[0] = "1e100".into();
        assert!(matches!(
            fit_weighted_l1(&bad, &limited, &token),
            Err(SolverError::IterationBudgetExhausted(_))
        ));
        let cancelled = CancellationToken::with_wall_time_limit(std::time::Duration::ZERO);
        assert!(matches!(
            fit_weighted_l1(&p, &WeightedL1Options::default(), &cancelled),
            Err(SolverError::Cancelled(_))
        ));
    }

    #[test]
    fn forged_records_and_bad_primal_or_dual_witnesses_are_rejected() {
        let p = example();
        let r = fit(&p);
        let options = WeightedL1Options::default();
        let token = CancellationToken::new();
        let mut other = p.clone();
        other.target_id.push('x');
        assert!(verify_weighted_l1_fit(&other, &options, &r, &token).is_err());
        let mut changed = options.clone();
        changed.absolute_optimality_gap = "1".into();
        assert!(verify_weighted_l1_fit(&p, &changed, &r, &token).is_err());
        let mut bad = r.clone();
        bad.scope = "continuous optimum".into();
        assert!(verify_weighted_l1_fit(&p, &options, &bad, &token).is_err());
        let mut bad = r.clone();
        bad.verification.objective_upper = "0".into();
        assert!(verify_weighted_l1_fit(&p, &options, &bad, &token).is_err());
        let mut bad = r.witness.clone();
        bad.coefficients[0] = "2".into();
        assert!(verify_weighted_l1_witness(&p, &options, &bad, &token).is_err());
        let mut bad = r.witness.clone();
        bad.dual_z[0] = "100".into();
        assert!(verify_weighted_l1_witness(&p, &options, &bad, &token).is_err());
        let mut bad = r.witness.clone();
        bad.dual_nu[0] = "100".into();
        assert!(verify_weighted_l1_witness(&p, &options, &bad, &token).is_err());
    }

    struct Loose;
    impl WeightedL1Backend for Loose {
        fn name(&self) -> &str {
            "independent-zero-dual-candidate"
        }
        fn propose(
            &self,
            _: &WeightedL1Problem,
            _: &WeightedL1Options,
            _: &CancellationToken,
        ) -> Result<(WeightedL1Witness, usize), SolverError> {
            Ok((
                WeightedL1Witness {
                    coefficients: vec_s(&["1", "0"]),
                    dual_z: vec_s(&["0", "0", "0"]),
                    dual_nu: vec_s(&["0"]),
                },
                0,
            ))
        }
    }

    #[test]
    fn replaceable_backend_keeps_unresolved_gaps_and_exact_tolerance_boundaries() {
        let p = example();
        let mut options = WeightedL1Options::default();
        let token = CancellationToken::new();
        let r = fit_weighted_l1_with_backend(&p, &options, &Loose, &token).unwrap();
        assert_eq!(r.verification.objective_upper, "12");
        assert_eq!(r.verification.optimality_gap, "12");
        assert_eq!(r.verification.status, WeightedL1Status::GapAboveTolerance);
        assert!(!r.verification.accepted);
        options.absolute_optimality_gap = "12".into();
        assert!(
            fit_weighted_l1_with_backend(&p, &options, &Loose, &token)
                .unwrap()
                .verification
                .accepted
        );
        options.absolute_optimality_gap =
            "11.99999999999999999999999999999999999999999999999".into();
        assert!(
            !fit_weighted_l1_with_backend(&p, &options, &Loose, &token)
                .unwrap()
                .verification
                .accepted
        );
        options.absolute_optimality_gap = "0".into();
        options.relative_optimality_gap = "1".into();
        assert!(
            fit_weighted_l1_with_backend(&p, &options, &Loose, &token)
                .unwrap()
                .verification
                .accepted
        );
        let mut forged = r;
        forged.verification.accepted = true;
        assert!(
            verify_weighted_l1_fit(&p, &WeightedL1Options::default(), &forged, &token).is_err()
        );
    }

    #[test]
    fn all_zero_targets_and_fully_determined_coefficients() {
        let mut p = example();
        p.target = vec_s(&["0", "0", "0"]);
        p.rhs = vec_s(&["0"]);
        let r = fit(&p);
        assert_eq!(r.verification.objective_upper, "0");
        assert_eq!(r.verification.optimality_gap, "0");
        p.constraints = vec![vec_s(&["1", "0"]), vec_s(&["0", "1"])];
        p.rhs = vec_s(&["-2", "3"]);
        let r = fit(&p);
        assert_eq!(r.witness.coefficients, vec_s(&["-2", "3"]));
        assert_eq!(r.verification.objective_upper, "9");
    }
}

mod streamed_trial_energy_contracts {
    use rug::Rational;
    use std::io::Cursor;
    use xc_core::{CancellationReason, CancellationToken};
    use xc_solver::{
        trial_energy::{self as dense, streaming::*, ExactBounds, TrialVector},
        SolverError,
    };

    fn pt(s: &str) -> ExactBounds {
        ExactBounds::point(s)
    }
    fn meta(id: &str, n: usize) -> FormMetadata {
        FormMetadata {
            source_id: id.into(),
            basis_id: "coordinates".into(),
            normalization_id: "physical".into(),
            dimension: n,
        }
    }
    struct Reader {
        values: Vec<ExactBounds>,
        pos: usize,
        chunk: usize,
    }
    impl Reader {
        fn new(values: &[ExactBounds], chunk: usize) -> Self {
            Self {
                values: values.to_vec(),
                pos: 0,
                chunk,
            }
        }
    }
    impl FormReader for Reader {
        fn next_block(
            &mut self,
            max: usize,
            _bytes: usize,
        ) -> Result<Option<FormBlock>, SolverError> {
            if self.pos == self.values.len() {
                return Ok(None);
            }
            let end = (self.pos + self.chunk.min(max)).min(self.values.len());
            let block = FormBlock {
                start: self.pos as u64,
                entries: self.values[self.pos..end].to_vec(),
            };
            self.pos = end;
            Ok(Some(block))
        }
    }
    fn source(m: FormMetadata, values: &[ExactBounds]) -> FormSource {
        let record = fingerprint(
            &m,
            &mut Reader::new(values, 2),
            &Options::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        FormSource {
            metadata: m,
            sha256: record.sha256,
        }
    }
    fn vector(label: &str, values: &[&str]) -> TrialVector {
        TrialVector {
            label: label.into(),
            source_id: format!("source-{label}"),
            operator_id: "A".into(),
            metric_id: "G".into(),
            basis_id: "coordinates".into(),
            normalization_id: "physical".into(),
            coefficients: values.iter().map(|v| pt(v)).collect(),
        }
    }
    fn fixture() -> (Problem, Vec<ExactBounds>) {
        let a = vec![pt("2"), pt("1"), pt("3")];
        let p = Problem {
            schema_version: 1,
            operator: source(meta("A", 2), &a),
            metric: Metric::Identity {
                metadata: meta("G", 2),
            },
            baseline: vector("base", &["1", "0"]),
            corrections: vec![
                vector("first", &["-1", "2"]),
                vector("second", &["2", "-1"]),
            ],
        };
        (p, a)
    }
    fn run(p: &Problem, a: &[ExactBounds]) -> Report {
        analyze(
            p,
            &Options::default(),
            &mut Reader::new(a, 1),
            None,
            &CancellationToken::new(),
        )
        .unwrap()
    }
    fn exact(b: &ExactBounds, s: &str) {
        assert_eq!(*b, pt(s));
    }
    #[test]
    fn dense_agreement_cross_terms_prefix_stages_and_portable_replay() {
        let (p, a) = fixture();
        let r = run(&p, &a);
        let form = |id: &str, values: Vec<ExactBounds>| dense::TrialForm {
            source_id: id.into(),
            basis_id: "coordinates".into(),
            normalization_id: "physical".into(),
            dimension: 2,
            entries: values,
        };
        let d = dense::analyze_trial_energy(
            &dense::TrialEnergyProblem {
                schema_version: 1,
                operator: form("A", vec![pt("2"), pt("1"), pt("1"), pt("3")]),
                metric: form("G", vec![pt("1"), pt("0"), pt("0"), pt("1")]),
                baseline: p.baseline.clone(),
                corrections: p.corrections.clone(),
            },
            &dense::TrialEnergyOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(r.energy, d.energy);
        assert_eq!(r.norm, d.norm);
        assert_eq!(
            r.stages[0].measurement.normalized.as_ref().unwrap(),
            &d.baseline
        );
        assert_eq!(
            r.stages[2].measurement.normalized.as_ref().unwrap(),
            &d.corrected
        );
        exact(&r.stages[1].measurement.energy, "12");
        exact(
            r.stages[1].energy_change_from_previous.as_ref().unwrap(),
            "10",
        );
        exact(
            r.stages[2].energy_change_from_previous.as_ref().unwrap(),
            "3",
        );
        exact(&r.energy_pairings[1], "0");
        assert_eq!(r.metric_global_lower_bound.as_deref(), Some("1"));
        let json = serde_json::to_vec(&r).unwrap();
        let copy: Report = serde_json::from_slice(&json).unwrap();
        verify(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 3),
            None,
            &copy,
            &CancellationToken::new(),
        )
        .unwrap();
    }
    #[test]
    fn digest_binds_metadata_and_endpoint_text_but_not_blocks() {
        let (p, a) = fixture();
        let x = fingerprint(
            &p.operator.metadata,
            &mut Reader::new(&a, 1),
            &Options::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let y = fingerprint(
            &p.operator.metadata,
            &mut Reader::new(&a, 3),
            &Options::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(x, y);
        let mut m = p.operator.metadata.clone();
        m.source_id = "other".into();
        assert_ne!(source(m, &a).sha256, x.sha256);
        let mut changed = a.clone();
        changed[0] = pt("2.0");
        assert_ne!(
            source(p.operator.metadata.clone(), &changed).sha256,
            x.sha256
        );
        assert!(analyze(
            &p,
            &Options::default(),
            &mut Reader::new(&changed, 3),
            None,
            &CancellationToken::new()
        )
        .is_err());
    }
    #[test]
    fn zero_and_dependent_corrections_do_not_require_a_full_rank_trial_basis() {
        let (mut p, a) = fixture();
        p.corrections = vec![vector("zero", &["0", "0"]), vector("same", &["2", "0"])];
        let r = run(&p, &a);
        exact(&r.energy.total, "18");
        exact(&r.norm.total, "9");
        assert_eq!(
            r.parts[1].status,
            MeasurementStatus::NormNotSeparatedFromZero
        );
        assert_eq!(
            r.stages[2].measurement.status,
            MeasurementStatus::CertifiedFinite
        );
        p.corrections = vec![vector("opposite", &["-1", "0"])];
        let r = run(&p, &a);
        exact(&r.energy.total, "0");
        exact(&r.norm.total, "0");
        assert!(r.stages[1].measurement.normalized.is_none());
        assert!(r.common_norm_energy_contributions.is_none());
        exact(r.stages[1].measurement.norm.as_ref().unwrap(), "0");
    }
    #[test]
    fn exact_large_cancellation_and_tiny_nonzero_energy_survive() {
        let big = Rational::from_str_radix(
            "100000000000000000000000000000000000000000000000000000000000000000000000000000000",
            10,
        )
        .unwrap();
        let tiny =
            Rational::from_str_radix("1/10000000000000000000000000000000000000000", 10).unwrap();
        let last = big.clone() + tiny.clone();
        let a = vec![
            pt(&big.to_string()),
            pt(&big.to_string()),
            pt(&last.to_string()),
        ];
        let (mut p, _) = fixture();
        p.operator = source(meta("A", 2), &a);
        p.corrections = vec![vector("cancel", &["0", "-1"])];
        let r = run(&p, &a);
        exact(&r.energy.total, &tiny.to_string());
        exact(
            &r.stages[1]
                .measurement
                .normalized
                .as_ref()
                .unwrap()
                .rayleigh_quotient,
            &(tiny.clone() / 2i32).to_string(),
        );
        let sum = big * 4i32 + tiny.clone();
        exact(
            &r.energy.coefficient_cancellation.absolute_term_sum,
            &sum.to_string(),
        );
        exact(
            r.energy
                .coefficient_cancellation
                .amplification
                .as_ref()
                .unwrap(),
            &(sum / tiny).to_string(),
        );
    }
    #[test]
    fn identity_norm_sqrt_bounds_and_unresolved_normalization() {
        let (mut p, a) = fixture();
        p.baseline = vector("base", &["1", "1"]);
        p.corrections.clear();
        let r = run(&p, &a);
        let norm = r.parts[0].norm.as_ref().unwrap();
        let lo = Rational::from_str_radix(&norm.lower, 10).unwrap();
        let hi = Rational::from_str_radix(&norm.upper, 10).unwrap();
        assert!(lo.clone() * lo <= 2);
        assert!(hi.clone() * hi >= 2);
        p.baseline.coefficients[0] = ExactBounds {
            lower: "-1".into(),
            upper: "1".into(),
        };
        p.baseline.coefficients[1] = pt("0");
        let r = run(&p, &a);
        assert_eq!(
            r.parts[0].status,
            MeasurementStatus::NormNotSeparatedFromZero
        );
        assert!(r.parts[0].normalized.is_none());
    }
    #[test]
    fn streamed_metric_global_and_restricted_positivity_are_distinguished() {
        let (mut p, a) = fixture();
        let g = vec![pt("2"), pt("1"), pt("2")];
        p.metric = Metric::Stream {
            source: source(meta("G", 2), &g),
        };
        let r = analyze(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 2),
            Some(&mut Reader::new(&g, 2)),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(
            r.metric_status,
            MetricStatus::PositiveStrictDiagonalDominance
        );
        assert_eq!(r.metric_global_lower_bound.as_deref(), Some("1"));
        exact(&r.norm.total, "14");
        let g = vec![pt("1"), pt("2"), pt("5")];
        p.metric = Metric::Stream {
            source: source(meta("G", 2), &g),
        };
        p.baseline = vector("base", &["1", "0"]);
        p.corrections = vec![vector("second", &["0", "1"])];
        let r = analyze(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 2),
            Some(&mut Reader::new(&g, 2)),
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(r.metric_status, MetricStatus::PositiveOnTrialSpan);
        assert!(r.metric_global_lower_bound.is_none());
        exact(&r.norm.total, "10");
        let mut corrupt = r.clone();
        corrupt.metric_status = MetricStatus::PositiveStrictDiagonalDominance;
        assert!(verify(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 2),
            Some(&mut Reader::new(&g, 2)),
            &corrupt,
            &CancellationToken::new()
        )
        .is_err());
    }
    #[test]
    fn indefinite_metric_retains_raw_data_without_certified_quotients() {
        let (mut p, a) = fixture();
        p.metric = Metric::Diagonal {
            metadata: meta("G", 2),
            entries: vec![pt("1"), pt("-1")],
        };
        let r = run(&p, &a);
        assert_eq!(r.metric_status, MetricStatus::Unresolved);
        assert!(r
            .parts
            .iter()
            .all(|m| m.normalized.is_none() && m.norm.is_none()));
        assert!(r.stages.iter().all(|m| m.measurement.normalized.is_none()));
        exact(&r.energy.total, "15");
        exact(&r.norm.total, "3");
        assert!(r.common_norm_energy_contributions.is_none());
    }
    #[test]
    fn diagonal_metric_and_structural_identity_failures() {
        let (mut p, a) = fixture();
        p.metric = Metric::Diagonal {
            metadata: meta("G", 2),
            entries: vec![pt("2"), pt("3")],
        };
        let r = run(&p, &a);
        assert_eq!(r.metric_status, MetricStatus::PositiveDiagonal);
        exact(&r.norm.total, "11");
        let token = CancellationToken::new();
        assert!(analyze(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 2),
            Some(&mut Reader::new(&a, 2)),
            &token
        )
        .is_err());
        p.corrections[0].basis_id = "foreign".into();
        assert!(analyze(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 2),
            None,
            &token
        )
        .is_err());
        let (mut p, a) = fixture();
        p.corrections[0].label = p.baseline.label.clone();
        assert!(analyze(
            &p,
            &Options::default(),
            &mut Reader::new(&a, 2),
            None,
            &token
        )
        .is_err());
    }
    struct Blocks(Vec<FormBlock>);
    impl FormReader for Blocks {
        fn next_block(
            &mut self,
            _: usize,
            _bytes: usize,
        ) -> Result<Option<FormBlock>, SolverError> {
            Ok(if self.0.is_empty() {
                None
            } else {
                Some(self.0.remove(0))
            })
        }
    }
    #[test]
    fn missing_duplicate_out_of_order_extra_and_oversized_blocks_fail_closed() {
        let (p, a) = fixture();
        let token = CancellationToken::new();
        for blocks in [
            vec![FormBlock {
                start: 0,
                entries: a[..2].to_vec(),
            }],
            vec![FormBlock {
                start: 1,
                entries: a.clone(),
            }],
            vec![
                FormBlock {
                    start: 0,
                    entries: a.clone(),
                },
                FormBlock {
                    start: 0,
                    entries: vec![pt("0")],
                },
            ],
            vec![
                FormBlock {
                    start: 0,
                    entries: a.clone(),
                },
                FormBlock {
                    start: 3,
                    entries: vec![pt("0")],
                },
            ],
            vec![FormBlock {
                start: 0,
                entries: vec![],
            }],
        ] {
            assert!(analyze(&p, &Options::default(), &mut Blocks(blocks), None, &token).is_err());
        }
        let o = Options {
            maximum_block_entries: 1,
            ..Options::default()
        };
        assert!(analyze(
            &p,
            &o,
            &mut Blocks(vec![FormBlock {
                start: 0,
                entries: a
            }]),
            None,
            &token
        )
        .is_err());
    }
    #[test]
    fn replay_rejects_tampered_scope_pairings_norms_changes_and_resource_counts() {
        let (p, a) = fixture();
        let r = run(&p, &a);
        let mut cases = Vec::new();
        let mut x = r.clone();
        x.scope = "continuum".into();
        cases.push(x);
        let mut x = r.clone();
        x.energy_pairings[0] = pt("0");
        cases.push(x);
        let mut x = r.clone();
        x.parts[0].norm = Some(pt("7"));
        cases.push(x);
        let mut x = r.clone();
        x.stages[1].energy_change_from_previous = Some(pt("0"));
        cases.push(x);
        let mut x = r.clone();
        x.interval_operations += 1;
        cases.push(x);
        let mut x = r.clone();
        x.streamed_scalar_bytes += 1;
        cases.push(x);
        let mut x = r.clone();
        x.energy.coefficient_cancellation.absolute_term_sum = pt("1");
        cases.push(x);
        for x in cases {
            assert!(verify(
                &p,
                &Options::default(),
                &mut Reader::new(&a, 3),
                None,
                &x,
                &CancellationToken::new()
            )
            .is_err());
        }
        let mut wire = serde_json::to_value(&p).unwrap();
        wire["infinite_tail_certified"] = true.into();
        assert!(serde_json::from_value::<Problem>(wire).is_err());
    }
    #[test]
    fn work_limits_are_not_converted_to_metric_unresolved() {
        let (p, a) = fixture();
        for o in [
            Options {
                maximum_interval_operations: 0,
                ..Options::default()
            },
            Options {
                maximum_stream_entries: 2,
                ..Options::default()
            },
            Options {
                maximum_stream_scalar_bytes: 1,
                ..Options::default()
            },
            Options {
                maximum_vector_scalar_bytes: 1,
                ..Options::default()
            },
            Options {
                maximum_dimension: 1,
                ..Options::default()
            },
            Options {
                maximum_parts: 2,
                ..Options::default()
            },
        ] {
            assert!(analyze(
                &p,
                &o,
                &mut Reader::new(&a, 2),
                None,
                &CancellationToken::new()
            )
            .is_err());
        }
        let mut b = a.clone();
        b[0] = pt("1e100");
        let o = Options {
            maximum_rational_bits: 64,
            ..Options::default()
        };
        assert!(fingerprint(
            &p.operator.metadata,
            &mut Reader::new(&b, 2),
            &o,
            &CancellationToken::new()
        )
        .is_err());
        let mut b = a;
        b[0] = ExactBounds {
            lower: "3".into(),
            upper: "2".into(),
        };
        assert!(fingerprint(
            &p.operator.metadata,
            &mut Reader::new(&b, 2),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
    }
    #[test]
    fn cancellation_before_and_during_a_stream_refuses_a_report() {
        let (p, a) = fixture();
        let token = CancellationToken::new();
        token.cancel(CancellationReason::UserRequested);
        assert!(matches!(
            analyze(
                &p,
                &Options::default(),
                &mut Reader::new(&a, 2),
                None,
                &token
            ),
            Err(SolverError::Cancelled(_))
        ));
        struct CancelReader<'a>(&'a CancellationToken, Reader);
        impl FormReader for CancelReader<'_> {
            fn next_block(
                &mut self,
                n: usize,
                bytes: usize,
            ) -> Result<Option<FormBlock>, SolverError> {
                if self.1.pos > 0 {
                    self.0.cancel(CancellationReason::UserRequested);
                }
                self.1.next_block(n, bytes)
            }
        }
        let token = CancellationToken::new();
        assert!(matches!(
            analyze(
                &p,
                &Options::default(),
                &mut CancelReader(&token, Reader::new(&a, 1)),
                None,
                &token
            ),
            Err(SolverError::Cancelled(_))
        ));
    }
    #[test]
    fn bounded_json_lines_reader_and_strict_record_schema() {
        let (p, a) = fixture();
        let bytes = a
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let r = analyze(
            &p,
            &Options::default(),
            &mut JsonLinesReader::new(Cursor::new(bytes)),
            None,
            &CancellationToken::new(),
        )
        .unwrap();
        exact(&r.energy.total, "15");
        for bytes in [
            "\n".to_string(),
            "{}\n".into(),
            "{\"lower\":\"1\",\"upper\":\"1\",\"extra\":0}".into(),
            "x".repeat(262_401),
        ] {
            assert!(JsonLinesReader::new(Cursor::new(bytes))
                .next_block(1, 16_777_216)
                .is_err());
        }
        assert!(JsonLinesReader::new(Cursor::new(Vec::<u8>::new()))
            .next_block(0, 16_777_216)
            .is_err());
    }

    #[test]
    fn per_block_byte_budget_splits_native_reader_and_rejects_noncompliant_readers() {
        let (p, a) = fixture();
        let wire = a
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let o = Options {
            maximum_block_scalar_bytes: 2,
            ..Options::default()
        };
        let r = analyze(
            &p,
            &o,
            &mut JsonLinesReader::new(Cursor::new(wire)),
            None,
            &CancellationToken::new(),
        )
        .unwrap();
        exact(&r.energy.total, "15");
        assert!(analyze(
            &p,
            &o,
            &mut Reader::new(&a, 3),
            None,
            &CancellationToken::new()
        )
        .is_err());
        let wire = serde_json::to_vec(&pt("100")).unwrap();
        assert!(JsonLinesReader::new(Cursor::new(wire))
            .next_block(1, 2)
            .is_err());
    }

    #[test]
    fn dimension_above_dense_limit_without_retaining_the_matrix() {
        struct DiagonalStream {
            n: usize,
            i: usize,
            j: usize,
            pos: u64,
        }
        impl FormReader for DiagonalStream {
            fn next_block(
                &mut self,
                max: usize,
                bytes: usize,
            ) -> Result<Option<FormBlock>, SolverError> {
                if self.i == self.n {
                    return Ok(None);
                }
                let start = self.pos;
                let mut entries = Vec::new();
                let mut used = 0;
                while self.i < self.n && entries.len() < max.min(97) {
                    let v = if self.i == self.j {
                        (self.i + 1).to_string()
                    } else {
                        "0".into()
                    };
                    if used + 2 * v.len() > bytes {
                        break;
                    }
                    used += 2 * v.len();
                    entries.push(ExactBounds::point(v));
                    self.pos += 1;
                    self.j += 1;
                    if self.j == self.n {
                        self.i += 1;
                        self.j = self.i;
                    }
                }
                Ok(Some(FormBlock { start, entries }))
            }
        }
        let make = || DiagonalStream {
            n: 577,
            i: 0,
            j: 0,
            pos: 0,
        };
        let m = meta("A", 577);
        let o = Options::default();
        let token = CancellationToken::new();
        let imported = fingerprint(&m, &mut make(), &o, &token).unwrap();
        let mut base = vector("base", &[]);
        base.coefficients = vec![pt("1"); 577];
        let p = Problem {
            schema_version: 1,
            operator: FormSource {
                metadata: m,
                sha256: imported.sha256,
            },
            metric: Metric::Identity {
                metadata: meta("G", 577),
            },
            baseline: base,
            corrections: vec![],
        };
        let r = analyze(&p, &o, &mut make(), None, &token).unwrap();
        exact(&r.energy.total, "166753");
        exact(&r.norm.total, "577");
        exact(
            &r.stages[0]
                .measurement
                .normalized
                .as_ref()
                .unwrap()
                .rayleigh_quotient,
            "289",
        );
        assert_eq!(r.streamed_entries, 166753);
    }
}

mod trial_energy_contracts {
    use rug::Rational;
    use xc_core::CancellationToken;
    use xc_solver::trial_energy::*;

    fn q(s: &str) -> Rational {
        Rational::from_str_radix(s, 10).unwrap()
    }
    fn form(id: &str, values: &[&str]) -> TrialForm {
        TrialForm {
            source_id: id.into(),
            basis_id: "two-coordinates".into(),
            normalization_id: "physical-mass".into(),
            dimension: 2,
            entries: values.iter().map(|x| ExactBounds::point(*x)).collect(),
        }
    }
    fn vector(label: &str, values: &[&str]) -> TrialVector {
        TrialVector {
            label: label.into(),
            source_id: format!("source-{label}"),
            operator_id: "matrix".into(),
            metric_id: "mass".into(),
            basis_id: "two-coordinates".into(),
            normalization_id: "physical-mass".into(),
            coefficients: values.iter().map(|x| ExactBounds::point(*x)).collect(),
        }
    }
    fn problem() -> TrialEnergyProblem {
        TrialEnergyProblem {
            schema_version: 1,
            operator: form("matrix", &["2", "1", "1", "3"]),
            metric: form("mass", &["2", "1", "1", "2"]),
            baseline: vector("base", &["1", "0"]),
            corrections: vec![
                vector("first", &["-1", "2"]),
                vector("second", &["2", "-1"]),
            ],
        }
    }
    fn run(p: &TrialEnergyProblem) -> TrialEnergyReport {
        analyze_trial_energy(p, &TrialEnergyOptions::default(), &CancellationToken::new()).unwrap()
    }
    fn exact(v: &ExactBounds, s: &str) {
        assert_eq!(*v, ExactBounds::point(s));
    }
    fn interval(a: &str, b: &str) -> ExactBounds {
        ExactBounds {
            lower: a.into(),
            upper: b.into(),
        }
    }

    #[test]
    fn exact_cross_terms_metric_and_portable_replay() {
        let p = problem();
        let r = run(&p);
        assert_eq!(r.metric_check, "positive_interval_ldl_metric");
        let terms: Vec<_> = r
            .energy
            .contributions
            .iter()
            .map(|t| t.value.lower.as_str())
            .collect();
        assert_eq!(terms, ["2", "0", "6", "10", "-10", "7"]);
        exact(&r.energy.total, "15");
        exact(&r.energy.closure_residual, "0");
        exact(&r.energy.change_from_baseline, "13");
        exact(&r.norm.total, "14");
        exact(&r.baseline.rayleigh_quotient, "1");
        exact(&r.corrected.rayleigh_quotient, "15/14");
        exact(&r.rayleigh_change, "1/14");
        let normalized: Vec<_> = r
            .common_norm_energy_contributions
            .iter()
            .map(|t| t.value.lower.as_str())
            .collect();
        assert_eq!(normalized, ["1/7", "0", "3/7", "5/7", "-5/7", "1/2"]);
        exact(&r.common_norm_expansion_sum, "15/14");
        exact(&r.common_norm_closure_residual, "0");
        assert_eq!(r.corrected.rayleigh_absolute_error_bound, "0");
        exact(
            r.energy
                .correction_cancellation
                .amplification
                .as_ref()
                .unwrap(),
            "7/3",
        );
        exact(
            r.energy
                .coefficient_cancellation
                .amplification
                .as_ref()
                .unwrap(),
            "1",
        );
        let replay: TrialEnergyReport =
            serde_json::from_slice(&serde_json::to_vec(&r).unwrap()).unwrap();
        verify_trial_energy(
            &p,
            &TrialEnergyOptions::default(),
            &replay,
            &CancellationToken::new(),
        )
        .unwrap();
    }

    #[test]
    fn common_vector_scaling_and_congruent_basis_change_preserve_rayleigh() {
        let p = problem();
        let original = run(&p);
        let mut scaled = p.clone();
        for v in std::iter::once(&mut scaled.baseline).chain(&mut scaled.corrections) {
            for x in &mut v.coefficients {
                *x = ExactBounds::point((q(&x.lower) * -7i32).to_string());
            }
        }
        let r = run(&scaled);
        exact(&r.energy.total, "735");
        exact(&r.norm.total, "686");
        assert_eq!(
            r.corrected.rayleigh_quotient,
            original.corrected.rayleigh_quotient
        );
        // x=S z, A'=S^T A S and G'=S^T G S, S=diag(2,-3).
        let mut changed = p.clone();
        let scales = [2i32, -3];
        for f in [&mut changed.operator, &mut changed.metric] {
            f.basis_id = "rescaled-coordinates".into();
            for i in 0..2 {
                for j in 0..2 {
                    f.entries[2 * i + j] = ExactBounds::point(
                        (q(&f.entries[2 * i + j].lower) * scales[i] * scales[j]).to_string(),
                    );
                }
            }
        }
        for v in std::iter::once(&mut changed.baseline).chain(&mut changed.corrections) {
            v.basis_id = "rescaled-coordinates".into();
            for (x, s) in v.coefficients.iter_mut().zip(scales) {
                *x = ExactBounds::point((q(&x.lower) / s).to_string());
            }
        }
        let r = run(&changed);
        assert_eq!(r.corrected, original.corrected);
        assert_ne!(r.problem_sha256, original.problem_sha256);
        changed.corrections.reverse();
        assert_eq!(run(&changed).corrected, original.corrected);
    }

    #[test]
    fn tiny_profile_change_can_have_large_energy_and_extreme_cancellation_survives() {
        let mut p = problem();
        p.metric = form("mass", &["1", "0", "0", "1"]);
        p.operator = form("matrix", &["1", "0", "0", "1e1000"]);
        p.corrections = vec![vector("small", &["0", "1e-100"])];
        let r = run(&p);
        assert!(q(&r.corrected.rayleigh_quotient.lower) > q(&format!("1{}", "0".repeat(799))));
        assert_eq!(r.corrected.rayleigh_absolute_error_bound, "0");
        p.corrections.clear();
        p.baseline = vector("base", &["1", "1"]);
        let almost = format!("1.{}1", "0".repeat(999));
        p.operator = form("matrix", &["1", "-1", "-1", &almost]);
        let r = run(&p);
        let tiny = format!("1/1{}", "0".repeat(1000));
        exact(&r.energy.total, &tiny);
        let ratio = r
            .energy
            .coefficient_cancellation
            .amplification
            .as_ref()
            .unwrap();
        assert_eq!(q(&ratio.lower), q(&format!("4{}1", "0".repeat(999))));
    }

    #[test]
    fn zero_energy_is_distinct_from_zero_norm_and_unresolved_cancellation() {
        let mut p = problem();
        p.corrections.clear();
        p.baseline = vector("base", &["1", "1"]);
        p.operator = form("matrix", &["1", "0", "0", "-1"]);
        let r = run(&p);
        exact(&r.corrected.energy, "0");
        assert_eq!(
            r.energy.coefficient_cancellation.status,
            CancellationStatus::ExactZeroTotal
        );
        assert!(r.energy.coefficient_cancellation.amplification.is_none());
        p.operator.entries[3] = interval("-101/100", "-99/100");
        let r = run(&p);
        assert_eq!(
            r.energy.coefficient_cancellation.status,
            CancellationStatus::TotalContainsZero
        );
        assert!(r.energy.coefficient_cancellation.amplification.is_none());
        p.operator = form("matrix", &["0", "0", "0", "0"]);
        assert_eq!(
            run(&p).energy.coefficient_cancellation.status,
            CancellationStatus::AllTermsZero
        );
    }

    #[test]
    fn interval_bounds_cover_source_uncertainty_and_explicit_radius() {
        let mut p = problem();
        p.corrections = vec![vector("first", &["1", "0"])];
        p.baseline = vector("base", &["1", "0"]);
        p.metric = form("mass", &["1", "0", "0", "1"]);
        p.operator = form("matrix", &["2", "0", "0", "3"]);
        p.operator.entries[0] = interval("19/10", "21/10");
        p.corrections[0].coefficients[0] = interval("9/10", "11/10");
        let r = run(&p);
        assert!(q(&r.energy.total.lower) <= q("6859/1000"));
        assert!(q(&r.energy.total.upper) >= q("9261/1000"));
        assert!(q(&r.corrected.rayleigh_quotient.lower) <= q("19/10"));
        assert!(q(&r.corrected.rayleigh_quotient.upper) >= q("21/10"));
        let mid = q(&r.corrected.rayleigh_midpoint);
        let rad = q(&r.corrected.rayleigh_absolute_error_bound);
        assert_eq!(mid.clone() - &rad, q(&r.corrected.rayleigh_quotient.lower));
        assert_eq!(mid + rad, q(&r.corrected.rayleigh_quotient.upper));
        assert!(
            q(&r.energy.closure_residual.lower) <= 0 && q(&r.energy.closure_residual.upper) >= 0
        );
        // No hidden renormalization: scaling one correction changes the problem.
        p.corrections[0].coefficients[0] = ExactBounds::point("3");
        assert_ne!(run(&p).problem_sha256, r.problem_sha256);
    }

    #[test]
    fn rejects_bad_coordinates_symmetry_metrics_and_zero_normalization() {
        let p = problem();
        let mut bad = vec![];
        let mut x = p.clone();
        x.baseline.operator_id = "another".into();
        bad.push(x);
        let mut x = p.clone();
        x.corrections[1].metric_id = "another".into();
        bad.push(x);
        let mut x = p.clone();
        x.metric.basis_id = "another".into();
        bad.push(x);
        let mut x = p.clone();
        x.baseline.normalization_id = "another".into();
        bad.push(x);
        let mut x = p.clone();
        x.operator.entries[1] = ExactBounds::point("0");
        bad.push(x);
        let mut x = p.clone();
        x.metric = form("mass", &["1", "2", "2", "1"]);
        bad.push(x);
        let mut x = p.clone();
        x.metric.entries[0] = interval("-1", "1");
        bad.push(x);
        let mut x = p.clone();
        x.baseline = vector("base", &["0", "0"]);
        bad.push(x);
        let mut x = p.clone();
        x.corrections = vec![vector("cancel", &["-1", "0"])];
        bad.push(x);
        let mut x = p.clone();
        x.corrections[1].label = "first".into();
        bad.push(x);
        let mut x = p.clone();
        x.operator.entries.pop();
        bad.push(x);
        let mut x = p.clone();
        x.operator.dimension = usize::MAX;
        bad.push(x);
        let mut x = p.clone();
        x.baseline.coefficients[0] = interval("-1", "1");
        x.corrections.clear();
        bad.push(x);
        for x in bad {
            assert!(analyze_trial_energy(
                &x,
                &TrialEnergyOptions::default(),
                &CancellationToken::new()
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_nonfinite_inputs_resource_exhaustion_and_cancellation() {
        let p = problem();
        for s in [
            "NaN",
            "inf",
            "1e999999999",
            "1e20000",
            "1/0",
            "1/-2",
            " 1",
            "1/2/3",
            "",
        ] {
            let mut x = p.clone();
            x.operator.entries[0] = ExactBounds::point(s);
            assert!(
                analyze_trial_energy(
                    &x,
                    &TrialEnergyOptions::default(),
                    &CancellationToken::new()
                )
                .is_err(),
                "{s}"
            );
        }
        let mut x = p.clone();
        x.operator.entries[0] = interval("2", "1");
        assert!(analyze_trial_energy(
            &x,
            &TrialEnergyOptions::default(),
            &CancellationToken::new()
        )
        .is_err());
        for o in [
            TrialEnergyOptions {
                maximum_interval_operations: 1,
                ..Default::default()
            },
            TrialEnergyOptions {
                maximum_input_bytes: 1,
                ..Default::default()
            },
            TrialEnergyOptions {
                maximum_parts: 2,
                ..Default::default()
            },
            TrialEnergyOptions {
                maximum_dimension: 1,
                ..Default::default()
            },
        ] {
            assert!(analyze_trial_energy(&p, &o, &CancellationToken::new()).is_err());
        }
        x = problem();
        x.baseline = vector("base", &["1e10", "1e10"]);
        x.operator = form("matrix", &["1e10", "0", "0", "1"]);
        assert!(analyze_trial_energy(
            &x,
            &TrialEnergyOptions {
                maximum_rational_bits: 64,
                ..Default::default()
            },
            &CancellationToken::new()
        )
        .is_err());
        let token = CancellationToken::with_wall_time_limit(std::time::Duration::ZERO);
        assert!(analyze_trial_energy(&p, &TrialEnergyOptions::default(), &token).is_err());
    }

    #[test]
    fn replay_rejects_changed_inputs_bounds_cross_terms_and_scope() {
        let p = problem();
        let r = run(&p);
        let o = TrialEnergyOptions::default();
        let token = CancellationToken::new();
        let mut bad = r.clone();
        bad.energy.contributions[4].value = ExactBounds::point("0");
        assert!(verify_trial_energy(&p, &o, &bad, &token).is_err());
        bad = r.clone();
        bad.corrected.rayleigh_absolute_error_bound = "1".into();
        assert!(verify_trial_energy(&p, &o, &bad, &token).is_err());
        bad = r.clone();
        bad.scope = "continuum".into();
        assert!(verify_trial_energy(&p, &o, &bad, &token).is_err());
        let mut changed = p.clone();
        changed.operator.entries[0] = ExactBounds::point("4");
        assert!(verify_trial_energy(&changed, &o, &r, &token).is_err());
        changed = p.clone();
        changed.corrections[0].source_id = "different".into();
        assert!(verify_trial_energy(&changed, &o, &r, &token).is_err());
        let mut o2 = o.clone();
        o2.maximum_interval_operations += 1;
        assert!(verify_trial_energy(&p, &o2, &r, &token).is_err());
    }
}

#[cfg(feature = "hp-reference")]
mod convergence_contracts {
    use rug::Rational;
    use xc_core::CancellationToken;
    use xc_solver::convergence::*;
    fn pt(s: &str) -> ExactBounds {
        ExactBounds::point(s)
    }
    fn q(s: &str) -> Rational {
        Rational::from_str_radix(s, 10).unwrap()
    }
    fn run(p: Problem) -> Output {
        let o = Options::default();
        let t = CancellationToken::new();
        let r = analyze(&p, &o, &t).unwrap();
        verify(&p, &o, &r, &t).unwrap();
        let wire = serde_json::to_vec(&r).unwrap();
        let decoded: Report = serde_json::from_slice(&wire).unwrap();
        assert_eq!(r, decoded);
        let mut bad = r.clone();
        bad.request_sha256 = "wrong".into();
        assert!(verify(&p, &o, &bad, &t).is_err());
        r.output
    }
    fn matrix(n: usize, entries: &[&str]) -> TrialForm {
        TrialForm {
            source_id: "matrix".into(),
            basis_id: "orthonormal".into(),
            normalization_id: "euclidean".into(),
            dimension: n,
            entries: entries.iter().map(|x| pt(x)).collect(),
        }
    }
    fn root_case() -> RootProblem {
        RootProblem {
            source_id: "rational".into(),
            branch_id: "positive".into(),
            requested_index: 1,
            weights: vec![pt("1"), pt("1")],
            poles: vec![pt("-1"), pt("1")],
            bracket: ExactBounds {
                lower: "-1/4".into(),
                upper: "1/4".into(),
            },
            center: "0".into(),
            value_error_upper: "0".into(),
            derivative_error_upper: "0".into(),
            taylor_order: 8,
        }
    }
    #[test]
    fn input_byte_cap_accepts_large_declared_limits_up_to_the_maximum() {
        let t = CancellationToken::new();
        let problem = Problem::Root(root_case());
        let at = |bytes| Options {
            maximum_input_bytes: bytes,
            ..Default::default()
        };
        assert!(analyze(&problem, &at(256 << 20), &t).is_ok());
        assert!(analyze(&problem, &at(MAXIMUM_INPUT_BYTES), &t).is_ok());
        assert!(analyze(&problem, &at(MAXIMUM_INPUT_BYTES + 1), &t).is_err());
        let rejected = analyze(&problem, &at(1), &t).unwrap_err().to_string();
        assert!(
            rejected.contains("convergence: input byte limit"),
            "{rejected}"
        );
    }
    #[test]
    fn root_certificate_and_c1_perturbation() {
        let p = root_case();
        let Output::Root(r) = run(Problem::Root(p.clone())) else {
            panic!()
        };
        assert_eq!(r.status, RootStatus::CertifiedLocal);
        assert_eq!(r.enclosure, Some(pt("0")));
        assert!(!r.global_ordinal_certified);
        let mut p = p;
        p.value_error_upper = "1/100".into();
        p.derivative_error_upper = "1/10".into();
        let Output::Root(r) = run(Problem::Root(p)) else {
            panic!()
        };
        assert_eq!(r.status, RootStatus::CertifiedLocal);
        let b = r.enclosure.unwrap();
        assert!(q(&b.lower) < 0 && q(&b.upper) > 0);
        // Both roots of 2*x/(x*x-1)+e=0 near zero are enclosed for e=+-0.01;
        // rational endpoint signs at +-1/200 certify their displacement.
        assert!(q(&b.upper) >= q("1/201"));
    }
    #[test]
    fn root_refusals_and_scale_invariance() {
        let mut p = root_case();
        p.poles[0] = pt("0");
        let Output::Root(r) = run(Problem::Root(p)) else {
            panic!()
        };
        assert_eq!(r.status, RootStatus::PoleContact);
        let mut p = root_case();
        p.derivative_error_upper = "100".into();
        let Output::Root(r) = run(Problem::Root(p)) else {
            panic!()
        };
        assert_eq!(r.status, RootStatus::UnresolvedDerivative);
        let mut p = root_case();
        p.value_error_upper = "100".into();
        let Output::Root(r) = run(Problem::Root(p)) else {
            panic!()
        };
        assert_eq!(r.status, RootStatus::NoSignChange);
        for s in ["-3", "1/1000000", "1e-1000"] {
            let mut p = root_case();
            p.weights = vec![pt(s), pt(s)];
            let Output::Root(r) = run(Problem::Root(p)) else {
                panic!()
            };
            assert_eq!(r.enclosure, Some(pt("0")));
        }
    }
    fn directional_case() -> DirectionalProblem {
        DirectionalProblem {
            matrix: matrix(2, &["2", "0", "0", "4"]),
            rhs: vec![pt("1"), pt("2")],
            approximate_solution: vec![pt("0"), pt("0")],
            functional: vec![pt("1"), pt("-1")],
            approximate_dual: vec![pt("1/2"), pt("-1/4")],
            coercivity_lower: "1".into(),
        }
    }
    #[test]
    fn signed_dual_cancellation_and_recomputed_defect() {
        let p = directional_case();
        let Output::Directional(r) = run(Problem::Directional(p.clone())) else {
            panic!()
        };
        assert_eq!(r.signed_correction, pt("0"));
        assert_eq!(r.exact_functional_enclosure, pt("0"));
        assert_eq!(r.correction_error_upper, "0");
        let mut p = p;
        p.approximate_dual[0] = pt("0");
        let Output::Directional(r) = run(Problem::Directional(p)) else {
            panic!()
        };
        assert!(q(&r.correction_error_upper) > 0);
        assert!(
            q(&r.exact_functional_enclosure.lower) <= 0
                && q(&r.exact_functional_enclosure.upper) >= 0
        );
    }
    #[test]
    fn directional_interval_matrix_and_bad_coercivity() {
        let mut p = directional_case();
        p.matrix.entries[0] = ExactBounds {
            lower: "2".into(),
            upper: "3".into(),
        };
        let Output::Directional(r) = run(Problem::Directional(p.clone())) else {
            panic!()
        };
        assert!(q(&r.exact_functional_enclosure.lower) <= q("-1/6"));
        assert!(q(&r.exact_functional_enclosure.upper) >= 0);
        p.coercivity_lower = "4".into();
        assert!(analyze(
            &Problem::Directional(p),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
    }
    #[test]
    fn finite_schur_tail_is_not_an_infinite_tail() {
        let p = TailBlockProblem {
            matrix: matrix(2, &["0", "2", "2", "5"]),
            retained_dimension: 1,
            shift: pt("1"),
            gap_lower: "3".into(),
        };
        let Output::TailBlock(r) = run(Problem::TailBlock(p.clone())) else {
            panic!()
        };
        assert_eq!(r.schur_correction_norm_upper, "4/3");
        assert_eq!(r.omitted_dimension, 1);
        let mut p = p;
        p.gap_lower = "5".into();
        assert!(analyze(
            &Problem::TailBlock(p),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
    }
    fn budget_case() -> BudgetProblem {
        BudgetProblem {
            branch_id: "positive".into(),
            target_absolute_error: "1/50".into(),
            candidates: vec![
                BudgetCandidate {
                    modes: 10,
                    precision_bits: 128,
                    root: root_case(),
                },
                BudgetCandidate {
                    modes: 100,
                    precision_bits: 256,
                    root: root_case(),
                },
            ],
            truncation: Some(TailMajorant {
                branch_id: "positive".into(),
                proof_id: "hypothesis".into(),
                hypotheses: vec!["term bound holds for every omitted index".into()],
                minimum_modes: 1,
                scale: "1".into(),
                offset: "0".into(),
                power: 2,
            }),
            cutoff: Some(DeclaredCutoffBound {
                branch_id: "positive".into(),
                proof_id: "cutoff".into(),
                hypotheses: vec!["selected branches match".into()],
                absolute_error_upper: "0".into(),
            }),
            reference_assisted: false,
        }
    }
    #[test]
    fn planner_selects_sufficient_candidate_and_keeps_unknowns() {
        let p = budget_case();
        let Output::Budget(r) = run(Problem::Budget(p.clone())) else {
            panic!()
        };
        assert_eq!(r.selected_candidate, Some(1));
        assert_eq!(r.rows[0].action, BudgetAction::IncreaseModes);
        assert_eq!(r.rows[1].total_error_upper.as_deref(), Some("1/100"));
        let mut p = p;
        p.truncation = None;
        let Output::Budget(r) = run(Problem::Budget(p.clone())) else {
            panic!()
        };
        assert_eq!(r.selected_candidate, None);
        assert_eq!(r.rows[1].action, BudgetAction::MissingTruncationBound);
        assert_eq!(r.rows[1].total_error_upper, None);
        p = budget_case();
        p.cutoff = None;
        let Output::Budget(r) = run(Problem::Budget(p)) else {
            panic!()
        };
        assert_eq!(r.selected_candidate, None);
        assert_eq!(r.rows[0].action, BudgetAction::MissingCutoffBound);
    }
    #[test]
    fn planner_rejects_wrong_branch_domain_and_duplicate_settings() {
        for kind in 0..3 {
            let mut p = budget_case();
            match kind {
                0 => p.truncation.as_mut().unwrap().branch_id = "other".into(),
                1 => p.truncation.as_mut().unwrap().minimum_modes = 11,
                _ => p.candidates[1] = p.candidates[0].clone(),
            };
            assert!(analyze(
                &Problem::Budget(p),
                &Options::default(),
                &CancellationToken::new()
            )
            .is_err());
        }
    }
    #[test]
    fn normalization_requires_a_trace_error_as_well_as_a_floor() {
        let p = NormalizationProblem {
            source_id: "vectors".into(),
            raw_norm_error_upper: "1/10".into(),
            reference_norm_upper: "2".into(),
            source_normalizer: pt("2"),
            reference_normalizer: pt("1"),
            normalizer_difference_upper: "1".into(),
        };
        let Output::Normalization(r) = run(Problem::Normalization(p.clone())) else {
            panic!()
        };
        assert_eq!(r.normalized_norm_error_upper, "21/20");
        let mut p = p;
        p.normalizer_difference_upper = "0".into();
        assert!(analyze(
            &Problem::Normalization(p.clone()),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
        p.source_normalizer = ExactBounds {
            lower: "-1".into(),
            upper: "1".into(),
        };
        assert!(analyze(
            &Problem::Normalization(p),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
    }
    fn profile_case() -> ProfileProblem {
        ProfileProblem {
            source_id: "profile".into(),
            validation_grid_id: "validation".into(),
            training_grid_id: Some("training".into()),
            cells: vec![ProfileCell {
                left: "0".into(),
                right: "2".into(),
                left_residual: pt("-1"),
                right_residual: pt("1"),
                weight: pt("3"),
                second_derivative_upper: "0".into(),
                additional_sup_error_upper: "0".into(),
            }],
        }
    }
    #[test]
    fn continuous_l1_integrates_sign_crossing_and_bounds_curvature() {
        let p = profile_case();
        let Output::Profile(r) = run(Problem::Profile(p.clone())) else {
            panic!()
        };
        assert_eq!(r.weighted_l1, pt("3"));
        assert!(r.independent_validation_grid);
        let mut p = p;
        p.cells[0].second_derivative_upper = "1".into();
        let Output::Profile(r) = run(Problem::Profile(p.clone())) else {
            panic!()
        };
        assert_eq!(
            r.weighted_l1,
            ExactBounds {
                lower: "0".into(),
                upper: "6".into()
            }
        );
        p.cells.push(p.cells[0].clone());
        assert!(analyze(
            &Problem::Profile(p),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
    }
    fn cluster_case() -> ClusterProblem {
        ClusterProblem {
            matrix: matrix(3, &["-2", "0", "0", "0", "-1", "0", "0", "0", "4"]),
            spectral_window: ExactBounds {
                lower: "-3".into(),
                upper: "0".into(),
            },
            columns: vec![
                vec![pt("1"), pt("0"), pt("0")],
                vec![pt("0"), pt("2"), pt("0")],
            ],
            shifts: vec!["-2".into(), "-1".into()],
            gram_lower: "1/2".into(),
            previous_columns: None,
        }
    }
    #[test]
    fn cluster_count_ground_selection_and_nonunit_basis() {
        let p = cluster_case();
        let Output::Cluster(r) = run(Problem::Cluster(p.clone())) else {
            panic!()
        };
        assert_eq!(r.eigenvalues_in_window, 2);
        assert!(r.contains_ground);
        assert!(!r.simple_ground);
        assert_eq!(r.projector_frobenius_squared_upper, "0");
        let mut p = p;
        p.spectral_window = ExactBounds {
            lower: "-3/2".into(),
            upper: "0".into(),
        };
        p.columns.remove(0);
        p.shifts.remove(0);
        let Output::Cluster(r) = run(Problem::Cluster(p)) else {
            panic!()
        };
        assert_eq!(r.eigenvalues_below, 1);
        assert!(!r.contains_ground);
        assert!(!r.simple_ground);
    }
    #[test]
    fn cluster_residual_bound_and_tracking_rotated_basis() {
        let mut p = cluster_case();
        p.columns = vec![
            vec![pt("3/5"), pt("0"), pt("4/5")],
            vec![pt("0"), pt("1"), pt("0")],
        ];
        p.previous_columns = Some(vec![
            vec![pt("1"), pt("0"), pt("0")],
            vec![pt("0"), pt("1"), pt("0")],
        ]);
        let Output::Cluster(r) = run(Problem::Cluster(p.clone())) else {
            panic!()
        };
        assert!(q(&r.projector_frobenius_squared_upper) >= q("32/25"));
        assert!(
            q(r.previous_trial_projector_frobenius_squared_upper
                .as_ref()
                .unwrap())
                >= q("32/25")
        );
        p.columns[1] = p.columns[0].clone();
        assert!(analyze(
            &Problem::Cluster(p),
            &Options::default(),
            &CancellationToken::new()
        )
        .is_err());
    }
    #[test]
    fn work_limits_cancellation_and_untrusted_serialization() {
        let p = Problem::Root(root_case());
        let mut o = Options {
            maximum_interval_operations: 0,
            ..Options::default()
        };
        assert!(analyze(&p, &o, &CancellationToken::new()).is_err());
        o = Options::default();
        o.maximum_input_bytes = 1;
        assert!(analyze(&p, &o, &CancellationToken::new()).is_err());
        let token = CancellationToken::new();
        token.cancel(xc_core::CancellationReason::UserRequested);
        assert!(analyze(&p, &Options::default(), &token).is_err());
        let mut wire = serde_json::to_value(p).unwrap();
        wire["input"]["global_ordinal_certified"] = true.into();
        assert!(serde_json::from_value::<Problem>(wire).is_err());
    }
    #[test]
    fn dense_directed_certificate_encloses_exact_inverse_and_rejects_bad_gap() {
        use rug::ops::Pow;
        // A=I+11^T. Sherman-Morrison gives A^-1 e0=e0-1/(n+1).
        // Dense interval LDL previously incurred exponential rational growth.
        let n = 41;
        let entries = (0..n * n)
            .map(|k| {
                let center = Rational::from(if k / n == k % n { 2 } else { 1 });
                let radius = Rational::from((1, rug::Integer::from(10).pow(30)));
                ExactBounds {
                    lower: Rational::from(&center - &radius).to_string(),
                    upper: Rational::from(&center + &radius).to_string(),
                }
            })
            .collect();
        let mut unit = vec![pt("0"); n];
        unit[0] = pt("1");
        let inverse: Vec<_> = (0..n)
            .map(|i| pt(&Rational::from((if i == 0 { n as i32 } else { -1 }, n + 1)).to_string()))
            .collect();
        let mut p = DirectionalProblem {
            matrix: TrialForm {
                source_id: "I+ones".into(),
                basis_id: "orthonormal".into(),
                normalization_id: "euclidean".into(),
                dimension: n,
                entries,
            },
            rhs: unit.clone(),
            approximate_solution: inverse.clone(),
            functional: unit,
            approximate_dual: inverse,
            coercivity_lower: "1/2".into(),
        };
        let o = Options {
            working_precision_bits: 128,
            maximum_rational_bits: 1024,
            ..Default::default()
        };
        let report = analyze(
            &Problem::Directional(p.clone()),
            &o,
            &CancellationToken::new(),
        )
        .unwrap();
        let Output::Directional(r) = report.output else {
            panic!()
        };
        let answer = Rational::from((n, n + 1));
        assert!(
            q(&r.exact_functional_enclosure.lower) <= answer
                && q(&r.exact_functional_enclosure.upper) >= answer
        );
        assert!(
            q(&r.exact_functional_enclosure.upper) - q(&r.exact_functional_enclosure.lower)
                < q("1/100000000000000000000")
        );
        p.coercivity_lower = "2".into();
        assert!(analyze(&Problem::Directional(p), &o, &CancellationToken::new()).is_err());
    }
    #[test]
    fn directed_inertia_never_rounds_a_zero_crossing_into_a_certificate() {
        let mut p = directional_case();
        p.matrix.entries[0] = ExactBounds {
            lower: "999999999999999999999999999999/1000000000000000000000000000000".into(),
            upper: "1000000000000000000000000000001/1000000000000000000000000000000".into(),
        };
        p.coercivity_lower = "1".into();
        let o = Options {
            working_precision_bits: 64,
            ..Default::default()
        };
        assert!(analyze(&Problem::Directional(p), &o, &CancellationToken::new()).is_err());
    }
}
