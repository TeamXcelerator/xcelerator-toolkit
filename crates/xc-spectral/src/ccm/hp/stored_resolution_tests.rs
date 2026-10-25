use super::*;
use rug::{Integer, Rational};

// Frozen stored-resolution contracts, with the new default dense-LU arithmetic
// isolated explicitly from the released solver identity. Literal old arithmetic
// names and the record shape deliberately do not use current version constants.
#[test]
fn default_dense_lu_isolates_released_eigenpair_identity_for_both_solvers() {
    let tau =
        super::tests::conditioning_test_manifest("ccm_tau_matrix", "released-eigenpair-source");
    let params = CcmParams::from_lambda_sq_integer(13, 1);
    let mut cfg = HighPrecConfig::for_decimal_digits(20);
    cfg.precision_bits = 128;
    cfg.inverse_iter_steps = 2;
    let common = serde_json::json!({
        "lambda_squared": "13", "n_modes": 1, "precision_bits": 128,
        "scalar_backend": "rug_mpfr", "force_even": true,
        "normalization": "sum_xi_equals_sqrt_log_lambda_squared",
        "source_identity_arithmetic": "validated_tau_dependency_and_exact_log_source_v2",
        "tau_source_identity": DependencyRef { key: tau.key.clone(), content_digest: tau.content_digest.clone(), required_quality: CacheQuality::Validated },
        "l2_normalization_arithmetic": "power_two_scaled_max_precision_l2_v2",
        "parity_basis_arithmetic": "directed_stored_point_orthonormal_parity_v2",
        "state_normalization_arithmetic": "directed_scaled_exact_sum_sqrt_l_state_v2",
        "tau_residual_arithmetic": "directed_scaled_stored_infinity_norm_v2",
        "ground_index_validation": "directed_stored_source_stable_ground_index_and_rounding_scale_gap_v3",
        "inverse_iteration_semantics": "dense-inverse-iteration-mixed-parity-residual-checked-wrapper-v3"
    });
    for solver in [
        CcmEigenstateSolver::LegacyInverseIteration,
        CcmEigenstateSolver::ShiftInvertKrylov,
    ] {
        cfg.eigenstate_solver = solver;
        cfg.ground_index_policy = CcmGroundIndexPolicy::Require;
        let mut parameters = common.clone();
        let (version, algorithm, suffix) = match solver {
            CcmEigenstateSolver::LegacyInverseIteration => {
                parameters["inverse_iteration_step_limit"] = serde_json::json!(2);
                ("ccm-smallest-weil-eigenpair-stored-resolution-v4",
                 "dense_inverse_iteration_with_half_precision_basin_shifted_rescue_and_full_tau_residual_gate_v1", "")
            }
            CcmEigenstateSolver::ShiftInvertKrylov => {
                for (key, value) in serde_json::json!({
                    "eigenstate_route": "shift_invert_krylov",
                    "krylov_subspace_dimension": 32, "krylov_maximum_restarts": 64,
                    "krylov_guard_eigenpairs": 2, "polishing_inverse_iteration_step_limit": 2,
                    "polishing_residual_policy": "directed_nonworsening_16_dim_ulp_matrix_floor_budgeted_phases_v2",
                    "krylov_working_precision_bits": 320,
                    "krylov_precision_policy": "same_stored_sector_double_source_plus_64_capped_one_million_v1",
                    "ccm_source_boundary_count": "budgeted_stable_outward_stored_source_inertia_v1",
                    "source_boundary_count_semantics": "hp_krylov_source_bound_budgeted_boundary_inertia_v5",
                    "boundary_cluster_resolution_policy": "stored_state_residual_and_one_norm_scale_gap_v1",
                    "borrowed_dense_action": "exact_products_guarded_sum_v2"
                }).as_object().unwrap() {
                    parameters[key] = value.clone();
                }
                (
                    "ccm-smallest-weil-eigenpair-shift-invert-krylov-budgeted-count-resolution-v7",
                    "ccm_even_zero_shift_krylov_budgeted_count_polish_resolution_v7",
                    "/shift_invert_krylov",
                )
            }
            CcmEigenstateSolver::Auto => unreachable!(),
        };
        let released = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "ccm_weil_eigenpair".into(),
            mathematical_semantics_version: version.into(),
            resolved_mathematical_parameters: parameters,
            normalization: Some("sum_xi_equals_sqrt_log_lambda_squared".into()),
            target: Some("smallest_weil_form_eigenpair".into()),
            subspace: Some("even".into()),
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: Some(algorithm.into()),
        };
        let (required, logical) = weil_eigenpair_cache_identity(&params, &cfg, &tau).unwrap();
        assert_eq!(logical, format!("ccm/weil-eigenpair/13/1/128/even{suffix}"));
        let mut without_dense_lu = required.clone();
        let dense_lu = without_dense_lu
            .resolved_mathematical_parameters
            .as_object_mut()
            .unwrap()
            .remove("dense_lu_arithmetic")
            .unwrap();
        assert_eq!(
            dense_lu,
            serde_json::json!("dense_lu_partial_pivot_dimension_panel512_exact_schur_v1")
        );
        assert_eq!(
            serde_json::to_vec(&without_dense_lu).unwrap(),
            serde_json::to_vec(&released).unwrap()
        );
        assert_ne!(required.digest().unwrap(), released.digest().unwrap());

        cfg.ground_index_policy = CcmGroundIndexPolicy::RecordOutcome;
        let (recorded, recorded_logical) =
            weil_eigenpair_cache_identity(&params, &cfg, &tau).unwrap();
        let mut old_recorded = released.clone();
        old_recorded.resolved_mathematical_parameters["ground_index_admission"] =
            serde_json::json!("record_outcome_v1");
        let mut corrected_recorded = old_recorded.clone();
        corrected_recorded.resolved_mathematical_parameters["stored_resolution_arithmetic"] =
            serde_json::json!("stored_state_residual_and_admitted_gap_v2");
        corrected_recorded.resolved_mathematical_parameters["dense_lu_arithmetic"] = dense_lu;
        assert_eq!(recorded, corrected_recorded);
        assert_ne!(recorded.digest().unwrap(), old_recorded.digest().unwrap());
        assert_ne!(recorded.digest().unwrap(), required.digest().unwrap());
        assert_eq!(recorded_logical, logical);
    }
}

#[test]
fn released_required_resolution_preserves_serialized_record_bytes() {
    #[derive(Serialize)]
    struct ReleasedResolution {
        arithmetic: &'static str,
        eigenvalue_resolution: &'static str,
        eigenvalue_lower: String,
        eigenvalue_upper: String,
        residual_upper: String,
        matrix_rounding_scale_upper: String,
        coefficient_sum_condition_upper: Option<String>,
        selected_gap_lower: Option<String>,
        residual_angle_upper: String,
        assembly_error_bound: Option<String>,
    }
    for p in [64, 128, 256] {
        for dimension in [1, 3] {
            let mut tau = vec![Float::with_val(p, 0); dimension * dimension];
            let mut xi = vec![Float::with_val(p, 0); dimension];
            for i in 0..dimension {
                tau[i * dimension + i] = Float::with_val(p, 2);
            }
            tau[(dimension / 2) * dimension + dimension / 2] = Float::with_val(p, 1);
            xi[dimension / 2] = Float::with_val(p, 1);
            let lambda = Float::with_val(p, 1);
            let parity = CcmParityPolicy::EvenSector;
            let outcome =
                ground_index::outcome_with_residual(&tau, &xi, &lambda, p, parity, None).unwrap();
            assert_eq!(outcome, ground_index::Outcome::Proven);
            let record = stored_resolution::bounds(&tau, &xi, &lambda, p, parity)
                .unwrap()
                .admit(&outcome, CcmGroundIndexPolicy::Require);
            let work = p + 64;
            let up = |x: &Float| x.to_string_radix_round(10, None, rug::float::Round::Up);
            let down = |x: &Float| x.to_string_radix_round(10, None, rug::float::Round::Down);
            let one = Float::with_val(work, 1);
            let zero = Float::with_val(work, 0);
            // Exact diagonal oracle: residual zero, floor n*max|A|*2^-p,
            // and neighbor gap floor + 2^-(p+8), with no eigensolver.
            let floor = Float::with_val(work, if dimension == 1 { 1 } else { 6 }) >> p;
            let gap = Float::with_val(work, Rational::from((1537, 256))) >> p;
            let released = ReleasedResolution {
                arithmetic: "stored_state_residual_and_one_norm_scale_gap_v1",
                eigenvalue_resolution: "resolved",
                eigenvalue_lower: down(&one),
                eigenvalue_upper: up(&one),
                residual_upper: up(&zero),
                matrix_rounding_scale_upper: up(&floor),
                coefficient_sum_condition_upper: Some(up(&one)),
                selected_gap_lower: (dimension != 1).then(|| down(&gap)),
                residual_angle_upper: up(&zero),
                assembly_error_bound: None,
            };
            let frozen_bytes = serde_json::to_vec(&released).unwrap();
            assert_eq!(serde_json::to_vec(&record).unwrap(), frozen_bytes);
            let decoded: CcmStoredStateResolution = serde_json::from_slice(&frozen_bytes).unwrap();
            assert_eq!(decoded, record);

            let recorded = stored_resolution::bounds(&tau, &xi, &lambda, p, parity)
                .unwrap()
                .admit(&outcome, CcmGroundIndexPolicy::RecordOutcome);
            assert_eq!(
                recorded.arithmetic,
                "stored_state_residual_and_admitted_gap_v2"
            );
            assert_eq!(recorded.selected_gap_lower, record.selected_gap_lower);
            assert_eq!(recorded.residual_angle_upper, record.residual_angle_upper);
            assert!(recorded.separation_unresolved_reason.is_none());
        }
    }
}

#[test]
fn exact_dyadic_gap_floor_threshold_and_scaling() {
    for p in [64u32, 128, 256] {
        for exponent in [-200i32, 0, 200] {
            for numerator in [7u32, 9, 17, 33] {
                let mut a = vec![Float::with_val(p, 0); 25];
                let scale = Float::with_val(p, 1) << exponent;
                let floor_exact =
                    Rational::from(5) * Rational::from((Integer::from(1), Integer::from(1) << p));
                let gap_unscaled = floor_exact.clone() * Rational::from((numerator, 8));
                let gap = Float::with_val(p, &gap_unscaled) << exponent;
                a[0] = scale.clone();
                a[24] = scale;
                a[6] = gap.clone();
                a[18] = gap;
                let mut v = vec![Float::with_val(p, 0); 5];
                v[2] = Float::with_val(p, 1);
                let zero = Float::with_val(p, 0);
                let actual = ground_index::validate(&a, &v, &zero, p, CcmParityPolicy::EvenSector);
                assert_eq!(
                    actual.is_ok(),
                    gap_unscaled > floor_exact,
                    "p={p} scale={exponent} ratio={numerator}/8: {actual:?}"
                );
                // The manufactured even pencil is diagonal with eigenvalues
                // exactly 0, gap, scale: this oracle uses no eigensolver.
                let record =
                    stored_resolution::resolved(&a, &v, &zero, p, CcmParityPolicy::EvenSector)
                        .unwrap();
                let bound = Float::with_val(
                    p + 64,
                    Float::parse(&record.matrix_rounding_scale_upper).unwrap(),
                );
                assert_eq!((bound >> exponent).to_rational().unwrap(), floor_exact);
            }
        }
    }
}

#[test]
fn eigenvalue_sign_floor_is_distinct_from_eigenvector_resolution() {
    let p = 128u32;
    for (multiple, expected) in [
        (0, StoredEigenvalueResolution::StoredSignUnresolved),
        (1, StoredEigenvalueResolution::BelowStorageFloor),
        (10, StoredEigenvalueResolution::Resolved),
    ] {
        let mut a = vec![Float::with_val(p, 0); 9];
        a[0] = Float::with_val(p, 1);
        a[8] = Float::with_val(p, 1);
        a[4] = Float::with_val(p, multiple) >> p;
        let v = vec![
            Float::with_val(p, 0),
            Float::with_val(p, 1),
            Float::with_val(p, 0),
        ];
        ground_index::validate(&a, &v, &a[4], p, CcmParityPolicy::EvenSector).unwrap();
        let record =
            stored_resolution::resolved(&a, &v, &a[4], p, CcmParityPolicy::EvenSector).unwrap();
        assert_eq!(record.eigenvalue_resolution, expected);
        assert_eq!(record.assembly_error_bound, None);
    }
}

#[test]
fn qr_agreement_accepts_normwise_roundoff_not_old_large_window() {
    let p = 128;
    let d = [1, 2, 3].map(|x| Float::with_val(p, x));
    let e = [Float::with_val(p, 0), Float::with_val(p, 0)];
    let radius = sector_gap_math::qr_agreement_allowance(&d, &e, p).unwrap();
    let exact_radius = Rational::from((Integer::from(72), Integer::from(1) << p));
    assert_eq!(radius.to_rational().unwrap(), exact_radius);
    let point = Float::with_val(p, 1);
    let close = Float::with_val(p, &point + (Float::with_val(p, 6) >> p));
    let distant = Float::with_val(p, &point + (Float::with_val(p, 1000) >> p));
    assert!(sector_gap_math::qr_point_agrees(
        &close, &point, &point, &radius
    ));
    assert!(!sector_gap_math::qr_point_agrees(
        &distant, &point, &point, &radius
    ));
    assert!(sector_gap_math::validate_complete_points(
        &d,
        &e,
        &[close, d[1].clone(), d[2].clone()],
        p
    )
    .is_ok());
    assert!(sector_gap_math::validate_complete_points(
        &d,
        &e,
        &[distant, d[1].clone(), d[2].clone()],
        p
    )
    .is_err());
}

#[test]
fn original_stored_gap_floor_counterexample_and_resolved_source() {
    let params = CcmParams::from_lambda_sq_integer(13, 20);
    for (p, admitted) in [(98, false), (131, true)] {
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.precision_bits = p;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        let result = run_inner(
            &params,
            &cfg,
            RootAcquisition::SourceOnly,
            CcmCacheRoute::Standalone,
        );
        if admitted {
            let source = result.unwrap();
            let record = source.stored_state_resolution.unwrap();
            assert!(record.selected_gap_lower.is_some());
            assert_eq!(record.assembly_error_bound, None);
        } else {
            let error = match result {
                Ok(_) => panic!("unresolved source must be rejected"),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains("stored precision resolution limit")
                    || error
                        .to_string()
                        .contains("ground state is unresolved or clustered"),
                "{error:#}"
            );
        }
    }
}

#[test]
fn cross_checked_original_n25_and_unresolved_n30_keep_sector_scope() {
    let mut cfg = HighPrecConfig::for_decimal_digits(30);
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    for n in [20usize, 25, 30] {
        let params = CcmParams::from_lambda_sq_integer(13, n);
        let result = analyze_sector_gap_inner(
            &params,
            &cfg,
            CcmSectorAnalysisOptions::cross_checked(2),
            None,
        );
        // The N30 source sector is retained with its unresolved conclusions
        // reported, not discarded; N20 and N25 resolve completely.
        let gap = result.unwrap();
        if n == 30 {
            assert!(
                !gap.unresolved.is_empty(),
                "N30 source sector interval remains unresolved"
            );
            assert!(
                gap.unresolved
                    .iter()
                    .all(|reason| reason.contains("CCM sector resolution limit:")),
                "{:?}",
                gap.unresolved
            );
        } else {
            assert!(gap.unresolved.is_empty(), "{:?}", gap.unresolved);
        }
    }
}

#[test]
fn polar_allowance_encloses_nonsymmetric_basis_exact_matrix_difference() {
    use crate::ccm::retained_evidence::finite_math::abs;
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    for p in [96u32, 128, 256] {
        for shift in [0u32, 7, 19] {
            let delta = Rational::from((Integer::from(1), Integer::from(1) << (p - 16 + shift)));
            let epsilon: Rational = delta.clone() / 4;
            let exact_a = [
                Rational::from(1),
                epsilon.clone(),
                epsilon,
                Rational::from(3),
            ];
            let q = [
                Rational::from(1),
                delta.clone(),
                Rational::from(0),
                Rational::from(1),
            ];
            let numerator = [
                Rational::from(2),
                delta.clone(),
                -delta.clone(),
                Rational::from(2),
            ];
            let denominator = Rational::from(4) + delta.clone().square();
            let mut difference = vec![Rational::from(0); 4];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            difference[2 * i + j] += numerator[2 * k + i].clone()
                                * &exact_a[2 * k + l]
                                * &numerator[2 * l + j]
                                / &denominator;
                        }
                    }
                }
            }
            difference[0] -= 1;
            difference[3] -= 3;
            let a: Vec<_> = exact_a.iter().map(|x| Float::with_val(p, x)).collect();
            let q: Vec<_> = q.iter().map(|x| Float::with_val(p, x)).collect();
            let d = [Float::with_val(p, 1), Float::with_val(p, 3)];
            let bound =
                sector_transform_validation::bounds(&a, &d, &[Float::with_val(p, 0)], &q, 2, p)
                    .unwrap();
            let w = p + 256;
            let x = I::from_rational(&difference[0], w);
            let y = I::from_rational(&difference[1], w);
            let z = I::from_rational(&difference[3], w);
            let disc = x
                .sub(&z)
                .square()
                .add(&y.square().mul(&I::from_i64(4, w)))
                .sqrt()
                .unwrap();
            for exact in [x.add(&z).sub(&disc), x.add(&z).add(&disc)] {
                let norm = abs(&exact.div(&I::from_i64(2, w)).unwrap()).unwrap();
                assert!(norm.upper() <= &bound.eigenvalue_allowance);
            }
        }
    }
}

#[test]
fn original_n40_source_allowance_admits_resolved_sector_minimum() {
    let params = CcmParams::from_lambda_sq_integer(13, 40);
    let mut cfg = HighPrecConfig::for_decimal_digits(40);
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let report =
        analyze_sector_gap_inner(&params, &cfg, CcmSectorAnalysisOptions::selected(2), None)
            .unwrap();
    assert!(report.even.eigenpairs[0].eigenvalue_lower > 0);
}

#[test]
fn full_first_positive_index_does_not_assume_even_or_ground() {
    for p in [96u32, 160] {
        for diagonal in [[-4, 2, 5], [-9, -2, 7], [1, 4, 9]] {
            let mut a = vec![Float::with_val(p, 0); 9];
            for i in 0..3 {
                a[3 * i + i] = Float::with_val(p, diagonal[i]);
            }
            let index = diagonal.iter().position(|x| *x > 0).unwrap();
            let pair = spectrum_accuracy::first_positive(&a, 3, p, 64).unwrap();
            assert_eq!(pair.index, index);
            assert_eq!(pair.eigenvalue, diagonal[index]);
            assert_eq!(pair.eigenvector[index].clone().abs(), 1);
            // A directed norm can enclose exact zero with a tiny positive
            // upper endpoint. Check that it bounds the independently formed
            // exact stored residual instead of requiring bitwise zero.
            let exact_residual_squared: Rational = (0..3)
                .map(|i| {
                    let term = (Rational::from(diagonal[i])
                        - pair.eigenvalue.to_rational().unwrap())
                        * pair.eigenvector[i].to_rational().unwrap();
                    term.clone() * term
                })
                .sum();
            let upper = pair.residual_upper_bound.to_rational().unwrap();
            assert!(exact_residual_squared <= upper.clone() * upper);
            assert!(pair.residual_upper_bound < Float::with_val(p, 1) >> (p - 8));
        }
    }
}

#[test]
fn full_spectrum_contains_exact_quadratic_stored_eigenvalues() {
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    for p in [96u32, 160] {
        // A nonsymmetric choice of eigenvectors: 2x2 block eigenvalues are
        // 3 +/- sqrt(2), with an independent isolated third diagonal value.
        let a = [2, 1, 0, 1, 4, 0, 0, 0, 9].map(|x| Float::with_val(p, x));
        let report = spectrum_accuracy::full_spectrum(&a, 3, p).unwrap();
        let w = p + 256;
        let root = I::from_i64(2, w).sqrt().unwrap();
        let expected = [
            I::from_i64(3, w).sub(&root),
            I::from_i64(3, w).add(&root),
            I::from_i64(9, w),
        ];
        for (bound, exact) in report.eigenvalue_bounds.iter().zip(expected) {
            assert!(bound.lower <= *exact.lower() && bound.upper >= *exact.upper());
            assert_eq!(bound.resolution, StoredEigenvalueResolution::Resolved);
        }
    }
}

#[test]
fn original_plunge_wrong_index_inputs_are_explicitly_unresolved() {
    for n in [20usize, 24] {
        let params = CcmParams::from_lambda_sq_integer(13, n);
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.precision_bits = 128;
        cfg.quad_points = 1;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let result = weil_plunge_cancellation_hp_inner(&params, &cfg);
        assert!(
            result.is_err(),
            "sub-floor plunge must not return a different eigenvalue at N={n}"
        );
    }
    // The independent stored-source indexed oracle is exercised above; this
    // separate cutoff is a valid public-pipeline control, not a doubled run.
    let params = CcmParams::from_lambda_sq_integer(7, 8);
    let mut cfg = HighPrecConfig::for_decimal_digits(40);
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let result = weil_plunge_cancellation_hp_inner(&params, &cfg).unwrap();
    assert!(result.stored_eigenvalue_lower > 0);
    assert!(
        result.stored_eigenvalue_lower <= result.eps_n
            && result.stored_eigenvalue_upper >= result.eps_n
    );
    assert_eq!(result.selected_algebraic_index, 0);
}

#[test]
fn explicit_krylov_admits_resolved_original_n20_and_n10_control() {
    use xc_cache::{
        ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
        FilesystemCacheStore,
    };
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0) {
                if !std::thread::panicking() {
                    panic!("remove isolated Krylov fixture: {error}");
                }
            }
        }
    }
    // `Owned` checks that the fixture created and can remove its cache; the
    // enclosing scratch guard, declared first, is removed after it.
    let scratch = xc_core::test_support::TestDir::new("r2-krylov");
    let dir = Owned(scratch.join("cache"));
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "workstation",
            &dir.0,
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
    let context = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["workstation".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let mut cfg = HighPrecConfig::for_decimal_digits(20);
    cfg.eigenstate_solver = CcmEigenstateSolver::ShiftInvertKrylov;
    cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    for n in [10, 20] {
        let params = CcmParams::from_lambda_sq_integer(13, n);
        let l = log_lambda_sq_hp(&params, cfg.precision_bits).unwrap();
        let tau = build_tau_hp_via_cache(&params, &l, &cfg, &context).unwrap();
        let cold = weil_eigenpair_via_cache_with_seed(
            &params, &cfg, &l, &tau.0, &tau.1, &context, None, None,
        )
        .unwrap();
        let warm = weil_eigenpair_via_cache_with_seed(
            &params, &cfg, &l, &tau.0, &tau.1, &context, None, None,
        )
        .unwrap();
        assert_eq!(cold.4, CcmEigenstateSolver::ShiftInvertKrylov);
        assert_eq!(cold.0, warm.0);
        assert_eq!(cold.1, warm.1);
        assert_eq!(cold.3.content_digest, warm.3.content_digest);
        ground_index::validate(
            &tau.0,
            &cold.1,
            &cold.0,
            cfg.precision_bits,
            CcmParityPolicy::EvenSector,
        )
        .unwrap();
    }
}

#[test]
fn read_only_source_preserves_explicit_solver_routes_without_creating_cache() {
    let scratch = xc_core::test_support::TestDir::new("r2-read-only-ccm");
    let root = scratch.join("cache");
    assert!(!root.exists());
    for solver in [
        CcmEigenstateSolver::LegacyInverseIteration,
        CcmEigenstateSolver::ShiftInvertKrylov,
    ] {
        let config = xc_cache::ManagedArtifactCacheConfig {
            profile: xc_cache::ManagedRunProfile::Normal,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            cache_root: root.clone(),
            staging_root: Some(root.join("staging")),
            publication_target: xc_core::PublicationTarget::None,
            repository_owner: "fixture".into(),
            remote_cache_mode: xc_cache::ManagedRemoteCacheMode::None,
            cache_mode: xc_cache::ArtifactExecutionCacheMode::PreferReuse,
            replace_existing_publication: false,
            execute_remote_mutations: false,
            output_validation: None,
        };
        let session = xc_cache::ManagedArtifactCacheSession::new_read_only(
            config,
            xc_core::ResourcePolicy::default(),
        )
        .unwrap();
        let context = session.context();
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.quad_points = 128;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        cfg.eigenstate_solver = solver;
        for n in [3, 4] {
            let params = CcmParams::from_lambda_sq_integer(13, n);
            let l = log_lambda_sq_hp(&params, cfg.precision_bits).unwrap();
            let tau = build_tau_hp_via_cache(&params, &l, &cfg, &context).unwrap();
            let cold = weil_eigenpair_via_cache_with_seed(
                &params, &cfg, &l, &tau.0, &tau.1, &context, None, None,
            )
            .unwrap();
            let warm = weil_eigenpair_via_cache_with_seed(
                &params, &cfg, &l, &tau.0, &tau.1, &context, None, None,
            )
            .unwrap();
            assert_eq!(cold.4, solver);
            assert_eq!(warm.4, solver);
            assert_eq!(cold.0, warm.0);
            assert_eq!(cold.1, warm.1);
            assert_eq!(cold.3.content_digest, warm.3.content_digest);
            ground_index::validate(
                &tau.0,
                &cold.1,
                &cold.0,
                cfg.precision_bits,
                CcmParityPolicy::EvenSector,
            )
            .unwrap();
            assert!(!root.exists());
        }
        assert!(session.finalize_publication_inventory().unwrap().is_none());
        assert!(!root.exists());
    }
}

#[test]
fn guarded_krylov_matches_exact_hadamard_spectrum_at_unchanged_tolerances() {
    let p = 131;
    let work = krylov_working_precision(4, 4, p).unwrap();
    assert!(krylov_working_precision(100_000, 4, p).is_err());
    assert_eq!(
        krylov_working_precision(4, 4, 1_000_000).unwrap(),
        1_000_000
    );
    let signs = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]];
    for (tiny_bits, next_bits, scale) in [(120u32, 90u32, 1), (110, 75, 2)] {
        // H/2 is exactly orthogonal. All entries of H diag(d) H^T/4
        // are exact dyadics at p bits, so its spectrum and vectors are an
        // algebraic oracle independent of the Krylov/LU implementation.
        let exact = [
            Float::with_val(p, 1) >> tiny_bits,
            Float::with_val(p, 1) >> next_bits,
            Float::with_val(p, scale),
            Float::with_val(p, 4 * scale),
        ];
        let mut matrix = vec![Float::with_val(p, 0); 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut entry = Rational::new();
                for k in 0..4 {
                    entry += exact[k].to_rational().unwrap() * (signs[i][k] * signs[j][k]);
                }
                entry /= 4;
                matrix[4 * i + j] = Float::with_val(p, &entry);
                assert_eq!(matrix[4 * i + j].to_rational().unwrap(), entry);
            }
        }
        let solve = |working| {
            let factors = xc_numerics::linalg::lu_factor_at_precision(&matrix, 4, working).unwrap();
            let operator = BorrowedDenseSymmetricHp {
                name: "manufactured-dyadic-sector",
                dimension: 4,
                entries: &matrix,
                precision_bits: working,
                inertia_maximum_bytes: 64 << 20,
            };
            let shifted = RetainedCcmLuShiftInvert {
                factors: &factors,
                dimension: 4,
                precision_bits: working,
                id: format!("manufactured-dyadic-sector-{working}"),
            };
            let tolerance = krylov_tolerance(p);
            xc_solver::ShiftInvertKrylovSolverHp
                .solve(
                    &operator,
                    &shifted,
                    &xc_solver::ShiftInvertKrylovConfigHp {
                        target: EigenTarget::SmallestMagnitude,
                        precision_bits: working,
                        requested_eigenpairs: 1,
                        guard_eigenpairs: 2,
                        maximum_subspace_dimension: 4,
                        maximum_restarts: 16,
                        minimum_restarts: 2,
                        maximum_projected_sweeps: 256,
                        absolute_residual_tolerance: tolerance.clone(),
                        scaled_backward_error_tolerance: tolerance.clone(),
                        ritz_value_stability_tolerance: tolerance,
                        boundary_cluster_tolerance: DecimalLiteral::new(
                            stored_resolution::matrix_rounding_scale(&matrix, 4, p)
                                .unwrap()
                                .to_string_radix_round(10, None, rug::float::Round::Up),
                        )
                        .unwrap(),
                    },
                )
                .unwrap()
        };
        assert_ne!(solve(p).status, ResultStatus::Converged);
        let report = solve(work);
        assert_eq!(report.status, ResultStatus::Converged);
        assert!(report.global_target_ordering_established);
        assert!(report.boundary_cluster.is_none());
        let relative_limit = Float::with_val(work, 1) >> 170;
        for (k, pair) in report.retained_eigenpairs.iter().enumerate() {
            let error = Float::with_val(work, &pair.eigenvalue - &exact[k]).abs();
            assert!(
                error <= Float::with_val(work, &exact[k] * &relative_limit),
                "exact eigenvalue oracle failed for retained position {k}: {error}"
            );
            let mut overlap = Float::with_val(work, 0);
            for (i, component) in pair.eigenvector.iter().enumerate() {
                overlap += Float::with_val(work, component * signs[i][k]) / 2;
            }
            assert!(Float::with_val(work, 1 - overlap.abs()).abs() <= relative_limit);
        }
    }
}

#[test]
fn current_root_keys_are_captured_from_real_producer() {
    let params = CcmParams::from_lambda_sq_integer(2, 3);
    let fixed = HighPrecConfig::for_decimal_digits(20);
    let adaptive = fixed.clone().with_adaptive_root_precision();
    let seeds = vec![Float::with_val(fixed.precision_bits, 1)];
    let source_digest = ContentDigest::sha256(b"manufactured C2 N3 exact stored secular source");
    let key = |cfg: &HighPrecConfig| {
        root_range_semantic_key(
            &params,
            cfg,
            1,
            &seeds,
            RootArtifactMode::Independent,
            None,
            RootWindowSemantics::strict_positive(1),
            Some(&source_digest),
        )
        .unwrap()
    };
    let fixed_key = key(&fixed);
    let adaptive_key = key(&adaptive);
    assert_ne!(fixed_key.digest().unwrap(), adaptive_key.digest().unwrap());
    println!(
        "R2_CURRENT_ROOT_KEYS={}",
        serde_json::json!({
            "fixed": fixed_key, "adaptive": adaptive_key,
        })
    );
}

#[test]
fn secular_source_rebinds_validation_identity_with_unchanged_state_bytes() {
    use crate::ccm::retained_evidence::{capture_root_window, RetainedRoots};
    use crate::ccm::state_geometry::RetainedState;
    use xc_cache::{
        ArtifactDraft, ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver,
        CacheStore, CacheVisibility, FilesystemCacheStore,
    };
    let directory = xc_core::test_support::TestDir::new("secular-validation-identity");
    let store = FilesystemCacheStore::new(
        "local",
        directory.to_path_buf(),
        true,
        CacheVisibility::Local,
    );
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "local",
            directory.to_path_buf(),
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
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let params = CcmParams::from_lambda_sq_integer(13, 2);
    let cfg = HighPrecConfig::for_decimal_digits(40).with_adaptive_root_precision();
    let xi = ["2.5", "-20", "36", "-20", "2.5"];
    let payload = serde_json::to_vec(&serde_json::json!({
        "schema_version": 3, "lambda_squared": "13", "n_modes": 2,
        "precision_bits": cfg.precision_bits, "force_even": true,
        "eigenvalue": "1", "eigenvector": xi,
    }))
    .unwrap();
    let draft = |key, dependencies, tags| ArtifactDraft {
        schema_version: 1,
        key,
        producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies,
        tags,
        provenance_digest: None,
    };
    let old = store
        .put(
            &draft(
                ArtifactKey::new("ccm_weil_eigenpair", "same-stored-state", b"validation-v2")
                    .unwrap(),
                vec![],
                BTreeMap::new(),
            ),
            &payload,
        )
        .unwrap();
    let current = store
        .put(
            &draft(
                ArtifactKey::new("ccm_weil_eigenpair", "same-stored-state", b"validation-v3")
                    .unwrap(),
                vec![],
                BTreeMap::new(),
            ),
            &payload,
        )
        .unwrap();
    assert_eq!(old.content_digest, current.content_digest);
    let old_source = resolve_secular_source_via_cache(&params, &cfg, &old, &cache).unwrap();
    let current_source = resolve_secular_source_via_cache(&params, &cfg, &current, &cache).unwrap();
    assert_ne!(
        old_source.key.parameters_digest,
        current_source.key.parameters_digest
    );
    assert_ne!(old_source.content_digest, current_source.content_digest);

    let l = log_lambda_sq_hp(&params, cfg.precision_bits).unwrap();
    let coefficients = xi
        .iter()
        .map(|s| Float::with_val(cfg.precision_bits, Float::parse(s).unwrap()))
        .collect::<Vec<_>>();
    // This exact rational point source has positive roots 3a and 4a,
    // where a=2*pi/log(13). No known zeta zeros are inputs to this fixture.
    let seed = Float::with_val(
        cfg.precision_bits,
        Float::with_val(cfg.precision_bits, rug::float::Constant::Pi) * 6,
    ) / &l;
    let roots_for = |source: &ArtifactManifest, context: &ArtifactCacheContext<'_>| {
        resolve_root_range_via_cache(
            &params,
            &cfg,
            CcmEigenstateSolver::Auto,
            &l,
            &coefficients,
            1,
            std::slice::from_ref(&seed),
            source,
            context,
            RootArtifactMode::Independent,
            None,
            RootWindowSemantics::strict_positive(1),
        )
        .unwrap()
    };
    let (old_roots, old_root_manifest, _) = roots_for(&old_source, &cache);
    let (current_roots, current_root_manifest, _) = roots_for(&current_source, &cache);
    assert!(current_roots.iter().all(EigenvalueResult::is_converged));
    assert_eq!(old_roots[0].value(), current_roots[0].value());
    assert_ne!(
        old_root_manifest.key.parameters_digest,
        current_root_manifest.key.parameters_digest
    );

    let warm = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["local".into()],
        mode: ArtifactExecutionCacheMode::RequireReuse,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let warm_source = resolve_secular_source_via_cache(&params, &cfg, &current, &warm).unwrap();
    assert_eq!(warm_source.key, current_source.key);
    let (_, warm_root, _) = roots_for(&warm_source, &warm);
    assert_eq!(warm_root.key, current_root_manifest.key);
    let state = RetainedState::from_payload(
        &current,
        &payload,
        std::slice::from_ref(&current.content_digest),
    )
    .unwrap();
    let source_bytes = resolver
        .resolve_exact(
            &warm_source.key,
            &warm_source.content_digest,
            CacheQuality::Validated,
            &policy,
        )
        .unwrap()
        .payload;
    let root_bytes = resolver
        .resolve_exact(
            &warm_root.key,
            &warm_root.content_digest,
            CacheQuality::Validated,
            &policy,
        )
        .unwrap()
        .payload;
    let retained = RetainedRoots::from_payload(
        &warm_root,
        &root_bytes,
        &warm_source,
        &source_bytes,
        &state,
        &[
            warm_root.content_digest.clone(),
            warm_source.content_digest.clone(),
        ],
    )
    .unwrap();
    let cold_capture = capture_root_window(&retained, &cache).unwrap();
    let warm_capture = capture_root_window(&retained, &warm).unwrap();
    assert_eq!(
        serde_json::to_value(cold_capture.value).unwrap(),
        serde_json::to_value(warm_capture.value).unwrap()
    );
    let old_source_bytes = resolver
        .resolve_exact(
            &old_source.key,
            &old_source.content_digest,
            CacheQuality::Validated,
            &policy,
        )
        .unwrap()
        .payload;
    let old_root_bytes = resolver
        .resolve_exact(
            &old_root_manifest.key,
            &old_root_manifest.content_digest,
            CacheQuality::Validated,
            &policy,
        )
        .unwrap()
        .payload;
    assert!(RetainedRoots::from_payload(
        &old_root_manifest,
        &old_root_bytes,
        &old_source,
        &old_source_bytes,
        &state,
        &[
            old_root_manifest.content_digest.clone(),
            old_source.content_digest.clone()
        ]
    )
    .is_err());

    let corrupt_dir = xc_core::test_support::TestDir::new("secular-wrong-validation");
    let corrupt_store = FilesystemCacheStore::new(
        "corrupt",
        corrupt_dir.to_path_buf(),
        true,
        CacheVisibility::Local,
    );
    let mut changed: serde_json::Value = serde_json::from_slice(&source_bytes).unwrap();
    changed["eigenpair_semantic_digest"] = serde_json::json!(old.key.parameters_digest.0);
    corrupt_store
        .put(
            &draft(
                current_source.key.clone(),
                current_source.dependencies.clone(),
                current_source.tags.clone(),
            ),
            &serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
    let corrupt_resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(corrupt_store),
    }]);
    let corrupt_cache = ArtifactCacheContext {
        resolver: Some(&corrupt_resolver),
        ..warm
    };
    assert!(resolve_secular_source_via_cache(&params, &cfg, &current, &corrupt_cache).is_err());
}

#[cfg(test)]
mod unresolved_separation_regression {
    use super::*;
    #[test]
    fn repeated_eigenvalue_never_retains_unproved_separation() {
        let p = 128;
        let tau = vec![Float::with_val(p, 0); 9];
        let xi = vec![
            Float::with_val(p, 0),
            Float::with_val(p, 1),
            Float::with_val(p, 0),
        ];
        let lambda = Float::with_val(p, 0);
        let bounds =
            stored_resolution::bounds(&tau, &xi, &lambda, p, CcmParityPolicy::EvenSector).unwrap();
        let outcome = ground_index::outcome_with_residual(
            &tau,
            &xi,
            &lambda,
            p,
            CcmParityPolicy::EvenSector,
            None,
        )
        .unwrap();
        assert!(matches!(outcome, ground_index::Outcome::NotEstablished(_)));
        let record = bounds.admit(&outcome, CcmGroundIndexPolicy::RecordOutcome);
        assert!(record.selected_gap_lower.is_none());
        assert!(record.residual_angle_upper.is_none());
        assert!(record.separation_unresolved_reason.is_some());

        let params = CcmParams::from_lambda_sq_integer(13, 1);
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.precision_bits = p;
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        cfg.ground_index_policy = CcmGroundIndexPolicy::RecordOutcome;
        let middle = log_lambda_sq_hp(&params, p).unwrap().sqrt();
        let normalized = vec![Float::with_val(p, 0), middle.clone(), Float::with_val(p, 0)];
        let mut artifact = PortableWeilEigenpair {
            stored_state_resolution: Some(
                stored_resolution::resolved(
                    &tau,
                    &normalized,
                    &lambda,
                    p,
                    cfg.effective_parity_policy(),
                )
                .unwrap(),
            ),
            schema_version: 4,
            lambda_squared: "13".into(),
            n_modes: 1,
            precision_bits: p,
            force_even: true,
            parity_policy: None,
            eigenstate_route: legacy_eigenstate_route_name(),
            eigenvalue: "0".into(),
            eigenvector: vec!["0".into(), middle.to_string(), "0".into()],
            inverse_iteration: PortableInverseIterationDiagnostics {
                configured_step_limit: cfg.inverse_iter_steps,
                unshifted_steps: 2,
                unshifted_converged: true,
                final_relative_rayleigh_change: Some("0".into()),
                shifted_refinement: "accepted".into(),
                final_relative_residual_norm: "0".into(),
            },
            shift_invert_krylov: None,
            ground_index: Some(CcmGroundIndexAdmission::from_outcome(
                ground_index::outcome_with_residual(
                    &tau,
                    &normalized,
                    &lambda,
                    p,
                    cfg.effective_parity_policy(),
                    None,
                )
                .unwrap(),
            )),
        };
        assert!(
            decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_ok(),
            "record-outcome decoder rejects fixture"
        );
        assert!(decode_weil_eigenpair_admitted(&artifact, &params, &cfg, &tau, false).is_ok());
        let original = artifact.stored_state_resolution.clone();
        // The previous outcome route published candidate separation despite
        // the failed proof. Its v1 bytes must not enter the corrected identity,
        // on either mathematical replay or previously admitted warm decoding.
        let old = artifact.stored_state_resolution.as_mut().unwrap();
        old.arithmetic = "stored_state_residual_and_one_norm_scale_gap_v1".into();
        old.selected_gap_lower = Some("1e-41".into());
        old.residual_angle_upper = Some("0".into());
        old.separation_unresolved_reason = None;
        for prove in [false, true] {
            let error =
                decode_weil_eigenpair_admitted(&artifact, &params, &cfg, &tau, prove).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("stored-state resolution failed source replay"),
                "{error}"
            );
        }
        artifact.stored_state_resolution = original.clone();
        artifact
            .stored_state_resolution
            .as_mut()
            .unwrap()
            .selected_gap_lower = Some("1e-41".into());
        for prove in [false, true] {
            assert!(decode_weil_eigenpair_admitted(&artifact, &params, &cfg, &tau, prove).is_err());
        }
        artifact.stored_state_resolution = original;
        let value = serde_json::to_value(&artifact.stored_state_resolution).unwrap();
        assert!(value["selected_gap_lower"].is_null());
        assert!(value["residual_angle_upper"].is_null());
        cfg.ground_index_policy = CcmGroundIndexPolicy::Require;
        artifact.ground_index = None;
        assert!(decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_err());
    }
}
