use super::*;
use rug::{Integer, Rational};

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
                    stored_resolution::bounds(&a, &v, &zero, p, CcmParityPolicy::EvenSector)
                        .unwrap()
                        .record;
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
        let record = stored_resolution::bounds(&a, &v, &a[4], p, CcmParityPolicy::EvenSector)
            .unwrap()
            .record;
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
        if n == 30 {
            let error = match result {
                Ok(_) => panic!("N30 source sector interval remains unresolved"),
                Err(error) => error,
            };
            assert!(is_sector_resolution_limit(&error), "{error:#}");
        } else {
            result.unwrap();
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
