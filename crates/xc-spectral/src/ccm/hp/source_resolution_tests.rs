use super::*;

#[test]
fn managed_auto_polishes_krylov_and_replays_the_retained_route() {
    use xc_cache::{
        ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
        FilesystemCacheStore,
    };
    // An explicitly created private directory avoids environment-dependent
    // standalone routing and any remote/publication side effects.
    let root_dir = xc_core::test_support::TestDir::new("confirmed-krylov");
    let root = root_dir.to_path_buf();
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "confirmed-local",
            &root,
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
        ordered_overlays: vec!["confirmed-local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let params = CcmParams::from_lambda_sq_integer(13, 10);
    let mut cfg = HighPrecConfig::for_decimal_digits(10);
    let solve = |config: &HighPrecConfig| {
        xc_numerics::hp_runtime::run_hp(|| {
            run_inner_retaining_source(
                &params,
                config,
                RootAcquisition::SourceOnly,
                CcmCacheRoute::Fabric(&context),
                None,
            )
        })
        .unwrap()
    };
    let (auto, source) = solve(&cfg);
    let manifest = source.eigenpair_manifest.as_ref().unwrap();
    let retained = resolver.resolve(&manifest.key, &policy).unwrap();
    let payload: PortableWeilEigenpair = serde_json::from_slice(&retained.payload).unwrap();
    assert_eq!(payload.eigenstate_route, "shift_invert_krylov");
    let diagnostics = payload.shift_invert_krylov.as_ref().unwrap();
    assert_eq!(
        diagnostics.algorithm_semantics,
        "ccm_even_zero_shift_krylov_guarded_lu_polish_resolution_v6"
    );
    assert!(diagnostics.polishing_candidate_adopted);
    assert_eq!(
        payload.inverse_iteration.configured_step_limit,
        cfg.inverse_iter_steps
    );
    let (warm, warm_source) = solve(&cfg);
    assert_eq!(warm_source.eigenpair_manifest, source.eigenpair_manifest);
    assert_eq!(warm.xi, auto.xi);
    assert_eq!(warm.weil_min_eigenvalue, auto.weil_min_eigenvalue);
    cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
    let (legacy, _) = solve(&cfg);
    let exact = |x: &Float| {
        let (mantissa, exponent) = x.to_integer_exp().unwrap();
        format!("{mantissa} {exponent}")
    };
    println!(
        "CCM_POLISH_ORACLE {}",
        serde_json::json!({
            "route_scope":"explicit_managed_auto_resolved_krylov_cold_and_warm",
            "precision_bits":cfg.precision_bits, "modes":params.n_modes,
            "tau":source.tau.iter().map(exact).collect::<Vec<_>>(),
            "auto":auto.xi.iter().map(exact).collect::<Vec<_>>(),
            "legacy":legacy.xi.iter().map(exact).collect::<Vec<_>>()
        })
    );
    assert!(auto
        .stored_eigenvalue_accuracy()
        .unwrap()
        .assembly_error_bound
        .is_none());
}

#[test]
fn documented_ground_index_budget_has_consistent_byte_admission() {
    let p = HighPrecConfig::for_decimal_digits(500).precision_bits;
    assert_eq!(p, 1725);
    ground_index::preflight(301, p, CcmParityPolicy::EvenSector).unwrap();
    assert!(ground_index::preflight(8193, 1_000_000, CcmParityPolicy::EvenSector).is_err());
}

#[test]
fn polynomially_separated_tiny_state_is_polished_against_exact_diagonal_oracle() {
    for p in [94, 128, 256] {
        let a0 = Float::with_val(p, 1) >> 80u32;
        let a1 = Float::with_val(p, 1) >> 40u32;
        let matrix = vec![
            a0.clone(),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            a1,
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 1),
        ];
        let factors = xc_numerics::linalg::lu_factor(&matrix, 3).unwrap();
        let vector = vec![
            Float::with_val(p, 1),
            Float::with_val(p, 1) >> 10u32,
            Float::with_val(p, 0),
        ];
        let polished =
            eigenstate_accuracy::polish(&matrix, &factors, &a0, &vector, p, 100).unwrap();
        assert_eq!(polished.value, a0);
        assert!(polished.vector[1].clone().abs() < (Float::with_val(p, 1) >> (p / 2)));
        assert_eq!(polished.diagnostics.configured_step_limit, 100);
        assert!(polished.diagnostics.unshifted_steps <= 100);
        if p == 256 {
            assert!(polished.diagnostics.unshifted_steps > 4);
            assert!(eigenstate_accuracy::polish(&matrix, &factors, &a0, &vector, p, 4).is_err());
        }
    }
}

#[test]
fn response_isolates_simple_ground_state_with_a_repeated_neighbor_cluster() {
    for p in [64, 128, 256] {
        let params = CcmParams::from_lambda_sq_integer(9, 2);
        let mut cfg = HighPrecConfig::for_decimal_digits(20);
        cfg.precision_bits = p;
        let t = SectorTridiagonalHp {
            diagonal: [1, 2, 2].map(|x| Float::with_val(p, x)).to_vec(),
            off_diagonal: vec![Float::with_val(p, 0); 2],
        };
        let policy = SectorIsolationPolicy::ResponseLowestAndNeighbor;
        let response = compute_sector_eigenvalues_with_policy(
            &t,
            3,
            2,
            CcmSectorEigenvalueRoute::Selected,
            p,
            policy,
        )
        .unwrap();
        assert_eq!(response.selected_enclosures[0].lower_count, 0);
        assert_eq!(response.selected_enclosures[0].upper_count, 1);
        assert_eq!(response.selected_enclosures[1].lower_count, 1);
        assert_eq!(response.selected_enclosures[1].upper_count, 3);
        // A response gap is valid, but these data must never be admitted as
        // individually simple sector eigenvectors by either cold or warm paths.
        assert!(
            compute_sector_eigenvalues(&t, 3, 2, CcmSectorEigenvalueRoute::Selected, p).is_err()
        );
        let portable = portable_sector_eigenvalues(&response, &params, &cfg, CcmParity::Even, 3, 2);
        assert!(decode_sector_eigenvalues(
            &portable,
            &params,
            &cfg,
            CcmParity::Even,
            3,
            2,
            CcmSectorEigenvalueRoute::Selected,
            &t
        )
        .is_err());
        let decoded = decode_sector_eigenvalues_with_policy(
            &portable,
            &params,
            &cfg,
            CcmParity::Even,
            3,
            2,
            CcmSectorEigenvalueRoute::Selected,
            &t,
            policy,
        )
        .unwrap();
        assert_eq!(decoded.selected_enclosures, response.selected_enclosures);
        for (index, lower, upper) in [(0, 0, 2), (1, 0, 3), (1, 1, 2), (1, 1, 4)] {
            let mut bad = portable.clone();
            bad.selected_enclosures[index].lower_count = lower;
            bad.selected_enclosures[index].upper_count = upper;
            assert!(decode_sector_eigenvalues_with_policy(
                &bad,
                &params,
                &cfg,
                CcmParity::Even,
                3,
                2,
                CcmSectorEigenvalueRoute::Selected,
                &t,
                policy
            )
            .is_err());
        }
        let prep = ResponseSpectralPreparation {
            even_sector_matrix: (0..9)
                .map(|k| Float::with_val(p, if k / 3 == k % 3 { [1, 2, 2][k / 3] } else { 0 }))
                .collect(),
            source_matrix_eigenvalue_allowance: Float::with_val(p, 0),
            selected_enclosures: decoded.selected_enclosures,
        };
        let state = [0, 0, 1, 0, 0].map(|x| Float::with_val(p, x));
        let report =
            response_spectral_isolation(&prep, &params, &cfg, &Float::with_val(p, 1), &state)
                .unwrap();
        let gap = Float::with_val(p, Float::parse(&report.sturm_gap_lower_bound).unwrap());
        assert!(gap > 0 && gap <= 1);
    }
}

#[test]
fn sector_bounds_prevent_zero_crossing_and_mislabeled_relative_accuracy() {
    let p = 128;
    let mut pair = CcmSectorEigenpairHp {
        algebraic_index: 0,
        eigenvalue: Float::with_val(p, 1),
        eigenvector: vec![],
        residual_norm: Float::with_val(p, 0),
        eigenvalue_lower: Float::with_val(p, -1),
        eigenvalue_upper: Float::with_val(p, 2),
    };
    assert!(sector_resolution::require_relative(&pair, p).is_err());
    pair.eigenvalue_lower = Float::with_val(p, 1) - (Float::with_val(p, 1) >> 4u32);
    pair.eigenvalue_upper = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 4u32);
    assert!(sector_resolution::require_relative(&pair, p).is_err());
    pair.eigenvalue_lower = Float::with_val(p, 1) - (Float::with_val(p, 1) >> 20u32);
    pair.eigenvalue_upper = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 20u32);
    sector_resolution::require_relative(&pair, p).unwrap();
}

#[test]
fn source_intervals_control_categories_and_the_gap_representative() {
    let p = 128;
    let point = |numerator: u32, denominator: u32| Float::with_val(p, numerator) / denominator;
    let spectrum = |parity, estimate: Float, exact: Float| CcmSectorSpectrumHp {
        parity,
        dimension: 2,
        eigenvalue_route: CcmSectorEigenvalueRoute::CompleteQr,
        complete_eigenvalues: Some(vec![estimate.clone(), Float::with_val(p, 2)]),
        eigenpairs: vec![
            CcmSectorEigenpairHp {
                algebraic_index: 0,
                eigenvalue: estimate,
                eigenvector: vec![Float::with_val(p, 1), Float::with_val(p, 0)],
                residual_norm: Float::with_val(p, 0),
                eigenvalue_lower: exact.clone(),
                eigenvalue_upper: exact,
            },
            CcmSectorEigenpairHp {
                algebraic_index: 1,
                eigenvalue: Float::with_val(p, 2),
                eigenvector: vec![Float::with_val(p, 0), Float::with_val(p, 1)],
                residual_norm: Float::with_val(p, 0),
                eigenvalue_lower: Float::with_val(p, 2),
                eigenvalue_upper: Float::with_val(p, 2),
            },
        ],
    };
    // Admissible sub-1/256 point errors can reverse the point ordering when
    // the exact spectra are close: 1 < 2049/2048, but 1025/1024 > 1.
    let even = spectrum(CcmParity::Even, point(1025, 1024), point(1, 1));
    let odd = spectrum(CcmParity::Odd, point(1, 1), point(2049, 2048));
    let result = compute_sector_gap(even, odd, p).unwrap();
    assert!(result.lambda_difference < 0); // retained point diagnostic
    assert_eq!(result.ordering, 1); // independently established source ordering
    assert!(result.even_simple);
    assert!(result.gap_log > 0);
    assert!(result.gap_log_lower <= result.gap_log);
    assert!(result.gap_log <= result.gap_log_upper);
    let expected = (Float::with_val(3 * p, 2049) / 2048u32).log10();
    assert!(result.gap_log_lower <= expected && expected <= result.gap_log_upper);
}
