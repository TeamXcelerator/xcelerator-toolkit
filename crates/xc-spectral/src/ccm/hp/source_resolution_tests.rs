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
        "ccm_even_zero_shift_krylov_budgeted_count_polish_resolution_v7"
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
fn primary_source_admission_uses_campaign_configuration_without_matrix_allocation() {
    let cfg = HighPrecConfig::for_decimal_digits(1000).with_adaptive_root_precision();
    assert_eq!(cfg.precision_bits, 3386);
    cfg.validate_source_admission(&CcmParams::from_lambda_sq_integer(13, 120))
        .unwrap();
    let budget = super::super::capture_runtime::CaptureResourcePolicy::from_environment()
        .unwrap()
        .maximum_working_bytes;
    assert_eq!(
        cfg.validate_source_admission(&CcmParams::from_lambda_sq_integer(100, 500))
            .is_ok(),
        budget >= 5_259_675_544
    );
    let mut invalid = cfg.clone();
    invalid.quad_points = 0;
    assert!(invalid
        .validate_source_admission(&CcmParams::from_lambda_sq_integer(13, 120))
        .is_err());
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
            even_sector_matrix: shared_float_vector(
                (0..9)
                    .map(|k| Float::with_val(p, if k / 3 == k % 3 { [1, 2, 2][k / 3] } else { 0 }))
                    .collect(),
            )
            .unwrap(),
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
    assert_eq!(result.ordering, Some(1)); // independently established source ordering
    assert!(result.even_simple);
    assert!(result.unresolved.is_empty());
    let (gap_log, lower, upper) = (
        result.gap_log.unwrap(),
        result.gap_log_lower.unwrap(),
        result.gap_log_upper.unwrap(),
    );
    assert!(gap_log > 0);
    assert!(lower <= gap_log);
    assert!(gap_log <= upper);
    let expected = (Float::with_val(3 * p, 2049) / 2048u32).log10();
    assert!(lower <= expected && expected <= upper);
}

#[test]
fn source_workspace_obeys_declared_budget_without_changing_domain_limits() {
    let budget = u128::from(
        super::super::capture_runtime::CaptureResourcePolicy::from_environment()
            .unwrap()
            .maximum_working_bytes,
    );
    source_workspace(budget, "boundary").unwrap();
    assert!(source_workspace(budget + 1, "boundary").is_err());
    // HP-2000/N=970 requires 14,090,378,940 bytes for these four source buffers.
    assert_eq!(
        validate_source_shape(970, 6708, 4000).is_ok(),
        budget >= 14_090_378_940
    );
    assert!(validate_source_shape(4097, 64, 1).is_err());
    assert!(validate_source_shape(1, 64, 1_000_001).is_err());
    assert!(validate_source_shape(1, 63, 1).is_err());
}

#[test]
fn quadrature_admission_charges_concurrent_work_and_rejects_invalid_orders() {
    let budget = u128::from(
        super::super::capture_runtime::CaptureResourcePolicy::from_environment()
            .unwrap()
            .maximum_working_bytes,
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        // One retained table plus two simultaneous guarded mode workspaces.
        let expected = 300_000u128 * 2 * (424 + 96) + 300_000u128 * 2 * (1359 + 96) * 16;
        assert_eq!(
            quadrature_workspace(1, 3386, &[300_000, 300_000]).is_ok(),
            expected <= budget
        );
        assert!(quadrature_workspace(1, 3386, &[300_000]).is_err());
        assert!(quadrature_workspace(1, 3386, &[300_000, 0]).is_err());
    });
}

#[test]
fn unresolved_sector_conclusions_are_reported_one_by_one() {
    let p = 128;
    let one = Float::with_val(p, 1);
    let pair = |index, estimate: &Float, lower: Float, upper: Float| CcmSectorEigenpairHp {
        algebraic_index: index,
        eigenvalue: estimate.clone(),
        eigenvector: vec![Float::with_val(p, 1), Float::with_val(p, 0)],
        residual_norm: Float::with_val(p, 0),
        eigenvalue_lower: lower,
        eigenvalue_upper: upper,
    };
    let exact = |index, value: &Float| pair(index, value, value.clone(), value.clone());
    let spectrum =
        |parity, first: CcmSectorEigenpairHp, second: CcmSectorEigenpairHp| CcmSectorSpectrumHp {
            parity,
            dimension: 2,
            eigenvalue_route: CcmSectorEigenvalueRoute::Selected,
            complete_eigenvalues: None,
            eigenpairs: vec![first, second],
        };
    let two = Float::with_val(p, 2);
    let eighth = one.clone() >> 3u32;

    // Even ground 2^-100 enclosed to only four relative bits: its depth and
    // the GapLog are unresolved; ordering, simplicity and the odd depth are not.
    let tiny = one.clone() >> 100u32;
    let wide = tiny.clone() >> 4u32;
    let even = spectrum(
        CcmParity::Even,
        pair(0, &tiny, tiny.clone() - &wide, tiny.clone() + &wide),
        exact(1, &two),
    );
    let odd = spectrum(CcmParity::Odd, exact(0, &one), exact(1, &two));
    let gap = compute_sector_gap(even, odd, p).unwrap();
    assert_eq!(gap.ordering, Some(1));
    assert!(gap.even_simple);
    assert!(gap.d_even.is_none() && gap.d_odd.is_some());
    assert!(gap.gap_log.is_none() && gap.gap_log_lower.is_none() && gap.gap_log_upper.is_none());
    assert!(gap.difference_depth.is_none());
    assert_eq!(
        gap.unresolved,
        ["even ground eigenvalue: CCM sector resolution limit: fewer than eight relative bits are resolved; increase precision"]
    );
    let params = CcmParams::from_lambda_sq_integer(13, 1);
    // The resolved ordering still determines the natural state's evenness.
    evenness_from_sector_gap(&params, p, &gap).unwrap();

    // Overlapping even and odd ground enclosures: ordering is unresolved, and
    // evenness is a resolution limit rather than a guess.
    let overlap =
        |estimate: &Float| pair(0, estimate, one.clone() - &eighth, one.clone() + &eighth);
    let gap = compute_sector_gap(
        spectrum(CcmParity::Even, overlap(&one), exact(1, &two)),
        spectrum(CcmParity::Odd, overlap(&one), exact(1, &two)),
        p,
    )
    .unwrap();
    assert_eq!(gap.ordering, None);
    assert!(gap
        .unresolved
        .iter()
        .any(|reason| reason == "CCM sector resolution limit: even/odd ordering is unresolved"));
    let Err(error) = evenness_from_sector_gap(&params, p, &gap) else {
        panic!("an unresolved ordering must not determine evenness");
    };
    assert!(is_sector_resolution_limit(&error), "{error}");

    // Even ground and first excited enclosures overlap: simplicity is not
    // established, and an even natural state is then not determined.
    let gap = compute_sector_gap(
        spectrum(
            CcmParity::Even,
            exact(0, &one),
            pair(1, &(one.clone() + &eighth), one.clone(), two.clone()),
        ),
        spectrum(
            CcmParity::Odd,
            exact(0, &two),
            exact(1, &(two.clone() + 1u32)),
        ),
        p,
    )
    .unwrap();
    assert_eq!(gap.ordering, Some(1));
    assert!(!gap.even_simple);
    assert!(gap.gap_log.is_some());
    assert_eq!(
        gap.unresolved,
        ["CCM sector resolution limit: even ground-state simplicity is unresolved"]
    );
    let Err(error) = evenness_from_sector_gap(&params, p, &gap) else {
        panic!("an unresolved even ground state must not determine evenness");
    };
    assert!(is_sector_resolution_limit(&error), "{error}");
}

#[test]
fn equal_point_estimates_leave_only_the_difference_depth_unresolved() {
    // Both 128-bit point estimates are 1, while the source enclosures are
    // exactly 1 and 1+2^-130: ordering, simplicity and both depths resolve,
    // but the point difference has no logarithmic depth.
    let p = 128;
    let wide = 256;
    let exact = |index, estimate: Float, value: Float| CcmSectorEigenpairHp {
        algebraic_index: index,
        eigenvalue: estimate,
        eigenvector: vec![Float::with_val(p, 1), Float::with_val(p, 0)],
        residual_norm: Float::with_val(p, 0),
        eigenvalue_lower: value.clone(),
        eigenvalue_upper: value,
    };
    let one = Float::with_val(p, 1);
    let above = Float::with_val(wide, 1) + (Float::with_val(wide, 1) >> 130u32);
    let two = Float::with_val(p, 2);
    let spectrum = |parity, first, second| CcmSectorSpectrumHp {
        parity,
        dimension: 2,
        eigenvalue_route: CcmSectorEigenvalueRoute::CompleteQr,
        complete_eigenvalues: None,
        eigenpairs: vec![first, second],
    };
    let gap = compute_sector_gap(
        spectrum(
            CcmParity::Even,
            exact(0, one.clone(), one.clone()),
            exact(1, two.clone(), two.clone()),
        ),
        spectrum(
            CcmParity::Odd,
            exact(0, one.clone(), above),
            exact(1, two.clone(), two.clone()),
        ),
        p,
    )
    .unwrap();
    assert_eq!(gap.ordering, Some(1));
    assert!(gap.even_simple);
    assert!(gap.d_even.is_some() && gap.d_odd.is_some());
    // GapLog uses its source-supported representative.
    let (gap_log, lower, upper) = (
        gap.gap_log.clone().unwrap(),
        gap.gap_log_lower.clone().unwrap(),
        gap.gap_log_upper.clone().unwrap(),
    );
    assert!(gap_log > 0 && lower <= gap_log && gap_log <= upper);
    assert!(gap.lambda_difference.is_zero());
    assert!(gap.difference_depth.is_none());
    assert_eq!(
        gap.unresolved,
        ["CCM sector resolution limit: the even/odd point difference is zero; its logarithmic depth is unresolved"]
    );
    // Serialized as null with the reason, and counted as unresolved coverage.
    let version = serde_json::json!({"major":0,"minor":16,"patch":0,"prerelease":null});
    let manifest = |name: &str| -> ArtifactManifest {
        serde_json::from_value(serde_json::json!({
            "schema_version":1,"key":{"kind":"ccm_sector_spectrum","logical_key":name,"parameters_digest":xc_cache::ContentDigest::sha256(name.as_bytes())},
            "content_digest":xc_cache::ContentDigest::sha256(name.as_bytes()),"size_bytes":0,"objects":[],"created_unix_seconds":1,
            "producer_toolkit_version":version,"minimum_reader_version":version,"maximum_reader_version":null,"quality":"validated","visibility":"local","immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
        }))
        .unwrap()
    };
    let payload = serde_json::to_value(portable_sector_gap(
        &gap,
        &CcmParams::from_lambda_sq_integer(13, 1),
        &HighPrecConfig::for_decimal_digits(20),
        &manifest("even"),
        &manifest("odd"),
    ))
    .unwrap();
    assert!(payload["difference_depth"].is_null());
    assert!(payload["gap_log"].is_string());
    assert_eq!(payload["unresolved"].as_array().unwrap().len(), 1);
    let coverage = xc_cache::NumericalCoverage::from_value(&payload);
    assert_eq!((coverage.resolved_rows, coverage.unresolved_rows), (0, 1));
    assert_eq!(coverage.outcome, "partial_unresolved");
}

#[test]
fn resolved_sector_gap_payload_keeps_its_fields_and_bytes() {
    let p = 128;
    let exact = |index, value: u32| CcmSectorEigenpairHp {
        algebraic_index: index,
        eigenvalue: Float::with_val(p, value),
        eigenvector: vec![Float::with_val(p, 1), Float::with_val(p, 0)],
        residual_norm: Float::with_val(p, 0),
        eigenvalue_lower: Float::with_val(p, value),
        eigenvalue_upper: Float::with_val(p, value),
    };
    let spectrum = |parity, first, second| CcmSectorSpectrumHp {
        parity,
        dimension: 2,
        eigenvalue_route: CcmSectorEigenvalueRoute::Selected,
        complete_eigenvalues: None,
        eigenpairs: vec![exact(0, first), exact(1, second)],
    };
    let gap = compute_sector_gap(
        spectrum(CcmParity::Even, 1, 3),
        spectrum(CcmParity::Odd, 2, 4),
        p,
    )
    .unwrap();
    assert!(gap.unresolved.is_empty());
    let version = serde_json::json!({"major":0,"minor":16,"patch":0,"prerelease":null});
    let manifest = |name: &str| -> ArtifactManifest {
        serde_json::from_value(serde_json::json!({
            "schema_version":1,"key":{"kind":"ccm_sector_spectrum","logical_key":name,"parameters_digest":xc_cache::ContentDigest::sha256(name.as_bytes())},
            "content_digest":xc_cache::ContentDigest::sha256(name.as_bytes()),"size_bytes":0,"objects":[],"created_unix_seconds":1,
            "producer_toolkit_version":version,"minimum_reader_version":version,"maximum_reader_version":null,"quality":"validated","visibility":"local","immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
        }))
        .unwrap()
    };
    let payload = portable_sector_gap(
        &gap,
        &CcmParams::from_lambda_sq_integer(13, 1),
        &HighPrecConfig::for_decimal_digits(20),
        &manifest("even"),
        &manifest("odd"),
    );
    let value = serde_json::to_value(&payload).unwrap();
    let object = value.as_object().unwrap();
    // Exactly the fields of the earlier record, each a string, number or
    // boolean as before: a resolved record serializes to the same bytes.
    let fields: Vec<_> = object.keys().map(String::as_str).collect();
    assert_eq!(
        fields,
        [
            "d_even",
            "d_odd",
            "difference_depth",
            "even_simple",
            "even_simplicity_margin",
            "even_spectrum_content_digest",
            "gap_log",
            "gap_log_lower",
            "gap_log_upper",
            "lambda_difference",
            "lambda_even",
            "lambda_odd",
            "lambda_squared",
            "n_modes",
            "odd_spectrum_content_digest",
            "ordering",
            "precision_bits",
            "schema_version",
        ]
    );
    assert!(object.values().all(|field| !field.is_null()));
    assert_eq!(object["ordering"], serde_json::json!(1));
    assert_eq!(object["schema_version"], serde_json::json!(2));
}

#[test]
fn shared_symmetry_preserves_owned_precision_and_signed_zero() {
    for (a, b) in [
        (Float::with_val(64, 1), Float::with_val(128, 1)),
        (Float::with_val(128, 0.0), Float::with_val(128, -0.0)),
        (Float::with_val(128, -0.0), Float::with_val(128, 0.0)),
    ] {
        let values = vec![Float::with_val(128, 2), a, b, Float::with_val(128, 3)];
        let mut expected = values.clone();
        force_symmetric(&mut expected, 2).unwrap();
        let mut shared = shared_float_vector(values).unwrap();
        let original = shared.clone();
        force_shared_symmetric(&mut shared, 2).unwrap();
        assert!(!shared.ptr_eq(&original));
        for (actual, expected) in shared.iter().zip(&expected) {
            assert_eq!(actual, expected);
            assert_eq!(actual.prec(), expected.prec());
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        }
        let normalized = shared.clone();
        force_shared_symmetric(&mut shared, 2).unwrap();
        assert!(shared.ptr_eq(&normalized));
    }
}
