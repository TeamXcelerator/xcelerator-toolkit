use super::*;

#[test]
fn certificate_endpoints_use_their_own_round_trip_precision() {
    let p = 64;
    let endpoint = Float::with_val(p, 1) + (Float::with_val(p, 1) >> 63u32);
    let text = endpoint.to_string();
    // p-bit endpoint identity must survive wider result precision and decimal rounding.
    let value = Float::with_val(256, &endpoint);
    assert!(stored_root_in_decimal_interval(&value, &text, &text, p).unwrap());
    let outside = value + (Float::with_val(256, 1) >> 100u32);
    assert!(!stored_root_in_decimal_interval(&outside, &text, &text, p).unwrap());
}
#[test]
fn event_edge_compares_exact_cutoff_values() {
    for cutoff in ["5", "5.0", "5.00000000000000000", "50e-1"] {
        assert!(cutoff_equals_event_power(cutoff, 5).unwrap());
    }
    for cutoff in ["5.00000000000000001", "4.99999999999999999"] {
        assert!(!cutoff_equals_event_power(cutoff, 5).unwrap());
    }
}
#[test]
fn standalone_precision_relabeling_is_rejected() {
    let old = Float::with_val(64, rug::Rational::from((1, 3))).to_string();
    assert!(parse_standalone_scalar(&old, 64).is_some());
    assert!(parse_standalone_scalar(&old, 256).is_none());
    let current = Float::with_val(256, rug::Rational::from((1, 3))).to_string();
    assert!(parse_standalone_scalar(&current, 256).is_some());
}
#[cfg(feature = "arb")]
#[test]
fn tau_assurance_uses_exact_independent_error_gate() {
    use rug::Rational;
    let scale = Rational::from(8);
    let allowance = Rational::from(1) >> 93u32;
    assert_eq!(
        tau_accuracy_agreement(&scale, &allowance, 128).unwrap(),
        allowance
    );
    assert!(
        tau_accuracy_agreement(&scale, &(allowance + (Rational::from(1) >> 200u32)), 128).is_err()
    );
    assert!(tau_accuracy_agreement(&Rational::from(0), &Rational::from(0), 128).is_ok());
    assert!(tau_accuracy_agreement(&Rational::from(0), &Rational::from(1), 128).is_err());
}
#[test]
fn fixed_root_outcomes_report_replayed_correction() {
    let p = 128;
    let xi = [Float::with_val(p, 1), Float::with_val(p, 2)];
    let poles = [Float::with_val(p, -1), Float::with_val(p, 1)];
    let root = solve_r_zero(
        &xi,
        &poles,
        &Float::with_val(p, 0),
        p,
        1,
        RootSolver::Newton,
    );
    let details = match &root {
        EigenvalueResult::Converged(r)
        | EigenvalueResult::Stagnated(r)
        | EigenvalueResult::Approximate(r) => r,
        _ => panic!("missing retained point: {root:?}"),
    };
    let expected = Float::with_val_round(
        p,
        secular_correction_at(
            &xi,
            &poles,
            &details.value,
            p + GUARD_BITS,
            RootSolver::Newton,
        )
        .unwrap(),
        rug::float::Round::Up,
    )
    .0;
    assert_eq!(details.diagnostics.final_correction, expected);
    assert!(details.diagnostics.final_correction > 0);
}
#[test]
fn crosschecked_capture_requests_conditioning() {
    let mut options = CcmResearchCaptureOptions::maximum(3);
    options.sector_analysis.as_mut().unwrap().eigenvalue_route =
        CcmSectorEigenvalueRoute::CrossChecked;
    assert!(options.captures_root_conditioning());
}

fn with_remaining_cache(test: impl FnOnce(&ArtifactCacheContext<'_>)) {
    use xc_cache::*;
    let root = crate::fresh_test_dir("remaining-ccm-cache");
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "remaining-local",
            &*root,
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
    let cache = ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["remaining-local".into()],
        mode: ArtifactExecutionCacheMode::PreferReuse,
        write_on_miss: true,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    test(&cache);
}
#[test]
fn admitted_empty_window_records_evidence_cold_and_warm() {
    with_remaining_cache(|cache| {
        let params = CcmParams::from_lambda_sq_integer(13, 1);
        let mut cfg = HighPrecConfig::for_decimal_digits(10);
        cfg.precision_bits = 128;
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        let target = ZeroTarget::HeightWindow {
            lower: "0.0001".into(),
            upper: "0.0002".into(),
        };
        for _ in 0..2 {
            let result = xc_numerics::hp_runtime::run_hp(|| {
                run_inner(
                    &params,
                    &cfg,
                    RootAcquisition::Independent {
                        target: &target,
                        options: IndependentRootDiscoveryOptions::complete_positive(true),
                    },
                    CcmCacheRoute::Fabric(cache),
                )
            })
            .unwrap();
            assert!(result.eigenvalues_pos.is_empty());
            assert!(result.spectral_root_index_range().is_none());
        }
    });
}
#[test]
fn managed_matrix_matches_off_computation() {
    with_remaining_cache(|cache| {
        let params = CcmParams::from_lambda_sq_integer(13, 1);
        let mut cfg = HighPrecConfig::for_decimal_digits(10);
        cfg.precision_bits = 128;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        xc_numerics::hp_runtime::run_hp(|| -> Result<()> {
            let l = log_lambda_sq_hp(&params, cfg.precision_bits)?;
            let expected = direct_matrix_tau(&params, &l, &cfg)?;
            let (cold, cm) = build_tau_hp_via_cache(&params, &l, &cfg, cache)?;
            let (warm, wm) = build_tau_hp_via_cache(&params, &l, &cfg, cache)?;
            assert_eq!(cold, expected);
            assert_eq!(warm, expected);
            assert_eq!(cm, wm);
            assert_eq!(cm.dependencies.len(), 2);
            Ok(())
        })
        .unwrap();
    });
}

#[test]
fn original_rounded_cancellation_retains_honest_stagnation() {
    let p = 96;
    let large = Float::with_val(p, 1) << 160u32;
    let mut xi = vec![Float::with_val(p, 0); 9];
    xi[0] = Float::with_val(p, rug::Rational::from((81, 4))) * &large;
    xi[1] = Float::with_val(p, 1);
    xi[2] = Float::with_val(p, rug::Rational::from((-25, 2))) * &large;
    xi[4] = Float::with_val(p, rug::Rational::from((1, 4))) * &large;
    xi[5] = Float::with_val(p, 1);
    xi[6] = Float::with_val(p, -3);
    let poles = (-4..=4).map(|j| Float::with_val(p, j)).collect::<Vec<_>>();
    let seed = Float::with_val(p, rug::Rational::from((1, 2)));
    let outcome = solve_r_zero(&xi, &poles, &seed, p, 8, RootSolver::Newton);
    let EigenvalueResult::Stagnated(result) = outcome else {
        panic!("expected retained precision-limited point: {outcome:?}");
    };
    let exact = rug::Rational::from((21, 202));
    let correction = result.diagnostics.final_correction.to_rational().unwrap();
    assert!(correction >= exact);
    assert!(correction < exact + (rug::Rational::from(1) >> 40u32));
    assert!(result.diagnostics.achieved_decimal_digits < 1);
    assert_eq!(result.value, seed);
}

#[test]
fn wide_cached_window_admits_only_the_requested_converged_prefix() {
    // Keep the original numerical conditions. Build its real source
    // once, then isolate root-window admission from source-assurance promotion.
    // CrossChecked below exercises the production converged-only root gate;
    // it does not assert that this Computed source has CrossChecked assurance.
    with_remaining_cache(|cache| {
        xc_numerics::hp_runtime::run_hp(|| -> Result<()> {
            let params = CcmParams::from_lambda_sq_integer(13, 64);
            let cfg = HighPrecConfig::for_decimal_digits(40);
            assert_eq!(cfg.precision_bits, 197);
            let dataset = xc_zeta::zeros::bundled_dataset_identity()?;
            let seeds = xc_zeta::zeros::bundled_first_n_strings(50)?
                .iter()
                .map(|text| Float::parse(text).map(|v| Float::with_val(cfg.precision_bits, v)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let l = log_lambda_sq_hp(&params, cfg.precision_bits)?;
            let (mut tau, tau_manifest) = build_tau_hp_via_cache(&params, &l, &cfg, cache)?;
            force_symmetric(&mut tau, params.matrix_size())?;
            let (_, xi, _, eigenpair_manifest, resolved_solver) =
                weil_eigenpair_via_cache_with_seed(
                    &params, &cfg, &l, &tau, &tau_manifest, cache, None, None,
                )?;
            let source = resolve_secular_source_via_cache(&params, &cfg, &eigenpair_manifest, cache)?;
            let (wide, wide_manifest, projected) = resolve_root_range_via_cache(
                &params, &cfg, resolved_solver, &l, &xi, 1, &seeds, &source, cache,
                RootArtifactMode::ReferenceSeededRefinement, Some(&dataset),
                RootWindowSemantics::strict_positive(50),
            )?;
            assert!(!projected);
            assert!(wide[..3].iter().all(EigenvalueResult::is_converged));
            let unresolved = wide.iter().position(|root| !root.is_converged())
                .expect("original numerical conditions must retain an unresolved tail root");
            assert!(unresolved >= 3);
            let context = |mode| ArtifactCacheContext {
                resolver: cache.resolver,
                reference_resolver: cache.reference_resolver,
                acceptance: cache.acceptance,
                ordered_overlays: cache.ordered_overlays.clone(),
                mode,
                write_on_miss: true,
                write_visibility: cache.write_visibility,
                requested_assurance: xc_core::AssuranceLevel::CrossChecked,
                certification_failure_policy: cache.certification_failure_policy,
                production_sink: None,
            };
            let resolve = |first, selected: &[Float], ctx: &ArtifactCacheContext<'_>| {
                resolve_root_range_via_cache(
                    &params, &cfg, resolved_solver, &l, &xi, first, selected, &source, ctx,
                    RootArtifactMode::ReferenceSeededRefinement, Some(&dataset),
                    RootWindowSemantics::strict_positive(selected.len()),
                )
            };
            let (head, reused_manifest, projected) = resolve(
                1, &seeds[..3], &context(xc_cache::ArtifactExecutionCacheMode::PreferReuse),
            )?;
            assert!(projected, "head must actually come from the retained wider window");
            assert_eq!(reused_manifest.content_digest, wide_manifest.content_digest);
            let (fresh, _, projected) = resolve(
                1, &seeds[..3], &context(xc_cache::ArtifactExecutionCacheMode::Refresh),
            )?;
            assert!(!projected, "control must compute the requested prefix afresh");
            assert_eq!(root_selection_digest(&head)?, root_selection_digest(&fresh)?);
            assert_eq!(root_selection_digest(&head)?, root_selection_digest(&wide[..3])?);
            for mode in [xc_cache::ArtifactExecutionCacheMode::PreferReuse,
                         xc_cache::ArtifactExecutionCacheMode::Refresh] {
                let rejected = resolve(unresolved + 1, &seeds[unresolved..unresolved + 1], &context(mode));
                assert!(rejected.is_err(), "requested unresolved ordinal must fail in {mode:?}");
            }
            eprintln!("C13 N64 p197, {} unresolved tail roots; retained prefix equals fresh prefix; ordinal {} rejected",
                wide.iter().filter(|root| !root.is_converged()).count(), unresolved + 1);
            Ok(())
        }).unwrap();
    });
}
#[test]
fn small_cutoff_archimedean_order_meets_its_precision_target() {
    // At lambda^2 = 2 the pole ellipse is wide and the fixed 3n allowance
    // under-resolved middle modes (about 2^-244 at p = 256 for n = 10).
    // An explicit low quad_points floor exposes the per-mode order itself.
    let p = 256;
    let length = Float::with_val(p, 2).ln();
    let orders =
        super::super::research::quadrature_orders_for_length(10, 1, p, 1, &length).unwrap();
    for n in [5usize, 10] {
        let order = orders[n];
        let coarse = xc_numerics::quadrature::try_gauss_legendre_nodes(
            order,
            p,
            xc_numerics::quadrature::CacheMode::Off,
        )
        .unwrap();
        let fine = xc_numerics::quadrature::try_gauss_legendre_nodes(
            4 * order,
            p,
            xc_numerics::quadrature::CacheMode::Off,
        )
        .unwrap();
        let a =
            compute_archimedean_integrals_l(n as i64, &length, p, &coarse.0, &coarse.1).unwrap();
        let b = compute_archimedean_integrals_l(n as i64, &length, p, &fine.0, &fine.1).unwrap();
        // Both rules store p-bit nodes and weights, so correctly rounded
        // results can differ by a few ulps; the former order missed by 2^-244.
        let tolerance = Float::with_val(p, 1) >> (p - 8);
        for (left, right) in [(&a.0, &b.0), (&a.1, &b.1), (&a.2, &b.2)] {
            let difference = Float::with_val(p, left - right).abs();
            assert!(
                difference <= tolerance,
                "n={n} order={order}: integral changed by {difference}"
            );
        }
    }
}
