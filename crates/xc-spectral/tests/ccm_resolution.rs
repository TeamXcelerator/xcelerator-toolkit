use xc_spectral::ccm::{run_f64, CcmParams};

#[test]
fn binary64_rejects_unresolved_states_and_preserves_resolved_control() {
    for (cutoff, modes) in [
        (13, 4),
        (13, 5),
        (13, 6),
        (13, 8),
        (13, 20),
        (5, 20),
        (9, 120),
    ] {
        let error = run_f64(&CcmParams::from_lambda_sq_integer(cutoff, modes))
            .expect_err("unresolved state must not return ordinary root data");
        assert!(
            error.to_string().contains("unresolved in binary64"),
            "{error}"
        );
    }
    let result = run_f64(&CcmParams::from_lambda_sq_integer(2, 20)).unwrap();
    // Independent mpmath/FLINT fixture fast_int_2_N20.json.
    let reference = [
        "14.349142580407094",
        "22.68170537589362",
        "31.17938936569662",
        "39.79791573548218",
        "48.50895393253196",
        "57.29108147362564",
        "66.12795104125227",
        "75.00730267108537",
        "83.92006486532992",
        "92.85958425054856",
        "101.82103240921651",
        "110.80098960466087",
        "119.7971908354194",
        "128.80843576054755",
        "137.83471417047145",
        "146.87773129854977",
        "155.94244420350178",
        "165.04202485221318",
        "174.2202661304796",
        "183.81868273616708",
    ];
    assert_eq!(result.spectral_roots().len(), reference.len());
    for (actual, reference) in result.spectral_roots().iter().zip(reference) {
        let reference = reference.parse::<f64>().unwrap();
        assert!((actual - reference).abs() < 1e-9, "{actual} vs {reference}");
    }
    assert!(result.spectral_roots().windows(2).all(|w| w[0] < w[1]));
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::Float;
    use xc_spectral::ccm::hp::{
        analyze_sector_gap_with_options, build_source, CcmEigenstateSolver,
        CcmSectorAnalysisOptions, HighPrecConfig,
    };

    #[test]
    fn selected_and_cross_checked_gaps_fail_when_source_uncertainty_dominates() {
        let cfg = HighPrecConfig::for_decimal_digits(30);
        for (modes, options) in [
            (30, CcmSectorAnalysisOptions::selected(2)),
            (39, CcmSectorAnalysisOptions::cross_checked(2)),
        ] {
            let outcome = analyze_sector_gap_with_options(
                &CcmParams::from_lambda_sq_integer(13, modes),
                &cfg,
                options,
            );
            assert!(
                outcome.is_err(),
                "an unresolved logarithmic gap must not be retained"
            );
        }
        let cfg = HighPrecConfig::for_decimal_digits(100);
        let resolved = analyze_sector_gap_with_options(
            &CcmParams::from_lambda_sq_integer(13, 6),
            &cfg,
            CcmSectorAnalysisOptions::cross_checked(2),
        )
        .unwrap();
        assert!(resolved.lambda_even > 0 && resolved.ordering == 1 && resolved.even_simple);
        assert!(resolved.gap_log_lower <= resolved.gap_log);
        assert!(resolved.gap_log <= resolved.gap_log_upper);
        assert!(resolved.gap_log.to_f64() > 2.69 && resolved.gap_log.to_f64() < 2.70);
    }

    #[test]
    #[ignore = "explicit documented N150/500-digit resource qualification"]
    fn documented_n150_500_digits_q600_build_source() {
        let mut cfg = HighPrecConfig::for_decimal_digits(500);
        cfg.quad_points = 600;
        let result = build_source(&CcmParams::from_lambda_sq_integer(5, 150), &cfg).unwrap();
        assert!(result.weil_min_eigenvalue > 0);
        assert_eq!(result.xi.len(), 301);
        assert_eq!(result.precision_bits, 1725);
        println!("N150_500_Q600_OK eigenvalue={}", result.weil_min_eigenvalue);
    }

    #[test]
    fn standalone_state_matches_explicit_inverse_iteration_control() {
        let params = CcmParams::from_lambda_sq_integer(13, 10);
        let mut cfg = HighPrecConfig::for_decimal_digits(10);
        let auto = build_source(&params, &cfg).unwrap();
        cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
        let legacy = build_source(&params, &cfg).unwrap();
        let p = cfg.precision_bits;
        let unit = |v: &[Float]| {
            let mut v = v.to_vec();
            xc_numerics::linalg::try_normalize_l2(&mut v).unwrap();
            v
        };
        let a = unit(&auto.xi);
        let b = unit(&legacy.xi);
        let mut distance = Float::with_val(p, 0);
        for (a, b) in a.iter().zip(&b) {
            let d = Float::with_val(p, a - b);
            distance += d.square();
        }
        assert!(distance.sqrt() < Float::with_val(p, Float::parse("1e-8").unwrap()));
        // Exact dyadic output for the independent Python eigsy oracle. This is
        // evidence for stored Tau, not for the unrounded assembly.
        let matrix = xc_spectral::ccm::hp::weil_matrix_hp(&params, &cfg, true).unwrap();
        let exact = |x: &Float| {
            let (mantissa, exponent) = x.to_integer_exp().unwrap();
            format!("{mantissa} {exponent}")
        };
        println!(
            "CCM_POLISH_ORACLE {}",
            serde_json::json!({
                "route_scope":"environment_selected_source_route_standalone_under_audit_cache_off",
                "precision_bits":p, "modes":10, "tau":matrix.iter().map(exact).collect::<Vec<_>>(),
                "auto":auto.xi.iter().map(exact).collect::<Vec<_>>(),
                "legacy":legacy.xi.iter().map(exact).collect::<Vec<_>>()
            })
        );
        let accuracy = auto.stored_eigenvalue_accuracy().unwrap();
        assert!(accuracy.absolute_error_upper >= 0);
        assert!(accuracy.assembly_error_bound.is_none());
    }
}
