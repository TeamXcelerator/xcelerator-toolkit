use super::*;

#[test]
fn retained_eigenstate_replays_parity_and_normalization() {
    let p = 128;
    let params = CcmParams::from_lambda_sq_integer(13, 1);
    let mut cfg = HighPrecConfig::for_decimal_digits(20);
    cfg.precision_bits = p;
    cfg.eigenstate_solver = CcmEigenstateSolver::LegacyInverseIteration;
    let a = log_lambda_sq_hp(&params, p).unwrap().sqrt();
    let tau = (0..9)
        .map(|j| {
            Float::with_val(
                p,
                if j == 4 {
                    1
                } else if j == 0 || j == 8 {
                    2
                } else {
                    0
                },
            )
        })
        .collect::<Vec<_>>();
    let mut artifact = PortableWeilEigenpair {
        stored_state_resolution: None,
        schema_version: 4,
        lambda_squared: "13".into(),
        n_modes: 1,
        precision_bits: p,
        force_even: true,
        parity_policy: None,
        eigenstate_route: legacy_eigenstate_route_name(),
        eigenvalue: "1".into(),
        eigenvector: vec!["0".into(), a.to_string(), "0".into()],
        inverse_iteration: PortableInverseIterationDiagnostics {
            configured_step_limit: cfg.inverse_iter_steps,
            unshifted_steps: 2,
            unshifted_converged: true,
            final_relative_rayleigh_change: Some("0".into()),
            shifted_refinement: "accepted".into(),
            final_relative_residual_norm: "0".into(),
        },
        shift_invert_krylov: None,
    };
    artifact.stored_state_resolution = Some(
        stored_resolution::bounds(
            &tau,
            &parse_hp_vector(&artifact.eigenvector, p).unwrap(),
            &Float::with_val(p, 1),
            p,
            cfg.effective_parity_policy(),
        )
        .unwrap()
        .record,
    );
    assert!(decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_ok());
    artifact.eigenvector = vec!["0".into(), "0".into(), a.to_string()];
    assert!(
        decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_err(),
        "zero residual did not establish declared even parity"
    );
    artifact.eigenvector = vec!["0".into(), (a.clone() * 2u32).to_string(), "0".into()];
    assert!(
        decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_err(),
        "zero residual did not establish declared normalization"
    );
    artifact.eigenvector = vec![a.to_string(), "0".into(), (-a.clone()).to_string()];
    assert!(decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_err());
    cfg.set_parity_policy(CcmParityPolicy::Natural);
    artifact.force_even = false;
    // Natural parity permits a nonsymmetric vector when it is the simple ground state.
    let tau = (0..9)
        .map(|j| {
            Float::with_val(
                p,
                if j == 8 {
                    1
                } else if j == 0 || j == 4 {
                    2
                } else {
                    0
                },
            )
        })
        .collect::<Vec<_>>();
    artifact.eigenvector = vec!["0".into(), "0".into(), a.to_string()];
    artifact.stored_state_resolution = Some(
        stored_resolution::bounds(
            &tau,
            &parse_hp_vector(&artifact.eigenvector, p).unwrap(),
            &Float::with_val(p, 1),
            p,
            cfg.effective_parity_policy(),
        )
        .unwrap()
        .record,
    );
    assert!(decode_weil_eigenpair(&artifact, &params, &cfg, &tau).is_ok());
    // Ordinary computed reuse keeps every quadratic gate but not the cubic
    // ground-index proof; fresh output and explicit verification keep it.
    ground_index::CHECKED_SOURCES.with(|cache| cache.borrow_mut().clear());
    ground_index::FULL_GROUND_CHECKS.with(|count| count.set(0));
    assert!(decode_weil_eigenpair_admitted(&artifact, &params, &cfg, &tau, false).is_ok());
    assert_eq!(
        ground_index::FULL_GROUND_CHECKS.with(|count| count.get()),
        0
    );
    assert!(decode_weil_eigenpair_admitted(&artifact, &params, &cfg, &tau, true).is_ok());
    assert_eq!(
        ground_index::FULL_GROUND_CHECKS.with(|count| count.get()),
        1
    );
    let mut stale = artifact.clone();
    stale.inverse_iteration.final_relative_residual_norm = "1e-10".into();
    assert!(decode_weil_eigenpair_admitted(&stale, &params, &cfg, &tau, false).is_err());
}

#[test]
fn inverse_iteration_metrics_must_be_finite() {
    for metric in ["NaN", "inf", "-inf"] {
        let report = PortableInverseIterationDiagnostics {
            configured_step_limit: 2,
            unshifted_steps: 2,
            unshifted_converged: true,
            final_relative_rayleigh_change: Some(metric.into()),
            shifted_refinement: "accepted".into(),
            final_relative_residual_norm: "0".into(),
        };
        assert!(report.to_runtime(128).is_err());
    }
}

#[test]
fn normalization_screen_accounts_for_rounding_of_canceling_coefficients() {
    let p = 128;
    let length = Float::with_val(p, 13).ln();
    let target = length.clone().sqrt();
    let big = Float::with_val(p, 1) << 50u32;
    let negative = Float::with_val(p, Float::with_val(p, &target / 2) - &big);
    let coefficients = vec![
        big.clone(),
        negative.clone(),
        Float::with_val(p, 0),
        negative,
        big,
    ];
    validate_eigenstate_contract(&coefficients, &length, p, CcmParityPolicy::EvenSector).unwrap();
    let doubled = coefficients
        .iter()
        .map(|x| Float::with_val(p, x * 2))
        .collect::<Vec<_>>();
    assert!(
        validate_eigenstate_contract(&doubled, &length, p, CcmParityPolicy::EvenSector).is_err()
    );
}
