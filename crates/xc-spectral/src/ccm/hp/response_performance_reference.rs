//! Frozen scalar response oracle from Toolkit 545041c, compiled only for tests.
use super::*;

fn reference_prime_power_velocity(
    n_modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    vector: &[Float],
    precision_bits: u32,
) -> Result<PrimePowerVelocityAction> {
    let dimension = 2 * n_modes + 1;
    if vector.len() != dimension || l <= &Float::with_val(precision_bits, 0) {
        bail!("prime-power response received incompatible dimensions or cutoff");
    }
    let log_power = Float::with_val(precision_bits, power).ln();
    let von_mangoldt_weight = Float::with_val(precision_bits, prime).ln();
    let sqrt_power = Float::with_val(precision_bits, power).sqrt();
    let mut reduced_position = Float::with_val(precision_bits, 1);
    let mut ratio = Float::with_val(precision_bits, &log_power);
    ratio /= l;
    reduced_position -= ratio;

    let mut velocity_coefficient = Float::with_val(precision_bits, &von_mangoldt_weight);
    velocity_coefficient *= &log_power;
    velocity_coefficient /= &sqrt_power;
    let mut l_squared = Float::with_val(precision_bits, l);
    l_squared.square_mut();
    velocity_coefficient /= l_squared;
    velocity_coefficient = -velocity_coefficient;

    let mut edge_jump_coefficient = Float::with_val(precision_bits, &von_mangoldt_weight);
    edge_jump_coefficient *= -2i32;
    edge_jump_coefficient /= &sqrt_power;
    edge_jump_coefficient /= &log_power;

    let pi_value = pi(precision_bits);
    let mut two_pi = Float::with_val(precision_bits, &pi_value);
    two_pi *= 2u32;
    let mut four_pi = Float::with_val(precision_bits, &pi_value);
    four_pi *= 4u32;
    let modes = (-(n_modes as i64)..=(n_modes as i64)).collect::<Vec<_>>();
    let phases = modes
        .iter()
        .map(|mode| {
            let mut phase = Float::with_val(precision_bits, &two_pi);
            phase *= fl_i(precision_bits, *mode);
            phase *= &reduced_position;
            phase
        })
        .collect::<Vec<_>>();
    let sines = phases
        .iter()
        .map(|phase| phase.clone().sin())
        .collect::<Vec<_>>();
    let cosines = phases.into_iter().map(Float::cos).collect::<Vec<_>>();

    let action = modes
        .par_iter()
        .enumerate()
        .map(|(row, n)| {
            let mut terms = Vec::with_capacity(dimension);
            for (column, m) in modes.iter().enumerate() {
                let derivative_kernel = if n == m {
                    let mut value = Float::with_val(precision_bits, &cosines[row]);
                    value *= 2u32;
                    let mut oscillatory = Float::with_val(precision_bits, &four_pi);
                    oscillatory *= fl_i(precision_bits, *n);
                    oscillatory *= &reduced_position;
                    oscillatory *= &sines[row];
                    value -= oscillatory;
                    value
                } else {
                    let mut value = Float::with_val(precision_bits, &cosines[row]);
                    value *= fl_i(precision_bits, *n);
                    let mut other = Float::with_val(precision_bits, &cosines[column]);
                    other *= fl_i(precision_bits, *m);
                    value -= other;
                    value *= 2u32;
                    value /= fl_i(precision_bits, n - m);
                    value
                };
                let mut term = derivative_kernel;
                term *= &vector[column];
                terms.push(term);
            }
            let mut value =
                xc_numerics::reduction::deterministic_pairwise_sum_hp(&terms, precision_bits);
            value *= &velocity_coefficient;
            value
        })
        .collect::<Vec<_>>();

    Ok(PrimePowerVelocityAction {
        log_power,
        von_mangoldt_weight,
        reduced_position,
        velocity_coefficient,
        edge_jump_coefficient,
        action,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reference_prime_power_response(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    tau: &[Float],
    state_eigenvalue: &Float,
    xi: &[Float],
    roots: &[EigenvalueResult],
    first_positive_root_index: usize,
    tau_manifest: &ArtifactManifest,
    eigenpair_manifest: &ArtifactManifest,
    root_manifest: &ArtifactManifest,
    secular_manifest: &ArtifactManifest,
    selection_digest: &ContentDigest,
    spectral_preparation: &ResponseSpectralPreparation,
) -> Result<PortablePrimePowerResponseAnalysis> {
    let precision_bits = cfg.precision_bits;
    let dimension = params.matrix_size();
    if roots.is_empty() || tau.len() != dimension * dimension || xi.len() != dimension {
        bail!("prime-power response capture requires a retained state and root window");
    }

    let xi_norm = deterministic_l2_norm_hp(xi, precision_bits);
    if xi_norm.is_zero() {
        bail!("prime-power response capture received a zero eigenstate");
    }
    let unit_state = xi
        .iter()
        .map(|value| {
            let mut normalized = Float::with_val(precision_bits, value);
            normalized /= &xi_norm;
            normalized
        })
        .collect::<Vec<_>>();
    let unit_state_sum =
        xc_numerics::reduction::deterministic_pairwise_sum_hp(&unit_state, precision_bits);
    if unit_state_sum.is_zero() {
        bail!("prime-power response cannot preserve the CCM zero-sum eigenstate normalization");
    }
    let mut ccm_scale = Float::with_val(precision_bits, l).sqrt();
    ccm_scale /= &unit_state_sum;
    let spectral_isolation = response_spectral_isolation(
        spectral_preparation,
        params,
        cfg,
        state_eigenvalue,
        &unit_state,
    )?;
    let bordered_solver = build_even_sector_bordered_response_solver(
        spectral_preparation,
        params,
        cfg,
        state_eigenvalue,
        &unit_state,
    )?;
    let shifted_frobenius_norm =
        shifted_matrix_frobenius_norm(tau, state_eigenvalue, dimension, precision_bits);
    let (poles, _) = ccm_secular_poles_and_u_velocities(l, params.n_modes, precision_bits);
    let portable_roots = ccm_response_roots(roots, first_positive_root_index);
    let prime_content = prime_powers_up_to(params.lambda_sq_int());
    let lambda_identity = lambda_squared_cache_identity(params);
    let mut events = Vec::with_capacity(prime_content.len());

    for (power, prime, exponent) in prime_content {
        let velocity = reference_prime_power_velocity(
            params.n_modes,
            power,
            prime,
            l,
            &unit_state,
            precision_bits,
        )?;
        let eigenvalue_response =
            deterministic_dot_hp(&unit_state, &velocity.action, precision_bits);
        let projected_forcing = velocity
            .action
            .iter()
            .zip(&unit_state)
            .map(|(action, state)| {
                let mut projection = Float::with_val(precision_bits, state);
                projection *= &eigenvalue_response;
                let mut value = Float::with_val(precision_bits, action);
                value -= projection;
                value
            })
            .collect::<Vec<_>>();
        let projected_forcing_norm = deterministic_l2_norm_hp(&projected_forcing, precision_bits);
        let (eigenvector_response, lagrange_multiplier) =
            solve_even_sector_bordered_response(&bordered_solver, &projected_forcing);
        let response_norm = deterministic_l2_norm_hp(&eigenvector_response, precision_bits);
        let response_sum = xc_numerics::reduction::deterministic_pairwise_sum_hp(
            &eigenvector_response,
            precision_bits,
        );
        let mut ccm_scale_response = Float::with_val(precision_bits, &ccm_scale);
        ccm_scale_response *= &response_sum;
        ccm_scale_response /= &unit_state_sum;
        ccm_scale_response = -ccm_scale_response;
        let root_velocity_responses = roots
            .iter()
            .map(|outcome| {
                outcome
                    .value()
                    .map(|root| {
                        prime_power_root_velocity_response(
                            &unit_state,
                            &eigenvector_response,
                            &poles,
                            root,
                            precision_bits,
                        )
                        .map(|response| lossless_hp_decimal(&response))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let relative_residual = bordered_response_relative_residual(
            tau,
            state_eigenvalue,
            &unit_state,
            &projected_forcing,
            &eigenvector_response,
            &lagrange_multiplier,
            &shifted_frobenius_norm,
            precision_bits,
        );
        if !weil_eigvec_cache::residual_within_precision_floor(&relative_residual, precision_bits) {
            bail!(
                "prime-power response bordered solve for {power} failed its precision-scaled residual gate"
            );
        }

        events.push(PortablePrimePowerResponseEvent {
            power,
            prime,
            exponent,
            log_power: lossless_hp_decimal(&velocity.log_power),
            von_mangoldt_weight: lossless_hp_decimal(&velocity.von_mangoldt_weight),
            reduced_position: lossless_hp_decimal(&velocity.reduced_position),
            velocity_coefficient: lossless_hp_decimal(&velocity.velocity_coefficient),
            edge_jump_coefficient: lossless_hp_decimal(&velocity.edge_jump_coefficient),
            observation_is_event_edge: lambda_identity == power.to_string(),
            eigenvalue_velocity_response: lossless_hp_decimal(&eigenvalue_response),
            projected_forcing_norm: lossless_hp_decimal(&projected_forcing_norm),
            l2_eigenvector_velocity_response_norm: lossless_hp_decimal(&response_norm),
            l2_eigenvector_velocity_response: encode_hp_vector(&eigenvector_response),
            ccm_normalization_scale_velocity_response: lossless_hp_decimal(&ccm_scale_response),
            bordered_lagrange_multiplier: lossless_hp_decimal(&lagrange_multiplier),
            bordered_solve_relative_residual: lossless_hp_decimal(&relative_residual),
            root_velocity_responses,
        });
    }

    let parity_policy = cfg.effective_parity_policy();
    Ok(PortablePrimePowerResponseAnalysis {
        schema_version: 2,
        lambda_squared: lambda_identity,
        prime_cutoff: params.lambda_sq_int(),
        n_modes: params.n_modes,
        dimension,
        precision_bits,
        force_even: parity_policy.legacy_force_even(),
        parity_policy: parity_policy.portable_marker(),
        tau_content_digest: tau_manifest.content_digest.0.clone(),
        eigenpair_content_digest: eigenpair_manifest.content_digest.0.clone(),
        root_range_content_digest: root_manifest.content_digest.0.clone(),
        secular_source_content_digest: secular_manifest.content_digest.0.clone(),
        root_selection_digest: selection_digest.0.clone(),
        normalization: PRIME_POWER_RESPONSE_NORMALIZATION.to_owned(),
        velocity_parameter: PRIME_POWER_RESPONSE_VELOCITY_PARAMETER.to_owned(),
        response_definition: PRIME_POWER_RESPONSE_DEFINITION.to_owned(),
        edge_jump_direction: PRIME_POWER_RESPONSE_EDGE_DIRECTION.to_owned(),
        state_eigenvalue: lossless_hp_decimal(state_eigenvalue),
        spectral_isolation,
        roots: portable_roots,
        events,
    })
}
