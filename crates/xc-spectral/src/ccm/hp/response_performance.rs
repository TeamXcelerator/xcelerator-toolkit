//! Reuse directed root geometry while preserving correctly rounded responses.
use super::*;

// Cached response records are computed observations. Routine admission checks
// their exact source joins and numerical shape; it does not manufacture a new
// independent cross-check or rerun derivative quadrature/bordered solves.
fn number(value: &str, p: u32, nonnegative: bool) -> std::result::Result<Float, CacheError> {
    let value = parse_hp_scalar(value, p)?;
    if nonnegative && value < 0 {
        return Err(CacheError::InvalidManifest(
            "negative response norm or bound".into(),
        ));
    }
    Ok(value)
}

fn root_values(
    values: &[Option<String>],
    roots: &[EigenvalueResult],
    p: u32,
) -> std::result::Result<(), CacheError> {
    if values.len() != roots.len() {
        return Err(CacheError::InvalidManifest(
            "response root count mismatch".into(),
        ));
    }
    for (value, root) in values.iter().zip(roots) {
        if value.is_some() != root.value().is_some() {
            return Err(CacheError::InvalidManifest(
                "response root availability mismatch".into(),
            ));
        }
        if let Some(value) = value {
            number(value, p, false)?;
        }
    }
    Ok(())
}

fn response_vector(
    values: &[String],
    norm: &str,
    dimension: usize,
    p: u32,
) -> std::result::Result<(), CacheError> {
    if values.len() != dimension {
        return Err(CacheError::InvalidManifest(
            "response vector dimension mismatch".into(),
        ));
    }
    let vector = parse_hp_vector(values, p)?;
    number(norm, p, true)?;
    if lossless_hp_decimal(&deterministic_l2_norm_hp(&vector, p)?) != norm {
        return Err(CacheError::InvalidManifest(
            "response vector norm mismatch".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sanity_prime(
    a: &PortablePrimePowerResponseAnalysis,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    state: &Float,
    xi: &[Float],
    roots: &[EigenvalueResult],
    first: usize,
    tau: &ArtifactManifest,
    eigenpair: &ArtifactManifest,
    root_range: &ArtifactManifest,
    secular: &ArtifactManifest,
    selection: &ContentDigest,
    spectral: &ResponseSpectralPreparation,
) -> std::result::Result<(), CacheError> {
    let p = cfg.precision_bits;
    let dimension = params.matrix_size();
    let content = prime_powers_up_to(params.lambda_sq_int());
    if a.schema_version != 2
        || a.lambda_squared != lambda_squared_cache_identity(params)
        || a.prime_cutoff != params.lambda_sq_int()
        || a.n_modes != params.n_modes
        || a.dimension != dimension
        || a.precision_bits != p
        || !payload_parity_matches(a.force_even, a.parity_policy, cfg.effective_parity_policy())
        || a.tau_content_digest != tau.content_digest.0
        || a.eigenpair_content_digest != eigenpair.content_digest.0
        || a.root_range_content_digest != root_range.content_digest.0
        || a.secular_source_content_digest != secular.content_digest.0
        || a.root_selection_digest != selection.0
        || a.normalization != PRIME_POWER_RESPONSE_NORMALIZATION
        || a.velocity_parameter != PRIME_POWER_RESPONSE_VELOCITY_PARAMETER
        || a.response_definition != PRIME_POWER_RESPONSE_DEFINITION
        || a.edge_jump_direction != PRIME_POWER_RESPONSE_EDGE_DIRECTION
        || a.state_eigenvalue != lossless_hp_decimal(state)
        || a.roots != ccm_response_roots(roots, first)?
        || a.events.len() != content.len()
    {
        return Err(CacheError::InvalidManifest(
            "CCM response identity mismatch".into(),
        ));
    }
    let unit = response_unit_state(xi, p)?;
    if a.spectral_isolation
        != response_spectral_isolation(spectral, params, cfg, state, &unit)
            .map_err(|e| CacheError::InvalidManifest(e.to_string()))?
    {
        return Err(CacheError::InvalidManifest(
            "CCM response source isolation mismatch".into(),
        ));
    }
    for (event, (power, prime, exponent)) in a.events.iter().zip(content) {
        if (event.power, event.prime, event.exponent) != (power, prime, exponent)
            || event.observation_is_event_edge
                != cutoff_equals_event_power(&a.lambda_squared, power)
                    .map_err(|e| CacheError::InvalidManifest(e.to_string()))?
        {
            return Err(CacheError::InvalidManifest(
                "CCM response event identity mismatch".into(),
            ));
        }
        for value in [
            &event.log_power,
            &event.von_mangoldt_weight,
            &event.reduced_position,
            &event.velocity_coefficient,
            &event.edge_jump_coefficient,
            &event.eigenvalue_velocity_response,
            &event.ccm_normalization_scale_velocity_response,
            &event.bordered_lagrange_multiplier,
        ] {
            number(value, p, false)?;
        }
        number(&event.projected_forcing_norm, p, true)?;
        let residual = number(&event.bordered_solve_relative_residual, p, true)?;
        if !weil_eigvec_cache::residual_within_precision_floor(&residual, p) {
            return Err(CacheError::InvalidManifest(
                "CCM response retained residual exceeds tolerance".into(),
            ));
        }
        response_vector(
            &event.l2_eigenvector_velocity_response,
            &event.l2_eigenvector_velocity_response_norm,
            dimension,
            p,
        )?;
        root_values(&event.root_velocity_responses, roots, p)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sanity_u_flow(
    a: &PortableUFlowResponseAnalysis,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    state: &Float,
    xi: &[Float],
    roots: &[EigenvalueResult],
    first: usize,
    tau: &ArtifactManifest,
    eigenpair: &ArtifactManifest,
    root_range: &ArtifactManifest,
    secular: &ArtifactManifest,
    selection: &ContentDigest,
    spectral: &ResponseSpectralPreparation,
) -> std::result::Result<(), CacheError> {
    let p = cfg.precision_bits;
    let dimension = params.matrix_size();
    if a.schema_version != 2
        || a.lambda_squared != lambda_squared_cache_identity(params)
        || a.prime_cutoff != params.lambda_sq_int()
        || a.active_prime_power_count != prime_powers_up_to(params.lambda_sq_int()).len()
        || a.n_modes != params.n_modes
        || a.dimension != dimension
        || a.precision_bits != p
        || !payload_parity_matches(a.force_even, a.parity_policy, cfg.effective_parity_policy())
        || a.tau_content_digest != tau.content_digest.0
        || a.eigenpair_content_digest != eigenpair.content_digest.0
        || a.root_range_content_digest != root_range.content_digest.0
        || a.secular_source_content_digest != secular.content_digest.0
        || a.root_selection_digest != selection.0
        || a.normalization != U_FLOW_RESPONSE_NORMALIZATION
        || a.velocity_parameter != U_FLOW_RESPONSE_VELOCITY_PARAMETER
        || a.derivative_convention != U_FLOW_RESPONSE_DERIVATIVE_CONVENTION
        || a.state_eigenvalue != lossless_hp_decimal(state)
        || a.roots != ccm_response_roots(roots, first)?
        || a.channels.len() != U_FLOW_CHANNELS.len()
        || a.normalization_target_velocity != lossless_hp_decimal(&response_target_velocity(l, p)?)
    {
        return Err(CacheError::InvalidManifest(
            "CCM u-flow response identity mismatch".into(),
        ));
    }
    let unit = response_unit_state(xi, p)?;
    if a.spectral_isolation
        != response_spectral_isolation(spectral, params, cfg, state, &unit)
            .map_err(|e| CacheError::InvalidManifest(e.to_string()))?
    {
        return Err(CacheError::InvalidManifest(
            "CCM u-flow source isolation mismatch".into(),
        ));
    }
    for (channel, expected) in a.channels.iter().zip(U_FLOW_CHANNELS) {
        if channel.channel != expected {
            return Err(CacheError::InvalidManifest(
                "CCM u-flow channel identity mismatch".into(),
            ));
        }
        for value in [
            &channel.eigenvalue_velocity_response,
            &channel.ccm_normalization_scale_velocity_response,
            &channel.bordered_lagrange_multiplier,
        ] {
            number(value, p, false)?;
        }
        number(&channel.projected_forcing_norm, p, true)?;
        let residual = number(&channel.bordered_solve_relative_residual, p, true)?;
        if !weil_eigvec_cache::residual_within_precision_floor(&residual, p) {
            return Err(CacheError::InvalidManifest(
                "CCM u-flow retained residual exceeds tolerance".into(),
            ));
        }
        response_vector(
            &channel.tau_velocity_action_on_state,
            &channel.tau_velocity_action_norm,
            dimension,
            p,
        )?;
        response_vector(
            &channel.l2_eigenvector_velocity_response,
            &channel.l2_eigenvector_velocity_response_norm,
            dimension,
            p,
        )?;
        root_values(&channel.fixed_pole_root_velocity_responses, roots, p)?;
    }
    root_values(&a.secular_pole_motion_root_velocity_responses, roots, p)?;
    root_values(&a.total_moving_pole_root_velocity_responses, roots, p)?;
    Ok(())
}
const GEOMETRY_BUDGET_BYTES: u128 = 512 * 1024 * 1024;
const ROOT_BATCH_SIZE: usize = 16;
struct PreparedRoot<'a> {
    value: &'a Float,
    geometry: root_response_math::Geometry,
}
pub(super) struct PreparedRootResponses<'a> {
    roots: Vec<Option<PreparedRoot<'a>>>,
    unit: &'a [Float],
    poles: &'a [Float],
    precision_bits: u32,
}
impl<'a> PreparedRootResponses<'a> {
    pub(super) fn new(
        unit: &'a [Float],
        poles: &'a [Float],
        roots: &'a [EigenvalueResult],
        bits: u32,
    ) -> Result<Self> {
        Self::with_budget(unit, poles, roots, bits, GEOMETRY_BUDGET_BYTES)
    }
    fn with_budget(
        unit: &'a [Float],
        poles: &'a [Float],
        roots: &'a [EigenvalueResult],
        bits: u32,
        budget: u128,
    ) -> Result<Self> {
        root_response_math::validate(unit, bits)?;
        root_response_math::validate(poles, bits)?;
        if unit.len() != poles.len() {
            bail!("root-response source dimensions differ");
        }
        let point_bytes = u128::from(bits + 4096).div_ceil(8) + 96;
        let bytes = roots.len() as u128 * poles.len() as u128 * point_bytes * 2;
        let retain = bytes <= budget.min(GEOMETRY_BUDGET_BYTES);
        let storage =
            roots.len() as u128 * (point_bytes * 4 + 128) + if retain { bytes } else { 0 };
        let workers = roots.len().min(rayon::current_num_threads()) as u128;
        if storage + workers * poles.len() as u128 * point_bytes * 24
            > super::source_working_budget()?
        {
            bail!("prepared root responses exceed combined workspace budget");
        }
        let prepared = roots
            .par_iter()
            .map(|outcome| {
                outcome
                    .value()
                    .map(|value| {
                        Ok(PreparedRoot {
                            value,
                            geometry: root_response_math::Geometry::prepare(
                                unit, poles, value, bits, 64, retain,
                            )?,
                        })
                    })
                    .transpose()
            })
            .collect::<Vec<Result<_>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            roots: prepared,
            unit,
            poles,
            precision_bits: bits,
        })
    }
    pub(super) fn values(&self, tangent: &[Float]) -> Result<Vec<Option<String>>> {
        root_response_math::validate(tangent, self.precision_bits)?;
        if tangent.len() != self.poles.len() {
            bail!("root-response tangent dimensions differ");
        }
        let chunks = self
            .roots
            .par_chunks(ROOT_BATCH_SIZE)
            .map(|roots| {
                roots
                    .iter()
                    .map(|root| {
                        root.as_ref()
                            .map(|root| {
                                root_response_math::evaluate(
                                    self.unit,
                                    tangent,
                                    self.poles,
                                    None,
                                    root.value,
                                    self.precision_bits,
                                    Some(&root.geometry),
                                )
                                .map(|x| lossless_hp_decimal(&x))
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        Ok(chunks.into_iter().flatten().collect())
    }
}

// Bound concurrent event workspaces while allowing each event's root batches
// and matrix rows to use the shared Rayon pool. Indexed collection fixes output
// order and the caller reports the first failure in canonical event order.
pub(super) fn event_chunk_size(events: usize) -> usize {
    events
        .div_ceil(rayon::current_num_threads().clamp(1, 16))
        .max(1)
}

pub(super) struct ResponseProgress {
    phase: &'static str,
    total: usize,
    started: Instant,
    state: std::sync::Mutex<(Instant, usize)>,
}
impl ResponseProgress {
    pub(super) fn new(phase: &'static str, total: usize, roots: usize) -> Self {
        let started = Instant::now();
        if total >= 64 {
            xc_core::progress_message!(
                "[HP] prime-power response {phase}: 0/{total} events; {roots} roots/event; {} workers",
                rayon::current_num_threads()
            );
        }
        Self {
            phase,
            total,
            started,
            state: std::sync::Mutex::new((started, 0)),
        }
    }
    pub(super) fn completed(&self) {
        let mut state = self.state.lock().expect("response progress mutex poisoned");
        state.1 += 1;
        if self.total >= 64 && (state.1 == self.total || state.0.elapsed().as_secs() >= 30) {
            xc_core::progress_message!(
                "[HP] prime-power response {}: {}/{} events; elapsed {:.1}s",
                self.phase,
                state.1,
                self.total,
                self.started.elapsed().as_secs_f64()
            );
            state.0 = Instant::now();
        }
    }
}

/// A process-local seal for the exact freshly computed response. Its producer
/// already ran spectral-isolation and bordered-residual gates. Cache reads never
/// receive this seal and must replay the numerical validator. Stream JSON so an
/// additional multi-gigabyte copy is never needed just to bind this witness.
#[derive(Default)]
pub(super) struct FreshResponseSeal(RefCell<Option<ContentDigest>>);
impl FreshResponseSeal {
    fn digest(value: &impl Serialize) -> std::result::Result<ContentDigest, CacheError> {
        use sha2::{Digest, Sha256};
        use std::io::{BufWriter, Write};
        struct HashWriter(Sha256);
        impl Write for HashWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut output = BufWriter::with_capacity(256 * 1024, HashWriter(Sha256::new()));
        serde_json::to_writer(&mut output, value)
            .map_err(|error| CacheError::Serialization(error.to_string()))?;
        output
            .flush()
            .map_err(|error| CacheError::Serialization(error.to_string()))?;
        let hash = output
            .into_inner()
            .map_err(|error| CacheError::Serialization(error.to_string()))?;
        Ok(ContentDigest(format!("{:x}", hash.0.finalize())))
    }
    pub(super) fn record(&self, value: &impl Serialize) -> std::result::Result<(), CacheError> {
        self.0.replace(Some(Self::digest(value)?));
        Ok(())
    }
    pub(super) fn verify_fresh(
        &self,
        value: &impl Serialize,
    ) -> std::result::Result<bool, CacheError> {
        let stored = self.0.borrow();
        let Some(expected) = stored.as_ref() else {
            return Ok(false);
        };
        if Self::digest(value)? != *expected {
            return Err(CacheError::InvalidManifest(
                "fresh response changed after its numerical production gates".into(),
            ));
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn outcome(value: Float) -> EigenvalueResult {
        let bits = value.prec();
        EigenvalueResult::Converged(RootRefinement {
            value,
            diagnostics: RootRefinementDiagnostics {
                iterations: 1,
                final_correction: Float::with_val(bits, 0),
                residual: Float::with_val(bits, 0),
                achieved_decimal_digits: Float::with_val(bits, 20),
            },
        })
    }
    fn source(
        bits: u32,
        n: usize,
        count: usize,
    ) -> (Vec<Float>, Vec<Float>, Vec<EigenvalueResult>, Vec<Float>) {
        let poles = (-(n as i64)..=n as i64)
            .map(|i| Float::with_val(bits, i))
            .collect::<Vec<_>>();
        let unit = poles
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut value = Float::with_val(bits, i + 1);
                value /= 2 * n + 3;
                value
            })
            .collect::<Vec<_>>();
        let tangent = unit
            .iter()
            .enumerate()
            .map(|(i, x)| if i % 2 == 0 { x.clone() } else { -x.clone() })
            .collect();
        let roots = (0..count)
            .map(|i| {
                let mut x = Float::with_val(bits, i + 1);
                x /= 8;
                x += n + 1;
                outcome(x)
            })
            .collect();
        (unit, poles, roots, tangent)
    }
    fn reference(
        unit: &[Float],
        poles: &[Float],
        roots: &[EigenvalueResult],
        tangent: &[Float],
        bits: u32,
    ) -> Vec<Option<String>> {
        roots
            .iter()
            .map(|root| {
                root.value().map(|value| {
                    lossless_hp_decimal(
                        &prime_power_root_velocity_response(unit, tangent, poles, value, bits)
                            .unwrap(),
                    )
                })
            })
            .collect()
    }

    #[test]
    fn exhaustive_response_range_prepared_extreme_scale() {
        let p = 128;
        for e in [-700_000_000i32, 700_000_000] {
            let h = Float::with_val(p, 1) << e;
            let unit = [Float::with_val(p, 1), Float::with_val(p, 1)];
            let tangent = [Float::with_val(p, 1), Float::with_val(p, -1)];
            let poles = [-h.clone(), h.clone()];
            let roots = [outcome(Float::with_val(p, 0))];
            for budget in [0, GEOMETRY_BUDGET_BYTES] {
                let prepared =
                    PreparedRootResponses::with_budget(&unit, &poles, &roots, p, budget).unwrap();
                assert_eq!(
                    prepared.values(&tangent).unwrap(),
                    vec![Some(lossless_hp_decimal(&h))]
                );
            }
        }
    }
    #[test]
    fn exhaustive_response_range_scalar_extreme_scale() {
        let p = 128;
        for e in [-700_000_000i32, 700_000_000] {
            let h = Float::with_val(p, 1) << e;
            let unit = [Float::with_val(p, 1), Float::with_val(p, 1)];
            let tangent = [Float::with_val(p, 1), Float::with_val(p, -1)];
            let poles = [-h.clone(), h.clone()];
            let root = Float::with_val(p, 0);
            assert_eq!(
                prime_power_root_velocity_response(&unit, &tangent, &poles, &root, p).unwrap(),
                h
            );
        }
    }
    #[test]
    fn exhaustive_response_range_moving_poles_preserve_finite_translation() {
        let p = 128;
        let big = Float::with_val(p, 1) << 700_000_000u32;
        let unit = [big.clone(), big.clone()];
        let tangent = vec![Float::with_val(p, 0); 2];
        let poles = [Float::with_val(p, -1), Float::with_val(p, 1)];
        let root = Float::with_val(p, 0);
        assert_eq!(
            secular_root_velocity_response(
                &unit,
                &tangent,
                &poles,
                &[big.clone(), big.clone()],
                &root,
                p
            )
            .unwrap(),
            big
        );
    }
    #[test]
    fn exhaustive_response_range_prepared_cancellation() {
        let p = 64;
        let big = Float::with_val(p, 1) << 100u32;
        let unit = [
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 1),
        ];
        let tangent = [Float::with_val(p, &big * 3u32), Float::with_val(p, 2), -big];
        let poles = [
            Float::with_val(p, -1),
            Float::with_val(p, 0),
            Float::with_val(p, 1),
        ];
        let roots = [outcome(Float::with_val(p, 2))];
        let prepared = PreparedRootResponses::new(&unit, &poles, &roots, p).unwrap();
        assert_eq!(
            prepared.values(&tangent).unwrap(),
            vec![Some(lossless_hp_decimal(&Float::with_val(p, 1)))]
        );
    }
    #[test]
    fn exhaustive_response_range_precision_preflight_is_fallible() {
        let unit = [Float::with_val(128, 1)];
        let poles = [Float::with_val(128, 0)];
        let roots = [outcome(Float::with_val(128, 1))];
        assert!(
            std::panic::catch_unwind(|| PreparedRootResponses::new(&unit, &poles, &roots, 0))
                .is_ok_and(|x| x.is_err())
        );
        assert!(PreparedRootResponses::new(&unit, &poles, &roots, 64).is_err());
    }
    #[test]
    fn exhaustive_response_range_nonfinite_source_rejected() {
        let unit = [Float::with_val(128, rug::float::Special::Nan)];
        let poles = [Float::with_val(128, 0)];
        let roots = [outcome(Float::with_val(128, 1))];
        assert!(PreparedRootResponses::new(&unit, &poles, &roots, 128).is_err());
    }
    #[test]
    fn exhaustive_response_range_nonfinite_tangent_rejected() {
        let unit = [Float::with_val(128, 1)];
        let poles = [Float::with_val(128, 0)];
        let roots = [outcome(Float::with_val(128, 1))];
        let prepared = PreparedRootResponses::new(&unit, &poles, &roots, 128).unwrap();
        assert!(prepared
            .values(&[Float::with_val(128, rug::float::Special::Nan)])
            .is_err());
    }
    #[test]
    fn exhaustive_response_range_unrepresentable_result_is_error() {
        let unit = [Float::with_val(128, 1)];
        let poles = [Float::with_val(128, 0)];
        let roots = [outcome(Float::with_val(128, 2))];
        let prepared = PreparedRootResponses::new(&unit, &poles, &roots, 128).unwrap();
        let tangent = [Float::with_val(128, 0.75) << rug::float::exp_max()];
        assert!(prepared.values(&tangent).is_err());
    }
    #[test]
    fn prepared_root_responses_preserve_bits_threads_and_budget_fallback() {
        for bits in [128, 192, 1024, 3386, 6708] {
            let (unit, poles, mut roots, tangent) = source(bits, 8, 35);
            let mut near_pole = Float::with_val(bits, 2).pow(-((bits / 3) as i32));
            near_pole += 2;
            roots.push(outcome(near_pole));
            let mut negative = Float::with_val(bits, -17);
            negative /= 3;
            roots.push(outcome(negative));
            roots.insert(
                17,
                EigenvalueResult::Failed {
                    iterations: 0,
                    reason: "retained failure".into(),
                },
            );
            let expected = reference(&unit, &poles, &roots, &tangent, bits);
            for workers in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                for budget in [0, GEOMETRY_BUDGET_BYTES] {
                    pool.install(|| {
                        let prepared =
                            PreparedRootResponses::with_budget(&unit, &poles, &roots, bits, budget)
                                .unwrap();
                        assert_eq!(prepared.values(&tangent).unwrap(), expected);
                        let zero = vec![Float::with_val(bits, 0); unit.len()];
                        assert_eq!(
                            prepared.values(&zero).unwrap(),
                            reference(&unit, &poles, &roots, &zero, bits)
                        );
                    });
                }
            }
        }
    }
    #[test]
    fn prepared_root_responses_preserve_invalid_geometry_rejection() {
        let (unit, poles, roots, tangent) = source(192, 2, 1);
        let at_pole = vec![outcome(poles[0].clone())];
        assert!(PreparedRootResponses::new(&unit, &poles, &at_pole, 192)
            .err()
            .unwrap()
            .to_string()
            .contains("secular pole"));
        let zero = vec![Float::with_val(192, 0); unit.len()];
        assert!(PreparedRootResponses::new(&zero, &poles, &roots, 192).is_err());
        assert!(PreparedRootResponses::new(&unit[..2], &poles, &roots, 192).is_err());
        let prepared = PreparedRootResponses::new(&unit, &poles, &roots, 192).unwrap();
        assert!(prepared.values(&tangent[..2]).is_err());
    }
    #[test]
    fn fresh_seal_is_exact_and_never_available_for_cached_or_altered_data() {
        let value = serde_json::json!({"source":"A", "roots":["0", "-0", "1e-2000"], "nested":{"response":"9"}});
        let seal = FreshResponseSeal::default();
        assert!(!seal.verify_fresh(&value).unwrap());
        assert_eq!(
            FreshResponseSeal::digest(&value).unwrap(),
            ContentDigest::sha256(&serde_json::to_vec(&value).unwrap())
        );
        seal.record(&value).unwrap();
        assert!(seal.verify_fresh(&value).unwrap());
        for key in ["source", "roots", "nested"] {
            let mut changed = value.clone();
            changed[key] = serde_json::Value::Null;
            assert!(seal.verify_fresh(&changed).is_err());
        }
        assert!(!FreshResponseSeal::default().verify_fresh(&value).unwrap());
    }
    #[test]
    #[ignore = "explicit bounded release-mode response-kernel benchmark"]
    fn response_root_kernel_benchmark() {
        let bits = 6708;
        let (unit, poles, roots, tangent) = source(bits, 400, 400);
        let repeats = 3;
        let started = Instant::now();
        let expected = (0..repeats)
            .map(|_| reference(&unit, &poles, &roots, &tangent, bits))
            .collect::<Vec<_>>();
        let baseline_seconds = started.elapsed().as_secs_f64();
        let expected_digest = ContentDigest::sha256(&serde_json::to_vec(&expected).unwrap());
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let preparation = Instant::now();
                let prepared = PreparedRootResponses::new(&unit, &poles, &roots, bits).unwrap();
                let preparation_seconds = preparation.elapsed().as_secs_f64();
                let started = Instant::now();
                let actual = (0..repeats).map(|_| prepared.values(&tangent).unwrap()).collect::<Vec<_>>();
                let candidate_seconds = started.elapsed().as_secs_f64();
                assert_eq!(actual, expected);
                eprintln!("RESPONSE_BENCH {}", serde_json::json!({"bits":bits,"dimension":801,"roots":400,"events":repeats,"workers":workers,"baseline_seconds":baseline_seconds,"preparation_seconds":preparation_seconds,"candidate_seconds":candidate_seconds,"payload_sha256":expected_digest.0,"bit_identical":true}));
            });
        }
    }
}
