//! Private restart batches for fully completed prime-power response events.
//!
//! These are trusted local execution snapshots, not mathematical certificates.
//! A batch contains one event, after all of its production numerical gates.
//! Its source-bound seal and canonical scalar checks are independent of the
//! portable artifact encoding, which is unchanged.
use super::*;
use crate::ccm::capture_runtime::{CaptureResourcePolicy, Checkpoints};

pub(super) fn allowed(
    mode: xc_cache::ArtifactExecutionCacheMode,
    assurance: xc_core::AssuranceLevel,
) -> bool {
    mode == xc_cache::ArtifactExecutionCacheMode::PreferReuse
        && assurance == xc_core::AssuranceLevel::Computed
}

pub(super) fn identity(
    semantic: &SemanticKeyEnvelope,
    cfg: &HighPrecConfig,
    l: &Float,
    state: &Float,
    xi: &[Float],
) -> Result<serde_json::Value> {
    // Bind operational policies as well as the mathematical request. A guard
    // or backend change must not turn a previously resolved event into a bypass
    // of the newly requested execution route.
    let policy = std::env::vars_os()
        .filter_map(|(name, value)| {
            let name = name.to_str()?;
            (name.starts_with("XC_PRIME_RESPONSE_")
                || name.starts_with("XC_ROOT_RESPONSE_")
                || name.starts_with("XC_CUDA_")
                || name.starts_with("XC_DENSE_")
                || name == "XC_RESPONSE_EVENT_BATCH")
                .then(|| {
                    value
                        .into_string()
                        .map(|value| (name.to_owned(), value))
                        .map_err(|_| {
                            anyhow::anyhow!("response checkpoint policy {name} is not Unicode")
                        })
                })
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(serde_json::json!({
        "semantics": "ccm-private-completed-prime-response-events-v1",
        "request": semantic,
        "precision_bits": cfg.precision_bits,
        "root_guard_bits": root_response_math::GuardPolicy::configured(cfg.precision_bits)?.maximum_guard_bits(),
        "policy": policy,
        "state": hp_values_memo_identity(xi),
        "eigenvalue": hp_values_memo_identity(std::slice::from_ref(state)),
        "length": hp_values_memo_identity(std::slice::from_ref(l)),
        "mpfr_range": [rug::float::exp_min(), rug::float::exp_max()],
    }))
}

pub(super) fn configured(
    semantic: &SemanticKeyEnvelope,
    cfg: &HighPrecConfig,
    l: &Float,
    state: &Float,
    xi: &[Float],
) -> Option<Checkpoints> {
    match identity(semantic, cfg, l, state, xi).and_then(|key| Checkpoints::new(&key)) {
        Ok(store) => Some(store),
        Err(error) => {
            xc_core::progress_message!("prime-response event checkpoints unavailable: {error}");
            None
        }
    }
}

// Fixed-count strings prevent a malformed sealed file from allocating an
// unbounded Vec of tiny strings before its planned dimensions are checked.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletedEvent {
    schema_version: u32,
    precision_bits: u32,
    event: (u64, u64, u32),
    edge: bool,
    scalars: [String; 11],
    vector: String,
    roots: String,
}

impl CompletedEvent {
    fn encode(event: &PortablePrimePowerResponseEvent, p: u32) -> Self {
        Self {
            schema_version: 1,
            precision_bits: p,
            event: (event.power, event.prime, event.exponent),
            edge: event.observation_is_event_edge,
            scalars: [
                event.log_power.clone(),
                event.von_mangoldt_weight.clone(),
                event.reduced_position.clone(),
                event.velocity_coefficient.clone(),
                event.edge_jump_coefficient.clone(),
                event.eigenvalue_velocity_response.clone(),
                event.projected_forcing_norm.clone(),
                event.l2_eigenvector_velocity_response_norm.clone(),
                event.ccm_normalization_scale_velocity_response.clone(),
                event.bordered_lagrange_multiplier.clone(),
                event.bordered_solve_relative_residual.clone(),
            ],
            vector: event.l2_eigenvector_velocity_response.join("\n"),
            roots: event
                .root_velocity_responses
                .iter()
                .map(|value| value.as_deref().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn decode(
        self,
        expected: (u64, u64, u32),
        edge: bool,
        dimension: usize,
        roots: &[EigenvalueResult],
        p: u32,
    ) -> Result<PortablePrimePowerResponseEvent> {
        anyhow::ensure!(
            self.schema_version == 1
                && self.precision_bits == p
                && self.event == expected
                && self.edge == edge
                && self.vector.split('\n').count() == dimension
                && self.roots.split('\n').count() == roots.len(),
            "completed response event shape or identity mismatch"
        );
        let canonical = |text: &str| -> Result<Float> {
            anyhow::ensure!(!text.is_empty() && text.len() <= p as usize + 128);
            let value = parse_hp_scalar(text, p)?;
            anyhow::ensure!(value.is_finite() && lossless_hp_decimal(&value) == text);
            Ok(value)
        };
        for (index, scalar) in self.scalars.iter().enumerate() {
            let value = canonical(scalar)?;
            if [6, 7, 10].contains(&index) {
                anyhow::ensure!(value >= 0, "negative response norm or residual");
            }
            if index == 10 {
                anyhow::ensure!(
                    weil_eigvec_cache::residual_within_precision_floor(&value, p),
                    "completed response residual gate failed"
                );
            }
        }
        let vector = self
            .vector
            .split('\n')
            .map(|text| -> Result<Float> {
                // Portable vectors use Float::to_string, while the scalar
                // fields use lossless_hp_decimal. Preserve both encodings.
                anyhow::ensure!(!text.is_empty() && text.len() <= p as usize + 128);
                let value = parse_hp_scalar(text, p)?;
                anyhow::ensure!(value.is_finite() && value.to_string() == text);
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            lossless_hp_decimal(&deterministic_l2_norm_hp(&vector, p)?) == self.scalars[7],
            "completed response vector norm mismatch"
        );
        for (text, root) in self.roots.split('\n').zip(roots) {
            anyhow::ensure!(!text.is_empty() == root.value().is_some());
            if !text.is_empty() {
                canonical(text)?;
            }
        }
        let [log_power, von_mangoldt_weight, reduced_position, velocity_coefficient, edge_jump_coefficient, eigenvalue_velocity_response, projected_forcing_norm, l2_eigenvector_velocity_response_norm, ccm_normalization_scale_velocity_response, bordered_lagrange_multiplier, bordered_solve_relative_residual] =
            self.scalars;
        Ok(PortablePrimePowerResponseEvent {
            power: expected.0,
            prime: expected.1,
            exponent: expected.2,
            log_power,
            von_mangoldt_weight,
            reduced_position,
            velocity_coefficient,
            edge_jump_coefficient,
            observation_is_event_edge: edge,
            eigenvalue_velocity_response,
            projected_forcing_norm,
            l2_eigenvector_velocity_response_norm,
            l2_eigenvector_velocity_response: self.vector.split('\n').map(str::to_owned).collect(),
            ccm_normalization_scale_velocity_response,
            bordered_lagrange_multiplier,
            bordered_solve_relative_residual,
            root_velocity_responses: self
                .roots
                .split('\n')
                .map(|s| (!s.is_empty()).then(|| s.to_owned()))
                .collect(),
        })
    }
}

fn admit(
    dimension: usize,
    roots: usize,
    p: u32,
) -> Option<(u64, Option<xc_core::MemoryReservation>)> {
    let cap = 4096u64.checked_add(
        u64::try_from(dimension.checked_add(roots)?.checked_add(12)?)
            .ok()?
            .checked_mul(u64::from(p) + 128)?,
    )?;
    // Fixed-count text, decoded scalar storage, duplicate output text, and the
    // checkpoint reader's 1 MiB buffer are admitted before any payload read.
    let workspace = cap.checked_mul(6)?.checked_add(1 << 20)?;
    if workspace
        > CaptureResourcePolicy::from_environment()
            .ok()?
            .maximum_working_bytes
    {
        return None;
    }
    let reservation = match xc_core::RunResources::current() {
        Some(resources) => Some(resources.try_reserve_working(workspace).ok()?),
        None => None,
    };
    Some((cap, reservation))
}

pub(super) fn load(
    store: &Checkpoints,
    expected: (u64, u64, u32),
    edge: bool,
    dimension: usize,
    roots: &[EigenvalueResult],
    p: u32,
) -> Option<PortablePrimePowerResponseEvent> {
    if !store.enabled() {
        return None;
    }
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range().ok()?;
    let (cap, _reservation) = admit(dimension, roots.len(), p)?;
    let label = format!("completed-prime-response-{}", expected.0);
    match store.limited_bytes(cap).load::<CompletedEvent>(&label) {
        Ok(Some(saved)) => match saved.decode(expected, edge, dimension, roots, p) {
            Ok(event) => Some(event),
            Err(error) => {
                xc_core::progress_message!(
                    "prime-response event {} checkpoint rejected ({error}); recomputing",
                    expected.0
                );
                None
            }
        },
        Ok(None) => None,
        Err(error) => {
            xc_core::progress_message!(
                "prime-response event {} checkpoint unavailable ({error}); recomputing",
                expected.0
            );
            None
        }
    }
}

pub(super) fn save(store: &Checkpoints, event: &PortablePrimePowerResponseEvent, p: u32) {
    if !store.enabled() {
        return;
    }
    let Some((cap, _reservation)) = admit(
        event.l2_eigenvector_velocity_response.len(),
        event.root_velocity_responses.len(),
        p,
    ) else {
        return;
    };
    if let Err(error) = store.limited_bytes(cap).save(
        &format!("completed-prime-response-{}", event.power),
        &CompletedEvent::encode(event, p),
    ) {
        xc_core::progress_message!(
            "prime-response event {} checkpoint not saved ({error})",
            event.power
        );
    }
}

#[cfg(test)]
thread_local! {
    static INTERRUPT_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static COMPLETED_EVENTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn event_completed() -> Result<()> {
    let count = COMPLETED_EVENTS.with(|count| {
        count.set(count.get() + 1);
        count.get()
    });
    if INTERRUPT_AFTER.with(|stop| stop.get()) == Some(count) {
        bail!("injected response interruption after completed event");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        params: CcmParams,
        cfg: HighPrecConfig,
        l: Float,
        state: Float,
        xi: Vec<Float>,
        tau: Vec<Float>,
        roots: Vec<EigenvalueResult>,
        manifests: [ArtifactManifest; 4],
        preparation: ResponseSpectralPreparation,
        semantic: SemanticKeyEnvelope,
    }

    impl Fixture {
        fn new() -> Self {
            let bits = 192;
            let params = CcmParams::from_lambda_sq_integer(13, 2);
            let mut cfg = HighPrecConfig::for_decimal_digits(40);
            cfg.precision_bits = bits;
            let l = log_lambda_sq_hp(&params, bits).unwrap();
            let dimension = params.matrix_size();
            let mut tau = vec![Float::with_val(bits, 0); dimension * dimension];
            for index in 0..dimension {
                let offset = index.abs_diff(params.n_modes);
                tau[index * dimension + index] = Float::with_val(bits, 1 + offset * offset);
            }
            let state = Float::with_val(bits, 1);
            let mut xi = vec![Float::with_val(bits, 0); dimension];
            xi[params.n_modes] = l.clone().sqrt();
            let roots = (0..2)
                .map(|i| {
                    EigenvalueResult::Converged(RootRefinement {
                        value: Float::with_val(bits, i + 3) + Float::with_val(bits, 0.25),
                        diagnostics: RootRefinementDiagnostics {
                            iterations: 1,
                            final_correction: Float::with_val(bits, 0),
                            residual: Float::with_val(bits, 0),
                            achieved_decimal_digits: Float::with_val(bits, 40),
                        },
                    })
                })
                .collect();
            let make = super::super::tests::conditioning_test_manifest;
            let manifests = [
                make("ccm_tau_matrix", "event-checkpoint-tau"),
                make("ccm_weil_eigenpair", "event-checkpoint-state"),
                make("ccm_root_discovery_window", "event-checkpoint-roots"),
                make("ccm_secular_source", "event-checkpoint-secular"),
            ];
            let preparation = compute_response_spectral_preparation(&params, &cfg, &tau).unwrap();
            let semantic = SemanticKeyEnvelope {
                schema_version: 1,
                artifact_kind: "event-checkpoint-test".into(),
                mathematical_semantics_version: "1".into(),
                resolved_mathematical_parameters: serde_json::json!({"C": 13, "N": 2}),
                normalization: None,
                target: None,
                subspace: None,
                source_data_identities: BTreeMap::from([(
                    "tau".into(),
                    manifests[0].content_digest.clone(),
                )]),
                algorithm_semantics: Some("checkpoint-test-v1".into()),
            };
            Self {
                params,
                cfg,
                l,
                state,
                xi,
                tau,
                roots,
                manifests,
                preparation,
                semantic,
            }
        }

        fn key(&self) -> serde_json::Value {
            identity(&self.semantic, &self.cfg, &self.l, &self.state, &self.xi).unwrap()
        }

        fn run(&self, store: Option<&Checkpoints>) -> Result<PortablePrimePowerResponseAnalysis> {
            compute_prime_power_response_analysis_with_policy(
                &self.params,
                &self.cfg,
                &self.l,
                &self.tau,
                &self.state,
                &self.xi,
                &self.roots,
                1,
                &self.manifests[0],
                &self.manifests[1],
                &self.manifests[2],
                &self.manifests[3],
                &root_selection_digest(&self.roots)?,
                &self.preparation,
                None,
                PrimeResponseActionPolicy::Prepared {
                    batch_size: 3,
                    optional_workspace_bytes: 1 << 29,
                },
                store,
            )
        }

        fn verify(&self, artifact: &PortablePrimePowerResponseAnalysis) {
            validate_prime_power_response_analysis(
                artifact,
                &self.params,
                &self.cfg,
                &self.l,
                &self.tau,
                &self.state,
                &self.xi,
                &self.roots,
                1,
                &self.manifests[0],
                &self.manifests[1],
                &self.manifests[2],
                &self.manifests[3],
                &root_selection_digest(&self.roots).unwrap(),
                &self.preparation,
            )
            .unwrap();
        }

        fn reference(&self) -> PortablePrimePowerResponseAnalysis {
            response_performance_reference::reference_prime_power_response(
                &self.params,
                &self.cfg,
                &self.l,
                &self.tau,
                &self.state,
                &self.xi,
                &self.roots,
                1,
                &self.manifests[0],
                &self.manifests[1],
                &self.manifests[2],
                &self.manifests[3],
                &root_selection_digest(&self.roots).unwrap(),
                &self.preparation,
            )
            .unwrap()
        }
    }

    struct InterruptReset;
    impl Drop for InterruptReset {
        fn drop(&mut self) {
            INTERRUPT_AFTER.with(|stop| stop.set(None));
        }
    }

    #[test]
    fn response_event_checkpoints_resume_exactly_after_interruption_and_reject_corruption() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::new(1 << 30).unwrap();
            let _scope = resources.enter();
            let fixture = Fixture::new();
            let directory = xc_core::test_support::TestDir::new("prime-event-checkpoints");
            let store = Checkpoints::local(&fixture.key(), directory.to_path_buf()).unwrap();
            let started = Instant::now();
            let fresh = fixture.run(None).unwrap();
            let fresh_seconds = started.elapsed().as_secs_f64();
            let bytes = serde_json::to_vec(&fresh).unwrap();
            assert_eq!(bytes, serde_json::to_vec(&fixture.reference()).unwrap());
            fixture.verify(&fresh);
            let _reset = InterruptReset;
            COMPLETED_EVENTS.with(|count| count.set(0));
            INTERRUPT_AFTER.with(|stop| stop.set(Some(2)));
            let error = fixture.run(Some(&store)).unwrap_err();
            assert!(error.to_string().contains("injected response interruption"));
            assert_eq!(COMPLETED_EVENTS.with(std::cell::Cell::get), 2);
            INTERRUPT_AFTER.with(|stop| stop.set(None));
            COMPLETED_EVENTS.with(|count| count.set(0));
            let started = Instant::now();
            let resumed = fixture.run(Some(&store)).unwrap();
            let resumed_seconds = started.elapsed().as_secs_f64();
            assert_eq!(
                COMPLETED_EVENTS.with(std::cell::Cell::get),
                fresh.events.len() - 2
            );
            assert_eq!(serde_json::to_vec(&resumed).unwrap(), bytes);
            fixture.verify(&resumed);
            COMPLETED_EVENTS.with(|count| count.set(0));
            let started = Instant::now();
            assert_eq!(
                serde_json::to_vec(&fixture.run(Some(&store)).unwrap()).unwrap(),
                bytes
            );
            let warm_seconds = started.elapsed().as_secs_f64();
            assert_eq!(COMPLETED_EVENTS.with(std::cell::Cell::get), 0);
            eprintln!("response checkpoint qualification: C=13 N=2 p=192 events={} resumed_events=2 fresh_seconds={fresh_seconds:.6} resumed_seconds={resumed_seconds:.6} fully_reused_seconds={warm_seconds:.6} mathematical_json_sha256={}", fresh.events.len(), ContentDigest::sha256(&bytes).0);

            // Tamper the first event without updating its seal: no later saved
            // event is used across this gap, and the canonical cold path wins.
            let key_dir = std::fs::read_dir(&directory)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            let label = format!("completed-prime-response-{}", fresh.events[0].power);
            let payload = key_dir.join(ContentDigest::sha256(label.as_bytes()).0);
            let mut damaged = std::fs::read(&payload).unwrap();
            damaged[0] = b'!';
            std::fs::write(payload, damaged).unwrap();
            COMPLETED_EVENTS.with(|count| count.set(0));
            let repaired = fixture.run(Some(&store)).unwrap();
            assert_eq!(
                COMPLETED_EVENTS.with(std::cell::Cell::get),
                fresh.events.len()
            );
            assert_eq!(serde_json::to_vec(&repaired).unwrap(), bytes);
            fixture.verify(&repaired);
        });
    }

    #[test]
    fn response_event_checkpoints_bind_sources_and_reject_invalid_sealed_payloads() {
        let fixture = Fixture::new();
        let directory = xc_core::test_support::TestDir::new("prime-event-binding");
        let key = fixture.key();
        let store = Checkpoints::local(&key, directory.to_path_buf()).unwrap();
        let fresh = fixture.run(None).unwrap();
        let event = &fresh.events[0];
        save(&store, event, fixture.cfg.precision_bits);
        let expected = (event.power, event.prime, event.exponent);
        let read = |store: &Checkpoints| {
            load(
                store,
                expected,
                event.observation_is_event_edge,
                fixture.params.matrix_size(),
                &fixture.roots,
                fixture.cfg.precision_bits,
            )
        };
        assert_eq!(read(&store), Some(event.clone()));
        for field in [
            "state",
            "eigenvalue",
            "length",
            "policy",
            "request",
            "root_guard_bits",
            "mpfr_range",
        ] {
            let mut other = key.clone();
            other[field] = serde_json::json!("changed");
            let other_store = Checkpoints::local(&other, directory.to_path_buf()).unwrap();
            assert!(
                read(&other_store).is_none(),
                "{field} must invalidate reuse"
            );
        }
        for mutation in 0..5 {
            let mut encoded = CompletedEvent::encode(event, fixture.cfg.precision_bits);
            match mutation {
                0 => encoded.precision_bits += 1,
                1 => encoded.event.0 += 1,
                2 => encoded.vector.push_str("\n0"),
                3 => encoded.scalars[10] = "1".to_owned(),
                _ => encoded.roots = "NaN\nNaN".to_owned(),
            }
            store
                .save(
                    &format!("completed-prime-response-{}", event.power),
                    &encoded,
                )
                .unwrap();
            assert!(read(&store).is_none());
        }
        let tiny = store.limited_bytes(1);
        assert!(read(&tiny).is_none());
        let resources = xc_core::RunResources::new(1024).unwrap();
        let _scope = resources.enter();
        assert!(read(&store).is_none());
    }

    #[test]
    fn response_event_checkpoints_are_only_computed_reuse() {
        use xc_cache::ArtifactExecutionCacheMode as Mode;
        use xc_core::AssuranceLevel as Assurance;
        assert!(allowed(Mode::PreferReuse, Assurance::Computed));
        for mode in [
            Mode::Disabled,
            Mode::Refresh,
            Mode::VerifyAgainstReference,
            Mode::RequireReuse,
        ] {
            assert!(!allowed(mode, Assurance::Computed));
        }
        for assurance in [Assurance::CrossChecked, Assurance::Certified] {
            assert!(!allowed(Mode::PreferReuse, assurance));
        }
    }
}
