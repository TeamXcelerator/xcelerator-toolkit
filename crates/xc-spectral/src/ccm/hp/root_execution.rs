//! Scheduling and private completion records for the unchanged root window.
use super::*;

/// Conservative scratch estimate for promoted source/pole vectors, directed
/// evaluation, and one root's portable conversion. It is not an RSS bound.
fn task_bytes(source_len: usize, cfg: &HighPrecConfig) -> u64 {
    let precision = if cfg.root_precision_policy == RootPrecisionPolicy::Adaptive {
        cfg.precision_bits
            .saturating_add(cfg.root_maximum_extra_precision_bits)
    } else {
        cfg.precision_bits.saturating_add(
            FIXED_CORRECTION_GUARDS
                .iter()
                .copied()
                .max()
                .unwrap_or(GUARD_BITS),
        )
    };
    runtime_float_bytes(source_len.saturating_mul(16).saturating_add(256), precision)
        .saturating_mul(2)
}

/// Indexed chunks bound simultaneously active scratch without creating a pool.
/// The owner reserves before dispatch, so workers never wait on their owner.
pub(super) fn ordered_map<T: Send>(
    count: usize,
    source_len: usize,
    cfg: &HighPrecConfig,
    f: impl Fn(usize) -> T + Sync,
) -> std::result::Result<Vec<T>, CacheError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let resources = xc_core::RunResources::current();
    let pool_workers = rayon::current_num_threads();
    let mut lanes = count.min(pool_workers);
    if resources
        .as_ref()
        .is_some_and(|r| r.check_workers(pool_workers).is_err())
        || (lanes > 1 && xc_numerics::mpfr_interval::ensure_uniform_exponent_range().is_err())
    {
        // The original serial validator is valid under the caller's range.
        lanes = 1;
    }
    let _reservation = if let Some(resources) = resources {
        let bytes = task_bytes(source_len, cfg);
        loop {
            match resources.try_reserve_working(bytes.saturating_mul(lanes as u64)) {
                Ok(reservation) => break Some(reservation),
                Err(error) if lanes == 1 => {
                    // Optional decoded/source retention must not turn a
                    // one-root workspace that fits into a numerical refusal.
                    // Active shared handles keep their reservations after clear.
                    if let Some(memo) = xc_cache::ValidatedValueMemo::current() {
                        memo.clear();
                        match resources.try_reserve_working(bytes) {
                            Ok(reservation) => break Some(reservation),
                            Err(error) => {
                                return Err(CacheError::InvalidTransition(error.to_string()));
                            }
                        }
                    }
                    return Err(CacheError::InvalidTransition(error.to_string()));
                }
                Err(_) => lanes -= 1,
            }
        }
    } else {
        None
    };
    if lanes == 1 {
        return Ok((0..count).map(f).collect());
    }
    let chunk = count.div_ceil(lanes);
    Ok((0..count.div_ceil(chunk))
        .into_par_iter()
        .map(|part| {
            let start = part * chunk;
            (start..(start + chunk).min(count))
                .map(&f)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect())
}

pub(super) fn portable(
    computed: &ComputedRootOutcome,
    solver: CcmEigenstateSolver,
) -> PortableRootOutcome {
    let encode = |result: &RootRefinement| match computed.adaptive_precision.clone() {
        Some(evidence) => PortableRootRefinement::from_runtime_adaptive(result, evidence),
        None => PortableRootRefinement::from_runtime_for_eigenstate_solver(result, solver),
    };
    match &computed.outcome {
        EigenvalueResult::Converged(result) => PortableRootOutcome::Converged(encode(result)),
        EigenvalueResult::Stagnated(result) => PortableRootOutcome::Stagnated(encode(result)),
        EigenvalueResult::Approximate(result) => PortableRootOutcome::Approximate(encode(result)),
        EigenvalueResult::Failed { iterations, reason } => PortableRootOutcome::Failed {
            iterations: *iterations,
            reason: reason.clone(),
        },
    }
}

fn identity(
    semantic: &SemanticKeyEnvelope,
    solver: CcmEigenstateSolver,
    xi: &[Float],
    l: &Float,
    seeds: &[Float],
) -> serde_json::Value {
    serde_json::json!({
        "semantics": "ccm-private-root-completion-v1",
        "request": semantic,
        "eigenstate_solver": format!("{solver:?}"),
        "coefficients": hp_values_memo_identity(xi),
        "length": hp_values_memo_identity(std::slice::from_ref(l)),
        "seeds": hp_values_memo_identity(seeds),
        "mpfr_range": [rug::float::exp_min(), rug::float::exp_max()],
    })
}

pub(super) fn checkpoints(
    semantic: &SemanticKeyEnvelope,
    solver: CcmEigenstateSolver,
    xi: &[Float],
    l: &Float,
    seeds: &[Float],
    mode: xc_cache::ArtifactExecutionCacheMode,
) -> Option<crate::ccm::capture_runtime::Checkpoints> {
    if mode != xc_cache::ArtifactExecutionCacheMode::PreferReuse {
        // Refresh and verification must perform fresh numerical work.
        return None;
    }
    match crate::ccm::capture_runtime::Checkpoints::new(&identity(semantic, solver, xi, l, seeds)) {
        Ok(store) => Some(store),
        Err(error) => {
            xc_core::progress_message!("root completion storage unavailable: {error}");
            None
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema_version: u32,
    index: usize,
    // Preserve each field's actual precision, including signed zero, instead
    // of round-tripping through the aggregate's fixed-precision decoder.
    runtime: PortableRootRefinementResult,
    adaptive_precision: Option<PortableAdaptiveRootPrecision>,
}

impl Completion {
    #[allow(clippy::too_many_arguments)]
    fn decode(
        self,
        index: usize,
        params: &CcmParams,
        cfg: &HighPrecConfig,
        xi: &[Float],
        poles: &[Float],
        solver: CcmEigenstateSolver,
        domain: IndependentRootDomain,
    ) -> Result<ComputedRootOutcome> {
        let ceiling = cfg
            .precision_bits
            .saturating_add(cfg.root_maximum_extra_precision_bits)
            .saturating_add(GUARD_BITS);
        if self.schema_version != 1
            || self.index != index
            || [
                &self.runtime.value,
                &self.runtime.final_correction,
                &self.runtime.residual,
                &self.runtime.achieved_decimal_digits,
            ]
            .iter()
            .any(|value| value.precision_bits > ceiling)
        {
            bail!("root completion metadata mismatch");
        }
        let computed = ComputedRootOutcome {
            outcome: EigenvalueResult::Converged(self.runtime.to_runtime()?),
            adaptive_precision: self.adaptive_precision,
        };
        let encoded = portable(&computed, solver);
        let decoded = decode_root_outcome(&encoded, cfg.precision_bits)?;
        validate_root_outcome(&encoded, &decoded, params, cfg, xi, poles, true)?;
        let value = decoded.value().expect("validated completion has a value");
        if value.is_zero() || (domain == IndependentRootDomain::Positive && value <= &0) {
            bail!("root completion is outside its requested domain");
        }
        Ok(computed)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute(
    xi: &[Float],
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    seeds: &[Float],
    solver: CcmEigenstateSolver,
    domain: IndependentRootDomain,
    store: Option<&crate::ccm::capture_runtime::Checkpoints>,
) -> std::result::Result<Vec<ComputedRootOutcome>, CacheError> {
    compute_with(
        xi,
        params,
        l,
        cfg,
        seeds,
        solver,
        domain,
        store,
        |index, poles| match cfg.root_precision_policy {
            RootPrecisionPolicy::FixedGuard => ComputedRootOutcome {
                outcome: solve_r_zero(
                    xi,
                    poles,
                    &seeds[index],
                    cfg.precision_bits,
                    cfg.solver_steps,
                    cfg.root_solver,
                ),
                adaptive_precision: None,
            },
            RootPrecisionPolicy::Adaptive => adaptive_root_outcome(xi, poles, &seeds[index], cfg),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_with(
    xi: &[Float],
    params: &CcmParams,
    l: &Float,
    cfg: &HighPrecConfig,
    seeds: &[Float],
    solver: CcmEigenstateSolver,
    domain: IndependentRootDomain,
    store: Option<&crate::ccm::capture_runtime::Checkpoints>,
    solve: impl Fn(usize, &[Float]) -> ComputedRootOutcome + Sync,
) -> std::result::Result<Vec<ComputedRootOutcome>, CacheError> {
    let spacing = match secular_spacing(l, cfg.precision_bits) {
        Ok(spacing) => spacing,
        Err(error) => {
            // Preserve the baseline failed-window outcome and its later
            // ensure_root_window_usable diagnostic, including an empty window.
            return Ok(seeds
                .iter()
                .take(1)
                .map(|_| ComputedRootOutcome {
                    outcome: EigenvalueResult::Failed {
                        iterations: 0,
                        reason: error.to_string(),
                    },
                    adaptive_precision: None,
                })
                .collect());
        }
    };
    let poles = secular_poles(&spacing, params.n_modes, cfg.precision_bits);
    let mut outcomes = ordered_map(seeds.len(), xi.len(), cfg, |index| {
        let label = format!("root-{index}");
        if let Some(store) = store {
            match store.load::<Completion>(&label) {
                Ok(Some(saved)) => {
                    match saved.decode(index, params, cfg, xi, &poles, solver, domain) {
                        Ok(computed) => return computed,
                        Err(error) => xc_core::progress_message!(
                            "root completion {index}: rejected ({error}); recomputing"
                        ),
                    }
                }
                Ok(None) => {}
                Err(error) => xc_core::progress_message!(
                    "root completion {index}: unavailable ({error}); recomputing"
                ),
            }
        }
        let computed = solve(index, &poles);
        // Only the solver's completed converged outcomes are progress. Failed,
        // resource-limited, stagnated and unfinished roots are always retried.
        if let (Some(store), EigenvalueResult::Converged(result)) = (store, &computed.outcome) {
            if let Ok(runtime) = PortableRootRefinementResult::from_runtime(result) {
                let saved = Completion {
                    schema_version: 1,
                    index,
                    runtime,
                    adaptive_precision: computed.adaptive_precision.clone(),
                };
                if let Err(error) = store.save(&label, &saved) {
                    xc_core::progress_message!("root completion {index}: not saved ({error})");
                }
            }
        }
        computed
    })?;
    if let Some(first_failure) = outcomes
        .iter()
        .position(|outcome| !outcome.outcome.has_value())
    {
        outcomes.truncate(first_failure + 1);
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // These synthetic secular-source fixtures have no automatic eigenstate
    // selection step. Bind their established legacy encoding explicitly.
    const FIXTURE_SOLVER: CcmEigenstateSolver = CcmEigenstateSolver::LegacyInverseIteration;

    #[test]
    fn root_spacing_failures_preserve_baseline_window_outcomes_and_errors() {
        let (params, cfg, _, xi, seeds) = fixture(false);
        let p = cfg.precision_bits;
        for length in [
            Float::with_val(p, 0),
            Float::with_val(p, -1),
            Float::with_val(p, rug::float::Special::Infinity),
            Float::with_val(p, 1) << (rug::float::exp_min() - 1),
        ] {
            assert!(secular_spacing(&length, p).is_err());
            for count in [0, 1, seeds.len()] {
                let seeds = &seeds[..count];
                let expected = compute_root_range_detailed(&xi, &params, &length, &cfg, seeds);
                let actual = compute_with(
                    &xi,
                    &params,
                    &length,
                    &cfg,
                    seeds,
                    FIXTURE_SOLVER,
                    IndependentRootDomain::Positive,
                    None,
                    |_, _| panic!("failed spacing must not invoke a root solver"),
                )
                .unwrap();
                let encoded = |outcomes: &[ComputedRootOutcome]| {
                    serde_json::to_vec(
                        &outcomes
                            .iter()
                            .map(|v| portable(v, FIXTURE_SOLVER))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                };
                assert_eq!(encoded(&actual), encoded(&expected));
                let checked = |outcomes: &[ComputedRootOutcome]| {
                    let roots = outcomes
                        .iter()
                        .map(|v| v.outcome.clone())
                        .collect::<Vec<_>>();
                    ensure_root_window_usable(&roots, count, true, IndependentRootDomain::Positive)
                        .map_err(|e| e.to_string())
                };
                assert_eq!(checked(&actual), checked(&expected));
                assert_eq!(actual.len(), usize::from(count != 0));
            }
        }
    }

    #[test]
    fn root_scheduling_evicts_idle_retention_but_keeps_active_handles_charged() {
        let (_, cfg, _, xi, _) = fixture(true);
        let bytes = task_bytes(xi.len(), &cfg);
        let resources = xc_core::RunResources::with_worker_limit(bytes, 1).unwrap();
        let _resources = resources.enter();
        let memo = xc_cache::ValidatedValueMemo::new(resources.clone(), bytes);
        let _memo = memo.enter();
        let retained = std::sync::Arc::new(Mutex::new(Some(std::sync::Arc::new(
            resources.try_reserve_retained(bytes).unwrap(),
        ))));
        let weak = std::sync::Arc::downgrade(&retained);
        let evictions = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = evictions.clone();
        let callback: std::sync::Arc<dyn Fn() + Send + Sync> = std::sync::Arc::new(move || {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Some(retained) = weak.upgrade() {
                retained.lock().unwrap().take();
            }
        });
        memo.register_eviction(&callback);
        assert_eq!(ordered_map(2, xi.len(), &cfg, |i| i).unwrap(), vec![0, 1]);
        assert_eq!(evictions.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(retained.lock().unwrap().is_none());
        assert_eq!(resources.usage().retained_bytes, 0);
        assert_eq!(resources.usage().working_bytes, 0);

        let active = std::sync::Arc::new(resources.try_reserve_retained(bytes).unwrap());
        *retained.lock().unwrap() = Some(active.clone());
        let computations = std::sync::atomic::AtomicUsize::new(0);
        let error = ordered_map(2, xi.len(), &cfg, |_| {
            computations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
        .unwrap_err();
        assert!(error.to_string().contains("run memory budget exceeded"));
        assert_eq!(computations.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(retained.lock().unwrap().is_none());
        assert_eq!(resources.usage().retained_bytes, bytes);
        assert_eq!(resources.usage().working_bytes, 0);
        drop(active);
        assert_eq!(resources.usage().retained_bytes, 0);
        assert_eq!(ordered_map(2, xi.len(), &cfg, |i| i).unwrap(), vec![0, 1]);
    }

    fn fixture(adaptive: bool) -> (CcmParams, HighPrecConfig, Float, Vec<Float>, Vec<Float>) {
        let params = CcmParams::from_lambda_sq_integer(13, 3);
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = 192;
        cfg.solver_steps = 64;
        cfg.eigenstate_solver = FIXTURE_SOLVER;
        if adaptive {
            cfg = cfg.with_adaptive_root_precision();
        }
        let l = log_lambda_sq_hp(&params, cfg.precision_bits).unwrap();
        let spacing = secular_spacing(&l, cfg.precision_bits).unwrap();
        let xi = vec![Float::with_val(cfg.precision_bits, 1); params.matrix_size()];
        let seeds = (0..3)
            .map(|i| Float::with_val(cfg.precision_bits, &spacing * (2 * i + 1)) / 2)
            .collect();
        (params, cfg, l, xi, seeds)
    }

    fn aggregate(
        params: &CcmParams,
        cfg: &HighPrecConfig,
        seeds: &[Float],
        outcomes: &[ComputedRootOutcome],
        source: &ContentDigest,
    ) -> PortableRootRange {
        let adaptive = cfg.root_precision_policy == RootPrecisionPolicy::Adaptive;
        PortableRootRange {
            schema_version: if adaptive { 5 } else { 6 },
            lambda_squared: lambda_squared_cache_identity(params),
            n_modes: params.n_modes,
            precision_bits: cfg.precision_bits,
            force_even: cfg.effective_parity_policy().legacy_force_even(),
            parity_policy: cfg.effective_parity_policy().portable_marker(),
            first_root_index: 1,
            root_domain: IndependentRootDomain::Positive,
            discovery_mode: RootArtifactMode::Independent.as_str().into(),
            reference_seeds_used: false,
            reference_dataset: None,
            completeness: RootWindowSemantics::strict_positive(seeds.len())
                .completeness(RootArtifactMode::Independent)
                .into(),
            starting_points: seeds.iter().map(Float::to_string).collect(),
            outcomes: outcomes
                .iter()
                .map(|outcome| portable(outcome, FIXTURE_SOLVER))
                .collect(),
            solver: cfg.root_solver.display_name().to_ascii_lowercase(),
            solver_steps: cfg.solver_steps,
            accuracy_guard_bits: GUARD_BITS,
            root_precision_policy: adaptive.then_some(RootPrecisionPolicy::Adaptive),
            target_precision_bits: adaptive.then_some(cfg.precision_bits - GUARD_BITS),
            maximum_extra_precision_bits: adaptive.then_some(cfg.root_maximum_extra_precision_bits),
            verification_precision_bits: adaptive.then_some(cfg.root_verification_precision_bits),
            secular_source_content_digest: Some(source.clone()),
        }
    }

    #[test]
    fn full_parallel_validation_preserves_values_and_first_error() {
        for adaptive in [false, true] {
            let (params, cfg, l, xi, seeds) = fixture(adaptive);
            let outcomes = compute_root_range_detailed(&xi, &params, &l, &cfg, &seeds);
            assert!(
                outcomes.iter().all(|r| r.outcome.is_converged()),
                "{outcomes:?}"
            );
            let source = ContentDigest::sha256(b"root-validation-fixture");
            let original = aggregate(&params, &cfg, &seeds, &outcomes, &source);
            let serial = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let parallel = rayon::ThreadPoolBuilder::new()
                .num_threads(3)
                .build()
                .unwrap();
            let decode = |artifact: &PortableRootRange| {
                decode_root_range(
                    artifact,
                    &params,
                    &cfg,
                    1,
                    &seeds,
                    RootArtifactMode::Independent,
                    None,
                    &xi,
                    &l,
                    RootWindowSemantics::strict_positive(seeds.len()),
                    Some(&source),
                    true,
                )
            };
            let a = serial.install(|| decode(&original)).unwrap();
            let b = parallel.install(|| decode(&original)).unwrap();
            let exact = |roots: &[EigenvalueResult]| {
                roots
                    .iter()
                    .map(|root| {
                        PortableRootRefinementResult::from_runtime(
                            outcome_refinement(root).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(exact(&a), exact(&b));
            let mut unordered = original.clone();
            unordered.outcomes.swap(0, 1);
            unordered.outcomes[2] = PortableRootOutcome::Failed {
                iterations: 1,
                reason: "later failure".into(),
            };
            let expected = serial
                .install(|| decode(&unordered))
                .unwrap_err()
                .to_string();
            assert!(expected.contains("strictly increasing"), "{expected}");
            assert_eq!(
                parallel
                    .install(|| decode(&unordered))
                    .unwrap_err()
                    .to_string(),
                expected
            );
            let PortableRootOutcome::Converged(first) = &mut unordered.outcomes[0] else {
                panic!()
            };
            first.iterations = cfg.solver_steps + 1;
            let expected = serial
                .install(|| decode(&unordered))
                .unwrap_err()
                .to_string();
            assert!(
                expected.contains("inconsistent convergence diagnostics"),
                "{expected}"
            );
            assert_eq!(
                parallel
                    .install(|| decode(&unordered))
                    .unwrap_err()
                    .to_string(),
                expected
            );
        }
    }

    #[test]
    fn durable_completion_resumes_interrupted_roots_and_rejects_changed_or_corrupt_data() {
        for adaptive in [false, true] {
            let (params, cfg, l, xi, seeds) = fixture(adaptive);
            let reference = compute_root_range_detailed(&xi, &params, &l, &cfg, &seeds);
            assert!(
                reference.iter().all(|r| r.outcome.is_converged()),
                "{reference:?}"
            );
            let source = ContentDigest::sha256(b"resume-fixture");
            let semantic = root_range_semantic_key(
                &params,
                &cfg,
                1,
                &seeds,
                RootArtifactMode::Independent,
                None,
                RootWindowSemantics::strict_positive(3),
                Some(&source),
            )
            .unwrap();
            let directory = xc_core::test_support::TestDir::new("root-progress");
            let binding = identity(&semantic, FIXTURE_SOLVER, &xi, &l, &seeds);
            let store =
                crate::ccm::capture_runtime::Checkpoints::local(&binding, directory.to_path_buf())
                    .unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.install(|| {
                    compute_with(
                        &xi,
                        &params,
                        &l,
                        &cfg,
                        &seeds,
                        FIXTURE_SOLVER,
                        IndependentRootDomain::Positive,
                        Some(&store),
                        |index, _| {
                            assert_ne!(index, 1, "simulated interruption after durable first root");
                            reference[index].clone()
                        },
                    )
                    .unwrap()
                })
            }));
            assert!(interrupted.is_err());
            assert!(store.load::<Completion>("root-0").unwrap().is_some());
            assert!(store.load::<Completion>("root-1").unwrap().is_none());
            let resumed_indices = Mutex::new(Vec::new());
            let resumed = pool.install(|| {
                compute_with(
                    &xi,
                    &params,
                    &l,
                    &cfg,
                    &seeds,
                    FIXTURE_SOLVER,
                    IndependentRootDomain::Positive,
                    Some(&store),
                    |index, _| {
                        resumed_indices.lock().unwrap().push(index);
                        reference[index].clone()
                    },
                )
                .unwrap()
            });
            assert_eq!(*resumed_indices.lock().unwrap(), vec![1, 2]);
            assert_eq!(
                serde_json::to_vec(&aggregate(&params, &cfg, &seeds, &reference, &source)).unwrap(),
                serde_json::to_vec(&aggregate(&params, &cfg, &seeds, &resumed, &source)).unwrap()
            );
            // A valid seal does not replace the root's numerical evidence.
            let mut bad = store.load::<Completion>("root-1").unwrap().unwrap();
            bad.runtime.residual = xc_numerics::fmt::PortableHpFloat::from_float(&Float::with_val(
                cfg.precision_bits,
                1,
            ))
            .unwrap();
            store.save("root-1", &bad).unwrap();
            let replayed = Mutex::new(Vec::new());
            pool.install(|| {
                compute_with(
                    &xi,
                    &params,
                    &l,
                    &cfg,
                    &seeds,
                    FIXTURE_SOLVER,
                    IndependentRootDomain::Positive,
                    Some(&store),
                    |index, _| {
                        replayed.lock().unwrap().push(index);
                        reference[index].clone()
                    },
                )
                .unwrap()
            });
            assert_eq!(*replayed.lock().unwrap(), vec![1]);
            // Corrupt payload bytes also fail the checkpoint seal before decode.
            let data_dir = std::fs::read_dir(&directory)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            std::fs::write(data_dir.join(ContentDigest::sha256(b"root-2").0), b"{}").unwrap();
            assert!(store.load::<Completion>("root-2").unwrap().is_none());
            let mut changed_seeds = seeds.clone();
            changed_seeds[0].next_up();
            let mut changed_xi = xi.clone();
            changed_xi[0].next_up();
            let mut changed_semantic = semantic.clone();
            changed_semantic.resolved_mathematical_parameters["solver_steps"] =
                serde_json::json!(cfg.solver_steps + 1);
            for changed in [
                identity(&semantic, FIXTURE_SOLVER, &xi, &l, &changed_seeds),
                identity(&semantic, FIXTURE_SOLVER, &changed_xi, &l, &seeds),
                identity(&changed_semantic, FIXTURE_SOLVER, &xi, &l, &seeds),
            ] {
                let other = crate::ccm::capture_runtime::Checkpoints::local(
                    &changed,
                    directory.to_path_buf(),
                )
                .unwrap();
                assert!(other.load::<Completion>("root-0").unwrap().is_none());
            }
            for mode in [
                xc_cache::ArtifactExecutionCacheMode::Refresh,
                xc_cache::ArtifactExecutionCacheMode::VerifyAgainstReference,
                xc_cache::ArtifactExecutionCacheMode::Disabled,
            ] {
                assert!(checkpoints(&semantic, FIXTURE_SOLVER, &xi, &l, &seeds, mode).is_none());
            }
        }
    }

    #[test]
    fn root_scheduling_admits_bounded_work_and_does_not_save_unfinished_outcomes() {
        let (params, cfg, l, xi, seeds) = fixture(true);
        let bytes = task_bytes(xi.len(), &cfg);
        let resources = xc_core::RunResources::with_worker_limit(bytes, 1).unwrap();
        let _scope = resources.enter();
        assert_eq!(
            ordered_map(5, xi.len(), &cfg, |index| index).unwrap(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(resources.usage().peak_bytes, bytes);
        assert_eq!(resources.usage().working_bytes, 0);
        let directory = xc_core::test_support::TestDir::new("root-no-failure-progress");
        let store = crate::ccm::capture_runtime::Checkpoints::local(
            &"failure-fixture",
            directory.to_path_buf(),
        )
        .unwrap();
        let outcomes = compute_with(
            &xi,
            &params,
            &l,
            &cfg,
            &seeds,
            FIXTURE_SOLVER,
            IndependentRootDomain::Positive,
            Some(&store),
            |_, _| ComputedRootOutcome {
                outcome: EigenvalueResult::Failed {
                    iterations: 0,
                    reason: "transient resource refusal".into(),
                },
                adaptive_precision: None,
            },
        )
        .unwrap();
        assert_eq!(outcomes.len(), 1);
        assert!(store.load::<Completion>("root-0").unwrap().is_none());
        let held = resources.try_reserve_retained(bytes).unwrap();
        assert!(compute(
            &xi,
            &params,
            &l,
            &cfg,
            &seeds,
            FIXTURE_SOLVER,
            IndependentRootDomain::Positive,
            Some(&store)
        )
        .is_err());
        assert!(store.load::<Completion>("root-0").unwrap().is_none());
        drop(held);
    }
}
