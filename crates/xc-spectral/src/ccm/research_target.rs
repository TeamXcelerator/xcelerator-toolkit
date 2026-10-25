//! Numeric research inputs from the explicitly configured runtime target.
//! Only samples and their finite Fourier projection are prepared here. No
//! infinite transform jets, tail model, atoms, or theorem bounds are inferred.
use super::{extended_research::*, retained_evidence::*, state_geometry::RetainedState};
use anyhow::{ensure, Result};
use rug::{float::Constant, Float};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use xc_cache::ContentDigest;
use xc_numerics::prefix::lossless_decimal as dec;

fn sample_native_versions() -> Result<Vec<u8>> {
    // These immutable NUL-terminated constants come from the same linked
    // MPFR/GMP libraries as Rug. No caller-owned allocation crosses the FFI.
    unsafe extern "C" {
        fn mpfr_get_version() -> *const std::ffi::c_char;
        static __gmp_version: *const std::ffi::c_char;
    }
    unsafe {
        let mpfr = mpfr_get_version();
        let gmp = __gmp_version;
        ensure!(
            !mpfr.is_null() && !gmp.is_null(),
            "target checkpoint native version unavailable"
        );
        let mut versions = std::ffi::CStr::from_ptr(mpfr).to_bytes_with_nul().to_vec();
        versions.extend_from_slice(std::ffi::CStr::from_ptr(gmp).to_bytes_with_nul());
        Ok(versions)
    }
}

pub(super) fn prepare(
    state: &RetainedState,
    spec: &crate::target::TargetProfileSpec,
    intervals: usize,
) -> Result<(ResearchInputs, ExternalResearchInputs)> {
    ensure!(
        intervals >= 2 * state.modes + 2 && intervals <= 131072 && intervals.is_multiple_of(2),
        "target research sampling requires an even grid resolving the retained Fourier modes"
    );
    let _stage = super::capture_runtime::Stage::new("runtime target research samples");
    let p = state.precision.saturating_add(64);
    precision(p)?;
    let store = super::capture_runtime::Checkpoints::new(&(
        "runtime-target-samples-and-fourier-v1",
        &state.manifest.content_digest,
        &state.cutoff,
        state.modes,
        p,
        intervals,
        spec.digest()?,
    ))?;
    if let Some((reference, input)) =
        store.load::<(ResearchInputs, ExternalResearchInputs)>("reference")?
    {
        reference.validate()?;
        input.matches(state)?;
        return Ok((reference, input));
    }
    let target = {
        let _phase = super::capture_runtime::Stage::new("runtime target evaluator setup");
        crate::target::hp::TargetEvaluator::from_spec(spec, p)?
    };
    let cutoff = scalar(&state.cutoff, p)?;
    target.validate_lambda(&cutoff.clone().sqrt())?;
    let l = cutoff.clone().ln();
    let definition = ContentDigest(spec.digest()?);
    let workers = sample_workers(spec, intervals + 1)?;
    let values = sample_values(&target, spec, p, &cutoff, &l, intervals, workers)?;
    let projection_phase = super::capture_runtime::Stage::new("runtime target Fourier projection");
    let mut samples = Vec::with_capacity(intervals + 1);
    let mut fine = vec![Float::with_val(p, 0); state.modes + 1];
    let mut coarse = fine.clone();
    for (j, value) in values.into_iter().enumerate() {
        let angle = Float::with_val(p, Constant::Pi) * (Float::with_val(p, j) / intervals + 1u32);
        let cosine = angle.cos();
        let mut previous = Float::with_val(p, 1);
        let mut current = cosine.clone();
        let weight = if j == 0 || j == intervals { 2u32 } else { 1u32 };
        for k in 0..=state.modes {
            let basis = if k == 0 {
                Float::with_val(p, 1)
            } else {
                current.clone()
            };
            let contribution = Float::with_val(p, &value) * basis / weight;
            fine[k] += &contribution;
            if j.is_multiple_of(2) {
                coarse[k] += &contribution;
            }
            if k > 0 {
                let next = Float::with_val(p, &cosine) * &current * 2u32 - &previous;
                previous = current;
                current = next;
            }
        }
        samples.push(dec(&value));
    }
    let mut refinement = Float::with_val(p, 0);
    for (f, c) in fine.iter_mut().zip(&mut coarse) {
        *f /= intervals;
        *c /= intervals / 2;
        refinement = refinement.max(&(Float::with_val(p, &*f) - &*c).abs());
    }
    let coefficients = (0..=2 * state.modes)
        .map(|j| dec(&fine[j.abs_diff(state.modes)]))
        .collect();
    drop(projection_phase);
    let _phase =
        super::capture_runtime::Stage::new("runtime target output validation and checkpoint");
    let scope = format!("supplied runtime target sampled on {intervals} uniform log-coordinate intervals; finite even Fourier projection through mode {}; composite trapezoid; maximum coefficient change on half grid={}; this measured refinement is not a quadrature, source-error, or infinite-tail certificate", state.modes, dec(&refinement));
    let reference = ResearchInputs {
        schema_version: 1,
        reference: ReferenceSpec {
            schema_version: 1,
            definition: format!(
                "finite Fourier projection of runtime target {}",
                definition.0
            ),
            lambda_squared: state.cutoff.clone(),
            precision_bits: p,
            coefficients,
            approximation_scope: scope.clone(),
        },
        basis: vec![],
        projection: ProjectionOptions {
            working_precision_bits: p + 64,
            normalization: "center_one".into(),
            fixed_second_component: None,
        },
    };
    reference.validate()?;
    let mut input: ExternalResearchInputs = serde_json::from_value(serde_json::json!({
        "schema_version":1,"source_eigenpair":state.manifest.content_digest,
        "lambda_squared":state.cutoff,"n_modes":state.modes,"precision_bits":p,
        "convention_id":"runtime-target-samples-and-fourier-v1",
        "definition_digest":definition,"approximation_scope":scope
    }))?;
    input.target = Some(SampledReference {
        definition_digest: definition,
        evaluation_policy: "configured target evaluator; uniform x=log(u) nodes; normalized at u=1"
            .into(),
        approximation_scope: scope,
        intervals,
        values: samples,
        basis_values: vec![],
        fixed_second_component: None,
        raw_normalizer: "1".into(),
        trial_coefficients: Some(reference.reference.coefficients.clone()),
    });
    input.validate()?;
    if let Err(e) = store.save("reference", &(&reference, &input)) {
        xc_core::progress_message!("target research checkpoint unavailable: {e}");
    }
    Ok((reference, input))
}

/// Largest default number of concurrent provider processes.
const MAXIMUM_DEFAULT_SAMPLE_WORKERS: usize = 64;

/// Provider processes used for sampling. A provider is evaluated concurrently
/// only when declared pure (`XC_TARGET_PROVIDER_PURE=1`): each value is then
/// the same function of its point in every process. The default is the Rayon
/// worker count, capped at 64; `XC_TARGET_PROVIDER_WORKERS` sets it
/// explicitly (1..=256).
pub(crate) fn sample_workers(
    spec: &crate::target::TargetProfileSpec,
    points: usize,
) -> Result<usize> {
    if spec.external_profile.is_none() || !crate::target::provider_declared_pure()? {
        return Ok(1);
    }
    let workers = match std::env::var("XC_TARGET_PROVIDER_WORKERS") {
        Ok(value) => {
            let workers = value
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("XC_TARGET_PROVIDER_WORKERS must be an integer"))?;
            ensure!(
                (1..=256).contains(&workers),
                "XC_TARGET_PROVIDER_WORKERS must be between 1 and 256"
            );
            workers
        }
        Err(std::env::VarError::NotPresent) => {
            rayon::current_num_threads().clamp(1, MAXIMUM_DEFAULT_SAMPLE_WORKERS)
        }
        Err(error) => return Err(error.into()),
    };
    Ok(workers.min(points))
}

/// Target values at `x_j = j*log(cutoff)/(2*intervals)`, `j=0..=intervals`,
/// in index order (see [`target_values_at`]).
fn sample_values(
    target: &crate::target::hp::TargetEvaluator,
    spec: &crate::target::TargetProfileSpec,
    p: u32,
    cutoff: &Float,
    l: &Float,
    intervals: usize,
    workers: usize,
) -> Result<Vec<Float>> {
    let points = {
        let _phase = super::capture_runtime::Stage::new("runtime target sample nodes");
        (0..=intervals)
            .map(|j| (Float::with_val(p, l) * j / (2 * intervals)).exp())
            .collect::<Vec<_>>()
    };
    target_values_at_impl(
        target,
        spec,
        p,
        &cutoff.clone().sqrt(),
        &points,
        workers,
        true,
    )
}

/// Target values at `points`, in order. With several workers, each evaluates
/// interleaved points through its own persistent provider process. The
/// values, and the sums a caller forms from them in point order, are identical
/// to sequential evaluation; an error is reported for the lowest failing point,
/// as sequentially.
pub(crate) fn target_values_at(
    target: &crate::target::hp::TargetEvaluator,
    spec: &crate::target::TargetProfileSpec,
    p: u32,
    lambda: &Float,
    points: &[Float],
    workers: usize,
) -> Result<Vec<Float>> {
    target_values_at_impl(target, spec, p, lambda, points, workers, false)
}

// Only the research preparation route (which already owns a complete-reference
// checkpoint) retains partial blocks. Ordinary measurement/verification callers
// of target_values_at never acquire a new persistent reuse path.
fn target_values_at_impl(
    target: &crate::target::hp::TargetEvaluator,
    spec: &crate::target::TargetProfileSpec,
    p: u32,
    lambda: &Float,
    points: &[Float],
    workers: usize,
    checkpoint_points: bool,
) -> Result<Vec<Float>> {
    let _phase = super::capture_runtime::Stage::new("runtime target point evaluation");
    let parallel = workers > 1
        && points.len() > 1
        && spec.external_profile.is_some()
        && crate::target::provider_declared_pure()?;
    let workers = if parallel {
        workers.min(points.len())
    } else {
        1
    };
    let workers =
        xc_core::RunResources::current().map_or(workers, |owner| workers.min(owner.worker_limit()));
    let progress = SampleProgress::new(points.len(), workers);
    // The authenticated evaluator already exists before a checkpoint is read.
    // Persist only declared-pure external evaluations. Checkpoints are local,
    // build-bound execution data and never replace a portable research artifact.
    let checkpoints = if checkpoint_points
        && spec.external_profile.is_some()
        && crate::target::provider_declared_pure()?
    {
        ensure!(
            target.definition_digest() == spec.digest()?,
            "runtime target checkpoint source mismatch"
        );
        // Admit the coordinate key before formatting it, plus serialized and
        // decoded blocks and each checkpoint reader's fixed buffer.
        let block_bytes = 16u64 * (u64::from(p) + 256) * 4;
        let key_bytes = points
            .iter()
            .map(|u| u128::from(u.prec()) + 256)
            .sum::<u128>()
            + u128::from(lambda.prec())
            + 256;
        let bytes = key_bytes + (workers as u128) * u128::from(2 * block_bytes + (1 << 20));
        let reservation = xc_core::RunResources::current().and_then(|owner| {
            u64::try_from(bytes)
                .ok()
                .and_then(|bytes| owner.try_reserve_working(bytes).ok())
        });
        if let Some(reservation) = reservation {
            let store = super::capture_runtime::Checkpoints::new(&(
                "runtime-target-point-blocks-v1",
                spec.digest()?,
                p,
                sample_native_versions()?,
                (lambda.prec(), lambda.to_string()),
                points
                    .iter()
                    .map(|u| (u.prec(), u.to_string()))
                    .collect::<Vec<_>>(),
            ))?;
            store
                .enabled()
                .then(|| (store.limited_bytes(block_bytes), reservation))
        } else {
            None
        }
    } else {
        None
    };
    if let Some((store, _reservation)) = checkpoints {
        let result = checkpointed_samples(
            &store,
            points.len(),
            workers,
            p,
            |_| {
                let start = Instant::now();
                let worker = crate::target::hp::TargetEvaluator::from_spec(spec, p)?;
                worker.validate_lambda(lambda)?;
                progress.provider_ready(start.elapsed());
                Ok(worker)
            },
            |worker, index| worker.try_value(&points[index]),
            || progress.point_done(),
        );
        progress.finish();
        return result;
    }
    if !parallel {
        let result = points
            .iter()
            .map(|u| {
                let value = target.try_value(u)?;
                progress.point_done();
                Ok(value)
            })
            .collect();
        progress.finish();
        return result;
    }
    let result = balanced_samples(
        points.len(),
        workers,
        |_| {
            let start = Instant::now();
            let worker = crate::target::hp::TargetEvaluator::from_spec(spec, p)?;
            worker.validate_lambda(lambda)?;
            progress.provider_ready(start.elapsed());
            Ok(worker)
        },
        |worker, index| {
            let value = worker.try_value(&points[index])?;
            progress.point_done();
            Ok(value)
        },
    );
    progress.finish();
    result
}

/// Save only complete blocks, preserving increasing point order within each
/// block and selecting the lowest failed block after joining all workers.
/// A restart loses at most one block per worker, not the whole profile.
fn checkpointed_samples<W>(
    store: &super::capture_runtime::Checkpoints,
    count: usize,
    workers: usize,
    p: u32,
    initialize: impl Fn(usize) -> Result<W> + Sync,
    evaluate: impl Fn(&W, usize) -> Result<Float> + Sync,
    point_done: impl Fn() + Sync,
) -> Result<Vec<Float>> {
    const BLOCK: usize = 16;
    let reused = AtomicUsize::new(0);
    let result = balanced_samples(
        count.div_ceil(BLOCK),
        workers,
        initialize,
        |worker, block| {
            let start = block * BLOCK;
            let end = (start + BLOCK).min(count);
            let label = format!(
                "points-{start}-{end}-range-{}-{}",
                rug::float::exp_min(),
                rug::float::exp_max()
            );
            let restored = store.load::<Vec<String>>(&label)?.and_then(|strings| {
                if strings.len() != end - start {
                    return None;
                }
                strings
                    .into_iter()
                    .map(|text| {
                        let value = Float::with_val(p, Float::parse(&text).ok()?);
                        (value.is_finite() && value.to_string() == text).then_some(value)
                    })
                    .collect::<Option<Vec<_>>>()
            });
            let values = if let Some(values) = restored {
                reused.fetch_add(values.len(), Ordering::Relaxed);
                values
            } else {
                let values = (start..end)
                    .map(|index| {
                        let value = evaluate(worker, index)?;
                        ensure!(
                            value.is_finite() && value.prec() == p,
                            "invalid runtime target checkpoint value"
                        );
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let strings = values.iter().map(Float::to_string).collect::<Vec<_>>();
                if let Err(error) = store.save(&label, &strings) {
                    xc_core::progress_message!("target sample checkpoint unavailable: {error}");
                }
                values
            };
            for _ in &values {
                point_done();
            }
            Ok(values)
        },
    );
    xc_core::progress_message!(
        "target sampling checkpoints: {} point(s) reused",
        reused.load(Ordering::Relaxed)
    );
    result.map(|blocks| blocks.into_iter().flatten().collect())
}

/// Every worker visits its indices in increasing order. An initialization
/// failure is associated with its first index, and all results are joined
/// before selecting the lowest failed index, independently of completion order.
fn balanced_samples<T: Send, W, I, E>(
    points: usize,
    workers: usize,
    initialize: I,
    evaluate: E,
) -> Result<Vec<T>>
where
    I: Fn(usize) -> Result<W> + Sync,
    E: Fn(&W, usize) -> Result<T> + Sync,
{
    if points == 0 {
        return Ok(Vec::new());
    }
    let workers = workers.max(1).min(points);
    let resources = xc_core::RunResources::current();
    let capture = xc_core::current_message_capture();
    let parts = std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|first| {
                let initialize = &initialize;
                let evaluate = &evaluate;
                let resources = resources.clone();
                let capture = capture.clone();
                scope.spawn(move || {
                    xc_core::with_message_capture(capture, || {
                        let _scope = resources.as_ref().map(xc_core::RunResources::enter);
                        let worker = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            initialize(first)
                        }))
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("target sampling worker panicked")))
                        .map_err(|error| (first, error))?;
                        let mut values = Vec::with_capacity((points - first).div_ceil(workers));
                        for index in (first..points).step_by(workers) {
                            let value =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    evaluate(&worker, index)
                                }))
                                .unwrap_or_else(|_| {
                                    Err(anyhow::anyhow!("target sampling worker panicked"))
                                })
                                .map_err(|error| (index, error))?;
                            values.push((index, value));
                        }
                        Ok::<_, (usize, anyhow::Error)>(values)
                    })
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .enumerate()
            .map(|(first, handle)| {
                handle.join().unwrap_or_else(|_| {
                    Err((first, anyhow::anyhow!("target sampling worker panicked")))
                })
            })
            .collect::<Vec<_>>()
    });
    let mut values = (0..points).map(|_| None).collect::<Vec<_>>();
    let mut failure: Option<(usize, anyhow::Error)> = None;
    for part in parts {
        match part {
            Ok(part) => {
                for (index, value) in part {
                    values[index] = Some(value);
                }
            }
            Err((index, error)) => {
                if failure.as_ref().is_none_or(|(first, _)| index < *first) {
                    failure = Some((index, error));
                }
            }
        }
    }
    if let Some((_, error)) = failure {
        return Err(error);
    }
    Ok(values
        .into_iter()
        .map(|value| value.expect("every sampling index was joined"))
        .collect())
}

/// Operational counters only; no source values or timing enter artifact bytes.
struct SampleProgress {
    start: Instant,
    next_report: AtomicU64,
    completed: AtomicUsize,
    initialized: AtomicUsize,
    setup_max_ms: AtomicU64,
    points: usize,
    workers: usize,
}
impl SampleProgress {
    fn new(points: usize, workers: usize) -> Self {
        xc_core::progress_message!("target sampling: {points} points; {workers} worker(s)");
        Self {
            start: Instant::now(),
            next_report: AtomicU64::new(30_000_000_000),
            completed: AtomicUsize::new(0),
            initialized: AtomicUsize::new(0),
            setup_max_ms: AtomicU64::new(0),
            points,
            workers,
        }
    }
    fn provider_ready(&self, elapsed: std::time::Duration) {
        self.initialized.fetch_add(1, Ordering::Relaxed);
        self.setup_max_ms.fetch_max(
            elapsed.as_millis().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
    fn point_done(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let next = self.next_report.load(Ordering::Relaxed);
        if elapsed >= next
            && self
                .next_report
                .compare_exchange(
                    next,
                    elapsed.saturating_add(30_000_000_000),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
        {
            xc_core::progress_message!(
                "target sampling: {}/{} points completed; {:.1}s elapsed",
                self.completed.load(Ordering::Relaxed),
                self.points,
                self.start.elapsed().as_secs_f64()
            );
        }
    }
    fn finish(&self) {
        xc_core::progress_message!("target sampling: {}/{} points completed with {} worker(s); {} new providers initialized; maximum provider setup {:.3}s; {:.3}s elapsed",
            self.completed.load(Ordering::Relaxed), self.points, self.workers,
            self.initialized.load(Ordering::Relaxed), self.setup_max_ms.load(Ordering::Relaxed) as f64 / 1000., self.start.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_partial_checkpoints_resume_exact_values_and_reject_invalid_data() {
        let directory = xc_core::test_support::TestDir::new("target-point-checkpoints");
        let p = 128;
        let store = super::super::capture_runtime::Checkpoints::local(
            &("point-fixture", p, "source-A"),
            directory.to_path_buf(),
        )
        .unwrap()
        .limited_bytes(1 << 20);
        let value = |i: usize| {
            if i == 0 {
                Float::with_val(p, rug::float::Special::NegZero)
            } else if i == 1 {
                Float::with_val(p, 1) >> 10000
            } else {
                Float::with_val(p, i) / 3
            }
        };
        let expected = (0..81).map(|i| value(i).to_string()).collect::<Vec<_>>();
        let error = checkpointed_samples(
            &store,
            81,
            2,
            p,
            |_| Ok(()),
            |_, i| {
                if i == 21 || i == 33 {
                    anyhow::bail!("point {i}");
                }
                Ok(value(i))
            },
            || {},
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "point 21");
        let computed = AtomicUsize::new(0);
        let resumed = checkpointed_samples(
            &store,
            81,
            4,
            p,
            |_| Ok(()),
            |_, i| {
                computed.fetch_add(1, Ordering::Relaxed);
                Ok(value(i))
            },
            || {},
        )
        .unwrap();
        assert!(
            computed.load(Ordering::Relaxed) <= 65,
            "first completed block survives failure"
        );
        assert_eq!(
            resumed.iter().map(Float::to_string).collect::<Vec<_>>(),
            expected
        );
        assert!(resumed[0].is_sign_negative());
        for workers in [1, 3, 8] {
            let warm = checkpointed_samples(
                &store,
                81,
                workers,
                p,
                |_| Ok(()),
                |_, _| -> Result<Float> {
                    panic!("warm checkpoint must avoid evaluation");
                },
                || {},
            )
            .unwrap();
            assert_eq!(
                warm.iter().map(Float::to_string).collect::<Vec<_>>(),
                expected
            );
        }
        let changed = super::super::capture_runtime::Checkpoints::local(
            &("point-fixture", p, "source-B"),
            directory.to_path_buf(),
        )
        .unwrap();
        computed.store(0, Ordering::Relaxed);
        checkpointed_samples(
            &changed,
            81,
            2,
            p,
            |_| Ok(()),
            |_, i| {
                computed.fetch_add(1, Ordering::Relaxed);
                Ok(value(i))
            },
            || {},
        )
        .unwrap();
        assert_eq!(computed.load(Ordering::Relaxed), 81);

        let first = format!(
            "points-0-16-range-{}-{}",
            rug::float::exp_min(),
            rug::float::exp_max()
        );
        for invalid in [vec!["NaN".to_owned(); 16], vec!["0".to_owned(); 15]] {
            store.save(&first, &invalid).unwrap();
            computed.store(0, Ordering::Relaxed);
            let restored = checkpointed_samples(
                &store,
                81,
                2,
                p,
                |_| Ok(()),
                |_, i| {
                    computed.fetch_add(1, Ordering::Relaxed);
                    Ok(value(i))
                },
                || {},
            )
            .unwrap();
            assert_eq!(computed.load(Ordering::Relaxed), 16);
            assert_eq!(
                restored.iter().map(Float::to_string).collect::<Vec<_>>(),
                expected
            );
        }
        // Alter a payload without its seal: integrity validation must miss.
        for source in std::fs::read_dir(&directory).unwrap() {
            for entry in std::fs::read_dir(source.unwrap().path()).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().is_none() {
                    std::fs::write(path, b"[]").unwrap();
                }
            }
        }
        computed.store(0, Ordering::Relaxed);
        let restored = checkpointed_samples(
            &store,
            81,
            2,
            p,
            |_| Ok(()),
            |_, i| {
                computed.fetch_add(1, Ordering::Relaxed);
                Ok(value(i))
            },
            || {},
        )
        .unwrap();
        assert_eq!(computed.load(Ordering::Relaxed), 81);
        assert_eq!(
            restored.iter().map(Float::to_string).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn balanced_sampling_preserves_order_at_uneven_worker_counts() {
        for points in [0, 1, 2, 17, 65] {
            for workers in [1, 2, 4, 8, 16] {
                let actual = balanced_samples(points, workers, Ok, |first, index| {
                    assert_eq!(index % workers.min(points).max(1), *first);
                    Ok(index * index)
                })
                .unwrap();
                assert_eq!(actual, (0..points).map(|i| i * i).collect::<Vec<_>>());
            }
        }
        assert_eq!(
            balanced_samples(3, 80, Ok, |_, index| Ok(index)).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn balanced_sampling_selects_lowest_failed_point_not_first_worker() {
        let error = balanced_samples(
            12,
            2,
            |_| Ok(()),
            |_, index| {
                if index == 3 || index == 4 {
                    anyhow::bail!("failed point {index}");
                }
                Ok(index)
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "failed point 3");
        // Initialization belongs to its first point; lower point failures win.
        let error = balanced_samples(
            12,
            2,
            |first| {
                if first == 1 {
                    anyhow::bail!("initialization failed");
                }
                Ok(())
            },
            |_, index| -> Result<usize> { anyhow::bail!("failed point {index}") },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "failed point 0");
        let error = balanced_samples(
            12,
            2,
            |first| {
                if first == 0 {
                    anyhow::bail!("initialization failed");
                }
                Ok(())
            },
            |_, index| -> Result<usize> { anyhow::bail!("failed point {index}") },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "initialization failed");
    }

    #[test]
    fn balanced_sampling_associates_panics_with_the_actual_point() {
        let error = balanced_samples(
            12,
            2,
            |_| Ok(()),
            |_, index| {
                assert_ne!(index, 3, "point panic");
                if index == 4 {
                    anyhow::bail!("higher point failed");
                }
                Ok(index)
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "target sampling worker panicked");
        let error = balanced_samples(
            12,
            2,
            |_| Ok(()),
            |_, index| {
                assert_ne!(index, 4, "higher point panic");
                if index == 3 {
                    anyhow::bail!("lower point failed");
                }
                Ok(index)
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "lower point failed");
        let error = balanced_samples(
            12,
            2,
            |first| {
                assert_ne!(first, 0, "initialization panic");
                Ok(())
            },
            |_, index| Ok(index),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "target sampling worker panicked");
    }

    #[test]
    fn balanced_sampling_inherits_the_run_owner() {
        let owner = xc_core::RunResources::with_worker_limit(123456, 4).unwrap();
        let _scope = owner.enter();
        let values = balanced_samples(
            9,
            4,
            |_| {
                assert_eq!(
                    xc_core::RunResources::current().unwrap().maximum_bytes(),
                    123456
                );
                Ok(())
            },
            |_, index| {
                assert_eq!(xc_core::RunResources::current().unwrap().worker_limit(), 4);
                Ok(index)
            },
        )
        .unwrap();
        assert_eq!(values, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn concurrent_pure_provider_sampling_equals_sequential_sampling() {
        use std::{fs, process::Command};
        struct RestoreEnvironment(&'static str, Option<std::ffi::OsString>);
        impl Drop for RestoreEnvironment {
            fn drop(&mut self) {
                match &self.1 {
                    Some(value) => std::env::set_var(self.0, value),
                    None => std::env::remove_var(self.0),
                }
            }
        }
        let root_dir = xc_core::test_support::TestDir::new("parallel-target-samples");
        let root = root_dir.to_path_buf();
        let source = root.join("provider.rs");
        // Replies with the requested point itself: distinct, deterministic values.
        fs::write(&source, r#"
use std::io::{self,BufRead,Write};
fn field<'a>(line:&'a str,name:&str)->Option<&'a str>{
 let marker=format!("\"{}\":",name);
 Some(line.split_once(&marker)?.1.split([',','}']).next().unwrap())
}
fn main(){
 for line in io::stdin().lock().lines(){
  let line=line.unwrap();
  let id=field(&line,"request_id").unwrap();let nonce=field(&line,"request_nonce").unwrap();
  let operation=field(&line,"operation").unwrap();let bits=field(&line,"precision_bits").unwrap();
  let value=field(&line,"u").unwrap_or("\"1\"");
  println!("{{\"protocol_version\":2,\"request_id\":{id},\"request_nonce\":{nonce},\"operation\":{operation},\"precision_bits\":{bits},\"ready\":true,\"value\":{value}}}");
  io::stdout().flush().unwrap();
 }
}
"#).unwrap();
        let executable = root.join("provider");
        let output = Command::new("rustc")
            .arg("--edition=2021")
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let digest = ContentDigest::sha256(&fs::read(&executable).unwrap()).0;
        let _previous = RestoreEnvironment(
            "XC_TARGET_PROVIDER_EXECUTABLE",
            std::env::var_os("XC_TARGET_PROVIDER_EXECUTABLE"),
        );
        let _previous_pure = RestoreEnvironment(
            "XC_TARGET_PROVIDER_PURE",
            std::env::var_os("XC_TARGET_PROVIDER_PURE"),
        );
        std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", &executable);
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
        let spec = crate::target::TargetProfileSpec::from_json(
            &serde_json::to_vec(&serde_json::json!({
                "schema_version":3,"profile_id":"parallel-samples","external_profile":{
                    "lambda_squared":"13","evaluation_precision_bits":320,"provider_sha256":digest,"input":{}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let p = 256;
        let cutoff = Float::with_val(p, 13);
        let l = cutoff.clone().ln();
        let target = crate::target::hp::TargetEvaluator::from_spec(&spec, p).unwrap();
        let sequential = sample_values(&target, &spec, p, &cutoff, &l, 64, 1);
        let concurrent = sample_values(&target, &spec, p, &cutoff, &l, 64, 4);
        let _previous_checkpoint = RestoreEnvironment(
            "XC_RESEARCH_CHECKPOINT_DIR",
            std::env::var_os("XC_RESEARCH_CHECKPOINT_DIR"),
        );
        std::env::set_var("XC_RESEARCH_CHECKPOINT_DIR", root.join("checkpoints"));
        let owner = xc_core::RunResources::with_worker_limit(32 << 20, 2).unwrap();
        {
            let _scope = owner.enter();
            let cold = sample_values(&target, &spec, p, &cutoff, &l, 64, 4).unwrap();
            let warm = sample_values(&target, &spec, p, &cutoff, &l, 64, 1).unwrap();
            assert_eq!(&cold, sequential.as_ref().unwrap());
            assert_eq!(cold, warm);
            assert_eq!(owner.usage().working_bytes, 0);
            let held = owner.try_reserve_working(owner.maximum_bytes()).unwrap();
            assert_eq!(
                sample_values(&target, &spec, p, &cutoff, &l, 64, 4).unwrap(),
                cold
            );
            drop(held);
        }
        assert!(std::fs::read_dir(root.join("checkpoints"))
            .unwrap()
            .next()
            .is_some());
        // A direct caller asking for parallelism must still use the existing
        // evaluator if purity is undeclared. No new process can initialize
        // once executable authorization is removed.
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
        std::env::remove_var("XC_TARGET_PROVIDER_EXECUTABLE");
        let undeclared = sample_values(&target, &spec, p, &cutoff, &l, 64, 8);
        let (sequential, concurrent) = (sequential.unwrap(), concurrent.unwrap());
        assert_eq!(sequential.len(), 65);
        assert_eq!(sequential, concurrent);
        assert_eq!(sequential, undeclared.unwrap());
        assert!(sequential.windows(2).all(|pair| pair[0] < pair[1]));
    }
    #[test]
    fn supplied_target_preparation_preserves_identity_and_does_not_invent_tail_inputs() {
        use crate::target::{GaussianPolynomialSeriesSpec, ScalarScaleSpec, TargetProfileSpec};
        let spec = TargetProfileSpec {
            schema_version: 1,
            profile_id: "research-sample-fixture".into(),
            external_profile: None,
            auxiliary_series: None,
            base_series: Some(GaussianPolynomialSeriesSpec {
                term_input_power: 0,
                polynomial_coefficients: vec!["1".into()],
                polynomial_scale: ScalarScaleSpec::default(),
                parameter_polynomial_coefficients: vec![],
                parameter_polynomial_scale: ScalarScaleSpec::default(),
                minimum_terms: 2,
                maximum_terms: 1000,
            }),
        };
        let digest = ContentDigest::sha256(b"sample-state");
        let manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"key":xc_cache::ArtifactKey::new("ccm_weil_eigenpair","test",b"test").unwrap(),
            "content_digest":digest,"size_bytes":12,
            "objects":[{"content_digest":digest,"size_bytes":12}],"created_unix_seconds":1,
            "producer_toolkit_version":xc_cache::ToolkitVersion::parse("0.16.0").unwrap(),
            "minimum_reader_version":xc_cache::ToolkitVersion::parse("0.16.0").unwrap(),
            "maximum_reader_version":null,"quality":"validated","visibility":"private",
            "immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
        })).unwrap();
        let state = RetainedState {
            manifest,
            cutoff: "9".into(),
            modes: 2,
            precision: 128,
            coefficients: vec![Float::with_val(128, 1); 5],
            eigenvalue: "1".into(),
            selection_policy: None,
        };
        let (reference, input) = prepare(&state, &spec, 256).unwrap();
        eprintln!(
            "target preparation fixture digest: {}",
            ContentDigest::sha256_json(&(&reference, &input)).unwrap().0
        );
        assert_eq!(input.source_eigenpair, state.manifest.content_digest);
        assert_eq!(input.definition_digest.0, spec.digest().unwrap());
        assert_eq!(reference.reference.coefficients.len(), 2 * state.modes + 1);
        assert_eq!(input.target.as_ref().unwrap().values.len(), 257);
        assert_eq!(
            scalar(&input.target.as_ref().unwrap().values[0], 192).unwrap(),
            1
        );
        assert!(input.reference_jets.is_empty());
        assert!(input.atoms.is_empty());
        assert!(input.energy_allowance.is_none());
        assert!(reference
            .reference
            .approximation_scope
            .contains("not a quadrature"));
        assert!(prepare(&state, &spec, 255).is_err());
    }
}
