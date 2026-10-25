//! CCM-specific provider preparation. Generic callback APIs stay serial.
use super::*;
use std::cell::RefCell;

struct BoundTarget {
    target: crate::target::hp::TargetEvaluator,
    spec: crate::target::TargetProfileSpec,
    working: u32,
}

/// Speculative pure-provider replies for an existing profile. Errors remain
/// attached to their index until the original serial loop visits that point.
pub(super) struct ProfileTargets {
    outcomes: Vec<Option<Result<Float>>>,
    range: (i32, i32),
    _reservation: xc_core::MemoryReservation,
}

impl ProfileTargets {
    pub(super) fn try_new(
        target: &crate::target::hp::TargetEvaluator,
        points: &[Option<&Float>],
        working: u32,
    ) -> Result<Option<Self>> {
        let Some(owner) = xc_core::RunResources::current() else {
            return Ok(None);
        };
        if points.len() < 2
            || !target.uses_external_provider()
            || !crate::target::provider_declared_pure()?
            || !fresh_threads_match()
        {
            return Ok(None);
        }
        let spec = crate::target::TargetProfileSpec::from_environment()?;
        anyhow::ensure!(
            spec.digest()? == target.definition_digest(),
            "runtime target specification changed after its semantic identity was fixed"
        );
        let workers = crate::ccm::research_target::sample_workers(&spec, points.len())?
            .min(owner.worker_limit());
        if workers < 2 {
            return Ok(None);
        }
        let bits = points
            .iter()
            .flatten()
            .map(|u| u.prec())
            .max()
            .unwrap_or(working)
            .max(working.saturating_add(64));
        let bytes = (points.len() as u128) * (u128::from(bits).div_ceil(8) * 16 + 1024)
            + (workers as u128) * 8192;
        let Some(reservation) = u64::try_from(bytes)
            .ok()
            .and_then(|bytes| owner.try_reserve_working(bytes).ok())
        else {
            return Ok(None);
        };
        // Existing memo replies make process setup more expensive than simply
        // visiting the original loop. This is only a scheduling hint; eviction
        // between here and consumption uses the authenticated serial evaluator.
        if points
            .iter()
            .flatten()
            .all(|u| target.has_memoized_value(u))
        {
            return Ok(None);
        }
        let outcomes = indexed_outcomes(
            points.len(),
            workers,
            |_| crate::target::hp::TargetEvaluator::from_spec(&spec, working),
            |worker, index| match points[index] {
                Some(u) => worker.try_value(u).map(Some),
                None => Ok(None),
            },
        );
        let outcomes = outcomes
            .into_iter()
            .map(|outcome| match outcome {
                Some(Ok(Some(value))) => Some(Ok(value)),
                Some(Err(error)) => Some(Err(error)),
                _ => None,
            })
            .collect();
        Ok(Some(Self {
            outcomes,
            range: (rug::float::exp_min(), rug::float::exp_max()),
            _reservation: reservation,
        }))
    }

    pub(super) fn value(
        &mut self,
        index: usize,
        target: &crate::target::hp::TargetEvaluator,
        u: &Float,
    ) -> Result<Float> {
        match self.outcomes.get_mut(index).and_then(Option::take) {
            Some(Err(error)) => Err(error),
            Some(Ok(value)) if self.range == (rug::float::exp_min(), rug::float::exp_max()) => {
                Ok(value)
            }
            _ => target.try_value(u),
        }
    }
}

impl BoundTarget {
    fn new(lambda: &Float, prec: u32, expected: Option<&str>) -> Result<Self> {
        let working = prec.saturating_add(GUARD_BITS);
        let spec = crate::target::TargetProfileSpec::from_environment()?;
        let target = crate::target::hp::TargetEvaluator::from_spec(&spec, working)?;
        target.validate_lambda(lambda)?;
        if expected.is_some_and(|value| value != target.definition_digest()) {
            anyhow::bail!(
                "runtime target specification changed after its semantic identity was fixed"
            );
        }
        Ok(Self {
            target,
            spec,
            working,
        })
    }
}

// The reservation drops after all buffers. No borrowed callback owns this data.
struct Buffers {
    points: Vec<Float>,
    keys: Vec<String>,
    outcomes: Vec<Option<Result<Float>>>,
    next: usize,
    _reservation: xc_core::MemoryReservation,
}

struct Prepared<'a> {
    bound: &'a BoundTarget,
    range: (i32, i32),
    buffers: RefCell<Buffers>,
}

fn fresh_threads_match() -> bool {
    let range = (rug::float::exp_min(), rug::float::exp_max());
    std::thread::Builder::new()
        .spawn(|| (rug::float::exp_min(), rug::float::exp_max()))
        .is_ok_and(|thread| thread.join().is_ok_and(|actual| actual == range))
}

/// Every attempted failure remains attached to its point; no global error
/// collection can select an unvisited failure ahead of serial weighted work.
fn indexed_outcomes<T: Send, W, I, E>(
    points: usize,
    workers: usize,
    init: I,
    eval: E,
) -> Vec<Option<Result<T>>>
where
    I: Fn(usize) -> Result<W> + Sync,
    E: Fn(&W, usize) -> Result<T> + Sync,
{
    let workers = workers.max(1).min(points.max(1));
    let owner = xc_core::RunResources::current();
    let capture = xc_core::current_message_capture();
    let parts = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for first in 0..workers.min(points) {
            let init = &init;
            let eval = &eval;
            let owner = owner.clone();
            let capture = capture.clone();
            let handle = std::thread::Builder::new().spawn_scoped(scope, move || {
                xc_core::with_message_capture(capture, || {
                    let _scope = owner.as_ref().map(xc_core::RunResources::enter);
                    let worker =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| init(first)))
                            .unwrap_or_else(|_| {
                                Err(anyhow::anyhow!("target sampling worker panicked"))
                            });
                    let worker = match worker {
                        Ok(worker) => worker,
                        Err(error) => return vec![(first, Err(error))],
                    };
                    let mut values = Vec::with_capacity((points - first).div_ceil(workers));
                    for index in (first..points).step_by(workers) {
                        let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            eval(&worker, index)
                        }))
                        .unwrap_or_else(|_| {
                            Err(anyhow::anyhow!("target sampling worker panicked"))
                        });
                        let failed = value.is_err();
                        values.push((index, value));
                        if failed {
                            break;
                        }
                    }
                    values
                })
            });
            handles.push((first, handle));
        }
        handles
            .into_iter()
            .map(|(first, handle)| match handle {
                Ok(handle) => handle.join().unwrap_or_else(|_| {
                    vec![(
                        first,
                        Err(anyhow::anyhow!("target sampling worker panicked")),
                    )]
                }),
                Err(error) => vec![(first, Err(error.into()))],
            })
            .collect::<Vec<_>>()
    });
    let mut outcomes = (0..points).map(|_| None).collect::<Vec<_>>();
    for part in parts {
        for (index, value) in part {
            outcomes[index] = Some(value);
        }
    }
    outcomes
}

impl<'a> Prepared<'a> {
    fn try_new(
        bound: &'a BoundTarget,
        lambda: &Float,
        rule: WeightedIntegrationRule,
        prec: u32,
        tables: Option<&SharedGlTables>,
    ) -> Result<Option<Self>> {
        // No optional policy parsing or new allocation without a run owner.
        let Some(owner) = xc_core::RunResources::current() else {
            return Ok(None);
        };
        if rule.variable() != GridVariable::U || bound.spec.external_profile.is_none() {
            return Ok(None);
        }
        let count = match rule {
            WeightedIntegrationRule::UniformGrid {
                scheme: UniformGridScheme::Trapezoid,
                steps,
                ..
            } => steps.checked_add(1),
            _ => Some(rule.resolution()),
        };
        let Some(count) = count.filter(|count| *count > 0) else {
            return Ok(None);
        };
        if let WeightedIntegrationRule::GaussLegendre { points, .. } = rule {
            if tables
                .and_then(|tables| tables.get(points, bound.working))
                .is_none()
            {
                return Ok(None);
            }
        }
        let maximum_workers = count.min(owner.worker_limit()).min(256);
        if maximum_workers < 2 {
            return Ok(None);
        }
        let bytes = u64::try_from(count).ok().and_then(|count| {
            let effective_bits = bound.working.saturating_add(64).max(lambda.prec());
            let scalar = u64::from(effective_bits.saturating_add(63)).div_ceil(64) * 8;
            count
                .checked_mul(16 * (std::mem::size_of::<Float>() as u64 + scalar) + 512)
                .and_then(|bytes| bytes.checked_add((maximum_workers as u64).saturating_mul(8192)))
        });
        let Some(reservation) = bytes.and_then(|bytes| owner.try_reserve_working(bytes).ok())
        else {
            return Ok(None);
        };
        let workers = crate::ccm::research_target::sample_workers(&bound.spec, count)?
            .min(owner.worker_limit());
        if workers < 2 || !fresh_threads_match() {
            return Ok(None);
        }
        // Use only existing GL tables. All later results, including errors, are
        // deferred until the original target callback visits their coordinate.
        let points = weighted_rule_abscissae(rule, lambda, prec, tables);
        if points.len() != count || points.iter().any(|u| !u.is_finite() || u < &1) {
            return Ok(None);
        }
        let keys = points
            .iter()
            .map(|u| provider_point_key(bound, u))
            .collect();
        let outcomes = indexed_outcomes(
            count,
            workers,
            |_| {
                let target =
                    crate::target::hp::TargetEvaluator::from_spec(&bound.spec, bound.working)?;
                target.validate_lambda(lambda)?;
                Ok(target)
            },
            |target, index| target.try_value(&points[index]),
        );
        Ok(Some(Self {
            bound,
            range: (rug::float::exp_min(), rug::float::exp_max()),
            buffers: RefCell::new(Buffers {
                points,
                keys,
                outcomes,
                next: 0,
                _reservation: reservation,
            }),
        }))
    }

    fn value(&self, u: &Float) -> Result<Float> {
        let key = provider_point_key(self.bound, u);
        let mut buffers = self.buffers.borrow_mut();
        let next = buffers.next;
        let index = (next..buffers.points.len()).find(|&index| {
            buffers.outcomes[index].is_some()
                && buffers.points[index] == *u
                && buffers.points[index].is_sign_negative() == u.is_sign_negative()
                && buffers.keys[index] == key
        });
        if let Some(index) = index {
            let outcome = buffers.outcomes[index]
                .take()
                .expect("observed deferred outcome");
            while buffers.next < buffers.outcomes.len() && buffers.outcomes[buffers.next].is_none()
            {
                buffers.next += 1;
            }
            drop(buffers);
            // A known failure never authorizes a retry, including after a
            // coordinate mismatch. Successful values require the same context.
            return match outcome {
                Err(error) => Err(error),
                Ok(value) if self.range == (rug::float::exp_min(), rug::float::exp_max()) => {
                    Ok(value)
                }
                Ok(_) => self.bound.target.try_value(u),
            };
        }
        drop(buffers);
        // Genuinely unplanned points, including an unexecuted worker tail,
        // use the existing evaluator. Keep all other failures for later visits.
        self.bound.target.try_value(u)
    }
}

fn provider_point_key(bound: &BoundTarget, u: &Float) -> String {
    // TargetEvaluator adds 64 internal guard bits. The external provider
    // protocol serializes the padded coordinate with to_string_radix(10, None).
    let provider_u = Float::with_val(bound.working.saturating_add(64).max(u.prec()), u);
    provider_u.to_string_radix(10, None)
}

fn validate_weighted(
    lambda: &Float,
    alpha: &Float,
    rule: WeightedIntegrationRule,
    prec: u32,
) -> Result<()> {
    validate_reconstruction_precision(prec)?;
    validate_lambda_alpha(lambda, alpha)?;
    rule.validate()
}

#[allow(clippy::too_many_arguments)]
fn replay<F: Fn(&Float) -> Float>(
    f: F,
    bound: &BoundTarget,
    prepared: Option<&Prepared<'_>>,
    lambda: &Float,
    alpha: &Float,
    rule: WeightedIntegrationRule,
    prec: u32,
    tables: Option<&mut SharedGlTables>,
) -> Result<WeightedGridValueHp> {
    let failure = super::super::EvaluationFailure::default();
    let result = weighted_alpha_distance_with_tables(
        f,
        |u| {
            failure.capture(
                || match prepared {
                    Some(prepared) => prepared.value(u),
                    None => bound.target.try_value(u),
                },
                || Float::with_val(bound.working, f64::NAN),
            )
        },
        lambda,
        alpha,
        rule,
        prec,
        tables,
    );
    failure.finish(result)
}

/// Only known CCM eigenfunction sources reach preparation. In particular no
/// generic user Fn is evaluated early or admitted as an implicitly pure source.
#[allow(clippy::too_many_arguments)]
pub(super) fn ccm_distance(
    eigenfunction: &WeilEigenfunction,
    values: Option<&PrecomputedEigenfunctionValues<'_>>,
    recorded: Option<&RefCell<Vec<Float>>>,
    lambda: &Float,
    alpha: &Float,
    rule: WeightedIntegrationRule,
    prec: u32,
    tables: Option<&mut SharedGlTables>,
    expected: Option<&str>,
) -> Result<WeightedGridValueHp> {
    let bound = BoundTarget::new(lambda, prec, expected)?;
    validate_weighted(lambda, alpha, rule, prec)?;
    let prepared = Prepared::try_new(&bound, lambda, rule, prec, tables.as_deref())?;
    replay(
        |u| {
            let value = values.map_or_else(|| eigenfunction.eval(u), |values| values.eval(u));
            if let Some(recorded) = recorded {
                recorded.borrow_mut().push(value.clone());
            }
            value
        },
        &bound,
        prepared.as_ref(),
        lambda,
        alpha,
        rule,
        prec,
        tables,
    )
}

#[cfg(test)]
mod tests;
