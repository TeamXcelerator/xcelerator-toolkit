//! Correctly rounded finite prime-response action at the exact stored length
//! and vector. Transcendental constants are enclosed, not rounded inputs.
use super::*;
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use rug::{Integer, Rational};
use xc_numerics::mpfr_interval::MpfrInterval as I;
#[path = "prime_response_gpu.rs"]
mod gpu;

pub(super) fn validate_batch_configuration() -> Result<()> {
    gpu::validate_configuration()
}

fn rounded(value: &I, p: u32) -> Result<Option<Float>> {
    value.validate()?;
    let lo = Float::with_val(p, value.lower());
    let hi = Float::with_val(p, value.upper());
    Ok((lo == hi
        && lo.is_finite()
        && (!lo.is_zero() || (value.lower().is_zero() && value.upper().is_zero())))
    .then_some(lo))
}
fn coefficients(
    vector: &[Rational],
    n_modes: usize,
    row: usize,
) -> Result<(Vec<Rational>, Rational)> {
    let n = row as i64 - n_modes as i64;
    let k = n.unsigned_abs() as usize;
    let mut cosine = vec![Rational::from(0); n_modes + 1];
    cosine[k] += vector[row].clone() * 2;
    for (column, v) in vector.iter().enumerate() {
        let m = column as i64 - n_modes as i64;
        if m == n {
            continue;
        }
        let factor: Rational = v.clone() * 2i32 / (n - m);
        cosine[k] += factor.clone() * n;
        cosine[m.unsigned_abs() as usize] -= factor * m;
    }
    let sine = vector[row].clone() * (-4 * (k as i64));
    let bits = cosine
        .iter()
        .chain(std::iter::once(&sine))
        .map(|x| {
            u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits())
        })
        .sum::<u128>();
    if bits > 67_108_864 {
        bail!("prime-response row exceeds exact coefficient budget");
    }
    Ok((cosine, sine))
}

/// Compact exact preparation shared across rows, guard precisions, and events.
/// No floating-point operation or complete coefficient matrix is retained.
pub(super) struct CoefficientSource<'a> {
    exact: &'a [Rational],
    modes: usize,
    common: Option<CommonDenominator>,
    storage_bytes: u128,
}
struct CommonDenominator {
    values: Vec<Integer>,
    reciprocals: Vec<Integer>,
    denominator: Integer,
    lcm: Integer,
    _reservation: xc_core::MemoryReservation,
}
pub(super) fn reserve_optional(bytes: u128) -> Option<xc_core::MemoryReservation> {
    xc_core::RunResources::current()?
        .try_reserve_retained(bytes.try_into().ok()?)
        .ok()
}
impl<'a> CoefficientSource<'a> {
    pub(super) fn new(exact: &'a [Rational], modes: usize, available: u128) -> Self {
        let mut source = Self {
            exact,
            modes,
            common: None,
            storage_bytes: 0,
        };
        // An optional optimization, never an additional source-admission cap.
        // The full row matrix would be quadratic in high-precision rationals.
        // Store only a scaled source vector and LCM/d for 1 <= d <= 2N.
        let distance = modes * 2;
        let lcm_bits =
            (distance as u128) * u128::from(usize::BITS - distance.max(1).leading_zeros()) + 1;
        let denominator_bits = exact
            .iter()
            .map(|v| v.denom().significant_bits())
            .max()
            .unwrap_or(1);
        let numerator_bits = exact
            .iter()
            .map(|v| v.numer().significant_bits())
            .max()
            .unwrap_or(0);
        let value_bits = u128::from(denominator_bits) + u128::from(numerator_bits);
        let retained = exact.len() as u128 * (value_bits.div_ceil(8) + 64)
            + (distance as u128 + 1) * (lcm_bits.div_ceil(8) + 64)
            + 4 * ((value_bits + lcm_bits).div_ceil(8) + 64);
        // Integer accumulators coexist with the resulting rational row while
        // collection finishes. Charge this extra scratch for every worker.
        let row_scratch = rayon::current_num_threads().min(exact.len()) as u128
            * (modes as u128 + 1)
            * ((value_bits + lcm_bits + 32).div_ceil(8) + 64);
        let bytes = retained + row_scratch;
        if retained > (64u128 << 20) || bytes > available {
            return source;
        }
        // Optional reuse requires an active shared ledger. Lack of a scope or
        // room in that ledger selects the original arithmetic path.
        let Some(reservation) = reserve_optional(bytes) else {
            return source;
        };
        let dyadic_denominator = exact
            .iter()
            .map(Rational::denom)
            .max()
            .cloned()
            .unwrap_or_else(|| Integer::from(1));
        // All production inputs come from finite Floats. Retain the exact
        // rational fallback if a caller supplies other denominators.
        if exact
            .iter()
            .any(|v| Integer::from(&dyadic_denominator % v.denom()) != 0)
        {
            return source;
        }
        let mut lcm = Integer::from(1);
        for d in 2..=distance {
            lcm.lcm_mut(&Integer::from(d));
        }
        let values = exact
            .iter()
            .map(|v| v.numer() * Integer::from(&dyadic_denominator / v.denom()))
            .collect();
        let reciprocals = (0..=distance)
            .map(|d| {
                if d == 0 {
                    Integer::from(0)
                } else {
                    Integer::from(&lcm / d)
                }
            })
            .collect();
        source.common = Some(CommonDenominator {
            values,
            reciprocals,
            denominator: dyadic_denominator * &lcm,
            lcm,
            _reservation: reservation,
        });
        source.storage_bytes = bytes;
        source
    }

    pub(super) fn storage_bytes(&self) -> u128 {
        self.storage_bytes
    }

    fn arch_numerators(&self, common: &CommonDenominator, row: usize) -> Vec<Integer> {
        let a = row as i64 - self.modes as i64;
        let mut alpha = vec![Integer::from(0); self.modes + 1];
        for (col, value) in common.values.iter().enumerate() {
            let b = col as i64 - self.modes as i64;
            if a == b {
                continue;
            }
            let mut term =
                Integer::from(value * &common.reciprocals[(a - b).unsigned_abs() as usize]);
            if a < b {
                term = -term;
            }
            alpha[b.unsigned_abs() as usize] -= term.clone() * b.signum();
            alpha[a.unsigned_abs() as usize] += term * a.signum();
        }
        alpha
    }

    pub(super) fn arch_coefficients(&self, row: usize) -> (Vec<Rational>, Rational) {
        let a = row as i64 - self.modes as i64;
        if let Some(common) = &self.common {
            let alpha = self.arch_numerators(common, row);
            (
                alpha
                    .into_iter()
                    .map(|value| Rational::from((value, common.denominator.clone())))
                    .collect(),
                self.exact[row].clone() * 2,
            )
        } else {
            let mut alpha = vec![Rational::from(0); self.modes + 1];
            for (col, value) in self.exact.iter().enumerate() {
                let b = col as i64 - self.modes as i64;
                if a == b {
                    continue;
                }
                let term = value.clone() / (a - b);
                alpha[b.unsigned_abs() as usize] -= term.clone() * b.signum();
                alpha[a.unsigned_abs() as usize] += term * a.signum();
            }
            (alpha, self.exact[row].clone() * 2)
        }
    }

    fn prime_coefficients(&self, row: usize) -> Result<(Vec<Rational>, Rational)> {
        let Some(common) = &self.common else {
            return coefficients(self.exact, self.modes, row);
        };
        let alpha = self.arch_numerators(common, row);
        // Exact rational identity: cosine[k] = 2k*alpha[k] + 2v[row]*delta,
        // and sine = -4|row-N|v[row]. No rounded arithmetic is reassociated.
        let mut numerators = alpha
            .into_iter()
            .enumerate()
            .map(|(k, a)| a * (2 * k))
            .collect::<Vec<_>>();
        numerators[row.abs_diff(self.modes)] +=
            Integer::from(&common.values[row] * &common.lcm) * 2;
        let cosine = numerators
            .into_iter()
            .map(|v| Rational::from((v, common.denominator.clone())))
            .collect::<Vec<_>>();
        let sine = self.exact[row].clone() * (-4 * row.abs_diff(self.modes) as i64);
        let bits = cosine
            .iter()
            .chain(std::iter::once(&sine))
            .map(|v| {
                u128::from(v.numer().significant_bits()) + u128::from(v.denom().significant_bits())
            })
            .sum::<u128>();
        if bits > 67_108_864 {
            bail!("prime-response row exceeds exact coefficient budget");
        }
        Ok((cosine, sine))
    }
}

pub(super) fn workspace_bytes(count: usize, base: u32, parallel: bool) -> u128 {
    let workers = if parallel {
        rayon::current_num_threads().min(count)
    } else {
        1
    };
    count as u128 * (u128::from(base * 2 + 4096).div_ceil(8) + 192) * ((workers as u128 + 1) * 32)
}
pub(super) fn evaluate(
    n_modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    vector: &[Float],
    p: u32,
    parallel: bool,
) -> Result<PrimePowerVelocityAction> {
    evaluate_impl(n_modes, power, prime, l, vector, p, parallel, None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_prepared(
    n_modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    vector: &[Float],
    p: u32,
    parallel: bool,
    prepared: &CoefficientSource<'_>,
) -> Result<PrimePowerVelocityAction> {
    evaluate_impl(
        n_modes,
        power,
        prime,
        l,
        vector,
        p,
        parallel,
        Some(prepared),
    )
}

struct BatchEventGuard {
    metadata: Vec<Float>,
    pi: I,
    reduced: I,
    coefficient: I,
    sines: Vec<I>,
    cosines: Vec<I>,
}

pub(super) struct BatchActions {
    pub(super) actions: Vec<PrimePowerVelocityAction>,
    _reservation: Option<xc_core::MemoryReservation>,
}

fn batch_unavailable(strict_gpu: bool, message: String) -> Result<Option<BatchActions>> {
    if strict_gpu {
        Err(anyhow::anyhow!(message)
            .context(xc_numerics::rational_interval_product::ContractionAdmission))
    } else {
        Ok(None)
    }
}

fn validate_batch_event(power: u64, prime: u64) -> Result<()> {
    if power < 2 || prime < 2 || prime > power {
        bail!("invalid finite prime-response source, shape, event or precision");
    }
    let mut remaining = power;
    while remaining.is_multiple_of(prime) {
        remaining /= prime;
    }
    if remaining != 1 {
        bail!("prime-response power and prime do not agree");
    }
    Ok(())
}

#[cfg(test)]
fn prepare_batch_event(
    modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    p: u32,
    work: u32,
    parallel: bool,
) -> Result<Option<BatchEventGuard>> {
    prepare_batch_event_in_context(modes, power, prime, l, p, work, parallel, &mut false)
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch_event_in_context(
    modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    p: u32,
    work: u32,
    parallel: bool,
    exponent_range_checked: &mut bool,
) -> Result<Option<BatchEventGuard>> {
    let one = I::from_i64(1, work);
    let two = I::from_i64(2, work);
    let pi = I::pi(work);
    let length = I::from_float(l, work)?;
    let log_power = I::from_u64(power, work).ln()?;
    let weight = I::from_u64(prime, work).ln()?;
    let sqrt_power = I::from_u64(power, work).sqrt()?;
    let reduced = one.sub(&log_power.div(&length)?);
    let coefficient = weight
        .mul(&log_power)
        .div(&sqrt_power)?
        .div(&length.square())?
        .neg();
    let edge = weight.mul(&two).div(&sqrt_power)?.div(&log_power)?.neg();
    let metadata = [&log_power, &weight, &reduced, &coefficient, &edge]
        .into_iter()
        .map(|v| rounded(v, p))
        .collect::<Result<Option<Vec<_>>>>()?;
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let resources = xc_core::RunResources::current();
    let workers = if parallel {
        rayon::current_num_threads()
            .min(
                resources
                    .as_ref()
                    .map_or(usize::MAX, |run| run.worker_limit()),
            )
            .min(modes + 1)
    } else {
        1
    };
    // MPFR exponent limits are thread-local. Check once, lazily, for this
    // guard's ordered event loop, after preserving metadata error selection.
    if workers > 1 && !*exponent_range_checked {
        xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
        *exponent_range_checked = true;
    }
    let phase_scale = pi.mul(&two).mul(&reduced);
    let phases = (0..=modes)
        .map(|n| phase_scale.mul(&I::from_u64(n as u64, work)))
        .collect::<Vec<_>>();
    let (sines, cosines) = if workers > 1 {
        // Indexed chunks preserve phase order and bound simultaneous native
        // calls even when the surrounding Rayon pool exceeds the run limit.
        // Reenter the run owner because Rayon does not inherit its TLS scope.
        phases
            .par_chunks(phases.len().div_ceil(workers))
            .map(|chunk| {
                let _scope = resources.as_ref().map(xc_core::RunResources::enter);
                chunk.iter().map(I::sin_cos).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .unzip()
    } else {
        phases.iter().map(I::sin_cos).unzip()
    };
    Ok(Some(BatchEventGuard {
        metadata,
        pi,
        reduced,
        coefficient,
        sines,
        cosines,
    }))
}

#[allow(clippy::too_many_arguments)]
fn apply_batch_row(
    cosine: &[(usize, I)],
    sine: Option<&I>,
    event: &BatchEventGuard,
    index: usize,
    modes: usize,
    exponent: i64,
    p: u32,
    work: u32,
) -> Result<Option<Float>> {
    let mut terms = Vec::with_capacity(cosine.len() + 1);
    for (k, factor) in cosine {
        terms.push(factor.mul(&event.cosines[*k]));
    }
    if let Some(sine) = sine {
        terms.push(
            sine.mul(&event.pi)
                .mul(&event.reduced)
                .mul(&event.sines[index.abs_diff(modes)]),
        );
    }
    let sum = I::new(
        Float::with_val_round(
            work,
            Float::sum(terms.iter().map(I::lower)),
            rug::float::Round::Down,
        )
        .0,
        Float::with_val_round(
            work,
            Float::sum(terms.iter().map(I::upper)),
            rug::float::Round::Up,
        )
        .0,
    )?;
    rounded(&scale(&sum.mul(&event.coefficient), exponent)?, p)
}

fn ordered_batch_rows(rows: Vec<Result<Option<Float>>>) -> Result<Option<Vec<Float>>> {
    // In particular, an earlier unresolved row suppresses a later row error,
    // just as the original Result<Option<Vec<_>>> collection does.
    rows.into_iter().collect()
}

type BatchRowResults = Vec<Vec<Result<Option<Float>>>>;

fn select_batch_rows(
    accelerated: Option<BatchRowResults>,
    reference: impl FnOnce() -> BatchRowResults,
) -> BatchRowResults {
    if let Some(rows) = accelerated {
        if !rows.iter().any(|row| row.iter().any(Result::is_err)) {
            return rows;
        }
        // Tighter bounds can resolve a row the reference leaves unresolved.
        // Before exposing any later mathematical error, restore the reference
        // row states and its None-before-Err reduction for every active event.
        // Drop speculative outputs before allocating their replacements.
        drop(rows);
    }
    reference()
}

fn batch_extra_bytes(events: usize, count: usize, modes: usize, base: u32, parallel: bool) -> u128 {
    let workers = if parallel {
        rayon::current_num_threads().min(count)
    } else {
        1
    } as u128;
    let scalar = u128::from(base * 2 + 4096).div_ceil(8) + 128;
    // Phase construction overlaps its input phases with sine/cosine output.
    // Completed p-bit actions coexist with the next guard's row results.
    // Exact rows/term products retain the single-event workspace allowance;
    // the additional converted row is charged for every live row worker.
    scalar
        * (events as u128 * ((modes as u128 + 1) * 6 + count as u128 * 2 + 32)
            + workers * (modes as u128 + 2) * 2)
}

#[cfg(test)]
thread_local! {
    // Exercise heterogeneous retry schedules without changing production
    // precision policy or relying on a platform-specific hard-rounding input.
    static BATCH_TEST_GUARD_FLOORS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    static BATCH_TEST_COMPLETED_GUARDS: std::cell::RefCell<Vec<(usize, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Reuse exact rows and their directed interval conversion across events at
/// each common work precision. None requests the caller's original streaming
/// route when optional batch/result storage is unavailable. Event and row
/// result selection remain ordered; ordinary errors never mean fallback.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_prepared_batch(
    n_modes: usize,
    events: &[(u64, u64)],
    l: &Float,
    vector: &[Float],
    p: u32,
    parallel: bool,
    prepared: &CoefficientSource<'_>,
    optional_workspace_bytes: u128,
) -> Result<Option<BatchActions>> {
    if events.is_empty() {
        return Ok(Some(BatchActions {
            actions: Vec::new(),
            _reservation: None,
        }));
    }
    if !(64..=1_000_000).contains(&p)
        || n_modes > 4096
        || n_modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(vector.len())
        || !l.is_finite()
        || l <= &0
        || l.prec() > 1_000_000
        || vector
            .iter()
            .any(|x| !x.is_finite() || x.prec() > 1_000_000)
    {
        bail!("invalid finite prime-response source, shape, event or precision");
    }
    validate_batch_event(events[0].0, events[0].1)?;
    let strict_gpu = gpu::requires_batch()?;
    if events.len() == 1 && !strict_gpu {
        return Ok(None);
    }
    let base = vector
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(p)
        .max(l.prec())
        .max(p);
    let bytes = workspace_bytes(vector.len(), base, parallel);
    let budget = super::source_working_budget()?;
    if bytes > budget {
        bail!("prime-response action exceeds workspace budget");
    }
    let exponent = vector
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = vector
        .iter()
        .map(|x| scale_float(x, -exponent, base))
        .collect::<Result<Vec<_>>>()?;
    let exact = crate::ccm::certified_roots::boundary::response_points(
        &normalized,
        n_modes,
        if parallel {
            rayon::current_num_threads()
        } else {
            1
        },
        bytes,
        budget,
    )?;
    let available = budget - bytes - exact.workspace_bytes;
    anyhow::ensure!(
        prepared.modes == n_modes && prepared.exact == exact.values.as_slice(),
        "prime-response prepared source mismatch"
    );
    anyhow::ensure!(
        prepared.storage_bytes <= available,
        "prime-response prepared source exceeds workspace budget"
    );
    let extra = batch_extra_bytes(events.len(), vector.len(), n_modes, base, parallel);
    let result_bytes =
        events.len() as u128 * (vector.len() as u128 + 5) * (u128::from(p).div_ceil(8) + 128);
    // The caller's exact source stays live alongside this precision-dependent
    // admission. Its outer table/scratch headroom has not charged this owner.
    let optional_available = (available - prepared.storage_bytes)
        .min(optional_workspace_bytes.saturating_sub(exact.workspace_bytes));
    if extra + result_bytes > optional_available {
        return batch_unavailable(
            strict_gpu,
            format!(
                "explicit CUDA prime-response batch exceeds optional workspace: required={}, available={optional_available}",
                extra + result_bytes,
            ),
        );
    }
    let Some(result_reservation) = reserve_optional(result_bytes) else {
        return batch_unavailable(
            strict_gpu,
            format!(
                "explicit CUDA prime-response batch cannot reserve {result_bytes} result bytes in the active run",
            ),
        );
    };
    let Some(_temporary_reservation) = reserve_optional(extra) else {
        return batch_unavailable(
            strict_gpu,
            format!(
                "explicit CUDA prime-response batch cannot reserve {extra} temporary bytes in the active run",
            ),
        );
    };
    let mut outcomes: Vec<Option<Result<PrimePowerVelocityAction>>> = events
        .iter()
        .map(|&(power, prime)| validate_batch_event(power, prime).err().map(Err))
        .collect();
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = base * 2 + guard;
        let mut active = Vec::new();
        let mut exponent_range_checked = false;
        for (index, &(power, prime)) in events.iter().enumerate() {
            if outcomes[index].is_some() {
                continue;
            }
            #[cfg(test)]
            if BATCH_TEST_GUARD_FLOORS
                .with(|floors| floors.borrow().get(index).copied().unwrap_or(0) > guard)
            {
                continue;
            }
            match prepare_batch_event_in_context(
                n_modes,
                power,
                prime,
                l,
                p,
                work,
                parallel,
                &mut exponent_range_checked,
            ) {
                Ok(Some(event)) => active.push((index, event)),
                Ok(None) => {}
                Err(error) => outcomes[index] = Some(Err(error)),
            }
        }
        if !active.is_empty() {
            let row = |index: usize| -> Vec<Result<Option<Float>>> {
                let (cosine, sine) = match prepared.prime_coefficients(index) {
                    Ok(value) => value,
                    Err(error) => {
                        let message = error.to_string();
                        return active
                            .iter()
                            .map(|_| Err(anyhow::anyhow!(message.clone())))
                            .collect();
                    }
                };
                let cosine = cosine
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| *value != &0)
                    .map(|(k, value)| (k, I::from_rational(value, work)))
                    .collect::<Vec<_>>();
                let sine = (sine != 0).then(|| I::from_rational(&sine, work));
                active
                    .iter()
                    .map(|(_, event)| {
                        apply_batch_row(
                            &cosine,
                            sine.as_ref(),
                            event,
                            index,
                            n_modes,
                            exponent,
                            p,
                            work,
                        )
                    })
                    .collect()
            };
            let accelerated = gpu::apply_rows(
                prepared,
                &active,
                n_modes,
                exponent,
                p,
                work,
                parallel,
                optional_available - extra - result_bytes,
            )?;
            let rows = select_batch_rows(accelerated, || {
                if parallel {
                    (0..vector.len())
                        .into_par_iter()
                        .map(row)
                        .collect::<Vec<_>>()
                } else {
                    (0..vector.len()).map(row).collect::<Vec<_>>()
                }
            });
            let mut columns = (0..active.len())
                .map(|_| Vec::with_capacity(vector.len()))
                .collect::<Vec<_>>();
            for row in rows {
                for (column, value) in columns.iter_mut().zip(row) {
                    column.push(value);
                }
            }
            for ((index, event), column) in active.into_iter().zip(columns) {
                match ordered_batch_rows(column) {
                    Ok(Some(action)) => {
                        #[cfg(test)]
                        BATCH_TEST_COMPLETED_GUARDS
                            .with(|values| values.borrow_mut().push((index, guard)));
                        outcomes[index] = Some(Ok(PrimePowerVelocityAction {
                            log_power: event.metadata[0].clone(),
                            von_mangoldt_weight: event.metadata[1].clone(),
                            reduced_position: event.metadata[2].clone(),
                            velocity_coefficient: event.metadata[3].clone(),
                            edge_jump_coefficient: event.metadata[4].clone(),
                            action,
                        }));
                    }
                    Ok(None) => {}
                    Err(error) => outcomes[index] = Some(Err(error)),
                }
            }
        }
        // Only the original event-order prefix can establish failure. Later
        // speculative errors never outrank an earlier unresolved event.
        let prefix_error = outcomes
            .iter()
            .take_while(|value| value.is_some())
            .position(|value| matches!(value, Some(Err(_))));
        if let Some(index) = prefix_error {
            return Err(outcomes[index].take().unwrap().err().unwrap());
        }
        if outcomes.iter().all(Option::is_some) {
            break;
        }
    }
    let actions = outcomes
        .into_iter()
        .map(|value| {
            value.unwrap_or_else(|| {
                Err(anyhow::anyhow!(
                    "prime-response action rounding unresolved within the guard budget"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(BatchActions {
        actions,
        _reservation: Some(result_reservation),
    }))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_impl(
    n_modes: usize,
    power: u64,
    prime: u64,
    l: &Float,
    vector: &[Float],
    p: u32,
    parallel: bool,
    prepared: Option<&CoefficientSource<'_>>,
) -> Result<PrimePowerVelocityAction> {
    if !(64..=1_000_000).contains(&p)
        || n_modes > 4096
        || n_modes.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(vector.len())
        || !l.is_finite()
        || l <= &0
        || l.prec() > 1_000_000
        || vector
            .iter()
            .any(|x| !x.is_finite() || x.prec() > 1_000_000)
        || power < 2
        || prime < 2
        || prime > power
    {
        bail!("invalid finite prime-response source, shape, event or precision");
    }
    let mut remaining = power;
    while remaining.is_multiple_of(prime) {
        remaining /= prime;
    }
    if remaining != 1 {
        bail!("prime-response power and prime do not agree");
    }
    let base = vector
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(p)
        .max(l.prec())
        .max(p);
    let bytes = workspace_bytes(vector.len(), base, parallel);
    let budget = super::source_working_budget()?;
    if bytes > budget {
        bail!("prime-response action exceeds workspace budget");
    }
    let exponent = vector
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = vector
        .iter()
        .map(|x| scale_float(x, -exponent, base))
        .collect::<Result<Vec<_>>>()?;
    let exact = crate::ccm::certified_roots::boundary::response_points(
        &normalized,
        n_modes,
        if parallel {
            rayon::current_num_threads()
        } else {
            1
        },
        bytes,
        budget,
    )?;
    let available = budget - bytes - exact.workspace_bytes;
    let local;
    let source = if let Some(source) = prepared {
        anyhow::ensure!(
            source.modes == n_modes && source.exact == exact.values.as_slice(),
            "prime-response prepared source mismatch"
        );
        anyhow::ensure!(
            source.storage_bytes <= available,
            "prime-response prepared source exceeds workspace budget"
        );
        source
    } else {
        local = CoefficientSource::new(&exact.values, n_modes, available);
        &local
    };
    // Double the source precision: at a stored approximation of log(power),
    // 1-log(power)/L can be about 2^-p and still needs p relative output bits.
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let work = base * 2 + guard;
        let one = I::from_i64(1, work);
        let two = I::from_i64(2, work);
        let pi = I::pi(work);
        let length = I::from_float(l, work)?;
        let log_power = I::from_u64(power, work).ln()?;
        let weight = I::from_u64(prime, work).ln()?;
        let sqrt_power = I::from_u64(power, work).sqrt()?;
        let reduced = one.sub(&log_power.div(&length)?);
        let coefficient = weight
            .mul(&log_power)
            .div(&sqrt_power)?
            .div(&length.square())?
            .neg();
        let edge = weight.mul(&two).div(&sqrt_power)?.div(&log_power)?.neg();
        let metadata = [&log_power, &weight, &reduced, &coefficient, &edge]
            .into_iter()
            .map(|v| rounded(v, p))
            .collect::<Result<Option<Vec<_>>>>()?;
        let Some(metadata) = metadata else {
            continue;
        };
        let phase_scale = pi.mul(&two).mul(&reduced);
        let phases = (0..=n_modes)
            .map(|n| phase_scale.mul(&I::from_u64(n as u64, work)))
            .collect::<Vec<_>>();
        // Paired MPFR evaluation preserves the separate directed endpoints.
        let (sines, cosines): (Vec<_>, Vec<_>) = phases.iter().map(I::sin_cos).unzip();
        let row = |index: usize| -> Result<Option<Float>> {
            let (cosine, sine) = source.prime_coefficients(index)?;
            let mut terms = Vec::with_capacity(cosine.len() + 1);
            for (factor, value) in cosine.iter().zip(&cosines) {
                if factor != &0 {
                    terms.push(I::from_rational(factor, work).mul(value));
                }
            }
            if sine != 0 {
                terms.push(
                    I::from_rational(&sine, work)
                        .mul(&pi)
                        .mul(&reduced)
                        .mul(&sines[index.abs_diff(n_modes)]),
                );
            }
            let sum = I::new(
                Float::with_val_round(
                    work,
                    Float::sum(terms.iter().map(I::lower)),
                    rug::float::Round::Down,
                )
                .0,
                Float::with_val_round(
                    work,
                    Float::sum(terms.iter().map(I::upper)),
                    rug::float::Round::Up,
                )
                .0,
            )?;
            rounded(&scale(&sum.mul(&coefficient), exponent)?, p)
        };
        let results = if parallel {
            (0..vector.len())
                .into_par_iter()
                .map(row)
                .collect::<Vec<_>>()
        } else {
            (0..vector.len()).map(row).collect::<Vec<_>>()
        };
        if let Some(action) = results.into_iter().collect::<Result<Option<Vec<_>>>>()? {
            return Ok(PrimePowerVelocityAction {
                log_power: metadata[0].clone(),
                von_mangoldt_weight: metadata[1].clone(),
                reduced_position: metadata[2].clone(),
                velocity_coefficient: metadata[3].clone(),
                edge_jump_coefficient: metadata[4].clone(),
                action,
            });
        }
    }
    bail!("prime-response action rounding unresolved within the guard budget")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_action_bits(action: &PrimePowerVelocityAction) -> Vec<(String, u32, bool)> {
        [
            &action.log_power,
            &action.von_mangoldt_weight,
            &action.reduced_position,
            &action.velocity_coefficient,
            &action.edge_jump_coefficient,
        ]
        .into_iter()
        .chain(action.action.iter())
        .map(|v| {
            (
                xc_numerics::prefix::lossless_decimal(v),
                v.prec(),
                v.is_sign_negative(),
            )
        })
        .collect()
    }

    fn batch_exact_source(vector: &[Float], p: u32) -> Vec<Rational> {
        let exponent = vector
            .iter()
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        vector
            .iter()
            .map(|v| scale_float(v, -exponent, p).unwrap().to_rational().unwrap())
            .collect()
    }

    #[test]
    fn batch_phase_parallel_endpoints_and_metadata_errors_match_serial() {
        let bits = |event: &BatchEventGuard| {
            event
                .metadata
                .iter()
                .chain(
                    [&event.pi, &event.reduced, &event.coefficient]
                        .into_iter()
                        .chain(event.sines.iter())
                        .chain(event.cosines.iter())
                        .flat_map(|value| [value.lower(), value.upper()]),
                )
                .map(|value| {
                    (
                        xc_numerics::prefix::lossless_decimal(value),
                        value.prec(),
                        value.is_sign_negative(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                // Exercise a run smaller than the surrounding pool as well.
                for limit in [1, workers] {
                    let resources =
                        xc_core::RunResources::with_worker_limit(1 << 28, limit).unwrap();
                    let _scope = resources.enter();
                    for p in [64, 128, 3386] {
                        let length = Float::with_val(p, 3);
                        let mut checked = false;
                        for (power, prime) in [(2, 2), (9, 3)] {
                            let serial = prepare_batch_event(
                                12,
                                power,
                                prime,
                                &length,
                                p,
                                p * 2 + 64,
                                false,
                            )
                            .unwrap()
                            .unwrap();
                            let parallel = prepare_batch_event_in_context(
                                12,
                                power,
                                prime,
                                &length,
                                p,
                                p * 2 + 64,
                                true,
                                &mut checked,
                            )
                            .unwrap()
                            .unwrap();
                            assert_eq!(bits(&serial), bits(&parallel));
                        }
                        assert_eq!(checked, workers > 1 && limit > 1);
                    }
                    let zero = Float::with_val(64, 0);
                    let serial = prepare_batch_event(12, 2, 2, &zero, 64, 192, false)
                        .err()
                        .unwrap();
                    let mut checked = false;
                    let parallel = prepare_batch_event_in_context(
                        12,
                        2,
                        2,
                        &zero,
                        64,
                        192,
                        true,
                        &mut checked,
                    )
                    .err()
                    .unwrap();
                    assert_eq!(format!("{serial:#}"), format!("{parallel:#}"));
                    assert!(!checked, "metadata errors must precede parallel preflight");
                }
            });
        }
    }

    #[test]
    fn batch_prime_actions_match_reference_bits_precision_and_signed_zero() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(1 << 29, 2).unwrap();
            let _scope = resources.enter();
            let events = [(2, 2), (4, 2), (9, 3), (16, 2), (17, 17)];
            for p in [64, 128] {
                for source in [[0, 0, 0], [1, 0, -1], [1, 2, 1], [-2, 3, 5]] {
                    let mut vector = source.map(|v| Float::with_val(p, v));
                    if source == [0, 0, 0] {
                        vector[0] = -vector[0].clone();
                    }
                    let exact = batch_exact_source(&vector, p);
                    let prepared = CoefficientSource::new(&exact, 1, 1 << 28);
                    for l in [Float::with_val(p, 3), Float::with_val(p, 17).ln()] {
                        let batch = evaluate_prepared_batch(
                            1,
                            &events,
                            &l,
                            &vector,
                            p + 64,
                            true,
                            &prepared,
                            u128::MAX,
                        )
                        .unwrap()
                        .expect("small batch must be admitted");
                        for (&(power, prime), actual) in events.iter().zip(&batch.actions) {
                            let expected = evaluate_prepared(
                                1,
                                power,
                                prime,
                                &l,
                                &vector,
                                p + 64,
                                false,
                                &prepared,
                            )
                            .unwrap();
                            assert_eq!(batch_action_bits(actual), batch_action_bits(&expected));
                        }
                    }
                }
            }
            assert_eq!(
                resources.usage().working_bytes + resources.usage().retained_bytes,
                0
            );
        });
    }

    #[test]
    fn batch_prime_heterogeneous_guard_retries_preserve_ordered_actions() {
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                BATCH_TEST_GUARD_FLOORS.with(|v| v.borrow_mut().clear());
                BATCH_TEST_COMPLETED_GUARDS.with(|v| v.borrow_mut().clear());
            }
        }
        let _clear = Clear;
        let resources = xc_core::RunResources::new(1 << 28).unwrap();
        let _scope = resources.enter();
        let p = 64;
        let vector = [Float::with_val(p, 1)];
        let exact = batch_exact_source(&vector, p);
        let source = CoefficientSource::new(&exact, 0, 0);
        let events = [(2, 2), (4, 2), (9, 3), (16, 2)];
        let l = Float::with_val(p, 3);
        BATCH_TEST_GUARD_FLOORS.with(|v| *v.borrow_mut() = vec![64, 256, 128, 512]);
        let batch = evaluate_prepared_batch(0, &events, &l, &vector, p, false, &source, u128::MAX)
            .unwrap()
            .unwrap();
        assert_eq!(
            BATCH_TEST_COMPLETED_GUARDS.with(|v| v.borrow().clone()),
            vec![(0, 64), (2, 128), (1, 256), (3, 512)]
        );
        for (&(power, prime), actual) in events.iter().zip(&batch.actions) {
            let expected =
                evaluate_prepared(0, power, prime, &l, &vector, p, false, &source).unwrap();
            assert_eq!(batch_action_bits(actual), batch_action_bits(&expected));
        }
    }

    #[test]
    fn batch_prime_errors_follow_original_event_and_row_selection() {
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                let resources = xc_core::RunResources::with_worker_limit(1 << 28, workers).unwrap();
                let _scope = resources.enter();
                let p = 64;
                let vector = [1, 2, 1].map(|value| Float::with_val(p, value));
                let exact = batch_exact_source(&vector, p);
                let source = CoefficientSource::new(&exact, 1, 0);
                let l = Float::with_val(p, 3);
                for events in [
                    vec![(8, 3), (1, 2)],
                    vec![(2, 2), (8, 3), (1, 2)],
                    vec![(2, 2), (1, 2), (8, 3)],
                ] {
                    let expected = events
                        .iter()
                        .map(|&(power, prime)| {
                            evaluate_prepared(1, power, prime, &l, &vector, p, false, &source)
                        })
                        .collect::<Result<Vec<_>>>()
                        .err()
                        .unwrap();
                    for parallel in [false, true] {
                        let actual = evaluate_prepared_batch(
                            1,
                            &events,
                            &l,
                            &vector,
                            p,
                            parallel,
                            &source,
                            u128::MAX,
                        )
                        .err()
                        .unwrap();
                        assert_eq!(format!("{actual:#}"), format!("{expected:#}"));
                    }
                }
            });
        }
        let unresolved =
            ordered_batch_rows(vec![Ok(None), Err(anyhow::anyhow!("later row error"))]).unwrap();
        assert!(unresolved.is_none());
        let earlier = ordered_batch_rows(vec![Err(anyhow::anyhow!("earlier row error")), Ok(None)])
            .err()
            .unwrap();
        assert_eq!(earlier.to_string(), "earlier row error");
    }

    #[test]
    fn batch_prime_optional_denial_releases_storage_and_keeps_streaming_usable() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(1 << 28, 2).unwrap();
            let _scope = resources.enter();
            let p = 128;
            let vector = [2, -3, 1].map(|v| Float::with_val(p, v));
            let exponent = vector
                .iter()
                .filter_map(Float::get_exp)
                .max()
                .map_or(0, i64::from);
            let normalized = vector
                .iter()
                .map(|v| scale_float(v, -exponent, p).unwrap())
                .collect::<Vec<_>>();
            let exact = normalized
                .iter()
                .map(|v| v.to_rational().unwrap())
                .collect::<Vec<_>>();
            let source = CoefficientSource::new(&exact, 1, 0);
            let exact_bytes =
                crate::ccm::certified_roots::boundary::response_point_workspace(&normalized, 1, 2)
                    .unwrap() as u64;
            let occupied = resources
                .try_reserve_retained(resources.maximum_bytes() - exact_bytes - 1)
                .unwrap();
            let before = resources.usage();
            let l = Float::with_val(p, 3);
            let events = [(2, 2), (4, 2)];
            assert!(
                evaluate_prepared_batch(1, &events, &l, &vector, p, true, &source, u128::MAX)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(resources.usage().retained_bytes, before.retained_bytes);
            assert_eq!(resources.usage().working_bytes, before.working_bytes);
            assert!(evaluate_prepared(1, 2, 2, &l, &vector, p, true, &source).is_ok());
            drop(occupied);
            assert!(
                evaluate_prepared_batch(1, &events, &l, &vector, p, true, &source, 0)
                    .unwrap()
                    .is_none()
            );
            let batch =
                evaluate_prepared_batch(1, &events, &l, &vector, p, true, &source, u128::MAX)
                    .unwrap()
                    .unwrap();
            assert!(
                resources.usage().retained_bytes > 0,
                "returned actions must retain their reservation"
            );
            assert_eq!(resources.usage().working_bytes, 0);
            drop(batch);
            assert_eq!(
                resources.usage().working_bytes + resources.usage().retained_bytes,
                0
            );
        });
    }

    #[test]
    fn batch_prime_source_admission_errors_are_not_optional_fallbacks() {
        let resources = xc_core::RunResources::new(1 << 28).unwrap();
        let _scope = resources.enter();
        let p = 64;
        let vector = [Float::with_val(p, 1)];
        let wrong = [Rational::from(0)];
        let source = CoefficientSource::new(&wrong, 0, 0);
        let l = Float::with_val(p, 3);
        let error = evaluate_prepared_batch(
            0,
            &[(2, 2), (1, 2)],
            &l,
            &vector,
            p,
            false,
            &source,
            u128::MAX,
        )
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "prime-response prepared source mismatch");
        let occupied = resources
            .try_reserve_retained(resources.maximum_bytes() - 1)
            .unwrap();
        let error = evaluate_prepared_batch(
            0,
            &[(2, 2), (4, 2)],
            &l,
            &vector,
            p,
            false,
            &source,
            u128::MAX,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("run memory budget exceeded"));
        drop(occupied);
    }

    #[test]
    fn batch_prime_strict_gpu_admission_is_explicit_and_typed() {
        use xc_numerics::rational_interval_product::ContractionAdmission;
        assert!(
            batch_unavailable(false, "optional bytes unavailable".to_owned())
                .unwrap()
                .is_none()
        );
        let error = batch_unavailable(true, "explicit CUDA result bytes unavailable".to_owned())
            .err()
            .unwrap();
        assert!(error.is::<ContractionAdmission>());
        assert!(format!("{error:#}").contains("result bytes unavailable"));
    }

    #[test]
    fn batch_prime_accelerated_error_replays_reference_unresolved_rows() {
        let accelerated = vec![
            vec![Ok(Some(Float::with_val(64, 1)))],
            vec![Err(anyhow::anyhow!("later row"))],
        ];
        let mut replayed = false;
        let rows = select_batch_rows(Some(accelerated), || {
            replayed = true;
            vec![vec![Ok(None)], vec![Err(anyhow::anyhow!("later row"))]]
        });
        assert!(replayed);
        assert!(ordered_batch_rows(rows.into_iter().flatten().collect())
            .unwrap()
            .is_none());
        let rows = select_batch_rows(Some(vec![vec![Ok(Some(Float::with_val(64, 1)))]]), || {
            panic!("successful accelerator rows must not replay")
        });
        assert!(rows[0][0].as_ref().unwrap().is_some());
    }

    #[test]
    #[ignore = "supervised four-worker response batching measurement; no claim-scale run"]
    fn measure_prepared_prime_event_batches() {
        use std::{hint::black_box, time::Instant};
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(512 << 20, 4).unwrap();
            let _scope = resources.enter();
            let events = [(2, 2), (4, 2), (8, 2), (9, 3), (16, 2), (25, 5), (27, 3), (997, 997)];
            for (modes, p) in [(48usize, 512u32), (120, 3386)] {
                let vector = (0..2 * modes + 1).map(|k| {
                    Float::with_val(p, k as i32 - modes as i32 / 2) / (k + 3)
                }).collect::<Vec<_>>();
                let l = Float::with_val(p, 1000).ln();
                let run = |batched| {
                    // Source normalization and compact exact preparation are
                    // timed in both routes, as are all event preparations.
                    let exact = batch_exact_source(&vector, p);
                    let source = CoefficientSource::new(&exact, modes, 512 << 20);
                    if batched {
                        evaluate_prepared_batch(modes, &events, &l, &vector, p, true, &source, u128::MAX)
                            .unwrap().expect("bounded benchmark batch must be admitted")
                    } else {
                        let actions = events.iter().map(|&(power, prime)| {
                            evaluate_prepared(modes, power, prime, &l, &vector, p, true, &source).unwrap()
                        }).collect();
                        BatchActions { actions, _reservation: None }
                    }
                };
                let expected = run(false);
                let actual = run(true);
                assert_eq!(expected.actions.iter().map(batch_action_bits).collect::<Vec<_>>(),
                           actual.actions.iter().map(batch_action_bits).collect::<Vec<_>>());
                drop(expected);
                drop(actual);
                let mut times = [Vec::new(), Vec::new()];
                for repetition in 0..3 {
                    for route in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let started = Instant::now();
                        let result = run(route == 1);
                        let elapsed = started.elapsed().as_secs_f64();
                        black_box(&result.actions);
                        times[route].push(elapsed);
                        drop(result);
                    }
                }
                for values in &mut times { values.sort_by(f64::total_cmp); }
                println!("prime_batch modes={modes} p={p} events=8 workers=4 trials=3 preparation=included bit_identity=passed baseline_median={:.6} batch_median={:.6} ratio={:.4} reserved_peak={}",
                         times[0][1], times[1][1], times[0][1]/times[1][1], resources.usage().peak_bytes);
            }
        });
    }

    #[test]
    fn capacity_prime_response_preserves_values_and_rejects_occupied_shared_budget() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(1 << 29, 2).unwrap();
            let _scope = resources.enter();
            let p = 128;
            let vector = [2, -3, 1].map(|v| Float::with_val(p, v));
            let length = Float::with_val(p, 3);
            let serial = evaluate(1, 2, 2, &length, &vector, p, false).unwrap();
            let parallel = evaluate(1, 2, 2, &length, &vector, p, true).unwrap();
            let bits = |action: &PrimePowerVelocityAction| {
                [
                    &action.log_power,
                    &action.von_mangoldt_weight,
                    &action.reduced_position,
                    &action.velocity_coefficient,
                    &action.edge_jump_coefficient,
                ]
                .into_iter()
                .chain(action.action.iter())
                .map(|v| {
                    (
                        xc_numerics::prefix::lossless_decimal(v),
                        v.prec(),
                        v.is_sign_negative(),
                    )
                })
                .collect::<Vec<_>>()
            };
            assert_eq!(bits(&serial), bits(&parallel));
            assert_eq!(
                resources.usage().working_bytes + resources.usage().retained_bytes,
                0
            );
            assert!(resources.usage().peak_bytes > 0);
            let occupied = resources
                .try_reserve_retained(resources.maximum_bytes() - 1)
                .unwrap();
            let before = resources.usage();
            let error = evaluate(1, 2, 2, &length, &vector, p, true).err().unwrap();
            assert!(format!("{error:#}").contains("run memory budget exceeded"));
            assert_eq!(resources.usage(), before);
            drop(occupied);
        });
    }

    #[test]
    #[ignore = "supervised exact-coefficient measurement; not claim-scale qualification"]
    fn measure_prepared_prime_coefficients() {
        use std::{hint::black_box, time::Instant};
        let resources = xc_core::RunResources::new(1 << 28).unwrap();
        let _scope = resources.enter();
        let (modes, p, events): (usize, u32, usize) = (48, 512, 4);
        let exact = (0..2 * modes + 1)
            .map(|k| {
                (Float::with_val(p, k as i32 - 40) / (k + 3))
                    .to_rational()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let source = CoefficientSource::new(&exact, modes, 1 << 28);
        assert!(source.common.is_some());
        for row in 0..exact.len() {
            assert_eq!(
                source.prime_coefficients(row).unwrap(),
                coefficients(&exact, modes, row).unwrap()
            );
        }
        drop(source);
        let mut times = [Vec::new(), Vec::new()];
        for repetition in 0..5 {
            for route in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                let started = Instant::now();
                let source =
                    CoefficientSource::new(&exact, modes, if route == 0 { 0 } else { 1 << 28 });
                for _ in 0..events {
                    for row in 0..exact.len() {
                        black_box(source.prime_coefficients(black_box(row)).unwrap());
                    }
                }
                times[route].push(started.elapsed().as_secs_f64());
            }
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "Exact coefficient kernel only: modes={modes}, p={p}, repeated_events={events}, trials=5; original/current median {:.6}/{:.6}s; preparation included; exact equality checked; not a claim-scale speed estimate",
            times[0][2], times[1][2]
        );
    }
    #[test]
    fn prepared_prime_storage_is_optional_and_releases_its_reservation() {
        let exact = vec![
            Rational::from((1, 4)),
            Rational::from((-3, 8)),
            Rational::from(0),
        ];
        // A worker without an inherited scope retains the original path.
        assert!(xc_core::RunResources::current().is_none());
        assert!(CoefficientSource::new(&exact, 1, 1 << 30).common.is_none());
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        {
            let source = CoefficientSource::new(&exact, 1, 1 << 30);
            assert!(source.common.is_some());
            assert_eq!(
                u128::from(resources.usage().retained_bytes),
                source.storage_bytes()
            );
        }
        assert_eq!(resources.usage().retained_bytes, 0);
        let _occupied = resources.try_reserve_retained((1 << 30) - 1).unwrap();
        let source = CoefficientSource::new(&exact, 1, 1 << 30);
        assert!(source.common.is_none());
        for row in 0..3 {
            assert_eq!(
                source.prime_coefficients(row).unwrap(),
                coefficients(&exact, 1, row).unwrap()
            );
        }
    }
    #[test]
    fn prepared_prime_coefficients_match_original_exact_rows() {
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        for modes in 0..=8 {
            let exact = (0..2 * modes + 1)
                .map(|k| Rational::from(((k as i32 - 5) * (k as i32 + 1), 1u32 << k)))
                .collect::<Vec<_>>();
            let prepared = CoefficientSource::new(&exact, modes, 1u128 << 30);
            let fallback = CoefficientSource::new(&exact, modes, 0);
            assert!(prepared.common.is_some());
            assert!(fallback.common.is_none());
            for row in 0..exact.len() {
                let expected = coefficients(&exact, modes, row).unwrap();
                assert_eq!(prepared.prime_coefficients(row).unwrap(), expected);
                assert_eq!(fallback.prime_coefficients(row).unwrap(), expected);
            }
        }
        let nondyadic = vec![
            Rational::from((1, 3)),
            Rational::from((1, 5)),
            Rational::from((1, 7)),
        ];
        let fallback = CoefficientSource::new(&nondyadic, 1, 1u128 << 30);
        assert!(fallback.common.is_none());
        for row in 0..3 {
            assert_eq!(
                fallback.prime_coefficients(row).unwrap(),
                coefficients(&nondyadic, 1, row).unwrap()
            );
        }
    }

    #[test]
    fn prepared_prime_actions_match_original_coefficients_across_guards_and_events() {
        let resources = xc_core::RunResources::new(1 << 30).unwrap();
        let _scope = resources.enter();
        for source_precision in [64, 128] {
            let modes = 3;
            let vector = (0..2 * modes + 1)
                .map(|k| {
                    Float::with_val(source_precision, (k as i32 - 2) * (k as i32 + 1)) / (1u32 << k)
                })
                .collect::<Vec<_>>();
            let exponent = vector
                .iter()
                .filter_map(Float::get_exp)
                .max()
                .map_or(0, i64::from);
            let exact = vector
                .iter()
                .map(|v| {
                    scale_float(v, -exponent, source_precision)
                        .unwrap()
                        .to_rational()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let prepared = CoefficientSource::new(&exact, modes, 1u128 << 30);
            let fallback = CoefficientSource::new(&exact, modes, 0);
            for guard in [64, 128, 256] {
                let output = source_precision + guard;
                for (power, prime) in [(2, 2), (4, 2), (9, 3), (16, 2)] {
                    let length = Float::with_val(source_precision, 3);
                    let actual = evaluate_prepared(
                        modes, power, prime, &length, &vector, output, true, &prepared,
                    )
                    .unwrap();
                    let expected = evaluate_prepared(
                        modes, power, prime, &length, &vector, output, false, &fallback,
                    )
                    .unwrap();
                    let fields = |result: &PrimePowerVelocityAction| {
                        [
                            &result.log_power,
                            &result.von_mangoldt_weight,
                            &result.reduced_position,
                            &result.velocity_coefficient,
                            &result.edge_jump_coefficient,
                        ]
                        .into_iter()
                        .chain(result.action.iter())
                        .map(|v| {
                            (
                                xc_numerics::prefix::lossless_decimal(v),
                                v.prec(),
                                v.is_sign_negative(),
                            )
                        })
                        .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        fields(&actual),
                        fields(&expected),
                        "source={source_precision},guard={guard},power={power}"
                    );
                }
            }
        }
    }
    #[test]
    fn verified_prime_kernel_matches_independent_direct_high_precision_matrix() {
        for p in [64, 128, 256] {
            for n in 1..=3usize {
                for length in [3, 5, 7] {
                    for exponent in [-1000i32, 0, 1000] {
                        let vector = (0..2 * n + 1)
                            .map(|k| {
                                Float::with_val(p, (k as i32 - 2) * (k as i32 + 1)) << exponent
                            })
                            .collect::<Vec<_>>();
                        let l = Float::with_val(p, length);
                        let actual = evaluate(n, 4, 2, &l, &vector, p, true).unwrap();
                        let reference=super::super::response_performance_reference::direct_prime_power_velocity_reference(n,4,2,&Float::with_val(1024,&l),&vector.iter().map(|x|Float::with_val(1024,x)).collect::<Vec<_>>(),1024).unwrap();
                        for (i, (got, want)) in
                            actual.action.iter().zip(&reference.action).enumerate()
                        {
                            assert_eq!(
                                *got,
                                Float::with_val(p, want),
                                "p={p},n={n},L={length},e={exponent},row={i}"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn exact_odd_cancellation_and_zero_vector_remain_exact() {
        let p = 128;
        let l = Float::with_val(p, 3);
        let a = Float::with_val(p, 1) << 400u32;
        let odd = evaluate(
            1,
            2,
            2,
            &l,
            &[a.clone(), Float::with_val(p, 0), -a],
            p,
            true,
        )
        .unwrap();
        assert!(odd.action[1].is_zero());
        assert_eq!(odd.action[0], -odd.action[2].clone());
        let zero = evaluate(2, 8, 2, &l, &vec![Float::with_val(p, 0); 5], p, true).unwrap();
        assert!(zero.action.iter().all(Float::is_zero));
        assert!(evaluate(0, 8, 3, &l, &[Float::with_val(p, 1)], p, true).is_err());
        for bad in [0, 63, 1_000_001, u32::MAX] {
            assert!(evaluate(0, 2, 2, &l, &[Float::with_val(p, 1)], bad, true).is_err());
        }
    }
}
