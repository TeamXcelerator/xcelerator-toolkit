//! Phase-local scheduling and ownership, independent of mathematical identity.
//! Uses the shared run ledger; these algorithmic allowances are not an OS RSS guarantee.
use super::*;
use xc_core::{MemoryReservation, RunResources};

pub(super) struct Admission {
    pub workers: usize,
    pub host_workers: Option<usize>,
    pub fixed_bytes: u64,
    pub gl_bytes: u64,
    pub output_bytes: u64,
    pub peak_bytes: u128,
}

fn reserve(resources: &RunResources, bytes: u128) -> Result<MemoryReservation> {
    Ok(resources.try_reserve_working(u64::try_from(bytes)?)?)
}

/// Exact numerator/denominator bit lengths of a finite dyadic, without creating
/// a potentially exponent-sized Rational. The small significand is owned by
/// the planning-prefix reservation.
fn dyadic_bits(value: &Float) -> Result<(u128, u128)> {
    let (mut integer, exponent) = value
        .to_integer_exp()
        .context("finite u-flow source required")?;
    if integer == 0 {
        return Ok((0, 1));
    }
    integer.abs_mut();
    let trailing = i64::from(integer.find_one(0).unwrap_or(0));
    let exponent = i64::from(exponent);
    let cancelled = trailing.min((-exponent).max(0));
    Ok((
        u128::from(integer.significant_bits()) + exponent.max(0) as u128 - cancelled as u128,
        ((-exponent).max(0) - cancelled) as u128 + 1,
    ))
}

pub(super) fn prepare(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    l: &Float,
    vector: &[Float],
    reuse: bool,
    event_batch: usize,
    resources: &RunResources,
) -> Result<Admission> {
    let n = params.n_modes;
    let p = cfg.precision_bits;
    let base = vector
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(p)
        .max(p)
        .max(l.prec());
    let max_work = base + 4096;
    let (host_workers, gpu_bytes) = if reuse {
        prime_response_kernel::admission_configuration()?
    } else {
        (None, 0)
    };
    let maximum_workers = rayon::current_num_threads().min(resources.worker_limit());
    anyhow::ensure!(
        maximum_workers > 0,
        "positive u-flow worker allowance required"
    );
    // Two normalized Float vectors plus a single temporary significand, shape
    // vectors, and checked scheduling metadata. No exact Rational is built here.
    let prefix_bytes = (vector.len() as u128 * 4 + 64) * (u128::from(max_work).div_ceil(8) + 128)
        + (n as u128 + 1) * 64;
    let prefix = reserve(resources, prefix_bytes)?;
    let exponent = vector
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = vector
        .iter()
        .map(|v| scale_float(v, -exponent, base))
        .collect::<Result<Vec<_>>>()?;
    let inner = normalized
        .iter()
        .map(|v| Float::with_val(max_work, v))
        .collect::<Vec<_>>();
    let orders = (0..=n)
        .map(|mode| cfg.quad_points.max(3 * mode + (p / 2) as usize))
        .collect::<Vec<_>>();
    let mut unique = orders.clone();
    unique.sort_unstable();
    unique.dedup();
    let count = unique.iter().map(|&order| order as u128).sum::<u128>();
    let max_order = *unique.last().context("u-flow quadrature order missing")? as u128;
    let slot = u128::from(max_work).div_ceil(8) + 96;
    let fixed = (count * 2 + vector.len() as u128 * 256) * slot;
    let output = vector.len() as u128 * 8 * (u128::from(p).div_ceil(8) + 96);
    let shared = orders
        .first()
        .copied()
        .filter(|order| reuse && orders.iter().filter(|v| *v == order).count() > 1)
        .map(|order| shared_rule_bytes(order, max_work))
        .filter(|bytes| *bytes <= (64u128 << 20))
        .unwrap_or(0);
    let mut numerator = 0;
    let mut denominator = 1;
    for value in &normalized {
        let bits = dyadic_bits(value)?;
        numerator = numerator.max(bits.0);
        denominator = denominator.max(bits.1);
    }
    let distance = n * 2;
    let lcm_bits = distance as u128 * u128::from(usize::BITS - distance.max(1).leading_zeros()) + 1;
    let value_bits = numerator + denominator;
    let common_retained = vector.len() as u128 * (value_bits.div_ceil(8) + 64)
        + (distance as u128 + 1) * (lcm_bits.div_ceil(8) + 64)
        + 4 * ((value_bits + lcm_bits).div_ceil(8) + 64);
    // Number of prime-power events cannot exceed the number of integers in
    // [2,C]. This bounds even an explicitly oversized batch without enumerating
    // events out of their original production order.
    let events = event_batch.min(params.lambda_sq_int().saturating_sub(1) as usize) as u128;
    let scalar = u128::from(max_work * 2 + 4096).div_ceil(8) + 128;
    let live = resources.usage();
    let already_owned =
        u128::from(live.retained_bytes) + u128::from(live.working_bytes) - prefix_bytes;
    let maximum = u128::from(resources.maximum_bytes());
    let mut one_worker = 0;
    let mut selected = None;
    for workers in (1..=maximum_workers).rev() {
        let w = workers.min(n + 1) as u128;
        let scratch = max_order * w * slot * 16;
        let prime = vector.len() as u128
            * (u128::from(max_work * 2 + 4096).div_ceil(8) + 192)
            * ((workers.min(vector.len()) as u128 + 1) * 32);
        let outer_exact = crate::ccm::certified_roots::boundary::response_point_workspace(
            &normalized,
            n,
            workers,
        )?;
        let inner_exact =
            crate::ccm::certified_roots::boundary::response_point_workspace(&inner, n, workers)?;
        let common = if reuse && common_retained <= (64u128 << 20) {
            common_retained
                + workers.min(vector.len()) as u128
                    * (n as u128 + 1)
                    * ((value_bits + lcm_bits + 32).div_ceil(8) + 64)
        } else {
            0
        };
        let batch = scalar
            * (events * ((n as u128 + 1) * 6 + vector.len() as u128 * 2 + 32)
                + workers.min(vector.len()) as u128 * (n as u128 + 2) * 2)
            + events * (vector.len() as u128 + 5) * (u128::from(max_work).div_ceil(8) + 128);
        // Compatibility GL decoding has a shape-derived expansion cap. Include
        // both JSON parses, retained strings, allocator slack, ZIP scratch and
        // recursion stacks for each simultaneous table. This deliberately loose
        // auxiliary allowance is separate from the retained numerical tables.
        let payload_cap = max_order * 2 * u128::from(p + 128) + 4096;
        let gl = if cfg.cache_mode == xc_numerics::quadrature::CacheMode::JsonZip {
            workers.min(unique.len()) as u128 * (128 * payload_cap + (8u128 << 20))
        } else {
            0
        };
        let base_owned = fixed + scratch + prime;
        let quadrature_phase = base_owned + outer_exact + gl;
        let actions_phase =
            base_owned + outer_exact + inner_exact + common + shared + batch + gpu_bytes;
        let peak = quadrature_phase.max(actions_phase);
        if workers == 1 {
            one_worker = already_owned + peak;
        }
        if already_owned + peak <= maximum {
            selected = Some(Admission {
                workers,
                host_workers,
                fixed_bytes: (base_owned - output).try_into()?,
                gl_bytes: gl.try_into()?,
                output_bytes: output.try_into()?,
                peak_bytes: peak,
            });
            break;
        }
    }
    if selected.is_none() {
        let first_other = fixed + max_order * slot * 16;
        if first_other > maximum {
            bail!("u-flow exceeds the declared workspace budget");
        }
        // Preserve the established exact-source failure class when even the
        // smallest pool cannot pass its original conversion gate. The tested
        // inequality guarantees response_points refuses BEFORE conversion.
        let first_exact =
            crate::ccm::certified_roots::boundary::response_point_workspace(&normalized, n, 1)?;
        if first_other + first_exact > maximum {
            return match crate::ccm::certified_roots::boundary::response_points(
                &normalized,
                n,
                1,
                first_other,
                maximum,
            ) {
                Err(error) => Err(error),
                Ok(_) => bail!("u-flow exact-source refusal invariant violated"),
            };
        }
    }
    drop(inner);
    drop(normalized);
    drop(prefix);
    selected.with_context(|| format!("u-flow exceeds the declared workspace budget: even one worker requires {one_worker} aggregate bytes, maximum={maximum}"))
}

pub(super) fn reserve_fixed(
    resources: &RunResources,
    admission: &Admission,
) -> Result<(MemoryReservation, MemoryReservation)> {
    let fixed = resources.try_reserve_working(admission.fixed_bytes)?;
    let output = resources.try_reserve_retained(admission.output_bytes)?;
    Ok((fixed, output))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_uflow_reservation_rechecks_live_pressure_without_leaking_owners() {
        let params = CcmParams::from_lambda_sq_integer(13, 3);
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.precision_bits = 128;
        cfg.quad_points = 1;
        cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
        let l = Float::with_val(128, 13).ln();
        let vector = vec![Float::with_val(128, 0.5); 7];
        let planning = RunResources::with_worker_limit(1 << 30, 4).unwrap();
        let admission = prepare(&params, &cfg, &l, &vector, false, 1, &planning).unwrap();
        assert_eq!(planning.usage().working_bytes, 0);
        let total = admission.fixed_bytes + admission.output_bytes;
        let resources = RunResources::with_worker_limit(total, 4).unwrap();
        let occupied = resources.try_reserve_retained(1).unwrap();
        assert!(reserve_fixed(&resources, &admission).is_err());
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(resources.usage().retained_bytes, 1);
        drop(occupied);
        let (fixed, output) = reserve_fixed(&resources, &admission).unwrap();
        assert_eq!(resources.usage().working_bytes, admission.fixed_bytes);
        assert_eq!(resources.usage().retained_bytes, admission.output_bytes);
        drop(fixed);
        drop(output);
        assert_eq!(resources.usage().working_bytes, 0);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn exact_dyadic_bit_estimate_matches_actual_rationals() {
        for p in [64, 129, 384] {
            for exponent in [-1000, -1, 0, 1, 1000] {
                for numerator in [-7, -2, 0, 1, 6] {
                    let mut value = Float::with_val(p, numerator);
                    value <<= exponent;
                    let exact = value.to_rational().unwrap();
                    assert_eq!(
                        dyadic_bits(&value).unwrap(),
                        (
                            u128::from(exact.numer().significant_bits()),
                            u128::from(exact.denom().significant_bits())
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn admitted_uflow_large_shapes_are_metadata_only_and_keep_existing_pressure() {
        rayon::ThreadPoolBuilder::new().num_threads(32).build().unwrap().install(|| {
            for n in [2400, 2800] {
              for cache_mode in [xc_numerics::quadrature::CacheMode::Off, xc_numerics::quadrature::CacheMode::JsonZip] {
                let mut cfg = HighPrecConfig::for_decimal_digits(1250);
                cfg.precision_bits = 4217;
                cfg.quad_points = 3750;
                cfg.cache_mode = cache_mode;
                let params = CcmParams::from_lambda_sq_integer(200, n);
                let l = Float::with_val(4217, 200).ln();
                let vector = vec![Float::with_val(4217, 0.5); 2 * n + 1];
                let resources = RunResources::with_worker_limit(84 << 30, 32).unwrap();
                let _retained = resources.try_reserve_retained(20 << 30).unwrap();
                let admitted = prepare(&params, &cfg, &l, &vector, true, 64, &resources).unwrap();
                assert!(admitted.workers < 32);
                assert!(admitted.peak_bytes + (20u128 << 30) <= (84u128 << 30));
                println!("ADMISSION_ONLY N={n} cache_mode={cache_mode:?} synthetic_source=true workers={} peak_bytes={} external_retained_bytes={} numerical_dispatch=false",
                    admitted.workers, admitted.peak_bytes, 20u64 << 30);
                assert_eq!(resources.usage().working_bytes, 0);
                let _occupied = resources.try_reserve_retained(60 << 30).unwrap();
                assert!(prepare(&params, &cfg, &l, &vector, true, 64, &resources).is_err());
                assert_eq!(resources.usage().working_bytes, 0);
              }
            }
        });
    }
}
