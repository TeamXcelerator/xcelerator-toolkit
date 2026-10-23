//! Optional exact GPU contraction of the shared prime cosine matrix.
//!
//! This child keeps the parent's event/guard selection and metadata. Exact
//! integer contractions enclose the same rational action, but need not produce
//! the parent's intermediate interval endpoints. Only a unique nonzero final
//! rounding is accepted here. Zero and unresolved results use the original
//! directed row calculation at this same guard, preserving signed-zero bytes
//! and avoiding an enclosure-width regression. Backend failures never retry.

use super::*;
use xc_numerics::exact_product::ProductShape;
use xc_numerics::modular_product::ProductBackend;
use xc_numerics::rational_interval_product::{
    self, ContractionAdmission, ContractionExecution, ContractionOptions,
};

type RowResults = Vec<Vec<Result<Option<Float>>>>;

fn parse_backend(value: &str) -> Result<ProductBackend> {
    match value {
        "cpu" => Ok(ProductBackend::Cpu),
        "auto" => Ok(ProductBackend::Auto),
        "cuda" => Ok(ProductBackend::Cuda),
        _ => bail!("XC_PRIME_RESPONSE_BACKEND must be cpu, auto, or cuda"),
    }
}

fn configured_backend() -> Result<ProductBackend> {
    match std::env::var("XC_PRIME_RESPONSE_BACKEND") {
        Ok(value) => parse_backend(&value),
        Err(std::env::VarError::NotPresent) => Ok(ProductBackend::Cpu),
        Err(error) => Err(error.into()),
    }
}

/// Strict CUDA must not disappear through the parent's optional batch refusal.
pub(super) fn requires_batch() -> Result<bool> {
    Ok(configured_backend()? == ProductBackend::Cuda)
}

/// Validate explicit execution policy before expensive quadrature preparation.
pub(super) fn validate_configuration() -> Result<()> {
    if configured_backend()? == ProductBackend::Cuda {
        configured_workspace()?;
        configured_tile_rows()?;
        configured_host_workers(true)?;
    }
    Ok(())
}

fn configured_workspace() -> Result<u64> {
    match std::env::var("XC_PRIME_RESPONSE_GPU_MAXIMUM_BYTES") {
        Ok(value) => {
            anyhow::ensure!(
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
                "XC_PRIME_RESPONSE_GPU_MAXIMUM_BYTES must be positive ASCII u64 bytes"
            );
            let bytes = value.parse::<u64>()?;
            anyhow::ensure!(bytes > 0, "positive prime-response GPU workspace required");
            Ok(bytes)
        }
        Err(std::env::VarError::NotPresent) => Ok(4 << 30),
        Err(error) => Err(error.into()),
    }
}

fn configured_tile_rows() -> Result<usize> {
    match std::env::var("XC_PRIME_RESPONSE_GPU_TILE_ROWS") {
        Ok(value) => {
            anyhow::ensure!(
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
                "XC_PRIME_RESPONSE_GPU_TILE_ROWS must be positive ASCII rows"
            );
            let rows = value.parse::<usize>()?;
            anyhow::ensure!(
                (1..=8193).contains(&rows),
                "GPU tile rows must be between 1 and 8193"
            );
            Ok(rows)
        }
        Err(std::env::VarError::NotPresent) => Ok(64),
        Err(error) => Err(error.into()),
    }
}

fn host_worker_policy(value: Option<&str>, allowed: usize) -> Result<usize> {
    let Some(value) = value else {
        return Ok(allowed.min(4));
    };
    anyhow::ensure!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "XC_PRIME_RESPONSE_GPU_HOST_WORKERS must be positive ASCII workers"
    );
    let workers = value.parse::<usize>()?;
    anyhow::ensure!(
        workers > 0,
        "positive prime-response GPU host workers required"
    );
    if workers > allowed {
        return Err(anyhow::anyhow!(
            "prime-response GPU host workers {workers} exceed admitted caller/pool/resource limit {allowed}"
        ).context(ContractionAdmission));
    }
    Ok(workers)
}

fn configured_host_workers(parallel: bool) -> Result<usize> {
    let allowed = rayon::current_num_threads()
        .min(
            xc_core::RunResources::current()
                .map_or(usize::MAX, |resources| resources.worker_limit()),
        )
        .min(if parallel { usize::MAX } else { 1 });
    match std::env::var("XC_PRIME_RESPONSE_GPU_HOST_WORKERS") {
        Ok(value) => host_worker_policy(Some(&value), allowed),
        Err(std::env::VarError::NotPresent) => host_worker_policy(None, allowed),
        Err(error) => Err(error.into()),
    }
}

fn admit_integer_row(
    numerators: &[Integer],
    denominator: &Integer,
    sine: &Rational,
    maximum_bits: u128,
) -> Result<()> {
    let sine_bits =
        u128::from(sine.numer().significant_bits()) + u128::from(sine.denom().significant_bits());
    // Canonical reduction only removes a common factor. This upper bound can
    // certify the ORIGINAL reduced-rational budget without performing GCDs.
    // Zero is canonical 0/1 regardless of the common denominator.
    let upper = sine_bits
        + numerators
            .iter()
            .map(|value| {
                if value == &0 {
                    1
                } else {
                    u128::from(value.significant_bits())
                        + u128::from(denominator.significant_bits())
                }
            })
            .sum::<u128>();
    if upper <= maximum_bits {
        return Ok(());
    }
    // A loose bound must not reject a source admitted by the original kernel.
    let exact = sine_bits
        + numerators
            .iter()
            .map(|value| {
                let reduced = Rational::from((value.clone(), denominator.clone()));
                u128::from(reduced.numer().significant_bits())
                    + u128::from(reduced.denom().significant_bits())
            })
            .sum::<u128>();
    if exact > maximum_bits {
        bail!("prime-response row exceeds exact coefficient budget");
    }
    Ok(())
}

fn integer_row(prepared: &CoefficientSource<'_>, row: usize) -> Result<(Vec<Integer>, Rational)> {
    let common = prepared
        .common
        .as_ref()
        .expect("common source admitted before row preparation");
    // This is the same exact identity used by prime_coefficients before it
    // reduces every A/D to Rational. Keep A directly instead of reducing and
    // then dividing D by the reduced denominator to reconstruct A again.
    let mut numerators = prepared
        .arch_numerators(common, row)
        .into_iter()
        .enumerate()
        .map(|(k, alpha)| alpha * (2 * k))
        .collect::<Vec<_>>();
    numerators[row.abs_diff(prepared.modes)] +=
        Integer::from(&common.values[row] * &common.lcm) * 2;
    let sine = prepared.exact[row].clone() * (-4 * row.abs_diff(prepared.modes) as i64);
    admit_integer_row(&numerators, &common.denominator, &sine, 67_108_864)?;
    Ok((numerators, sine))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_rows(
    prepared: &CoefficientSource<'_>,
    active: &[(usize, BatchEventGuard)],
    modes: usize,
    exponent: i64,
    p: u32,
    work: u32,
    parallel: bool,
    available: u128,
) -> Result<Option<RowResults>> {
    let backend = configured_backend()?;
    // Automatic GPU selection is deliberately unqualified until end-to-end
    // timings establish a winning range. The explicit CUDA path is strict.
    if backend != ProductBackend::Cuda {
        return Ok(None);
    }
    let workspace = available.min(u128::from(configured_workspace()?));
    Ok(Some(apply_rows_with_backend(
        prepared, active, modes, exponent, p, work, parallel, workspace, backend,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn apply_rows_with_backend(
    prepared: &CoefficientSource<'_>,
    active: &[(usize, BatchEventGuard)],
    modes: usize,
    exponent: i64,
    p: u32,
    work: u32,
    parallel: bool,
    workspace: u128,
    backend: ProductBackend,
) -> Result<RowResults> {
    let common = prepared.common.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "exact prime-response GPU contraction requires admitted common-denominator preparation"
        )
        .context(ContractionAdmission)
    })?;
    anyhow::ensure!(!active.is_empty(), "nonempty GPU response batch required");
    anyhow::ensure!(prepared.modes == modes, "GPU response source mode mismatch");
    let inner = modes + 1;
    let count = prepared.exact.len();
    let tile_rows = count.min(configured_tile_rows()?);
    let workers = configured_host_workers(parallel)?;
    let left_bits = common
        .values
        .iter()
        .map(Integer::significant_bits)
        .max()
        .unwrap_or(0)
        + common.lcm.significant_bits()
        + (usize::BITS - count.max(1).leading_zeros())
        + (usize::BITS - (2 * modes).max(1).leading_zeros())
        + 4;
    let scalar_bytes = u128::from(work).div_ceil(8) + 128;
    // This reservation covers borrowed operands excluded from the numerical
    // helper: the common integer row tile, cloned cosine columns, sine factors,
    // reduced rational row construction, and a serial reference-row fallback.
    // The parent's admitted extra/result space covers its events and row results.
    let operands_bytes = 4
        * (tile_rows as u128 * inner as u128 * (u128::from(left_bits).div_ceil(8) + 128)
            + active.len() as u128 * inner as u128 * 2 * scalar_bytes
            + tile_rows as u128 * 2 * scalar_bytes
            + workers.min(tile_rows) as u128
                * inner as u128
                * 8
                * (u128::from(left_bits).div_ceil(8) + scalar_bytes));
    if workspace <= operands_bytes || workspace > u128::from(u64::MAX) {
        return Err(
            anyhow::anyhow!("prime-response GPU operand workspace admission refused")
                .context(ContractionAdmission),
        );
    }
    let resources = xc_core::RunResources::current().ok_or_else(|| {
        anyhow::anyhow!("prime-response GPU contraction requires an active resource ledger")
            .context(ContractionAdmission)
    })?;
    let _operands_reservation = resources
        .try_reserve_working(operands_bytes as u64)
        .map_err(|error| anyhow::anyhow!(error).context(ContractionAdmission))?;
    let columns = (0..inner)
        .flat_map(|k| {
            active
                .iter()
                .map(move |(_, event)| event.cosines[k].clone())
        })
        .collect::<Vec<_>>();
    let options = ContractionOptions {
        backend,
        max_workspace_bytes: (workspace - operands_bytes) as u64,
        max_integer_bits: 67_108_864,
        max_exponent_span: 67_108_864,
        max_scalar_products: 1_000_000_000_000,
        max_modular_scalar_products: 1_000_000_000_000,
        workers,
    };
    let mut rows = Vec::with_capacity(count);
    for start in (0..count).step_by(tile_rows) {
        let end = (start + tile_rows).min(count);
        let mut numerators = Vec::with_capacity((end - start) * inner);
        let mut sines = Vec::with_capacity(end - start);
        let mut failures = Vec::with_capacity(end - start);
        let jobs = workers.min(end - start);
        // Exactly `jobs` serial row producers bound preparation concurrency,
        // even when the surrounding Rayon pool has hundreds of workers.
        let mut prepared_rows = (0..jobs)
            .into_par_iter()
            .map(|job| {
                (start + job..end)
                    .step_by(jobs)
                    .map(|row| (row, integer_row(prepared, row)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        prepared_rows.sort_by_key(|(row, _)| *row);
        for (_, result) in prepared_rows {
            match result {
                Ok((cosine, sine)) => {
                    numerators.extend(cosine);
                    sines.push((sine != 0).then(|| I::from_rational(&sine, work)));
                    failures.push(None);
                }
                Err(error) => {
                    numerators.extend((0..inner).map(|_| Integer::new()));
                    sines.push(None);
                    failures.push(Some(error.to_string()));
                }
            }
        }
        // No error from contract, including post-entry resource refusal, is
        // interpreted as permission to switch backends or arithmetic routes.
        let contraction = rational_interval_product::contract(
            &numerators,
            &common.denominator,
            &columns,
            ProductShape {
                rows: end - start,
                inner,
                cols: active.len(),
            },
            work,
            options,
        )
        .map_err(|error| {
            if start > 0 {
                error.context(ContractionExecution)
            } else {
                error
            }
        })?;
        for row in start..end {
            let local = row - start;
            if let Some(message) = &failures[local] {
                rows.push(
                    active
                        .iter()
                        .map(|_| Err(anyhow::anyhow!(message.clone())))
                        .collect(),
                );
                continue;
            }
            let mut reference = None;
            let mut values = Vec::with_capacity(active.len());
            for (column, (_, event)) in active.iter().enumerate() {
                let cosine = &contraction.values()[local * active.len() + column];
                let sum = if let Some(sine) = &sines[local] {
                    cosine.add(
                        &sine
                            .mul(&event.pi)
                            .mul(&event.reduced)
                            .mul(&event.sines[row.abs_diff(modes)]),
                    )
                } else {
                    cosine.clone()
                };
                let value = match scale(&sum.mul(&event.coefficient), exponent)
                    .and_then(|value| rounded(&value, p))
                {
                    Ok(value) => value,
                    Err(error) => {
                        values.push(Err(error));
                        continue;
                    }
                };
                if value.as_ref().is_some_and(|value| !value.is_zero()) {
                    values.push(Ok(value));
                    continue;
                }
                // A real exact zero does not specify MPFR's signed-zero byte.
                // Reuse the original operation order for that case, and for a
                // width-limited result that may still resolve at this guard.
                if reference.is_none() {
                    let (cosine, _) = prepared.prime_coefficients(row)?;
                    reference = Some(
                        cosine
                            .iter()
                            .enumerate()
                            .filter(|(_, value)| *value != &0)
                            .map(|(k, value)| (k, I::from_rational(value, work)))
                            .collect::<Vec<_>>(),
                    );
                }
                values.push(apply_batch_row(
                    reference.as_ref().unwrap(),
                    sines[local].as_ref(),
                    event,
                    row,
                    modes,
                    exponent,
                    p,
                    work,
                ));
            }
            rows.push(values);
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_selection_is_explicit() {
        assert_eq!(parse_backend("cpu").unwrap(), ProductBackend::Cpu);
        assert_eq!(parse_backend("auto").unwrap(), ProductBackend::Auto);
        assert_eq!(parse_backend("cuda").unwrap(), ProductBackend::Cuda);
        for invalid in ["", "CPU", " cuda", "cuda ", "approximate"] {
            assert!(parse_backend(invalid).is_err());
        }
    }

    #[test]
    fn host_workers_obey_explicit_and_shared_limits() {
        assert_eq!(host_worker_policy(None, 128).unwrap(), 4);
        assert_eq!(host_worker_policy(None, 2).unwrap(), 2);
        assert_eq!(host_worker_policy(Some("64"), 128).unwrap(), 64);
        for (value, allowed) in [("2", 1), ("129", 128)] {
            assert!(host_worker_policy(Some(value), allowed)
                .unwrap_err()
                .is::<ContractionAdmission>());
        }
        for invalid in ["", "0", "-1", "+4", " 4", "4.0", "18446744073709551616"] {
            assert!(host_worker_policy(Some(invalid), 128).is_err());
        }
    }

    #[test]
    fn raw_integer_rows_equal_original_reduced_coefficients() {
        let resources = xc_core::RunResources::new(256 << 20).unwrap();
        let _scope = resources.enter();
        for modes in [0usize, 1, 3, 8] {
            for precision in [64u32, 128, 3386] {
                let exact = (0..2 * modes + 1)
                    .map(|index| {
                        (Float::with_val(precision, index as i32 - modes as i32) / (index + 5))
                            .to_rational()
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let prepared = CoefficientSource::new(&exact, modes, 256 << 20);
                let denominator = &prepared.common.as_ref().unwrap().denominator;
                for row in 0..exact.len() {
                    let (raw, sine) = integer_row(&prepared, row).unwrap();
                    let actual = raw
                        .into_iter()
                        .map(|value| Rational::from((value, denominator.clone())))
                        .collect::<Vec<_>>();
                    let (expected, expected_sine) = prepared.prime_coefficients(row).unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(sine, expected_sine);
                }
            }
        }
    }

    #[test]
    fn integer_row_budget_fallback_preserves_cancellation_and_exact_refusal() {
        let denominator = (Integer::from(1) << 4096u32) + 17;
        let numerators = [
            Integer::from(&denominator * 3),
            Integer::from(&denominator * -5),
            Integer::new(),
        ];
        let sine = Rational::new();
        // Reduced coefficients 3,-5,0 plus sine0 require exactly 9 bits by
        // the legacy numerator+denominator measure, despite huge raw inputs.
        assert!(admit_integer_row(&numerators, &denominator, &sine, 9).is_ok());
        let error = admit_integer_row(&numerators, &denominator, &sine, 8).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prime-response row exceeds exact coefficient budget"
        );
        assert!(admit_integer_row(&[Integer::new()], &denominator, &sine, 2).is_ok());
        assert!(admit_integer_row(&[Integer::new()], &denominator, &sine, 1).is_err());
    }

    fn fixture_compare(backend: ProductBackend, p: u32, modes: usize, guard: u32) {
        let resources = xc_core::RunResources::with_worker_limit(1 << 30, 4).unwrap();
        let _scope = resources.enter();
        for kind in 0..3 {
            let vector = (0..2 * modes + 1)
                .map(|k| {
                    let value = match kind {
                        0 => (k as i32 - modes as i32) * (k as i32 + 3),
                        1 => k as i32 - modes as i32,
                        _ => 0,
                    };
                    let divisor = if kind == 0 {
                        k + 5
                    } else {
                        k.abs_diff(modes) + 5
                    };
                    (Float::with_val(p, value) / divisor) >> (k.abs_diff(modes) % 7) as u32
                })
                .collect::<Vec<_>>();
            let exponent = vector
                .iter()
                .filter_map(Float::get_exp)
                .max()
                .map_or(0, i64::from);
            let exact = vector
                .iter()
                .map(|v| scale_float(v, -exponent, p).unwrap().to_rational().unwrap())
                .collect::<Vec<_>>();
            let prepared = CoefficientSource::new(&exact, modes, 1 << 30);
            let length = Float::with_val(p, 3);
            let work = p * 2 + guard;
            let events = [(2, 2), (4, 2), (9, 3), (16, 2)];
            let active = events
                .iter()
                .enumerate()
                .map(|(index, &(power, prime))| {
                    (
                        index,
                        prepare_batch_event(modes, power, prime, &length, p, work)
                            .unwrap()
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let actual = apply_rows_with_backend(
                &prepared,
                &active,
                modes,
                exponent,
                p,
                work,
                false,
                512 << 20,
                backend,
            )
            .unwrap();
            for (row, values) in actual.into_iter().enumerate() {
                let (cosine, sine) = prepared.prime_coefficients(row).unwrap();
                let cosine = cosine
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| *v != &0)
                    .map(|(k, v)| (k, I::from_rational(v, work)))
                    .collect::<Vec<_>>();
                let sine = (sine != 0).then(|| I::from_rational(&sine, work));
                for ((_, event), got) in active.iter().zip(values) {
                    let expected = apply_batch_row(
                        &cosine,
                        sine.as_ref(),
                        event,
                        row,
                        modes,
                        exponent,
                        p,
                        work,
                    )
                    .unwrap()
                    .unwrap();
                    let got = got.unwrap().unwrap();
                    assert_eq!(got.to_integer_exp(), expected.to_integer_exp());
                    assert_eq!(got.prec(), expected.prec());
                    assert_eq!(got.is_sign_negative(), expected.is_sign_negative());
                }
            }
        }
    }

    #[test]
    fn exact_contraction_rows_match_legacy_actions_and_signed_zeros() {
        for guard in [64, 4096] {
            fixture_compare(ProductBackend::Cpu, 128, 3, guard);
        }
    }

    #[test]
    #[ignore = "supervised CUDA action qualification; includes zero and late guard cases"]
    fn cuda_contraction_rows_match_legacy_actions_and_signed_zeros() {
        for guard in [64, 4096] {
            fixture_compare(ProductBackend::Cuda, 128, 3, guard);
        }
    }

    #[test]
    #[ignore = "supervised high-precision complete action CUDA qualification; no concurrent claim GPU workload"]
    fn cuda_high_precision_actions_match_legacy_including_late_guards() {
        for p in [3386, 6708] {
            for guard in [64, 4096] {
                fixture_compare(ProductBackend::Cuda, p, 3, guard);
            }
        }
    }

    #[test]
    #[ignore = "supervised four-worker complete prime-action CPU/CUDA timing; all preparation included"]
    fn measure_high_precision_prime_actions_cpu_cuda() {
        use std::time::Instant;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(1 << 30, 4).unwrap();
            let _scope = resources.enter();
            let modes = 120usize;
            let events = [(2, 2), (4, 2), (8, 2), (9, 3), (16, 2), (25, 5), (27, 3), (997, 997)];
            for p in [3386u32, 6708] {
                let vector = (0..2 * modes + 1).map(|k| {
                    Float::with_val(p, k as i32 - 73) / (k + 5)
                }).collect::<Vec<_>>();
                let length = Float::with_val(p, 1000).ln();
                let run = |gpu: bool| {
                    let start = Instant::now();
                    let exponent = vector.iter().filter_map(Float::get_exp).max().map_or(0, i64::from);
                    let exact = vector.iter().map(|value| {
                        scale_float(value, -exponent, p).unwrap().to_rational().unwrap()
                    }).collect::<Vec<_>>();
                    let prepared = CoefficientSource::new(&exact, modes, 1 << 30);
                    let work = p * 2 + 64;
                    let active = events.iter().enumerate().map(|(index, &(power, prime))| {
                        (index, prepare_batch_event(modes, power, prime, &length, p, work)
                            .unwrap().expect("benchmark metadata must resolve at first guard"))
                    }).collect::<Vec<_>>();
                    let rows = if gpu {
                        apply_rows_with_backend(&prepared, &active, modes, exponent, p, work,
                            true, 768 << 20, ProductBackend::Cuda).unwrap()
                    } else {
                        (0..vector.len()).into_par_iter().map(|row| {
                            let (cosine, sine) = prepared.prime_coefficients(row).unwrap();
                            let cosine = cosine.iter().enumerate().filter(|(_, value)| *value != &0)
                                .map(|(k, value)| (k, I::from_rational(value, work))).collect::<Vec<_>>();
                            let sine = (sine != 0).then(|| I::from_rational(&sine, work));
                            active.iter().map(|(_, event)| apply_batch_row(&cosine, sine.as_ref(),
                                event, row, modes, exponent, p, work)).collect::<Vec<_>>()
                        }).collect::<Vec<_>>()
                    };
                    let mut actions = active.iter().map(|(_, event)| event.metadata.clone()).collect::<Vec<_>>();
                    for row in rows {
                        for (action, value) in actions.iter_mut().zip(row) {
                            action.push(value.unwrap().expect("benchmark action must resolve at first guard"));
                        }
                    }
                    (actions, start.elapsed().as_secs_f64())
                };
                let (cpu, cpu_seconds) = run(false);
                let (gpu, gpu_seconds) = run(true);
                for (cpu, gpu) in cpu.iter().flatten().zip(gpu.iter().flatten()) {
                    assert_eq!(cpu.to_integer_exp(), gpu.to_integer_exp());
                    assert_eq!(cpu.prec(), gpu.prec());
                    assert_eq!(cpu.is_sign_negative(), gpu.is_sign_negative());
                }
                println!("prime_actions modes=120 p={p} events=8 cpu_workers=4 gpu_host_workers=4 preparation=included metadata_and_action_identity=passed first_guard=64 cpu_seconds={cpu_seconds:.6} cuda_seconds={gpu_seconds:.6} ratio={:.4} ledger_peak={}", cpu_seconds/gpu_seconds, resources.usage().peak_bytes);
            }
        });
    }

    #[test]
    #[ignore = "isolated supervised CUDA batch/tile screening; run exact test with --test-threads=1"]
    fn measure_prime_actions_cuda_batch_tile_sweep() {
        use std::time::Instant;
        struct RestoreEnvironment(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl Drop for RestoreEnvironment {
            fn drop(&mut self) {
                for (key, value) in self.0.iter().rev() {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
        let _restore = RestoreEnvironment(
            [
                "XC_PRIME_RESPONSE_GPU_TILE_ROWS",
                "XC_PRIME_RESPONSE_GPU_HOST_WORKERS",
            ]
            .into_iter()
            .map(|key| (key, std::env::var_os(key)))
            .collect(),
        );
        std::env::set_var("XC_PRIME_RESPONSE_GPU_HOST_WORKERS", "4");
        let all_events = crate::ccm::try_prime_powers_up_to(1000).unwrap();
        // The full-row241 case is screened at32 events. At64 events its
        // admitted storage can exceed this deliberately bounded768MiB cap.
        // Two alternating trials are screening evidence, not an auto policy.
        for (event_count, tile_rows) in [(32, 64), (32, 128), (32, 241), (64, 64), (64, 128)] {
            std::env::set_var("XC_PRIME_RESPONSE_GPU_TILE_ROWS", tile_rows.to_string());
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(|| {
                let resources = xc_core::RunResources::with_worker_limit(1 << 30, 4).unwrap();
                let _scope = resources.enter();
                let (modes, p) = (120usize, 3386u32);
                let events = all_events[..event_count].iter()
                    .map(|&(power, prime, _)| (power, prime)).collect::<Vec<_>>();
                let vector = (0..2 * modes + 1).map(|k| {
                    Float::with_val(p, k as i32 - 73) / (k + 5)
                }).collect::<Vec<_>>();
                let length = Float::with_val(p, 1000).ln();
                let run = |gpu: bool| {
                    let start = Instant::now();
                    let exponent = vector.iter().filter_map(Float::get_exp).max().map_or(0, i64::from);
                    let exact = vector.iter().map(|value| {
                        scale_float(value, -exponent, p).unwrap().to_rational().unwrap()
                    }).collect::<Vec<_>>();
                    let prepared = CoefficientSource::new(&exact, modes, 1 << 30);
                    let work = p * 2 + 64;
                    let active = events.iter().enumerate().map(|(index, &(power, prime))| {
                        (index, prepare_batch_event(modes, power, prime, &length, p, work)
                            .unwrap().expect("screening metadata must resolve at first guard"))
                    }).collect::<Vec<_>>();
                    let rows = if gpu {
                        apply_rows_with_backend(&prepared, &active, modes, exponent, p, work,
                            true, 768 << 20, ProductBackend::Cuda).unwrap()
                    } else {
                        (0..vector.len()).into_par_iter().map(|row| {
                            let (cosine, sine) = prepared.prime_coefficients(row).unwrap();
                            let cosine = cosine.iter().enumerate().filter(|(_, value)| *value != &0)
                                .map(|(k, value)| (k, I::from_rational(value, work))).collect::<Vec<_>>();
                            let sine = (sine != 0).then(|| I::from_rational(&sine, work));
                            active.iter().map(|(_, event)| apply_batch_row(&cosine, sine.as_ref(),
                                event, row, modes, exponent, p, work)).collect::<Vec<_>>()
                        }).collect::<Vec<_>>()
                    };
                    let mut actions = active.iter().map(|(_, event)| event.metadata.clone()).collect::<Vec<_>>();
                    for row in rows {
                        for (action, value) in actions.iter_mut().zip(row) {
                            action.push(value.unwrap().expect("screening action must resolve at first guard"));
                        }
                    }
                    (actions, start.elapsed().as_secs_f64())
                };
                for trial in 0..2 {
                    let (cpu, gpu) = if trial == 0 {
                        (run(false), run(true))
                    } else {
                        let gpu = run(true);
                        (run(false), gpu)
                    };
                    for (cpu, gpu) in cpu.0.iter().flatten().zip(gpu.0.iter().flatten()) {
                        assert_eq!(cpu.to_integer_exp(), gpu.to_integer_exp());
                        assert_eq!(cpu.prec(), gpu.prec());
                        assert_eq!(cpu.is_sign_negative(), gpu.is_sign_negative());
                    }
                    println!("prime_action_sweep modes=120 p=3386 events={event_count} event_first={} event_last={} tile_rows={tile_rows} host_workers=4 trial={trial} preparation=included metadata_and_action_identity=passed cpu_seconds={:.6} cuda_seconds={:.6} ratio={:.4} ledger_peak={} screening_only=true",
                        events[0].0, events.last().unwrap().0, cpu.1, gpu.1, cpu.1/gpu.1,
                        resources.usage().peak_bytes);
                }
                assert!(resources.usage().peak_bytes <= 1 << 30);
            });
            drop(pool);
        }
    }

    #[test]
    #[ignore = "isolated supervised CUDA u-flow integration; run this exact test with --test-threads=1"]
    fn cuda_uflow_compute_path_matches_cpu_all_channels_and_tail_singleton() {
        // Follow the repository's restoring environment-guard convention.
        // This ignored test must run alone; each route's Rayon pool is dropped
        // before restoring/changing environment settings for the next route.
        struct RestoreEnvironment(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl RestoreEnvironment {
            fn set(values: &[(&'static str, &str)]) -> Self {
                let saved = values
                    .iter()
                    .map(|(key, _)| (*key, std::env::var_os(key)))
                    .collect();
                for (key, value) in values {
                    std::env::set_var(key, value);
                }
                Self(saved)
            }
        }
        impl Drop for RestoreEnvironment {
            fn drop(&mut self) {
                for (key, value) in self.0.iter().rev() {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
        type FloatBits = (Option<(Integer, i32)>, u32, bool);
        fn bits(values: &[Float]) -> Vec<FloatBits> {
            values
                .iter()
                .map(|value| {
                    (
                        value.to_integer_exp(),
                        value.prec(),
                        value.is_sign_negative(),
                    )
                })
                .collect()
        }
        let run = |backend: &str| {
            let _environment = RestoreEnvironment::set(&[
                ("XC_PRIME_RESPONSE_BACKEND", backend),
                ("XC_RESPONSE_EVENT_BATCH", "8"),
                ("XC_PRIME_RESPONSE_GPU_MAXIMUM_BYTES", "536870912"),
                ("XC_PRIME_RESPONSE_GPU_TILE_ROWS", "2"),
                ("XC_PRIME_RESPONSE_GPU_HOST_WORKERS", "2"),
            ]);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            pool.install(|| {
                let resources = xc_core::RunResources::with_worker_limit(1 << 30, 2).unwrap();
                let _scope = resources.enter();
                let p = 128;
                let mut cfg = HighPrecConfig::for_decimal_digits(20);
                cfg.precision_bits = p;
                cfg.quad_points = 96;
                cfg.cache_mode = xc_numerics::quadrature::CacheMode::Off;
                let params = CcmParams::from_lambda_sq_integer(13, 1);
                let length = Float::with_val(p, 13).ln();
                let events = crate::ccm::try_prime_powers_up_to(13).unwrap();
                assert!(events.len() > 8);
                assert_eq!(
                    events.len() % 8,
                    1,
                    "the final strict CUDA batch must be a singleton"
                );
                let mut reports = Vec::new();
                for kind in 0..3 {
                    let vector = match kind {
                        0 => [2, -3, 1].map(|value| Float::with_val(p, value)),
                        1 => [1, 0, -1].map(|value| Float::with_val(p, value)),
                        _ => [
                            -Float::with_val(p, 0),
                            Float::with_val(p, 0),
                            -Float::with_val(p, 0),
                        ],
                    };
                    // Use the normal compute entry, including table admission,
                    // integral/prime assembly, guard loops, and event batching.
                    let action =
                        compute_u_flow_velocity_actions(&params, &cfg, &length, &vector).unwrap();
                    let channels = [
                        &action.tau_pole,
                        &action.tau_archimedean,
                        &action.tau_prime,
                        &action.tau_total,
                    ]
                    .map(|values| bits(values));
                    assert!(channels.iter().all(|values| values.len() == 3));
                    assert_eq!(resources.usage().working_bytes, 0);
                    assert_eq!(resources.usage().retained_bytes, 0);

                    // UFlowVelocityActions contains only the four channels.
                    // Also compare every prime-action metadata field through
                    // the same public-to-parent batch delegate, with its tail.
                    let exponent = vector
                        .iter()
                        .filter_map(Float::get_exp)
                        .max()
                        .map_or(0, i64::from);
                    let exact = vector
                        .iter()
                        .map(|value| {
                            scale_float(value, -exponent, p)
                                .unwrap()
                                .to_rational()
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let prepared = CoefficientSource::new(&exact, 1, 1 << 30);
                    let mut metadata = Vec::new();
                    for batch in events.chunks(8) {
                        let pairs = batch
                            .iter()
                            .map(|&(power, prime, _)| (power, prime))
                            .collect::<Vec<_>>();
                        let owner = evaluate_prepared_batch(
                            1,
                            &pairs,
                            &length,
                            &vector,
                            p,
                            true,
                            &prepared,
                            768 << 20,
                        )
                        .unwrap();
                        let actions = match owner.as_ref() {
                            Some(owner) => &owner.actions,
                            None => {
                                assert_eq!(backend, "cpu");
                                // CPU singleton storage follows the existing
                                // streaming route; explicit CUDA must be Some.
                                let (power, prime) = pairs[0];
                                let value = evaluate_prepared(
                                    1, power, prime, &length, &vector, p, true, &prepared,
                                )
                                .unwrap();
                                metadata.push(bits(&[
                                    value.log_power,
                                    value.von_mangoldt_weight,
                                    value.reduced_position,
                                    value.velocity_coefficient,
                                    value.edge_jump_coefficient,
                                ]));
                                continue;
                            }
                        };
                        for value in actions {
                            metadata.push(bits(&[
                                value.log_power.clone(),
                                value.von_mangoldt_weight.clone(),
                                value.reduced_position.clone(),
                                value.velocity_coefficient.clone(),
                                value.edge_jump_coefficient.clone(),
                            ]));
                        }
                    }
                    drop(prepared);
                    assert_eq!(metadata.len(), events.len());
                    assert_eq!(resources.usage().working_bytes, 0);
                    assert_eq!(resources.usage().retained_bytes, 0);
                    reports.push((channels, metadata));
                }
                assert!(resources.usage().peak_bytes <= 1 << 30);
                reports
            })
        };
        let cpu = run("cpu");
        let cuda = run("cuda");
        assert_eq!(cpu, cuda);
    }
}
