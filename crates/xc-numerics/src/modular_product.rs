// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Exact signed integer matrix products using small prime residues and CRT.
//!
//! Inputs and outputs are arbitrary-precision integers. No floating-point
//! approximation or truncated reconstruction is used. A balanced product tree
//! handles conversion and reconstruction. The optional CUDA backend performs
//! exact INT8 products with checked INT32 accumulation; CPU reconstruction
//! recovers the unique result within a proved integer bound.
//!
//! Callers choose their solver and cache identities. An exact dot product
//! followed by rounding generally differs from sequentially rounded
//! floating-point operations.
//!
//! With `cuda-product`, products with at least 128 residue primes per device prefer a
//! two-stream engine with pinned staging and batches of three exact digit
//! products. The default keeps the synchronous engine if pipeline symbols or
//! additional admitted storage are unavailable, preserving its device set.
//! `XC_CUDA_PRODUCT_PIPELINE=0` forces synchronous execution; `1` requires the
//! pipeline and fails closed if unavailable. Runtime errors never trigger a
//! different arithmetic route. Hardware speed depends on the product and host.

use anyhow::{bail, ensure, Context, Result};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use std::time::Instant;

#[cfg(feature = "cuda-product")]
mod cuda;

/// Automatic selection falls back only before any GPU computation. A selected
/// device's numerical or transfer error is always returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductBackend {
    Cpu,
    Auto,
    #[cfg(feature = "cuda-product")]
    Cuda,
}

/// Runtime and visible-device information. Version integers use NVIDIA's own
/// encoding; a driver's supported CUDA version is not the installed runtime.
#[derive(Clone, Debug, Default)]
pub struct CudaStatus {
    pub available: bool,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub cublas_version: Option<i32>,
    pub devices: Vec<CudaDeviceInfo>,
    pub unavailable_reason: Option<String>,
}
impl CudaStatus {
    fn unavailable(reason: String) -> Self {
        Self {
            unavailable_reason: Some(reason),
            ..Self::default()
        }
    }
}
#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    /// Physical PCI identity; ordinals are relative to CUDA_VISIBLE_DEVICES.
    pub pci_bus_id: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub free_memory_bytes: u64,
    pub total_memory_bytes: u64,
}
/// Discover CUDA 12/13 and the visible compatible devices. CUDA libraries are
/// loaded on first use from XC_CUDA_LIB_DIR, CUDA_PATH, or the system loader.
/// XC_CUDA_DEVICES optionally restricts CUDA_VISIBLE_DEVICES-relative ordinals.
pub fn cuda_status() -> Result<CudaStatus> {
    #[cfg(feature = "cuda-product")]
    {
        cuda::status()
    }
    #[cfg(not(feature = "cuda-product"))]
    {
        Ok(CudaStatus::unavailable(
            "CUDA support was disabled at build time".to_owned(),
        ))
    }
}

/// Caller-supplied limits, checked before residue/answer arrays are allocated.
/// Workspace includes configured worker stacks, host arithmetic arrays, and
/// admitted device buffers, but not
/// caller-owned operands, allocator overhead, or CUDA context/library storage.
/// A process-level host admission remains necessary.
#[derive(Clone, Copy, Debug)]
pub struct ProductLimits {
    pub max_workspace_bytes: u64,
    pub max_input_bits: u64,
    pub max_modular_scalar_products: u64,
    pub workers: usize,
}

/// Allocation admission before residue conversion or any accelerator work.
/// Output tiles may be split on this error; numerical, domain, and CUDA runtime
/// failures deliberately use different errors and must never be retried as it.
#[derive(Debug)]
pub(crate) struct ModularWorkspaceLimit {
    stage: &'static str,
    pub(crate) required: u128,
    pub(crate) available: u128,
}
impl std::fmt::Display for ModularWorkspaceLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} workspace budget exceeded: requires {}, available {}",
            self.stage, self.required, self.available
        )
    }
}
impl std::error::Error for ModularWorkspaceLimit {}

/// A single-pass CRT domain refusal, established before any backend execution.
/// Only this marker permits the HP adapter to select exact limb products.
#[derive(Debug)]
pub(crate) struct ReconstructionCapacity(&'static str);
impl std::fmt::Display for ReconstructionCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ReconstructionCapacity {}

pub(crate) fn admit_workspace(required: u128, available: u128, stage: &'static str) -> Result<()> {
    if required > available {
        return Err(ModularWorkspaceLimit {
            stage,
            required,
            available,
        }
        .into());
    }
    Ok(())
}

/// Stage times describe this invocation, not a device-only benchmark.
#[derive(Clone, Debug, Default)]
pub struct ProductTimings {
    pub planning_seconds: f64,
    pub conversion_seconds: f64,
    pub products_seconds: f64,
    pub reconstruction_seconds: f64,
}

/// Successful residue products on an admitted physical device. These are
/// execution measurements and never enter numerical or cache identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceWork {
    pub ordinal: usize,
    pub successful_moduli: usize,
}

// Operational logging is independent of product reports and artifact identity.
// Keep the policy testable without CUDA, environment changes, or wall-clock waits.
#[cfg(any(feature = "cuda-product", test))]
mod cuda_logging {
    use super::CudaDeviceWork;
    use std::time::Duration;

    const SUMMARY_INTERVAL: Duration = Duration::from_secs(30);
    const MAX_DEVICES: usize = 64;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Mode {
        Summary,
        Verbose,
    }

    pub(super) fn parse(value: Option<&str>) -> (Mode, bool) {
        match value {
            None | Some("summary") => (Mode::Summary, false),
            Some("verbose") => (Mode::Verbose, false),
            _ => (Mode::Summary, true),
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(super) struct DeviceTotals {
        pub ordinal: usize,
        pub completed_products: u64,
        pub successful_moduli: u64,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(super) struct Totals {
        pub completed_exact_products: u64,
        pub synchronous_products: u64,
        pub pipelined_products: u64,
        pub devices: Vec<DeviceTotals>,
        // Assignments to ordinals beyond the first MAX_DEVICES distinct devices.
        pub overflow_device_assignments: u64,
        pub overflow_successful_moduli: u64,
        pub saturated: bool,
    }

    fn add(counter: &mut u64, amount: u64) -> bool {
        let overflow = counter.checked_add(amount).is_none();
        *counter = counter.saturating_add(amount);
        overflow
    }

    #[derive(Default)]
    pub(super) struct State {
        admitted: bool,
        last_summary: Option<Duration>,
        totals: Totals,
    }

    impl State {
        pub(super) fn first_admission(&mut self) -> bool {
            !std::mem::replace(&mut self.admitted, true)
        }

        pub(super) fn completed(
            &mut self,
            now: Duration,
            pipelined: bool,
            work: &[CudaDeviceWork],
        ) -> Option<Totals> {
            let totals = &mut self.totals;
            totals.saturated |= add(&mut totals.completed_exact_products, 1);
            totals.saturated |= add(
                if pipelined {
                    &mut totals.pipelined_products
                } else {
                    &mut totals.synchronous_products
                },
                1,
            );
            for device in work {
                let index = totals
                    .devices
                    .iter()
                    .position(|entry| entry.ordinal == device.ordinal)
                    .or_else(|| {
                        if totals.devices.len() == MAX_DEVICES {
                            None
                        } else {
                            totals.devices.push(DeviceTotals {
                                ordinal: device.ordinal,
                                ..DeviceTotals::default()
                            });
                            Some(totals.devices.len() - 1)
                        }
                    });
                let moduli = u64::try_from(device.successful_moduli).unwrap_or(u64::MAX);
                if let Some(index) = index {
                    let entry = &mut totals.devices[index];
                    totals.saturated |= add(&mut entry.completed_products, 1);
                    totals.saturated |= add(&mut entry.successful_moduli, moduli);
                } else {
                    totals.saturated |= add(&mut totals.overflow_device_assignments, 1);
                    totals.saturated |= add(&mut totals.overflow_successful_moduli, moduli);
                }
            }
            if self
                .last_summary
                .is_some_and(|last| now.saturating_sub(last) < SUMMARY_INTERVAL)
            {
                return None;
            }
            self.last_summary = Some(now);
            Some(totals.clone())
        }
    }

    #[cfg(feature = "cuda-product")]
    mod process {
        use super::{parse, Mode, State};
        use crate::modular_product::CudaDeviceWork;
        use std::sync::{Mutex, OnceLock};
        use std::time::Instant;

        fn settings() -> (Mode, bool) {
            static SETTINGS: OnceLock<(Mode, bool)> = OnceLock::new();
            *SETTINGS.get_or_init(|| match std::env::var("XC_CUDA_LOG") {
                Ok(value) => parse(Some(&value)),
                Err(std::env::VarError::NotPresent) => parse(None),
                Err(std::env::VarError::NotUnicode(_)) => parse(Some("")),
            })
        }

        fn state() -> &'static (Instant, Mutex<State>) {
            static STATE: OnceLock<(Instant, Mutex<State>)> = OnceLock::new();
            STATE.get_or_init(|| (Instant::now(), Mutex::new(State::default())))
        }

        pub(crate) fn verbose() -> bool {
            settings().0 == Mode::Verbose
        }

        pub(crate) fn admission(message: impl FnOnce() -> String) {
            let (mode, invalid) = settings();
            let emit = mode == Mode::Verbose || {
                let mut state = state().1.lock().unwrap_or_else(|error| error.into_inner());
                state.first_admission()
            };
            // Never format or invoke a progress callback under the state lock.
            if emit {
                if invalid {
                    xc_core::emit_message(
                        "[CUDA warning] unrecognized XC_CUDA_LOG; using summary (expected summary or verbose)".to_owned(),
                    );
                }
                xc_core::emit_message(message());
            }
        }

        pub(crate) fn completed(pipelined: bool, work: &[CudaDeviceWork]) {
            if verbose() {
                return;
            }
            let (start, mutex) = state();
            let snapshot = {
                let mut state = mutex.lock().unwrap_or_else(|error| error.into_inner());
                state.completed(start.elapsed(), pipelined, work)
            };
            if let Some(totals) = snapshot {
                xc_core::emit_message(format!(
                    "[CUDA summary] process cumulative completed exact products: {}; synchronous: {}; pipelined: {}; device work: {:?}; overflow device assignments: {}; overflow successful moduli: {}; counters saturated: {}",
                    totals.completed_exact_products,
                    totals.synchronous_products,
                    totals.pipelined_products,
                    totals.devices,
                    totals.overflow_device_assignments,
                    totals.overflow_successful_moduli,
                    totals.saturated,
                ));
            }
        }
    }

    #[cfg(feature = "cuda-product")]
    pub(super) use process::{admission, completed, verbose};

    #[cfg(test)]
    mod tests {
        use super::*;

        fn work(ordinal: usize, successful_moduli: usize) -> CudaDeviceWork {
            CudaDeviceWork {
                ordinal,
                successful_moduli,
            }
        }

        #[test]
        fn cuda_logging_selector_defaults_and_invalid_values_are_operational() {
            assert_eq!(parse(None), (Mode::Summary, false));
            assert_eq!(parse(Some("summary")), (Mode::Summary, false));
            assert_eq!(parse(Some("verbose")), (Mode::Verbose, false));
            for value in ["", "0", "Verbose", "invalid"] {
                assert_eq!(parse(Some(value)), (Mode::Summary, true));
            }
        }

        #[test]
        fn cuda_logging_admission_does_not_count_uncompleted_products() {
            let mut state = State::default();
            assert!(state.first_admission());
            assert!(!state.first_admission());
            assert_eq!(state.totals, Totals::default());
            assert_eq!(state.last_summary, None);
            let completed = state
                .completed(Duration::from_secs(80), false, &[work(7, 11)])
                .unwrap();
            assert_eq!(completed.completed_exact_products, 1);
            assert_eq!(completed.devices[0].successful_moduli, 11);
        }

        #[test]
        fn cuda_logging_summary_cadence_keeps_cumulative_actual_device_work() {
            let mut state = State::default();
            let first = state
                .completed(Duration::ZERO, false, &[work(4, 3), work(1, 2)])
                .unwrap();
            assert_eq!(first.completed_exact_products, 1);
            for second in 0..30 {
                assert!(state
                    .completed(Duration::from_secs(second), true, &[work(1, 7)])
                    .is_none());
            }
            let next = state
                .completed(Duration::from_secs(30), false, &[work(4, 5)])
                .unwrap();
            assert_eq!(next.completed_exact_products, 32);
            assert_eq!(next.synchronous_products, 2);
            assert_eq!(next.pipelined_products, 30);
            assert_eq!(next.devices[0].ordinal, 4);
            assert_eq!(next.devices[0].completed_products, 2);
            assert_eq!(next.devices[0].successful_moduli, 8);
            assert_eq!(next.devices[1].ordinal, 1);
            assert_eq!(next.devices[1].completed_products, 31);
            assert_eq!(next.devices[1].successful_moduli, 212);
            assert!(!next.saturated);
            // A backwards synthetic clock must not create an emission burst.
            assert!(state.completed(Duration::ZERO, false, &[]).is_none());
        }

        #[test]
        fn cuda_logging_device_storage_is_bounded_with_explicit_overflow() {
            let mut state = State::default();
            for ordinal in 100..200 {
                state.completed(Duration::ZERO, false, &[work(ordinal, 2)]);
            }
            assert_eq!(state.totals.devices.len(), MAX_DEVICES);
            assert_eq!(state.totals.overflow_device_assignments, 36);
            assert_eq!(state.totals.overflow_successful_moduli, 72);
            // Reuse a retained ordinal after the map fills, then an overflow one.
            state.completed(Duration::ZERO, true, &[work(100, 3), work(199, 5)]);
            assert_eq!(state.totals.devices.len(), MAX_DEVICES);
            assert_eq!(state.totals.devices[0].successful_moduli, 5);
            assert_eq!(state.totals.overflow_device_assignments, 37);
            assert_eq!(state.totals.overflow_successful_moduli, 77);
        }

        #[test]
        fn cuda_logging_counters_saturate_without_wrapping_or_failing_products() {
            let mut state = State::default();
            state.totals.completed_exact_products = u64::MAX;
            state.totals.synchronous_products = u64::MAX;
            state.totals.devices.push(DeviceTotals {
                ordinal: 0,
                completed_products: u64::MAX,
                successful_moduli: u64::MAX,
            });
            let totals = state
                .completed(Duration::ZERO, false, &[work(0, 1)])
                .unwrap();
            assert!(totals.saturated);
            assert_eq!(totals.completed_exact_products, u64::MAX);
            assert_eq!(totals.devices[0].completed_products, u64::MAX);
            assert_eq!(totals.devices[0].successful_moduli, u64::MAX);
        }

        #[test]
        fn cuda_logging_concurrent_completions_share_one_cadence_and_totals() {
            use std::sync::{Arc, Mutex};
            let state = Arc::new(Mutex::new(State::default()));
            let emissions = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..4)
                    .map(|ordinal| {
                        let state = Arc::clone(&state);
                        scope.spawn(move || {
                            let mut emissions = 0;
                            for _ in 0..100 {
                                emissions += usize::from(
                                    state
                                        .lock()
                                        .unwrap()
                                        .completed(
                                            Duration::ZERO,
                                            ordinal % 2 == 0,
                                            &[work(ordinal, 3)],
                                        )
                                        .is_some(),
                                );
                            }
                            emissions
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .sum::<usize>()
            });
            assert_eq!(emissions, 1);
            let state = state.lock().unwrap();
            assert_eq!(state.totals.completed_exact_products, 400);
            assert_eq!(state.totals.synchronous_products, 200);
            assert_eq!(state.totals.pipelined_products, 200);
            assert_eq!(state.totals.devices.len(), 4);
            assert!(state.totals.devices.iter().all(|device| {
                device.completed_products == 100 && device.successful_moduli == 300
            }));
        }
    }
}

#[derive(Debug)]
pub struct ExactModularProduct {
    pub entries: Vec<BigInt>,
    pub rows: usize,
    pub columns: usize,
    pub moduli: usize,
    pub reconstruction_bits: u64,
    pub required_bound_bits: u64,
    pub estimated_workspace_bytes: u64,
    pub timings: ProductTimings,
    pub backend_used: ProductBackend,
    pub cuda_devices: Vec<usize>,
    pub cuda_device_work: Vec<CudaDeviceWork>,
}

#[derive(Debug)]
enum Tree {
    Small {
        index: usize,
        len: usize,
        primes: [u32; 4],
        coefficients: [u64; 4],
        small_product: u64,
        product: BigUint,
    },
    Branch {
        left: Box<Tree>,
        right: Box<Tree>,
        product: BigUint,
        inverse: BigUint,
    },
}

impl Tree {
    fn product(&self) -> &BigUint {
        match self {
            Self::Small { product, .. } | Self::Branch { product, .. } => product,
        }
    }

    fn build(primes: &[u32], offset: usize) -> Self {
        assert!(!primes.is_empty());
        if primes.len() <= 4 {
            // Every admitted prime is at most 16191, so the four-prime
            // product is below 2^56 and every CRT coefficient is smaller
            // than that product. Build these leaves without bigint GCDs.
            let small_product: u64 = primes.iter().map(|&p| u64::from(p)).product();
            let mut small_primes = [0; 4];
            let mut coefficients = [0; 4];
            for (i, &p) in primes.iter().enumerate() {
                debug_assert!(p <= 16_191);
                small_primes[i] = p;
                let quotient = small_product / u64::from(p);
                let egcd = ((quotient % u64::from(p)) as i64).extended_gcd(&i64::from(p));
                debug_assert_eq!(egcd.gcd, 1);
                let inverse = egcd.x.rem_euclid(i64::from(p)) as u64;
                coefficients[i] = quotient * inverse;
            }
            return Self::Small {
                index: offset,
                len: primes.len(),
                primes: small_primes,
                coefficients,
                small_product,
                product: small_product.into(),
            };
        }
        // Balance complete four-prime groups, retaining only one possible
        // partial group instead of splitting many groups into two-prime leaves.
        let middle = (primes.len().div_ceil(4) / 2) * 4;
        let left = Box::new(Self::build(&primes[..middle], offset));
        let right = Box::new(Self::build(&primes[middle..], offset + middle));
        let l = BigInt::from(left.product().clone());
        let r = BigInt::from(right.product().clone());
        let egcd = l.extended_gcd(&r);
        debug_assert_eq!(egcd.gcd, BigInt::one());
        let inverse = egcd
            .x
            .mod_floor(&r)
            .to_biguint()
            .expect("nonnegative inverse");
        let product = left.product() * right.product();
        Self::Branch {
            left,
            right,
            product,
            inverse,
        }
    }

    fn residues(&self, value: &BigUint, negative: bool, output: &mut [u16]) {
        match self {
            Self::Small {
                index,
                len,
                primes,
                small_product,
                ..
            } => {
                // A parent already reduced to this group's product. The root
                // may be an unreduced input, so reduce it once when necessary.
                let value = match value.to_u64() {
                    Some(value) if value < *small_product => value,
                    Some(value) => value % *small_product,
                    None => (value % *small_product).to_u64().expect("small remainder"),
                };
                for (i, &prime) in primes[..*len].iter().enumerate() {
                    let r = (value % u64::from(prime)) as u32;
                    output[*index + i] = if negative && r != 0 { prime - r } else { r } as u16;
                }
            }
            Self::Branch { left, right, .. } => {
                let a = value % left.product();
                let b = value % right.product();
                left.residues(&a, negative, output);
                right.residues(&b, negative, output);
            }
        }
    }

    fn reconstruct(&self, residues: &[Vec<u16>], entry: usize) -> BigUint {
        match self {
            Self::Small {
                index,
                len,
                coefficients,
                small_product,
                ..
            } => {
                // Canonical residues are below 2^14 and coefficients below
                // 2^56: four weighted terms sum to less than 2^72. Even a
                // malformed u16 residue cannot overflow this u128 sum.
                let sum: u128 = coefficients[..*len]
                    .iter()
                    .enumerate()
                    .map(|(i, &coefficient)| {
                        u128::from(residues[*index + i][entry]) * u128::from(coefficient)
                    })
                    .sum();
                ((sum % u128::from(*small_product)) as u64).into()
            }
            Self::Branch {
                left,
                right,
                inverse,
                ..
            } => {
                let a = left.reconstruct(residues, entry);
                let b = right.reconstruct(residues, entry);
                let a_mod = &a % right.product();
                let difference = if b >= a_mod {
                    b - a_mod
                } else {
                    right.product() - (a_mod - b)
                };
                let correction = (difference * inverse) % right.product();
                a + left.product() * correction
            }
        }
    }
}

fn prime(p: u32) -> bool {
    let mut d = 3;
    while d * d <= p {
        if p.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    p >= 3 && p % 2 == 1
}

fn choose_primes(bound: &BigUint) -> Result<(Vec<u32>, BigUint)> {
    let mut product = BigUint::one();
    let mut primes = Vec::new();
    for p in (3..=16_191).rev().filter(|&p| prime(p)) {
        if !primes.is_empty() && product > *bound {
            break;
        }
        product *= p;
        primes.push(p);
    }
    if product <= *bound {
        return Err(ReconstructionCapacity(
            "input exceeds the exact small-prime reconstruction capacity",
        )
        .into());
    }
    Ok((primes, product))
}

/// Conservative supported reconstruction width. A bound strictly below
/// `2^maximum_reconstruction_bits()` fits the fixed exact residue family.
/// Callers can select a direct arbitrary-precision CPU product before conversion
/// when wider integer inputs are required.
pub fn maximum_reconstruction_bits() -> u64 {
    reconstruction_capacity().0
}

fn reconstruction_capacity() -> (u64, u64) {
    static CAPACITY: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *CAPACITY.get_or_init(|| {
        let mut product = BigUint::one();
        let mut count = 0;
        for p in (3..=16_191).filter(|&p| prime(p)) {
            product *= p;
            count += 1;
        }
        (product.bits() - 1, count)
    })
}

/// Exact signed-magnitude limb products for the HP adapter's wider domain.
/// Each child has its own proved CRT bound. Only their complete integer sum
/// crosses back into the HP adapter; no partial sum is rounded or returned.
/// The public single-pass product/report keeps its finite-modulus semantics.
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "hp", test))]
pub(crate) fn multiply_wide_entries(
    left: &[BigInt],
    right: &[BigInt],
    rows: usize,
    inner: usize,
    columns: usize,
    backend: ProductBackend,
    limits: ProductLimits,
) -> Result<Vec<BigInt>> {
    multiply_wide_entries_with(left, right, rows, inner, columns, limits, |a, b, limits| {
        Ok(multiply(a, b, rows, inner, columns, backend, limits)?.entries)
    })
}

#[cfg(any(feature = "hp", test))]
fn multiply_wide_entries_with(
    left: &[BigInt],
    right: &[BigInt],
    rows: usize,
    inner: usize,
    columns: usize,
    limits: ProductLimits,
    mut product: impl FnMut(&[BigInt], &[BigInt], ProductLimits) -> Result<Vec<BigInt>>,
) -> Result<Vec<BigInt>> {
    ensure!(
        rows > 0 && inner > 0 && columns > 0,
        "matrix dimensions must be positive"
    );
    ensure!(limits.workers > 0, "workers must be positive");
    ensure!(
        rows.checked_mul(inner) == Some(left.len())
            && inner.checked_mul(columns) == Some(right.len()),
        "matrix storage does not match shape"
    );
    let count = rows
        .checked_mul(columns)
        .context("product shape overflow")?;
    ensure!(
        rows <= i32::MAX as usize && columns <= i32::MAX as usize,
        "matrix exceeds kernel index domain"
    );
    let padded_k = inner.checked_add(3).context("inner dimension overflow")? & !3;
    ensure!(
        (padded_k as u128) * 127 * 127 <= i32::MAX as u128,
        "integer accumulation domain exceeded"
    );
    ensure!(
        (padded_k as u128) * (4096 + 128 * (16129 + 4096 + 3969) + 16384 * 3969)
            <= i64::MAX as u128,
        "integer recombination domain exceeded"
    );
    let a_bits = left.iter().map(BigInt::bits).max().unwrap_or(0);
    let b_bits = right.iter().map(BigInt::bits).max().unwrap_or(0);
    ensure!(
        a_bits <= limits.max_input_bits && b_bits <= limits.max_input_bits,
        "input precision budget exceeded"
    );
    let inner_bits = u64::from(usize::BITS - inner.leading_zeros());
    let (capacity, maximum_moduli) = reconstruction_capacity();
    // |digit| < 2^width, so twice each child dot product's absolute bound is
    // strictly below 2^(2*width + bit_length(inner) + 1), below the family.
    let width = capacity
        .checked_sub(inner_bits + 2)
        .context("inner dimension exceeds limb domain")?
        / 2;
    ensure!(width > 0, "empty reconstruction limb domain");
    let a_limbs = a_bits.max(1).div_ceil(width);
    let b_limbs = b_bits.max(1).div_ceil(width);
    let operations = (rows as u128)
        .checked_mul(inner as u128)
        .and_then(|v| v.checked_mul(columns as u128))
        .and_then(|v| v.checked_mul(u128::from(a_limbs)))
        .and_then(|v| v.checked_mul(u128::from(b_limbs)))
        .and_then(|v| v.checked_mul(u128::from(maximum_moduli)))
        .context("limb operation count overflow")?;
    ensure!(
        operations <= u128::from(limits.max_modular_scalar_products),
        "modular limb operation budget exceeded"
    );
    let output_bits = a_bits
        .checked_add(b_bits)
        .and_then(|v| v.checked_add(inner_bits))
        .context("limb output width overflow")?;
    let output_bytes = u128::from(output_bits.div_ceil(8)) + 64;
    let limb_bytes = u128::from(width.div_ceil(8)) + 64;
    let input_bytes = u128::from(a_bits.max(b_bits).div_ceil(8)) + 64;
    // Retain the full accumulator plus one shifted child entry, both limb
    // operands and extraction scratch. Shifting before masking can temporarily
    // allocate a full-width input. Child outputs/CRT/worker stacks are charged
    // separately by multiply, against the remainder of this same budget.
    let retained = (count as u128) * output_bytes * 4
        + ((left.len() as u128) + (right.len() as u128)) * limb_bytes * 4
        + 8 * (input_bytes + output_bytes + limb_bytes);
    admit_workspace(
        retained,
        u128::from(limits.max_workspace_bytes),
        "wide product retained",
    )?;
    let child_limits = ProductLimits {
        max_workspace_bytes: limits.max_workspace_bytes - retained as u64,
        max_input_bits: width,
        ..limits
    };
    static FIRST_WIDE_PRODUCT: std::sync::Once = std::sync::Once::new();
    FIRST_WIDE_PRODUCT.call_once(|| {
        xc_core::progress_message!(
            "[HP] exact wide product: {}x{}x{}; integer widths {} and {}; {}x{} signed-magnitude limbs, exact integer recombination before rounding",
            rows, inner, columns, a_bits, b_bits, a_limbs, b_limbs
        );
    });
    let width = usize::try_from(width).context("limb width exceeds address domain")?;
    let mask = (BigUint::one() << width) - 1u32;
    let limb = |values: &[BigInt], index: u64| -> Result<Vec<BigInt>> {
        let shift = usize::try_from(index)
            .ok()
            .and_then(|i| i.checked_mul(width))
            .context("limb shift overflow")?;
        Ok(values
            .iter()
            .map(|value| BigInt::from_biguint(value.sign(), (value.magnitude() >> shift) & &mask))
            .collect())
    };
    let mut output = vec![BigInt::from(0); count];
    for a_index in 0..a_limbs {
        let a = limb(left, a_index)?;
        for b_index in 0..b_limbs {
            let b = limb(right, b_index)?;
            let entries = product(&a, &b, child_limits)?;
            ensure!(entries.len() == count, "limb backend output shape mismatch");
            let shift = a_index
                .checked_add(b_index)
                .and_then(|i| usize::try_from(i).ok())
                .and_then(|i| i.checked_mul(width))
                .context("product limb shift overflow")?;
            for (sum, entry) in output.iter_mut().zip(entries) {
                *sum += entry << shift;
            }
        }
    }
    Ok(output)
}

fn centered(value: u16, prime: u32) -> i64 {
    let r = i64::from(value);
    if u32::from(value) > prime / 2 {
        r - i64::from(prime)
    } else {
        r
    }
}

fn cpu_residue_product(a: &[u16], b: &[u16], m: usize, k: usize, n: usize, p: u32) -> Vec<u16> {
    let aa: Vec<i64> = a.iter().map(|&v| centered(v, p)).collect();
    let bb: Vec<i64> = b.iter().map(|&v| centered(v, p)).collect();
    (0..m * n)
        .into_par_iter()
        .map(|entry| {
            let (i, j) = (entry / n, entry % n);
            let mut dot = 0i64;
            for t in 0..k {
                dot += aa[i * k + t] * bb[t * n + j];
            }
            dot.rem_euclid(i64::from(p)) as u16
        })
        .collect()
}

fn converted(values: &[BigInt], tree: &Tree, primes: usize) -> Vec<Vec<u16>> {
    values
        .par_iter()
        .map(|value| {
            let mut residues = vec![0; primes];
            tree.residues(
                value.magnitude(),
                value.sign() == Sign::Minus,
                &mut residues,
            );
            residues
        })
        .collect()
}

// Reuse the caller's pool when it already has the requested concurrency. A
// separate product pool would otherwise double allocated workers in a large
// run. Different explicit limits still receive a bounded dedicated pool.
struct ProductPool(Option<rayon::ThreadPool>);
impl ProductPool {
    fn new(workers: usize) -> Result<Self> {
        if workers == rayon::current_num_threads() {
            Ok(Self(None))
        } else {
            Ok(Self(Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .stack_size(2 << 20)
                    .build()?,
            )))
        }
    }
    fn install<T: Send>(&self, work: impl FnOnce() -> T + Send) -> T {
        match &self.0 {
            Some(pool) => pool.install(work),
            None => work(),
        }
    }
}

#[cfg(feature = "cuda-product")]
fn scheduled_residues<E>(
    devices: &[usize],
    moduli: usize,
    initialize: impl Fn(usize) -> Result<E> + Sync,
    product: impl Fn(&mut E, usize) -> Result<Vec<u16>> + Sync,
    finish: impl Fn(E) -> Result<()> + Sync,
) -> Result<(Vec<Vec<u16>>, Vec<CudaDeviceWork>)> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    ensure!(!devices.is_empty(), "CUDA scheduler has no admitted device");
    ensure!(
        devices.len() <= moduli,
        "CUDA scheduler has more devices than residues"
    );
    let next = AtomicUsize::new(devices.len());
    // Keep the original scalar dispatch/allocation path for the default engine.
    let batches: Vec<Result<CudaDeviceBatch>> = devices
        .par_iter()
        .enumerate()
        .map(|(first, &ordinal)| {
            let mut engine = initialize(ordinal)?;
            let mut batch = vec![(first, product(&mut engine, first))];
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= moduli {
                    break;
                }
                batch.push((index, product(&mut engine, index)));
            }
            Ok((ordinal, batch, finish(engine)))
        })
        .collect();
    collect_cuda_residues(batches, moduli)
}

#[cfg(feature = "cuda-product")]
type IndexedCudaResidue = (usize, Result<Vec<u16>>);
#[cfg(feature = "cuda-product")]
type CudaDeviceBatch = (usize, Vec<IndexedCudaResidue>, Result<()>);

#[cfg(feature = "cuda-product")]
fn scheduled_residue_batches<E>(
    devices: &[usize],
    moduli: usize,
    batch_size: usize,
    initialize: impl Fn(usize) -> Result<E> + Sync,
    products: impl Fn(&mut E, &[usize]) -> Vec<Result<Vec<u16>>> + Sync,
    finish: impl Fn(E) -> Result<()> + Sync,
) -> Result<(Vec<Vec<u16>>, Vec<CudaDeviceWork>)> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    ensure!(!devices.is_empty(), "CUDA scheduler has no admitted device");
    ensure!(
        (1..=2).contains(&batch_size),
        "CUDA scheduler batch size must be 1 or 2"
    );
    ensure!(
        devices.len() <= moduli,
        "CUDA scheduler has more devices than residues"
    );
    let next = AtomicUsize::new(devices.len());
    // Faster devices consume more tasks. Engines stay on their owning worker;
    // completed values and errors retain their immutable prime indices.
    let batches: Vec<Result<CudaDeviceBatch>> = devices
        .par_iter()
        .enumerate()
        .map(|(first, &ordinal)| {
            let mut engine = initialize(ordinal)?;
            // Reserve one distinct prime for every admitted GPU. A slow context
            // initialization cannot leave a reported device with zero work.
            let mut batch = Vec::new();
            let mut indices = vec![first];
            loop {
                // A bounded atomic claim keeps the dynamic multi-device
                // schedule; initial reserved primes cannot be stolen by a
                // faster device. Each engine/stream remains on its owner.
                let take = batch_size - indices.len();
                if take > 0 {
                    let start = next.fetch_add(take, Ordering::Relaxed);
                    indices.extend(start..start.saturating_add(take).min(moduli));
                }
                if indices.is_empty() {
                    break;
                }
                let values = products(&mut engine, &indices);
                if values.len() != indices.len() {
                    // Malformed batch callbacks cannot omit work or bypass
                    // cleanup. Associate the failure with every claimed prime.
                    let returned = values.len();
                    let expected = indices.len();
                    batch.extend(indices.drain(..).map(|index| {
                        (
                            index,
                            Err(anyhow::anyhow!(
                                "CUDA batch returned {returned} results for {expected} residues"
                            )),
                        )
                    }));
                } else {
                    batch.extend(indices.drain(..).zip(values));
                }
            }
            // Cleanup runs even when a product failed. Numerical errors keep
            // prime order; cleanup errors are checked afterward in device order.
            Ok((ordinal, batch, finish(engine)))
        })
        .collect();
    collect_cuda_residues(batches, moduli)
}

#[cfg(feature = "cuda-product")]
fn collect_cuda_residues(
    batches: Vec<Result<CudaDeviceBatch>>,
    moduli: usize,
) -> Result<(Vec<Vec<u16>>, Vec<CudaDeviceWork>)> {
    let mut indexed = Vec::with_capacity(moduli);
    let mut work = Vec::with_capacity(batches.len());
    let mut cleanup = Vec::with_capacity(batches.len());
    // Indexed collection preserves configured device order for init errors.
    for batch in batches {
        let (ordinal, batch, finished) = batch?;
        work.push(CudaDeviceWork {
            ordinal,
            successful_moduli: batch.iter().filter(|(_, value)| value.is_ok()).count(),
        });
        indexed.extend(batch);
        cleanup.push(finished);
    }
    indexed.sort_unstable_by_key(|(index, _)| *index);
    ensure!(indexed.len() == moduli, "CUDA scheduler omitted a residue");
    let residues = indexed
        .into_iter()
        .enumerate()
        .map(|(expected, (index, value))| {
            ensure!(index == expected, "CUDA scheduler duplicated a residue");
            value
        })
        .collect::<Result<Vec<_>>>()?;
    for finished in cleanup {
        finished?;
    }
    Ok((residues, work))
}

/// Exact row-major rectangular product. The reconstruction modulus is strictly
/// larger than twice `inner * max(abs(left)) * max(abs(right))`.
///
/// CUDA uses three signed-byte products per prime. The checked accumulation
/// bound also bounds every partial sum, irrespective of reduction order.
pub fn multiply(
    left: &[BigInt],
    right: &[BigInt],
    rows: usize,
    inner: usize,
    columns: usize,
    backend: ProductBackend,
    limits: ProductLimits,
) -> Result<ExactModularProduct> {
    let start = Instant::now();
    ensure!(
        rows > 0 && inner > 0 && columns > 0,
        "matrix dimensions must be positive"
    );
    ensure!(limits.workers > 0, "workers must be positive");
    ensure!(
        rows.checked_mul(inner) == Some(left.len())
            && inner.checked_mul(columns) == Some(right.len()),
        "matrix storage does not match shape"
    );
    let count = rows
        .checked_mul(columns)
        .context("product shape overflow")?;
    ensure!(
        rows <= i32::MAX as usize && columns <= i32::MAX as usize,
        "matrix exceeds kernel index domain"
    );
    let padded_k = inner.checked_add(3).context("inner dimension overflow")? & !3;
    ensure!(
        (padded_k as u128) * 127 * 127 <= i32::MAX as u128,
        "integer accumulation domain exceeded"
    );
    // This includes the absolute bound of every intermediate in the three-GEMM
    // base128 recombination, not just the final residue dot product.
    ensure!(
        (padded_k as u128) * (4096 + 128 * (16129 + 4096 + 3969) + 16384 * 3969)
            <= i64::MAX as u128,
        "integer recombination domain exceeded"
    );
    let max_a = left
        .iter()
        .map(BigInt::magnitude)
        .max()
        .expect("positive shape");
    let max_b = right
        .iter()
        .map(BigInt::magnitude)
        .max()
        .expect("positive shape");
    ensure!(
        max_a.bits() <= limits.max_input_bits && max_b.bits() <= limits.max_input_bits,
        "input precision budget exceeded"
    );
    let total_entries = (left.len() as u128) + (right.len() as u128) + (count as u128);
    // Refuse impossible minimum storage before allocating the large bound.
    admit_workspace(
        total_entries * 32,
        u128::from(limits.max_workspace_bytes),
        "product",
    )?;
    admit_workspace(
        u128::from(limits.max_input_bits).div_ceil(8),
        u128::from(limits.max_workspace_bytes),
        "product input width",
    )?;
    // The finite modulus set cannot represent arbitrary widths. Preflight this
    // and conservative bound-construction scratch before multiplying magnitudes;
    // otherwise a rejected request could allocate more than its workspace cap.
    let bound_bits = u128::from(max_a.bits())
        + u128::from(max_b.bits())
        + u128::from(usize::BITS - inner.leading_zeros())
        + 1;
    if bound_bits > 32_768 {
        return Err(ReconstructionCapacity("input exceeds the small-prime planning domain").into());
    }
    admit_workspace(
        bound_bits.div_ceil(8) * 64 + total_entries * 64,
        u128::from(limits.max_workspace_bytes),
        "product planning",
    )?;
    let bound = max_a * max_b * BigUint::from(inner) * 2u32;
    let (primes, modulus) = choose_primes(&bound)?;
    let operations = (rows as u128) * (inner as u128) * (columns as u128) * (primes.len() as u128);
    ensure!(
        operations <= u128::from(limits.max_modular_scalar_products),
        "modular operation budget exceeded"
    );
    let bit_storage = u128::from(modulus.bits().div_ceil(8));
    // Both conversion layouts, all residue answers, output integers, tree and
    // per-worker recursive bigint temporaries; excludes caller-owned inputs and
    // vendor CUDA context/library allocations, which require host admission.
    #[allow(unused_mut)] // CUDA adds its admitted per-device buffers.
    let mut workspace = total_entries * (4 * (primes.len() as u128) + 64)
        + (count as u128) * (bit_storage + 64) * 2
        + (limits.workers as u128) * (bit_storage * 128 + (primes.len() as u128) * 128)
        + bit_storage * (primes.len() as u128) * 32
        + (limits.workers as u128) * (2 << 20);
    admit_workspace(workspace, u128::from(limits.max_workspace_bytes), "product")?;
    #[allow(unused_mut)]
    let mut backend_used = ProductBackend::Cpu;
    #[allow(unused_mut)]
    let mut cuda_devices = Vec::new();
    #[allow(unused_mut)]
    let mut cuda_device_work = Vec::new();
    #[cfg(feature = "cuda-product")]
    let mut cuda_execution = cuda::ExecutionMode::Synchronous;
    #[cfg(feature = "cuda-product")]
    {
        let threshold = match std::env::var("XC_CUDA_MIN_SCALAR_PRODUCTS") {
            Ok(text) => text
                .parse::<u128>()
                .context("invalid XC_CUDA_MIN_SCALAR_PRODUCTS")?,
            Err(_) => 0,
        };
        if backend == ProductBackend::Cuda
            || (backend == ProductBackend::Auto && operations >= threshold)
        {
            let (requested, adaptive) = cuda::ExecutionMode::for_product(primes.len())?;
            cuda_execution = if adaptive {
                cuda::ExecutionMode::Synchronous
            } else {
                requested
            };
            let status = cuda_status()?;
            if !adaptive && requested == cuda::ExecutionMode::Pipelined {
                ensure!(
                    cuda::pipeline_symbols_available()?,
                    "requested CUDA pipeline is unavailable (missing asynchronous/batched symbols)"
                );
            }
            let (device_bytes, mut per_device_bytes) =
                cuda::workspace_for_mode(rows, inner, columns, cuda_execution)?;
            let mut selected = cuda::choose_devices(
                &status,
                device_bytes,
                per_device_bytes,
                limits.max_workspace_bytes - workspace as u64,
                limits.workers,
                primes.len(),
                backend == ProductBackend::Cuda,
            )?;
            // The automatic preference is optional only during planning.
            // Admit the established route first; do not lose an admitted
            // device, allocate extra storage, or retry after execution starts.
            if adaptive {
                let preference = cuda::default_pipeline_devices(
                    primes.len(),
                    &selected,
                    cuda::pipeline_symbols_available,
                    || {
                        let (device_bytes, aggregate_bytes) = cuda::workspace_for_mode(
                            rows,
                            inner,
                            columns,
                            cuda::ExecutionMode::Pipelined,
                        )?;
                        let devices = cuda::choose_devices(
                            &status,
                            device_bytes,
                            aggregate_bytes,
                            limits.max_workspace_bytes - workspace as u64,
                            limits.workers,
                            primes.len(),
                            backend == ProductBackend::Cuda,
                        )?;
                        Ok((devices, aggregate_bytes))
                    },
                )?;
                if let Some((devices, bytes)) = preference {
                    selected = devices;
                    per_device_bytes = bytes;
                    cuda_execution = cuda::ExecutionMode::Pipelined;
                }
            }
            ensure!(
                backend != ProductBackend::Cuda || !selected.is_empty(),
                "CUDA product unavailable or exceeds workspace/headroom: {}",
                status
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("no admitted device")
            );
            if !selected.is_empty() {
                backend_used = ProductBackend::Cuda;
                cuda_devices = selected.iter().map(|d| d.ordinal).collect();
                workspace += u128::from(per_device_bytes) * cuda_devices.len() as u128;
                cuda_logging::admission(|| {
                    format!(
                    "[CUDA] exact integer product {}x{}x{}; admitted devices {:?}; runtime {:?}, driver {:?}, cuBLAS {:?}; execution {:?}; admitted workspace {} bytes",
                    rows,
                    inner,
                    columns,
                    cuda_devices,
                    status.runtime_version,
                    status.driver_version,
                    status.cublas_version,
                    cuda_execution,
                    workspace
                )
                });
            }
        }
    }
    #[cfg(not(feature = "cuda-product"))]
    let _ = backend;
    let tree = Tree::build(&primes, 0);
    let pool = ProductPool::new(limits.workers)?;
    let mut timings = ProductTimings {
        planning_seconds: start.elapsed().as_secs_f64(),
        ..Default::default()
    };
    let start = Instant::now();
    let (a, b) = pool.install(|| {
        (
            converted(left, &tree, primes.len()),
            converted(right, &tree, primes.len()),
        )
    });
    timings.conversion_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let residues = match backend_used {
        #[cfg(feature = "cuda-product")]
        ProductBackend::Cuda => {
            let (residues, work) = pool.install(|| {
                let initialize = |ordinal| {
                    cuda::Engine::self_test(ordinal, cuda_execution)?;
                    cuda::Engine::new(ordinal, rows, inner, columns, cuda_execution)
                };
                if cuda_execution == cuda::ExecutionMode::Synchronous {
                    return scheduled_residues(
                        &cuda_devices,
                        primes.len(),
                        initialize,
                        |engine, index| {
                            let aa: Vec<u16> = a.iter().map(|v| v[index]).collect();
                            let bb: Vec<u16> = b.iter().map(|v| v[index]).collect();
                            engine.product(&aa, &bb, primes[index])
                        },
                        cuda::Engine::finish,
                    );
                }
                scheduled_residue_batches(
                    &cuda_devices,
                    primes.len(),
                    cuda_execution.batch_size(),
                    initialize,
                    |engine, indices| {
                        let operands: Vec<_> = indices
                            .iter()
                            .map(|&index| {
                                let aa: Vec<u16> = a.iter().map(|v| v[index]).collect();
                                let bb: Vec<u16> = b.iter().map(|v| v[index]).collect();
                                (aa, bb, primes[index])
                            })
                            .collect();
                        let inputs: Vec<_> = operands
                            .iter()
                            .map(|(aa, bb, prime)| (aa.as_slice(), bb.as_slice(), *prime))
                            .collect();
                        engine.products(&inputs)
                    },
                    cuda::Engine::finish,
                )
            })?;
            cuda_device_work = work;
            if cuda_device_work.len() > 1 && cuda_logging::verbose() {
                xc_core::emit_message(format!(
                    "[CUDA] completed residue products: {:?}",
                    cuda_device_work
                ));
            }
            residues
        }
        _ => {
            let mut residues = Vec::with_capacity(primes.len());
            for (index, &p) in primes.iter().enumerate() {
                let aa: Vec<u16> = a.iter().map(|v| v[index]).collect();
                let bb: Vec<u16> = b.iter().map(|v| v[index]).collect();
                residues
                    .push(pool.install(|| cpu_residue_product(&aa, &bb, rows, inner, columns, p)));
            }
            residues
        }
    };
    timings.products_seconds = start.elapsed().as_secs_f64();
    drop(a);
    drop(b);
    let start = Instant::now();
    let half = &modulus >> 1;
    let entries: Vec<BigInt> = pool.install(|| {
        (0..count)
            .into_par_iter()
            .map(|entry| {
                let value = tree.reconstruct(&residues, entry);
                if value > half {
                    BigInt::from(value) - BigInt::from(modulus.clone())
                } else {
                    BigInt::from(value)
                }
            })
            .collect()
    });
    // An additional range check catches malformed backend output in some cases;
    // it is not a substitute for the exact-kernel and CRT correctness contract.
    let absolute_limit = &bound >> 1;
    if entries.iter().any(|v| v.magnitude() > &absolute_limit) {
        bail!("reconstructed product violates its proved magnitude bound");
    }
    timings.reconstruction_seconds = start.elapsed().as_secs_f64();
    #[cfg(feature = "cuda-product")]
    if backend_used == ProductBackend::Cuda {
        // A residue result alone does not establish a completed exact product.
        cuda_logging::completed(
            cuda_execution == cuda::ExecutionMode::Pipelined,
            &cuda_device_work,
        );
    }
    if matches!(std::env::var("XC_CUDA_TIMINGS").as_deref(), Ok("1")) {
        xc_core::emit_message(format!(
            "[CUDA timings] exact integer product {}x{}x{}; backend={:?}; selected_gpus={}; moduli={}; required_bound_bits={}; reconstruction_bits={}; planning_seconds={:.6}; conversion_seconds={:.6}; products_seconds={:.6}; reconstruction_seconds={:.6}",
            rows,
            inner,
            columns,
            backend_used,
            cuda_devices.len(),
            primes.len(),
            bound.bits(),
            modulus.bits(),
            timings.planning_seconds,
            timings.conversion_seconds,
            timings.products_seconds,
            timings.reconstruction_seconds,
        ));
    }
    Ok(ExactModularProduct {
        entries,
        rows,
        columns,
        moduli: primes.len(),
        reconstruction_bits: modulus.bits(),
        required_bound_bits: bound.bits(),
        estimated_workspace_bytes: workspace as u64,
        timings,
        backend_used,
        cuda_devices,
        cuda_device_work,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    #[test]
    fn product_pool_reuses_caller_workers_and_honors_smaller_limits() {
        use std::collections::HashSet;
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        outer.install(|| {
            let caller_ids: HashSet<_> = rayon::broadcast(|_| std::thread::current().id())
                .into_iter()
                .collect();
            let shared = ProductPool::new(4).unwrap();
            assert!(shared.0.is_none());
            let shared_ids: HashSet<_> = shared.install(|| {
                rayon::broadcast(|_| std::thread::current().id())
                    .into_iter()
                    .collect()
            });
            assert_eq!(caller_ids, shared_ids);
            let limited = ProductPool::new(2).unwrap();
            assert!(limited.0.is_some());
            let limited_ids: HashSet<_> = limited.install(|| {
                rayon::broadcast(|_| std::thread::current().id())
                    .into_iter()
                    .collect()
            });
            assert_eq!(limited_ids.len(), 2);
            assert!(caller_ids.is_disjoint(&limited_ids));
            let a: Vec<_> = (0..35).map(|i| BigInt::from(i - 17)).collect();
            let b: Vec<_> = (0..21).map(|i| BigInt::from(10 - i)).collect();
            let mut bound = limits();
            bound.workers = 4;
            let same = multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, bound).unwrap();
            bound.workers = 2;
            let fewer = multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, bound).unwrap();
            assert_eq!(same.entries, fewer.entries);
        });
    }

    // Independent scalar CRT oracle: progressively extend a signed BigInt
    // result by one prime, without the tree or its native group coefficients.
    fn scalar_crt(primes: &[u32], residues: &[Vec<u16>], entry: usize) -> BigUint {
        let mut value = BigInt::zero();
        let mut product = BigInt::one();
        for (&prime, residues) in primes.iter().zip(residues) {
            let prime = BigInt::from(prime);
            let delta = (BigInt::from(residues[entry]) - &value).mod_floor(&prime);
            let inverse = product.extended_gcd(&prime).x.mod_floor(&prime);
            value += &product * ((delta * inverse).mod_floor(&prime));
            product *= prime;
        }
        value.to_biguint().unwrap()
    }

    #[test]
    fn grouped_crt_matches_independent_scalar_oracle_across_boundaries() {
        let primes: Vec<_> = (3..=16_191).rev().filter(|&p| prime(p)).collect();
        for count in [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            256,
            257,
            500,
            1025,
            primes.len(),
        ] {
            let primes = &primes[..count];
            let tree = Tree::build(primes, 0);
            let residues: Vec<_> = primes
                .iter()
                .enumerate()
                .map(|(index, &p)| {
                    vec![
                        0,
                        (p - 1) as u16,
                        (((index as u64 + 17) * 1_234_567) % u64::from(p)) as u16,
                        (((index as u64 + 43) * 8_765_431) % u64::from(p)) as u16,
                    ]
                })
                .collect();
            assert_eq!(tree.reconstruct(&residues, 0), BigUint::zero());
            assert_eq!(tree.reconstruct(&residues, 1), tree.product() - 1u32);
            for entry in 2..4 {
                assert_eq!(
                    tree.reconstruct(&residues, entry),
                    scalar_crt(primes, &residues, entry),
                    "prime count {count}, entry {entry}"
                );
            }
        }
    }

    #[test]
    fn grouped_conversion_matches_direct_signed_moduli_and_centering() {
        let primes: Vec<_> = (3..=16_191).rev().filter(|&p| prime(p)).collect();
        let wide: BigInt = (BigInt::one() << 6708usize) + (BigInt::one() << 3386usize) + 12345u32;
        for count in [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257, 500, 1025,
        ] {
            let primes = &primes[..count];
            // A nonzero offset also checks that groups retain original prime
            // indices and do not overwrite neighboring output storage.
            let tree = Tree::build(primes, 3);
            let modulus = BigInt::from(tree.product().clone());
            let half = &modulus >> 1usize;
            let values = [
                BigInt::zero(),
                BigInt::one(),
                -BigInt::one(),
                &half - 1u32,
                half.clone(),
                &half + 1u32,
                -&half,
                -(&half + 1u32),
                &modulus - 1u32,
                modulus.clone(),
                &modulus + 1u32,
                -(&modulus + 1u32),
                wide.clone(),
                -&wide,
            ];
            for value in values {
                let mut actual = vec![u16::MAX; count + 6];
                tree.residues(value.magnitude(), value.sign() == Sign::Minus, &mut actual);
                assert_eq!(actual[..3], [u16::MAX; 3]);
                assert_eq!(actual[count + 3..], [u16::MAX; 3]);
                for (index, &prime) in primes.iter().enumerate() {
                    assert_eq!(
                        actual[index + 3],
                        value.mod_floor(&BigInt::from(prime)).to_u16().unwrap(),
                        "prime count {count}, prime index {index}, value {value}"
                    );
                }
                let residues: Vec<_> = actual.into_iter().map(|r| vec![r]).collect();
                let reconstructed = BigInt::from(tree.reconstruct(&residues, 0));
                assert_eq!(reconstructed, value.mod_floor(&modulus));
                let centered = if reconstructed > half {
                    reconstructed - &modulus
                } else {
                    reconstructed
                };
                assert_eq!(centered, (&value + &half).mod_floor(&modulus) - &half);
                if value >= -&half && value <= half {
                    assert_eq!(centered, value);
                }
            }
        }
    }

    fn limits() -> ProductLimits {
        ProductLimits {
            max_workspace_bytes: 64 << 20,
            max_input_bits: 16_000,
            max_modular_scalar_products: 10_000_000,
            workers: 2,
        }
    }

    #[test]
    fn wide_limb_budgets_refuse_before_dispatch_and_late_error_is_atomic() {
        let h = BigInt::one() << 16_000usize;
        let a = vec![&h + 3, &h - 5];
        let b = vec![&h - 7, -(&h + 1u32)];
        let mut budget = limits();
        budget.max_workspace_bytes = 512 << 20;
        budget.max_input_bits = 32_000;
        for bad in [
            ProductLimits {
                max_modular_scalar_products: 1,
                ..budget
            },
            ProductLimits {
                max_workspace_bytes: 1,
                ..budget
            },
            ProductLimits {
                max_input_bits: 1,
                ..budget
            },
            ProductLimits {
                workers: 0,
                ..budget
            },
        ] {
            let result = multiply_wide_entries_with(&a, &b, 1, 2, 1, bad, |_, _, _| {
                panic!("invalid request dispatched a limb")
            });
            assert!(result.is_err());
        }
        let originals = (a.clone(), b.clone());
        let mut calls = 0;
        let error = multiply_wide_entries_with(&a, &b, 1, 2, 1, budget, |aa, bb, child| {
            calls += 1;
            if calls == 2 {
                bail!("injected second-child device failure");
            }
            Ok(multiply(aa, bb, 1, 2, 1, ProductBackend::Cpu, child)?.entries)
        })
        .unwrap_err();
        assert_eq!(calls, 2);
        assert!(error.to_string().contains("second-child device failure"));
        assert_eq!((a, b), originals);
    }

    #[test]
    fn wide_limb_boundaries_signs_and_zero_outputs_match_direct_integers() {
        let width = (maximum_reconstruction_bits() - 2 - 2) / 2;
        let base = BigInt::one() << width as usize;
        let square = &base * &base;
        let a = vec![
            square.clone(),
            -&square,
            BigInt::from(0),
            &base + 1,
            -&base,
            BigInt::from(-1),
        ];
        let b = vec![
            BigInt::from(1),
            -&base,
            BigInt::from(1),
            base.clone(),
            BigInt::from(1),
            BigInt::from(0),
        ];
        let mut budget = limits();
        budget.max_workspace_bytes = 512 << 20;
        budget.max_input_bits = 50_000;
        let actual = multiply_wide_entries(&a, &b, 2, 3, 2, ProductBackend::Cpu, budget).unwrap();
        let expected: Vec<BigInt> = (0..4)
            .map(|i| (0..3).map(|k| &a[i / 2 * 3 + k] * &b[k * 2 + i % 2]).sum())
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(actual[0], BigInt::from(0));
        assert_eq!(actual[2], BigInt::from(0));
    }
    fn cancellation(backend: ProductBackend, bits: usize) {
        cancellation_with_limits(backend, bits, limits());
    }
    fn cancellation_with_limits(backend: ProductBackend, bits: usize, limits: ProductLimits) {
        let t: BigInt = (BigInt::one() << bits) + 12345u32;
        let a = vec![t.clone(), &t - 1, BigInt::from(-7), BigInt::from(11)];
        let b = vec![t.clone(), BigInt::from(-3), -(&t + 1u32), BigInt::from(5)];
        let actual = multiply(&a, &b, 2, 2, 2, backend, limits).unwrap();
        let expected: Vec<BigInt> = (0..4)
            .map(|e| {
                (0..2)
                    .map(|k| &a[(e / 2) * 2 + k] * &b[k * 2 + e % 2])
                    .sum()
            })
            .collect();
        assert_eq!(actual.entries, expected);
        assert_eq!(actual.entries[0], BigInt::one());
        assert!(actual.reconstruction_bits > actual.required_bound_bits);
    }
    #[test]
    fn modular_product_exact_cancellation_thousands_of_bits() {
        cancellation(ProductBackend::Cpu, 3386);
        cancellation(ProductBackend::Cpu, 6708);
    }
    #[test]
    fn modular_product_rectangular_signed_zero_and_workers() {
        let a: Vec<BigInt> = (0..35).map(|i| BigInt::from(i - 17)).collect();
        let b: Vec<BigInt> = (0..21).map(|i| BigInt::from(10 - i)).collect();
        let expected: Vec<BigInt> = (0..15)
            .map(|e| {
                (0..7)
                    .map(|k| &a[(e / 3) * 7 + k] * &b[k * 3 + e % 3])
                    .sum()
            })
            .collect();
        let mut l = limits();
        l.workers = 1;
        assert_eq!(
            multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, l)
                .unwrap()
                .entries,
            expected
        );
        assert_eq!(
            multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, limits())
                .unwrap()
                .entries,
            expected
        );
        let zeros = vec![BigInt::zero(); 35];
        assert!(multiply(&zeros, &b, 5, 7, 3, ProductBackend::Cpu, limits())
            .unwrap()
            .entries
            .iter()
            .all(Zero::is_zero));
    }
    #[test]
    fn automatic_small_products_preserve_cpu_entries() {
        let a: Vec<BigInt> = (0..35).map(|i| BigInt::from(i - 17)).collect();
        let b: Vec<BigInt> = (0..21).map(|i| BigInt::from(10 - i)).collect();
        let cpu = multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, limits()).unwrap();
        let automatic = multiply(&a, &b, 5, 7, 3, ProductBackend::Auto, limits()).unwrap();
        assert_eq!(cpu.entries, automatic.entries);
    }
    #[cfg(feature = "cuda-product")]
    #[test]
    fn bounded_residue_batches_preserve_owner_order_tail_work_and_error_priority() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Owner {
            thread: std::thread::ThreadId,
            _not_send: std::rc::Rc<()>,
        }
        for workers in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let devices: Vec<_> = (0..workers).rev().collect();
            for moduli in [workers, workers + 1, 97] {
                let visited: Vec<_> = (0..moduli).map(|_| AtomicUsize::new(0)).collect();
                let finalized = AtomicUsize::new(0);
                let (actual, work) = pool
                    .install(|| {
                        scheduled_residue_batches(
                            &devices,
                            moduli,
                            2,
                            |_| {
                                Ok(Owner {
                                    thread: std::thread::current().id(),
                                    _not_send: std::rc::Rc::new(()),
                                })
                            },
                            |owner, indices| {
                                assert!(!indices.is_empty() && indices.len() <= 2);
                                assert_eq!(owner.thread, std::thread::current().id());
                                indices
                                    .iter()
                                    .map(|&index| {
                                        visited[index].fetch_add(1, Ordering::Relaxed);
                                        let _ = rayon::join(|| index, || index + 1);
                                        assert_eq!(owner.thread, std::thread::current().id());
                                        Ok(vec![index as u16])
                                    })
                                    .collect()
                            },
                            |owner| {
                                assert_eq!(owner.thread, std::thread::current().id());
                                finalized.fetch_add(1, Ordering::Relaxed);
                                Ok(())
                            },
                        )
                    })
                    .unwrap();
                assert_eq!(
                    actual,
                    (0..moduli).map(|i| vec![i as u16]).collect::<Vec<_>>()
                );
                assert_eq!(
                    work.iter().map(|work| work.ordinal).collect::<Vec<_>>(),
                    devices
                );
                assert!(work.iter().all(|work| work.successful_moduli > 0));
                assert!(visited
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1));
                assert_eq!(finalized.load(Ordering::Relaxed), workers);
            }
            let finalized = AtomicUsize::new(0);
            let error = pool
                .install(|| {
                    scheduled_residue_batches(
                        &devices,
                        97,
                        2,
                        Ok,
                        |_, indices| {
                            indices
                                .iter()
                                .map(|&index| {
                                    if index == 3 {
                                        std::thread::yield_now();
                                    }
                                    ensure!(index != 3 && index != 31, "prime {index} failed");
                                    Ok(vec![index as u16])
                                })
                                .collect()
                        },
                        |ordinal| {
                            finalized.fetch_add(1, Ordering::Relaxed);
                            bail!("device {ordinal} cleanup failed")
                        },
                    )
                })
                .unwrap_err();
            assert_eq!(error.to_string(), "prime 3 failed");
            assert_eq!(finalized.load(Ordering::Relaxed), workers);
        }
    }

    #[cfg(feature = "cuda-product")]
    #[test]
    fn bounded_residue_batches_reject_malformed_results_and_configuration_with_cleanup() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for returned in [0, 1, 3] {
            let finalized = AtomicUsize::new(0);
            let error = scheduled_residue_batches(
                &[0],
                2,
                2,
                Ok,
                |_, _| (0..returned).map(|_| Ok(vec![0])).collect(),
                |_| {
                    finalized.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                },
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("CUDA batch returned {returned} results for 2 residues")
            );
            assert_eq!(finalized.load(Ordering::Relaxed), 1);
        }
        for (devices, moduli, batch_size) in [
            (&[][..], 1, 2),
            (&[0][..], 0, 2),
            (&[0][..], 1, 0),
            (&[0][..], 1, 3),
        ] {
            assert!(scheduled_residue_batches(
                devices,
                moduli,
                batch_size,
                |_| -> Result<()> { panic!("invalid scheduling must fail before initialization") },
                |_, _| unreachable!(),
                |_| unreachable!(),
            )
            .is_err());
        }
    }

    #[cfg(feature = "cuda-product")]
    #[test]
    fn residue_scheduler_preserves_prime_order_and_first_error_across_workers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let expected: Vec<_> = (0..97).map(|i| vec![i as u16, (17 * i) as u16]).collect();
        for workers in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let devices: Vec<_> = (0..workers).rev().collect();
            let visited: Vec<_> = (0..97).map(|_| AtomicUsize::new(0)).collect();
            let actual = pool
                .install(|| {
                    scheduled_residues(
                        &devices,
                        97,
                        Ok,
                        |_, index| {
                            visited[index].fetch_add(1, Ordering::Relaxed);
                            if index % 7 == 0 {
                                std::thread::yield_now();
                            }
                            Ok(vec![index as u16, (17 * index) as u16])
                        },
                        |_| Ok(()),
                    )
                })
                .unwrap();
            assert_eq!(actual.0, expected);
            assert_eq!(
                actual.1.iter().map(|w| w.successful_moduli).sum::<usize>(),
                97
            );
            assert!(actual.1.iter().all(|w| w.successful_moduli > 0));
            assert!(visited
                .iter()
                .all(|count| count.load(Ordering::Relaxed) == 1));

            let error = pool
                .install(|| {
                    scheduled_residues(
                        &devices,
                        97,
                        Ok,
                        |_, index| {
                            if index == 7 {
                                std::thread::sleep(std::time::Duration::from_millis(2));
                            }
                            ensure!(index != 7 && index != 31, "prime {index} failed");
                            Ok(vec![index as u16])
                        },
                        |_| Ok(()),
                    )
                })
                .unwrap_err();
            assert_eq!(error.to_string(), "prime 7 failed");

            let error = pool
                .install(|| {
                    scheduled_residues(
                        &devices,
                        97,
                        |ordinal| -> Result<usize> { bail!("device {ordinal} failed") },
                        |_, _| unreachable!(),
                        |_| Ok(()),
                    )
                })
                .unwrap_err();
            assert_eq!(error.to_string(), format!("device {} failed", workers - 1));
        }
    }
    #[cfg(feature = "cuda-product")]
    #[test]
    fn residue_scheduler_uses_late_devices_and_finishes_on_the_owner_thread() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Owner {
            ordinal: usize,
            thread: std::thread::ThreadId,
            _not_send: std::rc::Rc<()>,
        }
        for workers in [2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let devices: Vec<_> = (0..workers).rev().collect();
            for moduli in [workers, 97] {
                let finalized: Vec<_> = (0..workers).map(|_| AtomicUsize::new(0)).collect();
                let (values, work) = pool
                    .install(|| {
                        scheduled_residues(
                            &devices,
                            moduli,
                            |ordinal| {
                                if ordinal == 0 {
                                    std::thread::sleep(std::time::Duration::from_millis(5));
                                }
                                Ok(Owner {
                                    ordinal,
                                    thread: std::thread::current().id(),
                                    _not_send: std::rc::Rc::new(()),
                                })
                            },
                            |owner, index| {
                                assert_eq!(owner.thread, std::thread::current().id());
                                // Nested host work may steal another device task, but
                                // the suspended owner never migrates to its worker.
                                let _ = rayon::join(|| index + 1, || index + 2);
                                assert_eq!(owner.thread, std::thread::current().id());
                                Ok(vec![index as u16])
                            },
                            |owner| {
                                assert_eq!(owner.thread, std::thread::current().id());
                                finalized[owner.ordinal].fetch_add(1, Ordering::Relaxed);
                                Ok(())
                            },
                        )
                    })
                    .unwrap();
                assert_eq!(
                    values,
                    (0..moduli).map(|i| vec![i as u16]).collect::<Vec<_>>()
                );
                assert_eq!(work.iter().map(|w| w.ordinal).collect::<Vec<_>>(), devices);
                assert!(work.iter().all(|w| w.successful_moduli > 0));
                assert_eq!(
                    work.iter().map(|w| w.successful_moduli).sum::<usize>(),
                    moduli
                );
                assert!(finalized.iter().all(|v| v.load(Ordering::Relaxed) == 1));
            }
        }
    }
    #[cfg(feature = "cuda-product")]
    #[test]
    fn residue_scheduler_propagates_cleanup_and_preserves_product_error_priority() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for product_fails in [false, true] {
            let finished = AtomicUsize::new(0);
            let error = pool
                .install(|| {
                    scheduled_residues(
                        &[1, 0],
                        17,
                        Ok,
                        |_, index| {
                            ensure!(!product_fails || index != 3, "prime 3 failed");
                            Ok(vec![index as u16])
                        },
                        |ordinal| {
                            finished.fetch_add(1, Ordering::Relaxed);
                            bail!("device {ordinal} cleanup failed")
                        },
                    )
                })
                .unwrap_err();
            assert_eq!(finished.load(Ordering::Relaxed), 2);
            assert_eq!(
                error.to_string(),
                if product_fails {
                    "prime 3 failed"
                } else {
                    "device 1 cleanup failed"
                }
            );
        }
    }
    #[test]
    fn input_width_workspace_refusal_is_distinct_from_precision_domain() {
        let input = vec![BigInt::one()];
        let mut budget = limits();
        budget.max_workspace_bytes = 128;
        budget.max_input_bits = 1025;
        // Minimum shape storage fits, but the configured input-width scratch
        // needs ceil(1025/8)=129 bytes. This is allocation admission, not a
        // violation of the actual input-precision contract.
        let error = multiply(&input, &input, 1, 1, 1, ProductBackend::Cpu, budget).unwrap_err();
        let refusal = error.downcast_ref::<ModularWorkspaceLimit>().unwrap();
        assert_eq!((refusal.required, refusal.available), (129, 128));
        budget = limits();
        budget.max_input_bits = 0;
        let error = multiply(&input, &input, 1, 1, 1, ProductBackend::Cpu, budget).unwrap_err();
        assert!(!error.is::<ModularWorkspaceLimit>());
        assert!(error
            .to_string()
            .contains("input precision budget exceeded"));
    }

    #[test]
    fn modular_product_refuses_shape_and_budgets() {
        let a = vec![BigInt::one(); 4];
        let mut l = limits();
        assert!(multiply(&a, &a, 3, 2, 2, ProductBackend::Cpu, l).is_err());
        l.max_workspace_bytes = 1;
        assert!(multiply(&a, &a, 2, 2, 2, ProductBackend::Cpu, l).is_err());
        l = limits();
        l.max_input_bits = 0;
        assert!(multiply(&a, &a, 2, 2, 2, ProductBackend::Cpu, l).is_err());
        l = limits();
        l.max_modular_scalar_products = 0;
        assert!(multiply(&a, &a, 2, 2, 2, ProductBackend::Cpu, l).is_err());
        l = limits();
        l.workers = 0;
        assert!(multiply(&a, &a, 2, 2, 2, ProductBackend::Cpu, l).is_err());
        // Refuse widths outside the finite prime domain before constructing a
        // large magnitude bound or residue arrays.
        let wide = vec![BigInt::one() << 17_000usize];
        l = limits();
        l.max_input_bits = 20_000;
        assert!(multiply(&wide, &wide, 1, 1, 1, ProductBackend::Cpu, l)
            .unwrap_err()
            .to_string()
            .contains("planning domain"));
        // Worker stacks count toward the admitted workspace even for tiny
        // numerical inputs.
        l = limits();
        l.max_workspace_bytes = 2 << 20;
        assert!(multiply(&a, &a, 2, 2, 2, ProductBackend::Cpu, l).is_err());
        l = limits();
        assert!(multiply(&[], &[], usize::MAX, 2, 2, ProductBackend::Cpu, l).is_err());
    }
    #[cfg(feature = "cuda-product")]
    #[test]
    #[ignore = "requires XC_CUDA_PRODUCT_PIPELINE=1 and explicitly selected idle CUDA devices"]
    fn modular_product_cuda_pipeline_preserves_wide_crt_and_multi_device_order() {
        assert_eq!(
            cuda::ExecutionMode::from_env().unwrap(),
            cuda::ExecutionMode::Pipelined,
            "set XC_CUDA_PRODUCT_PIPELINE=1 explicitly"
        );
        let status = cuda_status().unwrap();
        assert!(
            status.available,
            "{}",
            status.unavailable_reason.unwrap_or_default()
        );
        let mut budget = limits();
        budget.max_workspace_bytes = 256 << 20;
        cancellation_with_limits(ProductBackend::Cuda, 3386, budget);
        cancellation_with_limits(ProductBackend::Cuda, 6708, budget);
        let scale = BigInt::one() << 256usize;
        let a: Vec<_> = (0..35).map(|i| BigInt::from(i - 17) * &scale).collect();
        let b: Vec<_> = (0..21).map(|i| BigInt::from(10 - i) * &scale).collect();
        let gpu = multiply(&a, &b, 5, 7, 3, ProductBackend::Cuda, budget).unwrap();
        let cpu = multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, budget).unwrap();
        assert_eq!(gpu.entries, cpu.entries);
        let expected: Vec<_> = status
            .devices
            .iter()
            .take(budget.workers.min(gpu.moduli))
            .map(|device| device.ordinal)
            .collect();
        assert_eq!(gpu.cuda_devices, expected);
        assert_eq!(
            gpu.cuda_device_work
                .iter()
                .map(|work| work.ordinal)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(gpu
            .cuda_device_work
            .iter()
            .all(|work| work.successful_moduli > 0));
        assert_eq!(
            gpu.cuda_device_work
                .iter()
                .map(|work| work.successful_moduli)
                .sum::<usize>(),
            gpu.moduli
        );
    }

    #[cfg(feature = "cuda-product")]
    #[test]
    #[ignore = "requires an explicitly selected local CUDA device"]
    fn modular_product_cuda_exact_cancellation_and_rectangular() {
        let status = cuda_status().unwrap();
        println!("CUDA runtime discovery: {status:?}");
        assert!(
            status.available,
            "{}",
            status.unavailable_reason.unwrap_or_default()
        );
        cancellation(ProductBackend::Cuda, 3386);
        cancellation(ProductBackend::Cuda, 6708);
        let a: Vec<BigInt> = (0..35).map(|i| BigInt::from(i - 17)).collect();
        let b: Vec<BigInt> = (0..21).map(|i| BigInt::from(10 - i)).collect();
        assert_eq!(
            multiply(&a, &b, 5, 7, 3, ProductBackend::Cuda, limits())
                .unwrap()
                .entries,
            multiply(&a, &b, 5, 7, 3, ProductBackend::Cpu, limits())
                .unwrap()
                .entries
        );
    }
}
