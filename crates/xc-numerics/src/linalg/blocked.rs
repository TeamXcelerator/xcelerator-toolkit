// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Fixed-panel partial-pivot LU. Execution tiling changes no inner sums.
use super::{lu_factor, LuFactors};
use crate::exact_product::{
    prepare_borrowed, ExactProductLimits, PreparedProduct, ProductShape, WorkspaceLimit,
};
use crate::modular_product::{ModularWorkspaceLimit, ProductBackend, ProductLimits};
use anyhow::{anyhow, ensure, Result};
use rayon::prelude::*;
use rug::{float::Round, Float};

const PANEL: usize = 512;
const MIN_DIMENSION: usize = 2;
const ARITHMETIC: &str = "dense_lu_partial_pivot_dimension_panel512_exact_schur_v1";

/// Numerical identity depends on dimension, never device availability.
pub fn dense_lu_arithmetic(dimension: usize) -> Option<&'static str> {
    (dimension >= MIN_DIMENSION).then_some(ARITHMETIC)
}

/// Execution controls. Panel width and arithmetic are deliberately fixed.
#[derive(Clone, Copy, Debug)]
pub struct DenseLuOptions {
    pub backend: ProductBackend,
    /// Temporary product workspace, excluding the input and returned factors.
    pub max_workspace_bytes: u64,
    /// Maximum output tile edge. Smaller tiles have identical arithmetic.
    pub tile_edge: usize,
    /// Admission cap for the current CPU pool and size of the modular host pool.
    pub workers: usize,
}

/// Nonmathematical failure of an accelerated factorization. Callers must not
/// cache it as a solver refusal or reinterpret it as a singular matrix.
#[derive(Debug)]
pub struct DenseLuExecutionFailure;
impl std::fmt::Display for DenseLuExecutionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("accelerated dense LU execution failed")
    }
}
impl std::error::Error for DenseLuExecutionFailure {}

#[derive(Debug)]
pub(super) struct SingularPivot;
impl std::fmt::Display for SingularPivot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dense LU encountered a zero computed pivot")
    }
}
impl std::error::Error for SingularPivot {}

fn configured_number(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(value) => {
            let n = value
                .parse::<u64>()
                .map_err(|_| anyhow!("{name} must be a positive integer"))?;
            ensure!(n > 0, "{name} must be positive");
            Ok(n)
        }
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(_) => Err(anyhow!("{name} must be valid text")),
    }
}

fn default_workspace_bytes(
    dimension: usize,
    precision: u32,
    workers: usize,
    available: u64,
) -> u64 {
    let dimension = u64::try_from(dimension).unwrap_or(u64::MAX);
    let workers = u64::try_from(workers).unwrap_or(u64::MAX);
    let matrix = dimension
        .saturating_mul(dimension)
        .saturating_mul(u64::from(precision) * 8 + 4096);
    // Nominal-precision estimate, not a bound on arbitrary exponent spreads:
    // every actual product is admitted separately and may split its output tile.
    // The floor includes small CUDA buffers/self-tests on one or two devices.
    // Four MiB per worker leaves two MiB of stack allowance per modular worker
    // inside the half-budget product boundary, plus panel/caller headroom.
    matrix
        .saturating_add(64 << 20)
        .max(128 << 20)
        .saturating_add(workers.saturating_mul(4 << 20))
        .min(4 << 30)
        .min(available)
}

fn with_workspace_growth<T>(
    initial: u64,
    ceiling: u64,
    mut factor: impl FnMut(u64) -> Result<T>,
) -> Result<T> {
    let mut budget = initial;
    loop {
        match factor(budget) {
            Ok(factors) => return Ok(factors),
            Err(error) => {
                if (!error.is::<WorkspaceLimit>() && !error.is::<ModularWorkspaceLimit>())
                    || budget >= ceiling
                {
                    return Err(error);
                }
                // A nominal-precision default can underestimate irreducible
                // CRT storage when shifted inputs have a wide exponent span.
                // Output tiling has already been exhausted. The failed call
                // dropped every factor, product and reservation before this
                // retry; only allocation admission failures can reach here.
                // The automatic 128 MiB floor and 4 GiB ceiling allow at most
                // five growth steps, so successful earlier panels cannot be
                // repeated without a fixed bound.
                let next = budget.saturating_mul(2).max(budget + 1).min(ceiling);
                xc_core::progress_message!(
                    "[HP] dense LU: increasing automatic workspace from {budget} to {next} bytes after allocation admission refusal: {error}"
                );
                budget = next;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct OutputTile {
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
}

fn split_workspace_tile(
    tile: OutputTile,
    error: &anyhow::Error,
) -> Option<(OutputTile, OutputTile)> {
    if !error.is::<WorkspaceLimit>() && !error.is::<ModularWorkspaceLimit>() {
        return None;
    }
    if tile.rows >= tile.cols && tile.rows > 1 {
        let first = tile.rows / 2;
        Some((
            OutputTile {
                rows: first,
                ..tile
            },
            OutputTile {
                row: tile.row + first,
                rows: tile.rows - first,
                ..tile
            },
        ))
    } else if tile.cols > 1 {
        let first = tile.cols / 2;
        Some((
            OutputTile {
                cols: first,
                ..tile
            },
            OutputTile {
                col: tile.col + first,
                cols: tile.cols - first,
                ..tile
            },
        ))
    } else {
        None
    }
}

fn prepare_schur_tile(
    lu: &[Float],
    dim: usize,
    first: usize,
    end: usize,
    tile: OutputTile,
    precision: u32,
    product_budget: u64,
) -> Result<PreparedProduct> {
    let left =
        (tile.row..tile.row + tile.rows).flat_map(|i| lu[i * dim + first..i * dim + end].iter());
    let right =
        (first..end).flat_map(|i| lu[i * dim + tile.col..i * dim + tile.col + tile.cols].iter());
    prepare_borrowed(
        left,
        right,
        ProductShape {
            rows: tile.rows,
            inner: end - first,
            cols: tile.cols,
        },
        precision,
        ExactProductLimits {
            max_workspace_bytes: u128::from(product_budget),
            max_scalar_products: u128::MAX,
            max_exponent_span: 1_000_000,
            max_integer_bits: 2_100_000,
        },
    )?
    .with_exact_output(u128::from(product_budget))
}

#[allow(clippy::too_many_arguments)] // Explicit tile bounds and execution policy; no owned input copies.
fn update_schur_tile(
    lu: &mut [Float],
    dim: usize,
    first: usize,
    end: usize,
    tile: OutputTile,
    precision: u32,
    options: DenseLuOptions,
    cuda: bool,
) -> Result<()> {
    // Half the reservation leaves room for panel temporaries and allocator
    // headroom. Inputs are borrowed, so a refused tile has no cloned Floats.
    let product_budget = options.max_workspace_bytes / 2;
    // Keep the prepared integer operands and bridge inside this attempt. They
    // are dropped before retrying children, including a later CRT admission
    // refusal after exact preparation succeeded. No output is mutated until a
    // complete product has succeeded.
    let result = (|| {
        let prepared = prepare_schur_tile(lu, dim, first, end, tile, precision, product_budget)?;
        if cuda
            && (options.backend == ProductBackend::Cuda
                || u64::from(prepared.plan().output_integer_bits) + 2
                    <= crate::modular_product::maximum_reconstruction_bits())
        {
            prepared.modular_product(
                Round::Nearest,
                options.backend,
                ProductLimits {
                    max_workspace_bytes: product_budget,
                    max_input_bits: u64::from(
                        prepared
                            .plan()
                            .left_integer_bits
                            .max(prepared.plan().right_integer_bits),
                    )
                    .max(1),
                    max_modular_scalar_products: u64::MAX,
                    workers: options.workers,
                },
            )
        } else {
            prepared.cpu_product_parallel(Round::Nearest)
        }
    })();
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            if let Some((left, right)) = split_workspace_tile(tile, &error) {
                // Only output coordinates are split. Both children retain the
                // complete dimension-bound panel inner sum, and their updates
                // never touch either panel input. Earlier completed tiles need
                // no rollback or recomputation.
                update_schur_tile(lu, dim, first, end, left, precision, options, cuda)?;
                return update_schur_tile(lu, dim, first, end, right, precision, options, cuda);
            }
            return Err(error);
        }
    };
    for i in 0..tile.rows {
        for j in 0..tile.cols {
            lu[(tile.row + i) * dim + tile.col + j] -= &result[i * tile.cols + j];
        }
    }
    Ok(())
}

/// Default primary factorization: legacy for scalar matrices, exact blocked
/// Schur updates for every larger matrix, automatically using available CUDA devices.
/// CPU fallback uses exactly the same blocked arithmetic. Selected CUDA errors
/// propagate; no partial result is accepted or substituted with legacy factors.
/// The initial default temporary workspace scales with dimension, precision, and
/// workers. If even a scalar output tile cannot fit, allocation-admission errors
/// may grow that default and restart the factorization, capped by the original
/// available run memory and 4 GiB. Every attempt reserves its full allowance.
/// `XC_LU_WORKSPACE_BYTES` is a hard override and never grows automatically.
pub fn lu_factor_auto(a: &[Float], dim: usize) -> Result<LuFactors> {
    if dense_lu_arithmetic(dim).is_none() {
        return lu_factor(a, dim);
    }
    let result: Result<LuFactors> = (|| {
        let backend = match std::env::var("XC_DENSE_BACKEND").as_deref() {
            Ok("auto") | Err(std::env::VarError::NotPresent) => ProductBackend::Auto,
            Ok("cpu") => ProductBackend::Cpu,
            Ok("cuda") => ProductBackend::Cuda,
            _ => return Err(anyhow!("XC_DENSE_BACKEND must be auto, cpu, or cuda")),
        };
        let resources = xc_core::RunResources::current();
        let available = resources
            .as_ref()
            .map(|r| {
                let usage = r.usage();
                r.maximum_bytes()
                    .saturating_sub(usage.retained_bytes)
                    .saturating_sub(usage.working_bytes)
            })
            .unwrap_or(1 << 30);
        let workers = rayon::current_num_threads()
            .min(resources.as_ref().map_or(usize::MAX, |r| r.worker_limit()));
        let explicit_workspace = std::env::var_os("XC_LU_WORKSPACE_BYTES").is_some();
        let initial = configured_number(
            "XC_LU_WORKSPACE_BYTES",
            default_workspace_bytes(dim, a.first().map_or(0, Float::prec), workers, available),
        )?;
        let ceiling = if explicit_workspace {
            initial
        } else {
            available.min(4 << 30)
        };
        let tile_edge = usize::try_from(configured_number("XC_LU_TILE_EDGE", 1024)?)?;
        with_workspace_growth(initial, ceiling, |max_workspace_bytes| {
            lu_factor_blocked(
                a,
                dim,
                DenseLuOptions {
                    backend,
                    max_workspace_bytes,
                    tile_edge,
                    workers,
                },
            )
        })
    })();
    result.map_err(|error| {
        if error.is::<SingularPivot>() {
            error
        } else {
            error.context(DenseLuExecutionFailure)
        }
    })
}

/// Dimension-bound panels min(512, max(1, dimension / 4)), deterministic partial pivoting,
/// nearest-rounded panel arithmetic, and one rounded C - exact(L * U) update.
/// This is an approximate factorization, not an inertia or eigenvalue proof.
pub fn lu_factor_blocked(a: &[Float], dim: usize, options: DenseLuOptions) -> Result<LuFactors> {
    crate::mpfr_interval::ensure_uniform_exponent_range()?;
    ensure!(
        dim > 0 && dim.checked_mul(dim) == Some(a.len()),
        "invalid blocked LU shape"
    );
    let p = a[0].prec();
    ensure!(
        a.iter().all(|v| v.is_finite() && v.prec() == p),
        "blocked LU requires finite uniform precision"
    );
    ensure!(
        options.max_workspace_bytes > 0 && options.tile_edge > 0 && options.workers > 0,
        "invalid blocked LU execution budget"
    );
    let resources = xc_core::RunResources::current();
    ensure!(
        rayon::current_num_threads() <= options.workers,
        "blocked LU current CPU pool exceeds its worker allowance"
    );
    if let Some(r) = &resources {
        r.check_workers(rayon::current_num_threads())?;
        r.check_workers(options.workers)?;
    }
    let _reservation = resources
        .as_ref()
        .map(|r| r.try_reserve_working(options.max_workspace_bytes))
        .transpose()?;
    let cuda = match options.backend {
        ProductBackend::Cpu => false,
        ProductBackend::Auto => {
            let status = crate::modular_product::cuda_status()?;
            if !status.available {
                xc_core::progress_message!(
                    "[HP] dense LU: automatic CPU fallback because CUDA is unavailable: {}",
                    status
                        .unavailable_reason
                        .as_deref()
                        .unwrap_or("no runtime availability reason was reported")
                );
            }
            status.available
        }
        ProductBackend::Cuda => {
            let status = crate::modular_product::cuda_status()?;
            ensure!(
                status.available,
                "CUDA dense LU requested but unavailable: {:?}",
                status.unavailable_reason
            );
            true
        }
    };
    xc_core::progress_message!("[HP] dense LU: exact dimension-bound Schur updates, backend={}, dimension={dim}, workspace={} bytes", if cuda {"cuda"} else {"cpu"}, options.max_workspace_bytes);
    let mut lu = a.to_vec();
    let mut perm: Vec<_> = (0..dim).collect();
    let panel = PANEL.min((dim / 4).max(1));
    for first in (0..dim).step_by(panel) {
        let end = (first + panel).min(dim);
        for k in first..end {
            let mut pivot_row = k;
            for i in k + 1..dim {
                if lu[i * dim + k].clone().abs() > lu[pivot_row * dim + k].clone().abs() {
                    pivot_row = i;
                }
            }
            if pivot_row != k {
                for j in 0..dim {
                    lu.swap(k * dim + j, pivot_row * dim + j);
                }
                perm.swap(k, pivot_row);
            }
            let pivot = lu[k * dim + k].clone();
            ensure!(pivot.is_finite(), "blocked LU nonfinite pivot");
            if pivot.is_zero() {
                return Err(SingularPivot.into());
            }
            let row = lu[k * dim + k + 1..k * dim + end].to_vec();
            lu[(k + 1) * dim..].par_chunks_mut(dim).for_each(|target| {
                target[k] /= &pivot;
                let factor = target[k].clone();
                for j in k + 1..end {
                    let product = Float::with_val(p, &factor * &row[j - k - 1]);
                    target[j] -= product;
                }
            });
        }
        if end == dim {
            break;
        }
        // Unit lower-triangular solve for U12 after all tall-panel row swaps.
        for k in first..end {
            let (previous, trailing) = lu.split_at_mut(k * dim);
            let (current, _) = trailing.split_at_mut(dim);
            let lower = current[first..k].to_vec();
            current[end..]
                .par_iter_mut()
                .enumerate()
                .for_each(|(offset, value)| {
                    let j = end + offset;
                    for s in first..k {
                        let product =
                            Float::with_val(p, &lower[s - first] * &previous[s * dim + j]);
                        *value -= product;
                    }
                });
        }
        // Use a conservative tile allowance for conversion, complete residues,
        // reconstruction and exact Float products; re-admit actual exponent spans
        // in prepare(). Output tiling does not split the exact panel inner sum.
        let per_entry = u64::from(p).saturating_mul(8).saturating_add(4096);
        let host_stacks = (options.workers as u64).saturating_mul(2 << 20);
        ensure!(
            options.max_workspace_bytes > host_stacks,
            "blocked LU workspace cannot admit worker stacks"
        );
        let entries = (options.max_workspace_bytes - host_stacks) / per_entry;
        let mut tile = options.tile_edge.min(dim - end);
        while tile > 1 && (tile as u128) * (tile as u128) > entries as u128 {
            tile /= 2;
        }
        ensure!(
            entries > 0,
            "blocked LU workspace cannot admit an output tile"
        );
        for row0 in (end..dim).step_by(tile) {
            let rows = tile.min(dim - row0);
            for col0 in (end..dim).step_by(tile) {
                let cols = tile.min(dim - col0);
                update_schur_tile(
                    &mut lu,
                    dim,
                    first,
                    end,
                    OutputTile {
                        row: row0,
                        col: col0,
                        rows,
                        cols,
                    },
                    p,
                    options,
                    cuda,
                )?;
            }
        }
        ensure!(
            lu.iter().all(Float::is_finite),
            "blocked LU arithmetic exceeds finite range"
        );
    }
    ensure!(
        lu.iter().all(Float::is_finite),
        "blocked LU arithmetic exceeds finite range"
    );
    Ok(LuFactors { lu, perm })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;
    fn options(tile: usize) -> DenseLuOptions {
        DenseLuOptions {
            backend: ProductBackend::Cpu,
            max_workspace_bytes: 128 << 20,
            tile_edge: tile,
            workers: rayon::current_num_threads(),
        }
    }
    fn matrix(n: usize, p: u32) -> Vec<Float> {
        (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                Float::with_val(
                    p,
                    if i == j {
                        (n * 3) as i64
                    } else {
                        ((i * 19 + j * 7) % 17) as i64 - 8
                    },
                )
            })
            .collect()
    }
    #[test]
    fn blocked_lu_pivots_reconstruct_and_solve_with_tiling_invariance() {
        let (n, p) = (71, 192);
        let mut a = matrix(n, p);
        for j in 0..n {
            a.swap(j, 70 * n + j);
        }
        let one = lu_factor_blocked(&a, n, options(1)).unwrap();
        let many = lu_factor_blocked(&a, n, options(17)).unwrap();
        assert_eq!(one.perm, many.perm);
        assert_eq!(one.lu, many.lu);
        let mut max = Rational::new();
        for i in 0..n {
            for j in 0..n {
                let mut reconstructed = Rational::new();
                for k in 0..=i.min(j) {
                    let l = if k == i {
                        Rational::from(1)
                    } else {
                        one.lu[i * n + k].to_rational().unwrap()
                    };
                    reconstructed += l * one.lu[k * n + j].to_rational().unwrap();
                }
                let error = (a[one.perm[i] * n + j].to_rational().unwrap() - reconstructed).abs();
                max = max.max(error);
            }
        }
        assert!(max < (Rational::from(1) >> 170));
        let b: Vec<_> = a
            .chunks(n)
            .map(|row| Float::with_val(p, Float::sum(row.iter())))
            .collect();
        let x = super::super::try_lu_solve(&one, &b, n, p).unwrap();
        assert!(x
            .iter()
            .all(|v| Float::with_val(p, v - 1).abs() < Float::with_val(p, 1) >> 170));
    }
    #[test]
    fn wider_panels_preserve_known_dyadic_factors_and_late_pivots() {
        // Dimension 263 uses panels of 65 and a final panel of 3. Reverse rows
        // force pivots in later panels; every reference operation is dyadic.
        let (n, p) = (263, 192);
        let lower = |i: usize, j: usize| -> i64 {
            if i == j {
                64
            } else if i > j {
                ((i + j) % 7 + 1) as i64
            } else {
                0
            }
        };
        let upper = |i: usize, j: usize| -> i64 {
            if i == j {
                64
            } else if i < j {
                ((i * 3 + j) % 5) as i64 - 2
            } else {
                0
            }
        };
        let mut a = Vec::with_capacity(n * n);
        let mut expected = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let source_row = n - 1 - i;
                let integer: i64 = (0..=source_row.min(j))
                    .map(|k| lower(source_row, k) * upper(k, j))
                    .sum();
                a.push(Float::with_val(p, integer) >> 10);
                expected.push(if i > j {
                    Float::with_val(p, lower(i, j)) >> 6
                } else {
                    Float::with_val(p, upper(i, j)) >> 4
                });
            }
        }
        for tile in [31, 1024] {
            let factors = lu_factor_blocked(&a, n, options(tile)).unwrap();
            assert_eq!(factors.perm, (0..n).rev().collect::<Vec<_>>());
            assert_eq!(factors.lu, expected);
        }
    }
    #[test]
    fn blocked_lu_singular_and_workspace_refuse() {
        let a = vec![Float::with_val(128, 1); 70 * 70];
        assert!(lu_factor_blocked(&a, 70, options(8))
            .err()
            .unwrap()
            .is::<SingularPivot>());
        let mut limited = options(8);
        limited.max_workspace_bytes = 1;
        assert!(lu_factor_blocked(&matrix(70, 128), 70, limited).is_err());
    }

    #[test]
    fn small_default_lu_leaves_shared_run_headroom() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let resources = xc_core::RunResources::with_worker_limit(384 << 20, 1).unwrap();
                let _scope = resources.enter();
                let retained = resources.try_reserve_retained(64 << 20).unwrap();
                let other_work = resources.try_reserve_working(32 << 20).unwrap();
                let available = resources.maximum_bytes() - retained.bytes() - other_work.bytes();
                let (n, p) = (24, 128);
                let mut default = options(1024);
                // Exercise the same default allocation policy as lu_factor_auto
                // without mutating process-wide environment variables or requiring CUDA.
                default.max_workspace_bytes =
                    default_workspace_bytes(n, p, default.workers, available);
                let actual = lu_factor_blocked(&matrix(n, p), n, default).unwrap();
                let usage = resources.usage();
                assert_eq!(usage.retained_bytes, 64 << 20);
                assert_eq!(usage.working_bytes, 32 << 20);
                assert!(
                    usage.peak_bytes <= 256 << 20,
                    "small LU hoarded the shared run budget"
                );
                // The measured peak proves this additional work could have remained
                // admitted throughout the factorization, not merely after its release.
                assert!(usage.peak_bytes + (64 << 20) < resources.maximum_bytes());
                let overlapping = resources.try_reserve_working(64 << 20).unwrap();
                let expected = lu_factor_blocked(&matrix(n, p), n, options(1)).unwrap();
                assert_eq!(actual.perm, expected.perm);
                assert_eq!(actual.lu, expected.lu);
                drop(overlapping);
                drop(other_work);
                drop(retained);
                assert_eq!(
                    resources.usage().retained_bytes + resources.usage().working_bytes,
                    0
                );
            });
    }

    #[test]
    fn default_workspace_saturates_and_respects_remaining_memory() {
        assert_eq!(
            default_workspace_bytes(usize::MAX, u32::MAX, usize::MAX, u64::MAX),
            4 << 30
        );
        assert_eq!(
            default_workspace_bytes(usize::MAX, u32::MAX, usize::MAX, 7),
            7
        );
        assert_eq!(default_workspace_bytes(24, 128, 1, 0), 0);
        assert!(
            default_workspace_bytes(24, 128, 64, u64::MAX)
                > default_workspace_bytes(24, 128, 1, u64::MAX)
        );
    }

    #[test]
    fn automatic_workspace_grows_after_irreducible_small_cuda_admission() {
        // Shifted 3386-bit factorizations reached scalar output tiles: 1x5x1
        // at dimension 21 and 1x7x1 at dimension 31. Splitting could not reduce
        // either CUDA minimum. Exercise their observed admission boundaries
        // without loading CUDA, creating a CPU pool, or allocating products.
        for (dimension, inner, required, available) in [
            (21, 5, 5_243_542u64, 5_202_110u64),
            (31, 7, 5_243_550u64, 5_063_184u64),
        ] {
            // Padded m,n=4, k=16: device + host + bounded self-test storage.
            let cuda_minimum = (5 << 20) + 3 * (64 + 64) + 16 * 16 + 2 * (2 * inner + 1);
            assert_eq!(cuda_minimum, required);
            let fixed = crate::modular_product::admit_workspace(
                required.into(),
                available.into(),
                "CUDA product buffers",
            )
            .unwrap_err();
            assert!(fixed.is::<ModularWorkspaceLimit>());
            let initial = default_workspace_bytes(dimension, 3386, 16, 1 << 30);
            assert_eq!(initial, 192 << 20);
            let retained_product_charge = initial / 2 - available;
            let resources = xc_core::RunResources::with_worker_limit(1 << 30, 16).unwrap();
            let mut attempts = Vec::new();
            let result = with_workspace_growth(initial, 1 << 30, |budget| {
                assert_eq!(resources.usage().working_bytes, 0);
                let _reservation = resources.try_reserve_working(budget)?;
                attempts.push(budget);
                crate::modular_product::admit_workspace(
                    required.into(),
                    (budget / 2 - retained_product_charge).into(),
                    "CUDA product buffers",
                )
                .map_err(|error| error.context("shifted dense factorization"))?;
                Ok(())
            });
            result.unwrap();
            assert_eq!(attempts, [192 << 20, 384 << 20]);
            assert_eq!(resources.usage().working_bytes, 0);
            assert_eq!(resources.usage().peak_bytes, 384 << 20);
        }
    }

    #[test]
    fn automatic_workspace_honors_caps_and_does_not_retry_execution_errors() {
        // A final non-power-of-two allowance is attempted once; the final typed
        // refusal and its original context are preserved when the cap is reached.
        let mut attempts = Vec::new();
        let error = with_workspace_growth(100, 350, |budget| {
            attempts.push(budget);
            crate::modular_product::admit_workspace(400, budget.into(), "CUDA product buffers")
                .map_err(|error| error.context("retained context"))
        })
        .unwrap_err();
        assert_eq!(attempts, [100, 200, 350]);
        let refusal = error.downcast_ref::<ModularWorkspaceLimit>().unwrap();
        assert_eq!((refusal.required, refusal.available), (400, 350));
        assert_eq!(error.to_string(), "retained context");

        // The explicit override passes its original allowance as the ceiling.
        let mut calls = 0;
        let error = with_workspace_growth(100, 100, |budget| {
            calls += 1;
            crate::modular_product::admit_workspace(400, budget.into(), "product")
        })
        .unwrap_err();
        assert!(error.is::<ModularWorkspaceLimit>());
        assert_eq!(calls, 1);
        for message in [
            "CUDA workspace allocation failed",
            "input precision budget exceeded",
        ] {
            let mut calls = 0;
            let error = with_workspace_growth(100, 1000, |_| -> Result<()> {
                calls += 1;
                Err(anyhow!(message))
            })
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(error.to_string(), message);
        }
        let mut attempts = Vec::new();
        with_workspace_growth(u64::MAX - 1, u64::MAX, |budget| {
            attempts.push(budget);
            crate::modular_product::admit_workspace(u64::MAX.into(), budget.into(), "product")
        })
        .unwrap();
        assert_eq!(attempts, [u64::MAX - 1, u64::MAX]);
    }

    #[test]
    fn automatic_workspace_retries_exact_preparation_but_not_singularity_or_reservations() {
        let lu = matrix(21, 3386);
        let mut attempts = 0;
        let prepared = with_workspace_growth(1024, 1 << 20, |budget| {
            attempts += 1;
            prepare_schur_tile(
                &lu,
                21,
                15,
                20,
                OutputTile {
                    row: 20,
                    col: 20,
                    rows: 1,
                    cols: 1,
                },
                3386,
                budget,
            )
        })
        .unwrap();
        assert!(attempts > 1);
        assert_eq!(
            prepared.plan().shape,
            ProductShape {
                rows: 1,
                inner: 5,
                cols: 1
            }
        );
        let mut calls = 0;
        let error = with_workspace_growth(100, 1000, |_| -> Result<()> {
            calls += 1;
            Err(SingularPivot.into())
        })
        .unwrap_err();
        assert!(error.is::<SingularPivot>());
        assert_eq!(calls, 1);

        let resources = xc_core::RunResources::with_worker_limit(100, 1).unwrap();
        let mut calls = 0;
        let error = with_workspace_growth(101, 1000, |budget| {
            calls += 1;
            resources.try_reserve_working(budget).map(drop)?;
            Ok(())
        })
        .unwrap_err();
        assert!(!error.is::<WorkspaceLimit>() && !error.is::<ModularWorkspaceLimit>());
        assert_eq!(calls, 1);
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn exponent_workspace_split_preserves_exact_schur_values_and_signs() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let (n, p, end) = (10, 128, 2);
                let tile = OutputTile {
                    row: end,
                    col: end,
                    rows: 8,
                    cols: 8,
                };
                let tiny = Float::with_val(p, 1) >> 8192u32;
                let mut original = vec![Float::with_val(p, rug::float::Special::NegZero); n * n];
                for i in end..n {
                    original[i * n] = Float::with_val(p, (i + 1) as u32);
                    original[i * n + 1] = if i % 2 == 0 {
                        tiny.clone()
                    } else {
                        -tiny.clone()
                    };
                }
                for j in end..n {
                    original[j] = Float::with_val(p, (j + 2) as u32);
                    original[n + j] = Float::with_val(p, -3);
                }
                let mut limited = options(8);
                limited.max_workspace_bytes = 512 << 10;
                let refusal = prepare_schur_tile(
                    &original,
                    n,
                    0,
                    end,
                    tile,
                    p,
                    limited.max_workspace_bytes / 2,
                )
                .unwrap_err();
                assert!(refusal.is::<WorkspaceLimit>());
                assert!(split_workspace_tile(tile, &refusal).is_some());
                // Backend and domain errors must never enter the split path, even
                // if their text happens to mention workspace.
                assert!(split_workspace_tile(tile, &anyhow!("CUDA workspace failure")).is_none());
                assert!(split_workspace_tile(
                    OutputTile {
                        rows: 1,
                        cols: 1,
                        ..tile
                    },
                    &refusal
                )
                .is_none());

                let mut split = original.clone();
                let mut whole = original.clone();
                update_schur_tile(&mut split, n, 0, end, tile, p, limited, false).unwrap();
                update_schur_tile(&mut whole, n, 0, end, tile, p, options(8), false).unwrap();
                for (actual, expected) in split.iter().zip(&whole) {
                    assert_eq!(actual, expected);
                    assert_eq!(actual.prec(), expected.prec());
                    assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                }
                for i in end..n {
                    for j in end..n {
                        let product = original[i * n].to_rational().unwrap()
                            * original[j].to_rational().unwrap()
                            + original[i * n + 1].to_rational().unwrap()
                                * original[n + j].to_rational().unwrap();
                        assert_eq!(split[i * n + j], Float::with_val(p, -product));
                    }
                }
            });
    }

    #[test]
    fn modular_workspace_split_after_preparation_preserves_schur_values() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let (n, p, end) = (10, 128, 2);
                let tile = OutputTile {
                    row: end,
                    col: end,
                    rows: 8,
                    cols: 8,
                };
                let tiny = Float::with_val(p, 1) >> 8192u32;
                let mut original = vec![Float::with_val(p, rug::float::Special::NegZero); n * n];
                for i in end..n {
                    original[i * n] = Float::with_val(p, (i + 1) as u32);
                    original[i * n + 1] = if i % 2 == 0 {
                        tiny.clone()
                    } else {
                        -tiny.clone()
                    };
                }
                for j in end..n {
                    original[j] = Float::with_val(p, (j + 2) as u32);
                    original[n + j] = Float::with_val(p, -3);
                }
                let prepared = prepare_schur_tile(&original, n, 0, end, tile, p, 64 << 20).unwrap();
                let modular_limits = |bytes| ProductLimits {
                    max_workspace_bytes: bytes,
                    max_input_bits: 10_000,
                    max_modular_scalar_products: u64::MAX,
                    workers: 1,
                };
                // This budget admits the exact integers and binary bridge but not
                // the later CRT arrays/tree/worker storage. Obtain the required
                // amount from the real admission boundary, not a copied estimator.
                let probe_budget = 16 << 20;
                let refusal = prepared
                    .modular_product(
                        Round::Nearest,
                        ProductBackend::Cpu,
                        modular_limits(probe_budget),
                    )
                    .unwrap_err();
                let admission = refusal.downcast_ref::<ModularWorkspaceLimit>().unwrap();
                let product_budget = u64::try_from(
                    u128::from(probe_budget) + admission.required - admission.available - 1,
                )
                .unwrap();
                drop(prepared);
                let prepared =
                    prepare_schur_tile(&original, n, 0, end, tile, p, product_budget).unwrap();
                let refusal = prepared
                    .modular_product(
                        Round::Nearest,
                        ProductBackend::Cpu,
                        modular_limits(product_budget),
                    )
                    .unwrap_err();
                assert!(refusal.is::<ModularWorkspaceLimit>());
                assert!(split_workspace_tile(tile, &refusal).is_some());

                // The bridge's separate pre-execution admission also remains typed.
                let bridge_refusal = prepared
                    .modular_product(
                        Round::Nearest,
                        ProductBackend::Cpu,
                        modular_limits(
                            u64::try_from(prepared.plan().estimated_workspace_bytes).unwrap(),
                        ),
                    )
                    .unwrap_err();
                assert!(bridge_refusal.is::<WorkspaceLimit>());
                assert!(split_workspace_tile(tile, &bridge_refusal).is_some());

                let mut invalid_width = modular_limits(64 << 20);
                invalid_width.max_input_bits = 0;
                let domain_refusal = prepared
                    .modular_product(Round::Nearest, ProductBackend::Cpu, invalid_width)
                    .unwrap_err();
                assert!(split_workspace_tile(tile, &domain_refusal).is_none());
                assert!(
                    split_workspace_tile(tile, &anyhow!("CUDA workspace allocation failed"))
                        .is_none()
                );
                drop(prepared);

                let mut limited = options(8);
                limited.max_workspace_bytes = product_budget * 2;
                let mut split = original.clone();
                let mut whole = original.clone();
                // Select the modular path using its CPU backend: exercise the exact
                // failing admission/retry path without requiring a physical GPU.
                update_schur_tile(&mut split, n, 0, end, tile, p, limited, true).unwrap();
                update_schur_tile(&mut whole, n, 0, end, tile, p, options(8), false).unwrap();
                for (actual, expected) in split.iter().zip(&whole) {
                    assert_eq!(actual, expected);
                    assert_eq!(actual.prec(), expected.prec());
                    assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                }
            });
    }

    #[test]
    fn blocked_lu_splits_wide_exponent_tiles_without_changing_factors() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let (n, p) = (40, 128);
                let tiny = Float::with_val(p, 1) >> 100_000u32;
                let a: Vec<_> = (0..n * n)
                    .map(|index| {
                        let (row, col) = (index / n, index % n);
                        if row == col {
                            Float::with_val(p, 4)
                        } else if row > col {
                            if (row + col) % 2 == 0 {
                                tiny.clone()
                            } else {
                                Float::with_val(p, 2)
                            }
                        } else {
                            Float::with_val(p, 0)
                        }
                    })
                    .collect();
                let mut limited = options(30);
                limited.max_workspace_bytes = 4 << 20;
                // The nominal precision planner admits edge 15. Actual aligned
                // operand widths refuse that tile before any integer allocation.
                let refusal = prepare_schur_tile(
                    &a,
                    n,
                    0,
                    10,
                    OutputTile {
                        row: 10,
                        col: 10,
                        rows: 15,
                        cols: 15,
                    },
                    p,
                    2 << 20,
                )
                .unwrap_err();
                assert!(refusal.is::<WorkspaceLimit>());
                let split = lu_factor_blocked(&a, n, limited).unwrap();
                let whole = lu_factor_blocked(&a, n, options(30)).unwrap();
                assert_eq!(split.perm, whole.perm);
                for (actual, expected) in split.lu.iter().zip(&whole.lu) {
                    assert_eq!(actual, expected);
                    assert_eq!(actual.prec(), expected.prec());
                    assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                }
            });
    }

    #[test]
    #[ignore = "requires an admitted physical CUDA device"]
    fn blocked_lu_cuda_wide_limbs_preserve_cpu_factors() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                let (n, p) = (9, 128);
                let tiny = Float::with_val(p, 1) >> 13_000u32;
                let a: Vec<_> = (0..n * n)
                    .map(|index| {
                        let (row, col) = (index / n, index % n);
                        if row == col {
                            Float::with_val(p, 4)
                        } else if (row + col) % 2 == 0 {
                            tiny.clone()
                        } else {
                            Float::with_val(p, (row + col) as u32 + 1) / 128
                        }
                    })
                    .collect();
                let mut policy = options(4);
                policy.workers = 2;
                policy.max_workspace_bytes = 1 << 30;
                let expected = lu_factor_blocked(&a, n, policy).unwrap();
                policy.backend = ProductBackend::Cuda;
                let actual = lu_factor_blocked(&a, n, policy).unwrap();
                assert_eq!(actual.perm, expected.perm);
                for (left, right) in actual.lu.iter().zip(&expected.lu) {
                    assert_eq!(left, right);
                    assert_eq!(left.prec(), right.prec());
                    assert_eq!(left.is_sign_negative(), right.is_sign_negative());
                }
            });
    }
    #[test]
    fn default_scalar_lu_retains_legacy_bits() {
        let a = matrix(1, 128);
        let old = lu_factor(&a, 1).unwrap();
        let new = lu_factor_auto(&a, 1).unwrap();
        assert_eq!(old.lu, new.lu);
        assert_eq!(old.perm, new.perm);
        assert!(dense_lu_arithmetic(1).is_none());
        assert_eq!(dense_lu_arithmetic(2), Some(ARITHMETIC));
    }

    #[test]
    fn exact_schur_subtraction_preserves_a_pivot_below_product_rounding() {
        let (n, p) = (65, 64);
        let mut a = vec![Float::with_val(p, 0); n * n];
        for i in 0..n {
            a[i * n + i] = Float::with_val(p, 1);
        }
        // L21 = (1 + 2^-40)/2 and U12 = 2(1 - 2^-40).
        // Their exact product is 1 - 2^-80, which rounds to 1 at p=64.
        // Prematurely rounding that product makes the final pivot zero;
        // subtracting its exact value leaves the representable pivot 2^-80.
        let small = Float::with_val(p, 1) >> 40u32;
        let mut plus = Float::with_val(p, 1);
        plus += &small;
        let mut minus = Float::with_val(p, 1);
        minus -= &small;
        a[0] = Float::with_val(p, 2);
        a[n - 1] = Float::with_val(p, &minus * 2);
        a[(n - 1) * n] = plus.clone();
        let lower = Float::with_val(p, &plus / 2);
        let prematurely_rounded = Float::with_val(p, &lower * &a[n - 1]);
        assert_eq!(prematurely_rounded, 1);

        let factors = lu_factor_blocked(&a, n, options(1)).unwrap();
        let expected = Float::with_val(p, 1) >> 80u32;
        assert_eq!(factors.perm, (0..n).collect::<Vec<_>>());
        assert_eq!(factors.lu[n * n - 1], expected);
        // Exact rational reconstruction of the cancellation-sensitive entry.
        let reconstructed = factors.lu[(n - 1) * n].to_rational().unwrap()
            * factors.lu[n - 1].to_rational().unwrap()
            + factors.lu[n * n - 1].to_rational().unwrap();
        assert_eq!(reconstructed, 1);
    }

    #[test]
    fn later_panel_pivots_swap_already_computed_lower_columns() {
        let (n, p) = (131, 128);
        // Reverse the rows of L*(4I), with distinct exact dyadic multipliers.
        // Each diagonal pivot 4 strictly dominates all remaining candidates.
        // The second panel therefore must swap rows with unequal old L data;
        // a trailing-only swap cannot accidentally pass this fixture.
        let mut a = Vec::with_capacity(n * n);
        for stored_row in 0..n {
            let original_row = n - 1 - stored_row;
            for col in 0..n {
                let value = if col == original_row {
                    Float::with_val(p, 4)
                } else if col < original_row {
                    Float::with_val(p, ((original_row * 3 + col * 5) % 7 + 1) as u32) / 8
                } else {
                    Float::with_val(p, 0)
                };
                a.push(value);
            }
        }
        let factors = lu_factor_blocked(&a, n, options(19)).unwrap();
        assert_eq!(factors.perm, (0..n).rev().collect::<Vec<_>>());
        for row in 0..n {
            for col in 0..n {
                let expected = if row == col {
                    Float::with_val(p, 4)
                } else if col < row {
                    Float::with_val(p, ((row * 3 + col * 5) % 7 + 1) as u32) / 32
                } else {
                    Float::with_val(p, 0)
                };
                assert_eq!(factors.lu[row * n + col], expected, "entry ({row}, {col})");
            }
        }
    }

    #[test]
    fn blocked_lu_refuses_nonfinite_mixed_precision_and_malformed_inputs() {
        let mut a = matrix(3, 128);
        a[4] = Float::with_val(128, rug::float::Special::Nan);
        assert!(lu_factor_blocked(&a, 3, options(8)).is_err());
        a[4] = Float::with_val(192, 9);
        assert!(lu_factor_blocked(&a, 3, options(8)).is_err());
        assert!(lu_factor_blocked(&a, 4, options(8)).is_err());
        assert!(lu_factor_blocked(&[], 0, options(8)).is_err());
    }
}
