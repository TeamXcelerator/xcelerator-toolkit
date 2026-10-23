// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Rigorous batched contraction of an exact integer matrix divided by D with
//! interval columns. CPU and CUDA multiply the same stored dyadic operands
//! exactly. This does not reproduce entrywise interval-product endpoints.
//!
//! For a stored midpoint M and outward radius H, X is in [M-H,M+H]. Thus
//! A X / D is in [(A M - |A| H)/D, (A M + |A| H)/D] for D > 0.
//! The center product is exact, the radius product is rounded upward, and the
//! final subtraction/addition/division are directed. A caller may accept a
//! uniquely rounded final action, retaining its own exact-zero/sign policy.
//! There is no approximate GPU answer or numerical-error fallback here.

use crate::exact_product::{self, ExactProductLimits, PreparedProduct, ProductShape};
use crate::modular_product::{ProductBackend, ProductLimits};
use crate::mpfr_interval::{ensure_uniform_exponent_range, MpfrInterval};
use anyhow::{anyhow, ensure, Context, Result};
use rug::{float::Round, Float, Integer};

/// Explicit execution policy. Neither environment defaults nor artifact keys
/// are chosen by this helper. Workspace includes all helper-owned arrays,
/// integer/BigInt bridges, CRT and admitted device buffers. Borrowed inputs and
/// vendor context/library allocations remain the caller/process's responsibility.
#[derive(Clone, Copy, Debug)]
pub struct ContractionOptions {
    pub backend: ProductBackend,
    pub max_workspace_bytes: u64,
    pub max_integer_bits: u32,
    pub max_exponent_span: u32,
    pub max_scalar_products: u128,
    pub max_modular_scalar_products: u64,
    pub workers: usize,
}

/// Only a refusal established before entering either product backend has this
/// marker. A caller may select a smaller tile or its reference route on it.
#[derive(Debug)]
pub struct ContractionAdmission;
impl std::fmt::Display for ContractionAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("rational interval contraction admission refused")
    }
}
impl std::error::Error for ContractionAdmission {}

/// Once a backend call is entered, every failure is execution failure, including
/// later limb/second-product resource refusal. Do not catch a nested allocation
/// error and retry on another backend after this marker appears.
#[derive(Debug)]
pub struct ContractionExecution;
impl std::fmt::Display for ContractionExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("rational interval contraction execution failed")
    }
}
impl std::error::Error for ContractionExecution {}

/// Keep this owner alive while consuming its borrowed result slice. The whole
/// declared workspace remains reserved until drop, conservatively covering the
/// returned interval allocation after product scratch has been released.
#[derive(Debug)]
pub struct Contraction {
    values: Vec<MpfrInterval>,
    _reservation: Option<xc_core::MemoryReservation>,
}
impl Contraction {
    pub fn values(&self) -> &[MpfrInterval] {
        &self.values
    }
}

fn admitted(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(anyhow!(message.to_owned()).context(ContractionAdmission))
    }
}

fn bytes(bits: u32) -> u128 {
    u128::from(bits).div_ceil(64) * 8 + 96
}

// A bound on a common integer alignment of both midpoint and radius arrays,
// scanned before allocating them. Rounding cannot introduce a lower nonzero
// dyadic bit than the exact operands, except midpoint division by two. Two
// extra bits also cover outward carry in subtraction/radius and the top bound.
fn right_width(columns: &[MpfrInterval]) -> Result<u32> {
    let mut bottom = i64::MAX;
    let mut top = i64::MIN;
    for endpoint in columns.iter().flat_map(|x| [x.lower(), x.upper()]) {
        if let Some(significand) = endpoint.get_significand() {
            let exponent = i64::from(endpoint.get_exp().unwrap());
            let trailing = significand.find_one(0).unwrap();
            bottom = bottom
                .min(exponent - i64::from(significand.significant_bits()) + i64::from(trailing));
            top = top.max(exponent);
        }
    }
    if bottom == i64::MAX {
        return Ok(0);
    }
    u32::try_from(top - bottom + 2).map_err(|_| {
        anyhow!("interval dyadic alignment exceeds supported width").context(ContractionAdmission)
    })
}

fn product(
    prepared: &PreparedProduct,
    round: Round,
    options: ContractionOptions,
    product_budget: u64,
) -> Result<Vec<Float>> {
    // Cpu is also the exact GMP oracle; Auto/Cuda use the existing complete CRT
    // and exact wide-limb reconstruction, with their strict backend contracts.
    let result = match options.backend {
        ProductBackend::Cpu => prepared.cpu_product(round),
        backend => prepared.modular_product(
            round,
            backend,
            ProductLimits {
                max_workspace_bytes: product_budget,
                max_input_bits: u64::from(options.max_integer_bits),
                max_modular_scalar_products: options.max_modular_scalar_products / 2,
                workers: options.workers,
            },
        ),
    };
    result.context(ContractionExecution)
}

/// Contract row-major A/D (rows x inner) and interval columns (inner x cols).
///
/// Positive dimensions, D > 0 and uniform endpoint precision are required.
/// Scalar/modular operation limits cover BOTH products. The current shared run
/// ledger, when present, reserves the complete allowance before owned numeric
/// arrays are allocated. Without a shared run the explicit workspace policy
/// still applies. No active cache is accessed and no result is published.
pub fn contract(
    numerators: &[Integer],
    denominator: &Integer,
    columns: &[MpfrInterval],
    shape: ProductShape,
    precision: u32,
    options: ContractionOptions,
) -> Result<Contraction> {
    ensure_uniform_exponent_range()?;
    ensure!(
        shape.rows > 0 && shape.inner > 0 && shape.cols > 0,
        "positive contraction shape required"
    );
    ensure!(
        shape.rows.checked_mul(shape.inner) == Some(numerators.len())
            && shape.inner.checked_mul(shape.cols) == Some(columns.len()),
        "contraction shape mismatch"
    );
    ensure!(
        denominator > &0,
        "positive contraction denominator required"
    );
    ensure!(
        (rug::float::prec_min()..=rug::float::prec_max()).contains(&precision),
        "unsupported contraction precision"
    );
    ensure!(
        options.workers > 0 && options.max_workspace_bytes > 0,
        "positive contraction resource limits required"
    );
    for column in columns {
        column.validate()?;
        ensure!(
            column.precision() == precision,
            "uniform contraction precision required"
        );
    }
    let count = shape
        .rows
        .checked_mul(shape.cols)
        .context("contraction output shape overflow")?;
    for (length, width) in [
        (numerators.len(), std::mem::size_of::<Float>()),
        (columns.len(), std::mem::size_of::<Float>()),
        (count, std::mem::size_of::<MpfrInterval>()),
    ] {
        admitted(
            length
                .checked_mul(width)
                .is_some_and(|n| n <= isize::MAX as usize),
            "contraction vector capacity exceeded",
        )?;
    }
    let operations = (count as u128)
        .checked_mul(shape.inner as u128)
        .and_then(|n| n.checked_mul(2))
        .context("contraction operation count overflow")?;
    admitted(
        operations <= options.max_scalar_products,
        "contraction scalar operation budget exceeded",
    )?;
    let left_bits = numerators
        .iter()
        .map(Integer::significant_bits)
        .max()
        .unwrap_or(0);
    let right_bits = right_width(columns)?;
    let sum_bits = usize::BITS - (shape.inner - 1).leading_zeros();
    let output_bits = if left_bits == 0 || right_bits == 0 {
        0
    } else {
        left_bits
            .checked_add(right_bits)
            .and_then(|n| n.checked_add(sum_bits))
            .context("contraction output width overflow")?
    };
    admitted(
        left_bits <= options.max_integer_bits
            && right_bits <= options.max_integer_bits
            && output_bits <= options.max_integer_bits,
        "contraction integer width budget exceeded",
    )?;
    // The supplied denominator is borrowed, but division scratch must be charged.
    let widest = output_bits
        .max(precision)
        .max(denominator.significant_bits());
    // A, |A| (the latter changes A in place), M/H, retained exact center and
    // radius outputs, final interval outputs, and serial arithmetic scratch.
    // This is extra storage OUTSIDE the existing exact-product combined budget.
    let retained = 4
        * ((numerators.len() as u128) * bytes(left_bits)
            + 2 * (columns.len() as u128) * bytes(precision)
            + (count as u128) * (2 * bytes(output_bits) + 4 * bytes(precision))
            + 16 * bytes(widest));
    admitted(
        retained < u128::from(options.max_workspace_bytes),
        "contraction retained workspace budget exceeded",
    )?;
    let resources = xc_core::RunResources::current();
    if let Some(resources) = &resources {
        resources.check_workers(options.workers)?;
    }
    let reservation = resources
        .as_ref()
        .map(|r| r.try_reserve_working(options.max_workspace_bytes))
        .transpose()
        .map_err(|e| anyhow!(e).context(ContractionAdmission))?;
    let product_budget = options.max_workspace_bytes - retained as u64;
    let exact_limits = ExactProductLimits {
        max_workspace_bytes: u128::from(product_budget),
        max_scalar_products: options.max_scalar_products / 2,
        max_exponent_span: options.max_exponent_span,
        max_integer_bits: options.max_integer_bits,
    };
    let mut left: Vec<_> = numerators
        .iter()
        .map(|a| Float::with_val(a.significant_bits().max(2), a))
        .collect();
    let mut centers = Vec::with_capacity(columns.len());
    let mut radii = Vec::with_capacity(columns.len());
    for interval in columns {
        let mut center = Float::with_val(precision, interval.lower() + interval.upper());
        center >>= 1;
        ensure!(
            center.is_finite(),
            "contraction midpoint outside supported finite range"
        );
        let below = Float::with_val_round(precision, &center - interval.lower(), Round::Up).0;
        let above = Float::with_val_round(precision, interval.upper() - &center, Round::Up).0;
        let radius = if below > above { below } else { above };
        ensure!(
            radius.is_finite() && radius >= 0,
            "contraction radius outside supported finite range"
        );
        centers.push(center);
        radii.push(radius);
    }
    let prepared = exact_product::prepare(&left, &centers, shape, precision, exact_limits)
        .and_then(|p| p.with_exact_output(u128::from(product_budget)))
        .map_err(|e| {
            if e.is::<exact_product::WorkspaceLimit>() {
                e.context(ContractionAdmission)
            } else {
                e
            }
        })?;
    let center_product = product(&prepared, Round::Nearest, options, product_budget)?;
    drop(prepared);
    for value in &mut left {
        value.abs_mut();
    }
    // From this point even a preparation/admission error follows completed
    // backend work. Wrap it as execution failure, never a fallback signal.
    let prepared = exact_product::prepare(&left, &radii, shape, precision, exact_limits)
        .context(ContractionExecution)?;
    let radius_product = product(&prepared, Round::Up, options, product_budget)?;
    drop(prepared);
    let mut values = Vec::with_capacity(count);
    for (center, radius) in center_product.iter().zip(&radius_product) {
        let lo = Float::with_val_round(precision, center - radius, Round::Down).0;
        let hi = Float::with_val_round(precision, center + radius, Round::Up).0;
        let lower = Float::with_val_round(precision, &lo / denominator, Round::Down).0;
        let upper = Float::with_val_round(precision, &hi / denominator, Round::Up).0;
        values.push(MpfrInterval::new(lower, upper).context(ContractionExecution)?);
    }
    Ok(Contraction {
        values,
        _reservation: reservation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;

    fn options() -> ContractionOptions {
        ContractionOptions {
            backend: ProductBackend::Cpu,
            max_workspace_bytes: 64 << 20,
            max_integer_bits: 100_000,
            max_exponent_span: 100_000,
            max_scalar_products: 1_000_000,
            max_modular_scalar_products: 1_000_000_000,
            workers: 1,
        }
    }

    fn bounds(lo: i32, hi: i32, p: u32) -> MpfrInterval {
        MpfrInterval::new(Float::with_val(p, lo), Float::with_val(p, hi)).unwrap()
    }

    fn oracle(
        a: &[Integer],
        d: &Integer,
        x: &[MpfrInterval],
        shape: ProductShape,
    ) -> Vec<(Rational, Rational)> {
        (0..shape.rows)
            .flat_map(|i| {
                (0..shape.cols).map(move |j| {
                    let mut lo = Rational::new();
                    let mut hi = Rational::new();
                    for k in 0..shape.inner {
                        let value = &a[i * shape.inner + k];
                        let interval = &x[k * shape.cols + j];
                        let (l, u) = if value >= &0 {
                            (interval.lower(), interval.upper())
                        } else {
                            (interval.upper(), interval.lower())
                        };
                        lo += l.to_rational().unwrap() * value;
                        hi += u.to_rational().unwrap() * value;
                    }
                    (lo / d, hi / d)
                })
            })
            .collect()
    }

    #[test]
    fn mixed_sign_interval_columns_enclose_exact_rational_oracle() {
        let shape = ProductShape {
            rows: 3,
            inner: 4,
            cols: 3,
        };
        let a = [1, -3, 0, 7, -4, 2, 9, -1, 0, 0, 0, 0].map(Integer::from);
        let d = Integer::from(37);
        let x = (0..12)
            .map(|i| bounds(i - 7, i - 3, 96))
            .collect::<Vec<_>>();
        let result = contract(&a, &d, &x, shape, 96, options()).unwrap();
        for (got, (lo, hi)) in result.values().iter().zip(oracle(&a, &d, &x, shape)) {
            assert!(got.lower() <= &lo);
            assert!(got.upper() >= &hi);
        }
        assert!(result.values()[6..]
            .iter()
            .all(|v| v.lower().is_zero() && v.upper().is_zero()));
    }

    #[test]
    fn point_columns_preserve_cancellation_and_unique_output_rounding() {
        let p = 256;
        let a = [
            Integer::from(1) << 512u32,
            -(Integer::from(1) << 512u32),
            Integer::from(3),
        ];
        let x = [
            MpfrInterval::point(Float::with_val(p, 1)),
            MpfrInterval::point(Float::with_val(p, 1)),
            MpfrInterval::point(Float::with_val(p, 7)),
        ];
        let shape = ProductShape {
            rows: 1,
            inner: 3,
            cols: 1,
        };
        let result = contract(&a, &Integer::from(19), &x, shape, p, options()).unwrap();
        let expected = Float::with_val(64, Rational::from((21, 19)));
        assert_eq!(Float::with_val(64, result.values()[0].lower()), expected);
        assert_eq!(Float::with_val(64, result.values()[0].upper()), expected);
    }

    #[test]
    fn ledger_is_reserved_before_work_and_held_by_result() {
        let resources = xc_core::RunResources::with_worker_limit(128 << 20, 1).unwrap();
        let _scope = resources.enter();
        let a = [Integer::from(1)];
        let x = [bounds(-1, 1, 64)];
        let shape = ProductShape {
            rows: 1,
            inner: 1,
            cols: 1,
        };
        let occupied = resources.try_reserve_retained(100 << 20).unwrap();
        let error = contract(&a, &Integer::from(1), &x, shape, 64, options()).unwrap_err();
        assert!(error.is::<ContractionAdmission>());
        assert_eq!(resources.usage().working_bytes, 0);
        drop(occupied);
        let result = contract(&a, &Integer::from(1), &x, shape, 64, options()).unwrap();
        assert_eq!(resources.usage().working_bytes, 64 << 20);
        drop(result);
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn malformed_domain_and_budget_refuse_before_backend() {
        let a = [Integer::from(1)];
        let x = [bounds(-1, 1, 64)];
        let shape = ProductShape {
            rows: 1,
            inner: 1,
            cols: 1,
        };
        let err = contract(&a, &Integer::from(0), &x, shape, 64, options()).unwrap_err();
        assert!(!err.is::<ContractionAdmission>() && !err.is::<ContractionExecution>());
        let tiny = ContractionOptions {
            max_workspace_bytes: 1,
            ..options()
        };
        assert!(contract(&a, &Integer::from(1), &x, shape, 64, tiny)
            .unwrap_err()
            .is::<ContractionAdmission>());
        let narrow = ContractionOptions {
            max_integer_bits: 1,
            ..options()
        };
        assert!(contract(&a, &Integer::from(1), &x, shape, 64, narrow)
            .unwrap_err()
            .is::<ContractionAdmission>());
        assert!(contract(&a, &Integer::from(1), &x, shape, 96, options()).is_err());
    }

    #[test]
    fn directed_enclosure_survives_midpoint_rounding_and_extreme_scales() {
        let p = 96;
        let shape = ProductShape {
            rows: 2,
            inner: 3,
            cols: 2,
        };
        let a = [-7, 3, 11, 2, -13, 5].map(Integer::from);
        let d = (Integer::from(1) << 120u32) + 17;
        let x = (0..6)
            .map(|i| {
                let mut lo = Float::with_val(p, Rational::from((2 * i - 7, 19)));
                lo <<= if i % 2 == 0 { 300 } else { -300 };
                let mut hi = lo.clone();
                hi.next_up();
                hi.next_up();
                hi.next_up();
                MpfrInterval::new(lo, hi).unwrap()
            })
            .collect::<Vec<_>>();
        let result = contract(&a, &d, &x, shape, p, options()).unwrap();
        for (got, (lo, hi)) in result.values().iter().zip(oracle(&a, &d, &x, shape)) {
            assert!(got.lower() <= &lo);
            assert!(got.upper() >= &hi);
        }
    }

    #[test]
    #[ignore = "explicit supervised CUDA qualification only; never run beside an uncoordinated claim GPU workload"]
    fn cuda_contraction_matches_exact_cpu_endpoints() {
        let p = 192;
        let shape = ProductShape {
            rows: 16,
            inner: 17,
            cols: 9,
        };
        let a = (0..shape.rows * shape.inner)
            .map(|i| {
                let value = (Integer::from(1) << 120u32) + Integer::from(i * 17 + 3);
                if i % 3 == 0 {
                    -value
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let x = (0..shape.inner * shape.cols)
            .map(|i| {
                let lo = Float::with_val(p, Rational::from((i as i64 - 70, 197)));
                let mut hi = lo.clone();
                if hi.is_zero() {
                    hi = Float::with_val(p, 1) >> p;
                } else {
                    hi.next_up();
                }
                MpfrInterval::new(lo, hi).unwrap()
            })
            .collect::<Vec<_>>();
        let d = (Integer::from(1) << 121u32) + 17;
        let cpu = contract(&a, &d, &x, shape, p, options()).unwrap();
        let cuda = contract(
            &a,
            &d,
            &x,
            shape,
            p,
            ContractionOptions {
                backend: ProductBackend::Cuda,
                max_workspace_bytes: 512 << 20,
                ..options()
            },
        )
        .unwrap();
        for (a, b) in cpu.values().iter().zip(cuda.values()) {
            for (a, b) in [(a.lower(), b.lower()), (a.upper(), b.upper())] {
                assert_eq!(a.to_integer_exp(), b.to_integer_exp());
                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
                assert_eq!(a.prec(), b.prec());
            }
        }
    }

    #[test]
    fn zero_endpoints_have_no_alignment_exponent_and_keep_directed_zero_hull() {
        for p in [64, 192] {
            let positive = Float::with_val(p, 0);
            let negative = -positive.clone();
            for (lo, hi) in [
                (&positive, &positive),
                (&negative, &negative),
                (&negative, &positive),
            ] {
                let columns = [MpfrInterval::new(lo.clone(), hi.clone()).unwrap()];
                assert_eq!(right_width(&columns).unwrap(), 0);
                for numerator in [-1, 0, 1] {
                    let result = contract(
                        &[Integer::from(numerator)],
                        &Integer::from(3),
                        &columns,
                        ProductShape {
                            rows: 1,
                            inner: 1,
                            cols: 1,
                        },
                        p,
                        options(),
                    )
                    .unwrap();
                    let value = &result.values()[0];
                    assert!(value.lower().is_zero() && value.upper().is_zero());
                    assert!(value.lower().is_sign_negative());
                    assert!(!value.upper().is_sign_negative());
                }
            }
        }
    }

    #[test]
    fn extreme_zero_successor_span_is_refused_before_backend() {
        let p = 64;
        let lower = Float::with_val(p, 0);
        let mut upper = lower.clone();
        upper.next_up();
        let error = contract(
            &[Integer::from(1), Integer::from(1)],
            &Integer::from(1),
            &[
                MpfrInterval::new(lower, upper).unwrap(),
                MpfrInterval::point(Float::with_val(p, 1)),
            ],
            ProductShape {
                rows: 1,
                inner: 2,
                cols: 1,
            },
            p,
            options(),
        )
        .unwrap_err();
        assert!(error.is::<ContractionAdmission>());
        assert!(!error.is::<ContractionExecution>());
    }

    #[test]
    #[ignore = "supervised high-precision CPU/CUDA contraction timing; no concurrent claim GPU workload"]
    fn measure_high_precision_cpu_cuda_contractions() {
        use std::time::Instant;
        let shape = ProductShape {
            rows: 64,
            inner: 121,
            cols: 8,
        };
        for source_precision in [3386u32, 6708] {
            let work = 2 * source_precision + 64;
            let fixture = || {
                let a = (0..shape.rows * shape.inner)
                    .map(|i| {
                        let value =
                            ((Integer::from(1) << (source_precision + 350)) / (i % 113 + 3)) + i;
                        if i % 3 == 0 {
                            -value
                        } else {
                            value
                        }
                    })
                    .collect::<Vec<_>>();
                let x = (0..shape.inner * shape.cols)
                    .map(|i| {
                        let lower = Float::with_val(work, Rational::from((i as i64 - 471, 997)));
                        let mut upper = lower.clone();
                        if upper.is_zero() {
                            upper = Float::with_val(work, 1) >> work;
                        } else {
                            upper.next_up();
                        }
                        MpfrInterval::new(lower, upper).unwrap()
                    })
                    .collect::<Vec<_>>();
                let denominator = (Integer::from(1) << (source_precision + 351)) + 17;
                (a, denominator, x)
            };
            let run = |backend| {
                let start = Instant::now();
                let (a, d, x) = fixture();
                let result = contract(
                    &a,
                    &d,
                    &x,
                    shape,
                    work,
                    ContractionOptions {
                        backend,
                        max_workspace_bytes: 1 << 30,
                        workers: 4,
                        ..options()
                    },
                )
                .unwrap();
                (result, start.elapsed().as_secs_f64())
            };
            let (cpu, cpu_seconds) = run(ProductBackend::Cpu);
            let (cuda, cuda_seconds) = run(ProductBackend::Cuda);
            let (a, d, x) = fixture();
            for ((cpu, cuda), (lower, upper)) in cpu
                .values()
                .iter()
                .zip(cuda.values())
                .zip(oracle(&a, &d, &x, shape))
            {
                assert!(cuda.lower() <= &lower && cuda.upper() >= &upper);
                for (cpu, cuda) in [(cpu.lower(), cuda.lower()), (cpu.upper(), cuda.upper())] {
                    assert_eq!(cpu.to_integer_exp(), cuda.to_integer_exp());
                    assert_eq!(cpu.is_sign_negative(), cuda.is_sign_negative());
                    assert_eq!(cpu.prec(), cuda.prec());
                }
            }
            println!(
                "exact_interval_contraction source_p={source_precision} work={work} rows=64 inner=121 cols=8 preparation=included oracle=exact_rational endpoint_identity=passed cpu_seconds={cpu_seconds:.6} cuda_seconds={cuda_seconds:.6} ratio={:.4}",
                cpu_seconds / cuda_seconds
            );
        }
    }
}
