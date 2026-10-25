//! Directed residual bounds for actual finite symmetric source matrices.
use super::super::retained_evidence::finite_math::{abs, scale, scale_float};
use anyhow::{bail, Result};
use rayon::prelude::*;
use rug::{float::Round, Float};
use xc_numerics::mpfr_interval::MpfrInterval as I;

pub(super) struct ResidualBounds {
    /// Upper bound on ||A v-lambda v||_2 / ||v||_2, in eigenvalue units.
    pub eigenvalue_error_upper: Float,
    /// Upper bound on ||A v-lambda v||_infinity / ||v||_infinity.
    pub vector_scaled_residual_upper: Float,
}

/// Owned scratch only: the source matrix/vector remain borrowed. The bound
/// covers normalized vector/result collections and constant-sized arithmetic
/// scratch for at most one task per row, independently of the Rayon pool size.
pub(super) fn workspace_bytes(n: usize, p: u32) -> Result<u128> {
    if n == 0 || !(64..=1_000_000).contains(&p) {
        bail!("invalid source residual workspace dimensions or precision");
    }
    let scalar = u128::from(p + 64).div_ceil(64) * 8 + 160;
    (n as u128)
        .checked_mul(64)
        .and_then(|count| count.checked_add(512))
        .and_then(|count| count.checked_mul(scalar))
        .ok_or_else(|| anyhow::anyhow!("source residual workspace estimate overflow"))
}

/// One evaluation shared by the consumers of a single (A, v, lambda, p): the
/// value `weil_eigvec_cache::relative_residual_norm` returns, with the bounds
/// it was derived from. `None` exactly when that function returns `None`.
/// Every later consumer of the same inputs would recompute identical bounds.
pub(super) fn relative_residual(
    a: &[Float],
    dim: usize,
    v: &[Float],
    lambda: &Float,
    p: u32,
) -> Option<(Float, ResidualBounds)> {
    if v.len() != dim || dim.checked_mul(dim) != Some(a.len()) {
        return None;
    }
    let bounds = evaluate(a, v, lambda, p).ok()?;
    let bound = &bounds.vector_scaled_residual_upper;
    let result = Float::with_val_round(p, bound, Round::Up).0;
    (result.is_finite() && (bound.is_zero() || !result.is_zero())).then_some((result, bounds))
}

pub(super) fn evaluate(a: &[Float], v: &[Float], lambda: &Float, p: u32) -> Result<ResidualBounds> {
    let n = v.len();
    if n == 0
        || n.checked_mul(n) != Some(a.len())
        || !(64..=1_000_000).contains(&p)
        || a.iter()
            .chain(v)
            .chain(std::iter::once(lambda))
            .any(|x| !x.is_finite() || x.prec() > p)
        || v.iter().all(Float::is_zero)
    {
        bail!("invalid finite source-residual dimensions, precision or vector");
    }
    for i in 0..n {
        for j in i + 1..n {
            if a[i * n + j] != a[j * n + i] {
                bail!("source eigenvalue residual requires exact stored symmetry");
            }
        }
    }
    let work = p + 64;
    let scratch_bytes = workspace_bytes(n, p)?;
    let maximum_bytes = super::super::capture_runtime::CaptureResourcePolicy::from_environment()?
        .maximum_working_bytes;
    if scratch_bytes > u128::from(maximum_bytes) {
        bail!("source residual exceeds numerical workspace budget: estimated_bytes={scratch_bytes}, maximum_working_bytes={maximum_bytes}");
    }
    let _scratch_reservation = super::super::capture_runtime::reserve_working(scratch_bytes)?;
    let ae = a
        .iter()
        .chain(std::iter::once(lambda))
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let ve = i64::from(v.iter().filter_map(Float::get_exp).max().unwrap());
    // Entries and rows run in parallel. Each keeps its own operation order and
    // results are consumed in index order, so values and the first error are
    // those of the serial evaluation at every thread count.
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    // Preserve the allocating route's error order: every matrix normalization
    // is checked before normalizing the vector or evaluating a row. Retaining
    // only one result per row avoids an O(n^2) intermediate collection. The
    // same exact scaling is repeated below, without a dense interval image.
    let matrix_checks = a
        .par_chunks(n)
        .map(|row| -> Result<()> {
            for entry in row {
                I::from_float(&scale_float(entry, -ae, work)?, work)?;
            }
            Ok(())
        })
        .collect::<Vec<_>>();
    for result in matrix_checks {
        result?;
    }
    let normalized = |xs: &[Float], exponent: i64| -> Result<Vec<I>> {
        xs.par_iter()
            .map(|x| Ok(I::from_float(&scale_float(x, -exponent, work)?, work)?))
            .collect::<Vec<Result<I>>>()
            .into_iter()
            .collect()
    };
    let v = normalized(v, ve)?;
    let lambda = I::from_float(&scale_float(lambda, -ae, work)?, work)?;
    let rows = a
        .par_chunks(n)
        .enumerate()
        .map(|(i, row)| -> Result<I> {
            let mut value = I::from_i64(0, work);
            for (entry, x) in row.iter().zip(&v) {
                let entry = I::from_float(&scale_float(entry, -ae, work)?, work)?;
                value = value.add(&entry.mul(x));
            }
            abs(&value.sub(&lambda.mul(&v[i])))
        })
        .collect::<Vec<_>>();
    let mut residuals = Vec::with_capacity(n);
    let mut norm_squared = I::from_i64(0, work);
    let mut vector_max_lower = Float::with_val(work, 0);
    for (i, residual) in rows.into_iter().enumerate() {
        residuals.push(residual?);
        let x = abs(&v[i])?;
        norm_squared = norm_squared.add(&x.mul(&x));
        if x.lower() > &vector_max_lower {
            vector_max_lower = x.lower().clone();
        }
    }
    // Normalize residual intervals before squaring. Combine their exponent
    // with the matrix exponent only at the final step, preserving compensated
    // finite results rather than underflowing an intermediate residual square.
    let re = residuals
        .iter()
        .filter_map(|x| x.upper().get_exp())
        .max()
        .map_or(0, i64::from);
    let mut residual_squared = I::from_i64(0, work);
    let mut residual_max_upper = Float::with_val(work, 0);
    for residual in residuals {
        let normalized = scale(&residual, -re)?;
        residual_squared = residual_squared.add(&normalized.mul(&normalized));
        if normalized.upper() > &residual_max_upper {
            residual_max_upper = normalized.upper().clone();
        }
    }
    let two_norm = residual_squared.div(&norm_squared)?.sqrt()?;
    let infinity_norm =
        I::from_float(&residual_max_upper, work)?.div(&I::from_float(&vector_max_lower, work)?)?;
    Ok(ResidualBounds {
        eigenvalue_error_upper: scale_float(two_norm.upper(), ae + re, work)?,
        vector_scaled_residual_upper: scale_float(infinity_norm.upper(), ae + re, work)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;

    /// Explicit shape qualification only. A structured zero-heavy source has
    /// different arithmetic and RSS behavior from an arbitrary dense matrix.
    #[test]
    #[ignore = "explicit real-shape residual qualification; allocates a large source"]
    fn claim12_real_shape_structured_residual() {
        let modes: usize = std::env::var("XC_RESIDUAL_REAL_N")
            .expect("set XC_RESIDUAL_REAL_N to 2400 or 2800")
            .parse()
            .unwrap();
        assert!([2400, 2800].contains(&modes));
        let p = 4217u32;
        let n = 2 * modes + 1;
        let source_bytes = u64::try_from(
            ((n as u128) * (n as u128) + n as u128 + 1) * (u128::from(p).div_ceil(64) * 8 + 160),
        )
        .unwrap();
        let scratch_bytes = u64::try_from(workspace_bytes(n, p).unwrap()).unwrap();
        let budget = source_bytes + scratch_bytes + 65536;
        assert!(budget <= 84u64 << 30);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(budget, 4).unwrap();
            let _scope = resources.enter();
            let source_reservation = resources.try_reserve_retained(source_bytes).unwrap();
            let started = std::time::Instant::now();
            eprintln!("[real-shape residual] allocating N={modes}, D={n}, p={p}; admitted source={source_bytes}, scratch={scratch_bytes} bytes");
            let mut matrix = vec![Float::with_val(p, 0); n * n];
            for i in 0..n {
                matrix[i * n + i] = Float::with_val(p, i + 1);
            }
            let mut vector = vec![Float::with_val(p, 0); n];
            vector[0] = Float::with_val(p, 1);
            let lambda = Float::with_val(p, 1);
            eprintln!("[real-shape residual] source ready after {:.3}s; evaluating unchanged production residual", started.elapsed().as_secs_f64());
            let result = evaluate(&matrix, &vector, &lambda, p).unwrap();
            assert_eq!(result.eigenvalue_error_upper, 0);
            assert_eq!(result.vector_scaled_residual_upper, 0);
            assert_eq!(result.eigenvalue_error_upper.prec(), p + 64);
            assert_eq!(result.vector_scaled_residual_upper.prec(), p + 64);
            assert_eq!(resources.usage().working_bytes, 0);
            assert_eq!(resources.usage().retained_bytes, source_bytes);
            assert_eq!(resources.usage().peak_bytes, source_bytes + scratch_bytes);
            eprintln!("{}", serde_json::json!({
                "schema_version": 1, "passed": true, "n_modes": modes, "full_dimension": n,
                "precision_bits": p, "workers": 4, "elapsed_seconds": started.elapsed().as_secs_f64(),
                "admitted_source_bytes": source_bytes, "admitted_scratch_bytes": scratch_bytes,
                "reservation_peak_bytes_not_rss": resources.usage().peak_bytes,
                "eigenvalue_error_upper": "0", "vector_scaled_residual_upper": "0",
                "fixture": "exact diagonal A[i,i]=i+1, one-hot v[0]=1, lambda=1",
                "scope": "Real production residual shape with an exact closed-form oracle; structured diagonal/one-hot fixture RSS and timing are not representative of a dense arbitrary production matrix. No claim, ground proof, Krylov solve, cache, or publication."
            }));
            drop(matrix);
            drop(vector);
            drop(lambda);
            drop(source_reservation);
            assert_eq!(resources.usage().retained_bytes, 0);
        });
    }

    // Frozen serial evaluation preceding parallel rows.
    fn reference_evaluate(
        a: &[Float],
        v: &[Float],
        lambda: &Float,
        p: u32,
    ) -> Result<ResidualBounds> {
        let n = v.len();
        if n == 0
            || n.checked_mul(n) != Some(a.len())
            || !(64..=1_000_000).contains(&p)
            || a.iter()
                .chain(v)
                .chain(std::iter::once(lambda))
                .any(|x| !x.is_finite() || x.prec() > p)
            || v.iter().all(Float::is_zero)
        {
            bail!("invalid finite source-residual dimensions, precision or vector");
        }
        for i in 0..n {
            for j in i + 1..n {
                if a[i * n + j] != a[j * n + i] {
                    bail!("source eigenvalue residual requires exact stored symmetry");
                }
            }
        }
        let work = p + 64;
        let buffers = (a.len() as u64)
            .saturating_mul(2)
            .saturating_add((n as u64).saturating_mul(8))
            .saturating_add(192);
        if buffers.saturating_mul(u64::from(work).div_ceil(8) + 64) > (8u64 << 30) {
            bail!("source residual exceeds numerical workspace budget");
        }
        let ae = a
            .iter()
            .chain(std::iter::once(lambda))
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        let ve = i64::from(v.iter().filter_map(Float::get_exp).max().unwrap());
        let normalized = |xs: &[Float], exponent: i64| -> Result<Vec<I>> {
            xs.iter()
                .map(|x| Ok(I::from_float(&scale_float(x, -exponent, work)?, work)?))
                .collect()
        };
        let a = normalized(a, ae)?;
        let v = normalized(v, ve)?;
        let lambda = I::from_float(&scale_float(lambda, -ae, work)?, work)?;
        let mut residuals = Vec::with_capacity(n);
        let mut norm_squared = I::from_i64(0, work);
        let mut vector_max_lower = Float::with_val(work, 0);
        for (i, row) in a.chunks(n).enumerate() {
            let mut value = I::from_i64(0, work);
            for (entry, x) in row.iter().zip(&v) {
                value = value.add(&entry.mul(x));
            }
            residuals.push(abs(&value.sub(&lambda.mul(&v[i])))?);
            let x = abs(&v[i])?;
            norm_squared = norm_squared.add(&x.mul(&x));
            if x.lower() > &vector_max_lower {
                vector_max_lower = x.lower().clone();
            }
        }
        // Normalize residual intervals before squaring. Combine their exponent
        // with the matrix exponent only at the final step, preserving compensated
        // finite results rather than underflowing an intermediate residual square.
        let re = residuals
            .iter()
            .filter_map(|x| x.upper().get_exp())
            .max()
            .map_or(0, i64::from);
        let mut residual_squared = I::from_i64(0, work);
        let mut residual_max_upper = Float::with_val(work, 0);
        for residual in residuals {
            let normalized = scale(&residual, -re)?;
            residual_squared = residual_squared.add(&normalized.mul(&normalized));
            if normalized.upper() > &residual_max_upper {
                residual_max_upper = normalized.upper().clone();
            }
        }
        let two_norm = residual_squared.div(&norm_squared)?.sqrt()?;
        let infinity_norm = I::from_float(&residual_max_upper, work)?
            .div(&I::from_float(&vector_max_lower, work)?)?;
        Ok(ResidualBounds {
            eigenvalue_error_upper: scale_float(two_norm.upper(), ae + re, work)?,
            vector_scaled_residual_upper: scale_float(infinity_norm.upper(), ae + re, work)?,
        })
    }

    fn residual_fixture(n: usize, p: u32, case: u64) -> (Vec<Float>, Vec<Float>, Float) {
        let mut state = case.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (n as u64) << 20 ^ u64::from(p);
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as i64
        };
        let mut a = vec![Float::new(p); n * n];
        for i in 0..n {
            for j in i..n {
                // Full-precision quotients, scattered zeros of either sign.
                let value = match next() % 6 {
                    0 => Float::with_val(p, 0),
                    1 => -Float::with_val(p, 0),
                    _ => Float::with_val(p, next() % 2001 - 1000) / (next() % 997 + 3),
                };
                a[i * n + j] = value.clone();
                a[j * n + i] = value;
            }
        }
        let v = (0..n)
            .map(|_| Float::with_val(p, next() % 4001 - 2000) / (next() % 89 + 7))
            .collect::<Vec<_>>();
        let lambda = Float::with_val(p, next() % 101 - 50) / 7u32;
        (a, v, lambda)
    }

    fn same_bounds(left: &Result<ResidualBounds>, right: &Result<ResidualBounds>) -> bool {
        let same = |x: &Float, y: &Float| x.prec() == y.prec() && x.total_cmp(y).is_eq();
        match (left, right) {
            (Ok(x), Ok(y)) => {
                same(&x.eigenvalue_error_upper, &y.eigenvalue_error_upper)
                    && same(
                        &x.vector_scaled_residual_upper,
                        &y.vector_scaled_residual_upper,
                    )
            }
            (Err(x), Err(y)) => format!("{x:#}") == format!("{y:#}"),
            _ => false,
        }
    }

    #[test]
    fn parallel_rows_match_serial_reference_bitwise_at_every_pool_size() {
        let pools = [1, 2, 4, 8].map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        });
        let mut cases = Vec::new();
        for p in [64, 256, 3386] {
            for n in [1, 2, 3, 17, 241] {
                for case in 0..2 {
                    cases.push((residual_fixture(n, p, case), p, 10));
                }
            }
        }
        cases.push((residual_fixture(1001, 64, 0), 64, 1));
        // The actual Claim 12 source precision and the larger production
        // precision use the same arithmetic despite the different storage.
        for p in [4217, 6708] {
            for n in [3, 17, 65] {
                cases.push((residual_fixture(n, p, 3), p, 2));
            }
        }
        // Error paths: the first failing entry, in index order, is reported.
        let p = 128;
        let huge = Float::with_val(p, 1) << (rug::float::exp_max() - 2);
        let tiny = Float::with_val(p, 1) >> (-(rug::float::exp_min() + 2));
        let (mut a, v, lambda) = residual_fixture(3, p, 0);
        a[0] = huge.clone();
        a[4] = tiny.clone();
        cases.push(((a.clone(), v.clone(), lambda.clone()), p, 2));
        let mut wide_vector = v.clone();
        wide_vector[1] = huge;
        wide_vector[2] = tiny;
        cases.push(((residual_fixture(3, p, 1).0, wide_vector, lambda), p, 2));
        for ((a, v, lambda), p, repeats) in &cases {
            let expected = reference_evaluate(a, v, lambda, *p);
            for pool in &pools {
                for _ in 0..*repeats {
                    let actual = pool.install(|| evaluate(a, v, lambda, *p));
                    assert!(same_bounds(&actual, &expected), "n={} p={p}", v.len());
                }
            }
        }
        assert!(cases
            .iter()
            .any(|((a, v, l), p, _)| evaluate(a, v, l, *p).is_err()));
    }

    #[test]
    fn streaming_residual_admission_tracks_owned_linear_scratch() {
        for (n, bytes) in [(4801, 214_212_096u128), (5601, 249_847_296)] {
            assert_eq!(workspace_bytes(n, 4217).unwrap(), bytes);
            // This is an allocation-free admission calculation, not execution
            // of a Claim 12 matrix or a measurement of peak process memory.
            assert!(bytes < 256 << 20);
        }
        let n = 17;
        let p = 231;
        let (a, v, lambda) = residual_fixture(n, p, 3);
        let bytes = u64::try_from(workspace_bytes(n, p).unwrap()).unwrap();
        let expected = reference_evaluate(&a, &v, &lambda, p);
        let resources = xc_core::RunResources::new(bytes).unwrap();
        {
            let _scope = resources.enter();
            assert!(same_bounds(&evaluate(&a, &v, &lambda, p), &expected));
            assert_eq!(resources.usage().working_bytes, 0);
            assert_eq!(resources.usage().peak_bytes, bytes);
        }
        let resources = xc_core::RunResources::new(bytes - 1).unwrap();
        {
            let _scope = resources.enter();
            let error = evaluate(&a, &v, &lambda, p).err().unwrap();
            assert!(error
                .to_string()
                .contains("source residual exceeds numerical workspace budget"));
            assert_eq!(resources.usage().peak_bytes, 0);
        }
        let resources = xc_core::RunResources::new(bytes).unwrap();
        let _scope = resources.enter();
        let _other = resources.try_reserve_retained(1).unwrap();
        assert!(evaluate(&a, &v, &lambda, p).is_err());
        assert_eq!(resources.usage().working_bytes, 0);
        assert!(workspace_bytes(0, p).is_err());
        assert!(workspace_bytes(n, 63).is_err());
        assert!(workspace_bytes(n, 1_000_001).is_err());
    }

    #[test]
    fn shared_relative_residual_matches_the_full_tau_replay() {
        for p in [64, 256] {
            for n in [1, 3, 17] {
                for case in 0..3 {
                    let (a, v, lambda) = residual_fixture(n, p, case);
                    for dim in [n, n + 1] {
                        let shared = relative_residual(&a, dim, &v, &lambda, p);
                        let replay = super::super::weil_eigvec_cache::relative_residual_norm(
                            &a, dim, &v, &lambda, p,
                        );
                        assert_eq!(shared.as_ref().map(|(value, _)| value), replay.as_ref());
                        if let Some((_, bounds)) = shared {
                            assert!(same_bounds(&Ok(bounds), &evaluate(&a, &v, &lambda, p)));
                        }
                    }
                    let zero = vec![Float::with_val(p, 0); n];
                    assert!(relative_residual(&a, n, &zero, &lambda, p).is_none());
                }
            }
        }
    }
    #[test]
    fn directed_source_residuals_bound_independent_exact_rational_norms_and_scales() {
        let mut reports = 0;
        for p in [64, 128, 256] {
            for n in 1usize..=5 {
                for case in 0i32..8 {
                    let a = (0..n * n)
                        .map(|j| {
                            Float::with_val(p, ((j / n + j % n) as i32 * 3 + case) % 11 - 5) / 13u32
                        })
                        .collect::<Vec<_>>();
                    let v = (0..n)
                        .map(|j| Float::with_val(p, j as i32 * 2 + case + 1) / 7u32)
                        .collect::<Vec<_>>();
                    let lambda = Float::with_val(p, case - 4) / 3u32;
                    let av = a
                        .iter()
                        .map(|x| x.to_rational().unwrap())
                        .collect::<Vec<_>>();
                    let vv = v
                        .iter()
                        .map(|x| x.to_rational().unwrap())
                        .collect::<Vec<_>>();
                    let l = lambda.to_rational().unwrap();
                    let residual = (0..n)
                        .map(|i| {
                            (0..n)
                                .map(|j| av[i * n + j].clone() * &vv[j])
                                .sum::<Rational>()
                                - l.clone() * &vv[i]
                        })
                        .collect::<Vec<_>>();
                    let squared = residual
                        .iter()
                        .map(|x| x.clone().square())
                        .sum::<Rational>()
                        / vv.iter().map(|x| x.clone().square()).sum::<Rational>();
                    let infinity = residual.iter().map(|x| x.clone().abs()).max().unwrap()
                        / vv.iter().map(|x| x.clone().abs()).max().unwrap();
                    let base = evaluate(&a, &v, &lambda, p).unwrap();
                    for ascale in [-700_000_000i32, 0, 700_000_000] {
                        for vscale in [-700_000_000i32, 0, 700_000_000] {
                            let a = a.iter().map(|x| x.clone() << ascale).collect::<Vec<_>>();
                            let v = v.iter().map(|x| x.clone() << vscale).collect::<Vec<_>>();
                            let actual = evaluate(&a, &v, &(lambda.clone() << ascale), p).unwrap();
                            let two = (actual.eigenvalue_error_upper.clone() >> ascale)
                                .to_rational()
                                .unwrap();
                            let inf = (actual.vector_scaled_residual_upper.clone() >> ascale)
                                .to_rational()
                                .unwrap();
                            assert!(two >= 0 && two.square() >= squared);
                            assert!(inf >= infinity);
                            assert_eq!(
                                actual.eigenvalue_error_upper,
                                base.eigenvalue_error_upper.clone() << ascale
                            );
                            assert_eq!(
                                actual.vector_scaled_residual_upper,
                                base.vector_scaled_residual_upper.clone() << ascale
                            );
                            reports += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(reports, 1080);
    }
    #[test]
    fn source_residual_zero_and_domain_contracts() {
        let p = 128;
        let one = Float::with_val(p, 1);
        let zero = Float::with_val(p, 0);
        let exact = evaluate(
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            &one,
            p,
        )
        .unwrap();
        assert_eq!(exact.eigenvalue_error_upper, 0);
        assert_eq!(exact.vector_scaled_residual_upper, 0);
        assert!(evaluate(&[], &[], &one, p).is_err());
        assert!(evaluate(std::slice::from_ref(&one), &[zero], &one, p).is_err());
        assert!(evaluate(
            &[Float::with_val(p, f64::NAN)],
            std::slice::from_ref(&one),
            &one,
            p
        )
        .is_err());
        assert!(evaluate(
            &[one.clone(), one.clone(), Float::with_val(p, 0), one.clone()],
            &[one.clone(), one.clone()],
            &one,
            p
        )
        .is_err());
    }
}
