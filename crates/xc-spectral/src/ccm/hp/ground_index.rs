//! Source-bound algebraic index check in the requested subspace.
//! Two directed inertia counts isolate index zero of the exact stored matrix
//! for Natural/AdaptiveEven, or its even compression for EvenSector. The latter
//! does not assert an ordering against the odd block or the unrounded CCM form.
use super::{state_residual_bounds, stored_resolution, CcmParityPolicy};
use crate::ccm::capture_runtime::CaptureResourcePolicy;
use crate::ccm::retained_evidence::finite_math::scale_float;
use anyhow::{bail, Result};
use rug::{float::Round, Float};
use xc_certify::exact::{interval_symmetric_ldlt_inertia_mpfr_stable, IntervalInertiaResult};
use xc_numerics::{interval::RationalInterval, mpfr_interval::MpfrInterval as I};

pub(super) const ARITHMETIC: &str =
    "directed_stored_source_stable_ground_index_and_rounding_scale_gap_v3";

std::thread_local! {
    pub(super) static CHECKED_SOURCES: std::cell::RefCell<Vec<xc_cache::ContentDigest>> = const { std::cell::RefCell::new(Vec::new()) };
    #[cfg(test)]
    pub(super) static FULL_GROUND_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn inconclusive_summary(result: &IntervalInertiaResult, stage: &str, work: u32) -> String {
    let IntervalInertiaResult::Inconclusive {
        pivot_index,
        positive,
        negative,
        zero_or_unresolved,
        pivot_enclosures,
        reason,
    } = result
    else {
        unreachable!("only inconclusive inertia is summarized")
    };
    let last = pivot_enclosures.last().map(|pivot| {
        let lower = Float::with_val_round(work, pivot.lower(), Round::Down)
            .0
            .to_string_radix_round(10, Some(16), Round::Down);
        let upper = Float::with_val_round(work, pivot.upper(), Round::Up)
            .0
            .to_string_radix_round(10, Some(16), Round::Up);
        format!("[{lower}, {upper}]")
    });
    format!("stage={stage}, work_precision_bits={work}, pivot_index={pivot_index}, positive={positive}, negative={negative}, unresolved={zero_or_unresolved}, last_pivot={}, reason={reason}", last.as_deref().unwrap_or("absent"))
}

fn working_budget() -> Result<u128> {
    Ok(u128::from(
        CaptureResourcePolicy::from_environment()?.maximum_working_bytes,
    ))
}

fn check_workspace(bytes: u128, maximum_bytes: u128, stage: &str) -> Result<()> {
    if bytes > maximum_bytes {
        bail!("ground-index {stage} exceeds workspace budget: estimated_bytes={bytes}, maximum_working_bytes={maximum_bytes}; configure XC_RESEARCH_WORKING_BYTES within available memory");
    }
    Ok(())
}

fn rational_matrix(
    matrix: &[I],
    p: u32,
    retained_source_bytes: u128,
    maximum_bytes: u128,
) -> Result<Vec<RationalInterval>> {
    let mut bits = 0u128;
    for x in matrix {
        x.validate()?;
        for endpoint in [x.lower(), x.upper()] {
            bits += 2 * u128::from(endpoint.prec())
                + u128::from(i64::from(endpoint.get_exp().unwrap_or(0)).unsigned_abs())
                + 2;
        }
    }
    let bytes = bits.div_ceil(8) * 12
        + matrix.len() as u128 * (u128::from(p).div_ceil(8) + 160) * 12
        + retained_source_bytes;
    check_workspace(bytes, maximum_bytes, "rational inertia")?;
    Ok(matrix.iter().map(I::to_rational_interval).collect())
}

/// Reject impossible dimension/precision allocations before constructing Tau.
/// Actual exponent spans remain subject to rational_matrix's byte accounting.
pub(super) fn preflight(n: usize, p: u32, parity: CcmParityPolicy) -> Result<()> {
    preflight_with_budget(n, p, parity, working_budget()?)
}

fn preflight_with_budget(
    n: usize,
    p: u32,
    parity: CcmParityPolicy,
    maximum_bytes: u128,
) -> Result<()> {
    check_workspace(workspace_bytes(n, p, parity)?, maximum_bytes, "validation")
}

fn workspace_bytes(n: usize, p: u32, parity: CcmParityPolicy) -> Result<u128> {
    if !(64..=1_000_000).contains(&p) || n == 0 || n > 8193 || n.is_multiple_of(2) {
        bail!("invalid ground-index dimensions or precision");
    }
    let d = if parity == CcmParityPolicy::EvenSector {
        n / 2 + 1
    } else {
        n
    };
    let work = p.saturating_add(1024);
    let numerical = n as u128 * n as u128 * (u128::from(work).div_ceil(8) + 160) * 16;
    // Two dyadic endpoints, twelve conservative rational/LDLT work copies,
    // and twelve MPFR interval/storage work copies. Exponents are checked later.
    let rational_minimum = d as u128
        * d as u128
        * (2 * u128::from(2 * work + 2).div_ceil(8) * 12
            + (u128::from(work).div_ceil(8) + 160) * 12);
    Ok(numerical + rational_minimum)
}

// Integer reflection basis: e_center and e_(center-k)+e_(center+k).
// Congruence of A-tI preserves inertia without approximating sqrt(2).
fn shifted_matrix(
    a: &[Float],
    n: usize,
    shift: &Float,
    even: bool,
    p: u32,
    maximum_bytes: u128,
) -> Result<Vec<RationalInterval>> {
    let dimension = if even { n / 2 + 1 } else { n };
    let supports = (0..dimension)
        .map(|k| {
            if !even {
                vec![k]
            } else if k == 0 {
                vec![n / 2]
            } else {
                vec![n / 2 - k, n / 2 + k]
            }
        })
        .collect::<Vec<_>>();
    let shift = I::from_float(shift, p)?;
    let mut matrix = vec![I::from_i64(0, p); dimension * dimension];
    for i in 0..dimension {
        for j in i..dimension {
            let mut value = I::from_i64(0, p);
            for &row in &supports[i] {
                for &column in &supports[j] {
                    let entry = I::from_float(&a[row * n + column], p)?;
                    value = value.add(&if row == column {
                        entry.sub(&shift)
                    } else {
                        entry
                    });
                }
            }
            matrix[i * dimension + j] = value.clone();
            matrix[j * dimension + i] = value;
        }
    }
    // The full retained/normalized source, state, and residual work remain
    // alive while shifted interval matrices and their rational images exist.
    let retained_source_bytes = n as u128 * n as u128 * (u128::from(p).div_ceil(8) + 160) * 16;
    rational_matrix(&matrix, p, retained_source_bytes, maximum_bytes)
}

pub(super) fn validate(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
) -> Result<()> {
    validate_with_residual(a, v, lambda, p, parity, None)
}

/// `validate`, given `state_residual_bounds::evaluate(a, v, lambda, p)` when
/// the caller has already evaluated it successfully for exactly these inputs.
pub(super) fn validate_with_residual(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
    residual: Option<&state_residual_bounds::ResidualBounds>,
) -> Result<()> {
    // This remains a mathematical gate, not a claim that sparse probes prove
    // the ground index. Repeated consumers of exactly the same source reuse
    // the successful proof in this process rather than recounting its inertia.
    preflight(v.len(), p, parity)?;
    if v.len().checked_mul(v.len()) != Some(a.len()) {
        bail!("invalid ground-index source dimensions");
    }
    let namespace = format!("{ARITHMETIC}:{parity:?}");
    let digest = super::sector_transform_validation::source_digest(
        namespace.as_bytes(),
        &[a, v, std::slice::from_ref(lambda)],
        p,
    )?;
    if CHECKED_SOURCES.with(|cache| cache.borrow().contains(&digest)) {
        return Ok(());
    }
    validate_unretained(a, v, lambda, p, parity, residual)?;
    CHECKED_SOURCES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= 8 {
            cache.remove(0);
        }
        cache.push(digest);
    });
    Ok(())
}

fn validate_unretained(
    a: &[Float],
    v: &[Float],
    lambda: &Float,
    p: u32,
    parity: CcmParityPolicy,
    residual: Option<&state_residual_bounds::ResidualBounds>,
) -> Result<()> {
    #[cfg(test)]
    FULL_GROUND_CHECKS.with(|count| count.set(count.get() + 1));
    let n = v.len();
    if !(64..=1_000_000).contains(&p)
        || n == 0
        || n > 8193
        || n.is_multiple_of(2)
        || n.checked_mul(n) != Some(a.len())
        || a.iter()
            .chain(v)
            .chain(std::iter::once(lambda))
            .any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid ground-index source dimensions, precision or points");
    }
    // One execution budget applies to admission and all guarded inertia stages.
    // It does not change the arithmetic or the mathematical artifact identity.
    let maximum_bytes = working_budget()?;
    preflight_with_budget(n, p, parity, maximum_bytes)?;
    let even = parity == CcmParityPolicy::EvenSector;
    if even
        && (v.iter().zip(v.iter().rev()).any(|(x, y)| x != y)
            || a.iter().zip(a.iter().rev()).any(|(x, y)| x != y))
    {
        bail!("even ground-index source requires exact reflection symmetry");
    }
    let maximum_work = p.saturating_add(1024);
    check_workspace(
        a.len() as u128 * (u128::from(maximum_work).div_ceil(8) + 160) * 16,
        maximum_bytes,
        "numerical validation",
    )?;
    let exponent = a
        .iter()
        .chain(std::iter::once(lambda))
        .filter_map(Float::get_exp)
        .max()
        .map_or(0, i64::from);
    let normalized = a
        .iter()
        .map(|x| scale_float(x, -exponent, p))
        .collect::<Result<Vec<_>>>()?;
    let bounds = match residual {
        Some(residual) => {
            stored_resolution::bounds_with_residual(a, v, lambda, p, parity, residual)?
        }
        None => stored_resolution::bounds(a, v, lambda, p, parity)?,
    };
    let dimension = if even { n / 2 + 1 } else { n };
    let mut inconclusive = Vec::new();
    for guard in [64, 256, 1024] {
        let work = p.saturating_add(guard);
        let lower = scale_float(&bounds.lower, -exponent, work)?;
        let upper = scale_float(&bounds.upper, -exponent, work)?;
        let neighbor_probe = scale_float(&bounds.neighbor_probe, -exponent, work)?;
        let lower_matrix = shifted_matrix(&normalized, n, &lower, even, work, maximum_bytes)?;
        let lower_result =
            interval_symmetric_ldlt_inertia_mpfr_stable(&lower_matrix, dimension, work)?;
        drop(lower_matrix);
        match lower_result {
            IntervalInertiaResult::Conclusive { negative: 0, .. } => {}
            IntervalInertiaResult::Conclusive { .. } => {
                bail!("claimed CCM eigenpair is not the algebraic ground state in the requested subspace")
            }
            ref result @ IntervalInertiaResult::Inconclusive { .. } => {
                let detail = inconclusive_summary(result, "lower", work);
                xc_core::progress_message!("[HP] CCM ground-index inertia: {detail}");
                inconclusive.push(detail);
                continue;
            }
        }
        let upper_matrix = shifted_matrix(&normalized, n, &upper, even, work, maximum_bytes)?;
        match interval_symmetric_ldlt_inertia_mpfr_stable(&upper_matrix, dimension, work)? {
            IntervalInertiaResult::Conclusive { negative: 1, .. } => {
                if dimension == 1 {
                    return Ok(());
                }
                drop(upper_matrix);
                let neighbor_matrix =
                    shifted_matrix(&normalized, n, &neighbor_probe, even, work, maximum_bytes)?;
                match interval_symmetric_ldlt_inertia_mpfr_stable(&neighbor_matrix, dimension, work)? {
                    IntervalInertiaResult::Conclusive { negative: 1, .. } => return Ok(()),
                    IntervalInertiaResult::Conclusive { .. } => bail!(
                        "CCM stored precision resolution limit: selected-subspace gap is not separated from the matrix rounding scale at {p} bits"),
                    ref result @ IntervalInertiaResult::Inconclusive { .. } => {
                        let detail = inconclusive_summary(result, "neighbor", work);
                        xc_core::progress_message!("[HP] CCM ground-index inertia: {detail}");
                        inconclusive.push(detail);
                        continue;
                    }
                }
            }
            IntervalInertiaResult::Conclusive { negative, .. } => {
                bail!(
                    "CCM ground state is unresolved or clustered at the requested precision: upper_count={negative}, eigenvalue={lambda}, residual={}",
                    bounds.record.residual_upper
                )
            }
            ref result @ IntervalInertiaResult::Inconclusive { .. } => {
                let detail = inconclusive_summary(result, "upper", work);
                xc_core::progress_message!("[HP] CCM ground-index inertia: {detail}");
                inconclusive.push(detail);
            }
        }
    }
    bail!("CCM ground-index inertia is inconclusive within the guard precision budget: eigenvalue={}, residual_upper={}; {}", lambda.to_string_radix(10, Some(16)), bounds.record.residual_upper, inconclusive.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_ground_proof_reuses_only_identical_points_and_parity() {
        CHECKED_SOURCES.with(|cache| cache.borrow_mut().clear());
        FULL_GROUND_CHECKS.with(|count| count.set(0));
        let p = 128;
        let a: Vec<_> = [2, 0, 0, 0, 1, 0, 0, 0, 2]
            .into_iter()
            .map(|x| Float::with_val(p, x))
            .collect();
        let v: Vec<_> = [0, 1, 0]
            .into_iter()
            .map(|x| Float::with_val(p, x))
            .collect();
        let value = Float::with_val(p, 1);
        for _ in 0..2 {
            validate(&a, &v, &value, p, CcmParityPolicy::EvenSector).unwrap();
        }
        assert_eq!(FULL_GROUND_CHECKS.with(|count| count.get()), 1);
        validate(&a, &v, &value, p, CcmParityPolicy::Natural).unwrap();
        assert_eq!(FULL_GROUND_CHECKS.with(|count| count.get()), 2);
        let wrong_value = Float::with_val(p, 2);
        let wrong_vector: Vec<_> = [1, 0, 0]
            .into_iter()
            .map(|x| Float::with_val(p, x))
            .collect();
        for expected in [3, 4] {
            assert!(
                validate(&a, &wrong_vector, &wrong_value, p, CcmParityPolicy::Natural).is_err()
            );
            assert_eq!(FULL_GROUND_CHECKS.with(|count| count.get()), expected);
        }
        let mut wrong_source = a.clone();
        wrong_source[0] = Float::with_val(p, 0);
        assert!(validate(&wrong_source, &v, &value, p, CcmParityPolicy::Natural).is_err());
        assert_eq!(FULL_GROUND_CHECKS.with(|count| count.get()), 5);
    }

    #[test]
    fn inconclusive_diagnostics_retain_stage_precision_and_pivot_reason() {
        let matrix = vec![RationalInterval::point(rug::Rational::from(0))];
        let result = interval_symmetric_ldlt_inertia_mpfr_stable(&matrix, 1, 128).unwrap();
        let detail = inconclusive_summary(&result, "upper", 128);
        for expected in [
            "stage=upper",
            "work_precision_bits=128",
            "pivot_index=0",
            "positive=0",
            "negative=0",
            "unresolved=1",
            "last_pivot=[",
            "reason=no remaining signed",
        ] {
            assert!(detail.contains(expected), "{detail}");
        }
    }
    #[test]
    fn rational_budget_counts_full_significand_numerator_and_denominator() {
        for p in [64u32, 128, 1725] {
            let integer = (rug::Integer::from(1) << (p - 1)) + 1;
            for exponent in [-1000i32, 0, 1000] {
                let value: Float = (Float::with_val(p, &integer) >> p) << exponent;
                let exact = value.to_rational().unwrap();
                let actual = u128::from(exact.numer().significant_bits())
                    + u128::from(exact.denom().significant_bits());
                let estimate = 2 * u128::from(p)
                    + u128::from(i64::from(value.get_exp().unwrap()).unsigned_abs())
                    + 2;
                assert!(estimate >= actual);
            }
        }
        // Each constituent estimate fits alone, but their simultaneous sum does not.
        assert!(preflight_with_budget(1001, 1000, CcmParityPolicy::EvenSector, 8 << 30).is_err());
        preflight_with_budget(301, 1725, CcmParityPolicy::EvenSector, 8 << 30).unwrap();
        // Former 64-Mbit cap rejected this modest rational image even for zeros.
        let entries = vec![I::from_i64(0, 1789); 151 * 151];
        rational_matrix(&entries, 1789, 301 * 301 * 512 * 16, 8 << 30).unwrap();
    }
    #[test]
    fn ground_index_admission_respects_declared_budget_at_campaign_dimensions() {
        for (n, bytes) in [
            (241, 1_174_328_008u128),
            (1001, 20_203_846_408),
            (2001, 80_699_646_408),
        ] {
            assert_eq!(
                workspace_bytes(n, 3386, CcmParityPolicy::EvenSector).unwrap(),
                bytes
            );
            assert_eq!(
                preflight_with_budget(n, 3386, CcmParityPolicy::EvenSector, 8 << 30).is_ok(),
                n == 241
            );
            preflight_with_budget(n, 3386, CcmParityPolicy::EvenSector, 96 << 30).unwrap();
            preflight_with_budget(n, 3386, CcmParityPolicy::EvenSector, bytes).unwrap();
            assert!(
                preflight_with_budget(n, 3386, CcmParityPolicy::EvenSector, bytes - 1).is_err()
            );
        }
        // Actual rational conversion also uses the declared limit. Charge a
        // large retained source without allocating it in this admission test.
        let matrix = [I::from_i64(1, 3386)];
        assert!(rational_matrix(&matrix, 3386, 9 << 30, 8 << 30).is_err());
        let actual = rational_matrix(&matrix, 3386, 9 << 30, 96 << 30).unwrap();
        assert_eq!(actual[0], matrix[0].to_rational_interval());
    }
    #[test]
    fn ground_index_runtime_budget_controls_preflight() {
        // Run this test in separate processes with different environment
        // budgets; never mutate the environment under parallel Rust tests.
        let budget = working_budget().unwrap();
        for n in [241, 1001, 2001] {
            assert_eq!(
                preflight(n, 3386, CcmParityPolicy::EvenSector).is_ok(),
                workspace_bytes(n, 3386, CcmParityPolicy::EvenSector).unwrap() <= budget
            );
        }
    }
    #[test]
    fn exhaustive_ground_index_exact_congruence_and_extreme_scale_oracles() {
        let v = [2, -1, 2];
        let w = [1, 4, 1];
        let odd = [1, 0, -1];
        let mut cases = 0;
        for p in [64, 128, 256] {
            for s in [-700_000_000i32, 0, 700_000_000] {
                for vector_scale in [-700_000_000i32, 0, 700_000_000] {
                    // Exact integer spectral decomposition: eigenvalues 9,36,-20.
                    let a = (0..9)
                        .map(|k| {
                            let i = k / 3;
                            let j = k % 3;
                            Float::with_val(p, v[i] * v[j] + 2 * w[i] * w[j] - 10 * odd[i] * odd[j])
                                << s
                        })
                        .collect::<Vec<_>>();
                    let vec = |x: &[i32]| {
                        x.iter()
                            .map(|x| Float::with_val(p, *x) << vector_scale)
                            .collect::<Vec<_>>()
                    };
                    assert!(validate(
                        &a,
                        &vec(&v),
                        &(Float::with_val(p, 9) << s),
                        p,
                        CcmParityPolicy::EvenSector
                    )
                    .is_ok());
                    assert!(validate(
                        &a,
                        &vec(&v),
                        &(Float::with_val(p, 9) << s),
                        p,
                        CcmParityPolicy::Natural
                    )
                    .is_err());
                    assert!(validate(
                        &a,
                        &vec(&w),
                        &(Float::with_val(p, 36) << s),
                        p,
                        CcmParityPolicy::EvenSector
                    )
                    .is_err());
                    assert!(validate(
                        &a,
                        &vec(&odd),
                        &(Float::with_val(p, -20) << s),
                        p,
                        CcmParityPolicy::Natural
                    )
                    .is_ok());
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 27);
    }
    #[test]
    fn exact_small_eigenvalue_requires_a_gap_resolved_at_storage_precision() {
        for p in [64, 128, 256, 512] {
            let mut a = vec![Float::with_val(p, 0); 25];
            for j in [1, 3] {
                a[j * 5 + j] = Float::with_val(p, 1);
            }
            for j in [0, 4] {
                a[j * 5 + j] = Float::with_val(p, 1) >> 350u32;
            }
            a[12] = Float::with_val(p, 1) >> 400u32;
            let mut v = vec![Float::with_val(p, 0); 5];
            v[2] = Float::with_val(p, 1);
            for parity in [CcmParityPolicy::EvenSector, CcmParityPolicy::Natural] {
                let result = validate(&a, &v, &a[12], p, parity);
                // The exact selected value is 2^-400 and its neighbor is
                // 2^-350. Index isolation alone is possible at every p, but
                // the gap is below 5*2^-p for the first three precisions.
                assert_eq!(result.is_ok(), p == 512, "p={p}: {result:?}");
            }
        }
    }
    #[test]
    fn exhaustive_ground_index_domains_are_fallible() {
        let p = 128;
        let one = Float::with_val(p, 1);
        assert!(validate(&[], &[], &one, p, CcmParityPolicy::Natural).is_err());
        assert!(validate(
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            &one,
            0,
            CcmParityPolicy::Natural
        )
        .is_err());
        assert!(validate(
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            &one,
            p,
            CcmParityPolicy::Natural
        )
        .is_ok());
        let big = Float::with_val(p, 1) << 700_000_000i32;
        let little = Float::with_val(p, 1) >> 700_000_000i32;
        let intervals = [
            I::from_float(&big, p).unwrap(),
            I::from_float(&little, p).unwrap(),
            I::from_float(&big, p).unwrap(),
            I::from_float(&little, p).unwrap(),
            I::from_float(&big, p).unwrap(),
        ];
        // Five intervals exceed the current 8 GiB modeled workspace and must
        // reject before constructing any enormous exact rational endpoint.
        assert!(rational_matrix(&intervals, p, 0, 8 << 30).is_err());
    }
}
